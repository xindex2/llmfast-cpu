//! Compute kernels. Reference-quality but structured for the compiler to vectorize
//! (build with target-cpu=native, see .cargo/config.toml). Hand-written SIMD / int4 come in M2.

use crate::pool;

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

pub const ROWS_PER_CHUNK: usize = 16;

// ---------- NUMA-aware weight allocation ----------
//
// Linux places a page on the node of the thread that *first writes* it. Every weight buffer
// used to be filled by whichever thread was loading the model, so on a two-socket box all the
// weights landed on node 0 and half the workers streamed them over the interconnect for the
// life of the process. Decode is pure streaming, so it paid that penalty on every token.
//
// These two helpers hand each pool worker the rows it will later compute -- the same
// chunk->worker mapping `run_static` uses in every matvec -- so the first write to a page comes
// from the node that will read it. NUMA=0 restores the old single-threaded fill for A/B runs.

struct SendBytes(*mut u8);
unsafe impl Send for SendBytes {}
unsafe impl Sync for SendBytes {}
impl SendBytes {
    #[inline]
    fn get(&self) -> *mut u8 {
        self.0
    }
}

struct SrcBytes(*const u8);
unsafe impl Send for SrcBytes {}
unsafe impl Sync for SrcBytes {}
impl SrcBytes {
    #[inline]
    fn get(&self) -> *const u8 {
        self.0
    }
}

fn numa_placement() -> bool {
    use std::sync::OnceLock;
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("NUMA").map_or(true, |v| v != "0"))
}

/// Split `bytes` of a row-major buffer into the same row chunks the kernels use, and run `f`
/// on each chunk from the worker that will own those rows. Returns false when the buffer
/// cannot be split by rows (or placement is off), so callers fall back to a serial fill.
fn per_owner_rows(bytes: usize, rows: usize, f: impl Fn(usize, usize) + Sync) -> bool {
    if !numa_placement() || rows == 0 || bytes == 0 || bytes % rows != 0 {
        return false;
    }
    let row_bytes = bytes / rows;
    let chunks = (rows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(rows);
        f(r0 * row_bytes, (r1 - r0) * row_bytes);
    });
    true
}

/// A zeroed `Vec<T>` of `elems` elements whose pages are first touched by the workers that
/// will own those rows. Use instead of `vec![zero; elems]` for anything the kernels stream.
/// Pure memory read bandwidth: the same pool, the same row-chunk split, no arithmetic beyond
/// an xor so the loads cannot be optimised away. This is the ceiling every kernel is measured
/// against — if a matvec reaches it, the kernel is done and only more memory channels help;
/// if it falls well short, the kernel is the bottleneck, not the box.
pub fn read_bandwidth(bytes: usize) -> (f64, u64) {
    let elems = bytes / 8;
    let rows = 4096;
    let buf: Vec<u64> = alloc_rows(elems, rows);
    let per_row = elems / rows;
    let chunks = (rows + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    let mut sums = vec![0u64; chunks];
    let sp = SendBytes(sums.as_mut_ptr() as *mut u8);
    let bp = &buf;
    let run = || {
        pool::global().run_static(chunks, &|c| {
            let r0 = c * ROWS_PER_CHUNK;
            let r1 = (r0 + ROWS_PER_CHUNK).min(rows);
            // Eight independent accumulators: a single `acc ^= v` chain is latency-bound on
            // the xor, not on memory, and would understate the ceiling badly.
            let src = &bp[r0 * per_row..r1 * per_row];
            let mut acc = [0u64; 8];
            let mut it = src.chunks_exact(8);
            for w in &mut it {
                for j in 0..8 {
                    acc[j] ^= w[j];
                }
            }
            let mut a = acc.iter().fold(0u64, |x, y| x ^ y);
            for v in it.remainder() {
                a ^= *v;
            }
            unsafe { *(sp.get() as *mut u64).add(c) = a };
        });
    };
    run(); // warm: fault every page in before timing
    let t = std::time::Instant::now();
    let iters = 5;
    for _ in 0..iters {
        run();
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    (dt, sums.iter().fold(0u64, |a, b| a ^ b))
}

pub fn alloc_rows<T: Copy>(elems: usize, rows: usize) -> Vec<T> {
    let mut v: Vec<T> = Vec::with_capacity(elems);
    let bytes = elems * std::mem::size_of::<T>();
    let p = SendBytes(v.as_mut_ptr() as *mut u8);
    if !per_owner_rows(bytes, rows, |off, len| unsafe { std::ptr::write_bytes(p.get().add(off), 0, len) }) {
        unsafe { std::ptr::write_bytes(p.get(), 0, bytes) };
    }
    unsafe { v.set_len(elems) };
    v
}

/// `src` copied into a fresh `Vec<T>`, each row chunk copied by the worker that will own it.
pub fn copy_rows<T: Copy>(src: &[u8], rows: usize) -> Vec<T> {
    let elems = src.len() / std::mem::size_of::<T>();
    let mut v: Vec<T> = Vec::with_capacity(elems);
    let p = SendBytes(v.as_mut_ptr() as *mut u8);
    let sp = SrcBytes(src.as_ptr());
    if !per_owner_rows(src.len(), rows, |off, len| unsafe {
        std::ptr::copy_nonoverlapping(sp.get().add(off), p.get().add(off), len)
    }) {
        unsafe { std::ptr::copy_nonoverlapping(sp.get(), p.get(), src.len()) };
    }
    unsafe { v.set_len(elems) };
    v
}


/// y = W · x   (W: [n, k] bf16 row-major, x: f32[k]).  Decode hot path: bandwidth-bound.
pub fn matvec_bf16(w: &[u16], x: &[f32], n: usize, k: usize, y: &mut [f32]) {
    debug_assert_eq!(w.len(), n * k);
    let yp = SendPtr(y.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        for i in r0..r1 {
            unsafe { *yp.get().add(i) = dot_bf16(&w[i * k..(i + 1) * k], x) };
        }
    });
}

/// Y = X · Wᵀ for a batch of m inputs  (xs: m×k f32, ys: m×n f32).  Prefill hot path.
/// Each weight row is read from RAM once and reused against all m inputs, turning a
/// bandwidth-bound problem into a compute-bound one — this is why prefill can be 20× faster
/// per token than decode.
pub fn matmul_bf16(w: &[u16], xs: &[f32], m: usize, n: usize, k: usize, ys: &mut [f32]) {
    debug_assert_eq!(w.len(), n * k);
    debug_assert_eq!(xs.len(), m * k);
    if m == 1 {
        return matvec_bf16(w, xs, n, k, ys);
    }
    let yp = SendPtr(ys.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        // Convert this chunk's rows to f32 once; they then stay hot in L1/L2 for every input.
        with_scratch((r1 - r0) * k, |rows| {
            dequant_bf16(&w[r0 * k..r1 * k], rows);
            tile_rows(rows, r0, r1, xs, m, n, k, yp);
        });
    });
}


/// Rows r0..r1 (already f32, contiguous in `rows`) against all m inputs, 4x2 register tiles.
#[allow(clippy::too_many_arguments)]
fn tile_rows(rows: &[f32], r0: usize, r1: usize, xs: &[f32], m: usize, n: usize, k: usize, yp: SendPtr) {
        let mut i = 0;
        while i + 4 <= r1 - r0 {
            let (ra, rb, rc, rd) = (&rows[i * k..(i + 1) * k], &rows[(i + 1) * k..(i + 2) * k], &rows[(i + 2) * k..(i + 3) * k], &rows[(i + 3) * k..(i + 4) * k]);
            let mut j = 0;
            while j + 2 <= m {
                let (x0, x1) = (&xs[j * k..(j + 1) * k], &xs[(j + 1) * k..(j + 2) * k]);
                let r = tile4x2(ra, rb, rc, rd, x0, x1);
                unsafe {
                    for q in 0..4 {
                        *yp.get().add(j * n + r0 + i + q) = r[q];
                        *yp.get().add((j + 1) * n + r0 + i + q) = r[4 + q];
                    }
                }
                j += 2;
            }
            if j < m {
                let x0 = &xs[j * k..(j + 1) * k];
                unsafe {
                    *yp.get().add(j * n + r0 + i) = dot_f32(ra, x0);
                    *yp.get().add(j * n + r0 + i + 1) = dot_f32(rb, x0);
                    *yp.get().add(j * n + r0 + i + 2) = dot_f32(rc, x0);
                    *yp.get().add(j * n + r0 + i + 3) = dot_f32(rd, x0);
                }
            }
            i += 4;
        }
        while i < r1 - r0 {
            let row = &rows[i * k..(i + 1) * k];
            for j in 0..m {
                unsafe { *yp.get().add(j * n + r0 + i) = dot_f32(row, &xs[j * k..(j + 1) * k]) };
            }
            i += 1;
        }
}

/// Per-thread reusable f32 scratch (avoids a fresh zeroed allocation per chunk).
fn with_scratch(len: usize, f: impl FnOnce(&mut [f32])) {
    thread_local!(static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) });
    SCRATCH.with(|s| {
        let mut v = s.borrow_mut();
        if v.len() < len {
            v.resize(len, 0.0);
        }
        f(&mut v[..len]);
    });
}

fn dequant_bf16(w: &[u16], out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::dequant_bf16(w, out) };
    }
    if has_avx1() {
        return unsafe { avx1::dequant_bf16(w, out) };
    }
    for (r, &b) in out.iter_mut().zip(w) {
        *r = bf16_to_f32(b);
    }
}

/// Public Send+Sync raw pointer wrapper for pool closures outside this module.
#[derive(Clone, Copy)]
pub struct SendPtrPub(pub *mut f32);
unsafe impl Send for SendPtrPub {}
unsafe impl Sync for SendPtrPub {}
impl SendPtrPub {
    #[inline]
    pub fn get(&self) -> *mut f32 {
        self.0
    }
}

#[derive(Clone, Copy)]
struct SendPtr(*mut f32);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
impl SendPtr {
    // Method access keeps the closure capturing the whole wrapper (which is Sync), not the raw field.
    #[inline]
    fn get(&self) -> *mut f32 {
        self.0
    }
}

// ---------------------------------------------------------------------------------------
// Inner kernels. Explicit AVX2+FMA on x86_64 (runtime-detected), scalar fallback elsewhere.
// The compiler would not keep the accumulator arrays in registers on its own (~1 GFLOPS).
// ---------------------------------------------------------------------------------------

/// SIMD level: 2 = AVX2+FMA (Haswell+), 1 = AVX+SSE4.1 (Sandy/Ivy Bridge), 0 = scalar.
/// FORCE_SIMD=0|1|2 overrides detection (for testing the slower paths on a faster machine).
#[cfg(target_arch = "x86_64")]
fn simd_level() -> u8 {
    use std::sync::OnceLock;
    static F: OnceLock<u8> = OnceLock::new();
    *F.get_or_init(|| {
        let detected = if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") { 2 }
            else if is_x86_feature_detected!("avx") && is_x86_feature_detected!("sse4.1") { 1 } else { 0 };
        let forced = std::env::var("FORCE_SIMD").ok().and_then(|v| v.parse::<u8>().ok());
        let level = forced.map_or(detected, |f| f.min(detected));
        eprintln!("simd: level {level} (detected {detected}){}", if forced.is_some() { " [forced]" } else { "" });
        level
    })
}

#[cfg(target_arch = "x86_64")]
fn has_avx2() -> bool {
    simd_level() >= 2
}

#[cfg(target_arch = "x86_64")]
fn has_avx1() -> bool {
    simd_level() == 1
}

#[inline]
fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { dot_bf16_avx2(w, x) };
    }
    if has_avx1() {
        return unsafe { avx1::dot_bf16(w, x) };
    }
    dot_bf16_scalar(w, x)
}

#[inline]
fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { dot_f32_avx2(a, b) };
    }
    if has_avx1() {
        return unsafe { avx1::dot_f32(a, b) };
    }
    dot_f32_scalar(a, b)
}

pub fn tile4x2_pub(ra: &[f32], rb: &[f32], rc: &[f32], rd: &[f32], x0: &[f32], x1: &[f32]) -> [f32; 8] {
    tile4x2(ra, rb, rc, rd, x0, x1)
}

#[inline]
fn tile4x2(ra: &[f32], rb: &[f32], rc: &[f32], rd: &[f32], x0: &[f32], x1: &[f32]) -> [f32; 8] {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { tile4x2_avx2(ra, rb, rc, rd, x0, x1) };
    }
    if has_avx1() {
        return unsafe { avx1::tile4x2(ra, rb, rc, rd, x0, x1) };
    }
    [dot_f32_scalar(ra, x0), dot_f32_scalar(rb, x0), dot_f32_scalar(rc, x0), dot_f32_scalar(rd, x0),
     dot_f32_scalar(ra, x1), dot_f32_scalar(rb, x1), dot_f32_scalar(rc, x1), dot_f32_scalar(rd, x1)]
}

#[inline]
fn axpy(y: &mut [f32], x: &[f32], a: f32) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::axpy(y, x, a) };
    }
    if has_avx1() {
        return unsafe { avx1::axpy(y, x, a) };
    }
    for (yi, xi) in y.iter_mut().zip(x) {
        *yi += a * xi;
    }
}

fn dot_bf16_scalar(w: &[u16], x: &[f32]) -> f32 {
    w.iter().zip(x).map(|(&a, &b)| bf16_to_f32(a) * b).sum()
}

fn dot_f32_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&a, &b)| a * b).sum()
}

#[cfg(target_arch = "x86_64")]
mod avx {
    use std::arch::x86_64::*;

    /// 8 bf16 → 8 f32: widen u16→u32 and shift left 16 (bf16 is the top half of f32).
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn load_bf16x8(p: *const u16) -> __m256 {
        let h = _mm_loadu_si128(p as *const __m128i);
        _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(h), 16))
    }

    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn hsum(v: __m256) -> f32 {
        let lo = _mm256_castps256_ps128(v);
        let hi = _mm256_extractf128_ps(v, 1);
        let s = _mm_add_ps(lo, hi);
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(load_bf16x8(w.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(load_bf16x8(w.as_ptr().add(i + 8)), _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(load_bf16x8(w.as_ptr().add(i + 16)), _mm256_loadu_ps(x.as_ptr().add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(load_bf16x8(w.as_ptr().add(i + 24)), _mm256_loadu_ps(x.as_ptr().add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(load_bf16x8(w.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        while i < n {
            s += super::bf16_to_f32(w[i]) * x[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (mut a0, mut a1, mut a2, mut a3) = (_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 32 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), a1);
            a2 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 16)), _mm256_loadu_ps(b.as_ptr().add(i + 16)), a2);
            a3 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i + 24)), _mm256_loadu_ps(b.as_ptr().add(i + 24)), a3);
            i += 32;
        }
        while i + 8 <= n {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), a0);
            i += 8;
        }
        let mut s = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dequant_bf16(w: &[u16], out: &mut [f32]) {
        let n = w.len();
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(out.as_mut_ptr().add(i), load_bf16x8(w.as_ptr().add(i)));
            i += 8;
        }
        while i < n {
            out[i] = super::bf16_to_f32(w[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dequant_q8_row(q: &[i8], scales: &[f32], out: &mut [f32]) {
        for (b, &sc) in scales.iter().enumerate() {
            let base = b * super::QBLOCK;
            let s = _mm256_set1_ps(sc);
            let mut j = 0;
            while j < super::QBLOCK {
                let q8 = _mm_loadl_epi64(q.as_ptr().add(base + j) as *const __m128i);
                let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                _mm256_storeu_ps(out.as_mut_ptr().add(base + j), _mm256_mul_ps(qf, s));
                j += 8;
            }
        }
    }

    /// One 32-weight Q4 block → two __m256 of 8 f32 each per call (caller loops 4x), via
    /// nibble unpack → epi32 → ps, minus the +8 offset.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn q4_block_to_f32(bytes: *const u8, out: &mut [__m256; 4]) {
        let raw = _mm_loadu_si128(bytes as *const __m128i);           // 16 bytes = 32 nibbles
        let lo = _mm_and_si128(raw, _mm_set1_epi8(0x0F));             // w[0..16]
        let hi = _mm_and_si128(_mm_srli_epi16(raw, 4), _mm_set1_epi8(0x0F)); // w[16..32]
        let eight = _mm256_set1_epi32(8);
        let cv = |v: __m128i| _mm256_cvtepi32_ps(_mm256_sub_epi32(_mm256_cvtepu8_epi32(v), eight));
        out[0] = cv(lo);
        out[1] = cv(_mm_srli_si128(lo, 8));
        out[2] = cv(hi);
        out[3] = cv(_mm_srli_si128(hi, 8));
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q4(q: &[u8], scales: &[f32], x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        let mut w = [_mm256_setzero_ps(); 4];
        for (b, &sc) in scales.iter().enumerate() {
            q4_block_to_f32(q.as_ptr().add(b * 16), &mut w);
            let xb = x.as_ptr().add(b * super::QBLOCK);
            let mut acc = _mm256_mul_ps(w[0], _mm256_loadu_ps(xb));
            acc = _mm256_fmadd_ps(w[1], _mm256_loadu_ps(xb.add(8)), acc);
            acc = _mm256_fmadd_ps(w[2], _mm256_loadu_ps(xb.add(16)), acc);
            acc = _mm256_fmadd_ps(w[3], _mm256_loadu_ps(xb.add(24)), acc);
            total = _mm256_fmadd_ps(acc, _mm256_set1_ps(sc), total);
        }
        hsum(total)
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dequant_q4_row(q: &[u8], scales: &[f32], out: &mut [f32]) {
        let mut w = [_mm256_setzero_ps(); 4];
        for (b, &sc) in scales.iter().enumerate() {
            q4_block_to_f32(q.as_ptr().add(b * 16), &mut w);
            let s = _mm256_set1_ps(sc);
            let o = out.as_mut_ptr().add(b * super::QBLOCK);
            _mm256_storeu_ps(o, _mm256_mul_ps(w[0], s));
            _mm256_storeu_ps(o.add(8), _mm256_mul_ps(w[1], s));
            _mm256_storeu_ps(o.add(16), _mm256_mul_ps(w[2], s));
            _mm256_storeu_ps(o.add(24), _mm256_mul_ps(w[3], s));
        }
    }

    /// y += a * x
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy(y: &mut [f32], x: &[f32], a: f32) {
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(av, _mm256_loadu_ps(x.as_ptr().add(i)), yv));
            i += 8;
        }
        while i < n {
            y[i] += a * x[i];
            i += 1;
        }
    }


    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_f32_i8(x: &[f32], q: &[i8]) -> f32 {
        let n = x.len();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 16 <= n {
            let q8 = _mm_loadu_si128(q.as_ptr().add(i) as *const __m128i);
            let lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            let hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q8, 8)));
            a0 = _mm256_fmadd_ps(lo, _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = _mm256_fmadd_ps(hi, _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += x[i] * *q.get_unchecked(i) as f32;
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_i8(y: &mut [f32], q: &[i8], a: f32) {
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            let q8 = _mm_loadl_epi64(q.as_ptr().add(i) as *const __m128i);
            let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), _mm256_fmadd_ps(av, qf, yv));
            i += 8;
        }
        while i < n {
            y[i] += a * *q.get_unchecked(i) as f32;
            i += 1;
        }
    }

    /// int8 dot with per-block scales. `maddubs` needs one unsigned operand and saturates at
    /// i16, so the standard trick is used: |w| (always < 128) against x*sign(w), whose products
    /// stay inside 127*127*2 < 32767. Sums are exact in int32; only the scales are float.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_i8(wq: &[i8], wsc: &[f32], xq: &[i8], xsc: &[f32]) -> f32 {
        let ones = _mm256_set1_epi16(1);
        let mut acc = _mm256_setzero_ps();
        for b in 0..wsc.len() {
            let w = _mm256_loadu_si256(wq.as_ptr().add(b * super::QBLOCK) as *const __m256i);
            let x = _mm256_loadu_si256(xq.as_ptr().add(b * super::QBLOCK) as *const __m256i);
            let aw = _mm256_sign_epi8(w, w);   // |w|, unsigned
            let sx = _mm256_sign_epi8(x, w);   // x * sign(w)
            let p = _mm256_maddubs_epi16(aw, sx);
            let s = _mm256_madd_epi16(p, ones);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(*wsc.get_unchecked(b) * *xsc.get_unchecked(b)), acc);
        }
        hsum(acc)
    }

    /// Q8 dot: per 32-block, widen int8→f32 in registers, FMA against x, then scale.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn dot_q8(q: &[i8], scales: &[f32], x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        for (b, &sc) in scales.iter().enumerate() {
            let base = b * super::QBLOCK;
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j < super::QBLOCK {
                let q8 = _mm_loadl_epi64(q.as_ptr().add(base + j) as *const __m128i);
                let qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q8));
                acc = _mm256_fmadd_ps(qf, _mm256_loadu_ps(x.as_ptr().add(base + j)), acc);
                j += 8;
            }
            total = _mm256_fmadd_ps(acc, _mm256_set1_ps(sc), total);
        }
        hsum(total)
    }

    /// 4 rows × 2 inputs: 8 accumulators, 6 loads per 8 FMAs. Fits in 16 ymm registers.
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn tile4x2(ra: &[f32], rb: &[f32], rc: &[f32], rd: &[f32], x0: &[f32], x1: &[f32]) -> [f32; 8] {
        let n = ra.len();
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i + 8 <= n {
            let p = _mm256_loadu_ps(x0.as_ptr().add(i));
            let q = _mm256_loadu_ps(x1.as_ptr().add(i));
            let a = _mm256_loadu_ps(ra.as_ptr().add(i));
            acc[0] = _mm256_fmadd_ps(a, p, acc[0]);
            acc[4] = _mm256_fmadd_ps(a, q, acc[4]);
            let b = _mm256_loadu_ps(rb.as_ptr().add(i));
            acc[1] = _mm256_fmadd_ps(b, p, acc[1]);
            acc[5] = _mm256_fmadd_ps(b, q, acc[5]);
            let c = _mm256_loadu_ps(rc.as_ptr().add(i));
            acc[2] = _mm256_fmadd_ps(c, p, acc[2]);
            acc[6] = _mm256_fmadd_ps(c, q, acc[6]);
            let d = _mm256_loadu_ps(rd.as_ptr().add(i));
            acc[3] = _mm256_fmadd_ps(d, p, acc[3]);
            acc[7] = _mm256_fmadd_ps(d, q, acc[7]);
            i += 8;
        }
        let mut out = [0f32; 8];
        for t in 0..8 {
            out[t] = hsum(acc[t]);
        }
        while i < n {
            out[0] += ra[i] * x0[i];
            out[1] += rb[i] * x0[i];
            out[2] += rc[i] * x0[i];
            out[3] += rd[i] * x0[i];
            out[4] += ra[i] * x1[i];
            out[5] += rb[i] * x1[i];
            out[6] += rc[i] * x1[i];
            out[7] += rd[i] * x1[i];
            i += 1;
        }
        out
    }
}


/// AVX (2011 Sandy Bridge / 2013 Ivy Bridge Xeons): 256-bit float math but no FMA and no
/// 256-bit integer ops, so conversions go through 128-bit SSE4.1 and get stitched together.
#[cfg(target_arch = "x86_64")]
mod avx1 {
    use std::arch::x86_64::*;

    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn join(lo: __m128, hi: __m128) -> __m256 {
        _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1)
    }

    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn fma(a: __m256, b: __m256, c: __m256) -> __m256 {
        _mm256_add_ps(_mm256_mul_ps(a, b), c)
    }

    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn hsum(v: __m256) -> f32 {
        let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
        let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        let s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
        _mm_cvtss_f32(s)
    }

    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn load_bf16x8(p: *const u16) -> __m256 {
        let h = _mm_loadu_si128(p as *const __m128i);
        let lo = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(h), 16));
        let hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_cvtepu16_epi32(_mm_srli_si128(h, 8)), 16));
        join(lo, hi)
    }

    /// 8 int8 → 8 f32
    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn load_i8x8(p: *const i8) -> __m256 {
        let v = _mm_loadl_epi64(p as *const __m128i);
        join(_mm_cvtepi32_ps(_mm_cvtepi8_epi32(v)), _mm_cvtepi32_ps(_mm_cvtepi8_epi32(_mm_srli_si128(v, 4))))
    }

    /// 16 unsigned nibble codes (in a __m128i, one per byte, 0..15) → two __m256 of (code - 8)
    #[inline]
    #[target_feature(enable = "avx,sse4.1")]
    unsafe fn nib16_to_f32(v: __m128i) -> (__m256, __m256) {
        let eight = _mm_set1_epi32(8);
        let c = |x: __m128i| _mm_cvtepi32_ps(_mm_sub_epi32(_mm_cvtepu8_epi32(x), eight));
        (join(c(v), c(_mm_srli_si128(v, 4))), join(c(_mm_srli_si128(v, 8)), c(_mm_srli_si128(v, 12))))
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dot_bf16(w: &[u16], x: &[f32]) -> f32 {
        let n = w.len();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 16 <= n {
            a0 = fma(load_bf16x8(w.as_ptr().add(i)), _mm256_loadu_ps(x.as_ptr().add(i)), a0);
            a1 = fma(load_bf16x8(w.as_ptr().add(i + 8)), _mm256_loadu_ps(x.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += super::bf16_to_f32(w[i]) * x[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let n = a.len();
        let (mut a0, mut a1) = (_mm256_setzero_ps(), _mm256_setzero_ps());
        let mut i = 0;
        while i + 16 <= n {
            a0 = fma(_mm256_loadu_ps(a.as_ptr().add(i)), _mm256_loadu_ps(b.as_ptr().add(i)), a0);
            a1 = fma(_mm256_loadu_ps(a.as_ptr().add(i + 8)), _mm256_loadu_ps(b.as_ptr().add(i + 8)), a1);
            i += 16;
        }
        let mut s = hsum(_mm256_add_ps(a0, a1));
        while i < n {
            s += a[i] * b[i];
            i += 1;
        }
        s
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dequant_bf16(w: &[u16], out: &mut [f32]) {
        let n = w.len();
        let mut i = 0;
        while i + 8 <= n {
            _mm256_storeu_ps(out.as_mut_ptr().add(i), load_bf16x8(w.as_ptr().add(i)));
            i += 8;
        }
        while i < n {
            out[i] = super::bf16_to_f32(w[i]);
            i += 1;
        }
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dequant_q8_row(q: &[i8], scales: &[f32], out: &mut [f32]) {
        for (b, &sc) in scales.iter().enumerate() {
            let base = b * super::QBLOCK;
            let s = _mm256_set1_ps(sc);
            let mut j = 0;
            while j < super::QBLOCK {
                _mm256_storeu_ps(out.as_mut_ptr().add(base + j), _mm256_mul_ps(load_i8x8(q.as_ptr().add(base + j)), s));
                j += 8;
            }
        }
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dot_q8(q: &[i8], scales: &[f32], x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        for (b, &sc) in scales.iter().enumerate() {
            let base = b * super::QBLOCK;
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j < super::QBLOCK {
                acc = fma(load_i8x8(q.as_ptr().add(base + j)), _mm256_loadu_ps(x.as_ptr().add(base + j)), acc);
                j += 8;
            }
            total = fma(acc, _mm256_set1_ps(sc), total);
        }
        hsum(total)
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dot_q4(q: &[u8], scales: &[f32], x: &[f32]) -> f32 {
        let mut total = _mm256_setzero_ps();
        let mask = _mm_set1_epi8(0x0F);
        for (b, &sc) in scales.iter().enumerate() {
            let raw = _mm_loadu_si128(q.as_ptr().add(b * 16) as *const __m128i);
            let (w0, w1) = nib16_to_f32(_mm_and_si128(raw, mask));
            let (w2, w3) = nib16_to_f32(_mm_and_si128(_mm_srli_epi16(raw, 4), mask));
            let xb = x.as_ptr().add(b * super::QBLOCK);
            let mut acc = _mm256_mul_ps(w0, _mm256_loadu_ps(xb));
            acc = fma(w1, _mm256_loadu_ps(xb.add(8)), acc);
            acc = fma(w2, _mm256_loadu_ps(xb.add(16)), acc);
            acc = fma(w3, _mm256_loadu_ps(xb.add(24)), acc);
            total = fma(acc, _mm256_set1_ps(sc), total);
        }
        hsum(total)
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn dequant_q4_row(q: &[u8], scales: &[f32], out: &mut [f32]) {
        let mask = _mm_set1_epi8(0x0F);
        for (b, &sc) in scales.iter().enumerate() {
            let raw = _mm_loadu_si128(q.as_ptr().add(b * 16) as *const __m128i);
            let (w0, w1) = nib16_to_f32(_mm_and_si128(raw, mask));
            let (w2, w3) = nib16_to_f32(_mm_and_si128(_mm_srli_epi16(raw, 4), mask));
            let s = _mm256_set1_ps(sc);
            let o = out.as_mut_ptr().add(b * super::QBLOCK);
            _mm256_storeu_ps(o, _mm256_mul_ps(w0, s));
            _mm256_storeu_ps(o.add(8), _mm256_mul_ps(w1, s));
            _mm256_storeu_ps(o.add(16), _mm256_mul_ps(w2, s));
            _mm256_storeu_ps(o.add(24), _mm256_mul_ps(w3, s));
        }
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn axpy(y: &mut [f32], x: &[f32], a: f32) {
        let n = y.len();
        let av = _mm256_set1_ps(a);
        let mut i = 0;
        while i + 8 <= n {
            let yv = _mm256_loadu_ps(y.as_ptr().add(i));
            _mm256_storeu_ps(y.as_mut_ptr().add(i), fma(av, _mm256_loadu_ps(x.as_ptr().add(i)), yv));
            i += 8;
        }
        while i < n {
            y[i] += a * x[i];
            i += 1;
        }
    }

    #[target_feature(enable = "avx,sse4.1")]
    pub unsafe fn tile4x2(ra: &[f32], rb: &[f32], rc: &[f32], rd: &[f32], x0: &[f32], x1: &[f32]) -> [f32; 8] {
        let n = ra.len();
        let mut acc = [_mm256_setzero_ps(); 8];
        let mut i = 0;
        while i + 8 <= n {
            let p = _mm256_loadu_ps(x0.as_ptr().add(i));
            let q = _mm256_loadu_ps(x1.as_ptr().add(i));
            let a = _mm256_loadu_ps(ra.as_ptr().add(i));
            acc[0] = fma(a, p, acc[0]);
            acc[4] = fma(a, q, acc[4]);
            let b = _mm256_loadu_ps(rb.as_ptr().add(i));
            acc[1] = fma(b, p, acc[1]);
            acc[5] = fma(b, q, acc[5]);
            let c = _mm256_loadu_ps(rc.as_ptr().add(i));
            acc[2] = fma(c, p, acc[2]);
            acc[6] = fma(c, q, acc[6]);
            let d = _mm256_loadu_ps(rd.as_ptr().add(i));
            acc[3] = fma(d, p, acc[3]);
            acc[7] = fma(d, q, acc[7]);
            i += 8;
        }
        let mut out = [0f32; 8];
        for t in 0..8 {
            out[t] = hsum(acc[t]);
        }
        while i < n {
            out[0] += ra[i] * x0[i];
            out[1] += rb[i] * x0[i];
            out[2] += rc[i] * x0[i];
            out[3] += rd[i] * x0[i];
            out[4] += ra[i] * x1[i];
            out[5] += rb[i] * x1[i];
            out[6] += rc[i] * x1[i];
            out[7] += rd[i] * x1[i];
            i += 1;
        }
        out
    }
}

#[cfg(target_arch = "x86_64")]
use avx::{dot_bf16 as dot_bf16_avx2, dot_f32 as dot_f32_avx2, tile4x2 as tile4x2_avx2};

// ---------------------------------------------------------------------------------------
// Q8 block quantization: blocks of 32 int8 weights sharing one f32 scale (1.125 bytes/param).
// Decode is bandwidth-bound, so ~1.8x fewer bytes ≈ ~1.8x faster decode, at negligible quality cost.
// ---------------------------------------------------------------------------------------

pub const QBLOCK: usize = 32;

pub struct QMat {
    pub n: usize,
    pub k: usize,
    pub q: Vec<i8>,        // n * k
    pub scales: Vec<f32>,  // n * k / QBLOCK
}

impl QMat {
    pub fn from_bf16(w: &[u16], n: usize, k: usize) -> QMat {
        assert_eq!(w.len(), n * k);
        assert_eq!(k % QBLOCK, 0, "k must be a multiple of {QBLOCK}");
        let blocks = k / QBLOCK;
        let mut q: Vec<i8> = alloc_rows(n * k, n);
        let mut scales: Vec<f32> = alloc_rows(n * blocks, n);
        let qp = SendPtr(q.as_mut_ptr() as *mut f32); // reinterpreted below; only used for address math
        let sp = SendPtr(scales.as_mut_ptr());
        pool::global().run_static((n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK, &|c| {
            let r0 = c * ROWS_PER_CHUNK;
            let r1 = (r0 + ROWS_PER_CHUNK).min(n);
            let mut f = [0f32; QBLOCK];
            for i in r0..r1 {
                for b in 0..blocks {
                    let off = i * k + b * QBLOCK;
                    let mut amax = 0f32;
                    for j in 0..QBLOCK {
                        f[j] = bf16_to_f32(w[off + j]);
                        amax = amax.max(f[j].abs());
                    }
                    let scale = amax / 127.0;
                    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
                    unsafe {
                        *sp.get().add(i * blocks + b) = scale;
                        let qb = (qp.get() as *mut i8).add(off);
                        for j in 0..QBLOCK {
                            *qb.add(j) = (f[j] * inv).round().clamp(-127.0, 127.0) as i8;
                        }
                    }
                }
            }
        });
        QMat { n, k, q, scales }
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }

    /// Dequantize one row into f32.
    #[inline]
    pub fn row_f32(&self, i: usize, out: &mut [f32]) {
        let blocks = self.k / QBLOCK;
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            return unsafe { avx::dequant_q8_row(&self.q[i * self.k..(i + 1) * self.k], &self.scales[i * blocks..(i + 1) * blocks], out) };
        }
        if has_avx1() {
            return unsafe { avx1::dequant_q8_row(&self.q[i * self.k..(i + 1) * self.k], &self.scales[i * blocks..(i + 1) * blocks], out) };
        }
        for b in 0..blocks {
            let s = self.scales[i * blocks + b];
            let base = i * self.k + b * QBLOCK;
            for j in 0..QBLOCK {
                out[b * QBLOCK + j] = self.q[base + j] as f32 * s;
            }
        }
    }
}

pub fn matvec_q8(w: &QMat, x: &[f32], y: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    let yp = SendPtr(y.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        let blocks = k / QBLOCK;
        for i in r0..r1 {
            let v = dot_q8(&w.q[i * k..(i + 1) * k], &w.scales[i * blocks..(i + 1) * blocks], x);
            unsafe { *yp.get().add(i) = v };
        }
    });
}

/// Prefill path: dequantize each chunk of rows to f32 once, then the same register-tiled kernel.
pub fn matmul_q8(w: &QMat, xs: &[f32], m: usize, ys: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    if m == 1 {
        return matvec_q8(w, xs, ys);
    }
    let yp = SendPtr(ys.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        with_scratch((r1 - r0) * k, |rows| {
            for i in r0..r1 {
                w.row_f32(i, &mut rows[(i - r0) * k..(i - r0 + 1) * k]);
            }
            tile_rows(rows, r0, r1, xs, m, n, k, yp);
        });
    });
}

#[inline]
fn dot_q8(q: &[i8], scales: &[f32], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::dot_q8(q, scales, x) };
    }
    if has_avx1() {
        return unsafe { avx1::dot_q8(q, scales, x) };
    }
    let mut s = 0f32;
    for (b, &sc) in scales.iter().enumerate() {
        let mut acc = 0f32;
        for j in 0..QBLOCK {
            acc += q[b * QBLOCK + j] as f32 * x[b * QBLOCK + j];
        }
        s += acc * sc;
    }
    s
}

/// Attention for one query token over cache positions 0..=pos, all heads in parallel.
/// q: [heads*hd], kcache/vcache rows at (t * stride + kvi*hd), out: [heads*hd].
#[allow(clippy::too_many_arguments)]
pub fn attention(q: &[f32], kc: &[f32], vc: &[f32], stride: usize, pos: usize, heads: usize, kv_heads: usize, hd: usize, out: &mut [f32]) {
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let op = SendPtr(out.as_mut_ptr());
    pool::global().run(heads, &|hi| {
        let kvi = hi / group;
        let qh = &q[hi * hd..(hi + 1) * hd];
        let mut scores = vec![0f32; pos + 1];
        for t in 0..=pos {
            let kb = t * stride + kvi * hd;
            scores[t] = dot_f32(qh, &kc[kb..kb + hd]) * scale;
        }
        softmax(&mut scores);
        let mut o = vec![0f32; hd];
        for t in 0..=pos {
            let vb = t * stride + kvi * hd;
            let vt = &vc[vb..vb + hd];
            let p = scores[t];
            for d in 0..hd {
                o[d] += p * vt[d];
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), op.get().add(hi * hd), hd) };
    });
}

// ---------------------------------------------------------------------------------------
// Q4 block quantization: 32 weights share one f32 scale, each weight is a 4-bit code in
// [-8, 7] stored as an unsigned nibble (code + 8). 0.625 bytes/param: a 30B model in ~19 GB.
// ---------------------------------------------------------------------------------------

pub struct Q4Mat {
    pub n: usize,
    pub k: usize,
    pub q: Vec<u8>,       // n * k / 2, two nibbles per byte; byte i of a block holds w[i] (lo) and w[i+16] (hi)
    pub scales: Vec<f32>, // n * k / QBLOCK
}

impl Q4Mat {
    pub fn from_bf16(w: &[u16], n: usize, k: usize) -> Q4Mat {
        assert_eq!(w.len(), n * k);
        assert_eq!(k % QBLOCK, 0);
        let blocks = k / QBLOCK;
        let mut q: Vec<u8> = alloc_rows(n * k / 2, n);
        let mut scales: Vec<f32> = alloc_rows(n * blocks, n);
        let qp = SendPtr(q.as_mut_ptr() as *mut f32);
        let sp = SendPtr(scales.as_mut_ptr());
        pool::global().run_static((n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK, &|c| {
            let r0 = c * ROWS_PER_CHUNK;
            let r1 = (r0 + ROWS_PER_CHUNK).min(n);
            let mut f = [0f32; QBLOCK];
            for i in r0..r1 {
                for b in 0..blocks {
                    let off = i * k + b * QBLOCK;
                    let mut amax = 0f32;
                    let mut maxv = 0f32;
                    for j in 0..QBLOCK {
                        f[j] = bf16_to_f32(w[off + j]);
                        if f[j].abs() > amax {
                            amax = f[j].abs();
                            maxv = f[j];
                        }
                    }
                    // Map the largest-magnitude value to -8 (uses the full signed range, like llama.cpp Q4_0).
                    let scale = maxv / -8.0;
                    let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
                    unsafe {
                        *sp.get().add(i * blocks + b) = scale;
                        let qb = (qp.get() as *mut u8).add(off / 2);
                        for j in 0..QBLOCK / 2 {
                            let lo = ((f[j] * inv + 8.5).floor().clamp(0.0, 15.0)) as u8;
                            let hi = ((f[j + 16] * inv + 8.5).floor().clamp(0.0, 15.0)) as u8;
                            *qb.add(j) = lo | (hi << 4);
                        }
                    }
                }
            }
        });
        Q4Mat { n, k, q, scales }
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + self.scales.len() * 4
    }

    #[inline]
    pub fn row_f32(&self, i: usize, out: &mut [f32]) {
        let blocks = self.k / QBLOCK;
        #[cfg(target_arch = "x86_64")]
        if has_avx2() {
            return unsafe { avx::dequant_q4_row(&self.q[i * self.k / 2..(i + 1) * self.k / 2], &self.scales[i * blocks..(i + 1) * blocks], out) };
        }
        if has_avx1() {
            return unsafe { avx1::dequant_q4_row(&self.q[i * self.k / 2..(i + 1) * self.k / 2], &self.scales[i * blocks..(i + 1) * blocks], out) };
        }
        for b in 0..blocks {
            let s = self.scales[i * blocks + b];
            let base = i * self.k / 2 + b * 16;
            for j in 0..16 {
                let byte = self.q[base + j];
                out[b * QBLOCK + j] = ((byte & 0x0F) as i32 - 8) as f32 * s;
                out[b * QBLOCK + j + 16] = ((byte >> 4) as i32 - 8) as f32 * s;
            }
        }
    }
}

pub fn matvec_q4(w: &Q4Mat, x: &[f32], y: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    let yp = SendPtr(y.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        let blocks = k / QBLOCK;
        for i in r0..r1 {
            let v = dot_q4(&w.q[i * k / 2..(i + 1) * k / 2], &w.scales[i * blocks..(i + 1) * blocks], x);
            unsafe { *yp.get().add(i) = v };
        }
    });
}

pub fn matmul_q4(w: &Q4Mat, xs: &[f32], m: usize, ys: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    if m == 1 {
        return matvec_q4(w, xs, ys);
    }
    let yp = SendPtr(ys.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        with_scratch((r1 - r0) * k, |rows| {
            for i in r0..r1 {
                w.row_f32(i, &mut rows[(i - r0) * k..(i - r0 + 1) * k]);
            }
            tile_rows(rows, r0, r1, xs, m, n, k, yp);
        });
    });
}

#[inline]
fn dot_q4(q: &[u8], scales: &[f32], x: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::dot_q4(q, scales, x) };
    }
    if has_avx1() {
        return unsafe { avx1::dot_q4(q, scales, x) };
    }
    let mut s = 0f32;
    for (b, &sc) in scales.iter().enumerate() {
        let mut acc = 0f32;
        for j in 0..16 {
            let byte = q[b * 16 + j];
            acc += ((byte & 0x0F) as i32 - 8) as f32 * x[b * QBLOCK + j];
            acc += ((byte >> 4) as i32 - 8) as f32 * x[b * QBLOCK + j + 16];
        }
        s += acc * sc;
    }
    s
}


// ---------------------------------------------------------------------------------------
// int8 GEMM for prefill. The f32 path dequantizes every weight to float and multiplies in
// floats; here both sides stay 8-bit and accumulate in int32, which is what the hardware is
// fastest at. Activations are quantized per 32-block exactly like the weights, so the product
// of the two block scales rescales an exact int32 dot product.
// ---------------------------------------------------------------------------------------

/// Activations quantized to int8, blocks of QBLOCK sharing one scale.
pub struct Q8Act {
    pub q: Vec<i8>,
    pub scales: Vec<f32>,
    pub k: usize,
}

pub fn quantize_act(xs: &[f32], m: usize, k: usize) -> Q8Act {
    let blocks = k / QBLOCK;
    let mut q = vec![0i8; m * k];
    let mut scales = vec![0f32; m * blocks];
    for j in 0..m {
        for b in 0..blocks {
            let off = j * k + b * QBLOCK;
            let mut amax = 0f32;
            for i in 0..QBLOCK {
                amax = amax.max(xs[off + i].abs());
            }
            let scale = amax / 127.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            scales[j * blocks + b] = scale;
            for i in 0..QBLOCK {
                q[off + i] = (xs[off + i] * inv).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    Q8Act { q, scales, k }
}

/// Y = X · Wᵀ with both sides int8. Weight rows stay in cache across all m activations.
pub fn matmul_q8_int8(w: &QMat, x: &Q8Act, m: usize, ys: &mut [f32]) {
    let (n, k) = (w.n, w.k);
    debug_assert_eq!(x.k, k);
    let blocks = k / QBLOCK;
    let yp = SendPtr(ys.as_mut_ptr());
    let chunks = (n + ROWS_PER_CHUNK - 1) / ROWS_PER_CHUNK;
    pool::global().run_static(chunks, &|c| {
        let r0 = c * ROWS_PER_CHUNK;
        let r1 = (r0 + ROWS_PER_CHUNK).min(n);
        for i in r0..r1 {
            let wq = &w.q[i * k..(i + 1) * k];
            let wsc = &w.scales[i * blocks..(i + 1) * blocks];
            for j in 0..m {
                let xq = &x.q[j * k..(j + 1) * k];
                let xsc = &x.scales[j * blocks..(j + 1) * blocks];
                let v = dot_i8(wq, wsc, xq, xsc);
                unsafe { *yp.get().add(j * n + r0 + (i - r0)) = v };
            }
        }
    });
}

#[inline]
fn dot_i8(wq: &[i8], wsc: &[f32], xq: &[i8], xsc: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::dot_i8(wq, wsc, xq, xsc) };
    }
    let mut total = 0f32;
    for (b, (&ws, &xs)) in wsc.iter().zip(xsc).enumerate() {
        let mut acc = 0i32;
        for i in 0..QBLOCK {
            acc += wq[b * QBLOCK + i] as i32 * xq[b * QBLOCK + i] as i32;
        }
        total += acc as f32 * ws * xs;
    }
    total
}

/// dot(x_f32, q_i8) * scale — used to score int8 KV rows without materializing them as f32.
#[inline]
pub fn dot_f32_i8(x: &[f32], q: &[i8], scale: f32) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::dot_f32_i8(x, q) } * scale;
    }
    let mut s = 0f32;
    for i in 0..x.len() {
        s += x[i] * q[i] as f32;
    }
    s * scale
}

/// y += a * q_i8
#[inline]
pub fn axpy_i8(y: &mut [f32], q: &[i8], a: f32) {
    #[cfg(target_arch = "x86_64")]
    if has_avx2() {
        return unsafe { avx::axpy_i8(y, q, a) };
    }
    for (yi, &qi) in y.iter_mut().zip(q) {
        *yi += a * qi as f32;
    }
}

/// Attention over an int8 KV cache. Identical math to attention_multi; K and V rows carry one
/// scale per (position, kv head), so the cache costs ~1.03 bytes per value instead of 4.
#[allow(clippy::too_many_arguments)]
pub fn attention_multi_q8(q: &[f32], kv: &[(&[i8], &[f32], &[i8], &[f32], usize)], stride: usize, heads: usize, kv_heads: usize, hd: usize, out: &mut [f32]) {
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let qd = heads * hd;
    let op = SendPtr(out.as_mut_ptr());
    let m = kv.len();
    pool::global().run(m * heads, &|c| {
        let (j, hi) = (c / heads, c % heads);
        let (kc, ks, vc, vs, pos) = kv[j];
        let kvi = hi / group;
        let qh = &q[j * qd + hi * hd..j * qd + (hi + 1) * hd];
        let mut scores = vec![0f32; pos + 1];
        for t in 0..=pos {
            let kb = t * stride + kvi * hd;
            scores[t] = dot_f32_i8(qh, &kc[kb..kb + hd], ks[t * kv_heads + kvi]) * scale;
        }
        softmax(&mut scores);
        let mut o = vec![0f32; hd];
        for t in 0..=pos {
            let vb = t * stride + kvi * hd;
            axpy_i8(&mut o, &vc[vb..vb + hd], scores[t] * vs[t * kv_heads + kvi]);
        }
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), op.get().add(j * qd + hi * hd), hd) };
    });
}

/// Quantize one head vector to int8, returning its scale.
pub fn quantize_head(x: &[f32], out: &mut [i8]) -> f32 {
    let amax = x.iter().fold(0f32, |a, &b| a.max(b.abs()));
    let scale = amax / 127.0;
    let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    for (o, &v) in out.iter_mut().zip(x) {
        *o = (v * inv).round().clamp(-127.0, 127.0) as i8;
    }
    scale
}

/// Attention for many query items at once: one pool job over (item x head) pairs.
/// kv[j] = (k rows, v rows, pos) for item j; q/out rows are heads*hd per item.
pub fn attention_multi(q: &[f32], kv: &[(&[f32], &[f32], usize)], stride: usize, heads: usize, kv_heads: usize, hd: usize, out: &mut [f32]) {
    let group = heads / kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let qd = heads * hd;
    let op = SendPtr(out.as_mut_ptr());
    let m = kv.len();
    pool::global().run(m * heads, &|c| {
        let (j, hi) = (c / heads, c % heads);
        let (kc, vc, pos) = kv[j];
        let kvi = hi / group;
        let qh = &q[j * qd + hi * hd..j * qd + (hi + 1) * hd];
        let mut scores = vec![0f32; pos + 1];
        for t in 0..=pos {
            let kb = t * stride + kvi * hd;
            scores[t] = dot_f32(qh, &kc[kb..kb + hd]) * scale;
        }
        softmax(&mut scores);
        let mut o = vec![0f32; hd];
        for t in 0..=pos {
            let vb = t * stride + kvi * hd;
            axpy(&mut o, &vc[vb..vb + hd], scores[t]);
        }
        unsafe { std::ptr::copy_nonoverlapping(o.as_ptr(), op.get().add(j * qd + hi * hd), hd) };
    });
}

pub fn rmsnorm(x: &mut [f32], weight: &[f32], eps: f32) {
    let ss = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let scale = 1.0 / (ss + eps).sqrt();
    for (v, w) in x.iter_mut().zip(weight) {
        *v = *v * scale * w;
    }
}

pub fn softmax(x: &mut [f32]) {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Rotary embedding with precomputed cos/sin tables (len d/2).
pub fn rope_tab(v: &mut [f32], cos: &[f32], sin: &[f32]) {
    let half = v.len() / 2;
    for i in 0..half {
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * cos[i] - b * sin[i];
        v[i + half] = a * sin[i] + b * cos[i];
    }
}

/// Rotary embedding, HF "rotate_half" layout: pairs (i, i + d/2).
/// Reference RoPE, kept as the readable statement of what the fused paths implement.
#[allow(dead_code)]
pub fn rope(v: &mut [f32], pos: usize, theta: f32) {
    let d = v.len();
    let half = d / 2;
    for i in 0..half {
        let freq = 1.0 / theta.powf(2.0 * i as f32 / d as f32);
        let (s, c) = (pos as f32 * freq).sin_cos();
        let (a, b) = (v[i], v[i + half]);
        v[i] = a * c - b * s;
        v[i + half] = a * s + b * c;
    }
}

#[cfg(test)]
mod tests {

    // Placement must never change contents: these run over sizes that cross the row-chunk
    // boundary and sizes that cannot be split by rows at all (the serial fallback).
    #[test]
    fn copy_rows_matches_memcpy() {
        for (rows, row_bytes) in [(1usize, 7usize), (16, 4), (17, 64), (129, 96), (64, 4096)] {
            let src: Vec<u8> = (0..rows * row_bytes).map(|i| (i * 31 % 251) as u8).collect();
            let got: Vec<u8> = super::copy_rows(&src, rows);
            assert_eq!(got, src, "rows={rows} row_bytes={row_bytes}");
            // A row count the buffer does not divide by falls back to a plain copy.
            let odd: Vec<u8> = super::copy_rows(&src, rows * 2 + 1);
            assert_eq!(odd, src, "fallback rows={rows}");
        }
    }

    #[test]
    fn copy_rows_preserves_element_type() {
        let src: Vec<f32> = (0..1024).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, src.len() * 4) };
        let got: Vec<f32> = super::copy_rows(bytes, 64);
        assert_eq!(got, src);
    }

    #[test]
    fn alloc_rows_is_zeroed() {
        for (elems, rows) in [(0usize, 0usize), (16, 16), (4096, 128), (1000, 8)] {
            let v: Vec<i8> = super::alloc_rows(elems, rows);
            assert_eq!(v.len(), elems);
            assert!(v.iter().all(|&x| x == 0), "elems={elems} rows={rows}");
        }
    }

    // The whole point: the same weights must produce the same numbers whichever node their
    // pages live on. Quantize twice through the two allocation paths and compare exactly.
    #[test]
    fn numa_placement_does_not_change_results() {
        let (n, k) = (64usize, 128usize);
        let w: Vec<u16> = (0..n * k).map(|i| ((i * 2654435761usize >> 13) % 4001) as u16 + 15000).collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 11) as f32 * 0.1 - 0.5).collect();
        let q = super::QMat::from_bf16(&w, n, k);
        let mut a = vec![0f32; n];
        super::matvec_q8(&q, &x, &mut a);

        // Round-trip through the cache representation, which is where copy_rows runs.
        let bytes: Vec<u8> = q.q.iter().map(|&b| b as u8)
            .chain(q.scales.iter().flat_map(|s| s.to_le_bytes()))
            .collect();
        let (qb, sb) = bytes.split_at(n * k);
        let q2 = super::QMat { n, k, q: super::copy_rows(qb, n), scales: super::copy_rows(sb, n) };
        let mut b = vec![0f32; n];
        super::matvec_q8(&q2, &x, &mut b);
        assert_eq!(a, b, "weight placement changed the result");
    }
    use super::*;

    fn f2b(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }

    #[test]
    fn matvec_identity() {
        let w: Vec<u16> = [1.0, 0.0, 0.0, 1.0].iter().map(|&f| f2b(f)).collect();
        let mut y = [0.0; 2];
        matvec_bf16(&w, &[3.0, 4.0], 2, 2, &mut y);
        assert_eq!(y, [3.0, 4.0]);
    }

    #[test]
    fn matvec_threaded_matches_serial() {
        let (n, k) = (512, 256);
        let w: Vec<u16> = (0..n * k).map(|i| f2b(((i % 13) as f32 - 6.0) * 0.25)).collect();
        let x: Vec<f32> = (0..k).map(|i| (i % 7) as f32 * 0.5 - 1.0).collect();
        let mut y = vec![0.0; n];
        matvec_bf16(&w, &x, n, k, &mut y);
        for i in 0..n {
            let expect = dot_bf16(&w[i * k..(i + 1) * k], &x);
            assert!((y[i] - expect).abs() < 1e-3);
        }
    }

    #[test]
    fn matmul_matches_matvec() {
        let (m, n, k) = (7, 300, 200);
        let w: Vec<u16> = (0..n * k).map(|i| f2b(((i % 11) as f32 - 5.0) * 0.125)).collect();
        let xs: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 5) as f32 * 0.3 - 0.6).collect();
        let mut ys = vec![0.0; m * n];
        matmul_bf16(&w, &xs, m, n, k, &mut ys);
        for j in 0..m {
            let mut y = vec![0.0; n];
            matvec_bf16(&w, &xs[j * k..(j + 1) * k], n, k, &mut y);
            for i in 0..n {
                assert!((ys[j * n + i] - y[i]).abs() < 1e-3, "row {j} col {i}: {} vs {}", ys[j * n + i], y[i]);
            }
        }
    }

    #[test]
    fn quantized_dots_match_reference() {
        let (n, k) = (64, 256);
        let w: Vec<u16> = (0..n * k).map(|i| f2b((((i * 7919) % 2001) as f32 / 1000.0 - 1.0) * 0.3)).collect();
        let x: Vec<f32> = (0..k).map(|i| ((i * 31) % 17) as f32 * 0.1 - 0.8).collect();
        let q8 = QMat::from_bf16(&w, n, k);
        let q4 = Q4Mat::from_bf16(&w, n, k);
        let (mut y8, mut y4, mut yb) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
        matvec_q8(&q8, &x, &mut y8);
        matvec_q4(&q4, &x, &mut y4);
        matvec_bf16(&w, &x, n, k, &mut yb);
        let mut row = vec![0.0; k];
        for i in 0..n {
            // q8 must match its own dequantized row exactly-ish; q4 is lossier but must be close
            q8.row_f32(i, &mut row);
            let r8: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((y8[i] - r8).abs() < 1e-2, "q8 row {i}: {} vs {}", y8[i], r8);
            q4.row_f32(i, &mut row);
            let r4: f32 = row.iter().zip(&x).map(|(a, b)| a * b).sum();
            assert!((y4[i] - r4).abs() < 1e-2, "q4 row {i}: {} vs {}", y4[i], r4);
            assert!((y8[i] - yb[i]).abs() < 0.05 * yb[i].abs().max(1.0), "q8 vs bf16 row {i}: {} vs {}", y8[i], yb[i]);
        }
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = [1.0, 2.0, 3.0];
        softmax(&mut x);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bf16_roundtrip() {
        assert_eq!(bf16_to_f32(0x3F80), 1.0);
        assert_eq!(bf16_to_f32(0xC000), -2.0);
    }
}
