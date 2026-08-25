// llmfa.st GPU kernels. Q8 weights: 32-weight blocks, int8 codes packed 4 per u32, one f32 scale per block.

struct MatParams { n: u32, k: u32, m: u32, _pad: u32 };
struct LayerParams { m: u32, hidden: u32, heads: u32, kv_heads: u32, head_dim: u32, inter: u32, eps: f32, _p: u32 };

@group(0) @binding(0) var<storage, read> w_q: array<u32>;      // n*k/4 words
@group(0) @binding(1) var<storage, read> w_s: array<f32>;      // n*k/32 scales
@group(0) @binding(2) var<storage, read> xin: array<f32>;      // m*k
@group(0) @binding(3) var<storage, read_write> yout: array<f32>; // m*n
@group(0) @binding(4) var<uniform> mp: MatParams;
@group(0) @binding(5) var<uniform> mcount: LayerParams; // .m = rows of x this step

fn unpack4(word: u32) -> vec4<f32> {
    // four int8 codes, little-endian byte order, sign-extended
    let b0 = (i32(word << 24u)) >> 24u;
    let b1 = (i32(word << 16u)) >> 24u;
    let b2 = (i32(word << 8u)) >> 24u;
    let b3 = (i32(word)) >> 24u;
    return vec4<f32>(f32(b0), f32(b1), f32(b2), f32(b3));
}

// Interleaved layout: for a tile of 64 rows, word w of row r is at ((tile*words + w)*64 + r%64),
// so lanes reading the same word land on adjacent addresses → coalesced.
// Workgroup = 64 rows × KSPLIT k-partitions (512 threads); partials reduced in shared memory.
const KSPLIT: u32 = 4u;
var<workgroup> part: array<f32, 256>;

@compute @workgroup_size(256)
fn matvec_q8(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x % 64u;
    let p = lid.x / 64u;
    let tile = wid.x;
    let row = tile * 64u + lane;
    let words = mp.k / 4u;
    let blocks = mp.k / 32u;
    let wtile = tile * words * 64u + lane;
    let stile = tile * blocks * 64u + lane;
    for (var j = 0u; j < mcount.m; j = j + 1u) {
        var acc = 0.0;
        let xb = j * mp.k;
        for (var b = p; b < blocks; b = b + KSPLIT) {
            var bacc = 0.0;
            let wb = wtile + b * 8u * 64u;
            let kb = xb + b * 32u;
            for (var i = 0u; i < 8u; i = i + 1u) {
                let w4 = unpack4(w_q[wb + i * 64u]);
                let kk = kb + i * 4u;
                bacc = bacc + dot(w4, vec4<f32>(xin[kk], xin[kk + 1u], xin[kk + 2u], xin[kk + 3u]));
            }
            acc = acc + bacc * w_s[stile + b * 64u];
        }
        part[lid.x] = acc;
        workgroupBarrier();
        if (p == 0u && row < mp.n) {
            var total = 0.0;
            for (var q = 0u; q < KSPLIT; q = q + 1u) { total = total + part[q * 64u + lane]; }
            yout[j * mp.n + row] = total;
        }
        workgroupBarrier();
    }
}

// ---------------------------------------------------------------------------------------
// Per-step layer kernels. Activation buffers are fixed at load; per-item data comes from a
// dynamic-offset uniform so one command buffer can carry every dispatch of a step.
// ---------------------------------------------------------------------------------------

// items[i] = (j, pos, seq, 0) for item i of the step; SeqParams selects a contiguous item range.
struct SeqParams { start: u32, count: u32, _a: u32, _b: u32 };

// rmsnorm: out[j] = x[j] * rsqrt(mean(x^2)+eps) * w      (one workgroup of 256 per row)
@group(0) @binding(0) var<storage, read> rn_x: array<f32>;
@group(0) @binding(1) var<storage, read> rn_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> rn_out: array<f32>;
@group(0) @binding(3) var<uniform> lp: LayerParams;
var<workgroup> rn_red: array<f32, 256>;

@compute @workgroup_size(256)
fn rmsnorm(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x;
    let base = j * lp.hidden;
    var ss = 0.0;
    for (var i = lid.x; i < lp.hidden; i = i + 256u) { let v = rn_x[base + i]; ss = ss + v * v; }
    rn_red[lid.x] = ss;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (lid.x < st) { rn_red[lid.x] = rn_red[lid.x] + rn_red[lid.x + st]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(rn_red[0] / f32(lp.hidden) + lp.eps);
    for (var i = lid.x; i < lp.hidden; i = i + 256u) { rn_out[base + i] = rn_x[base + i] * scale * rn_w[i]; }
}

// qk_norm_rope: in-place on the fused qkv buffer (row j = [q: heads*hd | k: kvh*hd | v: kvh*hd]).
// One workgroup per (item, head) over q heads then k heads: per-head rmsnorm then rotary.
@group(0) @binding(0) var<storage, read_write> qr_qkv: array<f32>;
@group(0) @binding(1) var<storage, read> qr_qnorm: array<f32>;
@group(0) @binding(2) var<storage, read> qr_knorm: array<f32>;
@group(0) @binding(3) var<storage, read> qr_invfreq: array<f32>;
@group(0) @binding(4) var<uniform> qr_lp: LayerParams;
@group(0) @binding(5) var<storage, read> qr_items: array<vec4<u32>>;
var<workgroup> qr_red: array<f32, 64>;

@compute @workgroup_size(64)
fn qk_norm_rope(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hd = qr_lp.head_dim;
    let half = hd / 2u;
    let h = wid.x; // 0..heads+kv_heads
    let it = qr_items[wid.y];
    let is_q = h < qr_lp.heads;
    let rowlen = (qr_lp.heads + 2u * qr_lp.kv_heads) * hd;
    let base = it.x * rowlen + h * hd; // q heads then k heads are contiguous
    // rmsnorm over hd (each of 64 threads handles hd/64 elements)
    var ss = 0.0;
    for (var i = lid.x; i < hd; i = i + 64u) { let v = qr_qkv[base + i]; ss = ss + v * v; }
    qr_red[lid.x] = ss;
    workgroupBarrier();
    for (var st = 32u; st > 0u; st = st >> 1u) {
        if (lid.x < st) { qr_red[lid.x] = qr_red[lid.x] + qr_red[lid.x + st]; }
        workgroupBarrier();
    }
    let scale = inverseSqrt(qr_red[0] / f32(hd) + qr_lp.eps);
    // rope on pairs (i, i+half); thread handles pairs i = lid, lid+64, ...
    let pos = f32(it.y);
    for (var i = lid.x; i < half; i = i + 64u) {
        var a = qr_qkv[base + i] * scale;
        var b = qr_qkv[base + i + half] * scale;
        if (is_q) { a = a * qr_qnorm[i]; b = b * qr_qnorm[i + half]; } else { a = a * qr_knorm[i]; b = b * qr_knorm[i + half]; }
        let ang = pos * qr_invfreq[i];
        let c = cos(ang);
        let s = sin(ang);
        qr_qkv[base + i] = a * c - b * s;
        qr_qkv[base + i + half] = a * s + b * c;
    }
}

// attention for one item over its cache (positions 0..=pos): one workgroup per head, 128 threads.
@group(0) @binding(0) var<storage, read> at_qkv: array<f32>;
@group(0) @binding(1) var<storage, read> at_k: array<f32>;   // cache: pos * (kvh*hd)
@group(0) @binding(2) var<storage, read> at_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> at_out: array<f32>; // m * heads*hd
@group(0) @binding(4) var<uniform> at_lp: LayerParams;
@group(0) @binding(5) var<storage, read> at_items: array<vec4<u32>>;
@group(0) @binding(6) var<uniform> at_seq: SeqParams;
const MAXPOS: u32 = 2048u;
var<workgroup> at_scores: array<f32, 2048>;
var<workgroup> at_red: array<f32, 128>;

@compute @workgroup_size(128)
fn attention(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hd = at_lp.head_dim;
    let h = wid.x;
    let kvh = h / (at_lp.heads / at_lp.kv_heads);
    let stride = at_lp.kv_heads * hd;
    let rowlen = (at_lp.heads + 2u * at_lp.kv_heads) * hd;
    let it = at_items[at_seq.start + wid.y];
    let qbase = it.x * rowlen + h * hd;
    let n = it.y + 1u;
    let scale = inverseSqrt(f32(hd));
    // phase 1: scores
    var mx = -1e30;
    for (var t = lid.x; t < n; t = t + 128u) {
        var s = 0.0;
        let kb = t * stride + kvh * hd;
        for (var d = 0u; d < hd; d = d + 1u) { s = s + at_qkv[qbase + d] * at_k[kb + d]; }
        s = s * scale;
        at_scores[t] = s;
        mx = max(mx, s);
    }
    at_red[lid.x] = mx;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st = st >> 1u) {
        if (lid.x < st) { at_red[lid.x] = max(at_red[lid.x], at_red[lid.x + st]); }
        workgroupBarrier();
    }
    let gmax = at_red[0];
    workgroupBarrier();
    var sum = 0.0;
    for (var t = lid.x; t < n; t = t + 128u) { let e = exp(at_scores[t] - gmax); at_scores[t] = e; sum = sum + e; }
    at_red[lid.x] = sum;
    workgroupBarrier();
    for (var st = 64u; st > 0u; st = st >> 1u) {
        if (lid.x < st) { at_red[lid.x] = at_red[lid.x] + at_red[lid.x + st]; }
        workgroupBarrier();
    }
    let inv = 1.0 / at_red[0];
    // phase 2: thread d accumulates output dim d
    if (lid.x < hd) {
        var o = 0.0;
        for (var t = 0u; t < n; t = t + 1u) { o = o + at_scores[t] * at_v[t * stride + kvh * hd + lid.x]; }
        at_out[it.x * at_lp.heads * hd + h * hd + lid.x] = o * inv;
    }
}

// elementwise: x += y   (n = total elements, passed via lp.m * lp.hidden)
@group(0) @binding(0) var<storage, read_write> ad_x: array<f32>;
@group(0) @binding(1) var<storage, read> ad_y: array<f32>;
@group(0) @binding(2) var<uniform> ad_lp: LayerParams;
@compute @workgroup_size(256)
fn add_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < ad_lp.m * ad_lp.hidden) { ad_x[i] = ad_x[i] + ad_y[i]; }
}

// act[j][i] = silu(gu[j][i]) * gu[j][inter + i]   (gu rows are [gate | up])
@group(0) @binding(0) var<storage, read> sm_gu: array<f32>;
@group(0) @binding(1) var<storage, read_write> sm_act: array<f32>;
@group(0) @binding(2) var<uniform> sm_lp: LayerParams;
@compute @workgroup_size(256)
fn silu_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = sm_lp.m * sm_lp.inter;
    if (i < total) {
        let j = i / sm_lp.inter;
        let c = i % sm_lp.inter;
        let g = sm_gu[j * 2u * sm_lp.inter + c];
        let u = sm_gu[j * 2u * sm_lp.inter + sm_lp.inter + c];
        sm_act[i] = g / (1.0 + exp(-g)) * u;
    }
}

// kv_store: copy each item's k and v rows from the fused qkv buffer into its sequence's cache.
// One workgroup per item in the seq range; 256 threads stride over kvh*hd elements.
@group(0) @binding(0) var<storage, read> ks_qkv: array<f32>;
@group(0) @binding(1) var<storage, read_write> ks_k: array<f32>;
@group(0) @binding(2) var<storage, read_write> ks_v: array<f32>;
@group(0) @binding(3) var<uniform> ks_lp: LayerParams;
@group(0) @binding(4) var<storage, read> ks_items: array<vec4<u32>>;
@group(0) @binding(5) var<uniform> ks_seq: SeqParams;
@compute @workgroup_size(256)
fn kv_store(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let hd = ks_lp.head_dim;
    let stride = ks_lp.kv_heads * hd;
    let rowlen = (ks_lp.heads + 2u * ks_lp.kv_heads) * hd;
    let it = ks_items[ks_seq.start + wid.x];
    let src = it.x * rowlen + ks_lp.heads * hd;
    let dst = it.y * stride;
    for (var i = lid.x; i < stride; i = i + 256u) {
        ks_k[dst + i] = ks_qkv[src + i];
        ks_v[dst + i] = ks_qkv[src + stride + i];
    }
}

// ---------------------------------------------------------------------------------------
// Mixture of experts. All experts' weights live in ONE buffer, tiled per expert (rows per
// expert must be a multiple of 64 so each expert's tiles are self-contained); routing runs
// on the GPU and the expert matvec indexes the buffer by the routed expert id — so only the
// chosen experts' weights are ever streamed, which is the entire point of MoE, and nothing
// ever has to come back to the CPU mid-step.
// ---------------------------------------------------------------------------------------

struct MoeParams { ne: u32, topk: u32, inter: u32, norm: u32 };

// router_topk: per token, router logits → softmax over all experts → top-k (expert, weight).
// Mirrors moe_forward in model.rs exactly: weight = prob / (sum of top-k probs if norm else 1).
@group(0) @binding(0) var<storage, read> ro_w: array<f32>;               // ne × hidden
@group(0) @binding(1) var<storage, read> ro_x: array<f32>;               // m × hidden (hn)
@group(0) @binding(2) var<storage, read_write> ro_out: array<vec2<u32>>; // m × topk: (expert, weight bits)
@group(0) @binding(3) var<uniform> ro_lp: LayerParams;
@group(0) @binding(4) var<uniform> ro_mo: MoeParams;
var<workgroup> ro_logit: array<f32, 256>; // ne <= 256, enforced host-side

@compute @workgroup_size(256)
fn router_topk(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let j = wid.x;
    let xb = j * ro_lp.hidden;
    for (var e = lid.x; e < ro_mo.ne; e = e + 256u) {
        var s = 0.0;
        let wb = e * ro_lp.hidden;
        for (var d = 0u; d < ro_lp.hidden; d = d + 1u) { s = s + ro_w[wb + d] * ro_x[xb + d]; }
        ro_logit[e] = s;
    }
    workgroupBarrier();
    // ne is small (<=256): softmax + repeated argmax serially on one thread beats the
    // synchronization a parallel version would need.
    if (lid.x == 0u) {
        var mx = -1e30;
        for (var e = 0u; e < ro_mo.ne; e = e + 1u) { mx = max(mx, ro_logit[e]); }
        var sum = 0.0;
        for (var e = 0u; e < ro_mo.ne; e = e + 1u) { let v = exp(ro_logit[e] - mx); ro_logit[e] = v; sum = sum + v; }
        var wsum = 0.0;
        for (var s = 0u; s < ro_mo.topk; s = s + 1u) {
            var best = 0u;
            var bv = -1.0;
            for (var e = 0u; e < ro_mo.ne; e = e + 1u) { if (ro_logit[e] > bv) { bv = ro_logit[e]; best = e; } }
            let w = bv / sum;
            ro_out[j * ro_mo.topk + s] = vec2<u32>(best, bitcast<u32>(w));
            ro_logit[best] = -1.0;
            wsum = wsum + w;
        }
        if (ro_mo.norm != 0u) {
            for (var s = 0u; s < ro_mo.topk; s = s + 1u) {
                let r = ro_out[j * ro_mo.topk + s];
                ro_out[j * ro_mo.topk + s] = vec2<u32>(r.x, bitcast<u32>(bitcast<f32>(r.y) / wsum));
            }
        }
    }
}

// moe_matvec_q8: matvec_q8 with expert indirection. Workgroup (row tile, slot); slot
// s = token * topk + choice reads its routed expert id and streams only that expert's tile
// range. mp.n = rows per expert; mp._pad selects the input row: 0 = the token's hn row
// (gate/up), 1 = the slot's own row (down reading the slot's activations).
@group(0) @binding(0) var<storage, read> mv2_q: array<u32>;
@group(0) @binding(1) var<storage, read> mv2_s: array<f32>;
@group(0) @binding(2) var<storage, read> mv2_x: array<f32>;
@group(0) @binding(3) var<storage, read_write> mv2_y: array<f32>;
@group(0) @binding(4) var<uniform> mv2_mp: MatParams;
@group(0) @binding(5) var<storage, read> mv2_route: array<vec2<u32>>;
@group(0) @binding(6) var<uniform> mv2_mo: MoeParams;
var<workgroup> mv2_part: array<f32, 256>;

@compute @workgroup_size(256)
fn moe_matvec_q8(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let slot = wid.y;
    let e = mv2_route[slot].x;
    let lane = lid.x % 64u;
    let p = lid.x / 64u;
    let words = mv2_mp.k / 4u;
    let blocks = mv2_mp.k / 32u;
    let tile = e * (mv2_mp.n / 64u) + wid.x;
    let wtile = tile * words * 64u + lane;
    let stile = tile * blocks * 64u + lane;
    let xrow = select(slot, slot / mv2_mo.topk, mv2_mp._pad == 0u);
    let xb = xrow * mv2_mp.k;
    var acc = 0.0;
    for (var b = p; b < blocks; b = b + KSPLIT) {
        var bacc = 0.0;
        let wb = wtile + b * 8u * 64u;
        let kb = xb + b * 32u;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let b0 = mv2_q[wb + i * 64u];
            let w4 = vec4<f32>(f32((i32(b0 << 24u)) >> 24u), f32((i32(b0 << 16u)) >> 24u), f32((i32(b0 << 8u)) >> 24u), f32((i32(b0)) >> 24u));
            let kk = kb + i * 4u;
            bacc = bacc + dot(w4, vec4<f32>(mv2_x[kk], mv2_x[kk + 1u], mv2_x[kk + 2u], mv2_x[kk + 3u]));
        }
        acc = acc + bacc * mv2_s[stile + b * 64u];
    }
    mv2_part[lid.x] = acc;
    workgroupBarrier();
    let row = wid.x * 64u + lane;
    if (p == 0u && row < mv2_mp.n) {
        var total = 0.0;
        for (var q = 0u; q < KSPLIT; q = q + 1u) { total = total + mv2_part[q * 64u + lane]; }
        mv2_y[slot * mv2_mp.n + row] = total;
    }
}

// moe_reduce: o[token] = Σ over the token's slots of (routing weight × that slot's down row).
// Overwrites o (the residual add into x happens in the shared add_inplace pass afterwards).
@group(0) @binding(0) var<storage, read> mr_down: array<f32>;            // (m*topk) × hidden
@group(0) @binding(1) var<storage, read> mr_route: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> mr_o: array<f32>;         // m × hidden
@group(0) @binding(3) var<uniform> mr_lp: LayerParams;
@group(0) @binding(4) var<uniform> mr_mo: MoeParams;
@compute @workgroup_size(256)
fn moe_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i < mr_lp.m * mr_lp.hidden) {
        let j = i / mr_lp.hidden;
        let d = i % mr_lp.hidden;
        var acc = 0.0;
        for (var s = 0u; s < mr_mo.topk; s = s + 1u) {
            let r = mr_route[j * mr_mo.topk + s];
            acc = acc + bitcast<f32>(r.y) * mr_down[(j * mr_mo.topk + s) * mr_lp.hidden + d];
        }
        mr_o[i] = acc;
    }
}
