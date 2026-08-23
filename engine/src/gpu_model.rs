//! Qwen3 dense forward pass on the GPU. Mirrors model.rs's forward_impl: same items/caches
//! contract, so the scheduler treats CPU and GPU identically.

use crate::gpu::{Gpu, GQ8};
use crate::kernels::{bf16_to_f32, QMat};
use crate::model::{Config, Mlp, Model, Weight};
use std::sync::Arc;
use wgpu::util::DeviceExt;

const MAXM: usize = 512; // items per step (prefill is chunked to this)
pub const GPU_MAX_CONTEXT: usize = 2048; // attention shader's shared-memory limit

#[repr(C)]
#[derive(Clone, Copy)]
struct LayerParams {
    m: u32,
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    inter: u32,
    eps: f32,
    pad: u32,
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

struct GLayer {
    ln1: wgpu::Buffer,
    ln2: wgpu::Buffer,
    q_norm: wgpu::Buffer,
    k_norm: wgpu::Buffer,
    wqkv: GQ8,
    wo: GQ8,
    wgu: GQ8,
    wdown: GQ8,
    // static bind groups
    bg_norm1: wgpu::BindGroup,
    bg_norm2: wgpu::BindGroup,
    bg_qkv: wgpu::BindGroup,
    bg_rope: wgpu::BindGroup,
    bg_o: wgpu::BindGroup,
    bg_gu: wgpu::BindGroup,
    bg_down: wgpu::BindGroup,
}

struct Pipelines {
    matvec: wgpu::ComputePipeline,
    matvec_l: wgpu::BindGroupLayout,
    rmsnorm: wgpu::ComputePipeline,
    rmsnorm_l: wgpu::BindGroupLayout,
    rope: wgpu::ComputePipeline,
    rope_l: wgpu::BindGroupLayout,
    attention: wgpu::ComputePipeline,
    attention_l: wgpu::BindGroupLayout,
    add: wgpu::ComputePipeline,
    add_l: wgpu::BindGroupLayout,
    silu: wgpu::ComputePipeline,
    silu_l: wgpu::BindGroupLayout,
    kv_store: wgpu::ComputePipeline,
    kv_store_l: wgpu::BindGroupLayout,
}

pub struct GpuModel {
    pub config: Config,
    gpu: Arc<Gpu>,
    p: Pipelines,
    embed: Vec<u16>,
    layers: Vec<GLayer>,
    norm: wgpu::Buffer,
    lm_head: GQ8,
    inv_freq: wgpu::Buffer,
    // activations (MAXM rows)
    x: wgpu::Buffer,
    hn: wgpu::Buffer,
    qkv: wgpu::Buffer,
    attn: wgpu::Buffer,
    o: wgpu::Buffer,
    gu: wgpu::Buffer,
    act: wgpu::Buffer,
    xl: wgpu::Buffer,
    xln: wgpu::Buffer,
    logits: wgpu::Buffer,
    lp: wgpu::Buffer,       // LayerParams for this step (m = items)
    lp_last: wgpu::Buffer,  // LayerParams with m = number of logits rows
    items: wgpu::Buffer,    // storage: vec4<u32> per item (j, pos, seq, 0)
    seqp: wgpu::Buffer,     // dynamic-offset uniform: SeqParams per sequence slot (256 B apart)
    bg_add_o: wgpu::BindGroup,
    bg_silu: wgpu::BindGroup,
    bg_norm_last: wgpu::BindGroup,
    bg_head: wgpu::BindGroup,
    qd: usize,
    kd: usize,
}

/// Per-sequence KV cache on the GPU, plus the per-layer bind groups that reference it.
pub struct GpuKv {
    k: Vec<wgpu::Buffer>,
    v: Vec<wgpu::Buffer>,
    bg_attn: Vec<wgpu::BindGroup>,
    bg_store: Vec<wgpu::BindGroup>,
    pub len: usize,
    bytes: usize,
}

impl GpuKv {
    pub fn truncate(&mut self, n: usize) {
        self.len = n;
    }
    pub fn bytes(&self) -> usize {
        self.bytes
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None }, count: None }
}
fn uniform_entry(binding: u32, dynamic: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: None }, count: None }
}

fn bg(dev: &wgpu::Device, layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer]) -> wgpu::BindGroup {
    let entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

fn bg_dyn(dev: &wgpu::Device, layout: &wgpu::BindGroupLayout, bufs: &[&wgpu::Buffer], dyn_buf: &wgpu::Buffer) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
    entries.push(wgpu::BindGroupEntry { binding: bufs.len() as u32, resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: dyn_buf, offset: 0, size: wgpu::BufferSize::new(16) }) });
    dev.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
}

fn q8(w: &Weight) -> &QMat {
    match w {
        Weight::Q8(q) => q,
        _ => panic!("GPU backend needs QUANT=q8"),
    }
}

/// Concatenate Q8 matrices along rows (same k).
fn concat_q8(parts: &[&QMat]) -> QMat {
    let k = parts[0].k;
    let mut q = Vec::new();
    let mut scales = Vec::new();
    let mut n = 0;
    for p in parts {
        assert_eq!(p.k, k);
        q.extend_from_slice(&p.q);
        scales.extend_from_slice(&p.scales);
        n += p.n;
    }
    QMat { n, k, q, scales }
}

impl GpuModel {
    pub fn from_cpu(gpu: Arc<Gpu>, m: &Model) -> GpuModel {
        let t0 = std::time::Instant::now();
        let dev = &gpu.device;
        let c = m.config.clone();
        let (h, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
        let (qd, kd) = (h * hd, kvh * hd);
        let shader = dev.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some("forge"), source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()) });
        let mk = |name: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            let l = dev.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(name), entries });
            let pl = dev.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&l], push_constant_ranges: &[] });
            let p = dev.create_compute_pipeline(&wgpu::ComputePipelineDescriptor { label: Some(name), layout: Some(&pl), module: &shader, entry_point: Some(name), compilation_options: Default::default(), cache: None });
            (p, l)
        };
        let (matvec, matvec_l) = mk("matvec_q8", &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false), uniform_entry(4, false), uniform_entry(5, false)]);
        let (rmsnorm, rmsnorm_l) = mk("rmsnorm", &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, false), uniform_entry(3, false)]);
        let (rope, rope_l) = mk("qk_norm_rope", &[storage_entry(0, false), storage_entry(1, true), storage_entry(2, true), storage_entry(3, true), uniform_entry(4, false), storage_entry(5, true)]);
        let (attention, attention_l) = mk("attention", &[storage_entry(0, true), storage_entry(1, true), storage_entry(2, true), storage_entry(3, false), uniform_entry(4, false), storage_entry(5, true), uniform_entry(6, true)]);
        let (add, add_l) = mk("add_inplace", &[storage_entry(0, false), storage_entry(1, true), uniform_entry(2, false)]);
        let (silu, silu_l) = mk("silu_mul", &[storage_entry(0, true), storage_entry(1, false), uniform_entry(2, false)]);
        let (kv_store, kv_store_l) = mk("kv_store", &[storage_entry(0, true), storage_entry(1, false), storage_entry(2, false), uniform_entry(3, false), storage_entry(4, true), uniform_entry(5, true)]);
        let p = Pipelines { matvec, matvec_l, rmsnorm, rmsnorm_l, rope, rope_l, attention, attention_l, add, add_l, silu, silu_l, kv_store, kv_store_l };

        let f32buf = |v: &[f32], label: &str| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents: as_bytes(v), usage: wgpu::BufferUsages::STORAGE });
        let st = |n: usize, label: &str| gpu.storage(n * 4, label);
        let x = st(MAXM * c.hidden, "x");
        let hn = st(MAXM * c.hidden, "hn");
        let qkv = st(MAXM * (qd + 2 * kd), "qkv");
        let attn = st(MAXM * qd, "attn");
        let o = st(MAXM * c.hidden, "o");
        let gu = st(MAXM * 2 * c.intermediate, "gu");
        let act = st(MAXM * c.intermediate, "act");
        let xl = st(MAXM * c.hidden, "xl");
        let xln = st(MAXM * c.hidden, "xln");
        let logits = st(16 * c.vocab, "logits");
        let lpv = LayerParams { m: 1, hidden: c.hidden as u32, heads: h as u32, kv_heads: kvh as u32, head_dim: hd as u32, inter: c.intermediate as u32, eps: c.eps, pad: 0 };
        let lp = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("lp"), contents: as_bytes(&[lpv]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let lp_last = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("lp_last"), contents: as_bytes(&[lpv]), usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST });
        let items = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("items"), size: (MAXM * 16) as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let seqp = dev.create_buffer(&wgpu::BufferDescriptor { label: Some("seqp"), size: (64 * 256) as u64, usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let inv_freq = f32buf(&m.inv_freq, "inv_freq");

        let mut layers = Vec::with_capacity(c.layers);
        for l in &m.layers {
            let (wg, wu, wd) = match &l.mlp {
                Mlp::Dense(e) => (q8(&e.w_gate), q8(&e.w_up), q8(&e.w_down)),
                Mlp::Moe { .. } => panic!("GPU backend: MoE not implemented yet"),
            };
            let (lwq, lwk, lwv, lwo, lqn, lkn) = match &l.attn {
                crate::model::Attn::Full { wq, wk, wv, wo, q_norm, k_norm, gate: false } => (wq, wk, wv, wo, q_norm, k_norm),
                _ => panic!("GPU backend v1 supports plain softmax attention only (use DEVICE=cpu)"),
            };
            let qkv_m = concat_q8(&[q8(lwq), q8(lwk), q8(lwv)]);
            let gu_m = concat_q8(&[wg, wu]);
            let wqkv = gpu.upload_q8(&qkv_m.q, &qkv_m.scales, qkv_m.n, qkv_m.k);
            let wo = gpu.upload_q8(&q8(lwo).q, &q8(lwo).scales, q8(lwo).n, q8(lwo).k);
            let wgu = gpu.upload_q8(&gu_m.q, &gu_m.scales, gu_m.n, gu_m.k);
            let wdown = gpu.upload_q8(&wd.q, &wd.scales, wd.n, wd.k);
            let ln1 = f32buf(&l.ln1, "ln1");
            let ln2 = f32buf(&l.ln2, "ln2");
            let q_norm = f32buf(lqn, "q_norm");
            let k_norm = f32buf(lkn, "k_norm");
            let mp = |w: &GQ8| dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("mp"), contents: as_bytes(&[w.n as u32, w.k as u32, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
            let (mp_qkv, mp_o, mp_gu, mp_down) = (mp(&wqkv), mp(&wo), mp(&wgu), mp(&wdown));
            layers.push(GLayer {
                bg_norm1: bg(dev, &p.rmsnorm_l, &[&x, &ln1, &hn, &lp]),
                bg_norm2: bg(dev, &p.rmsnorm_l, &[&x, &ln2, &hn, &lp]),
                bg_qkv: bg(dev, &p.matvec_l, &[&wqkv.q, &wqkv.scales, &hn, &qkv, &mp_qkv, &lp]),
                bg_rope: bg(dev, &p.rope_l, &[&qkv, &q_norm, &k_norm, &inv_freq, &lp, &items]),
                bg_o: bg(dev, &p.matvec_l, &[&wo.q, &wo.scales, &attn, &o, &mp_o, &lp]),
                bg_gu: bg(dev, &p.matvec_l, &[&wgu.q, &wgu.scales, &hn, &gu, &mp_gu, &lp]),
                bg_down: bg(dev, &p.matvec_l, &[&wdown.q, &wdown.scales, &act, &o, &mp_down, &lp]),
                ln1, ln2, q_norm, k_norm, wqkv, wo, wgu, wdown,
            });
        }
        let head = q8(&m.lm_head);
        let lm_head = gpu.upload_q8(&head.q, &head.scales, head.n, head.k);
        let mp_head = dev.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("mp"), contents: as_bytes(&[lm_head.n as u32, lm_head.k as u32, 0u32, 0u32]), usage: wgpu::BufferUsages::UNIFORM });
        let norm = f32buf(&m.norm, "norm");
        let bg_add_o = bg(dev, &p.add_l, &[&x, &o, &lp]);
        let bg_silu = bg(dev, &p.silu_l, &[&gu, &act, &lp]);
        let bg_norm_last = bg(dev, &p.rmsnorm_l, &[&xl, &norm, &xln, &lp_last]);
        let bg_head = bg(dev, &p.matvec_l, &[&lm_head.q, &lm_head.scales, &xln, &logits, &mp_head, &lp_last]);
        let _ = &layers[0].ln1; // keep buffers alive via struct
        eprintln!("gpu model ready on {} in {:.1}s ({} layers, fused qkv/gate-up)", gpu.name, t0.elapsed().as_secs_f32(), layers.len());
        GpuModel { config: c, gpu, p, embed: m.embed.clone(), layers, norm, lm_head, inv_freq, x, hn, qkv, attn, o, gu, act, xl, xln, logits, lp, lp_last, items, seqp, bg_add_o, bg_silu, bg_norm_last, bg_head, qd, kd }
    }

    pub fn new_cache(&self) -> GpuKv {
        let c = &self.config;
        let dev = &self.gpu.device;
        let stride = self.kd;
        let per_layer = GPU_MAX_CONTEXT.min(c.max_context) * stride * 4;
        let mut k = Vec::new();
        let mut v = Vec::new();
        let mut bg_attn = Vec::new();
        let mut bg_store = Vec::new();
        for l in 0..c.layers {
            let kb = self.gpu.storage(per_layer, "kcache");
            let vb = self.gpu.storage(per_layer, "vcache");
            bg_attn.push(bg_dyn(dev, &self.p.attention_l, &[&self.qkv, &kb, &vb, &self.attn, &self.lp, &self.items], &self.seqp));
            bg_store.push(bg_dyn(dev, &self.p.kv_store_l, &[&self.qkv, &kb, &vb, &self.lp, &self.items], &self.seqp));
            let _ = l;
            k.push(kb);
            v.push(vb);
        }
        GpuKv { k, v, bg_attn, bg_store, len: 0, bytes: per_layer * 2 * c.layers }
    }

    pub fn clone_cache(&self, src: &GpuKv) -> GpuKv {
        let mut dst = self.new_cache();
        let bytes = (src.len * self.kd * 4) as u64;
        if bytes > 0 {
            let mut enc = self.gpu.device.create_command_encoder(&Default::default());
            for l in 0..self.config.layers {
                enc.copy_buffer_to_buffer(&src.k[l], 0, &dst.k[l], 0, bytes);
                enc.copy_buffer_to_buffer(&src.v[l], 0, &dst.v[l], 0, bytes);
            }
            self.gpu.queue.submit([enc.finish()]);
        }
        dst.len = src.len;
        dst
    }

    pub fn forward_batch(&self, tokens: &[u32], cache: &mut GpuKv) -> Vec<f32> {
        let mut out = Vec::new();
        for chunk in tokens.chunks(MAXM) {
            let items: Vec<(u32, usize)> = chunk.iter().map(|&t| (t, 0)).collect();
            out = self.forward_impl(&items, &mut [cache], false).pop().unwrap();
        }
        out
    }

    pub fn forward_multi(&self, items: &[(u32, usize)], caches: &mut [&mut GpuKv]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, false)
    }

    pub fn forward_multi_all(&self, items: &[(u32, usize)], caches: &mut [&mut GpuKv]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, true)
    }

    fn forward_impl(&self, items: &[(u32, usize)], caches: &mut [&mut GpuKv], all_logits: bool) -> Vec<Vec<f32>> {
        let c = &self.config;
        let m = items.len();
        assert!(m <= MAXM, "too many items for one GPU step");
        let gpu = &self.gpu;
        let (h, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);

        // positions, sequence ranges, last-of-sequence rows
        let mut pos = vec![0u32; m];
        let mut seen = vec![0usize; caches.len()];
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (seq, start, count)
        for (j, &(_, sq)) in items.iter().enumerate() {
            pos[j] = (caches[sq].len + seen[sq]) as u32;
            assert!((pos[j] as usize) < GPU_MAX_CONTEXT, "GPU context limit {GPU_MAX_CONTEXT}");
            seen[sq] += 1;
            match ranges.last_mut() {
                Some(r) if r.0 == sq => r.2 += 1,
                _ => ranges.push((sq, j, 1)),
            }
        }
        let mut is_last = vec![false; m];
        let mut seen_last = vec![false; caches.len()];
        for j in (0..m).rev() {
            let sq = items[j].1;
            if !seen_last[sq] {
                seen_last[sq] = true;
                is_last[j] = true;
            }
        }
        let lasts: Vec<usize> = (0..m).filter(|&j| all_logits || is_last[j]).collect();
        assert!(lasts.len() <= 16, "logits buffer holds 16 rows");

        // upload embeddings + per-step params
        let mut x = vec![0f32; m * c.hidden];
        for (j, &(t, _)) in items.iter().enumerate() {
            for (d, &b) in self.embed[t as usize * c.hidden..(t as usize + 1) * c.hidden].iter().enumerate() {
                x[j * c.hidden + d] = bf16_to_f32(b);
            }
        }
        gpu.write_f32(&self.x, &x);
        let lpv = LayerParams { m: m as u32, hidden: c.hidden as u32, heads: h as u32, kv_heads: kvh as u32, head_dim: hd as u32, inter: c.intermediate as u32, eps: c.eps, pad: 0 };
        gpu.queue.write_buffer(&self.lp, 0, as_bytes(&[lpv]));
        gpu.queue.write_buffer(&self.lp_last, 0, as_bytes(&[LayerParams { m: lasts.len() as u32, ..lpv }]));
        let itemv: Vec<[u32; 4]> = (0..m).map(|j| [j as u32, pos[j], items[j].1 as u32, 0]).collect();
        gpu.queue.write_buffer(&self.items, 0, as_bytes(&itemv));
        let mut seqv = vec![0u32; ranges.len() * 64];
        for (i, r) in ranges.iter().enumerate() {
            seqv[i * 64] = r.1 as u32;
            seqv[i * 64 + 1] = r.2 as u32;
        }
        gpu.queue.write_buffer(&self.seqp, 0, as_bytes(&seqv));

        let mut enc = gpu.device.create_command_encoder(&Default::default());
        let wg = |n: usize| ((n + 255) / 256) as u32;
        for l in &self.layers {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.p.rmsnorm);
            pass.set_bind_group(0, &l.bg_norm1, &[]);
            pass.dispatch_workgroups(m as u32, 1, 1);
            pass.set_pipeline(&self.p.matvec);
            pass.set_bind_group(0, &l.bg_qkv, &[]);
            pass.dispatch_workgroups(((l.wqkv.n + 63) / 64) as u32, 1, 1);
            pass.set_pipeline(&self.p.rope);
            pass.set_bind_group(0, &l.bg_rope, &[]);
            pass.dispatch_workgroups((h + kvh) as u32, m as u32, 1);
            // per sequence: store k/v, then attention over that sequence's cache
            let li = self.layers.iter().position(|x| std::ptr::eq(x, l)).unwrap();
            for (i, r) in ranges.iter().enumerate() {
                let off = (i * 256) as u32;
                let cache = &*caches[r.0];
                pass.set_pipeline(&self.p.kv_store);
                pass.set_bind_group(0, &cache.bg_store[li], &[off]);
                pass.dispatch_workgroups(r.2 as u32, 1, 1);
                pass.set_pipeline(&self.p.attention);
                pass.set_bind_group(0, &cache.bg_attn[li], &[off]);
                pass.dispatch_workgroups(h as u32, r.2 as u32, 1);
            }
            pass.set_pipeline(&self.p.matvec);
            pass.set_bind_group(0, &l.bg_o, &[]);
            pass.dispatch_workgroups(((l.wo.n + 63) / 64) as u32, 1, 1);
            pass.set_pipeline(&self.p.add);
            pass.set_bind_group(0, &self.bg_add_o, &[]);
            pass.dispatch_workgroups(wg(m * c.hidden), 1, 1);
            pass.set_pipeline(&self.p.rmsnorm);
            pass.set_bind_group(0, &l.bg_norm2, &[]);
            pass.dispatch_workgroups(m as u32, 1, 1);
            pass.set_pipeline(&self.p.matvec);
            pass.set_bind_group(0, &l.bg_gu, &[]);
            pass.dispatch_workgroups(((l.wgu.n + 63) / 64) as u32, 1, 1);
            pass.set_pipeline(&self.p.silu);
            pass.set_bind_group(0, &self.bg_silu, &[]);
            pass.dispatch_workgroups(wg(m * c.intermediate), 1, 1);
            pass.set_pipeline(&self.p.matvec);
            pass.set_bind_group(0, &l.bg_down, &[]);
            pass.dispatch_workgroups(((l.wdown.n + 63) / 64) as u32, 1, 1);
            pass.set_pipeline(&self.p.add);
            pass.set_bind_group(0, &self.bg_add_o, &[]);
            pass.dispatch_workgroups(wg(m * c.hidden), 1, 1);
        }
        // gather last rows → xl, final norm, lm_head
        for (r, &j) in lasts.iter().enumerate() {
            enc.copy_buffer_to_buffer(&self.x, (j * c.hidden * 4) as u64, &self.xl, (r * c.hidden * 4) as u64, (c.hidden * 4) as u64);
        }
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.p.rmsnorm);
            pass.set_bind_group(0, &self.bg_norm_last, &[]);
            pass.dispatch_workgroups(lasts.len() as u32, 1, 1);
            pass.set_pipeline(&self.p.matvec);
            pass.set_bind_group(0, &self.bg_head, &[]);
            pass.dispatch_workgroups(((self.lm_head.n + 63) / 64) as u32, 1, 1);
        }
        gpu.queue.submit([enc.finish()]);
        let logits = gpu.read_f32(&self.logits, lasts.len() * c.vocab);
        for (sq, n) in seen.iter().enumerate() {
            caches[sq].len += n;
        }
        logits.chunks(c.vocab).map(|l| l.to_vec()).collect()
    }
}
