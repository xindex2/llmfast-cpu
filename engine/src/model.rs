//! Qwen3 dense decoder: config, weights, and the per-token forward pass.
//!
//! Per layer:  x ──RMSNorm──► Q,K,V proj ──per-head RMSNorm (Qwen3-specific)──► RoPE
//!              ──GQA attention over KV cache──► O proj ──► + x
//!             x ──RMSNorm──► gate,up proj ──SiLU(gate)*up──► down proj ──► + x
//! Then final RMSNorm and logits = embedding matrix · x (weights are tied in the 0.6B model).

use crate::kernels::*;
use crate::safetensors::SafeTensors;

/// A linear layer's weights in whichever format we're running.
pub enum Weight {
    Bf16 { w: Vec<u16>, n: usize, k: usize },
    Q8(QMat),
    Q4(Q4Mat),
}

impl Weight {
    fn load(st: &SafeTensors, name: &str, quant: &str) -> Weight {
        let info = st.info(name);
        let (n, k) = (info.shape[0], info.shape[1]);
        let w = st.bf16(name);
        match quant {
            "q8" => Weight::Q8(QMat::from_bf16(&w, n, k)),
            "q4" => Weight::Q4(Q4Mat::from_bf16(&w, n, k)),
            _ => Weight::Bf16 { w, n, k },
        }
    }

    #[inline]
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        match self {
            Weight::Bf16 { w, n, k } => matvec_bf16(w, x, *n, *k, y),
            Weight::Q8(q) => matvec_q8(q, x, y),
            Weight::Q4(q) => matvec_q4(q, x, y),
        }
    }

    #[inline]
    pub fn matmul(&self, xs: &[f32], m: usize, ys: &mut [f32]) {
        match self {
            Weight::Bf16 { w, n, k } => matmul_bf16(w, xs, m, *n, *k, ys),
            Weight::Q8(q) => matmul_q8(q, xs, m, ys),
            Weight::Q4(q) => matmul_q4(q, xs, m, ys),
        }
    }

    pub fn bytes(&self) -> usize {
        match self {
            Weight::Bf16 { w, .. } => w.len() * 2,
            Weight::Q8(q) => q.bytes(),
            Weight::Q4(q) => q.bytes(),
        }
    }

    pub fn params(&self) -> usize {
        match self {
            Weight::Bf16 { n, k, .. } => n * k,
            Weight::Q8(q) => q.n * q.k,
            Weight::Q4(q) => q.n * q.k,
        }
    }
}

/// Gated-DeltaNet linear attention dimensions (Qwen3.5 hybrid layers).
#[derive(Debug, Clone)]
pub struct LinCfg {
    pub nk: usize,     // key heads
    pub nv: usize,     // value heads (nk divides nv)
    pub dk: usize,     // key head dim
    pub dv: usize,     // value head dim
    pub conv_k: usize, // causal conv kernel
}

impl LinCfg {
    pub fn key_dim(&self) -> usize { self.nk * self.dk }
    pub fn value_dim(&self) -> usize { self.nv * self.dv }
    pub fn conv_dim(&self) -> usize { 2 * self.key_dim() + self.value_dim() }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub rope_theta: f32,
    pub eps: f32,
    pub max_context: usize,
    // Mixture of experts (0 experts = dense MLP)
    pub num_experts: usize,
    pub experts_per_tok: usize,
    pub moe_intermediate: usize,
    pub norm_topk_prob: bool,
    // Qwen3.5 hybrid extras (defaults reproduce Qwen3 behavior)
    pub rotary_dim: usize,        // dims of each head that get RoPE (partial rotary)
    pub attn_gate: bool,          // q_proj also produces a sigmoid output gate per head
    pub layer_types: Vec<bool>,   // per layer: true = full attention; empty = all full
    pub lin: Option<LinCfg>,
    pub prefix: String,           // tensor name prefix: "model." or "model.language_model."
    /// Qwen3.5 stores RMSNorm weights zero-centered and applies (1 + w); Qwen3 applies w.
    /// (The DeltaNet gated norm always uses a plain weight.)
    pub norm_offset: f32,
}

impl Config {
    pub fn is_full(&self, layer: usize) -> bool {
        self.layer_types.get(layer).copied().unwrap_or(true)
    }
}

impl Config {
    pub fn load(dir: &str) -> Config {
        let root: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{dir}/config.json")).expect("config.json")).unwrap();
        // Multimodal checkpoints (Qwen3.5) nest the language model under text_config.
        let (j, prefix) = if root["text_config"].is_object() {
            (root["text_config"].clone(), "model.language_model.".to_string())
        } else {
            (root.clone(), "model.".to_string())
        };
        let u = |k: &str| j[k].as_u64().unwrap_or_else(|| panic!("config missing {k}")) as usize;
        let heads = u("num_attention_heads");
        let head_dim = j["head_dim"].as_u64().map(|v| v as usize).unwrap_or(u("hidden_size") / heads);
        let rope = &j["rope_parameters"];
        let rope_theta = rope["rope_theta"].as_f64().or(j["rope_theta"].as_f64()).unwrap_or(10000.0) as f32;
        let partial = rope["partial_rotary_factor"].as_f64().or(j["partial_rotary_factor"].as_f64()).unwrap_or(1.0);
        let layer_types: Vec<bool> = j["layer_types"].as_array().map(|a| a.iter().map(|t| t.as_str() == Some("full_attention")).collect()).unwrap_or_default();
        let lin = if j["linear_num_value_heads"].is_u64() {
            Some(LinCfg { nk: u("linear_num_key_heads"), nv: u("linear_num_value_heads"), dk: u("linear_key_head_dim"), dv: u("linear_value_head_dim"), conv_k: u("linear_conv_kernel_dim") })
        } else {
            None
        };
        let model_type = j["model_type"].as_str().unwrap_or("");
        Config {
            norm_offset: if model_type.starts_with("qwen3_5") { 1.0 } else { 0.0 },
            rotary_dim: ((head_dim as f64 * partial) as usize).max(2) & !1,
            attn_gate: j["attn_output_gate"].as_bool().unwrap_or(false),
            layer_types,
            lin,
            prefix,
            hidden: u("hidden_size"),
            intermediate: u("intermediate_size"),
            layers: std::env::var("NUM_LAYERS").ok().and_then(|v| v.parse().ok()).map(|n: usize| n.min(u("num_hidden_layers"))).unwrap_or_else(|| u("num_hidden_layers")),
            heads,
            kv_heads: u("num_key_value_heads"),
            head_dim,
            vocab: u("vocab_size"),
            rope_theta,
            eps: j["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
            max_context: std::env::var("MAX_CONTEXT").ok().and_then(|v| v.parse().ok()).unwrap_or(4096),
            num_experts: j["num_experts"].as_u64().unwrap_or(0) as usize,
            experts_per_tok: j["num_experts_per_tok"].as_u64().unwrap_or(0) as usize,
            moe_intermediate: j["moe_intermediate_size"].as_u64().unwrap_or(0) as usize,
            norm_topk_prob: j["norm_topk_prob"].as_bool().unwrap_or(true),
        }
    }
}

pub(crate) struct Layer {
    pub(crate) ln1: Vec<f32>,
    pub(crate) attn: Attn,
    pub(crate) ln2: Vec<f32>,
    pub(crate) mlp: Mlp,
}

pub(crate) enum Attn {
    /// Softmax attention (Qwen3, and Qwen3.5 full_attention layers). With `gate`, wq has 2x rows:
    /// per head [q(head_dim) | gate(head_dim)], and the head output is multiplied by sigmoid(gate).
    Full { wq: Weight, wk: Weight, wv: Weight, wo: Weight, q_norm: Vec<f32>, k_norm: Vec<f32>, gate: bool },
    /// Gated DeltaNet linear attention (Qwen3.5). Recurrent state instead of a KV cache.
    Lin {
        w_qkv: Weight,        // rows: [q: nk*dk | k: nk*dk | v: nv*dv]
        w_z: Weight,          // value_dim gate
        w_b: Weight,          // nv rows → beta
        w_a: Weight,          // nv rows → decay input
        conv_w: Vec<f32>,     // depthwise causal conv [conv_dim * conv_k], w[c*k + j], j = k-1 is current token
        dt_bias: Vec<f32>,    // nv
        a_log: Vec<f32>,      // nv
        norm_w: Vec<f32>,     // dv (gated rmsnorm weight)
        w_out: Weight,        // value_dim → hidden
    },
}

pub struct Expert {
    pub(crate) w_gate: Weight,
    pub(crate) w_up: Weight,
    pub(crate) w_down: Weight,
}

pub enum Mlp {
    Dense(Expert),
    Moe { router: Vec<f32>, experts: Vec<Expert> }, // router: [num_experts, hidden] f32
}

pub struct Model {
    pub config: Config,
    pub(crate) inv_freq: Vec<f32>, // head_dim/2 rotary frequencies
    pub(crate) embed: Vec<u16>, // [vocab, hidden]
    pub(crate) layers: Vec<Layer>,
    pub(crate) norm: Vec<f32>,
    pub(crate) lm_head: Weight, // tied to embed in small models; stored separately so it can be quantized
}

/// Per-layer sequence state: KV rows for softmax attention, recurrent state for linear attention.
#[derive(Clone)]
pub(crate) enum LayerCache {
    Kv { k: Vec<f32>, v: Vec<f32> },                 // pos * stride, growable
    Lin { state: Vec<f32>, conv: Vec<f32> },         // nv*dk*dv, conv_dim*(conv_k-1); empty until first use
}

#[derive(Clone)]
pub struct KvCache {
    pub(crate) layers: Vec<LayerCache>,
    pub(crate) stride: usize,
    pub len: usize,
    hybrid: bool,
}

impl KvCache {
    pub fn new(cfg: &Config) -> KvCache {
        let stride = cfg.kv_heads * cfg.head_dim;
        let layers = (0..cfg.layers).map(|l| if cfg.is_full(l) {
            LayerCache::Kv { k: Vec::new(), v: Vec::new() }
        } else {
            LayerCache::Lin { state: Vec::new(), conv: Vec::new() }
        }).collect();
        let hybrid = cfg.lin.is_some();
        KvCache { layers, stride, len: 0, hybrid }
    }

    pub fn bytes(&self) -> usize {
        self.layers.iter().map(|l| match l {
            LayerCache::Kv { k, .. } => k.capacity() * 8,
            LayerCache::Lin { state, conv } => (state.capacity() + conv.capacity()) * 4,
        }).sum()
    }

    /// Recurrent state cannot be rolled back: hybrid caches only "truncate" to their current length.
    pub fn can_truncate(&self, n: usize) -> bool {
        !self.hybrid || n == self.len
    }

    /// Drop everything after the first `n` positions (see can_truncate).
    pub fn truncate(&mut self, n: usize) {
        assert!(self.can_truncate(n), "cannot roll back recurrent state ({} -> {n})", self.len);
        for l in &mut self.layers {
            if let LayerCache::Kv { k, v } = l {
                k.truncate(n * self.stride);
                v.truncate(n * self.stride);
            }
        }
        self.len = n;
    }
}

impl Model {
    /// Bytes of weights streamed per token (what has to live on the device).
    pub fn weight_bytes(&self) -> usize {
        let mut b = self.lm_head.bytes();
        for l in &self.layers {
            b += match &l.attn {
                Attn::Full { wq, wk, wv, wo, .. } => wq.bytes() + wk.bytes() + wv.bytes() + wo.bytes(),
                Attn::Lin { w_qkv, w_z, w_b, w_a, w_out, .. } => w_qkv.bytes() + w_z.bytes() + w_b.bytes() + w_a.bytes() + w_out.bytes(),
            };
            if let Mlp::Dense(e) = &l.mlp {
                b += e.w_gate.bytes() + e.w_up.bytes() + e.w_down.bytes();
            }
        }
        b
    }

    /// True when the model needs no recurrent-state rollback (speculation-safe).
    pub fn supports_rollback(&self) -> bool {
        self.config.lin.is_none()
    }

    /// v1 GPU backend handles plain Qwen3 dense: softmax attention without gates, full rotary.
    pub fn gpu_supported(&self) -> bool {
        self.layers_mlp_dense()
            && self.config.lin.is_none()
            && !self.config.attn_gate
            && self.config.rotary_dim == self.config.head_dim
    }

    /// True when every MLP is dense (the GPU backend doesn't do MoE yet).
    pub fn layers_mlp_dense(&self) -> bool {
        self.layers.iter().all(|l| matches!(l.mlp, Mlp::Dense(_)))
    }

    pub fn load(dir: &str) -> Model {
        Model::load_with(dir, &std::env::var("QUANT").unwrap_or_else(|_| "q8".into()))
    }

    pub fn load_with(dir: &str, quant: &str) -> Model {
        let t0 = std::time::Instant::now();
        let config = Config::load(dir);
        let st = SafeTensors::open_dir(dir);
        let quant = quant.to_string();
        // qwen3_5 stores RMSNorm weights zero-centered and applies (1 + w); qwen3 applies w.
        let norm_w = |name: &str| {
            let mut v = st.f32(name);
            if config.norm_offset != 0.0 {
                for x in &mut v {
                    *x += config.norm_offset;
                }
            }
            v
        };
        let mut layers = Vec::with_capacity(config.layers);
        for l in 0..config.layers {
            if l % 8 == 0 {
                eprintln!("loading layer {l}/{} ({quant})...", config.layers);
            }
            let p = format!("{}layers.{l}.", config.prefix);
            let attn = if config.is_full(l) {
                Attn::Full {
                    wq: Weight::load(&st, &format!("{p}self_attn.q_proj.weight"), &quant),
                    wk: Weight::load(&st, &format!("{p}self_attn.k_proj.weight"), &quant),
                    wv: Weight::load(&st, &format!("{p}self_attn.v_proj.weight"), &quant),
                    wo: Weight::load(&st, &format!("{p}self_attn.o_proj.weight"), &quant),
                    q_norm: norm_w(&format!("{p}self_attn.q_norm.weight")),
                    k_norm: norm_w(&format!("{p}self_attn.k_norm.weight")),
                    gate: config.attn_gate,
                }
            } else {
                // The delta-rule recurrence compounds quantization noise across tokens, so the
                // linear-attention projections stay at least q8 even when the rest runs q4.
                let lq_env = std::env::var("LIN_QUANT").ok();
                let lq: &str = lq_env.as_deref().unwrap_or(if quant == "q4" { "q8" } else { quant.as_str() });
                let _ = &lq_env;
                Attn::Lin {
                    w_qkv: Weight::load(&st, &format!("{p}linear_attn.in_proj_qkv.weight"), lq),
                    w_z: Weight::load(&st, &format!("{p}linear_attn.in_proj_z.weight"), lq),
                    // beta/decay projections are tiny and numerically sensitive: keep full precision
                    w_b: Weight::load(&st, &format!("{p}linear_attn.in_proj_b.weight"), "bf16"),
                    w_a: Weight::load(&st, &format!("{p}linear_attn.in_proj_a.weight"), "bf16"),
                    conv_w: st.f32(&format!("{p}linear_attn.conv1d.weight")),
                    dt_bias: st.f32(&format!("{p}linear_attn.dt_bias")),
                    a_log: st.f32(&format!("{p}linear_attn.A_log")),
                    norm_w: st.f32(&format!("{p}linear_attn.norm.weight")),
                    w_out: Weight::load(&st, &format!("{p}linear_attn.out_proj.weight"), lq),
                }
            };
            layers.push(Layer {
                ln1: norm_w(&format!("{p}input_layernorm.weight")),
                attn,
                ln2: norm_w(&format!("{p}post_attention_layernorm.weight")),
                mlp: if st.has(&format!("{p}mlp.gate.weight")) {
                    let experts = (0..config.num_experts).map(|e| Expert {
                        w_gate: Weight::load(&st, &format!("{p}mlp.experts.{e}.gate_proj.weight"), &quant),
                        w_up: Weight::load(&st, &format!("{p}mlp.experts.{e}.up_proj.weight"), &quant),
                        w_down: Weight::load(&st, &format!("{p}mlp.experts.{e}.down_proj.weight"), &quant),
                    }).collect();
                    if l == 0 {
                        eprintln!("MoE: {} experts, top-{} per token, expert width {}", config.num_experts, config.experts_per_tok, config.moe_intermediate);
                    }
                    Mlp::Moe { router: st.f32(&format!("{p}mlp.gate.weight")), experts }
                } else {
                    Mlp::Dense(Expert {
                        w_gate: Weight::load(&st, &format!("{p}mlp.gate_proj.weight"), &quant),
                        w_up: Weight::load(&st, &format!("{p}mlp.up_proj.weight"), &quant),
                        w_down: Weight::load(&st, &format!("{p}mlp.down_proj.weight"), &quant),
                    })
                },
            });
        }
        let rd = config.rotary_dim;
        let inv_freq: Vec<f32> = (0..rd / 2).map(|i| 1.0 / config.rope_theta.powf(2.0 * i as f32 / rd as f32)).collect();
        let m = Model {
            inv_freq,
            embed: st.bf16(&format!("{}embed_tokens.weight", config.prefix)),
            norm: norm_w(&format!("{}norm.weight", config.prefix)),
            lm_head: {
                let name = if st.has("lm_head.weight") { "lm_head.weight".to_string() } else { format!("{}embed_tokens.weight", config.prefix) };
                let hq = if quant == "q4" && config.lin.is_some() { "q8" } else { quant.as_str() };
                Weight::load(&st, &name, hq)
            },
            layers,
            config,
        };
        // (total params, total bytes, bytes touched per token — for MoE only the active experts count)
        let ex = |e: &Expert| (e.w_gate.params() + e.w_up.params() + e.w_down.params(), e.w_gate.bytes() + e.w_up.bytes() + e.w_down.bytes());
        let mut params = m.lm_head.params();
        let mut bytes = m.lm_head.bytes();
        let mut active = m.lm_head.bytes();
        for l in &m.layers {
            let ws: Vec<&Weight> = match &l.attn {
                Attn::Full { wq, wk, wv, wo, .. } => vec![wq, wk, wv, wo],
                Attn::Lin { w_qkv, w_z, w_b, w_a, w_out, .. } => vec![w_qkv, w_z, w_b, w_a, w_out],
            };
            for w in ws {
                params += w.params();
                bytes += w.bytes();
                active += w.bytes();
            }
            match &l.mlp {
                Mlp::Dense(e) => { let (p, b) = ex(e); params += p; bytes += b; active += b; }
                Mlp::Moe { experts, .. } => {
                    for e in experts { let (p, b) = ex(e); params += p; bytes += b; }
                    let (_, b) = ex(&experts[0]);
                    active += b * m.config.experts_per_tok;
                }
            }
        }
        eprintln!("loaded {} layers, {:.2}B params, {:.2} GB in RAM, {:.2} GB streamed per token ({quant}) in {:.1}s",
            m.config.layers, params as f64 / 1e9, bytes as f64 / 1e9, active as f64 / 1e9, t0.elapsed().as_secs_f32());
        m
    }

    /// One token in, logits over the vocabulary out. `cache.len` is the position.
    pub fn forward(&self, token: u32, cache: &mut KvCache) -> Vec<f32> {
        let mut caches = [cache];
        self.forward_multi(&[(token, 0)], &mut caches).pop().unwrap()
    }

    /// Prefill: run `tokens` (positions cache.len..) through the network in one batched pass
    /// per layer, filling the KV cache. Returns logits for the last token only.
    pub fn forward_batch(&self, tokens: &[u32], cache: &mut KvCache) -> Vec<f32> {
        let items: Vec<(u32, usize)> = tokens.iter().map(|&t| (t, 0)).collect();
        let mut caches = [cache];
        self.forward_multi(&items, &mut caches).pop().unwrap()
    }

    /// The general step: `items` are (token, sequence index) pairs, in order; consecutive items
    /// of the same sequence take consecutive positions. Every weight matrix is streamed from
    /// RAM exactly once for the whole batch — this is what makes prefill fast (many tokens of
    /// one sequence) and batched decode efficient (one token each of many sequences).
    /// Returns logits for the last item of each sequence, in order of first appearance.
    pub fn forward_multi(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, false)
    }

    /// Like forward_multi but returns logits for EVERY item (needed to verify draft tokens).
    pub fn forward_multi_all(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache]) -> Vec<Vec<f32>> {
        self.forward_impl(items, caches, true)
    }

    fn forward_impl(&self, items: &[(u32, usize)], caches: &mut [&mut KvCache], all_logits: bool) -> Vec<Vec<f32>> {
        let m = items.len();
        let c = &self.config;
        let (h, hd, kvh) = (c.heads, c.head_dim, c.kv_heads);
        let qd = h * hd;
        let kd = kvh * hd;
        let stride = kd;

        // Position of each item and which items are the last of their sequence.
        let mut pos = vec![0usize; m];
        let mut seen = vec![0usize; caches.len()];
        for (j, &(_, sq)) in items.iter().enumerate() {
            pos[j] = caches[sq].len + seen[sq];
            seen[sq] += 1;
            assert!(pos[j] < c.max_context, "context full");
        }
        // Rotary cos/sin per item (same for every layer and head). Partial rotary: only the
        // first rotary_dim dims of each head rotate (rotate_half pairs within that slice).
        let half = c.rotary_dim / 2;
        let mut rcos = vec![0f32; m * half];
        let mut rsin = vec![0f32; m * half];
        for j in 0..m {
            for i in 0..half {
                let (sn, cs) = (pos[j] as f32 * self.inv_freq[i]).sin_cos();
                rcos[j * half + i] = cs;
                rsin[j * half + i] = sn;
            }
        }
        let mut is_last = vec![false; m];
        let mut order = Vec::new();
        for j in (0..m).rev() {
            let sq = items[j].1;
            if !order.contains(&sq) {
                order.push(sq);
                is_last[j] = true;
            }
        }
        order.reverse();
        for (sq, n) in seen.iter().enumerate() {
            let need = (caches[sq].len + n) * stride;
            for lc in caches[sq].layers.iter_mut() {
                match lc {
                    LayerCache::Kv { k, v } => {
                        if k.len() < need {
                            k.resize(need, 0.0);
                            v.resize(need, 0.0);
                        }
                    }
                    LayerCache::Lin { state, conv } => {
                        if let Some(lin) = &c.lin {
                            if state.is_empty() {
                                state.resize(lin.nv * lin.dk * lin.dv, 0.0);
                                conv.resize(lin.conv_dim() * (lin.conv_k - 1), 0.0);
                            }
                        }
                    }
                }
            }
        }
        // Contiguous per-sequence item ranges (required for the recurrent layers).
        let mut ranges: Vec<(usize, usize, usize)> = Vec::new(); // (seq, start, count)
        for (j, &(_, sq)) in items.iter().enumerate() {
            match ranges.last_mut() {
                Some(r) if r.0 == sq => r.2 += 1,
                _ => ranges.push((sq, j, 1)),
            }
        }

        let mut x = vec![0.0; m * c.hidden];
        for (j, &(t, _)) in items.iter().enumerate() {
            for (d, &b) in self.embed[t as usize * c.hidden..(t as usize + 1) * c.hidden].iter().enumerate() {
                x[j * c.hidden + d] = bf16_to_f32(b);
            }
        }
        let profile = std::env::var("PROFILE").is_ok();
        let layer_debug = std::env::var("LAYER_DEBUG").is_ok();
        if layer_debug {
            let x0 = &x[(m - 1) * c.hidden..m * c.hidden];
            let norm = x0.iter().map(|v| v * v).sum::<f32>().sqrt();
            eprintln!("embed      |x|={norm:10.3} first4=[{:.4}, {:.4}, {:.4}, {:.4}]", x0[0], x0[1], x0[2], x0[3]);
        }
        let mut tm = [0f64; 6]; // qkv, norm+rope+store, attention, o, mlp, head
        let mut tick = std::time::Instant::now();
        let mut lap = |slot: usize, tick: &mut std::time::Instant| {
            if profile {
                tm[slot] += tick.elapsed().as_secs_f64();
                *tick = std::time::Instant::now();
            }
        };
        let qrows = if c.attn_gate { 2 * qd } else { qd };
        let mut hn = vec![0.0; m * c.hidden];
        let mut q = vec![0.0; m * qrows];
        let mut k = vec![0.0; m * kd];
        let mut v = vec![0.0; m * kd];
        let mut attn = vec![0.0; m * qd];
        // linear-attention scratch (allocated only for hybrid models)
        let (mut lqkv, mut lz, mut la, mut lb, mut lo) = match &c.lin {
            Some(lin) => (vec![0.0f32; m * lin.conv_dim()], vec![0.0f32; m * lin.value_dim()], vec![0.0f32; m * lin.nv], vec![0.0f32; m * lin.nv], vec![0.0f32; m * lin.value_dim()]),
            None => (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()),
        };
        let mut o = vec![0.0; m * c.hidden];
        let mut gate = vec![0.0; m * c.intermediate];
        let mut up = vec![0.0; m * c.intermediate];

        for (li, l) in self.layers.iter().enumerate() {
            hn.copy_from_slice(&x);
            for j in 0..m {
                rmsnorm(&mut hn[j * c.hidden..(j + 1) * c.hidden], &l.ln1, c.eps);
            }
            match &l.attn {
                Attn::Full { wq, wk, wv, wo, q_norm, k_norm, gate } => {
                    wq.matmul(&hn, m, &mut q);
                    wk.matmul(&hn, m, &mut k);
                    wv.matmul(&hn, m, &mut v);
                    lap(0, &mut tick);
                    // Per head, wq rows are [q | gate] when gated; q_norm/rope apply to the q part.
                    let qstride = if *gate { 2 * hd } else { hd };
                    for j in 0..m {
                        let p = pos[j];
                        let (cs, sn) = (&rcos[j * half..(j + 1) * half], &rsin[j * half..(j + 1) * half]);
                        for hi in 0..h {
                            let qh = &mut q[j * qrows + hi * qstride..j * qrows + hi * qstride + hd];
                            rmsnorm(qh, q_norm, c.eps);
                            rope_tab(&mut qh[..c.rotary_dim], cs, sn);
                        }
                        for hi in 0..kvh {
                            let kh = &mut k[j * kd + hi * hd..j * kd + (hi + 1) * hd];
                            rmsnorm(kh, k_norm, c.eps);
                            rope_tab(&mut kh[..c.rotary_dim], cs, sn);
                        }
                        if let LayerCache::Kv { k: ck, v: cv } = &mut caches[items[j].1].layers[li] {
                            ck[p * stride..(p + 1) * stride].copy_from_slice(&k[j * kd..(j + 1) * kd]);
                            cv[p * stride..(p + 1) * stride].copy_from_slice(&v[j * kd..(j + 1) * kd]);
                        }
                    }
                    lap(1, &mut tick);
                    {
                        // Gather q parts contiguously when gated (attention expects heads*hd rows).
                        let qbuf: &[f32] = if *gate {
                            for j in 0..m {
                                for hi in 0..h {
                                    for d in 0..hd {
                                        attn[j * qd + hi * hd + d] = q[j * qrows + hi * qstride + d];
                                    }
                                }
                            }
                            // attn temporarily holds gathered q; use v buffer? keep simple: copy into a fresh slice
                            &attn
                        } else {
                            &q
                        };
                        let qgathered: Vec<f32> = qbuf[..m * qd].to_vec();
                        let kv: Vec<(&[f32], &[f32], usize)> = (0..m).map(|j| {
                            match &caches[items[j].1].layers[li] {
                                LayerCache::Kv { k: ck, v: cv } => {
                                    let end = (pos[j] + 1) * stride;
                                    (&ck[..end], &cv[..end], pos[j])
                                }
                                _ => unreachable!(),
                            }
                        }).collect();
                        attention_multi(&qgathered, &kv, stride, h, kvh, hd, &mut attn);
                    }
                    if *gate {
                        for j in 0..m {
                            for hi in 0..h {
                                for d in 0..hd {
                                    let g = q[j * qrows + hi * qstride + hd + d];
                                    attn[j * qd + hi * hd + d] *= 1.0 / (1.0 + (-g).exp());
                                }
                            }
                        }
                    }
                    lap(2, &mut tick);
                    wo.matmul(&attn, m, &mut o);
                }
                Attn::Lin { w_qkv, w_z, w_b, w_a, conv_w, dt_bias, a_log, norm_w, w_out } => {
                    let lin = c.lin.as_ref().unwrap();
                    w_qkv.matmul(&hn, m, &mut lqkv);
                    w_z.matmul(&hn, m, &mut lz);
                    w_b.matmul(&hn, m, &mut lb);
                    w_a.matmul(&hn, m, &mut la);
                    lap(0, &mut tick);
                    for &(sq, start, count) in &ranges {
                        let (state, conv) = match &mut caches[sq].layers[li] {
                            LayerCache::Lin { state, conv } => (state, conv),
                            _ => unreachable!(),
                        };
                        delta_net(lin, conv_w, dt_bias, a_log, norm_w, c.eps,
                            &mut lqkv[start * lin.conv_dim()..(start + count) * lin.conv_dim()],
                            &lz[start * lin.value_dim()..(start + count) * lin.value_dim()],
                            &la[start * lin.nv..(start + count) * lin.nv],
                            &lb[start * lin.nv..(start + count) * lin.nv],
                            state, conv, count,
                            &mut lo[start * lin.value_dim()..(start + count) * lin.value_dim()]);
                    }
                    lap(2, &mut tick);
                    w_out.matmul(&lo, m, &mut o);
                }
            }
            for i in 0..m * c.hidden {
                x[i] += o[i];
            }
            lap(3, &mut tick);

            hn.copy_from_slice(&x);
            for j in 0..m {
                rmsnorm(&mut hn[j * c.hidden..(j + 1) * c.hidden], &l.ln2, c.eps);
            }
            match &l.mlp {
                Mlp::Dense(e) => {
                    e.w_gate.matmul(&hn, m, &mut gate);
                    e.w_up.matmul(&hn, m, &mut up);
                    for i in 0..m * c.intermediate {
                        gate[i] = silu(gate[i]) * up[i];
                    }
                    e.w_down.matmul(&gate, m, &mut o);
                    for i in 0..m * c.hidden {
                        x[i] += o[i];
                    }
                }
                Mlp::Moe { router, experts } => self.moe_forward(router, experts, &hn, m, &mut x),
            }
            lap(4, &mut tick);
            if layer_debug {
                // last item's row: comparable to HF's hidden_states[...,-1] on identical ids
                let x0 = &x[(m - 1) * c.hidden..m * c.hidden];
                let norm = x0.iter().map(|v| v * v).sum::<f32>().sqrt();
                let mx = x0.iter().fold(0f32, |a, &b| a.max(b.abs()));
                let nan = x0.iter().filter(|v| !v.is_finite()).count();
                eprintln!("layer {li:2} ({}) |x|={norm:10.3} max|x|={mx:9.3} nonfinite={nan} first4=[{:.4}, {:.4}, {:.4}, {:.4}]",
                    if c.is_full(li) { "full" } else { "lin " }, x0[0], x0[1], x0[2], x0[3]);
            }
        }

        for (sq, n) in seen.iter().enumerate() {
            caches[sq].len += n;
        }

        // Logits only for the last item of each sequence, batched through lm_head.
        let lasts: Vec<usize> = (0..m).filter(|&j| all_logits || is_last[j]).collect();
        let mut xl = vec![0.0; lasts.len() * c.hidden];
        for (r, &j) in lasts.iter().enumerate() {
            let row = &mut xl[r * c.hidden..(r + 1) * c.hidden];
            row.copy_from_slice(&x[j * c.hidden..(j + 1) * c.hidden]);
            rmsnorm(row, &self.norm, c.eps);
        }
        let mut logits = vec![0.0; lasts.len() * c.vocab];
        self.lm_head.matmul(&xl, lasts.len(), &mut logits);
        lap(5, &mut tick);
        if profile {
            eprintln!("step m={m}: qkv {:.1}ms | norm/rope/kv {:.1}ms | attn {:.1}ms | o {:.1}ms | mlp {:.1}ms | head {:.1}ms",
                tm[0] * 1e3, tm[1] * 1e3, tm[2] * 1e3, tm[3] * 1e3, tm[4] * 1e3, tm[5] * 1e3);
        }
        logits.chunks(c.vocab).map(|l| l.to_vec()).collect()
    }
}

impl Model {
    /// Mixture of experts for m tokens: route each token to its top-k experts, then run each
    /// expert ONCE on all tokens routed to it (a batched matmul), and scatter the weighted
    /// outputs back. Only the chosen experts' weights are touched — that's the MoE speed win.
    fn moe_forward(&self, router: &[f32], experts: &[Expert], hn: &[f32], m: usize, x: &mut [f32]) {
        let c = &self.config;
        let (ne, k) = (c.num_experts, c.experts_per_tok);
        // routing
        let mut logits = vec![0f32; ne];
        let mut assign: Vec<Vec<(usize, f32)>> = vec![Vec::new(); ne]; // expert -> [(token, weight)]
        for j in 0..m {
            let h = &hn[j * c.hidden..(j + 1) * c.hidden];
            for e in 0..ne {
                logits[e] = router[e * c.hidden..(e + 1) * c.hidden].iter().zip(h).map(|(a, b)| a * b).sum();
            }
            softmax(&mut logits);
            let mut idx: Vec<usize> = (0..ne).collect();
            idx.select_nth_unstable_by(k - 1, |&a, &b| logits[b].total_cmp(&logits[a]));
            let top = &idx[..k];
            let norm: f32 = if c.norm_topk_prob { top.iter().map(|&e| logits[e]).sum() } else { 1.0 };
            for &e in top {
                assign[e].push((j, logits[e] / norm));
            }
        }
        // expert-grouped compute
        let mi = c.moe_intermediate;
        for (e, toks) in assign.iter().enumerate() {
            if toks.is_empty() {
                continue;
            }
            let n = toks.len();
            let mut xin = vec![0f32; n * c.hidden];
            for (r, &(j, _)) in toks.iter().enumerate() {
                xin[r * c.hidden..(r + 1) * c.hidden].copy_from_slice(&hn[j * c.hidden..(j + 1) * c.hidden]);
            }
            let mut g = vec![0f32; n * mi];
            let mut u = vec![0f32; n * mi];
            let ex = &experts[e];
            ex.w_gate.matmul(&xin, n, &mut g);
            ex.w_up.matmul(&xin, n, &mut u);
            for i in 0..n * mi {
                g[i] = silu(g[i]) * u[i];
            }
            let mut out = vec![0f32; n * c.hidden];
            ex.w_down.matmul(&g, n, &mut out);
            for (r, &(j, w)) in toks.iter().enumerate() {
                let xr = &mut x[j * c.hidden..(j + 1) * c.hidden];
                for d in 0..c.hidden {
                    xr[d] += w * out[r * c.hidden + d];
                }
            }
        }
    }
}


/// Gated DeltaNet for `count` consecutive tokens of one sequence (transcribed from the
/// Hugging Face qwen3_5 reference: causal_conv1d + torch_recurrent_gated_delta_rule).
///
/// qkv: count x conv_dim rows (modified in place by the conv), z: count x value_dim,
/// a/b: count x nv. state: nv*dk*dv recurrent memory, conv: conv_dim*(conv_k-1) rolling window.
/// out: count x value_dim = gated-rmsnorm(delta-rule output) per head.
#[allow(clippy::too_many_arguments)]
pub(crate) fn delta_net(
    lin: &LinCfg,
    conv_w: &[f32],
    dt_bias: &[f32],
    a_log: &[f32],
    norm_w: &[f32],
    eps: f32,
    qkv: &mut [f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    state: &mut [f32],
    conv: &mut [f32],
    count: usize,
    out: &mut [f32],
) {
    let cd = lin.conv_dim();
    let kk = lin.conv_k;
    let (nk, nv, dk, dv) = (lin.nk, lin.nv, lin.dk, lin.dv);
    let group = nv / nk;
    let key_dim = lin.key_dim();

    // --- depthwise causal conv over time, per channel, then SiLU (parallel over channel chunks) ---
    {
        let qp = crate::kernels::SendPtrPub(qkv.as_mut_ptr());
        let cp = crate::kernels::SendPtrPub(conv.as_mut_ptr());
        crate::pool::global().run((cd + 255) / 256, &|chunk| {
            let c0 = chunk * 256;
            let c1 = (c0 + 256).min(cd);
            for ch in c0..c1 {
                let w = &conv_w[ch * kk..(ch + 1) * kk];
                // rolling window: conv[ch*(kk-1) ..] holds the previous kk-1 raw inputs
                let mut win = [0f32; 8];
                for j in 0..kk - 1 {
                    win[j] = unsafe { *cp.get().add(ch * (kk - 1) + j) };
                }
                for t in 0..count {
                    let cur = unsafe { *qp.get().add(t * cd + ch) };
                    win[kk - 1] = cur;
                    let mut acc = 0f32;
                    for j in 0..kk {
                        acc += w[j] * win[j];
                    }
                    let y = acc / (1.0 + (-acc).exp()); // silu
                    unsafe { *qp.get().add(t * cd + ch) = y };
                    for j in 0..kk - 1 {
                        win[j] = win[j + 1];
                    }
                }
                for j in 0..kk - 1 {
                    unsafe { *cp.get().add(ch * (kk - 1) + j) = win[j] };
                }
            }
        });
    }

    // --- per-token decay/beta ---
    let mut gdecay = vec![0f32; count * nv];
    let mut beta = vec![0f32; count * nv];
    for t in 0..count {
        for hv in 0..nv {
            let softplus = {
                let x = a[t * nv + hv] + dt_bias[hv];
                if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
            };
            gdecay[t * nv + hv] = (-a_log[hv].exp() * softplus).exp();
            beta[t * nv + hv] = 1.0 / (1.0 + (-b[t * nv + hv]).exp());
        }
    }

    // --- delta rule recurrence, parallel over value heads (each head's chain is independent) ---
    let sp = crate::kernels::SendPtrPub(state.as_mut_ptr());
    let op = crate::kernels::SendPtrPub(out.as_mut_ptr());
    let qkv_ro: &[f32] = qkv;
    crate::pool::global().run(nv, &|hv| {
        let hk = hv / group; // key head serving this value head (repeat_interleave)
        let st = unsafe { std::slice::from_raw_parts_mut(sp.get().add(hv * dk * dv), dk * dv) };
        let scale = 1.0 / (dk as f32).sqrt();
        let mut qh = vec![0f32; dk];
        let mut kh = vec![0f32; dk];
        let mut delta = vec![0f32; dv];
        for t in 0..count {
            let row = &qkv_ro[t * cd..(t + 1) * cd];
            // l2-normalize q and k per key head (fla: x * rsqrt(sum(x^2) + eps))
            let qsrc = &row[hk * dk..(hk + 1) * dk];
            let ksrc = &row[key_dim + hk * dk..key_dim + (hk + 1) * dk];
            let qn = 1.0 / (qsrc.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            let knn = 1.0 / (ksrc.iter().map(|v| v * v).sum::<f32>() + 1e-6).sqrt();
            for d in 0..dk {
                qh[d] = qsrc[d] * qn * scale;
                kh[d] = ksrc[d] * knn;
            }
            let vh = &row[2 * key_dim + hv * dv..2 * key_dim + (hv + 1) * dv];
            let g = gdecay[t * nv + hv];
            let bt = beta[t * nv + hv];
            // state = g*state; kv_mem = k·state; delta = (v - kv_mem)*beta; state += k ⊗ delta; o = q·state
            for x in st.iter_mut() {
                *x *= g;
            }
            for dvi in 0..dv {
                delta[dvi] = vh[dvi];
            }
            for d in 0..dk {
                let kd_ = kh[d];
                if kd_ != 0.0 {
                    let srow = &st[d * dv..(d + 1) * dv];
                    for dvi in 0..dv {
                        delta[dvi] -= kd_ * srow[dvi];
                    }
                }
            }
            for dvi in 0..dv {
                delta[dvi] *= bt;
            }
            let mut oh = vec![0f32; dv];
            for d in 0..dk {
                let srow = &mut st[d * dv..(d + 1) * dv];
                let kd_ = kh[d];
                let qd_ = qh[d];
                for dvi in 0..dv {
                    srow[dvi] += kd_ * delta[dvi];
                    oh[dvi] += qd_ * srow[dvi];
                }
            }
            // gated rmsnorm: rmsnorm(o)*w * silu(z)
            let ss = oh.iter().map(|v| v * v).sum::<f32>() / dv as f32;
            let inv = 1.0 / (ss + eps).sqrt();
            let zh = &z[t * lin.value_dim() + hv * dv..t * lin.value_dim() + (hv + 1) * dv];
            let od = unsafe { std::slice::from_raw_parts_mut(op.get().add(t * lin.value_dim() + hv * dv), dv) };
            for dvi in 0..dv {
                let zg = zh[dvi];
                od[dvi] = oh[dvi] * inv * norm_w[dvi] * (zg / (1.0 + (-zg).exp()));
            }
        }
    });
}

/// Temperature + top-p sampling. temperature 0 → greedy.
pub struct Sampler {
    pub temperature: f32,
    pub top_p: f32,
    rng: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: f32, seed: u64) -> Sampler {
        Sampler { temperature, top_p, rng: seed | 1 }
    }

    fn next_f32(&mut self) -> f32 {
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn sample(&mut self, logits: &mut [f32]) -> u32 {
        if self.temperature <= 0.0 {
            return (0..logits.len()).max_by(|&a, &b| logits[a].total_cmp(&logits[b])).unwrap() as u32;
        }
        // Keep only the top-K logits (O(n) selection), then do temperature/top-p on those.
        // With top_p <= 0.95 the tail beyond K=256 carries negligible mass.
        const K: usize = 256;
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        if idx.len() > K {
            idx.select_nth_unstable_by(K, |&a, &b| logits[b].total_cmp(&logits[a]));
            idx.truncate(K);
        }
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let max = logits[idx[0]];
        let inv_t = 1.0 / self.temperature;
        let mut probs: Vec<f32> = idx.iter().map(|&t| ((logits[t] - max) * inv_t).exp()).collect();
        let sum: f32 = probs.iter().sum();
        probs.iter_mut().for_each(|p| *p /= sum);
        let mut cum = 0.0;
        let mut cutoff = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= self.top_p {
                cutoff = i + 1;
                break;
            }
        }
        let r = self.next_f32() * cum;
        let mut acc = 0.0;
        for i in 0..cutoff {
            acc += probs[i];
            if acc >= r {
                return idx[i] as u32;
            }
        }
        idx[0] as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f2b(f: f32) -> u16 {
        (f.to_bits() >> 16) as u16
    }

    fn rand_w(n: usize, k: usize, seed: u32) -> Weight {
        let mut st = seed.wrapping_mul(2654435761) | 1;
        let w: Vec<u16> = (0..n * k).map(|_| {
            st ^= st << 13; st ^= st >> 17; st ^= st << 5;
            f2b(((st % 2000) as f32 / 1000.0 - 1.0) * 0.05)
        }).collect();
        Weight::Bf16 { w, n, k }
    }

    /// A tiny random MoE model: checks that routing/batching is self-consistent — running
    /// tokens one at a time must give the same result as running them as one batch.
    fn tiny_moe() -> Model {
        let config = Config { hidden: 64, intermediate: 128, layers: 2, heads: 4, kv_heads: 2, head_dim: 16, vocab: 100,
            rope_theta: 10000.0, eps: 1e-6, max_context: 64, num_experts: 8, experts_per_tok: 2, moe_intermediate: 32, norm_topk_prob: true,
            rotary_dim: 16, attn_gate: false, layer_types: Vec::new(), lin: None, prefix: "model.".into() };
        let hd = config.head_dim;
        let layers = (0..config.layers).map(|l| Layer {
            ln1: vec![1.0; config.hidden],
            attn: Attn::Full {
                wq: rand_w(config.heads * hd, config.hidden, 1 + l as u32 * 10), wk: rand_w(config.kv_heads * hd, config.hidden, 2 + l as u32 * 10),
                wv: rand_w(config.kv_heads * hd, config.hidden, 3 + l as u32 * 10), wo: rand_w(config.hidden, config.heads * hd, 4 + l as u32 * 10),
                q_norm: vec![1.0; hd], k_norm: vec![1.0; hd], gate: false,
            },
            ln2: vec![1.0; config.hidden],
            mlp: Mlp::Moe {
                router: (0..config.num_experts * config.hidden).map(|i| ((i * 37 % 101) as f32 / 101.0 - 0.5) * 0.2).collect(),
                experts: (0..config.num_experts).map(|e| Expert {
                    w_gate: rand_w(config.moe_intermediate, config.hidden, 100 + e as u32 + l as u32 * 50),
                    w_up: rand_w(config.moe_intermediate, config.hidden, 200 + e as u32 + l as u32 * 50),
                    w_down: rand_w(config.hidden, config.moe_intermediate, 300 + e as u32 + l as u32 * 50),
                }).collect(),
            },
        }).collect();
        let inv_freq = (0..hd / 2).map(|i| 1.0 / config.rope_theta.powf(2.0 * i as f32 / hd as f32)).collect();
        let embed = match rand_w(config.vocab, config.hidden, 999) { Weight::Bf16 { w, .. } => w, _ => unreachable!() };
        Model { lm_head: Weight::Bf16 { w: embed.clone(), n: config.vocab, k: config.hidden }, embed, norm: vec![1.0; config.hidden], layers, inv_freq, config }
    }

    #[test]
    fn moe_batched_matches_sequential() {
        let m = tiny_moe();
        let toks = [3u32, 17, 42, 7, 99, 5];
        let mut c1 = KvCache::new(&m.config);
        let mut last = Vec::new();
        for &t in &toks {
            last = m.forward(t, &mut c1);
        }
        let mut c2 = KvCache::new(&m.config);
        let batched = m.forward_batch(&toks, &mut c2);
        assert_eq!(c1.len, c2.len);
        for (a, b) in last.iter().zip(&batched) {
            assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        }
        // and two independent sequences decoded together match decoding them alone
        let mut ca = KvCache::new(&m.config);
        let mut cb = KvCache::new(&m.config);
        let la = m.forward(11, &mut ca);
        let lb = m.forward(22, &mut cb);
        let mut ca2 = KvCache::new(&m.config);
        let mut cb2 = KvCache::new(&m.config);
        let both = m.forward_multi(&[(11, 0), (22, 1)], &mut [&mut ca2, &mut cb2]);
        for (a, b) in la.iter().zip(&both[0]) { assert!((a - b).abs() < 1e-3); }
        for (a, b) in lb.iter().zip(&both[1]) { assert!((a - b).abs() < 1e-3); }
    }
}
