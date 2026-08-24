//! Device abstraction: the scheduler talks to `Net`/`Kv` and doesn't care whether the math
//! runs on the CPU kernels or the GPU shaders.

use crate::gpu_model::{GpuKv, GpuModel};
use crate::model::{Config, KvCache, Model};
use std::sync::Arc;

pub enum Net {
    Cpu(Arc<Model>),
    Gpu(Arc<GpuModel>),
}

pub enum Kv {
    Cpu(KvCache),
    Gpu(GpuKv),
}

impl Kv {
    pub fn len(&self) -> usize {
        match self {
            Kv::Cpu(c) => c.len,
            Kv::Gpu(g) => g.len,
        }
    }
    pub fn can_truncate(&self, n: usize) -> bool {
        match self {
            Kv::Cpu(c) => c.can_truncate(n),
            Kv::Gpu(_) => true,
        }
    }

    pub fn truncate(&mut self, n: usize) {
        match self {
            Kv::Cpu(c) => c.truncate(n),
            Kv::Gpu(g) => g.truncate(n),
        }
    }
    pub fn bytes(&self) -> usize {
        match self {
            Kv::Cpu(c) => c.bytes(),
            Kv::Gpu(g) => g.bytes(),
        }
    }
}

impl Net {
    pub fn config(&self) -> &Config {
        match self {
            Net::Cpu(m) => &m.config,
            Net::Gpu(g) => &g.config,
        }
    }

    /// False for recurrent (hybrid linear-attention) models: their state cannot be rolled back,
    /// so speculative decoding and partial prefix reuse are disabled.
    pub fn supports_rollback(&self) -> bool {
        match self {
            Net::Cpu(m) => m.supports_rollback(),
            Net::Gpu(_) => true,
        }
    }

    pub fn device(&self) -> &'static str {
        match self {
            Net::Cpu(_) => "cpu",
            Net::Gpu(_) => "gpu",
        }
    }

    pub fn new_cache(&self) -> Kv {
        match self {
            Net::Cpu(m) => Kv::Cpu(KvCache::new(&m.config)),
            Net::Gpu(g) => Kv::Gpu(g.new_cache()),
        }
    }

    pub fn clone_cache(&self, kv: &Kv) -> Kv {
        match (self, kv) {
            (Net::Cpu(_), Kv::Cpu(c)) => Kv::Cpu(c.clone()),
            (Net::Gpu(g), Kv::Gpu(c)) => Kv::Gpu(g.clone_cache(c)),
            _ => panic!("cache/device mismatch"),
        }
    }

    pub fn forward_batch(&self, tokens: &[u32], cache: &mut Kv) -> Vec<f32> {
        match (self, cache) {
            (Net::Cpu(m), Kv::Cpu(c)) => m.forward_batch(tokens, c),
            (Net::Gpu(g), Kv::Gpu(c)) => g.forward_batch(tokens, c),
            _ => panic!("cache/device mismatch"),
        }
    }

    /// True when the checkpoint carries a multi-token-prediction head we can draft with.
    pub fn has_mtp(&self) -> bool {
        match self {
            Net::Cpu(m) => m.has_mtp(),
            Net::Gpu(_) => false,
        }
    }

    pub fn mtp_cache(&self) -> Option<KvCache> {
        match self {
            Net::Cpu(m) if m.has_mtp() => Some(m.mtp_cache()),
            _ => None,
        }
    }

    /// One MTP step: logits for the token after `token`, given the main model's hidden state.
    pub fn mtp_forward(&self, hidden: &[f32], token: u32, pos: usize, cache: &mut KvCache) -> Option<Vec<f32>> {
        match self {
            Net::Cpu(m) => m.mtp_forward(hidden, token, pos, cache),
            Net::Gpu(_) => None,
        }
    }

    /// forward_multi that also returns each row's pre-final-norm hidden state (MTP input).
    pub fn forward_multi_h(&self, items: &[(u32, usize)], caches: &mut [&mut Kv]) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
        match self {
            Net::Cpu(m) => {
                let mut cs: Vec<&mut KvCache> = caches.iter_mut().map(|k| match &mut **k { Kv::Cpu(c) => c, _ => panic!("mismatch") }).collect();
                m.forward_multi_all_h(items, &mut cs)
            }
            Net::Gpu(_) => (self.forward_multi(items, caches, true), Vec::new()),
        }
    }

    pub fn forward_multi(&self, items: &[(u32, usize)], caches: &mut [&mut Kv], all_logits: bool) -> Vec<Vec<f32>> {
        match self {
            Net::Cpu(m) => {
                let mut cs: Vec<&mut KvCache> = caches.iter_mut().map(|k| match &mut **k { Kv::Cpu(c) => c, _ => panic!("mismatch") }).collect();
                if all_logits { m.forward_multi_all(items, &mut cs) } else { m.forward_multi(items, &mut cs) }
            }
            Net::Gpu(g) => {
                let mut cs: Vec<&mut GpuKv> = caches.iter_mut().map(|k| match &mut **k { Kv::Gpu(c) => c, _ => panic!("mismatch") }).collect();
                if all_logits { g.forward_multi_all(items, &mut cs) } else { g.forward_multi(items, &mut cs) }
            }
        }
    }
}
