//! safetensors reader. Format: 8-byte little-endian header length, JSON header mapping tensor
//! name → {dtype, shape, data_offsets}, then raw tensor bytes. We mmap the file so loading a
//! multi-GB checkpoint costs nothing until a tensor is touched.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;

pub struct SafeTensors {
    shards: Vec<(Mmap, usize)>, // (mmap, data_start)
    tensors: HashMap<String, TensorInfo>,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub start: usize,
    pub end: usize,
    pub shard: usize,
}

impl SafeTensors {
    /// Open a checkpoint directory: either a single model.safetensors or the sharded
    /// model-0000X-of-0000Y.safetensors layout (all shards are mmap'd; nothing is read yet).
    pub fn open_dir(dir: &str) -> SafeTensors {
        let single = format!("{dir}/model.safetensors");
        let mut paths: Vec<String> = if std::path::Path::new(&single).exists() {
            vec![single]
        } else {
            std::fs::read_dir(dir).expect("model dir").flatten()
                .map(|e| e.path().to_string_lossy().into_owned())
                .filter(|p| p.ends_with(".safetensors")).collect()
        };
        paths.sort();
        assert!(!paths.is_empty(), "no .safetensors files in {dir}");
        let mut st = SafeTensors { shards: Vec::new(), tensors: HashMap::new() };
        for path in &paths {
            st.add_shard(path);
        }
        st
    }

    #[allow(dead_code)] // single-shard open, used by tools and tests rather than the server
    pub fn open(path: &str) -> SafeTensors {
        let mut st = SafeTensors { shards: Vec::new(), tensors: HashMap::new() };
        st.add_shard(path);
        st
    }

    fn add_shard(&mut self, path: &str) {
        let file = File::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
        let mmap = unsafe { Mmap::map(&file).expect("mmap") };
        let n = u64::from_le_bytes(mmap[..8].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&mmap[8..8 + n]).expect("safetensors header");
        let shard = self.shards.len();
        for (name, v) in header.as_object().unwrap() {
            if name == "__metadata__" {
                continue;
            }
            let off = v["data_offsets"].as_array().unwrap();
            self.tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype: v["dtype"].as_str().unwrap().to_string(),
                    shape: v["shape"].as_array().unwrap().iter().map(|x| x.as_u64().unwrap() as usize).collect(),
                    start: off[0].as_u64().unwrap() as usize,
                    end: off[1].as_u64().unwrap() as usize,
                    shard,
                },
            );
        }
        self.shards.push((mmap, 8 + n));
    }

    pub fn has(&self, name: &str) -> bool {
        self.tensors.contains_key(name)
    }

    pub fn info(&self, name: &str) -> &TensorInfo {
        self.tensors.get(name).unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    /// Tensor as bf16 bit patterns (the checkpoint's native dtype). Copied out of the mmap so the
    /// weights are aligned and contiguous in RAM.
    pub fn bf16(&self, name: &str) -> Vec<u16> {
        let t = self.info(name);
        assert_eq!(t.dtype, "BF16", "{name}: expected BF16, got {}", t.dtype);
        let (mmap, data_start) = &self.shards[t.shard];
        let bytes = &mmap[data_start + t.start..data_start + t.end];
        bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// Read a tensor as f32 whatever its stored precision. Checkpoints are inconsistent about
    /// small vectors: Qwen3.5-27B stores linear_attn.A_log/dt_bias as BF16, Qwen3.5-9B stores
    /// the same tensors as F32 -- asserting one dtype rejected a valid checkpoint.
    pub fn f32(&self, name: &str) -> Vec<f32> {
        let t = self.tensors.get(name).unwrap_or_else(|| panic!("tensor {name} not found"));
        match t.dtype.as_str() {
            "F32" => {
                let (mmap, data_start) = &self.shards[t.shard];
                let bytes = &mmap[data_start + t.start..data_start + t.end];
                bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
            }
            "BF16" => self.bf16(name).into_iter().map(crate::kernels::bf16_to_f32).collect(),
            other => panic!("{name}: expected F32 or BF16, got {other}"),
        }
    }

    #[allow(dead_code)] // used by the fixture generator and ad-hoc inspection
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}
