//! forge-engine: one process per server. Speaks the OpenAI-compatible SSE protocol the
//! gateway consumes (`POST /v1/chat/completions`, `GET /health`).
//!
//!   MODEL=../models/qwen3-0.6b ADDR=0.0.0.0:9000 forge-engine          # serve
//!   MODEL=../models/qwen3-0.6b forge-engine --prompt "hello"           # one-shot CLI test
//!
//! Env: MAX_CONTEXT (default 4096), THINK=1 to enable Qwen3 thinking mode, MODEL_NAME,
//!      DRAFT_MODEL=<dir> + SPEC_K=<n> for speculative decoding, QUANT / DRAFT_QUANT.

mod backend;
mod gpu;
mod gpu_model;
mod kernels;
mod model;
mod pool;
mod safetensors;
mod scheduler;
mod server;
mod tokenizer;

use std::sync::Arc;

fn main() {
    let dir = std::env::var("MODEL").expect("set MODEL=<checkpoint dir with config.json, model.safetensors, tokenizer.json>");
    let name = std::env::var("MODEL_NAME").unwrap_or_else(|_| std::path::Path::new(&dir).file_name().unwrap().to_string_lossy().into_owned());
    let think = std::env::var("THINK").map_or(false, |v| v == "1");

    let tokenizer = tokenizer::Tokenizer::load(&format!("{dir}/tokenizer.json"));
    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--tokenize") {
        let ids = tokenizer.encode(&args[i + 1].replace("\\n", "\n"));
        println!("{ids:?}");
        println!("{:?}", tokenizer.decode(&ids));
        return;
    }
    if args.iter().any(|a| a == "--bench") {
        bench();
        return;
    }
    if args.iter().any(|a| a == "--gpu-bench") {
        match gpu::Gpu::init() {
            Some(g) => gpu::bench(&g),
            None => eprintln!("no GPU adapter found"),
        }
        return;
    }
    let model = model::Model::load(&dir);
    if args.iter().any(|a| a == "--gpu-check") {
        let g = gpu::Gpu::init().expect("no GPU");
        let gm = gpu_model::GpuModel::from_cpu(g, &model);
        let toks: Vec<u32> = tokenizer.encode("<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
        let mut cc = model::KvCache::new(&model.config);
        let t = std::time::Instant::now();
        let lc = model.forward_batch(&toks, &mut cc);
        let cpu_s = t.elapsed().as_secs_f32();
        let mut gc = gm.new_cache();
        let t = std::time::Instant::now();
        let lg = gm.forward_batch(&toks, &mut gc);
        let gpu_s = t.elapsed().as_secs_f32();
        let maxerr = lc.iter().zip(&lg).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        let argc = (0..lc.len()).max_by(|&a, &b| lc[a].total_cmp(&lc[b])).unwrap();
        let argg = (0..lg.len()).max_by(|&a, &b| lg[a].total_cmp(&lg[b])).unwrap();
        eprintln!("prefill {} tok: cpu {:.3}s gpu {:.3}s | logits max abs err {maxerr:.4} | argmax cpu {argc} gpu {argg} ({:?})", toks.len(), cpu_s, gpu_s, tokenizer.decode(&[argg as u32]));
        // decode a few tokens on each and compare
        let (mut tc, mut tg) = (argc as u32, argg as u32);
        let mut same = 0;
        let t = std::time::Instant::now();
        for _ in 0..20 {
            let lc = model.forward(tc, &mut cc);
            tc = (0..lc.len()).max_by(|&a, &b| lc[a].total_cmp(&lc[b])).unwrap() as u32;
            let _ = &mut tg;
        }
        let cpu_dec = 20.0 / t.elapsed().as_secs_f32();
        let t = std::time::Instant::now();
        let mut outg = Vec::new();
        for _ in 0..20 {
            let lg = gm.forward_multi(&[(tg, 0)], &mut [&mut gc]).pop().unwrap();
            tg = (0..lg.len()).max_by(|&a, &b| lg[a].total_cmp(&lg[b])).unwrap() as u32;
            outg.push(tg);
        }
        let gpu_dec = 20.0 / t.elapsed().as_secs_f32();
        let _ = same;
        eprintln!("decode: cpu {cpu_dec:.1} tok/s, gpu {gpu_dec:.1} tok/s | gpu text: {:?}", tokenizer.decode(&outg));
        return;
    }
    // Optional draft model for speculative decoding (same tokenizer family required).
    let draft = std::env::var("DRAFT_MODEL").ok().map(|d| {
        let q = std::env::var("DRAFT_QUANT").unwrap_or_else(|_| "q8".into());
        eprintln!("draft model: {d} ({q})");
        model::Model::load_with(&d, &q)
    });
    // DEVICE=auto (default): use a GPU if one is present and the model fits; cpu | gpu to force.
    let want = std::env::var("DEVICE").unwrap_or_else(|_| "auto".into());
    // auto: only move to the GPU when the weights fit comfortably (GPU_MEM_MB, default 2048 → models < 1 GB).
    let gpu_mem_mb: usize = std::env::var("GPU_MEM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let fits = model.weight_bytes() < gpu_mem_mb * 1024 * 1024 / 2;
    if want == "auto" && !fits {
        eprintln!("device: auto → cpu (model {:.2} GB vs GPU_MEM_MB={gpu_mem_mb}; set DEVICE=gpu to force)", model.weight_bytes() as f64 / 1e9);
    }
    let net = if (want == "gpu" || (want == "auto" && fits)) && model.layers_mlp_dense() {
        // Any failure inside the GPU stack (driver quirks, limits) must not take the engine down.
        let attempt = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            gpu::Gpu::init().map(|g| {
                let cfg = model.config.clone();
                if cfg.max_context > gpu_model::GPU_MAX_CONTEXT {
                    eprintln!("gpu: context capped at {} (MAX_CONTEXT={})", gpu_model::GPU_MAX_CONTEXT, cfg.max_context);
                }
                gpu_model::GpuModel::from_cpu(g, &model)
            })
        }));
        match attempt {
            Ok(Some(gm)) => backend::Net::Gpu(Arc::new(gm)),
            Ok(None) => {
                if want == "gpu" { panic!("DEVICE=gpu but no usable GPU adapter found"); }
                backend::Net::Cpu(Arc::new(model))
            }
            Err(_) => {
                if want == "gpu" { panic!("DEVICE=gpu but GPU initialization failed (see messages above)"); }
                eprintln!("gpu: initialization failed, falling back to cpu");
                backend::Net::Cpu(Arc::new(model))
            }
        }
    } else {
        backend::Net::Cpu(Arc::new(model))
    };
    eprintln!("device: {}", net.device());
    let engine = Arc::new(server::Engine::new(net, draft, tokenizer, name, think));

    let args: Vec<String> = std::env::args().collect();
    if let Some(i) = args.iter().position(|a| a == "--prompt") {
        cli(&engine, &args[i + 1]);
        return;
    }
    let addr = std::env::var("ADDR").unwrap_or_else(|_| "0.0.0.0:9000".into());
    eprintln!("forge-engine serving {} on {addr} ({}, context {}, threads {})", engine.model_name, engine.model.device(), engine.model.config().max_context,
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    server::serve(&addr, engine);
}

/// Generate to stdout without HTTP — the quickest way to check the model is producing sense.
fn cli(engine: &server::Engine, prompt: &str) {
    use std::io::Write;
    let tk = &engine.tokenizer;
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{}", if engine.think { "" } else { "<think>\n\n</think>\n\n" });
    let ids = tk.encode(&text);
    eprintln!("prompt tokens: {:?}", ids);
    let mut cache = engine.model.new_cache();
    let t0 = std::time::Instant::now();
    let mut logits = engine.model.forward_batch(&ids, &mut cache);
    eprintln!("prefill: {:.1} tok/s", ids.len() as f32 / t0.elapsed().as_secs_f32());
    let mut sampler = model::Sampler::new(0.0, 1.0, 1);
    let t1 = std::time::Instant::now();
    let mut n = 0;
    let mut pending = Vec::new();
    for _ in 0..128 {
        let next = sampler.sample(&mut logits);
        if next == tk.im_end || next == tk.eos {
            break;
        }
        n += 1;
        pending.extend(tk.token_bytes(next));
        if let Ok(s) = std::str::from_utf8(&pending) {
            print!("{s}");
            std::io::stdout().flush().unwrap();
            pending.clear();
        }
        logits = engine.model.forward_multi(&[(next, 0)], &mut [&mut cache], false).pop().unwrap();
    }
    eprintln!("\ndecode: {:.1} tok/s ({n} tokens)", n as f32 / t1.elapsed().as_secs_f32());
}

/// Raw kernel throughput on typical Qwen3 shapes, independent of the model.
fn bench() {
    use std::time::Instant;
    let (n, k) = (3072, 1024);
    let w: Vec<u16> = (0..n * k).map(|i| ((((i * 2654435761usize) >> 13) % 2001) as f32 / 1000.0 - 1.0).to_bits() as u32 >> 16).map(|b| b as u16).collect();
    let mut y = vec![0f32; n];
    let x: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.01).collect();
    let iters = 200;
    kernels::matvec_bf16(&w, &x, n, k, &mut y);
    let t = Instant::now();
    for _ in 0..iters {
        kernels::matvec_bf16(&w, &x, n, k, &mut y);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("matvec  {n}x{k}: {:.2} ms  {:.1} GB/s  {:.1} GFLOPS", dt * 1e3, (n * k * 2) as f64 / dt / 1e9, (2 * n * k) as f64 / dt / 1e9);
    for m in [8usize, 32, 128] {
        let xs: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut ys = vec![0f32; m * n];
        kernels::matmul_bf16(&w, &xs, m, n, k, &mut ys);
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            kernels::matmul_bf16(&w, &xs, m, n, k, &mut ys);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!("matmul  m={m:<3} {n}x{k}: {:.2} ms  {:.1} GFLOPS", dt * 1e3, (2 * m * n * k) as f64 / dt / 1e9);
    }
    let qm = kernels::QMat::from_bf16(&w, n, k);
    kernels::matvec_q8(&qm, &x, &mut y);
    let t = Instant::now();
    for _ in 0..iters {
        kernels::matvec_q8(&qm, &x, &mut y);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("matvec_q8 {n}x{k}: {:.2} ms  {:.1} GB/s", dt * 1e3, qm.bytes() as f64 / dt / 1e9);
    for m in [8usize, 32, 128] {
        let xs: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut ys = vec![0f32; m * n];
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            kernels::matmul_q8(&qm, &xs, m, &mut ys);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!("matmul_q8 m={m:<3} {n}x{k}: {:.2} ms  {:.1} GFLOPS", dt * 1e3, (2 * m * n * k) as f64 / dt / 1e9);
    }
    let q4 = kernels::Q4Mat::from_bf16(&w, n, k);
    kernels::matvec_q4(&q4, &x, &mut y);
    let t = Instant::now();
    for _ in 0..iters {
        kernels::matvec_q4(&q4, &x, &mut y);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("matvec_q4 {n}x{k}: {:.2} ms  {:.1} GB/s", dt * 1e3, q4.bytes() as f64 / dt / 1e9);
    // Single-thread, L1-resident: how good is the inner tile kernel itself?
    let k = 1024;
    let rows: Vec<f32> = (0..4 * k).map(|i| (i % 9) as f32 * 0.1).collect();
    let xs: Vec<f32> = (0..2 * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let mut sink = 0f32;
    let iters = 20000;
    let t = Instant::now();
    for _ in 0..iters {
        let r = kernels::tile4x2_pub(&rows[..k], &rows[k..2 * k], &rows[2 * k..3 * k], &rows[3 * k..], &xs[..k], &xs[k..]);
        sink += r[0];
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("tile4x2 single-thread L1: {:.1} GFLOPS (sink {sink:.1})", (2 * 8 * k) as f64 / dt / 1e9);
    // Same work two ways: manual tile loop vs matmul_bf16 on n=32 rows, m=8 (single chunk).
    let (n2, m2) = (32usize, 8usize);
    let rows2: Vec<f32> = (0..n2 * k).map(|i| (i % 9) as f32 * 0.1).collect();
    let w2: Vec<u16> = rows2.iter().map(|f| (f.to_bits() >> 16) as u16).collect();
    let xs2: Vec<f32> = (0..m2 * k).map(|i| (i % 7) as f32 * 0.1).collect();
    let mut ys2 = vec![0f32; m2 * n2];
    let iters = 2000;
    let t = Instant::now();
    for _ in 0..iters {
        for i in (0..n2).step_by(4) {
            for j in (0..m2).step_by(2) {
                let r = kernels::tile4x2_pub(&rows2[i * k..(i + 1) * k], &rows2[(i + 1) * k..(i + 2) * k], &rows2[(i + 2) * k..(i + 3) * k], &rows2[(i + 3) * k..(i + 4) * k], &xs2[j * k..(j + 1) * k], &xs2[(j + 1) * k..(j + 2) * k]);
                ys2[j * n2 + i] = r[0];
                ys2[(j + 1) * n2 + i] = r[4];
            }
        }
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("manual tile loop n=32 m=8: {:.1} GFLOPS", (2 * m2 * n2 * k) as f64 / dt / 1e9);
    let t = Instant::now();
    for _ in 0..iters {
        kernels::matmul_bf16(&w2, &xs2, m2, n2, k, &mut ys2);
    }
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("matmul_bf16     n=32 m=8: {:.1} GFLOPS", (2 * m2 * n2 * k) as f64 / dt / 1e9);
    let (heads, kvh, hd, pos) = (16, 8, 128, 400);
    let stride = kvh * hd;
    let kc: Vec<f32> = (0..(pos + 1) * stride).map(|i| (i % 7) as f32 * 0.1).collect();
    let q: Vec<f32> = (0..heads * hd).map(|i| (i % 5) as f32 * 0.1).collect();
    let mut out = vec![0f32; heads * hd];
    let t = Instant::now();
    for _ in 0..100 {
        kernels::attention(&q, &kc, &kc, stride, pos, heads, kvh, hd, &mut out);
    }
    let dt = t.elapsed().as_secs_f64() / 100.0;
    eprintln!("attention pos={pos}: {:.3} ms/token/layer  (x28 layers = {:.1} ms/token)", dt * 1e3, dt * 28e3);
}
