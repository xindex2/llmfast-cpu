//! llmfast-engine: one process per server. Speaks the OpenAI-compatible SSE protocol the
//! gateway consumes (`POST /v1/chat/completions`, `GET /health`).
//!
//!   MODEL=../models/qwen3-0.6b ADDR=0.0.0.0:9000 llmfast-engine          # serve
//!   MODEL=../models/qwen3-0.6b llmfast-engine --prompt "hello"           # one-shot CLI test
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

/// Commit this binary was built from, stamped by build.rs. Reported by --version, the startup
/// line and /health, so "is the running engine current" is never again a matter of inference.
pub const COMMIT: &str = env!("LLMFAST_COMMIT");

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("llmfast-engine {COMMIT}");
        return;
    }
    // The kernel benchmarks synthesise their own matrices; requiring MODEL for them meant
    // pointing --bench at a checkpoint that it never opens.
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
    if let Some(i) = args.iter().position(|a| a == "--bench-model") {
        // Directory from the argument, falling back to MODEL — the first run in the field
        // set MODEL=... and passed "." as the argument, and "." silently won, so the bench
        // panicked on /opt/llmfast/config.json instead of using the checkpoint it was given.
        let dir = args
            .get(i + 1)
            .filter(|a| !a.starts_with("--") && a.as_str() != ".")
            .cloned()
            .or_else(|| std::env::var("MODEL").ok())
            .unwrap_or_else(|| {
                eprintln!("usage: llmfast-engine --bench-model <checkpoint dir>   (or set MODEL=<dir>)");
                std::process::exit(2);
            });
        if !std::path::Path::new(&dir).join("config.json").exists() {
            eprintln!("no config.json in {dir} — is the model downloaded? (add it in the admin Models page first)");
            std::process::exit(2);
        }
        bench_model(&dir);
        return;
    }

    let dir = std::env::var("MODEL").expect("set MODEL=<checkpoint dir with config.json, model.safetensors, tokenizer.json>");
    let name = std::env::var("MODEL_NAME").unwrap_or_else(|_| std::path::Path::new(&dir).file_name().unwrap().to_string_lossy().into_owned());
    let think = std::env::var("THINK").map_or(false, |v| v == "1");

    if let Some(i) = args.iter().position(|a| a == "--fixture") {
        run_fixture(&args[i + 1]);
        return;
    }
    // --trace-ids "1,2,3": forward raw token ids (set LAYER_DEBUG=1 for the per-layer trace)
    // and print the top-5 next tokens — for diffing against reference implementations.
    if let Some(i) = args.iter().position(|a| a == "--trace-ids") {
        let ids: Vec<u32> = args[i + 1].split(',').map(|t| t.trim().parse().unwrap()).collect();
        let tokenizer = tokenizer::Tokenizer::load(&format!("{dir}/tokenizer.json"));
        let model = model::Model::load(&dir);
        let mut cache = model::KvCache::new(&model.config);
        let logits = model.forward_batch(&ids, &mut cache);
        let mut idx: Vec<usize> = (0..logits.len()).collect();
        idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
        let top: Vec<String> = idx[..5].iter().map(|&t| format!("{t}={:?}({:.2})", tokenizer.decode(&[t as u32]), logits[t])).collect();
        println!("our top5: {}", top.join("  "));
        return;
    }
    let tokenizer = tokenizer::Tokenizer::load(&format!("{dir}/tokenizer.json"));
    if let Some(i) = args.iter().position(|a| a == "--tokenize") {
        let ids = tokenizer.encode(&args[i + 1].replace("\\n", "\n"));
        println!("{ids:?}");
        println!("{:?}", tokenizer.decode(&ids));
        return;
    }
    // --gpu-check and --prompt need the model synchronously; serving loads it in the background.
    let oneshot = args.iter().any(|a| a == "--gpu-check") || args.iter().any(|a| a == "--prompt");
    if oneshot {
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
            let same = 0;
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
        let net = build_net(model);
        eprintln!("device: {}", net.device());
        let engine = Arc::new(server::Engine::new(net, None, tokenizer, name, think));
        if let Some(i) = args.iter().position(|a| a == "--prompt") {
            cli(&engine, &args[i + 1]);
        }
        return;
    }

    let addr = std::env::var("ADDR").unwrap_or_else(|_| "0.0.0.0:9000".into());
    // Bind the port first and load the weights on a background thread: /health reports progress
    // from the first second, and requests get 503 + Retry-After instead of a refused connection.
    let slot: server::Slot = Arc::new(std::sync::RwLock::new(None));
    let slot2 = slot.clone();
    let name2 = name.clone();
    std::thread::spawn(move || {
        // A panic here must reach /health, not vanish with the thread: a dead loader behind a
        // live HTTP server reads as "loading 93%" forever, which is the worst possible error UX.
        let body = move || {
        let t0 = std::time::Instant::now();
        // The GPU shaders take q8 only. Say so before spending minutes loading weights that
        // can only end in a panic inside GpuModel::from_cpu.
        let quant = std::env::var("QUANT").unwrap_or_else(|_| "q8".into());
        if std::env::var("DEVICE").as_deref() == Ok("gpu") && quant != "q8" {
            panic!("GPU backend needs quant q8 (got {quant}) — set quant to q8, or device to cpu/auto");
        }
        let model = model::Model::load(&dir);
        let draft = std::env::var("DRAFT_MODEL").ok().map(|d| {
            let q = std::env::var("DRAFT_QUANT").unwrap_or_else(|_| "q8".into());
            eprintln!("draft model: {d} ({q})");
            model::Model::load_with(&d, &q)
        });
        let net = build_net(model);
        eprintln!("device: {}", net.device());
        let engine = Arc::new(server::Engine::new(net, draft, tokenizer, name2, think));
        model::LOAD_PROGRESS.store(1000, std::sync::atomic::Ordering::Relaxed);
        // available_parallelism() reports 1 once the pool has pinned its threads (see pool.rs),
        // so this line claimed "threads 1" on a 20-thread box for as long as it has existed.
        // Ask the pool. And say which decode kernel is live: every slow-decode investigation
        // so far has ended at "was the fast path even on", answerable only by inference.
        let (simd, i8) = kernels::kernel_report();
        eprintln!(
            "ready in {:.1}s (build {COMMIT}, context {}, threads {}, simd {}, {} decode, {} KV)",
            t0.elapsed().as_secs_f32(),
            engine.model.config().max_context,
            pool::global().threads(),
            simd,
            if i8 { "int8" } else { "float" },
            if model::kv_int8() { "int8" } else { "f32" },
        );
        *slot2.write().unwrap() = Some(engine);
        };
        if let Err(e) = std::panic::catch_unwind(std::panic::AssertUnwindSafe(body)) {
            let msg = e.downcast_ref::<String>().cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "model load panicked (see engine.log)".into());
            eprintln!("load failed: {msg}");
            server::set_load_error(msg);
        }
    });
    eprintln!("llmfast-engine listening on {addr} — loading {name}");
    server::serve(&addr, slot, name);
}

/// CPU or GPU for a loaded model: DEVICE=auto|cpu|gpu, with GPU_MEM_MB deciding what "fits".
fn build_net(model: model::Model) -> backend::Net {
    // DEVICE=auto (default): use a GPU if one is present and the model fits; cpu | gpu to force.
    let want = std::env::var("DEVICE").unwrap_or_else(|_| "auto".into());
    // auto: only move to the GPU when the weights fit comfortably (GPU_MEM_MB, default 2048 → models < 1 GB).
    let gpu_mem_mb: usize = std::env::var("GPU_MEM_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(2048);
    let fits = model.weight_bytes() < gpu_mem_mb * 1024 * 1024 / 2;
    if want == "auto" && !fits {
        eprintln!("device: auto → cpu (model {:.2} GB vs GPU_MEM_MB={gpu_mem_mb}; set DEVICE=gpu to force)", model.weight_bytes() as f64 / 1e9);
    }
    let net = if (want == "gpu" || (want == "auto" && fits)) && model.gpu_supported() {
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
    net
}

/// Generate to stdout without HTTP — the quickest way to check the model is producing sense.
fn cli(engine: &server::Engine, prompt: &str) {
    use std::io::Write;
    let tk = &engine.tokenizer;
    let suffix = if engine.model.config().lin.is_some() { "<think>\n" } else if engine.think { "" } else { "<think>\n\n</think>\n\n" };
    let text = format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{suffix}");
    let ids = tk.encode(&text);
    eprintln!("prompt tokens: {:?}", ids);
    let mut cache = engine.model.new_cache();
    let t0 = std::time::Instant::now();
    let mut logits = engine.model.forward_batch(&ids, &mut cache);
    eprintln!("prefill: {:.1} tok/s", ids.len() as f32 / t0.elapsed().as_secs_f32());
    // top-5 candidates for the first generated token: the quickest correctness signal
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<String> = idx[..5].iter().map(|&t| format!("{t}={:?}({:.2})", tk.decode(&[t as u32]), logits[t])).collect();
    eprintln!("top5: {}", top.join("  "));
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
        // print the longest valid prefix; replace a stuck invalid byte so output keeps flowing
        loop {
            match std::str::from_utf8(&pending) {
                Ok(s) => {
                    print!("{s}");
                    pending.clear();
                    break;
                }
                Err(e) => {
                    let ok = e.valid_up_to();
                    if ok > 0 {
                        print!("{}", std::str::from_utf8(&pending[..ok]).unwrap());
                        pending.drain(..ok);
                    }
                    match e.error_len() {
                        Some(bad) => {
                            print!("?");
                            pending.drain(..bad);
                        }
                        None => break, // incomplete tail: wait for more bytes
                    }
                }
            }
        }
        std::io::stdout().flush().unwrap();
        logits = engine.model.forward_multi(&[(next, 0)], &mut [&mut cache], false).pop().unwrap();
    }
    eprintln!("\ndecode: {:.1} tok/s ({n} tokens)", n as f32 / t1.elapsed().as_secs_f32());
}

/// Raw kernel throughput on typical Qwen3 shapes, independent of the model.

/// Decode-shaped benchmark on a matrix far larger than L3, which is the only way to see
/// memory placement. The 3072x1024 matrices below are ~6 MB and sit in cache on a Xeon, so
/// they measure the kernel and hide the interconnect. This one measures the memory system:
/// on a two-socket box, compare `NUMA=1` (default) against `NUMA=0`.
fn bench_stream() {
    use std::time::Instant;
    // The ceiling first, so every number below can be read as a fraction of it. Sized to the
    // same footprint as the matvecs below: a much larger probe would see no cache reuse at all
    // and report a ceiling the matvecs could then appear to "exceed".
    let probe_mb = 72usize;
    let (dt, sink) = kernels::read_bandwidth(probe_mb << 20);
    let peak = (probe_mb << 20) as f64 / dt / 1e9;
    eprintln!("memory read ceiling ({probe_mb} MB, no arithmetic): {peak:.1} GB/s  (sink {sink:x})");

    let (n, k) = (8192usize, 8192usize);
    let w: Vec<u16> = (0..n * k).map(|i| (((i * 2654435761usize) >> 11) % 4001) as u16 + 15000).collect();
    let x: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.01).collect();
    let mut y = vec![0f32; n];
    for (name, run) in [
        ("q8", Box::new({
            let q = kernels::QMat::from_bf16(&w, n, k);
            move |x: &[f32], y: &mut [f32]| kernels::matvec_q8(&q, x, y)
        }) as Box<dyn Fn(&[f32], &mut [f32])>),
        ("q4", Box::new({
            let q = kernels::Q4Mat::from_bf16(&w, n, k);
            move |x: &[f32], y: &mut [f32]| kernels::matvec_q4(&q, x, y)
        })),
    ] {
        let bytes = if name == "q8" { n * k + n * k / 32 * 4 } else { n * k / 2 + n * k / 32 * 4 };
        run(&x, &mut y);
        let iters = 30;
        let t = Instant::now();
        for _ in 0..iters {
            run(&x, &mut y);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes as f64 / dt / 1e9;
        eprintln!(
            "matvec_{name} {n}x{k} ({:.0} MB, exceeds L3): {:.2} ms  {gbs:.1} GB/s  ({:.0}% of ceiling)  -> {:.1} tok/s for a model of this size",
            bytes as f64 / 1e6,
            dt * 1e3,
            gbs / peak * 100.0,
            1.0 / dt,
        );
    }
    // The two decode kernels head to head, and how far they agree.
    {
        let q = kernels::QMat::from_bf16(&w, n, k);
        let bytes = n * k + n * k / 32 * 4;
        let xq = kernels::quantize_vec(&x);
        let mut yi = vec![0f32; n];
        kernels::matvec_q8_i8(&q, &xq, &mut yi);
        let iters = 30;
        let t = Instant::now();
        for _ in 0..iters {
            kernels::matvec_q8_i8(&q, &kernels::quantize_vec(&x), &mut yi);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes as f64 / dt / 1e9;
        eprintln!(
            "matvec_q8_i8 (integer decode) {n}x{k}: {:.2} ms  {gbs:.1} GB/s  ({:.0}% of ceiling)  -> {:.1} tok/s",
            dt * 1e3, gbs / peak * 100.0, 1.0 / dt,
        );
        let mut yf = vec![0f32; n];
        std::env::set_var("I8_DECODE", "0");
        kernels::matvec_q8(&q, &x, &mut yf);
        std::env::remove_var("I8_DECODE");
        let pk = yf.iter().fold(0f32, |m, v| m.max(v.abs())).max(1e-6);
        let err = yf.iter().zip(&yi).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        eprintln!("  int8 vs f32 decode: max abs err {err:.5} ({:.3}% of peak)", err / pk * 100.0);

        let q4 = kernels::Q4Mat::from_bf16(&w, n, k);
        let bytes4 = n * k / 2 + n * k / 32 * 4;
        let mut y4 = vec![0f32; n];
        kernels::matvec_q4_i8(&q4, &xq, &mut y4);
        let t = Instant::now();
        for _ in 0..iters {
            kernels::matvec_q4_i8(&q4, &kernels::quantize_vec(&x), &mut y4);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        let gbs = bytes4 as f64 / dt / 1e9;
        eprintln!(
            "matvec_q4_i8 (integer decode) {n}x{k}: {:.2} ms  {gbs:.1} GB/s  ({:.0}% of ceiling)  -> {:.1} tok/s",
            dt * 1e3, gbs / peak * 100.0, 1.0 / dt,
        );
    }
    eprintln!("  (near 100% of ceiling = memory-bound, only more channels help; well under = the kernel has headroom)");
}

/// Benchmark a real model's decode speed: load the actual checkpoint, run inference,
/// and print params/active-params, GB/token, and measured tok/s with the kernel config.
/// This is the only way to know "will this checkpoint hit 50 tok/s" before committing to it.
fn bench_model(dir: &str) {
    use std::time::Instant;
    let t0 = Instant::now();
    let model = model::Model::load(dir);
    let _load_s = t0.elapsed().as_secs_f32();
    // Exact numbers from the loaded weights, not re-derived from config — the first version
    // re-derived and printed 7.08B for a model the loader itself called 25.62B.
    let (params, active_bytes) = model.stats;
    let cfg = model.config.clone();
    let cfg = &cfg;
    let gb_per_token = active_bytes as f64 / 1e9;

    let tokenizer = tokenizer::Tokenizer::load(&format!("{dir}/tokenizer.json"));
    let prompt = "The future of artificial intelligence";
    let ids = tokenizer.encode(prompt);

    // Same device selection production uses: DEVICE=gpu benches the GPU path, auto/cpu the CPU.
    let num_experts = cfg.num_experts;
    let net = build_net(model);
    let device = net.device();
    let mut cache = net.new_cache();
    let t = Instant::now();
    let mut logits = net.forward_batch(&ids, &mut cache);
    let prefill_s = t.elapsed().as_secs_f32();

    // Time N decode steps
    let mut sampler = model::Sampler::new(0.7, 1.0, 1);
    let n_decode = 100usize;
    let t = Instant::now();
    for _ in 0..n_decode {
        let next = sampler.sample(&mut logits);
        if next == tokenizer.im_end || next == tokenizer.eos {
            break;
        }
        logits = net.forward_multi(&[(next, 0)], &mut [&mut cache], false).pop().unwrap();
    }
    let decode_s = t.elapsed().as_secs_f32();
    let tok_per_s = n_decode as f32 / decode_s;

    let (simd, i8) = kernels::kernel_report();
    // Achieved bandwidth = bytes streamed per token / step time. Compare directly with the
    // read ceiling from --bench: near it means memory-bound (done); well under means the time
    // is going somewhere other than streaming weights — run with PROFILE=1 to see where.
    let gbs = gb_per_token * tok_per_s as f64;
    eprintln!(
        "bench-model: {} ({device})\n  {:.2}B params · {:.2} GB streamed/token ({})\n  prefill {} tok in {:.1}s, decode {} tok in {:.2}s ({:.1} tok/s -> {gbs:.1} GB/s achieved)\n  build {} · {} threads · simd {} · {} decode · MoE {}",
        dir,
        params as f64 / 1e9, gb_per_token,
        std::env::var("QUANT").unwrap_or_else(|_| "bf16".into()),
        ids.len(), prefill_s, n_decode, decode_s, tok_per_s,
        COMMIT, pool::global().threads(), simd, if i8 { "int8" } else { "float" },
        if num_experts > 0 { format!("{num_experts} experts") } else { "no (dense)".into() },
    );
}

fn bench() {
    use std::time::Instant;
    bench_stream();
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
    // int8 GEMM vs the f32 path, same data — and check they agree
    let qm0 = kernels::QMat::from_bf16(&w, n, k);
    {
        let m = 8usize;
        let xs: Vec<f32> = (0..m * k).map(|i| ((i * 7) % 23) as f32 * 0.05 - 0.5).collect();
        let (mut a, mut b) = (vec![0f32; m * n], vec![0f32; m * n]);
        kernels::matmul_q8(&qm0, &xs, m, &mut a);
        kernels::matmul_q8_int8(&qm0, &kernels::quantize_act(&xs, m, k), m, &mut b);
        let scale = a.iter().fold(0f32, |x, &y| x.max(y.abs())).max(1.0);
        let err = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
        eprintln!("int8 vs f32 matmul: max abs err {err:.5} ({:.3}% of peak {scale:.2})", err / scale * 100.0);
    }
    for m in [8usize, 32, 128] {
        let xs: Vec<f32> = (0..m * k).map(|i| (i % 13) as f32 * 0.01).collect();
        let mut ys = vec![0f32; m * n];
        let xq = kernels::quantize_act(&xs, m, k);
        let iters = 20;
        let t = Instant::now();
        for _ in 0..iters {
            kernels::matmul_q8_int8(&qm0, &xq, m, &mut ys);
        }
        let dt = t.elapsed().as_secs_f64() / iters as f64;
        eprintln!("matmul_i8 m={m:<3} {n}x{k}: {:.2} ms  {:.1} GFLOPS", dt * 1e3, (2 * m * n * k) as f64 / dt / 1e9);
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

/// Compare our forward pass against reference logits produced by the NumPy transcription of the
/// Hugging Face modeling code (scripts in the repo history). Validates new architectures.
fn run_fixture(fdir: &str) {
    let fx: serde_json::Value = serde_json::from_slice(&std::fs::read(format!("{fdir}/fixture.json")).unwrap()).unwrap();
    let toks: Vec<u32> = fx["tokens"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let expect: Vec<f32> = fx["logits_last"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect();
    let quant = std::env::var("QUANT").unwrap_or_else(|_| "bf16".into());
    let model = model::Model::load_with(fdir, &quant);
    let mut cache = model::KvCache::new(&model.config);
    let got = model.forward_batch(&toks, &mut cache);
    let maxerr = got.iter().zip(&expect).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let mut c2 = model::KvCache::new(&model.config);
    let mut got2 = Vec::new();
    for &t in &toks {
        got2 = model.forward_batch(&[t], &mut c2);
    }
    let maxerr2 = got.iter().zip(&got2).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    println!("fixture: batched-vs-reference max abs err {maxerr:.6} | sequential-vs-batched {maxerr2:.6}");
    let scale = expect.iter().map(|v| v.abs()).fold(0f32, f32::max).max(1.0);
    let tol: f32 = std::env::var("FIXTURE_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-2);
    assert!(maxerr / scale < tol, "reference mismatch");
    assert!(maxerr2 / scale < 2e-2, "sequential/batched mismatch");
    println!("PASS");
}
