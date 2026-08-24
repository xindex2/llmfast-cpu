//! GPU backend on wgpu (Metal / Vulkan / DX12). All kernels are ours: see shaders.wgsl.

use std::sync::Arc;
use wgpu::util::DeviceExt;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    matvec_q8: wgpu::ComputePipeline,
    matvec_layout: wgpu::BindGroupLayout,
}

/// Q8 matrix resident on the GPU.
pub struct GQ8 {
    pub q: wgpu::Buffer,
    pub scales: wgpu::Buffer,
    pub n: usize,
    pub k: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct MatParams {
    n: u32,
    k: u32,
    m: u32,
    pad: u32,
}

fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

impl Gpu {
    /// Returns None when no usable GPU adapter exists (the engine then stays on CPU).
    pub fn init() -> Option<Arc<Gpu>> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?;
        let info = adapter.get_info();
        // Software rasterizers (llvmpipe, SwiftShader) show up as adapters on headless servers; they
        // are far slower than our CPU kernels and hit validation limits. Only real GPUs qualify.
        let software = matches!(info.device_type, wgpu::DeviceType::Cpu) || info.name.to_lowercase().contains("llvmpipe") || info.name.to_lowercase().contains("swiftshader");
        if software && std::env::var("DEVICE").map_or(true, |v| v != "gpu") {
            eprintln!("gpu: ignoring software adapter {} ({:?})", info.name, info.device_type);
            return None;
        }
        let limits = adapter.limits();
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("llmfast"),
                required_features: wgpu::Features::empty(),
                required_limits: limits.clone(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )).ok()?;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("llmfast-shaders"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders.wgsl").into()),
        });
        let entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None },
            count: None,
        };
        let matvec_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("matvec"),
            entries: &[entry(0, true), entry(1, true), entry(2, true), entry(3, false), wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }, wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                count: None,
            }],
        });
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: None, bind_group_layouts: &[&matvec_layout], push_constant_ranges: &[] });
        let matvec_q8 = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("matvec_q8"),
            layout: Some(&pl),
            module: &shader,
            entry_point: Some("matvec_q8"),
            compilation_options: Default::default(),
            cache: None,
        });
        eprintln!("gpu: {} ({:?}, {:?}), max buffer {} MB", info.name, info.backend, info.device_type, limits.max_buffer_size >> 20);
        Some(Arc::new(Gpu { device, queue, name: info.name, matvec_q8, matvec_layout }))
    }

    /// Upload a Q8 matrix, re-packed into the 64-row interleaved layout the shader expects
    /// (rows padded up to a multiple of 64).
    pub fn upload_q8(&self, q: &[i8], scales: &[f32], n: usize, k: usize) -> GQ8 {
        let words = k / 4;
        let blocks = k / 32;
        let tiles = (n + 63) / 64;
        let mut qi = vec![0u32; tiles * words * 64];
        let mut si = vec![0f32; tiles * blocks * 64];
        for r in 0..n {
            let (tile, lane) = (r / 64, r % 64);
            let src = &q[r * k..(r + 1) * k];
            for w in 0..words {
                let b = &src[w * 4..w * 4 + 4];
                qi[(tile * words + w) * 64 + lane] = u32::from_le_bytes([b[0] as u8, b[1] as u8, b[2] as u8, b[3] as u8]);
            }
            for b in 0..blocks {
                si[(tile * blocks + b) * 64 + lane] = scales[r * blocks + b];
            }
        }
        let q = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("q8"), contents: as_bytes(&qi), usage: wgpu::BufferUsages::STORAGE });
        let scales = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some("scales"), contents: as_bytes(&si), usage: wgpu::BufferUsages::STORAGE });
        GQ8 { q, scales, n, k }
    }

    pub fn storage(&self, bytes: usize, label: &str) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor { label: Some(label), size: bytes as u64, usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC, mapped_at_creation: false })
    }

    pub fn write_f32(&self, buf: &wgpu::Buffer, data: &[f32]) {
        self.queue.write_buffer(buf, 0, as_bytes(data));
    }

    pub fn read_f32(&self, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor { label: Some("staging"), size: (count * 4) as u64, usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST, mapped_at_creation: false });
        let mut enc = self.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(buf, 0, &staging, 0, (count * 4) as u64);
        self.queue.submit([enc.finish()]);
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { let _ = tx.send(r); });
        self.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let out: Vec<f32> = data.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        drop(data);
        staging.unmap();
        out
    }

    /// Encode y[m×n] = x[m×k] · Wᵀ into `enc`.
    pub fn matvec_q8(&self, enc: &mut wgpu::CommandEncoder, w: &GQ8, x: &wgpu::Buffer, y: &wgpu::Buffer, m: usize) {
        let params = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: as_bytes(&[MatParams { n: w.n as u32, k: w.k as u32, m: m as u32, pad: 0 }]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let count = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("count"),
            contents: as_bytes(&[m as u32, 0u32, 0, 0, 0, 0, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.matvec_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: w.q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: w.scales.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: x.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: y.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 5, resource: count.as_entire_binding() },
            ],
        });
        let mut pass = enc.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.matvec_q8);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(((w.n + 63) / 64) as u32, 1, 1);
    }
}

/// Correctness + bandwidth check of the GPU matvec against the CPU kernel.
pub fn bench(gpu: &Gpu) {
    use crate::kernels::QMat;
    let (n, k) = (3072usize, 1024usize);
    let w: Vec<u16> = (0..n * k).map(|i| ((((i * 2654435761usize) >> 13) % 2001) as f32 / 1000.0 - 1.0).to_bits() as u32 >> 16).map(|b| b as u16).collect();
    let qm = QMat::from_bf16(&w, n, k);
    let x: Vec<f32> = (0..k).map(|i| (i % 17) as f32 * 0.01).collect();
    let mut y_cpu = vec![0f32; n];
    crate::kernels::matvec_q8(&qm, &x, &mut y_cpu);

    let gw = gpu.upload_q8(&qm.q, &qm.scales, n, k);
    let xb = gpu.storage(k * 4, "x");
    let yb = gpu.storage(n * 4, "y");
    gpu.write_f32(&xb, &x);
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    gpu.matvec_q8(&mut enc, &gw, &xb, &yb, 1);
    gpu.queue.submit([enc.finish()]);
    let y_gpu = gpu.read_f32(&yb, n);
    let maxerr = y_cpu.iter().zip(&y_gpu).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    eprintln!("gpu matvec_q8 vs cpu: max abs err {maxerr:.5} (y[0] cpu {:.4} gpu {:.4})", y_cpu[0], y_gpu[0]);

    // throughput: many dispatches in one submission, like a decode step's 7 matvecs x 28 layers
    let iters = 200;
    let t = std::time::Instant::now();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        gpu.matvec_q8(&mut enc, &gw, &xb, &yb, 1);
    }
    let encode_s = t.elapsed().as_secs_f64();
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::Maintain::Wait);
    let _ = gpu.read_f32(&yb, 1);
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("gpu matvec_q8 {n}x{k}: {:.3} ms/dispatch ({:.3} ms of it CPU encoding)  {:.1} GB/s", dt * 1e3, encode_s / iters as f64 * 1e3, qm.bytes() as f64 / dt / 1e9);
    // bigger matrix (more like a real layer stack): 8192 x 4096
    let (n2, k2) = (8192usize, 4096usize);
    let w2: Vec<u16> = (0..n2 * k2).map(|i| ((((i * 2654435761usize) >> 13) % 2001) as f32 / 1000.0 - 1.0).to_bits() as u32 >> 16).map(|b| b as u16).collect();
    let q2 = QMat::from_bf16(&w2, n2, k2);
    let g2 = gpu.upload_q8(&q2.q, &q2.scales, n2, k2);
    let x2b = gpu.storage(k2 * 4, "x2");
    let y2b = gpu.storage(n2 * 4, "y2");
    let iters = 50;
    let t = std::time::Instant::now();
    let mut enc = gpu.device.create_command_encoder(&Default::default());
    for _ in 0..iters {
        gpu.matvec_q8(&mut enc, &g2, &x2b, &y2b, 1);
    }
    gpu.queue.submit([enc.finish()]);
    gpu.device.poll(wgpu::Maintain::Wait);
    let _ = gpu.read_f32(&y2b, 1);
    let dt = t.elapsed().as_secs_f64() / iters as f64;
    eprintln!("gpu matvec_q8 {n2}x{k2}: {:.3} ms/dispatch  {:.1} GB/s", dt * 1e3, q2.bytes() as f64 / dt / 1e9);
}
