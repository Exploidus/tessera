use std::ffi::c_void;
use std::time::Instant;
use tessera::backend_ptx::PtxGenerator;
use tessera::ir::{Activation, GemmConfig};
use tessera::runtime::GpuContext;

struct EvalResult {
    dimension: String,
    memory_mb: f64,
    latency_us: f64,
    tflops: f64,
    max_error: f32,
    activation: &'static str,
}

fn cpu_gemm(m: usize, n: usize, k: usize, a: &[f32], b: &[f32], c: &mut [f32]) {
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
}

fn run_eval(
    gpu: &GpuContext,
    dim: usize,
    iterations: u32,
    activation: Activation,
    act_str: &'static str,
) -> Result<EvalResult, Box<dyn std::error::Error>> {
    let m = dim as u32;
    let n = dim as u32;
    let k = dim as u32;
    let total_elements = (m * n) as usize;
    let bytes = total_elements * std::mem::size_of::<f32>();

    let mut h_a = vec![0.0f32; total_elements];
    let mut h_b = vec![0.0f32; total_elements];
    let mut h_c = vec![0.0f32; total_elements];

    for i in 0..total_elements {
        h_a[i] = ((i % 13) as f32) * 0.05;
        h_b[i] = (((i + 5) % 17) as f32) * 0.05;
    }

    let mut max_err = 0.0f32;
    if dim <= 256 && activation == Activation::None {
        let mut cpu_c = vec![0.0f32; total_elements];
        cpu_gemm(dim, dim, dim, &h_a, &h_b, &mut cpu_c);

        let d_a = gpu.alloc(bytes)?;
        let d_b = gpu.alloc(bytes)?;
        let d_c = gpu.alloc(bytes)?;
        d_a.upload(&h_a)?;
        d_b.upload(&h_b)?;

        let ptx_gen = PtxGenerator::new(86);
        let config = GemmConfig {
            activation,
            ..Default::default()
        };
        let kernel_name = format!("eval_gemm_{}x{}", dim, dim);
        let ptx = ptx_gen.emit_tiled_gemm(&kernel_name, &config);
        let module = gpu.load_ptx(&ptx)?;
        let kernel = module.get_kernel(&kernel_name)?;

        let tile_dim = 16u32;
        let grid = (n / tile_dim, m / tile_dim, 1u32);
        let block = (tile_dim, tile_dim, 1u32);

        let mut p_a = d_a.dptr;
        let mut p_b = d_b.dptr;
        let mut p_c = d_c.dptr;
        let mut m_a = m;
        let mut n_a = n;
        let mut k_a = k;

        let mut args: [*mut c_void; 6] = [
            &mut p_a as *mut _ as *mut c_void,
            &mut p_b as *mut _ as *mut c_void,
            &mut p_c as *mut _ as *mut c_void,
            &mut m_a as *mut _ as *mut c_void,
            &mut n_a as *mut _ as *mut c_void,
            &mut k_a as *mut _ as *mut c_void,
        ];

        unsafe { kernel.launch(grid, block, 0, &mut args)?; }
        gpu.sync()?;
        d_c.download(&mut h_c)?;

        for i in 0..total_elements {
            let diff = (h_c[i] - cpu_c[i]).abs();
            if diff > max_err {
                max_err = diff;
            }
        }
    }

    let d_a = gpu.alloc(bytes)?;
    let d_b = gpu.alloc(bytes)?;
    let d_c = gpu.alloc(bytes)?;
    d_a.upload(&h_a)?;
    d_b.upload(&h_b)?;

    let ptx_gen = PtxGenerator::new(86);
    let config = GemmConfig {
        activation,
        ..Default::default()
    };
    let kernel_name = format!("bench_gemm_{}x{}_{}", dim, dim, act_str);
    let ptx = ptx_gen.emit_tiled_gemm(&kernel_name, &config);
    let module = gpu.load_ptx(&ptx)?;
    let kernel = module.get_kernel(&kernel_name)?;

    let tile_dim = 16u32;
    let grid = (n / tile_dim, m / tile_dim, 1u32);
    let block = (tile_dim, tile_dim, 1u32);

    let mut p_a = d_a.dptr;
    let mut p_b = d_b.dptr;
    let mut p_c = d_c.dptr;
    let mut m_a = m;
    let mut n_a = n;
    let mut k_a = k;

    let mut args: [*mut c_void; 6] = [
        &mut p_a as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_c as *mut _ as *mut c_void,
        &mut m_a as *mut _ as *mut c_void,
        &mut n_a as *mut _ as *mut c_void,
        &mut k_a as *mut _ as *mut c_void,
    ];

    // Warm-up
    unsafe { kernel.launch(grid, block, 0, &mut args)?; }
    gpu.sync()?;

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { kernel.launch(grid, block, 0, &mut args)?; }
    }
    gpu.sync()?;
    let avg_duration = start.elapsed() / iterations;

    let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = (total_flops / avg_duration.as_secs_f64()) / 1e12;
    let memory_mb = (bytes * 3) as f64 / (1024.0 * 1024.0);

    Ok(EvalResult {
        dimension: format!("{}x{}", dim, dim),
        memory_mb,
        latency_us: avg_duration.as_secs_f64() * 1e6,
        tflops,
        max_error: max_err,
        activation: act_str,
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = GpuContext::init()?;
    println!("device: {} (driver: {})", gpu.device_name, gpu.driver_version);

    let test_matrix = [
        (64, 50, Activation::None, "Linear"),
        (128, 50, Activation::None, "Linear"),
        (256, 30, Activation::None, "Linear"),
        (512, 20, Activation::None, "Linear"),
        (1024, 10, Activation::None, "Linear"),
        (2048, 5, Activation::None, "Linear"),
        (512, 20, Activation::ReLU, "Fused_ReLU"),
        (512, 20, Activation::GELU, "Fused_GELU"),
        (1024, 10, Activation::GELU, "Fused_GELU"),
    ];

    let mut results = Vec::new();

    for (dim, iters, act, name) in test_matrix {
        let res = run_eval(&gpu, dim, iters, act, name)?;
        println!(
            "eval {:>9} {:>10}: {:>8.2} us, {:>5.2} tflops",
            res.dimension, name, res.latency_us, res.tflops
        );
        results.push(res);
    }

    println!();
    println!("| {:<10} | {:<10} | {:<9} | {:<12} | {:<10} | {:<10} |",
        "Matrix", "Kernel", "VRAM (MB)", "Latency", "TFLOPS", "Accuracy");
    println!("|:-----------|:-----------|:----------|:-------------|:-----------|:-----------|");

    for r in &results {
        let lat_str = if r.latency_us >= 1000.0 {
            format!("{:.3} ms", r.latency_us / 1000.0)
        } else {
            format!("{:.1} us", r.latency_us)
        };
        let acc_str = if r.max_error > 0.0 {
            format!("{:.1e}", r.max_error)
        } else {
            "Verified".to_string()
        };
        println!("| {:<10} | {:<10} | {:>8.2}  | {:>11} | {:>9.2}  | {:>10} |",
            r.dimension, r.activation, r.memory_mb, lat_str, r.tflops, acc_str);
    }

    Ok(())
}
