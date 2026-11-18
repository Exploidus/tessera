use std::ffi::c_void;
use std::time::Instant;
use tessera::backend_ptx::PtxGenerator;
use tessera::ir::{Activation, DType, GemmConfig};
use tessera::runtime::GpuContext;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = match GpuContext::init() {
        Ok(ctx) => {
            println!("device: {}", ctx.device_name);
            println!("driver: {}", ctx.driver_version);
            ctx
        }
        Err(e) => {
            eprintln!("failed to init gpu: {}", e);
            return Ok(());
        }
    };

    let m: u32 = 64;
    let n: u32 = 64;
    let k: u32 = 64;
    let total_elements = (m * n) as usize;
    let bytes = total_elements * std::mem::size_of::<f32>();

    let mut h_a = vec![0.0f32; total_elements];
    let mut h_b = vec![0.0f32; total_elements];
    let mut h_c_gpu = vec![0.0f32; total_elements];
    let mut h_c_cpu = vec![0.0f32; total_elements];

    for i in 0..total_elements {
        h_a[i] = (i % 17) as f32 * 0.1f32;
        h_b[i] = ((i + 3) % 19) as f32 * 0.1f32;
    }

    let start_cpu = Instant::now();
    cpu_gemm(m as usize, n as usize, k as usize, &h_a, &h_b, &mut h_c_cpu);
    let cpu_duration = start_cpu.elapsed();

    let kernel_name = "tessera_tiled_gemm_64x64";
    let config = GemmConfig {
        block_m: 64,
        block_n: 64,
        block_k: 16,
        warp_m: 32,
        warp_n: 32,
        num_threads: 128,
        a_dtype: DType::F32,
        b_dtype: DType::F32,
        c_dtype: DType::F32,
        acc_dtype: DType::F32,
        use_swizzle: true,
        use_tensor_cores: false,
        activation: Activation::None,
    };

    let ptx_gen = PtxGenerator::new(86);
    let ptx_code = ptx_gen.emit_tiled_gemm(kernel_name, &config);

    let start_jit = Instant::now();
    let module = gpu.load_ptx(&ptx_code)?;
    let kernel = module.get_kernel(kernel_name)?;
    let jit_duration = start_jit.elapsed();

    let d_a = gpu.alloc(bytes)?;
    let d_b = gpu.alloc(bytes)?;
    let d_c = gpu.alloc(bytes)?;

    d_a.upload(&h_a)?;
    d_b.upload(&h_b)?;

    let tile_dim = 16u32;
    let grid_dim = (n.div_ceil(tile_dim), m.div_ceil(tile_dim), 1u32);
    let block_dim = (tile_dim, tile_dim, 1u32);

    let mut param_a = d_a.dptr;
    let mut param_b = d_b.dptr;
    let mut param_c = d_c.dptr;
    let mut param_m = m;
    let mut param_n = n;
    let mut param_k = k;

    let mut args: [*mut c_void; 6] = [
        &mut param_a as *mut _ as *mut c_void,
        &mut param_b as *mut _ as *mut c_void,
        &mut param_c as *mut _ as *mut c_void,
        &mut param_m as *mut _ as *mut c_void,
        &mut param_n as *mut _ as *mut c_void,
        &mut param_k as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel.launch(grid_dim, block_dim, 0, &mut args)?;
    }
    gpu.sync()?;

    let iterations = 100;
    let start_gpu = Instant::now();
    for _ in 0..iterations {
        unsafe {
            kernel.launch(grid_dim, block_dim, 0, &mut args)?;
        }
    }
    gpu.sync()?;
    let gpu_duration = start_gpu.elapsed() / iterations;

    d_c.download(&mut h_c_gpu)?;

    let mut max_diff = 0.0f32;
    for i in 0..total_elements {
        let diff = (h_c_gpu[i] - h_c_cpu[i]).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = (flops / gpu_duration.as_secs_f64()) / 1e9;

    println!("m={}, n={}, k={}", m, n, k);
    println!("cpu time: {:?}", cpu_duration);
    println!("jit time: {:?}", jit_duration);
    println!("gpu time: {:?}", gpu_duration);
    println!("throughput: {:.2} gflops", gflops);
    println!("max diff: {:.6}", max_diff);

    // 1024x1024 run
    let lm: u32 = 1024;
    let ln: u32 = 1024;
    let lk: u32 = 1024;
    let l_bytes = (lm * ln) as usize * std::mem::size_of::<f32>();

    let d_la = gpu.alloc(l_bytes)?;
    let d_lb = gpu.alloc(l_bytes)?;
    let d_lc = gpu.alloc(l_bytes)?;

    let l_grid = (ln / tile_dim, lm / tile_dim, 1u32);
    let mut lp_a = d_la.dptr;
    let mut lp_b = d_lb.dptr;
    let mut lp_c = d_lc.dptr;
    let mut lm_arg = lm;
    let mut ln_arg = ln;
    let mut lk_arg = lk;

    let mut l_args: [*mut c_void; 6] = [
        &mut lp_a as *mut _ as *mut c_void,
        &mut lp_b as *mut _ as *mut c_void,
        &mut lp_c as *mut _ as *mut c_void,
        &mut lm_arg as *mut _ as *mut c_void,
        &mut ln_arg as *mut _ as *mut c_void,
        &mut lk_arg as *mut _ as *mut c_void,
    ];

    unsafe { kernel.launch(l_grid, block_dim, 0, &mut l_args)?; }
    gpu.sync()?;

    let l_iters = 20;
    let l_start = Instant::now();
    for _ in 0..l_iters {
        unsafe { kernel.launch(l_grid, block_dim, 0, &mut l_args)?; }
    }
    gpu.sync()?;
    let l_duration = l_start.elapsed() / l_iters;
    let l_flops = 2.0 * (lm as f64) * (ln as f64) * (lk as f64);
    let l_tflops = (l_flops / l_duration.as_secs_f64()) / 1e12;

    println!("1024x1024 gpu time: {:?}", l_duration);
    println!("1024x1024 throughput: {:.2} tflops", l_tflops);

    Ok(())
}
