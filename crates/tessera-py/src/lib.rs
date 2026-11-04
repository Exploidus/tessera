#![allow(clippy::all)]

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ffi::c_void;
use std::time::Instant;

use tessera_backend_metal::MetalGenerator;
use tessera_backend_ptx::PtxGenerator;
use tessera_backend_spirv::SpirvGenerator;
use tessera_ir::{Activation, DType, GemmConfig, RMSNormConfig, SwiGLUConfig};
use tessera_runtime::autotune::Autotuner;
use tessera_runtime::GpuContext;

#[pyfunction]
fn version() -> &'static str {
    "0.1.0"
}

#[pyfunction]
fn device_info(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let dict = PyDict::new(py);
    dict.set_item("device_name", gpu.device_name.clone())?;
    dict.set_item("driver_version", gpu.driver_version)?;
    dict.set_item("backend", "cuda_driver")?;
    Ok(dict)
}

#[pyfunction]
fn compile_ptx(name: &str, block_m: usize, block_n: usize, block_k: usize) -> PyResult<String> {
    let config = GemmConfig {
        block_m,
        block_n,
        block_k,
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
    let gen = PtxGenerator::new(86);
    Ok(gen.emit_tiled_gemm(name, &config))
}

#[pyfunction]
fn compile_vulkan(name: &str) -> PyResult<String> {
    let gen = SpirvGenerator::new();
    let config = GemmConfig::default();
    Ok(gen.emit_vulkan_compute_shader(name, &config))
}

#[pyfunction]
fn compile_metal(name: &str) -> PyResult<String> {
    let gen = MetalGenerator::new();
    let config = GemmConfig::default();
    Ok(gen.emit_metal_gemm(name, &config))
}

#[pyfunction]
#[pyo3(signature = (a, b, m, n, k, activation=None))]
fn matmul(
    _py: Python<'_>,
    a: Vec<f32>,
    b: Vec<f32>,
    m: u32,
    n: u32,
    k: u32,
    activation: Option<&str>,
) -> PyResult<Vec<f32>> {
    if a.len() != (m * k) as usize {
        return Err(PyRuntimeError::new_err("Matrix A size mismatch"));
    }
    if b.len() != (k * n) as usize {
        return Err(PyRuntimeError::new_err("Matrix B size mismatch"));
    }

    let act = match activation.unwrap_or("none").to_lowercase().as_str() {
        "relu" => Activation::ReLU,
        "gelu" => Activation::GELU,
        "sigmoid" => Activation::Sigmoid,
        "swish" => Activation::Swish,
        _ => Activation::None,
    };

    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_py_gemm";
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
        activation: act,
    };

    let ptx_gen = PtxGenerator::new(86);
    let ptx_code = ptx_gen.emit_tiled_gemm(kernel_name, &config);
    let module = gpu.load_ptx(&ptx_code).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let a_bytes = a.len() * 4;
    let b_bytes = b.len() * 4;
    let c_elements = (m * n) as usize;
    let c_bytes = c_elements * 4;

    let d_a = gpu.alloc(a_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_b = gpu.alloc(b_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_c = gpu.alloc(c_bytes).map_err(|e| PyRuntimeError::new_err(e))?;

    d_a.upload(&a).map_err(|e| PyRuntimeError::new_err(e))?;
    d_b.upload(&b).map_err(|e| PyRuntimeError::new_err(e))?;

    let tile_dim = 16u32;
    let grid_dim = ((n + tile_dim - 1) / tile_dim, (m + tile_dim - 1) / tile_dim, 1u32);
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
        kernel
            .launch(grid_dim, block_dim, 0, &mut args)
            .map_err(|e| PyRuntimeError::new_err(e))?;
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let mut c_host = vec![0.0f32; c_elements];
    d_c.download(&mut c_host).map_err(|e| PyRuntimeError::new_err(e))?;

    Ok(c_host)
}

#[pyfunction]
fn matmul_tensor_core(
    _py: Python<'_>,
    a_u16: Vec<u16>,
    b_u16: Vec<u16>,
    m: u32,
    n: u32,
    k: u32,
) -> PyResult<Vec<f32>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_wmma_gemm";
    let config = GemmConfig {
        block_m: 16,
        block_n: 16,
        block_k: 16,
        warp_m: 16,
        warp_n: 16,
        num_threads: 32,
        a_dtype: DType::F16,
        b_dtype: DType::F16,
        c_dtype: DType::F32,
        acc_dtype: DType::F32,
        use_swizzle: false,
        use_tensor_cores: true,
        activation: Activation::None,
    };

    let ptx_gen = PtxGenerator::new(86);
    let ptx_code = ptx_gen.emit_tensor_core_gemm(kernel_name, &config);
    let module = gpu.load_ptx(&ptx_code).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let c_elements = (m * n) as usize;
    let d_a = gpu.alloc(a_u16.len() * 2).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_b = gpu.alloc(b_u16.len() * 2).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_c = gpu.alloc(c_elements * 4).map_err(|e| PyRuntimeError::new_err(e))?;

    d_a.upload(&a_u16).map_err(|e| PyRuntimeError::new_err(e))?;
    d_b.upload(&b_u16).map_err(|e| PyRuntimeError::new_err(e))?;

    let grid_dim = (n / 16, m / 16, 1u32);
    let block_dim = (32u32, 1u32, 1u32);

    let mut p_a = d_a.dptr;
    let mut p_b = d_b.dptr;
    let mut p_c = d_c.dptr;
    let mut p_m = m;
    let mut p_n = n;
    let mut p_k = k;

    let mut args: [*mut c_void; 6] = [
        &mut p_a as *mut _ as *mut c_void,
        &mut p_b as *mut _ as *mut c_void,
        &mut p_c as *mut _ as *mut c_void,
        &mut p_m as *mut _ as *mut c_void,
        &mut p_n as *mut _ as *mut c_void,
        &mut p_k as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(grid_dim, block_dim, 0, &mut args)
            .map_err(|e| PyRuntimeError::new_err(e))?;
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let mut c_host = vec![0.0f32; c_elements];
    d_c.download(&mut c_host).map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(c_host)
}

#[pyfunction]
fn rmsnorm(
    _py: Python<'_>,
    input: Vec<f32>,
    weight: Vec<f32>,
    num_rows: u32,
    hidden_dim: u32,
    eps: f32,
) -> PyResult<Vec<f32>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_rmsnorm";
    let config = RMSNormConfig {
        num_rows: num_rows as usize,
        hidden_dim: hidden_dim as usize,
        eps,
    };

    let ptx_gen = PtxGenerator::new(86);
    let ptx = ptx_gen.emit_rmsnorm(kernel_name, &config);
    let module = gpu.load_ptx(&ptx).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let total = (num_rows * hidden_dim) as usize;
    let d_in = gpu.alloc(total * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_w = gpu.alloc((hidden_dim as usize) * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_out = gpu.alloc(total * 4).map_err(|e| PyRuntimeError::new_err(e))?;

    d_in.upload(&input).map_err(|e| PyRuntimeError::new_err(e))?;
    d_w.upload(&weight).map_err(|e| PyRuntimeError::new_err(e))?;

    let grid = (num_rows, 1u32, 1u32);
    let block = (256u32, 1u32, 1u32);

    let mut p_in = d_in.dptr;
    let mut p_w = d_w.dptr;
    let mut p_out = d_out.dptr;
    let mut p_dim = hidden_dim;

    let mut args: [*mut c_void; 4] = [
        &mut p_in as *mut _ as *mut c_void,
        &mut p_w as *mut _ as *mut c_void,
        &mut p_out as *mut _ as *mut c_void,
        &mut p_dim as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?;
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let mut out_host = vec![0.0f32; total];
    d_out.download(&mut out_host).map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out_host)
}

#[pyfunction]
fn swiglu(_py: Python<'_>, x: Vec<f32>, gate: Vec<f32>) -> PyResult<Vec<f32>> {
    if x.len() != gate.len() {
        return Err(PyRuntimeError::new_err("Size mismatch between x and gate"));
    }
    let n = x.len() as u32;
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_swiglu";
    let config = SwiGLUConfig {
        num_elements: n as usize,
    };

    let ptx_gen = PtxGenerator::new(86);
    let ptx = ptx_gen.emit_swiglu(kernel_name, &config);
    let module = gpu.load_ptx(&ptx).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let d_x = gpu.alloc(x.len() * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_g = gpu.alloc(gate.len() * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_out = gpu.alloc(x.len() * 4).map_err(|e| PyRuntimeError::new_err(e))?;

    d_x.upload(&x).map_err(|e| PyRuntimeError::new_err(e))?;
    d_g.upload(&gate).map_err(|e| PyRuntimeError::new_err(e))?;

    let grid = ((n + 255) / 256, 1u32, 1u32);
    let block = (256u32, 1u32, 1u32);

    let mut p_x = d_x.dptr;
    let mut p_g = d_g.dptr;
    let mut p_out = d_out.dptr;
    let mut p_n = n;

    let mut args: [*mut c_void; 4] = [
        &mut p_x as *mut _ as *mut c_void,
        &mut p_g as *mut _ as *mut c_void,
        &mut p_out as *mut _ as *mut c_void,
        &mut p_n as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel.launch(grid, block, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?;
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let mut out_host = vec![0.0f32; x.len()];
    d_out.download(&mut out_host).map_err(|e| PyRuntimeError::new_err(e))?;
    Ok(out_host)
}

#[pyfunction]
fn autotune_gemm(py: Python<'_>, m: usize, n: usize, k: usize) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let tuner = Autotuner::new();
    let profile = tuner.find_optimal(&gpu, m, n, k).map_err(|e| PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("use_tensor_cores", profile.config.use_tensor_cores)?;
    dict.set_item("block_m", profile.config.block_m)?;
    dict.set_item("block_n", profile.config.block_n)?;
    dict.set_item("latency_us", profile.latency_us)?;
    dict.set_item("tflops", profile.tflops)?;
    Ok(dict)
}

#[pyfunction]
fn bench_gemm(py: Python<'_>, m: u32, n: u32, k: u32, iterations: u32) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_bench_gemm";
    let config = GemmConfig::default();
    let ptx_gen = PtxGenerator::new(86);
    let ptx_code = ptx_gen.emit_tiled_gemm(kernel_name, &config);
    let module = gpu.load_ptx(&ptx_code).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let a_bytes = (m * k) as usize * 4;
    let b_bytes = (k * n) as usize * 4;
    let c_bytes = (m * n) as usize * 4;

    let d_a = gpu.alloc(a_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_b = gpu.alloc(b_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_c = gpu.alloc(c_bytes).map_err(|e| PyRuntimeError::new_err(e))?;

    let tile_dim = 16u32;
    let grid_dim = ((n + tile_dim - 1) / tile_dim, (m + tile_dim - 1) / tile_dim, 1u32);
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

    unsafe { kernel.launch(grid_dim, block_dim, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?; }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { kernel.launch(grid_dim, block_dim, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?; }
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;
    let avg_duration = start.elapsed() / iterations;

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let gflops = (flops / avg_duration.as_secs_f64()) / 1e9;

    let dict = PyDict::new(py);
    dict.set_item("device", gpu.device_name.clone())?;
    dict.set_item("latency_us", avg_duration.as_micros())?;
    dict.set_item("gflops", gflops)?;
    dict.set_item("m", m)?;
    dict.set_item("n", n)?;
    dict.set_item("k", k)?;
    Ok(dict)
}

#[pyfunction]
fn bench_tensor_core(py: Python<'_>, m: u32, n: u32, k: u32, iterations: u32) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_bench_wmma";
    let config = GemmConfig {
        block_m: 16,
        block_n: 16,
        block_k: 16,
        warp_m: 16,
        warp_n: 16,
        num_threads: 32,
        a_dtype: DType::F16,
        b_dtype: DType::F16,
        c_dtype: DType::F32,
        acc_dtype: DType::F32,
        use_swizzle: false,
        use_tensor_cores: true,
        activation: Activation::None,
    };
    let ptx_gen = PtxGenerator::new(86);
    let ptx_code = ptx_gen.emit_tensor_core_gemm(kernel_name, &config);
    let module = gpu.load_ptx(&ptx_code).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let a_bytes = (m * k) as usize * 2;
    let b_bytes = (k * n) as usize * 2;
    let c_bytes = (m * n) as usize * 4;

    let d_a = gpu.alloc(a_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_b = gpu.alloc(b_bytes).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_c = gpu.alloc(c_bytes).map_err(|e| PyRuntimeError::new_err(e))?;

    let grid_dim = (n / 16, m / 16, 1u32);
    let block_dim = (32u32, 1u32, 1u32);

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

    unsafe { kernel.launch(grid_dim, block_dim, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?; }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let start = Instant::now();
    for _ in 0..iterations {
        unsafe { kernel.launch(grid_dim, block_dim, 0, &mut args).map_err(|e| PyRuntimeError::new_err(e))?; }
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;
    let avg_duration = start.elapsed() / iterations;

    let flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = (flops / avg_duration.as_secs_f64()) / 1e12;

    let dict = PyDict::new(py);
    dict.set_item("device", gpu.device_name.clone())?;
    dict.set_item("latency_us", avg_duration.as_micros())?;
    dict.set_item("tflops", tflops)?;
    dict.set_item("m", m)?;
    dict.set_item("n", n)?;
    dict.set_item("k", k)?;
    Ok(dict)
}

#[pyfunction]
fn matmul_arbitrary(
    _py: Python<'_>,
    a: Vec<f32>,
    b: Vec<f32>,
    m: usize,
    n: usize,
    k: usize,
) -> PyResult<Vec<f32>> {
    if a.len() != m * k || b.len() != k * n {
        return Err(PyRuntimeError::new_err("Matrix shape mismatch for arbitrary GEMM"));
    }

    let (padded_a, pad_m, pad_k) = tessera_runtime::pad_matrix_2d(&a, m, k, 16);
    let (padded_b, _, pad_n) = tessera_runtime::pad_matrix_2d(&b, k, n, 16);

    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel_name = "tessera_arbitrary_gemm";
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
    let module = gpu.load_ptx(&ptx_code).map_err(|e| PyRuntimeError::new_err(e))?;
    let kernel = module.get_kernel(kernel_name).map_err(|e| PyRuntimeError::new_err(e))?;

    let d_a = gpu.alloc(padded_a.len() * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_b = gpu.alloc(padded_b.len() * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_c = gpu.alloc(pad_m * pad_n * 4).map_err(|e| PyRuntimeError::new_err(e))?;

    d_a.upload(&padded_a).map_err(|e| PyRuntimeError::new_err(e))?;
    d_b.upload(&padded_b).map_err(|e| PyRuntimeError::new_err(e))?;

    let grid_dim = (pad_n as u32 / 16, pad_m as u32 / 16, 1u32);
    let block_dim = (16u32, 16u32, 1u32);

    let mut param_a = d_a.dptr;
    let mut param_b = d_b.dptr;
    let mut param_c = d_c.dptr;
    let mut param_m = pad_m as u32;
    let mut param_n = pad_n as u32;
    let mut param_k = pad_k as u32;

    let mut args: [*mut c_void; 6] = [
        &mut param_a as *mut _ as *mut c_void,
        &mut param_b as *mut _ as *mut c_void,
        &mut param_c as *mut _ as *mut c_void,
        &mut param_m as *mut _ as *mut c_void,
        &mut param_n as *mut _ as *mut c_void,
        &mut param_k as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(grid_dim, block_dim, 0, &mut args)
            .map_err(|e| PyRuntimeError::new_err(e))?;
    }
    gpu.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let mut padded_c = vec![0.0f32; pad_m * pad_n];
    d_c.download(&mut padded_c).map_err(|e| PyRuntimeError::new_err(e))?;

    let unpadded_c = tessera_runtime::unpad_matrix_2d(&padded_c, pad_n, m, n);
    Ok(unpadded_c)
}

#[pyfunction]
fn test_async_streams(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let stream1 = gpu.create_stream(true).map_err(|e| PyRuntimeError::new_err(e))?;
    let stream2 = gpu.create_stream(true).map_err(|e| PyRuntimeError::new_err(e))?;
    let event_start = gpu.create_event().map_err(|e| PyRuntimeError::new_err(e))?;
    let event_end = gpu.create_event().map_err(|e| PyRuntimeError::new_err(e))?;

    let n = 1024 * 1024;
    let d_buf1 = gpu.alloc(n * 4).map_err(|e| PyRuntimeError::new_err(e))?;
    let d_buf2 = gpu.alloc(n * 4).map_err(|e| PyRuntimeError::new_err(e))?;

    let host_data = vec![1.0f32; n];

    event_start.record(&stream1).map_err(|e| PyRuntimeError::new_err(e))?;

    // Async concurrent transfers on two independent streams
    stream1.upload_async(&d_buf1, &host_data).map_err(|e| PyRuntimeError::new_err(e))?;
    stream2.upload_async(&d_buf2, &host_data).map_err(|e| PyRuntimeError::new_err(e))?;

    event_end.record(&stream1).map_err(|e| PyRuntimeError::new_err(e))?;

    stream1.sync().map_err(|e| PyRuntimeError::new_err(e))?;
    stream2.sync().map_err(|e| PyRuntimeError::new_err(e))?;
    event_end.sync().map_err(|e| PyRuntimeError::new_err(e))?;

    let elapsed_ms = event_start.elapsed_time(&event_end).map_err(|e| PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("streams_created", 2)?;
    dict.set_item("concurrency", "non_blocking_async")?;
    dict.set_item("elapsed_ms", elapsed_ms)?;
    Ok(dict)
}

#[pyfunction]
fn multi_gpu_topology(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let gpu = GpuContext::init().map_err(|e| PyRuntimeError::new_err(e))?;
    let topo = gpu.get_topology().map_err(|e| PyRuntimeError::new_err(e))?;

    let dict = PyDict::new(py);
    dict.set_item("device_count", topo.devices.len())?;
    let dev_list = pyo3::types::PyList::empty(py);
    for dev in &topo.devices {
        let dev_dict = PyDict::new(py);
        dev_dict.set_item("ordinal", dev.ordinal)?;
        dev_dict.set_item("name", dev.name.clone())?;
        dev_list.append(dev_dict)?;
    }
    dict.set_item("devices", dev_list)?;
    Ok(dict)
}

#[pyfunction]
fn collective_all_reduce(_py: Python<'_>, rank_data: Vec<Vec<f32>>) -> PyResult<Vec<Vec<f32>>> {
    if rank_data.is_empty() {
        return Ok(Vec::new());
    }
    let mut copy = rank_data;
    let mut slice_refs: Vec<&mut [f32]> = copy.iter_mut().map(|v| v.as_mut_slice()).collect();
    tessera_runtime::CollectiveComm::all_reduce_sum(&mut slice_refs);
    Ok(copy)
}

#[pymodule]
fn tessera(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(device_info, m)?)?;
    m.add_function(wrap_pyfunction!(compile_ptx, m)?)?;
    m.add_function(wrap_pyfunction!(compile_vulkan, m)?)?;
    m.add_function(wrap_pyfunction!(compile_metal, m)?)?;
    m.add_function(wrap_pyfunction!(matmul, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_arbitrary, m)?)?;
    m.add_function(wrap_pyfunction!(matmul_tensor_core, m)?)?;
    m.add_function(wrap_pyfunction!(rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(swiglu, m)?)?;
    m.add_function(wrap_pyfunction!(autotune_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(bench_gemm, m)?)?;
    m.add_function(wrap_pyfunction!(bench_tensor_core, m)?)?;
    m.add_function(wrap_pyfunction!(test_async_streams, m)?)?;
    m.add_function(wrap_pyfunction!(multi_gpu_topology, m)?)?;
    m.add_function(wrap_pyfunction!(collective_all_reduce, m)?)?;
    Ok(())
}
