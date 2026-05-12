use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::Mutex;
use std::time::Instant;

use tessera_backend_ptx::PtxGenerator;
use tessera_ir::{Activation, DType, GemmConfig};

use crate::GpuContext;

#[derive(Debug, Clone)]
pub struct TunedProfile {
    pub config: GemmConfig,
    pub latency_us: f64,
    pub tflops: f64,
}

pub struct Autotuner {
    cache: Mutex<HashMap<(usize, usize, usize), TunedProfile>>,
}

impl Autotuner {
    pub fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }
}

impl Default for Autotuner {
    fn default() -> Self {
        Self::new()
    }
}

impl Autotuner {

    pub fn find_optimal(
        &self,
        gpu: &GpuContext,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<TunedProfile, String> {
        let key = (m, n, k);
        {
            let lock = self.cache.lock().unwrap();
            if let Some(profile) = lock.get(&key) {
                return Ok(profile.clone());
            }
        }

        let candidates = vec![
            GemmConfig {
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
            },
            GemmConfig {
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
            },
        ];

        let mut best_profile = None;
        let ptx_gen = PtxGenerator::new(86);

        for (idx, cfg) in candidates.into_iter().enumerate() {
            let kernel_name = format!("autotune_candidate_{}_{}x{}", idx, m, n);
            let ptx = if cfg.use_tensor_cores {
                ptx_gen.emit_tensor_core_gemm(&kernel_name, &cfg)
            } else {
                ptx_gen.emit_tiled_gemm(&kernel_name, &cfg)
            };

            let module = match gpu.load_ptx(&ptx) {
                Ok(m) => m,
                Err(_) => continue,
            };
            let kernel = match module.get_kernel(&kernel_name) {
                Ok(k) => k,
                Err(_) => continue,
            };

            let elem_bytes = if cfg.use_tensor_cores { 2 } else { 4 };
            let a_bytes = m * k * elem_bytes;
            let b_bytes = k * n * elem_bytes;
            let c_bytes = m * n * 4;

            let d_a = gpu.alloc(a_bytes)?;
            let d_b = gpu.alloc(b_bytes)?;
            let d_c = gpu.alloc(c_bytes)?;

            let grid = if cfg.use_tensor_cores {
                ((n as u32) / 16, (m as u32) / 16, 1u32)
            } else {
                (((n as u32) + 15) / 16, ((m as u32) + 15) / 16, 1u32)
            };

            let block = if cfg.use_tensor_cores {
                (32u32, 1u32, 1u32)
            } else {
                (16u32, 16u32, 1u32)
            };

            let mut p_a = d_a.dptr;
            let mut p_b = d_b.dptr;
            let mut p_c = d_c.dptr;
            let mut p_m = m as u32;
            let mut p_n = n as u32;
            let mut p_k = k as u32;

            let mut args: [*mut c_void; 6] = [
                &mut p_a as *mut _ as *mut c_void,
                &mut p_b as *mut _ as *mut c_void,
                &mut p_c as *mut _ as *mut c_void,
                &mut p_m as *mut _ as *mut c_void,
                &mut p_n as *mut _ as *mut c_void,
                &mut p_k as *mut _ as *mut c_void,
            ];

            // Warm-up
            unsafe { let _ = kernel.launch(grid, block, 0, &mut args); }
            let _ = gpu.sync();

            let iters = 10;
            let start = Instant::now();
            for _ in 0..iters {
                unsafe { let _ = kernel.launch(grid, block, 0, &mut args); }
            }
            let _ = gpu.sync();
            let avg_time = start.elapsed() / iters;

            let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
            let tflops = (total_flops / avg_time.as_secs_f64()) / 1e12;
            let latency_us = avg_time.as_secs_f64() * 1e6;

            let profile = TunedProfile {
                config: cfg,
                latency_us,
                tflops,
            };

            match &best_profile {
                None => best_profile = Some(profile),
                Some(prev) if profile.tflops > prev.tflops => best_profile = Some(profile),
                _ => {}
            }
        }

        let chosen = best_profile.ok_or_else(|| "No valid configuration found".to_string())?;
        let mut lock = self.cache.lock().unwrap();
        lock.insert(key, chosen.clone());
        Ok(chosen)
    }
}
