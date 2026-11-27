use std::ffi::c_void;
use std::time::Instant;
use tessera::runtime::GpuContext;

fn half_to_f32(h: u16) -> f32 {
    let s = (h >> 15) & 0x0001;
    let e = (h >> 10) & 0x001f;
    let f = h & 0x03ff;

    if e == 0 {
        if f == 0 {
            return if s == 1 { -0.0 } else { 0.0 };
        }
        return (if s == 1 { -1.0 } else { 1.0 }) * (f as f32 / 1024.0) * 2.0f32.powi(-14);
    } else if e == 31 {
        return if f == 0 {
            if s == 1 {
                f32::NEG_INFINITY
            } else {
                f32::INFINITY
            }
        } else {
            f32::NAN
        };
    }
    (if s == 1 { -1.0 } else { 1.0 }) * (1.0 + f as f32 / 1024.0) * 2.0f32.powi(e as i32 - 15)
}

fn f32_to_half(val: f32) -> u16 {
    let bits = val.to_bits();
    let s = (bits >> 31) & 1;
    let e = ((bits >> 23) & 0xff) as i32;
    let m = bits & 0x7fffff;

    if e == 0xff {
        return ((s << 15) | 0x7c00 | if m != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_e = e - 127 + 15;
    if new_e >= 31 {
        return ((s << 15) | 0x7c00) as u16;
    }
    if new_e <= 0 {
        return (s << 15) as u16;
    }
    let new_m = m >> 13;
    ((s << 15) | ((new_e as u32) << 10) | new_m) as u16
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gpu = GpuContext::init()?;
    println!("GPU: {}", gpu.device_name);

    let ptx = r#"
.version 8.0
.target sm_86
.address_size 64
.visible .entry wmma_gemm_16x16(
    .param .u64 param_a,
    .param .u64 param_b,
    .param .u64 param_c,
    .param .u32 param_m,
    .param .u32 param_n,
    .param .u32 param_k
)
.maxntid 32, 1, 1
{
    .reg .b32 %a<8>;
    .reg .b32 %b<8>;
    .reg .f32 %c<8>;
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<16>;

    ld.param.u64 %rd1, [param_a];
    ld.param.u64 %rd2, [param_b];
    ld.param.u64 %rd3, [param_c];
    ld.param.u32 %r1,  [param_m];
    ld.param.u32 %r2,  [param_n];
    ld.param.u32 %r3,  [param_k];

    mov.f32 %c0, 0f00000000;
    mov.f32 %c1, 0f00000000;
    mov.f32 %c2, 0f00000000;
    mov.f32 %c3, 0f00000000;
    mov.f32 %c4, 0f00000000;
    mov.f32 %c5, 0f00000000;
    mov.f32 %c6, 0f00000000;
    mov.f32 %c7, 0f00000000;

    mov.u32 %r4, %ctaid.y;
    shl.b32 %r5, %r4, 4;
    mov.u32 %r6, %ctaid.x;
    shl.b32 %r7, %r6, 4;

    mov.u32 %r8, 0;

$k_loop:
    setp.ge.u32 %p1, %r8, %r3;
    @%p1 bra $k_done;

    mad.lo.u32 %r9, %r5, %r3, %r8;
    mul.wide.u32 %rd4, %r9, 2;
    add.u64 %rd5, %rd1, %rd4;

    mad.lo.u32 %r10, %r8, %r2, %r7;
    mul.wide.u32 %rd6, %r10, 2;
    add.u64 %rd7, %rd2, %rd6;

    wmma.load.a.sync.aligned.row.m16n16k16.f16 {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, [%rd5], %r3;
    wmma.load.b.sync.aligned.row.m16n16k16.f16 {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, [%rd7], %r2;

    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7};

    add.u32 %r8, %r8, 16;
    bra $k_loop;

$k_done:
    mad.lo.u32 %r11, %r5, %r2, %r7;
    mul.wide.u32 %rd8, %r11, 4;
    add.u64 %rd9, %rd3, %rd8;
    wmma.store.d.sync.aligned.row.m16n16k16.f32 [%rd9], {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, %r2;

    ret;
}
"#;

    let module = gpu.load_ptx(ptx)?;
    let kernel = module.get_kernel("wmma_gemm_16x16")?;

    let m: u32 = 1024;
    let n: u32 = 1024;
    let k: u32 = 1024;
    let total_elements = (m * n) as usize;

    let mut h_a = vec![0u16; total_elements];
    let mut h_b = vec![0u16; total_elements];
    let mut h_c = vec![0.0f32; total_elements];

    for i in 0..total_elements {
        h_a[i] = f32_to_half(((i % 17) as f32) * 0.05);
        h_b[i] = f32_to_half((((i + 3) % 19) as f32) * 0.05);
    }

    let a_bytes = total_elements * 2;
    let b_bytes = total_elements * 2;
    let c_bytes = total_elements * 4;

    let d_a = gpu.alloc(a_bytes)?;
    let d_b = gpu.alloc(b_bytes)?;
    let d_c = gpu.alloc(c_bytes)?;

    d_a.upload(&h_a)?;
    d_b.upload(&h_b)?;

    let grid_dim = (n / 16, m / 16, 1u32);
    let block_dim = (32u32, 1u32, 1u32);

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
    unsafe {
        kernel.launch(grid_dim, block_dim, 0, &mut args)?;
    }
    gpu.sync()?;

    let iters = 50;
    let start = Instant::now();
    for _ in 0..iters {
        unsafe {
            kernel.launch(grid_dim, block_dim, 0, &mut args)?;
        }
    }
    gpu.sync()?;
    let duration = start.elapsed() / iters;

    d_c.download(&mut h_c)?;

    let total_flops = 2.0 * (m as f64) * (n as f64) * (k as f64);
    let tflops = (total_flops / duration.as_secs_f64()) / 1e12;

    println!("Matrix: {}x{}x{}", m, n, k);
    println!("Tensor Core Latency: {:.3?}", duration);
    println!("Tensor Core Throughput: {:.2} TFLOPS", tflops);

    // Verify small CPU reference
    let mut cpu_c_0 = 0.0f32;
    for p in 0..k as usize {
        let val_a = half_to_f32(h_a[p]);
        let val_b = half_to_f32(h_b[p * (n as usize)]);
        cpu_c_0 += val_a * val_b;
    }
    println!(
        "C[0, 0] GPU (sync): {:.4}, CPU: {:.4}, diff: {:.6}",
        h_c[0],
        cpu_c_0,
        (h_c[0] - cpu_c_0).abs()
    );

    let ptx_gen = tessera::backend_ptx::PtxGenerator::new(86);
    let async_cfg = tessera::ir::GemmConfig {
        use_tensor_cores: true,
        ..tessera::ir::GemmConfig::default()
    };
    let async_ptx = ptx_gen.emit_tensor_core_gemm("wmma_async_16x16", &async_cfg);
    let async_mod = gpu.load_ptx(&async_ptx)?;
    let async_kernel = async_mod.get_kernel("wmma_async_16x16")?;

    unsafe {
        async_kernel.launch(grid_dim, block_dim, 0, &mut args)?;
    }
    gpu.sync()?;

    let start_async = Instant::now();
    for _ in 0..iters {
        unsafe {
            async_kernel.launch(grid_dim, block_dim, 0, &mut args)?;
        }
    }
    gpu.sync()?;
    let async_duration = start_async.elapsed() / iters;
    let async_tflops = (total_flops / async_duration.as_secs_f64()) / 1e12;

    d_c.download(&mut h_c)?;
    println!("Async cp.async Latency: {:.3?}", async_duration);
    println!("Async cp.async Throughput: {:.2} TFLOPS", async_tflops);
    println!(
        "C[0, 0] GPU (cp.async): {:.4}, CPU: {:.4}, diff: {:.6}",
        h_c[0],
        cpu_c_0,
        (h_c[0] - cpu_c_0).abs()
    );

    Ok(())
}
