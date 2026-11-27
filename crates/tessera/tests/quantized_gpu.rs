use std::ffi::c_void;
use tessera::backend_ptx::PtxGenerator;
use tessera::ir::{DType, GemmConfig};
use tessera::runtime::GpuContext;

#[test]
fn test_int4_tensor_core_jit_and_launch_gpu() {
    let Ok(gpu) = GpuContext::init() else {
        return;
    };

    let cfg = GemmConfig {
        use_tensor_cores: true,
        a_dtype: DType::I4,
        b_dtype: DType::I4,
        c_dtype: DType::I32,
        acc_dtype: DType::I32,
        ..GemmConfig::default()
    };
    let ptx = PtxGenerator::new(86).emit_tensor_core_gemm("gemm_int4_16x8x32", &cfg);
    let cu_mod = gpu.load_ptx(&ptx).expect("INT4 PTX JIT load failed");
    let kernel = cu_mod
        .get_kernel("gemm_int4_16x8x32")
        .expect("INT4 kernel lookup failed");

    let m = 16u32;
    let n = 8u32;
    let k = 32u32;

    // Pack two 4-bit signed 1s (0x11) per byte -> every element is +1
    let a_packed = vec![0x11u8; (m * k / 2) as usize];
    let b_packed = vec![0x11u8; (n * k / 2) as usize];

    let d_a = gpu.alloc(a_packed.len()).unwrap();
    let d_b = gpu.alloc(b_packed.len()).unwrap();
    let d_c = gpu.alloc((m * n * 4) as usize).unwrap();
    d_a.upload(&a_packed).unwrap();
    d_b.upload(&b_packed).unwrap();

    let mut pa = d_a.dptr;
    let mut pb = d_b.dptr;
    let mut pc = d_c.dptr;
    let mut m_u32 = m;
    let mut n_u32 = n;
    let mut k_u32 = k;

    let mut args: [*mut c_void; 6] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut m_u32 as *mut _ as *mut c_void,
        &mut n_u32 as *mut _ as *mut c_void,
        &mut k_u32 as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel.launch((1, 1, 1), (32, 1, 1), 0, &mut args).unwrap();
    }
    gpu.sync().unwrap();

    let mut out = vec![0i32; (m * n) as usize];
    d_c.download(&mut out).unwrap();

    // Dot product of 32 ones with 32 ones = 32 for every output element
    assert!(
        out.iter().all(|&v| v == 32),
        "Expected all 32s from INT4 m16n8k32 MMA, got {:?}",
        &out[..8]
    );
}
