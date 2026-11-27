use std::ffi::c_void;
use tessera::backend_ptx::PtxGenerator;
use tessera::ir::{fusion::FusionEngine, Activation, GemmConfig, KernelModule, TileOp};
use tessera::runtime::GpuContext;

fn cpu_gemm_relu(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for r in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for p in 0..k {
                acc += a[r * k + p] * b[p * n + col];
            }
            c[r * n + col] = acc.max(0.0);
        }
    }
    c
}

#[test]
fn test_horizontal_and_vertical_fusion_gpu() {
    let Ok(gpu) = GpuContext::init() else {
        return;
    };

    // Unfused IR: 3 parallel GEMMs each followed by ReLU (6 ops -> 1 FusedHorizontal op)
    let mut module = KernelModule::new("fused_qkv_relu");
    for _ in 0..3 {
        module.ops.push(TileOp::TileMatMul(GemmConfig::default()));
        module.ops.push(TileOp::Elementwise(Activation::ReLU));
    }

    let (optimized, stats) = FusionEngine::optimize(&module);
    assert_eq!(stats.vertical_fusions, 3);
    assert_eq!(stats.horizontal_fusions, 2);
    assert_eq!(stats.fused_ops, 1);

    let ptx = PtxGenerator::new(86).compile(&optimized);
    let cu_mod = gpu.load_ptx(&ptx).expect("PTX load failed");
    let kernel = cu_mod
        .get_kernel("fused_qkv_relu")
        .expect("kernel lookup failed");

    let m = 32usize;
    let n = 32usize;
    let k = 32usize;
    let elems = m * n;

    let mut h_a_list = Vec::new();
    let mut h_b_list = Vec::new();
    let mut d_a_bufs = Vec::new();
    let mut d_b_bufs = Vec::new();
    let mut d_c_bufs = Vec::new();
    let mut refs = Vec::new();

    for batch in 0..3 {
        let a: Vec<f32> = (0..elems)
            .map(|i| (((i + batch * 5) % 19) as f32 - 9.0) * 0.1)
            .collect();
        let b: Vec<f32> = (0..elems)
            .map(|i| (((i * 3 + batch) % 17) as f32 - 8.0) * 0.1)
            .collect();
        refs.push(cpu_gemm_relu(&a, &b, m, n, k));

        let da = gpu.alloc(elems * 4).unwrap();
        let db = gpu.alloc(elems * 4).unwrap();
        let dc = gpu.alloc(elems * 4).unwrap();
        da.upload(&a).unwrap();
        db.upload(&b).unwrap();

        h_a_list.push(da.dptr);
        h_b_list.push(db.dptr);
        d_a_bufs.push(da);
        d_b_bufs.push(db);
        d_c_bufs.push(dc);
    }

    let h_c_list: Vec<u64> = d_c_bufs.iter().map(|b| b.dptr).collect();
    let d_a_ptrs = gpu.alloc(3 * 8).unwrap();
    let d_b_ptrs = gpu.alloc(3 * 8).unwrap();
    let d_c_ptrs = gpu.alloc(3 * 8).unwrap();
    d_a_ptrs.upload(&h_a_list).unwrap();
    d_b_ptrs.upload(&h_b_list).unwrap();
    d_c_ptrs.upload(&h_c_list).unwrap();

    let mut pa = d_a_ptrs.dptr;
    let mut pb = d_b_ptrs.dptr;
    let mut pc = d_c_ptrs.dptr;
    let mut m_u32 = m as u32;
    let mut n_u32 = n as u32;
    let mut k_u32 = k as u32;

    let mut args: [*mut c_void; 6] = [
        &mut pa as *mut _ as *mut c_void,
        &mut pb as *mut _ as *mut c_void,
        &mut pc as *mut _ as *mut c_void,
        &mut m_u32 as *mut _ as *mut c_void,
        &mut n_u32 as *mut _ as *mut c_void,
        &mut k_u32 as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(
                ((n / 16) as u32, (m / 16) as u32, 3),
                (16, 16, 1),
                0,
                &mut args,
            )
            .unwrap();
    }
    gpu.sync().unwrap();

    for (batch, cpu_ref) in refs.iter().enumerate() {
        let mut gpu_out = vec![0.0f32; elems];
        d_c_bufs[batch].download(&mut gpu_out).unwrap();
        let max_diff = gpu_out
            .iter()
            .zip(cpu_ref.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "Fused batch {} mismatch: max_diff={}",
            batch,
            max_diff
        );
    }
}
