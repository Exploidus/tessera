use std::ffi::c_void;
use tessera::backend_ptx::PtxGenerator;
use tessera::ir::{FlashAttentionConfig, KernelModule};
use tessera::runtime::GpuContext;

fn cpu_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    is_causal: bool,
) -> Vec<f32> {
    let mut out = vec![0.0f32; num_heads * seq_len * head_dim];
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let head_off = h * seq_len * head_dim;
        for i in 0..seq_len {
            let j_end = if is_causal {
                (i + 1).min(seq_len)
            } else {
                seq_len
            };
            let mut scores = Vec::with_capacity(j_end);
            let mut max_s = f32::NEG_INFINITY;
            for j in 0..j_end {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[head_off + i * head_dim + d] * k[head_off + j * head_dim + d];
                }
                let s = dot * scale;
                if s > max_s {
                    max_s = s;
                }
                scores.push(s);
            }

            let mut sum_exp = 0.0f32;
            let exps: Vec<f32> = scores
                .iter()
                .map(|&s| {
                    let e = (s - max_s).exp();
                    sum_exp += e;
                    e
                })
                .collect();

            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for (j, &e) in exps.iter().enumerate() {
                    acc += e * v[head_off + j * head_dim + d];
                }
                out[head_off + i * head_dim + d] = acc / sum_exp;
            }
        }
    }
    out
}

fn run_flash_attention_case(is_causal: bool) {
    let Ok(gpu) = GpuContext::init() else {
        return;
    };

    let num_heads = 2usize;
    let seq_len = 64usize;
    let head_dim = 64usize;
    let total = num_heads * seq_len * head_dim;

    let q: Vec<f32> = (0..total)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.05)
        .collect();
    let k: Vec<f32> = (0..total)
        .map(|i| (((i * 3) % 31) as f32 - 15.0) * 0.05)
        .collect();
    let v: Vec<f32> = (0..total)
        .map(|i| (((i * 7) % 23) as f32 - 11.0) * 0.05)
        .collect();

    let cpu_ref = cpu_attention(&q, &k, &v, num_heads, seq_len, head_dim, is_causal);

    let cfg = FlashAttentionConfig {
        batch_size: 1,
        num_heads,
        seq_len,
        head_dim,
        is_causal,
    };
    let module = KernelModule::with_flash_attention("flash_attn_v2", cfg);
    let ptx = PtxGenerator::new(86).compile(&module);

    let cu_mod = gpu.load_ptx(&ptx).expect("PTX load failed");
    let kernel = cu_mod
        .get_kernel("flash_attn_v2")
        .expect("kernel lookup failed");

    let d_q = gpu.alloc(total * 4).unwrap();
    let d_k = gpu.alloc(total * 4).unwrap();
    let d_v = gpu.alloc(total * 4).unwrap();
    let d_out = gpu.alloc(total * 4).unwrap();

    d_q.upload(&q).unwrap();
    d_k.upload(&k).unwrap();
    d_v.upload(&v).unwrap();

    let mut p_q = d_q.dptr;
    let mut p_k = d_k.dptr;
    let mut p_v = d_v.dptr;
    let mut p_out = d_out.dptr;
    let mut seq_u32 = seq_len as u32;
    let mut dim_u32 = head_dim as u32;

    let mut args: [*mut c_void; 6] = [
        &mut p_q as *mut _ as *mut c_void,
        &mut p_k as *mut _ as *mut c_void,
        &mut p_v as *mut _ as *mut c_void,
        &mut p_out as *mut _ as *mut c_void,
        &mut seq_u32 as *mut _ as *mut c_void,
        &mut dim_u32 as *mut _ as *mut c_void,
    ];

    unsafe {
        kernel
            .launch(
                (seq_len as u32, num_heads as u32, 1),
                (64, 1, 1),
                0,
                &mut args,
            )
            .unwrap();
    }
    gpu.sync().unwrap();

    let mut gpu_out = vec![0.0f32; total];
    d_out.download(&mut gpu_out).unwrap();

    let max_diff = gpu_out
        .iter()
        .zip(cpu_ref.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_diff < 1e-4,
        "FlashAttention-2 mismatch (is_causal={}): max_diff={}",
        is_causal,
        max_diff
    );
}

#[test]
fn test_flash_attention_2_causal_gpu() {
    run_flash_attention_case(true);
}

#[test]
fn test_flash_attention_2_non_causal_gpu() {
    run_flash_attention_case(false);
}
