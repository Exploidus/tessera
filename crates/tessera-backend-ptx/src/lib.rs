use tessera_ir::{
    Activation, FlashAttentionConfig, GemmConfig, KernelModule, RMSNormConfig, SwiGLUConfig,
};

pub struct PtxGenerator {
    target_sm: u32,
}

impl PtxGenerator {
    pub fn new(target_sm: u32) -> Self {
        Self { target_sm }
    }

    pub fn compile(&self, module: &KernelModule) -> String {
        let config = module.gemm_config.clone().unwrap_or_default();
        if config.use_tensor_cores {
            self.emit_tensor_core_gemm(&module.name, &config)
        } else {
            self.emit_tiled_gemm(&module.name, &config)
        }
    }

    pub fn emit_tensor_core_gemm(&self, kernel_name: &str, config: &GemmConfig) -> String {
        let mut ptx = String::with_capacity(4096);
        ptx.push_str(".version 8.0\n");
        ptx.push_str(&format!(".target sm_{}\n", self.target_sm));
        ptx.push_str(".address_size 64\n\n");

        ptx.push_str(&format!(".visible .entry {}(\n", kernel_name));
        ptx.push_str("    .param .u64 param_a,\n");
        ptx.push_str("    .param .u64 param_b,\n");
        ptx.push_str("    .param .u64 param_c,\n");
        ptx.push_str("    .param .u32 param_m,\n");
        ptx.push_str("    .param .u32 param_n,\n");
        ptx.push_str("    .param .u32 param_k\n");
        ptx.push_str(")\n");
        ptx.push_str(".maxntid 32, 1, 1\n");
        ptx.push_str("{\n");
        ptx.push_str("    .reg .b32 %a<8>;\n");
        ptx.push_str("    .reg .b32 %b<8>;\n");
        ptx.push_str("    .reg .f32 %c<8>;\n");
        ptx.push_str("    .reg .pred %p<8>;\n");
        ptx.push_str("    .reg .b32 %r<16>;\n");
        ptx.push_str("    .reg .b64 %rd<16>;\n");
        ptx.push_str("    .reg .f32 %f<8>;\n\n");

        ptx.push_str("    ld.param.u64 %rd1, [param_a];\n");
        ptx.push_str("    ld.param.u64 %rd2, [param_b];\n");
        ptx.push_str("    ld.param.u64 %rd3, [param_c];\n");
        ptx.push_str("    ld.param.u32 %r1,  [param_m];\n");
        ptx.push_str("    ld.param.u32 %r2,  [param_n];\n");
        ptx.push_str("    ld.param.u32 %r3,  [param_k];\n\n");

        ptx.push_str("    mov.f32 %c0, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c1, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c2, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c3, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c4, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c5, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c6, 0f00000000;\n");
        ptx.push_str("    mov.f32 %c7, 0f00000000;\n\n");

        ptx.push_str("    mov.u32 %r4, %ctaid.y;\n");
        ptx.push_str("    shl.b32 %r5, %r4, 4;\n");
        ptx.push_str("    mov.u32 %r6, %ctaid.x;\n");
        ptx.push_str("    shl.b32 %r7, %r6, 4;\n\n");

        ptx.push_str("    mov.u32 %r8, 0;\n");
        ptx.push_str("$wmma_k_loop:\n");
        ptx.push_str("    setp.ge.u32 %p1, %r8, %r3;\n");
        ptx.push_str("    @%p1 bra $wmma_k_done;\n\n");

        ptx.push_str("    mad.lo.u32 %r9, %r5, %r3, %r8;\n");
        ptx.push_str("    mul.wide.u32 %rd4, %r9, 2;\n");
        ptx.push_str("    add.u64 %rd5, %rd1, %rd4;\n\n");

        ptx.push_str("    mad.lo.u32 %r10, %r8, %r2, %r7;\n");
        ptx.push_str("    mul.wide.u32 %rd6, %r10, 2;\n");
        ptx.push_str("    add.u64 %rd7, %rd2, %rd6;\n\n");

        ptx.push_str("    wmma.load.a.sync.aligned.row.m16n16k16.f16 {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, [%rd5], %r3;\n");
        ptx.push_str("    wmma.load.b.sync.aligned.row.m16n16k16.f16 {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, [%rd7], %r2;\n");
        ptx.push_str("    wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32 {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, {%a0, %a1, %a2, %a3, %a4, %a5, %a6, %a7}, {%b0, %b1, %b2, %b3, %b4, %b5, %b6, %b7}, {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7};\n\n");

        ptx.push_str("    add.u32 %r8, %r8, 16;\n");
        ptx.push_str("    bra $wmma_k_loop;\n");
        ptx.push_str("$wmma_k_done:\n\n");

        if config.activation == Activation::ReLU {
            for i in 0..8 {
                ptx.push_str(&format!("    setp.lt.f32 %p2, %c{}, 0f00000000;\n", i));
                ptx.push_str(&format!("    @%p2 mov.f32 %c{}, 0f00000000;\n", i));
            }
        }

        ptx.push_str("    mad.lo.u32 %r11, %r5, %r2, %r7;\n");
        ptx.push_str("    mul.wide.u32 %rd8, %r11, 4;\n");
        ptx.push_str("    add.u64 %rd9, %rd3, %rd8;\n");
        ptx.push_str("    wmma.store.d.sync.aligned.row.m16n16k16.f32 [%rd9], {%c0, %c1, %c2, %c3, %c4, %c5, %c6, %c7}, %r2;\n\n");

        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");

        ptx
    }

    pub fn emit_tiled_gemm(&self, kernel_name: &str, config: &GemmConfig) -> String {
        let tile_dim = 16;
        let smem_elements = tile_dim * tile_dim;
        let smem_bytes_per_tile = smem_elements * 4;
        let total_smem_bytes = smem_bytes_per_tile * 2;

        let mut ptx = String::with_capacity(4096);
        ptx.push_str(".version 8.0\n");
        ptx.push_str(&format!(".target sm_{}\n", self.target_sm));
        ptx.push_str(".address_size 64\n\n");

        ptx.push_str(&format!(".visible .entry {}(\n", kernel_name));
        ptx.push_str("    .param .u64 param_a,\n");
        ptx.push_str("    .param .u64 param_b,\n");
        ptx.push_str("    .param .u64 param_c,\n");
        ptx.push_str("    .param .u32 param_m,\n");
        ptx.push_str("    .param .u32 param_n,\n");
        ptx.push_str("    .param .u32 param_k\n");
        ptx.push_str(")\n");
        ptx.push_str(".maxntid 16, 16, 1\n");
        ptx.push_str("{\n");

        ptx.push_str(&format!(
            "    .shared .align 16 .b8 smem[{}];\n\n",
            total_smem_bytes
        ));

        ptx.push_str("    .reg .pred %p<16>;\n");
        ptx.push_str("    .reg .b32  %r<48>;\n");
        ptx.push_str("    .reg .b64  %rd<48>;\n");
        ptx.push_str("    .reg .f32  %f<32>;\n\n");

        ptx.push_str("    ld.param.u64 %rd1, [param_a];\n");
        ptx.push_str("    ld.param.u64 %rd2, [param_b];\n");
        ptx.push_str("    ld.param.u64 %rd3, [param_c];\n");
        ptx.push_str("    ld.param.u32 %r1,  [param_m];\n");
        ptx.push_str("    ld.param.u32 %r2,  [param_n];\n");
        ptx.push_str("    ld.param.u32 %r3,  [param_k];\n\n");

        ptx.push_str("    mov.u32 %r4, %tid.x;\n");
        ptx.push_str("    mov.u32 %r5, %tid.y;\n");
        ptx.push_str("    mov.u32 %r6, %ctaid.x;\n");
        ptx.push_str("    mov.u32 %r7, %ctaid.y;\n\n");

        ptx.push_str(&format!("    mad.lo.s32 %r8, %r7, {}, %r5;\n", tile_dim));
        ptx.push_str(&format!("    mad.lo.s32 %r9, %r6, {}, %r4;\n\n", tile_dim));

        ptx.push_str("    mov.f32 %f0, 0.0;\n");
        ptx.push_str("    mov.u64 %rd4, smem;\n\n");

        ptx.push_str("    mov.u32 %r10, 0;\n");
        ptx.push_str("$k_tile_loop:\n");
        ptx.push_str("    setp.ge.u32 %p1, %r10, %r3;\n");
        ptx.push_str("    @%p1 bra $k_tile_done;\n\n");

        ptx.push_str("    add.s32 %r11, %r10, %r4;\n");
        ptx.push_str("    setp.lt.u32 %p2, %r8, %r1;\n");
        ptx.push_str("    setp.lt.u32 %p3, %r11, %r3;\n");
        ptx.push_str("    and.pred %p4, %p2, %p3;\n");
        ptx.push_str("    @!%p4 mov.f32 %f1, 0.0;\n");
        ptx.push_str("    @!%p4 bra $load_a_store_smem;\n");
        ptx.push_str("    mul.lo.u32 %r12, %r8, %r3;\n");
        ptx.push_str("    add.s32 %r12, %r12, %r11;\n");
        ptx.push_str("    mul.wide.u32 %rd5, %r12, 4;\n");
        ptx.push_str("    add.u64 %rd6, %rd1, %rd5;\n");
        ptx.push_str("    ld.global.f32 %f1, [%rd6];\n");
        ptx.push_str("$load_a_store_smem:\n");
        ptx.push_str(&format!("    mad.lo.s32 %r13, %r5, {}, %r4;\n", tile_dim));
        ptx.push_str("    mul.wide.u32 %rd7, %r13, 4;\n");
        ptx.push_str("    add.u64 %rd8, %rd4, %rd7;\n");
        ptx.push_str("    st.shared.f32 [%rd8], %f1;\n\n");

        ptx.push_str("    add.s32 %r14, %r10, %r5;\n");
        ptx.push_str("    setp.lt.u32 %p5, %r14, %r3;\n");
        ptx.push_str("    setp.lt.u32 %p6, %r9, %r2;\n");
        ptx.push_str("    and.pred %p7, %p5, %p6;\n");
        ptx.push_str("    @!%p7 mov.f32 %f2, 0.0;\n");
        ptx.push_str("    @!%p7 bra $load_b_store_smem;\n");
        ptx.push_str("    mul.lo.u32 %r15, %r14, %r2;\n");
        ptx.push_str("    add.s32 %r15, %r15, %r9;\n");
        ptx.push_str("    mul.wide.u32 %rd9, %r15, 4;\n");
        ptx.push_str("    add.u64 %rd10, %rd2, %rd9;\n");
        ptx.push_str("    ld.global.f32 %f2, [%rd10];\n");
        ptx.push_str("$load_b_store_smem:\n");
        ptx.push_str(&format!("    add.s32 %r16, %r13, {};\n", smem_elements));
        ptx.push_str("    mul.wide.u32 %rd11, %r16, 4;\n");
        ptx.push_str("    add.u64 %rd12, %rd4, %rd11;\n");
        ptx.push_str("    st.shared.f32 [%rd12], %f2;\n\n");

        ptx.push_str("    bar.sync 0;\n\n");

        ptx.push_str("    mov.u32 %r17, 0;\n");
        ptx.push_str("$inner_p_loop:\n");
        ptx.push_str(&format!("    setp.ge.u32 %p8, %r17, {};\n", tile_dim));
        ptx.push_str("    @%p8 bra $inner_p_done;\n");
        ptx.push_str(&format!("    mad.lo.s32 %r18, %r5, {}, %r17;\n", tile_dim));
        ptx.push_str("    mul.wide.u32 %rd13, %r18, 4;\n");
        ptx.push_str("    add.u64 %rd14, %rd4, %rd13;\n");
        ptx.push_str("    ld.shared.f32 %f3, [%rd14];\n");
        ptx.push_str(&format!("    mad.lo.s32 %r19, %r17, {}, %r4;\n", tile_dim));
        ptx.push_str(&format!("    add.s32 %r19, %r19, {};\n", smem_elements));
        ptx.push_str("    mul.wide.u32 %rd15, %r19, 4;\n");
        ptx.push_str("    add.u64 %rd16, %rd4, %rd15;\n");
        ptx.push_str("    ld.shared.f32 %f4, [%rd16];\n");
        ptx.push_str("    fma.rn.f32 %f0, %f3, %f4, %f0;\n");
        ptx.push_str("    add.s32 %r17, %r17, 1;\n");
        ptx.push_str("    bra $inner_p_loop;\n");
        ptx.push_str("$inner_p_done:\n\n");

        ptx.push_str("    bar.sync 0;\n");
        ptx.push_str(&format!("    add.s32 %r10, %r10, {};\n", tile_dim));
        ptx.push_str("    bra $k_tile_loop;\n");
        ptx.push_str("$k_tile_done:\n\n");

        match config.activation {
            Activation::None => {}
            Activation::ReLU => {
                ptx.push_str("    setp.lt.f32 %p12, %f0, 0f00000000;\n");
                ptx.push_str("    @%p12 mov.f32 %f0, 0f00000000;\n\n");
            }
            Activation::GELU => {
                ptx.push_str("    mul.f32 %f5, %f0, 0f3FD9D89D;\n");
                ptx.push_str("    neg.f32 %f5, %f5;\n");
                ptx.push_str("    ex2.approx.f32 %f5, %f5;\n");
                ptx.push_str("    add.f32 %f5, %f5, 0f3F800000;\n");
                ptx.push_str("    rcp.approx.f32 %f5, %f5;\n");
                ptx.push_str("    mul.f32 %f0, %f0, %f5;\n\n");
            }
            Activation::Sigmoid => {
                ptx.push_str("    neg.f32 %f5, %f0;\n");
                ptx.push_str("    ex2.approx.f32 %f5, %f5;\n");
                ptx.push_str("    add.f32 %f5, %f5, 0f3F800000;\n");
                ptx.push_str("    rcp.approx.f32 %f0, %f5;\n\n");
            }
            Activation::Swish => {
                ptx.push_str("    neg.f32 %f5, %f0;\n");
                ptx.push_str("    ex2.approx.f32 %f5, %f5;\n");
                ptx.push_str("    add.f32 %f5, %f5, 0f3F800000;\n");
                ptx.push_str("    rcp.approx.f32 %f5, %f5;\n");
                ptx.push_str("    mul.f32 %f0, %f0, %f5;\n\n");
            }
        }

        ptx.push_str("    setp.lt.u32 %p9, %r8, %r1;\n");
        ptx.push_str("    setp.lt.u32 %p10, %r9, %r2;\n");
        ptx.push_str("    and.pred %p11, %p9, %p10;\n");
        ptx.push_str("    @!%p11 bra $exit;\n");
        ptx.push_str("    mul.lo.u32 %r20, %r8, %r2;\n");
        ptx.push_str("    add.s32 %r20, %r20, %r9;\n");
        ptx.push_str("    mul.wide.u32 %rd17, %r20, 4;\n");
        ptx.push_str("    add.u64 %rd18, %rd3, %rd17;\n");
        ptx.push_str("    st.global.f32 [%rd18], %f0;\n\n");

        ptx.push_str("$exit:\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");

        ptx
    }

    pub fn emit_rmsnorm(&self, kernel_name: &str, config: &RMSNormConfig) -> String {
        let mut ptx = String::with_capacity(2048);
        ptx.push_str(".version 8.0\n");
        ptx.push_str(&format!(".target sm_{}\n", self.target_sm));
        ptx.push_str(".address_size 64\n\n");

        ptx.push_str(&format!(".visible .entry {}(\n", kernel_name));
        ptx.push_str("    .param .u64 param_input,\n");
        ptx.push_str("    .param .u64 param_weight,\n");
        ptx.push_str("    .param .u64 param_output,\n");
        ptx.push_str("    .param .u32 param_hidden_dim\n");
        ptx.push_str(")\n");
        ptx.push_str(".maxntid 256, 1, 1\n");
        ptx.push_str("{\n");
        ptx.push_str("    .shared .align 4 .f32 smem_sum[256];\n");
        ptx.push_str("    .reg .pred %p<8>;\n");
        ptx.push_str("    .reg .b32 %r<16>;\n");
        ptx.push_str("    .reg .b64 %rd<16>;\n");
        ptx.push_str("    .reg .f32 %f<16>;\n\n");

        ptx.push_str("    ld.param.u64 %rd1, [param_input];\n");
        ptx.push_str("    ld.param.u64 %rd2, [param_weight];\n");
        ptx.push_str("    ld.param.u64 %rd3, [param_output];\n");
        ptx.push_str("    ld.param.u32 %r1,  [param_hidden_dim];\n\n");

        ptx.push_str("    mov.u32 %r2, %ctaid.x;\n"); // row index
        ptx.push_str("    mov.u32 %r3, %tid.x;\n"); // thread index
        ptx.push_str("    mul.lo.u32 %r4, %r2, %r1;\n"); // row offset

        // Pass 1: compute sum(x^2)
        ptx.push_str("    mov.f32 %f1, 0f00000000;\n");
        ptx.push_str("    mov.u32 %r5, %r3;\n");
        ptx.push_str("$sum_loop:\n");
        ptx.push_str("    setp.ge.u32 %p1, %r5, %r1;\n");
        ptx.push_str("    @%p1 bra $sum_done;\n");
        ptx.push_str("    add.u32 %r6, %r4, %r5;\n");
        ptx.push_str("    mul.wide.u32 %rd4, %r6, 4;\n");
        ptx.push_str("    add.u64 %rd5, %rd1, %rd4;\n");
        ptx.push_str("    ld.global.f32 %f2, [%rd5];\n");
        ptx.push_str("    fma.rn.f32 %f1, %f2, %f2, %f1;\n");
        ptx.push_str("    add.u32 %r5, %r5, 256;\n");
        ptx.push_str("    bra $sum_loop;\n");
        ptx.push_str("$sum_done:\n");

        // Store to shared memory for reduction
        ptx.push_str("    mul.wide.u32 %rd6, %r3, 4;\n");
        ptx.push_str("    mov.u64 %rd7, smem_sum;\n");
        ptx.push_str("    add.u64 %rd8, %rd7, %rd6;\n");
        ptx.push_str("    st.shared.f32 [%rd8], %f1;\n");
        ptx.push_str("    bar.sync 0;\n\n");

        // Reduction in shared memory (stride 128, 64, 32, 16, 8, 4, 2, 1)
        for s in [128, 64, 32, 16, 8, 4, 2, 1] {
            ptx.push_str(&format!("    setp.lt.u32 %p2, %r3, {};\n", s));
            ptx.push_str("    @!%p2 bra $skip_red_");
            ptx.push_str(&format!("{};\n", s));
            ptx.push_str(&format!("    add.u32 %r7, %r3, {};\n", s));
            ptx.push_str("    mul.wide.u32 %rd9, %r7, 4;\n");
            ptx.push_str("    add.u64 %rd10, %rd7, %rd9;\n");
            ptx.push_str("    ld.shared.f32 %f3, [%rd10];\n");
            ptx.push_str("    ld.shared.f32 %f4, [%rd8];\n");
            ptx.push_str("    add.f32 %f4, %f4, %f3;\n");
            ptx.push_str("    st.shared.f32 [%rd8], %f4;\n");
            ptx.push_str(&format!("$skip_red_{}:\n", s));
            ptx.push_str("    bar.sync 0;\n");
        }

        // Thread 0 computes rsqrt(mean + eps)
        ptx.push_str("    ld.shared.f32 %f5, [%rd7];\n");
        ptx.push_str("    cvt.rn.f32.u32 %f6, %r1;\n");
        ptx.push_str("    div.rn.f32 %f5, %f5, %f6;\n");
        ptx.push_str(&format!(
            "    add.f32 %f5, %f5, 0f{:08X};\n",
            config.eps.to_bits()
        ));
        ptx.push_str("    rsqrt.approx.f32 %f5, %f5;\n");
        ptx.push_str("    st.shared.f32 [%rd7], %f5;\n");
        ptx.push_str("    bar.sync 0;\n\n");

        // Broadcast inv_rms to all threads
        ptx.push_str("    ld.shared.f32 %f7, [%rd7];\n");

        // Pass 2: output = (input * inv_rms) * weight
        ptx.push_str("    mov.u32 %r8, %r3;\n");
        ptx.push_str("$norm_loop:\n");
        ptx.push_str("    setp.ge.u32 %p3, %r8, %r1;\n");
        ptx.push_str("    @%p3 bra $norm_done;\n");
        ptx.push_str("    add.u32 %r9, %r4, %r8;\n");
        ptx.push_str("    mul.wide.u32 %rd11, %r9, 4;\n");
        ptx.push_str("    add.u64 %rd12, %rd1, %rd11;\n");
        ptx.push_str("    ld.global.f32 %f8, [%rd12];\n");
        ptx.push_str("    mul.wide.u32 %rd13, %r8, 4;\n");
        ptx.push_str("    add.u64 %rd14, %rd2, %rd13;\n");
        ptx.push_str("    ld.global.f32 %f9, [%rd14];\n");
        ptx.push_str("    mul.f32 %f10, %f8, %f7;\n");
        ptx.push_str("    mul.f32 %f10, %f10, %f9;\n");
        ptx.push_str("    add.u64 %rd15, %rd3, %rd11;\n");
        ptx.push_str("    st.global.f32 [%rd15], %f10;\n");
        ptx.push_str("    add.u32 %r8, %r8, 256;\n");
        ptx.push_str("    bra $norm_loop;\n");
        ptx.push_str("$norm_done:\n");

        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");
        ptx
    }

    pub fn emit_swiglu(&self, kernel_name: &str, _config: &SwiGLUConfig) -> String {
        let mut ptx = String::with_capacity(2048);
        ptx.push_str(".version 8.0\n");
        ptx.push_str(&format!(".target sm_{}\n", self.target_sm));
        ptx.push_str(".address_size 64\n\n");

        ptx.push_str(&format!(".visible .entry {}(\n", kernel_name));
        ptx.push_str("    .param .u64 param_x,\n");
        ptx.push_str("    .param .u64 param_gate,\n");
        ptx.push_str("    .param .u64 param_output,\n");
        ptx.push_str("    .param .u32 param_n\n");
        ptx.push_str(")\n");
        ptx.push_str(".maxntid 256, 1, 1\n");
        ptx.push_str("{\n");
        ptx.push_str("    .reg .pred %p<4>;\n");
        ptx.push_str("    .reg .b32 %r<8>;\n");
        ptx.push_str("    .reg .b64 %rd<8>;\n");
        ptx.push_str("    .reg .f32 %f<8>;\n\n");

        ptx.push_str("    ld.param.u64 %rd1, [param_x];\n");
        ptx.push_str("    ld.param.u64 %rd2, [param_gate];\n");
        ptx.push_str("    ld.param.u64 %rd3, [param_output];\n");
        ptx.push_str("    ld.param.u32 %r1,  [param_n];\n\n");

        ptx.push_str("    mov.u32 %r2, %ctaid.x;\n");
        ptx.push_str("    mov.u32 %r3, %tid.x;\n");
        ptx.push_str("    mad.lo.u32 %r4, %r2, 256, %r3;\n");
        ptx.push_str("    setp.ge.u32 %p1, %r4, %r1;\n");
        ptx.push_str("    @%p1 bra $swiglu_exit;\n\n");

        ptx.push_str("    mul.wide.u32 %rd4, %r4, 4;\n");
        ptx.push_str("    add.u64 %rd5, %rd1, %rd4;\n");
        ptx.push_str("    add.u64 %rd6, %rd2, %rd4;\n");
        ptx.push_str("    ld.global.f32 %f1, [%rd5];\n"); // x
        ptx.push_str("    ld.global.f32 %f2, [%rd6];\n"); // gate

        // swish(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
        // exp(-x) = 2^(-x * log2(e)), log2(e) = 1.44269504 (0f3FB8AA3B)
        ptx.push_str("    neg.f32 %f3, %f1;\n");
        ptx.push_str("    mul.f32 %f3, %f3, 0f3FB8AA3B;\n");
        ptx.push_str("    ex2.approx.f32 %f3, %f3;\n");
        ptx.push_str("    add.f32 %f3, %f3, 0f3F800000;\n");
        ptx.push_str("    rcp.approx.f32 %f3, %f3;\n");
        ptx.push_str("    mul.f32 %f4, %f1, %f3;\n"); // swish_x
        ptx.push_str("    mul.f32 %f5, %f4, %f2;\n"); // swish_x * gate

        ptx.push_str("    add.u64 %rd7, %rd3, %rd4;\n");
        ptx.push_str("    st.global.f32 [%rd7], %f5;\n");

        ptx.push_str("$swiglu_exit:\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");

        ptx
    }

    pub fn emit_flash_attention(&self, kernel_name: &str, config: &FlashAttentionConfig) -> String {
        let mut ptx = String::with_capacity(4096);
        ptx.push_str(".version 8.0\n");
        ptx.push_str(&format!(".target sm_{}\n", self.target_sm));
        ptx.push_str(".address_size 64\n\n");

        ptx.push_str(&format!(".visible .entry {}(\n", kernel_name));
        ptx.push_str("    .param .u64 param_q,\n");
        ptx.push_str("    .param .u64 param_k,\n");
        ptx.push_str("    .param .u64 param_v,\n");
        ptx.push_str("    .param .u64 param_out,\n");
        ptx.push_str("    .param .u32 param_seq_len,\n");
        ptx.push_str("    .param .u32 param_head_dim\n");
        ptx.push_str(")\n");
        ptx.push_str(".maxntid 64, 1, 1\n");
        ptx.push_str("{\n");
        ptx.push_str("    .reg .pred %p<8>;\n");
        ptx.push_str("    .reg .b32 %r<16>;\n");
        ptx.push_str("    .reg .b64 %rd<16>;\n");
        ptx.push_str("    .reg .f32 %f<16>;\n\n");

        ptx.push_str("    ld.param.u64 %rd1, [param_q];\n");
        ptx.push_str("    ld.param.u64 %rd2, [param_k];\n");
        ptx.push_str("    ld.param.u64 %rd3, [param_v];\n");
        ptx.push_str("    ld.param.u64 %rd4, [param_out];\n");
        ptx.push_str("    ld.param.u32 %r1,  [param_seq_len];\n");
        ptx.push_str("    ld.param.u32 %r2,  [param_head_dim];\n\n");

        ptx.push_str("    mov.u32 %r3, %ctaid.x;\n"); // q token index
        ptx.push_str("    mov.u32 %r4, %tid.x;\n"); // dimension index (d < head_dim)
        ptx.push_str("    setp.ge.u32 %p1, %r4, %r2;\n");
        ptx.push_str("    @%p1 bra $fa_exit;\n\n");

        let scale = 1.0f32 / (config.head_dim as f32).sqrt();
        ptx.push_str(&format!("    mov.f32 %f1, 0f{:08X};\n", scale.to_bits()));

        ptx.push_str("    mul.lo.u32 %r5, %r3, %r2;\n");
        ptx.push_str("    add.u32 %r6, %r5, %r4;\n");
        ptx.push_str("    mul.wide.u32 %rd5, %r6, 4;\n");
        ptx.push_str("    add.u64 %rd6, %rd1, %rd5;\n");
        ptx.push_str("    ld.global.f32 %f2, [%rd6];\n"); // q_val

        // Simplified fast projection accumulation for demo
        ptx.push_str("    add.u64 %rd7, %rd3, %rd5;\n");
        ptx.push_str("    ld.global.f32 %f3, [%rd7];\n"); // v_val
        ptx.push_str("    mul.f32 %f4, %f2, %f1;\n");
        ptx.push_str("    mul.f32 %f5, %f4, %f3;\n");

        ptx.push_str("    add.u64 %rd8, %rd4, %rd5;\n");
        ptx.push_str("    st.global.f32 [%rd8], %f5;\n");

        ptx.push_str("$fa_exit:\n");
        ptx.push_str("    ret;\n");
        ptx.push_str("}\n");

        ptx
    }
}
