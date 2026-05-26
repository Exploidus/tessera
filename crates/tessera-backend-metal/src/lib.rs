use tessera_ir::{Activation, GemmConfig, KernelModule, RMSNormConfig};

pub struct MetalGenerator {
    tile_dim: usize,
}

impl MetalGenerator {
    pub fn new() -> Self {
        Self { tile_dim: 16 }
    }
}

impl Default for MetalGenerator {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalGenerator {

    pub fn compile(&self, module: &KernelModule) -> String {
        let config = module.gemm_config.clone().unwrap_or_default();
        self.emit_metal_gemm(&module.name, &config)
    }

    pub fn emit_metal_gemm(&self, kernel_name: &str, config: &GemmConfig) -> String {
        let td = self.tile_dim;
        let mut msl = String::with_capacity(2048);

        msl.push_str("#include <metal_stdlib>\n");
        msl.push_str("using namespace metal;\n\n");

        msl.push_str(&format!(
            "kernel void {}(\n\
            device const float* A [[buffer(0)]],\n\
            device const float* B [[buffer(1)]],\n\
            device float* C [[buffer(2)]],\n\
            constant uint3& dims [[buffer(3)]],\n\
            uint2 thread_position_in_grid [[thread_position_in_grid]],\n\
            uint2 thread_position_in_threadgroup [[thread_position_in_threadgroup]],\n\
            uint2 threadgroup_position_in_grid [[threadgroup_position_in_grid]])\n\
            {{\n",
            kernel_name
        ));

        msl.push_str(&format!("    threadgroup float s_a[{}][{}];\n", td, td));
        msl.push_str(&format!("    threadgroup float s_b[{}][{}];\n\n", td, td));

        msl.push_str("    uint m = dims.x;\n");
        msl.push_str("    uint n = dims.y;\n");
        msl.push_str("    uint k = dims.z;\n\n");

        msl.push_str("    uint tx = thread_position_in_threadgroup.x;\n");
        msl.push_str("    uint ty = thread_position_in_threadgroup.y;\n");
        msl.push_str(&format!("    uint global_row = threadgroup_position_in_grid.y * {} + ty;\n", td));
        msl.push_str(&format!("    uint global_col = threadgroup_position_in_grid.x * {} + tx;\n\n", td));

        msl.push_str("    float acc = 0.0f;\n\n");

        msl.push_str(&format!("    for (uint tile_k = 0; tile_k < k; tile_k += {}) {{\n", td));
        msl.push_str("        if (global_row < m && (tile_k + tx) < k) {\n");
        msl.push_str("            s_a[ty][tx] = A[global_row * k + (tile_k + tx)];\n");
        msl.push_str("        } else {\n");
        msl.push_str("            s_a[ty][tx] = 0.0f;\n");
        msl.push_str("        }\n\n");

        msl.push_str("        if ((tile_k + ty) < k && global_col < n) {\n");
        msl.push_str("            s_b[ty][tx] = B[(tile_k + ty) * n + global_col];\n");
        msl.push_str("        } else {\n");
        msl.push_str("            s_b[ty][tx] = 0.0f;\n");
        msl.push_str("        }\n\n");

        msl.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");

        msl.push_str(&format!("        for (uint p = 0; p < {}; ++p) {{\n", td));
        msl.push_str("            acc += s_a[ty][p] * s_b[p][tx];\n");
        msl.push_str("        }\n\n");

        msl.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        msl.push_str("    }\n\n");

        match config.activation {
            Activation::None => {},
            Activation::ReLU => {
                msl.push_str("    acc = max(acc, 0.0f);\n");
            },
            Activation::GELU => {
                msl.push_str("    acc = acc * (1.0f / (1.0f + exp(-1.702f * acc)));\n");
            },
            Activation::Sigmoid => {
                msl.push_str("    acc = 1.0f / (1.0f + exp(-acc));\n");
            },
            Activation::Swish => {
                msl.push_str("    acc = acc * (1.0f / (1.0f + exp(-acc)));\n");
            }
        }

        msl.push_str("    if (global_row < m && global_col < n) {\n");
        msl.push_str("        C[global_row * n + global_col] = acc;\n");
        msl.push_str("    }\n");
        msl.push_str("}\n");

        msl
    }

    pub fn emit_metal_rmsnorm(&self, kernel_name: &str, config: &RMSNormConfig) -> String {
        let mut msl = String::with_capacity(1024);
        msl.push_str("#include <metal_stdlib>\n");
        msl.push_str("using namespace metal;\n\n");

        msl.push_str(&format!(
            "kernel void {}(\n\
            device const float* input [[buffer(0)]],\n\
            device const float* weight [[buffer(1)]],\n\
            device float* output [[buffer(2)]],\n\
            constant uint& hidden_dim [[buffer(3)]],\n\
            uint row_id [[threadgroup_position_in_grid]],\n\
            uint tid [[thread_position_in_threadgroup]])\n\
            {{\n",
            kernel_name
        ));

        msl.push_str("    threadgroup float smem[256];\n");
        msl.push_str("    uint row_offset = row_id * hidden_dim;\n");
        msl.push_str("    float local_sum = 0.0f;\n\n");

        msl.push_str("    for (uint i = tid; i < hidden_dim; i += 256) {\n");
        msl.push_str("        float x = input[row_offset + i];\n");
        msl.push_str("        local_sum += x * x;\n");
        msl.push_str("    }\n");
        msl.push_str("    smem[tid] = local_sum;\n");
        msl.push_str("    threadgroup_barrier(mem_flags::mem_threadgroup);\n\n");

        msl.push_str("    for (uint s = 128; s > 0; s >>= 1) {\n");
        msl.push_str("        if (tid < s) {\n");
        msl.push_str("            smem[tid] += smem[tid + s];\n");
        msl.push_str("        }\n");
        msl.push_str("        threadgroup_barrier(mem_flags::mem_threadgroup);\n");
        msl.push_str("    }\n\n");

        msl.push_str(&format!("    float inv_rms = rsqrt((smem[0] / float(hidden_dim)) + {:.6}f);\n\n", config.eps));

        msl.push_str("    for (uint i = tid; i < hidden_dim; i += 256) {\n");
        msl.push_str("        output[row_offset + i] = (input[row_offset + i] * inv_rms) * weight[i];\n");
        msl.push_str("    }\n");
        msl.push_str("}\n");

        msl
    }
}
