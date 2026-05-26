#![allow(clippy::all)]

use tessera_ir::{GemmConfig, KernelModule};

pub const SPIRV_MAGIC: u32 = 0x07230203;
pub const SPIRV_VERSION_1_5: u32 = 0x00010500;
pub const TESSERA_GENERATOR_ID: u32 = 0x54455353;

pub struct SpirvGenerator {
    tile_dim: usize,
}

impl SpirvGenerator {
    pub fn new() -> Self {
        Self { tile_dim: 16 }
    }

    pub fn emit_vulkan_compute_shader(&self, kernel_name: &str, _config: &GemmConfig) -> String {
        let td = self.tile_dim;
        let mut code = String::with_capacity(2048);

        code.push_str("#version 450\n");
        code.push_str(&format!(
            "layout(local_size_x = {}, local_size_y = {}, local_size_z = 1) in;\n\n",
            td, td
        ));

        code.push_str("layout(std430, set = 0, binding = 0) readonly buffer BufferA {\n");
        code.push_str("    float a[];\n");
        code.push_str("};\n\n");

        code.push_str("layout(std430, set = 0, binding = 1) readonly buffer BufferB {\n");
        code.push_str("    float b[];\n");
        code.push_str("};\n\n");

        code.push_str("layout(std430, set = 0, binding = 2) writeonly buffer BufferC {\n");
        code.push_str("    float c[];\n");
        code.push_str("};\n\n");

        code.push_str("layout(push_constant) uniform Dimensions {\n");
        code.push_str("    uint m;\n");
        code.push_str("    uint n;\n");
        code.push_str("    uint k;\n");
        code.push_str("} dims;\n\n");

        code.push_str(&format!(
            "shared float s_a[{}][{}];\n",
            td, td
        ));
        code.push_str(&format!(
            "shared float s_b[{}][{}];\n\n",
            td, td
        ));

        code.push_str(&format!("void main() /* {} */ {{\n", kernel_name));
        code.push_str("    uint tx = gl_LocalInvocationID.x;\n");
        code.push_str("    uint ty = gl_LocalInvocationID.y;\n");
        code.push_str("    uint bx = gl_WorkGroupID.x;\n");
        code.push_str("    uint by = gl_WorkGroupID.y;\n\n");

        code.push_str(&format!("    uint global_row = by * {}u + ty;\n", td));
        code.push_str(&format!("    uint global_col = bx * {}u + tx;\n\n", td));

        code.push_str("    float acc = 0.0;\n\n");

        code.push_str(&format!("    for (uint k_idx = 0u; k_idx < dims.k; k_idx += {}u) {{\n", td));
        code.push_str("        uint curr_k_a = k_idx + tx;\n");
        code.push_str("        if (global_row < dims.m && curr_k_a < dims.k) {\n");
        code.push_str("            s_a[ty][tx] = a[global_row * dims.k + curr_k_a];\n");
        code.push_str("        } else {\n");
        code.push_str("            s_a[ty][tx] = 0.0;\n");
        code.push_str("        }\n\n");

        code.push_str("        uint curr_k_b = k_idx + ty;\n");
        code.push_str("        if (curr_k_b < dims.k && global_col < dims.n) {\n");
        code.push_str("            s_b[ty][tx] = b[curr_k_b * dims.n + global_col];\n");
        code.push_str("        } else {\n");
        code.push_str("            s_b[ty][tx] = 0.0;\n");
        code.push_str("        }\n\n");

        code.push_str("        barrier();\n\n");

        code.push_str(&format!("        for (uint p = 0u; p < {}u; ++p) {{\n", td));
        code.push_str("            acc = fma(s_a[ty][p], s_b[p][tx], acc);\n");
        code.push_str("        }\n\n");

        code.push_str("        barrier();\n");
        code.push_str("    }\n\n");

        code.push_str("    if (global_row < dims.m && global_col < dims.n) {\n");
        code.push_str("        c[global_row * dims.n + global_col] = acc;\n");
        code.push_str("    }\n");
        code.push_str("}\n");

        code
    }

    pub fn emit_spirv_binary(&self, _module: &KernelModule) -> Vec<u32> {
        let mut words = Vec::new();
        words.push(SPIRV_MAGIC);
        words.push(SPIRV_VERSION_1_5);
        words.push(TESSERA_GENERATOR_ID);
        words.push(100);
        words.push(0);

        // OpCapability Shader
        words.push((2 << 16) | 17);
        words.push(1);

        // OpMemoryModel Logical GLSL450
        words.push((3 << 16) | 14);
        words.push(0);
        words.push(1);

        // OpEntryPoint GLCompute
        words.push((6 << 16) | 15);
        words.push(5);
        words.push(1);
        words.push(0x6e69616d); // "main"
        words.push(0x00000000);
        words.push(0);

        // OpExecutionMode LocalSize 16 16 1
        words.push((6 << 16) | 16);
        words.push(1);
        words.push(17);
        words.push(self.tile_dim as u32);
        words.push(self.tile_dim as u32);
        words.push(1);

        words
    }
}

impl Default for SpirvGenerator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_compute_shader_generation() {
        let gen = SpirvGenerator::new();
        let config = GemmConfig::default();
        let code = gen.emit_vulkan_compute_shader("tessera_vulkan_gemm", &config);
        assert!(code.contains("#version 450"));
        assert!(code.contains("layout(local_size_x = 16, local_size_y = 16"));
        assert!(code.contains("barrier()"));
        assert!(code.contains("shared float s_a[16][16]"));
        assert!(code.contains("shared float s_b[16][16]"));
    }

    #[test]
    fn test_spirv_binary_header() {
        let gen = SpirvGenerator::new();
        let module = KernelModule::new("test");
        let binary = gen.emit_spirv_binary(&module);
        assert_eq!(binary[0], SPIRV_MAGIC);
        assert_eq!(binary[1], SPIRV_VERSION_1_5);
        assert_eq!(binary[2], TESSERA_GENERATOR_ID);
    }
}
