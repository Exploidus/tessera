pub mod layout;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    BF16,
    I32,
    I64,
    U32,
    U64,
}

impl DType {
    pub const fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 => 2,
            DType::I64 | DType::U64 => 8,
        }
    }

    pub const fn ptx_type_str(&self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::F16 => "f16",
            DType::BF16 => "bf16",
            DType::I32 => "s32",
            DType::I64 => "s64",
            DType::U32 => "u32",
            DType::U64 => "u64",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileShape {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

impl TileShape {
    pub const fn new_2d(rows: usize, cols: usize) -> Self {
        Self {
            m: rows,
            n: cols,
            k: 1,
        }
    }

    pub const fn new_gemm(m: usize, n: usize, k: usize) -> Self {
        Self { m, n, k }
    }

    pub const fn num_elements_2d(&self) -> usize {
        self.m * self.n
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Activation {
    None,
    ReLU,
    GELU,
    Sigmoid,
    Swish,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GemmConfig {
    pub block_m: usize,
    pub block_n: usize,
    pub block_k: usize,
    pub warp_m: usize,
    pub warp_n: usize,
    pub num_threads: usize,
    pub a_dtype: DType,
    pub b_dtype: DType,
    pub c_dtype: DType,
    pub acc_dtype: DType,
    pub use_swizzle: bool,
    pub use_tensor_cores: bool,
    pub activation: Activation,
}

impl Default for GemmConfig {
    fn default() -> Self {
        Self {
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
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RMSNormConfig {
    pub num_rows: usize,
    pub hidden_dim: usize,
    pub eps: f32,
}

impl Default for RMSNormConfig {
    fn default() -> Self {
        Self {
            num_rows: 1,
            hidden_dim: 4096,
            eps: 1e-5,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SwiGLUConfig {
    pub num_elements: usize,
}

impl Default for SwiGLUConfig {
    fn default() -> Self {
        Self { num_elements: 4096 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FlashAttentionConfig {
    pub batch_size: usize,
    pub num_heads: usize,
    pub seq_len: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            batch_size: 1,
            num_heads: 32,
            seq_len: 512,
            head_dim: 64,
            is_causal: true,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TileOp {
    LoadGlobal {
        dst_tile_id: usize,
        src_ptr: String,
        stride: usize,
    },
    StoreGlobal {
        dst_ptr: String,
        src_tile_id: usize,
        stride: usize,
    },
    TileMatMul(GemmConfig),
    RMSNorm(RMSNormConfig),
    SwiGLU(SwiGLUConfig),
    FlashAttention(FlashAttentionConfig),
}

#[derive(Debug, Clone)]
pub struct KernelModule {
    pub name: String,
    pub ops: Vec<TileOp>,
    pub gemm_config: Option<GemmConfig>,
}

impl KernelModule {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ops: Vec::new(),
            gemm_config: None,
        }
    }

    pub fn with_gemm(name: impl Into<String>, config: GemmConfig) -> Self {
        Self {
            name: name.into(),
            ops: vec![TileOp::TileMatMul(config.clone())],
            gemm_config: Some(config),
        }
    }
}
