use crate::{Activation, GemmConfig, KernelModule, TileOp};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FusionStats {
    pub original_ops: usize,
    pub fused_ops: usize,
    pub vertical_fusions: usize,
    pub horizontal_fusions: usize,
}

pub struct FusionEngine;

impl FusionEngine {
    /// Runs vertical (producer-consumer epilogue) and horizontal (parallel batch)
    /// fusion passes over a `KernelModule`.
    pub fn optimize(module: &KernelModule) -> (KernelModule, FusionStats) {
        let original_ops = module.ops.len();
        let (after_vertical, vertical_fusions) = Self::fuse_vertical(&module.ops);
        let (after_horizontal, horizontal_fusions) = Self::fuse_horizontal(&after_vertical);

        let fused_gemm_config = after_horizontal
            .iter()
            .find_map(|op| match op {
                TileOp::TileMatMul(cfg) => Some(cfg.clone()),
                TileOp::FusedHorizontal(group) => group.iter().find_map(|inner| match inner {
                    TileOp::TileMatMul(cfg) => Some(cfg.clone()),
                    _ => None,
                }),
                _ => None,
            })
            .or_else(|| module.gemm_config.clone());

        let fused_module = KernelModule {
            name: module.name.clone(),
            ops: after_horizontal,
            gemm_config: fused_gemm_config,
        };

        let stats = FusionStats {
            original_ops,
            fused_ops: fused_module.ops.len(),
            vertical_fusions,
            horizontal_fusions,
        };

        (fused_module, stats)
    }

    fn fuse_vertical(ops: &[TileOp]) -> (Vec<TileOp>, usize) {
        let mut out = Vec::with_capacity(ops.len());
        let mut count = 0;
        let mut i = 0;

        while i < ops.len() {
            if i + 1 < ops.len() {
                if let (TileOp::TileMatMul(gemm), TileOp::Elementwise(act)) = (&ops[i], &ops[i + 1])
                {
                    if gemm.activation == Activation::None && *act != Activation::None {
                        let mut fused = gemm.clone();
                        fused.activation = *act;
                        out.push(TileOp::TileMatMul(fused));
                        count += 1;
                        i += 2;
                        continue;
                    }
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }

        (out, count)
    }

    fn gemms_horizontally_compatible(a: &GemmConfig, b: &GemmConfig) -> bool {
        a.block_m == b.block_m
            && a.block_n == b.block_n
            && a.block_k == b.block_k
            && a.a_dtype == b.a_dtype
            && a.b_dtype == b.b_dtype
            && a.c_dtype == b.c_dtype
            && a.use_tensor_cores == b.use_tensor_cores
            && a.activation == b.activation
    }

    fn fuse_horizontal(ops: &[TileOp]) -> (Vec<TileOp>, usize) {
        let mut out = Vec::with_capacity(ops.len());
        let mut count = 0;
        let mut i = 0;

        while i < ops.len() {
            if let TileOp::TileMatMul(first_cfg) = &ops[i] {
                let mut group = vec![ops[i].clone()];
                let mut j = i + 1;
                while j < ops.len() {
                    if let TileOp::TileMatMul(next_cfg) = &ops[j] {
                        if Self::gemms_horizontally_compatible(first_cfg, next_cfg) {
                            group.push(ops[j].clone());
                            j += 1;
                            continue;
                        }
                    }
                    break;
                }

                if group.len() > 1 {
                    count += group.len() - 1;
                    out.push(TileOp::FusedHorizontal(group));
                    i = j;
                    continue;
                }
            }
            out.push(ops[i].clone());
            i += 1;
        }

        (out, count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vertical_fusion_gemm_activation() {
        let mut module = KernelModule::new("gemm_relu");
        module.ops.push(TileOp::TileMatMul(GemmConfig::default()));
        module.ops.push(TileOp::Elementwise(Activation::ReLU));

        let (fused, stats) = FusionEngine::optimize(&module);
        assert_eq!(stats.vertical_fusions, 1);
        assert_eq!(stats.horizontal_fusions, 0);
        assert_eq!(stats.original_ops, 2);
        assert_eq!(stats.fused_ops, 1);
        assert_eq!(
            fused.ops[0],
            TileOp::TileMatMul(GemmConfig {
                activation: Activation::ReLU,
                ..GemmConfig::default()
            })
        );
    }

    #[test]
    fn test_horizontal_fusion_qkv_projections() {
        let mut module = KernelModule::new("qkv_proj");
        for _ in 0..3 {
            module.ops.push(TileOp::TileMatMul(GemmConfig::default()));
        }

        let (fused, stats) = FusionEngine::optimize(&module);
        assert_eq!(stats.vertical_fusions, 0);
        assert_eq!(stats.horizontal_fusions, 2);
        assert_eq!(stats.fused_ops, 1);
        assert!(matches!(&fused.ops[0], TileOp::FusedHorizontal(g) if g.len() == 3));
    }

    #[test]
    fn test_combined_vertical_and_horizontal_fusion() {
        let mut module = KernelModule::new("qkv_gelu");
        for _ in 0..3 {
            module.ops.push(TileOp::TileMatMul(GemmConfig::default()));
            module.ops.push(TileOp::Elementwise(Activation::GELU));
        }

        let (fused, stats) = FusionEngine::optimize(&module);
        assert_eq!(stats.original_ops, 6);
        assert_eq!(stats.vertical_fusions, 3);
        assert_eq!(stats.horizontal_fusions, 2);
        assert_eq!(stats.fused_ops, 1);
        match &fused.ops[0] {
            TileOp::FusedHorizontal(g) => {
                assert_eq!(g.len(), 3);
                for op in g {
                    assert!(matches!(
                        op,
                        TileOp::TileMatMul(cfg) if cfg.activation == Activation::GELU
                    ));
                }
            }
            other => panic!("expected FusedHorizontal, got {:?}", other),
        }
    }
}
