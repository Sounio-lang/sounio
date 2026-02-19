//! Epistemic GEMM intrinsic definitions
//!
//! Intrinsic operations for epistemic matrix multiplication.

use super::tensor_epistemic::{
    EpistemicTensorCategory, EpistemicTensorIntrinsic, EpsilonPropagationRule, TensorCoreOp,
};

/// Get all epistemic GEMM intrinsics
pub fn all_epistemic_gemm_intrinsics() -> Vec<EpistemicTensorIntrinsic> {
    vec![
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm",
            short_name: "gemm",
            description: "General matrix multiply with scaling factors and uncertainty propagation. Computes C = α * A * B + β * C.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Mma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: Some("wmma.mma.sync"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm.f32",
            short_name: "gemm_f32",
            description: "Single-precision GEMM with epistemic uncertainty tracking.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Mma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm.f16",
            short_name: "gemm_f16",
            description: "Half-precision GEMM with Tensor Core acceleration and uncertainty propagation.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: Some("wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm.confidence_gated",
            short_name: "gemm_confidence_gated",
            description: "GEMM that gates low-confidence values based on threshold.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Mma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm.batched",
            short_name: "gemm_batched",
            description: "Batched GEMM with per-batch uncertainty tracking.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::BatchMm),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gemm.strassen",
            short_name: "gemm_strassen",
            description: "Strassen algorithm for GEMM with uncertainty propagation through recursive decomposition.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Mma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
    ]
}

/// Check if an intrinsic name is an epistemic GEMM operation
pub fn is_epistemic_gemm_intrinsic(name: &str) -> bool {
    name.starts_with("epistemic.gemm")
}

/// Get epistemic GEMM intrinsic by name
pub fn get_epistemic_gemm_intrinsic(name: &str) -> Option<EpistemicTensorIntrinsic> {
    all_epistemic_gemm_intrinsics()
        .into_iter()
        .find(|i| i.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gemm_intrinsics_count() {
        let intrinsics = all_epistemic_gemm_intrinsics();
        assert!(intrinsics.len() >= 6);
    }

    #[test]
    fn test_is_gemm_intrinsic() {
        assert!(is_epistemic_gemm_intrinsic("epistemic.gemm"));
        assert!(is_epistemic_gemm_intrinsic("epistemic.gemm.f32"));
        assert!(is_epistemic_gemm_intrinsic("epistemic.gemm.f16"));
        assert!(!is_epistemic_gemm_intrinsic("epistemic.add"));
    }

    #[test]
    fn test_get_gemm_intrinsic() {
        let intr = get_epistemic_gemm_intrinsic("epistemic.gemm.f16").unwrap();
        assert_eq!(intr.short_name, "gemm_f16");
        assert_eq!(intr.category, EpistemicTensorCategory::MatrixOps);
        assert_eq!(
            intr.ptx_instruction,
            Some("wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16")
        );
    }
}
