//! Tensor Core Epistemic Operations
//!
//! GPU tensor operations with built-in uncertainty propagation for ML applications.
//! Uses NVIDIA Tensor Cores (WMMA API) when available, with fallback to standard ops.
//!
//! # Epistemic Tensor Model
//!
//! Each tensor element carries uncertainty information:
//! ```text
//! EpistemicTensor<T, ε> = {
//!     data: Tensor<T>,           // Main values
//!     epsilon: Tensor<f16>,      // Per-element uncertainty bounds
//!     provenance: ProvenanceMask // Bit-packed provenance markers
//! }
//! ```
//!
//! # Tensor Core Operations
//!
//! - `epistemic_wmma`: Matrix multiply-accumulate with uncertainty propagation
//! - `epistemic_reduce`: Reduction with confidence weighting
//! - `epistemic_softmax`: Softmax with uncertainty-aware normalization
//! - `epistemic_attention`: Attention with confidence gating

use std::fmt;

/// Tensor Core operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorCoreOp {
    /// Matrix multiply-accumulate: D = A * B + C
    Wmma,
    /// Matrix multiply: C = A * B
    Mma,
    /// Tensor contraction
    Contract,
    /// Batch matrix multiply
    BatchMm,
}

/// Epistemic tensor intrinsic
#[derive(Debug, Clone)]
pub struct EpistemicTensorIntrinsic {
    pub name: &'static str,
    pub short_name: &'static str,
    pub description: &'static str,
    pub category: EpistemicTensorCategory,
    pub tensor_op: Option<TensorCoreOp>,
    /// How epsilon propagates through this operation
    pub epsilon_propagation: EpsilonPropagationRule,
    /// PTX instruction (if directly mappable)
    pub ptx_instruction: Option<&'static str>,
}

/// Category of epistemic tensor operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EpistemicTensorCategory {
    /// Core matrix operations (WMMA)
    MatrixOps,
    /// Elementwise operations with uncertainty
    Elementwise,
    /// Reduction operations (sum, mean, max with confidence)
    Reduction,
    /// Attention and transformer operations
    Attention,
    /// Activation functions with uncertainty
    Activation,
    /// Normalization operations
    Normalization,
    /// Memory and layout operations
    Memory,
    /// Confidence/uncertainty manipulation
    Confidence,
}

impl EpistemicTensorCategory {
    pub fn all() -> &'static [Self] {
        &[
            Self::MatrixOps,
            Self::Elementwise,
            Self::Reduction,
            Self::Attention,
            Self::Activation,
            Self::Normalization,
            Self::Memory,
            Self::Confidence,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::MatrixOps => "Matrix Operations (Tensor Core)",
            Self::Elementwise => "Elementwise with Uncertainty",
            Self::Reduction => "Confidence-Weighted Reduction",
            Self::Attention => "Epistemic Attention",
            Self::Activation => "Uncertain Activations",
            Self::Normalization => "Epistemic Normalization",
            Self::Memory => "Tensor Memory",
            Self::Confidence => "Confidence Manipulation",
        }
    }
}

/// How epsilon propagates through an operation
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EpsilonPropagationRule {
    /// ε_out = ε_in (no change)
    Passthrough,
    /// ε_out = max(ε_a, ε_b)
    Maximum,
    /// ε_out = ε_a + ε_b (additive)
    Additive,
    /// ε_out = f(ε_a, ε_b, values) - computed at runtime
    Computed,
    /// ε_out = ε_a * |b| + ε_b * |a| (multiplicative)
    Multiplicative,
    /// ε_out = weighted_mean(ε_i, confidence_i)
    WeightedMean,
    /// ε_out depends on softmax temperature
    SoftmaxScaled,
    /// ε_out = 0 (introduces certainty, e.g., from ground truth)
    Reset,
}

impl fmt::Display for EpsilonPropagationRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Passthrough => write!(f, "passthrough"),
            Self::Maximum => write!(f, "max(ε_a, ε_b)"),
            Self::Additive => write!(f, "ε_a + ε_b"),
            Self::Computed => write!(f, "computed"),
            Self::Multiplicative => write!(f, "|a|ε_b + |b|ε_a"),
            Self::WeightedMean => write!(f, "weighted_mean"),
            Self::SoftmaxScaled => write!(f, "softmax_scaled"),
            Self::Reset => write!(f, "reset(0)"),
        }
    }
}

/// All epistemic tensor intrinsics
pub fn all_epistemic_tensor_intrinsics() -> Vec<EpistemicTensorIntrinsic> {
    vec![
        // === Matrix Operations (Tensor Core) ===
        EpistemicTensorIntrinsic {
            name: "epistemic.wmma.mma_sync",
            short_name: "wmma_mma",
            description: "Warp-level matrix multiply-accumulate with uncertainty propagation. \
                         Computes D = A*B + C where each matrix carries epsilon bounds.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: Some("wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.wmma.load_a",
            short_name: "wmma_load_a",
            description: "Load matrix A fragment with epsilon bounds from global memory.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: Some("wmma.load.a.sync.aligned.row.m16n16k16.global.f16"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.wmma.load_b",
            short_name: "wmma_load_b",
            description: "Load matrix B fragment with epsilon bounds from global memory.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: Some("wmma.load.b.sync.aligned.col.m16n16k16.global.f16"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.wmma.store_d",
            short_name: "wmma_store_d",
            description: "Store result matrix D with computed epsilon bounds to global memory.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: Some("wmma.store.d.sync.aligned.row.m16n16k16.global.f16"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.batch_matmul",
            short_name: "batch_matmul",
            description: "Batched matrix multiplication with per-batch uncertainty tracking.",
            category: EpistemicTensorCategory::MatrixOps,
            tensor_op: Some(TensorCoreOp::BatchMm),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        // === Elementwise Operations ===
        EpistemicTensorIntrinsic {
            name: "epistemic.add",
            short_name: "tensor_add",
            description: "Elementwise addition with additive epsilon propagation.",
            category: EpistemicTensorCategory::Elementwise,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Additive,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.mul",
            short_name: "tensor_mul",
            description: "Elementwise multiplication with multiplicative epsilon propagation.",
            category: EpistemicTensorCategory::Elementwise,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Multiplicative,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.div",
            short_name: "tensor_div",
            description: "Elementwise division with uncertainty widening near zero.",
            category: EpistemicTensorCategory::Elementwise,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.fma",
            short_name: "tensor_fma",
            description: "Fused multiply-add with combined uncertainty propagation.",
            category: EpistemicTensorCategory::Elementwise,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: Some("fma.rn.f16"),
        },
        // === Reduction Operations ===
        EpistemicTensorIntrinsic {
            name: "epistemic.reduce.sum",
            short_name: "reduce_sum",
            description: "Sum reduction with accumulated uncertainty.",
            category: EpistemicTensorCategory::Reduction,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Additive,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.reduce.mean",
            short_name: "reduce_mean",
            description: "Mean reduction with confidence-weighted averaging.",
            category: EpistemicTensorCategory::Reduction,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::WeightedMean,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.reduce.max",
            short_name: "reduce_max",
            description: "Max reduction, returns both value and its uncertainty.",
            category: EpistemicTensorCategory::Reduction,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.reduce.confidence_weighted",
            short_name: "reduce_confidence",
            description: "Reduction weighted by inverse uncertainty (high confidence = high weight).",
            category: EpistemicTensorCategory::Reduction,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::WeightedMean,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.warp_reduce_confidence",
            short_name: "warp_reduce_confidence",
            description: "Warp-level confidence voting reduction using ballot/shuffle.",
            category: EpistemicTensorCategory::Reduction,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::WeightedMean,
            ptx_instruction: Some("redux.sync.add.s32"),
        },
        // === Attention Operations ===
        EpistemicTensorIntrinsic {
            name: "epistemic.attention.scores",
            short_name: "attention_scores",
            description: "Compute attention scores Q*K^T with uncertainty propagation.",
            category: EpistemicTensorCategory::Attention,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Multiplicative,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.attention.softmax",
            short_name: "attention_softmax",
            description: "Softmax with temperature-scaled uncertainty.",
            category: EpistemicTensorCategory::Attention,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::SoftmaxScaled,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.attention.weighted_values",
            short_name: "attention_values",
            description: "Apply attention weights to values with confidence gating.",
            category: EpistemicTensorCategory::Attention,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.attention.full",
            short_name: "attention",
            description: "Full attention block: softmax(Q*K^T/sqrt(d)) * V with epistemic tracking.",
            category: EpistemicTensorCategory::Attention,
            tensor_op: Some(TensorCoreOp::Wmma),
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.attention.confidence_mask",
            short_name: "confidence_mask",
            description: "Generate attention mask based on confidence thresholds.",
            category: EpistemicTensorCategory::Attention,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Reset,
            ptx_instruction: None,
        },
        // === Activation Functions ===
        EpistemicTensorIntrinsic {
            name: "epistemic.relu",
            short_name: "relu",
            description: "ReLU with uncertainty preserved in positive region, widened at boundary.",
            category: EpistemicTensorCategory::Activation,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.gelu",
            short_name: "gelu",
            description: "GELU activation with smooth uncertainty transition.",
            category: EpistemicTensorCategory::Activation,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.silu",
            short_name: "silu",
            description: "SiLU/Swish activation with uncertainty propagation.",
            category: EpistemicTensorCategory::Activation,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.softmax",
            short_name: "softmax",
            description: "Softmax normalization with temperature-dependent uncertainty.",
            category: EpistemicTensorCategory::Activation,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::SoftmaxScaled,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.tanh",
            short_name: "tanh",
            description: "Tanh with uncertainty compression at saturation.",
            category: EpistemicTensorCategory::Activation,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        // === Normalization ===
        EpistemicTensorIntrinsic {
            name: "epistemic.layer_norm",
            short_name: "layer_norm",
            description: "Layer normalization with uncertainty-aware statistics.",
            category: EpistemicTensorCategory::Normalization,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.rms_norm",
            short_name: "rms_norm",
            description: "RMS normalization with epistemic tracking.",
            category: EpistemicTensorCategory::Normalization,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.batch_norm",
            short_name: "batch_norm",
            description: "Batch normalization with running uncertainty estimates.",
            category: EpistemicTensorCategory::Normalization,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        // === Memory Operations ===
        EpistemicTensorIntrinsic {
            name: "epistemic.load_tensor",
            short_name: "load_tensor",
            description: "Load epistemic tensor (value + epsilon) from memory.",
            category: EpistemicTensorCategory::Memory,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.store_tensor",
            short_name: "store_tensor",
            description: "Store epistemic tensor (value + epsilon) to memory.",
            category: EpistemicTensorCategory::Memory,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.load_ontology",
            short_name: "load_ontology",
            description: "Load ontology constants from constant memory.",
            category: EpistemicTensorCategory::Memory,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Reset,
            ptx_instruction: Some("ld.const"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.transpose",
            short_name: "transpose",
            description: "Transpose with epsilon reordering.",
            category: EpistemicTensorCategory::Memory,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.reshape",
            short_name: "reshape",
            description: "Reshape tensor maintaining epsilon correspondence.",
            category: EpistemicTensorCategory::Memory,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        // === Confidence Manipulation ===
        EpistemicTensorIntrinsic {
            name: "epistemic.get_confidence",
            short_name: "get_confidence",
            description: "Extract confidence (1 - ε) as a tensor.",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Reset,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.set_confidence",
            short_name: "set_confidence",
            description: "Set confidence (update epsilon bounds).",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Reset,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.threshold_confidence",
            short_name: "threshold_confidence",
            description: "Mask elements below confidence threshold.",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.calibrate",
            short_name: "calibrate",
            description: "Calibrate epsilon based on validation data.",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Computed,
            ptx_instruction: None,
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.provenance_and",
            short_name: "provenance_and",
            description: "AND provenance masks (intersection of sources).",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: Some("and.b32"),
        },
        EpistemicTensorIntrinsic {
            name: "epistemic.provenance_or",
            short_name: "provenance_or",
            description: "OR provenance masks (union of sources).",
            category: EpistemicTensorCategory::Confidence,
            tensor_op: None,
            epsilon_propagation: EpsilonPropagationRule::Passthrough,
            ptx_instruction: Some("or.b32"),
        },
    ]
}

/// Check if a name is an epistemic tensor intrinsic
pub fn is_epistemic_tensor_intrinsic(name: &str) -> bool {
    name.starts_with("epistemic.")
}

/// Get epistemic intrinsic by name
pub fn get_epistemic_intrinsic(name: &str) -> Option<EpistemicTensorIntrinsic> {
    all_epistemic_tensor_intrinsics()
        .into_iter()
        .find(|i| i.name == name)
}

/// Get epistemic intrinsic by short name
pub fn get_epistemic_intrinsic_by_short_name(short_name: &str) -> Option<EpistemicTensorIntrinsic> {
    all_epistemic_tensor_intrinsics()
        .into_iter()
        .find(|i| i.short_name == short_name)
}

/// Get all epistemic intrinsics by category
pub fn get_epistemic_intrinsics_by_category(
    category: EpistemicTensorCategory,
) -> Vec<EpistemicTensorIntrinsic> {
    all_epistemic_tensor_intrinsics()
        .into_iter()
        .filter(|i| i.category == category)
        .collect()
}

/// Get intrinsics that use Tensor Cores
pub fn get_tensor_core_intrinsics() -> Vec<EpistemicTensorIntrinsic> {
    all_epistemic_tensor_intrinsics()
        .into_iter()
        .filter(|i| i.tensor_op.is_some())
        .collect()
}

/// Compute epsilon propagation for matrix multiply
///
/// For C = A * B where A is m×k and B is k×n:
/// ε_C[i,j] ≈ Σ_l (|A[i,l]| * ε_B[l,j] + |B[l,j]| * ε_A[i,l])
///
/// This is conservative (upper bound) but fast to compute.
pub fn compute_wmma_epsilon(
    a_values: &[f32],
    a_epsilon: &[f32],
    b_values: &[f32],
    b_epsilon: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c_epsilon = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut eps_sum = 0.0f32;
            for l in 0..k {
                let a_idx = i * k + l;
                let b_idx = l * n + j;

                // Multiplicative uncertainty propagation
                eps_sum += a_values[a_idx].abs() * b_epsilon[b_idx];
                eps_sum += b_values[b_idx].abs() * a_epsilon[a_idx];
                eps_sum += a_epsilon[a_idx] * b_epsilon[b_idx];
            }
            c_epsilon[i * n + j] = eps_sum;
        }
    }

    c_epsilon
}

/// Compute epsilon for softmax
///
/// For softmax(x)_i = exp(x_i) / Σ_j exp(x_j):
/// The uncertainty depends on the temperature and input magnitudes.
pub fn compute_softmax_epsilon(values: &[f32], epsilons: &[f32], temperature: f32) -> Vec<f32> {
    let n = values.len();

    // Compute softmax
    let max_val = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_vals: Vec<f32> = values
        .iter()
        .map(|&x| ((x - max_val) / temperature).exp())
        .collect();
    let sum_exp: f32 = exp_vals.iter().sum();
    let softmax: Vec<f32> = exp_vals.iter().map(|&e| e / sum_exp).collect();

    // Propagate epsilon through softmax
    // Simplified: ε_out ≈ softmax * (1 - softmax) * ε_in / temperature
    softmax
        .iter()
        .zip(epsilons.iter())
        .map(|(&s, &e)| s * (1.0 - s) * e / temperature)
        .collect()
}

/// Compute epsilon for ReLU
///
/// ReLU(x) = max(0, x)
/// - For x >> 0: ε_out = ε_in
/// - For x << 0: ε_out = 0
/// - Near 0: ε_out depends on uncertainty of sign
pub fn compute_relu_epsilon(values: &[f32], epsilons: &[f32]) -> Vec<f32> {
    values
        .iter()
        .zip(epsilons.iter())
        .map(|(&x, &e)| {
            if x > e {
                // Clearly positive
                e
            } else if x < -e {
                // Clearly negative
                0.0
            } else {
                // Uncertain region - widen epsilon
                e * 2.0
            }
        })
        .collect()
}

/// Confidence-weighted reduction
///
/// Computes weighted average where weights are 1/(1 + ε).
/// High-confidence values contribute more to the result.
pub fn confidence_weighted_mean(values: &[f32], epsilons: &[f32]) -> (f32, f32) {
    let weights: Vec<f32> = epsilons.iter().map(|&e| 1.0 / (1.0 + e)).collect();
    let weight_sum: f32 = weights.iter().sum();

    let weighted_sum: f32 = values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum();

    let result_value = weighted_sum / weight_sum;

    // Result epsilon is the weighted average of epsilons
    let weighted_eps: f32 = epsilons
        .iter()
        .zip(weights.iter())
        .map(|(&e, &w)| e * w)
        .sum();

    let result_epsilon = weighted_eps / weight_sum;

    (result_value, result_epsilon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_epistemic_intrinsic() {
        assert!(is_epistemic_tensor_intrinsic("epistemic.wmma.mma_sync"));
        assert!(is_epistemic_tensor_intrinsic("epistemic.attention.full"));
        assert!(!is_epistemic_tensor_intrinsic("gpu.thread_id.x"));
        assert!(!is_epistemic_tensor_intrinsic("wmma"));
    }

    #[test]
    fn test_get_epistemic_intrinsic() {
        let intr = get_epistemic_intrinsic("epistemic.wmma.mma_sync").unwrap();
        assert_eq!(intr.short_name, "wmma_mma");
        assert_eq!(intr.tensor_op, Some(TensorCoreOp::Wmma));
    }

    #[test]
    fn test_get_by_short_name() {
        let intr = get_epistemic_intrinsic_by_short_name("attention").unwrap();
        assert_eq!(intr.name, "epistemic.attention.full");
        assert_eq!(intr.category, EpistemicTensorCategory::Attention);
    }

    #[test]
    fn test_tensor_core_intrinsics() {
        let tc_intrs = get_tensor_core_intrinsics();
        assert!(tc_intrs.len() >= 5);

        for intr in &tc_intrs {
            assert!(intr.tensor_op.is_some());
        }
    }

    #[test]
    fn test_category_coverage() {
        for category in EpistemicTensorCategory::all() {
            let intrs = get_epistemic_intrinsics_by_category(*category);
            assert!(
                !intrs.is_empty(),
                "Category {:?} has no intrinsics",
                category
            );
        }
    }

    #[test]
    fn test_wmma_epsilon_propagation() {
        // 2x2 * 2x2 matrix multiply
        let a_values = [1.0f32, 2.0, 3.0, 4.0];
        let a_epsilon = [0.01f32, 0.01, 0.01, 0.01];
        let b_values = [5.0f32, 6.0, 7.0, 8.0];
        let b_epsilon = [0.02f32, 0.02, 0.02, 0.02];

        let c_epsilon = compute_wmma_epsilon(&a_values, &a_epsilon, &b_values, &b_epsilon, 2, 2, 2);

        // Result should have propagated uncertainty
        assert_eq!(c_epsilon.len(), 4);
        for &e in &c_epsilon {
            assert!(e > 0.0);
            assert!(e < 1.0); // Should be reasonable
        }
    }

    #[test]
    fn test_softmax_epsilon() {
        let values = [1.0f32, 2.0, 3.0];
        let epsilons = [0.1f32, 0.1, 0.1];

        let result_eps = compute_softmax_epsilon(&values, &epsilons, 1.0);

        assert_eq!(result_eps.len(), 3);
        // Softmax output is bounded [0,1], so epsilon should be bounded
        for &e in &result_eps {
            assert!(e >= 0.0);
            assert!(e <= 0.25); // max is at softmax = 0.5
        }
    }

    #[test]
    fn test_relu_epsilon() {
        let values = [5.0f32, 0.01, -5.0];
        let epsilons = [0.1f32, 0.1, 0.1];

        let result_eps = compute_relu_epsilon(&values, &epsilons);

        // Positive value: preserve epsilon
        assert!((result_eps[0] - 0.1).abs() < 0.01);

        // Near zero: widen epsilon
        assert!(result_eps[1] > 0.1);

        // Negative value: zero epsilon
        assert!((result_eps[2] - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_weighted_mean() {
        // High-confidence value (low epsilon) should dominate
        let values = [10.0f32, 20.0];
        let epsilons = [0.01f32, 1.0]; // First has 100x more confidence

        let (mean, mean_eps) = confidence_weighted_mean(&values, &epsilons);

        // Result should be closer to 10.0 (the high-confidence value)
        assert!(mean < 15.0);
        assert!(mean_eps < 0.5);
    }

    #[test]
    fn test_epsilon_propagation_display() {
        assert_eq!(format!("{}", EpsilonPropagationRule::Additive), "ε_a + ε_b");
        assert_eq!(
            format!("{}", EpsilonPropagationRule::Maximum),
            "max(ε_a, ε_b)"
        );
    }

    #[test]
    fn test_all_intrinsics_count() {
        let all = all_epistemic_tensor_intrinsics();
        assert!(
            all.len() >= 30,
            "Expected at least 30 intrinsics, got {}",
            all.len()
        );
    }
}
