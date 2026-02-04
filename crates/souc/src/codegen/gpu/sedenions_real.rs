//! Real GPU kernels for Sedenion (16D hypercomplex) operations
//!
//! Sedenions are 16-dimensional hypercomplex numbers in the Cayley-Dickson sequence:
//! ℝ → ℂ → ℍ (quaternions) → 𝕆 (octonions) → 𝕊 (sedenions)
//!
//! NOVEL: Sedenions have ZERO DIVISORS - non-zero elements whose product is zero.
//! This module includes a zero divisor detection kernel for numerical safety.
//!
//! Medical applications:
//! - 16-compartment PBPK (pharmacokinetic modeling)
//! - Multi-modal imaging fusion (16 imaging channels)
//! - EEG/MEG analysis (16-electrode clusters)
//!
//! References:
//! - Moreno, G. (1998). "The zero divisors of the Cayley-Dickson algebras"
//! - Cawagas, R. E. (2004). "On the Structure and Zero Divisors of the Sedenion Algebra"

use super::ir::{
    BlockId, GpuBlock, GpuKernel, GpuOp, GpuParam, GpuTerminator, GpuType, MemorySpace, ValueId,
};

/// Helper for sedenion parameters (16 f32 values = 512 bits)
fn sedenion_param(name: &str) -> GpuParam {
    GpuParam {
        name: name.into(),
        ty: GpuType::Array(Box::new(GpuType::F32), 16),
        space: MemorySpace::Global,
        restrict: true,
    }
}

/// Generate sedenion multiplication kernel (Cayley-Dickson)
///
/// (a, b) × (c, d) = (ac - d̄b, da + bc̄) where a, b, c, d are octonions
pub fn gen_sedenion_mul_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_mul");

    kernel.add_param(sedenion_param("s1"));
    kernel.add_param(sedenion_param("s2"));
    kernel.add_param(sedenion_param("out"));
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            // Thread index calculation
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(3)), // n
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load s1
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Load s2
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            // Sedenion multiplication (Cayley-Dickson)
            (ValueId(30), GpuOp::SedenionMul(ValueId(12), ValueId(22))),
            // Store result
            (ValueId(40), GpuOp::Param(2)),
            (
                ValueId(41),
                GpuOp::GetElementPtr(ValueId(40), vec![ValueId(4)]),
            ),
            (
                ValueId(42),
                GpuOp::Store(ValueId(41), ValueId(30), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Generate sedenion norm squared kernel
///
/// |s|² = Σᵢ eᵢ² (sum of squares of all 16 components)
pub fn gen_sedenion_norm_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_norm");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(GpuParam {
        name: "norm".into(),
        ty: GpuType::F32,
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)), // n
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load sedenion
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Compute norm squared
            (ValueId(13), GpuOp::SedenionNormSq(ValueId(12))),
            // Store norm
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Generate sedenion normalize kernel
///
/// Returns s / |s| (unit sedenion)
pub fn gen_sedenion_normalize_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_normalize");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(sedenion_param("out"));
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load sedenion
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Normalize
            (ValueId(13), GpuOp::SedenionNormalize(ValueId(12))),
            // Store result
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Generate sedenion ReLU kernel (component-wise)
///
/// ReLU(s) = (max(0, e₀), max(0, e₁), ..., max(0, e₁₅))
pub fn gen_sedenion_relu_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_relu");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(sedenion_param("out"));
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load sedenion
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Apply ReLU
            (ValueId(13), GpuOp::SedenionRelu(ValueId(12))),
            // Store result
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// NOVEL: Generate sedenion zero divisor check kernel
///
/// Returns 1.0 if the sedenion is a zero divisor, 0.0 otherwise.
/// Zero divisors are non-zero elements s such that ∃ t ≠ 0 where s × t = 0
///
/// Example zero divisor: (e₃ + e₁₀) × (e₆ - e₁₅) = 0
pub fn gen_sedenion_zero_divisor_check_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_zero_divisor_check");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(GpuParam {
        name: "is_zero_div".into(),
        ty: GpuType::F32,
        space: MemorySpace::Global,
        restrict: true,
    });
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load sedenion
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Check if zero divisor (native operation tests against known zero divisors)
            (ValueId(13), GpuOp::SedenionIsZeroDivisor(ValueId(12))),
            // Store result
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Generate sedenion sigmoid kernel (component-wise)
pub fn gen_sedenion_sigmoid_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_sigmoid");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(sedenion_param("out"));
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            (ValueId(13), GpuOp::SedenionSigmoid(ValueId(12))),
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Generate sedenion tanh kernel (component-wise)
pub fn gen_sedenion_tanh_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("sedenion_tanh");

    kernel.add_param(sedenion_param("s"));
    kernel.add_param(sedenion_param("out"));
    kernel.add_param(GpuParam {
        name: "n".into(),
        ty: GpuType::I32,
        space: MemorySpace::Generic,
        restrict: false,
    });

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(2)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            (ValueId(13), GpuOp::SedenionTanh(ValueId(12))),
            (ValueId(14), GpuOp::Param(1)),
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (
                ValueId(16),
                GpuOp::Store(ValueId(15), ValueId(13), MemorySpace::Global),
            ),
        ],
        terminator: GpuTerminator::Br(BlockId(2)),
    };

    let exit_block = GpuBlock {
        id: BlockId(2),
        label: "exit".into(),
        instructions: vec![],
        terminator: GpuTerminator::ReturnVoid,
    };

    kernel.blocks = vec![entry, compute_block, exit_block];
    kernel.entry = BlockId(0);
    kernel
}

/// Create all sedenion GPU kernels
pub fn create_all_sedenion_kernels() -> Vec<GpuKernel> {
    vec![
        gen_sedenion_mul_kernel(),
        gen_sedenion_norm_kernel(),
        gen_sedenion_normalize_kernel(),
        gen_sedenion_relu_kernel(),
        gen_sedenion_sigmoid_kernel(),
        gen_sedenion_tanh_kernel(),
        gen_sedenion_zero_divisor_check_kernel(),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sedenion_mul_kernel_structure() {
        let kernel = gen_sedenion_mul_kernel();
        assert_eq!(kernel.name, "sedenion_mul");
        assert_eq!(kernel.params.len(), 4); // s1, s2, out, n
        assert_eq!(kernel.blocks.len(), 3); // entry, compute, exit
    }

    #[test]
    fn test_sedenion_zero_divisor_kernel_structure() {
        let kernel = gen_sedenion_zero_divisor_check_kernel();
        assert_eq!(kernel.name, "sedenion_zero_divisor_check");
        assert_eq!(kernel.params.len(), 3); // s, is_zero_div, n
    }

    #[test]
    fn test_all_kernels_created() {
        let kernels = create_all_sedenion_kernels();
        assert_eq!(kernels.len(), 7);

        let names: Vec<_> = kernels.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"sedenion_mul"));
        assert!(names.contains(&"sedenion_norm"));
        assert!(names.contains(&"sedenion_normalize"));
        assert!(names.contains(&"sedenion_relu"));
        assert!(names.contains(&"sedenion_sigmoid"));
        assert!(names.contains(&"sedenion_tanh"));
        assert!(names.contains(&"sedenion_zero_divisor_check"));
    }
}
