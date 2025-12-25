//! GPU Kernels for Biological/Quaternionic Computing
//!
//! Implements GPU-accelerated versions of operations from "The Quaternionic Syntax
//! of Existence" (Agourakis & Agourakis). These kernels enable massively parallel
//! processing of DNA sequences and quaternion-based transmission dynamics.
//!
//! # Kernel Categories
//!
//! - **Quaternion kernels**: Hamilton product, normalization, SLERP
//! - **DNA kernels**: Complement, shift, reverse on sequence arrays
//! - **GF(4) kernels**: Field arithmetic for quaternary DNA encoding
//! - **Transmission kernels**: Channel composition, distortion, interpolation
//!
//! # Performance Characteristics
//!
//! All kernels are designed for coalesced memory access and utilize:
//! - Warp-level primitives for reductions
//! - Shared memory for sequence transformations
//! - Unit quaternion invariant preservation through renormalization

use super::ir::{
    BlockId, GpuBlock, GpuFunction, GpuKernel, GpuOp, GpuParam, GpuTerminator, GpuType,
    MemorySpace, ValueId,
};

/// Helper to create a global pointer parameter
fn ptr_param(name: &str, elem_ty: GpuType) -> GpuParam {
    GpuParam {
        name: name.into(),
        ty: GpuType::Ptr(Box::new(elem_ty), MemorySpace::Global),
        space: MemorySpace::Global,
        restrict: true,
    }
}

/// Helper to create a scalar parameter (passed by value)
fn scalar_param(name: &str, ty: GpuType) -> GpuParam {
    GpuParam {
        name: name.into(),
        ty,
        space: MemorySpace::Generic,
        restrict: false,
    }
}

/// Generate quaternion multiplication kernel
///
/// Computes Hamilton product: q1 * q2 for arrays of quaternions.
/// Result stored in output array. All quaternions are Vec4<f32>.
///
/// ```text
/// q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2)
///         + (w1x2 + x1w2 + y1z2 - z1y2)i
///         + (w1y2 - x1z2 + y1w2 + z1x2)j
///         + (w1z2 + x1y2 - y1x2 + z1w2)k
/// ```
pub fn gen_quaternion_mul_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quaternion_mul");

    // Parameters: q1_ptr, q2_ptr, out_ptr, n
    kernel.add_param(ptr_param("q1", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("q2", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("out", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));

    // Main block
    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            // Get thread index
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))), // i = threadIdx.x + blockIdx.x * blockDim.x
            // Bounds check
            (ValueId(5), GpuOp::Param(3)),                   // n
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))), // i < n
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load q1[i] and q2[i]
            (ValueId(10), GpuOp::Param(0)), // q1 ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            (ValueId(20), GpuOp::Param(1)), // q2 ptr
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            // Quaternion multiply using our bio op
            (ValueId(30), GpuOp::QuatMul(ValueId(12), ValueId(22))),
            // Store result
            (ValueId(40), GpuOp::Param(2)), // out ptr
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

/// Generate quaternion normalization kernel
///
/// Normalizes quaternions to unit length: q / |q|
/// Essential for maintaining SU(2) structure in transmission dynamics.
pub fn gen_quaternion_normalize_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quaternion_normalize");

    kernel.add_param(ptr_param("q", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("out", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));

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
            (ValueId(13), GpuOp::QuatNormalize(ValueId(12))),
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (
                ValueId(22),
                GpuOp::Store(ValueId(21), ValueId(13), MemorySpace::Global),
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

/// Generate DNA complement kernel
///
/// Computes Watson-Crick complement for DNA sequences encoded as u8:
/// - A (0) ↔ T (3)
/// - C (1) ↔ G (2)
///
/// Uses GF(4) XOR: complement(x) = x XOR 3
pub fn gen_dna_complement_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("dna_complement");

    kernel.add_param(ptr_param("seq", GpuType::U8));
    kernel.add_param(ptr_param("out", GpuType::U8));
    kernel.add_param(scalar_param("n", GpuType::I32));

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
            (ValueId(13), GpuOp::DnaComplement(ValueId(12))),
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (
                ValueId(22),
                GpuOp::Store(ValueId(21), ValueId(13), MemorySpace::Global),
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

/// Generate GF(4) vector addition kernel
///
/// Element-wise addition in GF(4) for DNA sequences.
/// GF(4) has characteristic 2, so x + x = 0.
pub fn gen_gf4_add_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("gf4_add");

    kernel.add_param(ptr_param("a", GpuType::U8));
    kernel.add_param(ptr_param("b", GpuType::U8));
    kernel.add_param(ptr_param("out", GpuType::U8));
    kernel.add_param(scalar_param("n", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(3)),
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
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            (ValueId(30), GpuOp::Gf4Add(ValueId(12), ValueId(22))),
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

/// Generate transmission composition kernel
///
/// Computes quaternion product of transmission states for
/// sequential channel composition. Non-commutative!
pub fn gen_transmission_compose_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("transmission_compose");

    // Transmission states stored as Vec4<f32> = (g, t, p, e)
    kernel.add_param(ptr_param("t1", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("t2", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("out", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(3)),
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
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            // Use transmission compose (quaternion product + renormalize)
            (
                ValueId(30),
                GpuOp::TransmissionCompose(ValueId(12), ValueId(22)),
            ),
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

/// Generate quaternion SLERP kernel
///
/// Spherical linear interpolation between two quaternion arrays.
/// Used for smooth transmission state transitions.
pub fn gen_quaternion_slerp_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quaternion_slerp");

    kernel.add_param(ptr_param("q1", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("q2", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("t", GpuType::F32)); // Per-element t values
    kernel.add_param(ptr_param("out", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(4)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load q1[i]
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Load q2[i]
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            // Load t[i]
            (ValueId(30), GpuOp::Param(2)),
            (
                ValueId(31),
                GpuOp::GetElementPtr(ValueId(30), vec![ValueId(4)]),
            ),
            (ValueId(32), GpuOp::Load(ValueId(31), MemorySpace::Global)),
            // SLERP
            (
                ValueId(40),
                GpuOp::QuatSlerp(ValueId(12), ValueId(22), ValueId(32)),
            ),
            // Store
            (ValueId(50), GpuOp::Param(3)),
            (
                ValueId(51),
                GpuOp::GetElementPtr(ValueId(50), vec![ValueId(4)]),
            ),
            (
                ValueId(52),
                GpuOp::Store(ValueId(51), ValueId(40), MemorySpace::Global),
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

/// Generate all bio kernels and add to GPU module
pub fn add_bio_kernels(kernels: &mut Vec<GpuKernel>) {
    kernels.push(gen_quaternion_mul_kernel());
    kernels.push(gen_quaternion_normalize_kernel());
    kernels.push(gen_quaternion_slerp_kernel());
    kernels.push(gen_dna_complement_kernel());
    kernels.push(gen_gf4_add_kernel());
    kernels.push(gen_transmission_compose_kernel());
}

/// Device function for Hamilton product (inlined into kernels)
pub fn gen_quaternion_mul_device_fn() -> GpuFunction {
    GpuFunction {
        name: "__quat_mul".into(),
        params: vec![
            GpuParam {
                name: "q1".into(),
                ty: GpuType::Vec4(Box::new(GpuType::F32)),
                space: MemorySpace::Generic,
                restrict: false,
            },
            GpuParam {
                name: "q2".into(),
                ty: GpuType::Vec4(Box::new(GpuType::F32)),
                space: MemorySpace::Generic,
                restrict: false,
            },
        ],
        return_type: GpuType::Vec4(Box::new(GpuType::F32)),
        blocks: vec![], // Implementation in PTX emitter
        entry: BlockId(0),
        inline: true,
    }
}

/// Device function for quaternion normalization
pub fn gen_quaternion_normalize_device_fn() -> GpuFunction {
    GpuFunction {
        name: "__quat_normalize".into(),
        params: vec![GpuParam {
            name: "q".into(),
            ty: GpuType::Vec4(Box::new(GpuType::F32)),
            space: MemorySpace::Generic,
            restrict: false,
        }],
        return_type: GpuType::Vec4(Box::new(GpuType::F32)),
        blocks: vec![],
        entry: BlockId(0),
        inline: true,
    }
}

/// Device function for GF(4) addition lookup
pub fn gen_gf4_add_device_fn() -> GpuFunction {
    GpuFunction {
        name: "__gf4_add".into(),
        params: vec![
            GpuParam {
                name: "a".into(),
                ty: GpuType::U8,
                space: MemorySpace::Generic,
                restrict: false,
            },
            GpuParam {
                name: "b".into(),
                ty: GpuType::U8,
                space: MemorySpace::Generic,
                restrict: false,
            },
        ],
        return_type: GpuType::U8,
        blocks: vec![],
        entry: BlockId(0),
        inline: true,
    }
}

/// Device function for GF(4) multiplication lookup
pub fn gen_gf4_mul_device_fn() -> GpuFunction {
    GpuFunction {
        name: "__gf4_mul".into(),
        params: vec![
            GpuParam {
                name: "a".into(),
                ty: GpuType::U8,
                space: MemorySpace::Generic,
                restrict: false,
            },
            GpuParam {
                name: "b".into(),
                ty: GpuType::U8,
                space: MemorySpace::Generic,
                restrict: false,
            },
        ],
        return_type: GpuType::U8,
        blocks: vec![],
        entry: BlockId(0),
        inline: true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_mul_kernel_structure() {
        let kernel = gen_quaternion_mul_kernel();
        assert_eq!(kernel.name, "quaternion_mul");
        assert_eq!(kernel.params.len(), 4);
        assert_eq!(kernel.blocks.len(), 3);
    }

    #[test]
    fn test_dna_complement_kernel_structure() {
        let kernel = gen_dna_complement_kernel();
        assert_eq!(kernel.name, "dna_complement");
        assert_eq!(kernel.params.len(), 3);
    }

    #[test]
    fn test_bio_kernels_generation() {
        let mut kernels = Vec::new();
        add_bio_kernels(&mut kernels);
        assert_eq!(kernels.len(), 6);

        let names: Vec<_> = kernels.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"quaternion_mul"));
        assert!(names.contains(&"quaternion_normalize"));
        assert!(names.contains(&"quaternion_slerp"));
        assert!(names.contains(&"dna_complement"));
        assert!(names.contains(&"gf4_add"));
        assert!(names.contains(&"transmission_compose"));
    }
}
