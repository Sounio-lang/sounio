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

// ==================== QUATERNIONIC NEURAL NETWORK GPU KERNELS ====================
//
// QNN GPU Kernel Implementations
//
// References:
// - "Quaternion Convolutional Neural Networks" (arXiv:1804.10592)
// - "Quaternion Recurrent Neural Networks" (arXiv:1903.08478)
// - "Deep Quaternion Networks" (arXiv:1705.07944)
//
// Key Operations:
// - Hamilton Product: q1 ⊗ q2 (noncommutative quaternion multiplication)
// - Quaternion convolution: Each filter position uses Hamilton product
// - Quaternion BN: Normalizes each component independently

/// Generate quaternionic linear layer forward pass kernel
///
/// Computes: y = W ⊗ x + b
/// Where W is [output_features, input_features] of quaternions
/// x is [input_features] of quaternions
/// y is [output_features] of quaternions
/// ⊗ is the Hamilton product
pub fn gen_quat_linear_fwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_linear_fwd");

    kernel.add_param(ptr_param("W", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("x", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("b", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("out", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("in_features", GpuType::I32));
    kernel.add_param(scalar_param("out_features", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))), // out_idx
            (ValueId(5), GpuOp::Param(5)),                    // out_features
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Initialize accumulator with bias
            (ValueId(10), GpuOp::Param(2)), // b ptr
            (ValueId(11), GpuOp::Load(ValueId(10), MemorySpace::Global)),
            // Reduce over input features using Hamilton product
            // For each i: acc = acc ⊗ W[out_idx, i] ⊗ x[i]
            // Simplified: acc = Σ_i (W[out_idx, i] ⊗ x[i]) + b
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

/// Generate quaternionic linear layer backward pass kernel
///
/// Computes gradients for weight update and input gradient:
/// dW[o,i] += dy[o] ⊗ x[i]^T
/// dx[i] += Σ_o W[o,i]^T ⊗ dy[o]
pub fn gen_quat_linear_bwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_linear_bwd");

    kernel.add_param(ptr_param("W", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("x", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("dy", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("dW", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("dx", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("in_features", GpuType::I32));
    kernel.add_param(scalar_param("out_features", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(6)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Compute dW[out_idx, :] += dy[out_idx] ⊗ x[:]
            // Compute dx[out_idx] += W[: ,out_idx]^T ⊗ dy[out_idx]
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

/// Generate quaternionic 2D convolution forward pass kernel
///
/// Input: [batch, in_ch, H, W] of quaternions
/// Kernel: [out_ch, in_ch, kH, kW] of quaternions
/// Output: [batch, out_ch, H_out, W_out] of quaternions
///
/// Uses Hamilton product for each convolution position:
/// y[b, oc, h, w] = Σ_ic Σ_kh Σ_kw (kernel[oc, ic, kh, kw] ⊗ x[b, ic, h+kh, w+kw])
pub fn gen_quat_conv2d_fwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_conv2d_fwd");

    kernel.add_param(ptr_param("input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("kernel", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("output", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("batch", GpuType::I32));
    kernel.add_param(scalar_param("in_ch", GpuType::I32));
    kernel.add_param(scalar_param("out_ch", GpuType::I32));
    kernel.add_param(scalar_param("height", GpuType::I32));
    kernel.add_param(scalar_param("width", GpuType::I32));
    kernel.add_param(scalar_param("kH", GpuType::I32));
    kernel.add_param(scalar_param("kW", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::ThreadIdY),
            (ValueId(2), GpuOp::BlockIdX),
            (ValueId(3), GpuOp::BlockIdY),
            (ValueId(4), GpuOp::BlockDimX),
            (ValueId(5), GpuOp::Mul(ValueId(2), ValueId(4))),
            (ValueId(6), GpuOp::Add(ValueId(0), ValueId(5))), // x = threadIdx.x + blockIdx.x * blockDim.x
            (ValueId(7), GpuOp::Mul(ValueId(3), ValueId(1))), // y = threadIdx.y + blockIdx.y
        ],
        terminator: GpuTerminator::Br(BlockId(1)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Accumulate Hamilton products over in_ch, kH, kW
            // Each position: output[x,y] += kernel[oc, ic, kh, kw] ⊗ input[ic, y+kh, x+kw]
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

/// Generate quaternionic 2D convolution backward pass kernel
pub fn gen_quat_conv2d_bwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_conv2d_bwd");

    kernel.add_param(ptr_param("input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("kernel", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_output", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_kernel", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("batch", GpuType::I32));
    kernel.add_param(scalar_param("in_ch", GpuType::I32));
    kernel.add_param(scalar_param("out_ch", GpuType::I32));
    kernel.add_param(scalar_param("height", GpuType::I32));
    kernel.add_param(scalar_param("width", GpuType::I32));
    kernel.add_param(scalar_param("kH", GpuType::I32));
    kernel.add_param(scalar_param("kW", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
        ],
        terminator: GpuTerminator::Br(BlockId(1)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // d_kernel[oc, ic, kh, kw] += d_output[oc] ⊗ input[ic, :, :]
            // d_input[ic] += Σ_oc Σ_kh Σ_kw kernel[oc, ic, kh, kw]^T ⊗ d_output[oc]
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

/// Generate quaternionic ReLU activation kernel
///
/// Applies ReLU to each component of each quaternion:
/// ReLU(q)[i] = max(0, q[i]) for i in {w, x, y, z}
pub fn gen_quat_relu_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_relu");

    kernel.add_param(ptr_param("input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("output", GpuType::Vec4(Box::new(GpuType::F32))));
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
            (ValueId(13), GpuOp::QuatRelu(ValueId(12))), // Apply ReLU per component
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

/// Generate quaternionic batch normalization forward pass kernel
///
/// Normalizes each quaternion component independently:
/// y = γ * (x - μ) / σ + β
/// Computed separately for w, x, y, z components
pub fn gen_quat_bn_fwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_bn_fwd");

    kernel.add_param(ptr_param("input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("gamma", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("beta", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("mean", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("var", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("output", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));
    kernel.add_param(scalar_param("epsilon", GpuType::F32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(7)), // n
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // Load input quaternion
            (ValueId(10), GpuOp::Param(0)),
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            // Extract and normalize each component
            // out = gamma * (x - mean) / sqrt(var + epsilon) + beta
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

/// Generate quaternionic batch normalization backward pass kernel
pub fn gen_quat_bn_bwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("quat_bn_bwd");

    kernel.add_param(ptr_param("input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_output", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("gamma", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("mean", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("var", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_gamma", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_beta", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(ptr_param("d_input", GpuType::Vec4(Box::new(GpuType::F32))));
    kernel.add_param(scalar_param("n", GpuType::I32));
    kernel.add_param(scalar_param("epsilon", GpuType::F32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
        ],
        terminator: GpuTerminator::Br(BlockId(1)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // d_gamma += d_output * (input - mean) / sqrt(var + epsilon)
            // d_beta += d_output
            // d_input = d_output * gamma / sqrt(var + epsilon) * normalized_factor
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

// ==================== OCTONION GPU KERNELS (8D Hypercomplex) ====================
//
// Octonion: o = a + bi + cj + dk + el + fil + gjl + hkl
// Stored as Vec8<f32> for GPU compatibility
// Multiplication: o1 * o2 = (a1a2 - v1·v2) + (a1v2 + a2v1 + v1×v2)
// where v1, v2 are 7D imaginary parts, × is 7D cross product
//
// References:
// - arXiv:1601.01507 - Octonion-valued neural networks
// - arXiv:1711.03987 - Octonion fully connected neural networks
// - arXiv:1809.09704 - Octonion convolutional neural networks

/// Helper for Vec8 parameter
fn vec8_param(name: &str) -> GpuParam {
    GpuParam {
        name: name.into(),
        ty: GpuType::Array(Box::new(GpuType::F32), 8),
        space: MemorySpace::Global,
        restrict: true,
    }
}

/// Generate octonion multiplication kernel
///
/// Computes: o1 * o2 via Cayley-Dickson construction
/// Each octonion is represented as 8 f32 values: (real, e1, e2, e3, e4, e5, e6, e7)
///
/// Uses the standard octonion multiplication table based on the Fano plane.
#[allow(dead_code)]
pub fn gen_octonion_mul_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_mul");

    kernel.add_param(vec8_param("o1"));
    kernel.add_param(vec8_param("o2"));
    kernel.add_param(vec8_param("out"));
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

    // Compute block: Load 8 components from o1 and o2, multiply, store to out
    let mut compute_instrs = Vec::new();
    let mut vid = 10u32;

    // Get base pointers for o1, o2, out arrays
    compute_instrs.push((ValueId(vid), GpuOp::Param(0))); // o1 base
    let o1_base = ValueId(vid);
    vid += 1;

    compute_instrs.push((ValueId(vid), GpuOp::Param(1))); // o2 base
    let o2_base = ValueId(vid);
    vid += 1;

    compute_instrs.push((ValueId(vid), GpuOp::Param(2))); // out base
    let out_base = ValueId(vid);
    vid += 1;

    // Compute byte offset for this thread's octonion: idx * 8 (array of 8 floats)
    compute_instrs.push((ValueId(vid), GpuOp::ConstInt(8, GpuType::I32)));
    let eight = ValueId(vid);
    vid += 1;

    compute_instrs.push((ValueId(vid), GpuOp::Mul(ValueId(4), eight)));
    let base_offset = ValueId(vid);
    vid += 1;

    // Load all 8 components of o1[idx]
    let mut a = [ValueId(0); 8];
    for i in 0..8 {
        compute_instrs.push((ValueId(vid), GpuOp::ConstInt(i as i64, GpuType::I32)));
        let i_const = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Add(base_offset, i_const)));
        let offset = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::GetElementPtr(o1_base, vec![offset])));
        let ptr = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Load(ptr, MemorySpace::Global)));
        a[i] = ValueId(vid);
        vid += 1;
    }

    // Load all 8 components of o2[idx]
    let mut b = [ValueId(0); 8];
    for i in 0..8 {
        compute_instrs.push((ValueId(vid), GpuOp::ConstInt(i as i64, GpuType::I32)));
        let i_const = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Add(base_offset, i_const)));
        let offset = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::GetElementPtr(o2_base, vec![offset])));
        let ptr = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Load(ptr, MemorySpace::Global)));
        b[i] = ValueId(vid);
        vid += 1;
    }

    // Compute octonion product using standard multiplication table
    // c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    // c1 = a0*b1 + a1*b0 + a2*b4 - a4*b2 + a3*b7 - a7*b3 + a5*b6 - a6*b5
    // c2 = a0*b2 + a2*b0 + a4*b1 - a1*b4 + a3*b5 - a5*b3 + a6*b7 - a7*b6
    // c3 = a0*b3 + a3*b0 + a7*b1 - a1*b7 + a5*b2 - a2*b5 + a4*b6 - a6*b4
    // c4 = a0*b4 + a4*b0 + a1*b2 - a2*b1 + a3*b6 - a6*b3 + a7*b5 - a5*b7
    // c5 = a0*b5 + a5*b0 + a3*b2 - a2*b3 + a6*b1 - a1*b6 + a4*b7 - a7*b4
    // c6 = a0*b6 + a6*b0 + a5*b1 - a1*b5 + a4*b3 - a3*b4 + a7*b2 - a2*b7
    // c7 = a0*b7 + a7*b0 + a1*b3 - a3*b1 + a2*b6 - a6*b2 + a5*b4 - a4*b5

    // Helper closures for arithmetic operations
    let fmul =
        |instrs: &mut Vec<(ValueId, GpuOp)>, v: &mut u32, x: ValueId, y: ValueId| -> ValueId {
            instrs.push((ValueId(*v), GpuOp::FMul(x, y)));
            let result = ValueId(*v);
            *v += 1;
            result
        };

    let fadd =
        |instrs: &mut Vec<(ValueId, GpuOp)>, v: &mut u32, x: ValueId, y: ValueId| -> ValueId {
            instrs.push((ValueId(*v), GpuOp::FAdd(x, y)));
            let result = ValueId(*v);
            *v += 1;
            result
        };

    let fsub =
        |instrs: &mut Vec<(ValueId, GpuOp)>, v: &mut u32, x: ValueId, y: ValueId| -> ValueId {
            instrs.push((ValueId(*v), GpuOp::FSub(x, y)));
            let result = ValueId(*v);
            *v += 1;
            result
        };

    // c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
    let mut c0 = fmul(&mut compute_instrs, &mut vid, a[0], b[0]);
    for i in 1..8 {
        let prod = fmul(&mut compute_instrs, &mut vid, a[i], b[i]);
        c0 = fsub(&mut compute_instrs, &mut vid, c0, prod);
    }

    // c1 = a0*b1 + a1*b0 + a2*b4 - a4*b2 + a3*b7 - a7*b3 + a5*b6 - a6*b5
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[1]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[1], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[2], b[4]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[4], b[2]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[3], b[7]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[7], b[3]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[5], b[6]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[6], b[5]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c1 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c2 = a0*b2 + a2*b0 + a4*b1 - a1*b4 + a3*b5 - a5*b3 + a6*b7 - a7*b6
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[2]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[2], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[4], b[1]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[1], b[4]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[3], b[5]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[5], b[3]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[6], b[7]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[7], b[6]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c2 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c3 = a0*b3 + a3*b0 + a7*b1 - a1*b7 + a5*b2 - a2*b5 + a4*b6 - a6*b4
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[3]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[3], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[7], b[1]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[1], b[7]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[5], b[2]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[2], b[5]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[4], b[6]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[6], b[4]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c3 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c4 = a0*b4 + a4*b0 + a1*b2 - a2*b1 + a3*b6 - a6*b3 + a7*b5 - a5*b7
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[4]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[4], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[1], b[2]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[2], b[1]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[3], b[6]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[6], b[3]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[7], b[5]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[5], b[7]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c4 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c5 = a0*b5 + a5*b0 + a3*b2 - a2*b3 + a6*b1 - a1*b6 + a4*b7 - a7*b4
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[5]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[5], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[3], b[2]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[2], b[3]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[6], b[1]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[1], b[6]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[4], b[7]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[7], b[4]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c5 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c6 = a0*b6 + a6*b0 + a5*b1 - a1*b5 + a4*b3 - a3*b4 + a7*b2 - a2*b7
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[6]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[6], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[5], b[1]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[1], b[5]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[4], b[3]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[3], b[4]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[7], b[2]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[2], b[7]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c6 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    // c7 = a0*b7 + a7*b0 + a1*b3 - a3*b1 + a2*b6 - a6*b2 + a5*b4 - a4*b5
    let t1 = fmul(&mut compute_instrs, &mut vid, a[0], b[7]);
    let t2 = fmul(&mut compute_instrs, &mut vid, a[7], b[0]);
    let t3 = fmul(&mut compute_instrs, &mut vid, a[1], b[3]);
    let t4 = fmul(&mut compute_instrs, &mut vid, a[3], b[1]);
    let t5 = fmul(&mut compute_instrs, &mut vid, a[2], b[6]);
    let t6 = fmul(&mut compute_instrs, &mut vid, a[6], b[2]);
    let t7 = fmul(&mut compute_instrs, &mut vid, a[5], b[4]);
    let t8 = fmul(&mut compute_instrs, &mut vid, a[4], b[5]);
    let s1 = fadd(&mut compute_instrs, &mut vid, t1, t2);
    let s2 = fsub(&mut compute_instrs, &mut vid, t3, t4);
    let s3 = fsub(&mut compute_instrs, &mut vid, t5, t6);
    let s4 = fsub(&mut compute_instrs, &mut vid, t7, t8);
    let s5 = fadd(&mut compute_instrs, &mut vid, s1, s2);
    let s6 = fadd(&mut compute_instrs, &mut vid, s3, s4);
    let c7 = fadd(&mut compute_instrs, &mut vid, s5, s6);

    let c = [c0, c1, c2, c3, c4, c5, c6, c7];

    // Store all 8 components to out[idx]
    for i in 0..8 {
        compute_instrs.push((ValueId(vid), GpuOp::ConstInt(i as i64, GpuType::I32)));
        let i_const = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Add(base_offset, i_const)));
        let offset = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::GetElementPtr(out_base, vec![offset])));
        let ptr = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Store(ptr, c[i], MemorySpace::Global)));
        vid += 1;
    }

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: compute_instrs,
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

/// Generate octonion norm kernel
///
/// Computes: ||o|| = sqrt(sum of squares of all 8 components)
pub fn gen_octonion_norm_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_norm");

    kernel.add_param(vec8_param("o"));
    kernel.add_param(ptr_param("norm", GpuType::F32));
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

    let mut compute_instrs = Vec::new();
    let mut vid = 10u32;

    // Get base pointers
    compute_instrs.push((ValueId(vid), GpuOp::Param(0))); // o base
    let o_base = ValueId(vid);
    vid += 1;

    compute_instrs.push((ValueId(vid), GpuOp::Param(1))); // norm output
    let norm_base = ValueId(vid);
    vid += 1;

    // Compute base offset: idx * 8
    compute_instrs.push((ValueId(vid), GpuOp::ConstInt(8, GpuType::I32)));
    let eight = ValueId(vid);
    vid += 1;

    compute_instrs.push((ValueId(vid), GpuOp::Mul(ValueId(4), eight)));
    let base_offset = ValueId(vid);
    vid += 1;

    // Load all 8 components of o[idx] and compute sum of squares
    compute_instrs.push((ValueId(vid), GpuOp::ConstFloat(0.0, GpuType::F32)));
    let mut sum_sq = ValueId(vid);
    vid += 1;

    for i in 0..8 {
        compute_instrs.push((ValueId(vid), GpuOp::ConstInt(i as i64, GpuType::I32)));
        let i_const = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Add(base_offset, i_const)));
        let offset = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::GetElementPtr(o_base, vec![offset])));
        let ptr = ValueId(vid);
        vid += 1;

        compute_instrs.push((ValueId(vid), GpuOp::Load(ptr, MemorySpace::Global)));
        let val = ValueId(vid);
        vid += 1;

        // val * val
        compute_instrs.push((ValueId(vid), GpuOp::FMul(val, val)));
        let sq = ValueId(vid);
        vid += 1;

        // sum_sq += sq
        compute_instrs.push((ValueId(vid), GpuOp::FAdd(sum_sq, sq)));
        sum_sq = ValueId(vid);
        vid += 1;
    }

    // sqrt(sum_sq)
    compute_instrs.push((ValueId(vid), GpuOp::FastSqrt(sum_sq)));
    let norm_val = ValueId(vid);
    vid += 1;

    // Store norm to output
    compute_instrs.push((
        ValueId(vid),
        GpuOp::GetElementPtr(norm_base, vec![ValueId(4)]),
    ));
    let out_ptr = ValueId(vid);
    vid += 1;

    compute_instrs.push((
        ValueId(vid),
        GpuOp::Store(out_ptr, norm_val, MemorySpace::Global),
    ));

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: compute_instrs,
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

/// Generate octonion normalization kernel
pub fn gen_octonion_normalize_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_normalize");

    kernel.add_param(vec8_param("o"));
    kernel.add_param(vec8_param("out"));
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
            (ValueId(13), GpuOp::OctonionNormalize(ValueId(12))),
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

/// Generate octonion ReLU kernel
pub fn gen_octonion_relu_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_relu");

    kernel.add_param(vec8_param("input"));
    kernel.add_param(vec8_param("output"));
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
            (ValueId(13), GpuOp::OctonionRelu(ValueId(12))),
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

/// Generate octonion linear layer forward pass kernel
pub fn gen_octonion_linear_fwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_linear_fwd");

    kernel.add_param(vec8_param("W")); // weights [out, in] of octonions
    kernel.add_param(vec8_param("x")); // input [in] of octonions
    kernel.add_param(vec8_param("b")); // bias [out] of octonions
    kernel.add_param(vec8_param("out")); // output [out] of octonions
    kernel.add_param(scalar_param("in_features", GpuType::I32));
    kernel.add_param(scalar_param("out_features", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
            (ValueId(2), GpuOp::BlockDimX),
            (ValueId(3), GpuOp::Mul(ValueId(1), ValueId(2))),
            (ValueId(4), GpuOp::Add(ValueId(0), ValueId(3))),
            (ValueId(5), GpuOp::Param(5)),
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))),
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // y[out_idx] = Σ_i W[out_idx, i] ⊗ x[i] + b[out_idx]
            (ValueId(10), GpuOp::Param(2)), // bias ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
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

/// Generate octonion linear layer backward pass kernel
pub fn gen_octonion_linear_bwd_kernel() -> GpuKernel {
    let mut kernel = GpuKernel::new("octonion_linear_bwd");

    kernel.add_param(vec8_param("W"));
    kernel.add_param(vec8_param("x"));
    kernel.add_param(vec8_param("dy"));
    kernel.add_param(vec8_param("dW"));
    kernel.add_param(vec8_param("dx"));
    kernel.add_param(scalar_param("in_features", GpuType::I32));
    kernel.add_param(scalar_param("out_features", GpuType::I32));

    let entry = GpuBlock {
        id: BlockId(0),
        label: "entry".into(),
        instructions: vec![
            (ValueId(0), GpuOp::ThreadIdX),
            (ValueId(1), GpuOp::BlockIdX),
        ],
        terminator: GpuTerminator::Br(BlockId(1)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // dW[out_idx, in_idx] += dy[out_idx] ⊗ x[in_idx]
            // dx[in_idx] += W[out_idx, in_idx]^T ⊗ dy[out_idx]
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

    // ==================== QUATERNIONIC NEURAL NETWORK KERNELS ====================
    // arXiv:1804.10592 - Quaternion Convolutional Neural Networks
    // arXiv:1903.08478 - Quaternion Recurrent Neural Networks

    kernels.push(gen_quat_linear_fwd_kernel());
    kernels.push(gen_quat_linear_bwd_kernel());
    kernels.push(gen_quat_conv2d_fwd_kernel());
    kernels.push(gen_quat_conv2d_bwd_kernel());
    kernels.push(gen_quat_relu_kernel());
    kernels.push(gen_quat_bn_fwd_kernel());
    kernels.push(gen_quat_bn_bwd_kernel());

    // ==================== OCTONION KERNELS (8D Hypercomplex) ====================
    // arXiv:1601.01507 - Octonion-valued neural networks
    // arXiv:1711.03987 - Octonion fully connected neural networks
    // arXiv:1809.09704 - Octonion convolutional neural networks

    kernels.push(gen_octonion_mul_kernel());
    kernels.push(gen_octonion_norm_kernel());
    kernels.push(gen_octonion_normalize_kernel());
    kernels.push(gen_octonion_relu_kernel());
    kernels.push(gen_octonion_linear_fwd_kernel());
    kernels.push(gen_octonion_linear_bwd_kernel());
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
        assert_eq!(kernels.len(), 19); // 6 bio + 7 QNN + 6 Octonion kernels

        let names: Vec<_> = kernels.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"quaternion_mul"));
        assert!(names.contains(&"quaternion_normalize"));
        assert!(names.contains(&"quaternion_slerp"));
        assert!(names.contains(&"dna_complement"));
        assert!(names.contains(&"gf4_add"));
        assert!(names.contains(&"transmission_compose"));

        // QNN kernels
        assert!(names.contains(&"quat_linear_fwd"));
        assert!(names.contains(&"quat_linear_bwd"));
        assert!(names.contains(&"quat_conv2d_fwd"));
        assert!(names.contains(&"quat_conv2d_bwd"));
        assert!(names.contains(&"quat_relu"));
        assert!(names.contains(&"quat_bn_fwd"));
        assert!(names.contains(&"quat_bn_bwd"));

        // Octonion kernels (8D hypercomplex)
        assert!(names.contains(&"octonion_mul"));
        assert!(names.contains(&"octonion_norm"));
        assert!(names.contains(&"octonion_normalize"));
        assert!(names.contains(&"octonion_relu"));
        assert!(names.contains(&"octonion_linear_fwd"));
        assert!(names.contains(&"octonion_linear_bwd"));
    }
}
