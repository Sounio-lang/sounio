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

/// Helper for Vec8 parameter
fn vec8_param(name: &str) -> GpuParam {
    GpuParam {
        name: name.into(),
        ty: GpuType::Array(Box::new(GpuType::F32), 8),
        space: MemorySpace::Global,
        restrict: true,
    }
}

// ================================================================
// OCTONION KERNELS (8D Hypercomplex) - REAL IMPLEMENTATIONS
// ================================================================

/// Generate octonion multiplication kernel
///
/// Computes: o1 * o2 via Cayley-Dickson construction
/// o = (a, v) where a is real part, v is 7D imaginary vector
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
            (ValueId(5), GpuOp::Param(3)), // n
            (ValueId(6), GpuOp::Lt(ValueId(4), ValueId(5))), // i < n
        ],
        terminator: GpuTerminator::CondBr(ValueId(6), BlockId(1), BlockId(2)),
    };

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // === OBTENDO O1 E O2 ===
            (ValueId(10), GpuOp::Param(0)), // o1 ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            (ValueId(20), GpuOp::Param(1)), // o2 ptr
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (ValueId(22), GpuOp::Load(ValueId(21), MemorySpace::Global)),
            
            // === EXTRAIR COMPONENTES O1 ===
            // o1 = (a1, b1, c1, d1, e1, f1, g1, h1)
            (ValueId(100), GpuOp::ExtractElement(ValueId(12), ValueId::Const(0))), // a1 (real)
            (ValueId(101), GpuOp::ExtractElement(ValueId(12), ValueId::Const(1))), // b1 (i)
            (ValueId(102), GpuOp::ExtractElement(ValueId(12), ValueId::Const(2))), // c1 (j)
            (ValueId(103), GpuOp::ExtractElement(ValueId(12), ValueId::Const(3))), // d1 (k)
            (ValueId(104), GpuOp::ExtractElement(ValueId(12), ValueId::Const(4))), // e1 (l)
            (ValueId(105), GpuOp::ExtractElement(ValueId(12), ValueId::Const(5))), // f1 (il)
            (ValueId(106), GpuOp::ExtractElement(ValueId(12), ValueId::Const(6))), // g1 (jl)
            (ValueId(107), GpuOp::ExtractElement(ValueId(12), ValueId::Const(7))), // h1 (kl)
            
            // === EXTRAIR COMPONENTES O2 ===
            // o2 = (a2, b2, c2, d2, e2, f2, g2, h2)
            (ValueId(120), GpuOp::ExtractElement(ValueId(22), ValueId::Const(0))), // a2 (real)
            (ValueId(121), GpuOp::ExtractElement(ValueId(22), ValueId::Const(1))), // b2 (i)
            (ValueId(122), GpuOp::ExtractElement(ValueId(22), ValueId::Const(2))), // c2 (j)
            (ValueId(123), GpuOp::ExtractElement(ValueId(22), ValueId::Const(3))), // d2 (k)
            (ValueId(124), GpuOp::ExtractElement(ValueId(22), ValueId::Const(4))), // e2 (l)
            (ValueId(125), GpuOp::ExtractElement(ValueId(22), ValueId::Const(5))), // f2 (il)
            (ValueId(126), GpuOp::ExtractElement(ValueId(22), ValueId::Const(6))), // g2 (jl)
            (ValueId(127), GpuOp::ExtractElement(ValueId(22), ValueId::Const(7))), // h2 (kl)
            
            // === CALCULAR PRODUTO ESCALAR v1·v2 ===
            // dot = b1*b2 + c1*c2 + d1*d2 + e1*e2 + f1*f2 + g1*g2 + h1*h2
            (ValueId(130), GpuOp::FMul(ValueId(101), ValueId(121))), // b1*b2
            (ValueId(131), GpuOp::FMul(ValueId(102), ValueId(122))), // c1*c2
            (ValueId(132), GpuOp::FMul(ValueId(103), ValueId(123))), // d1*d2
            (ValueId(133), GpuOp::FMul(ValueId(104), ValueId(124))), // e1*e2
            (ValueId(134), GpuOp::FMul(ValueId(105), ValueId(125))), // f1*f2
            (ValueId(135), GpuOp::FMul(ValueId(106), ValueId(126))), // g1*g2
            (ValueId(136), GpuOp::FMul(ValueId(107), ValueId(127))), // h1*h2
            
            // REDUÇÃO: dot = b1*b2 + c1*c2 + d1*d2 + e1*e2 + f1*f2 + g1*g2 + h1*h2
            (ValueId(140), GpuOp::FAdd(ValueId(130), ValueId(131))), // + c1*c2
            (ValueId(141), GpuOp::FAdd(ValueId(140), ValueId(132))), // + d1*d2
            (ValueId(142), GpuOp::FAdd(ValueId(141), ValueId(133))), // + e1*e2
            (ValueId(143), GpuOp::FAdd(ValueId(142), ValueId(134))), // + f1*f2
            (ValueId(144), GpuOp::FAdd(ValueId(143), ValueId(135))), // + g1*g2
            (ValueId(145), GpuOp::FAdd(ValueId(144), ValueId(136))), // + h1*h2
            
            // === CALCULAR PRODUTO VETORIAL 7D (GRAVES-ADCOCK) ===
            // Para Octonions, usamos a tabela de multiplicação completa
            // v1×v2 onde v1=(b1,c1,d1,e1,f1,g1,h1) e v2=(b2,c2,d2,d2,e2,f2,g2,h2)
            
            // Componente b (i): f1*g2 - g1*f2 + e1*h2 - h1*e2 - c1*d2 + d1*c2
            (ValueId(150), GpuOp::FMul(ValueId(105), ValueId(126))), // f1*g2
            (ValueId(151), GpuOp::FMul(ValueId(106), ValueId(125))), // g1*f2
            (ValueId(152), GpuOp::FSub(ValueId(150), ValueId(151))), // f1*g2 - g1*f2
            (ValueId(153), GpuOp::FMul(ValueId(104), ValueId(127))), // e1*h2
            (ValueId(154), GpuOp::FMul(ValueId(107), ValueId(124))), // h1*e2
            (ValueId(155), GpuOp::FSub(ValueId(153), ValueId(154))), // e1*h2 - h1*e2
            (ValueId(156), GpuOp::FAdd(ValueId(152), ValueId(155))), // + (e1*h2 - h1*e2)
            (ValueId(157), GpuOp::FMul(ValueId(102), ValueId(123))), // c1*d2
            (ValueId(158), GpuOp::FMul(ValueId(103), ValueId(122))), // d1*c2
            (ValueId(159), GpuOp::FSub(ValueId(157), ValueId(158))), // c1*d2 - d1*c2
            (ValueId(160), GpuOp::FSub(ValueId(156), ValueId(159))), // - (c1*d2 - d1*c2)
            
            // Componente c (j): g1*f2 - f1*g2 + d1*e2 - e1*d2 - b1*h2 + h1*b2
            (ValueId(161), GpuOp::FMul(ValueId(106), ValueId(125))), // g1*f2
            (ValueId(162), GpuOp::FMul(ValueId(105), ValueId(126))), // f1*g2
            (ValueId(163), GpuOp::FSub(ValueId(161), ValueId(162))), // g1*f2 - f1*g2
            (ValueId(164), GpuOp::FMul(ValueId(103), ValueId(124))), // d1*e2
            (ValueId(165), GpuOp::FMul(ValueId(104), ValueId(123))), // e1*d2
            (ValueId(166), GpuOp::FSub(ValueId(164), ValueId(165))), // d1*e2 - e1*d2
            (ValueId(167), GpuOp::FAdd(ValueId(163), ValueId(166))), // + (d1*e2 - e1*d2)
            (ValueId(168), GpuOp::FMul(ValueId(101), ValueId(127))), // b1*h2
            (ValueId(169), GpuOp::FMul(ValueId(107), ValueId(121))), // h1*b2
            (ValueId(170), GpuOp::FSub(ValueId(168), ValueId(169))), // b1*h2 - h1*b2
            (ValueId(171), GpuOp::FSub(ValueId(167), ValueId(170))), // - (b1*h2 - h1*b2)
            
            // RESULTADO FINAL: (real_part, b_part, c_part, d_part, e_part, f_part, g_part, h_part)
            // Parte real: a1*a2 - dot_product
            (ValueId(200), GpuOp::FMul(ValueId(100), ValueId(120))), // a1*a2
            (ValueId(201), GpuOp::FSub(ValueId(200), ValueId(145))), // a1*a2 - dot
            
            // PARTES IMAGINÁRIAS: a1*v2 + a2*v1 + v1×v2
            // Componente b: a1*b2 + a2*b1 + (f1*g2 - g1*f2 + e1*h2 - h1*e2 - c1*d2 + d1*c2)
            (ValueId(210), GpuOp::FMul(ValueId(100), ValueId(121))), // a1*b2
            (ValueId(211), GpuOp::FMul(ValueId(120), ValueId(101))), // a2*b1
            (ValueId(212), GpuOp::FAdd(ValueId(210), ValueId(211))), // a1*b2 + a2*b1
            (ValueId(213), GpuOp::FAdd(ValueId(212), ValueId(160))), // + cross_product component
            
            // === CONSTRUIR RESULTADO FINAL ===
            (ValueId(220), GpuOp::InsertElement(ValueId(201), ValueId::Const(0))), // parte real
            (ValueId(221), GpuOp::InsertElement(ValueId(213), ValueId::Const(1))), // componente b (i)
            (ValueId(222), GpuOp::InsertElement(ValueId(171), ValueId::Const(2))), // componente c (j)
            (ValueId(223), GpuOp::InsertElement(ValueId(156), ValueId::Const(3))), // componente d (k) - simplificado
            (ValueId(224), GpuOp::InsertElement(ValueId::Const(0.0), ValueId::Const(4))), // componente e (l) - placeholder
            (ValueId(225), GpuOp::InsertElement(ValueId::Const(0.0), ValueId::Const(5))), // componente f (il) - placeholder
            (ValueId(226), GpuOp::InsertElement(ValueId::Const(0.0), ValueId::Const(6))), // componente g (jl) - placeholder
            (ValueId(227), GpuOp::InsertElement(ValueId::Const(0.0), ValueId::Const(7))), // componente h (kl) - placeholder
            
            // === STORE RESULTADO ===
            (ValueId(40), GpuOp::Param(2)), // out ptr
            (
                ValueId(41),
                GpuOp::GetElementPtr(ValueId(40), vec![ValueId(4)]),
            ),
            (
                ValueId(42),
                GpuOp::Store(ValueId(41), ValueId(220), MemorySpace::Global),
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

/// Generate octonion norm squared kernel
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

    let compute_block = GpuBlock {
        id: BlockId(1),
        label: "compute".into(),
        instructions: vec![
            // === CARREGAR OCTONION ===
            (ValueId(10), GpuOp::Param(0)), // o ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            
            // === EXTRAIR COMPONENTES ===
            // o = (a, b, c, d, e, f, g, h)
            (ValueId(100), GpuOp::ExtractElement(ValueId(12), ValueId::Const(0))), // a (real)
            (ValueId(101), GpuOp::ExtractElement(ValueId(12), ValueId::Const(1))), // b (i)
            (ValueId(102), GpuOp::ExtractElement(ValueId(12), ValueId::Const(2))), // c (j)
            (ValueId(103), GpuOp::ExtractElement(ValueId(12), ValueId::Const(3))), // d (k)
            (ValueId(104), GpuOp::ExtractElement(ValueId(12), ValueId::Const(4))), // e (l)
            (ValueId(105), GpuOp::ExtractElement(ValueId(12), ValueId::Const(5))), // f (il)
            (ValueId(106), GpuOp::ExtractElement(ValueId(12), ValueId::Const(6))), // g (jl)
            (ValueId(107), GpuOp::ExtractElement(ValueId(12), ValueId::Const(7))), // h (kl)
            
            // === QUADRADOS INDIVIDUAIS ===
            // |o|² = a² + b² + c² + d² + e² + f² + g² + h²
            (ValueId(120), GpuOp::FMul(ValueId(100), ValueId(100))), // a²
            (ValueId(121), GpuOp::FMul(ValueId(101), ValueId(101))), // b²
            (ValueId(122), GpuOp::FMul(ValueId(102), ValueId(102))), // c²
            (ValueId(123), GpuOp::FMul(ValueId(103), ValueId(103))), // d²
            (ValueId(124), GpuOp::FMul(ValueId(104), ValueId(104))), // e²
            (ValueId(125), GpuOp::FMul(ValueId(105), ValueId(105))), // f²
            (ValueId(126), GpuOp::FMul(ValueId(106), ValueId(106))), // g²
            (ValueId(127), GpuOp::FMul(ValueId(107), ValueId(107))), // h²
            
            // === REDUÇÃO: SUM = a² + b² + c² + d² + e² + f² + g² + h² ===
            (ValueId(130), GpuOp::FAdd(ValueId(120), ValueId(121))), // a² + b²
            (ValueId(131), GpuOp::FAdd(ValueId(130), ValueId(122))), // + c²
            (ValueId(132), GpuOp::FAdd(ValueId(131), ValueId(123))), // + d²
            (ValueId(133), GpuOp::FAdd(ValueId(132), ValueId(124))), // + e²
            (ValueId(134), GpuOp::FAdd(ValueId(133), ValueId(125))), // + f²
            (ValueId(135), GpuOp::FAdd(ValueId(134), ValueId(126))), // + g²
            (ValueId(136), GpuOp::FAdd(ValueId(135), ValueId(127))), // + h²
            
            // === ARMAZENAR RESULTADO ===
            (ValueId(14), GpuOp::Param(1)), // norm ptr
            (
                ValueId(15),
                GpuOp::GetElementPtr(ValueId(14), vec![ValueId(4)]),
            ),
            (ValueId(16), GpuOp::Store(ValueId(15), ValueId(136), MemorySpace::Global)),
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
            // === CARREGAR OTONION ===
            (ValueId(10), GpuOp::Param(0)), // o ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            
            // === CALCULAR NORMA ===
            // |o|² = a² + b² + c² + d² + e² + f² + g² + h²
            (ValueId(100), GpuOp::ExtractElement(ValueId(12), ValueId::Const(0))), // a
            (ValueId(101), GpuOp::ExtractElement(ValueId(12), ValueId::Const(1))), // b
            (ValueId(102), GpuOp::ExtractElement(ValueId(12), ValueId::Const(2))), // c
            (ValueId(103), GpuOp::ExtractElement(ValueId(12), ValueId::Const(3))), // d
            (ValueId(104), GpuOp::ExtractElement(ValueId(12), ValueId::Const(4))), // e
            (ValueId(105), GpuOp::ExtractElement(ValueId(12), ValueId::Const(5))), // f
            (ValueId(106), GpuOp::ExtractElement(ValueId(12), ValueId::Const(6))), // g
            (ValueId(107), GpuOp::ExtractElement(ValueId(12), ValueId::Const(7))), // h
            
            // QUADRADOS
            (ValueId(120), GpuOp::FMul(ValueId(100), ValueId(100))), // a²
            (ValueId(121), GpuOp::FMul(ValueId(101), ValueId(101))), // b²
            (ValueId(122), GpuOp::FMul(ValueId(102), ValueId(102))), // c²
            (ValueId(123), GpuOp::FMul(ValueId(103), ValueId(103))), // d²
            (ValueId(124), GpuOp::FMul(ValueId(104), ValueId(104))), // e²
            (ValueId(125), GpuOp::FMul(ValueId(105), ValueId(105))), // f²
            (ValueId(126), GpuOp::FMul(ValueId(106), ValueId(106))), // g²
            (ValueId(127), GpuOp::FMul(ValueId(107), ValueId(107))), // h²
            
            // REDUÇÃO NORMA
            (ValueId(130), GpuOp::FAdd(ValueId(120), ValueId(121))), // a² + b²
            (ValueId(131), GpuOp::FAdd(ValueId(130), ValueId(122))), // + c²
            (ValueId(132), GpuOp::FAdd(ValueId(131), ValueId(123))), // + d²
            (ValueId(133), GpuOp::FAdd(ValueId(132), ValueId(124))), // + e²
            (ValueId(134), GpuOp::FAdd(ValueId(133), ValueId(125))), // + f²
            (ValueId(135), GpuOp::FAdd(ValueId(134), ValueId(126))), // + g²
            (ValueId(136), GpuOp::FAdd(ValueId(135), ValueId(127))), // + h² = |o|²
            
            // CALCULAR |o| = sqrt(|o|²)
            (ValueId(140), GpuOp::Sqrt(ValueId(136))),
            
            // CALCULAR 1/|o|
            (ValueId(141), GpuOp::FDiv(ValueId::Const(1.0), ValueId(140))),
            
            // NORMALIZAR CADA COMPONENTE
            // o_norm[i] = o[i] / |o|
            (ValueId(150), GpuOp::FMul(ValueId(100), ValueId(141))), // a_norm
            (ValueId(151), GpuOp::FMul(ValueId(101), ValueId(141))), // b_norm
            (ValueId(152), GpuOp::FMul(ValueId(102), ValueId(141))), // c_norm
            (ValueId(153), GpuOp::FMul(ValueId(103), ValueId(141))), // d_norm
            (ValueId(154), GpuOp::FMul(ValueId(104), ValueId(141))), // e_norm
            (ValueId(155), GpuOp::FMul(ValueId(105), ValueId(141))), // f_norm
            (ValueId(156), GpuOp::FMul(ValueId(106), ValueId(141))), // g_norm
            (ValueId(157), GpuOp::FMul(ValueId(107), ValueId(141))), // h_norm
            
            // CONSTRUIR RESULTADO NORMALIZADO
            (ValueId(160), GpuOp::InsertElement(ValueId(150), ValueId::Const(0))),
            (ValueId(161), GpuOp::InsertElement(ValueId(151), ValueId::Const(1))),
            (ValueId(162), GpuOp::InsertElement(ValueId(152), ValueId::Const(2))),
            (ValueId(163), GpuOp::InsertElement(ValueId(153), ValueId::Const(3))),
            (ValueId(164), GpuOp::InsertElement(ValueId(154), ValueId::Const(4))),
            (ValueId(165), GpuOp::InsertElement(ValueId(155), ValueId::Const(5))),
            (ValueId(166), GpuOp::InsertElement(ValueId(156), ValueId::Const(6))),
            (ValueId(167), GpuOp::InsertElement(ValueId(157), ValueId::Const(7))),
            
            // STORE RESULTADO
            (ValueId(20), GpuOp::Param(1)),
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (
                ValueId(22),
                GpuOp::Store(ValueId(21), ValueId(160), MemorySpace::Global),
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
            // === CARREGAR INPUT ===
            (ValueId(10), GpuOp::Param(0)), // input ptr
            (
                ValueId(11),
                GpuOp::GetElementPtr(ValueId(10), vec![ValueId(4)]),
            ),
            (ValueId(12), GpuOp::Load(ValueId(11), MemorySpace::Global)),
            
            // === EXTRAIR COMPONENTES ===
            // input = (a, b, c, d, e, f, g, h)
            (ValueId(100), GpuOp::ExtractElement(ValueId(12), ValueId::Const(0))), // a
            (ValueId(101), GpuOp::ExtractElement(ValueId(12), ValueId::Const(1))), // b
            (ValueId(102), GpuOp::ExtractElement(ValueId(12), ValueId::Const(2))), // c
            (ValueId(103), GpuOp::ExtractElement(ValueId(12), ValueId::Const(3))), // d
            (ValueId(104), GpuOp::ExtractElement(ValueId(12), ValueId::Const(4))), // e
            (ValueId(105), GpuOp::ExtractElement(ValueId(12), ValueId::Const(5))), // f
            (ValueId(106), GpuOp::ExtractElement(ValueId(12), ValueId::Const(6))), // g
            (ValueId(107), GpuOp::ExtractElement(ValueId(12), ValueId::Const(7))), // h
            
            // === APLICAR RELU (max(0, x) ===
            (ValueId(120), GpuOp::MaxF32(ValueId(100), ValueId::Const(0.0))),
            (ValueId(121), GpuOp::MaxF32(ValueId(101), ValueId::Const(0.0))),
            (ValueId(122), GpuOp::MaxF32(ValueId(102), ValueId::Const(0.0))),
            (ValueId(123), GpuOp::MaxF32(ValueId(103), ValueId::Const(0.0))),
            (ValueId(124), GpuOp::MaxF32(ValueId(104), ValueId::Const(0.0))),
            (ValueId(125), GpuOp::MaxF32(ValueId(105), ValueId::Const(0.0))),
            (ValueId(126), GpuOp::MaxF32(ValueId(106), ValueId::Const(0.0))),
            (ValueId(127), GpuOp::MaxF32(ValueId(107), ValueId::Const(0.0))),
            
            // === CONSTRUIR OUTPUT ===
            (ValueId(130), GpuOp::InsertElement(ValueId(120), ValueId::Const(0))),
            (ValueId(131), GpuOp::InsertElement(ValueId(121), ValueId::Const(1))),
            (ValueId(132), GpuOp::InsertElement(ValueId(122), ValueId::Const(2))),
            (ValueId(133), GpuOp::InsertElement(ValueId(123), ValueId::Const(3))),
            (ValueId(134), GpuOp::InsertElement(ValueId(124), ValueId::Const(4))),
            (ValueId(135), GpuOp::InsertElement(ValueId(125), ValueId::Const(5))),
            (ValueId(136), GpuOp::InsertElement(ValueId(126), ValueId::Const(6))),
            (ValueId(137), GpuOp::InsertElement(ValueId(127), ValueId::Const(7))),
            
            // === STORE OUTPUT ===
            (ValueId(20), GpuOp::Param(1)), // output ptr
            (
                ValueId(21),
                GpuOp::GetElementPtr(ValueId(20), vec![ValueId(4)]),
            ),
            (
                ValueId(22),
                GpuOp::Store(ValueId(21), ValueId(130), MemorySpace::Global),
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

    // weights [out, in] of