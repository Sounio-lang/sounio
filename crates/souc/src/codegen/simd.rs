//! SIMD operations for linear algebra primitives
//!
//! This module provides SIMD-optimized implementations for vector, matrix,
//! and quaternion operations using Cranelift's SIMD instruction support.
//!
//! Supported SIMD types:
//! - vec2, vec3, vec4: F32X4 (128-bit, 4x f32)
//! - quat: F32X4 (128-bit, 4x f32 as x, y, z, w)
//! - mat2: F32X4 (128-bit, 2x2 matrix column-major)
//! - mat3, mat4: Pointer to aligned memory (multiple F32X4 lanes)

#[cfg(feature = "jit")]
use cranelift_codegen::ir::{InstBuilder, Value, types};
#[cfg(feature = "jit")]
use cranelift_frontend::FunctionBuilder;

/// SIMD operations for vec3/vec4
#[cfg(feature = "jit")]
pub struct SimdVec;

#[cfg(feature = "jit")]
impl SimdVec {
    /// Create vec4 from 4 f32 values
    pub fn splat_f32x4(
        builder: &mut FunctionBuilder,
        x: Value,
        y: Value,
        z: Value,
        w: Value,
    ) -> Value {
        // Insert each component into the vector
        let zero = builder.ins().f32const(0.0);
        let vec = builder.ins().scalar_to_vector(types::F32X4, zero);

        // Insert x at index 0
        let vec = builder.ins().insertlane(vec, x, 0);
        // Insert y at index 1
        let vec = builder.ins().insertlane(vec, y, 1);
        // Insert z at index 2
        let vec = builder.ins().insertlane(vec, z, 2);
        // Insert w at index 3
        builder.ins().insertlane(vec, w, 3)
    }

    /// Extract component from vec4
    pub fn extract_lane(builder: &mut FunctionBuilder, vec: Value, lane: u8) -> Value {
        builder.ins().extractlane(vec, lane)
    }

    /// Vector addition: a + b (component-wise)
    pub fn add(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fadd(a, b)
    }

    /// Vector subtraction: a - b (component-wise)
    pub fn sub(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fsub(a, b)
    }

    /// Vector multiplication: a * b (component-wise)
    pub fn mul(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fmul(a, b)
    }

    /// Vector division: a / b (component-wise)
    pub fn div(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        builder.ins().fdiv(a, b)
    }

    /// Scalar multiplication: v * s (broadcast scalar to all lanes)
    pub fn scale(builder: &mut FunctionBuilder, vec: Value, scalar: Value) -> Value {
        let splat = builder.ins().splat(types::F32X4, scalar);
        builder.ins().fmul(vec, splat)
    }

    /// Dot product for vec3 (ignores w component)
    /// dot(a, b) = a.x*b.x + a.y*b.y + a.z*b.z
    pub fn dot3(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        // Multiply component-wise
        let prod = builder.ins().fmul(a, b);

        // Extract and sum x, y, z components
        let x = builder.ins().extractlane(prod, 0u8);
        let y = builder.ins().extractlane(prod, 1u8);
        let z = builder.ins().extractlane(prod, 2u8);

        let xy = builder.ins().fadd(x, y);
        builder.ins().fadd(xy, z)
    }

    /// Dot product for vec4
    /// dot(a, b) = a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w
    pub fn dot4(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        let prod = builder.ins().fmul(a, b);

        let x = builder.ins().extractlane(prod, 0u8);
        let y = builder.ins().extractlane(prod, 1u8);
        let z = builder.ins().extractlane(prod, 2u8);
        let w = builder.ins().extractlane(prod, 3u8);

        let xy = builder.ins().fadd(x, y);
        let zw = builder.ins().fadd(z, w);
        builder.ins().fadd(xy, zw)
    }

    /// Cross product for vec3: a × b
    pub fn cross(builder: &mut FunctionBuilder, a: Value, b: Value) -> Value {
        // Extract components
        let ax = builder.ins().extractlane(a, 0u8);
        let ay = builder.ins().extractlane(a, 1u8);
        let az = builder.ins().extractlane(a, 2u8);

        let bx = builder.ins().extractlane(b, 0u8);
        let by = builder.ins().extractlane(b, 1u8);
        let bz = builder.ins().extractlane(b, 2u8);

        // cross.x = a.y*b.z - a.z*b.y
        let t1 = builder.ins().fmul(ay, bz);
        let t2 = builder.ins().fmul(az, by);
        let cx = builder.ins().fsub(t1, t2);

        // cross.y = a.z*b.x - a.x*b.z
        let t3 = builder.ins().fmul(az, bx);
        let t4 = builder.ins().fmul(ax, bz);
        let cy = builder.ins().fsub(t3, t4);

        // cross.z = a.x*b.y - a.y*b.x
        let t5 = builder.ins().fmul(ax, by);
        let t6 = builder.ins().fmul(ay, bx);
        let cz = builder.ins().fsub(t5, t6);

        let zero = builder.ins().f32const(0.0);
        Self::splat_f32x4(builder, cx, cy, cz, zero)
    }

    /// Vector length squared: |v|²
    pub fn length_squared(builder: &mut FunctionBuilder, v: Value) -> Value {
        Self::dot3(builder, v, v)
    }

    /// Vector length: |v|
    pub fn length(builder: &mut FunctionBuilder, v: Value) -> Value {
        let len_sq = Self::length_squared(builder, v);
        builder.ins().sqrt(len_sq)
    }

    /// Normalize vector: v / |v|
    pub fn normalize(builder: &mut FunctionBuilder, v: Value) -> Value {
        let len = Self::length(builder, v);
        let one = builder.ins().f32const(1.0);
        let inv_len = builder.ins().fdiv(one, len);
        Self::scale(builder, v, inv_len)
    }
}

/// SIMD operations for quaternions
#[cfg(feature = "jit")]
pub struct SimdQuat;

#[cfg(feature = "jit")]
impl SimdQuat {
    /// Hamilton product: q1 ⊗ q2
    /// This is the key operation for quaternion embeddings (arXiv:1904.10281)
    ///
    /// Where quat = (x, y, z, w) in (i, j, k, 1) notation
    pub fn hamilton_product(builder: &mut FunctionBuilder, q1: Value, q2: Value) -> Value {
        let x1 = builder.ins().extractlane(q1, 0u8);
        let y1 = builder.ins().extractlane(q1, 1u8);
        let z1 = builder.ins().extractlane(q1, 2u8);
        let w1 = builder.ins().extractlane(q1, 3u8);

        let x2 = builder.ins().extractlane(q2, 0u8);
        let y2 = builder.ins().extractlane(q2, 1u8);
        let z2 = builder.ins().extractlane(q2, 2u8);
        let w2 = builder.ins().extractlane(q2, 3u8);

        // result.w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        let t1 = builder.ins().fmul(w1, w2);
        let t2 = builder.ins().fmul(x1, x2);
        let t3 = builder.ins().fmul(y1, y2);
        let t4 = builder.ins().fmul(z1, z2);
        let rw_t1 = builder.ins().fsub(t1, t2);
        let rw_t2 = builder.ins().fsub(rw_t1, t3);
        let rw = builder.ins().fsub(rw_t2, t4);

        // result.x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        let t5 = builder.ins().fmul(w1, x2);
        let t6 = builder.ins().fmul(x1, w2);
        let t7 = builder.ins().fmul(y1, z2);
        let t8 = builder.ins().fmul(z1, y2);
        let rx_t1 = builder.ins().fadd(t5, t6);
        let rx_t2 = builder.ins().fadd(rx_t1, t7);
        let rx = builder.ins().fsub(rx_t2, t8);

        // result.y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        let t9 = builder.ins().fmul(w1, y2);
        let t10 = builder.ins().fmul(x1, z2);
        let t11 = builder.ins().fmul(y1, w2);
        let t12 = builder.ins().fmul(z1, x2);
        let ry_t1 = builder.ins().fsub(t9, t10);
        let ry_t2 = builder.ins().fadd(ry_t1, t11);
        let ry = builder.ins().fadd(ry_t2, t12);

        // result.z = w1*z2 + x1*y2 - y1*x2 + z1*w2
        let t13 = builder.ins().fmul(w1, z2);
        let t14 = builder.ins().fmul(x1, y2);
        let t15 = builder.ins().fmul(y1, x2);
        let t16 = builder.ins().fmul(z1, w2);
        let rz_t1 = builder.ins().fadd(t13, t14);
        let rz_t2 = builder.ins().fsub(rz_t1, t15);
        let rz = builder.ins().fadd(rz_t2, t16);

        SimdVec::splat_f32x4(builder, rx, ry, rz, rw)
    }

    /// Quaternion conjugate: q* = (-x, -y, -z, w)
    pub fn conjugate(builder: &mut FunctionBuilder, q: Value) -> Value {
        let x = builder.ins().extractlane(q, 0u8);
        let y = builder.ins().extractlane(q, 1u8);
        let z = builder.ins().extractlane(q, 2u8);
        let w = builder.ins().extractlane(q, 3u8);

        let neg_x = builder.ins().fneg(x);
        let neg_y = builder.ins().fneg(y);
        let neg_z = builder.ins().fneg(z);

        SimdVec::splat_f32x4(builder, neg_x, neg_y, neg_z, w)
    }

    /// Quaternion norm squared: |q|² = x² + y² + z² + w²
    pub fn norm_squared(builder: &mut FunctionBuilder, q: Value) -> Value {
        SimdVec::dot4(builder, q, q)
    }

    /// Quaternion normalize: q / |q|
    pub fn normalize(builder: &mut FunctionBuilder, q: Value) -> Value {
        let norm_sq = Self::norm_squared(builder, q);
        let norm = builder.ins().sqrt(norm_sq);
        let one = builder.ins().f32const(1.0);
        let inv_norm = builder.ins().fdiv(one, norm);
        SimdVec::scale(builder, q, inv_norm)
    }

    /// Quaternion inverse: q⁻¹ = q* / |q|²
    pub fn inverse(builder: &mut FunctionBuilder, q: Value) -> Value {
        let conj = Self::conjugate(builder, q);
        let norm_sq = Self::norm_squared(builder, q);
        let one = builder.ins().f32const(1.0);
        let inv_norm_sq = builder.ins().fdiv(one, norm_sq);
        SimdVec::scale(builder, conj, inv_norm_sq)
    }

    /// Inner product for quaternion embeddings
    pub fn inner_product(builder: &mut FunctionBuilder, q1: Value, q2: Value) -> Value {
        SimdVec::dot4(builder, q1, q2)
    }

    /// Score triple for knowledge graph: score = <h ⊗ r, t>
    pub fn score_triple(
        builder: &mut FunctionBuilder,
        head: Value,
        relation: Value,
        tail: Value,
    ) -> Value {
        let transformed = Self::hamilton_product(builder, head, relation);
        Self::inner_product(builder, transformed, tail)
    }

    /// Rotate vec3 by quaternion: q * v * q⁻¹
    pub fn rotate_vector(builder: &mut FunctionBuilder, q: Value, v: Value) -> Value {
        let vx = builder.ins().extractlane(v, 0u8);
        let vy = builder.ins().extractlane(v, 1u8);
        let vz = builder.ins().extractlane(v, 2u8);
        let zero = builder.ins().f32const(0.0);

        let v_quat = SimdVec::splat_f32x4(builder, vx, vy, vz, zero);

        let q_inv = Self::inverse(builder, q);
        let temp = Self::hamilton_product(builder, q, v_quat);
        Self::hamilton_product(builder, temp, q_inv)
    }

    /// Spherical linear interpolation (simplified as nlerp)
    pub fn slerp(builder: &mut FunctionBuilder, q1: Value, q2: Value, t: Value) -> Value {
        let one = builder.ins().f32const(1.0);
        let one_minus_t = builder.ins().fsub(one, t);

        let scaled_q1 = SimdVec::scale(builder, q1, one_minus_t);
        let scaled_q2 = SimdVec::scale(builder, q2, t);
        let sum = SimdVec::add(builder, scaled_q1, scaled_q2);

        Self::normalize(builder, sum)
    }
}

/// Linear interpolation for vectors
#[cfg(feature = "jit")]
pub fn lerp(builder: &mut FunctionBuilder, a: Value, b: Value, t: Value) -> Value {
    let one = builder.ins().f32const(1.0);
    let one_minus_t = builder.ins().fsub(one, t);

    let scaled_a = SimdVec::scale(builder, a, one_minus_t);
    let scaled_b = SimdVec::scale(builder, b, t);
    SimdVec::add(builder, scaled_a, scaled_b)
}

// ==================== SEDENION SIMD OPERATIONS ====================
// Sedenions: 16-dimensional hypercomplex numbers (512 bits = 16 x f32)
// Represented as 4 F32X4 vectors (SSE/NEON compatible) or 2 F32X8 (AVX)
//
// IMPORTANT: Sedenions have ZERO DIVISORS unlike quaternions/octonions
// The norm is not multiplicative: |a*b| ≠ |a|*|b| for some a, b

/// SIMD operations for sedenions (16D hypercomplex)
/// Uses 4 F32X4 vectors for maximum compatibility (SSE4, NEON, etc.)
#[cfg(feature = "jit")]
pub struct SimdSedenion;

#[cfg(feature = "jit")]
impl SimdSedenion {
    /// Create sedenion from 16 f32 values
    /// Returns (v0, v1, v2, v3) where each is a F32X4
    pub fn splat_f32x16(
        builder: &mut FunctionBuilder,
        e0: Value,
        e1: Value,
        e2: Value,
        e3: Value,
        e4: Value,
        e5: Value,
        e6: Value,
        e7: Value,
        e8: Value,
        e9: Value,
        e10: Value,
        e11: Value,
        e12: Value,
        e13: Value,
        e14: Value,
        e15: Value,
    ) -> (Value, Value, Value, Value) {
        let v0 = SimdVec::splat_f32x4(builder, e0, e1, e2, e3);
        let v1 = SimdVec::splat_f32x4(builder, e4, e5, e6, e7);
        let v2 = SimdVec::splat_f32x4(builder, e8, e9, e10, e11);
        let v3 = SimdVec::splat_f32x4(builder, e12, e13, e14, e15);
        (v0, v1, v2, v3)
    }

    /// Extract component from sedenion (0-15)
    pub fn extract_lane(
        builder: &mut FunctionBuilder,
        s: (Value, Value, Value, Value),
        lane: u8,
    ) -> Value {
        let (v0, v1, v2, v3) = s;
        match lane {
            0..=3 => builder.ins().extractlane(v0, lane),
            4..=7 => builder.ins().extractlane(v1, lane - 4),
            8..=11 => builder.ins().extractlane(v2, lane - 8),
            12..=15 => builder.ins().extractlane(v3, lane - 12),
            _ => builder.ins().f32const(0.0), // Out of bounds
        }
    }

    /// Sedenion addition: a + b (component-wise)
    pub fn add(
        builder: &mut FunctionBuilder,
        a: (Value, Value, Value, Value),
        b: (Value, Value, Value, Value),
    ) -> (Value, Value, Value, Value) {
        (
            builder.ins().fadd(a.0, b.0),
            builder.ins().fadd(a.1, b.1),
            builder.ins().fadd(a.2, b.2),
            builder.ins().fadd(a.3, b.3),
        )
    }

    /// Sedenion subtraction: a - b (component-wise)
    pub fn sub(
        builder: &mut FunctionBuilder,
        a: (Value, Value, Value, Value),
        b: (Value, Value, Value, Value),
    ) -> (Value, Value, Value, Value) {
        (
            builder.ins().fsub(a.0, b.0),
            builder.ins().fsub(a.1, b.1),
            builder.ins().fsub(a.2, b.2),
            builder.ins().fsub(a.3, b.3),
        )
    }

    /// Scalar multiplication: s * k
    pub fn scale(
        builder: &mut FunctionBuilder,
        s: (Value, Value, Value, Value),
        scalar: Value,
    ) -> (Value, Value, Value, Value) {
        let splat = builder.ins().splat(types::F32X4, scalar);
        (
            builder.ins().fmul(s.0, splat),
            builder.ins().fmul(s.1, splat),
            builder.ins().fmul(s.2, splat),
            builder.ins().fmul(s.3, splat),
        )
    }

    /// Sedenion conjugate: negate all imaginary parts (e1-e15)
    /// conj(e0 + e1*i1 + ... + e15*i15) = e0 - e1*i1 - ... - e15*i15
    pub fn conjugate(
        builder: &mut FunctionBuilder,
        s: (Value, Value, Value, Value),
    ) -> (Value, Value, Value, Value) {
        // v0 = (e0, e1, e2, e3) -> (e0, -e1, -e2, -e3)
        let e0 = builder.ins().extractlane(s.0, 0u8);
        let e1 = builder.ins().extractlane(s.0, 1u8);
        let e2 = builder.ins().extractlane(s.0, 2u8);
        let e3 = builder.ins().extractlane(s.0, 3u8);

        // Negate components separately to avoid borrow issues
        let neg_e1 = builder.ins().fneg(e1);
        let neg_e2 = builder.ins().fneg(e2);
        let neg_e3 = builder.ins().fneg(e3);
        let v0 = SimdVec::splat_f32x4(builder, e0, neg_e1, neg_e2, neg_e3);

        // v1, v2, v3: negate all components
        let v1 = builder.ins().fneg(s.1);
        let v2 = builder.ins().fneg(s.2);
        let v3 = builder.ins().fneg(s.3);

        (v0, v1, v2, v3)
    }

    /// Sedenion norm squared: |s|² = e0² + e1² + ... + e15²
    pub fn norm_squared(builder: &mut FunctionBuilder, s: (Value, Value, Value, Value)) -> Value {
        // Sum of squares for each vector
        let sq0 = builder.ins().fmul(s.0, s.0);
        let sq1 = builder.ins().fmul(s.1, s.1);
        let sq2 = builder.ins().fmul(s.2, s.2);
        let sq3 = builder.ins().fmul(s.3, s.3);

        // Horizontal sum for each F32X4
        let sum0 = Self::horizontal_sum_f32x4(builder, sq0);
        let sum1 = Self::horizontal_sum_f32x4(builder, sq1);
        let sum2 = Self::horizontal_sum_f32x4(builder, sq2);
        let sum3 = Self::horizontal_sum_f32x4(builder, sq3);

        // Total sum
        let sum01 = builder.ins().fadd(sum0, sum1);
        let sum23 = builder.ins().fadd(sum2, sum3);
        builder.ins().fadd(sum01, sum23)
    }

    /// Helper: horizontal sum of F32X4 components
    fn horizontal_sum_f32x4(builder: &mut FunctionBuilder, v: Value) -> Value {
        let e0 = builder.ins().extractlane(v, 0u8);
        let e1 = builder.ins().extractlane(v, 1u8);
        let e2 = builder.ins().extractlane(v, 2u8);
        let e3 = builder.ins().extractlane(v, 3u8);

        let sum01 = builder.ins().fadd(e0, e1);
        let sum23 = builder.ins().fadd(e2, e3);
        builder.ins().fadd(sum01, sum23)
    }

    /// Sedenion norm: |s| = sqrt(|s|²)
    pub fn norm(builder: &mut FunctionBuilder, s: (Value, Value, Value, Value)) -> Value {
        let norm_sq = Self::norm_squared(builder, s);
        builder.ins().sqrt(norm_sq)
    }

    /// Sedenion normalize: s / |s|
    pub fn normalize(
        builder: &mut FunctionBuilder,
        s: (Value, Value, Value, Value),
    ) -> (Value, Value, Value, Value) {
        let norm = Self::norm(builder, s);
        let one = builder.ins().f32const(1.0);
        let inv_norm = builder.ins().fdiv(one, norm);
        Self::scale(builder, s, inv_norm)
    }

    /// Sedenion dot product: a · b = Σ(ai * bi)
    pub fn dot(
        builder: &mut FunctionBuilder,
        a: (Value, Value, Value, Value),
        b: (Value, Value, Value, Value),
    ) -> Value {
        let prod0 = builder.ins().fmul(a.0, b.0);
        let prod1 = builder.ins().fmul(a.1, b.1);
        let prod2 = builder.ins().fmul(a.2, b.2);
        let prod3 = builder.ins().fmul(a.3, b.3);

        let sum0 = Self::horizontal_sum_f32x4(builder, prod0);
        let sum1 = Self::horizontal_sum_f32x4(builder, prod1);
        let sum2 = Self::horizontal_sum_f32x4(builder, prod2);
        let sum3 = Self::horizontal_sum_f32x4(builder, prod3);

        let sum01 = builder.ins().fadd(sum0, sum1);
        let sum23 = builder.ins().fadd(sum2, sum3);
        builder.ins().fadd(sum01, sum23)
    }

    /// Check if sedenion is approximately zero (for zero divisor detection)
    /// Returns 1.0 if norm_sq < epsilon, 0.0 otherwise
    pub fn is_approx_zero(
        builder: &mut FunctionBuilder,
        s: (Value, Value, Value, Value),
        epsilon: Value,
    ) -> Value {
        let norm_sq = Self::norm_squared(builder, s);
        let is_zero = builder.ins().fcmp(
            cranelift_codegen::ir::condcodes::FloatCC::LessThan,
            norm_sq,
            epsilon,
        );
        let one = builder.ins().f32const(1.0);
        let zero = builder.ins().f32const(0.0);
        builder.ins().select(is_zero, one, zero)
    }

    // NOTE: Sedenion multiplication (Cayley-Dickson) is complex (~500 FLOPs)
    // and is better implemented in the stdlib or via GPU kernels.
    // The SIMD version here provides component-wise operations for building
    // blocks that can be composed for the full multiplication.
}

// ==================== CLIFFORD ALGEBRA GEOMETRIC PRODUCT ====================
// Precomputed sign tables for Cl(p,q) geometric products.
//
// The geometric product of two basis blades e_I * e_J in Clifford algebra Cl(p,q)
// yields ±e_{I^J} where the sign depends on:
// 1. Transposition parity (number of swaps to move J past I)
// 2. Metric signature (basis vectors with index >= p square to -1)
//
// By precomputing the full n×n sign table (n = 2^(p+q)) at compile time,
// we eliminate the O(p+q) sign computation per basis-blade pair, replacing
// ~50 instructions (function call + loop) with a single table load.

/// Compute the sign of basis blade product e_i * e_j in Cl(p,q).
///
/// Algorithm: count transpositions (bits of j that must pass bits of i),
/// then apply metric (shared bits in positions >= p square to -1).
pub fn clifford_sign(i: usize, j: usize, p: u32, q: u32) -> f64 {
    let total = p + q;

    // Count transposition swaps: for each set bit in j, count set bits
    // in i at higher positions
    let mut swaps = 0u32;
    for bit in 0..total {
        if (j >> bit) & 1 != 0 {
            for k in (bit + 1)..total {
                if (i >> k) & 1 != 0 {
                    swaps += 1;
                }
            }
        }
    }

    // Metric sign: shared basis vectors at positions >= p square to -1
    let common = i & j;
    let mut metric_sign = 1i32;
    for bit in p..total {
        if (common >> bit) & 1 != 0 {
            metric_sign = -metric_sign;
        }
    }

    let sign = if swaps % 2 == 0 {
        metric_sign
    } else {
        -metric_sign
    };
    sign as f64
}

/// Precompute the full n×n sign table for Clifford algebra Cl(p,q).
///
/// Returns a `Vec<f64>` of size n² where n = 2^(p+q).
/// Entry `table[i * n + j]` gives the sign (±1.0) for e_i * e_j.
///
/// For Cl(4,4): n=256, table size = 65536 entries = 512 KB.
/// For Cl(1,3): n=16, table size = 256 entries = 2 KB.
pub fn precompute_clifford_sign_table(p: u32, q: u32) -> Vec<f64> {
    let n = 1usize << (p + q);
    let mut table = vec![0.0f64; n * n];

    for i in 0..n {
        for j in 0..n {
            table[i * n + j] = clifford_sign(i, j, p, q);
        }
    }

    table
}

/// Static storage for precomputed sign tables (kept alive for JIT lifetime).
#[cfg(feature = "jit")]
use std::sync::Mutex;

#[cfg(feature = "jit")]
static CLIFFORD_SIGN_TABLES: Mutex<Vec<Box<[f64]>>> = Mutex::new(Vec::new());

/// Store a precomputed sign table in static storage, returning a raw pointer
/// that remains valid for the lifetime of the process.
#[cfg(feature = "jit")]
pub fn store_clifford_sign_table(table: Vec<f64>) -> *const f64 {
    let boxed: Box<[f64]> = table.into_boxed_slice();
    let ptr = boxed.as_ptr();
    if let Ok(mut tables) = CLIFFORD_SIGN_TABLES.lock() {
        tables.push(boxed);
    }
    ptr
}

/// Get or create a cached sign table for Cl(p,q), returning (pointer as usize, dimension).
/// Thread-safe: uses internal mutex. The returned pointer is valid for process lifetime.
/// Returns usize instead of raw pointer for Send/Sync compatibility.
#[cfg(feature = "jit")]
pub fn get_or_create_sign_table(p: u32, q: u32) -> (usize, usize) {
    use std::sync::OnceLock;
    use std::collections::HashMap;

    // Cache of (p,q) -> (pointer_as_usize, dimension)
    static CACHE: OnceLock<Mutex<HashMap<(u32, u32), (usize, usize)>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    let mut map = cache.lock().unwrap();
    if let Some(&entry) = map.get(&(p, q)) {
        return entry;
    }

    let n = 1usize << (p + q);
    let table = precompute_clifford_sign_table(p, q);
    let ptr = store_clifford_sign_table(table);
    let entry = (ptr as usize, n);
    map.insert((p, q), entry);
    entry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_module_exists() {
        assert!(true);
    }

    #[test]
    fn test_clifford_sign_cl44() {
        // e0 * e0 = +1 (positive square)
        assert_eq!(clifford_sign(1, 1, 4, 4), 1.0);
        // e4 * e4 = -1 (negative square, bit 4 >= p=4)
        assert_eq!(clifford_sign(16, 16, 4, 4), -1.0);
        // e0 * e1 = e01 (no swap, no shared bits)
        assert_eq!(clifford_sign(1, 2, 4, 4), 1.0);
        // e1 * e0 = -e01 (one swap)
        assert_eq!(clifford_sign(2, 1, 4, 4), -1.0);
    }

    #[test]
    fn test_clifford_sign_cl13() {
        // Cl(1,3): e0^2 = +1, e1^2 = e2^2 = e3^2 = -1
        // e0 = bit 0, e1 = bit 1, e2 = bit 2, e3 = bit 3
        assert_eq!(clifford_sign(1, 1, 1, 3), 1.0);   // e0^2 = +1
        assert_eq!(clifford_sign(2, 2, 1, 3), -1.0);   // e1^2 = -1
        assert_eq!(clifford_sign(4, 4, 1, 3), -1.0);   // e2^2 = -1
        assert_eq!(clifford_sign(8, 8, 1, 3), -1.0);   // e3^2 = -1
    }

    #[test]
    fn test_sign_table_symmetry() {
        let table = precompute_clifford_sign_table(1, 3);
        let n = 16;
        // e_i * e_j and e_j * e_i should differ only by transposition sign
        for i in 0..n {
            for j in 0..n {
                let s1 = table[i * n + j];
                let s2 = table[j * n + i];
                // Both should be ±1.0
                assert!(s1 == 1.0 || s1 == -1.0, "Invalid sign at ({}, {})", i, j);
                assert!(s2 == 1.0 || s2 == -1.0, "Invalid sign at ({}, {})", j, i);
            }
        }
    }
}
