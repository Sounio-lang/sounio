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

#[cfg(test)]
mod tests {
    #[test]
    fn test_simd_module_exists() {
        assert!(true);
    }
}
