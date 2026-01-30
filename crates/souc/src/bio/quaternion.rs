//! Quaternion Operations for Biological Transmission
//!
//! Implements unit quaternions for SU(2) representations of transmission channels
//! from the preprint. Unit quaternions satisfy |q| = 1 and form a group under
//! multiplication (noncommutative).
//!
//! # Mathematical Background
//!
//! A quaternion q = w + xi + yj + zk where:
//! - w is the scalar (real) part
//! - (x, y, z) is the vector (imaginary) part
//! - i² = j² = k² = ijk = -1
//!
//! Multiplication is noncommutative:
//! - ij = k, ji = -k
//! - jk = i, kj = -i
//! - ki = j, ik = -j
//!
//! # Unit Quaternions (SU(2))
//!
//! Unit quaternions |q| = 1 form SU(2), double-covering SO(3).
//! They are used for norm-preserving transmission channels:
//!
//! ```text
//! q = g + ti + pj + ek
//! |q|² = g² + t² + p² + e² = 1
//! ```
//!
//! Where (g, t, p, e) represent (genetic, transcriptional, post-transcriptional, epigenetic)
//! channel weights.

use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};

/// Quaternion with f64 components
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    /// Scalar part (w)
    pub w: f64,
    /// i component (x)
    pub x: f64,
    /// j component (y)
    pub y: f64,
    /// k component (z)
    pub z: f64,
}

impl Quaternion {
    /// Create new quaternion
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { w, x, y, z }
    }

    /// Create zero quaternion
    pub const fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Create identity quaternion (1 + 0i + 0j + 0k)
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Create pure imaginary quaternion (0 + xi + yj + zk)
    pub const fn pure(x: f64, y: f64, z: f64) -> Self {
        Self::new(0.0, x, y, z)
    }

    /// Create quaternion i
    pub const fn i() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Create quaternion j
    pub const fn j() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0)
    }

    /// Create quaternion k
    pub const fn k() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Get squared norm |q|²
    pub fn norm_squared(&self) -> f64 {
        self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Get norm |q|
    pub fn norm(&self) -> f64 {
        self.norm_squared().sqrt()
    }

    /// Check if unit quaternion (|q| ≈ 1)
    pub fn is_unit(&self, tolerance: f64) -> bool {
        (self.norm() - 1.0).abs() < tolerance
    }

    /// Conjugate: q* = w - xi - yj - zk
    pub fn conjugate(&self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// Inverse: q⁻¹ = q* / |q|²
    pub fn inverse(&self) -> Option<Self> {
        let norm_sq = self.norm_squared();
        if norm_sq < 1e-15 {
            return None;
        }
        let conj = self.conjugate();
        Some(Self::new(
            conj.w / norm_sq,
            conj.x / norm_sq,
            conj.y / norm_sq,
            conj.z / norm_sq,
        ))
    }

    /// Normalize to unit quaternion
    pub fn normalize(&self) -> Option<Self> {
        let n = self.norm();
        if n < 1e-15 {
            return None;
        }
        Some(Self::new(self.w / n, self.x / n, self.y / n, self.z / n))
    }

    /// Scalar part
    pub fn scalar(&self) -> f64 {
        self.w
    }

    /// Vector part as tuple
    pub fn vector(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    /// Dot product (Euclidean inner product)
    pub fn dot(&self, other: &Self) -> f64 {
        self.w * other.w + self.x * other.x + self.y * other.y + self.z * other.z
    }

    /// Scale by scalar
    pub fn scale(&self, s: f64) -> Self {
        Self::new(self.w * s, self.x * s, self.y * s, self.z * s)
    }
}

impl Add for Quaternion {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self::new(
            self.w + other.w,
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )
    }
}

impl Sub for Quaternion {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self::new(
            self.w - other.w,
            self.x - other.x,
            self.y - other.y,
            self.z - other.z,
        )
    }
}

impl Neg for Quaternion {
    type Output = Self;

    fn neg(self) -> Self {
        Self::new(-self.w, -self.x, -self.y, -self.z)
    }
}

/// Hamilton product (quaternion multiplication)
///
/// q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2)
///         + (w1x2 + x1w2 + y1z2 - z1y2)i
///         + (w1y2 - x1z2 + y1w2 + z1x2)j
///         + (w1z2 + x1y2 - y1x2 + z1w2)k
impl Mul for Quaternion {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        Self::new(
            self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        )
    }
}

impl fmt::Display for Quaternion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} + {}i + {}j + {}k", self.w, self.x, self.y, self.z)
    }
}

/// Unit quaternion (constrained |q| = 1)
///
/// This is the refined type for SU(2) elements used in transmission dynamics.
/// Invariant: norm is always 1.0 (within floating-point tolerance)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UnitQuat {
    inner: Quaternion,
}

impl UnitQuat {
    /// Create unit quaternion from components (normalizes automatically)
    pub fn new(w: f64, x: f64, y: f64, z: f64) -> Option<Self> {
        Quaternion::new(w, x, y, z)
            .normalize()
            .map(|q| Self { inner: q })
    }

    /// Create from quaternion (normalizes)
    pub fn from_quaternion(q: Quaternion) -> Option<Self> {
        q.normalize().map(|q| Self { inner: q })
    }

    /// Create identity unit quaternion
    pub fn identity() -> Self {
        Self {
            inner: Quaternion::identity(),
        }
    }

    /// Create from axis-angle representation
    pub fn from_axis_angle(axis: (f64, f64, f64), angle: f64) -> Option<Self> {
        let (ax, ay, az) = axis;
        let len = (ax * ax + ay * ay + az * az).sqrt();
        if len < 1e-15 {
            return None;
        }
        let half_angle = angle / 2.0;
        let s = half_angle.sin() / len;
        Self::new(half_angle.cos(), ax * s, ay * s, az * s)
    }

    /// Get inner quaternion
    pub fn as_quaternion(&self) -> &Quaternion {
        &self.inner
    }

    /// Get norm (always 1.0)
    pub fn norm(&self) -> f64 {
        self.inner.norm()
    }

    /// Conjugate (inverse for unit quaternions)
    pub fn conjugate(&self) -> Self {
        Self {
            inner: self.inner.conjugate(),
        }
    }

    /// Inverse (same as conjugate for unit quaternions)
    pub fn inverse(&self) -> Self {
        self.conjugate()
    }

    /// Multiply two unit quaternions (result is unit quaternion)
    pub fn mul(&self, other: &Self) -> Self {
        // Product of unit quaternions is unit quaternion
        Self {
            inner: self.inner * other.inner,
        }
    }

    /// Spherical linear interpolation (SLERP)
    pub fn slerp(&self, other: &Self, t: f64) -> Self {
        let mut dot = self.inner.dot(&other.inner);

        // If dot is negative, negate one quaternion to take shorter path
        let other_inner = if dot < 0.0 {
            dot = -dot;
            -other.inner
        } else {
            other.inner
        };

        // If quaternions are very close, use linear interpolation
        if dot > 0.9995 {
            let result = Quaternion::new(
                self.inner.w + t * (other_inner.w - self.inner.w),
                self.inner.x + t * (other_inner.x - self.inner.x),
                self.inner.y + t * (other_inner.y - self.inner.y),
                self.inner.z + t * (other_inner.z - self.inner.z),
            );
            return Self::from_quaternion(result).unwrap_or(Self::identity());
        }

        // Standard SLERP
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let s0 = ((1.0 - t) * theta).sin() / sin_theta;
        let s1 = (t * theta).sin() / sin_theta;

        Self {
            inner: self.inner.scale(s0) + other_inner.scale(s1),
        }
    }
}

impl fmt::Display for UnitQuat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.inner)
    }
}

/// Quaternion operations trait for extensibility
pub trait QuatOps {
    /// Rotate a 3D vector by this quaternion
    fn rotate_vector(&self, v: (f64, f64, f64)) -> (f64, f64, f64);

    /// Convert to rotation matrix (3x3)
    fn to_rotation_matrix(&self) -> [[f64; 3]; 3];
}

impl QuatOps for UnitQuat {
    fn rotate_vector(&self, v: (f64, f64, f64)) -> (f64, f64, f64) {
        let p = Quaternion::pure(v.0, v.1, v.2);
        let rotated = self.inner * p * self.inner.conjugate();
        (rotated.x, rotated.y, rotated.z)
    }

    fn to_rotation_matrix(&self) -> [[f64; 3]; 3] {
        let q = &self.inner;
        let (w, x, y, z) = (q.w, q.x, q.y, q.z);

        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_quaternion_multiplication() {
        // i * j = k
        let i = Quaternion::i();
        let j = Quaternion::j();
        let k = Quaternion::k();

        let ij = i * j;
        assert!((ij.z - 1.0).abs() < EPSILON);

        // j * i = -k
        let ji = j * i;
        assert!((ji.z + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_quaternion_noncommutativity() {
        let i = Quaternion::i();
        let j = Quaternion::j();

        let ij = i * j;
        let ji = j * i;

        assert!((ij.z - 1.0).abs() < EPSILON); // ij = k
        assert!((ji.z + 1.0).abs() < EPSILON); // ji = -k
        assert_ne!(ij, ji);
    }

    #[test]
    fn test_unit_quaternion_norm() {
        let q = UnitQuat::new(1.0, 2.0, 3.0, 4.0).unwrap();
        assert!((q.norm() - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_unit_quaternion_inverse() {
        let q = UnitQuat::new(0.5, 0.5, 0.5, 0.5).unwrap();
        let q_inv = q.inverse();
        let product = q.mul(&q_inv);

        // q * q^(-1) = 1
        assert!((product.as_quaternion().w - 1.0).abs() < EPSILON);
        assert!(product.as_quaternion().x.abs() < EPSILON);
        assert!(product.as_quaternion().y.abs() < EPSILON);
        assert!(product.as_quaternion().z.abs() < EPSILON);
    }

    #[test]
    fn test_rotation() {
        // 180° rotation around z-axis
        let q = UnitQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::PI).unwrap();

        // Rotate (1, 0, 0) -> (-1, 0, 0)
        let v = (1.0, 0.0, 0.0);
        let rotated = q.rotate_vector(v);

        assert!((rotated.0 + 1.0).abs() < EPSILON);
        assert!(rotated.1.abs() < EPSILON);
        assert!(rotated.2.abs() < EPSILON);
    }

    #[test]
    fn test_slerp() {
        let q1 = UnitQuat::identity();
        let q2 = UnitQuat::from_axis_angle((0.0, 0.0, 1.0), std::f64::consts::PI / 2.0).unwrap();

        // Midpoint interpolation
        let mid = q1.slerp(&q2, 0.5);

        // Should be 45° rotation
        let v = (1.0, 0.0, 0.0);
        let rotated = mid.rotate_vector(v);

        // x and y should be equal (45°)
        assert!((rotated.0 - rotated.1).abs() < 0.01);
    }
}
