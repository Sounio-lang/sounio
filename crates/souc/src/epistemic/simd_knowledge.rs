//! SIMD-Accelerated Epistemic Knowledge Operations
//!
//! This module provides vectorized operations for Knowledge<T> values,
//! enabling high-performance uncertainty propagation using NEON (ARM) or AVX2 (x86).
//!
//! # Key Types
//!
//! - `KnowledgeVec4`: 4 Knowledge<f32> values with SIMD-accelerated operations
//! - `KnowledgeVec8`: 8 Knowledge<f32> values (AVX2 only, falls back to 2x Vec4)
//!
//! # Uncertainty Propagation Formulas
//!
//! - Addition: sigma_out = sqrt(sigma_a^2 + sigma_b^2)
//! - Multiplication: sigma_out = sqrt((a * sigma_b)^2 + (b * sigma_a)^2)
//! - Linear combination: sigma_out = sqrt(sum(w_i^2 * sigma_i^2))

#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// ============================================================================
// SIMD PRIMITIVE OPERATIONS (128-bit, 4x f32)
// ============================================================================

/// Add two f32x4 vectors
#[inline(always)]
pub fn simd_add_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            return unsafe { add_x86(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { add_arm(a, b) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn add_x86(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let res = _mm_add_ps(va, vb);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), res);
    r
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_arm(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = vld1q_f32(a.as_ptr());
    let vb = vld1q_f32(b.as_ptr());
    let res = vaddq_f32(va, vb);
    let mut r = [0.0f32; 4];
    vst1q_f32(r.as_mut_ptr(), res);
    r
}

/// Subtract two f32x4 vectors
#[inline(always)]
pub fn simd_sub_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            return unsafe { sub_x86(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sub_arm(a, b) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    [a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn sub_x86(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let res = _mm_sub_ps(va, vb);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), res);
    r
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sub_arm(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = vld1q_f32(a.as_ptr());
    let vb = vld1q_f32(b.as_ptr());
    let res = vsubq_f32(va, vb);
    let mut r = [0.0f32; 4];
    vst1q_f32(r.as_mut_ptr(), res);
    r
}

/// Multiply two f32x4 vectors
#[inline(always)]
pub fn simd_mul_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            return unsafe { mul_x86(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { mul_arm(a, b) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    [a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn mul_x86(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let res = _mm_mul_ps(va, vb);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), res);
    r
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn mul_arm(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = vld1q_f32(a.as_ptr());
    let vb = vld1q_f32(b.as_ptr());
    let res = vmulq_f32(va, vb);
    let mut r = [0.0f32; 4];
    vst1q_f32(r.as_mut_ptr(), res);
    r
}

/// Square root of f32x4 vector
#[inline(always)]
pub fn simd_sqrt_f32x4(a: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            return unsafe { sqrt_x86(a) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sqrt_arm(a) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()]
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    [a[0].sqrt(), a[1].sqrt(), a[2].sqrt(), a[3].sqrt()]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn sqrt_x86(a: [f32; 4]) -> [f32; 4] {
    let va = _mm_loadu_ps(a.as_ptr());
    let res = _mm_sqrt_ps(va);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), res);
    r
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqrt_arm(a: [f32; 4]) -> [f32; 4] {
    let va = vld1q_f32(a.as_ptr());
    let res = vsqrtq_f32(va);
    let mut r = [0.0f32; 4];
    vst1q_f32(r.as_mut_ptr(), res);
    r
}

/// Element-wise minimum of two f32x4 vectors
#[inline(always)]
pub fn simd_min_f32x4(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse") {
            return unsafe { min_x86(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { min_arm(a, b) };
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        [
            a[0].min(b[0]),
            a[1].min(b[1]),
            a[2].min(b[2]),
            a[3].min(b[3]),
        ]
    }
    #[cfg(all(target_arch = "x86_64", not(target_arch = "aarch64")))]
    [
        a[0].min(b[0]),
        a[1].min(b[1]),
        a[2].min(b[2]),
        a[3].min(b[3]),
    ]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse")]
unsafe fn min_x86(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = _mm_loadu_ps(a.as_ptr());
    let vb = _mm_loadu_ps(b.as_ptr());
    let res = _mm_min_ps(va, vb);
    let mut r = [0.0f32; 4];
    _mm_storeu_ps(r.as_mut_ptr(), res);
    r
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn min_arm(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    let va = vld1q_f32(a.as_ptr());
    let vb = vld1q_f32(b.as_ptr());
    let res = vminq_f32(va, vb);
    let mut r = [0.0f32; 4];
    vst1q_f32(r.as_mut_ptr(), res);
    r
}

/// Horizontal sum of f32x4 vector
#[inline(always)]
pub fn simd_horizontal_sum_f32x4(v: [f32; 4]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse3") {
            return unsafe { hsum_x86(v) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { hsum_arm(v) };
    }
    v[0] + v[1] + v[2] + v[3]
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse3")]
unsafe fn hsum_x86(v: [f32; 4]) -> f32 {
    let vv = _mm_loadu_ps(v.as_ptr());
    let sum1 = _mm_hadd_ps(vv, vv);
    let sum2 = _mm_hadd_ps(sum1, sum1);
    _mm_cvtss_f32(sum2)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hsum_arm(v: [f32; 4]) -> f32 {
    let vv = vld1q_f32(v.as_ptr());
    vaddvq_f32(vv)
}

/// Horizontal minimum of f32x4 vector
#[inline(always)]
pub fn simd_horizontal_min_f32x4(v: [f32; 4]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { hmin_arm(v) };
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        v[0].min(v[1]).min(v[2]).min(v[3])
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn hmin_arm(v: [f32; 4]) -> f32 {
    let vv = vld1q_f32(v.as_ptr());
    vminvq_f32(vv)
}

/// Horizontal product of f32x4 vector
#[inline(always)]
pub fn simd_horizontal_product_f32x4(v: [f32; 4]) -> f32 {
    // No direct SIMD instruction for this, use scalar
    v[0] * v[1] * v[2] * v[3]
}

// ============================================================================
// KNOWLEDGE VEC4 - 4 Knowledge<f32> values
// ============================================================================

/// Four Knowledge<f32> values with SIMD-accelerated operations.
///
/// Each knowledge value consists of:
/// - `values`: The actual f32 values
/// - `uncertainties`: Standard deviations (sigma) for each value
/// - `confidences`: Confidence levels (0.0-1.0) for each value
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnowledgeVec4 {
    /// The central values
    pub values: [f32; 4],
    /// Standard deviations (uncertainties)
    pub uncertainties: [f32; 4],
    /// Confidence levels (0.0 to 1.0)
    pub confidences: [f32; 4],
}

impl KnowledgeVec4 {
    /// Create a new KnowledgeVec4 with given values, uncertainties, and confidences
    #[inline]
    pub fn new(values: [f32; 4], uncertainties: [f32; 4], confidences: [f32; 4]) -> Self {
        Self {
            values,
            uncertainties,
            confidences,
        }
    }

    /// Create from certain values (zero uncertainty, full confidence)
    #[inline]
    pub fn certain(values: [f32; 4]) -> Self {
        Self {
            values,
            uncertainties: [0.0; 4],
            confidences: [1.0; 4],
        }
    }

    /// Create from values with uniform uncertainty and confidence
    #[inline]
    pub fn uniform(values: [f32; 4], uncertainty: f32, confidence: f32) -> Self {
        Self {
            values,
            uncertainties: [uncertainty; 4],
            confidences: [confidence; 4],
        }
    }

    /// Create a zeroed KnowledgeVec4
    #[inline]
    pub fn zero() -> Self {
        Self {
            values: [0.0; 4],
            uncertainties: [0.0; 4],
            confidences: [1.0; 4],
        }
    }

    /// Get the relative uncertainties (sigma / |value|)
    #[inline]
    pub fn relative_uncertainties(&self) -> [f32; 4] {
        let abs_values = [
            self.values[0].abs().max(f32::EPSILON),
            self.values[1].abs().max(f32::EPSILON),
            self.values[2].abs().max(f32::EPSILON),
            self.values[3].abs().max(f32::EPSILON),
        ];
        [
            self.uncertainties[0] / abs_values[0],
            self.uncertainties[1] / abs_values[1],
            self.uncertainties[2] / abs_values[2],
            self.uncertainties[3] / abs_values[3],
        ]
    }

    /// Add two KnowledgeVec4 with uncertainty propagation.
    ///
    /// For independent random variables:
    /// - result.value = a.value + b.value
    /// - result.sigma = sqrt(a.sigma^2 + b.sigma^2)
    /// - result.confidence = min(a.confidence, b.confidence)
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        let values = simd_add_f32x4(self.values, other.values);

        // Uncertainty propagation: sqrt(sigma_a^2 + sigma_b^2)
        let sigma_a_sq = simd_mul_f32x4(self.uncertainties, self.uncertainties);
        let sigma_b_sq = simd_mul_f32x4(other.uncertainties, other.uncertainties);
        let sum_sq = simd_add_f32x4(sigma_a_sq, sigma_b_sq);
        let uncertainties = simd_sqrt_f32x4(sum_sq);

        // Confidence: take minimum (conservative)
        let confidences = simd_min_f32x4(self.confidences, other.confidences);

        Self {
            values,
            uncertainties,
            confidences,
        }
    }

    /// Subtract two KnowledgeVec4 with uncertainty propagation.
    ///
    /// Same uncertainty propagation as addition.
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        let values = simd_sub_f32x4(self.values, other.values);

        // Uncertainty propagation: sqrt(sigma_a^2 + sigma_b^2)
        let sigma_a_sq = simd_mul_f32x4(self.uncertainties, self.uncertainties);
        let sigma_b_sq = simd_mul_f32x4(other.uncertainties, other.uncertainties);
        let sum_sq = simd_add_f32x4(sigma_a_sq, sigma_b_sq);
        let uncertainties = simd_sqrt_f32x4(sum_sq);

        let confidences = simd_min_f32x4(self.confidences, other.confidences);

        Self {
            values,
            uncertainties,
            confidences,
        }
    }

    /// Multiply two KnowledgeVec4 with uncertainty propagation.
    ///
    /// For z = a * b:
    /// - sigma_z = sqrt((a * sigma_b)^2 + (b * sigma_a)^2)
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        let values = simd_mul_f32x4(self.values, other.values);

        // a * sigma_b
        let a_sigma_b = simd_mul_f32x4(self.values, other.uncertainties);
        let a_sigma_b_sq = simd_mul_f32x4(a_sigma_b, a_sigma_b);

        // b * sigma_a
        let b_sigma_a = simd_mul_f32x4(other.values, self.uncertainties);
        let b_sigma_a_sq = simd_mul_f32x4(b_sigma_a, b_sigma_a);

        let sum_sq = simd_add_f32x4(a_sigma_b_sq, b_sigma_a_sq);
        let uncertainties = simd_sqrt_f32x4(sum_sq);

        let confidences = simd_min_f32x4(self.confidences, other.confidences);

        Self {
            values,
            uncertainties,
            confidences,
        }
    }

    /// Scale by a certain scalar value
    #[inline]
    pub fn scale(&self, scalar: f32) -> Self {
        let scalar_vec = [scalar; 4];
        let values = simd_mul_f32x4(self.values, scalar_vec);
        let abs_scalar = [scalar.abs(); 4];
        let uncertainties = simd_mul_f32x4(self.uncertainties, abs_scalar);

        Self {
            values,
            uncertainties,
            confidences: self.confidences,
        }
    }

    /// Compute weighted linear combination with uncertainty propagation.
    ///
    /// result = sum(weights[i] * self[i])
    /// sigma_result = sqrt(sum(weights[i]^2 * sigma[i]^2))
    #[inline]
    pub fn weighted_sum(&self, weights: [f32; 4]) -> (f32, f32, f32) {
        // Weighted values
        let weighted_values = simd_mul_f32x4(self.values, weights);
        let value = simd_horizontal_sum_f32x4(weighted_values);

        // Weighted uncertainties: sqrt(sum(w^2 * sigma^2))
        let weights_sq = simd_mul_f32x4(weights, weights);
        let sigma_sq = simd_mul_f32x4(self.uncertainties, self.uncertainties);
        let weighted_sigma_sq = simd_mul_f32x4(weights_sq, sigma_sq);
        let uncertainty = simd_horizontal_sum_f32x4(weighted_sigma_sq).sqrt();

        // Confidence: minimum
        let confidence = simd_horizontal_min_f32x4(self.confidences);

        (value, uncertainty, confidence)
    }

    /// Aggregate confidences using minimum rule
    #[inline]
    pub fn aggregate_confidence_min(&self) -> f32 {
        simd_horizontal_min_f32x4(self.confidences)
    }

    /// Aggregate confidences using product rule (for independent events)
    #[inline]
    pub fn aggregate_confidence_product(&self) -> f32 {
        simd_horizontal_product_f32x4(self.confidences)
    }

    /// Aggregate confidences using quadrature (uncertainty-based)
    ///
    /// combined = 1 - sqrt(sum((1 - c_i)^2))
    #[inline]
    pub fn aggregate_confidence_quadrature(&self) -> f32 {
        let ones = [1.0f32; 4];
        let one_minus_c = simd_sub_f32x4(ones, self.confidences);
        let squared = simd_mul_f32x4(one_minus_c, one_minus_c);
        let sum = simd_horizontal_sum_f32x4(squared);
        (1.0 - sum.sqrt()).max(0.0)
    }

    /// Compute mean value with propagated uncertainty
    #[inline]
    pub fn mean(&self) -> (f32, f32, f32) {
        let weights = [0.25f32; 4];
        self.weighted_sum(weights)
    }
}

impl std::ops::Add for KnowledgeVec4 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        KnowledgeVec4::add(&self, &rhs)
    }
}

impl std::ops::Sub for KnowledgeVec4 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        KnowledgeVec4::sub(&self, &rhs)
    }
}

impl std::ops::Mul for KnowledgeVec4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        KnowledgeVec4::mul(&self, &rhs)
    }
}

impl std::ops::Mul<f32> for KnowledgeVec4 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// KNOWLEDGE VEC8 - 8 Knowledge<f32> values (AVX2 or 2x Vec4)
// ============================================================================

/// Eight Knowledge<f32> values with SIMD-accelerated operations.
///
/// On AVX2-capable hardware, uses 256-bit operations.
/// Falls back to two KnowledgeVec4 operations on other platforms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KnowledgeVec8 {
    /// The central values
    pub values: [f32; 8],
    /// Standard deviations (uncertainties)
    pub uncertainties: [f32; 8],
    /// Confidence levels (0.0 to 1.0)
    pub confidences: [f32; 8],
}

impl KnowledgeVec8 {
    /// Create a new KnowledgeVec8
    #[inline]
    pub fn new(values: [f32; 8], uncertainties: [f32; 8], confidences: [f32; 8]) -> Self {
        Self {
            values,
            uncertainties,
            confidences,
        }
    }

    /// Create from certain values
    #[inline]
    pub fn certain(values: [f32; 8]) -> Self {
        Self {
            values,
            uncertainties: [0.0; 8],
            confidences: [1.0; 8],
        }
    }

    /// Create from uniform uncertainty and confidence
    #[inline]
    pub fn uniform(values: [f32; 8], uncertainty: f32, confidence: f32) -> Self {
        Self {
            values,
            uncertainties: [uncertainty; 8],
            confidences: [confidence; 8],
        }
    }

    /// Split into two KnowledgeVec4
    #[inline]
    fn split(&self) -> (KnowledgeVec4, KnowledgeVec4) {
        let lo = KnowledgeVec4::new(
            [
                self.values[0],
                self.values[1],
                self.values[2],
                self.values[3],
            ],
            [
                self.uncertainties[0],
                self.uncertainties[1],
                self.uncertainties[2],
                self.uncertainties[3],
            ],
            [
                self.confidences[0],
                self.confidences[1],
                self.confidences[2],
                self.confidences[3],
            ],
        );
        let hi = KnowledgeVec4::new(
            [
                self.values[4],
                self.values[5],
                self.values[6],
                self.values[7],
            ],
            [
                self.uncertainties[4],
                self.uncertainties[5],
                self.uncertainties[6],
                self.uncertainties[7],
            ],
            [
                self.confidences[4],
                self.confidences[5],
                self.confidences[6],
                self.confidences[7],
            ],
        );
        (lo, hi)
    }

    /// Merge two KnowledgeVec4 into KnowledgeVec8
    #[inline]
    fn merge(lo: KnowledgeVec4, hi: KnowledgeVec4) -> Self {
        Self {
            values: [
                lo.values[0],
                lo.values[1],
                lo.values[2],
                lo.values[3],
                hi.values[0],
                hi.values[1],
                hi.values[2],
                hi.values[3],
            ],
            uncertainties: [
                lo.uncertainties[0],
                lo.uncertainties[1],
                lo.uncertainties[2],
                lo.uncertainties[3],
                hi.uncertainties[0],
                hi.uncertainties[1],
                hi.uncertainties[2],
                hi.uncertainties[3],
            ],
            confidences: [
                lo.confidences[0],
                lo.confidences[1],
                lo.confidences[2],
                lo.confidences[3],
                hi.confidences[0],
                hi.confidences[1],
                hi.confidences[2],
                hi.confidences[3],
            ],
        }
    }

    /// Add two KnowledgeVec8 with uncertainty propagation
    #[inline]
    pub fn add(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.add_avx2(other) };
            }
        }
        // Fallback: use two Vec4 operations
        let (lo1, hi1) = self.split();
        let (lo2, hi2) = other.split();
        Self::merge(lo1.add(&lo2), hi1.add(&hi2))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_avx2(&self, other: &Self) -> Self {
        let va = _mm256_loadu_ps(self.values.as_ptr());
        let vb = _mm256_loadu_ps(other.values.as_ptr());
        let values_vec = _mm256_add_ps(va, vb);

        // Uncertainty propagation
        let sa = _mm256_loadu_ps(self.uncertainties.as_ptr());
        let sb = _mm256_loadu_ps(other.uncertainties.as_ptr());
        let sa_sq = _mm256_mul_ps(sa, sa);
        let sb_sq = _mm256_mul_ps(sb, sb);
        let sum_sq = _mm256_add_ps(sa_sq, sb_sq);
        let sigma_vec = _mm256_sqrt_ps(sum_sq);

        // Confidence: minimum
        let ca = _mm256_loadu_ps(self.confidences.as_ptr());
        let cb = _mm256_loadu_ps(other.confidences.as_ptr());
        let conf_vec = _mm256_min_ps(ca, cb);

        let mut result = Self::certain([0.0; 8]);
        _mm256_storeu_ps(result.values.as_mut_ptr(), values_vec);
        _mm256_storeu_ps(result.uncertainties.as_mut_ptr(), sigma_vec);
        _mm256_storeu_ps(result.confidences.as_mut_ptr(), conf_vec);
        result
    }

    /// Subtract two KnowledgeVec8 with uncertainty propagation
    #[inline]
    pub fn sub(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.sub_avx2(other) };
            }
        }
        let (lo1, hi1) = self.split();
        let (lo2, hi2) = other.split();
        Self::merge(lo1.sub(&lo2), hi1.sub(&hi2))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn sub_avx2(&self, other: &Self) -> Self {
        let va = _mm256_loadu_ps(self.values.as_ptr());
        let vb = _mm256_loadu_ps(other.values.as_ptr());
        let values_vec = _mm256_sub_ps(va, vb);

        let sa = _mm256_loadu_ps(self.uncertainties.as_ptr());
        let sb = _mm256_loadu_ps(other.uncertainties.as_ptr());
        let sa_sq = _mm256_mul_ps(sa, sa);
        let sb_sq = _mm256_mul_ps(sb, sb);
        let sum_sq = _mm256_add_ps(sa_sq, sb_sq);
        let sigma_vec = _mm256_sqrt_ps(sum_sq);

        let ca = _mm256_loadu_ps(self.confidences.as_ptr());
        let cb = _mm256_loadu_ps(other.confidences.as_ptr());
        let conf_vec = _mm256_min_ps(ca, cb);

        let mut result = Self::certain([0.0; 8]);
        _mm256_storeu_ps(result.values.as_mut_ptr(), values_vec);
        _mm256_storeu_ps(result.uncertainties.as_mut_ptr(), sigma_vec);
        _mm256_storeu_ps(result.confidences.as_mut_ptr(), conf_vec);
        result
    }

    /// Multiply two KnowledgeVec8 with uncertainty propagation
    #[inline]
    pub fn mul(&self, other: &Self) -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.mul_avx2(other) };
            }
        }
        let (lo1, hi1) = self.split();
        let (lo2, hi2) = other.split();
        Self::merge(lo1.mul(&lo2), hi1.mul(&hi2))
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn mul_avx2(&self, other: &Self) -> Self {
        let va = _mm256_loadu_ps(self.values.as_ptr());
        let vb = _mm256_loadu_ps(other.values.as_ptr());
        let values_vec = _mm256_mul_ps(va, vb);

        // sigma_z = sqrt((a * sigma_b)^2 + (b * sigma_a)^2)
        let sa = _mm256_loadu_ps(self.uncertainties.as_ptr());
        let sb = _mm256_loadu_ps(other.uncertainties.as_ptr());

        let a_sb = _mm256_mul_ps(va, sb);
        let b_sa = _mm256_mul_ps(vb, sa);
        let a_sb_sq = _mm256_mul_ps(a_sb, a_sb);
        let b_sa_sq = _mm256_mul_ps(b_sa, b_sa);
        let sum_sq = _mm256_add_ps(a_sb_sq, b_sa_sq);
        let sigma_vec = _mm256_sqrt_ps(sum_sq);

        let ca = _mm256_loadu_ps(self.confidences.as_ptr());
        let cb = _mm256_loadu_ps(other.confidences.as_ptr());
        let conf_vec = _mm256_min_ps(ca, cb);

        let mut result = Self::certain([0.0; 8]);
        _mm256_storeu_ps(result.values.as_mut_ptr(), values_vec);
        _mm256_storeu_ps(result.uncertainties.as_mut_ptr(), sigma_vec);
        _mm256_storeu_ps(result.confidences.as_mut_ptr(), conf_vec);
        result
    }

    /// Scale by a certain scalar
    #[inline]
    pub fn scale(&self, scalar: f32) -> Self {
        let (lo, hi) = self.split();
        Self::merge(lo.scale(scalar), hi.scale(scalar))
    }

    /// Aggregate confidence using minimum
    #[inline]
    pub fn aggregate_confidence_min(&self) -> f32 {
        let (lo, hi) = self.split();
        lo.aggregate_confidence_min()
            .min(hi.aggregate_confidence_min())
    }

    /// Aggregate confidence using product
    #[inline]
    pub fn aggregate_confidence_product(&self) -> f32 {
        let (lo, hi) = self.split();
        lo.aggregate_confidence_product() * hi.aggregate_confidence_product()
    }

    /// Aggregate confidence using quadrature
    #[inline]
    pub fn aggregate_confidence_quadrature(&self) -> f32 {
        let mut sum_sq = 0.0f32;
        for i in 0..8 {
            let diff = 1.0 - self.confidences[i];
            sum_sq += diff * diff;
        }
        (1.0 - sum_sq.sqrt()).max(0.0)
    }
}

impl std::ops::Add for KnowledgeVec8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        KnowledgeVec8::add(&self, &rhs)
    }
}

impl std::ops::Sub for KnowledgeVec8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        KnowledgeVec8::sub(&self, &rhs)
    }
}

impl std::ops::Mul for KnowledgeVec8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        KnowledgeVec8::mul(&self, &rhs)
    }
}

impl std::ops::Mul<f32> for KnowledgeVec8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f32) -> Self::Output {
        self.scale(rhs)
    }
}

// ============================================================================
// BATCH OPERATIONS
// ============================================================================

/// Process a batch of knowledge values with SIMD acceleration.
///
/// Applies an operation to arrays of knowledge values in chunks of 4.
pub fn batch_add_uncertainties(
    values_a: &[f32],
    sigmas_a: &[f32],
    values_b: &[f32],
    sigmas_b: &[f32],
    out_values: &mut [f32],
    out_sigmas: &mut [f32],
) {
    debug_assert_eq!(values_a.len(), values_b.len());
    debug_assert_eq!(values_a.len(), sigmas_a.len());
    debug_assert_eq!(values_a.len(), out_values.len());

    let len = values_a.len();
    let chunks = len / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let va = [
            values_a[offset],
            values_a[offset + 1],
            values_a[offset + 2],
            values_a[offset + 3],
        ];
        let sa = [
            sigmas_a[offset],
            sigmas_a[offset + 1],
            sigmas_a[offset + 2],
            sigmas_a[offset + 3],
        ];
        let vb = [
            values_b[offset],
            values_b[offset + 1],
            values_b[offset + 2],
            values_b[offset + 3],
        ];
        let sb = [
            sigmas_b[offset],
            sigmas_b[offset + 1],
            sigmas_b[offset + 2],
            sigmas_b[offset + 3],
        ];

        let kv_a = KnowledgeVec4::new(va, sa, [1.0; 4]);
        let kv_b = KnowledgeVec4::new(vb, sb, [1.0; 4]);
        let result = kv_a.add(&kv_b);

        out_values[offset..offset + 4].copy_from_slice(&result.values);
        out_sigmas[offset..offset + 4].copy_from_slice(&result.uncertainties);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        out_values[i] = values_a[i] + values_b[i];
        out_sigmas[i] = (sigmas_a[i].powi(2) + sigmas_b[i].powi(2)).sqrt();
    }
}

/// Compute weighted mean with uncertainty for a batch of values
pub fn batch_weighted_mean(values: &[f32], sigmas: &[f32], weights: &[f32]) -> (f32, f32) {
    debug_assert_eq!(values.len(), sigmas.len());
    debug_assert_eq!(values.len(), weights.len());

    let len = values.len();
    let chunks = len / 4;

    let mut total_value = 0.0f32;
    let mut total_sigma_sq = 0.0f32;

    for i in 0..chunks {
        let offset = i * 4;
        let v = [
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
        ];
        let s = [
            sigmas[offset],
            sigmas[offset + 1],
            sigmas[offset + 2],
            sigmas[offset + 3],
        ];
        let w = [
            weights[offset],
            weights[offset + 1],
            weights[offset + 2],
            weights[offset + 3],
        ];

        let kv = KnowledgeVec4::new(v, s, [1.0; 4]);
        let (val, unc, _) = kv.weighted_sum(w);
        total_value += val;
        total_sigma_sq += unc.powi(2);
    }

    // Handle remainder
    for i in (chunks * 4)..len {
        total_value += values[i] * weights[i];
        total_sigma_sq += (weights[i] * sigmas[i]).powi(2);
    }

    (total_value, total_sigma_sq.sqrt())
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    #[test]
    fn test_simd_add_f32x4() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [5.0, 6.0, 7.0, 8.0];
        let result = simd_add_f32x4(a, b);
        assert!(approx_eq(result[0], 6.0));
        assert!(approx_eq(result[1], 8.0));
        assert!(approx_eq(result[2], 10.0));
        assert!(approx_eq(result[3], 12.0));
    }

    #[test]
    fn test_simd_sqrt_f32x4() {
        let a = [1.0, 4.0, 9.0, 16.0];
        let result = simd_sqrt_f32x4(a);
        assert!(approx_eq(result[0], 1.0));
        assert!(approx_eq(result[1], 2.0));
        assert!(approx_eq(result[2], 3.0));
        assert!(approx_eq(result[3], 4.0));
    }

    #[test]
    fn test_simd_horizontal_sum() {
        let v = [1.0, 2.0, 3.0, 4.0];
        let sum = simd_horizontal_sum_f32x4(v);
        assert!(approx_eq(sum, 10.0));
    }

    #[test]
    fn test_knowledge_vec4_add_certain() {
        let a = KnowledgeVec4::certain([1.0, 2.0, 3.0, 4.0]);
        let b = KnowledgeVec4::certain([5.0, 6.0, 7.0, 8.0]);
        let result = a + b;

        assert!(approx_eq(result.values[0], 6.0));
        assert!(approx_eq(result.values[1], 8.0));
        assert!(approx_eq(result.values[2], 10.0));
        assert!(approx_eq(result.values[3], 12.0));

        // Uncertainties should be 0 for certain values
        for u in result.uncertainties {
            assert!(approx_eq(u, 0.0));
        }
    }

    #[test]
    fn test_knowledge_vec4_add_uncertain() {
        let a = KnowledgeVec4::uniform([1.0; 4], 0.1, 0.95);
        let b = KnowledgeVec4::uniform([2.0; 4], 0.2, 0.90);
        let result = a + b;

        // Values should add
        for v in result.values {
            assert!(approx_eq(v, 3.0));
        }

        // Uncertainty: sqrt(0.1^2 + 0.2^2) = sqrt(0.01 + 0.04) = sqrt(0.05)
        let expected_sigma = (0.01f32 + 0.04).sqrt();
        for u in result.uncertainties {
            assert!(approx_eq(u, expected_sigma));
        }

        // Confidence: min(0.95, 0.90) = 0.90
        for c in result.confidences {
            assert!(approx_eq(c, 0.90));
        }
    }

    #[test]
    fn test_knowledge_vec4_mul_uncertain() {
        let a = KnowledgeVec4::new([2.0; 4], [0.1; 4], [0.95; 4]);
        let b = KnowledgeVec4::new([3.0; 4], [0.2; 4], [0.90; 4]);
        let result = a * b;

        // Values: 2 * 3 = 6
        for v in result.values {
            assert!(approx_eq(v, 6.0));
        }

        // Uncertainty: sqrt((a*sigma_b)^2 + (b*sigma_a)^2)
        // = sqrt((2*0.2)^2 + (3*0.1)^2) = sqrt(0.16 + 0.09) = sqrt(0.25) = 0.5
        for u in result.uncertainties {
            assert!(approx_eq(u, 0.5));
        }
    }

    #[test]
    fn test_knowledge_vec4_weighted_sum() {
        let kv = KnowledgeVec4::new(
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.1, 0.1, 0.1],
            [0.9, 0.8, 0.95, 0.85],
        );

        let weights = [0.25, 0.25, 0.25, 0.25];
        let (value, uncertainty, confidence) = kv.weighted_sum(weights);

        // Value: (1+2+3+4)/4 = 2.5
        assert!(approx_eq(value, 2.5));

        // Uncertainty: sqrt(4 * (0.25^2 * 0.1^2)) = sqrt(4 * 0.000625) = 0.05
        assert!(approx_eq(uncertainty, 0.05));

        // Confidence: min = 0.8
        assert!(approx_eq(confidence, 0.8));
    }

    #[test]
    fn test_knowledge_vec4_confidence_aggregation() {
        let kv = KnowledgeVec4::new([1.0; 4], [0.1; 4], [0.9, 0.8, 0.95, 0.85]);

        assert!(approx_eq(kv.aggregate_confidence_min(), 0.8));

        let product = 0.9 * 0.8 * 0.95 * 0.85;
        assert!(approx_eq(kv.aggregate_confidence_product(), product));
    }

    #[test]
    fn test_knowledge_vec8_add() {
        let a = KnowledgeVec8::uniform([1.0; 8], 0.1, 0.95);
        let b = KnowledgeVec8::uniform([2.0; 8], 0.2, 0.90);
        let result = a + b;

        for v in result.values {
            assert!(approx_eq(v, 3.0));
        }

        let expected_sigma = (0.01f32 + 0.04).sqrt();
        for u in result.uncertainties {
            assert!(approx_eq(u, expected_sigma));
        }

        for c in result.confidences {
            assert!(approx_eq(c, 0.90));
        }
    }

    #[test]
    fn test_batch_add_uncertainties() {
        let values_a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let sigmas_a = vec![0.1, 0.1, 0.1, 0.1, 0.1];
        let values_b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let sigmas_b = vec![0.2, 0.2, 0.2, 0.2, 0.2];

        let mut out_values = vec![0.0; 5];
        let mut out_sigmas = vec![0.0; 5];

        batch_add_uncertainties(
            &values_a,
            &sigmas_a,
            &values_b,
            &sigmas_b,
            &mut out_values,
            &mut out_sigmas,
        );

        let expected_sigma = (0.01f32 + 0.04).sqrt();
        for i in 0..5 {
            assert!(approx_eq(out_values[i], values_a[i] + values_b[i]));
            assert!(approx_eq(out_sigmas[i], expected_sigma));
        }
    }

    #[test]
    fn test_scalar_fallback_matches_simd() {
        // Ensure scalar and SIMD produce same results
        let a = [1.5, 2.5, 3.5, 4.5];
        let b = [0.5, 1.5, 2.5, 3.5];

        let add_result = simd_add_f32x4(a, b);
        let mul_result = simd_mul_f32x4(a, b);
        let sqrt_result = simd_sqrt_f32x4(a);

        // Verify against scalar
        for i in 0..4 {
            assert!(approx_eq(add_result[i], a[i] + b[i]));
            assert!(approx_eq(mul_result[i], a[i] * b[i]));
            assert!(approx_eq(sqrt_result[i], a[i].sqrt()));
        }
    }
}
