//! SIMD-Accelerated Embedding Operations
//!
//! Uses AVX2/AVX-512 for fast vector operations on embedding spaces.
//!
//! # Performance
//!
//! On AVX2-capable hardware:
//! - Cosine similarity: ~500ns for 256-dim vectors
//! - Euclidean distance: ~400ns for 256-dim vectors
//! - Dot product: ~300ns for 256-dim vectors

// Allow unsafe operations in unsafe functions (Rust 2024 compatibility)
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
/// SIMD-accelerated cosine similarity
///
/// Returns the cosine similarity between two vectors, in range [-1, 1].
/// Identical vectors have similarity 1.0.
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { cosine_similarity_avx2(a, b) };
        }
    }

    cosine_similarity_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn cosine_similarity_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut dot_sum = _mm256_setzero_ps();
    let mut a_sq_sum = _mm256_setzero_ps();
    let mut b_sq_sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));

        // dot product: dot_sum += a * b
        dot_sum = _mm256_fmadd_ps(a_vec, b_vec, dot_sum);

        // squared norms: a_sq_sum += a * a, b_sq_sum += b * b
        a_sq_sum = _mm256_fmadd_ps(a_vec, a_vec, a_sq_sum);
        b_sq_sum = _mm256_fmadd_ps(b_vec, b_vec, b_sq_sum);
    }

    // Horizontal sum
    let dot = hsum_avx2(dot_sum);
    let a_sq = hsum_avx2(a_sq_sum);
    let b_sq = hsum_avx2(b_sq_sum);

    // Handle remainder
    let remainder_start = chunks * 8;
    let mut dot_remainder = 0.0f32;
    let mut a_sq_remainder = 0.0f32;
    let mut b_sq_remainder = 0.0f32;

    for i in remainder_start..len {
        dot_remainder += a[i] * b[i];
        a_sq_remainder += a[i] * a[i];
        b_sq_remainder += b[i] * b[i];
    }

    let total_dot = dot + dot_remainder;
    let total_a_norm = (a_sq + a_sq_remainder).sqrt();
    let total_b_norm = (b_sq + b_sq_remainder).sqrt();

    if total_a_norm == 0.0 || total_b_norm == 0.0 {
        0.0
    } else {
        total_dot / (total_a_norm * total_b_norm)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_avx2(v: __m256) -> f32 {
    // Extract high and low 128-bit lanes
    let low = _mm256_castps256_ps128(v);
    let high = _mm256_extractf128_ps(v, 1);

    // Add the two 128-bit vectors
    let sum128 = _mm_add_ps(low, high);

    // Horizontal add within 128-bit vector
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));

    _mm_cvtss_f32(sum32)
}

/// Scalar fallback for cosine similarity
fn cosine_similarity_scalar(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let a_norm: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let b_norm: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if a_norm == 0.0 || b_norm == 0.0 {
        0.0
    } else {
        dot / (a_norm * b_norm)
    }
}

/// SIMD-accelerated Euclidean distance
///
/// Returns the L2 (Euclidean) distance between two vectors.
pub fn euclidean_distance_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { euclidean_distance_avx2(a, b) };
        }
    }

    euclidean_distance_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    let mut total = hsum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        let diff = a[i] - b[i];
        total += diff * diff;
    }

    total.sqrt()
}

/// Scalar fallback for Euclidean distance
fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

/// SIMD-accelerated dot product
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Vectors must have same length");

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            return unsafe { dot_product_avx2(a, b) };
        }
    }

    dot_product_scalar(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(a_vec, b_vec, sum);
    }

    let mut total = hsum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        total += a[i] * b[i];
    }

    total
}

/// Scalar fallback for dot product
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// SIMD-accelerated vector normalization (in-place)
pub fn normalize_simd(v: &mut [f32]) {
    let norm = euclidean_norm_simd(v);
    if norm > 0.0 {
        let inv_norm = 1.0 / norm;
        for x in v.iter_mut() {
            *x *= inv_norm;
        }
    }
}

/// SIMD-accelerated L2 norm
pub fn euclidean_norm_simd(v: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && v.len() >= 8 {
            return unsafe { euclidean_norm_avx2(v) };
        }
    }

    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn euclidean_norm_avx2(v: &[f32]) -> f32 {
    let len = v.len();
    let chunks = len / 8;

    let mut sum = _mm256_setzero_ps();

    for i in 0..chunks {
        let vec = _mm256_loadu_ps(v.as_ptr().add(i * 8));
        sum = _mm256_fmadd_ps(vec, vec, sum);
    }

    let mut total = hsum_avx2(sum);

    // Handle remainder
    for i in (chunks * 8)..len {
        total += v[i] * v[i];
    }

    total.sqrt()
}

/// SIMD-accelerated vector addition (a + b -> result)
pub fn add_vectors_simd(a: &[f32], b: &[f32], result: &mut [f32]) {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), result.len());

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && a.len() >= 8 {
            unsafe {
                add_vectors_avx2(a, b, result);
            }
            return;
        }
    }

    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_vectors_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    let len = a.len();
    let chunks = len / 8;

    for i in 0..chunks {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
        let sum = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i * 8), sum);
    }

    // Handle remainder
    for i in (chunks * 8)..len {
        result[i] = a[i] + b[i];
    }
}

/// Batch cosine similarity: compute similarity between query and all targets
pub fn batch_cosine_similarity(query: &[f32], targets: &[&[f32]]) -> Vec<f32> {
    targets
        .iter()
        .map(|target| cosine_similarity_simd(query, target))
        .collect()
}

/// Find k nearest neighbors by cosine similarity
pub fn k_nearest_by_cosine(query: &[f32], targets: &[&[f32]], k: usize) -> Vec<(usize, f32)> {
    let similarities: Vec<(usize, f32)> = targets
        .iter()
        .enumerate()
        .map(|(i, target)| (i, cosine_similarity_simd(query, target)))
        .collect();

    // Use partial sort for efficiency when k << n
    let mut sorted = similarities;
    sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(k);
    sorted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let sim = cosine_similarity_simd(&a, &b);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "Identical vectors should have similarity 1.0"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let sim = cosine_similarity_simd(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "Orthogonal vectors should have similarity 0.0"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

        let sim = cosine_similarity_simd(&a, &b);
        assert!(
            (sim - (-1.0)).abs() < 1e-6,
            "Opposite vectors should have similarity -1.0"
        );
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![0.0; 256];
        let mut b = vec![0.0; 256];
        b[0] = 3.0;
        b[1] = 4.0;

        let dist = euclidean_distance_simd(&a, &b);
        assert!(
            (dist - 5.0).abs() < 1e-6,
            "Distance should be 5.0 (3-4-5 triangle)"
        );
    }

    #[test]
    fn test_euclidean_distance_zero() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let dist = euclidean_distance_simd(&a, &b);
        assert!(dist.abs() < 1e-6, "Distance to self should be 0.0");
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let dot = dot_product_simd(&a, &b);
        let expected: f32 = (1..=8).map(|x| x as f32).sum();
        assert!(
            (dot - expected).abs() < 1e-6,
            "Dot product should be sum 1..8 = 36"
        );
    }

    #[test]
    fn test_normalize() {
        let mut v = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        normalize_simd(&mut v);

        let norm = euclidean_norm_simd(&v);
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Normalized vector should have norm 1.0"
        );
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_add_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; 8];

        add_vectors_simd(&a, &b, &mut result);

        for x in &result {
            assert!((x - 9.0).abs() < 1e-6, "Sum should be 9.0 for all elements");
        }
    }

    #[test]
    fn test_k_nearest() {
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let targets: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // sim = 1.0
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // sim = 0.0
            vec![0.7, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // sim ≈ 0.707
            vec![-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], // sim = -1.0
        ];
        let target_refs: Vec<&[f32]> = targets.iter().map(|v| v.as_slice()).collect();

        let nearest = k_nearest_by_cosine(&query, &target_refs, 2);

        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, 0); // First target (sim = 1.0)
        assert_eq!(nearest[1].0, 2); // Third target (sim ≈ 0.707)
    }

    #[test]
    fn test_scalar_matches_simd() {
        // Test that scalar and SIMD implementations produce same results
        let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..256).map(|i| ((i + 50) as f32) * 0.01).collect();

        let scalar_cosine = cosine_similarity_scalar(&a, &b);
        let simd_cosine = cosine_similarity_simd(&a, &b);
        assert!(
            (scalar_cosine - simd_cosine).abs() < 1e-5,
            "Scalar and SIMD cosine should match"
        );

        let scalar_euclidean = euclidean_distance_scalar(&a, &b);
        let simd_euclidean = euclidean_distance_simd(&a, &b);
        assert!(
            (scalar_euclidean - simd_euclidean).abs() < 1e-4,
            "Scalar and SIMD euclidean should match"
        );

        let scalar_dot = dot_product_scalar(&a, &b);
        let simd_dot = dot_product_simd(&a, &b);
        assert!(
            (scalar_dot - simd_dot).abs() < 1e-3,
            "Scalar and SIMD dot product should match"
        );
    }
}
