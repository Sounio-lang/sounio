//! Common utilities for Phase 2 integration tests
//!
//! Provides:
//! - Data generators (quaternions, sparse patterns)
//! - Validators (precision flow, sparsity, quantization)
//! - Test harness (Phase2TestHarness)

use std::collections::HashMap;

/// Quaternion representation for testing
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    pub fn identity() -> Self {
        Self { w: 1.0, x: 0.0, y: 0.0, z: 0.0 }
    }

    pub fn norm(&self) -> f32 {
        (self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    pub fn mul(&self, other: &Quaternion) -> Quaternion {
        Quaternion {
            w: self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
            x: self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y: self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z: self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
        }
    }

    pub fn as_array(&self) -> [f32; 4] {
        [self.w, self.x, self.y, self.z]
    }
}

/// Generates random quaternions for testing
pub struct QuaternionGenerator {
    seed: u32,
}

impl QuaternionGenerator {
    pub fn new(seed: u32) -> Self {
        Self { seed }
    }

    pub fn next(&mut self) -> Quaternion {
        // Simple LCG random number generator
        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let f = (self.seed as f32) / (u32::MAX as f32);

        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let g = (self.seed as f32) / (u32::MAX as f32);

        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let h = (self.seed as f32) / (u32::MAX as f32);

        self.seed = self.seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let i = (self.seed as f32) / (u32::MAX as f32);

        let mut q = Quaternion::new(f, g, h, i);
        let n = q.norm();
        if n > 0.0 {
            q.w /= n;
            q.x /= n;
            q.y /= n;
            q.z /= n;
        }
        q
    }

    pub fn generate_batch(&mut self, size: usize) -> Vec<Quaternion> {
        (0..size).map(|_| self.next()).collect()
    }
}

/// Sparse pattern generator for testing sparsity patterns
pub enum SparsePattern {
    CSR,
    Structured2x4,
    StructuredNxM { n: usize, m: usize },
}

pub struct SparsePatternGenerator;

impl SparsePatternGenerator {
    /// Generate CSR (Compressed Sparse Row) pattern
    pub fn generate_csr(sparsity: f64, size: usize) -> Vec<bool> {
        let mut rng = QuaternionGenerator::new(42);
        (0..size)
            .map(|_| {
                let q = rng.next();
                q.norm() > (1.0 - sparsity as f32)
            })
            .collect()
    }

    /// Generate 2:4 structured sparsity (keep 2 of 4)
    pub fn generate_2x4(size: usize) -> Vec<bool> {
        let mut mask = vec![false; size];
        let mut rng = QuaternionGenerator::new(42);

        for chunk in mask.chunks_mut(4) {
            let mut indices: Vec<usize> = (0..chunk.len()).collect();
            // Simple shuffle
            for i in 0..chunk.len() {
                let j = (rng.next().w * chunk.len() as f32) as usize;
                indices.swap(i, j % chunk.len());
            }
            // Keep first 2
            for i in 0..2.min(chunk.len()) {
                chunk[indices[i]] = true;
            }
        }
        mask
    }

    /// Generate N:M structured sparsity
    pub fn generate_nxm(n: usize, m: usize, size: usize) -> Vec<bool> {
        let mut mask = vec![false; size];
        let mut rng = QuaternionGenerator::new(42);

        for chunk in mask.chunks_mut(m) {
            let mut indices: Vec<usize> = (0..chunk.len()).collect();
            for i in 0..chunk.len() {
                let j = (rng.next().w * chunk.len() as f32) as usize;
                indices.swap(i, j % chunk.len());
            }
            for i in 0..n.min(chunk.len()) {
                chunk[indices[i]] = true;
            }
        }
        mask
    }
}

/// Validators module for checking correctness
pub mod validators {
    use super::*;

    /// Validate precision flow through computation
    pub fn validate_precision_flow(
        input_fp32: &[f32],
        output_fp16: &[f32],
        max_error: f32,
    ) -> bool {
        if input_fp32.len() != output_fp16.len() {
            return false;
        }

        input_fp32.iter().zip(output_fp16.iter()).all(|(a, b)| {
            (a - b).abs() <= max_error
        })
    }

    /// Validate sparsity pattern
    pub fn validate_sparsity(mask: &[bool], expected_sparsity: f64) -> bool {
        if mask.is_empty() {
            return false;
        }

        let actual_sparsity = mask.iter().filter(|&&b| !b).count() as f64 / mask.len() as f64;
        (actual_sparsity - expected_sparsity).abs() < 0.05
    }

    /// Validate fake quantization result
    pub fn validate_fake_quantization(
        original: &[f32],
        quantized: &[f32],
        scale: f32,
    ) -> bool {
        if original.len() != quantized.len() {
            return false;
        }

        original.iter().zip(quantized.iter()).all(|(orig, quant)| {
            let max_error = scale * 0.5; // Quantization error < 0.5 * scale
            (orig - quant).abs() <= max_error
        })
    }

    /// Validate loss scale update
    pub fn validate_loss_scale(old_scale: f32, new_scale: f32, direction: &str) -> bool {
        match direction {
            "up" => new_scale > old_scale && (new_scale / old_scale - 2.0).abs() < 0.01,
            "down" => new_scale < old_scale && (old_scale / new_scale - 2.0).abs() < 0.01,
            "none" => (new_scale - old_scale).abs() < 1e-6,
            _ => false,
        }
    }
}

/// Test harness for Phase 2 features
pub struct Phase2TestHarness {
    config: HashMap<String, f32>,
}

impl Phase2TestHarness {
    pub fn new() -> Self {
        let mut config = HashMap::new();
        // Mixed-precision defaults
        config.insert("initial_loss_scale".to_string(), 32768.0);
        config.insert("scale_growth_factor".to_string(), 2.0);
        config.insert("scale_growth_interval".to_string(), 2000.0);

        // QAT defaults
        config.insert("warmup_batches".to_string(), 1000.0);
        config.insert("momentum".to_string(), 0.01);

        // Sparsity defaults
        config.insert("target_sparsity".to_string(), 0.5);

        Self { config }
    }

    pub fn set_config(&mut self, key: &str, value: f32) {
        self.config.insert(key.to_string(), value);
    }

    pub fn get_config(&self, key: &str) -> Option<f32> {
        self.config.get(key).copied()
    }

    /// Simulate forward pass with mixed precision
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        // Convert FP32 -> FP16 -> FP32 (simulated)
        input.iter().map(|&x| {
            // Simulate FP16 quantization
            let quantized = ((x * 65520.0).round()) / 65520.0;
            quantized
        }).collect()
    }

    /// Simulate backward pass with gradient scaling
    pub fn backward(&self, gradient: &[f32]) -> Vec<f32> {
        let loss_scale = self.get_config("initial_loss_scale").unwrap_or(32768.0);
        gradient.iter().map(|&g| g / loss_scale).collect()
    }

    /// Simulate fused layer (Linear+BN+ReLU)
    pub fn fused_layer(&self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&x| {
            // Linear: y = x + 0.5
            let linear = x + 0.5;
            // BN: normalize
            let bn = linear * 0.95;
            // ReLU: max(0, x)
            bn.max(0.0)
        }).collect()
    }

    /// Simulate sparse GEMM with 2:4 pattern
    pub fn sparse_gemm(&self, input: &[f32], mask: &[bool]) -> Vec<f32> {
        input.iter().zip(mask.iter())
            .map(|(&x, &keep)| if keep { x } else { 0.0 })
            .collect()
    }

    /// Simulate QAT forward (fake quantization)
    pub fn qat_forward(&self, input: &[f32], scale: f32) -> Vec<f32> {
        input.iter().map(|&x| {
            let quantized = (x / scale).round() * scale;
            quantized
        }).collect()
    }
}

impl Default for Phase2TestHarness {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quaternion_identity() {
        let q = Quaternion::identity();
        assert_eq!(q.w, 1.0);
        assert_eq!(q.x, 0.0);
    }

    #[test]
    fn test_quaternion_norm() {
        let q = Quaternion::new(3.0, 4.0, 0.0, 0.0);
        assert!((q.norm() - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_quat_generator() {
        let mut r#gen = QuaternionGenerator::new(123);
        let q = r#gen.next();
        assert!((q.norm() - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sparse_pattern_2x4() {
        let mask = SparsePatternGenerator::generate_2x4(16);
        assert_eq!(mask.len(), 16);
        // Each group of 4 should have at most 2 trues
        for chunk in mask.chunks(4) {
            let count = chunk.iter().filter(|&&b| b).count();
            assert!(count <= 2);
        }
    }

    #[test]
    fn test_validator_sparsity() {
        let mask = vec![true, false, true, false, true, false, false, false];
        assert!(validators::validate_sparsity(&mask, 0.5));
    }

    #[test]
    fn test_test_harness() {
        let harness = Phase2TestHarness::new();
        assert_eq!(harness.get_config("initial_loss_scale"), Some(32768.0));

        let result = harness.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3);
    }
}
