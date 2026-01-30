//! Sparse Quaternion Operations
//!
//! Quaternion-aware sparsity patterns that prune at quaternion granularity (4 components together).
//! Supports 2:4 structured sparsity compatible with Ampere+ Tensor Cores for 2-4x speedup.

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use super::ir::SparseQuatFormat;

/// Quaternion pruning method
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuatPruningMethod {
    /// Global magnitude pruning: prune quaternions with smallest norm
    GlobalMagnitude,
    /// Local magnitude: prune within each row/channel independently
    LocalMagnitude,
    /// Structured 2:4: force exactly 2 of every 4 quaternions to zero
    Structured2x4,
    /// Structured N:M: force exactly M-N of every M quaternions to zero
    StructuredNxM { n: u32, m: u32 },
    /// Gradual magnitude pruning during training
    GradualMagnitude {
        start_sparsity: f64,
        end_sparsity: f64,
    },
}

/// Sparse quaternion tensor metadata
#[derive(Debug, Clone)]
pub struct SparseQuatTensor {
    /// Storage format
    pub format: SparseQuatFormat,
    /// Shape in quaternions [batch, channels, height, width] or [out_features, in_features]
    pub quat_shape: Vec<usize>,
    /// Number of non-zero quaternions
    pub nnz_quats: usize,
    /// Sparsity ratio (fraction of pruned quaternions)
    pub sparsity: f64,
}

/// Quaternion pruning engine
#[derive(Debug, Clone)]
pub struct QuatPruningEngine {
    /// Target sparsity ratio (0.0 to 1.0)
    target_sparsity: f64,
    /// Pruning method
    method: QuatPruningMethod,
    /// Minimum quaternion norm to consider (epsilon for numerical stability)
    epsilon: f32,
}

impl QuatPruningEngine {
    /// Create a new pruning engine with the given method
    pub fn new(method: QuatPruningMethod, target_sparsity: f64) -> Self {
        Self {
            target_sparsity: target_sparsity.clamp(0.0, 1.0),
            method,
            epsilon: 1e-7,
        }
    }

    /// Create 2:4 structured pruning engine (50% sparsity)
    pub fn structured_2x4() -> Self {
        Self {
            target_sparsity: 0.5,
            method: QuatPruningMethod::Structured2x4,
            epsilon: 1e-7,
        }
    }

    /// Create global magnitude pruning engine
    pub fn global_magnitude(sparsity: f64) -> Self {
        Self {
            target_sparsity: sparsity.clamp(0.0, 1.0),
            method: QuatPruningMethod::GlobalMagnitude,
            epsilon: 1e-7,
        }
    }

    /// Prune a quaternion weight tensor
    ///
    /// # Arguments
    /// * `weights` - Slice of quaternions, each as [w, x, y, z]
    ///
    /// # Returns
    /// * Sparse tensor metadata
    /// * Boolean mask (true = keep, false = prune)
    pub fn prune(&self, weights: &[[f32; 4]]) -> (SparseQuatTensor, Vec<bool>) {
        let num_quats = weights.len();
        if num_quats == 0 {
            return (
                SparseQuatTensor {
                    format: SparseQuatFormat::Dense,
                    quat_shape: vec![0],
                    nnz_quats: 0,
                    sparsity: 1.0,
                },
                vec![],
            );
        }

        // Compute quaternion norms
        let norms: Vec<f32> = weights
            .iter()
            .map(|q| (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt())
            .collect();

        let (mask, format) = match &self.method {
            QuatPruningMethod::GlobalMagnitude => {
                let m = self.global_magnitude_mask(&norms);
                (m, SparseQuatFormat::QuatCSR)
            }
            QuatPruningMethod::LocalMagnitude => {
                let m = self.local_magnitude_mask(&norms, 4);
                (m, SparseQuatFormat::QuatCSR)
            }
            QuatPruningMethod::Structured2x4 => {
                let m = Self::structured_2x4_mask(&norms);
                (m, SparseQuatFormat::QuatStructured2x4)
            }
            QuatPruningMethod::StructuredNxM { n, m } => {
                let mask = self.structured_nxm_mask(&norms, *n as usize, *m as usize);
                (mask, SparseQuatFormat::QuatStructuredNxM { n: *n, m: *m })
            }
            QuatPruningMethod::GradualMagnitude { .. } => {
                // Use average sparsity for gradual pruning
                let m = self.global_magnitude_mask(&norms);
                (m, SparseQuatFormat::QuatCSR)
            }
        };

        let nnz_quats = mask.iter().filter(|&&m| m).count();
        let sparsity = if num_quats == 0 {
            1.0
        } else {
            1.0 - (nnz_quats as f64 / num_quats as f64)
        };

        let tensor = SparseQuatTensor {
            format,
            quat_shape: vec![num_quats],
            nnz_quats,
            sparsity,
        };

        (tensor, mask)
    }

    /// Global magnitude mask: keep quaternions with largest norms
    fn global_magnitude_mask(&self, norms: &[f32]) -> Vec<bool> {
        let num_quats = norms.len();
        let keep_fraction = 1.0 - self.target_sparsity;
        let keep_num = (keep_fraction * num_quats as f64).round() as usize;
        let actual_keep = keep_num.min(num_quats);

        // Sort by norm descending
        let mut indexed: Vec<(usize, f32)> = norms.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep top-k
        let mut mask = vec![false; num_quats];
        for i in 0..actual_keep {
            mask[indexed[i].0] = true;
        }
        mask
    }

    /// Local magnitude mask: prune within each group independently
    fn local_magnitude_mask(&self, norms: &[f32], group_size: usize) -> Vec<bool> {
        let num_quats = norms.len();
        let keep_fraction = 1.0 - self.target_sparsity;
        let mut mask = vec![false; num_quats];

        for chunk_start in (0..num_quats).step_by(group_size) {
            let chunk_end = (chunk_start + group_size).min(num_quats);
            let chunk_len = chunk_end - chunk_start;
            let keep_num = ((keep_fraction * chunk_len as f64).round() as usize).min(chunk_len);

            // Sort chunk by norm descending
            let mut chunk: Vec<(usize, f32)> =
                (chunk_start..chunk_end).map(|i| (i, norms[i])).collect();
            chunk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top-k in chunk
            for i in 0..keep_num {
                mask[chunk[i].0] = true;
            }
        }
        mask
    }

    /// Structured 2:4 mask: keep top 2 quaternions by norm in each group of 4
    pub fn structured_2x4_mask(norms: &[f32]) -> Vec<bool> {
        let num_quats = norms.len();
        let mut mask = vec![false; num_quats];

        for group_start in (0..num_quats).step_by(4) {
            let group_end = (group_start + 4).min(num_quats);
            let chunk_len = group_end - group_start;
            let keep_num = 2.min(chunk_len);

            // Sort chunk by norm descending
            let mut chunk: Vec<(usize, f32)> =
                (group_start..group_end).map(|i| (i, norms[i])).collect();
            chunk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top 2
            for i in 0..keep_num {
                mask[chunk[i].0] = true;
            }
        }
        mask
    }

    /// Structured N:M mask: keep top N quaternions by norm in each group of M
    fn structured_nxm_mask(&self, norms: &[f32], n: usize, m: usize) -> Vec<bool> {
        let num_quats = norms.len();
        let mut mask = vec![false; num_quats];

        for group_start in (0..num_quats).step_by(m) {
            let group_end = (group_start + m).min(num_quats);
            let chunk_len = group_end - group_start;
            let keep_num = n.min(chunk_len);

            let mut chunk: Vec<(usize, f32)> =
                (group_start..group_end).map(|i| (i, norms[i])).collect();
            chunk.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for i in 0..keep_num {
                mask[chunk[i].0] = true;
            }
        }
        mask
    }

    /// Generate 2:4 metadata for Tensor Core acceleration
    ///
    /// Each byte encodes positions of 2 non-zero quaternions in a group of 4.
    /// Format: bits [1:0] = first position, bits [3:2] = second position
    pub fn generate_2x4_metadata(mask: &[bool]) -> Vec<u8> {
        let num_groups = (mask.len() + 3) / 4;
        let mut metadata = Vec::with_capacity(num_groups);

        for chunk in mask.chunks(4) {
            let mut byte: u8 = 0;
            let mut pos = 0u8;

            for (i, &keep) in chunk.iter().enumerate() {
                if keep {
                    // Each position uses 2 bits (0, 1, 2, or 3)
                    byte |= (i as u8) << (pos * 2);
                    pos += 1;
                    if pos >= 2 {
                        break;
                    }
                }
            }

            metadata.push(byte);
        }

        metadata
    }

    /// Decode 2:4 metadata back to positions
    pub fn decode_2x4_metadata(metadata: &[u8]) -> Vec<(usize, usize)> {
        metadata
            .iter()
            .map(|&byte| {
                let pos0 = (byte & 0x03) as usize;
                let pos1 = ((byte >> 2) & 0x03) as usize;
                (pos0, pos1)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structured_2x4() {
        let engine = QuatPruningEngine::structured_2x4();
        let weights = [
            [1.0, 0.0, 0.0, 0.0], // norm = 1.0
            [2.0, 0.0, 0.0, 0.0], // norm = 2.0
            [0.5, 0.0, 0.0, 0.0], // norm = 0.5
            [1.5, 0.0, 0.0, 0.0], // norm = 1.5
        ];

        let (tensor, mask) = engine.prune(&weights);

        assert_eq!(tensor.format, SparseQuatFormat::QuatStructured2x4);
        assert_eq!(tensor.nnz_quats, 2);
        assert!((tensor.sparsity - 0.5).abs() < 0.01);

        // Should keep indices 1 (norm=2.0) and 3 (norm=1.5)
        assert!(!mask[0]); // norm=1.0, pruned
        assert!(mask[1]); // norm=2.0, kept
        assert!(!mask[2]); // norm=0.5, pruned
        assert!(mask[3]); // norm=1.5, kept
    }

    #[test]
    fn test_2x4_metadata() {
        let mask = vec![false, true, false, true]; // positions 1 and 3
        let metadata = QuatPruningEngine::generate_2x4_metadata(&mask);
        assert_eq!(metadata.len(), 1);

        let positions = QuatPruningEngine::decode_2x4_metadata(&metadata);
        assert_eq!(positions[0], (1, 3));
    }

    #[test]
    fn test_global_magnitude() {
        let engine = QuatPruningEngine::global_magnitude(0.5);
        let weights = [
            [1.0, 0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0, 0.0],
        ];

        let (tensor, mask) = engine.prune(&weights);

        assert_eq!(tensor.nnz_quats, 2);
        // Should keep top 2 by norm: indices 1 and 3
        assert!(mask[1]);
        assert!(mask[3]);
    }
}
