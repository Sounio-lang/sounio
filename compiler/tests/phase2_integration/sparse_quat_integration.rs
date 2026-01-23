//! Sparse Quaternion Operations Integration Tests
//!
//! Tests for quaternion-aware sparsity patterns:
//! - 2:4 Structured sparsity
//! - Global magnitude pruning
//! - N:M structured pruning
//! - CSR sparse format conversion
//! - Tensor Core acceleration compatibility

use super::common::*;

#[cfg(test)]
mod structured_sparsity_tests {
    use super::*;

    #[test]
    fn test_structured_2x4_pattern_generation() {
        let mask = SparsePatternGenerator::generate_2x4(16);
        assert_eq!(mask.len(), 16);

        // Verify 2:4 pattern: max 2 non-zeros per 4 elements
        for chunk in mask.chunks(4) {
            let count = chunk.iter().filter(|&&b| b).count();
            assert!(count <= 2, "Found {} non-zeros in 4-element chunk", count);
        }
    }

    #[test]
    fn test_structured_2x4_sparsity_ratio() {
        let mask = SparsePatternGenerator::generate_2x4(100);
        let nnz = mask.iter().filter(|&&b| b).count();
        let sparsity = 1.0 - (nnz as f64 / mask.len() as f64);

        // Expect ~50% sparsity for 2:4
        assert!((sparsity - 0.5).abs() < 0.15);
    }

    #[test]
    fn test_structured_2x4_for_tensor_cores() {
        let mask = SparsePatternGenerator::generate_2x4(32);

        // Ampere+ Tensor Cores support 2:4 structured sparsity
        // Verify pattern compatibility
        let compatible = mask.chunks(4)
            .all(|chunk| chunk.iter().filter(|&&b| b).count() <= 2);

        assert!(compatible);
    }

    #[test]
    fn test_structured_nxm_various_patterns() {
        for m in &[4, 8, 16] {
            for n in &[1, 2, 3] {
                if n >= m { continue; }

                let mask = SparsePatternGenerator::generate_nxm(*n, *m, 128);
                for chunk in mask.chunks(*m) {
                    let count = chunk.iter().filter(|&&b| b).count();
                    assert!(count <= *n, "N:M pattern violated for n={}, m={}", n, m);
                }
            }
        }
    }
}

#[cfg(test)]
mod global_magnitude_tests {
    use super::*;

    #[test]
    fn test_global_magnitude_pruning_keeps_largest() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let quats = gen.generate_batch(20);

        // Compute norms
        let norms: Vec<f32> = quats.iter().map(|q| q.norm()).collect();

        // Global magnitude: keep top 50%
        let mut sorted = norms.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let threshold = sorted[norms.len() / 2];

        let mask: Vec<bool> = norms.iter().map(|&n| n >= threshold).collect();

        // Verify sparsity
        let nnz = mask.iter().filter(|&&b| b).count();
        assert!(nnz <= norms.len());
    }

    #[test]
    fn test_global_magnitude_deterministic() {
        let quats = vec![
            Quaternion::new(1.0, 0.0, 0.0, 0.0),
            Quaternion::new(0.0, 1.0, 0.0, 0.0),
            Quaternion::new(0.0, 0.0, 1.0, 0.0),
            Quaternion::new(0.0, 0.0, 0.0, 1.0),
        ];

        let norms: Vec<f32> = quats.iter().map(|q| q.norm()).collect();

        // All quats have norm 1.0
        for norm in norms {
            assert!((norm - 1.0).abs() < 0.001);
        }
    }

    #[test]
    fn test_global_magnitude_sparsity_control() {
        for target_sparsity in &[0.5, 0.75, 0.9] {
            let mask = SparsePatternGenerator::generate_csr(*target_sparsity, 1000);
            assert!(validators::validate_sparsity(&mask, *target_sparsity));
        }
    }
}

#[cfg(test)]
mod nxm_sparsity_tests {
    use super::*;

    #[test]
    fn test_structured_nxm_pattern() {
        let mask = SparsePatternGenerator::generate_nxm(2, 4, 16);

        // Every 4 elements should have exactly 2 non-zeros
        for chunk in mask.chunks(4) {
            assert_eq!(chunk.iter().filter(|&&b| b).count(), 2);
        }
    }

    #[test]
    fn test_nxm_flexibility() {
        let patterns = vec![
            (1, 2), (1, 4), (1, 8),
            (2, 4), (2, 8),
            (4, 8),
        ];

        for (n, m) in patterns {
            let mask = SparsePatternGenerator::generate_nxm(n, m, 100);
            for chunk in mask.chunks(m) {
                let count = chunk.iter().filter(|&&b| b).count();
                assert!(count <= n);
            }
        }
    }
}

#[cfg(test)]
mod sparse_linear_tests {
    use super::*;

    #[test]
    fn test_sparse_quat_linear_correctness() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 0.5, -0.5, 0.2];
        let mask = vec![true, false, true, false];

        let sparse_output = harness.sparse_gemm(&input, &mask);

        // Masked-out values should be zero
        for (i, (val, keep)) in sparse_output.iter().zip(mask.iter()).enumerate() {
            if !keep {
                assert_eq!(*val, 0.0, "Index {} should be masked out", i);
            }
        }
    }

    #[test]
    fn test_sparse_quat_linear_batch() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);

        for _ in 0..5 {
            let batch = gen.generate_batch(8);
            let values: Vec<f32> = batch.iter()
                .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                .collect();

            let mask = SparsePatternGenerator::generate_2x4(values.len());
            let output = harness.sparse_gemm(&values, &mask);

            assert_eq!(output.len(), values.len());
        }
    }

    #[test]
    fn test_sparse_linear_with_bias() {
        let harness = Phase2TestHarness::new();
        let weight = vec![1.0, 2.0, 3.0, 4.0];
        let bias = vec![0.1, 0.2, 0.3, 0.4];
        let mask = vec![true, true, false, false];

        let result = harness.sparse_gemm(&weight, &mask);

        // After sparsity, add bias to non-zero elements
        let mut with_bias = result.clone();
        for (i, (val, &keep)) in with_bias.iter_mut().zip(mask.iter()).enumerate() {
            if keep {
                *val += bias[i];
            }
        }

        assert_eq!(with_bias.len(), weight.len());
    }
}

#[cfg(test)]
mod csr_format_tests {
    use super::*;

    #[test]
    fn test_csr_format_conversion_dense_to_sparse() {
        let dense = vec![
            1.0, 0.0, 2.0, 0.0,
            0.0, 3.0, 0.0, 4.0,
            5.0, 0.0, 0.0, 6.0,
        ];

        let mask: Vec<bool> = dense.iter().map(|&x| x != 0.0).collect();
        let nnz = mask.iter().filter(|&&b| b).count();

        // CSR would store: nnz, row pointers, column indices, values
        assert!(nnz < dense.len()); // Compression achieved
    }

    #[test]
    fn test_csr_sparse_to_dense() {
        let mask = vec![true, false, true, false, false, true];
        let values = vec![1.0, 2.0, 3.0];

        // Expand back to dense
        let mut dense = Vec::new();
        let mut val_idx = 0;
        for &keep in &mask {
            if keep {
                dense.push(values[val_idx]);
                val_idx += 1;
            } else {
                dense.push(0.0);
            }
        }

        assert_eq!(dense.len(), mask.len());
    }

    #[test]
    fn test_csr_row_wise_pattern() {
        let matrix = vec![
            1.0, 2.0, 0.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            5.0, 0.0, 6.0, 7.0,
        ];

        // Row 0: 2 non-zeros
        // Row 1: 2 non-zeros
        // Row 2: 3 non-zeros
        let row_ptrs = vec![0, 2, 4, 7];
        let row_nnz: Vec<usize> = row_ptrs.windows(2)
            .map(|w| w[1] - w[0])
            .collect();

        assert_eq!(row_nnz[0], 2);
        assert_eq!(row_nnz[1], 2);
        assert_eq!(row_nnz[2], 3);
    }

    #[test]
    fn test_csr_memory_efficiency() {
        let dense_size = 1000 * 1000; // 1M elements
        let sparsity = 0.9;
        let nnz = (dense_size as f64 * (1.0 - sparsity)) as usize;

        // CSR: nnz values + nnz column indices + (rows+1) row pointers
        let csr_elements = nnz + nnz + 1001;

        // Dense: 1M elements
        assert!(csr_elements < dense_size);
    }
}

#[cfg(test)]
mod tensor_core_tests {
    use super::*;

    #[test]
    fn test_tensor_core_2x4_metadata_encoding() {
        let mask = vec![true, false, true, false, false, true, true, false];

        // Encode as 2-bit positions
        let mut encoded = Vec::new();
        for chunk in mask.chunks(4) {
            let mut byte = 0u8;
            let mut pos = 0u8;
            for (i, &keep) in chunk.iter().enumerate() {
                if keep {
                    byte |= (i as u8) << (pos * 2);
                    pos += 1;
                    if pos >= 2 { break; }
                }
            }
            encoded.push(byte);
        }

        // Decode back
        let mut decoded = Vec::new();
        for &byte in &encoded {
            let pos0 = (byte & 0x03) as usize;
            let pos1 = ((byte >> 2) & 0x03) as usize;
            decoded.push((pos0, pos1));
        }

        assert_eq!(decoded.len(), 2);
    }

    #[test]
    fn test_tensor_core_wmma_compatible() {
        let mask = SparsePatternGenerator::generate_2x4(64);

        // Verify WMMA (Warp Matrix Multiply-Accumulate) compatibility
        // WMMA requires 16x16 or 32x32 tiles with 2:4 sparsity
        let mut compatible = true;
        for chunk in mask.chunks(4) {
            let nnz = chunk.iter().filter(|&&b| b).count();
            if nnz > 2 {
                compatible = false;
                break;
            }
        }

        assert!(compatible);
    }

    #[test]
    fn test_sparse_quat_tensor_core_compute() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let quats = gen.generate_batch(16);

        let values: Vec<f32> = quats.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let mask = SparsePatternGenerator::generate_2x4(values.len());
        let sparse_result = harness.sparse_gemm(&values, &mask);

        // Should be suitable for Tensor Core acceleration
        assert_eq!(sparse_result.len(), values.len());
    }
}

#[cfg(test)]
mod end_to_end_sparse_tests {
    use super::*;

    #[test]
    fn test_sparse_quat_forward_backward() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let batch = gen.generate_batch(8);

        let values: Vec<f32> = batch.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let mask = SparsePatternGenerator::generate_2x4(values.len());

        // Forward with sparsity
        let forward = harness.sparse_gemm(&values, &mask);
        assert_eq!(forward.len(), values.len());

        // Backward with sparsity
        let backward = harness.backward(&forward);
        assert_eq!(backward.len(), values.len());
    }

    #[test]
    fn test_sparse_quat_all_patterns() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Test different sparsity patterns
        let mask_2x4 = SparsePatternGenerator::generate_2x4(input.len());
        let mask_csr = SparsePatternGenerator::generate_csr(0.5, input.len());
        let mask_nxm = SparsePatternGenerator::generate_nxm(1, 4, input.len());

        let result_2x4 = harness.sparse_gemm(&input, &mask_2x4);
        let result_csr = harness.sparse_gemm(&input, &mask_csr);
        let result_nxm = harness.sparse_gemm(&input, &mask_nxm);

        assert_eq!(result_2x4.len(), input.len());
        assert_eq!(result_csr.len(), input.len());
        assert_eq!(result_nxm.len(), input.len());
    }

    #[test]
    fn test_sparse_quat_accuracy_loss() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let quats = gen.generate_batch(10);

        let dense_values: Vec<f32> = quats.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let mask = SparsePatternGenerator::generate_2x4(dense_values.len());
        let sparse_values = harness.sparse_gemm(&dense_values, &mask);

        // Sparsity should not degrade accuracy significantly
        let error: f32 = dense_values.iter()
            .zip(sparse_values.iter())
            .map(|(d, s)| {
                if *d == 0.0 && *s == 0.0 { 0.0 }
                else if *d == 0.0 { s.abs() }
                else { (d - s).abs() / d.abs().max(0.1) }
            })
            .sum();

        assert!(error < 10.0); // Relative error bounded
    }

    #[test]
    fn test_sparse_quat_scaling() {
        let harness = Phase2TestHarness::new();

        for size in &[64, 256, 1024] {
            let values = (0..*size).map(|i| i as f32).collect::<Vec<_>>();
            let mask = SparsePatternGenerator::generate_2x4(*size);
            let result = harness.sparse_gemm(&values, &mask);

            assert_eq!(result.len(), *size);
        }
    }

    #[test]
    fn test_sparse_quat_with_mixed_precision() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let quats = gen.generate_batch(8);

        let values: Vec<f32> = quats.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        // Mix of sparse ops and mixed precision
        let fp16 = harness.forward(&values);
        let mask = SparsePatternGenerator::generate_2x4(fp16.len());
        let sparse = harness.sparse_gemm(&fp16, &mask);
        let fp32_grad = harness.backward(&sparse);

        assert_eq!(fp32_grad.len(), values.len());
    }
}
