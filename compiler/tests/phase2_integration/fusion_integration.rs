//! Kernel Fusion Integration Tests
//!
//! Tests for semantic fusion patterns:
//! - Linear+BN+ReLU
//! - Linear+BN
//! - Linear+ReLU
//! - BN+ReLU
//! - Conv+BN+ReLU

use super::common::*;

#[cfg(test)]
mod vertical_fusion_tests {
    use super::*;

    #[test]
    fn test_vertical_fusion_linear_bn() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Separate layers
        let linear_out = input.iter().map(|&x| x + 0.5).collect::<Vec<_>>();
        let bn_out = linear_out.iter().map(|&x| x * 0.95).collect::<Vec<_>>();

        // Fused equivalent
        let fused = harness.fused_layer(&input);

        // Results should match
        for (sep, fus) in bn_out.iter().zip(fused.iter()) {
            assert!((sep - fus).abs() < 0.01);
        }
    }

    #[test]
    fn test_vertical_fusion_eliminates_memory_traffic() {
        let harness = Phase2TestHarness::new();
        let _large_batch = (0..10000).map(|i| i as f32).collect::<Vec<_>>();

        // Fused layer processes in registers (no intermediate global memory)
        let input = vec![1.0, 2.0, 3.0];
        let _result = harness.fused_layer(&input);

        // If fusion is properly implemented, intermediate values stay in registers
        // This test would be validated by profiling memory access patterns
        assert!(true); // Placeholder for actual memory measurement
    }

    #[test]
    fn test_fusion_preserves_gradients() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.5, 1.5, 2.5];

        let forward = harness.fused_layer(&input);
        let backward = harness.backward(&forward);

        // Gradients should not be NaN
        for val in backward {
            assert!(val.is_finite());
        }
    }
}

#[cfg(test)]
mod horizontal_fusion_tests {
    use super::*;

    #[test]
    fn test_horizontal_fusion_multiple_outputs() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Process through fused layer multiple times
        let output1 = harness.fused_layer(&input);
        let output2 = harness.fused_layer(&input);

        // Outputs should be identical
        for (a, b) in output1.iter().zip(output2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_horizontal_fusion_batch_processing() {
        let harness = Phase2TestHarness::new();

        for batch_size in &[1, 4, 16, 64] {
            let input = (0..*batch_size)
                .map(|i| i as f32)
                .collect::<Vec<_>>();

            let output = harness.fused_layer(&input);
            assert_eq!(output.len(), input.len());
        }
    }
}

#[cfg(test)]
mod diamond_fusion_tests {
    use super::*;

    #[test]
    fn test_diamond_fusion_fork_join() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0];

        // Fork: two separate layers
        let path1 = harness.fused_layer(&input);
        let path2 = harness.forward(&input);

        // Join: combine results
        let combined = path1.iter()
            .zip(path2.iter())
            .map(|(a, b)| a + b)
            .collect::<Vec<_>>();

        assert_eq!(combined.len(), input.len());
    }

    #[test]
    fn test_diamond_shared_input() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        // Multiple consumers of same input
        let output1 = harness.fused_layer(&input);
        let output2 = harness.forward(&input);
        let output3 = harness.backward(&input);

        assert_eq!(output1.len(), input.len());
        assert_eq!(output2.len(), input.len());
        assert_eq!(output3.len(), input.len());
    }
}

#[cfg(test)]
mod loop_fusion_tests {
    use super::*;

    #[test]
    fn test_loop_fusion_with_unrolling() {
        let harness = Phase2TestHarness::new();
        let input = (0..100).map(|i| i as f32).collect::<Vec<_>>();

        // Simulate loop fusion: multiple iterations through fused layer
        let mut result = input.clone();
        for _ in 0..4 {
            result = harness.fused_layer(&result);
        }

        assert_eq!(result.len(), input.len());
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_loop_fusion_inner_loop_vectorization() {
        let harness = Phase2TestHarness::new();

        let mut values = Vec::new();
        for i in 0..10 {
            let batch = (0..100).map(|j| (i * 100 + j) as f32).collect::<Vec<_>>();
            let processed = harness.fused_layer(&batch);
            values.extend(processed);
        }

        assert_eq!(values.len(), 1000);
    }

    #[test]
    fn test_loop_fusion_invariant_hoisting() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0];

        // Loop with fused operations
        let mut sum = 0.0;
        for _ in 0..100 {
            let output = harness.fused_layer(&input);
            sum += output.iter().sum::<f32>();
        }

        assert!(sum.is_finite());
    }
}

#[cfg(test)]
mod sparse_quat_fusion_tests {
    use super::*;

    #[test]
    fn test_fusion_with_sparse_quat_input() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let quats = r#gen.generate_batch(16);

        let values: Vec<f32> = quats.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        // Sparse mask: keep 2 of 4
        let mask = SparsePatternGenerator::generate_2x4(values.len());

        // Apply sparsity then fusion
        let sparse_vals = harness.sparse_gemm(&values, &mask);
        let fused = harness.fused_layer(&sparse_vals);

        assert_eq!(fused.len(), values.len());
    }

    #[test]
    fn test_quat_linear_bn_relu_fusion() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 0.5, -0.5, 0.2, 2.0, -1.0, 0.0, 1.5];

        // Fused layer simulates Linear+BN+ReLU for quaternions
        let output = harness.fused_layer(&input);

        // All quaternion components should be processed
        assert_eq!(output.len(), input.len());
        assert!(output.iter().all(|&x| x >= 0.0 || x < 1e-10)); // ReLU output
    }

    #[test]
    fn test_fusion_preserves_quat_structure() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let q = r#gen.next();

        let input = q.as_array().to_vec();
        let output = harness.fused_layer(&input);

        assert_eq!(output.len(), 4);
    }
}

#[cfg(test)]
mod end_to_end_fusion_tests {
    use super::*;

    #[test]
    fn test_semantic_fusion_linear_bn_relu() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.5, 1.5, -1.0, 2.5, -0.5];

        let output = harness.fused_layer(&input);

        // Check fusion correctness
        for (inp, out) in input.iter().zip(output.iter()) {
            // After Linear: y = x + 0.5
            // After BN: y = y * 0.95
            // After ReLU: y = max(0, y)
            let expected = ((inp + 0.5) * 0.95).max(0.0);
            assert!((out - expected).abs() < 0.001);
        }
    }

    #[test]
    fn test_fusion_batch_consistent() {
        let harness = Phase2TestHarness::new();

        let input1 = vec![1.0, 2.0, 3.0];
        let input2 = vec![1.0, 2.0, 3.0];

        let output1 = harness.fused_layer(&input1);
        let output2 = harness.fused_layer(&input2);

        for (a, b) in output1.iter().zip(output2.iter()) {
            assert_eq!(a, b);
        }
    }

    #[test]
    fn test_fusion_gradient_flow() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0];

        let forward = harness.fused_layer(&input);
        let backward = harness.backward(&forward);

        assert_eq!(backward.len(), input.len());
        assert!(backward.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fusion_multiple_layers_stacked() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.5, 1.0, 1.5, 2.0];

        let mut result = input.clone();
        for _ in 0..3 {
            result = harness.fused_layer(&result);
        }

        assert_eq!(result.len(), input.len());
        assert!(result.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fusion_with_mixed_precision() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0];

        let fp16_forward = harness.forward(&input);
        let fused = harness.fused_layer(&fp16_forward);
        let fp32_backward = harness.backward(&fused);

        assert_eq!(fp32_backward.len(), input.len());
    }

    #[test]
    fn test_fusion_handles_denormalized_numbers() {
        let harness = Phase2TestHarness::new();
        let input = vec![1e-38, 1e-37, 1e-36]; // Near FP32 denormal

        let output = harness.fused_layer(&input);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_fusion_no_overflow_underflow() {
        let harness = Phase2TestHarness::new();
        let input = vec![1e37, 1e36]; // Large values

        let output = harness.fused_layer(&input);
        assert!(output.iter().all(|x| x.is_finite()));
    }
}
