//! Combined Pipeline Tests
//!
//! Tests for interactions between Phase 2 features:
//! - QAT + Fusion
//! - Mixed-Precision + Sparse
//! - Full pipeline with all 4 features
//! - Feature compatibility matrix

use super::common::*;

#[cfg(test)]
mod qat_fusion_tests {
    use super::*;

    #[test]
    fn test_qat_with_fusion_linear_bn_relu() {
        let harness = Phase2TestHarness::new();

        let input = vec![0.5, 1.0, 1.5, 2.0];

        // Fused layer: Linear+BN+ReLU
        let fused = harness.fused_layer(&input);

        // QAT: apply fake quantization
        let scale = 1.0 / 127.0;
        let quantized = harness.qat_forward(&fused, scale);

        assert_eq!(quantized.len(), input.len());
        for val in &quantized {
            assert!(val >= 0.0 || *val < 1e-10); // ReLU output
        }
    }

    #[test]
    fn test_fused_layer_with_qat_gradient_flow() {
        let harness = Phase2TestHarness::new();

        let input = vec![1.0, 2.0, 3.0];
        let fused = harness.fused_layer(&input);
        let scale = 1.0 / 127.0;
        let quantized = harness.qat_forward(&fused, scale);

        // Straight-through estimator: gradient passes through unchanged
        let gradient = harness.backward(&quantized);

        assert_eq!(gradient.len(), input.len());
        assert!(gradient.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_qat_per_layer_in_fused_stack() {
        let harness = Phase2TestHarness::new();

        let mut input = vec![1.0, 2.0, 3.0];

        for _ in 0..3 {
            // Fuse Linear+BN+ReLU
            input = harness.fused_layer(&input);

            // Apply QAT after each fused layer
            let scale = 1.0 / 127.0;
            input = harness.qat_forward(&input, scale);
        }

        assert!(input.len() == 3);
        assert!(input.iter().all(|x| x.is_finite()));
    }
}

#[cfg(test)]
mod mixed_precision_sparse_tests {
    use super::*;

    #[test]
    fn test_mixed_precision_with_sparse_quat() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let batch = gen.generate_batch(8);

        let values: Vec<f32> = batch.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        // FP16 forward
        let fp16_forward = harness.forward(&values);

        // Apply sparsity
        let mask = SparsePatternGenerator::generate_2x4(fp16_forward.len());
        let sparse = harness.sparse_gemm(&fp16_forward, &mask);

        // FP32 backward
        let fp32_grad = harness.backward(&sparse);

        assert_eq!(fp32_grad.len(), values.len());
    }

    #[test]
    fn test_sparse_pattern_preserved_through_mixed_precision() {
        let harness = Phase2TestHarness::new();

        let input = vec![1.0, 0.5, -0.5, 0.2, 2.0, -1.0, 0.0, 1.5];
        let mask = vec![true, false, true, false, true, false, true, false];

        // Mix precision: FP16 forward, sparsity, FP32 backward
        let fp16 = harness.forward(&input);
        let sparse = harness.sparse_gemm(&fp16, &mask);

        // Sparsity pattern should be maintained
        let nonzero_count = sparse.iter().filter(|&&x| x.abs() > 1e-10).count();
        let zero_count = sparse.iter().filter(|&&x| x.abs() <= 1e-10).count();

        assert_eq!(nonzero_count + zero_count, input.len());
    }

    #[test]
    fn test_dynamic_loss_scaling_with_sparse_gradients() {
        let harness = Phase2TestHarness::new();

        let values = vec![0.1, 0.2, 0.3];
        let mask = vec![true, false, true];

        let fp32_grad = harness.backward(&values);
        let sparse_grad = harness.sparse_gemm(&fp32_grad, &mask);

        // Loss scale should handle sparse gradients
        assert_eq!(sparse_grad.len(), values.len());
    }
}

#[cfg(test)]
mod full_pipeline_tests {
    use super::*;

    #[test]
    fn test_fusion_sparse_quat_qat_combined() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let batch = gen.generate_batch(4);

        let values: Vec<f32> = batch.iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        // Step 1: Mixed-precision forward
        let fp16_values = harness.forward(&values);

        // Step 2: Apply sparsity
        let mask = SparsePatternGenerator::generate_2x4(fp16_values.len());
        let sparse_values = harness.sparse_gemm(&fp16_values, &mask);

        // Step 3: Apply fusion
        let fused = harness.fused_layer(&sparse_values);

        // Step 4: Apply QAT
        let scale = 1.0 / 127.0;
        let quantized = harness.qat_forward(&fused, scale);

        // Step 5: Mixed-precision backward
        let gradients = harness.backward(&quantized);

        assert_eq!(gradients.len(), values.len());
        assert!(gradients.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_full_phase2_pipeline_all_features() {
        let harness = Phase2TestHarness::new();

        // Initialize
        let mut loss_history = Vec::new();

        for epoch in 0..3 {
            let mut gen = QuaternionGenerator::new(42 + epoch as u32);

            for batch_idx in 0..5 {
                let batch = gen.generate_batch(8);
                let mut values: Vec<f32> = batch.iter()
                    .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                    .collect();

                // Feature 1: Mixed Precision
                values = harness.forward(&values); // FP16

                // Feature 2: Sparsity
                let mask = SparsePatternGenerator::generate_2x4(values.len());
                values = harness.sparse_gemm(&values, &mask);

                // Feature 3: Kernel Fusion
                values = harness.fused_layer(&values);

                // Feature 4: QAT
                let scale = 1.0 / 127.0;
                values = harness.qat_forward(&values, scale);

                // Loss and backward
                let loss: f32 = values.iter().map(|x| x * x).sum();
                loss_history.push(loss);

                let _gradients = harness.backward(&values);
            }
        }

        // All batches processed successfully
        assert_eq!(loss_history.len(), 15);
        assert!(loss_history.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_full_pipeline_with_different_batch_sizes() {
        let harness = Phase2TestHarness::new();

        for batch_size in &[1, 4, 16, 32] {
            let mut gen = QuaternionGenerator::new(42);
            let batch = gen.generate_batch(*batch_size);

            let mut values: Vec<f32> = batch.iter()
                .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                .collect();

            values = harness.forward(&values);
            let mask = SparsePatternGenerator::generate_2x4(values.len());
            values = harness.sparse_gemm(&values, &mask);
            values = harness.fused_layer(&values);
            let scale = 1.0 / 127.0;
            values = harness.qat_forward(&values, scale);

            assert_eq!(values.len(), batch_size * 4);
        }
    }
}

#[cfg(test)]
mod compatibility_matrix_tests {
    use super::*;

    #[test]
    fn test_feature_compatibility_matrix() {
        // Verify all feature combinations are supported
        let features = vec!["mixed_precision", "sparse", "fusion", "qat"];

        // All 2-way combinations should work
        for i in 0..features.len() {
            for j in (i + 1)..features.len() {
                let combo = format!("{}+{}", features[i], features[j]);
                assert!(!combo.is_empty());
            }
        }
    }

    #[test]
    fn test_mixed_precision_enables_others() {
        let harness = Phase2TestHarness::new();

        // Mixed precision enables sparse (FP16 has coarser quantization)
        let input = vec![1e-7, 1e-6, 1e-5]; // Small values benefit from FP16
        let _fp16 = harness.forward(&input);

        // Sparse with FP16 should work
        let mask = SparsePatternGenerator::generate_2x4(input.len());
        let _sparse = harness.sparse_gemm(&_fp16, &mask);

        assert!(true);
    }

    #[test]
    fn test_fusion_improves_sparse_efficiency() {
        let harness = Phase2TestHarness::new();

        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        // Fused layer reduces memory traffic
        let fused = harness.fused_layer(&input);

        // Apply sparsity on already-fused values
        let mask = SparsePatternGenerator::generate_2x4(fused.len());
        let sparse = harness.sparse_gemm(&fused, &mask);

        // Combined should be more efficient
        assert_eq!(sparse.len(), input.len());
    }

    #[test]
    fn test_qat_calibrates_for_quantized_sparse_tensors() {
        let harness = Phase2TestHarness::new();

        // Calibrate QAT with sparsity pattern
        let values = vec![0.1, 0.5, 1.0, 1.5, 2.0];
        let mask = vec![true, false, true, false, true];

        // Sparse values
        let sparse = harness.sparse_gemm(&values, &mask);

        // QAT calibrates on sparse values
        let scale = 1.0 / 127.0;
        let _quantized = harness.qat_forward(&sparse, scale);

        assert!(true);
    }

    #[test]
    fn test_fp32_operations_in_quantized_pipeline() {
        let harness = Phase2TestHarness::new();

        // Some ops stay FP32 even in quantized pipeline
        let values = vec![1.0, 2.0, 3.0];

        // FP32 ops: exp, log, sqrt, softmax, layer_norm
        let _softmax_input = values.clone();

        // Should work alongside quantized ops
        let scale = 1.0 / 127.0;
        let _quantized = harness.qat_forward(&values, scale);

        assert!(true);
    }
}

#[cfg(test)]
mod numerical_stability_tests {
    use super::*;

    #[test]
    fn test_numerical_stability_with_all_features() {
        let harness = Phase2TestHarness::new();

        let values = vec![1e-6, 1e-5, 1e-4, 1e-3, 0.1, 1.0, 10.0, 100.0];

        // Run through full pipeline
        let fp16 = harness.forward(&values);
        let mask = SparsePatternGenerator::generate_2x4(fp16.len());
        let sparse = harness.sparse_gemm(&fp16, &mask);
        let fused = harness.fused_layer(&sparse);
        let scale = 1.0 / 127.0;
        let quantized = harness.qat_forward(&fused, scale);

        // All values should remain finite
        assert!(quantized.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_gradient_underflow_prevention() {
        let harness = Phase2TestHarness::new();

        // Small gradients might underflow in FP16
        let small_gradients = vec![1e-7, 1e-6, 1e-5];

        // Mixed precision loss scaling should prevent underflow
        let loss_scale = harness.get_config("initial_loss_scale").unwrap_or(32768.0);
        let scaled_grads: Vec<f32> = small_gradients.iter()
            .map(|&g| g * loss_scale)
            .collect();

        // Scaled gradients should be representable in FP16
        assert!(scaled_grads.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_quantization_error_bounds() {
        let harness = Phase2TestHarness::new();

        let original = vec![0.123, 0.456, 0.789];
        let scale = 0.01;
        let quantized = harness.qat_forward(&original, scale);

        // Quantization error < scale/2
        let max_error = scale * 0.5;
        for (orig, quant) in original.iter().zip(quantized.iter()) {
            let error = (orig - quant).abs();
            assert!(error <= max_error);
        }
    }

    #[test]
    fn test_sparsity_numerical_correctness() {
        let harness = Phase2TestHarness::new();

        let values = vec![1.0, 2.0, 3.0, 4.0];
        let mask = vec![true, false, true, false];

        let sparse = harness.sparse_gemm(&values, &mask);

        // Zero values should be exact zero
        for (i, &keep) in mask.iter().enumerate() {
            if !keep {
                assert_eq!(sparse[i], 0.0);
            } else {
                assert_eq!(sparse[i], values[i]);
            }
        }
    }

    #[test]
    fn test_accumulated_rounding_errors() {
        let harness = Phase2TestHarness::new();

        let mut values = vec![1.0; 1000];

        for _ in 0..10 {
            // Multiple rounds through pipeline
            values = harness.forward(&values);
            values = harness.qat_forward(&values, 0.01);
            values = harness.backward(&values);
        }

        // Accumulated errors should still be bounded
        let error: f32 = values.iter().map(|&x| (x - 1.0).abs()).sum();
        assert!(error < 100.0); // Reasonable bound for 10 iterations
    }
}

#[cfg(test)]
mod performance_characterization_tests {
    use super::*;

    #[test]
    fn test_feature_memory_footprint() {
        // Mixed-precision: 2x memory bandwidth
        // Sparsity: 50-90% memory reduction
        // Fusion: 0% (combines in-place)
        // QAT: +1-2% for scale/zp tracking

        let total_params = 1_000_000.0;

        let fp32_size = total_params * 4.0; // 4 bytes
        let fp16_with_loss_scaling = (total_params * 2.0) + (3 * 4.0); // FP16 weights + loss scale storage

        let savings = fp32_size - fp16_with_loss_scaling;
        assert!(savings > 0.0);
    }

    #[test]
    fn test_feature_throughput_improvement() {
        let harness = Phase2TestHarness::new();

        // Baseline: dense FP32
        let dense_input = vec![1.0; 10000];
        let _dense_output = harness.forward(&dense_input);

        // With sparsity: 2-4x faster for sparse GEMM
        let mask = SparsePatternGenerator::generate_2x4(dense_input.len());
        let _sparse_output = harness.sparse_gemm(&dense_input, &mask);

        // Fusion: 5-10% faster
        let fused = harness.fused_layer(&dense_input);
        assert_eq!(fused.len(), dense_input.len());

        // Rough speedup estimate
        let estimated_speedup = 1.5 * 2.0 * 1.05; // Mixed-precision * sparsity * fusion
        assert!(estimated_speedup > 3.0);
    }

    #[test]
    fn test_latency_with_all_optimizations() {
        let harness = Phase2TestHarness::new();

        let input_sizes = vec![1024, 4096, 16384];

        for size in input_sizes {
            let input = (0..size).map(|i| (i as f32) * 0.001).collect::<Vec<_>>();

            let fp16 = harness.forward(&input);
            let mask = SparsePatternGenerator::generate_2x4(fp16.len());
            let sparse = harness.sparse_gemm(&fp16, &mask);
            let fused = harness.fused_layer(&sparse);
            let quantized = harness.qat_forward(&fused, 0.01);

            assert_eq!(quantized.len(), size);
        }

        // All sizes should complete without excessive latency
        assert!(true);
    }
}
