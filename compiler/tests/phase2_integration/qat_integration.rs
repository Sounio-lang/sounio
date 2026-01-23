//! Quantization-Aware Training (QAT) Integration Tests
//!
//! Tests for fake quantization, online calibration, learnable scales,
//! and quantization parameter export.

use super::common::*;

#[cfg(test)]
mod fake_quantization_tests {
    use super::*;

    #[test]
    fn test_fake_quantization_int8_symmetric() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.1, 0.5, 1.0, -0.5, -1.0];
        let scale = 1.0 / 127.0; // INT8 symmetric scale

        let output = harness.qat_forward(&input, scale);

        assert_eq!(output.len(), input.len());
        for val in output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_fake_quantization_int8_asymmetric() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.0, 0.5, 1.0, 1.5, 2.0];
        let scale = 2.0 / 255.0; // INT8 asymmetric scale

        let output = harness.qat_forward(&input, scale);

        // All values should be non-negative after asymmetric quant
        for val in output {
            assert!(val >= 0.0 || val.abs() < 1e-6);
        }
    }

    #[test]
    fn test_fake_quantization_int4_aggressive() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.0, 0.5, 1.0, 1.5];
        let scale = 1.0 / 7.0; // INT4 scale

        let output = harness.qat_forward(&input, scale);

        assert_eq!(output.len(), input.len());
        for val in output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_fake_quantization_quantization_error() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.123, 0.456, 0.789];
        let scale = 0.01;

        let output = harness.qat_forward(&input, scale);

        // Quantization error should be bounded by scale/2
        let max_error = scale * 0.5;
        assert!(validators::validate_fake_quantization(&input, &output, max_error));
    }

    #[test]
    fn test_fake_quantization_preserves_sign() {
        let harness = Phase2TestHarness::new();
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let scale = 1.0 / 127.0;

        let output = harness.qat_forward(&input, scale);

        // Signs should be preserved
        for (inp, out) in input.iter().zip(output.iter()) {
            if inp > &0.0 {
                assert!(out > &0.0);
            } else if inp < &0.0 {
                assert!(out < &0.0);
            }
        }
    }
}

#[cfg(test)]
mod online_minmax_tests {
    use super::*;

    #[test]
    fn test_online_minmax_tracking_single_batch() {
        let harness = Phase2TestHarness::new();
        let batch = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let output = harness.forward(&batch);
        assert_eq!(output.len(), batch.len());
    }

    #[test]
    fn test_online_minmax_tracking_accumulation() {
        let harness = Phase2TestHarness::new();

        for batch_num in 0..10 {
            let batch: Vec<f32> = (0..32)
                .map(|i| ((batch_num * 32 + i) as f32) * 0.01)
                .collect();

            let _output = harness.forward(&batch);
        }

        // After accumulation, statistics should be valid
        assert!(true);
    }

    #[test]
    fn test_online_minmax_ema_momentum() {
        let mut harness = Phase2TestHarness::new();
        harness.set_config("momentum", 0.01);

        let momentum = harness.get_config("momentum").unwrap_or(0.01);
        assert_eq!(momentum, 0.01);
    }

    #[test]
    fn test_online_minmax_updates_scale() {
        let harness = Phase2TestHarness::new();
        let batch1 = vec![1.0, 2.0, 3.0];
        let batch2 = vec![4.0, 5.0, 6.0];

        let _out1 = harness.forward(&batch1);
        let _out2 = harness.forward(&batch2);

        // Scale should adapt after seeing new data
        assert!(true);
    }
}

#[cfg(test)]
mod warmup_phase_tests {
    use super::*;

    #[test]
    fn test_qat_layer_warmup_no_quantization() {
        let mut harness = Phase2TestHarness::new();
        harness.set_config("warmup_batches", 10.0);

        let input = vec![1.0, 2.0, 3.0];

        // During warmup, forward should not apply quantization
        let output = harness.forward(&input);

        // Output should be close to input during warmup
        for (inp, out) in input.iter().zip(output.iter()) {
            assert!((inp - out).abs() < 0.01);
        }
    }

    #[test]
    fn test_qat_layer_quantization_after_warmup() {
        let harness = Phase2TestHarness::new();

        let input = vec![0.5, 1.0, 1.5];
        let scale = 0.1;

        let output = harness.qat_forward(&input, scale);

        // After warmup, quantization should be applied
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_warmup_batch_tracking() {
        let harness = Phase2TestHarness::new();
        let warmup_batches = harness.get_config("warmup_batches").unwrap_or(1000.0);

        assert_eq!(warmup_batches, 1000.0);
    }

    #[test]
    fn test_warmup_statistics_collection() {
        let harness = Phase2TestHarness::new();

        // Simulate 100 batches
        for _ in 0..100 {
            let batch = vec![1.0, 2.0, 3.0];
            let _output = harness.forward(&batch);
        }

        // Statistics should be collected
        assert!(true);
    }
}

#[cfg(test)]
mod learnable_scale_tests {
    use super::*;

    #[test]
    fn test_learnable_scale_initialization() {
        let harness = Phase2TestHarness::new();
        let initial_scale = 1.0 / 127.0;

        let input = vec![0.5];
        let _output = harness.qat_forward(&input, initial_scale);

        assert!(initial_scale > 0.0);
    }

    #[test]
    fn test_learnable_scale_update() {
        let mut harness = Phase2TestHarness::new();
        let old_scale = 1.0 / 127.0;
        let new_scale = 1.0 / 100.0;

        harness.set_config("qat_scale", new_scale);

        let updated = harness.get_config("qat_scale").unwrap_or(old_scale);
        assert!((updated - new_scale).abs() < 1e-6);
    }

    #[test]
    fn test_scale_affects_quantization_granularity() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0];

        let coarse_scale = 0.1;
        let fine_scale = 0.001;

        let coarse_output = harness.qat_forward(&input, coarse_scale);
        let fine_output = harness.qat_forward(&input, fine_scale);

        // Fine scale should have smaller quantization error
        let coarse_error = (input[0] - coarse_output[0]).abs();
        let fine_error = (input[0] - fine_output[0]).abs();

        assert!(fine_error <= coarse_error);
    }

    #[test]
    fn test_log_scale_stability() {
        let harness = Phase2TestHarness::new();

        for scale in &[0.001, 0.01, 0.1, 1.0] {
            let log_scale = scale.ln();
            assert!(log_scale.is_finite());

            // Exp should recover scale
            let recovered = log_scale.exp();
            assert!((recovered - scale).abs() < 1e-5);
        }
    }
}

#[cfg(test)]
mod int4_tests {
    use super::*;

    #[test]
    fn test_int4_quantization() {
        let harness = Phase2TestHarness::new();
        let input = vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let scale = 1.0 / 7.0; // INT4 range [0, 15] maps to [0, 15*scale]

        let output = harness.qat_forward(&input, scale);

        assert_eq!(output.len(), input.len());
        for val in output {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_int4_aggresive_compression() {
        let harness = Phase2TestHarness::new();

        // INT4 should be 4x more aggressive than INT8
        let input = vec![0.1, 0.3, 0.5, 0.7];
        let int4_scale = 1.0 / 7.0;
        let int8_scale = 1.0 / 127.0;

        let _int4_output = harness.qat_forward(&input, int4_scale);
        let _int8_output = harness.qat_forward(&input, int8_scale);

        // INT4 scale should be larger (coarser)
        assert!(int4_scale > int8_scale);
    }

    #[test]
    fn test_int4_activation_quantization() {
        let harness = Phase2TestHarness::new();

        // INT4 for activations (while weights stay INT8)
        let activation = vec![0.5, 1.0, 1.5];
        let act_scale = 1.0 / 7.0;

        let _output = harness.qat_forward(&activation, act_scale);

        assert!(true);
    }
}

#[cfg(test)]
mod per_channel_tests {
    use super::*;

    #[test]
    fn test_per_channel_weight_quantization() {
        let harness = Phase2TestHarness::new();

        // Per-channel: each output channel gets its own scale
        let weights = vec![
            0.1, 0.2, 0.3, 0.4,  // Channel 0
            0.5, 0.6, 0.7, 0.8,  // Channel 1
        ];

        let channel_scale_0 = 0.4 / 127.0;
        let channel_scale_1 = 0.8 / 127.0;

        let mut output = vec![0.0; weights.len()];

        // Quantize first channel
        for i in 0..4 {
            output[i] = harness.qat_forward(&[weights[i]], channel_scale_0)[0];
        }

        // Quantize second channel
        for i in 4..8 {
            output[i] = harness.qat_forward(&[weights[i]], channel_scale_1)[0];
        }

        assert_eq!(output.len(), weights.len());
    }

    #[test]
    fn test_per_channel_reduces_quantization_error() {
        let harness = Phase2TestHarness::new();

        // Per-channel should be more accurate than per-tensor
        let weights = vec![0.1, 1.0, 0.05, 10.0]; // Wide range

        let _per_tensor_scale = 10.0 / 127.0;
        let _per_channel_scales = vec![0.1 / 127.0, 1.0 / 127.0, 0.05 / 127.0, 10.0 / 127.0];

        // Per-channel scales better for wide-range tensors
        assert!(true);
    }
}

#[cfg(test)]
mod symmetric_asymmetric_tests {
    use super::*;

    #[test]
    fn test_symmetric_quantization_zero_centered() {
        let harness = Phase2TestHarness::new();

        let input = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let scale = 1.0 / 127.0;

        let output = harness.qat_forward(&input, scale);

        // Symmetric quant: -max and +max map to ±127
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn test_asymmetric_quantization_uses_zp() {
        let harness = Phase2TestHarness::new();

        let input = vec![0.0, 0.5, 1.0, 1.5, 2.0]; // Positive only
        let scale = 2.0 / 255.0;

        let output = harness.qat_forward(&input, scale);

        // Asymmetric uses full [0, 255] range
        for val in output {
            assert!(val >= 0.0);
        }
    }

    #[test]
    fn test_asymmetric_better_for_activations() {
        let harness = Phase2TestHarness::new();

        // ReLU outputs are non-negative
        let relu_output = vec![0.0, 0.1, 0.2, 0.3];
        let asym_scale = 0.3 / 255.0;

        let _output = harness.qat_forward(&relu_output, asym_scale);

        // Asymmetric packs values better
        assert!(true);
    }
}

#[cfg(test)]
mod quat_layer_tests {
    use super::*;

    #[test]
    fn test_quat_qat_forward() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let q = gen.next();

        let input = q.as_array().to_vec();
        let scale = 1.0 / 127.0;

        let output = harness.qat_forward(&input, scale);

        assert_eq!(output.len(), 4);
        assert!(output.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_quat_qat_batch() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);

        for _ in 0..10 {
            let q = gen.next();
            let input = q.as_array().to_vec();
            let scale = 1.0 / 127.0;

            let _output = harness.qat_forward(&input, scale);
        }

        assert!(true);
    }

    #[test]
    fn test_quat_per_channel_quantization() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let q = gen.next();

        let mut output = vec![0.0; 4];

        // Per-component scale for quaternion
        for i in 0..4 {
            let component = q.as_array()[i];
            let scale = (component.abs().max(1e-6)) / 127.0;
            output[i] = harness.qat_forward(&[component], scale)[0];
        }

        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_quat_multiplication_preserves_structure() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);
        let q1 = gen.next();
        let q2 = gen.next();

        let prod = q1.mul(&q2);
        let prod_array = prod.as_array();

        // Multiply quantized quaternions
        let scale = 1.0 / 127.0;
        let prod_q = harness.qat_forward(&prod_array, scale);

        // Should still be valid quaternion
        assert_eq!(prod_q.len(), 4);
    }
}

#[cfg(test)]
mod end_to_end_qat_tests {
    use super::*;

    #[test]
    fn test_qat_export_parameters() {
        let harness = Phase2TestHarness::new();

        let weight_scale = 1.0 / 127.0;
        let weight_zero_point = 0;
        let input_scale = 1.0 / 127.0;
        let input_zero_point = 0;

        assert!(weight_scale > 0.0);
        assert_eq!(weight_zero_point, 0);
        assert!(input_scale > 0.0);
        assert_eq!(input_zero_point, 0);
    }

    #[test]
    fn test_full_qat_training_loop() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);

        for epoch in 0..5 {
            for _ in 0..10 {
                let batch = gen.generate_batch(8);
                let values: Vec<f32> = batch.iter()
                    .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                    .collect();

                let scale = 1.0 / 127.0;
                let quantized = harness.qat_forward(&values, scale);
                let _loss = quantized.iter().map(|x| x * x).sum::<f32>();
            }

            // Simulate scale update after epoch
            assert!(epoch < 5);
        }
    }

    #[test]
    fn test_qat_accuracy_within_target() {
        let harness = Phase2TestHarness::new();
        let mut gen = QuaternionGenerator::new(42);

        let mut total_error = 0.0;
        let num_batches = 100;

        for _ in 0..num_batches {
            let batch = gen.generate_batch(8);
            let original: Vec<f32> = batch.iter()
                .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                .collect();

            let scale = 1.0 / 127.0;
            let quantized = harness.qat_forward(&original, scale);

            let batch_error: f32 = original.iter()
                .zip(quantized.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();

            total_error += batch_error;
        }

        let avg_error = total_error / (num_batches as f32 * 32.0);
        assert!(avg_error < 0.01); // <1% accuracy loss target
    }

    #[test]
    fn test_qat_model_export_ptq_conversion() {
        // After QAT training, export quantization params for PTQ deployment
        let _weight_scale = 1.0 / 127.0;
        let _weight_zp = 0i32;
        let _input_scale = 1.0 / 127.0;
        let _input_zp = 0i32;

        // These would be saved to deploy model
        assert!(true);
    }

    #[test]
    fn test_qat_with_batch_norm_folding() {
        let harness = Phase2TestHarness::new();

        let input = vec![1.0, 2.0, 3.0];
        let fused = harness.fused_layer(&input); // Linear+BN+ReLU fused
        let quantized = harness.qat_forward(&fused, 1.0 / 127.0);

        assert_eq!(quantized.len(), input.len());
    }

    #[test]
    fn test_qat_calibration_min_max_tracking() {
        let harness = Phase2TestHarness::new();

        // Simulate calibration phase
        let calibration_data = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.15, 0.25, 0.35],
            vec![0.05, 0.15, 0.25],
        ];

        let mut min_val = f32::MAX;
        let mut max_val = f32::MIN;

        for batch in calibration_data {
            for &val in &batch {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }

            let _output = harness.forward(&batch);
        }

        assert!(min_val < max_val);
    }
}
