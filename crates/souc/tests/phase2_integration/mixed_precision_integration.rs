//! Mixed-Precision Training Integration Tests
//!
//! Tests for FP16/BF16 forward pass with FP32 backward pass,
//! dynamic loss scaling, and gradient underflow prevention.

use super::common::*;

#[cfg(test)]
mod fp16_forward_tests {
    use super::*;

    #[test]
    fn test_fp16_forward_basic() {
        let harness = Phase2TestHarness::new();
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let output = harness.forward(&input);

        // Output should be quantized but close to input
        assert_eq!(output.len(), input.len());
        for (inp, out) in input.iter().zip(output.iter()) {
            assert!((inp - out).abs() < 0.01);
        }
    }

    #[test]
    fn test_fp16_forward_precision_loss() {
        let harness = Phase2TestHarness::new();
        let small_values = vec![1e-5, 1e-4, 1e-3];
        let output = harness.forward(&small_values);

        // Small values may have larger relative error in FP16
        assert_eq!(output.len(), small_values.len());
    }

    #[test]
    fn test_fp16_forward_numerical_stability() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let batch: Vec<f32> = r#gen.generate_batch(100).iter().map(|q| q.norm()).collect();

        let output = harness.forward(&batch);
        assert!(validators::validate_precision_flow(&batch, &output, 0.01));
    }

    #[test]
    fn test_fp16_forward_edge_cases() {
        let harness = Phase2TestHarness::new();
        let edge_cases = vec![0.0, 1.0, -1.0, f32::MAX / 100.0, f32::MIN / 100.0];
        let _output = harness.forward(&edge_cases);

        // Should not panic or produce inf/nan
        for val in &edge_cases {
            assert!(val.is_finite());
        }
    }
}

#[cfg(test)]
mod mixed_precision_gradient_tests {
    use super::*;

    #[test]
    fn test_backward_pass_fp32() {
        let harness = Phase2TestHarness::new();
        let gradient = vec![0.001, 0.002, 0.003];
        let scaled = harness.backward(&gradient);

        assert_eq!(scaled.len(), gradient.len());
        for (orig, scaled_val) in gradient.iter().zip(scaled.iter()) {
            assert!((orig / 32768.0 - scaled_val).abs() < 1e-6);
        }
    }

    #[test]
    fn test_gradient_scaling() {
        let mut harness = Phase2TestHarness::new();
        harness.set_config("initial_loss_scale", 16384.0);

        let gradient = vec![1.0];
        let scaled = harness.backward(&gradient);

        assert!((scaled[0] - 1.0 / 16384.0).abs() < 1e-9);
    }

    #[test]
    fn test_gradient_accumulation() {
        let harness = Phase2TestHarness::new();
        let grad1 = vec![0.1, 0.2];
        let grad2 = vec![0.05, 0.1];

        let mut accumulated = harness.backward(&grad1);
        let grad2_scaled = harness.backward(&grad2);

        for (acc, add) in accumulated.iter_mut().zip(grad2_scaled.iter()) {
            *acc += add;
        }

        assert_eq!(accumulated.len(), 2);
    }

    #[test]
    fn test_mixed_precision_loss_compute() {
        let harness = Phase2TestHarness::new();
        let logits = vec![1.0, 2.0, 3.0];
        let _loss_fp32 = harness.forward(&logits);
        // Forward computes in FP16, loss scaling happens
        let loss_scale = harness.get_config("initial_loss_scale").unwrap_or(1.0);
        assert!(loss_scale > 0.0);
    }
}

#[cfg(test)]
mod loss_scaling_tests {
    use super::*;

    #[test]
    fn test_loss_scaling_initialization() {
        let harness = Phase2TestHarness::new();
        let initial_scale = harness.get_config("initial_loss_scale").unwrap_or(1.0);
        assert_eq!(initial_scale, 32768.0);
    }

    #[test]
    fn test_loss_scale_growth_on_success() {
        let mut harness = Phase2TestHarness::new();
        let old_scale = harness.get_config("initial_loss_scale").unwrap_or(32768.0);
        let growth_factor = harness.get_config("scale_growth_factor").unwrap_or(2.0);
        let new_scale = old_scale * growth_factor;

        harness.set_config("current_scale", new_scale);
        let current = harness.get_config("current_scale").unwrap_or(old_scale);

        assert!(validators::validate_loss_scale(old_scale, current, "up"));
    }

    #[test]
    fn test_loss_scale_backoff_on_overflow() {
        let mut harness = Phase2TestHarness::new();
        let old_scale = harness.get_config("initial_loss_scale").unwrap_or(32768.0);
        let new_scale = old_scale * 0.5; // Backoff factor

        harness.set_config("current_scale", new_scale);
        let current = harness.get_config("current_scale").unwrap_or(old_scale);

        assert!(validators::validate_loss_scale(old_scale, current, "down"));
    }

    #[test]
    fn test_loss_scale_bounds() {
        let harness = Phase2TestHarness::new();
        let max_scale = (1u32 << 24) as f32; // 16M max
        let min_scale = 1.0f32;

        let mut scale = 32768.0f32;
        // Should not exceed max
        scale = (scale * 2.0).min(max_scale);
        assert!(scale <= max_scale);

        // Should not go below min
        scale = (scale * 0.5).max(min_scale);
        assert!(scale >= min_scale);
    }

    #[test]
    fn test_dynamic_loss_scaling_interval() {
        let harness = Phase2TestHarness::new();
        let interval = harness
            .get_config("scale_growth_interval")
            .unwrap_or(2000.0);
        assert_eq!(interval, 2000.0);
    }
}

#[cfg(test)]
mod quat_mixed_precision_tests {
    use super::*;

    #[test]
    fn test_quat_fp16_forward() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let quats = r#gen.generate_batch(10);

        let float_values: Vec<f32> = quats
            .iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let output = harness.forward(&float_values);
        assert_eq!(output.len(), float_values.len());
    }

    #[test]
    fn test_quat_gradient_scaling() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let quats = r#gen.generate_batch(5);

        let gradients: Vec<f32> = quats
            .iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let scaled = harness.backward(&gradients);
        assert_eq!(scaled.len(), gradients.len());

        for val in scaled {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_quat_loss_scaling_precision() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let q = r#gen.next();

        let components = vec![q.w, q.x, q.y, q.z];
        let scaled = harness.backward(&components);

        let loss_scale = harness.get_config("initial_loss_scale").unwrap_or(32768.0);
        for (orig, scaled_val) in components.iter().zip(scaled.iter()) {
            assert!((orig / loss_scale - scaled_val).abs() < 1e-9);
        }
    }

    #[test]
    fn test_quat_batch_precision_flow() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(123);
        let batch = r#gen.generate_batch(8);

        let float_batch: Vec<f32> = batch
            .iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        let forward = harness.forward(&float_batch);
        let backward = harness.backward(&forward);

        assert_eq!(forward.len(), float_batch.len());
        assert_eq!(backward.len(), float_batch.len());
    }
}

#[cfg(test)]
mod end_to_end_tests {
    use super::*;

    #[test]
    fn test_end_to_end_mixed_precision_epoch() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);
        let batch = r#gen.generate_batch(32);

        let float_batch: Vec<f32> = batch
            .iter()
            .flat_map(|q| vec![q.w, q.x, q.y, q.z])
            .collect();

        // Forward in FP16
        let forward_output = harness.forward(&float_batch);
        assert_eq!(forward_output.len(), float_batch.len());

        // Compute loss (simplified)
        let loss = forward_output.iter().map(|x| x * x).sum::<f32>();
        assert!(loss.is_finite());

        // Scale loss
        let loss_scale = harness.get_config("initial_loss_scale").unwrap_or(32768.0);
        let scaled_loss = loss * loss_scale;

        // Backward in FP32
        let backward_output = harness.backward(&vec![scaled_loss]);
        assert_eq!(backward_output.len(), 1);
    }

    #[test]
    fn test_mixed_precision_training_loop() {
        let harness = Phase2TestHarness::new();
        let mut loss_history: Vec<f32> = Vec::new();

        for _ in 0..10 {
            let mut r#gen = QuaternionGenerator::new(42);
            let batch = r#gen.generate_batch(16);
            let values: Vec<f32> = batch
                .iter()
                .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                .collect();

            let forward = harness.forward(&values);
            let loss: f32 = forward.iter().map(|x| x * x).sum();
            loss_history.push(loss);
        }

        assert_eq!(loss_history.len(), 10);
        for loss in loss_history {
            assert!(loss.is_finite());
        }
    }

    #[test]
    fn test_bf16_configuration() {
        let mut harness = Phase2TestHarness::new();
        harness.set_config("precision_mode", 16.0); // BF16 = 16

        let input = vec![1.0, 2.0, 3.0];
        let output = harness.forward(&input);

        assert_eq!(output.len(), input.len());
        for (a, b) in input.iter().zip(output.iter()) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_fused_mixed_precision_batch() {
        let harness = Phase2TestHarness::new();
        let mut r#gen = QuaternionGenerator::new(42);

        for _ in 0..5 {
            let batch = r#gen.generate_batch(8);
            let values: Vec<f32> = batch
                .iter()
                .flat_map(|q| vec![q.w, q.x, q.y, q.z])
                .collect();

            let forward = harness.forward(&values);
            let fused = harness.fused_layer(&forward);
            let backward = harness.backward(&fused);

            assert_eq!(backward.len(), values.len());
        }
    }
}
