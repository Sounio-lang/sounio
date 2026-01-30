//! Integration tests for GPU Quantization Pipeline (Phase 11)
//!
//! Tests INT8/INT4 quantization, calibration algorithms, and PTQ workflow.

use sounio::codegen::gpu::{
    // PTQ
    ActivationQuantConfig,
    BlockId,
    // Calibration
    CalibrationCollector,
    CalibrationMethod,
    GpuBlock,
    GpuKernel,
    // IR
    GpuModule,
    GpuOp,
    GpuTarget,
    GpuTerminator,
    GpuType,
    LayerInfo,
    PerChannelCalibrator,
    PtqConfig,
    PtqEngine,
    // PTX
    PtxCodegen,
    // Quantization types
    QuantDtype,
    QuantErrorAnalyzer,
    QuantParams,
    QuantScheme,
    ValueId,
    WeightQuantConfig,
    pack_int4,
    quantize_tensor_int8,
    unpack_int4,
};

// ============================================================================
// Quantization Parameter Tests
// ============================================================================

#[test]
fn test_symmetric_int8_quantization() {
    // Create symmetric INT8 params from min/max
    let params = QuantParams::from_minmax_symmetric(-1.0, 1.0, QuantDtype::Int8);

    assert_eq!(params.zero_point, 0);
    assert_eq!(params.dtype, QuantDtype::Int8);
    assert!(matches!(params.scheme, QuantScheme::Symmetric));

    // Test quantization roundtrip
    let value = 0.5f32;
    let quantized = params.quantize(value);
    let dequantized = params.dequantize(quantized);

    // Should be close to original
    assert!(
        (value - dequantized).abs() < 0.02,
        "Expected ~{}, got {}",
        value,
        dequantized
    );
}

#[test]
fn test_asymmetric_uint8_quantization() {
    // Asymmetric quantization for non-negative activations
    let params = QuantParams::from_minmax_asymmetric(0.0, 2.0, QuantDtype::UInt8);

    assert_eq!(params.dtype, QuantDtype::UInt8);
    assert!(matches!(params.scheme, QuantScheme::Asymmetric));

    // Test edge values
    let quantized_min = params.quantize(0.0);
    let quantized_max = params.quantize(2.0);

    assert!(quantized_min <= quantized_max);
}

#[test]
fn test_int4_pack_unpack() {
    // Test INT4 packing and unpacking
    let lo: i8 = 3;
    let hi: i8 = -2;

    let packed = pack_int4(lo, hi);
    let (unpacked_lo, unpacked_hi) = unpack_int4(packed);

    assert_eq!(unpacked_lo, lo, "Low nibble mismatch");
    assert_eq!(unpacked_hi, hi, "High nibble mismatch");
}

#[test]
fn test_int4_range() {
    // INT4 range is [-8, 7]
    for lo in -8i8..=7 {
        for hi in -8i8..=7 {
            let packed = pack_int4(lo, hi);
            let (unpacked_lo, unpacked_hi) = unpack_int4(packed);
            assert_eq!(unpacked_lo, lo);
            assert_eq!(unpacked_hi, hi);
        }
    }
}

#[test]
fn test_tensor_quantization_int8() {
    let data: Vec<f32> = vec![-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, -0.75];
    let shape = vec![2, 4];

    let quantized = quantize_tensor_int8(&data, shape.clone(), "test_tensor".to_string(), true);

    assert_eq!(quantized.shape, shape);
    assert_eq!(quantized.data.len(), data.len());
    assert_eq!(quantized.name, "test_tensor");
    assert_eq!(quantized.params.dtype, QuantDtype::Int8);
}

// ============================================================================
// Quantization Error Analysis Tests
// ============================================================================

#[test]
fn test_quant_error_analyzer() {
    let mut analyzer = QuantErrorAnalyzer::new();

    // Simulate quantization with small errors
    for i in 0..100 {
        let original = (i as f32) * 0.1;
        let quantized = ((i as f32) * 0.1 * 10.0).round() / 10.0; // Simulate rounding
        analyzer.add(original, quantized, false);
    }

    let error = analyzer.compute();

    assert!(error.mse < 0.01, "MSE too high: {}", error.mse);
    assert!(error.mae < 0.1, "MAE too high: {}", error.mae);
    assert_eq!(error.count, 100);
}

#[test]
fn test_perfect_quantization_error() {
    let mut analyzer = QuantErrorAnalyzer::new();

    // Perfect quantization (no error)
    for i in 0..100 {
        let value = i as f32;
        analyzer.add(value, value, false);
    }

    let error = analyzer.compute();

    assert_eq!(error.mse, 0.0);
    assert_eq!(error.mae, 0.0);
    assert_eq!(error.max_error, 0.0);
}

// ============================================================================
// Calibration Tests
// ============================================================================

#[test]
fn test_minmax_calibration() {
    let mut collector = CalibrationCollector::new(CalibrationMethod::MinMax);

    // Collect samples
    let samples: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
    collector.collect(&samples);

    let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);

    assert!(params.scale > 0.0, "Scale should be positive");
}

#[test]
fn test_histogram_calibration() {
    let mut collector = CalibrationCollector::new(CalibrationMethod::Histogram { num_bins: 256 });

    // Collect samples with outliers
    let mut samples: Vec<f32> = (-100..=100).map(|i| i as f32 * 0.01).collect();
    samples.push(10.0); // Outlier
    samples.push(-10.0); // Outlier

    collector.collect(&samples);

    let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);

    // Histogram calibration should handle outliers better than MinMax
    assert!(params.scale > 0.0);
}

#[test]
fn test_per_channel_calibration() {
    let mut calibrator = PerChannelCalibrator::new(
        4, // num_channels
        0, // axis
        CalibrationMethod::MinMax,
    );

    // Collect per-channel data
    for ch in 0..4 {
        let scale = (ch + 1) as f32;
        let samples: Vec<f32> = (-10..=10).map(|i| i as f32 * 0.1 * scale).collect();
        calibrator.collect_channel(ch, &samples);
    }

    let params = calibrator.compute_params(QuantDtype::Int8);

    assert_eq!(params.num_channels, 4);
}

// ============================================================================
// PTQ Workflow Tests
// ============================================================================

#[test]
fn test_ptq_config_presets() {
    let accuracy = PtqConfig::accuracy_focused();
    assert_eq!(accuracy.num_calibration_batches, 200);
    assert!(accuracy.analyze_errors);
    assert!(matches!(
        accuracy.activation_config.calibration_method,
        CalibrationMethod::Entropy { .. }
    ));

    let speed = PtqConfig::speed_focused();
    assert_eq!(speed.num_calibration_batches, 50);
    assert!(!speed.analyze_errors);
    assert!(matches!(
        speed.activation_config.calibration_method,
        CalibrationMethod::MinMax
    ));

    let int4 = PtqConfig::int4_aggressive();
    assert_eq!(int4.weight_config.dtype, QuantDtype::Int4);
}

#[test]
fn test_ptq_engine_workflow() {
    let config = PtqConfig::default();
    let mut engine = PtqEngine::new(config);

    // Register a layer
    let layer = LayerInfo {
        name: "conv1".to_string(),
        layer_type: "conv".to_string(),
        weight_shape: vec![64, 3, 3, 3],
        has_bias: true,
        input_shape: vec![1, 3, 224, 224],
        output_shape: vec![1, 64, 224, 224],
    };
    engine.register_layer(layer);

    // Simulate calibration
    let activations: Vec<f32> = (0..1000).map(|i| (i as f32 / 500.0) - 1.0).collect();

    for _ in 0..100 {
        engine.calibrate_input("conv1", &activations);
        engine.calibrate_output("conv1", &activations);
        engine.increment_calibration_batch();
    }

    assert!(engine.is_calibration_complete());
    assert_eq!(engine.calibration_progress(), 1.0);

    // Compute quantization parameters
    engine.compute_quant_params();

    // Verify params were computed
    assert!(engine.get_input_params("conv1").is_some());
    assert!(engine.get_output_params("conv1").is_some());
}

#[test]
fn test_ptq_layer_status() {
    let config = PtqConfig::default();
    let mut engine = PtqEngine::new(config);

    let layers = vec![
        LayerInfo {
            name: "layer1".to_string(),
            layer_type: "linear".to_string(),
            weight_shape: vec![128, 64],
            has_bias: true,
            input_shape: vec![1, 64],
            output_shape: vec![1, 128],
        },
        LayerInfo {
            name: "layer2".to_string(),
            layer_type: "linear".to_string(),
            weight_shape: vec![256, 128],
            has_bias: true,
            input_shape: vec![1, 128],
            output_shape: vec![1, 256],
        },
    ];

    for layer in layers {
        engine.register_layer(layer);
    }

    let status = engine.get_layer_status();
    assert_eq!(status.len(), 2);
    assert_eq!(status[0].info.name, "layer1");
    assert_eq!(status[1].info.name, "layer2");
}

// ============================================================================
// PTX Codegen Tests for INT8 Operations
// ============================================================================

#[test]
fn test_ptx_quantize_int8() {
    let target = GpuTarget::Cuda {
        compute_capability: (7, 5),
    };
    let mut module = GpuModule::new("quantize_test", target);

    let mut kernel = GpuKernel::new("quantize_kernel");

    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Add quantization operations
    block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.5, GpuType::F32));
    block.add_instruction(ValueId(1), GpuOp::ConstFloat(0.01, GpuType::F32)); // scale
    block.add_instruction(ValueId(2), GpuOp::ConstInt(0, GpuType::I32)); // zero_point
    block.add_instruction(
        ValueId(3),
        GpuOp::QuantizeF32ToInt8 {
            value: ValueId(0),
            scale: ValueId(1),
            zero_point: ValueId(2),
            symmetric: true,
        },
    );

    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);
    module.add_kernel(kernel);

    // Generate PTX
    let mut codegen = PtxCodegen::new((7, 5));
    let ptx = codegen.generate(&module);

    // Verify PTX contains INT8 quantization instructions
    assert!(
        ptx.contains("Quantize F32 to INT8"),
        "Missing quantize comment"
    );
    assert!(ptx.contains("div.rn.f32"), "Missing scale division");
}

#[test]
fn test_ptx_dp4a() {
    // dp4a requires sm_61+ (Pascal)
    let target = GpuTarget::Cuda {
        compute_capability: (6, 1),
    };
    let mut module = GpuModule::new("dp4a_test", target);

    let mut kernel = GpuKernel::new("dp4a_kernel");

    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Pack 4 INT8 values into u32
    block.add_instruction(ValueId(0), GpuOp::ConstInt(0x01020304, GpuType::U32)); // a packed
    block.add_instruction(ValueId(1), GpuOp::ConstInt(0x05060708, GpuType::U32)); // b packed
    block.add_instruction(ValueId(2), GpuOp::ConstInt(0, GpuType::I32)); // accumulator
    block.add_instruction(
        ValueId(3),
        GpuOp::Dp4a {
            a: ValueId(0),
            b: ValueId(1),
            c: ValueId(2),
        },
    );

    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);
    module.add_kernel(kernel);

    let mut codegen = PtxCodegen::new((6, 1));
    let ptx = codegen.generate(&module);

    assert!(ptx.contains("dp4a.s32.s32"), "Missing dp4a instruction");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_quantize_edge_values() {
    let params = QuantParams::from_minmax_symmetric(-1.0, 1.0, QuantDtype::Int8);

    // Test extremes
    let q_min = params.quantize(-1.0);
    let q_max = params.quantize(1.0);
    let q_zero = params.quantize(0.0);

    assert!(q_min < q_zero, "Min should be less than zero");
    assert!(q_max > q_zero, "Max should be greater than zero");
    assert!(
        q_zero == 0 || q_zero.abs() <= 1,
        "Zero should quantize near 0"
    );
}

#[test]
fn test_quantize_clipping() {
    let params = QuantParams::from_minmax_symmetric(-1.0, 1.0, QuantDtype::Int8);

    // Values outside range should clip
    let q_overflow = params.quantize(10.0);
    let q_underflow = params.quantize(-10.0);

    // Should clip to valid INT8 range
    assert!((-128..=127).contains(&q_overflow));
    assert!((-128..=127).contains(&q_underflow));
}

#[test]
fn test_empty_calibration() {
    let collector = CalibrationCollector::new(CalibrationMethod::MinMax);

    // Computing params with no data should return valid defaults
    let params = collector.compute_params(QuantDtype::Int8, QuantScheme::Symmetric);

    assert!(params.scale.is_finite());
    assert!(params.zero_point >= -128 && params.zero_point <= 127);
}

// ============================================================================
// Integration Test: Full PTQ Pipeline
// ============================================================================

#[test]
fn test_full_ptq_pipeline() {
    // Simulate a simple neural network quantization
    let config = PtqConfig {
        weight_config: WeightQuantConfig {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::PerChannel { axis: 0 },
            skip_bias: true,
            excluded_layers: vec![],
        },
        activation_config: ActivationQuantConfig {
            dtype: QuantDtype::Int8,
            scheme: QuantScheme::PerTensor,
            calibration_method: CalibrationMethod::MinMax,
            quantize_inputs: true,
            quantize_outputs: true,
        },
        num_calibration_batches: 10,
        analyze_errors: true,
        error_threshold: Some(0.01),
        first_layer_per_channel: true,
        last_layer_per_channel: true,
    };

    let mut engine = PtqEngine::new(config);

    // Register layers
    for i in 0..3 {
        engine.register_layer(LayerInfo {
            name: format!("layer{}", i),
            layer_type: "linear".to_string(),
            weight_shape: vec![64, 64],
            has_bias: true,
            input_shape: vec![1, 64],
            output_shape: vec![1, 64],
        });
    }

    // Simulate calibration pass
    for _ in 0..10 {
        for i in 0..3 {
            let activations: Vec<f32> = (0..64).map(|j| ((j as f32) - 32.0) / 32.0).collect();
            engine.calibrate_input(&format!("layer{}", i), &activations);
            engine.calibrate_output(&format!("layer{}", i), &activations);
        }
        engine.increment_calibration_batch();
    }

    // Compute quantization parameters
    engine.compute_quant_params();

    // Verify all layers have params
    for i in 0..3 {
        let name = format!("layer{}", i);
        assert!(
            engine.get_input_params(&name).is_some(),
            "Missing input params for {}",
            name
        );
        assert!(
            engine.get_output_params(&name).is_some(),
            "Missing output params for {}",
            name
        );
    }

    // Get error summary
    let summary = engine.get_error_summary();
    assert_eq!(summary.num_layers, 0); // No weights quantized yet
}
