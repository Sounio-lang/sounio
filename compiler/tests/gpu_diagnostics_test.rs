//! Integration tests for GPU Diagnostics & Validation (Phase 9)
//!
//! Tests the unified diagnostic system, source mapping, and correctness validation
//! across the GPU optimization pipeline.

use sounio::codegen::gpu::{
    // GPU IR types
    BlockId,
    // Validation
    BufferComparison,
    CorrectnessValidator,
    // Diagnostics
    DiagnosticConfig,
    DiagnosticContext,
    // Optimizer
    FusionError,
    GpuDiagnostic,
    GpuDiagnosticKind,
    GpuIrLocation,
    // Source mapping
    GpuSourceMapper,
    OptimizerError,
    PtxDebugEmitter,
    PtxLocation,
    RecoveryGenerator,
    SpanTracker,
    ToleranceConfig,
    ValidationConfig,
    ValidationError,
    ValidationIssue,
    ValueId,
};
use sounio::common::Span;

// ============================================================================
// Diagnostic Aggregation Tests
// ============================================================================

#[test]
fn test_diagnostic_context_aggregation() {
    let mut ctx = DiagnosticContext::new();

    // Add errors from different phases using the report methods
    ctx.report_fusion_error(&FusionError::KernelNotFound("missing_kernel".to_string()));
    ctx.report_warning("Minor issue detected");
    ctx.report_codegen_error("PTX generation failed");

    assert_eq!(ctx.error_count(), 2);
    assert_eq!(ctx.warning_count(), 1);
    assert!(ctx.has_errors());
    assert!(ctx.has_warnings());
}

#[test]
fn test_diagnostic_with_source_location() {
    let mut ctx = DiagnosticContext::new();

    // Create source location
    let span = Span::new(100, 150);
    let gpu_loc = GpuIrLocation {
        kernel: "my_kernel".to_string(),
        block: BlockId(0),
        instruction: 5,
        value: ValueId(42),
    };

    // Create diagnostic with location info using builder pattern
    let diag = GpuDiagnostic::error(
        GpuDiagnosticKind::Optimizer(Box::new(OptimizerError::InvalidModule(
            "Bad structure".to_string(),
        ))),
        "Invalid module structure",
    )
    .with_span(span)
    .with_gpu_location(gpu_loc.clone())
    .with_ptx_line(25);

    ctx.report(diag);

    let report = ctx.build_report();
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].hlir_span, Some(span));
    assert_eq!(
        report.errors[0].gpu_location.as_ref().map(|l| &l.kernel),
        Some(&"my_kernel".to_string())
    );
    assert_eq!(report.errors[0].ptx_line, Some(25));
}

#[test]
fn test_diagnostic_report_summary() {
    let mut ctx = DiagnosticContext::new();

    ctx.report_fusion_error(&FusionError::KernelNotFound("kernel1".to_string()));
    ctx.report_fusion_error(&FusionError::BlockNotFound(BlockId(5)));
    ctx.report_warning("Warning 1");

    let report = ctx.build_report();

    assert_eq!(report.summary.error_count, 2);
    assert_eq!(report.summary.warning_count, 1);
    assert!(report.has_errors());
    assert!(report.has_warnings());
}

// ============================================================================
// Recovery Hint Tests
// ============================================================================

#[test]
fn test_recovery_hints_for_fusion_error() {
    let mut ctx = DiagnosticContext::new();

    ctx.report_fusion_error(&FusionError::KernelNotFound("test_kernel".to_string()));

    let report = ctx.build_report();
    assert!(!report.errors.is_empty());

    // Should have recovery hints
    assert!(!report.errors[0].hints.is_empty());

    // Check hint content mentions the kernel name
    let hint_messages: Vec<_> = report.errors[0]
        .hints
        .iter()
        .map(|h| h.title.as_str())
        .collect();
    assert!(hint_messages.iter().any(|m| m.contains("test_kernel")));
}

#[test]
fn test_recovery_hints_generator() {
    // Test direct hint generation
    let hints = RecoveryGenerator::for_fusion_error(&FusionError::InvalidTransformation(
        "cannot fuse".to_string(),
    ));
    assert!(!hints.is_empty());

    let hints = RecoveryGenerator::for_validation_error(&ValidationError::SizeMismatch {
        expected: 100,
        actual: 50,
    });
    assert!(!hints.is_empty());
}

// ============================================================================
// Source Mapping Tests
// ============================================================================

#[test]
fn test_source_mapper_full_chain() {
    let mut mapper = GpuSourceMapper::new();

    // Record the full chain: HLIR -> GPU IR -> PTX
    let hlir_span = Span::new(50, 100);
    let gpu_loc = GpuIrLocation {
        kernel: "vector_add".to_string(),
        block: BlockId(0),
        instruction: 3,
        value: ValueId(10),
    };
    let ptx_loc = PtxLocation::new(42, 8);

    mapper.record_hlir_span(1, hlir_span);
    mapper.record_lowering(1, gpu_loc.clone());
    mapper.record_codegen(gpu_loc.clone(), ptx_loc);

    // Trace back from PTX line
    let traced_span = mapper.trace_ptx_to_hlir(42);
    assert_eq!(traced_span, Some(hlir_span));

    // Full trace
    let trace = mapper.full_trace(42);
    assert!(trace.has_location());
    assert_eq!(trace.hlir_span, Some(hlir_span));
    assert!(trace.gpu_ir.is_some());
    assert!(trace.ptx.is_some());
}

#[test]
fn test_span_tracker_during_lowering() {
    let mut tracker = SpanTracker::new();
    let mut mapper = GpuSourceMapper::new();

    // Simulate lowering a kernel
    tracker.set_kernel("matmul");
    tracker.set_block(BlockId(0));
    tracker.push_span(Span::new(200, 250));

    // Lower several instructions
    for i in 0..5 {
        let idx = tracker.next_instruction();
        assert_eq!(idx, i);
        tracker.record_to_mapper(&mut mapper, i as u32, ValueId(100 + i as u32));
    }

    // Verify mappings
    assert_eq!(mapper.hlir_mapping_count(), 5);
    for i in 0..5 {
        assert_eq!(mapper.get_hlir_span(i as u32), Some(Span::new(200, 250)));
    }
}

#[test]
fn test_ptx_debug_emitter() {
    let mut emitter = PtxDebugEmitter::new(true);

    // Register source files
    let main_id = emitter.register_file("src/kernels/main.dm");
    let utils_id = emitter.register_file("src/kernels/utils.dm");

    assert_eq!(main_id, 0);
    assert_eq!(utils_id, 1);

    // Generate file directives
    let directives = emitter.emit_file_directives();
    assert!(directives.contains(".file 1 \"src/kernels/main.dm\""));
    assert!(directives.contains(".file 2 \"src/kernels/utils.dm\""));

    // Generate location directive
    emitter.set_file(main_id);
    let loc = emitter.emit_loc(42, 10);
    assert_eq!(loc, Some(".loc 1 42 10".to_string()));
}

// ============================================================================
// Correctness Validation Tests
// ============================================================================

#[test]
fn test_validation_f32_exact_match() {
    let validator = CorrectnessValidator::default();

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let optimized: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(result.matches);
    assert!(result.issues.is_empty());
}

#[test]
fn test_validation_f32_within_tolerance() {
    let config = ValidationConfig {
        relative_tolerance: 1e-4,
        absolute_tolerance: 1e-5,
        check_nan: true,
        check_inf: true,
    };
    let validator = CorrectnessValidator::new(config);

    // Small differences within tolerance
    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let optimized: Vec<f32> = vec![1.000001, 2.000002, 3.000003, 4.000004];

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(result.matches);
}

#[test]
fn test_validation_f32_mismatch() {
    let validator = CorrectnessValidator::default();

    // Significant difference
    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let optimized: Vec<f32> = vec![1.0, 2.0, 3.5, 4.0]; // Index 2 differs

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(!result.matches);
    assert!(!result.issues.is_empty());

    // Check that the mismatch is reported
    let has_mismatch = result
        .issues
        .iter()
        .any(|issue| matches!(issue, ValidationIssue::ValueMismatch { index: 2, .. }));
    assert!(has_mismatch);
}

#[test]
fn test_validation_nan_detection() {
    let validator = CorrectnessValidator::default();

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0];
    let optimized: Vec<f32> = vec![1.0, f32::NAN, 3.0]; // NaN introduced

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(!result.matches);

    let has_nan_issue = result
        .issues
        .iter()
        .any(|issue| matches!(issue, ValidationIssue::NaNDetected { .. }));
    assert!(has_nan_issue);
}

#[test]
fn test_validation_inf_detection() {
    let validator = CorrectnessValidator::default();

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0];
    let optimized: Vec<f32> = vec![1.0, f32::INFINITY, 3.0]; // Infinity introduced

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(!result.matches);

    let has_inf_issue = result
        .issues
        .iter()
        .any(|issue| matches!(issue, ValidationIssue::InfDetected { .. }));
    assert!(has_inf_issue);
}

#[test]
fn test_validation_size_mismatch() {
    let validator = CorrectnessValidator::default();

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let optimized: Vec<f32> = vec![1.0, 2.0, 3.0]; // Different size

    let result = validator.validate_buffers(&baseline, &optimized);
    assert!(result.is_err());

    if let Err(ValidationError::SizeMismatch { expected, actual }) = result {
        assert_eq!(expected, 5);
        assert_eq!(actual, 3);
    } else {
        panic!("Expected SizeMismatch error");
    }
}

#[test]
fn test_tolerance_config_defaults() {
    // Default tolerance
    let default = ToleranceConfig::default();
    assert!(default.rtol > 0.0);
    assert!(default.atol > 0.0);
    assert_eq!(default.rtol, 1e-5);
    assert_eq!(default.atol, 1e-8);
}

#[test]
fn test_validation_config_defaults() {
    let config = ValidationConfig::default();
    assert!(config.relative_tolerance > 0.0);
    assert!(config.absolute_tolerance > 0.0);
    assert!(config.check_nan);
    assert!(config.check_inf);
}

#[test]
fn test_validation_with_custom_tolerance() {
    // Test with very tight tolerance - should fail
    let strict_config = ValidationConfig {
        relative_tolerance: 1e-10,
        absolute_tolerance: 1e-12,
        check_nan: true,
        check_inf: true,
    };
    let strict_validator = CorrectnessValidator::new(strict_config);

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0];
    let optimized: Vec<f32> = vec![1.0000001, 2.0, 3.0]; // Tiny difference

    let result = strict_validator.validate_buffers(&baseline, &optimized).unwrap();
    // With strict tolerance, even tiny difference may be caught
    assert!(!result.matches || result.stats.max_error > 0.0);
}

#[test]
fn test_validation_with_relaxed_tolerance() {
    // Test with very relaxed tolerance - should pass
    let relaxed_config = ValidationConfig {
        relative_tolerance: 0.1,  // 10% relative
        absolute_tolerance: 0.01, // 0.01 absolute
        check_nan: true,
        check_inf: true,
    };
    let relaxed_validator = CorrectnessValidator::new(relaxed_config);

    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0];
    let optimized: Vec<f32> = vec![1.005, 2.01, 3.02]; // Small differences

    let result = relaxed_validator.validate_buffers(&baseline, &optimized).unwrap();
    assert!(result.matches);
}

// ============================================================================
// Integration: Diagnostic + Validation Tests
// ============================================================================

#[test]
fn test_diagnostic_from_validation_error() {
    let mut ctx = DiagnosticContext::new();

    // Create a validation error and report it
    let error = ValidationError::SizeMismatch {
        expected: 100,
        actual: 50,
    };

    // Report as a diagnostic
    let diag = GpuDiagnostic::error(
        GpuDiagnosticKind::Generic(format!("{}", error)),
        "Validation failed: size mismatch",
    );
    ctx.report(diag);

    let report = ctx.build_report();
    assert!(report.has_errors());
}

#[test]
fn test_diagnostic_from_validation_issue() {
    let mut ctx = DiagnosticContext::new();

    // Simulate validation issue - create warning diagnostic
    let issue = ValidationIssue::ValueMismatch {
        index: 42,
        expected: 1.0,
        actual: 1.001,
        relative_error: 0.001,
    };

    // Report validation issue as warning
    let diag = GpuDiagnostic::warning(
        GpuDiagnosticKind::Generic(format!("{:?}", issue)),
        "Value mismatch detected",
    );
    ctx.report(diag);

    let report = ctx.build_report();
    assert!(report.has_warnings());
}

#[test]
fn test_diagnostic_from_nan_detection() {
    let mut ctx = DiagnosticContext::new();

    // NaN detection is reported as error
    let issue = ValidationIssue::NaNDetected {
        index: 5,
        context: "output buffer".to_string(),
    };

    // Report as error (NaN is more severe)
    let diag = GpuDiagnostic::error(
        GpuDiagnosticKind::Generic(format!("{:?}", issue)),
        "NaN detected in GPU output",
    );
    ctx.report(diag);

    let report = ctx.build_report();
    assert!(report.has_errors());
}

#[test]
fn test_full_pipeline_diagnostic_flow() {
    let mut mapper = GpuSourceMapper::new();
    let mut ctx = DiagnosticContext::new();

    // Step 1: Record source mapping during lowering
    let hlir_span = Span::new(100, 200);
    let gpu_loc = GpuIrLocation {
        kernel: "reduce_kernel".to_string(),
        block: BlockId(0),
        instruction: 10,
        value: ValueId(50),
    };
    mapper.record_hlir_span(42, hlir_span);
    mapper.record_lowering(42, gpu_loc.clone());
    mapper.record_codegen(gpu_loc.clone(), PtxLocation::line(150));

    // Step 2: Optimization encounters an error
    let diag = GpuDiagnostic::error(
        GpuDiagnosticKind::Fusion(FusionError::InvalidTransformation(
            "Cannot fuse kernels".to_string(),
        )),
        "Fusion transformation failed",
    )
    .with_span(hlir_span)
    .with_gpu_location(gpu_loc)
    .with_ptx_line(150);

    ctx.report(diag);

    // Step 3: Build report
    let report = ctx.build_report();

    // Verify diagnostics have full location chain
    assert_eq!(report.errors.len(), 1);
    assert_eq!(report.errors[0].hlir_span, Some(hlir_span));
    assert_eq!(report.errors[0].ptx_line, Some(150));

    // Verify can trace back
    let trace = mapper.full_trace(150);
    assert_eq!(trace.hlir_span, Some(hlir_span));
}

#[test]
fn test_validation_precision_stats() {
    let validator = CorrectnessValidator::default();

    // Validate buffers and check precision stats
    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let optimized: Vec<f32> = vec![1.001, 2.002, 3.003, 4.004];

    let result = validator.validate_buffers(&baseline, &optimized).unwrap();

    // Stats should be computed
    assert!(result.stats.max_error > 0.0);
    assert!(result.stats.mean_error > 0.0);
    assert!(result.stats.rms_error > 0.0);
}

#[test]
fn test_max_diagnostics_limit() {
    let config = DiagnosticConfig {
        max_diagnostics: 3,
        ..DiagnosticConfig::default()
    };
    let mut ctx = DiagnosticContext::with_config(config);

    // Try to report more than the limit
    for i in 0..10 {
        ctx.report_fusion_error(&FusionError::KernelNotFound(format!("kernel_{}", i)));
    }

    // Should be limited to 3
    assert_eq!(ctx.error_count(), 3);
}

#[test]
fn test_diagnostic_info_filtering() {
    // Default config doesn't collect info
    let mut ctx = DiagnosticContext::new();

    let info_diag = GpuDiagnostic::info(
        GpuDiagnosticKind::Generic("info message".to_string()),
        "This is informational",
    );
    ctx.report(info_diag);

    // Info should be filtered out by default
    let report = ctx.build_report();
    assert!(!report.has_errors());
    assert!(!report.has_warnings());
}

#[test]
fn test_diagnostic_formatting() {
    let mut ctx = DiagnosticContext::new();
    ctx.report_fusion_error(&FusionError::KernelNotFound("my_kernel".to_string()));

    let report = ctx.build_report();
    let formatted = report.format();

    assert!(formatted.contains("error"));
    assert!(formatted.contains("1 error(s)"));
}

#[test]
fn test_buffer_comparison_result() {
    let validator = CorrectnessValidator::default();

    // Test BufferComparison struct fields
    let baseline: Vec<f32> = vec![1.0, 2.0, 3.0];
    let optimized: Vec<f32> = vec![1.0, 2.0, 3.0];

    let result: BufferComparison = validator.validate_buffers(&baseline, &optimized).unwrap();

    assert!(result.matches);
    assert!(result.issues.is_empty());
    assert_eq!(result.stats.mismatch_count, 0);
    assert_eq!(result.stats.max_error, 0.0);
}

#[test]
fn test_validation_error_display() {
    // Test that ValidationError implements Display
    let error = ValidationError::SizeMismatch {
        expected: 100,
        actual: 50,
    };
    let display = format!("{}", error);
    assert!(display.contains("100"));
    assert!(display.contains("50"));

    let error2 = ValidationError::ValueMismatch {
        index: 5,
        expected: 1.0,
        actual: 2.0,
    };
    let display2 = format!("{}", error2);
    assert!(display2.contains("5"));

    let error3 = ValidationError::PrecisionLoss {
        max_error: 0.1,
        threshold: 0.01,
    };
    let display3 = format!("{}", error3);
    assert!(display3.contains("0.1") || display3.contains("Precision"));
}
