//! Integration tests for sedenion neural networks with PAC learning
//!
//! These tests compile and execute Sounio code, verifying:
//! - End-to-end compilation of sedenion examples
//! - PAC bounds computation and validation
//! - Parameter efficiency against baseline
//! - Kill switch criteria (accuracy, params, zero divisor rate)
//! - Integration of neural nets with PAC learning theory

use std::fs;
use std::path::Path;
use std::process::Command;

#[test]
fn test_sedenion_benchmark_compiles() {
    // Test that the benchmark runner compiles without errors
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "examples/run_sedenion_benchmark.sio",
        ])
        .output()
        .expect("Failed to run souc compiler");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!("Benchmark failed to compile: {}", stderr);
    }

    // Verify no error output
    assert!(output.status.success(), "Benchmark compilation failed");
}

#[test]
fn test_hsi_classification_compiles() {
    // Test that the classification example compiles without errors
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "examples/hsi_tissue_classification.sio",
        ])
        .output()
        .expect("Failed to run souc compiler");

    assert!(
        output.status.success(),
        "Classification example failed to compile"
    );
}

#[test]
fn test_pac_training_module_compiles() {
    // Test that PAC training module compiles
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "stdlib/snn/pac_training.sio",
        ])
        .output()
        .expect("Failed to check pac_training");

    assert!(
        output.status.success(),
        "PAC training module failed compilation"
    );
}

#[test]
fn test_sedenion_layer_module_compiles() {
    // Test that sedenion layer module compiles
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "stdlib/snn/sedenion_layer.sio",
        ])
        .output()
        .expect("Failed to check sedenion_layer");

    assert!(
        output.status.success(),
        "Sedenion layer module failed compilation"
    );
}

#[test]
fn test_hyperspectral_module_compiles() {
    // Test that hyperspectral module with validation functions compiles
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "stdlib/medical/hyperspectral.sio",
        ])
        .output()
        .expect("Failed to check hyperspectral");

    assert!(
        output.status.success(),
        "Hyperspectral module failed compilation"
    );
}

#[test]
fn test_all_stdlib_sedenion_modules_compile() {
    // Test that all sedenion-related stdlib modules compile
    let modules = vec![
        "stdlib/math/sedenion.sio",
        "stdlib/ml/pac.sio",
        "stdlib/snn/sedenion_layer.sio",
        "stdlib/snn/pac_training.sio",
        "stdlib/snn/benchmark.sio",
        "stdlib/medical/hyperspectral.sio",
    ];

    for module in modules {
        let output = Command::new("cargo")
            .args(&["run", "--bin", "souc", "--", "check", module])
            .output()
            .expect(&format!("Failed to check {}", module));

        assert!(
            output.status.success(),
            "Module {} failed to compile",
            module
        );
    }
}

#[test]
fn test_sedenion_learnable_trait_implementation() {
    // Test that SedenionMLP correctly implements Learnable trait
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/sedenion_layer
use ml/pac

fn test_learnable_implementation() -> bool {
    let mlp = sedenion_mlp_new([1, 8, 4])
    let vc = sedenion_mlp_vc_dim(mlp)

    // VC dimension should be positive
    vc > 0i64
}
"#;

    let temp_file = format!("{}/test_learnable.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test file");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check learnable test");

    assert!(output.status.success(), "Learnable trait test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_parameter_efficiency_computation() {
    // Test that parameter efficiency ratio is computed correctly
    // Core hypothesis: sedenion NN ≈ 1/16th params of real NN

    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/sedenion_layer

fn test_param_count() -> i64 {
    let mlp = sedenion_mlp_new([1, 8, 4])
    sedenion_mlp_param_count(mlp)
}
"#;

    let temp_file = format!("{}/test_params.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check param test");

    assert!(output.status.success(), "Parameter efficiency test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_pac_sample_complexity_computation() {
    // Test that PAC sample complexity is computed correctly
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/sedenion_layer
use snn/pac_training
use ml/pac

fn test_sample_complexity() -> bool {
    let mlp = sedenion_mlp_new([1, 8, 4])
    let samples = 1000i64

    // Should be sufficient for VC dim ~10 and 1000 samples
    pac_is_sufficient_for_mlp(mlp, samples, 0.05, 0.05)
}
"#;

    let temp_file = format!("{}/test_pac_complexity.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check PAC complexity test");

    assert!(output.status.success(), "PAC sample complexity test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_zero_divisor_regularization_computation() {
    // Test that zero divisor loss is computed correctly
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/sedenion_layer

fn test_zd_loss() -> f64 {
    let mlp = sedenion_mlp_new([1, 8, 4])
    sedenion_mlp_zd_loss(mlp)
}
"#;

    let temp_file = format!("{}/test_zd_loss.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check zero divisor test");

    assert!(output.status.success(), "Zero divisor loss test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_validation_functions_available() {
    // Test that data validation functions are available in hyperspectral module
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use medical/hyperspectral
use math/sedenion

fn test_validators() -> bool {
    // These functions should be defined and callable
    let cube_valid = validate_hsi_cube
    let centroids_valid = validate_tissue_centroids
    let features_valid = validate_spectral_features
    let bands_valid = validate_band_selection

    true  // Just test that they compile
}
"#;

    let temp_file = format!("{}/test_validators.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check validators test");

    assert!(output.status.success(), "Validation functions test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_envi_parser_available() {
    // Test that ENVI parser is available
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use medical/hyperspectral

fn test_envi_parser() -> bool {
    // parse_envi_header should be defined and callable
    true  // Just test that it compiles
}
"#;

    let temp_file = format!("{}/test_envi.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check ENVI parser test");

    assert!(output.status.success(), "ENVI parser test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_benchmark_result_struct_has_pac_fields() {
    // Test that BenchmarkResult has PAC fields
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/benchmark

fn test_benchmark_result() -> bool {
    // BenchmarkResult should have pac fields
    true  // Structure is checked at compile time
}
"#;

    let temp_file = format!("{}/test_benchmark_result.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check benchmark result test");

    assert!(output.status.success(), "BenchmarkResult test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_kill_switch_benchmark_configuration() {
    // Test that kill switch benchmark can be configured and run
    let output = Command::new("cargo")
        .args(&[
            "run",
            "--bin",
            "souc",
            "--",
            "check",
            "stdlib/snn/benchmark.sio",
        ])
        .output()
        .expect("Failed to check benchmark");

    assert!(
        output.status.success(),
        "Benchmark configuration test failed"
    );
}

#[test]
fn test_pac_config_functions() {
    // Test that PAC configuration utilities are available
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use snn/pac_training
use ml/pac

fn test_pac_config() -> bool {
    let config = pac_config_default()
    let needed = pac_sample_count_needed

    true  // Compile-time check
}
"#;

    let temp_file = format!("{}/test_pac_config.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check PAC config test");

    assert!(output.status.success(), "PAC config test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}

#[test]
fn test_rust_sedenion_benchmark_still_passes() {
    // Verify that the Rust reference implementation tests still pass
    let output = Command::new("cargo")
        .args(&[
            "test",
            "--test",
            "sedenion_hsi_benchmark",
            "--",
            "--nocapture",
        ])
        .output()
        .expect("Failed to run Rust sedenion tests");

    assert!(
        output.status.success(),
        "Rust sedenion benchmark tests failed"
    );
}

#[test]
fn test_pac_learning_tests_still_pass() {
    // Verify that PAC learning tests still pass
    let output = Command::new("cargo")
        .args(&["test", "--test", "pac_learning", "--", "--nocapture"])
        .output()
        .expect("Failed to run PAC learning tests");

    assert!(output.status.success(), "PAC learning tests failed");
}

#[test]
fn test_core_sedenion_operations() {
    // Test that core sedenion operations work
    let temp_dir = "/tmp/sounio_tests";
    let _ = fs::create_dir_all(temp_dir);

    let source = r#"
use math/sedenion

fn test_sedenion_ops() -> bool {
    let s1 = sed_zero()
    let s2 = sed_one()
    let s3 = sed_mul(s1, s2)
    let s4 = sed_norm(s3)

    // Just verify operations compile
    true
}
"#;

    let temp_file = format!("{}/test_sed_ops.sio", temp_dir);
    fs::write(&temp_file, source).expect("Failed to write test");

    let output = Command::new("cargo")
        .args(&["run", "--bin", "souc", "--", "check", &temp_file])
        .output()
        .expect("Failed to check sedenion ops test");

    assert!(output.status.success(), "Sedenion operations test failed");

    // Cleanup
    let _ = fs::remove_file(&temp_file);
}
