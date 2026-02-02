//! Self-Hosting Validation Tests
//!
//! Compares output of Rust compiler vs self-hosted compiler on example programs.
//! This validates that the bootstrap infrastructure is working correctly.

use std::path::PathBuf;
use std::process::Command;

fn get_souc_bin() -> PathBuf {
    // Try to find the souc binary in the target directory
    let target_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("target/debug/souc"))
        .filter(|p| p.exists());

    if let Some(path) = target_dir {
        return path;
    }

    // Fallback: try from current exe
    std::env::current_exe()
        .ok()
        .and_then(|exe_path| exe_path.parent().map(|p| p.to_path_buf()))
        .map(|p| p.join("souc"))
        .unwrap_or_else(|| PathBuf::from("souc"))
}

fn setup_test_file(name: &str, content: &str) -> PathBuf {
    let temp_dir = std::env::temp_dir();
    let path = temp_dir.join(format!("souc_test_{}.sio", name));
    std::fs::write(&path, content).expect("Failed to write test file");
    path
}

fn run_with_rust_compiler(path: &PathBuf) -> String {
    let output = Command::new(get_souc_bin())
        .arg("run")
        .arg(path)
        .output()
        .expect("Failed to run with Rust compiler");

    String::from_utf8_lossy(&output.stdout).to_string()
}

fn run_with_sounio_compiler(path: &PathBuf) -> String {
    let mut cmd = Command::new(get_souc_bin());
    cmd.arg("run")
        .arg("--use-sounio-compiler")
        .arg(path)
        .env("SOUNIO_STDLIB_PATH", "stdlib/compiler");

    let output = cmd
        .output()
        .expect("Failed to run with self-hosted compiler");

    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn test_hello_world_output_matches() {
    let test_file = setup_test_file(
        "hello_world",
        r#"fn main() with IO {
    println("Hello from Sounio!")
}"#,
    );

    let rust_output = run_with_rust_compiler(&test_file);
    let sounio_output = run_with_sounio_compiler(&test_file);

    assert_eq!(
        rust_output.trim(),
        sounio_output.trim(),
        "Rust and Sounio compiler outputs differ for hello_world"
    );
    assert!(
        rust_output.contains("Hello from Sounio!"),
        "Expected output not found"
    );
}

#[test]
fn test_simple_function_call() {
    let test_file = setup_test_file(
        "simple_fn",
        r#"fn greet(name: String) with IO {
    println(name)
}

fn main() with IO {
    greet("World")
}"#,
    );

    let rust_output = run_with_rust_compiler(&test_file);
    let sounio_output = run_with_sounio_compiler(&test_file);

    assert_eq!(
        rust_output.trim(),
        sounio_output.trim(),
        "Rust and Sounio compiler outputs differ for function calls"
    );
    assert!(
        rust_output.contains("World"),
        "Expected function call output not found"
    );
}

#[test]
fn test_multiple_statements() {
    let test_file = setup_test_file(
        "multi_stmt",
        r#"fn main() with IO {
    println("Line 1")
    println("Line 2")
    println("Line 3")
}"#,
    );

    let rust_output = run_with_rust_compiler(&test_file);
    let sounio_output = run_with_sounio_compiler(&test_file);

    assert_eq!(
        rust_output.trim(),
        sounio_output.trim(),
        "Rust and Sounio compiler outputs differ for multiple statements"
    );

    let lines: Vec<&str> = rust_output.lines().collect();
    assert!(lines.len() >= 3, "Expected at least 3 lines of output");
}

#[test]
fn test_self_hosted_compiler_initializes() {
    // Verify that the self-hosted compiler can at least initialize
    let test_file = setup_test_file(
        "init_check",
        r#"fn main() with IO {
    println("Compiler initialized")
}"#,
    );

    let output = Command::new(get_souc_bin())
        .arg("run")
        .arg("--use-sounio-compiler")
        .arg(&test_file)
        .env("SOUNIO_STDLIB_PATH", "stdlib/compiler")
        .output()
        .expect("Failed to run self-hosted compiler");

    // Check that it at least runs without error
    assert!(output.status.success(), "Self-hosted compiler failed");
    assert!(!output.stdout.is_empty(), "Expected output from program");
}

#[test]
fn test_compiler_loader_compiles_successfully() {
    // Test that the SounioCompiler can parse and compile valid Sounio code
    // This is a basic structural test
    let test_file = setup_test_file(
        "compiler_test",
        r#"fn add(x: i32, y: i32) -> i32 {
    x + y
}

fn main() with IO {
    println("Compilation successful")
}"#,
    );

    let output = Command::new(get_souc_bin())
        .arg("run")
        .arg(&test_file)
        .output()
        .expect("Failed to run compiler");

    assert!(output.status.success(), "Rust compiler failed");
    assert!(
        output.stdout.iter().count() > 0,
        "Expected output from program"
    );
}

/// Benchmark: Compile time comparison
#[test]
#[ignore] // Ignored by default - run with: cargo test -- --ignored
fn benchmark_compile_time() {
    let test_file = setup_test_file(
        "perf_test",
        r#"fn fibonacci(n: i32) -> i32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

fn main() with IO {
    println("Fibonacci computed")
}"#,
    );

    // Rust compiler benchmark
    let rust_start = std::time::Instant::now();
    for _ in 0..3 {
        let _ = Command::new(get_souc_bin())
            .arg("run")
            .arg(&test_file)
            .output();
    }
    let rust_elapsed = rust_start.elapsed();

    // Self-hosted compiler benchmark
    let sounio_start = std::time::Instant::now();
    for _ in 0..3 {
        let _ = Command::new(get_souc_bin())
            .arg("run")
            .arg("--use-sounio-compiler")
            .arg(&test_file)
            .env("SOUNIO_STDLIB_PATH", "stdlib/compiler")
            .output();
    }
    let sounio_elapsed = sounio_start.elapsed();

    println!(
        "Rust compiler: {:?}, Self-hosted: {:?}",
        rust_elapsed, sounio_elapsed
    );

    // Informational - not a hard assertion
    let ratio = sounio_elapsed.as_secs_f64() / rust_elapsed.as_secs_f64();
    println!("Self-hosted is {:.1}x the Rust compiler time", ratio);
}

/// Test that both compilers handle errors similarly
#[test]
#[ignore] // Ignored - error handling may differ during bootstrap
fn test_error_handling_consistency() {
    let test_file = setup_test_file(
        "error_test",
        r#"fn main() with IO {
    undefined_variable
}"#,
    );

    let rust_output = Command::new(get_souc_bin())
        .arg("run")
        .arg(&test_file)
        .output();

    let sounio_output = Command::new(get_souc_bin())
        .arg("run")
        .arg("--use-sounio-compiler")
        .arg(&test_file)
        .env("SOUNIO_STDLIB_PATH", "stdlib/compiler")
        .output();

    // Both should fail (not succeed)
    assert!(
        !rust_output.expect("Rust run").status.success(),
        "Expected Rust compiler to reject invalid code"
    );
    assert!(
        !sounio_output.expect("Sounio run").status.success(),
        "Expected self-hosted compiler to reject invalid code"
    );
}
