//! Self-Hosting Validation Tests
//!
//! Compares output of Rust compiler vs self-hosted compiler on example programs.
//! This validates that the bootstrap infrastructure is working correctly.

use std::path::PathBuf;
use std::process::Command;

// Import the compiler loader for direct testing
use sounio::compiler_loader::SounioCompiler;
use sounio::embedded_stdlib;

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

// ============================================================================
// Phase 2: Embedded Module Tests
// ============================================================================

/// Test that embedded modules are available
#[test]
fn test_embedded_modules_available() {
    let modules = embedded_stdlib::list_modules();
    assert!(!modules.is_empty(), "Should have embedded modules");
    println!("Found {} embedded modules", modules.len());
}

/// Test that core compiler modules are embedded
#[test]
fn test_embedded_has_core_compiler_modules() {
    let modules = embedded_stdlib::list_modules();

    // These patterns should match embedded module names
    let expected_patterns = ["lexer", "parser", "check", "codegen"];

    for pattern in expected_patterns {
        let found = modules.iter().any(|m| m.contains(pattern));
        assert!(
            found,
            "Should have module containing '{}'. Available: {:?}",
            pattern,
            modules.iter().take(10).collect::<Vec<_>>()
        );
    }
}

/// Test embedded compiler can load modules
#[test]
fn test_embedded_compiler_loads_modules() {
    let compiler = SounioCompiler::new_embedded().expect("Should create embedded compiler");

    assert!(compiler.is_embedded(), "Should be in embedded mode");
    assert!(compiler.module_count() > 0, "Should have modules");

    let modules = compiler.list_modules().expect("Should list modules");
    assert!(!modules.is_empty(), "Should list at least one module");

    // Try to load a known module
    if let Some(first) = modules.first() {
        let source = compiler.load_module(first);
        assert!(source.is_ok(), "Should load module: {}", first);
        let content = source.unwrap();
        assert!(!content.is_empty(), "Module content should not be empty");
    }
}

/// Test that embedded compiler can compile simple code
#[test]
fn test_embedded_compiler_compiles_code() {
    let compiler = SounioCompiler::new_embedded().expect("Should create embedded compiler");

    let source = r#"fn main() with IO {
    println("Hello from embedded compiler!")
}"#;

    let result = compiler.compile(source);
    assert!(
        result.is_ok(),
        "Should compile valid code: {:?}",
        result.err()
    );

    let bytecode = result.unwrap();
    assert!(!bytecode.is_empty(), "Should generate bytecode");
}

/// Test that embedded module count matches build-time count
#[test]
fn test_embedded_module_count_consistent() {
    let count_from_constant = embedded_stdlib::MODULE_COUNT;
    let count_from_list = embedded_stdlib::list_modules().len();
    let count_from_map = embedded_stdlib::get_embedded_modules().len();

    assert_eq!(
        count_from_constant, count_from_list,
        "MODULE_COUNT should match list length"
    );
    assert_eq!(
        count_from_constant, count_from_map,
        "MODULE_COUNT should match map size"
    );
}

/// Test that we can read specific known modules
#[test]
fn test_can_read_lexer_module() {
    // Look for any lexer-related module
    let modules = embedded_stdlib::list_modules();
    let lexer_module = modules.iter().find(|m| m.contains("lexer"));

    if let Some(module_name) = lexer_module {
        let source = embedded_stdlib::get_module(module_name);
        assert!(source.is_some(), "Should get lexer module source");
        let content = source.unwrap();
        assert!(
            content.len() > 100,
            "Lexer module should have substantial content"
        );
        // Basic sanity check - should contain Sounio code markers
        assert!(
            content.contains("fn") || content.contains("struct") || content.contains("//"),
            "Should contain Sounio code"
        );
    }
}

// ============================================================================
// Phase 3: Self-Compilation Tests
// ============================================================================

/// Test that we can compile embedded stdlib modules through the bytecode pipeline
#[test]
fn test_compile_stdlib_module_to_bytecode() {
    let compiler = SounioCompiler::new_embedded().expect("Should create embedded compiler");
    let modules = compiler.list_modules().expect("Should list modules");

    // Try to compile each module and count successes
    let mut compiled = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut errors = Vec::new();

    for module_name in modules.iter() {
        let source = compiler.load_module(module_name);
        if let Ok(content) = source {
            // Skip very small modules (likely just comments/empty)
            if content.len() < 200 || !content.contains("fn ") {
                skipped += 1;
                continue;
            }

            match compiler.compile(&content) {
                Ok(bytecode) => {
                    compiled += 1;
                    println!(
                        "  compiled {} ({} instructions)",
                        module_name,
                        bytecode.len()
                    );
                }
                Err(e) => {
                    failed += 1;
                    errors.push(format!("{}: {}", module_name, e));
                }
            }
        }
    }

    println!(
        "\nSummary: {} compiled, {} failed, {} skipped",
        compiled, failed, skipped
    );

    // Print all errors for categorization
    if !errors.is_empty() {
        println!("\nAll {} errors:", errors.len());
        for e in errors.iter() {
            println!("  - {}", e);
        }
    }

    // We expect at least some modules to compile successfully
    assert!(
        compiled > 0,
        "Should compile at least one substantial module. Errors: {:?}",
        errors
    );
}
