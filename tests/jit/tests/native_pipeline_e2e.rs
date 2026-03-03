// tests/jit/tests/native_pipeline_e2e.rs
//
// Horizon 4.3 — Source-to-ELF pipeline tests.
//
// Compiles .sio source files through `souc compile --native-stub` and
// runs the resulting ELF binaries, verifying stdout and exit codes.
//
// Requires the `souc` binary to be built and available in the repo's
// target/debug directory (or as /tmp/souc-patched).

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static FILE_COUNTER: AtomicU64 = AtomicU64::new(0);

fn find_souc() -> PathBuf {
    // Try repo-local target/debug/souc first
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let local = repo_root.join("target/debug/souc");
    if local.exists() {
        return local;
    }
    // Fallback: /tmp/souc-patched (used during development)
    let fallback = PathBuf::from("/tmp/souc-patched");
    if fallback.exists() {
        return fallback;
    }
    panic!("Cannot find souc binary. Build with `cargo build --bin souc` or place at /tmp/souc-patched");
}

fn unique_path(prefix: &str) -> PathBuf {
    let id = FILE_COUNTER.fetch_add(1, Ordering::Relaxed);
    PathBuf::from(format!("/tmp/sounio_test_{}_{}", prefix, id))
}

fn compile_and_run(sio_source: &str, test_name: &str) -> (String, i32) {
    let souc = find_souc();
    let src_path = unique_path(&format!("{}_src", test_name));
    let elf_path = unique_path(&format!("{}_elf", test_name));

    // Write source to temp file
    std::fs::write(&src_path, sio_source).expect("write source file");

    // Compile
    let compile_out = Command::new(&souc)
        .args(["compile", "--native-stub", "-o"])
        .arg(&elf_path)
        .arg(&src_path)
        .output()
        .expect("run souc compile");

    if !compile_out.status.success() {
        let stderr = String::from_utf8_lossy(&compile_out.stderr);
        let stdout = String::from_utf8_lossy(&compile_out.stdout);
        panic!(
            "souc compile failed for {}:\nstdout: {}\nstderr: {}",
            test_name, stdout, stderr
        );
    }

    // Run the ELF
    let run_out = Command::new(&elf_path)
        .output()
        .expect("run compiled ELF");

    // Cleanup
    let _ = std::fs::remove_file(&src_path);
    let _ = std::fs::remove_file(&elf_path);

    let stdout = String::from_utf8_lossy(&run_out.stdout).to_string();
    let exit_code = run_out.status.code().unwrap_or(-1);

    (stdout, exit_code)
}

// ============================================================================
// A1: Hello world
// ============================================================================

#[test]
fn test_hello_world() {
    let src = r#"
fn main() -> i64 with IO {
    print("Hello from self-hosted Sounio!\n")
    let x = 21 + 21
    print_int(x)
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "hello");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "Hello from self-hosted Sounio!\n42\n", "stdout mismatch");
}

// ============================================================================
// A2: Recursive fibonacci
// ============================================================================

#[test]
fn test_fibonacci() {
    let src = r#"
fn fib(n: i64) -> i64 {
    if n <= 1 { n } else { fib(n - 1) + fib(n - 2) }
}
fn main() -> i64 with IO {
    print_int(fib(10))
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "fib");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "55\n", "fib(10) should be 55");
}

// ============================================================================
// A3: While loop + mutable variables
// ============================================================================

#[test]
fn test_while_loop() {
    let src = r#"
fn main() -> i64 with IO, Mut {
    var i: i64 = 0
    while i < 5 {
        print_int(i)
        print_char(32)
        i = i + 1
    }
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "while");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "0 1 2 3 4 \n", "while loop output mismatch");
}

// ============================================================================
// A5: Exit code propagation
// ============================================================================

#[test]
fn test_exit_code() {
    let src = r#"
fn main() -> i64 with IO {
    42
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "exit_code");
    assert_eq!(exit_code, 42, "exit code should be 42");
    assert_eq!(stdout, "", "should produce no output");
}

// ============================================================================
// A6: Nested if/else
// ============================================================================

#[test]
fn test_nested_if_else() {
    let src = r#"
fn classify(n: i64) -> i64 {
    if n < 0 { 0 - 1 } else { if n == 0 { 0 } else { 1 } }
}
fn main() -> i64 with IO {
    print_int(classify(0 - 5))
    print_char(32)
    print_int(classify(0))
    print_char(32)
    print_int(classify(7))
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "nested_if");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "-1 0 1\n", "classify output mismatch");
}

// ============================================================================
// A7: Multiple functions + arithmetic
// ============================================================================

#[test]
fn test_multi_function() {
    let src = r#"
fn square(x: i64) -> i64 { x * x }
fn add(a: i64, b: i64) -> i64 { a + b }
fn main() -> i64 with IO {
    let a = square(3)
    let b = square(4)
    let c = add(a, b)
    print_int(c)
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "multi_fn");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "25\n", "3*3 + 4*4 = 25");
}

// ============================================================================
// A8: Negative numbers
// ============================================================================

#[test]
fn test_negative_numbers() {
    let src = r#"
fn main() -> i64 with IO {
    let x = 0 - 42
    print_int(x)
    print_char(10)
    0
}
"#;
    let (stdout, exit_code) = compile_and_run(src, "negative");
    assert_eq!(exit_code, 0, "exit code should be 0");
    assert_eq!(stdout, "-42\n", "negative number output");
}

// ============================================================================
// A9: Float arithmetic (SSE2)
// ============================================================================

#[test]
fn test_float_arithmetic() {
    let src = r#"
fn main() -> i64 with IO {
    let a = 3.0 + 2.0
    let b = 10.0 - a
    if b == a { 0 } else { 1 }
}
"#;
    let (_, exit_code) = compile_and_run(src, "float");
    assert_eq!(exit_code, 0, "3.0 + 2.0 = 5.0 = 10.0 - 5.0");
}

// ============================================================================
// A10: Float multiply + divide
// ============================================================================

#[test]
fn test_float_mul_div() {
    let src = r#"
fn main() -> i64 with IO {
    let product = 3.0 * 4.0
    let quotient = product / 2.0
    if quotient == 6.0 { 0 } else { 1 }
}
"#;
    let (_, exit_code) = compile_and_run(src, "float_muldiv");
    assert_eq!(exit_code, 0, "3.0 * 4.0 / 2.0 = 6.0");
}
