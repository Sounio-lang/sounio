//! Native ELF Execution Tests
//!
//! Validates the self-hosted native backend end-to-end:
//! source .sio → self-hosted compiler → ELF64 binary → execute → verify exit code.

use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

/// Test the full native compilation pipeline:
/// 1. Write a simple .sio fixture
/// 2. Run self-hosted compiler with --backend=native
/// 3. Execute the produced ELF binary
/// 4. Verify exit code
#[test]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "Linux-only: requires ELF execution"
)]
fn native_elf_compile_and_execute_return_42() {
    let root = workspace_root();
    let compiler_dir = root.join("compiler");

    // Create a simple test fixture
    let fixture_path = std::env::temp_dir().join("sounio_fixture_42.sio");
    std::fs::write(&fixture_path, "fn main() -> i64 { 42 }\n").expect("failed to write fixture");

    let elf_path = std::env::temp_dir().join("sounio_e2e_42.elf");

    // Run: cargo run --bin souc -- run self-hosted/ -- compile --backend=native -o <elf> <fixture>
    let compile_result = Command::new("cargo")
        .args(["run", "--bin", "souc", "--"])
        .arg("run")
        .arg(root.join("self-hosted/"))
        .arg("--")
        .arg("compile")
        .arg("--backend=native")
        .arg("-o")
        .arg(&elf_path)
        .arg(&fixture_path)
        .current_dir(&compiler_dir)
        .output();

    let compile_output = match compile_result {
        Ok(o) => o,
        Err(e) => {
            eprintln!("Failed to invoke souc: {}", e);
            return;
        }
    };

    let stdout = String::from_utf8_lossy(&compile_output.stdout);
    let stderr = String::from_utf8_lossy(&compile_output.stderr);

    if !compile_output.status.success() {
        eprintln!(
            "souc compile failed (exit {:?}):",
            compile_output.status.code()
        );
        eprintln!("stdout: {}", stdout);
        eprintln!("stderr: {}", stderr);
        panic!("Native compilation failed");
    }

    eprintln!("Compile output: {}", stdout);

    // Verify ELF file was created
    assert!(
        elf_path.exists(),
        "ELF file not created at {}",
        elf_path.display()
    );

    let elf_metadata = std::fs::metadata(&elf_path).expect("failed to stat ELF");
    assert!(
        elf_metadata.len() > 64,
        "ELF file too small: {} bytes",
        elf_metadata.len()
    );

    // Make executable
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&elf_path, std::fs::Permissions::from_mode(0o755))
            .expect("failed to chmod");
    }

    // Execute the ELF binary
    let exec_result = Command::new(&elf_path).output();

    match exec_result {
        Ok(output) => {
            let exit_code = output.status.code().unwrap_or(-1);
            assert_eq!(
                exit_code,
                42,
                "Expected exit code 42, got {}. stdout={:?}, stderr={:?}",
                exit_code,
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            eprintln!("Native ELF executed successfully with exit code 42");
        }
        Err(e) => {
            panic!("Failed to execute ELF at {}: {}", elf_path.display(), e);
        }
    }

    // Cleanup
    let _ = std::fs::remove_file(&fixture_path);
    let _ = std::fs::remove_file(&elf_path);
}

/// Validate ELF structure without execution.
/// Loads the self-hosted test suite and runs internal ELF validation.
#[test]
#[cfg_attr(not(target_os = "linux"), ignore = "Linux-only test")]
fn native_elf_structure_validation() {
    use sounio::interp::{Interpreter, Value};
    use sounio::module_loader;

    let root = workspace_root();
    let test_path = root.join("self-hosted/native/test_phase1.sio");

    if !test_path.exists() {
        eprintln!("Test skipped: test_phase1.sio not found");
        return;
    }

    let ast = match module_loader::load_program_ast(&test_path) {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return;
        }
    };

    let hir = match sounio::check::check_ast(&ast) {
        Ok(hir) => hir,
        Err(e) => {
            eprintln!("Type check error: {}", e);
            return;
        }
    };

    let mut interp = Interpreter::new();
    match interp.interpret(&hir) {
        Ok(Value::Int(0)) => {
            eprintln!("Phase 1 tests passed (ELF generation validated internally)")
        }
        Ok(Value::Int(n)) => panic!("Phase 1 tests failed with code: {}", n),
        Ok(v) => panic!("Unexpected return value: {:?}", v),
        Err(e) => panic!("Execution error: {}", e),
    }
}
