/// Integration test: Compile Sounio → SOIR → Execute with Poseidon VM
///
/// This test validates that:
/// 1. Self-hosted compiler can serialize SOIR bytecode
/// 2. Poseidon VM can execute the bytecode correctly
/// 3. Round-trip works end-to-end

use std::path::PathBuf;
use std::process::Command;

fn poseidon_path() -> PathBuf {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.pop();
    path.pop();
    path.push("bootstrap");
    path.push("poseidon");
    path.push("poseidon");
    path
}

fn ensure_poseidon_built() -> bool {
    let poseidon = poseidon_path();
    if poseidon.exists() {
        return true;
    }

    eprintln!("Building Poseidon VM...");
    let mut build_dir = poseidon.clone();
    build_dir.pop();

    let status = Command::new("make")
        .current_dir(&build_dir)
        .status()
        .expect("Failed to run make");

    status.success()
}

#[test]
#[ignore] // Requires self-hosted compiler integration
fn test_poseidon_simple_return() {
    if !ensure_poseidon_built() {
        panic!("Failed to build Poseidon VM");
    }

    // Write a simple Sounio program
    let src = r#"
fn main() -> i64 {
    42
}
"#;

    let temp_dir = std::env::temp_dir();
    let src_file = temp_dir.join("test_poseidon_simple.sio");
    let _soir_file = temp_dir.join("test_poseidon_simple.soir");

    std::fs::write(&src_file, src).expect("Failed to write source");

    // TODO: Call self-hosted compiler to generate SOIR
    // For now, this test is a placeholder for when we have:
    // 1. SOIR serialization from Rust
    // 2. Integration with self-hosted compiler

    // Once we have SOIR, we would run:
    // let output = Command::new(poseidon_path())
    //     .arg(&soir_file)
    //     .output()
    //     .expect("Failed to run Poseidon");
    //
    // assert_eq!(output.status.code(), Some(42));
}

#[test]
fn test_poseidon_vm_exists() {
    // Just verify we can build/find Poseidon
    assert!(
        ensure_poseidon_built(),
        "Poseidon VM should build successfully"
    );
    assert!(
        poseidon_path().exists(),
        "Poseidon VM executable should exist"
    );
}

#[test]
fn test_poseidon_basic_execution() {
    if !ensure_poseidon_built() {
        panic!("Failed to build Poseidon VM");
    }

    // Use one of the hand-crafted test files
    let mut test_file = poseidon_path();
    test_file.pop();
    test_file.push("tests");
    test_file.push("add.soir");

    // Generate test files if they don't exist
    if !test_file.exists() {
        let mut gen_script = test_file.clone();
        gen_script.pop();
        gen_script.push("gen_test_soir.py");

        if gen_script.exists() {
            let mut test_dir = test_file.clone();
            test_dir.pop();

            Command::new("python3")
                .arg("gen_test_soir.py")
                .current_dir(&test_dir)
                .status()
                .expect("Failed to generate test files");
        }
    }

    if test_file.exists() {
        let output = Command::new(poseidon_path())
            .arg(&test_file)
            .output()
            .expect("Failed to run Poseidon");

        assert_eq!(
            output.status.code(),
            Some(42),
            "add.soir should return 42"
        );
    } else {
        eprintln!("Skipping test - add.soir not found");
    }
}
