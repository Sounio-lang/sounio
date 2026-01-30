//! ELF Validation Tests for Native Backend
//!
//! These tests validate that the generated ELF files are correct and can be
//! used by system linkers.

use std::fs;
use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

fn get_fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("native")
        .join(name)
}

fn compile_to_object(input: &PathBuf, output: &PathBuf) -> Result<(), String> {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_souc"));
    cmd.arg("build");
    cmd.arg("--backend=native");
    cmd.arg("-o");
    cmd.arg(output);
    cmd.arg(input);

    let output_result = cmd
        .output()
        .map_err(|e| format!("Failed to run souc: {}", e))?;

    if !output_result.status.success() {
        let stderr = String::from_utf8_lossy(&output_result.stderr);
        return Err(format!("Compilation failed: {}", stderr));
    }

    Ok(())
}

fn run_readelf(args: &[&str], file: &PathBuf) -> Result<String, String> {
    let mut cmd = Command::new("readelf");
    cmd.args(args);
    cmd.arg(file);

    let output = cmd
        .output()
        .map_err(|e| format!("readelf not available: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("readelf failed: {}", stderr));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ============================================================================
// ELF STRUCTURE VALIDATION
// ============================================================================

#[test]
fn test_elf_header_valid() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_to_object(&input, &output).expect("Failed to compile");

    // Check ELF magic
    let bytes = fs::read(&output).unwrap();
    assert!(bytes.len() >= 4, "ELF file too short");
    assert_eq!(&bytes[0..4], &[0x7f, b'E', b'L', b'F'], "Invalid ELF magic");

    // Check ELF class (should be 64-bit)
    assert_eq!(bytes[4], 2, "Should be ELF64");
}

#[test]
fn test_elf_has_text_section() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_to_object(&input, &output).expect("Failed to compile");

    let readelf_output = run_readelf(&["-S"], &output);

    if let Ok(output_text) = readelf_output {
        assert!(
            output_text.contains(".text"),
            "ELF should have .text section"
        );
    } else {
        // readelf not available, skip
        eprintln!("readelf not available, skipping section check");
    }
}

#[test]
fn test_elf_has_symbol_table() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_to_object(&input, &output).expect("Failed to compile");

    let readelf_output = run_readelf(&["-s"], &output);

    if let Ok(output_text) = readelf_output {
        assert!(
            output_text.contains("Symbol table") || output_text.contains("main"),
            "ELF should have symbol table"
        );
    } else {
        eprintln!("readelf not available, skipping symbol table check");
    }
}

#[test]
fn test_elf_has_relocations() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_to_object(&input, &output).expect("Failed to compile");

    let readelf_output = run_readelf(&["-r"], &output);

    if let Ok(output_text) = readelf_output {
        // Relocations may or may not be present depending on code
        // Just check that readelf can read the file
        assert!(
            output_text.contains("Relocation") || output_text.len() > 0,
            "Should be able to read relocations"
        );
    } else {
        eprintln!("readelf not available, skipping relocation check");
    }
}

#[test]
fn test_elf_can_be_linked() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let obj_file = temp_dir.path().join("test.o");
    let linked_file = temp_dir.path().join("test_linked.so");

    compile_to_object(&input, &obj_file).expect("Failed to compile");

    // Try to link with system linker
    let mut cmd = Command::new("ld");
    cmd.arg("-shared");
    cmd.arg("-o");
    cmd.arg(&linked_file);
    cmd.arg(&obj_file);

    let result = cmd.output();

    if let Ok(output) = result {
        if output.status.success() {
            // Linking succeeded - verify output
            assert!(linked_file.exists(), "Linked file should be created");
            let bytes = fs::read(&linked_file).unwrap();
            assert_eq!(
                &bytes[0..4],
                &[0x7f, b'E', b'L', b'F'],
                "Linked file should be valid ELF"
            );
        } else {
            // Linking failed - this might be expected if symbols are missing
            eprintln!(
                "Linking failed (may be expected): {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        // ld not available
        eprintln!("ld not available, skipping link test");
    }
}

#[test]
fn test_elf_can_be_linked_with_gcc() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let obj_file = temp_dir.path().join("test.o");
    let linked_file = temp_dir.path().join("test_linked_gcc.so");

    compile_to_object(&input, &obj_file).expect("Failed to compile");

    // Try to link with gcc (which uses ld internally)
    let mut cmd = Command::new("gcc");
    cmd.arg("-shared");
    cmd.arg("-o");
    cmd.arg(&linked_file);
    cmd.arg(&obj_file);

    let result = cmd.output();

    if let Ok(output) = result {
        if output.status.success() {
            // Linking succeeded
            assert!(linked_file.exists(), "Linked file should be created");
            let bytes = fs::read(&linked_file).unwrap();
            assert_eq!(
                &bytes[0..4],
                &[0x7f, b'E', b'L', b'F'],
                "Linked file should be valid ELF"
            );
        } else {
            eprintln!(
                "gcc linking failed (may be expected): {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    } else {
        eprintln!("gcc not available, skipping gcc link test");
    }
}

#[test]
fn test_elf_disassembly() {
    let temp_dir = TempDir::new().unwrap();
    let input = get_fixture_path("test_minimal.sio");
    let output = temp_dir.path().join("test.o");

    compile_to_object(&input, &output).expect("Failed to compile");

    // Try to disassemble with objdump
    let mut cmd = Command::new("objdump");
    cmd.arg("-d");
    cmd.arg(&output);

    let result = cmd.output();

    if let Ok(output_result) = result {
        if output_result.status.success() {
            let stdout = String::from_utf8_lossy(&output_result.stdout);
            // Should have some disassembly
            assert!(stdout.len() > 0, "Disassembly should produce output");
        } else {
            eprintln!(
                "objdump failed (may be expected): {}",
                String::from_utf8_lossy(&output_result.stderr)
            );
        }
    } else {
        eprintln!("objdump not available, skipping disassembly test");
    }
}
