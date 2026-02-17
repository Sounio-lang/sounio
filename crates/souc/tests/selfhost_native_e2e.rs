//! End-to-end test: Sounio source → Self-hosted compilation → Native ELF
//!
//! This test validates the complete self-hosted native pipeline:
//! 1. Parse source with Rust frontend (for now)
//! 2. Lower to IR with Rust (for now)
//! 3. Call self-hosted compile_to_elf() via interpreter
//! 4. Extract ELF bytes
//! 5. Write to disk and execute
//!
//! Future: Replace Rust frontend with self-hosted parser/checker

#[test]
#[ignore] // Enable when wiring is complete
fn selfhost_native_simple_return() -> anyhow::Result<()> {
    // Test: fn main() -> i64 { 42 }

    // Step 1: Parse to IR (using Rust for now)
    let source = "fn main() -> i64 { 42 }";
    let ir_module = parse_and_lower_to_ir(source)?;

    // Step 2: Call self-hosted compile_to_elf()
    let elf_bytes = call_selfhosted_compile_to_elf(&ir_module)?;

    // Step 3: Write to disk
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::process::Command;

    let temp_path = "/tmp/selfhost_native_test";
    fs::write(temp_path, &elf_bytes)?;
    fs::set_permissions(temp_path, fs::Permissions::from_mode(0o755))?;

    // Step 4: Execute
    let output = Command::new(temp_path).output()?;

    // Step 5: Verify
    assert_eq!(output.status.code(), Some(42), "Expected exit code 42");

    // Cleanup
    let _ = fs::remove_file(temp_path);

    println!("✅ Self-hosted native compilation SUCCESS!");
    println!("   Source: {}", source);
    println!("   ELF size: {} bytes", elf_bytes.len());
    println!("   Exit code: 42");

    Ok(())
}

#[test]
#[ignore]
fn selfhost_native_multimodule() -> anyhow::Result<()> {
    // Test multi-module compilation with imports
    // math.sio: fn add(x, y) -> i64 { x + y }
    // main.sio: import math; fn main() { math::add(10, 32) }

    // This tests the FULL STACK:
    // - Multi-module parsing
    // - Import resolution
    // - Module linking
    // - Native code generation
    // - Execution

    todo!("Wire multi-module native compilation");
}

/// Parse source and lower to HIR using Rust frontend
fn parse_and_lower_to_ir(_source: &str) -> anyhow::Result<sounio::hir::Hir> {
    todo!("Use Rust parser/checker/lowerer")
}

/// Call self-hosted compile_to_elf() via interpreter
fn call_selfhosted_compile_to_elf(_ir: &sounio::hir::Hir) -> anyhow::Result<Vec<u8>> {
    // Strategy:
    // 1. Serialize HIR to self-hosted format
    // 2. Load self-hosted/native/codegen.sio into interpreter
    // 3. Call compile_to_elf(hir_module)
    // 4. Extract Elf64Binary struct
    // 5. Return bytes[0..len]

    todo!("Wire self-hosted compile_to_elf via interpreter")
}
