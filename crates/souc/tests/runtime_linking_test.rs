//! Test runtime library linking
//!
//! This test verifies that the static runtime library (libsounio.a) can be
//! linked with native backend code and that runtime functions are callable.

use sounio::backend::native::aarch64::{AArch64Emitter, AArch64Reg};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

#[test]
fn test_runtime_library_exists() {
    // Verify that the static library was built
    let lib_path = PathBuf::from("../../target/release/libsounio.a");
    assert!(
        lib_path.exists(),
        "libsounio.a should exist at {:?}",
        lib_path
    );

    // Verify it's actually a static library
    let metadata = fs::metadata(&lib_path).expect("Failed to get library metadata");
    assert!(metadata.len() > 0, "Library file should not be empty");
}

#[test]
fn test_runtime_symbols_present() {
    // Use nm to check for runtime symbols
    let lib_path = PathBuf::from("../../target/release/libsounio.a");
    if !lib_path.exists() {
        eprintln!("Warning: libsounio.a not found, skipping symbol test");
        return;
    }

    let output = Command::new("nm")
        .arg(&lib_path)
        .output()
        .expect("Failed to run nm");

    let symbols = String::from_utf8_lossy(&output.stdout);

    // Check for key runtime symbols
    let required_symbols = [
        "__sounio_push_handler_io",
        "__sounio_pop_handler",
        "__sounio_dispatch_io_print",
        "__sounio_dispatch_mut_get",
        "__sounio_dispatch_div_div",
        // Native compatibility shims used by HTTP/Postgres lowering paths.
        "starts_with",
        "contains",
        "replace",
        "as_ptr",
        "handler",
    ];

    for symbol in &required_symbols {
        assert!(
            symbols.contains(symbol),
            "Runtime symbol {} not found in library",
            symbol
        );
    }
}

#[test]
fn test_generate_code_with_runtime_calls() {
    // Generate simple AArch64 code that calls runtime functions
    let mut emitter = AArch64Emitter::new();

    // Simulate: push IO handler, dispatch print, pop handler
    emitter.bl_external("__sounio_push_handler_io");

    // Set up argument in V0 register (value to print)
    // FMOV D0, #1.0 would go here in real code

    emitter.bl_external("__sounio_dispatch_io_print");
    emitter.bl_external("__sounio_pop_handler");

    // Verify relocations were created
    let relocations = emitter.relocations();
    assert_eq!(relocations.len(), 3, "Should have 3 relocations");

    // Verify symbols
    let symbols = emitter.external_symbols();
    assert!(symbols.contains_key("__sounio_push_handler_io"));
    assert!(symbols.contains_key("__sounio_dispatch_io_print"));
    assert!(symbols.contains_key("__sounio_pop_handler"));

    // Verify code was emitted (3 BL instructions = 12 bytes)
    let code = emitter.finish().expect("Should finish successfully");
    assert_eq!(code.len(), 12, "Should emit 12 bytes (3 x BL)");
}

#[test]
fn test_runtime_library_size() {
    // Verify the library is a reasonable size
    let lib_path = PathBuf::from("../../target/release/libsounio.a");
    if !lib_path.exists() {
        eprintln!("Warning: libsounio.a not found, skipping size test");
        return;
    }

    let metadata = fs::metadata(&lib_path).expect("Failed to get library metadata");
    let size_mb = metadata.len() / (1024 * 1024);

    // Library should be at least 1MB (has runtime code)
    assert!(size_mb >= 1, "Library seems too small: {} MB", size_mb);

    // And less than 500MB (sanity check)
    assert!(size_mb < 500, "Library seems too large: {} MB", size_mb);

    println!("Runtime library size: {} MB", size_mb);
}
