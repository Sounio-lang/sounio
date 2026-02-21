//! Phase 6 Self-Compilation Integration Tests
//!
//! Tests that the self-hosted compiler can generate native ELF binaries
//! via compile_to_elf(), now that module imports are resolved.

use sounio::ast::Item;
use std::path::PathBuf;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn load_phase6_selfhost_ast() -> miette::Result<sounio::ast::Ast> {
    let selfhost_dir = workspace_root().join("self-hosted");
    let manifest_path = workspace_root().join("bootstrap/selfhost-kernel.manifest");
    assert!(
        manifest_path.exists(),
        "Phase 6 bootstrap manifest missing: {}",
        manifest_path.display()
    );

    let options = sounio::module_loader::ProgramLoadOptions {
        selfhost_bootstrap_manifest: Some(manifest_path),
        selfhost_strict_module_gating: Some(true),
    };
    sounio::module_loader::load_program_ast_with_options(&selfhost_dir, options)
}

fn fn_names_in_ast(ast: &sounio::ast::Ast) -> Vec<&str> {
    ast.items
        .iter()
        .filter_map(|item| match item {
            Item::Function(f) => Some(f.name.as_str()),
            _ => None,
        })
        .collect()
}

/// Verify the self-hosted compiler loads with native/ directory included.
/// This was previously blocked by native/ being in the EXCLUDED list.
#[test]
fn selfhosted_loads_native_module() {
    let ast = load_phase6_selfhost_ast();
    assert!(
        ast.is_ok(),
        "Self-hosted Phase 6 profile should load (including native/): {:?}",
        ast.err()
    );

    let ast = ast.unwrap();
    let names = fn_names_in_ast(&ast);
    assert!(
        names.contains(&"compile_to_elf"),
        "compile_to_elf() should be visible after module import fix"
    );
}

/// Verify write_elf_to_file is accessible in the self-hosted suite.
#[test]
fn selfhosted_loads_write_elf_to_file() {
    let ast = load_phase6_selfhost_ast().unwrap();
    let names = fn_names_in_ast(&ast);
    assert!(
        names.contains(&"write_elf_to_file"),
        "write_elf_to_file() should be visible in self-hosted suite"
    );
}

/// Verify no duplicate symbol conflicts after enabling native/ directory.
#[test]
fn selfhosted_no_duplicate_symbols() {
    let ast = load_phase6_selfhost_ast().unwrap();

    let mut names: Vec<&str> = fn_names_in_ast(&ast);
    names.sort();

    let mut duplicates = Vec::new();
    for window in names.windows(2) {
        if window[0] == window[1] && !duplicates.contains(&window[0]) {
            duplicates.push(window[0]);
        }
    }

    let unique_count = {
        let mut deduped = names.clone();
        deduped.dedup();
        deduped.len()
    };

    assert_eq!(
        names.len(),
        unique_count,
        "No duplicate function definitions should exist. Duplicates: {:?}",
        duplicates
    );
}

/// Test that compile_file_to_native API exists and validates inputs.
#[test]
fn compile_file_to_native_validates_input() {
    let mut compiler = sounio::compiler_loader::SounioCompiler::new_embedded()
        .expect("Should create embedded compiler");

    let result = compiler.compile_file_to_native("/nonexistent/path.sio", "/tmp/test_output");
    assert!(result.is_err(), "Should fail for nonexistent input");

    let err = result.unwrap_err().to_string();
    assert!(
        err.contains("not found"),
        "Error should mention file not found: {}",
        err
    );
}
