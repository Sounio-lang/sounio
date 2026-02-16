// Integration test: self-hosted multi-module compilation
//
// Verifies that the self-hosted compiler can compile programs with imports
// using the compile_multimodule_program pipeline.

use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

#[test]
fn selfhost_multimodule_simple_import() {
    let _root = workspace_root();
    let dir = TempDir::new().expect("temp dir");

    // Create math/basic.sio with helper functions (local module, not stdlib)
    let math_dir = dir.path().join("math");
    fs::create_dir_all(&math_dir).expect("create math dir");

    fs::write(
        math_dir.join("basic.sio"),
        r#"// math::basic

fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(a: i32, b: i32) -> i32 {
    a * b
}
"#,
    )
    .expect("write basic.sio");

    // Create main.sio that imports math::basic
    fs::write(
        dir.path().join("main.sio"),
        r#"use math::basic

fn main() -> i32 with IO {
    let sum = add(10, 20)
    let product = multiply(3, 4)

    print("Sum: ")
    print_int(sum)
    print("\n")

    print("Product: ")
    print_int(product)
    print("\n")

    0
}
"#,
    )
    .expect("write main.sio");

    // Note: This test verifies that the module loader compiles without errors.
    // Actually invoking compile_multimodule_program requires the self-hosted
    // compiler to support a "compile-multi" CLI command, which can be tested
    // once native codegen M1-M6 are complete and we have a self-hosted binary.

    // For now, we verify the Rust compiler can handle this via its module_loader
    use sounio::module_loader;
    let main_path = dir.path().join("main.sio");

    let ast = match module_loader::load_program_ast(&main_path) {
        Ok(ast) => ast,
        Err(e) => {
            panic!("Failed to load multi-module program: {:?}", e);
        }
    };

    // Verify we loaded functions from both modules
    assert!(
        ast.items.len() >= 3,
        "Expected at least 3 functions (add, multiply, main), got {}",
        ast.items.len()
    );
}
