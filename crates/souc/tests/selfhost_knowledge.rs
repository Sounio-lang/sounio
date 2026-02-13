// Integration test: self-hosted Knowledge type parsing
//
// Verifies that the self-hosted suite loads and parses correctly with
// the new Knowledge type AST nodes. Execution correctness is verified
// by Gate 2 (cargo run --bin souc -- run self-hosted/).

use sounio::module_loader;
use std::path::Path;

#[test]
fn knowledge_selfhost_parses() {
    // Navigate to repo root from crate directory (crates/souc/)
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.join("../..").canonicalize().unwrap();
    let suite_path = repo_root.join("self-hosted");

    // Verify the self-hosted suite loads without parse errors
    let ast = match module_loader::load_program_ast(&suite_path) {
        Ok(ast) => ast,
        Err(e) => {
            panic!("Failed to load self-hosted suite with Knowledge types: {:?}", e);
        }
    };

    // Verify the AST has a reasonable number of items (sanity check)
    assert!(
        ast.items.len() > 100,
        "Expected 100+ items in self-hosted suite, got {}",
        ast.items.len()
    );
}
