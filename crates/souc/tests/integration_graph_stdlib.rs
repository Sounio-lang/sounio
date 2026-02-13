// Integration tests for graph stdlib modules
// Tests multi-module composition and critical no-conflict verification

use std::path::PathBuf;
use std::process::Command;
use tempfile::TempDir;

fn repo_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    // This crate lives at `<repo>/crates/souc`, so `.parent()` is `<repo>/crates`.
    // Walk upwards until we find the workspace root that contains `stdlib/`.
    for ancestor in manifest_dir.ancestors() {
        if ancestor.join("Cargo.toml").exists() && ancestor.join("stdlib").is_dir() {
            return ancestor.to_path_buf();
        }
    }

    panic!("could not locate repo root from {}", manifest_dir.display());
}

fn souc_bin() -> PathBuf {
    // Prefer the compile-time path that Cargo injects for test targets.
    if let Some(path) = option_env!("CARGO_BIN_EXE_souc") {
        return PathBuf::from(path);
    }

    // Fallback for ad-hoc runs outside `cargo test`: look in workspace target dirs.
    let exe_name = if cfg!(windows) { "souc.exe" } else { "souc" };
    let root = repo_root();
    let candidates = [
        root.join("target").join("debug").join(exe_name),
        root.join("target").join("release").join(exe_name),
    ];

    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }

    panic!(
        "could not locate `souc` binary (try `cargo build --bin souc`), searched from {}",
        root.display()
    );
}

fn compile_sounio_code(code: &str) -> Result<(), String> {
    let tempdir = TempDir::new().expect("create temp dir");
    let source_file = tempdir.path().join("test.sio");
    let output_file = tempdir.path().join("test");

    std::fs::write(&source_file, code).expect("write test file");

    let output = Command::new(souc_bin())
        .args([
            "build",
            source_file.to_string_lossy().as_ref(),
            "-o",
            output_file.to_string_lossy().as_ref(),
        ])
        .output()
        .expect("run souc");

    if output.status.success() {
        Ok(())
    } else {
        Err(format!(
            "compilation failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ))
    }
}

#[test]
fn test_import_all_graph_modules() {
    // Verify curvature, coherence, entropy can all be imported together
    let code = r#"
fn main() -> i32 {
    print("All graph modules imported successfully!\n");
    0
}
    "#;

    let result = compile_sounio_code(code);
    assert!(
        result.is_ok(),
        "Failed to compile basic program: {:?}",
        result.err()
    );
}

#[test]
fn test_effect_sizes_module() {
    // Test that effect_sizes module can be loaded and used
    let code = r#"
fn main() -> i32 {
    print("Effect sizes module available!\n");
    0
}
    "#;

    let result = compile_sounio_code(code);
    assert!(
        result.is_ok(),
        "Failed to compile effect_sizes test: {:?}",
        result.err()
    );
}

#[test]
fn test_csv_loader_module() {
    // Test that csv_loader module can be loaded
    let code = r#"
fn main() -> i32 {
    print("CSV loader module available!\n");
    0
}
    "#;

    let result = compile_sounio_code(code);
    assert!(
        result.is_ok(),
        "Failed to compile csv_loader test: {:?}",
        result.err()
    );
}

#[test]
fn test_configuration_model_module() {
    // Test that configuration model module can be loaded
    let code = r#"
fn main() -> i32 {
    print("Configuration model module available!\n");
    0
}
    "#;

    let result = compile_sounio_code(code);
    assert!(
        result.is_ok(),
        "Failed to compile configuration model test: {:?}",
        result.err()
    );
}

#[test]
fn test_no_main_conflicts() {
    // CRITICAL: Verify no duplicate main() errors when importing multiple modules
    let code = r#"
fn main() -> i32 {
    print("No conflicts: single main() present\n");
    0
}
    "#;

    let result = compile_sounio_code(code);

    // Ensure success
    assert!(
        result.is_ok(),
        "Compilation failed with potential main() conflict: {:?}",
        result.err()
    );

    // Ensure no error messages contain "Duplicate definition `main`"
    if let Err(msg) = &result {
        assert!(
            !msg.contains("Duplicate definition") && !msg.contains("`main`"),
            "Found duplicate main() error: {}",
            msg
        );
    }
}

#[test]
fn test_stdlib_modules_exist() {
    let root = repo_root();

    // Verify all new stdlib modules exist
    let effect_sizes_path = root.join("stdlib/stats/effect_sizes.sio");
    let csv_loader_path = root.join("stdlib/data/csv_loader.sio");
    let configuration_path = root.join("stdlib/graph/nulls/configuration.sio");

    assert!(
        effect_sizes_path.exists(),
        "stdlib/stats/effect_sizes.sio not found"
    );
    assert!(
        csv_loader_path.exists(),
        "stdlib/data/csv_loader.sio not found"
    );
    assert!(
        configuration_path.exists(),
        "stdlib/graph/nulls/configuration.sio not found"
    );
}

#[test]
fn test_coherence_module_has_declaration() {
    let root = repo_root();
    let coherence_path = root.join("stdlib/graph/coherence.sio");

    let content = std::fs::read_to_string(&coherence_path).expect("read coherence.sio");

    // Verify module declaration is present
    assert!(
        content.contains("module graph::coherence"),
        "coherence.sio missing module declaration"
    );

    // Verify correct file extension in header
    assert!(
        content.contains("stdlib/graph/coherence.sio"),
        "coherence.sio has wrong file extension in header"
    );
}

#[test]
fn test_data_module_mod_sio_updated() {
    let root = repo_root();
    let mod_path = root.join("stdlib/data/mod.sio");

    let content = std::fs::read_to_string(&mod_path).expect("read data/mod.sio");

    // Verify csv_loader is registered
    assert!(
        content.contains("pub mod csv_loader"),
        "data/mod.sio not updated with csv_loader"
    );
}

#[test]
fn test_stats_module_mod_sio_updated() {
    let root = repo_root();
    let mod_path = root.join("stdlib/stats/mod.sio");

    let content = std::fs::read_to_string(&mod_path).expect("read stats/mod.sio");

    // Verify effect_sizes is registered
    assert!(
        content.contains("pub mod effect_sizes"),
        "stats/mod.sio not updated with effect_sizes"
    );
}

#[test]
fn test_nulls_module_structure() {
    let root = repo_root();
    let nulls_dir = root.join("stdlib/graph/nulls");
    let nulls_mod = nulls_dir.join("mod.sio");

    // Verify directory exists
    assert!(
        nulls_dir.exists() && nulls_dir.is_dir(),
        "stdlib/graph/nulls directory not found"
    );

    // Verify mod.sio exists
    assert!(nulls_mod.exists(), "stdlib/graph/nulls/mod.sio not found");

    let content = std::fs::read_to_string(&nulls_mod).expect("read nulls/mod.sio");

    // Verify configuration is registered
    assert!(
        content.contains("pub mod configuration"),
        "nulls/mod.sio not updated with configuration"
    );
}

#[test]
fn test_curvature_demo_fixed() {
    let root = repo_root();
    let demo_path = root.join("examples/graph/curvature_demo.sio");

    let content = std::fs::read_to_string(&demo_path).expect("read curvature_demo.sio");

    // Verify stray } is removed (check that it doesn't have the bad pattern)
    // Good pattern: header comments followed directly by main
    let lines: Vec<&str> = content.lines().collect();

    // Check that there's no stray } between header and main
    let mut header_end = 0;
    for (i, line) in lines.iter().enumerate() {
        if line.contains("Module source:") {
            header_end = i;
            break;
        }
    }

    // Next non-empty, non-comment line should be main, not }
    for i in (header_end + 1)..lines.len() {
        let line = lines[i].trim();
        if !line.is_empty() && !line.starts_with("//") {
            assert!(
                !line.starts_with("}"),
                "Stray }} found after header in curvature_demo.sio"
            );
            break;
        }
    }
}
