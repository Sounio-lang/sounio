use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

fn souc_bin() -> PathBuf {
    if let Some(path) = std::env::var_os("CARGO_BIN_EXE_souc") {
        return PathBuf::from(path);
    }

    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let candidate = workspace_root
        .join("target")
        .join("debug")
        .join(if cfg!(windows) { "souc.exe" } else { "souc" });
    if !candidate.exists() {
        let status = Command::new("cargo")
            .args(["build", "--bin", "souc"])
            .current_dir(&workspace_root)
            .status()
            .expect("build souc binary");
        assert!(status.success(), "failed to build souc binary");
    }
    assert!(candidate.exists(), "souc binary not found at {:?}", candidate);
    candidate
}

fn run_check(source: &str) -> Output {
    let temp_dir = TempDir::new().expect("temp dir");
    let file = temp_dir.path().join("test.sio");
    fs::write(&file, source).expect("write fixture");

    Command::new(souc_bin())
        .args([
            "check",
            file.to_str().expect("utf8 path"),
            "--error-format=json",
        ])
        .output()
        .expect("execute souc")
}

fn combined_output(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn knightian_argument_refused_for_confidence_gate() {
    let source = r#"
fn require_confidence(signal: Knowledge[f64, ε >= 0.8]) -> Knowledge[f64, ε >= 0.8] {
    signal
}

fn main() with IO {
    let unknown: Knowledge[f64, ε=⊥] = Knowledge { value: 0.7 }
    let _ = require_confidence(unknown)
}
"#;

    let output = run_check(source);
    assert!(!output.status.success(), "check unexpectedly succeeded");
    let text = combined_output(&output);
    assert!(
        text.contains("Knightian uncertainty (ε=⊥) cannot satisfy required confidence"),
        "missing Knightian refusal diagnostic: {text}"
    );
}

#[test]
fn alias_wrapped_constraints_still_enforced() {
    let source = r#"
type StrongSignal = Knowledge[f64, ε >= 0.9];

fn require_strong(signal: StrongSignal) -> StrongSignal {
    signal
}

fn main() with IO {
    let weak: Knowledge[f64, ε=0.4] = Knowledge { value: 1.0 }
    let _ = require_strong(weak)
}
"#;

    let output = run_check(source);
    assert!(!output.status.success(), "check unexpectedly succeeded");
    let text = combined_output(&output);
    assert!(
        text.contains("Type mismatch: expected confidence ε >= 0.9, found ε >= 0.4"),
        "missing alias enforcement mismatch: {text}"
    );
}

#[test]
fn temporal_validity_mismatch_is_refused() {
    let source = r#"
fn require_march_window(
    signal: Knowledge[f64, ValidUntil("2020-03-31"), Derived]
) -> Knowledge[f64, ValidUntil("2020-03-31"), Derived] {
    signal
}

fn main() with IO {
    let stale: Knowledge[f64, ValidUntil("2020-03-10"), Derived] = Knowledge {
        value: 1.0,
        epsilon: 0.92,
        validity: "2020-03-10",
        provenance: "icu_report_2020_03_10"
    }
    let _ = require_march_window(stale)
}
"#;

    let output = run_check(source);
    assert!(!output.status.success(), "check unexpectedly succeeded");
    let text = combined_output(&output);
    assert!(
        text.contains("Temporal validity window"),
        "missing temporal refusal diagnostic: {text}"
    );
}
