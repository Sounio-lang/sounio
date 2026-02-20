// crates/souc/tests/formal_verification.rs
//! Phase 8 — Lean 4 formal verification integration test.
//!
//! Verifies that the Lean 4 proofs in `formal/lean4/` compile cleanly via
//! `lake build`.  Requires `elan` / `lake` to be installed.  Run with:
//!
//! ```bash
//! cargo test -p souc --test formal_verification -- --include-ignored
//! ```

/// Locate the `lake` binary: check PATH first, then the default elan install.
fn lake_binary() -> std::path::PathBuf {
    // 1. Check if lake is already on PATH
    if let Ok(output) = std::process::Command::new("which").arg("lake").output() {
        if output.status.success() {
            let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if !path.is_empty() {
                return std::path::PathBuf::from(path);
            }
        }
    }
    // 2. Fall back to default elan install location
    let home = std::env::var("HOME").unwrap_or_else(|_| "/root".to_string());
    let elan_lake = std::path::PathBuf::from(&home).join(".elan/bin/lake");
    if elan_lake.exists() {
        return elan_lake;
    }
    // 3. Return bare name — Command will give a clear error
    std::path::PathBuf::from("lake")
}

#[test]
#[ignore = "Phase 8: requires elan; run with -- --include-ignored"]
fn lean4_linear_and_effects_proofs_compile() {
    let formal_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("formal/lean4");

    assert!(
        formal_dir.exists(),
        "formal/lean4/ not found at {}",
        formal_dir.display()
    );

    // Confirm no sorry is present (comments mentioning sorry are allowed)
    let grep_out = std::process::Command::new("grep")
        .args(["-r", "--include=*.lean", "sorry", formal_dir.to_str().unwrap()])
        .output()
        .expect("grep not available");

    let sorry_lines: Vec<String> = String::from_utf8_lossy(&grep_out.stdout)
        .lines()
        .filter(|l| !l.contains("No `sorry`") && !l.contains("No sorry"))
        .map(|l| l.to_string())
        .collect();
    assert!(
        sorry_lines.is_empty(),
        "sorry found in Lean sources:\n{}",
        sorry_lines.join("\n")
    );

    let lake = lake_binary();
    let status = std::process::Command::new(&lake)
        .arg("build")
        .current_dir(&formal_dir)
        .status()
        .unwrap_or_else(|e| {
            panic!(
                "Failed to run lake ({}) — install elan:\n\
                 curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh -s -- -y\n\
                 Error: {}",
                lake.display(),
                e
            )
        });

    assert!(status.success(), "lake build failed in {}", formal_dir.display());
}
