//! Phase 7 — Diverse Double-Compilation (DDC) Check
//!
//! Verifies the Wheeler DDC property: when `souc` is compiled by two different
//! host toolchains (GCC-linked vs Clang-linked), both resulting binaries produce
//! **byte-identical output** for the same Sounio source input.
//!
//! A malicious backdoor injected by one host compiler cannot survive compilation
//! by the other, so divergent output is a reliable signal of compromise.
//!
//! # Running
//!
//! These tests are `#[ignore]` by default because they require two pre-built
//! souc binaries. Build them first:
//!
//! ```bash
//! bash scripts/diverse_double_compile_check.sh --skip-build || \
//!   bash scripts/diverse_double_compile_check.sh
//! ```
//!
//! Then run:
//!
//! ```bash
//! cargo test -p souc --test ddc_check -- --include-ignored
//! ```

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn ddc_bin(compiler: &str) -> PathBuf {
    workspace_root()
        .join(format!("target/ddc-{}/debug/souc", compiler))
}

fn stdlib_path() -> PathBuf {
    workspace_root().join("stdlib")
}

/// Run `souc run <program>` and return combined stdout+stderr as a String.
fn run_souc(bin: &PathBuf, program: &PathBuf) -> Result<String, String> {
    let output = Command::new(bin)
        .arg("run")
        .arg(program)
        .env("SOUNIO_STDLIB_PATH", stdlib_path())
        .output()
        .map_err(|e| format!("failed to spawn {:?}: {}", bin, e))?;

    let mut combined = String::new();
    combined.push_str(&String::from_utf8_lossy(&output.stdout));
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    Ok(combined)
}

/// The reference Sounio program used for DDC comparison.
/// Must be deterministic and produce visible output.
const DDC_REF_SOURCE: &str = r#"
fn main() with IO {
    print("DDC-REF-PASS\n")
    print_int(6 * 7)
    print("\n")
}
"#;

#[test]
#[ignore = "DDC check: build gcc/clang variants first with scripts/diverse_double_compile_check.sh"]
fn test_diverse_double_compile_output_identical() {
    let gcc_bin = ddc_bin("gcc");
    let clang_bin = ddc_bin("clang");

    // Skip gracefully if binaries not built yet
    if !gcc_bin.exists() || !clang_bin.exists() {
        eprintln!(
            "SKIP: DDC binaries not found.\n\
             Build them with: bash scripts/diverse_double_compile_check.sh\n\
             Expected:\n  {}\n  {}",
            gcc_bin.display(),
            clang_bin.display()
        );
        return;
    }

    // Write reference program to a temp file
    let tmp_dir = PathBuf::from("/tmp/sounio-ddc-test");
    fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let ref_file = tmp_dir.join("ddc_ref.sio");
    fs::write(&ref_file, DDC_REF_SOURCE).expect("write ref program");

    // Run with GCC-linked souc
    let gcc_out = run_souc(&gcc_bin, &ref_file)
        .expect("GCC-linked souc run failed");

    // Run with Clang-linked souc
    let clang_out = run_souc(&clang_bin, &ref_file)
        .expect("Clang-linked souc run failed");

    // Cleanup
    let _ = fs::remove_file(&ref_file);

    // DDC assertion: outputs must be identical
    assert_eq!(
        gcc_out, clang_out,
        "DDC FAIL: GCC-linked and Clang-linked souc produced divergent output!\n\
         GCC output:\n{}\n\
         Clang output:\n{}",
        gcc_out, clang_out
    );

    eprintln!("DDC PASS: outputs identical ({} bytes)", gcc_out.len());
}

#[test]
#[ignore = "DDC check: build gcc/clang variants first with scripts/diverse_double_compile_check.sh"]
fn test_ddc_determinism_within_single_binary() {
    // Sanity: a single binary must also be deterministic across two runs.
    // This is a precondition for DDC to be meaningful.
    let gcc_bin = ddc_bin("gcc");
    if !gcc_bin.exists() {
        eprintln!("SKIP: GCC DDC binary not found");
        return;
    }

    let tmp_dir = PathBuf::from("/tmp/sounio-ddc-test");
    fs::create_dir_all(&tmp_dir).expect("create tmp dir");
    let ref_file = tmp_dir.join("ddc_ref_det.sio");
    fs::write(&ref_file, DDC_REF_SOURCE).expect("write ref program");

    let out1 = run_souc(&gcc_bin, &ref_file).expect("run 1 failed");
    let out2 = run_souc(&gcc_bin, &ref_file).expect("run 2 failed");

    let _ = fs::remove_file(&ref_file);

    assert_eq!(
        out1, out2,
        "DETERMINISM FAIL: same binary produced different output on two runs"
    );
    eprintln!("DETERMINISM PASS: single binary is reproducible");
}
