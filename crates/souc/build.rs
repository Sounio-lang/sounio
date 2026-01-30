//! Build script for the Sounio compiler
//!
//! This script captures build-time information such as:
//! - Git commit hash
//! - Build date
//! - Target triple

use std::process::Command;

fn main() {
    // Capture git information
    let git_hash = get_git_hash();
    let git_dirty = is_git_dirty();
    let build_date = get_build_date();

    // Set environment variables for use in the binary
    println!("cargo:rerun-if-changed=.git/HEAD");
    println!("cargo:rerun-if-changed=.git/index");

    let hash_suffix = if git_dirty { "-dirty" } else { "" };
    println!(
        "cargo:rustc-env=SOUNIO_GIT_HASH={}{}",
        git_hash.unwrap_or_else(|| "unknown".to_string()),
        hash_suffix
    );

    println!("cargo:rustc-env=SOUNIO_BUILD_DATE={}", build_date);

    // Compile AArch64 continuation assembly (only on AArch64 targets)
    #[cfg(target_arch = "aarch64")]
    {
        let asm_file = "src/backend/native/aarch64_continuation.S";
        println!("cargo:rerun-if-changed={}", asm_file);

        cc::Build::new()
            .file(asm_file)
            .compile("aarch64_continuation");

        println!("cargo:rustc-link-lib=static=aarch64_continuation");
    }
}

fn get_git_hash() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short=10", "HEAD"])
        .output()
        .ok()?;

    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn is_git_dirty() -> bool {
    Command::new("git")
        .args(["diff-index", "--quiet", "HEAD", "--"])
        .status()
        .map(|s| !s.success())
        .unwrap_or(false)
}

fn get_build_date() -> String {
    // Use a simple date format via system date command
    Command::new("date")
        .arg("+%Y-%m-%d")
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}
