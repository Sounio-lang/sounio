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
    // Use RFC 3339 format
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}
