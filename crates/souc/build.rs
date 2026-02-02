//! Build script for the Sounio compiler
//!
//! This script captures build-time information such as:
//! - Git commit hash
//! - Build date
//! - Target triple

use std::fs;
use std::path::Path;
use std::process::Command;

fn main() {
    // Embed stdlib/compiler modules
    embed_stdlib_modules();

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

    // Compile x86-64 continuation assembly (only on x86-64 targets)
    #[cfg(target_arch = "x86_64")]
    {
        let asm_file = "src/backend/native/x86_64_continuation.S";
        println!("cargo:rerun-if-changed={}", asm_file);

        cc::Build::new()
            .file(asm_file)
            .compile("x86_64_continuation");

        println!("cargo:rustc-link-lib=static=x86_64_continuation");
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

fn embed_stdlib_modules() {
    // Tell Cargo to rerun if stdlib/compiler changes
    println!("cargo:rerun-if-changed=../../stdlib/compiler");

    let stdlib_path = Path::new("../../stdlib/compiler");

    if !stdlib_path.exists() {
        println!("cargo:warning=stdlib/compiler not found, skipping embedding");
        return;
    }

    // Discover all .sio files
    let mut modules = Vec::new();
    discover_sio_files(stdlib_path, stdlib_path, &mut modules);

    if modules.is_empty() {
        println!("cargo:warning=No .sio files found in stdlib/compiler");
        return;
    }

    println!(
        "cargo:warning=Embedding {} stdlib/compiler modules",
        modules.len()
    );
}

fn discover_sio_files(base_path: &Path, current_path: &Path, modules: &mut Vec<String>) {
    if let Ok(entries) = fs::read_dir(current_path) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() && path.extension().map_or(false, |ext| ext == "sio") {
                if let Some(rel_path) = path.strip_prefix(base_path).ok() {
                    if let Some(module_path) = rel_path.to_str() {
                        modules.push(module_path.to_string());
                        println!(
                            "cargo:rerun-if-changed=../../stdlib/compiler/{}",
                            module_path
                        );
                    }
                }
            } else if path.is_dir() {
                // Recursively discover in subdirectories
                discover_sio_files(base_path, &path, modules);
            }
        }
    }
}
