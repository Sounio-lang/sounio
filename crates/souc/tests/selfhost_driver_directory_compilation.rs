use std::fs;
use std::path::PathBuf;
use std::process::Command;

use tempfile::TempDir;

const DRIVER_ENTRYPOINT_COMPILE_FILE: &str = "bootstrap::driver::compile_file";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn assert_stderr_contains(stderr: &str, needle: &str, context: &str) {
    assert!(
        stderr.contains(needle),
        "{context}. expected to find `{needle}` in stderr, got: {stderr:?}"
    );
}

fn write_temp_sio_file(dir: &TempDir, name: &str, content: &str) -> PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, content).expect("write temp .sio file");
    path
}

#[test]
fn selfhost_driver_compiles_directory_with_multiple_sio_files() {
    let root = workspace_root();
    let stdlib_path = root.join("stdlib").join("compiler");

    // Directory compilation via the self-hosted driver (bootstrap::driver::compile_file)
    // uses `list_dir()` to find `.sio` siblings. This regression test ensures that
    // passing a directory path (not a file path) works end-to-end.
    let dir = TempDir::new().expect("temp dir");
    write_temp_sio_file(
        &dir,
        "extra.sio",
        r#"
// Intentionally unused; ensures directory compilation sees >1 .sio file.
"#,
    );
    write_temp_sio_file(
        &dir,
        "main.sio",
        r#"
fn main() -> i32 {
    return 42
}
"#,
    );

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(dir.path())
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_STRICT", "1")
        .env("SOUNIO_SELFHOST_PIPELINE", "driver")
        .env("SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT", "1")
        .output()
        .expect("run souc on directory");

    assert!(
        output.status.success(),
        "expected exit 0, got {:?} (stderr={})",
        output.status.code(),
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert_eq!(stdout.trim(), "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        "SELFHOST=driver-first schema=v1 event=compile_start",
        "expected compile_start telemetry",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected driver_output marker for directory compilation",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=stage_boundary entrypoint={} status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected stage_boundary completion marker",
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in strict mode, got: {stderr:?}"
    );
}
