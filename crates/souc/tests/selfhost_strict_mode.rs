use std::path::PathBuf;
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf()
}

fn write_temp_program(name: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    let path = std::env::temp_dir().join(format!("souc_selfhost_{name}_{unique}.sio"));

    std::fs::write(
        &path,
        r#"fn main() -> i32 {
    let x: i32 = 42
    return x
}
"#,
    )
    .expect("write temp program");

    path
}

#[test]
fn selfhost_fallback_emits_marker_and_succeeds_by_default() {
    let root = workspace_root();
    let program = write_temp_program("fallback_marker");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg("--use-sounio-compiler")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("run souc");

    let _ = std::fs::remove_file(&program);

    assert!(output.status.success(), "expected exit 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("SELFHOST=fallback backend=rust"),
        "expected fallback marker in stderr, got: {stderr:?}"
    );
}

#[test]
fn selfhost_strict_exits_2_when_fallback_occurs() {
    let root = workspace_root();
    let program = write_temp_program("strict_exit_2");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg("--use-sounio-compiler")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_STRICT", "1")
        .output()
        .expect("run souc strict");

    let _ = std::fs::remove_file(&program);

    assert_eq!(output.status.code(), Some(2), "expected exit code 2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("SELFHOST=fallback backend=rust"),
        "expected fallback marker in stderr, got: {stderr:?}"
    );
}

