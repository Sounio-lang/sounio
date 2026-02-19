use std::path::PathBuf;
use std::process::Command;

const DRIVER_ENTRYPOINT_COMPILE_FILE: &str = "bootstrap::driver::compile_file";

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

fn write_temp_dir_path(name: &str) -> PathBuf {
    let unique = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_nanos();
    std::env::temp_dir().join(format!("souc_selfhost_{name}_{unique}_missing"))
}

fn assert_stderr_contains(stderr: &str, needle: &str, context: &str) {
    assert!(
        stderr.contains(needle),
        "{context}. expected to find `{needle}` in stderr, got: {stderr:?}"
    );
}

#[test]
fn selfhost_default_vm_mode_has_no_oracle_marker() {
    let root = workspace_root();
    let program = write_temp_program("default_vm_mode");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .output()
        .expect("run souc");

    let _ = std::fs::remove_file(&program);

    assert!(output.status.success(), "expected exit 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        "SELFHOST=driver-first schema=v1 event=compile_start",
        "expected compile_start telemetry",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "entrypoint={} strict=0 parity=0",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected deterministic default-mode compile metadata",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict=0 status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected successful driver orchestration marker",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=stage_boundary entrypoint={} status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected stage-boundary completion marker",
    );
    assert!(
        !stderr.contains("SELFHOST=oracle backend=rust"),
        "did not expect oracle marker in default vm mode, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in default vm mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_strict_still_succeeds_in_default_vm_mode() {
    let root = workspace_root();
    let program = write_temp_program("strict_default_vm");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_STRICT", "1")
        .output()
        .expect("run souc strict default vm");

    let _ = std::fs::remove_file(&program);

    assert!(
        output.status.success(),
        "expected strict mode to succeed in vm mode"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        "SELFHOST=driver-first schema=v1 event=compile_start",
        "expected compile_start telemetry",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "entrypoint={} strict=1 parity=0",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected deterministic strict-mode compile metadata",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict=1 status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected successful driver orchestration marker in strict mode",
    );
    assert!(
        !stderr.contains("SELFHOST=oracle backend=rust"),
        "did not expect oracle marker in strict vm mode, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in strict vm mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_oracle_mode_is_rejected_with_clear_error() {
    let root = workspace_root();
    let program = write_temp_program("oracle_mode_strict");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_STRICT", "1")
        .env("SOUNIO_SELFHOST_MODE", "oracle")
        .output()
        .expect("run souc strict oracle mode");

    let _ = std::fs::remove_file(&program);

    assert_eq!(output.status.code(), Some(2), "expected exit code 2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported SOUNIO_SELFHOST_MODE='oracle'"),
        "expected unsupported mode message in stderr, got: {stderr:?}"
    );
    assert!(
        stderr.contains("Only 'vm' is supported"),
        "expected supported-mode guidance in stderr, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=oracle backend=rust"),
        "did not expect oracle rust marker in stderr, got: {stderr:?}"
    );
}

#[test]
fn selfhost_default_env_compat_still_succeeds() {
    let root = workspace_root();
    let program = write_temp_program("default_env_switch");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_DEFAULT", "1")
        .output()
        .expect("run souc with selfhost default env");

    let _ = std::fs::remove_file(&program);

    assert!(output.status.success(), "expected exit 0");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        !stderr.contains("SELFHOST=oracle backend=rust"),
        "did not expect oracle marker when SOUNIO_SELFHOST_DEFAULT=1, got: {stderr:?}"
    );
}

#[test]
fn selfhost_oracle_mode_blocked_when_no_rust_fallback_is_enabled() {
    let root = workspace_root();
    let program = write_temp_program("oracle_no_rust_fallback");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_MODE", "oracle")
        .env("SOUNIO_SELFHOST_NO_RUST_FALLBACK", "1")
        .output()
        .expect("run souc oracle with no-rust-fallback");

    let _ = std::fs::remove_file(&program);

    assert_eq!(output.status.code(), Some(2), "expected exit code 2");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unsupported SOUNIO_SELFHOST_MODE='oracle'"),
        "expected unsupported mode message in stderr, got: {stderr:?}"
    );
    assert!(
        stderr.contains("Only 'vm' is supported"),
        "expected supported-mode guidance in stderr, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=oracle backend=rust"),
        "did not expect oracle rust marker in stderr, got: {stderr:?}"
    );
}

#[test]
fn selfhost_parity_mode_emits_oracle_marker_and_succeeds() {
    let root = workspace_root();
    let program = write_temp_program("parity_marker");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_PARITY", "1")
        .output()
        .expect("run souc parity mode");

    let _ = std::fs::remove_file(&program);

    assert!(output.status.success(), "expected parity mode to succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=compile_start entrypoint={} strict=0 parity=1",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected parity-mode compile_start marker",
    );
    assert!(
        stderr.contains("SELFHOST=oracle backend=rust status=match label=file"),
        "expected oracle parity marker in stderr, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in parity mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_default_mode_allows_stage_boundary_fallback_when_driver_unavailable() {
    let root = workspace_root();
    let program = write_temp_program("fallback_allowed_default");
    let missing_stdlib = write_temp_dir_path("stdlib_missing_default");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &missing_stdlib)
        .output()
        .expect("run souc default fallback behavior");

    let _ = std::fs::remove_file(&program);

    assert!(
        output.status.success(),
        "expected default mode to allow fallback when driver orchestration is unavailable"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict=0 status=fallback error_kind=load",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected deterministic fallback marker in default mode",
    );
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=stage_boundary entrypoint={} status=ok",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected stage-boundary completion marker after fallback",
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in default mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_strict_rejects_stage_boundary_fallback_when_driver_unavailable() {
    let root = workspace_root();
    let program = write_temp_program("fallback_rejected_strict");
    let missing_stdlib = write_temp_dir_path("stdlib_missing_strict");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &missing_stdlib)
        .env("SOUNIO_SELFHOST_STRICT", "1")
        .output()
        .expect("run souc strict fallback behavior");

    let _ = std::fs::remove_file(&program);

    assert!(
        !output.status.success(),
        "expected strict mode to reject fallback when driver orchestration is unavailable"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict=1 status=strict_error error_kind=load",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected deterministic strict-error marker",
    );
    assert!(
        stderr.contains("Self-hosted compile failed"),
        "expected strict compile failure message, got: {stderr:?}"
    );
    assert!(
        stderr.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
        "expected stable strict driver failure token, got: {stderr:?}"
    );
    assert!(
        stderr.contains("error_kind=load"),
        "expected strict failure diagnostics to include deterministic error kind, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in strict mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_no_rust_fallback_rejects_stage_boundary_fallback_when_driver_unavailable() {
    let root = workspace_root();
    let program = write_temp_program("fallback_rejected_no_rust");
    let missing_stdlib = write_temp_dir_path("stdlib_missing_no_rust");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &missing_stdlib)
        .env("SOUNIO_SELFHOST_NO_RUST_FALLBACK", "1")
        .output()
        .expect("run souc no-rust-fallback behavior");

    let _ = std::fs::remove_file(&program);

    assert!(
        !output.status.success(),
        "expected no-rust-fallback mode to reject fallback when driver orchestration is unavailable"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        &format!(
            "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict=0 status=strict_error error_kind=load",
            DRIVER_ENTRYPOINT_COMPILE_FILE
        ),
        "expected deterministic strict-error marker without strict mode",
    );
    assert!(
        stderr.contains("Self-hosted compile failed"),
        "expected no-rust-fallback compile failure message, got: {stderr:?}"
    );
    assert!(
        stderr.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
        "expected stable strict driver failure token, got: {stderr:?}"
    );
    assert!(
        stderr.contains("error_kind=load"),
        "expected failure diagnostics to include deterministic error kind, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in no-rust-fallback mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_root_seed_enforce_fails_closed_when_seed_is_missing() {
    let root = workspace_root();
    let stdlib_path = root.join("stdlib").join("compiler");
    let missing_seed = write_temp_dir_path("missing_seed_root").join("seed.sio.bin");
    let missing_checksum = missing_seed.with_extension("sio.bin.sha256");
    let missing_sig = missing_seed.with_extension("sio.bin.sig");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg("self-hosted")
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_BOOTSTRAP_SEED_ENFORCE", "1")
        .env("SOUNIO_BOOTSTRAP_SEED_PATH", &missing_seed)
        .env("SOUNIO_BOOTSTRAP_SEED_SHA256_PATH", &missing_checksum)
        .env("SOUNIO_BOOTSTRAP_SEED_SIG_PATH", &missing_sig)
        .output()
        .expect("run souc self-hosted seed-enforced behavior");

    assert!(
        !output.status.success(),
        "expected seed-enforced self-hosted root run to fail when seed is missing"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        "SELFHOST=run schema=v1 event=selfhost_input_check status=resolved input=self-hosted/main.sio args=1",
        "expected deterministic self-hosted root resolution marker",
    );
    assert!(
        stderr.contains("Self-hosted compile failed for self-hosted/main.sio"),
        "expected compile failure prefix, got: {stderr:?}"
    );
    assert!(
        stderr.contains("BOOTSTRAP_SEED_MISSING"),
        "expected missing-seed failure token, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=driver-first schema=v1 event=compile_start"),
        "did not expect driver pipeline markers when seed enforcement is fail-closed, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=fallback backend=rust"),
        "did not expect rust fallback marker in seed-enforced root mode, got: {stderr:?}"
    );
}

#[test]
fn selfhost_pipeline_rust_without_ghost_stays_on_driver_path() {
    let root = workspace_root();
    let program = write_temp_program("rust_pipeline_no_ghost");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_PIPELINE", "rust")
        .output()
        .expect("run souc with rust pipeline without ghost");

    let _ = std::fs::remove_file(&program);

    assert!(
        output.status.success(),
        "expected rust pipeline request without ghost to succeed on driver path"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_stderr_contains(
        &stderr,
        "SELFHOST=driver-first schema=v1 event=compile_start",
        "expected driver pipeline marker when rust pipeline is requested without ghost",
    );
    assert!(
        !stderr.contains("GHOST MODE"),
        "did not expect ghost warning when SOUNIO_RUST_GHOST is not enabled, got: {stderr:?}"
    );
}

#[test]
fn selfhost_pipeline_rust_with_ghost_emits_transition_warning() {
    let root = workspace_root();
    let program = write_temp_program("rust_pipeline_with_ghost");
    let stdlib_path = root.join("stdlib").join("compiler");

    let output = Command::new(env!("CARGO_BIN_EXE_souc"))
        .current_dir(&root)
        .arg("run")
        .arg(&program)
        .env("SOUNIO_STDLIB_PATH", &stdlib_path)
        .env("SOUNIO_SELFHOST_PIPELINE", "rust")
        .env("SOUNIO_RUST_GHOST", "1")
        .output()
        .expect("run souc with rust pipeline in ghost mode");

    let _ = std::fs::remove_file(&program);

    assert!(
        output.status.success(),
        "expected rust ghost mode to compile and execute successfully"
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.trim() == "42", "expected stdout 42, got: {stdout:?}");

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("GHOST MODE"),
        "expected ghost mode warning in stderr, got: {stderr:?}"
    );
    assert!(
        stderr.contains("Rust is dead. This path will vanish in 0.x.y+1"),
        "expected transition warning payload in stderr, got: {stderr:?}"
    );
    assert!(
        !stderr.contains("SELFHOST=driver-first schema=v1 event=compile_start"),
        "did not expect driver-first compile marker when rust pipeline is explicitly ghost-enabled, got: {stderr:?}"
    );
}
