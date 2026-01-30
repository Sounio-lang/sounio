#[cfg(feature = "gpu")]
mod gpu_backend {
    use std::path::PathBuf;
    use std::process::Command;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("compiler crate should live under repo root")
            .to_path_buf()
    }

    fn souc_bin() -> PathBuf {
        std::env::var_os("CARGO_BIN_EXE_souc")
            .map(PathBuf::from)
            .or_else(|| {
                let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("target")
                    .join("debug")
                    .join(if cfg!(windows) { "souc.exe" } else { "souc" });
                candidate.exists().then_some(candidate)
            })
            .expect("could not locate `souc` binary (try `cargo build --bin souc`)")
    }

    #[test]
    fn gpu_backend_builds_ptx() {
        let root = repo_root();
        let souc = souc_bin();
        let fixture = root.join("scripts/fixtures/gpu_minimal.sio");
        let tempdir = tempfile::tempdir().expect("create temp dir");
        let out = tempdir.path().join("gpu_minimal.ptx");

        let output = Command::new(&souc)
            .args([
                "build",
                fixture.to_string_lossy().as_ref(),
                "--backend",
                "gpu",
                "-o",
                out.to_string_lossy().as_ref(),
            ])
            .output()
            .expect("run souc");

        assert!(
            output.status.success(),
            "souc build failed\nstdout:\n{}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );

        let ptx = std::fs::read_to_string(&out).expect("read PTX output");
        assert!(ptx.contains(".entry"), "PTX output missing .entry");
    }
}
