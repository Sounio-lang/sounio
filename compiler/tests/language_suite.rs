//! Language test suite runner for repo-level `tests/`.
//!
//! Runs `.sio` fixtures annotated with:
//! - `//@ run-pass`
//! - `//@ compile-fail`
//! - `//@ error-pattern: ...`
//! - `//@ ignore`
//! - `//@ ignore-platform: <linux|macos|windows>`

use std::ffi::OsStr;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TestKind {
    RunPass,
    CompileFail,
}

#[derive(Debug, Default)]
struct Directives {
    kind: Option<TestKind>,
    error_patterns: Vec<String>,
    ignore: bool,
    ignore_platforms: Vec<String>,
    check_only: bool,
    timeout_ms: Option<u64>,
    run_timeout_ms: Option<u64>,
}

impl Directives {
    fn should_ignore(&self) -> bool {
        if self.ignore {
            return true;
        }
        let os = std::env::consts::OS;
        self.ignore_platforms.iter().any(|p| p == os)
    }
}

#[derive(Debug)]
struct RunResult {
    code: i32,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

fn parse_u64_directive_value(rest: &str) -> Option<u64> {
    let value = rest.trim();
    if value.is_empty() {
        return None;
    }
    value.parse::<u64>().ok()
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("compiler crate should live under repo root")
        .to_path_buf()
}

fn discover_sio_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();

    let Ok(read_dir) = std::fs::read_dir(dir) else {
        return files;
    };

    for entry in read_dir.flatten() {
        let path = entry.path();
        if path.is_dir() {
            files.extend(discover_sio_files(&path));
            continue;
        }
        if path.extension().and_then(OsStr::to_str) == Some("sio") {
            files.push(path);
        }
    }

    files.sort();
    files
}

fn parse_directives(source: &str) -> Directives {
    let mut directives = Directives::default();

    for line in source.lines() {
        let trimmed = line.trim_start();

        if let Some(rest) = trimmed.strip_prefix("//@") {
            let rest = rest.trim();
            if rest == "run-pass" {
                directives.kind = Some(TestKind::RunPass);
                continue;
            }
            if rest == "compile-fail" {
                directives.kind = Some(TestKind::CompileFail);
                continue;
            }
            if rest == "check-only" {
                directives.check_only = true;
                continue;
            }
            if rest == "ignore" {
                directives.ignore = true;
                continue;
            }
            if let Some(platform) = rest.strip_prefix("ignore-platform:") {
                directives
                    .ignore_platforms
                    .push(platform.trim().to_string());
                continue;
            }
            if let Some(pat) = rest.strip_prefix("error-pattern:") {
                directives.error_patterns.push(pat.trim().to_string());
                continue;
            }
            if let Some(rest) = rest.strip_prefix("timeout-ms:") {
                directives.timeout_ms = parse_u64_directive_value(rest);
                continue;
            }
            if let Some(rest) = rest.strip_prefix("run-timeout-ms:") {
                directives.run_timeout_ms = parse_u64_directive_value(rest);
                continue;
            }
        }

        // Stop scanning once we hit non-comment code.
        if !trimmed.starts_with("//") && !trimmed.is_empty() {
            break;
        }
    }

    directives
}

fn run_command_with_timeout(mut command: Command, timeout: Duration) -> RunResult {
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let mut child = command.spawn().expect("failed to spawn command");

    let stdout_handle = child
        .stdout
        .take()
        .map(|mut stdout| {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = stdout.read_to_end(&mut buf);
                buf
            })
        })
        .expect("failed to capture stdout");

    let stderr_handle = child
        .stderr
        .take()
        .map(|mut stderr| {
            std::thread::spawn(move || {
                let mut buf = Vec::new();
                let _ = stderr.read_to_end(&mut buf);
                buf
            })
        })
        .expect("failed to capture stderr");

    let start = Instant::now();
    let mut timed_out = false;
    let status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => {}
            Err(e) => panic!("failed to wait on command: {e}"),
        }

        if start.elapsed() >= timeout {
            timed_out = true;
            let _ = child.kill();
            break child
                .wait()
                .expect("failed to wait on killed command");
        }

        std::thread::sleep(Duration::from_millis(10));
    };

    let stdout = stdout_handle.join().unwrap_or_default();
    let stderr = stderr_handle.join().unwrap_or_default();

    RunResult {
        code: status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&stdout).to_string(),
        stderr: String::from_utf8_lossy(&stderr).to_string(),
        timed_out,
    }
}

fn run_souc(args: &[&str], timeout: Duration) -> RunResult {
    let souc = std::env::var_os("CARGO_BIN_EXE_souc")
        .map(PathBuf::from)
        .or_else(|| {
            let candidate = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target").join("debug").join(if cfg!(windows) { "souc.exe" } else { "souc" });
            candidate.exists().then_some(candidate)
        })
        .expect("could not locate `souc` binary (try `cargo build --bin souc`)");

    let mut command = Command::new(&souc);
    command.args(args);
    run_command_with_timeout(command, timeout)
}

fn assert_contains_all(haystack: &str, patterns: &[String]) -> Result<(), String> {
    for pat in patterns {
        if !haystack.contains(pat) {
            return Err(format!("missing error-pattern `{}`", pat));
        }
    }
    Ok(())
}

#[test]
fn language_suite() {
    let root = repo_root();
    let tests_dir = root.join("tests");
    let default_check_timeout = Duration::from_millis(20_000);
    let default_run_timeout = Duration::from_millis(10_000);

    let mut failures: Vec<String> = Vec::new();

    for file in discover_sio_files(&tests_dir) {
        let rel = file
            .strip_prefix(&root)
            .unwrap_or(&file)
            .display()
            .to_string();

        let source = match std::fs::read_to_string(&file) {
            Ok(s) => s,
            Err(e) => {
                failures.push(format!("{rel}: failed to read: {e}"));
                continue;
            }
        };

        let directives = parse_directives(&source);
        if directives.should_ignore() {
            continue;
        }

        let kind = directives.kind.unwrap_or_else(|| {
            if rel.contains("/compile-fail/") || rel.contains("\\compile-fail\\") {
                TestKind::CompileFail
            } else if rel.contains("/ui/") || rel.contains("\\ui\\") {
                TestKind::CompileFail
            } else {
                TestKind::RunPass
            }
        });

        let check_timeout = directives
            .timeout_ms
            .map(Duration::from_millis)
            .unwrap_or(default_check_timeout);
        let run_timeout = directives
            .run_timeout_ms
            .or(directives.timeout_ms)
            .map(Duration::from_millis)
            .unwrap_or(default_run_timeout);

        match kind {
            TestKind::CompileFail => {
                let result = run_souc(
                    &["check", file.to_string_lossy().as_ref(), "--error-format=json"],
                    check_timeout,
                );
                if result.timed_out {
                    failures.push(format!(
                        "{rel}: timed out after {}ms during `check`\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        check_timeout.as_millis(),
                        result.stdout,
                        result.stderr
                    ));
                    continue;
                }
                if result.code == 0 {
                    failures.push(format!("{rel}: expected compile failure, got success"));
                    continue;
                }
                if let Err(msg) = assert_contains_all(&result.stderr, &directives.error_patterns) {
                    failures.push(format!(
                        "{rel}: {msg}\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        result.stdout, result.stderr
                    ));
                }
            }
            TestKind::RunPass => {
                let result = run_souc(
                    &["check", file.to_string_lossy().as_ref(), "--error-format=json"],
                    check_timeout,
                );
                if result.timed_out {
                    failures.push(format!(
                        "{rel}: timed out after {}ms during `check`\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        check_timeout.as_millis(),
                        result.stdout,
                        result.stderr
                    ));
                    continue;
                }
                if result.code != 0 {
                    failures.push(format!(
                        "{rel}: expected check success\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        result.stdout, result.stderr
                    ));
                    continue;
                }

                if directives.check_only {
                    continue;
                }

                let result =
                    run_souc(&["run", file.to_string_lossy().as_ref()], run_timeout);
                if result.timed_out {
                    failures.push(format!(
                        "{rel}: timed out after {}ms during `run`\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        run_timeout.as_millis(),
                        result.stdout,
                        result.stderr
                    ));
                    continue;
                }
                if result.code != 0 {
                    failures.push(format!(
                        "{rel}: expected run success\n--- stdout ---\n{}\n--- stderr ---\n{}",
                        result.stdout, result.stderr
                    ));
                }
            }
        }
    }

    if !failures.is_empty() {
        panic!(
            "language suite failures ({}):\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }
}
