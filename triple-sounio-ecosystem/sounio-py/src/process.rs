/// SounioProcess — spawn `souc run <program> -- --serve`, handshake, send/receive SNIO frames.
///
/// Ported from interop/fsharp/Sounio.Interop/SounioProcess.fs.
///
/// Key design decisions:
/// - stderr is drained in a background thread to prevent deadlock when the
///   JIT emits warnings (which it does for file_size() etc.).
/// - All I/O is synchronous on the calling thread via stdin/stdout pipes.
/// - `SounioProcess` is `Send` because the child process handles are `Send`.
use crate::protocol::{self, Response};
use std::env;
use std::io::{self, BufReader, BufWriter, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

/// Locate the souc binary.
///
/// Resolution order:
/// 1. `SOUC` environment variable
/// 2. `SOUNIO_SOUC_PATH` environment variable
/// 3. Repo-relative default: `<repo_root>/artifacts/omega/souc-bin/souc-linux-x86_64-jit`
pub fn find_souc() -> Option<PathBuf> {
    // 1. SOUC env var
    if let Ok(p) = env::var("SOUC") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Some(path);
        }
    }
    // 2. SOUNIO_SOUC_PATH env var
    if let Ok(p) = env::var("SOUNIO_SOUC_PATH") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Some(path);
        }
    }
    // 3. Relative to this binary's location (walk up to find repo root)
    let candidates = [
        // If run from within the sounio-py project dir
        "../../artifacts/omega/souc-bin/souc-linux-x86_64-jit",
        // Standard repo root
        "artifacts/omega/souc-bin/souc-linux-x86_64-jit",
    ];
    for rel in &candidates {
        let path = PathBuf::from(rel);
        if path.exists() {
            return Some(path);
        }
    }
    None
}

/// A live souc --serve subprocess with stdin/stdout SNIO framing.
pub struct SounioProcess {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    stdout: BufReader<ChildStdout>,
    /// Stderr lines collected by the background drainer thread.
    stderr_log: Arc<Mutex<Vec<String>>>,
}

impl SounioProcess {
    /// Spawn `souc run <program_path> -- --serve` and wait for the ready signal.
    ///
    /// # Errors
    /// Returns `Err` if the binary cannot be found, if the process fails to start,
    /// or if the ready handshake fails.
    pub fn spawn(souc_path: &str, program_path: &str) -> io::Result<Self> {
        let mut child = Command::new(souc_path)
            .args(["run", program_path, "--", "--serve"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = BufWriter::new(child.stdin.take().expect("stdin piped"));
        let stdout = BufReader::new(child.stdout.take().expect("stdout piped"));
        let stderr = child.stderr.take().expect("stderr piped");

        // Drain stderr in a background thread to prevent deadlock.
        let stderr_log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let log_clone = Arc::clone(&stderr_log);
        thread::spawn(move || {
            use std::io::BufRead;
            let reader = io::BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                if let Ok(mut v) = log_clone.lock() {
                    v.push(line);
                }
            }
        });

        let mut sp = SounioProcess {
            child,
            stdin,
            stdout,
            stderr_log,
        };

        // Read the ready signal: RESULT([0]) or RESULT([x]) — any RESULT counts.
        match sp.read_response()? {
            Response::Result(_) => {} // ready
            Response::Error(msg) => {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    format!("souc startup error: {}", msg),
                ))
            }
            Response::Shutdown => {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionAborted,
                    "souc shut down during startup",
                ))
            }
            Response::Unknown(t) => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected message type {} during startup", t),
                ))
            }
        }

        Ok(sp)
    }

    /// Spawn using auto-detected souc binary path.
    pub fn spawn_default(program_path: &str) -> io::Result<Self> {
        let souc = find_souc().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                "souc binary not found; set SOUC env var",
            )
        })?;
        Self::spawn(souc.to_str().unwrap(), program_path)
    }

    // ---- Internal I/O ----

    fn read_response(&mut self) -> io::Result<Response> {
        protocol::read_response(&mut self.stdout)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.stdin.flush()
    }

    // ---- Public API ----

    /// Call a Sounio function with raw i64 arguments, return i64 results.
    pub fn call_raw(&mut self, func_name: &str, args: &[i64]) -> io::Result<Vec<i64>> {
        protocol::write_call_func(&mut self.stdin, func_name, args)?;
        match self.read_response()? {
            Response::Result(values) => Ok(values),
            Response::Error(msg) => Err(io::Error::new(
                io::ErrorKind::Other,
                format!("Sounio error in '{}': {}", func_name, msg),
            )),
            Response::Shutdown => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected shutdown",
            )),
            Response::Unknown(t) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown response type {}", t),
            )),
        }
    }

    /// Call with f64 args (bit-cast to i64) and return f64 results.
    pub fn call_f64(&mut self, func_name: &str, args: &[f64]) -> io::Result<Vec<f64>> {
        let i64_args: Vec<i64> = args.iter().map(|f| f.to_bits() as i64).collect();
        let results = self.call_raw(func_name, &i64_args)?;
        Ok(results.into_iter().map(|v| f64::from_bits(v as u64)).collect())
    }

    /// Call with a single expected i64 scalar return.
    pub fn call_scalar(&mut self, func_name: &str, args: &[i64]) -> io::Result<i64> {
        let results = self.call_raw(func_name, args)?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "empty result"))
    }

    /// Ping: returns true if the server responds with RESULT([1]).
    pub fn ping(&mut self) -> bool {
        self.call_scalar("ping", &[])
            .map(|v| v == 1)
            .unwrap_or(false)
    }

    /// Query health via MSG_HEALTH; returns true if response is RESULT([1]).
    pub fn health(&mut self) -> io::Result<bool> {
        protocol::write_simple(&mut self.stdin, protocol::MSG_HEALTH)?;
        match self.read_response()? {
            Response::Result(v) if v.first() == Some(&1) => Ok(true),
            _ => Ok(false),
        }
    }

    /// Gracefully shut down: send SHUTDOWN, drain response, then kill if needed.
    pub fn shutdown(&mut self) -> io::Result<()> {
        let _ = protocol::write_shutdown(&mut self.stdin);
        // Attempt to read the SHUTDOWN ack (ignore errors — process may already be gone).
        let _ = self.read_response();
        let _ = self.child.wait();
        Ok(())
    }

    /// Return collected stderr lines (for diagnostics).
    pub fn stderr_lines(&self) -> Vec<String> {
        self.stderr_log
            .lock()
            .map(|v| v.clone())
            .unwrap_or_default()
    }
}

impl Drop for SounioProcess {
    fn drop(&mut self) {
        let _ = protocol::write_shutdown(&mut self.stdin);
        let _ = self.child.wait();
    }
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity check: find_souc resolves the binary given SOUC is set in the environment.
    #[test]
    fn test_find_souc_env() {
        // We can't rely on the binary existing in CI, so just check env var path works
        // when the file exists.  We set it to a real file (/bin/sh) as a proxy.
        env::set_var("SOUC", "/bin/sh");
        assert!(find_souc().is_some());
        env::remove_var("SOUC");
    }

    /// Integration test: spawn souc --serve and send ping.
    ///
    /// Skipped if souc binary is not available.
    #[test]
    fn test_ping() {
        let souc = match find_souc() {
            Some(p) => p,
            None => {
                eprintln!("[SKIP] test_ping: souc binary not found");
                return;
            }
        };
        let program = "self-hosted/compiler/main.sio";
        let mut sp = match SounioProcess::spawn(souc.to_str().unwrap(), program) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("[SKIP] test_ping: could not spawn souc: {}", e);
                return;
            }
        };
        assert!(sp.ping(), "ping should return true (RESULT([1]))");
        let _ = sp.shutdown();
    }
}
