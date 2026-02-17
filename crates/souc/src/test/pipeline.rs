//! Internal production-oriented test pipeline for source → native executable.
//!
//! This module centralizes the critical path used by smoke tests and CI gates.
//! It avoids panics in favor of explicit `Err` with operational diagnostics.

use crate::codegen::native::NativeCodegen;
use crate::lexer;
use crate::module_loader;
use crate::parser;
use std::any::Any;
use std::ffi::OsString;
use std::panic::{self, AssertUnwindSafe, UnwindSafe};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

static PIPELINE_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Native backends supported by the internal production test pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NativeBackendChoice {
    /// Rust-native x86-64 Linux production backend.
    Native,
    /// Self-hosted compiler native backend via `souc` self-hosted pipeline.
    SelfhostedNative,
}

impl Default for NativeBackendChoice {
    fn default() -> Self {
        Self::SelfhostedNative
    }
}

impl std::fmt::Display for NativeBackendChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => write!(f, "native"),
            Self::SelfhostedNative => write!(f, "selfhosted-native"),
        }
    }
}

/// Production test pipeline options for native ELF compilation.
#[derive(Debug, Clone)]
pub struct NativeCompileOptions {
    /// Backend used to compile `.sio` source into native ELF bytes.
    pub backend: NativeBackendChoice,
    /// Optional temporary working root.
    pub stdlib_path: Option<PathBuf>,
    /// Enable experimental CPS pipeline (off by default).
    pub enable_cps: bool,
    /// Print extra diagnostics.
    pub verbose: bool,
}

impl Default for NativeCompileOptions {
    fn default() -> Self {
        Self {
            backend: NativeBackendChoice::SelfhostedNative,
            stdlib_path: None,
            enable_cps: false,
            verbose: false,
        }
    }
}

/// Pipeline-local result type.
pub type PipelineResult<T> = Result<T, String>;

struct WorkingDirGuard {
    original: PathBuf,
    should_restore: bool,
}

impl WorkingDirGuard {
    fn enter(path: &Path) -> PipelineResult<Self> {
        let original = std::env::current_dir()
            .map_err(|e| format!("Failed to read working directory: {}", e))?;

        if original == path {
            return Ok(Self {
                original,
                should_restore: false,
            });
        }

        std::env::set_current_dir(path).map_err(|e| {
            format!(
                "Failed to set working directory to {}: {}",
                path.display(),
                e
            )
        })?;

        Ok(Self {
            original,
            should_restore: true,
        })
    }
}

impl Drop for WorkingDirGuard {
    fn drop(&mut self) {
        if self.should_restore {
            let _ = std::env::set_current_dir(&self.original);
        }
    }
}

struct StdlibPathGuard {
    previous: Option<OsString>,
    was_set: bool,
}

impl StdlibPathGuard {
    fn enter(path: Option<&Path>) -> PipelineResult<Self> {
        let previous = std::env::var_os("SOUNIO_STDLIB_PATH");

        if let Some(path) = path {
            // SAFETY: We restore the variable deterministically in Drop.
            unsafe {
                std::env::set_var("SOUNIO_STDLIB_PATH", path);
            }
            return Ok(Self {
                previous,
                was_set: true,
            });
        }

        Ok(Self {
            previous,
            was_set: false,
        })
    }
}

impl Drop for StdlibPathGuard {
    fn drop(&mut self) {
        if self.was_set {
            if let Some(prev) = &self.previous {
                // SAFETY: Caller holds exclusive access to this guard for the
                // current thread's scope.
                unsafe {
                    std::env::set_var("SOUNIO_STDLIB_PATH", prev);
                }
            } else {
                // SAFETY: Caller holds exclusive access to this guard for the
                // current thread's scope.
                unsafe {
                    std::env::remove_var("SOUNIO_STDLIB_PATH");
                }
            }
        }
    }
}

fn panic_payload_to_string(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "panic payload is not a string".to_string()
    }
}

fn catch_pipeline_panic<T, F>(operation: &str, f: F) -> PipelineResult<T>
where
    F: FnOnce() -> PipelineResult<T> + UnwindSafe,
{
    panic::catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|payload| {
        Err(format!(
            "{} failed due to panic: {}",
            operation,
            panic_payload_to_string(payload)
        ))
    })
}

fn make_workdir(root: &Option<PathBuf>) -> PipelineResult<PathBuf> {
    if let Some(dir) = root {
        if !dir.is_dir() {
            return Err(format!(
                "Requested working directory is not a directory: {}",
                dir.display()
            ));
        }
        return Ok(dir.clone());
    }

    Ok(std::env::temp_dir())
}

fn make_temp_path(root: &Path, suffix: &str) -> PipelineResult<PathBuf> {
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|e| format!("Failed to read system time: {}", e))?;
    let pid = process::id();
    let seq = PIPELINE_ID_COUNTER.fetch_add(1, Ordering::Relaxed);

    Ok(root
        .join(format!(
            "sounio_native_pipeline_{pid}_{}_{}",
            ts.as_nanos(),
            seq
        ))
        .with_extension(suffix))
}

/// Compile a Sounio source string to a native ELF executable image.
pub fn compile_source_to_native_elf(
    source: &str,
    options: &NativeCompileOptions,
) -> PipelineResult<Vec<u8>> {
    let workdir = make_workdir(&options.stdlib_path)?;
    let source_path = make_temp_path(&workdir, "sio")?;
    let output_path = make_temp_path(&workdir, "elf")?;

    if options.verbose {
        eprintln!(
            "native pipeline start: backend={} source={} output={}",
            options.backend,
            source_path.display(),
            output_path.display()
        );
    }

    std::fs::write(&source_path, source).map_err(|e| {
        format!(
            "Failed to write temporary source file {}: {}",
            source_path.display(),
            e
        )
    })?;

    let _stdlib_guard = StdlibPathGuard::enter(options.stdlib_path.as_deref())?;

    let bytes = match options.backend {
        NativeBackendChoice::Native => {
            compile_source_to_native_with_rust_pipeline(&source_path, options.enable_cps)?
        }
        NativeBackendChoice::SelfhostedNative => {
            compile_source_to_native_with_selfhost_pipeline(&source_path, options.enable_cps)?
        }
    };

    // Best-effort cleanup for temporary artifacts.
    let _ = std::fs::remove_file(&source_path);
    let _ = std::fs::remove_file(&output_path);

    if bytes.is_empty() {
        return Err("Native compilation produced an empty binary image".to_string());
    }

    Ok(bytes)
}

/// Compile a self-hosted native program from a source file path.
pub fn compile_native_file_to_native_elf(
    source_path: &Path,
    _output_path: &Path,
) -> PipelineResult<Vec<u8>> {
    if !source_path.exists() {
        return Err(format!("Source file not found: {}", source_path.display()));
    }

    let bytes = compile_native_file_to_native_with_rust_pipeline(source_path, false, None)?;

    if bytes.is_empty() {
        return Err("Self-hosted native compiler produced an empty binary image".to_string());
    }

    Ok(bytes)
}

fn compile_source_to_native_with_selfhost_pipeline(
    source_path: &Path,
    enable_cps: bool,
) -> PipelineResult<Vec<u8>> {
    compile_native_file_to_native_with_rust_pipeline(source_path, enable_cps, None)
}

fn compile_source_to_native_with_rust_pipeline(
    source_path: &Path,
    enable_cps: bool,
) -> PipelineResult<Vec<u8>> {
    if enable_cps {
        return Err(
            "CPS transformation is only implemented on the native backend path; use SelfhostedNative or disable --enable-cps."
                .to_string(),
        );
    }

    catch_pipeline_panic("Native pipeline for source string", || {
        let source = std::fs::read_to_string(source_path).map_err(|e| {
            format!(
                "Failed to read temporary source file {}: {}",
                source_path.display(),
                e
            )
        })?;

        let tokens = lexer::lex(&source).map_err(|e| format!("Lexing failed: {}", e))?;
        let ast = parser::parse(&tokens, &source).map_err(|e| format!("Parser error: {}", e))?;
        let hir = crate::check::check_ast(&ast).map_err(|e| format!("Type-check error: {}", e))?;

        let mut codegen = NativeCodegen::new();
        codegen
            .compile(&hir)
            .map_err(|e| format!("Native codegen error: {}", e))
    })
}

fn compile_native_file_to_native_with_rust_pipeline(
    source_path: &Path,
    enable_cps: bool,
    stdlib_path: Option<&Path>,
) -> PipelineResult<Vec<u8>> {
    if enable_cps {
        return Err(
            "CPS transformation is only implemented on the native backend path; use SelfhostedNative or disable --enable-cps."
                .to_string(),
        );
    }

    catch_pipeline_panic("Self-hosted native pipeline for source file", || {
        let source_parent = source_path
            .parent()
            .map(Path::new)
            .map(Path::to_path_buf)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));

        let _stdlib_guard = StdlibPathGuard::enter(stdlib_path)?;
        let _cwd_guard = WorkingDirGuard::enter(&source_parent)?;

        let ast = module_loader::load_program_ast(source_path)
            .map_err(|e| format!("Parser/load error for {}: {}", source_path.display(), e))?;

        let hir = crate::check::check_ast(&ast)
            .map_err(|e| format!("Type-check error for {}: {}", source_path.display(), e))?;

        let mut codegen = NativeCodegen::new();
        codegen
            .compile(&hir)
            .map_err(|e| format!("Native codegen error: {}", e))
    })
}

/// Persist a compiled executable image and apply platform executable permissions.
pub fn emit_native_executable(bytes: &[u8], path: &Path) -> PipelineResult<()> {
    std::fs::write(path, bytes).map_err(|e| {
        format!(
            "Failed to write native executable image to {}: {}",
            path.display(),
            e
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o755))
            .map_err(|e| format!("Failed to mark {} as executable: {}", path.display(), e))?;
    }

    Ok(())
}
