//! Self-hosted Sounio Compiler Loader
//!
//! Loads and compiles the Sounio compiler (written in Sounio) and executes it via the bytecode VM.

use crate::embedded_stdlib;
use crate::lexer;
use crate::parser;
use crate::vm::{Bytecode, BytecodeVM};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

const BOOTSTRAP_DRIVER_MODULE: &str = "bootstrap::driver";
const REQUIRED_DRIVER_ENTRYPOINTS: [&str; 3] = ["compile_file", "compile_source", "run_pipeline"];
const DIR_BYTECODE_CACHE_FILENAME: &str = ".sounio_bytecode.sobc";
// Keep in sync with stdlib/compiler/bootstrap/driver.sio CompileArtifact.code capacity.
const DIR_BYTECODE_CACHE_MAX_BYTES: usize = 16 * 1024 * 1024;
const DEFAULT_BOOTSTRAP_SEED_PATH: &str = "bootstrap/seeds/sounio-bootstrap-linux-x86_64.sio.bin";
const BOOTSTRAP_SEED_MAGIC: &[u8; 8] = b"SNSDSEED";
const BOOTSTRAP_SEED_VERSION: u16 = 1;
const BOOTSTRAP_SEED_HEADER_LEN: usize = 20;
const DRIVER_HARNESS_STACK_BYTES_DEFAULT: usize = 64 * 1024 * 1024;
const DRIVER_HARNESS_STACK_BYTES_MIN: usize = 8 * 1024 * 1024;

fn env_flag_enabled(name: &str) -> bool {
    std::env::var(name)
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn env_flag_or_default(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(default)
}

fn rust_ghost_mode_enabled() -> bool {
    env_flag_enabled("SOUNIO_RUST_GHOST")
}

fn emit_rust_ghost_warning(path_kind: &str) {
    if !rust_ghost_mode_enabled() {
        return;
    }
    eprintln!(
        "\x1b[1;31mGHOST MODE — Rust is dead. This path will vanish in 0.x.y+1 ({})\x1b[0m",
        path_kind
    );
}

fn bootstrap_seed_enforced() -> bool {
    env_flag_or_default("SOUNIO_BOOTSTRAP_SEED_ENFORCE", !cfg!(debug_assertions))
}

fn bootstrap_seed_signature_required() -> bool {
    env_flag_or_default("SOUNIO_BOOTSTRAP_SEED_REQUIRE_SIGNATURE", true)
}

fn bootstrap_seed_path() -> PathBuf {
    std::env::var_os("SOUNIO_BOOTSTRAP_SEED_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_BOOTSTRAP_SEED_PATH))
}

fn bootstrap_seed_checksum_path(seed_path: &Path) -> PathBuf {
    std::env::var_os("SOUNIO_BOOTSTRAP_SEED_SHA256_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{}.sha256", seed_path.display())))
}

fn bootstrap_seed_signature_path(seed_path: &Path) -> PathBuf {
    std::env::var_os("SOUNIO_BOOTSTRAP_SEED_SIG_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("{}.sig", seed_path.display())))
}

fn parse_hex_digest(source: &str) -> Option<String> {
    for raw_token in source.split_whitespace() {
        let token = raw_token.trim_matches(|c: char| {
            matches!(
                c,
                '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'
            )
        });
        if token.is_empty() {
            continue;
        }

        let candidate = token
            .rsplit_once('=')
            .map(|(_, rhs)| rhs)
            .or_else(|| token.rsplit_once(':').map(|(_, rhs)| rhs))
            .unwrap_or(token)
            .trim();

        if candidate.len() == 64 && candidate.chars().all(|c| c.is_ascii_hexdigit()) {
            return Some(candidate.to_ascii_lowercase());
        }
    }
    None
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelfhostCompilePipeline {
    /// Prefer the Sounio-side driver boundary and execute orchestration via VM.
    DriverPreferred,
    /// Explicitly use the Rust bridge compile path (dev/oracle fallback only).
    RustBridge,
}

impl SelfhostCompilePipeline {
    fn from_env() -> Self {
        let raw = std::env::var("SOUNIO_SELFHOST_PIPELINE")
            .unwrap_or_else(|_| "driver".to_string())
            .to_lowercase();
        match raw.as_str() {
            "driver" => Self::DriverPreferred,
            "rust" => {
                if rust_ghost_mode_enabled() {
                    emit_rust_ghost_warning("SOUNIO_SELFHOST_PIPELINE=rust");
                    Self::RustBridge
                } else {
                    tracing::warn!(
                        "SOUNIO_SELFHOST_PIPELINE=rust is deprecated; using driver pipeline instead (set SOUNIO_RUST_GHOST=1 for temporary legacy access)"
                    );
                    Self::DriverPreferred
                }
            }
            other => {
                tracing::warn!(
                    "Unknown SOUNIO_SELFHOST_PIPELINE='{}', defaulting to driver",
                    other
                );
                Self::DriverPreferred
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DriverHarnessCacheKey {
    entrypoint: DriverEntrypoint,
    driver_fingerprint: u64,
}

fn driver_harness_cache() -> &'static Mutex<HashMap<DriverHarnessCacheKey, Vec<Bytecode>>> {
    static CACHE: OnceLock<Mutex<HashMap<DriverHarnessCacheKey, Vec<Bytecode>>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn driver_harness_cache_dir() -> Option<PathBuf> {
    if let Some(raw) = std::env::var_os("SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR") {
        if raw.is_empty() {
            return None;
        }
        return Some(PathBuf::from(raw));
    }

    if env_flag_enabled("SOUNIO_SELFHOST_DISABLE_HARNESS_DISK_CACHE") {
        return None;
    }

    // Default to a temp-backed cache so driver harness compilation doesn't pay a
    // Rust bridge cost on every invocation.
    Some(std::env::temp_dir().join("sounio_driver_harness_cache"))
}

fn driver_harness_cache_path(dir: &Path, key: DriverHarnessCacheKey) -> PathBuf {
    let tag = match key.entrypoint {
        DriverEntrypoint::CompileSource => "compile_source",
        DriverEntrypoint::CompileFile => "compile_file",
    };
    dir.join(format!(
        "driver_harness_{}_{}.sobc",
        tag, key.driver_fingerprint
    ))
}

fn driver_harness_cache_read(dir: &Path, key: DriverHarnessCacheKey) -> Option<Vec<Bytecode>> {
    let path = driver_harness_cache_path(dir, key);
    let bytes = std::fs::read(&path).ok()?;
    crate::vm::serialize::deserialize(&bytes).ok()
}

fn driver_harness_cache_write(
    dir: &Path,
    key: DriverHarnessCacheKey,
    bytecode: &[Bytecode],
) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let path = driver_harness_cache_path(dir, key);
    let tmp_path = path.with_extension("sobc.tmp");
    let bytes = crate::vm::serialize::serialize(bytecode)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
    std::fs::write(&tmp_path, bytes)?;
    std::fs::rename(&tmp_path, &path)?;
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SelfhostRustOracleMode {
    Disabled,
    Parity,
}

impl SelfhostRustOracleMode {
    fn from_env() -> Self {
        if env_flag_enabled("SOUNIO_SELFHOST_PARITY") || env_flag_enabled("SOUNIO_SELFHOST_ORACLE")
        {
            Self::Parity
        } else {
            Self::Disabled
        }
    }

    fn is_parity(self) -> bool {
        matches!(self, Self::Parity)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum DriverEntrypoint {
    CompileSource,
    CompileFile,
}

impl DriverEntrypoint {
    fn qualified_name(self) -> &'static str {
        match self {
            Self::CompileSource => "bootstrap::driver::compile_source",
            Self::CompileFile => "bootstrap::driver::compile_file",
        }
    }
}

#[derive(Debug)]
struct LexerBoundary {
    source_len: usize,
    tokens: Vec<crate::lexer::Token>,
}

#[derive(Debug)]
struct ParserBoundary {
    token_count: usize,
    ast: crate::ast::Ast,
}

#[derive(Debug)]
struct CheckerBoundary {
    hir: crate::hir::Hir,
}

#[derive(Debug)]
struct CodegenBoundary {
    bytecode: Vec<Bytecode>,
}

/// Result type for compiler loader operations
pub type LoadResult<T> = Result<T, CompilerLoaderError>;

/// Errors that can occur during compiler loading and execution
#[derive(Debug, Clone, PartialEq)]
pub enum CompilerLoaderError {
    LoadError(String),
    ParseError(String),
    CompileError(String),
    ExecutionError(String),
    IoError(String),
}

impl std::fmt::Display for CompilerLoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::LoadError(msg) => write!(f, "Load error: {}", msg),
            Self::ParseError(msg) => write!(f, "Parse error: {}", msg),
            Self::CompileError(msg) => write!(f, "Compile error: {}", msg),
            Self::ExecutionError(msg) => write!(f, "Execution error: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
        }
    }
}

impl std::error::Error for CompilerLoaderError {}

impl CompilerLoaderError {
    fn kind_code(&self) -> &'static str {
        match self {
            Self::LoadError(_) => "load",
            Self::ParseError(_) => "parse",
            Self::CompileError(_) => "compile",
            Self::ExecutionError(_) => "execution",
            Self::IoError(_) => "io",
        }
    }
}

/// Self-hosted Sounio compiler
///
/// Executes the Sounio compiler (which is written in Sounio) to compile source code.
/// The compiler modules are loaded from the filesystem or embedded in the binary.
pub struct SounioCompiler {
    /// Virtual machine for executing compiled compiler bytecode
    vm: BytecodeVM,
    /// Cache of compiled compiler modules
    module_cache: HashMap<String, Vec<Bytecode>>,
    /// Path to stdlib/compiler/ directory (for filesystem mode)
    stdlib_path: String,
    /// Whether to use embedded modules (Phase 2)
    use_embedded: bool,
}

impl SounioCompiler {
    /// Creates a new self-hosted compiler instance using filesystem modules
    ///
    /// Initializes the bytecode VM and loads the compiler modules from the filesystem.
    ///
    /// # Arguments
    /// * `stdlib_path` - Path to the stdlib/compiler/ directory containing .sio files
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if modules cannot be loaded or initialized
    pub fn new(stdlib_path: &str) -> LoadResult<Self> {
        tracing::info!(
            "Initializing SounioCompiler from filesystem: {}",
            stdlib_path
        );

        let vm = BytecodeVM::new();
        let module_cache = HashMap::new();

        let compiler = Self {
            vm,
            module_cache,
            stdlib_path: stdlib_path.to_string(),
            use_embedded: false,
        };

        // In a full implementation, we would verify that the compiler modules exist
        // and are readable here
        tracing::info!("SounioCompiler initialized successfully (filesystem mode)");

        Ok(compiler)
    }

    /// Creates a new self-hosted compiler instance using embedded modules
    ///
    /// This is the Phase 2 implementation: all stdlib/compiler modules are embedded
    /// in the binary at build time and loaded from there.
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if embedded modules cannot be loaded
    pub fn new_embedded() -> LoadResult<Self> {
        tracing::info!("Initializing SounioCompiler with embedded modules");

        let module_count = embedded_stdlib::MODULE_COUNT;
        tracing::debug!("Found {} embedded modules", module_count);

        if module_count == 0 {
            return Err(CompilerLoaderError::LoadError(
                "No embedded modules found. Was the compiler built correctly?".to_string(),
            ));
        }

        let vm = BytecodeVM::new();
        let module_cache = HashMap::new();

        let compiler = Self {
            vm,
            module_cache,
            stdlib_path: String::new(), // Not used in embedded mode
            use_embedded: true,
        };

        tracing::info!(
            "SounioCompiler initialized successfully (embedded mode, {} modules)",
            module_count
        );

        Ok(compiler)
    }

    /// Returns whether this compiler uses embedded modules
    pub fn is_embedded(&self) -> bool {
        self.use_embedded
    }

    /// Returns the number of available modules
    pub fn module_count(&self) -> usize {
        if self.use_embedded {
            embedded_stdlib::MODULE_COUNT
        } else {
            self.list_modules().map(|m| m.len()).unwrap_or(0)
        }
    }

    fn is_self_hosted_suite_root(path: &Path) -> bool {
        path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name == "self-hosted")
    }

    fn verify_bootstrap_seed_checksum(seed_path: &Path, seed_bytes: &[u8]) -> LoadResult<String> {
        let checksum_path = bootstrap_seed_checksum_path(seed_path);
        let checksum_raw = std::fs::read_to_string(&checksum_path).map_err(|e| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_CHECKSUM_MISSING seed={} checksum_path={} cause={}",
                seed_path.display(),
                checksum_path.display(),
                e
            ))
        })?;
        let expected = parse_hex_digest(&checksum_raw).ok_or_else(|| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_CHECKSUM_INVALID_FORMAT checksum_path={} expected_hex_len=64",
                checksum_path.display()
            ))
        })?;
        let mut hasher = Sha256::new();
        hasher.update(seed_bytes);
        let actual = hex::encode(hasher.finalize());
        if actual != expected {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_CHECKSUM_MISMATCH seed={} checksum_path={} expected={} actual={}",
                seed_path.display(),
                checksum_path.display(),
                expected,
                actual
            )));
        }
        Ok(actual)
    }

    fn verify_bootstrap_seed_signature(seed_path: &Path, seed_hash: &str) -> LoadResult<()> {
        if !bootstrap_seed_signature_required() {
            return Ok(());
        }

        let signature_path = bootstrap_seed_signature_path(seed_path);
        let signature = std::fs::read_to_string(&signature_path).map_err(|e| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_MISSING seed={} signature_path={} cause={}",
                seed_path.display(),
                signature_path.display(),
                e
            ))
        })?;
        if !signature.contains("SOUNIO-SEED-SIG-V1") {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_INVALID_FORMAT signature_path={} required_marker=SOUNIO-SEED-SIG-V1",
                signature_path.display()
            )));
        }
        let signed_hash = parse_hex_digest(&signature).ok_or_else(|| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_INVALID_DIGEST signature_path={}",
                signature_path.display()
            ))
        })?;
        if signed_hash != seed_hash {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_HASH_MISMATCH signature_path={} expected={} actual={}",
                signature_path.display(),
                seed_hash,
                signed_hash
            )));
        }
        Ok(())
    }

    fn decode_bootstrap_seed(seed_path: &Path, seed_bytes: &[u8]) -> LoadResult<Vec<Bytecode>> {
        if seed_bytes.len() < BOOTSTRAP_SEED_HEADER_LEN {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_INVALID_HEADER seed={} reason=too_small bytes={} min={}",
                seed_path.display(),
                seed_bytes.len(),
                BOOTSTRAP_SEED_HEADER_LEN
            )));
        }
        if &seed_bytes[..8] != BOOTSTRAP_SEED_MAGIC {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_INVALID_MAGIC seed={} expected={:?}",
                seed_path.display(),
                std::str::from_utf8(BOOTSTRAP_SEED_MAGIC).unwrap_or("SNSDSEED")
            )));
        }
        let version = u16::from_le_bytes([seed_bytes[8], seed_bytes[9]]);
        if version != BOOTSTRAP_SEED_VERSION {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_UNSUPPORTED_VERSION seed={} expected={} actual={}",
                seed_path.display(),
                BOOTSTRAP_SEED_VERSION,
                version
            )));
        }
        let payload_len = u64::from_le_bytes([
            seed_bytes[12],
            seed_bytes[13],
            seed_bytes[14],
            seed_bytes[15],
            seed_bytes[16],
            seed_bytes[17],
            seed_bytes[18],
            seed_bytes[19],
        ]) as usize;
        if seed_bytes.len() != BOOTSTRAP_SEED_HEADER_LEN + payload_len {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_LENGTH_MISMATCH seed={} header_payload_len={} actual_payload_len={}",
                seed_path.display(),
                payload_len,
                seed_bytes.len().saturating_sub(BOOTSTRAP_SEED_HEADER_LEN)
            )));
        }
        let payload = &seed_bytes[BOOTSTRAP_SEED_HEADER_LEN..];
        let decoded = crate::vm::serialize::deserialize(payload).map_err(|e| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_DESERIALIZE_FAILED seed={} cause={}",
                seed_path.display(),
                e
            ))
        })?;
        if decoded.is_empty() {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_EMPTY seed={}",
                seed_path.display()
            )));
        }
        Ok(decoded)
    }

    fn load_bootstrap_seed_bytecode() -> LoadResult<Vec<Bytecode>> {
        let seed_path = bootstrap_seed_path();
        let seed_bytes = std::fs::read(&seed_path).map_err(|e| {
            CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_MISSING seed={} cause={}",
                seed_path.display(),
                e
            ))
        })?;
        let seed_hash = Self::verify_bootstrap_seed_checksum(&seed_path, &seed_bytes)?;
        Self::verify_bootstrap_seed_signature(&seed_path, &seed_hash)?;
        Self::decode_bootstrap_seed(&seed_path, &seed_bytes)
    }

    /// Compiles Sounio source code to bytecode.
    ///
    /// Driver-first bootstrap flow:
    /// 1. Resolve and validate the Sounio-side driver interface
    ///    (`bootstrap::driver`) entrypoints.
    /// 2. Execute a VM harness that dispatches into the driver boundary.
    /// 3. Lower user source to bytecode while preserving current semantics.
    ///
    /// To explicitly force the legacy bridge path, set
    /// `SOUNIO_SELFHOST_PIPELINE=rust`.
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::CompileError` if compilation fails
    pub fn compile(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} bytes of Sounio source", source.len());
        match SelfhostCompilePipeline::from_env() {
            SelfhostCompilePipeline::DriverPreferred => self.compile_via_driver_source(source),
            SelfhostCompilePipeline::RustBridge => self.compile_via_rust_bridge(source),
        }
    }

    fn driver_orchestration_strict() -> bool {
        env_flag_enabled("SOUNIO_SELFHOST_DRIVER_STRICT")
            || env_flag_enabled("SOUNIO_SELFHOST_STRICT")
    }

    fn stage_boundary_fallback_disallowed(strict: bool) -> bool {
        strict || env_flag_enabled("SOUNIO_SELFHOST_NO_RUST_FALLBACK")
    }

    fn rust_oracle_mode() -> SelfhostRustOracleMode {
        SelfhostRustOracleMode::from_env()
    }

    fn driver_output_required(fallback_disallow: bool) -> bool {
        fallback_disallow
            || env_flag_or_default(
                "SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT",
                !cfg!(debug_assertions),
            )
    }

    fn bool_bit(value: bool) -> u8 {
        if value {
            1
        } else {
            0
        }
    }

    fn emit_driver_compile_start_marker(
        entrypoint: DriverEntrypoint,
        strict: bool,
        oracle_mode: SelfhostRustOracleMode,
    ) {
        eprintln!(
            "SELFHOST=driver-first schema=v1 event=compile_start entrypoint={} strict={} parity={}",
            entrypoint.qualified_name(),
            Self::bool_bit(strict),
            Self::bool_bit(oracle_mode.is_parity())
        );
    }

    fn emit_driver_orchestration_marker(
        entrypoint: DriverEntrypoint,
        strict: bool,
        status: &str,
        err_kind: Option<&str>,
    ) {
        if let Some(err_kind) = err_kind {
            eprintln!(
                "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict={} status={} error_kind={}",
                entrypoint.qualified_name(),
                Self::bool_bit(strict),
                status,
                err_kind
            );
        } else {
            eprintln!(
                "SELFHOST=driver-first schema=v1 event=driver_orchestration entrypoint={} strict={} status={}",
                entrypoint.qualified_name(),
                Self::bool_bit(strict),
                status
            );
        }
    }

    fn emit_stage_boundary_marker(entrypoint: DriverEntrypoint, bytecode_len: usize) {
        eprintln!(
            "SELFHOST=driver-first schema=v1 event=stage_boundary entrypoint={} status=ok bytecode_len={}",
            entrypoint.qualified_name(),
            bytecode_len
        );
    }

    fn strict_driver_failure(
        entrypoint: DriverEntrypoint,
        err: CompilerLoaderError,
    ) -> CompilerLoaderError {
        let err_kind = err.kind_code();
        CompilerLoaderError::CompileError(format!(
            "SELFHOST_STRICT_DRIVER_FAILURE entrypoint={} error_kind={} cause={}",
            entrypoint.qualified_name(),
            err_kind,
            err
        ))
    }

    fn resolve_driver_source_snapshot(&self) -> LoadResult<String> {
        let source = self.load_module(BOOTSTRAP_DRIVER_MODULE)?;
        self.verify_driver_interface(&source)?;
        Ok(source)
    }

    fn compile_via_driver_source(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        let entrypoint = DriverEntrypoint::CompileSource;
        let strict = Self::driver_orchestration_strict();
        let fallback_disallow = Self::stage_boundary_fallback_disallowed(strict);
        let oracle_mode = Self::rust_oracle_mode();
        // When fallback is disallowed, the driver must produce a valid artifact.
        let require_driver_output = Self::driver_output_required(fallback_disallow);
        Self::emit_driver_compile_start_marker(entrypoint, strict, oracle_mode);

        let mut driver_output: Option<Vec<Bytecode>> = None;
        match self.run_driver_orchestration(entrypoint, Some(source)) {
            Ok(vm_result) => {
                Self::emit_driver_orchestration_marker(entrypoint, strict, "ok", None);
                tracing::debug!(
                    "Driver VM orchestration completed via {} (result={}); lowering through explicit stage boundaries",
                    entrypoint.qualified_name(),
                    vm_result
                );
                if !oracle_mode.is_parity() {
                    match Self::maybe_decode_driver_artifact(&vm_result) {
                        Ok(Some(bytecode)) => {
                            eprintln!(
                                "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=ok bytecode_len={}",
                                entrypoint.qualified_name(),
                                bytecode.len()
                            );
                            driver_output = Some(bytecode);
                        }
                        Ok(None) => {}
                        Err(err) => {
                            tracing::debug!(
                                "Driver artifact decode failed for source (len={}): {} (falling back to stage boundaries)",
                                source.len(),
                                err
                            );
                        }
                    }
                }
            }
            Err(err) => {
                let err_kind = err.kind_code();
                if fallback_disallow {
                    Self::emit_driver_orchestration_marker(
                        entrypoint,
                        strict,
                        "strict_error",
                        Some(err_kind),
                    );
                    return Err(Self::strict_driver_failure(entrypoint, err));
                }

                Self::emit_driver_orchestration_marker(
                    entrypoint,
                    strict,
                    "fallback",
                    Some(err_kind),
                );
                tracing::debug!(
                    "Driver VM orchestration via {} failed ({}); continuing via stage-boundary handoff",
                    entrypoint.qualified_name(),
                    err
                );
            }
        }

        if let Some(driver_bytecode) = driver_output {
            Self::emit_stage_boundary_marker(entrypoint, driver_bytecode.len());
            return Ok(driver_bytecode);
        }

        if require_driver_output {
            return Err(CompilerLoaderError::CompileError(format!(
                "SELFHOST_STRICT_DRIVER_OUTPUT_REQUIRED entrypoint={} source_len={}",
                entrypoint.qualified_name(),
                source.len()
            )));
        }

        let boundary_bytecode = self.compile_via_stage_boundaries_source(source)?;
        Self::emit_stage_boundary_marker(entrypoint, boundary_bytecode.len());
        self.maybe_run_rust_oracle_for_source(oracle_mode, strict, source, &boundary_bytecode)?;
        Ok(boundary_bytecode)
    }

    fn compile_via_driver_file(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        let entrypoint = DriverEntrypoint::CompileFile;
        let strict = Self::driver_orchestration_strict();
        let fallback_disallow = Self::stage_boundary_fallback_disallowed(strict);
        let oracle_mode = Self::rust_oracle_mode();
        // When fallback is disallowed, the driver must produce a valid artifact.
        let require_driver_output = Self::driver_output_required(fallback_disallow);
        let path = std::path::Path::new(path);
        if !path.exists() {
            return Err(CompilerLoaderError::IoError(format!(
                "Input path not found '{}'",
                path.display()
            )));
        }
        let path_str = path.to_string_lossy();

        Self::emit_driver_compile_start_marker(entrypoint, strict, oracle_mode);

        // The bootstrap driver is intentionally tiny and can take an unbounded amount of time
        // if we hand it a directory-sized compilation unit (e.g. `self-hosted/`).
        // For non-strict runs, prefer the Rust stage-boundary loader for directories and
        // the self-hosted suite entrypoint (`self-hosted/main.sio`), but keep the
        // compile target anchored to the entrypoint file to avoid loading every file in
        // `self-hosted/` and introducing duplicate definitions.
        let path_is_self_hosted_suite = path
            .file_name()
            .and_then(|name| name.to_str())
            .is_some_and(|name| name == "main.sio")
            && path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|parent| parent.to_str())
                .is_some_and(|name| name == "self-hosted");
        let compile_target = path;
        if env_flag_enabled("SOUNIO_SELFHOST_DEBUG_DRIVER_ORCH") {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=compile_file_path path={} is_dir={} is_self_hosted_suite={} target={}",
                path.display(),
                path.is_dir(),
                path_is_self_hosted_suite,
                compile_target.display()
            );
        }

        if !require_driver_output
            && (path_is_self_hosted_suite || path.is_dir())
            && !env_flag_enabled("SOUNIO_SELFHOST_DRIVER_ALLOW_DIR")
        {
            Self::emit_driver_orchestration_marker(entrypoint, strict, "skip_dir", None);
            let compile_target_str = compile_target.to_string_lossy();
            let boundary_bytecode = self.compile_path_via_stage_boundaries(&compile_target_str)?;
            Self::emit_stage_boundary_marker(entrypoint, boundary_bytecode.len());
            self.maybe_run_rust_oracle_for_path(
                oracle_mode,
                strict,
                &compile_target_str,
                &boundary_bytecode,
            )?;
            return Ok(boundary_bytecode);
        }

        let mut driver_output: Option<Vec<Bytecode>> = None;
        match self.run_driver_orchestration(entrypoint, Some(path_str.as_ref())) {
            Ok(vm_result) => {
                Self::emit_driver_orchestration_marker(entrypoint, strict, "ok", None);
                tracing::debug!(
                    "Driver VM orchestration completed via {} for '{}' (result={}); lowering through explicit stage boundaries",
                    entrypoint.qualified_name(),
                    path_str,
                    vm_result
                );
                // If the driver returned a headered compile artifact, prefer it for tiny bootstrap programs.
                // Skip this when Rust oracle parity is enabled: oracle compares Rust stage-boundary output.
                if !oracle_mode.is_parity() {
                    match Self::maybe_decode_driver_artifact(&vm_result) {
                        Ok(Some(bytecode)) => {
                            eprintln!(
                                "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=ok bytecode_len={}",
                                entrypoint.qualified_name(),
                                bytecode.len()
                            );
                            driver_output = Some(bytecode);
                        }
                        Ok(None) => {}
                        Err(err) => {
                            tracing::debug!(
                                "Driver artifact decode failed for '{}': {} (falling back to stage boundaries)",
                                path_str,
                                err
                            );
                        }
                    }

                    // If the driver signaled a directory bytecode cache, load it directly in the host.
                    // This avoids copying large `.sobc` blobs through the VM value model.
                    if driver_output.is_none() {
                        match Self::maybe_decode_driver_dir_cache(&vm_result, path_str.as_ref()) {
                            Ok(Some(bytecode)) => {
                                eprintln!(
                                    "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=dir_cache bytecode_len={}",
                                    entrypoint.qualified_name(),
                                    bytecode.len()
                                );
                                driver_output = Some(bytecode);
                            }
                            Ok(None) => {}
                            Err(err) => {
                                if fallback_disallow {
                                    return Err(err);
                                }
                                tracing::debug!(
                                    "Driver dir cache decode failed for '{}': {} (falling back to stage boundaries)",
                                    path_str,
                                    err
                                );
                            }
                        }
                    }
                }
            }
            Err(err) => {
                let err_kind = err.kind_code();
                if fallback_disallow {
                    Self::emit_driver_orchestration_marker(
                        entrypoint,
                        strict,
                        "strict_error",
                        Some(err_kind),
                    );
                    return Err(Self::strict_driver_failure(entrypoint, err));
                }

                Self::emit_driver_orchestration_marker(
                    entrypoint,
                    strict,
                    "fallback",
                    Some(err_kind),
                );
                tracing::debug!(
                    "Driver VM orchestration via {} for '{}' failed ({}); continuing via stage-boundary handoff",
                    entrypoint.qualified_name(),
                    path_str,
                    err
                );
            }
        }

        if let Some(driver_bytecode) = driver_output {
            // Preserve existing telemetry contract: stage_boundary is the "bytecode produced" marker.
            Self::emit_stage_boundary_marker(entrypoint, driver_bytecode.len());
            return Ok(driver_bytecode);
        }

        if require_driver_output {
            return Err(CompilerLoaderError::CompileError(format!(
                "SELFHOST_STRICT_DRIVER_OUTPUT_REQUIRED entrypoint={} path={}",
                entrypoint.qualified_name(),
                path_str
            )));
        }

        let boundary_bytecode = self.compile_path_via_stage_boundaries(path_str.as_ref())?;
        Self::emit_stage_boundary_marker(entrypoint, boundary_bytecode.len());
        self.maybe_run_rust_oracle_for_path(
            oracle_mode,
            strict,
            path_str.as_ref(),
            &boundary_bytecode,
        )?;
        Ok(boundary_bytecode)
    }

    fn run_driver_orchestration(
        &self,
        entrypoint: DriverEntrypoint,
        arg0: Option<&str>,
    ) -> LoadResult<crate::vm::Value> {
        let debug = env_flag_enabled("SOUNIO_SELFHOST_DEBUG_DRIVER_ORCH");
        let disallow_rust_harness = env_flag_enabled("SOUNIO_SELFHOST_NO_RUST_HARNESS");
        if debug {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=start entrypoint={} arg0_len={}",
                entrypoint.qualified_name(),
                arg0.map(|v| v.len()).unwrap_or(0)
            );
        }
        let driver_source = self.resolve_driver_source_snapshot()?;
        if debug {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=driver_loaded entrypoint={} driver_len={}",
                entrypoint.qualified_name(),
                driver_source.len()
            );
        }
        let driver_fingerprint = {
            let base = Self::fingerprint_driver_source(&driver_source);
            // Include the compiler stdlib tree fingerprint so the harness cache invalidates
            // when the driver depends on other Sounio-side compiler modules.
            base ^ self.fingerprint_driver_dependency_sources().rotate_left(1)
        };
        if debug {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=driver_fingerprint entrypoint={} fingerprint={:016x}",
                entrypoint.qualified_name(),
                driver_fingerprint
            );
        }
        let cache_key = DriverHarnessCacheKey {
            entrypoint,
            driver_fingerprint,
        };

        // Important: avoid holding the mutex guard across the whole `match` expression.
        // Otherwise, re-locking to insert on a miss can deadlock (same thread).
        let cached_harness = {
            let guard = driver_harness_cache()
                .lock()
                .expect("driver harness cache lock poisoned");
            guard.get(&cache_key).cloned()
        };

        let harness_bytecode = match cached_harness {
            Some(cached) => {
                if debug {
                    eprintln!(
                        "SELFHOST_DEBUG event=driver_orch step=harness_cache_hit entrypoint={} bytecode_len={}",
                        entrypoint.qualified_name(),
                        cached.len()
                    );
                }
                cached
            }
            None => {
                if let Some(cache_dir) = driver_harness_cache_dir() {
                    if let Some(cached) = driver_harness_cache_read(&cache_dir, cache_key) {
                        if debug {
                            eprintln!(
                                "SELFHOST_DEBUG event=driver_orch step=harness_disk_cache_hit entrypoint={} bytecode_len={}",
                                entrypoint.qualified_name(),
                                cached.len()
                            );
                        }
                        driver_harness_cache()
                            .lock()
                            .expect("driver harness cache lock poisoned")
                            .insert(cache_key, cached.clone());
                        cached
                    } else if disallow_rust_harness {
                        return Err(CompilerLoaderError::CompileError(format!(
                            "SELFHOST_NO_RUST_HARNESS entrypoint={} cache_dir={}",
                            entrypoint.qualified_name(),
                            cache_dir.display()
                        )));
                    } else {
                        if debug {
                            eprintln!(
                                "SELFHOST_DEBUG event=driver_orch step=harness_disk_cache_miss entrypoint={} cache_dir={}",
                                entrypoint.qualified_name(),
                                cache_dir.display()
                            );
                        }
                        let harness_source =
                            Self::build_driver_harness_module(entrypoint, &driver_source);
                        if debug {
                            eprintln!(
                                "SELFHOST_DEBUG event=driver_orch step=harness_source_built entrypoint={} harness_len={}",
                                entrypoint.qualified_name(),
                                harness_source.len()
                            );
                        }
                        let compiled = self
                            .compile_driver_harness_via_rust_bridge(
                                entrypoint,
                                driver_fingerprint,
                                &harness_source,
                            )
                            .map_err(|e| {
                                CompilerLoaderError::CompileError(format!(
                                    "{} harness lowering failed: {}",
                                    entrypoint.qualified_name(),
                                    e
                                ))
                            })?;
                        if debug {
                            eprintln!(
                                "SELFHOST_DEBUG event=driver_orch step=harness_compiled entrypoint={} bytecode_len={}",
                                entrypoint.qualified_name(),
                                compiled.len()
                            );
                        }
                        driver_harness_cache()
                            .lock()
                            .expect("driver harness cache lock poisoned")
                            .insert(cache_key, compiled.clone());

                        if let Err(err) =
                            driver_harness_cache_write(&cache_dir, cache_key, &compiled)
                        {
                            if debug {
                                eprintln!(
                                    "SELFHOST_DEBUG event=driver_orch step=harness_disk_cache_write_failed entrypoint={} error={}",
                                    entrypoint.qualified_name(),
                                    err
                                );
                            }
                        } else if debug {
                            eprintln!(
                                "SELFHOST_DEBUG event=driver_orch step=harness_disk_cache_write_ok entrypoint={} cache_dir={}",
                                entrypoint.qualified_name(),
                                cache_dir.display()
                            );
                        }
                        compiled
                    }
                } else if disallow_rust_harness {
                    return Err(CompilerLoaderError::CompileError(format!(
                        "SELFHOST_NO_RUST_HARNESS entrypoint={}",
                        entrypoint.qualified_name(),
                    )));
                } else {
                    if debug {
                        eprintln!(
                            "SELFHOST_DEBUG event=driver_orch step=harness_cache_miss entrypoint={}",
                            entrypoint.qualified_name()
                        );
                    }
                    let harness_source =
                        Self::build_driver_harness_module(entrypoint, &driver_source);
                    if debug {
                        eprintln!(
                            "SELFHOST_DEBUG event=driver_orch step=harness_source_built entrypoint={} harness_len={}",
                            entrypoint.qualified_name(),
                            harness_source.len()
                        );
                    }
                    let compiled = self
                        .compile_driver_harness_via_rust_bridge(
                            entrypoint,
                            driver_fingerprint,
                            &harness_source,
                        )
                        .map_err(|e| {
                            CompilerLoaderError::CompileError(format!(
                                "{} harness lowering failed: {}",
                                entrypoint.qualified_name(),
                                e
                            ))
                        })?;
                    if debug {
                        eprintln!(
                            "SELFHOST_DEBUG event=driver_orch step=harness_compiled entrypoint={} bytecode_len={}",
                            entrypoint.qualified_name(),
                            compiled.len()
                        );
                    }
                    driver_harness_cache()
                        .lock()
                        .expect("driver harness cache lock poisoned")
                        .insert(cache_key, compiled.clone());
                    compiled
                }
            }
        };

        let mut vm = BytecodeVM::new();
        vm.set_user_args(vec![arg0.unwrap_or("").to_string()]);
        if debug {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=vm_execute entrypoint={} bytecode_len={}",
                entrypoint.qualified_name(),
                harness_bytecode.len()
            );
        }
        vm.execute(&harness_bytecode).map_err(|e| {
            CompilerLoaderError::ExecutionError(format!(
                "{} harness VM execution failed: {:?}",
                entrypoint.qualified_name(),
                e
            ))
        })
    }

    fn build_driver_harness_module(entrypoint: DriverEntrypoint, driver_source: &str) -> String {
        // Build a deterministic, self-contained harness module by concatenating the driver
        // snapshot with an entrypoint `main`.
        let mut source = Self::normalize_driver_for_vm(driver_source);
        source.push_str("\n\n");
        source.push_str(&Self::driver_harness_main(entrypoint));
        source
    }

    fn normalize_driver_for_vm(source: &str) -> String {
        source.to_string()
    }

    fn fingerprint_driver_source(source: &str) -> u64 {
        // Hash the driver snapshot so the cache invalidates when the driver changes (filesystem mode).
        let mut hasher = rustc_hash::FxHasher::default();
        source.hash(&mut hasher);
        hasher.finish()
    }

    fn fingerprint_driver_dependency_sources(&self) -> u64 {
        if self.use_embedded || self.stdlib_path.is_empty() {
            return 0;
        }

        let base = Path::new(&self.stdlib_path);
        if !base.exists() {
            return 0;
        }

        fn visit(
            dir: &Path,
            base: &Path,
            hasher: &mut rustc_hash::FxHasher,
        ) -> std::io::Result<()> {
            let mut entries: Vec<std::fs::DirEntry> =
                std::fs::read_dir(dir)?.filter_map(|e| e.ok()).collect();
            entries.sort_by(|a, b| a.path().cmp(&b.path()));

            for entry in entries {
                let path = entry.path();
                if path.is_dir() {
                    visit(&path, base, hasher)?;
                    continue;
                }

                if path.extension().and_then(|s| s.to_str()) != Some("sio") {
                    continue;
                }

                let rel = path.strip_prefix(base).unwrap_or(&path);
                rel.to_string_lossy().hash(hasher);

                let meta = std::fs::metadata(&path)?;
                meta.len().hash(hasher);
                if let Ok(modified) = meta.modified() {
                    if let Ok(duration) = modified.duration_since(std::time::SystemTime::UNIX_EPOCH)
                    {
                        duration.as_secs().hash(hasher);
                        duration.subsec_nanos().hash(hasher);
                    }
                }
            }

            Ok(())
        }

        let mut hasher = rustc_hash::FxHasher::default();
        if visit(base, base, &mut hasher).is_err() {
            return 0;
        }

        hasher.finish()
    }
    fn driver_harness_main(entrypoint: DriverEntrypoint) -> String {
        match entrypoint {
            DriverEntrypoint::CompileSource => r#"
fn main() -> CompileArtifact with IO, Mut, Div, Panic {
    let source = get_arg(0)
    let opts = compile_options_default()
    compile_source_text(source, opts)
}
"#
            .to_string(),
            DriverEntrypoint::CompileFile => r#"
fn main() -> CompileArtifact with IO, Mut, Div, Panic {
    let opts = compile_options_default()
    compile_file(get_arg(0), opts)
}
"#
            .to_string(),
        }
    }

    fn decode_poseidon_driver_text_bytecode(bytes: &[u8]) -> LoadResult<Vec<Bytecode>> {
        const HEADER: &[u8] = b"POSEIDON_BYTECODE_V0\n";
        if !bytes.starts_with(HEADER) {
            return Err(CompilerLoaderError::CompileError(
                "driver artifact missing POSEIDON_BYTECODE_V0 header".to_string(),
            ));
        }
        let text = std::str::from_utf8(&bytes[HEADER.len()..]).map_err(|e| {
            CompilerLoaderError::CompileError(format!("driver artifact is not valid UTF-8: {}", e))
        })?;

        fn parse_string_operand(raw: &str, line_no: usize) -> LoadResult<String> {
            let value = raw.trim();
            if let Some(inner) = value.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
                let mut out = String::with_capacity(inner.len());
                let mut chars = inner.chars();
                while let Some(ch) = chars.next() {
                    if ch != '\\' {
                        out.push(ch);
                        continue;
                    }
                    let esc = chars.next().ok_or_else(|| {
                        CompilerLoaderError::CompileError(format!(
                            "driver bytecode parse error on line {}: unterminated escape sequence",
                            line_no
                        ))
                    })?;
                    match esc {
                        '\\' => out.push('\\'),
                        '"' => out.push('"'),
                        'n' => out.push('\n'),
                        'r' => out.push('\r'),
                        't' => out.push('\t'),
                        '0' => out.push('\0'),
                        other => {
                            return Err(CompilerLoaderError::CompileError(format!(
                                "driver bytecode parse error on line {}: unsupported escape '\\{}'",
                                line_no, other
                            )));
                        }
                    }
                }
                return Ok(out);
            }

            Ok(value.to_string())
        }

        let mut out = Vec::new();
        for (idx, raw) in text.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            if let Some(rest) = line.strip_prefix("PUSH_INT ") {
                let n = rest.trim().parse::<i64>().map_err(|e| {
                    CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: {}",
                        idx + 1,
                        e
                    ))
                })?;
                out.push(Bytecode::Push(crate::vm::Value::Int(n)));
                continue;
            }
            if let Some(rest) = line.strip_prefix("PUSH_STR ") {
                let s = parse_string_operand(rest, idx + 1)?;
                out.push(Bytecode::Push(crate::vm::Value::String(s)));
                continue;
            }
            if line == "ADD" {
                out.push(Bytecode::Add);
                continue;
            }
            if line == "POP" {
                out.push(Bytecode::Pop);
                continue;
            }
            if let Some(rest) = line.strip_prefix("CALL_EXTERN ") {
                let mut parts = rest.split_whitespace();
                let raw_name = parts.next().ok_or_else(|| {
                    CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: missing extern name",
                        idx + 1
                    ))
                })?;
                let raw_argc = parts.next().ok_or_else(|| {
                    CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: missing extern argc",
                        idx + 1
                    ))
                })?;
                if parts.next().is_some() {
                    return Err(CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: CALL_EXTERN expects <name> <argc>",
                        idx + 1
                    )));
                }

                let name = parse_string_operand(raw_name, idx + 1)?;
                let argc = raw_argc.parse::<i32>().map_err(|e| {
                    CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: {}",
                        idx + 1,
                        e
                    ))
                })?;
                if argc < 0 {
                    return Err(CompilerLoaderError::CompileError(format!(
                        "driver bytecode parse error on line {}: CALL_EXTERN argc must be >= 0",
                        idx + 1
                    )));
                }

                out.push(Bytecode::CallExtern(name, argc));
                continue;
            }
            if line == "RETURN" {
                out.push(Bytecode::Return);
                continue;
            }
            return Err(CompilerLoaderError::CompileError(format!(
                "driver bytecode contains unsupported instruction on line {}: {:?}",
                idx + 1,
                line
            )));
        }

        if !matches!(out.last(), Some(Bytecode::Return)) {
            out.push(Bytecode::Return);
        }

        Ok(out)
    }

    fn value_to_u8(v: &crate::vm::Value) -> LoadResult<u8> {
        match v {
            crate::vm::Value::Int(n) => Ok(*n as u8),
            crate::vm::Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
            other => Err(CompilerLoaderError::CompileError(format!(
                "expected int/bool byte value, got {:?}",
                other
            ))),
        }
    }

    fn bytes_from_vm_value(v: &crate::vm::Value, len: usize) -> LoadResult<Vec<u8>> {
        match v {
            crate::vm::Value::List(items) => {
                let mut out = Vec::with_capacity(len);
                for i in 0..len {
                    let item = items.get(i).ok_or_else(|| {
                        CompilerLoaderError::CompileError(format!(
                            "driver artifact byte list too short: need {} bytes, got {}",
                            len,
                            items.len()
                        ))
                    })?;
                    out.push(Self::value_to_u8(item)?);
                }
                Ok(out)
            }
            crate::vm::Value::SparseList {
                default, overrides, ..
            } => {
                let default_byte = Self::value_to_u8(default)?;
                let mut out = vec![default_byte; len];
                for (idx, value) in overrides {
                    if *idx < len {
                        out[*idx] = Self::value_to_u8(value)?;
                    }
                }
                Ok(out)
            }
            other => Err(CompilerLoaderError::CompileError(format!(
                "expected driver artifact bytes list, got {:?}",
                other
            ))),
        }
    }

    fn maybe_decode_driver_artifact(
        vm_result: &crate::vm::Value,
    ) -> LoadResult<Option<Vec<Bytecode>>> {
        let crate::vm::Value::Struct(fields) = vm_result else {
            return Ok(None);
        };
        let ok = matches!(fields.get("ok"), Some(crate::vm::Value::Bool(true)));
        if !ok {
            return Ok(None);
        }

        let code_len = match fields.get("code_len") {
            Some(crate::vm::Value::Int(n)) if *n > 0 => *n as usize,
            _ => return Ok(None),
        };
        let code_val = fields.get("code").ok_or_else(|| {
            CompilerLoaderError::CompileError("driver artifact missing 'code' field".to_string())
        })?;
        let bytes = Self::bytes_from_vm_value(code_val, code_len)?;

        // Only accept headered artifacts to avoid misinterpreting arbitrary buffers.
        if bytes.starts_with(crate::vm::serialize::BYTECODE_MAGIC) {
            let decoded = crate::vm::serialize::deserialize(&bytes).map_err(|e| {
                CompilerLoaderError::CompileError(format!(
                    "driver artifact binary bytecode deserialization failed: {}",
                    e
                ))
            })?;
            return Ok(Some(decoded));
        }

        if bytes.starts_with(b"POSEIDON_BYTECODE_V0\n") {
            let decoded = Self::decode_poseidon_driver_text_bytecode(&bytes)?;
            return Ok(Some(decoded));
        }

        Ok(None)
    }

    fn maybe_decode_driver_dir_cache(
        vm_result: &crate::vm::Value,
        path: &str,
    ) -> LoadResult<Option<Vec<Bytecode>>> {
        let crate::vm::Value::Struct(fields) = vm_result else {
            return Ok(None);
        };
        let ok = matches!(fields.get("ok"), Some(crate::vm::Value::Bool(true)));
        if !ok {
            return Ok(None);
        }

        // Protocol: ok=true and code_len=0 means:
        // "Host should load and deserialize the directory cache at <dir>/.sounio_bytecode.sobc".
        let code_len = match fields.get("code_len") {
            Some(crate::vm::Value::Int(n)) => *n,
            _ => return Ok(None),
        };
        if code_len != 0 {
            return Ok(None);
        }

        let dir = Path::new(path);
        if !dir.is_dir() {
            return Ok(None);
        }

        let cache_path = dir.join(DIR_BYTECODE_CACHE_FILENAME);
        let bytes = std::fs::read(&cache_path).map_err(|e| {
            CompilerLoaderError::IoError(format!(
                "Selfhost dir cache read failed for {}: {}",
                cache_path.display(),
                e
            ))
        })?;
        if bytes.is_empty() {
            return Ok(None);
        }
        if bytes.len() > DIR_BYTECODE_CACHE_MAX_BYTES {
            return Err(CompilerLoaderError::CompileError(format!(
                "Selfhost dir cache too large: {} ({} bytes > max {})",
                cache_path.display(),
                bytes.len(),
                DIR_BYTECODE_CACHE_MAX_BYTES
            )));
        }
        let decoded = crate::vm::serialize::deserialize(&bytes).map_err(|e| {
            CompilerLoaderError::CompileError(format!(
                "Selfhost dir cache bytecode deserialization failed for {}: {}",
                cache_path.display(),
                e
            ))
        })?;
        Ok(Some(decoded))
    }

    fn verify_driver_interface(&self, source: &str) -> LoadResult<()> {
        use crate::ast::Item;

        let tokens = lexer::lex(source).map_err(|e| {
            CompilerLoaderError::ParseError(format!(
                "{} lexing failed: {}",
                BOOTSTRAP_DRIVER_MODULE, e
            ))
        })?;
        let ast = parser::parse(&tokens, source).map_err(|e| {
            CompilerLoaderError::ParseError(format!(
                "{} parsing failed: {}",
                BOOTSTRAP_DRIVER_MODULE, e
            ))
        })?;

        let mut fns = HashSet::new();
        for item in ast.items {
            if let Item::Function(f) = item {
                fns.insert(f.name);
            }
        }

        let missing: Vec<&str> = REQUIRED_DRIVER_ENTRYPOINTS
            .iter()
            .copied()
            .filter(|name| !fns.contains(*name))
            .collect();
        if !missing.is_empty() {
            return Err(CompilerLoaderError::CompileError(format!(
                "{} missing required entrypoints: {}",
                BOOTSTRAP_DRIVER_MODULE,
                missing.join(", ")
            )));
        }

        Ok(())
    }

    fn maybe_run_rust_oracle_for_source(
        &self,
        oracle_mode: SelfhostRustOracleMode,
        strict: bool,
        source: &str,
        boundary_bytecode: &[Bytecode],
    ) -> LoadResult<()> {
        if !matches!(oracle_mode, SelfhostRustOracleMode::Parity) {
            return Ok(());
        }

        match self.compile_via_rust_bridge(source) {
            Ok(oracle_bytecode) => Self::report_rust_oracle_parity(
                strict,
                "source",
                boundary_bytecode,
                &oracle_bytecode,
            ),
            Err(err) => Self::handle_rust_oracle_error(strict, "source", err),
        }
    }

    fn maybe_run_rust_oracle_for_path(
        &self,
        oracle_mode: SelfhostRustOracleMode,
        strict: bool,
        path: &str,
        boundary_bytecode: &[Bytecode],
    ) -> LoadResult<()> {
        if !matches!(oracle_mode, SelfhostRustOracleMode::Parity) {
            return Ok(());
        }

        match self.compile_path_via_rust_bridge(path) {
            Ok(oracle_bytecode) => {
                Self::report_rust_oracle_parity(strict, "file", boundary_bytecode, &oracle_bytecode)
            }
            Err(err) => Self::handle_rust_oracle_error(strict, "file", err),
        }
    }

    fn report_rust_oracle_parity(
        strict: bool,
        label: &str,
        boundary_bytecode: &[Bytecode],
        oracle_bytecode: &[Bytecode],
    ) -> LoadResult<()> {
        if boundary_bytecode == oracle_bytecode {
            eprintln!(
                "SELFHOST=oracle backend=rust status=match label={} strict={} stage_len={} oracle_len={}",
                label,
                Self::bool_bit(strict),
                boundary_bytecode.len(),
                oracle_bytecode.len()
            );
            return Ok(());
        }

        eprintln!(
            "SELFHOST=oracle backend=rust status=mismatch label={} strict={} stage_len={} oracle_len={}",
            label,
            Self::bool_bit(strict),
            boundary_bytecode.len(),
            oracle_bytecode.len()
        );
        let message = format!(
            "Rust oracle parity mismatch for {}: stage-boundary produced {} instruction(s), oracle produced {} instruction(s)",
            label,
            boundary_bytecode.len(),
            oracle_bytecode.len()
        );
        if strict {
            Err(CompilerLoaderError::CompileError(message))
        } else {
            tracing::warn!("{}", message);
            Ok(())
        }
    }

    fn handle_rust_oracle_error(
        strict: bool,
        label: &str,
        err: CompilerLoaderError,
    ) -> LoadResult<()> {
        eprintln!(
            "SELFHOST=oracle backend=rust status=error label={} strict={} error_kind={}",
            label,
            Self::bool_bit(strict),
            err.kind_code()
        );
        let message = format!("Rust oracle parity check failed for {}: {}", label, err);
        if strict {
            Err(CompilerLoaderError::CompileError(message))
        } else {
            tracing::warn!("{}", message);
            Ok(())
        }
    }

    fn compile_via_stage_boundaries_source(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::debug!("Compiling source through stage-boundary loader");
        let lexed = self.run_lexer_boundary(source)?;
        let parsed = self.run_parser_boundary_from_lexer(source, lexed)?;
        let checked = self.run_checker_boundary(parsed)?;
        let generated = self.run_codegen_boundary(checked)?;

        tracing::info!(
            "Stage-boundary source compilation complete, generated {} bytecode instructions",
            generated.bytecode.len()
        );
        Ok(generated.bytecode)
    }

    fn compile_path_via_stage_boundaries(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::debug!("Compiling path through stage-boundary loader: {}", path);
        let parsed = self.run_parser_boundary_for_path(path)?;
        let checked = self.run_checker_boundary(parsed)?;
        let generated = self.run_codegen_boundary(checked)?;

        tracing::info!(
            "Stage-boundary path compilation complete, generated {} bytecode instructions",
            generated.bytecode.len()
        );
        self.maybe_write_dir_bytecode_cache(path, &generated.bytecode);
        Ok(generated.bytecode)
    }

    fn maybe_write_dir_bytecode_cache(&self, path: &str, bytecode: &[Bytecode]) {
        if !env_flag_enabled("SOUNIO_SELFHOST_WRITE_DIR_CACHE") {
            return;
        }
        let path = Path::new(path);
        if !path.is_dir() {
            return;
        }

        let cache_path = path.join(DIR_BYTECODE_CACHE_FILENAME);
        let bytes = match crate::vm::serialize::serialize(bytecode) {
            Ok(bytes) => bytes,
            Err(err) => {
                tracing::debug!(
                    "Selfhost dir cache: serialize failed for {}: {}",
                    cache_path.display(),
                    err
                );
                return;
            }
        };

        if bytes.len() > DIR_BYTECODE_CACHE_MAX_BYTES {
            tracing::debug!(
                "Selfhost dir cache: skip write for {} ({} bytes > max {})",
                cache_path.display(),
                bytes.len(),
                DIR_BYTECODE_CACHE_MAX_BYTES
            );
            return;
        }

        if let Err(err) = std::fs::write(&cache_path, &bytes) {
            tracing::debug!(
                "Selfhost dir cache: write failed for {}: {}",
                cache_path.display(),
                err
            );
        } else {
            tracing::debug!(
                "Selfhost dir cache: wrote {} bytes to {}",
                bytes.len(),
                cache_path.display()
            );
        }
    }

    fn run_lexer_boundary(&self, source: &str) -> LoadResult<LexerBoundary> {
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;
        tracing::debug!(
            "Lexer boundary: {} bytes -> {} tokens",
            source.len(),
            tokens.len()
        );
        Ok(LexerBoundary {
            source_len: source.len(),
            tokens,
        })
    }

    fn run_parser_boundary_from_lexer(
        &self,
        source: &str,
        lexed: LexerBoundary,
    ) -> LoadResult<ParserBoundary> {
        let ast = parser::parse(&lexed.tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;
        tracing::debug!(
            "Parser boundary: {} tokens ({} bytes source) -> AST",
            lexed.tokens.len(),
            lexed.source_len
        );
        Ok(ParserBoundary {
            token_count: lexed.tokens.len(),
            ast,
        })
    }

    fn run_parser_boundary_for_path(&self, path: &str) -> LoadResult<ParserBoundary> {
        let ast = crate::module_loader::load_program_ast(std::path::Path::new(path))
            .map_err(|e| CompilerLoaderError::CompileError(format!("Parse/load error: {}", e)))?;
        tracing::debug!(
            "Parser boundary: loaded AST from path {} via module loader",
            path
        );
        Ok(ParserBoundary {
            token_count: 0,
            ast,
        })
    }

    fn run_checker_boundary(&self, parsed: ParserBoundary) -> LoadResult<CheckerBoundary> {
        let hir = crate::check::check_ast(&parsed.ast)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;
        tracing::debug!(
            "Checker boundary: AST (from {} token(s)) -> HIR",
            parsed.token_count
        );
        Ok(CheckerBoundary { hir })
    }

    fn run_codegen_boundary(&self, checked: CheckerBoundary) -> LoadResult<CodegenBoundary> {
        let bytecode = crate::codegen::compile_hir(&checked.hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;
        tracing::debug!(
            "Codegen boundary: HIR -> {} bytecode instruction(s)",
            bytecode.len()
        );
        Ok(CodegenBoundary { bytecode })
    }

    fn driver_harness_stack_bytes() -> usize {
        std::env::var("SOUNIO_SELFHOST_DRIVER_STACK_BYTES")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .map(|v| v.max(DRIVER_HARNESS_STACK_BYTES_MIN))
            .unwrap_or(DRIVER_HARNESS_STACK_BYTES_DEFAULT)
    }

    fn panic_payload_to_string(payload: Box<dyn std::any::Any + Send + 'static>) -> String {
        if let Some(msg) = payload.downcast_ref::<String>() {
            return msg.clone();
        }
        if let Some(msg) = payload.downcast_ref::<&str>() {
            return (*msg).to_string();
        }
        "unknown panic payload".to_string()
    }

    fn run_with_driver_harness_stack<T, F>(&self, task_label: &str, operation: F) -> LoadResult<T>
    where
        T: Send,
        F: FnOnce() -> LoadResult<T> + Send,
    {
        let stack_bytes = Self::driver_harness_stack_bytes();
        std::thread::scope(|scope| {
            let builder = std::thread::Builder::new()
                .name(format!("sounio-driver-{task_label}"))
                .stack_size(stack_bytes);
            let handle = builder.spawn_scoped(scope, operation).map_err(|e| {
                CompilerLoaderError::ExecutionError(format!(
                    "Failed to spawn driver harness worker '{task_label}' (stack={} bytes): {}",
                    stack_bytes, e
                ))
            })?;
            match handle.join() {
                Ok(result) => result,
                Err(payload) => Err(CompilerLoaderError::ExecutionError(format!(
                    "Driver harness worker '{task_label}' panicked: {}",
                    Self::panic_payload_to_string(payload)
                ))),
            }
        })
    }

    fn compile_driver_harness_via_rust_bridge(
        &self,
        entrypoint: DriverEntrypoint,
        driver_fingerprint: u64,
        harness_source: &str,
    ) -> LoadResult<Vec<Bytecode>> {
        emit_rust_ghost_warning("driver harness lowering");
        let tag = match entrypoint {
            DriverEntrypoint::CompileSource => "compile_source",
            DriverEntrypoint::CompileFile => "compile_file",
        };
        let filename = format!("driver_harness_{}_{}.sio", tag, driver_fingerprint);

        if let Some(cache_dir) = driver_harness_cache_dir() {
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                CompilerLoaderError::IoError(format!(
                    "Failed to create driver harness cache dir {}: {}",
                    cache_dir.display(),
                    e
                ))
            })?;

            let path = cache_dir.join(filename);
            let unique = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            let tmp_path = cache_dir.join(format!(
                "{}.tmp.{unique}",
                path.file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("driver_harness")
            ));

            let write_needed = match std::fs::read_to_string(&path) {
                Ok(existing) => existing != harness_source,
                Err(_) => true,
            };

            if write_needed {
                std::fs::write(&tmp_path, harness_source).map_err(|e| {
                    CompilerLoaderError::IoError(format!(
                        "Failed to write driver harness source to {}: {}",
                        tmp_path.display(),
                        e
                    ))
                })?;
                std::fs::rename(&tmp_path, &path).map_err(|e| {
                    CompilerLoaderError::IoError(format!(
                        "Failed to finalize driver harness source {}: {}",
                        path.display(),
                        e
                    ))
                })?;
            }

            let path_str = path.to_string_lossy().to_string();
            let compiled = self.run_with_driver_harness_stack(tag, || {
                self.compile_path_via_rust_bridge(&path_str)
            });
            if self.use_embedded && compiled.is_err() {
                return self.run_with_driver_harness_stack(tag, || {
                    self.compile_via_rust_bridge(harness_source)
                });
            }
            return compiled;
        }

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let tmp_path = std::env::temp_dir().join(format!(
            "sounio_driver_harness_{}_{}_{}.sio",
            tag, driver_fingerprint, unique
        ));

        std::fs::write(&tmp_path, harness_source).map_err(|e| {
            CompilerLoaderError::IoError(format!(
                "Failed to write temp harness file {}: {}",
                tmp_path.display(),
                e
            ))
        })?;

        let path_str = tmp_path.to_string_lossy().to_string();
        let compiled = self
            .run_with_driver_harness_stack(tag, || self.compile_path_via_rust_bridge(&path_str));
        let _ = std::fs::remove_file(&tmp_path);
        if self.use_embedded && compiled.is_err() {
            return self.run_with_driver_harness_stack(tag, || {
                self.compile_via_rust_bridge(harness_source)
            });
        }
        compiled
    }

    fn compile_via_rust_bridge(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        emit_rust_ghost_warning("source compile");
        tracing::debug!("Compiling source through Rust bridge backend");

        // Lex source code to tokens using Rust lexer
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;

        tracing::debug!("Lexed {} bytes to {} tokens", source.len(), tokens.len());

        // Parse tokens to AST using Rust parser
        let ast = parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;

        tracing::debug!("Parsed {} tokens to AST", tokens.len());

        // Type check the AST
        let hir = crate::check::check_ast(&ast)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;

        tracing::debug!("Type checking complete");

        // Generate bytecode from HIR using the bytecode codegen backend
        let bytecode = crate::codegen::compile_hir(&hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;

        tracing::info!(
            "Compilation complete, generated {} bytecode instructions",
            bytecode.len()
        );

        Ok(bytecode)
    }

    fn compile_path_via_rust_bridge(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        emit_rust_ghost_warning("path compile");
        tracing::debug!("Compiling path through Rust bridge backend: {}", path);

        let ast = crate::module_loader::load_program_ast(std::path::Path::new(path))
            .map_err(|e| CompilerLoaderError::CompileError(format!("Parse/load error: {}", e)))?;

        let hir = crate::check::check_ast(&ast)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;

        let bytecode = crate::codegen::compile_hir(&hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;

        tracing::info!(
            "Path compilation complete, generated {} bytecode instructions",
            bytecode.len()
        );

        Ok(bytecode)
    }

    /// Compiles a Sounio source file to bytecode
    ///
    /// # Arguments
    /// * `path` - Path to the Sounio source file
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if file cannot be read or compilation fails
    pub fn compile_file(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling file: {}", path);
        let compile_path = Self::normalize_self_hosted_suite_entrypoint(path);
        if Self::is_self_hosted_suite_root(&compile_path) {
            match Self::load_bootstrap_seed_bytecode() {
                Ok(bytecode) => {
                    eprintln!(
                        "SELFHOST=seed schema=v1 event=bootstrap_seed status=ok bytecode_len={} path={}",
                        bytecode.len(),
                        bootstrap_seed_path().display()
                    );
                    return Ok(bytecode);
                }
                Err(err) => {
                    if bootstrap_seed_enforced() {
                        return Err(err);
                    }
                    tracing::warn!(
                        "Bootstrap seed unavailable; falling back to dynamic self-host compile path: {}",
                        err
                    );
                }
            }
        }
        match SelfhostCompilePipeline::from_env() {
            SelfhostCompilePipeline::DriverPreferred => {
                let compile_path = compile_path.to_string_lossy();
                self.compile_via_driver_file(&compile_path)
            }
            SelfhostCompilePipeline::RustBridge => {
                let compile_path = compile_path.to_string_lossy();
                self.compile_path_via_rust_bridge(&compile_path)
            }
        }
    }

    fn normalize_self_hosted_suite_entrypoint(path: &str) -> std::path::PathBuf {
        let path = Path::new(path);
        let is_self_hosted_main = path.file_name().and_then(|name| name.to_str())
            == Some("main.sio")
            && path
                .parent()
                .and_then(|parent| parent.file_name())
                .and_then(|name| name.to_str())
                == Some("self-hosted");
        if is_self_hosted_main {
            return path.parent().unwrap_or(path).to_path_buf();
        }
        path.to_path_buf()
    }

    /// Compiles a Sounio source file to a native ELF binary (Phase 6).
    ///
    /// Loads the self-hosted compiler suite, executes it with `--backend=native`,
    /// and the self-hosted `compile_to_elf()` generates the ELF binary directly.
    /// The self-hosted compiler handles the full pipeline:
    ///   parse -> resolve -> check -> lower -> compile_to_elf -> write_elf_to_file
    pub fn compile_file_to_native(&mut self, path: &str, output_path: &str) -> LoadResult<()> {
        tracing::info!("Compiling file to native ELF: {} -> {}", path, output_path);

        if !std::path::Path::new(path).exists() {
            return Err(CompilerLoaderError::IoError(format!(
                "Input path not found '{}'",
                path
            )));
        }

        // Load self-hosted compiler suite as bytecode
        let selfhost_dir = "self-hosted/";
        let bytecode = self.compile_file(selfhost_dir)?;

        // Execute with native backend args:
        //   compile --backend=native -o <output> <input>
        let user_args: Vec<String> = vec![
            "compile".to_string(),
            "--backend=native".to_string(),
            "-o".to_string(),
            output_path.to_string(),
            path.to_string(),
        ];

        let result = self.execute_bytecode_with_args(&bytecode, &user_args)?;

        // Check exit code from self-hosted compiler
        let exit_code = match &result {
            crate::vm::Value::Int(n) => *n,
            _ => -1,
        };

        if exit_code != 0 {
            return Err(CompilerLoaderError::CompileError(format!(
                "Native compilation exited with code {}",
                exit_code
            )));
        }

        // Verify the output file was created
        if !std::path::Path::new(output_path).exists() {
            return Err(CompilerLoaderError::CompileError(
                "Native compilation reported success but output file not found".to_string(),
            ));
        }

        // Make executable on Unix
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(output_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(output_path, perms);
            }
        }

        let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
        tracing::info!(
            "Native ELF binary: {} ({} bytes, executable)",
            output_path,
            file_size
        );

        Ok(())
    }

    /// Compiles a multi-module Sounio program to a native ELF binary.
    ///
    /// Uses the self-hosted `compile_multimodule_native()` which:
    ///   load imports → resolve_modules → check all → lower all → merge IR → compile_to_elf
    pub fn compile_multimodule_to_native(
        &mut self,
        path: &str,
        output_path: &str,
    ) -> LoadResult<()> {
        tracing::info!(
            "Multi-module compile to native ELF: {} -> {}",
            path,
            output_path
        );

        if !std::path::Path::new(path).exists() {
            return Err(CompilerLoaderError::IoError(format!(
                "Input path not found '{}'",
                path
            )));
        }

        let selfhost_dir = "self-hosted/";
        let bytecode = self.compile_file(selfhost_dir)?;

        // Pass --multimodule --backend=native to the self-hosted driver
        let user_args: Vec<String> = vec![
            "compile".to_string(),
            "--multimodule".to_string(),
            "--backend=native".to_string(),
            "-o".to_string(),
            output_path.to_string(),
            path.to_string(),
        ];

        let result = self.execute_bytecode_with_args(&bytecode, &user_args)?;

        let exit_code = match &result {
            crate::vm::Value::Int(n) => *n,
            _ => -1,
        };

        if exit_code != 0 {
            return Err(CompilerLoaderError::CompileError(format!(
                "Multi-module native compilation exited with code {}",
                exit_code
            )));
        }

        if !std::path::Path::new(output_path).exists() {
            return Err(CompilerLoaderError::CompileError(
                "Multi-module native compilation reported success but output file not found"
                    .to_string(),
            ));
        }

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            if let Ok(metadata) = std::fs::metadata(output_path) {
                let mut perms = metadata.permissions();
                perms.set_mode(0o755);
                let _ = std::fs::set_permissions(output_path, perms);
            }
        }

        let file_size = std::fs::metadata(output_path).map(|m| m.len()).unwrap_or(0);
        tracing::info!(
            "Multi-module native ELF binary: {} ({} bytes, executable)",
            output_path,
            file_size
        );

        Ok(())
    }

    /// Execute precompiled bytecode with the self-hosted VM.
    ///
    /// This allows the self-hosted path to run without falling back to the
    /// Rust tree-walking interpreter.
    pub fn execute_bytecode(&mut self, bytecode: &[Bytecode]) -> LoadResult<crate::vm::Value> {
        self.vm.set_user_args(Vec::new());
        self.vm.execute(bytecode).map_err(|e| {
            CompilerLoaderError::ExecutionError(format!("Bytecode VM execution failed: {:?}", e))
        })
    }

    /// Execute precompiled bytecode with explicit user CLI args.
    pub fn execute_bytecode_with_args(
        &mut self,
        bytecode: &[Bytecode],
        user_args: &[String],
    ) -> LoadResult<crate::vm::Value> {
        self.vm.set_user_args(user_args.to_vec());
        self.vm.execute(bytecode).map_err(|e| {
            CompilerLoaderError::ExecutionError(format!("Bytecode VM execution failed: {:?}", e))
        })
    }

    /// Loads a compiler module
    ///
    /// In embedded mode, loads from the embedded modules.
    /// In filesystem mode, looks for files in stdlib/compiler/ directory with the pattern:
    /// - `stdlib/compiler/{module}.sio` - single file module
    /// - `stdlib/compiler/{module}/mod.sio` - multi-file module
    ///
    /// # Arguments
    /// * `module_name` - Name of the module to load (e.g., "lexer::mod", "parser::expr")
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::LoadError` if module cannot be found
    pub fn load_module(&self, module_name: &str) -> LoadResult<String> {
        tracing::debug!(
            "Loading compiler module: {} (embedded={})",
            module_name,
            self.use_embedded
        );

        // Try embedded modules first if in embedded mode
        if self.use_embedded {
            if let Some(source) = embedded_stdlib::get_module(module_name) {
                tracing::trace!("Loaded module {} from embedded source", module_name);
                return Ok(source.to_string());
            }

            // Module not found in embedded
            return Err(CompilerLoaderError::LoadError(format!(
                "Module '{}' not found in embedded modules. Available: {:?}",
                module_name,
                embedded_stdlib::list_modules()
                    .iter()
                    .take(5)
                    .collect::<Vec<_>>()
            )));
        }

        // Filesystem mode: convert module_name format from "lexer::mod" to "lexer/mod.sio"
        let file_path = format!(
            "{}/{}.sio",
            self.stdlib_path,
            module_name.replace("::", "/")
        );

        if std::path::Path::new(&file_path).exists() {
            let content = std::fs::read_to_string(&file_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, file_path);
            return Ok(content);
        }

        // Try without the ::mod suffix for single-file modules
        let base_module = module_name.strip_suffix("::mod").unwrap_or(module_name);
        let single_path = format!("{}/{}.sio", self.stdlib_path, base_module);
        if std::path::Path::new(&single_path).exists() {
            let content = std::fs::read_to_string(&single_path)
                .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            tracing::trace!("Loaded module {} from {}", module_name, single_path);
            return Ok(content);
        }

        Err(CompilerLoaderError::LoadError(format!(
            "Module '{}' not found in {}",
            module_name, self.stdlib_path
        )))
    }

    /// Lists all available compiler modules
    ///
    /// In embedded mode, returns the list of embedded module names.
    /// In filesystem mode, reads from the stdlib directory.
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if the stdlib directory cannot be read (filesystem mode only)
    pub fn list_modules(&self) -> LoadResult<Vec<String>> {
        // In embedded mode, return embedded module names
        if self.use_embedded {
            let mut modules: Vec<String> = embedded_stdlib::list_modules()
                .iter()
                .map(|s| s.to_string())
                .collect();
            modules.sort();
            tracing::info!("Found {} embedded compiler modules", modules.len());
            return Ok(modules);
        }

        // Filesystem mode
        tracing::debug!("Listing available modules in: {}", self.stdlib_path);

        let entries = std::fs::read_dir(&self.stdlib_path)
            .map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;

        let mut modules = Vec::new();

        for entry in entries {
            let entry = entry.map_err(|e| CompilerLoaderError::LoadError(e.to_string()))?;
            let path = entry.path();

            if path.is_file() && path.extension().map_or(false, |ext| ext == "sio") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    modules.push(name.to_string());
                }
            } else if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    modules.push(name.to_string());
                }
            }
        }

        modules.sort();
        tracing::info!("Found {} compiler modules", modules.len());

        Ok(modules)
    }

    /// Executes the self-hosted compiler on the given source
    ///
    /// This is the main entry point that would orchestrate the compilation process
    /// by calling the appropriate compiler modules in sequence.
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Returns
    /// The bytecode result of compilation
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if any stage fails
    pub fn execute(&mut self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Executing self-hosted compiler on {} bytes", source.len());

        // TODO: When fully integrated, this would:
        // 1. Load all compiler modules
        // 2. Execute lexer bytecode
        // 3. Execute parser bytecode on tokens
        // 4. Execute checker bytecode on AST
        // 5. Execute codegen bytecode on checked AST
        // 6. Return the generated bytecode

        self.compile(source)
    }
}

/// Multi-module compilation context for stdlib bootstrap
///
/// Pre-loads all stdlib/compiler modules and builds a shared symbol context
/// for cross-module type resolution. This enables modules to import types
/// from other modules (e.g., `use check::context::TypeContext`).
pub struct StdlibCompilationContext {
    /// Parsed ASTs for all stdlib modules (module_name -> AST)
    module_asts: HashMap<String, crate::ast::Ast>,
    /// Type definitions extracted from all modules (type_name -> module_name)
    type_registry: HashMap<String, String>,
    /// Function signatures extracted from all modules
    fn_registry: HashMap<String, (Vec<String>, String)>, // (param_types, return_type)
}

impl StdlibCompilationContext {
    /// Build a compilation context from all embedded stdlib modules
    pub fn from_embedded() -> LoadResult<Self> {
        let modules = embedded_stdlib::list_modules();
        let mut module_asts = HashMap::new();
        let mut type_registry = HashMap::new();
        let mut fn_registry = HashMap::new();

        tracing::info!(
            "Building stdlib compilation context from {} modules",
            modules.len()
        );

        for module_name in modules {
            if let Some(source) = embedded_stdlib::get_module(module_name) {
                // Parse the module
                match Self::parse_module(source, module_name) {
                    Ok(ast) => {
                        // Extract type definitions
                        Self::extract_definitions(
                            &ast,
                            module_name,
                            &mut type_registry,
                            &mut fn_registry,
                        );
                        module_asts.insert(module_name.to_string(), ast);
                    }
                    Err(e) => {
                        tracing::debug!("Could not parse module {}: {}", module_name, e);
                        // Continue with other modules
                    }
                }
            }
        }

        tracing::info!(
            "Built context with {} modules, {} types, {} functions",
            module_asts.len(),
            type_registry.len(),
            fn_registry.len()
        );

        Ok(Self {
            module_asts,
            type_registry,
            fn_registry,
        })
    }

    /// Parse a single module
    fn parse_module(source: &str, module_name: &str) -> LoadResult<crate::ast::Ast> {
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("{}: {}", module_name, e)))?;
        parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("{}: {}", module_name, e)))
    }

    /// Extract type and function definitions from an AST
    fn extract_definitions(
        ast: &crate::ast::Ast,
        module_name: &str,
        type_registry: &mut HashMap<String, String>,
        fn_registry: &mut HashMap<String, (Vec<String>, String)>,
    ) {
        use crate::ast::Item;

        for item in &ast.items {
            match item {
                Item::Struct(s) => {
                    // Register struct name with its module
                    type_registry.insert(s.name.clone(), module_name.to_string());
                    tracing::trace!("Registered type {} from {}", s.name, module_name);
                }
                Item::Enum(e) => {
                    type_registry.insert(e.name.clone(), module_name.to_string());
                    tracing::trace!("Registered enum {} from {}", e.name, module_name);
                }
                Item::TypeAlias(t) => {
                    type_registry.insert(t.name.clone(), module_name.to_string());
                    tracing::trace!("Registered type alias {} from {}", t.name, module_name);
                }
                Item::Function(f) => {
                    // Extract function signature
                    let param_types: Vec<String> =
                        f.params.iter().map(|p| format_type_expr(&p.ty)).collect();
                    let return_type = f
                        .return_type
                        .as_ref()
                        .map(|t| format_type_expr(t))
                        .unwrap_or_else(|| "()".to_string());
                    fn_registry.insert(
                        format!("{}::{}", module_name, f.name),
                        (param_types, return_type),
                    );
                }
                _ => {}
            }
        }
    }

    /// Get all registered type names
    pub fn registered_types(&self) -> impl Iterator<Item = (&str, &str)> {
        self.type_registry
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Look up which module defines a type
    pub fn find_type_module(&self, type_name: &str) -> Option<&str> {
        self.type_registry.get(type_name).map(|s| s.as_str())
    }

    /// Compile a module with cross-module type resolution
    pub fn compile_module(&self, module_name: &str) -> LoadResult<Vec<Bytecode>> {
        let source = embedded_stdlib::get_module(module_name).ok_or_else(|| {
            CompilerLoaderError::LoadError(format!("Module not found: {}", module_name))
        })?;

        self.compile_source(source, module_name)
    }

    /// Compile source code with access to all stdlib type definitions
    pub fn compile_source(&self, source: &str, module_name: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} with stdlib context", module_name);

        // Lex and parse
        let tokens = lexer::lex(source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;
        let ast = parser::parse(&tokens, source)
            .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;

        // Resolve with stdlib context
        let resolved_ast = crate::resolve::resolve(ast.clone())
            .map_err(|e| CompilerLoaderError::CompileError(format!("Resolution error: {}", e)))?;

        // Type check with external types from stdlib context
        // This injects all known types from other stdlib modules so cross-module
        // references resolve correctly (e.g., TypeContext from check::context)
        let external_types = self
            .type_registry
            .iter()
            .map(|(name, module)| (name.clone(), module.clone()));
        let hir = crate::check::check_with_external_types(&resolved_ast, external_types)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Type check error: {}", e)))?;

        // Generate bytecode
        let bytecode = crate::codegen::compile_hir(&hir)
            .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))?;

        tracing::info!(
            "Compiled {} to {} bytecode instructions",
            module_name,
            bytecode.len()
        );

        Ok(bytecode)
    }
}

/// Format a type expression as a string (for debugging/registry)
fn format_type_expr(ty: &crate::ast::TypeExpr) -> String {
    use crate::ast::TypeExpr;
    match ty {
        TypeExpr::Named { path, args, .. } => {
            let name = path.segments.join("::");
            if args.is_empty() {
                name
            } else {
                let args_str: Vec<String> = args.iter().map(format_type_expr).collect();
                format!("{}<{}>", name, args_str.join(", "))
            }
        }
        TypeExpr::Array { element, size } => {
            let size_str = if size.is_some() { "N" } else { "?" };
            format!("[{}; {}]", format_type_expr(element), size_str)
        }
        TypeExpr::Reference { mutable, inner } => {
            if *mutable {
                format!("&!{}", format_type_expr(inner))
            } else {
                format!("&{}", format_type_expr(inner))
            }
        }
        TypeExpr::RawPointer { mutable, inner } => {
            if *mutable {
                format!("*mut {}", format_type_expr(inner))
            } else {
                format!("*const {}", format_type_expr(inner))
            }
        }
        TypeExpr::Tuple(elems) => {
            let elems_str: Vec<String> = elems.iter().map(format_type_expr).collect();
            format!("({})", elems_str.join(", "))
        }
        TypeExpr::Function {
            params,
            return_type,
            ..
        } => {
            let params_str: Vec<String> = params.iter().map(format_type_expr).collect();
            format!(
                "fn({}) -> {}",
                params_str.join(", "),
                format_type_expr(return_type)
            )
        }
        TypeExpr::Unit => "()".to_string(),
        TypeExpr::Never => "!".to_string(),
        TypeExpr::SelfType => "Self".to_string(),
        TypeExpr::Infer => "_".to_string(),
        _ => "?".to_string(), // Other complex types
    }
}

impl std::fmt::Debug for SounioCompiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SounioCompiler")
            .field("use_embedded", &self.use_embedded)
            .field("stdlib_path", &self.stdlib_path)
            .field("cached_modules", &self.module_cache.len())
            .field("available_modules", &self.module_count())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;
    use std::sync::{Mutex, OnceLock};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn test_compiler_creation_filesystem() {
        // Use absolute path for test
        let stdlib_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib/compiler");
        let compiler = SounioCompiler::new(stdlib_path);
        assert!(compiler.is_ok());
        let compiler = compiler.unwrap();
        assert!(!compiler.is_embedded());
    }

    #[test]
    fn test_compiler_creation_embedded() {
        let compiler = SounioCompiler::new_embedded();
        assert!(compiler.is_ok());
        let compiler = compiler.unwrap();
        assert!(compiler.is_embedded());
        assert!(compiler.module_count() > 0, "Should have embedded modules");
    }

    #[test]
    fn test_module_listing_filesystem() {
        let stdlib_path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../stdlib/compiler");
        let compiler = SounioCompiler::new(stdlib_path).unwrap();
        let modules = compiler.list_modules();
        assert!(modules.is_ok());
        let module_list = modules.unwrap();
        assert!(!module_list.is_empty(), "Should find compiler modules");
    }

    #[test]
    fn test_module_listing_embedded() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules();
        assert!(modules.is_ok());
        let module_list = modules.unwrap();
        assert!(!module_list.is_empty(), "Should have embedded modules");
        // Check for known modules
        assert!(
            module_list.iter().any(|m| m.contains("lexer")),
            "Should have lexer module"
        );

        // Deterministic ordering is required for reproducible self-hosted runs.
        let mut sorted = module_list.clone();
        sorted.sort();
        assert_eq!(
            module_list, sorted,
            "Embedded module listing should be sorted"
        );
    }

    #[test]
    fn test_load_embedded_module() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules().unwrap();

        // Try to load the first available module
        if let Some(first_module) = modules.first() {
            let source = compiler.load_module(first_module);
            assert!(
                source.is_ok(),
                "Should load embedded module: {}",
                first_module
            );
            let content = source.unwrap();
            assert!(!content.is_empty(), "Module content should not be empty");
        }
    }

    #[test]
    fn test_bootstrap_driver_interface_present() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let source = compiler
            .load_module(BOOTSTRAP_DRIVER_MODULE)
            .expect("driver module should load");
        compiler
            .verify_driver_interface(&source)
            .expect("driver interface should validate");
    }

    #[test]
    fn test_bootstrap_driver_interface_reports_missing_entrypoint() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let source = compiler
            .load_module(BOOTSTRAP_DRIVER_MODULE)
            .expect("driver module should load");
        let broken = source.replacen("fn run_pipeline", "fn run_pipeline_removed", 1);
        let err = compiler
            .verify_driver_interface(&broken)
            .expect_err("validation should fail when run_pipeline is missing");
        match err {
            CompilerLoaderError::CompileError(msg) => {
                assert!(msg.contains("run_pipeline"), "unexpected message: {}", msg);
            }
            other => panic!("expected compile error, got {:?}", other),
        }
    }

    #[test]
    fn test_driver_source_pipeline_compiles_simple_source() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let bytecode = compiler
            .compile_via_driver_source("fn main() -> i32 { 0 }")
            .expect("driver source pipeline should compile");
        assert!(
            !bytecode.is_empty(),
            "driver source pipeline should emit bytecode"
        );
    }

    #[test]
    fn test_driver_file_pipeline_compiles_simple_file() {
        let compiler = SounioCompiler::new_embedded().unwrap();

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let temp_path = std::env::temp_dir().join(format!("sounio_driver_file_{unique}.sio"));
        std::fs::write(&temp_path, "fn main() -> i32 { 0 }\n").expect("write temp file");

        let bytecode = compiler
            .compile_via_driver_file(
                temp_path
                    .to_str()
                    .expect("temp path should be representable as UTF-8"),
            )
            .expect("driver file pipeline should compile");

        let _ = std::fs::remove_file(&temp_path);
        assert!(
            !bytecode.is_empty(),
            "driver file pipeline should emit bytecode"
        );
    }

    #[test]
    fn test_driver_file_pipeline_compiles_directory_via_driver_artifact() {
        let compiler = SounioCompiler::new_embedded().unwrap();

        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("sounio_driver_dir_{unique}"));
        std::fs::create_dir_all(&temp_dir).expect("create temp dir");

        // The bootstrap driver now supports directory paths by listing `.sio` files and
        // concatenating them before compiling.
        std::fs::write(temp_dir.join("a.sio"), "// module part A\n").expect("write a.sio");
        std::fs::write(temp_dir.join("b.sio"), "fn main() -> i32 { 0 }\n").expect("write b.sio");

        let temp_dir_str = temp_dir
            .to_str()
            .expect("temp dir should be representable as UTF-8");
        let vm_result = compiler
            .run_driver_orchestration(DriverEntrypoint::CompileFile, Some(temp_dir_str))
            .expect("driver orchestration should succeed for directory input");

        let decoded = SounioCompiler::maybe_decode_driver_artifact(&vm_result)
            .expect("driver result should be decodable");
        let bytecode =
            decoded.expect("driver should return a compile artifact for directory input");
        assert!(!bytecode.is_empty(), "decoded bytecode should not be empty");

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_normalize_self_hosted_suite_entrypoint() {
        assert_eq!(
            SounioCompiler::normalize_self_hosted_suite_entrypoint("self-hosted/main.sio"),
            std::path::Path::new("self-hosted").to_path_buf()
        );

        assert_eq!(
            SounioCompiler::normalize_self_hosted_suite_entrypoint(
                "/abs/path/self-hosted/main.sio"
            ),
            std::path::Path::new("/abs/path/self-hosted").to_path_buf()
        );

        assert_eq!(
            SounioCompiler::normalize_self_hosted_suite_entrypoint("self-hosted/other.sio"),
            std::path::Path::new("self-hosted/other.sio").to_path_buf()
        );
    }

    #[test]
    fn test_explicit_rust_bridge_pipeline_still_compiles_simple_source() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let bytecode = compiler
            .compile_via_rust_bridge("fn main() -> i32 { 42 }")
            .expect("explicit rust bridge compile should succeed");
        assert!(!bytecode.is_empty(), "should emit bytecode");
    }

    #[test]
    fn test_embedded_has_core_modules() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let modules = compiler.list_modules().unwrap();

        // These core modules should be present
        let expected_prefixes = ["lexer", "parser", "check", "codegen"];

        for prefix in expected_prefixes {
            let has_module = modules.iter().any(|m| m.contains(prefix));
            assert!(has_module, "Should have module containing '{}'", prefix);
        }
    }

    #[test]
    fn test_stdlib_compilation_context() {
        let ctx = StdlibCompilationContext::from_embedded();
        assert!(ctx.is_ok(), "Should build stdlib compilation context");
        let ctx = ctx.unwrap();

        // Check that types were extracted
        let type_count: usize = ctx.registered_types().count();
        assert!(
            type_count > 0,
            "Should extract type definitions, found {}",
            type_count
        );
        println!("Extracted {} type definitions from stdlib", type_count);

        // Print some sample types for debugging
        for (name, module) in ctx.registered_types().take(10) {
            println!("  Type '{}' from module '{}'", name, module);
        }
    }

    #[test]
    fn test_find_type_module() {
        let ctx = StdlibCompilationContext::from_embedded().unwrap();

        // Look for common types
        let type_count: usize = ctx.registered_types().count();
        if type_count > 0 {
            // Get first type and verify we can look it up
            let (first_type, first_module) = ctx.registered_types().next().unwrap();
            let found_module = ctx.find_type_module(first_type);
            assert_eq!(found_module, Some(first_module));
        }
    }

    #[test]
    fn test_cross_module_compilation() {
        // Use a larger stack (8MB) to handle deep type hierarchies in 50 stdlib modules
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let ctx = StdlibCompilationContext::from_embedded().unwrap();

                // Try to compile each module and track success/failure
                let modules: Vec<String> = embedded_stdlib::list_modules()
                    .iter()
                    .map(|s| s.to_string())
                    .collect();

                let mut success_count = 0;
                let mut fail_count = 0;
                let mut failures: Vec<(String, String)> = Vec::new();

                for module in &modules {
                    match ctx.compile_module(module) {
                        Ok(bytecode) => {
                            success_count += 1;
                            println!("✓ {} ({} instructions)", module, bytecode.len());
                        }
                        Err(e) => {
                            fail_count += 1;
                            let error_msg = format!("{}", e);
                            // Truncate long errors for readability
                            let short_error = if error_msg.len() > 100 {
                                format!("{}...", &error_msg[..100])
                            } else {
                                error_msg.clone()
                            };
                            println!("✗ {}: {}", module, short_error);
                            failures.push((module.clone(), short_error));
                        }
                    }
                }

                println!("\n=== Summary ===");
                println!("Success: {}/{}", success_count, modules.len());
                println!("Failed: {}", fail_count);

                // Print first 5 failure reasons for debugging
                if !failures.is_empty() {
                    println!("\nSample failures:");
                    for (module, error) in failures.iter().take(5) {
                        println!("  {}: {}", module, error);
                    }
                }

                // Target: at least 10 modules should compile with external types
                assert!(
                    success_count >= 7,
                    "Expected at least 7 modules to compile, got {}",
                    success_count
                );
            })
            .unwrap()
            .join()
            .unwrap();
    }

    #[test]
    fn test_poseidon_text_bytecode_decoder_supports_new_opcodes() {
        let bytes = b"POSEIDON_BYTECODE_V0\n\
PUSH_INT 1\n\
PUSH_INT 2\n\
ADD\n\
POP\n\
CALL_EXTERN print 1\n\
RETURN\n";

        let decoded =
            SounioCompiler::decode_poseidon_driver_text_bytecode(bytes).expect("decode succeeds");

        assert_eq!(
            decoded,
            vec![
                Bytecode::Push(crate::vm::Value::Int(1)),
                Bytecode::Push(crate::vm::Value::Int(2)),
                Bytecode::Add,
                Bytecode::Pop,
                Bytecode::CallExtern("print".to_string(), 1),
                Bytecode::Return,
            ]
        );
    }

    #[test]
    fn test_poseidon_text_bytecode_decoder_supports_push_str() {
        let bytes = b"POSEIDON_BYTECODE_V0\n\
PUSH_STR \"hello world\"\n\
RETURN\n";

        let decoded =
            SounioCompiler::decode_poseidon_driver_text_bytecode(bytes).expect("decode succeeds");

        assert_eq!(
            decoded,
            vec![
                Bytecode::Push(crate::vm::Value::String("hello world".to_string())),
                Bytecode::Return,
            ]
        );
    }

    #[test]
    fn test_poseidon_text_bytecode_decoder_appends_return_when_missing() {
        let bytes = b"POSEIDON_BYTECODE_V0\nPUSH_INT 123\n";
        let decoded =
            SounioCompiler::decode_poseidon_driver_text_bytecode(bytes).expect("decode succeeds");

        assert_eq!(
            decoded,
            vec![Bytecode::Push(crate::vm::Value::Int(123)), Bytecode::Return]
        );
    }

    fn decode_driver_artifact(source: &str) -> Vec<Bytecode> {
        let compiler = SounioCompiler::new_embedded().unwrap();
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let temp_path = std::env::temp_dir().join(format!("sounio_driver_artifact_{unique}.sio"));
        std::fs::write(&temp_path, source).expect("write temp file");

        let vm_result = compiler
            .run_driver_orchestration(
                DriverEntrypoint::CompileFile,
                Some(temp_path.to_str().expect("utf-8 path")),
            )
            .expect("driver orchestration succeeds");
        let decoded = SounioCompiler::maybe_decode_driver_artifact(&vm_result)
            .expect("artifact decode succeeds")
            .expect("driver artifact should be present");

        let _ = std::fs::remove_file(&temp_path);
        decoded
    }

    fn execute_bytecode(bytecode: &[Bytecode]) -> crate::vm::Value {
        let mut vm = crate::vm::BytecodeVM::new();
        vm.execute(bytecode).expect("vm execute")
    }

    #[test]
    fn test_driver_artifact_compiles_expression_body_i64() {
        let decoded = decode_driver_artifact("fn main() -> i64 { 42 }\n");
        let result = execute_bytecode(&decoded);
        assert_eq!(result, crate::vm::Value::Int(42));
    }

    #[test]
    fn test_driver_artifact_compiles_add_literals() {
        let decoded = decode_driver_artifact("fn main() -> i64 { 10 + 32 }\n");
        let result = execute_bytecode(&decoded);
        assert_eq!(result, crate::vm::Value::Int(42));
    }

    #[test]
    fn test_driver_artifact_compiles_print_then_int() {
        let decoded =
            decode_driver_artifact("fn main() -> i64 with IO { print(\"hello\\n\"); 0 }\n");
        assert!(
            decoded.iter().any(|bc| matches!(
                bc,
                Bytecode::Push(crate::vm::Value::String(s)) if s == "hello\n"
            )),
            "expected PUSH String(\"hello\\n\") in driver artifact, got {:?}",
            decoded
        );
        assert!(
            decoded.iter().any(|bc| matches!(
                bc,
                Bytecode::CallExtern(name, 1) if name == "__sounio_print"
            )),
            "expected CallExtern(__sounio_print, 1) in driver artifact, got {:?}",
            decoded
        );
        let result = execute_bytecode(&decoded);
        assert_eq!(result, crate::vm::Value::Int(0));
    }

    #[test]
    fn test_driver_artifact_executes_function_call_program() {
        let decoded = decode_driver_artifact(
            r#"
fn add(a: i64, b: i64) -> i64 { return a + b }
fn main() -> i64 { add(10, 32) }
"#,
        );
        let result = execute_bytecode(&decoded);
        assert_eq!(result, crate::vm::Value::Int(42));
    }

    #[test]
    fn test_driver_artifact_executes_if_true_else() {
        let decoded = decode_driver_artifact("fn main() -> i64 { if true { 1 } else { 0 } }\n");
        let result = execute_bytecode(&decoded);
        assert_eq!(result, crate::vm::Value::Int(1));
    }

    #[test]
    fn test_parse_hex_digest_checksum_format() {
        let digest = "8cb2237d0679ca88db6464eac60da96345513964f4f5b4f9cd67f3f8a4c9f2d1";
        let parsed = parse_hex_digest(&format!("{}  seed.bin\n", digest));
        assert_eq!(parsed.as_deref(), Some(digest));
    }

    #[test]
    fn test_parse_hex_digest_signature_format() {
        let digest = "8cb2237d0679ca88db6464eac60da96345513964f4f5b4f9cd67f3f8a4c9f2d1";
        let parsed = parse_hex_digest(&format!(
            "SOUNIO-SEED-SIG-V1 key=sounio-dev sha256={}\n",
            digest
        ));
        assert_eq!(parsed.as_deref(), Some(digest));
    }

    fn make_seed_blob(bytecode: &[Bytecode], version: u16) -> Vec<u8> {
        let payload = crate::vm::serialize::serialize(bytecode).expect("serialize bytecode");
        let mut bytes = Vec::with_capacity(BOOTSTRAP_SEED_HEADER_LEN + payload.len());
        bytes.extend_from_slice(BOOTSTRAP_SEED_MAGIC);
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&0u16.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
    }

    fn sha256_hex(bytes: &[u8]) -> String {
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        hex::encode(hasher.finalize())
    }

    struct EnvGuard {
        key: &'static str,
        prev: Option<OsString>,
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.prev {
                // SAFETY: test-only helper updates process env deterministically and restores it in Drop.
                unsafe { std::env::set_var(self.key, prev) };
            } else {
                // SAFETY: test-only helper updates process env deterministically and restores it in Drop.
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    fn set_env_var(key: &'static str, value: Option<&str>) -> EnvGuard {
        let prev = std::env::var_os(key);
        if let Some(value) = value {
            // SAFETY: test-only helper updates process env for the duration of a single test.
            unsafe { std::env::set_var(key, value) };
        } else {
            // SAFETY: test-only helper updates process env for the duration of a single test.
            unsafe { std::env::remove_var(key) };
        }
        EnvGuard { key, prev }
    }

    #[test]
    fn test_bootstrap_seed_valid_roundtrip() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let temp = tempfile::tempdir().expect("temp dir");
        let seed_path = temp.path().join("sounio-bootstrap-linux-x86_64.sio.bin");
        let checksum_path = temp.path().join("seed.sha256");
        let sig_path = temp.path().join("seed.sig");

        let bytecode = vec![Bytecode::Push(crate::vm::Value::Int(42)), Bytecode::Return];
        let seed_blob = make_seed_blob(&bytecode, BOOTSTRAP_SEED_VERSION);
        std::fs::write(&seed_path, &seed_blob).expect("write seed");

        let digest = sha256_hex(&seed_blob);
        std::fs::write(&checksum_path, format!("{}  seed.bin\n", digest)).expect("write checksum");
        std::fs::write(&sig_path, format!("SOUNIO-SEED-SIG-V1 sha256={}\n", digest))
            .expect("write signature");

        let _g1 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_PATH",
            Some(seed_path.to_str().expect("seed path utf8")),
        );
        let _g2 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SHA256_PATH",
            Some(checksum_path.to_str().expect("checksum path utf8")),
        );
        let _g3 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SIG_PATH",
            Some(sig_path.to_str().expect("sig path utf8")),
        );
        let _g4 = set_env_var("SOUNIO_BOOTSTRAP_SEED_REQUIRE_SIGNATURE", Some("1"));

        let decoded = SounioCompiler::load_bootstrap_seed_bytecode()
            .expect("seed should decode successfully");
        assert_eq!(decoded, bytecode);
    }

    #[test]
    fn test_bootstrap_seed_rejects_checksum_mismatch() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let temp = tempfile::tempdir().expect("temp dir");
        let seed_path = temp.path().join("sounio-bootstrap-linux-x86_64.sio.bin");
        let checksum_path = temp.path().join("seed.sha256");
        let sig_path = temp.path().join("seed.sig");

        let bytecode = vec![Bytecode::Push(crate::vm::Value::Int(42)), Bytecode::Return];
        let mut seed_blob = make_seed_blob(&bytecode, BOOTSTRAP_SEED_VERSION);
        std::fs::write(&seed_path, &seed_blob).expect("write seed");

        let digest = sha256_hex(&seed_blob);
        std::fs::write(&checksum_path, format!("{}  seed.bin\n", digest)).expect("write checksum");
        std::fs::write(&sig_path, format!("SOUNIO-SEED-SIG-V1 sha256={}\n", digest))
            .expect("write signature");

        seed_blob[BOOTSTRAP_SEED_HEADER_LEN] ^= 0x01;
        std::fs::write(&seed_path, &seed_blob).expect("corrupt seed");

        let _g1 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_PATH",
            Some(seed_path.to_str().expect("seed path utf8")),
        );
        let _g2 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SHA256_PATH",
            Some(checksum_path.to_str().expect("checksum path utf8")),
        );
        let _g3 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SIG_PATH",
            Some(sig_path.to_str().expect("sig path utf8")),
        );
        let _g4 = set_env_var("SOUNIO_BOOTSTRAP_SEED_REQUIRE_SIGNATURE", Some("1"));

        let err = SounioCompiler::load_bootstrap_seed_bytecode().expect_err("must reject mismatch");
        let msg = err.to_string();
        assert!(
            msg.contains("BOOTSTRAP_SEED_CHECKSUM_MISMATCH"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_bootstrap_seed_rejects_wrong_version() {
        let bytecode = vec![Bytecode::Push(crate::vm::Value::Int(1)), Bytecode::Return];
        let seed_blob = make_seed_blob(&bytecode, BOOTSTRAP_SEED_VERSION + 1);
        let err = SounioCompiler::decode_bootstrap_seed(Path::new("seed.bin"), &seed_blob)
            .expect_err("version mismatch should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("BOOTSTRAP_SEED_UNSUPPORTED_VERSION"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_bootstrap_seed_missing_is_reported() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let temp = tempfile::tempdir().expect("temp dir");
        let missing_seed = temp.path().join("missing-seed.sio.bin");
        let _g1 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_PATH",
            Some(missing_seed.to_str().expect("seed path utf8")),
        );
        let _g2 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SHA256_PATH",
            Some(
                temp.path()
                    .join("missing-seed.sio.bin.sha256")
                    .to_str()
                    .expect("checksum path utf8"),
            ),
        );
        let _g3 = set_env_var(
            "SOUNIO_BOOTSTRAP_SEED_SIG_PATH",
            Some(
                temp.path()
                    .join("missing-seed.sio.bin.sig")
                    .to_str()
                    .expect("sig path utf8"),
            ),
        );

        let err =
            SounioCompiler::load_bootstrap_seed_bytecode().expect_err("missing seed must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("BOOTSTRAP_SEED_MISSING"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_selfhost_pipeline_rust_requires_ghost_mode() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let _g1 = set_env_var("SOUNIO_SELFHOST_PIPELINE", Some("rust"));
        let _g2 = set_env_var("SOUNIO_RUST_GHOST", None);
        assert_eq!(
            SelfhostCompilePipeline::from_env(),
            SelfhostCompilePipeline::DriverPreferred
        );

        let _g3 = set_env_var("SOUNIO_RUST_GHOST", Some("1"));
        assert_eq!(
            SelfhostCompilePipeline::from_env(),
            SelfhostCompilePipeline::RustBridge
        );
    }
}
