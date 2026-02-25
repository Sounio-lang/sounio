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
const BOOTSTRAP_SEED_SIGNATURE_MARKER: &str = "SOUNIO-SEED-SIG-V1";
const DEFAULT_BOOTSTRAP_SEED_TRUSTED_KEY: &str = "sounio-dev";

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

fn bootstrap_seed_enforced() -> bool {
    env_flag_or_default("SOUNIO_BOOTSTRAP_SEED_ENFORCE", !cfg!(debug_assertions))
}

fn bootstrap_seed_enforced_for_selfhost_root() -> bool {
    bootstrap_seed_enforced()
}

pub fn selfhost_root_seed_enforced() -> bool {
    bootstrap_seed_enforced_for_selfhost_root()
}

pub fn verify_selfhost_bootstrap_seed_policy() -> LoadResult<()> {
    if bootstrap_seed_enforced_for_selfhost_root() {
        SounioCompiler::load_bootstrap_seed_bytecode().map(|_| ())
    } else {
        Err(CompilerLoaderError::CompileError(
            "SELFHOST_BOOTSTRAP_ARTIFACTS_MISSING guidance='Run souc bootstrap init --bundle <dir> --state <dir>'".to_string(),
        ))
    }
}

const REMOVED_SELFHOST_ENV_KNOBS: &[&str] = &[
    "SOUNIO_SELFHOST_PIPELINE",
    "SOUNIO_RUST_GHOST",
    "SOUNIO_SELFHOST_NO_RUST_FALLBACK",
    "SOUNIO_SELFHOST_NO_RUST_HARNESS",
    "SOUNIO_SELFHOST_DRIVER_REQUIRE_OUTPUT",
];

pub fn verify_removed_legacy_env_contracts() -> LoadResult<()> {
    for knob in REMOVED_SELFHOST_ENV_KNOBS {
        if let Ok(value) = std::env::var(knob) {
            return Err(CompilerLoaderError::CompileError(format!(
                "LEGACY_SELFHOST_ENV_REMOVED var={} value={} guidance='Use souc bootstrap init --bundle <dir> --state <dir>, souc bootstrap verify --bundle <dir>, and souc bootstrap cycle --state <dir>'",
                knob, value
            )));
        }
    }
    Ok(())
}

fn bootstrap_seed_signature_required() -> bool {
    env_flag_or_default("SOUNIO_BOOTSTRAP_SEED_REQUIRE_SIGNATURE", true)
}

fn bootstrap_seed_trusted_key() -> String {
    std::env::var("SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY")
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_BOOTSTRAP_SEED_TRUSTED_KEY.to_string())
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

fn trim_signature_token(raw: &str) -> &str {
    raw.trim_matches(|c: char| {
        matches!(
            c,
            '"' | '\'' | ',' | ';' | '(' | ')' | '[' | ']' | '{' | '}'
        )
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SeedSignatureFields {
    key: String,
    sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SeedSignatureParseError {
    InvalidFormat,
    MissingKey,
    InvalidDigest,
}

fn parse_seed_signature_fields(
    signature: &str,
) -> Result<SeedSignatureFields, SeedSignatureParseError> {
    let mut tokens = signature
        .split_whitespace()
        .map(trim_signature_token)
        .filter(|token| !token.is_empty());

    let marker = tokens
        .next()
        .ok_or(SeedSignatureParseError::InvalidFormat)?;
    if marker != BOOTSTRAP_SEED_SIGNATURE_MARKER {
        return Err(SeedSignatureParseError::InvalidFormat);
    }

    let mut key = None::<String>;
    let mut sha256 = None::<String>;

    for token in tokens {
        if let Some((field, value)) = token.split_once('=') {
            match field {
                "key" => {
                    if value.trim().is_empty() {
                        return Err(SeedSignatureParseError::MissingKey);
                    }
                    if key.is_none() {
                        key = Some(value.trim().to_string());
                    }
                }
                "sha256" => {
                    let value = value.trim();
                    if value.len() != 64 || !value.chars().all(|c| c.is_ascii_hexdigit()) {
                        return Err(SeedSignatureParseError::InvalidDigest);
                    }
                    if sha256.is_none() {
                        sha256 = Some(value.to_ascii_lowercase());
                    }
                }
                _ => {}
            }
        }
    }

    let key = key.ok_or(SeedSignatureParseError::MissingKey)?;
    let sha256 = sha256.ok_or(SeedSignatureParseError::InvalidDigest)?;
    Ok(SeedSignatureFields { key, sha256 })
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
        let parsed_signature = parse_seed_signature_fields(&signature).map_err(|err| match err {
            SeedSignatureParseError::InvalidFormat => CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_INVALID_FORMAT signature_path={} required_marker={} required_fields=key,sha256",
                signature_path.display(),
                BOOTSTRAP_SEED_SIGNATURE_MARKER
            )),
            SeedSignatureParseError::MissingKey => CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_MISSING_KEY signature_path={} required_field=key",
                signature_path.display()
            )),
            SeedSignatureParseError::InvalidDigest => CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_INVALID_DIGEST signature_path={}",
                signature_path.display()
            )),
        })?;
        let trusted_key = bootstrap_seed_trusted_key();
        if parsed_signature.key != trusted_key {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_UNTRUSTED_KEY signature_path={} expected={} actual={}",
                signature_path.display(),
                trusted_key,
                parsed_signature.key
            )));
        }
        if parsed_signature.sha256 != seed_hash {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_SIGNATURE_HASH_MISMATCH signature_path={} expected={} actual={}",
                signature_path.display(),
                seed_hash,
                parsed_signature.sha256
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
        let reserved = u16::from_le_bytes([seed_bytes[10], seed_bytes[11]]);
        if reserved != 0 {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_INVALID_RESERVED seed={} expected=0 actual={}",
                seed_path.display(),
                reserved
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
        if payload_len > DIR_BYTECODE_CACHE_MAX_BYTES {
            return Err(CompilerLoaderError::CompileError(format!(
                "BOOTSTRAP_SEED_PAYLOAD_TOO_LARGE seed={} payload_len={} max={}",
                seed_path.display(),
                payload_len,
                DIR_BYTECODE_CACHE_MAX_BYTES
            )));
        }
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
    /// 3. Require a valid driver compile artifact and fail closed otherwise.
    ///
    /// # Arguments
    /// * `source` - The Sounio source code to compile
    ///
    /// # Errors
    /// Returns `CompilerLoaderError::CompileError` if compilation fails
    pub fn compile(&self, source: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling {} bytes of Sounio source", source.len());
        verify_removed_legacy_env_contracts()?;
        self.compile_via_driver_source(source)
    }

    fn driver_orchestration_strict() -> bool {
        true
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
    ) {
        eprintln!(
            "SELFHOST=driver-first schema=v1 event=compile_start entrypoint={} strict={} parity={}",
            entrypoint.qualified_name(),
            Self::bool_bit(strict),
            0
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
        Self::emit_driver_compile_start_marker(entrypoint, strict);

        let vm_result = match self.run_driver_orchestration(entrypoint, Some(source)) {
            Ok(vm_result) => {
                Self::emit_driver_orchestration_marker(entrypoint, strict, "ok", None);
                vm_result
            }
            Err(err) => {
                let err_kind = err.kind_code();
                Self::emit_driver_orchestration_marker(
                    entrypoint,
                    strict,
                    "strict_error",
                    Some(err_kind),
                );
                return Err(Self::strict_driver_failure(entrypoint, err));
            }
        };

        let driver_bytecode = Self::maybe_decode_driver_artifact(&vm_result)?
            .ok_or_else(|| {
                CompilerLoaderError::CompileError(format!(
                    "SELFHOST_STRICT_DRIVER_OUTPUT_REQUIRED entrypoint={} source_len={}",
                    entrypoint.qualified_name(),
                    source.len()
                ))
            })?;
        eprintln!(
            "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=ok bytecode_len={}",
            entrypoint.qualified_name(),
            driver_bytecode.len()
        );
        Self::emit_stage_boundary_marker(entrypoint, driver_bytecode.len());
        Ok(driver_bytecode)
    }

    fn compile_via_driver_file(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        let entrypoint = DriverEntrypoint::CompileFile;
        let strict = Self::driver_orchestration_strict();
        let path = std::path::Path::new(path);
        if !path.exists() {
            return Err(CompilerLoaderError::IoError(format!(
                "Input path not found '{}'",
                path.display()
            )));
        }
        let path_str = path.to_string_lossy();

        Self::emit_driver_compile_start_marker(entrypoint, strict);
        if env_flag_enabled("SOUNIO_SELFHOST_DEBUG_DRIVER_ORCH") {
            eprintln!(
                "SELFHOST_DEBUG event=driver_orch step=compile_file_path path={} is_dir={} is_self_hosted_suite={} target={}",
                path.display(),
                path.is_dir(),
                false,
                path.display()
            );
        }

        let vm_result = match self.run_driver_orchestration(entrypoint, Some(path_str.as_ref())) {
            Ok(vm_result) => {
                Self::emit_driver_orchestration_marker(entrypoint, strict, "ok", None);
                vm_result
            }
            Err(err) => {
                let err_kind = err.kind_code();
                Self::emit_driver_orchestration_marker(
                    entrypoint,
                    strict,
                    "strict_error",
                    Some(err_kind),
                );
                return Err(Self::strict_driver_failure(entrypoint, err));
            }
        };

        let driver_output = match Self::maybe_decode_driver_artifact(&vm_result)? {
            Some(bytecode) => {
                eprintln!(
                    "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=ok bytecode_len={}",
                    entrypoint.qualified_name(),
                    bytecode.len()
                );
                Some(bytecode)
            }
            None => match Self::maybe_decode_driver_dir_cache(&vm_result, path_str.as_ref())? {
                Some(bytecode) => {
                    eprintln!(
                        "SELFHOST=driver-first schema=v1 event=driver_output entrypoint={} status=dir_cache bytecode_len={}",
                        entrypoint.qualified_name(),
                        bytecode.len()
                    );
                    Some(bytecode)
                }
                None => None,
            },
        };

        let driver_bytecode = driver_output.ok_or_else(|| {
            CompilerLoaderError::CompileError(format!(
                "SELFHOST_STRICT_DRIVER_OUTPUT_REQUIRED entrypoint={} path={}",
                entrypoint.qualified_name(),
                path_str
            ))
        })?;
        Self::emit_stage_boundary_marker(entrypoint, driver_bytecode.len());
        Ok(driver_bytecode)
    }

    fn run_driver_orchestration(
        &self,
        entrypoint: DriverEntrypoint,
        arg0: Option<&str>,
    ) -> LoadResult<crate::vm::Value> {
        let debug = env_flag_enabled("SOUNIO_SELFHOST_DEBUG_DRIVER_ORCH");
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
                    } else {
                        return Err(CompilerLoaderError::CompileError(format!(
                            "SELFHOST_DRIVER_HARNESS_UNAVAILABLE entrypoint={} cache_dir={}",
                            entrypoint.qualified_name(),
                            cache_dir.display()
                        )));
                    }
                } else {
                    return Err(CompilerLoaderError::CompileError(format!(
                        "SELFHOST_DRIVER_HARNESS_UNAVAILABLE entrypoint={} cache=disabled",
                        entrypoint.qualified_name(),
                    )));
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

    /// Compiles a Sounio source file to bytecode
    ///
    /// # Arguments
    /// * `path` - Path to the Sounio source file
    ///
    /// # Errors
    /// Returns `CompilerLoaderError` if file cannot be read or compilation fails
    pub fn compile_file(&self, path: &str) -> LoadResult<Vec<Bytecode>> {
        tracing::info!("Compiling file: {}", path);
        verify_removed_legacy_env_contracts()?;
        let compile_path = Self::normalize_self_hosted_suite_entrypoint(path);
        if Self::is_self_hosted_suite_root(&compile_path) {
            let seed_required = bootstrap_seed_enforced_for_selfhost_root();
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
                    if seed_required {
                        return Err(err);
                    }
                    tracing::warn!(
                        "Bootstrap seed unavailable; falling back to dynamic self-host compile path: {}",
                        err
                    );
                }
            }
        }
        let compile_path = compile_path.to_string_lossy();
        self.compile_via_driver_file(&compile_path)
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

    fn prime_driver_harness_cache_for_test(
        compiler: &SounioCompiler,
        entrypoint: DriverEntrypoint,
    ) {
        let driver_source = compiler
            .resolve_driver_source_snapshot()
            .expect("driver source snapshot should resolve");
        let driver_fingerprint = {
            let base = SounioCompiler::fingerprint_driver_source(&driver_source);
            base ^ compiler.fingerprint_driver_dependency_sources().rotate_left(1)
        };
        let harness_source = SounioCompiler::build_driver_harness_module(entrypoint, &driver_source);
        let harness_bytecode =
            compile_harness_source_for_test(&harness_source).expect("driver harness should compile for test priming");
        driver_harness_cache()
            .lock()
            .expect("driver harness cache lock poisoned")
            .insert(
                DriverHarnessCacheKey {
                    entrypoint,
                    driver_fingerprint,
                },
                harness_bytecode,
            );
    }

    fn compile_harness_source_for_test(source: &str) -> LoadResult<Vec<Bytecode>> {
        let source_owned = source.to_string();
        let handle = std::thread::Builder::new()
            .name("sounio-test-harness-compile".to_string())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let tokens = lexer::lex(&source_owned)
                    .map_err(|e| CompilerLoaderError::ParseError(format!("Lexer error: {}", e)))?;
                let ast = parser::parse(&tokens, &source_owned)
                    .map_err(|e| CompilerLoaderError::ParseError(format!("Parser error: {}", e)))?;
                let hir = crate::check::check_ast(&ast).map_err(|e| {
                    CompilerLoaderError::CompileError(format!("Type check error: {}", e))
                })?;
                crate::codegen::compile_hir(&hir)
                    .map_err(|e| CompilerLoaderError::CompileError(format!("Codegen error: {}", e)))
            })
            .map_err(|e| {
                CompilerLoaderError::ExecutionError(format!(
                    "Failed to spawn harness compile worker: {}",
                    e
                ))
            })?;

        handle.join().map_err(|_| {
            CompilerLoaderError::ExecutionError(
                "Harness compile worker panicked while compiling test harness".to_string(),
            )
        })?
    }

    fn new_filesystem_compiler_with_embedded_driver(root: &std::path::Path) -> SounioCompiler {
        let embedded = SounioCompiler::new_embedded().expect("embedded compiler");
        let driver_source = embedded
            .load_module(BOOTSTRAP_DRIVER_MODULE)
            .expect("embedded driver module should load");
        let driver_path = root.join("bootstrap").join("driver.sio");
        std::fs::create_dir_all(
            driver_path
                .parent()
                .expect("driver path should have parent directory"),
        )
        .expect("create bootstrap directory");
        std::fs::write(&driver_path, driver_source).expect("write bootstrap::driver source");
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let nonce_path = root.join("bootstrap").join(format!("nonce_{nonce}.sio"));
        std::fs::write(&nonce_path, format!("// nonce {nonce}\n"))
            .expect("write nonce source file");
        SounioCompiler::new(root.to_str().expect("filesystem stdlib root should be utf-8"))
            .expect("filesystem compiler should initialize")
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
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileSource);
        let bytecode = compiler
            .compile_via_driver_source("fn main() -> i32 { 0 }")
            .expect("driver source pipeline should compile");
        assert!(
            !bytecode.is_empty(),
            "driver source pipeline should emit bytecode"
        );
    }

    #[test]
    fn test_driver_source_pipeline_fails_closed_when_driver_unavailable() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");

        let temp = tempfile::tempdir().expect("temp dir");
        let missing_stdlib = temp.path().join("missing-stdlib");
        let compiler = SounioCompiler::new(
            missing_stdlib
                .to_str()
                .expect("missing stdlib path should be representable as UTF-8"),
        )
        .expect("compiler init should succeed even when stdlib path is missing");

        let err = compiler
            .compile_via_driver_source("fn main() -> i32 { 0 }\n")
            .expect_err("driver-first path should fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
            "expected strict failure token, got: {msg}"
        );
        assert!(
            msg.contains("entrypoint=bootstrap::driver::compile_source"),
            "expected compile_source entrypoint metadata, got: {msg}"
        );
        assert!(
            msg.contains("error_kind=load"),
            "expected deterministic load error kind, got: {msg}"
        );
    }

    #[test]
    fn test_driver_file_pipeline_compiles_simple_file() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileFile);

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
    fn test_driver_file_pipeline_fails_closed_when_driver_unavailable() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");

        let temp = tempfile::tempdir().expect("temp dir");
        let missing_stdlib = temp.path().join("missing-stdlib");
        let compiler = SounioCompiler::new(
            missing_stdlib
                .to_str()
                .expect("missing stdlib path should be representable as UTF-8"),
        )
        .expect("compiler init should succeed even when stdlib path is missing");

        let source = temp.path().join("sample.sio");
        std::fs::write(&source, "fn main() -> i32 { 0 }\n").expect("write temp source");
        let err = compiler
            .compile_via_driver_file(
                source
                    .to_str()
                    .expect("temp source path should be representable as UTF-8"),
            )
            .expect_err("driver-first path should fail closed");
        let msg = err.to_string();
        assert!(
            msg.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
            "expected strict failure token, got: {msg}"
        );
        assert!(
            msg.contains("entrypoint=bootstrap::driver::compile_file"),
            "expected compile_file entrypoint metadata, got: {msg}"
        );
        assert!(
            msg.contains("error_kind=load"),
            "expected deterministic load error kind, got: {msg}"
        );
    }

    #[test]
    fn test_driver_file_pipeline_compiles_directory_via_driver_artifact() {
        let compiler = SounioCompiler::new_embedded().unwrap();
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileFile);

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
    fn test_harness_compile_helper_compiles_simple_source() {
        let bytecode = compile_harness_source_for_test("fn main() -> i32 { 42 }")
            .expect("harness compile helper should succeed");
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
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileFile);
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

    fn execute_custom_driver_harness(
        compiler: &SounioCompiler,
        entrypoint: DriverEntrypoint,
        main_source: &str,
        user_args: Vec<String>,
    ) -> crate::vm::Value {
        let driver_source = compiler
            .resolve_driver_source_snapshot()
            .expect("driver source snapshot should resolve");
        let harness_source = format!("{driver_source}\n{main_source}\n");
        let _ = entrypoint; // Keeps parity with call sites that provide entrypoint context.
        let bytecode =
            compile_harness_source_for_test(&harness_source).expect("custom harness should compile");

        let mut vm = crate::vm::BytecodeVM::new();
        vm.set_user_args(user_args);
        vm.execute(&bytecode)
            .expect("custom harness should execute successfully")
    }

    fn compile_artifact_diag_string(vm_result: &crate::vm::Value) -> String {
        let crate::vm::Value::Struct(fields) = vm_result else {
            panic!("expected compile artifact struct, got {:?}", vm_result);
        };
        let diag_len = match fields.get("diagnostics_len") {
            Some(crate::vm::Value::Int(n)) if *n > 0 => *n as usize,
            _ => 0,
        };
        if diag_len == 0 {
            return String::new();
        }
        let diag_val = fields
            .get("diagnostics")
            .expect("compile artifact should include diagnostics bytes");
        let bytes = SounioCompiler::bytes_from_vm_value(diag_val, diag_len)
            .expect("diagnostics bytes should decode");
        String::from_utf8(bytes).expect("diagnostics should be valid UTF-8")
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
    fn test_driver_multimodule_pipeline_fails_closed_with_explicit_stub_diag() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let cache_dir = tempfile::tempdir().expect("temp dir for harness cache");
        let _cache_guard = set_env_var(
            "SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR",
            Some(cache_dir.path().to_str().expect("cache path utf8")),
        );

        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        let temp = tempfile::NamedTempFile::new().expect("temp source file");
        std::fs::write(temp.path(), "fn main() -> i32 { 0 }\n").expect("write temp source");

        let vm_result = execute_custom_driver_harness(
            &compiler,
            DriverEntrypoint::CompileFile,
            r#"
fn main() -> CompileArtifact with IO, Mut, Div, Panic, Alloc {
    compile_multimodule_program(get_arg(0))
}
"#,
            vec![temp.path().to_string_lossy().to_string()],
        );

        let crate::vm::Value::Struct(fields) = &vm_result else {
            panic!("expected compile artifact struct, got {:?}", vm_result);
        };
        assert_eq!(fields.get("ok"), Some(&crate::vm::Value::Bool(false)));
        assert_eq!(fields.get("exit_code"), Some(&crate::vm::Value::Int(6)));
        assert_eq!(
            compile_artifact_diag_string(&vm_result),
            "multimodule_pipeline_unavailable",
            "multimodule bootstrap stubs must fail closed with deterministic diagnostics"
        );
    }

    #[test]
    fn test_driver_compile_source_text_rejects_oversize_input_with_explicit_diag() {
        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileSource);
        let oversize = "a".repeat(65_537);
        let vm_result = compiler
            .run_driver_orchestration(DriverEntrypoint::CompileSource, Some(&oversize))
            .expect("driver orchestration should return compile artifact");

        let crate::vm::Value::Struct(fields) = &vm_result else {
            panic!("expected compile artifact struct, got {:?}", vm_result);
        };
        assert_eq!(fields.get("ok"), Some(&crate::vm::Value::Bool(false)));
        assert_eq!(fields.get("exit_code"), Some(&crate::vm::Value::Int(2)));
        assert_eq!(compile_artifact_diag_string(&vm_result), "source_text_too_large");
    }

    #[test]
    fn test_driver_compile_source_text_rejects_non_ascii_input_with_explicit_diag() {
        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        prime_driver_harness_cache_for_test(&compiler, DriverEntrypoint::CompileSource);
        let non_ascii_source = format!("fn main() -> i32 {{ 3{} }}", '\u{00E9}');
        let vm_result = compiler
            .run_driver_orchestration(DriverEntrypoint::CompileSource, Some(&non_ascii_source))
            .expect("driver orchestration should return compile artifact");

        let crate::vm::Value::Struct(fields) = &vm_result else {
            panic!("expected compile artifact struct, got {:?}", vm_result);
        };
        assert_eq!(fields.get("ok"), Some(&crate::vm::Value::Bool(false)));
        assert_eq!(fields.get("exit_code"), Some(&crate::vm::Value::Int(2)));
        assert_eq!(compile_artifact_diag_string(&vm_result), "source_text_non_ascii");
    }

    #[test]
    fn test_driver_run_pipeline_native_stage_returns_explicit_failure_code() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let cache_dir = tempfile::tempdir().expect("temp dir for harness cache");
        let _cache_guard = set_env_var(
            "SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR",
            Some(cache_dir.path().to_str().expect("cache path utf8")),
        );

        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        let vm_result = execute_custom_driver_harness(
            &compiler,
            DriverEntrypoint::CompileSource,
            r#"
fn main() -> StageOutput with Mut, Panic {
    var input = StageInput { stage: STAGE_CODEGEN_NATIVE, payload: [0; 65536], payload_len: 0 }
    run_pipeline(input)
}
"#,
            Vec::new(),
        );

        let crate::vm::Value::Struct(fields) = &vm_result else {
            panic!("expected stage output struct, got {:?}", vm_result);
        };
        assert_eq!(fields.get("stage"), Some(&crate::vm::Value::Int(5)));
        assert_eq!(fields.get("ok"), Some(&crate::vm::Value::Bool(false)));
        assert_eq!(fields.get("diag_code"), Some(&crate::vm::Value::Int(5)));
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

    #[test]
    fn test_parse_seed_signature_fields_valid() {
        let digest = "8cb2237d0679ca88db6464eac60da96345513964f4f5b4f9cd67f3f8a4c9f2d1";
        let parsed = parse_seed_signature_fields(&format!(
            "{BOOTSTRAP_SEED_SIGNATURE_MARKER} key=sounio-dev sha256={digest}\n"
        ))
        .expect("signature fields should parse");
        assert_eq!(parsed.key, "sounio-dev");
        assert_eq!(parsed.sha256, digest);
    }

    #[test]
    fn test_parse_seed_signature_fields_missing_key() {
        let digest = "8cb2237d0679ca88db6464eac60da96345513964f4f5b4f9cd67f3f8a4c9f2d1";
        let err = parse_seed_signature_fields(&format!(
            "{BOOTSTRAP_SEED_SIGNATURE_MARKER} sha256={digest}\n"
        ))
        .expect_err("keyless signature should fail");
        assert_eq!(err, SeedSignatureParseError::MissingKey);
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
        std::fs::write(
            &sig_path,
            format!(
                "{BOOTSTRAP_SEED_SIGNATURE_MARKER} key={} sha256={}\n",
                DEFAULT_BOOTSTRAP_SEED_TRUSTED_KEY, digest
            ),
        )
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
        std::fs::write(
            &sig_path,
            format!(
                "{BOOTSTRAP_SEED_SIGNATURE_MARKER} key={} sha256={}\n",
                DEFAULT_BOOTSTRAP_SEED_TRUSTED_KEY, digest
            ),
        )
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
    fn test_bootstrap_seed_rejects_non_zero_reserved_header() {
        let bytecode = vec![Bytecode::Push(crate::vm::Value::Int(7)), Bytecode::Return];
        let mut seed_blob = make_seed_blob(&bytecode, BOOTSTRAP_SEED_VERSION);
        seed_blob[10..12].copy_from_slice(&1u16.to_le_bytes());
        let err = SounioCompiler::decode_bootstrap_seed(Path::new("seed.bin"), &seed_blob)
            .expect_err("non-zero reserved header should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("BOOTSTRAP_SEED_INVALID_RESERVED"),
            "unexpected error: {}",
            msg
        );
    }

    #[test]
    fn test_bootstrap_seed_rejects_untrusted_signature_key() {
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
        std::fs::write(
            &sig_path,
            format!(
                "{BOOTSTRAP_SEED_SIGNATURE_MARKER} key=staging-key sha256={}\n",
                digest
            ),
        )
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
        let _g5 = set_env_var("SOUNIO_BOOTSTRAP_SEED_TRUSTED_KEY", Some("sounio-dev"));

        let err =
            SounioCompiler::load_bootstrap_seed_bytecode().expect_err("untrusted key must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("BOOTSTRAP_SEED_SIGNATURE_UNTRUSTED_KEY"),
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
    fn test_compile_rejects_removed_legacy_env_contracts() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        for knob in REMOVED_SELFHOST_ENV_KNOBS {
            let _guard = set_env_var(knob, Some("1"));
            let err = compiler
                .compile("fn main() -> i32 { 0 }\n")
                .expect_err("compile must reject removed legacy env contract");
            let msg = err.to_string();
            assert!(
                msg.contains("LEGACY_SELFHOST_ENV_REMOVED") && msg.contains(knob),
                "unexpected compile() error for {}: {}",
                knob,
                msg
            );
        }
    }

    #[test]
    fn test_compile_file_rejects_removed_legacy_env_contracts() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let compiler = SounioCompiler::new_embedded().expect("embedded compiler");
        let temp = tempfile::NamedTempFile::new().expect("temp source");
        std::fs::write(temp.path(), "fn main() -> i32 { 0 }\n").expect("write temp source");
        for knob in REMOVED_SELFHOST_ENV_KNOBS {
            let _guard = set_env_var(knob, Some("1"));
            let err = compiler
                .compile_file(temp.path().to_str().expect("temp path utf-8"))
                .expect_err("compile_file must reject removed legacy env contract");
            let msg = err.to_string();
            assert!(
                msg.contains("LEGACY_SELFHOST_ENV_REMOVED") && msg.contains(knob),
                "unexpected compile_file() error for {}: {}",
                knob,
                msg
            );
        }
    }

    #[test]
    fn test_compile_fails_closed_when_driver_harness_unavailable() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let cache_dir = tempfile::tempdir().expect("temp harness cache dir");
        let _cache_guard = set_env_var(
            "SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR",
            Some(cache_dir.path().to_str().expect("cache path utf8")),
        );

        let stdlib_root = tempfile::tempdir().expect("temp stdlib root");
        let compiler = new_filesystem_compiler_with_embedded_driver(stdlib_root.path());
        let err = compiler
            .compile("fn main() -> i32 { 0 }\n")
            .expect_err("compile should fail closed when harness is unavailable");
        let msg = err.to_string();
        assert!(
            msg.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
            "expected strict failure token, got: {msg}"
        );
        assert!(
            msg.contains("SELFHOST_DRIVER_HARNESS_UNAVAILABLE"),
            "expected harness unavailable token, got: {msg}"
        );
        assert!(
            msg.contains("entrypoint=bootstrap::driver::compile_source"),
            "expected compile_source entrypoint metadata, got: {msg}"
        );
    }

    #[test]
    fn test_compile_file_fails_closed_when_driver_harness_unavailable() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let cache_dir = tempfile::tempdir().expect("temp harness cache dir");
        let _cache_guard = set_env_var(
            "SOUNIO_SELFHOST_DRIVER_HARNESS_CACHE_DIR",
            Some(cache_dir.path().to_str().expect("cache path utf8")),
        );

        let stdlib_root = tempfile::tempdir().expect("temp stdlib root");
        let compiler = new_filesystem_compiler_with_embedded_driver(stdlib_root.path());
        let source = stdlib_root.path().join("sample.sio");
        std::fs::write(&source, "fn main() -> i32 { 0 }\n").expect("write temp source");

        let err = compiler
            .compile_file(source.to_str().expect("source path utf-8"))
            .expect_err("compile_file should fail closed when harness is unavailable");
        let msg = err.to_string();
        assert!(
            msg.contains("SELFHOST_STRICT_DRIVER_FAILURE"),
            "expected strict failure token, got: {msg}"
        );
        assert!(
            msg.contains("SELFHOST_DRIVER_HARNESS_UNAVAILABLE"),
            "expected harness unavailable token, got: {msg}"
        );
        assert!(
            msg.contains("entrypoint=bootstrap::driver::compile_file"),
            "expected compile_file entrypoint metadata, got: {msg}"
        );
    }

    #[test]
    fn test_removed_legacy_env_contracts_are_rejected() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        for knob in REMOVED_SELFHOST_ENV_KNOBS {
            let _guard = set_env_var(knob, Some("1"));
            let err = verify_removed_legacy_env_contracts()
                .expect_err("removed legacy env contract must fail");
            let msg = err.to_string();
            assert!(
                msg.contains("LEGACY_SELFHOST_ENV_REMOVED") && msg.contains(knob),
                "unexpected error for {}: {}",
                knob,
                msg
            );
        }
    }

    #[test]
    fn test_removed_legacy_env_contracts_pass_when_absent() {
        let _env_guard = env_lock().lock().expect("env lock poisoned");
        let _guards: Vec<_> = REMOVED_SELFHOST_ENV_KNOBS
            .iter()
            .map(|name| set_env_var(name, None))
            .collect();
        verify_removed_legacy_env_contracts().expect("all legacy env contracts unset");
    }
}
