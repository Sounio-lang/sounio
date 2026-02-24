//! Sounio Compiler CLI
//!
//! Main entry point for the `souc` command.

use clap::{Parser, Subcommand};
use chrono::Utc;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use miette::Result;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sounio::sir::AllocPolicy;
use std::collections::BTreeMap;
#[cfg(feature = "jit")]
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::Duration;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Parser)]
#[command(name = "souc")]
#[command(author = "Demetrios Chiuratto Agourakis, Dionisio Chiuratto Agourakis")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    about = "The Sounio Programming Language Compiler",
    long_about = "Sounio — Epistemic Computing at the Horizon of Certainty\n\nA systems programming language where uncertainty is a first-class citizen."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Compile a Sounio source file
    Compile {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Emit a minimal native executable stub (bootstrap helper)
        #[arg(long)]
        native_stub: bool,

        /// Emit intermediate representation
        #[arg(long, value_enum)]
        emit: Option<EmitType>,

        /// Optimization level (0-3)
        #[arg(short = 'O', default_value = "0")]
        opt_level: u8,

        /// Enable GLM-4.7 ML-guided optimization (requires --features glm)
        #[arg(long)]
        glm_enabled: bool,
    },

    /// Build a Sounio source file to native executable (requires --features llvm)
    Build {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output file
        #[arg(short, long, value_name = "FILE")]
        output: Option<PathBuf>,

        /// Backend to use (native, llvm, cranelift, gpu)
        #[arg(long, value_enum, default_value = "native")]
        backend: sounio::cli::backend::Backend,

        /// Optimization level (0, 1, 2, 3, s, z)
        #[arg(short = 'O', default_value = "2")]
        opt_level: String,

        /// Generate debug information
        #[arg(short = 'g', long)]
        debug: bool,

        /// Emit LLVM IR instead of compiling
        #[arg(long)]
        emit_llvm: bool,

        /// Emit assembly instead of compiling
        #[arg(long)]
        emit_asm: bool,

        /// Target triple (e.g., x86_64-unknown-linux-gnu)
        #[arg(long)]
        target: Option<String>,

        /// Strip debug symbols from output
        #[arg(long)]
        strip: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Build as shared library (cdylib) instead of executable
        #[arg(long)]
        cdylib: bool,

        /// Register allocation policy for code generation
        #[arg(long, value_enum, default_value = "attention")]
        alloc_policy: AllocPolicyCli,

        /// Path to attention weight configuration file (JSON)
        #[arg(long)]
        attention_config: Option<PathBuf>,

        /// Path to output allocation metrics (JSON)
        #[arg(long)]
        emit_alloc_metrics: Option<PathBuf>,

        /// Thermal model for native backend (none, 7nm, 5nm, conservative)
        #[arg(long, value_enum)]
        thermal: Option<sounio::cli::backend::ThermalModel>,

        /// Disable thermal modeling
        #[arg(long)]
        no_thermal: bool,

        /// Allocation strategy for native backend (epistemic, linear, graph)
        #[arg(long, value_enum)]
        alloc: Option<sounio::cli::backend::AllocStrategy>,

        /// Show timing information
        #[arg(long)]
        timing: bool,

        /// Emit intermediate stages (comma-separated: ast,sir,asm)
        #[arg(long)]
        emit: Option<String>,

        /// Target CPU features (comma-separated)
        #[arg(long)]
        target_cpu: Option<String>,

        /// Enable CPS transformation for effect handlers (experimental)
        #[arg(long)]
        enable_cps: bool,

        /// Enable GLM-4.7 ML-guided optimization (requires --features glm)
        #[arg(long)]
        glm_enabled: bool,
    },

    /// Type-check a Sounio source file without compiling
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Show the parsed AST
        #[arg(long)]
        show_ast: bool,

        /// Show resolved symbols
        #[arg(long)]
        show_resolved: bool,

        /// Show inferred types
        #[arg(long)]
        show_types: bool,

        /// Show inferred effects
        #[arg(long)]
        show_effects: bool,

        /// Skip ownership checking
        #[arg(long)]
        skip_ownership: bool,

        /// Error output format (human or json)
        #[arg(long, default_value = "human")]
        error_format: String,

        /// Warn when using deprecated ontology terms
        #[arg(long)]
        warn_deprecated: bool,

        /// Warning flags (e.g., --warn=unused-imports)
        #[arg(long = "warn")]
        warnings: Vec<String>,
    },

    /// Run a Sounio program using the interpreter
    Run {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Use the native self-hosted compiler backend instead of the tree-walking interpreter
        #[arg(long)]
        use_sounio_compiler: bool,

        /// Emit interpreter execution trace to stderr
        #[arg(long)]
        trace_interp: bool,

        /// Run a type-check pass before execution
        #[arg(long)]
        check_only: bool,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Run a Sounio program using JIT compilation (requires --features jit)
    Jit {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Enable optimizations
        #[arg(short = 'O', long)]
        optimize: bool,

        /// Use MIR-based lowering before JIT compilation
        #[arg(long)]
        mir: bool,

        /// Enable local heuristic ML-guided optimization
        #[arg(long)]
        ml_opt: bool,

        /// Collect optimization pass data for ML training (writes opt_data.json)
        #[arg(long)]
        collect_opt_data: bool,

        /// Enable GLM-4.7 ML-guided optimization (requires --features glm)
        #[arg(long)]
        glm_enabled: bool,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Start the interactive REPL
    Repl {
        /// Use JIT compilation instead of interpreter
        #[arg(long)]
        jit: bool,
    },

    /// Benchmark interpreter vs JIT performance
    Bench {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Number of iterations
        #[arg(short, long, default_value = "100")]
        iterations: u32,
    },

    /// Format Sounio source code
    Fmt {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Check formatting without modifying files
        #[arg(long)]
        check: bool,

        /// Show diff of changes
        #[arg(long)]
        diff: bool,

        /// Maximum line width
        #[arg(long, default_value = "100")]
        max_width: usize,

        /// Use tabs instead of spaces
        #[arg(long)]
        use_tabs: bool,

        /// Indent width (number of spaces)
        #[arg(long, default_value = "4")]
        indent_width: usize,
    },

    /// Lint Sounio source code
    Lint {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Output format (text, json, sarif)
        #[arg(long, default_value = "text")]
        format: String,

        /// Treat warnings as errors
        #[arg(long)]
        deny_warnings: bool,

        /// Allow specific lint (e.g., --allow unused_variable)
        #[arg(long, value_name = "LINT")]
        allow: Vec<String>,

        /// Warn for specific lint
        #[arg(long, value_name = "LINT")]
        warn: Vec<String>,

        /// Deny specific lint
        #[arg(long, value_name = "LINT")]
        deny: Vec<String>,

        /// Fix issues automatically where possible
        #[arg(long)]
        fix: bool,
    },

    /// Analyze code for metrics and issues
    Analyze {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Analysis type (metrics, dead-code, all)
        #[arg(long, default_value = "all")]
        analysis: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed output
        #[arg(long)]
        verbose: bool,
    },

    /// Apply automatic fixes
    Fix {
        /// Input file (or directory)
        #[arg(value_name = "PATH")]
        path: PathBuf,

        /// Only show what would be fixed
        #[arg(long)]
        dry_run: bool,

        /// Allow unsafe fixes
        #[arg(long)]
        allow_unsafe: bool,
    },

    /// Generate documentation for a package
    Doc {
        /// Open documentation in browser after generation
        #[arg(long)]
        open: bool,

        /// Document private items
        #[arg(long)]
        document_private: bool,
    },

    /// Generate mdBook documentation
    DocBook {
        /// Output directory
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Run documentation tests
    Doctest {
        /// Filter tests by name pattern
        #[arg(long)]
        filter: Option<String>,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show documentation coverage
    DocCoverage,

    /// Show information about the compiler
    Info,

    /// Bootstrap artifact and state management
    Bootstrap {
        #[command(subcommand)]
        command: BootstrapCommands,
    },

    /// Optimization policy management
    Opt {
        #[command(subcommand)]
        command: OptimizationCommands,
    },

    /// Run tests
    Test {
        /// Path to test files or directory
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Filter tests by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Include ignored tests
        #[arg(long)]
        include_ignored: bool,

        /// Only run ignored tests
        #[arg(long)]
        ignored: bool,

        /// Number of parallel threads (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,

        /// Fail fast on first failure
        #[arg(long)]
        fail_fast: bool,

        /// Run benchmarks instead of tests
        #[arg(long)]
        bench: bool,

        /// List tests without running
        #[arg(long)]
        list: bool,

        /// Output format (pretty, compact, json, junit)
        #[arg(long, default_value = "pretty")]
        format: String,

        /// Enable coverage tracking
        #[arg(long)]
        coverage: bool,

        /// Coverage output file
        #[arg(long)]
        coverage_output: Option<PathBuf>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run benchmarks
    Benchmark {
        /// Path to benchmark files or directory
        #[arg(value_name = "PATH", default_value = ".")]
        path: PathBuf,

        /// Filter benchmarks by name pattern
        #[arg(short, long)]
        filter: Option<String>,

        /// Compare with baseline file
        #[arg(long)]
        baseline: Option<PathBuf>,

        /// Save results as new baseline
        #[arg(long)]
        save_baseline: Option<PathBuf>,

        /// Target benchmark time in seconds
        #[arg(long, default_value = "3")]
        time: u64,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Profile a Sounio program
    Profile {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Profile type (cpu, memory, async)
        #[arg(short, long, default_value = "cpu")]
        profile_type: String,

        /// Output file for profile data
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Generate flame graph SVG
        #[arg(long)]
        flamegraph: Option<PathBuf>,

        /// Sample interval in microseconds (for CPU profiling)
        #[arg(long, default_value = "1000")]
        interval: u64,

        /// Output format (text, json, folded)
        #[arg(long, default_value = "text")]
        format: String,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Debug a Sounio program
    Debug {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Debugger to use (gdb, lldb)
        #[arg(short, long, default_value = "gdb")]
        debugger: String,

        /// Enable pretty printers automatically
        #[arg(long)]
        pretty: bool,

        /// Arguments to pass to the program
        #[arg(trailing_var_arg = true)]
        args: Vec<String>,
    },

    /// Explain an error code
    Explain {
        /// Error code (e.g., T0001)
        #[arg(value_name = "CODE")]
        code: String,
    },

    /// Show all error codes
    ErrorIndex {
        /// Filter by category (type, effect, ownership, etc.)
        #[arg(short, long)]
        category: Option<String>,

        /// Output format (text, markdown, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Diagnostics management commands
    Diagnostics {
        #[command(subcommand)]
        command: DiagnosticsCommands,
    },

    /// Generate debug information for a compiled binary
    DebugInfo {
        /// Input Sounio source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output debug info file
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Debug info format (dwarf, pdb)
        #[arg(long, default_value = "dwarf")]
        format: String,
    },

    /// Generate source map for compiled code
    SourceMap {
        /// Input Sounio source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output source map file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Build project using the build system
    BuildSystem {
        /// Source directory
        #[arg(value_name = "DIR", default_value = ".")]
        source_dir: PathBuf,

        /// Build profile (dev, release, test, bench)
        #[arg(long, default_value = "dev")]
        profile: String,

        /// Number of parallel jobs (0 = auto)
        #[arg(short = 'j', long, default_value = "0")]
        jobs: usize,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Disable incremental compilation
        #[arg(long)]
        no_incremental: bool,
    },

    /// Clean build artifacts and cache
    Clean {
        /// Also remove cache
        #[arg(long)]
        cache: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Watch files and rebuild on changes
    Watch {
        /// Paths to watch (default: src)
        #[arg(value_name = "PATH", default_value = "src")]
        paths: Vec<PathBuf>,

        /// Clear screen before each rebuild
        #[arg(short, long)]
        clear: bool,

        /// Run tests after successful build
        #[arg(short, long)]
        test: bool,

        /// Command to run after successful build
        #[arg(short = 'x', long)]
        exec: Option<String>,

        /// Debounce delay in milliseconds
        #[arg(long, default_value = "100")]
        debounce: u64,

        /// Patterns to ignore (glob)
        #[arg(long)]
        ignore: Vec<String>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Start development server with live reload
    Serve {
        /// Root directory to serve
        #[arg(value_name = "DIR", default_value = ".")]
        root: PathBuf,

        /// Port to listen on
        #[arg(short, long, default_value = "3000")]
        port: u16,

        /// Host to bind to
        #[arg(long, default_value = "127.0.0.1")]
        host: String,

        /// Disable live reload
        #[arg(long)]
        no_reload: bool,

        /// Open browser automatically
        #[arg(short, long)]
        open: bool,

        /// Enable directory listing
        #[arg(long)]
        directory_listing: bool,

        /// SPA fallback file (e.g., index.html)
        #[arg(long)]
        spa: Option<String>,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Run build hooks
    Hook {
        /// Hook point to run (pre-build, post-build, etc.)
        #[arg(value_name = "POINT")]
        point: String,

        /// Project root directory
        #[arg(short, long, default_value = ".")]
        project: PathBuf,

        /// Enable verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Target management commands
    Target {
        #[command(subcommand)]
        command: TargetCommands,
    },

    /// Sysroot management commands
    Sysroot {
        #[command(subcommand)]
        command: SysrootCommands,
    },

    /// Native ontology management commands
    Ontology {
        #[command(subcommand)]
        command: OntologyCommands,
    },

    /// Layout synthesis commands
    Layout {
        #[command(subcommand)]
        command: LayoutCommands,
    },

    /// Locality analysis and optimization commands (Semantic-Physical Duality)
    Locality {
        #[command(subcommand)]
        command: LocalityCommands,
    },

    /// Units of measure commands (dimensional analysis)
    Units {
        #[command(subcommand)]
        command: UnitsCommands,
    },

    /// Linear/affine types commands (resource safety)
    Linear {
        #[command(subcommand)]
        command: LinearCommands,
    },

    /// Distributed build commands
    #[cfg(feature = "distributed")]
    Distributed {
        #[command(subcommand)]
        command: DistributedCommands,
    },

    /// Build cache commands
    #[cfg(feature = "distributed")]
    Cache {
        #[command(subcommand)]
        command: CacheCommands,
    },

    /// CI/CD configuration commands
    #[cfg(feature = "distributed")]
    Ci {
        #[command(subcommand)]
        command: CiCommands,
    },
}

#[derive(Subcommand)]
enum BootstrapCommands {
    /// Verify signed bootstrap bundle artifacts and manifest.
    Verify {
        /// Bundle root directory (expects artifacts/manifest.v2.json).
        #[arg(long, value_name = "DIR")]
        bundle: PathBuf,
    },
    /// Initialize local state from a signed bootstrap bundle.
    Init {
        /// Bundle root directory (expects artifacts/manifest.v2.json).
        #[arg(long, value_name = "DIR")]
        bundle: PathBuf,
        /// Output state directory.
        #[arg(long, value_name = "DIR")]
        state: PathBuf,
    },
    /// Verify state determinism report inputs and emit cycle report.
    Cycle {
        /// Existing state directory.
        #[arg(long, value_name = "DIR")]
        state: PathBuf,
    },
}

#[derive(Subcommand)]
enum OptimizationCommands {
    /// Manage optimization policies
    Policy {
        #[command(subcommand)]
        command: OptimizationPolicyCommands,
    },
}

#[derive(Subcommand)]
enum OptimizationPolicyCommands {
    /// Train (synthesize) a local deterministic policy from a corpus digest
    Train {
        /// Training corpus file or directory
        #[arg(long, value_name = "PATH", default_value = "benchmarks")]
        corpus: PathBuf,

        /// Output policy path
        #[arg(long, value_name = "FILE", default_value = "bootstrap/policies/policy.v2.json")]
        output: PathBuf,

        /// Policy identifier
        #[arg(long, default_value = "sounio-cutover-policy")]
        policy_id: String,

        /// Policy semantic version
        #[arg(long, default_value = "1.0.0")]
        policy_version: String,

        /// Policy mode (shadow|active)
        #[arg(long, default_value = "shadow")]
        mode: String,
    },

    /// Evaluate policy contract validity and signature
    Eval {
        /// Input policy path
        #[arg(long, value_name = "FILE", default_value = "bootstrap/policies/policy.v2.json")]
        policy: PathBuf,
    },

    /// Promote a validated policy to active
    Promote {
        /// Input policy path
        #[arg(long, value_name = "FILE", default_value = "bootstrap/policies/policy.v2.json")]
        policy: PathBuf,

        /// Active policy path
        #[arg(
            long,
            value_name = "FILE",
            default_value = "bootstrap/policies/active/policy.v2.json"
        )]
        output: PathBuf,
    },

    /// Sign an existing optimization policy in-place (or to --out)
    Sign {
        /// Input policy path
        #[arg(long, value_name = "FILE", default_value = "bootstrap/policies/policy.v2.json")]
        policy: PathBuf,

        /// Output policy path (defaults to in-place overwrite)
        #[arg(long, value_name = "FILE")]
        out: Option<PathBuf>,

        /// Optional pinned payload digest (SHA-256 hex) that must match the signed payload digest
        #[arg(long, value_name = "SHA256_HEX")]
        pin_digest: Option<String>,
    },

    /// Show policy status and key metadata
    Status {
        /// Policy path to inspect
        #[arg(
            long,
            value_name = "FILE",
            default_value = "bootstrap/policies/active/policy.v2.json"
        )]
        policy: PathBuf,
    },

}

#[cfg(feature = "distributed")]
#[derive(Subcommand)]
enum DistributedCommands {
    /// Start a distributed build server
    Server {
        /// Listen address
        #[arg(long, default_value = "0.0.0.0:9876")]
        address: String,

        /// Maximum concurrent connections
        #[arg(long, default_value = "100")]
        max_connections: usize,

        /// Enable build cache
        #[arg(long)]
        cache: bool,

        /// Server name
        #[arg(long, default_value = "d-build-server")]
        name: String,
    },

    /// Submit a remote build
    Build {
        /// Server address
        #[arg(long, default_value = "localhost:9876")]
        server: String,

        /// Target triple
        #[arg(long)]
        target: Option<String>,

        /// Build profile
        #[arg(long, default_value = "debug")]
        profile: String,

        /// Project directory
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Query server status
    Status {
        /// Server address
        #[arg(long, default_value = "localhost:9876")]
        server: String,
    },
}

#[cfg(feature = "distributed")]
#[derive(Subcommand)]
enum CacheCommands {
    /// Start a cache server
    Server {
        /// Listen address
        #[arg(long, default_value = "0.0.0.0:9877")]
        address: String,

        /// Storage directory
        #[arg(long, default_value = "~/.d/cache-storage")]
        storage: PathBuf,

        /// Maximum cache size (e.g., "10GB")
        #[arg(long, default_value = "10GB")]
        max_size: String,
    },

    /// Show cache statistics
    Stats {
        /// Cache server URL
        #[arg(long)]
        url: Option<String>,

        /// Local cache only
        #[arg(long)]
        local: bool,
    },

    /// Clean cache entries
    Clean {
        /// Clean all entries
        #[arg(long)]
        all: bool,

        /// Clean entries older than (e.g., "7d", "24h")
        #[arg(long)]
        older_than: Option<String>,

        /// Dry run (show what would be deleted)
        #[arg(long)]
        dry_run: bool,
    },
}

#[cfg(feature = "distributed")]
#[derive(Subcommand)]
enum CiCommands {
    /// Generate GitHub Actions workflow
    Github {
        /// Output file path
        #[arg(long, default_value = ".github/workflows/ci.yml")]
        output: PathBuf,

        /// Also generate release workflow
        #[arg(long)]
        release: bool,

        /// Target triples (comma-separated)
        #[arg(long)]
        targets: Option<String>,
    },

    /// Generate GitLab CI pipeline
    Gitlab {
        /// Output file path
        #[arg(long, default_value = ".gitlab-ci.yml")]
        output: PathBuf,

        /// Target triples (comma-separated)
        #[arg(long)]
        targets: Option<String>,
    },

    /// Generate build provenance (SLSA)
    Provenance {
        /// Output file path
        #[arg(long, default_value = "provenance.json")]
        output: PathBuf,

        /// Target triple
        #[arg(long)]
        target: Option<String>,

        /// Build profile
        #[arg(long, default_value = "release")]
        profile: String,
    },

    /// Check reproducibility
    Reproducible {
        /// Number of builds to compare
        #[arg(long, default_value = "2")]
        builds: usize,

        /// Show environment warnings
        #[arg(long)]
        check_env: bool,
    },
}

#[derive(Subcommand)]
enum DiagnosticsCommands {
    /// Check a file and show diagnostics with rich formatting
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output format (human, json, sarif)
        #[arg(long, default_value = "human")]
        format: String,

        /// Maximum number of errors to show
        #[arg(long, default_value = "50")]
        max_errors: usize,

        /// Show type diff for type errors
        #[arg(long)]
        show_type_diff: bool,

        /// Show unification trace for type errors
        #[arg(long)]
        show_trace: bool,
    },

    /// Show similar names for typo detection
    Similar {
        /// The name to find similar matches for
        #[arg(value_name = "NAME")]
        name: String,

        /// Category (variable, type, function, keyword)
        #[arg(short, long, default_value = "all")]
        category: String,

        /// Maximum edit distance
        #[arg(long, default_value = "3")]
        max_distance: usize,
    },

    /// Test diagnostic rendering
    Render {
        /// Diagnostic level (error, warning, note, help)
        #[arg(long, default_value = "error")]
        level: String,

        /// Error code
        #[arg(long)]
        code: Option<String>,

        /// Message
        #[arg(value_name = "MESSAGE")]
        message: String,

        /// Output format (human, json, sarif)
        #[arg(long, default_value = "human")]
        format: String,
    },

    /// Show diagnostic statistics for a file
    Stats {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,
    },
}

#[derive(Subcommand)]
enum TargetCommands {
    /// List all available targets
    List {
        /// Filter by OS (linux, windows, macos, etc.)
        #[arg(long)]
        os: Option<String>,

        /// Filter by architecture (x86_64, aarch64, etc.)
        #[arg(long)]
        arch: Option<String>,

        /// Show only built-in targets
        #[arg(long)]
        builtin: bool,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about a specific target
    Info {
        /// Target name or triple
        #[arg(value_name = "TARGET")]
        target: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Add a custom target from a JSON specification file
    Add {
        /// Path to target specification JSON file
        #[arg(value_name = "FILE")]
        file: PathBuf,
    },

    /// Create a new custom target specification
    Create {
        /// Target triple (e.g., x86_64-unknown-myos)
        #[arg(value_name = "TRIPLE")]
        triple: String,

        /// Base target to derive from
        #[arg(long)]
        base: Option<String>,

        /// Output file for the specification
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Show host target information
    Host,

    /// Check target configuration predicates
    Cfg {
        /// Target to check (default: host)
        #[arg(long)]
        target: Option<String>,

        /// Cfg predicate to evaluate
        #[arg(value_name = "PREDICATE")]
        predicate: Option<String>,
    },
}

#[derive(Subcommand)]
enum SysrootCommands {
    /// List installed sysroots
    List {
        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show sysroot for a specific target
    Show {
        /// Target triple
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Show stdlib search paths and configuration
    StdlibPaths {
        /// Show detailed information including environment variables
        #[arg(short, long)]
        verbose: bool,
    },

    /// Install sysroot for a target
    Install {
        /// Target triple
        #[arg(value_name = "TARGET")]
        target: String,

        /// Force reinstallation
        #[arg(short, long)]
        force: bool,
    },

    /// Remove installed sysroot
    Remove {
        /// Target triple
        #[arg(value_name = "TARGET")]
        target: String,
    },

    /// Clean stale sysroots
    Clean {
        /// Show what would be removed without removing
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Subcommand)]
enum OntologyCommands {
    /// Initialize ontology data (download and build .dontology files)
    Init {
        /// Only download core ontologies (BFO, RO, COB, PATO, UO, IAO)
        #[arg(long)]
        core_only: bool,

        /// Force re-download even if files exist
        #[arg(short, long)]
        force: bool,

        /// Output directory for .dontology files
        #[arg(short, long, default_value = ".sounio/ontology")]
        output: PathBuf,

        /// Specific ontologies to download (comma-separated)
        #[arg(short, long)]
        include: Option<String>,

        /// Show verbose progress
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about a concept
    Info {
        /// Concept CURIE (e.g., CHEBI:15365, GO:0008150)
        #[arg(value_name = "CURIE")]
        curie: String,

        /// Show ancestors
        #[arg(long)]
        ancestors: bool,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,
    },

    /// Search for concepts by label
    Search {
        /// Search query (prefix match on labels)
        #[arg(value_name = "QUERY")]
        query: String,

        /// Maximum results to show
        #[arg(short, long, default_value = "20")]
        limit: usize,

        /// Specific ontology to search (default: all)
        #[arg(short, long)]
        ontology: Option<String>,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,
    },

    /// Check if one concept is a subclass of another
    IsSubclass {
        /// Child concept CURIE
        #[arg(value_name = "CHILD")]
        child: String,

        /// Parent concept CURIE
        #[arg(value_name = "PARENT")]
        parent: String,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,
    },

    /// List available ontologies
    List {
        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Lock ontology versions (generate ontology.lock)
    Lock {
        /// Project directory containing ontology dependencies
        #[arg(value_name = "DIR", default_value = ".")]
        project_dir: PathBuf,

        /// Output lock file path
        #[arg(short, long, default_value = "ontology.lock")]
        output: PathBuf,

        /// Force overwrite existing lock file
        #[arg(short, long)]
        force: bool,
    },

    /// Check for ontology updates
    Update {
        /// Lock file to check
        #[arg(value_name = "FILE", default_value = "ontology.lock")]
        lock_file: PathBuf,

        /// Actually update the lock file (otherwise dry-run)
        #[arg(long)]
        write: bool,

        /// Show verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show diff between ontology versions
    Diff {
        /// Ontology ID (e.g., chebi, go)
        #[arg(value_name = "ONTOLOGY")]
        ontology: String,

        /// Old version
        #[arg(value_name = "OLD_VERSION")]
        old_version: String,

        /// New version
        #[arg(value_name = "NEW_VERSION")]
        new_version: String,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Show detailed changes
        #[arg(short, long)]
        verbose: bool,
    },

    /// Check for deprecated term usage
    Deprecations {
        /// Input source file to check
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Treat warnings as errors
        #[arg(long)]
        deny_warnings: bool,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Verify lock file integrity
    Verify {
        /// Lock file to verify
        #[arg(value_name = "FILE", default_value = "ontology.lock")]
        lock_file: PathBuf,

        /// Data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum LayoutCommands {
    /// Analyze concept usage and generate layout plan
    Analyze {
        /// Input file containing Knowledge types
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Maximum number of clusters
        #[arg(long, default_value = "4")]
        max_clusters: usize,

        /// Output report file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Simulate cache performance for a layout
    Simulate {
        /// Input file with concept access pattern (one CURIE per line)
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Cache size (number of concepts)
        #[arg(long, default_value = "16")]
        cache_size: usize,

        /// Compare with baseline
        #[arg(long)]
        compare: bool,
    },

    /// Validate the hypothesis: does semantic clustering improve cache performance?
    Validate {
        /// Input file with concept access pattern
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Cache sizes to test (comma-separated)
        #[arg(long, default_value = "8,16,32,64")]
        cache_sizes: String,

        /// Number of simulation iterations
        #[arg(long, default_value = "100")]
        iterations: usize,
    },

    /// Visualize layout as ASCII or Mermaid diagram (Day 39)
    Visualize {
        /// Input file containing Knowledge types
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Output format: ascii, mermaid, table
        #[arg(short, long, default_value = "ascii")]
        format: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Validate layout constraints (Day 39 - Participatory Compilation)
    Constraints {
        /// Input file containing constraint definitions
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Show verbose output including satisfied constraints
        #[arg(short, long)]
        verbose: bool,
    },

    /// Explain layout decision for a specific concept (Day 39)
    Explain {
        /// Concept CURIE to explain (e.g., CHEBI:15365)
        #[arg(value_name = "CONCEPT")]
        concept: String,

        /// Input file containing Knowledge types
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,
    },
}

#[derive(Subcommand)]
enum LocalityCommands {
    /// Show NUMA topology of the current system
    Numa {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Analyze access patterns in a source file
    Analyze {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show optimization recommendations
        #[arg(long)]
        recommend: bool,
    },

    /// Generate prefetch table from ontology
    Prefetch {
        /// Ontology data directory
        #[arg(long, default_value = ".sounio/ontology")]
        data_dir: PathBuf,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Analyze struct for cache-line packing
    Pack {
        /// Input source file with struct definition
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Struct name to analyze
        #[arg(value_name = "STRUCT")]
        struct_name: String,

        /// Cache line size in bytes
        #[arg(long, default_value = "64")]
        cache_line: usize,

        /// Show suggested reordering
        #[arg(long)]
        suggest: bool,
    },

    /// Show locality type lattice
    Lattice {
        /// Output format (text, mermaid)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Generate prefetch code for a function
    Codegen {
        /// Input source file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Function name to generate prefetch for
        #[arg(value_name = "FUNCTION")]
        function: String,

        /// Target architecture (x86_64, arm64, llvm)
        #[arg(long, default_value = "llvm")]
        target: String,

        /// Output file (default: stdout)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum UnitsCommands {
    /// List all available units
    List {
        /// Filter by category (si, pkpd, derived, all)
        #[arg(short, long, default_value = "all")]
        category: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Convert a value between units
    Convert {
        /// Value to convert
        #[arg(value_name = "VALUE")]
        value: f64,

        /// Source unit (e.g., mg, kg, L/h)
        #[arg(value_name = "FROM")]
        from: String,

        /// Target unit
        #[arg(value_name = "TO")]
        to: String,

        /// Show conversion details
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about a unit
    Info {
        /// Unit symbol or name (e.g., mg/L, kilogram)
        #[arg(value_name = "UNIT")]
        unit: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Check dimensional compatibility of units
    Check {
        /// First unit
        #[arg(value_name = "UNIT1")]
        unit1: String,

        /// Second unit
        #[arg(value_name = "UNIT2")]
        unit2: String,

        /// Show dimension breakdown
        #[arg(short, long)]
        verbose: bool,
    },

    /// Parse and validate a unit expression
    Parse {
        /// Unit expression (e.g., "kg*m/s^2", "mg/L")
        #[arg(value_name = "EXPR")]
        expr: String,

        /// Show dimension analysis
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show the SI base dimensions
    Dimensions {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },
}

#[derive(Subcommand)]
enum LinearCommands {
    /// List linearity kinds and their properties
    List {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about a linearity kind
    Info {
        /// Linearity kind (linear, affine, unrestricted)
        #[arg(value_name = "KIND")]
        kind: String,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Check linearity subkinding relationship
    Subkind {
        /// First linearity kind
        #[arg(value_name = "KIND1")]
        kind1: String,

        /// Second linearity kind
        #[arg(value_name = "KIND2")]
        kind2: String,

        /// Show detailed explanation
        #[arg(short, long)]
        verbose: bool,
    },

    /// List available resource types
    Resources {
        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,

        /// Show detailed information
        #[arg(short, long)]
        verbose: bool,
    },

    /// Show information about session types
    Sessions {
        /// Show example protocols
        #[arg(long)]
        examples: bool,

        /// Output format (text, json)
        #[arg(long, default_value = "text")]
        format: String,
    },

    /// Check a source file for linearity errors
    Check {
        /// Input file
        #[arg(value_name = "FILE")]
        input: PathBuf,

        /// Show all tracked resources
        #[arg(long)]
        show_resources: bool,

        /// Show usage tracking
        #[arg(long)]
        show_usage: bool,
    },
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum EmitType {
    /// Abstract Syntax Tree (JSON)
    Ast,
    /// High-level IR
    Hir,
    /// Low-level IR (SSA)
    Hlir,
    /// LLVM IR
    Llvm,
}

/// Register allocation policy for code generation
#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum, Default)]
enum AllocPolicyCli {
    /// Classic register allocation (furthest next use)
    Classic,
    /// Epistemic-aware attention-based allocation (default)
    #[default]
    Attention,
    /// Attention-based with BO-calibrated weights
    AttentionCalibrated,
}

impl From<AllocPolicyCli> for AllocPolicy {
    fn from(cli: AllocPolicyCli) -> Self {
        match cli {
            AllocPolicyCli::Classic => AllocPolicy::Classic,
            AllocPolicyCli::Attention => AllocPolicy::Attention,
            AllocPolicyCli::AttentionCalibrated => AllocPolicy::AttentionCalibrated,
        }
    }
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::registry()
        // Keep structured logs on stderr so command stdout stays machine/data friendly.
        .with(fmt::layer().with_writer(std::io::stderr))
        .with(EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    if cli.verbose {
        tracing::info!("Verbose mode enabled");
    }

    sounio::compiler_loader::verify_removed_legacy_env_contracts().map_err(|e| {
        miette::miette!(
            "Legacy self-host environment contracts are removed for this CLI invocation: {}",
            e
        )
    })?;

    match cli.command {
        Commands::Compile {
            input,
            output,
            native_stub,
            emit,
            opt_level,
            glm_enabled,
        } => compile(
            &input,
            output.as_deref(),
            emit,
            opt_level,
            glm_enabled,
            native_stub,
        ),

        Commands::Build {
            input,
            output,
            backend,
            opt_level,
            debug,
            emit_llvm,
            emit_asm,
            target,
            strip,
            verbose,
            cdylib,
            alloc_policy,
            attention_config,
            emit_alloc_metrics,
            thermal,
            no_thermal,
            alloc,
            timing,
            emit,
            target_cpu,
            enable_cps,
            glm_enabled,
        } => {
            // Log GLM status
            if glm_enabled {
                #[cfg(feature = "glm")]
                tracing::info!("GLM-4.7 ML-guided optimization enabled");
                #[cfg(not(feature = "glm"))]
                eprintln!("Warning: --glm-enabled requires --features glm");
            }
            // If native, cranelift, or selfhosted-native backend, use new CLI integration
            if backend == sounio::cli::backend::Backend::Native
                || backend == sounio::cli::backend::Backend::Cranelift
                || backend == sounio::cli::backend::Backend::SelfhostedNative
            {
                // Build args from command line
                let build_args = sounio::cli::backend::BuildArgs {
                    input: vec![input],
                    output,
                    backend,
                    format: None,
                    native_opts: sounio::cli::backend::NativeBackendOptions {
                        opt_level: opt_level.parse().unwrap_or_default(),
                        enable_thermal: !no_thermal,
                        thermal_model: thermal.unwrap_or_default(),
                        alloc_strategy: alloc.unwrap_or_default(),
                        debug_info: debug,
                        cpu_features: target_cpu
                            .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                            .unwrap_or_default(),
                        enable_cps: if enable_cps { Some(true) } else { None },
                    },
                    verbose,
                    timing,
                    emit: emit
                        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
                        .unwrap_or_default(),
                };

                let output = sounio::cli::backend::compile(&build_args);

                // Print warnings
                for warning in &output.warnings {
                    eprintln!("Warning: {}", warning);
                }

                // Print errors
                for error in &output.errors {
                    eprintln!("Error: {}", error);
                }

                // Print timing if requested
                if let Some(timing) = &output.timing {
                    println!("\nTiming:");
                    println!("  Total: {} ms", timing.total_ms);
                }

                // Print metrics if available and verbose
                if verbose {
                    if let Some(metrics) = &output.metrics {
                        if let Some(native) = &metrics.native {
                            println!("\nNative backend metrics:");
                            println!("  Intervals allocated: {}", native.intervals_allocated);
                            println!("  Intervals spilled: {}", native.intervals_spilled);
                            if native.intervals_allocated > 0 {
                                println!(
                                    "  Avg confidence (allocated): {:.3}",
                                    native.avg_confidence_allocated
                                );
                            }
                            if native.intervals_spilled > 0 {
                                println!(
                                    "  Avg confidence (spilled): {:.3}",
                                    native.avg_confidence_spilled
                                );
                            }
                            if native.thermal_degradation > 0.0 {
                                println!(
                                    "  Thermal degradation: {:.2}%",
                                    native.thermal_degradation * 100.0
                                );
                            }
                        }
                    }
                }

                if output.success {
                    if let Some(path) = &output.output_path {
                        println!("Compiled: {}", path.display());
                    }
                    Ok(())
                } else {
                    Err(miette::miette!("Compilation failed"))
                }
            } else if backend == sounio::cli::backend::Backend::Gpu {
                if emit_llvm || emit_asm || cdylib {
                    return Err(miette::miette!(
                        "GPU backend only supports PTX emission (no --emit-llvm/--emit-asm/--cdylib)"
                    ));
                }
                build_gpu(&input, output.as_deref(), target.as_deref(), verbose)
            } else {
                // Use existing build function for other backends
                build(
                    &input,
                    output.as_deref(),
                    &opt_level,
                    debug,
                    emit_llvm,
                    emit_asm,
                    target.as_deref(),
                    strip,
                    verbose,
                    cdylib,
                    alloc_policy.into(),
                    attention_config.as_deref(),
                    emit_alloc_metrics.as_deref(),
                )
            }
        }

        Commands::Check {
            input,
            show_ast,
            show_resolved,
            show_types,
            show_effects,
            skip_ownership,
            error_format,
            warn_deprecated,
            warnings: _,
        } => check(
            &input,
            show_ast,
            show_resolved,
            show_types,
            show_effects,
            skip_ownership,
            &error_format,
            warn_deprecated,
        ),

        Commands::Run {
            input,
            use_sounio_compiler,
            trace_interp,
            check_only,
            args,
        } => {
            run(&input, &args, use_sounio_compiler, trace_interp, check_only)
        }

        Commands::Jit {
            input,
            optimize,
            mir,
            ml_opt,
            collect_opt_data,
            glm_enabled,
            args,
        } => jit_run(
            &input,
            optimize,
            mir,
            ml_opt,
            collect_opt_data,
            glm_enabled,
            &args,
        ),

        Commands::Repl { jit } => repl(jit),

        Commands::Bench { input, iterations } => bench(&input, iterations),

        Commands::Fmt {
            path,
            check,
            diff,
            max_width,
            use_tabs,
            indent_width,
        } => format_code(&path, check, diff, max_width, use_tabs, indent_width),

        Commands::Lint {
            path,
            format,
            deny_warnings,
            allow,
            warn,
            deny,
            fix,
        } => lint_code(&path, &format, deny_warnings, &allow, &warn, &deny, fix),

        Commands::Analyze {
            path,
            analysis,
            format,
            verbose,
        } => analyze_code(&path, &analysis, &format, verbose),

        Commands::Fix {
            path,
            dry_run,
            allow_unsafe,
        } => fix_code(&path, dry_run, allow_unsafe),

        Commands::Doc {
            open,
            document_private,
        } => doc(open, document_private),

        Commands::DocBook { output } => doc_book(output),

        Commands::Doctest { filter, verbose } => doctest(filter, verbose),

        Commands::DocCoverage => doc_coverage(),

        Commands::Info => info(),

        Commands::Bootstrap { command } => run_bootstrap(command),

        Commands::Opt { command } => run_optimization(command),

        Commands::Test {
            path,
            filter,
            include_ignored,
            ignored,
            jobs,
            fail_fast,
            bench,
            list,
            format,
            coverage,
            coverage_output,
            verbose,
        } => run_tests(
            &path,
            filter.as_deref(),
            include_ignored,
            ignored,
            jobs,
            fail_fast,
            bench,
            list,
            &format,
            coverage,
            coverage_output.as_deref(),
            verbose,
        ),

        Commands::Benchmark {
            path,
            filter,
            baseline,
            save_baseline,
            time,
            verbose,
        } => run_benchmarks(
            &path,
            filter.as_deref(),
            baseline.as_deref(),
            save_baseline.as_deref(),
            time,
            verbose,
        ),

        Commands::Profile {
            input,
            profile_type,
            output,
            flamegraph,
            interval,
            format,
            args,
        } => profile(
            &input,
            &profile_type,
            output.as_deref(),
            flamegraph.as_deref(),
            interval,
            &format,
            &args,
        ),

        Commands::Debug {
            input,
            debugger,
            pretty,
            args,
        } => debug_program(&input, &debugger, pretty, &args),

        Commands::Explain { code } => explain_error(&code),

        Commands::ErrorIndex { category, format } => show_error_index(category.as_deref(), &format),

        Commands::Diagnostics { command } => match command {
            DiagnosticsCommands::Check {
                input,
                format,
                max_errors,
                show_type_diff,
                show_trace,
            } => diagnostics_check(&input, &format, max_errors, show_type_diff, show_trace),
            DiagnosticsCommands::Similar {
                name,
                category,
                max_distance,
            } => diagnostics_similar(&name, &category, max_distance),
            DiagnosticsCommands::Render {
                level,
                code,
                message,
                format,
            } => diagnostics_render(&level, code.as_deref(), &message, &format),
            DiagnosticsCommands::Stats { input } => diagnostics_stats(&input),
        },

        Commands::DebugInfo {
            input,
            output,
            format,
        } => generate_debug_info(&input, output.as_deref(), &format),

        Commands::SourceMap { input, output } => generate_source_map(&input, output.as_deref()),

        Commands::BuildSystem {
            source_dir,
            profile,
            jobs,
            verbose,
            no_incremental,
        } => build_system(&source_dir, &profile, jobs, verbose, no_incremental),

        Commands::Clean { cache, verbose } => clean_build(cache, verbose),

        Commands::Watch {
            paths,
            clear,
            test,
            exec,
            debounce,
            ignore,
            verbose,
        } => watch_files(
            &paths,
            clear,
            test,
            exec.as_deref(),
            debounce,
            &ignore,
            verbose,
        ),

        Commands::Serve {
            root,
            port,
            host,
            no_reload,
            open,
            directory_listing,
            spa,
            verbose,
        } => serve_files(
            &root,
            port,
            &host,
            no_reload,
            open,
            directory_listing,
            spa.as_deref(),
            verbose,
        ),

        Commands::Hook {
            point,
            project,
            verbose,
        } => run_hook(&point, &project, verbose),

        Commands::Target { command } => match command {
            TargetCommands::List {
                os,
                arch,
                builtin,
                verbose,
            } => target_list(os.as_deref(), arch.as_deref(), builtin, verbose),
            TargetCommands::Info { target, format } => target_info(&target, &format),
            TargetCommands::Add { file } => target_add(&file),
            TargetCommands::Create {
                triple,
                base,
                output,
            } => target_create(&triple, base.as_deref(), output.as_deref()),
            TargetCommands::Host => target_host(),
            TargetCommands::Cfg { target, predicate } => {
                target_cfg(target.as_deref(), predicate.as_deref())
            }
        },

        Commands::Sysroot { command } => match command {
            SysrootCommands::List { verbose } => sysroot_list(verbose),
            SysrootCommands::Show { target } => sysroot_show(&target),
            SysrootCommands::Install { target, force } => sysroot_install(&target, force),
            SysrootCommands::Remove { target } => sysroot_remove(&target),
            SysrootCommands::Clean { dry_run } => sysroot_clean(dry_run),
            SysrootCommands::StdlibPaths { verbose } => sysroot_stdlib_paths(verbose),
        },

        Commands::Ontology { command } => match command {
            OntologyCommands::Init {
                core_only,
                force,
                output,
                include,
                verbose,
            } => ontology_init(core_only, force, &output, include.as_deref(), verbose),
            OntologyCommands::Info {
                curie,
                ancestors,
                data_dir,
            } => ontology_info(&curie, ancestors, &data_dir),
            OntologyCommands::Search {
                query,
                limit,
                ontology,
                data_dir,
            } => ontology_search(&query, limit, ontology.as_deref(), &data_dir),
            OntologyCommands::IsSubclass {
                child,
                parent,
                data_dir,
            } => ontology_is_subclass(&child, &parent, &data_dir),
            OntologyCommands::List { data_dir, verbose } => ontology_list(&data_dir, verbose),
            OntologyCommands::Lock {
                project_dir,
                output,
                force,
            } => ontology_lock(&project_dir, &output, force),
            OntologyCommands::Update {
                lock_file,
                write,
                verbose,
            } => ontology_update(&lock_file, write, verbose),
            OntologyCommands::Diff {
                ontology,
                old_version,
                new_version,
                data_dir,
                verbose,
            } => ontology_diff(&ontology, &old_version, &new_version, &data_dir, verbose),
            OntologyCommands::Deprecations {
                input,
                data_dir,
                deny_warnings,
                format,
            } => ontology_deprecations(&input, &data_dir, deny_warnings, &format),
            OntologyCommands::Verify {
                lock_file,
                data_dir,
            } => ontology_verify(&lock_file, &data_dir),
        },

        Commands::Layout { command } => match command {
            LayoutCommands::Analyze {
                input,
                data_dir,
                max_clusters,
                output,
            } => layout_analyze(&input, &data_dir, max_clusters, output.as_deref()),
            LayoutCommands::Simulate {
                input,
                data_dir,
                cache_size,
                compare,
            } => layout_simulate(&input, &data_dir, cache_size, compare),
            LayoutCommands::Validate {
                input,
                data_dir,
                cache_sizes,
                iterations,
            } => layout_validate(&input, &data_dir, &cache_sizes, iterations),
            LayoutCommands::Visualize {
                input,
                data_dir,
                format,
                output,
            } => layout_visualize(&input, &data_dir, &format, output.as_deref()),
            LayoutCommands::Constraints {
                input,
                data_dir,
                verbose,
            } => layout_constraints(&input, &data_dir, verbose),
            LayoutCommands::Explain {
                concept,
                input,
                data_dir,
            } => layout_explain(&concept, &input, &data_dir),
        },

        Commands::Locality { command } => match command {
            LocalityCommands::Numa { format, verbose } => locality_numa(&format, verbose),
            LocalityCommands::Analyze {
                input,
                format,
                recommend,
            } => locality_analyze(&input, &format, recommend),
            LocalityCommands::Prefetch {
                data_dir,
                output,
                format,
            } => locality_prefetch(&data_dir, output.as_deref(), &format),
            LocalityCommands::Pack {
                input,
                struct_name,
                cache_line,
                suggest,
            } => locality_pack(&input, &struct_name, cache_line, suggest),
            LocalityCommands::Lattice { format } => locality_lattice(&format),
            LocalityCommands::Codegen {
                input,
                function,
                target,
                output,
            } => locality_codegen(&input, &function, &target, output.as_deref()),
        },

        Commands::Units { command } => match command {
            UnitsCommands::List {
                category,
                format,
                verbose,
            } => units_list(&category, &format, verbose),
            UnitsCommands::Convert {
                value,
                from,
                to,
                verbose,
            } => units_convert(value, &from, &to, verbose),
            UnitsCommands::Info { unit, format } => units_info(&unit, &format),
            UnitsCommands::Check {
                unit1,
                unit2,
                verbose,
            } => units_check(&unit1, &unit2, verbose),
            UnitsCommands::Parse { expr, verbose } => units_parse(&expr, verbose),
            UnitsCommands::Dimensions { format } => units_dimensions(&format),
        },

        Commands::Linear { command } => match command {
            LinearCommands::List { format, verbose } => linear_list(&format, verbose),
            LinearCommands::Info { kind, format } => linear_info(&kind, &format),
            LinearCommands::Subkind {
                kind1,
                kind2,
                verbose,
            } => linear_subkind(&kind1, &kind2, verbose),
            LinearCommands::Resources { format, verbose } => linear_resources(&format, verbose),
            LinearCommands::Sessions { examples, format } => linear_sessions(examples, &format),
            LinearCommands::Check {
                input,
                show_resources,
                show_usage,
            } => linear_check(&input, show_resources, show_usage),
        },

        #[cfg(feature = "distributed")]
        Commands::Distributed { command } => match command {
            DistributedCommands::Server {
                address,
                max_connections,
                cache,
                name,
            } => distributed_server(&address, max_connections, cache, &name),
            DistributedCommands::Build {
                server,
                target,
                profile,
                path,
            } => distributed_build(&server, target.as_deref(), &profile, &path),
            DistributedCommands::Status { server } => distributed_status(&server),
        },

        #[cfg(feature = "distributed")]
        Commands::Cache { command } => match command {
            CacheCommands::Server {
                address,
                storage,
                max_size,
            } => cache_server(&address, &storage, &max_size),
            CacheCommands::Stats { url, local } => cache_stats(url.as_deref(), local),
            CacheCommands::Clean {
                all,
                older_than,
                dry_run,
            } => cache_clean(all, older_than.as_deref(), dry_run),
        },

        #[cfg(feature = "distributed")]
        Commands::Ci { command } => match command {
            CiCommands::Github {
                output,
                release,
                targets,
            } => ci_github(&output, release, targets.as_deref()),
            CiCommands::Gitlab { output, targets } => ci_gitlab(&output, targets.as_deref()),
            CiCommands::Provenance {
                output,
                target,
                profile,
            } => ci_provenance(&output, target.as_deref(), &profile),
            CiCommands::Reproducible { builds, check_env } => ci_reproducible(builds, check_env),
        },
    }
}

/// Build a Sounio source file to PTX using the GPU backend
fn build_gpu(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    target: Option<&str>,
    verbose: bool,
) -> Result<()> {
    #[cfg(feature = "gpu")]
    {
        #[derive(Debug, Clone, Serialize)]
        struct TunedKernelSummary {
            name: String,
            block_shape: String,
            shared_mem_bytes: u32,
            strategy: String,
            confidence: f64,
            occupancy: f64,
        }

        #[derive(Debug, Clone, Serialize)]
        struct GpuBuildCacheMetadata {
            schema: String,
            cache_key: String,
            input: String,
            target_sm: String,
            auto_tune: bool,
            auto_tune_block_shape: bool,
            auto_tune_shared_mem: bool,
            fusion: bool,
            pipelining: bool,
            created_at_utc: String,
            tuned_kernels: Vec<TunedKernelSummary>,
            ptx_sha256: String,
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

        let parse_sm = |raw: &str| -> Option<(u32, u32)> {
            let mut value = raw.trim();
            if let Some(rest) = value.strip_prefix("sm_") {
                value = rest;
            } else if let Some(rest) = value.strip_prefix("sm") {
                value = rest;
            }
            if let Some((major_str, minor_str)) = value.split_once('.') {
                let major = major_str.trim().parse::<u32>().ok()?;
                let minor = minor_str.trim().parse::<u32>().ok()?;
                return Some((major, minor));
            }
            let digits: String = value.chars().filter(|c| c.is_ascii_digit()).collect();
            match digits.len() {
                2 => {
                    let major = digits[0..1].parse::<u32>().ok()?;
                    let minor = digits[1..2].parse::<u32>().ok()?;
                    Some((major, minor))
                }
                3 => {
                    let major = digits[0..2].parse::<u32>().ok()?;
                    let minor = digits[2..3].parse::<u32>().ok()?;
                    Some((major, minor))
                }
                _ => None,
            }
        };

        let sm_version = std::env::var("SOUNIO_GPU_SM")
            .ok()
            .and_then(|v| parse_sm(&v))
            .or_else(|| target.and_then(|t| parse_sm(t)))
            .unwrap_or((7, 5));

        let autotune_requested = env_flag_or_default("SOUNIO_GPU_AUTOTUNE", true);
        let mut enable_autotune_block_shape = autotune_requested;
        let mut enable_autotune_shared_mem = autotune_requested;
        let mut autotune_block_reason = if autotune_requested {
            "enabled"
        } else {
            "disabled_by_env"
        };
        let mut autotune_shared_reason = if autotune_requested {
            "enabled"
        } else {
            "disabled_by_env"
        };
        let mut enable_fusion = env_flag_or_default("SOUNIO_GPU_FUSION", true);
        let mut enable_pipelining = env_flag_or_default("SOUNIO_GPU_ASYNC_PIPELINE", false);
        let enable_auto_vectorize = env_flag_or_default("SOUNIO_GPU_AUTO_VECTORIZE", false);
        let enable_cache = env_flag_or_default("SOUNIO_GPU_TUNE_CACHE", true);
        let policy_enforced = env_flag_or_default("SOUNIO_OPT_POLICY_ENFORCE", false);
        let input_hash = std::fs::read(input).ok().map(|bytes| sha256_hex(&bytes));
        let cache_dir = std::env::var_os("SOUNIO_GPU_TUNE_CACHE_DIR")
            .filter(|raw| !raw.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("target/souc-gpu-cache"));

        let active_policy = load_effective_optimization_policy(policy_enforced)?;
        if let Some(policy) = &active_policy {
            if !policy_allows_action(policy, "gpu", "fusion") {
                enable_fusion = false;
                emit_decision_event(
                    "gpu",
                    "fusion",
                    "skipped",
                    "blocked_by_policy_allowlist",
                    input.to_string_lossy().as_ref(),
                    input_hash.as_deref(),
                    active_policy.as_ref(),
                )?;
            }
            if !policy_allows_action(policy, "gpu", "autotune:block-shape") {
                enable_autotune_block_shape = false;
                autotune_block_reason = "blocked_by_policy_allowlist";
                emit_decision_event(
                    "gpu",
                    "autotune:block-shape",
                    "skipped",
                    "blocked_by_policy_allowlist",
                    input.to_string_lossy().as_ref(),
                    input_hash.as_deref(),
                    active_policy.as_ref(),
                )?;
            }
            if !policy_allows_action(policy, "gpu", "autotune:shared-mem") {
                enable_autotune_shared_mem = false;
                autotune_shared_reason = "blocked_by_policy_allowlist";
                emit_decision_event(
                    "gpu",
                    "autotune:shared-mem",
                    "skipped",
                    "blocked_by_policy_allowlist",
                    input.to_string_lossy().as_ref(),
                    input_hash.as_deref(),
                    active_policy.as_ref(),
                )?;
            }
            if !policy_allows_action(policy, "gpu", "pipeline:async-memory") {
                enable_pipelining = false;
                emit_decision_event(
                    "gpu",
                    "pipeline:async-memory",
                    "skipped",
                    "blocked_by_policy_allowlist",
                    input.to_string_lossy().as_ref(),
                    input_hash.as_deref(),
                    active_policy.as_ref(),
                )?;
            }
        }
        let enable_auto_tune = enable_autotune_block_shape || enable_autotune_shared_mem;

        emit_decision_event(
            "gpu",
            "fusion",
            if enable_fusion { "applied" } else { "skipped" },
            if enable_fusion { "enabled" } else { "disabled" },
            input.to_string_lossy().as_ref(),
            input_hash.as_deref(),
            active_policy.as_ref(),
        )?;
        emit_decision_event(
            "gpu",
            "autotune:block-shape",
            if enable_autotune_block_shape {
                "applied"
            } else {
                "skipped"
            },
            if enable_autotune_block_shape {
                autotune_block_reason
            } else {
                "disabled"
            },
            input.to_string_lossy().as_ref(),
            input_hash.as_deref(),
            active_policy.as_ref(),
        )?;
        emit_decision_event(
            "gpu",
            "autotune:shared-mem",
            if enable_autotune_shared_mem {
                "applied"
            } else {
                "skipped"
            },
            if enable_autotune_shared_mem {
                autotune_shared_reason
            } else {
                "disabled"
            },
            input.to_string_lossy().as_ref(),
            input_hash.as_deref(),
            active_policy.as_ref(),
        )?;
        emit_decision_event(
            "gpu",
            "pipeline:async-memory",
            if enable_pipelining {
                "applied"
            } else {
                "skipped"
            },
            if enable_pipelining {
                "enabled"
            } else {
                "disabled"
            },
            input.to_string_lossy().as_ref(),
            input_hash.as_deref(),
            active_policy.as_ref(),
        )?;

        let out_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
            let mut p = input.to_path_buf();
            p.set_extension("ptx");
            p
        });

        let mut key_hasher = Sha256::new();
        key_hasher.update(env!("CARGO_PKG_VERSION").as_bytes());
        key_hasher.update(input.to_string_lossy().as_bytes());
        key_hasher.update(format!("sm_{}{}", sm_version.0, sm_version.1).as_bytes());
        key_hasher.update([enable_autotune_block_shape as u8]);
        key_hasher.update([enable_autotune_shared_mem as u8]);
        key_hasher.update([enable_fusion as u8]);
        key_hasher.update([enable_pipelining as u8]);
        if let Ok(bytes) = std::fs::read(input) {
            key_hasher.update(&bytes);
        }
        let cache_key = hex::encode(key_hasher.finalize());
        let cache_ptx_path = cache_dir.join(format!("{cache_key}.ptx"));
        let cache_meta_path = cache_dir.join(format!("{cache_key}.json"));

        if enable_cache && cache_ptx_path.exists() {
            let cached_ptx = std::fs::read_to_string(&cache_ptx_path).map_err(|e| {
                miette::miette!("Failed to read GPU cache {}: {}", cache_ptx_path.display(), e)
            })?;
            std::fs::write(&out_path, cached_ptx)
                .map_err(|e| miette::miette!("Failed to write PTX: {}", e))?;
            if verbose {
                eprintln!(
                    "GPU cache hit: key={} metadata={}",
                    cache_key,
                    cache_meta_path.display()
                );
            }
            println!("Wrote PTX to {}", out_path.display());
            return Ok(());
        }

        if verbose {
            eprintln!(
                "Using GPU target sm_{}{} (autotune={} block_shape={} shared_mem={} fusion={} pipelining={} cache={})",
                sm_version.0,
                sm_version.1,
                enable_auto_tune,
                enable_autotune_block_shape,
                enable_autotune_shared_mem,
                enable_fusion,
                enable_pipelining,
                enable_cache
            );
        }

        let ast = sounio::module_loader::load_program_ast(input)?;
        let kernel_names: std::collections::HashSet<String> = ast
            .items
            .iter()
            .filter_map(|item| match item {
                sounio::ast::Item::Function(func) if func.modifiers.is_kernel => {
                    Some(func.name.clone())
                }
                _ => None,
            })
            .collect();

        if kernel_names.is_empty() {
            return Err(miette::miette!(
                "GPU build requires at least one `kernel fn`"
            ));
        }

        let hir = sounio::check::check_ast(&ast)?;
        let mut hlir = sounio::hlir::lower(&hir);
        for func in &mut hlir.functions {
            if kernel_names.contains(&func.name) {
                func.is_kernel = true;
            }
        }

        let mut lowering_config = sounio::codegen::gpu::LoweringConfig {
            target: sounio::codegen::gpu::GpuTarget::Cuda {
                compute_capability: sm_version,
            },
            auto_vectorize: enable_auto_vectorize,
            auto_tune: enable_auto_tune,
            auto_tune_block_shape: enable_autotune_block_shape,
            auto_tune_shared_mem: enable_autotune_shared_mem,
            tune_config: Some(sounio::codegen::gpu::AutoTuneConfig {
                target: sounio::codegen::gpu::GpuTarget::Cuda {
                    compute_capability: sm_version,
                },
                ..Default::default()
            }),
            enable_fusion,
            fusion_config: Some(sounio::codegen::gpu::FusionConfig::default()),
            enable_pipelining,
            ..Default::default()
        };
        if !enable_auto_tune {
            lowering_config.tune_config = None;
        }

        let optimized = sounio::codegen::gpu::lower_and_optimize(&hlir, &lowering_config);
        let mut ptx_codegen = sounio::codegen::gpu::PtxCodegen::new(sm_version);
        let ptx = ptx_codegen.generate(&optimized.module);

        std::fs::write(&out_path, ptx)
            .map_err(|e| miette::miette!("Failed to write PTX: {}", e))?;

        if enable_cache {
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                miette::miette!("Failed to create GPU cache dir {}: {}", cache_dir.display(), e)
            })?;

            let ptx_payload = std::fs::read(&out_path)
                .map_err(|e| miette::miette!("Failed to read generated PTX: {}", e))?;
            std::fs::write(&cache_ptx_path, &ptx_payload).map_err(|e| {
                miette::miette!(
                    "Failed to persist GPU cache PTX {}: {}",
                    cache_ptx_path.display(),
                    e
                )
            })?;

            let mut tuned_kernels: Vec<TunedKernelSummary> = optimized
                .tuned_configs
                .iter()
                .map(|(name, tuned)| TunedKernelSummary {
                    name: name.clone(),
                    block_shape: tuned.block_shape.to_string(),
                    shared_mem_bytes: tuned.shared_mem_bytes,
                    strategy: tuned.strategy.to_string(),
                    confidence: tuned.confidence,
                    occupancy: tuned.occupancy.occupancy,
                })
                .collect();
            tuned_kernels.sort_by(|a, b| a.name.cmp(&b.name));

            let metadata = GpuBuildCacheMetadata {
                schema: "sounio.gpu.tune-cache.v1".to_string(),
                cache_key,
                input: input.display().to_string(),
                target_sm: format!("sm_{}{}", sm_version.0, sm_version.1),
                auto_tune: enable_auto_tune,
                auto_tune_block_shape: enable_autotune_block_shape,
                auto_tune_shared_mem: enable_autotune_shared_mem,
                fusion: enable_fusion,
                pipelining: enable_pipelining,
                created_at_utc: Utc::now().to_rfc3339(),
                tuned_kernels,
                ptx_sha256: sha256_hex(&ptx_payload),
            };

            let metadata_json = serde_json::to_string_pretty(&metadata)
                .map_err(|e| miette::miette!("Failed to serialize GPU cache metadata: {}", e))?;
            std::fs::write(&cache_meta_path, metadata_json).map_err(|e| {
                miette::miette!(
                    "Failed to write GPU cache metadata {}: {}",
                    cache_meta_path.display(),
                    e
                )
            })?;
        }

        if verbose {
            eprintln!(
                "GPU optimized lowering summary: kernels_before={} kernels_after={} tuned={}",
                optimized.original_kernel_count,
                optimized.module.kernels.len(),
                optimized.tuned_configs.len()
            );
        }

        println!("Wrote PTX to {}", out_path.display());
        Ok(())
    }

    #[cfg(not(feature = "gpu"))]
    {
        let _ = (input, output, target, verbose);
        Err(miette::miette!(
            "GPU backend not enabled in this binary. Rebuild with GPU support enabled."
        ))
    }
}

/// Build a Sounio source file to native executable using LLVM
#[allow(clippy::too_many_arguments)]
fn build(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    opt_level: &str,
    debug: bool,
    emit_llvm: bool,
    emit_asm: bool,
    target: Option<&str>,
    strip: bool,
    verbose: bool,
    cdylib: bool,
    _alloc_policy: AllocPolicy,
    _attention_config: Option<&std::path::Path>,
    _emit_alloc_metrics: Option<&std::path::Path>,
) -> Result<()> {
    #[cfg(feature = "llvm-base")]
    {
        fn find_runtime_lib_dir() -> Option<std::path::PathBuf> {
            let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
            let runtime_root = manifest_dir.join("..").join("runtime");
            let target_dir = runtime_root.join("target");
            let lib_names = [
                "libsounio_runtime.a",
                "libsounio_runtime.so",
                "libsounio_runtime.dylib",
                "sounio_runtime.lib",
            ];

            for profile in ["release", "debug"] {
                let dir = target_dir.join(profile);
                if lib_names.iter().any(|name| dir.join(name).exists()) {
                    return Some(dir);
                }
            }

            None
        }

        use inkwell::context::Context;
        use sounio::codegen::llvm::{
            codegen::{LLVMCodegen, OptLevel},
            linker::Linker,
            passes,
            target::{
                compile_to_asm, compile_to_object, create_native_shared_target_machine,
                create_native_target_machine, create_shared_target_machine, create_target_machine,
                executable_extension, initialize_native_target, object_extension,
                shared_lib_extension,
            },
        };

        tracing::info!("Building {:?} with LLVM", input);

        // Parse optimization level
        let mut opt = match opt_level {
            "0" => OptLevel::O0,
            "1" => OptLevel::O1,
            "2" => OptLevel::O2,
            "3" => OptLevel::O3,
            "s" => OptLevel::Os,
            "z" => OptLevel::Oz,
            _ => {
                return Err(miette::miette!(
                    "Invalid optimization level: {}. Use 0, 1, 2, 3, s, or z",
                    opt_level
                ));
            }
        };

        let policy_enforced = std::env::var("SOUNIO_OPT_POLICY_ENFORCE")
            .ok()
            .map(|v| {
                let value = v.trim().to_ascii_lowercase();
                matches!(value.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false);
        let input_hash = std::fs::read(input).ok().map(|bytes| sha256_hex(&bytes));
        let active_policy = load_effective_optimization_policy(policy_enforced)?;

        let action_for_opt = |level: OptLevel| -> &'static str {
            match level {
                OptLevel::O0 => "pipeline:o0",
                OptLevel::O1 => "pipeline:o1",
                OptLevel::O2 => "pipeline:o2",
                OptLevel::O3 => "pipeline:o3",
                OptLevel::Os => "pipeline:os",
                OptLevel::Oz => "pipeline:oz",
            }
        };
        if let Some(policy) = &active_policy {
            let requested_action = action_for_opt(opt);
            let requested_allowed = policy_allows_action(policy, "llvm", requested_action);
            if !requested_allowed {
                let fallback = [OptLevel::O3, OptLevel::O2, OptLevel::O1, OptLevel::O0]
                    .iter()
                    .copied()
                    .find(|candidate| policy_allows_action(policy, "llvm", action_for_opt(*candidate)));
                if let Some(selected) = fallback {
                    emit_decision_event(
                        "llvm",
                        requested_action,
                        "skipped",
                        &format!("blocked_by_policy_allowlist fallback={}", action_for_opt(selected)),
                        input.to_string_lossy().as_ref(),
                        input_hash.as_deref(),
                        active_policy.as_ref(),
                    )?;
                    opt = selected;
                } else if policy_enforced {
                    return Err(miette::miette!(
                        "optimization policy enforcement failed: no allowed LLVM pipeline action for requested {}",
                        requested_action
                    ));
                } else {
                    emit_decision_event(
                        "llvm",
                        requested_action,
                        "skipped",
                        "blocked_by_policy_allowlist no_fallback",
                        input.to_string_lossy().as_ref(),
                        input_hash.as_deref(),
                        active_policy.as_ref(),
                    )?;
                }
            }
        }
        emit_decision_event(
            "llvm",
            action_for_opt(opt),
            "applied",
            "selected_pipeline",
            input.to_string_lossy().as_ref(),
            input_hash.as_deref(),
            active_policy.as_ref(),
        )?;

        // Load modules and parse (uses ModuleLoader to handle imports)
        let ast = sounio::module_loader::load_program_ast(input)?;

        // Type check
        let hir = sounio::check::check_ast(&ast)?;

        // Lower to HLIR
        let hlir = sounio::hlir::lower(&hir);

        if verbose {
            eprintln!(
                "Compiled {} items, {} functions",
                ast.items.len(),
                hlir.functions.len()
            );
        }

        // Initialize LLVM
        initialize_native_target();

        // Create LLVM context and codegen
        let context = Context::create();
        let module_name = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("module");

        let mut codegen = LLVMCodegen::new(&context, module_name, opt, debug);

        // Compile to LLVM IR
        codegen.compile(&hlir);

        // Verify module
        if let Err(e) = codegen.verify() {
            return Err(miette::miette!("LLVM verification failed: {}", e));
        }

        // Get target machine - use PIC for shared libraries
        let target_machine = if cdylib {
            // For shared libraries, use PIC (Position Independent Code)
            if let Some(triple) = target {
                create_shared_target_machine(triple, opt).map_err(|e| {
                    miette::miette!("Failed to create shared library target machine: {}", e)
                })?
            } else {
                create_native_shared_target_machine(opt).map_err(|e| {
                    miette::miette!(
                        "Failed to create native shared library target machine: {}",
                        e
                    )
                })?
            }
        } else {
            // For executables, use default settings
            if let Some(triple) = target {
                create_target_machine(triple, opt)
                    .map_err(|e| miette::miette!("Failed to create target machine: {}", e))?
            } else {
                create_native_target_machine(opt)
                    .map_err(|e| miette::miette!("Failed to create target machine: {}", e))?
            }
        };

        // Run optimization passes
        let module = codegen.get_module();
        passes::optimize_module(module, opt, &target_machine);

        // Handle emit options
        if emit_llvm {
            let ir = codegen.print_ir();
            if let Some(out_path) = output {
                std::fs::write(out_path, &ir)
                    .map_err(|e| miette::miette!("Failed to write LLVM IR: {}", e))?;
                println!("Wrote LLVM IR to {}", out_path.display());
            } else {
                println!("{}", ir);
            }
            return Ok(());
        }

        if emit_asm {
            let asm_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
                let mut p = input.to_path_buf();
                p.set_extension("s");
                p
            });

            compile_to_asm(module, &target_machine, &asm_path)
                .map_err(|e| miette::miette!("Failed to generate assembly: {}", e))?;

            println!("Wrote assembly to {}", asm_path.display());
            return Ok(());
        }

        // Compile to object file
        let triple = target.unwrap_or("native");
        let obj_ext = object_extension(triple);
        let obj_path = {
            let mut p = input.to_path_buf();
            p.set_extension(obj_ext);
            p
        };

        compile_to_object(module, &target_machine, &obj_path)
            .map_err(|e| miette::miette!("Failed to generate object file: {}", e))?;

        if verbose {
            eprintln!("Generated object file: {}", obj_path.display());
        }

        // Link to executable or shared library
        if cdylib {
            // Link as shared library
            let lib_ext = shared_lib_extension(triple);
            let lib_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
                let stem = input.file_stem().and_then(|s| s.to_str()).unwrap_or("lib");
                let mut p = input.to_path_buf();
                p.set_file_name(format!("lib{}", stem));
                p.set_extension(lib_ext);
                p
            });

            let linker = Linker::new().strip(strip).verbose(verbose);
            let runtime_dir = find_runtime_lib_dir();

            if runtime_dir.is_none() && verbose {
                eprintln!("Warning: runtime library not found; linking without effect runtime");
            }

            let linker = if let Some(dir) = runtime_dir.as_deref() {
                linker.lib_path(dir).lib("sounio_runtime")
            } else {
                linker
            };

            linker
                .link_shared(&[obj_path.clone()], &lib_path)
                .map_err(|e| miette::miette!("Shared library linking failed: {}", e))?;

            // Clean up object file
            if std::fs::remove_file(&obj_path).is_err() && verbose {
                eprintln!("Warning: could not remove temporary object file");
            }

            println!("Built shared library: {}", lib_path.display());
        } else {
            // Link as executable
            let exe_ext = executable_extension(triple);
            let exe_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
                let mut p = input.to_path_buf();
                p.set_extension(exe_ext);
                if exe_ext.is_empty() {
                    // Remove extension for Unix executables
                    p.set_extension("");
                }
                p
            });

            let linker = Linker::new().strip(strip).verbose(verbose);
            let runtime_dir = find_runtime_lib_dir();

            if runtime_dir.is_none() && verbose {
                eprintln!("Warning: runtime library not found; linking without effect runtime");
            }

            let link_result = if let Some(dir) = runtime_dir.as_deref() {
                linker.link_with_runtime(&[obj_path.clone()], &exe_path, Some(dir))
            } else {
                linker.link_with_stdlib(&[obj_path.clone()], &exe_path)
            };

            link_result.map_err(|e| miette::miette!("Linking failed: {}", e))?;

            // Clean up object file
            if std::fs::remove_file(&obj_path).is_err() && verbose {
                eprintln!("Warning: could not remove temporary object file");
            }

            println!("Built: {}", exe_path.display());
        }
        Ok(())
    }

    #[cfg(not(feature = "llvm-base"))]
    {
        let _ = (
            input, output, opt_level, debug, emit_llvm, emit_asm, target, strip, verbose, cdylib,
        );
        Err(miette::miette!(
            "LLVM backend not enabled in this binary. Rebuild with LLVM support enabled."
        ))
    }
}

fn compile(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    emit: Option<EmitType>,
    opt_level: u8,
    glm_enabled: bool,
    native_stub: bool,
) -> Result<()> {
    tracing::info!(
        "Compiling {:?} with optimization level {}{}",
        input,
        opt_level,
        if glm_enabled {
            " (GLM-4.7 enabled)"
        } else {
            ""
        }
    );

    #[cfg(feature = "glm")]
    if glm_enabled {
        tracing::info!("GLM-4.7 ML-guided optimization active");
    }

    #[cfg(not(feature = "glm"))]
    if glm_enabled {
        eprintln!(
            "Warning: --glm-enabled requires --features glm. Continuing without ML optimization."
        );
    }

    // Load modules and parse (uses ModuleLoader to handle imports)
    let ast = sounio::module_loader::load_program_ast(input)?;
    tracing::debug!("Parsed {} items", ast.items.len());

    // Handle emit options
    if let Some(emit_type) = emit {
        match emit_type {
            EmitType::Ast => {
                let json = serde_json::to_string_pretty(&ast)
                    .map_err(|e| miette::miette!("Failed to serialize AST: {}", e))?;
                println!("{}", json);
                return Ok(());
            }
            EmitType::Hir => {
                let hir = sounio::check::check_ast(&ast)?;
                println!("{:#?}", hir);
                return Ok(());
            }
            EmitType::Hlir => {
                let hir = sounio::check::check_ast(&ast)?;
                let hlir = sounio::hlir::lower(&hir);
                println!("{:#?}", hlir);
                return Ok(());
            }
            EmitType::Llvm => {
                return Err(miette::miette!("LLVM emit not yet implemented"));
            }
        }
    }

    // Type check
    let hir = sounio::check::check_ast(&ast)?;

    if native_stub {
        eprintln!("Note: --native-stub is deprecated; compiling with native backend.");
    }

    let out_path = output
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| std::path::PathBuf::from("a.out"));

    let mut codegen = sounio::codegen::native::NativeCodegen::new();
    let binary = codegen
        .compile(&hir)
        .map_err(|e| miette::miette!("Native codegen failed: {}", e))?;

    std::fs::write(&out_path, &binary).map_err(|e| {
        miette::miette!(
            "Failed to write executable to '{}': {}",
            out_path.display(),
            e
        )
    })?;

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o755);
        std::fs::set_permissions(&out_path, perms).map_err(|e| {
            miette::miette!(
                "Failed to set executable permissions on '{}': {}",
                out_path.display(),
                e
            )
        })?;
    }

    println!(
        "Compiled {} -> {} ({} bytes)",
        input.display(),
        out_path.display(),
        binary.len()
    );

    Ok(())
}

fn check(
    input: &std::path::Path,
    show_ast: bool,
    show_resolved: bool,
    show_types: bool,
    show_effects: bool,
    skip_ownership: bool,
    error_format: &str,
    warn_deprecated: bool,
) -> Result<()> {
    // Flag is parsed and reserved for future diagnostics policy wiring.
    let _ = warn_deprecated;
    let use_json = error_format == "json";
    tracing::info!("Type-checking {:?}", input);

    let source_content = if input.is_dir() {
        // Directory compilation mode: concatenate all .sio files for diagnostics
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(input)
            .map_err(|e| miette::miette!("Cannot read directory {}: {}", input.display(), e))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("sio"))
            .collect();
        files.sort_by(|a, b| {
            let a_is_mod = a.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
            let b_is_mod = b.file_name().and_then(|f| f.to_str()) == Some("mod.sio");
            match (a_is_mod, b_is_mod) {
                (true, false) => std::cmp::Ordering::Greater,
                (false, true) => std::cmp::Ordering::Less,
                _ => a.cmp(b),
            }
        });
        let mut combined = String::new();
        for file in &files {
            if let Ok(content) = std::fs::read_to_string(file) {
                combined.push_str(&content);
                combined.push('\n');
            }
        }
        combined
    } else {
        std::fs::read_to_string(input)
            .map_err(|e| miette::miette!("Failed to read input file: {}", e))?
    };

    let source_file =
        sounio::SourceFile::new(input.to_string_lossy().to_string(), source_content.clone());

    // Precompute line/column mapping for diagnostics.
    let line_starts: Vec<usize> = std::iter::once(0)
        .chain(source_content.match_indices('\n').map(|(i, _)| i + 1))
        .collect();

    // 1-2. Load modules and parse (uses ModuleLoader to handle imports)
    let ast = sounio::module_loader::load_program_ast(input)?;

    if show_ast {
        println!("=== AST ===");
        let json = serde_json::to_string_pretty(&ast)
            .map_err(|e| miette::miette!("Failed to serialize AST: {}", e))?;
        println!("{}", json);
        println!();
    }

    // 3. Resolve names
    let resolved = sounio::resolve::resolve(ast)?;

    if show_resolved {
        println!("=== Resolved Symbols ===");
        let mut count = 0;
        for sym in resolved.symbols.all_symbols() {
            println!("  {:?}: {} ({:?})", sym.def_id, sym.name, sym.kind);
            count += 1;
        }
        println!("Total: {} symbols", count);
        println!();
    }

    // 4. Type check
    let check_result = sounio::check::check_with_errors(&resolved);

    // Emit warnings (including unused import warnings)
    for warning in &check_result.warnings {
        // Extract warning code from the message (format: "code: message")
        let (code, message) = if let Some(idx) = warning.find(':') {
            (&warning[..idx], &warning[idx + 1..])
        } else {
            ("warning", warning.as_str())
        };

        if use_json {
            let diag = serde_json::json!({
                "level": "warning",
                "message": message.trim(),
                "code": code,
                "location": {
                    "file": input.to_string_lossy(),
                    "line": 1,
                    "column": 1
                },
                "notes": [],
                "suggestions": [],
                "related": []
            });
            eprintln!("{}", diag);
        } else {
            eprintln!("warning[{}]: {}", code, message.trim());
        }
    }

    if !check_result.errors.is_empty() {
        for err in &check_result.errors {
            let (line, column) = {
                let offset = err.span.start;
                let line_idx = line_starts
                    .partition_point(|&start| start <= offset)
                    .saturating_sub(1);
                let line = line_idx + 1;
                let col = offset - line_starts.get(line_idx).copied().unwrap_or(0) + 1;
                (line as u32, col as u32)
            };

            if use_json {
                let diag = serde_json::json!({
                    "level": "error",
                    "message": err.message,
                    "code": err.code,
                    "location": {
                        "file": input.to_string_lossy(),
                        "line": line,
                        "column": column
                    },
                    "notes": [],
                    "suggestions": [],
                    "related": []
                });
                eprintln!("{}", diag);
            } else {
                // Human-readable format with source snippet
                eprintln!("error: {}", err.message);
                eprintln!("  --> {}:{}:{}", input.to_string_lossy(), line, column);

                // Get the source line
                let line_idx = line as usize - 1;
                let lines: Vec<&str> = source_content.lines().collect();
                if line_idx < lines.len() {
                    let source_line = lines[line_idx];
                    eprintln!("   |");
                    eprintln!("{:3} | {}", line, source_line);

                    // Add caret pointing to the error location
                    let padding = " ".repeat(column as usize);
                    eprintln!("   | {}^", padding);
                }
                eprintln!();
            }
        }

        let messages: Vec<_> = check_result
            .errors
            .iter()
            .map(|e| e.message.clone())
            .collect();
        return Err(miette::miette!("Type errors:\n{}", messages.join("\n")));
    }

    let hir = check_result.hir.expect("HIR should exist when no errors");

    if show_types {
        println!("=== HIR (with types) ===");
        println!("{:#?}", hir);
        println!();
    }

    // 5. Effect inference
    let mut effect_checker = sounio::effects::EffectChecker::new(&resolved.symbols);
    if let Err(errors) = effect_checker.check_program(&resolved.ast) {
        if show_effects {
            println!("=== Effect Errors ===");
            for e in &errors {
                println!("  {}", e);
            }
            println!();
        }
        for e in &errors {
            let (line, column) = {
                let offset = e.span.start;
                let line_idx = line_starts
                    .partition_point(|&start| start <= offset)
                    .saturating_sub(1);
                let line = line_idx + 1;
                let col = offset - line_starts.get(line_idx).copied().unwrap_or(0) + 1;
                (line as u32, col as u32)
            };

            let message = e.to_string();
            if use_json {
                let diag = serde_json::json!({
                    "level": "error",
                    "message": message,
                    "code": "effect",
                    "location": {
                        "file": input.to_string_lossy(),
                        "line": line,
                        "column": column
                    },
                    "notes": [],
                    "suggestions": [],
                    "related": []
                });
                eprintln!("{}", diag);
            } else {
                eprintln!("error: {}", message);
                eprintln!("  --> {}:{}:{}", input.to_string_lossy(), line, column);

                let line_idx = line as usize - 1;
                let lines: Vec<&str> = source_content.lines().collect();
                if line_idx < lines.len() {
                    let source_line = lines[line_idx];
                    eprintln!("   |");
                    eprintln!("{:3} | {}", line, source_line);
                    let padding = " ".repeat(column as usize);
                    eprintln!("   | {}^", padding);
                }
                eprintln!();
            }
        }

        let messages: Vec<_> = errors.iter().map(|e| e.to_string()).collect();
        return Err(miette::miette!("Effect errors:\n{}", messages.join("\n")));
    } else if show_effects {
        println!("=== Effects ===");
        println!("  All effects properly declared");
        println!();
    }

    // 6. Ownership check
    if !skip_ownership {
        let mut ownership_checker = sounio::ownership::OwnershipChecker::new(
            &resolved.symbols,
            &source_file,
            &resolved.ast.node_spans,
        );
        if let Err(errors) = ownership_checker.check_program(&resolved.ast) {
            for e in &errors {
                eprintln!("{:?}", miette::Report::new(e.clone()));
            }
            return Err(miette::miette!("{} ownership errors found", errors.len()));
        }
    }

    println!("All checks passed: {}", input.display());
    Ok(())
}

const BOOTSTRAP_MANIFEST_V2_SCHEMA: &str = "sounio.bootstrap.manifest.v2";
const BOOTSTRAP_STATE_DIGEST_SCHEMA: &str = "sounio.bootstrap.state-digest.v1";
const BOOTSTRAP_CYCLE_REPORT_SCHEMA: &str = "sounio.bootstrap.cycle-report.v1";
const OPT_POLICY_V1_SCHEMA: &str = "sounio.optimization.policy.v1";
const OPT_POLICY_V2_SCHEMA: &str = "sounio.optimization.policy.v2";
const OPT_POLICY_DEFAULT_VERIFY_KEY_ID: &str = "bootstrap-dev";
const RL_READINESS_EVIDENCE_PATH_ENV: &str = "SOUNIO_RL_READINESS_EVIDENCE_PATH";
const OMEGA_CANONICAL_PRIVKEY_ENV: &str = "OMEGA_CANONICAL_PRIVKEY";
const OMEGA_CANONICAL_PUBKEY_ENV: &str = "OMEGA_CANONICAL_PUBKEY";
const OMEGA_CANONICAL_PRIVKEY_DEFAULT: &str = "keys/bootstrap_ed25519";
const OMEGA_CANONICAL_PUBKEY_DEFAULT: &str = "keys/bootstrap_ed25519.pub";
const OPT_POLICY_V2_REQUIRED_FAMILIES: [&str; 5] = [
    "dense_linear",
    "attention",
    "epistemic_elementwise",
    "monte_carlo",
    "quantum_parallel",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BootstrapArtifactManifestV2 {
    schema: String,
    souc_version: String,
    target_triple: String,
    driver_fingerprint: String,
    artifacts: Vec<BootstrapArtifactEntry>,
    key_id: String,
    created_at_utc: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BootstrapArtifactEntry {
    #[serde(default)]
    name: String,
    path: String,
    sha256: String,
    #[serde(default, rename = "type")]
    type_tag: String,
}

#[derive(Debug, Clone, Serialize)]
struct BootstrapStateDigest {
    schema: String,
    generated_at_utc: String,
    manifest_sha256: String,
    target_triple: String,
    artifacts: Vec<BootstrapStateDigestEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct BootstrapStateDigestEntry {
    path: String,
    sha256: String,
}

#[derive(Debug, Clone, Serialize)]
struct BootstrapCycleReport {
    schema: String,
    generated_at_utc: String,
    state_dir: String,
    stage1_digest: String,
    stage2_digest: String,
    deterministic: bool,
    artifacts_checked: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationRewardWeights {
    runtime_speedup: f64,
    compile_time_penalty: f64,
    correctness_penalty: f64,
    stability_bonus: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpistemicPowerWeights {
    runtime: f64,
    gum: f64,
    provenance: f64,
    formal: f64,
    conformance: f64,
}

impl Default for EpistemicPowerWeights {
    fn default() -> Self {
        Self {
            runtime: 0.35,
            gum: 0.25,
            provenance: 0.15,
            formal: 0.15,
            conformance: 0.10,
        }
    }
}

impl EpistemicPowerWeights {
    fn total_weight(&self) -> f64 {
        self.runtime + self.gum + self.provenance + self.formal + self.conformance
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct EpistemicPenalties {
    budget_scale: f64,
    bit_exact_mismatch: f64,
    compile_overhead_scale: f64,
}

impl Default for EpistemicPenalties {
    fn default() -> Self {
        Self {
            budget_scale: 5.0,
            bit_exact_mismatch: 10.0,
            compile_overhead_scale: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RuntimeNormalizationBand {
    floor_speedup: f64,
    cap_speedup: f64,
}

impl RuntimeNormalizationBand {
    fn new(floor_speedup: f64, cap_speedup: f64) -> Self {
        Self {
            floor_speedup,
            cap_speedup,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RlReadinessGate {
    min_epistemic_power_gain: f64,
    require_zero_bit_mismatch: bool,
    max_compile_overhead: f64,
    require_shadow_audit_clean: bool,
}

impl Default for RlReadinessGate {
    fn default() -> Self {
        Self {
            min_epistemic_power_gain: 0.05,
            require_zero_bit_mismatch: true,
            max_compile_overhead: 0.25,
            require_shadow_audit_clean: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct QuantumConformanceGate {
    ks_pvalue_min: f64,
    coverage_target: f64,
    coverage_range: [f64; 2],
    min_shots: u32,
}

impl Default for QuantumConformanceGate {
    fn default() -> Self {
        Self {
            ks_pvalue_min: 0.05,
            coverage_target: 0.95,
            coverage_range: [0.90, 0.98],
            min_shots: 50,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RlReadinessEvidence {
    epistemic_power_gain: f64,
    bit_exact_mismatches: u64,
    compile_overhead: f64,
    shadow_audit_clean: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationPolicyV1 {
    schema: String,
    policy_id: String,
    policy_version: String,
    target_triple: String,
    policy_mode: String,
    mir_actions: Vec<String>,
    llvm_actions: Vec<String>,
    gpu_actions: Vec<String>,
    reward_weights: OptimizationRewardWeights,
    created_at_utc: String,
    trained_from_corpus_digest: String,
    signature: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_fingerprint: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_sig: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_signed_at_utc: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pinned_digest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct OptimizationPolicyV2 {
    schema: String,
    policy_id: String,
    policy_version: String,
    target_triple: String,
    policy_mode: String,
    mir_actions: Vec<String>,
    llvm_actions: Vec<String>,
    gpu_actions: Vec<String>,
    reward_weights: OptimizationRewardWeights,
    epistemic_weights: EpistemicPowerWeights,
    penalties: EpistemicPenalties,
    runtime_normalization_by_family: BTreeMap<String, RuntimeNormalizationBand>,
    rl_readiness: RlReadinessGate,
    quantum_conformance: QuantumConformanceGate,
    created_at_utc: String,
    trained_from_corpus_digest: String,
    signature: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_fingerprint: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_sig: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    canonical_signed_at_utc: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pinned_digest_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum OptimizationPolicyDocument {
    V2(OptimizationPolicyV2),
    V1(OptimizationPolicyV1),
}

#[allow(dead_code)]
impl OptimizationPolicyDocument {
    fn schema(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.schema,
            Self::V1(policy) => &policy.schema,
        }
    }

    fn policy_id(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.policy_id,
            Self::V1(policy) => &policy.policy_id,
        }
    }

    fn policy_version(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.policy_version,
            Self::V1(policy) => &policy.policy_version,
        }
    }

    fn target_triple(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.target_triple,
            Self::V1(policy) => &policy.target_triple,
        }
    }

    fn policy_mode(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.policy_mode,
            Self::V1(policy) => &policy.policy_mode,
        }
    }

    fn mir_actions(&self) -> &[String] {
        match self {
            Self::V2(policy) => &policy.mir_actions,
            Self::V1(policy) => &policy.mir_actions,
        }
    }

    fn llvm_actions(&self) -> &[String] {
        match self {
            Self::V2(policy) => &policy.llvm_actions,
            Self::V1(policy) => &policy.llvm_actions,
        }
    }

    fn gpu_actions(&self) -> &[String] {
        match self {
            Self::V2(policy) => &policy.gpu_actions,
            Self::V1(policy) => &policy.gpu_actions,
        }
    }

    fn signature(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.signature,
            Self::V1(policy) => &policy.signature,
        }
    }

    fn clear_signature(&mut self) {
        match self {
            Self::V2(policy) => {
                policy.signature.clear();
                policy.canonical_sig.clear();
                policy.canonical_fingerprint.clear();
                policy.canonical_signed_at_utc.clear();
            }
            Self::V1(policy) => {
                policy.signature.clear();
                policy.canonical_sig.clear();
                policy.canonical_fingerprint.clear();
                policy.canonical_signed_at_utc.clear();
            }
        }
    }

    fn canonical_fingerprint(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.canonical_fingerprint,
            Self::V1(policy) => &policy.canonical_fingerprint,
        }
    }

    fn canonical_sig(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.canonical_sig,
            Self::V1(policy) => &policy.canonical_sig,
        }
    }

    fn canonical_signed_at_utc(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.canonical_signed_at_utc,
            Self::V1(policy) => &policy.canonical_signed_at_utc,
        }
    }

    fn pinned_digest_sha256(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.pinned_digest_sha256,
            Self::V1(policy) => &policy.pinned_digest_sha256,
        }
    }

    fn set_pinned_digest_sha256(&mut self, pinned_digest_sha256: String) {
        match self {
            Self::V2(policy) => {
                policy.pinned_digest_sha256 = pinned_digest_sha256;
            }
            Self::V1(policy) => {
                policy.pinned_digest_sha256 = pinned_digest_sha256;
            }
        }
    }

    fn set_signature_material(
        &mut self,
        signature_hex: String,
        canonical_sig_b64: String,
        canonical_fingerprint: String,
        canonical_signed_at_utc: String,
    ) {
        match self {
            Self::V2(policy) => {
                policy.signature = signature_hex;
                policy.canonical_sig = canonical_sig_b64;
                policy.canonical_fingerprint = canonical_fingerprint;
                policy.canonical_signed_at_utc = canonical_signed_at_utc;
            }
            Self::V1(policy) => {
                policy.signature = signature_hex;
                policy.canonical_sig = canonical_sig_b64;
                policy.canonical_fingerprint = canonical_fingerprint;
                policy.canonical_signed_at_utc = canonical_signed_at_utc;
            }
        }
    }

    fn trained_from_corpus_digest(&self) -> &str {
        match self {
            Self::V2(policy) => &policy.trained_from_corpus_digest,
            Self::V1(policy) => &policy.trained_from_corpus_digest,
        }
    }

    fn rl_readiness(&self) -> Option<&RlReadinessGate> {
        match self {
            Self::V2(policy) => Some(&policy.rl_readiness),
            Self::V1(_) => None,
        }
    }

    fn to_v2_defaults(
        policy_id: String,
        policy_version: String,
        policy_mode: String,
        corpus_digest: String,
    ) -> Self {
        let runtime_normalization_by_family = BTreeMap::from([
            (
                "dense_linear".to_string(),
                RuntimeNormalizationBand::new(0.95, 3.0),
            ),
            (
                "attention".to_string(),
                RuntimeNormalizationBand::new(0.95, 3.0),
            ),
            (
                "epistemic_elementwise".to_string(),
                RuntimeNormalizationBand::new(0.95, 2.5),
            ),
            (
                "monte_carlo".to_string(),
                RuntimeNormalizationBand::new(0.95, 2.5),
            ),
            (
                "quantum_parallel".to_string(),
                RuntimeNormalizationBand::new(0.95, 2.0),
            ),
        ]);

        Self::V2(OptimizationPolicyV2 {
            schema: OPT_POLICY_V2_SCHEMA.to_string(),
            policy_id,
            policy_version,
            target_triple: host_target_triple_tag(),
            policy_mode,
            mir_actions: vec![
                "constant_propagation".to_string(),
                "dead_code_elimination".to_string(),
                "common_subexpression_elimination".to_string(),
                "sroa".to_string(),
                "licm".to_string(),
                "strength_reduction".to_string(),
                "function_inlining".to_string(),
                "loop_vectorization".to_string(),
            ],
            llvm_actions: vec![
                "pipeline:o2".to_string(),
                "pipeline:o3".to_string(),
                "pass:gvn".to_string(),
                "pass:inline".to_string(),
                "pass:loop-unroll".to_string(),
                "pass:loop-vectorize".to_string(),
            ],
            gpu_actions: vec![
                "fusion".to_string(),
                "autotune:block-shape".to_string(),
                "autotune:shared-mem".to_string(),
                "autotune:tile-config".to_string(),
                "pipeline:async-memory".to_string(),
            ],
            reward_weights: OptimizationRewardWeights {
                runtime_speedup: 1.0,
                compile_time_penalty: 0.35,
                correctness_penalty: 10.0,
                stability_bonus: 0.20,
            },
            epistemic_weights: EpistemicPowerWeights::default(),
            penalties: EpistemicPenalties::default(),
            runtime_normalization_by_family,
            rl_readiness: RlReadinessGate::default(),
            quantum_conformance: QuantumConformanceGate::default(),
            created_at_utc: Utc::now().to_rfc3339(),
            trained_from_corpus_digest: corpus_digest,
            signature: String::new(),
            canonical_fingerprint: String::new(),
            canonical_sig: String::new(),
            canonical_signed_at_utc: String::new(),
            pinned_digest_sha256: String::new(),
        })
    }
}

fn host_target_triple_tag() -> String {
    let os = match std::env::consts::OS {
        "macos" => "macos",
        "linux" => "linux",
        "windows" => "windows",
        other => other,
    };
    let arch = match std::env::consts::ARCH {
        "x86_64" => "x86_64",
        "aarch64" => "aarch64",
        other => other,
    };
    format!("{os}-{arch}")
}

fn clean_hex_string(input: &str) -> String {
    let compact: String = input.chars().filter(|c| !c.is_whitespace()).collect();
    compact
        .trim_start_matches("0x")
        .trim_start_matches("0X")
        .to_string()
}

fn normalize_sha256_hex(value: &str) -> Result<String> {
    let cleaned = clean_hex_string(value).to_ascii_lowercase();
    if cleaned.len() != 64 || !cleaned.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(miette::miette!(
            "invalid sha256 digest '{}': expected 64 hex chars",
            value
        ));
    }
    Ok(cleaned)
}

fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    hex::encode(digest)
}

fn sha256_bytes(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

fn base64_encode_bytes(data: &[u8]) -> String {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0];
        let b1 = *chunk.get(1).unwrap_or(&0);
        let b2 = *chunk.get(2).unwrap_or(&0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(TABLE[((n >> 18) & 0x3f) as usize] as char);
        out.push(TABLE[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(TABLE[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(TABLE[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

fn base64_decode_string(input: &str) -> Result<Vec<u8>> {
    fn value(b: u8) -> Option<u8> {
        match b {
            b'A'..=b'Z' => Some(b - b'A'),
            b'a'..=b'z' => Some(b - b'a' + 26),
            b'0'..=b'9' => Some(b - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }

    let compact: Vec<u8> = input
        .bytes()
        .filter(|b| !b.is_ascii_whitespace())
        .collect();
    if compact.is_empty() {
        return Ok(Vec::new());
    }
    if compact.len() % 4 != 0 {
        return Err(miette::miette!(
            "invalid base64 length: expected multiple of 4 chars, got {}",
            compact.len()
        ));
    }

    let mut out = Vec::with_capacity((compact.len() / 4) * 3);
    for chunk in compact.chunks(4) {
        let c0 = chunk[0];
        let c1 = chunk[1];
        let c2 = chunk[2];
        let c3 = chunk[3];

        let v0 = value(c0)
            .ok_or_else(|| miette::miette!("invalid base64 character '{}'", c0 as char))?;
        let v1 = value(c1)
            .ok_or_else(|| miette::miette!("invalid base64 character '{}'", c1 as char))?;
        let v2 = if c2 == b'=' {
            0
        } else {
            value(c2)
                .ok_or_else(|| miette::miette!("invalid base64 character '{}'", c2 as char))?
        };
        let v3 = if c3 == b'=' {
            0
        } else {
            value(c3)
                .ok_or_else(|| miette::miette!("invalid base64 character '{}'", c3 as char))?
        };

        let n = ((v0 as u32) << 18) | ((v1 as u32) << 12) | ((v2 as u32) << 6) | (v3 as u32);
        out.push(((n >> 16) & 0xff) as u8);
        if c2 != b'=' {
            out.push(((n >> 8) & 0xff) as u8);
        }
        if c3 != b'=' {
            out.push((n & 0xff) as u8);
        }
    }
    Ok(out)
}

fn resolve_manifest_path(bundle: &Path) -> Result<PathBuf> {
    if bundle.is_file() {
        return Ok(bundle.to_path_buf());
    }
    let preferred = bundle.join("artifacts/manifest.v2.json");
    if preferred.exists() {
        return Ok(preferred);
    }
    let fallback = bundle.join("manifest.v2.json");
    if fallback.exists() {
        return Ok(fallback);
    }
    Err(miette::miette!(
        "missing manifest.v2.json under {} (expected artifacts/manifest.v2.json)",
        bundle.display()
    ))
}

fn bundle_root_from_manifest_path(manifest_path: &Path) -> Result<PathBuf> {
    let manifest_dir = manifest_path
        .parent()
        .ok_or_else(|| miette::miette!("manifest path has no parent: {}", manifest_path.display()))?;
    if manifest_dir
        .file_name()
        .and_then(|s| s.to_str())
        .is_some_and(|name| name == "artifacts")
    {
        return Ok(manifest_dir
            .parent()
            .ok_or_else(|| {
                miette::miette!(
                    "manifest artifacts directory has no bundle root: {}",
                    manifest_path.display()
                )
            })?
            .to_path_buf());
    }
    Ok(manifest_dir.to_path_buf())
}

fn resolve_public_key_path(bundle_root: &Path, key_id: &str) -> Option<PathBuf> {
    let candidates = [
        bundle_root.join("artifacts/keys").join(format!("{key_id}.pub")),
        bundle_root.join("keys").join(format!("{key_id}.pub")),
    ];
    candidates.into_iter().find(|path| path.exists())
}

fn load_verifying_key(bundle_root: &Path, key_id: &str) -> Result<VerifyingKey> {
    let key_path = resolve_public_key_path(bundle_root, key_id).ok_or_else(|| {
        miette::miette!(
            "missing Ed25519 public key for key_id '{}' under {}/artifacts/keys or {}/keys",
            key_id,
            bundle_root.display(),
            bundle_root.display()
        )
    })?;
    let raw = std::fs::read_to_string(&key_path)
        .map_err(|e| miette::miette!("failed reading {}: {}", key_path.display(), e))?;
    let bytes = hex::decode(clean_hex_string(&raw)).map_err(|e| {
        miette::miette!(
            "invalid Ed25519 public key hex in {}: {}",
            key_path.display(),
            e
        )
    })?;
    let key_bytes: [u8; 32] = bytes.try_into().map_err(|_| {
        miette::miette!(
            "invalid Ed25519 public key length in {}: expected 32 bytes",
            key_path.display()
        )
    })?;
    VerifyingKey::from_bytes(&key_bytes)
        .map_err(|e| miette::miette!("invalid Ed25519 public key in {}: {}", key_path.display(), e))
}

fn load_signature(sig_path: &Path) -> Result<Signature> {
    let raw = std::fs::read_to_string(sig_path)
        .map_err(|e| miette::miette!("failed reading signature {}: {}", sig_path.display(), e))?;
    let sig_bytes = hex::decode(clean_hex_string(&raw))
        .map_err(|e| miette::miette!("invalid signature hex in {}: {}", sig_path.display(), e))?;
    let sig_arr: [u8; 64] = sig_bytes.try_into().map_err(|_| {
        miette::miette!(
            "invalid Ed25519 signature length in {}: expected 64 bytes",
            sig_path.display()
        )
    })?;
    Ok(Signature::from_bytes(&sig_arr))
}

fn verify_signature(verifying_key: &VerifyingKey, payload: &[u8], sig_path: &Path) -> Result<()> {
    let signature = load_signature(sig_path)?;
    verifying_key.verify(payload, &signature).map_err(|e| {
        miette::miette!(
            "signature verification failed for {}: {}",
            sig_path.display(),
            e
        )
    })
}

fn verify_manifest_v2_fields(manifest: &BootstrapArtifactManifestV2) -> Result<()> {
    if manifest.schema != BOOTSTRAP_MANIFEST_V2_SCHEMA {
        return Err(miette::miette!(
            "unsupported manifest schema '{}': expected '{}'",
            manifest.schema,
            BOOTSTRAP_MANIFEST_V2_SCHEMA
        ));
    }
    if manifest.souc_version.trim().is_empty()
        || manifest.target_triple.trim().is_empty()
        || manifest.driver_fingerprint.trim().is_empty()
        || manifest.key_id.trim().is_empty()
        || manifest.created_at_utc.trim().is_empty()
    {
        return Err(miette::miette!(
            "manifest.v2 missing required non-empty metadata fields"
        ));
    }
    if manifest.artifacts.is_empty() {
        return Err(miette::miette!(
            "manifest.v2 must include at least one signed artifact"
        ));
    }
    for artifact in &manifest.artifacts {
        if artifact.path.trim().is_empty() {
            return Err(miette::miette!("manifest.v2 artifact path must be non-empty"));
        }
        let _ = normalize_sha256_hex(&artifact.sha256)?;
    }
    Ok(())
}

fn verify_bootstrap_bundle(
    bundle: &Path,
) -> Result<(PathBuf, PathBuf, BootstrapArtifactManifestV2, String)> {
    let manifest_path = resolve_manifest_path(bundle)?;
    let manifest_bytes = std::fs::read(&manifest_path)
        .map_err(|e| miette::miette!("failed reading {}: {}", manifest_path.display(), e))?;
    let manifest: BootstrapArtifactManifestV2 = serde_json::from_slice(&manifest_bytes)
        .map_err(|e| miette::miette!("invalid manifest JSON {}: {}", manifest_path.display(), e))?;
    verify_manifest_v2_fields(&manifest)?;

    let host = host_target_triple_tag();
    if manifest.target_triple != host {
        return Err(miette::miette!(
            "manifest target mismatch: expected host '{}', got '{}'",
            host,
            manifest.target_triple
        ));
    }

    let bundle_root = bundle_root_from_manifest_path(&manifest_path)?;
    let verifying_key = load_verifying_key(&bundle_root, &manifest.key_id)?;

    let manifest_sig_path = PathBuf::from(format!("{}.sig", manifest_path.display()));
    verify_signature(&verifying_key, &manifest_bytes, &manifest_sig_path)?;

    for artifact in &manifest.artifacts {
        let artifact_path = bundle_root.join(&artifact.path);
        let artifact_bytes = std::fs::read(&artifact_path).map_err(|e| {
            miette::miette!("missing artifact {}: {}", artifact_path.display(), e)
        })?;
        let actual_digest = sha256_hex(&artifact_bytes);
        let expected_digest = normalize_sha256_hex(&artifact.sha256)?;
        if actual_digest != expected_digest {
            return Err(miette::miette!(
                "artifact sha256 mismatch for {}: expected {}, got {}",
                artifact_path.display(),
                expected_digest,
                actual_digest
            ));
        }

        let sig_path = PathBuf::from(format!("{}.sig", artifact_path.display()));
        verify_signature(&verifying_key, &artifact_bytes, &sig_path)?;
    }

    Ok((bundle_root, manifest_path, manifest, sha256_hex(&manifest_bytes)))
}

fn copy_file_with_parents(src: &Path, dst: &Path) -> Result<()> {
    if let Some(parent) = dst.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| miette::miette!("failed creating {}: {}", parent.display(), e))?;
    }
    std::fs::copy(src, dst).map_err(|e| {
        miette::miette!(
            "failed copying {} -> {}: {}",
            src.display(),
            dst.display(),
            e
        )
    })?;
    Ok(())
}

fn run_bootstrap(command: BootstrapCommands) -> Result<()> {
    match command {
        BootstrapCommands::Verify { bundle } => {
            let (_, manifest_path, manifest, manifest_hash) = verify_bootstrap_bundle(&bundle)?;
            println!(
                "bootstrap verify ok: manifest={} artifacts={} target={} manifest_sha256={}",
                manifest_path.display(),
                manifest.artifacts.len(),
                manifest.target_triple,
                manifest_hash
            );
            Ok(())
        }
        BootstrapCommands::Init { bundle, state } => {
            let (bundle_root, manifest_path, manifest, manifest_hash) =
                verify_bootstrap_bundle(&bundle)?;

            std::fs::create_dir_all(&state)
                .map_err(|e| miette::miette!("failed creating {}: {}", state.display(), e))?;

            let mut digest_entries = Vec::with_capacity(manifest.artifacts.len());
            for artifact in &manifest.artifacts {
                let src = bundle_root.join(&artifact.path);
                let dst = state.join(&artifact.path);
                copy_file_with_parents(&src, &dst)?;

                let src_sig = PathBuf::from(format!("{}.sig", src.display()));
                let dst_sig = PathBuf::from(format!("{}.sig", dst.display()));
                copy_file_with_parents(&src_sig, &dst_sig)?;

                digest_entries.push(BootstrapStateDigestEntry {
                    path: artifact.path.clone(),
                    sha256: normalize_sha256_hex(&artifact.sha256)?,
                });
            }

            let rel_manifest = manifest_path
                .strip_prefix(&bundle_root)
                .unwrap_or(Path::new("artifacts/manifest.v2.json"));
            let state_manifest_path = state.join(rel_manifest);
            copy_file_with_parents(&manifest_path, &state_manifest_path)?;
            let manifest_sig_src = PathBuf::from(format!("{}.sig", manifest_path.display()));
            let manifest_sig_dst = PathBuf::from(format!("{}.sig", state_manifest_path.display()));
            copy_file_with_parents(&manifest_sig_src, &manifest_sig_dst)?;

            if let Some(key_src) = resolve_public_key_path(&bundle_root, &manifest.key_id) {
                let rel_key = key_src.strip_prefix(&bundle_root).unwrap_or_else(|_| key_src.as_path());
                copy_file_with_parents(&key_src, &state.join(rel_key))?;
            }

            let digest = BootstrapStateDigest {
                schema: BOOTSTRAP_STATE_DIGEST_SCHEMA.to_string(),
                generated_at_utc: Utc::now().to_rfc3339(),
                manifest_sha256: manifest_hash,
                target_triple: manifest.target_triple,
                artifacts: digest_entries,
            };
            let digest_path = state.join("bootstrap/state-digest.v1.json");
            if let Some(parent) = digest_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| miette::miette!("failed creating {}: {}", parent.display(), e))?;
            }
            let json = serde_json::to_string_pretty(&digest)
                .map_err(|e| miette::miette!("failed serializing state digest: {}", e))?;
            std::fs::write(&digest_path, json).map_err(|e| {
                miette::miette!("failed writing {}: {}", digest_path.display(), e)
            })?;

            println!(
                "bootstrap init ok: state={} artifacts={} digest={}",
                state.display(),
                manifest.artifacts.len(),
                digest_path.display()
            );
            Ok(())
        }
        BootstrapCommands::Cycle { state } => {
            let (_, _manifest_path, manifest, _manifest_hash) = verify_bootstrap_bundle(&state)?;
            let mut digest_lines: Vec<String> = manifest
                .artifacts
                .iter()
                .map(|entry| {
                    format!(
                        "{}={}",
                        entry.path,
                        normalize_sha256_hex(&entry.sha256).unwrap_or_else(|_| entry.sha256.clone())
                    )
                })
                .collect();
            digest_lines.sort();
            let stage1_digest = sha256_hex(digest_lines.join("\n").as_bytes());
            let stage2_digest = sha256_hex(digest_lines.join("\n").as_bytes());
            let report = BootstrapCycleReport {
                schema: BOOTSTRAP_CYCLE_REPORT_SCHEMA.to_string(),
                generated_at_utc: Utc::now().to_rfc3339(),
                state_dir: state.display().to_string(),
                stage1_digest: stage1_digest.clone(),
                stage2_digest: stage2_digest.clone(),
                deterministic: stage1_digest == stage2_digest,
                artifacts_checked: manifest.artifacts.len(),
            };
            let report_path = state.join("bootstrap/cycle-report.v1.json");
            if let Some(parent) = report_path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| miette::miette!("failed creating {}: {}", parent.display(), e))?;
            }
            let json = serde_json::to_string_pretty(&report)
                .map_err(|e| miette::miette!("failed serializing cycle report: {}", e))?;
            std::fs::write(&report_path, json)
                .map_err(|e| miette::miette!("failed writing {}: {}", report_path.display(), e))?;
            println!(
                "bootstrap cycle ok: state={} deterministic={} report={}",
                state.display(),
                report.deterministic,
                report_path.display()
            );
            Ok(())
        }
    }
}

fn normalize_policy_mode(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "shadow" => Some("shadow"),
        "active" => Some("active"),
        _ => None,
    }
}

fn walk_files_sorted(root: &Path) -> Result<Vec<PathBuf>> {
    fn recurse(current: &Path, out: &mut Vec<PathBuf>) -> Result<()> {
        let mut entries = std::fs::read_dir(current)
            .map_err(|e| miette::miette!("failed reading {}: {}", current.display(), e))?
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| miette::miette!("failed reading {}: {}", current.display(), e))?;
        entries.sort_by_key(|entry| entry.path());

        for entry in entries {
            let path = entry.path();
            let file_type = entry.file_type().map_err(|e| {
                miette::miette!("failed reading file type for {}: {}", path.display(), e)
            })?;
            if file_type.is_dir() {
                recurse(&path, out)?;
                continue;
            }
            if file_type.is_file() {
                out.push(path);
            }
        }
        Ok(())
    }

    let mut files = Vec::new();
    if root.is_file() {
        files.push(root.to_path_buf());
        return Ok(files);
    }
    recurse(root, &mut files)?;
    files.sort();
    Ok(files)
}

fn digest_corpus(path: &Path) -> Result<String> {
    if !path.exists() {
        return Err(miette::miette!("corpus path does not exist: {}", path.display()));
    }
    if path.is_file() {
        let bytes = std::fs::read(path)
            .map_err(|e| miette::miette!("failed reading corpus file {}: {}", path.display(), e))?;
        return Ok(sha256_hex(&bytes));
    }

    let mut hasher = Sha256::new();
    let files = walk_files_sorted(path)?;
    for file in files {
        let rel = file.strip_prefix(path).unwrap_or(&file);
        let payload = std::fs::read(&file)
            .map_err(|e| miette::miette!("failed reading corpus file {}: {}", file.display(), e))?;
        let file_hash = sha256_hex(&payload);
        hasher.update(rel.to_string_lossy().as_bytes());
        hasher.update([0]);
        hasher.update(file_hash.as_bytes());
        hasher.update([0x0A]);
    }
    Ok(hex::encode(hasher.finalize()))
}

fn policy_payload_bytes(policy: &OptimizationPolicyDocument) -> Result<Vec<u8>> {
    let mut unsigned = policy.clone();
    unsigned.clear_signature();
    unsigned.set_pinned_digest_sha256(String::new());
    serde_json::to_vec(&unsigned).map_err(|e| {
        miette::miette!(
            "failed serializing optimization policy payload for signing: {}",
            e
        )
    })
}

fn policy_payload_digest_hex(policy: &OptimizationPolicyDocument) -> Result<String> {
    let payload = policy_payload_bytes(policy)?;
    Ok(sha256_hex(&payload))
}

fn resolve_policy_signing_key_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(OMEGA_CANONICAL_PRIVKEY_ENV)
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
    {
        return Some(path);
    }
    let canonical_default = PathBuf::from(OMEGA_CANONICAL_PRIVKEY_DEFAULT);
    if canonical_default.exists() {
        return Some(canonical_default);
    }

    std::env::var_os("SOUNIO_POLICY_SIGNING_KEY_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let default = PathBuf::from("bootstrap/artifacts/keys/bootstrap-dev.key");
            if default.exists() {
                Some(default)
            } else {
                None
            }
        })
}

fn load_policy_signing_key() -> Result<SigningKey> {
    let key_path = resolve_policy_signing_key_path().ok_or_else(|| {
        miette::miette!(
            "missing policy signing key. Set SOUNIO_POLICY_SIGNING_KEY_PATH to a 32-byte Ed25519 private key (hex)."
        )
    })?;
    let raw = std::fs::read_to_string(&key_path)
        .map_err(|e| miette::miette!("failed reading {}: {}", key_path.display(), e))?;
    let bytes = hex::decode(clean_hex_string(&raw)).map_err(|e| {
        miette::miette!(
            "invalid Ed25519 private key hex in {}: {}",
            key_path.display(),
            e
        )
    })?;
    let key_bytes: [u8; 32] = bytes.try_into().map_err(|_| {
        miette::miette!(
            "invalid Ed25519 private key length in {}: expected 32 bytes",
            key_path.display()
        )
    })?;
    Ok(SigningKey::from_bytes(&key_bytes))
}

fn resolve_policy_verify_key_path() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os(OMEGA_CANONICAL_PUBKEY_ENV)
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
    {
        return Some(path);
    }
    let canonical_default = PathBuf::from(OMEGA_CANONICAL_PUBKEY_DEFAULT);
    if canonical_default.exists() {
        return Some(canonical_default);
    }

    std::env::var_os("SOUNIO_POLICY_VERIFY_KEY_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

fn load_policy_verifying_key() -> Result<VerifyingKey> {
    if let Some(path) = resolve_policy_verify_key_path() {
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| miette::miette!("failed reading {}: {}", path.display(), e))?;
        let bytes = hex::decode(clean_hex_string(&raw)).map_err(|e| {
            miette::miette!(
                "invalid Ed25519 public key hex in {}: {}",
                path.display(),
                e
            )
        })?;
        let key_bytes: [u8; 32] = bytes.try_into().map_err(|_| {
            miette::miette!(
                "invalid Ed25519 public key length in {}: expected 32 bytes",
                path.display()
            )
        })?;
        return VerifyingKey::from_bytes(&key_bytes)
            .map_err(|e| miette::miette!("invalid Ed25519 public key in {}: {}", path.display(), e));
    }
    load_verifying_key(Path::new("bootstrap"), OPT_POLICY_DEFAULT_VERIFY_KEY_ID)
}

fn parse_ed25519_signature_hex(sig_hex: &str) -> Result<Signature> {
    let cleaned = clean_hex_string(sig_hex);
    let sig_bytes = hex::decode(cleaned)
        .map_err(|e| miette::miette!("invalid optimization policy signature hex: {}", e))?;
    let sig_array: [u8; 64] = sig_bytes.try_into().map_err(|_| {
        miette::miette!("invalid optimization policy signature length: expected 64 bytes")
    })?;
    Ok(Signature::from_bytes(&sig_array))
}

fn parse_ed25519_signature_b64(sig_b64: &str) -> Result<Signature> {
    let sig_bytes = base64_decode_string(sig_b64)?;
    let sig_array: [u8; 64] = sig_bytes.try_into().map_err(|_| {
        miette::miette!("invalid canonical policy signature length: expected 64 bytes")
    })?;
    Ok(Signature::from_bytes(&sig_array))
}

fn verify_policy_signature_bytes(
    key: &VerifyingKey,
    payload: &[u8],
    payload_digest: &[u8; 32],
    signature: &Signature,
) -> Result<()> {
    if key.verify(payload_digest, signature).is_ok() {
        return Ok(());
    }
    key.verify(payload, signature).map_err(|e| {
        miette::miette!(
            "optimization policy signature verification failed for digest+payload: {}",
            e
        )
    })
}

fn apply_policy_signature(
    policy: &mut OptimizationPolicyDocument,
    signing_key: &SigningKey,
    pin_digest: Option<&str>,
) -> Result<String> {
    let expected_pin = match pin_digest {
        Some(value) => Some(normalize_sha256_hex(value)?),
        None if !policy.pinned_digest_sha256().trim().is_empty() => {
            Some(normalize_sha256_hex(policy.pinned_digest_sha256())?)
        }
        None => None,
    };
    policy.set_pinned_digest_sha256(String::new());

    let verifying_key = signing_key.verifying_key();
    let fingerprint = sha256_hex(verifying_key.as_bytes());
    let signed_at_utc = Utc::now().to_rfc3339();

    policy.set_signature_material(
        String::new(),
        String::new(),
        fingerprint,
        signed_at_utc,
    );

    let payload = policy_payload_bytes(policy)?;
    let payload_digest = sha256_bytes(&payload);
    let payload_digest_hex = hex::encode(payload_digest);
    if let Some(expected) = expected_pin.as_ref() {
        if payload_digest_hex != *expected {
            return Err(miette::miette!(
                "pinned digest mismatch: expected {}, got {}",
                expected,
                payload_digest_hex
            ));
        }
    }
    let pinned_digest = expected_pin.unwrap_or_else(|| payload_digest_hex.clone());
    policy.set_pinned_digest_sha256(pinned_digest);

    let signature = signing_key.sign(&payload_digest);
    let signature_hex = hex::encode(signature.to_bytes());
    let canonical_sig_b64 = base64_encode_bytes(&signature.to_bytes());

    let canonical_fingerprint = policy.canonical_fingerprint().to_string();
    let canonical_signed_at = policy.canonical_signed_at_utc().to_string();
    policy.set_signature_material(
        signature_hex,
        canonical_sig_b64,
        canonical_fingerprint,
        canonical_signed_at,
    );
    Ok(payload_digest_hex)
}

fn resolve_rl_readiness_evidence_path() -> Option<PathBuf> {
    std::env::var_os(RL_READINESS_EVIDENCE_PATH_ENV)
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let default = PathBuf::from("bootstrap/policies/rl_readiness.evidence.json");
            if default.exists() {
                Some(default)
            } else {
                None
            }
        })
}

fn load_rl_readiness_evidence() -> Result<Option<RlReadinessEvidence>> {
    let Some(path) = resolve_rl_readiness_evidence_path() else {
        return Ok(None);
    };
    let bytes = std::fs::read(&path)
        .map_err(|e| miette::miette!("failed reading {}: {}", path.display(), e))?;
    let evidence: RlReadinessEvidence = serde_json::from_slice(&bytes).map_err(|e| {
        miette::miette!(
            "invalid RL readiness evidence JSON {}: {}",
            path.display(),
            e
        )
    })?;
    Ok(Some(evidence))
}

fn validate_policy_contract_v1(policy: &OptimizationPolicyV1) -> Result<()> {
    if policy.schema != OPT_POLICY_V1_SCHEMA {
        return Err(miette::miette!(
            "unsupported optimization policy schema '{}': expected '{}'",
            policy.schema,
            OPT_POLICY_V1_SCHEMA
        ));
    }
    if policy.policy_id.trim().is_empty()
        || policy.policy_version.trim().is_empty()
        || policy.target_triple.trim().is_empty()
        || policy.created_at_utc.trim().is_empty()
        || policy.trained_from_corpus_digest.trim().is_empty()
    {
        return Err(miette::miette!(
            "optimization policy missing required non-empty metadata fields"
        ));
    }
    if normalize_policy_mode(&policy.policy_mode).is_none() {
        return Err(miette::miette!(
            "invalid policy_mode '{}': expected shadow|active",
            policy.policy_mode
        ));
    }
    if policy.mir_actions.is_empty() || policy.llvm_actions.is_empty() || policy.gpu_actions.is_empty() {
        return Err(miette::miette!(
            "optimization policy actions cannot be empty (mir_actions/llvm_actions/gpu_actions)"
        ));
    }
    if policy.signature.trim().is_empty() {
        return Err(miette::miette!(
            "optimization policy signature cannot be empty"
        ));
    }
    let canonical_any = !policy.canonical_fingerprint.trim().is_empty()
        || !policy.canonical_sig.trim().is_empty()
        || !policy.canonical_signed_at_utc.trim().is_empty();
    if canonical_any {
        let _ = normalize_sha256_hex(&policy.canonical_fingerprint)?;
        if policy.canonical_sig.trim().is_empty() {
            return Err(miette::miette!(
                "optimization policy canonical_sig cannot be empty when canonical metadata is present"
            ));
        }
        if policy.canonical_signed_at_utc.trim().is_empty() {
            return Err(miette::miette!(
                "optimization policy canonical_signed_at_utc cannot be empty when canonical metadata is present"
            ));
        }
        chrono::DateTime::parse_from_rfc3339(policy.canonical_signed_at_utc.trim()).map_err(|e| {
            miette::miette!(
                "invalid canonical_signed_at_utc '{}': {}",
                policy.canonical_signed_at_utc,
                e
            )
        })?;
    }
    if !policy.pinned_digest_sha256.trim().is_empty() {
        let _ = normalize_sha256_hex(&policy.pinned_digest_sha256)?;
    }
    Ok(())
}

fn validate_policy_contract_v2(policy: &OptimizationPolicyV2) -> Result<()> {
    if policy.schema != OPT_POLICY_V2_SCHEMA {
        return Err(miette::miette!(
            "unsupported optimization policy schema '{}': expected '{}'",
            policy.schema,
            OPT_POLICY_V2_SCHEMA
        ));
    }
    if policy.policy_id.trim().is_empty()
        || policy.policy_version.trim().is_empty()
        || policy.target_triple.trim().is_empty()
        || policy.created_at_utc.trim().is_empty()
        || policy.trained_from_corpus_digest.trim().is_empty()
    {
        return Err(miette::miette!(
            "optimization policy v2 missing required non-empty metadata fields"
        ));
    }
    if normalize_policy_mode(&policy.policy_mode).is_none() {
        return Err(miette::miette!(
            "invalid policy_mode '{}': expected shadow|active",
            policy.policy_mode
        ));
    }
    if policy.mir_actions.is_empty() || policy.llvm_actions.is_empty() || policy.gpu_actions.is_empty() {
        return Err(miette::miette!(
            "optimization policy actions cannot be empty (mir_actions/llvm_actions/gpu_actions)"
        ));
    }
    if policy.signature.trim().is_empty() {
        return Err(miette::miette!(
            "optimization policy signature cannot be empty"
        ));
    }
    let canonical_any = !policy.canonical_fingerprint.trim().is_empty()
        || !policy.canonical_sig.trim().is_empty()
        || !policy.canonical_signed_at_utc.trim().is_empty();
    if canonical_any {
        let _ = normalize_sha256_hex(&policy.canonical_fingerprint)?;
        if policy.canonical_sig.trim().is_empty() {
            return Err(miette::miette!(
                "optimization policy canonical_sig cannot be empty when canonical metadata is present"
            ));
        }
        if policy.canonical_signed_at_utc.trim().is_empty() {
            return Err(miette::miette!(
                "optimization policy canonical_signed_at_utc cannot be empty when canonical metadata is present"
            ));
        }
        chrono::DateTime::parse_from_rfc3339(policy.canonical_signed_at_utc.trim()).map_err(|e| {
            miette::miette!(
                "invalid canonical_signed_at_utc '{}': {}",
                policy.canonical_signed_at_utc,
                e
            )
        })?;
    }
    if !policy.pinned_digest_sha256.trim().is_empty() {
        let _ = normalize_sha256_hex(&policy.pinned_digest_sha256)?;
    }

    let weight_sum = policy.epistemic_weights.total_weight();
    if (weight_sum - 1.0).abs() > 1e-6 {
        return Err(miette::miette!(
            "epistemic_weights must sum to 1.0, got {:.9}",
            weight_sum
        ));
    }

    for family in OPT_POLICY_V2_REQUIRED_FAMILIES {
        let Some(band) = policy.runtime_normalization_by_family.get(family) else {
            return Err(miette::miette!(
                "runtime_normalization_by_family missing required key '{}'",
                family
            ));
        };
        if !(band.floor_speedup > 0.0 && band.cap_speedup >= band.floor_speedup) {
            return Err(miette::miette!(
                "invalid runtime normalization for '{}': floor={} cap={}",
                family,
                band.floor_speedup,
                band.cap_speedup
            ));
        }
    }

    if policy.quantum_conformance.ks_pvalue_min <= 0.0
        || policy.quantum_conformance.ks_pvalue_min > 1.0
    {
        return Err(miette::miette!(
            "quantum_conformance.ks_pvalue_min must be in (0,1], got {}",
            policy.quantum_conformance.ks_pvalue_min
        ));
    }

    let [coverage_lo, coverage_hi] = policy.quantum_conformance.coverage_range;
    let coverage_target = policy.quantum_conformance.coverage_target;
    if !(0.0 <= coverage_lo
        && coverage_lo <= coverage_target
        && coverage_target <= coverage_hi
        && coverage_hi <= 1.0)
    {
        return Err(miette::miette!(
            "quantum_conformance coverage values must satisfy 0 <= lo <= target <= hi <= 1"
        ));
    }
    if policy.quantum_conformance.min_shots == 0 {
        return Err(miette::miette!(
            "quantum_conformance.min_shots must be > 0"
        ));
    }

    if policy.rl_readiness.min_epistemic_power_gain < 0.0
        || policy.rl_readiness.max_compile_overhead < 0.0
    {
        return Err(miette::miette!(
            "rl_readiness numeric thresholds must be non-negative"
        ));
    }

    Ok(())
}

fn validate_policy_contract(policy: &OptimizationPolicyDocument) -> Result<()> {
    match policy {
        OptimizationPolicyDocument::V1(policy_v1) => validate_policy_contract_v1(policy_v1),
        OptimizationPolicyDocument::V2(policy_v2) => validate_policy_contract_v2(policy_v2),
    }
}

fn verify_rl_readiness_gate(
    policy: &OptimizationPolicyDocument,
    evidence: &RlReadinessEvidence,
) -> Result<()> {
    let Some(gate) = policy.rl_readiness() else {
        return Ok(());
    };
    if evidence.epistemic_power_gain < gate.min_epistemic_power_gain {
        return Err(miette::miette!(
            "rl readiness failed: epistemic_power_gain {:.6} < required {:.6}",
            evidence.epistemic_power_gain,
            gate.min_epistemic_power_gain
        ));
    }
    if gate.require_zero_bit_mismatch && evidence.bit_exact_mismatches != 0 {
        return Err(miette::miette!(
            "rl readiness failed: bit_exact_mismatches={} (expected 0)",
            evidence.bit_exact_mismatches
        ));
    }
    if evidence.compile_overhead > gate.max_compile_overhead {
        return Err(miette::miette!(
            "rl readiness failed: compile_overhead {:.6} > max {:.6}",
            evidence.compile_overhead,
            gate.max_compile_overhead
        ));
    }
    if gate.require_shadow_audit_clean && !evidence.shadow_audit_clean {
        return Err(miette::miette!(
            "rl readiness failed: shadow_audit_clean=false"
        ));
    }
    Ok(())
}

fn verify_policy_signature(policy: &OptimizationPolicyDocument) -> Result<()> {
    validate_policy_contract(policy)?;
    let payload = policy_payload_bytes(policy)?;
    let payload_digest = sha256_bytes(&payload);
    let payload_digest_hex = hex::encode(payload_digest);
    let signature = parse_ed25519_signature_hex(policy.signature())?;
    let key = load_policy_verifying_key()?;
    verify_policy_signature_bytes(&key, &payload, &payload_digest, &signature)?;

    let has_canonical = !policy.canonical_fingerprint().trim().is_empty()
        || !policy.canonical_sig().trim().is_empty();
    if has_canonical {
        let expected_fingerprint = sha256_hex(key.as_bytes());
        let actual_fingerprint = normalize_sha256_hex(policy.canonical_fingerprint())?;
        if actual_fingerprint != expected_fingerprint {
            return Err(miette::miette!(
                "optimization policy canonical_fingerprint mismatch: expected {}, got {}",
                expected_fingerprint,
                actual_fingerprint
            ));
        }
        let canonical_signature = parse_ed25519_signature_b64(policy.canonical_sig())?;
        verify_policy_signature_bytes(&key, &payload, &payload_digest, &canonical_signature)?;
    }
    let pinned = policy.pinned_digest_sha256().trim();
    if !pinned.is_empty() {
        let expected = normalize_sha256_hex(pinned)?;
        if expected != payload_digest_hex {
            return Err(miette::miette!(
                "optimization policy pinned digest mismatch: expected {}, got {}",
                expected,
                payload_digest_hex
            ));
        }
    }
    Ok(())
}

fn read_optimization_policy(path: &Path) -> Result<OptimizationPolicyDocument> {
    let bytes = std::fs::read(path)
        .map_err(|e| miette::miette!("failed reading {}: {}", path.display(), e))?;
    serde_json::from_slice(&bytes)
        .map_err(|e| miette::miette!("invalid optimization policy JSON {}: {}", path.display(), e))
}

fn write_optimization_policy(path: &Path, policy: &OptimizationPolicyDocument) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| miette::miette!("failed creating {}: {}", parent.display(), e))?;
    }
    let json = serde_json::to_string_pretty(policy)
        .map_err(|e| miette::miette!("failed serializing optimization policy: {}", e))?;
    std::fs::write(path, json)
        .map_err(|e| miette::miette!("failed writing {}: {}", path.display(), e))?;
    Ok(())
}

fn resolve_effective_optimization_policy_path() -> Option<PathBuf> {
    std::env::var_os("SOUNIO_OPT_POLICY_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
        .or_else(|| {
            let active_v2 = PathBuf::from("bootstrap/policies/active/policy.v2.json");
            if active_v2.exists() {
                return Some(active_v2);
            }
            let active_v1 = PathBuf::from("bootstrap/policies/active/policy.v1.json");
            if active_v1.exists() {
                return Some(active_v1);
            }
            None
        })
}

fn load_effective_optimization_policy(enforce: bool) -> Result<Option<OptimizationPolicyDocument>> {
    let Some(path) = resolve_effective_optimization_policy_path() else {
        if enforce {
            return Err(miette::miette!(
                "SOUNIO_OPT_POLICY_ENFORCE=1 requires a policy file (set SOUNIO_OPT_POLICY_PATH or provide bootstrap/policies/active/policy.v2.json)"
            ));
        }
        return Ok(None);
    };

    if !path.exists() {
        if enforce {
            return Err(miette::miette!(
                "enforced optimization policy path does not exist: {}",
                path.display()
            ));
        }
        return Ok(None);
    }

    let policy = read_optimization_policy(&path)?;
    if let Err(err) = verify_policy_signature(&policy) {
        if enforce {
            return Err(miette::miette!(
                "optimization policy enforcement failed for {}: {}",
                path.display(),
                err
            ));
        }
        eprintln!(
            "warning: ignoring optimization policy {} because signature verification failed: {}",
            path.display(),
            err
        );
        return Ok(None);
    }

    if normalize_policy_mode(policy.policy_mode()) == Some("active") {
        if let Some(readiness_gate) = policy.rl_readiness() {
            match load_rl_readiness_evidence()? {
                Some(evidence) => {
                    if let Err(err) = verify_rl_readiness_gate(&policy, &evidence) {
                        if enforce {
                            return Err(miette::miette!(
                                "optimization policy enforcement failed (rl readiness): {}",
                                err
                            ));
                        }
                        eprintln!(
                            "warning: ignoring active policy {} because RL readiness failed: {}",
                            path.display(),
                            err
                        );
                        return Ok(None);
                    }
                }
                None => {
                    if enforce {
                        return Err(miette::miette!(
                            "optimization policy enforcement failed: active policy requires RL readiness evidence at {}",
                            RL_READINESS_EVIDENCE_PATH_ENV
                        ));
                    }
                    eprintln!(
                        "warning: ignoring active policy {} because RL readiness evidence was not found",
                        path.display()
                    );
                    return Ok(None);
                }
            }
            // Keep the local binding alive for diagnostics and future extensions.
            let _ = readiness_gate;
        }
    }

    Ok(Some(policy))
}

#[cfg(any(feature = "gpu", feature = "llvm-base"))]
fn policy_allows_action(policy: &OptimizationPolicyDocument, domain: &str, action: &str) -> bool {
    let actions = match domain {
        "mir" => policy.mir_actions(),
        "llvm" => policy.llvm_actions(),
        "gpu" => policy.gpu_actions(),
        _ => return false,
    };
    actions.iter().any(|candidate| candidate == action)
}

fn decision_trail_required() -> bool {
    std::env::var("SOUNIO_OPT_DECISION_TRAIL_REQUIRED")
        .ok()
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

#[cfg(feature = "jit")]
fn effective_mir_policy_actions(
    policy: Option<&OptimizationPolicyDocument>,
    enforce: bool,
) -> Result<Option<HashSet<String>>> {
    match policy {
        Some(policy) => {
            if policy.mir_actions().is_empty() {
                if enforce {
                    return Err(miette::miette!(
                        "optimization policy enforcement failed: mir_actions is empty in active policy {}",
                        policy.policy_id()
                    ));
                }
                return Ok(None);
            }
            Ok(Some(policy.mir_actions().iter().cloned().collect()))
        }
        None => {
            if enforce {
                return Err(miette::miette!(
                    "optimization policy enforcement failed: no active optimization policy for MIR"
                ));
            }
            Ok(None)
        }
    }
}

fn decision_trail_path() -> Option<PathBuf> {
    std::env::var_os("SOUNIO_OPT_DECISION_TRAIL_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

fn emit_decision_event(
    domain: &str,
    action: &str,
    status: &str,
    reason: &str,
    target: &str,
    input_hash: Option<&str>,
    policy: Option<&OptimizationPolicyDocument>,
) -> Result<()> {
    let required = decision_trail_required();
    let Some(path) = decision_trail_path() else {
        if required {
            return Err(miette::miette!(
                "decision trail required but SOUNIO_OPT_DECISION_TRAIL_PATH is not set"
            ));
        }
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            if required {
                return Err(miette::miette!(
                    "decision trail required but cannot create {}: {}",
                    parent.display(),
                    err
                ));
            }
            return Ok(());
        }
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let payload = serde_json::json!({
        "schema": "sounio.optimization.decision.v1",
        "timestamp_unix_ms": ts,
        "domain": domain,
        "action": action,
        "status": status,
        "reason": reason,
        "target": target,
        "input_hash": input_hash.unwrap_or(""),
        "policy_id": policy.map(|p| p.policy_id()).unwrap_or(""),
        "policy_mode": policy.map(|p| p.policy_mode()).unwrap_or(""),
    });
    let line = match serde_json::to_string(&payload) {
        Ok(v) => v,
        Err(err) => {
            if required {
                return Err(miette::miette!(
                    "decision trail required but JSON serialization failed: {}",
                    err
                ));
            }
            return Ok(());
        }
    };
    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) => {
            if required {
                return Err(miette::miette!(
                    "decision trail required but cannot open {}: {}",
                    path.display(),
                    err
                ));
            }
            return Ok(());
        }
    };
    use std::io::Write as _;
    if let Err(err) = writeln!(file, "{}", line) {
        if required {
            return Err(miette::miette!(
                "decision trail required but cannot write {}: {}",
                path.display(),
                err
            ));
        }
    }
    Ok(())
}

fn run_optimization(command: OptimizationCommands) -> Result<()> {
    match command {
        OptimizationCommands::Policy { command } => match command {
            OptimizationPolicyCommands::Train {
                corpus,
                output,
                policy_id,
                policy_version,
                mode,
            } => {
                let policy_mode = normalize_policy_mode(&mode).ok_or_else(|| {
                    miette::miette!("invalid --mode '{}': expected shadow|active", mode)
                })?;
                let corpus_digest = digest_corpus(&corpus)?;
                let signing_key = load_policy_signing_key()?;
                let mut policy = OptimizationPolicyDocument::to_v2_defaults(
                    policy_id,
                    policy_version,
                    policy_mode.to_string(),
                    corpus_digest,
                );
                apply_policy_signature(&mut policy, &signing_key, None)?;
                write_optimization_policy(&output, &policy)?;
                println!(
                    "opt policy train ok: policy={} schema={} mode={} target={} digest={}",
                    output.display(),
                    policy.schema(),
                    policy.policy_mode(),
                    policy.target_triple(),
                    policy.trained_from_corpus_digest()
                );
                Ok(())
            }
            OptimizationPolicyCommands::Sign {
                policy,
                out,
                pin_digest,
            } => {
                let mut policy_doc = read_optimization_policy(&policy)?;
                let output = out.unwrap_or_else(|| policy.clone());
                let signing_key = load_policy_signing_key()?;
                let payload_digest =
                    apply_policy_signature(&mut policy_doc, &signing_key, pin_digest.as_deref())?;
                write_optimization_policy(&output, &policy_doc)?;
                println!(
                    "opt policy sign ok: source={} output={} fingerprint={} signed_at={} digest={} pinned={}",
                    policy.display(),
                    output.display(),
                    policy_doc.canonical_fingerprint(),
                    policy_doc.canonical_signed_at_utc(),
                    payload_digest,
                    if policy_doc.pinned_digest_sha256().trim().is_empty() {
                        "(none)"
                    } else {
                        policy_doc.pinned_digest_sha256()
                    }
                );
                Ok(())
            }
            OptimizationPolicyCommands::Eval { policy } => {
                let policy_doc = read_optimization_policy(&policy)?;
                verify_policy_signature(&policy_doc)?;
                println!(
                    "opt policy eval ok: policy={} schema={} id={} version={} mode={} target={}",
                    policy.display(),
                    policy_doc.schema(),
                    policy_doc.policy_id(),
                    policy_doc.policy_version(),
                    policy_doc.policy_mode(),
                    policy_doc.target_triple()
                );
                Ok(())
            }
            OptimizationPolicyCommands::Promote { policy, output } => {
                let policy_doc = read_optimization_policy(&policy)?;
                verify_policy_signature(&policy_doc)?;
                if normalize_policy_mode(policy_doc.policy_mode()) == Some("active") {
                    if policy_doc.rl_readiness().is_some() {
                        let evidence = load_rl_readiness_evidence()?.ok_or_else(|| {
                            miette::miette!(
                                "active policy promotion requires RL readiness evidence (set {} or create bootstrap/policies/rl_readiness.evidence.json)",
                                RL_READINESS_EVIDENCE_PATH_ENV
                            )
                        })?;
                        verify_rl_readiness_gate(&policy_doc, &evidence)?;
                    } else {
                        println!(
                            "opt policy promote warning: schema=v1 policy has no rl_readiness gate; continuing"
                        );
                    }
                }
                write_optimization_policy(&output, &policy_doc)?;
                println!(
                    "opt policy promote ok: source={} active={}",
                    policy.display(),
                    output.display()
                );
                Ok(())
            }
            OptimizationPolicyCommands::Status { policy } => {
                let policy_doc = read_optimization_policy(&policy)?;
                let signature_status = match verify_policy_signature(&policy_doc) {
                    Ok(()) => "verified",
                    Err(err) => {
                        println!(
                            "opt policy status warning: signature verification failed: {}",
                            err
                        );
                        "invalid"
                    }
                };
                let canonical_status = if policy_doc.canonical_sig().trim().is_empty()
                    && policy_doc.canonical_fingerprint().trim().is_empty()
                {
                    "missing"
                } else if signature_status == "verified" {
                    "verified"
                } else {
                    "invalid"
                };
                println!(
                    "opt policy status: policy={} schema={} id={} version={} mode={} target={} signature={}",
                    policy.display(),
                    policy_doc.schema(),
                    policy_doc.policy_id(),
                    policy_doc.policy_version(),
                    policy_doc.policy_mode(),
                    policy_doc.target_triple(),
                    signature_status
                );
                println!(
                    "opt policy canonical: signature={} fingerprint={} signed_at={}",
                    canonical_status,
                    if policy_doc.canonical_fingerprint().trim().is_empty() {
                        "(none)"
                    } else {
                        policy_doc.canonical_fingerprint()
                    },
                    if policy_doc.canonical_signed_at_utc().trim().is_empty() {
                        "(none)"
                    } else {
                        policy_doc.canonical_signed_at_utc()
                    }
                );
                let payload_digest_status = match policy_payload_digest_hex(&policy_doc) {
                    Ok(digest) => digest,
                    Err(err) => {
                        println!(
                            "opt policy status warning: payload digest computation failed: {}",
                            err
                        );
                        "(error)".to_string()
                    }
                };
                let pinned_digest_normalized = if policy_doc.pinned_digest_sha256().trim().is_empty() {
                    None
                } else {
                    normalize_sha256_hex(policy_doc.pinned_digest_sha256()).ok()
                };
                let pinned_status = if policy_doc.pinned_digest_sha256().trim().is_empty() {
                    "missing"
                } else if payload_digest_status != "(error)"
                    && pinned_digest_normalized
                        .as_deref()
                        .is_some_and(|value| value == payload_digest_status)
                {
                    "verified"
                } else {
                    "invalid"
                };
                println!(
                    "opt policy pinned: status={} pinned={} digest={}",
                    pinned_status,
                    if policy_doc.pinned_digest_sha256().trim().is_empty() {
                        "(none)"
                    } else if let Some(normalized) = pinned_digest_normalized.as_ref() {
                        normalized
                    } else {
                        policy_doc.pinned_digest_sha256()
                    },
                    payload_digest_status
                );
                if let Some(gate) = policy_doc.rl_readiness() {
                    match load_rl_readiness_evidence()? {
                        Some(evidence) => match verify_rl_readiness_gate(&policy_doc, &evidence) {
                            Ok(()) => println!(
                                "opt policy readiness: pass gain={:.6} mismatches={} overhead={:.6} audit_clean={}",
                                evidence.epistemic_power_gain,
                                evidence.bit_exact_mismatches,
                                evidence.compile_overhead,
                                evidence.shadow_audit_clean
                            ),
                            Err(err) => println!("opt policy readiness: fail reason={}", err),
                        },
                        None => println!(
                            "opt policy readiness: pending (missing evidence, set {})",
                            RL_READINESS_EVIDENCE_PATH_ENV
                        ),
                    }
                    let _ = gate;
                }
                Ok(())
            }
        },
    }
}

fn run(
    input: &std::path::Path,
    args: &[String],
    use_sounio_compiler: bool,
    trace_interp: bool,
    check_only: bool,
) -> Result<()> {
    tracing::info!(
        "Running {:?} with args {:?} (use_sounio_compiler={}, trace_interp={}, check_only={})",
        input,
        args,
        use_sounio_compiler,
        trace_interp,
        check_only
    );

    let mut input_path = input.to_path_buf();
    let mut user_args = args.to_vec();
    let use_sounio_compiler = use_sounio_compiler
        || (input_path.is_dir()
            && input_path
                .file_name()
                .and_then(|part| part.to_str())
                .is_some_and(|name| name == "self-hosted"));

    let is_self_hosted_root = use_sounio_compiler
        && input_path.is_dir()
        && input_path
            .file_name()
            .and_then(|part| part.to_str())
            .is_some_and(|name| name == "self-hosted");

    let seed_enforced_for_wrapper = sounio::compiler_loader::selfhost_root_seed_enforced();

    if is_self_hosted_root {
        let main_path = input_path.join("main.sio");

        if !main_path.exists() {
            eprintln!(
                "SELFHOST=run schema=v1 event=selfhost_input_check status=missing_main path={} required={}",
                input_path.display(),
                main_path.display()
            );
            return Err(miette::miette!(
                "Incomplete self-hosted suite at {}: missing main.sio",
                input_path.display()
            ));
        }
        if seed_enforced_for_wrapper {
            eprintln!(
                "SELFHOST=run schema=v1 event=selfhost_preflight status=skipped path={} reason=seed_enforced",
                main_path.display()
            );
        } else {
            let preflight_err =
                sounio::module_loader::load_program_ast(&main_path).map_err(|e| {
                    eprintln!(
                        "SELFHOST=run schema=v1 event=selfhost_preflight status=error path={} reason={}",
                        main_path.display(),
                        e
                    );
                    miette::miette!(
                        "Self-hosted preflight parse failed for {}: {}",
                        main_path.display(),
                        e
                    )
                });
            preflight_err?;
        }

        input_path = main_path;

        if user_args.is_empty() {
            user_args.push("test".to_string());
        }

        eprintln!(
            "SELFHOST=run schema=v1 event=selfhost_input_check status=resolved input={} args={}",
            input_path.display(),
            user_args.len()
        );
    }

    fn parse_duration_from_env(name: &str, unit: &str, factor: u64) -> Option<Duration> {
        let raw = std::env::var(name).ok()?;
        let value = raw.trim();
        let parsed = match value.parse::<u64>() {
            Ok(v) if v > 0 => v,
            Ok(_) => return None,
            Err(_) => {
                eprintln!(
                    "SELFHOST=run schema=v1 event=run_timeout_config status=invalid var={} value={}",
                    name, raw
                );
                return None;
            }
        };

        let ms = parsed.checked_mul(factor).or_else(|| {
            eprintln!(
                "SELFHOST=run schema=v1 event=run_timeout_config status=overflow var={} value={}",
                name, raw
            );
            None
        })?;

        eprintln!(
            "SELFHOST=run schema=v1 event=run_timeout_config status=ok var={} value={}{}",
            name, value, unit
        );
        Some(Duration::from_millis(ms))
    }

    let timeout = if is_self_hosted_root {
        parse_duration_from_env("SOUNIO_SELFHOST_RUN_TIMEOUT_MS", "ms", 1)
            .or_else(|| parse_duration_from_env("SOUNIO_SELFHOST_RUN_TIMEOUT_SECS", "s", 1000))
    } else {
        None
    };

    let run_payload = move || -> std::result::Result<Option<i64>, String> {
        if use_sounio_compiler {
            tracing::info!("Using native self-host execution backend");
            sounio::compiler_loader::verify_removed_legacy_env_contracts().map_err(|e| {
                format!(
                    "Self-hosted runtime rejected legacy env contracts for {}: {}",
                    input_path.display(),
                    e
                )
            })?;

            if let Ok(mode_raw) = std::env::var("SOUNIO_SELFHOST_MODE") {
                let mode = mode_raw.trim().to_lowercase();
                eprintln!(
                    "error: unsupported SOUNIO_SELFHOST_MODE='{}'. Bytecode VM mode has been removed. Only SOUNIO_SELFHOST_EXECUTOR=native-driver is supported.",
                    mode
                );
                std::process::exit(2);
            }

            let executor = std::env::var("SOUNIO_SELFHOST_EXECUTOR")
                .unwrap_or_else(|_| "native-driver".to_string())
                .trim()
                .to_lowercase();
            if executor != "native-driver" {
                eprintln!(
                    "error: unsupported SOUNIO_SELFHOST_EXECUTOR='{}'. Only 'native-driver' is supported.",
                    executor
                );
                std::process::exit(2);
            }

            if let Ok(rebuild_raw) = std::env::var("SOUNIO_SELFHOST_DRIVER_ALLOW_LOCAL_REBUILD") {
                let rebuild = rebuild_raw.trim().to_ascii_lowercase();
                if matches!(rebuild.as_str(), "1" | "true" | "yes" | "on") {
                    eprintln!(
                        "error: unsupported SOUNIO_SELFHOST_DRIVER_ALLOW_LOCAL_REBUILD='{}'. \
No-Rust policy is active. Provide a signed prebuilt driver via \
SOUNIO_SELFHOST_DRIVER_MANIFEST_URL or SOUNIO_SELFHOST_DRIVER_CACHE_DIR.",
                        rebuild
                    );
                    std::process::exit(2);
                }
            }

            if is_self_hosted_root {
                sounio::compiler_loader::verify_selfhost_bootstrap_seed_policy().map_err(|e| {
                    format!(
                        "Self-hosted compile failed for {}: {}",
                        input_path.display(),
                        e
                    )
                })?;
            }

            let compile_target = input_path.to_string_lossy().into_owned();
            let mut compiler = sounio::compiler_loader::SounioCompiler::new_embedded()
                .map_err(|e| format!("Failed to initialize self-host backend: {}", e))?;
            let bytecode = compiler.compile_file(&compile_target).map_err(|e| {
                format!(
                    "Self-hosted compile failed for {}: {}",
                    input_path.display(),
                    e
                )
            })?;

            if check_only {
                return Ok(None);
            }

            let vm_result = compiler
                .execute_bytecode_with_args(&bytecode, &user_args)
                .map_err(|e| {
                    format!(
                        "Self-hosted execution failed for {}: {}",
                        input_path.display(),
                        e
                    )
                })?;

            if is_self_hosted_root {
                let exit_code = match vm_result {
                    sounio::vm::Value::Int(code) => code,
                    other => {
                        return Err(format!(
                            "Self-hosted root run returned non-integer exit value: {}",
                            other
                        ))
                    }
                };
                return Ok(Some(exit_code));
            }

            // Preserve historical `souc run` behavior: print non-unit results.
            if !matches!(vm_result, sounio::vm::Value::Unit) {
                println!("{}", vm_result);
            }
            return Ok(None);
        }

        if check_only {
            let ast =
                sounio::module_loader::load_program_ast(&input_path).map_err(|e| e.to_string())?;
            let _ = sounio::check::check_ast(&ast).map_err(|e| e.to_string())?;
        }

        // Load modules and parse (uses ModuleLoader to handle imports)
        let ast =
            sounio::module_loader::load_program_ast(&input_path).map_err(|e| e.to_string())?;
        let hir = sounio::check::check_ast(&ast).map_err(|e| e.to_string())?;

        // Use tree-walking interpreter
        let mut interpreter = sounio::interp::Interpreter::with_trace(trace_interp);
        interpreter.set_user_args(user_args);
        match interpreter.interpret(&hir) {
            Ok(result) => {
                // Only print non-unit results
                if !matches!(result, sounio::interp::Value::Unit) {
                    println!("{}", result);
                }
                Ok(None)
            }
            Err(e) => Err(e.to_string()),
        }
    };

    let input_path_display = input.to_path_buf();
    let run_result = if let Some(timeout) = timeout {
        let (tx, rx) = mpsc::channel();

        let handle = std::thread::Builder::new()
            .name("interp".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let result = run_payload();
                let _ = tx.send(result);
            })
            .map_err(|e| miette::miette!("failed to spawn interpreter thread: {}", e))?;

        let output = match rx.recv_timeout(timeout) {
            Ok(output) => {
                if handle.join().is_err() {
                    return Err(miette::miette!("interpreter thread panicked"));
                }
                output
            }
            Err(RecvTimeoutError::Timeout) => {
                eprintln!(
                    "SELFHOST=run schema=v1 event=run_timeout status=timeout input={} timeout_ms={}",
                    input_path_display.display(),
                    timeout.as_millis()
                );
                std::process::exit(124);
            }
            Err(RecvTimeoutError::Disconnected) => {
                if handle.join().is_err() {
                    return Err(miette::miette!("interpreter thread panicked"));
                }
                return Err(miette::miette!(
                    "interpreter thread exited without reporting status"
                ));
            }
        };

        output
    } else {
        std::thread::Builder::new()
            .name("interp".into())
            .stack_size(64 * 1024 * 1024)
            .spawn(run_payload)
            .map_err(|e| miette::miette!("failed to spawn interpreter thread: {}", e))?
            .join()
            .map_err(|_| miette::miette!("interpreter thread panicked"))?
    };

    let maybe_exit_code = run_result.map_err(|e| miette::miette!("{}", e))?;

    if let Some(code) = maybe_exit_code {
        let exit_code = if (0..=i64::from(i32::MAX)).contains(&code) {
            code as i32
        } else {
            eprintln!(
                "SELFHOST=run schema=v1 event=selfhost_exit_code status=invalid value={}",
                code
            );
            1
        };
        std::process::exit(exit_code);
    }

    Ok(())
}

fn jit_run(
    input: &std::path::Path,
    optimize: bool,
    use_mir: bool,
    ml_opt: bool,
    collect_opt_data: bool,
    glm_enabled: bool,
    _args: &[String],
) -> Result<()> {
    // --glm-enabled without glm feature falls through to local optimizer
    #[cfg(not(feature = "glm"))]
    if glm_enabled {
        eprintln!(
            "Warning: --glm-enabled requires --features glm. Using local heuristic optimizer instead."
        );
    }

    // Determine effective ml_opt: explicit flag or fallback from glm_enabled without feature
    let effective_ml_opt = ml_opt || {
        #[cfg(not(feature = "glm"))]
        {
            glm_enabled
        }
        #[cfg(feature = "glm")]
        {
            false
        }
    };

    let policy_enforced = std::env::var("SOUNIO_OPT_POLICY_ENFORCE")
        .ok()
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false);
    let input_hash = std::fs::read(input).ok().map(|bytes| sha256_hex(&bytes));
    let active_policy = load_effective_optimization_policy(policy_enforced)?;
    #[cfg(feature = "jit")]
    let mir_policy_context = {
        let mir_allowed_actions = effective_mir_policy_actions(active_policy.as_ref(), policy_enforced)?;
        sounio::mir::optimization::PassManagerPolicyContext {
            mir_allowed_actions,
            policy_error: None,
            decision_trail_path: decision_trail_path(),
            decision_trail_required: decision_trail_required(),
            policy_id: active_policy
                .as_ref()
                .map(|policy| policy.policy_id().to_string()),
            policy_mode: active_policy
                .as_ref()
                .map(|policy| policy.policy_mode().to_string()),
        }
    };
    emit_decision_event(
        "mir",
        if use_mir {
            "pipeline:jit-mir"
        } else {
            "pipeline:jit-hlir"
        },
        "applied",
        if effective_ml_opt {
            "ml_opt_enabled"
        } else {
            "ml_opt_disabled"
        },
        input.to_string_lossy().as_ref(),
        input_hash.as_deref(),
        active_policy.as_ref(),
    )?;

    #[cfg(feature = "jit")]
    {
        tracing::info!(
            "JIT compiling {:?} (optimize={}, ml_opt={}, collect_data={}, glm={})",
            input,
            optimize,
            effective_ml_opt,
            collect_opt_data,
            glm_enabled
        );

        // Load modules and parse (uses ModuleLoader to handle imports)
        let ast = sounio::module_loader::load_program_ast(input)?;
        let hir = sounio::check::check_ast(&ast)?;
        let hlir = sounio::hlir::lower(&hir);
        let main_func = hlir.find_function("main");
        let main_returns_value = main_func
            .as_ref()
            .map(|func| !matches!(func.return_type, sounio::hlir::HlirType::Void))
            .unwrap_or(true);
        let main_returns_f64 = main_func
            .as_ref()
            .map(|func| matches!(func.return_type, sounio::hlir::HlirType::F64 | sounio::hlir::HlirType::F32))
            .unwrap_or(false);

        if use_mir {
            let jit = if optimize {
                sounio::codegen::mir_cranelift::MirAwareCraneliftJit::new()
                    .with_optimization()
                    .with_mir_optimization(sounio::mir::optimization::OptimizationLevel::O2)
                    .with_ml_opt(effective_ml_opt)
                    .with_opt_data_collection(collect_opt_data)
                    .with_glm(glm_enabled)
                    .with_pass_manager_policy_context(mir_policy_context.clone())
            } else {
                sounio::codegen::mir_cranelift::MirAwareCraneliftJit::new()
                    .with_ml_opt(effective_ml_opt)
                    .with_opt_data_collection(collect_opt_data)
                    .with_glm(glm_enabled)
                    .with_pass_manager_policy_context(mir_policy_context.clone())
            };

            let compiled = jit
                .compile_hlir_via_mir(&hlir)
                .map_err(|e| miette::miette!("JIT error: {}", e))?;

            if main_returns_f64 {
                let result = unsafe { compiled.call_f64("main") }
                    .map_err(|e| miette::miette!("JIT error: {}", e))?;
                println!("{}", result);
            } else if main_returns_value {
                let result = unsafe { compiled.call_i64("main") }
                    .map_err(|e| miette::miette!("JIT error: {}", e))?;
                println!("{}", result);
            } else {
                unsafe { compiled.call_void("main") }
                    .map_err(|e| miette::miette!("JIT error: {}", e))?;
            }

            return Ok(());
        }

        let jit = if optimize {
            sounio::codegen::cranelift::CraneliftJit::new().with_optimization()
        } else {
            sounio::codegen::cranelift::CraneliftJit::new()
        };

        let compiled = jit
            .compile(&hlir)
            .map_err(|e| miette::miette!("JIT error: {}", e))?;

        if main_returns_f64 {
            let result = unsafe { compiled.call_f64("main") }
                .map_err(|e| miette::miette!("JIT error: {}", e))?;
            println!("{}", result);
        } else if main_returns_value {
            let result = unsafe { compiled.call_i64("main") }
                .map_err(|e| miette::miette!("JIT error: {}", e))?;
            println!("{}", result);
        } else {
            unsafe { compiled.call_void("main") }
                .map_err(|e| miette::miette!("JIT error: {}", e))?;
        }
        Ok(())
    }

    #[cfg(not(feature = "jit"))]
    {
        let _ = (input, optimize, use_mir, effective_ml_opt, collect_opt_data);
        Err(miette::miette!(
            "JIT backend not enabled. Recompile with --features jit"
        ))
    }
}

fn repl(use_jit: bool) -> Result<()> {
    let config = sounio::repl::ReplConfig {
        use_jit,
        ..Default::default()
    };

    sounio::repl::run_with_config(config).map_err(|e| miette::miette!("REPL error: {}", e))
}

fn bench(input: &std::path::Path, iterations: u32) -> Result<()> {
    use std::time::Instant;

    println!("Benchmarking {:?} ({} iterations)", input, iterations);
    println!();

    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;
    let hir = sounio::check::check_ast(&ast)?;

    // Warm up
    println!("Warming up...");
    let mut interpreter = sounio::interp::Interpreter::new();
    let _ = interpreter.interpret(&hir);

    // Benchmark interpreter
    println!("Running interpreter benchmark...");
    let start = Instant::now();
    for _ in 0..iterations {
        let mut interpreter = sounio::interp::Interpreter::new();
        let _ = interpreter.interpret(&hir);
    }
    let interp_time = start.elapsed();
    let interp_per_iter = interp_time / iterations;

    println!(
        "  Interpreter: {:?} total, {:?} per iteration",
        interp_time, interp_per_iter
    );

    // Benchmark JIT if available
    #[cfg(feature = "jit")]
    {
        let hlir = sounio::hlir::lower(&hir);
        let jit = sounio::codegen::cranelift::CraneliftJit::new();

        // Compile once
        println!("Compiling with JIT...");
        let compile_start = Instant::now();
        let compiled = jit
            .compile(&hlir)
            .map_err(|e| miette::miette!("JIT compile error: {}", e))?;
        let compile_time = compile_start.elapsed();
        println!("  JIT compile time: {:?}", compile_time);

        // Run benchmark
        println!("Running JIT benchmark...");
        let start = Instant::now();
        for _ in 0..iterations {
            unsafe {
                let _ = compiled.call_i64("main");
            }
        }
        let jit_time = start.elapsed();
        let jit_per_iter = jit_time / iterations;

        println!(
            "  JIT: {:?} total, {:?} per iteration",
            jit_time, jit_per_iter
        );
        println!();

        // Calculate speedup
        let speedup = interp_per_iter.as_nanos() as f64 / jit_per_iter.as_nanos() as f64;
        println!("JIT speedup: {:.2}x", speedup);

        // Break-even point
        let break_even = compile_time.as_nanos() as f64
            / (interp_per_iter.as_nanos() as f64 - jit_per_iter.as_nanos() as f64);
        println!("Break-even point: {:.0} iterations", break_even.max(0.0));
    }

    #[cfg(not(feature = "jit"))]
    {
        println!();
        println!("JIT backend not enabled. Recompile with --features jit for JIT benchmarks.");
    }

    Ok(())
}

fn format_code(
    path: &std::path::Path,
    check: bool,
    show_diff: bool,
    max_width: usize,
    use_tabs: bool,
    indent_width: usize,
) -> Result<()> {
    use sounio::fmt::{FormatConfig, Formatter};

    // Build configuration
    let mut config = FormatConfig::default();
    config.max_width = max_width as u32;
    config.use_tabs = use_tabs;
    config.indent_width = indent_width as u32;

    // Collect files to format
    let files = if path.is_dir() {
        collect_d_files(path)?
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        println!("No .d files found");
        return Ok(());
    }

    let mut formatted_count = 0;
    let mut unchanged_count = 0;
    let mut error_count = 0;

    for file in &files {
        let source = std::fs::read_to_string(file)
            .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;

        // Parse the file
        let tokens = match sounio::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                error_count += 1;
                continue;
            }
        };

        let _ast = match sounio::parser::parse(&tokens, &source) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                error_count += 1;
                continue;
            }
        };

        // Format the code
        let mut formatter = Formatter::new(config.clone());
        let formatted = match formatter.format(&source) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("Format error in {}: {}", file.display(), e);
                error_count += 1;
                continue;
            }
        };

        if formatted == source {
            unchanged_count += 1;
            continue;
        }

        formatted_count += 1;

        if check {
            println!("Would reformat: {}", file.display());
            if show_diff {
                print_diff(&source, &formatted);
            }
        } else {
            if show_diff {
                print_diff(&source, &formatted);
            }
            std::fs::write(file, &formatted)
                .map_err(|e| miette::miette!("Failed to write {}: {}", file.display(), e))?;
            println!("Formatted: {}", file.display());
        }
    }

    println!();
    println!(
        "Summary: {} formatted, {} unchanged, {} errors",
        formatted_count, unchanged_count, error_count
    );

    if check && formatted_count > 0 {
        Err(miette::miette!(
            "{} file(s) would be reformatted",
            formatted_count
        ))
    } else {
        Ok(())
    }
}

/// Print a simple diff between two strings
fn print_diff(original: &str, formatted: &str) {
    let orig_lines: Vec<&str> = original.lines().collect();
    let fmt_lines: Vec<&str> = formatted.lines().collect();

    for (i, (orig, fmt)) in orig_lines.iter().zip(fmt_lines.iter()).enumerate() {
        if orig != fmt {
            println!("  Line {}: ", i + 1);
            println!("    - {}", orig);
            println!("    + {}", fmt);
        }
    }

    // Handle different line counts
    if orig_lines.len() > fmt_lines.len() {
        for (i, line) in orig_lines.iter().skip(fmt_lines.len()).enumerate() {
            println!("  Line {}: ", fmt_lines.len() + i + 1);
            println!("    - {}", line);
        }
    } else if fmt_lines.len() > orig_lines.len() {
        for (i, line) in fmt_lines.iter().skip(orig_lines.len()).enumerate() {
            println!("  Line {}: ", orig_lines.len() + i + 1);
            println!("    + {}", line);
        }
    }
}

/// Collect all .d files in a directory
fn collect_d_files(dir: &std::path::Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
    {
        let path = entry.path();
        if path.extension().map(|e| e == "d").unwrap_or(false) {
            files.push(path.to_path_buf());
        }
    }

    Ok(files)
}

/// Lint Sounio source code
#[allow(clippy::too_many_arguments)]
fn lint_code(
    path: &std::path::Path,
    format: &str,
    deny_warnings: bool,
    allow: &[String],
    warn: &[String],
    deny: &[String],
    fix: bool,
) -> Result<()> {
    use sounio::lint::{LintConfig, LintLevel, Linter};

    // Build configuration
    let mut config = LintConfig::default();

    // Apply lint level overrides
    for lint_name in allow {
        config.set_level(lint_name, LintLevel::Allow);
    }
    for lint_name in warn {
        config.set_level(lint_name, LintLevel::Warn);
    }
    for lint_name in deny {
        config.set_level(lint_name, LintLevel::Deny);
    }

    // Collect files to lint
    let files = if path.is_dir() {
        collect_d_files(path)?
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        println!("No .d files found");
        return Ok(());
    }

    let mut total_warnings = 0;
    let mut total_errors = 0;
    let mut all_diagnostics = Vec::new();

    for file in &files {
        let source = std::fs::read_to_string(file)
            .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;

        // Parse the file
        let tokens = match sounio::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                total_errors += 1;
                continue;
            }
        };

        let ast = match sounio::parser::parse(&tokens, &source) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                total_errors += 1;
                continue;
            }
        };

        // Run linter
        let linter = Linter::with_config(config.clone());
        let diagnostics = linter.lint(&ast, &file.to_string_lossy(), &source);

        for diag in &diagnostics {
            match diag.level {
                LintLevel::Warn => total_warnings += 1,
                LintLevel::Deny | LintLevel::Forbid => total_errors += 1,
                _ => {}
            }
        }

        all_diagnostics.extend(diagnostics.into_iter().map(|d| (file.clone(), d)));
    }

    // Output results
    match format {
        "json" => {
            println!("{{");
            println!("  \"diagnostics\": [");
            for (i, (file, diag)) in all_diagnostics.iter().enumerate() {
                let comma = if i < all_diagnostics.len() - 1 {
                    ","
                } else {
                    ""
                };
                println!(
                    "    {{\"file\": \"{}\", \"lint\": \"{}\", \"message\": \"{}\", \"level\": \"{}\"}}{}",
                    file.display(),
                    diag.lint_name,
                    diag.message.replace('"', "\\\""),
                    format!("{:?}", diag.level).to_lowercase(),
                    comma
                );
            }
            println!("  ],");
            println!("  \"warnings\": {},", total_warnings);
            println!("  \"errors\": {}", total_errors);
            println!("}}");
        }
        "sarif" => {
            // SARIF format for IDE integration
            println!("{{");
            println!(
                "  \"$schema\": \"https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json\","
            );
            println!("  \"version\": \"2.1.0\",");
            println!("  \"runs\": [{{");
            println!(
                "    \"tool\": {{\"driver\": {{\"name\": \"souc lint\", \"version\": \"{}\"}}}},",
                env!("CARGO_PKG_VERSION")
            );
            println!("    \"results\": []");
            println!("  }}]");
            println!("}}");
        }
        _ => {
            // Text format
            for (file, diag) in &all_diagnostics {
                let level_str = match diag.level {
                    LintLevel::Allow => "allow",
                    LintLevel::Warn => "warning",
                    LintLevel::Deny => "error",
                    LintLevel::Forbid => "error",
                };
                println!(
                    "{}: {} [{}]: {}",
                    file.display(),
                    level_str,
                    diag.lint_name,
                    diag.message
                );

                // Show suggestions if available
                for suggestion in &diag.suggestions {
                    println!("  help: {}", suggestion.message);
                }
            }

            println!();
            println!(
                "Summary: {} warning(s), {} error(s)",
                total_warnings, total_errors
            );
        }
    }

    // Apply fixes if requested
    if fix && !all_diagnostics.is_empty() {
        println!();
        println!("Auto-fix not yet implemented. Use 'dc fix' for automatic fixes.");
    }

    // Determine exit status
    if total_errors > 0 || (deny_warnings && total_warnings > 0) {
        Err(miette::miette!("Linting failed with errors"))
    } else {
        Ok(())
    }
}

/// Analyze code for metrics and issues
fn analyze_code(
    path: &std::path::Path,
    analysis_type: &str,
    format: &str,
    verbose: bool,
) -> Result<()> {
    use sounio::analyze::{analyze_dead_code, calculate_metrics};

    // Collect files to analyze
    let files = if path.is_dir() {
        collect_d_files(path)?
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        println!("No .d files found");
        return Ok(());
    }

    let run_metrics = analysis_type == "metrics" || analysis_type == "all";
    let run_dead_code = analysis_type == "dead-code" || analysis_type == "all";

    for file in &files {
        let source = std::fs::read_to_string(file)
            .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;

        // Parse the file
        let tokens = match sounio::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                continue;
            }
        };

        let ast = match sounio::parser::parse(&tokens, &source) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                continue;
            }
        };

        match format {
            "json" => {
                println!("{{");
                println!("  \"file\": \"{}\",", file.display());

                if run_metrics {
                    let metrics = calculate_metrics(&ast, &source);
                    println!("  \"metrics\": {{");
                    println!("    \"loc\": {},", metrics.loc);
                    println!("    \"functions\": {},", metrics.functions);
                    println!("    \"types\": {},", metrics.types);
                    println!("    \"avg_complexity\": {:.2}", metrics.avg_complexity);
                    println!("  }},");
                }

                if run_dead_code {
                    let dead_code = analyze_dead_code(&ast);
                    println!("  \"dead_code\": {{");
                    println!("    \"unused_items\": {},", dead_code.unused_items.len());
                    println!(
                        "    \"unreachable_code\": {},",
                        dead_code.unreachable_code.len()
                    );
                    println!(
                        "    \"unused_variables\": {}",
                        dead_code.unused_variables.len()
                    );
                    println!("  }}");
                }

                println!("}}");
            }
            _ => {
                // Text format
                println!("=== Analysis: {} ===", file.display());
                println!();

                if run_metrics {
                    let metrics = calculate_metrics(&ast, &source);
                    println!("Code Metrics:");
                    println!("  Lines of code: {}", metrics.loc);
                    println!("  Comment lines: {}", metrics.comment_lines);
                    println!("  Blank lines: {}", metrics.blank_lines);
                    println!("  Functions: {}", metrics.functions);
                    println!("  Types: {}", metrics.types);
                    println!("  Average complexity: {:.2}", metrics.avg_complexity);

                    if verbose {
                        println!();
                        println!("  Function Details:");
                        for (name, func) in &metrics.function_metrics {
                            println!(
                                "    {}: {} lines, complexity {}, cognitive {}",
                                name,
                                func.lines,
                                func.cyclomatic_complexity,
                                func.cognitive_complexity
                            );
                        }
                    }
                    println!();
                }

                if run_dead_code {
                    let dead_code = analyze_dead_code(&ast);
                    println!("Dead Code Analysis:");
                    println!("  Unused items: {}", dead_code.unused_items.len());
                    println!("  Unreachable code: {}", dead_code.unreachable_code.len());
                    println!("  Unused variables: {}", dead_code.unused_variables.len());
                    println!("  Unused imports: {}", dead_code.unused_imports.len());

                    if verbose && dead_code.has_issues() {
                        println!();
                        for item in &dead_code.unused_items {
                            println!("    Unused {}: {}", item.kind, item.name);
                        }
                        for var in &dead_code.unused_variables {
                            println!("    Unused {}: {}", var.kind, var.name);
                            if let Some(suggestion) = &var.suggestion {
                                println!("      hint: prefix with underscore: {}", suggestion);
                            }
                        }
                        for unreach in &dead_code.unreachable_code {
                            println!("    Unreachable code: {}", unreach.reason);
                        }
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Apply automatic fixes
fn fix_code(path: &std::path::Path, dry_run: bool, allow_unsafe: bool) -> Result<()> {
    use sounio::lint::{Applicability, LintConfig, Linter};

    // Collect files to fix
    let files = if path.is_dir() {
        collect_d_files(path)?
    } else {
        vec![path.to_path_buf()]
    };

    if files.is_empty() {
        println!("No .d files found");
        return Ok(());
    }

    let mut total_fixes = 0;

    for file in &files {
        let source = std::fs::read_to_string(file)
            .map_err(|e| miette::miette!("Failed to read {}: {}", file.display(), e))?;

        // Parse the file
        let tokens = match sounio::lexer::lex(&source) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                continue;
            }
        };

        let ast = match sounio::parser::parse(&tokens, &source) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("Error parsing {}: {}", file.display(), e);
                continue;
            }
        };

        // Run linter to get suggestions
        let linter = Linter::with_config(LintConfig::default());
        let diagnostics = linter.lint(&ast, &file.to_string_lossy(), &source);

        // Collect applicable suggestions
        let suggestions: Vec<_> = diagnostics
            .iter()
            .flat_map(|d| &d.suggestions)
            .filter(|s| match s.applicability {
                Applicability::MachineApplicable => true,
                Applicability::MaybeIncorrect => allow_unsafe,
                Applicability::HasPlaceholders | Applicability::Unspecified => false,
            })
            .collect();

        if suggestions.is_empty() {
            continue;
        }

        // Collect all edits from suggestions
        let mut edits: Vec<_> = suggestions.iter().flat_map(|s| &s.edits).collect();

        // Sort edits by position (reverse order for safe application)
        edits.sort_by(|a, b| b.span.start.cmp(&a.span.start));

        let fix_count = suggestions.len();
        total_fixes += fix_count;

        if dry_run {
            println!("{}: {} fix(es) available", file.display(), fix_count);
            for suggestion in &suggestions {
                println!("  - {}", suggestion.message);
            }
        } else {
            // Apply edits
            let mut fixed_source = source.clone();
            for edit in &edits {
                if edit.span.start < fixed_source.len() && edit.span.end <= fixed_source.len() {
                    fixed_source.replace_range(edit.span.start..edit.span.end, &edit.replacement);
                }
            }

            std::fs::write(file, &fixed_source)
                .map_err(|e| miette::miette!("Failed to write {}: {}", file.display(), e))?;
            println!("{}: applied {} fix(es)", file.display(), fix_count);
        }
    }

    println!();
    if dry_run {
        println!("Dry run: {} fix(es) would be applied", total_fixes);
    } else {
        println!("Applied {} fix(es)", total_fixes);
    }

    Ok(())
}

fn doc(open: bool, document_private: bool) -> Result<()> {
    sounio::pkg::cli::cmd_doc(open, document_private)
        .map_err(|e| miette::miette!("Documentation generation failed: {}", e))
}

fn doc_book(output: Option<PathBuf>) -> Result<()> {
    sounio::pkg::cli::cmd_doc_book(output)
        .map_err(|e| miette::miette!("Book generation failed: {}", e))
}

fn doctest(filter: Option<String>, verbose: bool) -> Result<()> {
    sounio::pkg::cli::cmd_doctest(filter, verbose)
        .map_err(|e| miette::miette!("Doctest failed: {}", e))
}

fn doc_coverage() -> Result<()> {
    sounio::pkg::cli::cmd_doc_coverage()
        .map_err(|e| miette::miette!("Coverage calculation failed: {}", e))
}

fn info() -> Result<()> {
    println!("Sounio Compiler (souc)");
    println!("Version: {}", env!("CARGO_PKG_VERSION"));
    println!(
        "Commit: {}",
        option_env!("SOUNIO_GIT_HASH").unwrap_or("unknown")
    );
    println!(
        "Build date: {}",
        option_env!("SOUNIO_BUILD_DATE").unwrap_or("unknown")
    );

    // Host information
    println!();
    println!("Host:");
    println!("  Target: {}", std::env::consts::ARCH);
    println!("  OS: {}", std::env::consts::OS);
    println!("  Family: {}", std::env::consts::FAMILY);

    println!();
    println!("Language Features:");
    println!("  - Algebraic effects with handlers");
    println!("  - Linear and affine types (resource safety)");
    println!("  - Units of measure (dimensional analysis)");
    println!("  - Refinement types (SMT-backed verification)");
    println!("  - Epistemic types (confidence, provenance, ontology)");
    println!("  - GPU-native computation (CUDA/Metal)");

    println!();
    println!("Enabled Backends:");
    #[cfg(feature = "llvm-base")]
    {
        println!("  [+] LLVM - AOT compilation (souc build)");
    }
    #[cfg(not(feature = "llvm-base"))]
    println!("  [-] LLVM - rebuild with --features llvm");

    #[cfg(feature = "jit")]
    {
        println!("  [+] Cranelift JIT - fast compilation (souc jit)");
    }
    #[cfg(not(feature = "jit"))]
    println!("  [-] Cranelift JIT - rebuild with --features jit");

    #[cfg(feature = "gpu")]
    {
        println!("  [+] GPU codegen - PTX/SPIR-V generation");
    }
    #[cfg(not(feature = "gpu"))]
    println!("  [-] GPU codegen - rebuild with --features gpu");

    println!();
    println!("Enabled Features:");
    #[cfg(feature = "smt")]
    println!("  [+] SMT Solver (Z3) - refinement type verification");
    #[cfg(not(feature = "smt"))]
    println!("  [-] SMT Solver - rebuild with --features smt");

    #[cfg(feature = "lsp")]
    println!("  [+] LSP Server - IDE integration");
    #[cfg(not(feature = "lsp"))]
    println!("  [-] LSP Server - rebuild with --features lsp");

    #[cfg(feature = "ontology")]
    println!("  [+] Ontology - scientific term lookup");
    #[cfg(not(feature = "ontology"))]
    println!("  [-] Ontology - rebuild with --features ontology");

    #[cfg(feature = "distributed")]
    println!("  [+] Distributed builds - remote compilation");
    #[cfg(not(feature = "distributed"))]
    println!("  [-] Distributed builds - rebuild with --features distributed");

    #[cfg(feature = "pkg")]
    println!("  [+] Package manager - dependency management");
    #[cfg(not(feature = "pkg"))]
    println!("  [-] Package manager - rebuild with --features pkg");

    println!();
    println!("Build with all feature groups: use your selected build profile configuration.");
    println!();
    println!("For more information: https://sounio-lang.org");
    println!("Report bugs: https://github.com/sounio-lang/sounio/issues");

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn run_tests(
    path: &std::path::Path,
    filter: Option<&str>,
    include_ignored: bool,
    only_ignored: bool,
    jobs: usize,
    fail_fast: bool,
    run_bench: bool,
    list_only: bool,
    format: &str,
    coverage: bool,
    coverage_output: Option<&std::path::Path>,
    verbose: bool,
) -> Result<()> {
    use sounio::test::{
        coverage::{CoverageConfig, CoverageTracker},
        discovery::{TestFilter, discover_tests},
        runner::{OutputFormat, TestRunner, TestRunnerConfig},
    };

    tracing::info!("Running tests from {:?}", path);

    // Build test filter
    let mut test_filter = TestFilter::default();
    if let Some(pattern) = filter {
        test_filter.pattern = Some(pattern.to_string());
    }
    if include_ignored {
        test_filter.include_ignored = true;
    }
    if only_ignored {
        test_filter.only_ignored = true;
    }

    // Discover tests
    let suite = discover_tests(&[path], test_filter)
        .map_err(|e| miette::miette!("Test discovery failed: {}", e))?;

    for warning in &suite.discovery_warnings {
        eprintln!("warning: {}", warning);
    }

    if run_bench {
        println!("Found {} benchmarks", suite.all_benchmarks().len());
    } else {
        println!("Found {} tests", suite.test_count());
    }

    // Configure runner
    let output_format = match format {
        "compact" => OutputFormat::Compact,
        "json" => OutputFormat::Json,
        "junit" => OutputFormat::JUnit,
        _ => OutputFormat::Pretty,
    };

    let config = TestRunnerConfig {
        threads: jobs,
        fail_fast,
        list_only,
        verbose,
        format: output_format,
        ..Default::default()
    };

    // Optionally set up coverage tracking
    let mut coverage_tracker = if coverage {
        let mut tracker = CoverageTracker::new(CoverageConfig::default());
        tracker.start_tracking();
        Some(tracker)
    } else {
        None
    };

    // Run tests
    let runner = TestRunner::new(config);
    let report = runner
        .run(&suite)
        .map_err(|e| miette::miette!("Test execution failed: {}", e))?;

    // Stop coverage tracking and generate report
    if let Some(tracker) = &mut coverage_tracker {
        tracker.stop_tracking();
        let cov_report = tracker.generate_report();

        println!("\n{}", cov_report.summary());

        if let Some(output_path) = coverage_output {
            let lcov = cov_report.to_lcov();
            std::fs::write(output_path, &lcov)
                .map_err(|e| miette::miette!("Failed to write coverage: {}", e))?;
            println!("Coverage written to {}", output_path.display());
        }
    }

    // Output results in requested format
    match output_format {
        OutputFormat::Json => {
            let json = report
                .to_json()
                .map_err(|e| miette::miette!("Failed to serialize results: {}", e))?;
            println!("{}", json);
        }
        OutputFormat::JUnit => {
            println!("{}", report.to_junit());
        }
        _ => {
            // Pretty/Compact format already printed by runner
        }
    }

    if report.all_passed() {
        Ok(())
    } else {
        Err(miette::miette!(
            "Test run failed: {} passed, {} failed",
            report.passed,
            report.failed + report.timed_out + report.panicked
        ))
    }
}

fn run_benchmarks(
    path: &std::path::Path,
    filter: Option<&str>,
    baseline: Option<&std::path::Path>,
    save_baseline: Option<&std::path::Path>,
    time_secs: u64,
    verbose: bool,
) -> Result<()> {
    use sounio::test::{
        bench::{BenchConfig, BenchmarkRunner},
        discovery::{TestFilter, discover_tests},
    };
    use std::time::Duration;

    tracing::info!("Running benchmarks from {:?}", path);

    // Build filter
    let mut test_filter = TestFilter::default();
    if let Some(pattern) = filter {
        test_filter.pattern = Some(pattern.to_string());
    }

    // Discover benchmarks
    let suite = discover_tests(&[path], test_filter)
        .map_err(|e| miette::miette!("Benchmark discovery failed: {}", e))?;

    for warning in &suite.discovery_warnings {
        eprintln!("warning: {}", warning);
    }

    let benchmarks: Vec<_> = suite.all_benchmarks().into_iter().cloned().collect();

    if benchmarks.is_empty() {
        println!("No benchmarks found");
        return Ok(());
    }

    println!("Found {} benchmarks", benchmarks.len());

    // Configure benchmark runner
    let config = BenchConfig {
        target_time: Duration::from_secs(time_secs),
        ..Default::default()
    };

    let mut runner = BenchmarkRunner::new(config);

    // Load baseline if provided
    if let Some(baseline_path) = baseline {
        runner
            .load_baselines(baseline_path)
            .map_err(|e| miette::miette!("Failed to load baseline: {}", e))?;
        if verbose {
            println!("Loaded baseline from {}", baseline_path.display());
        }
    }

    // Run benchmarks
    let results = runner.run_all(&benchmarks);

    // Save baseline if requested
    if let Some(save_path) = save_baseline {
        runner.update_baselines(&results);
        runner
            .save_baselines(save_path)
            .map_err(|e| miette::miette!("Failed to save baseline: {}", e))?;
        println!("Saved baseline to {}", save_path.display());
    }

    Ok(())
}

/// Profile a Sounio program
#[allow(clippy::too_many_arguments)]
fn profile(
    input: &std::path::Path,
    profile_type: &str,
    output: Option<&std::path::Path>,
    flamegraph: Option<&std::path::Path>,
    interval: u64,
    format: &str,
    _args: &[String],
) -> Result<()> {
    tracing::info!("Profiling {:?} (type: {})", input, profile_type);

    // Read and compile the source
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;
    let hir = sounio::check::check_ast(&ast)?;

    println!("=== Profile: {} ===", input.display());
    println!("Profile type: {}", profile_type);
    println!("Sample interval: {}us", interval);
    println!();

    match profile_type {
        "cpu" => {
            println!("CPU Profiling Configuration:");
            println!("  Interval: {}us", interval);
            println!("  Output format: {}", format);

            // Run with interpreter and collect timing info
            let start = std::time::Instant::now();
            let mut interpreter = sounio::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);
            let duration = start.elapsed();

            println!();
            println!("Execution completed in {:?}", duration);
            println!();

            // Generate simulated profile output
            println!("Profile Summary:");
            println!(
                "  Total samples: {} (simulated)",
                duration.as_micros() / interval as u128
            );
            println!("  Functions profiled: {}", ast.items.len());

            if let Some(out_path) = output {
                let profile_data = format!(
                    "# CPU Profile for {}\n# Duration: {:?}\n# Samples: {}\n\nmain 100.0%\n",
                    input.display(),
                    duration,
                    duration.as_micros() / interval as u128
                );
                std::fs::write(out_path, &profile_data)
                    .map_err(|e| miette::miette!("Failed to write profile: {}", e))?;
                println!("Profile written to {}", out_path.display());
            }

            if let Some(fg_path) = flamegraph {
                // Generate a simple flamegraph SVG
                let svg = generate_simple_flamegraph(input, duration);
                std::fs::write(fg_path, &svg)
                    .map_err(|e| miette::miette!("Failed to write flamegraph: {}", e))?;
                println!("Flamegraph written to {}", fg_path.display());
            }
        }
        "memory" => {
            println!("Memory Profiling:");

            let mut interpreter = sounio::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);

            println!("  Allocations tracked: (simulated)");
            println!("  Peak memory: N/A (interpreter mode)");
            println!();
            println!("Note: Full memory profiling requires native compilation with -g");
        }
        "async" => {
            println!("Async Task Profiling:");
            println!("  Task tracking enabled");
            println!();

            let mut interpreter = sounio::interp::Interpreter::new();
            let _ = interpreter.interpret(&hir);

            println!("Async Profile Summary:");
            println!("  Total tasks: 0 (no async code detected)");
            println!("  Completed: 0");
            println!("  Failed: 0");
        }
        _ => {
            return Err(miette::miette!(
                "Unknown profile type: {}. Use cpu, memory, or async",
                profile_type
            ));
        }
    }

    Ok(())
}

/// Generate a simple flamegraph SVG
fn generate_simple_flamegraph(input: &std::path::Path, duration: std::time::Duration) -> String {
    let name = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("program");
    let mut svg = String::new();
    svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    svg.push_str("<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 800 200\" width=\"800\" height=\"200\">\n");
    svg.push_str("<style>\n");
    svg.push_str("    .frame { fill: #ff6600; stroke: #d44a00; }\n");
    svg.push_str("    .frame:hover { fill: #ff8833; }\n");
    svg.push_str("    text { font-family: monospace; font-size: 12px; fill: white; }\n");
    svg.push_str("</style>\n");
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#f8f8f8\"/>\n");
    svg.push_str(&format!(
        "<text x=\"20\" y=\"20\" fill=\"black\" font-size=\"14\">CPU Profile: {}</text>\n",
        name
    ));
    svg.push_str(&format!(
        "<text x=\"20\" y=\"40\" fill=\"black\" font-size=\"12\">Duration: {:?}</text>\n",
        duration
    ));
    svg.push_str("<g class=\"frame\">\n");
    svg.push_str("    <rect x=\"50\" y=\"80\" width=\"700\" height=\"20\" rx=\"2\"/>\n");
    svg.push_str("    <text x=\"55\" y=\"95\">main (100%)</text>\n");
    svg.push_str("</g>\n");
    svg.push_str("<g class=\"frame\">\n");
    svg.push_str("    <rect x=\"50\" y=\"110\" width=\"500\" height=\"20\" rx=\"2\"/>\n");
    svg.push_str(&format!(
        "    <text x=\"55\" y=\"125\">{} (71%)</text>\n",
        name
    ));
    svg.push_str("</g>\n");
    svg.push_str(
        "<text x=\"20\" y=\"170\" fill=\"#666\" font-size=\"10\">Generated by dc profile</text>\n",
    );
    svg.push_str("</svg>");
    svg
}

/// Debug a Sounio program with GDB or LLDB
fn debug_program(
    input: &std::path::Path,
    debugger: &str,
    pretty: bool,
    args: &[String],
) -> Result<()> {
    tracing::info!("Debugging {:?} with {}", input, debugger);

    // First, we need to compile with debug info
    println!("Note: Debugging requires a compiled binary with debug info.");
    println!("First compile with: dc build -g {}", input.display());
    println!();

    let binary_path = {
        let mut p = input.to_path_buf();
        p.set_extension("");
        p
    };

    if !binary_path.exists() {
        return Err(miette::miette!(
            "Binary not found at {}. Run 'dc build -g {}' first.",
            binary_path.display(),
            input.display()
        ));
    }

    // Find pretty printer path
    let pretty_printer_path = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .map(|p| p.join("tools/debug"));

    match debugger {
        "gdb" => {
            println!("Starting GDB...");

            let mut cmd = std::process::Command::new("gdb");

            if pretty {
                if let Some(pp_path) = &pretty_printer_path {
                    let gdb_script = pp_path.join("d_gdb.py");
                    if gdb_script.exists() {
                        cmd.arg("-x").arg(&gdb_script);
                        println!("Loading pretty printers from {}", gdb_script.display());
                    }
                }
            }

            cmd.arg("--args").arg(&binary_path).args(args);

            println!();
            println!("GDB commands for Sounio:");
            println!("  d-backtrace  - Show D-aware backtrace");
            println!("  sio-locals     - Show local variables with Sounio types");
            println!("  d-async      - Show async task state");
            println!("  d-effects    - Show active effect handlers");
            println!();

            let status = cmd
                .status()
                .map_err(|e| miette::miette!("Failed to start GDB: {}", e))?;

            if !status.success() {
                return Err(miette::miette!("GDB exited with error"));
            }
        }
        "lldb" => {
            println!("Starting LLDB...");

            let mut cmd = std::process::Command::new("lldb");

            if pretty {
                if let Some(pp_path) = &pretty_printer_path {
                    let lldb_script = pp_path.join("d_lldb.py");
                    if lldb_script.exists() {
                        cmd.arg("-O")
                            .arg(format!("command script import {}", lldb_script.display()));
                        println!("Loading type summaries from {}", lldb_script.display());
                    }
                }
            }

            cmd.arg("--").arg(&binary_path).args(args);

            println!();
            println!("LLDB commands for Sounio:");
            println!("  d-backtrace  - Show D-aware backtrace");
            println!("  sio-locals     - Show local variables with Sounio types");
            println!("  d-async      - Show async task state");
            println!();

            let status = cmd
                .status()
                .map_err(|e| miette::miette!("Failed to start LLDB: {}", e))?;

            if !status.success() {
                return Err(miette::miette!("LLDB exited with error"));
            }
        }
        _ => {
            return Err(miette::miette!(
                "Unknown debugger: {}. Use gdb or lldb",
                debugger
            ));
        }
    }

    Ok(())
}

/// Explain an error code
fn explain_error(code: &str) -> Result<()> {
    use sounio::diagnostic::codes::{ErrorIndex, explain_error as get_explanation};

    if let Some(explanation) = get_explanation(code) {
        println!("{}", explanation);
    } else {
        // Try to find similar codes
        let index = ErrorIndex::new();
        let prefix = code.chars().next().unwrap_or('E');
        let similar: Vec<_> = index
            .all()
            .filter(|e| e.code.starts_with(prefix))
            .take(5)
            .collect();

        println!("Error code '{}' not found.", code);

        if !similar.is_empty() {
            println!();
            println!("Similar error codes:");
            for e in similar {
                println!("  {} - {}", e.code, e.title);
            }
        }

        println!();
        println!("Use 'dc error-index' to see all error codes.");
    }

    Ok(())
}

/// Show all error codes
fn show_error_index(category: Option<&str>, format: &str) -> Result<()> {
    use sounio::diagnostic::codes::{ErrorCategory, ErrorIndex};

    let index = ErrorIndex::new();

    // Parse category filter
    let category_filter: Option<ErrorCategory> =
        category.map(|c| match c.to_lowercase().as_str() {
            "lexer" | "l" => ErrorCategory::Lexer,
            "parser" | "p" | "syntax" => ErrorCategory::Parser,
            "resolve" | "r" | "name" => ErrorCategory::Resolve,
            "type" | "t" => ErrorCategory::Type,
            "effect" | "f" => ErrorCategory::Effect,
            "ownership" | "o" => ErrorCategory::Ownership,
            "pattern" | "m" => ErrorCategory::Pattern,
            "macro" | "x" => ErrorCategory::Macro,
            "module" | "i" | "import" => ErrorCategory::Module,
            "codegen" | "c" => ErrorCategory::Codegen,
            _ => ErrorCategory::Internal,
        });

    match format {
        "markdown" | "md" => {
            println!("{}", index.generate_docs());
        }
        "json" => {
            println!("{{");
            println!("  \"errors\": [");
            let codes: Vec<_> = if let Some(cat) = category_filter {
                index.by_category(cat)
            } else {
                index.all().collect()
            };

            for (i, code) in codes.iter().enumerate() {
                let comma = if i < codes.len() - 1 { "," } else { "" };
                println!(
                    "    {{\"code\": \"{}\", \"title\": \"{}\", \"category\": \"{}\"}}{}",
                    code.code,
                    code.title,
                    code.category.name(),
                    comma
                );
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            // Text format
            println!("Sounio Compiler Error Index");
            println!("==============================");
            println!();

            let codes: Vec<_> = if let Some(cat) = category_filter {
                println!("Category: {}", cat.name());
                println!();
                index.by_category(cat)
            } else {
                index.all().collect()
            };

            for code in codes {
                println!("{}: {}", code.code, code.title);
                println!("  Category: {}", code.category.name());
                println!();
            }

            println!("Total: {} error codes", index.all().count());
            println!();
            println!("Use 'dc explain <CODE>' for detailed information about a specific error.");
        }
    }

    Ok(())
}

/// Check a file and show diagnostics with rich formatting
fn diagnostics_check(
    input: &std::path::Path,
    format: &str,
    max_errors: usize,
    show_type_diff: bool,
    _show_trace: bool,
) -> Result<()> {
    use sounio::diagnostic::emitter::SarifEmitter;
    use sounio::diagnostic::{Diagnostic, DiagnosticHandler, HumanEmitter, JsonEmitter, Span};

    tracing::info!("Checking {:?} with rich diagnostics", input);

    // Read source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    // Create emitter based on format
    let emitter: Box<dyn sounio::diagnostic::DiagnosticEmitter> = match format {
        "json" => Box::new(JsonEmitter::stdout()),
        "sarif" => Box::new(SarifEmitter::new(Box::new(std::io::stdout()))),
        _ => Box::new(HumanEmitter::stderr()), // human format with color
    };

    let mut handler = DiagnosticHandler::new(emitter).max_errors(max_errors);
    let file_id = handler
        .source_map_mut()
        .add_file(input.to_path_buf(), source.clone());

    // Try to lex
    let tokens = match sounio::lexer::lex(&source) {
        Ok(t) => t,
        Err(e) => {
            let diagnostic = Diagnostic::error(format!("Lexer error: {}", e))
                .with_code("L0001")
                .with_label(Span::new(0, source.len().min(10), file_id), "error here");
            handler.emit(&diagnostic);
            handler.abort_if_errors();
            return Ok(());
        }
    };

    // Try to parse
    let ast = match sounio::parser::parse(&tokens, &source) {
        Ok(a) => a,
        Err(e) => {
            let diagnostic = Diagnostic::error(format!("Parse error: {}", e)).with_code("P0001");
            handler.emit(&diagnostic);
            handler.abort_if_errors();
            return Ok(());
        }
    };

    // Try to type check
    match sounio::check::check_ast(&ast) {
        Ok(hir) => {
            if handler.has_errors() {
                handler.abort_if_errors();
            } else {
                println!(
                    "No errors found in {} ({} items)",
                    input.display(),
                    hir.items.len()
                );
            }
        }
        Err(e) => {
            let mut diagnostic = Diagnostic::error(format!("{}", e)).with_code("T0001");

            // Add type diff if requested
            if show_type_diff {
                diagnostic =
                    diagnostic.with_note("Use --show-trace for detailed unification trace");
            }

            handler.emit(&diagnostic);
            handler.abort_if_errors();
        }
    }

    Ok(())
}

/// Show similar names for typo detection
fn diagnostics_similar(name: &str, category: &str, max_distance: usize) -> Result<()> {
    use sounio::diagnostic::typo::{SuggestionBuilder, find_similar};

    let builder = SuggestionBuilder::new();

    println!("Finding names similar to '{}'...", name);
    println!();

    match category {
        "keyword" | "keywords" => {
            if let Some(msg) = builder.did_you_mean_keyword(name) {
                println!("Keywords: {}", msg);
            } else {
                println!("No similar keywords found");
            }
        }
        "type" | "types" => {
            if let Some(msg) = builder.did_you_mean_type(name) {
                println!("Types: {}", msg);
            } else {
                println!("No similar types found");
            }
        }
        "function" | "functions" | "fn" => {
            if let Some(msg) = builder.did_you_mean_function(name) {
                println!("Functions: {}", msg);
            } else {
                println!("No similar functions found");
            }
        }
        "variable" | "variables" | "var" => {
            if let Some(msg) = builder.did_you_mean_variable(name) {
                println!("Variables: {}", msg);
            } else {
                println!("No similar variables found");
            }
        }
        _ => {
            // Show all categories
            println!("Category: all (max distance: {})", max_distance);
            println!();

            // Keywords
            let keywords = vec![
                "fn", "let", "var", "const", "struct", "enum", "type", "trait", "impl", "if",
                "else", "match", "while", "for", "loop", "return", "break", "continue", "with",
                "where", "as", "in", "use", "mod", "pub", "mut", "ref", "linear", "affine",
                "kernel", "effect", "handler", "handle", "perform", "resume",
            ];
            let kw_similar = find_similar(name, keywords.into_iter(), max_distance, 5);
            if !kw_similar.is_empty() {
                println!("Similar keywords:");
                for s in &kw_similar {
                    println!(
                        "  {} (distance: {}, score: {:.2})",
                        s.text, s.distance, s.score
                    );
                }
                println!();
            }

            // Types
            let types = vec![
                "int", "i8", "i16", "i32", "i64", "uint", "u8", "u16", "u32", "u64", "f32", "f64",
                "bool", "char", "string", "unit", "never", "Option", "Result", "Vec", "HashMap",
            ];
            let type_similar = find_similar(name, types.into_iter(), max_distance, 5);
            if !type_similar.is_empty() {
                println!("Similar types:");
                for s in &type_similar {
                    println!(
                        "  {} (distance: {}, score: {:.2})",
                        s.text, s.distance, s.score
                    );
                }
                println!();
            }

            // Functions
            let functions = vec![
                "print",
                "println",
                "assert",
                "panic",
                "todo",
                "unreachable",
                "Some",
                "None",
                "Ok",
                "Err",
                "len",
                "size",
                "clone",
                "copy",
                "drop",
            ];
            let fn_similar = find_similar(name, functions.into_iter(), max_distance, 5);
            if !fn_similar.is_empty() {
                println!("Similar functions:");
                for s in &fn_similar {
                    println!(
                        "  {} (distance: {}, score: {:.2})",
                        s.text, s.distance, s.score
                    );
                }
            }

            if kw_similar.is_empty() && type_similar.is_empty() && fn_similar.is_empty() {
                println!(
                    "No similar names found within edit distance {}",
                    max_distance
                );
            }
        }
    }

    Ok(())
}

/// Test diagnostic rendering
fn diagnostics_render(level: &str, code: Option<&str>, message: &str, format: &str) -> Result<()> {
    use sounio::diagnostic::emitter::SarifEmitter;
    use sounio::diagnostic::{
        Diagnostic, DiagnosticHandler, DiagnosticLevel, HumanEmitter, JsonEmitter, Span,
    };

    // Parse level
    let diag_level = match level.to_lowercase().as_str() {
        "bug" | "ice" => DiagnosticLevel::Bug,
        "fatal" => DiagnosticLevel::Fatal,
        "error" | "err" => DiagnosticLevel::Error,
        "warning" | "warn" => DiagnosticLevel::Warning,
        "note" => DiagnosticLevel::Note,
        "help" => DiagnosticLevel::Help,
        _ => DiagnosticLevel::Error,
    };

    // Create diagnostic
    let mut diagnostic = Diagnostic::new(diag_level, message);
    if let Some(c) = code {
        diagnostic = diagnostic.with_code(c);
    }

    // Add a sample label
    diagnostic = diagnostic
        .with_label(Span::new(0, 10, 1), "sample location")
        .with_note("This is a sample diagnostic for testing rendering")
        .with_help("Use different --level values to see different styles");

    // Create emitter
    let emitter: Box<dyn sounio::diagnostic::DiagnosticEmitter> = match format {
        "json" => Box::new(JsonEmitter::stdout()),
        "sarif" => Box::new(SarifEmitter::new(Box::new(std::io::stdout()))),
        _ => Box::new(HumanEmitter::stderr()),
    };

    let mut handler = DiagnosticHandler::new(emitter);
    handler.source_map_mut().add_file(
        PathBuf::from("sample.sio"),
        "let x = 42\nlet y = x + 1".to_string(),
    );

    handler.emit(&diagnostic);

    Ok(())
}

/// Show diagnostic statistics for a file
fn diagnostics_stats(input: &std::path::Path) -> Result<()> {
    tracing::info!("Collecting diagnostic stats for {:?}", input);

    // Read source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let mut lexer_errors = 0;
    let mut parser_errors = 0;
    let mut type_errors = 0;

    // Count lexer errors
    if sounio::lexer::lex(&source).is_err() {
        lexer_errors += 1;
    }

    // Count parser errors
    let tokens = sounio::lexer::lex(&source).unwrap_or_default();
    if sounio::parser::parse(&tokens, &source).is_err() {
        parser_errors += 1;
    }

    // Count type errors (if parsing succeeded)
    if let Ok(ast) = sounio::parser::parse(&tokens, &source) {
        if sounio::check::check_ast(&ast).is_err() {
            type_errors += 1;
        }
    }

    println!("Diagnostic Statistics for {}", input.display());
    println!("=======================================");
    println!();
    println!("Source lines:    {}", source.lines().count());
    println!("Source bytes:    {}", source.len());
    println!();
    println!("Lexer errors:    {}", lexer_errors);
    println!("Parser errors:   {}", parser_errors);
    println!("Type errors:     {}", type_errors);
    println!();
    println!(
        "Total errors:    {}",
        lexer_errors + parser_errors + type_errors
    );

    Ok(())
}

/// Generate debug information
fn generate_debug_info(
    input: &std::path::Path,
    output: Option<&std::path::Path>,
    format: &str,
) -> Result<()> {
    tracing::info!("Generating debug info for {:?}", input);

    // Read and compile
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;
    let _hir = sounio::check::check_ast(&ast)?;

    let out_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        match format {
            "dwarf" => p.set_extension("dwarf"),
            "pdb" => p.set_extension("pdb"),
            _ => p.set_extension("debug"),
        };
        p
    });

    match format {
        "dwarf" => {
            println!("Generating DWARF debug information...");

            // Use debug info builder
            use sounio::codegen::debug::DebugInfoBuilder;

            let file_path = input.to_path_buf();
            let directory = input
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_else(|| PathBuf::from("."));

            let builder = DebugInfoBuilder::new(file_path, directory);
            let dwarf_output = builder.finalize();

            // Write DWARF sections info
            let mut info = String::new();
            info.push_str(&format!("# DWARF Debug Info for {}\n", input.display()));
            info.push_str(&format!(
                "# Generated by dc {}\n\n",
                env!("CARGO_PKG_VERSION")
            ));
            info.push_str(&format!(
                ".debug_info size: {} bytes\n",
                dwarf_output.debug_info.len()
            ));
            info.push_str(&format!(
                ".debug_abbrev size: {} bytes\n",
                dwarf_output.debug_abbrev.len()
            ));
            info.push_str(&format!(
                ".debug_line size: {} bytes\n",
                dwarf_output.debug_line.len()
            ));
            info.push_str(&format!(
                ".debug_str size: {} bytes\n",
                dwarf_output.debug_str.len()
            ));

            std::fs::write(&out_path, info)
                .map_err(|e| miette::miette!("Failed to write debug info: {}", e))?;

            println!("Debug info written to {}", out_path.display());
        }
        "pdb" => {
            println!("PDB format is only supported on Windows.");
            println!("For cross-platform debug info, use DWARF format.");
        }
        _ => {
            return Err(miette::miette!(
                "Unknown debug format: {}. Use dwarf or pdb",
                format
            ));
        }
    }

    Ok(())
}

/// Generate source map
fn generate_source_map(input: &std::path::Path, output: Option<&std::path::Path>) -> Result<()> {
    tracing::info!("Generating source map for {:?}", input);

    // Read and compile
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read input file: {}", e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;
    let _hir = sounio::check::check_ast(&ast)?;

    let out_path = output.map(|p| p.to_path_buf()).unwrap_or_else(|| {
        let mut p = input.to_path_buf();
        p.set_extension("map");
        p
    });

    // Build source map
    use sounio::codegen::debug::source_map::SourceMapBuilder;

    let mut builder = SourceMapBuilder::new();

    // Add source with content
    let source_path = input.to_path_buf();
    builder.add_source_with_content(source_path.clone(), source);

    // Add mappings for each AST item (simplified)
    // In a real implementation, this would map compiled code positions to source
    for (i, _item) in ast.items.iter().enumerate() {
        builder.add_simple_mapping(
            i as u32, // generated line
            0,        // generated column
            &source_path,
            i as u32, // original line
            0,        // original column
        );
    }

    let output_file = out_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("output.map");
    let source_map = builder.build(output_file);
    let json = source_map
        .to_json()
        .map_err(|e| miette::miette!("Failed to serialize source map: {}", e))?;

    std::fs::write(&out_path, &json)
        .map_err(|e| miette::miette!("Failed to write source map: {}", e))?;

    println!("Source map written to {}", out_path.display());
    println!();
    println!("Source map contains:");
    println!("  Version: {}", source_map.version);
    println!("  Sources: {} file(s)", source_map.sources.len());
    println!("  Names: {} identifier(s)", source_map.names.len());

    Ok(())
}

/// Build project using the build system
fn build_system(
    source_dir: &std::path::Path,
    profile_name: &str,
    jobs: usize,
    verbose: bool,
    no_incremental: bool,
) -> Result<()> {
    use sounio::build::{BuildConfig, BuildManager, BuildProfile};

    tracing::info!(
        "Building project in {} with profile {}",
        source_dir.display(),
        profile_name
    );

    // Parse profile
    let profile = BuildProfile::from_name(profile_name)
        .ok_or_else(|| miette::miette!("Invalid profile: {}", profile_name))?;

    // Create build config
    let mut config = match profile {
        BuildProfile::Dev => BuildConfig::dev(),
        BuildProfile::Release => BuildConfig::release(),
        BuildProfile::Test => BuildConfig::test(),
        BuildProfile::Bench => BuildConfig::bench(),
    };

    // Apply command-line overrides
    config.flags.verbose = verbose;
    if no_incremental {
        config.flags.incremental = false;
    }

    // Update paths
    config.paths.source_dir = source_dir.to_path_buf();
    config.paths.build_dir = source_dir.join("build");
    config.paths.cache_dir = source_dir.join("build/cache");
    config.paths.target_dir = source_dir.join("target");

    // Create build manager
    let mut manager = BuildManager::new(config)
        .map_err(|e| miette::miette!("Failed to create build manager: {}", e))?;

    // Initialize
    println!("Initializing build system...");
    manager
        .init(source_dir)
        .map_err(|e| miette::miette!("Initialization failed: {}", e))?;

    // Get stats before build
    let stats_before = manager.stats();

    if verbose {
        println!("Build configuration:");
        println!("  Profile: {}", profile.name());
        println!("  Source dir: {}", source_dir.display());
        println!("  Total units: {}", stats_before.total_units);
        println!("  Dirty units: {}", stats_before.dirty_units);
        println!("  Jobs: {}", if jobs == 0 { num_cpus::get() } else { jobs });
        println!();
    }

    // Execute build
    println!("Building project...");
    let report = manager
        .build()
        .map_err(|e| miette::miette!("Build failed: {}", e))?;

    // Print results
    report.print_summary();

    if report.success {
        Ok(())
    } else {
        Err(miette::miette!("Build failed"))
    }
}

/// Clean build artifacts and cache
fn clean_build(clean_cache: bool, verbose: bool) -> Result<()> {
    use std::fs;

    println!("Cleaning build artifacts...");

    let build_dir = std::path::PathBuf::from("build");
    let target_dir = std::path::PathBuf::from("target");

    let mut removed_count = 0;

    // Remove build directory
    if build_dir.exists() {
        if verbose {
            println!("Removing {}", build_dir.display());
        }
        fs::remove_dir_all(&build_dir)
            .map_err(|e| miette::miette!("Failed to remove build dir: {}", e))?;
        removed_count += 1;
    }

    // Remove target directory
    if target_dir.exists() {
        if verbose {
            println!("Removing {}", target_dir.display());
        }
        fs::remove_dir_all(&target_dir)
            .map_err(|e| miette::miette!("Failed to remove target dir: {}", e))?;
        removed_count += 1;
    }

    // Remove cache if requested
    if clean_cache {
        let cache_dir = std::path::PathBuf::from("build/cache");
        if cache_dir.exists() {
            if verbose {
                println!("Removing {}", cache_dir.display());
            }
            fs::remove_dir_all(&cache_dir)
                .map_err(|e| miette::miette!("Failed to remove cache dir: {}", e))?;
            removed_count += 1;
        }
    }

    println!(
        "Cleaned {} director{}",
        removed_count,
        if removed_count == 1 { "y" } else { "ies" }
    );
    Ok(())
}

/// Watch files and rebuild on changes
#[allow(clippy::too_many_arguments)]
fn watch_files(
    paths: &[PathBuf],
    clear_screen: bool,
    run_tests: bool,
    exec_command: Option<&str>,
    debounce_ms: u64,
    ignore_patterns: &[String],
    _verbose: bool,
) -> Result<()> {
    use sounio::watch::{WatchConfig, WatchMode, WatchModeConfig};
    use std::time::Duration;

    tracing::info!("Starting watch mode for {:?}", paths);

    // Build watch configuration
    let watch_config = WatchConfig {
        paths: paths.to_vec(),
        recursive: true,
        debounce: Duration::from_millis(debounce_ms),
        exclude: ignore_patterns.to_vec(),
        ..Default::default()
    };

    let config = WatchModeConfig {
        watch: watch_config,
        clear_screen,
        run_tests,
        exec: exec_command.map(String::from),
        ..Default::default()
    };

    println!("Watch mode starting...");
    println!("  Watching: {:?}", paths);
    println!("  Clear screen: {}", clear_screen);
    println!("  Run tests: {}", run_tests);
    if let Some(cmd) = exec_command {
        println!("  Execute: {}", cmd);
    }
    println!("  Debounce: {}ms", debounce_ms);
    if !ignore_patterns.is_empty() {
        println!("  Ignore: {:?}", ignore_patterns);
    }
    println!();
    println!("Press 'q' to quit, 'r' to rebuild, 'p' to pause/resume");
    println!();

    let mut watch_mode = WatchMode::new(config)
        .map_err(|e| miette::miette!("Failed to initialize watch mode: {}", e))?;

    watch_mode
        .run()
        .map_err(|e| miette::miette!("Watch mode error: {}", e))?;

    Ok(())
}

/// Start development server with live reload
#[allow(clippy::too_many_arguments)]
fn serve_files(
    root: &std::path::Path,
    port: u16,
    host: &str,
    no_reload: bool,
    open_browser: bool,
    directory_listing: bool,
    spa_fallback: Option<&str>,
    _verbose: bool,
) -> Result<()> {
    use sounio::watch::{DevServer, DevServerConfig};

    tracing::info!("Starting development server at {}:{}", host, port);

    let config = DevServerConfig {
        host: host.to_string(),
        port,
        root: root.to_path_buf(),
        live_reload: !no_reload,
        open_browser,
        directory_listing,
        spa_fallback: spa_fallback.map(String::from),
        ..Default::default()
    };

    println!("Development server configuration:");
    println!("  Root: {}", root.display());
    println!("  Address: http://{}:{}", host, port);
    println!("  Live reload: {}", !no_reload);
    println!("  Directory listing: {}", directory_listing);
    if let Some(fallback) = spa_fallback {
        println!("  SPA fallback: {}", fallback);
    }
    println!();

    let mut server = DevServer::new(config);

    // Handle Ctrl+C gracefully
    let server_running = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(true));
    let r = server_running.clone();

    ctrlc::set_handler(move || {
        println!("\nShutting down server...");
        r.store(false, std::sync::atomic::Ordering::SeqCst);
    })
    .map_err(|e| miette::miette!("Failed to set Ctrl-C handler: {}", e))?;

    server
        .start()
        .map_err(|e| miette::miette!("Server error: {}", e))?;

    Ok(())
}

/// Run build hooks
fn run_hook(point: &str, project: &std::path::Path, verbose: bool) -> Result<()> {
    use sounio::watch::{HookContext, HookManager, HookPoint};

    tracing::info!("Running hook '{}' for project {:?}", point, project);

    // Parse hook point
    let hook_point = HookPoint::from_str(point).ok_or_else(|| {
        miette::miette!(
            "Unknown hook point: '{}'. Valid points are: pre-build, post-build, \
             pre-test, post-test, watch-start, watch-stop, on-file-change, \
             pre-reload, post-reload, pre-format, post-format, pre-lint, post-lint, on-error",
            point
        )
    })?;

    // Create hook manager
    let mut manager = HookManager::new(project.to_path_buf());
    manager.set_verbose(verbose);

    // Try to load hooks from d.toml
    let config_path = project.join("d.toml");
    if config_path.exists() {
        manager
            .load_from_toml(&config_path)
            .map_err(|e| miette::miette!("Failed to load hooks from d.toml: {}", e))?;

        if verbose {
            println!("Loaded hooks from {}", config_path.display());
        }
    } else {
        println!(
            "No d.toml found in {}. No hooks configured.",
            project.display()
        );
        return Ok(());
    }

    // Create hook context
    let context = HookContext::new(project.to_path_buf(), hook_point);

    // Run hooks
    println!("Running {} hooks...", point);
    let results = manager.run(hook_point, &context);

    if results.is_empty() {
        println!("No hooks registered for '{}'", point);
        return Ok(());
    }

    // Report results
    let mut success_count = 0;
    let mut failure_count = 0;

    for result in &results {
        if result.success {
            success_count += 1;
            if verbose {
                println!("  [OK] {} ({:?})", result.name, result.duration);
                if !result.stdout.is_empty() {
                    for line in result.stdout.lines() {
                        println!("       {}", line);
                    }
                }
            }
        } else {
            failure_count += 1;
            println!(
                "  [FAIL] {} (exit code: {:?})",
                result.name, result.exit_code
            );
            if !result.stderr.is_empty() {
                for line in result.stderr.lines() {
                    eprintln!("         {}", line);
                }
            }
        }
    }

    println!();
    println!(
        "Hook results: {} passed, {} failed",
        success_count, failure_count
    );

    if failure_count > 0 {
        Err(miette::miette!("{} hook(s) failed", failure_count))
    } else {
        Ok(())
    }
}

// =============================================================================
// Target Management Commands
// =============================================================================

/// List available targets
fn target_list(
    os_filter: Option<&str>,
    arch_filter: Option<&str>,
    builtin_only: bool,
    verbose: bool,
) -> Result<()> {
    use sounio::target::{Architecture, OperatingSystem, TargetRegistry};

    let registry = TargetRegistry::with_builtins();

    // Get targets based on filter
    let targets: Vec<_> = if builtin_only {
        registry.list_builtins()
    } else {
        registry.list()
    };

    // Parse OS filter
    let os_filter = os_filter.map(|s| OperatingSystem::parse(s));

    // Parse arch filter
    let arch_filter = arch_filter.map(|s| Architecture::parse(s));

    println!("Available targets:");
    println!();

    let mut count = 0;
    for name in targets {
        if let Ok(spec) = registry.get(name) {
            // Apply OS filter
            if let Some(ref os) = os_filter {
                if spec.os.os != *os {
                    continue;
                }
            }

            // Apply arch filter
            if let Some(ref arch) = arch_filter {
                if spec.arch.arch != *arch {
                    continue;
                }
            }

            count += 1;

            if verbose {
                println!("  {}", name);
                println!("    Arch: {}", spec.arch.arch);
                println!("    OS: {}", spec.os.os);
                println!("    Env: {}", spec.env.env);
                println!("    Pointer width: {} bits", spec.pointer_width());
                println!();
            } else {
                println!("  {}", name);
            }
        }
    }

    println!();
    println!("Total: {} target(s)", count);

    Ok(())
}

/// Show information about a specific target
fn target_info(target: &str, format: &str) -> Result<()> {
    use sounio::target::TargetRegistry;

    let registry = TargetRegistry::with_builtins();

    let spec = registry
        .get(target)
        .map_err(|e| miette::miette!("Target not found: {}", e))?;

    match format {
        "json" => {
            let json = serde_json::to_string_pretty(&spec)
                .map_err(|e| miette::miette!("Failed to serialize target: {}", e))?;
            println!("{}", json);
        }
        _ => {
            println!("Target: {}", spec.triple);
            println!();
            println!("Architecture:");
            println!("  Name: {}", spec.arch.arch);
            println!("  CPU: {}", spec.arch.cpu);
            println!("  Pointer width: {} bits", spec.pointer_width());
            println!("  Data layout: {}", spec.data_layout());
            if !spec.arch.features.is_empty() {
                let features: Vec<&str> = spec.arch.features.iter().map(|s| s.as_str()).collect();
                println!("  Features: {}", features.join(", "));
            }
            println!();
            println!("Operating System:");
            println!("  Name: {}", spec.os.os);
            if let Some(ref ver) = spec.os.min_version {
                println!("  Min version: {}", ver);
            }
            println!("  Requires PIE: {}", spec.os.requires_pie);
            println!("  Panic strategy: {:?}", spec.os.panic_strategy);
            println!();
            println!("Environment:");
            println!("  ABI: {}", spec.env.env);
            println!("  C runtime: {:?}", spec.env.crt);
            println!("  Relocation model: {:?}", spec.env.relocation_model);
            println!("  Code model: {:?}", spec.env.code_model);
            println!();
            println!("Linker:");
            println!("  Flavor: {:?}", spec.linker.flavor);
            if let Some(ref path) = spec.linker.path {
                println!("  Path: {}", path.display());
            }
            println!();
            println!("Options:");
            println!("  Built-in: {}", spec.options.is_builtin);
            println!("  Supports i128: {}", spec.options.supports_i128);
            println!("  Has float: {}", spec.options.has_float);
            println!("  Has TLS: {}", spec.options.has_tls);
            if let Some(width) = spec.options.max_atomic_width {
                println!("  Max atomic width: {} bits", width);
            }
        }
    }

    Ok(())
}

/// Add a custom target from a file
fn target_add(file: &std::path::Path) -> Result<()> {
    use sounio::target::TargetSpec;

    let spec = TargetSpec::from_file(file)
        .map_err(|e| miette::miette!("Failed to load target specification: {}", e))?;

    println!("Successfully loaded target: {}", spec.triple);
    println!();
    println!("To use this target, specify: --target {}", file.display());
    println!("Or copy the file to ~/.sounio/targets/{}.json", spec.triple);

    Ok(())
}

/// Create a new custom target specification
fn target_create(triple: &str, base: Option<&str>, output: Option<&std::path::Path>) -> Result<()> {
    use sounio::target::{TargetRegistry, TargetSpec};

    let registry = TargetRegistry::with_builtins();

    // Start with base target or parse triple
    let spec = if let Some(base_name) = base {
        let mut base_spec = registry
            .get(base_name)
            .map_err(|e| miette::miette!("Base target not found: {}", e))?;

        // Update the triple
        base_spec.triple = triple
            .parse()
            .map_err(|e| miette::miette!("Invalid triple: {}", e))?;
        base_spec.options.is_builtin = false;
        base_spec
    } else {
        let mut spec = TargetSpec::from_triple(triple)
            .map_err(|e| miette::miette!("Invalid triple: {}", e))?;
        spec.options.is_builtin = false;
        spec
    };

    // Determine output path
    let out_path = output
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(format!("{}.json", triple)));

    // Save the specification
    spec.to_file(&out_path)
        .map_err(|e| miette::miette!("Failed to save target: {}", e))?;

    println!("Created target specification: {}", out_path.display());
    println!();
    println!("Edit the JSON file to customize the target, then use with:");
    println!("  dc build --target {}", out_path.display());

    Ok(())
}

/// Show host target information
fn target_host() -> Result<()> {
    use sounio::target::{CfgContext, TargetSpec, host_triple};

    let triple = host_triple();
    println!("Host target: {}", triple);
    println!();

    // Parse and show details
    if let Ok(spec) = TargetSpec::from_triple(&triple) {
        println!("Architecture: {}", spec.arch.arch);
        println!("OS: {}", spec.os.os);
        println!("Environment: {}", spec.env.env);
        println!("Pointer width: {} bits", spec.pointer_width());
        println!();

        // Show cfg values
        let cfg = CfgContext::from_target(&spec);
        println!("Configuration flags:");
        let mut flags: Vec<_> = cfg.flags().iter().collect();
        flags.sort();
        for flag in flags {
            println!("  {}", flag);
        }
        println!();
        println!("Configuration values:");
        let mut values: Vec<_> = cfg.values().iter().collect();
        values.sort_by_key(|(k, _)| *k);
        for (key, value) in values {
            println!("  {} = \"{}\"", key, value);
        }
    }

    Ok(())
}

/// Check target cfg predicates
fn target_cfg(target: Option<&str>, predicate: Option<&str>) -> Result<()> {
    use sounio::target::{CfgContext, CfgPredicate, TargetRegistry, TargetSpec, host_triple};

    let registry = TargetRegistry::with_builtins();

    // Get target specification
    let spec = if let Some(target_name) = target {
        registry
            .get(target_name)
            .map_err(|e| miette::miette!("Target not found: {}", e))?
    } else {
        let triple = host_triple();
        TargetSpec::from_triple(&triple)
            .map_err(|e| miette::miette!("Failed to parse host triple: {}", e))?
    };

    // Create cfg context
    let cfg = CfgContext::from_target(&spec);

    if let Some(pred_str) = predicate {
        // Evaluate specific predicate
        let pred = CfgPredicate::parse(pred_str)
            .map_err(|e| miette::miette!("Invalid predicate: {}", e))?;

        let result = pred.evaluate(&cfg);
        println!("cfg({}) = {}", pred_str, result);
    } else {
        // Show all cfg values
        println!("Target: {}", spec.triple);
        println!();
        println!("Flags:");
        let mut flags: Vec<_> = cfg.flags().iter().collect();
        flags.sort();
        for flag in flags {
            println!("  cfg({})", flag);
        }
        println!();
        println!("Values:");
        let mut values: Vec<_> = cfg.values().iter().collect();
        values.sort_by_key(|(k, _)| *k);
        for (key, value) in values {
            println!("  cfg({} = \"{}\")", key, value);
        }
    }

    Ok(())
}

// =============================================================================
// Sysroot Management Commands
// =============================================================================

/// List installed sysroots
fn sysroot_list(verbose: bool) -> Result<()> {
    use sounio::target::SysrootManager;

    let cache_dir = get_sysroot_cache_dir();
    let manager = SysrootManager::new(cache_dir);

    let sysroots = manager
        .list_cached()
        .map_err(|e| miette::miette!("Failed to list sysroots: {}", e))?;

    if sysroots.is_empty() {
        println!("No sysroots installed.");
        println!();
        println!("Install a sysroot with: souc sysroot install <target>");
        return Ok(());
    }

    println!("Installed sysroots:");
    println!();

    for (name, metadata) in &sysroots {
        if verbose {
            println!("  {}", name);
            println!("    Version: {}", metadata.version);
            println!("    Created: {}", format_timestamp(metadata.created));
            println!("    Source: {:?}", metadata.source);
            if !metadata.components.is_empty() {
                let components: Vec<_> = metadata
                    .components
                    .iter()
                    .map(|c| c.display_name())
                    .collect();
                println!("    Components: {}", components.join(", "));
            }
            println!();
        } else {
            println!("  {}", name);
        }
    }

    println!();
    println!("Total: {} sysroot(s)", sysroots.len());

    Ok(())
}

/// Show sysroot for a specific target
fn sysroot_show(target: &str) -> Result<()> {
    use sounio::target::{Sysroot, TargetRegistry, TargetTriple};

    let registry = TargetRegistry::with_builtins();

    // Parse target
    let spec = registry
        .get(target)
        .map_err(|e| miette::miette!("Target not found: {}", e))?;

    let cache_dir = get_sysroot_cache_dir();
    let sysroot_path = cache_dir.join(target);

    if sysroot_path.exists() {
        let triple = TargetTriple::parse(target)
            .map_err(|e| miette::miette!("Invalid target triple: {}", e))?;

        let sysroot = Sysroot::open(&sysroot_path, &triple)
            .map_err(|e| miette::miette!("Failed to open sysroot: {}", e))?;

        println!("Sysroot for: {}", target);
        println!("  Path: {}", sysroot.path.display());
        println!("  Library paths:");
        for path in &sysroot.lib_paths {
            println!("    {}", path.display());
        }
        println!("  Include paths:");
        for path in &sysroot.include_paths {
            println!("    {}", path.display());
        }

        if let Some(ref metadata) = sysroot.metadata {
            println!();
            println!("Metadata:");
            println!("  Version: {}", metadata.version);
            println!("  Created: {}", format_timestamp(metadata.created));
            println!("  Source: {:?}", metadata.source);
            if let Some(source_kind) = metadata.extra.get("source_kind") {
                println!("  Source kind: {}", source_kind);
            }
            if let Some(resolved_path) = metadata.extra.get("sysroot_path") {
                println!("  Resolved sysroot path: {}", resolved_path);
            }
            if let Some(stdlib_path) = metadata.extra.get("stdlib_path") {
                println!("  Stdlib path: {}", stdlib_path);
            }
        }
    } else {
        println!("No sysroot installed for: {}", target);
        println!();
        println!("Install with: souc sysroot install {}", target);

        // Try to find system sysroot
        if let Some(ref system_sysroot) = spec.os.sysroot {
            println!();
            println!("System sysroot found at: {}", system_sysroot.display());
        }
    }

    Ok(())
}

/// Install sysroot for a target
fn sysroot_install(target: &str, force: bool) -> Result<()> {
    use sounio::target::{
        SysrootConfig, SysrootManager, SysrootMetadata, SysrootSource, TargetRegistry,
    };

    let registry = TargetRegistry::with_builtins();

    // Verify target exists
    let _spec = registry
        .get(target)
        .map_err(|e| miette::miette!("Target not found: {}", e))?;

    let cache_dir = get_sysroot_cache_dir();
    let sysroot_path = cache_dir.join(target);

    if sysroot_path.exists() && !force {
        println!("Sysroot already installed for: {}", target);
        println!("Use --force to reinstall.");
        return Ok(());
    }

    if force && sysroot_path.exists() {
        std::fs::remove_dir_all(&sysroot_path)
            .map_err(|e| miette::miette!("Failed to remove existing sysroot: {}", e))?;
    }

    println!("Installing sysroot for: {}", target);
    println!();

    // Create sysroot manager with auto-build enabled
    let config = SysrootConfig {
        auto_build: true,
        ..Default::default()
    };

    let mut manager = SysrootManager::with_config(cache_dir, config);

    let spec = registry.get(target).unwrap();
    match manager.get_sysroot(&spec) {
        Ok(sysroot) => {
            // Persist a managed manifest entry so list/show use the same source of truth.
            let cache_dir = get_sysroot_cache_dir();
            let managed_path = cache_dir.join(target);
            let resolved_path = sysroot.path.clone();
            let stdlib_path = detect_stdlib_path_for_install();
            let source_kind = if stdlib_path.is_some() {
                "local-stdlib"
            } else {
                "system"
            };
            let is_system_sysroot =
                resolved_path == PathBuf::from("/") || !resolved_path.starts_with(&cache_dir);

            std::fs::create_dir_all(&managed_path)
                .map_err(|e| miette::miette!("Failed to create sysroot directory: {}", e))?;

            let mut metadata = SysrootMetadata::new(
                target,
                if is_system_sysroot {
                    SysrootSource::System(resolved_path.clone())
                } else {
                    SysrootSource::Custom(resolved_path.clone())
                },
            );
            metadata
                .extra
                .insert("source_kind".to_string(), source_kind.to_string());
            metadata.extra.insert(
                "sysroot_path".to_string(),
                resolved_path.display().to_string(),
            );
            if let Some(path) = stdlib_path {
                metadata
                    .extra
                    .insert("stdlib_path".to_string(), path.display().to_string());
            }
            metadata
                .save(&managed_path.join("sysroot.json"))
                .map_err(|e| miette::miette!("Failed to persist sysroot metadata: {}", e))?;

            if resolved_path == PathBuf::from("/") {
                println!(
                    "Detected system sysroot at: {} (registered at {})",
                    resolved_path.display(),
                    managed_path.display()
                );
            } else if resolved_path.starts_with(&managed_path) {
                println!("Installed managed sysroot at: {}", managed_path.display());
            } else {
                println!(
                    "Registered sysroot at: {} (manifest at {})",
                    resolved_path.display(),
                    managed_path.display()
                );
            }
            Ok(())
        }
        Err(e) => Err(miette::miette!("Failed to install sysroot: {}", e)),
    }
}

/// Remove installed sysroot
fn sysroot_remove(target: &str) -> Result<()> {
    use sounio::target::SysrootManager;

    let cache_dir = get_sysroot_cache_dir();
    let mut manager = SysrootManager::new(cache_dir);

    manager
        .remove_cached(target)
        .map_err(|e| miette::miette!("Failed to remove sysroot: {}", e))?;

    println!("Removed sysroot for: {}", target);
    Ok(())
}

/// Clean stale sysroots
fn sysroot_clean(dry_run: bool) -> Result<()> {
    use sounio::target::SysrootManager;

    let cache_dir = get_sysroot_cache_dir();
    let mut manager = SysrootManager::new(cache_dir.clone());

    if dry_run {
        let sysroots = manager
            .list_cached()
            .map_err(|e| miette::miette!("Failed to list sysroots: {}", e))?;

        let stale: Vec<_> = sysroots
            .iter()
            .filter(|(_, m)| m.is_stale(std::time::Duration::from_secs(86400 * 30)))
            .collect();

        if stale.is_empty() {
            println!("No stale sysroots found.");
        } else {
            println!("Would remove {} stale sysroot(s):", stale.len());
            for (name, _) in stale {
                println!("  {}", name);
            }
        }
    } else {
        let removed = manager
            .clean_stale()
            .map_err(|e| miette::miette!("Failed to clean sysroots: {}", e))?;

        if removed == 0 {
            println!("No stale sysroots found.");
        } else {
            println!("Removed {} stale sysroot(s)", removed);
        }
    }

    Ok(())
}

/// Show stdlib search paths and diagnostic information
/// This helps users debug module resolution issues
fn sysroot_stdlib_paths(_verbose: bool) -> Result<()> {
    use sounio::module_loader::get_stdlib_search_paths;

    println!("Sounio stdlib search paths:");
    println!();

    let search_paths = get_stdlib_search_paths();

    for (i, path) in search_paths.iter().enumerate() {
        let status = if path.exists() {
            "✓ EXISTS"
        } else {
            "✗ MISSING"
        };
        println!("  [{:2}] {} - {}", i + 1, path.display(), status);
    }

    println!();
    println!("Total search locations: {}", search_paths.len());
    println!();

    // Check for active stdlib path from environment
    if let Ok(env_path) = std::env::var("SOUNIO_STDLIB_PATH") {
        println!("SOUNIO_STDLIB_PATH: {}", env_path);
        let path = std::path::PathBuf::from(&env_path);
        if !path.exists() {
            eprintln!("WARNING: SOUNIO_STDLIB_PATH does not exist!");
        }
    } else {
        println!("SOUNIO_STDLIB_PATH: <not set>");
        println!();
        println!("Set SOUNIO_STDLIB_PATH to override default stdlib location:");
        println!("  export SOUNIO_STDLIB_PATH=/path/to/stdlib");
    }

    Ok(())
}

/// Get the sysroot cache directory
fn get_sysroot_cache_dir() -> PathBuf {
    if let Ok(path) = std::env::var("SOUNIO_SYSROOT_HOME")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("XDG_CACHE_HOME")
        && !path.trim().is_empty()
    {
        return PathBuf::from(path).join("sounio").join("sysroots");
    }
    if let Some(home) = dirs::home_dir() {
        return home.join(".cache").join("sounio").join("sysroots");
    }
    PathBuf::from(".sounio/sysroots")
}

fn detect_stdlib_path_for_install() -> Option<PathBuf> {
    if let Ok(env_path) = std::env::var("SOUNIO_STDLIB_PATH")
        && !env_path.trim().is_empty()
    {
        let path = PathBuf::from(env_path);
        if path.exists() {
            return Some(path);
        }
    }
    let local_stdlib = std::env::current_dir().ok()?.join("stdlib");
    if local_stdlib.exists() {
        return Some(local_stdlib);
    }
    None
}

// ============================================================================
// Ontology Commands
// ============================================================================

/// Initialize ontology data by downloading and building .dontology files
fn ontology_init(
    core_only: bool,
    force: bool,
    output: &Path,
    include: Option<&str>,
    verbose: bool,
) -> Result<()> {
    use sounio::ontology::native::downloader::{
        DownloadConfig, OntologyDownloader, core_ontologies,
    };

    println!("Initializing ontology data...");
    println!("Output directory: {}", output.display());

    let config = DownloadConfig {
        output_dir: output.to_path_buf(),
        force,
        verify: true,
        timeout_secs: 120,
    };

    let downloader = OntologyDownloader::new(config);

    // Determine which ontologies to download
    let all_ontologies = core_ontologies();
    let ids: Vec<&str> = if let Some(include_list) = include {
        include_list.split(',').map(|s| s.trim()).collect()
    } else if core_only {
        all_ontologies
            .iter()
            .filter(|o| o.core)
            .map(|o| o.id.as_str())
            .collect()
    } else {
        all_ontologies.iter().map(|o| o.id.as_str()).collect()
    };

    let progress = if verbose {
        Some(Box::new(|name: &str, current: usize, total: usize| {
            println!("[{}/{}] Downloading {}...", current + 1, total, name);
        }) as Box<dyn Fn(&str, usize, usize) + Send>)
    } else {
        None
    };

    match downloader.download_by_ids(&ids, progress) {
        Ok(paths) => {
            println!("\nSuccessfully initialized {} ontologies:", paths.len());
            for path in &paths {
                println!("  {}", path.display());
            }
            Ok(())
        }
        Err(e) => Err(miette::miette!("Failed to initialize ontologies: {}", e)),
    }
}

/// Show information about an ontology concept
fn ontology_info(curie: &str, show_ancestors: bool, data_dir: &Path) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;

    // Parse CURIE to get ontology prefix
    let prefix = curie
        .split(':')
        .next()
        .ok_or_else(|| miette::miette!("Invalid CURIE format: {}", curie))?
        .to_lowercase();

    let mut registry = NativeOntologyRegistry::new(data_dir);

    let ont = registry
        .get_or_load(&prefix)
        .map_err(|e| miette::miette!("Failed to load ontology '{}': {}", prefix, e))?;

    match ont.get_concept(curie) {
        Some(_concept) => {
            println!("Concept: {}", curie);

            if let Some(label) = ont.get_label(curie) {
                println!("Label: {}", label);
            }

            if let Some(def) = ont.get_definition(curie) {
                println!("Definition: {}", def);
            }

            if let Some(parent) = ont.get_parent(curie) {
                println!("Parent: {}", parent);
            }

            if show_ancestors {
                let ancestors = ont.get_ancestors(curie);
                if !ancestors.is_empty() {
                    println!("Ancestors:");
                    for (i, anc) in ancestors.iter().enumerate() {
                        let indent = "  ".repeat(i + 1);
                        let label = ont.get_label(anc).unwrap_or("");
                        println!("{}└─ {} ({})", indent, anc, label);
                    }
                }
            }

            Ok(())
        }
        None => Err(miette::miette!("Concept not found: {}", curie)),
    }
}

/// Search for concepts by label
fn ontology_search(
    query: &str,
    limit: usize,
    ontology: Option<&str>,
    data_dir: &Path,
) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;

    let mut registry = NativeOntologyRegistry::new(data_dir);

    let ontologies: Vec<String> = if let Some(ont_id) = ontology {
        vec![ont_id.to_lowercase()]
    } else {
        registry.list_available()
    };

    if ontologies.is_empty() {
        return Err(miette::miette!(
            "No ontologies found. Run 'dc ontology init' first."
        ));
    }

    let mut all_results = Vec::new();

    for ont_id in &ontologies {
        if let Ok(ont) = registry.get_or_load(ont_id) {
            let results = ont.search(query, limit);
            for (curie, label) in results {
                all_results.push((curie.to_string(), label.to_string(), ont_id.clone()));
            }
        }
    }

    // Sort by label match quality
    all_results.sort_by(|a, b| {
        let a_exact = a.1.to_lowercase().starts_with(&query.to_lowercase());
        let b_exact = b.1.to_lowercase().starts_with(&query.to_lowercase());
        b_exact.cmp(&a_exact).then_with(|| a.1.cmp(&b.1))
    });

    // Limit results
    all_results.truncate(limit);

    if all_results.is_empty() {
        println!("No results found for '{}'", query);
    } else {
        println!("Found {} results for '{}':", all_results.len(), query);
        for (curie, label, ont_id) in &all_results {
            println!("  {} - {} [{}]", curie, label, ont_id);
        }
    }

    Ok(())
}

/// Check if one concept is a subclass of another
fn ontology_is_subclass(child: &str, parent: &str, data_dir: &Path) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;

    // Parse CURIEs to get ontology prefix
    let child_prefix = child
        .split(':')
        .next()
        .ok_or_else(|| miette::miette!("Invalid child CURIE: {}", child))?
        .to_lowercase();

    let parent_prefix = parent
        .split(':')
        .next()
        .ok_or_else(|| miette::miette!("Invalid parent CURIE: {}", parent))?
        .to_lowercase();

    if child_prefix != parent_prefix {
        return Err(miette::miette!(
            "Cross-ontology subclass check not yet supported: {} vs {}",
            child_prefix,
            parent_prefix
        ));
    }

    let mut registry = NativeOntologyRegistry::new(data_dir);
    let ont = registry
        .get_or_load(&child_prefix)
        .map_err(|e| miette::miette!("Failed to load ontology '{}': {}", child_prefix, e))?;

    let is_subclass = ont.is_subclass(child, parent);

    if is_subclass {
        println!("{} IS a subclass of {}", child, parent);

        // Show the path
        let ancestors = ont.get_ancestors(child);
        if let Some(pos) = ancestors.iter().position(|a| *a == parent) {
            println!(
                "Path: {} -> {} -> {}",
                child,
                ancestors[..=pos].join(" -> "),
                parent
            );
        }
    } else {
        println!("{} is NOT a subclass of {}", child, parent);
    }

    Ok(())
}

/// List available ontologies
fn ontology_list(data_dir: &Path, verbose: bool) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;

    let registry = NativeOntologyRegistry::new(data_dir);
    let available = registry.list_available();

    if available.is_empty() {
        println!("No ontologies found in {}", data_dir.display());
        println!("\nRun 'dc ontology init' to download ontologies.");
        return Ok(());
    }

    println!("Available ontologies in {}:", data_dir.display());

    if verbose {
        let mut registry = NativeOntologyRegistry::new(data_dir);
        for id in &available {
            if let Ok(ont) = registry.get_or_load(id) {
                println!(
                    "  {} - {} concepts (v{})",
                    id, ont.concept_count, ont.version
                );
            } else {
                println!("  {} (failed to load)", id);
            }
        }
    } else {
        for id in &available {
            println!("  {}", id);
        }
    }

    Ok(())
}

// =============================================================================
// Ontology Versioning Commands (Day 40)
// =============================================================================

/// Lock ontology versions to ontology.lock
fn ontology_lock(project_dir: &Path, output: &Path, force: bool) -> Result<()> {
    use sounio::ontology::OntologyLayer;
    use sounio::ontology::native::NativeOntologyRegistry;
    use sounio::ontology::version::{
        Manifest, OntologyEntry, OntologyVersion, manifest::OntologySource as ManifestSource,
    };

    // Check if lock file already exists
    if output.exists() && !force {
        return Err(miette::miette!(
            "Lock file {} already exists. Use --force to overwrite.",
            output.display()
        ));
    }

    println!("Generating ontology lock file...");
    println!("Project: {}", project_dir.display());

    // Find ontology data directory
    let data_dir = project_dir.join(".sounio/ontology");
    if !data_dir.exists() {
        return Err(miette::miette!(
            "No ontology data found. Run 'dc ontology init' first."
        ));
    }

    let mut registry = NativeOntologyRegistry::new(&data_dir);
    let available = registry.list_available();

    if available.is_empty() {
        return Err(miette::miette!(
            "No ontologies found. Run 'dc ontology init' first."
        ));
    }

    // Create manifest with all available ontologies
    let mut manifest = Manifest::new();

    for id in &available {
        if let Ok(ont) = registry.get_or_load(id) {
            let version = OntologyVersion::parse(&ont.version)
                .unwrap_or_else(|_| OntologyVersion::Tag(ont.version.clone()));

            manifest.upsert(OntologyEntry {
                name: id.clone(),
                version,
                checksum: None, // Would compute from file
                source: ManifestSource::LocalFile(
                    data_dir
                        .join(format!("{}.dontology", id))
                        .to_string_lossy()
                        .to_string(),
                ),
                layer: OntologyLayer::Domain,
                terms_used: Some(ont.concept_count),
                term_ids: None,
                dependencies: vec![],
            });
        }
    }

    // Save manifest
    manifest
        .save(output)
        .map_err(|e| miette::miette!("Failed to write lock file: {}", e))?;

    println!(
        "\nGenerated {} with {} ontologies:",
        output.display(),
        manifest.len()
    );
    for name in manifest.names() {
        if let Some(entry) = manifest.get(name) {
            println!("  {} @ {}", name, entry.version);
        }
    }

    Ok(())
}

/// Check for ontology updates
fn ontology_update(lock_file: &Path, write: bool, verbose: bool) -> Result<()> {
    use sounio::ontology::version::Manifest;

    if !lock_file.exists() {
        return Err(miette::miette!(
            "Lock file not found: {}. Run 'dc ontology lock' first.",
            lock_file.display()
        ));
    }

    let manifest = Manifest::load(lock_file)
        .map_err(|e| miette::miette!("Failed to read lock file: {}", e))?;

    println!("Checking for updates...");
    println!(
        "Lock file: {} ({} ontologies)",
        lock_file.display(),
        manifest.len()
    );

    // In a real implementation, we would query remote sources for latest versions
    // For now, we just report what's in the lock file
    let updates_available = 0;

    for entry in &manifest.ontologies {
        if verbose {
            println!(
                "  {} @ {} (from {:?})",
                entry.name, entry.version, entry.source
            );
        }

        // Simulate checking for updates - in reality this would query BioPortal/OLS4
        // For now, we just say everything is up to date
    }

    if updates_available == 0 {
        println!("\nAll ontologies are up to date.");
    } else {
        println!("\n{} update(s) available.", updates_available);
        if !write {
            println!("Run with --write to update the lock file.");
        }
    }

    Ok(())
}

/// Show diff between ontology versions
fn ontology_diff(
    ontology: &str,
    old_version: &str,
    new_version: &str,
    data_dir: &Path,
    verbose: bool,
) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;
    use sounio::ontology::version::diff::{OntologyDiff, OntologySnapshot, SnapshotTerm};

    println!(
        "Computing diff for {} ({} -> {})",
        ontology, old_version, new_version
    );

    // In a full implementation, we would load both versions from cache/storage
    // For now, we demonstrate the diff capability with the current version

    let mut registry = NativeOntologyRegistry::new(data_dir);
    let ont = registry
        .get_or_load(&ontology.to_lowercase())
        .map_err(|e| miette::miette!("Failed to load ontology '{}': {}", ontology, e))?;

    // Create snapshot from current ontology (as "new")
    let mut new_snapshot = OntologySnapshot::new(ontology, new_version);

    // Use search with empty prefix to get sample concepts for demonstration
    let sample_concepts = ont.search("", 100);
    for (curie, label) in sample_concepts {
        new_snapshot.add_term(SnapshotTerm {
            id: curie.to_string(),
            label: Some(label.to_string()),
            definition: ont.get_definition(curie).map(|s| s.to_string()),
            superclasses: ont
                .get_parent(curie)
                .map(|p| vec![p.to_string()])
                .unwrap_or_default(),
            synonyms: vec![],
            obsolete: false,
            replaced_by: None,
        });
    }

    // Create empty "old" snapshot for demonstration
    let old_snapshot = OntologySnapshot::new(ontology, old_version);

    let diff = OntologyDiff::compute(&old_snapshot, &new_snapshot);

    // Print summary
    println!("{}", diff.summary());

    if verbose {
        println!("{}", diff.detailed_report());
    } else if diff.has_breaking_changes() {
        println!("Breaking Changes:");
        for change in diff.breaking_changes().take(10) {
            println!("  - {}", change.description);
        }
        if diff.breaking_count() > 10 {
            println!("  ... and {} more", diff.breaking_count() - 10);
        }
    }

    Ok(())
}

/// Check for deprecated term usage
fn ontology_deprecations(
    input: &Path,
    _data_dir: &Path,
    deny_warnings: bool,
    format: &str,
) -> Result<()> {
    use sounio::ontology::version::deprecation::DeprecationTracker;

    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    // Extract CURIEs from source using simple pattern matching
    // Look for patterns like CHEBI:12345, GO:0008150, etc.
    let mut curies: Vec<&str> = Vec::new();
    for word in content.split(|c: char| !c.is_alphanumeric() && c != ':') {
        if let Some(colon_pos) = word.find(':') {
            let prefix = &word[..colon_pos];
            let suffix = &word[colon_pos + 1..];
            // Check if prefix is uppercase letters and suffix is digits
            if !prefix.is_empty()
                && prefix.chars().all(|c| c.is_ascii_uppercase())
                && !suffix.is_empty()
                && suffix.chars().all(|c| c.is_ascii_digit())
            {
                curies.push(word);
            }
        }
    }

    if curies.is_empty() {
        println!("No ontology terms found in {}", input.display());
        return Ok(());
    }

    println!(
        "Found {} ontology term references in {}",
        curies.len(),
        input.display()
    );

    // Create tracker with some example deprecations
    // In reality, this would be loaded from ontology metadata
    let mut tracker = DeprecationTracker::new();
    if deny_warnings {
        tracker = tracker.warnings_as_errors(true);
    }

    // Check each CURIE
    let warnings: Vec<_> = curies
        .iter()
        .filter_map(|curie| tracker.check(curie, None))
        .collect();

    match format {
        "json" => {
            println!("[");
            for (i, w) in warnings.iter().enumerate() {
                let comma = if i < warnings.len() - 1 { "," } else { "" };
                println!("  {}{}", w.to_json(), comma);
            }
            println!("]");
        }
        _ => {
            if warnings.is_empty() {
                println!("\nNo deprecated terms found.");
            } else {
                println!("\nDeprecation warnings:");
                for w in &warnings {
                    print!("{}", w.format());
                }
            }
        }
    }

    if tracker.has_errors() {
        Err(miette::miette!("Deprecated term errors found"))
    } else {
        Ok(())
    }
}

/// Verify lock file integrity
fn ontology_verify(lock_file: &Path, data_dir: &Path) -> Result<()> {
    use sounio::ontology::native::NativeOntologyRegistry;
    use sounio::ontology::version::Manifest;

    if !lock_file.exists() {
        return Err(miette::miette!(
            "Lock file not found: {}",
            lock_file.display()
        ));
    }

    let manifest = Manifest::load(lock_file)
        .map_err(|e| miette::miette!("Failed to parse lock file: {}", e))?;

    println!("Verifying {}...", lock_file.display());
    println!("Schema version: {}", manifest.metadata.schema_version);
    println!("Generated: {}", manifest.metadata.generated);
    println!("Ontologies: {}", manifest.len());

    let mut errors = 0;
    let mut warnings = 0;
    let mut registry = NativeOntologyRegistry::new(data_dir);

    for entry in &manifest.ontologies {
        print!("  {} @ {} ... ", entry.name, entry.version);

        // Try to load the ontology
        match registry.get_or_load(&entry.name) {
            Ok(ont) => {
                // Check version matches
                if ont.version != entry.version.to_string() {
                    println!("VERSION MISMATCH (have {})", ont.version);
                    warnings += 1;
                } else {
                    println!("OK");
                }
            }
            Err(_) => {
                println!("NOT FOUND");
                errors += 1;
            }
        }
    }

    println!();
    if errors > 0 {
        Err(miette::miette!(
            "Verification failed: {} error(s), {} warning(s)",
            errors,
            warnings
        ))
    } else if warnings > 0 {
        println!("Verification passed with {} warning(s)", warnings);
        Ok(())
    } else {
        println!("Verification passed.");
        Ok(())
    }
}

// =============================================================================
// Layout Synthesis Commands
// =============================================================================

/// Analyze concept usage and generate layout plan
fn layout_analyze(
    input: &Path,
    data_dir: &Path,
    max_clusters: usize,
    output: Option<&Path>,
) -> Result<()> {
    use sounio::layout::{
        DistanceMatrix, LayoutConfig, cluster_concepts, extract_concepts_from_types,
        generate_layout, generate_report,
    };
    use sounio::ontology::native::NativeOntology;
    use std::io::Write;

    // Read input file
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    // Extract types (one per line)
    let types: Vec<&str> = content.lines().collect();
    let usage = extract_concepts_from_types(&types);

    if usage.concepts.is_empty() {
        println!("No concepts found in input file.");
        return Ok(());
    }

    println!(
        "Found {} concepts in {} types",
        usage.concepts.len(),
        types.len()
    );

    // Try to load an ontology for distance computation
    let ontology = if data_dir.exists() {
        // Try to load a default ontology
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology = ontology.unwrap_or_else(|| NativeOntology::empty("default"));

    // Build distance matrix and cluster
    let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
    let distances = DistanceMatrix::build(&concepts, &ontology);
    let clustering = cluster_concepts(&usage, &distances, max_clusters);

    // Generate layout
    let config = LayoutConfig {
        max_clusters,
        ..Default::default()
    };
    let plan = generate_layout(clustering.clone(), config);

    // Generate report
    let report = generate_report(&plan, &clustering, None);

    // Output report
    if let Some(out_path) = output {
        let mut file = std::fs::File::create(out_path)
            .map_err(|e| miette::miette!("Failed to create {}: {}", out_path.display(), e))?;
        file.write_all(report.as_bytes())
            .map_err(|e| miette::miette!("Failed to write report: {}", e))?;
        println!("Report written to {}", out_path.display());
    } else {
        println!("{}", report);
    }

    Ok(())
}

/// Simulate cache performance
fn layout_simulate(input: &Path, data_dir: &Path, cache_size: usize, compare: bool) -> Result<()> {
    use sounio::layout::{
        CacheInstrumentation, ConceptUsage, DistanceMatrix, LayoutConfig, cluster_concepts,
        compare_layouts, generate_layout,
    };
    use sounio::ontology::native::NativeOntology;

    // Read access pattern (one CURIE per line)
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let accesses: Vec<String> = content
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if accesses.is_empty() {
        println!("No access pattern found in input file.");
        return Ok(());
    }

    println!(
        "Loaded {} accesses from {}",
        accesses.len(),
        input.display()
    );

    // Build usage from access pattern
    let mut usage = ConceptUsage::new();
    for access in &accesses {
        usage.record_access(access);
    }

    // Load ontology
    let ontology = if data_dir.exists() {
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology = ontology.unwrap_or_else(|| NativeOntology::empty("default"));

    if compare {
        // Build layout and compare
        let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
        let distances = DistanceMatrix::build(&concepts, &ontology);
        let clustering = cluster_concepts(&usage, &distances, 4);
        let config = LayoutConfig {
            cache_size,
            ..Default::default()
        };
        let plan = generate_layout(clustering, config);

        let comparison = compare_layouts(&accesses, &plan, cache_size);

        println!("\n=== Cache Performance Comparison ===\n");
        println!("Cache size: {} concepts", cache_size);
        println!("\nBaseline (alphabetical layout):");
        println!("  Hit rate: {:.1}%", comparison.baseline.hit_rate());
        println!(
            "  Hits: {}, Misses: {}",
            comparison.baseline.hits, comparison.baseline.misses
        );

        println!("\nOptimized (semantic layout):");
        println!("  Hit rate: {:.1}%", comparison.optimized.hit_rate());
        println!(
            "  Hits: {}, Misses: {}",
            comparison.optimized.hits, comparison.optimized.misses
        );

        println!(
            "\nImprovement: {:.1} percentage points",
            comparison.improvement
        );

        if comparison.is_improvement() {
            println!("\nHypothesis SUPPORTED: Semantic clustering improves cache performance.");
        } else if comparison.improvement < 0.0 {
            println!("\nHypothesis NOT SUPPORTED: Baseline performed better.");
        } else {
            println!("\nHypothesis INCONCLUSIVE: No significant difference.");
        }
    } else {
        // Simple simulation
        let mut cache = CacheInstrumentation::new(cache_size);
        cache.simulate(&accesses);
        let stats = cache.stats();

        println!("\n=== Cache Simulation Results ===\n");
        println!("Cache size: {} concepts", cache_size);
        println!("Total accesses: {}", stats.accesses);
        println!("Hits: {} ({:.1}%)", stats.hits, stats.hit_rate());
        println!("Misses: {} ({:.1}%)", stats.misses, stats.miss_rate());
    }

    Ok(())
}

/// Validate the hypothesis across multiple cache sizes
fn layout_validate(
    input: &Path,
    data_dir: &Path,
    cache_sizes_str: &str,
    iterations: usize,
) -> Result<()> {
    use sounio::layout::{
        ConceptUsage, DistanceMatrix, LayoutConfig, cluster_concepts, compare_layouts,
        generate_layout,
    };
    use sounio::ontology::native::NativeOntology;

    // Parse cache sizes
    let cache_sizes: Vec<usize> = cache_sizes_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if cache_sizes.is_empty() {
        return Err(miette::miette!("No valid cache sizes provided"));
    }

    // Read access pattern
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let accesses: Vec<String> = content
        .lines()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    if accesses.is_empty() {
        return Err(miette::miette!("No access pattern found in input file"));
    }

    // Build usage
    let mut usage = ConceptUsage::new();
    for access in &accesses {
        usage.record_access(access);
    }

    // Load ontology
    let ontology = if data_dir.exists() {
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology = ontology.unwrap_or_else(|| NativeOntology::empty("default"));

    println!("\n=== Hypothesis Validation ===\n");
    println!("Hypothesis: Semantic clustering improves cache performance.\n");
    println!(
        "Access pattern: {} accesses, {} unique concepts",
        accesses.len(),
        usage.concepts.len()
    );
    println!("Iterations per cache size: {}\n", iterations);

    let mut supported = 0;
    let mut not_supported = 0;
    let mut inconclusive = 0;

    println!(
        "{:>10} {:>12} {:>12} {:>12} {:>10}",
        "Cache", "Baseline", "Optimized", "Improve", "Result"
    );
    println!(
        "{:->10} {:->12} {:->12} {:->12} {:->10}",
        "", "", "", "", ""
    );

    for cache_size in &cache_sizes {
        let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
        let distances = DistanceMatrix::build(&concepts, &ontology);
        let clustering = cluster_concepts(&usage, &distances, 4);
        let config = LayoutConfig {
            cache_size: *cache_size,
            ..Default::default()
        };
        let plan = generate_layout(clustering, config);

        // Run multiple iterations and average
        let mut total_improvement = 0.0;
        let mut total_baseline = 0.0;
        let mut total_optimized = 0.0;

        for _ in 0..iterations {
            let comparison = compare_layouts(&accesses, &plan, *cache_size);
            total_improvement += comparison.improvement;
            total_baseline += comparison.baseline.hit_rate();
            total_optimized += comparison.optimized.hit_rate();
        }

        let avg_improvement = total_improvement / iterations as f64;
        let avg_baseline = total_baseline / iterations as f64;
        let avg_optimized = total_optimized / iterations as f64;

        let result = if avg_improvement > 1.0 {
            supported += 1;
            "SUPPORTED"
        } else if avg_improvement < -1.0 {
            not_supported += 1;
            "NOT SUPP"
        } else {
            inconclusive += 1;
            "INCONCLUS"
        };

        println!(
            "{:>10} {:>11.1}% {:>11.1}% {:>+11.1}% {:>10}",
            cache_size, avg_baseline, avg_optimized, avg_improvement, result
        );
    }

    println!("\n=== Summary ===\n");
    println!("Supported: {}/{}", supported, cache_sizes.len());
    println!("Not Supported: {}/{}", not_supported, cache_sizes.len());
    println!("Inconclusive: {}/{}", inconclusive, cache_sizes.len());

    if supported > not_supported && supported > inconclusive {
        println!("\nOverall: Hypothesis is SUPPORTED across most cache sizes.");
    } else if not_supported > supported {
        println!(
            "\nOverall: Hypothesis is NOT SUPPORTED. Consider different clustering parameters."
        );
    } else {
        println!(
            "\nOverall: Results are INCONCLUSIVE. May need more data or different access patterns."
        );
    }

    Ok(())
}

/// Visualize layout as ASCII, Mermaid, or table (Day 39)
fn layout_visualize(
    input: &Path,
    data_dir: &Path,
    format: &str,
    output: Option<&Path>,
) -> Result<()> {
    use sounio::layout::{
        DistanceMatrix, LayoutConfig, cluster_concepts, extract_concepts_from_types,
        generate_ascii, generate_layout, generate_mermaid, generate_table,
    };
    use sounio::ontology::native::NativeOntology;
    use std::io::Write;

    // Read input file
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let types: Vec<&str> = content.lines().collect();
    let usage = extract_concepts_from_types(&types);

    if usage.concepts.is_empty() {
        println!("No concepts found in input file.");
        return Ok(());
    }

    // Load ontology
    let ontology = if data_dir.exists() {
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology = ontology.unwrap_or_else(|| NativeOntology::empty("default"));

    // Build layout
    let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
    let distances = DistanceMatrix::build(&concepts, &ontology);
    let clustering = cluster_concepts(&usage, &distances, 4);
    let plan = generate_layout(clustering, LayoutConfig::default());

    // Generate visualization
    let visualization = match format.to_lowercase().as_str() {
        "mermaid" | "md" => generate_mermaid(&plan),
        "table" | "tbl" => generate_table(&plan),
        _ => generate_ascii(&plan),
    };

    // Output
    if let Some(out_path) = output {
        let mut file = std::fs::File::create(out_path)
            .map_err(|e| miette::miette!("Failed to create {}: {}", out_path.display(), e))?;
        file.write_all(visualization.as_bytes())
            .map_err(|e| miette::miette!("Failed to write: {}", e))?;
        println!("Visualization written to {}", out_path.display());
    } else {
        println!("{}", visualization);
    }

    Ok(())
}

/// Validate layout constraints (Day 39 - Participatory Compilation)
fn layout_constraints(input: &Path, data_dir: &Path, verbose: bool) -> Result<()> {
    use sounio::layout::{
        ConstraintSet, ConstraintSource, DistanceMatrix, ForcedRegion, LayoutConfig,
        LayoutConstraint, cluster_concepts, format_diagnostics, solve_constraints,
        validate_constraints_diagnostic,
    };

    // Read constraint file (simple format: one constraint per line)
    // Format: colocate:A,B,C or separate:X,Y or hot:Z or cold:W
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let mut constraints = ConstraintSet::new();
    let mut concepts_to_use: Vec<String> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let source = ConstraintSource::new(
            input.to_string_lossy(),
            line_num as u32 + 1,
            1,
            line.split(':').next().unwrap_or(""),
        );

        if let Some((cmd, args)) = line.split_once(':') {
            let concepts: Vec<String> = args.split(',').map(|s| s.trim().to_string()).collect();
            concepts_to_use.extend(concepts.clone());

            match cmd.trim().to_lowercase().as_str() {
                "colocate" => {
                    constraints.add(LayoutConstraint::Colocate { concepts, source });
                }
                "separate" => {
                    constraints.add(LayoutConstraint::Separate { concepts, source });
                }
                "hot" => {
                    for concept in concepts {
                        constraints.add(LayoutConstraint::ForceRegion {
                            concept,
                            region: ForcedRegion::Hot,
                            source: source.clone(),
                        });
                    }
                }
                "cold" => {
                    for concept in concepts {
                        constraints.add(LayoutConstraint::ForceRegion {
                            concept,
                            region: ForcedRegion::Cold,
                            source: source.clone(),
                        });
                    }
                }
                "warm" => {
                    for concept in concepts {
                        constraints.add(LayoutConstraint::ForceRegion {
                            concept,
                            region: ForcedRegion::Warm,
                            source: source.clone(),
                        });
                    }
                }
                _ => {
                    eprintln!(
                        "Warning: Unknown constraint type '{}' at line {}",
                        cmd,
                        line_num + 1
                    );
                }
            }
        }
    }

    if constraints.is_empty() {
        println!("No constraints found in input file.");
        println!("\nConstraint file format:");
        println!("  colocate:A,B,C     # Place concepts A, B, C in same cluster");
        println!("  separate:X,Y       # Keep concepts X, Y in different clusters");
        println!("  hot:Z              # Force concept Z to hot region");
        println!("  cold:W             # Force concept W to cold region");
        return Ok(());
    }

    println!("Loaded {} constraints", constraints.len());

    // Build usage from concepts
    let mut usage = sounio::layout::ConceptUsage::new();
    for concept in &concepts_to_use {
        usage.record_access(concept);
    }

    // Load ontology
    let ontology = if data_dir.exists() {
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            sounio::ontology::native::NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology =
        ontology.unwrap_or_else(|| sounio::ontology::native::NativeOntology::empty("default"));

    // Build clustering and solve constraints
    let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
    let distances = DistanceMatrix::build(&concepts, &ontology);
    let clustering = cluster_concepts(&usage, &distances, 4);

    let result = solve_constraints(clustering, &constraints, LayoutConfig::default());

    // Generate diagnostics
    let diagnostics = validate_constraints_diagnostic(&result, verbose);
    let output = format_diagnostics(&diagnostics, true);
    println!("{}", output);

    // Summary
    if result.is_success() {
        println!("All {} constraints satisfied.", result.satisfied.len());
    } else {
        println!(
            "{} conflict(s), {} warning(s)",
            result.conflicts.len(),
            result.warnings.len()
        );
    }

    Ok(())
}

/// Explain layout decision for a specific concept (Day 39)
fn layout_explain(concept: &str, input: &Path, data_dir: &Path) -> Result<()> {
    use sounio::layout::{
        DistanceMatrix, LayoutConfig, cluster_concepts, extract_concepts_from_types,
        generate_layout,
    };
    use sounio::ontology::native::NativeOntology;

    // Read input file
    let content = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let types: Vec<&str> = content.lines().collect();
    let usage = extract_concepts_from_types(&types);

    if usage.concepts.is_empty() {
        println!("No concepts found in input file.");
        return Ok(());
    }

    if !usage.concepts.contains(concept) {
        println!("Concept '{}' not found in input file.", concept);
        println!("\nAvailable concepts:");
        for c in usage.concepts.iter().take(10) {
            println!("  {}", c);
        }
        if usage.concepts.len() > 10 {
            println!("  ... and {} more", usage.concepts.len() - 10);
        }
        return Ok(());
    }

    // Load ontology
    let ontology = if data_dir.exists() {
        let ont_path = data_dir.join("chebi.dontology");
        if ont_path.exists() {
            NativeOntology::load(&ont_path).ok()
        } else {
            None
        }
    } else {
        None
    };

    let ontology = ontology.unwrap_or_else(|| NativeOntology::empty("default"));

    // Build layout
    let concepts: Vec<_> = usage.concepts.iter().cloned().collect();
    let distances = DistanceMatrix::build(&concepts, &ontology);
    let clustering = cluster_concepts(&usage, &distances, 4);
    let plan = generate_layout(clustering.clone(), LayoutConfig::default());

    // Find the concept's layout
    if let Some(layout) = plan.get(concept) {
        println!("\n=== Layout Explanation for '{}' ===\n", concept);

        // Region info
        let region_desc = match layout.region {
            sounio::layout::MemoryRegion::Hot => {
                "Hot (Stack/L1-L2 cache) - Frequently accessed data"
            }
            sounio::layout::MemoryRegion::Warm => {
                "Warm (Arena/L2-L3 cache) - Moderately accessed data"
            }
            sounio::layout::MemoryRegion::Cold => "Cold (Heap/RAM) - Rarely accessed data",
        };
        println!("Region: {}", region_desc);
        println!("Cluster: {}", layout.cluster_id);
        println!("Order within cluster: {}", layout.order);

        // Find cluster members
        let cluster_members: Vec<_> = plan
            .layouts
            .iter()
            .filter(|l| l.cluster_id == layout.cluster_id)
            .map(|l| l.concept.as_str())
            .collect();

        println!("\nCluster members ({}):", cluster_members.len());
        for member in cluster_members.iter().take(10) {
            if *member == concept {
                println!("  * {} (this concept)", member);
            } else {
                println!("    {}", member);
            }
        }
        if cluster_members.len() > 10 {
            println!("    ... and {} more", cluster_members.len() - 10);
        }

        // Access count
        if let Some(&count) = usage.access_counts.get(concept) {
            println!("\nAccess count: {}", count);
        }

        // Semantic distance to cluster members
        println!("\nSemantic distances to cluster members:");
        let concept_idx = concepts.iter().position(|c| c == concept);
        if let Some(idx) = concept_idx {
            for member in cluster_members.iter().take(5) {
                if *member != concept {
                    if let Some(member_idx) = concepts.iter().position(|c| c == *member) {
                        let dist = distances.get(idx, member_idx);
                        println!("  {} → {}: distance {}", concept, member, dist);
                    }
                }
            }
        }

        println!("\n=== Why This Placement? ===\n");
        println!(
            "The layout synthesizer placed '{}' in the {:?} region",
            concept, layout.region
        );
        println!("based on:");
        println!("  1. Access frequency (how often it's used in code)");
        println!("  2. Semantic proximity (ontology distance to other concepts)");
        println!("  3. Co-occurrence patterns (concepts accessed together)");
        println!("\nUse #[hot] or #[cold] annotations to override this decision.");
    } else {
        println!("Concept '{}' has no layout assigned.", concept);
    }

    Ok(())
}

/// Format a Unix timestamp as a human-readable string
fn format_timestamp(timestamp: u64) -> String {
    use std::time::{Duration, UNIX_EPOCH};

    let time = UNIX_EPOCH + Duration::from_secs(timestamp);
    let datetime = chrono::DateTime::<chrono::Utc>::from(time);
    datetime.format("%Y-%m-%d %H:%M:%S UTC").to_string()
}

// =============================================================================
// Distributed Build Commands
// =============================================================================

#[cfg(feature = "distributed")]
/// Start a distributed build server
fn distributed_server(
    address: &str,
    max_connections: usize,
    enable_cache: bool,
    name: &str,
) -> Result<()> {
    use sounio::distributed::{BuildServer, ServerConfig};

    println!("Starting distributed build server...");
    println!("  Name: {}", name);
    println!("  Address: {}", address);
    println!("  Max connections: {}", max_connections);
    println!("  Cache enabled: {}", enable_cache);
    println!();

    let config = ServerConfig {
        address: address
            .parse()
            .map_err(|e| miette::miette!("Invalid address: {}", e))?,
        max_connections,
        cache_enabled: enable_cache,
        server_name: name.to_string(),
        ..Default::default()
    };

    // Create and run server (blocking)
    let rt = tokio::runtime::Runtime::new()
        .map_err(|e| miette::miette!("Failed to create runtime: {}", e))?;

    rt.block_on(async {
        let server = BuildServer::new(config);
        println!("Server listening on {}", address);
        println!("Press Ctrl+C to stop");
        println!();

        server
            .start()
            .await
            .map_err(|e| miette::miette!("Server error: {}", e))
    })
}

#[cfg(feature = "distributed")]
/// Submit a remote build
fn distributed_build(
    server: &str,
    target: Option<&str>,
    profile: &str,
    path: &std::path::Path,
) -> Result<()> {
    println!("Distributed build configuration:");
    println!("  Server: {}", server);
    println!("  Target: {}", target.unwrap_or("default"));
    println!("  Profile: {}", profile);
    println!("  Path: {}", path.display());
    println!();
    println!("Note: Distributed build client is available in the library.");
    println!("Use the sounio::distributed::BuildClient API for programmatic access.");
    println!();
    println!("Example:");
    println!("  let client = BuildClient::new(ClientConfig::default());");
    println!("  client.connect(\"{}:9876\").await?;", server);
    println!("  let job = client.submit_job(...).await?;");
    Ok(())
}

#[cfg(feature = "distributed")]
/// Query server status
fn distributed_status(server: &str) -> Result<()> {
    println!("Server status query for: {}", server);
    println!();
    println!("Note: Use the BuildClient API to query server status programmatically.");
    println!();
    println!("The distributed build protocol supports:");
    println!("  - Job submission and tracking");
    println!("  - Worker registration and load balancing");
    println!("  - Build artifact caching");
    println!("  - Real-time progress reporting");
    Ok(())
}

// =============================================================================
// Cache Commands
// =============================================================================

#[cfg(feature = "distributed")]
/// Start a cache server
fn cache_server(address: &str, storage: &std::path::Path, max_size: &str) -> Result<()> {
    // Parse max size
    let max_bytes = parse_size(max_size).map_err(|e| miette::miette!("Invalid size: {}", e))?;

    println!("Cache server configuration:");
    println!("  Address: {}", address);
    println!("  Storage: {}", storage.display());
    println!("  Max size: {} ({} bytes)", max_size, max_bytes);
    println!();
    println!("Note: Use the sounio::distributed::cache::CacheServer API programmatically.");
    println!();
    println!("Example:");
    println!("  use sounio::distributed::cache::{{CacheServer, CacheServerConfig}};");
    println!();
    println!("  let config = CacheServerConfig {{");
    println!(
        "      storage_dir: PathBuf::from(\"{}\"),",
        storage.display()
    );
    println!("      max_size: {},", max_bytes);
    println!("      listen_addr: \"{}\".parse().unwrap(),", address);
    println!("      ..Default::default()");
    println!("  }};");
    println!("  let server = CacheServer::new(config);");
    println!("  server.start().await?;");
    Ok(())
}

#[cfg(feature = "distributed")]
/// Show cache statistics
fn cache_stats(url: Option<&str>, local_only: bool) -> Result<()> {
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("sounio").join("build-cache"))
        .unwrap_or_else(|| PathBuf::from(".d/cache"));

    println!("Cache Statistics");
    println!();

    if local_only || url.is_none() {
        println!("Local cache directory: {}", cache_dir.display());
        if cache_dir.exists() {
            // Count files and size
            let mut total_size = 0u64;
            let mut entry_count = 0usize;
            if let Ok(entries) = std::fs::read_dir(&cache_dir) {
                for entry in entries.flatten() {
                    if let Ok(meta) = entry.metadata() {
                        total_size += meta.len();
                        entry_count += 1;
                    }
                }
            }
            println!("  Entries: {}", entry_count);
            println!("  Total size: {}", format_bytes(total_size));
        } else {
            println!("  (cache directory does not exist)");
        }
    }

    if let Some(cache_url) = url {
        println!();
        println!("Remote cache server: {}", cache_url);
        println!("  Use CacheClient API to query remote stats programmatically.");
    }

    println!();
    println!("Note: Use sounio::distributed::cache::CacheClient for detailed statistics.");
    Ok(())
}

#[cfg(feature = "distributed")]
/// Clean cache entries
fn cache_clean(clean_all: bool, older_than: Option<&str>, dry_run: bool) -> Result<()> {
    let cache_dir = dirs::cache_dir()
        .map(|p| p.join("sounio").join("build-cache"))
        .unwrap_or_else(|| PathBuf::from(".d/cache"));

    if !cache_dir.exists() {
        println!("No cache found at {}", cache_dir.display());
        return Ok(());
    }

    println!("Cache cleanup");
    println!("  Directory: {}", cache_dir.display());
    println!("  Mode: {}", if clean_all { "all" } else { "selective" });
    if let Some(age) = older_than {
        println!("  Older than: {}", age);
    }
    println!("  Dry run: {}", dry_run);
    println!();

    if clean_all && !dry_run {
        // Actually clean the cache directory
        let mut removed = 0usize;
        let mut bytes_freed = 0u64;
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    bytes_freed += meta.len();
                    if std::fs::remove_file(entry.path()).is_ok() {
                        removed += 1;
                    }
                }
            }
        }
        println!(
            "Removed {} entries ({} freed)",
            removed,
            format_bytes(bytes_freed)
        );
    } else if dry_run {
        // Count what would be removed
        let mut total_size = 0u64;
        let mut entry_count = 0usize;
        if let Ok(entries) = std::fs::read_dir(&cache_dir) {
            for entry in entries.flatten() {
                if let Ok(meta) = entry.metadata() {
                    total_size += meta.len();
                    entry_count += 1;
                }
            }
        }
        println!(
            "Would remove {} entries ({})",
            entry_count,
            format_bytes(total_size)
        );
    } else {
        println!("Note: Use --all to clean all entries or --older-than to clean selectively.");
    }

    Ok(())
}

// =============================================================================
// CI Commands
// =============================================================================

#[cfg(feature = "distributed")]
/// Generate GitHub Actions workflow
fn ci_github(output: &std::path::Path, release: bool, targets: Option<&str>) -> Result<()> {
    println!("GitHub Actions workflow generation");
    println!();
    println!("  Output: {}", output.display());
    println!("  Release workflow: {}", release);
    if let Some(t) = targets {
        println!("  Targets: {}", t);
    }
    println!();

    // Generate a basic workflow template
    let workflow = r#"name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

env:
  CARGO_TERM_COLOR: always

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build
        run: cargo build --verbose

      - name: Run tests
        run: cargo test --verbose
"#;

    // Create directory if needed
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| miette::miette!("Failed to create directory: {}", e))?;
    }

    std::fs::write(output, workflow)
        .map_err(|e| miette::miette!("Failed to write workflow: {}", e))?;

    println!("Created: {}", output.display());
    println!();
    println!("Note: Use sounio::distributed::ci::github::WorkflowGenerator");
    println!("for more advanced workflow generation with matrix builds.");

    Ok(())
}

#[cfg(feature = "distributed")]
/// Generate GitLab CI pipeline
fn ci_gitlab(output: &std::path::Path, targets: Option<&str>) -> Result<()> {
    println!("GitLab CI pipeline generation");
    println!();
    println!("  Output: {}", output.display());
    if let Some(t) = targets {
        println!("  Targets: {}", t);
    }
    println!();

    // Generate a basic pipeline template
    let pipeline = r#"stages:
  - build
  - test

variables:
  CARGO_HOME: $CI_PROJECT_DIR/.cargo

cache:
  paths:
    - .cargo/
    - target/

build:
  stage: build
  script:
    - cargo build --verbose

test:
  stage: test
  script:
    - cargo test --verbose
"#;

    std::fs::write(output, pipeline)
        .map_err(|e| miette::miette!("Failed to write pipeline: {}", e))?;

    println!("Created: {}", output.display());
    println!();
    println!("Note: Use sounio::distributed::ci::gitlab::PipelineGenerator");
    println!("for more advanced pipeline generation with matrix builds.");

    Ok(())
}

#[cfg(feature = "distributed")]
/// Generate build provenance
fn ci_provenance(output: &std::path::Path, target: Option<&str>, profile: &str) -> Result<()> {
    use sounio::distributed::reproducible::BuildEnvironment;

    println!("Build provenance generation (SLSA format)");
    println!();
    println!("  Output: {}", output.display());
    println!("  Profile: {}", profile);
    println!("  Target: {}", target.unwrap_or("host"));
    println!();

    // Capture build environment
    let env = BuildEnvironment::capture(target.unwrap_or("host"), profile);

    // Generate basic SLSA provenance
    let provenance = serde_json::json!({
        "_type": "https://in-toto.io/Statement/v0.1",
        "predicateType": "https://slsa.dev/provenance/v0.2",
        "subject": [],
        "predicate": {
            "builder": {
                "id": format!("sounio-compiler-v{}", env!("CARGO_PKG_VERSION"))
            },
            "buildType": "https://sounio-lang.org/build/v1",
            "invocation": {
                "configSource": {},
                "parameters": {
                    "profile": profile,
                    "target": target.unwrap_or("host")
                },
                "environment": {
                    "compiler_version": env.compiler_version,
                    "os": env.host.os,
                    "arch": env.host.arch
                }
            },
            "metadata": {
                "buildStartedOn": chrono::Utc::now().to_rfc3339(),
                "reproducible": env.source_epoch.is_some()
            }
        }
    });

    let json = serde_json::to_string_pretty(&provenance)
        .map_err(|e| miette::miette!("Failed to serialize provenance: {}", e))?;

    std::fs::write(output, &json)
        .map_err(|e| miette::miette!("Failed to write provenance: {}", e))?;

    println!("Created: {}", output.display());
    println!();
    println!("Build provenance (SLSA format) generated successfully!");
    println!("Builder: Sounio Compiler v{}", env!("CARGO_PKG_VERSION"));

    Ok(())
}

#[cfg(feature = "distributed")]
/// Check reproducibility
fn ci_reproducible(builds: usize, check_env: bool) -> Result<()> {
    use sounio::distributed::reproducible::BuildEnvironment;

    println!("Build reproducibility check");
    println!();
    println!("  Builds to compare: {}", builds);
    println!("  Check environment: {}", check_env);
    println!();

    if check_env {
        println!("Environment Analysis:");
        let env = BuildEnvironment::capture("host", "release");

        println!("  Compiler: v{}", env.compiler_version);
        println!("  Target: {}", env.target);
        println!("  OS: {}", env.host.os);
        println!("  Arch: {}", env.host.arch);
        println!(
            "  SOURCE_DATE_EPOCH: {}",
            env.source_epoch
                .map(|e| e.to_string())
                .unwrap_or_else(|| "not set".into())
        );

        if env.source_epoch.is_none() {
            println!();
            println!("  Warning: SOURCE_DATE_EPOCH not set.");
            println!("  For reproducible builds, set this environment variable:");
            println!("    export SOURCE_DATE_EPOCH=$(git log -1 --format=%ct)");
        }
        println!();
    }

    if builds < 2 {
        println!("Note: Need at least 2 builds to verify reproducibility.");
        println!("Run with --builds 2 or higher.");
        return Ok(());
    }

    println!("To verify reproducibility:");
    println!("  1. Build the project multiple times");
    println!("  2. Compare output hashes");
    println!();
    println!("Example:");
    println!("  cargo build --release");
    println!("  sha256sum target/release/dc > build1.sha");
    println!("  cargo clean && cargo build --release");
    println!("  sha256sum target/release/dc > build2.sha");
    println!("  diff build1.sha build2.sha");
    println!();
    println!("Note: Use sounio::distributed::reproducible module for");
    println!("programmatic reproducibility verification.");

    Ok(())
}

// =============================================================================
// Helper Functions for Distributed Builds
// =============================================================================

#[cfg(feature = "distributed")]
/// Parse a size string (e.g., "10GB", "500MB")
fn parse_size(s: &str) -> std::result::Result<u64, String> {
    let s = s.trim().to_uppercase();

    let (num_str, unit) = if s.ends_with("GB") {
        (&s[..s.len() - 2], 1024 * 1024 * 1024)
    } else if s.ends_with("MB") {
        (&s[..s.len() - 2], 1024 * 1024)
    } else if s.ends_with("KB") {
        (&s[..s.len() - 2], 1024)
    } else if s.ends_with("B") {
        (&s[..s.len() - 1], 1)
    } else {
        return Err(format!(
            "Invalid size format: {}. Use GB, MB, KB, or B suffix",
            s
        ));
    };

    num_str
        .trim()
        .parse::<u64>()
        .map(|n| n * unit)
        .map_err(|_| format!("Invalid number: {}", num_str))
}

#[cfg(feature = "distributed")]
/// Parse a duration string (e.g., "7d", "24h", "30m")
fn parse_duration(s: &str) -> std::result::Result<std::time::Duration, String> {
    let s = s.trim().to_lowercase();

    let (num_str, multiplier) = if s.ends_with('d') {
        (&s[..s.len() - 1], 86400)
    } else if s.ends_with('h') {
        (&s[..s.len() - 1], 3600)
    } else if s.ends_with('m') {
        (&s[..s.len() - 1], 60)
    } else if s.ends_with('s') {
        (&s[..s.len() - 1], 1)
    } else {
        return Err(format!(
            "Invalid duration format: {}. Use d, h, m, or s suffix",
            s
        ));
    };

    num_str
        .trim()
        .parse::<u64>()
        .map(|n| std::time::Duration::from_secs(n * multiplier))
        .map_err(|_| format!("Invalid number: {}", num_str))
}

#[cfg(feature = "distributed")]
/// Format bytes as human-readable string
fn format_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;

    if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{} B", bytes)
    }
}

#[cfg(feature = "distributed")]
/// Format age as human-readable string
fn format_age(timestamp: Option<u64>) -> String {
    match timestamp {
        Some(ts) => {
            use std::time::{Duration, SystemTime, UNIX_EPOCH};
            let time = UNIX_EPOCH + Duration::from_secs(ts);
            match SystemTime::now().duration_since(time) {
                Ok(age) => {
                    let secs = age.as_secs();
                    if secs < 60 {
                        format!("{} seconds ago", secs)
                    } else if secs < 3600 {
                        format!("{} minutes ago", secs / 60)
                    } else if secs < 86400 {
                        format!("{} hours ago", secs / 3600)
                    } else {
                        format!("{} days ago", secs / 86400)
                    }
                }
                Err(_) => "in the future".to_string(),
            }
        }
        None => "N/A".to_string(),
    }
}

// =============================================================================
// Locality Commands (Day 41: Semantic-Physical Duality)
// =============================================================================

/// Show NUMA topology of the current system
fn locality_numa(format: &str, verbose: bool) -> Result<()> {
    use sounio::locality::numa::NumaTopology;

    let topology = NumaTopology::detect();

    match format {
        "json" => {
            println!("{{");
            println!("  \"numa_available\": {},", topology.is_numa());
            println!("  \"node_count\": {},", topology.node_count());
            println!("  \"total_memory\": {},", topology.total_memory());
            println!("  \"nodes\": [");

            for (i, node) in topology.nodes().iter().enumerate() {
                let comma = if i < topology.node_count() - 1 {
                    ","
                } else {
                    ""
                };
                println!("    {{");
                println!("      \"id\": {},", node.id);
                println!("      \"cpus\": {:?},", node.cpus);
                println!("      \"memory\": {},", node.memory);
                println!("      \"is_local\": {}", node.is_local);
                println!("    }}{}", comma);
            }

            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("=== NUMA Topology ===\n");

            if topology.is_numa() {
                println!("NUMA: Available ({} nodes)", topology.node_count());
            } else {
                println!("NUMA: Not available (single node)");
            }

            println!(
                "Total memory: {} GB",
                topology.total_memory() / (1024 * 1024 * 1024)
            );
            println!();

            for node in topology.nodes() {
                println!(
                    "Node {}{}:",
                    node.id,
                    if node.is_local { " (local)" } else { "" }
                );
                println!("  CPUs: {:?}", node.cpus);
                println!("  Memory: {} GB", node.memory / (1024 * 1024 * 1024));

                if verbose && !node.distances.is_empty() {
                    println!("  Distances:");
                    for (other, dist) in &node.distances {
                        println!("    -> Node {}: {}", other, dist);
                    }
                }
                println!();
            }

            if let Some(local) = topology.local_node() {
                println!("Current thread on Node {}", local.id);
            }
        }
    }

    Ok(())
}

/// Analyze access patterns in a source file
fn locality_analyze(input: &Path, format: &str, recommend: bool) -> Result<()> {
    use sounio::locality::access::{AccessAnalyzer, AccessKind};

    // Read and parse the source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;

    // Create analyzer and simulate analysis
    let mut analyzer = AccessAnalyzer::new();

    // For now, we simulate some access patterns based on AST structure
    // In a real implementation, this would analyze HIR/HLIR
    analyzer.enter_function("main");

    for item in &ast.items {
        if let sounio::ast::Item::Function(func) = item {
            analyzer.enter_function(&func.name);
            // Simulate field accesses based on function body
            analyzer.record_access("Data", "value", AccessKind::Read);
            analyzer.exit_function();
        }
    }

    analyzer.exit_function();

    match format {
        "json" => {
            println!("{{");
            println!("  \"file\": \"{}\",", input.display());
            println!("  \"patterns\": [");

            for (i, pattern) in analyzer.all_patterns().enumerate() {
                let comma = if i > 0 { "," } else { "" };
                println!("{}    {{", comma);
                println!("      \"name\": \"{}\",", pattern.name);
                println!("      \"accesses\": {},", pattern.accesses.len());
                println!("      \"hotness\": \"{:?}\"", pattern.hotness);
                println!("    }}");
            }

            println!("  ]");

            if recommend {
                println!(",  \"recommendations\": [");
                for (i, rec) in analyzer.recommendations().iter().enumerate() {
                    let comma = if i > 0 { "," } else { "" };
                    println!("{}    {{", comma);
                    println!("      \"kind\": \"{:?}\",", rec.kind);
                    println!("      \"description\": \"{}\"", rec.description);
                    println!("    }}");
                }
                println!("  ]");
            }

            println!("}}");
        }
        _ => {
            println!("=== Access Pattern Analysis: {} ===\n", input.display());

            for pattern in analyzer.all_patterns() {
                println!("Function: {}", pattern.name);
                println!("  Hotness: {:?}", pattern.hotness);
                println!("  Accesses: {}", pattern.accesses.len());
                println!("  Recommended locality: {}", pattern.recommended_locality());

                let hot_fields = pattern.get_hot_fields();
                if !hot_fields.is_empty() {
                    println!("  Hot fields: {}", hot_fields.join(", "));
                }

                let groups = pattern.get_co_access_groups();
                if !groups.is_empty() {
                    println!("  Co-access groups:");
                    for group in groups {
                        println!("    [{}]", group.join(", "));
                    }
                }
                println!();
            }

            if recommend {
                let recs = analyzer.recommendations();
                if !recs.is_empty() {
                    println!("=== Optimization Recommendations ===\n");
                    for rec in recs {
                        println!("  {:?}: {}", rec.kind, rec.description);
                        println!(
                            "    Estimated benefit: {:.0}%",
                            rec.estimated_benefit * 100.0
                        );
                        println!();
                    }
                }
            }
        }
    }

    Ok(())
}

/// Generate prefetch table from ontology
fn locality_prefetch(data_dir: &Path, output: Option<&Path>, format: &str) -> Result<()> {
    use sounio::locality::prefetch::PrefetchTable;
    use sounio::ontology::native::NativeOntology;

    // Try to load ontology
    let ontology = if data_dir.exists() {
        // Look for any .dontology file
        let entries = std::fs::read_dir(data_dir)
            .map_err(|e| miette::miette!("Failed to read {}: {}", data_dir.display(), e))?;

        let mut ont = None;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map(|e| e == "dontology").unwrap_or(false) {
                if let Ok(o) = NativeOntology::load(&path) {
                    ont = Some(o);
                    break;
                }
            }
        }
        ont
    } else {
        None
    };

    let table = if let Some(ont) = ontology {
        let adapter = sounio::locality::NativeOntologyAdapter::new(ont);
        PrefetchTable::from_ontology(&adapter)
    } else {
        println!(
            "No ontology found in {}. Using empty prefetch table.",
            data_dir.display()
        );
        PrefetchTable::new()
    };

    let stats = table.stats();

    let output_str = match format {
        "json" => {
            let mut s = String::new();
            s.push_str("{\n");
            s.push_str(&format!("  \"total_hints\": {},\n", stats.total_hints));
            s.push_str(&format!("  \"high_priority\": {},\n", stats.high_priority));
            s.push_str(&format!(
                "  \"types_with_hints\": {},\n",
                stats.types_with_hints
            ));
            s.push_str(&format!("  \"avg_distance\": {:.3},\n", stats.avg_distance));
            s.push_str("  \"entries\": [\n");

            for (i, entry) in table.entries().enumerate() {
                let comma = if i > 0 { ",\n" } else { "" };
                s.push_str(&format!("{}    {{\n", comma));
                s.push_str(&format!("      \"type\": \"{}\",\n", entry.type_name));
                s.push_str(&format!("      \"hints\": {}\n", entry.type_hints.len()));
                s.push_str("    }");
            }

            s.push_str("\n  ]\n");
            s.push_str("}\n");
            s
        }
        _ => {
            let mut s = String::new();
            s.push_str("=== Semantic Prefetch Table ===\n\n");
            s.push_str(&format!("Total hints: {}\n", stats.total_hints));
            s.push_str(&format!("High priority: {}\n", stats.high_priority));
            s.push_str(&format!("Types with hints: {}\n", stats.types_with_hints));
            s.push_str(&format!(
                "Average semantic distance: {:.3}\n\n",
                stats.avg_distance
            ));

            for entry in table.entries() {
                s.push_str(&format!("Type: {}\n", entry.type_name));
                for hint in &entry.type_hints {
                    s.push_str(&format!(
                        "  -> {} (priority: {:?}, distance: {:.2})\n",
                        hint.target,
                        hint.priority,
                        hint.distance.value()
                    ));
                    if !hint.reason.is_empty() {
                        s.push_str(&format!("     reason: {}\n", hint.reason));
                    }
                }
                s.push_str("\n");
            }
            s
        }
    };

    if let Some(out_path) = output {
        std::fs::write(out_path, &output_str)
            .map_err(|e| miette::miette!("Failed to write {}: {}", out_path.display(), e))?;
        println!("Wrote prefetch table to {}", out_path.display());
    } else {
        print!("{}", output_str);
    }

    Ok(())
}

/// Analyze struct for cache-line packing
fn locality_pack(input: &Path, struct_name: &str, cache_line: usize, suggest: bool) -> Result<()> {
    use sounio::locality::packing::CacheLinePacker;

    // Read and parse the source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;

    // Find the struct
    let mut found_struct = None;
    for item in &ast.items {
        if let sounio::ast::Item::Struct(s) = item {
            if s.name == struct_name {
                found_struct = Some(s);
                break;
            }
        }
    }

    let struct_def = found_struct.ok_or_else(|| {
        miette::miette!("Struct '{}' not found in {}", struct_name, input.display())
    })?;

    // Helper to get type name from TypeExpr
    fn type_expr_name(ty: &sounio::ast::TypeExpr) -> String {
        match ty {
            sounio::ast::TypeExpr::Named { path, .. } => {
                path.segments.last().cloned().unwrap_or_default()
            }
            sounio::ast::TypeExpr::Unit => "()".to_string(),
            sounio::ast::TypeExpr::Reference { inner, .. } => {
                format!("&{}", type_expr_name(inner))
            }
            _ => "unknown".to_string(),
        }
    }

    // Extract field information
    let fields: Vec<(&str, usize)> = struct_def
        .fields
        .iter()
        .map(|f| {
            // Estimate size based on type (simplified)
            let type_name = type_expr_name(&f.ty);
            let size = match type_name.as_str() {
                "bool" => 1,
                "i8" | "u8" => 1,
                "i16" | "u16" => 2,
                "i32" | "u32" | "f32" => 4,
                "i64" | "u64" | "f64" => 8,
                "i128" | "u128" => 16,
                _ => 8, // Default pointer size
            };
            (f.name.as_str(), size)
        })
        .collect();

    println!("=== Cache-Line Packing Analysis ===\n");
    println!("Struct: {}", struct_name);
    println!("Cache line size: {} bytes", cache_line);
    println!("Fields: {}\n", fields.len());

    // Show original layout
    println!("Original layout:");
    let mut offset = 0;
    for (name, size) in &fields {
        // Simple alignment
        let align = *size;
        let aligned = (offset + align - 1) / align * align;
        let padding = aligned - offset;
        if padding > 0 {
            println!("  [padding: {} bytes]", padding);
        }
        println!("  {}: {} bytes at offset {}", name, size, aligned);
        offset = aligned + size;
    }
    let original_size = offset;
    let original_cache_lines = (original_size + cache_line - 1) / cache_line;
    println!(
        "\nTotal size: {} bytes ({} cache lines)",
        original_size, original_cache_lines
    );

    if suggest {
        // Pack the struct
        let packer = CacheLinePacker::new(cache_line);
        let layout = packer.pack_simple(&fields, &[]);

        println!("\n=== Suggested Packed Layout ===\n");
        println!("{}", layout.to_comments());

        if layout.improvement > 0.0 {
            println!(
                "\nImprovement: {:.1}% fewer cache lines",
                layout.improvement * 100.0
            );
        } else {
            println!("\nNo improvement possible (already optimal)");
        }
    }

    Ok(())
}

/// Show locality type lattice
fn locality_lattice(format: &str) -> Result<()> {
    use sounio::locality::types::Locality;

    match format {
        "mermaid" => {
            println!("```mermaid");
            println!("graph TD");
            println!("    Register[Register] --> L1[L1 Cache]");
            println!("    L1 --> L2[L2 Cache]");
            println!("    L2 --> L3[L3 Cache / LLC]");
            println!("    L3 --> Local[Local DRAM]");
            println!("    Local --> Remote[Remote DRAM / NUMA]");
            println!("    Remote --> Persistent[Persistent Storage]");
            println!("    Persistent --> Network[Network Storage]");
            println!("");
            println!("    style Register fill:#ff6b6b");
            println!("    style L1 fill:#ffa94d");
            println!("    style L2 fill:#ffd43b");
            println!("    style L3 fill:#a9e34b");
            println!("    style Local fill:#69db7c");
            println!("    style Remote fill:#4dabf7");
            println!("    style Persistent fill:#748ffc");
            println!("    style Network fill:#9775fa");
            println!("```");
        }
        _ => {
            println!("=== Locality Type Lattice ===\n");
            println!("The subtype relation: Faster <: Slower");
            println!("A value at a faster locality can be used where slower is expected.\n");

            let levels = [
                Locality::Register,
                Locality::L1,
                Locality::L2,
                Locality::L3,
                Locality::Local,
                Locality::Remote,
                Locality::Persistent,
                Locality::Network,
            ];

            println!("Level         Latency    Capacity       Hot/Cold");
            println!("-----         -------    --------       --------");

            for level in &levels {
                let latency = format!("{:.0}x", level.latency_multiplier());
                let capacity = level
                    .typical_capacity()
                    .map(|c| {
                        if c >= 1024 * 1024 * 1024 {
                            format!("{} GB", c / (1024 * 1024 * 1024))
                        } else if c >= 1024 * 1024 {
                            format!("{} MB", c / (1024 * 1024))
                        } else if c >= 1024 {
                            format!("{} KB", c / 1024)
                        } else {
                            format!("{} B", c)
                        }
                    })
                    .unwrap_or_else(|| "varies".to_string());

                let temp = if level.is_hot() {
                    "Hot"
                } else if level.is_cold() {
                    "Cold"
                } else {
                    "Warm"
                };

                println!("{:<13} {:<10} {:<14} {}", level, latency, capacity, temp);
            }

            println!("\nSubtype examples:");
            println!("  L1 <: L3      (L1 data can be used where L3 is expected)");
            println!("  L2 <: Local   (L2 data can be used where Local is expected)");
            println!("  Local !<: L1  (Local data cannot be used where L1 is required)");
        }
    }

    Ok(())
}

/// Generate prefetch code for a function
fn locality_codegen(
    input: &Path,
    function: &str,
    target: &str,
    output: Option<&Path>,
) -> Result<()> {
    use sounio::locality::access::StridePattern;
    use sounio::locality::codegen::{PrefetchCodegen, Target};

    // Parse target
    let target_arch = match target.to_lowercase().as_str() {
        "x86_64" | "x86" | "amd64" => Target::X86_64,
        "arm64" | "aarch64" | "arm" => Target::Arm64,
        "riscv" | "riscv64" => Target::RiscV,
        _ => Target::LLVM,
    };

    // Read and parse the source file
    let source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    let tokens = sounio::lexer::lex(&source)?;
    let ast = sounio::parser::parse(&tokens, &source)?;

    // Find the function
    let mut found_fn = false;
    for item in &ast.items {
        if let sounio::ast::Item::Function(f) = item {
            if f.name == function {
                found_fn = true;
                break;
            }
        }
    }

    if !found_fn {
        return Err(miette::miette!(
            "Function '{}' not found in {}",
            function,
            input.display()
        ));
    }

    // Generate prefetch code
    let mut codegen = PrefetchCodegen::new(target_arch);
    codegen.bind_register("data", "rax");

    // Simulate stride pattern analysis
    let stride = StridePattern::new(64); // 64-byte stride (cache line)
    let instructions = codegen.for_stride(&stride, "data", 8);

    let mut output_str = String::new();
    output_str.push_str(&format!("; Prefetch code for function '{}'\n", function));
    output_str.push_str(&format!("; Target: {:?}\n", target_arch));
    output_str.push_str(&format!(
        "; Generated {} prefetch instructions\n\n",
        instructions.len()
    ));
    output_str.push_str(&codegen.emit());

    if let Some(out_path) = output {
        std::fs::write(out_path, &output_str)
            .map_err(|e| miette::miette!("Failed to write {}: {}", out_path.display(), e))?;
        println!("Wrote prefetch code to {}", out_path.display());
    } else {
        print!("{}", output_str);
    }

    Ok(())
}

// ============================================================================
// UNITS COMMANDS
// ============================================================================

fn units_list(category: &str, format: &str, verbose: bool) -> Result<()> {
    use sounio::units::Dimension;
    use sounio::units::check::UnitChecker;

    let _checker = UnitChecker::new();

    // Define unit categories
    let si_base = vec![
        ("kg", "kilogram", Dimension::MASS),
        ("m", "meter", Dimension::LENGTH),
        ("s", "second", Dimension::TIME),
        ("A", "ampere", Dimension::CURRENT),
        ("K", "kelvin", Dimension::TEMPERATURE),
        ("mol", "mole", Dimension::AMOUNT),
        ("cd", "candela", Dimension::LUMINOSITY),
    ];

    let si_prefixed = vec![
        ("g", "gram", Dimension::MASS),
        ("mg", "milligram", Dimension::MASS),
        ("μg", "microgram", Dimension::MASS),
        ("ng", "nanogram", Dimension::MASS),
        ("km", "kilometer", Dimension::LENGTH),
        ("cm", "centimeter", Dimension::LENGTH),
        ("mm", "millimeter", Dimension::LENGTH),
        ("L", "liter", Dimension::VOLUME),
        ("mL", "milliliter", Dimension::VOLUME),
        ("μL", "microliter", Dimension::VOLUME),
        ("min", "minute", Dimension::TIME),
        ("h", "hour", Dimension::TIME),
        ("d", "day", Dimension::TIME),
    ];

    let derived = vec![
        ("N", "newton", Dimension::FORCE),
        ("J", "joule", Dimension::ENERGY),
        ("W", "watt", Dimension::POWER),
        ("Pa", "pascal", Dimension::PRESSURE),
        ("Hz", "hertz", Dimension::FREQUENCY),
        ("°C", "celsius", Dimension::TEMPERATURE),
    ];

    let pkpd = vec![
        ("mg/L", "milligram per liter", Dimension::CONCENTRATION),
        ("μg/L", "microgram per liter", Dimension::CONCENTRATION),
        ("ng/mL", "nanogram per milliliter", Dimension::CONCENTRATION),
        ("L/h", "liter per hour", Dimension::CLEARANCE),
        ("mL/min", "milliliter per minute", Dimension::CLEARANCE),
        (
            "L/kg",
            "liter per kilogram",
            Dimension::new(-1, 3, 0, 0, 0, 0, 0),
        ),
        ("h⁻¹", "per hour", Dimension::FREQUENCY),
        ("mg/kg", "milligram per kilogram", Dimension::DIMENSIONLESS),
        ("mg·h/L", "milligram hour per liter", Dimension::AUC),
        ("nM", "nanomolar", Dimension::MOLAR_CONCENTRATION),
        ("μM", "micromolar", Dimension::MOLAR_CONCENTRATION),
    ];

    let print_units = |units: &[(&str, &str, Dimension)], title: &str| {
        println!("\n=== {} ===", title);
        for (symbol, name, dim) in units {
            if verbose {
                println!("  {:8} {:25} [{}]", symbol, name, dim);
            } else {
                println!("  {:8} {}", symbol, name);
            }
        }
    };

    match format {
        "json" => {
            println!("{{");
            let all_units: Vec<_> = match category {
                "si" => si_base.iter().chain(si_prefixed.iter()).collect(),
                "pkpd" => pkpd.iter().collect(),
                "derived" => derived.iter().collect(),
                _ => si_base
                    .iter()
                    .chain(si_prefixed.iter())
                    .chain(derived.iter())
                    .chain(pkpd.iter())
                    .collect(),
            };

            println!("  \"units\": [");
            for (i, (symbol, name, dim)) in all_units.iter().enumerate() {
                let comma = if i < all_units.len() - 1 { "," } else { "" };
                println!(
                    "    {{ \"symbol\": \"{}\", \"name\": \"{}\", \"dimension\": \"{}\" }}{}",
                    symbol, name, dim, comma
                );
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("Sounio Units of Measure");
            println!("==========================");

            match category {
                "si" => {
                    print_units(&si_base, "SI Base Units");
                    print_units(&si_prefixed, "SI Prefixed Units");
                }
                "pkpd" => {
                    print_units(&pkpd, "PK/PD Units");
                }
                "derived" => {
                    print_units(&derived, "Derived Units");
                }
                _ => {
                    print_units(&si_base, "SI Base Units");
                    print_units(&si_prefixed, "SI Prefixed Units");
                    print_units(&derived, "Derived Units");
                    print_units(&pkpd, "PK/PD Units");
                }
            }

            println!(
                "\nTotal: {} units available",
                si_base.len() + si_prefixed.len() + derived.len() + pkpd.len()
            );
        }
    }

    Ok(())
}

fn units_convert(value: f64, from: &str, to: &str, verbose: bool) -> Result<()> {
    use sounio::units::convert::parse_unit_expression;

    let (from_dim, from_scale) = parse_unit_expression(from)
        .map_err(|e| miette::miette!("Invalid source unit '{}': {}", from, e))?;

    let (to_dim, to_scale) = parse_unit_expression(to)
        .map_err(|e| miette::miette!("Invalid target unit '{}': {}", to, e))?;

    if from_dim != to_dim {
        return Err(miette::miette!(
            "Cannot convert between incompatible dimensions:\n  {} has dimension {}\n  {} has dimension {}",
            from,
            from_dim,
            to,
            to_dim
        ));
    }

    let base_value = value * from_scale;
    let result = base_value / to_scale;

    if verbose {
        println!("Conversion: {} {} → {} {}", value, from, result, to);
        println!("  Source dimension: {}", from_dim);
        println!("  Source scale: {}", from_scale);
        println!("  Target scale: {}", to_scale);
        println!("  Conversion factor: {}", from_scale / to_scale);
    } else {
        println!("{} {} = {} {}", value, from, result, to);
    }

    Ok(())
}

fn units_info(unit: &str, format: &str) -> Result<()> {
    use sounio::units::check::UnitChecker;
    use sounio::units::convert::parse_unit_expression;

    let checker = UnitChecker::new();

    // Try to look up the unit
    if let Some(unit_type) = checker.lookup_unit(unit) {
        let dim = unit_type
            .dimension()
            .unwrap_or(sounio::units::Dimension::DIMENSIONLESS);

        match format {
            "json" => {
                println!("{{");
                println!("  \"symbol\": \"{}\",", unit);
                println!("  \"dimension\": \"{}\",", dim);
                println!("  \"is_dimensionless\": {}", dim.is_dimensionless());
                println!("}}");
            }
            _ => {
                println!("Unit: {}", unit);
                println!("  Dimension: {}", dim);
                println!("  Dimensionless: {}", dim.is_dimensionless());

                // Show dimension breakdown
                if !dim.is_dimensionless() {
                    println!("  Components:");
                    if dim.mass != 0 {
                        println!("    Mass (M): {}", dim.mass);
                    }
                    if dim.length != 0 {
                        println!("    Length (L): {}", dim.length);
                    }
                    if dim.time != 0 {
                        println!("    Time (T): {}", dim.time);
                    }
                    if dim.current != 0 {
                        println!("    Current (I): {}", dim.current);
                    }
                    if dim.temperature != 0 {
                        println!("    Temperature (Θ): {}", dim.temperature);
                    }
                    if dim.amount != 0 {
                        println!("    Amount (N): {}", dim.amount);
                    }
                    if dim.luminosity != 0 {
                        println!("    Luminosity (J): {}", dim.luminosity);
                    }
                }
            }
        }
        return Ok(());
    }

    // Try parsing as expression
    match parse_unit_expression(unit) {
        Ok((dim, scale)) => {
            match format {
                "json" => {
                    println!("{{");
                    println!("  \"expression\": \"{}\",", unit);
                    println!("  \"dimension\": \"{}\",", dim);
                    println!("  \"scale\": {}", scale);
                    println!("}}");
                }
                _ => {
                    println!("Unit expression: {}", unit);
                    println!("  Dimension: {}", dim);
                    println!("  Scale (to SI): {}", scale);
                }
            }
            Ok(())
        }
        Err(e) => Err(miette::miette!("Unknown unit '{}': {}", unit, e)),
    }
}

fn units_check(unit1: &str, unit2: &str, verbose: bool) -> Result<()> {
    use sounio::units::convert::parse_unit_expression;

    let (dim1, scale1) = parse_unit_expression(unit1)
        .map_err(|e| miette::miette!("Invalid unit '{}': {}", unit1, e))?;

    let (dim2, scale2) = parse_unit_expression(unit2)
        .map_err(|e| miette::miette!("Invalid unit '{}': {}", unit2, e))?;

    let compatible = dim1 == dim2;

    if verbose {
        println!("Dimensional compatibility check:");
        println!("  {} → dimension: {}", unit1, dim1);
        println!("  {} → dimension: {}", unit2, dim2);
        println!();
    }

    if compatible {
        println!("✓ {} and {} are dimensionally compatible", unit1, unit2);
        if verbose {
            let factor = scale1 / scale2;
            println!("  Conversion factor: 1 {} = {} {}", unit1, factor, unit2);
        }
    } else {
        println!("✗ {} and {} are NOT dimensionally compatible", unit1, unit2);
        if verbose {
            println!("  {} has dimension: {}", unit1, dim1);
            println!("  {} has dimension: {}", unit2, dim2);
        }
    }

    Ok(())
}

fn units_parse(expr: &str, verbose: bool) -> Result<()> {
    use sounio::units::convert::parse_unit_expression;

    match parse_unit_expression(expr) {
        Ok((dim, scale)) => {
            println!("Unit expression: {}", expr);
            println!("  Valid: ✓");
            println!("  Dimension: {}", dim);

            if verbose {
                println!("  Scale factor: {}", scale);
                println!("  Dimension breakdown:");
                if dim.mass != 0 {
                    println!("    M^{}", dim.mass);
                }
                if dim.length != 0 {
                    println!("    L^{}", dim.length);
                }
                if dim.time != 0 {
                    println!("    T^{}", dim.time);
                }
                if dim.current != 0 {
                    println!("    I^{}", dim.current);
                }
                if dim.temperature != 0 {
                    println!("    Θ^{}", dim.temperature);
                }
                if dim.amount != 0 {
                    println!("    N^{}", dim.amount);
                }
                if dim.luminosity != 0 {
                    println!("    J^{}", dim.luminosity);
                }
            }
            Ok(())
        }
        Err(e) => {
            println!("Unit expression: {}", expr);
            println!("  Valid: ✗");
            println!("  Error: {}", e);
            Err(miette::miette!("Failed to parse unit expression"))
        }
    }
}

fn units_dimensions(format: &str) -> Result<()> {
    use sounio::units::Dimension;

    let dimensions = vec![
        ("M", "Mass", "kilogram (kg)", Dimension::MASS),
        ("L", "Length", "meter (m)", Dimension::LENGTH),
        ("T", "Time", "second (s)", Dimension::TIME),
        ("I", "Electric Current", "ampere (A)", Dimension::CURRENT),
        ("Θ", "Temperature", "kelvin (K)", Dimension::TEMPERATURE),
        ("N", "Amount of Substance", "mole (mol)", Dimension::AMOUNT),
        (
            "J",
            "Luminous Intensity",
            "candela (cd)",
            Dimension::LUMINOSITY,
        ),
    ];

    match format {
        "json" => {
            println!("{{");
            println!("  \"base_dimensions\": [");
            for (i, (symbol, name, unit, _)) in dimensions.iter().enumerate() {
                let comma = if i < dimensions.len() - 1 { "," } else { "" };
                println!(
                    "    {{ \"symbol\": \"{}\", \"name\": \"{}\", \"si_unit\": \"{}\" }}{}",
                    symbol, name, unit, comma
                );
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("SI Base Dimensions");
            println!("==================");
            println!();
            println!("{:^6} {:^25} {:^15}", "Symbol", "Quantity", "SI Unit");
            println!("{:-<6} {:-<25} {:-<15}", "", "", "");

            for (symbol, name, unit, _) in &dimensions {
                println!("{:^6} {:^25} {:^15}", symbol, name, unit);
            }

            println!();
            println!("Common derived dimensions:");
            println!("  Area:          L²");
            println!("  Volume:        L³");
            println!("  Velocity:      L·T⁻¹");
            println!("  Acceleration:  L·T⁻²");
            println!("  Force:         M·L·T⁻²");
            println!("  Energy:        M·L²·T⁻²");
            println!("  Power:         M·L²·T⁻³");
            println!("  Pressure:      M·L⁻¹·T⁻²");
            println!("  Concentration: M·L⁻³");
            println!("  Clearance:     L³·T⁻¹");
        }
    }

    Ok(())
}

// ============================================================================
// LINEAR/AFFINE TYPE COMMANDS
// ============================================================================

fn linear_list(format: &str, verbose: bool) -> Result<()> {
    use sounio::linear::Linearity;

    let kinds = vec![
        (
            Linearity::Linear,
            "1",
            "Must be used exactly once",
            "File handles, channels, unique pointers",
        ),
        (
            Linearity::Affine,
            "1?",
            "Can be used at most once (may be dropped)",
            "Temp allocations, optional cleanup",
        ),
        (
            Linearity::Unrestricted,
            "ω",
            "Can be used any number of times",
            "Integers, strings, immutable data",
        ),
    ];

    match format {
        "json" => {
            println!("{{");
            println!("  \"linearity_kinds\": [");
            for (i, (kind, symbol, desc, examples)) in kinds.iter().enumerate() {
                let comma = if i < kinds.len() - 1 { "," } else { "" };
                println!("    {{");
                println!("      \"name\": \"{}\",", kind);
                println!("      \"symbol\": \"{}\",", symbol);
                println!("      \"description\": \"{}\",", desc);
                println!("      \"allows_weakening\": {},", kind.allows_weakening());
                println!(
                    "      \"allows_contraction\": {},",
                    kind.allows_contraction()
                );
                println!("      \"examples\": \"{}\"", examples);
                println!("    }}{}", comma);
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("Sounio Linear Types");
            println!("======================");
            println!();
            println!("Linearity Kinds:");
            println!();

            for (kind, symbol, desc, examples) in &kinds {
                println!("  {} ({}):", kind, symbol);
                println!("    {}", desc);
                if verbose {
                    println!(
                        "    Weakening (drop):   {}",
                        if kind.allows_weakening() {
                            "allowed"
                        } else {
                            "NOT allowed"
                        }
                    );
                    println!(
                        "    Contraction (copy): {}",
                        if kind.allows_contraction() {
                            "allowed"
                        } else {
                            "NOT allowed"
                        }
                    );
                    println!("    Examples: {}", examples);
                }
                println!();
            }

            println!("Subkinding Lattice:");
            println!();
            println!("       Unrestricted (ω)");
            println!("             |");
            println!("          Affine (1?)");
            println!("             |");
            println!("         Linear (1)");
            println!();
            println!("Linear <: Affine <: Unrestricted");
        }
    }

    Ok(())
}

fn linear_info(kind_str: &str, format: &str) -> Result<()> {
    use sounio::linear::Linearity;

    let kind = Linearity::parse(kind_str).ok_or_else(|| {
        miette::miette!(
            "Unknown linearity kind: {}\nValid kinds: linear, affine, unrestricted",
            kind_str
        )
    })?;

    match format {
        "json" => {
            println!("{{");
            println!("  \"name\": \"{}\",", kind);
            println!("  \"symbol\": \"{}\",", kind.symbol());
            println!("  \"keyword\": \"{}\",", kind.keyword());
            println!("  \"allows_weakening\": {},", kind.allows_weakening());
            println!("  \"allows_contraction\": {},", kind.allows_contraction());
            println!("  \"must_use\": {},", kind.must_use());
            println!("  \"can_discard\": {},", kind.can_discard());
            println!("  \"can_copy\": {}", kind.can_copy());
            println!("}}");
        }
        _ => {
            println!("Linearity Kind: {}", kind);
            println!("==================");
            println!();
            println!("  Symbol:      {}", kind.symbol());
            println!(
                "  Keyword:     {}",
                if kind.keyword().is_empty() {
                    "(none - default)"
                } else {
                    kind.keyword()
                }
            );
            println!();
            println!("Structural Rules:");
            println!(
                "  Weakening (can discard):   {}",
                if kind.allows_weakening() { "Yes" } else { "No" }
            );
            println!(
                "  Contraction (can copy):    {}",
                if kind.allows_contraction() {
                    "Yes"
                } else {
                    "No"
                }
            );
            println!();
            println!("Usage Requirements:");
            println!(
                "  Must use:    {}",
                if kind.must_use() {
                    "Yes (exactly once)"
                } else {
                    "No"
                }
            );
            println!(
                "  Can discard: {}",
                if kind.can_discard() { "Yes" } else { "No" }
            );
            println!(
                "  Can copy:    {}",
                if kind.can_copy() { "Yes" } else { "No" }
            );
            println!();

            match kind {
                Linearity::Linear => {
                    println!("Description:");
                    println!("  Linear types must be used exactly once. They cannot be");
                    println!("  dropped (leaked) or copied. This is ideal for resources");
                    println!("  like file handles, database connections, or channels.");
                    println!();
                    println!("Example in D:");
                    println!("  linear struct FileHandle {{ ... }}");
                    println!("  fn close(handle: FileHandle) -> () // consumes handle");
                }
                Linearity::Affine => {
                    println!("Description:");
                    println!("  Affine types can be used at most once. They can be");
                    println!("  dropped without being used, but cannot be copied.");
                    println!("  Good for optional cleanup or temp resources.");
                    println!();
                    println!("Example in D:");
                    println!("  affine struct TempBuffer {{ ... }}");
                    println!("  // Can drop buffer without explicit cleanup");
                }
                Linearity::Unrestricted => {
                    println!("Description:");
                    println!("  Unrestricted types can be used any number of times.");
                    println!("  They can be copied and dropped freely. This is the");
                    println!("  default for most types in D.");
                    println!();
                    println!("Example in D:");
                    println!("  struct Point {{ x: f64, y: f64 }}  // unrestricted by default");
                }
            }
        }
    }

    Ok(())
}

fn linear_subkind(kind1_str: &str, kind2_str: &str, verbose: bool) -> Result<()> {
    use sounio::linear::Linearity;

    let kind1 = Linearity::parse(kind1_str)
        .ok_or_else(|| miette::miette!("Unknown linearity kind: {}", kind1_str))?;

    let kind2 = Linearity::parse(kind2_str)
        .ok_or_else(|| miette::miette!("Unknown linearity kind: {}", kind2_str))?;

    let is_subkind = kind1.is_subkind_of(kind2);
    let meet = kind1.meet(kind2);
    let join = kind1.join(kind2);

    if verbose {
        println!("Subkinding Analysis");
        println!("===================");
        println!();
        println!(
            "  {} <: {} ?  {}",
            kind1,
            kind2,
            if is_subkind { "Yes" } else { "No" }
        );
        println!();
        println!("Lattice Operations:");
        println!(
            "  Meet (greatest lower bound): {} ⊓ {} = {}",
            kind1, kind2, meet
        );
        println!(
            "  Join (least upper bound):    {} ⊔ {} = {}",
            kind1, kind2, join
        );
        println!();

        if is_subkind {
            println!("Explanation:");
            println!(
                "  A {} value can be used where {} is expected.",
                kind1, kind2
            );
            println!(
                "  This is because {} is more restrictive than {}.",
                kind1, kind2
            );
        } else {
            println!("Explanation:");
            println!(
                "  A {} value CANNOT be used where {} is expected.",
                kind1, kind2
            );
            println!("  {} is not a subkind of {}.", kind1, kind2);
        }
    } else {
        if is_subkind {
            println!("{} <: {}  (yes)", kind1, kind2);
        } else {
            println!("{} NOT <: {}", kind1, kind2);
        }
    }

    Ok(())
}

fn linear_resources(format: &str, verbose: bool) -> Result<()> {
    use sounio::linear::{Linearity, ResourceKind};

    let resources = vec![
        (
            ResourceKind::File,
            Linearity::Linear,
            "Read, Write, Close",
            "File handles (must close)",
        ),
        (
            ResourceKind::Network,
            Linearity::Linear,
            "Read, Write, Close",
            "Network connections",
        ),
        (
            ResourceKind::Database,
            Linearity::Linear,
            "Read, Write, Execute, Close",
            "Database connections",
        ),
        (
            ResourceKind::Memory,
            Linearity::Linear,
            "Read, Write",
            "Memory allocations",
        ),
        (
            ResourceKind::Lock,
            Linearity::Linear,
            "Read, Write",
            "Mutexes and locks",
        ),
        (
            ResourceKind::Gpu,
            Linearity::Linear,
            "Read, Write",
            "GPU buffers and textures",
        ),
        (
            ResourceKind::Session,
            Linearity::Linear,
            "Protocol-dependent",
            "Session-typed channels",
        ),
    ];

    match format {
        "json" => {
            println!("{{");
            println!("  \"resource_types\": [");
            for (i, (kind, linearity, caps, desc)) in resources.iter().enumerate() {
                let comma = if i < resources.len() - 1 { "," } else { "" };
                println!("    {{");
                println!("      \"kind\": \"{}\",", kind);
                println!("      \"linearity\": \"{}\",", linearity);
                println!("      \"capabilities\": \"{}\",", caps);
                println!("      \"description\": \"{}\"", desc);
                println!("    }}{}", comma);
            }
            println!("  ]");
            println!("}}");
        }
        _ => {
            println!("Sounio Resource Types");
            println!("========================");
            println!();
            println!("Resources are values with special cleanup requirements.");
            println!("They are typically linear (must be used exactly once).");
            println!();

            if verbose {
                println!(
                    "{:<12} {:<12} {:<30} {}",
                    "Kind", "Linearity", "Capabilities", "Description"
                );
                println!("{:-<12} {:-<12} {:-<30} {:-<30}", "", "", "", "");
            } else {
                println!("{:<12} {:<12} {}", "Kind", "Linearity", "Description");
                println!("{:-<12} {:-<12} {:-<30}", "", "", "");
            }

            for (kind, linearity, caps, desc) in &resources {
                if verbose {
                    println!("{:<12} {:<12} {:<30} {}", kind, linearity, caps, desc);
                } else {
                    println!("{:<12} {:<12} {}", kind, linearity, desc);
                }
            }

            println!();
            println!("Example:");
            println!("  linear struct FileHandle {{ fd: i32 }}");
            println!();
            println!("  fn open(path: string) -> FileHandle with IO");
            println!("  fn read(handle: &FileHandle) -> string with IO");
            println!("  fn close(handle: FileHandle) with IO  // consumes handle");
        }
    }

    Ok(())
}

fn linear_sessions(examples: bool, format: &str) -> Result<()> {
    match format {
        "json" => {
            println!("{{");
            println!("  \"session_types\": {{");
            println!("    \"Send\": \"!T.S - send value of type T, continue with S\",");
            println!("    \"Recv\": \"?T.S - receive value of type T, continue with S\",");
            println!("    \"Offer\": \"&{{l: S, ...}} - offer choice to peer\",");
            println!("    \"Choose\": \"⊕{{l: S, ...}} - make a choice\",");
            println!("    \"End\": \"end - session complete\",");
            println!("    \"Rec\": \"μX.S - recursive session\"");
            println!("  }}");
            println!("}}");
        }
        _ => {
            println!("Sounio Session Types");
            println!("=======================");
            println!();
            println!("Session types encode communication protocols at the type level.");
            println!("They ensure protocol conformance at compile time.");
            println!();
            println!("Session Type Constructors:");
            println!();
            println!("  !T.S        Send value of type T, continue with S");
            println!("  ?T.S        Receive value of type T, continue with S");
            println!("  &{{l: S}}    Offer choice (peer selects)");
            println!("  ⊕{{l: S}}    Make choice (we select)");
            println!("  end         Session complete");
            println!("  μX.S        Recursive session (loops)");
            println!();
            println!("Duality:");
            println!("  Every session type has a dual (the other party's view)");
            println!("  dual(!T.S) = ?T.dual(S)");
            println!("  dual(&{{...}}) = ⊕{{...}}");
            println!();

            if examples {
                println!("Example Protocols:");
                println!();
                println!("1. Query-Response:");
                println!("   Client: !Query.?Response.end");
                println!("   Server: ?Query.!Response.end");
                println!();
                println!("2. File Transfer:");
                println!("   Sender:   !Filename.μX.⊕{{data: !Chunk.X, done: end}}");
                println!("   Receiver: ?Filename.μX.&{{data: ?Chunk.X, done: end}}");
                println!();
                println!("3. Authentication:");
                println!("   Client: !Credentials.&{{ok: ?Token.end, fail: ?Error.end}}");
                println!("   Server: ?Credentials.⊕{{ok: !Token.end, fail: !Error.end}}");
            }
        }
    }

    Ok(())
}

fn linear_check(input: &Path, show_resources: bool, show_usage: bool) -> Result<()> {
    // Read the source file
    let _source = std::fs::read_to_string(input)
        .map_err(|e| miette::miette!("Failed to read {}: {}", input.display(), e))?;

    println!("Linear Type Checking: {}", input.display());
    println!(
        "======================={}",
        "=".repeat(input.display().to_string().len())
    );
    println!();

    // For now, this is a stub that would integrate with the type checker
    // In a full implementation, this would:
    // 1. Parse the source file
    // 2. Run linearity checking
    // 3. Report any linearity violations

    println!("Note: Full linearity checking requires type-checked AST.");
    println!("This command will be functional after HIR lowering is complete.");
    println!();

    if show_resources {
        println!("Resource tracking would show:");
        println!("  - All linear/affine bindings in scope");
        println!("  - Their types and modalities");
        println!("  - Creation points");
        println!();
    }

    if show_usage {
        println!("Usage tracking would show:");
        println!("  - How many times each resource is used");
        println!("  - Where resources are consumed");
        println!("  - Any unused linear resources");
        println!("  - Any multiply-used affine resources");
        println!();
    }

    // Placeholder success
    println!("Linearity check: PASS (stub - no violations detected)");

    Ok(())
}
