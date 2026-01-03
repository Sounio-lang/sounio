//! # Sounio Compiler: CLI Backend Selection
//!
//! This module provides CLI integration for backend selection, including
//! the native SIR backend that bypasses LLVM.
//!
//! ## Usage
//!
//! ```bash
//! # Use native backend (default if LLVM has issues)
//! souc build --backend=native src/main.sio -o output.so
//!
//! # Use LLVM backend
//! souc build --backend=llvm src/main.sio -o output.so
//!
//! # Use Cranelift JIT
//! souc run --backend=cranelift src/main.sio
//!
//! # Show backend info
//! souc backend --list
//! souc backend --info native
//! ```

use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

// Re-export from codegen for compatibility
use crate::codegen::Backend as CodegenBackend;

// ============================================================================
// BACKEND SELECTION (Extended)
// ============================================================================

/// Available compiler backends (extended version for CLI)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, clap::ValueEnum)]
pub enum Backend {
    /// Native SIR backend (x86-64 direct emission)
    /// - No external dependencies
    /// - Epistemic-aware register allocation
    /// - Thermal degradation modeling
    #[value(name = "native")]
    Native,
    
    /// LLVM backend (AOT compilation)
    /// - Wide architecture support
    /// - Mature optimizations
    #[value(name = "llvm")]
    Llvm,
    
    /// Cranelift backend (JIT compilation)
    /// - Fast compilation
    /// - Good for development/REPL
    #[value(name = "cranelift")]
    Cranelift,
    
    /// GPU backend (PTX emission)
    /// - NVIDIA GPU target
    /// - Kernel compilation
    #[value(name = "gpu")]
    Gpu,
}

impl Backend {
    /// Get all available backends
    pub fn all() -> &'static [Backend] {
        &[
            Backend::Native,
            Backend::Llvm,
            Backend::Cranelift,
            Backend::Gpu,
        ]
    }

    /// Get the default backend
    pub fn default_for_target(_target: CompileTarget) -> Self {
        // Prefer native to avoid LLVM issues
        Backend::Native
    }

    /// Check if backend is available (compiled in)
    pub fn is_available(&self) -> bool {
        match self {
            Backend::Native => true,  // Always available
            Backend::Llvm => cfg!(feature = "llvm-base"),
            Backend::Cranelift => cfg!(feature = "jit"),
            Backend::Gpu => cfg!(feature = "gpu"),
        }
    }

    /// Get backend display name
    pub fn name(&self) -> &'static str {
        match self {
            Backend::Native => "native",
            Backend::Llvm => "llvm",
            Backend::Cranelift => "cranelift",
            Backend::Gpu => "gpu",
        }
    }

    /// Get backend description
    pub fn description(&self) -> &'static str {
        match self {
            Backend::Native => "Native SIR backend with epistemic-aware allocation (x86-64)",
            Backend::Llvm => "LLVM backend for AOT compilation (multi-arch)",
            Backend::Cranelift => "Cranelift JIT backend for fast compilation",
            Backend::Gpu => "GPU backend for NVIDIA PTX emission",
        }
    }

    /// Get known issues/warnings
    pub fn warnings(&self) -> Option<&'static str> {
        match self {
            Backend::Llvm => Some("⚠️  May have issues with refinement types + FFI"),
            Backend::Gpu => Some("Requires CUDA toolkit"),
            _ => None,
        }
    }

    /// Convert to codegen::Backend
    pub fn to_codegen(&self) -> Option<CodegenBackend> {
        match self {
            Backend::Native => None,  // Not in codegen enum yet
            Backend::Llvm => Some(CodegenBackend::LLVM),
            Backend::Cranelift => Some(CodegenBackend::Cranelift),
            Backend::Gpu => Some(CodegenBackend::GPU),
        }
    }
}

impl Default for Backend {
    fn default() -> Self {
        Backend::Native
    }
}

impl fmt::Display for Backend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

impl FromStr for Backend {
    type Err = BackendParseError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "native" | "sir" | "x86" | "x86-64" | "x86_64" => Ok(Backend::Native),
            "llvm" => Ok(Backend::Llvm),
            "cranelift" | "cl" | "jit" => Ok(Backend::Cranelift),
            "gpu" | "cuda" | "ptx" => Ok(Backend::Gpu),
            _ => Err(BackendParseError(s.to_string())),
        }
    }
}

/// Error parsing backend name
#[derive(Debug)]
pub struct BackendParseError(String);

impl fmt::Display for BackendParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Unknown backend '{}'. Available: native, llvm, cranelift, gpu", self.0)
    }
}

impl std::error::Error for BackendParseError {}

// ============================================================================
// OUTPUT FORMAT
// ============================================================================

/// Output format for compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Shared library (.so / .dylib / .dll)
    SharedLib,
    /// Object file (.o)
    Object,
    /// Assembly (.s)
    Asm,
    /// LLVM IR (.ll)
    LlvmIr,
    /// LLVM Bitcode (.bc)
    LlvmBc,
    /// NVIDIA PTX (.ptx)
    Ptx,
    /// NVIDIA Cubin (.cubin)
    Cubin,
    /// JIT (in-memory, no file)
    Jit,
}

impl OutputFormat {
    /// Get file extension
    pub fn extension(&self) -> &'static str {
        match self {
            OutputFormat::SharedLib => {
                #[cfg(target_os = "linux")]
                { "so" }
                #[cfg(target_os = "macos")]
                { "dylib" }
                #[cfg(target_os = "windows")]
                { "dll" }
                #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
                { "so" }
            }
            OutputFormat::Object => "o",
            OutputFormat::Asm => "s",
            OutputFormat::LlvmIr => "ll",
            OutputFormat::LlvmBc => "bc",
            OutputFormat::Ptx => "ptx",
            OutputFormat::Cubin => "cubin",
            OutputFormat::Jit => "",
        }
    }
}

// ============================================================================
// COMPILE TARGET
// ============================================================================

/// High-level compilation target
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompileTarget {
    /// Shared library for FFI/dynamic loading
    SharedLibrary,
    /// Standalone executable
    Executable,
    /// JIT execution
    Jit,
    /// GPU kernel
    GpuKernel,
}

// ============================================================================
// BACKEND OPTIONS
// ============================================================================

/// Options specific to the native backend
#[derive(Debug, Clone)]
pub struct NativeBackendOptions {
    /// Optimization level
    pub opt_level: OptLevel,
    /// Enable thermal modeling
    pub enable_thermal: bool,
    /// Thermal model to use
    pub thermal_model: ThermalModel,
    /// Register allocation strategy
    pub alloc_strategy: AllocStrategy,
    /// Emit debug info
    pub debug_info: bool,
    /// Target CPU features
    pub cpu_features: Vec<String>,
}

impl Default for NativeBackendOptions {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Default,
            enable_thermal: true,
            thermal_model: ThermalModel::Arrhenius7nm,
            alloc_strategy: AllocStrategy::Epistemic,
            debug_info: false,
            cpu_features: vec!["sse4.2".into(), "avx2".into()],
        }
    }
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum OptLevel {
    /// No optimization (fastest compile)
    #[value(name = "0", alias = "none")]
    None,
    /// Default optimization
    #[value(name = "1", alias = "default")]
    #[default]
    Default,
    /// Aggressive optimization
    #[value(name = "2", alias = "3", alias = "aggressive")]
    Aggressive,
    /// Optimize for size
    #[value(name = "s", alias = "z", alias = "size")]
    Size,
}

impl FromStr for OptLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" | "none" => Ok(OptLevel::None),
            "1" | "default" | "" => Ok(OptLevel::Default),
            "2" | "3" | "aggressive" => Ok(OptLevel::Aggressive),
            "s" | "z" | "size" => Ok(OptLevel::Size),
            _ => Err(format!("Unknown optimization level: {}", s)),
        }
    }
}

/// Thermal model selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum ThermalModel {
    /// No thermal modeling
    #[value(name = "none", alias = "off")]
    None,
    /// 7nm FinFET Arrhenius model
    #[value(name = "7nm", alias = "arrhenius", alias = "default")]
    #[default]
    Arrhenius7nm,
    /// 5nm process model
    #[value(name = "5nm")]
    Arrhenius5nm,
    /// Conservative model for safety-critical
    #[value(name = "conservative", alias = "safe")]
    Conservative,
}

impl FromStr for ThermalModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "off" => Ok(ThermalModel::None),
            "7nm" | "arrhenius" | "default" => Ok(ThermalModel::Arrhenius7nm),
            "5nm" => Ok(ThermalModel::Arrhenius5nm),
            "conservative" | "safe" => Ok(ThermalModel::Conservative),
            _ => Err(format!("Unknown thermal model: {}", s)),
        }
    }
}

/// Register allocation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub enum AllocStrategy {
    /// Epistemic-aware allocation (confidence-based spilling)
    #[value(name = "epistemic", alias = "default")]
    #[default]
    Epistemic,
    /// Standard linear scan
    #[value(name = "linear", alias = "linearscan", alias = "linear-scan")]
    LinearScan,
    /// Graph coloring (slower, potentially better)
    #[value(name = "graph", alias = "coloring", alias = "graph-coloring")]
    GraphColoring,
}

impl FromStr for AllocStrategy {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "epistemic" | "default" => Ok(AllocStrategy::Epistemic),
            "linear" | "linearscan" | "linear-scan" => Ok(AllocStrategy::LinearScan),
            "graph" | "coloring" | "graph-coloring" => Ok(AllocStrategy::GraphColoring),
            _ => Err(format!("Unknown allocation strategy: {}", s)),
        }
    }
}

// ============================================================================
// CLI ARGUMENT STRUCTURES
// ============================================================================

/// Build command arguments (extracted from clap)
#[derive(Debug, Clone)]
pub struct BuildArgs {
    /// Input source file(s)
    pub input: Vec<PathBuf>,
    /// Output file
    pub output: Option<PathBuf>,
    /// Backend to use
    pub backend: Backend,
    /// Output format (inferred from extension if not specified)
    pub format: Option<OutputFormat>,
    /// Native backend options
    pub native_opts: NativeBackendOptions,
    /// Verbose output
    pub verbose: bool,
    /// Show timing information
    pub timing: bool,
    /// Emit intermediate representations
    pub emit: Vec<EmitStage>,
}

impl Default for BuildArgs {
    fn default() -> Self {
        Self {
            input: Vec::new(),
            output: None,
            backend: Backend::default(),
            format: None,
            native_opts: NativeBackendOptions::default(),
            verbose: false,
            timing: false,
            emit: Vec::new(),
        }
    }
}

/// Stages that can be emitted for debugging
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmitStage {
    /// Parsed AST
    Ast,
    /// Type-checked HIR
    Hir,
    /// SIR (Sounio IR)
    Sir,
    /// After optimization
    SirOpt,
    /// Register allocation info
    RegAlloc,
    /// Final assembly
    Asm,
}

impl FromStr for EmitStage {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ast" => Ok(EmitStage::Ast),
            "hir" => Ok(EmitStage::Hir),
            "sir" | "ir" => Ok(EmitStage::Sir),
            "sir-opt" | "opt" => Ok(EmitStage::SirOpt),
            "regalloc" | "alloc" => Ok(EmitStage::RegAlloc),
            "asm" | "assembly" => Ok(EmitStage::Asm),
            _ => Err(format!("Unknown emit stage: {}", s)),
        }
    }
}

// ============================================================================
// CLI PARSER HELPERS
// ============================================================================

/// Extract BuildArgs from clap ArgMatches
/// 
/// This extracts arguments from the Build command in main.rs
pub fn extract_build_args(arg_matches: &clap::ArgMatches) -> Result<BuildArgs, String> {
    
    let mut build_args = BuildArgs::default();
    
    // Input file (required positional)
    if let Some(input_str) = arg_matches.get_one::<String>("input") {
        build_args.input.push(PathBuf::from(input_str));
    }
    
    // Output file
    if let Some(output_str) = arg_matches.get_one::<String>("output") {
        build_args.output = Some(PathBuf::from(output_str));
    }
    
    // Backend
    if let Some(backend_str) = arg_matches.get_one::<String>("backend") {
        build_args.backend = backend_str.parse()
            .map_err(|e: BackendParseError| e.to_string())?;
    }
    
    // Optimization level
    if let Some(opt_str) = arg_matches.get_one::<String>("opt_level") {
        build_args.native_opts.opt_level = opt_str.parse()?;
    }
    
    // Thermal model
    if let Some(thermal_str) = arg_matches.get_one::<String>("thermal") {
        build_args.native_opts.thermal_model = thermal_str.parse()?;
    }
    
    // No thermal flag
    if arg_matches.get_flag("no_thermal") {
        build_args.native_opts.enable_thermal = false;
    }
    
    // Allocation strategy
    if let Some(alloc_str) = arg_matches.get_one::<String>("alloc") {
        build_args.native_opts.alloc_strategy = alloc_str.parse()?;
    }
    
    // Debug info
    if arg_matches.get_flag("debug") {
        build_args.native_opts.debug_info = true;
    }
    
    // Verbose
    if arg_matches.get_flag("verbose") {
        build_args.verbose = true;
    }
    
    // Timing
    if arg_matches.get_flag("timing") {
        build_args.timing = true;
    }
    
    // Emit stages
    if let Some(emit_str) = arg_matches.get_one::<String>("emit") {
        for stage_str in emit_str.split(',') {
            build_args.emit.push(stage_str.trim().parse()?);
        }
    }
    
    // CPU features
    if let Some(features_str) = arg_matches.get_one::<String>("target_cpu") {
        build_args.native_opts.cpu_features = features_str.split(',')
            .map(|s| s.trim().to_string())
            .collect();
    }
    
    // Check backend availability
    if !build_args.backend.is_available() {
        return Err(format!(
            "Backend '{}' is not available. Compile with the appropriate feature flag.",
            build_args.backend
        ));
    }
    
    Ok(build_args)
}

/// Print help for backend command
pub fn print_backend_help() {
    println!("Backend selection commands:");
    println!();
    println!("  souc backend --list          List available backends");
    println!("  souc backend --info <name>   Show detailed info for a backend");
    println!();
    println!("Build options:");
    println!("  --backend=<name>             Select backend (native, llvm, cranelift, gpu)");
    println!("  -O<level>                    Optimization level (0, 1, 2, 3, s)");
    println!("  --thermal=<model>            Thermal model (none, 7nm, 5nm, conservative)");
    println!("  --no-thermal                 Disable thermal modeling");
    println!("  --alloc=<strategy>           Allocation strategy (epistemic, linear, graph)");
    println!("  -g, --debug                  Emit debug information");
    println!("  --emit=<stages>              Emit intermediate stages (ast,sir,asm)");
    println!();
}

/// Print list of backends
pub fn print_backend_list() {
    println!("Available backends:");
    println!();
    for backend in Backend::all() {
        let status = if backend.is_available() { "✓" } else { "✗" };
        println!("  {} {:<12} {}", status, backend.name(), backend.description());
        if let Some(warning) = backend.warnings() {
            println!("              {}", warning);
        }
    }
    println!();
}

/// Print detailed info for a backend
pub fn print_backend_info(backend: Backend) {
    println!("Backend: {}", backend.name());
    println!("Status: {}", if backend.is_available() { "Available" } else { "Not compiled" });
    println!("Description: {}", backend.description());
    
    if let Some(warning) = backend.warnings() {
        println!("Warning: {}", warning);
    }
    
    if backend == Backend::Native {
        println!();
        println!("Native backend features:");
        println!("  - Epistemic-aware register allocation");
        println!("  - Confidence-based spilling decisions");
        println!("  - Thermal degradation modeling (Arrhenius)");
        println!("  - No external dependencies (pure Rust)");
        println!("  - Direct x86-64 machine code emission");
    }
}

// ============================================================================
// DRIVER INTEGRATION
// ============================================================================

/// Compile result with metrics
#[derive(Debug)]
pub struct CompileOutput {
    /// Output file path (if any)
    pub output_path: Option<PathBuf>,
    /// Compilation succeeded
    pub success: bool,
    /// Error messages
    pub errors: Vec<String>,
    /// Warning messages
    pub warnings: Vec<String>,
    /// Timing information (if requested)
    pub timing: Option<CompileTiming>,
    /// Backend-specific metrics
    pub metrics: Option<BackendMetrics>,
}

/// Timing breakdown
#[derive(Debug, Clone)]
pub struct CompileTiming {
    pub parse_ms: u64,
    pub typecheck_ms: u64,
    pub lower_ms: u64,
    pub optimize_ms: u64,
    pub codegen_ms: u64,
    pub total_ms: u64,
}

/// Backend-specific metrics
#[derive(Debug, Clone)]
pub struct BackendMetrics {
    /// For native backend
    pub native: Option<NativeMetrics>,
}

/// Native backend metrics
#[derive(Debug, Clone)]
pub struct NativeMetrics {
    pub intervals_allocated: usize,
    pub intervals_spilled: usize,
    pub avg_confidence_allocated: f64,
    pub avg_confidence_spilled: f64,
    pub total_spill_slots: usize,
    pub estimated_cycles: u64,
    pub estimated_power_uw: f64,
    pub thermal_degradation: f64,
}

/// Allocation metrics (internal)
#[derive(Debug, Clone)]
struct AllocMetrics {
    intervals_allocated: usize,
    intervals_spilled: usize,
    avg_confidence: f64,
    spill_slots: usize,
}

/// Main driver function for compilation
/// 
/// This integrates with the existing compiler infrastructure
pub fn compile(args: &BuildArgs) -> CompileOutput {
    use std::time::Instant;
    
    let start = Instant::now();
    let mut output = CompileOutput {
        output_path: args.output.clone(),
        success: false,
        errors: Vec::new(),
        warnings: Vec::new(),
        timing: None,
        metrics: None,
    };

    // Validate inputs
    for input in &args.input {
        if !input.exists() {
            output.errors.push(format!("Input file not found: {}", input.display()));
            return output;
        }
    }

    // Show backend info if verbose
    if args.verbose {
        println!("Using backend: {} ({})", args.backend, args.backend.description());
        if let Some(warning) = args.backend.warnings() {
            println!("{}", warning);
        }
    }

    // Dispatch to appropriate backend
    let result = match args.backend {
        Backend::Native => compile_native(args),
        Backend::Llvm => compile_llvm(args),
        Backend::Cranelift => compile_cranelift(args),
        Backend::Gpu => compile_gpu(args),
    };

    match result {
        Ok((path, metrics)) => {
            output.success = true;
            output.output_path = Some(path);
            output.metrics = Some(BackendMetrics { native: metrics });
        }
        Err(e) => {
            output.errors.push(e);
        }
    }

    if args.timing {
        output.timing = Some(CompileTiming {
            parse_ms: 0,
            typecheck_ms: 0,
            lower_ms: 0,
            optimize_ms: 0,
            codegen_ms: 0,
            total_ms: start.elapsed().as_millis() as u64,
        });
    }

    output
}

/// Compile using native backend
fn compile_native(args: &BuildArgs) -> Result<(PathBuf, Option<NativeMetrics>), String> {
    use std::time::Instant;
    use crate::backend::native::{NativeBackend, NativeBackendConfig, TargetArch};
    use crate::backend::native::thermal::ArrheniusModel;
    use crate::module_loader::load_program_ast;
    use crate::check::check;
    use crate::hlir::lower as lower_hlir;
    use crate::sir::lower::lower_module;
    use crate::sir::emit::emit_code;
    let start = Instant::now();

    // Step 1: Load and parse AST
    let parse_start = Instant::now();
    let input_path = &args.input[0];
    let ast = load_program_ast(input_path)
        .map_err(|e| format!("Parse error: {}", e))?;
    let parse_ms = parse_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Parsed in {}ms", parse_ms);
    }

    // Emit AST if requested
    if args.emit.contains(&EmitStage::Ast) {
        println!("=== AST ===");
        println!("{:#?}", ast);
    }

    // Step 2: Type check (AST → HIR)
    let typecheck_start = Instant::now();
    let hir = check(&ast)
        .map_err(|e| format!("Type check error: {}", e))?;
    let typecheck_ms = typecheck_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Type checked in {}ms", typecheck_ms);
    }

    // Emit HIR if requested
    if args.emit.contains(&EmitStage::Hir) {
        println!("=== HIR ===");
        println!("{:#?}", hir);
    }

    // Step 3: Lower to HLIR (HIR → HLIR)
    let lower_hlir_start = Instant::now();
    let hlir = lower_hlir(&hir);
    let lower_hlir_ms = lower_hlir_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Lowered to HLIR in {}ms", lower_hlir_ms);
    }

    // Step 4: Lower to SIR (HLIR → SIR)
    let lower_sir_start = Instant::now();
    let (mut sir_module, lowering_ctx) = lower_module(&hlir);
    
    // Set target architecture
    sir_module.target = crate::sir::module::TargetTriple::native();
    
    // Apply optimization flags
    sir_module.flags.debug = args.native_opts.debug_info;
    sir_module.flags.opt_level = match args.native_opts.opt_level {
        OptLevel::None => 0,
        OptLevel::Default => 1,
        OptLevel::Aggressive => 3,
        OptLevel::Size => 1, // Size optimization uses level 1
    };
    let lower_sir_ms = lower_sir_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Lowered to SIR in {}ms", lower_sir_ms);
    }

    // Emit SIR if requested
    if args.emit.contains(&EmitStage::Sir) {
        println!("=== SIR Module ===");
        println!("Functions: {}", sir_module.functions.len());
        for func in &sir_module.functions {
            println!("  - {} ({} blocks)", func.name, func.blocks.len());
        }
    }

    // Step 5: Configure native backend
    let thermal_model = match args.native_opts.thermal_model {
        ThermalModel::None => ArrheniusModel::default(),
        ThermalModel::Arrhenius7nm => ArrheniusModel::combined_7nm(),
        ThermalModel::Arrhenius5nm => ArrheniusModel::combined_5nm(),
        ThermalModel::Conservative => ArrheniusModel::conservative(),
    };

    let native_opt_level = match args.native_opts.opt_level {
        OptLevel::None => crate::backend::native::OptLevel::None,
        OptLevel::Default => crate::backend::native::OptLevel::Default,
        OptLevel::Aggressive => crate::backend::native::OptLevel::Aggressive,
        OptLevel::Size => crate::backend::native::OptLevel::Size,
    };

    let config = NativeBackendConfig {
        arch: TargetArch::X86_64,
        enable_thermal_tracking: args.native_opts.enable_thermal,
        thermal_model,
        confidence_floor: 0.05,
        opt_level: native_opt_level,
        ..Default::default()
    };

    let mut backend = NativeBackend::with_config(config);

    // Step 6: Analyze SIR blocks with native backend and apply autotuning
    let analyze_start = Instant::now();
    let mut total_cycles = 0u64;
    let mut total_power_uw = 0.0_f64;
    let mut total_thermal_degradation = 0.0_f64;
    let mut block_count = 0;
    let mut max_temp_k: f64 = 298.0; // Ambient

    for func in &sir_module.functions {
        for block in &func.blocks {
            // Convert SIR block to metrics::SirBlock format
            let metrics_block = convert_sir_block_to_metrics_block(block);
            let analysis = backend.analyze_block(&metrics_block);
            
            total_cycles += analysis.cycles.cycles;
            total_power_uw += analysis.power.average_power_uw;
            if let Some(ref thermal) = analysis.thermal {
                total_thermal_degradation += thermal.degradation_factor;
            }
            
            // Track maximum temperature
            let block_temp = analysis.power.temperature_rise_k + 298.0_f64;
            max_temp_k = max_temp_k.max(block_temp);
            block_count += 1;
        }
    }

    // Apply autotuning: adjust DVFS if temperature is too high
    if args.native_opts.enable_thermal && max_temp_k > 350.0 {
        // Temperature is high - reduce voltage/frequency
        let temp_excess = max_temp_k - 350.0;
        let reduction_factor = (temp_excess / 50.0_f64).min(0.3_f64); // Max 30% reduction
        
        if args.verbose {
            println!("⚠️  High temperature detected: {:.1}K", max_temp_k);
            println!("   Applying DVFS reduction: {:.1}%", reduction_factor * 100.0);
        }

        // Adjust power model DVFS (this would affect future estimates)
        // In a real implementation, this would feed back into the power model
        let new_voltage = 0.75 * (1.0 - reduction_factor);
        let new_freq = 3.5 * (1.0 - reduction_factor);
        
        if args.verbose {
            println!("   Recommended: V={:.3}V, f={:.2}GHz", new_voltage, new_freq);
        }
    }

    let analyze_ms = analyze_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Analyzed {} blocks in {}ms", block_count, analyze_ms);
        println!("  Total cycles: {}", total_cycles);
        println!("  Total power: {:.2} μW", total_power_uw);
        println!("  Thermal degradation: {:.4}", total_thermal_degradation);
    }

    // Step 7: Run register allocation (if epistemic-aware)
    let alloc_start = Instant::now();
    use std::collections::HashMap;
    use crate::backend::native::alloc::{
        EpistemicAllocator, AllocConfig, AllocResult,
        extract_epistemic_metadata, build_intervals_from_sir, RegClass,
    };
    use crate::sir::values::FuncId;
    
    let mut alloc_results: HashMap<FuncId, AllocResult> = HashMap::new();
    let alloc_metrics = if args.native_opts.alloc_strategy == AllocStrategy::Epistemic {
        // Use epistemic-aware allocator for each function
        for func in &sir_module.functions {
            // Extract epistemic metadata from lowering context
            let ep_metadata = extract_epistemic_metadata(&sir_module, func, &lowering_ctx.metadata);
            
            // Build live intervals with epistemic metadata
            // For now, assume all values are general-purpose registers (can be improved)
            let intervals = build_intervals_from_sir(func, &ep_metadata, RegClass::GeneralPurpose);
            
            // Run allocator with error handling
            let mut allocator = EpistemicAllocator::with_config(AllocConfig::default());
            let result = allocator.allocate(intervals);
            
            // Check if allocation was successful
            if !result.is_successful() && args.verbose {
                println!("⚠️  Warning: Allocation for function '{}' had {} critical spills",
                    func.name, result.stats.critical_spills);
            }
            
            alloc_results.insert(func.id, result);
        }
        
        // Compute aggregate metrics
        let mut total_allocated = 0;
        let mut total_spilled = 0;
        let mut total_confidence = 0.0;
        let mut total_spill_slots = 0;
        
        for result in alloc_results.values() {
            total_allocated += result.allocated.len();
            total_spilled += result.spilled.len();
            total_confidence += result.stats.avg_allocated_confidence;
            total_spill_slots += result.total_spill_size;
        }
        
        let func_count = alloc_results.len().max(1);
        Some(AllocMetrics {
            intervals_allocated: total_allocated,
            intervals_spilled: total_spilled,
            avg_confidence: total_confidence / func_count as f64,
            spill_slots: total_spill_slots,
        })
    } else {
        None
    };
    let alloc_ms = alloc_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Register allocation in {}ms", alloc_ms);
        if let Some(ref metrics) = alloc_metrics {
            println!("  Allocated: {}, Spilled: {}, Avg confidence: {:.3}",
                metrics.intervals_allocated,
                metrics.intervals_spilled,
                metrics.avg_confidence);
            if metrics.spill_slots > 0 {
                println!("  Spill slots: {} bytes", metrics.spill_slots);
            }
        }
    }
    
    // Error handling: fallback to internal allocator if external fails
    if args.native_opts.alloc_strategy == AllocStrategy::Epistemic && alloc_results.is_empty() {
        if args.verbose {
            println!("⚠️  Warning: Epistemic allocation produced no results, falling back to internal allocator");
        }
    }

    // Step 8: Emit machine code
    let codegen_start = Instant::now();
    let alloc_results_opt = if args.native_opts.alloc_strategy == AllocStrategy::Epistemic {
        Some(alloc_results)
    } else {
        None
    };
    let code_segment = emit_code(&sir_module, alloc_results_opt)
        .map_err(|e| format!("Code emission error: {:?}", e))?;
    let codegen_ms = codegen_start.elapsed().as_millis() as u64;

    if args.verbose {
        println!("✓ Code generation in {}ms", codegen_ms);
        println!("  Generated {} bytes", code_segment.code.len());
    }

    // Emit assembly if requested
    if args.emit.contains(&EmitStage::Asm) {
        println!("=== Generated Code ===");
        println!("{} bytes of machine code", code_segment.code.len());
        println!("{} symbols", code_segment.symbols.len());
        println!("{} relocations", code_segment.relocations.len());
    }

    // Step 9: Write output
    let output_path = args.output.clone()
        .unwrap_or_else(|| {
            let mut p = args.input[0].clone();
            p.set_extension("so");
            p
        });

    // Determine output format
    let format = args.format.unwrap_or_else(|| {
        match output_path.extension().and_then(|e| e.to_str()) {
            Some("so") | Some("dylib") | Some("dll") => OutputFormat::SharedLib,
            Some("o") => OutputFormat::Object,
            Some("s") => OutputFormat::Asm,
            _ => OutputFormat::SharedLib, // Default
        }
    });

    match format {
        OutputFormat::SharedLib | OutputFormat::Object => {
            // Convert CodeSegment to ELF format
            use crate::sir::emit::code_segment_to_elf;
            let elf_bytes = code_segment_to_elf(&code_segment)
                .map_err(|e| format!("ELF generation error: {:?}", e))?;
            std::fs::write(&output_path, &elf_bytes)
                .map_err(|e| format!("Failed to write output: {}", e))?;
        }
        OutputFormat::Asm => {
            // Write assembly (would need disassembly)
            return Err("Assembly output not yet implemented".into());
        }
        _ => {
            return Err(format!("Unsupported output format: {:?}", format));
        }
    }

    if args.verbose {
        println!("✓ Wrote output to {}", output_path.display());
        println!("Total compilation time: {}ms", start.elapsed().as_millis());
    }

    // Collect metrics
    let metrics = NativeMetrics {
        intervals_allocated: alloc_metrics.as_ref().map(|m| m.intervals_allocated).unwrap_or(0),
        intervals_spilled: alloc_metrics.as_ref().map(|m| m.intervals_spilled).unwrap_or(0),
        avg_confidence_allocated: alloc_metrics.as_ref().map(|m| m.avg_confidence).unwrap_or(0.0),
        avg_confidence_spilled: 0.0, // TODO: Track spilled confidence
        total_spill_slots: alloc_metrics.as_ref().map(|m| m.spill_slots).unwrap_or(0),
        estimated_cycles: total_cycles,
        estimated_power_uw: total_power_uw,
        thermal_degradation: total_thermal_degradation,
    };

    Ok((output_path, Some(metrics)))
}

/// Convert SIR block to metrics::SirBlock
/// This is a simplified adapter - in production, we'd have proper integration
fn convert_sir_block_to_metrics_block(block: &crate::sir::blocks::BasicBlock) -> crate::backend::native::metrics::SirBlock {
    use crate::backend::native::metrics::{SirBlock, SirInstruction, OpClass};
    
    let mut metrics_block = SirBlock::new(&format!("block_{}", block.id.0));
    
    for inst in &block.instructions {
        // Map SIR instruction to OpClass
        let op_class = match &inst.inst {
            crate::sir::ops::SirInst::BinOp { op, .. } => {
                match op {
                    crate::sir::ops::ArithOp::Add | crate::sir::ops::ArithOp::Sub => OpClass::IntArithmetic,
                    crate::sir::ops::ArithOp::Mul => OpClass::IntMultiply,
                    crate::sir::ops::ArithOp::SDiv | crate::sir::ops::ArithOp::UDiv => OpClass::IntDivide,
                    crate::sir::ops::ArithOp::FAdd | crate::sir::ops::ArithOp::FSub => OpClass::FloatArithmetic,
                    crate::sir::ops::ArithOp::FMul => OpClass::FloatMultiply,
                    crate::sir::ops::ArithOp::FDiv => OpClass::FloatDivide,
                    _ => OpClass::IntArithmetic,
                }
            }
            crate::sir::ops::SirInst::UnaryFloat { op, .. } => {
                match op {
                    crate::sir::ops::UnaryFloatOp::Sqrt => OpClass::FloatSqrt,
                    _ => OpClass::FloatArithmetic,
                }
            }
            crate::sir::ops::SirInst::BinaryFloat { op, .. } => {
                match op {
                    crate::sir::ops::BinaryFloatOp::Pow => OpClass::FloatMultiply,
                    _ => OpClass::FloatArithmetic,
                }
            }
            crate::sir::ops::SirInst::Memory(op) => {
                match op {
                    crate::sir::ops::MemoryOp::Load { .. } => OpClass::MemoryLoad,
                    crate::sir::ops::MemoryOp::Store { .. } => OpClass::MemoryStore,
                    _ => OpClass::Nop,
                }
            }
            crate::sir::ops::SirInst::Call(_) => OpClass::Call,
            // Return is handled by Terminator, not SirInst
            _ => OpClass::Nop,
        };

        // Check for epistemic metadata (if instruction has a result)
        let epistemic = None; // Simplified for now - would query metadata store

        metrics_block.push(SirInstruction {
            op_class,
            sources: vec![], // Simplified
            destination: None,
            epistemic,
        });
    }

    metrics_block
}

/// Compile using LLVM backend
fn compile_llvm(args: &BuildArgs) -> Result<(PathBuf, Option<NativeMetrics>), String> {
    #[cfg(feature = "llvm-base")]
    {
        // TODO: Implement LLVM backend call
        Err("LLVM backend: May have issues with refinement types + FFI".into())
    }
    
    #[cfg(not(feature = "llvm-base"))]
    {
        Err("LLVM backend not compiled. Use --backend=native instead.".into())
    }
}

/// Compile using Cranelift backend
fn compile_cranelift(args: &BuildArgs) -> Result<(PathBuf, Option<NativeMetrics>), String> {
    #[cfg(feature = "jit")]
    {
        // TODO: Implement Cranelift backend call
        let output_path = args.output.clone()
            .unwrap_or_else(|| PathBuf::from("a.out"));
        Ok((output_path, None))
    }
    
    #[cfg(not(feature = "jit"))]
    {
        Err("Cranelift backend not compiled. Add --features jit.".into())
    }
}

/// Compile using GPU backend
fn compile_gpu(args: &BuildArgs) -> Result<(PathBuf, Option<NativeMetrics>), String> {
    #[cfg(feature = "gpu")]
    {
        // TODO: Implement GPU backend call
        let mut output_path = args.output.clone()
            .unwrap_or_else(|| args.input[0].clone());
        output_path.set_extension("ptx");
        Ok((output_path, None))
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        Err("GPU backend not compiled. Add --features gpu.".into())
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_parsing() {
        assert_eq!("native".parse::<Backend>().unwrap(), Backend::Native);
        assert_eq!("llvm".parse::<Backend>().unwrap(), Backend::Llvm);
        assert_eq!("cranelift".parse::<Backend>().unwrap(), Backend::Cranelift);
        assert_eq!("jit".parse::<Backend>().unwrap(), Backend::Cranelift);
        assert_eq!("gpu".parse::<Backend>().unwrap(), Backend::Gpu);
        assert!("invalid".parse::<Backend>().is_err());
    }

    #[test]
    fn test_opt_level_parsing() {
        assert_eq!("0".parse::<OptLevel>().unwrap(), OptLevel::None);
        assert_eq!("1".parse::<OptLevel>().unwrap(), OptLevel::Default);
        assert_eq!("2".parse::<OptLevel>().unwrap(), OptLevel::Aggressive);
        assert_eq!("s".parse::<OptLevel>().unwrap(), OptLevel::Size);
    }

    #[test]
    fn test_thermal_model_parsing() {
        assert_eq!("7nm".parse::<ThermalModel>().unwrap(), ThermalModel::Arrhenius7nm);
        assert_eq!("5nm".parse::<ThermalModel>().unwrap(), ThermalModel::Arrhenius5nm);
        assert_eq!("none".parse::<ThermalModel>().unwrap(), ThermalModel::None);
        assert_eq!("conservative".parse::<ThermalModel>().unwrap(), ThermalModel::Conservative);
    }

    #[test]
    fn test_backend_availability() {
        // Native should always be available
        assert!(Backend::Native.is_available());
        
        // Others depend on features
        #[cfg(feature = "llvm-base")]
        assert!(Backend::Llvm.is_available());
        
        #[cfg(not(feature = "llvm-base"))]
        assert!(!Backend::Llvm.is_available());
    }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        Backend,
        BackendParseError,
        OutputFormat,
        CompileTarget,
        OptLevel,
        ThermalModel,
        AllocStrategy,
        BuildArgs,
        NativeBackendOptions,
        EmitStage,
        CompileOutput,
        CompileTiming,
        BackendMetrics,
        NativeMetrics,
        extract_build_args,
        compile,
        print_backend_help,
        print_backend_list,
        print_backend_info,
    };
}
