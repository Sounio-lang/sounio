//! # Sounio Compiler: Native Backend
//!
//! This module provides the native x86-64 backend for the Sounio compiler,
//! bypassing LLVM entirely for direct machine code generation with epistemic
//! awareness at every level.
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │                    NATIVE BACKEND PIPELINE                          │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  SIR (Sounio IR)                                                    │
//! │       │                                                             │
//! │       ▼                                                             │
//! │  ┌─────────────────┐                                                │
//! │  │ Metrics Analysis │ ◄── metrics.rs                               │
//! │  │ • Cycle estimation                                               │
//! │  │ • Power estimation                                               │
//! │  │ • Epistemic propagation                                          │
//! │  └────────┬────────┘                                                │
//! │           │                                                         │
//! │           ▼                                                         │
//! │  ┌─────────────────┐                                                │
//! │  │ Thermal Analysis │ ◄── thermal.rs                               │
//! │  │ • Arrhenius degradation                                          │
//! │  │ • Self-heating feedback                                          │
//! │  │ • Confidence degradation                                         │
//! │  └────────┬────────┘                                                │
//! │           │                                                         │
//! │           ▼                                                         │
//! │  ┌─────────────────┐                                                │
//! │  │ Register Alloc  │ ◄── alloc.rs (epistemic-aware)                │
//! │  │ • Confidence-based spilling                                      │
//! │  │ • LiveInterval with metadata                                     │
//! │  └────────┬────────┘                                                │
//! │           │                                                         │
//! │           ▼                                                         │
//! │  ┌─────────────────┐                                                │
//! │  │ Code Emission   │ ◄── emit.rs                                   │
//! │  │ • x86-64 machine code                                            │
//! │  │ • .so generation                                                 │
//! │  └─────────────────┘                                                │
//! │                                                                     │
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Key Features
//!
//! - **No LLVM Dependency**: Direct x86-64 emission without external toolchains
//! - **Epistemic-Aware Allocation**: Register allocation considers confidence metadata
//! - **Thermal Modeling**: Arrhenius-based degradation affects epistemic confidence
//! - **Self-Contained Metrics**: All cycle/power estimates computed internally
//!
//! ## Usage
//!
//! ```rust,ignore
//! use sounio::backend::native::{
//!     metrics::{SirMetricsEstimator, SirBlock},
//!     thermal::{ArrheniusModel, apply_degradation_full},
//! };
//!
//! let estimator = SirMetricsEstimator::default();
//! let block = /* build SIR block */;
//!
//! let (cycles, power) = estimator.estimate_all(&block);
//!
//! // Apply thermal degradation to epistemic values
//! let model = ArrheniusModel::default();
//! let result = apply_degradation_full(0.95, 0.1, &model, cycles.cycles, 350.0);
//! ```
//!
//! ## Module Structure
//!
//! - `metrics`: Cycle and power estimation based on microarchitecture models
//! - `thermal`: Arrhenius degradation and self-heating models
//! - `alloc`: Epistemic-aware register allocation
//! - `emit`: x86-64 machine code emission (TODO: connect)

pub mod metrics;
pub mod thermal;
pub mod alloc;
pub mod elf;
pub mod linker;
pub mod runtime;
pub mod aarch64;
pub mod ode_runtime;
pub mod autodiff_runtime;
pub mod tensor_runtime;
pub mod uncertain_runtime;
pub mod quat_runtime;

// Re-export commonly used types
pub mod prelude {
    pub use super::metrics::prelude::*;
    pub use super::thermal::prelude::*;
    pub use super::alloc::prelude::*;
}

/// Backend configuration
#[derive(Debug, Clone)]
pub struct NativeBackendConfig {
    /// Target architecture
    pub arch: TargetArch,
    /// Cycle model to use
    pub cycle_model: metrics::CycleModel,
    /// Power model to use
    pub power_model: metrics::PowerModel,
    /// Thermal degradation model
    pub thermal_model: thermal::ArrheniusModel,
    /// Whether to enable thermal tracking
    pub enable_thermal_tracking: bool,
    /// Confidence floor (minimum allowed confidence)
    pub confidence_floor: f64,
    /// Optimization level
    pub opt_level: OptLevel,
}

impl Default for NativeBackendConfig {
    fn default() -> Self {
        Self {
            arch: TargetArch::X86_64,
            cycle_model: metrics::CycleModel::default(),
            power_model: metrics::PowerModel::default(),
            thermal_model: thermal::ArrheniusModel::default(),
            enable_thermal_tracking: true,
            confidence_floor: 0.05,
            opt_level: OptLevel::Default,
        }
    }
}

/// Target architecture
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetArch {
    X86_64,
    AArch64,  // Future
    RISCV64,  // Future
}

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimization (fastest compile)
    None,
    /// Default optimization
    Default,
    /// Aggressive optimization (slower compile)
    Aggressive,
    /// Size optimization
    Size,
}

impl Default for OptLevel {
    fn default() -> Self {
        OptLevel::Default
    }
}

/// Native backend entry point
pub struct NativeBackend {
    config: NativeBackendConfig,
    metrics_estimator: metrics::SirMetricsEstimator,
    thermal_state: thermal::ThermalState,
}

impl NativeBackend {
    /// Create with default configuration
    pub fn new() -> Self {
        Self::with_config(NativeBackendConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: NativeBackendConfig) -> Self {
        let metrics_estimator = metrics::SirMetricsEstimator::new(
            config.cycle_model.clone(),
            config.power_model.clone(),
        );

        Self {
            config,
            metrics_estimator,
            thermal_state: thermal::ThermalState::with_history(),
        }
    }

    /// Compile a SIR module to native code
    ///
    /// Returns compiled bytes and compilation metadata
    pub fn compile(&mut self, module: &crate::sir::module::SirModule) -> Result<CompileResult, CompileError> {
        use crate::sir::emit::{CodeEmitter, X86_64Emitter};

        // Step 1: Create x86-64 emitter
        let mut emitter = X86_64Emitter::new();

        // Step 2: Emit code for the module
        let code_segment = emitter
            .emit_module(module)
            .map_err(|e| CompileError::EmissionFailed(format!("{:?}", e)))?;

        // Step 3: Analyze blocks for metrics if thermal tracking enabled
        let total_cycles = 0u64;
        let total_energy = 0.0f64;

        // For now, we don't have block-level metrics from the SirModule
        // The metrics analysis can be added when SirModule exposes its blocks

        // Step 4: Build result
        let symbols: Vec<Symbol> = code_segment
            .symbols
            .iter()
            .map(|s| Symbol {
                name: s.name.clone(),
                offset: s.offset,
                size: 0, // Size not tracked in emit.rs Symbol
            })
            .collect();

        // Calculate thermal degradation if tracking enabled
        let final_confidence = if self.config.enable_thermal_tracking {
            let result = thermal::apply_degradation_full(
                1.0,
                0.0,
                &self.config.thermal_model,
                total_cycles,
                self.thermal_state.current_temp_k,
            );
            result.epsilon  // epsilon is the epistemic confidence
        } else {
            1.0
        };

        Ok(CompileResult {
            code: code_segment.code,
            symbols,
            metrics: CompileMetrics {
                total_cycles,
                total_energy_pj: total_energy,
                final_confidence,
                thermal_degradation: 1.0 - final_confidence,
            },
        })
    }

    /// Compile a SIR module to an ELF object file
    pub fn compile_to_object(&mut self, module: &crate::sir::module::SirModule) -> Result<Vec<u8>, CompileError> {
        use crate::sir::emit::{CodeEmitter, X86_64Emitter};

        let mut emitter = X86_64Emitter::new();
        let code_segment = emitter
            .emit_module(module)
            .map_err(|e| CompileError::EmissionFailed(format!("{:?}", e)))?;

        // Use ELF conversion helper
        let elf_bytes = elf::code_segment_to_elf(&code_segment)
            .map_err(|e| CompileError::EmissionFailed(format!("ELF emission failed: {}", e)))?;

        Ok(elf_bytes)
    }

    /// Analyze a SIR block for metrics without compilation
    pub fn analyze_block(&mut self, block: &metrics::SirBlock) -> BlockAnalysis {
        let (cycles, power) = self.metrics_estimator.estimate_all(block);

        // Update thermal state if tracking enabled
        let thermal_result = if self.config.enable_thermal_tracking {
            self.thermal_state.update(
                &self.config.thermal_model,
                power.temperature_rise_k + 298.0,
                cycles.cycles,
            );

            Some(thermal::apply_degradation_full(
                1.0,  // Base confidence
                0.0,  // Base variance
                &self.config.thermal_model,
                self.thermal_state.accumulated_cycles,
                self.thermal_state.current_temp_k,
            ))
        } else {
            None
        };

        BlockAnalysis {
            cycles,
            power,
            thermal: thermal_result,
        }
    }

    /// Get current thermal state
    pub fn thermal_state(&self) -> &thermal::ThermalState {
        &self.thermal_state
    }

    /// Reset thermal state (e.g., for new compilation unit)
    pub fn reset_thermal_state(&mut self) {
        self.thermal_state.reset();
    }

    /// Get configuration
    pub fn config(&self) -> &NativeBackendConfig {
        &self.config
    }
}

impl Default for NativeBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of block analysis
#[derive(Debug)]
pub struct BlockAnalysis {
    pub cycles: metrics::CycleEstimate,
    pub power: metrics::PowerEstimate,
    pub thermal: Option<thermal::DegradationResult>,
}

/// Placeholder for SIR module type
/// TODO: Import from actual SIR module when available
#[derive(Debug)]
pub struct SirModule {
    pub name: String,
    pub blocks: Vec<metrics::SirBlock>,
}

/// Compilation result
#[derive(Debug)]
pub struct CompileResult {
    /// Compiled machine code
    pub code: Vec<u8>,
    /// Symbol table
    pub symbols: Vec<Symbol>,
    /// Compilation metrics
    pub metrics: CompileMetrics,
}

/// Symbol in compiled output
#[derive(Debug)]
pub struct Symbol {
    pub name: String,
    pub offset: usize,
    pub size: usize,
}

/// Compilation metrics
#[derive(Debug)]
pub struct CompileMetrics {
    pub total_cycles: u64,
    pub total_energy_pj: f64,
    pub final_confidence: f64,
    pub thermal_degradation: f64,
}

/// Compilation error
#[derive(Debug)]
pub enum CompileError {
    NotImplemented(String),
    InvalidSir(String),
    AllocationFailed(String),
    EmissionFailed(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotImplemented(s) => write!(f, "Not implemented: {}", s),
            Self::InvalidSir(s) => write!(f, "Invalid SIR: {}", s),
            Self::AllocationFailed(s) => write!(f, "Register allocation failed: {}", s),
            Self::EmissionFailed(s) => write!(f, "Code emission failed: {}", s),
        }
    }
}

impl std::error::Error for CompileError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use metrics::{OpClass, Operand, SirInstruction, SirBlock};

    fn sample_block() -> SirBlock {
        let mut block = SirBlock::new("test");
        block.push(SirInstruction {
            op_class: OpClass::MemoryLoad,
            sources: vec![Operand::Memory { base: 0, offset: 0, scale: 1 }],
            destination: Some(Operand::Register(0)),
            epistemic: None,
        });
        block.push(SirInstruction {
            op_class: OpClass::FloatArithmetic,
            sources: vec![Operand::Register(0), Operand::Register(1)],
            destination: Some(Operand::Register(2)),
            epistemic: None,
        });
        block
    }

    #[test]
    fn test_backend_creation() {
        let backend = NativeBackend::new();
        assert_eq!(backend.config.arch, TargetArch::X86_64);
        assert!(backend.config.enable_thermal_tracking);
    }

    #[test]
    fn test_block_analysis() {
        let mut backend = NativeBackend::new();
        let block = sample_block();
        
        let analysis = backend.analyze_block(&block);
        
        assert!(analysis.cycles.cycles > 0);
        assert!(analysis.power.average_power_uw > 0.0);
        assert!(analysis.thermal.is_some());
    }

    #[test]
    fn test_thermal_accumulation() {
        let mut backend = NativeBackend::new();
        let block = sample_block();
        
        // Multiple analyses should accumulate thermal state
        for _ in 0..10 {
            backend.analyze_block(&block);
        }
        
        assert!(backend.thermal_state().accumulated_cycles > 0);
        assert!(backend.thermal_state().accumulated_degradation > 0.0);
    }
}
