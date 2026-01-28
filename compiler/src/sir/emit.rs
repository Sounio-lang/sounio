//! Machine Code Emission
//!
//! Translates SIR to native machine code. This is the final stage of
//! the Sounio Native Backend.
//!
//! # Architecture Support
//!
//! - **x86-64**: Primary target (Linux, macOS, Windows)
//! - **AArch64**: Secondary target (macOS M1/M2, Linux ARM)
//! - **RISC-V**: Future target
//!
//! # Code Generation Strategy
//!
//! 1. **Instruction Selection**: SIR ops → machine instructions
//! 2. **Register Allocation**: Virtual registers → physical registers
//! 3. **Instruction Scheduling**: Reorder for pipeline efficiency
//! 4. **Code Emission**: Encode instructions to bytes

use super::alloc_policy::{
    AllocPolicy, EpistemicMetadata, MetricsCollector,
    SpillEvent, SpillReason, compute_attention_score,
};

// Re-export alloc_policy types for external use
pub use super::alloc_policy::{AttentionConfig, AllocationMetrics};

use super::blocks::{SirFunction, Terminator};
use super::module::{Architecture, SirModule};
use super::ops::*;
use super::types::{ScalarType, SirType};
use super::values::BlockId;
use super::values::{Constant, ValueId};
use std::collections::{HashMap, HashSet};

/// Emitted code segment
#[derive(Debug, Clone)]
pub struct CodeSegment {
    /// Machine code bytes
    pub code: Vec<u8>,
    /// Relocations to apply
    pub relocations: Vec<Relocation>,
    /// Symbol definitions
    pub symbols: Vec<Symbol>,
}

/// A relocation entry
#[derive(Debug, Clone)]
pub struct Relocation {
    /// Offset in code where relocation applies
    pub offset: usize,
    /// Kind of relocation
    pub kind: RelocKind,
    /// Symbol being referenced
    pub symbol: String,
    /// Addend
    pub addend: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelocKind {
    /// Absolute 64-bit address
    Abs64,
    /// PC-relative 32-bit
    PCRel32,
    /// PLT entry
    PLT32,
    /// GOT entry
    GOT32,
}

/// A symbol definition
#[derive(Debug, Clone)]
pub struct Symbol {
    /// Symbol name
    pub name: String,
    /// Offset in code
    pub offset: usize,
    /// Is externally visible?
    pub global: bool,
}

#[derive(Debug, Clone)]
struct RoDataEntry {
    func_name: String,
    data_name: String,
}

#[derive(Clone)]
struct PhiMove {
    dst: ValueId,
    src: ValueId,
}

/// Code emitter trait
pub trait CodeEmitter {
    /// Emit code for a module
    fn emit_module(&mut self, module: &SirModule) -> Result<CodeSegment, EmitError>;

    /// Emit code for a function
    fn emit_function(&mut self, func: &SirFunction) -> Result<Vec<u8>, EmitError>;
}

/// Code emission error
#[derive(Debug)]
pub enum EmitError {
    /// Unsupported instruction
    UnsupportedInstruction(String),
    /// Unsupported target
    UnsupportedTarget(String),
    /// Register allocation failed
    RegisterAllocationFailed,
    /// Too many spills
    TooManySpills,
    /// Invalid instruction encoding
    InvalidEncoding(String),
    /// I/O error during code emission
    IoError(String),
    /// Unit mismatch in operation
    UnitMismatch {
        op: String,
        lhs_unit: String,
        rhs_unit: String,
    },
    /// Invalid input (e.g., invalid register class, dimension, etc.)
    InvalidInput(String),
}

// ============================================================================
// X86-64 CODE GENERATOR
// ============================================================================

/// X86-64 registers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum X86Reg {
    RAX = 0,
    RCX = 1,
    RDX = 2,
    RBX = 3,
    RSP = 4,
    RBP = 5,
    RSI = 6,
    RDI = 7,
    R8 = 8,
    R9 = 9,
    R10 = 10,
    R11 = 11,
    R12 = 12,
    R13 = 13,
    R14 = 14,
    R15 = 15,

    // XMM registers for floating point
    XMM0 = 16,
    XMM1 = 17,
    XMM2 = 18,
    XMM3 = 19,
    XMM4 = 20,
    XMM5 = 21,
    XMM6 = 22,
    XMM7 = 23,
    XMM8 = 24,
    XMM9 = 25,
    XMM10 = 26,
    XMM11 = 27,
    XMM12 = 28,
    XMM13 = 29,
    XMM14 = 30,
    XMM15 = 31,
}

impl X86Reg {
    /// Is this a callee-saved register?
    pub fn is_callee_saved(&self) -> bool {
        matches!(
            self,
            X86Reg::RBX | X86Reg::RBP | X86Reg::R12 | X86Reg::R13 | X86Reg::R14 | X86Reg::R15
        )
    }

    /// Is this an XMM register?
    pub fn is_xmm(&self) -> bool {
        (*self as u8) >= 16
    }

    /// Get the register encoding (0-7, 8-15)
    pub fn encoding(&self) -> u8 {
        (*self as u8) & 0x7
    }

    /// Does this register need REX.B prefix?
    pub fn needs_rex(&self) -> bool {
        (*self as u8) >= 8 && (*self as u8) < 16
    }

    /// Allocatable integer registers (System V ABI)
    pub fn allocatable_int() -> &'static [X86Reg] {
        &[
            X86Reg::RAX,
            X86Reg::RCX,
            X86Reg::RDX,
            X86Reg::RSI,
            X86Reg::RDI,
            X86Reg::R8,
            X86Reg::R9,
            X86Reg::R10,
            X86Reg::R11,
            X86Reg::RBX,
            X86Reg::R12,
            X86Reg::R13,
            X86Reg::R14,
            X86Reg::R15,
        ]
    }

    /// Allocatable XMM registers
    pub fn allocatable_xmm() -> &'static [X86Reg] {
        &[
            X86Reg::XMM0,
            X86Reg::XMM1,
            X86Reg::XMM2,
            X86Reg::XMM3,
            X86Reg::XMM4,
            X86Reg::XMM5,
            X86Reg::XMM6,
            X86Reg::XMM7,
            X86Reg::XMM8,
            X86Reg::XMM9,
            X86Reg::XMM10,
            X86Reg::XMM11,
            X86Reg::XMM12,
            X86Reg::XMM13,
            X86Reg::XMM14,
            X86Reg::XMM15,
        ]
    }

    /// Argument registers (System V AMD64 ABI)
    pub fn arg_regs() -> &'static [X86Reg] {
        &[
            X86Reg::RDI,
            X86Reg::RSI,
            X86Reg::RDX,
            X86Reg::RCX,
            X86Reg::R8,
            X86Reg::R9,
        ]
    }

    /// Floating-point argument registers
    pub fn arg_xmm_regs() -> &'static [X86Reg] {
        &[
            X86Reg::XMM0,
            X86Reg::XMM1,
            X86Reg::XMM2,
            X86Reg::XMM3,
            X86Reg::XMM4,
            X86Reg::XMM5,
            X86Reg::XMM6,
            X86Reg::XMM7,
        ]
    }

    /// Callee-saved integer registers (System V AMD64 ABI)
    /// These must be preserved across function calls
    pub fn callee_saved_int() -> &'static [X86Reg] {
        &[
            X86Reg::RBX,
            X86Reg::R12,
            X86Reg::R13,
            X86Reg::R14,
            X86Reg::R15,
        ]
    }

    /// Caller-saved integer registers (scratch registers)
    /// These can be freely modified by called functions
    pub fn caller_saved_int() -> &'static [X86Reg] {
        &[
            X86Reg::RAX,
            X86Reg::RCX,
            X86Reg::RDX,
            X86Reg::RSI,
            X86Reg::RDI,
            X86Reg::R8,
            X86Reg::R9,
            X86Reg::R10,
            X86Reg::R11,
        ]
    }
}

// ============================================================================
// LIVE INTERVAL AND REGISTER ALLOCATION
// ============================================================================
//
// Note: AttentionConfig and AllocationMetrics are now imported from alloc_policy module

/// A live interval representing when a value is live in the program
///
/// In linear scan register allocation, we track the lifetime of each
/// SSA value as an interval [start, end). The allocator assigns physical
/// registers to non-overlapping intervals.
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// The SSA value this interval represents
    pub value: ValueId,
    /// Instruction index where this value is first defined
    pub start: usize,
    /// Instruction index where this value is last used (exclusive)
    pub end: usize,
    /// The type of the value (determines int vs float register)
    pub ty: SirType,
    /// Assigned physical register (None if spilled)
    pub reg: Option<X86Reg>,
    /// Stack spill slot offset from RBP (negative, e.g., -8, -16)
    pub spill_slot: Option<i32>,
    /// Whether this value is used across a call (must be callee-saved or spilled)
    pub crosses_call: bool,

    // === USE TRACKING (for attention score) ===
    /// Instruction indices where this value is used
    pub use_positions: Vec<usize>,

    // === EPISTEMIC-AWARE FIELDS ===
    // These fields enable confidence-aware register allocation - the first
    // compiler in history to optimize for TRUST, not just speed.

    /// Maximum confidence bound [0.0, 1.0] for this value (τ).
    /// 1.0 = fully certain (constants, proven values)
    /// 0.0 = completely unknown (untracked external data)
    /// Values with high confidence are preferentially kept in registers.
    pub max_confidence: f64,

    /// Uncertainty bound [0.0, 1.0] for this value (σ).
    /// Opposite of confidence but tracked separately for GUM propagation.
    /// High uncertainty values are better candidates for spilling.
    pub uncertainty: f64,

    /// Whether this value has tracked provenance (data lineage, ρ).
    /// Provenanced values are prioritized for register allocation
    /// because they represent scientifically rigorous computations.
    pub has_provenance: bool,

    /// Combined epistemic weight for allocation decisions.
    /// Higher weight = more important to keep in registers.
    /// Computed as: confidence * (1.0 + provenance_bonus)
    pub epistemic_weight: f64,
}

impl LiveInterval {
    /// Create a new live interval starting at the given instruction index
    pub fn new(value: ValueId, start: usize, ty: SirType) -> Self {
        Self {
            value,
            start,
            end: start + 1, // Minimum interval length of 1
            ty,
            reg: None,
            spill_slot: None,
            crosses_call: false,
            use_positions: Vec::new(),
            // Epistemic fields - default to uncertain until proven otherwise
            max_confidence: 0.5,
            uncertainty: 0.5,
            has_provenance: false,
            epistemic_weight: 0.5,
        }
    }

    /// Create an interval with known epistemic properties
    pub fn with_epistemic(
        value: ValueId,
        start: usize,
        ty: SirType,
        confidence: f64,
        has_provenance: bool,
    ) -> Self {
        let provenance_bonus = if has_provenance { 0.2 } else { 0.0 };
        let epistemic_weight = confidence * (1.0 + provenance_bonus);

        Self {
            value,
            start,
            end: start + 1,
            ty,
            reg: None,
            spill_slot: None,
            crosses_call: false,
            use_positions: Vec::new(),
            max_confidence: confidence,
            uncertainty: 1.0 - confidence, // Default uncertainty is inverse of confidence
            has_provenance,
            epistemic_weight,
        }
    }

    /// Create an interval for a fully certain value (constants, proven refinements)
    pub fn certain(value: ValueId, start: usize, ty: SirType) -> Self {
        Self::with_epistemic(value, start, ty, 1.0, true)
    }

    /// Update epistemic properties based on instruction metadata
    pub fn set_epistemic(&mut self, confidence: f64, has_provenance: bool) {
        self.max_confidence = confidence;
        self.uncertainty = 1.0 - confidence;
        self.has_provenance = has_provenance;
        let provenance_bonus = if has_provenance { 0.2 } else { 0.0 };
        self.epistemic_weight = confidence * (1.0 + provenance_bonus);
    }

    /// Set explicit uncertainty (when different from 1 - confidence)
    pub fn set_uncertainty(&mut self, uncertainty: f64) {
        self.uncertainty = uncertainty;
    }

    /// Record a use of this value at the given instruction index
    pub fn record_use(&mut self, idx: usize) {
        self.use_positions.push(idx);
        self.extend_to(idx);
    }

    /// Extend the live range to include the given instruction index
    pub fn extend_to(&mut self, idx: usize) {
        self.end = self.end.max(idx + 1);
    }

    /// Calculate use density: number of uses per unit of lifetime
    pub fn use_density(&self) -> f64 {
        let lifetime = (self.end - self.start).max(1) as f64;
        self.use_positions.len() as f64 / lifetime
    }

    /// Calculate the attention score for this interval.
    ///
    /// This unified score combines classical register allocation heuristics
    /// with epistemic (confidence-aware) scoring.
    ///
    /// Higher score = more important to keep in register
    /// Lower score = better candidate for spilling
    ///
    /// Formula:
    /// score = w1 * use_density + w2 * crosses_call  // classical
    ///       + w3 * τ - w4 * σ + w5 * ρ              // epistemic
    pub fn attention_score(&self, config: &AttentionConfig) -> f64 {
        // Classical component
        let classic = config.w_use_density * self.use_density()
            + config.w_crosses_call * (if self.crosses_call { 1.0 } else { 0.0 });

        // Epistemic component
        let tau = self.max_confidence;
        let sigma = self.uncertainty;
        let rho = if self.has_provenance { 1.0 } else { 0.0 };

        let epistemic = config.w_confidence * tau
            - config.w_uncertainty * sigma
            + config.w_provenance * rho;

        classic + epistemic
    }

    /// Check if this interval overlaps with another
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Check if the interval is still live at the given position
    pub fn is_live_at(&self, pos: usize) -> bool {
        self.start <= pos && pos < self.end
    }

    /// Returns true if this value needs a floating-point register
    pub fn needs_float_reg(&self) -> bool {
        self.ty.is_float()
    }

    /// Returns true if this interval has been assigned a register
    pub fn has_register(&self) -> bool {
        self.reg.is_some()
    }

    /// Returns true if this interval has been spilled to the stack
    pub fn is_spilled(&self) -> bool {
        self.spill_slot.is_some()
    }
}

/// Linear scan register allocator
///
/// Implements the classic linear scan algorithm from Poletto & Sarkar (1999):
/// 1. Compute live intervals for all values
/// 2. Sort intervals by start position
/// 3. Walk through intervals in order, assigning registers greedily
/// 4. When registers are exhausted, spill the interval ending furthest in the future
///
/// # System V AMD64 ABI Compliance
///
/// - **Integer arguments**: RDI, RSI, RDX, RCX, R8, R9
/// - **Float arguments**: XMM0-XMM7
/// - **Return values**: RAX (integer), XMM0 (float)
/// - **Callee-saved**: RBX, RBP, R12-R15
/// - **Caller-saved**: RAX, RCX, RDX, RSI, RDI, R8-R11, XMM0-XMM15
#[derive(Debug)]
pub struct RegisterAllocator {
    /// All live intervals, sorted by start position
    pub intervals: Vec<LiveInterval>,
    /// Indices of currently active intervals (allocated and not yet expired)
    active: Vec<usize>,
    /// Free integer registers available for allocation
    free_int_regs: Vec<X86Reg>,
    /// Free XMM registers available for allocation
    free_xmm_regs: Vec<X86Reg>,
    /// Current stack frame size for spills (grows downward from RBP)
    pub stack_size: i32,
    /// Mapping from ValueId to interval index for quick lookup
    value_to_interval: HashMap<ValueId, usize>,
    /// Set of callee-saved registers that we used (need to save/restore in prologue/epilogue)
    pub used_callee_saved: HashSet<X86Reg>,
    /// Instruction positions of call instructions (for crosses_call analysis)
    call_positions: Vec<usize>,

    // === A/B POLICY COMPARISON FIELDS ===
    /// Register allocation policy (Classic, Attention, AttentionCalibrated)
    policy: AllocPolicy,
    /// Attention weight configuration (only used when policy is Attention*)
    attention_config: AttentionConfig,
    /// Metrics collector for A/B comparison
    pub metrics: MetricsCollector,
}

impl RegisterAllocator {
    /// Create a new register allocator with default Attention policy
    pub fn new() -> Self {
        Self::with_policy(AllocPolicy::default(), AttentionConfig::default())
    }

    /// Create a register allocator with specified policy and config
    pub fn with_policy(policy: AllocPolicy, config: AttentionConfig) -> Self {
        let attention_config = match policy {
            AllocPolicy::AttentionCalibrated => AttentionConfig::calibrated(),
            _ => config,
        };

        Self {
            intervals: Vec::new(),
            active: Vec::new(),
            free_int_regs: X86Reg::allocatable_int().to_vec(),
            free_xmm_regs: X86Reg::allocatable_xmm().to_vec(),
            stack_size: 0,
            value_to_interval: HashMap::new(),
            used_callee_saved: HashSet::new(),
            call_positions: Vec::new(),
            policy,
            attention_config,
            metrics: MetricsCollector::new(policy, true),
        }
    }

    /// Get the current allocation policy
    pub fn policy(&self) -> AllocPolicy {
        self.policy
    }

    /// Get the attention configuration
    pub fn attention_config(&self) -> &AttentionConfig {
        &self.attention_config
    }

    /// Reset the allocator for a new function
    pub fn reset(&mut self) {
        self.intervals.clear();
        self.active.clear();
        self.free_int_regs = X86Reg::allocatable_int().to_vec();
        self.free_xmm_regs = X86Reg::allocatable_xmm().to_vec();
        self.stack_size = 0;
        self.value_to_interval.clear();
        self.used_callee_saved.clear();
        self.call_positions.clear();
        self.metrics.reset();
    }

    /// Main entry point: allocate registers for a function
    ///
    /// This performs the complete linear scan algorithm:
    /// 1. Compute live intervals from the SIR function
    /// 2. Sort intervals by start position
    /// 3. Perform linear scan allocation with spilling
    pub fn allocate_registers(&mut self, func: &SirFunction) -> Result<(), EmitError> {
        self.reset();

        // Step 1: Compute live intervals
        self.compute_live_intervals(func)?;

        // Step 2: Sort intervals by start position
        self.intervals.sort_by_key(|iv| iv.start);

        // Rebuild value_to_interval mapping after sorting
        self.value_to_interval.clear();
        for (idx, iv) in self.intervals.iter().enumerate() {
            self.value_to_interval.insert(iv.value, idx);
        }

        // Step 3: Linear scan allocation
        // Debug: print intervals before allocation
        if std::env::var("SOUNIO_DEBUG_ALLOC").is_ok() {
            eprintln!("=== Live Intervals (sorted by start) ===");
            for (idx, iv) in self.intervals.iter().enumerate() {
                eprintln!("  [{}] v{}: [{}, {}) crosses_call={} uses={:?}",
                    idx, iv.value.0, iv.start, iv.end, iv.crosses_call, iv.use_positions);
            }
        }

        self.linear_scan()?;

        // Debug: print allocation results
        if std::env::var("SOUNIO_DEBUG_ALLOC").is_ok() {
            eprintln!("=== Allocation Results ===");
            for iv in &self.intervals {
                eprintln!("  v{}: reg={:?} spill={:?}",
                    iv.value.0, iv.reg, iv.spill_slot);
            }
        }

        Ok(())
    }

    /// Compute live intervals for all values in the function
    ///
    /// This performs a single pass over the function, recording:
    /// - Definition points for each value
    /// - Use points for each value
    /// - Call instruction positions (for crosses_call detection)
    fn compute_live_intervals(&mut self, func: &SirFunction) -> Result<(), EmitError> {
        let mut inst_idx: usize = 0;
        let mut value_defs: HashMap<ValueId, (usize, SirType)> = HashMap::new();
        let mut value_uses: HashMap<ValueId, Vec<usize>> = HashMap::new();
        let mut alloca_values: HashSet<ValueId> = HashSet::new();
        let mut block_indices: HashMap<BlockId, usize> = HashMap::new();
        let mut block_ranges: HashMap<BlockId, (usize, usize)> = HashMap::new();
        let mut backedge_targets: HashSet<BlockId> = HashSet::new();
        let mut has_backedge = false;

        for (block_idx, block) in func.blocks.iter().enumerate() {
            block_indices.insert(block.id, block_idx);
        }

        // Handle function parameters - they are live from instruction 0
        // System V AMD64 ABI: integers in RDI, RSI, RDX, RCX, R8, R9
        // Floats in XMM0-XMM7
        for (param_idx, (_, param_ty)) in func.params.iter().enumerate() {
            let value_id = ValueId::new(param_idx as u32);
            value_defs.insert(value_id, (0, param_ty.clone()));
            value_uses.entry(value_id).or_default();
        }

        // Process each basic block
        for (block_idx, block) in func.blocks.iter().enumerate() {
            let block_start = inst_idx;
            // Block parameters (from phi elimination) are defined at block entry
            for param in &block.params {
                value_defs.insert(param.id, (inst_idx, param.ty.clone()));
            }

            // Process each instruction in the block
            for inst in &block.instructions {
                // Record definition if this instruction produces a value
                if let Some(result) = inst.result {
                    let ty = self.infer_instruction_type(&inst.inst, func);
                    value_defs.insert(result, (inst_idx, ty));
                    if matches!(inst.inst, SirInst::Memory(MemoryOp::Alloca { .. })) {
                        alloca_values.insert(result);
                    }
                }

                // Record uses of operand values
                for operand in inst.inst.operands() {
                    value_uses.entry(operand).or_default().push(inst_idx);
                }

                // Track call positions for crosses_call analysis
                if matches!(inst.inst, SirInst::Call(_)) {
                    self.call_positions.push(inst_idx);
                }

                inst_idx += 1;
            }

            // Handle terminator operands
            if let Some(term) = &block.terminator {
                for operand in term.operands() {
                    value_uses.entry(operand).or_default().push(inst_idx);
                }
                for succ in term.successors() {
                    if let Some(&succ_idx) = block_indices.get(&succ) {
                        if succ_idx <= block_idx {
                            has_backedge = true;
                            backedge_targets.insert(succ);
                        }
                    }
                }
                inst_idx += 1;
            }
            block_ranges.insert(block.id, (block_start, inst_idx));
        }
        let function_end = inst_idx;

        // Build live intervals from definitions and uses
        for (value_id, (def_pos, ty)) in value_defs {
            let mut interval = LiveInterval::new(value_id, def_pos, ty);

            // Extend interval to cover all uses
            if let Some(uses) = value_uses.get(&value_id) {
                for &use_pos in uses {
                    interval.extend_to(use_pos);
                }
            }

            // Check if this interval crosses any call instructions
            // If so, we must either use a callee-saved register or spill
            for &call_pos in &self.call_positions {
                if interval.start <= call_pos && call_pos < interval.end {
                    interval.crosses_call = true;
                    break;
                }
            }

            let idx = self.intervals.len();
            self.value_to_interval.insert(value_id, idx);
            self.intervals.push(interval);
        }

        if !backedge_targets.is_empty() {
            let mut header_ranges: Vec<(usize, usize)> = Vec::new();
            for header in &backedge_targets {
                if let Some(&(start, end)) = block_ranges.get(header) {
                    header_ranges.push((start, end));
                }
            }

            for interval in &mut self.intervals {
                if let Some(uses) = value_uses.get(&interval.value) {
                    let used_in_header = uses.iter().any(|&pos| {
                        header_ranges
                            .iter()
                            .any(|(start, end)| pos >= *start && pos < *end)
                    });
                    if used_in_header {
                        interval.end = interval.end.max(function_end);
                    }
                }
            }
        }

        // Conservatively keep alloca pointers in callee-saved regs or spills.
        if !alloca_values.is_empty() {
            for value_id in alloca_values {
                if let Some(&idx) = self.value_to_interval.get(&value_id) {
                    if let Some(interval) = self.intervals.get_mut(idx) {
                        interval.crosses_call = true;
                        if has_backedge {
                            interval.end = interval.end.max(function_end);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Infer the type of an instruction's result
    fn infer_instruction_type(&self, inst: &SirInst, _func: &SirFunction) -> SirType {
        match inst {
            SirInst::BinOp { op, .. } => match op {
                ArithOp::FAdd | ArithOp::FSub | ArithOp::FMul | ArithOp::FDiv | ArithOp::FRem => {
                    SirType::f64()
                }
                _ => SirType::i64(),
            },
            SirInst::Cmp { .. } => SirType::bool(),
            SirInst::Cast { to_ty, .. } => to_ty.clone(),
            SirInst::UnaryFloat { .. } | SirInst::BinaryFloat { .. } => SirType::f64(),
            SirInst::Const(c) => c.ty(),
            SirInst::Memory(MemoryOp::Load { ty, .. }) => ty.clone(),
            SirInst::Memory(MemoryOp::Alloca { ty, .. }) => SirType::ptr(ty.clone()),
            SirInst::Memory(MemoryOp::GetElementPtr { ty, .. }) => SirType::ptr(ty.clone()),
            SirInst::Call(info) => info.ret_ty.clone(),
            SirInst::Phi { ty, .. } => ty.clone(),
            SirInst::BuildAggregate { ty, .. } => ty.clone(),
            SirInst::Select { .. } => SirType::i64(), // Would need context for precise type
            _ => SirType::i64(),                      // Default fallback
        }
    }

    /// EPISTEMIC-AWARE: Extract confidence and provenance from an instruction
    ///
    /// This is the breakthrough: analyzing each instruction to determine its
    /// epistemic properties. The compiler becomes an epistemic guardian.
    ///
    /// Returns: (confidence, has_provenance)
    /// - confidence: [0.0, 1.0] - how certain is this value?
    /// - has_provenance: whether the value has tracked data lineage
    fn extract_epistemic_metadata(&self, inst: &SirInst) -> (f64, bool) {
        match inst {
            // Constants are FULLY CERTAIN - they're compile-time known
            SirInst::Const(c) => {
                match c {
                    // Epistemic constants already have confidence
                    Constant::Epistemic { confidence, .. } => (*confidence, true),
                    // Regular constants are fully certain
                    _ => (1.0, true),
                }
            }

            // EPISTEMIC OPERATIONS - these carry explicit uncertainty
            SirInst::Epistemic(op) => match op {
                // Creating an epistemic value - we know the confidence
                EpistemicOp::Create { .. } => {
                    // The actual confidence is a runtime value, but we know
                    // this operation produces tracked epistemic data
                    (0.8, true) // Conservative estimate, fully provenanced
                }
                // Extracting value from epistemic - loses some confidence info
                EpistemicOp::ExtractValue(_) => (0.7, true),
                // Extracting confidence itself - fully certain about the confidence
                EpistemicOp::ExtractConfidence(_) => (1.0, true),
                // Propagation operations - maintain tracking
                EpistemicOp::PropagateAdd { .. }
                | EpistemicOp::PropagateSub { .. }
                | EpistemicOp::PropagateMul { .. }
                | EpistemicOp::PropagateDiv { .. } => (0.75, true),
                // Fused multiplication - highly optimized, maintains tracking
                EpistemicOp::FusedMul { .. } => (0.8, true),
                // Meet/Join operations (lattice operations)
                EpistemicOp::Meet { .. } | EpistemicOp::Join { .. } => (0.7, true),
                // Any other epistemic operations
                _ => (0.75, true),
            },

            // PROBABILITY OPERATIONS - inherently uncertain but tracked
            SirInst::Prob(op) => match op {
                // Creating a distribution - we know what we're modeling
                ProbOp::CreateDist { .. } => (0.9, true),
                // Sampling - introduces randomness, but tracked
                ProbOp::Sample { .. } | ProbOp::SampleN { .. } => (0.6, true),
                // PDF/CDF computations - mathematical, certain given inputs
                ProbOp::Pdf { .. } | ProbOp::LogPdf { .. } | ProbOp::Cdf { .. } => (0.95, true),
                // Quantile function
                ProbOp::Quantile { .. } => (0.95, true),
                // Distribution combining
                ProbOp::Combine { .. } => (0.85, true),
            },

            // SCIENTIFIC OPERATIONS - domain-specific, high confidence
            SirInst::Scientific(op) => match op {
                // ODE step - numerical, but well-understood error bounds
                ScientificOp::OdeStep { .. } => (0.85, true),
                // Matrix operations - exact computations
                ScientificOp::MatVecMul { .. } | ScientificOp::DotProduct { .. } => (1.0, true),
                // Compartment models - domain-specific
                ScientificOp::CompartmentStep { .. } => (0.9, true),
                // Lerp - linear interpolation, exact given inputs
                ScientificOp::Lerp { .. } => (1.0, true),
                // Autodiff - symbolic differentiation, high confidence
                ScientificOp::Autodiff { .. } => (0.9, true),
            },

            // MEMORY LOADS - external data, LOW confidence unless proven
            SirInst::Memory(MemoryOp::Load { .. }) => {
                // Data from memory could be from anywhere - external files,
                // user input, sensors. LOW confidence by default.
                (0.3, false)
            }

            // CALLS TO EXTERNAL FUNCTIONS - unknown provenance
            SirInst::Call(info) => {
                match &info.callee {
                    // Named calls might be known functions
                    Callee::Named(name) if name.starts_with("sounio_") => (0.8, true),
                    Callee::Named(_) => (0.5, false),
                    // Direct calls to known functions
                    Callee::Direct(_) => (0.7, true),
                    // Indirect calls - no idea what we're calling
                    Callee::Indirect(_) => (0.2, false),
                }
            }

            // PURE ARITHMETIC - maintains input confidence
            // These operations don't introduce new uncertainty
            SirInst::BinOp { .. }
            | SirInst::UnaryFloat { .. }
            | SirInst::BinaryFloat { .. }
            | SirInst::Cmp { .. } => (0.8, false),

            // PHI NODES - merge paths with potentially different confidences
            SirInst::Phi { .. } => (0.5, false), // Conservative

            // CASTS - maintain confidence through type changes
            SirInst::Cast { .. } => (0.9, true),

            // DEFAULT - unknown operations get moderate confidence
            _ => (0.5, false),
        }
    }

    /// Perform linear scan register allocation
    ///
    /// For each interval (in order of start position):
    /// 1. Expire intervals that have ended
    /// 2. Try to allocate a free register
    /// 3. If no register available, spill
    fn linear_scan(&mut self) -> Result<(), EmitError> {
        for i in 0..self.intervals.len() {
            let start = self.intervals[i].start;

            // Expire old intervals (those that end before current starts)
            self.expire_old_intervals(start);

            // Determine register class needed
            let needs_float = self.intervals[i].needs_float_reg();
            let crosses_call = self.intervals[i].crosses_call;
            let epistemic_weight = self.intervals[i].epistemic_weight;

            // Try to allocate a register from the appropriate pool
            // EPISTEMIC-AWARE: high-confidence values get preferential treatment
            let reg = if needs_float {
                self.allocate_xmm_reg(crosses_call)
            } else {
                self.allocate_int_reg_with_confidence(crosses_call, epistemic_weight)
            };

            if let Some(r) = reg {
                // Successfully allocated a register
                self.intervals[i].reg = Some(r);
                self.active.push(i);

                // Keep active list sorted by end position for efficient expiry
                self.active.sort_by_key(|&idx| self.intervals[idx].end);

                // Track if we used a callee-saved register
                if r.is_callee_saved() {
                    self.used_callee_saved.insert(r);
                }
            } else {
                // No free register - need to spill
                self.spill_at_interval(i)?;
            }
        }

        Ok(())
    }

    /// Remove intervals that have expired (ended before the given position)
    fn expire_old_intervals(&mut self, current_pos: usize) {
        let mut expired = Vec::new();

        for (active_idx, &interval_idx) in self.active.iter().enumerate() {
            let interval = &self.intervals[interval_idx];
            if interval.end <= current_pos {
                // This interval has expired - free its register
                if let Some(reg) = interval.reg {
                    if reg.is_xmm() {
                        self.free_xmm_regs.push(reg);
                    } else {
                        self.free_int_regs.push(reg);
                    }
                }
                expired.push(active_idx);
            }
        }

        // Remove expired intervals from active list (in reverse order to preserve indices)
        for idx in expired.into_iter().rev() {
            self.active.remove(idx);
        }
    }

    /// Allocate an integer register - EPISTEMIC-AWARE
    ///
    /// Allocation strategy (in order of priority):
    /// 1. If the value crosses a call, prefer callee-saved registers (RBX, R12-R15)
    ///    to avoid spilling around the call
    /// 2. HIGH-CONFIDENCE values (epistemic_weight > 0.8) also prefer callee-saved
    ///    registers to ensure they stay register-resident through complex computations
    /// 3. Otherwise, prefer caller-saved (scratch) registers to minimize
    ///    callee-saved register saves in prologue/epilogue
    ///
    /// This ensures that scientifically rigorous, high-confidence computations
    /// execute with maximum register efficiency.
    fn allocate_int_reg(&mut self, crosses_call: bool) -> Option<X86Reg> {
        self.allocate_int_reg_with_confidence(crosses_call, 0.5) // Default confidence
    }

    /// Epistemic-aware integer register allocation
    fn allocate_int_reg_with_confidence(
        &mut self,
        crosses_call: bool,
        epistemic_weight: f64,
    ) -> Option<X86Reg> {
        if self.free_int_regs.is_empty() {
            return None;
        }

        // HIGH-CONFIDENCE or CROSSES-CALL: prefer callee-saved registers
        // These values represent trusted computations that should stay in registers
        let prefer_callee_saved = crosses_call || epistemic_weight > 0.8;

        if prefer_callee_saved {
            // Value lives across a call OR is high-confidence - prefer callee-saved
            for (idx, &reg) in self.free_int_regs.iter().enumerate() {
                if reg.is_callee_saved() {
                    return Some(self.free_int_regs.remove(idx));
                }
            }
        }

        // Prefer caller-saved (scratch) registers when possible
        for (idx, &reg) in self.free_int_regs.iter().enumerate() {
            if !reg.is_callee_saved() {
                return Some(self.free_int_regs.remove(idx));
            }
        }

        // Take any available register
        Some(self.free_int_regs.remove(0))
    }

    /// Allocate an XMM register
    ///
    /// Note: All XMM registers are caller-saved in System V ABI.
    /// If a value crosses a call, it will need to be spilled regardless.
    fn allocate_xmm_reg(&mut self, _crosses_call: bool) -> Option<X86Reg> {
        self.free_xmm_regs.pop()
    }

    /// Handle register spilling - POLICY-AWARE
    ///
    /// Supports multiple allocation policies:
    /// - **Classic**: Furthest next use heuristic (traditional)
    /// - **Attention**: Epistemic-aware scoring with configurable weights
    /// - **AttentionCalibrated**: Attention with BO-optimized weights
    ///
    /// The Attention policy is the breakthrough: the first compiler in history
    /// that makes spill decisions based on TRUST, not just lifetime.
    ///
    /// The result: Code paths with strong epistemic backing execute faster
    /// (more register-resident). Dubious data is naturally penalized.
    /// The machine itself rewards scientific rigor.
    fn spill_at_interval(&mut self, current: usize) -> Result<(), EmitError> {
        if self.active.is_empty() {
            // No active intervals - must spill the current interval
            self.record_spill_event(current, SpillReason::RegisterPressure);
            return self.spill_interval(current);
        }

        let spill_candidate = match self.policy {
            AllocPolicy::Classic => self.select_spill_classic(current),
            AllocPolicy::Attention | AllocPolicy::AttentionCalibrated => {
                self.select_spill_attention(current)
            }
        };

        if spill_candidate != current {
            // Spill an active interval and give its register to current
            let reg = self.intervals[spill_candidate]
                .reg
                .expect("Active interval must have a register");

            // Transfer register to current interval
            self.intervals[current].reg = Some(reg);

            // Record the spill event
            self.record_spill_event(spill_candidate, SpillReason::RegisterPressure);

            // Spill the candidate to stack
            self.spill_interval(spill_candidate)?;

            // Update active list
            self.active.retain(|&idx| idx != spill_candidate);
            self.active.push(current);
            self.active.sort_by_key(|&idx| self.intervals[idx].end);

            if reg.is_callee_saved() {
                self.used_callee_saved.insert(reg);
            }
        } else {
            // Spill current interval
            self.record_spill_event(current, SpillReason::RegisterPressure);
            self.spill_interval(current)?;
        }

        Ok(())
    }

    /// Classic spill selection: furthest next use heuristic
    fn select_spill_classic(&self, current: usize) -> usize {
        let current_end = self.intervals[current].end;

        // Find the interval with the furthest end point
        let mut best_spill_idx = current;
        let mut furthest_end = current_end;

        for &active_idx in &self.active {
            let interval = &self.intervals[active_idx];
            if interval.end > furthest_end {
                furthest_end = interval.end;
                best_spill_idx = active_idx;
            }
        }

        best_spill_idx
    }

    /// Attention-based spill selection: epistemic-aware scoring
    fn select_spill_attention(&self, current: usize) -> usize {
        let current_interval = &self.intervals[current];
        let current_score = self.compute_interval_attention_score(current);

        // Find the best spill candidate (LOWEST attention score = best to spill)
        let mut best_spill_idx = current;
        let mut lowest_score = current_score;

        for &active_idx in &self.active {
            let score = self.compute_interval_attention_score(active_idx);

            // Lower score = better spill candidate
            // (we want to keep high-scoring values in registers)
            if score < lowest_score {
                lowest_score = score;
                best_spill_idx = active_idx;
            }
        }

        best_spill_idx
    }

    /// Compute attention score for a live interval
    fn compute_interval_attention_score(&self, idx: usize) -> f64 {
        let interval = &self.intervals[idx];

        // Compute use density (uses per instruction in live range)
        let live_range = (interval.end - interval.start).max(1) as f64;
        let use_density = interval.use_positions.len() as f64 / live_range;

        // Compute next use distance (normalized)
        let current_pos = interval.start; // Approximation
        let next_use_distance = interval
            .use_positions
            .iter()
            .filter(|&&pos| pos >= current_pos)
            .min()
            .map(|&pos| (pos - current_pos) as f64 / 100.0)
            .unwrap_or(1.0);

        // Build epistemic metadata
        let epistemic = EpistemicMetadata {
            confidence: Some(interval.max_confidence),
            uncertainty: Some(interval.uncertainty),
            has_provenance: interval.has_provenance,
            provenance_depth: if interval.has_provenance { 1 } else { 0 },
            source_op: None,
        };

        compute_attention_score(
            use_density,
            interval.crosses_call,
            next_use_distance,
            &epistemic,
            &self.attention_config,
        )
    }

    /// Record a spill event for metrics collection
    fn record_spill_event(&mut self, idx: usize, reason: SpillReason) {
        let interval = &self.intervals[idx];

        // Build epistemic metadata for attention score computation
        let epistemic = EpistemicMetadata {
            confidence: Some(interval.max_confidence),
            uncertainty: Some(interval.uncertainty),
            has_provenance: interval.has_provenance,
            provenance_depth: if interval.has_provenance { 1 } else { 0 },
            source_op: None,
        };

        // Classify confidence level
        let confidence_class = epistemic.confidence_class(&self.attention_config);

        // Compute the attention score for this interval
        let live_range = (interval.end - interval.start).max(1) as f64;
        let use_density = interval.use_positions.len() as f64 / live_range;
        let next_use_distance = interval
            .use_positions
            .first()
            .map(|&pos| (pos.saturating_sub(interval.start)) as f64 / 100.0)
            .unwrap_or(1.0);

        let attention_score = compute_attention_score(
            use_density,
            interval.crosses_call,
            next_use_distance,
            &epistemic,
            &self.attention_config,
        );

        let event = SpillEvent {
            value: interval.value.index() as u32,
            position: interval.start as u32,
            attention_score,
            confidence: Some(interval.max_confidence),
            uncertainty: Some(interval.uncertainty),
            has_provenance: interval.has_provenance,
            confidence_class,
            reason,
        };

        self.metrics.record_spill(event);
    }

    /// Assign a stack spill slot to an interval
    fn spill_interval(&mut self, idx: usize) -> Result<(), EmitError> {
        let interval = &mut self.intervals[idx];

        // Remove any register assignment
        if let Some(reg) = interval.reg.take() {
            if reg.is_xmm() {
                self.free_xmm_regs.push(reg);
            } else {
                self.free_int_regs.push(reg);
            }
        }

        // Allocate stack slot (8 bytes for 64-bit values)
        // Stack grows downward from RBP
        self.stack_size += 8;
        interval.spill_slot = Some(-self.stack_size);

        // Safety check: don't allow excessive stack usage
        if self.stack_size > 64 * 1024 {
            return Err(EmitError::TooManySpills);
        }

        Ok(())
    }

    /// Get the register assigned to a value (if any)
    pub fn get_reg(&self, value: ValueId) -> Option<X86Reg> {
        self.value_to_interval
            .get(&value)
            .and_then(|&idx| self.intervals[idx].reg)
    }

    /// Get the spill slot for a value (if spilled)
    pub fn get_spill_slot(&self, value: ValueId) -> Option<i32> {
        self.value_to_interval
            .get(&value)
            .and_then(|&idx| self.intervals[idx].spill_slot)
    }

    /// Check if a value is spilled to the stack
    pub fn is_spilled(&self, value: ValueId) -> bool {
        self.get_spill_slot(value).is_some()
    }

    /// Get the total stack space needed (spill slots + callee-saved registers)
    ///
    /// Returns a 16-byte aligned value for System V ABI compliance
    pub fn total_stack_size(&self) -> i32 {
        let spill_space = self.stack_size;
        let callee_saved_space = (self.used_callee_saved.len() as i32) * 8;
        let total = spill_space + callee_saved_space;

        // Round up to 16-byte alignment (required by System V AMD64 ABI)
        (total + 15) & !15
    }

    /// Get the ordered list of callee-saved registers to save/restore
    pub fn callee_saved_to_save(&self) -> Vec<X86Reg> {
        let order = X86Reg::callee_saved_int();
        order
            .iter()
            .filter(|r| self.used_callee_saved.contains(r))
            .copied()
            .collect()
    }
}

impl Default for RegisterAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// X86-64 code emitter
pub struct X86_64Emitter {
    /// Current code buffer
    code: Vec<u8>,
    /// Read-only data appended after code
    rodata: Vec<u8>,
    /// Symbols for read-only data (offsets are relative to rodata start)
    rodata_symbols: Vec<Symbol>,
    /// Counter for unique rodata symbol names
    rodata_counter: usize,
    /// Read-only data entries that need stub functions
    rodata_entries: Vec<RoDataEntry>,
    /// Relocations
    relocations: Vec<Relocation>,
    /// Value to register mapping (legacy, use register_allocator instead)
    reg_alloc: HashMap<ValueId, X86Reg>,
    /// Stack frame size
    frame_size: usize,
    /// Labels (block ID -> code offset)
    labels: HashMap<u32, usize>,
    /// Forward references to patch
    forward_refs: Vec<(usize, u32)>, // (patch offset, block id)
    /// Register allocator for sophisticated allocation
    register_allocator: RegisterAllocator,
    /// External allocation result (from native backend allocator)
    external_alloc_result: Option<crate::backend::native::alloc::AllocResult>,
    /// Spill/reload operations indexed by instruction position
    spill_ops_by_position: std::collections::HashMap<usize, Vec<crate::backend::native::alloc::SpillReloadOp>>,
    /// Current instruction position (for spill/reload insertion)
    current_instruction_pos: usize,
    /// Function name map for resolving Callee::Direct calls
    func_names: HashMap<crate::sir::values::FuncId, String>,
    /// Phi moves keyed by (pred, succ)
    phi_moves: HashMap<(BlockId, BlockId), Vec<PhiMove>>,
    /// Current block being emitted
    current_block: Option<BlockId>,
}

/// Condition codes for x86-64 Jcc/SETcc instructions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CondCode {
    /// Overflow (OF=1)
    O = 0,
    /// No overflow (OF=0)
    NO = 1,
    /// Below/carry (CF=1) - unsigned <
    B = 2,
    /// Above or equal/no carry (CF=0) - unsigned >=
    AE = 3,
    /// Equal/zero (ZF=1)
    E = 4,
    /// Not equal/not zero (ZF=0)
    NE = 5,
    /// Below or equal (CF=1 or ZF=1) - unsigned <=
    BE = 6,
    /// Above (CF=0 and ZF=0) - unsigned >
    A = 7,
    /// Sign (SF=1)
    S = 8,
    /// Not sign (SF=0)
    NS = 9,
    /// Parity (PF=1) - used for unordered float comparison
    P = 10,
    /// Not parity (PF=0) - used for ordered float comparison
    NP = 11,
    /// Less (SF!=OF) - signed <
    L = 12,
    /// Greater or equal (SF=OF) - signed >=
    GE = 13,
    /// Less or equal (ZF=1 or SF!=OF) - signed <=
    LE = 14,
    /// Greater (ZF=0 and SF=OF) - signed >
    G = 15,
}

/// Stack slot information for local variables
#[derive(Debug, Clone)]
pub struct StackSlot {
    /// Offset from RBP (negative)
    pub offset: i32,
    /// Size in bytes
    pub size: usize,
    /// Alignment
    pub align: usize,
}

impl X86_64Emitter {
    pub fn new() -> Self {
        Self {
            code: Vec::with_capacity(4096),
            rodata: Vec::new(),
            rodata_symbols: Vec::new(),
            rodata_counter: 0,
            rodata_entries: Vec::new(),
            relocations: vec![],
            reg_alloc: HashMap::new(),
            frame_size: 0,
            labels: HashMap::new(),
            forward_refs: vec![],
            register_allocator: RegisterAllocator::new(),
            external_alloc_result: None,
            spill_ops_by_position: std::collections::HashMap::new(),
            current_instruction_pos: 0,
            func_names: HashMap::new(),
            phi_moves: HashMap::new(),
            current_block: None,
        }
    }

    /// Set external allocation result from native backend allocator
    /// 
    /// Optimization: Accept reference to avoid unnecessary clone when possible
    pub fn set_external_alloc_result(&mut self, result: &crate::backend::native::alloc::AllocResult) {
        // We need to clone here because we store it for later use
        // But this is better than cloning at the call site
        self.external_alloc_result = Some(result.clone());
    }

    /// Apply external allocation result to register allocator
    fn apply_external_alloc_result(&mut self, func: &SirFunction) {
        if let Some(ref alloc_result) = self.external_alloc_result {
            // Clear previous spill operations
            self.spill_ops_by_position.clear();

            // Build map of spill/reload operations by position
            for op in &alloc_result.spill_code {
                let pos = op.position();
                self.spill_ops_by_position.entry(pos).or_insert_with(Vec::new).push(op.clone());
            }

            // Validate that we have allocation results
            if alloc_result.allocated.is_empty() && alloc_result.spilled.is_empty() {
                // Fallback: use internal allocator if external result is empty
                return;
            }

            // Clear existing intervals and create new ones from external allocation
            self.register_allocator.intervals.clear();
            self.register_allocator.value_to_interval.clear();
            self.register_allocator.used_callee_saved.clear();
            self.register_allocator.stack_size = alloc_result.total_spill_size as i32;

            // Apply allocated registers - create a new interval for each
            for interval in &alloc_result.allocated {
                if let Some(reg) = interval.assigned {
                    let value_id = ValueId(interval.vreg.0);
                    if let Some(x86_reg) = self.virt_reg_to_x86_reg(reg) {
                        // Create a new interval entry for this value
                        let interval_idx = self.register_allocator.intervals.len();
                        self.register_allocator.value_to_interval.insert(value_id, interval_idx);

                        // Create the LiveInterval with the assigned register
                        let has_prov = !matches!(interval.epistemic.provenance, crate::backend::native::metrics::Provenance::Unknown);
                        let new_interval = LiveInterval {
                            value: value_id,
                            start: interval.start,
                            end: interval.end,
                            ty: SirType::i64(), // Default type, actual type not critical for allocation
                            reg: Some(x86_reg),
                            spill_slot: None,
                            crosses_call: interval.crosses_call,
                            use_positions: interval.uses.iter().map(|u| u.pos).collect(),
                            max_confidence: interval.epistemic.confidence,
                            uncertainty: 1.0 - interval.epistemic.confidence,
                            has_provenance: has_prov,
                            epistemic_weight: interval.epistemic.confidence,
                        };
                        self.register_allocator.intervals.push(new_interval);
                        if x86_reg.is_callee_saved() {
                            self.register_allocator.used_callee_saved.insert(x86_reg);
                        }

                        if std::env::var("SOUNIO_DEBUG_ALLOC").is_ok() {
                            eprintln!("  Applied: v{} -> {:?}", value_id.0, x86_reg);
                        }
                    }
                }
            }

            // Apply spill slots
            for interval in &alloc_result.spilled {
                if let Some(slot) = interval.spill_slot {
                    let value_id = ValueId(interval.vreg.0);

                    // Create a new interval for spilled value
                    let interval_idx = self.register_allocator.intervals.len();
                    self.register_allocator.value_to_interval.insert(value_id, interval_idx);

                    let has_prov = !matches!(interval.epistemic.provenance, crate::backend::native::metrics::Provenance::Unknown);
                    let spill_offset = -((slot.0 as i32 + 1) * 8);
                    let new_interval = LiveInterval {
                        value: value_id,
                        start: interval.start,
                        end: interval.end,
                        ty: SirType::i64(),
                        reg: None,
                        spill_slot: Some(spill_offset),
                        crosses_call: interval.crosses_call,
                        use_positions: interval.uses.iter().map(|u| u.pos).collect(),
                        max_confidence: interval.epistemic.confidence,
                        uncertainty: 1.0 - interval.epistemic.confidence,
                        has_provenance: has_prov,
                        epistemic_weight: interval.epistemic.confidence,
                    };
                    self.register_allocator.intervals.push(new_interval);

                    if std::env::var("SOUNIO_DEBUG_ALLOC").is_ok() {
                        eprintln!("  Spilled: v{} -> slot {}", value_id.0, slot.0);
                    }
                }
            }
        }
    }

    /// Convert PhysReg to X86Reg with proper register class support
    fn virt_reg_to_x86_reg(&self, phys_reg: crate::backend::native::alloc::PhysReg) -> Option<X86Reg> {
        match phys_reg.class {
            crate::backend::native::alloc::RegClass::GeneralPurpose => {
                // Map general-purpose registers
                match phys_reg.num {
                    0 => Some(X86Reg::RAX),
                    1 => Some(X86Reg::RCX),
                    2 => Some(X86Reg::RDX),
                    3 => Some(X86Reg::RBX),
                    4 => Some(X86Reg::RSP), // Stack pointer (rarely allocated)
                    5 => Some(X86Reg::RBP), // Base pointer (rarely allocated)
                    6 => Some(X86Reg::RSI),
                    7 => Some(X86Reg::RDI),
                    8 => Some(X86Reg::R8),
                    9 => Some(X86Reg::R9),
                    10 => Some(X86Reg::R10),
                    11 => Some(X86Reg::R11),
                    12 => Some(X86Reg::R12),
                    13 => Some(X86Reg::R13),
                    14 => Some(X86Reg::R14),
                    15 => Some(X86Reg::R15),
                    _ => None,
                }
            }
            crate::backend::native::alloc::RegClass::FloatingPoint => {
                // Map XMM registers (0-15)
                if phys_reg.num <= 15 {
                    match phys_reg.num {
                        0 => Some(X86Reg::XMM0),
                        1 => Some(X86Reg::XMM1),
                        2 => Some(X86Reg::XMM2),
                        3 => Some(X86Reg::XMM3),
                        4 => Some(X86Reg::XMM4),
                        5 => Some(X86Reg::XMM5),
                        6 => Some(X86Reg::XMM6),
                        7 => Some(X86Reg::XMM7),
                        8 => Some(X86Reg::XMM8),
                        9 => Some(X86Reg::XMM9),
                        10 => Some(X86Reg::XMM10),
                        11 => Some(X86Reg::XMM11),
                        12 => Some(X86Reg::XMM12),
                        13 => Some(X86Reg::XMM13),
                        14 => Some(X86Reg::XMM14),
                        15 => Some(X86Reg::XMM15),
                        _ => None,
                    }
                } else {
                    None
                }
            }
            crate::backend::native::alloc::RegClass::ExtendedSimd => {
                // ZMM registers not yet supported in X86Reg enum
                // TODO: Add ZMM support when implementing AVX-512
                None
            }
            crate::backend::native::alloc::RegClass::Special => {
                // Special registers (RSP, RBP) - rarely allocated
                match phys_reg.num {
                    0 => Some(X86Reg::RSP),
                    1 => Some(X86Reg::RBP),
                    _ => None,
                }
            }
        }
    }

    /// Emit a byte
    fn emit_byte(&mut self, b: u8) {
        self.code.push(b);
    }

    /// Emit bytes
    fn emit_bytes(&mut self, bytes: &[u8]) {
        self.code.extend_from_slice(bytes);
    }

    /// Emit 32-bit value (little-endian)
    fn emit_u32(&mut self, val: u32) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Emit 64-bit value (little-endian)
    fn emit_u64(&mut self, val: u64) {
        self.emit_bytes(&val.to_le_bytes());
    }

    /// Get current code offset
    fn offset(&self) -> usize {
        self.code.len()
    }

    // =========================================================================
    // X86-64 INSTRUCTION ENCODING
    // =========================================================================

    /// Emit REX prefix if needed
    fn emit_rex(&mut self, w: bool, r: bool, x: bool, b: bool) {
        let rex = 0x40 | ((w as u8) << 3) | ((r as u8) << 2) | ((x as u8) << 1) | (b as u8);
        if rex != 0x40 {
            self.emit_byte(rex);
        }
    }

    /// Emit REX.W prefix (64-bit operand)
    fn emit_rex_w(&mut self, reg: X86Reg, rm: X86Reg) {
        self.emit_rex(true, reg.needs_rex(), false, rm.needs_rex());
    }

    /// Emit ModR/M byte
    fn emit_modrm(&mut self, mod_: u8, reg: u8, rm: u8) {
        self.emit_byte((mod_ << 6) | ((reg & 7) << 3) | (rm & 7));
    }

    /// MOV reg, reg
    fn emit_mov_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x89); // MOV r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// MOV reg, imm64
    fn emit_mov_ri64(&mut self, dst: X86Reg, imm: i64) {
        self.emit_rex(true, false, false, dst.needs_rex());
        self.emit_byte(0xB8 + dst.encoding()); // MOV r64, imm64
        self.emit_u64(imm as u64);
    }

    /// ADD reg, reg
    fn emit_add_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x01); // ADD r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// SUB reg, reg
    fn emit_sub_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x29); // SUB r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// ADD reg, imm32
    fn emit_add_ri32(&mut self, dst: X86Reg, imm: i32) {
        if imm == 0 {
            return;
        }
        self.emit_rex(true, false, false, dst.needs_rex());
        if imm >= -128 && imm <= 127 {
            // ADD r/m64, imm8
            self.emit_byte(0x83);
            self.emit_modrm(0b11, 0, dst.encoding());
            self.emit_byte(imm as u8);
        } else {
            // ADD r/m64, imm32
            self.emit_byte(0x81);
            self.emit_modrm(0b11, 0, dst.encoding());
            self.emit_u32(imm as u32);
        }
    }

    /// IMUL reg, reg
    fn emit_imul_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(dst, src);
        self.emit_bytes(&[0x0F, 0xAF]); // IMUL r64, r/m64
        self.emit_modrm(0b11, dst.encoding(), src.encoding());
    }

    /// CMP reg, reg
    fn emit_cmp_rr(&mut self, lhs: X86Reg, rhs: X86Reg) {
        self.emit_rex_w(rhs, lhs);
        self.emit_byte(0x39); // CMP r/m64, r64
        self.emit_modrm(0b11, rhs.encoding(), lhs.encoding());
    }

    /// RET
    fn emit_ret(&mut self) {
        self.emit_byte(0xC3);
    }

    /// JMP rel32
    fn emit_jmp_rel32(&mut self, target_offset: i32) {
        self.emit_byte(0xE9);
        self.emit_u32(target_offset as u32);
    }

    /// Jcc rel32 (conditional jump)
    fn emit_jcc_rel32(&mut self, cc: u8, target_offset: i32) {
        self.emit_bytes(&[0x0F, 0x80 + cc]);
        self.emit_u32(target_offset as u32);
    }

    /// PUSH reg
    fn emit_push(&mut self, reg: X86Reg) {
        if reg.needs_rex() {
            self.emit_byte(0x41);
        }
        self.emit_byte(0x50 + reg.encoding());
    }

    /// POP reg
    fn emit_pop(&mut self, reg: X86Reg) {
        if reg.needs_rex() {
            self.emit_byte(0x41);
        }
        self.emit_byte(0x58 + reg.encoding());
    }

    // =========================================================================
    // SSE/AVX INSTRUCTIONS (for f64)
    // =========================================================================

    /// MOVSD xmm, xmm
    fn emit_movsd_rr(&mut self, dst: X86Reg, src: X86Reg) {
        // F2 0F 10 /r — MOVSD xmm1, xmm2
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x10]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// CVTSS2SD xmm, xmm
    fn emit_cvtss2sd(&mut self, dst: X86Reg, src: X86Reg) {
        // F3 0F 5A /r — CVTSS2SD xmm1, xmm2
        self.emit_byte(0xF3);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x5A]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// ADDSD xmm, xmm
    fn emit_addsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x58]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// MULSD xmm, xmm
    fn emit_mulsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x59]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// DIVSD xmm, xmm
    fn emit_divsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x5E]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// SQRTSD xmm, xmm
    fn emit_sqrtsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x51]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// SUBSD xmm, xmm
    fn emit_subsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x5C]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// UCOMISD xmm, xmm (unordered compare scalar double)
    fn emit_ucomisd(&mut self, lhs: X86Reg, rhs: X86Reg) {
        self.emit_byte(0x66);
        if lhs.needs_rex() || rhs.needs_rex() {
            self.emit_rex(false, lhs.needs_rex(), false, rhs.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x2E]);
        self.emit_modrm(0b11, (lhs as u8) & 0x7, (rhs as u8) & 0x7);
    }

    /// MINSD xmm, xmm (minimum scalar double)
    fn emit_minsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x5D]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// MAXSD xmm, xmm (maximum scalar double)
    fn emit_maxsd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0xF2);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x5F]);
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// MOVSD [mem+disp], xmm (store double to memory)
    fn emit_movsd_mem_disp_r64(&mut self, base: X86Reg, disp: i32, src: X86Reg) {
        self.emit_byte(0xF2);
        if base.needs_rex() || src.needs_rex() {
            self.emit_rex(false, false, false, base.needs_rex() || src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x11]);
        if disp == 0 {
            self.emit_modrm(0b00, (src as u8) & 0x7, base.encoding());
        } else if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, (src as u8) & 0x7, base.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, (src as u8) & 0x7, base.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// MOVSD xmm, [mem+disp] (load double from memory)
    fn emit_movsd_r64_mem_disp(&mut self, dst: X86Reg, base: X86Reg, disp: i32) {
        self.emit_byte(0xF2);
        if base.needs_rex() || dst.needs_rex() {
            self.emit_rex(false, false, false, base.needs_rex() || dst.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x10]);
        if disp == 0 {
            self.emit_modrm(0b00, (dst as u8) & 0x7, base.encoding());
        } else if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, (dst as u8) & 0x7, base.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, (dst as u8) & 0x7, base.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// AND reg, reg
    fn emit_and_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x21); // AND r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// OR reg, reg
    fn emit_or_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x09); // OR r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// XOR reg, reg
    fn emit_xor_rr(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex_w(src, dst);
        self.emit_byte(0x31); // XOR r/m64, r64
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// SHL reg, CL
    fn emit_shl_cl(&mut self, dst: X86Reg) {
        self.emit_rex(true, false, false, dst.needs_rex());
        self.emit_byte(0xD3); // SHL r/m64, CL
        self.emit_modrm(0b11, 4, dst.encoding());
    }

    /// SHR reg, CL (logical)
    fn emit_shr_cl(&mut self, dst: X86Reg) {
        self.emit_rex(true, false, false, dst.needs_rex());
        self.emit_byte(0xD3); // SHR r/m64, CL
        self.emit_modrm(0b11, 5, dst.encoding());
    }

    /// SAR reg, CL (arithmetic)
    fn emit_sar_cl(&mut self, dst: X86Reg) {
        self.emit_rex(true, false, false, dst.needs_rex());
        self.emit_byte(0xD3); // SAR r/m64, CL
        self.emit_modrm(0b11, 7, dst.encoding());
    }

    /// IDIV r/m64 - signed divide RDX:RAX by r/m64
    fn emit_idiv(&mut self, divisor: X86Reg) {
        self.emit_rex(true, false, false, divisor.needs_rex());
        self.emit_byte(0xF7); // IDIV r/m64
        self.emit_modrm(0b11, 7, divisor.encoding());
    }

    /// DIV r/m64 - unsigned divide RDX:RAX by r/m64
    fn emit_div(&mut self, divisor: X86Reg) {
        self.emit_rex(true, false, false, divisor.needs_rex());
        self.emit_byte(0xF7); // DIV r/m64
        self.emit_modrm(0b11, 6, divisor.encoding());
    }

    /// CQO - sign extend RAX into RDX:RAX
    fn emit_cqo(&mut self) {
        self.emit_rex(true, false, false, false);
        self.emit_byte(0x99); // CQO
    }

    /// XOR reg, reg (for zeroing RDX before unsigned division)
    fn emit_xor_rr_32(&mut self, dst: X86Reg, src: X86Reg) {
        // 32-bit XOR clears upper 32 bits too
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, src.needs_rex(), false, dst.needs_rex());
        }
        self.emit_byte(0x31);
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// SETcc r/m8 - set byte based on condition code
    fn emit_setcc(&mut self, cc: CondCode, dst: X86Reg) {
        if dst.needs_rex() {
            self.emit_rex(false, false, false, true);
        }
        self.emit_bytes(&[0x0F, 0x90 + cc as u8]);
        self.emit_modrm(0b11, 0, dst.encoding());
    }

    /// MOVZX r64, r/m8 - zero extend byte to 64-bit
    fn emit_movzx_byte(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_rex(true, dst.needs_rex(), false, src.needs_rex());
        self.emit_bytes(&[0x0F, 0xB6]);
        self.emit_modrm(0b11, dst.encoding(), src.encoding());
    }

    /// MOV [rbp + disp], reg - store to stack
    fn emit_mov_mem_rbp_r64(&mut self, disp: i32, src: X86Reg) {
        self.emit_rex(true, src.needs_rex(), false, false);
        self.emit_byte(0x89); // MOV r/m64, r64
        if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, src.encoding(), X86Reg::RBP.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, src.encoding(), X86Reg::RBP.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// MOV reg, [rbp + disp] - load from stack
    fn emit_mov_r64_mem_rbp(&mut self, dst: X86Reg, disp: i32) {
        self.emit_rex(true, dst.needs_rex(), false, false);
        self.emit_byte(0x8B); // MOV r64, r/m64
        if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, dst.encoding(), X86Reg::RBP.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, dst.encoding(), X86Reg::RBP.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// MOV [reg + disp], src - store to memory with displacement
    fn emit_mov_mem_disp_r64(&mut self, base: X86Reg, disp: i32, src: X86Reg) {
        self.emit_rex(true, src.needs_rex(), false, base.needs_rex());
        self.emit_byte(0x89); // MOV r/m64, r64
        let base_enc = base.encoding();
        let needs_sib = base_enc == X86Reg::RSP.encoding();
        if disp == 0 && base_enc != X86Reg::RBP.encoding() {
            if needs_sib {
                self.emit_modrm(0b00, src.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b00, src.encoding(), base_enc);
            }
        } else if disp >= -128 && disp <= 127 {
            if needs_sib {
                self.emit_modrm(0b01, src.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b01, src.encoding(), base_enc);
            }
            self.emit_byte(disp as u8);
        } else {
            if needs_sib {
                self.emit_modrm(0b10, src.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b10, src.encoding(), base_enc);
            }
            self.emit_u32(disp as u32);
        }
    }

    /// MOV dst, [reg + disp] - load from memory with displacement
    fn emit_mov_r64_mem_disp(&mut self, dst: X86Reg, base: X86Reg, disp: i32) {
        self.emit_rex(true, dst.needs_rex(), false, base.needs_rex());
        self.emit_byte(0x8B); // MOV r64, r/m64
        let base_enc = base.encoding();
        let needs_sib = base_enc == X86Reg::RSP.encoding();
        if disp == 0 && base_enc != X86Reg::RBP.encoding() {
            if needs_sib {
                self.emit_modrm(0b00, dst.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b00, dst.encoding(), base_enc);
            }
        } else if disp >= -128 && disp <= 127 {
            if needs_sib {
                self.emit_modrm(0b01, dst.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b01, dst.encoding(), base_enc);
            }
            self.emit_byte(disp as u8);
        } else {
            if needs_sib {
                self.emit_modrm(0b10, dst.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b10, dst.encoding(), base_enc);
            }
            self.emit_u32(disp as u32);
        }
    }

    /// LEA reg, [base + disp] - load effective address
    fn emit_lea(&mut self, dst: X86Reg, base: X86Reg, disp: i32) {
        self.emit_rex(true, dst.needs_rex(), false, base.needs_rex());
        self.emit_byte(0x8D); // LEA r64, m
        let base_enc = base.encoding();
        let needs_sib = base_enc == X86Reg::RSP.encoding();
        if disp >= -128 && disp <= 127 {
            if needs_sib {
                self.emit_modrm(0b01, dst.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b01, dst.encoding(), base_enc);
            }
            self.emit_byte(disp as u8);
        } else {
            if needs_sib {
                self.emit_modrm(0b10, dst.encoding(), 0b100);
                self.emit_byte((0b00 << 6) | (0b100 << 3) | base_enc);
            } else {
                self.emit_modrm(0b10, dst.encoding(), base_enc);
            }
            self.emit_u32(disp as u32);
        }
    }

    /// LEA reg, [base + index*scale + disp] - SIB form
    fn emit_lea_sib(&mut self, dst: X86Reg, base: X86Reg, index: X86Reg, scale: u8, disp: i32) {
        let scale_bits = match scale {
            1 => 0b00,
            2 => 0b01,
            4 => 0b10,
            8 => 0b11,
            _ => 0b00, // default to scale 1
        };

        self.emit_rex(true, dst.needs_rex(), index.needs_rex(), base.needs_rex());
        self.emit_byte(0x8D); // LEA r64, m

        // Need SIB byte when using index
        if disp >= -128 && disp <= 127 && disp != 0 {
            self.emit_modrm(0b01, dst.encoding(), 0b100); // SIB follows
            self.emit_byte((scale_bits << 6) | (index.encoding() << 3) | base.encoding());
            self.emit_byte(disp as u8);
        } else if disp == 0 {
            self.emit_modrm(0b00, dst.encoding(), 0b100);
            self.emit_byte((scale_bits << 6) | (index.encoding() << 3) | base.encoding());
        } else {
            self.emit_modrm(0b10, dst.encoding(), 0b100);
            self.emit_byte((scale_bits << 6) | (index.encoding() << 3) | base.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// SUB rsp, imm32 - for stack allocation
    fn emit_sub_rsp_imm(&mut self, size: i32) {
        self.emit_rex(true, false, false, false);
        self.emit_byte(0x81); // SUB r/m64, imm32
        self.emit_modrm(0b11, 5, X86Reg::RSP.encoding());
        self.emit_u32(size as u32);
    }

    /// ADD rsp, imm32 - for stack deallocation
    fn emit_add_rsp_imm(&mut self, imm: i32) {
        self.emit_rex(true, false, false, false);
        self.emit_byte(0x81); // ADD r/m64, imm32
        self.emit_modrm(0b11, 0, X86Reg::RSP.encoding());
        self.emit_u32(imm as u32);
    }

    /// CALL rel32 - relative call
    fn emit_call_rel32(&mut self, rel: i32) {
        self.emit_byte(0xE8);
        self.emit_u32(rel as u32);
    }

    /// CALL reg - indirect call
    fn emit_call_reg(&mut self, reg: X86Reg) {
        if reg.needs_rex() {
            self.emit_rex(false, false, false, true);
        }
        self.emit_byte(0xFF); // CALL r/m64
        self.emit_modrm(0b11, 2, reg.encoding());
    }

    /// Call external function by name (adds PLT relocation)
    fn emit_call_extern(&mut self, name: &str) {
        let offset = self.code.len() + 1; // +1 for the E8 opcode
        self.relocations.push(Relocation {
            offset,
            kind: RelocKind::PLT32,
            symbol: name.to_string(),
            addend: -4, // PC-relative adjustment
        });
        self.emit_call_rel32(0); // Placeholder, will be patched by linker
    }

    /// LEA reg, [RIP + disp32] with relocation to a symbol
    fn emit_lea_rip_relative(&mut self, dst: X86Reg, symbol: &str) {
        self.emit_rex(true, dst.needs_rex(), false, false);
        self.emit_byte(0x8D); // LEA r64, m
        self.emit_modrm(0b00, dst.encoding(), 0b101); // RIP-relative
        let offset = self.code.len();
        self.emit_u32(0);
        self.relocations.push(Relocation {
            offset,
            kind: RelocKind::PCRel32,
            symbol: symbol.to_string(),
            addend: -4,
        });
    }

    /// Intern read-only data and return a symbol name
    fn intern_rodata(&mut self, bytes: &[u8], align: usize) -> String {
        let align = align.max(1);
        let padding = (align - (self.rodata.len() % align)) % align;
        if padding > 0 {
            self.rodata.extend(std::iter::repeat(0).take(padding));
        }
        let offset = self.rodata.len();
        self.rodata.extend_from_slice(bytes);
        let func_name = format!("__sounio_rodata_{}", self.rodata_counter);
        let data_name = format!("{}_data", func_name);
        self.rodata_counter += 1;
        self.rodata_symbols.push(Symbol {
            name: data_name.clone(),
            offset,
            global: false,
        });
        self.rodata_entries.push(RoDataEntry {
            func_name,
            data_name: data_name.clone(),
        });
        data_name
    }

    fn append_constant_bytes(constant: &Constant, out: &mut Vec<u8>) -> bool {
        match constant {
            Constant::Bool(v) => {
                out.push(if *v { 1 } else { 0 });
                true
            }
            Constant::I8(v) => {
                out.push(*v as u8);
                true
            }
            Constant::I16(v) => {
                out.extend_from_slice(&v.to_le_bytes());
                true
            }
            Constant::I32(v) => {
                out.extend_from_slice(&v.to_le_bytes());
                true
            }
            Constant::I64(v) => {
                out.extend_from_slice(&v.to_le_bytes());
                true
            }
            Constant::F32(v) => {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
                true
            }
            Constant::F64(v) => {
                out.extend_from_slice(&v.to_bits().to_le_bytes());
                true
            }
            Constant::NullPtr => {
                out.extend_from_slice(&0u64.to_le_bytes());
                true
            }
            Constant::Undef(ty) | Constant::ZeroInit(ty) => {
                out.extend(std::iter::repeat(0).take(ty.size_bytes()));
                true
            }
            Constant::Aggregate(elems) => {
                for elem in elems {
                    if !Self::append_constant_bytes(elem, out) {
                        return false;
                    }
                }
                true
            }
            Constant::Epistemic { value, .. } => Self::append_constant_bytes(value, out),
            Constant::Distribution { .. } => false,
        }
    }

    /// XORPD xmm, xmm - XOR packed doubles (for sign manipulation)
    fn emit_xorpd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0x66); // Operand size prefix
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x57]); // XORPD
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// ANDPD xmm, xmm - AND packed doubles (for abs via sign bit masking)
    fn emit_andpd(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0x66);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x54]); // ANDPD
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
    }

    /// ROUNDSD xmm, xmm, imm8 - Round scalar double (SSE4.1)
    /// imm8: 0 = round to nearest, 1 = floor, 2 = ceil, 3 = truncate
    fn emit_roundsd(&mut self, dst: X86Reg, src: X86Reg, mode: u8) {
        self.emit_byte(0x66);
        if dst.needs_rex() || src.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, src.needs_rex());
        }
        self.emit_bytes(&[0x0F, 0x3A, 0x0B]); // ROUNDSD
        self.emit_modrm(0b11, (dst as u8) & 0x7, (src as u8) & 0x7);
        self.emit_byte(mode);
    }

    /// MOVQ xmm, r64 - Move quadword from GPR to XMM
    fn emit_movq_xmm_r64(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0x66);
        self.emit_rex(true, dst.needs_rex(), false, src.needs_rex());
        self.emit_bytes(&[0x0F, 0x6E]); // MOVQ xmm, r/m64
        self.emit_modrm(0b11, (dst as u8) & 0x7, src.encoding());
    }

    /// MOVQ r64, xmm - Move quadword from XMM to GPR
    fn emit_movq_r64_xmm(&mut self, dst: X86Reg, src: X86Reg) {
        self.emit_byte(0x66);
        self.emit_rex(true, dst.needs_rex(), false, src.needs_rex());
        self.emit_bytes(&[0x0F, 0x7E]); // MOVQ r/m64, xmm
        self.emit_modrm(0b11, src.encoding(), dst.encoding());
    }

    /// Load 64-bit constant into XMM via stack
    fn emit_load_f64_const(&mut self, dst: X86Reg, bits: u64) {
        // Move constant to RAX, then through stack to XMM
        self.emit_mov_ri64(X86Reg::RAX, bits as i64);
        self.emit_mov_mem_rbp_r64(-16, X86Reg::RAX);
        self.emit_movsd_rbp_mem(dst, -16);
    }
    
    /// Load 64-bit constant into XMM via RSP (for use in function prologue)
    /// Allocates temporary stack space, stores constant, loads to XMM, then cleans up
    fn emit_load_f64_const_rsp(&mut self, dst: X86Reg, bits: u64) {
        // Allocate temporary stack space (8 bytes)
        self.emit_sub_rsp_imm(8);
        
        // Store constant bytes at [RSP]
        // Move constant to RAX first
        self.emit_mov_ri64(X86Reg::RAX, bits as i64);
        // Store RAX to [RSP]
        self.emit_mov_mem_disp_r64(X86Reg::RSP, 0, X86Reg::RAX);
        
        // Load from [RSP] into XMM register
        self.emit_movsd_r64_mem_disp(dst, X86Reg::RSP, 0);
        
        // Clean up stack
        self.emit_add_rsp_imm(8);
    }

    /// MOVSD [rbp + disp], xmm - store float to stack
    fn emit_movsd_mem_rbp(&mut self, disp: i32, src: X86Reg) {
        self.emit_byte(0xF2);
        if src.needs_rex() {
            self.emit_rex(false, src.needs_rex(), false, false);
        }
        self.emit_bytes(&[0x0F, 0x11]); // MOVSD m64, xmm
        if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, (src as u8) & 0x7, X86Reg::RBP.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, (src as u8) & 0x7, X86Reg::RBP.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// MOVSD xmm, [rbp + disp] - load float from stack
    fn emit_movsd_rbp_mem(&mut self, dst: X86Reg, disp: i32) {
        self.emit_byte(0xF2);
        if dst.needs_rex() {
            self.emit_rex(false, dst.needs_rex(), false, false);
        }
        self.emit_bytes(&[0x0F, 0x10]); // MOVSD xmm, m64
        if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, (dst as u8) & 0x7, X86Reg::RBP.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, (dst as u8) & 0x7, X86Reg::RBP.encoding());
            self.emit_u32(disp as u32);
        }
    }

    // =========================================================================
    // FUNCTION PROLOGUE/EPILOGUE
    // =========================================================================

    /// Emit basic prologue (without callee-saved register saves)
    fn emit_prologue(&mut self, frame_size: usize) {
        // push rbp
        self.emit_push(X86Reg::RBP);
        // mov rbp, rsp
        self.emit_mov_rr(X86Reg::RBP, X86Reg::RSP);

        if frame_size > 0 {
            // sub rsp, frame_size
            self.emit_rex(true, false, false, false);
            self.emit_byte(0x81); // SUB r/m64, imm32
            self.emit_modrm(0b11, 5, X86Reg::RSP.encoding());
            self.emit_u32(frame_size as u32);
        }
    }

    /// Emit prologue with callee-saved register preservation
    ///
    /// System V AMD64 ABI requires preserving: RBX, RBP, R12-R15
    fn emit_prologue_with_saves(&mut self, frame_size: usize, callee_saved: &[X86Reg]) {
        // push rbp
        self.emit_push(X86Reg::RBP);
        // mov rbp, rsp
        self.emit_mov_rr(X86Reg::RBP, X86Reg::RSP);

        // Push callee-saved registers we're using
        for &reg in callee_saved {
            self.emit_push(reg);
        }

        if frame_size > 0 {
            // sub rsp, frame_size
            self.emit_sub_rsp_imm(frame_size as i32);
        }
    }

    /// Emit basic epilogue (without callee-saved register restores)
    fn emit_epilogue(&mut self) {
        // mov rsp, rbp
        self.emit_mov_rr(X86Reg::RSP, X86Reg::RBP);
        // pop rbp
        self.emit_pop(X86Reg::RBP);
        // ret
        self.emit_ret();
    }

    /// Emit epilogue with callee-saved register restoration
    fn emit_epilogue_with_restores(&mut self, callee_saved: &[X86Reg]) {
        // Restore stack pointer to where we pushed callee-saved regs
        // Since we used RBP-relative addressing, we need to restore RSP first
        // by computing the address based on number of pushed registers

        // If we allocated stack space, the callee-saved regs are at known
        // offsets from RBP. We restore by popping in reverse order.

        // lea rsp, [rbp - callee_saved_count * 8]
        if !callee_saved.is_empty() {
            let offset = -((callee_saved.len() as i32) * 8);
            self.emit_lea(X86Reg::RSP, X86Reg::RBP, offset);
        } else {
            self.emit_mov_rr(X86Reg::RSP, X86Reg::RBP);
        }

        // Pop callee-saved registers in reverse order
        for &reg in callee_saved.iter().rev() {
            self.emit_pop(reg);
        }

        // pop rbp
        self.emit_pop(X86Reg::RBP);
        // ret
        self.emit_ret();
    }

    // =========================================================================
    // SPILL/RELOAD HELPERS
    // =========================================================================

    /// Emit code to spill a value to its stack slot
    ///
    /// Used when a value needs to be saved to memory (e.g., around calls
    /// or when register pressure is high)
    fn emit_spill(&mut self, reg: X86Reg, spill_slot: i32) {
        if reg.is_xmm() {
            // MOVSD [rbp + offset], xmm
            self.emit_movsd_mem_rbp(spill_slot, reg);
        } else {
            // MOV [rbp + offset], reg
            self.emit_mov_mem_rbp_r64(spill_slot, reg);
        }
    }

    /// Emit code to reload a value from its stack slot
    ///
    /// Used when a spilled value needs to be loaded back into a register
    fn emit_reload(&mut self, reg: X86Reg, spill_slot: i32) {
        if reg.is_xmm() {
            // MOVSD xmm, [rbp + offset]
            self.emit_movsd_rbp_mem(reg, spill_slot);
        } else {
            // MOV reg, [rbp + offset]
            self.emit_mov_r64_mem_rbp(reg, spill_slot);
        }
    }

    /// Get a register for a value, loading from spill slot if necessary
    ///
    /// If the value is in a register, returns it directly.
    /// If spilled, emits a reload into a scratch register.
    fn get_value_reg(&mut self, value: ValueId, scratch: X86Reg) -> X86Reg {
        if let Some(reg) = self.register_allocator.get_reg(value) {
            reg
        } else if let Some(slot) = self.register_allocator.get_spill_slot(value) {
            self.emit_reload(scratch, slot);
            scratch
        } else {
            // Value not found - this shouldn't happen in correct code
            // Return scratch register as fallback
            scratch
        }
    }

    /// Store a value to its location (register or spill slot)
    ///
    /// If the destination is a register, emits a move if source differs.
    /// If spilled, emits a store to the stack slot.
    fn store_value(&mut self, value: ValueId, src_reg: X86Reg) {
        if let Some(dst_reg) = self.register_allocator.get_reg(value) {
            if dst_reg != src_reg {
                if dst_reg.is_xmm() && src_reg.is_xmm() {
                    self.emit_movsd_rr(dst_reg, src_reg);
                } else if !dst_reg.is_xmm() && !src_reg.is_xmm() {
                    self.emit_mov_rr(dst_reg, src_reg);
                }
            }
        } else if let Some(slot) = self.register_allocator.get_spill_slot(value) {
            self.emit_spill(src_reg, slot);
        }
    }

    // =========================================================================
    // CALLING CONVENTION HELPERS (System V AMD64 ABI)
    // =========================================================================

    /// Move function arguments to their ABI-specified locations
    ///
    /// For outgoing calls, moves values to argument registers.
    /// Integer args: RDI, RSI, RDX, RCX, R8, R9
    /// Float args: XMM0-XMM7
    /// Additional args go on stack (right to left)
    fn setup_call_args(&mut self, args: &[ValueId]) {
        let int_arg_regs = X86Reg::arg_regs();
        let xmm_arg_regs = X86Reg::arg_xmm_regs();

        let mut int_idx = 0;
        let mut xmm_idx = 0;

        for &arg in args {
            // Determine if this is a float argument
            let is_float = self
                .register_allocator
                .value_to_interval
                .get(&arg)
                .map(|&idx| self.register_allocator.intervals[idx].ty.is_float())
                .unwrap_or(false);

            if is_float && xmm_idx < xmm_arg_regs.len() {
                // Float argument - move to XMM register
                let dst = xmm_arg_regs[xmm_idx];
                let src = self.get_value_reg(arg, dst);
                if src != dst {
                    self.emit_movsd_rr(dst, src);
                }
                xmm_idx += 1;
            } else if !is_float && int_idx < int_arg_regs.len() {
                // Integer argument - move to integer register
                let dst = int_arg_regs[int_idx];
                let src = self.get_value_reg(arg, dst);
                if src != dst {
                    self.emit_mov_rr(dst, src);
                }
                int_idx += 1;
            } else {
                // Stack argument - push onto stack
                // Note: In a full implementation, stack args are pushed right-to-left
                let scratch = X86Reg::R11;
                let src = self.get_value_reg(arg, scratch);
                self.emit_push(src);
            }
        }
    }

    /// Save caller-saved registers around a call
    ///
    /// System V AMD64 ABI: caller must save RAX, RCX, RDX, RSI, RDI, R8-R11, XMM0-XMM15
    fn save_caller_saved_regs(&mut self, live_values: &[ValueId]) {
        // For each live value that's in a caller-saved register, save it
        for &value in live_values {
            if let Some(reg) = self.register_allocator.get_reg(value) {
                if !reg.is_callee_saved() {
                    // This value is in a caller-saved register and is live across the call
                    // We need to save it to its spill slot (or allocate one)
                    if let Some(slot) = self.register_allocator.get_spill_slot(value) {
                        self.emit_spill(reg, slot);
                    }
                }
            }
        }
    }

    /// Restore caller-saved registers after a call
    fn restore_caller_saved_regs(&mut self, live_values: &[ValueId]) {
        for &value in live_values {
            if let Some(reg) = self.register_allocator.get_reg(value) {
                if !reg.is_callee_saved() {
                    if let Some(slot) = self.register_allocator.get_spill_slot(value) {
                        self.emit_reload(reg, slot);
                    }
                }
            }
        }
    }

    /// Move incoming function parameters from ABI locations to their allocated registers
    fn setup_incoming_params(&mut self, func: &SirFunction) {
        let int_arg_regs = X86Reg::arg_regs();
        let xmm_arg_regs = X86Reg::arg_xmm_regs();

        let mut int_idx = 0;
        let mut xmm_idx = 0;

        for (param_idx, (_, param_ty)) in func.params.iter().enumerate() {
            let value_id = ValueId::new(param_idx as u32);
            let is_float = param_ty.is_float();

            if is_float && xmm_idx < xmm_arg_regs.len() {
                let src = xmm_arg_regs[xmm_idx];
                self.store_value(value_id, src);
                xmm_idx += 1;
            } else if !is_float && int_idx < int_arg_regs.len() {
                let src = int_arg_regs[int_idx];
                self.store_value(value_id, src);
                int_idx += 1;
            } else {
                // Parameter was passed on stack
                // Would need to compute the correct offset from RBP
            }
        }
    }
}

impl Default for X86_64Emitter {
    fn default() -> Self {
        Self::new()
    }
}

impl CodeEmitter for X86_64Emitter {
    fn emit_module(&mut self, module: &SirModule) -> Result<CodeSegment, EmitError> {
        self.emit_module_with_alloc_results(module, None)
    }

    fn emit_function(&mut self, func: &SirFunction) -> Result<Vec<u8>, EmitError> {
        // Delegate to the implementation in impl X86_64Emitter block
        // (The actual implementation is below, this just satisfies the trait)
        X86_64Emitter::emit_function(self, func)
    }
}

impl X86_64Emitter {
    /// Emit module with optional allocation results
    pub fn emit_module_with_alloc_results(
        &mut self,
        module: &SirModule,
        alloc_results: Option<std::collections::HashMap<crate::sir::values::FuncId, crate::backend::native::alloc::AllocResult>>,
    ) -> Result<CodeSegment, EmitError> {
        let mut symbols = vec![];

        // Build function name map for resolving Callee::Direct calls
        for func in &module.functions {
            self.func_names.insert(func.id, func.name.clone());
        }

        for func in &module.functions {
            // Record symbol
            symbols.push(Symbol {
                name: func.name.clone(),
                offset: self.offset(),
                global: true,
            });

            // Set allocation result for this function if available
            // Optimization: Pass reference to avoid clone at call site
            if let Some(ref alloc_results_map) = alloc_results {
                if let Some(alloc_result) = alloc_results_map.get(&func.id) {
                    self.set_external_alloc_result(alloc_result);
                }
            }

            // Emit function
            self.emit_function(func)?;
            
            // Clear external alloc result after function
            self.external_alloc_result = None;
        }

        self.emit_rodata_stubs(&mut symbols);

        let mut code = std::mem::take(&mut self.code);
        let mut symbols_out = symbols;

        let rodata = std::mem::take(&mut self.rodata);
        let rodata_symbols = std::mem::take(&mut self.rodata_symbols);
        if !rodata.is_empty() {
            let align = 8usize;
            let padding = (align - (code.len() % align)) % align;
            if padding > 0 {
                code.extend(std::iter::repeat(0).take(padding));
            }
            let rodata_base = code.len();
            for sym in rodata_symbols {
                symbols_out.push(Symbol {
                    name: sym.name,
                    offset: rodata_base + sym.offset,
                    global: sym.global,
                });
            }
            code.extend_from_slice(&rodata);
        }

        self.rodata_counter = 0;

        Ok(CodeSegment {
            code,
            relocations: std::mem::take(&mut self.relocations),
            symbols: symbols_out,
        })
    }

    fn emit_rodata_stubs(&mut self, symbols: &mut Vec<Symbol>) {
        let entries = std::mem::take(&mut self.rodata_entries);
        for entry in entries {
            symbols.push(Symbol {
                name: entry.func_name.clone(),
                offset: self.offset(),
                global: false,
            });
            self.emit_lea_rip_relative(X86Reg::RAX, &entry.data_name);
            self.emit_byte(0xC3); // ret
        }
    }

    fn build_phi_moves(&mut self, func: &SirFunction) {
        self.phi_moves.clear();
        for block in &func.blocks {
            for inst in &block.instructions {
                if let SirInst::Phi { incoming, .. } = &inst.inst {
                    let Some(dst) = inst.result else {
                        continue;
                    };
                    for (pred, src) in incoming {
                        self.phi_moves
                            .entry((*pred, block.id))
                            .or_default()
                            .push(PhiMove { dst, src: *src });
                    }
                }
            }
        }
    }

    fn emit_phi_moves_for_edge(
        &mut self,
        pred: BlockId,
        succ: BlockId,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        let moves = match self.phi_moves.get(&(pred, succ)) {
            Some(moves) => moves.clone(),
            None => return Ok(()),
        };

        for mv in moves {
            self.emit_phi_move(mv.dst, mv.src, func)?;
        }
        Ok(())
    }

    fn emit_phi_move(
        &mut self,
        dst: ValueId,
        src: ValueId,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        if let Some(dst_reg) = self.register_allocator.get_reg(dst) {
            let src_reg = self.get_value_reg(src, dst_reg);
            if dst_reg.is_xmm() && src_reg.is_xmm() {
                if dst_reg != src_reg {
                    self.emit_movsd_rr(dst_reg, src_reg);
                }
            } else if !dst_reg.is_xmm() && !src_reg.is_xmm() {
                if dst_reg != src_reg {
                    self.emit_mov_rr(dst_reg, src_reg);
                }
            } else if dst_reg.is_xmm() && !src_reg.is_xmm() {
                self.emit_movq_xmm_r64(dst_reg, src_reg);
            } else if !dst_reg.is_xmm() && src_reg.is_xmm() {
                self.emit_movq_r64_xmm(dst_reg, src_reg);
            }
            return Ok(());
        }

        if let Some(slot) = self.register_allocator.get_spill_slot(dst) {
            let is_float = self.get_value_type(src, func).is_float();
            let scratch = if is_float { X86Reg::XMM15 } else { X86Reg::R11 };
            let src_reg = self.get_value_reg(src, scratch);
            self.emit_spill(src_reg, slot);
        }

        Ok(())
    }

    fn patch_rel32(&mut self, patch_offset: usize, rel: i32) {
        let bytes = rel.to_le_bytes();
        self.code[patch_offset] = bytes[0];
        self.code[patch_offset + 1] = bytes[1];
        self.code[patch_offset + 2] = bytes[2];
        self.code[patch_offset + 3] = bytes[3];
    }

    fn emit_function(&mut self, func: &SirFunction) -> Result<Vec<u8>, EmitError> {
        let start = self.offset();

        // Step 1: Run register allocation (or apply external result)
        if self.external_alloc_result.is_some() {
            // Use external allocation result from native backend
            self.register_allocator.compute_live_intervals(func)?;
            self.apply_external_alloc_result(func);
        } else {
            // Use internal register allocator
            self.register_allocator.allocate_registers(func)?;
        }

        // Step 2: Calculate frame size from register allocation
        // Include space for spilled values and alignment
        let spill_space = self.register_allocator.stack_size as usize;
        let callee_saved = self.register_allocator.callee_saved_to_save();
        let callee_saved_space = callee_saved.len() * 8;

        // Total frame needs to maintain 16-byte alignment
        // Stack layout:
        //   [return address]     <- 8 bytes, pushed by CALL
        //   [saved RBP]          <- 8 bytes, pushed in prologue
        //   [callee-saved regs]  <- callee_saved.len() * 8
        //   [spill slots]        <- spill_space
        //   [local variables]    <- any additional stack allocation
        //
        // RSP must be 16-byte aligned before CALL instructions
        let mut frame_size = if spill_space > 0 {
            // Round up to 16-byte alignment
            (spill_space + 15) & !15
        } else {
            0
        };
        // Account for callee-saved pushes so RSP stays 16-byte aligned before calls.
        if (callee_saved_space % 16) != 0 {
            frame_size += 8;
        }
        self.frame_size = frame_size;

        // Step 3: Emit prologue with callee-saved register saves
        self.emit_prologue_with_saves(self.frame_size, &callee_saved);

        // Step 4: Move incoming parameters to their allocated locations
        self.setup_incoming_params(func);

        // Step 5: Emit instructions for each basic block
        // Clear labels from previous functions
        self.labels.clear();
        self.forward_refs.clear();
        self.current_instruction_pos = 0;
        self.build_phi_moves(func);

        for block in &func.blocks {
            self.current_block = Some(block.id);
            // Record label for this block
            self.labels.insert(block.id.0, self.offset());

            // Emit instructions
            for inst in &block.instructions {
                // Insert spill/reload operations before this instruction if needed
                // Optimization: Group operations and eliminate redundancies
                if let Some(ops) = self.spill_ops_by_position.get(&self.current_instruction_pos) {
                    // Separate spills and reloads for better ordering
                    let mut spills: Vec<_> = Vec::new();
                    let mut reloads: Vec<_> = Vec::new();
                    
                    for op in ops {
                        match op {
                            crate::backend::native::alloc::SpillReloadOp::Spill { reg, slot, .. } => {
                                spills.push((*reg, *slot));
                            }
                            crate::backend::native::alloc::SpillReloadOp::Reload { reg, slot, .. } => {
                                reloads.push((*reg, *slot));
                            }
                        }
                    }
                    
                    // Remove duplicate spills/reloads for the same register/slot
                    // (same register being spilled/reloaded to/from same slot is redundant)
                    spills.sort_by_key(|(reg, slot)| (reg.class, reg.num, slot.0));
                    spills.dedup_by_key(|(reg, slot)| (reg.class, reg.num, slot.0));
                    
                    reloads.sort_by_key(|(reg, slot)| (reg.class, reg.num, slot.0));
                    reloads.dedup_by_key(|(reg, slot)| (reg.class, reg.num, slot.0));
                    
                    // Emit reloads first (values needed for instruction)
                    for (reg, slot) in &reloads {
                        if let Some(x86_reg) = self.virt_reg_to_x86_reg(*reg) {
                            self.emit_reload(x86_reg, slot.0 as i32);
                        }
                    }
                    
                    // Then emit spills (values no longer needed)
                    for (reg, slot) in &spills {
                        if let Some(x86_reg) = self.virt_reg_to_x86_reg(*reg) {
                            self.emit_spill(x86_reg, slot.0 as i32);
                        }
                    }
                }

                self.emit_instruction(inst, func)?;
                self.current_instruction_pos += 1;
            }

            // Emit terminator
            if let Some(term) = &block.terminator {
                self.emit_terminator(term, func, &callee_saved)?;
            }
        }

        self.current_block = None;
        self.phi_moves.clear();

        // Step 6: Patch forward references
        self.patch_forward_refs();

        Ok(self.code[start..].to_vec())
    }
}

// Additional implementation methods for X86_64Emitter (not part of CodeEmitter trait)
impl X86_64Emitter {
    /// Emit a single instruction
    fn emit_instruction(
        &mut self,
        inst: &super::blocks::Instruction,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        match &inst.inst {
            SirInst::BinOp { op, lhs, rhs } => {
                if let Some(result) = inst.result {
                    self.emit_binop(result, *op, *lhs, *rhs)?;
                }
            }
            SirInst::Cmp { op, lhs, rhs } => {
                if let Some(result) = inst.result {
                    self.emit_cmp(result, *op, *lhs, *rhs)?;
                }
            }
            SirInst::Const(constant) => {
                if let Some(result) = inst.result {
                    self.emit_const(result, constant)?;
                }
            }
            SirInst::UnaryFloat { op, val } => {
                if let Some(result) = inst.result {
                    self.emit_unary_float(result, *op, *val)?;
                }
            }
            SirInst::Call(info) => {
                self.emit_call(inst.result, info, func)?;
            }
            SirInst::Memory(mem_op) => {
                self.emit_memory_op(inst.result, mem_op)?;
            }
            SirInst::Select {
                cond,
                then_val,
                else_val,
            } => {
                if let Some(result) = inst.result {
                    self.emit_select(result, *cond, *then_val, *else_val)?;
                }
            }
            SirInst::BuildAggregate { fields, ty } => {
                if let Some(result) = inst.result {
                    self.emit_build_aggregate(result, fields, ty)?;
                }
            }
            SirInst::Cast { op, val, to_ty } => {
                if let Some(result) = inst.result {
                    self.emit_cast(result, *op, *val, to_ty)?;
                }
            }
            SirInst::BinaryFloat { op, lhs, rhs } => {
                if let Some(result) = inst.result {
                    self.emit_binary_float(result, *op, *lhs, *rhs)?;
                }
            }
            SirInst::Phi { ty, incoming } => {
                // Phi nodes should be resolved by register allocation (coalescing)
                // At codegen time, all incoming values should be in the same location
                // No code emission needed if properly coalesced
            }
            SirInst::DebugValue { .. } => {
                // Debug values are no-ops in release builds
            }
            SirInst::Assert { cond, message, failure_mode } => {
                // In debug mode, emit assertion check
                // For now, skip assertions in codegen
            }
            SirInst::Assume(_) => {
                // Optimization hints - no code generation needed
            }
            SirInst::Epistemic(ep_op) => {
                if let Some(result) = inst.result {
                    self.emit_epistemic_op(result, ep_op, func)?;
                }
            }
            SirInst::Scientific(sci_op) => {
                if let Some(result) = inst.result {
                    self.emit_scientific_op(result, sci_op, func)?;
                }
            }
            _ => {
                // Other instructions not yet implemented (Vector, Prob)
            }
        }
        Ok(())
    }

    /// Emit aggregate (struct/slice) construction
    /// For slices: builds {ptr, len} fat pointer
    fn emit_build_aggregate(
        &mut self,
        result: ValueId,
        fields: &[ValueId],
        _ty: &SirType,
    ) -> Result<(), EmitError> {
        // For 2-field aggregates like slices {ptr, len}, we can use register pairs
        // System V ABI: return small structs in RAX:RDX
        if fields.len() == 2 {
            // Get field values into registers
            let ptr_reg = self.get_value_reg(fields[0], X86Reg::RAX);
            let len_reg = self.get_value_reg(fields[1], X86Reg::RDX);

            // Move to RAX:RDX if not already there
            if ptr_reg != X86Reg::RAX {
                self.emit_mov_rr(X86Reg::RAX, ptr_reg);
            }
            if len_reg != X86Reg::RDX {
                self.emit_mov_rr(X86Reg::RDX, len_reg);
            }

            // Record that result is in RAX (ptr) with RDX holding len
            // For now, we treat the aggregate as residing in RAX
            if let Some(dst) = self.register_allocator.get_reg(result) {
                if dst != X86Reg::RAX {
                    self.emit_mov_rr(dst, X86Reg::RAX);
                }
            }
        } else {
            // Larger aggregates: allocate stack space and store fields
            // Calculate total size (assume 8 bytes per field for now)
            let size = fields.len() * 8;

            // Allocate stack space
            self.emit_sub_rsp_imm(size as i32);

            // Store each field
            for (i, field) in fields.iter().enumerate() {
                let field_reg = self.get_value_reg(*field, X86Reg::R10);
                let offset = (i * 8) as i32;
                self.emit_mov_mem_disp_r64(X86Reg::RSP, offset, field_reg);
            }

            // Result is pointer to stack allocation
            if let Some(dst) = self.register_allocator.get_reg(result) {
                self.emit_mov_rr(dst, X86Reg::RSP);
            }
        }
        Ok(())
    }

    /// Emit a binary operation
    fn emit_binop(
        &mut self,
        result: ValueId,
        op: ArithOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), EmitError> {
        // Note: Unit checking is performed at the type-checking phase.
        // By the time we reach codegen, units have been verified and erased.
        // This is a zero-cost abstraction - no runtime overhead.

        // Get registers or load from spill slots
        let lhs_reg = self.get_value_reg(lhs, X86Reg::R10);
        let rhs_reg = self.get_value_reg(rhs, X86Reg::R11);

        // Determine destination
        let dst_reg = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        match op {
            ArithOp::Add => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_add_rr(dst_reg, rhs_reg);
            }
            ArithOp::Sub => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_sub_rr(dst_reg, rhs_reg);
            }
            ArithOp::Mul => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_imul_rr(dst_reg, rhs_reg);
            }
            ArithOp::SDiv => {
                // IDIV uses RDX:RAX, result in RAX (quotient) and RDX (remainder)
                self.emit_mov_rr(X86Reg::RAX, lhs_reg);
                self.emit_cqo(); // Sign extend RAX into RDX:RAX
                self.emit_idiv(rhs_reg);
                if dst_reg != X86Reg::RAX {
                    self.emit_mov_rr(dst_reg, X86Reg::RAX);
                }
            }
            ArithOp::UDiv => {
                self.emit_mov_rr(X86Reg::RAX, lhs_reg);
                self.emit_xor_rr_32(X86Reg::RDX, X86Reg::RDX); // Zero RDX
                self.emit_div(rhs_reg);
                if dst_reg != X86Reg::RAX {
                    self.emit_mov_rr(dst_reg, X86Reg::RAX);
                }
            }
            ArithOp::SRem => {
                self.emit_mov_rr(X86Reg::RAX, lhs_reg);
                self.emit_cqo();
                self.emit_idiv(rhs_reg);
                // Remainder is in RDX
                if dst_reg != X86Reg::RDX {
                    self.emit_mov_rr(dst_reg, X86Reg::RDX);
                }
            }
            ArithOp::URem => {
                self.emit_mov_rr(X86Reg::RAX, lhs_reg);
                self.emit_xor_rr_32(X86Reg::RDX, X86Reg::RDX);
                self.emit_div(rhs_reg);
                if dst_reg != X86Reg::RDX {
                    self.emit_mov_rr(dst_reg, X86Reg::RDX);
                }
            }
            ArithOp::And => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_and_rr(dst_reg, rhs_reg);
            }
            ArithOp::Or => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_or_rr(dst_reg, rhs_reg);
            }
            ArithOp::Xor => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                self.emit_xor_rr(dst_reg, rhs_reg);
            }
            ArithOp::Shl => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                // Shift amount must be in CL
                if rhs_reg != X86Reg::RCX {
                    self.emit_mov_rr(X86Reg::RCX, rhs_reg);
                }
                self.emit_shl_cl(dst_reg);
            }
            ArithOp::LShr => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                if rhs_reg != X86Reg::RCX {
                    self.emit_mov_rr(X86Reg::RCX, rhs_reg);
                }
                self.emit_shr_cl(dst_reg);
            }
            ArithOp::AShr => {
                if dst_reg != lhs_reg {
                    self.emit_mov_rr(dst_reg, lhs_reg);
                }
                if rhs_reg != X86Reg::RCX {
                    self.emit_mov_rr(X86Reg::RCX, rhs_reg);
                }
                self.emit_sar_cl(dst_reg);
            }
            // Floating point operations
            ArithOp::FAdd => {
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let lhs_xmm = self.get_value_reg(lhs, X86Reg::XMM14);
                let rhs_xmm = self.get_value_reg(rhs, X86Reg::XMM15);
                if dst_xmm != lhs_xmm {
                    self.emit_movsd_rr(dst_xmm, lhs_xmm);
                }
                self.emit_addsd(dst_xmm, rhs_xmm);
            }
            ArithOp::FSub => {
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let lhs_xmm = self.get_value_reg(lhs, X86Reg::XMM14);
                let rhs_xmm = self.get_value_reg(rhs, X86Reg::XMM15);
                if dst_xmm != lhs_xmm {
                    self.emit_movsd_rr(dst_xmm, lhs_xmm);
                }
                self.emit_subsd(dst_xmm, rhs_xmm);
            }
            ArithOp::FMul => {
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let lhs_xmm = self.get_value_reg(lhs, X86Reg::XMM14);
                let rhs_xmm = self.get_value_reg(rhs, X86Reg::XMM15);
                if dst_xmm != lhs_xmm {
                    self.emit_movsd_rr(dst_xmm, lhs_xmm);
                }
                self.emit_mulsd(dst_xmm, rhs_xmm);
            }
            ArithOp::FDiv => {
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let lhs_xmm = self.get_value_reg(lhs, X86Reg::XMM14);
                let rhs_xmm = self.get_value_reg(rhs, X86Reg::XMM15);
                if dst_xmm != lhs_xmm {
                    self.emit_movsd_rr(dst_xmm, lhs_xmm);
                }
                self.emit_divsd(dst_xmm, rhs_xmm);
            }
            ArithOp::FRem => {
                // FRem requires a function call to fmod
                // System V AMD64: XMM0 = arg1, XMM1 = arg2, return in XMM0
                let lhs_xmm = self.get_value_reg(lhs, X86Reg::XMM14);
                let rhs_xmm = self.get_value_reg(rhs, X86Reg::XMM15);
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);

                if lhs_xmm != X86Reg::XMM0 {
                    self.emit_movsd_rr(X86Reg::XMM0, lhs_xmm);
                }
                if rhs_xmm != X86Reg::XMM1 {
                    self.emit_movsd_rr(X86Reg::XMM1, rhs_xmm);
                }
                self.emit_call_extern("fmod");
                // Result is in XMM0
                if dst_xmm != X86Reg::XMM0 {
                    self.emit_movsd_rr(dst_xmm, X86Reg::XMM0);
                }
            }
        }

        // Store result if spilled
        if self.register_allocator.is_spilled(result) {
            if let Some(slot) = self.register_allocator.get_spill_slot(result) {
                self.emit_spill(dst_reg, slot);
            }
        }

        Ok(())
    }

    /// Emit a comparison operation
    fn emit_cmp(
        &mut self,
        result: ValueId,
        op: CmpOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), EmitError> {
        let dst_reg = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        // Check if this is a float comparison
        let is_float_cmp = matches!(
            op,
            CmpOp::FOEq | CmpOp::FONe | CmpOp::FOLt | CmpOp::FOLe | CmpOp::FOGt | CmpOp::FOGe
        );

        if is_float_cmp {
            // Float comparison using UCOMISD
            let lhs_reg = self.get_value_reg(lhs, X86Reg::XMM14);
            let rhs_reg = self.get_value_reg(rhs, X86Reg::XMM15);

            // UCOMISD sets: ZF=1,PF=1,CF=1 if unordered (NaN)
            //              ZF=0,PF=0,CF=0 if lhs > rhs
            //              ZF=1,PF=0,CF=0 if lhs == rhs
            //              ZF=0,PF=0,CF=1 if lhs < rhs
            self.emit_ucomisd(lhs_reg, rhs_reg);

            // For ordered comparisons, we need to handle NaN properly
            // Using SETA/SETAE/SETB/SETBE which check CF and ZF
            let cc = match op {
                CmpOp::FOEq => {
                    // Equal: ZF=1 and not unordered (PF=0)
                    // SETE sets if ZF=1, then AND with SETNP
                    self.emit_setcc(CondCode::E, dst_reg);
                    self.emit_setcc(CondCode::NP, X86Reg::R11);
                    self.emit_and_rr(dst_reg, X86Reg::R11);
                    self.emit_movzx_byte(dst_reg, dst_reg);
                    if self.register_allocator.is_spilled(result) {
                        if let Some(slot) = self.register_allocator.get_spill_slot(result) {
                            self.emit_spill(dst_reg, slot);
                        }
                    }
                    return Ok(());
                }
                CmpOp::FONe => {
                    // Not equal: ZF=0 OR unordered (PF=1)
                    // SETNE sets if ZF=0, then OR with SETP
                    self.emit_setcc(CondCode::NE, dst_reg);
                    self.emit_setcc(CondCode::P, X86Reg::R11);
                    self.emit_or_rr(dst_reg, X86Reg::R11);
                    self.emit_movzx_byte(dst_reg, dst_reg);
                    if self.register_allocator.is_spilled(result) {
                        if let Some(slot) = self.register_allocator.get_spill_slot(result) {
                            self.emit_spill(dst_reg, slot);
                        }
                    }
                    return Ok(());
                }
                CmpOp::FOLt => CondCode::B,  // Below: CF=1 (and not unordered)
                CmpOp::FOLe => CondCode::BE, // Below or Equal: CF=1 || ZF=1
                CmpOp::FOGt => CondCode::A,  // Above: CF=0 && ZF=0
                CmpOp::FOGe => CondCode::AE, // Above or Equal: CF=0
                _ => unreachable!(),
            };

            // For FOLt/FOLe/FOGt/FOGe, need to also check for unordered (PF=1)
            // and return false for unordered in ordered comparisons
            self.emit_setcc(cc, dst_reg);
            self.emit_setcc(CondCode::NP, X86Reg::R11); // Not parity = not unordered
            self.emit_and_rr(dst_reg, X86Reg::R11);
        } else {
            // Integer comparison
            let lhs_reg = self.get_value_reg(lhs, X86Reg::R10);
            let rhs_reg = self.get_value_reg(rhs, X86Reg::R11);

            self.emit_cmp_rr(lhs_reg, rhs_reg);

            // Set result based on condition code
            let cc = match op {
                CmpOp::Eq => CondCode::E,
                CmpOp::Ne => CondCode::NE,
                CmpOp::SLt => CondCode::L,
                CmpOp::SLe => CondCode::LE,
                CmpOp::SGt => CondCode::G,
                CmpOp::SGe => CondCode::GE,
                CmpOp::ULt => CondCode::B,
                CmpOp::ULe => CondCode::BE,
                CmpOp::UGt => CondCode::A,
                CmpOp::UGe => CondCode::AE,
                _ => unreachable!("Float comparisons handled above"),
            };

            self.emit_setcc(cc, dst_reg);
        }

        // SETcc sets the low byte, we need to zero-extend
        self.emit_movzx_byte(dst_reg, dst_reg);

        if self.register_allocator.is_spilled(result) {
            if let Some(slot) = self.register_allocator.get_spill_slot(result) {
                self.emit_spill(dst_reg, slot);
            }
        }

        Ok(())
    }

    /// Emit a constant load
    fn emit_const(
        &mut self,
        result: ValueId,
        constant: &super::values::Constant,
    ) -> Result<(), EmitError> {
        let dst_reg = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        match constant {
            super::values::Constant::I8(val) => {
                self.emit_mov_ri64(dst_reg, *val as i64);
            }
            super::values::Constant::I16(val) => {
                self.emit_mov_ri64(dst_reg, *val as i64);
            }
            super::values::Constant::I64(val) => {
                self.emit_mov_ri64(dst_reg, *val);
            }
            super::values::Constant::I32(val) => {
                self.emit_mov_ri64(dst_reg, *val as i64);
            }
            super::values::Constant::Bool(val) => {
                self.emit_mov_ri64(dst_reg, if *val { 1 } else { 0 });
            }
            super::values::Constant::NullPtr => {
                self.emit_mov_ri64(dst_reg, 0);
            }
            super::values::Constant::F64(val) => {
                // For floats, we'd need to load from a constant pool
                // For now, move the bits through an integer register
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let bits = val.to_bits() as i64;
                self.emit_mov_ri64(X86Reg::RAX, bits);
                // MOVQ xmm, rax would go here (0x66 0x48 0x0F 0x6E ...)
                // Simplified: just store to stack and reload
                self.emit_mov_mem_rbp_r64(-8, X86Reg::RAX);
                self.emit_movsd_rbp_mem(dst_xmm, -8);
            }
            super::values::Constant::F32(val) => {
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                let bits = val.to_bits() as i64;
                self.emit_mov_ri64(X86Reg::RAX, bits);
                self.emit_movq_xmm_r64(dst_xmm, X86Reg::RAX);
            }
            super::values::Constant::Undef(_) | super::values::Constant::ZeroInit(_) => {
                self.emit_mov_ri64(dst_reg, 0);
            }
            super::values::Constant::Aggregate(_) => {
                let mut bytes = Vec::new();
                if Self::append_constant_bytes(constant, &mut bytes) {
                    let symbol = self.intern_rodata(&bytes, 8);
                    self.emit_lea_rip_relative(dst_reg, &symbol);
                } else {
                    self.emit_mov_ri64(dst_reg, 0);
                }
            }
            _ => {
                // Other constant types not yet implemented
            }
        }

        if self.register_allocator.is_spilled(result) {
            if let Some(slot) = self.register_allocator.get_spill_slot(result) {
                self.emit_spill(dst_reg, slot);
            }
        }

        Ok(())
    }

    /// Emit a unary float operation
    fn emit_unary_float(
        &mut self,
        result: ValueId,
        op: UnaryFloatOp,
        val: ValueId,
    ) -> Result<(), EmitError> {
        let src = self.get_value_reg(val, X86Reg::XMM15);
        let dst = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::XMM0);

        match op {
            UnaryFloatOp::Neg => {
                // Negate by XORing with sign bit mask (0x8000000000000000)
                // Load sign bit mask into XMM14
                self.emit_load_f64_const(X86Reg::XMM14, 0x8000_0000_0000_0000);
                if dst != src {
                    self.emit_movsd_rr(dst, src);
                }
                self.emit_xorpd(dst, X86Reg::XMM14);
            }
            UnaryFloatOp::Abs => {
                // Absolute value by ANDing with mask that clears sign bit (0x7FFFFFFFFFFFFFFF)
                self.emit_load_f64_const(X86Reg::XMM14, 0x7FFF_FFFF_FFFF_FFFF);
                if dst != src {
                    self.emit_movsd_rr(dst, src);
                }
                self.emit_andpd(dst, X86Reg::XMM14);
            }
            UnaryFloatOp::Sqrt => {
                self.emit_sqrtsd(dst, src);
            }
            UnaryFloatOp::Floor => {
                // ROUNDSD with mode 1 = floor (round toward -infinity)
                self.emit_roundsd(dst, src, 0x01 | 0x08); // 0x08 = use MXCSR
            }
            UnaryFloatOp::Ceil => {
                // ROUNDSD with mode 2 = ceil (round toward +infinity)
                self.emit_roundsd(dst, src, 0x02 | 0x08);
            }
            UnaryFloatOp::Round => {
                // ROUNDSD with mode 0 = round to nearest (banker's rounding)
                self.emit_roundsd(dst, src, 0x00 | 0x08);
            }
            UnaryFloatOp::Trunc => {
                // ROUNDSD with mode 3 = truncate (round toward zero)
                self.emit_roundsd(dst, src, 0x03 | 0x08);
            }
            // Library calls for transcendental functions
            UnaryFloatOp::Sin => {
                self.emit_libm_call("sin", src, dst);
            }
            UnaryFloatOp::Cos => {
                self.emit_libm_call("cos", src, dst);
            }
            UnaryFloatOp::Tan => {
                self.emit_libm_call("tan", src, dst);
            }
            UnaryFloatOp::Exp => {
                self.emit_libm_call("exp", src, dst);
            }
            UnaryFloatOp::Exp2 => {
                self.emit_libm_call("exp2", src, dst);
            }
            UnaryFloatOp::Log => {
                self.emit_libm_call("log", src, dst);
            }
            UnaryFloatOp::Log2 => {
                self.emit_libm_call("log2", src, dst);
            }
            UnaryFloatOp::Log10 => {
                self.emit_libm_call("log10", src, dst);
            }
        }

        Ok(())
    }

    /// Emit a call to a libm function (sin, cos, exp, log, etc.)
    /// Follows System V AMD64 ABI: float arg in XMM0, return in XMM0
    fn emit_libm_call(&mut self, name: &str, src: X86Reg, dst: X86Reg) {
        // Move argument to XMM0 if not already there
        if src != X86Reg::XMM0 {
            self.emit_movsd_rr(X86Reg::XMM0, src);
        }
        // Call the libm function
        self.emit_call_extern(name);
        // Result is in XMM0, move to dst if different
        if dst != X86Reg::XMM0 {
            self.emit_movsd_rr(dst, X86Reg::XMM0);
        }
    }

    /// Emit a binary float operation (pow, min, max, copysign, atan2)
    fn emit_binary_float(
        &mut self,
        result: ValueId,
        op: BinaryFloatOp,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Result<(), EmitError> {
        let lhs_reg = self.get_value_reg(lhs, X86Reg::XMM14);
        let rhs_reg = self.get_value_reg(rhs, X86Reg::XMM15);
        let dst = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::XMM0);

        match op {
            BinaryFloatOp::Min => {
                // MINSD dst, src - minimum of two doubles
                if dst != lhs_reg {
                    self.emit_movsd_rr(dst, lhs_reg);
                }
                // MINSD: F2 0F 5D /r
                self.emit_byte(0xF2);
                if dst.needs_rex() || rhs_reg.needs_rex() {
                    self.emit_rex(false, dst.needs_rex(), false, rhs_reg.needs_rex());
                }
                self.emit_bytes(&[0x0F, 0x5D]);
                self.emit_modrm(0b11, (dst as u8) & 0x7, (rhs_reg as u8) & 0x7);
            }
            BinaryFloatOp::Max => {
                // MAXSD dst, src - maximum of two doubles
                if dst != lhs_reg {
                    self.emit_movsd_rr(dst, lhs_reg);
                }
                // MAXSD: F2 0F 5F /r
                self.emit_byte(0xF2);
                if dst.needs_rex() || rhs_reg.needs_rex() {
                    self.emit_rex(false, dst.needs_rex(), false, rhs_reg.needs_rex());
                }
                self.emit_bytes(&[0x0F, 0x5F]);
                self.emit_modrm(0b11, (dst as u8) & 0x7, (rhs_reg as u8) & 0x7);
            }
            BinaryFloatOp::Pow => {
                // pow(x, y) - call libm
                self.emit_libm_call_2arg("pow", lhs_reg, rhs_reg, dst);
            }
            BinaryFloatOp::Atan2 => {
                // atan2(y, x) - call libm
                self.emit_libm_call_2arg("atan2", lhs_reg, rhs_reg, dst);
            }
            BinaryFloatOp::CopySign => {
                // copysign(x, y) - call libm (or implement with bit manipulation)
                self.emit_libm_call_2arg("copysign", lhs_reg, rhs_reg, dst);
            }
        }

        Ok(())
    }

    /// Emit a call to a 2-argument libm function (pow, atan2, copysign)
    fn emit_libm_call_2arg(&mut self, name: &str, arg1: X86Reg, arg2: X86Reg, dst: X86Reg) {
        // System V ABI: first float arg in XMM0, second in XMM1
        if arg1 != X86Reg::XMM0 {
            self.emit_movsd_rr(X86Reg::XMM0, arg1);
        }
        if arg2 != X86Reg::XMM1 {
            self.emit_movsd_rr(X86Reg::XMM1, arg2);
        }
        self.emit_call_extern(name);
        if dst != X86Reg::XMM0 {
            self.emit_movsd_rr(dst, X86Reg::XMM0);
        }
    }

    /// Emit a cast/conversion operation
    fn emit_cast(
        &mut self,
        result: ValueId,
        op: CastOp,
        val: ValueId,
        _to_ty: &SirType,
    ) -> Result<(), EmitError> {
        let src = self.get_value_reg(val, X86Reg::R10);
        let dst = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        match op {
            CastOp::Trunc => {
                // Truncate to smaller type - just move (high bits will be ignored)
                if dst != src {
                    self.emit_mov_rr(dst, src);
                }
            }
            CastOp::ZExt => {
                // Zero extend - for 32->64, the mov already zero-extends
                // For smaller sizes, use movzx
                if dst != src {
                    self.emit_mov_rr(dst, src);
                }
            }
            CastOp::SExt => {
                // Sign extend - MOVSXD for 32->64
                // MOVSXD r64, r32: REX.W + 63 /r
                self.emit_rex(true, dst.needs_rex(), false, src.needs_rex());
                self.emit_byte(0x63);
                self.emit_modrm(0b11, dst.encoding(), src.encoding());
            }
            CastOp::FPTrunc => {
                // Double to single: CVTSD2SS
                let src_xmm = self.get_value_reg(val, X86Reg::XMM15);
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                // CVTSD2SS: F2 0F 5A /r
                self.emit_byte(0xF2);
                if dst_xmm.needs_rex() || src_xmm.needs_rex() {
                    self.emit_rex(false, dst_xmm.needs_rex(), false, src_xmm.needs_rex());
                }
                self.emit_bytes(&[0x0F, 0x5A]);
                self.emit_modrm(0b11, (dst_xmm as u8) & 0x7, (src_xmm as u8) & 0x7);
            }
            CastOp::FPExt => {
                // Single to double: CVTSS2SD
                let src_xmm = self.get_value_reg(val, X86Reg::XMM15);
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                // CVTSS2SD: F3 0F 5A /r
                self.emit_byte(0xF3);
                if dst_xmm.needs_rex() || src_xmm.needs_rex() {
                    self.emit_rex(false, dst_xmm.needs_rex(), false, src_xmm.needs_rex());
                }
                self.emit_bytes(&[0x0F, 0x5A]);
                self.emit_modrm(0b11, (dst_xmm as u8) & 0x7, (src_xmm as u8) & 0x7);
            }
            CastOp::FPToSI => {
                // Float to signed int: CVTTSD2SI (truncate toward zero)
                let src_xmm = self.get_value_reg(val, X86Reg::XMM15);
                // CVTTSD2SI r64, xmm: F2 REX.W 0F 2C /r
                self.emit_byte(0xF2);
                self.emit_rex(true, dst.needs_rex(), false, src_xmm.needs_rex());
                self.emit_bytes(&[0x0F, 0x2C]);
                self.emit_modrm(0b11, dst.encoding(), (src_xmm as u8) & 0x7);
            }
            CastOp::FPToUI => {
                // Float to unsigned int - no direct instruction, use CVTTSD2SI
                // For values that fit, this works; for full range, need more complex code
                let src_xmm = self.get_value_reg(val, X86Reg::XMM15);
                self.emit_byte(0xF2);
                self.emit_rex(true, dst.needs_rex(), false, src_xmm.needs_rex());
                self.emit_bytes(&[0x0F, 0x2C]);
                self.emit_modrm(0b11, dst.encoding(), (src_xmm as u8) & 0x7);
            }
            CastOp::SIToFP => {
                // Signed int to float: CVTSI2SD
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                // CVTSI2SD xmm, r64: F2 REX.W 0F 2A /r
                self.emit_byte(0xF2);
                self.emit_rex(true, dst_xmm.needs_rex(), false, src.needs_rex());
                self.emit_bytes(&[0x0F, 0x2A]);
                self.emit_modrm(0b11, (dst_xmm as u8) & 0x7, src.encoding());
            }
            CastOp::UIToFP => {
                // Unsigned int to float - for values that fit in signed range, use CVTSI2SD
                let dst_xmm = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                self.emit_byte(0xF2);
                self.emit_rex(true, dst_xmm.needs_rex(), false, src.needs_rex());
                self.emit_bytes(&[0x0F, 0x2A]);
                self.emit_modrm(0b11, (dst_xmm as u8) & 0x7, src.encoding());
            }
            CastOp::PtrToInt => {
                // Pointer to integer - just move
                if dst != src {
                    self.emit_mov_rr(dst, src);
                }
            }
            CastOp::IntToPtr => {
                // Integer to pointer - just move
                if dst != src {
                    self.emit_mov_rr(dst, src);
                }
            }
            CastOp::BitCast => {
                // Reinterpret bits - move between register classes if needed
                let src_is_float = src.is_xmm();
                let dst_is_float = dst.is_xmm();
                if src_is_float && !dst_is_float {
                    // XMM -> GPR: MOVQ r64, xmm
                    self.emit_byte(0x66);
                    self.emit_rex(true, src.needs_rex(), false, dst.needs_rex());
                    self.emit_bytes(&[0x0F, 0x7E]); // MOVQ r/m64, xmm
                    self.emit_modrm(0b11, (src as u8) & 0x7, dst.encoding());
                } else if !src_is_float && dst_is_float {
                    // GPR -> XMM: MOVQ xmm, r64
                    self.emit_movq_xmm_r64(dst, src);
                } else if dst != src {
                    // Same register class
                    if dst.is_xmm() {
                        self.emit_movsd_rr(dst, src);
                    } else {
                        self.emit_mov_rr(dst, src);
                    }
                }
            }
        }

        Ok(())
    }

    /// Emit epistemic operations for Knowledge<T> types
    /// 
    /// Knowledge<T> is represented as a struct { value: T, confidence: f64 }
    /// This function generates code for epistemic operations while preserving
    /// confidence propagation through computations.
    fn emit_epistemic_op(
        &mut self,
        result: ValueId,
        op: &super::ops::EpistemicOp,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        use super::ops::EpistemicOp;
        
        match op {
            EpistemicOp::Create { value, confidence } => {
                // Create epistemic value: allocate struct { value, confidence }
                // OPTIMIZATION: For small types, keep value and confidence in adjacent registers
                // when possible (XMM0:XMM1 for floats, RAX:RDX for integers)
                
                // Get value and confidence registers
                let value_reg = self.get_value_reg(*value, X86Reg::RAX);
                let conf_reg = self.get_value_reg(*confidence, X86Reg::RDX);
                
                // Check if result type is epistemic
                let result_ty = self.get_value_type(result, func);
                if let SirType::Epistemic(inner_ty) = &result_ty {
                    let inner_size = inner_ty.size_bytes();
                    let is_float = matches!(&**inner_ty, SirType::Scalar(ScalarType::F32) | SirType::Scalar(ScalarType::F64));
                    
                    // OPTIMIZATION: For small types (<= 8 bytes), try to keep in register pairs
                    if inner_size <= 8 {
                        if is_float {
                            // Float value - optimize: keep in XMM0:XMM1 pair
                            // XMM0 = value, XMM1 = confidence
                            if value_reg != X86Reg::XMM0 {
                                self.emit_movsd_rr(X86Reg::XMM0, value_reg);
                            }
                            if conf_reg.is_xmm() {
                                if conf_reg != X86Reg::XMM1 {
                                    self.emit_movsd_rr(X86Reg::XMM1, conf_reg);
                                }
                            } else {
                                // Convert confidence from GPR to XMM1
                                self.emit_movq_xmm_r64(X86Reg::XMM1, conf_reg);
                            }
                            // Result is in XMM0 (value) with XMM1 (confidence) adjacent
                            // For now, we'll need to allocate on stack for the struct
                            // In future optimization, could track register pairs
                            self.emit_sub_rsp_imm(16);
                            self.emit_movsd_mem_disp_r64(X86Reg::RSP, 0, X86Reg::XMM0);
                            self.emit_movsd_mem_disp_r64(X86Reg::RSP, 8, X86Reg::XMM1);
                            let result_ptr_reg = self
                                .register_allocator
                                .get_reg(result)
                                .unwrap_or(X86Reg::RAX);
                            self.emit_mov_rr(result_ptr_reg, X86Reg::RSP);
                        } else {
                            // Integer value - optimize: keep in RAX:RDX pair
                            // RAX = value, RDX = confidence (as f64, but stored separately)
                            if value_reg != X86Reg::RAX {
                                self.emit_mov_rr(X86Reg::RAX, value_reg);
                            }
                            // Confidence is f64 - allocate on stack adjacent to value
                            self.emit_sub_rsp_imm(16);
                            // Store value at offset 0
                            self.emit_mov_mem_disp_r64(X86Reg::RSP, 0, X86Reg::RAX);
                            // Store confidence at offset 8
                            if conf_reg.is_xmm() {
                                self.emit_movsd_mem_disp_r64(X86Reg::RSP, 8, conf_reg);
                            } else {
                                // Convert confidence to f64 and store
                                // For now, assume it's already f64 - would need conversion
                                self.emit_mov_mem_disp_r64(X86Reg::RSP, 8, conf_reg);
                            }
                            let result_ptr_reg = self
                                .register_allocator
                                .get_reg(result)
                                .unwrap_or(X86Reg::RAX);
                            self.emit_mov_rr(result_ptr_reg, X86Reg::RSP);
                        }
                    } else {
                        // Large type - allocate on stack with value and confidence adjacent
                        let total_size = inner_size + 8; // value + confidence
                        let aligned_size = (total_size + 15) & !15;
                        self.emit_sub_rsp_imm(aligned_size as i32);
                        
                        // Store value at offset 0
                        // Store confidence at offset inner_size (adjacent in memory)
                        // This ensures cache locality
                        let value_ptr_reg = X86Reg::R10;
                        self.emit_mov_rr(value_ptr_reg, X86Reg::RSP);
                        
                        // Copy value to stack (would need type-specific code)
                        // For now, assume value is already in memory or register
                        
                        // Store confidence at offset inner_size
                        if conf_reg.is_xmm() {
                            self.emit_movsd_mem_disp_r64(X86Reg::RSP, inner_size as i32, conf_reg);
                        } else {
                            self.emit_mov_mem_disp_r64(X86Reg::RSP, inner_size as i32, conf_reg);
                        }
                        
                        let result_ptr_reg = self
                            .register_allocator
                            .get_reg(result)
                            .unwrap_or(X86Reg::RAX);
                        self.emit_mov_rr(result_ptr_reg, X86Reg::RSP);
                    }
                } else {
                    // Result is not epistemic - shouldn't happen, but handle gracefully
                    return Err(EmitError::UnsupportedInstruction(
                        format!("Create epistemic value but result type is not epistemic: {:?}", result_ty)
                    ));
                }
            }
            
            EpistemicOp::ExtractValue(epistemic_val) => {
                // Extract value from Knowledge<T> -> T
                // Load value field (offset 0) from epistemic struct
                let epistemic_reg = self.get_value_reg(*epistemic_val, X86Reg::R10);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::RAX);
                
                // Load value field
                if epistemic_reg.is_xmm() {
                    // Epistemic value is in XMM - value is in same register
                    self.emit_movsd_rr(dst_reg, epistemic_reg);
                } else {
                    // Epistemic value is pointer to struct - load value field
                    self.emit_mov_r64_mem_disp(dst_reg, epistemic_reg, 0);
                }
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::ExtractConfidence(epistemic_val) => {
                // Extract confidence from Knowledge<T> -> f64
                let epistemic_reg = self.get_value_reg(*epistemic_val, X86Reg::R10);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                
                // Load confidence field (offset = size of inner type)
                // For now, assume inner type is <= 8 bytes, so confidence is at offset 8
                if epistemic_reg.is_xmm() {
                    // Epistemic value is in XMM pair - confidence is in second register
                    // Would need to track which XMM register pair
                    // For now, assume we need to load from memory
                    self.emit_sub_rsp_imm(16);
                    self.emit_movsd_mem_disp_r64(X86Reg::RSP, 0, epistemic_reg);
                    self.emit_movsd_rr(dst_reg, X86Reg::XMM1); // Assume confidence in XMM1
                    self.emit_add_rsp_imm(16);
                } else {
                    // Load confidence from memory at offset 8
                    let inner_ty = self.get_value_type(*epistemic_val, func);
                    let offset = if let SirType::Epistemic(inner) = &inner_ty {
                        inner.size_bytes() as i32
                    } else {
                        8 // Default offset
                    };
                    self.emit_movsd_r64_mem_disp(dst_reg, epistemic_reg, offset);
                }
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::PropagateAdd { conf_a, conf_b } => {
                // Confidence propagation for addition: min(conf_a, conf_b)
                // This is the GUM (Guide to Uncertainty in Measurement) rule
                let conf_a_reg = self.get_value_reg(*conf_a, X86Reg::XMM0);
                let conf_b_reg = self.get_value_reg(*conf_b, X86Reg::XMM1);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM2);
                
                // MINSD dst, src (minimum of two doubles)
                // F2 0F 5D /r
                self.emit_byte(0xF2);
                self.emit_rex(false, false, false, false);
                self.emit_bytes(&[0x0F, 0x5D]);
                self.emit_modrm(0b11, (dst_reg as u8) & 0x7, (conf_b_reg as u8) & 0x7);
                
                // Move conf_a to dst first
                if conf_a_reg != dst_reg {
                    self.emit_movsd_rr(dst_reg, conf_a_reg);
                }
                // Then compute min(dst, conf_b)
                self.emit_minsd(dst_reg, conf_b_reg);
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::PropagateSub { conf_a, conf_b } => {
                // Confidence propagation for subtraction: min(conf_a, conf_b)
                // Same as addition (GUM rule)
                self.emit_epistemic_op(
                    result,
                    &EpistemicOp::PropagateAdd {
                        conf_a: *conf_a,
                        conf_b: *conf_b,
                    },
                    func,
                )?;
            }
            
            EpistemicOp::PropagateMul {
                val_a,
                conf_a,
                val_b,
                conf_b,
            } => {
                // Confidence propagation for multiplication: conf_a * conf_b
                // This follows GUM propagation rules
                let conf_a_reg = self.get_value_reg(*conf_a, X86Reg::XMM0);
                let conf_b_reg = self.get_value_reg(*conf_b, X86Reg::XMM1);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM2);
                
                // MULSD dst, src (multiply doubles)
                self.emit_movsd_rr(dst_reg, conf_a_reg);
                self.emit_mulsd(dst_reg, conf_b_reg);
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::PropagateDiv {
                val_a: _,
                conf_a,
                val_b: _,
                conf_b,
            } => {
                // Confidence propagation for division: conf_a * conf_b
                // Same as multiplication (GUM rule)
                let conf_a_reg = self.get_value_reg(*conf_a, X86Reg::XMM0);
                let conf_b_reg = self.get_value_reg(*conf_b, X86Reg::XMM1);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM2);
                
                // MULSD dst, src (multiply doubles)
                self.emit_movsd_rr(dst_reg, conf_a_reg);
                self.emit_mulsd(dst_reg, conf_b_reg);
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::FusedMul {
                val_a,
                conf_a,
                val_b,
                conf_b,
            } => {
                // Fused operation: compute both value * value and conf_a * conf_b
                // Optimization: keep both in XMM register pair
                let val_a_reg = self.get_value_reg(*val_a, X86Reg::XMM0);
                let val_b_reg = self.get_value_reg(*val_b, X86Reg::XMM1);
                let conf_a_reg = self.get_value_reg(*conf_a, X86Reg::XMM2);
                let conf_b_reg = self.get_value_reg(*conf_b, X86Reg::XMM3);
                
                // Compute value result: val_a * val_b
                let val_result_reg = X86Reg::XMM4;
                self.emit_movsd_rr(val_result_reg, val_a_reg);
                self.emit_mulsd(val_result_reg, val_b_reg);
                
                // Compute confidence result: conf_a * conf_b
                let conf_result_reg = X86Reg::XMM5;
                self.emit_movsd_rr(conf_result_reg, conf_a_reg);
                self.emit_mulsd(conf_result_reg, conf_b_reg);
                
                // Result is epistemic - would need to construct struct
                // For now, store both in result location (would need stack allocation)
                // In optimized version, keep in XMM register pair
                self.store_value(result, val_result_reg);
            }
            
            EpistemicOp::Meet { conf_a, conf_b } => {
                // Meet operation: min(conf_a, conf_b) - same as PropagateAdd
                self.emit_epistemic_op(
                    result,
                    &EpistemicOp::PropagateAdd {
                        conf_a: *conf_a,
                        conf_b: *conf_b,
                    },
                    func,
                )?;
            }
            
            EpistemicOp::Join { conf_a, conf_b } => {
                // Join operation: max(conf_a, conf_b) - for branch merging
                let conf_a_reg = self.get_value_reg(*conf_a, X86Reg::XMM0);
                let conf_b_reg = self.get_value_reg(*conf_b, X86Reg::XMM1);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM2);
                
                // Move conf_a to dst first
                if conf_a_reg != dst_reg {
                    self.emit_movsd_rr(dst_reg, conf_a_reg);
                }
                // Then compute max(dst, conf_b)
                self.emit_maxsd(dst_reg, conf_b_reg);
                
                self.store_value(result, dst_reg);
            }
            
            EpistemicOp::CondPropagate {
                condition_conf,
                then_conf,
                else_conf,
            } => {
                // Conditional confidence: if condition is certain (conf=1), use then_conf,
                // otherwise use weighted average or minimum
                // For now, use minimum of all three (conservative)
                let cond_reg = self.get_value_reg(*condition_conf, X86Reg::XMM0);
                let then_reg = self.get_value_reg(*then_conf, X86Reg::XMM1);
                let else_reg = self.get_value_reg(*else_conf, X86Reg::XMM2);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM3);
                
                // Compute min(then_conf, else_conf) first
                self.emit_movsd_rr(dst_reg, then_reg);
                self.emit_minsd(dst_reg, else_reg);
                
                // Then take minimum with condition confidence
                // (if condition is uncertain, result is also uncertain)
                self.emit_minsd(dst_reg, cond_reg);
                
                self.store_value(result, dst_reg);
            }
        }
        
        Ok(())
    }
    
    /// Helper: Get value type from function
    fn get_value_type(&self, value_id: ValueId, func: &SirFunction) -> SirType {
        // Try to find value in function parameters
        for (idx, (_, param_ty)) in func.params.iter().enumerate() {
            if ValueId::new(idx as u32) == value_id {
                return param_ty.clone();
            }
        }
        
        // Try to find in instructions
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    if result == value_id {
                        // Infer type from instruction
                        return self.infer_instruction_type(&inst.inst);
                    }
                }
            }
        }
        
        // Default to f64 if unknown
        SirType::Scalar(ScalarType::F64)
    }
    
    /// Helper: Get physical unit from value (for unit checking)
    fn get_value_unit(&self, value_id: ValueId, func: &SirFunction) -> Option<super::values::PhysicalUnit> {
        // Try to find value in function parameters
        for (idx, (_, _)) in func.params.iter().enumerate() {
            if ValueId::new(idx as u32) == value_id {
                // Would need access to parameter metadata - for now return None
                // In full implementation, would access func.params metadata
                return None;
            }
        }
        
        // Try to find in instructions
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    if result == value_id {
                        // Check if instruction has unit metadata
                        // In full implementation, would access inst.metadata.unit
                        return None;
                    }
                }
            }
        }
        
        None
    }
    
    /// Helper: Check if two units are compatible (same dimension)
    /// Zero-cost: this is compile-time only, no runtime overhead
    fn units_compatible(&self, lhs: &super::values::PhysicalUnit, rhs: &super::values::PhysicalUnit) -> bool {
        // Units are compatible if they have the same dimensions
        // Scale factors don't matter for compatibility (only for conversion)
        lhs.dimensions == rhs.dimensions
    }
    
    /// Helper: Infer type from instruction
    fn infer_instruction_type(&self, inst: &super::ops::SirInst) -> SirType {
        match inst {
            super::ops::SirInst::BinOp { op, .. } => match op {
                super::ops::ArithOp::FAdd
                | super::ops::ArithOp::FSub
                | super::ops::ArithOp::FMul
                | super::ops::ArithOp::FDiv
                | super::ops::ArithOp::FRem => SirType::f64(),
                _ => SirType::i64(),
            },
            super::ops::SirInst::Cmp { .. } => SirType::bool(),
            super::ops::SirInst::Cast { to_ty, .. } => to_ty.clone(),
            super::ops::SirInst::UnaryFloat { .. } | super::ops::SirInst::BinaryFloat { .. } => {
                SirType::f64()
            }
            super::ops::SirInst::Const(c) => c.ty(),
            super::ops::SirInst::Memory(super::ops::MemoryOp::Load { ty, .. }) => ty.clone(),
            super::ops::SirInst::Memory(super::ops::MemoryOp::Alloca { ty, .. }) => {
                SirType::ptr(ty.clone())
            }
            super::ops::SirInst::Memory(super::ops::MemoryOp::GetElementPtr { ty, .. }) => {
                SirType::ptr(ty.clone())
            }
            super::ops::SirInst::Call(info) => info.ret_ty.clone(),
            super::ops::SirInst::BuildAggregate { ty, .. } => ty.clone(),
            super::ops::SirInst::Phi { ty, .. } => ty.clone(),
            super::ops::SirInst::Select { .. } => SirType::i64(),
            super::ops::SirInst::Epistemic(_) => {
                SirType::Epistemic(Box::new(SirType::Scalar(ScalarType::F64)))
            }
            _ => SirType::i64(),
        }
    }

    /// Emit scientific operations (ODE, matrix ops, etc.)
    fn emit_scientific_op(
        &mut self,
        result: ValueId,
        op: &super::ops::ScientificOp,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        use super::ops::{OdeMethod, ScientificOp};
        
        match op {
            ScientificOp::OdeStep {
                method,
                state,
                derivatives,
                t,
                dt,
            } => {
                // Generate code for ODE step
                // For simple methods (Euler, RK4), inline the computation
                // For complex methods (DoPri5, CashKarp), call runtime
                
                match method {
                    OdeMethod::Euler => {
                        // Euler: y_new = y + dt * f(t, y)
                        // For now, call runtime - full implementation would inline
                        let state_reg = self.get_value_reg(*state, X86Reg::R10);
                        let t_reg = self.get_value_reg(*t, X86Reg::XMM0);
                        let dt_reg = self.get_value_reg(*dt, X86Reg::XMM1);
                        
                        // Setup arguments: state (RDI), t (XMM0), dt (XMM1)
                        if !state_reg.is_xmm() {
                            self.emit_mov_rr(X86Reg::RDI, state_reg);
                        }
                        if !t_reg.is_xmm() {
                            self.emit_movsd_rr(X86Reg::XMM0, t_reg);
                        }
                        if !dt_reg.is_xmm() {
                            self.emit_movsd_rr(X86Reg::XMM1, dt_reg);
                        }
                        
                        self.emit_call_extern("sounio_ode_euler");
                        
                        let result_reg = self
                            .register_allocator
                            .get_reg(result)
                            .unwrap_or(X86Reg::XMM0);
                        if result_reg != X86Reg::XMM0 {
                            self.emit_movsd_rr(result_reg, X86Reg::XMM0);
                        }
                    }
                    
                    OdeMethod::RK4 => {
                        // RK4: call runtime implementation
                        let state_reg = self.get_value_reg(*state, X86Reg::R10);
                        let t_reg = self.get_value_reg(*t, X86Reg::XMM0);
                        let dt_reg = self.get_value_reg(*dt, X86Reg::XMM1);
                        
                        if !state_reg.is_xmm() {
                            self.emit_mov_rr(X86Reg::RDI, state_reg);
                        }
                        if !t_reg.is_xmm() {
                            self.emit_movsd_rr(X86Reg::XMM0, t_reg);
                        }
                        if !dt_reg.is_xmm() {
                            self.emit_movsd_rr(X86Reg::XMM1, dt_reg);
                        }
                        
                        self.emit_call_extern("sounio_ode_rk4");
                        
                        let result_reg = self
                            .register_allocator
                            .get_reg(result)
                            .unwrap_or(X86Reg::XMM0);
                        if result_reg != X86Reg::XMM0 {
                            self.emit_movsd_rr(result_reg, X86Reg::XMM0);
                        }
                    }
                    
                    OdeMethod::DoPri5 | OdeMethod::CashKarp | OdeMethod::BDF | OdeMethod::LSODA => {
                        // ============================================================
                        // COMPLEX ADAPTIVE ODE SOLVERS (DoPri5, CashKarp)
                        // ============================================================
                        // These methods require full C runtime interface:
                        // C signature: double sounio_ode_*_step(
                        //     double* state,      // RDI: state vector (modified in-place)
                        //     int n,              // RSI: dimension
                        //     double* t,          // RDX: pointer to current time (updated)
                        //     double* dt,         // RCX: pointer to step size (updated)
                        //     double rtol,        // XMM0: relative tolerance
                        //     double atol,        // XMM1: absolute tolerance
                        //     DerivativeFn f      // R8: derivatives function pointer
                        // );
                        // Returns: error estimate in XMM0 (finite = success, INF = step rejected)
                        
                        // ============================================================
                        // STEP 1: Extract state dimension from type
                        // ============================================================
                        let state_type = self.get_value_type(*state, func);
                        let state_dim = match &state_type {
                            SirType::Array(arr) => {
                                // Direct array type - use array length
                                arr.len as i32
                            }
                            SirType::Pointer(ptr) => {
                                // Pointer type - check if it points to an array
                                match ptr.pointee.as_ref() {
                                    SirType::Array(arr) => arr.len as i32,
                                    _ => {
                                        // Pointer to scalar or unknown - default to 1
                                        // In production, would extract from metadata or type info
                                        1
                                    }
                                }
                            }
                            _ => {
                                // Scalar or unknown type - default to 1
                                // In production, would validate this is correct
                                1
                            }
                        };
                        
                        // Validate dimension is reasonable
                        if state_dim <= 0 || state_dim > 1_000_000 {
                            return Err(EmitError::InvalidInput(
                                format!("Invalid ODE state dimension: {}", state_dim)
                            ));
                        }
                        
                        // ============================================================
                        // STEP 2: Get registers for state, t, dt
                        // ============================================================
                        let state_reg = self.get_value_reg(*state, X86Reg::R10);
                        let t_reg = self.get_value_reg(*t, X86Reg::XMM2);
                        let dt_reg = self.get_value_reg(*dt, X86Reg::XMM3);
                        
                        // ============================================================
                        // STEP 3: Allocate stack space for t and dt (16 bytes aligned)
                        // ============================================================
                        // We need 16 bytes: 8 for t, 8 for dt
                        // Align to 16 bytes for System V ABI
                        self.emit_sub_rsp_imm(16);
                        
                        // ============================================================
                        // STEP 4: Store t and dt values on stack
                        // ============================================================
                        // Store t at [RSP] (must be f64 in XMM register)
                        if t_reg.is_xmm() {
                            // t is in XMM register - store directly
                            self.emit_movsd_mem_disp_r64(X86Reg::RSP, 0, t_reg);
                        } else {
                            // t is in integer register - this shouldn't happen for f64
                            // In production, would convert or handle error
                            // For now, attempt to store (may cause issues)
                            // TODO: Add proper conversion or error handling
                            return Err(EmitError::InvalidInput(
                                "ODE time value must be in XMM register (f64)".into()
                            ));
                        }
                        
                        // Store dt at [RSP+8] (must be f64 in XMM register)
                        if dt_reg.is_xmm() {
                            // dt is in XMM register - store directly
                            self.emit_movsd_mem_disp_r64(X86Reg::RSP, 8, dt_reg);
                        } else {
                            // dt is in integer register - this shouldn't happen for f64
                            return Err(EmitError::InvalidInput(
                                "ODE step size must be in XMM register (f64)".into()
                            ));
                        }
                        
                        // ============================================================
                        // STEP 5: Setup arguments for C calling convention (System V AMD64)
                        // ============================================================
                        // RDI = state pointer (must be in general-purpose register)
                        if state_reg.is_xmm() {
                            // State is in XMM register - this is invalid for pointers
                            // In production, would handle this error or convert
                            return Err(EmitError::InvalidInput(
                                "ODE state must be in general-purpose register (pointer)".into()
                            ));
                        } else {
                            // State is in GP register - move to RDI (first argument)
                            self.emit_mov_rr(X86Reg::RDI, state_reg);
                        }
                        
                        // RSI = n (dimension)
                        self.emit_mov_ri64(X86Reg::RSI, state_dim as i64);
                        
                        // RDX = t pointer (address of [RSP])
                        self.emit_lea(X86Reg::RDX, X86Reg::RSP, 0);
                        
                        // RCX = dt pointer (address of [RSP+8])
                        self.emit_lea(X86Reg::RCX, X86Reg::RSP, 8);
                        
                        // ============================================================
                        // STEP 6: Load tolerance constants (rtol, atol)
                        // ============================================================
                        // Extract tolerances from metadata if available
                        // OdeSolverMetadata contains tolerance information
                        // For now, use defaults; in production, extract from metadata
                        let rtol: f64 = 1e-6;
                        let atol: f64 = 1e-8;
                        
                        // TODO: Extract from OdeSolverMetadata if available
                        // Example:
                        // if let Some(metadata) = func.metadata.get(result) {
                        //     for meta in metadata {
                        //         if let Metadata::OdeSolver(ode_meta) = meta {
                        //             if let Some(tol) = ode_meta.tolerance {
                        //                 rtol = tol;
                        //                 atol = tol * 0.1; // Common pattern: atol = rtol * 0.1
                        //             }
                        //         }
                        //     }
                        // }
                        
                        // Load rtol into XMM0 using helper function
                        // Use RSP-based version since we're in the middle of function
                        self.emit_load_f64_const_rsp(X86Reg::XMM0, rtol.to_bits());
                        
                        // Load atol into XMM1 using helper function
                        self.emit_load_f64_const_rsp(X86Reg::XMM1, atol.to_bits());
                        
                        // ============================================================
                        // STEP 7: Get derivatives function pointer
                        // ============================================================
                        // The derivatives function ID is in the op
                        let derivatives_func_id = *derivatives;
                        
                        // Look up function name or address
                        // In production, this would resolve the function address via PLT
                        // For now, we'll use a relocation to resolve the function
                        if let Some(func_name) = self.func_names.get(&derivatives_func_id) {
                            // Function has a name - use relocation to resolve address
                            // Load function address into R8 via relocation
                            let offset = self.code.len() + 2; // +2 for MOV opcode and ModR/M
                            self.relocations.push(Relocation {
                                offset,
                                kind: RelocKind::PLT32,
                                symbol: func_name.clone(),
                                addend: -4,
                            });
                            // MOV R8, [RIP+rel32] - will be patched by linker
                            self.emit_rex(true, false, false, true); // REX.W + R8 needs REX.B
                            self.emit_byte(0x8B); // MOV r64, r/m64
                            self.emit_modrm(0b00, X86Reg::R8.encoding(), 0b101); // [RIP+disp32]
                            self.emit_u32(0); // Placeholder, will be patched
                        } else {
                            // Function doesn't have a name - use direct address if available
                            // For now, set to NULL (runtime will handle)
                            self.emit_mov_ri64(X86Reg::R8, 0);
                        }
                        
                        // ============================================================
                        // STEP 8: Call the ODE solver function
                        // ============================================================
                        let func_name = match method {
                            OdeMethod::DoPri5 => "sounio_ode_dopri5_step",
                            OdeMethod::CashKarp => "sounio_ode_cashkarp_step",
                            OdeMethod::BDF => "sounio_ode_bdf_step",
                            OdeMethod::LSODA => "sounio_ode_lsoda_step",
                            _ => "sounio_ode_step", // Fallback for Euler, Midpoint, RK4
                        };
                        
                        // All arguments are now set up:
                        // RDI = state pointer ✓
                        // RSI = n (dimension) ✓
                        // RDX = t pointer ✓
                        // RCX = dt pointer ✓
                        // XMM0 = rtol ✓
                        // XMM1 = atol ✓
                        // R8 = derivatives function pointer ✓

                        // For BDF and LSODA, we need R9 = Jacobian function pointer
                        // Currently set to NULL (0) for numerical Jacobian
                        if matches!(method, OdeMethod::BDF | OdeMethod::LSODA) {
                            // R9 = NULL (use numerical Jacobian)
                            // TODO: Support user-provided Jacobian functions
                            self.emit_mov_ri64(X86Reg::R9, 0);
                        }

                        // Call the function
                        self.emit_call_extern(func_name);
                        
                        // ============================================================
                        // STEP 9: Clean up stack and handle return value
                        // ============================================================
                        // Clean up stack: 16 bytes for t/dt
                        self.emit_add_rsp_imm(16);
                        
                        // Return value (error estimate) is in XMM0
                        let result_reg = self
                            .register_allocator
                            .get_reg(result)
                            .unwrap_or(X86Reg::XMM0);
                        if result_reg != X86Reg::XMM0 {
                            self.emit_movsd_rr(result_reg, X86Reg::XMM0);
                        }
                    }
                    
                    _ => {
                        self.emit_call_extern("sounio_ode_step");
                    }
                }
                
                self.store_value(result, X86Reg::XMM0);
            }
            
            ScientificOp::DotProduct { a, b, len, .. } => {
                // Dot product: optimize with SIMD when len >= 4
                let a_reg = self.get_value_reg(*a, X86Reg::R10);
                let b_reg = self.get_value_reg(*b, X86Reg::R11);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM0);
                
                if *len >= 4 {
                    // Use SIMD
                    if !a_reg.is_xmm() {
                        self.emit_mov_rr(X86Reg::RDI, a_reg);
                    }
                    if !b_reg.is_xmm() {
                        self.emit_mov_rr(X86Reg::RSI, b_reg);
                    }
                    self.emit_mov_ri64(X86Reg::RDX, *len as i64);
                    self.emit_call_extern("sounio_dot_product_simd");
                } else {
                    // Small vectors - scalar
                    if !a_reg.is_xmm() {
                        self.emit_mov_rr(X86Reg::RDI, a_reg);
                    }
                    if !b_reg.is_xmm() {
                        self.emit_mov_rr(X86Reg::RSI, b_reg);
                    }
                    self.emit_mov_ri64(X86Reg::RDX, *len as i64);
                    self.emit_call_extern("sounio_dot_product");
                }
                
                if dst_reg != X86Reg::XMM0 {
                    self.emit_movsd_rr(dst_reg, X86Reg::XMM0);
                }
                self.store_value(result, dst_reg);
            }
            
            ScientificOp::MatVecMul {
                matrix,
                vector,
                rows,
                cols,
            } => {
                // Matrix-vector multiplication: y = A @ x
                // Call sounio_tensor_matvec(A, x, y, M, N)
                let matrix_reg = self.get_value_reg(*matrix, X86Reg::R10);
                let vector_reg = self.get_value_reg(*vector, X86Reg::R11);
                
                // Setup arguments for sounio_tensor_matvec:
                // RDI = A (matrix pointer)
                // RSI = x (vector pointer)
                // RDX = y (output vector pointer - need to allocate)
                // RCX = M (rows)
                // R8 = N (columns)
                
                // Move matrix pointer to RDI
                if !matrix_reg.is_xmm() {
                    self.emit_mov_rr(X86Reg::RDI, matrix_reg);
                } else {
                    return Err(EmitError::InvalidInput(
                        "Matrix pointer must be in general-purpose register".into()
                    ));
                }
                
                // Move vector pointer to RSI
                if !vector_reg.is_xmm() {
                    self.emit_mov_rr(X86Reg::RSI, vector_reg);
                } else {
                    return Err(EmitError::InvalidInput(
                        "Vector pointer must be in general-purpose register".into()
                    ));
                }
                
                // Allocate output vector on stack
                // Size: rows * 8 bytes (f64)
                let output_size = (*rows as usize) * 8;
                self.emit_sub_rsp_imm(output_size as i32);
                
                // Pass output pointer (RSP) to RDX
                self.emit_lea(X86Reg::RDX, X86Reg::RSP, 0);
                
                // Pass rows (M) to RCX
                self.emit_mov_ri64(X86Reg::RCX, *rows as i64);
                
                // Pass columns (N) to R8
                self.emit_mov_ri64(X86Reg::R8, *cols as i64);
                
                // Call tensor matvec function
                self.emit_call_extern("sounio_tensor_matvec");
                
                // Result vector is at [RSP]
                // For now, we'll leave it on the stack and return a pointer
                // In production, would copy to heap or use result register
                
                // Store result pointer (RSP) in result register
                let result_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::RAX);
                if result_reg != X86Reg::RSP {
                    self.emit_mov_rr(result_reg, X86Reg::RSP);
                }
                
                // Note: Stack space will be cleaned up in epilogue
                // For now, we assume the result is used immediately
            }
            
            ScientificOp::Autodiff {
                mode,
                function,
                input,
                output: _,
            } => {
                // Automatic differentiation codegen
                // Forward mode: uses dual numbers
                // Reverse mode: uses tape-based backpropagation
                
                match mode {
                    super::ops::AutodiffMode::Forward => {
                        // Forward-mode autodiff using dual numbers
                        // For now, call the runtime function
                        // In production, would inline dual number operations when possible
                        
                        // Get input value register
                        let input_reg = self.get_value_reg(*input, X86Reg::XMM0);
                        
                        // Allocate stack space for gradient result
                        self.emit_sub_rsp_imm(8);
                        
                        // Setup arguments for sounio_autodiff_forward:
                        // RDI = function pointer
                        // XMM0 = input value
                        // RSI = gradient pointer (on stack)
                        
                        // Get function pointer via relocation
                        // Look up function name
                        if let Some(func_name) = self.func_names.get(function) {
                            // Create relocation to load function address
                            let offset = self.code.len() + 2; // +2 for MOV opcode and ModR/M
                            self.relocations.push(Relocation {
                                offset,
                                kind: RelocKind::PLT32,
                                symbol: func_name.clone(),
                                addend: -4,
                            });
                            
                            // MOV RDI, [RIP+rel32] - will be patched by linker
                            self.emit_rex(true, false, false, false); // REX.W
                            self.emit_byte(0x8B); // MOV r64, r/m64
                            self.emit_modrm(0b00, X86Reg::RDI.encoding(), 0b101); // [RIP+disp32]
                            self.emit_u32(0); // Placeholder, will be patched
                        } else {
                            // Function doesn't have a name - set to NULL
                            self.emit_mov_ri64(X86Reg::RDI, 0);
                        }
                        
                        // Load input value into XMM0
                        if input_reg.is_xmm() {
                            if input_reg != X86Reg::XMM0 {
                                self.emit_movsd_rr(X86Reg::XMM0, input_reg);
                            }
                        } else {
                            // Input is in integer register - this shouldn't happen for f64
                            return Err(EmitError::InvalidInput(
                                "Autodiff input must be in XMM register (f64)".into()
                            ));
                        }
                        
                        // Pass gradient pointer (RSP)
                        self.emit_lea(X86Reg::RSI, X86Reg::RSP, 0);
                        
                        // Call forward-mode autodiff
                        self.emit_call_extern("sounio_autodiff_forward");
                        
                        // Function value is in XMM0 (return value)
                        // Gradient is at [RSP]
                        
                        // Load gradient from stack if needed
                        // For now, assume result is the function value
                        // In production, would handle gradient separately
                        
                        // Clean up stack
                        self.emit_add_rsp_imm(8);
                        
                        // Result (function value) is in XMM0
                        let result_reg = self
                            .register_allocator
                            .get_reg(result)
                            .unwrap_or(X86Reg::XMM0);
                        if result_reg != X86Reg::XMM0 {
                            self.emit_movsd_rr(result_reg, X86Reg::XMM0);
                        }
                    }
                    
                    super::ops::AutodiffMode::Reverse => {
                        // Reverse-mode autodiff using tape
                        // For now, call the runtime function
                        // In production, would build tape during forward pass
                        
                        // TODO: Implement tape-based reverse mode
                        // This requires:
                        // 1. Building a computation tape during forward pass
                        // 2. Storing tape pointer
                        // 3. Calling sounio_autodiff_reverse with tape and output gradient
                        
                        // For now, return error - reverse mode needs more infrastructure
                        return Err(EmitError::UnsupportedInstruction(
                            "Reverse-mode autodiff not yet implemented in codegen".into()
                        ));
                    }
                }
            }
            
            ScientificOp::Lerp { a, b, t } => {
                // Linear interpolation: a + t * (b - a)
                let a_reg = self.get_value_reg(*a, X86Reg::XMM0);
                let b_reg = self.get_value_reg(*b, X86Reg::XMM1);
                let t_reg = self.get_value_reg(*t, X86Reg::XMM2);
                let dst_reg = self
                    .register_allocator
                    .get_reg(result)
                    .unwrap_or(X86Reg::XMM3);
                
                // result = a + t * (b - a)
                self.emit_movsd_rr(dst_reg, b_reg);
                self.emit_subsd(dst_reg, a_reg);  // (b - a)
                self.emit_mulsd(dst_reg, t_reg);  // t * (b - a)
                self.emit_addsd(dst_reg, a_reg);   // a + t * (b - a)
                
                self.store_value(result, dst_reg);
            }

            _ => {
                return Err(EmitError::UnsupportedInstruction(
                    format!("Scientific operation {:?} not yet implemented", op)
                ));
            }
        }
        
        Ok(())
    }

    /// Emit a function call
    fn emit_call(
        &mut self,
        result: Option<ValueId>,
        info: &CallInfo,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        if let Callee::Named(name) = &info.callee {
            if name == "print" || name == "println" {
                return self.emit_builtin_print(name, info, func);
            }
            if name == "len" || name == "array_len" {
                return self.emit_builtin_len(result, info, func);
            }
        }

        // Set up arguments according to calling convention
        self.setup_call_args(&info.args);

        // Emit the call
        match &info.callee {
            Callee::Named(name) => {
                // Add relocation for external call
                let offset = self.offset() + 1; // +1 for the E8 opcode
                self.relocations.push(Relocation {
                    offset,
                    kind: RelocKind::PLT32,
                    symbol: name.clone(),
                    addend: -4, // PC-relative adjustment
                });
                self.emit_call_rel32(0); // Placeholder, will be patched
            }
            Callee::Indirect(ptr) => {
                let ptr_reg = self.get_value_reg(*ptr, X86Reg::R11);
                self.emit_call_reg(ptr_reg);
            }
            Callee::Direct(func_id) => {
                // Internal function call - use PLT32 for PIC compatibility
                // PLT32 works for both executables and shared libraries
                if let Some(name) = self.func_names.get(func_id) {
                    let offset = self.offset() + 1; // +1 for the E8 opcode
                    self.relocations.push(Relocation {
                        offset,
                        kind: RelocKind::PLT32,
                        symbol: name.clone(),
                        addend: -4, // PC-relative adjustment
                    });
                }
                self.emit_call_rel32(0); // Placeholder, will be patched by linker
            }
        }

        // Store return value if present
        if let Some(result_id) = result {
            // Return value is in RAX (integer) or XMM0 (float)
            let is_float = info.ret_ty.is_float();
            let ret_reg = if is_float { X86Reg::XMM0 } else { X86Reg::RAX };
            self.store_value(result_id, ret_reg);
        }

        Ok(())
    }

    fn emit_builtin_print(
        &mut self,
        name: &str,
        info: &CallInfo,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        let arg = match info.args.first() {
            Some(arg) => *arg,
            None => {
                if name == "println" {
                    self.emit_call_extern("_demetrios_print_newline");
                }
                return Ok(());
            }
        };

        let arg_ty = self.get_value_type(arg, func);
        match arg_ty {
            SirType::Scalar(ScalarType::F32) => {
                let src = self.get_value_reg(arg, X86Reg::XMM0);
                if src != X86Reg::XMM0 {
                    self.emit_cvtss2sd(X86Reg::XMM0, src);
                } else {
                    self.emit_cvtss2sd(X86Reg::XMM0, X86Reg::XMM0);
                }
                self.emit_call_extern("_demetrios_print_f64");
            }
            SirType::Scalar(ScalarType::F64) => {
                let src = self.get_value_reg(arg, X86Reg::XMM0);
                if src != X86Reg::XMM0 {
                    self.emit_movsd_rr(X86Reg::XMM0, src);
                }
                self.emit_call_extern("_demetrios_print_f64");
            }
            SirType::Scalar(ScalarType::Bool) => {
                let src = self.get_value_reg(arg, X86Reg::RDI);
                if src != X86Reg::RDI {
                    self.emit_mov_rr(X86Reg::RDI, src);
                }
                self.emit_call_extern("_demetrios_print_bool");
            }
            SirType::Array(arr) => {
                if matches!(arr.elem.as_ref(), SirType::Scalar(ScalarType::I8)) {
                    let src = self.get_value_reg(arg, X86Reg::RDI);
                    if src != X86Reg::RDI {
                        self.emit_mov_rr(X86Reg::RDI, src);
                    }
                    self.emit_mov_ri64(X86Reg::RSI, arr.len as i64);
                    self.emit_call_extern("_demetrios_print");
                } else {
                    let src = self.get_value_reg(arg, X86Reg::RDI);
                    if src != X86Reg::RDI {
                        self.emit_mov_rr(X86Reg::RDI, src);
                    }
                    self.emit_call_extern("_demetrios_print_i64");
                }
            }
            SirType::Pointer(ptr) => {
                if let SirType::Array(arr) = ptr.pointee.as_ref() {
                    let src = self.get_value_reg(arg, X86Reg::RDI);
                    if src != X86Reg::RDI {
                        self.emit_mov_rr(X86Reg::RDI, src);
                    }
                    self.emit_mov_ri64(X86Reg::RSI, arr.len as i64);
                    self.emit_call_extern("_demetrios_print");
                } else {
                    let src = self.get_value_reg(arg, X86Reg::RDI);
                    if src != X86Reg::RDI {
                        self.emit_mov_rr(X86Reg::RDI, src);
                    }
                    self.emit_call_extern("_demetrios_print_i64");
                }
            }
            _ => {
                let src = self.get_value_reg(arg, X86Reg::RDI);
                if src != X86Reg::RDI {
                    self.emit_mov_rr(X86Reg::RDI, src);
                }
                self.emit_call_extern("_demetrios_print_i64");
            }
        }

        if name == "println" {
            self.emit_call_extern("_demetrios_print_newline");
        }

        Ok(())
    }

    fn emit_builtin_len(
        &mut self,
        result: Option<ValueId>,
        info: &CallInfo,
        func: &SirFunction,
    ) -> Result<(), EmitError> {
        let Some(result_id) = result else {
            return Ok(());
        };
        let Some(arg) = info.args.first() else {
            return Ok(());
        };

        let arg_ty = self.get_value_type(*arg, func);
        let len = match arg_ty {
            SirType::Array(arr) => arr.len as i64,
            SirType::Pointer(ptr) => match ptr.pointee.as_ref() {
                SirType::Array(arr) => arr.len as i64,
                _ => 0,
            },
            _ => 0,
        };

        let dst = self
            .register_allocator
            .get_reg(result_id)
            .unwrap_or(X86Reg::RAX);
        self.emit_mov_ri64(dst, len);
        self.store_value(result_id, dst);

        Ok(())
    }

    /// Emit a memory operation
    fn emit_memory_op(&mut self, result: Option<ValueId>, op: &MemoryOp) -> Result<(), EmitError> {
        match op {
            MemoryOp::Load { ptr, ty, .. } => {
                if let Some(result_id) = result {
                    let ptr_reg = self.get_value_reg(*ptr, X86Reg::R10);
                    let dst_reg = self
                        .register_allocator
                        .get_reg(result_id)
                        .unwrap_or(X86Reg::RAX);

                    if ty.is_float() {
                        // MOVSD dst, [ptr]
                        self.emit_byte(0xF2);
                        self.emit_bytes(&[0x0F, 0x10]);
                        self.emit_modrm(0b00, (dst_reg as u8) & 0x7, ptr_reg.encoding());
                    } else {
                        // MOV dst, [ptr]
                        self.emit_mov_r64_mem_disp(dst_reg, ptr_reg, 0);
                    }
                }
            }
            MemoryOp::Store { ptr, val, .. } => {
                let ptr_reg = self.get_value_reg(*ptr, X86Reg::R10);
                let val_reg = self.get_value_reg(*val, X86Reg::R11);

                // MOV [ptr], val
                self.emit_mov_mem_disp_r64(ptr_reg, 0, val_reg);
            }
            MemoryOp::Alloca { ty, .. } => {
                // Stack allocation - adjust RSP and return pointer
                if let Some(result_id) = result {
                    let size = ty.size_bytes();
                    let aligned_size = (size + 15) & !15;
                    self.emit_sub_rsp_imm(aligned_size as i32);
                    let dst_reg = self
                        .register_allocator
                        .get_reg(result_id)
                        .unwrap_or(X86Reg::RAX);
                    self.emit_mov_rr(dst_reg, X86Reg::RSP);
                }
            }
            MemoryOp::GetElementPtr { ptr, ty, indices } => {
                // Compute address: base + sum(indices * element_sizes)
                if let Some(result_id) = result {
                    let base_reg = self.get_value_reg(*ptr, X86Reg::R10);
                    let dst_reg = self
                        .register_allocator
                        .get_reg(result_id)
                        .unwrap_or(X86Reg::RAX);

                    // Start with base pointer
                    self.emit_mov_rr(dst_reg, base_reg);

                    let mut current_ty = ty.clone();
                    for index in indices {
                        let elem_size = match &current_ty {
                            SirType::Array(arr_ty) => {
                                let size = arr_ty.elem.size_bytes();
                                current_ty = (*arr_ty.elem).clone();
                                size
                            }
                            SirType::Struct(struct_ty) => {
                                // For struct, index is field number - compute offset
                                if let GepIndex::Const(idx) = index {
                                    if let Some(field) = struct_ty.fields.get(*idx as usize) {
                                        // Use pre-computed field offset
                                        let offset = field.offset;
                                        current_ty = field.ty.clone();
                                        if offset > 0 {
                                            self.emit_add_ri32(dst_reg, offset as i32);
                                        }
                                        continue;
                                    }
                                }
                                8 // Default element size
                            }
                            SirType::Pointer(ptr_ty) => {
                                let size = ptr_ty.pointee.size_bytes();
                                current_ty = (*ptr_ty.pointee).clone();
                                size
                            }
                            _ => 8, // Default to 8 bytes
                        };

                        match index {
                            GepIndex::Const(c) => {
                                let offset = (*c as usize) * elem_size;
                                if offset != 0 {
                                    self.emit_add_ri32(dst_reg, offset as i32);
                                }
                            }
                            GepIndex::Value(v) => {
                                // Dynamic index: dst += index * elem_size
                                let mut idx_reg = self.get_value_reg(*v, X86Reg::R11);
                                if elem_size == 1 {
                                    // Just add index
                                    self.emit_add_rr(dst_reg, idx_reg);
                                } else if matches!(elem_size, 2 | 4 | 8) {
                                    if idx_reg == dst_reg {
                                        let scratch = if dst_reg == X86Reg::R11 {
                                            X86Reg::R10
                                        } else {
                                            X86Reg::R11
                                        };
                                        self.emit_mov_rr(scratch, idx_reg);
                                        idx_reg = scratch;
                                    }
                                    self.emit_lea_sib(dst_reg, dst_reg, idx_reg, elem_size as u8, 0);
                                } else {
                                    // Use IMUL for non-power-of-two sizes
                                    self.emit_mov_ri64(X86Reg::RAX, elem_size as i64);
                                    self.emit_imul_rr(X86Reg::RAX, idx_reg);
                                    self.emit_add_rr(dst_reg, X86Reg::RAX);
                                }
                            }
                        }
                    }
                }
            }
            MemoryOp::Memcpy {
                dst, src, len, ..
            } => {
                // Call memcpy(dst, src, len)
                // System V AMD64: RDI = dst, RSI = src, RDX = len
                let dst_reg = self.get_value_reg(*dst, X86Reg::RDI);
                let src_reg = self.get_value_reg(*src, X86Reg::RSI);
                let len_reg = self.get_value_reg(*len, X86Reg::RDX);

                // Move to ABI registers if not already there
                if dst_reg != X86Reg::RDI {
                    self.emit_mov_rr(X86Reg::RDI, dst_reg);
                }
                if src_reg != X86Reg::RSI {
                    self.emit_mov_rr(X86Reg::RSI, src_reg);
                }
                if len_reg != X86Reg::RDX {
                    self.emit_mov_rr(X86Reg::RDX, len_reg);
                }

                // Call memcpy
                self.emit_call_extern("memcpy");
            }
            MemoryOp::Memset {
                dst, val, len, ..
            } => {
                // Call memset(dst, val, len)
                // System V AMD64: RDI = dst, RSI = val, RDX = len
                let dst_reg = self.get_value_reg(*dst, X86Reg::RDI);
                let val_reg = self.get_value_reg(*val, X86Reg::RSI);
                let len_reg = self.get_value_reg(*len, X86Reg::RDX);

                // Move to ABI registers if not already there
                if dst_reg != X86Reg::RDI {
                    self.emit_mov_rr(X86Reg::RDI, dst_reg);
                }
                if val_reg != X86Reg::RSI {
                    self.emit_mov_rr(X86Reg::RSI, val_reg);
                }
                if len_reg != X86Reg::RDX {
                    self.emit_mov_rr(X86Reg::RDX, len_reg);
                }

                // Call memset
                self.emit_call_extern("memset");
            }
        }
        Ok(())
    }

    /// Emit a select (ternary) operation
    fn emit_select(
        &mut self,
        result: ValueId,
        cond: ValueId,
        then_val: ValueId,
        else_val: ValueId,
    ) -> Result<(), EmitError> {
        let cond_reg = self.get_value_reg(cond, X86Reg::R10);
        let then_reg = self.get_value_reg(then_val, X86Reg::R11);
        let else_reg = self.get_value_reg(else_val, X86Reg::RAX);
        let dst_reg = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        // Test condition
        self.emit_cmp_rr(cond_reg, cond_reg); // Will set ZF based on cond
        // We actually need: test cond_reg, cond_reg
        // For now, emit CMOVZ (move if zero)
        self.emit_mov_rr(dst_reg, then_reg);
        // CMOVZ dst, else - if condition was 0, use else value
        // 0F 44 /r - CMOVZ
        self.emit_rex_w(dst_reg, else_reg);
        self.emit_bytes(&[0x0F, 0x44]);
        self.emit_modrm(0b11, dst_reg.encoding(), else_reg.encoding());

        Ok(())
    }

    /// Emit a terminator instruction
    fn emit_terminator(
        &mut self,
        term: &Terminator,
        func: &SirFunction,
        callee_saved: &[X86Reg],
    ) -> Result<(), EmitError> {
        match term {
            Terminator::Return(None) => {
                if matches!(func.ret_ty, SirType::Void) {
                    self.emit_xor_rr_32(X86Reg::RAX, X86Reg::RAX);
                }
                self.emit_epilogue_with_restores(callee_saved);
            }
            Terminator::Return(Some(val)) => {
                // Move return value to RAX (integer) or XMM0 (float)
                let val_reg = self.get_value_reg(*val, X86Reg::RAX);
                if val_reg != X86Reg::RAX {
                    // Check if it's a float value
                    if val_reg.is_xmm() {
                        if val_reg != X86Reg::XMM0 {
                            self.emit_movsd_rr(X86Reg::XMM0, val_reg);
                        }
                    } else {
                        self.emit_mov_rr(X86Reg::RAX, val_reg);
                    }
                }
                self.emit_epilogue_with_restores(callee_saved);
            }
            Terminator::Br(target) => {
                if let Some(curr_block) = self.current_block {
                    self.emit_phi_moves_for_edge(curr_block, *target, func)?;
                }
                // Unconditional branch
                if let Some(&target_offset) = self.labels.get(&target.0) {
                    let rel = (target_offset as i32) - (self.offset() as i32) - 5;
                    self.emit_jmp_rel32(rel);
                } else {
                    // Forward reference - emit placeholder
                    self.forward_refs.push((self.offset() + 1, target.0));
                    self.emit_jmp_rel32(0);
                }
            }
            Terminator::CondBr {
                cond,
                then_block,
                else_block,
            } => {
                let cond_reg = self.get_value_reg(*cond, X86Reg::R10);

                // TEST cond, cond
                self.emit_rex_w(cond_reg, cond_reg);
                self.emit_byte(0x85);
                self.emit_modrm(0b11, cond_reg.encoding(), cond_reg.encoding());

                let curr_block = match self.current_block {
                    Some(id) => id,
                    None => {
                        return Ok(());
                    }
                };

                // JNZ then_path (patched after else path is emitted)
                let jcc_disp_offset = self.offset() + 2;
                self.emit_jcc_rel32(CondCode::NE as u8, 0);

                // Else path: phi moves then jump to else block
                self.emit_phi_moves_for_edge(curr_block, *else_block, func)?;
                if let Some(&else_offset) = self.labels.get(&else_block.0) {
                    let rel = (else_offset as i32) - (self.offset() as i32) - 5;
                    self.emit_jmp_rel32(rel);
                } else {
                    self.forward_refs.push((self.offset() + 1, else_block.0));
                    self.emit_jmp_rel32(0);
                }

                // Then path label is here
                let then_offset = self.offset();
                let rel = (then_offset as i32) - (jcc_disp_offset as i32) - 4;
                self.patch_rel32(jcc_disp_offset, rel);

                // Then path: phi moves then jump to then block
                self.emit_phi_moves_for_edge(curr_block, *then_block, func)?;
                if let Some(&then_block_offset) = self.labels.get(&then_block.0) {
                    let rel = (then_block_offset as i32) - (self.offset() as i32) - 5;
                    self.emit_jmp_rel32(rel);
                } else {
                    self.forward_refs.push((self.offset() + 1, then_block.0));
                    self.emit_jmp_rel32(0);
                }
            }
            Terminator::Unreachable => {
                // UD2 instruction (undefined - will trap)
                self.emit_bytes(&[0x0F, 0x0B]);
            }
            Terminator::Switch {
                val,
                default,
                cases,
            } => {
                // Emit series of compare-and-branch for each case
                let val_reg = self.get_value_reg(*val, X86Reg::R10);

                for (case_val, case_block) in cases {
                    // CMP val, case_val
                    self.emit_mov_ri64(X86Reg::R11, *case_val);
                    self.emit_cmp_rr(val_reg, X86Reg::R11);

                    // JE case_block
                    if let Some(&target_offset) = self.labels.get(&case_block.0) {
                        let rel = (target_offset as i32) - (self.offset() as i32) - 6;
                        self.emit_jcc_rel32(CondCode::E as u8, rel);
                    } else {
                        self.forward_refs.push((self.offset() + 2, case_block.0));
                        self.emit_jcc_rel32(CondCode::E as u8, 0);
                    }
                }

                // JMP default
                if let Some(&default_offset) = self.labels.get(&default.0) {
                    let rel = (default_offset as i32) - (self.offset() as i32) - 5;
                    self.emit_jmp_rel32(rel);
                } else {
                    self.forward_refs.push((self.offset() + 1, default.0));
                    self.emit_jmp_rel32(0);
                }
            }
            Terminator::TailCall { callee, args } => {
                // For tail calls, we reuse the current stack frame
                // Setup args first using System V AMD64 ABI
                let arg_regs = [
                    X86Reg::RDI,
                    X86Reg::RSI,
                    X86Reg::RDX,
                    X86Reg::RCX,
                    X86Reg::R8,
                    X86Reg::R9,
                ];

                for (i, arg) in args.iter().enumerate() {
                    if i < arg_regs.len() {
                        let arg_reg = self.get_value_reg(*arg, arg_regs[i]);
                        if arg_reg != arg_regs[i] {
                            self.emit_mov_rr(arg_regs[i], arg_reg);
                        }
                    }
                }

                // Restore callee-saved registers and RBP before jump
                self.emit_epilogue_with_restores(callee_saved);

                // Jump (not call) to target
                match callee {
                    crate::sir::ops::Callee::Named(name) => {
                        // JMP via PLT
                        let offset = self.code.len() + 1;
                        self.relocations.push(Relocation {
                            offset,
                            kind: RelocKind::PLT32,
                            symbol: name.clone(),
                            addend: -4,
                        });
                        self.emit_jmp_rel32(0);
                    }
                    crate::sir::ops::Callee::Indirect(ptr) => {
                        // JMP *ptr
                        let ptr_reg = self.get_value_reg(*ptr, X86Reg::R10);
                        // JMP r/m64: FF /4
                        self.emit_rex(true, false, false, ptr_reg.needs_rex());
                        self.emit_byte(0xFF);
                        self.emit_modrm(0b11, 4, ptr_reg.encoding());
                    }
                    crate::sir::ops::Callee::Direct(func_id) => {
                        // Direct tail call - use PLT32 for PIC compatibility
                        if let Some(name) = self.func_names.get(func_id) {
                            let offset = self.code.len() + 1;
                            self.relocations.push(Relocation {
                                offset,
                                kind: RelocKind::PLT32,
                                symbol: name.clone(),
                                addend: -4,
                            });
                        }
                        self.emit_jmp_rel32(0);
                    }
                }
            }
        }
        Ok(())
    }

    /// Patch forward references after all blocks are emitted
    fn patch_forward_refs(&mut self) {
        for (patch_offset, block_id) in &self.forward_refs {
            if let Some(&target_offset) = self.labels.get(block_id) {
                let rel = (target_offset as i32) - (*patch_offset as i32) - 4;
                let bytes = rel.to_le_bytes();
                self.code[*patch_offset] = bytes[0];
                self.code[*patch_offset + 1] = bytes[1];
                self.code[*patch_offset + 2] = bytes[2];
                self.code[*patch_offset + 3] = bytes[3];
            }
        }
    }
}

// ============================================================================
// PUBLIC API
// ============================================================================

/// Emit code for a SIR module
pub fn emit_code(
    module: &SirModule,
    alloc_results: Option<std::collections::HashMap<crate::sir::values::FuncId, crate::backend::native::alloc::AllocResult>>,
) -> Result<CodeSegment, EmitError> {
    match module.target.arch {
        Architecture::X86_64 => {
            let mut emitter = X86_64Emitter::new();
            emitter.emit_module_with_alloc_results(module, alloc_results)
        }
        Architecture::AArch64 => {
            // TODO: Implement AArch64 emitter
            Err(EmitError::UnsupportedTarget(
                "AArch64 not yet implemented".into(),
            ))
        }
        arch => Err(EmitError::UnsupportedTarget(format!("{:?}", arch))),
    }
}

/// Emit code for a SIR module and wrap it in ELF format
pub fn emit_code_to_elf(
    module: &SirModule,
    alloc_results: Option<std::collections::HashMap<crate::sir::values::FuncId, crate::backend::native::alloc::AllocResult>>,
) -> Result<Vec<u8>, EmitError> {
    let code_segment = emit_code(module, alloc_results)?;
    code_segment_to_elf(&code_segment)
}

/// Convert a CodeSegment to ELF object file bytes
pub fn code_segment_to_elf(segment: &CodeSegment) -> Result<Vec<u8>, EmitError> {
    use crate::backend::native::elf::{ElfConfig, ElfWriter, RelocationType};
    use std::collections::HashMap;

    let mut elf = ElfWriter::new(ElfConfig::default());

    // Add the text section with machine code
    let text_section_idx = elf.add_text_section(&segment.code);

    // Build symbol name -> index map
    let mut symbol_indices: HashMap<String, usize> = HashMap::new();

    // Add defined symbols
    // Note: ELF symbol table has null symbol at index 0, so actual symbols start at 1
    for sym in &segment.symbols {
        let idx = elf.add_function(&sym.name, sym.offset as u64, 0, sym.global);
        symbol_indices.insert(sym.name.clone(), idx + 1);
    }

    // Collect symbols referenced by relocations that aren't defined
    for reloc in &segment.relocations {
        if !symbol_indices.contains_key(&reloc.symbol) {
            let idx = elf.add_undefined_symbol(&reloc.symbol);
            symbol_indices.insert(reloc.symbol.clone(), idx + 1);
        }
    }

    // Add relocations
    for reloc in &segment.relocations {
        let reloc_type = match reloc.kind {
            RelocKind::Abs64 => RelocationType::Abs64,
            RelocKind::PCRel32 => RelocationType::Pc32,
            RelocKind::PLT32 => RelocationType::Plt32,
            RelocKind::GOT32 => RelocationType::GotPcRel,
        };

        let symbol_idx = symbol_indices.get(&reloc.symbol)
            .copied()
            .unwrap_or(0);

        elf.add_relocation(
            text_section_idx,
            reloc.offset as u64,
            symbol_idx,
            reloc_type,
            reloc.addend,
        );
    }

    // Generate ELF bytes
    elf.finish().map_err(|e| EmitError::IoError(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_64_encoding() {
        let mut emitter = X86_64Emitter::new();

        // mov rax, rbx
        emitter.emit_mov_rr(X86Reg::RAX, X86Reg::RBX);
        assert_eq!(&emitter.code, &[0x48, 0x89, 0xD8]);

        emitter.code.clear();

        // add rcx, rdx
        emitter.emit_add_rr(X86Reg::RCX, X86Reg::RDX);
        assert_eq!(&emitter.code, &[0x48, 0x01, 0xD1]);

        emitter.code.clear();

        // ret
        emitter.emit_ret();
        assert_eq!(&emitter.code, &[0xC3]);
    }

    #[test]
    fn test_simple_function_emission() {
        let mut module = SirModule::new("test");

        module.create_function("empty", vec![], super::super::types::SirType::Void);

        let result = emit_code(&module, None);
        assert!(result.is_ok());

        let segment = result.unwrap();
        assert!(!segment.code.is_empty());
        assert_eq!(segment.symbols.len(), 1);
        assert_eq!(segment.symbols[0].name, "empty");
    }

    #[test]
    fn test_live_interval_overlap() {
        // Create two intervals: [0, 5) and [3, 8) - these should overlap
        let mut iv1 = LiveInterval::new(ValueId::new(0), 0, SirType::i64());
        iv1.extend_to(4); // Now [0, 5) - end is exclusive
        let mut iv2 = LiveInterval::new(ValueId::new(1), 3, SirType::i64());
        iv2.extend_to(7); // Now [3, 8)

        assert!(iv1.overlaps(&iv2), "Intervals [0,5) and [3,8) should overlap");
        assert!(iv2.overlaps(&iv1), "Intervals [3,8) and [0,5) should overlap");

        // Non-overlapping: [0, 3) and [5, 8)
        let mut iv3 = LiveInterval::new(ValueId::new(2), 0, SirType::i64());
        iv3.extend_to(2); // Now [0, 3)
        let mut iv4 = LiveInterval::new(ValueId::new(3), 5, SirType::i64());
        iv4.extend_to(7); // Now [5, 8)

        assert!(!iv3.overlaps(&iv4), "Intervals [0,3) and [5,8) should not overlap");
        assert!(!iv4.overlaps(&iv3), "Intervals [5,8) and [0,3) should not overlap");
    }

    #[test]
    fn test_live_interval_extend() {
        let mut iv = LiveInterval::new(ValueId::new(0), 5, SirType::i64());
        assert_eq!(iv.start, 5);
        assert_eq!(iv.end, 6); // Minimum length of 1

        iv.extend_to(10);
        assert_eq!(iv.end, 11);

        // Extending to earlier position should not shrink
        iv.extend_to(3);
        assert_eq!(iv.end, 11);
    }

    #[test]
    fn test_register_allocator_basic() {
        let mut allocator = RegisterAllocator::new();

        // Create a simple interval
        let iv = LiveInterval::new(ValueId::new(0), 0, SirType::i64());
        allocator.intervals.push(iv);
        allocator.value_to_interval.insert(ValueId::new(0), 0);

        // Allocate
        allocator.linear_scan().unwrap();

        // Should have assigned a register
        assert!(allocator.intervals[0].reg.is_some());
        assert!(allocator.intervals[0].spill_slot.is_none());
    }

    #[test]
    fn test_callee_saved_preference() {
        let mut allocator = RegisterAllocator::new();

        // Create an interval that crosses a call
        let mut iv = LiveInterval::new(ValueId::new(0), 0, SirType::i64());
        iv.extend_to(10);
        iv.crosses_call = true;

        allocator.intervals.push(iv);
        allocator.value_to_interval.insert(ValueId::new(0), 0);

        allocator.linear_scan().unwrap();

        // Should prefer a callee-saved register
        let reg = allocator.intervals[0].reg.unwrap();
        assert!(
            reg.is_callee_saved(),
            "Expected callee-saved register for value crossing call, got {:?}",
            reg
        );
    }

    #[test]
    fn test_spill_when_registers_exhausted() {
        let mut allocator = RegisterAllocator::new();

        // Create more intervals than available registers
        // Integer registers: RAX, RCX, RDX, RSI, RDI, R8-R15 = 14 registers
        for i in 0..20 {
            let mut iv = LiveInterval::new(ValueId::new(i), 0, SirType::i64());
            iv.extend_to(100); // All live for entire range - forces spilling
            allocator.intervals.push(iv);
            allocator
                .value_to_interval
                .insert(ValueId::new(i), i as usize);
        }

        allocator.linear_scan().unwrap();

        // Count how many got registers vs spilled
        let registered: usize = allocator
            .intervals
            .iter()
            .filter(|iv| iv.reg.is_some())
            .count();
        let spilled: usize = allocator
            .intervals
            .iter()
            .filter(|iv| iv.spill_slot.is_some())
            .count();

        // Should have used all available integer registers and spilled the rest
        assert!(
            registered <= 14,
            "Should not have more than 14 integer registers"
        );
        assert!(spilled > 0, "Should have spilled at least some values");
        assert_eq!(
            registered + spilled,
            20,
            "All values should be either registered or spilled"
        );
    }

    #[test]
    fn test_float_registers() {
        let mut allocator = RegisterAllocator::new();

        // Create a float interval
        let iv = LiveInterval::new(ValueId::new(0), 0, SirType::f64());
        allocator.intervals.push(iv);
        allocator.value_to_interval.insert(ValueId::new(0), 0);

        allocator.linear_scan().unwrap();

        // Should have assigned an XMM register
        let reg = allocator.intervals[0].reg.unwrap();
        assert!(
            reg.is_xmm(),
            "Expected XMM register for float value, got {:?}",
            reg
        );
    }

    #[test]
    fn test_total_stack_size_alignment() {
        let mut allocator = RegisterAllocator::new();

        // Spill a single value (8 bytes)
        allocator.stack_size = 8;

        // Total should be 16-byte aligned
        let total = allocator.total_stack_size();
        assert_eq!(total % 16, 0, "Stack size should be 16-byte aligned");
        assert!(total >= 8, "Stack size should be at least the spill space");
    }

    #[test]
    fn test_x86_reg_properties() {
        // Test callee-saved detection
        assert!(X86Reg::RBX.is_callee_saved());
        assert!(X86Reg::R12.is_callee_saved());
        assert!(X86Reg::R15.is_callee_saved());
        assert!(!X86Reg::RAX.is_callee_saved());
        assert!(!X86Reg::RCX.is_callee_saved());
        assert!(!X86Reg::R11.is_callee_saved());

        // Test XMM detection
        assert!(X86Reg::XMM0.is_xmm());
        assert!(X86Reg::XMM15.is_xmm());
        assert!(!X86Reg::RAX.is_xmm());
        assert!(!X86Reg::R15.is_xmm());

        // Test REX requirement
        assert!(X86Reg::R8.needs_rex());
        assert!(X86Reg::R15.needs_rex());
        assert!(!X86Reg::RAX.needs_rex());
        assert!(!X86Reg::RDI.needs_rex());
    }
}
