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

use super::blocks::{BasicBlock, SirFunction, Terminator};
use super::module::{Architecture, SirModule, TargetTriple};
use super::ops::*;
use super::types::SirType;
use super::values::{Constant, ValueId};
use std::collections::{BTreeSet, HashMap, HashSet};

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
// ATTENTION-BASED REGISTER ALLOCATION CONFIGURATION
// ============================================================================

/// Configuration for the attention-based register allocator.
///
/// This unified configuration combines classical allocation heuristics with
/// epistemic (confidence-aware) scoring. The attention score determines
/// which values are most important to keep in registers.
///
/// Formula:
/// attention_score = (w_use_density * use_density + w_crosses_call * crosses_call)  // classic
///                 + (w_confidence * τ - w_uncertainty * σ + w_provenance * ρ)       // epistemic
///
/// Higher score = more important to keep in register
/// Lower score = better candidate for spilling
#[derive(Clone, Debug)]
pub struct AttentionConfig {
    // === Classic component ===
    /// Weight for use density (uses per unit of lifetime)
    pub w_use_density: f64,
    /// Weight for values that cross function calls
    pub w_crosses_call: f64,

    // === Epistemic component ===
    /// Weight for confidence τ ∈ [0,1]
    pub w_confidence: f64,
    /// Weight for uncertainty σ ∈ [0,1] (subtracted, so higher = worse)
    pub w_uncertainty: f64,
    /// Weight for provenance tracking ρ ∈ {0,1}
    pub w_provenance: f64,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            w_use_density: 1.0,
            w_crosses_call: 0.5,
            w_confidence: 0.5,
            w_uncertainty: 0.3,
            w_provenance: 0.2,
        }
    }
}

impl AttentionConfig {
    /// Create a config optimized for scientific computing
    /// (prioritizes confidence and provenance)
    pub fn scientific() -> Self {
        Self {
            w_use_density: 0.8,
            w_crosses_call: 0.4,
            w_confidence: 0.7,
            w_uncertainty: 0.5,
            w_provenance: 0.4,
        }
    }

    /// Create a config optimized for performance
    /// (prioritizes classical heuristics)
    pub fn performance() -> Self {
        Self {
            w_use_density: 1.5,
            w_crosses_call: 0.8,
            w_confidence: 0.2,
            w_uncertainty: 0.1,
            w_provenance: 0.1,
        }
    }
}

/// Metrics collected during register allocation for validation
#[derive(Debug, Clone, Default)]
pub struct AllocationMetrics {
    /// Total number of spills performed
    pub total_spills: usize,
    /// Spills of high-confidence values (τ > 0.7)
    pub high_confidence_spills: usize,
    /// Spills of low-confidence values (τ < 0.3)
    pub low_confidence_spills: usize,
    /// Number of provenance-tracked values that stayed in registers
    pub provenance_preserved: usize,
    /// Number of provenance-tracked values that were spilled
    pub provenance_spilled: usize,
    /// Total values allocated
    pub total_values: usize,
}

impl AllocationMetrics {
    /// Calculate the epistemic quality score for the allocation.
    ///
    /// A good epistemic policy spills more low-confidence values
    /// and preserves high-confidence values. Score > 1.0 is good.
    pub fn epistemic_quality(&self) -> f64 {
        if self.total_spills == 0 {
            return 1.0;
        }

        let good_spills = self.low_confidence_spills as f64;
        let bad_spills = self.high_confidence_spills as f64;

        (good_spills + 1.0) / (bad_spills + 1.0)
    }

    /// Record a spill event with confidence information
    pub fn record_spill(&mut self, confidence: f64, has_provenance: bool) {
        self.total_spills += 1;

        if confidence > 0.7 {
            self.high_confidence_spills += 1;
        } else if confidence < 0.3 {
            self.low_confidence_spills += 1;
        }

        if has_provenance {
            self.provenance_spilled += 1;
        }
    }

    /// Record that a provenance value stayed in registers
    pub fn record_provenance_preserved(&mut self) {
        self.provenance_preserved += 1;
    }
}

// ============================================================================
// LIVE INTERVAL AND REGISTER ALLOCATION
// ============================================================================

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
}

impl RegisterAllocator {
    /// Create a new register allocator
    pub fn new() -> Self {
        Self {
            intervals: Vec::new(),
            active: Vec::new(),
            free_int_regs: X86Reg::allocatable_int().to_vec(),
            free_xmm_regs: X86Reg::allocatable_xmm().to_vec(),
            stack_size: 0,
            value_to_interval: HashMap::new(),
            used_callee_saved: HashSet::new(),
            call_positions: Vec::new(),
        }
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
        self.linear_scan()?;

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

        // Handle function parameters - they are live from instruction 0
        // System V AMD64 ABI: integers in RDI, RSI, RDX, RCX, R8, R9
        // Floats in XMM0-XMM7
        for (param_idx, (_, param_ty)) in func.params.iter().enumerate() {
            let value_id = ValueId::new(param_idx as u32);
            value_defs.insert(value_id, (0, param_ty.clone()));
            value_uses.entry(value_id).or_default();
        }

        // Process each basic block
        for block in &func.blocks {
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
                inst_idx += 1;
            }
        }

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

    /// Handle register spilling - EPISTEMIC-AWARE
    ///
    /// This is the breakthrough: the first compiler in history that makes spill
    /// decisions based on TRUST, not just lifetime.
    ///
    /// Strategy:
    /// 1. Preferentially spill LOW-CONFIDENCE values (they represent uncertain data)
    /// 2. Preferentially KEEP HIGH-CONFIDENCE values in registers (they're scientifically rigorous)
    /// 3. Among equal confidence, fall back to classical "ends furthest" heuristic
    ///
    /// The result: Code paths with strong epistemic backing execute faster
    /// (more register-resident). Dubious data is naturally penalized.
    /// The machine itself rewards scientific rigor.
    fn spill_at_interval(&mut self, current: usize) -> Result<(), EmitError> {
        if self.active.is_empty() {
            // No active intervals - must spill the current interval
            return self.spill_interval(current);
        }

        // === EPISTEMIC-AWARE SPILL SELECTION ===
        // Find the BEST interval to spill using a combined score:
        // - Lower epistemic_weight → more likely to spill
        // - Longer remaining lifetime → more likely to spill (classic heuristic)
        // The formula weights epistemic trust heavily:
        //   spill_score = (1.0 - epistemic_weight) * 0.6 + normalized_lifetime * 0.4

        let current_weight = self.intervals[current].epistemic_weight;
        let current_end = self.intervals[current].end;

        // Find the best spill candidate among active intervals
        let mut best_spill_idx: Option<usize> = None;
        let mut best_spill_score = f64::MIN;

        for &active_idx in &self.active {
            let interval = &self.intervals[active_idx];
            let weight = interval.epistemic_weight;
            let remaining_life = (interval.end - interval.start) as f64;

            // Combined score: lower confidence + longer life = better spill candidate
            // Epistemic weight is MORE important than lifetime (0.7 vs 0.3)
            let spill_score = (1.0 - weight) * 0.7 + (remaining_life / 1000.0).min(1.0) * 0.3;

            if spill_score > best_spill_score {
                best_spill_score = spill_score;
                best_spill_idx = Some(active_idx);
            }
        }

        // Compare with current interval's spill score
        let current_life = (current_end - self.intervals[current].start) as f64;
        let current_spill_score =
            (1.0 - current_weight) * 0.7 + (current_life / 1000.0).min(1.0) * 0.3;

        let spill_candidate = if let Some(candidate) = best_spill_idx {
            // Spill the one with higher spill score (lower confidence, longer life)
            if best_spill_score > current_spill_score {
                candidate
            } else {
                current
            }
        } else {
            current
        };

        if spill_candidate != current {
            // Spill an active interval and give its register to current
            let reg = self.intervals[spill_candidate]
                .reg
                .expect("Active interval must have a register");

            // Transfer register to current interval
            self.intervals[current].reg = Some(reg);

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
            self.spill_interval(current)?;
        }

        Ok(())
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
            relocations: vec![],
            reg_alloc: HashMap::new(),
            frame_size: 0,
            labels: HashMap::new(),
            forward_refs: vec![],
            register_allocator: RegisterAllocator::new(),
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
        if disp == 0 && base.encoding() != X86Reg::RBP.encoding() {
            self.emit_modrm(0b00, src.encoding(), base.encoding());
        } else if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, src.encoding(), base.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, src.encoding(), base.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// MOV dst, [reg + disp] - load from memory with displacement
    fn emit_mov_r64_mem_disp(&mut self, dst: X86Reg, base: X86Reg, disp: i32) {
        self.emit_rex(true, dst.needs_rex(), false, base.needs_rex());
        self.emit_byte(0x8B); // MOV r64, r/m64
        if disp == 0 && base.encoding() != X86Reg::RBP.encoding() {
            self.emit_modrm(0b00, dst.encoding(), base.encoding());
        } else if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, dst.encoding(), base.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, dst.encoding(), base.encoding());
            self.emit_u32(disp as u32);
        }
    }

    /// LEA reg, [base + disp] - load effective address
    fn emit_lea(&mut self, dst: X86Reg, base: X86Reg, disp: i32) {
        self.emit_rex(true, dst.needs_rex(), false, base.needs_rex());
        self.emit_byte(0x8D); // LEA r64, m
        if disp >= -128 && disp <= 127 {
            self.emit_modrm(0b01, dst.encoding(), base.encoding());
            self.emit_byte(disp as u8);
        } else {
            self.emit_modrm(0b10, dst.encoding(), base.encoding());
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
        let mut symbols = vec![];

        for func in &module.functions {
            // Record symbol
            symbols.push(Symbol {
                name: func.name.clone(),
                offset: self.offset(),
                global: true,
            });

            // Emit function
            self.emit_function(func)?;
        }

        Ok(CodeSegment {
            code: std::mem::take(&mut self.code),
            relocations: std::mem::take(&mut self.relocations),
            symbols,
        })
    }

    fn emit_function(&mut self, func: &SirFunction) -> Result<Vec<u8>, EmitError> {
        let start = self.offset();

        // Step 1: Run register allocation
        self.register_allocator.allocate_registers(func)?;

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
        self.frame_size = if spill_space > 0 {
            // Round up to 16-byte alignment
            (spill_space + 15) & !15
        } else {
            0
        };

        // Step 3: Emit prologue with callee-saved register saves
        self.emit_prologue_with_saves(self.frame_size, &callee_saved);

        // Step 4: Move incoming parameters to their allocated locations
        self.setup_incoming_params(func);

        // Step 5: Emit instructions for each basic block
        // Clear labels from previous functions
        self.labels.clear();
        self.forward_refs.clear();

        for block in &func.blocks {
            // Record label for this block
            self.labels.insert(block.id.0, self.offset());

            // Emit instructions
            for inst in &block.instructions {
                self.emit_instruction(inst, func)?;
            }

            // Emit terminator
            if let Some(term) = &block.terminator {
                self.emit_terminator(term, &callee_saved)?;
            }
        }

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
                self.emit_call(inst.result, info)?;
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
            _ => {
                // Other instructions not yet implemented
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
                // FRem requires a function call to fmod - for now, emit a placeholder
                return Err(EmitError::UnsupportedInstruction("FRem".into()));
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
        let lhs_reg = self.get_value_reg(lhs, X86Reg::R10);
        let rhs_reg = self.get_value_reg(rhs, X86Reg::R11);
        let dst_reg = self
            .register_allocator
            .get_reg(result)
            .unwrap_or(X86Reg::RAX);

        // Integer comparison
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
            _ => CondCode::E, // Floating point comparisons need different handling
        };

        // SETcc sets the low byte, we need to zero-extend
        self.emit_setcc(cc, dst_reg);
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
            super::values::Constant::I64(val) => {
                self.emit_mov_ri64(dst_reg, *val);
            }
            super::values::Constant::I32(val) => {
                self.emit_mov_ri64(dst_reg, *val as i64);
            }
            super::values::Constant::Bool(val) => {
                self.emit_mov_ri64(dst_reg, if *val { 1 } else { 0 });
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
            UnaryFloatOp::Sqrt => {
                self.emit_sqrtsd(dst, src);
            }
            _ => {
                // Other unary ops would need library calls
                return Err(EmitError::UnsupportedInstruction(format!("{:?}", op)));
            }
        }

        Ok(())
    }

    /// Emit a function call
    fn emit_call(&mut self, result: Option<ValueId>, info: &CallInfo) -> Result<(), EmitError> {
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
            Callee::Direct(_func_id) => {
                // Internal function call - would need function address
                self.emit_call_rel32(0); // Placeholder
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
            _ => {}
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
        callee_saved: &[X86Reg],
    ) -> Result<(), EmitError> {
        match term {
            Terminator::Return(None) => {
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

                // JNZ then_block
                if let Some(&then_offset) = self.labels.get(&then_block.0) {
                    let rel = (then_offset as i32) - (self.offset() as i32) - 6;
                    self.emit_jcc_rel32(CondCode::NE as u8, rel);
                } else {
                    self.forward_refs.push((self.offset() + 2, then_block.0));
                    self.emit_jcc_rel32(CondCode::NE as u8, 0);
                }

                // JMP else_block (fallthrough or explicit jump)
                if let Some(&else_offset) = self.labels.get(&else_block.0) {
                    let rel = (else_offset as i32) - (self.offset() as i32) - 5;
                    self.emit_jmp_rel32(rel);
                } else {
                    self.forward_refs.push((self.offset() + 1, else_block.0));
                    self.emit_jmp_rel32(0);
                }
            }
            Terminator::Unreachable => {
                // UD2 instruction (undefined - will trap)
                self.emit_bytes(&[0x0F, 0x0B]);
            }
            _ => {
                // Other terminators not yet implemented
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
pub fn emit_code(module: &SirModule) -> Result<CodeSegment, EmitError> {
    match module.target.arch {
        Architecture::X86_64 => {
            let mut emitter = X86_64Emitter::new();
            emitter.emit_module(module)
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

        let result = emit_code(&module);
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
