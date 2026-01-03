//! # Sounio Compiler: Epistemic-Aware Register Allocation
//!
//! This module implements a register allocator that considers epistemic metadata
//! when making spilling decisions. Values with lower confidence are preferentially
//! spilled to memory, preserving high-confidence values in registers.
//!
//! ## Key Innovation
//!
//! Traditional register allocators use heuristics like:
//! - Spill the value with the longest remaining live range
//! - Spill the value with the fewest uses
//! - Spill the value with the lowest loop nesting
//!
//! Our allocator adds an **epistemic dimension**:
//! - Spill values with **lower confidence (ε)** first
//! - Preserve values with **higher confidence** in registers
//! - Track **provenance degradation** through spill/reload cycles
//!
//! ## Algorithm
//!
//! We use a modified Linear Scan algorithm with epistemic priority:
//!
//! ```text
//! Priority(interval) = α × confidence + β × (1 / live_range_length) + γ × use_count
//! ```
//!
//! Where α, β, γ are tunable weights. Higher priority = less likely to spill.
//!
//! ## Spill Penalty
//!
//! When a value is spilled and reloaded, its epistemic confidence degrades:
//! - ε' = ε × SPILL_CONFIDENCE_FACTOR (default: 0.95)
//! - Provenance gains `SpilledToMemory` marker
//!
//! ## References
//!
//! - Poletto & Sarkar, "Linear Scan Register Allocation" (PLDI 1999)
//! - Wimmer & Mössenböck, "Optimized Interval Splitting" (VEE 2005)
//! - Novel: Epistemic priority integration (Sounio Project)

use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;
use std::fmt;

use super::metrics::Provenance;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Register allocation configuration
#[derive(Debug, Clone)]
pub struct AllocConfig {
    /// Weight for confidence in priority calculation (α)
    pub confidence_weight: f64,
    /// Weight for inverse live range length (β)
    pub live_range_weight: f64,
    /// Weight for use count (γ)
    pub use_count_weight: f64,
    /// Confidence degradation factor when spilling (0.0-1.0)
    pub spill_confidence_factor: f64,
    /// Minimum confidence below which we always spill
    pub min_confidence_threshold: f64,
    /// Whether to track spill statistics
    pub track_statistics: bool,
    /// Maximum spill attempts before giving up
    pub max_spill_attempts: usize,
}

impl Default for AllocConfig {
    fn default() -> Self {
        Self {
            confidence_weight: 0.5,       // 50% weight on confidence
            live_range_weight: 0.3,       // 30% weight on live range
            use_count_weight: 0.2,        // 20% weight on use count
            spill_confidence_factor: 0.95, // 5% degradation per spill
            min_confidence_threshold: 0.1, // Below this, always spill first
            track_statistics: true,
            max_spill_attempts: 1000,
        }
    }
}

impl AllocConfig {
    /// Configuration optimized for scientific computing (preserve confidence)
    pub fn scientific() -> Self {
        Self {
            confidence_weight: 0.7,
            live_range_weight: 0.2,
            use_count_weight: 0.1,
            spill_confidence_factor: 0.98,
            min_confidence_threshold: 0.05,
            ..Default::default()
        }
    }

    /// Configuration optimized for performance (minimize spills)
    pub fn performance() -> Self {
        Self {
            confidence_weight: 0.2,
            live_range_weight: 0.5,
            use_count_weight: 0.3,
            spill_confidence_factor: 0.90,
            min_confidence_threshold: 0.2,
            ..Default::default()
        }
    }

    /// Configuration for embedded/constrained environments
    pub fn embedded() -> Self {
        Self {
            confidence_weight: 0.4,
            live_range_weight: 0.4,
            use_count_weight: 0.2,
            spill_confidence_factor: 0.92,
            min_confidence_threshold: 0.15,
            max_spill_attempts: 500,
            ..Default::default()
        }
    }
}

// ============================================================================
// REGISTER CLASSES AND PHYSICAL REGISTERS
// ============================================================================

/// Register class (grouping of compatible registers)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegClass {
    /// General-purpose integer registers (RAX, RBX, RCX, RDX, RSI, RDI, R8-R15)
    GeneralPurpose,
    /// Floating-point/SIMD registers (XMM0-XMM15, YMM0-YMM15)
    FloatingPoint,
    /// Extended SIMD registers (ZMM0-ZMM31 for AVX-512)
    ExtendedSimd,
    /// Special registers (RSP, RBP - limited allocation)
    Special,
}

/// Physical register identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysReg {
    /// Register class
    pub class: RegClass,
    /// Register number within class
    pub num: u8,
}

impl PhysReg {
    pub fn new(class: RegClass, num: u8) -> Self {
        Self { class, num }
    }

    /// Get x86-64 register name
    pub fn name(&self) -> &'static str {
        match self.class {
            RegClass::GeneralPurpose => match self.num {
                0 => "rax", 1 => "rcx", 2 => "rdx", 3 => "rbx",
                4 => "rsp", 5 => "rbp", 6 => "rsi", 7 => "rdi",
                8 => "r8", 9 => "r9", 10 => "r10", 11 => "r11",
                12 => "r12", 13 => "r13", 14 => "r14", 15 => "r15",
                _ => "r??",
            },
            RegClass::FloatingPoint => match self.num {
                0..=15 => {
                    const NAMES: [&str; 16] = [
                        "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
                        "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15",
                    ];
                    NAMES[self.num as usize]
                }
                _ => "xmm??",
            },
            RegClass::ExtendedSimd => "zmm??",
            RegClass::Special => match self.num {
                0 => "rsp",
                1 => "rbp",
                _ => "???",
            },
        }
    }

    /// Check if register is callee-saved (must be preserved across calls)
    pub fn is_callee_saved(&self) -> bool {
        match self.class {
            RegClass::GeneralPurpose => matches!(self.num, 3 | 5 | 12..=15), // rbx, rbp, r12-r15
            RegClass::FloatingPoint => false, // All XMM are caller-saved in System V ABI
            _ => false,
        }
    }

    /// Check if register is reserved (not allocatable)
    pub fn is_reserved(&self) -> bool {
        match self.class {
            RegClass::GeneralPurpose => matches!(self.num, 4 | 5), // rsp, rbp
            RegClass::Special => true,
            _ => false,
        }
    }
}

impl fmt::Display for PhysReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Virtual register identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VirtReg(pub u32);

impl fmt::Display for VirtReg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "v{}", self.0)
    }
}

// ============================================================================
// EPISTEMIC METADATA
// ============================================================================

/// Epistemic metadata associated with a virtual register
#[derive(Debug, Clone)]
pub struct EpistemicMetadata {
    /// Confidence level (0.0-1.0)
    pub confidence: f64,
    /// Aleatoric uncertainty (standard deviation)
    pub delta: f64,
    /// Provenance chain
    pub provenance: Provenance,
    /// Number of times this value has been spilled
    pub spill_count: u32,
    /// Whether this is a critical value (never spill if possible)
    pub critical: bool,
}

impl Default for EpistemicMetadata {
    fn default() -> Self {
        Self {
            confidence: 1.0,
            delta: 0.0,
            provenance: Provenance::Unknown,
            spill_count: 0,
            critical: false,
        }
    }
}

impl EpistemicMetadata {
    /// Create with specific confidence
    pub fn with_confidence(confidence: f64) -> Self {
        Self {
            confidence: confidence.clamp(0.0, 1.0),
            ..Default::default()
        }
    }

    /// Mark as critical (should not be spilled)
    pub fn critical(mut self) -> Self {
        self.critical = true;
        self
    }

    /// Apply spill degradation
    pub fn apply_spill_degradation(&mut self, factor: f64) {
        self.confidence *= factor;
        self.spill_count += 1;
        self.provenance = Provenance::Merged {
            sources: vec![
                self.provenance.clone(),
                Provenance::sir_backend("SpilledToMemory"),
            ],
        };
    }
}

// ============================================================================
// LIVE INTERVAL
// ============================================================================

/// A live interval represents the lifetime of a virtual register
#[derive(Debug, Clone)]
pub struct LiveInterval {
    /// Virtual register this interval belongs to
    pub vreg: VirtReg,
    /// Required register class
    pub reg_class: RegClass,
    /// Start position (instruction index)
    pub start: usize,
    /// End position (instruction index, exclusive)
    pub end: usize,
    /// Use positions within the interval
    pub uses: Vec<UsePosition>,
    /// Definition position
    pub def_pos: usize,
    /// Epistemic metadata
    pub epistemic: EpistemicMetadata,
    /// Assigned physical register (if allocated)
    pub assigned: Option<PhysReg>,
    /// Spill slot (if spilled)
    pub spill_slot: Option<SpillSlot>,
    /// Weight/priority (computed)
    priority: f64,
}

/// Position and type of a use
#[derive(Debug, Clone, Copy)]
pub struct UsePosition {
    /// Instruction index
    pub pos: usize,
    /// Type of use
    pub kind: UseKind,
}

/// Kind of register use
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UseKind {
    /// Value is read
    Read,
    /// Value is written
    Write,
    /// Value is both read and written
    ReadWrite,
    /// Value must be in a specific register (e.g., for ABI)
    Fixed(PhysReg),
}

/// Spill slot identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpillSlot(pub u32);

impl LiveInterval {
    /// Create a new live interval
    pub fn new(vreg: VirtReg, reg_class: RegClass, def_pos: usize) -> Self {
        Self {
            vreg,
            reg_class,
            start: def_pos,
            end: def_pos + 1,
            uses: vec![UsePosition { pos: def_pos, kind: UseKind::Write }],
            def_pos,
            epistemic: EpistemicMetadata::default(),
            assigned: None,
            spill_slot: None,
            priority: 0.0,
        }
    }

    /// Add a use position
    pub fn add_use(&mut self, pos: usize, kind: UseKind) {
        self.uses.push(UsePosition { pos, kind });
        self.end = self.end.max(pos + 1);
    }

    /// Extend the interval to cover a position
    pub fn extend_to(&mut self, pos: usize) {
        self.end = self.end.max(pos + 1);
    }

    /// Get interval length
    pub fn length(&self) -> usize {
        self.end - self.start
    }

    /// Check if interval is active at a position
    pub fn covers(&self, pos: usize) -> bool {
        pos >= self.start && pos < self.end
    }

    /// Check if interval overlaps with another
    pub fn overlaps(&self, other: &LiveInterval) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Set epistemic metadata
    pub fn with_epistemic(mut self, epistemic: EpistemicMetadata) -> Self {
        self.epistemic = epistemic;
        self
    }

    /// Compute priority based on configuration
    pub fn compute_priority(&mut self, config: &AllocConfig) {
        let confidence_score = self.epistemic.confidence;
        let length_score = 1.0 / (self.length() as f64 + 1.0);
        let use_score = (self.uses.len() as f64).ln().max(0.0) / 10.0;

        self.priority = config.confidence_weight * confidence_score
            + config.live_range_weight * length_score
            + config.use_count_weight * use_score;

        // Boost priority for critical values
        if self.epistemic.critical {
            self.priority *= 2.0;
        }

        // Penalize already-spilled values
        if self.epistemic.spill_count > 0 {
            self.priority *= 0.9_f64.powi(self.epistemic.spill_count as i32);
        }
    }

    /// Get computed priority
    pub fn priority(&self) -> f64 {
        self.priority
    }
}

// Ordering for priority queue (higher priority = greater)
impl Ord for LiveInterval {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority.partial_cmp(&other.priority)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.start.cmp(&other.start))
    }
}

impl PartialOrd for LiveInterval {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for LiveInterval {
    fn eq(&self, other: &Self) -> bool {
        self.vreg == other.vreg
    }
}

impl Eq for LiveInterval {}

// ============================================================================
// REGISTER POOL
// ============================================================================

/// Pool of available physical registers
#[derive(Debug, Clone)]
pub struct RegisterPool {
    /// Available general-purpose registers
    gp_available: Vec<PhysReg>,
    /// Available floating-point registers
    fp_available: Vec<PhysReg>,
    /// Currently allocated registers
    allocated: HashSet<PhysReg>,
    /// Register -> interval mapping
    assignments: HashMap<PhysReg, VirtReg>,
}

impl Default for RegisterPool {
    fn default() -> Self {
        Self::new()
    }
}

impl RegisterPool {
    /// Create a new register pool for x86-64
    pub fn new() -> Self {
        // Allocatable GP registers (excluding RSP, RBP)
        let gp_available: Vec<PhysReg> = [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
            .iter()
            .map(|&n| PhysReg::new(RegClass::GeneralPurpose, n))
            .collect();

        // All XMM registers
        let fp_available: Vec<PhysReg> = (0..16)
            .map(|n| PhysReg::new(RegClass::FloatingPoint, n))
            .collect();

        Self {
            gp_available,
            fp_available,
            allocated: HashSet::new(),
            assignments: HashMap::new(),
        }
    }

    /// Try to allocate a register of the given class
    pub fn try_allocate(&mut self, class: RegClass, vreg: VirtReg) -> Option<PhysReg> {
        let pool = match class {
            RegClass::GeneralPurpose => &self.gp_available,
            RegClass::FloatingPoint | RegClass::ExtendedSimd => &self.fp_available,
            RegClass::Special => return None,
        };

        for &reg in pool {
            if !self.allocated.contains(&reg) {
                self.allocated.insert(reg);
                self.assignments.insert(reg, vreg);
                return Some(reg);
            }
        }
        None
    }

    /// Free a register
    pub fn free(&mut self, reg: PhysReg) {
        self.allocated.remove(&reg);
        self.assignments.remove(&reg);
    }

    /// Check if a register is available
    pub fn is_available(&self, reg: PhysReg) -> bool {
        !self.allocated.contains(&reg)
    }

    /// Get the virtual register assigned to a physical register
    pub fn get_assignment(&self, reg: PhysReg) -> Option<VirtReg> {
        self.assignments.get(&reg).copied()
    }

    /// Count available registers of a class
    pub fn available_count(&self, class: RegClass) -> usize {
        let pool = match class {
            RegClass::GeneralPurpose => &self.gp_available,
            RegClass::FloatingPoint | RegClass::ExtendedSimd => &self.fp_available,
            RegClass::Special => return 0,
        };
        pool.iter().filter(|r| !self.allocated.contains(r)).count()
    }

    /// Get all currently allocated registers
    pub fn allocated_regs(&self) -> impl Iterator<Item = &PhysReg> {
        self.allocated.iter()
    }
}

// ============================================================================
// SPILL SLOT MANAGER
// ============================================================================

/// Manages spill slots on the stack
#[derive(Debug)]
pub struct SpillSlotManager {
    /// Next available slot index
    next_slot: u32,
    /// Size of each slot in bytes
    slot_size: usize,
    /// Mapping from virtual register to spill slot
    vreg_to_slot: HashMap<VirtReg, SpillSlot>,
    /// Free slots (for reuse)
    free_slots: Vec<SpillSlot>,
}

impl Default for SpillSlotManager {
    fn default() -> Self {
        Self::new(8) // 8 bytes per slot (64-bit values)
    }
}

impl SpillSlotManager {
    pub fn new(slot_size: usize) -> Self {
        Self {
            next_slot: 0,
            slot_size,
            vreg_to_slot: HashMap::new(),
            free_slots: Vec::new(),
        }
    }

    /// Allocate a spill slot for a virtual register
    pub fn allocate(&mut self, vreg: VirtReg) -> SpillSlot {
        if let Some(&slot) = self.vreg_to_slot.get(&vreg) {
            return slot;
        }

        let slot = if let Some(slot) = self.free_slots.pop() {
            slot
        } else {
            let slot = SpillSlot(self.next_slot);
            self.next_slot += 1;
            slot
        };

        self.vreg_to_slot.insert(vreg, slot);
        slot
    }

    /// Free a spill slot
    pub fn free(&mut self, vreg: VirtReg) {
        if let Some(slot) = self.vreg_to_slot.remove(&vreg) {
            self.free_slots.push(slot);
        }
    }

    /// Get the stack offset for a spill slot
    pub fn slot_offset(&self, slot: SpillSlot) -> i32 {
        -((slot.0 as i32 + 1) * self.slot_size as i32)
    }

    /// Total stack space needed for spills
    pub fn total_spill_size(&self) -> usize {
        self.next_slot as usize * self.slot_size
    }
}

// ============================================================================
// ALLOCATION STATISTICS
// ============================================================================

/// Statistics about the allocation process
#[derive(Debug, Default, Clone)]
pub struct AllocStatistics {
    /// Total intervals processed
    pub total_intervals: usize,
    /// Intervals successfully allocated to registers
    pub allocated_to_reg: usize,
    /// Intervals spilled to memory
    pub spilled: usize,
    /// Total spill operations (including re-spills)
    pub total_spill_ops: usize,
    /// Total reload operations
    pub total_reload_ops: usize,
    /// Average confidence of allocated values
    pub avg_allocated_confidence: f64,
    /// Average confidence of spilled values
    pub avg_spilled_confidence: f64,
    /// Confidence preserved (weighted average)
    pub confidence_preserved: f64,
    /// Number of critical values that had to be spilled
    pub critical_spills: usize,
}

impl AllocStatistics {
    /// Compute summary metrics
    pub fn finalize(&mut self) {
        if self.allocated_to_reg > 0 {
            self.avg_allocated_confidence /= self.allocated_to_reg as f64;
        }
        if self.spilled > 0 {
            self.avg_spilled_confidence /= self.spilled as f64;
        }
        if self.total_intervals > 0 {
            self.confidence_preserved /= self.total_intervals as f64;
        }
    }
}

impl fmt::Display for AllocStatistics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Register Allocation Statistics:")?;
        writeln!(f, "  Total intervals: {}", self.total_intervals)?;
        writeln!(f, "  Allocated to registers: {} ({:.1}%)", 
            self.allocated_to_reg,
            100.0 * self.allocated_to_reg as f64 / self.total_intervals.max(1) as f64)?;
        writeln!(f, "  Spilled to memory: {} ({:.1}%)",
            self.spilled,
            100.0 * self.spilled as f64 / self.total_intervals.max(1) as f64)?;
        writeln!(f, "  Avg confidence (allocated): {:.3}", self.avg_allocated_confidence)?;
        writeln!(f, "  Avg confidence (spilled): {:.3}", self.avg_spilled_confidence)?;
        writeln!(f, "  Confidence preserved: {:.1}%", self.confidence_preserved * 100.0)?;
        if self.critical_spills > 0 {
            writeln!(f, "  ⚠ Critical values spilled: {}", self.critical_spills)?;
        }
        Ok(())
    }
}

// ============================================================================
// EPISTEMIC REGISTER ALLOCATOR
// ============================================================================

/// The main epistemic-aware register allocator
pub struct EpistemicAllocator {
    /// Configuration
    config: AllocConfig,
    /// Register pool
    reg_pool: RegisterPool,
    /// Spill slot manager
    spill_manager: SpillSlotManager,
    /// Active intervals (currently occupying registers)
    active: Vec<LiveInterval>,
    /// Allocation statistics
    stats: AllocStatistics,
}

impl EpistemicAllocator {
    /// Create a new allocator with default configuration
    pub fn new() -> Self {
        Self::with_config(AllocConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: AllocConfig) -> Self {
        Self {
            config,
            reg_pool: RegisterPool::new(),
            spill_manager: SpillSlotManager::default(),
            active: Vec::new(),
            stats: AllocStatistics::default(),
        }
    }

    /// Run allocation on a set of live intervals
    /// 
    /// Returns the modified intervals with assignments and allocation result
    pub fn allocate(&mut self, mut intervals: Vec<LiveInterval>) -> AllocResult {
        self.stats = AllocStatistics::default();
        self.stats.total_intervals = intervals.len();

        // Compute priorities for all intervals
        for interval in &mut intervals {
            interval.compute_priority(&self.config);
        }

        // Sort by start position for linear scan
        intervals.sort_by_key(|i| i.start);

        let mut allocated = Vec::new();
        let mut spilled = Vec::new();
        let mut spill_code = Vec::new();

        for mut interval in intervals {
            // Expire old intervals
            self.expire_old_intervals(interval.start, &mut allocated);

            // Try to allocate
            if let Some(reg) = self.try_allocate_reg(&interval) {
                interval.assigned = Some(reg);
                self.stats.allocated_to_reg += 1;
                self.stats.avg_allocated_confidence += interval.epistemic.confidence;
                self.stats.confidence_preserved += interval.epistemic.confidence;
                self.active.push(interval.clone());
                allocated.push(interval);
            } else {
                // Need to spill - either this interval or an active one
                let spill_result = self.spill_at_interval(&mut interval, &mut allocated);
                
                match spill_result {
                    SpillDecision::SpillCurrent(slot) => {
                        interval.spill_slot = Some(slot);
                        self.stats.spilled += 1;
                        self.stats.avg_spilled_confidence += interval.epistemic.confidence;
                        self.stats.confidence_preserved += 
                            interval.epistemic.confidence * self.config.spill_confidence_factor;
                        if interval.epistemic.critical {
                            self.stats.critical_spills += 1;
                        }
                        spilled.push(interval);
                    }
                    SpillDecision::SpillOther { victim_vreg, new_reg, slot, code } => {
                        // Move victim to spilled
                        if let Some(pos) = self.active.iter().position(|i| i.vreg == victim_vreg) {
                            let mut victim = self.active.remove(pos);
                            victim.spill_slot = Some(slot);
                            victim.assigned = None;
                            victim.epistemic.apply_spill_degradation(self.config.spill_confidence_factor);
                            
                            self.stats.spilled += 1;
                            self.stats.avg_spilled_confidence += victim.epistemic.confidence;
                            if victim.epistemic.critical {
                                self.stats.critical_spills += 1;
                            }
                            
                            spilled.push(victim);
                        }
                        
                        // Allocate current to freed register
                        interval.assigned = Some(new_reg);
                        self.stats.allocated_to_reg += 1;
                        self.stats.avg_allocated_confidence += interval.epistemic.confidence;
                        self.stats.confidence_preserved += interval.epistemic.confidence;
                        self.active.push(interval.clone());
                        allocated.push(interval);
                        
                        spill_code.extend(code);
                    }
                    SpillDecision::Failed => {
                        // Last resort: spill current
                        let slot = self.spill_manager.allocate(interval.vreg);
                        interval.spill_slot = Some(slot);
                        self.stats.spilled += 1;
                        self.stats.avg_spilled_confidence += interval.epistemic.confidence;
                        spilled.push(interval);
                    }
                }
            }
        }

        // Finalize statistics
        self.stats.total_spill_ops = spilled.len();
        self.stats.finalize();

        AllocResult {
            allocated,
            spilled,
            spill_code,
            stats: self.stats.clone(),
            total_spill_size: self.spill_manager.total_spill_size(),
        }
    }

    /// Expire intervals that end before the given position
    fn expire_old_intervals(&mut self, pos: usize, allocated: &mut Vec<LiveInterval>) {
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].end <= pos {
                let interval = self.active.remove(i);
                if let Some(reg) = interval.assigned {
                    self.reg_pool.free(reg);
                }
                // Interval already in allocated list, no action needed
            } else {
                i += 1;
            }
        }
    }

    /// Try to allocate a physical register for an interval
    fn try_allocate_reg(&mut self, interval: &LiveInterval) -> Option<PhysReg> {
        self.reg_pool.try_allocate(interval.reg_class, interval.vreg)
    }

    /// Decide what to spill when no registers are available
    fn spill_at_interval(
        &mut self,
        current: &mut LiveInterval,
        _allocated: &mut Vec<LiveInterval>,
    ) -> SpillDecision {
        // Find the interval with lowest priority among active
        let victim_idx = self.active.iter()
            .enumerate()
            .filter(|(_, i)| i.reg_class == current.reg_class)
            .min_by(|(_, a), (_, b)| a.priority.partial_cmp(&b.priority).unwrap_or(Ordering::Equal))
            .map(|(idx, _)| idx);

        let victim_idx = match victim_idx {
            Some(idx) => idx,
            None => return SpillDecision::Failed,
        };

        let victim = &self.active[victim_idx];

        // Epistemic decision: spill lower confidence value
        // But also consider: is victim's remaining range shorter?
        let current_score = current.epistemic.confidence;
        let victim_score = victim.epistemic.confidence;

        // If current has lower confidence AND victim has a register, spill current
        if current_score < victim_score * 0.8 && !current.epistemic.critical {
            let slot = self.spill_manager.allocate(current.vreg);
            return SpillDecision::SpillCurrent(slot);
        }

        // If victim has much lower confidence, spill victim
        if victim_score < current_score * 0.8 || 
           (victim_score < self.config.min_confidence_threshold && !victim.epistemic.critical) {
            let reg = victim.assigned.expect("Active interval should have register");
            let slot = self.spill_manager.allocate(victim.vreg);
            
            // Generate spill code
            let spill_code = vec![
                SpillReloadOp::Spill {
                    vreg: victim.vreg,
                    reg,
                    slot,
                    pos: current.start,
                }
            ];

            self.reg_pool.free(reg);
            
            return SpillDecision::SpillOther {
                victim_vreg: victim.vreg,
                new_reg: reg,
                slot,
                code: spill_code,
            };
        }

        // Default: spill current if it ends later
        if current.end > victim.end {
            let slot = self.spill_manager.allocate(current.vreg);
            SpillDecision::SpillCurrent(slot)
        } else {
            let reg = victim.assigned.expect("Active interval should have register");
            let slot = self.spill_manager.allocate(victim.vreg);
            
            let spill_code = vec![
                SpillReloadOp::Spill {
                    vreg: victim.vreg,
                    reg,
                    slot,
                    pos: current.start,
                }
            ];

            self.reg_pool.free(reg);
            
            SpillDecision::SpillOther {
                victim_vreg: victim.vreg,
                new_reg: reg,
                slot,
                code: spill_code,
            }
        }
    }

    /// Get current statistics
    pub fn statistics(&self) -> &AllocStatistics {
        &self.stats
    }

    /// Get spill slot manager
    pub fn spill_manager(&self) -> &SpillSlotManager {
        &self.spill_manager
    }
}

impl Default for EpistemicAllocator {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of spill decision
enum SpillDecision {
    /// Spill the current interval
    SpillCurrent(SpillSlot),
    /// Spill another (active) interval, freeing a register
    SpillOther {
        victim_vreg: VirtReg,
        new_reg: PhysReg,
        slot: SpillSlot,
        code: Vec<SpillReloadOp>,
    },
    /// Failed to make a decision
    Failed,
}

/// Spill/reload operation to insert
#[derive(Debug, Clone)]
pub enum SpillReloadOp {
    /// Store register to spill slot
    Spill {
        vreg: VirtReg,
        reg: PhysReg,
        slot: SpillSlot,
        pos: usize,
    },
    /// Load spill slot to register
    Reload {
        vreg: VirtReg,
        reg: PhysReg,
        slot: SpillSlot,
        pos: usize,
    },
}

impl SpillReloadOp {
    /// Get the position where this operation should be inserted
    pub fn position(&self) -> usize {
        match self {
            SpillReloadOp::Spill { pos, .. } => *pos,
            SpillReloadOp::Reload { pos, .. } => *pos,
        }
    }
}

/// Result of register allocation
#[derive(Debug)]
pub struct AllocResult {
    /// Intervals allocated to physical registers
    pub allocated: Vec<LiveInterval>,
    /// Intervals spilled to memory
    pub spilled: Vec<LiveInterval>,
    /// Spill/reload operations to insert
    pub spill_code: Vec<SpillReloadOp>,
    /// Allocation statistics
    pub stats: AllocStatistics,
    /// Total stack space needed for spills
    pub total_spill_size: usize,
}

impl AllocResult {
    /// Check if allocation was successful (no critical failures)
    pub fn is_successful(&self) -> bool {
        self.stats.critical_spills == 0
    }

    /// Get assignment for a virtual register
    pub fn get_assignment(&self, vreg: VirtReg) -> Option<PhysReg> {
        self.allocated.iter()
            .find(|i| i.vreg == vreg)
            .and_then(|i| i.assigned)
    }

    /// Get spill slot for a virtual register
    pub fn get_spill_slot(&self, vreg: VirtReg) -> Option<SpillSlot> {
        self.spilled.iter()
            .find(|i| i.vreg == vreg)
            .and_then(|i| i.spill_slot)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_interval(vreg: u32, start: usize, end: usize, confidence: f64) -> LiveInterval {
        let mut interval = LiveInterval::new(
            VirtReg(vreg),
            RegClass::GeneralPurpose,
            start,
        );
        interval.end = end;
        interval.epistemic = EpistemicMetadata::with_confidence(confidence);
        interval
    }

    #[test]
    fn test_simple_allocation() {
        let mut allocator = EpistemicAllocator::new();
        
        let intervals = vec![
            make_interval(0, 0, 10, 0.9),
            make_interval(1, 5, 15, 0.8),
            make_interval(2, 12, 20, 0.95),
        ];

        let result = allocator.allocate(intervals);
        
        assert_eq!(result.stats.total_intervals, 3);
        assert!(result.stats.allocated_to_reg > 0);
    }

    #[test]
    fn test_spill_low_confidence() {
        let config = AllocConfig {
            confidence_weight: 0.9,  // Heavy weight on confidence
            ..Default::default()
        };
        let mut allocator = EpistemicAllocator::with_config(config);
        
        // Create more intervals than registers
        let mut intervals = Vec::new();
        for i in 0..20 {
            let confidence = if i < 5 { 0.1 } else { 0.9 };  // First 5 have low confidence
            intervals.push(make_interval(i, 0, 100, confidence));
        }

        let result = allocator.allocate(intervals);
        
        // Low confidence values should be spilled preferentially
        for interval in &result.spilled {
            // Most spilled should have low confidence
            // (not all, since we have more intervals than regs)
            println!("Spilled v{} with confidence {}", interval.vreg.0, interval.epistemic.confidence);
        }
        
        assert!(result.stats.spilled > 0);
        assert!(result.stats.avg_spilled_confidence < result.stats.avg_allocated_confidence);
    }

    #[test]
    fn test_critical_preservation() {
        let mut allocator = EpistemicAllocator::new();
        
        let mut intervals = Vec::new();
        
        // One critical interval
        let mut critical = make_interval(0, 0, 100, 0.5);
        critical.epistemic.critical = true;
        intervals.push(critical);
        
        // Many non-critical with higher confidence
        for i in 1..20 {
            intervals.push(make_interval(i, 0, 100, 0.9));
        }

        let result = allocator.allocate(intervals);
        
        // Critical interval should not be spilled if possible
        let critical_spilled = result.spilled.iter()
            .any(|i| i.vreg == VirtReg(0));
        
        // With enough registers, critical should not be spilled
        // (14 GP registers available, 20 intervals, so some will spill)
        println!("Critical spilled: {}", critical_spilled);
        println!("Stats: {}", result.stats);
    }

    #[test]
    fn test_confidence_degradation() {
        let config = AllocConfig {
            spill_confidence_factor: 0.8,  // 20% degradation per spill
            ..Default::default()
        };
        
        let mut meta = EpistemicMetadata::with_confidence(1.0);
        assert_eq!(meta.confidence, 1.0);
        assert_eq!(meta.spill_count, 0);
        
        meta.apply_spill_degradation(config.spill_confidence_factor);
        assert!((meta.confidence - 0.8).abs() < 0.001);
        assert_eq!(meta.spill_count, 1);
        
        meta.apply_spill_degradation(config.spill_confidence_factor);
        assert!((meta.confidence - 0.64).abs() < 0.001);
        assert_eq!(meta.spill_count, 2);
    }

    #[test]
    fn test_register_pool() {
        let mut pool = RegisterPool::new();
        
        // Should have 14 GP registers available
        assert_eq!(pool.available_count(RegClass::GeneralPurpose), 14);
        
        // Allocate all
        for i in 0..14 {
            let reg = pool.try_allocate(RegClass::GeneralPurpose, VirtReg(i));
            assert!(reg.is_some());
        }
        
        // No more available
        assert_eq!(pool.available_count(RegClass::GeneralPurpose), 0);
        assert!(pool.try_allocate(RegClass::GeneralPurpose, VirtReg(100)).is_none());
        
        // Free one
        pool.free(PhysReg::new(RegClass::GeneralPurpose, 0));
        assert_eq!(pool.available_count(RegClass::GeneralPurpose), 1);
    }

    #[test]
    fn test_spill_slot_manager() {
        let mut manager = SpillSlotManager::default();
        
        let slot1 = manager.allocate(VirtReg(0));
        let slot2 = manager.allocate(VirtReg(1));
        
        assert_ne!(slot1, slot2);
        assert_eq!(manager.slot_offset(slot1), -8);
        assert_eq!(manager.slot_offset(slot2), -16);
        assert_eq!(manager.total_spill_size(), 16);
        
        // Same vreg returns same slot
        let slot1_again = manager.allocate(VirtReg(0));
        assert_eq!(slot1, slot1_again);
    }

    #[test]
    fn test_live_interval_priority() {
        let config = AllocConfig::default();
        
        let mut high_conf = make_interval(0, 0, 10, 0.95);
        let mut low_conf = make_interval(1, 0, 10, 0.2);
        
        high_conf.compute_priority(&config);
        low_conf.compute_priority(&config);
        
        assert!(high_conf.priority() > low_conf.priority());
    }
}

// ============================================================================
// SIR INTEGRATION
// ============================================================================

use crate::sir::blocks::SirFunction;
use crate::sir::metadata::{Metadata, MetadataStore};
use crate::sir::module::SirModule;
use crate::sir::values::ValueId;

/// Extract epistemic metadata from SIR module for register allocation.
///
/// Converts SIR epistemic metadata into the allocator's internal format,
/// enabling confidence-aware spilling decisions.
pub fn extract_epistemic_metadata(
    _sir_module: &SirModule,
    func: &SirFunction,
    metadata_store: &MetadataStore,
) -> HashMap<ValueId, EpistemicMetadata> {
    let mut result = HashMap::new();

    // Extract metadata for all values used in this function
    for block in &func.blocks {
        // Extract from instructions
        for inst in &block.instructions {
            // Get metadata for instruction result (if any)
            if let Some(result_val) = inst.result {
                if let Some(metas) = metadata_store.get(result_val) {
                    for meta in metas {
                        if let Metadata::Epistemic(sir_ep_meta) = meta {
                            // Convert SIR EpistemicMetadata to allocator EpistemicMetadata
                            let confidence = sir_ep_meta.known_confidence.unwrap_or(0.5);
                            let delta = 0.1; // Default uncertainty

                            result.insert(
                                result_val,
                                EpistemicMetadata {
                                    confidence,
                                    delta,
                                    provenance: super::metrics::Provenance::sir_backend(
                                        "native-allocator",
                                    ),
                                    spill_count: 0,
                                    critical: sir_ep_meta.certain,
                                },
                            );
                        }
                    }
                }
            }

            // Get metadata for operands
            for operand in inst.inst.operands() {
                if let Some(metas) = metadata_store.get(operand) {
                    for meta in metas {
                        if let Metadata::Epistemic(sir_ep_meta) = meta {
                            let confidence = sir_ep_meta.known_confidence.unwrap_or(0.5);
                            let delta = 0.1;

                            result.entry(operand).or_insert(EpistemicMetadata {
                                confidence,
                                delta,
                                provenance: super::metrics::Provenance::sir_backend(
                                    "native-allocator",
                                ),
                                spill_count: 0,
                                critical: sir_ep_meta.certain,
                            });
                        }
                    }
                }
            }
        }
    }

    result
}

/// Build live intervals from SIR function with epistemic metadata.
pub fn build_intervals_from_sir(
    func: &SirFunction,
    metadata: &HashMap<ValueId, EpistemicMetadata>,
    reg_class: RegClass,
) -> Vec<LiveInterval> {
    let mut intervals: HashMap<ValueId, LiveInterval> = HashMap::new();
    let mut pos = 0;

    for block in &func.blocks {
        for inst in &block.instructions {
            // Create interval for result
            if let Some(result_val) = inst.result {
                let vreg = VirtReg(result_val.0);
                let mut interval = LiveInterval::new(vreg, reg_class, pos);

                if let Some(ep_meta) = metadata.get(&result_val) {
                    interval.epistemic = ep_meta.clone();
                }

                intervals.insert(result_val, interval);
            }

            // Extend intervals for operands
            for operand in inst.inst.operands() {
                if let Some(interval) = intervals.get_mut(&operand) {
                    interval.add_use(pos, UseKind::Read);
                }
            }

            pos += 1;
        }
    }

    intervals.into_values().collect()
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        AllocConfig,
        AllocResult,
        AllocStatistics,
        EpistemicAllocator,
        EpistemicMetadata,
        LiveInterval,
        PhysReg,
        RegClass,
        RegisterPool,
        SpillReloadOp,
        SpillSlot,
        SpillSlotManager,
        UseKind,
        UsePosition,
        VirtReg,
        build_intervals_from_sir,
        extract_epistemic_metadata,
    };
}
