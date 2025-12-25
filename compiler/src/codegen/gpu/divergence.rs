//! Dynamic Control Flow & Warp Divergence Analysis for GPU
//!
//! Provides comprehensive analysis and optimization of control flow divergence
//! on GPU architectures, with focus on warp-level execution efficiency.
//!
//! # Overview
//!
//! On GPUs, threads execute in lockstep groups (warps on NVIDIA, wavefronts on AMD).
//! When threads in a warp take different control flow paths, the hardware must
//! serialize execution, reducing throughput. This module analyzes and optimizes
//! divergent control flow.
//!
//! # Architecture
//!
//! ```text
//! GpuKernel
//!    │
//!    ▼
//! ┌──────────────────────┐
//! │ Divergence Analyzer  │ ← Identify divergent branches
//! └──────────────────────┘
//!    │
//!    ▼
//! ┌──────────────────────┐
//! │   Cost Estimator     │ ← Calculate serialization cost
//! └──────────────────────┘
//!    │
//!    ▼
//! ┌──────────────────────┐
//! │ Control Flow Optimizer│ ← Apply optimizations
//! └──────────────────────┘
//!    │
//!    ▼
//! OptimizedKernel
//! ```

use super::ir::{BlockId, GpuKernel, GpuOp, GpuTerminator, ValueId};
use rustc_hash::{FxHashMap, FxHashSet};

// ============================================================================
// Divergence Analysis
// ============================================================================

/// Divergence classification for control flow
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DivergenceKind {
    /// All threads take the same path (uniform execution)
    Uniform,
    /// Threads may diverge (non-uniform execution)
    Divergent,
    /// Uniform within warps, but warps may differ
    PartiallyUniform,
    /// Divergence depends on input data patterns
    DataDependent,
}

impl DivergenceKind {
    /// Check if this kind allows optimization
    pub fn is_optimizable(&self) -> bool {
        matches!(
            self,
            DivergenceKind::Uniform | DivergenceKind::PartiallyUniform
        )
    }

    /// Get the severity score (0 = no divergence, 1 = worst)
    pub fn severity_score(&self) -> f64 {
        match self {
            DivergenceKind::Uniform => 0.0,
            DivergenceKind::PartiallyUniform => 0.3,
            DivergenceKind::DataDependent => 0.5,
            DivergenceKind::Divergent => 1.0,
        }
    }
}

/// Thread mask representing active threads in a warp
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadMask {
    /// Bitmask for 32 threads (NVIDIA warp size)
    pub mask: u32,
}

impl ThreadMask {
    /// All threads active
    pub const ALL: Self = ThreadMask { mask: 0xFFFFFFFF };

    /// No threads active
    pub const NONE: Self = ThreadMask { mask: 0 };

    /// Create mask from bitmask
    pub fn from_mask(mask: u32) -> Self {
        Self { mask }
    }

    /// Check if all threads are active
    pub fn is_full(&self) -> bool {
        self.mask == 0xFFFFFFFF
    }

    /// Check if no threads are active
    pub fn is_empty(&self) -> bool {
        self.mask == 0
    }

    /// Count active threads
    pub fn count(&self) -> u32 {
        self.mask.count_ones()
    }

    /// Compute intersection (AND)
    pub fn intersect(&self, other: ThreadMask) -> ThreadMask {
        ThreadMask {
            mask: self.mask & other.mask,
        }
    }

    /// Compute union (OR)
    pub fn union(&self, other: ThreadMask) -> ThreadMask {
        ThreadMask {
            mask: self.mask | other.mask,
        }
    }

    /// Compute complement (NOT)
    pub fn complement(&self) -> ThreadMask {
        ThreadMask { mask: !self.mask }
    }

    /// Check if masks overlap
    pub fn overlaps(&self, other: ThreadMask) -> bool {
        (self.mask & other.mask) != 0
    }
}

/// Information about divergence at a specific point
#[derive(Debug, Clone)]
pub struct DivergenceInfo {
    /// Type of divergence
    pub kind: DivergenceKind,
    /// Nesting level of divergent control flow
    pub divergence_depth: u32,
    /// Point where control flow reconverges
    pub reconvergence_point: Option<BlockId>,
    /// Which threads are affected by this divergence
    pub affected_threads: ThreadMask,
    /// Estimated execution cost multiplier (1.0 = no overhead)
    pub cost_multiplier: f64,
}

impl DivergenceInfo {
    /// Create info for uniform execution
    pub fn uniform() -> Self {
        Self {
            kind: DivergenceKind::Uniform,
            divergence_depth: 0,
            reconvergence_point: None,
            affected_threads: ThreadMask::ALL,
            cost_multiplier: 1.0,
        }
    }

    /// Create info for divergent execution
    pub fn divergent(depth: u32, reconverge: Option<BlockId>) -> Self {
        Self {
            kind: DivergenceKind::Divergent,
            divergence_depth: depth,
            reconvergence_point: reconverge,
            affected_threads: ThreadMask::ALL,
            cost_multiplier: 2.0, // Assume worst-case doubling
        }
    }

    /// Check if this represents actual divergence
    pub fn is_divergent(&self) -> bool {
        self.kind != DivergenceKind::Uniform
    }

    /// Get the estimated overhead percentage
    pub fn overhead_percent(&self) -> f64 {
        (self.cost_multiplier - 1.0) * 100.0
    }
}

/// Warp divergence analyzer
pub struct WarpDivergenceAnalyzer {
    /// Cache of analyzed blocks
    block_info: FxHashMap<BlockId, DivergenceInfo>,
    /// Control flow graph (block -> successors)
    cfg: FxHashMap<BlockId, Vec<BlockId>>,
    /// Reverse CFG (block -> predecessors)
    reverse_cfg: FxHashMap<BlockId, Vec<BlockId>>,
    /// Dominator tree
    dominators: FxHashMap<BlockId, FxHashSet<BlockId>>,
    /// Post-dominator tree
    post_dominators: FxHashMap<BlockId, FxHashSet<BlockId>>,
}

impl WarpDivergenceAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self {
            block_info: FxHashMap::default(),
            cfg: FxHashMap::default(),
            reverse_cfg: FxHashMap::default(),
            dominators: FxHashMap::default(),
            post_dominators: FxHashMap::default(),
        }
    }

    /// Analyze divergence for an entire kernel
    pub fn analyze_kernel(&mut self, kernel: &GpuKernel) -> KernelDivergenceAnalysis {
        // Build control flow graph
        self.build_cfg(kernel);

        // Compute dominators and post-dominators
        self.compute_dominators(kernel.entry);
        self.compute_post_dominators(kernel);

        // Analyze each block
        let mut block_divergence = FxHashMap::default();
        let mut divergent_branches = Vec::new();

        for block in &kernel.blocks {
            let info = self.analyze_block(block, kernel);
            if info.is_divergent() {
                divergent_branches.push((block.id, info.clone()));
            }
            block_divergence.insert(block.id, info);
        }

        // Compute overall statistics
        let total_blocks = kernel.blocks.len();
        let divergent_blocks = divergent_branches.len();
        let divergence_ratio = divergent_blocks as f64 / total_blocks as f64;
        let max_divergence_depth = self.max_depth(&block_divergence);
        let estimated_overhead = self.estimate_total_overhead(&block_divergence);

        KernelDivergenceAnalysis {
            kernel_name: kernel.name.clone(),
            block_divergence,
            divergent_branches,
            divergence_ratio,
            max_divergence_depth,
            estimated_overhead,
        }
    }

    /// Build control flow graph from kernel
    fn build_cfg(&mut self, kernel: &GpuKernel) {
        self.cfg.clear();
        self.reverse_cfg.clear();

        for block in &kernel.blocks {
            let successors = match &block.terminator {
                GpuTerminator::Br(target) => vec![*target],
                GpuTerminator::CondBr(_, true_bb, false_bb) => vec![*true_bb, *false_bb],
                GpuTerminator::ReturnVoid | GpuTerminator::Return(_) => vec![],
                GpuTerminator::Unreachable => vec![],
            };

            for &succ in &successors {
                self.reverse_cfg.entry(succ).or_default().push(block.id);
            }

            self.cfg.insert(block.id, successors);
        }
    }

    /// Compute dominator tree
    fn compute_dominators(&mut self, entry: BlockId) {
        self.dominators.clear();

        // Initialize: entry dominates only itself
        let mut dom = FxHashMap::default();
        dom.insert(entry, vec![entry].into_iter().collect());

        // All other blocks: dominated by all blocks initially
        for &block in self.cfg.keys() {
            if block != entry {
                dom.insert(block, self.cfg.keys().copied().collect());
            }
        }

        // Iterative dataflow until fixpoint
        let mut changed = true;
        while changed {
            changed = false;
            for &block in self.cfg.keys() {
                if block == entry {
                    continue;
                }

                let preds = self.reverse_cfg.get(&block).cloned().unwrap_or_default();
                if preds.is_empty() {
                    continue;
                }

                // New dominators = {block} ∪ (∩ dom(pred) for pred in preds)
                let mut new_dom: FxHashSet<BlockId> =
                    dom.get(&preds[0]).cloned().unwrap_or_default();

                for &pred in &preds[1..] {
                    if let Some(pred_dom) = dom.get(&pred) {
                        new_dom = new_dom.intersection(pred_dom).copied().collect();
                    }
                }
                new_dom.insert(block);

                if dom.get(&block) != Some(&new_dom) {
                    dom.insert(block, new_dom);
                    changed = true;
                }
            }
        }

        self.dominators = dom;
    }

    /// Compute post-dominator tree
    fn compute_post_dominators(&mut self, kernel: &GpuKernel) {
        self.post_dominators.clear();

        // Find exit blocks (no successors)
        let exit_blocks: Vec<BlockId> = self
            .cfg
            .iter()
            .filter(|(_, succs)| succs.is_empty())
            .map(|(&block, _)| block)
            .collect();

        if exit_blocks.is_empty() {
            return;
        }

        // For simplicity, assume single exit or create virtual exit
        let virtual_exit = BlockId(u32::MAX);

        // Build reverse CFG for post-dominance
        let mut reverse_analysis = FxHashMap::default();
        reverse_analysis.insert(virtual_exit, vec![virtual_exit].into_iter().collect());

        for &block in self.cfg.keys() {
            reverse_analysis.insert(block, self.cfg.keys().copied().collect());
        }

        // Iterative dataflow
        let mut changed = true;
        while changed {
            changed = false;
            for &block in self.cfg.keys() {
                let succs = self.cfg.get(&block).cloned().unwrap_or_default();
                if succs.is_empty() {
                    continue;
                }

                let mut new_postdom: FxHashSet<BlockId> =
                    reverse_analysis.get(&succs[0]).cloned().unwrap_or_default();

                for &succ in &succs[1..] {
                    if let Some(succ_postdom) = reverse_analysis.get(&succ) {
                        new_postdom = new_postdom.intersection(succ_postdom).copied().collect();
                    }
                }
                new_postdom.insert(block);

                if reverse_analysis.get(&block) != Some(&new_postdom) {
                    reverse_analysis.insert(block, new_postdom);
                    changed = true;
                }
            }
        }

        self.post_dominators = reverse_analysis;
    }

    /// Analyze a single basic block
    fn analyze_block(&mut self, block: &super::ir::GpuBlock, kernel: &GpuKernel) -> DivergenceInfo {
        // Check if already cached
        if let Some(info) = self.block_info.get(&block.id) {
            return info.clone();
        }

        // Analyze terminator for divergence
        let info = match &block.terminator {
            GpuTerminator::CondBr(cond_val, true_bb, false_bb) => {
                // Check if condition is uniform or divergent
                let cond_uniform = self.is_uniform_value(*cond_val, block, kernel);

                if cond_uniform {
                    DivergenceInfo::uniform()
                } else {
                    // Find reconvergence point (immediate post-dominator)
                    let reconverge = self.find_reconvergence_point(block.id, *true_bb, *false_bb);

                    // Compute depth
                    let depth = self.compute_divergence_depth(block.id);

                    DivergenceInfo::divergent(depth, reconverge)
                }
            }
            GpuTerminator::Br(_) => DivergenceInfo::uniform(),
            GpuTerminator::ReturnVoid | GpuTerminator::Return(_) => DivergenceInfo::uniform(),
            GpuTerminator::Unreachable => DivergenceInfo::uniform(),
        };

        self.block_info.insert(block.id, info.clone());
        info
    }

    /// Check if a value is uniform across the warp
    fn is_uniform_value(
        &self,
        _value: ValueId,
        block: &super::ir::GpuBlock,
        _kernel: &GpuKernel,
    ) -> bool {
        // Look for the instruction that produces this value
        for (vid, op) in &block.instructions {
            if *vid == _value {
                return self.is_uniform_op(op);
            }
        }

        // Conservative: assume divergent if unknown
        false
    }

    /// Check if an operation produces uniform results
    fn is_uniform_op(&self, op: &GpuOp) -> bool {
        match op {
            // Thread indices are divergent
            GpuOp::ThreadIdX | GpuOp::ThreadIdY | GpuOp::ThreadIdZ => false,
            GpuOp::LaneId => false,

            // Block/grid dimensions are uniform
            GpuOp::BlockIdX
            | GpuOp::BlockIdY
            | GpuOp::BlockIdZ
            | GpuOp::BlockDimX
            | GpuOp::BlockDimY
            | GpuOp::BlockDimZ
            | GpuOp::GridDimX
            | GpuOp::GridDimY
            | GpuOp::GridDimZ
            | GpuOp::WarpId
            | GpuOp::WarpSize => true,

            // Constants are uniform
            GpuOp::ConstInt(_, _) | GpuOp::ConstFloat(_, _) | GpuOp::ConstBool(_) => true,

            // Parameters are uniform (loaded from same address)
            GpuOp::Param(_) => true,

            // Warp vote operations produce uniform results
            GpuOp::WarpVote(_, _) => true,

            // Most other operations depend on their operands
            _ => false,
        }
    }

    /// Find the reconvergence point for divergent branches
    fn find_reconvergence_point(
        &self,
        branch_block: BlockId,
        true_bb: BlockId,
        false_bb: BlockId,
    ) -> Option<BlockId> {
        // Find the immediate post-dominator of both branches
        let true_postdom = self.post_dominators.get(&true_bb)?;
        let false_postdom = self.post_dominators.get(&false_bb)?;

        // Intersection gives common post-dominators
        let common: Vec<BlockId> = true_postdom
            .intersection(false_postdom)
            .copied()
            .filter(|&b| b != branch_block) // Exclude the branch itself
            .collect();

        // The immediate post-dominator is the closest one
        common.first().copied()
    }

    /// Compute divergence depth (nesting level)
    fn compute_divergence_depth(&self, block: BlockId) -> u32 {
        let mut depth = 0;
        let preds = self.reverse_cfg.get(&block).cloned().unwrap_or_default();

        for &pred in &preds {
            if let Some(info) = self.block_info.get(&pred) {
                depth = depth.max(info.divergence_depth + 1);
            }
        }

        depth
    }

    /// Get maximum divergence depth
    fn max_depth(&self, block_info: &FxHashMap<BlockId, DivergenceInfo>) -> u32 {
        block_info
            .values()
            .map(|info| info.divergence_depth)
            .max()
            .unwrap_or(0)
    }

    /// Estimate total overhead from divergence
    fn estimate_total_overhead(&self, block_info: &FxHashMap<BlockId, DivergenceInfo>) -> f64 {
        if block_info.is_empty() {
            return 0.0;
        }

        let total_cost: f64 = block_info.values().map(|info| info.cost_multiplier).sum();
        let ideal_cost = block_info.len() as f64;

        ((total_cost / ideal_cost) - 1.0) * 100.0
    }
}

impl Default for WarpDivergenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Complete divergence analysis for a kernel
#[derive(Debug, Clone)]
pub struct KernelDivergenceAnalysis {
    /// Kernel name
    pub kernel_name: String,
    /// Divergence info per block
    pub block_divergence: FxHashMap<BlockId, DivergenceInfo>,
    /// List of divergent branches (block, info)
    pub divergent_branches: Vec<(BlockId, DivergenceInfo)>,
    /// Ratio of divergent blocks to total blocks
    pub divergence_ratio: f64,
    /// Maximum nesting depth of divergent control flow
    pub max_divergence_depth: u32,
    /// Estimated overhead percentage
    pub estimated_overhead: f64,
}

impl KernelDivergenceAnalysis {
    /// Check if kernel has significant divergence
    pub fn has_significant_divergence(&self) -> bool {
        self.divergence_ratio > 0.2 || self.max_divergence_depth > 2
    }

    /// Get divergence severity (0.0 to 1.0)
    pub fn severity(&self) -> f64 {
        (self.divergence_ratio + (self.max_divergence_depth as f64 / 10.0)).min(1.0)
    }
}

// ============================================================================
// Predicated Execution
// ============================================================================

/// Predicate register for conditional execution
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PredicateReg {
    /// Register index
    pub index: u32,
}

impl PredicateReg {
    /// Create a new predicate register
    pub fn new(index: u32) -> Self {
        Self { index }
    }
}

/// Compiler for converting branches to predicated execution
pub struct PredicateCompiler {
    /// Next available predicate register
    next_pred_reg: u32,
    /// Maximum predicate registers available
    max_pred_regs: u32,
    /// Mapping from block to predicate
    block_predicates: FxHashMap<BlockId, PredicateReg>,
}

impl PredicateCompiler {
    /// Create a new predicate compiler
    pub fn new(max_pred_regs: u32) -> Self {
        Self {
            next_pred_reg: 0,
            max_pred_regs,
            block_predicates: FxHashMap::default(),
        }
    }

    /// Allocate a new predicate register
    pub fn allocate_predicate(&mut self) -> Option<PredicateReg> {
        if self.next_pred_reg >= self.max_pred_regs {
            return None;
        }

        let pred = PredicateReg::new(self.next_pred_reg);
        self.next_pred_reg += 1;
        Some(pred)
    }

    /// Generate predicate mask for a branch
    pub fn generate_predicate_mask(
        &mut self,
        condition: ValueId,
        true_bb: BlockId,
        false_bb: BlockId,
    ) -> PredicateMask {
        let true_pred = self
            .allocate_predicate()
            .expect("Out of predicate registers");
        let false_pred = self
            .allocate_predicate()
            .expect("Out of predicate registers");

        self.block_predicates.insert(true_bb, true_pred);
        self.block_predicates.insert(false_bb, false_pred);

        PredicateMask {
            condition,
            true_pred,
            false_pred,
            nesting_level: 0,
        }
    }

    /// Check if predication is beneficial
    pub fn should_predicate(&self, analysis: &DivergenceInfo, block_size: usize) -> bool {
        // Only predicate small blocks to avoid register pressure
        if block_size > 10 {
            return false;
        }

        // Only predicate if divergence is significant
        if !analysis.is_divergent() {
            return false;
        }

        // Check if we have enough predicate registers
        if self.next_pred_reg + 2 > self.max_pred_regs {
            return false;
        }

        true
    }

    /// Reset the compiler
    pub fn reset(&mut self) {
        self.next_pred_reg = 0;
        self.block_predicates.clear();
    }
}

/// Predicate mask for conditional execution
#[derive(Debug, Clone)]
pub struct PredicateMask {
    /// Condition value
    pub condition: ValueId,
    /// Predicate for true path
    pub true_pred: PredicateReg,
    /// Predicate for false path
    pub false_pred: PredicateReg,
    /// Nesting level for nested predicates
    pub nesting_level: u32,
}

// ============================================================================
// Branch Cost Model
// ============================================================================

/// Estimator for branch divergence costs
pub struct BranchCostEstimator {
    /// Warp size (32 for NVIDIA, 64 for AMD)
    warp_size: u32,
    /// Base cost for a uniform branch (cycles)
    uniform_branch_cost: f64,
    /// Cost per serialized execution path
    serialization_cost: f64,
}

impl BranchCostEstimator {
    /// Create a new cost estimator
    pub fn new(warp_size: u32) -> Self {
        Self {
            warp_size,
            uniform_branch_cost: 1.0,
            serialization_cost: 10.0,
        }
    }

    /// Estimate cost of a divergent branch
    pub fn estimate_divergent_cost(&self, info: &DivergenceInfo, branch_size: usize) -> BranchCost {
        if !info.is_divergent() {
            return BranchCost {
                base_cost: self.uniform_branch_cost,
                serialization_penalty: 0.0,
                reconvergence_overhead: 0.0,
                total_cost: self.uniform_branch_cost,
            };
        }

        // Base cost for the branch itself
        let base_cost = self.uniform_branch_cost;

        // Serialization penalty: assume worst case (all threads diverge)
        let serialization_penalty = self.serialization_cost * info.divergence_depth as f64;

        // Reconvergence overhead: cost to merge execution paths
        let reconvergence_overhead = if info.reconvergence_point.is_some() {
            2.0 // Barrier + merge overhead
        } else {
            0.0
        };

        // Total cost includes instruction execution under divergence
        let instruction_cost = branch_size as f64 * info.cost_multiplier;

        let total_cost =
            base_cost + serialization_penalty + reconvergence_overhead + instruction_cost;

        BranchCost {
            base_cost,
            serialization_penalty,
            reconvergence_overhead,
            total_cost,
        }
    }

    /// Estimate cost of predicated execution
    pub fn estimate_predicated_cost(&self, branch_size: usize) -> f64 {
        // Predicated execution: all threads execute, but masked
        // Cost = instruction_count + predicate_overhead
        let instruction_cost = branch_size as f64;
        let predicate_overhead = 0.5; // Small overhead for predicate management

        instruction_cost + predicate_overhead
    }

    /// Determine if predication is better than branching
    pub fn should_use_predication(&self, info: &DivergenceInfo, branch_size: usize) -> bool {
        let branch_cost = self.estimate_divergent_cost(info, branch_size);
        let predicate_cost = self.estimate_predicated_cost(branch_size);

        predicate_cost < branch_cost.total_cost
    }
}

impl Default for BranchCostEstimator {
    fn default() -> Self {
        Self::new(32) // Default to NVIDIA warp size
    }
}

/// Cost breakdown for a branch
#[derive(Debug, Clone)]
pub struct BranchCost {
    /// Base cost of the branch instruction
    pub base_cost: f64,
    /// Penalty from serialization
    pub serialization_penalty: f64,
    /// Overhead from reconvergence
    pub reconvergence_overhead: f64,
    /// Total estimated cost
    pub total_cost: f64,
}

impl BranchCost {
    /// Get the overhead ratio (total / base)
    pub fn overhead_ratio(&self) -> f64 {
        if self.base_cost == 0.0 {
            0.0
        } else {
            self.total_cost / self.base_cost
        }
    }
}

// ============================================================================
// Control Flow Optimization
// ============================================================================

/// Optimizer for divergent control flow
pub struct ControlFlowOptimizer {
    /// Divergence analyzer
    analyzer: WarpDivergenceAnalyzer,
    /// Cost estimator
    cost_estimator: BranchCostEstimator,
    /// Predicate compiler
    predicate_compiler: PredicateCompiler,
}

impl ControlFlowOptimizer {
    /// Create a new optimizer
    pub fn new(warp_size: u32, max_pred_regs: u32) -> Self {
        Self {
            analyzer: WarpDivergenceAnalyzer::new(),
            cost_estimator: BranchCostEstimator::new(warp_size),
            predicate_compiler: PredicateCompiler::new(max_pred_regs),
        }
    }

    /// Optimize control flow in a kernel
    pub fn optimize(&mut self, kernel: &GpuKernel) -> ControlFlowOptReport {
        // Analyze divergence
        let analysis = self.analyzer.analyze_kernel(kernel);

        // Identify optimization opportunities
        let mut optimizations = Vec::new();

        for (block_id, info) in &analysis.block_divergence {
            if !info.is_divergent() {
                continue;
            }

            // Find the block
            let block = kernel.blocks.iter().find(|b| b.id == *block_id);
            if block.is_none() {
                continue;
            }
            let block = block.unwrap();

            // Check if we should apply optimizations
            let block_size = block.instructions.len();

            // Option 1: Convert to predicated execution
            if self.predicate_compiler.should_predicate(info, block_size) {
                optimizations.push(ControlFlowOpt::ConvertToPredicated(*block_id));
            }

            // Option 2: Hoist uniform code
            if self.has_hoistable_code(block) {
                optimizations.push(ControlFlowOpt::HoistUniformCode(*block_id));
            }

            // Option 3: Loop unrolling for uniform bounds
            if self.is_uniform_loop(block) {
                optimizations.push(ControlFlowOpt::UnrollLoop(*block_id));
            }
        }

        let estimated_speedup = self.estimate_speedup(&analysis);

        ControlFlowOptReport {
            analysis,
            applied_optimizations: optimizations,
            estimated_speedup,
        }
    }

    /// Check if block has code that can be hoisted
    fn has_hoistable_code(&self, block: &super::ir::GpuBlock) -> bool {
        // Look for uniform operations that can be moved out of divergent regions
        for (_, op) in &block.instructions {
            if self.analyzer.is_uniform_op(op) {
                return true;
            }
        }
        false
    }

    /// Check if this is a loop with uniform bounds
    fn is_uniform_loop(&self, _block: &super::ir::GpuBlock) -> bool {
        // Simplified: would need full loop detection
        false
    }

    /// Estimate speedup from optimizations
    fn estimate_speedup(&self, analysis: &KernelDivergenceAnalysis) -> f64 {
        if analysis.estimated_overhead == 0.0 {
            return 1.0;
        }

        // Assume we can eliminate 50% of divergence overhead
        let improvement = analysis.estimated_overhead * 0.5;
        1.0 + (improvement / 100.0)
    }

    /// Detect warp-uniform conditions
    pub fn detect_uniform_conditions(&self, kernel: &GpuKernel) -> Vec<(BlockId, ValueId)> {
        let mut uniform_conditions = Vec::new();

        for block in &kernel.blocks {
            if let GpuTerminator::CondBr(cond, _, _) = &block.terminator
                && self.analyzer.is_uniform_value(*cond, block, kernel)
            {
                uniform_conditions.push((block.id, *cond));
            }
        }

        uniform_conditions
    }
}

impl Default for ControlFlowOptimizer {
    fn default() -> Self {
        Self::new(32, 7) // NVIDIA defaults: warp=32, 7 predicate regs
    }
}

/// Control flow optimization types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlFlowOpt {
    /// Convert branch to predicated execution
    ConvertToPredicated(BlockId),
    /// Hoist uniform code out of divergent region
    HoistUniformCode(BlockId),
    /// Unroll loop with uniform bounds
    UnrollLoop(BlockId),
    /// Merge branches with same target
    MergeBranches(BlockId),
}

/// Report from control flow optimization
#[derive(Debug, Clone)]
pub struct ControlFlowOptReport {
    /// Divergence analysis results
    pub analysis: KernelDivergenceAnalysis,
    /// Optimizations that were applied
    pub applied_optimizations: Vec<ControlFlowOpt>,
    /// Estimated speedup factor
    pub estimated_speedup: f64,
}

impl ControlFlowOptReport {
    /// Check if any optimizations were applied
    pub fn has_optimizations(&self) -> bool {
        !self.applied_optimizations.is_empty()
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        format!(
            "Divergence: {:.1}% overhead, {} optimizations applied, {:.2}x estimated speedup",
            self.analysis.estimated_overhead,
            self.applied_optimizations.len(),
            self.estimated_speedup
        )
    }
}

// ============================================================================
// Adaptive Dispatch
// ============================================================================

/// Adaptive kernel dispatcher for runtime conditions
pub struct AdaptiveDispatcher {
    /// Uniform kernel variant (optimized for no divergence)
    uniform_variant: Option<String>,
    /// Divergent kernel variant (handles arbitrary divergence)
    divergent_variant: Option<String>,
    /// Threshold for choosing variant (divergence ratio)
    divergence_threshold: f64,
}

impl AdaptiveDispatcher {
    /// Create a new adaptive dispatcher
    pub fn new() -> Self {
        Self {
            uniform_variant: None,
            divergent_variant: None,
            divergence_threshold: 0.3,
        }
    }

    /// Register a uniform kernel variant
    pub fn register_uniform_variant(&mut self, kernel_name: String) {
        self.uniform_variant = Some(kernel_name);
    }

    /// Register a divergent kernel variant
    pub fn register_divergent_variant(&mut self, kernel_name: String) {
        self.divergent_variant = Some(kernel_name);
    }

    /// Select the best kernel variant based on runtime conditions
    pub fn select_variant(&self, expected_divergence: f64) -> Option<&str> {
        if expected_divergence < self.divergence_threshold {
            self.uniform_variant.as_deref()
        } else {
            self.divergent_variant.as_deref()
        }
    }

    /// Set the threshold for variant selection
    pub fn set_threshold(&mut self, threshold: f64) {
        self.divergence_threshold = threshold.clamp(0.0, 1.0);
    }
}

impl Default for AdaptiveDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thread_mask() {
        let mask1 = ThreadMask::from_mask(0xFF);
        let mask2 = ThreadMask::from_mask(0xF0);

        assert_eq!(mask1.count(), 8);
        assert_eq!(mask2.count(), 4);

        let intersection = mask1.intersect(mask2);
        assert_eq!(intersection.mask, 0xF0);

        let union = mask1.union(mask2);
        assert_eq!(union.mask, 0xFF);
    }

    #[test]
    fn test_divergence_kind() {
        assert!(DivergenceKind::Uniform.is_optimizable());
        assert!(!DivergenceKind::Divergent.is_optimizable());

        assert_eq!(DivergenceKind::Uniform.severity_score(), 0.0);
        assert_eq!(DivergenceKind::Divergent.severity_score(), 1.0);
    }

    #[test]
    fn test_divergence_info() {
        let uniform = DivergenceInfo::uniform();
        assert!(!uniform.is_divergent());
        assert_eq!(uniform.overhead_percent(), 0.0);

        let divergent = DivergenceInfo::divergent(2, None);
        assert!(divergent.is_divergent());
        assert!(divergent.overhead_percent() > 0.0);
    }

    #[test]
    fn test_predicate_compiler() {
        let mut compiler = PredicateCompiler::new(7);

        let pred1 = compiler.allocate_predicate();
        assert!(pred1.is_some());
        assert_eq!(pred1.unwrap().index, 0);

        let pred2 = compiler.allocate_predicate();
        assert!(pred2.is_some());
        assert_eq!(pred2.unwrap().index, 1);

        compiler.reset();
        let pred3 = compiler.allocate_predicate();
        assert_eq!(pred3.unwrap().index, 0);
    }

    #[test]
    fn test_branch_cost_estimator() {
        let estimator = BranchCostEstimator::new(32);

        let uniform_info = DivergenceInfo::uniform();
        let cost = estimator.estimate_divergent_cost(&uniform_info, 10);
        assert_eq!(cost.serialization_penalty, 0.0);

        let divergent_info = DivergenceInfo::divergent(1, None);
        let div_cost = estimator.estimate_divergent_cost(&divergent_info, 10);
        assert!(div_cost.serialization_penalty > 0.0);
        assert!(div_cost.total_cost > cost.total_cost);
    }

    #[test]
    fn test_adaptive_dispatcher() {
        let mut dispatcher = AdaptiveDispatcher::new();
        dispatcher.register_uniform_variant("kernel_uniform".to_string());
        dispatcher.register_divergent_variant("kernel_divergent".to_string());

        let variant = dispatcher.select_variant(0.1);
        assert_eq!(variant, Some("kernel_uniform"));

        let variant = dispatcher.select_variant(0.8);
        assert_eq!(variant, Some("kernel_divergent"));
    }

    #[test]
    fn test_control_flow_optimizer() {
        let optimizer = ControlFlowOptimizer::new(32, 7);
        assert!(optimizer.cost_estimator.warp_size == 32);
    }
}
