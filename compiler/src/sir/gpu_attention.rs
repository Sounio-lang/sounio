//! GPU-Specific Attention-Based Register Allocation
//!
//! This module extends the CPU attention-based register allocation to handle
//! GPU's unique multi-resource constraints. The key insight: "+1 register/thread
//! can reduce occupancy more than a few spills cost."
//!
//! # GPU Resource Model
//!
//! Unlike CPUs where registers are the primary scarce resource, GPUs have
//! multiple interacting constraints:
//!
//! 1. **Registers**: Limited per-thread (e.g., 255 for CUDA)
//! 2. **Shared Memory**: Per-block, affects block size
//! 3. **Occupancy**: Warps per SM, determined by register/shared usage
//! 4. **Divergence**: Warp divergence from confidence-based branches
//!
//! # Occupancy-Register Tradeoff
//!
//! NVIDIA GPUs have a non-linear occupancy curve:
//! - 32 regs/thread → 100% occupancy (2048 threads/SM)
//! - 64 regs/thread → 50% occupancy (1024 threads/SM)
//! - 128 regs/thread → 25% occupancy (512 threads/SM)
//! - 255 regs/thread → 12.5% occupancy (256 threads/SM)
//!
//! The allocator must balance register pressure against occupancy loss.
//!
//! # Epistemic-GPU Integration
//!
//! Confidence information affects GPU allocation in unique ways:
//! - High-variance confidence → likely divergent branches → penalize
//! - Low confidence values → may not need registers at all
//! - Provenance tracking → shared memory for audit logs
//!
//! # References
//!
//! - CUDA Occupancy Calculator: https://developer.nvidia.com/cuda-occupancy-calculator
//! - "Occupancy-Aware Register Allocation" (Coutinho et al., CGO 2016)

use super::alloc_policy::AttentionConfig;
use super::emit::LiveInterval;
use std::cmp::Ordering;

// Re-export for tests
#[cfg(test)]
use super::types::SirType;
#[cfg(test)]
use super::values::ValueId;

// ============================================================================
// GPU RESOURCE BUDGET
// ============================================================================

/// The hardware resources available on a GPU compute unit (SM/CU).
///
/// These limits determine how register pressure translates to occupancy
/// and how shared memory affects block sizing.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuResourceBudget {
    /// Maximum registers available per thread (e.g., 255 for CUDA)
    pub max_registers_per_thread: u32,

    /// Maximum shared memory per block in bytes (e.g., 48KB for most GPUs)
    pub max_shared_per_block: u32,

    /// Target occupancy ratio [0.0, 1.0]
    /// 1.0 = maximum concurrent warps, but may require aggressive spilling
    /// 0.5 = reasonable balance for compute-bound kernels
    pub target_occupancy: f64,

    /// Warp size (32 for NVIDIA, 64 for AMD)
    pub warp_size: u32,

    /// Maximum warps per SM at full occupancy
    pub max_warps_per_sm: u32,

    /// Total register file size per SM (e.g., 65536 for modern NVIDIA)
    pub register_file_size: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Maximum blocks per SM
    pub max_blocks_per_sm: u32,

    /// Shared memory banks (for conflict analysis)
    pub shared_memory_banks: u32,

    /// GPU compute capability (major, minor) for NVIDIA
    pub compute_capability: (u32, u32),
}

impl Default for GpuResourceBudget {
    /// Default to NVIDIA Ampere (SM 8.0) characteristics
    fn default() -> Self {
        Self {
            max_registers_per_thread: 255,
            max_shared_per_block: 163840,    // 160KB for Ampere
            target_occupancy: 0.5,           // Reasonable default
            warp_size: 32,
            max_warps_per_sm: 64,            // 2048 threads / 32
            register_file_size: 65536,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 32,
            shared_memory_banks: 32,
            compute_capability: (8, 0),
        }
    }
}

impl GpuResourceBudget {
    /// Create budget for specific NVIDIA compute capability
    pub fn nvidia(major: u32, minor: u32) -> Self {
        match (major, minor) {
            // Volta (V100)
            (7, 0) => Self {
                max_registers_per_thread: 255,
                max_shared_per_block: 98304,  // 96KB
                target_occupancy: 0.5,
                warp_size: 32,
                max_warps_per_sm: 64,
                register_file_size: 65536,
                max_threads_per_block: 1024,
                max_blocks_per_sm: 32,
                shared_memory_banks: 32,
                compute_capability: (7, 0),
            },
            // Turing (RTX 2080)
            (7, 5) => Self {
                max_registers_per_thread: 255,
                max_shared_per_block: 65536,  // 64KB
                target_occupancy: 0.5,
                warp_size: 32,
                max_warps_per_sm: 32,
                register_file_size: 65536,
                max_threads_per_block: 1024,
                max_blocks_per_sm: 16,
                shared_memory_banks: 32,
                compute_capability: (7, 5),
            },
            // Ampere (A100, RTX 3090)
            (8, 0) | (8, 6) => Self::default(),
            // Hopper (H100)
            (9, 0) => Self {
                max_registers_per_thread: 255,
                max_shared_per_block: 232448, // 227KB
                target_occupancy: 0.5,
                warp_size: 32,
                max_warps_per_sm: 64,
                register_file_size: 65536,
                max_threads_per_block: 1024,
                max_blocks_per_sm: 32,
                shared_memory_banks: 32,
                compute_capability: (9, 0),
            },
            // Default to Ampere
            _ => Self::default(),
        }
    }

    /// Create budget for AMD CDNA/RDNA architecture
    pub fn amd_cdna() -> Self {
        Self {
            max_registers_per_thread: 256,
            max_shared_per_block: 65536,     // LDS
            target_occupancy: 0.5,
            warp_size: 64,                   // Wavefront
            max_warps_per_sm: 32,            // Waves per CU
            register_file_size: 65536,
            max_threads_per_block: 1024,
            max_blocks_per_sm: 16,
            shared_memory_banks: 32,
            compute_capability: (0, 0),      // N/A for AMD
        }
    }

    /// Calculate maximum threads per SM given register usage
    pub fn max_threads_for_registers(&self, regs_per_thread: u32) -> u32 {
        if regs_per_thread == 0 {
            return self.max_warps_per_sm * self.warp_size;
        }
        let threads = self.register_file_size / regs_per_thread;
        // Round down to warp granularity
        (threads / self.warp_size) * self.warp_size
    }

    /// Calculate maximum blocks per SM given shared memory usage
    pub fn max_blocks_for_shared(&self, shared_per_block: u32) -> u32 {
        if shared_per_block == 0 {
            return self.max_blocks_per_sm;
        }
        // Most SMs have ~164KB shared memory
        let sm_shared = self.max_shared_per_block;
        (sm_shared / shared_per_block).min(self.max_blocks_per_sm)
    }
}

// ============================================================================
// GPU LIVE INTERVAL
// ============================================================================

/// Extends CPU LiveInterval with GPU-specific considerations.
///
/// GPU register allocation must consider factors that don't exist on CPUs:
/// - Divergence risk from conditional branches
/// - Bank conflicts in shared memory access
/// - Memory coalescing requirements
#[derive(Debug, Clone)]
pub struct GpuLiveInterval {
    /// The underlying CPU-style live interval
    pub base: LiveInterval,

    // === GPU-Specific Fields ===

    /// Risk of warp divergence from confidence-dependent branches [0.0, 1.0].
    ///
    /// Values with high confidence variance may cause threads in a warp to
    /// take different branch paths. High divergence risk values should be
    /// prioritized for register allocation to minimize divergence overhead.
    pub divergence_risk: f64,

    /// Risk of shared memory bank conflicts [0.0, 1.0].
    ///
    /// Values accessed by multiple threads that map to the same bank cause
    /// serialization. High bank conflict risk suggests register allocation.
    pub bank_conflict_risk: f64,

    /// Memory coalescing score [0.0, 1.0].
    ///
    /// Higher scores indicate the value participates in coalesced memory
    /// accesses. Good coalescing means spilling is relatively cheap.
    pub coalescing_score: f64,

    /// Whether this value is used in a divergent control flow path
    pub in_divergent_region: bool,

    /// Memory space this value is associated with (for spill decisions)
    pub memory_space: GpuMemorySpace,

    /// Number of threads that access this value (1 = private, N = shared)
    pub thread_sharing: u32,

    /// Estimated spill cost in cycles (higher = prefer register)
    pub spill_cost_cycles: f64,
}

/// GPU memory spaces for spill location decisions
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuMemorySpace {
    /// Per-thread register (fastest, most constrained)
    #[default]
    Register,
    /// Per-thread local memory (L1 cached, slow)
    Local,
    /// Per-block shared memory (fast, limited)
    Shared,
    /// Global memory (L2 cached, slow)
    Global,
    /// Constant memory (broadcast, read-only)
    Constant,
}

impl GpuLiveInterval {
    /// Create a GPU interval from a CPU interval with default GPU metrics
    pub fn from_cpu(base: LiveInterval) -> Self {
        Self {
            base,
            divergence_risk: 0.0,
            bank_conflict_risk: 0.0,
            coalescing_score: 1.0, // Assume good coalescing by default
            in_divergent_region: false,
            memory_space: GpuMemorySpace::Register,
            thread_sharing: 1,
            spill_cost_cycles: 200.0, // Default global memory latency
        }
    }

    /// Create a GPU interval with epistemic-derived divergence risk
    pub fn with_epistemic(base: LiveInterval, confidence_variance: f64) -> Self {
        let mut interval = Self::from_cpu(base);
        interval.divergence_risk = divergence_probability(confidence_variance);
        interval
    }

    /// Set divergence risk from confidence variance
    pub fn set_divergence_from_confidence(&mut self, confidence_variance: f64) {
        self.divergence_risk = divergence_probability(confidence_variance);
    }

    /// Calculate spill cost considering GPU memory hierarchy
    pub fn calculate_spill_cost(&self, budget: &GpuResourceBudget) -> f64 {
        let base_cost = match self.memory_space {
            GpuMemorySpace::Register => 0.0,     // No spill
            GpuMemorySpace::Shared => 5.0,       // ~5 cycles
            GpuMemorySpace::Local => 50.0,       // L1 hit
            GpuMemorySpace::Global => 200.0,     // L2 hit
            GpuMemorySpace::Constant => 1.0,     // Broadcast
        };

        // Add bank conflict penalty for shared memory
        let conflict_penalty = if self.memory_space == GpuMemorySpace::Shared {
            self.bank_conflict_risk * (budget.shared_memory_banks as f64)
        } else {
            0.0
        };

        // Add divergence penalty
        let divergence_penalty = if self.in_divergent_region {
            self.divergence_risk * (budget.warp_size as f64)
        } else {
            0.0
        };

        base_cost + conflict_penalty + divergence_penalty
    }
}

// ============================================================================
// GPU ATTENTION SCORE (Multi-Objective)
// ============================================================================

/// Multi-objective attention score for GPU register allocation.
///
/// Unlike CPU allocation where we optimize a single metric, GPU allocation
/// must balance multiple competing objectives that interact non-linearly.
#[derive(Debug, Clone, Default)]
pub struct GpuAttentionScore {
    /// Register pressure contribution [0.0, 1.0].
    /// Higher = uses more registers, should consider spilling.
    pub register_pressure: f64,

    /// Occupancy impact [0.0, 1.0].
    /// Higher = keeping in register hurts occupancy more.
    pub occupancy_impact: f64,

    /// Shared memory cost [0.0, 1.0].
    /// Higher = expensive to spill to shared memory.
    pub shared_mem_cost: f64,

    /// Divergence cost [0.0, 1.0].
    /// Higher = spilling may cause/worsen divergence.
    pub divergence_cost: f64,

    /// Epistemic weight from confidence/provenance [0.0, 1.0].
    /// Higher = more important to keep due to epistemic properties.
    pub epistemic_weight: f64,

    /// Classic attention score (from CPU allocator)
    pub classic_score: f64,
}

impl GpuAttentionScore {
    /// Compute weighted sum of all objectives
    pub fn weighted_sum(&self, config: &GpuAttentionConfig) -> f64 {
        // Note: register_pressure and occupancy_impact are "costs" (higher = worse for keeping)
        // while the others are "benefits" (higher = better for keeping)
        let cost = config.w_register * self.register_pressure
                 + config.w_occupancy * self.occupancy_impact;

        let benefit = config.w_shared * (1.0 - self.shared_mem_cost)
                    + config.w_divergence * (1.0 - self.divergence_cost)
                    + config.w_epistemic * self.epistemic_weight
                    + self.classic_score;

        benefit - cost
    }

    /// Check Pareto dominance: does self dominate other?
    ///
    /// Self dominates other if:
    /// - Self is at least as good in all objectives
    /// - Self is strictly better in at least one objective
    ///
    /// For allocation: higher scores are better (more reason to keep in register)
    pub fn dominates(&self, other: &Self) -> bool {
        let self_scores = [
            -self.register_pressure, // Negate costs
            -self.occupancy_impact,
            1.0 - self.shared_mem_cost,
            1.0 - self.divergence_cost,
            self.epistemic_weight,
        ];
        let other_scores = [
            -other.register_pressure,
            -other.occupancy_impact,
            1.0 - other.shared_mem_cost,
            1.0 - other.divergence_cost,
            other.epistemic_weight,
        ];

        let mut strictly_better = false;

        for (s, o) in self_scores.iter().zip(other_scores.iter()) {
            if s < o {
                return false; // Other is better in this objective
            }
            if s > o {
                strictly_better = true;
            }
        }

        strictly_better
    }

    /// Compute score for a GPU live interval
    pub fn compute(
        interval: &GpuLiveInterval,
        budget: &GpuResourceBudget,
        config: &GpuAttentionConfig,
        current_reg_usage: u32,
    ) -> Self {
        // Calculate register pressure: how much would this add to pressure?
        let reg_pressure = if current_reg_usage >= budget.max_registers_per_thread / 2 {
            // Above 50% usage, pressure increases non-linearly
            let ratio = current_reg_usage as f64 / budget.max_registers_per_thread as f64;
            (ratio * 2.0 - 1.0).max(0.0).min(1.0)
        } else {
            0.0
        };

        // Calculate occupancy impact
        let current_occupancy = occupancy_for_registers(current_reg_usage, budget);
        let new_occupancy = occupancy_for_registers(current_reg_usage + 1, budget);
        let occupancy_impact = (current_occupancy - new_occupancy).max(0.0);

        // Shared memory cost based on spill estimate
        let spill_cost = interval.calculate_spill_cost(budget);
        let shared_mem_cost = (spill_cost / 200.0).min(1.0); // Normalize to [0,1]

        // Divergence cost
        let divergence_cost = interval.divergence_risk;

        // Epistemic weight from base interval
        let epistemic_weight = interval.base.epistemic_weight;

        // Classic score from CPU allocator
        let cpu_config = AttentionConfig::default();
        let classic_score = interval.base.attention_score(&cpu_config);

        Self {
            register_pressure: reg_pressure,
            occupancy_impact,
            shared_mem_cost,
            divergence_cost,
            epistemic_weight,
            classic_score,
        }
    }
}

// ============================================================================
// GPU ATTENTION CONFIG
// ============================================================================

/// Configuration weights for GPU attention-based allocation.
///
/// These weights determine how the multi-objective score is combined
/// into a single allocation decision.
#[derive(Debug, Clone)]
pub struct GpuAttentionConfig {
    /// Weight for register pressure objective
    pub w_register: f64,

    /// Weight for occupancy impact objective
    pub w_occupancy: f64,

    /// Weight for shared memory cost objective
    pub w_shared: f64,

    /// Weight for divergence cost objective
    pub w_divergence: f64,

    /// Weight for epistemic importance objective
    pub w_epistemic: f64,

    /// Minimum acceptable occupancy [0.0, 1.0]
    pub min_occupancy: f64,

    /// Maximum acceptable spill cost
    pub max_spill_cost: f64,

    /// Enable Pareto-based selection (vs weighted sum)
    pub use_pareto: bool,
}

impl Default for GpuAttentionConfig {
    fn default() -> Self {
        Self {
            w_register: 0.3,
            w_occupancy: 0.4,  // Occupancy is crucial on GPUs
            w_shared: 0.2,
            w_divergence: 0.3,
            w_epistemic: 0.4,
            min_occupancy: 0.25,
            max_spill_cost: 1000.0,
            use_pareto: false,
        }
    }
}

impl GpuAttentionConfig {
    /// Configuration optimized for compute-bound kernels
    pub fn compute_bound() -> Self {
        Self {
            w_register: 0.2,
            w_occupancy: 0.3,  // Less focus on occupancy
            w_shared: 0.3,
            w_divergence: 0.4,
            w_epistemic: 0.5,
            min_occupancy: 0.125, // Can tolerate low occupancy
            max_spill_cost: 500.0,
            use_pareto: false,
        }
    }

    /// Configuration optimized for memory-bound kernels
    pub fn memory_bound() -> Self {
        Self {
            w_register: 0.4,
            w_occupancy: 0.6,  // High occupancy crucial for hiding latency
            w_shared: 0.1,
            w_divergence: 0.2,
            w_epistemic: 0.3,
            min_occupancy: 0.5,
            max_spill_cost: 200.0,
            use_pareto: false,
        }
    }

    /// Configuration for epistemic-heavy scientific kernels
    pub fn scientific() -> Self {
        Self {
            w_register: 0.25,
            w_occupancy: 0.35,
            w_shared: 0.2,
            w_divergence: 0.5,  // Divergence from confidence variance
            w_epistemic: 0.7,   // Prioritize epistemic values
            min_occupancy: 0.25,
            max_spill_cost: 300.0,
            use_pareto: true,   // Use Pareto for multi-objective
        }
    }
}

// ============================================================================
// OCCUPANCY COST MODEL
// ============================================================================

/// Calculate occupancy given register usage per thread.
///
/// This models the non-linear relationship between registers and occupancy
/// on NVIDIA GPUs (similar for AMD with different constants).
///
/// # Returns
/// Occupancy as a fraction [0.0, 1.0] where 1.0 = max warps running
pub fn occupancy_for_registers(regs: u32, budget: &GpuResourceBudget) -> f64 {
    if regs == 0 {
        return 1.0;
    }

    // Calculate max threads possible given register usage
    let max_threads = budget.max_threads_for_registers(regs);

    // Calculate max warps
    let max_warps = max_threads / budget.warp_size;

    // Occupancy relative to hardware maximum
    max_warps as f64 / budget.max_warps_per_sm as f64
}

/// Calculate the optimal register target to achieve minimum occupancy.
///
/// This is the inverse of occupancy_for_registers: given a minimum
/// occupancy constraint, what's the maximum registers we can use?
///
/// # Returns
/// Maximum registers per thread that still achieves min_occupancy
pub fn optimal_register_target(budget: &GpuResourceBudget, min_occupancy: f64) -> u32 {
    // Target warps for this occupancy
    let target_warps = (min_occupancy * budget.max_warps_per_sm as f64).ceil() as u32;
    let target_threads = target_warps * budget.warp_size;

    if target_threads == 0 {
        return budget.max_registers_per_thread;
    }

    // Registers per thread = total registers / threads
    let max_regs = budget.register_file_size / target_threads;

    // Clamp to hardware limit
    max_regs.min(budget.max_registers_per_thread)
}

/// Calculate occupancy considering both registers and shared memory.
///
/// The true occupancy is the minimum of register-limited and shared-limited.
pub fn combined_occupancy(
    regs_per_thread: u32,
    shared_per_block: u32,
    threads_per_block: u32,
    budget: &GpuResourceBudget,
) -> f64 {
    // Register-limited occupancy
    let reg_occupancy = occupancy_for_registers(regs_per_thread, budget);

    // Shared memory-limited occupancy
    let max_blocks = budget.max_blocks_for_shared(shared_per_block);
    let threads_per_sm = max_blocks * threads_per_block;
    let warps_per_sm = threads_per_sm / budget.warp_size;
    let shared_occupancy = warps_per_sm as f64 / budget.max_warps_per_sm as f64;

    // True occupancy is the minimum
    reg_occupancy.min(shared_occupancy)
}

// ============================================================================
// DIVERGENCE COST MODEL
// ============================================================================

/// Calculate probability of warp divergence from confidence variance.
///
/// Values with high confidence variance lead to data-dependent branches
/// where threads in a warp take different paths.
///
/// # Model
/// P(divergence) = 1 - exp(-k * variance)
/// where k is a tuning parameter based on branch structure
///
/// # Arguments
/// * `confidence_variance` - Variance of confidence values [0.0, 1.0]
///
/// # Returns
/// Probability of divergence [0.0, 1.0]
pub fn divergence_probability(confidence_variance: f64) -> f64 {
    // Higher variance → higher divergence probability
    // k = 4.0 means 50% divergence at variance = 0.17
    let k = 4.0;
    1.0 - (-k * confidence_variance).exp()
}

/// Calculate the cost of divergence in terms of wasted cycles.
///
/// When a warp diverges, each path must be executed serially.
/// The cost depends on how many threads take each path.
///
/// # Arguments
/// * `divergence_prob` - Probability of divergence
/// * `warp_size` - Threads per warp (32 for NVIDIA)
/// * `branch_depth` - Nesting depth of divergent branches
///
/// # Returns
/// Relative cost multiplier (1.0 = no divergence, up to warp_size)
pub fn divergence_cost(divergence_prob: f64, warp_size: u32, branch_depth: u32) -> f64 {
    if divergence_prob < 0.01 {
        return 1.0; // Negligible divergence
    }

    // Expected threads per path assuming uniform distribution
    // With probability p of divergence, expected cost is:
    // cost = p * (2^depth) + (1-p) * 1
    let paths = (2_u32.pow(branch_depth) as f64).min(warp_size as f64);
    let cost = divergence_prob * paths + (1.0 - divergence_prob) * 1.0;

    cost
}

/// Calculate expected active threads per cycle given divergence.
///
/// This is useful for estimating ALU utilization.
pub fn active_threads_ratio(divergence_prob: f64, warp_size: u32) -> f64 {
    // At 50% divergence, expect ~half the threads active on average
    // Model: active = warp_size * (1 - divergence_prob/2)
    (warp_size as f64) * (1.0 - divergence_prob / 2.0) / (warp_size as f64)
}

// ============================================================================
// GPU SPILL SELECTOR
// ============================================================================

/// Selects the best spill victim from active GPU intervals.
///
/// Unlike CPU spill selection which primarily considers interval length
/// and use density, GPU spill selection must balance:
/// - Register pressure relief
/// - Occupancy improvement
/// - Divergence avoidance
/// - Epistemic importance
///
/// # Returns
/// Index into `active` slice of the best spill candidate, or None if
/// no suitable victim exists.
pub fn select_spill_victim(
    active: &[GpuLiveInterval],
    budget: &GpuResourceBudget,
    config: &GpuAttentionConfig,
    current_reg_usage: u32,
) -> Option<usize> {
    if active.is_empty() {
        return None;
    }

    // Calculate scores for all candidates
    let mut candidates: Vec<(usize, GpuAttentionScore)> = active
        .iter()
        .enumerate()
        .map(|(idx, interval)| {
            let score = GpuAttentionScore::compute(interval, budget, config, current_reg_usage);
            (idx, score)
        })
        .collect();

    if config.use_pareto {
        // Pareto-based selection: find non-dominated set then pick lowest weighted sum
        select_spill_pareto(&mut candidates, config)
    } else {
        // Weighted sum selection: pick lowest weighted sum
        select_spill_weighted(&candidates, config)
    }
}

/// Select spill victim using weighted sum of objectives.
fn select_spill_weighted(
    candidates: &[(usize, GpuAttentionScore)],
    config: &GpuAttentionConfig,
) -> Option<usize> {
    candidates
        .iter()
        .min_by(|(_, a), (_, b)| {
            let score_a = a.weighted_sum(config);
            let score_b = b.weighted_sum(config);
            score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
        })
        .map(|(idx, _)| *idx)
}

/// Select spill victim using Pareto dominance.
///
/// 1. Find the Pareto frontier (non-dominated solutions)
/// 2. Among the frontier, select by weighted sum
fn select_spill_pareto(
    candidates: &mut [(usize, GpuAttentionScore)],
    config: &GpuAttentionConfig,
) -> Option<usize> {
    if candidates.is_empty() {
        return None;
    }

    // Find Pareto frontier
    let mut frontier: Vec<usize> = Vec::new();

    for i in 0..candidates.len() {
        let mut dominated = false;

        for j in 0..candidates.len() {
            if i != j && candidates[j].1.dominates(&candidates[i].1) {
                dominated = true;
                break;
            }
        }

        if !dominated {
            frontier.push(i);
        }
    }

    if frontier.is_empty() {
        // Fallback to weighted sum if all dominated somehow
        return select_spill_weighted(candidates, config);
    }

    // Among frontier, select lowest weighted sum
    frontier
        .iter()
        .min_by(|&&a, &&b| {
            let score_a = candidates[a].1.weighted_sum(config);
            let score_b = candidates[b].1.weighted_sum(config);
            score_a.partial_cmp(&score_b).unwrap_or(Ordering::Equal)
        })
        .map(|&i| candidates[i].0)
}

/// Batch spill selection for register pressure relief.
///
/// When we need to free multiple registers, it's more efficient to
/// select all victims at once considering their interactions.
///
/// # Arguments
/// * `active` - Currently active intervals
/// * `budget` - GPU resource constraints
/// * `config` - Allocation weights
/// * `target_regs` - Target register usage after spilling
///
/// # Returns
/// Indices of intervals to spill, in recommended spill order
pub fn select_spill_batch(
    active: &[GpuLiveInterval],
    budget: &GpuResourceBudget,
    config: &GpuAttentionConfig,
    current_reg_usage: u32,
    target_regs: u32,
) -> Vec<usize> {
    if current_reg_usage <= target_regs {
        return Vec::new();
    }

    let needed = (current_reg_usage - target_regs) as usize;
    let mut spill_list = Vec::with_capacity(needed);
    let mut remaining: Vec<_> = active.iter().cloned().enumerate().collect();

    for _ in 0..needed {
        if remaining.is_empty() {
            break;
        }

        // Select victim from remaining
        let intervals: Vec<_> = remaining.iter().map(|(_, iv)| iv.clone()).collect();
        let adjusted_usage = current_reg_usage.saturating_sub(spill_list.len() as u32);

        if let Some(local_idx) = select_spill_victim(&intervals, budget, config, adjusted_usage) {
            let (original_idx, _) = remaining.remove(local_idx);
            spill_list.push(original_idx);
        } else {
            break;
        }
    }

    spill_list
}

// ============================================================================
// GPU REGISTER PRESSURE ANALYSIS
// ============================================================================

/// Analyze register pressure across a GPU kernel's control flow.
#[derive(Debug, Clone, Default)]
pub struct GpuPressureAnalysis {
    /// Peak register pressure at each program point
    pub pressure_by_point: Vec<u32>,

    /// Maximum pressure across all points
    pub max_pressure: u32,

    /// Points where pressure exceeds target occupancy threshold
    pub pressure_hotspots: Vec<usize>,

    /// Estimated occupancy at peak pressure
    pub estimated_occupancy: f64,

    /// Spills needed to achieve target occupancy
    pub spills_for_target: u32,
}

impl GpuPressureAnalysis {
    /// Analyze pressure for a set of GPU intervals
    pub fn analyze(
        intervals: &[GpuLiveInterval],
        budget: &GpuResourceBudget,
        num_points: usize,
    ) -> Self {
        let mut pressure_by_point = vec![0u32; num_points];

        // Count live intervals at each point
        for interval in intervals {
            for point in interval.base.start..interval.base.end.min(num_points) {
                pressure_by_point[point] += 1;
            }
        }

        let max_pressure = *pressure_by_point.iter().max().unwrap_or(&0);

        // Find target for minimum occupancy
        let target = optimal_register_target(budget, budget.target_occupancy);

        let pressure_hotspots: Vec<usize> = pressure_by_point
            .iter()
            .enumerate()
            .filter(|(_, p)| **p > target)
            .map(|(i, _)| i)
            .collect();

        let estimated_occupancy = occupancy_for_registers(max_pressure, budget);
        let spills_for_target = max_pressure.saturating_sub(target);

        Self {
            pressure_by_point,
            max_pressure,
            pressure_hotspots,
            estimated_occupancy,
            spills_for_target,
        }
    }

    /// Check if target occupancy can be achieved
    pub fn meets_occupancy_target(&self, budget: &GpuResourceBudget) -> bool {
        self.estimated_occupancy >= budget.target_occupancy
    }
}

// ============================================================================
// GPU ALLOCATION METRICS
// ============================================================================

/// Metrics collected during GPU register allocation.
#[derive(Debug, Clone, Default)]
pub struct GpuAllocationMetrics {
    /// Total intervals allocated
    pub total_intervals: usize,

    /// Intervals kept in registers
    pub in_registers: usize,

    /// Intervals spilled to shared memory
    pub spilled_to_shared: usize,

    /// Intervals spilled to local memory
    pub spilled_to_local: usize,

    /// Final register usage per thread
    pub final_registers: u32,

    /// Final shared memory usage per block
    pub final_shared: u32,

    /// Achieved occupancy
    pub achieved_occupancy: f64,

    /// High-confidence values that were spilled
    pub high_confidence_spills: usize,

    /// Low-confidence values that were kept
    pub low_confidence_kept: usize,

    /// Divergence-prone values spilled (good)
    pub divergent_spills: usize,
}

impl GpuAllocationMetrics {
    /// Calculate epistemic quality score for the allocation
    pub fn epistemic_quality(&self) -> f64 {
        if self.total_intervals == 0 {
            return 1.0;
        }

        // Good: low confidence spills, high confidence kept
        // Bad: high confidence spills, low confidence kept
        let good = (self.total_intervals - self.high_confidence_spills) as f64
                 + self.divergent_spills as f64;
        let bad = self.high_confidence_spills as f64 + self.low_confidence_kept as f64;

        (good + 1.0) / (bad + 1.0)
    }

    /// Calculate overall allocation quality
    pub fn overall_quality(&self, target_occupancy: f64) -> f64 {
        let occupancy_score = self.achieved_occupancy / target_occupancy;
        let epistemic_score = self.epistemic_quality();

        // Geometric mean of the two scores
        (occupancy_score * epistemic_score).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_resource_budget_default() {
        let budget = GpuResourceBudget::default();
        assert_eq!(budget.max_registers_per_thread, 255);
        assert_eq!(budget.warp_size, 32);
        assert_eq!(budget.max_warps_per_sm, 64);
    }

    #[test]
    fn test_occupancy_for_registers() {
        let budget = GpuResourceBudget::default();

        // 32 registers/thread → 2048 threads → 64 warps → 100%
        let occ_32 = occupancy_for_registers(32, &budget);
        assert!((occ_32 - 1.0).abs() < 0.01, "Expected ~100% occupancy at 32 regs");

        // 64 registers/thread → 1024 threads → 32 warps → 50%
        let occ_64 = occupancy_for_registers(64, &budget);
        assert!((occ_64 - 0.5).abs() < 0.01, "Expected ~50% occupancy at 64 regs");

        // 128 registers/thread → 512 threads → 16 warps → 25%
        let occ_128 = occupancy_for_registers(128, &budget);
        assert!((occ_128 - 0.25).abs() < 0.01, "Expected ~25% occupancy at 128 regs");
    }

    #[test]
    fn test_optimal_register_target() {
        let budget = GpuResourceBudget::default();

        // For 50% occupancy, should target ~64 regs
        let target_50 = optimal_register_target(&budget, 0.5);
        assert!(target_50 >= 32 && target_50 <= 64, "Expected ~64 regs for 50% occupancy");

        // For 25% occupancy, can use more registers
        let target_25 = optimal_register_target(&budget, 0.25);
        assert!(target_25 > target_50, "Lower occupancy should allow more registers");
    }

    #[test]
    fn test_divergence_probability() {
        // No variance → no divergence
        let p0 = divergence_probability(0.0);
        assert!(p0.abs() < 0.01);

        // High variance → high divergence
        let p_high = divergence_probability(1.0);
        assert!(p_high > 0.9);

        // Moderate variance → moderate divergence
        let p_mid = divergence_probability(0.25);
        assert!(p_mid > 0.3 && p_mid < 0.8);
    }

    #[test]
    fn test_divergence_cost() {
        let warp_size = 32;

        // No divergence → cost 1.0
        let cost_none = divergence_cost(0.0, warp_size, 1);
        assert!((cost_none - 1.0).abs() < 0.01);

        // 50% divergence, depth 1 → cost ~1.5
        let cost_50 = divergence_cost(0.5, warp_size, 1);
        assert!(cost_50 > 1.0 && cost_50 < 2.0);

        // High divergence, deep nesting → high cost
        let cost_high = divergence_cost(0.8, warp_size, 3);
        assert!(cost_high > cost_50);
    }

    #[test]
    fn test_gpu_attention_score_dominates() {
        let score_a = GpuAttentionScore {
            register_pressure: 0.3,
            occupancy_impact: 0.2,
            shared_mem_cost: 0.4,
            divergence_cost: 0.1,
            epistemic_weight: 0.8,
            classic_score: 1.0,
        };

        let score_b = GpuAttentionScore {
            register_pressure: 0.4, // Worse
            occupancy_impact: 0.3, // Worse
            shared_mem_cost: 0.5, // Worse
            divergence_cost: 0.2, // Worse
            epistemic_weight: 0.7, // Worse
            classic_score: 0.9,
        };

        assert!(score_a.dominates(&score_b));
        assert!(!score_b.dominates(&score_a));
    }

    #[test]
    fn test_gpu_live_interval_spill_cost() {
        let base = LiveInterval::new(ValueId::new(0), 0, SirType::f64());
        let mut interval = GpuLiveInterval::from_cpu(base);
        let budget = GpuResourceBudget::default();

        // Register → no cost
        interval.memory_space = GpuMemorySpace::Register;
        assert_eq!(interval.calculate_spill_cost(&budget), 0.0);

        // Shared → low cost
        interval.memory_space = GpuMemorySpace::Shared;
        let shared_cost = interval.calculate_spill_cost(&budget);
        assert!(shared_cost > 0.0 && shared_cost < 50.0);

        // Global → high cost
        interval.memory_space = GpuMemorySpace::Global;
        let global_cost = interval.calculate_spill_cost(&budget);
        assert!(global_cost >= 200.0);
    }

    #[test]
    fn test_pressure_analysis() {
        let budget = GpuResourceBudget::default();
        let base = LiveInterval::new(ValueId::new(0), 0, SirType::f64());
        let interval = GpuLiveInterval::from_cpu(base);

        // Create overlapping intervals
        let intervals: Vec<GpuLiveInterval> = (0..100)
            .map(|i| {
                let mut base = LiveInterval::new(ValueId::new(i), i as usize, SirType::f64());
                base.extend_to((i + 20) as usize);
                GpuLiveInterval::from_cpu(base)
            })
            .collect();

        let analysis = GpuPressureAnalysis::analyze(&intervals, &budget, 120);

        assert!(analysis.max_pressure > 0);
        assert!(analysis.estimated_occupancy > 0.0);
    }

    #[test]
    fn test_select_spill_victim() {
        let budget = GpuResourceBudget::default();
        let config = GpuAttentionConfig::default();

        // Create intervals with varying scores
        let intervals: Vec<GpuLiveInterval> = vec![
            {
                let mut base = LiveInterval::new(ValueId::new(0), 0, SirType::f64());
                base.max_confidence = 0.9;
                base.epistemic_weight = 0.9;
                let mut gpu = GpuLiveInterval::from_cpu(base);
                gpu.divergence_risk = 0.1;
                gpu
            },
            {
                let mut base = LiveInterval::new(ValueId::new(1), 0, SirType::f64());
                base.max_confidence = 0.3;
                base.epistemic_weight = 0.3;
                let mut gpu = GpuLiveInterval::from_cpu(base);
                gpu.divergence_risk = 0.8;
                gpu
            },
        ];

        let victim = select_spill_victim(&intervals, &budget, &config, 64);

        // Should select interval 1 (lower epistemic weight, higher divergence)
        assert_eq!(victim, Some(1));
    }

    #[test]
    fn test_nvidia_budget_variants() {
        let volta = GpuResourceBudget::nvidia(7, 0);
        assert_eq!(volta.compute_capability, (7, 0));
        assert_eq!(volta.max_shared_per_block, 98304);

        let hopper = GpuResourceBudget::nvidia(9, 0);
        assert_eq!(hopper.compute_capability, (9, 0));
        assert!(hopper.max_shared_per_block > volta.max_shared_per_block);
    }

    #[test]
    fn test_combined_occupancy() {
        let budget = GpuResourceBudget::default();

        // Low register, low shared → high occupancy
        let occ_low = combined_occupancy(32, 4096, 256, &budget);
        assert!(occ_low > 0.5);

        // High register, high shared → low occupancy
        let occ_high = combined_occupancy(128, 65536, 256, &budget);
        assert!(occ_high < occ_low);
    }

    #[test]
    fn test_allocation_metrics() {
        let mut metrics = GpuAllocationMetrics::default();
        metrics.total_intervals = 100;
        metrics.in_registers = 80;
        metrics.spilled_to_shared = 15;
        metrics.spilled_to_local = 5;
        metrics.high_confidence_spills = 2;
        metrics.divergent_spills = 10;
        metrics.achieved_occupancy = 0.5;

        let quality = metrics.epistemic_quality();
        assert!(quality > 1.0, "Quality should be good with few high-confidence spills");

        let overall = metrics.overall_quality(0.5);
        // Geometric mean of occupancy ratio (1.0) and epistemic quality (>1.0)
        // can be > 1.0 but should be reasonable
        assert!(overall > 0.0, "Overall quality should be positive");
        assert!(overall.is_finite(), "Overall quality should be finite");
    }
}
