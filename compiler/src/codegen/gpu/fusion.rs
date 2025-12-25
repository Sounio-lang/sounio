//! GPU Kernel Fusion
//!
//! Implements kernel fusion optimizations to reduce launch overhead and improve
//! memory locality by merging sequential (vertical) and parallel (horizontal) kernels.
//!
//! # Fusion Types
//!
//! - **Vertical Fusion**: Merge producer-consumer kernel pairs where the output
//!   of kernel K1 feeds directly into kernel K2. Eliminates intermediate global
//!   memory traffic.
//!
//! - **Horizontal Fusion**: Merge independent kernels with compatible launch
//!   configurations. Reduces kernel launch overhead.
//!
//! - **Diamond Fusion**: Merge fork-join patterns (A -> B, A -> C, B&C -> D)
//!   into a single kernel with internal synchronization.
//!
//! - **Loop Fusion**: Merge kernels inside a loop body with optional unrolling.
//!
//! # Architecture
//!
//! ```text
//! GpuModule + GpuGraph → FusionAnalysis → FusionPlan → FusedGpuModule
//!                             │               │
//!                     DependencyGraph    CostModel
//! ```
//!
//! # 6-Pass Pipeline
//!
//! 1. Build kernel registry (name → KernelId mapping)
//! 2. Build dependency graph from GpuGraph edges
//! 3. Find fusion candidates (vertical + horizontal)
//! 4. Evaluate candidates with cost model
//! 5. Select non-conflicting fusion groups (greedy)
//! 6. Plan transformations (value/block remapping)

use rustc_hash::{FxHashMap, FxHashSet};
use std::fmt;

use super::graph::{BufferId, GpuGraph, GraphNodeId, GraphNodeType};
use super::ir::{
    BlockId, CudaArch, GpuBlock, GpuKernel, GpuModule, GpuOp, GpuTarget, GpuTerminator, GpuType,
    MemorySpace, SharedMemDecl, ValueId,
};

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a kernel in the fusion analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KernelId(pub u32);

impl fmt::Display for KernelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "K{}", self.0)
    }
}

/// Unique identifier for a fusion candidate
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusionCandidateId(pub u32);

impl fmt::Display for FusionCandidateId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FC{}", self.0)
    }
}

/// Unique identifier for a fusion group
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FusionGroupId(pub u32);

impl fmt::Display for FusionGroupId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FG{}", self.0)
    }
}

/// Type of kernel fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionType {
    /// Producer-consumer fusion (output of K1 → input of K2)
    Vertical,

    /// Independent kernels with same launch config
    Horizontal,

    /// Fork-join pattern (A → B, A → C, B&C → D)
    Diamond,

    /// Kernels in a loop body with optional unrolling
    LoopFusion {
        /// Number of iterations to unroll (1 = no unrolling)
        unroll_factor: u32,
    },
}

impl fmt::Display for FusionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionType::Vertical => write!(f, "vertical"),
            FusionType::Horizontal => write!(f, "horizontal"),
            FusionType::Diamond => write!(f, "diamond"),
            FusionType::LoopFusion { unroll_factor } => {
                write!(f, "loop(unroll={})", unroll_factor)
            }
        }
    }
}

// ============================================================================
// Resource Estimation
// ============================================================================

/// Resource requirements for a kernel or fused kernel group
#[derive(Debug, Clone, Default)]
pub struct ResourceEstimate {
    /// Estimated registers per thread
    pub registers_per_thread: u32,

    /// Total shared memory bytes
    pub shared_memory_bytes: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Estimated occupancy (0.0 - 1.0)
    pub occupancy: f64,

    /// Number of synchronization barriers
    pub barrier_count: u32,

    /// Estimated instruction count
    pub instruction_count: u32,

    /// Number of global memory loads
    pub global_loads: u32,

    /// Number of global memory stores
    pub global_stores: u32,
}

impl ResourceEstimate {
    /// Create a new empty resource estimate
    pub fn new() -> Self {
        Self::default()
    }

    /// Estimate resources for a kernel
    pub fn from_kernel(kernel: &GpuKernel) -> Self {
        let mut estimate = Self::new();

        // Count shared memory
        estimate.shared_memory_bytes = kernel.shared_mem_size;

        // Count instructions and memory ops
        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                estimate.instruction_count += 1;
                match op {
                    GpuOp::Load(_, MemorySpace::Global) => estimate.global_loads += 1,
                    GpuOp::Store(_, _, MemorySpace::Global) => estimate.global_stores += 1,
                    GpuOp::SyncThreads => estimate.barrier_count += 1,
                    _ => {}
                }
            }
        }

        // Rough register estimate: 2 registers per instruction
        estimate.registers_per_thread = (estimate.instruction_count * 2).min(255);

        // Default max threads
        estimate.max_threads_per_block = kernel.max_threads.unwrap_or(1024);

        // Rough occupancy estimate (simplified)
        estimate.occupancy = if estimate.shared_memory_bytes > 48 * 1024 {
            0.25
        } else if estimate.registers_per_thread > 64 {
            0.5
        } else {
            1.0
        };

        estimate
    }

    /// Combine two estimates (for fusion)
    pub fn combine(&self, other: &Self) -> Self {
        Self {
            registers_per_thread: (self.registers_per_thread + other.registers_per_thread).min(255),
            shared_memory_bytes: self.shared_memory_bytes + other.shared_memory_bytes,
            max_threads_per_block: self.max_threads_per_block.min(other.max_threads_per_block),
            occupancy: self.occupancy.min(other.occupancy),
            barrier_count: self.barrier_count + other.barrier_count + 1, // +1 for inter-kernel sync
            instruction_count: self.instruction_count + other.instruction_count,
            global_loads: self.global_loads + other.global_loads,
            global_stores: self.global_stores + other.global_stores,
        }
    }
}

// ============================================================================
// Architecture Constraints
// ============================================================================

/// Hardware constraints for a specific GPU architecture
#[derive(Debug, Clone)]
pub struct ArchConstraints {
    /// Maximum registers per thread
    pub max_registers: u32,

    /// Maximum shared memory per block (bytes)
    pub max_shared_memory: u32,

    /// Maximum threads per block
    pub max_threads_per_block: u32,

    /// Warp size (32 for NVIDIA)
    pub warp_size: u32,

    /// L2 cache size (bytes)
    pub l2_cache_size: u32,

    /// Tensor Core generation (0 = no Tensor Cores)
    pub tensor_core_gen: u32,
}

impl ArchConstraints {
    /// Get constraints from CUDA compute capability
    pub fn from_cuda_cc(cc: (u32, u32)) -> Self {
        if let Some(arch) = CudaArch::from_compute_capability(cc) {
            Self {
                max_registers: arch.max_registers_per_thread(),
                max_shared_memory: arch.max_shared_memory(),
                max_threads_per_block: arch.max_threads_per_block(),
                warp_size: 32,
                l2_cache_size: arch.l2_cache_size(),
                tensor_core_gen: arch.tensor_core_gen(),
            }
        } else {
            // Default to conservative Turing limits
            Self {
                max_registers: 255,
                max_shared_memory: 64 * 1024,
                max_threads_per_block: 1024,
                warp_size: 32,
                l2_cache_size: 6 * 1024 * 1024,
                tensor_core_gen: 2,
            }
        }
    }

    /// Get constraints from GPU target
    pub fn from_target(target: &GpuTarget) -> Self {
        match target {
            GpuTarget::Cuda { compute_capability } => Self::from_cuda_cc(*compute_capability),
            GpuTarget::Metal { gpu_family } => Self {
                max_registers: 255,
                max_shared_memory: gpu_family.max_threadgroup_memory(),
                max_threads_per_block: gpu_family.max_threads_per_threadgroup(),
                warp_size: gpu_family.simd_width(),
                l2_cache_size: 0, // Unknown for Metal
                tensor_core_gen: 0,
            },
            _ => Self::default(),
        }
    }

    /// Check if a resource estimate fits within constraints
    pub fn fits(&self, estimate: &ResourceEstimate) -> bool {
        estimate.registers_per_thread <= self.max_registers
            && estimate.shared_memory_bytes <= self.max_shared_memory
            && estimate.max_threads_per_block <= self.max_threads_per_block
    }
}

impl Default for ArchConstraints {
    fn default() -> Self {
        // Conservative defaults (Turing)
        Self {
            max_registers: 255,
            max_shared_memory: 64 * 1024,
            max_threads_per_block: 1024,
            warp_size: 32,
            l2_cache_size: 6 * 1024 * 1024,
            tensor_core_gen: 2,
        }
    }
}

// ============================================================================
// Fusion Candidate
// ============================================================================

/// A potential fusion opportunity
#[derive(Debug, Clone)]
pub struct FusionCandidate {
    /// Unique identifier
    pub id: FusionCandidateId,

    /// Kernels to be fused (in execution order)
    pub kernels: Vec<KernelId>,

    /// Type of fusion
    pub fusion_type: FusionType,

    /// Computed benefit score (higher = better)
    pub benefit_score: f64,

    /// Combined resource estimate
    pub resource_estimate: ResourceEstimate,

    /// Producer-consumer edges (producer_kernel, consumer_kernel, buffer)
    pub producer_consumer_edges: Vec<(KernelId, KernelId, BufferId)>,

    /// Whether this candidate is valid (passes constraint checks)
    pub valid: bool,

    /// Reason if invalid
    pub invalid_reason: Option<String>,
}

impl FusionCandidate {
    /// Create a new fusion candidate
    pub fn new(id: FusionCandidateId, kernels: Vec<KernelId>, fusion_type: FusionType) -> Self {
        Self {
            id,
            kernels,
            fusion_type,
            benefit_score: 0.0,
            resource_estimate: ResourceEstimate::default(),
            producer_consumer_edges: Vec::new(),
            valid: true,
            invalid_reason: None,
        }
    }

    /// Mark candidate as invalid
    pub fn invalidate(&mut self, reason: &str) {
        self.valid = false;
        self.invalid_reason = Some(reason.to_string());
    }
}

// ============================================================================
// Fusion Group
// ============================================================================

/// Launch configuration for a fused kernel
#[derive(Debug, Clone, Default)]
pub struct LaunchConfig {
    /// Grid dimensions
    pub grid: (u32, u32, u32),

    /// Block dimensions
    pub block: (u32, u32, u32),

    /// Dynamic shared memory size
    pub dynamic_shared_mem: u32,
}

/// Shared memory layout for fused kernel
#[derive(Debug, Clone)]
pub struct SharedMemLayout {
    /// Allocations: (name, offset, size, alignment)
    pub allocations: Vec<(String, u32, u32, u32)>,

    /// Total size
    pub total_size: u32,
}

impl SharedMemLayout {
    /// Create an empty layout
    pub fn new() -> Self {
        Self {
            allocations: Vec::new(),
            total_size: 0,
        }
    }

    /// Add an allocation with proper alignment
    pub fn add(&mut self, name: String, size: u32, alignment: u32) -> u32 {
        // Align current offset
        let aligned_offset = align_to(self.total_size, alignment);
        self.allocations
            .push((name, aligned_offset, size, alignment));
        self.total_size = aligned_offset + size;
        aligned_offset
    }
}

impl Default for SharedMemLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// Align value to alignment boundary
fn align_to(value: u32, alignment: u32) -> u32 {
    if alignment == 0 {
        return value;
    }
    (value + alignment - 1) & !(alignment - 1)
}

/// A selected group of kernels to fuse
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Unique identifier
    pub id: FusionGroupId,

    /// Kernels in this group (in execution order)
    pub kernels: Vec<KernelId>,

    /// Launch configuration for the fused kernel
    pub launch_config: LaunchConfig,

    /// Combined shared memory layout
    pub shared_mem_layout: SharedMemLayout,

    /// Block IDs where barriers should be inserted
    pub barrier_points: Vec<BlockId>,

    /// Value remapping: (original_kernel, original_value) → new_value
    pub value_map: FxHashMap<(KernelId, ValueId), ValueId>,

    /// Block remapping: (original_kernel, original_block) → new_block
    pub block_map: FxHashMap<(KernelId, BlockId), BlockId>,

    /// Name of the fused kernel
    pub fused_name: String,
}

impl FusionGroup {
    /// Create a new fusion group
    pub fn new(id: FusionGroupId, kernels: Vec<KernelId>) -> Self {
        let fused_name = format!(
            "fused_{}",
            kernels
                .iter()
                .map(|k| k.0.to_string())
                .collect::<Vec<_>>()
                .join("_")
        );

        Self {
            id,
            kernels,
            launch_config: LaunchConfig::default(),
            shared_mem_layout: SharedMemLayout::new(),
            barrier_points: Vec::new(),
            value_map: FxHashMap::default(),
            block_map: FxHashMap::default(),
            fused_name,
        }
    }
}

// ============================================================================
// Cost Model
// ============================================================================

/// Weights for the fusion cost model
#[derive(Debug, Clone)]
pub struct CostWeights {
    /// Weight for kernel launch overhead savings
    pub launch_overhead: f64,

    /// Weight for memory traffic reduction
    pub memory_traffic: f64,

    /// Weight for occupancy change
    pub occupancy: f64,

    /// Weight for synchronization overhead (negative impact)
    pub sync_overhead: f64,

    /// Weight for register pressure (negative impact)
    pub register_pressure: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            launch_overhead: 1.0,
            memory_traffic: 2.0,
            occupancy: 0.5,
            sync_overhead: -0.3,
            register_pressure: -0.2,
        }
    }
}

/// Cost model for evaluating fusion candidates
#[derive(Debug, Clone)]
pub struct FusionCostModel {
    /// Target GPU
    pub target: GpuTarget,

    /// Architecture constraints
    pub constraints: ArchConstraints,

    /// Cost weights
    pub weights: CostWeights,

    /// Minimum benefit threshold for fusion
    pub min_benefit_threshold: f64,
}

impl FusionCostModel {
    /// Create a new cost model for a target
    pub fn new(target: GpuTarget) -> Self {
        let constraints = ArchConstraints::from_target(&target);
        Self {
            target,
            constraints,
            weights: CostWeights::default(),
            min_benefit_threshold: 0.1,
        }
    }

    /// Create cost model from compute capability
    pub fn from_cuda_cc(cc: (u32, u32)) -> Self {
        Self::new(GpuTarget::Cuda {
            compute_capability: cc,
        })
    }

    /// Check if a candidate satisfies hardware constraints
    pub fn check_constraints(&self, candidate: &FusionCandidate) -> Result<(), String> {
        let estimate = &candidate.resource_estimate;

        if estimate.registers_per_thread > self.constraints.max_registers {
            return Err(format!(
                "Register usage {} exceeds max {}",
                estimate.registers_per_thread, self.constraints.max_registers
            ));
        }

        if estimate.shared_memory_bytes > self.constraints.max_shared_memory {
            return Err(format!(
                "Shared memory {} exceeds max {}",
                estimate.shared_memory_bytes, self.constraints.max_shared_memory
            ));
        }

        if estimate.max_threads_per_block > self.constraints.max_threads_per_block {
            return Err(format!(
                "Threads per block {} exceeds max {}",
                estimate.max_threads_per_block, self.constraints.max_threads_per_block
            ));
        }

        Ok(())
    }

    /// Evaluate the benefit score of a fusion candidate
    ///
    /// Score = W_launch * launch_savings
    ///       + W_memory * memory_savings
    ///       + W_occupancy * occupancy_delta
    ///       - W_sync * barrier_count
    ///       - W_register * register_increase
    pub fn evaluate(
        &self,
        candidate: &FusionCandidate,
        original_estimates: &[ResourceEstimate],
    ) -> f64 {
        let kernel_count = candidate.kernels.len() as f64;
        let estimate = &candidate.resource_estimate;

        // Launch savings: each fused kernel saves (n-1) launches
        let launch_savings = (kernel_count - 1.0) * self.weights.launch_overhead;

        // Memory traffic savings: estimate based on producer-consumer edges
        // Each vertical fusion eliminates intermediate buffer traffic
        let memory_savings = match candidate.fusion_type {
            FusionType::Vertical => {
                let edges = candidate.producer_consumer_edges.len() as f64;
                edges * self.weights.memory_traffic
            }
            FusionType::Diamond => {
                // Diamond patterns save more memory
                candidate.producer_consumer_edges.len() as f64 * self.weights.memory_traffic * 1.5
            }
            _ => 0.0,
        };

        // Occupancy impact
        let avg_original_occupancy: f64 =
            original_estimates.iter().map(|e| e.occupancy).sum::<f64>() / kernel_count;
        let occupancy_delta =
            (estimate.occupancy - avg_original_occupancy) * self.weights.occupancy;

        // Synchronization overhead
        let sync_cost = estimate.barrier_count as f64 * self.weights.sync_overhead;

        // Register pressure
        let avg_original_regs: u32 = original_estimates
            .iter()
            .map(|e| e.registers_per_thread)
            .sum::<u32>()
            / original_estimates.len() as u32;
        let reg_increase =
            (estimate.registers_per_thread as i32 - avg_original_regs as i32).max(0) as f64;
        let reg_cost = reg_increase * self.weights.register_pressure;

        launch_savings + memory_savings + occupancy_delta + sync_cost + reg_cost
    }

    /// Check if a candidate's benefit exceeds the threshold
    pub fn is_beneficial(&self, candidate: &FusionCandidate) -> bool {
        candidate.benefit_score >= self.min_benefit_threshold
    }
}

// ============================================================================
// Kernel Dependency Graph
// ============================================================================

/// Edge type in the dependency graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependencyType {
    /// Data dependency (output → input)
    DataFlow(BufferId),

    /// Control dependency
    Control,

    /// Memory ordering dependency
    MemoryOrder,
}

/// Dependency graph for kernels
#[derive(Debug, Clone)]
pub struct KernelDependencyGraph {
    /// Forward edges: kernel → [(dependent_kernel, dependency_type)]
    pub forward: FxHashMap<KernelId, Vec<(KernelId, DependencyType)>>,

    /// Backward edges: kernel → [(predecessor_kernel, dependency_type)]
    pub backward: FxHashMap<KernelId, Vec<(KernelId, DependencyType)>>,

    /// All kernels
    pub kernels: FxHashSet<KernelId>,
}

impl KernelDependencyGraph {
    /// Create an empty dependency graph
    pub fn new() -> Self {
        Self {
            forward: FxHashMap::default(),
            backward: FxHashMap::default(),
            kernels: FxHashSet::default(),
        }
    }

    /// Add a kernel to the graph
    pub fn add_kernel(&mut self, kernel: KernelId) {
        self.kernels.insert(kernel);
        self.forward.entry(kernel).or_default();
        self.backward.entry(kernel).or_default();
    }

    /// Add a dependency edge
    pub fn add_edge(&mut self, from: KernelId, to: KernelId, dep_type: DependencyType) {
        self.forward.entry(from).or_default().push((to, dep_type));
        self.backward.entry(to).or_default().push((from, dep_type));
    }

    /// Get successors of a kernel
    pub fn successors(
        &self,
        kernel: KernelId,
    ) -> impl Iterator<Item = (KernelId, DependencyType)> + '_ {
        self.forward
            .get(&kernel)
            .into_iter()
            .flat_map(|v| v.iter().copied())
    }

    /// Get predecessors of a kernel
    pub fn predecessors(
        &self,
        kernel: KernelId,
    ) -> impl Iterator<Item = (KernelId, DependencyType)> + '_ {
        self.backward
            .get(&kernel)
            .into_iter()
            .flat_map(|v| v.iter().copied())
    }

    /// Check if two kernels are independent (no path between them)
    pub fn are_independent(&self, k1: KernelId, k2: KernelId) -> bool {
        !self.has_path(k1, k2) && !self.has_path(k2, k1)
    }

    /// Check if there's a path from k1 to k2
    pub fn has_path(&self, from: KernelId, to: KernelId) -> bool {
        let mut visited = FxHashSet::default();
        let mut queue = vec![from];

        while let Some(current) = queue.pop() {
            if current == to {
                return true;
            }
            if visited.insert(current) {
                for (next, _) in self.successors(current) {
                    queue.push(next);
                }
            }
        }

        false
    }

    /// Get data flow edge between two kernels if it exists
    pub fn get_data_dependency(&self, from: KernelId, to: KernelId) -> Option<BufferId> {
        self.forward.get(&from).and_then(|edges| {
            edges.iter().find_map(|(k, dep)| {
                if *k == to {
                    match dep {
                        DependencyType::DataFlow(buf) => Some(*buf),
                        _ => None,
                    }
                } else {
                    None
                }
            })
        })
    }
}

impl Default for KernelDependencyGraph {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Fusion Configuration
// ============================================================================

/// Configuration for fusion analysis
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// Enable vertical (producer-consumer) fusion
    pub enable_vertical: bool,

    /// Enable horizontal (parallel) fusion
    pub enable_horizontal: bool,

    /// Enable diamond pattern fusion
    pub enable_diamond: bool,

    /// Enable loop fusion
    pub enable_loop_fusion: bool,

    /// Maximum kernels per fusion group
    pub max_kernels_per_group: usize,

    /// Minimum benefit threshold
    pub min_benefit: f64,

    /// Maximum chain length for vertical fusion
    pub max_chain_length: usize,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            enable_vertical: true,
            enable_horizontal: true,
            enable_diamond: true,
            enable_loop_fusion: true,
            max_kernels_per_group: 4,
            min_benefit: 0.1,
            max_chain_length: 3,
        }
    }
}

// ============================================================================
// Fusion Plan
// ============================================================================

/// The result of fusion analysis
#[derive(Debug, Clone)]
pub struct FusionPlan {
    /// Selected fusion groups
    pub groups: Vec<FusionGroup>,

    /// Kernels that remain unfused
    pub unfused_kernels: Vec<KernelId>,

    /// Total estimated benefit
    pub total_benefit: f64,

    /// Statistics
    pub stats: FusionStats,
}

/// Statistics from fusion analysis
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    /// Total candidates found
    pub candidates_found: usize,

    /// Candidates rejected by constraints
    pub candidates_rejected: usize,

    /// Candidates rejected by cost model
    pub candidates_below_threshold: usize,

    /// Groups formed
    pub groups_formed: usize,

    /// Kernels fused
    pub kernels_fused: usize,
}

// ============================================================================
// Fusion Analysis
// ============================================================================

/// Main fusion analysis engine
#[derive(Debug)]
pub struct FusionAnalysis {
    /// Configuration
    config: FusionConfig,

    /// Cost model
    cost_model: FusionCostModel,

    /// Kernel registry: name → KernelId
    kernel_registry: FxHashMap<String, KernelId>,

    /// Reverse registry: KernelId → name
    kernel_names: FxHashMap<KernelId, String>,

    /// Resource estimates for each kernel
    kernel_estimates: FxHashMap<KernelId, ResourceEstimate>,

    /// Dependency graph
    dep_graph: KernelDependencyGraph,

    /// Found candidates
    candidates: Vec<FusionCandidate>,

    /// Next candidate ID
    next_candidate_id: u32,

    /// Selected fusion groups
    fusion_groups: Vec<FusionGroup>,

    /// Next group ID
    next_group_id: u32,
}

impl FusionAnalysis {
    /// Create a new fusion analysis with default config
    pub fn new(target: GpuTarget) -> Self {
        Self::with_config(FusionConfig::default(), FusionCostModel::new(target))
    }

    /// Create a new fusion analysis with custom config
    pub fn with_config(config: FusionConfig, cost_model: FusionCostModel) -> Self {
        Self {
            config,
            cost_model,
            kernel_registry: FxHashMap::default(),
            kernel_names: FxHashMap::default(),
            kernel_estimates: FxHashMap::default(),
            dep_graph: KernelDependencyGraph::new(),
            candidates: Vec::new(),
            next_candidate_id: 0,
            fusion_groups: Vec::new(),
            next_group_id: 0,
        }
    }

    /// Run the full 6-pass analysis pipeline
    pub fn analyze(&mut self, module: &GpuModule, graph: &GpuGraph) -> FusionPlan {
        // Pass 1: Build kernel registry
        self.build_kernel_registry(module);

        // Pass 2: Build dependency graph
        self.build_dependency_graph(graph);

        // Pass 3: Find fusion candidates
        self.find_candidates();

        // Pass 4: Evaluate candidates with cost model
        self.evaluate_candidates();

        // Pass 5: Select non-conflicting fusion groups
        self.select_fusion_groups();

        // Pass 6: Plan transformations
        self.plan_transformations(module);

        // Build and return the fusion plan
        self.build_plan()
    }

    /// Pass 1: Build kernel registry from GpuModule
    fn build_kernel_registry(&mut self, module: &GpuModule) {
        let mut id = 0u32;
        for (name, kernel) in &module.kernels {
            let kernel_id = KernelId(id);
            self.kernel_registry.insert(name.clone(), kernel_id);
            self.kernel_names.insert(kernel_id, name.clone());
            self.kernel_estimates
                .insert(kernel_id, ResourceEstimate::from_kernel(kernel));
            self.dep_graph.add_kernel(kernel_id);
            id += 1;
        }
    }

    /// Pass 2: Build dependency graph from GpuGraph
    fn build_dependency_graph(&mut self, graph: &GpuGraph) {
        // Map graph nodes to kernel IDs
        let mut node_to_kernel: FxHashMap<GraphNodeId, KernelId> = FxHashMap::default();

        for node in &graph.nodes {
            if let GraphNodeType::Kernel(kernel_node) = &node.node_type
                && let Some(&kernel_id) = self.kernel_registry.get(&kernel_node.kernel_name)
            {
                node_to_kernel.insert(node.id, kernel_id);
            }
        }

        // Build edges from graph dependencies
        for (from_node, to_node) in &graph.edges {
            if let (Some(&from_kernel), Some(&to_kernel)) =
                (node_to_kernel.get(from_node), node_to_kernel.get(to_node))
            {
                // Check if there's a data flow (buffer connection)
                let dep_type = self
                    .find_data_dependency(graph, *from_node, *to_node)
                    .map(DependencyType::DataFlow)
                    .unwrap_or(DependencyType::Control);

                self.dep_graph.add_edge(from_kernel, to_kernel, dep_type);
            }
        }
    }

    /// Find data dependency between two graph nodes
    fn find_data_dependency(
        &self,
        graph: &GpuGraph,
        from: GraphNodeId,
        to: GraphNodeId,
    ) -> Option<BufferId> {
        let from_node = graph.get_node(from)?;
        let to_node = graph.get_node(to)?;

        // Check if from_node writes to a buffer that to_node reads
        if let GraphNodeType::Kernel(from_kernel) = &from_node.node_type
            && let GraphNodeType::Kernel(to_kernel) = &to_node.node_type
        {
            // Find common buffer (simplified: check if any buffer is shared)
            for from_arg in &from_kernel.args {
                for to_arg in &to_kernel.args {
                    if let (
                        super::graph::GraphKernelArg::Buffer(buf1),
                        super::graph::GraphKernelArg::Buffer(buf2),
                    ) = (from_arg, to_arg)
                        && buf1 == buf2
                    {
                        return Some(*buf1);
                    }
                }
            }
        }

        None
    }

    /// Pass 3: Find fusion candidates
    fn find_candidates(&mut self) {
        if self.config.enable_vertical {
            self.find_vertical_candidates();
        }

        if self.config.enable_horizontal {
            self.find_horizontal_candidates();
        }

        if self.config.enable_diamond {
            self.find_diamond_candidates();
        }
    }

    /// Find vertical (producer-consumer) fusion candidates
    fn find_vertical_candidates(&mut self) {
        for &kernel in &self.dep_graph.kernels.iter().copied().collect::<Vec<_>>() {
            for (successor, dep_type) in self.dep_graph.successors(kernel).collect::<Vec<_>>() {
                if let DependencyType::DataFlow(buffer) = dep_type {
                    // Check launch config compatibility
                    if self.have_compatible_configs(kernel, successor) {
                        let candidate_id = self.alloc_candidate_id();
                        let mut candidate = FusionCandidate::new(
                            candidate_id,
                            vec![kernel, successor],
                            FusionType::Vertical,
                        );
                        candidate
                            .producer_consumer_edges
                            .push((kernel, successor, buffer));

                        // Combine resource estimates
                        if let (Some(est1), Some(est2)) = (
                            self.kernel_estimates.get(&kernel),
                            self.kernel_estimates.get(&successor),
                        ) {
                            candidate.resource_estimate = est1.combine(est2);
                        }

                        self.candidates.push(candidate);
                    }
                }
            }
        }

        // Extend to chains if within limits
        self.extend_vertical_chains();
    }

    /// Extend vertical candidates to chains (A→B→C)
    fn extend_vertical_chains(&mut self) {
        if self.config.max_chain_length <= 2 {
            return;
        }

        let existing_pairs: Vec<(KernelId, KernelId)> = self
            .candidates
            .iter()
            .filter(|c| c.fusion_type == FusionType::Vertical && c.kernels.len() == 2)
            .map(|c| (c.kernels[0], c.kernels[1]))
            .collect();

        let mut chains = Vec::new();

        for (a, b) in &existing_pairs {
            for (c, d) in &existing_pairs {
                if b == c && a != d {
                    // Found chain A→B→D
                    chains.push(vec![*a, *b, *d]);
                }
            }
        }

        for chain in chains {
            if chain.len() <= self.config.max_chain_length {
                let candidate_id = self.alloc_candidate_id();
                let mut candidate =
                    FusionCandidate::new(candidate_id, chain.clone(), FusionType::Vertical);

                // Add all edges
                for window in chain.windows(2) {
                    if let Some(buffer) = self.dep_graph.get_data_dependency(window[0], window[1]) {
                        candidate
                            .producer_consumer_edges
                            .push((window[0], window[1], buffer));
                    }
                }

                // Combine all estimates
                let estimates: Vec<_> = chain
                    .iter()
                    .filter_map(|k| self.kernel_estimates.get(k).cloned())
                    .collect();
                if !estimates.is_empty() {
                    candidate.resource_estimate = estimates
                        .iter()
                        .skip(1)
                        .fold(estimates[0].clone(), |acc, e| acc.combine(e));
                }

                self.candidates.push(candidate);
            }
        }
    }

    /// Find horizontal (parallel) fusion candidates
    fn find_horizontal_candidates(&mut self) {
        let kernels: Vec<KernelId> = self.dep_graph.kernels.iter().copied().collect();

        for i in 0..kernels.len() {
            for j in (i + 1)..kernels.len() {
                let k1 = kernels[i];
                let k2 = kernels[j];

                // Check independence and config compatibility
                if self.dep_graph.are_independent(k1, k2) && self.have_compatible_configs(k1, k2) {
                    let candidate_id = self.alloc_candidate_id();
                    let mut candidate =
                        FusionCandidate::new(candidate_id, vec![k1, k2], FusionType::Horizontal);

                    // Combine resource estimates
                    if let (Some(est1), Some(est2)) = (
                        self.kernel_estimates.get(&k1),
                        self.kernel_estimates.get(&k2),
                    ) {
                        candidate.resource_estimate = est1.combine(est2);
                    }

                    self.candidates.push(candidate);
                }
            }
        }
    }

    /// Find diamond pattern fusion candidates (A→B, A→C, B&C→D)
    fn find_diamond_candidates(&mut self) {
        let kernels: Vec<KernelId> = self.dep_graph.kernels.iter().copied().collect();

        for &root in &kernels {
            let successors: Vec<KernelId> =
                self.dep_graph.successors(root).map(|(k, _)| k).collect();

            if successors.len() >= 2 {
                // Look for common successor of the branches
                for i in 0..successors.len() {
                    for j in (i + 1)..successors.len() {
                        let b = successors[i];
                        let c = successors[j];

                        // Find common successor
                        let b_succs: FxHashSet<KernelId> =
                            self.dep_graph.successors(b).map(|(k, _)| k).collect();
                        let c_succs: FxHashSet<KernelId> =
                            self.dep_graph.successors(c).map(|(k, _)| k).collect();

                        for &d in b_succs.intersection(&c_succs) {
                            if self.have_compatible_configs_multi(&[root, b, c, d]) {
                                let candidate_id = self.alloc_candidate_id();
                                let mut candidate = FusionCandidate::new(
                                    candidate_id,
                                    vec![root, b, c, d],
                                    FusionType::Diamond,
                                );

                                // Add all edges
                                for &(from, to) in &[(root, b), (root, c), (b, d), (c, d)] {
                                    if let Some(buf) = self.dep_graph.get_data_dependency(from, to)
                                    {
                                        candidate.producer_consumer_edges.push((from, to, buf));
                                    }
                                }

                                // Combine estimates
                                let estimates: Vec<_> = [root, b, c, d]
                                    .iter()
                                    .filter_map(|k| self.kernel_estimates.get(k).cloned())
                                    .collect();
                                if !estimates.is_empty() {
                                    candidate.resource_estimate = estimates
                                        .iter()
                                        .skip(1)
                                        .fold(estimates[0].clone(), |acc, e| acc.combine(e));
                                }

                                self.candidates.push(candidate);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check if two kernels have compatible launch configurations
    fn have_compatible_configs(&self, k1: KernelId, k2: KernelId) -> bool {
        // For now, assume compatible if both have similar thread counts
        // In a full implementation, we'd check grid/block dims from kernel metadata
        if let (Some(est1), Some(est2)) = (
            self.kernel_estimates.get(&k1),
            self.kernel_estimates.get(&k2),
        ) {
            est1.max_threads_per_block == est2.max_threads_per_block
        } else {
            true // Assume compatible if no estimates
        }
    }

    /// Check if multiple kernels have compatible launch configurations
    fn have_compatible_configs_multi(&self, kernels: &[KernelId]) -> bool {
        if kernels.len() < 2 {
            return true;
        }

        let first = kernels[0];
        kernels
            .iter()
            .skip(1)
            .all(|&k| self.have_compatible_configs(first, k))
    }

    /// Allocate a new candidate ID
    fn alloc_candidate_id(&mut self) -> FusionCandidateId {
        let id = FusionCandidateId(self.next_candidate_id);
        self.next_candidate_id += 1;
        id
    }

    /// Allocate a new group ID
    fn alloc_group_id(&mut self) -> FusionGroupId {
        let id = FusionGroupId(self.next_group_id);
        self.next_group_id += 1;
        id
    }

    /// Pass 4: Evaluate candidates with cost model
    fn evaluate_candidates(&mut self) {
        for candidate in &mut self.candidates {
            // Check hardware constraints
            if let Err(reason) = self.cost_model.check_constraints(candidate) {
                candidate.invalidate(&reason);
                continue;
            }

            // Get original estimates for comparison
            let original_estimates: Vec<ResourceEstimate> = candidate
                .kernels
                .iter()
                .filter_map(|k| self.kernel_estimates.get(k).cloned())
                .collect();

            // Calculate benefit score
            candidate.benefit_score = self.cost_model.evaluate(candidate, &original_estimates);
        }
    }

    /// Pass 5: Select non-conflicting fusion groups (greedy algorithm)
    fn select_fusion_groups(&mut self) {
        // Sort candidates by benefit score (descending)
        let mut sorted_indices: Vec<usize> = (0..self.candidates.len()).collect();
        sorted_indices.sort_by(|&a, &b| {
            self.candidates[b]
                .benefit_score
                .partial_cmp(&self.candidates[a].benefit_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut used_kernels: FxHashSet<KernelId> = FxHashSet::default();

        for idx in sorted_indices {
            // Extract info from candidate first to avoid borrow conflicts
            let (is_valid, is_beneficial, kernels_clone, kernel_count) = {
                let candidate = &self.candidates[idx];
                (
                    candidate.valid,
                    self.cost_model.is_beneficial(candidate),
                    candidate.kernels.clone(),
                    candidate.kernels.len(),
                )
            };

            // Skip invalid or low-benefit candidates
            if !is_valid || !is_beneficial {
                continue;
            }

            // Check if any kernel is already used
            let conflicts = kernels_clone.iter().any(|k| used_kernels.contains(k));
            if conflicts {
                continue;
            }

            // Check max kernels per group
            if kernel_count > self.config.max_kernels_per_group {
                continue;
            }

            // Create fusion group
            let group_id = self.alloc_group_id();
            let group = FusionGroup::new(group_id, kernels_clone.clone());
            self.fusion_groups.push(group);

            // Mark kernels as used
            for kernel in kernels_clone {
                used_kernels.insert(kernel);
            }
        }
    }

    /// Pass 6: Plan transformations (value/block remapping)
    fn plan_transformations(&mut self, module: &GpuModule) {
        // Process groups by index to avoid borrow conflicts
        for i in 0..self.fusion_groups.len() {
            // Get kernels for this group
            let kernels = self.fusion_groups[i].kernels.clone();

            // Compute launch config
            let launch_config = self.compute_launch_config(&kernels);
            self.fusion_groups[i].launch_config = launch_config;

            // Plan shared memory layout
            let shared_mem_layout = self.plan_shared_mem_layout(&kernels, module);
            self.fusion_groups[i].shared_mem_layout = shared_mem_layout;

            // Plan value and block remapping
            self.plan_remapping_for_group(i, module);
        }
    }

    /// Plan value and block remapping for a fusion group by index
    fn plan_remapping_for_group(&mut self, group_idx: usize, module: &GpuModule) {
        let mut next_value_id = 0u32;
        let mut next_block_id = 0u32;

        // Get kernels for this group
        let kernels = self.fusion_groups[group_idx].kernels.clone();
        let last_kernel = kernels.last().copied();

        let mut value_map = FxHashMap::default();
        let mut block_map = FxHashMap::default();
        let mut barrier_points = Vec::new();

        for &kernel_id in &kernels {
            if let Some(name) = self.kernel_names.get(&kernel_id)
                && let Some(kernel) = module.kernels.get(name)
            {
                // Map all values
                for block in &kernel.blocks {
                    for (vid, _) in &block.instructions {
                        value_map.insert((kernel_id, *vid), ValueId(next_value_id));
                        next_value_id += 1;
                    }
                }

                // Map all blocks
                for block in &kernel.blocks {
                    block_map.insert((kernel_id, block.id), BlockId(next_block_id));
                    next_block_id += 1;
                }

                // Add barrier point between kernels (except after last)
                if Some(kernel_id) != last_kernel {
                    barrier_points.push(BlockId(next_block_id - 1));
                }
            }
        }

        // Update the group
        self.fusion_groups[group_idx].value_map = value_map;
        self.fusion_groups[group_idx].block_map = block_map;
        self.fusion_groups[group_idx].barrier_points = barrier_points;
    }

    /// Compute launch configuration for a fused kernel group
    fn compute_launch_config(&self, kernels: &[KernelId]) -> LaunchConfig {
        // Take the max of all thread requirements
        let max_threads = kernels
            .iter()
            .filter_map(|k| self.kernel_estimates.get(k))
            .map(|e| e.max_threads_per_block)
            .max()
            .unwrap_or(256);

        // Ensure it's a multiple of warp size
        let threads = max_threads.div_ceil(32) * 32;
        let threads = threads.min(self.cost_model.constraints.max_threads_per_block);

        LaunchConfig {
            grid: (1, 1, 1), // Will be determined at runtime
            block: (threads, 1, 1),
            dynamic_shared_mem: 0,
        }
    }

    /// Plan shared memory layout for fused kernels
    fn plan_shared_mem_layout(&self, kernels: &[KernelId], module: &GpuModule) -> SharedMemLayout {
        let mut layout = SharedMemLayout::new();

        for &kernel_id in kernels {
            if let Some(name) = self.kernel_names.get(&kernel_id)
                && let Some(kernel) = module.kernels.get(name)
            {
                for decl in &kernel.shared_memory {
                    let size = decl.elem_type.size_bytes() * decl.size;
                    let prefix_name = format!("{}_{}", kernel_id.0, decl.name);
                    layout.add(prefix_name, size, decl.align);
                }
            }
        }

        layout
    }

    /// Build the final fusion plan
    fn build_plan(&self) -> FusionPlan {
        let fused_kernels: FxHashSet<KernelId> = self
            .fusion_groups
            .iter()
            .flat_map(|g| g.kernels.iter().copied())
            .collect();

        let unfused_kernels: Vec<KernelId> = self
            .dep_graph
            .kernels
            .iter()
            .filter(|k| !fused_kernels.contains(k))
            .copied()
            .collect();

        let total_benefit: f64 = self
            .candidates
            .iter()
            .filter(|c| self.fusion_groups.iter().any(|g| g.kernels == c.kernels))
            .map(|c| c.benefit_score)
            .sum();

        let stats = FusionStats {
            candidates_found: self.candidates.len(),
            candidates_rejected: self.candidates.iter().filter(|c| !c.valid).count(),
            candidates_below_threshold: self
                .candidates
                .iter()
                .filter(|c| c.valid && !self.cost_model.is_beneficial(c))
                .count(),
            groups_formed: self.fusion_groups.len(),
            kernels_fused: fused_kernels.len(),
        };

        FusionPlan {
            groups: self.fusion_groups.clone(),
            unfused_kernels,
            total_benefit,
            stats,
        }
    }
}

// ============================================================================
// Fusion Transformer
// ============================================================================

/// Error type for fusion transformation
#[derive(Debug, Clone)]
pub enum FusionError {
    /// Kernel not found
    KernelNotFound(String),

    /// Block not found
    BlockNotFound(BlockId),

    /// Value not found
    ValueNotFound(ValueId),

    /// Invalid transformation
    InvalidTransformation(String),
}

impl fmt::Display for FusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusionError::KernelNotFound(name) => write!(f, "Kernel not found: {}", name),
            FusionError::BlockNotFound(id) => write!(f, "Block not found: {}", id),
            FusionError::ValueNotFound(id) => write!(f, "Value not found: {}", id),
            FusionError::InvalidTransformation(msg) => write!(f, "Invalid transformation: {}", msg),
        }
    }
}

impl std::error::Error for FusionError {}

/// Transforms GPU modules by applying fusion
pub struct FusionTransformer {
    /// Name registry for fused kernels
    kernel_names: FxHashMap<KernelId, String>,
}

impl FusionTransformer {
    /// Create a new fusion transformer
    pub fn new() -> Self {
        Self {
            kernel_names: FxHashMap::default(),
        }
    }

    /// Apply a fusion plan to a GPU module
    pub fn apply(
        &mut self,
        module: &GpuModule,
        plan: &FusionPlan,
    ) -> Result<GpuModule, FusionError> {
        let mut result = GpuModule::new(module.name.clone(), module.target);

        // Build kernel name registry
        let mut id = 0u32;
        for name in module.kernels.keys() {
            self.kernel_names.insert(KernelId(id), name.clone());
            id += 1;
        }

        // Fuse groups
        for group in &plan.groups {
            let fused_kernel = self.fuse_group(module, group)?;
            result.add_kernel(fused_kernel);
        }

        // Copy unfused kernels
        for kernel_id in &plan.unfused_kernels {
            if let Some(name) = self.kernel_names.get(kernel_id)
                && let Some(kernel) = module.kernels.get(name)
            {
                result.add_kernel(kernel.clone());
            }
        }

        // Copy device functions
        for (name, func) in &module.device_functions {
            result.device_functions.insert(name.clone(), func.clone());
        }

        // Copy constants
        for constant in &module.constants {
            result.constants.push(constant.clone());
        }

        Ok(result)
    }

    /// Fuse a group of kernels into a single kernel
    fn fuse_group(
        &mut self,
        module: &GpuModule,
        group: &FusionGroup,
    ) -> Result<GpuKernel, FusionError> {
        let mut fused = GpuKernel::new(&group.fused_name);

        // Collect and deduplicate parameters
        let mut param_set: FxHashSet<String> = FxHashSet::default();
        for &kernel_id in &group.kernels {
            if let Some(name) = self.kernel_names.get(&kernel_id)
                && let Some(kernel) = module.kernels.get(name)
            {
                for param in &kernel.params {
                    if !param_set.contains(&param.name) {
                        fused.add_param(param.clone());
                        param_set.insert(param.name.clone());
                    }
                }
            }
        }

        // Add shared memory from layout
        for (name, offset, size, align) in &group.shared_mem_layout.allocations {
            fused.add_shared_memory(SharedMemDecl {
                name: name.clone(),
                elem_type: GpuType::U8, // Use bytes
                size: *size,
                align: *align,
            });
        }

        // Clone and remap blocks from each kernel
        let mut all_blocks = Vec::new();
        for (i, &kernel_id) in group.kernels.iter().enumerate() {
            if let Some(name) = self.kernel_names.get(&kernel_id)
                && let Some(kernel) = module.kernels.get(name)
            {
                let blocks = self.clone_kernel_blocks(kernel, kernel_id, group)?;
                all_blocks.extend(blocks);

                // Insert barrier between kernels (except after last)
                if i < group.kernels.len() - 1
                    && let Some(&barrier_block) = group.barrier_points.get(i)
                {
                    // Add sync instruction to the last block of this kernel
                    if let Some(last_block) = all_blocks.last_mut() {
                        // Insert sync before terminator
                        let sync_id = ValueId(last_block.instructions.len() as u32 + 10000);
                        last_block.instructions.push((sync_id, GpuOp::SyncThreads));
                    }
                }
            }
        }

        // Connect control flow between kernel sections
        self.connect_control_flow(&mut all_blocks, group)?;

        // Add all blocks to fused kernel
        for block in all_blocks {
            fused.add_block(block);
        }

        // Set entry point
        fused.entry = BlockId(0);

        // Set max threads
        fused.max_threads = Some(group.launch_config.block.0);

        Ok(fused)
    }

    /// Clone blocks from a kernel with value/block remapping
    fn clone_kernel_blocks(
        &self,
        kernel: &GpuKernel,
        kernel_id: KernelId,
        group: &FusionGroup,
    ) -> Result<Vec<GpuBlock>, FusionError> {
        let mut blocks = Vec::new();

        for block in &kernel.blocks {
            let new_block_id = group
                .block_map
                .get(&(kernel_id, block.id))
                .copied()
                .unwrap_or(block.id);

            let mut new_block =
                GpuBlock::new(new_block_id, format!("{}_{}", kernel_id.0, block.label));

            // Clone and remap instructions
            for (value_id, op) in &block.instructions {
                let new_value_id = group
                    .value_map
                    .get(&(kernel_id, *value_id))
                    .copied()
                    .unwrap_or(*value_id);

                let new_op = self.remap_op(op, kernel_id, group);
                new_block.add_instruction(new_value_id, new_op);
            }

            // Remap terminator
            let new_terminator = self.remap_terminator(&block.terminator, kernel_id, group);
            new_block.set_terminator(new_terminator);

            blocks.push(new_block);
        }

        Ok(blocks)
    }

    /// Remap an operation's value references
    fn remap_op(&self, op: &GpuOp, kernel_id: KernelId, group: &FusionGroup) -> GpuOp {
        let remap = |v: ValueId| group.value_map.get(&(kernel_id, v)).copied().unwrap_or(v);

        match op {
            // Binary ops
            GpuOp::Add(a, b) => GpuOp::Add(remap(*a), remap(*b)),
            GpuOp::Sub(a, b) => GpuOp::Sub(remap(*a), remap(*b)),
            GpuOp::Mul(a, b) => GpuOp::Mul(remap(*a), remap(*b)),
            GpuOp::Div(a, b) => GpuOp::Div(remap(*a), remap(*b)),
            GpuOp::Rem(a, b) => GpuOp::Rem(remap(*a), remap(*b)),
            GpuOp::FAdd(a, b) => GpuOp::FAdd(remap(*a), remap(*b)),
            GpuOp::FSub(a, b) => GpuOp::FSub(remap(*a), remap(*b)),
            GpuOp::FMul(a, b) => GpuOp::FMul(remap(*a), remap(*b)),
            GpuOp::FDiv(a, b) => GpuOp::FDiv(remap(*a), remap(*b)),

            // Unary ops
            GpuOp::Neg(a) => GpuOp::Neg(remap(*a)),
            GpuOp::FNeg(a) => GpuOp::FNeg(remap(*a)),
            GpuOp::Not(a) => GpuOp::Not(remap(*a)),
            GpuOp::BitNot(a) => GpuOp::BitNot(remap(*a)),

            // Comparisons
            GpuOp::Eq(a, b) => GpuOp::Eq(remap(*a), remap(*b)),
            GpuOp::Ne(a, b) => GpuOp::Ne(remap(*a), remap(*b)),
            GpuOp::Lt(a, b) => GpuOp::Lt(remap(*a), remap(*b)),
            GpuOp::Le(a, b) => GpuOp::Le(remap(*a), remap(*b)),
            GpuOp::Gt(a, b) => GpuOp::Gt(remap(*a), remap(*b)),
            GpuOp::Ge(a, b) => GpuOp::Ge(remap(*a), remap(*b)),
            GpuOp::FEq(a, b) => GpuOp::FEq(remap(*a), remap(*b)),
            GpuOp::FNe(a, b) => GpuOp::FNe(remap(*a), remap(*b)),
            GpuOp::FLt(a, b) => GpuOp::FLt(remap(*a), remap(*b)),
            GpuOp::FLe(a, b) => GpuOp::FLe(remap(*a), remap(*b)),
            GpuOp::FGt(a, b) => GpuOp::FGt(remap(*a), remap(*b)),
            GpuOp::FGe(a, b) => GpuOp::FGe(remap(*a), remap(*b)),

            // Memory ops
            GpuOp::Load(ptr, space) => GpuOp::Load(remap(*ptr), *space),
            GpuOp::Store(ptr, val, space) => GpuOp::Store(remap(*ptr), remap(*val), *space),

            // Conversions
            GpuOp::Trunc(v, ty) => GpuOp::Trunc(remap(*v), ty.clone()),
            GpuOp::ZExt(v, ty) => GpuOp::ZExt(remap(*v), ty.clone()),
            GpuOp::SExt(v, ty) => GpuOp::SExt(remap(*v), ty.clone()),
            GpuOp::FpTrunc(v, ty) => GpuOp::FpTrunc(remap(*v), ty.clone()),
            GpuOp::FpExt(v, ty) => GpuOp::FpExt(remap(*v), ty.clone()),
            GpuOp::Bitcast(v, ty) => GpuOp::Bitcast(remap(*v), ty.clone()),

            // Select
            GpuOp::Select(cond, t, f) => GpuOp::Select(remap(*cond), remap(*t), remap(*f)),

            // Phi - remap both values and blocks
            GpuOp::Phi(entries) => {
                let new_entries: Vec<_> = entries
                    .iter()
                    .map(|(block, val)| {
                        let new_block = group
                            .block_map
                            .get(&(kernel_id, *block))
                            .copied()
                            .unwrap_or(*block);
                        (new_block, remap(*val))
                    })
                    .collect();
                GpuOp::Phi(new_entries)
            }

            // Call
            GpuOp::Call(name, args) => {
                let new_args: Vec<_> = args.iter().map(|a| remap(*a)).collect();
                GpuOp::Call(name.clone(), new_args)
            }

            // Atomics
            GpuOp::AtomicAdd(ptr, val) => GpuOp::AtomicAdd(remap(*ptr), remap(*val)),
            GpuOp::AtomicSub(ptr, val) => GpuOp::AtomicSub(remap(*ptr), remap(*val)),
            GpuOp::AtomicMin(ptr, val) => GpuOp::AtomicMin(remap(*ptr), remap(*val)),
            GpuOp::AtomicMax(ptr, val) => GpuOp::AtomicMax(remap(*ptr), remap(*val)),

            // GEP
            GpuOp::GetElementPtr(base, indices) => {
                let new_indices: Vec<_> = indices.iter().map(|i| remap(*i)).collect();
                GpuOp::GetElementPtr(remap(*base), new_indices)
            }

            // Pass through unchanged ops
            _ => op.clone(),
        }
    }

    /// Remap terminator block references
    fn remap_terminator(
        &self,
        term: &GpuTerminator,
        kernel_id: KernelId,
        group: &FusionGroup,
    ) -> GpuTerminator {
        let remap_block = |b: BlockId| group.block_map.get(&(kernel_id, b)).copied().unwrap_or(b);

        let remap_value = |v: ValueId| group.value_map.get(&(kernel_id, v)).copied().unwrap_or(v);

        match term {
            GpuTerminator::Br(target) => GpuTerminator::Br(remap_block(*target)),
            GpuTerminator::CondBr(cond, then_block, else_block) => GpuTerminator::CondBr(
                remap_value(*cond),
                remap_block(*then_block),
                remap_block(*else_block),
            ),
            GpuTerminator::Return(val) => GpuTerminator::Return(remap_value(*val)),
            GpuTerminator::ReturnVoid => GpuTerminator::ReturnVoid,
            GpuTerminator::Unreachable => GpuTerminator::Unreachable,
        }
    }

    /// Connect control flow between fused kernel sections
    fn connect_control_flow(
        &self,
        blocks: &mut [GpuBlock],
        group: &FusionGroup,
    ) -> Result<(), FusionError> {
        // Find blocks that end with ReturnVoid and aren't the last kernel's exit
        // Replace them with branches to the next kernel's entry

        let mut kernel_entries: Vec<BlockId> = Vec::new();
        let mut kernel_exits: Vec<usize> = Vec::new(); // indices into blocks

        let mut current_kernel_idx = 0;
        for (i, block) in blocks.iter().enumerate() {
            // Detect kernel boundary by checking block label prefix
            let expected_prefix = format!("{}_", group.kernels[current_kernel_idx].0);
            if !block.label.starts_with(&expected_prefix) {
                // New kernel started
                kernel_entries.push(block.id);
                if i > 0 {
                    kernel_exits.push(i - 1);
                }
                current_kernel_idx = (current_kernel_idx + 1).min(group.kernels.len() - 1);
            }

            if i == 0 {
                kernel_entries.push(block.id);
            }
        }
        kernel_exits.push(blocks.len() - 1);

        // Replace ReturnVoid with branches to next kernel's entry
        for (exit_idx, entry_block) in kernel_exits.iter().zip(kernel_entries.iter().skip(1)) {
            if let Some(block) = blocks.get_mut(*exit_idx)
                && matches!(block.terminator, GpuTerminator::ReturnVoid)
            {
                block.terminator = GpuTerminator::Br(*entry_block);
            }
        }

        Ok(())
    }
}

impl Default for FusionTransformer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Analyze and fuse kernels in a GPU module
pub fn analyze_and_fuse_kernels(
    module: &GpuModule,
    graph: &GpuGraph,
    config: Option<FusionConfig>,
) -> Result<GpuModule, FusionError> {
    let config = config.unwrap_or_default();
    let cost_model = FusionCostModel::new(module.target);

    let mut analysis = FusionAnalysis::with_config(config, cost_model);
    let plan = analysis.analyze(module, graph);

    let mut transformer = FusionTransformer::new();
    transformer.apply(module, &plan)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kernel_id_display() {
        let id = KernelId(42);
        assert_eq!(format!("{}", id), "K42");
    }

    #[test]
    fn test_fusion_type_display() {
        assert_eq!(format!("{}", FusionType::Vertical), "vertical");
        assert_eq!(format!("{}", FusionType::Horizontal), "horizontal");
        assert_eq!(format!("{}", FusionType::Diamond), "diamond");
        assert_eq!(
            format!("{}", FusionType::LoopFusion { unroll_factor: 4 }),
            "loop(unroll=4)"
        );
    }

    #[test]
    fn test_arch_constraints_from_cuda_cc() {
        // Test Turing (sm_75)
        let turing = ArchConstraints::from_cuda_cc((7, 5));
        assert_eq!(turing.max_shared_memory, 64 * 1024);
        assert_eq!(turing.max_registers, 255);
        assert_eq!(turing.warp_size, 32);

        // Test Ampere (sm_80)
        let ampere = ArchConstraints::from_cuda_cc((8, 0));
        assert_eq!(ampere.max_shared_memory, 164 * 1024);

        // Test Hopper (sm_90)
        let hopper = ArchConstraints::from_cuda_cc((9, 0));
        assert_eq!(hopper.max_shared_memory, 228 * 1024);

        // Test Blackwell (sm_100)
        let blackwell = ArchConstraints::from_cuda_cc((10, 0));
        assert_eq!(blackwell.max_shared_memory, 256 * 1024);
    }

    #[test]
    fn test_arch_constraints_fits() {
        let constraints = ArchConstraints::from_cuda_cc((7, 5)); // Turing

        // Should fit
        let small = ResourceEstimate {
            registers_per_thread: 64,
            shared_memory_bytes: 32 * 1024,
            max_threads_per_block: 512,
            ..Default::default()
        };
        assert!(constraints.fits(&small));

        // Should not fit (too much shared memory)
        let large = ResourceEstimate {
            registers_per_thread: 64,
            shared_memory_bytes: 128 * 1024, // Exceeds Turing's 64KB
            max_threads_per_block: 512,
            ..Default::default()
        };
        assert!(!constraints.fits(&large));
    }

    #[test]
    fn test_resource_estimate_combine() {
        let est1 = ResourceEstimate {
            registers_per_thread: 32,
            shared_memory_bytes: 1024,
            max_threads_per_block: 256,
            occupancy: 0.75,
            barrier_count: 1,
            instruction_count: 100,
            global_loads: 10,
            global_stores: 5,
        };

        let est2 = ResourceEstimate {
            registers_per_thread: 48,
            shared_memory_bytes: 2048,
            max_threads_per_block: 256,
            occupancy: 0.5,
            barrier_count: 2,
            instruction_count: 150,
            global_loads: 20,
            global_stores: 10,
        };

        let combined = est1.combine(&est2);
        assert_eq!(combined.registers_per_thread, 80);
        assert_eq!(combined.shared_memory_bytes, 3072);
        assert_eq!(combined.max_threads_per_block, 256);
        assert_eq!(combined.occupancy, 0.5);
        assert_eq!(combined.barrier_count, 4); // 1 + 2 + 1 inter-kernel
        assert_eq!(combined.instruction_count, 250);
        assert_eq!(combined.global_loads, 30);
        assert_eq!(combined.global_stores, 15);
    }

    #[test]
    fn test_cost_model_check_constraints() {
        let cost_model = FusionCostModel::from_cuda_cc((7, 5)); // Turing

        // Valid candidate
        let mut valid = FusionCandidate::new(
            FusionCandidateId(0),
            vec![KernelId(0), KernelId(1)],
            FusionType::Vertical,
        );
        valid.resource_estimate.registers_per_thread = 64;
        valid.resource_estimate.shared_memory_bytes = 32 * 1024;
        assert!(cost_model.check_constraints(&valid).is_ok());

        // Invalid: too many registers
        let mut invalid_regs = valid.clone();
        invalid_regs.resource_estimate.registers_per_thread = 300;
        assert!(cost_model.check_constraints(&invalid_regs).is_err());

        // Invalid: too much shared memory
        let mut invalid_smem = valid.clone();
        invalid_smem.resource_estimate.shared_memory_bytes = 128 * 1024;
        assert!(cost_model.check_constraints(&invalid_smem).is_err());
    }

    #[test]
    fn test_cost_model_evaluate() {
        let cost_model = FusionCostModel::from_cuda_cc((8, 0));

        // Use low register counts to ensure fusion benefits outweigh penalties
        let original_estimates = vec![
            ResourceEstimate {
                registers_per_thread: 8,
                shared_memory_bytes: 1024,
                occupancy: 0.75,
                barrier_count: 0,
                ..Default::default()
            },
            ResourceEstimate {
                registers_per_thread: 8,
                shared_memory_bytes: 1024,
                occupancy: 0.75,
                barrier_count: 0,
                ..Default::default()
            },
        ];

        // Vertical fusion should have positive score due to memory savings
        // Score breakdown:
        // - Launch savings: 1.0 * 1.0 = 1.0
        // - Memory savings: 1 edge * 2.0 = 2.0
        // - Occupancy delta: (0.75 - 0.75) * 0.5 = 0.0 (after combine)
        // - Sync cost: 1 barrier * -0.3 = -0.3
        // - Register cost: (16 - 8) * -0.2 = -1.6
        // Total: 1.0 + 2.0 + 0.0 - 0.3 - 1.6 = 1.1 > 0
        let mut vertical = FusionCandidate::new(
            FusionCandidateId(0),
            vec![KernelId(0), KernelId(1)],
            FusionType::Vertical,
        );
        vertical
            .producer_consumer_edges
            .push((KernelId(0), KernelId(1), BufferId(0)));
        vertical.resource_estimate = original_estimates[0].combine(&original_estimates[1]);

        let score = cost_model.evaluate(&vertical, &original_estimates);
        assert!(
            score > 0.0,
            "Vertical fusion should have positive score, got {}",
            score
        );

        // Horizontal fusion should have lower score (no memory savings)
        // Score = 1.0 (launch) - 0.3 (sync) - 1.6 (regs) = -0.9 (negative)
        let mut horizontal = FusionCandidate::new(
            FusionCandidateId(1),
            vec![KernelId(0), KernelId(1)],
            FusionType::Horizontal,
        );
        horizontal.resource_estimate = original_estimates[0].combine(&original_estimates[1]);

        let h_score = cost_model.evaluate(&horizontal, &original_estimates);
        assert!(
            h_score < score,
            "Horizontal fusion should have lower score than vertical"
        );
    }

    #[test]
    fn test_dependency_graph() {
        let mut graph = KernelDependencyGraph::new();

        graph.add_kernel(KernelId(0));
        graph.add_kernel(KernelId(1));
        graph.add_kernel(KernelId(2));

        // 0 → 1 → 2
        graph.add_edge(
            KernelId(0),
            KernelId(1),
            DependencyType::DataFlow(BufferId(0)),
        );
        graph.add_edge(
            KernelId(1),
            KernelId(2),
            DependencyType::DataFlow(BufferId(1)),
        );

        // Check paths
        assert!(graph.has_path(KernelId(0), KernelId(1)));
        assert!(graph.has_path(KernelId(0), KernelId(2)));
        assert!(!graph.has_path(KernelId(1), KernelId(0)));

        // Check independence
        assert!(!graph.are_independent(KernelId(0), KernelId(1)));
        assert!(!graph.are_independent(KernelId(0), KernelId(2)));
    }

    #[test]
    fn test_shared_mem_layout() {
        let mut layout = SharedMemLayout::new();

        let offset1 = layout.add("buf_a".to_string(), 256, 16);
        assert_eq!(offset1, 0);
        assert_eq!(layout.total_size, 256);

        let offset2 = layout.add("buf_b".to_string(), 128, 32);
        assert_eq!(offset2, 256); // Aligned to 32
        assert_eq!(layout.total_size, 384);

        let offset3 = layout.add("buf_c".to_string(), 64, 4);
        assert_eq!(offset3, 384);
        assert_eq!(layout.total_size, 448);
    }

    #[test]
    fn test_align_to() {
        assert_eq!(align_to(0, 16), 0);
        assert_eq!(align_to(1, 16), 16);
        assert_eq!(align_to(15, 16), 16);
        assert_eq!(align_to(16, 16), 16);
        assert_eq!(align_to(17, 16), 32);
        assert_eq!(align_to(100, 32), 128);
    }

    #[test]
    fn test_fusion_group_creation() {
        let group = FusionGroup::new(
            FusionGroupId(0),
            vec![KernelId(0), KernelId(1), KernelId(2)],
        );

        assert_eq!(group.id, FusionGroupId(0));
        assert_eq!(group.kernels.len(), 3);
        assert_eq!(group.fused_name, "fused_0_1_2");
    }

    #[test]
    fn test_fusion_candidate_invalidate() {
        let mut candidate = FusionCandidate::new(
            FusionCandidateId(0),
            vec![KernelId(0)],
            FusionType::Vertical,
        );
        assert!(candidate.valid);
        assert!(candidate.invalid_reason.is_none());

        candidate.invalidate("exceeds shared memory limit");
        assert!(!candidate.valid);
        assert_eq!(
            candidate.invalid_reason,
            Some("exceeds shared memory limit".to_string())
        );
    }

    #[test]
    fn test_fusion_config_default() {
        let config = FusionConfig::default();
        assert!(config.enable_vertical);
        assert!(config.enable_horizontal);
        assert!(config.enable_diamond);
        assert_eq!(config.max_kernels_per_group, 4);
        assert_eq!(config.max_chain_length, 3);
    }
}
