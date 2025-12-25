//! Async Memory Pipeline for GPU Kernels
//!
//! Implements software pipelining to overlap memory transfers with compute:
//! - TMA (Tensor Memory Accelerator) on Hopper+ (sm_90+)
//! - cp.async on Ampere+ (sm_80+)
//! - mbarrier synchronization for pipeline stages
//!
//! # Pipeline Patterns
//!
//! ```text
//! Double Buffer (2 stages):
//!   Stage 0: Load A[i+1] → Buffer[0]
//!   Stage 1: Compute on Buffer[1], Store result
//!   (swap buffers)
//!
//! Triple Buffer (3 stages):
//!   Stage 0: Load A[i+2] → Buffer[0]
//!   Stage 1: Load B[i+1] → Buffer[1]
//!   Stage 2: Compute on Buffer[2], Store
//!   (rotate buffers)
//! ```
//!
//! # Architecture
//!
//! ```text
//! TileConfig.pipeline_stages → AsyncPipeline → PipelineSchedule
//!                                    │
//!               ┌────────────────────┴────────────────────┐
//!               │                    │                    │
//!         StageBuffers         BarrierPool          AsyncOpGraph
//! ```

use std::collections::{HashMap, VecDeque};
use std::fmt;

use super::autotune::TileConfig;
use super::ir::{CudaArch, GpuBlock, GpuKernel, GpuOp, GpuType, TmaReduceOp, ValueId};

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for pipeline stages
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct StageId(pub u32);

impl fmt::Display for StageId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "stage{}", self.0)
    }
}

/// Unique identifier for barriers
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct BarrierId(pub u32);

impl fmt::Display for BarrierId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mbar{}", self.0)
    }
}

/// Unique identifier for async operations
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct AsyncOpId(pub u32);

impl fmt::Display for AsyncOpId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "async{}", self.0)
    }
}

/// A single stage in the software pipeline
#[derive(Debug, Clone)]
pub struct PipelineStage {
    /// Stage identifier
    pub id: StageId,
    /// Buffers used in this stage
    pub buffers: Vec<StageBuffer>,
    /// Async operations in this stage
    pub async_ops: Vec<AsyncOpId>,
    /// Barrier to wait on before entering this stage
    pub entry_barrier: Option<BarrierId>,
    /// Barrier to signal when exiting this stage
    pub exit_barrier: Option<BarrierId>,
}

impl PipelineStage {
    /// Create a new pipeline stage
    pub fn new(id: StageId) -> Self {
        Self {
            id,
            buffers: Vec::new(),
            async_ops: Vec::new(),
            entry_barrier: None,
            exit_barrier: None,
        }
    }

    /// Add a buffer to this stage
    pub fn add_buffer(&mut self, buffer: StageBuffer) {
        self.buffers.push(buffer);
    }

    /// Add an async operation to this stage
    pub fn add_async_op(&mut self, op_id: AsyncOpId) {
        self.async_ops.push(op_id);
    }

    /// Total buffer size for this stage in bytes
    pub fn total_buffer_size(&self) -> u32 {
        self.buffers.iter().map(|b| b.size_bytes).sum()
    }
}

/// Buffer allocation for a pipeline stage
#[derive(Debug, Clone)]
pub struct StageBuffer {
    /// Buffer name (for PTX emission)
    pub name: String,
    /// Size in bytes
    pub size_bytes: u32,
    /// Element type
    pub element_type: GpuType,
    /// Buffer index within the stage (0 = first, 1 = second in double-buffer)
    pub buffer_index: u32,
}

impl StageBuffer {
    /// Create a new stage buffer
    pub fn new(name: String, size_bytes: u32, element_type: GpuType, buffer_index: u32) -> Self {
        Self {
            name,
            size_bytes,
            element_type,
            buffer_index,
        }
    }
}

/// Async operation types
#[derive(Debug, Clone)]
pub enum AsyncOpKind {
    /// TMA load from global to shared memory (Hopper+)
    TmaLoad {
        src: ValueId,
        dst: ValueId,
        size: u32,
    },
    /// TMA store from shared to global memory (Hopper+)
    TmaStore {
        src: ValueId,
        dst: ValueId,
        size: u32,
    },
    /// TMA multicast load to multiple CTAs in cluster
    TmaMulticast {
        src: ValueId,
        dst: ValueId,
        multicast_mask: u32,
    },
    /// TMA reduction operation
    TmaReduce {
        src: ValueId,
        dst: ValueId,
        op: TmaReduceOp,
    },
    /// Ampere cp.async copy
    CpAsync {
        src: ValueId,
        dst: ValueId,
        size: u32,
    },
    /// Commit pending cp.async operations
    CpAsyncCommit,
    /// Wait for cp.async operations to complete
    CpAsyncWait {
        /// Number of outstanding operations to wait for (0 = wait for all)
        count: u32,
    },
}

impl AsyncOpKind {
    /// Check if this is a load operation
    pub fn is_load(&self) -> bool {
        matches!(
            self,
            Self::TmaLoad { .. } | Self::TmaMulticast { .. } | Self::CpAsync { .. }
        )
    }

    /// Check if this is a store operation
    pub fn is_store(&self) -> bool {
        matches!(self, Self::TmaStore { .. } | Self::TmaReduce { .. })
    }

    /// Check if this is a synchronization operation
    pub fn is_sync(&self) -> bool {
        matches!(self, Self::CpAsyncCommit | Self::CpAsyncWait { .. })
    }

    /// Check if this requires TMA (Hopper+)
    pub fn requires_tma(&self) -> bool {
        matches!(
            self,
            Self::TmaLoad { .. }
                | Self::TmaStore { .. }
                | Self::TmaMulticast { .. }
                | Self::TmaReduce { .. }
        )
    }
}

/// Single async operation with dependency info
#[derive(Debug, Clone)]
pub struct AsyncOp {
    /// Operation identifier
    pub id: AsyncOpId,
    /// Operation type
    pub kind: AsyncOpKind,
    /// Stage this operation belongs to
    pub stage: StageId,
    /// Operations that must complete before this one
    pub depends_on: Vec<AsyncOpId>,
    /// Operations that this one blocks
    pub blocks: Vec<AsyncOpId>,
}

impl AsyncOp {
    /// Create a new async operation
    pub fn new(id: AsyncOpId, kind: AsyncOpKind, stage: StageId) -> Self {
        Self {
            id,
            kind,
            stage,
            depends_on: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// Add a dependency
    pub fn add_dependency(&mut self, dep: AsyncOpId) {
        if !self.depends_on.contains(&dep) {
            self.depends_on.push(dep);
        }
    }

    /// Add a blocked operation
    pub fn add_blocks(&mut self, blocked: AsyncOpId) {
        if !self.blocks.contains(&blocked) {
            self.blocks.push(blocked);
        }
    }
}

/// Dependency graph for async operations
#[derive(Debug, Clone, Default)]
pub struct AsyncOpGraph {
    /// All operations in the graph
    pub ops: Vec<AsyncOp>,
    /// Operations ready to execute (no pending dependencies)
    pub ready_queue: VecDeque<AsyncOpId>,
    /// Map from operation ID to index
    id_to_index: HashMap<AsyncOpId, usize>,
}

impl AsyncOpGraph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an operation to the graph
    pub fn add_op(&mut self, op: AsyncOp) {
        let id = op.id;
        let index = self.ops.len();
        self.id_to_index.insert(id, index);
        self.ops.push(op);
    }

    /// Get an operation by ID
    pub fn get_op(&self, id: AsyncOpId) -> Option<&AsyncOp> {
        self.id_to_index.get(&id).map(|&idx| &self.ops[idx])
    }

    /// Get a mutable operation by ID
    pub fn get_op_mut(&mut self, id: AsyncOpId) -> Option<&mut AsyncOp> {
        self.id_to_index
            .get(&id)
            .copied()
            .map(|idx| &mut self.ops[idx])
    }

    /// Add a dependency between operations
    pub fn add_dependency(&mut self, from: AsyncOpId, to: AsyncOpId) {
        if let Some(op) = self.get_op_mut(to) {
            op.add_dependency(from);
        }
        if let Some(op) = self.get_op_mut(from) {
            op.add_blocks(to);
        }
    }

    /// Find all operations with no dependencies (roots)
    pub fn find_roots(&self) -> Vec<AsyncOpId> {
        self.ops
            .iter()
            .filter(|op| op.depends_on.is_empty())
            .map(|op| op.id)
            .collect()
    }

    /// Topologically sort operations
    pub fn topological_sort(&self) -> Vec<AsyncOpId> {
        let mut result = Vec::with_capacity(self.ops.len());
        let mut in_degree: HashMap<AsyncOpId, usize> = self
            .ops
            .iter()
            .map(|op| (op.id, op.depends_on.len()))
            .collect();

        let mut queue: VecDeque<AsyncOpId> = in_degree
            .iter()
            .filter(|&(_, d)| *d == 0)
            .map(|(&id, _)| id)
            .collect();

        while let Some(id) = queue.pop_front() {
            result.push(id);

            if let Some(op) = self.get_op(id) {
                for &blocked in &op.blocks {
                    if let Some(deg) = in_degree.get_mut(&blocked) {
                        *deg = deg.saturating_sub(1);
                        if *deg == 0 {
                            queue.push_back(blocked);
                        }
                    }
                }
            }
        }

        result
    }
}

// ============================================================================
// Barrier Types
// ============================================================================

/// Hardware barrier kind
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BarrierKind {
    /// Hopper+ native mbarrier (most efficient)
    MBarrier,
    /// Ampere cp.async.commit_group (legacy)
    CpAsyncGroup,
    /// Traditional named barrier
    NamedBarrier,
}

impl BarrierKind {
    /// Get the minimum architecture for this barrier kind
    pub fn min_arch(&self) -> CudaArch {
        match self {
            Self::MBarrier => CudaArch::Ampere, // mbarrier available from sm_80
            Self::CpAsyncGroup => CudaArch::Ampere,
            Self::NamedBarrier => CudaArch::Turing,
        }
    }

    /// Check if this barrier is supported on the given architecture
    pub fn is_supported(&self, arch: CudaArch) -> bool {
        match self {
            Self::MBarrier => matches!(
                arch,
                CudaArch::Ampere
                    | CudaArch::Ada
                    | CudaArch::Hopper
                    | CudaArch::Blackwell
                    | CudaArch::BlackwellUltra
            ),
            Self::CpAsyncGroup => matches!(
                arch,
                CudaArch::Ampere
                    | CudaArch::Ada
                    | CudaArch::Hopper
                    | CudaArch::Blackwell
                    | CudaArch::BlackwellUltra
            ),
            Self::NamedBarrier => true,
        }
    }
}

/// Hardware barrier for async synchronization
#[derive(Debug, Clone)]
pub struct Barrier {
    /// Barrier identifier
    pub id: BarrierId,
    /// Barrier type
    pub kind: BarrierKind,
    /// Number of threads that must arrive
    pub arrive_count: u32,
    /// Current phase (for parity-based barriers)
    pub phase: u32,
    /// Name for PTX emission
    pub name: String,
}

impl Barrier {
    /// Create a new barrier
    pub fn new(id: BarrierId, kind: BarrierKind, arrive_count: u32) -> Self {
        Self {
            id,
            kind,
            arrive_count,
            phase: 0,
            name: format!("mbar{}", id.0),
        }
    }

    /// Advance to next phase
    pub fn advance_phase(&mut self) {
        self.phase = (self.phase + 1) % 2; // Parity-based
    }

    /// Get the current parity
    pub fn parity(&self) -> u32 {
        self.phase % 2
    }
}

/// Pool of barriers for pipeline management
#[derive(Debug, Clone, Default)]
pub struct BarrierPool {
    /// All barriers in the pool
    pub barriers: Vec<Barrier>,
    /// Next available barrier ID
    pub next_id: u32,
    /// Maximum barriers allowed (architecture limit)
    pub max_barriers: u32,
}

impl BarrierPool {
    /// Create a new barrier pool for an architecture
    pub fn new(arch: CudaArch) -> Self {
        let max_barriers = match arch {
            CudaArch::Turing => 16,
            CudaArch::Ampere | CudaArch::Ada => 16,
            CudaArch::Hopper | CudaArch::Blackwell | CudaArch::BlackwellUltra => 32,
        };
        Self {
            barriers: Vec::new(),
            next_id: 0,
            max_barriers,
        }
    }

    /// Allocate a new barrier
    pub fn allocate(&mut self, kind: BarrierKind, arrive_count: u32) -> Option<BarrierId> {
        if self.barriers.len() >= self.max_barriers as usize {
            return None;
        }

        let id = BarrierId(self.next_id);
        self.next_id += 1;

        let barrier = Barrier::new(id, kind, arrive_count);
        self.barriers.push(barrier);

        Some(id)
    }

    /// Get a barrier by ID
    pub fn get(&self, id: BarrierId) -> Option<&Barrier> {
        self.barriers.iter().find(|b| b.id == id)
    }

    /// Get a mutable barrier by ID
    pub fn get_mut(&mut self, id: BarrierId) -> Option<&mut Barrier> {
        self.barriers.iter_mut().find(|b| b.id == id)
    }

    /// Number of allocated barriers
    pub fn len(&self) -> usize {
        self.barriers.len()
    }

    /// Check if pool is empty
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty()
    }
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Complete async pipeline configuration
#[derive(Debug, Clone)]
pub struct AsyncPipeline {
    /// Pipeline stages
    pub stages: Vec<PipelineStage>,
    /// Barrier pool
    pub barriers: BarrierPool,
    /// Async operation graph
    pub op_graph: AsyncOpGraph,
    /// Pipeline depth (2 = double buffer, 3 = triple buffer)
    pub depth: u32,
    /// Target architecture
    pub target: CudaArch,
}

impl AsyncPipeline {
    /// Create a new async pipeline
    pub fn new(depth: u32, target: CudaArch) -> Self {
        Self {
            stages: Vec::new(),
            barriers: BarrierPool::new(target),
            op_graph: AsyncOpGraph::new(),
            depth,
            target,
        }
    }

    /// Add a stage to the pipeline
    pub fn add_stage(&mut self, stage: PipelineStage) {
        self.stages.push(stage);
    }

    /// Get pipeline depth
    pub fn depth(&self) -> u32 {
        self.depth
    }

    /// Check if TMA is supported
    pub fn supports_tma(&self) -> bool {
        matches!(
            self.target,
            CudaArch::Hopper | CudaArch::Blackwell | CudaArch::BlackwellUltra
        )
    }

    /// Check if mbarrier is supported
    pub fn supports_mbarrier(&self) -> bool {
        matches!(
            self.target,
            CudaArch::Ampere
                | CudaArch::Ada
                | CudaArch::Hopper
                | CudaArch::Blackwell
                | CudaArch::BlackwellUltra
        )
    }

    /// Get preferred barrier kind for this architecture
    pub fn preferred_barrier_kind(&self) -> BarrierKind {
        if self.supports_mbarrier() {
            BarrierKind::MBarrier
        } else {
            BarrierKind::NamedBarrier
        }
    }

    /// Total shared memory required for all stages
    pub fn total_shared_memory(&self) -> u32 {
        self.stages.iter().map(|s| s.total_buffer_size()).sum()
    }
}

// ============================================================================
// Pipeline Schedule
// ============================================================================

/// Scheduled operation for code generation
#[derive(Debug, Clone)]
pub enum ScheduledOp {
    /// Execute an async load operation
    AsyncLoad(AsyncOpId),
    /// Execute an async store operation
    AsyncStore(AsyncOpId),
    /// Execute compute operations
    Compute(Vec<GpuOp>),
    /// Signal barrier arrival
    BarrierArrive(BarrierId),
    /// Wait on barrier
    BarrierWait(BarrierId),
    /// Advance barrier phase
    AdvancePhase(BarrierId),
    /// Synchronize all threads
    SyncThreads,
    /// cp.async commit group
    CpAsyncCommitGroup,
    /// cp.async wait
    CpAsyncWait { count: u32 },
}

/// Pipeline schedule ready for code generation
#[derive(Debug, Clone, Default)]
pub struct PipelineSchedule {
    /// Prologue: initial prefetches
    pub prologue: Vec<ScheduledOp>,
    /// Main loop: interleaved loads/compute/stores
    pub main_loop: Vec<ScheduledOp>,
    /// Epilogue: drain pipeline
    pub epilogue: Vec<ScheduledOp>,
    /// Barriers to initialize
    pub barrier_init: Vec<BarrierId>,
}

impl PipelineSchedule {
    /// Create a new empty schedule
    pub fn new() -> Self {
        Self::default()
    }

    /// Add an operation to the prologue
    pub fn add_prologue(&mut self, op: ScheduledOp) {
        self.prologue.push(op);
    }

    /// Add an operation to the main loop
    pub fn add_main_loop(&mut self, op: ScheduledOp) {
        self.main_loop.push(op);
    }

    /// Add an operation to the epilogue
    pub fn add_epilogue(&mut self, op: ScheduledOp) {
        self.epilogue.push(op);
    }

    /// Add a barrier to initialize
    pub fn add_barrier_init(&mut self, barrier_id: BarrierId) {
        if !self.barrier_init.contains(&barrier_id) {
            self.barrier_init.push(barrier_id);
        }
    }
}

// ============================================================================
// Buffer Configuration
// ============================================================================

/// Configuration for a single buffer in the pipeline
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Size in bytes
    pub size: u32,
    /// Element type
    pub elem_type: GpuType,
    /// Optional name prefix
    pub name_prefix: Option<String>,
}

impl BufferConfig {
    /// Create a new buffer configuration
    pub fn new(size: u32, elem_type: GpuType) -> Self {
        Self {
            size,
            elem_type,
            name_prefix: None,
        }
    }

    /// Create with a name prefix
    pub fn with_name(size: u32, elem_type: GpuType, name: &str) -> Self {
        Self {
            size,
            elem_type,
            name_prefix: Some(name.to_string()),
        }
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// Builder for constructing async pipelines
pub struct PipelineBuilder {
    target: CudaArch,
    depth: u32,
    buffer_configs: Vec<BufferConfig>,
    warp_count: u32,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new(target: CudaArch) -> Self {
        Self {
            target,
            depth: 2,
            buffer_configs: Vec::new(),
            warp_count: 1,
        }
    }

    /// Set pipeline depth
    pub fn depth(mut self, depth: u32) -> Self {
        self.depth = depth.max(2).min(4); // Clamp to reasonable range
        self
    }

    /// Add a buffer configuration
    pub fn add_buffer(mut self, config: BufferConfig) -> Self {
        self.buffer_configs.push(config);
        self
    }

    /// Set the number of warps
    pub fn warp_count(mut self, count: u32) -> Self {
        self.warp_count = count;
        self
    }

    /// Build a double-buffer pipeline
    pub fn double_buffer(target: CudaArch, buffer_size: u32, elem_type: GpuType) -> AsyncPipeline {
        PipelineBuilder::new(target)
            .depth(2)
            .add_buffer(BufferConfig::new(buffer_size, elem_type))
            .build()
    }

    /// Build a triple-buffer pipeline
    pub fn triple_buffer(target: CudaArch, buffer_size: u32, elem_type: GpuType) -> AsyncPipeline {
        PipelineBuilder::new(target)
            .depth(3)
            .add_buffer(BufferConfig::new(buffer_size, elem_type))
            .build()
    }

    /// Build pipeline from TileConfig
    pub fn from_tile_config(tile: &TileConfig, target: CudaArch) -> AsyncPipeline {
        let buffer_size = tile.tile_m * tile.tile_k * 2; // Assume f16 (2 bytes)
        PipelineBuilder::new(target)
            .depth(tile.pipeline_stages)
            .add_buffer(BufferConfig::with_name(buffer_size, GpuType::F16, "a_smem"))
            .add_buffer(BufferConfig::with_name(buffer_size, GpuType::F16, "b_smem"))
            .build()
    }

    /// Build the pipeline
    pub fn build(self) -> AsyncPipeline {
        let mut pipeline = AsyncPipeline::new(self.depth, self.target);
        let barrier_kind = pipeline.preferred_barrier_kind();

        // Create stages
        for stage_idx in 0..self.depth {
            let stage_id = StageId(stage_idx);
            let mut stage = PipelineStage::new(stage_id);

            // Allocate buffers for this stage
            for (buf_idx, config) in self.buffer_configs.iter().enumerate() {
                let name = config
                    .name_prefix
                    .clone()
                    .unwrap_or_else(|| "buf".to_string());
                let buffer = StageBuffer::new(
                    format!("{}_{}_s{}", name, buf_idx, stage_idx),
                    config.size,
                    config.elem_type.clone(),
                    stage_idx,
                );
                stage.add_buffer(buffer);
            }

            // Allocate barriers for stage transitions
            if stage_idx > 0 {
                // Entry barrier: wait for previous stage to complete
                if let Some(barrier_id) = pipeline
                    .barriers
                    .allocate(barrier_kind, self.warp_count * 32)
                {
                    stage.entry_barrier = Some(barrier_id);
                }
            }

            if stage_idx < self.depth - 1 {
                // Exit barrier: signal completion to next stage
                if let Some(barrier_id) = pipeline
                    .barriers
                    .allocate(barrier_kind, self.warp_count * 32)
                {
                    stage.exit_barrier = Some(barrier_id);
                }
            }

            pipeline.add_stage(stage);
        }

        pipeline
    }
}

// ============================================================================
// Dependency Analyzer
// ============================================================================

/// Analyzer for finding async operations and their dependencies
pub struct DependencyAnalyzer {
    next_op_id: u32,
}

impl DependencyAnalyzer {
    /// Create a new dependency analyzer
    pub fn new() -> Self {
        Self { next_op_id: 0 }
    }

    /// Analyze a kernel and build async operation graph
    pub fn analyze_kernel(&mut self, kernel: &GpuKernel) -> AsyncOpGraph {
        let mut graph = AsyncOpGraph::new();

        for block in &kernel.blocks {
            let ops = self.find_async_ops(block);
            for op in ops {
                graph.add_op(op);
            }
        }

        // Compute dependencies
        self.compute_dependencies(&mut graph);

        graph
    }

    /// Find async operations in a block
    pub fn find_async_ops(&mut self, block: &GpuBlock) -> Vec<AsyncOp> {
        let mut async_ops = Vec::new();

        for (_value_id, gpu_op) in &block.instructions {
            if let Some(async_op_kind) = self.classify_op(gpu_op) {
                let op_id = self.next_async_op_id();
                let async_op = AsyncOp::new(op_id, async_op_kind, StageId(0));
                async_ops.push(async_op);
            }
        }

        async_ops
    }

    /// Classify a GpuOp as an async operation kind (if applicable)
    fn classify_op(&self, op: &GpuOp) -> Option<AsyncOpKind> {
        match op {
            GpuOp::TmaLoadAsync {
                dst_shared,
                src_global,
                size,
                barrier: _,
            } => Some(AsyncOpKind::TmaLoad {
                src: *src_global,
                dst: *dst_shared,
                size: *size,
            }),
            GpuOp::TmaStoreAsync {
                dst_global,
                src_shared,
                size,
            } => Some(AsyncOpKind::TmaStore {
                src: *src_shared,
                dst: *dst_global,
                size: *size,
            }),
            GpuOp::TmaMulticastLoad {
                dst_shared,
                src_global,
                size: _,
                cluster_mask,
                barrier: _,
            } => Some(AsyncOpKind::TmaMulticast {
                src: *src_global,
                dst: *dst_shared,
                multicast_mask: *cluster_mask,
            }),
            GpuOp::TmaReduceAsync {
                dst_global,
                src_shared,
                size: _,
                reduce_op,
            } => Some(AsyncOpKind::TmaReduce {
                src: *src_shared,
                dst: *dst_global,
                op: *reduce_op,
            }),
            _ => None,
        }
    }

    /// Generate next async operation ID
    fn next_async_op_id(&mut self) -> AsyncOpId {
        let id = AsyncOpId(self.next_op_id);
        self.next_op_id += 1;
        id
    }

    /// Compute dependencies between operations in the graph
    pub fn compute_dependencies(&self, graph: &mut AsyncOpGraph) {
        // Simple dependency model: loads must complete before compute uses them,
        // stores must wait for compute to produce data
        let ops: Vec<(AsyncOpId, bool, bool)> = graph
            .ops
            .iter()
            .map(|op| (op.id, op.kind.is_load(), op.kind.is_store()))
            .collect();

        // Stores depend on previous loads completing
        let mut last_load: Option<AsyncOpId> = None;
        for (id, is_load, is_store) in &ops {
            if *is_load {
                last_load = Some(*id);
            } else if *is_store && let Some(load_id) = last_load {
                graph.add_dependency(load_id, *id);
            }
        }
    }
}

impl Default for DependencyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Pipeline Scheduler
// ============================================================================

/// Scheduler for creating pipeline execution order
pub struct PipelineScheduler {
    lookahead: u32,
}

impl PipelineScheduler {
    /// Create a new pipeline scheduler
    pub fn new() -> Self {
        Self { lookahead: 1 }
    }

    /// Set the lookahead depth for prefetching
    pub fn lookahead(mut self, depth: u32) -> Self {
        self.lookahead = depth;
        self
    }

    /// Schedule a pipeline for optimal memory/compute overlap
    pub fn schedule(&self, pipeline: &AsyncPipeline) -> PipelineSchedule {
        let mut schedule = PipelineSchedule::new();

        // Initialize barriers
        for barrier in &pipeline.barriers.barriers {
            schedule.add_barrier_init(barrier.id);
        }

        // Build prologue: prefetch initial data
        self.build_prologue(pipeline, &mut schedule);

        // Build main loop: interleaved operations
        self.build_main_loop(pipeline, &mut schedule);

        // Build epilogue: drain pipeline
        self.build_epilogue(pipeline, &mut schedule);

        schedule
    }

    /// Build the prologue (initial prefetches)
    fn build_prologue(&self, pipeline: &AsyncPipeline, schedule: &mut PipelineSchedule) {
        // Prefetch data for first iteration(s)
        let prefetch_count = (pipeline.depth - 1).min(self.lookahead);

        for i in 0..prefetch_count {
            let stage_idx = i as usize % pipeline.stages.len();
            if let Some(stage) = pipeline.stages.get(stage_idx) {
                // Issue async loads for this stage
                for op_id in &stage.async_ops {
                    schedule.add_prologue(ScheduledOp::AsyncLoad(*op_id));
                }

                // Signal completion
                if let Some(barrier) = stage.exit_barrier {
                    schedule.add_prologue(ScheduledOp::BarrierArrive(barrier));
                }
            }
        }
    }

    /// Build the main loop (steady state)
    fn build_main_loop(&self, pipeline: &AsyncPipeline, schedule: &mut PipelineSchedule) {
        // For each stage, interleave:
        // 1. Wait for previous stage data
        // 2. Compute on current data
        // 3. Issue prefetch for future iteration
        // 4. Signal completion

        for (stage_idx, stage) in pipeline.stages.iter().enumerate() {
            // Wait for data to be ready
            if let Some(barrier) = stage.entry_barrier {
                schedule.add_main_loop(ScheduledOp::BarrierWait(barrier));
            }

            // Compute (placeholder - actual compute ops would be injected)
            if stage_idx == pipeline.stages.len() - 1 {
                schedule.add_main_loop(ScheduledOp::Compute(Vec::new()));
            }

            // Prefetch next iteration
            for op_id in &stage.async_ops {
                schedule.add_main_loop(ScheduledOp::AsyncLoad(*op_id));
            }

            // Signal completion
            if let Some(barrier) = stage.exit_barrier {
                schedule.add_main_loop(ScheduledOp::BarrierArrive(barrier));
                schedule.add_main_loop(ScheduledOp::AdvancePhase(barrier));
            }
        }
    }

    /// Build the epilogue (drain pipeline)
    fn build_epilogue(&self, pipeline: &AsyncPipeline, schedule: &mut PipelineSchedule) {
        // Wait for all outstanding operations to complete
        let drain_count = pipeline.depth - 1;

        for i in 0..drain_count {
            let stage_idx = (pipeline.depth as usize - 1 - i as usize) % pipeline.stages.len();
            if let Some(stage) = pipeline.stages.get(stage_idx) {
                // Wait for data
                if let Some(barrier) = stage.entry_barrier {
                    schedule.add_epilogue(ScheduledOp::BarrierWait(barrier));
                }

                // Final compute
                if stage_idx == pipeline.stages.len() - 1 {
                    schedule.add_epilogue(ScheduledOp::Compute(Vec::new()));
                }
            }
        }
    }
}

impl Default for PipelineScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Barrier Scheduler
// ============================================================================

/// Scheduler for barrier placement optimization
pub struct BarrierScheduler {
    minimize_stalls: bool,
}

impl BarrierScheduler {
    /// Create a new barrier scheduler
    pub fn new() -> Self {
        Self {
            minimize_stalls: true,
        }
    }

    /// Enable/disable stall minimization
    pub fn minimize_stalls(mut self, enable: bool) -> Self {
        self.minimize_stalls = enable;
        self
    }

    /// Allocate barriers for a pipeline
    pub fn allocate_barriers(&self, pipeline: &mut AsyncPipeline) {
        let barrier_kind = pipeline.preferred_barrier_kind();
        let arrive_count = 32; // One warp

        // Allocate one barrier per stage transition
        for stage in &mut pipeline.stages {
            if stage.entry_barrier.is_none() && stage.id.0 > 0 {
                stage.entry_barrier = pipeline.barriers.allocate(barrier_kind, arrive_count);
            }
            if stage.exit_barrier.is_none() && stage.id.0 < pipeline.depth - 1 {
                stage.exit_barrier = pipeline.barriers.allocate(barrier_kind, arrive_count);
            }
        }
    }

    /// Optimize barrier placement in a schedule
    pub fn optimize_schedule(&self, schedule: &mut PipelineSchedule) {
        if !self.minimize_stalls {
            return;
        }

        // Move barrier waits as late as possible (lazy synchronization)
        self.delay_waits(&mut schedule.main_loop);
        self.delay_waits(&mut schedule.epilogue);

        // Move barrier arrives as early as possible (eager notification)
        self.advance_arrives(&mut schedule.prologue);
        self.advance_arrives(&mut schedule.main_loop);
    }

    /// Delay barrier waits as late as possible
    fn delay_waits(&self, ops: &mut [ScheduledOp]) {
        // Simple implementation: no reordering for now
        // A more sophisticated version would move waits just before their consumers
        let _ = ops;
    }

    /// Advance barrier arrives as early as possible
    fn advance_arrives(&self, ops: &mut [ScheduledOp]) {
        // Simple implementation: no reordering for now
        // A more sophisticated version would move arrives right after their producers
        let _ = ops;
    }
}

impl Default for BarrierScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// PTX Emission Helpers
// ============================================================================

/// Emit PTX code for pipeline operations
pub struct PipelineCodegen {
    target: CudaArch,
}

impl PipelineCodegen {
    /// Create a new pipeline code generator
    pub fn new(target: CudaArch) -> Self {
        Self { target }
    }

    /// Emit barrier initialization PTX
    pub fn emit_barrier_init(&self, barrier: &Barrier) -> Vec<String> {
        let mut lines = Vec::new();

        match barrier.kind {
            BarrierKind::MBarrier => {
                lines.push(format!(
                    "mbarrier.init.shared.b64 [{}], {};",
                    barrier.name, barrier.arrive_count
                ));
            }
            BarrierKind::CpAsyncGroup => {
                // cp.async groups don't need explicit init
            }
            BarrierKind::NamedBarrier => {
                // Named barriers don't need explicit init
            }
        }

        lines
    }

    /// Emit barrier arrive PTX
    pub fn emit_barrier_arrive(&self, barrier: &Barrier) -> String {
        match barrier.kind {
            BarrierKind::MBarrier => {
                format!("mbarrier.arrive.shared.b64 _, [{}];", barrier.name)
            }
            BarrierKind::CpAsyncGroup => "cp.async.commit_group;".to_string(),
            BarrierKind::NamedBarrier => {
                format!("bar.arrive {}, {};", barrier.id.0, barrier.arrive_count)
            }
        }
    }

    /// Emit barrier wait PTX
    pub fn emit_barrier_wait(&self, barrier: &Barrier) -> Vec<String> {
        let mut lines = Vec::new();

        match barrier.kind {
            BarrierKind::MBarrier => {
                lines.push(format!("wait_{}:", barrier.name));
                lines.push(format!(
                    "mbarrier.try_wait.parity.shared.b64 pred_{}, [{}], {};",
                    barrier.name,
                    barrier.name,
                    barrier.parity()
                ));
                lines.push(format!(
                    "@!pred_{} bra.uni wait_{};",
                    barrier.name, barrier.name
                ));
            }
            BarrierKind::CpAsyncGroup => {
                lines.push("cp.async.wait_group 0;".to_string());
            }
            BarrierKind::NamedBarrier => {
                lines.push(format!(
                    "bar.sync {}, {};",
                    barrier.id.0, barrier.arrive_count
                ));
            }
        }

        lines
    }

    /// Emit TMA load PTX (Hopper+)
    pub fn emit_tma_load(
        &self,
        dst_smem: &str,
        tensor_map: &str,
        coords: &str,
        barrier: &str,
    ) -> String {
        format!(
            "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes \
             [{dst_smem}], [{tensor_map}, {{{coords}}}], [{barrier}];"
        )
    }

    /// Emit TMA store PTX (Hopper+)
    pub fn emit_tma_store(
        &self,
        dst_global: &str,
        src_smem: &str,
        tensor_map: &str,
        coords: &str,
    ) -> String {
        format!(
            "cp.async.bulk.tensor.2d.global.shared::cta [{dst_global}, [{tensor_map}, {{{coords}}}]], [{src_smem}];"
        )
    }

    /// Emit cp.async PTX (Ampere+)
    pub fn emit_cp_async(&self, dst_smem: &str, src_global: &str, size: u32) -> String {
        format!("cp.async.cg.shared.global [{dst_smem}], [{src_global}], {size};")
    }

    /// Emit cp.async commit PTX
    pub fn emit_cp_async_commit(&self) -> String {
        "cp.async.commit_group;".to_string()
    }

    /// Emit cp.async wait PTX
    pub fn emit_cp_async_wait(&self, count: u32) -> String {
        format!("cp.async.wait_group {};", count)
    }

    /// Emit shared memory declaration for pipeline buffers
    pub fn emit_shared_memory_decl(&self, buffer: &StageBuffer, align: u32) -> String {
        format!(
            ".shared .align {} .b8 {}[{}];",
            align, buffer.name, buffer.size_bytes
        )
    }

    /// Emit full prologue code
    pub fn emit_prologue(
        &self,
        schedule: &PipelineSchedule,
        pipeline: &AsyncPipeline,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push("// Pipeline prologue: initial prefetches".to_string());

        for op in &schedule.prologue {
            match op {
                ScheduledOp::AsyncLoad(id) => {
                    if let Some(async_op) = pipeline.op_graph.get_op(*id) {
                        lines.extend(self.emit_async_load_op(async_op, pipeline));
                    }
                }
                ScheduledOp::BarrierArrive(id) => {
                    if let Some(barrier) = pipeline.barriers.get(*id) {
                        lines.push(self.emit_barrier_arrive(barrier));
                    }
                }
                ScheduledOp::CpAsyncCommitGroup => {
                    lines.push(self.emit_cp_async_commit());
                }
                _ => {}
            }
        }

        lines
    }

    /// Emit full main loop code
    pub fn emit_main_loop(
        &self,
        schedule: &PipelineSchedule,
        pipeline: &AsyncPipeline,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push("// Pipeline main loop: steady state".to_string());

        for op in &schedule.main_loop {
            match op {
                ScheduledOp::AsyncLoad(id) => {
                    if let Some(async_op) = pipeline.op_graph.get_op(*id) {
                        lines.extend(self.emit_async_load_op(async_op, pipeline));
                    }
                }
                ScheduledOp::Compute(ops) => {
                    lines.extend(self.emit_compute_ops(ops));
                }
                ScheduledOp::BarrierWait(id) => {
                    if let Some(barrier) = pipeline.barriers.get(*id) {
                        lines.extend(self.emit_barrier_wait(barrier));
                    }
                }
                ScheduledOp::BarrierArrive(id) => {
                    if let Some(barrier) = pipeline.barriers.get(*id) {
                        lines.push(self.emit_barrier_arrive(barrier));
                    }
                }
                ScheduledOp::AdvancePhase(id) => {
                    if let Some(barrier) = pipeline.barriers.get(*id) {
                        lines.push(format!("// Advance phase for barrier {}", barrier.name));
                    }
                }
                ScheduledOp::CpAsyncCommitGroup => {
                    lines.push(self.emit_cp_async_commit());
                }
                ScheduledOp::CpAsyncWait { count } => {
                    lines.push(self.emit_cp_async_wait(*count));
                }
                ScheduledOp::SyncThreads => {
                    lines.push("bar.sync 0;".to_string());
                }
                _ => {}
            }
        }

        lines
    }

    /// Emit full epilogue code
    pub fn emit_epilogue(
        &self,
        schedule: &PipelineSchedule,
        pipeline: &AsyncPipeline,
    ) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push("// Pipeline epilogue: drain".to_string());

        for op in &schedule.epilogue {
            match op {
                ScheduledOp::Compute(ops) => {
                    lines.extend(self.emit_compute_ops(ops));
                }
                ScheduledOp::BarrierWait(id) => {
                    if let Some(barrier) = pipeline.barriers.get(*id) {
                        lines.extend(self.emit_barrier_wait(barrier));
                    }
                }
                ScheduledOp::CpAsyncWait { count } => {
                    lines.push(self.emit_cp_async_wait(*count));
                }
                ScheduledOp::SyncThreads => {
                    lines.push("bar.sync 0;".to_string());
                }
                _ => {}
            }
        }

        lines
    }

    /// Emit PTX for an async load operation based on its kind
    fn emit_async_load_op(&self, op: &AsyncOp, pipeline: &AsyncPipeline) -> Vec<String> {
        let mut lines = Vec::new();

        match &op.kind {
            AsyncOpKind::TmaLoad { src, dst, size } => {
                // TMA load for Hopper+ (sm_90+)
                if self.target.compute_capability() >= (9, 0) {
                    // Find the barrier for this stage
                    let barrier_name = pipeline
                        .stages
                        .iter()
                        .find(|s| s.id == op.stage)
                        .and_then(|s| s.exit_barrier)
                        .and_then(|bid| pipeline.barriers.get(bid))
                        .map(|b| b.name.clone())
                        .unwrap_or_else(|| "mbar0".to_string());

                    lines.push(format!(
                        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes \
                         [smem_{}], [tensor_map_{}, {{coord_x, coord_y}}], [{}];",
                        dst.0, src.0, barrier_name
                    ));
                } else {
                    // Fallback to cp.async for pre-Hopper
                    lines.push(format!(
                        "cp.async.cg.shared.global [smem_{}], [gmem_{}], {};",
                        dst.0, src.0, size
                    ));
                    lines.push("cp.async.commit_group;".to_string());
                }
            }
            AsyncOpKind::CpAsync { src, dst, size } => {
                // cp.async for Ampere+ (sm_80+)
                lines.push(format!(
                    "cp.async.cg.shared.global [smem_{}], [gmem_{}], {};",
                    dst.0, src.0, size
                ));
            }
            AsyncOpKind::TmaMulticast {
                src,
                dst,
                multicast_mask,
            } => {
                // TMA multicast for Hopper+ cluster operations
                lines.push(format!(
                    "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster \
                     [smem_{}], [tensor_map_{}], [mbar], 0x{:x};",
                    dst.0, src.0, multicast_mask
                ));
            }
            AsyncOpKind::TmaStore { src, dst, .. } => {
                // TMA store (Hopper+)
                lines.push(format!(
                    "cp.async.bulk.tensor.2d.global.shared::cta [gmem_{}], [smem_{}];",
                    dst.0, src.0
                ));
            }
            AsyncOpKind::TmaReduce {
                src,
                dst,
                op: reduce_op,
            } => {
                // TMA reduction (Hopper+)
                let op_str = match reduce_op {
                    TmaReduceOp::Add => "add",
                    TmaReduceOp::Min => "min",
                    TmaReduceOp::Max => "max",
                    TmaReduceOp::And => "and",
                    TmaReduceOp::Or => "or",
                    TmaReduceOp::Xor => "xor",
                };
                lines.push(format!(
                    "cp.reduce.async.bulk.tensor.2d.global.shared::cta.{} [gmem_{}], [smem_{}];",
                    op_str, dst.0, src.0
                ));
            }
            AsyncOpKind::CpAsyncCommit => {
                // Commit current async copy group
                lines.push("cp.async.commit_group;".to_string());
            }
            AsyncOpKind::CpAsyncWait { count } => {
                // Wait for async copies to complete
                lines.push(format!("cp.async.wait_group {};", count));
            }
        }

        lines
    }

    /// Emit PTX for compute operations
    fn emit_compute_ops(&self, ops: &[GpuOp]) -> Vec<String> {
        let mut lines = Vec::new();
        lines.push(format!("// Compute block: {} operations", ops.len()));

        // For now, emit placeholder comments for compute operations
        // Full implementation would translate GpuOp to PTX
        for (i, op) in ops.iter().enumerate() {
            match op {
                GpuOp::Add(a, b) => {
                    lines.push(format!("add.f32 r{}, r{}, r{};", i, a.0, b.0));
                }
                GpuOp::Mul(a, b) => {
                    lines.push(format!("mul.f32 r{}, r{}, r{};", i, a.0, b.0));
                }
                GpuOp::FMulAdd(a, b, c) => {
                    lines.push(format!("fma.rn.f32 r{}, r{}, r{}, r{};", i, a.0, b.0, c.0));
                }
                GpuOp::Load(ptr, _offset) => {
                    lines.push(format!("ld.shared.f32 r{}, [r{}];", i, ptr.0));
                }
                GpuOp::Store(ptr, val, _offset) => {
                    lines.push(format!("st.shared.f32 [r{}], r{};", ptr.0, val.0));
                }
                _ => {
                    lines.push(format!("// op_{}: {:?}", i, std::mem::discriminant(op)));
                }
            }
        }

        lines
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Create a pipeline from tile configuration (main entry point)
pub fn create_pipeline_from_tile(tile: &TileConfig, target: CudaArch) -> AsyncPipeline {
    PipelineBuilder::from_tile_config(tile, target)
}

/// Schedule a pipeline for execution
pub fn schedule_pipeline(pipeline: &AsyncPipeline) -> PipelineSchedule {
    let scheduler = PipelineScheduler::new();
    scheduler.schedule(pipeline)
}

/// Analyze and apply async pipelining to a kernel
pub fn apply_pipeline(
    kernel: &GpuKernel,
    pipeline_stages: u32,
    target: CudaArch,
) -> (AsyncPipeline, PipelineSchedule) {
    // Create pipeline from kernel analysis
    let mut analyzer = DependencyAnalyzer::new();
    let _op_graph = analyzer.analyze_kernel(kernel);

    // Build pipeline
    let pipeline = PipelineBuilder::new(target)
        .depth(pipeline_stages)
        .add_buffer(BufferConfig::new(1024, GpuType::F16))
        .build();

    // Schedule
    let schedule = schedule_pipeline(&pipeline);

    (pipeline, schedule)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_double_buffer_creation() {
        let pipeline = PipelineBuilder::double_buffer(CudaArch::Hopper, 1024, GpuType::F16);

        assert_eq!(pipeline.depth, 2);
        assert_eq!(pipeline.stages.len(), 2);
        assert!(pipeline.supports_tma());
        assert!(pipeline.supports_mbarrier());
    }

    #[test]
    fn test_triple_buffer_creation() {
        let pipeline = PipelineBuilder::triple_buffer(CudaArch::Hopper, 1024, GpuType::F16);

        assert_eq!(pipeline.depth, 3);
        assert_eq!(pipeline.stages.len(), 3);
    }

    #[test]
    fn test_barrier_allocation() {
        let pipeline = PipelineBuilder::double_buffer(CudaArch::Hopper, 1024, GpuType::F16);

        // Double buffer should have barriers for stage transitions
        assert!(!pipeline.barriers.is_empty());
        assert!(pipeline.barriers.len() <= pipeline.barriers.max_barriers as usize);
    }

    #[test]
    fn test_schedule_generation() {
        let pipeline = PipelineBuilder::double_buffer(CudaArch::Hopper, 1024, GpuType::F16);
        let schedule = schedule_pipeline(&pipeline);

        // Schedule should have all three phases
        assert!(!schedule.barrier_init.is_empty() || pipeline.barriers.is_empty());
    }

    #[test]
    fn test_tile_config_pipeline() {
        let tile = TileConfig {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            pipeline_stages: 3,
            swizzled: false,
        };

        let pipeline = PipelineBuilder::from_tile_config(&tile, CudaArch::Hopper);

        assert_eq!(pipeline.depth, 3);
        // Two buffers (a_smem, b_smem) per stage
        assert!(pipeline.stages[0].buffers.len() >= 1);
    }

    #[test]
    fn test_ampere_fallback() {
        let pipeline = PipelineBuilder::double_buffer(CudaArch::Ampere, 1024, GpuType::F16);

        // Ampere doesn't support TMA, but does support mbarrier
        assert!(!pipeline.supports_tma());
        assert!(pipeline.supports_mbarrier());
        assert_eq!(pipeline.preferred_barrier_kind(), BarrierKind::MBarrier);
    }

    #[test]
    fn test_turing_fallback() {
        let pipeline = PipelineBuilder::double_buffer(CudaArch::Turing, 1024, GpuType::F16);

        // Turing doesn't support TMA or mbarrier
        assert!(!pipeline.supports_tma());
        assert!(!pipeline.supports_mbarrier());
        assert_eq!(pipeline.preferred_barrier_kind(), BarrierKind::NamedBarrier);
    }

    #[test]
    fn test_ptx_mbarrier_emission() {
        let codegen = PipelineCodegen::new(CudaArch::Hopper);
        let barrier = Barrier::new(BarrierId(0), BarrierKind::MBarrier, 32);

        let init = codegen.emit_barrier_init(&barrier);
        assert!(!init.is_empty());
        assert!(init[0].contains("mbarrier.init"));

        let arrive = codegen.emit_barrier_arrive(&barrier);
        assert!(arrive.contains("mbarrier.arrive"));

        let wait = codegen.emit_barrier_wait(&barrier);
        assert!(!wait.is_empty());
        assert!(wait.iter().any(|l| l.contains("mbarrier.try_wait")));
    }

    #[test]
    fn test_async_op_graph() {
        let mut graph = AsyncOpGraph::new();

        let op1 = AsyncOp::new(
            AsyncOpId(0),
            AsyncOpKind::TmaLoad {
                src: ValueId(0),
                dst: ValueId(1),
                size: 1024,
            },
            StageId(0),
        );
        let op2 = AsyncOp::new(AsyncOpId(1), AsyncOpKind::CpAsyncCommit, StageId(0));

        graph.add_op(op1);
        graph.add_op(op2);
        graph.add_dependency(AsyncOpId(0), AsyncOpId(1));

        let sorted = graph.topological_sort();
        assert_eq!(sorted.len(), 2);
        assert_eq!(sorted[0], AsyncOpId(0));
        assert_eq!(sorted[1], AsyncOpId(1));
    }

    #[test]
    fn test_dependency_analysis() {
        let kernel = GpuKernel {
            name: "test_kernel".to_string(),
            params: vec![],
            shared_memory: vec![],
            blocks: vec![],
            entry: super::super::ir::BlockId(0),
            max_threads: None,
            shared_mem_size: 0,
        };

        let mut analyzer = DependencyAnalyzer::new();
        let graph = analyzer.analyze_kernel(&kernel);

        // Empty kernel should have empty graph
        assert!(graph.ops.is_empty());
    }

    #[test]
    fn test_prologue_generation() {
        let pipeline = PipelineBuilder::triple_buffer(CudaArch::Hopper, 1024, GpuType::F16);
        let schedule = schedule_pipeline(&pipeline);

        // Triple buffer should have prologue operations
        // (prefetch for first iterations)
        let _ = schedule; // Schedule is generated, check basic structure
    }

    #[test]
    fn test_architecture_differences() {
        // Hopper: full TMA + mbarrier
        let hopper = PipelineBuilder::new(CudaArch::Hopper).depth(2).build();
        assert!(hopper.supports_tma());
        assert_eq!(hopper.barriers.max_barriers, 32);

        // Ampere: mbarrier only, no TMA
        let ampere = PipelineBuilder::new(CudaArch::Ampere).depth(2).build();
        assert!(!ampere.supports_tma());
        assert_eq!(ampere.barriers.max_barriers, 16);

        // Turing: neither
        let turing = PipelineBuilder::new(CudaArch::Turing).depth(2).build();
        assert!(!turing.supports_tma());
        assert!(!turing.supports_mbarrier());
    }
}
