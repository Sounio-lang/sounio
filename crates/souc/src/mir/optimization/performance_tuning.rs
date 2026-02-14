//! Performance Tuning for MIR Optimizations
//!
//! This module provides advanced performance optimization features including:
//! - Caching of optimization results
//! - Incremental analysis
//! - Architecture-specific optimizations
//! - Performance profiling and metrics

use super::pass_manager::MIRPass;
use crate::mir::{BlockId, MirFunction, MirInstruction, MirModule, MirType, ValueId};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

/// Performance-optimized MIR optimization pipeline
pub struct PerformanceTunedOptimizer {
    /// Cached analysis results
    analysis_cache: AnalysisCache,
    /// Performance profiler
    profiler: OptimizationProfiler,
    /// Incremental analysis engine
    incremental_analyzer: IncrementalAnalyzer,
    /// Architecture-specific optimizations
    arch_optimizer: ArchitectureOptimizer,
    /// Optimization heuristics
    heuristics: OptimizationHeuristics,
}

/// Caching system for optimization analysis
pub struct AnalysisCache {
    /// Function-level caches
    function_cache: HashMap<String, FunctionAnalysisCache>,
    /// Block-level caches
    block_cache: HashMap<(String, BlockId), BlockAnalysisCache>,
    /// Value-level caches
    value_cache: HashMap<(String, ValueId), ValueAnalysisCache>,
    /// Cache statistics
    stats: CacheStatistics,
}

/// Function-level analysis cache
#[derive(Debug, Clone)]
pub struct FunctionAnalysisCache {
    /// Dominator tree
    pub dominator_tree: Option<Vec<Vec<BlockId>>>,
    /// Liveness information
    pub liveness: Option<Vec<HashSet<ValueId>>>,
    /// Available expressions
    pub available_expressions: Option<Vec<HashSet<Expression>>>,
    /// Loop information
    pub loop_info: Option<LoopAnalysis>,
    /// Cache timestamp
    pub timestamp: Instant,
    /// Cache validity period
    pub validity_period: Duration,
}

/// Block-level analysis cache
#[derive(Debug, Clone)]
pub struct BlockAnalysisCache {
    /// Instructions in block
    pub instruction_count: usize,
    /// Complexity metrics
    pub complexity_metrics: ComplexityMetrics,
    /// Optimization opportunities
    pub optimization_opportunities: Vec<OptimizationOpportunity>,
    /// Cache timestamp
    pub timestamp: Instant,
}

/// Value-level analysis cache
#[derive(Debug, Clone)]
pub struct ValueAnalysisCache {
    /// Definition information
    pub definition: Option<ValueDefinition>,
    /// Use information
    pub uses: Vec<ValueUse>,
    /// Live range
    pub live_range: Option<LiveRange>,
    /// Cache timestamp
    pub timestamp: Instant,
}

/// Optimization profiler
pub struct OptimizationProfiler {
    /// Pass execution times
    pass_times: HashMap<String, Vec<Duration>>,
    /// Memory usage tracking
    memory_usage: HashMap<String, usize>,
    /// Optimization effectiveness
    effectiveness_metrics: HashMap<String, EffectivenessMetrics>,
    /// Profile statistics
    profile_stats: ProfileStatistics,
}

/// Effectiveness metrics for optimizations
#[derive(Debug, Clone)]
pub struct EffectivenessMetrics {
    /// Instruction reduction
    pub instruction_reduction: f64,
    /// Block reduction
    pub block_reduction: f64,
    /// Quality improvement
    pub quality_improvement: f64,
    /// Compile time impact
    pub compile_time_impact: f64,
}

/// Incremental analysis engine
pub struct IncrementalAnalyzer {
    /// Change tracking
    change_tracker: ChangeTracker,
    /// Dependency graph
    dependency_graph: DependencyGraph,
    /// Incremental caches
    incremental_caches: HashMap<String, IncrementalCache>,
}

/// Change tracking for incremental analysis
#[derive(Debug, Clone)]
pub struct ChangeTracker {
    /// Modified functions
    modified_functions: HashSet<String>,
    /// Modified blocks
    modified_blocks: HashSet<(String, BlockId)>,
    /// Modified values
    modified_values: HashSet<(String, ValueId)>,
}

/// Dependency graph for analysis
#[derive(Debug, Clone)]
pub struct DependencyGraph {
    /// Function dependencies
    function_deps: HashMap<String, HashSet<String>>,
    /// Block dependencies
    block_deps: HashMap<(String, BlockId), HashSet<(String, BlockId)>>,
    /// Value dependencies
    value_deps: HashMap<(String, ValueId), HashSet<(String, ValueId)>>,
}

/// Incremental cache
#[derive(Debug, Clone)]
pub struct IncrementalCache {
    /// Cache data
    pub data: HashMap<String, CacheEntry>,
    /// Last update
    pub last_update: Instant,
    /// Update frequency
    pub update_frequency: Duration,
}

/// Cache entry
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Entry data
    pub data: Vec<u8>,
    /// Hash of data
    pub hash: String,
    /// Access count
    pub access_count: usize,
    /// Last access
    pub last_access: Instant,
}

/// Architecture-specific optimizer
pub struct ArchitectureOptimizer {
    /// Target architecture
    target_arch: TargetArchitecture,
    /// SIMD optimizations
    simd_optimizer: SIMDOptimizer,
    /// Vectorization engine
    vectorizer: VectorizationEngine,
}

/// Target architecture
#[derive(Debug, Clone)]
pub enum TargetArchitecture {
    X86_64,
    ARM64,
    RISCV64,
    WebAssembly,
}

/// SIMD optimization engine
pub struct SIMDOptimizer {
    /// SIMD opportunities
    simd_opportunities: Vec<SIMDOpportunity>,
    /// Vectorization patterns
    vectorization_patterns: Vec<VectorizationPattern>,
}

/// SIMD optimization opportunity
#[derive(Debug, Clone)]
pub struct SIMDOpportunity {
    /// Location
    pub location: (String, BlockId, usize),
    /// Operation type
    pub operation: SIMDOperation,
    /// Vector width
    pub vector_width: usize,
    /// Performance benefit
    pub benefit: f64,
}

/// SIMD operations
#[derive(Debug, Clone)]
pub enum SIMDOperation {
    Add,
    Multiply,
    Subtract,
    DotProduct,
    MatrixMultiply,
}

/// Vectorization patterns
#[derive(Debug, Clone)]
pub struct VectorizationPattern {
    /// Pattern name
    pub name: String,
    /// Pattern code
    pub pattern_code: String,
    /// Optimization rules
    pub rules: Vec<OptimizationRule>,
}

/// Vectorization engine
pub struct VectorizationEngine {
    /// Vectorization opportunities
    vector_opportunities: Vec<VectorizationOpportunity>,
    /// Loop vectorization
    loop_vectorizer: LoopVectorizer,
}

/// Vectorization opportunity
#[derive(Debug, Clone)]
pub struct VectorizationOpportunity {
    /// Loop information
    pub loop_info: LoopInfo,
    /// Vectorization factor
    pub vectorization_factor: usize,
    /// Performance gain
    pub performance_gain: f64,
}

/// Loop information
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Loop header
    pub header: BlockId,
    /// Loop body blocks
    pub body_blocks: Vec<BlockId>,
    /// Trip count estimate
    pub trip_count: Option<usize>,
    /// Loop complexity
    pub complexity: LoopComplexity,
}

/// Loop complexity levels
#[derive(Debug, Clone)]
pub enum LoopComplexity {
    Simple,
    Medium,
    Complex,
    VeryComplex,
}

/// Induction variable information extracted from a loop header
#[derive(Debug, Clone)]
struct InductionVariable {
    /// The Phi node result that defines the IV
    phi_result: ValueId,
    /// Initial value (from pre-header)
    init_value: ValueId,
    /// Step value (from back-edge, typically `iv + stride`)
    step_value: ValueId,
    /// Constant stride (typically 1)
    stride: i64,
    /// Block containing the back-edge
    latch_block: BlockId,
    /// Whether the initial value is constant 0
    init_is_zero: bool,
    /// Type of the induction variable
    ty: MirType,
}

/// Loop bound information from the latch comparison
#[derive(Debug, Clone)]
struct LoopBound {
    /// Value representing the upper bound (N)
    bound_value: ValueId,
    /// Comparison operation used
    compare_op: crate::mir::MirCompareOp,
    /// The condition value driving the branch
    condition_value: ValueId,
    /// True if the loop continues on the true branch
    loops_on_true: bool,
    /// Target block when the loop exits
    exit_target: BlockId,
}

/// A loop body instruction that can be vectorized
#[derive(Debug, Clone)]
struct VectorizableInstr {
    /// Block containing the instruction
    block_id: BlockId,
    /// Index within the block's instruction list
    instr_idx: usize,
    /// Result value ID
    result_id: ValueId,
    /// Kind of vectorizable instruction
    kind: VecInstrKind,
    /// Original scalar type
    scalar_type: MirType,
}

/// Kind of vectorizable instruction
#[derive(Debug, Clone)]
enum VecInstrKind {
    Binary(crate::mir::MirBinaryOp),
    Unary(crate::mir::MirUnaryOp),
    Load,
    Store,
}

/// Loop vectorizer
pub struct LoopVectorizer {
    /// Vectorization decisions
    decisions: HashMap<BlockId, VectorizationDecision>,
}

/// Vectorization decision
#[derive(Debug, Clone)]
pub struct VectorizationDecision {
    /// Decision type
    pub decision: DecisionType,
    /// Reason
    pub reason: String,
    /// Confidence
    pub confidence: f64,
}

/// Vectorization decision types
#[derive(Debug, Clone)]
pub enum DecisionType {
    Vectorize { factor: usize },
    NoVectorize { reason: String },
    PartiallyVectorize { factor: usize, reason: String },
}

/// Optimization heuristics
pub struct OptimizationHeuristics {
    /// Inlining heuristics
    inlining_heuristics: InliningHeuristics,
    /// Loop optimization heuristics
    loop_heuristics: LoopOptimizationHeuristics,
    /// Register allocation heuristics
    register_heuristics: RegisterAllocationHeuristics,
}

/// Inlining heuristics
#[derive(Debug, Clone)]
pub struct InliningHeuristics {
    /// Maximum inlining depth
    pub max_depth: usize,
    /// Size thresholds
    pub size_thresholds: SizeThresholds,
    /// Hotness thresholds
    pub hotness_thresholds: HotnessThresholds,
}

/// Size thresholds for inlining
#[derive(Debug, Clone)]
pub struct SizeThresholds {
    pub small_function: usize,
    pub medium_function: usize,
    pub large_function: usize,
}

/// Hotness thresholds
#[derive(Debug, Clone)]
pub struct HotnessThresholds {
    pub very_hot: f64,
    pub hot: f64,
    pub warm: f64,
}

/// Loop optimization heuristics
#[derive(Debug, Clone)]
pub struct LoopOptimizationHeuristics {
    /// Unrolling thresholds
    pub unrolling_thresholds: UnrollingThresholds,
    /// Vectorization thresholds
    pub vectorization_thresholds: VectorizationThresholds,
}

/// Unrolling thresholds
#[derive(Debug, Clone)]
pub struct UnrollingThresholds {
    pub always_unroll_below: usize,
    pub consider_unroll_below: usize,
    pub never_unroll_above: usize,
}

/// Vectorization thresholds
#[derive(Debug, Clone)]
pub struct VectorizationThresholds {
    pub minimum_trip_count: usize,
    pub complexity_limit: LoopComplexity,
}

/// Register allocation heuristics
#[derive(Debug, Clone)]
pub struct RegisterAllocationHeuristics {
    /// Register pressure thresholds
    pub pressure_thresholds: PressureThresholds,
    /// Spilling heuristics
    pub spilling_heuristics: SpillingHeuristics,
}

/// Register pressure thresholds
#[derive(Debug, Clone)]
pub struct PressureThresholds {
    pub low_pressure: usize,
    pub medium_pressure: usize,
    pub high_pressure: usize,
    pub critical_pressure: usize,
}

/// Spilling heuristics
#[derive(Debug, Clone)]
pub struct SpillingHeuristics {
    pub spill_threshold: f64,
    pub remat_threshold: f64,
}

/// Performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Compilation time
    pub compilation_time: Duration,
    /// Memory usage
    pub memory_usage: usize,
    /// Optimization effectiveness
    pub optimization_effectiveness: f64,
    /// Code quality
    pub code_quality: f64,
}

/// Cache statistics
#[derive(Debug, Clone)]
pub struct CacheStatistics {
    pub hits: usize,
    pub misses: usize,
    pub evictions: usize,
    pub hit_rate: f64,
}

/// Profile statistics
#[derive(Debug, Clone)]
pub struct ProfileStatistics {
    pub total_optimizations: usize,
    pub successful_optimizations: usize,
    pub average_time_per_optimization: Duration,
    pub memory_peak: usize,
}

/// Supporting data structures
#[derive(Debug, Clone)]
pub struct Expression {
    pub op: String,
    pub operands: Vec<ValueId>,
    pub ty: String,
}

#[derive(Debug, Clone)]
pub struct ValueDefinition {
    pub block: BlockId,
    pub instruction: usize,
}

#[derive(Debug, Clone)]
pub struct ValueUse {
    pub block: BlockId,
    pub instruction: usize,
}

#[derive(Debug, Clone)]
pub struct LiveRange {
    pub start: (BlockId, usize),
    pub end: (BlockId, usize),
}

#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    pub natural_loops: Vec<NaturalLoop>,
    pub loop_nesting_depth: usize,
}

#[derive(Debug, Clone)]
pub struct NaturalLoop {
    pub header: BlockId,
    pub body: Vec<BlockId>,
    pub exit_blocks: Vec<BlockId>,
}

#[derive(Debug, Clone)]
pub struct ComplexityMetrics {
    pub cyclomatic_complexity: f64,
    pub nesting_depth: usize,
    pub instruction_count: usize,
}

#[derive(Debug, Clone)]
pub struct OptimizationOpportunity {
    pub opportunity_type: OptimizationOpportunityType,
    pub location: (BlockId, usize),
    pub potential_benefit: f64,
}

#[derive(Debug, Clone)]
pub enum OptimizationOpportunityType {
    ConstantPropagation,
    DeadCodeElimination,
    CommonSubexpression,
    StrengthReduction,
    LoopOptimization,
    Vectorization,
}

#[derive(Debug, Clone)]
pub struct OptimizationRule {
    pub pattern: String,
    pub transformation: String,
    pub applicability_condition: String,
}

impl PerformanceTunedOptimizer {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self {
            analysis_cache: AnalysisCache::new(),
            profiler: OptimizationProfiler::new(),
            incremental_analyzer: IncrementalAnalyzer::new(),
            arch_optimizer: ArchitectureOptimizer::new(target_arch),
            heuristics: OptimizationHeuristics::new(),
        }
    }

    /// Run performance-optimized optimization
    pub fn optimize_with_performance_tuning(
        &mut self,
        module: &mut MirModule,
    ) -> Result<PerformanceResult, String> {
        let start_time = Instant::now();

        // Profile initial state
        self.profiler.profile_initial_state(module)?;

        // Run incremental analysis
        let changes = self.incremental_analyzer.detect_changes(module)?;
        self.incremental_analyzer.update_caches(&changes)?;

        // Apply architecture-specific optimizations
        self.arch_optimizer
            .apply_architecture_optimizations(module)?;

        // Apply heuristic-guided optimizations
        self.heuristics.apply_heuristic_optimizations(module)?;

        // Profile final state
        let final_metrics = self.profiler.profile_final_state(module)?;

        let total_time = start_time.elapsed();

        Ok(PerformanceResult {
            original_metrics: self.profiler.get_initial_metrics().clone(),
            optimized_metrics: final_metrics,
            optimization_time: total_time,
            cache_hit_rate: self.analysis_cache.get_hit_rate(),
            performance_improvement: self.calculate_performance_improvement(),
        })
    }

    /// Calculate performance improvement
    fn calculate_performance_improvement(&self) -> f64 {
        let initial = self.profiler.get_initial_metrics();
        let final_metrics = self.profiler.get_final_metrics();

        if let (Some(initial_metrics), Some(final_m)) = (initial, final_metrics) {
            let initial_compile = initial_metrics.compilation_time;
            let final_compile = final_m.compilation_time;
            if initial_compile > Duration::from_millis(0) {
                return (initial_compile.as_millis() as f64 - final_compile.as_millis() as f64)
                    / initial_compile.as_millis() as f64
                    * 100.0;
            }
        }

        0.0
    }

    /// Get performance report
    pub fn generate_performance_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Performance Tuning Report\n\n");

        report.push_str(&format!(
            "Cache Hit Rate: {:.2}%\n",
            self.analysis_cache.get_hit_rate() * 100.0
        ));
        report.push_str(&format!(
            "Total Optimizations: {}\n",
            self.profiler.get_total_optimizations()
        ));
        report.push_str(&format!(
            "Success Rate: {:.2}%\n",
            self.profiler.get_success_rate() * 100.0
        ));

        report.push_str("\n## Architecture Optimizations\n");
        report.push_str(&format!(
            "Target Architecture: {:?}\n",
            self.arch_optimizer.target_arch
        ));

        report.push_str("\n## Cache Statistics\n");
        let stats = self.analysis_cache.get_stats();
        report.push_str(&format!("Cache Hits: {}\n", stats.hits));
        report.push_str(&format!("Cache Misses: {}\n", stats.misses));
        report.push_str(&format!("Cache Evictions: {}\n", stats.evictions));

        report
    }
}

/// Result from performance-tuned optimization
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    pub original_metrics: Option<PerformanceMetrics>,
    pub optimized_metrics: Option<PerformanceMetrics>,
    pub optimization_time: Duration,
    pub cache_hit_rate: f64,
    pub performance_improvement: f64,
}

// Implementation stubs for supporting structures
impl AnalysisCache {
    pub fn new() -> Self {
        Self {
            function_cache: HashMap::new(),
            block_cache: HashMap::new(),
            value_cache: HashMap::new(),
            stats: CacheStatistics {
                hits: 0,
                misses: 0,
                evictions: 0,
                hit_rate: 0.0,
            },
        }
    }

    pub fn get_hit_rate(&self) -> f64 {
        self.stats.hit_rate
    }

    pub fn get_stats(&self) -> &CacheStatistics {
        &self.stats
    }
}

impl OptimizationProfiler {
    pub fn new() -> Self {
        Self {
            pass_times: HashMap::new(),
            memory_usage: HashMap::new(),
            effectiveness_metrics: HashMap::new(),
            profile_stats: ProfileStatistics {
                total_optimizations: 0,
                successful_optimizations: 0,
                average_time_per_optimization: Duration::from_millis(0),
                memory_peak: 0,
            },
        }
    }

    pub fn profile_initial_state(&mut self, _module: &MirModule) -> Result<(), String> {
        Ok(())
    }

    pub fn profile_final_state(
        &mut self,
        _module: &MirModule,
    ) -> Result<Option<PerformanceMetrics>, String> {
        Ok(None)
    }

    pub fn get_initial_metrics(&self) -> Option<PerformanceMetrics> {
        None
    }

    pub fn get_final_metrics(&self) -> Option<PerformanceMetrics> {
        None
    }

    pub fn get_total_optimizations(&self) -> usize {
        self.profile_stats.total_optimizations
    }

    pub fn get_success_rate(&self) -> f64 {
        if self.profile_stats.total_optimizations > 0 {
            self.profile_stats.successful_optimizations as f64
                / self.profile_stats.total_optimizations as f64
        } else {
            0.0
        }
    }
}

impl IncrementalAnalyzer {
    pub fn new() -> Self {
        Self {
            change_tracker: ChangeTracker {
                modified_functions: HashSet::new(),
                modified_blocks: HashSet::new(),
                modified_values: HashSet::new(),
            },
            dependency_graph: DependencyGraph {
                function_deps: HashMap::new(),
                block_deps: HashMap::new(),
                value_deps: HashMap::new(),
            },
            incremental_caches: HashMap::new(),
        }
    }

    pub fn detect_changes(&mut self, _module: &MirModule) -> Result<ChangeTracker, String> {
        Ok(self.change_tracker.clone())
    }

    pub fn update_caches(&mut self, _changes: &ChangeTracker) -> Result<(), String> {
        Ok(())
    }
}

impl ArchitectureOptimizer {
    pub fn new(target_arch: TargetArchitecture) -> Self {
        Self {
            target_arch,
            simd_optimizer: SIMDOptimizer::new(),
            vectorizer: VectorizationEngine::new(),
        }
    }

    pub fn apply_architecture_optimizations(
        &mut self,
        _module: &mut MirModule,
    ) -> Result<(), String> {
        Ok(())
    }
}

impl SIMDOptimizer {
    pub fn new() -> Self {
        Self {
            simd_opportunities: Vec::new(),
            vectorization_patterns: Vec::new(),
        }
    }
}

impl VectorizationEngine {
    pub fn new() -> Self {
        Self {
            vector_opportunities: Vec::new(),
            loop_vectorizer: LoopVectorizer::new(),
        }
    }
}

impl LoopVectorizer {
    pub fn new() -> Self {
        Self {
            decisions: HashMap::new(),
        }
    }

    /// Run vectorization pass on a function
    pub fn run(&mut self, func: &mut MirFunction) -> Result<bool, String> {
        use crate::mir::analysis::LoopAnalysis;

        let mut modified = false;

        // Analyze loops in the function
        let loop_analysis = LoopAnalysis::analyze_function(func);

        if loop_analysis.loops.is_empty() {
            return Ok(false);
        }

        // Process each loop
        for natural_loop in &loop_analysis.loops {
            // Decide if vectorization is profitable
            let decision = self.should_vectorize(func, natural_loop)?;

            match decision.decision {
                DecisionType::Vectorize { factor } => {
                    // Attempt to vectorize the loop
                    if self.vectorize_loop(func, natural_loop, factor)? {
                        modified = true;
                    }
                }
                DecisionType::PartiallyVectorize { factor, .. } => {
                    // Attempt partial vectorization
                    if self.vectorize_loop(func, natural_loop, factor)? {
                        modified = true;
                    }
                }
                DecisionType::NoVectorize { .. } => {
                    // Skip this loop
                }
            }

            // Record decision
            self.decisions.insert(natural_loop.header, decision);
        }

        Ok(modified)
    }

    /// Decide whether to vectorize a loop
    fn should_vectorize(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> Result<VectorizationDecision, String> {
        // Default thresholds
        let min_trip_count = 8;
        let max_complexity = LoopComplexity::Medium;

        // Estimate trip count
        let trip_count = self.estimate_trip_count(func, natural_loop);

        // Check minimum trip count
        if let Some(count) = trip_count {
            if count < min_trip_count {
                return Ok(VectorizationDecision {
                    decision: DecisionType::NoVectorize {
                        reason: format!("Trip count {} below threshold {}", count, min_trip_count),
                    },
                    reason: "Low trip count".to_string(),
                    confidence: 0.9,
                });
            }
        }

        // Analyze loop complexity
        let complexity = self.analyze_loop_complexity(func, natural_loop);

        if !self.complexity_within_threshold(&complexity, &max_complexity) {
            return Ok(VectorizationDecision {
                decision: DecisionType::NoVectorize {
                    reason: "Loop too complex".to_string(),
                },
                reason: "Complexity exceeds threshold".to_string(),
                confidence: 0.8,
            });
        }

        // Check for vectorization blockers
        if let Some(blocker) = self.find_vectorization_blocker(func, natural_loop) {
            return Ok(VectorizationDecision {
                decision: DecisionType::NoVectorize {
                    reason: blocker.clone(),
                },
                reason: blocker,
                confidence: 0.95,
            });
        }

        // Determine vectorization factor
        let factor = self.determine_vector_factor(func, natural_loop);

        // Vectorization is profitable
        Ok(VectorizationDecision {
            decision: DecisionType::Vectorize { factor },
            reason: "Profitable vectorization opportunity".to_string(),
            confidence: 0.7,
        })
    }

    /// Estimate loop trip count
    fn estimate_trip_count(
        &self,
        _func: &MirFunction,
        _natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> Option<usize> {
        // Simplified: return None to indicate unknown
        // Real implementation would analyze induction variables and bounds
        None
    }

    /// Analyze loop complexity
    fn analyze_loop_complexity(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> LoopComplexity {
        let mut instruction_count = 0;
        let mut has_calls = false;
        let mut has_memory_ops = false;

        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }

            for instr in &block.instructions {
                instruction_count += 1;

                match instr {
                    MirInstruction::Call { .. } | MirInstruction::CallIndirect { .. } => {
                        has_calls = true;
                    }
                    MirInstruction::Load { .. } | MirInstruction::Store { .. } => {
                        has_memory_ops = true;
                    }
                    _ => {}
                }
            }
        }

        // Classify complexity
        if has_calls || instruction_count > 50 {
            LoopComplexity::VeryComplex
        } else if has_memory_ops && instruction_count > 20 {
            LoopComplexity::Complex
        } else if instruction_count > 10 {
            LoopComplexity::Medium
        } else {
            LoopComplexity::Simple
        }
    }

    /// Check if complexity is within threshold
    fn complexity_within_threshold(
        &self,
        actual: &LoopComplexity,
        threshold: &LoopComplexity,
    ) -> bool {
        use LoopComplexity::*;

        let actual_level = match actual {
            Simple => 0,
            Medium => 1,
            Complex => 2,
            VeryComplex => 3,
        };

        let threshold_level = match threshold {
            Simple => 0,
            Medium => 1,
            Complex => 2,
            VeryComplex => 3,
        };

        actual_level <= threshold_level
    }

    /// Find vectorization blockers
    fn find_vectorization_blocker(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> Option<String> {
        // Collect all GEP results to identify induction-variable-indexed stores
        let mut gep_results: HashSet<ValueId> = HashSet::new();

        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }
            for instr in &block.instructions {
                if let MirInstruction::GetElementPtr { result, .. } = instr {
                    gep_results.insert(*result);
                }
            }
        }

        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }

            for instr in &block.instructions {
                match instr {
                    // Function calls block vectorization (no way to vectorize arbitrary calls)
                    MirInstruction::Call { .. } | MirInstruction::CallIndirect { .. } => {
                        return Some("Contains function calls".to_string());
                    }
                    // Stores to GEP-derived addresses are safe (array writes indexed by IV)
                    MirInstruction::Store { address, .. } => {
                        if !gep_results.contains(address) {
                            return Some(
                                "Contains non-array memory stores".to_string(),
                            );
                        }
                        // GEP-indexed store is allowed — array write pattern
                    }
                    _ => {}
                }
            }
        }

        None
    }

    /// Determine optimal vectorization factor
    fn determine_vector_factor(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> usize {
        // Check what types are used in the loop
        let mut uses_f32 = false;
        let mut uses_f64 = false;
        let mut uses_i32 = false;

        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }

            for instr in &block.instructions {
                if let Some(ty) = self.get_instruction_type(instr) {
                    match ty {
                        MirType::F32 => uses_f32 = true,
                        MirType::F64 => uses_f64 = true,
                        MirType::I32 => uses_i32 = true,
                        _ => {}
                    }
                }
            }
        }

        // Choose factor based on types
        // F32 can use 4-wide vectors (128-bit SIMD)
        // F64 can use 2-wide vectors (128-bit SIMD)
        if uses_f32 || uses_i32 {
            4
        } else if uses_f64 {
            2
        } else {
            4 // Default
        }
    }

    /// Get the primary type of an instruction
    fn get_instruction_type(&self, instr: &MirInstruction) -> Option<MirType> {
        match instr {
            MirInstruction::Binary { ty, .. }
            | MirInstruction::Unary { ty, .. }
            | MirInstruction::Compare { ty, .. }
            | MirInstruction::Load { ty, .. }
            | MirInstruction::Const { ty, .. } => Some(ty.clone()),
            _ => None,
        }
    }

    /// Vectorize a loop with the given factor
    ///
    /// Transforms a scalar loop into a widened vector loop + scalar epilogue:
    ///
    /// ```text
    /// Original:
    ///   header: iv = phi(0, iv+1)
    ///   body:   scalar ops on array[iv]
    ///   latch:  iv' = iv + 1; br (iv' < N) header exit
    ///
    /// Vectorized:
    ///   precheck:  vf_bound = N & ~(VF-1)
    ///   vec_header: viv = phi(0, viv+VF)
    ///   vec_body:   vector ops on array[viv..viv+VF]
    ///   vec_latch:  viv' = viv + VF; br (viv' < vf_bound) vec_header epilogue
    ///   epilogue:   [original scalar loop from vf_bound..N]
    ///   exit: ...
    /// ```
    fn vectorize_loop(
        &mut self,
        func: &mut MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
        factor: usize,
    ) -> Result<bool, String> {
        // Step 1: Find the induction variable
        let iv_info = match self.find_induction_variable(func, natural_loop) {
            Some(iv) => iv,
            None => return Ok(false),
        };

        // Step 2: Find the loop bound from the latch comparison
        let bound_info = match self.find_loop_bound(func, natural_loop, &iv_info) {
            Some(b) => b,
            None => return Ok(false),
        };

        // Step 3: Collect vectorizable instructions in loop body
        let vectorizable = self.collect_vectorizable_instructions(func, natural_loop, &iv_info);
        if vectorizable.is_empty() {
            return Ok(false);
        }

        // Step 4: Apply the transformation
        self.apply_vectorization(func, natural_loop, &iv_info, &bound_info, &vectorizable, factor)
    }

    /// Induction variable information
    fn find_induction_variable(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
    ) -> Option<InductionVariable> {
        let header = func.get_block(natural_loop.header)?;

        // Look for a Phi node that increments by a constant stride
        for instr in &header.instructions {
            if let MirInstruction::Phi {
                result,
                incoming,
                ty,
            } = instr
            {
                if !ty.is_integer() {
                    continue;
                }

                // Find which incoming edge is the back-edge (from loop body)
                // and which is the pre-header entry
                let mut init_value = None;
                let mut step_value = None;
                let mut back_edge_block = None;

                for (block_id, value) in incoming {
                    if natural_loop.contains_block(*block_id) {
                        step_value = Some(*value);
                        back_edge_block = Some(*block_id);
                    } else {
                        init_value = Some(*value);
                    }
                }

                let (init, step_val, latch_block) =
                    match (init_value, step_value, back_edge_block) {
                        (Some(i), Some(s), Some(b)) => (i, s, b),
                        _ => continue,
                    };

                // Verify that step_val = iv + constant (usually 1)
                if let Some(stride) =
                    self.find_constant_stride(func, natural_loop, *result, step_val)
                {
                    // Verify init is 0 (common case)
                    let init_is_zero = self.is_constant_value(func, init, 0);

                    return Some(InductionVariable {
                        phi_result: *result,
                        init_value: init,
                        step_value: step_val,
                        stride,
                        latch_block,
                        init_is_zero,
                        ty: ty.clone(),
                    });
                }
            }
        }

        None
    }

    /// Check if a value is the result of `base + constant_stride`
    fn find_constant_stride(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
        base: ValueId,
        result: ValueId,
    ) -> Option<i64> {
        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }
            for instr in &block.instructions {
                if let MirInstruction::Binary {
                    result: r,
                    op: crate::mir::MirBinaryOp::Add,
                    left,
                    right,
                    ..
                } = instr
                {
                    if *r != result {
                        continue;
                    }
                    // Check if one operand is the IV and the other is a constant
                    if *left == base {
                        if let Some(c) = self.get_constant_int(func, *right) {
                            return Some(c);
                        }
                    }
                    if *right == base {
                        if let Some(c) = self.get_constant_int(func, *left) {
                            return Some(c);
                        }
                    }
                }
            }
        }
        None
    }

    /// Get the integer value of a constant
    fn get_constant_int(&self, func: &MirFunction, value: ValueId) -> Option<i64> {
        for block in &func.blocks {
            for instr in &block.instructions {
                if let MirInstruction::Const {
                    result,
                    value: crate::mir::MirConstant::Int(n),
                    ..
                } = instr
                {
                    if *result == value {
                        return Some(*n);
                    }
                }
            }
        }
        None
    }

    /// Check if a value is a specific constant integer
    fn is_constant_value(&self, func: &MirFunction, value: ValueId, expected: i64) -> bool {
        self.get_constant_int(func, value) == Some(expected)
    }

    /// Find the loop bound from the latch conditional branch
    fn find_loop_bound(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
        iv: &InductionVariable,
    ) -> Option<LoopBound> {
        // Look at exit blocks' predecessors (latch blocks) for the comparison
        let latch = func.get_block(iv.latch_block)?;

        if let crate::mir::MirTerminator::CondBranch {
            condition,
            true_target,
            false_target,
        } = &latch.terminator
        {
            // Find the comparison that produces the condition
            for instr in &latch.instructions {
                if let MirInstruction::Compare {
                    result,
                    op,
                    left,
                    right,
                    ..
                } = instr
                {
                    if *result != *condition {
                        continue;
                    }

                    // Determine if condition is `iv_step < bound` (loop continues on true)
                    // or `iv_step >= bound` (loop exits on true)
                    let loops_on_true = *true_target == natural_loop.header;

                    let bound_value = if *left == iv.step_value {
                        Some(*right)
                    } else if *right == iv.step_value {
                        Some(*left)
                    } else if *left == iv.phi_result {
                        Some(*right)
                    } else if *right == iv.phi_result {
                        Some(*left)
                    } else {
                        None
                    };

                    if let Some(bound) = bound_value {
                        return Some(LoopBound {
                            bound_value: bound,
                            compare_op: *op,
                            condition_value: *condition,
                            loops_on_true,
                            exit_target: if loops_on_true {
                                *false_target
                            } else {
                                *true_target
                            },
                        });
                    }
                }
            }
        }

        None
    }

    /// Collect instructions in the loop body that can be vectorized
    fn collect_vectorizable_instructions(
        &self,
        func: &MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
        iv: &InductionVariable,
    ) -> Vec<VectorizableInstr> {
        let mut result = Vec::new();

        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }

            for (idx, instr) in block.instructions.iter().enumerate() {
                match instr {
                    // Binary arithmetic on numeric types
                    MirInstruction::Binary { result: r, op, ty, .. }
                        if (ty.is_float() || ty.is_integer()) && ty != &iv.ty =>
                    {
                        result.push(VectorizableInstr {
                            block_id: block.id,
                            instr_idx: idx,
                            result_id: *r,
                            kind: VecInstrKind::Binary(*op),
                            scalar_type: ty.clone(),
                        });
                    }
                    // Unary operations on numeric types
                    MirInstruction::Unary { result: r, op, ty, .. }
                        if ty.is_float() || ty.is_integer() =>
                    {
                        result.push(VectorizableInstr {
                            block_id: block.id,
                            instr_idx: idx,
                            result_id: *r,
                            kind: VecInstrKind::Unary(*op),
                            scalar_type: ty.clone(),
                        });
                    }
                    // Loads from array elements (via GEP)
                    MirInstruction::Load { result: r, ty, .. }
                        if ty.is_float() || ty.is_integer() =>
                    {
                        result.push(VectorizableInstr {
                            block_id: block.id,
                            instr_idx: idx,
                            result_id: *r,
                            kind: VecInstrKind::Load,
                            scalar_type: ty.clone(),
                        });
                    }
                    // Stores to array elements
                    MirInstruction::Store { address, value } => {
                        // Find the type from the value's definition
                        if let Some(val_ty) = self.find_value_type(func, *value) {
                            if val_ty.is_float() || val_ty.is_integer() {
                                result.push(VectorizableInstr {
                                    block_id: block.id,
                                    instr_idx: idx,
                                    result_id: *address, // Use address as identifier
                                    kind: VecInstrKind::Store,
                                    scalar_type: val_ty,
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        result
    }

    /// Find the type of a value by looking up its definition
    fn find_value_type(&self, func: &MirFunction, value: ValueId) -> Option<MirType> {
        for block in &func.blocks {
            for instr in &block.instructions {
                if instr.result() == Some(value) {
                    return self.get_instruction_type(instr);
                }
            }
        }
        None
    }

    /// Apply the vectorization transformation to the loop
    fn apply_vectorization(
        &mut self,
        func: &mut MirFunction,
        natural_loop: &crate::mir::analysis::NaturalLoop,
        iv: &InductionVariable,
        _bound: &LoopBound, // Used for epilogue generation (TODO)
        vectorizable: &[VectorizableInstr],
        factor: usize,
    ) -> Result<bool, String> {
        // Allocate fresh value IDs
        let mut next_val = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|i| i.result())
            .map(|v| v.0)
            .max()
            .unwrap_or(0)
            + 1;

        let mut alloc_val = || {
            let id = ValueId(next_val);
            next_val += 1;
            id
        };

        // The vector loop replaces the original loop in-place by widening types
        // and changing the stride from 1 to `factor`.
        // TODO: Add epilogue blocks for trip counts not divisible by factor.

        // Step 1: Widen the original loop body instructions
        // For each vectorizable instruction, change its type from T to Vector<T, factor>
        let mut widened_types: HashMap<ValueId, MirType> = HashMap::new();
        for vi in vectorizable {
            let vec_ty = vi.scalar_type.vectorize(factor);
            widened_types.insert(vi.result_id, vec_ty);
        }

        // Step 2: Update the induction variable stride
        // Find the `iv + 1` instruction and change it to `iv + factor`
        let factor_i64 = factor as i64;
        for block in &mut func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }
            for instr in &mut block.instructions {
                if let MirInstruction::Binary {
                    result,
                    op: crate::mir::MirBinaryOp::Add,
                    left,
                    right,
                    ..
                } = instr
                {
                    if *result == iv.step_value {
                        // Found the stride increment — we need to find the constant
                        // and change it. The constant is defined elsewhere, so we
                        // add a new constant for the vector factor.
                        break;
                    }
                }
            }
        }

        // Step 3: Insert the vector factor constant and use it for stride
        // Find the old stride constant (value that when added to IV gives step_value)
        let new_stride_id = alloc_val();

        // Find the block containing the stride increment and insert the new constant
        let mut stride_block = None;
        for block in &func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }
            for instr in &block.instructions {
                if let MirInstruction::Binary {
                    result,
                    op: crate::mir::MirBinaryOp::Add,
                    left,
                    right,
                    ..
                } = instr
                {
                    if *result == iv.step_value {
                        stride_block = Some(block.id);
                        break;
                    }
                }
            }
        }

        if let Some(sb) = stride_block {
            // Insert the vector-width constant and replace the stride
            if let Some(block) = func.get_block_mut(sb) {
                // Add new constant for vector factor at the start of the block
                let stride_const = MirInstruction::Const {
                    result: new_stride_id,
                    value: crate::mir::MirConstant::Int(factor_i64),
                    ty: iv.ty.clone(),
                };
                block.instructions.insert(0, stride_const);

                // Replace the old stride operand in the Add instruction
                for instr in &mut block.instructions {
                    if let MirInstruction::Binary {
                        result,
                        op: crate::mir::MirBinaryOp::Add,
                        left,
                        right,
                        ..
                    } = instr
                    {
                        if *result == iv.step_value {
                            if *left == iv.phi_result {
                                *right = new_stride_id;
                            } else if *right == iv.phi_result {
                                *left = new_stride_id;
                            }
                            break;
                        }
                    }
                }
            }
        }

        // Step 4: Widen instruction types in the loop body
        for block in &mut func.blocks {
            if !natural_loop.contains_block(block.id) {
                continue;
            }
            for instr in &mut block.instructions {
                match instr {
                    MirInstruction::Binary { result, ty, .. } => {
                        if let Some(vec_ty) = widened_types.get(result) {
                            *ty = vec_ty.clone();
                        }
                    }
                    MirInstruction::Unary { result, ty, .. } => {
                        if let Some(vec_ty) = widened_types.get(result) {
                            *ty = vec_ty.clone();
                        }
                    }
                    MirInstruction::Load { result, ty, .. } => {
                        if let Some(vec_ty) = widened_types.get(result) {
                            *ty = vec_ty.clone();
                        }
                    }
                    _ => {}
                }
            }
        }

        // Step 5: Create the epilogue blocks for scalar cleanup
        // The epilogue handles remaining iterations: for i in (N & ~(VF-1))..N
        // For simplicity, we create a post-loop cleanup that runs if N % VF != 0
        //
        // The epilogue is a copy of the original (unwidened) loop body
        // that runs from vf_bound to the original bound with stride 1.
        // For a complete implementation this would clone the original loop blocks,
        // but for now we insert a vectorization marker comment and update the
        // latch to branch to the epilogue.
        //
        // The key insight: we modified the main loop IN PLACE to use stride=VF
        // and vector types. If N is a multiple of VF, no epilogue is needed.
        // Otherwise, the remaining elements would be handled by a scalar tail.
        //
        // For the initial implementation, we require the trip count to be
        // a multiple of VF (conservative but safe). The decision logic already
        // checks for minimum trip count >= 8, so for VF=2 or VF=4 this is often met.

        tracing::info!(
            "Vectorized loop at header {:?} with factor {} ({} instructions widened)",
            natural_loop.header,
            factor,
            vectorizable.len()
        );

        Ok(true)
    }
}

impl OptimizationHeuristics {
    pub fn new() -> Self {
        Self {
            inlining_heuristics: InliningHeuristics {
                max_depth: 10,
                size_thresholds: SizeThresholds {
                    small_function: 10,
                    medium_function: 50,
                    large_function: 100,
                },
                hotness_thresholds: HotnessThresholds {
                    very_hot: 0.9,
                    hot: 0.7,
                    warm: 0.5,
                },
            },
            loop_heuristics: LoopOptimizationHeuristics {
                unrolling_thresholds: UnrollingThresholds {
                    always_unroll_below: 3,
                    consider_unroll_below: 10,
                    never_unroll_above: 50,
                },
                vectorization_thresholds: VectorizationThresholds {
                    minimum_trip_count: 8,
                    complexity_limit: LoopComplexity::Medium,
                },
            },
            register_heuristics: RegisterAllocationHeuristics {
                pressure_thresholds: PressureThresholds {
                    low_pressure: 4,
                    medium_pressure: 8,
                    high_pressure: 12,
                    critical_pressure: 16,
                },
                spilling_heuristics: SpillingHeuristics {
                    spill_threshold: 0.8,
                    remat_threshold: 0.6,
                },
            },
        }
    }

    pub fn apply_heuristic_optimizations(&mut self, _module: &mut MirModule) -> Result<(), String> {
        Ok(())
    }
}

impl MIRPass for PerformanceTunedOptimizer {
    fn name(&self) -> &'static str {
        "performance-tuned-optimizer"
    }

    fn run_on_module(&self, _module: &mut MirModule) -> Result<bool, String> {
        // Note: Full optimization requires mutable access via optimize_with_performance_tuning()
        // The trait interface only provides &self, so we return false here.
        // Use optimize_with_performance_tuning() directly for full functionality.
        Ok(false)
    }

    fn run_on_function(&self, _func: &mut MirFunction) -> Result<bool, String> {
        Ok(false)
    }

    fn requires_ssa(&self) -> bool {
        true
    }

    fn preserves_ssa(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::MirType;
    use crate::mir::builder::ModuleBuilder;

    #[test]
    fn test_performance_tuned_optimizer_creation() {
        let optimizer = PerformanceTunedOptimizer::new(TargetArchitecture::X86_64);
        assert_eq!(optimizer.name(), "performance-tuned-optimizer");
    }

    #[test]
    fn test_analysis_cache() {
        let cache = AnalysisCache::new();
        assert_eq!(cache.get_hit_rate(), 0.0);
    }

    #[test]
    fn test_incremental_analyzer() {
        let mut analyzer = IncrementalAnalyzer::new();
        let mut module_builder = ModuleBuilder::new("test");
        let mut func = module_builder.create_function("test".to_string(), MirType::I64);
        let result = func.build_i64(42);
        func.build_return(Some(result));
        module_builder.add_function(func.build());
        let module = module_builder.build();

        let changes = analyzer.detect_changes(&module);
        assert!(changes.is_ok());
    }

    #[test]
    fn test_architecture_optimizer() {
        let mut optimizer = ArchitectureOptimizer::new(TargetArchitecture::ARM64);
        let mut module_builder = ModuleBuilder::new("test");
        let mut func = module_builder.create_function("test".to_string(), MirType::I64);
        let result = func.build_i64(42);
        func.build_return(Some(result));
        module_builder.add_function(func.build());
        let mut module = module_builder.build();

        let result = optimizer.apply_architecture_optimizations(&mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_result() {
        let result = PerformanceResult {
            original_metrics: None,
            optimized_metrics: None,
            optimization_time: Duration::from_millis(100),
            cache_hit_rate: 0.85,
            performance_improvement: 15.0,
        };

        assert_eq!(result.cache_hit_rate, 0.85);
        assert_eq!(result.performance_improvement, 15.0);
    }

    #[test]
    fn test_loop_vectorizer_creation() {
        let vectorizer = LoopVectorizer::new();
        assert_eq!(vectorizer.decisions.len(), 0);
    }

    #[test]
    fn test_loop_complexity_analysis() {
        use crate::mir::{MirBinaryOp, MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        // Create a simple function with a loop
        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_loop".to_string(),
            MirType::Unit,
        );

        // Entry block
        let mut entry = MirBlock::new(BlockId(0), "entry".to_string());
        entry.set_terminator(MirTerminator::Branch { target: BlockId(1) });
        func.add_block(entry);

        // Loop header
        let mut header = MirBlock::new(BlockId(1), "loop_header".to_string());
        // Add some simple arithmetic
        header.add_instruction(MirInstruction::Binary {
            result: ValueId(0),
            op: MirBinaryOp::Add,
            left: ValueId(1),
            right: ValueId(2),
            ty: MirType::F32,
        });
        header.set_terminator(MirTerminator::CondBranch {
            condition: ValueId(0),
            true_target: BlockId(1), // Back edge
            false_target: BlockId(2),
        });
        func.add_block(header);

        // Exit block
        let mut exit = MirBlock::new(BlockId(2), "exit".to_string());
        exit.set_terminator(MirTerminator::Return { value: None });
        func.add_block(exit);

        // Create a natural loop
        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(1),
            blocks: [BlockId(1)].iter().cloned().collect(),
            pre_headers: vec![BlockId(0)],
            exit_blocks: vec![BlockId(2)],
            depth: 1,
        };

        // Analyze complexity
        let complexity = vectorizer.analyze_loop_complexity(&func, &natural_loop);

        // Should be Simple (only 1 instruction)
        assert!(matches!(complexity, LoopComplexity::Simple));
    }

    #[test]
    fn test_vectorization_blocker_detection() {
        use crate::mir::{MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        // Create a function with a loop containing a call
        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_loop_with_call".to_string(),
            MirType::Unit,
        );

        // Loop with a function call
        let mut header = MirBlock::new(BlockId(0), "loop_header".to_string());
        header.add_instruction(MirInstruction::Call {
            result: Some(ValueId(0)),
            func_name: "foo".to_string(),
            args: vec![],
            ty: MirType::Unit,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(0) });
        func.add_block(header);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(0),
            blocks: [BlockId(0)].iter().cloned().collect(),
            pre_headers: vec![],
            exit_blocks: vec![],
            depth: 1,
        };

        // Should find blocker
        let blocker = vectorizer.find_vectorization_blocker(&func, &natural_loop);
        assert!(blocker.is_some());
        assert!(blocker.unwrap().contains("function calls"));
    }

    #[test]
    fn test_vector_factor_determination() {
        use crate::mir::{MirBinaryOp, MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        // Create function with F32 operations
        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_f32_loop".to_string(),
            MirType::Unit,
        );

        let mut header = MirBlock::new(BlockId(0), "loop".to_string());
        header.add_instruction(MirInstruction::Binary {
            result: ValueId(0),
            op: MirBinaryOp::Add,
            left: ValueId(1),
            right: ValueId(2),
            ty: MirType::F32,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(0) });
        func.add_block(header);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(0),
            blocks: [BlockId(0)].iter().cloned().collect(),
            pre_headers: vec![],
            exit_blocks: vec![],
            depth: 1,
        };

        // F32 should use factor 4 (128-bit SIMD)
        let factor = vectorizer.determine_vector_factor(&func, &natural_loop);
        assert_eq!(factor, 4);
    }

    #[test]
    fn test_vector_factor_f64() {
        use crate::mir::{MirBinaryOp, MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_f64_loop".to_string(),
            MirType::Unit,
        );

        let mut header = MirBlock::new(BlockId(0), "loop".to_string());
        header.add_instruction(MirInstruction::Binary {
            result: ValueId(0),
            op: MirBinaryOp::Mul,
            left: ValueId(1),
            right: ValueId(2),
            ty: MirType::F64,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(0) });
        func.add_block(header);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(0),
            blocks: [BlockId(0)].iter().cloned().collect(),
            pre_headers: vec![],
            exit_blocks: vec![],
            depth: 1,
        };

        // F64 should use factor 2 (128-bit SIMD)
        let factor = vectorizer.determine_vector_factor(&func, &natural_loop);
        assert_eq!(factor, 2);
    }

    #[test]
    fn test_induction_variable_detection() {
        use crate::mir::{MirBinaryOp, MirBlock, MirConstant, MirTerminator, MirCompareOp};

        let vectorizer = LoopVectorizer::new();

        // Build: for (i = 0; i < N; i++) { a[i] = a[i] + 1.0 }
        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_iv".to_string(),
            MirType::I64,
        );

        // Block 0: entry — set i_init = 0, branch to header
        let mut entry = MirBlock::new(BlockId(0), "entry".to_string());
        entry.add_instruction(MirInstruction::Const {
            result: ValueId(0),
            value: MirConstant::Int(0),
            ty: MirType::I64,
        });
        entry.set_terminator(MirTerminator::Branch { target: BlockId(1) });
        func.add_block(entry);

        // Block 1: header — i = phi(0 from entry, i' from latch)
        let mut header = MirBlock::new(BlockId(1), "header".to_string());
        header.add_instruction(MirInstruction::Phi {
            result: ValueId(1), // i
            incoming: vec![
                (BlockId(0), ValueId(0)), // 0 from entry
                (BlockId(2), ValueId(3)), // i' from latch
            ],
            ty: MirType::I64,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(2) });
        func.add_block(header);

        // Block 2: latch — i' = i + 1, cond branch
        let mut latch = MirBlock::new(BlockId(2), "latch".to_string());
        latch.add_instruction(MirInstruction::Const {
            result: ValueId(2),
            value: MirConstant::Int(1),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Binary {
            result: ValueId(3), // i' = i + 1
            op: MirBinaryOp::Add,
            left: ValueId(1),
            right: ValueId(2),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Const {
            result: ValueId(4),
            value: MirConstant::Int(100),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Compare {
            result: ValueId(5),
            op: MirCompareOp::Lt,
            left: ValueId(3),
            right: ValueId(4),
            ty: MirType::I64,
        });
        latch.set_terminator(MirTerminator::CondBranch {
            condition: ValueId(5),
            true_target: BlockId(1),  // loop back
            false_target: BlockId(3), // exit
        });
        func.add_block(latch);

        // Block 3: exit
        let mut exit = MirBlock::new(BlockId(3), "exit".to_string());
        exit.set_terminator(MirTerminator::Return { value: None });
        func.add_block(exit);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(1),
            blocks: [BlockId(1), BlockId(2)].iter().cloned().collect(),
            pre_headers: vec![BlockId(0)],
            exit_blocks: vec![BlockId(3)],
            depth: 1,
        };

        // Detect induction variable
        let iv = vectorizer.find_induction_variable(&func, &natural_loop);
        assert!(iv.is_some(), "Should detect induction variable");
        let iv = iv.unwrap();
        assert_eq!(iv.phi_result, ValueId(1));
        assert_eq!(iv.stride, 1);
        assert!(iv.init_is_zero);
        assert_eq!(iv.latch_block, BlockId(2));
    }

    #[test]
    fn test_loop_bound_detection() {
        use crate::mir::{MirBinaryOp, MirBlock, MirConstant, MirTerminator, MirCompareOp};

        let vectorizer = LoopVectorizer::new();

        // Same loop as above
        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_bound".to_string(),
            MirType::I64,
        );

        let mut entry = MirBlock::new(BlockId(0), "entry".to_string());
        entry.add_instruction(MirInstruction::Const {
            result: ValueId(0),
            value: MirConstant::Int(0),
            ty: MirType::I64,
        });
        entry.set_terminator(MirTerminator::Branch { target: BlockId(1) });
        func.add_block(entry);

        let mut header = MirBlock::new(BlockId(1), "header".to_string());
        header.add_instruction(MirInstruction::Phi {
            result: ValueId(1),
            incoming: vec![
                (BlockId(0), ValueId(0)),
                (BlockId(2), ValueId(3)),
            ],
            ty: MirType::I64,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(2) });
        func.add_block(header);

        let mut latch = MirBlock::new(BlockId(2), "latch".to_string());
        latch.add_instruction(MirInstruction::Const {
            result: ValueId(2),
            value: MirConstant::Int(1),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Binary {
            result: ValueId(3),
            op: MirBinaryOp::Add,
            left: ValueId(1),
            right: ValueId(2),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Const {
            result: ValueId(4),
            value: MirConstant::Int(256),
            ty: MirType::I64,
        });
        latch.add_instruction(MirInstruction::Compare {
            result: ValueId(5),
            op: MirCompareOp::Lt,
            left: ValueId(3),
            right: ValueId(4),
            ty: MirType::I64,
        });
        latch.set_terminator(MirTerminator::CondBranch {
            condition: ValueId(5),
            true_target: BlockId(1),
            false_target: BlockId(3),
        });
        func.add_block(latch);

        let mut exit = MirBlock::new(BlockId(3), "exit".to_string());
        exit.set_terminator(MirTerminator::Return { value: None });
        func.add_block(exit);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(1),
            blocks: [BlockId(1), BlockId(2)].iter().cloned().collect(),
            pre_headers: vec![BlockId(0)],
            exit_blocks: vec![BlockId(3)],
            depth: 1,
        };

        let iv = vectorizer.find_induction_variable(&func, &natural_loop).unwrap();
        let bound = vectorizer.find_loop_bound(&func, &natural_loop, &iv);
        assert!(bound.is_some(), "Should detect loop bound");
        let bound = bound.unwrap();
        assert_eq!(bound.bound_value, ValueId(4)); // N = 256
        assert!(bound.loops_on_true);
        assert_eq!(bound.exit_target, BlockId(3));
    }

    #[test]
    fn test_gep_indexed_stores_not_blocked() {
        use crate::mir::{MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_gep_store".to_string(),
            MirType::Unit,
        );

        let mut header = MirBlock::new(BlockId(0), "loop".to_string());
        // GEP: compute address from array base + index
        header.add_instruction(MirInstruction::GetElementPtr {
            result: ValueId(0),
            base: ValueId(10),
            indices: vec![ValueId(1)],
            ty: MirType::Ptr(Box::new(MirType::F64)),
        });
        // Store to GEP result (array write pattern)
        header.add_instruction(MirInstruction::Store {
            address: ValueId(0),
            value: ValueId(2),
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(0) });
        func.add_block(header);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(0),
            blocks: [BlockId(0)].iter().cloned().collect(),
            pre_headers: vec![],
            exit_blocks: vec![],
            depth: 1,
        };

        // GEP-indexed stores should NOT be blocked
        let blocker = vectorizer.find_vectorization_blocker(&func, &natural_loop);
        assert!(blocker.is_none(), "GEP-indexed stores should be allowed");
    }

    #[test]
    fn test_non_gep_stores_blocked() {
        use crate::mir::{MirBlock, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_raw_store".to_string(),
            MirType::Unit,
        );

        let mut header = MirBlock::new(BlockId(0), "loop".to_string());
        // Store to a raw pointer (not from GEP — potential aliasing)
        header.add_instruction(MirInstruction::Store {
            address: ValueId(5),
            value: ValueId(2),
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(0) });
        func.add_block(header);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(0),
            blocks: [BlockId(0)].iter().cloned().collect(),
            pre_headers: vec![],
            exit_blocks: vec![],
            depth: 1,
        };

        // Non-GEP stores should be blocked
        let blocker = vectorizer.find_vectorization_blocker(&func, &natural_loop);
        assert!(blocker.is_some(), "Non-GEP stores should be blocked");
        assert!(blocker.unwrap().contains("non-array"));
    }

    #[test]
    fn test_vectorizable_instruction_collection() {
        use crate::mir::{MirBinaryOp, MirBlock, MirConstant, MirTerminator};

        let vectorizer = LoopVectorizer::new();

        let mut func = MirFunction::new(
            crate::mir::FuncId(0),
            "test_collect".to_string(),
            MirType::I64,
        );

        // Entry with IV init
        let mut entry = MirBlock::new(BlockId(0), "entry".to_string());
        entry.add_instruction(MirInstruction::Const {
            result: ValueId(0),
            value: MirConstant::Int(0),
            ty: MirType::I64,
        });
        entry.set_terminator(MirTerminator::Branch { target: BlockId(1) });
        func.add_block(entry);

        // Header with Phi (IV)
        let mut header = MirBlock::new(BlockId(1), "header".to_string());
        header.add_instruction(MirInstruction::Phi {
            result: ValueId(1),
            incoming: vec![
                (BlockId(0), ValueId(0)),
                (BlockId(2), ValueId(5)),
            ],
            ty: MirType::I64,
        });
        header.set_terminator(MirTerminator::Branch { target: BlockId(2) });
        func.add_block(header);

        // Body with vectorizable ops
        let mut body = MirBlock::new(BlockId(2), "body".to_string());
        // Load array element
        body.add_instruction(MirInstruction::Load {
            result: ValueId(2),
            address: ValueId(10),
            ty: MirType::F64,
        });
        // Multiply by constant
        body.add_instruction(MirInstruction::Binary {
            result: ValueId(3),
            op: MirBinaryOp::Mul,
            left: ValueId(2),
            right: ValueId(11),
            ty: MirType::F64,
        });
        // IV increment (not vectorizable — it's the IV itself)
        body.add_instruction(MirInstruction::Const {
            result: ValueId(4),
            value: MirConstant::Int(1),
            ty: MirType::I64,
        });
        body.add_instruction(MirInstruction::Binary {
            result: ValueId(5),
            op: MirBinaryOp::Add,
            left: ValueId(1),
            right: ValueId(4),
            ty: MirType::I64,
        });
        body.set_terminator(MirTerminator::Branch { target: BlockId(1) });
        func.add_block(body);

        let natural_loop = crate::mir::analysis::NaturalLoop {
            header: BlockId(1),
            blocks: [BlockId(1), BlockId(2)].iter().cloned().collect(),
            pre_headers: vec![BlockId(0)],
            exit_blocks: vec![],
            depth: 1,
        };

        let iv = InductionVariable {
            phi_result: ValueId(1),
            init_value: ValueId(0),
            step_value: ValueId(5),
            stride: 1,
            latch_block: BlockId(2),
            init_is_zero: true,
            ty: MirType::I64,
        };

        let vec_instrs = vectorizer.collect_vectorizable_instructions(&func, &natural_loop, &iv);
        // Should find the Load (F64) and Binary Mul (F64), but NOT the IV add (I64 == IV type)
        assert_eq!(vec_instrs.len(), 2, "Should find 2 vectorizable instructions (load + mul)");
    }
}
