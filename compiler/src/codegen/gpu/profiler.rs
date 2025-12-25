//! Performance Profiler & Bottleneck Detection (Phase 12)
//!
//! Provides static performance analysis for GPU kernels including
//! simulated hardware counters, bottleneck detection, and optimization
//! recommendations.

use super::costs::{CostDatabase, FlopsCount, KernelCostEstimate, MemoryTraffic};
use super::diagnostics::GpuIrLocation;
use super::ir::{CudaArch, GpuKernel, GpuModule, GpuOp, MemorySpace};
use super::roofline::{OptimizationHint, RooflineAnalysis, RooflineModel};

// ============================================================================
// Performance Counters
// ============================================================================

/// Simulated hardware performance counters
#[derive(Debug, Clone, Default)]
pub struct PerfCounters {
    // Instruction counts
    /// FP32 floating-point instructions
    pub fp32_instructions: u64,
    /// FP16 floating-point instructions
    pub fp16_instructions: u64,
    /// FP64 floating-point instructions
    pub fp64_instructions: u64,
    /// Integer instructions
    pub int_instructions: u64,
    /// Tensor core instructions
    pub tensor_instructions: u64,
    /// Special function unit instructions
    pub sfu_instructions: u64,

    // Memory transactions
    /// Global memory load transactions
    pub global_load_transactions: u64,
    /// Global memory store transactions
    pub global_store_transactions: u64,
    /// Shared memory load transactions
    pub shared_load_transactions: u64,
    /// Shared memory store transactions
    pub shared_store_transactions: u64,
    /// L2 cache read transactions
    pub l2_read_transactions: u64,
    /// L2 cache write transactions
    pub l2_write_transactions: u64,

    // Efficiency metrics (0.0 - 1.0)
    /// Warp execution efficiency
    pub warp_execution_efficiency: f64,
    /// Global load efficiency (coalescing)
    pub global_load_efficiency: f64,
    /// Global store efficiency (coalescing)
    pub global_store_efficiency: f64,
    /// Shared memory bank conflicts
    pub shared_bank_conflicts: u64,

    // Occupancy
    /// Achieved occupancy (0.0 - 1.0)
    pub achieved_occupancy: f64,
    /// Theoretical maximum occupancy
    pub theoretical_occupancy: f64,

    // Stall reasons (cycle counts)
    /// Stalls due to memory dependency
    pub stall_memory_dependency: u64,
    /// Stalls due to execution dependency
    pub stall_execution_dependency: u64,
    /// Stalls due to synchronization
    pub stall_sync: u64,
    /// Stalls due to other reasons
    pub stall_other: u64,

    // Branch statistics
    /// Total branch instructions
    pub branch_instructions: u64,
    /// Divergent branch instructions
    pub divergent_branches: u64,
}

impl PerfCounters {
    /// Calculate total instruction count
    pub fn total_instructions(&self) -> u64 {
        self.fp32_instructions
            + self.fp16_instructions
            + self.fp64_instructions
            + self.int_instructions
            + self.tensor_instructions
            + self.sfu_instructions
    }

    /// Calculate total memory transactions
    pub fn total_memory_transactions(&self) -> u64 {
        self.global_load_transactions
            + self.global_store_transactions
            + self.shared_load_transactions
            + self.shared_store_transactions
    }

    /// Calculate total stall cycles
    pub fn total_stalls(&self) -> u64 {
        self.stall_memory_dependency
            + self.stall_execution_dependency
            + self.stall_sync
            + self.stall_other
    }

    /// Calculate branch divergence ratio
    pub fn branch_divergence_ratio(&self) -> f64 {
        if self.branch_instructions == 0 {
            0.0
        } else {
            self.divergent_branches as f64 / self.branch_instructions as f64
        }
    }
}

// ============================================================================
// Kernel Profiler
// ============================================================================

/// Performance profiler for GPU kernels
pub struct KernelProfiler {
    costs: CostDatabase,
    roofline: RooflineModel,
    arch: CudaArch,
}

impl KernelProfiler {
    /// Create profiler for a specific architecture
    pub fn for_arch(arch: CudaArch) -> Self {
        let costs = CostDatabase::for_arch(arch);
        let roofline = RooflineModel::for_arch(arch);
        Self {
            costs,
            roofline,
            arch,
        }
    }

    /// Get the architecture this profiler targets
    pub fn arch(&self) -> CudaArch {
        self.arch
    }

    /// Profile a single kernel (static analysis)
    pub fn profile_kernel(&self, kernel: &GpuKernel) -> KernelPerfProfile {
        let counters = self.collect_counters(kernel);
        let cost_estimate = self.costs.estimate_kernel_cycles(kernel);
        let roofline = self.roofline.analyze_kernel(kernel, &self.costs);
        let bottlenecks = self.detect_bottlenecks_internal(&counters, &cost_estimate, &roofline);
        let score = self.compute_score(&counters, &roofline, &bottlenecks);

        KernelPerfProfile {
            name: kernel.name.clone(),
            counters,
            cost_estimate,
            roofline,
            bottlenecks,
            score,
        }
    }

    /// Profile an entire module
    pub fn profile_module(&self, module: &GpuModule) -> ModulePerfProfile {
        let mut kernel_profiles = Vec::new();
        let mut total_flops = FlopsCount::default();
        let mut total_memory = MemoryTraffic::default();

        for kernel in module.kernels.values() {
            let profile = self.profile_kernel(kernel);

            // Accumulate totals
            total_flops.fp32_flops += profile.roofline.flops.fp32_flops;
            total_flops.fp16_flops += profile.roofline.flops.fp16_flops;
            total_flops.fp64_flops += profile.roofline.flops.fp64_flops;
            total_flops.int_ops += profile.roofline.flops.int_ops;
            total_flops.tensor_flops += profile.roofline.flops.tensor_flops;
            total_flops.total_flops += profile.roofline.flops.total_flops;

            total_memory.global_loads += profile.roofline.memory.global_loads;
            total_memory.global_stores += profile.roofline.memory.global_stores;
            total_memory.shared_loads += profile.roofline.memory.shared_loads;
            total_memory.shared_stores += profile.roofline.memory.shared_stores;
            total_memory.total_bytes += profile.roofline.memory.total_bytes;

            kernel_profiles.push(profile);
        }

        // Identify hotspots (kernels with most cycles)
        let mut hotspots: Vec<_> = kernel_profiles
            .iter()
            .map(|p| (p.name.clone(), p.cost_estimate.total_cycles as f64))
            .collect();
        hotspots.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        hotspots.truncate(5);

        // Build optimization priority list
        let mut optimization_priority: Vec<_> = kernel_profiles
            .iter()
            .filter(|p| !p.bottlenecks.is_empty())
            .map(|p| {
                let hints: Vec<_> = p.roofline.recommendations.clone();
                (p.name.clone(), hints)
            })
            .collect();
        optimization_priority.sort_by(|a, b| b.1.len().cmp(&a.1.len()));

        // Compute overall score
        let overall_score = if kernel_profiles.is_empty() {
            PerfScore::default()
        } else {
            let avg_compute = kernel_profiles
                .iter()
                .map(|p| p.score.compute_efficiency)
                .sum::<f64>()
                / kernel_profiles.len() as f64;
            let avg_memory = kernel_profiles
                .iter()
                .map(|p| p.score.memory_efficiency)
                .sum::<f64>()
                / kernel_profiles.len() as f64;
            let avg_occupancy = kernel_profiles
                .iter()
                .map(|p| p.score.occupancy_score)
                .sum::<f64>()
                / kernel_profiles.len() as f64;
            let avg_instruction = kernel_profiles
                .iter()
                .map(|p| p.score.instruction_efficiency)
                .sum::<f64>()
                / kernel_profiles.len() as f64;

            PerfScore {
                overall: (avg_compute + avg_memory + avg_occupancy + avg_instruction) / 4.0 * 100.0,
                compute_efficiency: avg_compute,
                memory_efficiency: avg_memory,
                occupancy_score: avg_occupancy,
                instruction_efficiency: avg_instruction,
            }
        };

        ModulePerfProfile {
            kernels: kernel_profiles,
            total_flops,
            total_memory_bytes: total_memory.total_bytes,
            overall_score,
            hotspots,
            optimization_priority,
        }
    }

    /// Collect simulated performance counters for a kernel
    fn collect_counters(&self, kernel: &GpuKernel) -> PerfCounters {
        let mut counters = PerfCounters::default();

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                self.classify_instruction(op, &mut counters);
            }
        }

        // Estimate efficiencies based on access patterns
        counters.warp_execution_efficiency = self.estimate_warp_efficiency(kernel);
        counters.global_load_efficiency = self.estimate_coalescing_efficiency(kernel, true);
        counters.global_store_efficiency = self.estimate_coalescing_efficiency(kernel, false);
        counters.achieved_occupancy = self.estimate_occupancy(kernel);
        counters.theoretical_occupancy = 1.0; // Assume max possible

        // Estimate stalls based on instruction mix
        let total_mem = counters.global_load_transactions + counters.global_store_transactions;
        counters.stall_memory_dependency = total_mem * 10; // Rough estimate
        counters.stall_sync = counters.int_instructions / 100; // Sync overhead estimate

        counters
    }

    /// Classify an instruction and update counters
    fn classify_instruction(&self, op: &GpuOp, counters: &mut PerfCounters) {
        use super::costs::InstructionClass;

        let class = CostDatabase::classify_op(op);
        let cost = self.costs.get_cost(class);

        match class {
            InstructionClass::Fp32Add
            | InstructionClass::Fp32Mul
            | InstructionClass::Fp32Fma
            | InstructionClass::Fp32Div
            | InstructionClass::Fp32Sqrt
            | InstructionClass::Fp32Rsqrt => {
                counters.fp32_instructions += 1;
            }

            InstructionClass::Fp16Add | InstructionClass::Fp16Mul | InstructionClass::Fp16Fma => {
                counters.fp16_instructions += 1;
            }

            InstructionClass::Bf16Add | InstructionClass::Bf16Mul | InstructionClass::Bf16Fma => {
                counters.fp16_instructions += 1; // Count as FP16
            }

            InstructionClass::Fp64Add
            | InstructionClass::Fp64Mul
            | InstructionClass::Fp64Fma
            | InstructionClass::Fp64Div => {
                counters.fp64_instructions += 1;
            }

            InstructionClass::Fp32Sin
            | InstructionClass::Fp32Cos
            | InstructionClass::Fp32Exp
            | InstructionClass::Fp32Log
            | InstructionClass::Fp32Pow
            | InstructionClass::Fp32Tanh => {
                counters.sfu_instructions += 1;
            }

            InstructionClass::IntAdd
            | InstructionClass::IntMul
            | InstructionClass::IntDiv
            | InstructionClass::IntMod
            | InstructionClass::IntBitwise
            | InstructionClass::IntShift
            | InstructionClass::IntCompare => {
                counters.int_instructions += 1;
            }

            InstructionClass::GlobalLoad => {
                counters.global_load_transactions += 1;
                counters.l2_read_transactions += 1;
            }

            InstructionClass::GlobalStore => {
                counters.global_store_transactions += 1;
                counters.l2_write_transactions += 1;
            }

            InstructionClass::SharedLoad => {
                counters.shared_load_transactions += 1;
            }

            InstructionClass::SharedStore => {
                counters.shared_store_transactions += 1;
            }

            InstructionClass::TensorMmaFp16
            | InstructionClass::TensorMmaBf16
            | InstructionClass::TensorMmaFp8
            | InstructionClass::TensorMmaInt8
            | InstructionClass::TensorMmaFp4 => {
                counters.tensor_instructions += 1;
            }

            InstructionClass::Branch | InstructionClass::PredicatedExec => {
                counters.branch_instructions += 1;
            }

            _ => {
                counters.int_instructions += 1; // Default to int
            }
        }

        // Track SFU usage
        if cost.uses_sfu {
            counters.sfu_instructions += 1;
        }

        // Track tensor core usage
        if cost.uses_tensor_core {
            counters.tensor_instructions += 1;
        }
    }

    /// Estimate warp execution efficiency
    fn estimate_warp_efficiency(&self, kernel: &GpuKernel) -> f64 {
        let mut branch_count = 0;
        let mut total_ops = 0;

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                total_ops += 1;
                if matches!(op, GpuOp::Select { .. }) {
                    branch_count += 1;
                }
            }
        }

        // More branches = lower efficiency (simplified model)
        let branch_ratio = branch_count as f64 / total_ops.max(1) as f64;
        (1.0 - branch_ratio * 0.5).max(0.5)
    }

    /// Estimate memory coalescing efficiency
    fn estimate_coalescing_efficiency(&self, kernel: &GpuKernel, is_load: bool) -> f64 {
        let mut coalesced = 0;
        let mut total = 0;

        for block in &kernel.blocks {
            for (_, op) in &block.instructions {
                let is_target = if is_load {
                    matches!(op, GpuOp::Load(_, MemorySpace::Global))
                } else {
                    matches!(op, GpuOp::Store(_, _, MemorySpace::Global))
                };

                if is_target {
                    total += 1;
                    // Assume contiguous accesses are coalesced (simplified)
                    coalesced += 1;
                }
            }
        }

        if total == 0 {
            1.0
        } else {
            coalesced as f64 / total as f64
        }
    }

    /// Estimate achieved occupancy
    fn estimate_occupancy(&self, kernel: &GpuKernel) -> f64 {
        // Use the existing autotune occupancy calculation if available
        // For now, return a reasonable default based on kernel complexity
        let total_ops: usize = kernel.blocks.iter().map(|b| b.instructions.len()).sum();

        if total_ops < 100 {
            0.9 // Small kernels typically have high occupancy
        } else if total_ops < 1000 {
            0.7
        } else {
            0.5 // Large kernels may have register pressure
        }
    }

    /// Detect performance bottlenecks
    fn detect_bottlenecks_internal(
        &self,
        counters: &PerfCounters,
        cost: &KernelCostEstimate,
        roofline: &RooflineAnalysis,
    ) -> Vec<Bottleneck> {
        let mut bottlenecks = Vec::new();

        // Memory bandwidth bottleneck
        if roofline.boundedness.is_memory_bound() && counters.global_load_efficiency < 0.7 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::MemoryBandwidth,
                severity: BottleneckSeverity::High,
                description: format!(
                    "Memory-bound with {:.0}% load efficiency",
                    counters.global_load_efficiency * 100.0
                ),
                location: None,
                impact_estimate: 0.3,
            });
        }

        // Uncoalesced access detection
        if counters.global_load_efficiency < 0.5 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::UncoalescedAccess,
                severity: BottleneckSeverity::High,
                description: format!(
                    "Poor memory coalescing ({:.0}% efficiency)",
                    counters.global_load_efficiency * 100.0
                ),
                location: None,
                impact_estimate: 0.4,
            });
        }

        // Low occupancy
        if counters.achieved_occupancy < 0.5 * counters.theoretical_occupancy {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::Occupancy,
                severity: BottleneckSeverity::Medium,
                description: format!(
                    "Low occupancy ({:.0}%)",
                    counters.achieved_occupancy * 100.0
                ),
                location: None,
                impact_estimate: 0.2,
            });
        }

        // Bank conflicts
        if counters.shared_bank_conflicts > counters.shared_load_transactions / 10 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::SharedMemoryBankConflict,
                severity: BottleneckSeverity::Medium,
                description: format!("{} bank conflicts detected", counters.shared_bank_conflicts),
                location: None,
                impact_estimate: 0.15,
            });
        }

        // Warp divergence
        if counters.warp_execution_efficiency < 0.8 && counters.branch_instructions > 0 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::WarpDivergence,
                severity: BottleneckSeverity::Medium,
                description: format!(
                    "Warp divergence ({:.0}% efficiency)",
                    counters.warp_execution_efficiency * 100.0
                ),
                location: None,
                impact_estimate: 0.2,
            });
        }

        // Synchronization overhead
        if cost.sync_cycles > cost.compute_cycles / 4 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::SyncOverhead,
                severity: BottleneckSeverity::Low,
                description: format!(
                    "{} sync cycles ({:.0}% of compute)",
                    cost.sync_cycles,
                    cost.sync_cycles as f64 / cost.compute_cycles.max(1) as f64 * 100.0
                ),
                location: None,
                impact_estimate: 0.1,
            });
        }

        // Memory latency (high stalls)
        if counters.stall_memory_dependency > counters.total_instructions() / 2 {
            bottlenecks.push(Bottleneck {
                kind: BottleneckKind::MemoryLatency,
                severity: BottleneckSeverity::High,
                description: "High memory latency stalls".to_string(),
                location: None,
                impact_estimate: 0.35,
            });
        }

        // Sort by severity
        bottlenecks.sort_by(|a, b| b.severity.cmp(&a.severity));

        bottlenecks
    }

    /// Compute performance score
    fn compute_score(
        &self,
        counters: &PerfCounters,
        roofline: &RooflineAnalysis,
        bottlenecks: &[Bottleneck],
    ) -> PerfScore {
        // Compute efficiency (0-1)
        let compute_efficiency = if counters.total_instructions() == 0 {
            1.0
        } else {
            let compute_ratio = (counters.fp32_instructions
                + counters.fp16_instructions
                + counters.tensor_instructions) as f64
                / counters.total_instructions() as f64;
            compute_ratio.min(1.0)
        };

        // Memory efficiency (0-1)
        let memory_efficiency =
            (counters.global_load_efficiency + counters.global_store_efficiency) / 2.0;

        // Occupancy score (0-1)
        let occupancy_score =
            counters.achieved_occupancy / counters.theoretical_occupancy.max(0.01);

        // Instruction efficiency (based on bottleneck impact)
        let bottleneck_penalty: f64 = bottlenecks.iter().map(|b| b.impact_estimate).sum();
        let instruction_efficiency = (1.0 - bottleneck_penalty).max(0.0);

        // Overall score (0-100)
        let overall =
            (compute_efficiency + memory_efficiency + occupancy_score + instruction_efficiency)
                / 4.0
                * 100.0;

        PerfScore {
            overall: overall.min(100.0),
            compute_efficiency,
            memory_efficiency,
            occupancy_score,
            instruction_efficiency,
        }
    }

    /// Detect bottlenecks for a kernel profile
    pub fn detect_bottlenecks(&self, profile: &KernelPerfProfile) -> Vec<Bottleneck> {
        self.detect_bottlenecks_internal(
            &profile.counters,
            &profile.cost_estimate,
            &profile.roofline,
        )
    }

    /// Generate optimization recommendations
    pub fn recommend_optimizations(&self, profile: &KernelPerfProfile) -> Vec<OptimizationHint> {
        let mut hints = profile.roofline.recommendations.clone();

        // Add hints based on bottlenecks
        for bottleneck in &profile.bottlenecks {
            match bottleneck.kind {
                BottleneckKind::UncoalescedAccess => {
                    hints.push(OptimizationHint::UseVectorizedAccess {
                        current_width: 4,
                        target_width: 16,
                    });
                }
                BottleneckKind::Occupancy => {
                    hints.push(OptimizationHint::ReduceRegisterPressure {
                        current: 64,
                        target: 32,
                    });
                }
                BottleneckKind::SharedMemoryBankConflict => {
                    hints.push(OptimizationHint::ReduceSharedMemory {
                        current: 48 * 1024,
                        limit: 32 * 1024,
                    });
                }
                _ => {}
            }
        }

        // Deduplicate hints
        hints.sort_by(|a, b| format!("{:?}", a).cmp(&format!("{:?}", b)));
        hints.dedup_by(|a, b| std::mem::discriminant(a) == std::mem::discriminant(b));

        hints
    }

    /// Compare two kernel profiles (before/after optimization)
    pub fn compare(&self, before: &KernelPerfProfile, after: &KernelPerfProfile) -> PerfComparison {
        let speedup = if after.cost_estimate.total_cycles > 0 {
            before.cost_estimate.total_cycles as f64 / after.cost_estimate.total_cycles as f64
        } else {
            1.0
        };

        let flops_improvement = if before.roofline.flops.total_flops > 0 {
            after.roofline.flops.total_flops as f64 / before.roofline.flops.total_flops as f64
        } else {
            1.0
        };

        let memory_reduction = if before.roofline.memory.total_bytes > 0 {
            1.0 - (after.roofline.memory.total_bytes as f64
                / before.roofline.memory.total_bytes as f64)
        } else {
            0.0
        };

        let efficiency_delta = after.roofline.efficiency - before.roofline.efficiency;

        let mut changes = Vec::new();
        if speedup > 1.1 {
            changes.push(format!("{:.1}x speedup", speedup));
        }
        if memory_reduction > 0.1 {
            changes.push(format!("{:.0}% memory reduction", memory_reduction * 100.0));
        }
        if efficiency_delta > 0.05 {
            changes.push(format!("+{:.0}% efficiency", efficiency_delta * 100.0));
        }

        PerfComparison {
            speedup,
            flops_improvement,
            memory_reduction,
            efficiency_delta,
            changes,
        }
    }
}

// ============================================================================
// Profile Types
// ============================================================================

/// Complete performance profile for a kernel
#[derive(Debug, Clone)]
pub struct KernelPerfProfile {
    /// Kernel name
    pub name: String,
    /// Simulated performance counters
    pub counters: PerfCounters,
    /// Cost estimate
    pub cost_estimate: KernelCostEstimate,
    /// Roofline analysis
    pub roofline: RooflineAnalysis,
    /// Detected bottlenecks
    pub bottlenecks: Vec<Bottleneck>,
    /// Performance score
    pub score: PerfScore,
}

impl KernelPerfProfile {
    /// Check if the kernel has critical bottlenecks
    pub fn has_critical_bottlenecks(&self) -> bool {
        self.bottlenecks
            .iter()
            .any(|b| b.severity == BottleneckSeverity::Critical)
    }

    /// Get the primary bottleneck (if any)
    pub fn primary_bottleneck(&self) -> Option<&Bottleneck> {
        self.bottlenecks.first()
    }

    /// Get optimization priority (0 = highest priority)
    pub fn optimization_priority(&self) -> u32 {
        let severity_score: u32 = self
            .bottlenecks
            .iter()
            .map(|b| b.severity.as_priority())
            .sum();
        let efficiency_penalty = ((1.0 - self.roofline.efficiency) * 100.0) as u32;
        severity_score + efficiency_penalty
    }
}

/// Module-level performance profile
#[derive(Debug, Clone)]
pub struct ModulePerfProfile {
    /// Per-kernel profiles
    pub kernels: Vec<KernelPerfProfile>,
    /// Total FLOPS across all kernels
    pub total_flops: FlopsCount,
    /// Total memory traffic
    pub total_memory_bytes: u64,
    /// Overall performance score
    pub overall_score: PerfScore,
    /// Hotspot kernels (name, cycles)
    pub hotspots: Vec<(String, f64)>,
    /// Optimization priority (kernel name, hints)
    pub optimization_priority: Vec<(String, Vec<OptimizationHint>)>,
}

impl ModulePerfProfile {
    /// Get the number of kernels profiled
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Get kernels with bottlenecks
    pub fn kernels_with_bottlenecks(&self) -> Vec<&KernelPerfProfile> {
        self.kernels
            .iter()
            .filter(|k| !k.bottlenecks.is_empty())
            .collect()
    }

    /// Get the top N hotspot kernel names
    pub fn top_hotspots(&self, n: usize) -> Vec<&str> {
        self.hotspots
            .iter()
            .take(n)
            .map(|(name, _)| name.as_str())
            .collect()
    }
}

// ============================================================================
// Bottleneck Types
// ============================================================================

/// Performance bottleneck
#[derive(Debug, Clone)]
pub struct Bottleneck {
    /// Type of bottleneck
    pub kind: BottleneckKind,
    /// Severity level
    pub severity: BottleneckSeverity,
    /// Human-readable description
    pub description: String,
    /// Location in the GPU IR (if known)
    pub location: Option<GpuIrLocation>,
    /// Estimated performance impact (0.0 - 1.0)
    pub impact_estimate: f64,
}

/// Types of performance bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BottleneckKind {
    /// Limited by memory bandwidth
    MemoryBandwidth,
    /// Limited by compute throughput
    ComputeThroughput,
    /// Limited by memory latency
    MemoryLatency,
    /// Limited by occupancy
    Occupancy,
    /// Shared memory bank conflicts
    SharedMemoryBankConflict,
    /// Uncoalesced global memory access
    UncoalescedAccess,
    /// Warp divergence due to branches
    WarpDivergence,
    /// Synchronization overhead
    SyncOverhead,
    /// Register spilling to local memory
    RegisterSpilling,
    /// Instruction cache misses
    InstructionCacheMiss,
}

impl BottleneckKind {
    /// Get a short name for the bottleneck kind
    pub fn short_name(&self) -> &'static str {
        match self {
            BottleneckKind::MemoryBandwidth => "mem_bw",
            BottleneckKind::ComputeThroughput => "compute",
            BottleneckKind::MemoryLatency => "mem_lat",
            BottleneckKind::Occupancy => "occupancy",
            BottleneckKind::SharedMemoryBankConflict => "bank_conflict",
            BottleneckKind::UncoalescedAccess => "uncoalesced",
            BottleneckKind::WarpDivergence => "divergence",
            BottleneckKind::SyncOverhead => "sync",
            BottleneckKind::RegisterSpilling => "reg_spill",
            BottleneckKind::InstructionCacheMiss => "icache_miss",
        }
    }
}

/// Severity level for bottlenecks
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BottleneckSeverity {
    /// Minor impact (<10%)
    Low,
    /// Moderate impact (10-25%)
    Medium,
    /// Significant impact (25-50%)
    High,
    /// Major impact (>50%)
    Critical,
}

impl BottleneckSeverity {
    /// Get priority value (lower = more important)
    pub fn as_priority(&self) -> u32 {
        match self {
            BottleneckSeverity::Critical => 100,
            BottleneckSeverity::High => 50,
            BottleneckSeverity::Medium => 20,
            BottleneckSeverity::Low => 5,
        }
    }
}

// ============================================================================
// Performance Score
// ============================================================================

/// Overall performance score (0-100)
#[derive(Debug, Clone, Default)]
pub struct PerfScore {
    /// Overall score (0-100)
    pub overall: f64,
    /// Compute efficiency (0-1)
    pub compute_efficiency: f64,
    /// Memory efficiency (0-1)
    pub memory_efficiency: f64,
    /// Occupancy score (0-1)
    pub occupancy_score: f64,
    /// Instruction efficiency (0-1)
    pub instruction_efficiency: f64,
}

impl PerfScore {
    /// Check if the score indicates good performance
    pub fn is_good(&self) -> bool {
        self.overall >= 70.0
    }

    /// Check if the score indicates poor performance
    pub fn is_poor(&self) -> bool {
        self.overall < 50.0
    }

    /// Get a letter grade
    pub fn grade(&self) -> char {
        if self.overall >= 90.0 {
            'A'
        } else if self.overall >= 80.0 {
            'B'
        } else if self.overall >= 70.0 {
            'C'
        } else if self.overall >= 60.0 {
            'D'
        } else {
            'F'
        }
    }
}

// ============================================================================
// Comparison Types
// ============================================================================

/// Before/after performance comparison
#[derive(Debug, Clone)]
pub struct PerfComparison {
    /// Speedup factor (>1 = faster)
    pub speedup: f64,
    /// FLOPS improvement factor
    pub flops_improvement: f64,
    /// Memory reduction (0-1, >0 = less memory)
    pub memory_reduction: f64,
    /// Efficiency delta (-1 to +1)
    pub efficiency_delta: f64,
    /// Human-readable changes
    pub changes: Vec<String>,
}

impl PerfComparison {
    /// Check if the optimization was beneficial
    pub fn is_improvement(&self) -> bool {
        self.speedup > 1.0 || self.efficiency_delta > 0.0
    }

    /// Get a summary string
    pub fn summary(&self) -> String {
        if self.changes.is_empty() {
            "No significant changes".to_string()
        } else {
            self.changes.join(", ")
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = KernelProfiler::for_arch(CudaArch::Ampere);
        assert_eq!(profiler.arch(), CudaArch::Ampere);
    }

    #[test]
    fn test_perf_counters() {
        let mut counters = PerfCounters::default();
        counters.fp32_instructions = 100;
        counters.int_instructions = 50;
        counters.global_load_transactions = 20;

        assert_eq!(counters.total_instructions(), 150);
        assert_eq!(counters.total_memory_transactions(), 20);
    }

    #[test]
    fn test_bottleneck_severity_ordering() {
        assert!(BottleneckSeverity::Critical > BottleneckSeverity::High);
        assert!(BottleneckSeverity::High > BottleneckSeverity::Medium);
        assert!(BottleneckSeverity::Medium > BottleneckSeverity::Low);
    }

    #[test]
    fn test_perf_score_grades() {
        let excellent = PerfScore {
            overall: 95.0,
            ..Default::default()
        };
        assert_eq!(excellent.grade(), 'A');

        let poor = PerfScore {
            overall: 45.0,
            ..Default::default()
        };
        assert_eq!(poor.grade(), 'F');
    }

    #[test]
    fn test_perf_comparison() {
        let comparison = PerfComparison {
            speedup: 2.0,
            flops_improvement: 1.0,
            memory_reduction: 0.3,
            efficiency_delta: 0.1,
            changes: vec!["2.0x speedup".to_string()],
        };

        assert!(comparison.is_improvement());
        assert!(comparison.summary().contains("speedup"));
    }

    #[test]
    fn test_bottleneck_kind_short_names() {
        assert_eq!(BottleneckKind::MemoryBandwidth.short_name(), "mem_bw");
        assert_eq!(BottleneckKind::WarpDivergence.short_name(), "divergence");
    }
}
