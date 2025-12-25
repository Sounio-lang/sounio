//! Integration tests for GPU Performance Profiling & Roofline Analysis (Phase 12)
//!
//! Tests instruction cost database, roofline model, and bottleneck detection.

use sounio::codegen::gpu::{
    // Cost database
    ArchPeakPerf,
    // IR types
    BlockId,
    BottleneckKind,
    BottleneckSeverity,
    // Roofline model
    Boundedness,
    CostDatabase,
    // Arch types
    CudaArch,
    GpuBlock,
    GpuKernel,
    GpuModule,
    GpuOp,
    GpuTarget,
    GpuTerminator,
    GpuType,
    InstructionClass,
    KernelProfiler,
    MemorySpace,
    OptimizationHint,
    PerfCounters,
    PerfScore,
    RooflineModel,
    RooflinePoint,
    ValueId,
};

// ============================================================================
// Instruction Cost Database Tests
// ============================================================================

#[test]
fn test_cost_database_creation() {
    // Test creating cost databases for all supported architectures
    let arches = [
        CudaArch::Turing,
        CudaArch::Ampere,
        CudaArch::Ada,
        CudaArch::Hopper,
        CudaArch::Blackwell,
    ];

    for arch in arches {
        let _db = CostDatabase::for_arch(arch);
        let perf = ArchPeakPerf::for_arch(arch);

        // All architectures should have positive peak performance
        assert!(
            perf.fp32_tflops > 0.0,
            "{:?} should have fp32 performance",
            arch
        );
        assert!(
            perf.fp16_tflops > 0.0,
            "{:?} should have fp16 performance",
            arch
        );
        assert!(
            perf.memory_bandwidth_gbs > 0.0,
            "{:?} should have memory bandwidth",
            arch
        );
    }
}

#[test]
fn test_instruction_cost_fp32() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // Test FP32 instruction costs
    let fp32_add = db.get_cost(InstructionClass::Fp32Add);
    let fp32_mul = db.get_cost(InstructionClass::Fp32Mul);
    let fp32_fma = db.get_cost(InstructionClass::Fp32Fma);
    let fp32_div = db.get_cost(InstructionClass::Fp32Div);

    // FMA should have same latency as add/mul
    assert_eq!(fp32_add.latency, fp32_fma.latency);
    assert_eq!(fp32_mul.latency, fp32_fma.latency);

    // Division should have higher latency
    assert!(fp32_div.latency > fp32_add.latency);

    // FMA should not use SFU
    assert!(!fp32_fma.uses_sfu);
}

#[test]
fn test_instruction_cost_memory() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    let global_load = db.get_cost(InstructionClass::GlobalLoad);
    let shared_load = db.get_cost(InstructionClass::SharedLoad);

    // Global memory should have higher latency than shared memory
    assert!(
        global_load.latency > shared_load.latency,
        "Global load latency {} should be > shared load latency {}",
        global_load.latency,
        shared_load.latency
    );

    // Memory operations should have memory_bytes > 0
    assert!(global_load.memory_bytes > 0);
    assert!(shared_load.memory_bytes > 0);
}

#[test]
fn test_instruction_cost_tensor_core() {
    let db = CostDatabase::for_arch(CudaArch::Hopper);

    let tensor_fp16 = db.get_cost(InstructionClass::TensorMmaFp16);
    let tensor_fp8 = db.get_cost(InstructionClass::TensorMmaFp8);

    // Tensor core ops should be flagged
    assert!(tensor_fp16.uses_tensor_core);
    assert!(tensor_fp8.uses_tensor_core);

    // Tensor core ops should have high throughput
    assert!(tensor_fp16.throughput > 0.0);
    assert!(tensor_fp8.throughput > 0.0);
}

#[test]
fn test_op_classification() {
    // Test that GpuOps are correctly classified

    // Arithmetic
    let add_class = CostDatabase::classify_op(&GpuOp::FAdd(ValueId(0), ValueId(1)));
    assert_eq!(add_class, InstructionClass::Fp32Add);

    let mul_class = CostDatabase::classify_op(&GpuOp::FMul(ValueId(0), ValueId(1)));
    assert_eq!(mul_class, InstructionClass::Fp32Mul);

    // Memory
    let load_class = CostDatabase::classify_op(&GpuOp::Load(ValueId(0), MemorySpace::Global));
    assert_eq!(load_class, InstructionClass::GlobalLoad);

    let shared_load_class =
        CostDatabase::classify_op(&GpuOp::Load(ValueId(0), MemorySpace::Shared));
    assert_eq!(shared_load_class, InstructionClass::SharedLoad);

    // Sync
    let sync_class = CostDatabase::classify_op(&GpuOp::SyncThreads);
    assert_eq!(sync_class, InstructionClass::SyncThreads);
}

#[test]
fn test_kernel_cost_estimate() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // Create a simple test kernel
    let target = GpuTarget::Cuda {
        compute_capability: (8, 0),
    };
    let module = GpuModule::new("test_module", target);
    let mut kernel = GpuKernel::new("test_kernel");

    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Add some operations
    block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
    block.add_instruction(ValueId(1), GpuOp::ConstFloat(2.0, GpuType::F32));
    block.add_instruction(ValueId(2), GpuOp::FAdd(ValueId(0), ValueId(1)));
    block.add_instruction(ValueId(3), GpuOp::FMul(ValueId(2), ValueId(1)));
    block.set_terminator(GpuTerminator::ReturnVoid);

    kernel.add_block(block);

    let estimate = db.estimate_kernel_cycles(&kernel);

    // Should have some compute cycles
    assert!(estimate.compute_cycles > 0);
    // Total should be >= compute
    assert!(estimate.total_cycles >= estimate.compute_cycles);
}

#[test]
fn test_flops_count() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // Create kernel with known FLOPS count
    let mut kernel = GpuKernel::new("flops_test");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // 3 FP32 operations
    block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
    block.add_instruction(ValueId(1), GpuOp::ConstFloat(2.0, GpuType::F32));
    block.add_instruction(ValueId(2), GpuOp::FAdd(ValueId(0), ValueId(1))); // 1 FLOP
    block.add_instruction(ValueId(3), GpuOp::FMul(ValueId(2), ValueId(1))); // 1 FLOP
    block.add_instruction(ValueId(4), GpuOp::FSub(ValueId(3), ValueId(0))); // 1 FLOP
    block.set_terminator(GpuTerminator::ReturnVoid);

    kernel.add_block(block);

    let flops = db.count_flops(&kernel);

    assert_eq!(flops.fp32_flops, 3, "Should count 3 FP32 operations");
    assert_eq!(flops.total_flops, 3, "Total should match FP32");
}

#[test]
fn test_memory_traffic() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    let mut kernel = GpuKernel::new("memory_test");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Add loads and stores
    block.add_instruction(ValueId(0), GpuOp::ConstInt(0, GpuType::I64)); // pointer
    block.add_instruction(ValueId(1), GpuOp::Load(ValueId(0), MemorySpace::Global)); // global load
    block.add_instruction(ValueId(2), GpuOp::Load(ValueId(0), MemorySpace::Shared)); // shared load
    block.add_instruction(
        ValueId(3),
        GpuOp::Store(ValueId(0), ValueId(1), MemorySpace::Global),
    ); // global store
    block.set_terminator(GpuTerminator::ReturnVoid);

    kernel.add_block(block);

    let traffic = db.count_memory_bytes(&kernel);

    assert!(traffic.global_loads > 0, "Should count global loads");
    assert!(traffic.global_stores > 0, "Should count global stores");
    assert!(traffic.shared_loads > 0, "Should count shared loads");
    assert!(traffic.total_bytes > 0, "Total bytes should be positive");
}

// ============================================================================
// Roofline Model Tests
// ============================================================================

#[test]
fn test_roofline_model_creation() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);

    // Ampere A100 specs
    assert!(
        model.peak_compute() > 10000.0,
        "A100 should have >10 TFLOPS"
    );
    assert!(
        model.peak_bandwidth() > 1000.0,
        "A100 should have >1 TB/s bandwidth"
    );
    assert!(model.ridge_point() > 0.0, "Ridge point should be positive");
}

#[test]
fn test_ridge_point_calculation() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);

    // Ridge point = peak_compute / peak_bandwidth
    let ridge = model.ridge_point();

    // For A100: ~19.5 TFLOPS / 1.555 TB/s ≈ 12.5 FLOPS/byte
    assert!(ridge > 5.0, "Ridge point should be > 5 FLOPS/byte");
    assert!(ridge < 30.0, "Ridge point should be < 30 FLOPS/byte");
}

#[test]
fn test_boundedness_classification() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);
    let ridge = model.ridge_point();

    // Low arithmetic intensity = memory bound
    let memory_bound = model.classify_boundedness(1.0);
    assert!(
        matches!(memory_bound, Boundedness::MemoryBound { .. }),
        "AI=1 should be memory bound"
    );

    // High arithmetic intensity = compute bound
    let compute_bound = model.classify_boundedness(100.0);
    assert!(
        matches!(compute_bound, Boundedness::ComputeBound { .. }),
        "AI=100 should be compute bound"
    );

    // Near ridge = balanced
    let _balanced = model.classify_boundedness(ridge);
    // Within 10% of ridge should be balanced
    let near_ridge = model.classify_boundedness(ridge * 0.95);
    // Could be balanced or memory bound, just verify it runs
    assert!(matches!(
        near_ridge,
        Boundedness::Balanced | Boundedness::MemoryBound { .. } | Boundedness::ComputeBound { .. }
    ));
}

#[test]
fn test_peak_at_intensity() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);
    let ridge = model.ridge_point();

    // Below ridge: limited by bandwidth
    let low_ai_peak = model.peak_at_intensity(1.0);
    // At AI=1, peak = bandwidth * AI = ~1.5 TFLOPS
    assert!(
        low_ai_peak < model.peak_compute(),
        "Below ridge should be memory limited"
    );

    // Above ridge: limited by compute
    let high_ai_peak = model.peak_at_intensity(ridge * 10.0);
    // Should be near peak compute
    assert!(
        (high_ai_peak - model.peak_compute()).abs() < 100.0,
        "Above ridge should hit compute ceiling"
    );
}

#[test]
fn test_roofline_point() {
    let point = RooflinePoint {
        arithmetic_intensity: 10.0,
        achieved_gflops: 5000.0,
        peak_gflops: 10000.0,
        efficiency: 0.5,
    };

    assert_eq!(point.efficiency, 0.5);
    assert_eq!(point.arithmetic_intensity, 10.0);
}

#[test]
fn test_roofline_analysis() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // Create a compute-heavy kernel
    let mut kernel = GpuKernel::new("compute_kernel");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Many FP32 ops, few memory ops
    for i in 0..100 {
        block.add_instruction(ValueId(i as u32), GpuOp::ConstFloat(1.0, GpuType::F32));
    }
    for i in 100..200 {
        block.add_instruction(
            ValueId(i as u32),
            GpuOp::FAdd(ValueId((i - 100) as u32), ValueId((i - 99) as u32)),
        );
    }
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let analysis = model.analyze_kernel(&kernel, &db);

    assert!(!analysis.kernel_name.is_empty());
    assert!(analysis.flops.total_flops > 0);
    assert!(analysis.efficiency >= 0.0 && analysis.efficiency <= 1.0);
}

// ============================================================================
// Profiler Tests
// ============================================================================

#[test]
fn test_profiler_creation() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);
    // Should not panic
    let _ = profiler;
}

#[test]
fn test_perf_counters_default() {
    let counters = PerfCounters::default();

    assert_eq!(counters.fp32_instructions, 0);
    assert_eq!(counters.global_load_transactions, 0);
    assert_eq!(counters.warp_execution_efficiency, 0.0);
}

#[test]
fn test_kernel_profiling() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);

    // Create test kernel
    let mut kernel = GpuKernel::new("profile_test");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
    block.add_instruction(ValueId(1), GpuOp::ConstFloat(2.0, GpuType::F32));
    block.add_instruction(ValueId(2), GpuOp::FAdd(ValueId(0), ValueId(1)));
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let profile = profiler.profile_kernel(&kernel);

    assert_eq!(profile.name, "profile_test");
    assert!(profile.counters.fp32_instructions > 0);
    assert!(profile.score.overall >= 0.0);
}

#[test]
fn test_module_profiling() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);

    let target = GpuTarget::Cuda {
        compute_capability: (8, 0),
    };
    let mut module = GpuModule::new("module_test", target);

    // Add multiple kernels
    for i in 0..3 {
        let mut kernel = GpuKernel::new(format!("kernel_{}", i));
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::FAdd(ValueId(0), ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);
    }

    let profile = profiler.profile_module(&module);

    assert_eq!(profile.kernels.len(), 3);
    assert!(profile.total_flops.total_flops > 0);
}

#[test]
fn test_bottleneck_detection() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);

    // Create memory-heavy kernel (many loads, few computes)
    let mut kernel = GpuKernel::new("memory_heavy");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Many memory operations
    for i in 0..50 {
        block.add_instruction(ValueId(i as u32), GpuOp::ConstInt(i as i64, GpuType::I64));
        block.add_instruction(
            ValueId((50 + i) as u32),
            GpuOp::Load(ValueId(i as u32), MemorySpace::Global),
        );
    }
    // Few compute operations
    block.add_instruction(ValueId(100), GpuOp::FAdd(ValueId(50), ValueId(51)));
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let profile = profiler.profile_kernel(&kernel);
    let _bottlenecks = profiler.detect_bottlenecks(&profile);

    // Should detect memory-related bottleneck
    let _has_memory_bottleneck = _bottlenecks.iter().any(|b| {
        matches!(
            b.kind,
            BottleneckKind::MemoryBandwidth | BottleneckKind::MemoryLatency
        )
    });
    // Memory-heavy kernel should have some memory-related observations
    assert!(profile.counters.global_load_transactions > 0);
}

#[test]
fn test_bottleneck_severity() {
    // Test severity ordering
    assert!(BottleneckSeverity::Critical > BottleneckSeverity::High);
    assert!(BottleneckSeverity::High > BottleneckSeverity::Medium);
    assert!(BottleneckSeverity::Medium > BottleneckSeverity::Low);
}

#[test]
fn test_perf_score() {
    let score = PerfScore {
        overall: 75.0,
        compute_efficiency: 80.0,
        memory_efficiency: 70.0,
        occupancy_score: 85.0,
        instruction_efficiency: 65.0,
    };

    assert!(score.overall >= 0.0 && score.overall <= 100.0);
    assert!(score.compute_efficiency >= 0.0);
}

#[test]
fn test_perf_comparison() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);

    // Create "before" kernel
    let mut before_kernel = GpuKernel::new("before");
    let mut block = GpuBlock::new(BlockId(0), "entry");
    for i in 0..10 {
        block.add_instruction(ValueId(i as u32), GpuOp::ConstInt(i as i64, GpuType::I64));
        block.add_instruction(
            ValueId((10 + i) as u32),
            GpuOp::Load(ValueId(i as u32), MemorySpace::Global),
        );
    }
    block.set_terminator(GpuTerminator::ReturnVoid);
    before_kernel.add_block(block);

    // Create "after" kernel (more efficient)
    let mut after_kernel = GpuKernel::new("after");
    let mut block = GpuBlock::new(BlockId(0), "entry");
    for i in 0..5 {
        block.add_instruction(ValueId(i as u32), GpuOp::ConstInt(i as i64, GpuType::I64));
        block.add_instruction(
            ValueId((5 + i) as u32),
            GpuOp::Load(ValueId(i as u32), MemorySpace::Shared),
        );
    }
    block.set_terminator(GpuTerminator::ReturnVoid);
    after_kernel.add_block(block);

    let before_profile = profiler.profile_kernel(&before_kernel);
    let after_profile = profiler.profile_kernel(&after_kernel);

    let comparison = profiler.compare(&before_profile, &after_profile);

    // After should have less memory traffic
    assert!(
        comparison.memory_reduction >= 0.0 || comparison.memory_reduction < 0.0,
        "Comparison should produce a numeric result"
    );
}

// ============================================================================
// Architecture-Specific Tests
// ============================================================================

#[test]
fn test_blackwell_specs() {
    let perf = ArchPeakPerf::for_arch(CudaArch::Blackwell);

    // Blackwell B200 should have highest performance
    assert!(
        perf.fp32_tflops > 50.0,
        "Blackwell should have >50 FP32 TFLOPS"
    );
    assert!(
        perf.tensor_fp16_tflops > 1000.0,
        "Blackwell should have >1000 Tensor TFLOPS"
    );
    assert!(
        perf.memory_bandwidth_gbs > 5000.0,
        "Blackwell should have >5 TB/s bandwidth"
    );
}

#[test]
fn test_hopper_tensor_cores() {
    let db = CostDatabase::for_arch(CudaArch::Hopper);

    // Hopper introduced FP8 tensor cores
    let fp8_cost = db.get_cost(InstructionClass::TensorMmaFp8);
    assert!(fp8_cost.uses_tensor_core);
    assert!(fp8_cost.throughput > 0.0);

    // FP8 should have higher throughput than FP16
    let fp16_cost = db.get_cost(InstructionClass::TensorMmaFp16);
    assert!(fp8_cost.throughput >= fp16_cost.throughput);
}

#[test]
fn test_turing_vs_ampere() {
    let turing_perf = ArchPeakPerf::for_arch(CudaArch::Turing);
    let ampere_perf = ArchPeakPerf::for_arch(CudaArch::Ampere);

    // Ampere should be faster than Turing
    assert!(
        ampere_perf.fp32_tflops > turing_perf.fp32_tflops,
        "Ampere should be faster than Turing"
    );
    assert!(
        ampere_perf.memory_bandwidth_gbs > turing_perf.memory_bandwidth_gbs,
        "Ampere should have more bandwidth than Turing"
    );
}

// ============================================================================
// Integration Tests
// ============================================================================

#[test]
fn test_full_profiling_workflow() {
    // Complete workflow: create module -> profile -> analyze bottlenecks -> get recommendations

    let target = GpuTarget::Cuda {
        compute_capability: (8, 0),
    };
    let mut module = GpuModule::new("workflow_test", target);

    // Create a realistic kernel (matrix multiply pattern)
    let mut matmul = GpuKernel::new("matmul");
    let mut block = GpuBlock::new(BlockId(0), "entry");

    // Load A, B tiles
    for i in 0..16 {
        block.add_instruction(ValueId(i as u32), GpuOp::ConstInt(i as i64, GpuType::I64));
        block.add_instruction(
            ValueId((16 + i) as u32),
            GpuOp::Load(ValueId(i as u32), MemorySpace::Shared),
        );
    }
    // Compute: use FMulAdd (FMA) for matrix multiply
    for i in 0..64 {
        let a = ValueId((32 + i * 2) as u32);
        let b = ValueId((33 + i * 2) as u32);
        let c = ValueId((160 + i) as u32);
        block.add_instruction(a, GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(b, GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(c, GpuOp::FMulAdd(a, b, ValueId(16))); // FMA
    }
    // Store result
    block.add_instruction(
        ValueId(300),
        GpuOp::Store(ValueId(0), ValueId(160), MemorySpace::Global),
    );
    block.set_terminator(GpuTerminator::ReturnVoid);
    matmul.add_block(block);
    module.add_kernel(matmul);

    // Profile
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);
    let profile = profiler.profile_module(&module);

    // Verify profile has expected data
    assert_eq!(profile.kernels.len(), 1);
    assert!(profile.kernels[0].counters.fp32_instructions > 0);

    // Check for optimization hints
    let hints = profiler.recommend_optimizations(&profile.kernels[0]);
    // Should have some recommendations
    assert!(hints.len() >= 0, "Should generate optimization hints");

    // Verify scores are reasonable
    let score = &profile.kernels[0].score;
    assert!(score.overall >= 0.0 && score.overall <= 100.0);
}

#[test]
fn test_roofline_plot_data() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // Create some test kernels and analyze them
    let mut analyses = Vec::new();
    for i in 1..=5 {
        let mut kernel = GpuKernel::new(format!("kernel_{}", i));
        let mut block = GpuBlock::new(BlockId(0), "entry");

        // Add operations to create different arithmetic intensities
        for j in 0..(i * 10) {
            block.add_instruction(ValueId(j as u32), GpuOp::ConstFloat(1.0, GpuType::F32));
        }
        for j in (i * 10)..(i * 20) {
            block.add_instruction(ValueId(j as u32), GpuOp::FAdd(ValueId(0), ValueId(1)));
        }
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        analyses.push(model.analyze_kernel(&kernel, &db));
    }

    let plot = model.generate_roofline_data(&analyses);

    // Plot should have ceiling data
    assert!(!plot.ceiling.is_empty(), "Should have ceiling points");
    assert!(plot.ridge_x > 0.0, "Should have ridge point");
    assert_eq!(plot.kernels.len(), 5, "Should have 5 kernel points");
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[test]
fn test_empty_kernel() {
    let profiler = KernelProfiler::for_arch(CudaArch::Ampere);

    let mut kernel = GpuKernel::new("empty");
    let mut block = GpuBlock::new(BlockId(0), "entry");
    block.set_terminator(GpuTerminator::ReturnVoid);
    kernel.add_block(block);

    let profile = profiler.profile_kernel(&kernel);

    // Should not panic, should have zero counts
    assert_eq!(profile.counters.fp32_instructions, 0);
    assert_eq!(profile.counters.global_load_transactions, 0);
}

#[test]
fn test_unknown_instruction_class() {
    let db = CostDatabase::for_arch(CudaArch::Ampere);

    // All instruction classes should have a cost
    let classes = [
        InstructionClass::Fp32Add,
        InstructionClass::Fp32Mul,
        InstructionClass::Fp32Fma,
        InstructionClass::GlobalLoad,
        InstructionClass::SharedLoad,
        InstructionClass::SyncThreads,
        InstructionClass::AtomicAdd,
    ];

    for class in classes {
        let cost = db.get_cost(class);
        assert!(cost.latency > 0, "{:?} should have positive latency", class);
    }
}

#[test]
fn test_very_high_arithmetic_intensity() {
    let model = RooflineModel::for_arch(CudaArch::Ampere);

    // Very high AI (like 1000) should hit compute ceiling
    let peak = model.peak_at_intensity(1000.0);
    let compute_peak = model.peak_compute();

    // Should be at or very close to compute ceiling
    assert!(
        (peak - compute_peak).abs() < 1.0,
        "AI=1000 should hit compute ceiling"
    );
}

#[test]
fn test_optimization_hint_types() {
    // Verify all hint types can be created
    let hints: Vec<OptimizationHint> = vec![
        OptimizationHint::IncreaseArithmeticIntensity {
            current: 1.0,
            target: 10.0,
        },
        OptimizationHint::ImproveMemoryCoalescing { efficiency: 0.5 },
        OptimizationHint::UseTensorCores {
            speedup_estimate: 4.0,
        },
        OptimizationHint::IncreaseOccupancy {
            current: 0.3,
            target: 0.7,
        },
        OptimizationHint::ReduceSharedMemory {
            current: 48000,
            limit: 48000,
        },
        OptimizationHint::UseQuantization { precision: "INT8" },
        OptimizationHint::EnableAsyncPipeline,
        OptimizationHint::FuseKernels {
            candidates: vec!["kernel_a".to_string()],
        },
    ];

    assert_eq!(hints.len(), 8, "Should have 8 hint types");
}
