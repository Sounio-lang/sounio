//! Integration Tests for GPU Optimization Pipeline
//!
//! Tests the full optimization pipeline including fusion, auto-tuning,
//! and PTX code generation.

use rustc_hash::FxHashMap;
use sounio::codegen::gpu::{
    autotune::{AutoTuneConfig, AutoTuner, KernelAnalyzer, OccupancyCalculator},
    fusion::{FusionAnalysis, FusionConfig, FusionCostModel},
    graph::build_graph_from_module,
    ir::{
        BlockId, CudaArch, GpuBlock, GpuKernel, GpuModule, GpuOp, GpuParam, GpuTarget,
        GpuTerminator, GpuType, MemorySpace, ValueId,
    },
    optimizer::{GpuOptimizer, OptimizerConfig},
    ptx::PtxCodegen,
};

/// Helper to create a simple elementwise kernel
fn make_elementwise_kernel(name: &str, num_ops: usize) -> GpuKernel {
    let mut instructions = Vec::new();

    // tid = threadIdx.x
    instructions.push((ValueId(0), GpuOp::ThreadIdX));

    // Load, compute, store pattern
    for i in 0..num_ops {
        let base_id = (i * 3 + 1) as u32;
        // load
        instructions.push((
            ValueId(base_id),
            GpuOp::Load(ValueId(0), MemorySpace::Global),
        ));
        // mul by 2.0
        instructions.push((
            ValueId(base_id + 1),
            GpuOp::FMul(ValueId(base_id), ValueId(base_id)),
        ));
        // store
        instructions.push((
            ValueId(base_id + 2),
            GpuOp::Store(ValueId(0), ValueId(base_id + 1), MemorySpace::Global),
        ));
    }

    GpuKernel {
        name: name.to_string(),
        params: vec![
            GpuParam {
                name: "input".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "output".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "n".to_string(),
                ty: GpuType::I32,
                space: MemorySpace::Local,
                restrict: false,
            },
        ],
        shared_memory: vec![],
        blocks: vec![GpuBlock {
            id: BlockId(0),
            label: "entry".to_string(),
            instructions,
            terminator: GpuTerminator::ReturnVoid,
        }],
        entry: BlockId(0),
        max_threads: None,
        shared_mem_size: 0,
    }
}

/// Helper to create a reduction kernel
fn make_reduction_kernel(name: &str) -> GpuKernel {
    let instructions = vec![
        (ValueId(0), GpuOp::ThreadIdX),
        (ValueId(1), GpuOp::BlockIdX),
        (ValueId(2), GpuOp::BlockDimX),
        // Simplified reduction - actual impl would have more ops
        (ValueId(3), GpuOp::Load(ValueId(0), MemorySpace::Global)),
        (ValueId(4), GpuOp::FAdd(ValueId(3), ValueId(3))),
        (
            ValueId(5),
            GpuOp::Store(ValueId(0), ValueId(4), MemorySpace::Global),
        ),
    ];

    GpuKernel {
        name: name.to_string(),
        params: vec![
            GpuParam {
                name: "data".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "result".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
        ],
        shared_memory: vec![],
        blocks: vec![GpuBlock {
            id: BlockId(0),
            label: "entry".to_string(),
            instructions,
            terminator: GpuTerminator::ReturnVoid,
        }],
        entry: BlockId(0),
        max_threads: Some(256),
        shared_mem_size: 1024, // 1KB shared memory
    }
}

/// Create a test module with multiple kernels
fn make_test_module(kernels: Vec<GpuKernel>) -> GpuModule {
    let mut kernel_map = FxHashMap::default();
    for k in kernels {
        kernel_map.insert(k.name.clone(), k);
    }

    GpuModule {
        name: "test_module".to_string(),
        target: GpuTarget::Cuda {
            compute_capability: (8, 0), // Ampere
        },
        kernels: kernel_map,
        device_functions: FxHashMap::default(),
        constants: vec![],
    }
}

// =============================================================================
// Integration Tests
// =============================================================================

#[test]
fn test_fusion_analysis_finds_candidates() {
    let module = make_test_module(vec![
        make_elementwise_kernel("kernel_a", 5),
        make_elementwise_kernel("kernel_b", 5),
        make_elementwise_kernel("kernel_c", 5),
    ]);

    let graph = build_graph_from_module(&module);
    let cost_model = FusionCostModel::new(module.target);
    let mut analysis = FusionAnalysis::with_config(FusionConfig::default(), cost_model);

    let plan = analysis.analyze(&module, &graph);

    // Plan should exist (even if empty for independent kernels)
    assert!(plan.groups.len() <= module.kernels.len());
}

#[test]
fn test_autotune_elementwise_kernel() {
    let kernel = make_elementwise_kernel("elementwise", 10);
    let profile = KernelAnalyzer::analyze(&kernel);

    // Kernel should have a valid pattern detected
    // The pattern depends on the analysis algorithm
    let _ = profile.pattern; // Just verify it has a pattern

    let tuner = AutoTuner::new(AutoTuneConfig::default());
    let tuned = tuner.tune_kernel(&kernel);

    // Block shape should be reasonable for elementwise
    assert!(tuned.block_shape.total_threads() >= 64);
    assert!(tuned.block_shape.total_threads() <= 1024);
}

#[test]
fn test_autotune_reduction_kernel() {
    let kernel = make_reduction_kernel("reduce_sum");
    let _profile = KernelAnalyzer::analyze(&kernel);

    let tuner = AutoTuner::new(AutoTuneConfig::default());
    let tuned = tuner.tune_kernel(&kernel);

    // Reduction kernels typically use 128-512 threads
    assert!(tuned.block_shape.total_threads() >= 64);
}

#[test]
fn test_occupancy_calculator_architectures() {
    let archs = [
        CudaArch::Turing,
        CudaArch::Ampere,
        CudaArch::Hopper,
        CudaArch::Blackwell,
    ];

    for arch in archs {
        let calc = OccupancyCalculator::from_cuda_arch(arch);

        // Test typical kernel configuration
        let info = calc.calculate_occupancy(256, 32, 0);

        // Occupancy should be positive
        assert!(
            info.occupancy > 0.0,
            "Occupancy should be > 0 for {:?}, got {}",
            arch,
            info.occupancy
        );
        assert!(
            info.occupancy <= 1.0,
            "Occupancy should be <= 1.0 for {:?}, got {}",
            arch,
            info.occupancy
        );
    }
}

#[test]
fn test_optimizer_full_pipeline() {
    let module = make_test_module(vec![
        make_elementwise_kernel("vec_add", 3),
        make_reduction_kernel("reduce"),
    ]);

    let optimizer = GpuOptimizer::new();
    let (optimized, report) = optimizer.optimize(&module).expect("Optimization failed");

    // Module should still have kernels
    assert!(!optimized.kernels.is_empty());

    // Auto-tune should have run
    assert!(report.autotune.enabled);
    assert!(!report.tuned_configs.is_empty());

    // Each kernel should have a tuned config
    assert_eq!(report.tuned_configs.len(), optimized.kernels.len());
}

#[test]
fn test_optimizer_aggressive_config() {
    let module = make_test_module(vec![make_elementwise_kernel("kernel", 5)]);

    let optimizer = GpuOptimizer::with_config(OptimizerConfig::aggressive());
    let (_, report) = optimizer.optimize(&module).expect("Optimization failed");

    // Aggressive config enables async pipeline analysis
    assert!(report.async_pipeline.enabled);
}

#[test]
fn test_optimizer_minimal_config() {
    let module = make_test_module(vec![make_elementwise_kernel("kernel", 5)]);

    let optimizer = GpuOptimizer::with_config(OptimizerConfig::minimal());
    let (_, report) = optimizer.optimize(&module).expect("Optimization failed");

    // Minimal config skips fusion
    assert!(!report.fusion.enabled);
    // But still does auto-tuning
    assert!(report.autotune.enabled);
}

#[test]
fn test_optimizer_quick_tune() {
    let module = make_test_module(vec![
        make_elementwise_kernel("k1", 3),
        make_elementwise_kernel("k2", 3),
        make_elementwise_kernel("k3", 3),
    ]);

    let optimizer = GpuOptimizer::new();
    let tuned = optimizer.quick_tune(&module);

    // Should tune all 3 kernels
    assert_eq!(tuned.len(), 3);

    // Each should have valid block shape
    for (name, config) in &tuned {
        assert!(
            config.block_shape.total_threads() > 0,
            "Kernel {} has invalid block size",
            name
        );
    }
}

#[test]
fn test_ptx_generation_after_optimization() {
    let module = make_test_module(vec![make_elementwise_kernel("test_kernel", 3)]);

    // Run optimizer
    let optimizer = GpuOptimizer::new();
    let (optimized, _) = optimizer.optimize(&module).expect("Optimization failed");

    // Generate PTX
    let mut codegen = PtxCodegen::new((8, 0)); // Ampere
    let ptx = codegen.generate(&optimized);

    // PTX should be valid
    assert!(ptx.contains(".version"));
    assert!(ptx.contains(".target sm_80"));
    assert!(ptx.contains(".entry test_kernel"));
}

#[test]
fn test_end_to_end_optimization_pipeline() {
    // Create a realistic module
    let mut kernels = FxHashMap::default();

    // Add several kernels of different patterns
    kernels.insert(
        "preprocess".to_string(),
        make_elementwise_kernel("preprocess", 4),
    );
    kernels.insert("compute".to_string(), make_elementwise_kernel("compute", 8));
    kernels.insert("reduce".to_string(), make_reduction_kernel("reduce"));

    let module = GpuModule {
        name: "pipeline_test".to_string(),
        target: GpuTarget::Cuda {
            compute_capability: (9, 0), // Hopper
        },
        kernels,
        device_functions: FxHashMap::default(),
        constants: vec![],
    };

    // Run full optimization
    let optimizer = GpuOptimizer::with_config(OptimizerConfig::aggressive());
    let (optimized, report) = optimizer.optimize(&module).expect("Optimization failed");

    // Verify optimization ran
    assert!(report.fusion.enabled);
    assert!(report.autotune.enabled);
    assert!(report.async_pipeline.enabled);

    // Generate PTX for the optimized module
    let mut codegen = PtxCodegen::new((9, 0)); // Hopper
    let ptx = codegen.generate(&optimized);

    // Should target Hopper
    assert!(ptx.contains(".target sm_90"));

    // All kernels should be present
    for name in optimized.kernels.keys() {
        assert!(
            ptx.contains(&format!(".entry {}", name)),
            "Missing kernel {} in PTX",
            name
        );
    }
}

#[test]
fn test_optimization_report_summary() {
    let module = make_test_module(vec![make_elementwise_kernel("kernel", 3)]);

    let optimizer = GpuOptimizer::new();
    let (_, report) = optimizer.optimize(&module).expect("Optimization failed");

    let summary = report.summary();

    // Summary should contain key information
    assert!(summary.contains("Total optimization time"));
    assert!(summary.contains("Fusion") || summary.contains("AutoTune"));
}

#[test]
fn test_optimizer_preserves_module_name() {
    let module = make_test_module(vec![make_elementwise_kernel("kernel", 3)]);

    let optimizer = GpuOptimizer::new();
    let (optimized, _) = optimizer.optimize(&module).expect("Optimization failed");

    assert_eq!(optimized.name, "test_module");
}

#[test]
fn test_occupancy_with_shared_memory() {
    let calc = OccupancyCalculator::from_cuda_arch(CudaArch::Ampere);

    // No shared memory
    let info_no_smem = calc.calculate_occupancy(256, 32, 0);

    // With shared memory (16KB)
    let info_with_smem = calc.calculate_occupancy(256, 32, 16384);

    // Shared memory typically reduces occupancy
    assert!(info_no_smem.occupancy >= info_with_smem.occupancy);
}

#[test]
fn test_occupancy_with_high_register_usage() {
    let calc = OccupancyCalculator::from_cuda_arch(CudaArch::Ampere);

    // Low register usage
    let info_low_regs = calc.calculate_occupancy(256, 16, 0);

    // High register usage
    let info_high_regs = calc.calculate_occupancy(256, 128, 0);

    // Higher register usage typically reduces occupancy
    assert!(info_low_regs.occupancy >= info_high_regs.occupancy);
}
