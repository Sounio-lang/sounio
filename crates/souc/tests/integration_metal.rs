//! Integration Tests for Sounio Metal GPU Backend
//!
//! These tests verify the Metal Shading Language (MSL) code generation,
//! epistemic tracking, counterfactual execution, and runtime bindings
//! for Apple Silicon GPUs.

use sounio::codegen::gpu::{
    // Counterfactual context
    CounterfactualContext,
    // Metal counterfactual
    CounterfactualMetalConfig,
    CounterfactualMetalEmitter,
    CounterfactualValue,
    EpistemicMetalRunner,
    GpuKernel,
    GpuModule,
    GpuParam,
    // Core GPU types
    GpuTarget,
    GpuType,
    MemorySpace,
    // Metal codegen
    MetalCodegen,
    MetalCodegenConfig,
    MetalDeviceInfo,
    MetalDispatchSize,
    MetalGpuFamily,
    // Metal runtime
    MetalRuntime,
    MetalStorageMode,
    compile_counterfactual_metal,
    generate_counterfactual_metal_library,
};

// =============================================================================
// PART 1: Metal GPU Family and Target Tests
// =============================================================================

mod metal_target {
    use super::*;

    #[test]
    fn test_metal_gpu_families() {
        let families = [
            MetalGpuFamily::Apple7,
            MetalGpuFamily::Apple8,
            MetalGpuFamily::Apple9,
            MetalGpuFamily::Apple10,
            MetalGpuFamily::Mac2,
            MetalGpuFamily::Common,
        ];

        for family in &families {
            assert!(family.supports_simdgroup());
            assert_eq!(family.simd_width(), 32);
        }
    }

    #[test]
    fn test_apple_silicon_families() {
        // Apple7 (M1/A14)
        assert_eq!(MetalGpuFamily::Apple7.max_threads_per_threadgroup(), 1024);
        assert_eq!(MetalGpuFamily::Apple7.msl_version(), "2.4");

        // Apple8 (M2/A15/A16)
        assert_eq!(MetalGpuFamily::Apple8.max_threads_per_threadgroup(), 1024);
        assert_eq!(MetalGpuFamily::Apple8.msl_version(), "3.0");
        assert!(MetalGpuFamily::Apple8.supports_simdgroup_matrix());

        // Apple9 (M3/A17)
        assert_eq!(MetalGpuFamily::Apple9.max_threads_per_threadgroup(), 1024);
        assert_eq!(MetalGpuFamily::Apple9.msl_version(), "3.1");
        assert!(MetalGpuFamily::Apple9.supports_simdgroup_matrix());

        // Apple10 (M4/A18) - futureproof
        assert_eq!(MetalGpuFamily::Apple10.max_threads_per_threadgroup(), 1024);
        assert_eq!(MetalGpuFamily::Apple10.msl_version(), "3.2");
        assert!(MetalGpuFamily::Apple10.supports_simdgroup_matrix());
        assert!(MetalGpuFamily::Apple10.supports_mesh_shaders());
        assert!(MetalGpuFamily::Apple10.supports_ray_tracing());
    }

    #[test]
    fn test_intel_mac_family() {
        let family = MetalGpuFamily::Mac2;
        assert_eq!(family.max_threads_per_threadgroup(), 1024);
        assert_eq!(family.msl_version(), "2.4");
        assert!(!family.supports_simdgroup_matrix());
    }

    #[test]
    fn test_gpu_target_metal_variant() {
        let target = GpuTarget::Metal {
            gpu_family: MetalGpuFamily::Apple8,
        };

        match target {
            GpuTarget::Metal { gpu_family } => {
                assert_eq!(gpu_family, MetalGpuFamily::Apple8);
            }
            _ => panic!("Expected Metal target"),
        }
    }

    #[test]
    fn test_gpu_target_display() {
        let target = GpuTarget::Metal {
            gpu_family: MetalGpuFamily::Apple9,
        };
        let display = format!("{}", target);
        assert!(display.contains("Metal") || display.contains("Apple9"));
    }
}

// =============================================================================
// PART 2: MSL Codegen Tests
// =============================================================================

mod msl_codegen {
    use super::*;

    #[test]
    fn test_metal_codegen_config() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: true,
            max_threads_per_threadgroup: 1024,
        };

        assert_eq!(config.gpu_family, MetalGpuFamily::Apple8);
        assert!(config.fast_math);
        assert!(config.epistemic_enabled);
    }

    #[test]
    fn test_metal_codegen_creation() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: false,
            max_threads_per_threadgroup: 1024,
        };

        let codegen = MetalCodegen::new(config);
        let _ = codegen;
    }

    #[test]
    fn test_msl_header_generation() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: false,
            debug_info: false,
            epistemic_enabled: false,
            max_threads_per_threadgroup: 1024,
        };

        let mut codegen = MetalCodegen::new(config);
        let module = GpuModule::new(
            "test_module",
            GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple8,
            },
        );

        let msl = codegen.generate(&module);

        // Check MSL headers
        assert!(msl.contains("#include <metal_stdlib>"));
        assert!(msl.contains("using namespace metal;"));
    }

    #[test]
    fn test_msl_epistemic_types() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple9,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: true,
            max_threads_per_threadgroup: 1024,
        };

        let mut codegen = MetalCodegen::new(config);
        let module = GpuModule::new(
            "epistemic_test",
            GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple9,
            },
        );

        let msl = codegen.generate(&module);

        // Check epistemic type definitions
        assert!(msl.contains("EpistemicFloat"));
        assert!(msl.contains("epsilon"));
        assert!(msl.contains("valid"));
        assert!(msl.contains("provenance"));
    }

    #[test]
    fn test_msl_epistemic_functions() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: true,
            max_threads_per_threadgroup: 1024,
        };

        let mut codegen = MetalCodegen::new(config);
        let module = GpuModule::new(
            "epistemic_funcs",
            GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple8,
            },
        );

        let msl = codegen.generate(&module);

        // Check epistemic helper functions
        assert!(msl.contains("epistemic_add"));
        assert!(msl.contains("epistemic_mul"));
    }

    #[test]
    fn test_msl_simdgroup_helpers() {
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: true,
            max_threads_per_threadgroup: 1024,
        };

        let mut codegen = MetalCodegen::new(config);
        let module = GpuModule::new(
            "simdgroup_test",
            GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple8,
            },
        );

        let msl = codegen.generate(&module);

        // Check simdgroup helpers (Metal's warp equivalent)
        assert!(msl.contains("simd"));
    }
}

// =============================================================================
// PART 3: Counterfactual Metal Tests
// =============================================================================

mod counterfactual_metal {
    use super::*;

    #[test]
    fn test_counterfactual_metal_config() {
        let config = CounterfactualMetalConfig::default();

        assert_eq!(config.worlds_per_simdgroup, 2);
        assert_eq!(config.max_depth, 8);
        assert!(config.track_divergence);
        assert!(config.track_depth);
    }

    #[test]
    fn test_counterfactual_metal_emitter_creation() {
        let config = CounterfactualMetalConfig {
            gpu_family: MetalGpuFamily::Apple9,
            worlds_per_simdgroup: 4,
            track_divergence: true,
            track_depth: true,
            max_depth: 4,
            fast_math: true,
        };

        let emitter = CounterfactualMetalEmitter::new(config);
        let _ = emitter;
    }

    #[test]
    fn test_counterfactual_library_generation() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Check key components
        assert!(library.contains("CounterfactualContext"));
        assert!(library.contains("cf_init"));
        assert!(library.contains("cf_intervene"));
        assert!(library.contains("cf_compute_ite"));
        assert!(library.contains("cf_compute_ate"));
        assert!(library.contains("simd_shuffle_xor"));
    }

    #[test]
    fn test_counterfactual_structural_equations() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Check structural equation models
        assert!(library.contains("sem_linear"));
        assert!(library.contains("sem_logistic"));
        assert!(library.contains("sem_multiplicative"));
        assert!(library.contains("sem_threshold"));
    }

    #[test]
    fn test_counterfactual_causal_estimators() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple9);

        // Check causal effect estimators
        assert!(library.contains("cf_cate")); // CATE
        assert!(library.contains("cf_attributable_fraction")); // AF
        assert!(library.contains("cf_nnt")); // NNT
        assert!(library.contains("cf_probability_causation")); // PoC
    }

    #[test]
    fn test_counterfactual_kernel_generation() {
        let mut ctx = CounterfactualContext::new();
        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.5));
        ctx.intervene("treatment", CounterfactualValue::F32(1.0));

        let msl = compile_counterfactual_metal(&ctx, MetalGpuFamily::Apple8);

        // Check kernel structure
        assert!(msl.contains("kernel void counterfactual_main"));
        assert!(msl.contains("cf_intervene"));
        assert!(msl.contains("thread_position_in_grid"));
        assert!(msl.contains("thread_index_in_simdgroup"));
    }

    #[test]
    fn test_counterfactual_world_helpers() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // World ID helpers
        assert!(library.contains("WORLD_FACTUAL"));
        assert!(library.contains("WORLD_CF_MARKER"));
        assert!(library.contains("world_is_factual"));
        assert!(library.contains("world_is_counterfactual"));
        assert!(library.contains("create_cf_world"));
    }

    #[test]
    fn test_counterfactual_nested_interventions() {
        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Nested intervention support
        assert!(library.contains("cf_nested_intervene"));
        assert!(library.contains("bit 0"));
        assert!(library.contains("bit 1"));
    }
}

// =============================================================================
// PART 4: Metal Runtime Tests
// =============================================================================

mod metal_runtime {
    use super::*;

    #[test]
    fn test_metal_runtime_creation() {
        let runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();
        assert_eq!(runtime.gpu_family(), MetalGpuFamily::Apple8);
        assert_eq!(runtime.allocated_bytes(), 0);
    }

    #[test]
    fn test_metal_runtime_all_families() {
        for family in &[
            MetalGpuFamily::Apple7,
            MetalGpuFamily::Apple8,
            MetalGpuFamily::Apple9,
            MetalGpuFamily::Mac2,
            MetalGpuFamily::Common,
        ] {
            let runtime = MetalRuntime::new(*family).unwrap();
            assert_eq!(runtime.gpu_family(), *family);
        }
    }

    #[test]
    fn test_metal_device_info() {
        let info = MetalDeviceInfo::default_for(MetalGpuFamily::Apple9);

        assert_eq!(info.family, MetalGpuFamily::Apple9);
        assert!(info.supports_simdgroup_matrix);
        assert!(info.supports_apple_silicon_features);
        assert!(info.supports_simdgroup);
        assert_eq!(info.max_threads_per_threadgroup, 1024);
    }

    #[test]
    fn test_metal_storage_modes() {
        // Apple Silicon prefers Shared for CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Apple8, true),
            MetalStorageMode::Shared
        );

        // Intel Mac prefers Managed for CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Mac2, true),
            MetalStorageMode::Managed
        );

        // Private for no CPU access
        assert_eq!(
            MetalStorageMode::recommended_for(MetalGpuFamily::Apple8, false),
            MetalStorageMode::Private
        );
    }

    #[test]
    fn test_metal_buffer_allocation() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let buffer = runtime
            .alloc_buffer(1024, MetalStorageMode::Shared)
            .unwrap();
        assert_eq!(buffer.size(), 1024);
        assert!(buffer.is_valid());
        assert_eq!(runtime.allocated_bytes(), 1024);

        runtime.free_buffer(buffer).unwrap();
        assert_eq!(runtime.allocated_bytes(), 0);
    }

    #[test]
    fn test_metal_typed_allocation() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let buffer = runtime
            .alloc_typed::<f32>(256, MetalStorageMode::Shared)
            .unwrap();
        assert_eq!(buffer.size(), 256 * std::mem::size_of::<f32>());

        runtime.free_buffer(buffer).unwrap();
    }

    #[test]
    fn test_metal_dispatch_size() {
        // 1D dispatch
        let dispatch_1d = MetalDispatchSize::new_1d(64, 256);
        assert_eq!(dispatch_1d.total_threads(), 64 * 256);
        assert_eq!(dispatch_1d.total_threadgroups(), 64);
        assert_eq!(dispatch_1d.threads_per_group(), 256);

        // 2D dispatch
        let dispatch_2d = MetalDispatchSize::new_2d((32, 32), (16, 16));
        assert_eq!(dispatch_2d.total_threads(), 32 * 32 * 16 * 16);
        assert_eq!(dispatch_2d.total_threadgroups(), 32 * 32);
        assert_eq!(dispatch_2d.threads_per_group(), 16 * 16);

        // 3D dispatch
        let dispatch_3d = MetalDispatchSize::new((8, 8, 8), (8, 8, 8));
        assert_eq!(dispatch_3d.total_threads(), 8 * 8 * 8 * 8 * 8 * 8);
    }

    #[test]
    fn test_metal_msl_compilation() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let msl = r#"
            #include <metal_stdlib>
            using namespace metal;

            kernel void test_kernel(
                device const float* input [[buffer(0)]],
                device float* output [[buffer(1)]],
                uint id [[thread_position_in_grid]]
            ) {
                output[id] = input[id] * 2.0f;
            }
        "#;

        let library = runtime.compile_msl(msl).unwrap();
        assert!(library.functions().contains(&"test_kernel".to_string()));
    }

    #[test]
    fn test_metal_kernel_retrieval() {
        let mut runtime = MetalRuntime::new(MetalGpuFamily::Apple8).unwrap();

        let msl = "kernel void my_kernel() {}";
        let library = runtime.compile_msl(msl).unwrap();

        let kernel = runtime.get_kernel(&library, "my_kernel").unwrap();
        assert_eq!(kernel.name(), "my_kernel");
        assert_eq!(kernel.thread_execution_width(), 32);
    }

    #[test]
    fn test_epistemic_metal_runner() {
        let runner = EpistemicMetalRunner::new(MetalGpuFamily::Apple8).unwrap();
        assert_eq!(runner.runtime().gpu_family(), MetalGpuFamily::Apple8);
    }
}

// =============================================================================
// PART 5: End-to-End Pipeline Tests
// =============================================================================

mod e2e_pipeline {
    use super::*;

    #[test]
    fn test_full_metal_epistemic_pipeline() {
        // 1. Create GPU module targeting Metal
        let mut module = GpuModule::new(
            "epistemic_computation",
            GpuTarget::Metal {
                gpu_family: MetalGpuFamily::Apple8,
            },
        );

        // 2. Add epistemic kernel
        let mut kernel = GpuKernel::new("epistemic_vector_add");
        kernel.params = vec![
            GpuParam {
                name: "a".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "b".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "out".to_string(),
                ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
                space: MemorySpace::Global,
                restrict: true,
            },
            GpuParam {
                name: "n".to_string(),
                ty: GpuType::U32,
                space: MemorySpace::Local,
                restrict: false,
            },
        ];
        kernel.max_threads = Some(256);
        module
            .kernels
            .insert("epistemic_vector_add".to_string(), kernel);

        // 3. Generate MSL with epistemic support
        let config = MetalCodegenConfig {
            gpu_family: MetalGpuFamily::Apple8,
            fast_math: true,
            debug_info: false,
            epistemic_enabled: true,
            max_threads_per_threadgroup: 256,
        };

        let mut codegen = MetalCodegen::new(config);
        let msl = codegen.generate(&module);

        // 4. Verify generated code
        assert!(msl.contains("#include <metal_stdlib>"));
        assert!(msl.contains("EpistemicFloat"));
        assert!(msl.contains("epistemic_add"));
    }

    #[test]
    fn test_full_metal_counterfactual_pipeline() {
        // 1. Create counterfactual context
        let mut ctx = CounterfactualContext::new();
        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("age", CounterfactualValue::F32(45.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.3));

        // 2. Apply intervention
        let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));
        assert!(cf_world.is_counterfactual());

        // 3. Generate counterfactual MSL
        let msl = compile_counterfactual_metal(&ctx, MetalGpuFamily::Apple9);

        // 4. Verify generated code
        assert!(msl.contains("CounterfactualContext"));
        assert!(msl.contains("cf_intervene"));
        assert!(msl.contains("cf_compute_ite"));
        assert!(msl.contains("simd_shuffle_xor"));

        // 5. Create runtime
        let runtime = MetalRuntime::new(MetalGpuFamily::Apple9).unwrap();
        assert!(runtime.device_info().supports_simdgroup_matrix);
    }

    #[test]
    fn test_metal_vs_cuda_target_differentiation() {
        let cuda_target = GpuTarget::Cuda {
            compute_capability: (8, 0),
        };
        let metal_target = GpuTarget::Metal {
            gpu_family: MetalGpuFamily::Apple8,
        };

        // Different targets
        assert!(matches!(cuda_target, GpuTarget::Cuda { .. }));
        assert!(matches!(metal_target, GpuTarget::Metal { .. }));

        // Create modules for each
        let cuda_module = GpuModule::new("cuda_mod", cuda_target);
        let metal_module = GpuModule::new("metal_mod", metal_target);

        assert!(matches!(cuda_module.target, GpuTarget::Cuda { .. }));
        assert!(matches!(metal_module.target, GpuTarget::Metal { .. }));
    }
}

// =============================================================================
// PART 6: Cross-Platform Compatibility Tests
// =============================================================================

mod cross_platform {
    use super::*;

    #[test]
    fn test_epistemic_concepts_across_backends() {
        // Epistemic uncertainty is tracked consistently across CUDA and Metal

        // CUDA: uses shadow registers
        // Metal: uses EpistemicFloat struct

        // Both should support:
        // 1. Value + epsilon pairs
        // 2. Validity tracking
        // 3. Provenance tracking
        // 4. Warp/simdgroup level operations

        let metal_library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Metal uses simd_sum for reduction (CUDA uses warp shuffles)
        assert!(metal_library.contains("simd_sum"));

        // Both track individual treatment effect (ITE)
        assert!(metal_library.contains("cf_compute_ite"));

        // Both compute average treatment effect (ATE)
        assert!(metal_library.contains("cf_compute_ate"));
    }

    #[test]
    fn test_terminology_mapping() {
        // CUDA -> Metal terminology
        // warp -> simdgroup
        // block -> threadgroup
        // grid -> grid (same)
        // __shared__ -> threadgroup
        // __device__ -> device
        // __syncthreads() -> threadgroup_barrier()
        // laneid -> thread_index_in_simdgroup
        // __shfl_xor_sync -> simd_shuffle_xor

        let library = generate_counterfactual_metal_library(MetalGpuFamily::Apple8);

        // Metal-specific terminology
        assert!(library.contains("simd"));
        assert!(library.contains("thread_index_in_simdgroup") || library.contains("simd_lane_id"));
    }
}
