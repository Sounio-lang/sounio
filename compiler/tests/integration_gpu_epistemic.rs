//! Integration Tests for Sounio GPU + Epistemic Pipeline
//!
//! These tests verify the end-to-end compilation pipeline from source
//! to GPU code (PTX), including epistemic state tracking.

use sounio::codegen::gpu::{
    // Counterfactual execution
    CounterfactualContext,
    CounterfactualPtxConfig,
    CounterfactualPtxEmitter,
    CounterfactualValue,
    // Epistemic PTX emission
    EpistemicPtxConfig,
    EpistemicPtxEmitter,
    EpistemicShadowRegs,
    // Core GPU IR types
    GpuKernel,
    GpuModule,
    GpuParam,
    GpuTarget,
    GpuType,
    // HLIR to GPU lowering
    LoweringConfig,
    MemorySpace,
    PtxCodegen,
    StructuralEqType,
    WarpEpsilonOp,
    WorldId,
    WorldSnapshot,
    compile_to_ptx,
    compile_to_ptx_epistemic,
    lower,
    lower_with_config,
};

use sounio::hlir::HlirModule;

// =============================================================================
// PART 1: Basic GPU IR Construction Tests
// =============================================================================

mod gpu_ir_construction {
    use super::*;

    #[test]
    fn test_create_empty_gpu_module() {
        let module = GpuModule::new(
            "test_module",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );

        assert_eq!(module.name, "test_module");
        assert!(matches!(
            module.target,
            GpuTarget::Cuda {
                compute_capability: (8, 0)
            }
        ));
    }

    #[test]
    fn test_create_gpu_kernel() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );

        let mut kernel = GpuKernel::new("vector_add");
        kernel.params = vec![
            GpuParam {
                name: "a".to_string(),
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
        ];
        kernel.max_threads = Some(256);

        module.kernels.insert("vector_add".to_string(), kernel);

        assert_eq!(module.kernels.len(), 1);
        assert!(module.kernels.contains_key("vector_add"));
    }

    #[test]
    fn test_gpu_types() {
        let types = vec![
            GpuType::F32,
            GpuType::F64,
            GpuType::I32,
            GpuType::I64,
            GpuType::U32,
            GpuType::U64,
            GpuType::Bool,
            GpuType::Void,
            GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Shared),
        ];

        assert_eq!(types.len(), 10);
    }

    #[test]
    fn test_memory_spaces() {
        let spaces = [
            MemorySpace::Global,
            MemorySpace::Shared,
            MemorySpace::Local,
            MemorySpace::Constant,
        ];

        for space in &spaces {
            let ptr_type = GpuType::Ptr(Box::new(GpuType::F32), *space);
            match ptr_type {
                GpuType::Ptr(inner, s) => {
                    assert!(matches!(*inner, GpuType::F32));
                    assert_eq!(s, *space);
                }
                _ => panic!("Expected pointer type"),
            }
        }
    }
}

// =============================================================================
// PART 2: PTX Codegen Tests
// =============================================================================

mod ptx_codegen {
    use super::*;

    #[test]
    fn test_ptx_codegen_creation() {
        let codegen = PtxCodegen::new((8, 0));
        let _ = codegen;
    }

    #[test]
    fn test_ptx_generation_empty_module() {
        let module = GpuModule::new(
            "empty",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );
        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        // Should produce valid PTX
        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_ptx_generation_with_kernel() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );

        let mut kernel = GpuKernel::new("simple_kernel");
        kernel.params = vec![GpuParam {
            name: "x".to_string(),
            ty: GpuType::F32,
            space: MemorySpace::Local,
            restrict: false,
        }];
        kernel.max_threads = Some(256);

        module.kernels.insert("simple_kernel".to_string(), kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        assert!(!ptx.is_empty());
    }
}

// =============================================================================
// PART 3: HLIR to GPU Lowering Tests
// =============================================================================

mod hlir_lowering {
    use super::*;

    fn create_simple_hlir_module() -> HlirModule {
        HlirModule::new("test")
    }

    #[test]
    fn test_lowering_config_default() {
        let config = LoweringConfig::default();

        assert!(matches!(config.target, GpuTarget::Cuda { .. }));
        assert!(config.epistemic_enabled);
    }

    #[test]
    fn test_lower_empty_module() {
        let hlir = create_simple_hlir_module();
        let target = GpuTarget::Cuda {
            compute_capability: (8, 0),
        };

        let gpu_module = lower(&hlir, target);

        // Module is created successfully (name may differ from hlir name)
        assert!(!gpu_module.name.is_empty());
    }

    #[test]
    fn test_lower_with_config() {
        let hlir = create_simple_hlir_module();
        let config = LoweringConfig {
            target: GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
            epistemic_enabled: true,
            ..Default::default()
        };

        let gpu_module = lower_with_config(&hlir, &config);

        // Module is created successfully (name may differ from hlir name)
        assert!(!gpu_module.name.is_empty());
    }

    #[test]
    fn test_compile_to_ptx_empty() {
        let hlir = create_simple_hlir_module();
        let sm_version = (8, 0);

        let ptx = compile_to_ptx(&hlir, sm_version);

        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_compile_to_ptx_epistemic() {
        let hlir = create_simple_hlir_module();
        let sm_version = (8, 0);

        let ptx = compile_to_ptx_epistemic(&hlir, sm_version, true);

        assert!(!ptx.is_empty());
    }
}

// =============================================================================
// PART 4: Epistemic PTX Emission Tests
// =============================================================================

mod epistemic_ptx {
    use super::*;

    #[test]
    fn test_epistemic_config_default() {
        let config = EpistemicPtxConfig::default();

        assert!(config.confidence_threshold > 0.0);
        assert!(config.provenance_tracking);
    }

    #[test]
    fn test_epistemic_emitter_creation() {
        let config = EpistemicPtxConfig::default();
        let emitter = EpistemicPtxEmitter::new(config);

        let output = emitter.output();
        assert!(output.is_empty());
    }

    #[test]
    fn test_epistemic_shadow_regs() {
        let regs = EpistemicShadowRegs {
            value: "%r_value".to_string(),
            epsilon: "%r_epsilon".to_string(),
            validity: "%p_valid".to_string(),
            provenance: "%r_prov".to_string(),
        };

        assert_eq!(regs.value, "%r_value");
        assert_eq!(regs.epsilon, "%r_epsilon");
        assert_eq!(regs.validity, "%p_valid");
        assert_eq!(regs.provenance, "%r_prov");
    }

    #[test]
    fn test_epistemic_add_emission() {
        let config = EpistemicPtxConfig::default();
        let mut emitter = EpistemicPtxEmitter::new(config);

        let a = EpistemicShadowRegs {
            value: "%r_a".to_string(),
            epsilon: "%r_eps_a".to_string(),
            validity: "%p_valid_a".to_string(),
            provenance: "%r_prov_a".to_string(),
        };

        let b = EpistemicShadowRegs {
            value: "%r_b".to_string(),
            epsilon: "%r_eps_b".to_string(),
            validity: "%p_valid_b".to_string(),
            provenance: "%r_prov_b".to_string(),
        };

        let result = EpistemicShadowRegs {
            value: "%r_result".to_string(),
            epsilon: "%r_eps_result".to_string(),
            validity: "%p_valid_result".to_string(),
            provenance: "%r_prov_result".to_string(),
        };

        emitter.emit_epistemic_add(&result, &a, &b, false);

        let output = emitter.output();
        assert!(output.contains("add") || output.contains("Epistemic"));
    }

    #[test]
    fn test_epistemic_mul_emission() {
        let config = EpistemicPtxConfig::default();
        let mut emitter = EpistemicPtxEmitter::new(config);

        let a = EpistemicShadowRegs {
            value: "%r_a".to_string(),
            epsilon: "%r_eps_a".to_string(),
            validity: "%p_valid_a".to_string(),
            provenance: "%r_prov_a".to_string(),
        };

        let b = EpistemicShadowRegs {
            value: "%r_b".to_string(),
            epsilon: "%r_eps_b".to_string(),
            validity: "%p_valid_b".to_string(),
            provenance: "%r_prov_b".to_string(),
        };

        let result = EpistemicShadowRegs {
            value: "%r_result".to_string(),
            epsilon: "%r_eps_result".to_string(),
            validity: "%p_valid_result".to_string(),
            provenance: "%r_prov_result".to_string(),
        };

        emitter.emit_epistemic_mul(&result, &a, &b);

        let output = emitter.output();
        assert!(output.contains("mul") || output.contains("Epistemic"));
    }

    #[test]
    fn test_warp_epsilon_reduce() {
        let config = EpistemicPtxConfig::default();
        let mut emitter = EpistemicPtxEmitter::new(config);

        let shadow = EpistemicShadowRegs {
            value: "%r_val".to_string(),
            epsilon: "%r_eps".to_string(),
            validity: "%p_valid".to_string(),
            provenance: "%r_prov".to_string(),
        };

        emitter.emit_warp_epsilon_reduce(&shadow, "%r_result", WarpEpsilonOp::Max);

        let output = emitter.output();
        assert!(output.contains("shfl") || output.contains("warp") || output.contains("reduce"));
    }

    #[test]
    fn test_confidence_gate() {
        let config = EpistemicPtxConfig::default();
        let mut emitter = EpistemicPtxEmitter::new(config);

        let shadow = EpistemicShadowRegs {
            value: "%r_val".to_string(),
            epsilon: "%r_eps".to_string(),
            validity: "%p_valid".to_string(),
            provenance: "%r_prov".to_string(),
        };

        emitter.emit_confidence_gate(&shadow, 0.1, "high_conf", "low_conf");

        let output = emitter.output();
        assert!(
            output.contains("setp") || output.contains("confidence") || output.contains("gate")
        );
    }
}

// =============================================================================
// PART 5: Counterfactual Execution Tests
// =============================================================================

mod counterfactual_execution {
    use super::*;

    #[test]
    fn test_world_id_factual() {
        let factual = WorldId::FACTUAL;

        assert!(factual.is_factual());
        assert!(!factual.is_counterfactual());
        assert_eq!(factual.intervention_id(), None);
    }

    #[test]
    fn test_world_id_counterfactual() {
        let cf = WorldId::counterfactual(42);

        assert!(!cf.is_factual());
        assert!(cf.is_counterfactual());
        assert_eq!(cf.intervention_id(), Some(42));
    }

    #[test]
    fn test_counterfactual_context_creation() {
        let ctx = CounterfactualContext::new();

        assert!(ctx.current_world.is_factual());
        assert!(ctx.interventions.is_empty());
    }

    #[test]
    fn test_counterfactual_set_factual() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("age", CounterfactualValue::F32(45.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.3));

        assert_eq!(
            ctx.get_value("treatment", WorldId::FACTUAL)
                .and_then(|v| v.as_f32()),
            Some(0.0)
        );
        assert_eq!(
            ctx.get_value("age", WorldId::FACTUAL)
                .and_then(|v| v.as_f32()),
            Some(45.0)
        );
    }

    #[test]
    fn test_counterfactual_intervention() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.3));

        let cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

        assert_eq!(
            ctx.get_value("treatment", WorldId::FACTUAL)
                .and_then(|v| v.as_f32()),
            Some(0.0)
        );

        assert_eq!(
            ctx.get_value("treatment", cf_world)
                .and_then(|v| v.as_f32()),
            Some(1.0)
        );
    }

    #[test]
    fn test_counterfactual_divergence() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("x", CounterfactualValue::F32(10.0));
        let cf_world = ctx.intervene("x", CounterfactualValue::F32(20.0));

        let divergence = ctx.compute_divergence("x", cf_world);

        assert!(divergence.is_some());
        let div = divergence.unwrap();
        assert!((div.absolute - 10.0).abs() < 0.001);
        assert!((div.relative - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_counterfactual_ptx_config() {
        let config = CounterfactualPtxConfig::default();

        assert!(config.worlds_per_warp >= 2);
        assert!(config.track_divergence);
        assert!(config.max_depth > 0);
    }

    #[test]
    fn test_counterfactual_ptx_emitter() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_cf_declarations();
        emitter.emit_cf_init();

        let output = emitter.output();

        assert!(output.contains("world_id") || output.contains("r_world"));
        assert!(output.contains("factual") || output.contains("p_is"));
    }

    #[test]
    fn test_counterfactual_intervention_emission() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_intervention("treatment", "%r_treatment", 1.0, 1);

        let output = emitter.output();

        assert!(output.contains("do(") || output.contains("Intervention"));
        assert!(output.contains("selp") || output.contains("select"));
    }

    #[test]
    fn test_counterfactual_divergence_emission() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_divergence_compute("%r_outcome", "%r_divergence");

        let output = emitter.output();

        assert!(output.contains("shfl") || output.contains("xor"));
        assert!(
            output.contains("ITE") || output.contains("divergence") || output.contains("effect")
        );
    }

    #[test]
    fn test_ate_computation() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_ate_compute("%r_ite", "%r_ate");

        let output = emitter.output();

        assert!(output.contains("shfl") || output.contains("ATE") || output.contains("Average"));
    }

    #[test]
    fn test_structural_equation_linear() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_structural_eq(
            "%r_y",
            StructuralEqType::Linear,
            &["%r_x1", "%r_x2"],
            &[0.5, 2.0, -1.0],
        );

        let output = emitter.output();

        assert!(output.contains("fma") || output.contains("Linear") || output.contains("mul"));
    }

    #[test]
    fn test_structural_equation_logistic() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_structural_eq("%r_y", StructuralEqType::Logistic, &["%r_x"], &[0.0, 1.0]);

        let output = emitter.output();

        assert!(output.contains("ex2") || output.contains("rcp") || output.contains("Logistic"));
    }

    #[test]
    fn test_structural_equation_threshold() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_structural_eq("%r_y", StructuralEqType::Threshold, &["%r_x"], &[0.5]);

        let output = emitter.output();

        assert!(output.contains("setp") || output.contains("Threshold") || output.contains("selp"));
    }

    #[test]
    fn test_probability_causation() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_probability_causation("%r_x", "%r_y", "%r_y_cf", "%r_prob");

        let output = emitter.output();

        assert!(
            output.contains("Causation") || output.contains("caused") || output.contains("setp")
        );
    }

    #[test]
    fn test_nested_intervention() {
        let config = CounterfactualPtxConfig {
            track_depth: true,
            max_depth: 4,
            ..Default::default()
        };
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_nested_intervention(
            "treatment",
            1.0,
            "dosage",
            100.0,
            "%r_treatment",
            "%r_dosage",
        );

        let output = emitter.output();

        assert!(output.contains("Nested") || output.contains("do("));
    }
}

// =============================================================================
// PART 6: World Snapshot Tests
// =============================================================================

mod world_snapshots {
    use super::*;

    #[test]
    fn test_factual_snapshot() {
        let snapshot = WorldSnapshot::factual();

        assert!(snapshot.world_id.is_factual());
        assert_eq!(snapshot.depth, 0);
        assert!(snapshot.values.is_empty());
    }

    #[test]
    fn test_intervention_creates_snapshot() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("x", CounterfactualValue::F32(1.0));
        let cf_world = ctx.intervene("x", CounterfactualValue::F32(2.0));

        let snapshot = ctx.snapshots.get(&cf_world);

        assert!(snapshot.is_some());
        let snap = snapshot.unwrap();
        assert_eq!(snap.depth, 1);
        assert!(snap.world_id.is_counterfactual());
    }

    #[test]
    fn test_multiple_interventions() {
        let mut ctx = CounterfactualContext::new();

        ctx.set_factual("a", CounterfactualValue::F32(1.0));
        ctx.set_factual("b", CounterfactualValue::F32(2.0));

        let world1 = ctx.intervene("a", CounterfactualValue::F32(10.0));
        let world2 = ctx.intervene("b", CounterfactualValue::F32(20.0));

        assert_ne!(world1, world2);
        assert!(world1.is_counterfactual());
        assert!(world2.is_counterfactual());
    }
}

// =============================================================================
// PART 7: Full Pipeline Integration Tests
// =============================================================================

mod full_pipeline {
    use super::*;

    #[test]
    fn test_end_to_end_simple() {
        let hlir = HlirModule::new("test_e2e");

        let gpu_module = lower(
            &hlir,
            GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
        );

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&gpu_module);

        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_end_to_end_epistemic() {
        let hlir = HlirModule::new("test_epistemic_e2e");

        let config = LoweringConfig {
            target: GpuTarget::Cuda {
                compute_capability: (8, 0),
            },
            epistemic_enabled: true,
            ..Default::default()
        };

        let gpu_module = lower_with_config(&hlir, &config);
        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&gpu_module);

        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_counterfactual_full_pipeline() {
        let mut ctx = CounterfactualContext::new();
        ctx.set_factual("treatment", CounterfactualValue::F32(0.0));
        ctx.set_factual("outcome", CounterfactualValue::F32(0.3));

        let _cf_world = ctx.intervene("treatment", CounterfactualValue::F32(1.0));

        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_cf_declarations();
        emitter.emit_cf_init();
        emitter.emit_parallel_worlds(&ctx);
        emitter.emit_divergence_compute("%r_outcome", "%r_ite");
        emitter.emit_ate_compute("%r_ite", "%r_ate");

        let ptx = emitter.output();

        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_compile_to_ptx_api() {
        let hlir = HlirModule::new("api_test");

        let ptx = compile_to_ptx(&hlir, (8, 0));

        assert!(!ptx.is_empty());
    }

    #[test]
    fn test_compile_to_ptx_epistemic_api() {
        let hlir = HlirModule::new("epistemic_api_test");

        let ptx = compile_to_ptx_epistemic(&hlir, (8, 0), true);

        assert!(!ptx.is_empty());
    }
}

// =============================================================================
// PART 8: Edge Cases and Error Handling
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    fn test_empty_intervention_list() {
        let ctx = CounterfactualContext::new();

        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_parallel_worlds(&ctx);

        let output = emitter.output();
        assert!(output.is_empty() || output.contains("Counterfactual"));
    }

    #[test]
    fn test_large_intervention_id() {
        let cf = WorldId::counterfactual(u32::MAX);

        assert!(cf.is_counterfactual());
        assert_eq!(cf.intervention_id(), Some(u32::MAX));
    }

    #[test]
    fn test_counterfactual_value_conversions() {
        assert_eq!(CounterfactualValue::F32(1.0).as_f32(), Some(1.0));
        assert_eq!(CounterfactualValue::F64(2.0).as_f32(), Some(2.0));
        assert_eq!(CounterfactualValue::I32(3).as_f32(), Some(3.0));
        assert_eq!(CounterfactualValue::I64(4).as_f32(), Some(4.0));
        assert_eq!(CounterfactualValue::Bool(true).as_f32(), None);
        assert_eq!(CounterfactualValue::Vector(vec![1.0, 2.0]).as_f32(), None);
    }
}

// =============================================================================
// PART 9: Performance Characteristics Tests
// =============================================================================

mod performance {
    use super::*;

    #[test]
    fn test_multiple_worlds_creation() {
        let mut ctx = CounterfactualContext::new();

        for i in 0..100 {
            ctx.set_factual(&format!("var_{}", i), CounterfactualValue::F32(i as f32));
        }

        for i in 0..50 {
            ctx.intervene(
                &format!("var_{}", i),
                CounterfactualValue::F32((i * 2) as f32),
            );
        }

        assert_eq!(ctx.interventions.len(), 50);
        assert_eq!(ctx.snapshots.len(), 51);
    }

    #[test]
    fn test_ptx_emission_size() {
        let config = CounterfactualPtxConfig::default();
        let mut emitter = CounterfactualPtxEmitter::new(config);

        emitter.emit_cf_declarations();
        emitter.emit_cf_init();
        emitter.emit_intervention("x", "%r_x", 1.0, 1);
        emitter.emit_divergence_compute("%r_y", "%r_div");
        emitter.emit_ate_compute("%r_div", "%r_ate");
        emitter.emit_structural_eq(
            "%r_z",
            StructuralEqType::Linear,
            &["%r_a", "%r_b"],
            &[1.0, 2.0, 3.0],
        );

        let output = emitter.output();

        assert!(output.len() < 100_000);
        assert!(output.len() > 100);
    }
}
