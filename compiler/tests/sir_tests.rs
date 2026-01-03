//! SIR (Sounio IR) Integration Tests
//!
//! Tests for the Sounio Native Backend infrastructure.

use sounio::sir::*;

#[test]
fn test_sir_module_creation() {
    let module = SirModule::new("test_module");
    assert_eq!(module.name, "test_module");
    assert!(module.functions.is_empty());
}

#[test]
fn test_sir_type_sizes() {
    use types::*;

    assert_eq!(SirType::i32().size_bytes(), 4);
    assert_eq!(SirType::i64().size_bytes(), 8);
    assert_eq!(SirType::f32().size_bytes(), 4);
    assert_eq!(SirType::f64().size_bytes(), 8);
    assert_eq!(SirType::bool().size_bytes(), 1);
}

#[test]
fn test_sir_struct_layout() {
    use types::*;

    // Struct { i32, f64 } should have padding
    let s = StructType::new(vec![
        (Some("x".into()), SirType::i32()),
        (Some("y".into()), SirType::f64()),
    ]);

    // i32 at offset 0
    assert_eq!(s.fields[0].offset, 0);
    // f64 at offset 8 (aligned to 8 bytes)
    assert_eq!(s.fields[1].offset, 8);
    // Total size 16
    assert_eq!(s.size, 16);
    // Alignment from f64
    assert_eq!(s.align, 8);
}

#[test]
fn test_sir_vector_types() {
    use types::*;

    let v = VectorType::f64x4();
    assert_eq!(v.size_bytes(), 32); // 4 * 8 bytes
    assert_eq!(v.lanes, 4);
}

#[test]
fn test_sir_builder_simple_function() {
    let mut module = SirModule::new("test");

    {
        let mut builder = builder::SirBuilder::new(&mut module);

        // Create: fn add(a: i32, b: i32) -> i32 { a + b }
        let _func_id = builder.create_function(
            "add",
            vec![("a".into(), SirType::i32()), ("b".into(), SirType::i32())],
            SirType::i32(),
        );

        // Parameters are at indices 0 and 1
        let a = values::ValueId::new(0);
        let b = values::ValueId::new(1);

        let sum = builder.add(a, b);
        builder.ret(sum);
    }

    // Validate
    assert!(module.validate().is_ok());
    assert_eq!(module.functions.len(), 1);
    assert_eq!(module.functions[0].name, "add");
}

#[test]
fn test_sir_builder_epistemic_function() {
    let mut module = SirModule::new("test");

    {
        let mut builder = builder::SirBuilder::new(&mut module);

        // fn uncertain_mul(a: f64, conf_a: f64, b: f64, conf_b: f64) -> epistemic<f64>
        builder.create_function(
            "uncertain_mul",
            vec![
                ("a".into(), SirType::f64()),
                ("conf_a".into(), SirType::f64()),
                ("b".into(), SirType::f64()),
                ("conf_b".into(), SirType::f64()),
            ],
            SirType::Epistemic(Box::new(SirType::f64())),
        );

        let a = values::ValueId::new(0);
        let conf_a = values::ValueId::new(1);
        let b = values::ValueId::new(2);
        let conf_b = values::ValueId::new(3);

        // Fused epistemic multiplication
        let result = builder.epistemic_fused_mul(a, conf_a, b, conf_b);
        builder.ret(result);
    }

    assert!(module.validate().is_ok());
    println!("{}", module);
}

#[test]
fn test_sir_builder_control_flow() {
    let mut module = SirModule::new("test");

    {
        let mut builder = builder::SirBuilder::new(&mut module);

        // fn abs(x: i32) -> i32 { if x < 0 { -x } else { x } }
        builder.create_function("abs", vec![("x".into(), SirType::i32())], SirType::i32());

        let x = values::ValueId::new(0);
        let zero = builder.const_i32(0);

        // Compare x < 0
        let cond = builder.icmp_slt(x, zero);

        // Create blocks
        let neg_block = builder.create_block_named("negative");
        let pos_block = builder.create_block_named("positive");

        // Branch
        builder.cond_br(cond, neg_block, pos_block);

        // Negative block: return 0 - x
        builder.position_at(neg_block);
        let neg_x = builder.sub(zero, x);
        builder.ret(neg_x);

        // Positive block: return x
        builder.position_at(pos_block);
        builder.ret(x);
    }

    assert!(module.validate().is_ok());
}

#[test]
fn test_sir_pass_manager() {
    use passes::*;

    let mut module = SirModule::new("test");

    // Create a simple function
    {
        let mut builder = builder::SirBuilder::new(&mut module);
        builder.create_function("main", vec![], SirType::i32());
        let zero = builder.const_i32(0);
        builder.ret(zero);
    }

    // Run optimization passes
    let mut pm = PassManager::with_defaults();
    let results = pm.run(&mut module);

    // Should have run multiple passes
    assert!(!results.is_empty());
}

#[test]
fn test_sir_metadata_certainty() {
    use metadata::*;

    let mut store = MetadataStore::new();
    let val = values::ValueId::new(0);

    // Mark value as certain
    store.attach(val, Metadata::Epistemic(EpistemicMetadata::certain()));

    assert!(store.is_certain(val));

    // Check epistemic metadata
    let ep = store.epistemic(val).unwrap();
    assert!(ep.certain);
    assert_eq!(ep.known_confidence, Some(1.0));
}

#[test]
fn test_sir_physical_units() {
    use values::PhysicalUnit;

    let mg = PhysicalUnit::milligram();
    let l = PhysicalUnit::liter();

    // mg/L = concentration
    let conc = mg.divide(&l);

    // Dimensions should be mass / volume = kg / m³
    assert_eq!(conc.dimensions[0], -3); // m^-3 (from liter = 10^-3 m³)
    assert_eq!(conc.dimensions[1], 1); // kg^1
}

#[test]
fn test_sir_constant_values() {
    use values::Constant;

    let c1 = Constant::F64(3.14159);
    assert!(matches!(c1.ty(), SirType::Scalar(types::ScalarType::F64)));

    let c2 = Constant::I32(42);
    assert!(matches!(c2.ty(), SirType::Scalar(types::ScalarType::I32)));

    // Epistemic constant
    let c3 = Constant::Epistemic {
        value: Box::new(Constant::F64(0.5)),
        confidence: 0.9,
    };
    assert!(matches!(c3.ty(), SirType::Epistemic(_)));
}

#[test]
fn test_sir_distribution_types() {
    use types::*;

    let normal_ty = SirType::Distribution(DistributionKind::Normal);
    let lognormal_ty = SirType::Distribution(DistributionKind::LogNormal);

    // Distributions have known sizes
    assert_eq!(normal_ty.size_bytes(), 24);
    assert_eq!(lognormal_ty.size_bytes(), 24);
}

#[test]
#[cfg(target_arch = "x86_64")]
fn test_sir_x86_64_emit_basic() {
    use emit::*;
    use module::*;

    let mut module = SirModule::new("test");
    module.create_function("empty", vec![], SirType::Void);

    // Should successfully emit code
    let result = emit_code(&module, None);
    assert!(result.is_ok());

    let segment = result.unwrap();
    // Should have generated some code
    assert!(!segment.code.is_empty());
    // Should have defined the 'empty' symbol
    assert_eq!(segment.symbols.len(), 1);
    assert_eq!(segment.symbols[0].name, "empty");
}

#[test]
fn test_sir_target_triple() {
    use module::*;

    let native = TargetTriple::native();
    assert!(native.is_64bit());

    // Pointer size should be 8 on 64-bit
    assert_eq!(native.pointer_size(), 8);
}

#[test]
fn test_sir_module_validation() {
    let mut module = SirModule::new("test");

    // Create function without terminator
    let func_id = module.create_function("incomplete", vec![], SirType::Void);

    if let Some(func) = module.function_mut(func_id) {
        func.blocks
            .push(blocks::BasicBlock::new(values::BlockId::new(0)));
    }

    // Validation should fail
    let result = module.validate();
    assert!(result.is_err());
}

#[test]
fn test_sir_instruction_operands() {
    use ops::*;

    let add = SirInst::BinOp {
        op: ArithOp::Add,
        lhs: values::ValueId::new(0),
        rhs: values::ValueId::new(1),
    };

    let operands = add.operands();
    assert_eq!(operands.len(), 2);
    assert_eq!(operands[0], values::ValueId::new(0));
    assert_eq!(operands[1], values::ValueId::new(1));
}

#[test]
fn test_sir_instruction_side_effects() {
    use ops::*;

    // Pure arithmetic has no side effects
    let add = SirInst::BinOp {
        op: ArithOp::Add,
        lhs: values::ValueId::new(0),
        rhs: values::ValueId::new(1),
    };
    assert!(!add.has_side_effects());

    // Store has side effects
    let store = SirInst::Memory(MemoryOp::Store {
        ptr: values::ValueId::new(0),
        val: values::ValueId::new(1),
        volatile: false,
        align: None,
    });
    assert!(store.has_side_effects());

    // Sampling has side effects (RNG state)
    let sample = SirInst::Prob(ProbOp::Sample {
        dist: values::ValueId::new(0),
        rng: values::ValueId::new(1),
    });
    assert!(sample.has_side_effects());
}

// ============================================================================
// ATTENTION-BASED REGISTER ALLOCATION TESTS
// ============================================================================

#[test]
fn test_attention_config_default() {
    use emit::AttentionConfig;

    let config = AttentionConfig::default();
    assert_eq!(config.w_use_density, 1.0);
    assert_eq!(config.w_crosses_call, 0.5);
    assert_eq!(config.w_confidence, 0.5);
    assert_eq!(config.w_uncertainty, 0.3);
    assert_eq!(config.w_provenance, 0.2);
}

#[test]
fn test_attention_config_scientific() {
    use emit::AttentionConfig;

    let config = AttentionConfig::scientific();
    // Scientific config prioritizes epistemic properties
    assert!(config.w_confidence > config.w_use_density * 0.5);
    assert!(config.w_provenance > AttentionConfig::default().w_provenance);
}

#[test]
fn test_attention_config_performance() {
    use emit::AttentionConfig;

    let config = AttentionConfig::performance();
    // Performance config prioritizes classical heuristics
    assert!(config.w_use_density > config.w_confidence);
    assert!(config.w_use_density > AttentionConfig::default().w_use_density);
}

#[test]
fn test_live_interval_use_density() {
    use emit::LiveInterval;
    use values::ValueId;

    let mut interval = LiveInterval::new(ValueId::new(0), 0, SirType::i32());

    // Record uses at different positions
    interval.record_use(10);
    interval.record_use(20);
    interval.record_use(30);
    interval.record_use(40);

    // Lifetime is 40 - 0 = 40 (but we use extend_to, so end = 41)
    // Uses = 4
    // Density = 4 / 41 ≈ 0.097
    assert!(interval.use_density() > 0.0);
    assert!(interval.use_density() < 1.0);
}

#[test]
fn test_attention_score_ordering() {
    use emit::{AttentionConfig, LiveInterval};
    use values::ValueId;

    let config = AttentionConfig::default();

    // Critical value: high confidence, low uncertainty, with provenance
    let mut critical = LiveInterval::with_epistemic(
        ValueId::new(1),
        0,
        SirType::f64(),
        0.95,
        true, // has provenance
    );
    critical.set_uncertainty(0.1);
    critical.crosses_call = true;
    // Add uses
    for i in 0..5 {
        critical.record_use(10 + i * 10);
    }

    // Dispensable value: low confidence, high uncertainty, no provenance
    let mut dispensable = LiveInterval::with_epistemic(
        ValueId::new(2),
        0,
        SirType::f64(),
        0.2,
        false, // no provenance
    );
    dispensable.set_uncertainty(0.9);
    dispensable.crosses_call = false;
    // Fewer uses
    dispensable.record_use(10);
    dispensable.record_use(50);

    let score_critical = critical.attention_score(&config);
    let score_dispensable = dispensable.attention_score(&config);

    assert!(
        score_critical > score_dispensable,
        "Critical value must have higher score than dispensable. critical={:.3}, dispensable={:.3}",
        score_critical,
        score_dispensable
    );
}

#[test]
fn test_allocation_metrics_epistemic_quality() {
    use emit::AllocationMetrics;

    let mut metrics = AllocationMetrics::default();

    // Good policy: spill mostly low-confidence values
    metrics.record_spill(0.1, false); // low conf
    metrics.record_spill(0.2, false); // low conf
    metrics.record_spill(0.25, false); // low conf
    metrics.record_spill(0.8, true); // high conf (bad)

    let quality = metrics.epistemic_quality();
    assert!(
        quality > 1.0,
        "Good epistemic policy should have quality > 1.0. Got: {}",
        quality
    );
}

#[test]
fn test_allocation_metrics_bad_policy() {
    use emit::AllocationMetrics;

    let mut metrics = AllocationMetrics::default();

    // Bad policy: spill mostly high-confidence values
    metrics.record_spill(0.9, true); // high conf (bad)
    metrics.record_spill(0.85, true); // high conf (bad)
    metrics.record_spill(0.8, true); // high conf (bad)
    metrics.record_spill(0.1, false); // low conf (good)

    let quality = metrics.epistemic_quality();
    assert!(
        quality < 1.0,
        "Bad epistemic policy should have quality < 1.0. Got: {}",
        quality
    );
}

#[test]
fn test_live_interval_uncertainty() {
    use emit::LiveInterval;
    use values::ValueId;

    let mut interval = LiveInterval::with_epistemic(
        ValueId::new(0),
        0,
        SirType::f64(),
        0.8, // confidence
        true,
    );

    // Default uncertainty is 1 - confidence
    assert!((interval.uncertainty - 0.2).abs() < 0.01);

    // Can set explicit uncertainty
    interval.set_uncertainty(0.5);
    assert_eq!(interval.uncertainty, 0.5);
}
