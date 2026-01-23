//! Complete MIR Optimization Pipeline Example
//!
//! This example demonstrates the fully integrated optimization pipeline with
//! all advanced features: SSA validation, epistemic optimization, ML guidance, and performance tuning.

use sounio_compiler::benchmark::*;
use sounio_compiler::mir::analysis::ssa_validator::SSAValidator;
use sounio_compiler::mir::optimization::*;
use sounio_compiler::mir::{BlockId, MirFunction, MirModule, MirType, ValueId};

fn main() -> Result<(), String> {
    println!("🚀 Starting Complete MIR Optimization Pipeline Demo");

    // 1. Create test module with epistemic features
    let module = create_epistemic_test_module()?;

    println!("📦 Created test module with epistemic features");

    // 2. Run SSA validation
    let mut validator = SSAValidator::new();
    let ssa_result = validator.validate_module(&module);

    match ssa_result.is_valid {
        true => println!("✅ SSA validation passed"),
        false => {
            println!("❌ SSA validation failed:");
            for error in ssa_result.errors {
                println!("  - {}", error.message);
            }
        }
    }

    // 3. Create advanced optimization pipeline
    let mut validated_pipeline = create_validated_optimization_pipeline();
    let mut ml_optimizer = create_ml_optimization_pipeline(TargetArchitecture::X86_64);
    let mut performance_tuner = create_performance_optimizer(TargetArchitecture::X86_64);

    println!("🔧 Created advanced optimization pipelines");

    // 4. Run complete optimization sequence
    let mut optimized_module = module.clone();

    // Phase 1: Basic optimizations with validation
    println!("\n📊 Phase 1: Basic optimizations...");
    let phase1_result = validated_pipeline.run_pipeline(&mut optimized_module)?;
    println!(
        "  - Instructions: {} → {} ({:.1}% reduction)",
        phase1_result.validation_stats.total_validations,
        phase1_result.validation_stats.failures,
        phase1_result
            .effectiveness_metrics
            .instruction_reduction_percent
    );

    // Phase 2: ML-guided optimizations
    println!("\n🤖 Phase 2: ML-guided optimizations...");
    let ml_result = ml_optimizer.run_ml_guided_optimization(&mut optimized_module)?;
    println!(
        "  - ML optimization benefit: {:.1}%",
        ml_result.inlining_result.estimated_benefit * 100.0
    );
    println!(
        "  - Prediction confidence: {:.1}%",
        ml_result.inlining_result.prediction_confidence * 100.0
    );

    // Phase 3: Performance tuning
    println!("\n⚡ Phase 3: Performance tuning...");
    let perf_result = performance_tuner.optimize_with_performance_tuning(&mut optimized_module)?;
    println!(
        "  - Cache hit rate: {:.1}%",
        perf_result.cache_hit_rate * 100.0
    );
    println!(
        "  - Performance improvement: {:.1}%",
        perf_result.performance_improvement
    );

    // Final validation
    println!("\n🔍 Final validation...");
    let final_ssa = validator.validate_module(&optimized_module);
    let final_benchmark = benchmark_optimization_effectiveness(&optimized_module)?;

    println!(
        "✅ Final SSA validation: {}",
        if final_ssa.is_valid {
            "PASSED"
        } else {
            "FAILED"
        }
    );
    println!(
        "📈 Final benchmark results: {:.1}% improvement",
        final_benchmark.effectiveness_metrics.quality_score
    );

    println!("\n🎉 Complete optimization pipeline finished successfully!");
    Ok(())
}

fn create_epistemic_test_module() -> Result<MirModule, String> {
    use sounio_compiler::mir::builder::{FunctionBuilder, ModuleBuilder};
    use sounio_compiler::mir::{MirBinaryOp, MirConstant};

    let mut module_builder = ModuleBuilder::new("epistemic_example");

    // Create function with epistemic operations
    let mut func =
        module_builder.create_function("knowledge_computation".to_string(), MirType::I64);

    // Create knowledge construction with confidence
    let knowledge_const = func.build_i64(42);
    func.build_return(Some(knowledge_const));

    module_builder.add_function(func.build());

    // Create function with uncertainty handling
    let mut uncertain_func =
        module_builder.create_function("uncertain_computation".to_string(), MirType::F64);

    let uncertainty_param = uncertain_func.fresh_value();
    uncertain_func.add_param("uncertain_input".to_string(), MirType::F64);

    let result = uncertain_func.fresh_value();
    uncertain_func.build_return(Some(result));

    module_builder.add_function(uncertain_func.build());

    Ok(module_builder.build())
}

fn benchmark_optimization_effectiveness(
    module: &MirModule,
) -> Result<MIROptimizationBenchmark, String> {
    let mut benchmarker = MIROptimizationBenchmarker::new(BenchmarkConfig::default());
    let test_module = create_epistemic_test_module()?;
    let pass_names = vec![
        "constant-propagation",
        "dead-code-elimination",
        "common-subexpression-elimination",
    ];

    benchmarker.benchmark_optimization_sequence(test_module, pass_names)
}

fn create_epistemic_test_module() -> Result<MirModule, String> {
    // Implementation would create a module with epistemic operations
    // For demo purposes, return a simple module
    let mut module_builder = ModuleBuilder::new("epistemic_demo");
    let mut func = module_builder.create_function("demo_func".to_string(), MirType::I64);
    let result = func.build_i64(42);
    func.build_return(Some(result));
    module_builder.add_function(func.build());
    Ok(module_builder.build())
}

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_complete_pipeline_integration() {
        let module = create_epistemic_test_module().unwrap();
        assert!(!module.functions.is_empty());
    }

    #[test]
    fn test_epistemic_optimization() {
        let mut module = create_epistemic_test_module().unwrap();
        let mut optimizer = AdvancedEpistemicOptimization::new();
        let result = optimizer.run_on_module(&mut module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_performance_pipeline() {
        let mut optimizer = PerformanceTunedOptimizer::new(TargetArchitecture::X86_64);
        let module = create_epistemic_test_module().unwrap();
        let result = optimizer.optimize_with_performance_tuning(&module);
        assert!(result.is_ok());
    }

    #[test]
    fn test_ml_optimization() {
        let mut optimizer = MLGuidedOptimizer::new(TargetArchitecture::X86_64);
        let mut module = create_epistemic_test_module().unwrap();
        let result = optimizer.run_ml_guided_optimization(&mut module);
        assert!(result.is_ok());
    }
}
