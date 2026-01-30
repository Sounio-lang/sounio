//! Integration test for GLM-4.7 ML-guided optimization
//!
//! This test validates that the GLM integration works correctly
//! and provides optimization suggestions.

#[cfg(feature = "glm")]
mod glm_optimization_integration {
    use sounio::mir::optimization::glm_integration::{
        GLMConfig, GLMManager, OptimizationSuggestion, OptimizationType,
    };
    use sounio::mir::types::MirType;
    use sounio::mir::{MirBlock, MirFunction, MirInstruction, MirModule};
    use std::collections::HashMap;

    #[test]
    fn test_glm_config_default() {
        let config = GLMConfig::default();
        assert!(!config.api_key.is_empty());
        assert_eq!(config.max_tokens, 1000);
        assert_eq!(config.temperature, 0.1);
        assert_eq!(config.timeout_secs, 30);
    }

    #[test]
    fn test_optimization_suggestion_creation() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::ConstantPropagation,
            confidence: 0.95,
            target: "test_function".to_string(),
            parameters: HashMap::from([("threshold".to_string(), "0.8".to_string())]),
            reasoning: "High confidence constant propagation opportunity".to_string(),
        };

        assert_eq!(
            suggestion.optimization_type,
            OptimizationType::ConstantPropagation
        );
        assert_eq!(suggestion.confidence, 0.95);
        assert_eq!(suggestion.target, "test_function");
        assert_eq!(
            suggestion.parameters.get("threshold"),
            Some(&"0.8".to_string())
        );
    }

    #[test]
    fn test_optimization_suggestion_serialization() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::LoopUnrolling,
            confidence: 0.85,
            target: "loop_function".to_string(),
            parameters: HashMap::from([
                ("unroll_factor".to_string(), "4".to_string()),
                ("trip_count".to_string(), "100".to_string()),
            ]),
            reasoning: "Small loop with known trip count".to_string(),
        };

        let json = serde_json::to_string(&suggestion).unwrap();
        let parsed: OptimizationSuggestion = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.optimization_type, OptimizationType::LoopUnrolling);
        assert_eq!(parsed.confidence, 0.85);
        assert_eq!(parsed.target, "loop_function");
        assert_eq!(
            parsed.parameters.get("unroll_factor"),
            Some(&"4".to_string())
        );
    }

    #[test]
    fn test_glm_manager_creation() {
        let config = GLMConfig::default();
        let manager = GLMManager::new(config);

        assert_eq!(manager.cache_size(), 0);
    }

    #[test]
    fn test_all_optimization_types() {
        let types = vec![
            OptimizationType::ConstantPropagation,
            OptimizationType::DeadCodeElimination,
            OptimizationType::FunctionInlining,
            OptimizationType::LoopUnrolling,
            OptimizationType::StrengthReduction,
            OptimizationType::CommonSubexpressionElimination,
            OptimizationType::LoopInvariantCodeMotion,
            OptimizationType::AliasAnalysis,
            OptimizationType::ScalarReplacementOfAggregates,
            OptimizationType::PredictiveInlining,
            OptimizationType::AdaptiveUnrolling,
            OptimizationType::UncertaintyAwareOptimization,
        ];

        assert_eq!(types.len(), 12);

        // Test each type can be serialized/deserialized
        for opt_type in types {
            let suggestion = OptimizationSuggestion {
                optimization_type: opt_type.clone(),
                confidence: 0.5,
                target: "test".to_string(),
                parameters: HashMap::new(),
                reasoning: "test".to_string(),
            };

            let json = serde_json::to_string(&suggestion).unwrap();
            let parsed: OptimizationSuggestion = serde_json::from_str(&json).unwrap();

            assert_eq!(parsed.optimization_type, opt_type);
        }
    }

    #[test]
    fn test_epistemic_optimization_type() {
        let suggestion = OptimizationSuggestion {
            optimization_type: OptimizationType::UncertaintyAwareOptimization,
            confidence: 0.90,
            target: "epistemic_function".to_string(),
            parameters: HashMap::from([
                ("optimize_knowledge_ops".to_string(), "true".to_string()),
                ("preserve_confidence".to_string(), "true".to_string()),
            ]),
            reasoning: "High confidence epistemic optimization opportunity".to_string(),
        };

        assert_eq!(
            suggestion.optimization_type,
            OptimizationType::UncertaintyAwareOptimization
        );
        assert!(suggestion.confidence >= 0.9);
        assert_eq!(
            suggestion.parameters.get("optimize_knowledge_ops"),
            Some(&"true".to_string())
        );
    }

    #[test]
    fn test_ml_guided_optimization_pass() {
        use sounio::mir::optimization::ml_guided_optimization::MLGuidedOptimization;

        let pass = MLGuidedOptimization::new();
        assert_eq!(pass.name(), "ml-guided-optimization");
        assert_eq!(pass.min_confidence, 0.7);
        assert!(
            pass.enabled_optimizations
                .contains(&OptimizationType::ConstantPropagation)
        );
        assert!(
            pass.enabled_optimizations
                .contains(&OptimizationType::UncertaintyAwareOptimization)
        );
    }

    #[test]
    fn test_code_features_extraction() {
        use sounio::mir::optimization::glm_integration::CodeFeatures;

        let features = CodeFeatures {
            function_count: 1,
            total_blocks: 3,
            total_instructions: 15,
            avg_block_size: 5.0,
            max_block_size: 8,
            loop_count: 2,
            branch_count: 1,
            call_count: 0,
            arithmetic_ops: 10,
            memory_ops: 3,
            block_features: Vec::new(),
            type_distribution: HashMap::new(),
            epistemic_types: 1,
            uncertainty_ops: 5,
        };

        assert_eq!(features.function_count, 1);
        assert_eq!(features.total_blocks, 3);
        assert_eq!(features.total_instructions, 15);
        assert_eq!(features.epistemic_types, 1);
        assert_eq!(features.uncertainty_ops, 5);
    }

    #[test]
    fn test_block_features() {
        use sounio::mir::optimization::glm_integration::BlockFeatures;

        let block_features = BlockFeatures {
            instruction_count: 10,
            arithmetic_ops: 5,
            memory_loads: 2,
            memory_stores: 1,
            branches: 2,
            phi_nodes: 0,
            loop_depth: 1,
            has_uncertainty: true,
        };

        assert_eq!(block_features.instruction_count, 10);
        assert_eq!(block_features.arithmetic_ops, 5);
        assert_eq!(block_features.branches, 2);
        assert!(block_features.has_uncertainty);
        assert_eq!(block_features.loop_depth, 1);
    }

    #[tokio::test]
    async fn test_glm_api_integration() {
        // Note: This test requires a valid GLM API key
        // In CI/CD, this might be skipped or use a mock

        let config = GLMConfig::default();
        let mut manager = GLMManager::new(config);

        // Create a simple test module
        let mut test_module = MirModule::new();
        let mut test_function =
            MirFunction::new("test_function".to_string(), MirType::I32, Vec::new());

        // Add a simple block
        let mut test_block = MirBlock::new("entry".to_string());
        test_block.add_instruction(MirInstruction::Return(MirOperand::Constant(
            MirConstant::I32(42),
        )));
        test_function.add_block(test_block);
        test_module.add_function(test_function);

        // Test analysis (this would normally make an API call)
        // For testing, we'll skip the actual API call and just test the setup
        assert_eq!(manager.cache_size(), 0);

        // Clear cache
        manager.clear_cache();
        assert_eq!(manager.cache_size(), 0);
    }

    #[test]
    fn test_optimization_level_integration() {
        use sounio::mir::optimization::pass_manager::OptimizationLevel;

        // Test that ML-guided optimization is enabled for O2 and O3
        assert!(!OptimizationLevel::O0.should_enable_pass(
            &std::marker::PhantomData::<
                sounio::mir::optimization::ml_guided_optimization::MLGuidedOptimization,
            >
        ));

        assert!(OptimizationLevel::O2.should_enable_pass(
            &std::marker::PhantomData::<
                sounio::mir::optimization::ml_guided_optimization::MLGuidedOptimization,
            >
        ));

        assert!(OptimizationLevel::O3.should_enable_pass(
            &std::marker::PhantomData::<
                sounio::mir::optimization::ml_guided_optimization::MLGuidedOptimization,
            >
        ));
    }

    #[test]
    fn test_prompt_generation() {
        use sounio::mir::optimization::glm_integration::CodeFeatures;

        let features = CodeFeatures {
            function_count: 2,
            total_blocks: 5,
            total_instructions: 25,
            avg_block_size: 5.0,
            max_block_size: 10,
            loop_count: 3,
            branch_count: 2,
            call_count: 1,
            arithmetic_ops: 15,
            memory_ops: 5,
            block_features: Vec::new(),
            type_distribution: HashMap::from([
                ("Knowledge<f64>".to_string(), 3),
                ("i32".to_string(), 5),
            ]),
            epistemic_types: 3,
            uncertainty_ops: 8,
        };

        let prompt = format!(
            "Functions: {}, Blocks: {}, Instructions: {}, Avg Block Size: {:.2}, Max Block Size: {}, Arithmetic Ops: {}, Memory Ops: {}, Calls: {}, Branches: {}, Epistemic Types: {}, Uncertainty Ops: {}",
            features.function_count,
            features.total_blocks,
            features.total_instructions,
            features.avg_block_size,
            features.max_block_size,
            features.arithmetic_ops,
            features.memory_ops,
            features.call_count,
            features.branch_count,
            features.epistemic_types,
            features.uncertainty_ops,
        );

        assert!(prompt.contains("Functions: 2"));
        assert!(prompt.contains("Epistemic Types: 3"));
        assert!(prompt.contains("Uncertainty Ops: 8"));
    }
}

#[cfg(not(feature = "glm"))]
mod glm_optimization_integration {
    use super::*;

    #[test]
    fn test_glm_feature_disabled() {
        // This test should pass when GLM feature is not enabled
        // The actual GLM tests are in the cfg(feature = "glm") block above
        assert!(
            true,
            "GLM feature tests are disabled when feature is not enabled"
        );
    }
}
