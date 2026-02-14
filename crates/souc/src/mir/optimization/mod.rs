//! MIR Optimization Module
//!
//! This module provides optimization passes for MIR code.

pub mod common_subexpression_elimination;
pub mod constant_propagation;
pub mod dead_code_elimination;
pub mod function_inlining;
pub mod pass_manager;
pub mod strength_reduction;

// Advanced optimization modules
pub mod advanced_epistemic_optimization;
pub mod function_inlining_complete;
pub mod ml_guided_optimization;
pub mod performance_tuning;
pub mod pipeline_with_validation;

// Workstream 4: New optimization passes
pub mod alias_analysis;
pub mod licm;
pub mod sroa;

// Shared types for ML-guided and heuristic optimization (always available)
pub mod local_optimizer;
pub mod optimization_types;

// GLM-4.7 API-based ML-guided optimization
#[cfg(feature = "glm")]
pub mod glm_integration;

// Re-export commonly used optimization types
pub use common_subexpression_elimination::CommonSubexpressionElimination;
pub use constant_propagation::ConstantPropagation;
pub use dead_code_elimination::DeadCodeElimination;
pub use function_inlining::FunctionInlining;
pub use pass_manager::{AnalysisPass, MIRPass};
pub use strength_reduction::StrengthReduction;

// Advanced exports
pub use advanced_epistemic_optimization::AdvancedEpistemicOptimization;
pub use function_inlining_complete::CompleteFunctionInlining;
pub use ml_guided_optimization::{MLGuidedOptimizer, MLOptimizationResult, TargetArchitecture};
pub use performance_tuning::{PerformanceResult, PerformanceTunedOptimizer};
pub use pipeline_with_validation::{PipelineResult, ValidatedOptimizationPipeline};

// Workstream 4 exports: LICM, Alias Analysis, SROA
pub use alias_analysis::{AliasAnalysis, AliasAnalysisResult, AliasQuery, AliasResult};
pub use licm::LoopInvariantCodeMotion;
pub use sroa::Sroa;

// Shared optimization types (always available — no feature gate)
pub use local_optimizer::{DataCollector, HeuristicOptimizer};
pub use optimization_types::{
    BlockFeatures, CodeFeatures, OptimizationSuggestion, OptimizationType,
};

// GLM-4.7 exports (only GLMConfig and GLMManager; types come from optimization_types)
#[cfg(feature = "glm")]
pub use glm_integration::{GLMConfig, GLMManager};

/// Create a default optimization pipeline with validation
pub fn create_validated_optimization_pipeline() -> ValidatedOptimizationPipeline {
    ValidatedOptimizationPipeline::new()
}

/// Create an advanced ML-guided optimization pipeline
pub fn create_ml_optimization_pipeline(target_arch: TargetArchitecture) -> MLGuidedOptimizer {
    MLGuidedOptimizer::new(target_arch)
}

/// Create a performance-tuned optimizer
pub fn create_performance_optimizer(
    target_arch: performance_tuning::TargetArchitecture,
) -> PerformanceTunedOptimizer {
    PerformanceTunedOptimizer::new(target_arch)
}

/// Create a complete function inliner
pub fn create_complete_inliner() -> CompleteFunctionInlining {
    CompleteFunctionInlining::new()
}

/// Create an advanced epistemic optimizer
pub fn create_epistemic_optimizer() -> AdvancedEpistemicOptimization {
    AdvancedEpistemicOptimization::new()
}

/// Create a default optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    O0,
    O1,
    O2,
    O3,
}

/// Pass manager for running optimization passes
pub struct PassManager {
    level: OptimizationLevel,
    glm_enabled: bool,
    ml_opt_enabled: bool,
    collect_opt_data: bool,
    data_collector: Option<local_optimizer::DataCollector>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            level: OptimizationLevel::O2,
            glm_enabled: false,
            ml_opt_enabled: false,
            collect_opt_data: false,
            data_collector: None,
        }
    }

    pub fn new_with_level(level: OptimizationLevel) -> Self {
        Self {
            level,
            glm_enabled: false,
            ml_opt_enabled: false,
            collect_opt_data: false,
            data_collector: None,
        }
    }

    pub fn new_with_glm(level: OptimizationLevel, glm_enabled: bool) -> Self {
        Self {
            level,
            glm_enabled,
            ml_opt_enabled: false,
            collect_opt_data: false,
            data_collector: None,
        }
    }

    /// Create a pass manager with local ML-guided optimization
    pub fn new_with_ml_opt(level: OptimizationLevel, ml_opt: bool, collect_data: bool) -> Self {
        Self {
            level,
            glm_enabled: false,
            ml_opt_enabled: ml_opt,
            collect_opt_data: collect_data,
            data_collector: if collect_data {
                Some(local_optimizer::DataCollector::new())
            } else {
                None
            },
        }
    }

    pub fn run_function_passes(
        &mut self,
        func: &mut crate::mir::MirFunction,
    ) -> Result<PassResult, String> {
        let mut total_modified = false;

        // Apply constant propagation for all levels
        let cp = ConstantPropagation::new();
        if cp.run_on_function(func)? {
            total_modified = true;
        }

        // Apply other optimizations based on level
        match self.level {
            OptimizationLevel::O1 | OptimizationLevel::O2 | OptimizationLevel::O3 => {
                let dce = DeadCodeElimination::new();
                if dce.run_on_function(func)? {
                    total_modified = true;
                }
            }
            _ => {}
        }

        match self.level {
            OptimizationLevel::O2 | OptimizationLevel::O3 => {
                let cse = CommonSubexpressionElimination::new();
                if cse.run_on_function(func)? {
                    total_modified = true;
                }

                // SROA: Scalar Replacement of Aggregates
                // Runs at O2+ to reduce memory traffic for aggregate types
                let sroa = Sroa::new();
                if sroa.run_on_function(func)? {
                    total_modified = true;
                }

                // LICM: Loop Invariant Code Motion
                // Runs at O2+ to hoist loop-invariant computations
                let licm = LoopInvariantCodeMotion;
                if licm.run_on_function(func)? {
                    total_modified = true;
                }
            }
            _ => {}
        }

        if self.level == OptimizationLevel::O3 {
            let sr = StrengthReduction::new();
            if sr.run_on_function(func)? {
                total_modified = true;
            }

            let fi = FunctionInlining::new();
            if fi.run_on_function(func)? {
                total_modified = true;
            }

            // Loop vectorization: widen simple array-processing loops to SIMD width
            let mut vectorizer =
                performance_tuning::LoopVectorizer::new();
            if vectorizer.run(func)? {
                total_modified = true;
            }
        }

        Ok(PassResult {
            modified: total_modified,
            passes_applied: vec![],
            instructions_reduced: 0,
        })
    }

    /// Run standard passes plus local heuristic ML-guided suggestions (no feature gate)
    pub fn run_function_passes_with_ml_opt(
        &mut self,
        func: &mut crate::mir::MirFunction,
        module: &crate::mir::MirModule,
    ) -> Result<PassResult, String> {
        let mut result = self.run_function_passes(func)?;

        if self.ml_opt_enabled
            && (self.level == OptimizationLevel::O2 || self.level == OptimizationLevel::O3)
        {
            let features = optimization_types::extract_features(module, &func.name);
            let optimizer = HeuristicOptimizer::new(0.7);
            let suggestions = optimizer.suggest(&features, &func.name);

            for suggestion in &suggestions {
                let before = count_instructions(func);
                tracing::info!(
                    "ML-opt suggests {:?} on '{}' (confidence: {:.2}): {}",
                    suggestion.optimization_type,
                    suggestion.target,
                    suggestion.confidence,
                    suggestion.reasoning
                );
                match self.apply_suggestion(func, suggestion) {
                    Ok(modified) => {
                        if modified {
                            result.modified = true;
                        }
                        if self.collect_opt_data {
                            let after = count_instructions(func);
                            if let Some(ref mut collector) = self.data_collector {
                                collector.collect_pass_result(
                                    &func.name,
                                    &features,
                                    &format!("{:?}", suggestion.optimization_type),
                                    modified,
                                    before,
                                    after,
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::debug!(
                            "ML-opt suggestion {:?} failed: {}",
                            suggestion.optimization_type,
                            e
                        );
                    }
                }
            }
        }

        Ok(result)
    }

    /// Run standard passes plus GLM-4.7 ML-guided suggestions
    #[cfg(feature = "glm")]
    pub fn run_function_passes_with_glm(
        &mut self,
        func: &mut crate::mir::MirFunction,
        module: &crate::mir::MirModule,
    ) -> Result<PassResult, String> {
        let mut result = self.run_function_passes(func)?;

        if self.glm_enabled
            && (self.level == OptimizationLevel::O2 || self.level == OptimizationLevel::O3)
        {
            match self.apply_glm_suggestions(func, module) {
                Ok(true) => {
                    result.modified = true;
                }
                Ok(false) => {}
                Err(e) => {
                    tracing::warn!("GLM optimization skipped: {}", e);
                }
            }
        }

        Ok(result)
    }

    #[cfg(feature = "glm")]
    fn apply_glm_suggestions(
        &mut self,
        func: &mut crate::mir::MirFunction,
        module: &crate::mir::MirModule,
    ) -> Result<bool, String> {
        let config = glm_integration::GLMConfig::default();
        if config.api_key.is_empty() {
            tracing::warn!("GLM_API_KEY not set, skipping GLM optimization");
            return Ok(false);
        }

        let mut manager = glm_integration::GLMManager::new(config);

        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| format!("Tokio runtime init failed: {}", e))?;

        let suggestions = rt.block_on(manager.analyze_and_suggest(module, &func.name))?;

        let mut modified = false;
        for suggestion in &suggestions {
            if suggestion.confidence < 0.7 {
                tracing::debug!(
                    "GLM suggestion {:?} skipped (confidence {:.2} < 0.7)",
                    suggestion.optimization_type,
                    suggestion.confidence
                );
                continue;
            }
            tracing::info!(
                "GLM suggests {:?} on '{}' (confidence: {:.2}): {}",
                suggestion.optimization_type,
                suggestion.target,
                suggestion.confidence,
                suggestion.reasoning
            );
            match self.apply_suggestion(func, suggestion) {
                Ok(true) => {
                    modified = true;
                }
                Ok(false) => {}
                Err(e) => {
                    tracing::debug!(
                        "GLM suggestion {:?} failed: {}",
                        suggestion.optimization_type,
                        e
                    );
                }
            }
        }

        Ok(modified)
    }

    /// Apply a single optimization suggestion (shared by both local and GLM paths)
    fn apply_suggestion(
        &self,
        func: &mut crate::mir::MirFunction,
        suggestion: &optimization_types::OptimizationSuggestion,
    ) -> Result<bool, String> {
        use optimization_types::OptimizationType as OT;
        match suggestion.optimization_type {
            OT::ConstantPropagation => ConstantPropagation::new().run_on_function(func),
            OT::DeadCodeElimination => DeadCodeElimination::new().run_on_function(func),
            OT::CommonSubexpressionElimination => {
                CommonSubexpressionElimination::new().run_on_function(func)
            }
            OT::StrengthReduction => StrengthReduction::new().run_on_function(func),
            OT::FunctionInlining => FunctionInlining::new().run_on_function(func),
            OT::LoopInvariantCodeMotion => LoopInvariantCodeMotion.run_on_function(func),
            OT::ScalarReplacementOfAggregates => Sroa::new().run_on_function(func),
            OT::LoopVectorization => {
                let mut vectorizer = performance_tuning::LoopVectorizer::new();
                vectorizer.run(func)
            }
            _ => {
                tracing::debug!(
                    "Suggested {:?} but no pass implements it yet",
                    suggestion.optimization_type
                );
                Ok(false)
            }
        }
    }

    /// Take the data collector (transfers ownership for writing to file)
    pub fn take_data_collector(&mut self) -> Option<local_optimizer::DataCollector> {
        self.data_collector.take()
    }
}

/// Count total instructions across all blocks in a function
fn count_instructions(func: &crate::mir::MirFunction) -> usize {
    func.blocks.iter().map(|b| b.instructions.len()).sum()
}

/// Result from running optimization passes
#[derive(Debug, Clone)]
pub struct PassResult {
    pub modified: bool,
    pub passes_applied: Vec<PassInfo>,
    pub instructions_reduced: usize,
}

/// Information about a pass that was applied
#[derive(Debug, Clone)]
pub struct PassInfo {
    pub name: String,
    pub time_ms: f64,
    pub modified: bool,
}

// Workstream 4: Factory functions for new optimization passes

/// Create a Loop Invariant Code Motion (LICM) pass
pub fn create_licm_pass() -> LoopInvariantCodeMotion {
    LoopInvariantCodeMotion
}

/// Create a Scalar Replacement of Aggregates (SROA) pass
pub fn create_sroa_pass() -> Sroa {
    Sroa::new()
}

/// Create an Alias Analysis pass
pub fn create_alias_analysis() -> AliasAnalysis {
    AliasAnalysis::new()
}

/// Create a Loop Vectorization pass
pub fn create_loop_vectorizer() -> performance_tuning::LoopVectorizer {
    performance_tuning::LoopVectorizer::new()
}
