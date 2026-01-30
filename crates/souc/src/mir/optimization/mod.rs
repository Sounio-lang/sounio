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
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            level: OptimizationLevel::O2,
        }
    }

    pub fn new_with_level(level: OptimizationLevel) -> Self {
        Self { level }
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
        }

        Ok(PassResult {
            modified: total_modified,
            passes_applied: vec![],
            instructions_reduced: 0,
        })
    }
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
