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
pub mod incremental_query;

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
pub use incremental_query::{IncrementalFunctionOptimizer, IncrementalOptimizationOutput};

// Shared optimization types (always available — no feature gate)
pub use local_optimizer::{DataCollector, HeuristicOptimizer};
pub use optimization_types::{
    BlockFeatures, CodeFeatures, OptimizationSuggestion, OptimizationType,
};

// GLM-4.7 exports (only GLMConfig and GLMManager; types come from optimization_types)
#[cfg(feature = "glm")]
pub use glm_integration::{GLMConfig, GLMManager};

use std::collections::HashSet;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

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

#[derive(Debug, Clone, Default)]
pub struct PassManagerPolicyContext {
    pub mir_allowed_actions: Option<HashSet<String>>,
    pub policy_error: Option<String>,
    pub decision_trail_path: Option<PathBuf>,
    pub decision_trail_required: bool,
    pub policy_id: Option<String>,
    pub policy_mode: Option<String>,
}

/// Pass manager for running optimization passes
pub struct PassManager {
    level: OptimizationLevel,
    glm_enabled: bool,
    ml_opt_enabled: bool,
    collect_opt_data: bool,
    data_collector: Option<local_optimizer::DataCollector>,
    mir_allowed_actions: Option<HashSet<String>>,
    policy_error: Option<String>,
    decision_trail_path: Option<PathBuf>,
    decision_trail_required: bool,
    policy_id: Option<String>,
    policy_mode: Option<String>,
}

impl PassManager {
    fn from_context(
        level: OptimizationLevel,
        glm_enabled: bool,
        ml_opt_enabled: bool,
        collect_opt_data: bool,
        context: PassManagerPolicyContext,
    ) -> Self {
        Self {
            level,
            glm_enabled,
            ml_opt_enabled,
            collect_opt_data,
            data_collector: if collect_opt_data {
                Some(local_optimizer::DataCollector::new())
            } else {
                None
            },
            mir_allowed_actions: context.mir_allowed_actions,
            policy_error: context.policy_error,
            decision_trail_path: context.decision_trail_path,
            decision_trail_required: context.decision_trail_required,
            policy_id: context.policy_id,
            policy_mode: context.policy_mode,
        }
    }

    pub fn new() -> Self {
        Self::from_context(
            OptimizationLevel::O2,
            false,
            false,
            false,
            default_policy_context(),
        )
    }

    pub fn new_with_level(level: OptimizationLevel) -> Self {
        Self::from_context(level, false, false, false, default_policy_context())
    }

    pub fn new_with_glm(level: OptimizationLevel, glm_enabled: bool) -> Self {
        Self::new_with_glm_and_context(level, glm_enabled, default_policy_context())
    }

    pub fn new_with_glm_and_context(
        level: OptimizationLevel,
        glm_enabled: bool,
        context: PassManagerPolicyContext,
    ) -> Self {
        Self::from_context(level, glm_enabled, false, false, context)
    }

    /// Create a pass manager with local ML-guided optimization
    pub fn new_with_ml_opt(level: OptimizationLevel, ml_opt: bool, collect_data: bool) -> Self {
        Self::new_with_ml_opt_and_context(level, ml_opt, collect_data, default_policy_context())
    }

    pub fn new_with_ml_opt_and_context(
        level: OptimizationLevel,
        ml_opt: bool,
        collect_data: bool,
        context: PassManagerPolicyContext,
    ) -> Self {
        Self::from_context(level, false, ml_opt, collect_data, context)
    }

    fn emit_mir_decision(
        &self,
        function: &str,
        action: &str,
        status: &str,
        reason: &str,
    ) -> Result<(), String> {
        emit_mir_decision(
            function,
            action,
            status,
            reason,
            self.decision_trail_path.as_ref(),
            self.decision_trail_required,
            self.policy_id.as_deref(),
            self.policy_mode.as_deref(),
        )
    }

    fn mir_action_allowed(&self, action: &str) -> bool {
        self.mir_allowed_actions
            .as_ref()
            .map(|set| set.contains(action))
            .unwrap_or(true)
    }

    fn run_guarded_pass<F>(
        &self,
        func: &mut crate::mir::MirFunction,
        action: &'static str,
        run_pass: F,
        total_modified: &mut bool,
    ) -> Result<(), String>
    where
        F: FnOnce(&mut crate::mir::MirFunction) -> Result<bool, String>,
    {
        if !self.mir_action_allowed(action) {
            self.emit_mir_decision(&func.name, action, "skipped", "blocked_by_policy_allowlist")?;
            return Ok(());
        }

        let modified = run_pass(func)?;
        if modified {
            *total_modified = true;
        }
        self.emit_mir_decision(
            &func.name,
            action,
            "applied",
            if modified {
                "pass_modified"
            } else {
                "pass_no_change"
            },
        )?;
        Ok(())
    }

    pub fn run_function_passes(
        &mut self,
        func: &mut crate::mir::MirFunction,
    ) -> Result<PassResult, String> {
        if let Some(err) = &self.policy_error {
            return Err(err.clone());
        }
        let mut total_modified = false;

        // Apply constant propagation for all levels
        let cp = ConstantPropagation::new();
        self.run_guarded_pass(
            func,
            "constant_propagation",
            |func| cp.run_on_function(func),
            &mut total_modified,
        )?;

        // Apply other optimizations based on level
        match self.level {
            OptimizationLevel::O1 | OptimizationLevel::O2 | OptimizationLevel::O3 => {
                let dce = DeadCodeElimination::new();
                self.run_guarded_pass(
                    func,
                    "dead_code_elimination",
                    |func| dce.run_on_function(func),
                    &mut total_modified,
                )?;
            }
            _ => {}
        }

        match self.level {
            OptimizationLevel::O2 | OptimizationLevel::O3 => {
                let cse = CommonSubexpressionElimination::new();
                self.run_guarded_pass(
                    func,
                    "common_subexpression_elimination",
                    |func| cse.run_on_function(func),
                    &mut total_modified,
                )?;

                // SROA: Scalar Replacement of Aggregates
                // Runs at O2+ to reduce memory traffic for aggregate types
                let sroa = Sroa::new();
                self.run_guarded_pass(
                    func,
                    "sroa",
                    |func| sroa.run_on_function(func),
                    &mut total_modified,
                )?;

                // LICM: Loop Invariant Code Motion
                // Runs at O2+ to hoist loop-invariant computations
                let licm = LoopInvariantCodeMotion;
                self.run_guarded_pass(
                    func,
                    "licm",
                    |func| licm.run_on_function(func),
                    &mut total_modified,
                )?;
            }
            _ => {}
        }

        if self.level == OptimizationLevel::O3 {
            let sr = StrengthReduction::new();
            self.run_guarded_pass(
                func,
                "strength_reduction",
                |func| sr.run_on_function(func),
                &mut total_modified,
            )?;

            let fi = FunctionInlining::new();
            self.run_guarded_pass(
                func,
                "function_inlining",
                |func| fi.run_on_function(func),
                &mut total_modified,
            )?;

            // Loop vectorization: widen simple array-processing loops to SIMD width
            let mut vectorizer = performance_tuning::LoopVectorizer::new();
            self.run_guarded_pass(
                func,
                "loop_vectorization",
                |func| vectorizer.run(func),
                &mut total_modified,
            )?;
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
                        if e.starts_with("decision_trail_") {
                            return Err(e);
                        }
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
                    if e.starts_with("decision_trail_") {
                        return Err(e);
                    }
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
        let action_name = mir_action_for_optimization_type(suggestion.optimization_type.clone());
        if !self.mir_action_allowed(action_name) {
            self.emit_mir_decision(
                &func.name,
                action_name,
                "skipped",
                "blocked_by_policy_allowlist",
            )?;
            return Ok(false);
        }

        let apply_result = match suggestion.optimization_type {
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
        };

        match &apply_result {
            Ok(true) => self.emit_mir_decision(&func.name, action_name, "applied", "modified"),
            Ok(false) => self.emit_mir_decision(&func.name, action_name, "applied", "no_change"),
            Err(err) => self.emit_mir_decision(
                &func.name,
                action_name,
                "error",
                &format!("apply_error:{}", err),
            ),
        }?;

        apply_result
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

fn decision_trail_path_from_env() -> Option<PathBuf> {
    std::env::var_os("SOUNIO_OPT_DECISION_TRAIL_PATH")
        .filter(|raw| !raw.is_empty())
        .map(PathBuf::from)
}

fn decision_trail_required_from_env() -> bool {
    std::env::var("SOUNIO_OPT_DECISION_TRAIL_REQUIRED")
        .ok()
        .map(|v| {
            let value = v.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
        })
        .unwrap_or(false)
}

fn default_policy_context() -> PassManagerPolicyContext {
    PassManagerPolicyContext {
        mir_allowed_actions: None,
        policy_error: None,
        decision_trail_path: decision_trail_path_from_env(),
        decision_trail_required: decision_trail_required_from_env(),
        policy_id: None,
        policy_mode: None,
    }
}

fn mir_action_for_optimization_type(ty: optimization_types::OptimizationType) -> &'static str {
    use optimization_types::OptimizationType as OT;
    match ty {
        OT::ConstantPropagation => "constant_propagation",
        OT::DeadCodeElimination => "dead_code_elimination",
        OT::CommonSubexpressionElimination => "common_subexpression_elimination",
        OT::StrengthReduction => "strength_reduction",
        OT::FunctionInlining => "function_inlining",
        OT::LoopInvariantCodeMotion => "licm",
        OT::ScalarReplacementOfAggregates => "sroa",
        OT::LoopVectorization => "loop_vectorization",
        OT::UncertaintyAwareOptimization => "uncertainty_aware_optimization",
        OT::AdaptiveUnrolling => "adaptive_unrolling",
        OT::LoopUnrolling => "loop_unrolling",
        OT::AliasAnalysis => "alias_analysis",
        OT::PredictiveInlining => "predictive_inlining",
    }
}

fn emit_mir_decision(
    function: &str,
    action: &str,
    status: &str,
    reason: &str,
    decision_trail_path: Option<&PathBuf>,
    decision_trail_required: bool,
    policy_id: Option<&str>,
    policy_mode: Option<&str>,
) -> Result<(), String> {
    let Some(path) = decision_trail_path else {
        if decision_trail_required {
            return Err(
                "decision_trail_required: SOUNIO_OPT_DECISION_TRAIL_PATH is not configured"
                    .to_string(),
            );
        }
        return Ok(());
    };
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            if decision_trail_required {
                return Err(format!(
                    "decision_trail_io: failed creating {}: {}",
                    parent.display(),
                    err
                ));
            }
            return Ok(());
        }
    }

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    let record = serde_json::json!({
        "schema": "sounio.optimization.decision.v1",
        "timestamp_unix_ms": ts,
        "domain": "mir",
        "function": function,
        "action": action,
        "status": status,
        "reason": reason,
        "policy_id": policy_id.unwrap_or(""),
        "policy_mode": policy_mode.unwrap_or(""),
    });

    let line = match serde_json::to_string(&record) {
        Ok(json) => json,
        Err(err) => {
            if decision_trail_required {
                return Err(format!("decision_trail_json: {}", err));
            }
            return Ok(());
        }
    };

    let mut file = match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(file) => file,
        Err(err) => {
            if decision_trail_required {
                return Err(format!(
                    "decision_trail_io: failed opening {}: {}",
                    path.display(),
                    err
                ));
            }
            return Ok(());
        }
    };
    if let Err(err) = writeln!(file, "{}", line) {
        if decision_trail_required {
            return Err(format!(
                "decision_trail_io: failed writing {}: {}",
                path.display(),
                err
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::emit_mir_decision;

    #[test]
    fn emit_mir_decision_required_without_path_fails_closed() {
        let err = emit_mir_decision(
            "main",
            "constant_propagation",
            "applied",
            "pass_modified",
            None,
            true,
            None,
            None,
        )
        .expect_err("missing decision trail path should fail when required");
        assert!(
            err.contains("decision_trail_required"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn emit_mir_decision_without_path_succeeds_when_not_required() {
        emit_mir_decision(
            "main",
            "constant_propagation",
            "applied",
            "pass_modified",
            None,
            false,
            None,
            None,
        )
        .expect("best-effort decision trail should be non-fatal");
    }
}
