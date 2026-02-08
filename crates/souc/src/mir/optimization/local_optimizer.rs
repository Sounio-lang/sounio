//! Local heuristic optimizer for ML-guided pass selection
//!
//! Replaces the remote GLM-4.7 API with a deterministic, zero-latency
//! decision engine based on compiler optimization heuristics.
//!
//! Phase 1: Rule-based heuristics (this module)
//! Phase 2: Data collection for training (DataCollector)
//! Phase 3: Trained model inference via ONNX (future)

use super::optimization_types::{CodeFeatures, OptimizationSuggestion, OptimizationType};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Deterministic heuristic-based optimization pass selector
pub struct HeuristicOptimizer {
    confidence_threshold: f32,
}

impl HeuristicOptimizer {
    pub fn new(confidence_threshold: f32) -> Self {
        Self {
            confidence_threshold,
        }
    }

    /// Suggest optimization passes based on code features.
    /// Returns suggestions sorted by confidence (descending), filtered by threshold.
    pub fn suggest(
        &self,
        features: &CodeFeatures,
        function_name: &str,
    ) -> Vec<OptimizationSuggestion> {
        let mut suggestions = Vec::new();
        let target = function_name.to_string();

        // Rule 1: Always suggest CP and DCE — nearly always beneficial, no downside
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::ConstantPropagation,
            confidence: 0.95,
            target: target.clone(),
            parameters: HashMap::new(),
            reasoning: "Constant propagation is nearly always beneficial".into(),
        });
        suggestions.push(OptimizationSuggestion {
            optimization_type: OptimizationType::DeadCodeElimination,
            confidence: 0.95,
            target: target.clone(),
            parameters: HashMap::new(),
            reasoning: "Dead code elimination reduces code size with no cost".into(),
        });

        // Rule 2: Loop-bearing code with arithmetic → LICM
        if features.loop_count > 0 && features.arithmetic_ops > 5 {
            suggestions.push(OptimizationSuggestion {
                optimization_type: OptimizationType::LoopInvariantCodeMotion,
                confidence: 0.85,
                target: target.clone(),
                parameters: HashMap::new(),
                reasoning:
                    "Loop-bearing code with arithmetic operations benefits from hoisting invariants"
                        .into(),
            });
        }

        // Rule 3: Arithmetic-heavy functions → Strength Reduction
        if features.total_instructions > 0 {
            let arith_ratio = features.arithmetic_ops as f64 / features.total_instructions as f64;
            if features.total_instructions > 100 && arith_ratio > 0.3 {
                suggestions.push(OptimizationSuggestion {
                    optimization_type: OptimizationType::StrengthReduction,
                    confidence: 0.80,
                    target: target.clone(),
                    parameters: HashMap::new(),
                    reasoning: "High arithmetic density suggests strength reduction opportunities"
                        .into(),
                });
            }
        }

        // Rule 4: Many calls with small blocks → Function Inlining
        if features.call_count > 3 && features.avg_block_size < 20.0 {
            suggestions.push(OptimizationSuggestion {
                optimization_type: OptimizationType::FunctionInlining,
                confidence: 0.75,
                target: target.clone(),
                parameters: HashMap::new(),
                reasoning: "Multiple calls to small functions are good inlining candidates".into(),
            });
        }

        // Rule 5: Large basic blocks → CSE
        if features.max_block_size > 30 {
            suggestions.push(OptimizationSuggestion {
                optimization_type: OptimizationType::CommonSubexpressionElimination,
                confidence: 0.82,
                target: target.clone(),
                parameters: HashMap::new(),
                reasoning: "Large basic blocks likely contain redundant computations".into(),
            });
        }

        // Rule 6: Memory-heavy code → SROA
        if features.memory_ops > features.arithmetic_ops && features.memory_ops > 5 {
            suggestions.push(OptimizationSuggestion {
                optimization_type: OptimizationType::ScalarReplacementOfAggregates,
                confidence: 0.78,
                target: target.clone(),
                parameters: HashMap::new(),
                reasoning: "Memory-dominated code benefits from scalar replacement of aggregates"
                    .into(),
            });
        }

        // Rule 7: Epistemic types → Uncertainty-aware optimization
        if features.epistemic_types > 0 || features.uncertainty_ops > 0 {
            suggestions.push(OptimizationSuggestion {
                optimization_type: OptimizationType::UncertaintyAwareOptimization,
                confidence: 0.70,
                target: target.clone(),
                parameters: HashMap::new(),
                reasoning: "Epistemic types present; uncertainty-aware optimization may help"
                    .into(),
            });
        }

        // Rule 8: Block-level refinements
        for bf in &features.block_features {
            // Deep nested loops with heavy computation → adaptive unrolling
            if bf.loop_depth > 1 && bf.arithmetic_ops > 10 {
                suggestions.push(OptimizationSuggestion {
                    optimization_type: OptimizationType::AdaptiveUnrolling,
                    confidence: 0.72,
                    target: target.clone(),
                    parameters: HashMap::new(),
                    reasoning: "Deep loop nest with heavy computation is an unrolling candidate"
                        .into(),
                });
                break; // One suggestion per type is enough
            }
        }

        // Filter by threshold and sort by confidence
        suggestions.retain(|s| s.confidence >= self.confidence_threshold);
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        suggestions
    }
}

/// Record of a pass application for training data collection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRecord {
    pub function_name: String,
    pub features: CodeFeatures,
    pub pass_name: String,
    pub modified: bool,
    pub instructions_before: usize,
    pub instructions_after: usize,
}

/// Collects optimization pass results for future ML model training
pub struct DataCollector {
    records: Vec<DataRecord>,
}

impl DataCollector {
    pub fn new() -> Self {
        Self {
            records: Vec::new(),
        }
    }

    pub fn collect_pass_result(
        &mut self,
        function_name: &str,
        features: &CodeFeatures,
        pass_name: &str,
        modified: bool,
        instructions_before: usize,
        instructions_after: usize,
    ) {
        self.records.push(DataRecord {
            function_name: function_name.to_string(),
            features: features.clone(),
            pass_name: pass_name.to_string(),
            modified,
            instructions_before,
            instructions_after,
        });
    }

    pub fn write_to_file(&self, path: &std::path::Path) -> Result<(), String> {
        let json = serde_json::to_string_pretty(&self.records)
            .map_err(|e| format!("Failed to serialize optimization data: {}", e))?;
        std::fs::write(path, json).map_err(|e| {
            format!(
                "Failed to write optimization data to {}: {}",
                path.display(),
                e
            )
        })
    }

    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(
        total_instructions: usize,
        arithmetic_ops: usize,
        memory_ops: usize,
        loop_count: usize,
        call_count: usize,
        max_block_size: usize,
        avg_block_size: f64,
        epistemic_types: usize,
    ) -> CodeFeatures {
        CodeFeatures {
            function_count: 1,
            total_blocks: if total_instructions > 0 { 1 } else { 0 },
            total_instructions,
            avg_block_size,
            max_block_size,
            loop_count,
            branch_count: 0,
            call_count,
            arithmetic_ops,
            memory_ops,
            block_features: Vec::new(),
            type_distribution: HashMap::new(),
            epistemic_types,
            uncertainty_ops: 0,
        }
    }

    #[test]
    fn test_always_suggests_cp_and_dce() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(10, 2, 1, 0, 0, 10, 10.0, 0);
        let suggestions = opt.suggest(&features, "test_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::ConstantPropagation));
        assert!(types.contains(&&OptimizationType::DeadCodeElimination));
    }

    #[test]
    fn test_loop_bearing_suggests_licm() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 10, 3, 2, 0, 25, 25.0, 0);
        let suggestions = opt.suggest(&features, "loop_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::LoopInvariantCodeMotion));
    }

    #[test]
    fn test_no_licm_without_loops() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 10, 3, 0, 0, 25, 25.0, 0);
        let suggestions = opt.suggest(&features, "no_loop_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(!types.contains(&&OptimizationType::LoopInvariantCodeMotion));
    }

    #[test]
    fn test_arithmetic_heavy_suggests_strength_reduction() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(200, 80, 10, 0, 0, 50, 50.0, 0);
        let suggestions = opt.suggest(&features, "arith_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::StrengthReduction));
    }

    #[test]
    fn test_many_calls_suggests_inlining() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 5, 3, 0, 5, 10, 10.0, 0);
        let suggestions = opt.suggest(&features, "caller_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::FunctionInlining));
    }

    #[test]
    fn test_large_blocks_suggests_cse() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 10, 5, 0, 0, 40, 40.0, 0);
        let suggestions = opt.suggest(&features, "big_block_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::CommonSubexpressionElimination));
    }

    #[test]
    fn test_memory_heavy_suggests_sroa() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 5, 20, 0, 0, 30, 30.0, 0);
        let suggestions = opt.suggest(&features, "mem_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::ScalarReplacementOfAggregates));
    }

    #[test]
    fn test_epistemic_suggests_uncertainty_aware() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(50, 10, 5, 0, 0, 25, 25.0, 3);
        let suggestions = opt.suggest(&features, "epistemic_fn");

        let types: Vec<_> = suggestions.iter().map(|s| &s.optimization_type).collect();
        assert!(types.contains(&&OptimizationType::UncertaintyAwareOptimization));
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        // High threshold should filter out low-confidence suggestions
        let opt = HeuristicOptimizer::new(0.90);
        let features = make_features(50, 10, 5, 1, 0, 25, 25.0, 0);
        let suggestions = opt.suggest(&features, "test_fn");

        // Only CP and DCE have 0.95 confidence
        for s in &suggestions {
            assert!(s.confidence >= 0.90);
        }
    }

    #[test]
    fn test_sorted_by_confidence_descending() {
        let opt = HeuristicOptimizer::new(0.7);
        // Trigger multiple rules
        let features = make_features(200, 80, 10, 2, 5, 40, 10.0, 1);
        let suggestions = opt.suggest(&features, "complex_fn");

        for window in suggestions.windows(2) {
            assert!(window[0].confidence >= window[1].confidence);
        }
    }

    #[test]
    fn test_deterministic_output() {
        let opt = HeuristicOptimizer::new(0.7);
        let features = make_features(100, 40, 15, 2, 4, 35, 25.0, 1);

        let run1 = opt.suggest(&features, "det_fn");
        let run2 = opt.suggest(&features, "det_fn");

        assert_eq!(run1.len(), run2.len());
        for (a, b) in run1.iter().zip(run2.iter()) {
            assert_eq!(a.optimization_type, b.optimization_type);
            assert_eq!(a.confidence, b.confidence);
        }
    }

    #[test]
    fn test_data_collector() {
        let mut collector = DataCollector::new();
        let features = make_features(50, 10, 5, 0, 0, 25, 25.0, 0);

        collector.collect_pass_result("test_fn", &features, "ConstantPropagation", true, 50, 45);
        collector.collect_pass_result("test_fn", &features, "DeadCodeElimination", false, 45, 45);

        assert_eq!(collector.record_count(), 2);
    }
}
