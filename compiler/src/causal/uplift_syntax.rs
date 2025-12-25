//! Uplift Modeling Syntax Integration for Causal Blocks
//!
//! Extends Sounio causal model blocks with uplift modeling syntax:
//!
//! ```d
//! causal model MarketingUplift {
//!     nodes: [Treatment, Age, Income, Purchase, Segment]
//!
//!     // Causal structure
//!     Age -> Purchase
//!     Income -> Purchase
//!     Treatment -> Purchase
//!
//!     // Uplift estimation block
//!     uplift {
//!         treatment: Treatment
//!         outcome: Purchase
//!         covariates: [Age, Income]
//!
//!         // Meta-learner selection
//!         estimator: XLearner {
//!             base_learner: GradientBoosting
//!             propensity_model: LogisticRegression
//!         }
//!
//!         // Epistemic configuration
//!         epistemic {
//!             prior: Beta(1, 1)          // Uniform prior
//!             min_confidence: 0.8        // Minimum for decisions
//!             track_provenance: true
//!         }
//!
//!         // Segmentation rules
//!         segments {
//!             Persuadable: tau > 0.1 with confidence > 0.8
//!             SleepingDog: tau < -0.05 with confidence > 0.8
//!             SureThing: baseline > 0.7
//!             LostCause: baseline < 0.1
//!         }
//!
//!         // Refutation tests (run automatically)
//!         refutations: [
//!             PlaceboTreatment,
//!             RandomCommonCause { n_confounders: 3 },
//!             Bootstrap { n_samples: 1000 },
//!             Monotonicity
//!         ]
//!     }
//! }
//! ```
//!
//! # Generated Code
//!
//! The uplift block generates:
//! 1. Type-safe uplift estimator instantiation
//! 2. Automatic epistemic tracking
//! 3. Segment classification functions
//! 4. Refutation test harness

use crate::common::{NodeId, Span};
use serde::{Deserialize, Serialize};

/// Uplift modeling block within a causal model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpliftBlock {
    pub id: NodeId,
    /// Treatment variable name
    pub treatment: String,
    /// Outcome variable name
    pub outcome: String,
    /// Covariate variable names
    pub covariates: Vec<String>,
    /// Meta-learner estimator configuration
    pub estimator: UpliftEstimatorConfig,
    /// Epistemic configuration
    pub epistemic: UpliftEpistemicConfig,
    /// Customer segment definitions
    pub segments: Vec<SegmentRule>,
    /// Refutation tests to run
    pub refutations: Vec<RefutationSpec>,
    pub span: Span,
}

/// Uplift estimator configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpliftEstimatorConfig {
    pub id: NodeId,
    /// Estimator type
    pub estimator_type: EstimatorType,
    /// Base learner for CATE estimation
    pub base_learner: Option<BaseLearnerType>,
    /// Propensity model (for doubly-robust methods)
    pub propensity_model: Option<BaseLearnerType>,
    /// GPU acceleration
    pub gpu_accelerated: bool,
    /// Honest estimation (sample splitting)
    pub honest: bool,
    /// Cross-fitting folds
    pub n_folds: u32,
    pub span: Span,
}

impl Default for UpliftEstimatorConfig {
    fn default() -> Self {
        Self {
            id: NodeId(0),
            estimator_type: EstimatorType::XLearner,
            base_learner: Some(BaseLearnerType::GradientBoosting),
            propensity_model: Some(BaseLearnerType::LogisticRegression),
            gpu_accelerated: false,
            honest: true,
            n_folds: 5,
            span: Span::default(),
        }
    }
}

/// Meta-learner estimator types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EstimatorType {
    /// S-Learner: Single model with treatment as feature
    SLearner,
    /// T-Learner: Separate models for treatment/control
    TLearner,
    /// X-Learner: Cross-learner (Künzel et al., 2019)
    XLearner,
    /// R-Learner: Robinson's transformation
    RLearner,
    /// DR-Learner: Doubly robust
    DRLearner,
    /// Causal Forest (Athey & Imbens)
    CausalForest,
    /// GPU-accelerated uplift tree
    GpuUpliftTree,
}

impl EstimatorType {
    /// Get the default base learner for this estimator
    pub fn default_base_learner(&self) -> BaseLearnerType {
        match self {
            EstimatorType::SLearner => BaseLearnerType::GradientBoosting,
            EstimatorType::TLearner => BaseLearnerType::RandomForest,
            EstimatorType::XLearner => BaseLearnerType::GradientBoosting,
            EstimatorType::RLearner => BaseLearnerType::Lasso,
            EstimatorType::DRLearner => BaseLearnerType::GradientBoosting,
            EstimatorType::CausalForest => BaseLearnerType::DecisionTree,
            EstimatorType::GpuUpliftTree => BaseLearnerType::DecisionTree,
        }
    }

    /// Check if this estimator supports GPU acceleration
    pub fn supports_gpu(&self) -> bool {
        matches!(
            self,
            EstimatorType::GpuUpliftTree | EstimatorType::XLearner | EstimatorType::CausalForest
        )
    }
}

/// Base learner types for meta-learners
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaseLearnerType {
    LinearRegression,
    LogisticRegression,
    Lasso,
    Ridge,
    ElasticNet,
    DecisionTree,
    RandomForest,
    GradientBoosting,
    XGBoost,
    LightGBM,
    NeuralNetwork,
}

/// Epistemic configuration for uplift estimation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpliftEpistemicConfig {
    pub id: NodeId,
    /// Prior distribution for CATE (Beta parameters)
    pub prior_alpha: f64,
    pub prior_beta: f64,
    /// Minimum confidence for actionable decisions
    pub min_confidence: f64,
    /// Track data provenance
    pub track_provenance: bool,
    /// Propagate uncertainty through predictions
    pub propagate_uncertainty: bool,
    /// Credible interval level (e.g., 0.95 for 95% CI)
    pub credible_level: f64,
    pub span: Span,
}

impl Default for UpliftEpistemicConfig {
    fn default() -> Self {
        Self {
            id: NodeId(0),
            prior_alpha: 1.0, // Uniform prior
            prior_beta: 1.0,
            min_confidence: 0.8,
            track_provenance: true,
            propagate_uncertainty: true,
            credible_level: 0.95,
            span: Span::default(),
        }
    }
}

/// Segment classification rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentRule {
    pub id: NodeId,
    /// Segment name
    pub name: String,
    /// Segment type
    pub segment_type: SegmentType,
    /// Condition expression (simplified)
    pub condition: SegmentCondition,
    pub span: Span,
}

/// Customer segment types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SegmentType {
    /// τ(x) > 0 with high confidence
    Persuadable,
    /// τ(x) < 0 with high confidence
    SleepingDog,
    /// High baseline conversion
    SureThing,
    /// Low baseline conversion
    LostCause,
    /// Custom segment
    Custom,
}

/// Segment condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SegmentCondition {
    /// tau > threshold with confidence > conf_threshold
    TauGreaterThan {
        threshold: f64,
        min_confidence: Option<f64>,
    },
    /// tau < threshold with confidence > conf_threshold
    TauLessThan {
        threshold: f64,
        min_confidence: Option<f64>,
    },
    /// baseline > threshold
    BaselineGreaterThan { threshold: f64 },
    /// baseline < threshold
    BaselineLessThan { threshold: f64 },
    /// Custom expression (as string for now)
    Custom { expr: String },
}

/// Refutation test specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefutationSpec {
    pub id: NodeId,
    /// Refutation type
    pub refutation_type: UpliftRefutationType,
    /// Parameters
    pub params: RefutationParams,
    pub span: Span,
}

/// Refutation test types for uplift modeling (distinct from general causal refutation)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpliftRefutationType {
    /// Placebo treatment (random treatment)
    PlaceboTreatment,
    /// Add random common causes
    RandomCommonCause,
    /// Subset data refutation
    DataSubset,
    /// Bootstrap refutation
    Bootstrap,
    /// Sensitivity analysis (Rosenbaum bounds)
    SensitivityAnalysis,
    /// Monotonicity check (uplift-specific)
    Monotonicity,
    /// Persuadable segment validity
    PersuadableValidity,
    /// Sleeping dog detection
    SleepingDogDetection,
}

/// Refutation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefutationParams {
    None,
    RandomCommonCause { n_confounders: u32 },
    DataSubset { fraction: f64 },
    Bootstrap { n_samples: u32 },
    SensitivityAnalysis { gamma_range: (f64, f64) },
}

/// Code generator for uplift blocks
pub struct UpliftCodeGen {
    /// Generated Sounio code
    output: String,
    /// Indent level
    indent: usize,
}

impl UpliftCodeGen {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
        }
    }

    pub fn output(&self) -> &str {
        &self.output
    }

    fn emit(&mut self, s: &str) {
        let indent = "    ".repeat(self.indent);
        self.output.push_str(&indent);
        self.output.push_str(s);
        self.output.push('\n');
    }

    /// Generate Sounio code for an uplift block
    pub fn generate(&mut self, block: &UpliftBlock, model_name: &str) {
        self.emit(&format!("// Generated uplift estimator for {}", model_name));
        self.emit("");

        // Generate estimator struct
        self.generate_estimator_struct(block, model_name);

        // Generate estimation function
        self.generate_estimate_function(block, model_name);

        // Generate segment classifier
        self.generate_segment_classifier(block, model_name);

        // Generate refutation harness
        self.generate_refutation_harness(block, model_name);
    }

    fn generate_estimator_struct(&mut self, block: &UpliftBlock, model_name: &str) {
        self.emit(&format!("struct {}UpliftEstimator {{", model_name));
        self.indent += 1;

        // Treatment and outcome
        self.emit(&format!(
            "treatment_var: string  // \"{}\"",
            block.treatment
        ));
        self.emit(&format!("outcome_var: string    // \"{}\"", block.outcome));

        // Covariates
        let cov_str = block.covariates.join(", ");
        self.emit(&format!("covariates: [string]   // [{}]", cov_str));

        // Epistemic state
        self.emit("");
        self.emit("// Epistemic configuration");
        self.emit(&format!(
            "prior_alpha: f64 = {}",
            block.epistemic.prior_alpha
        ));
        self.emit(&format!("prior_beta: f64 = {}", block.epistemic.prior_beta));
        self.emit(&format!(
            "min_confidence: f64 = {}",
            block.epistemic.min_confidence
        ));

        // Estimator type
        self.emit("");
        self.emit(&format!(
            "// Estimator: {:?}",
            block.estimator.estimator_type
        ));
        if block.estimator.gpu_accelerated {
            self.emit("gpu_accelerated: bool = true");
        }

        self.indent -= 1;
        self.emit("}");
        self.emit("");
    }

    fn generate_estimate_function(&mut self, block: &UpliftBlock, model_name: &str) {
        self.emit(&format!(
            "fn estimate_uplift(est: &{}UpliftEstimator, X: Matrix, T: Vector, Y: Vector) -> UpliftResult with Prob {{",
            model_name
        ));
        self.indent += 1;

        // Generate estimator-specific code
        match block.estimator.estimator_type {
            EstimatorType::XLearner => {
                self.emit("// X-Learner estimation");
                self.emit("let mu0 = fit_outcome_model(X[T == 0], Y[T == 0])");
                self.emit("let mu1 = fit_outcome_model(X[T == 1], Y[T == 1])");
                self.emit("");
                self.emit("// Imputed treatment effects");
                self.emit("let D0 = mu1.predict(X[T == 0]) - Y[T == 0]");
                self.emit("let D1 = Y[T == 1] - mu0.predict(X[T == 1])");
                self.emit("");
                self.emit("// CATE models");
                self.emit("let tau0_model = fit_cate_model(X[T == 0], D0)");
                self.emit("let tau1_model = fit_cate_model(X[T == 1], D1)");
                self.emit("");
                self.emit("// Propensity weighting");
                self.emit("let e = fit_propensity(X, T)");
                self.emit("let tau = e * tau0_model.predict(X) + (1 - e) * tau1_model.predict(X)");
            }
            EstimatorType::TLearner => {
                self.emit("// T-Learner estimation");
                self.emit("let mu0 = fit_outcome_model(X[T == 0], Y[T == 0])");
                self.emit("let mu1 = fit_outcome_model(X[T == 1], Y[T == 1])");
                self.emit("let tau = mu1.predict(X) - mu0.predict(X)");
            }
            EstimatorType::GpuUpliftTree => {
                self.emit("// GPU-accelerated uplift tree");
                self.emit("let tree = gpu_uplift_tree_fit(X, T, Y)");
                self.emit("let tau = tree.predict_epistemic(X)");
            }
            _ => {
                self.emit(&format!(
                    "// {:?} estimation",
                    block.estimator.estimator_type
                ));
                self.emit("let tau = estimate_cate(X, T, Y)");
            }
        }

        // Epistemic wrapping
        self.emit("");
        self.emit("// Wrap in epistemic type with Beta posterior");
        self.emit(&format!(
            "let epistemic_tau = Knowledge::from_estimate(tau, prior: Beta({}, {}))",
            block.epistemic.prior_alpha, block.epistemic.prior_beta
        ));

        self.emit("");
        self.emit("UpliftResult {");
        self.indent += 1;
        self.emit("tau: epistemic_tau");
        self.emit("confidence: epistemic_tau.confidence()");
        self.emit(&format!("treatment: \"{}\"", block.treatment));
        self.emit(&format!("outcome: \"{}\"", block.outcome));
        self.indent -= 1;
        self.emit("}");

        self.indent -= 1;
        self.emit("}");
        self.emit("");
    }

    fn generate_segment_classifier(&mut self, block: &UpliftBlock, model_name: &str) {
        self.emit("fn classify_segment(result: &UpliftResult) -> CustomerSegment {");
        self.indent += 1;

        for rule in &block.segments {
            let condition = match &rule.condition {
                SegmentCondition::TauGreaterThan {
                    threshold,
                    min_confidence,
                } => {
                    let conf = min_confidence.unwrap_or(block.epistemic.min_confidence);
                    format!(
                        "result.tau > {} and result.confidence > {}",
                        threshold, conf
                    )
                }
                SegmentCondition::TauLessThan {
                    threshold,
                    min_confidence,
                } => {
                    let conf = min_confidence.unwrap_or(block.epistemic.min_confidence);
                    format!(
                        "result.tau < {} and result.confidence > {}",
                        threshold, conf
                    )
                }
                SegmentCondition::BaselineGreaterThan { threshold } => {
                    format!("result.baseline > {}", threshold)
                }
                SegmentCondition::BaselineLessThan { threshold } => {
                    format!("result.baseline < {}", threshold)
                }
                SegmentCondition::Custom { expr } => expr.clone(),
            };

            self.emit(&format!("if {} {{", condition));
            self.indent += 1;
            self.emit(&format!("return CustomerSegment::{}", rule.name));
            self.indent -= 1;
            self.emit("}");
        }

        self.emit("CustomerSegment::Uncertain");

        self.indent -= 1;
        self.emit("}");
        self.emit("");
    }

    fn generate_refutation_harness(&mut self, block: &UpliftBlock, model_name: &str) {
        self.emit(&format!(
            "fn run_refutations(est: &{}UpliftEstimator, X: Matrix, T: Vector, Y: Vector) -> RefutationReport with IO {{",
            model_name
        ));
        self.indent += 1;

        self.emit("var report = RefutationReport::new()");
        self.emit("");

        for refutation in &block.refutations {
            let test_call = match (&refutation.refutation_type, &refutation.params) {
                (UpliftRefutationType::PlaceboTreatment, _) => {
                    "refute_placebo_treatment(est, X, T, Y)".to_string()
                }
                (
                    UpliftRefutationType::RandomCommonCause,
                    RefutationParams::RandomCommonCause { n_confounders },
                ) => {
                    format!(
                        "refute_random_common_cause(est, X, T, Y, n_confounders: {})",
                        n_confounders
                    )
                }
                (UpliftRefutationType::Bootstrap, RefutationParams::Bootstrap { n_samples }) => {
                    format!("refute_bootstrap(est, X, T, Y, n_samples: {})", n_samples)
                }
                (UpliftRefutationType::Monotonicity, _) => {
                    "refute_monotonicity(est, X, T, Y)".to_string()
                }
                (UpliftRefutationType::PersuadableValidity, _) => {
                    "refute_persuadable_validity(est, X, T, Y)".to_string()
                }
                _ => format!("refute_{:?}(est, X, T, Y)", refutation.refutation_type),
            };

            self.emit(&format!("report.add({})", test_call));
        }

        self.emit("");
        self.emit("report");

        self.indent -= 1;
        self.emit("}");
    }
}

impl Default for UpliftCodeGen {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse an uplift block from tokens (simplified parser)
pub fn parse_uplift_block_simple(
    treatment: &str,
    outcome: &str,
    covariates: Vec<String>,
    estimator_type: EstimatorType,
) -> UpliftBlock {
    UpliftBlock {
        id: NodeId(0),
        treatment: treatment.to_string(),
        outcome: outcome.to_string(),
        covariates,
        estimator: UpliftEstimatorConfig {
            estimator_type,
            ..Default::default()
        },
        epistemic: UpliftEpistemicConfig::default(),
        segments: vec![
            SegmentRule {
                id: NodeId(0),
                name: "Persuadable".to_string(),
                segment_type: SegmentType::Persuadable,
                condition: SegmentCondition::TauGreaterThan {
                    threshold: 0.1,
                    min_confidence: Some(0.8),
                },
                span: Span::default(),
            },
            SegmentRule {
                id: NodeId(0),
                name: "SleepingDog".to_string(),
                segment_type: SegmentType::SleepingDog,
                condition: SegmentCondition::TauLessThan {
                    threshold: -0.05,
                    min_confidence: Some(0.8),
                },
                span: Span::default(),
            },
        ],
        refutations: vec![
            RefutationSpec {
                id: NodeId(0),
                refutation_type: UpliftRefutationType::PlaceboTreatment,
                params: RefutationParams::None,
                span: Span::default(),
            },
            RefutationSpec {
                id: NodeId(0),
                refutation_type: UpliftRefutationType::Bootstrap,
                params: RefutationParams::Bootstrap { n_samples: 1000 },
                span: Span::default(),
            },
        ],
        span: Span::default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uplift_block_codegen() {
        let block = parse_uplift_block_simple(
            "Treatment",
            "Purchase",
            vec!["Age".to_string(), "Income".to_string()],
            EstimatorType::XLearner,
        );

        let mut codegen = UpliftCodeGen::new();
        codegen.generate(&block, "Marketing");

        let output = codegen.output();

        assert!(output.contains("MarketingUpliftEstimator"));
        assert!(output.contains("X-Learner"));
        assert!(output.contains("estimate_uplift"));
        assert!(output.contains("classify_segment"));
        assert!(output.contains("Persuadable"));
    }

    #[test]
    fn test_segment_conditions() {
        let cond = SegmentCondition::TauGreaterThan {
            threshold: 0.1,
            min_confidence: Some(0.9),
        };

        match cond {
            SegmentCondition::TauGreaterThan {
                threshold,
                min_confidence,
            } => {
                assert!((threshold - 0.1).abs() < 1e-6);
                assert_eq!(min_confidence, Some(0.9));
            }
            _ => panic!("Wrong condition type"),
        }
    }

    #[test]
    fn test_estimator_types() {
        assert!(EstimatorType::GpuUpliftTree.supports_gpu());
        assert!(EstimatorType::XLearner.supports_gpu());
        assert!(!EstimatorType::SLearner.supports_gpu());
    }
}
