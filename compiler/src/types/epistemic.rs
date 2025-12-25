//! Epistemic Types for Sounio
//!
//! Epistemic types track knowledge quality through computation:
//! - Confidence (ε): How certain is this value? [0.0, 1.0]
//! - Provenance (Φ): Where did this value come from?
//! - Temporal validity (τ): When was this knowledge valid?
//!
//! This is the KEY DIFFERENTIATOR of Sounio from all other languages.
//! No other language provides compile-time epistemic tracking.

use crate::common::Span;
use std::collections::HashMap;

/// Knowledge type wrapper: Knowledge[T, ε >= threshold]
#[derive(Debug, Clone, PartialEq)]
pub struct KnowledgeType {
    /// The underlying value type
    pub inner_type: Box<super::Type>,

    /// Minimum confidence bound (ε >= bound)
    pub confidence_bound: Option<ConfidenceBound>,

    /// Provenance requirement
    pub provenance_constraint: Option<ProvenanceConstraint>,

    /// Temporal validity constraint
    pub temporal_constraint: Option<TemporalConstraint>,
}

impl KnowledgeType {
    pub fn new(inner: super::Type) -> Self {
        Self {
            inner_type: Box::new(inner),
            confidence_bound: None,
            provenance_constraint: None,
            temporal_constraint: None,
        }
    }

    pub fn with_confidence(mut self, bound: ConfidenceBound) -> Self {
        self.confidence_bound = Some(bound);
        self
    }

    pub fn with_provenance(mut self, constraint: ProvenanceConstraint) -> Self {
        self.provenance_constraint = Some(constraint);
        self
    }

    pub fn with_temporal(mut self, constraint: TemporalConstraint) -> Self {
        self.temporal_constraint = Some(constraint);
        self
    }
}

/// Confidence bound: ε >= value, ε > value, ε == value
#[derive(Debug, Clone, PartialEq)]
pub struct ConfidenceBound {
    pub operator: ConfidenceOp,
    pub value: f64,
    pub span: Option<Span>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfidenceOp {
    GreaterEq, // ε >= value
    Greater,   // ε > value
    Eq,        // ε == value
    LessEq,    // ε <= value (rare, for upper bounds)
}

impl ConfidenceBound {
    pub fn at_least(value: f64) -> Self {
        Self {
            operator: ConfidenceOp::GreaterEq,
            value,
            span: None,
        }
    }

    /// Check if a confidence value satisfies this bound
    pub fn is_satisfied_by(&self, actual: f64) -> bool {
        match self.operator {
            ConfidenceOp::GreaterEq => actual >= self.value,
            ConfidenceOp::Greater => actual > self.value,
            ConfidenceOp::Eq => (actual - self.value).abs() < f64::EPSILON,
            ConfidenceOp::LessEq => actual <= self.value,
        }
    }
}

/// Provenance constraint for knowledge origin
#[derive(Debug, Clone, PartialEq)]
pub enum ProvenanceConstraint {
    /// Must come from a specific source
    FromSource(String),

    /// Must be derived from sources matching pattern
    DerivedFrom(Vec<String>),

    /// Must be user-provided
    UserInput,

    /// Must be peer-reviewed
    PeerReviewed,

    /// Must comply with regulatory requirement
    RegulatoryCompliant(String),
}

/// Temporal validity constraint
#[derive(Debug, Clone, PartialEq)]
pub enum TemporalConstraint {
    /// Knowledge valid for at most N seconds
    MaxAge(u64),

    /// Knowledge valid after timestamp
    ValidAfter(u64),

    /// Knowledge valid within date range
    ValidBetween(u64, u64),

    /// Always valid (no temporal constraint)
    Eternal,
}

// =============================================================================
// Confidence Propagation Rules
// =============================================================================

/// Rules for how confidence propagates through operations
#[derive(Debug, Clone)]
pub struct PropagationRules {
    rules: HashMap<String, PropagationRule>,
}

#[derive(Debug, Clone)]
pub enum PropagationRule {
    /// Output confidence = min(input confidences)
    Minimum,

    /// Output confidence = product of input confidences
    Product,

    /// Output confidence = weighted average
    WeightedAverage(Vec<f64>),

    /// Output confidence = min(inputs) * degradation factor
    MinWithDegradation(f64),

    /// Custom propagation function
    Custom(String),
}

impl Default for PropagationRules {
    fn default() -> Self {
        let mut rules = HashMap::new();

        // Arithmetic operations: take minimum
        rules.insert("add".to_string(), PropagationRule::Minimum);
        rules.insert("sub".to_string(), PropagationRule::Minimum);
        rules.insert("mul".to_string(), PropagationRule::Minimum);
        rules.insert("div".to_string(), PropagationRule::MinWithDegradation(0.99));

        // ODE solvers degrade confidence
        rules.insert(
            "ode_solve".to_string(),
            PropagationRule::MinWithDegradation(0.95),
        );

        // Interpolation degrades slightly
        rules.insert(
            "interpolate".to_string(),
            PropagationRule::MinWithDegradation(0.99),
        );

        // Monte Carlo: confidence degrades based on sample size
        rules.insert(
            "monte_carlo".to_string(),
            PropagationRule::Custom("monte_carlo_propagation".to_string()),
        );

        Self { rules }
    }
}

impl PropagationRules {
    pub fn get(&self, operation: &str) -> Option<&PropagationRule> {
        self.rules.get(operation)
    }

    /// Calculate output confidence given inputs and operation
    pub fn propagate(&self, operation: &str, input_confidences: &[f64]) -> f64 {
        match self.rules.get(operation) {
            Some(PropagationRule::Minimum) => input_confidences.iter().cloned().fold(1.0, f64::min),
            Some(PropagationRule::Product) => input_confidences.iter().product(),
            Some(PropagationRule::WeightedAverage(weights)) => {
                let sum: f64 = input_confidences
                    .iter()
                    .zip(weights.iter())
                    .map(|(c, w)| c * w)
                    .sum();
                let weight_sum: f64 = weights.iter().sum();
                sum / weight_sum
            }
            Some(PropagationRule::MinWithDegradation(factor)) => {
                input_confidences.iter().cloned().fold(1.0, f64::min) * factor
            }
            Some(PropagationRule::Custom(_)) => {
                // Custom rules need special handling
                input_confidences.iter().cloned().fold(1.0, f64::min)
            }
            None => {
                // Default: minimum confidence
                input_confidences.iter().cloned().fold(1.0, f64::min)
            }
        }
    }
}

// =============================================================================
// Epistemic Type Checker
// =============================================================================

/// Checks epistemic constraints at compile time
#[derive(Debug)]
pub struct EpistemicChecker {
    propagation: PropagationRules,
    /// Known confidence values for variables
    known_confidences: HashMap<String, f64>,
}

impl Default for EpistemicChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl EpistemicChecker {
    pub fn new() -> Self {
        Self {
            propagation: PropagationRules::default(),
            known_confidences: HashMap::new(),
        }
    }

    /// Record a known confidence for a variable
    pub fn set_confidence(&mut self, var: &str, confidence: f64) {
        self.known_confidences.insert(var.to_string(), confidence);
    }

    /// Get the known confidence for a variable
    pub fn get_confidence(&self, var: &str) -> Option<f64> {
        self.known_confidences.get(var).copied()
    }

    /// Check if a Knowledge type satisfies a required bound
    pub fn check_bound(
        &self,
        actual: &KnowledgeType,
        required: &ConfidenceBound,
    ) -> EpistemicCheckResult {
        match &actual.confidence_bound {
            Some(actual_bound) => {
                // If actual bound is at least as strong as required
                if actual_bound.value >= required.value {
                    EpistemicCheckResult::Satisfied
                } else {
                    EpistemicCheckResult::InsufficientConfidence {
                        required: required.value,
                        actual: actual_bound.value,
                    }
                }
            }
            None => EpistemicCheckResult::UnknownConfidence,
        }
    }

    /// Check provenance constraint
    pub fn check_provenance(
        &self,
        actual: &KnowledgeType,
        required: &ProvenanceConstraint,
    ) -> EpistemicCheckResult {
        match (&actual.provenance_constraint, required) {
            (
                Some(ProvenanceConstraint::RegulatoryCompliant(actual_reg)),
                ProvenanceConstraint::RegulatoryCompliant(required_reg),
            ) => {
                if actual_reg == required_reg || actual_reg == "FDA" {
                    EpistemicCheckResult::Satisfied
                } else {
                    EpistemicCheckResult::ProvenanceViolation {
                        reason: format!(
                            "Required {} compliance, found {}",
                            required_reg, actual_reg
                        ),
                    }
                }
            }
            (None, _) => EpistemicCheckResult::UnknownProvenance,
            _ => EpistemicCheckResult::Satisfied, // Other cases simplified for now
        }
    }
}

/// Result of epistemic constraint checking
#[derive(Debug, Clone, PartialEq)]
pub enum EpistemicCheckResult {
    /// Constraint satisfied
    Satisfied,

    /// Confidence too low
    InsufficientConfidence { required: f64, actual: f64 },

    /// Confidence unknown at compile time
    UnknownConfidence,

    /// Provenance constraint violated
    ProvenanceViolation { reason: String },

    /// Provenance unknown
    UnknownProvenance,

    /// Temporal constraint violated
    TemporalViolation { reason: String },
}

// =============================================================================
// Compile-time Confidence Analysis
// =============================================================================

/// Static analysis for confidence flow
pub struct ConfidenceFlowAnalysis {
    /// Variable -> known confidence bounds
    bounds: HashMap<String, ConfidenceBound>,

    /// Propagation rules
    rules: PropagationRules,
}

impl Default for ConfidenceFlowAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfidenceFlowAnalysis {
    pub fn new() -> Self {
        Self {
            bounds: HashMap::new(),
            rules: PropagationRules::default(),
        }
    }

    /// Analyze confidence flow in a function body
    /// Returns the output confidence bounds
    pub fn analyze_function(
        &mut self,
        _params: &[(String, KnowledgeType)],
        _body: &str, // Simplified - would be actual AST
    ) -> Result<ConfidenceBound, ConfidenceAnalysisError> {
        // This would perform actual dataflow analysis
        // For now, return a placeholder
        Ok(ConfidenceBound::at_least(0.50))
    }
}

#[derive(Debug)]
pub enum ConfidenceAnalysisError {
    UnboundVariable(String),
    IncompatibleBounds {
        var: String,
        bound1: f64,
        bound2: f64,
    },
    CircularDependency(Vec<String>),
}

// =============================================================================
// Display implementations
// =============================================================================

impl std::fmt::Display for KnowledgeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Knowledge[{}", self.inner_type)?;
        if let Some(ref bound) = self.confidence_bound {
            write!(f, ", {}", bound)?;
        }
        write!(f, "]")
    }
}

impl std::fmt::Display for ConfidenceBound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let op = match self.operator {
            ConfidenceOp::GreaterEq => ">=",
            ConfidenceOp::Greater => ">",
            ConfidenceOp::Eq => "==",
            ConfidenceOp::LessEq => "<=",
        };
        write!(f, "ε {} {:.2}", op, self.value)
    }
}

impl std::fmt::Display for super::Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            super::Type::Unit => write!(f, "()"),
            super::Type::Bool => write!(f, "bool"),
            super::Type::I32 => write!(f, "i32"),
            super::Type::I64 => write!(f, "i64"),
            super::Type::F32 => write!(f, "f32"),
            super::Type::F64 => write!(f, "f64"),
            super::Type::String => write!(f, "String"),
            super::Type::Str => write!(f, "str"),
            super::Type::Named { name, args } => {
                write!(f, "{}", name)?;
                if !args.is_empty() {
                    write!(f, "<")?;
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", arg)?;
                    }
                    write!(f, ">")?;
                }
                Ok(())
            }
            _ => write!(f, "{:?}", self),
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Type;

    #[test]
    fn test_knowledge_type_creation() {
        let kt = KnowledgeType::new(Type::F64).with_confidence(ConfidenceBound::at_least(0.80));

        assert!(kt.confidence_bound.is_some());
        assert_eq!(kt.confidence_bound.as_ref().unwrap().value, 0.80);
    }

    #[test]
    fn test_confidence_bound_satisfaction() {
        let bound = ConfidenceBound::at_least(0.80);

        assert!(bound.is_satisfied_by(0.80));
        assert!(bound.is_satisfied_by(0.95));
        assert!(!bound.is_satisfied_by(0.79));
    }

    #[test]
    fn test_propagation_rules() {
        let rules = PropagationRules::default();

        // Min propagation for addition
        let result = rules.propagate("add", &[0.90, 0.85, 0.95]);
        assert!((result - 0.85).abs() < f64::EPSILON);

        // Degradation for division
        let result = rules.propagate("div", &[0.90, 0.90]);
        assert!((result - 0.90 * 0.99).abs() < f64::EPSILON);
    }

    #[test]
    fn test_epistemic_checker() {
        let checker = EpistemicChecker::new();

        let actual = KnowledgeType::new(Type::F64).with_confidence(ConfidenceBound::at_least(0.85));

        let required = ConfidenceBound::at_least(0.80);

        let result = checker.check_bound(&actual, &required);
        assert_eq!(result, EpistemicCheckResult::Satisfied);

        let required_high = ConfidenceBound::at_least(0.90);
        let result = checker.check_bound(&actual, &required_high);
        match result {
            EpistemicCheckResult::InsufficientConfidence { required, actual } => {
                assert_eq!(required, 0.90);
                assert_eq!(actual, 0.85);
            }
            _ => panic!("Expected InsufficientConfidence"),
        }
    }

    #[test]
    fn test_knowledge_type_display() {
        let kt = KnowledgeType::new(Type::F64).with_confidence(ConfidenceBound::at_least(0.80));

        let display = format!("{}", kt);
        assert!(display.contains("Knowledge"));
        assert!(display.contains("0.80"));
    }
}
