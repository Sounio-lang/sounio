//! Knowledge operations: assert, query, revise, translate
//!
//! These are the primitive operations on Knowledge types.
//! They form the core API for working with epistemic values in Sounio.
//!
//! # Operations
//!
//! - `assert`: Create new knowledge with explicit epistemic status
//! - `query`: Search for knowledge matching constraints
//! - `revise`: Update knowledge with new evidence
//! - `translate`: Convert knowledge between ontology domains
//! - `merge`: Combine multiple knowledge sources
//! - `inspect`: Extract epistemic metadata

use super::knowledge::OntologyBinding;
use super::{Confidence, EpistemicStatus, Source};
use crate::ast::Expr;

/// Operations on Knowledge values
#[derive(Debug, Clone)]
pub enum KnowledgeOp {
    /// Assert new knowledge
    Assert(AssertOp),

    /// Query existing knowledge
    Query(QueryOp),

    /// Revise knowledge with new evidence
    Revise(ReviseOp),

    /// Translate between ontologies
    Translate(TranslateOp),

    /// Merge multiple knowledge sources
    Merge(MergeOp),

    /// Extract epistemic metadata
    Inspect(InspectOp),
}

/// Assert new knowledge into the system
///
/// ```sounio
/// assert mass : Knowledge[PATO:mass] = 5.2_kg with {
///     confidence: 0.99,
///     source: Measurement { instrument: "scale_001" }
/// };
/// ```
#[derive(Debug, Clone)]
pub struct AssertOp {
    /// The value being asserted
    pub value: Box<Expr>,

    /// Target domain binding
    pub domain: OntologyBinding,

    /// Epistemic status to assign
    pub status: Option<EpistemicStatus>,

    /// Evidence for the assertion
    pub evidence: Vec<super::Evidence>,
}

impl AssertOp {
    /// Create a new assert operation
    pub fn new(value: Expr, domain: OntologyBinding) -> Self {
        Self {
            value: Box::new(value),
            domain,
            status: None,
            evidence: vec![],
        }
    }

    /// Set the epistemic status
    pub fn with_status(mut self, status: EpistemicStatus) -> Self {
        self.status = Some(status);
        self
    }

    /// Add evidence
    pub fn with_evidence(mut self, evidence: super::Evidence) -> Self {
        self.evidence.push(evidence);
        self
    }
}

/// Query knowledge with constraints
///
/// ```sounio
/// let results = query Knowledge[δ: ChEBI, ε.confidence > 0.9]
///     where relation(_, "inhibits", target);
/// ```
#[derive(Debug, Clone, Default)]
pub struct QueryOp {
    /// Domain constraint
    pub domain: Option<OntologyBinding>,

    /// Epistemic constraints
    pub epistemic_constraints: Vec<EpistemicConstraint>,

    /// Relational constraints (ontology-based)
    pub relational_constraints: Vec<RelationalConstraint>,

    /// Temporal constraints
    pub temporal_constraints: Option<super::ContextTime>,
}

impl QueryOp {
    /// Create a new empty query
    pub fn new() -> Self {
        Self::default()
    }

    /// Add domain constraint
    pub fn with_domain(mut self, domain: OntologyBinding) -> Self {
        self.domain = Some(domain);
        self
    }

    /// Add minimum confidence constraint
    pub fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.epistemic_constraints
            .push(EpistemicConstraint::MinConfidence(confidence));
        self
    }

    /// Add relational constraint
    pub fn with_relation(mut self, constraint: RelationalConstraint) -> Self {
        self.relational_constraints.push(constraint);
        self
    }
}

/// Epistemic constraints for queries
#[derive(Debug, Clone)]
pub enum EpistemicConstraint {
    /// Minimum confidence
    MinConfidence(f64),
    /// Maximum confidence
    MaxConfidence(f64),
    /// Specific source kind
    SourceKind(Source),
    /// Must be revisable
    Revisable(bool),
    /// Confidence in range
    ConfidenceRange(f64, f64),
}

impl EpistemicConstraint {
    /// Check if an epistemic status satisfies this constraint
    pub fn satisfied_by(&self, status: &EpistemicStatus) -> bool {
        match self {
            EpistemicConstraint::MinConfidence(min) => status.confidence.value() >= *min,
            EpistemicConstraint::MaxConfidence(max) => status.confidence.value() <= *max,
            EpistemicConstraint::SourceKind(kind) => {
                std::mem::discriminant(&status.source) == std::mem::discriminant(kind)
            }
            EpistemicConstraint::Revisable(expected) => {
                status.revisability.is_revisable() == *expected
            }
            EpistemicConstraint::ConfidenceRange(min, max) => {
                let c = status.confidence.value();
                c >= *min && c <= *max
            }
        }
    }
}

/// Relational constraint for queries
#[derive(Debug, Clone)]
pub struct RelationalConstraint {
    /// Subject (None = wildcard)
    pub subject: Option<String>,
    /// Relation (required)
    pub relation: String,
    /// Object (None = wildcard)
    pub object: Option<String>,
}

impl RelationalConstraint {
    /// Create a new relational constraint
    pub fn new(subject: Option<&str>, relation: &str, object: Option<&str>) -> Self {
        Self {
            subject: subject.map(String::from),
            relation: relation.to_string(),
            object: object.map(String::from),
        }
    }

    /// Create constraint: ?x relation object
    pub fn with_object(relation: &str, object: &str) -> Self {
        Self::new(None, relation, Some(object))
    }

    /// Create constraint: subject relation ?x
    pub fn with_subject(subject: &str, relation: &str) -> Self {
        Self::new(Some(subject), relation, None)
    }
}

/// Revise existing knowledge
///
/// ```sounio
/// revise patient_data with new_measurement {
///     strategy: Bayesian,
///     prior_weight: 0.3
/// };
/// ```
#[derive(Debug, Clone)]
pub struct ReviseOp {
    /// Knowledge to revise
    pub target: Box<Expr>,

    /// New evidence
    pub evidence: Box<Expr>,

    /// Revision strategy
    pub strategy: RevisionStrategy,
}

impl ReviseOp {
    /// Create a new revision operation
    pub fn new(target: Expr, evidence: Expr, strategy: RevisionStrategy) -> Self {
        Self {
            target: Box::new(target),
            evidence: Box::new(evidence),
            strategy,
        }
    }

    /// Create Bayesian revision
    pub fn bayesian(target: Expr, evidence: Expr, prior_weight: f64) -> Self {
        Self::new(
            target,
            evidence,
            RevisionStrategy::Bayesian { prior_weight },
        )
    }

    /// Create replacement revision
    pub fn replace(target: Expr, evidence: Expr) -> Self {
        Self::new(target, evidence, RevisionStrategy::Replace)
    }
}

/// Strategy for revising knowledge
#[derive(Debug, Clone)]
pub enum RevisionStrategy {
    /// Bayesian update
    Bayesian { prior_weight: f64 },

    /// Replace entirely
    Replace,

    /// Maximum entropy update
    MaxEntropy,

    /// Custom function
    Custom { function: String },
}

impl RevisionStrategy {
    /// Apply this strategy to combine old and new confidence
    pub fn apply(&self, old_confidence: Confidence, new_confidence: Confidence) -> Confidence {
        match self {
            RevisionStrategy::Bayesian { prior_weight } => {
                let combined = old_confidence.value() * prior_weight
                    + new_confidence.value() * (1.0 - prior_weight);
                Confidence::new(combined)
            }
            RevisionStrategy::Replace => new_confidence,
            RevisionStrategy::MaxEntropy => {
                // Simple approximation: average
                Confidence::new((old_confidence.value() + new_confidence.value()) / 2.0)
            }
            RevisionStrategy::Custom { .. } => {
                // Would need runtime evaluation
                new_confidence
            }
        }
    }
}

impl Default for RevisionStrategy {
    fn default() -> Self {
        RevisionStrategy::Bayesian { prior_weight: 0.5 }
    }
}

/// Translate knowledge between ontologies
///
/// ```sounio
/// let fhir_obs = translate measurement from PATO to FHIR:Observation;
/// ```
#[derive(Debug, Clone)]
pub struct TranslateOp {
    /// Source knowledge
    pub source: Box<Expr>,

    /// Source ontology (inferred if not specified)
    pub from_ontology: Option<OntologyBinding>,

    /// Target ontology
    pub to_ontology: OntologyBinding,

    /// Translation options
    pub options: TranslateOptions,
}

impl TranslateOp {
    /// Create a new translation operation
    pub fn new(source: Expr, to_ontology: OntologyBinding) -> Self {
        Self {
            source: Box::new(source),
            from_ontology: None,
            to_ontology,
            options: TranslateOptions::default(),
        }
    }

    /// Specify source ontology explicitly
    pub fn from(mut self, from: OntologyBinding) -> Self {
        self.from_ontology = Some(from);
        self
    }

    /// Set options
    pub fn with_options(mut self, options: TranslateOptions) -> Self {
        self.options = options;
        self
    }
}

/// Options for translation operations
#[derive(Debug, Clone, Default)]
pub struct TranslateOptions {
    /// Fail if no exact mapping exists
    pub require_exact: bool,

    /// Maximum allowed confidence loss
    pub max_confidence_loss: Option<f64>,

    /// Preferred mapping source
    pub mapping_source: Option<String>,
}

impl TranslateOptions {
    /// Create default options
    pub fn new() -> Self {
        Self::default()
    }

    /// Require exact mapping
    pub fn exact(mut self) -> Self {
        self.require_exact = true;
        self
    }

    /// Set maximum confidence loss
    pub fn max_loss(mut self, loss: f64) -> Self {
        self.max_confidence_loss = Some(loss);
        self
    }

    /// Set preferred mapping source
    pub fn prefer_source(mut self, source: &str) -> Self {
        self.mapping_source = Some(source.to_string());
        self
    }
}

/// Merge multiple knowledge sources
#[derive(Debug, Clone)]
pub struct MergeOp {
    /// Sources to merge
    pub sources: Vec<Box<Expr>>,

    /// Merge strategy
    pub strategy: MergeStrategy,
}

impl MergeOp {
    /// Create a new merge operation
    pub fn new(sources: Vec<Expr>, strategy: MergeStrategy) -> Self {
        Self {
            sources: sources.into_iter().map(Box::new).collect(),
            strategy,
        }
    }

    /// Merge with most confident strategy
    pub fn most_confident(sources: Vec<Expr>) -> Self {
        Self::new(sources, MergeStrategy::MostConfident)
    }

    /// Merge with weighted average
    pub fn weighted_average(sources: Vec<Expr>) -> Self {
        Self::new(sources, MergeStrategy::WeightedAverage)
    }
}

/// Strategy for merging multiple knowledge sources
#[derive(Debug, Clone, Default)]
pub enum MergeStrategy {
    /// Take the most confident
    MostConfident,

    /// Weighted average by confidence
    #[default]
    WeightedAverage,

    /// Consensus (require agreement)
    Consensus { threshold: f64 },

    /// Union (keep all)
    Union,
}

/// Inspect epistemic metadata
#[derive(Debug, Clone)]
pub struct InspectOp {
    /// Knowledge to inspect
    pub target: Box<Expr>,

    /// What to extract
    pub field: InspectField,
}

impl InspectOp {
    /// Create a new inspect operation
    pub fn new(target: Expr, field: InspectField) -> Self {
        Self {
            target: Box::new(target),
            field,
        }
    }

    /// Inspect confidence
    pub fn confidence(target: Expr) -> Self {
        Self::new(target, InspectField::Confidence)
    }

    /// Inspect provenance
    pub fn provenance(target: Expr) -> Self {
        Self::new(target, InspectField::Provenance)
    }
}

/// Field to inspect
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectField {
    /// Get confidence value
    Confidence,
    /// Get source information
    Source,
    /// Get domain binding
    Domain,
    /// Get provenance trace
    Provenance,
    /// Get temporal context
    Temporal,
    /// Get all metadata
    All,
}

// === Public API functions ===

/// Assert knowledge into the system
pub fn assert_knowledge(
    value: Expr,
    domain: OntologyBinding,
    status: Option<EpistemicStatus>,
) -> KnowledgeOp {
    let mut op = AssertOp::new(value, domain);
    if let Some(s) = status {
        op = op.with_status(s);
    }
    KnowledgeOp::Assert(op)
}

/// Query knowledge from the system
pub fn query_knowledge(
    domain: Option<OntologyBinding>,
    min_confidence: Option<f64>,
) -> KnowledgeOp {
    let mut query = QueryOp::new();
    if let Some(d) = domain {
        query = query.with_domain(d);
    }
    if let Some(c) = min_confidence {
        query = query.with_min_confidence(c);
    }
    KnowledgeOp::Query(query)
}

/// Revise existing knowledge
pub fn revise_knowledge(target: Expr, evidence: Expr, strategy: RevisionStrategy) -> KnowledgeOp {
    KnowledgeOp::Revise(ReviseOp::new(target, evidence, strategy))
}

/// Translate between ontologies
pub fn translate_knowledge(source: Expr, to_ontology: OntologyBinding) -> KnowledgeOp {
    KnowledgeOp::Translate(TranslateOp::new(source, to_ontology))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_epistemic_constraint_min_confidence() {
        let constraint = EpistemicConstraint::MinConfidence(0.8);
        let high_conf = EpistemicStatus::empirical(0.9, Source::Axiom);
        let low_conf = EpistemicStatus::empirical(0.5, Source::Axiom);

        assert!(constraint.satisfied_by(&high_conf));
        assert!(!constraint.satisfied_by(&low_conf));
    }

    #[test]
    fn test_revision_strategy_bayesian() {
        let strategy = RevisionStrategy::Bayesian { prior_weight: 0.3 };
        let old = Confidence::new(0.8);
        let new = Confidence::new(0.9);

        let result = strategy.apply(old, new);
        // 0.8 * 0.3 + 0.9 * 0.7 = 0.24 + 0.63 = 0.87
        assert!((result.value() - 0.87).abs() < 0.001);
    }

    #[test]
    fn test_relational_constraint() {
        let constraint = RelationalConstraint::with_object("inhibits", "target_gene");
        assert!(constraint.subject.is_none());
        assert_eq!(constraint.relation, "inhibits");
        assert_eq!(constraint.object, Some("target_gene".to_string()));
    }
}
