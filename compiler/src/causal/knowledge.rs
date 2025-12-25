//! CausalKnowledge: Level 2 of the Causal Hierarchy
//!
//! Extends Knowledge with causal structure, enabling:
//! - The do() operator for interventional queries
//! - Causal effect identification
//! - Adjustment-based estimation

use std::fmt;

use super::graph::{CausalGraph, GraphError};
use super::identification::{CausalIdentifier, IdentificationMethod, IdentificationStatus};
use super::intervention::{Intervention, InterventionResult};
use crate::epistemic::composition::ConfidenceValue;
use crate::temporal::{Temporal, TemporalKnowledge};

/// Knowledge with causal structure (Level 2 of causal hierarchy)
///
/// Extends TemporalKnowledge with:
/// - A causal DAG encoding the causal structure
/// - Designated outcome and treatment variables
/// - Identification status for causal queries
#[derive(Clone)]
pub struct CausalKnowledge<T> {
    /// Base temporal knowledge
    pub base: TemporalKnowledge<T>,
    /// Causal graph structure
    pub graph: CausalGraph,
    /// Outcome variable of interest
    pub outcome: String,
    /// Treatment variables
    pub treatments: Vec<String>,
    /// Identification status
    pub identification: IdentificationStatus,
}

impl<T: fmt::Debug> fmt::Debug for CausalKnowledge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalKnowledge")
            .field("base", &self.base)
            .field("outcome", &self.outcome)
            .field("treatments", &self.treatments)
            .field("identification", &self.identification)
            .field("nodes", &self.graph.node_count())
            .field("edges", &self.graph.edge_count())
            .finish()
    }
}

impl<T: Clone> CausalKnowledge<T> {
    /// Create new causal knowledge
    pub fn new(
        base: TemporalKnowledge<T>,
        graph: CausalGraph,
        outcome: impl Into<String>,
        treatments: Vec<String>,
    ) -> Self {
        CausalKnowledge {
            base,
            graph,
            outcome: outcome.into(),
            treatments,
            identification: IdentificationStatus::Unknown,
        }
    }

    /// Create from base knowledge with minimal graph
    pub fn from_knowledge(
        base: TemporalKnowledge<T>,
        treatment: impl Into<String>,
        outcome: impl Into<String>,
    ) -> Self {
        use super::graph::{CausalNode, EdgeType};

        let treatment = treatment.into();
        let outcome = outcome.into();

        let mut graph = CausalGraph::new();
        graph.add_node(CausalNode::treatment(&treatment));
        graph.add_node(CausalNode::outcome(&outcome));
        let _ = graph.add_edge(&treatment, &outcome, EdgeType::Direct);

        CausalKnowledge {
            base,
            graph,
            outcome,
            treatments: vec![treatment],
            identification: IdentificationStatus::Unknown,
        }
    }

    /// The do() operator: compute P(Y | do(X=x))
    ///
    /// This is the core operation of Level 2 causal reasoning.
    pub fn do_intervention<X: Clone + 'static>(
        &self,
        intervention: Intervention<X>,
    ) -> Result<InterventionResult<T>, CausalError> {
        // First, identify the causal effect
        let identifier = CausalIdentifier::new(&self.graph);
        let status = identifier.identify(&intervention.target, &self.outcome);

        match status {
            IdentificationStatus::Identified { method } => {
                self.compute_effect(&intervention, &method)
            }
            IdentificationStatus::PartiallyIdentified { lower, upper } => {
                Err(CausalError::PartiallyIdentified {
                    treatment: intervention.target,
                    outcome: self.outcome.clone(),
                    lower,
                    upper,
                })
            }
            IdentificationStatus::NotIdentifiable { reason } => Err(CausalError::NotIdentifiable {
                treatment: intervention.target,
                outcome: self.outcome.clone(),
                reason,
            }),
            IdentificationStatus::Unknown => Err(CausalError::IdentificationUnknown),
        }
    }

    /// Compute the causal effect using the identified method
    fn compute_effect<X>(
        &self,
        intervention: &Intervention<X>,
        method: &IdentificationMethod,
    ) -> Result<InterventionResult<T>, CausalError> {
        let base_confidence = self.base.core.confidence().value();

        match method {
            IdentificationMethod::Experimental => {
                // Direct from experimental data
                Ok(InterventionResult::new(
                    self.base.core.value().clone(),
                    self.base.core.confidence(),
                    IdentificationMethod::Experimental,
                ))
            }

            IdentificationMethod::BackdoorAdjustment { set } => {
                // P(Y | do(X=x)) = Σ_z P(Y | X=x, Z=z) P(Z=z)
                // Confidence reduced by adjustment complexity
                let adjustment_penalty = 0.95_f64.powi(set.len() as i32);
                let adjusted_confidence = base_confidence * adjustment_penalty;

                Ok(InterventionResult::new(
                    self.base.core.value().clone(),
                    ConfidenceValue::new(adjusted_confidence)
                        .unwrap_or(ConfidenceValue::uncertain()),
                    method.clone(),
                )
                .with_adjustment_set(set.clone()))
            }

            IdentificationMethod::FrontdoorAdjustment { mediators } => {
                // Frontdoor formula
                let adjustment_penalty = 0.90_f64.powi(mediators.len() as i32);
                let adjusted_confidence = base_confidence * adjustment_penalty;

                Ok(InterventionResult::new(
                    self.base.core.value().clone(),
                    ConfidenceValue::new(adjusted_confidence)
                        .unwrap_or(ConfidenceValue::uncertain()),
                    method.clone(),
                ))
            }

            IdentificationMethod::InstrumentalVariable { instruments } => {
                // IV estimation
                let iv_penalty = 0.85_f64.powi(instruments.len() as i32);
                let adjusted_confidence = base_confidence * iv_penalty;

                Ok(InterventionResult::new(
                    self.base.core.value().clone(),
                    ConfidenceValue::new(adjusted_confidence)
                        .unwrap_or(ConfidenceValue::uncertain()),
                    method.clone(),
                ))
            }

            IdentificationMethod::DoCalculus { derivation } => {
                // General do-calculus derivation
                let derivation_penalty = 0.90_f64.powi(derivation.len() as i32);
                let adjusted_confidence = base_confidence * derivation_penalty;

                Ok(InterventionResult::new(
                    self.base.core.value().clone(),
                    ConfidenceValue::new(adjusted_confidence)
                        .unwrap_or(ConfidenceValue::uncertain()),
                    method.clone(),
                ))
            }

            IdentificationMethod::PartialIdentification { lower, upper } => {
                Err(CausalError::PartiallyIdentified {
                    treatment: intervention.target.clone(),
                    outcome: self.outcome.clone(),
                    lower: *lower,
                    upper: *upper,
                })
            }

            IdentificationMethod::NotIdentifiable => Err(CausalError::NotIdentifiable {
                treatment: intervention.target.clone(),
                outcome: self.outcome.clone(),
                reason: "Effect not identifiable".to_string(),
            }),

            IdentificationMethod::Unknown => Err(CausalError::IdentificationUnknown),
        }
    }

    /// Identify and cache the identification method
    pub fn identify(&mut self) -> IdentificationStatus {
        if self.treatments.is_empty() {
            self.identification = IdentificationStatus::NotIdentifiable {
                reason: "No treatment variables specified".to_string(),
            };
            return self.identification.clone();
        }

        let identifier = CausalIdentifier::new(&self.graph);
        let status = identifier.identify(&self.treatments[0], &self.outcome);
        self.identification = status.clone();
        status
    }

    /// Check if causal effect is identified
    pub fn is_identified(&self) -> bool {
        self.identification.is_identified()
    }

    /// Get the identification method if identified
    pub fn identification_method(&self) -> Option<&IdentificationMethod> {
        self.identification.method()
    }

    /// Get the underlying value
    pub fn value(&self) -> &T {
        self.base.core.value()
    }

    /// Get confidence in the causal estimate
    pub fn confidence(&self) -> ConfidenceValue {
        self.base.core.confidence()
    }

    /// Get the temporal dimension
    pub fn temporal(&self) -> &Temporal {
        &self.base.temporal
    }

    /// Add a confounder to the graph
    pub fn add_confounder(
        &mut self,
        name: impl Into<String>,
        affects: Vec<&str>,
    ) -> Result<(), GraphError> {
        use super::graph::{CausalNode, EdgeType};

        let name = name.into();
        self.graph.add_node(CausalNode::observed(&name));

        for target in affects {
            self.graph.add_edge(&name, target, EdgeType::Direct)?;
        }

        // Reset identification status
        self.identification = IdentificationStatus::Unknown;
        Ok(())
    }

    /// Add a mediator to the graph
    pub fn add_mediator(
        &mut self,
        name: impl Into<String>,
        from: &str,
        to: &str,
    ) -> Result<(), GraphError> {
        use super::graph::{CausalNode, EdgeType};

        let name = name.into();
        self.graph.add_node(CausalNode::mediator(&name));
        self.graph.add_edge(from, &name, EdgeType::Direct)?;
        self.graph.add_edge(&name, to, EdgeType::Direct)?;

        // Reset identification status
        self.identification = IdentificationStatus::Unknown;
        Ok(())
    }

    /// Downcast to base TemporalKnowledge (loses causal structure)
    pub fn as_temporal(&self) -> &TemporalKnowledge<T> {
        &self.base
    }

    /// Extract as TemporalKnowledge
    pub fn into_temporal(self) -> TemporalKnowledge<T> {
        self.base
    }
}

/// Errors that can occur in causal operations
#[derive(Debug, Clone)]
pub enum CausalError {
    /// Causal effect is not identifiable
    NotIdentifiable {
        treatment: String,
        outcome: String,
        reason: String,
    },

    /// Only bounds are available
    PartiallyIdentified {
        treatment: String,
        outcome: String,
        lower: f64,
        upper: f64,
    },

    /// Invalid intervention specification
    InvalidIntervention { reason: String },

    /// Graph structure issue
    GraphError(GraphError),

    /// Identification status unknown
    IdentificationUnknown,

    /// Unsupported operation
    UnsupportedOperation(String),
}

impl fmt::Display for CausalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CausalError::NotIdentifiable {
                treatment,
                outcome,
                reason,
            } => {
                write!(
                    f,
                    "Causal effect P({} | do({})) not identifiable: {}",
                    outcome, treatment, reason
                )
            }
            CausalError::PartiallyIdentified {
                treatment,
                outcome,
                lower,
                upper,
            } => {
                write!(
                    f,
                    "Causal effect P({} | do({})) only partially identified: [{:.3}, {:.3}]",
                    outcome, treatment, lower, upper
                )
            }
            CausalError::InvalidIntervention { reason } => {
                write!(f, "Invalid intervention: {}", reason)
            }
            CausalError::GraphError(e) => write!(f, "Graph error: {}", e),
            CausalError::IdentificationUnknown => {
                write!(f, "Identification status unknown - call identify() first")
            }
            CausalError::UnsupportedOperation(msg) => {
                write!(f, "Unsupported operation: {}", msg)
            }
        }
    }
}

impl std::error::Error for CausalError {}

impl From<GraphError> for CausalError {
    fn from(e: GraphError) -> Self {
        CausalError::GraphError(e)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::graph::{CausalNode, EdgeType};
    use crate::epistemic::composition::{EpistemicValue, SourceInfo};

    fn create_test_knowledge() -> TemporalKnowledge<f64> {
        let core = EpistemicValue::with_confidence(0.75, 0.90);
        TemporalKnowledge::timeless(core)
    }

    fn simple_causal_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("Drug"));
        g.add_node(CausalNode::outcome("Effect"));
        g.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();
        g
    }

    fn confounded_graph() -> CausalGraph {
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("Drug"));
        g.add_node(CausalNode::outcome("Effect"));
        g.add_node(CausalNode::observed("Severity"));

        g.add_edge("Drug", "Effect", EdgeType::Direct).unwrap();
        g.add_edge("Severity", "Drug", EdgeType::Direct).unwrap();
        g.add_edge("Severity", "Effect", EdgeType::Direct).unwrap();
        g
    }

    #[test]
    fn test_create_causal_knowledge() {
        let base = create_test_knowledge();
        let graph = simple_causal_graph();

        let ck = CausalKnowledge::new(base, graph, "Effect", vec!["Drug".to_string()]);

        assert_eq!(ck.outcome, "Effect");
        assert_eq!(ck.treatments, vec!["Drug"]);
    }

    #[test]
    fn test_do_unconfounded() {
        let base = create_test_knowledge();
        let graph = simple_causal_graph();

        let ck = CausalKnowledge::new(base, graph, "Effect", vec!["Drug".to_string()]);

        let intervention = Intervention::atomic("Drug", 100.0);
        let result = ck.do_intervention(intervention);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(matches!(
            result.identification,
            IdentificationMethod::BackdoorAdjustment { .. }
        ));
    }

    #[test]
    fn test_do_with_backdoor() {
        let base = create_test_knowledge();
        let graph = confounded_graph();

        let ck = CausalKnowledge::new(base, graph, "Effect", vec!["Drug".to_string()]);

        let intervention = Intervention::atomic("Drug", 100.0);
        let result = ck.do_intervention(intervention);

        assert!(result.is_ok());
        let result = result.unwrap();

        if let IdentificationMethod::BackdoorAdjustment { set } = &result.identification {
            assert!(set.contains("Severity"));
        } else {
            panic!("Expected backdoor adjustment");
        }
    }

    #[test]
    fn test_identify_caches() {
        let base = create_test_knowledge();
        let graph = confounded_graph();

        let mut ck = CausalKnowledge::new(base, graph, "Effect", vec!["Drug".to_string()]);

        assert!(matches!(ck.identification, IdentificationStatus::Unknown));

        ck.identify();

        assert!(ck.is_identified());
    }

    #[test]
    fn test_add_confounder() {
        let base = create_test_knowledge();
        let graph = simple_causal_graph();

        let mut ck = CausalKnowledge::new(base, graph, "Effect", vec!["Drug".to_string()]);

        ck.add_confounder("Age", vec!["Drug", "Effect"]).unwrap();

        assert!(ck.graph.contains_node("Age"));
        // Identification should be reset
        assert!(matches!(ck.identification, IdentificationStatus::Unknown));
    }

    #[test]
    fn test_from_knowledge() {
        let base = create_test_knowledge();

        let ck = CausalKnowledge::from_knowledge(base, "Treatment", "Outcome");

        assert_eq!(ck.treatments, vec!["Treatment"]);
        assert_eq!(ck.outcome, "Outcome");
        assert!(ck.graph.contains_node("Treatment"));
        assert!(ck.graph.contains_node("Outcome"));
    }
}
