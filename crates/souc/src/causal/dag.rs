//! Epistemic Causal DAG with Uncertain Edges
//!
//! This module implements a causal directed acyclic graph where:
//! - Nodes represent variables with optional ontology bindings
//! - Edges have BetaConfidence indicating uncertainty about edge existence
//! - Edge strengths are EffectEstimate<f64> with full epistemic metadata
//!
//! # Example
//!
//! ```ignore
//! use sounio::causal::dag::*;
//! use sounio::epistemic::BetaConfidence;
//!
//! let mut dag = CausalDAG::new();
//!
//! // Add nodes with ontology bindings
//! dag.add_node(EpistemicCausalNode::treatment("Aspirin")
//!     .with_ontology("CHEBI", "15365"));
//!
//! dag.add_node(EpistemicCausalNode::outcome("HeadacheRelief")
//!     .with_ontology("PATO", "0000467"));
//!
//! // Add edge with uncertain existence
//! dag.add_edge(UncertainCausalEdge::new(
//!     "Aspirin",
//!     "HeadacheRelief",
//!     BetaConfidence::new(9.0, 2.0),  // 90% confident edge exists
//!     EffectEstimate::certain(0.7),   // Effect size = 0.7
//!     UncertainEdgeType::Direct,
//! ))?;
//! ```

use std::collections::HashMap;
use std::fmt;

use crate::epistemic::BetaConfidence;

/// Effect estimate wrapper for causal effect values
///
/// Tracks a value with its uncertainty (variance) and confidence.
/// Named to avoid conflict with epistemic::Knowledge.
#[derive(Clone, Debug, PartialEq)]
pub struct EffectEstimate<T> {
    /// The value itself
    pub value: T,
    /// Variance/uncertainty in the value
    pub variance: Option<f64>,
    /// Confidence in this estimate [0, 1]
    pub confidence: f64,
    /// Source of this estimate
    pub source: String,
}

impl<T> EffectEstimate<T> {
    /// Create estimate with full certainty
    pub fn certain(value: T) -> Self {
        EffectEstimate {
            value,
            variance: None,
            confidence: 1.0,
            source: "Certain".to_string(),
        }
    }

    /// Create estimate with specified confidence
    pub fn with_confidence(value: T, confidence: f64) -> Self {
        EffectEstimate {
            value,
            variance: None,
            confidence: confidence.clamp(0.0, 1.0),
            source: "WithConfidence".to_string(),
        }
    }

    /// Create estimate with variance
    pub fn with_variance(value: T, variance: f64, confidence: f64) -> Self {
        EffectEstimate {
            value,
            variance: Some(variance),
            confidence: confidence.clamp(0.0, 1.0),
            source: "WithVariance".to_string(),
        }
    }
}

/// Type of causal node in epistemic DAG
///
/// Extended from graph::NodeType with Confounder and Collider.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EpistemicNodeType {
    /// Treatment/intervention variable
    Treatment,
    /// Outcome variable of interest
    Outcome,
    /// Confounder (common cause of treatment and outcome)
    Confounder,
    /// Mediator (on causal path from treatment to outcome)
    Mediator,
    /// Collider (common effect of two variables)
    Collider,
    /// Generic observed variable
    Observed,
    /// Latent/unobserved variable
    Latent,
}

/// Type of uncertain causal edge
///
/// Named to avoid conflict with graph::EdgeType.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum UncertainEdgeType {
    /// Direct causal effect: X → Y
    Direct,
    /// Confounding path (through latent variable)
    Confounding,
    /// Mediating path
    Mediating,
}

/// A node in the epistemic causal DAG
///
/// Named to avoid conflict with graph::CausalNode.
#[derive(Clone, Debug, PartialEq)]
pub struct EpistemicCausalNode {
    /// Node name/identifier
    pub name: String,
    /// Type of node in causal structure
    pub node_type: EpistemicNodeType,
    /// Prior distribution over this variable (if specified)
    pub distribution: Option<BetaConfidence>,
    /// Optional ontology binding (e.g., "CHEBI:15365" for aspirin)
    pub ontology_uri: Option<String>,
}

impl EpistemicCausalNode {
    /// Create a new causal node
    pub fn new(name: impl Into<String>, node_type: EpistemicNodeType) -> Self {
        EpistemicCausalNode {
            name: name.into(),
            node_type,
            distribution: None,
            ontology_uri: None,
        }
    }

    /// Create a treatment node
    pub fn treatment(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Treatment)
    }

    /// Create an outcome node
    pub fn outcome(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Outcome)
    }

    /// Create a confounder node
    pub fn confounder(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Confounder)
    }

    /// Create a mediator node
    pub fn mediator(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Mediator)
    }

    /// Create a collider node
    pub fn collider(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Collider)
    }

    /// Create an observed variable node
    pub fn observed(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Observed)
    }

    /// Create a latent/unobserved node
    pub fn latent(name: impl Into<String>) -> Self {
        Self::new(name, EpistemicNodeType::Latent)
    }

    /// Set the prior distribution over this variable
    pub fn with_distribution(mut self, dist: BetaConfidence) -> Self {
        self.distribution = Some(dist);
        self
    }

    /// Bind to an ontology term
    pub fn with_ontology(mut self, ontology: &str, term_id: &str) -> Self {
        self.ontology_uri = Some(format!("{}:{}", ontology, term_id));
        self
    }

    /// Set ontology URI directly
    pub fn with_ontology_uri(mut self, uri: impl Into<String>) -> Self {
        self.ontology_uri = Some(uri.into());
        self
    }
}

/// An edge in the epistemic causal DAG with uncertainty
///
/// Edges carry two types of uncertainty:
/// 1. **Existence uncertainty**: BetaConfidence on whether the edge exists
/// 2. **Strength uncertainty**: EffectEstimate<f64> for the effect size
///
/// Named to avoid conflict with graph edge types.
#[derive(Clone, Debug)]
pub struct UncertainCausalEdge {
    /// Source node name
    pub from: String,
    /// Target node name
    pub to: String,
    /// Confidence that this edge exists
    pub confidence: BetaConfidence,
    /// Causal effect strength (with uncertainty)
    pub strength: EffectEstimate<f64>,
    /// Type of edge
    pub edge_type: UncertainEdgeType,
}

impl UncertainCausalEdge {
    /// Create a new causal edge
    pub fn new(
        from: impl Into<String>,
        to: impl Into<String>,
        confidence: BetaConfidence,
        strength: EffectEstimate<f64>,
        edge_type: UncertainEdgeType,
    ) -> Self {
        UncertainCausalEdge {
            from: from.into(),
            to: to.into(),
            confidence,
            strength,
            edge_type,
        }
    }

    /// Create a direct causal edge
    pub fn direct(
        from: impl Into<String>,
        to: impl Into<String>,
        confidence: BetaConfidence,
        strength: EffectEstimate<f64>,
    ) -> Self {
        Self::new(from, to, confidence, strength, UncertainEdgeType::Direct)
    }

    /// Create a confounding edge
    pub fn confounding(
        from: impl Into<String>,
        to: impl Into<String>,
        confidence: BetaConfidence,
        strength: EffectEstimate<f64>,
    ) -> Self {
        Self::new(
            from,
            to,
            confidence,
            strength,
            UncertainEdgeType::Confounding,
        )
    }

    /// Create a mediating edge
    pub fn mediating(
        from: impl Into<String>,
        to: impl Into<String>,
        confidence: BetaConfidence,
        strength: EffectEstimate<f64>,
    ) -> Self {
        Self::new(from, to, confidence, strength, UncertainEdgeType::Mediating)
    }

    /// Get the expected (mean) edge strength
    pub fn expected_strength(&self) -> f64 {
        self.strength.value
    }

    /// Get the probability this edge exists
    pub fn existence_probability(&self) -> f64 {
        self.confidence.mean()
    }
}

/// Epistemic Causal DAG with uncertain edges
///
/// Unlike traditional causal graphs where edges either exist or don't,
/// this DAG models uncertainty about:
/// - Whether an edge exists (BetaConfidence)
/// - The strength of causal effects (EffectEstimate<f64>)
#[derive(Clone, Debug)]
pub struct CausalDAG {
    /// Nodes in the graph
    nodes: HashMap<String, EpistemicCausalNode>,
    /// Edges with epistemic uncertainty
    edges: Vec<UncertainCausalEdge>,
}

impl Default for CausalDAG {
    fn default() -> Self {
        Self::new()
    }
}

impl CausalDAG {
    /// Create a new empty causal DAG
    pub fn new() -> Self {
        CausalDAG {
            nodes: HashMap::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the DAG
    pub fn add_node(&mut self, node: EpistemicCausalNode) {
        self.nodes.insert(node.name.clone(), node);
    }

    /// Add an edge to the DAG
    ///
    /// Returns error if source or target nodes don't exist
    pub fn add_edge(&mut self, edge: UncertainCausalEdge) -> Result<(), DAGError> {
        if !self.nodes.contains_key(&edge.from) {
            return Err(DAGError::NodeNotFound(edge.from.clone()));
        }
        if !self.nodes.contains_key(&edge.to) {
            return Err(DAGError::NodeNotFound(edge.to.clone()));
        }

        // Check for cycles (simplified - full cycle detection would use DFS)
        if edge.from == edge.to {
            return Err(DAGError::CycleDetected {
                from: edge.from.clone(),
                to: edge.to.clone(),
            });
        }

        self.edges.push(edge);
        Ok(())
    }

    /// Get a node by name
    pub fn get_node(&self, name: &str) -> Option<&EpistemicCausalNode> {
        self.nodes.get(name)
    }

    /// Get all nodes
    pub fn nodes(&self) -> impl Iterator<Item = &EpistemicCausalNode> {
        self.nodes.values()
    }

    /// Get all edges
    pub fn edges(&self) -> &[UncertainCausalEdge] {
        &self.edges
    }

    /// Get edges originating from a node
    pub fn edges_from(&self, node: &str) -> Vec<&UncertainCausalEdge> {
        self.edges.iter().filter(|e| e.from == node).collect()
    }

    /// Get edges pointing to a node
    pub fn edges_to(&self, node: &str) -> Vec<&UncertainCausalEdge> {
        self.edges.iter().filter(|e| e.to == node).collect()
    }

    /// Get parent nodes (direct causes)
    pub fn parents(&self, node: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.to == node)
            .map(|e| e.from.as_str())
            .collect()
    }

    /// Get children nodes (direct effects)
    pub fn children(&self, node: &str) -> Vec<&str> {
        self.edges
            .iter()
            .filter(|e| e.from == node)
            .map(|e| e.to.as_str())
            .collect()
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Check if node exists
    pub fn contains_node(&self, name: &str) -> bool {
        self.nodes.contains_key(name)
    }

    /// Get all treatment nodes
    pub fn treatments(&self) -> Vec<&EpistemicCausalNode> {
        self.nodes
            .values()
            .filter(|n| matches!(n.node_type, EpistemicNodeType::Treatment))
            .collect()
    }

    /// Get all outcome nodes
    pub fn outcomes(&self) -> Vec<&EpistemicCausalNode> {
        self.nodes
            .values()
            .filter(|n| matches!(n.node_type, EpistemicNodeType::Outcome))
            .collect()
    }

    /// Get all confounders
    pub fn confounders(&self) -> Vec<&EpistemicCausalNode> {
        self.nodes
            .values()
            .filter(|n| matches!(n.node_type, EpistemicNodeType::Confounder))
            .collect()
    }

    /// Get all mediators
    pub fn mediators(&self) -> Vec<&EpistemicCausalNode> {
        self.nodes
            .values()
            .filter(|n| matches!(n.node_type, EpistemicNodeType::Mediator))
            .collect()
    }

    /// Find edge between two nodes
    pub fn find_edge(&self, from: &str, to: &str) -> Option<&UncertainCausalEdge> {
        self.edges.iter().find(|e| e.from == from && e.to == to)
    }

    /// Remove an edge
    pub fn remove_edge(&mut self, from: &str, to: &str) -> bool {
        let initial_len = self.edges.len();
        self.edges.retain(|e| !(e.from == from && e.to == to));
        self.edges.len() < initial_len
    }

    /// Create a modified graph with incoming edges to X removed (for do(X))
    ///
    /// This is the G_X̄ operation from Pearl's do-calculus
    pub fn intervene_remove_incoming(&self, node: &str) -> CausalDAG {
        let mut dag = self.clone();
        dag.edges.retain(|e| e.to != node);
        dag
    }

    /// Create a modified graph with outgoing edges from X removed
    ///
    /// This is the G_X_ operation from Pearl's do-calculus
    pub fn remove_outgoing(&self, node: &str) -> CausalDAG {
        let mut dag = self.clone();
        dag.edges.retain(|e| e.from != node);
        dag
    }

    /// Compute the expected total causal effect along a path
    ///
    /// Multiplies edge strengths along the path, accounting for edge existence probability
    pub fn path_effect(&self, path: &[String]) -> EffectEstimate<f64> {
        if path.len() < 2 {
            return EffectEstimate::certain(0.0);
        }

        let mut total_effect = 1.0;
        let mut total_variance = 0.0;
        let mut min_confidence: f64 = 1.0;

        for i in 0..path.len() - 1 {
            if let Some(edge) = self.find_edge(&path[i], &path[i + 1]) {
                total_effect *= edge.expected_strength();
                total_variance += edge.strength.variance.unwrap_or(0.0);
                min_confidence = min_confidence.min(edge.existence_probability());
            } else {
                // Edge doesn't exist
                return EffectEstimate::certain(0.0);
            }
        }

        EffectEstimate {
            value: total_effect,
            variance: Some(total_variance),
            confidence: min_confidence,
            source: "PathEffect".to_string(),
        }
    }
}

/// Errors that can occur in DAG operations
#[derive(Debug, Clone)]
pub enum DAGError {
    /// Node not found in DAG
    NodeNotFound(String),
    /// Adding edge would create a cycle
    CycleDetected { from: String, to: String },
    /// Invalid operation
    InvalidOperation(String),
}

impl fmt::Display for DAGError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DAGError::NodeNotFound(name) => write!(f, "Node '{}' not found in DAG", name),
            DAGError::CycleDetected { from, to } => {
                write!(f, "Adding edge {} -> {} would create a cycle", from, to)
            }
            DAGError::InvalidOperation(msg) => write!(f, "Invalid operation: {}", msg),
        }
    }
}

impl std::error::Error for DAGError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_dag() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    fn test_add_edge() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );

        assert!(dag.add_edge(edge).is_ok());
        assert_eq!(dag.edge_count(), 1);
    }

    #[test]
    fn test_edge_not_found() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));

        let edge = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );

        let result = dag.add_edge(edge);
        assert!(matches!(result, Err(DAGError::NodeNotFound(_))));
    }

    #[test]
    fn test_parents_children() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::mediator("M"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge1 = UncertainCausalEdge::direct(
            "X",
            "M",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge2 = UncertainCausalEdge::direct(
            "M",
            "Y",
            BetaConfidence::new(8.0, 2.0),
            EffectEstimate::certain(0.3),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();

        assert_eq!(dag.parents("M"), vec!["X"]);
        assert_eq!(dag.children("M"), vec!["Y"]);
        assert!(dag.parents("X").is_empty());
        assert!(dag.children("Y").is_empty());
    }

    #[test]
    fn test_node_types() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("T"));
        dag.add_node(EpistemicCausalNode::outcome("O"));
        dag.add_node(EpistemicCausalNode::confounder("C"));
        dag.add_node(EpistemicCausalNode::mediator("M"));

        assert_eq!(dag.treatments().len(), 1);
        assert_eq!(dag.outcomes().len(), 1);
        assert_eq!(dag.confounders().len(), 1);
        assert_eq!(dag.mediators().len(), 1);
    }

    #[test]
    fn test_ontology_binding() {
        let node = EpistemicCausalNode::treatment("Aspirin").with_ontology("CHEBI", "15365");

        assert_eq!(node.ontology_uri, Some("CHEBI:15365".to_string()));
    }

    #[test]
    fn test_intervene_remove_incoming() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::confounder("U"));
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge1 = UncertainCausalEdge::direct(
            "U",
            "X",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge2 = UncertainCausalEdge::direct(
            "U",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge3 = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.3),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();
        dag.add_edge(edge3).unwrap();

        let dag_x = dag.intervene_remove_incoming("X");

        // U -> X should be removed
        assert_eq!(dag_x.edge_count(), 2);
        assert!(dag_x.parents("X").is_empty());
        assert_eq!(dag_x.children("X"), vec!["Y"]);
    }

    #[test]
    fn test_path_effect() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::mediator("M"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge1 = UncertainCausalEdge::direct(
            "X",
            "M",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        let edge2 = UncertainCausalEdge::direct(
            "M",
            "Y",
            BetaConfidence::new(8.0, 2.0),
            EffectEstimate::certain(0.4),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();

        let path = vec!["X".to_string(), "M".to_string(), "Y".to_string()];
        let effect = dag.path_effect(&path);

        // Expected: 0.5 * 0.4 = 0.2
        assert!((effect.value - 0.2).abs() < 0.001);
    }
}
