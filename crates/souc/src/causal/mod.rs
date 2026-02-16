//! Causal Primitives: Do-Calculus as L0 Operator
//!
//! Day 34 of the Sounio compiler implements Pearl's causal hierarchy:
//!
//! # Ladder of Causation
//!
//! ```text
//! Level 1: ASSOCIATION (Seeing)
//!   Query: P(Y | X)
//!   Type: Knowledge[τ, ε, δ, Φ, t]
//!
//! Level 2: INTERVENTION (Doing)
//!   Query: P(Y | do(X))
//!   Type: CausalKnowledge[τ, ε, δ, Φ, t, G]
//!
//! Level 3: COUNTERFACTUAL (Imagining)
//!   Query: P(Y_x | X=x', Y=y)
//!   Type: StructuralKnowledge[τ, ε, δ, Φ, t, M]
//! ```
//!
//! # Key Components
//!
//! - [`CausalGraph`]: Directed acyclic graph with d-separation
//! - [`CausalKnowledge`]: Knowledge with causal structure (Level 2)
//! - [`StructuralKnowledge`]: Full structural causal model (Level 3)
//! - [`Intervention`]: The do() operator
//! - [`CounterfactualResult`]: Results of counterfactual queries
//!
//! # Example
//!
//! ```ignore
//! use sounio::causal::*;
//!
//! // Build causal graph
//! let mut graph = CausalGraph::new();
//! graph.add_node(CausalNode::treatment("Dose"));
//! graph.add_node(CausalNode::outcome("Effect"));
//! graph.add_edge("Dose", "Effect", EdgeType::Direct)?;
//!
//! // Create causal knowledge
//! let causal = CausalKnowledge::new(knowledge, graph, "Effect", vec!["Dose"]);
//!
//! // Compute causal effect
//! let result = causal.do_intervention(Intervention::atomic("Dose", 100.0))?;
//! ```

pub mod composition;
pub mod counterfactual;
pub mod graph;
pub mod identification;
pub mod intervention;
pub mod knowledge;
pub mod refutation;
pub mod structural;
pub mod uplift;
pub mod uplift_syntax;
pub mod z3_identify;

// Causal discovery (PC algorithm)
pub mod discovery;

// Epistemic-causal modules (new)
pub mod dag;
pub mod do_calculus;
pub mod effects;
pub mod identifiability;

// Re-exports
pub use composition::*;
pub use counterfactual::*;
pub use graph::*;
pub use identification::*;
pub use intervention::*;
pub use knowledge::*;
pub use refutation::*;
pub use structural::*;
pub use uplift::*;
pub use uplift_syntax::*;
pub use z3_identify::*;

// Epistemic-causal re-exports (renamed to avoid conflicts with graph.rs types)
pub use dag::{
    CausalDAG, DAGError, EffectEstimate, EpistemicCausalNode, EpistemicNodeType,
    UncertainCausalEdge, UncertainEdgeType,
};
pub use do_calculus::{AdjustmentSet, AdjustmentType, CausalQuery, DoCalculus};
pub use effects::{
    AverageTreatmentEffect, ConditionalAverageTreatmentEffect, LocalAverageTreatmentEffect,
    MediationEffects, average_treatment_effect, conditional_average_treatment_effect,
    local_average_treatment_effect, mediation_effects,
};
pub use identifiability::{
    BackdoorPath, BackdoorPathAnalysis, check_frontdoor_criterion, d_separation,
    find_backdoor_paths, find_valid_adjustment_sets,
};

// ============================================================================
// Unified Causal API: Builder, Analysis, and Type System Integration
// ============================================================================

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::epistemic::composition::ConfidenceValue;

/// Builder for causal graphs with ergonomic chainable API.
///
/// Wraps [`CausalGraph`] with a fluent interface for constructing
/// causal DAGs, adding nodes/edges, and immediately querying
/// identifiability.
///
/// # Example
///
/// ```ignore
/// let analysis = CausalGraphBuilder::new()
///     .treatment("Drug")
///     .outcome("Recovery")
///     .observed("Age")
///     .edge("Drug", "Recovery")
///     .edge("Age", "Drug")
///     .edge("Age", "Recovery")
///     .analyze();
///
/// assert!(analysis.is_identified());
/// ```
pub struct CausalGraphBuilder {
    graph: CausalGraph,
    treatments: Vec<String>,
    outcomes: Vec<String>,
    bidirected: Vec<(String, String)>,
}

impl CausalGraphBuilder {
    /// Create an empty causal graph builder
    pub fn new() -> Self {
        CausalGraphBuilder {
            graph: CausalGraph::new(),
            treatments: Vec::new(),
            outcomes: Vec::new(),
            bidirected: Vec::new(),
        }
    }

    /// Add a treatment (intervention) variable
    pub fn treatment(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.add_node(CausalNode::treatment(&name));
        self.treatments.push(name);
        self
    }

    /// Add an outcome variable
    pub fn outcome(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.add_node(CausalNode::outcome(&name));
        self.outcomes.push(name);
        self
    }

    /// Add an observed variable
    pub fn observed(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.add_node(CausalNode::observed(&name));
        self
    }

    /// Add a latent (unobserved) variable
    pub fn latent(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.add_node(CausalNode::latent(&name));
        self
    }

    /// Add a mediator variable
    pub fn mediator(mut self, name: impl Into<String>) -> Self {
        let name = name.into();
        self.graph.add_node(CausalNode::mediator(&name));
        self
    }

    /// Add a directed causal edge: from → to
    pub fn edge(mut self, from: impl Into<String>, to: impl Into<String>) -> Self {
        let _ = self
            .graph
            .add_edge(&from.into(), &to.into(), EdgeType::Direct);
        self
    }

    /// Add a bidirected edge (latent common cause): from ↔ to
    pub fn bidirected_edge(mut self, a: impl Into<String>, b: impl Into<String>) -> Self {
        let a = a.into();
        let b = b.into();
        let _ = self.graph.add_edge(&a, &b, EdgeType::Bidirected);
        self.bidirected.push((a, b));
        self
    }

    /// Build the graph without analysis
    pub fn build(self) -> CausalGraph {
        self.graph
    }

    /// Build and immediately run identification analysis
    pub fn analyze(self) -> CausalAnalysis {
        CausalAnalysis::new(self.graph, self.treatments, self.outcomes, self.bidirected)
    }
}

impl Default for CausalGraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Unified causal analysis facade.
///
/// Combines identification, effect estimation, and refutation into
/// a single entry point. Constructed via [`CausalGraphBuilder::analyze`]
/// or directly from components.
///
/// # Causal Hierarchy Integration
///
/// - **Level 1** (Association): Use [`CausalAnalysis::d_separated`]
/// - **Level 2** (Intervention): Use [`CausalAnalysis::identify`], [`CausalAnalysis::ate`]
/// - **Level 3** (Counterfactual): Requires [`StructuralCausalModel`] via [`CausalAnalysis::with_scm`]
pub struct CausalAnalysis {
    /// The causal graph
    pub graph: CausalGraph,
    /// Treatment variables
    pub treatments: Vec<String>,
    /// Outcome variables
    pub outcomes: Vec<String>,
    /// Bidirected edges (latent confounders)
    pub bidirected: Vec<(String, String)>,
    /// Cached identification results per (treatment, outcome) pair
    identification_cache: HashMap<(String, String), IdentificationStatus>,
}

impl CausalAnalysis {
    /// Create a new causal analysis from components
    pub fn new(
        graph: CausalGraph,
        treatments: Vec<String>,
        outcomes: Vec<String>,
        bidirected: Vec<(String, String)>,
    ) -> Self {
        CausalAnalysis {
            graph,
            treatments,
            outcomes,
            bidirected,
            identification_cache: HashMap::new(),
        }
    }

    /// Create from an existing CausalGraph, inferring treatments/outcomes from node types
    pub fn from_graph(graph: CausalGraph) -> Self {
        let mut treatments = Vec::new();
        let mut outcomes = Vec::new();

        for name in graph.node_names() {
            if let Some(node) = graph.get_node(name) {
                match node.node_type {
                    NodeType::Treatment => treatments.push(name.clone()),
                    NodeType::Outcome => outcomes.push(name.clone()),
                    _ => {}
                }
            }
        }

        CausalAnalysis {
            graph,
            treatments,
            outcomes,
            bidirected: Vec::new(),
            identification_cache: HashMap::new(),
        }
    }

    // ---- Level 1: Association (d-separation) ----

    /// Test d-separation: X ⊥⊥ Y | Z in the graph
    pub fn d_separated(&self, x: &str, y: &str, conditioning: &HashSet<String>) -> bool {
        self.graph.d_separated(x, y, conditioning)
    }

    /// Find all ancestors of a node
    pub fn ancestors(&self, node: &str) -> HashSet<String> {
        self.graph.ancestors_of(node)
    }

    /// Find all descendants of a node
    pub fn descendants(&self, node: &str) -> HashSet<String> {
        self.graph.descendants_of(node)
    }

    /// Get topological ordering of the graph
    pub fn topological_order(&self) -> Vec<String> {
        self.graph.topological_order()
    }

    // ---- Level 2: Intervention (identification) ----

    /// Identify the causal effect of treatment on outcome.
    ///
    /// Returns the identification status, which includes the method
    /// (backdoor, frontdoor, IV) and any required adjustment set.
    pub fn identify(&mut self, treatment: &str, outcome: &str) -> &IdentificationStatus {
        let key = (treatment.to_string(), outcome.to_string());
        if !self.identification_cache.contains_key(&key) {
            let identifier = CausalIdentifier::new(&self.graph);
            let status = identifier.identify(treatment, outcome);
            self.identification_cache.insert(key.clone(), status);
        }
        &self.identification_cache[&key]
    }

    /// Check if any treatment → outcome effect is identified
    pub fn is_identified(&mut self) -> bool {
        if self.treatments.is_empty() || self.outcomes.is_empty() {
            return false;
        }
        // Check first treatment-outcome pair
        let t = self.treatments[0].clone();
        let o = self.outcomes[0].clone();
        self.identify(&t, &o).is_identified()
    }

    /// Identify all treatment-outcome pairs and return summary
    pub fn identify_all(&mut self) -> Vec<IdentificationSummary> {
        let treatments = self.treatments.clone();
        let outcomes = self.outcomes.clone();
        let mut results = Vec::new();

        for t in &treatments {
            for o in &outcomes {
                let status = self.identify(t, o).clone();
                results.push(IdentificationSummary {
                    treatment: t.clone(),
                    outcome: o.clone(),
                    identified: status.is_identified(),
                    method: status.method().cloned(),
                    adjustment_set: self.backdoor_set(t, o),
                });
            }
        }

        results
    }

    /// Find a valid backdoor adjustment set for (treatment, outcome)
    pub fn backdoor_set(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let identifier = CausalIdentifier::new(&self.graph);
        identifier.find_backdoor_set(treatment, outcome)
    }

    /// Find a valid frontdoor set for (treatment, outcome)
    pub fn frontdoor_set(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let identifier = CausalIdentifier::new(&self.graph);
        identifier.find_frontdoor_set(treatment, outcome)
    }

    /// Find instrumental variables for (treatment, outcome)
    pub fn instruments(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let identifier = CausalIdentifier::new(&self.graph);
        identifier.find_instruments(treatment, outcome)
    }

    /// Find mediators on directed paths from treatment to outcome
    pub fn mediators(&self, treatment: &str, outcome: &str) -> Vec<String> {
        self.graph.find_mediators(treatment, outcome)
    }

    // ---- Level 3: Counterfactual (requires SCM) ----

    /// Attach a structural causal model for counterfactual queries
    pub fn with_scm(self, model: StructuralCausalModel) -> CausalAnalysisWithSCM {
        CausalAnalysisWithSCM {
            analysis: self,
            model,
        }
    }

    // ---- Uncertainty decomposition ----

    /// Decompose total uncertainty into components for a causal effect.
    ///
    /// Causal uncertainty has three independent sources:
    /// - **Statistical**: sampling variance from finite data
    /// - **Identification**: uncertainty about which adjustment set is valid
    /// - **Structural**: uncertainty about graph structure (edge existence)
    pub fn decompose_uncertainty(
        &mut self,
        treatment: &str,
        outcome: &str,
        statistical_variance: f64,
        edge_confidences: &HashMap<(String, String), f64>,
    ) -> CausalUncertaintyDecomposition {
        // Identification uncertainty: based on how many valid adjustment sets exist
        let backdoor_paths = {
            let dag = CausalDAG::new();
            find_backdoor_paths(&dag, treatment, outcome)
        };
        // Each backdoor path that isn't blocked suggests alternative adjustment possibilities
        let n_adjustment_sets = backdoor_paths.len().max(1);
        let identification_uncertainty: f64 = if n_adjustment_sets <= 1 {
            0.0 // Unique or no adjustment set — low identification uncertainty
        } else {
            // More possible sets → more uncertainty about correct one
            1.0 - 1.0 / (n_adjustment_sets as f64)
        };

        // Structural uncertainty: average uncertainty about edge existence
        let structural_uncertainty: f64 = if edge_confidences.is_empty() {
            0.0
        } else {
            let total: f64 = edge_confidences.values().map(|c| 1.0 - c).sum();
            total / edge_confidences.len() as f64
        };

        // Total: GUM-style quadrature (independent sources)
        let total = (statistical_variance
            + identification_uncertainty * identification_uncertainty
            + structural_uncertainty * structural_uncertainty)
            .sqrt();

        CausalUncertaintyDecomposition {
            statistical: statistical_variance.sqrt(),
            identification: identification_uncertainty,
            structural: structural_uncertainty,
            total,
            n_adjustment_sets,
        }
    }

    // ---- Graph queries ----

    /// Number of nodes in the graph
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of edges in the graph
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Get the mutated graph G_X̄ (remove incoming edges to X)
    pub fn graph_do(&self, x: &str) -> CausalGraph {
        self.graph.graph_do(x)
    }

    /// Check if two graphs are structurally equivalent
    pub fn structurally_equivalent(&self, other: &CausalGraph) -> bool {
        self.graph.structurally_equivalent(other)
    }
}

impl fmt::Debug for CausalAnalysis {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CausalAnalysis")
            .field("nodes", &self.graph.node_count())
            .field("edges", &self.graph.edge_count())
            .field("treatments", &self.treatments)
            .field("outcomes", &self.outcomes)
            .field("bidirected", &self.bidirected.len())
            .field("cached_results", &self.identification_cache.len())
            .finish()
    }
}

/// Causal analysis extended with a structural causal model for Level 3 queries
pub struct CausalAnalysisWithSCM {
    /// Base causal analysis (Level 1 + 2)
    pub analysis: CausalAnalysis,
    /// Structural causal model for counterfactual reasoning
    pub model: StructuralCausalModel,
}

impl CausalAnalysisWithSCM {
    /// Compute counterfactual: what would Y be if X had been x'?
    ///
    /// Uses the SCM's intervene-then-evaluate approach:
    /// 1. Create intervened model M_{x'} via do(X = x')
    /// 2. Evaluate the target variable in the modified world
    /// 3. Return the counterfactual value with evidence as context
    pub fn counterfactual(
        &self,
        target: &str,
        intervention_var: &str,
        intervention_value: f64,
        evidence: &HashMap<String, f64>,
    ) -> CounterfactualResult {
        // Counterfactual world: intervene and evaluate
        let intervened_model = self.model.intervene(intervention_var, intervention_value);
        let cf_values = intervened_model.evaluate(evidence);
        let cf_value = cf_values.get(target).copied().unwrap_or(0.0);

        CounterfactualResult {
            query: CounterfactualQuery {
                target: target.to_string(),
                intervention: intervention_var.to_string(),
                intervention_value,
                evidence: evidence.clone(),
            },
            value: cf_value,
            confidence: ConfidenceValue::new_unchecked(1.0),
            posterior_u: HashMap::new(),
            counterfactual_world: cf_values,
        }
    }

    /// Access the base analysis for Level 1/2 queries
    pub fn analysis(&self) -> &CausalAnalysis {
        &self.analysis
    }

    /// Access the SCM
    pub fn model(&self) -> &StructuralCausalModel {
        &self.model
    }
}

/// Summary of identification result for a single treatment-outcome pair
#[derive(Clone, Debug)]
pub struct IdentificationSummary {
    /// Treatment variable name
    pub treatment: String,
    /// Outcome variable name
    pub outcome: String,
    /// Whether the causal effect is identified
    pub identified: bool,
    /// The identification method (if identified)
    pub method: Option<IdentificationMethod>,
    /// Valid adjustment set (if backdoor-identifiable)
    pub adjustment_set: Option<HashSet<String>>,
}

impl fmt::Display for IdentificationSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let status = if self.identified {
            "IDENTIFIED"
        } else {
            "NOT IDENTIFIED"
        };
        write!(f, "{} -> {}: {}", self.treatment, self.outcome, status)?;
        if let Some(ref method) = self.method {
            write!(f, " via {}", method)?;
        }
        if let Some(ref adj) = self.adjustment_set {
            write!(f, " (adjust: {:?})", adj)?;
        }
        Ok(())
    }
}

/// Decomposition of causal uncertainty into independent components.
///
/// Follows GUM (Guide to the Expression of Uncertainty in Measurement)
/// principles applied to causal inference:
///
/// ```text
/// ε_total = √(ε_stat² + ε_id² + ε_struct²)
/// ```
///
/// where:
/// - ε_stat  = statistical uncertainty from finite sample
/// - ε_id    = identification uncertainty (multiple valid adjustment sets)
/// - ε_struct = structural uncertainty (uncertain graph edges)
#[derive(Clone, Debug)]
pub struct CausalUncertaintyDecomposition {
    /// Statistical uncertainty (standard error from sampling)
    pub statistical: f64,
    /// Identification uncertainty (0.0 = unique adjustment set, higher = ambiguous)
    pub identification: f64,
    /// Structural uncertainty (average edge existence uncertainty)
    pub structural: f64,
    /// Total combined uncertainty (quadrature sum)
    pub total: f64,
    /// Number of valid adjustment sets found
    pub n_adjustment_sets: usize,
}

impl fmt::Display for CausalUncertaintyDecomposition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Causal Uncertainty Decomposition:")?;
        writeln!(f, "  Statistical:     {:.4}", self.statistical)?;
        writeln!(
            f,
            "  Identification:  {:.4} ({} adjustment sets)",
            self.identification, self.n_adjustment_sets
        )?;
        writeln!(f, "  Structural:      {:.4}", self.structural)?;
        write!(f, "  Total (GUM):     {:.4}", self.total)
    }
}

/// Type-level predicate for causal identifiability.
///
/// Used by the dependent type checker to verify at compile time that
/// a causal effect can be estimated from observational data. This bridges
/// the runtime causal module with the type system's refinement predicates.
///
/// # Integration with dependent/predicates.rs
///
/// ```ignore
/// // In type signatures:
/// fn safe_ate<G: CausalGraph>(
///     g: G,
///     treatment: Node,
///     outcome: Node,
/// ) where CausalTypePredicate::Identifiable(g, treatment, outcome) -> f64
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CausalTypePredicate {
    /// The causal effect P(Y|do(X)) is identifiable in graph G
    Identifiable { treatment: String, outcome: String },

    /// Variables X and Y are d-separated given Z in graph G
    DSeparated {
        x: String,
        y: String,
        conditioning: Vec<String>,
    },

    /// Set Z satisfies the backdoor criterion for (X, Y)
    BackdoorValid {
        treatment: String,
        outcome: String,
        adjustment_set: Vec<String>,
    },

    /// Set M satisfies the frontdoor criterion for (X, Y)
    FrontdoorValid {
        treatment: String,
        outcome: String,
        mediators: Vec<String>,
    },

    /// The graph is a valid DAG (no cycles)
    IsDAG,
}

impl CausalTypePredicate {
    /// Evaluate the predicate against a concrete causal graph.
    ///
    /// Returns `true` if the predicate holds, `false` otherwise.
    /// Used by the SMT integration to provide compile-time guarantees.
    pub fn evaluate(&self, graph: &CausalGraph) -> bool {
        match self {
            CausalTypePredicate::Identifiable { treatment, outcome } => {
                let identifier = CausalIdentifier::new(graph);
                identifier.identify(treatment, outcome).is_identified()
            }
            CausalTypePredicate::DSeparated { x, y, conditioning } => {
                let z: HashSet<String> = conditioning.iter().cloned().collect();
                graph.d_separated(x, y, &z)
            }
            CausalTypePredicate::BackdoorValid {
                treatment,
                outcome,
                adjustment_set,
            } => {
                let z: HashSet<String> = adjustment_set.iter().cloned().collect();
                let identifier = CausalIdentifier::new(graph);
                identifier.satisfies_backdoor(treatment, outcome, &z)
            }
            CausalTypePredicate::FrontdoorValid {
                treatment,
                outcome,
                mediators,
            } => {
                let m: HashSet<String> = mediators.iter().cloned().collect();
                let identifier = CausalIdentifier::new(graph);
                identifier.satisfies_frontdoor(treatment, outcome, &m)
            }
            CausalTypePredicate::IsDAG => {
                // A valid topological order exists iff the graph is a DAG
                !graph.topological_order().is_empty() || graph.node_count() == 0
            }
        }
    }

    /// Convert to a human-readable description for error messages
    pub fn describe(&self) -> String {
        match self {
            CausalTypePredicate::Identifiable { treatment, outcome } => {
                format!("P({} | do({})) is identifiable", outcome, treatment)
            }
            CausalTypePredicate::DSeparated { x, y, conditioning } => {
                format!("{} ⊥⊥ {} | {:?}", x, y, conditioning)
            }
            CausalTypePredicate::BackdoorValid {
                treatment,
                outcome,
                adjustment_set,
            } => {
                format!(
                    "{:?} satisfies backdoor criterion for ({}, {})",
                    adjustment_set, treatment, outcome
                )
            }
            CausalTypePredicate::FrontdoorValid {
                treatment,
                outcome,
                mediators,
            } => {
                format!(
                    "{:?} satisfies frontdoor criterion for ({}, {})",
                    mediators, treatment, outcome
                )
            }
            CausalTypePredicate::IsDAG => "graph is a valid DAG".to_string(),
        }
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Quick identification check: is the causal effect of X on Y identifiable?
///
/// This is the simplest entry point for causal analysis. For more detailed
/// results, use [`CausalGraphBuilder`] or [`CausalAnalysis`].
///
/// # Example
///
/// ```ignore
/// let mut graph = CausalGraph::new();
/// // ... build graph ...
/// let (identified, method) = identify_effect(&graph, "Treatment", "Outcome");
/// ```
pub fn identify_effect(
    graph: &CausalGraph,
    treatment: &str,
    outcome: &str,
) -> (bool, IdentificationMethod) {
    let identifier = CausalIdentifier::new(graph);
    let status = identifier.identify(treatment, outcome);
    let identified = status.is_identified();
    let method = status
        .method()
        .cloned()
        .unwrap_or(IdentificationMethod::Unknown);
    (identified, method)
}

/// Quick d-separation test
pub fn is_d_separated(graph: &CausalGraph, x: &str, y: &str, conditioning: &[&str]) -> bool {
    let z: HashSet<String> = conditioning.iter().map(|s| s.to_string()).collect();
    graph.d_separated(x, y, &z)
}

/// Build the canonical "confounded" graph: X ← U → Y with X → Y
///
/// This is the simplest graph where backdoor adjustment fails without
/// observing U. Useful for testing and pedagogy.
pub fn confounded_graph(treatment: &str, outcome: &str, confounder: &str) -> CausalGraph {
    CausalGraphBuilder::new()
        .treatment(treatment)
        .outcome(outcome)
        .latent(confounder)
        .edge(treatment, outcome)
        .edge(confounder, treatment)
        .edge(confounder, outcome)
        .build()
}

/// Build a mediation graph: X → M → Y (with possible direct X → Y)
pub fn mediation_graph(
    treatment: &str,
    mediator: &str,
    outcome: &str,
    direct_effect: bool,
) -> CausalGraph {
    let mut builder = CausalGraphBuilder::new()
        .treatment(treatment)
        .mediator(mediator)
        .outcome(outcome)
        .edge(treatment, mediator)
        .edge(mediator, outcome);

    if direct_effect {
        builder = builder.edge(treatment, outcome);
    }

    builder.build()
}

/// Build an instrumental variable graph: Z → X → Y with U → X, U → Y
pub fn iv_graph(instrument: &str, treatment: &str, outcome: &str, confounder: &str) -> CausalGraph {
    CausalGraphBuilder::new()
        .observed(instrument)
        .treatment(treatment)
        .outcome(outcome)
        .latent(confounder)
        .edge(instrument, treatment)
        .edge(treatment, outcome)
        .edge(confounder, treatment)
        .edge(confounder, outcome)
        .build()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_simple_graph() {
        let analysis = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .edge("X", "Y")
            .analyze();

        assert_eq!(analysis.node_count(), 2);
        assert_eq!(analysis.edge_count(), 1);
        assert_eq!(analysis.treatments, vec!["X"]);
        assert_eq!(analysis.outcomes, vec!["Y"]);
    }

    #[test]
    fn test_builder_with_confounder() {
        let mut analysis = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .observed("Z")
            .edge("X", "Y")
            .edge("Z", "X")
            .edge("Z", "Y")
            .analyze();

        // Should be identifiable via backdoor adjustment on Z
        assert!(analysis.is_identified());
        let adj = analysis.backdoor_set("X", "Y");
        assert!(adj.is_some());
    }

    #[test]
    fn test_identify_effect_convenience() {
        let graph = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .edge("X", "Y")
            .build();

        let (identified, _method) = identify_effect(&graph, "X", "Y");
        assert!(identified);
    }

    #[test]
    fn test_d_separation() {
        let analysis = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .observed("Z")
            .edge("X", "Z")
            .edge("Z", "Y")
            .analyze();

        // X and Y are d-separated given Z (Z is a mediator)
        let z = HashSet::from(["Z".to_string()]);
        assert!(analysis.d_separated("X", "Y", &z));

        // X and Y are NOT d-separated given empty set
        let empty = HashSet::new();
        assert!(!analysis.d_separated("X", "Y", &empty));
    }

    #[test]
    fn test_confounded_graph() {
        let graph = confounded_graph("Drug", "Effect", "Genetics");
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3);
    }

    #[test]
    fn test_mediation_graph() {
        let graph = mediation_graph("X", "M", "Y", true);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 3); // X→M, M→Y, X→Y

        let graph_no_direct = mediation_graph("X", "M", "Y", false);
        assert_eq!(graph_no_direct.edge_count(), 2); // X→M, M→Y only
    }

    #[test]
    fn test_iv_graph() {
        let graph = iv_graph("Z", "X", "Y", "U");
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 4); // Z→X, X→Y, U→X, U→Y
    }

    #[test]
    fn test_type_predicate_identifiable() {
        let graph = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .edge("X", "Y")
            .build();

        let pred = CausalTypePredicate::Identifiable {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        };
        assert!(pred.evaluate(&graph));
    }

    #[test]
    fn test_type_predicate_d_separated() {
        let graph = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .observed("Z")
            .edge("X", "Z")
            .edge("Z", "Y")
            .analyze();

        let pred = CausalTypePredicate::DSeparated {
            x: "X".to_string(),
            y: "Y".to_string(),
            conditioning: vec!["Z".to_string()],
        };
        assert!(pred.evaluate(&graph.graph));
    }

    #[test]
    fn test_type_predicate_describe() {
        let pred = CausalTypePredicate::Identifiable {
            treatment: "Drug".to_string(),
            outcome: "Recovery".to_string(),
        };
        assert_eq!(pred.describe(), "P(Recovery | do(Drug)) is identifiable");

        let pred = CausalTypePredicate::IsDAG;
        assert_eq!(pred.describe(), "graph is a valid DAG");
    }

    #[test]
    fn test_uncertainty_decomposition_display() {
        let decomp = CausalUncertaintyDecomposition {
            statistical: 0.05,
            identification: 0.1,
            structural: 0.02,
            total: 0.115,
            n_adjustment_sets: 3,
        };
        let s = format!("{}", decomp);
        assert!(s.contains("Statistical"));
        assert!(s.contains("Identification"));
        assert!(s.contains("Structural"));
        assert!(s.contains("Total (GUM)"));
    }

    #[test]
    fn test_identification_summary_display() {
        let summary = IdentificationSummary {
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            identified: true,
            method: Some(IdentificationMethod::BackdoorAdjustment {
                set: HashSet::from(["Z".to_string()]),
            }),
            adjustment_set: Some(HashSet::from(["Z".to_string()])),
        };
        let s = format!("{}", summary);
        assert!(s.contains("IDENTIFIED"));
        assert!(s.contains("Backdoor"));
    }

    #[test]
    fn test_identify_all() {
        let mut analysis = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .edge("X", "Y")
            .analyze();

        let results = analysis.identify_all();
        assert_eq!(results.len(), 1);
        assert!(results[0].identified);
    }

    #[test]
    fn test_from_graph() {
        let mut graph = CausalGraph::new();
        graph.add_node(CausalNode::treatment("X"));
        graph.add_node(CausalNode::outcome("Y"));
        let _ = graph.add_edge("X", "Y", EdgeType::Direct);

        let mut analysis = CausalAnalysis::from_graph(graph);
        assert_eq!(analysis.treatments, vec!["X"]);
        assert_eq!(analysis.outcomes, vec!["Y"]);
        assert!(analysis.is_identified());
    }

    #[test]
    fn test_is_d_separated_convenience() {
        let graph = CausalGraphBuilder::new()
            .treatment("X")
            .outcome("Y")
            .observed("Z")
            .edge("X", "Z")
            .edge("Z", "Y")
            .build();

        assert!(is_d_separated(&graph, "X", "Y", &["Z"]));
        assert!(!is_d_separated(&graph, "X", "Y", &[]));
    }
}
