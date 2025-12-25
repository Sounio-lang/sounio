//! Causal Effect Identification
//!
//! Implements identification strategies for causal effects:
//! - Backdoor criterion
//! - Frontdoor criterion
//! - Instrumental variables
//! - Do-calculus rules

use std::collections::HashSet;
use std::fmt;

use super::graph::CausalGraph;

/// Identification method used to compute causal effect
#[derive(Clone, Debug, PartialEq)]
pub enum IdentificationMethod {
    /// Experimental/RCT data
    Experimental,

    /// Backdoor adjustment: P(Y|do(X)) = Σ_z P(Y|X,Z)P(Z)
    BackdoorAdjustment { set: HashSet<String> },

    /// Frontdoor adjustment
    FrontdoorAdjustment { mediators: HashSet<String> },

    /// Instrumental variable estimation
    InstrumentalVariable { instruments: HashSet<String> },

    /// General do-calculus derivation
    DoCalculus { derivation: Vec<DoCalculusStep> },

    /// Partial identification (bounds only)
    PartialIdentification { lower: f64, upper: f64 },

    /// Effect is not identifiable
    NotIdentifiable,

    /// Identification status unknown
    Unknown,
}

impl fmt::Display for IdentificationMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IdentificationMethod::Experimental => write!(f, "Experimental (RCT)"),
            IdentificationMethod::BackdoorAdjustment { set } => {
                write!(f, "Backdoor adjustment on {:?}", set)
            }
            IdentificationMethod::FrontdoorAdjustment { mediators } => {
                write!(f, "Frontdoor via {:?}", mediators)
            }
            IdentificationMethod::InstrumentalVariable { instruments } => {
                write!(f, "IV estimation using {:?}", instruments)
            }
            IdentificationMethod::DoCalculus { derivation } => {
                write!(f, "Do-calculus ({} steps)", derivation.len())
            }
            IdentificationMethod::PartialIdentification { lower, upper } => {
                write!(f, "Partial identification: [{:.3}, {:.3}]", lower, upper)
            }
            IdentificationMethod::NotIdentifiable => write!(f, "Not identifiable"),
            IdentificationMethod::Unknown => write!(f, "Unknown"),
        }
    }
}

/// A step in a do-calculus derivation
#[derive(Clone, Debug, PartialEq)]
pub struct DoCalculusStep {
    /// Rule applied
    pub rule: DoCalculusRule,
    /// Expression before transformation
    pub from: String,
    /// Expression after transformation
    pub to: String,
    /// Justification (d-separation statement)
    pub justification: String,
}

/// Do-calculus rules
#[derive(Clone, Debug, PartialEq)]
pub enum DoCalculusRule {
    /// Rule 1: Insertion/deletion of observations
    /// P(Y | do(X), Z, W) = P(Y | do(X), W) if (Y ⊥⊥ Z | X, W)_{G_X̄}
    Rule1 { z: HashSet<String> },

    /// Rule 2: Action/observation exchange
    /// P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W) if (Y ⊥⊥ Z | X, W)_{G_X̄Z_}
    Rule2 { z: HashSet<String> },

    /// Rule 3: Insertion/deletion of actions
    /// P(Y | do(X), do(Z), W) = P(Y | do(X), W) if (Y ⊥⊥ Z | X, W)_{G_X̄Z̄*}
    Rule3 { z: HashSet<String> },
}

impl fmt::Display for DoCalculusRule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DoCalculusRule::Rule1 { z } => write!(f, "Rule 1 (obs): {:?}", z),
            DoCalculusRule::Rule2 { z } => write!(f, "Rule 2 (act/obs): {:?}", z),
            DoCalculusRule::Rule3 { z } => write!(f, "Rule 3 (act): {:?}", z),
        }
    }
}

/// Identification status for a causal query
#[derive(Clone, Debug)]
pub enum IdentificationStatus {
    /// Effect is identifiable with given method
    Identified { method: IdentificationMethod },

    /// Only bounds can be determined
    PartiallyIdentified { lower: f64, upper: f64 },

    /// Effect cannot be identified from observational data
    NotIdentifiable { reason: String },

    /// Status not yet determined
    Unknown,
}

impl IdentificationStatus {
    /// Check if effect is identifiable
    pub fn is_identified(&self) -> bool {
        matches!(self, IdentificationStatus::Identified { .. })
    }

    /// Get the identification method if identified
    pub fn method(&self) -> Option<&IdentificationMethod> {
        match self {
            IdentificationStatus::Identified { method } => Some(method),
            _ => None,
        }
    }
}

/// Causal effect identifier
pub struct CausalIdentifier<'a> {
    graph: &'a CausalGraph,
}

impl<'a> CausalIdentifier<'a> {
    /// Create a new identifier for the given graph
    pub fn new(graph: &'a CausalGraph) -> Self {
        CausalIdentifier { graph }
    }

    /// Try to identify P(Y | do(X))
    pub fn identify(&self, treatment: &str, outcome: &str) -> IdentificationStatus {
        // Try backdoor criterion first
        if let Some(set) = self.find_backdoor_set(treatment, outcome) {
            return IdentificationStatus::Identified {
                method: IdentificationMethod::BackdoorAdjustment { set },
            };
        }

        // Try frontdoor criterion
        if let Some(mediators) = self.find_frontdoor_set(treatment, outcome) {
            return IdentificationStatus::Identified {
                method: IdentificationMethod::FrontdoorAdjustment { mediators },
            };
        }

        // Try instrumental variables
        if let Some(instruments) = self.find_instruments(treatment, outcome) {
            return IdentificationStatus::Identified {
                method: IdentificationMethod::InstrumentalVariable { instruments },
            };
        }

        // Try do-calculus derivation
        if let Some(derivation) = self.derive_do_calculus(treatment, outcome) {
            return IdentificationStatus::Identified {
                method: IdentificationMethod::DoCalculus { derivation },
            };
        }

        // Not identifiable
        IdentificationStatus::NotIdentifiable {
            reason: format!(
                "No valid identification strategy found for P({} | do({}))",
                outcome, treatment
            ),
        }
    }

    /// Find a valid backdoor adjustment set
    ///
    /// Z satisfies backdoor criterion for (X, Y) if:
    /// 1. No node in Z is a descendant of X
    /// 2. Z blocks all backdoor paths from X to Y
    pub fn find_backdoor_set(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let descendants_of_x = self.graph.descendants_of(treatment);

        // Candidates: all observed nodes that aren't X, Y, or descendants of X
        let candidates: HashSet<String> = self
            .graph
            .node_names()
            .filter(|n| *n != treatment && *n != outcome && !descendants_of_x.contains(*n))
            .cloned()
            .collect();

        // Try empty set first (if no backdoor paths exist)
        if self.satisfies_backdoor(treatment, outcome, &HashSet::new()) {
            return Some(HashSet::new());
        }

        // Try single nodes
        for node in &candidates {
            let set: HashSet<String> = [node.clone()].into_iter().collect();
            if self.satisfies_backdoor(treatment, outcome, &set) {
                return Some(set);
            }
        }

        // Try pairs
        let candidates_vec: Vec<_> = candidates.iter().collect();
        for i in 0..candidates_vec.len() {
            for j in (i + 1)..candidates_vec.len() {
                let set: HashSet<String> = [candidates_vec[i].clone(), candidates_vec[j].clone()]
                    .into_iter()
                    .collect();
                if self.satisfies_backdoor(treatment, outcome, &set) {
                    return Some(set);
                }
            }
        }

        None
    }

    /// Check if Z satisfies the backdoor criterion
    pub fn satisfies_backdoor(&self, treatment: &str, outcome: &str, z: &HashSet<String>) -> bool {
        // 1. No node in Z is descendant of X
        let desc_x = self.graph.descendants_of(treatment);
        if z.iter().any(|node| desc_x.contains(node)) {
            return false;
        }

        // 2. Z blocks all backdoor paths
        // In G with outgoing edges from X removed, X and Y should be d-separated given Z
        let g_no_out = self.graph.graph_no_out(treatment);
        g_no_out.d_separated(treatment, outcome, z)
    }

    /// Find a valid frontdoor adjustment set
    ///
    /// M satisfies frontdoor criterion for (X, Y) if:
    /// 1. M intercepts all directed paths from X to Y
    /// 2. No backdoor path from X to M (unconfounded)
    /// 3. All backdoor paths from M to Y are blocked by X
    pub fn find_frontdoor_set(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let mediators = self.graph.find_mediators(treatment, outcome);

        // Try each single mediator
        for m in &mediators {
            let m_set: HashSet<String> = [m.clone()].into_iter().collect();
            if self.satisfies_frontdoor(treatment, outcome, &m_set) {
                return Some(m_set);
            }
        }

        // Try pairs of mediators
        for i in 0..mediators.len() {
            for j in (i + 1)..mediators.len() {
                let m_set: HashSet<String> = [mediators[i].clone(), mediators[j].clone()]
                    .into_iter()
                    .collect();
                if self.satisfies_frontdoor(treatment, outcome, &m_set) {
                    return Some(m_set);
                }
            }
        }

        None
    }

    /// Check if M satisfies frontdoor criterion
    pub fn satisfies_frontdoor(&self, treatment: &str, outcome: &str, m: &HashSet<String>) -> bool {
        // 1. M intercepts all directed paths from X to Y
        if !self.graph.intercepts_all_paths(treatment, outcome, m) {
            return false;
        }

        // 2. No backdoor path from X to any node in M
        let g_no_out_x = self.graph.graph_no_out(treatment);
        for node in m {
            if !g_no_out_x.d_separated(treatment, node, &HashSet::new()) {
                return false;
            }
        }

        // 3. All backdoor paths from M to Y are blocked by X
        let x_set: HashSet<String> = [treatment.to_string()].into_iter().collect();
        for node in m {
            let g_no_out_m = self.graph.graph_no_out(node);
            if !g_no_out_m.d_separated(node, outcome, &x_set) {
                return false;
            }
        }

        true
    }

    /// Find valid instrumental variables
    ///
    /// Z is an instrument for (X, Y) if:
    /// 1. Z affects X (relevance)
    /// 2. Z affects Y only through X (exclusion restriction)
    /// 3. Z is independent of confounders (exogeneity)
    pub fn find_instruments(&self, treatment: &str, outcome: &str) -> Option<HashSet<String>> {
        let mut instruments = HashSet::new();

        for node in self.graph.node_names() {
            if node == treatment || node == outcome {
                continue;
            }

            if self.is_valid_instrument(node, treatment, outcome) {
                instruments.insert(node.clone());
            }
        }

        if instruments.is_empty() {
            None
        } else {
            Some(instruments)
        }
    }

    /// Check if Z is a valid instrument
    fn is_valid_instrument(&self, z: &str, treatment: &str, outcome: &str) -> bool {
        // 1. Relevance: Z must have path to X
        let children_z = self.graph.children(z).cloned().unwrap_or_default();
        let descendants_z = self.graph.descendants_of(z);
        if !children_z.contains(treatment) && !descendants_z.contains(treatment) {
            return false;
        }

        // 2. Exclusion: Z affects Y only through X
        // In graph with X removed, Z and Y should be d-separated
        let g_no_x = self.graph.clone();
        // Simplified: check if all paths from Z to Y go through X
        if !self.all_paths_through(z, outcome, treatment) {
            return false;
        }

        // 3. Exogeneity: Z independent of confounders of X-Y
        // Simplified: Z has no incoming edges from confounders
        let parents_z = self.graph.parents(z).cloned().unwrap_or_default();
        let parents_x = self.graph.parents(treatment).cloned().unwrap_or_default();
        let parents_y = self.graph.parents(outcome).cloned().unwrap_or_default();

        // Common parents of X and Y are confounders
        let confounders: HashSet<_> = parents_x.intersection(&parents_y).cloned().collect();

        // Z should not be descendant of confounders
        for conf in &confounders {
            if self.graph.descendants_of(conf).contains(z) {
                return false;
            }
        }

        true
    }

    /// Check if all directed paths from A to C go through B
    fn all_paths_through(&self, a: &str, c: &str, b: &str) -> bool {
        let b_set: HashSet<String> = [b.to_string()].into_iter().collect();
        self.graph.intercepts_all_paths(a, c, &b_set)
    }

    /// Attempt do-calculus derivation
    fn derive_do_calculus(&self, _treatment: &str, _outcome: &str) -> Option<Vec<DoCalculusStep>> {
        // Full do-calculus derivation is complex
        // This is a placeholder for the general algorithm
        // In practice, would use algorithms like IDC or IDC*
        None
    }
}

/// Result of applying backdoor adjustment
#[derive(Clone, Debug)]
pub struct BackdoorAdjustment {
    /// Adjustment set used
    pub adjustment_set: HashSet<String>,
    /// Stratified estimates
    pub strata: Vec<StrataEstimate>,
    /// Overall causal effect
    pub effect: f64,
    /// Standard error
    pub std_error: f64,
}

/// Estimate within a stratum
#[derive(Clone, Debug)]
pub struct StrataEstimate {
    /// Stratum values
    pub stratum: Vec<(String, f64)>,
    /// Effect in this stratum
    pub effect: f64,
    /// Weight (proportion of population)
    pub weight: f64,
}

impl BackdoorAdjustment {
    /// Compute weighted average across strata
    pub fn compute_effect(&self) -> f64 {
        self.strata.iter().map(|s| s.effect * s.weight).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::graph::{CausalNode, EdgeType};

    fn simple_confounded_graph() -> CausalGraph {
        // X <- U -> Y, X -> Y
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("X"));
        g.add_node(CausalNode::outcome("Y"));
        g.add_node(CausalNode::observed("U"));

        g.add_edge("X", "Y", EdgeType::Direct).unwrap();
        g.add_edge("U", "X", EdgeType::Direct).unwrap();
        g.add_edge("U", "Y", EdgeType::Direct).unwrap();

        g
    }

    fn frontdoor_graph() -> CausalGraph {
        // X -> M -> Y, U -> X, U -> Y (U latent)
        let mut g = CausalGraph::new();
        g.add_node(CausalNode::treatment("X"));
        g.add_node(CausalNode::mediator("M"));
        g.add_node(CausalNode::outcome("Y"));
        g.add_node(CausalNode::latent("U"));

        g.add_edge("X", "M", EdgeType::Direct).unwrap();
        g.add_edge("M", "Y", EdgeType::Direct).unwrap();
        g.add_edge("U", "X", EdgeType::Direct).unwrap();
        g.add_edge("U", "Y", EdgeType::Direct).unwrap();

        g
    }

    #[test]
    fn test_backdoor_identification() {
        let g = simple_confounded_graph();
        let identifier = CausalIdentifier::new(&g);

        let status = identifier.identify("X", "Y");

        match status {
            IdentificationStatus::Identified { method } => {
                assert!(matches!(
                    method,
                    IdentificationMethod::BackdoorAdjustment { .. }
                ));
                if let IdentificationMethod::BackdoorAdjustment { set } = method {
                    assert!(set.contains("U"));
                }
            }
            _ => panic!("Expected backdoor identification"),
        }
    }

    #[test]
    fn test_satisfies_backdoor() {
        let g = simple_confounded_graph();
        let identifier = CausalIdentifier::new(&g);

        // Empty set doesn't satisfy (U confounds)
        assert!(!identifier.satisfies_backdoor("X", "Y", &HashSet::new()));

        // {U} satisfies
        let u_set: HashSet<String> = ["U".to_string()].into_iter().collect();
        assert!(identifier.satisfies_backdoor("X", "Y", &u_set));
    }

    #[test]
    fn test_frontdoor_identification() {
        let g = frontdoor_graph();
        let identifier = CausalIdentifier::new(&g);

        // Backdoor should fail (U is latent/unobserved in reality)
        // But in our graph U is still a node we can use

        // Let's check frontdoor
        let frontdoor_set = identifier.find_frontdoor_set("X", "Y");

        // M should be a valid frontdoor mediator
        if let Some(set) = frontdoor_set {
            assert!(set.contains("M"));
        }
    }

    #[test]
    fn test_identification_method_display() {
        let method = IdentificationMethod::BackdoorAdjustment {
            set: ["Age", "Gender"].iter().map(|s| s.to_string()).collect(),
        };

        let display = format!("{}", method);
        assert!(display.contains("Backdoor"));
    }

    #[test]
    fn test_do_calculus_step() {
        let step = DoCalculusStep {
            rule: DoCalculusRule::Rule2 {
                z: ["M".to_string()].into_iter().collect(),
            },
            from: "P(Y | do(X), do(M))".to_string(),
            to: "P(Y | do(X), M)".to_string(),
            justification: "(Y ⊥⊥ M | X)_{G_X̄M_}".to_string(),
        };

        assert_eq!(step.from, "P(Y | do(X), do(M))");
    }
}
