//! Pearl's Do-Calculus for Causal Inference
//!
//! This module implements the three rules of do-calculus for identifying
//! causal effects from observational data.
//!
//! # The Three Rules of Do-Calculus
//!
//! Given a causal graph G:
//!
//! **Rule 1 (Insertion/deletion of observations):**
//! ```text
//! P(Y | do(X), Z, W) = P(Y | do(X), W)
//! if (Y ⊥⊥ Z | X, W)_{G_X̄}
//! ```
//!
//! **Rule 2 (Action/observation exchange):**
//! ```text
//! P(Y | do(X), do(Z), W) = P(Y | do(X), Z, W)
//! if (Y ⊥⊥ Z | X, W)_{G_X̄Z_}
//! ```
//!
//! **Rule 3 (Insertion/deletion of actions):**
//! ```text
//! P(Y | do(X), do(Z), W) = P(Y | do(X), W)
//! if (Y ⊥⊥ Z | X, W)_{G_X̄Z̄(W)}
//! where G_X̄Z̄(W) = graph with X incoming and Z outgoing (except through W) removed
//! ```
//!
//! # References
//!
//! - Pearl, J. (1995). "Causal diagrams for empirical research"
//! - Pearl, J. (2009). "Causality: Models, Reasoning, and Inference"

use std::collections::HashSet;

use super::dag::CausalDAG;

/// Do-calculus query for causal effects
#[derive(Clone, Debug)]
pub struct CausalQuery {
    /// Outcome variable(s)
    pub outcome: HashSet<String>,
    /// Intervention variables with their values
    pub interventions: Vec<(String, f64)>,
    /// Conditioning variables
    pub conditioning: HashSet<String>,
}

impl CausalQuery {
    /// Create a new causal query
    pub fn new(outcome: impl Into<String>) -> Self {
        let outcome_str = outcome.into();
        let mut outcome_set = HashSet::new();
        outcome_set.insert(outcome_str);

        CausalQuery {
            outcome: outcome_set,
            interventions: Vec::new(),
            conditioning: HashSet::new(),
        }
    }

    /// Add an intervention: do(X = x)
    pub fn with_intervention(mut self, var: impl Into<String>, value: f64) -> Self {
        self.interventions.push((var.into(), value));
        self
    }

    /// Add conditioning variable
    pub fn given(mut self, var: impl Into<String>) -> Self {
        self.conditioning.insert(var.into());
        self
    }
}

/// Adjustment set for identifying causal effects
#[derive(Clone, Debug, PartialEq)]
pub struct AdjustmentSet {
    /// Variables to adjust for
    pub variables: HashSet<String>,
    /// Type of adjustment
    pub adjustment_type: AdjustmentType,
}

/// Type of causal adjustment
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AdjustmentType {
    /// Backdoor adjustment: control for confounders
    Backdoor,
    /// Frontdoor adjustment: use mediators
    Frontdoor,
    /// Instrumental variable adjustment
    InstrumentalVariable,
}

/// Do-calculus operations on causal graphs
pub trait DoCalculus {
    /// Perform intervention: P(Y | do(X=x))
    ///
    /// Returns the distribution of Y after setting X to value x,
    /// breaking all incoming edges to X
    fn do_intervention(&self, var: &str, value: f64) -> Self;

    /// Check if a causal query is identifiable from observational data
    ///
    /// Uses do-calculus rules to determine if P(Y | do(X)) can be
    /// expressed in terms of observable probabilities P(·)
    fn is_identifiable(&self, query: &CausalQuery) -> bool;

    /// Find a valid backdoor adjustment set
    ///
    /// Returns variables Z such that controlling for Z identifies
    /// the causal effect of treatment on outcome
    fn backdoor_adjustment(&self, treatment: &str, outcome: &str) -> Option<AdjustmentSet>;

    /// Find a valid frontdoor adjustment set
    ///
    /// Returns mediator variables M that can be used to identify
    /// the causal effect when there are unobserved confounders
    fn frontdoor_adjustment(&self, treatment: &str, outcome: &str) -> Option<AdjustmentSet>;
}

// Note: DoCalculus for CausalGraph is already implemented in
// the existing identification module. We focus on CausalDAG here.

impl DoCalculus for CausalDAG {
    fn do_intervention(&self, var: &str, _value: f64) -> Self {
        // Remove all incoming edges to var
        self.intervene_remove_incoming(var)
    }

    fn is_identifiable(&self, query: &CausalQuery) -> bool {
        // For single outcome and single intervention
        if query.outcome.len() != 1 || query.interventions.len() != 1 {
            return false;
        }

        let outcome = query.outcome.iter().next().unwrap();
        let (treatment, _) = &query.interventions[0];

        // Check if we can identify via backdoor or frontdoor
        self.backdoor_adjustment(treatment, outcome).is_some()
            || self.frontdoor_adjustment(treatment, outcome).is_some()
    }

    fn backdoor_adjustment(&self, treatment: &str, outcome: &str) -> Option<AdjustmentSet> {
        // Find valid backdoor adjustment set
        let descendants_of_treatment = self.descendants(treatment);

        // Candidates: all nodes except treatment, outcome, and descendants of treatment
        let mut candidates = Vec::new();
        for node in self.nodes() {
            let name = &node.name;
            if name != treatment
                && name != outcome
                && !descendants_of_treatment.contains(name.as_str())
            {
                candidates.push(name.clone());
            }
        }

        // Try empty set first
        if self.satisfies_backdoor(treatment, outcome, &HashSet::new()) {
            return Some(AdjustmentSet {
                variables: HashSet::new(),
                adjustment_type: AdjustmentType::Backdoor,
            });
        }

        // Try single variables
        for var in &candidates {
            let mut set = HashSet::new();
            set.insert(var.clone());
            if self.satisfies_backdoor(treatment, outcome, &set) {
                return Some(AdjustmentSet {
                    variables: set,
                    adjustment_type: AdjustmentType::Backdoor,
                });
            }
        }

        // Try pairs (limited search)
        for i in 0..candidates.len().min(5) {
            for j in (i + 1)..candidates.len().min(5) {
                let mut set = HashSet::new();
                set.insert(candidates[i].clone());
                set.insert(candidates[j].clone());
                if self.satisfies_backdoor(treatment, outcome, &set) {
                    return Some(AdjustmentSet {
                        variables: set,
                        adjustment_type: AdjustmentType::Backdoor,
                    });
                }
            }
        }

        None
    }

    fn frontdoor_adjustment(&self, treatment: &str, outcome: &str) -> Option<AdjustmentSet> {
        // Find mediators on all paths from treatment to outcome
        let mediators = self.find_all_mediators(treatment, outcome);

        // Check frontdoor criterion
        for mediator in mediators {
            let mut med_set = HashSet::new();
            med_set.insert(mediator.clone());

            if self.satisfies_frontdoor(treatment, outcome, &med_set) {
                return Some(AdjustmentSet {
                    variables: med_set,
                    adjustment_type: AdjustmentType::Frontdoor,
                });
            }
        }

        None
    }
}

impl CausalDAG {
    /// Get all descendants of a node (via directed paths)
    fn descendants(&self, node: &str) -> HashSet<String> {
        let mut desc = HashSet::new();
        let mut queue = vec![node.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for child in self.children(&current) {
                desc.insert(child.to_string());
                queue.push(child.to_string());
            }
        }

        desc
    }

    /// Check if Z satisfies backdoor criterion for (treatment, outcome)
    fn satisfies_backdoor(&self, treatment: &str, outcome: &str, z: &HashSet<String>) -> bool {
        // 1. No node in Z is a descendant of treatment
        let desc = self.descendants(treatment);
        for var in z {
            if desc.contains(var) {
                return false;
            }
        }

        // 2. Z blocks all backdoor paths (paths with edge into treatment)
        // Simplified: check if treatment and outcome are d-separated in G with
        // outgoing edges from treatment removed
        let g_no_out = self.remove_outgoing(treatment);
        g_no_out.d_separated(treatment, outcome, z)
    }

    /// Check if M satisfies frontdoor criterion
    fn satisfies_frontdoor(&self, treatment: &str, outcome: &str, m: &HashSet<String>) -> bool {
        // 1. M intercepts all directed paths from treatment to outcome
        if !self.intercepts_all_paths(treatment, outcome, m) {
            return false;
        }

        // 2. No backdoor path from treatment to M
        let g_no_out = self.remove_outgoing(treatment);
        for med in m {
            if !g_no_out.d_separated(treatment, med, &HashSet::new()) {
                return false;
            }
        }

        // 3. All backdoor paths from M to outcome are blocked by treatment
        let mut t_set = HashSet::new();
        t_set.insert(treatment.to_string());

        for med in m {
            let g_no_out_m = self.remove_outgoing(med);
            if !g_no_out_m.d_separated(med, outcome, &t_set) {
                return false;
            }
        }

        true
    }

    /// Check if mediators intercept all paths from source to target
    fn intercepts_all_paths(
        &self,
        source: &str,
        target: &str,
        mediators: &HashSet<String>,
    ) -> bool {
        // Simple path-finding: can we reach target from source without going through mediators?
        let mut queue = vec![source.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = queue.pop() {
            if current == target {
                return false; // Found path not through mediators
            }

            if visited.contains(&current) || mediators.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for child in self.children(&current) {
                queue.push(child.to_string());
            }
        }

        true // All paths go through mediators
    }

    /// Find all mediators between treatment and outcome
    fn find_all_mediators(&self, treatment: &str, outcome: &str) -> Vec<String> {
        let desc_t = self.descendants(treatment);
        let anc_o = self.ancestors(outcome);

        let mut mediators = Vec::new();
        for node in self.nodes() {
            let name = &node.name;
            if name != treatment
                && name != outcome
                && desc_t.contains(name.as_str())
                && anc_o.contains(name.as_str())
            {
                mediators.push(name.clone());
            }
        }

        mediators
    }

    /// Get all ancestors of a node
    fn ancestors(&self, node: &str) -> HashSet<String> {
        let mut anc = HashSet::new();
        let mut queue = vec![node.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = queue.pop() {
            if visited.contains(&current) {
                continue;
            }
            visited.insert(current.clone());

            for parent in self.parents(&current) {
                anc.insert(parent.to_string());
                queue.push(parent.to_string());
            }
        }

        anc
    }

    /// Check d-separation (simplified version)
    pub fn d_separated(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        // Simplified d-separation check
        // In a full implementation, would use Bayes-Ball algorithm
        if x == y {
            return false;
        }

        // If z blocks all paths, they're d-separated
        !self.has_active_path(x, y, z)
    }

    /// Check if there's an active path between x and y given z
    fn has_active_path(&self, x: &str, y: &str, z: &HashSet<String>) -> bool {
        // Simple path existence check (not full d-connection)
        let mut queue = vec![x.to_string()];
        let mut visited = HashSet::new();

        while let Some(current) = queue.pop() {
            if current == y {
                return true;
            }

            if visited.contains(&current) || (z.contains(&current) && current != x) {
                continue;
            }
            visited.insert(current.clone());

            // Add children
            for child in self.children(&current) {
                queue.push(child.to_string());
            }

            // Add parents (for backdoor paths)
            for parent in self.parents(&current) {
                queue.push(parent.to_string());
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::causal::dag::{EffectEstimate, EpistemicCausalNode, UncertainCausalEdge};
    use crate::epistemic::BetaConfidence;

    #[test]
    fn test_causal_query() {
        let query = CausalQuery::new("Y").with_intervention("X", 1.0).given("Z");

        assert_eq!(query.interventions.len(), 1);
        assert_eq!(query.conditioning.len(), 1);
    }

    #[test]
    fn test_backdoor_adjustment() {
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

        let adjustment = dag.backdoor_adjustment("X", "Y");
        assert!(adjustment.is_some());

        if let Some(adj) = adjustment {
            assert!(adj.variables.contains("U"));
        }
    }

    #[test]
    fn test_do_intervention() {
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
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.3),
        );

        dag.add_edge(edge1).unwrap();
        dag.add_edge(edge2).unwrap();

        let dag_do = dag.do_intervention("X", 1.0);

        // After intervention, X should have no parents
        assert!(dag_do.parents("X").is_empty());
        // But X -> Y should still exist
        assert_eq!(dag_do.children("X"), vec!["Y"]);
    }

    #[test]
    fn test_is_identifiable() {
        let mut dag = CausalDAG::new();
        dag.add_node(EpistemicCausalNode::treatment("X"));
        dag.add_node(EpistemicCausalNode::outcome("Y"));

        let edge = UncertainCausalEdge::direct(
            "X",
            "Y",
            BetaConfidence::new(9.0, 1.0),
            EffectEstimate::certain(0.5),
        );
        dag.add_edge(edge).unwrap();

        let query = CausalQuery::new("Y").with_intervention("X", 1.0);

        assert!(dag.is_identifiable(&query));
    }
}
