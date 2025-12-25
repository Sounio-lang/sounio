//! Proof State
//!
//! The central data structure for geometric reasoning.
//! Maintains the current state of knowledge with epistemic tracking.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::epistemic::bayesian::BetaConfidence;

use super::predicates::{Predicate, PredicateEpistemic, PredicateId, PredicateKind};
use super::primitives::{Circle, Line, Point, PointConstruction};

/// A node in the provenance tree
#[derive(Debug, Clone)]
pub struct ProvenanceNode {
    /// Predicate this node represents
    pub predicate_id: PredicateId,
    /// Rule used to derive this (None for axioms)
    pub rule: Option<String>,
    /// Parent nodes (premises)
    pub parents: Vec<PredicateId>,
    /// Depth in proof tree
    pub depth: usize,
    /// Timestamp (for ordering)
    pub timestamp: u64,
}

/// A step in the proof trace
#[derive(Debug, Clone)]
pub struct ProofStep {
    /// Predicate derived
    pub predicate: Predicate,
    /// Rule applied
    pub rule: String,
    /// Premises used
    pub premises: Vec<PredicateId>,
    /// Confidence at this step
    pub confidence: f64,
}

impl fmt::Display for ProofStep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} [{}] (conf: {:.3})",
            self.predicate, self.rule, self.confidence
        )
    }
}

/// Goal to prove
#[derive(Debug, Clone)]
pub struct ProofGoal {
    /// The predicate to prove
    pub predicate: Predicate,
    /// Minimum confidence required
    pub min_confidence: f64,
}

/// Current state of geometric knowledge during proof search
#[derive(Debug, Clone)]
pub struct ProofState {
    /// All known points (label -> Point)
    pub points: HashMap<String, Point>,

    /// All known lines
    pub lines: HashSet<Line>,

    /// All known circles
    pub circles: Vec<Circle>,

    /// All established predicates (key -> Predicate)
    predicates: HashMap<String, Predicate>,

    /// Predicate ID to key mapping
    id_to_key: HashMap<PredicateId, String>,

    /// Provenance graph
    provenance: HashMap<PredicateId, ProvenanceNode>,

    /// Proof trace (ordered steps)
    pub trace: Vec<ProofStep>,

    /// Goal to prove
    pub goal: Option<ProofGoal>,

    /// Overall epistemic confidence (Beta distribution)
    pub confidence: BetaConfidence,

    /// Counter for timestamps
    timestamp_counter: u64,

    /// Constructions applied
    pub constructions: Vec<Construction>,
}

/// An auxiliary construction
#[derive(Debug, Clone)]
pub struct Construction {
    /// Type of construction
    pub kind: ConstructionKind,
    /// Points created
    pub new_points: Vec<String>,
    /// Confidence from source
    pub confidence: f64,
    /// Source (neural or symbolic)
    pub source: ConstructionSource,
}

#[derive(Debug, Clone)]
pub enum ConstructionKind {
    Midpoint {
        p1: String,
        p2: String,
    },
    Perpendicular {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    Parallel {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    Circumcircle {
        p1: String,
        p2: String,
        p3: String,
    },
    Incircle {
        p1: String,
        p2: String,
        p3: String,
    },
    AngleBisector {
        p1: String,
        vertex: String,
        p2: String,
    },
    LineIntersection {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    CircleLineIntersection {
        center: String,
        on_circle: String,
        line_p1: String,
        line_p2: String,
    },
    CircleCircleIntersection {
        c1_center: String,
        c1_on: String,
        c2_center: String,
        c2_on: String,
    },
    Foot {
        point: String,
        line_p1: String,
        line_p2: String,
    },
    Reflection {
        point: String,
        line_p1: String,
        line_p2: String,
    },
}

#[derive(Debug, Clone)]
pub enum ConstructionSource {
    Symbolic,
    Neural { model: String, confidence: f64 },
}

impl ProofState {
    /// Create empty state
    pub fn new() -> Self {
        ProofState {
            points: HashMap::new(),
            lines: HashSet::new(),
            circles: Vec::new(),
            predicates: HashMap::new(),
            id_to_key: HashMap::new(),
            provenance: HashMap::new(),
            trace: Vec::new(),
            goal: None,
            confidence: BetaConfidence::uniform_prior(),
            timestamp_counter: 0,
            constructions: Vec::new(),
        }
    }

    /// Add a free point (from problem statement)
    pub fn add_point(&mut self, label: impl Into<String>) -> &mut Self {
        let label = label.into();
        self.points.insert(label.clone(), Point::free(label));
        self
    }

    /// Add a constructed point
    pub fn add_constructed_point(
        &mut self,
        label: impl Into<String>,
        construction: PointConstruction,
    ) -> &mut Self {
        let label = label.into();
        self.points
            .insert(label.clone(), Point::constructed(label, construction));
        self
    }

    /// Add multiple points
    pub fn add_points(&mut self, labels: &[&str]) -> &mut Self {
        for label in labels {
            self.add_point(*label);
        }
        self
    }

    /// Add a line
    pub fn add_line(&mut self, p1: &str, p2: &str) -> &mut Self {
        self.lines.insert(Line::new(p1, p2));
        self
    }

    /// Add a circle
    pub fn add_circle(&mut self, center: &str, on_circle: &str) -> &mut Self {
        self.circles.push(Circle::new(center, on_circle));
        self
    }

    /// Add a predicate as axiom (from problem statement)
    pub fn add_axiom(&mut self, predicate: Predicate) -> &mut Self {
        let pred = predicate.with_epistemic(PredicateEpistemic::axiom());
        self.add_predicate_internal(pred, None);
        self
    }

    /// Add a derived predicate
    pub fn add_derived(
        &mut self,
        predicate: Predicate,
        rule: &str,
        parents: &[PredicateId],
        decay: f64,
    ) -> bool {
        let key = predicate.key();

        // Check if already exists with higher confidence
        if let Some(existing) = self.predicates.get(&key)
            && existing.epistemic.confidence.mean() >= predicate.epistemic.confidence.mean()
        {
            return false; // Already have better
        }

        // Get parent predicates
        let parent_preds: Vec<&Predicate> = parents
            .iter()
            .filter_map(|id| self.id_to_key.get(id))
            .filter_map(|key| self.predicates.get(key))
            .collect();

        // Create epistemic status
        let epistemic = PredicateEpistemic::derived(&parent_preds, rule, decay);
        let pred = predicate.with_epistemic(epistemic);

        // Update overall confidence with Bayesian update
        let conf = pred.epistemic.confidence.mean();
        self.confidence.update(conf, 1.0 - conf);

        self.add_predicate_internal(pred, Some((rule.to_string(), parents.to_vec())));
        true
    }

    /// Internal: add predicate with provenance
    fn add_predicate_internal(
        &mut self,
        pred: Predicate,
        derivation: Option<(String, Vec<PredicateId>)>,
    ) {
        let key = pred.key();
        let id = pred.id;
        let conf = pred.epistemic.confidence.mean();

        // Create provenance node
        let (rule, parents) = derivation.unwrap_or((String::new(), vec![]));
        let depth = if parents.is_empty() {
            0
        } else {
            parents
                .iter()
                .filter_map(|pid| self.provenance.get(pid))
                .map(|n| n.depth)
                .max()
                .unwrap_or(0)
                + 1
        };

        self.timestamp_counter += 1;
        let prov_node = ProvenanceNode {
            predicate_id: id,
            rule: if rule.is_empty() {
                None
            } else {
                Some(rule.clone())
            },
            parents: parents.clone(),
            depth,
            timestamp: self.timestamp_counter,
        };

        // Record proof step if derived
        if !parents.is_empty() {
            self.trace.push(ProofStep {
                predicate: pred.clone(),
                rule: rule.clone(),
                premises: parents,
                confidence: conf,
            });
        }

        self.predicates.insert(key.clone(), pred);
        self.id_to_key.insert(id, key);
        self.provenance.insert(id, prov_node);
    }

    /// Set the goal to prove
    pub fn set_goal(&mut self, predicate: Predicate, min_confidence: f64) -> &mut Self {
        self.goal = Some(ProofGoal {
            predicate,
            min_confidence,
        });
        self
    }

    /// Check if goal is satisfied
    pub fn goal_satisfied(&self) -> bool {
        match &self.goal {
            Some(goal) => {
                let key = goal.predicate.key();
                if let Some(pred) = self.predicates.get(&key) {
                    pred.epistemic.confidence.mean() >= goal.min_confidence
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Check if goal is satisfied with uncertainty threshold
    ///
    /// Returns true only if confidence is high enough AND variance is low enough
    pub fn goal_satisfied_with_certainty(&self, max_variance: f64) -> bool {
        match &self.goal {
            Some(goal) => {
                let key = goal.predicate.key();
                if let Some(pred) = self.predicates.get(&key) {
                    pred.epistemic.confidence.mean() >= goal.min_confidence
                        && pred.epistemic.confidence.variance() <= max_variance
                } else {
                    false
                }
            }
            None => false,
        }
    }

    /// Get global epistemic uncertainty (aggregate variance across all predicates)
    pub fn global_uncertainty(&self) -> f64 {
        if self.predicates.is_empty() {
            return 1.0; // Maximum uncertainty when we know nothing
        }

        // Compute mean variance across all predicates
        let total_variance: f64 = self
            .predicates
            .values()
            .map(|p| p.epistemic.confidence.variance())
            .sum();

        total_variance / self.predicates.len() as f64
    }

    /// Get predicates with high uncertainty (candidates for neural re-evaluation)
    pub fn uncertain_predicates(&self, threshold: f64) -> Vec<&Predicate> {
        self.predicates
            .values()
            .filter(|p| p.epistemic.confidence.variance() > threshold)
            .collect()
    }

    /// Get a predicate by key
    pub fn get_predicate(&self, key: &str) -> Option<&Predicate> {
        self.predicates.get(key)
    }

    /// Get a predicate by ID
    pub fn get_predicate_by_id(&self, id: PredicateId) -> Option<&Predicate> {
        self.id_to_key
            .get(&id)
            .and_then(|key| self.predicates.get(key))
    }

    /// Get all predicates
    pub fn all_predicates(&self) -> impl Iterator<Item = &Predicate> {
        self.predicates.values()
    }

    /// Get predicates by kind
    pub fn predicates_by_kind(&self, kind: PredicateKind) -> Vec<&Predicate> {
        self.predicates
            .values()
            .filter(|p| p.kind == kind)
            .collect()
    }

    /// Get number of predicates
    pub fn num_predicates(&self) -> usize {
        self.predicates.len()
    }

    /// Check if predicate exists
    pub fn has_predicate(&self, key: &str) -> bool {
        self.predicates.contains_key(key)
    }

    /// Get point by label
    pub fn get_point(&self, label: &str) -> Option<&Point> {
        self.points.get(label)
    }

    /// Get all point labels
    pub fn point_labels(&self) -> Vec<&str> {
        self.points.keys().map(|s| s.as_str()).collect()
    }

    /// Apply a construction
    pub fn apply_construction(&mut self, construction: Construction) {
        match &construction.kind {
            ConstructionKind::Midpoint { p1, p2 } => {
                let mid_label = format!("M_{}_{}", p1, p2);
                self.add_constructed_point(
                    &mid_label,
                    PointConstruction::Midpoint(p1.clone(), p2.clone()),
                );
                // Add midpoint predicate
                let pred = Predicate::midpoint(&mid_label, p1, p2);
                self.add_axiom(pred); // Axiom because construction guarantees it
            }
            ConstructionKind::Circumcircle { p1, p2, p3 } => {
                let center_label = format!("O_{}_{}{}", p1, p2, p3);
                self.add_constructed_point(
                    &center_label,
                    PointConstruction::Circumcenter(p1.clone(), p2.clone(), p3.clone()),
                );
                self.add_circle(&center_label, p1);

                // Add on_circle predicates
                self.add_axiom(Predicate::on_circle(p1, &center_label, p1));
                self.add_axiom(Predicate::on_circle(p2, &center_label, p1));
                self.add_axiom(Predicate::on_circle(p3, &center_label, p1));
            }
            ConstructionKind::LineIntersection {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => {
                let inter_label = format!("I_{}{}_{}{}", l1_p1, l1_p2, l2_p1, l2_p2);
                self.add_constructed_point(
                    &inter_label,
                    PointConstruction::LineLineIntersection(
                        format!("{}{}", l1_p1, l1_p2),
                        format!("{}{}", l2_p1, l2_p2),
                    ),
                );
                // Add collinear predicates
                self.add_axiom(Predicate::collinear(&inter_label, l1_p1, l1_p2));
                self.add_axiom(Predicate::collinear(&inter_label, l2_p1, l2_p2));
            }
            ConstructionKind::Foot {
                point,
                line_p1,
                line_p2,
            } => {
                let foot_label = format!("F_{}_{}{}", point, line_p1, line_p2);
                self.add_constructed_point(
                    &foot_label,
                    PointConstruction::PerpendicularFoot(
                        point.clone(),
                        line_p1.clone(),
                        line_p2.clone(),
                    ),
                );
                // Add collinear (foot on line) and perpendicular
                self.add_axiom(Predicate::collinear(&foot_label, line_p1, line_p2));
                self.add_axiom(Predicate::perpendicular(
                    point,
                    &foot_label,
                    line_p1,
                    line_p2,
                ));
            }
            // ... other constructions
            _ => {}
        }

        self.constructions.push(construction);
    }

    /// Get provenance trace for a predicate
    pub fn get_provenance_trace(&self, pred_id: PredicateId) -> Vec<&ProofStep> {
        // BFS to collect all ancestors
        let mut visited: HashSet<PredicateId> = HashSet::new();
        let mut indices: Vec<usize> = Vec::new();

        fn collect(
            state: &ProofState,
            id: PredicateId,
            visited: &mut HashSet<PredicateId>,
            result: &mut Vec<usize>,
        ) {
            if visited.contains(&id) {
                return;
            }
            visited.insert(id);

            if let Some(node) = state.provenance.get(&id) {
                for parent in &node.parents {
                    collect(state, *parent, visited, result);
                }
                // Find index in trace
                if let Some(idx) = state.trace.iter().position(|s| s.predicate.id == id) {
                    result.push(idx);
                }
            }
        }

        collect(self, pred_id, &mut visited, &mut indices);
        indices.sort();

        indices.into_iter().map(|i| &self.trace[i]).collect()
    }

    /// Generate human-readable proof
    pub fn generate_proof_text(&self) -> String {
        let mut proof = String::new();

        proof.push_str("=== PROOF ===\n\n");

        // Given points
        proof.push_str("Given:\n");
        for (label, point) in &self.points {
            if point.is_free {
                proof.push_str(&format!("  Point {}\n", label));
            }
        }
        proof.push('\n');

        // Axioms
        proof.push_str("Axioms:\n");
        for pred in self.predicates.values() {
            if matches!(pred.epistemic.source, crate::epistemic::Source::Axiom) {
                proof.push_str(&format!("  {}\n", pred));
            }
        }
        proof.push('\n');

        // Constructions
        if !self.constructions.is_empty() {
            proof.push_str("Constructions:\n");
            for (i, constr) in self.constructions.iter().enumerate() {
                proof.push_str(&format!(
                    "  {}. {:?} (conf: {:.2})\n",
                    i + 1,
                    constr.kind,
                    constr.confidence
                ));
            }
            proof.push('\n');
        }

        // Proof steps
        proof.push_str("Proof:\n");
        for (i, step) in self.trace.iter().enumerate() {
            proof.push_str(&format!(
                "  {}. {} [{}] (conf: {:.3})\n",
                i + 1,
                step.predicate,
                step.rule,
                step.confidence
            ));
        }
        proof.push('\n');

        // Goal
        if let Some(goal) = &self.goal {
            if self.goal_satisfied() {
                proof.push_str(&format!("GOAL PROVEN: {}\n", goal.predicate));
            } else {
                proof.push_str(&format!("GOAL NOT PROVEN: {}\n", goal.predicate));
            }
        }

        proof.push_str(&format!(
            "\nOverall confidence: {:.3}\n",
            self.confidence.mean()
        ));

        proof
    }
}

impl Default for ProofState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_state_basic() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));

        assert_eq!(state.points.len(), 3);
        assert_eq!(state.num_predicates(), 1);
    }

    #[test]
    fn test_goal_satisfaction() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C"]);

        let goal = Predicate::collinear("A", "B", "C");
        state.set_goal(goal.clone(), 0.9);

        assert!(!state.goal_satisfied());

        state.add_axiom(goal);
        assert!(state.goal_satisfied());
    }

    #[test]
    fn test_derived_predicate() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);

        let p1 = Predicate::collinear("A", "B", "C");
        let p2 = Predicate::collinear("A", "B", "D");
        state.add_axiom(p1.clone());
        state.add_axiom(p2.clone());

        let derived = Predicate::collinear("C", "B", "D");
        let added = state.add_derived(derived, "collinear_trans", &[p1.id, p2.id], 0.99);

        assert!(added);
        assert_eq!(state.num_predicates(), 3);
        assert_eq!(state.trace.len(), 1);
    }

    #[test]
    fn test_construction() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B"]);

        let constr = Construction {
            kind: ConstructionKind::Midpoint {
                p1: "A".to_string(),
                p2: "B".to_string(),
            },
            new_points: vec!["M_A_B".to_string()],
            confidence: 1.0,
            source: ConstructionSource::Symbolic,
        };

        state.apply_construction(constr);

        assert!(state.points.contains_key("M_A_B"));
        assert!(state.has_predicate(&Predicate::midpoint("M_A_B", "A", "B").key()));
    }
}
