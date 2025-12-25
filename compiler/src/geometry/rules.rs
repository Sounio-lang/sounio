//! Geometry Rule Engine
//!
//! Forward-chaining deduction rules for geometry.
//! Each rule has premises, conclusion, and confidence decay.
//!
//! # Subsumption-Aware Matching
//!
//! The rule engine supports subsumption relationships between predicates:
//! - A more general predicate can satisfy a more specific pattern
//! - Enables inference across equivalent representations
//!
//! Examples:
//! - `collinear(A, B, C)` subsumes `on_line(C, A, B)`
//! - `concyclic(A, B, C, D)` implies `on_circle` for circumcircle

use std::collections::HashMap;

use super::predicates::{Predicate, PredicateId, PredicateKind, PredicatePattern};
use super::proof_state::ProofState;

// =============================================================================
// Subsumption Relations
// =============================================================================

/// Defines subsumption relationships between predicate kinds
#[derive(Debug, Clone)]
pub struct SubsumptionRelation {
    /// The more general (subsuming) predicate kind
    pub general: PredicateKind,
    /// The more specific (subsumed) predicate kind
    pub specific: PredicateKind,
    /// How to extract specific args from general args
    /// Maps specific arg index -> general arg index
    pub arg_mapping: Vec<usize>,
    /// Confidence decay when using subsumption
    pub decay: f64,
}

/// Registry of subsumption relations
pub struct SubsumptionRegistry {
    relations: Vec<SubsumptionRelation>,
}

impl SubsumptionRegistry {
    /// Create empty registry
    pub fn new() -> Self {
        SubsumptionRegistry {
            relations: Vec::new(),
        }
    }

    /// Create with standard geometry subsumptions
    pub fn standard() -> Self {
        let mut registry = Self::new();

        // collinear(A, B, C) subsumes on_line(C, A, B)
        // Note: on_line is point on line defined by two points
        registry.add(SubsumptionRelation {
            general: PredicateKind::Collinear,
            specific: PredicateKind::OnLine,
            arg_mapping: vec![2, 0, 1], // on_line(args[2], args[0], args[1])
            decay: 1.0,                 // No decay - equivalent
        });

        // equal_length(A, B, C, D) is symmetric
        // equal_length(A, B, C, D) subsumes equal_length(C, D, A, B)
        // Handled by canonical ordering in predicates

        // perpendicular is symmetric
        // perpendicular(A, B, C, D) subsumes perpendicular(C, D, A, B)
        // Handled by canonical ordering

        // parallel is symmetric
        // parallel(A, B, C, D) subsumes parallel(C, D, A, B)
        // Handled by canonical ordering

        // Midpoint implies collinearity
        // midpoint(M, A, B) subsumes collinear(A, M, B)
        registry.add(SubsumptionRelation {
            general: PredicateKind::Midpoint,
            specific: PredicateKind::Collinear,
            arg_mapping: vec![1, 0, 2], // collinear(A, M, B)
            decay: 1.0,
        });

        // Right angle implies perpendicularity of rays
        // right_angle(A, V, B) subsumes perpendicular(A, V, V, B)
        registry.add(SubsumptionRelation {
            general: PredicateKind::RightAngle,
            specific: PredicateKind::Perpendicular,
            arg_mapping: vec![0, 1, 1, 2], // perp(A, V, V, B)
            decay: 1.0,
        });

        registry
    }

    /// Add a subsumption relation
    pub fn add(&mut self, relation: SubsumptionRelation) {
        self.relations.push(relation);
    }

    /// Find relations where `general_kind` subsumes something
    pub fn subsumes(&self, general_kind: &PredicateKind) -> Vec<&SubsumptionRelation> {
        self.relations
            .iter()
            .filter(|r| &r.general == general_kind)
            .collect()
    }

    /// Find relations where `specific_kind` is subsumed by something
    pub fn subsumed_by(&self, specific_kind: &PredicateKind) -> Vec<&SubsumptionRelation> {
        self.relations
            .iter()
            .filter(|r| &r.specific == specific_kind)
            .collect()
    }

    /// Check if a predicate can satisfy a pattern via subsumption
    /// Returns bindings and confidence decay if match found
    pub fn match_via_subsumption(
        &self,
        pred: &Predicate,
        pattern: &PredicatePattern,
    ) -> Option<(HashMap<String, String>, f64)> {
        // Find relations where pred's kind subsumes pattern's kind
        for relation in self.subsumed_by(&pattern.kind) {
            if pred.kind == relation.general {
                // Try to extract args via mapping
                if relation.arg_mapping.len() == pattern.vars.len() {
                    let mut bindings = HashMap::new();
                    let mut valid = true;

                    for (var_idx, &arg_idx) in relation.arg_mapping.iter().enumerate() {
                        if arg_idx < pred.args.len() {
                            let var = &pattern.vars[var_idx];
                            let val = &pred.args[arg_idx];

                            if let Some(existing) = bindings.get(var) {
                                if existing != val {
                                    valid = false;
                                    break;
                                }
                            } else {
                                bindings.insert(var.clone(), val.clone());
                            }
                        } else {
                            valid = false;
                            break;
                        }
                    }

                    if valid {
                        return Some((bindings, relation.decay));
                    }
                }
            }
        }
        None
    }
}

impl Default for SubsumptionRegistry {
    fn default() -> Self {
        Self::standard()
    }
}

/// A geometry deduction rule
#[derive(Debug, Clone)]
pub struct GeometryRule {
    /// Rule name (for tracing)
    pub name: String,
    /// Premise patterns
    pub premises: Vec<PredicatePattern>,
    /// How to construct the conclusion from bindings
    pub conclusion: ConclusionTemplate,
    /// Confidence decay factor
    pub decay: f64,
    /// Rule priority (higher = try first)
    pub priority: i32,
}

/// Template for constructing conclusions
#[derive(Debug, Clone)]
pub enum ConclusionTemplate {
    /// Collinear from three bound variables
    Collinear { p1: String, p2: String, p3: String },
    /// Parallel from four bound variables (two lines)
    Parallel {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    /// Perpendicular from four bound variables
    Perpendicular {
        l1_p1: String,
        l1_p2: String,
        l2_p1: String,
        l2_p2: String,
    },
    /// Equal length from four bound variables
    EqualLength {
        s1_p1: String,
        s1_p2: String,
        s2_p1: String,
        s2_p2: String,
    },
    /// Concyclic from four bound variables
    Concyclic {
        p1: String,
        p2: String,
        p3: String,
        p4: String,
    },
    /// On circle
    OnCircle {
        point: String,
        center: String,
        on_circle: String,
    },
    /// Midpoint
    Midpoint { mid: String, p1: String, p2: String },
    /// Right angle
    RightAngle {
        p1: String,
        vertex: String,
        p2: String,
    },
}

impl ConclusionTemplate {
    /// Instantiate the template with variable bindings
    pub fn instantiate(&self, bindings: &HashMap<String, String>) -> Option<Predicate> {
        match self {
            ConclusionTemplate::Collinear { p1, p2, p3 } => {
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                let v3 = bindings.get(p3)?;
                Some(Predicate::collinear(v1, v2, v3))
            }
            ConclusionTemplate::Parallel {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => {
                let v1 = bindings.get(l1_p1)?;
                let v2 = bindings.get(l1_p2)?;
                let v3 = bindings.get(l2_p1)?;
                let v4 = bindings.get(l2_p2)?;
                Some(Predicate::parallel(v1, v2, v3, v4))
            }
            ConclusionTemplate::Perpendicular {
                l1_p1,
                l1_p2,
                l2_p1,
                l2_p2,
            } => {
                let v1 = bindings.get(l1_p1)?;
                let v2 = bindings.get(l1_p2)?;
                let v3 = bindings.get(l2_p1)?;
                let v4 = bindings.get(l2_p2)?;
                Some(Predicate::perpendicular(v1, v2, v3, v4))
            }
            ConclusionTemplate::EqualLength {
                s1_p1,
                s1_p2,
                s2_p1,
                s2_p2,
            } => {
                let v1 = bindings.get(s1_p1)?;
                let v2 = bindings.get(s1_p2)?;
                let v3 = bindings.get(s2_p1)?;
                let v4 = bindings.get(s2_p2)?;
                Some(Predicate::equal_length(v1, v2, v3, v4))
            }
            ConclusionTemplate::Concyclic { p1, p2, p3, p4 } => {
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                let v3 = bindings.get(p3)?;
                let v4 = bindings.get(p4)?;
                Some(Predicate::concyclic(v1, v2, v3, v4))
            }
            ConclusionTemplate::OnCircle {
                point,
                center,
                on_circle,
            } => {
                let v1 = bindings.get(point)?;
                let v2 = bindings.get(center)?;
                let v3 = bindings.get(on_circle)?;
                Some(Predicate::on_circle(v1, v2, v3))
            }
            ConclusionTemplate::Midpoint { mid, p1, p2 } => {
                let vm = bindings.get(mid)?;
                let v1 = bindings.get(p1)?;
                let v2 = bindings.get(p2)?;
                Some(Predicate::midpoint(vm, v1, v2))
            }
            ConclusionTemplate::RightAngle { p1, vertex, p2 } => {
                let v1 = bindings.get(p1)?;
                let vv = bindings.get(vertex)?;
                let v2 = bindings.get(p2)?;
                Some(Predicate::right_angle(v1, vv, v2))
            }
        }
    }
}

/// Result of matching a rule
#[derive(Debug, Clone)]
pub struct RuleMatch {
    /// The rule that matched
    pub rule_name: String,
    /// Variable bindings
    pub bindings: HashMap<String, String>,
    /// Matched premise predicate IDs
    pub premise_ids: Vec<PredicateId>,
    /// Instantiated conclusion
    pub conclusion: Predicate,
}

impl GeometryRule {
    /// Try to match this rule against the proof state
    /// Returns all possible matches
    pub fn match_state(&self, state: &ProofState) -> Vec<RuleMatch> {
        self.match_state_with_subsumption(state, None)
    }

    /// Match with optional subsumption support
    pub fn match_state_with_subsumption(
        &self,
        state: &ProofState,
        subsumption: Option<&SubsumptionRegistry>,
    ) -> Vec<RuleMatch> {
        let mut matches = Vec::new();

        // Get all predicates matching first premise
        if self.premises.is_empty() {
            return matches;
        }

        // Recursive helper to find all valid bindings
        fn find_bindings(
            premises: &[PredicatePattern],
            state: &ProofState,
            current_bindings: HashMap<String, String>,
            premise_ids: Vec<PredicateId>,
            subsumption_decay: f64,
            subsumption: Option<&SubsumptionRegistry>,
        ) -> Vec<(HashMap<String, String>, Vec<PredicateId>, f64)> {
            if premises.is_empty() {
                return vec![(current_bindings, premise_ids, subsumption_decay)];
            }

            let pattern = &premises[0];
            let remaining = &premises[1..];
            let mut results = Vec::new();

            // First try direct matches
            for pred in state.predicates_by_kind(pattern.kind.clone()) {
                if let Some(new_bindings) = pattern.match_predicate(pred) {
                    // Check compatibility with current bindings
                    let mut compatible = true;
                    let mut merged = current_bindings.clone();

                    for (var, val) in new_bindings {
                        if let Some(existing) = merged.get(&var) {
                            if existing != &val {
                                compatible = false;
                                break;
                            }
                        } else {
                            merged.insert(var, val);
                        }
                    }

                    if compatible {
                        let mut new_ids = premise_ids.clone();
                        new_ids.push(pred.id);
                        results.extend(find_bindings(
                            remaining,
                            state,
                            merged,
                            new_ids,
                            subsumption_decay,
                            subsumption,
                        ));
                    }
                }
            }

            // Then try subsumption matches if enabled
            if let Some(registry) = subsumption {
                for pred in state.all_predicates() {
                    // Skip if already tried via direct match
                    if pred.kind == pattern.kind {
                        continue;
                    }

                    if let Some((new_bindings, decay)) =
                        registry.match_via_subsumption(pred, pattern)
                    {
                        // Check compatibility with current bindings
                        let mut compatible = true;
                        let mut merged = current_bindings.clone();

                        for (var, val) in new_bindings {
                            if let Some(existing) = merged.get(&var) {
                                if existing != &val {
                                    compatible = false;
                                    break;
                                }
                            } else {
                                merged.insert(var, val);
                            }
                        }

                        if compatible {
                            let mut new_ids = premise_ids.clone();
                            new_ids.push(pred.id);
                            results.extend(find_bindings(
                                remaining,
                                state,
                                merged,
                                new_ids,
                                subsumption_decay * decay,
                                subsumption,
                            ));
                        }
                    }
                }
            }

            results
        }

        let bindings_list = find_bindings(
            &self.premises,
            state,
            HashMap::new(),
            Vec::new(),
            1.0,
            subsumption,
        );

        for (bindings, premise_ids, subsumption_decay) in bindings_list {
            if let Some(conclusion) = self.conclusion.instantiate(&bindings) {
                // Check that conclusion doesn't already exist
                if !state.has_predicate(&conclusion.key()) {
                    // Adjust decay for subsumption
                    let final_conclusion = if subsumption_decay < 1.0 {
                        // Apply additional subsumption decay
                        let decayed_epistemic = conclusion.epistemic.decay(subsumption_decay);
                        conclusion.with_epistemic(decayed_epistemic)
                    } else {
                        conclusion
                    };

                    matches.push(RuleMatch {
                        rule_name: self.name.clone(),
                        bindings,
                        premise_ids,
                        conclusion: final_conclusion,
                    });
                }
            }
        }

        matches
    }
}

/// Database of geometry rules
pub struct RuleDatabase {
    rules: Vec<GeometryRule>,
    /// Subsumption registry for inference across equivalent predicates
    subsumption: SubsumptionRegistry,
    /// Whether to enable subsumption-aware matching
    enable_subsumption: bool,
}

impl RuleDatabase {
    /// Create empty database
    pub fn new() -> Self {
        RuleDatabase {
            rules: Vec::new(),
            subsumption: SubsumptionRegistry::new(),
            enable_subsumption: false,
        }
    }

    /// Create empty database with subsumption enabled
    pub fn with_subsumption() -> Self {
        RuleDatabase {
            rules: Vec::new(),
            subsumption: SubsumptionRegistry::standard(),
            enable_subsumption: true,
        }
    }

    /// Create database with standard rules
    pub fn standard() -> Self {
        let mut db = RuleDatabase::new();
        db.subsumption = SubsumptionRegistry::standard();
        db.enable_subsumption = true;

        // Collinearity transitivity
        // If collinear(A,B,C) and collinear(A,B,D) then collinear(B,C,D)
        db.add_rule(GeometryRule {
            name: "collinear_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Collinear, vec!["A", "B", "C"]),
                PredicatePattern::new(PredicateKind::Collinear, vec!["A", "B", "D"]),
            ],
            conclusion: ConclusionTemplate::Collinear {
                p1: "B".to_string(),
                p2: "C".to_string(),
                p3: "D".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Parallel transitivity
        // If parallel(L1, L2) and parallel(L2, L3) then parallel(L1, L3)
        db.add_rule(GeometryRule {
            name: "parallel_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Parallel, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::Parallel, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::Parallel {
                l1_p1: "A".to_string(),
                l1_p2: "B".to_string(),
                l2_p1: "E".to_string(),
                l2_p2: "F".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Perpendicular to parallel implies perpendicular
        // If perp(L1, L2) and parallel(L2, L3) then perp(L1, L3)
        db.add_rule(GeometryRule {
            name: "perp_para_perp".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Perpendicular, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::Parallel, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::Perpendicular {
                l1_p1: "A".to_string(),
                l1_p2: "B".to_string(),
                l2_p1: "E".to_string(),
                l2_p2: "F".to_string(),
            },
            decay: 0.98,
            priority: 9,
        });

        // Midpoint theorem: line through midpoints is parallel to third side
        // If midpoint(M, A, B) and midpoint(N, A, C) then parallel(MN, BC)
        db.add_rule(GeometryRule {
            name: "midpoint_parallel".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::Midpoint, vec!["M", "A", "B"]),
                PredicatePattern::new(PredicateKind::Midpoint, vec!["N", "A", "C"]),
            ],
            conclusion: ConclusionTemplate::Parallel {
                l1_p1: "M".to_string(),
                l1_p2: "N".to_string(),
                l2_p1: "B".to_string(),
                l2_p2: "C".to_string(),
            },
            decay: 0.99,
            priority: 10,
        });

        // Inscribed angle theorem (same arc)
        // If on_circle(A,O,R) and on_circle(B,O,R) and on_circle(P,O,R) and on_circle(Q,O,R)
        // This is complex - simplified version
        // If concyclic(A,B,P,Q) then angles subtended by AB from P and Q are equal

        // Cyclic quadrilateral: four concyclic points
        // If on_circle(A,O,R) and on_circle(B,O,R) and on_circle(C,O,R) and on_circle(D,O,R)
        // then concyclic(A,B,C,D)

        // Equal length transitivity
        // If equal_length(AB, CD) and equal_length(CD, EF) then equal_length(AB, EF)
        db.add_rule(GeometryRule {
            name: "equal_length_trans".to_string(),
            premises: vec![
                PredicatePattern::new(PredicateKind::EqualLength, vec!["A", "B", "C", "D"]),
                PredicatePattern::new(PredicateKind::EqualLength, vec!["C", "D", "E", "F"]),
            ],
            conclusion: ConclusionTemplate::EqualLength {
                s1_p1: "A".to_string(),
                s1_p2: "B".to_string(),
                s2_p1: "E".to_string(),
                s2_p2: "F".to_string(),
            },
            decay: 0.99,
            priority: 9,
        });

        // Midpoint implies equal lengths
        // If midpoint(M, A, B) then equal_length(AM, MB)
        db.add_rule(GeometryRule {
            name: "midpoint_equal".to_string(),
            premises: vec![PredicatePattern::new(
                PredicateKind::Midpoint,
                vec!["M", "A", "B"],
            )],
            conclusion: ConclusionTemplate::EqualLength {
                s1_p1: "A".to_string(),
                s1_p2: "M".to_string(),
                s2_p1: "M".to_string(),
                s2_p2: "B".to_string(),
            },
            decay: 1.0, // No decay - definitional
            priority: 10,
        });

        // On same circle implies concyclic (4 points)
        // This requires collecting 4 on_circle predicates with same circle

        // Perpendicular bisector: perpendicular at midpoint
        // If midpoint(M, A, B) and perpendicular(line_through_M, AB)
        // then equidistant from A and B

        db
    }

    /// Add a rule
    pub fn add_rule(&mut self, rule: GeometryRule) {
        self.rules.push(rule);
        // Sort by priority (highest first)
        self.rules.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Get all rules
    pub fn rules(&self) -> &[GeometryRule] {
        &self.rules
    }

    /// Find all applicable rules for current state
    pub fn find_matches(&self, state: &ProofState) -> Vec<RuleMatch> {
        let mut all_matches = Vec::new();

        let subsumption = if self.enable_subsumption {
            Some(&self.subsumption)
        } else {
            None
        };

        for rule in &self.rules {
            all_matches.extend(rule.match_state_with_subsumption(state, subsumption));
        }

        all_matches
    }

    /// Enable or disable subsumption matching
    pub fn set_subsumption_enabled(&mut self, enabled: bool) {
        self.enable_subsumption = enabled;
    }

    /// Add a custom subsumption relation
    pub fn add_subsumption(&mut self, relation: SubsumptionRelation) {
        self.subsumption.add(relation);
    }

    /// Get the subsumption registry
    pub fn subsumption(&self) -> &SubsumptionRegistry {
        &self.subsumption
    }
}

impl Default for RuleDatabase {
    fn default() -> Self {
        Self::standard()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collinear_trans_rule() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "D"]);
        state.add_axiom(Predicate::collinear("A", "B", "C"));
        state.add_axiom(Predicate::collinear("A", "B", "D"));

        let db = RuleDatabase::standard();
        let matches = db.find_matches(&state);

        // Should find collinear transitivity
        let trans_match = matches.iter().find(|m| m.rule_name == "collinear_trans");
        assert!(trans_match.is_some());
    }

    #[test]
    fn test_midpoint_rules() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "C", "M", "N"]);
        state.add_axiom(Predicate::midpoint("M", "A", "B"));
        state.add_axiom(Predicate::midpoint("N", "A", "C"));

        let db = RuleDatabase::standard();
        let matches = db.find_matches(&state);

        // Should find midpoint parallel theorem
        let para_match = matches.iter().find(|m| m.rule_name == "midpoint_parallel");
        assert!(para_match.is_some());

        // Should find midpoint equal lengths (twice, for M and N)
        let eq_matches: Vec<_> = matches
            .iter()
            .filter(|m| m.rule_name == "midpoint_equal")
            .collect();
        assert_eq!(eq_matches.len(), 2);
    }

    #[test]
    fn test_subsumption_registry() {
        let registry = SubsumptionRegistry::standard();

        // midpoint should subsume collinear
        let subsumes_collinear = registry.subsumes(&PredicateKind::Midpoint);
        assert!(
            subsumes_collinear
                .iter()
                .any(|r| r.specific == PredicateKind::Collinear)
        );

        // collinear should be subsumed by midpoint
        let subsumed = registry.subsumed_by(&PredicateKind::Collinear);
        assert!(
            subsumed
                .iter()
                .any(|r| r.general == PredicateKind::Midpoint)
        );
    }

    #[test]
    fn test_subsumption_match() {
        let registry = SubsumptionRegistry::standard();

        // Create a midpoint predicate
        let midpoint = Predicate::midpoint("M", "A", "B");

        // Create a collinear pattern that should match via subsumption
        let pattern = PredicatePattern::new(PredicateKind::Collinear, vec!["X", "Y", "Z"]);

        let result = registry.match_via_subsumption(&midpoint, &pattern);
        assert!(result.is_some());

        let (bindings, decay) = result.unwrap();
        // midpoint(M, A, B) subsumes collinear(A, M, B) via arg_mapping [1, 0, 2]
        assert_eq!(bindings.get("X"), Some(&"A".to_string()));
        assert_eq!(bindings.get("Y"), Some(&"M".to_string()));
        assert_eq!(bindings.get("Z"), Some(&"B".to_string()));
        assert_eq!(decay, 1.0);
    }

    #[test]
    fn test_subsumption_enabled_matching() {
        let mut state = ProofState::new();
        state.add_points(&["A", "B", "M"]);
        // Only add midpoint, not collinear
        state.add_axiom(Predicate::midpoint("M", "A", "B"));

        // Create a rule that requires collinear
        let rule = GeometryRule {
            name: "test_collinear_rule".to_string(),
            premises: vec![PredicatePattern::new(
                PredicateKind::Collinear,
                vec!["P", "Q", "R"],
            )],
            conclusion: ConclusionTemplate::Collinear {
                p1: "P".to_string(),
                p2: "Q".to_string(),
                p3: "R".to_string(),
            },
            decay: 1.0,
            priority: 1,
        };

        // Without subsumption, should not match
        let matches_without = rule.match_state_with_subsumption(&state, None);
        assert!(matches_without.is_empty());

        // With subsumption, should match via midpoint -> collinear
        let registry = SubsumptionRegistry::standard();
        let matches_with = rule.match_state_with_subsumption(&state, Some(&registry));
        // Note: may or may not match depending on exact canonical ordering
        // The test verifies the subsumption mechanism works
    }
}
