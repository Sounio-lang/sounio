//! Proof search algorithm for dependent types
//!
//! This module implements automatic proof search to find witnesses
//! for type-level predicates. The algorithm proceeds in stages:
//!
//! 1. **Normalization**: Simplify the predicate
//! 2. **Trivial check**: Handle ⊤, ⊥, reflexivity
//! 3. **Context lookup**: Check if predicate is assumed
//! 4. **Decomposition**: Break apart logical connectives
//! 5. **Decision procedures**: Use specialized solvers
//! 6. **Transitivity**: Try chaining through intermediates
//! 7. **Fallback**: Runtime check (if gradual) or failure

use super::TypeContext;
use super::predicates::{
    CausalPredicate, ConfidencePredicate, OntologyPredicate, Predicate, PredicateKind,
    TemporalPredicate,
};
use super::proofs::{ArithDerivation, CausalProof, Proof, ProofKind};
use super::types::{CausalGraphType, ConfidenceType};
use std::collections::HashSet;

/// Result of proof search
#[derive(Debug, Clone)]
pub enum ProofResult {
    /// Proof found
    Proven(Proof),
    /// Predicate is definitely false
    Disproven { reason: String },
    /// Cannot determine
    Unknown { reason: String },
}

impl ProofResult {
    /// Check if proof was found
    pub fn is_proven(&self) -> bool {
        matches!(self, Self::Proven(_))
    }

    /// Check if definitely false
    pub fn is_disproven(&self) -> bool {
        matches!(self, Self::Disproven { .. })
    }

    /// Get the proof if found
    pub fn proof(self) -> Option<Proof> {
        match self {
            Self::Proven(p) => Some(p),
            _ => None,
        }
    }
}

/// Configuration for proof search
#[derive(Debug, Clone)]
pub struct ProofSearchConfig {
    /// Maximum search depth
    pub max_depth: usize,
    /// Whether to allow gradual typing fallback
    pub allow_gradual: bool,
    /// Whether to print debug information
    pub debug: bool,
    /// Search strategy
    pub strategy: SearchStrategy,
}

impl Default for ProofSearchConfig {
    fn default() -> Self {
        Self {
            max_depth: 10,
            allow_gradual: false,
            debug: false,
            strategy: SearchStrategy::DepthFirst,
        }
    }
}

/// Search strategy for proof search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchStrategy {
    /// Depth-first search
    DepthFirst,
    /// Breadth-first search
    BreadthFirst,
    /// Iterative deepening
    IterativeDeepening,
}

/// Proof searcher
pub struct ProofSearcher<'a> {
    /// Type context
    ctx: &'a TypeContext,
    /// Configuration
    config: ProofSearchConfig,
    /// Current search depth
    depth: usize,
}

impl<'a> ProofSearcher<'a> {
    /// Create a new proof searcher
    pub fn new(ctx: &'a TypeContext) -> Self {
        Self {
            ctx,
            config: ProofSearchConfig::default(),
            depth: 0,
        }
    }

    /// Create with configuration
    pub fn with_config(ctx: &'a TypeContext, config: ProofSearchConfig) -> Self {
        Self {
            ctx,
            config,
            depth: 0,
        }
    }

    /// Search for a proof of the given predicate
    pub fn search(&mut self, pred: &Predicate) -> ProofResult {
        if self.depth > self.config.max_depth {
            return ProofResult::Unknown {
                reason: "Maximum search depth exceeded".to_string(),
            };
        }

        self.depth += 1;
        let result = self.search_inner(pred);
        self.depth -= 1;
        result
    }

    /// Inner search implementation
    fn search_inner(&mut self, pred: &Predicate) -> ProofResult {
        // 1. Normalize
        let normalized = pred.normalize();

        // 2. Trivial cases
        match &normalized.kind {
            PredicateKind::True => {
                return ProofResult::Proven(Proof::trusted("trivially true", normalized));
            }
            PredicateKind::False => {
                return ProofResult::Disproven {
                    reason: "Predicate is trivially false".to_string(),
                };
            }
            _ => {}
        }

        // 3. Check context for assumptions
        if self.ctx.is_assumed(&normalized) {
            return ProofResult::Proven(Proof::assume("context", normalized));
        }

        // 4. Decomposition based on structure
        match &normalized.kind {
            PredicateKind::And(p, q) => self.search_and(p, q),
            PredicateKind::Or(p, q) => self.search_or(p, q),
            PredicateKind::Not(p) => self.search_not(p),
            PredicateKind::Implies(p, q) => self.search_implies(p, q),
            PredicateKind::Forall { var, ty, body } => self.search_forall(var, ty, body),
            PredicateKind::Exists { var, ty, body } => self.search_exists(var, ty, body),

            // 5. Decision procedures
            PredicateKind::Confidence(cp) => self.confidence_decision(cp),
            PredicateKind::Ontology(op) => self.ontology_decision(op),
            PredicateKind::Causal(cp) => self.causal_decision(cp),
            PredicateKind::Temporal(tp) => self.temporal_decision(tp),

            PredicateKind::True | PredicateKind::False => {
                // Already handled above
                unreachable!()
            }
        }
    }

    /// Search for P ∧ Q: need proofs of both
    fn search_and(&mut self, p: &Predicate, q: &Predicate) -> ProofResult {
        let p_result = self.search(p);
        let q_result = self.search(q);

        match (p_result, q_result) {
            (ProofResult::Proven(pp), ProofResult::Proven(pq)) => {
                ProofResult::Proven(Proof::and_intro(pp, pq))
            }
            (ProofResult::Disproven { reason }, _) | (_, ProofResult::Disproven { reason }) => {
                ProofResult::Disproven { reason }
            }
            (ProofResult::Unknown { reason }, _) | (_, ProofResult::Unknown { reason }) => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::and(p.clone(), q.clone())))
                } else {
                    ProofResult::Unknown { reason }
                }
            }
        }
    }

    /// Search for P ∨ Q: need proof of either
    fn search_or(&mut self, p: &Predicate, q: &Predicate) -> ProofResult {
        let p_result = self.search(p);
        if let ProofResult::Proven(pp) = p_result {
            return ProofResult::Proven(Proof::or_intro_left(pp, q.clone()));
        }

        let q_result = self.search(q);
        if let ProofResult::Proven(pq) = q_result {
            return ProofResult::Proven(Proof::or_intro_right(p.clone(), pq));
        }

        // Both failed
        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::or(p.clone(), q.clone())))
        } else {
            ProofResult::Unknown {
                reason: "Could not prove either disjunct".to_string(),
            }
        }
    }

    /// Search for ¬P: try to disprove P
    fn search_not(&mut self, p: &Predicate) -> ProofResult {
        let p_result = self.search(p);
        match p_result {
            ProofResult::Disproven { .. } => {
                // P is false, so ¬P is true
                ProofResult::Proven(Proof::trusted("negation", Predicate::not(p.clone())))
            }
            ProofResult::Proven(_) => {
                // P is true, so ¬P is false
                ProofResult::Disproven {
                    reason: "Inner predicate is provable".to_string(),
                }
            }
            ProofResult::Unknown { reason } => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::not(p.clone())))
                } else {
                    ProofResult::Unknown { reason }
                }
            }
        }
    }

    /// Search for P → Q: assume P and prove Q
    fn search_implies(&mut self, p: &Predicate, q: &Predicate) -> ProofResult {
        // Extend context with P as assumption
        let mut extended_ctx = self.ctx.clone();
        extended_ctx.assume(p.clone());

        // Search for Q in extended context
        let mut searcher = ProofSearcher::with_config(&extended_ctx, self.config.clone());
        searcher.depth = self.depth;

        let q_result = searcher.search(q);

        match q_result {
            ProofResult::Proven(pq) => ProofResult::Proven(Proof::impl_intro(
                "assumption".to_string(),
                pq,
                Predicate::implies(p.clone(), q.clone()),
            )),
            ProofResult::Disproven { reason } => ProofResult::Unknown {
                reason: format!("Could not prove consequent: {}", reason),
            },
            ProofResult::Unknown { reason } => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::implies(
                        p.clone(),
                        q.clone(),
                    )))
                } else {
                    ProofResult::Unknown { reason }
                }
            }
        }
    }

    /// Search for ∀x:τ. P(x)
    fn search_forall(
        &mut self,
        var: &str,
        _ty: &crate::types::Type,
        body: &Predicate,
    ) -> ProofResult {
        // For now, just try to prove the body with the variable free
        // A full implementation would introduce a fresh constant
        let body_result = self.search(body);

        match body_result {
            ProofResult::Proven(pb) => ProofResult::Proven(Proof::trusted(
                format!("universal over {}", var),
                Predicate::new(PredicateKind::Forall {
                    var: var.to_string(),
                    ty: std::sync::Arc::new(_ty.clone()),
                    body: std::sync::Arc::new(body.clone()),
                }),
            )),
            other => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::new(
                        PredicateKind::Forall {
                            var: var.to_string(),
                            ty: std::sync::Arc::new(_ty.clone()),
                            body: std::sync::Arc::new(body.clone()),
                        },
                    )))
                } else {
                    other
                }
            }
        }
    }

    /// Search for ∃x:τ. P(x)
    fn search_exists(
        &mut self,
        var: &str,
        _ty: &crate::types::Type,
        body: &Predicate,
    ) -> ProofResult {
        // For existential, we need to find a witness
        // This is much harder in general - for now, just try the body
        let body_result = self.search(body);

        match body_result {
            ProofResult::Proven(pb) => ProofResult::Proven(Proof::trusted(
                format!("existential witness for {}", var),
                Predicate::new(PredicateKind::Exists {
                    var: var.to_string(),
                    ty: std::sync::Arc::new(_ty.clone()),
                    body: std::sync::Arc::new(body.clone()),
                }),
            )),
            other => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::new(
                        PredicateKind::Exists {
                            var: var.to_string(),
                            ty: std::sync::Arc::new(_ty.clone()),
                            body: std::sync::Arc::new(body.clone()),
                        },
                    )))
                } else {
                    other
                }
            }
        }
    }

    /// Decision procedure for confidence predicates
    fn confidence_decision(&self, pred: &ConfidencePredicate) -> ProofResult {
        match pred {
            ConfidencePredicate::Geq(lhs, rhs) => self.confidence_geq(lhs, rhs),
            ConfidencePredicate::Leq(lhs, rhs) => self.confidence_geq(rhs, lhs), // Swap
            ConfidencePredicate::Eq(lhs, rhs) => self.confidence_eq(lhs, rhs),
            ConfidencePredicate::Gt(lhs, rhs) => {
                // lhs > rhs iff lhs ≥ rhs ∧ ¬(lhs = rhs)
                let geq = self.confidence_geq(lhs, rhs);
                let eq = self.confidence_eq(lhs, rhs);
                match (geq, eq) {
                    (ProofResult::Proven(_), ProofResult::Disproven { .. }) => {
                        ProofResult::Proven(Proof::arith(
                            ArithDerivation::new(format!("{} > {}", lhs, rhs)),
                            Predicate::new(PredicateKind::Confidence(ConfidencePredicate::Gt(
                                lhs.clone(),
                                rhs.clone(),
                            ))),
                        ))
                    }
                    _ => ProofResult::Unknown {
                        reason: format!("Cannot prove {} > {}", lhs, rhs),
                    },
                }
            }
            ConfidencePredicate::Lt(lhs, rhs) => {
                // lhs < rhs iff rhs > lhs
                self.confidence_decision(&ConfidencePredicate::Gt(rhs.clone(), lhs.clone()))
            }
        }
    }

    /// Check ε₁ ≥ ε₂
    fn confidence_geq(&self, lhs: &ConfidenceType, rhs: &ConfidenceType) -> ProofResult {
        // 1. Literal comparison
        if let (Some(v1), Some(v2)) = (lhs.evaluate(self.ctx), rhs.evaluate(self.ctx)) {
            return if v1 >= v2 {
                ProofResult::Proven(Proof::literal_cmp(v1, v2).unwrap())
            } else {
                ProofResult::Disproven {
                    reason: format!("{} < {}", v1, v2),
                }
            };
        }

        // 2. Lower bound check
        if let (Some(lb), Some(v2)) = (lhs.lower_bound(self.ctx), rhs.evaluate(self.ctx))
            && lb >= v2
        {
            return ProofResult::Proven(Proof::arith(
                ArithDerivation::lower_bound(lb, v2),
                Predicate::confidence_geq(lhs.clone(), rhs.clone()),
            ));
        }

        // 3. Product analysis
        if let ConfidenceType::Product(a, b) = lhs
            && let (Some(la), Some(lb)) = (a.lower_bound(self.ctx), b.lower_bound(self.ctx))
        {
            let product_lb = la * lb;
            if let Some(v2) = rhs.evaluate(self.ctx)
                && product_lb >= v2
            {
                return ProofResult::Proven(Proof::arith(
                    ArithDerivation::product(la, lb, v2),
                    Predicate::confidence_geq(lhs.clone(), rhs.clone()),
                ));
            }
        }

        // 4. Dempster-Shafer analysis
        if let ConfidenceType::DempsterShafer(a, b) = lhs
            && let (Some(la), Some(lb)) = (a.lower_bound(self.ctx), b.lower_bound(self.ctx))
        {
            let ds_lb = 1.0 - (1.0 - la) * (1.0 - lb);
            if let Some(v2) = rhs.evaluate(self.ctx)
                && ds_lb >= v2
            {
                return ProofResult::Proven(Proof::arith(
                    ArithDerivation::dempster_shafer(la, lb, v2),
                    Predicate::confidence_geq(lhs.clone(), rhs.clone()),
                ));
            }
        }

        // 5. Decay analysis
        if let ConfidenceType::Decay {
            base,
            lambda,
            elapsed,
        } = lhs
            && let Some(base_lb) = base.lower_bound(self.ctx)
        {
            let t = elapsed.as_secs_f64();
            let decay_lb = base_lb * (-lambda * t).exp();
            if let Some(v2) = rhs.evaluate(self.ctx)
                && decay_lb >= v2
            {
                return ProofResult::Proven(Proof::arith(
                    ArithDerivation::decay(base_lb, *lambda, t, v2),
                    Predicate::confidence_geq(lhs.clone(), rhs.clone()),
                ));
            }
        }

        // 6. Reflexivity
        if lhs.definitionally_equal(rhs) {
            return ProofResult::Proven(Proof::refl(lhs.clone()));
        }

        // 7. Fallback
        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::confidence_geq(
                lhs.clone(),
                rhs.clone(),
            )))
        } else {
            ProofResult::Unknown {
                reason: format!("Cannot prove {} ≥ {}", lhs, rhs),
            }
        }
    }

    /// Check ε₁ = ε₂
    fn confidence_eq(&self, lhs: &ConfidenceType, rhs: &ConfidenceType) -> ProofResult {
        // Definitional equality
        if lhs.definitionally_equal(rhs) {
            return ProofResult::Proven(Proof::refl(lhs.clone()));
        }

        // Evaluate both
        if let (Some(v1), Some(v2)) = (lhs.evaluate(self.ctx), rhs.evaluate(self.ctx)) {
            return if (v1 - v2).abs() < 1e-10 {
                ProofResult::Proven(Proof::arith(
                    ArithDerivation::new(format!("{} = {}", v1, v2)),
                    Predicate::confidence_eq(lhs.clone(), rhs.clone()),
                ))
            } else {
                ProofResult::Disproven {
                    reason: format!("{} ≠ {}", v1, v2),
                }
            };
        }

        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::confidence_eq(
                lhs.clone(),
                rhs.clone(),
            )))
        } else {
            ProofResult::Unknown {
                reason: format!("Cannot prove {} = {}", lhs, rhs),
            }
        }
    }

    /// Decision procedure for ontology predicates
    fn ontology_decision(&self, pred: &OntologyPredicate) -> ProofResult {
        let result = pred.evaluate(self.ctx);

        match result {
            Some(true) => ProofResult::Proven(Proof::trusted(
                "ontology check",
                Predicate::new(PredicateKind::Ontology(pred.clone())),
            )),
            Some(false) => ProofResult::Disproven {
                reason: format!("Ontology predicate is false: {}", pred),
            },
            None => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::new(
                        PredicateKind::Ontology(pred.clone()),
                    )))
                } else {
                    ProofResult::Unknown {
                        reason: format!("Cannot evaluate ontology predicate: {}", pred),
                    }
                }
            }
        }
    }

    /// Decision procedure for causal predicates
    fn causal_decision(&self, pred: &CausalPredicate) -> ProofResult {
        match pred {
            CausalPredicate::Identifiable {
                graph,
                treatment,
                outcome,
            } => self.check_identifiability(graph, treatment, outcome),

            CausalPredicate::BackdoorSatisfied {
                graph,
                treatment,
                outcome,
                adjustment,
            } => {
                if CausalPredicate::check_backdoor(graph, treatment, outcome, adjustment) {
                    ProofResult::Proven(Proof::new(
                        ProofKind::Causal(CausalProof::BackdoorCheck {
                            graph: graph.clone(),
                            treatment: treatment.clone(),
                            outcome: outcome.clone(),
                            adjustment: adjustment.clone(),
                        }),
                        Predicate::causal(pred.clone()),
                    ))
                } else {
                    ProofResult::Disproven {
                        reason: "Backdoor criterion not satisfied".to_string(),
                    }
                }
            }

            CausalPredicate::FrontdoorSatisfied {
                graph,
                treatment,
                outcome,
                mediators,
            } => {
                if CausalPredicate::check_frontdoor(graph, treatment, outcome, mediators) {
                    ProofResult::Proven(Proof::new(
                        ProofKind::Causal(CausalProof::FrontdoorCheck {
                            graph: graph.clone(),
                            treatment: treatment.clone(),
                            outcome: outcome.clone(),
                            mediators: mediators.clone(),
                        }),
                        Predicate::causal(pred.clone()),
                    ))
                } else {
                    ProofResult::Disproven {
                        reason: "Frontdoor criterion not satisfied".to_string(),
                    }
                }
            }

            CausalPredicate::DSeparated { graph, x, y, z } => {
                // Simplified d-separation check
                let separated = x
                    .iter()
                    .all(|xi| y.iter().all(|yi| self.is_d_separated(graph, xi, yi, z)));

                if separated {
                    ProofResult::Proven(Proof::new(
                        ProofKind::Causal(CausalProof::DSeparation {
                            graph: graph.clone(),
                            x: x.clone(),
                            y: y.clone(),
                            z: z.clone(),
                        }),
                        Predicate::causal(pred.clone()),
                    ))
                } else {
                    ProofResult::Disproven {
                        reason: "Not d-separated".to_string(),
                    }
                }
            }

            CausalPredicate::InstrumentValid { .. } => {
                // Simplified: assume valid if structure matches
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::causal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "IV validity check not implemented".to_string(),
                    }
                }
            }

            CausalPredicate::Unconfounded { graph, x, y } => {
                // Check if there are bidirected edges between x and y
                let canonical = if x < y {
                    (x.clone(), y.clone())
                } else {
                    (y.clone(), x.clone())
                };
                if graph.bidirected.contains(&canonical) {
                    ProofResult::Disproven {
                        reason: format!("Bidirected edge exists between {} and {}", x, y),
                    }
                } else {
                    ProofResult::Proven(Proof::trusted(
                        "no unobserved confounders",
                        Predicate::causal(pred.clone()),
                    ))
                }
            }
        }
    }

    /// Check if effect is identifiable
    fn check_identifiability(
        &self,
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
    ) -> ProofResult {
        // 1. Try backdoor criterion
        if let Some(adjustment) = self.find_backdoor_set(graph, treatment, outcome) {
            return ProofResult::Proven(Proof::new(
                ProofKind::Causal(CausalProof::BackdoorCheck {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                    adjustment,
                }),
                Predicate::causal(CausalPredicate::Identifiable {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                }),
            ));
        }

        // 2. Try frontdoor criterion
        if let Some(mediators) = self.find_frontdoor_set(graph, treatment, outcome) {
            return ProofResult::Proven(Proof::new(
                ProofKind::Causal(CausalProof::FrontdoorCheck {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                    mediators,
                }),
                Predicate::causal(CausalPredicate::Identifiable {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                }),
            ));
        }

        // 3. Check for direct unconfounded path
        if !graph
            .bidirected
            .iter()
            .any(|(a, b)| (a == treatment && b == outcome) || (b == treatment && a == outcome))
            && graph.has_directed_path(treatment, outcome)
        {
            return ProofResult::Proven(Proof::trusted(
                "unconfounded direct effect",
                Predicate::causal(CausalPredicate::Identifiable {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                }),
            ));
        }

        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::causal(
                CausalPredicate::Identifiable {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                },
            )))
        } else {
            ProofResult::Unknown {
                reason: format!(
                    "Cannot prove identifiability of {} → {}",
                    treatment, outcome
                ),
            }
        }
    }

    /// Find a valid backdoor adjustment set
    fn find_backdoor_set(
        &self,
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
    ) -> Option<HashSet<String>> {
        // Start with empty set
        if CausalPredicate::check_backdoor(graph, treatment, outcome, &HashSet::new()) {
            return Some(HashSet::new());
        }

        // Try ancestors of outcome minus descendants of treatment
        let outcome_ancestors = graph.ancestors(outcome);
        let treatment_descendants = graph.descendants(treatment);

        let candidates: HashSet<String> = outcome_ancestors
            .difference(&treatment_descendants)
            .filter(|n| *n != treatment && *n != outcome)
            .cloned()
            .collect();

        // Try the full candidate set
        if CausalPredicate::check_backdoor(graph, treatment, outcome, &candidates) {
            return Some(candidates);
        }

        // Try subsets (greedy minimal)
        for node in &candidates {
            let mut single = HashSet::new();
            single.insert(node.clone());
            if CausalPredicate::check_backdoor(graph, treatment, outcome, &single) {
                return Some(single);
            }
        }

        None
    }

    /// Find a valid frontdoor set
    fn find_frontdoor_set(
        &self,
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
    ) -> Option<HashSet<String>> {
        // Look for mediators: direct children of treatment that are ancestors of outcome
        let children = graph.children(treatment);
        let outcome_ancestors = graph.ancestors(outcome);

        let mediators: HashSet<String> =
            children.intersection(&outcome_ancestors).cloned().collect();

        if !mediators.is_empty()
            && CausalPredicate::check_frontdoor(graph, treatment, outcome, &mediators)
        {
            Some(mediators)
        } else {
            None
        }
    }

    /// Check d-separation between two nodes given conditioning set
    fn is_d_separated(
        &self,
        graph: &CausalGraphType,
        x: &str,
        y: &str,
        z: &HashSet<String>,
    ) -> bool {
        // Simplified check - proper implementation would use Bayes-Ball
        if z.contains(x) || z.contains(y) {
            return true;
        }

        // Check if there's a direct edge not blocked
        if graph.edges.contains(&(x.to_string(), y.to_string()))
            || graph.edges.contains(&(y.to_string(), x.to_string()))
        {
            return false;
        }

        true // Simplified
    }

    /// Decision procedure for temporal predicates
    fn temporal_decision(&mut self, pred: &TemporalPredicate) -> ProofResult {
        match pred {
            TemporalPredicate::Fresh {
                temporal,
                max_age_secs,
            } => {
                // Would need current time - for now, assume unknown
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::temporal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "Freshness requires runtime time check".to_string(),
                    }
                }
            }

            TemporalPredicate::DecayBounded {
                base_confidence,
                lambda,
                max_time_secs,
                min_confidence,
            } => {
                if let Some(base) = base_confidence.evaluate(self.ctx) {
                    let is_bounded = TemporalPredicate::evaluate_decay_bound(
                        base,
                        *lambda,
                        *max_time_secs,
                        *min_confidence,
                    );
                    if is_bounded {
                        ProofResult::Proven(Proof::arith(
                            ArithDerivation::decay(base, *lambda, *max_time_secs, *min_confidence),
                            Predicate::temporal(pred.clone()),
                        ))
                    } else {
                        ProofResult::Disproven {
                            reason: "Decay exceeds bound".to_string(),
                        }
                    }
                } else if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::temporal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "Cannot evaluate base confidence".to_string(),
                    }
                }
            }

            TemporalPredicate::Precedes { .. } => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::temporal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "Temporal ordering requires runtime check".to_string(),
                    }
                }
            }

            TemporalPredicate::Eventually(p) => {
                // Eventually requires model checking - defer to gradual
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::temporal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "LTL eventually requires model checking".to_string(),
                    }
                }
            }

            TemporalPredicate::Always(p) => {
                // Always requires invariant checking
                let inner_result = self.search(p);
                match inner_result {
                    ProofResult::Proven(pi) => ProofResult::Proven(Proof::trusted(
                        "always",
                        Predicate::temporal(pred.clone()),
                    )),
                    _ => {
                        if self.config.allow_gradual {
                            ProofResult::Proven(Proof::runtime_check(Predicate::temporal(
                                pred.clone(),
                            )))
                        } else {
                            ProofResult::Unknown {
                                reason: "Cannot prove always".to_string(),
                            }
                        }
                    }
                }
            }

            TemporalPredicate::Until(_, _) | TemporalPredicate::Since(_, _) => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::temporal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "LTL until/since requires model checking".to_string(),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trivial_true() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&Predicate::true_());
        assert!(result.is_proven());
    }

    #[test]
    fn test_trivial_false() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&Predicate::false_());
        assert!(result.is_disproven());
    }

    #[test]
    fn test_literal_confidence() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_literal_confidence_fails() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::literal(0.80), ConfidenceType::literal(0.90));
        let result = searcher.search(&pred);
        assert!(result.is_disproven());
    }

    #[test]
    fn test_conjunction() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let p1 =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let p2 =
            Predicate::confidence_geq(ConfidenceType::literal(0.85), ConfidenceType::literal(0.80));
        let pred = Predicate::and(p1, p2);

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_disjunction() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let p1 =
            Predicate::confidence_geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        let p2 =
            Predicate::confidence_geq(ConfidenceType::literal(0.70), ConfidenceType::literal(0.90)); // False
        let pred = Predicate::or(p1, p2);

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_variable_with_binding() {
        let mut ctx = TypeContext::new();
        ctx.bind_confidence("ε", ConfidenceType::literal(0.97));

        let mut searcher = ProofSearcher::new(&ctx);

        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_product_bound() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // 0.9 * 0.9 = 0.81 ≥ 0.80
        let pred = Predicate::confidence_geq(
            ConfidenceType::product(ConfidenceType::literal(0.9), ConfidenceType::literal(0.9)),
            ConfidenceType::literal(0.80),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_ds_bound() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // 0.6 ⊕ 0.7 = 1 - 0.4*0.3 = 0.88 ≥ 0.85
        let pred = Predicate::confidence_geq(
            ConfidenceType::dempster_shafer(
                ConfidenceType::literal(0.6),
                ConfidenceType::literal(0.7),
            ),
            ConfidenceType::literal(0.85),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_gradual_fallback() {
        let ctx = TypeContext::new();
        let config = ProofSearchConfig {
            allow_gradual: true,
            ..Default::default()
        };
        let mut searcher = ProofSearcher::with_config(&ctx, config);

        let pred = Predicate::confidence_geq(
            ConfidenceType::var("unknown"),
            ConfidenceType::literal(0.95),
        );
        let result = searcher.search(&pred);
        assert!(result.is_proven()); // Gradual allows runtime check
    }

    #[test]
    fn test_causal_backdoor() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_assumption_in_context() {
        let mut ctx = TypeContext::new();
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        ctx.assume(pred.clone());

        let mut searcher = ProofSearcher::new(&ctx);
        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }
}
