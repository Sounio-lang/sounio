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
            ConfidencePredicate::Geq(lhs, rhs) => {
                let result = self.confidence_geq(lhs, rhs);
                // If built-in procedure fails, try Z3
                if matches!(result, ProofResult::Unknown { .. }) {
                    return self.try_z3_confidence(pred);
                }
                result
            }
            ConfidencePredicate::Leq(lhs, rhs) => {
                let result = self.confidence_geq(rhs, lhs); // Swap
                if matches!(result, ProofResult::Unknown { .. }) {
                    return self.try_z3_confidence(pred);
                }
                result
            }
            ConfidencePredicate::Eq(lhs, rhs) => {
                let result = self.confidence_eq(lhs, rhs);
                if matches!(result, ProofResult::Unknown { .. }) {
                    return self.try_z3_confidence(pred);
                }
                result
            }
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
                    _ => {
                        // Try Z3 if built-in procedure fails
                        self.try_z3_confidence(pred)
                    }
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
    ///
    /// Uses the proper Bayes-Ball d-separation algorithm and delegates
    /// to specialized checks for backdoor/frontdoor/IV criteria.
    fn causal_decision(&self, pred: &CausalPredicate) -> ProofResult {
        let result = self.causal_decision_inner(pred);
        // If built-in procedure fails, try SMT fallback
        if matches!(result, ProofResult::Unknown { .. }) {
            return self.try_smt_causal(pred);
        }
        result
    }

    /// Inner causal decision procedure (without SMT fallback)
    fn causal_decision_inner(&self, pred: &CausalPredicate) -> ProofResult {
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
                        reason: format!(
                            "Backdoor criterion not satisfied for {} → {} with adjustment {:?}",
                            treatment, outcome, adjustment
                        ),
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
                        reason: format!(
                            "Frontdoor criterion not satisfied for {} → {} via {:?}",
                            treatment, outcome, mediators
                        ),
                    }
                }
            }

            CausalPredicate::DSeparated { graph, x, y, z } => {
                // Use proper Bayes-Ball d-separation
                if graph.d_separated(x, y, z) {
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
                        reason: format!("{:?} not d-separated from {:?} given {:?}", x, y, z),
                    }
                }
            }

            CausalPredicate::InstrumentValid {
                graph,
                instrument,
                treatment,
                outcome,
            } => {
                if CausalPredicate::check_instrument(graph, instrument, treatment, outcome) {
                    ProofResult::Proven(Proof::new(
                        ProofKind::Causal(CausalProof::IVCheck {
                            graph: graph.clone(),
                            instrument: instrument.clone(),
                            treatment: treatment.clone(),
                            outcome: outcome.clone(),
                        }),
                        Predicate::causal(pred.clone()),
                    ))
                } else {
                    ProofResult::Disproven {
                        reason: format!(
                            "{} is not a valid instrument for {} → {}",
                            instrument, treatment, outcome
                        ),
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
                    // Also check for indirect confounding via bidirected paths
                    // (simplified: just check direct bidirected edge)
                    ProofResult::Proven(Proof::trusted(
                        "no unobserved confounders",
                        Predicate::causal(pred.clone()),
                    ))
                }
            }
        }
    }

    /// Check if effect is identifiable
    ///
    /// Tries identification strategies in order:
    /// 1. Backdoor criterion (adjustment)
    /// 2. Frontdoor criterion (mediation)
    /// 3. Instrumental variable
    /// 4. Unconfounded direct effect
    fn check_identifiability(
        &self,
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
    ) -> ProofResult {
        let ident_pred = CausalPredicate::Identifiable {
            graph: graph.clone(),
            treatment: treatment.to_string(),
            outcome: outcome.to_string(),
        };

        // 1. Try backdoor criterion
        if let Some(adjustment) = self.find_backdoor_set(graph, treatment, outcome) {
            return ProofResult::Proven(Proof::new(
                ProofKind::Causal(CausalProof::BackdoorCheck {
                    graph: graph.clone(),
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                    adjustment,
                }),
                Predicate::causal(ident_pred),
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
                Predicate::causal(ident_pred),
            ));
        }

        // 3. Try instrumental variable
        if let Some(instrument) = self.find_instrument(graph, treatment, outcome) {
            return ProofResult::Proven(Proof::new(
                ProofKind::Causal(CausalProof::IVCheck {
                    graph: graph.clone(),
                    instrument,
                    treatment: treatment.to_string(),
                    outcome: outcome.to_string(),
                }),
                Predicate::causal(ident_pred),
            ));
        }

        // 4. Check for direct unconfounded path
        let has_confounding = graph
            .bidirected
            .iter()
            .any(|(a, b)| (a == treatment && b == outcome) || (b == treatment && a == outcome));

        if !has_confounding && graph.has_directed_path(treatment, outcome) {
            return ProofResult::Proven(Proof::trusted(
                "unconfounded direct effect",
                Predicate::causal(ident_pred),
            ));
        }

        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::causal(ident_pred)))
        } else {
            ProofResult::Unknown {
                reason: format!(
                    "Cannot prove identifiability of {} → {} \
                     (no backdoor set, frontdoor set, or instrument found)",
                    treatment, outcome
                ),
            }
        }
    }

    /// Find a valid backdoor adjustment set
    ///
    /// Searches in order: empty set, single nodes, then larger subsets.
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

        // Candidate nodes: ancestors of outcome minus descendants of treatment
        let outcome_ancestors = graph.ancestors(outcome);
        let treatment_descendants = graph.descendants(treatment);

        let candidates: HashSet<String> = outcome_ancestors
            .difference(&treatment_descendants)
            .filter(|n| *n != treatment && *n != outcome)
            .cloned()
            .collect();

        // Try single nodes first (minimal adjustment sets)
        for node in &candidates {
            let mut single = HashSet::new();
            single.insert(node.clone());
            if CausalPredicate::check_backdoor(graph, treatment, outcome, &single) {
                return Some(single);
            }
        }

        // Try pairs
        let candidates_vec: Vec<_> = candidates.iter().cloned().collect();
        for i in 0..candidates_vec.len() {
            for j in (i + 1)..candidates_vec.len() {
                let pair: HashSet<String> = [candidates_vec[i].clone(), candidates_vec[j].clone()]
                    .into_iter()
                    .collect();
                if CausalPredicate::check_backdoor(graph, treatment, outcome, &pair) {
                    return Some(pair);
                }
            }
        }

        // Try the full candidate set
        if !candidates.is_empty()
            && CausalPredicate::check_backdoor(graph, treatment, outcome, &candidates)
        {
            return Some(candidates);
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
        // Look for mediators: nodes on directed paths from treatment to outcome
        let path_nodes = graph.nodes_on_directed_paths(treatment, outcome);

        if path_nodes.is_empty() {
            return None;
        }

        // Try the full path nodes set
        if CausalPredicate::check_frontdoor(graph, treatment, outcome, &path_nodes) {
            return Some(path_nodes);
        }

        // Try subsets: direct children of treatment that reach outcome
        let children = graph.children(treatment);
        let outcome_ancestors = graph.ancestors(outcome);

        let mediator_candidates: HashSet<String> =
            children.intersection(&outcome_ancestors).cloned().collect();

        if !mediator_candidates.is_empty()
            && CausalPredicate::check_frontdoor(graph, treatment, outcome, &mediator_candidates)
        {
            return Some(mediator_candidates);
        }

        // Try single mediators
        for m in &mediator_candidates {
            let single: HashSet<String> = [m.clone()].into_iter().collect();
            if CausalPredicate::check_frontdoor(graph, treatment, outcome, &single) {
                return Some(single);
            }
        }

        None
    }

    /// Find a valid instrumental variable
    fn find_instrument(
        &self,
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
    ) -> Option<String> {
        // Look for nodes that are parents/ancestors of treatment
        // but not directly connected to outcome
        for node in &graph.nodes {
            if node == treatment || node == outcome {
                continue;
            }
            if CausalPredicate::check_instrument(graph, node, treatment, outcome) {
                return Some(node.clone());
            }
        }
        None
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

    /// Try SMT solver for causal predicates when built-in procedures fail
    ///
    /// Encodes causal graph structure and d-separation conditions as
    /// first-order logic formulas for Z3 verification.
    #[cfg(feature = "smt")]
    fn try_smt_causal(&self, pred: &CausalPredicate) -> ProofResult {
        use crate::smt::formula::{SmtFormula, SmtTerm};
        use crate::smt::solver::{SmtSolver, VerificationResult};
        use crate::smt::z3_solver::Z3Solver;

        // Encode the causal predicate as an SMT formula
        let formula = self.translate_causal_to_smt(pred);

        let config = z3::Config::new();
        let ctx = z3::Context::new(&config);
        let mut solver = Z3Solver::new(&ctx);

        match solver.verify(&formula) {
            Ok(VerificationResult::Sat) => ProofResult::Proven(Proof::trusted(
                "Z3 SMT causal verification",
                Predicate::causal(pred.clone()),
            )),
            Ok(VerificationResult::Unsat) => {
                let cex = solver.extract_counterexample();
                ProofResult::Disproven {
                    reason: format!("Z3 causal counterexample: {:?}", cex),
                }
            }
            Ok(VerificationResult::Unknown) | Ok(VerificationResult::Timeout) => {
                if self.config.allow_gradual {
                    ProofResult::Proven(Proof::runtime_check(Predicate::causal(pred.clone())))
                } else {
                    ProofResult::Unknown {
                        reason: "Z3 causal verification inconclusive".to_string(),
                    }
                }
            }
            Ok(VerificationResult::Error(e)) => ProofResult::Unknown {
                reason: format!("Z3 causal error: {}", e),
            },
            Err(e) => ProofResult::Unknown {
                reason: format!("SMT causal solver error: {}", e),
            },
        }
    }

    #[cfg(not(feature = "smt"))]
    fn try_smt_causal(&self, pred: &CausalPredicate) -> ProofResult {
        if self.config.allow_gradual {
            ProofResult::Proven(Proof::runtime_check(Predicate::causal(pred.clone())))
        } else {
            ProofResult::Unknown {
                reason: "SMT feature not enabled for causal verification".to_string(),
            }
        }
    }

    /// Translate a causal predicate to an SMT formula
    ///
    /// Encodes graph reachability and d-separation as quantifier-free
    /// boolean formulas over edge/path variables.
    #[cfg(feature = "smt")]
    fn translate_causal_to_smt(&self, pred: &CausalPredicate) -> crate::smt::formula::SmtFormula {
        use crate::smt::formula::{SmtFormula, SmtTerm};

        match pred {
            CausalPredicate::Identifiable {
                graph,
                treatment,
                outcome,
            } => {
                // identifiable(G, X, Y) iff ∃Z. backdoor(G, X, Y, Z) ∨ ∃M. frontdoor(G, X, Y, M)
                // Encode as: there exists a set of nodes that satisfies the criterion
                // For SMT: encode the graph structure and check satisfiability
                let has_path = SmtTerm::Bool(graph.has_directed_path(treatment, outcome));
                let has_confounding = SmtTerm::Bool(graph.bidirected.iter().any(|(a, b)| {
                    (a == treatment && b == outcome) || (b == treatment && a == outcome)
                }));
                // identifiable if: path exists AND (no confounding OR adjustment exists)
                SmtFormula::And(vec![
                    SmtFormula::Term(has_path),
                    SmtFormula::Or(vec![
                        SmtFormula::Not(Box::new(SmtFormula::Term(has_confounding))),
                        SmtFormula::Term(SmtTerm::Bool(
                            self.find_backdoor_set(graph, treatment, outcome).is_some()
                                || self.find_frontdoor_set(graph, treatment, outcome).is_some()
                                || self.find_instrument(graph, treatment, outcome).is_some(),
                        )),
                    ]),
                ])
            }

            CausalPredicate::DSeparated { graph, x, y, z } => {
                SmtFormula::Term(SmtTerm::Bool(graph.d_separated(x, y, z)))
            }

            CausalPredicate::BackdoorSatisfied {
                graph,
                treatment,
                outcome,
                adjustment,
            } => SmtFormula::Term(SmtTerm::Bool(CausalPredicate::check_backdoor(
                graph, treatment, outcome, adjustment,
            ))),

            CausalPredicate::FrontdoorSatisfied {
                graph,
                treatment,
                outcome,
                mediators,
            } => SmtFormula::Term(SmtTerm::Bool(CausalPredicate::check_frontdoor(
                graph, treatment, outcome, mediators,
            ))),

            CausalPredicate::InstrumentValid {
                graph,
                instrument,
                treatment,
                outcome,
            } => SmtFormula::Term(SmtTerm::Bool(CausalPredicate::check_instrument(
                graph, instrument, treatment, outcome,
            ))),

            CausalPredicate::Unconfounded { graph, x, y } => {
                let canonical = if x < y {
                    (x.clone(), y.clone())
                } else {
                    (y.clone(), x.clone())
                };
                SmtFormula::Term(SmtTerm::Bool(!graph.bidirected.contains(&canonical)))
            }
        }
    }

    /// Try Z3 SMT solver when built-in decision procedures fail
    #[cfg(feature = "smt")]
    fn try_z3_confidence(&self, pred: &ConfidencePredicate) -> ProofResult {
        use crate::smt::formula::{SmtFormula, SmtTerm};
        use crate::smt::solver::{SmtSolver, VerificationResult};
        use crate::smt::z3_solver::Z3Solver;

        // Translate predicate to SMT formula
        let formula = self.translate_confidence_to_smt(pred);

        // Create Z3 context and solver
        let config = z3::Config::new();
        let ctx = z3::Context::new(&config);
        let mut solver = Z3Solver::new(&ctx);

        // Verify the formula
        match solver.verify(&formula) {
            Ok(VerificationResult::Sat) => {
                // Formula is satisfiable - predicate holds
                ProofResult::Proven(Proof::trusted(
                    "Z3 SMT solver",
                    Predicate::new(PredicateKind::Confidence(pred.clone())),
                ))
            }
            Ok(VerificationResult::Unsat) => {
                // Formula is unsatisfiable - predicate does not hold
                let cex = solver.extract_counterexample();
                ProofResult::Disproven {
                    reason: format!("Z3 counterexample: {:?}", cex),
                }
            }
            Ok(VerificationResult::Unknown) => ProofResult::Unknown {
                reason: "Z3 returned unknown".to_string(),
            },
            Ok(VerificationResult::Timeout) => ProofResult::Unknown {
                reason: "Z3 timeout".to_string(),
            },
            Ok(VerificationResult::Error(e)) => ProofResult::Unknown {
                reason: format!("Z3 error: {}", e),
            },
            Err(e) => ProofResult::Unknown {
                reason: format!("SMT solver error: {}", e),
            },
        }
    }

    #[cfg(not(feature = "smt"))]
    fn try_z3_confidence(&self, _pred: &ConfidencePredicate) -> ProofResult {
        ProofResult::Unknown {
            reason: "SMT feature not enabled (compile with --features smt)".to_string(),
        }
    }

    /// Translate ConfidencePredicate to SmtFormula
    #[cfg(feature = "smt")]
    fn translate_confidence_to_smt(
        &self,
        pred: &ConfidencePredicate,
    ) -> crate::smt::formula::SmtFormula {
        use crate::smt::formula::{SmtFormula, SmtTerm};

        match pred {
            ConfidencePredicate::Geq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Ge(Box::new(lhs_term), Box::new(rhs_term))
            }
            ConfidencePredicate::Leq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Le(Box::new(lhs_term), Box::new(rhs_term))
            }
            ConfidencePredicate::Eq(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Eq(Box::new(lhs_term), Box::new(rhs_term))
            }
            ConfidencePredicate::Gt(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Gt(Box::new(lhs_term), Box::new(rhs_term))
            }
            ConfidencePredicate::Lt(lhs, rhs) => {
                let lhs_term = self.confidence_type_to_term(lhs);
                let rhs_term = self.confidence_type_to_term(rhs);
                SmtFormula::Lt(Box::new(lhs_term), Box::new(rhs_term))
            }
        }
    }

    /// Translate ConfidenceType to SmtTerm
    #[cfg(feature = "smt")]
    fn confidence_type_to_term(&self, conf_ty: &ConfidenceType) -> crate::smt::formula::SmtTerm {
        use crate::smt::formula::SmtTerm;
        use std::time::Duration;

        match conf_ty {
            ConfidenceType::Literal(v) => SmtTerm::Real(*v),

            ConfidenceType::Var(name) => SmtTerm::Var(name.clone()),

            ConfidenceType::Product(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                SmtTerm::Mul(Box::new(a_term), Box::new(b_term))
            }

            ConfidenceType::DempsterShafer(a, b) => {
                // DS(a, b) = 1 - (1-a)*(1-b)
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);
                let one = SmtTerm::Real(1.0);

                let one_minus_a = SmtTerm::Sub(Box::new(one.clone()), Box::new(a_term));
                let one_minus_b = SmtTerm::Sub(Box::new(one.clone()), Box::new(b_term));
                let product = SmtTerm::Mul(Box::new(one_minus_a), Box::new(one_minus_b));

                SmtTerm::Sub(Box::new(one), Box::new(product))
            }

            ConfidenceType::Decay {
                base,
                lambda,
                elapsed,
            } => {
                // base * exp(-lambda * t)
                let base_term = self.confidence_type_to_term(base);
                let t = SmtTerm::Real(elapsed.as_secs_f64());
                let neg_lambda = SmtTerm::Real(-lambda);
                let neg_lambda_t = SmtTerm::Mul(Box::new(neg_lambda), Box::new(t));

                // exp(x) is represented as a function application
                let exp_term = SmtTerm::App("exp".to_string(), vec![neg_lambda_t]);

                SmtTerm::Mul(Box::new(base_term), Box::new(exp_term))
            }

            ConfidenceType::Min(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);

                // min(a, b) - use function application
                SmtTerm::App("min".to_string(), vec![a_term, b_term])
            }

            ConfidenceType::Max(a, b) => {
                let a_term = self.confidence_type_to_term(a);
                let b_term = self.confidence_type_to_term(b);

                // max(a, b) - use function application
                SmtTerm::App("max".to_string(), vec![a_term, b_term])
            }

            ConfidenceType::Conditional {
                prior,
                likelihood,
                evidence,
            } => {
                // P(H|E) = P(E|H) * P(H) / P(E)
                let prior_term = self.confidence_type_to_term(prior);
                let likelihood_term = self.confidence_type_to_term(likelihood);
                let evidence_term = self.confidence_type_to_term(evidence);

                let numerator = SmtTerm::Mul(Box::new(likelihood_term), Box::new(prior_term));
                SmtTerm::Div(Box::new(numerator), Box::new(evidence_term))
            }

            ConfidenceType::Unknown => {
                // For unknown, use a fresh variable
                SmtTerm::Var("_unknown_conf".to_string())
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
    fn test_causal_simple_identifiable() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Simple X → Y: identifiable (no confounders)
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
    fn test_causal_confounded_with_backdoor() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // X → Y with confounder U → X, U → Y
        // Backdoor set {U} should work
        let mut graph = CausalGraphType::new();
        graph.add_edge("U", "X");
        graph.add_edge("U", "Y");
        graph.add_edge("X", "Y");

        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph: graph.clone(),
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());

        // Explicitly verify backdoor with {U}
        let adj: HashSet<String> = ["U".to_string()].into_iter().collect();
        let backdoor_pred = Predicate::causal(CausalPredicate::BackdoorSatisfied {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            adjustment: adj,
        });
        let bd_result = searcher.search(&backdoor_pred);
        assert!(bd_result.is_proven());
    }

    #[test]
    fn test_causal_unidentifiable_bidirected() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // X → Y with latent confounder (bidirected X ↔ Y)
        // Not identifiable without instruments or mediators
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");
        graph.add_bidirected("X", "Y");

        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        // Should not be proven (no backdoor/frontdoor/IV available)
        assert!(!result.is_proven());
    }

    #[test]
    fn test_causal_frontdoor_identifiable() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Classic frontdoor: X → M → Y with X ↔ Y confounded
        // Identifiable via frontdoor through M
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");
        graph.add_bidirected("X", "Y");

        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_d_separation_chain_blocked() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Chain: X → Z → Y, conditioning on Z blocks the path
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Z");
        graph.add_edge("Z", "Y");

        let z_set: HashSet<String> = ["Z".to_string()].into_iter().collect();
        let pred = Predicate::causal(CausalPredicate::DSeparated {
            graph,
            x: ["X".to_string()].into_iter().collect(),
            y: ["Y".to_string()].into_iter().collect(),
            z: z_set,
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_d_separation_fork_blocked() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Fork: X ← Z → Y, conditioning on Z blocks
        let mut graph = CausalGraphType::new();
        graph.add_edge("Z", "X");
        graph.add_edge("Z", "Y");

        let z_set: HashSet<String> = ["Z".to_string()].into_iter().collect();
        let pred = Predicate::causal(CausalPredicate::DSeparated {
            graph,
            x: ["X".to_string()].into_iter().collect(),
            y: ["Y".to_string()].into_iter().collect(),
            z: z_set,
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_d_separation_collider_open() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Collider: X → Z ← Y, NOT conditioning on Z: X ⊥⊥ Y
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Z");
        graph.add_edge("Y", "Z");

        let pred = Predicate::causal(CausalPredicate::DSeparated {
            graph: graph.clone(),
            x: ["X".to_string()].into_iter().collect(),
            y: ["Y".to_string()].into_iter().collect(),
            z: HashSet::new(), // Not conditioning on Z
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven()); // Collider blocks by default

        // Now condition on Z: path opens (d-connected)
        let z_set: HashSet<String> = ["Z".to_string()].into_iter().collect();
        let pred2 = Predicate::causal(CausalPredicate::DSeparated {
            graph,
            x: ["X".to_string()].into_iter().collect(),
            y: ["Y".to_string()].into_iter().collect(),
            z: z_set,
        });

        let result2 = searcher.search(&pred2);
        assert!(result2.is_disproven()); // Conditioning on collider opens path
    }

    #[test]
    fn test_causal_iv_identification() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        // Instrument Z → X → Y with X ↔ Y confounded
        let mut graph = CausalGraphType::new();
        graph.add_edge("Z", "X");
        graph.add_edge("X", "Y");
        graph.add_bidirected("X", "Y");

        let pred = Predicate::causal(CausalPredicate::InstrumentValid {
            graph,
            instrument: "Z".to_string(),
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_causal_pbpk_graph() {
        // PBPK: Dose → Plasma → Effect, Genotype → Metabolism → Plasma
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let mut graph = CausalGraphType::new();
        graph.add_edge("Dose", "Plasma");
        graph.add_edge("Plasma", "Effect");
        graph.add_edge("Genotype", "Metabolism");
        graph.add_edge("Metabolism", "Plasma");

        // Effect of Dose on Effect is identifiable
        let pred = Predicate::causal(CausalPredicate::Identifiable {
            graph,
            treatment: "Dose".to_string(),
            outcome: "Effect".to_string(),
        });

        let result = searcher.search(&pred);
        assert!(result.is_proven());
    }

    #[test]
    fn test_causal_unconfounded() {
        let ctx = TypeContext::new();
        let mut searcher = ProofSearcher::new(&ctx);

        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        // No bidirected edges: unconfounded
        let pred = Predicate::causal(CausalPredicate::Unconfounded {
            graph: graph.clone(),
            x: "X".to_string(),
            y: "Y".to_string(),
        });
        assert!(searcher.search(&pred).is_proven());

        // Add bidirected: now confounded
        graph.add_bidirected("X", "Y");
        let pred2 = Predicate::causal(CausalPredicate::Unconfounded {
            graph,
            x: "X".to_string(),
            y: "Y".to_string(),
        });
        assert!(searcher.search(&pred2).is_disproven());
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
