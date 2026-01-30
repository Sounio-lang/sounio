//! Predicates for refinement types
//!
//! This module defines the predicates that can appear in refinement types
//! and dependent type constraints:
//!
//! - Confidence predicates: ε₁ ≥ ε₂, ε₁ ≤ ε₂, etc.
//! - Ontology predicates: δ₁ ⊇ δ₂, δ₁ ⊆ δ₂, etc.
//! - Causal predicates: identifiable(G, X, Y), d_separated(G, X, Y, Z)
//! - Temporal predicates: fresh(K, Δt), decay_bounded(K, ε_min)
//! - Logical connectives: ∧, ∨, ¬, →, ∀, ∃

use super::types::{CausalGraphType, ConfidenceType, OntologyType, TemporalType};
use std::collections::HashSet;
use std::sync::Arc;

/// A predicate that can appear in refinement types
///
/// # Examples
///
/// ```rust,ignore
/// // Confidence constraint
/// let p1 = Predicate::confidence_geq(
///     ConfidenceType::var("ε"),
///     ConfidenceType::literal(0.95)
/// );
///
/// // Ontology constraint
/// let p2 = Predicate::ontology_superset(
///     OntologyType::var("δ"),
///     OntologyType::concrete("PKPD")
/// );
///
/// // Causal identifiability
/// let p3 = Predicate::causal(CausalPredicate::Identifiable {
///     graph: graph_type,
///     treatment: "Drug".to_string(),
///     outcome: "Effect".to_string(),
/// });
///
/// // Conjunction
/// let p4 = Predicate::and(p1, p2);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Predicate {
    /// The kind of predicate
    pub kind: PredicateKind,
}

impl Predicate {
    /// Create a new predicate
    pub fn new(kind: PredicateKind) -> Self {
        Self { kind }
    }

    /// Create a trivially true predicate
    pub fn true_() -> Self {
        Self::new(PredicateKind::True)
    }

    /// Create a trivially false predicate
    pub fn false_() -> Self {
        Self::new(PredicateKind::False)
    }

    /// Create a confidence predicate: ε₁ ≥ ε₂
    pub fn confidence_geq(lhs: ConfidenceType, rhs: ConfidenceType) -> Self {
        Self::new(PredicateKind::Confidence(ConfidencePredicate::Geq(
            lhs, rhs,
        )))
    }

    /// Create a confidence predicate: ε₁ ≤ ε₂
    pub fn confidence_leq(lhs: ConfidenceType, rhs: ConfidenceType) -> Self {
        Self::new(PredicateKind::Confidence(ConfidencePredicate::Leq(
            lhs, rhs,
        )))
    }

    /// Create a confidence predicate: ε₁ = ε₂
    pub fn confidence_eq(lhs: ConfidenceType, rhs: ConfidenceType) -> Self {
        Self::new(PredicateKind::Confidence(ConfidencePredicate::Eq(lhs, rhs)))
    }

    /// Create an ontology predicate: δ₁ ⊇ δ₂
    pub fn ontology_superset(lhs: OntologyType, rhs: OntologyType) -> Self {
        Self::new(PredicateKind::Ontology(OntologyPredicate::Superset(
            lhs, rhs,
        )))
    }

    /// Create an ontology predicate: δ₁ ⊆ δ₂
    pub fn ontology_subset(lhs: OntologyType, rhs: OntologyType) -> Self {
        Self::new(PredicateKind::Ontology(OntologyPredicate::Subset(lhs, rhs)))
    }

    /// Create an ontology predicate: δ₁ ∩ δ₂ ≠ ∅
    pub fn ontology_overlaps(lhs: OntologyType, rhs: OntologyType) -> Self {
        Self::new(PredicateKind::Ontology(OntologyPredicate::Overlaps(
            lhs, rhs,
        )))
    }

    /// Create a causal predicate
    pub fn causal(pred: CausalPredicate) -> Self {
        Self::new(PredicateKind::Causal(pred))
    }

    /// Create a temporal predicate
    pub fn temporal(pred: TemporalPredicate) -> Self {
        Self::new(PredicateKind::Temporal(pred))
    }

    /// Create conjunction: P ∧ Q
    pub fn and(p: Predicate, q: Predicate) -> Self {
        Self::new(PredicateKind::And(Arc::new(p), Arc::new(q)))
    }

    /// Create disjunction: P ∨ Q
    pub fn or(p: Predicate, q: Predicate) -> Self {
        Self::new(PredicateKind::Or(Arc::new(p), Arc::new(q)))
    }

    /// Create negation: ¬P
    pub fn not(p: Predicate) -> Self {
        Self::new(PredicateKind::Not(Arc::new(p)))
    }

    /// Create implication: P → Q
    pub fn implies(p: Predicate, q: Predicate) -> Self {
        Self::new(PredicateKind::Implies(Arc::new(p), Arc::new(q)))
    }

    /// Create universal quantification: ∀x:τ. P(x)
    pub fn forall(var: impl Into<String>, ty: crate::types::Type, body: Predicate) -> Self {
        Self::new(PredicateKind::Forall {
            var: var.into(),
            ty: Arc::new(ty),
            body: Arc::new(body),
        })
    }

    /// Create existential quantification: ∃x:τ. P(x)
    pub fn exists(var: impl Into<String>, ty: crate::types::Type, body: Predicate) -> Self {
        Self::new(PredicateKind::Exists {
            var: var.into(),
            ty: Arc::new(ty),
            body: Arc::new(body),
        })
    }

    /// Check if this predicate is trivially true
    pub fn is_trivially_true(&self) -> bool {
        matches!(self.kind, PredicateKind::True)
    }

    /// Check if this predicate is trivially false
    pub fn is_trivially_false(&self) -> bool {
        matches!(self.kind, PredicateKind::False)
    }

    /// Get free variables in this predicate
    pub fn free_vars(&self) -> HashSet<String> {
        match &self.kind {
            PredicateKind::True | PredicateKind::False => HashSet::new(),
            PredicateKind::Confidence(cp) => cp.free_vars(),
            PredicateKind::Ontology(op) => op.free_vars(),
            PredicateKind::Causal(cp) => cp.free_vars(),
            PredicateKind::Temporal(tp) => tp.free_vars(),
            PredicateKind::And(p, q) | PredicateKind::Or(p, q) | PredicateKind::Implies(p, q) => {
                let mut vars = p.free_vars();
                vars.extend(q.free_vars());
                vars
            }
            PredicateKind::Not(p) => p.free_vars(),
            PredicateKind::Forall { var, body, .. } | PredicateKind::Exists { var, body, .. } => {
                let mut vars = body.free_vars();
                vars.remove(var);
                vars
            }
        }
    }

    /// Normalize the predicate (simplify)
    pub fn normalize(&self) -> Self {
        match &self.kind {
            // Double negation elimination
            PredicateKind::Not(inner) => {
                if let PredicateKind::Not(p) = &inner.kind {
                    return p.normalize();
                }
                Self::not(inner.normalize())
            }
            // And simplification
            PredicateKind::And(p, q) => {
                let np = p.normalize();
                let nq = q.normalize();
                if np.is_trivially_true() {
                    return nq;
                }
                if nq.is_trivially_true() {
                    return np;
                }
                if np.is_trivially_false() || nq.is_trivially_false() {
                    return Self::false_();
                }
                Self::and(np, nq)
            }
            // Or simplification
            PredicateKind::Or(p, q) => {
                let np = p.normalize();
                let nq = q.normalize();
                if np.is_trivially_false() {
                    return nq;
                }
                if nq.is_trivially_false() {
                    return np;
                }
                if np.is_trivially_true() || nq.is_trivially_true() {
                    return Self::true_();
                }
                Self::or(np, nq)
            }
            // Implies to Or
            PredicateKind::Implies(p, q) => {
                let np = p.normalize();
                let nq = q.normalize();
                // P → Q ≡ ¬P ∨ Q
                Self::or(Self::not(np), nq).normalize()
            }
            // Keep others as-is
            _ => self.clone(),
        }
    }

    /// Substitute a variable with a confidence type
    pub fn substitute_confidence(&self, var: &str, value: &ConfidenceType) -> Self {
        match &self.kind {
            PredicateKind::Confidence(cp) => {
                Self::new(PredicateKind::Confidence(cp.substitute(var, value)))
            }
            PredicateKind::And(p, q) => Self::and(
                p.substitute_confidence(var, value),
                q.substitute_confidence(var, value),
            ),
            PredicateKind::Or(p, q) => Self::or(
                p.substitute_confidence(var, value),
                q.substitute_confidence(var, value),
            ),
            PredicateKind::Not(p) => Self::not(p.substitute_confidence(var, value)),
            PredicateKind::Implies(p, q) => Self::implies(
                p.substitute_confidence(var, value),
                q.substitute_confidence(var, value),
            ),
            PredicateKind::Forall { var: v, ty, body } if v != var => Self::forall(
                v.clone(),
                (**ty).clone(),
                body.substitute_confidence(var, value),
            ),
            PredicateKind::Exists { var: v, ty, body } if v != var => Self::exists(
                v.clone(),
                (**ty).clone(),
                body.substitute_confidence(var, value),
            ),
            _ => self.clone(),
        }
    }
}

impl std::fmt::Display for Predicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            PredicateKind::True => write!(f, "⊤"),
            PredicateKind::False => write!(f, "⊥"),
            PredicateKind::Confidence(cp) => write!(f, "{}", cp),
            PredicateKind::Ontology(op) => write!(f, "{}", op),
            PredicateKind::Causal(cp) => write!(f, "{}", cp),
            PredicateKind::Temporal(tp) => write!(f, "{}", tp),
            PredicateKind::And(p, q) => write!(f, "({} ∧ {})", p, q),
            PredicateKind::Or(p, q) => write!(f, "({} ∨ {})", p, q),
            PredicateKind::Not(p) => write!(f, "¬{}", p),
            PredicateKind::Implies(p, q) => write!(f, "({} → {})", p, q),
            PredicateKind::Forall { var, ty, body } => write!(f, "∀{}:{:?}. {}", var, ty, body),
            PredicateKind::Exists { var, ty, body } => write!(f, "∃{}:{:?}. {}", var, ty, body),
        }
    }
}

/// The kind of predicate
#[derive(Clone, Debug, PartialEq)]
pub enum PredicateKind {
    /// Trivially true
    True,
    /// Trivially false
    False,
    /// Confidence predicate
    Confidence(ConfidencePredicate),
    /// Ontology predicate
    Ontology(OntologyPredicate),
    /// Causal predicate
    Causal(CausalPredicate),
    /// Temporal predicate
    Temporal(TemporalPredicate),
    /// Conjunction: P ∧ Q
    And(Arc<Predicate>, Arc<Predicate>),
    /// Disjunction: P ∨ Q
    Or(Arc<Predicate>, Arc<Predicate>),
    /// Negation: ¬P
    Not(Arc<Predicate>),
    /// Implication: P → Q
    Implies(Arc<Predicate>, Arc<Predicate>),
    /// Universal quantification: ∀x:τ. P(x)
    Forall {
        var: String,
        ty: Arc<crate::types::Type>,
        body: Arc<Predicate>,
    },
    /// Existential quantification: ∃x:τ. P(x)
    Exists {
        var: String,
        ty: Arc<crate::types::Type>,
        body: Arc<Predicate>,
    },
}

/// Confidence predicates
#[derive(Clone, Debug, PartialEq)]
pub enum ConfidencePredicate {
    /// ε₁ ≥ ε₂
    Geq(ConfidenceType, ConfidenceType),
    /// ε₁ ≤ ε₂
    Leq(ConfidenceType, ConfidenceType),
    /// ε₁ = ε₂
    Eq(ConfidenceType, ConfidenceType),
    /// ε₁ > ε₂
    Gt(ConfidenceType, ConfidenceType),
    /// ε₁ < ε₂
    Lt(ConfidenceType, ConfidenceType),
}

impl ConfidencePredicate {
    /// Get free variables
    pub fn free_vars(&self) -> HashSet<String> {
        match self {
            Self::Geq(a, b)
            | Self::Leq(a, b)
            | Self::Eq(a, b)
            | Self::Gt(a, b)
            | Self::Lt(a, b) => {
                let mut vars = a.free_vars();
                vars.extend(b.free_vars());
                vars
            }
        }
    }

    /// Substitute a variable
    pub fn substitute(&self, var: &str, value: &ConfidenceType) -> Self {
        match self {
            Self::Geq(a, b) => Self::Geq(a.substitute(var, value), b.substitute(var, value)),
            Self::Leq(a, b) => Self::Leq(a.substitute(var, value), b.substitute(var, value)),
            Self::Eq(a, b) => Self::Eq(a.substitute(var, value), b.substitute(var, value)),
            Self::Gt(a, b) => Self::Gt(a.substitute(var, value), b.substitute(var, value)),
            Self::Lt(a, b) => Self::Lt(a.substitute(var, value), b.substitute(var, value)),
        }
    }

    /// Evaluate if possible
    pub fn evaluate(&self, ctx: &super::TypeContext) -> Option<bool> {
        match self {
            Self::Geq(a, b) => Some(a.evaluate(ctx)? >= b.evaluate(ctx)?),
            Self::Leq(a, b) => Some(a.evaluate(ctx)? <= b.evaluate(ctx)?),
            Self::Eq(a, b) => Some((a.evaluate(ctx)? - b.evaluate(ctx)?).abs() < 1e-10),
            Self::Gt(a, b) => Some(a.evaluate(ctx)? > b.evaluate(ctx)?),
            Self::Lt(a, b) => Some(a.evaluate(ctx)? < b.evaluate(ctx)?),
        }
    }
}

impl std::fmt::Display for ConfidencePredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Geq(a, b) => write!(f, "{} ≥ {}", a, b),
            Self::Leq(a, b) => write!(f, "{} ≤ {}", a, b),
            Self::Eq(a, b) => write!(f, "{} = {}", a, b),
            Self::Gt(a, b) => write!(f, "{} > {}", a, b),
            Self::Lt(a, b) => write!(f, "{} < {}", a, b),
        }
    }
}

/// Ontology predicates
#[derive(Clone, Debug, PartialEq)]
pub enum OntologyPredicate {
    /// δ₁ ⊇ δ₂ (superset)
    Superset(OntologyType, OntologyType),
    /// δ₁ ⊆ δ₂ (subset)
    Subset(OntologyType, OntologyType),
    /// δ₁ = δ₂ (equal)
    Eq(OntologyType, OntologyType),
    /// δ₁ ∩ δ₂ ≠ ∅ (overlap)
    Overlaps(OntologyType, OntologyType),
    /// δ₁ ∩ δ₂ = ∅ (disjoint)
    Disjoint(OntologyType, OntologyType),
}

impl OntologyPredicate {
    /// Get free variables (ontology variables are different from confidence vars)
    pub fn free_vars(&self) -> HashSet<String> {
        // Ontology types with Var contain variables
        HashSet::new() // Simplified - would need to traverse OntologyType
    }

    /// Evaluate if possible
    pub fn evaluate(&self, _ctx: &super::TypeContext) -> Option<bool> {
        match self {
            Self::Superset(a, b) => Some(a.contains(b)),
            Self::Subset(a, b) => Some(b.contains(a)),
            Self::Eq(a, b) => Some(a.definitionally_equal(b)),
            Self::Overlaps(a, b) => {
                let set_a = a.to_set();
                let set_b = b.to_set();
                Some(!set_a.is_disjoint(&set_b))
            }
            Self::Disjoint(a, b) => {
                let set_a = a.to_set();
                let set_b = b.to_set();
                Some(set_a.is_disjoint(&set_b))
            }
        }
    }
}

impl std::fmt::Display for OntologyPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Superset(a, b) => write!(f, "{} ⊇ {}", a, b),
            Self::Subset(a, b) => write!(f, "{} ⊆ {}", a, b),
            Self::Eq(a, b) => write!(f, "{} = {}", a, b),
            Self::Overlaps(a, b) => write!(f, "{} ∩ {} ≠ ∅", a, b),
            Self::Disjoint(a, b) => write!(f, "{} ∩ {} = ∅", a, b),
        }
    }
}

/// Causal predicates for identifiability and d-separation
#[derive(Clone, Debug, PartialEq)]
pub enum CausalPredicate {
    /// Effect is identifiable from data
    Identifiable {
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
    },

    /// d-separation: X ⊥⊥ Y | Z in G
    DSeparated {
        graph: CausalGraphType,
        x: HashSet<String>,
        y: HashSet<String>,
        z: HashSet<String>,
    },

    /// Backdoor criterion is satisfied
    BackdoorSatisfied {
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        adjustment: HashSet<String>,
    },

    /// Frontdoor criterion is satisfied
    FrontdoorSatisfied {
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        mediators: HashSet<String>,
    },

    /// Instrumental variable is valid
    InstrumentValid {
        graph: CausalGraphType,
        instrument: String,
        treatment: String,
        outcome: String,
    },

    /// No unobserved confounders between X and Y
    Unconfounded {
        graph: CausalGraphType,
        x: String,
        y: String,
    },
}

impl CausalPredicate {
    /// Get free variables
    pub fn free_vars(&self) -> HashSet<String> {
        HashSet::new() // Causal predicates don't have free vars in our system
    }

    /// Check if backdoor criterion is satisfied
    pub fn check_backdoor(
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
        adjustment: &HashSet<String>,
    ) -> bool {
        // 1. Z does not contain any descendant of X
        let x_descendants = graph.descendants(treatment);
        if adjustment.iter().any(|z| x_descendants.contains(z)) {
            return false;
        }

        // 2. Z blocks all backdoor paths from X to Y
        // A backdoor path is a path that starts with an edge INTO X
        // For simplicity, we check that after removing X→Y edges,
        // Z d-separates X from Y

        // This is a simplified check - full d-separation would need
        // to be implemented properly
        let g_x = graph.remove_outgoing(treatment);
        Self::check_d_separation(&g_x, treatment, outcome, adjustment)
    }

    /// Check if frontdoor criterion is satisfied
    pub fn check_frontdoor(
        graph: &CausalGraphType,
        treatment: &str,
        outcome: &str,
        mediators: &HashSet<String>,
    ) -> bool {
        // 1. M intercepts all directed paths from X to Y
        let x_children = graph.children(treatment);
        if !mediators.iter().all(|m| x_children.contains(m)) {
            // Mediators must be direct children of treatment
            // (simplified check)
        }

        // 2. No unblocked backdoor path from X to M
        // 3. All backdoor paths from M to Y are blocked by X

        // Simplified: just check that M is between X and Y
        for m in mediators {
            if !graph.has_directed_path(treatment, m) {
                return false;
            }
            if !graph.has_directed_path(m, outcome) {
                return false;
            }
        }

        true
    }

    /// Simple d-separation check (Bayes-Ball simplified)
    fn check_d_separation(graph: &CausalGraphType, x: &str, y: &str, z: &HashSet<String>) -> bool {
        // If Z contains X or Y, trivially separated
        if z.contains(x) || z.contains(y) {
            return true;
        }

        // Simplified: check if all paths are blocked
        // A proper implementation would use Bayes-Ball algorithm
        // For now, we just check direct connectivity

        // If there's a direct edge and no intervention, not separated
        if graph.edges.contains(&(x.to_string(), y.to_string())) {
            return false;
        }
        if graph.edges.contains(&(y.to_string(), x.to_string())) {
            return false;
        }

        // Check through common ancestors (simplified)
        let x_ancestors = graph.ancestors(x);
        let y_ancestors = graph.ancestors(y);
        let common: HashSet<_> = x_ancestors.intersection(&y_ancestors).collect();

        // If common ancestors exist and none are in Z, might not be separated
        if !common.is_empty() && !common.iter().any(|a| z.contains(*a)) {
            // This is a very simplified check
            return false;
        }

        true
    }
}

impl std::fmt::Display for CausalPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Identifiable {
                treatment, outcome, ..
            } => {
                write!(f, "identifiable({} → {})", treatment, outcome)
            }
            Self::DSeparated { x, y, z, .. } => {
                write!(f, "{:?} ⊥⊥ {:?} | {:?}", x, y, z)
            }
            Self::BackdoorSatisfied {
                treatment,
                outcome,
                adjustment,
                ..
            } => {
                write!(
                    f,
                    "backdoor({} → {} | {:?})",
                    treatment, outcome, adjustment
                )
            }
            Self::FrontdoorSatisfied {
                treatment,
                outcome,
                mediators,
                ..
            } => {
                write!(
                    f,
                    "frontdoor({} → {} via {:?})",
                    treatment, outcome, mediators
                )
            }
            Self::InstrumentValid {
                instrument,
                treatment,
                outcome,
                ..
            } => {
                write!(f, "IV({} → {} → {})", instrument, treatment, outcome)
            }
            Self::Unconfounded { x, y, .. } => {
                write!(f, "unconfounded({}, {})", x, y)
            }
        }
    }
}

/// Temporal predicates
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalPredicate {
    /// Knowledge is fresh (created within Δt)
    Fresh {
        temporal: TemporalType,
        max_age_secs: i64,
    },

    /// Decay is bounded (ε after decay ≥ ε_min)
    DecayBounded {
        base_confidence: ConfidenceType,
        lambda: f64,
        max_time_secs: f64,
        min_confidence: f64,
    },

    /// Knowledge precedes another temporally
    Precedes {
        earlier: TemporalType,
        later: TemporalType,
    },

    /// Eventually (LTL: ◇P)
    Eventually(Arc<Predicate>),

    /// Always (LTL: □P)
    Always(Arc<Predicate>),

    /// Until (LTL: P U Q)
    Until(Arc<Predicate>, Arc<Predicate>),

    /// Since (LTL: P S Q)
    Since(Arc<Predicate>, Arc<Predicate>),
}

impl TemporalPredicate {
    /// Get free variables
    pub fn free_vars(&self) -> HashSet<String> {
        match self {
            Self::Fresh { .. } => HashSet::new(),
            Self::DecayBounded {
                base_confidence, ..
            } => base_confidence.free_vars(),
            Self::Precedes { .. } => HashSet::new(),
            Self::Eventually(p) | Self::Always(p) => p.free_vars(),
            Self::Until(p, q) | Self::Since(p, q) => {
                let mut vars = p.free_vars();
                vars.extend(q.free_vars());
                vars
            }
        }
    }

    /// Evaluate freshness predicate
    pub fn evaluate_fresh(
        temporal: &TemporalType,
        max_age_secs: i64,
        current_time: i64,
    ) -> Option<bool> {
        temporal.is_fresh(max_age_secs, current_time)
    }

    /// Evaluate decay bound
    pub fn evaluate_decay_bound(base: f64, lambda: f64, max_time: f64, min_conf: f64) -> bool {
        let decayed = base * (-lambda * max_time).exp();
        decayed >= min_conf
    }
}

impl std::fmt::Display for TemporalPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fresh {
                temporal,
                max_age_secs,
            } => {
                write!(f, "fresh({}, {}s)", temporal, max_age_secs)
            }
            Self::DecayBounded {
                base_confidence,
                lambda,
                max_time_secs,
                min_confidence,
            } => {
                write!(
                    f,
                    "decay({}, λ={}, t≤{}) ≥ {}",
                    base_confidence, lambda, max_time_secs, min_confidence
                )
            }
            Self::Precedes { earlier, later } => {
                write!(f, "{} ≺ {}", earlier, later)
            }
            Self::Eventually(p) => write!(f, "◇{}", p),
            Self::Always(p) => write!(f, "□{}", p),
            Self::Until(p, q) => write!(f, "({} U {})", p, q),
            Self::Since(p, q) => write!(f, "({} S {})", p, q),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_predicate_true() {
        let p = Predicate::true_();
        assert!(p.is_trivially_true());
        assert!(!p.is_trivially_false());
    }

    #[test]
    fn test_confidence_predicate_eval() {
        let ctx = super::super::TypeContext::new();
        let pred =
            ConfidencePredicate::Geq(ConfidenceType::literal(0.95), ConfidenceType::literal(0.90));
        assert_eq!(pred.evaluate(&ctx), Some(true));

        let pred2 =
            ConfidencePredicate::Geq(ConfidenceType::literal(0.80), ConfidenceType::literal(0.90));
        assert_eq!(pred2.evaluate(&ctx), Some(false));
    }

    #[test]
    fn test_predicate_normalize_double_negation() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let double_neg = Predicate::not(Predicate::not(p.clone()));
        let normalized = double_neg.normalize();
        assert_eq!(normalized, p);
    }

    #[test]
    fn test_predicate_normalize_and_true() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let with_true = Predicate::and(p.clone(), Predicate::true_());
        let normalized = with_true.normalize();
        assert_eq!(normalized, p);
    }

    #[test]
    fn test_predicate_and_false() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let with_false = Predicate::and(p, Predicate::false_());
        let normalized = with_false.normalize();
        assert!(normalized.is_trivially_false());
    }

    #[test]
    fn test_ontology_predicate() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");
        let union = OntologyType::union(pkpd.clone(), chebi.clone());

        let pred = OntologyPredicate::Superset(union, pkpd);
        let ctx = super::super::TypeContext::new();
        assert_eq!(pred.evaluate(&ctx), Some(true));
    }

    #[test]
    fn test_causal_predicate_display() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let pred = CausalPredicate::Identifiable {
            graph,
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
        };
        let s = format!("{}", pred);
        assert!(s.contains("identifiable"));
    }

    #[test]
    fn test_free_vars() {
        let p = Predicate::and(
            Predicate::confidence_geq(ConfidenceType::var("α"), ConfidenceType::literal(0.9)),
            Predicate::confidence_geq(ConfidenceType::var("β"), ConfidenceType::var("α")),
        );
        let vars = p.free_vars();
        assert!(vars.contains("α"));
        assert!(vars.contains("β"));
    }

    #[test]
    fn test_substitution() {
        let p = Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let substituted = p.substitute_confidence("ε", &ConfidenceType::literal(0.97));

        if let PredicateKind::Confidence(ConfidencePredicate::Geq(lhs, _)) = &substituted.kind {
            assert!(matches!(lhs, ConfidenceType::Literal(v) if (*v - 0.97).abs() < 0.001));
        } else {
            panic!("Expected confidence predicate");
        }
    }
}
