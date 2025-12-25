//! Proof terms for dependent types
//!
//! This module implements proof terms that serve as evidence for
//! type-level claims. Under the Curry-Howard correspondence:
//!
//! - Propositions are types
//! - Proofs are values
//! - If we have a value of type `Proof[P]`, then P is proven
//!
//! # Proof Kinds
//!
//! - **Reflexivity**: `refl` proves `a = a`
//! - **Literal comparison**: `lit_cmp(0.95, 0.90)` proves `0.95 ≥ 0.90`
//! - **Transitivity**: `trans(p1, p2)` from `a ≥ b` and `b ≥ c` proves `a ≥ c`
//! - **Arithmetic**: `arith(...)` proves arithmetic facts
//! - **Causal**: `backdoor_check(...)` proves identifiability
//! - **Runtime**: `runtime_check(P)` defers to runtime (gradual typing)
//! - **Trusted**: `trusted(reason)` escape hatch (requires annotation)

use super::predicates::{CausalPredicate, Predicate, PredicateKind};
use super::types::{CausalGraphType, ConfidenceType};
use std::collections::HashSet;
use std::sync::Arc;

/// A proof term that witnesses a predicate
///
/// Under Curry-Howard, this is the "program" that proves a "proposition".
#[derive(Clone, Debug)]
pub struct Proof {
    /// The kind of proof
    pub kind: ProofKind,
    /// The predicate this proof witnesses
    pub witnesses: Predicate,
}

impl Proof {
    /// Create a new proof
    pub fn new(kind: ProofKind, witnesses: Predicate) -> Self {
        Self { kind, witnesses }
    }

    /// Create a reflexivity proof
    pub fn refl(conf: ConfidenceType) -> Self {
        let pred = Predicate::confidence_eq(conf.clone(), conf);
        Self::new(ProofKind::Refl, pred)
    }

    /// Create a literal comparison proof
    pub fn literal_cmp(lhs: f64, rhs: f64) -> Option<Self> {
        if lhs >= rhs {
            let pred = Predicate::confidence_geq(
                ConfidenceType::literal(lhs),
                ConfidenceType::literal(rhs),
            );
            Some(Self::new(
                ProofKind::LiteralCmp {
                    lhs,
                    rhs,
                    relation: std::cmp::Ordering::Greater,
                },
                pred,
            ))
        } else {
            None
        }
    }

    /// Create a transitivity proof
    pub fn trans(p1: Proof, p2: Proof) -> Option<Self> {
        // Check that p1 and p2 chain: p1: a ≥ b, p2: b ≥ c → a ≥ c
        // This is a simplified check
        let witnesses = Predicate::true_(); // Would need to compute actual predicate
        Some(Self::new(
            ProofKind::Trans(Arc::new(p1), Arc::new(p2)),
            witnesses,
        ))
    }

    /// Create an arithmetic derivation proof
    pub fn arith(derivation: ArithDerivation, witnesses: Predicate) -> Self {
        Self::new(ProofKind::Arith(derivation), witnesses)
    }

    /// Create a backdoor criterion proof
    pub fn backdoor(
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        adjustment: HashSet<String>,
    ) -> Option<Self> {
        // Verify the backdoor criterion
        if CausalPredicate::check_backdoor(&graph, &treatment, &outcome, &adjustment) {
            let pred = Predicate::causal(CausalPredicate::BackdoorSatisfied {
                graph: graph.clone(),
                treatment: treatment.clone(),
                outcome: outcome.clone(),
                adjustment: adjustment.clone(),
            });
            Some(Self::new(
                ProofKind::Causal(CausalProof::BackdoorCheck {
                    graph,
                    treatment,
                    outcome,
                    adjustment,
                }),
                pred,
            ))
        } else {
            None
        }
    }

    /// Create a frontdoor criterion proof
    pub fn frontdoor(
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        mediators: HashSet<String>,
    ) -> Option<Self> {
        if CausalPredicate::check_frontdoor(&graph, &treatment, &outcome, &mediators) {
            let pred = Predicate::causal(CausalPredicate::FrontdoorSatisfied {
                graph: graph.clone(),
                treatment: treatment.clone(),
                outcome: outcome.clone(),
                mediators: mediators.clone(),
            });
            Some(Self::new(
                ProofKind::Causal(CausalProof::FrontdoorCheck {
                    graph,
                    treatment,
                    outcome,
                    mediators,
                }),
                pred,
            ))
        } else {
            None
        }
    }

    /// Create a runtime check proof (for gradual typing)
    pub fn runtime_check(pred: Predicate) -> Self {
        Self::new(ProofKind::RuntimeCheck(pred.clone()), pred)
    }

    /// Create a trusted proof (escape hatch)
    pub fn trusted(reason: impl Into<String>, pred: Predicate) -> Self {
        Self::new(ProofKind::Trusted(reason.into()), pred)
    }

    /// Create an assumption proof (from context)
    pub fn assume(name: impl Into<String>, pred: Predicate) -> Self {
        Self::new(ProofKind::Assume(name.into()), pred)
    }

    /// Create an and-introduction proof
    pub fn and_intro(p1: Proof, p2: Proof) -> Self {
        let pred = Predicate::and(p1.witnesses.clone(), p2.witnesses.clone());
        Self::new(ProofKind::AndIntro(Arc::new(p1), Arc::new(p2)), pred)
    }

    /// Create an or-introduction (left) proof
    pub fn or_intro_left(p: Proof, q: Predicate) -> Self {
        let pred = Predicate::or(p.witnesses.clone(), q);
        Self::new(ProofKind::OrIntroL(Arc::new(p)), pred)
    }

    /// Create an or-introduction (right) proof
    pub fn or_intro_right(p: Predicate, q: Proof) -> Self {
        let pred = Predicate::or(p, q.witnesses.clone());
        Self::new(ProofKind::OrIntroR(Arc::new(q)), pred)
    }

    /// Create an implication introduction
    pub fn impl_intro(assumption: String, body: Proof, conclusion: Predicate) -> Self {
        Self::new(
            ProofKind::ImplIntro {
                assumption,
                body: Arc::new(body),
            },
            conclusion,
        )
    }

    /// Create an implication elimination (modus ponens)
    pub fn impl_elim(impl_proof: Proof, arg_proof: Proof) -> Self {
        // impl_proof: P → Q, arg_proof: P → proof of Q
        // Extract Q from the implication
        let conclusion = if let PredicateKind::Implies(_, q) = &impl_proof.witnesses.kind {
            (**q).clone()
        } else {
            Predicate::true_()
        };
        Self::new(
            ProofKind::ImplElim(Arc::new(impl_proof), Arc::new(arg_proof)),
            conclusion,
        )
    }

    /// Create a forall introduction
    pub fn forall_intro(var: String, ty: crate::types::Type, body: Proof) -> Self {
        let pred = Predicate::forall(var.clone(), ty.clone(), body.witnesses.clone());
        Self::new(
            ProofKind::ForallIntro {
                var,
                ty: Arc::new(ty),
                body: Arc::new(body),
            },
            pred,
        )
    }

    /// Create a forall elimination
    pub fn forall_elim(forall_proof: Proof, arg: ProofTerm) -> Self {
        // Would substitute var with arg in the body predicate
        let conclusion = Predicate::true_(); // Simplified
        Self::new(
            ProofKind::ForallElim {
                forall_proof: Arc::new(forall_proof),
                arg: Arc::new(arg),
            },
            conclusion,
        )
    }

    /// Create an exists introduction
    pub fn exists_intro(witness: ProofTerm, body: Proof) -> Self {
        // Would construct ∃x. P where witness proves P[witness/x]
        let pred = Predicate::true_(); // Simplified
        Self::new(
            ProofKind::ExistsIntro {
                witness: Arc::new(witness),
                body: Arc::new(body),
            },
            pred,
        )
    }

    /// Check if this is a static proof (no runtime checks)
    pub fn is_static(&self) -> bool {
        match &self.kind {
            ProofKind::RuntimeCheck(_) => false,
            ProofKind::Trusted(_) => false, // Trusted is also not "proven"
            ProofKind::AndIntro(p1, p2) => p1.is_static() && p2.is_static(),
            ProofKind::OrIntroL(p) | ProofKind::OrIntroR(p) => p.is_static(),
            ProofKind::Trans(p1, p2) => p1.is_static() && p2.is_static(),
            ProofKind::ImplIntro { body, .. } => body.is_static(),
            ProofKind::ImplElim(p1, p2) => p1.is_static() && p2.is_static(),
            ProofKind::ForallIntro { body, .. } => body.is_static(),
            ProofKind::ForallElim { forall_proof, .. } => forall_proof.is_static(),
            ProofKind::ExistsIntro { body, .. } => body.is_static(),
            _ => true,
        }
    }

    /// Get a human-readable description of the proof
    pub fn describe(&self) -> String {
        match &self.kind {
            ProofKind::Refl => "reflexivity".to_string(),
            ProofKind::LiteralCmp { lhs, rhs, .. } => {
                format!("literal comparison: {} ≥ {}", lhs, rhs)
            }
            ProofKind::Trans(_, _) => "transitivity".to_string(),
            ProofKind::Arith(deriv) => format!("arithmetic: {}", deriv.expression),
            ProofKind::Causal(cp) => format!("causal: {}", cp.describe()),
            ProofKind::RuntimeCheck(_) => "runtime check (gradual)".to_string(),
            ProofKind::Trusted(reason) => format!("trusted: {}", reason),
            ProofKind::Assume(name) => format!("assumption: {}", name),
            ProofKind::AndIntro(_, _) => "conjunction introduction".to_string(),
            ProofKind::OrIntroL(_) => "disjunction introduction (left)".to_string(),
            ProofKind::OrIntroR(_) => "disjunction introduction (right)".to_string(),
            ProofKind::ImplIntro { .. } => "implication introduction".to_string(),
            ProofKind::ImplElim(_, _) => "modus ponens".to_string(),
            ProofKind::ForallIntro { .. } => "universal introduction".to_string(),
            ProofKind::ForallElim { .. } => "universal elimination".to_string(),
            ProofKind::ExistsIntro { .. } => "existential introduction".to_string(),
            ProofKind::MonoMul(_) => "monotonicity of multiplication".to_string(),
            ProofKind::MonoDS(_) => "monotonicity of Dempster-Shafer".to_string(),
            ProofKind::DecayBound { .. } => "decay bound computation".to_string(),
        }
    }
}

impl std::fmt::Display for Proof {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "⊢ {} by {}", self.witnesses, self.describe())
    }
}

/// The kind of proof term
#[derive(Clone, Debug)]
pub enum ProofKind {
    /// Reflexivity: a = a
    Refl,

    /// Literal comparison: r₁ ≥ r₂ where r₁, r₂ are constants
    LiteralCmp {
        lhs: f64,
        rhs: f64,
        relation: std::cmp::Ordering,
    },

    /// Transitivity: from a ≥ b and b ≥ c, derive a ≥ c
    Trans(Arc<Proof>, Arc<Proof>),

    /// Arithmetic derivation
    Arith(ArithDerivation),

    /// Causal proof (backdoor, frontdoor, etc.)
    Causal(CausalProof),

    /// Runtime check (for gradual typing)
    RuntimeCheck(Predicate),

    /// Trusted assertion (escape hatch)
    Trusted(String),

    /// Assumption from context
    Assume(String),

    /// Conjunction introduction: from P and Q, derive P ∧ Q
    AndIntro(Arc<Proof>, Arc<Proof>),

    /// Disjunction introduction (left): from P, derive P ∨ Q
    OrIntroL(Arc<Proof>),

    /// Disjunction introduction (right): from Q, derive P ∨ Q
    OrIntroR(Arc<Proof>),

    /// Implication introduction: assuming P, prove Q, derive P → Q
    ImplIntro {
        assumption: String,
        body: Arc<Proof>,
    },

    /// Implication elimination (modus ponens): from P → Q and P, derive Q
    ImplElim(Arc<Proof>, Arc<Proof>),

    /// Universal introduction: prove P(x) for arbitrary x, derive ∀x. P(x)
    ForallIntro {
        var: String,
        ty: Arc<crate::types::Type>,
        body: Arc<Proof>,
    },

    /// Universal elimination: from ∀x. P(x) and term t, derive P(t)
    ForallElim {
        forall_proof: Arc<Proof>,
        arg: Arc<ProofTerm>,
    },

    /// Existential introduction: from P(t), derive ∃x. P(x)
    ExistsIntro {
        witness: Arc<ProofTerm>,
        body: Arc<Proof>,
    },

    /// Monotonicity of multiplication: from ε₁ ≥ ε₂, derive ε₁ * ε₃ ≥ ε₂ * ε₃
    MonoMul(Arc<Proof>),

    /// Monotonicity of Dempster-Shafer
    MonoDS(Arc<Proof>),

    /// Decay bound: prove that decay(ε₀, λ, t) ≥ ε_min
    DecayBound {
        base: f64,
        lambda: f64,
        time: f64,
        bound: f64,
    },
}

/// Arithmetic derivation details
#[derive(Clone, Debug)]
pub struct ArithDerivation {
    /// The expression being proven
    pub expression: String,
    /// Steps in the derivation
    pub steps: Vec<ArithStep>,
}

impl ArithDerivation {
    /// Create a new arithmetic derivation
    pub fn new(expression: impl Into<String>) -> Self {
        Self {
            expression: expression.into(),
            steps: vec![],
        }
    }

    /// Add a step to the derivation
    pub fn add_step(&mut self, step: ArithStep) {
        self.steps.push(step);
    }

    /// Create a simple lower-bound derivation
    pub fn lower_bound(value: f64, bound: f64) -> Self {
        let mut deriv = Self::new(format!("{} ≥ {}", value, bound));
        deriv.add_step(ArithStep::Compare {
            lhs: value,
            rhs: bound,
            result: value >= bound,
        });
        deriv
    }

    /// Create a product derivation
    pub fn product(a: f64, b: f64, bound: f64) -> Self {
        let product = a * b;
        let mut deriv = Self::new(format!("{} * {} = {} ≥ {}", a, b, product, bound));
        deriv.add_step(ArithStep::Multiply {
            a,
            b,
            result: product,
        });
        deriv.add_step(ArithStep::Compare {
            lhs: product,
            rhs: bound,
            result: product >= bound,
        });
        deriv
    }

    /// Create a Dempster-Shafer derivation
    pub fn dempster_shafer(a: f64, b: f64, bound: f64) -> Self {
        let ds = 1.0 - (1.0 - a) * (1.0 - b);
        let mut deriv = Self::new(format!("{} ⊕ {} = {} ≥ {}", a, b, ds, bound));
        deriv.add_step(ArithStep::DempsterShafer { a, b, result: ds });
        deriv.add_step(ArithStep::Compare {
            lhs: ds,
            rhs: bound,
            result: ds >= bound,
        });
        deriv
    }

    /// Create a decay derivation
    pub fn decay(base: f64, lambda: f64, time: f64, bound: f64) -> Self {
        let decayed = base * (-lambda * time).exp();
        let mut deriv = Self::new(format!(
            "decay({}, {}, {}) = {} ≥ {}",
            base, lambda, time, decayed, bound
        ));
        deriv.add_step(ArithStep::Decay {
            base,
            lambda,
            time,
            result: decayed,
        });
        deriv.add_step(ArithStep::Compare {
            lhs: decayed,
            rhs: bound,
            result: decayed >= bound,
        });
        deriv
    }
}

/// A step in an arithmetic derivation
#[derive(Clone, Debug)]
pub enum ArithStep {
    /// Direct comparison
    Compare { lhs: f64, rhs: f64, result: bool },
    /// Multiplication
    Multiply { a: f64, b: f64, result: f64 },
    /// Dempster-Shafer combination
    DempsterShafer { a: f64, b: f64, result: f64 },
    /// Decay computation
    Decay {
        base: f64,
        lambda: f64,
        time: f64,
        result: f64,
    },
    /// Lower bound extraction
    LowerBound { expr: String, bound: f64 },
    /// Upper bound extraction
    UpperBound { expr: String, bound: f64 },
}

/// Causal proof types
#[derive(Clone, Debug)]
pub enum CausalProof {
    /// Backdoor criterion proof
    BackdoorCheck {
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        adjustment: HashSet<String>,
    },

    /// Frontdoor criterion proof
    FrontdoorCheck {
        graph: CausalGraphType,
        treatment: String,
        outcome: String,
        mediators: HashSet<String>,
    },

    /// Instrumental variable proof
    IVCheck {
        graph: CausalGraphType,
        instrument: String,
        treatment: String,
        outcome: String,
    },

    /// Do-calculus derivation
    DoCalculus {
        graph: CausalGraphType,
        steps: Vec<DoCalculusStep>,
    },

    /// D-separation proof
    DSeparation {
        graph: CausalGraphType,
        x: HashSet<String>,
        y: HashSet<String>,
        z: HashSet<String>,
    },
}

impl CausalProof {
    /// Get a description of the proof
    pub fn describe(&self) -> String {
        match self {
            Self::BackdoorCheck {
                treatment,
                outcome,
                adjustment,
                ..
            } => {
                format!("backdoor({} → {} | {:?})", treatment, outcome, adjustment)
            }
            Self::FrontdoorCheck {
                treatment,
                outcome,
                mediators,
                ..
            } => {
                format!("frontdoor({} → {} via {:?})", treatment, outcome, mediators)
            }
            Self::IVCheck {
                instrument,
                treatment,
                outcome,
                ..
            } => {
                format!("IV({} → {} → {})", instrument, treatment, outcome)
            }
            Self::DoCalculus { steps, .. } => {
                format!("do-calculus ({} steps)", steps.len())
            }
            Self::DSeparation { x, y, z, .. } => {
                format!("{:?} ⊥⊥ {:?} | {:?}", x, y, z)
            }
        }
    }
}

/// A step in a do-calculus derivation
#[derive(Clone, Debug)]
pub struct DoCalculusStep {
    /// Which rule was applied
    pub rule: DoCalculusRule,
    /// The expression before this step
    pub before: String,
    /// The expression after this step
    pub after: String,
    /// Justification
    pub justification: String,
}

/// Do-calculus rules
#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum DoCalculusRule {
    /// Rule 1: Insertion/deletion of observations
    /// P(y | do(x), z, w) = P(y | do(x), w) if (Y ⊥⊥ Z | X, W)_{G_X̄}
    Rule1,

    /// Rule 2: Action/observation exchange
    /// P(y | do(x), do(z), w) = P(y | do(x), z, w) if (Y ⊥⊥ Z | X, W)_{G_X̄Z̄}
    Rule2,

    /// Rule 3: Insertion/deletion of actions
    /// P(y | do(x), do(z), w) = P(y | do(x), w) if (Y ⊥⊥ Z | X, W)_{G_X̄Z(W)̄}
    Rule3,
}

impl std::fmt::Display for DoCalculusRule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rule1 => write!(f, "Rule 1 (observation)"),
            Self::Rule2 => write!(f, "Rule 2 (action/observation)"),
            Self::Rule3 => write!(f, "Rule 3 (action)"),
        }
    }
}

/// A proof term (for use in proof construction)
#[derive(Clone, Debug)]
pub enum ProofTerm {
    /// A variable
    Var(String),
    /// A literal confidence value
    ConfLit(f64),
    /// An application
    App(Arc<ProofTerm>, Arc<ProofTerm>),
    /// A lambda abstraction
    Lam(String, Arc<ProofTerm>),
    /// A pair
    Pair(Arc<ProofTerm>, Arc<ProofTerm>),
    /// First projection
    Fst(Arc<ProofTerm>),
    /// Second projection
    Snd(Arc<ProofTerm>),
}

impl ProofTerm {
    /// Create a variable term
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Create a confidence literal term
    pub fn conf_lit(value: f64) -> Self {
        Self::ConfLit(value)
    }

    /// Create an application
    pub fn app(f: ProofTerm, arg: ProofTerm) -> Self {
        Self::App(Arc::new(f), Arc::new(arg))
    }

    /// Create a lambda abstraction
    pub fn lam(var: impl Into<String>, body: ProofTerm) -> Self {
        Self::Lam(var.into(), Arc::new(body))
    }

    /// Create a pair
    pub fn pair(fst: ProofTerm, snd: ProofTerm) -> Self {
        Self::Pair(Arc::new(fst), Arc::new(snd))
    }

    /// Project first component
    pub fn fst(p: ProofTerm) -> Self {
        Self::Fst(Arc::new(p))
    }

    /// Project second component
    pub fn snd(p: ProofTerm) -> Self {
        Self::Snd(Arc::new(p))
    }
}

impl std::fmt::Display for ProofTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var(name) => write!(f, "{}", name),
            Self::ConfLit(v) => write!(f, "{:.2}", v),
            Self::App(func, arg) => write!(f, "({} {})", func, arg),
            Self::Lam(var, body) => write!(f, "(λ{}. {})", var, body),
            Self::Pair(a, b) => write!(f, "⟨{}, {}⟩", a, b),
            Self::Fst(p) => write!(f, "π₁({})", p),
            Self::Snd(p) => write!(f, "π₂({})", p),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_literal_cmp_proof() {
        let proof = Proof::literal_cmp(0.95, 0.90);
        assert!(proof.is_some());
        let p = proof.unwrap();
        assert!(p.is_static());
    }

    #[test]
    fn test_literal_cmp_fails() {
        let proof = Proof::literal_cmp(0.80, 0.90);
        assert!(proof.is_none());
    }

    #[test]
    fn test_refl_proof() {
        let conf = ConfidenceType::literal(0.95);
        let proof = Proof::refl(conf);
        assert!(proof.is_static());
    }

    #[test]
    fn test_runtime_check_not_static() {
        let pred =
            Predicate::confidence_geq(ConfidenceType::var("ε"), ConfidenceType::literal(0.95));
        let proof = Proof::runtime_check(pred);
        assert!(!proof.is_static());
    }

    #[test]
    fn test_and_intro() {
        let p1 = Proof::literal_cmp(0.95, 0.90).unwrap();
        let p2 = Proof::literal_cmp(0.85, 0.80).unwrap();
        let and_proof = Proof::and_intro(p1, p2);
        assert!(and_proof.is_static());
    }

    #[test]
    fn test_arith_derivation_product() {
        let deriv = ArithDerivation::product(0.9, 0.8, 0.7);
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_arith_derivation_ds() {
        let deriv = ArithDerivation::dempster_shafer(0.6, 0.7, 0.85);
        // 1 - (0.4 * 0.3) = 0.88 ≥ 0.85
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_arith_derivation_decay() {
        let deriv = ArithDerivation::decay(1.0, 0.1, 5.0, 0.5);
        // e^(-0.5) ≈ 0.606 ≥ 0.5
        assert_eq!(deriv.steps.len(), 2);
    }

    #[test]
    fn test_backdoor_proof() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");
        // Simple case: no confounders, empty adjustment set works
        let proof = Proof::backdoor(graph, "X".to_string(), "Y".to_string(), HashSet::new());
        // This should succeed for the simple X → Y graph
        assert!(proof.is_some());
    }

    #[test]
    fn test_proof_description() {
        let proof = Proof::literal_cmp(0.95, 0.90).unwrap();
        let desc = proof.describe();
        assert!(desc.contains("literal comparison"));
    }

    #[test]
    fn test_proof_term_display() {
        let term = ProofTerm::lam(
            "x",
            ProofTerm::app(ProofTerm::var("f"), ProofTerm::var("x")),
        );
        let s = format!("{}", term);
        assert!(s.contains("λx"));
    }

    #[test]
    fn test_causal_proof_describe() {
        let mut adj = HashSet::new();
        adj.insert("Z".to_string());
        let proof = CausalProof::BackdoorCheck {
            graph: CausalGraphType::new(),
            treatment: "X".to_string(),
            outcome: "Y".to_string(),
            adjustment: adj,
        };
        let desc = proof.describe();
        assert!(desc.contains("backdoor"));
    }
}
