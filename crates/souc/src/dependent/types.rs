//! Type-level representations for dependent epistemic types
//!
//! This module defines the core types that exist at the type level:
//! - [`ConfidenceType`]: Confidence expressions (literals, variables, operations)
//! - [`OntologyType`]: Ontology specifications at type level
//! - [`TemporalType`]: Temporal semantics at type level
//! - [`CausalGraphType`]: Causal graph structure at type level
//! - [`EpistemicType`]: The full dependent epistemic type with Π and Σ

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

/// Confidence at the type level
///
/// Represents confidence expressions that can be manipulated
/// during type checking and proof search.
///
/// # Examples
///
/// ```rust,ignore
/// // Literal confidence
/// let c1 = ConfidenceType::Literal(0.95);
///
/// // Variable (from function parameter)
/// let c2 = ConfidenceType::Var("ε".to_string());
///
/// // Tensor product (from composition)
/// let c3 = ConfidenceType::Product(Arc::new(c1), Arc::new(c2));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum ConfidenceType {
    /// Literal confidence value r ∈ [0,1]
    Literal(f64),

    /// Type variable (from function parameters or inference)
    Var(String),

    /// Product of two confidences (tensor composition)
    /// ε₁ * ε₂
    Product(Arc<ConfidenceType>, Arc<ConfidenceType>),

    /// Dempster-Shafer combination
    /// ε₁ ⊕ ε₂ = 1 - (1-ε₁)(1-ε₂)
    DempsterShafer(Arc<ConfidenceType>, Arc<ConfidenceType>),

    /// Temporal decay
    /// decay(ε₀, λ, t) = ε₀ * e^(-λt)
    Decay {
        base: Arc<ConfidenceType>,
        lambda: f64,
        elapsed: Duration,
    },

    /// Minimum of two confidences
    Min(Arc<ConfidenceType>, Arc<ConfidenceType>),

    /// Maximum of two confidences
    Max(Arc<ConfidenceType>, Arc<ConfidenceType>),

    /// Conditional confidence (Bayesian update)
    /// P(H|E) = P(E|H) * P(H) / P(E)
    Conditional {
        prior: Arc<ConfidenceType>,
        likelihood: Arc<ConfidenceType>,
        evidence: Arc<ConfidenceType>,
    },

    /// Unknown confidence (for gradual typing)
    Unknown,
}

impl ConfidenceType {
    /// Create a literal confidence
    pub fn literal(value: f64) -> Self {
        Self::Literal(value.clamp(0.0, 1.0))
    }

    /// Create a variable confidence
    pub fn var(name: impl Into<String>) -> Self {
        Self::Var(name.into())
    }

    /// Create product of two confidences
    pub fn product(a: ConfidenceType, b: ConfidenceType) -> Self {
        Self::Product(Arc::new(a), Arc::new(b))
    }

    /// Create Dempster-Shafer combination
    pub fn dempster_shafer(a: ConfidenceType, b: ConfidenceType) -> Self {
        Self::DempsterShafer(Arc::new(a), Arc::new(b))
    }

    /// Create decaying confidence
    pub fn decay(base: ConfidenceType, lambda: f64, elapsed: Duration) -> Self {
        Self::Decay {
            base: Arc::new(base),
            lambda,
            elapsed,
        }
    }

    /// Create minimum of two confidences
    pub fn min(a: ConfidenceType, b: ConfidenceType) -> Self {
        Self::Min(Arc::new(a), Arc::new(b))
    }

    /// Create maximum of two confidences
    pub fn max(a: ConfidenceType, b: ConfidenceType) -> Self {
        Self::Max(Arc::new(a), Arc::new(b))
    }

    /// Try to evaluate to a concrete value
    ///
    /// Returns `Some(value)` if the type can be fully evaluated,
    /// `None` if it contains unbound variables.
    pub fn evaluate(&self, ctx: &super::TypeContext) -> Option<f64> {
        match self {
            Self::Literal(v) => Some(*v),
            Self::Var(name) => ctx.lookup_confidence(name)?.evaluate(ctx),
            Self::Product(a, b) => Some(a.evaluate(ctx)? * b.evaluate(ctx)?),
            Self::DempsterShafer(a, b) => {
                let va = a.evaluate(ctx)?;
                let vb = b.evaluate(ctx)?;
                Some(1.0 - (1.0 - va) * (1.0 - vb))
            }
            Self::Decay {
                base,
                lambda,
                elapsed,
            } => {
                let base_val = base.evaluate(ctx)?;
                let t = elapsed.as_secs_f64();
                Some(base_val * (-lambda * t).exp())
            }
            Self::Min(a, b) => Some(a.evaluate(ctx)?.min(b.evaluate(ctx)?)),
            Self::Max(a, b) => Some(a.evaluate(ctx)?.max(b.evaluate(ctx)?)),
            Self::Conditional {
                prior,
                likelihood,
                evidence,
            } => {
                let p = prior.evaluate(ctx)?;
                let l = likelihood.evaluate(ctx)?;
                let e = evidence.evaluate(ctx)?;
                if e > 0.0 {
                    Some((l * p / e).clamp(0.0, 1.0))
                } else {
                    None
                }
            }
            Self::Unknown => None,
        }
    }

    /// Compute a lower bound (conservative estimate)
    ///
    /// Used for proof search when exact values aren't available.
    pub fn lower_bound(&self, ctx: &super::TypeContext) -> Option<f64> {
        match self {
            Self::Literal(v) => Some(*v),
            Self::Var(name) => ctx.lookup_confidence(name)?.lower_bound(ctx),
            Self::Product(a, b) => Some(a.lower_bound(ctx)? * b.lower_bound(ctx)?),
            Self::DempsterShafer(a, b) => {
                // DS is monotonic, so use lower bounds
                let la = a.lower_bound(ctx)?;
                let lb = b.lower_bound(ctx)?;
                Some(1.0 - (1.0 - la) * (1.0 - lb))
            }
            Self::Decay {
                base,
                lambda,
                elapsed,
            } => {
                // Decay reduces confidence, so lower bound uses lower bound of base
                let base_lb = base.lower_bound(ctx)?;
                let t = elapsed.as_secs_f64();
                Some(base_lb * (-lambda * t).exp())
            }
            Self::Min(a, b) => Some(a.lower_bound(ctx)?.min(b.lower_bound(ctx)?)),
            Self::Max(a, b) => {
                // For max, lower bound is the max of lower bounds
                Some(a.lower_bound(ctx)?.max(b.lower_bound(ctx)?))
            }
            Self::Conditional { .. } => {
                // Conservative: can't bound without more info
                Some(0.0)
            }
            Self::Unknown => Some(0.0),
        }
    }

    /// Compute an upper bound
    pub fn upper_bound(&self, ctx: &super::TypeContext) -> Option<f64> {
        match self {
            Self::Literal(v) => Some(*v),
            Self::Var(name) => ctx.lookup_confidence(name)?.upper_bound(ctx),
            Self::Product(a, b) => Some(a.upper_bound(ctx)? * b.upper_bound(ctx)?),
            Self::DempsterShafer(a, b) => {
                let ua = a.upper_bound(ctx)?;
                let ub = b.upper_bound(ctx)?;
                Some(1.0 - (1.0 - ua) * (1.0 - ub))
            }
            Self::Decay {
                base,
                lambda,
                elapsed,
            } => {
                let base_ub = base.upper_bound(ctx)?;
                let t = elapsed.as_secs_f64();
                Some(base_ub * (-lambda * t).exp())
            }
            Self::Min(a, b) => Some(a.upper_bound(ctx)?.min(b.upper_bound(ctx)?)),
            Self::Max(a, b) => Some(a.upper_bound(ctx)?.max(b.upper_bound(ctx)?)),
            Self::Conditional { .. } => Some(1.0),
            Self::Unknown => Some(1.0),
        }
    }

    /// Check definitional equality (syntactic)
    pub fn definitionally_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Literal(a), Self::Literal(b)) => (a - b).abs() < 1e-10,
            (Self::Var(a), Self::Var(b)) => a == b,
            (Self::Product(a1, b1), Self::Product(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (Self::DempsterShafer(a1, b1), Self::DempsterShafer(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (
                Self::Decay {
                    base: b1,
                    lambda: l1,
                    elapsed: e1,
                },
                Self::Decay {
                    base: b2,
                    lambda: l2,
                    elapsed: e2,
                },
            ) => b1.definitionally_equal(b2) && (l1 - l2).abs() < 1e-10 && e1 == e2,
            (Self::Min(a1, b1), Self::Min(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (Self::Max(a1, b1), Self::Max(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (Self::Unknown, Self::Unknown) => true,
            _ => false,
        }
    }

    /// Get free variables in this type
    pub fn free_vars(&self) -> HashSet<String> {
        match self {
            Self::Literal(_) => HashSet::new(),
            Self::Var(name) => {
                let mut set = HashSet::new();
                set.insert(name.clone());
                set
            }
            Self::Product(a, b)
            | Self::DempsterShafer(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                let mut vars = a.free_vars();
                vars.extend(b.free_vars());
                vars
            }
            Self::Decay { base, .. } => base.free_vars(),
            Self::Conditional {
                prior,
                likelihood,
                evidence,
            } => {
                let mut vars = prior.free_vars();
                vars.extend(likelihood.free_vars());
                vars.extend(evidence.free_vars());
                vars
            }
            Self::Unknown => HashSet::new(),
        }
    }

    /// Substitute a variable with a value
    pub fn substitute(&self, var: &str, value: &ConfidenceType) -> Self {
        match self {
            Self::Literal(v) => Self::Literal(*v),
            Self::Var(name) if name == var => value.clone(),
            Self::Var(name) => Self::Var(name.clone()),
            Self::Product(a, b) => Self::Product(
                Arc::new(a.substitute(var, value)),
                Arc::new(b.substitute(var, value)),
            ),
            Self::DempsterShafer(a, b) => Self::DempsterShafer(
                Arc::new(a.substitute(var, value)),
                Arc::new(b.substitute(var, value)),
            ),
            Self::Decay {
                base,
                lambda,
                elapsed,
            } => Self::Decay {
                base: Arc::new(base.substitute(var, value)),
                lambda: *lambda,
                elapsed: *elapsed,
            },
            Self::Min(a, b) => Self::Min(
                Arc::new(a.substitute(var, value)),
                Arc::new(b.substitute(var, value)),
            ),
            Self::Max(a, b) => Self::Max(
                Arc::new(a.substitute(var, value)),
                Arc::new(b.substitute(var, value)),
            ),
            Self::Conditional {
                prior,
                likelihood,
                evidence,
            } => Self::Conditional {
                prior: Arc::new(prior.substitute(var, value)),
                likelihood: Arc::new(likelihood.substitute(var, value)),
                evidence: Arc::new(evidence.substitute(var, value)),
            },
            Self::Unknown => Self::Unknown,
        }
    }
}

impl std::fmt::Display for ConfidenceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Literal(v) => write!(f, "{:.2}", v),
            Self::Var(name) => write!(f, "{}", name),
            Self::Product(a, b) => write!(f, "({} * {})", a, b),
            Self::DempsterShafer(a, b) => write!(f, "({} ⊕ {})", a, b),
            Self::Decay {
                base,
                lambda,
                elapsed,
            } => write!(f, "decay({}, {}, {:?})", base, lambda, elapsed),
            Self::Min(a, b) => write!(f, "min({}, {})", a, b),
            Self::Max(a, b) => write!(f, "max({}, {})", a, b),
            Self::Conditional {
                prior,
                likelihood,
                evidence,
            } => write!(f, "P({}|{}/{}", prior, likelihood, evidence),
            Self::Unknown => write!(f, "?"),
        }
    }
}

/// Ontology at the type level
///
/// Specifies which ontology validates a piece of knowledge.
#[derive(Clone, Debug, PartialEq)]
pub enum OntologyType {
    /// Specific ontology binding
    Concrete {
        /// Ontology identifier (e.g., "PKPD", "ChEBI", "GO")
        ontology: String,
        /// Optional specific term
        term: Option<String>,
    },

    /// Type variable for ontology
    Var(String),

    /// Union of ontologies (δ₁ ∪ δ₂)
    Union(Arc<OntologyType>, Arc<OntologyType>),

    /// Intersection of ontologies (δ₁ ∩ δ₂)
    Intersection(Arc<OntologyType>, Arc<OntologyType>),

    /// Any ontology (top)
    Any,

    /// No ontology (bottom)
    None,

    /// Unknown (for gradual typing)
    Unknown,
}

impl OntologyType {
    /// Create a concrete ontology type
    pub fn concrete(ontology: impl Into<String>) -> Self {
        Self::Concrete {
            ontology: ontology.into(),
            term: None,
        }
    }

    /// Create a concrete ontology type with term
    pub fn with_term(ontology: impl Into<String>, term: impl Into<String>) -> Self {
        Self::Concrete {
            ontology: ontology.into(),
            term: Some(term.into()),
        }
    }

    /// Create union of two ontologies
    pub fn union(a: OntologyType, b: OntologyType) -> Self {
        Self::Union(Arc::new(a), Arc::new(b))
    }

    /// Create intersection of two ontologies
    pub fn intersection(a: OntologyType, b: OntologyType) -> Self {
        Self::Intersection(Arc::new(a), Arc::new(b))
    }

    /// Check if this ontology contains another
    pub fn contains(&self, other: &OntologyType) -> bool {
        match (self, other) {
            (Self::Any, _) => true,
            (_, Self::None) => true,
            (Self::None, _) => false,
            (_, Self::Any) => false,
            (Self::Concrete { ontology: o1, .. }, Self::Concrete { ontology: o2, .. }) => o1 == o2,
            (Self::Union(a, b), other) => a.contains(other) || b.contains(other),
            (self_, Self::Union(a, b)) => self_.contains(a) && self_.contains(b),
            (Self::Intersection(a, b), other) => a.contains(other) && b.contains(other),
            (self_, Self::Intersection(a, b)) => self_.contains(a) || self_.contains(b),
            (Self::Var(v1), Self::Var(v2)) => v1 == v2,
            (Self::Unknown, _) | (_, Self::Unknown) => true, // Gradual: assume compatible
            _ => false,
        }
    }

    /// Get the set of concrete ontologies
    pub fn to_set(&self) -> HashSet<String> {
        match self {
            Self::Concrete { ontology, .. } => {
                let mut set = HashSet::new();
                set.insert(ontology.clone());
                set
            }
            Self::Union(a, b) => {
                let mut set = a.to_set();
                set.extend(b.to_set());
                set
            }
            Self::Intersection(a, b) => {
                let set_a = a.to_set();
                let set_b = b.to_set();
                set_a.intersection(&set_b).cloned().collect()
            }
            Self::Any | Self::None | Self::Var(_) | Self::Unknown => HashSet::new(),
        }
    }

    /// Check definitional equality
    pub fn definitionally_equal(&self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Concrete {
                    ontology: o1,
                    term: t1,
                },
                Self::Concrete {
                    ontology: o2,
                    term: t2,
                },
            ) => o1 == o2 && t1 == t2,
            (Self::Var(v1), Self::Var(v2)) => v1 == v2,
            (Self::Union(a1, b1), Self::Union(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (Self::Intersection(a1, b1), Self::Intersection(a2, b2)) => {
                a1.definitionally_equal(a2) && b1.definitionally_equal(b2)
            }
            (Self::Any, Self::Any) => true,
            (Self::None, Self::None) => true,
            (Self::Unknown, Self::Unknown) => true,
            _ => false,
        }
    }
}

impl std::fmt::Display for OntologyType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concrete { ontology, term } => match term {
                Some(t) => write!(f, "{}:{}", ontology, t),
                None => write!(f, "{}", ontology),
            },
            Self::Var(name) => write!(f, "{}", name),
            Self::Union(a, b) => write!(f, "({} ∪ {})", a, b),
            Self::Intersection(a, b) => write!(f, "({} ∩ {})", a, b),
            Self::Any => write!(f, "⊤"),
            Self::None => write!(f, "⊥"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

/// Provenance at the type level
#[derive(Clone, Debug, PartialEq)]
pub enum ProvenanceType {
    /// Known transformation chain
    Chain(Vec<String>),

    /// Type variable
    Var(String),

    /// Composed provenances
    Composed(Arc<ProvenanceType>, Arc<ProvenanceType>),

    /// Unknown provenance
    Unknown,
}

impl ProvenanceType {
    /// Create a chain from transformations
    pub fn chain(steps: Vec<String>) -> Self {
        Self::Chain(steps)
    }

    /// Compose two provenances
    pub fn compose(a: ProvenanceType, b: ProvenanceType) -> Self {
        Self::Composed(Arc::new(a), Arc::new(b))
    }
}

impl std::fmt::Display for ProvenanceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Chain(steps) => write!(f, "[{}]", steps.join(" → ")),
            Self::Var(name) => write!(f, "{}", name),
            Self::Composed(a, b) => write!(f, "({} ∘ {})", a, b),
            Self::Unknown => write!(f, "?"),
        }
    }
}

/// Temporal semantics at the type level
#[derive(Clone, Debug, PartialEq)]
pub enum TemporalType {
    /// Instant (point in time)
    Instant(Option<i64>), // Unix timestamp, None if unknown

    /// Decaying knowledge
    Decaying { created: Option<i64>, lambda: f64 },

    /// Versioned knowledge
    Versioned { version: Option<String> },

    /// Type variable
    Var(String),

    /// Eternal (no temporal decay)
    Eternal,

    /// Unknown
    Unknown,
}

impl TemporalType {
    /// Create an instant temporal type
    pub fn instant(timestamp: i64) -> Self {
        Self::Instant(Some(timestamp))
    }

    /// Create a decaying temporal type
    pub fn decaying(created: i64, lambda: f64) -> Self {
        Self::Decaying {
            created: Some(created),
            lambda,
        }
    }

    /// Check if temporal type represents fresh knowledge
    pub fn is_fresh(&self, max_age_secs: i64, current_time: i64) -> Option<bool> {
        match self {
            Self::Instant(Some(t)) => Some(current_time - t < max_age_secs),
            Self::Decaying {
                created: Some(t), ..
            } => Some(current_time - t < max_age_secs),
            Self::Eternal => Some(true),
            _ => None,
        }
    }
}

impl std::fmt::Display for TemporalType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Instant(Some(t)) => write!(f, "instant({})", t),
            Self::Instant(None) => write!(f, "instant(?)"),
            Self::Decaying {
                created: Some(t),
                lambda,
            } => write!(f, "decaying({}, λ={})", t, lambda),
            Self::Decaying {
                created: None,
                lambda,
            } => write!(f, "decaying(?, λ={})", lambda),
            Self::Versioned { version: Some(v) } => write!(f, "v{}", v),
            Self::Versioned { version: None } => write!(f, "versioned"),
            Self::Var(name) => write!(f, "{}", name),
            Self::Eternal => write!(f, "eternal"),
            Self::Unknown => write!(f, "?"),
        }
    }
}

/// Causal graph at the type level
///
/// Represents the structure of a causal graph for type-level
/// reasoning about identifiability.
#[derive(Clone, Debug, PartialEq)]
pub struct CausalGraphType {
    /// Node names
    pub nodes: HashSet<String>,

    /// Directed edges (from, to)
    pub edges: HashSet<(String, String)>,

    /// Bidirected edges (unobserved confounders)
    pub bidirected: HashSet<(String, String)>,

    /// Treatment variables
    pub treatments: HashSet<String>,

    /// Outcome variables
    pub outcomes: HashSet<String>,
}

impl CausalGraphType {
    /// Create an empty causal graph type
    pub fn new() -> Self {
        Self {
            nodes: HashSet::new(),
            edges: HashSet::new(),
            bidirected: HashSet::new(),
            treatments: HashSet::new(),
            outcomes: HashSet::new(),
        }
    }

    /// Add a node
    pub fn add_node(&mut self, name: impl Into<String>) {
        self.nodes.insert(name.into());
    }

    /// Add a directed edge
    pub fn add_edge(&mut self, from: impl Into<String>, to: impl Into<String>) {
        let from = from.into();
        let to = to.into();
        self.nodes.insert(from.clone());
        self.nodes.insert(to.clone());
        self.edges.insert((from, to));
    }

    /// Add a bidirected edge
    pub fn add_bidirected(&mut self, a: impl Into<String>, b: impl Into<String>) {
        let a = a.into();
        let b = b.into();
        self.nodes.insert(a.clone());
        self.nodes.insert(b.clone());
        // Store in canonical order
        if a < b {
            self.bidirected.insert((a, b));
        } else {
            self.bidirected.insert((b, a));
        }
    }

    /// Mark a node as treatment
    pub fn set_treatment(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.nodes.insert(name.clone());
        self.treatments.insert(name);
    }

    /// Mark a node as outcome
    pub fn set_outcome(&mut self, name: impl Into<String>) {
        let name = name.into();
        self.nodes.insert(name.clone());
        self.outcomes.insert(name);
    }

    /// Get parents of a node
    pub fn parents(&self, node: &str) -> HashSet<String> {
        self.edges
            .iter()
            .filter(|(_, to)| to == node)
            .map(|(from, _)| from.clone())
            .collect()
    }

    /// Get children of a node
    pub fn children(&self, node: &str) -> HashSet<String> {
        self.edges
            .iter()
            .filter(|(from, _)| from == node)
            .map(|(_, to)| to.clone())
            .collect()
    }

    /// Get descendants of a node
    pub fn descendants(&self, node: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![node.to_string()];

        while let Some(current) = stack.pop() {
            for child in self.children(&current) {
                if visited.insert(child.clone()) {
                    stack.push(child);
                }
            }
        }

        visited
    }

    /// Get ancestors of a node
    pub fn ancestors(&self, node: &str) -> HashSet<String> {
        let mut visited = HashSet::new();
        let mut stack = vec![node.to_string()];

        while let Some(current) = stack.pop() {
            for parent in self.parents(&current) {
                if visited.insert(parent.clone()) {
                    stack.push(parent);
                }
            }
        }

        visited
    }

    /// Check if there's a directed path from a to b
    pub fn has_directed_path(&self, from: &str, to: &str) -> bool {
        self.descendants(from).contains(to)
    }

    /// Remove incoming edges to a node (for graph surgery G_X̄)
    pub fn remove_incoming(&self, node: &str) -> Self {
        let mut new_graph = self.clone();
        new_graph.edges.retain(|(_, to)| to != node);
        new_graph
    }

    /// Remove outgoing edges from a node (for G_X_)
    pub fn remove_outgoing(&self, node: &str) -> Self {
        let mut new_graph = self.clone();
        new_graph.edges.retain(|(from, _)| from != node);
        new_graph
    }
}

impl Default for CausalGraphType {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for CausalGraphType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "G({} nodes, {} edges)",
            self.nodes.len(),
            self.edges.len()
        )
    }
}

/// The full dependent epistemic type
///
/// This is the core type that represents knowledge with all its
/// epistemic parameters tracked at the type level.
#[derive(Clone, Debug)]
pub enum EpistemicType {
    /// Basic knowledge type
    /// Knowledge[τ, ε, δ, Φ, t]
    Knowledge {
        /// The underlying content type
        inner: Arc<crate::types::Type>,
        /// Type-level confidence
        confidence: ConfidenceType,
        /// Type-level ontology
        ontology: OntologyType,
        /// Type-level provenance
        provenance: ProvenanceType,
        /// Type-level temporal semantics
        temporal: TemporalType,
    },

    /// Causal knowledge (Level 2)
    /// CausalKnowledge[τ, ε, δ, Φ, t, G]
    CausalKnowledge {
        inner: Arc<crate::types::Type>,
        confidence: ConfidenceType,
        ontology: OntologyType,
        provenance: ProvenanceType,
        temporal: TemporalType,
        /// The causal graph structure
        graph: CausalGraphType,
    },

    /// Structural knowledge (Level 3)
    /// StructuralKnowledge[τ, ε, δ, Φ, t, M]
    StructuralKnowledge {
        inner: Arc<crate::types::Type>,
        confidence: ConfidenceType,
        ontology: OntologyType,
        provenance: ProvenanceType,
        temporal: TemporalType,
        /// The causal graph
        graph: CausalGraphType,
        /// Has structural equations
        has_equations: bool,
    },

    /// Refinement type
    /// {x : τ | P}
    Refinement {
        /// Base epistemic type
        base: Arc<EpistemicType>,
        /// The refinement predicate
        predicate: super::Predicate,
    },

    /// Dependent function type (Π-type)
    /// Π(x : A). B(x)
    Pi {
        /// Parameter name
        param_name: String,
        /// Parameter type
        param_type: Arc<crate::types::Type>,
        /// Body type (may reference param_name)
        body: Arc<EpistemicType>,
    },

    /// Dependent pair type (Σ-type)
    /// Σ(x : A). B(x)
    Sigma {
        /// First component name
        fst_name: String,
        /// First component type
        fst_type: Arc<crate::types::Type>,
        /// Second component type (may reference fst_name)
        snd_type: Arc<EpistemicType>,
    },

    /// Proof type
    /// Proof[P]
    Proof(super::Predicate),

    /// Type variable (for polymorphism)
    Var(String),

    /// Unknown type (for gradual typing)
    Unknown,
}

impl EpistemicType {
    /// Create a basic knowledge type
    pub fn knowledge(
        inner: crate::types::Type,
        confidence: ConfidenceType,
        ontology: OntologyType,
    ) -> Self {
        Self::Knowledge {
            inner: Arc::new(inner),
            confidence,
            ontology,
            provenance: ProvenanceType::Unknown,
            temporal: TemporalType::Unknown,
        }
    }

    /// Create a knowledge type with all parameters
    pub fn knowledge_full(
        inner: crate::types::Type,
        confidence: ConfidenceType,
        ontology: OntologyType,
        provenance: ProvenanceType,
        temporal: TemporalType,
    ) -> Self {
        Self::Knowledge {
            inner: Arc::new(inner),
            confidence,
            ontology,
            provenance,
            temporal,
        }
    }

    /// Create a causal knowledge type
    pub fn causal_knowledge(
        inner: crate::types::Type,
        confidence: ConfidenceType,
        ontology: OntologyType,
        graph: CausalGraphType,
    ) -> Self {
        Self::CausalKnowledge {
            inner: Arc::new(inner),
            confidence,
            ontology,
            provenance: ProvenanceType::Unknown,
            temporal: TemporalType::Unknown,
            graph,
        }
    }

    /// Create a refinement type
    pub fn refinement(base: EpistemicType, predicate: super::Predicate) -> Self {
        Self::Refinement {
            base: Arc::new(base),
            predicate,
        }
    }

    /// Create a Π-type (dependent function)
    pub fn pi(
        param_name: impl Into<String>,
        param_type: crate::types::Type,
        body: EpistemicType,
    ) -> Self {
        Self::Pi {
            param_name: param_name.into(),
            param_type: Arc::new(param_type),
            body: Arc::new(body),
        }
    }

    /// Create a Σ-type (dependent pair)
    pub fn sigma(
        fst_name: impl Into<String>,
        fst_type: crate::types::Type,
        snd_type: EpistemicType,
    ) -> Self {
        Self::Sigma {
            fst_name: fst_name.into(),
            fst_type: Arc::new(fst_type),
            snd_type: Arc::new(snd_type),
        }
    }

    /// Create a proof type
    pub fn proof(predicate: super::Predicate) -> Self {
        Self::Proof(predicate)
    }

    /// Get the confidence type if this is a knowledge type
    pub fn confidence(&self) -> Option<&ConfidenceType> {
        match self {
            Self::Knowledge { confidence, .. }
            | Self::CausalKnowledge { confidence, .. }
            | Self::StructuralKnowledge { confidence, .. } => Some(confidence),
            Self::Refinement { base, .. } => base.confidence(),
            _ => None,
        }
    }

    /// Get the ontology type if this is a knowledge type
    pub fn ontology(&self) -> Option<&OntologyType> {
        match self {
            Self::Knowledge { ontology, .. }
            | Self::CausalKnowledge { ontology, .. }
            | Self::StructuralKnowledge { ontology, .. } => Some(ontology),
            Self::Refinement { base, .. } => base.ontology(),
            _ => None,
        }
    }

    /// Get the causal graph if this is a causal knowledge type
    pub fn causal_graph(&self) -> Option<&CausalGraphType> {
        match self {
            Self::CausalKnowledge { graph, .. } | Self::StructuralKnowledge { graph, .. } => {
                Some(graph)
            }
            Self::Refinement { base, .. } => base.causal_graph(),
            _ => None,
        }
    }

    /// Check if this is a subtype of another (basic structural check)
    pub fn is_knowledge_hierarchy_subtype(&self, other: &Self) -> bool {
        match (self, other) {
            // StructuralKnowledge <: CausalKnowledge <: Knowledge
            (Self::StructuralKnowledge { .. }, Self::CausalKnowledge { .. }) => true,
            (Self::StructuralKnowledge { .. }, Self::Knowledge { .. }) => true,
            (Self::CausalKnowledge { .. }, Self::Knowledge { .. }) => true,
            // Same level
            (Self::Knowledge { .. }, Self::Knowledge { .. }) => true,
            (Self::CausalKnowledge { .. }, Self::CausalKnowledge { .. }) => true,
            (Self::StructuralKnowledge { .. }, Self::StructuralKnowledge { .. }) => true,
            // Refinement inherits from base
            (Self::Refinement { base, .. }, other) => base.is_knowledge_hierarchy_subtype(other),
            (self_, Self::Refinement { base, .. }) => self_.is_knowledge_hierarchy_subtype(base),
            _ => false,
        }
    }
}

impl std::fmt::Display for EpistemicType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Knowledge {
                inner,
                confidence,
                ontology,
                ..
            } => {
                write!(
                    f,
                    "Knowledge[{:?}, ε={}, δ={}]",
                    inner, confidence, ontology
                )
            }
            Self::CausalKnowledge {
                inner,
                confidence,
                ontology,
                graph,
                ..
            } => {
                write!(
                    f,
                    "CausalKnowledge[{:?}, ε={}, δ={}, {}]",
                    inner, confidence, ontology, graph
                )
            }
            Self::StructuralKnowledge {
                inner,
                confidence,
                ontology,
                graph,
                ..
            } => {
                write!(
                    f,
                    "StructuralKnowledge[{:?}, ε={}, δ={}, {}]",
                    inner, confidence, ontology, graph
                )
            }
            Self::Refinement { base, predicate } => {
                write!(f, "{{x : {} | {}}}", base, predicate)
            }
            Self::Pi {
                param_name,
                param_type,
                body,
            } => {
                write!(f, "Π({} : {:?}). {}", param_name, param_type, body)
            }
            Self::Sigma {
                fst_name,
                fst_type,
                snd_type,
            } => {
                write!(f, "Σ({} : {:?}). {}", fst_name, fst_type, snd_type)
            }
            Self::Proof(pred) => write!(f, "Proof[{}]", pred),
            Self::Var(name) => write!(f, "{}", name),
            Self::Unknown => write!(f, "?"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_confidence_literal() {
        let c = ConfidenceType::literal(0.95);
        let ctx = super::super::TypeContext::new();
        assert!((c.evaluate(&ctx).unwrap() - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_confidence_product() {
        let c1 = ConfidenceType::literal(0.9);
        let c2 = ConfidenceType::literal(0.8);
        let product = ConfidenceType::product(c1, c2);
        let ctx = super::super::TypeContext::new();
        assert!((product.evaluate(&ctx).unwrap() - 0.72).abs() < 0.001);
    }

    #[test]
    fn test_confidence_dempster_shafer() {
        let c1 = ConfidenceType::literal(0.6);
        let c2 = ConfidenceType::literal(0.7);
        let ds = ConfidenceType::dempster_shafer(c1, c2);
        let ctx = super::super::TypeContext::new();
        // 1 - (1-0.6)*(1-0.7) = 1 - 0.4*0.3 = 1 - 0.12 = 0.88
        assert!((ds.evaluate(&ctx).unwrap() - 0.88).abs() < 0.001);
    }

    #[test]
    fn test_confidence_decay() {
        let base = ConfidenceType::literal(1.0);
        let decay = ConfidenceType::decay(base, 0.1, Duration::from_secs(10));
        let ctx = super::super::TypeContext::new();
        // e^(-0.1 * 10) = e^(-1) ≈ 0.368
        assert!((decay.evaluate(&ctx).unwrap() - 0.368).abs() < 0.01);
    }

    #[test]
    fn test_confidence_variable() {
        let c = ConfidenceType::var("ε");
        let mut ctx = super::super::TypeContext::new();
        assert!(c.evaluate(&ctx).is_none());

        ctx.bind_confidence("ε", ConfidenceType::literal(0.85));
        assert!((c.evaluate(&ctx).unwrap() - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_confidence_free_vars() {
        let c = ConfidenceType::product(
            ConfidenceType::var("α"),
            ConfidenceType::product(ConfidenceType::var("β"), ConfidenceType::literal(0.5)),
        );
        let vars = c.free_vars();
        assert!(vars.contains("α"));
        assert!(vars.contains("β"));
        assert_eq!(vars.len(), 2);
    }

    #[test]
    fn test_ontology_contains() {
        let pkpd = OntologyType::concrete("PKPD");
        let chebi = OntologyType::concrete("ChEBI");
        let union = OntologyType::union(pkpd.clone(), chebi.clone());

        assert!(union.contains(&pkpd));
        assert!(union.contains(&chebi));
        assert!(OntologyType::Any.contains(&pkpd));
        assert!(pkpd.contains(&OntologyType::None));
    }

    #[test]
    fn test_causal_graph_descendants() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");
        graph.add_edge("X", "Y");

        let desc = graph.descendants("X");
        assert!(desc.contains("M"));
        assert!(desc.contains("Y"));
        assert!(!desc.contains("X"));
    }

    #[test]
    fn test_causal_graph_ancestors() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "M");
        graph.add_edge("M", "Y");
        graph.add_edge("U", "Y");

        let anc = graph.ancestors("Y");
        assert!(anc.contains("M"));
        assert!(anc.contains("X"));
        assert!(anc.contains("U"));
    }

    #[test]
    fn test_causal_graph_surgery() {
        let mut graph = CausalGraphType::new();
        graph.add_edge("U", "X");
        graph.add_edge("X", "Y");
        graph.add_edge("U", "Y");

        // G_X̄: remove incoming to X
        let g_x_bar = graph.remove_incoming("X");
        assert!(!g_x_bar.edges.contains(&("U".to_string(), "X".to_string())));
        assert!(g_x_bar.edges.contains(&("X".to_string(), "Y".to_string())));

        // G_X_: remove outgoing from X
        let g_x_under = graph.remove_outgoing("X");
        assert!(
            g_x_under
                .edges
                .contains(&("U".to_string(), "X".to_string()))
        );
        assert!(
            !g_x_under
                .edges
                .contains(&("X".to_string(), "Y".to_string()))
        );
    }

    #[test]
    fn test_epistemic_type_hierarchy() {
        let inner = crate::types::Type::F64;
        let conf = ConfidenceType::literal(0.9);
        let ont = OntologyType::concrete("PKPD");
        let mut graph = CausalGraphType::new();
        graph.add_edge("X", "Y");

        let knowledge = EpistemicType::knowledge(inner.clone(), conf.clone(), ont.clone());
        let causal =
            EpistemicType::causal_knowledge(inner.clone(), conf.clone(), ont.clone(), graph);

        assert!(causal.is_knowledge_hierarchy_subtype(&knowledge));
        assert!(!knowledge.is_knowledge_hierarchy_subtype(&causal));
    }
}
