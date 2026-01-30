//! Integration of Linear Types with Epistemic and Units Systems
//!
//! This module provides integration between the linear type system and
//! other Sounio type system features:
//!
//! - **Epistemic Integration**: Linear knowledge types, confidence tracking
//! - **Units Integration**: Linearly-tracked quantities
//! - **Effects Integration**: Linear resources and effect handlers
//!
//! # Epistemic-Linear Types
//!
//! Knowledge can be linear when:
//! - It comes from a one-time observation (quantum measurement)
//! - It represents a one-time credential or token
//! - It must be consumed in a proof
//!
//! ```text
//! linear Knowledge[Observation, 0.95]    -- Must use this observation
//! affine Knowledge[Credential, 1.0]      -- Can drop if unused
//! ```
//!
//! # Quantity-Linear Types
//!
//! Physical quantities can be linear for:
//! - Resource tracking (total amount consumed)
//! - Conservation laws (mass/energy balance)
//! - Batch processing (fixed input amounts)
//!
//! ```text
//! linear Quantity[100.0, mg]    -- Must consume exactly this amount
//! ```

use std::fmt;
use std::marker::PhantomData;

use super::kind::Linearity;
use super::typed::{Affine, Linear, Unrestricted};

// ============================================================================
// Linear Knowledge - Epistemic types with linearity
// ============================================================================

/// Linear knowledge type combining epistemic and linear aspects.
///
/// This represents knowledge that has usage constraints:
/// - Must be used (linear) - destructive observations
/// - Can be dropped (affine) - optional evidence
/// - Can be shared (unrestricted) - published knowledge
#[derive(Clone, Debug)]
pub struct LinearKnowledge<T> {
    /// The knowledge content
    content: T,
    /// Confidence level [0, 1]
    confidence: f64,
    /// Provenance information
    provenance: String,
    /// Linearity constraint
    linearity: Linearity,
}

impl<T> LinearKnowledge<T> {
    /// Create new linear knowledge (must be used exactly once).
    pub fn linear(content: T, confidence: f64, provenance: impl Into<String>) -> Linear<Self> {
        Linear::new(Self {
            content,
            confidence,
            provenance: provenance.into(),
            linearity: Linearity::Linear,
        })
    }

    /// Create new affine knowledge (can be dropped).
    pub fn affine(content: T, confidence: f64, provenance: impl Into<String>) -> Affine<Self> {
        Affine::new(Self {
            content,
            confidence,
            provenance: provenance.into(),
            linearity: Linearity::Affine,
        })
    }

    /// Create unrestricted knowledge (can be shared).
    pub fn unrestricted(
        content: T,
        confidence: f64,
        provenance: impl Into<String>,
    ) -> Unrestricted<Self> {
        Unrestricted::new(Self {
            content,
            confidence,
            provenance: provenance.into(),
            linearity: Linearity::Unrestricted,
        })
    }

    /// Get the content.
    pub fn content(&self) -> &T {
        &self.content
    }

    /// Get the confidence level.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Get the provenance.
    pub fn provenance(&self) -> &str {
        &self.provenance
    }

    /// Get the linearity.
    pub fn linearity(&self) -> Linearity {
        self.linearity
    }

    /// Consume and extract the content.
    pub fn into_content(self) -> T {
        self.content
    }

    /// Map over the content.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> LinearKnowledge<U> {
        LinearKnowledge {
            content: f(self.content),
            confidence: self.confidence,
            provenance: self.provenance,
            linearity: self.linearity,
        }
    }

    /// Combine with another knowledge using a function.
    ///
    /// The resulting confidence is the minimum of the two.
    pub fn combine<U, V>(
        self,
        other: LinearKnowledge<U>,
        f: impl FnOnce(T, U) -> V,
    ) -> LinearKnowledge<V> {
        LinearKnowledge {
            content: f(self.content, other.content),
            confidence: self.confidence.min(other.confidence),
            provenance: format!("{} + {}", self.provenance, other.provenance),
            linearity: self.linearity.meet(other.linearity),
        }
    }
}

impl<T: fmt::Display> fmt::Display for LinearKnowledge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}Knowledge[{}, {:.2}]",
            self.linearity.keyword(),
            self.content,
            self.confidence
        )
    }
}

// ============================================================================
// Linear Quantity - Physical quantities with linearity
// ============================================================================

/// A physical quantity with linearity tracking.
///
/// This is useful for:
/// - Conservation laws (linear quantities must be conserved)
/// - Resource accounting (track total consumption)
/// - Batch processing (fixed amounts that must be used)
#[derive(Clone, Debug)]
pub struct LinearQuantity<N, U> {
    /// The numerical value
    value: N,
    /// The linearity constraint
    linearity: Linearity,
    /// Phantom for the unit type
    _unit: PhantomData<U>,
}

impl<N, U> LinearQuantity<N, U> {
    /// Create a new linear quantity (must be consumed).
    pub fn linear(value: N) -> Linear<Self> {
        Linear::new(Self {
            value,
            linearity: Linearity::Linear,
            _unit: PhantomData,
        })
    }

    /// Create a new affine quantity (can be discarded).
    pub fn affine(value: N) -> Affine<Self> {
        Affine::new(Self {
            value,
            linearity: Linearity::Affine,
            _unit: PhantomData,
        })
    }

    /// Create a new unrestricted quantity.
    pub fn unrestricted(value: N) -> Unrestricted<Self> {
        Unrestricted::new(Self {
            value,
            linearity: Linearity::Unrestricted,
            _unit: PhantomData,
        })
    }

    /// Get the value.
    pub fn value(&self) -> &N {
        &self.value
    }

    /// Get the linearity.
    pub fn linearity(&self) -> Linearity {
        self.linearity
    }

    /// Consume and extract the value.
    pub fn into_value(self) -> N {
        self.value
    }
}

impl<N: Copy, U> LinearQuantity<N, U> {
    /// Get a copy of the value (only for Copy types).
    pub fn get(&self) -> N {
        self.value
    }
}

impl<N: std::ops::Add<Output = N>, U> LinearQuantity<N, U> {
    /// Add two quantities, consuming both.
    ///
    /// The resulting linearity is the meet (most restrictive).
    pub fn add(self, other: Self) -> Self {
        Self {
            value: self.value + other.value,
            linearity: self.linearity.meet(other.linearity),
            _unit: PhantomData,
        }
    }
}

impl<N: std::ops::Sub<Output = N>, U> LinearQuantity<N, U> {
    /// Subtract two quantities, consuming both.
    pub fn sub(self, other: Self) -> Self {
        Self {
            value: self.value - other.value,
            linearity: self.linearity.meet(other.linearity),
            _unit: PhantomData,
        }
    }
}

impl<N: fmt::Display, U> fmt::Display for LinearQuantity<N, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}Quantity[{}]", self.linearity.keyword(), self.value)
    }
}

// ============================================================================
// Observation - One-time observations (linear by nature)
// ============================================================================

/// A one-time observation that must be consumed.
///
/// Observations are inherently linear because:
/// - They represent a single event in time
/// - Re-observation would be a different observation
/// - They carry provenance that shouldn't be duplicated
#[derive(Clone, Debug)]
pub struct Observation<T> {
    /// The observed value
    value: T,
    /// When the observation was made (as a string timestamp)
    timestamp: String,
    /// The instrument or method used
    instrument: String,
    /// Confidence in the observation
    confidence: f64,
}

impl<T> Observation<T> {
    /// Create a new observation (always linear).
    pub fn new(
        value: T,
        timestamp: impl Into<String>,
        instrument: impl Into<String>,
        confidence: f64,
    ) -> Linear<Self> {
        Linear::new(Self {
            value,
            timestamp: timestamp.into(),
            instrument: instrument.into(),
            confidence,
        })
    }

    /// Get the observed value.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Get the timestamp.
    pub fn timestamp(&self) -> &str {
        &self.timestamp
    }

    /// Get the instrument.
    pub fn instrument(&self) -> &str {
        &self.instrument
    }

    /// Get the confidence.
    pub fn confidence(&self) -> f64 {
        self.confidence
    }

    /// Consume and extract the value.
    pub fn into_value(self) -> T {
        self.value
    }

    /// Map over the observed value.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Observation<U> {
        Observation {
            value: f(self.value),
            timestamp: self.timestamp,
            instrument: self.instrument,
            confidence: self.confidence,
        }
    }
}

impl<T: fmt::Display> fmt::Display for Observation<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Observation[{} @ {} via {}, conf={:.2}]",
            self.value, self.timestamp, self.instrument, self.confidence
        )
    }
}

// ============================================================================
// Credential - Affine authorization tokens
// ============================================================================

/// An authorization credential that can be used at most once.
///
/// Credentials are affine because:
/// - They can expire without being used
/// - Using them consumes them (one-time tokens)
/// - They shouldn't be duplicated
#[derive(Clone, Debug)]
pub struct Credential {
    /// The scope/permission this credential grants
    scope: String,
    /// Who issued the credential
    issuer: String,
    /// When it expires (as a string timestamp)
    expires: Option<String>,
    /// Unique identifier
    id: String,
}

impl Credential {
    /// Create a new credential (always affine).
    pub fn new(
        scope: impl Into<String>,
        issuer: impl Into<String>,
        expires: Option<String>,
    ) -> Affine<Self> {
        use std::time::{SystemTime, UNIX_EPOCH};
        let id = format!(
            "cred_{:x}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        );

        Affine::new(Self {
            scope: scope.into(),
            issuer: issuer.into(),
            expires,
            id,
        })
    }

    /// Get the scope.
    pub fn scope(&self) -> &str {
        &self.scope
    }

    /// Get the issuer.
    pub fn issuer(&self) -> &str {
        &self.issuer
    }

    /// Get the expiration time.
    pub fn expires(&self) -> Option<&str> {
        self.expires.as_deref()
    }

    /// Get the credential ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Check if this credential has a specific permission.
    pub fn has_permission(&self, permission: &str) -> bool {
        self.scope == "*"
            || self.scope == permission
            || self.scope.starts_with(&format!("{}:", permission))
    }
}

impl fmt::Display for Credential {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Credential[{} from {}]", self.scope, self.issuer)
    }
}

// ============================================================================
// Intervention - Causal interventions (linear by nature)
// ============================================================================

/// A causal intervention that must be applied exactly once.
///
/// Interventions are linear because:
/// - They change the state of the world
/// - Applying twice would have different effects
/// - They must be tracked for causal reasoning
#[derive(Clone, Debug)]
pub struct Intervention<T, V> {
    /// The target of the intervention
    target: T,
    /// The value being set
    value: V,
    /// Description of the intervention
    description: String,
}

impl<T, V> Intervention<T, V> {
    /// Create a new intervention (always linear).
    pub fn new(target: T, value: V, description: impl Into<String>) -> Linear<Self> {
        Linear::new(Self {
            target,
            value,
            description: description.into(),
        })
    }

    /// Get the target.
    pub fn target(&self) -> &T {
        &self.target
    }

    /// Get the value.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Get the description.
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Consume and extract the components.
    pub fn into_parts(self) -> (T, V) {
        (self.target, self.value)
    }
}

impl<T: fmt::Display, V: fmt::Display> fmt::Display for Intervention<T, V> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "do({} := {})", self.target, self.value)
    }
}

// ============================================================================
// Published Knowledge - Unrestricted shared knowledge
// ============================================================================

/// Published knowledge that can be freely shared.
///
/// Published knowledge is unrestricted because:
/// - It's been peer-reviewed and validated
/// - It's intended for wide dissemination
/// - Citation creates new references, not copies
#[derive(Clone, Debug)]
pub struct PublishedKnowledge<T> {
    /// The knowledge content
    content: T,
    /// The publication reference
    reference: String,
    /// DOI or other identifier
    identifier: Option<String>,
    /// Publication date
    date: String,
    /// Confidence/quality score
    quality: f64,
}

impl<T> PublishedKnowledge<T> {
    /// Create new published knowledge (always unrestricted).
    pub fn new(
        content: T,
        reference: impl Into<String>,
        identifier: Option<String>,
        date: impl Into<String>,
        quality: f64,
    ) -> Unrestricted<Self> {
        Unrestricted::new(Self {
            content,
            reference: reference.into(),
            identifier,
            date: date.into(),
            quality,
        })
    }

    /// Get the content.
    pub fn content(&self) -> &T {
        &self.content
    }

    /// Get the reference.
    pub fn reference(&self) -> &str {
        &self.reference
    }

    /// Get the identifier.
    pub fn identifier(&self) -> Option<&str> {
        self.identifier.as_deref()
    }

    /// Get the date.
    pub fn date(&self) -> &str {
        &self.date
    }

    /// Get the quality score.
    pub fn quality(&self) -> f64 {
        self.quality
    }

    /// Cite this knowledge (returns a reference).
    pub fn cite(&self) -> String {
        if let Some(id) = &self.identifier {
            format!("{} ({})", self.reference, id)
        } else {
            self.reference.clone()
        }
    }
}

impl<T: fmt::Display> fmt::Display for PublishedKnowledge<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Published[{}, {}]", self.content, self.reference)
    }
}

// ============================================================================
// Effect-Linear Integration
// ============================================================================

/// A linear effect handler.
///
/// Effect handlers that manage linear resources must themselves be linear
/// to ensure proper resource cleanup.
pub struct LinearEffectHandler<E, R> {
    /// The effect type
    _effect: PhantomData<E>,
    /// The handler function
    handler: Box<dyn FnOnce(E) -> R + Send>,
}

impl<E, R> std::fmt::Debug for LinearEffectHandler<E, R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LinearEffectHandler")
            .field("_effect", &self._effect)
            .field("handler", &"<function>")
            .finish()
    }
}

impl<E, R> LinearEffectHandler<E, R> {
    /// Create a new linear effect handler.
    pub fn new(handler: impl FnOnce(E) -> R + Send + 'static) -> Linear<Self> {
        Linear::new(Self {
            _effect: PhantomData,
            handler: Box::new(handler),
        })
    }

    /// Handle an effect, consuming this handler.
    pub fn handle(self, effect: E) -> R {
        (self.handler)(effect)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_knowledge() {
        let knowledge = LinearKnowledge::linear(42, 0.95, "measurement");
        let content = knowledge.consume().into_content();
        assert_eq!(content, 42);
    }

    #[test]
    fn test_affine_knowledge() {
        let knowledge = LinearKnowledge::affine("data", 0.9, "observation");
        // Can be dropped without using
        drop(knowledge);
    }

    #[test]
    fn test_unrestricted_knowledge() {
        let knowledge = LinearKnowledge::unrestricted(100, 0.99, "published");
        // Can access multiple times
        assert_eq!(*knowledge.as_ref().content(), 100);
        assert_eq!(*knowledge.as_ref().content(), 100);
    }

    #[test]
    fn test_observation() {
        let obs = Observation::new(42.0, "2024-01-01", "thermometer", 0.95);
        assert_eq!(*obs.as_ref().value(), 42.0);
        assert_eq!(obs.as_ref().instrument(), "thermometer");
        assert_eq!(obs.consume().into_value(), 42.0);
    }

    #[test]
    fn test_credential() {
        let cred = Credential::new("admin", "auth-service", None);
        assert_eq!(cred.as_ref().unwrap().scope(), "admin");
        assert!(cred.as_ref().unwrap().has_permission("admin"));
        // Can be dropped without using (affine)
        drop(cred);
    }

    #[test]
    fn test_intervention() {
        let interv = Intervention::new("temperature", 37.5, "set thermostat");
        let (target, value) = interv.consume().into_parts();
        assert_eq!(target, "temperature");
        assert_eq!(value, 37.5);
    }

    #[test]
    fn test_published_knowledge() {
        let pub_know = PublishedKnowledge::new(
            "E = mc^2",
            "Einstein (1905)",
            Some("doi:10.1234/example".to_string()),
            "1905",
            1.0,
        );

        // Can cite multiple times
        let cite1 = pub_know.as_ref().cite();
        let cite2 = pub_know.as_ref().cite();
        assert_eq!(cite1, cite2);
        assert!(cite1.contains("Einstein"));
    }

    #[test]
    fn test_linear_quantity() {
        // Create linear quantities
        let q1 = LinearQuantity::<f64, ()>::linear(10.0);
        let q2 = LinearQuantity::<f64, ()>::linear(5.0);

        // Consume and add
        let sum = q1.consume().add(q2.consume());
        assert_eq!(*sum.value(), 15.0);
    }

    #[test]
    fn test_knowledge_combine() {
        let k1 = LinearKnowledge {
            content: 10,
            confidence: 0.9,
            provenance: "A".to_string(),
            linearity: Linearity::Linear,
        };
        let k2 = LinearKnowledge {
            content: 20,
            confidence: 0.8,
            provenance: "B".to_string(),
            linearity: Linearity::Linear,
        };

        let combined = k1.combine(k2, |a, b| a + b);
        assert_eq!(*combined.content(), 30);
        assert_eq!(combined.confidence(), 0.8); // min
        assert_eq!(combined.linearity(), Linearity::Linear);
    }
}
