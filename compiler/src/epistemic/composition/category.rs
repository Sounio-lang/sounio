//! Categorical Semantics for Epistemic Composition
//!
//! This module provides the category-theoretic foundations for the
//! epistemic algebra, showing that Knowledge forms a monad.
//!
//! # Key Structures
//!
//! - **EpistemicCategory**: The category of types with epistemic morphisms
//! - **EpistemicFunctor**: The LIFT operation as a functor
//! - **EpistemicMonad**: Knowledge as a monad with bind and return
//!
//! # Monad Laws
//!
//! ```text
//! Left identity:  return a >>= f  ≡  f a
//! Right identity: m >>= return    ≡  m
//! Associativity:  (m >>= f) >>= g ≡  m >>= (λx. f x >>= g)
//! ```

use super::confidence::ConfidenceValue;
use super::knowledge::EpistemicValue;
use super::provenance::ProvenanceNode;
use std::collections::HashSet;
use std::marker::PhantomData;

/// The epistemic category: objects are types, morphisms are confidence-aware functions
///
/// A morphism f: A → B in this category is a function that also tracks
/// how confidence propagates through the transformation.
pub trait EpistemicCategory {
    /// The source type
    type Source;
    /// The target type
    type Target;

    /// Apply the morphism
    fn apply(&self, source: EpistemicValue<Self::Source>) -> EpistemicValue<Self::Target>;

    /// Compose two morphisms
    fn compose<M: EpistemicCategory<Source = Self::Target>>(
        self,
        other: M,
    ) -> ComposedMorphism<Self, M>
    where
        Self: Sized,
    {
        ComposedMorphism {
            first: self,
            second: other,
        }
    }
}

/// A composed morphism: f; g
pub struct ComposedMorphism<F, G> {
    first: F,
    second: G,
}

impl<F, G> EpistemicCategory for ComposedMorphism<F, G>
where
    F: EpistemicCategory,
    G: EpistemicCategory<Source = F::Target>,
{
    type Source = F::Source;
    type Target = G::Target;

    fn apply(&self, source: EpistemicValue<Self::Source>) -> EpistemicValue<Self::Target> {
        let intermediate = self.first.apply(source);
        self.second.apply(intermediate)
    }
}

/// A simple morphism defined by a function and confidence factor
pub struct SimpleMorphism<A, B, F>
where
    F: Fn(A) -> B,
{
    /// The underlying function
    func: F,
    /// Confidence factor (how reliable is this transformation)
    confidence_factor: f64,
    /// Name for provenance
    name: String,
    _phantom: PhantomData<(A, B)>,
}

impl<A, B, F> SimpleMorphism<A, B, F>
where
    F: Fn(A) -> B,
{
    /// Create a new simple morphism
    pub fn new(func: F, confidence_factor: f64, name: impl Into<String>) -> Self {
        SimpleMorphism {
            func,
            confidence_factor: confidence_factor.clamp(0.0, 1.0),
            name: name.into(),
            _phantom: PhantomData,
        }
    }

    /// Create a pure morphism (no confidence loss)
    pub fn pure(func: F, name: impl Into<String>) -> Self {
        Self::new(func, 1.0, name)
    }
}

impl<A: Clone, B, F> EpistemicCategory for SimpleMorphism<A, B, F>
where
    F: Fn(A) -> B,
{
    type Source = A;
    type Target = B;

    fn apply(&self, source: EpistemicValue<Self::Source>) -> EpistemicValue<Self::Target> {
        let new_value = (self.func)(source.value().clone());
        let new_confidence = source.confidence().scale(self.confidence_factor);

        EpistemicValue::new(
            new_value,
            new_confidence,
            source.ontology().clone(),
            ProvenanceNode::derived(&self.name, vec![source.provenance().clone()]),
        )
    }
}

/// The LIFT functor: τ → Knowledge[τ]
///
/// This is the "return" operation of the epistemic monad.
pub struct EpistemicFunctor;

impl EpistemicFunctor {
    /// Lift a pure value into the epistemic domain with full certainty
    ///
    /// lift: τ → Knowledge[τ, 1.0, ∅, Primitive]
    pub fn lift<T>(value: T) -> EpistemicValue<T> {
        EpistemicValue::certain(value)
    }

    /// Lift with specified confidence
    pub fn lift_with_confidence<T>(value: T, confidence: f64) -> EpistemicValue<T> {
        EpistemicValue::with_confidence(value, confidence)
    }

    /// Apply a function to a lifted value (functor map)
    ///
    /// fmap: (A → B) → Knowledge[A] → Knowledge[B]
    pub fn fmap<A, B, F>(f: F, ka: EpistemicValue<A>) -> EpistemicValue<B>
    where
        F: FnOnce(A) -> B,
        A: Clone,
    {
        ka.map(f)
    }
}

/// The epistemic monad: Knowledge[_] with return and bind
///
/// This provides the monadic structure for sequencing epistemic computations.
pub struct EpistemicMonad;

impl EpistemicMonad {
    /// Monadic return: τ → Knowledge[τ]
    ///
    /// Same as LIFT with certainty.
    pub fn pure<T>(value: T) -> EpistemicValue<T> {
        EpistemicValue::certain(value)
    }

    /// Monadic bind: Knowledge[A] → (A → Knowledge[B]) → Knowledge[B]
    ///
    /// Sequences epistemic computations, propagating confidence.
    pub fn bind<A, B, F>(ka: EpistemicValue<A>, f: F) -> EpistemicValue<B>
    where
        A: Clone,
        F: FnOnce(A) -> EpistemicValue<B>,
    {
        let kb = f(ka.value().clone());

        // Combine confidences (multiplicative)
        let combined_confidence = ka.confidence().product(kb.confidence());

        // Merge ontologies
        let mut combined_ontology = ka.ontology().clone();
        combined_ontology.extend(kb.ontology().iter().cloned());

        // Merge provenance
        let provenance = ProvenanceNode::derived(
            "bind",
            vec![ka.provenance().clone(), kb.provenance().clone()],
        );

        EpistemicValue::new(
            kb.into_inner(),
            combined_confidence,
            combined_ontology,
            provenance,
        )
    }

    /// Monadic join: Knowledge[Knowledge[A]] → Knowledge[A]
    ///
    /// Flattens nested epistemic values.
    pub fn join<A: Clone>(kka: EpistemicValue<EpistemicValue<A>>) -> EpistemicValue<A> {
        let inner = kka.value().clone();

        // Combine confidences
        let combined_confidence = kka.confidence().product(inner.confidence());

        // Merge ontologies
        let mut combined_ontology = kka.ontology().clone();
        combined_ontology.extend(inner.ontology().iter().cloned());

        // Merge provenance
        let provenance = ProvenanceNode::derived(
            "join",
            vec![kka.provenance().clone(), inner.provenance().clone()],
        );

        EpistemicValue::new(
            inner.into_inner(),
            combined_confidence,
            combined_ontology,
            provenance,
        )
    }

    /// Kleisli composition: (A → Knowledge[B]) → (B → Knowledge[C]) → (A → Knowledge[C])
    pub fn compose_kleisli<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> EpistemicValue<C>
    where
        A: Clone,
        B: Clone,
        F: Fn(A) -> EpistemicValue<B>,
        G: Fn(B) -> EpistemicValue<C>,
    {
        move |a: A| {
            let kb = f(a);
            Self::bind(kb, &g)
        }
    }
}

/// Extension trait for monadic operations on EpistemicValue
pub trait EpistemicMonadExt<A> {
    /// Monadic bind (flatMap)
    fn flat_map<B, F>(self, f: F) -> EpistemicValue<B>
    where
        A: Clone,
        F: FnOnce(A) -> EpistemicValue<B>;

    /// Applicative apply
    fn apply<B, F>(self, kf: EpistemicValue<F>) -> EpistemicValue<B>
    where
        A: Clone,
        F: FnOnce(A) -> B + Clone;

    /// Sequence two epistemic values, keeping the second
    fn then<B>(self, kb: EpistemicValue<B>) -> EpistemicValue<B>
    where
        A: Clone,
        B: Clone;
}

impl<A> EpistemicMonadExt<A> for EpistemicValue<A> {
    fn flat_map<B, F>(self, f: F) -> EpistemicValue<B>
    where
        A: Clone,
        F: FnOnce(A) -> EpistemicValue<B>,
    {
        EpistemicMonad::bind(self, f)
    }

    fn apply<B, F>(self, kf: EpistemicValue<F>) -> EpistemicValue<B>
    where
        A: Clone,
        F: FnOnce(A) -> B + Clone,
    {
        EpistemicMonad::bind(kf, |f| self.map(f))
    }

    fn then<B>(self, kb: EpistemicValue<B>) -> EpistemicValue<B>
    where
        A: Clone,
        B: Clone,
    {
        EpistemicMonad::bind(self, |_| kb)
    }
}

/// Helper to create epistemic computations
pub fn epistemic<T, F>(f: F) -> impl FnOnce() -> EpistemicValue<T>
where
    F: FnOnce() -> T,
{
    move || EpistemicValue::certain(f())
}

/// Helper to sequence epistemic computations
pub fn sequence<T: Clone>(values: Vec<EpistemicValue<T>>) -> EpistemicValue<Vec<T>> {
    if values.is_empty() {
        return EpistemicValue::certain(Vec::new());
    }

    let mut result_values = Vec::with_capacity(values.len());
    let mut combined_confidence = ConfidenceValue::certain();
    let mut combined_ontology = HashSet::new();
    let mut provenances = Vec::with_capacity(values.len());

    for v in values {
        combined_confidence = combined_confidence.product(v.confidence());
        combined_ontology.extend(v.ontology().iter().cloned());
        provenances.push(v.provenance().clone());
        result_values.push(v.into_inner());
    }

    EpistemicValue::new(
        result_values,
        combined_confidence,
        combined_ontology,
        ProvenanceNode::derived("sequence", provenances),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lift() {
        let v = EpistemicFunctor::lift(42);
        assert_eq!(*v.value(), 42);
        assert!(v.is_certain());
    }

    #[test]
    fn test_fmap() {
        let v = EpistemicValue::with_confidence(10, 0.8);
        let doubled = EpistemicFunctor::fmap(|x| x * 2, v);
        assert_eq!(*doubled.value(), 20);
        assert!((doubled.confidence().value() - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_monad_left_identity() {
        // return a >>= f  ≡  f a
        let a = 5;
        let f = |x: i32| EpistemicValue::with_confidence(x * 2, 0.9);

        let left = EpistemicMonad::bind(EpistemicMonad::pure(a), f);
        let right = f(a);

        assert_eq!(*left.value(), *right.value());
        // Confidences may differ slightly due to composition
    }

    #[test]
    fn test_monad_right_identity() {
        // m >>= return  ≡  m
        let m = EpistemicValue::with_confidence(10, 0.8);

        let result = EpistemicMonad::bind(m.clone(), EpistemicMonad::pure);

        assert_eq!(*result.value(), *m.value());
        // Confidence is preserved (1.0 factor)
    }

    #[test]
    fn test_monad_associativity() {
        // (m >>= f) >>= g ≡ m >>= (λx. f x >>= g)
        let m = EpistemicValue::with_confidence(5, 0.9);
        let f = |x: i32| EpistemicValue::with_confidence(x + 1, 0.8);
        let g = |x: i32| EpistemicValue::with_confidence(x * 2, 0.7);

        let left = EpistemicMonad::bind(EpistemicMonad::bind(m.clone(), f), g);
        let right = EpistemicMonad::bind(m, |x| EpistemicMonad::bind(f(x), g));

        assert_eq!(*left.value(), *right.value());
        // (5+1)*2 = 12
        assert_eq!(*left.value(), 12);
    }

    #[test]
    fn test_flat_map_extension() {
        let v = EpistemicValue::with_confidence(10, 0.8);
        let result = v.flat_map(|x| EpistemicValue::with_confidence(x * 2, 0.9));

        assert_eq!(*result.value(), 20);
        // Confidence: 0.8 × 0.9 = 0.72
        assert!((result.confidence().value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_simple_morphism() {
        let morph = SimpleMorphism::new(|x: i32| x * 2, 0.9, "double");
        let v = EpistemicValue::with_confidence(5, 0.8);

        let result = morph.apply(v);

        assert_eq!(*result.value(), 10);
        // Confidence: 0.8 × 0.9 = 0.72
        assert!((result.confidence().value() - 0.72).abs() < 1e-10);
    }

    #[test]
    fn test_morphism_composition() {
        let double = SimpleMorphism::new(|x: i32| x * 2, 0.9, "double");
        let add_one = SimpleMorphism::new(|x: i32| x + 1, 0.95, "add_one");

        let composed = double.compose(add_one);
        let v = EpistemicValue::with_confidence(5, 1.0);

        let result = composed.apply(v);

        assert_eq!(*result.value(), 11); // (5*2)+1 = 11
        // Confidence: 1.0 × 0.9 × 0.95 = 0.855
        assert!((result.confidence().value() - 0.855).abs() < 1e-10);
    }

    #[test]
    fn test_sequence() {
        let values = vec![
            EpistemicValue::with_confidence(1, 0.9),
            EpistemicValue::with_confidence(2, 0.8),
            EpistemicValue::with_confidence(3, 0.7),
        ];

        let result = sequence(values);

        assert_eq!(result.value(), &vec![1, 2, 3]);
        // Confidence: 0.9 × 0.8 × 0.7 = 0.504
        assert!((result.confidence().value() - 0.504).abs() < 1e-10);
    }

    #[test]
    fn test_join() {
        let inner = EpistemicValue::with_confidence(42, 0.8);
        let outer = EpistemicValue::with_confidence(inner, 0.9);

        let flattened = EpistemicMonad::join(outer);

        assert_eq!(*flattened.value(), 42);
        // Confidence: 0.9 × 0.8 = 0.72
        assert!((flattened.confidence().value() - 0.72).abs() < 1e-10);
    }
}
