//! Effect system implementation
//!
//! This module re-exports the effect types from the types module and provides
//! additional runtime support for effect handling.

pub mod inference;

pub use crate::types::effects::*;
pub use inference::{EffectChecker, EffectError, EffectErrorKind};

/// Runtime effect handler trait
pub trait Handler<E> {
    type Output;

    fn handle(&self, effect: E) -> Self::Output;
}

/// Effect continuation
pub struct Continuation<T> {
    _marker: std::marker::PhantomData<T>,
}

impl<T> Continuation<T> {
    /// Resume the continuation with a value
    pub fn resume(self, _value: T) {
        // Placeholder for continuation implementation
    }
}
