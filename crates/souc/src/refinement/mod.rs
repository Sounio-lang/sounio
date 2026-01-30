//! Refinement Types System
//!
//! Extends the D type system with logical predicates verified by Z3.
//!
//! Refinement types allow specifying properties that must hold for values:
//! - `{ x: int | x > 0 }` - positive integers
//! - `{ dose: mg | 0.0 < dose <= max_dose }` - safe medication doses
//! - `{ arr: [T] | len(arr) > 0 }` - non-empty arrays
//!
//! # Theory Background
//!
//! A refinement type `{ v: T | P }` extends a base type `T` with a predicate `P`
//! over the refinement variable `v`. Subtyping follows:
//!
//! ```text
//! { v: T | P } <: { v: T | Q }  iff  ∀v. P(v) ⟹ Q(v)
//! ```
//!
//! The implication is verified by an SMT solver (Z3).
//!
//! # Liquid Types
//!
//! This implementation uses Liquid Types for automatic inference:
//! - Predicates are conjunctions of "qualifiers" from a predefined set
//! - Inference finds the strongest valid refinement
//! - Enables automatic verification without explicit annotations
//!
//! # References
//!
//! - Rondon, P. M., Kawaguchi, M., & Jhala, R. (2008). Liquid types.
//! - Vazou, N., et al. (2014). Refinement types for Haskell.
//! - Xi, H., & Pfenning, F. (1999). Dependent types in practical programming.

pub mod constraint;
pub mod infer;
pub mod predicate;
pub mod qualifiers;
pub mod solver;
pub mod subtype;

pub use constraint::*;
pub use infer::RefinementInference;
pub use predicate::*;
pub use qualifiers::{Qualifier, medical_qualifiers, standard_qualifiers};
pub use solver::{Counterexample, VerifyResult, Z3Solver};
pub use subtype::SubtypeChecker;
