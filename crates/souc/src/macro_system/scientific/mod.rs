//! Scientific domain-specific macros
//!
//! Provides compile-time code generation for:
//! - Dimensional analysis (units of measure)
//! - Automatic differentiation
//! - Linear algebra DSL
//! - Statistical modeling

pub mod autodiff;
pub mod units;

pub use autodiff::*;
pub use units::*;
