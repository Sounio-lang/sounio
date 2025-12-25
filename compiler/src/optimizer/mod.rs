//! Optimization Passes for Sounio HIR
//!
//! This module contains optimization passes that run on the HIR before
//! lowering to HLIR. The key optimizer is the EpistemicOptimizer which
//! handles Knowledge type optimization.

pub mod epistemic;

pub use epistemic::EpistemicOptimizer;
