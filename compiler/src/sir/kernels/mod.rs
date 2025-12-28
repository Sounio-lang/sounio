//! SIR Kernels
//!
//! Canonical computational kernels expressed in SIR form.
//! These kernels serve as:
//! 1. Proof-of-concept for epistemic-aware parallel execution
//! 2. Templates for domain-specific optimizations
//! 3. Reference implementations for testing codegen
//!
//! # The Soul of Epistemic Computation
//!
//! The Monte Carlo kernel is "the soul" - where epistemic-aware allocation
//! meets parallel execution. It demonstrates how confidence flows through
//! computation and how low-confidence particles can be handled differently
//! (the "attention score" concept applied to particles).

pub mod monte_carlo;

pub use monte_carlo::*;
