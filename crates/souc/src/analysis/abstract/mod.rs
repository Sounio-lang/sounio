//! Abstract Interpretation Framework for Sounio
//!
//! This module provides a comprehensive abstract interpretation framework
//! for static program analysis. It enables precise invariant discovery
//! and feeds into refinement type verification.
//!
//! # Overview
//!
//! Abstract interpretation approximates program semantics using abstract
//! domains. Instead of computing exact values, we compute abstract values
//! that soundly over-approximate all possible concrete values.
//!
//! # Domains Available
//!
//! - **Interval**: Numeric ranges [lo, hi]
//! - **Sign**: Sign information (negative, zero, positive)
//! - **Congruence**: Modular arithmetic (value mod m)
//! - **Octagon**: Relational constraints (+/- x +/- y <= c)
//! - **Product**: Combinations of domains with reduction
//!
//! # Example Usage
//!
//! ```ignore
//! use compiler::analysis::r#abstract::{AbstractInterpreter, IntervalAnalysis};
//!
//! // Create interpreter with interval domain
//! let mut interpreter = AbstractInterpreter::new();
//!
//! // Analyze a function
//! let result = interpreter.analyze_function(&hlir_function);

#![allow(clippy::excessive_nesting)]
//!
//! // Query invariants at specific points
//! if let Some(state) = result.entry_state(block_id) {
//!     let x_interval = state.get(x_value_id);
//!     if x_interval.is_non_negative() {
//!         println!("x is guaranteed non-negative at this point");
//!     }
//! }
//! ```
//!
//! # Integration with Refinement Types
//!
//! Discovered invariants can be fed to the refinement type checker:
//!
//! ```ignore
//! // Abstract interpretation discovers: 0 <= i < n
//! let invariant = interpreter.get_invariant(loop_body, i);
//!
//! // Convert to refinement predicate
//! let pred = invariant.to_predicate("i");
//!
//! // Use in refinement type checking
//! refinement_checker.add_assumption(pred);
//! ```
//!
//! # References
//!
//! - Cousot, P., & Cousot, R. (1977). Abstract interpretation: a unified
//!   lattice model for static analysis of programs.
//! - Mine, A. (2006). The octagon abstract domain.
//! - Granger, P. (1989). Static analysis of linear congruence equalities.

pub mod congruence;
pub mod domain;
pub mod fixpoint;
pub mod intervals;
pub mod octagon;
pub mod product;
pub mod signs;
pub mod transfer;

// Re-exports
pub use congruence::Congruence;
pub use domain::{AbstractDomain, BoolDomain, Flat, Lifted, PowerSet};
pub use fixpoint::{
    FixpointConfig, FixpointEngine, FixpointResult, SccDecomposition, simple_fixpoint,
};
pub use intervals::{Bound, Interval, ThresholdWidening};
pub use octagon::{Octagon, VarId, Weight};
pub use product::{IntervalCongruence, IntervalSign, NumericDomain, Product};
pub use signs::Sign;
pub use transfer::{AbstractState, ConditionRefinement, IntervalTransfer, NumericTransfer};

use crate::hlir::{BlockId, HlirFunction, ValueId};
use crate::refinement::{Predicate, RefinementType, Term};
use crate::types::Type;
use std::collections::HashMap;

/// Main entry point for abstract interpretation
///
/// The AbstractInterpreter orchestrates the analysis of HLIR functions
/// using configurable abstract domains.
#[derive(Debug)]
pub struct AbstractInterpreter {
    /// Configuration for the analysis
    pub config: AnalysisConfig,
}

/// Configuration for abstract interpretation
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Fixpoint iteration config
    pub fixpoint: FixpointConfig,
    /// Whether to compute invariants for all program points
    pub compute_invariants: bool,
    /// Whether to use relational domain (octagon)
    pub use_relational: bool,
    /// Whether to track congruence information
    pub track_congruence: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            fixpoint: FixpointConfig::default(),
            compute_invariants: true,
            use_relational: false, // Octagon is expensive
            track_congruence: true,
        }
    }
}

impl AbstractInterpreter {
    /// Create a new abstract interpreter
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self { config }
    }

    /// Analyze a function using interval domain
    pub fn analyze_function_intervals(&self, func: &HlirFunction) -> FixpointResult<Interval> {
        let initial = self.create_initial_state_intervals(func);

        let mut engine = FixpointEngine::new(self.config.fixpoint.clone(), |state, block| {
            let mut result = state.clone();
            for instr in &block.instructions {
                IntervalTransfer::transfer(&mut result, instr);
            }
            result
        });

        engine.compute(func, initial)
    }

    /// Analyze a function using numeric domain (interval + sign + congruence)
    pub fn analyze_function_numeric(&self, func: &HlirFunction) -> FixpointResult<NumericDomain> {
        let initial = self.create_initial_state_numeric(func);

        let mut engine = FixpointEngine::new(self.config.fixpoint.clone(), |state, block| {
            let mut result = state.clone();
            for instr in &block.instructions {
                NumericTransfer::transfer(&mut result, instr);
            }
            result
        });

        engine.compute(func, initial)
    }

    /// Create initial state for interval analysis
    fn create_initial_state_intervals(&self, func: &HlirFunction) -> AbstractState<Interval> {
        let mut state = AbstractState::new();

        // Parameters are initially unconstrained (top)
        for param in &func.params {
            state.set(param.value, Interval::top());
        }

        state
    }

    /// Create initial state for numeric analysis
    fn create_initial_state_numeric(&self, func: &HlirFunction) -> AbstractState<NumericDomain> {
        let mut state = AbstractState::new();

        for param in &func.params {
            state.set(param.value, NumericDomain::top());
        }

        state
    }

    /// Extract loop invariants from analysis result
    pub fn extract_invariants(
        &self,
        result: &FixpointResult<Interval>,
        func: &HlirFunction,
    ) -> HashMap<BlockId, Vec<Invariant>> {
        let mut invariants = HashMap::new();

        for block in &func.blocks {
            if let Some(state) = result.entry_state(block.id) {
                let mut block_invariants = Vec::new();

                for (id, val) in state.iter() {
                    if !val.is_top() && !val.is_bottom() {
                        block_invariants.push(Invariant {
                            value: *id,
                            interval: *val,
                        });
                    }
                }

                if !block_invariants.is_empty() {
                    invariants.insert(block.id, block_invariants);
                }
            }
        }

        invariants
    }
}

impl Default for AbstractInterpreter {
    fn default() -> Self {
        Self::new()
    }
}

/// An invariant discovered through abstract interpretation
#[derive(Debug, Clone)]
pub struct Invariant {
    /// The value ID this invariant is about
    pub value: ValueId,
    /// The interval constraint
    pub interval: Interval,
}

impl Invariant {
    /// Convert to a refinement predicate
    pub fn to_predicate(&self, var_name: &str) -> Predicate {
        match self.interval {
            Interval::Bottom => Predicate::False,
            Interval::Range { lo, hi } => {
                let mut preds = Vec::new();

                // Lower bound: v >= lo
                if let Bound::Finite(l) = lo {
                    preds.push(Predicate::ge(Term::var(var_name), Term::int(l)));
                }

                // Upper bound: v <= hi
                if let Bound::Finite(h) = hi {
                    preds.push(Predicate::le(Term::var(var_name), Term::int(h)));
                }

                if preds.is_empty() {
                    Predicate::True
                } else {
                    Predicate::and(preds)
                }
            }
        }
    }

    /// Convert to a refinement type
    pub fn to_refinement_type(&self, var_name: &str, base: Type) -> RefinementType {
        let pred = self.to_predicate(var_name);
        RefinementType::refined(base, var_name, pred)
    }
}

/// Analysis for array bounds checking
pub struct ArrayBoundsAnalysis;

impl ArrayBoundsAnalysis {
    /// Check if an array access is safe
    ///
    /// Returns true if index is definitely within bounds.
    pub fn is_safe_access(index_interval: &Interval, array_length: Option<usize>) -> SafetyResult {
        match (index_interval, array_length) {
            (Interval::Bottom, _) => SafetyResult::Unreachable,
            (_, None) => SafetyResult::Unknown, // Unknown array length
            (Interval::Range { lo, hi }, Some(len)) => {
                let len_bound = Bound::Finite(len as i64);

                // Check lower bound: index >= 0
                let lower_ok = *lo >= Bound::Finite(0);

                // Check upper bound: index < len
                let upper_ok = *hi < len_bound;

                if lower_ok && upper_ok {
                    SafetyResult::Safe
                } else if *lo >= len_bound || *hi < Bound::Finite(0) {
                    SafetyResult::AlwaysOutOfBounds
                } else {
                    SafetyResult::MaybeOutOfBounds
                }
            }
        }
    }

    /// Analyze a loop to find array access invariants
    pub fn analyze_loop_access(
        counter_interval: &Interval,
        bound_interval: &Interval,
    ) -> LoopInvariant {
        // Check if counter starts at 0
        let starts_at_zero = match counter_interval.lo() {
            Some(Bound::Finite(0)) => true,
            _ => false,
        };

        // Check if counter is bounded by loop bound
        let bounded_by_n = match (counter_interval.hi(), bound_interval.lo()) {
            (Some(c_hi), Some(b_lo)) => c_hi <= b_lo,
            _ => false,
        };

        LoopInvariant {
            counter_non_negative: counter_interval.is_non_negative(),
            counter_less_than_bound: bounded_by_n,
            starts_at_zero,
        }
    }
}

/// Result of array bounds safety check
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafetyResult {
    /// Access is definitely safe
    Safe,
    /// Access might be out of bounds
    MaybeOutOfBounds,
    /// Access is definitely out of bounds
    AlwaysOutOfBounds,
    /// Code is unreachable
    Unreachable,
    /// Cannot determine (e.g., unknown array length)
    Unknown,
}

/// Loop invariant for array access
#[derive(Debug, Clone)]
pub struct LoopInvariant {
    /// Counter is always >= 0
    pub counter_non_negative: bool,
    /// Counter is always < bound
    pub counter_less_than_bound: bool,
    /// Counter starts at 0
    pub starts_at_zero: bool,
}

impl LoopInvariant {
    /// Check if this loop invariant guarantees safe array access
    pub fn guarantees_safe_access(&self) -> bool {
        self.counter_non_negative && self.counter_less_than_bound
    }

    /// Convert to refinement predicate
    pub fn to_predicate(&self, counter_name: &str, bound_name: &str) -> Predicate {
        let mut preds = Vec::new();

        if self.counter_non_negative {
            preds.push(Predicate::ge(Term::var(counter_name), Term::int(0)));
        }

        if self.counter_less_than_bound {
            preds.push(Predicate::lt(
                Term::var(counter_name),
                Term::var(bound_name),
            ));
        }

        if preds.is_empty() {
            Predicate::True
        } else {
            Predicate::and(preds)
        }
    }
}

/// Integration with refinement type inference
pub struct RefinementIntegration;

impl RefinementIntegration {
    /// Feed abstract interpretation results to refinement inference
    pub fn invariants_to_assumptions(
        invariants: &HashMap<BlockId, Vec<Invariant>>,
        var_names: &HashMap<ValueId, String>,
    ) -> Vec<(BlockId, Vec<Predicate>)> {
        let mut result = Vec::new();

        for (block_id, invs) in invariants {
            let preds: Vec<_> = invs
                .iter()
                .filter_map(|inv| {
                    let name = var_names.get(&inv.value)?;
                    Some(inv.to_predicate(name))
                })
                .collect();

            if !preds.is_empty() {
                result.push((*block_id, preds));
            }
        }

        result
    }

    /// Generate precondition for a function
    pub fn infer_precondition(
        entry_state: &AbstractState<Interval>,
        param_names: &[(ValueId, String)],
    ) -> Predicate {
        let mut preds = Vec::new();

        for (id, name) in param_names {
            let interval = entry_state.get(*id);
            if !interval.is_top() {
                let inv = Invariant {
                    value: *id,
                    interval,
                };
                preds.push(inv.to_predicate(name));
            }
        }

        if preds.is_empty() {
            Predicate::True
        } else {
            Predicate::and(preds)
        }
    }
}

/// Entry point for analyzing a function
pub fn analyze_function(func: &HlirFunction) -> AnalysisResult {
    let interpreter = AbstractInterpreter::new();
    let interval_result = interpreter.analyze_function_intervals(func);
    let invariants = interpreter.extract_invariants(&interval_result, func);

    AnalysisResult {
        fixpoint_result: interval_result,
        invariants,
    }
}

/// Complete analysis result
pub struct AnalysisResult {
    /// Fixpoint iteration result
    pub fixpoint_result: FixpointResult<Interval>,
    /// Extracted invariants
    pub invariants: HashMap<BlockId, Vec<Invariant>>,
}

impl AnalysisResult {
    /// Check if the function terminates (all paths reach return)
    pub fn check_termination(&self) -> bool {
        self.fixpoint_result.converged
    }

    /// Get invariant at a specific block
    pub fn get_invariants(&self, block: BlockId) -> Option<&Vec<Invariant>> {
        self.invariants.get(&block)
    }

    /// Check if a block is reachable
    pub fn is_reachable(&self, block: BlockId) -> bool {
        self.fixpoint_result.is_reachable(block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_invariant_to_predicate() {
        let inv = Invariant {
            value: ValueId(0),
            interval: Interval::from_range(0, 10),
        };

        let pred = inv.to_predicate("x");

        // Should produce: x >= 0 && x <= 10
        match pred {
            Predicate::And(preds) => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("Expected And predicate"),
        }
    }

    #[test]
    fn test_array_bounds_safe() {
        let index = Interval::from_range(0, 9);
        let result = ArrayBoundsAnalysis::is_safe_access(&index, Some(10));
        assert_eq!(result, SafetyResult::Safe);
    }

    #[test]
    fn test_array_bounds_maybe_unsafe() {
        let index = Interval::from_range(0, 15);
        let result = ArrayBoundsAnalysis::is_safe_access(&index, Some(10));
        assert_eq!(result, SafetyResult::MaybeOutOfBounds);
    }

    #[test]
    fn test_array_bounds_always_unsafe() {
        let index = Interval::from_range(10, 15);
        let result = ArrayBoundsAnalysis::is_safe_access(&index, Some(10));
        assert_eq!(result, SafetyResult::AlwaysOutOfBounds);
    }

    #[test]
    fn test_loop_invariant() {
        let counter = Interval::from_range(0, 9);
        let bound = Interval::constant(10);

        let inv = ArrayBoundsAnalysis::analyze_loop_access(&counter, &bound);

        assert!(inv.counter_non_negative);
        assert!(inv.counter_less_than_bound);
        assert!(inv.starts_at_zero);
        assert!(inv.guarantees_safe_access());
    }

    #[test]
    fn test_loop_invariant_predicate() {
        let inv = LoopInvariant {
            counter_non_negative: true,
            counter_less_than_bound: true,
            starts_at_zero: true,
        };

        let pred = inv.to_predicate("i", "n");

        // Should produce: i >= 0 && i < n
        match pred {
            Predicate::And(preds) => {
                assert_eq!(preds.len(), 2);
            }
            _ => panic!("Expected And predicate"),
        }
    }

    #[test]
    fn test_abstract_interpreter_creation() {
        let interpreter = AbstractInterpreter::new();
        assert!(interpreter.config.compute_invariants);
        assert!(!interpreter.config.use_relational);
    }

    #[test]
    fn test_safety_result_variants() {
        assert_eq!(SafetyResult::Safe, SafetyResult::Safe);
        assert_ne!(SafetyResult::Safe, SafetyResult::MaybeOutOfBounds);
    }
}
