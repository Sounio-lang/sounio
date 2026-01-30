//! Ownership and borrow checking
//!
//! This module implements the ownership and borrow checker for Sounio.
//! It enforces:
//! - Move semantics for non-Copy types
//! - Borrow rules (at most one exclusive borrow, or any number of shared borrows)
//! - Linear type constraints (must be used exactly once)
//! - Affine type constraints (may be used at most once)

mod checker;
mod state;

pub use checker::OwnershipChecker;
pub use state::{BorrowState, Linearity, OwnershipState, Place, PlaceId, ScopeState, TrackedValue};
