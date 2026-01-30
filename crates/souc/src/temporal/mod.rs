//! Temporal Epistemic Logic - Knowledge that Evolves in Time
//!
//! This module implements Day 33 of the Sounio compiler: temporal
//! reasoning about epistemic states. It answers the question: "How does
//! knowledge evolve over time?"
//!
//! # Core Concepts
//!
//! - **Decay**: Knowledge confidence decreases over time
//! - **Versioning**: Same phenomenon, different temporal snapshots
//! - **Temporal Operators**: LTL operators (always, eventually, since, until)
//! - **Causal Precedence**: Intervention at t₁ → effect at t₂
//!
//! # Theoretical Foundation
//!
//! Based on Kripke frame epistemic-temporal logic:
//! ```text
//! 𝔽 = ⟨W, T, ≤, R, V⟩
//!
//! where:
//!   W = set of possible worlds (epistemic states)
//!   T = set of temporal instants (ℝ⁺ or ℕ)
//!   ≤ ⊆ T × T = temporal order (total, reflexive, transitive)
//!   R ⊆ W × W = epistemic accessibility (S4)
//!   V : W × T → P(Prop) = valuation function
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! // Create knowledge with exponential decay
//! let clearance = TemporalKnowledge::decaying(
//!     Knowledge::new(5.2, Confidence::new(0.95).unwrap(), ...),
//!     0.3, // 30% decay per year
//!     TimeUnit::Years,
//! );
//!
//! // Get confidence at current time
//! let now = clearance.now();
//! println!("Current confidence: {}", now.core.confidence().value());
//!
//! // Project into future
//! let future = clearance.eventually(Duration::days(365));
//! println!("Projected confidence: {}", future.projected_confidence.value());
//! ```

pub mod composition;
pub mod decay;
pub mod knowledge;
pub mod operators;
pub mod types;
pub mod versioning;

// Re-exports
pub use composition::{TemporalComposition, TemporalJoinResult};
pub use decay::{DecayFunction, TimeUnit};
pub use knowledge::TemporalKnowledge;
pub use operators::{
    FuturePrediction, HistoricalAssessment, SinceAssessment, TemporalEvent, UntilMonitor,
    UntilStatus, ValidityConstraint,
};
pub use types::{Temporal, Version, VersionInfo, VersionRelation};
pub use versioning::VersionedKnowledge;
