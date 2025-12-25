//! Interoperability with external proof systems
//!
//! This module provides integration with external theorem provers and
//! proof assistants, enabling Sounio to leverage powerful proof
//! backends while maintaining epistemic semantics.
//!
//! # Supported Systems
//!
//! - **Lean 4**: Full dependent type verification, mathlib integration
//! - **Coq**: Calculus of Inductive Constructions, mathcomp/ssreflect
//! - **Isabelle/HOL**: Higher-order logic, sledgehammer, AFP
//!
//! # Architecture
//!
//! All proof assistants share a common pattern:
//!
//! 1. **Export**: Translate Sounio refinements/geometry/causal to prover syntax
//! 2. **Prove**: Run the external prover to verify the theorem
//! 3. **Import**: Convert proof certificate to Knowledge<Theorem> with confidence 1.0
//!
//! # Epistemic Semantics
//!
//! Theorems proven by external provers are imported with:
//! - `confidence = 1.0` (axiomatic - machine-verified proof)
//! - `provenance = MerkleNode` tracking the proof tree/script
//! - Integration with Sounio epistemic propagation for approximate numerics

pub mod coq;
pub mod isabelle;
pub mod lean;

// Re-export main types from each module
pub use lean::{
    LeanConfig, LeanExportError, LeanImportError, LeanInterop, LeanServer, LeanTheorem,
    ProofCertificate,
};

pub use coq::{
    CoqConfig, CoqConnectionError, CoqImportError, CoqInterop, CoqProofCertificate, CoqProofError,
    CoqServer, CoqTheorem,
};

pub use isabelle::{
    IsabelleConfig, IsabelleConnectionError, IsabelleImportError, IsabelleInterop,
    IsabelleProofCertificate, IsabelleProofError, IsabelleServer, IsabelleTheorem,
    SledgehammerResult,
};

// =============================================================================
// Unified Prover Backend Trait
// =============================================================================

use crate::epistemic::bayesian::BetaConfidence;
use std::time::Duration;

/// A unified trait for theorem prover backends
///
/// This trait abstracts over different proof assistants (Lean, Coq, Isabelle)
/// providing a common interface for Sounio to interact with them.
pub trait ProverBackend: Send + Sync {
    /// Error type for proof attempts
    type ProofError: std::error::Error;

    /// Error type for imports
    type ImportError: std::error::Error;

    /// Proof certificate type
    type Certificate;

    /// Theorem type with epistemic metadata
    type Theorem;

    /// Name of the prover (e.g., "Lean 4", "Coq", "Isabelle/HOL")
    fn name(&self) -> &'static str;

    /// Check if the prover is available on this system
    fn is_available(&self) -> bool;

    /// Get the prover version
    fn version(&self) -> Option<String>;

    /// Export a refinement constraint to prover syntax
    fn export_refinement(&self, refinement: &UnifiedRefinement) -> String;

    /// Export a geometry predicate to prover syntax
    fn export_geometry(&self, predicate: &UnifiedGeometry) -> String;

    /// Export a causal constraint to prover syntax
    fn export_causal(&self, constraint: &UnifiedCausal) -> String;

    /// Attempt to prove the exported code
    fn prove(&mut self, code: &str) -> Result<Self::Certificate, Self::ProofError>;

    /// Import a proof certificate as epistemic knowledge
    fn import_theorem(&self, cert: &Self::Certificate) -> Self::Theorem;

    /// Get the confidence of an imported theorem (always 1.0 for formal proofs)
    fn theorem_confidence(&self) -> BetaConfidence {
        BetaConfidence::from_confidence(1.0, 10000.0)
    }
}

/// Unified refinement constraint (prover-agnostic)
#[derive(Debug, Clone)]
pub struct UnifiedRefinement {
    pub name: Option<String>,
    pub variable: String,
    pub kind: UnifiedRefinementKind,
}

/// Kind of refinement (prover-agnostic)
#[derive(Debug, Clone)]
pub enum UnifiedRefinementKind {
    Positive,
    NonNegative,
    Range { min: String, max: String },
    Predicate(String),
    Custom(String),
}

/// Unified geometry predicate (prover-agnostic)
#[derive(Debug, Clone)]
pub enum UnifiedGeometry {
    Collinear(String, String, String),
    Perpendicular(String, String),
    Parallel(String, String),
    Congruent(String, String),
    Cyclic(Vec<String>),
    Custom(String),
}

/// Unified causal constraint (prover-agnostic)
#[derive(Debug, Clone)]
pub enum UnifiedCausal {
    Independence {
        x: String,
        y: String,
        given: Vec<String>,
    },
    DoIntervention {
        target: String,
        value: String,
    },
    Counterfactual {
        condition: String,
        outcome: String,
    },
}

/// Result of a multi-prover proof attempt
#[derive(Debug, Clone)]
pub struct MultiProverResult {
    /// Which prover succeeded (if any)
    pub succeeded_prover: Option<String>,
    /// Proof attempts by each prover
    pub attempts: Vec<ProverAttempt>,
    /// Total time across all attempts
    pub total_time: Duration,
}

/// A single prover's attempt
#[derive(Debug, Clone)]
pub struct ProverAttempt {
    pub prover: String,
    pub success: bool,
    pub time: Duration,
    pub error: Option<String>,
}

/// Multi-prover orchestrator
///
/// Tries multiple proof backends in parallel or sequence to find a proof.
pub struct MultiProver {
    pub lean: Option<LeanInterop>,
    pub coq: Option<CoqInterop>,
    pub isabelle: Option<IsabelleInterop>,
    /// Order to try provers
    pub priority: Vec<ProverType>,
    /// Whether to try in parallel
    pub parallel: bool,
}

/// Type of prover
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProverType {
    Lean,
    Coq,
    Isabelle,
}

impl MultiProver {
    /// Create a new multi-prover with default settings
    pub fn new() -> Self {
        Self {
            lean: LeanInterop::new().ok(),
            coq: CoqInterop::new().ok(),
            isabelle: IsabelleInterop::new().ok(),
            priority: vec![ProverType::Lean, ProverType::Isabelle, ProverType::Coq],
            parallel: false,
        }
    }

    /// Check which provers are available
    pub fn available_provers(&self) -> Vec<ProverType> {
        let mut available = vec![];
        if self
            .lean
            .as_ref()
            .map(|l| l.is_available())
            .unwrap_or(false)
        {
            available.push(ProverType::Lean);
        }
        if self.coq.as_ref().map(|c| c.is_available()).unwrap_or(false) {
            available.push(ProverType::Coq);
        }
        if self
            .isabelle
            .as_ref()
            .map(|i| i.is_available())
            .unwrap_or(false)
        {
            available.push(ProverType::Isabelle);
        }
        available
    }

    /// Try to prove a refinement with any available prover
    pub fn prove_refinement(&mut self, refinement: &UnifiedRefinement) -> MultiProverResult {
        let start = std::time::Instant::now();
        let mut attempts = vec![];

        for prover_type in &self.priority {
            let attempt_start = std::time::Instant::now();

            match prover_type {
                ProverType::Lean => {
                    if let Some(ref mut lean) = self.lean {
                        let lean_ref = lean::RefinementConstraint {
                            name: refinement.name.clone(),
                            variable: refinement.variable.clone(),
                            kind: match &refinement.kind {
                                UnifiedRefinementKind::Positive => lean::RefinementKind::Positive,
                                UnifiedRefinementKind::NonNegative => {
                                    lean::RefinementKind::NonNegative
                                }
                                UnifiedRefinementKind::Range { min, max } => {
                                    lean::RefinementKind::Range {
                                        min: min.clone(),
                                        max: max.clone(),
                                    }
                                }
                                UnifiedRefinementKind::Predicate(p) => {
                                    lean::RefinementKind::Predicate(p.clone())
                                }
                                UnifiedRefinementKind::Custom(c) => {
                                    lean::RefinementKind::Custom(c.clone())
                                }
                            },
                        };
                        let code = lean.export_refinement(&lean_ref);
                        match lean.prove(&code) {
                            Ok(_) => {
                                attempts.push(ProverAttempt {
                                    prover: "Lean".to_string(),
                                    success: true,
                                    time: attempt_start.elapsed(),
                                    error: None,
                                });
                                return MultiProverResult {
                                    succeeded_prover: Some("Lean".to_string()),
                                    attempts,
                                    total_time: start.elapsed(),
                                };
                            }
                            Err(e) => {
                                attempts.push(ProverAttempt {
                                    prover: "Lean".to_string(),
                                    success: false,
                                    time: attempt_start.elapsed(),
                                    error: Some(e.to_string()),
                                });
                            }
                        }
                    }
                }
                ProverType::Coq => {
                    if let Some(ref mut coq) = self.coq {
                        let coq_ref = coq::RefinementConstraint {
                            name: refinement.name.clone(),
                            variable: refinement.variable.clone(),
                            kind: match &refinement.kind {
                                UnifiedRefinementKind::Positive => coq::RefinementKind::Positive,
                                UnifiedRefinementKind::NonNegative => {
                                    coq::RefinementKind::NonNegative
                                }
                                UnifiedRefinementKind::Range { min, max } => {
                                    coq::RefinementKind::Range {
                                        min: min.clone(),
                                        max: max.clone(),
                                    }
                                }
                                UnifiedRefinementKind::Predicate(p) => {
                                    coq::RefinementKind::Predicate(p.clone())
                                }
                                UnifiedRefinementKind::Custom(c) => {
                                    coq::RefinementKind::Custom(c.clone())
                                }
                            },
                        };
                        let code = coq.export_refinement(&coq_ref);
                        match coq.prove(&code) {
                            Ok(_) => {
                                attempts.push(ProverAttempt {
                                    prover: "Coq".to_string(),
                                    success: true,
                                    time: attempt_start.elapsed(),
                                    error: None,
                                });
                                return MultiProverResult {
                                    succeeded_prover: Some("Coq".to_string()),
                                    attempts,
                                    total_time: start.elapsed(),
                                };
                            }
                            Err(e) => {
                                attempts.push(ProverAttempt {
                                    prover: "Coq".to_string(),
                                    success: false,
                                    time: attempt_start.elapsed(),
                                    error: Some(e.to_string()),
                                });
                            }
                        }
                    }
                }
                ProverType::Isabelle => {
                    if let Some(ref mut isabelle) = self.isabelle {
                        let isa_ref = isabelle::RefinementConstraint {
                            name: refinement.name.clone(),
                            variable: refinement.variable.clone(),
                            kind: match &refinement.kind {
                                UnifiedRefinementKind::Positive => {
                                    isabelle::RefinementKind::Positive
                                }
                                UnifiedRefinementKind::NonNegative => {
                                    isabelle::RefinementKind::NonNegative
                                }
                                UnifiedRefinementKind::Range { min, max } => {
                                    isabelle::RefinementKind::Range {
                                        min: min.clone(),
                                        max: max.clone(),
                                    }
                                }
                                UnifiedRefinementKind::Predicate(p) => {
                                    isabelle::RefinementKind::Predicate(p.clone())
                                }
                                UnifiedRefinementKind::Custom(c) => {
                                    isabelle::RefinementKind::Custom(c.clone())
                                }
                            },
                        };
                        let code = isabelle.export_refinement(&isa_ref);
                        match isabelle.prove(&code) {
                            Ok(_) => {
                                attempts.push(ProverAttempt {
                                    prover: "Isabelle".to_string(),
                                    success: true,
                                    time: attempt_start.elapsed(),
                                    error: None,
                                });
                                return MultiProverResult {
                                    succeeded_prover: Some("Isabelle".to_string()),
                                    attempts,
                                    total_time: start.elapsed(),
                                };
                            }
                            Err(e) => {
                                attempts.push(ProverAttempt {
                                    prover: "Isabelle".to_string(),
                                    success: false,
                                    time: attempt_start.elapsed(),
                                    error: Some(e.to_string()),
                                });
                            }
                        }
                    }
                }
            }
        }

        MultiProverResult {
            succeeded_prover: None,
            attempts,
            total_time: start.elapsed(),
        }
    }
}

impl Default for MultiProver {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multi_prover_creation() {
        let mp = MultiProver::new();
        // Just test it creates without panic
        assert!(mp.priority.len() == 3);
    }

    #[test]
    fn test_unified_refinement() {
        let refinement = UnifiedRefinement {
            name: Some("positive_x".to_string()),
            variable: "x".to_string(),
            kind: UnifiedRefinementKind::Positive,
        };
        assert_eq!(refinement.variable, "x");
    }

    #[test]
    fn test_unified_geometry() {
        let geom = UnifiedGeometry::Collinear("A".to_string(), "B".to_string(), "C".to_string());
        match geom {
            UnifiedGeometry::Collinear(a, b, c) => {
                assert_eq!(a, "A");
                assert_eq!(b, "B");
                assert_eq!(c, "C");
            }
            _ => panic!("Wrong variant"),
        }
    }

    #[test]
    fn test_prover_type_equality() {
        assert_eq!(ProverType::Lean, ProverType::Lean);
        assert_ne!(ProverType::Lean, ProverType::Coq);
    }

    #[test]
    fn test_multi_prover_result() {
        let result = MultiProverResult {
            succeeded_prover: Some("Lean".to_string()),
            attempts: vec![ProverAttempt {
                prover: "Lean".to_string(),
                success: true,
                time: Duration::from_millis(100),
                error: None,
            }],
            total_time: Duration::from_millis(100),
        };
        assert!(result.succeeded_prover.is_some());
        assert_eq!(result.attempts.len(), 1);
    }
}
