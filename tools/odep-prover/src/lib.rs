//! ODEP — Oblivious Differential Epistemic Privacy prover stub.
//!
//! This crate verifies the algebraic constraints of Sounio's ZD-based
//! unlearning operation and emits a JSON witness envelope that a
//! downstream verifier (e.g. Halo2/Nova in Phase 2+) can consume.
//!
//! See `spec.md` in the crate root for the R1CS circuit specification
//! and the Lean soundness reduction.

pub mod ffi;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The canonical kernel basis for the sedenion operator A = e3 + e10.
/// Indices (6, 15) with signs (+, -) and (7, 14) with signs (+, +),
/// each scaled by 1/sqrt(2). Matches `zd_basis_u1` / `zd_basis_u2` in
/// `stdlib/epistemic/revocable.sio` and the `alice_kernel` object in
/// `formal/lean4/SounioSurgicalInterventions.lean`.
pub const KERNEL_DIM: usize = 16;
pub const SQRT2_INV: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// Public inputs to the ODEP prover.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnlearnClaim {
    /// Pre-surgery weight vector (the model as trained).
    pub w_pre: [f64; KERNEL_DIM],
    /// Post-surgery weight vector (after the ZD kernel projection subtract).
    pub w_post: [f64; KERNEL_DIM],
    /// Kernel projection coefficients used to compute w_post.
    pub c1: f64,
    pub c2: f64,
    /// The primitive ZD operator indices (i, j).  For Sounio's default
    /// `primA = e3 + e10` we expect (3, 10).
    pub prim_a_i: usize,
    pub prim_a_j: usize,
}

/// The witness envelope emitted by a successful ODEP proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OdepEnvelope {
    pub scheme: &'static str,
    pub regulation: String,
    pub algebraic_theorem: &'static str,
    pub lean_term_hash: String,
    pub r1cs_satisfied: bool,
    pub residual_l2: f64,
    pub prim_a_spec: PrimaSpec,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimaSpec {
    pub i: usize,
    pub j: usize,
}

/// Reasons a witness can fail the constraint check.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OdepError {
    KernelMismatch,
    ProjectionCoeffMismatch,
    PostWeightMismatch,
    ResidualAboveTolerance,
}

/// Main entry point: verify the claim's R1CS constraints and emit an
/// envelope.
pub fn prove(claim: &UnlearnClaim, regulation: &str) -> Result<OdepEnvelope, OdepError> {
    // --- Kernel basis u1, u2 ---
    let mut u1 = [0.0_f64; KERNEL_DIM];
    let mut u2 = [0.0_f64; KERNEL_DIM];
    u1[6]  =  SQRT2_INV;
    u1[15] = -SQRT2_INV;
    u2[7]  =  SQRT2_INV;
    u2[14] =  SQRT2_INV;

    // --- Constraint 3 / 4: projection coefficients ---
    let mut c1_chk = 0.0;
    let mut c2_chk = 0.0;
    for i in 0..KERNEL_DIM {
        c1_chk += claim.w_pre[i] * u1[i];
        c2_chk += claim.w_pre[i] * u2[i];
    }
    if (c1_chk - claim.c1).abs() > 1e-10 || (c2_chk - claim.c2).abs() > 1e-10 {
        return Err(OdepError::ProjectionCoeffMismatch);
    }

    // --- Constraint 5: post-weight reconstruction ---
    let mut residual = 0.0;
    for i in 0..KERNEL_DIM {
        let expected = claim.w_pre[i] - claim.c1 * u1[i] - claim.c2 * u2[i];
        let diff = claim.w_post[i] - expected;
        residual += diff * diff;
    }
    if residual > 1e-10 {
        return Err(OdepError::PostWeightMismatch);
    }

    // --- Hash the "Lean term" (stubbed as a canonical ASCII
    // rendering of the claim) ---
    let ascii_term = format!(
        "unlearning_kernel_exact primA=({},{}) c1={:.10} c2={:.10}",
        claim.prim_a_i, claim.prim_a_j, claim.c1, claim.c2
    );
    let mut hasher = Sha256::new();
    hasher.update(ascii_term.as_bytes());
    let digest = hasher.finalize();
    let lean_term_hash = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &digest);

    Ok(OdepEnvelope {
        scheme: "ODEP-v1",
        regulation: regulation.to_string(),
        algebraic_theorem: "unlearning_kernel_exact",
        lean_term_hash,
        r1cs_satisfied: true,
        residual_l2: residual,
        prim_a_spec: PrimaSpec { i: claim.prim_a_i, j: claim.prim_a_j },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end round trip: build a w_pre with a known kernel
    /// component, compute c1/c2, apply the kernel subtract, and verify
    /// the prover accepts.
    #[test]
    fn roundtrip_accepts() {
        let mut w_pre = [0.0_f64; KERNEL_DIM];
        w_pre[0] = 1.2;
        w_pre[1] = 0.5;
        w_pre[6]  =  0.9 * SQRT2_INV;
        w_pre[15] = -0.9 * SQRT2_INV;

        let mut u1 = [0.0_f64; KERNEL_DIM];
        let mut u2 = [0.0_f64; KERNEL_DIM];
        u1[6]  =  SQRT2_INV; u1[15] = -SQRT2_INV;
        u2[7]  =  SQRT2_INV; u2[14] =  SQRT2_INV;

        let mut c1 = 0.0;
        let mut c2 = 0.0;
        for i in 0..KERNEL_DIM {
            c1 += w_pre[i] * u1[i];
            c2 += w_pre[i] * u2[i];
        }
        let mut w_post = [0.0_f64; KERNEL_DIM];
        for i in 0..KERNEL_DIM {
            w_post[i] = w_pre[i] - c1 * u1[i] - c2 * u2[i];
        }

        let claim = UnlearnClaim {
            w_pre, w_post, c1, c2,
            prim_a_i: 3, prim_a_j: 10,
        };
        let env = prove(&claim, "GDPR-Art17").expect("prover accepts");
        assert_eq!(env.algebraic_theorem, "unlearning_kernel_exact");
        assert!(env.r1cs_satisfied);
        assert!(env.residual_l2 < 1e-10);
    }

    /// A malformed post-weight should be rejected.
    #[test]
    fn tampered_post_rejected() {
        let mut w_pre = [0.0_f64; KERNEL_DIM];
        w_pre[0] = 1.0;
        let claim = UnlearnClaim {
            w_pre,
            w_post: [42.0; KERNEL_DIM],
            c1: 0.0,
            c2: 0.0,
            prim_a_i: 3, prim_a_j: 10,
        };
        assert!(matches!(prove(&claim, "GDPR-Art17"), Err(OdepError::PostWeightMismatch)));
    }
}
