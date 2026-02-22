//! Grand Challenges — source programs live in examples/grand_challenges/*.sio
//!
//! The actual proofs are written in Sounio and run through the compiler/interpreter.
//! This module only holds the source text constants for embedding in tests.

pub const TOXIC_PEAK_SIO: &str = include_str!("../../../../examples/grand_challenges/toxic_peak.sio");
pub const EULER_LINE_SIO: &str = include_str!("../../../../examples/grand_challenges/euler_line.sio");
pub const TURBULENCE_SIO: &str = include_str!("../../../../examples/grand_challenges/turbulence.sio");

/// Backwards-compat alias used by mod.rs re-export.
pub const SOUNIO_PROOF_DSL: &str = TOXIC_PEAK_SIO;

/// Thin result type kept for backwards-compat with mod.rs re-export.
pub struct StabilityProof {
    pub safety_guaranteed: bool,
    pub proof_confidence: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_files_are_non_empty() {
        assert!(!TOXIC_PEAK_SIO.is_empty());
        assert!(!EULER_LINE_SIO.is_empty());
        assert!(!TURBULENCE_SIO.is_empty());
    }
}
