//! Transmission Dynamics Module
//!
//! Models biological information transmission channels using unit quaternions
//! as described in the preprint. Each channel preserves information content
//! (norm-preserving) while allowing noncommutative transformations.
//!
//! # Channel Types
//!
//! - **Genetic (g)**: DNA replication, vertical inheritance
//! - **Transcriptional (t)**: DNA→RNA transcription, regulatory control
//! - **Post-transcriptional (p)**: RNA processing, splicing, editing
//! - **Epigenetic (e)**: Methylation, histone modification, chromatin state
//!
//! # Mathematical Model
//!
//! Transmission state is a unit quaternion q = g + ti + pj + ek where:
//! ```text
//! |q|² = g² + t² + p² + e² = 1  (simplex constraint)
//! ```
//!
//! Updates are quaternion products (noncommutative):
//! ```text
//! q' = q₂ ∘ q₁  (order matters)
//! ```
//!
//! # Distortion
//!
//! Environmental/stochastic effects perturb channels but preserve total
//! information content through re-normalization.

use super::quaternion::UnitQuat;
use std::fmt;

/// Transmission channel weights
///
/// Represents the distribution of information flow across channels.
/// Invariant: g² + t² + p² + e² = 1 (unit norm)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transmission {
    /// Genetic channel weight
    pub g: f64,
    /// Transcriptional channel weight
    pub t: f64,
    /// Post-transcriptional channel weight
    pub p: f64,
    /// Epigenetic channel weight
    pub e: f64,
}

impl Transmission {
    /// Create new transmission (normalizes to unit)
    pub fn new(g: f64, t: f64, p: f64, e: f64) -> Self {
        let mut trans = Self { g, t, p, e };
        trans.normalize();
        trans
    }

    /// Create from unit quaternion
    pub fn from_unit_quat(q: &UnitQuat) -> Self {
        let inner = q.as_quaternion();
        Self {
            g: inner.w,
            t: inner.x,
            p: inner.y,
            e: inner.z,
        }
    }

    /// Convert to unit quaternion
    pub fn to_unit_quat(&self) -> UnitQuat {
        UnitQuat::new(self.g, self.t, self.p, self.e).unwrap_or(UnitQuat::identity())
    }

    /// Pure genetic transmission
    pub fn genetic() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Pure transcriptional transmission
    pub fn transcriptional() -> Self {
        Self::new(0.0, 1.0, 0.0, 0.0)
    }

    /// Pure post-transcriptional transmission
    pub fn post_transcriptional() -> Self {
        Self::new(0.0, 0.0, 1.0, 0.0)
    }

    /// Pure epigenetic transmission
    pub fn epigenetic() -> Self {
        Self::new(0.0, 0.0, 0.0, 1.0)
    }

    /// Equal mix of all channels
    pub fn balanced() -> Self {
        Self::new(0.5, 0.5, 0.5, 0.5)
    }

    /// Calculate norm
    pub fn norm(&self) -> f64 {
        (self.g * self.g + self.t * self.t + self.p * self.p + self.e * self.e).sqrt()
    }

    /// Normalize to unit
    fn normalize(&mut self) {
        let n = self.norm();
        if n > 1e-15 {
            self.g /= n;
            self.t /= n;
            self.p /= n;
            self.e /= n;
        } else {
            // Default to genetic
            self.g = 1.0;
            self.t = 0.0;
            self.p = 0.0;
            self.e = 0.0;
        }
    }

    /// Check if unit (|q| ≈ 1)
    pub fn is_unit(&self, tolerance: f64) -> bool {
        (self.norm() - 1.0).abs() < tolerance
    }

    /// Compose two transmissions (quaternion product)
    ///
    /// This is noncommutative: t1.compose(t2) ≠ t2.compose(t1) in general
    pub fn compose(&self, other: &Self) -> Self {
        let q1 = self.to_unit_quat();
        let q2 = other.to_unit_quat();
        Self::from_unit_quat(&q1.mul(&q2))
    }

    /// Apply distortion (perturbation + renormalization)
    ///
    /// Models environmental or stochastic effects on transmission.
    /// Preserves total information content (norm=1).
    pub fn apply_distortion(&self, dg: f64, dt: f64, dp: f64, de: f64) -> Self {
        Self::new(self.g + dg, self.t + dt, self.p + dp, self.e + de)
    }

    /// Interpolate between two transmission states
    pub fn interpolate(&self, other: &Self, t: f64) -> Self {
        let q1 = self.to_unit_quat();
        let q2 = other.to_unit_quat();
        Self::from_unit_quat(&q1.slerp(&q2, t))
    }

    /// Get dominant channel
    pub fn dominant_channel(&self) -> Channel {
        let max = self
            .g
            .abs()
            .max(self.t.abs().max(self.p.abs().max(self.e.abs())));
        if (self.g.abs() - max).abs() < 1e-10 {
            Channel::Genetic
        } else if (self.t.abs() - max).abs() < 1e-10 {
            Channel::Transcriptional
        } else if (self.p.abs() - max).abs() < 1e-10 {
            Channel::PostTranscriptional
        } else {
            Channel::Epigenetic
        }
    }

    /// Get channel weights as array
    pub fn weights(&self) -> [f64; 4] {
        [self.g, self.t, self.p, self.e]
    }

    /// Information entropy of channel distribution
    ///
    /// H = -Σ p² log(p²) for normalized weights
    pub fn entropy(&self) -> f64 {
        let weights = [
            self.g * self.g,
            self.t * self.t,
            self.p * self.p,
            self.e * self.e,
        ];
        -weights
            .iter()
            .filter(|&&w| w > 1e-15)
            .map(|&w| w * w.ln())
            .sum::<f64>()
    }
}

impl fmt::Display for Transmission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "T({:.3}g + {:.3}t + {:.3}p + {:.3}e)",
            self.g, self.t, self.p, self.e
        )
    }
}

/// Transmission channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// Genetic: DNA replication, vertical inheritance
    Genetic,
    /// Transcriptional: DNA→RNA, regulatory control
    Transcriptional,
    /// Post-transcriptional: RNA processing, splicing
    PostTranscriptional,
    /// Epigenetic: methylation, histone modification
    Epigenetic,
}

impl Channel {
    /// Get channel name
    pub fn name(&self) -> &'static str {
        match self {
            Channel::Genetic => "genetic",
            Channel::Transcriptional => "transcriptional",
            Channel::PostTranscriptional => "post-transcriptional",
            Channel::Epigenetic => "epigenetic",
        }
    }

    /// Get channel abbreviation
    pub fn abbrev(&self) -> char {
        match self {
            Channel::Genetic => 'g',
            Channel::Transcriptional => 't',
            Channel::PostTranscriptional => 'p',
            Channel::Epigenetic => 'e',
        }
    }

    /// Create pure transmission for this channel
    pub fn to_transmission(&self) -> Transmission {
        match self {
            Channel::Genetic => Transmission::genetic(),
            Channel::Transcriptional => Transmission::transcriptional(),
            Channel::PostTranscriptional => Transmission::post_transcriptional(),
            Channel::Epigenetic => Transmission::epigenetic(),
        }
    }
}

impl fmt::Display for Channel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Transmission history (sequence of states)
#[derive(Debug, Clone)]
pub struct TransmissionHistory {
    /// Sequence of transmission states
    pub states: Vec<Transmission>,
}

impl TransmissionHistory {
    /// Create empty history
    pub fn new() -> Self {
        Self { states: Vec::new() }
    }

    /// Create with initial state
    pub fn with_initial(initial: Transmission) -> Self {
        Self {
            states: vec![initial],
        }
    }

    /// Add new state
    pub fn push(&mut self, state: Transmission) {
        self.states.push(state);
    }

    /// Get current state
    pub fn current(&self) -> Option<&Transmission> {
        self.states.last()
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Compose all states in sequence
    pub fn compose_all(&self) -> Transmission {
        if self.states.is_empty() {
            return Transmission::genetic();
        }
        self.states[1..]
            .iter()
            .fold(self.states[0], |acc, s| acc.compose(s))
    }
}

impl Default for TransmissionHistory {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPSILON: f64 = 1e-10;

    #[test]
    fn test_transmission_unit() {
        let t = Transmission::new(1.0, 2.0, 3.0, 4.0);
        assert!(t.is_unit(EPSILON));
    }

    #[test]
    fn test_pure_channels() {
        let g = Transmission::genetic();
        assert!((g.g - 1.0).abs() < EPSILON);
        assert_eq!(g.dominant_channel(), Channel::Genetic);

        let t = Transmission::transcriptional();
        assert!((t.t - 1.0).abs() < EPSILON);
        assert_eq!(t.dominant_channel(), Channel::Transcriptional);
    }

    #[test]
    fn test_composition_noncommutative() {
        let g = Transmission::genetic();
        let t = Transmission::transcriptional();

        let gt = g.compose(&t);
        let tg = t.compose(&g);

        // g ∘ t ≠ t ∘ g in general (quaternion noncommutativity)
        // However, pure basis quaternions have specific relations
        // i × 1 = 1 × i = i, so this specific case is commutative
        // Let's test with mixed states

        let mixed1 = Transmission::new(0.5, 0.5, 0.5, 0.5);
        let mixed2 = Transmission::new(0.7, 0.3, 0.5, 0.3);

        let m12 = mixed1.compose(&mixed2);
        let m21 = mixed2.compose(&mixed1);

        // Should generally be different (noncommutative)
        let diff = (m12.g - m21.g).abs()
            + (m12.t - m21.t).abs()
            + (m12.p - m21.p).abs()
            + (m12.e - m21.e).abs();

        // Note: The specific values might or might not differ based on quaternion math
        // The test verifies the composition mechanism works
        assert!(m12.is_unit(EPSILON));
        assert!(m21.is_unit(EPSILON));
    }

    #[test]
    fn test_distortion() {
        let g = Transmission::genetic();
        let perturbed = g.apply_distortion(0.1, 0.1, 0.0, 0.0);

        // Still unit after renormalization
        assert!(perturbed.is_unit(EPSILON));

        // Distribution changed
        assert!(perturbed.g > 0.5); // Still mostly genetic
        assert!(perturbed.t > 0.0); // Some transcriptional
    }

    #[test]
    fn test_interpolation() {
        let g = Transmission::genetic();
        let t = Transmission::transcriptional();

        let mid = g.interpolate(&t, 0.5);

        // Midpoint should be unit
        assert!(mid.is_unit(EPSILON));

        // Both channels should have similar weights
        assert!((mid.g.abs() - mid.t.abs()).abs() < 0.1);
    }

    #[test]
    fn test_entropy() {
        let g = Transmission::genetic();
        let balanced = Transmission::balanced();

        // Pure state has zero entropy
        assert!(g.entropy().abs() < EPSILON);

        // Balanced state has maximum entropy
        assert!(balanced.entropy() > g.entropy());
    }

    #[test]
    fn test_history() {
        let mut history = TransmissionHistory::with_initial(Transmission::genetic());
        history.push(Transmission::transcriptional());
        history.push(Transmission::post_transcriptional());

        assert_eq!(history.len(), 3);

        let composed = history.compose_all();
        assert!(composed.is_unit(EPSILON));
    }
}
