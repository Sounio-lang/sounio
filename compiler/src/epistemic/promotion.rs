//! Uncertainty Model Promotion Lattice
//!
//! Implements a formal lattice structure for uncertainty representations:
//!
//! ```text
//!                    Particles (SMC)
//!                         │
//!                    Distribution
//!                    ╱         ╲
//!               Affine      DempsterShafer
//!                  │              │
//!              Interval        Fuzzy
//!                    ╲        ╱
//!                      Point
//! ```

use std::cmp::Ordering;
use std::fmt;

/// Uncertainty model identifier with lattice ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UncertaintyLevel {
    Point = 0,
    Interval = 1,
    Fuzzy = 2,
    Affine = 3,
    DempsterShafer = 4,
    Distribution = 5,
    Particles = 6,
}

impl UncertaintyLevel {
    pub const fn height(&self) -> u8 {
        match self {
            Self::Point => 0,
            Self::Interval | Self::Fuzzy => 1,
            Self::Affine | Self::DempsterShafer => 2,
            Self::Distribution => 3,
            Self::Particles => 4,
        }
    }

    pub const fn info_capacity(&self) -> u32 {
        match self {
            Self::Point => 64,
            Self::Interval => 128,
            Self::Fuzzy => 256,
            Self::Affine => 512,
            Self::DempsterShafer => 1024,
            Self::Distribution => 4096,
            Self::Particles => 65536,
        }
    }

    pub const fn cost_multiplier(&self) -> f64 {
        match self {
            Self::Point => 1.0,
            Self::Interval => 2.0,
            Self::Fuzzy => 4.0,
            Self::Affine => 8.0,
            Self::DempsterShafer => 16.0,
            Self::Distribution => 100.0,
            Self::Particles => 1000.0,
        }
    }

    /// Check if this level can be promoted to the target level.
    /// This respects the lattice structure where Interval and Fuzzy are on different branches.
    pub fn can_promote_to(&self, target: Self) -> bool {
        if *self == target {
            return true;
        }
        match (*self, target) {
            // Point can promote to anything
            (Self::Point, _) => true,
            // Interval branch: Interval -> Affine -> Distribution -> Particles
            (Self::Interval, Self::Affine | Self::Distribution | Self::Particles) => true,
            // Fuzzy branch: Fuzzy -> DempsterShafer -> Distribution -> Particles
            (Self::Fuzzy, Self::DempsterShafer | Self::Distribution | Self::Particles) => true,
            // Affine -> Distribution -> Particles
            (Self::Affine, Self::Distribution | Self::Particles) => true,
            // DempsterShafer -> Distribution -> Particles
            (Self::DempsterShafer, Self::Distribution | Self::Particles) => true,
            // Distribution -> Particles
            (Self::Distribution, Self::Particles) => true,
            // Cross-branch: Interval can promote to DempsterShafer via Distribution
            // But NOT directly - must go through Distribution
            // Same for Fuzzy -> Affine
            _ => false,
        }
    }

    pub fn promotable_targets(&self) -> Vec<Self> {
        ALL_LEVELS.iter().filter(|l| self.can_promote_to(**l)).copied().collect()
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "point" | "deterministic" => Some(Self::Point),
            "interval" | "bounds" => Some(Self::Interval),
            "fuzzy" | "membership" => Some(Self::Fuzzy),
            "affine" | "aa" => Some(Self::Affine),
            "dempster-shafer" | "ds" | "belief" => Some(Self::DempsterShafer),
            "distribution" | "dist" | "probabilistic" => Some(Self::Distribution),
            "particles" | "smc" | "pf" => Some(Self::Particles),
            _ => None,
        }
    }
}

impl fmt::Display for UncertaintyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Point => write!(f, "Point"),
            Self::Interval => write!(f, "Interval"),
            Self::Fuzzy => write!(f, "Fuzzy"),
            Self::Affine => write!(f, "Affine"),
            Self::DempsterShafer => write!(f, "Dempster-Shafer"),
            Self::Distribution => write!(f, "Distribution"),
            Self::Particles => write!(f, "Particles"),
        }
    }
}

impl PartialOrd for UncertaintyLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other { return Some(Ordering::Equal); }
        let (sh, oh) = (self.height(), other.height());
        if sh < oh && self.can_promote_to(*other) { Some(Ordering::Less) }
        else if oh < sh && other.can_promote_to(*self) { Some(Ordering::Greater) }
        else { None }
    }
}

pub const ALL_LEVELS: [UncertaintyLevel; 7] = [
    UncertaintyLevel::Point, UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy,
    UncertaintyLevel::Affine, UncertaintyLevel::DempsterShafer,
    UncertaintyLevel::Distribution, UncertaintyLevel::Particles,
];

#[derive(Debug, Clone, Default)]
pub struct PromotionLattice;

impl PromotionLattice {
    pub fn new() -> Self { Self }

    pub fn meet(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel {
        if a == b { return a; }
        match (a.height(), b.height()) {
            (ah, bh) if ah == bh => UncertaintyLevel::Point,
            (ah, bh) if ah < bh => a,
            _ => b,
        }
    }

    pub fn join(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel {
        if a == b { return a; }
        let (ah, bh) = (a.height(), b.height());
        if ah > bh { return a; }
        if bh > ah { return b; }
        match a.height() {
            0 => UncertaintyLevel::Interval,
            1 => UncertaintyLevel::Affine,
            2 => UncertaintyLevel::Distribution,
            _ => UncertaintyLevel::Particles,
        }
    }

    pub fn is_subtype(&self, sub: UncertaintyLevel, sup: UncertaintyLevel) -> bool {
        sub.can_promote_to(sup)
    }

    pub fn join_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel {
        levels.iter().copied().reduce(|a, b| self.join(a, b)).unwrap_or(UncertaintyLevel::Point)
    }

    pub fn meet_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel {
        levels.iter().copied().reduce(|a, b| self.meet(a, b)).unwrap_or(UncertaintyLevel::Particles)
    }

    pub fn ascii_diagram(&self) -> String {
        "                    Particles (SMC)\n                         │\n                    Distribution\n                    ╱         ╲\n               Affine      Dempster-Shafer\n                  │              │\n              Interval        Fuzzy\n                    ╲        ╱\n                      Point\n".to_string()
    }
}

pub trait Promotable: Sized {
    fn uncertainty_level(&self) -> UncertaintyLevel;
    fn can_promote(&self, target: UncertaintyLevel) -> bool { self.uncertainty_level().can_promote_to(target) }
    fn promote_to(&self, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError>;
    fn point_estimate(&self) -> f64;
    fn uncertainty_bounds(&self) -> (f64, f64);
}

#[derive(Debug, Clone)]
pub enum PromotedValue {
    Point { value: f64, confidence: f64 },
    Interval { lower: f64, upper: f64 },
    Fuzzy { support_lower: f64, support_upper: f64, peak: f64, alpha_cut: f64 },
    Affine { center: f64, noise_terms: Vec<(u32, f64)> },
    DempsterShafer { focal_elements: Vec<(f64, f64, f64)> },
    Distribution { samples: Vec<f64>, mean: f64, variance: f64 },
    Particles { particles: Vec<f64>, weights: Vec<f64>, effective_sample_size: f64 },
}

impl PromotedValue {
    pub fn level(&self) -> UncertaintyLevel {
        match self {
            Self::Point { .. } => UncertaintyLevel::Point,
            Self::Interval { .. } => UncertaintyLevel::Interval,
            Self::Fuzzy { .. } => UncertaintyLevel::Fuzzy,
            Self::Affine { .. } => UncertaintyLevel::Affine,
            Self::DempsterShafer { .. } => UncertaintyLevel::DempsterShafer,
            Self::Distribution { .. } => UncertaintyLevel::Distribution,
            Self::Particles { .. } => UncertaintyLevel::Particles,
        }
    }

    pub fn point_estimate(&self) -> f64 {
        match self {
            Self::Point { value, .. } => *value,
            Self::Interval { lower, upper } => (lower + upper) / 2.0,
            Self::Fuzzy { peak, .. } => *peak,
            Self::Affine { center, .. } => *center,
            Self::DempsterShafer { focal_elements } => {
                let total: f64 = focal_elements.iter().map(|(_, _, m)| m).sum();
                if total == 0.0 { 0.0 } else { focal_elements.iter().map(|(l, u, m)| (l + u) / 2.0 * m).sum::<f64>() / total }
            }
            Self::Distribution { mean, .. } => *mean,
            Self::Particles { particles, weights, .. } => {
                let total: f64 = weights.iter().sum();
                if total == 0.0 { 0.0 } else { particles.iter().zip(weights).map(|(p, w)| p * w).sum::<f64>() / total }
            }
        }
    }

    pub fn bounds(&self) -> (f64, f64) {
        match self {
            Self::Point { value, confidence } => { let hw = value.abs() * (1.0 - confidence); (value - hw, value + hw) }
            Self::Interval { lower, upper } => (*lower, *upper),
            Self::Fuzzy { support_lower, support_upper, .. } => (*support_lower, *support_upper),
            Self::Affine { center, noise_terms } => { let t: f64 = noise_terms.iter().map(|(_, n)| n.abs()).sum(); (center - t, center + t) }
            Self::DempsterShafer { focal_elements } => {
                (focal_elements.iter().map(|(l,_,_)| *l).fold(f64::INFINITY, f64::min),
                 focal_elements.iter().map(|(_,u,_)| *u).fold(f64::NEG_INFINITY, f64::max))
            }
            Self::Distribution { samples, .. } => {
                (samples.iter().copied().fold(f64::INFINITY, f64::min), samples.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            }
            Self::Particles { particles, .. } => {
                (particles.iter().copied().fold(f64::INFINITY, f64::min), particles.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum PromotionError {
    CannotDemote { from: UncertaintyLevel, to: UncertaintyLevel },
    IncompatiblePath { from: UncertaintyLevel, to: UncertaintyLevel },
    InsufficientInfo { from: UncertaintyLevel, to: UncertaintyLevel, reason: String },
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CannotDemote { from, to } => write!(f, "Cannot demote from {} to {}", from, to),
            Self::IncompatiblePath { from, to } => write!(f, "No path from {} to {}", from, to),
            Self::InsufficientInfo { from, to, reason } => write!(f, "Cannot promote {} to {}: {}", from, to, reason),
        }
    }
}

impl std::error::Error for PromotionError {}

#[derive(Debug, Clone)]
pub struct Promoter {
    pub default_samples: usize,
    pub default_particles: usize,
    pub seed: Option<u64>,
}

impl Default for Promoter {
    fn default() -> Self { Self { default_samples: 10000, default_particles: 1000, seed: None } }
}

impl Promoter {
    pub fn new() -> Self { Self::default() }
    pub fn with_samples(mut self, n: usize) -> Self { self.default_samples = n; self }
    pub fn with_particles(mut self, n: usize) -> Self { self.default_particles = n; self }

    pub fn promote_point(&self, value: f64, confidence: f64, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError> {
        let hw = value.abs() * (1.0 - confidence) + 1e-10;
        match target {
            UncertaintyLevel::Point => Ok(PromotedValue::Point { value, confidence }),
            UncertaintyLevel::Interval => Ok(PromotedValue::Interval { lower: value - hw, upper: value + hw }),
            UncertaintyLevel::Fuzzy => Ok(PromotedValue::Fuzzy { support_lower: value - hw * 1.5, support_upper: value + hw * 1.5, peak: value, alpha_cut: confidence }),
            UncertaintyLevel::Affine => Ok(PromotedValue::Affine { center: value, noise_terms: vec![(0, hw)] }),
            UncertaintyLevel::DempsterShafer => Ok(PromotedValue::DempsterShafer { focal_elements: vec![(value - hw, value + hw, confidence)] }),
            UncertaintyLevel::Distribution => {
                let std = hw / 1.96;
                let samples = self.generate_normal_samples(value, std);
                Ok(PromotedValue::Distribution { samples, mean: value, variance: std * std })
            }
            UncertaintyLevel::Particles => {
                let std = hw / 1.96;
                let particles = self.generate_normal_samples_n(value, std, self.default_particles);
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles { particles, weights, effective_sample_size: self.default_particles as f64 })
            }
        }
    }

    pub fn promote_interval(&self, lower: f64, upper: f64, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError> {
        if target == UncertaintyLevel::Point { return Err(PromotionError::CannotDemote { from: UncertaintyLevel::Interval, to: target }); }
        let center = (lower + upper) / 2.0;
        let hw = (upper - lower) / 2.0;
        match target {
            UncertaintyLevel::Interval => Ok(PromotedValue::Interval { lower, upper }),
            UncertaintyLevel::Fuzzy => Ok(PromotedValue::Fuzzy { support_lower: lower - hw * 0.1, support_upper: upper + hw * 0.1, peak: center, alpha_cut: 1.0 }),
            UncertaintyLevel::Affine => Ok(PromotedValue::Affine { center, noise_terms: vec![(0, hw)] }),
            UncertaintyLevel::DempsterShafer => Ok(PromotedValue::DempsterShafer { focal_elements: vec![(lower, upper, 1.0)] }),
            UncertaintyLevel::Distribution => {
                let samples: Vec<f64> = (0..self.default_samples).map(|i| lower + (upper - lower) * (i as f64 / self.default_samples as f64)).collect();
                Ok(PromotedValue::Distribution { samples, mean: center, variance: (upper - lower).powi(2) / 12.0 })
            }
            UncertaintyLevel::Particles => {
                let particles: Vec<f64> = (0..self.default_particles).map(|i| lower + (upper - lower) * (i as f64 / self.default_particles as f64)).collect();
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles { particles, weights, effective_sample_size: self.default_particles as f64 })
            }
            _ => Err(PromotionError::CannotDemote { from: UncertaintyLevel::Interval, to: target }),
        }
    }

    fn generate_normal_samples(&self, mean: f64, std: f64) -> Vec<f64> {
        self.generate_normal_samples_n(mean, std, self.default_samples)
    }

    fn generate_normal_samples_n(&self, mean: f64, std: f64, n: usize) -> Vec<f64> {
        (0..n).map(|i| { let u = (i as f64 + 0.5) / n as f64; mean + Self::inv_norm(u) * std }).collect()
    }

    fn inv_norm(p: f64) -> f64 {
        if p <= 0.0 { return f64::NEG_INFINITY; }
        if p >= 1.0 { return f64::INFINITY; }
        let pa = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * pa.ln()).sqrt();
        let z = t - (2.515517 + 0.802853 * t + 0.010328 * t * t) / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
        if p > 0.5 { -z } else { z }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lattice_ordering() {
        let l = PromotionLattice::new();
        assert!(l.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Interval));
        assert!(l.is_subtype(UncertaintyLevel::Point, UncertaintyLevel::Particles));
        assert!(!l.is_subtype(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy));
    }

    #[test]
    fn test_meet_join() {
        let l = PromotionLattice::new();
        assert_eq!(l.meet(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy), UncertaintyLevel::Point);
        assert_eq!(l.join(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy), UncertaintyLevel::Affine);
    }

    #[test]
    fn test_promotion() {
        let p = Promoter::new().with_samples(100);
        let r = p.promote_point(10.0, 0.95, UncertaintyLevel::Distribution).unwrap();
        if let PromotedValue::Distribution { samples, mean, .. } = r {
            assert_eq!(samples.len(), 100);
            assert!((mean - 10.0).abs() < 0.01);
        }
    }
}
