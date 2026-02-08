//! Uncertainty Model Promotion Lattice
//!
//! Implements a formal lattice structure for uncertainty representations:
//!
//! ```text
//!                         Particles (SMC)
//!                              │
//!                         Distribution
//!                    ╱    │    │    ╲
//!               Affine Bootstrap CV DempsterShafer
//!                  │       │    │        │
//!              Interval   (sample-based) Fuzzy
//!                    ╲        │        ╱
//!                          Point
//! ```
//!
//! Bootstrap and CrossValidation are sample-based uncertainty representations
//! that sit between Point and Distribution in the lattice.

use std::cmp::Ordering;
use std::fmt;

/// Uncertainty model identifier with lattice ordering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UncertaintyLevel {
    Point = 0,
    Interval = 1,
    Fuzzy = 2,
    /// Bootstrap resampling-based uncertainty (sample-based)
    Bootstrap = 3,
    /// K-fold cross-validation uncertainty (model performance)
    CrossValidation = 4,
    Affine = 5,
    DempsterShafer = 6,
    Distribution = 7,
    Particles = 8,
}

impl UncertaintyLevel {
    pub const fn height(&self) -> u8 {
        match self {
            Self::Point => 0,
            Self::Interval | Self::Fuzzy => 1,
            // Bootstrap and CrossValidation are sample-based, between Interval and Affine
            Self::Bootstrap | Self::CrossValidation => 2,
            Self::Affine | Self::DempsterShafer => 3,
            Self::Distribution => 4,
            Self::Particles => 5,
        }
    }

    pub const fn info_capacity(&self) -> u32 {
        match self {
            Self::Point => 64,
            Self::Interval => 128,
            Self::Fuzzy => 256,
            // Bootstrap stores summary stats from many samples
            Self::Bootstrap => 384,
            // CrossValidation stores k fold values
            Self::CrossValidation => 320,
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
            // Bootstrap has moderate cost (summary-based operations)
            Self::Bootstrap => 6.0,
            // CrossValidation is relatively cheap (small k)
            Self::CrossValidation => 5.0,
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

            // Interval branch: Interval -> Bootstrap -> Affine -> Distribution -> Particles
            (
                Self::Interval,
                Self::Bootstrap | Self::Affine | Self::Distribution | Self::Particles,
            ) => true,

            // Fuzzy branch: Fuzzy -> DempsterShafer -> Distribution -> Particles
            (Self::Fuzzy, Self::DempsterShafer | Self::Distribution | Self::Particles) => true,

            // Bootstrap: sample-based, can promote to Distribution or Particles
            (Self::Bootstrap, Self::Distribution | Self::Particles) => true,

            // CrossValidation: fold-based, can promote to Distribution or Particles
            (Self::CrossValidation, Self::Distribution | Self::Particles) => true,

            // Affine -> Distribution -> Particles
            (Self::Affine, Self::Distribution | Self::Particles) => true,

            // DempsterShafer -> Distribution -> Particles
            (Self::DempsterShafer, Self::Distribution | Self::Particles) => true,

            // Distribution -> Particles
            (Self::Distribution, Self::Particles) => true,

            // Cross-branch promotions:
            // Bootstrap and CrossValidation are incomparable (different use cases)
            // Bootstrap is sample-based, CV is model-evaluation-based
            (Self::Bootstrap, Self::CrossValidation) => false,
            (Self::CrossValidation, Self::Bootstrap) => false,

            // Bootstrap can demote to Interval (loses sample information)
            (Self::Bootstrap, Self::Interval) => false, // Cannot demote
            (Self::CrossValidation, Self::Interval) => false, // Cannot demote

            _ => false,
        }
    }

    pub fn promotable_targets(&self) -> Vec<Self> {
        ALL_LEVELS
            .iter()
            .filter(|l| self.can_promote_to(**l))
            .copied()
            .collect()
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "point" | "deterministic" => Some(Self::Point),
            "interval" | "bounds" => Some(Self::Interval),
            "fuzzy" | "membership" => Some(Self::Fuzzy),
            "bootstrap" | "boot" | "resampling" => Some(Self::Bootstrap),
            "crossvalidation" | "cv" | "kfold" | "cross-validation" => Some(Self::CrossValidation),
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
            Self::Bootstrap => write!(f, "Bootstrap"),
            Self::CrossValidation => write!(f, "CrossValidation"),
            Self::Affine => write!(f, "Affine"),
            Self::DempsterShafer => write!(f, "Dempster-Shafer"),
            Self::Distribution => write!(f, "Distribution"),
            Self::Particles => write!(f, "Particles"),
        }
    }
}

impl PartialOrd for UncertaintyLevel {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self == other {
            return Some(Ordering::Equal);
        }
        let (sh, oh) = (self.height(), other.height());
        if sh < oh && self.can_promote_to(*other) {
            Some(Ordering::Less)
        } else if oh < sh && other.can_promote_to(*self) {
            Some(Ordering::Greater)
        } else {
            None
        }
    }
}

pub const ALL_LEVELS: [UncertaintyLevel; 9] = [
    UncertaintyLevel::Point,
    UncertaintyLevel::Interval,
    UncertaintyLevel::Fuzzy,
    UncertaintyLevel::Bootstrap,
    UncertaintyLevel::CrossValidation,
    UncertaintyLevel::Affine,
    UncertaintyLevel::DempsterShafer,
    UncertaintyLevel::Distribution,
    UncertaintyLevel::Particles,
];

#[derive(Debug, Clone, Default)]
pub struct PromotionLattice;

impl PromotionLattice {
    pub fn new() -> Self {
        Self
    }

    pub fn meet(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel {
        if a == b {
            return a;
        }
        // If a can promote to b, a is below b in the lattice
        if a.can_promote_to(b) {
            return a;
        }
        // If b can promote to a, b is below a in the lattice
        if b.can_promote_to(a) {
            return b;
        }
        // Neither can promote to the other - find common lower bound
        // For our lattice, incomparable elements meet at Point
        UncertaintyLevel::Point
    }

    pub fn join(&self, a: UncertaintyLevel, b: UncertaintyLevel) -> UncertaintyLevel {
        if a == b {
            return a;
        }
        // If a can promote to b, b is above a in the lattice
        if a.can_promote_to(b) {
            return b;
        }
        // If b can promote to a, a is above b in the lattice
        if b.can_promote_to(a) {
            return a;
        }
        // Neither can promote to the other - find least upper bound
        // For our branching lattice:
        // - Interval and Fuzzy join at Distribution (via their respective branches)
        // - Affine and DempsterShafer join at Distribution
        // - Anything incomparable at height 1 joins at Distribution
        // - Anything incomparable at height 2 joins at Distribution
        UncertaintyLevel::Distribution
    }

    pub fn is_subtype(&self, sub: UncertaintyLevel, sup: UncertaintyLevel) -> bool {
        sub.can_promote_to(sup)
    }

    pub fn join_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel {
        levels
            .iter()
            .copied()
            .reduce(|a, b| self.join(a, b))
            .unwrap_or(UncertaintyLevel::Point)
    }

    pub fn meet_all(&self, levels: &[UncertaintyLevel]) -> UncertaintyLevel {
        levels
            .iter()
            .copied()
            .reduce(|a, b| self.meet(a, b))
            .unwrap_or(UncertaintyLevel::Particles)
    }

    pub fn ascii_diagram(&self) -> String {
        r#"                         Particles (SMC)
                              │
                         Distribution
                    ╱    │    │    ╲
               Affine Bootstrap CV Dempster-Shafer
                  │       │    │        │
              Interval   (sample-based) Fuzzy
                    ╲        │        ╱
                          Point
"#
        .to_string()
    }
}

pub trait Promotable: Sized {
    fn uncertainty_level(&self) -> UncertaintyLevel;
    fn can_promote(&self, target: UncertaintyLevel) -> bool {
        self.uncertainty_level().can_promote_to(target)
    }
    fn promote_to(&self, target: UncertaintyLevel) -> Result<PromotedValue, PromotionError>;
    fn point_estimate(&self) -> f64;
    fn uncertainty_bounds(&self) -> (f64, f64);
}

#[derive(Debug, Clone)]
pub enum PromotedValue {
    Point {
        value: f64,
        confidence: f64,
    },
    Interval {
        lower: f64,
        upper: f64,
    },
    Fuzzy {
        support_lower: f64,
        support_upper: f64,
        peak: f64,
        alpha_cut: f64,
    },
    /// Bootstrap resampling-based uncertainty
    Bootstrap {
        /// Original point estimate
        estimate: f64,
        /// Lower percentile bound (e.g., 2.5th percentile)
        percentile_lower: f64,
        /// Upper percentile bound (e.g., 97.5th percentile)
        percentile_upper: f64,
        /// Bootstrap standard error
        bootstrap_se: f64,
        /// Bias estimate
        bias: f64,
        /// Number of samples used
        n_samples: u32,
    },
    /// K-fold cross-validation uncertainty
    CrossValidation {
        /// Mean across folds
        mean: f64,
        /// Standard error of the mean
        std_error: f64,
        /// Standard deviation across folds
        std_dev: f64,
        /// Number of folds
        k: u32,
    },
    Affine {
        center: f64,
        noise_terms: Vec<(u32, f64)>,
    },
    DempsterShafer {
        focal_elements: Vec<(f64, f64, f64)>,
    },
    Distribution {
        samples: Vec<f64>,
        mean: f64,
        variance: f64,
    },
    Particles {
        particles: Vec<f64>,
        weights: Vec<f64>,
        effective_sample_size: f64,
    },
}

impl PromotedValue {
    pub fn level(&self) -> UncertaintyLevel {
        match self {
            Self::Point { .. } => UncertaintyLevel::Point,
            Self::Interval { .. } => UncertaintyLevel::Interval,
            Self::Fuzzy { .. } => UncertaintyLevel::Fuzzy,
            Self::Bootstrap { .. } => UncertaintyLevel::Bootstrap,
            Self::CrossValidation { .. } => UncertaintyLevel::CrossValidation,
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
            Self::Bootstrap { estimate, .. } => *estimate,
            Self::CrossValidation { mean, .. } => *mean,
            Self::Affine { center, .. } => *center,
            Self::DempsterShafer { focal_elements } => {
                let total: f64 = focal_elements.iter().map(|(_, _, m)| m).sum();
                if total == 0.0 {
                    0.0
                } else {
                    focal_elements
                        .iter()
                        .map(|(l, u, m)| (l + u) / 2.0 * m)
                        .sum::<f64>()
                        / total
                }
            }
            Self::Distribution { mean, .. } => *mean,
            Self::Particles {
                particles, weights, ..
            } => {
                let total: f64 = weights.iter().sum();
                if total == 0.0 {
                    0.0
                } else {
                    particles
                        .iter()
                        .zip(weights)
                        .map(|(p, w)| p * w)
                        .sum::<f64>()
                        / total
                }
            }
        }
    }

    pub fn bounds(&self) -> (f64, f64) {
        match self {
            Self::Point { value, confidence } => {
                let hw = value.abs() * (1.0 - confidence);
                (value - hw, value + hw)
            }
            Self::Interval { lower, upper } => (*lower, *upper),
            Self::Fuzzy {
                support_lower,
                support_upper,
                ..
            } => (*support_lower, *support_upper),
            Self::Bootstrap {
                percentile_lower,
                percentile_upper,
                ..
            } => (*percentile_lower, *percentile_upper),
            Self::CrossValidation {
                mean, std_error, k, ..
            } => {
                // Use t-distribution approximation for CI
                let t_value = if *k <= 2 {
                    12.71
                } else if *k <= 5 {
                    2.78
                } else if *k <= 10 {
                    2.26
                } else {
                    1.96
                };
                (mean - t_value * std_error, mean + t_value * std_error)
            }
            Self::Affine {
                center,
                noise_terms,
            } => {
                let t: f64 = noise_terms.iter().map(|(_, n)| n.abs()).sum();
                (center - t, center + t)
            }
            Self::DempsterShafer { focal_elements } => (
                focal_elements
                    .iter()
                    .map(|(l, _, _)| *l)
                    .fold(f64::INFINITY, f64::min),
                focal_elements
                    .iter()
                    .map(|(_, u, _)| *u)
                    .fold(f64::NEG_INFINITY, f64::max),
            ),
            Self::Distribution { samples, .. } => (
                samples.iter().copied().fold(f64::INFINITY, f64::min),
                samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ),
            Self::Particles { particles, .. } => (
                particles.iter().copied().fold(f64::INFINITY, f64::min),
                particles.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ),
        }
    }
}

#[derive(Debug, Clone)]
pub enum PromotionError {
    CannotDemote {
        from: UncertaintyLevel,
        to: UncertaintyLevel,
    },
    IncompatiblePath {
        from: UncertaintyLevel,
        to: UncertaintyLevel,
    },
    InsufficientInfo {
        from: UncertaintyLevel,
        to: UncertaintyLevel,
        reason: String,
    },
}

impl fmt::Display for PromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CannotDemote { from, to } => write!(f, "Cannot demote from {} to {}", from, to),
            Self::IncompatiblePath { from, to } => write!(f, "No path from {} to {}", from, to),
            Self::InsufficientInfo { from, to, reason } => {
                write!(f, "Cannot promote {} to {}: {}", from, to, reason)
            }
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
    fn default() -> Self {
        Self {
            default_samples: 10000,
            default_particles: 1000,
            seed: None,
        }
    }
}

impl Promoter {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn with_samples(mut self, n: usize) -> Self {
        self.default_samples = n;
        self
    }
    pub fn with_particles(mut self, n: usize) -> Self {
        self.default_particles = n;
        self
    }

    pub fn promote_point(
        &self,
        value: f64,
        confidence: f64,
        target: UncertaintyLevel,
    ) -> Result<PromotedValue, PromotionError> {
        let hw = value.abs() * (1.0 - confidence) + 1e-10;
        match target {
            UncertaintyLevel::Point => Ok(PromotedValue::Point { value, confidence }),
            UncertaintyLevel::Interval => Ok(PromotedValue::Interval {
                lower: value - hw,
                upper: value + hw,
            }),
            UncertaintyLevel::Fuzzy => Ok(PromotedValue::Fuzzy {
                support_lower: value - hw * 1.5,
                support_upper: value + hw * 1.5,
                peak: value,
                alpha_cut: confidence,
            }),
            UncertaintyLevel::Affine => Ok(PromotedValue::Affine {
                center: value,
                noise_terms: vec![(0, hw)],
            }),
            UncertaintyLevel::DempsterShafer => Ok(PromotedValue::DempsterShafer {
                focal_elements: vec![(value - hw, value + hw, confidence)],
            }),
            UncertaintyLevel::Distribution => {
                let std = hw / 1.96;
                let samples = self.generate_normal_samples(value, std);
                Ok(PromotedValue::Distribution {
                    samples,
                    mean: value,
                    variance: std * std,
                })
            }
            UncertaintyLevel::Particles => {
                let std = hw / 1.96;
                let particles = self.generate_normal_samples_n(value, std, self.default_particles);
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles {
                    particles,
                    weights,
                    effective_sample_size: self.default_particles as f64,
                })
            }
            UncertaintyLevel::Bootstrap => {
                // Generate bootstrap-style uncertainty from point estimate
                let std = hw / 1.96;
                let bootstrap_se = std / (self.default_particles as f64).sqrt();
                Ok(PromotedValue::Bootstrap {
                    estimate: value,
                    percentile_lower: value - hw,
                    percentile_upper: value + hw,
                    bootstrap_se,
                    bias: 0.0,
                    n_samples: self.default_particles as u32,
                })
            }
            UncertaintyLevel::CrossValidation => {
                // Cannot directly promote to cross-validation (requires model)
                Err(PromotionError::InsufficientInfo {
                    from: UncertaintyLevel::Point,
                    to: target,
                    reason: "Cross-validation requires model fitting".to_string(),
                })
            }
        }
    }

    pub fn promote_interval(
        &self,
        lower: f64,
        upper: f64,
        target: UncertaintyLevel,
    ) -> Result<PromotedValue, PromotionError> {
        if target == UncertaintyLevel::Point {
            return Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::Interval,
                to: target,
            });
        }
        let center = (lower + upper) / 2.0;
        let hw = (upper - lower) / 2.0;
        match target {
            UncertaintyLevel::Interval => Ok(PromotedValue::Interval { lower, upper }),
            UncertaintyLevel::Fuzzy => Ok(PromotedValue::Fuzzy {
                support_lower: lower - hw * 0.1,
                support_upper: upper + hw * 0.1,
                peak: center,
                alpha_cut: 1.0,
            }),
            UncertaintyLevel::Affine => Ok(PromotedValue::Affine {
                center,
                noise_terms: vec![(0, hw)],
            }),
            UncertaintyLevel::DempsterShafer => Ok(PromotedValue::DempsterShafer {
                focal_elements: vec![(lower, upper, 1.0)],
            }),
            UncertaintyLevel::Distribution => {
                let samples: Vec<f64> = (0..self.default_samples)
                    .map(|i| lower + (upper - lower) * (i as f64 / self.default_samples as f64))
                    .collect();
                Ok(PromotedValue::Distribution {
                    samples,
                    mean: center,
                    variance: (upper - lower).powi(2) / 12.0,
                })
            }
            UncertaintyLevel::Particles => {
                let particles: Vec<f64> = (0..self.default_particles)
                    .map(|i| lower + (upper - lower) * (i as f64 / self.default_particles as f64))
                    .collect();
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles {
                    particles,
                    weights,
                    effective_sample_size: self.default_particles as f64,
                })
            }
            _ => Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::Interval,
                to: target,
            }),
        }
    }

    fn generate_normal_samples(&self, mean: f64, std: f64) -> Vec<f64> {
        self.generate_normal_samples_n(mean, std, self.default_samples)
    }

    fn generate_normal_samples_n(&self, mean: f64, std: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let u = (i as f64 + 0.5) / n as f64;
                mean + Self::inv_norm(u) * std
            })
            .collect()
    }

    fn inv_norm(p: f64) -> f64 {
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }
        let pa = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * pa.ln()).sqrt();
        let z = t
            - (2.515517 + 0.802853 * t + 0.010328 * t * t)
                / (1.0 + 1.432788 * t + 0.189269 * t * t + 0.001308 * t * t * t);
        if p > 0.5 { -z } else { z }
    }

    /// Promote a Bootstrap value to a target uncertainty level
    pub fn promote_bootstrap(
        &self,
        estimate: f64,
        percentile_lower: f64,
        percentile_upper: f64,
        bootstrap_se: f64,
        bias: f64,
        n_samples: u32,
        target: UncertaintyLevel,
    ) -> Result<PromotedValue, PromotionError> {
        match target {
            UncertaintyLevel::Point => Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::Bootstrap,
                to: target,
            }),
            UncertaintyLevel::Interval => Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::Bootstrap,
                to: target,
            }),
            UncertaintyLevel::Fuzzy => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::Bootstrap,
                to: target,
            }),
            UncertaintyLevel::Bootstrap => Ok(PromotedValue::Bootstrap {
                estimate,
                percentile_lower,
                percentile_upper,
                bootstrap_se,
                bias,
                n_samples,
            }),
            UncertaintyLevel::CrossValidation => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::Bootstrap,
                to: target,
            }),
            UncertaintyLevel::Affine => {
                // Convert bootstrap CI to affine form
                let center = estimate + bias;
                let hw = (percentile_upper - percentile_lower) / 2.0;
                Ok(PromotedValue::Affine {
                    center,
                    noise_terms: vec![(0, hw)],
                })
            }
            UncertaintyLevel::DempsterShafer => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::Bootstrap,
                to: target,
            }),
            UncertaintyLevel::Distribution => {
                // Generate samples from bootstrap summary statistics
                let samples = self.generate_normal_samples(estimate + bias, bootstrap_se);
                Ok(PromotedValue::Distribution {
                    samples,
                    mean: estimate + bias,
                    variance: bootstrap_se * bootstrap_se,
                })
            }
            UncertaintyLevel::Particles => {
                let particles = self.generate_normal_samples_n(
                    estimate + bias,
                    bootstrap_se,
                    self.default_particles,
                );
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles {
                    particles,
                    weights,
                    effective_sample_size: self.default_particles as f64,
                })
            }
        }
    }

    /// Promote a CrossValidation value to a target uncertainty level
    pub fn promote_cv(
        &self,
        mean: f64,
        std_error: f64,
        std_dev: f64,
        k: u32,
        target: UncertaintyLevel,
    ) -> Result<PromotedValue, PromotionError> {
        match target {
            UncertaintyLevel::Point => Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::Interval => Err(PromotionError::CannotDemote {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::Fuzzy => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::Bootstrap => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::CrossValidation => Ok(PromotedValue::CrossValidation {
                mean,
                std_error,
                std_dev,
                k,
            }),
            UncertaintyLevel::Affine => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::DempsterShafer => Err(PromotionError::IncompatiblePath {
                from: UncertaintyLevel::CrossValidation,
                to: target,
            }),
            UncertaintyLevel::Distribution => {
                // CV uncertainty represents model variance; generate samples
                let samples = self.generate_normal_samples(mean, std_dev);
                Ok(PromotedValue::Distribution {
                    samples,
                    mean,
                    variance: std_dev * std_dev,
                })
            }
            UncertaintyLevel::Particles => {
                let particles =
                    self.generate_normal_samples_n(mean, std_dev, self.default_particles);
                let weights = vec![1.0 / self.default_particles as f64; self.default_particles];
                Ok(PromotedValue::Particles {
                    particles,
                    weights,
                    effective_sample_size: self.default_particles as f64,
                })
            }
        }
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
        // Interval and Fuzzy are on different branches - their meet is Point (greatest lower bound)
        assert_eq!(
            l.meet(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Point
        );
        // Interval and Fuzzy are on different branches - their join is Distribution (least upper bound)
        assert_eq!(
            l.join(UncertaintyLevel::Interval, UncertaintyLevel::Fuzzy),
            UncertaintyLevel::Distribution
        );
    }

    #[test]
    fn test_promotion() {
        let p = Promoter::new().with_samples(100);
        let r = p
            .promote_point(10.0, 0.95, UncertaintyLevel::Distribution)
            .unwrap();
        if let PromotedValue::Distribution { samples, mean, .. } = r {
            assert_eq!(samples.len(), 100);
            assert!((mean - 10.0).abs() < 0.01);
        }
    }

    // ========================================================================
    // Bootstrap tests
    // ========================================================================

    #[test]
    fn test_bootstrap_level() {
        assert_eq!(UncertaintyLevel::Bootstrap.height(), 2);
        assert!(UncertaintyLevel::Bootstrap.can_promote_to(UncertaintyLevel::Distribution));
        assert!(UncertaintyLevel::Bootstrap.can_promote_to(UncertaintyLevel::Particles));
        assert!(!UncertaintyLevel::Bootstrap.can_promote_to(UncertaintyLevel::Point));
        assert!(!UncertaintyLevel::Bootstrap.can_promote_to(UncertaintyLevel::CrossValidation));
    }

    #[test]
    fn test_bootstrap_parse() {
        assert_eq!(
            UncertaintyLevel::parse("bootstrap"),
            Some(UncertaintyLevel::Bootstrap)
        );
        assert_eq!(
            UncertaintyLevel::parse("boot"),
            Some(UncertaintyLevel::Bootstrap)
        );
        assert_eq!(
            UncertaintyLevel::parse("resampling"),
            Some(UncertaintyLevel::Bootstrap)
        );
    }

    #[test]
    fn test_promote_bootstrap_to_distribution() {
        let p = Promoter::new().with_samples(100);
        let r = p
            .promote_bootstrap(
                10.0, // estimate
                9.5,  // percentile_lower
                10.5, // percentile_upper
                0.25, // bootstrap_se
                0.0,  // bias
                1000, // n_samples
                UncertaintyLevel::Distribution,
            )
            .unwrap();

        if let PromotedValue::Distribution { samples, mean, .. } = r {
            assert_eq!(samples.len(), 100);
            assert!((mean - 10.0).abs() < 0.01);
        } else {
            panic!("Expected Distribution");
        }
    }

    #[test]
    fn test_bootstrap_promoted_value() {
        let boot = PromotedValue::Bootstrap {
            estimate: 10.0,
            percentile_lower: 9.5,
            percentile_upper: 10.5,
            bootstrap_se: 0.25,
            bias: 0.01,
            n_samples: 1000,
        };

        assert_eq!(boot.level(), UncertaintyLevel::Bootstrap);
        assert!((boot.point_estimate() - 10.0).abs() < 1e-10);
        let (lo, hi) = boot.bounds();
        assert!((lo - 9.5).abs() < 1e-10);
        assert!((hi - 10.5).abs() < 1e-10);
    }

    // ========================================================================
    // CrossValidation tests
    // ========================================================================

    #[test]
    fn test_cv_level() {
        assert_eq!(UncertaintyLevel::CrossValidation.height(), 2);
        assert!(UncertaintyLevel::CrossValidation.can_promote_to(UncertaintyLevel::Distribution));
        assert!(UncertaintyLevel::CrossValidation.can_promote_to(UncertaintyLevel::Particles));
        assert!(!UncertaintyLevel::CrossValidation.can_promote_to(UncertaintyLevel::Point));
        assert!(!UncertaintyLevel::CrossValidation.can_promote_to(UncertaintyLevel::Bootstrap));
    }

    #[test]
    fn test_cv_parse() {
        assert_eq!(
            UncertaintyLevel::parse("crossvalidation"),
            Some(UncertaintyLevel::CrossValidation)
        );
        assert_eq!(
            UncertaintyLevel::parse("cv"),
            Some(UncertaintyLevel::CrossValidation)
        );
        assert_eq!(
            UncertaintyLevel::parse("kfold"),
            Some(UncertaintyLevel::CrossValidation)
        );
    }

    #[test]
    fn test_promote_cv_to_distribution() {
        let p = Promoter::new().with_samples(100);
        let r = p
            .promote_cv(
                0.85, // mean
                0.01, // std_error
                0.02, // std_dev
                5,    // k
                UncertaintyLevel::Distribution,
            )
            .unwrap();

        if let PromotedValue::Distribution { samples, mean, .. } = r {
            assert_eq!(samples.len(), 100);
            assert!((mean - 0.85).abs() < 0.01);
        } else {
            panic!("Expected Distribution");
        }
    }

    #[test]
    fn test_cv_promoted_value() {
        let cv = PromotedValue::CrossValidation {
            mean: 0.85,
            std_error: 0.01,
            std_dev: 0.02,
            k: 5,
        };

        assert_eq!(cv.level(), UncertaintyLevel::CrossValidation);
        assert!((cv.point_estimate() - 0.85).abs() < 1e-10);
        let (lo, hi) = cv.bounds();
        assert!(lo < 0.85);
        assert!(hi > 0.85);
    }

    #[test]
    fn test_bootstrap_cv_incomparable() {
        let l = PromotionLattice::new();
        // Bootstrap and CV are on different branches - incomparable
        assert_eq!(
            l.meet(
                UncertaintyLevel::Bootstrap,
                UncertaintyLevel::CrossValidation
            ),
            UncertaintyLevel::Point
        );
        assert_eq!(
            l.join(
                UncertaintyLevel::Bootstrap,
                UncertaintyLevel::CrossValidation
            ),
            UncertaintyLevel::Distribution
        );
    }
}
