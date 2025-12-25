//! Decay Functions for Temporal Knowledge
//!
//! This module implements confidence decay over time. As knowledge ages,
//! its reliability decreases according to various decay models.
//!
//! # General Model
//!
//! ```text
//! ε(t) = ε₀ × D(t - t₀)
//!
//! where:
//!   ε₀ = initial confidence
//!   t₀ = creation timestamp
//!   D : ℝ⁺ → [0,1] = decay function
//! ```
//!
//! # Available Decay Functions
//!
//! | Function | Formula | Typical Use |
//! |----------|---------|-------------|
//! | Exponential | D(Δt) = e^(-λ·Δt) | Scientific literature |
//! | Step | D(Δt) = 1 if Δt≤τ, else 0 | Lab tests |
//! | Linear | D(Δt) = max(0, 1-Δt/τ) | Simple interpolation |
//! | Sigmoid | D(Δt) = 1/(1+e^(k(Δt-τ))) | Smooth transition |

use chrono::Duration;
use std::fmt;
use std::sync::Arc;

/// Time unit for decay rate specification
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeUnit {
    Seconds,
    Minutes,
    Hours,
    Days,
    Weeks,
    Months,
    Years,
}

impl TimeUnit {
    /// Convert to seconds
    pub fn to_seconds(&self) -> f64 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Minutes => 60.0,
            TimeUnit::Hours => 3600.0,
            TimeUnit::Days => 86400.0,
            TimeUnit::Weeks => 604800.0,
            TimeUnit::Months => 2629746.0, // Average month
            TimeUnit::Years => 31556952.0, // Average year
        }
    }

    /// Convert a duration to this unit
    pub fn from_duration(&self, duration: Duration) -> f64 {
        duration.num_seconds() as f64 / self.to_seconds()
    }
}

impl fmt::Display for TimeUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeUnit::Seconds => write!(f, "s"),
            TimeUnit::Minutes => write!(f, "min"),
            TimeUnit::Hours => write!(f, "h"),
            TimeUnit::Days => write!(f, "d"),
            TimeUnit::Weeks => write!(f, "w"),
            TimeUnit::Months => write!(f, "mo"),
            TimeUnit::Years => write!(f, "y"),
        }
    }
}

/// Decay function specification
pub enum DecayFunction {
    /// D(Δt) = e^(-λ·Δt)
    /// Half-life = ln(2)/λ
    Exponential { lambda: f64, time_unit: TimeUnit },

    /// D(Δt) = 1 if Δt ≤ τ, else 0
    Step { validity_period: Duration },

    /// D(Δt) = max(0, 1 - Δt/τ)
    Linear { full_decay_period: Duration },

    /// D(Δt) = 1 / (1 + e^(k·(Δt-τ)))
    /// τ = midpoint (50% decay)
    /// k = sharpness of transition
    Sigmoid { midpoint: Duration, sharpness: f64 },

    /// No decay - always returns 1.0
    None,

    /// Custom decay function
    Custom {
        name: String,
        evaluator: Arc<dyn Fn(Duration) -> f64 + Send + Sync>,
    },

    /// Product of two decay functions
    Product {
        d1: Box<DecayFunction>,
        d2: Box<DecayFunction>,
    },
}

impl DecayFunction {
    /// Create an exponential decay function
    pub fn exponential(lambda: f64, time_unit: TimeUnit) -> Self {
        DecayFunction::Exponential { lambda, time_unit }
    }

    /// Create exponential decay from half-life
    pub fn from_half_life(half_life: Duration, time_unit: TimeUnit) -> Self {
        let half_life_units = time_unit.from_duration(half_life);
        let lambda = 2.0_f64.ln() / half_life_units;
        DecayFunction::Exponential { lambda, time_unit }
    }

    /// Create a step decay function
    pub fn step(validity_period: Duration) -> Self {
        DecayFunction::Step { validity_period }
    }

    /// Create a linear decay function
    pub fn linear(full_decay_period: Duration) -> Self {
        DecayFunction::Linear { full_decay_period }
    }

    /// Create a sigmoid decay function
    pub fn sigmoid(midpoint: Duration, sharpness: f64) -> Self {
        DecayFunction::Sigmoid {
            midpoint,
            sharpness,
        }
    }

    /// Create a no-decay function
    pub fn none() -> Self {
        DecayFunction::None
    }

    /// Create a custom decay function
    pub fn custom<F>(name: impl Into<String>, f: F) -> Self
    where
        F: Fn(Duration) -> f64 + Send + Sync + 'static,
    {
        DecayFunction::Custom {
            name: name.into(),
            evaluator: Arc::new(f),
        }
    }

    /// Evaluate the decay function at a given time delta
    pub fn evaluate(&self, delta: Duration) -> f64 {
        let delta_secs = delta.num_seconds() as f64;

        if delta_secs < 0.0 {
            return 1.0; // Future knowledge has full confidence
        }

        match self {
            DecayFunction::Exponential { lambda, time_unit } => {
                let delta_units = delta_secs / time_unit.to_seconds();
                (-lambda * delta_units).exp()
            }

            DecayFunction::Step { validity_period } => {
                if delta <= *validity_period {
                    1.0
                } else {
                    0.0
                }
            }

            DecayFunction::Linear { full_decay_period } => {
                let period_secs = full_decay_period.num_seconds() as f64;
                if period_secs <= 0.0 {
                    1.0
                } else {
                    (1.0 - delta_secs / period_secs).max(0.0)
                }
            }

            DecayFunction::Sigmoid {
                midpoint,
                sharpness,
            } => {
                let mid_secs = midpoint.num_seconds() as f64;
                1.0 / (1.0 + (sharpness * (delta_secs - mid_secs)).exp())
            }

            DecayFunction::None => 1.0,

            DecayFunction::Custom { evaluator, .. } => evaluator(delta).clamp(0.0, 1.0),

            DecayFunction::Product { d1, d2 } => d1.evaluate(delta) * d2.evaluate(delta),
        }
    }

    /// Compute the half-life (time for 50% decay)
    pub fn half_life(&self) -> Option<Duration> {
        match self {
            DecayFunction::Exponential { lambda, time_unit } => {
                let half_life_units = 2.0_f64.ln() / lambda;
                let half_life_secs = half_life_units * time_unit.to_seconds();
                Some(Duration::seconds(half_life_secs as i64))
            }

            DecayFunction::Linear { full_decay_period } => Some(Duration::seconds(
                (full_decay_period.num_seconds() as f64 / 2.0) as i64,
            )),

            DecayFunction::Sigmoid { midpoint, .. } => Some(*midpoint),

            DecayFunction::Step { validity_period } => Some(*validity_period),

            DecayFunction::None => None,

            DecayFunction::Custom { .. } => None,

            DecayFunction::Product { d1, d2 } => {
                // Approximate: use the shorter half-life
                match (d1.half_life(), d2.half_life()) {
                    (Some(h1), Some(h2)) => Some(h1.min(h2)),
                    (Some(h), None) | (None, Some(h)) => Some(h),
                    (None, None) => None,
                }
            }
        }
    }

    /// Time until confidence drops below threshold
    pub fn time_to_threshold(&self, threshold: f64) -> Option<Duration> {
        if threshold <= 0.0 || threshold >= 1.0 {
            return None;
        }

        match self {
            DecayFunction::Exponential { lambda, time_unit } => {
                // e^(-λt) = threshold
                // t = -ln(threshold) / λ
                let t_units = -(threshold.ln()) / lambda;
                let t_secs = t_units * time_unit.to_seconds();
                Some(Duration::seconds(t_secs as i64))
            }

            DecayFunction::Step { validity_period } => {
                if threshold < 1.0 {
                    Some(*validity_period)
                } else {
                    Some(Duration::zero())
                }
            }

            DecayFunction::Linear { full_decay_period } => {
                // 1 - t/τ = threshold
                // t = τ × (1 - threshold)
                let ratio = 1.0 - threshold;
                let t_secs = full_decay_period.num_seconds() as f64 * ratio;
                Some(Duration::seconds(t_secs as i64))
            }

            DecayFunction::Sigmoid {
                midpoint,
                sharpness,
            } => {
                // 1 / (1 + e^(k(t-τ))) = threshold
                // 1/threshold - 1 = e^(k(t-τ))
                // t = τ + ln(1/threshold - 1) / k
                let t_secs =
                    midpoint.num_seconds() as f64 + (1.0 / threshold - 1.0).ln() / sharpness;
                Some(Duration::seconds(t_secs as i64))
            }

            DecayFunction::None => None,

            _ => None,
        }
    }

    /// Product of two decay functions
    pub fn product(d1: &DecayFunction, d2: &DecayFunction) -> DecayFunction {
        DecayFunction::Product {
            d1: Box::new(d1.clone()),
            d2: Box::new(d2.clone()),
        }
    }

    /// Chain two decay functions (apply d2 after d1's half-life)
    pub fn chain(d1: DecayFunction, d2: DecayFunction) -> DecayFunction {
        let d1_half = d1.half_life().unwrap_or(Duration::days(365));
        let d1_clone = d1.clone();

        DecayFunction::Custom {
            name: format!("chain({:?}, {:?})", d1, d2),
            evaluator: Arc::new(move |delta| {
                if delta < d1_half {
                    d1_clone.evaluate(delta)
                } else {
                    let remaining = delta - d1_half;
                    d1_clone.evaluate(d1_half) * d2.evaluate(remaining)
                }
            }),
        }
    }
}

impl fmt::Debug for DecayFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecayFunction::Exponential { lambda, time_unit } => {
                write!(f, "Exponential(λ={:.4}/{})", lambda, time_unit)
            }
            DecayFunction::Step { validity_period } => {
                write!(f, "Step(τ={}d)", validity_period.num_days())
            }
            DecayFunction::Linear { full_decay_period } => {
                write!(f, "Linear(τ={}d)", full_decay_period.num_days())
            }
            DecayFunction::Sigmoid {
                midpoint,
                sharpness,
            } => {
                write!(f, "Sigmoid(τ={}d, k={})", midpoint.num_days(), sharpness)
            }
            DecayFunction::None => write!(f, "None"),
            DecayFunction::Custom { name, .. } => write!(f, "Custom({})", name),
            DecayFunction::Product { d1, d2 } => write!(f, "Product({:?}, {:?})", d1, d2),
        }
    }
}

// Manual Clone for Custom variant
impl Clone for DecayFunction {
    fn clone(&self) -> Self {
        match self {
            DecayFunction::Exponential { lambda, time_unit } => DecayFunction::Exponential {
                lambda: *lambda,
                time_unit: *time_unit,
            },
            DecayFunction::Step { validity_period } => DecayFunction::Step {
                validity_period: *validity_period,
            },
            DecayFunction::Linear { full_decay_period } => DecayFunction::Linear {
                full_decay_period: *full_decay_period,
            },
            DecayFunction::Sigmoid {
                midpoint,
                sharpness,
            } => DecayFunction::Sigmoid {
                midpoint: *midpoint,
                sharpness: *sharpness,
            },
            DecayFunction::None => DecayFunction::None,
            DecayFunction::Custom { name, evaluator } => DecayFunction::Custom {
                name: name.clone(),
                evaluator: evaluator.clone(),
            },
            DecayFunction::Product { d1, d2 } => DecayFunction::Product {
                d1: d1.clone(),
                d2: d2.clone(),
            },
        }
    }
}

/// Predefined decay functions for common knowledge types
pub mod presets {
    use super::*;

    /// Scientific literature: 5-15% decay per year
    pub fn scientific_literature() -> DecayFunction {
        DecayFunction::exponential(0.10, TimeUnit::Years)
    }

    /// Clinical guidelines: valid for 3-5 years
    pub fn clinical_guidelines() -> DecayFunction {
        DecayFunction::step(Duration::days(4 * 365))
    }

    /// Patient data: 50-200% decay per year
    pub fn patient_data() -> DecayFunction {
        DecayFunction::exponential(1.0, TimeUnit::Years)
    }

    /// Lab results: valid for 30-90 days
    pub fn lab_results() -> DecayFunction {
        DecayFunction::step(Duration::days(60))
    }

    /// Vital signs: rapid decay (10-100 per hour)
    pub fn vital_signs() -> DecayFunction {
        DecayFunction::exponential(50.0, TimeUnit::Hours)
    }

    /// Market prices: rapid decay (1-10 per day)
    pub fn market_prices() -> DecayFunction {
        DecayFunction::exponential(5.0, TimeUnit::Days)
    }

    /// LLM-generated content: 10-50% decay per month
    pub fn llm_generated() -> DecayFunction {
        DecayFunction::exponential(0.3, TimeUnit::Months)
    }

    /// Physical constants: no decay
    pub fn physical_constants() -> DecayFunction {
        DecayFunction::None
    }

    /// Mathematical facts: no decay
    pub fn mathematical() -> DecayFunction {
        DecayFunction::None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exponential_decay() {
        let decay = DecayFunction::exponential(1.0, TimeUnit::Years);

        // At t=0, decay = 1
        assert!((decay.evaluate(Duration::zero()) - 1.0).abs() < 1e-10);

        // At t=1 year, decay = e^(-1) ≈ 0.368
        let one_year = Duration::days(365);
        let expected = (-1.0_f64).exp();
        assert!((decay.evaluate(one_year) - expected).abs() < 0.01);
    }

    #[test]
    fn test_half_life() {
        let decay = DecayFunction::exponential(0.693, TimeUnit::Years);

        // Half-life should be approximately 1 year
        let half_life = decay.half_life().unwrap();
        let half_life_years = half_life.num_days() as f64 / 365.0;
        assert!((half_life_years - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_from_half_life() {
        let half_life = Duration::days(365);
        let decay = DecayFunction::from_half_life(half_life, TimeUnit::Years);

        // At half-life, decay should be 0.5
        let at_half = decay.evaluate(half_life);
        assert!((at_half - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_step_decay() {
        let decay = DecayFunction::step(Duration::days(30));

        // Within validity: 1.0
        assert_eq!(decay.evaluate(Duration::days(15)), 1.0);

        // At boundary: 1.0
        assert_eq!(decay.evaluate(Duration::days(30)), 1.0);

        // After validity: 0.0
        assert_eq!(decay.evaluate(Duration::days(31)), 0.0);
    }

    #[test]
    fn test_linear_decay() {
        let decay = DecayFunction::linear(Duration::days(100));

        // At t=0: 1.0
        assert!((decay.evaluate(Duration::zero()) - 1.0).abs() < 1e-10);

        // At t=50: 0.5
        assert!((decay.evaluate(Duration::days(50)) - 0.5).abs() < 0.01);

        // At t=100: 0.0
        assert!((decay.evaluate(Duration::days(100)) - 0.0).abs() < 0.01);

        // After full decay: 0.0
        assert_eq!(decay.evaluate(Duration::days(150)), 0.0);
    }

    #[test]
    fn test_sigmoid_decay() {
        let decay = DecayFunction::sigmoid(Duration::days(30), 0.1);

        // At t=0: close to 1.0
        let at_zero = decay.evaluate(Duration::zero());
        assert!(at_zero > 0.9);

        // At midpoint: 0.5
        let at_mid = decay.evaluate(Duration::days(30));
        assert!((at_mid - 0.5).abs() < 0.01);

        // After midpoint: close to 0.0
        let at_far = decay.evaluate(Duration::days(100));
        assert!(at_far < 0.1);
    }

    #[test]
    fn test_no_decay() {
        let decay = DecayFunction::none();

        assert_eq!(decay.evaluate(Duration::zero()), 1.0);
        assert_eq!(decay.evaluate(Duration::days(1000)), 1.0);
        assert_eq!(decay.evaluate(Duration::days(36500)), 1.0);
    }

    #[test]
    fn test_product_decay() {
        let d1 = DecayFunction::exponential(0.5, TimeUnit::Years);
        let d2 = DecayFunction::exponential(0.3, TimeUnit::Years);
        let product = DecayFunction::product(&d1, &d2);

        let one_year = Duration::days(365);
        let expected = (-0.5_f64).exp() * (-0.3_f64).exp();
        assert!((product.evaluate(one_year) - expected).abs() < 0.01);
    }

    #[test]
    fn test_time_to_threshold() {
        let decay = DecayFunction::exponential(1.0, TimeUnit::Years);

        // Time to 50% (half-life)
        let to_50 = decay.time_to_threshold(0.5).unwrap();
        let to_50_years = to_50.num_days() as f64 / 365.0;
        assert!((to_50_years - 0.693).abs() < 0.1);

        // Time to 10%
        let to_10 = decay.time_to_threshold(0.1).unwrap();
        let to_10_years = to_10.num_days() as f64 / 365.0;
        assert!((to_10_years - 2.3).abs() < 0.1);
    }

    #[test]
    fn test_presets() {
        // Scientific literature: slow decay
        let sci = presets::scientific_literature();
        assert!(sci.evaluate(Duration::days(365)) > 0.8);

        // Vital signs: rapid decay
        let vital = presets::vital_signs();
        assert!(vital.evaluate(Duration::hours(1)) < 0.1);

        // Physical constants: no decay
        let phys = presets::physical_constants();
        assert_eq!(phys.evaluate(Duration::days(36500)), 1.0);
    }

    #[test]
    fn test_negative_duration() {
        let decay = DecayFunction::exponential(1.0, TimeUnit::Years);

        // Negative duration (future knowledge) should have full confidence
        let future = Duration::days(-100);
        assert_eq!(decay.evaluate(future), 1.0);
    }
}
