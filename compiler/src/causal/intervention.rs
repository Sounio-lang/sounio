//! Intervention Types and the do() Operator
//!
//! Implements Pearl's do-calculus interventions:
//! - Atomic interventions: do(X=x)
//! - Conditional interventions: do(X=x | Z=z)
//! - Stochastic interventions: do(X ~ P')
//! - Dynamic interventions: do(X = f(PA_X))

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use super::identification::IdentificationMethod;
use crate::epistemic::composition::ConfidenceValue;

/// Intervention specification
#[derive(Clone)]
pub struct Intervention<T> {
    /// Target variable to intervene on
    pub target: String,
    /// Value to set
    pub value: T,
    /// Type of intervention
    pub intervention_type: InterventionType,
}

impl<T: Clone> Intervention<T> {
    /// Create atomic intervention: do(X=x)
    pub fn atomic(target: impl Into<String>, value: T) -> Self {
        Intervention {
            target: target.into(),
            value,
            intervention_type: InterventionType::Atomic,
        }
    }

    /// Create conditional intervention: do(X=x | Z=z)
    pub fn conditional(
        target: impl Into<String>,
        value: T,
        condition_var: impl Into<String>,
        condition_value: f64,
    ) -> Self {
        Intervention {
            target: target.into(),
            value,
            intervention_type: InterventionType::Conditional {
                condition: condition_var.into(),
                condition_value,
            },
        }
    }

    /// Create stochastic intervention: do(X ~ distribution)
    pub fn stochastic(target: impl Into<String>, value: T, distribution: Distribution) -> Self {
        Intervention {
            target: target.into(),
            value,
            intervention_type: InterventionType::Stochastic { distribution },
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for Intervention<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Intervention")
            .field("target", &self.target)
            .field("value", &self.value)
            .field("type", &self.intervention_type)
            .finish()
    }
}

/// Types of interventions
#[derive(Clone)]
pub enum InterventionType {
    /// Atomic: do(X=x)
    Atomic,

    /// Conditional: do(X=x | Z=z)
    Conditional {
        condition: String,
        condition_value: f64,
    },

    /// Stochastic: do(X ~ P')
    Stochastic { distribution: Distribution },

    /// Dynamic: do(X = f(PA_X))
    Dynamic {
        policy: Arc<dyn Fn(&std::collections::HashMap<String, f64>) -> f64 + Send + Sync>,
    },
}

impl fmt::Debug for InterventionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InterventionType::Atomic => write!(f, "Atomic"),
            InterventionType::Conditional {
                condition,
                condition_value,
            } => {
                write!(f, "Conditional({} = {})", condition, condition_value)
            }
            InterventionType::Stochastic { distribution } => {
                write!(f, "Stochastic({:?})", distribution)
            }
            InterventionType::Dynamic { .. } => write!(f, "Dynamic(<policy>)"),
        }
    }
}

/// Result of an intervention
#[derive(Clone, Debug)]
pub struct InterventionResult<T> {
    /// The causal effect estimate
    pub effect: T,
    /// Confidence in the estimate
    pub confidence: ConfidenceValue,
    /// Method used for identification
    pub identification: IdentificationMethod,
    /// Adjustment set used (if any)
    pub adjustment_set: Option<HashSet<String>>,
    /// Bounds on effect (for partial identification)
    pub bounds: Option<(T, T)>,
}

impl<T> InterventionResult<T> {
    /// Create a new intervention result
    pub fn new(
        effect: T,
        confidence: ConfidenceValue,
        identification: IdentificationMethod,
    ) -> Self {
        InterventionResult {
            effect,
            confidence,
            identification,
            adjustment_set: None,
            bounds: None,
        }
    }

    /// Add adjustment set information
    pub fn with_adjustment_set(mut self, set: HashSet<String>) -> Self {
        self.adjustment_set = Some(set);
        self
    }

    /// Add bounds for partial identification
    pub fn with_bounds(mut self, lower: T, upper: T) -> Self {
        self.bounds = Some((lower, upper));
        self
    }
}

/// Probability distribution for stochastic interventions
#[derive(Clone, Debug)]
pub enum Distribution {
    /// Uniform distribution over [min, max]
    Uniform { min: f64, max: f64 },
    /// Normal/Gaussian distribution
    Normal { mean: f64, std: f64 },
    /// Beta distribution
    Beta { alpha: f64, beta: f64 },
    /// Bernoulli (binary)
    Bernoulli { p: f64 },
    /// Categorical
    Categorical { probs: Vec<f64> },
    /// Point mass (degenerate)
    PointMass { value: f64 },
}

impl Distribution {
    /// Get the mean of the distribution
    pub fn mean(&self) -> f64 {
        match self {
            Distribution::Uniform { min, max } => (min + max) / 2.0,
            Distribution::Normal { mean, .. } => *mean,
            Distribution::Beta { alpha, beta } => alpha / (alpha + beta),
            Distribution::Bernoulli { p } => *p,
            Distribution::Categorical { probs } => {
                probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
            }
            Distribution::PointMass { value } => *value,
        }
    }

    /// Get the variance of the distribution
    pub fn variance(&self) -> f64 {
        match self {
            Distribution::Uniform { min, max } => (max - min).powi(2) / 12.0,
            Distribution::Normal { std, .. } => std.powi(2),
            Distribution::Beta { alpha, beta } => {
                (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0))
            }
            Distribution::Bernoulli { p } => p * (1.0 - p),
            Distribution::Categorical { probs } => {
                let mean = self.mean();
                probs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| p * (i as f64 - mean).powi(2))
                    .sum()
            }
            Distribution::PointMass { .. } => 0.0,
        }
    }

    /// Sample from the distribution (using simple RNG)
    pub fn sample(&self) -> f64 {
        // Simple pseudo-random sampling
        let u: f64 = rand_simple();

        match self {
            Distribution::Uniform { min, max } => min + u * (max - min),
            Distribution::Normal { mean, std } => {
                // Box-Muller transform
                let u2: f64 = rand_simple();
                let z = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                mean + std * z
            }
            Distribution::Beta { alpha, beta } => {
                // Approximate using inverse CDF
                beta_sample(*alpha, *beta, u)
            }
            Distribution::Bernoulli { p } => {
                if u < *p {
                    1.0
                } else {
                    0.0
                }
            }
            Distribution::Categorical { probs } => {
                let mut cumsum = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    cumsum += p;
                    if u < cumsum {
                        return i as f64;
                    }
                }
                (probs.len() - 1) as f64
            }
            Distribution::PointMass { value } => *value,
        }
    }
}

/// Simple pseudo-random number generator (for testing)
fn rand_simple() -> f64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .subsec_nanos();
    (nanos as f64 % 1000.0) / 1000.0
}

/// Approximate beta distribution sampling
fn beta_sample(alpha: f64, beta: f64, u: f64) -> f64 {
    // Simple approximation using gamma function ratio
    // For more accuracy, would use proper gamma sampling
    let mean = alpha / (alpha + beta);
    let var = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
    let std = var.sqrt();

    // Clamp to [0, 1]
    (mean + std * (u - 0.5) * 2.0 * 3.0_f64.sqrt()).clamp(0.0, 1.0)
}

/// Average Treatment Effect (ATE)
#[derive(Clone, Debug)]
pub struct AverageTreatmentEffect {
    /// E[Y | do(X=1)] - E[Y | do(X=0)]
    pub ate: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Sample size used
    pub sample_size: usize,
    /// Standard error
    pub std_error: f64,
}

impl AverageTreatmentEffect {
    /// Check if effect is statistically significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        // Z-test for significance
        let z = self.ate.abs() / self.std_error;
        let z_critical = if alpha <= 0.01 {
            2.576
        } else if alpha <= 0.05 {
            1.96
        } else {
            1.645
        };
        z > z_critical
    }
}

/// Conditional Average Treatment Effect (CATE)
#[derive(Clone, Debug)]
pub struct ConditionalATE {
    /// Subgroup definition
    pub subgroup: String,
    /// Effect in this subgroup
    pub cate: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Sample size in subgroup
    pub subgroup_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_atomic_intervention() {
        let int: Intervention<f64> = Intervention::atomic("X", 1.0);
        assert_eq!(int.target, "X");
        assert_eq!(int.value, 1.0);
        assert!(matches!(int.intervention_type, InterventionType::Atomic));
    }

    #[test]
    fn test_conditional_intervention() {
        let int: Intervention<f64> = Intervention::conditional("X", 1.0, "Z", 0.5);
        assert_eq!(int.target, "X");
        assert!(matches!(
            int.intervention_type,
            InterventionType::Conditional { .. }
        ));
    }

    #[test]
    fn test_distribution_mean() {
        let uniform = Distribution::Uniform { min: 0.0, max: 1.0 };
        assert!((uniform.mean() - 0.5).abs() < 0.001);

        let normal = Distribution::Normal {
            mean: 5.0,
            std: 1.0,
        };
        assert!((normal.mean() - 5.0).abs() < 0.001);

        let beta = Distribution::Beta {
            alpha: 2.0,
            beta: 2.0,
        };
        assert!((beta.mean() - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_intervention_result() {
        let result: InterventionResult<f64> = InterventionResult::new(
            0.5,
            ConfidenceValue::new(0.95).unwrap(),
            IdentificationMethod::Experimental,
        );
        assert_eq!(result.effect, 0.5);
        assert_eq!(result.confidence.value(), 0.95);
    }

    #[test]
    fn test_ate_significance() {
        let ate = AverageTreatmentEffect {
            ate: 0.3,
            confidence_interval: (0.1, 0.5),
            sample_size: 1000,
            std_error: 0.1,
        };

        // z = 0.3 / 0.1 = 3.0 > 1.96
        assert!(ate.is_significant(0.05));
    }
}
