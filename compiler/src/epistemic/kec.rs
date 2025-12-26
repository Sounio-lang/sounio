//! KEC (Knowledge-Entropy-Complexity) Auto-Selection
//!
//! Automatically selects the optimal uncertainty representation based on:
//! - **K**: Knowledge available (measurement quality, prior information)
//! - **E**: Entropy of the uncertainty (information content)
//! - **C**: Computational complexity constraints
//!
//! This implements the decision logic for the Sounio epistemic type system's
//! automatic model selection feature.

use super::promotion::UncertaintyLevel;
use std::collections::HashMap;

// ============================================================================
// CONFIGURATION
// ============================================================================

/// Configuration for KEC auto-selection
#[derive(Debug, Clone)]
pub struct KECConfig {
    /// Maximum computational budget (relative units)
    pub max_complexity: f64,
    
    /// Minimum acceptable confidence level
    pub min_confidence: f64,
    
    /// Whether to prefer simpler models when equivalent
    pub prefer_simplicity: bool,
    
    /// Enable correlation tracking
    pub track_correlations: bool,
    
    /// Maximum number of particles for SMC
    pub max_particles: usize,
    
    /// Entropy threshold for upgrading to Distribution
    pub entropy_threshold_distribution: f64,
    
    /// Entropy threshold for upgrading to Particles
    pub entropy_threshold_particles: f64,
    
    /// Coefficient of variation threshold for Interval
    pub cv_threshold_interval: f64,
}

impl Default for KECConfig {
    fn default() -> Self {
        Self {
            max_complexity: 1000.0,
            min_confidence: 0.95,
            prefer_simplicity: true,
            track_correlations: true,
            max_particles: 10000,
            entropy_threshold_distribution: 2.0,
            entropy_threshold_particles: 4.0,
            cv_threshold_interval: 0.1,
        }
    }
}

impl KECConfig {
    /// Scientific computing preset - high accuracy, moderate complexity
    pub fn scientific() -> Self {
        Self {
            max_complexity: 10000.0,
            min_confidence: 0.99,
            prefer_simplicity: false,
            track_correlations: true,
            max_particles: 50000,
            entropy_threshold_distribution: 1.5,
            entropy_threshold_particles: 3.0,
            cv_threshold_interval: 0.05,
        }
    }
    
    /// Real-time preset - fast execution, acceptable approximations
    pub fn realtime() -> Self {
        Self {
            max_complexity: 100.0,
            min_confidence: 0.90,
            prefer_simplicity: true,
            track_correlations: false,
            max_particles: 1000,
            entropy_threshold_distribution: 3.0,
            entropy_threshold_particles: 5.0,
            cv_threshold_interval: 0.2,
        }
    }
    
    /// PKPD modeling preset - balanced for pharmacometric applications
    pub fn pkpd() -> Self {
        Self {
            max_complexity: 5000.0,
            min_confidence: 0.95,
            prefer_simplicity: false,
            track_correlations: true,
            max_particles: 20000,
            entropy_threshold_distribution: 2.0,
            entropy_threshold_particles: 3.5,
            cv_threshold_interval: 0.1,
        }
    }
    
    /// Safety-critical preset - maximum accuracy, no shortcuts
    pub fn safety_critical() -> Self {
        Self {
            max_complexity: f64::INFINITY,
            min_confidence: 0.999,
            prefer_simplicity: false,
            track_correlations: true,
            max_particles: 100000,
            entropy_threshold_distribution: 1.0,
            entropy_threshold_particles: 2.0,
            cv_threshold_interval: 0.01,
        }
    }
}

// ============================================================================
// UNCERTAINTY METRICS
// ============================================================================

/// Metrics describing uncertainty characteristics
#[derive(Debug, Clone, Default)]
pub struct UncertaintyMetrics {
    /// Shannon entropy estimate
    pub entropy: f64,
    
    /// Coefficient of variation (std/mean)
    pub cv: f64,
    
    /// Lower bound (if known)
    pub lower_bound: Option<f64>,
    
    /// Upper bound (if known)
    pub upper_bound: Option<f64>,
    
    /// Skewness (0 = symmetric)
    pub skewness: f64,
    
    /// Kurtosis (3 = normal)
    pub kurtosis: f64,
    
    /// Is the distribution multi-modal?
    pub multimodal: bool,
    
    /// Number of modes (if multimodal)
    pub mode_count: usize,
    
    /// Are there correlations with other variables?
    pub has_correlations: bool,
    
    /// Correlation strength (0-1)
    pub correlation_strength: f64,
}

impl UncertaintyMetrics {
    /// Create from sample statistics
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self::default();
        }
        
        let n = samples.len() as f64;
        let mean: f64 = samples.iter().sum::<f64>() / n;
        
        let variance: f64 = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / n;
        let std = variance.sqrt();
        
        let cv = if mean.abs() > 1e-10 { std / mean.abs() } else { 0.0 };
        
        // Skewness
        let skewness = if std > 1e-10 {
            samples.iter()
                .map(|x| ((x - mean) / std).powi(3))
                .sum::<f64>() / n
        } else {
            0.0
        };
        
        // Kurtosis
        let kurtosis = if std > 1e-10 {
            samples.iter()
                .map(|x| ((x - mean) / std).powi(4))
                .sum::<f64>() / n
        } else {
            3.0
        };
        
        // Entropy estimate (histogram-based)
        let entropy = Self::estimate_entropy(samples);
        
        Self {
            entropy,
            cv,
            lower_bound: samples.iter().copied().reduce(f64::min),
            upper_bound: samples.iter().copied().reduce(f64::max),
            skewness,
            kurtosis,
            multimodal: false, // Would need kernel density estimation
            mode_count: 1,
            has_correlations: false,
            correlation_strength: 0.0,
        }
    }
    
    /// Estimate entropy using histogram method
    fn estimate_entropy(samples: &[f64]) -> f64 {
        if samples.len() < 2 {
            return 0.0;
        }
        
        let min = samples.iter().copied().reduce(f64::min).unwrap();
        let max = samples.iter().copied().reduce(f64::max).unwrap();
        
        if (max - min).abs() < 1e-10 {
            return 0.0;
        }
        
        // Use Sturges' rule for bin count
        let num_bins = (1.0 + (samples.len() as f64).log2()).ceil() as usize;
        let bin_width = (max - min) / num_bins as f64;
        
        let mut counts = vec![0usize; num_bins];
        for &x in samples {
            let bin = ((x - min) / bin_width).floor() as usize;
            let bin = bin.min(num_bins - 1);
            counts[bin] += 1;
        }
        
        let n = samples.len() as f64;
        let mut entropy = 0.0;
        for count in counts {
            if count > 0 {
                let p = count as f64 / n;
                entropy -= p * p.ln();
            }
        }
        
        entropy
    }
    
    /// Create for a known interval
    pub fn from_interval(lower: f64, upper: f64) -> Self {
        let width = upper - lower;
        let mean = (lower + upper) / 2.0;
        let cv = if mean.abs() > 1e-10 {
            (width / 12.0_f64.sqrt()) / mean.abs() // uniform distribution std
        } else {
            0.0
        };
        
        Self {
            entropy: width.ln().max(0.0), // entropy of uniform
            cv,
            lower_bound: Some(lower),
            upper_bound: Some(upper),
            skewness: 0.0,
            kurtosis: 1.8, // uniform kurtosis
            multimodal: false,
            mode_count: 0, // uniform has no mode
            has_correlations: false,
            correlation_strength: 0.0,
        }
    }
    
    /// Create for a point estimate (no uncertainty)
    pub fn from_point(_value: f64) -> Self {
        Self {
            entropy: 0.0,
            cv: 0.0,
            lower_bound: None,
            upper_bound: None,
            skewness: 0.0,
            kurtosis: 3.0,
            multimodal: false,
            mode_count: 1,
            has_correlations: false,
            correlation_strength: 0.0,
        }
    }
}

// ============================================================================
// COMPLEXITY METRICS
// ============================================================================

/// Metrics describing computational complexity
#[derive(Debug, Clone, Default)]
pub struct ComplexityMetrics {
    /// Number of operations in the computation graph
    pub operation_count: usize,
    
    /// Depth of the computation graph
    pub graph_depth: usize,
    
    /// Number of variables involved
    pub variable_count: usize,
    
    /// Estimated floating-point operations
    pub flop_estimate: f64,
    
    /// Memory requirements (bytes)
    pub memory_bytes: usize,
    
    /// Is the computation embarrassingly parallel?
    pub parallelizable: bool,
    
    /// Nonlinearity factor (0 = linear, 1 = highly nonlinear)
    pub nonlinearity: f64,
}

impl ComplexityMetrics {
    /// Complexity cost for each uncertainty level
    pub fn level_cost(level: UncertaintyLevel, particles: usize) -> f64 {
        match level {
            UncertaintyLevel::Point => 1.0,
            UncertaintyLevel::Interval => 2.0,
            UncertaintyLevel::Fuzzy => 10.0,
            UncertaintyLevel::Affine => 5.0 * (1.0 + 0.1 * 10.0), // base + terms
            UncertaintyLevel::DempsterShafer => 20.0,
            UncertaintyLevel::Distribution => 50.0,
            UncertaintyLevel::Particles => particles as f64,
        }
    }
    
    /// Estimate total complexity for a computation
    pub fn estimate_total(&self, level: UncertaintyLevel, particles: usize) -> f64 {
        let level_cost = Self::level_cost(level, particles);
        let ops_factor = self.operation_count as f64;
        let depth_factor = 1.0 + 0.1 * self.graph_depth as f64;
        let nonlin_factor = 1.0 + self.nonlinearity;
        
        level_cost * ops_factor * depth_factor * nonlin_factor
    }
}

// ============================================================================
// KEC SELECTOR
// ============================================================================

/// Result of KEC auto-selection
#[derive(Debug, Clone)]
pub struct KECResult {
    /// Recommended uncertainty level
    pub recommended: UncertaintyLevel,
    
    /// Confidence in this recommendation (0-1)
    pub confidence: f64,
    
    /// Reasoning for the selection
    pub reasoning: Vec<String>,
    
    /// Alternative options considered
    pub alternatives: Vec<(UncertaintyLevel, f64)>, // (level, score)
    
    /// Warnings about potential issues
    pub warnings: Vec<String>,
    
    /// Suggested configuration parameters
    pub suggested_params: HashMap<String, f64>,
}

/// Main KEC selector
pub struct KECSelector {
    config: KECConfig,
}

impl KECSelector {
    /// Create a new selector with the given configuration
    pub fn new(config: KECConfig) -> Self {
        Self { config }
    }
    
    /// Create with default configuration
    pub fn default_selector() -> Self {
        Self::new(KECConfig::default())
    }
    
    /// Select the optimal uncertainty model
    pub fn select(
        &self,
        uncertainty: &UncertaintyMetrics,
        complexity: &ComplexityMetrics,
    ) -> KECResult {
        let mut scores: HashMap<UncertaintyLevel, f64> = HashMap::new();
        let mut reasoning = Vec::new();
        let mut warnings = Vec::new();
        
        // Calculate scores for each level
        for level in [
            UncertaintyLevel::Point,
            UncertaintyLevel::Interval,
            UncertaintyLevel::Fuzzy,
            UncertaintyLevel::Affine,
            UncertaintyLevel::DempsterShafer,
            UncertaintyLevel::Distribution,
            UncertaintyLevel::Particles,
        ] {
            let score = self.score_level(level, uncertainty, complexity);
            scores.insert(level, score);
        }
        
        // Decision logic
        let mut recommended = UncertaintyLevel::Point;
        let mut best_score = f64::NEG_INFINITY;
        
        for (&level, &score) in &scores {
            if score > best_score {
                best_score = score;
                recommended = level;
            }
        }
        
        // Build reasoning
        reasoning.push(format!(
            "Entropy: {:.2} (threshold for Distribution: {:.2})",
            uncertainty.entropy, self.config.entropy_threshold_distribution
        ));
        reasoning.push(format!(
            "CV: {:.2}% (threshold for Interval: {:.2}%)",
            uncertainty.cv * 100.0, self.config.cv_threshold_interval * 100.0
        ));
        reasoning.push(format!(
            "Complexity budget: {:.0} (estimated: {:.0})",
            self.config.max_complexity,
            complexity.estimate_total(recommended, self.config.max_particles)
        ));
        
        // Add warnings
        if uncertainty.multimodal && recommended != UncertaintyLevel::Particles {
            warnings.push("Multimodal distribution detected; consider Particles".to_string());
        }
        if uncertainty.has_correlations && !self.config.track_correlations {
            warnings.push("Correlations present but not being tracked".to_string());
        }
        if complexity.nonlinearity > 0.5 && recommended == UncertaintyLevel::Interval {
            warnings.push("High nonlinearity may cause interval explosion".to_string());
        }
        
        // Suggested parameters
        let mut suggested_params = HashMap::new();
        match recommended {
            UncertaintyLevel::Particles => {
                let particles = self.suggest_particle_count(uncertainty, complexity);
                suggested_params.insert("particles".to_string(), particles as f64);
            }
            UncertaintyLevel::Affine => {
                suggested_params.insert("noise_terms".to_string(), 10.0);
            }
            _ => {}
        }
        
        // Build alternatives list
        let mut alternatives: Vec<(UncertaintyLevel, f64)> = scores.into_iter().collect();
        alternatives.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        KECResult {
            recommended,
            confidence: self.compute_confidence(&alternatives),
            reasoning,
            alternatives,
            warnings,
            suggested_params,
        }
    }
    
    /// Score a specific uncertainty level
    fn score_level(
        &self,
        level: UncertaintyLevel,
        uncertainty: &UncertaintyMetrics,
        complexity: &ComplexityMetrics,
    ) -> f64 {
        let mut score = 0.0;
        
        // Adequacy score: how well does this level capture the uncertainty?
        let adequacy = self.adequacy_score(level, uncertainty);
        score += 50.0 * adequacy;
        
        // Complexity score: can we afford this level?
        let cost = complexity.estimate_total(level, self.config.max_particles);
        let complexity_score = if cost <= self.config.max_complexity {
            1.0 - (cost / self.config.max_complexity).min(1.0) * 0.5
        } else {
            -1.0 // Penalize exceeding budget
        };
        score += 30.0 * complexity_score;
        
        // Simplicity bonus
        if self.config.prefer_simplicity {
            let simplicity = 1.0 - (level as u8 as f64 / 6.0);
            score += 20.0 * simplicity;
        }
        
        score
    }
    
    /// How well does this level capture the uncertainty?
    fn adequacy_score(&self, level: UncertaintyLevel, uncertainty: &UncertaintyMetrics) -> f64 {
        match level {
            UncertaintyLevel::Point => {
                // Only adequate if uncertainty is negligible
                if uncertainty.cv < 0.01 && uncertainty.entropy < 0.1 {
                    1.0
                } else {
                    0.1
                }
            }
            UncertaintyLevel::Interval => {
                // Good for bounded, symmetric uncertainty
                if uncertainty.cv < self.config.cv_threshold_interval
                    && uncertainty.skewness.abs() < 0.5
                    && !uncertainty.multimodal
                {
                    0.9
                } else if uncertainty.lower_bound.is_some() && uncertainty.upper_bound.is_some() {
                    0.7
                } else {
                    0.4
                }
            }
            UncertaintyLevel::Fuzzy => {
                // Good for imprecise or linguistic uncertainty
                0.6 // Generally applicable but not optimal
            }
            UncertaintyLevel::Affine => {
                // Good for moderate uncertainty with correlation tracking
                if uncertainty.cv < 0.3 && !uncertainty.multimodal {
                    if uncertainty.has_correlations {
                        0.95
                    } else {
                        0.8
                    }
                } else {
                    0.5
                }
            }
            UncertaintyLevel::DempsterShafer => {
                // Good for conflicting evidence
                0.6 // Specialized use case
            }
            UncertaintyLevel::Distribution => {
                // Good for well-characterized parametric uncertainty
                if uncertainty.entropy < self.config.entropy_threshold_distribution
                    && !uncertainty.multimodal
                {
                    0.9
                } else if uncertainty.entropy < self.config.entropy_threshold_particles {
                    0.75
                } else {
                    0.5
                }
            }
            UncertaintyLevel::Particles => {
                // Always adequate but expensive
                if uncertainty.multimodal {
                    1.0
                } else if uncertainty.entropy > self.config.entropy_threshold_particles {
                    0.95
                } else {
                    0.7 // Overkill for simple cases
                }
            }
        }
    }
    
    /// Suggest particle count based on metrics
    fn suggest_particle_count(
        &self,
        uncertainty: &UncertaintyMetrics,
        complexity: &ComplexityMetrics,
    ) -> usize {
        // Base particle count
        let base = 1000;
        
        // Scale with entropy
        let entropy_factor = (1.0 + uncertainty.entropy).min(5.0);
        
        // Scale with complexity (inverse)
        let complexity_factor = if complexity.operation_count > 0 {
            (100.0 / complexity.operation_count as f64).max(0.1).min(2.0)
        } else {
            1.0
        };
        
        let suggested = (base as f64 * entropy_factor * complexity_factor) as usize;
        suggested.min(self.config.max_particles).max(100)
    }
    
    /// Compute confidence in the recommendation
    fn compute_confidence(&self, alternatives: &[(UncertaintyLevel, f64)]) -> f64 {
        if alternatives.len() < 2 {
            return 1.0;
        }
        
        let best = alternatives[0].1;
        let second = alternatives[1].1;
        
        if best <= 0.0 {
            return 0.5;
        }
        
        // Confidence based on gap to second-best
        let gap = (best - second) / best.abs();
        (gap * 2.0).min(1.0).max(0.5)
    }
}

// ============================================================================
// AUTO-SELECTION INTEGRATION
// ============================================================================

/// Automatic model selection based on input characteristics
pub fn auto_select_model(
    samples: Option<&[f64]>,
    interval: Option<(f64, f64)>,
    config: Option<KECConfig>,
) -> KECResult {
    let uncertainty = if let Some(s) = samples {
        UncertaintyMetrics::from_samples(s)
    } else if let Some((lo, hi)) = interval {
        UncertaintyMetrics::from_interval(lo, hi)
    } else {
        UncertaintyMetrics::default()
    };
    
    let complexity = ComplexityMetrics::default();
    let selector = KECSelector::new(config.unwrap_or_default());
    
    selector.select(&uncertainty, &complexity)
}

/// Select model for a specific operation
pub fn select_for_operation(
    op: &str,
    input_levels: &[UncertaintyLevel],
    config: Option<KECConfig>,
) -> UncertaintyLevel {
    let config = config.unwrap_or_default();
    
    // Find the highest level among inputs (by height in the lattice)
    let max_level = input_levels
        .iter()
        .copied()
        .max_by_key(|l| l.height())
        .unwrap_or(UncertaintyLevel::Point);

    // Operations that may require upgrade
    match op {
        "divide" | "exp" | "log" | "pow" => {
            // Nonlinear operations may need higher precision
            if max_level == UncertaintyLevel::Interval && config.max_complexity > 100.0 {
                UncertaintyLevel::Affine
            } else {
                max_level
            }
        }
        "integrate" | "solve_ode" => {
            // Long computations benefit from affine or particles
            if max_level.height() <= UncertaintyLevel::Interval.height() && config.max_complexity > 1000.0 {
                UncertaintyLevel::Affine
            } else if max_level.height() <= UncertaintyLevel::Distribution.height() && config.max_complexity > 5000.0 {
                UncertaintyLevel::Particles
            } else {
                max_level
            }
        }
        _ => max_level,
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_kec_config_presets() {
        let scientific = KECConfig::scientific();
        assert!(scientific.max_complexity > KECConfig::realtime().max_complexity);
        assert!(scientific.min_confidence > KECConfig::realtime().min_confidence);
    }
    
    #[test]
    fn test_uncertainty_metrics_from_samples() {
        let samples: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let metrics = UncertaintyMetrics::from_samples(&samples);
        
        assert!(metrics.entropy > 0.0);
        assert!(metrics.cv > 0.0);
        assert_eq!(metrics.lower_bound, Some(0.0));
        assert_eq!(metrics.upper_bound, Some(99.0));
    }
    
    #[test]
    fn test_kec_selection_point() {
        let uncertainty = UncertaintyMetrics::from_point(42.0);
        let complexity = ComplexityMetrics::default();
        let selector = KECSelector::default_selector();
        
        let result = selector.select(&uncertainty, &complexity);
        assert_eq!(result.recommended, UncertaintyLevel::Point);
    }
    
    #[test]
    fn test_kec_selection_interval() {
        let uncertainty = UncertaintyMetrics::from_interval(10.0, 20.0);
        let complexity = ComplexityMetrics::default();
        let selector = KECSelector::default_selector();
        
        let result = selector.select(&uncertainty, &complexity);
        // Should recommend Interval or higher for bounded uncertainty
        assert!(result.recommended as u8 >= UncertaintyLevel::Interval as u8);
    }
    
    #[test]
    fn test_auto_select_from_samples() {
        let samples: Vec<f64> = vec![1.0, 1.1, 0.9, 1.05, 0.95];
        let result = auto_select_model(Some(&samples), None, None);
        
        assert!(!result.reasoning.is_empty());
        assert!(result.confidence > 0.0);
    }
    
    #[test]
    fn test_select_for_operation() {
        let level = select_for_operation(
            "divide",
            &[UncertaintyLevel::Interval],
            Some(KECConfig::scientific()),
        );
        
        // Division with interval may upgrade to affine
        assert!(level as u8 >= UncertaintyLevel::Interval as u8);
    }
}
