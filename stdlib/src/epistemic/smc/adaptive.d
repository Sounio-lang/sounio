/// Adaptive Temperature Schedules for SMC Samplers
///
/// Automatically determines the temperature sequence for SMC samplers
/// based on Effective Sample Size (ESS) criterion.
///
/// # Theory
///
/// The goal is to find temperatures 0 = β₀ < β₁ < ... < βₙ = 1 such that
/// the ESS after each reweighting step stays above a threshold.
///
/// Given current particles {xᵢ, wᵢ} at temperature βₜ, the weights after
/// moving to βₜ₊₁ are:
///
///   w̃ᵢ ∝ wᵢ · exp((βₜ₊₁ - βₜ) · log π(xᵢ))
///
/// We choose βₜ₊₁ to maintain ESS ≥ α·N where α ∈ (0,1) is the threshold.
///
/// # Algorithms
///
/// 1. **Bisection**: Binary search for next temperature
/// 2. **CESS (Conditional ESS)**: Solve for ESS = target analytically
/// 3. **Adaptive**: Combine bisection with local optimization
///
/// # References
///
/// - Jasra et al. (2011) - Inference for Lévy-driven SV models
/// - Zhou et al. (2016) - Toward automatic model comparison

module sio.epistemic.smc.adaptive

use sio.core.numeric.{f64, usize}
use sio.core.collections.{Vec}
use sio.core.option.{Option, Some, None}

// =============================================================================
// Temperature Schedule Types
// =============================================================================

/// A temperature schedule for SMC
pub struct TemperatureSchedule {
    /// Sequence of temperatures [0, ..., 1]
    temperatures: Vec<f64>,
    /// ESS values at each step
    ess_values: Vec<f64>,
    /// Number of resampling events
    resample_count: usize,
    /// Total iterations
    total_steps: usize,
}

impl TemperatureSchedule {
    /// Create a fixed linear schedule
    pub fn linear(n_steps: usize) -> Self {
        let temperatures: Vec<f64> = (0..=n_steps)
            .map(|i| i as f64 / n_steps as f64)
            .collect();

        TemperatureSchedule {
            temperatures,
            ess_values: Vec::new(),
            resample_count: 0,
            total_steps: n_steps,
        }
    }

    /// Create a fixed geometric schedule (more steps near 0)
    pub fn geometric(n_steps: usize, base: f64) -> Self {
        let temperatures: Vec<f64> = (0..=n_steps)
            .map(|i| {
                let t = i as f64 / n_steps as f64;
                // Transform: more resolution near 0
                1.0 - (1.0 - t).powf(base)
            })
            .collect();

        TemperatureSchedule {
            temperatures,
            ess_values: Vec::new(),
            resample_count: 0,
            total_steps: n_steps,
        }
    }

    /// Create empty schedule for adaptive filling
    pub fn adaptive() -> Self {
        TemperatureSchedule {
            temperatures: vec![0.0],
            ess_values: Vec::new(),
            resample_count: 0,
            total_steps: 0,
        }
    }

    /// Current temperature
    pub fn current(&self) -> f64 {
        *self.temperatures.last().unwrap_or(&0.0)
    }

    /// Is schedule complete?
    pub fn is_complete(&self) -> bool {
        self.current() >= 1.0 - 1e-10
    }

    /// Add a new temperature
    pub fn push(&mut self, temp: f64) {
        self.temperatures.push(temp.min(1.0));
        self.total_steps += 1;
    }

    /// Record ESS
    pub fn record_ess(&mut self, ess: f64) {
        self.ess_values.push(ess);
    }

    /// Record resampling event
    pub fn record_resample(&mut self) {
        self.resample_count += 1;
    }

    /// Get all temperatures
    pub fn temperatures(&self) -> &[f64] {
        &self.temperatures
    }

    /// Summary statistics
    pub fn summary(&self) -> ScheduleSummary {
        ScheduleSummary {
            n_temperatures: self.temperatures.len(),
            n_resamples: self.resample_count,
            min_ess: self.ess_values.iter().copied().fold(f64::INFINITY, f64::min),
            mean_ess: if self.ess_values.is_empty() { 0.0 }
                      else { self.ess_values.iter().sum::<f64>() / self.ess_values.len() as f64 },
            temperature_schedule: self.temperatures.clone(),
        }
    }
}

/// Summary of schedule execution
pub struct ScheduleSummary {
    pub n_temperatures: usize,
    pub n_resamples: usize,
    pub min_ess: f64,
    pub mean_ess: f64,
    pub temperature_schedule: Vec<f64>,
}

// =============================================================================
// Adaptive Temperature Selection
// =============================================================================

/// Configuration for adaptive temperature selection
pub struct AdaptiveConfig {
    /// Target ESS ratio (0 < α < 1)
    pub ess_threshold: f64,
    /// Minimum temperature increment
    pub min_increment: f64,
    /// Maximum temperature increment
    pub max_increment: f64,
    /// Bisection tolerance
    pub tolerance: f64,
    /// Maximum bisection iterations
    pub max_bisection_iters: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        AdaptiveConfig {
            ess_threshold: 0.5,
            min_increment: 1e-6,
            max_increment: 0.5,
            tolerance: 1e-4,
            max_bisection_iters: 50,
        }
    }
}

/// Adaptive temperature selector using bisection
pub struct AdaptiveTemperatureSelector {
    config: AdaptiveConfig,
}

impl AdaptiveTemperatureSelector {
    pub fn new(config: AdaptiveConfig) -> Self {
        AdaptiveTemperatureSelector { config }
    }

    pub fn with_ess_threshold(mut self, threshold: f64) -> Self {
        self.config.ess_threshold = threshold.clamp(0.1, 0.99);
        self
    }

    /// Find next temperature using bisection on ESS
    ///
    /// # Arguments
    /// * `current_temp` - Current temperature βₜ
    /// * `log_likelihoods` - Log-likelihood values for each particle
    /// * `current_weights` - Current normalized weights
    /// * `n_particles` - Total number of particles
    ///
    /// # Returns
    /// Next temperature βₜ₊₁ such that ESS ≈ threshold * N
    pub fn find_next_temperature(
        &self,
        current_temp: f64,
        log_likelihoods: &[f64],
        current_weights: &[f64],
        n_particles: usize,
    ) -> f64 {
        let target_ess = self.config.ess_threshold * n_particles as f64;

        // Check if we can go directly to 1.0
        let ess_at_one = self.compute_ess_at_temp(
            current_temp, 1.0, log_likelihoods, current_weights
        );
        if ess_at_one >= target_ess {
            return 1.0;
        }

        // Bisection search
        var lo = current_temp + self.config.min_increment;
        var hi = (current_temp + self.config.max_increment).min(1.0);

        // Ensure hi gives ESS below target
        while self.compute_ess_at_temp(current_temp, hi, log_likelihoods, current_weights) >= target_ess
              && hi < 1.0 {
            hi = (hi + (1.0 - hi) * 0.5).min(1.0);
        }

        // Bisection
        for _ in 0..self.config.max_bisection_iters {
            if hi - lo < self.config.tolerance {
                break;
            }

            let mid = (lo + hi) / 2.0;
            let ess_mid = self.compute_ess_at_temp(current_temp, mid, log_likelihoods, current_weights);

            if ess_mid > target_ess {
                lo = mid;
            } else {
                hi = mid;
            }
        }

        lo  // Return conservative (lower) bound
    }

    /// Compute ESS if we move from current_temp to next_temp
    fn compute_ess_at_temp(
        &self,
        current_temp: f64,
        next_temp: f64,
        log_likelihoods: &[f64],
        current_weights: &[f64],
    ) -> f64 {
        let delta = next_temp - current_temp;

        // Compute incremental weights
        let max_log_w: f64 = log_likelihoods.iter()
            .map(|ll| delta * ll)
            .fold(f64::NEG_INFINITY, f64::max);

        var sum_w = 0.0;
        var sum_w_sq = 0.0;

        for (w, ll) in current_weights.iter().zip(log_likelihoods.iter()) {
            let inc_log_w = delta * ll - max_log_w;
            let inc_w = inc_log_w.exp();
            let new_w = w * inc_w;
            sum_w += new_w;
            sum_w_sq += new_w * new_w;
        }

        if sum_w_sq > 0.0 {
            sum_w * sum_w / sum_w_sq
        } else {
            0.0
        }
    }
}

impl Default for AdaptiveTemperatureSelector {
    fn default() -> Self {
        Self::new(AdaptiveConfig::default())
    }
}

// =============================================================================
// CESS (Conditional ESS) Method
// =============================================================================

/// Conditional ESS temperature selector
///
/// Uses the analytical solution for the temperature that achieves
/// exactly the target ESS, assuming uniform current weights.
pub struct CESSSelector {
    target_ess_ratio: f64,
}

impl CESSSelector {
    pub fn new(target_ess_ratio: f64) -> Self {
        CESSSelector {
            target_ess_ratio: target_ess_ratio.clamp(0.1, 0.99),
        }
    }

    /// Find next temperature using CESS criterion
    ///
    /// For uniform weights, the ESS after reweighting is:
    ///   ESS = N / (1 + Var(exp(Δβ · log π)))
    ///
    /// We solve for Δβ given target ESS.
    pub fn find_next_temperature(
        &self,
        current_temp: f64,
        log_likelihoods: &[f64],
        n_particles: usize,
    ) -> f64 {
        let target_ess = self.target_ess_ratio * n_particles as f64;

        // Start with a guess based on variance of log-likelihoods
        let ll_mean: f64 = log_likelihoods.iter().sum::<f64>() / n_particles as f64;
        let ll_var: f64 = log_likelihoods.iter()
            .map(|ll| (ll - ll_mean).powi(2))
            .sum::<f64>() / n_particles as f64;

        if ll_var < 1e-10 {
            // All log-likelihoods are equal → go directly to 1
            return 1.0;
        }

        // Approximate: ESS ≈ N / (1 + Δβ² · Var(log π))
        // Solve: target = N / (1 + Δβ² · V)
        // → Δβ = sqrt((N/target - 1) / V)
        let target_ratio = n_particles as f64 / target_ess;
        let delta_beta_sq = (target_ratio - 1.0) / ll_var;

        if delta_beta_sq <= 0.0 {
            return 1.0;
        }

        let delta_beta = delta_beta_sq.sqrt();
        let next_temp = (current_temp + delta_beta).min(1.0);

        next_temp
    }
}

// =============================================================================
// Combined Adaptive Scheduler
// =============================================================================

/// Full adaptive SMC scheduler combining temperature selection with resampling
pub struct AdaptiveSMCScheduler {
    temp_selector: AdaptiveTemperatureSelector,
    schedule: TemperatureSchedule,
    resample_threshold: f64,
}

impl AdaptiveSMCScheduler {
    pub fn new(ess_threshold: f64) -> Self {
        let config = AdaptiveConfig {
            ess_threshold,
            ..Default::default()
        };

        AdaptiveSMCScheduler {
            temp_selector: AdaptiveTemperatureSelector::new(config),
            schedule: TemperatureSchedule::adaptive(),
            resample_threshold: ess_threshold,
        }
    }

    /// Initialize with first temperature (0)
    pub fn initialize(&mut self) {
        self.schedule = TemperatureSchedule::adaptive();
    }

    /// Advance to next temperature
    ///
    /// Returns: (new_temperature, should_resample)
    pub fn advance(
        &mut self,
        log_likelihoods: &[f64],
        current_weights: &[f64],
        n_particles: usize,
    ) -> (f64, bool) {
        let current_temp = self.schedule.current();

        if current_temp >= 1.0 - 1e-10 {
            return (1.0, false);
        }

        // Find next temperature
        let next_temp = self.temp_selector.find_next_temperature(
            current_temp,
            log_likelihoods,
            current_weights,
            n_particles,
        );

        self.schedule.push(next_temp);

        // Compute ESS at new temperature
        let ess = self.temp_selector.compute_ess_at_temp(
            current_temp,
            next_temp,
            log_likelihoods,
            current_weights,
        );
        self.schedule.record_ess(ess);

        // Should we resample?
        let should_resample = ess / n_particles as f64 < self.resample_threshold;
        if should_resample {
            self.schedule.record_resample();
        }

        (next_temp, should_resample)
    }

    /// Is sampling complete?
    pub fn is_complete(&self) -> bool {
        self.schedule.is_complete()
    }

    /// Get current schedule
    pub fn schedule(&self) -> &TemperatureSchedule {
        &self.schedule
    }

    /// Get schedule summary
    pub fn summary(&self) -> ScheduleSummary {
        self.schedule.summary()
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Compute ESS from weights
pub fn compute_ess(weights: &[f64]) -> f64 {
    let sum_sq: f64 = weights.iter().map(|w| w * w).sum();
    if sum_sq > 0.0 {
        1.0 / sum_sq
    } else {
        0.0
    }
}

/// Normalize log-weights to weights
pub fn normalize_log_weights(log_weights: &[f64]) -> Vec<f64> {
    if log_weights.is_empty() {
        return Vec::new();
    }

    let max_log_w = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);

    let weights: Vec<f64> = log_weights.iter()
        .map(|lw| (lw - max_log_w).exp())
        .collect();

    let sum: f64 = weights.iter().sum();

    if sum > 0.0 {
        weights.iter().map(|w| w / sum).collect()
    } else {
        vec![1.0 / log_weights.len() as f64; log_weights.len()]
    }
}

/// Estimate log marginal likelihood from temperature schedule
pub fn estimate_log_marginal_likelihood(
    schedule: &TemperatureSchedule,
    log_likelihoods_by_step: &[Vec<f64>],
    weights_by_step: &[Vec<f64>],
) -> f64 {
    var log_z = 0.0;

    let temps = schedule.temperatures();
    for i in 1..temps.len() {
        let delta = temps[i] - temps[i-1];
        let ll = &log_likelihoods_by_step[i-1];
        let w = &weights_by_step[i-1];

        // log Z += log E[exp(Δβ · log π)]
        let max_ll: f64 = ll.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = ll.iter().zip(w.iter())
            .map(|(l, wi)| wi * (delta * l - delta * max_ll).exp())
            .sum();

        log_z += delta * max_ll + sum.ln();
    }

    log_z
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn test_linear_schedule() {
    let schedule = TemperatureSchedule::linear(10);
    assert_eq!(schedule.temperatures().len(), 11);
    assert!((schedule.temperatures()[0] - 0.0).abs() < 1e-10);
    assert!((schedule.temperatures()[10] - 1.0).abs() < 1e-10);
}

#[test]
fn test_geometric_schedule() {
    let schedule = TemperatureSchedule::geometric(10, 2.0);
    assert_eq!(schedule.temperatures().len(), 11);
    // Geometric should have more resolution near 0
    let mid_temp = schedule.temperatures()[5];
    assert!(mid_temp > 0.5, "Geometric schedule should be skewed toward 1");
}

#[test]
fn test_adaptive_selector() {
    let selector = AdaptiveTemperatureSelector::default();

    // Uniform log-likelihoods → should go to 1 immediately
    let uniform_ll = vec![-1.0; 100];
    let uniform_w = vec![0.01; 100];

    let next = selector.find_next_temperature(0.0, &uniform_ll, &uniform_w, 100);
    assert!((next - 1.0).abs() < 1e-6, "Uniform LL should allow direct jump to 1");

    // Highly variable log-likelihoods → should take small step
    var varied_ll = vec![-1.0; 100];
    varied_ll[0] = -100.0;  // One very low likelihood
    varied_ll[1] = 0.0;     // One very high likelihood

    let next = selector.find_next_temperature(0.0, &varied_ll, &uniform_w, 100);
    assert!(next < 1.0, "Varied LL should not allow direct jump to 1");
    assert!(next > 0.0, "Next temperature should be positive");
}

#[test]
fn test_cess_selector() {
    let selector = CESSSelector::new(0.5);

    // Test with moderate variance
    let n = 1000;
    let log_likelihoods: Vec<f64> = (0..n)
        .map(|i| -((i as f64 / n as f64 - 0.5).powi(2) * 10.0))
        .collect();

    let next = selector.find_next_temperature(0.0, &log_likelihoods, n);

    assert!(next > 0.0 && next <= 1.0, "CESS should return valid temperature");
}

#[test]
fn test_full_adaptive_scheduler() {
    var scheduler = AdaptiveSMCScheduler::new(0.5);
    scheduler.initialize();

    let n = 100;
    // Simulate: log-likelihoods get more concentrated as we approach target
    var current_temp = 0.0;
    var weights = vec![1.0 / n as f64; n];

    var steps = 0;
    while !scheduler.is_complete() && steps < 100 {
        // Generate log-likelihoods (simulated)
        let log_likelihoods: Vec<f64> = (0..n)
            .map(|i| {
                let x = i as f64 / n as f64;
                -(x - 0.3).powi(2) * 5.0  // Peak at x=0.3
            })
            .collect();

        let (new_temp, should_resample) = scheduler.advance(&log_likelihoods, &weights, n);

        if should_resample {
            // Reset to uniform weights after resampling
            weights = vec![1.0 / n as f64; n];
        }

        current_temp = new_temp;
        steps += 1;
    }

    let summary = scheduler.summary();
    assert!(summary.n_temperatures >= 2, "Should have at least start and end");
    assert!(scheduler.is_complete(), "Should complete within 100 steps");
}

#[test]
fn test_ess_computation() {
    // Uniform weights → ESS = N
    let uniform = vec![0.1; 10];
    let ess = compute_ess(&uniform);
    assert!((ess - 10.0).abs() < 0.01);

    // Degenerate weights → ESS = 1
    var degen = vec![0.0; 10];
    degen[0] = 1.0;
    let ess = compute_ess(&degen);
    assert!((ess - 1.0).abs() < 0.01);
}
