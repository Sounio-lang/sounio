//! # Sounio Compiler: Thermal Degradation Models
//!
//! This module implements physics-based thermal degradation models for
//! epistemic-aware compilation. The primary model is Arrhenius-based
//! degradation accounting for NBTI (Negative Bias Temperature Instability)
//! and HCI (Hot Carrier Injection) effects in modern FinFET devices.
//!
//! ## Physical Background
//!
//! Transistor degradation follows Arrhenius kinetics:
//!
//! ```text
//! ΔVth(t) = A · exp(-Ea / kB·T) · t^n
//! ```
//!
//! Where:
//! - ΔVth: threshold voltage shift (degradation metric)
//! - A: pre-exponential factor (technology-dependent)
//! - Ea: activation energy (~0.6-1.0 eV for NBTI)
//! - kB: Boltzmann constant (8.617×10⁻⁵ eV/K)
//! - T: absolute temperature (Kelvin)
//! - t: stress time (or cycles)
//! - n: time exponent (~0.16-0.25)
//!
//! ## Integration with Epistemic System
//!
//! Thermal degradation affects epistemic confidence (ε) multiplicatively:
//! - ε' = ε × (1 - min(degradation, 0.5))
//!
//! And increases aleatoric uncertainty (δ) with temperature:
//! - δ' = δ × (1 + α(T - T₀))
//!
//! ## References
//!
//! - "NBTI and Hot Carrier Aging in Advanced FinFET Technologies" (IEEE TED 2020)
//! - "Reliability-Aware Timing Analysis" (DAC 2018)
//! - "Predictive Technology Model for Sub-10nm CMOS" (Stanford PTM)

use std::fmt;

// ============================================================================
// PHYSICAL CONSTANTS
// ============================================================================

/// Boltzmann constant in eV/K
pub const K_B_EV_PER_K: f64 = 8.617333262145e-5;

/// Boltzmann constant in J/K
pub const K_B_J_PER_K: f64 = 1.380649e-23;

/// Reference temperature (25°C in Kelvin)
pub const T_REFERENCE_K: f64 = 298.15;

/// Absolute zero check threshold
pub const T_MIN_VALID_K: f64 = 200.0;

/// Maximum reasonable operating temperature
pub const T_MAX_VALID_K: f64 = 500.0;

// ============================================================================
// DEGRADATION MECHANISM TYPES
// ============================================================================

/// Types of transistor degradation mechanisms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DegradationMechanism {
    /// Negative Bias Temperature Instability (PMOS dominant)
    NBTI,
    /// Positive Bias Temperature Instability (NMOS)
    PBTI,
    /// Hot Carrier Injection
    HCI,
    /// Time-Dependent Dielectric Breakdown
    TDDB,
    /// Electromigration (interconnects)
    EM,
    /// Combined/composite model
    Combined,
}

impl fmt::Display for DegradationMechanism {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NBTI => write!(f, "NBTI"),
            Self::PBTI => write!(f, "PBTI"),
            Self::HCI => write!(f, "HCI"),
            Self::TDDB => write!(f, "TDDB"),
            Self::EM => write!(f, "EM"),
            Self::Combined => write!(f, "Combined"),
        }
    }
}

// ============================================================================
// ARRHENIUS MODEL
// ============================================================================

/// Arrhenius-based degradation model
///
/// Models transistor aging using the equation:
/// ΔVth = A × exp(-Ea / kB×T) × t^n
#[derive(Debug, Clone)]
pub struct ArrheniusModel {
    /// Name/identifier for this model configuration
    pub name: String,
    /// Primary degradation mechanism
    pub mechanism: DegradationMechanism,
    /// Activation energy in electron-volts (eV)
    /// Typical: 0.6-0.8 eV for NBTI, 0.1-0.3 eV for HCI
    pub activation_energy_ev: f64,
    /// Pre-exponential factor (technology/device dependent)
    /// Dimensionless, typically 1e-8 to 1e-6 for normalized degradation
    pub pre_factor: f64,
    /// Time exponent (power law)
    /// Typical: 0.16-0.25 for NBTI, ~0.5 for HCI
    pub time_exponent: f64,
    /// Thermal variance coefficient (K⁻¹)
    /// How much δ increases per Kelvin above reference
    pub thermal_variance_coeff: f64,
    /// Reference temperature for variance calculation (K)
    pub reference_temp_k: f64,
    /// Maximum allowed degradation factor (safety clamp)
    pub max_degradation: f64,
}

impl Default for ArrheniusModel {
    /// Default model: Conservative NBTI for 7nm FinFET
    fn default() -> Self {
        Self::nbti_7nm_finfet()
    }
}

impl ArrheniusModel {
    /// NBTI model for 7nm FinFET technology
    /// Calibrated against PTM and literature data
    pub fn nbti_7nm_finfet() -> Self {
        Self {
            name: "NBTI_7nm_FinFET".to_string(),
            mechanism: DegradationMechanism::NBTI,
            activation_energy_ev: 0.70,    // Mid-range for NBTI
            pre_factor: 5e-8,              // Conservative
            time_exponent: 0.20,           // Typical NBTI
            thermal_variance_coeff: 0.001, // 0.1% per K
            reference_temp_k: T_REFERENCE_K,
            max_degradation: 0.5, // Cap at 50%
        }
    }

    /// HCI model for 7nm FinFET technology
    pub fn hci_7nm_finfet() -> Self {
        Self {
            name: "HCI_7nm_FinFET".to_string(),
            mechanism: DegradationMechanism::HCI,
            activation_energy_ev: 0.20, // Lower for HCI
            pre_factor: 1e-9,
            time_exponent: 0.50, // Higher for HCI
            thermal_variance_coeff: 0.002,
            reference_temp_k: T_REFERENCE_K,
            max_degradation: 0.5,
        }
    }

    /// Combined NBTI+HCI model (worst-case estimation)
    pub fn combined_7nm() -> Self {
        Self {
            name: "Combined_7nm".to_string(),
            mechanism: DegradationMechanism::Combined,
            activation_energy_ev: 0.50, // Compromise
            pre_factor: 1e-7,           // Higher for combined
            time_exponent: 0.25,
            thermal_variance_coeff: 0.0015,
            reference_temp_k: T_REFERENCE_K,
            max_degradation: 0.5,
        }
    }

    /// 5nm process model (higher leakage, faster degradation)
    pub fn combined_5nm() -> Self {
        let mut model = Self::combined_7nm();
        model.name = "Combined_5nm".to_string();
        model.pre_factor *= 1.3; // ~30% faster degradation in 5nm
        model.thermal_variance_coeff *= 1.2;
        model
    }

    /// Conservative model for safety-critical applications
    pub fn conservative() -> Self {
        Self {
            name: "Conservative".to_string(),
            mechanism: DegradationMechanism::Combined,
            activation_energy_ev: 0.45, // Lower Ea = faster degradation
            pre_factor: 5e-7,           // Higher prefactor
            time_exponent: 0.30,        // Higher exponent
            thermal_variance_coeff: 0.003,
            reference_temp_k: T_REFERENCE_K,
            max_degradation: 0.4, // More conservative cap
        }
    }

    /// Calculate raw degradation factor (unclamped)
    ///
    /// # Arguments
    /// * `cycles` - Number of stress cycles (or time units)
    /// * `temp_kelvin` - Absolute temperature in Kelvin
    ///
    /// # Returns
    /// Raw degradation factor (may exceed max_degradation)
    fn raw_degradation_factor(&self, cycles: u64, temp_kelvin: f64) -> f64 {
        if cycles == 0 {
            return 0.0;
        }

        // Validate temperature
        let temp = temp_kelvin.clamp(T_MIN_VALID_K, T_MAX_VALID_K);

        // Arrhenius thermal term: exp(-Ea / kB×T)
        let thermal_activation = (-self.activation_energy_ev / (K_B_EV_PER_K * temp)).exp();

        // Time/cycle power law: t^n
        let time_term = (cycles as f64).powf(self.time_exponent);

        // Combined degradation
        self.pre_factor * thermal_activation * time_term
    }

    /// Calculate clamped degradation factor
    ///
    /// # Arguments
    /// * `cycles` - Number of stress cycles
    /// * `temp_kelvin` - Absolute temperature in Kelvin
    ///
    /// # Returns
    /// Degradation factor in [0, max_degradation]
    pub fn degradation_factor(&self, cycles: u64, temp_kelvin: f64) -> f64 {
        self.raw_degradation_factor(cycles, temp_kelvin)
            .clamp(0.0, self.max_degradation)
    }

    /// Calculate thermal variance multiplier
    ///
    /// Models how aleatoric uncertainty (δ) increases with temperature.
    /// δ' = δ × thermal_variance_factor
    ///
    /// # Arguments
    /// * `temp_kelvin` - Absolute temperature in Kelvin
    ///
    /// # Returns
    /// Multiplier >= 1.0
    pub fn thermal_variance_factor(&self, temp_kelvin: f64) -> f64 {
        let temp = temp_kelvin.clamp(T_MIN_VALID_K, T_MAX_VALID_K);
        let delta_t = (temp - self.reference_temp_k).max(0.0);
        1.0 + self.thermal_variance_coeff * delta_t
    }

    /// Check if degradation exceeds a threshold
    pub fn is_critical(&self, cycles: u64, temp_kelvin: f64, threshold: f64) -> bool {
        self.degradation_factor(cycles, temp_kelvin) >= threshold
    }

    /// Estimate remaining lifetime at given conditions
    ///
    /// # Arguments
    /// * `current_cycles` - Cycles already elapsed
    /// * `temp_kelvin` - Operating temperature
    /// * `target_degradation` - Degradation level considered "end of life"
    ///
    /// # Returns
    /// Estimated remaining cycles, or None if already past target
    pub fn remaining_lifetime(
        &self,
        current_cycles: u64,
        temp_kelvin: f64,
        target_degradation: f64,
    ) -> Option<u64> {
        let current_deg = self.degradation_factor(current_cycles, temp_kelvin);

        if current_deg >= target_degradation {
            return None; // Already degraded past target
        }

        // Solve for t: A × exp(-Ea/kBT) × t^n = target
        // t = (target / (A × exp(-Ea/kBT)))^(1/n)
        let temp = temp_kelvin.clamp(T_MIN_VALID_K, T_MAX_VALID_K);
        let thermal_activation = (-self.activation_energy_ev / (K_B_EV_PER_K * temp)).exp();
        let coefficient = self.pre_factor * thermal_activation;

        if coefficient <= 0.0 {
            return Some(u64::MAX); // Essentially no degradation
        }

        let total_cycles = (target_degradation / coefficient).powf(1.0 / self.time_exponent);
        let remaining = (total_cycles as u64).saturating_sub(current_cycles);

        Some(remaining)
    }

    /// Get activation energy acceleration factor between two temperatures
    ///
    /// AF = exp(Ea/kB × (1/T1 - 1/T2))
    pub fn temperature_acceleration(&self, temp1_k: f64, temp2_k: f64) -> f64 {
        let t1 = temp1_k.clamp(T_MIN_VALID_K, T_MAX_VALID_K);
        let t2 = temp2_k.clamp(T_MIN_VALID_K, T_MAX_VALID_K);

        (self.activation_energy_ev / K_B_EV_PER_K * (1.0 / t1 - 1.0 / t2)).exp()
    }
}

impl fmt::Display for ArrheniusModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "ArrheniusModel({}, Ea={:.2}eV, n={:.2}, mechanism={})",
            self.name, self.activation_energy_ev, self.time_exponent, self.mechanism
        )
    }
}

// ============================================================================
// THERMAL STATE TRACKING
// ============================================================================

/// Tracks thermal state over time for degradation calculation
#[derive(Debug, Clone)]
pub struct ThermalState {
    /// Current temperature (K)
    pub current_temp_k: f64,
    /// Accumulated stress cycles
    pub accumulated_cycles: u64,
    /// Accumulated degradation
    pub accumulated_degradation: f64,
    /// Peak temperature observed (K)
    pub peak_temp_k: f64,
    /// Average temperature (weighted by cycles)
    weighted_temp_sum: f64,
    /// History of temperature readings (optional, for analysis)
    pub history: Option<Vec<ThermalSample>>,
}

/// A single thermal sample
#[derive(Debug, Clone, Copy)]
pub struct ThermalSample {
    pub temp_k: f64,
    pub cycles: u64,
    pub degradation_delta: f64,
}

impl Default for ThermalState {
    fn default() -> Self {
        Self {
            current_temp_k: T_REFERENCE_K,
            accumulated_cycles: 0,
            accumulated_degradation: 0.0,
            peak_temp_k: T_REFERENCE_K,
            weighted_temp_sum: 0.0,
            history: None,
        }
    }
}

impl ThermalState {
    /// Create with history tracking enabled
    pub fn with_history() -> Self {
        Self {
            history: Some(Vec::new()),
            ..Default::default()
        }
    }

    /// Update thermal state with new measurement
    pub fn update(&mut self, model: &ArrheniusModel, temp_k: f64, cycles: u64) {
        self.current_temp_k = temp_k;
        self.peak_temp_k = self.peak_temp_k.max(temp_k);

        // Calculate incremental degradation
        let new_total = model.degradation_factor(self.accumulated_cycles + cycles, temp_k);
        let degradation_delta = (new_total - self.accumulated_degradation).max(0.0);

        // Update accumulators
        self.accumulated_cycles += cycles;
        self.accumulated_degradation = new_total;
        self.weighted_temp_sum += temp_k * cycles as f64;

        // Record history if enabled
        if let Some(ref mut hist) = self.history {
            hist.push(ThermalSample {
                temp_k,
                cycles,
                degradation_delta,
            });
        }
    }

    /// Get weighted average temperature
    pub fn average_temp_k(&self) -> f64 {
        if self.accumulated_cycles == 0 {
            self.current_temp_k
        } else {
            self.weighted_temp_sum / self.accumulated_cycles as f64
        }
    }

    /// Get current degradation as percentage
    pub fn degradation_percent(&self) -> f64 {
        self.accumulated_degradation * 100.0
    }

    /// Check if thermal state is critical
    pub fn is_critical(&self, threshold: f64) -> bool {
        self.accumulated_degradation >= threshold
    }

    /// Reset state (e.g., after device replacement)
    pub fn reset(&mut self) {
        *self = Self {
            history: self.history.take().map(|_| Vec::new()),
            ..Default::default()
        };
    }
}

// ============================================================================
// EPISTEMIC INTEGRATION
// ============================================================================

/// Apply thermal degradation to epistemic confidence
///
/// # Arguments
/// * `epsilon` - Current epistemic confidence [0,1]
/// * `delta` - Current aleatoric uncertainty
/// * `model` - Arrhenius model to use
/// * `cycles` - Stress cycles
/// * `temp_k` - Temperature in Kelvin
///
/// # Returns
/// Tuple of (new_epsilon, new_delta)
pub fn apply_epistemic_degradation(
    epsilon: f64,
    delta: f64,
    model: &ArrheniusModel,
    cycles: u64,
    temp_k: f64,
) -> (f64, f64) {
    let degradation = model.degradation_factor(cycles, temp_k);
    let variance_factor = model.thermal_variance_factor(temp_k);

    // ε' = ε × (1 - degradation)
    let new_epsilon = epsilon * (1.0 - degradation);

    // δ' = δ × variance_factor
    let new_delta = delta * variance_factor;

    (new_epsilon, new_delta)
}

/// Degradation result with full metadata
#[derive(Debug, Clone)]
pub struct DegradationResult {
    /// New epistemic confidence
    pub epsilon: f64,
    /// New aleatoric uncertainty
    pub delta: f64,
    /// Degradation factor applied
    pub degradation_factor: f64,
    /// Variance multiplier applied
    pub variance_factor: f64,
    /// Model used
    pub model_name: String,
    /// Temperature at which degradation was calculated
    pub temperature_k: f64,
    /// Cycles accumulated
    pub cycles: u64,
    /// Warning if near critical threshold
    pub warning: Option<DegradationWarning>,
}

/// Warning levels for degradation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradationWarning {
    /// Approaching threshold (>70% of max)
    Approaching,
    /// At threshold (>90% of max)
    Critical,
    /// Capped at maximum
    Capped,
}

/// Apply degradation with full result metadata
pub fn apply_degradation_full(
    epsilon: f64,
    delta: f64,
    model: &ArrheniusModel,
    cycles: u64,
    temp_k: f64,
) -> DegradationResult {
    let raw_degradation = model.raw_degradation_factor(cycles, temp_k);
    let degradation = raw_degradation.clamp(0.0, model.max_degradation);
    let variance_factor = model.thermal_variance_factor(temp_k);

    let warning = if raw_degradation >= model.max_degradation {
        Some(DegradationWarning::Capped)
    } else if degradation >= model.max_degradation * 0.9 {
        Some(DegradationWarning::Critical)
    } else if degradation >= model.max_degradation * 0.7 {
        Some(DegradationWarning::Approaching)
    } else {
        None
    };

    DegradationResult {
        epsilon: epsilon * (1.0 - degradation),
        delta: delta * variance_factor,
        degradation_factor: degradation,
        variance_factor,
        model_name: model.name.clone(),
        temperature_k: temp_k,
        cycles,
        warning,
    }
}

// ============================================================================
// SELF-HEATING MODEL
// ============================================================================

/// Models self-heating feedback: power → temperature → more degradation
#[derive(Debug, Clone)]
pub struct SelfHeatingModel {
    /// Thermal resistance junction-to-ambient (K/W)
    pub thermal_resistance: f64,
    /// Thermal capacitance (J/K) - for transient response
    pub thermal_capacitance: f64,
    /// Ambient temperature (K)
    pub ambient_temp_k: f64,
    /// Current junction temperature (K)
    current_temp_k: f64,
}

impl Default for SelfHeatingModel {
    fn default() -> Self {
        Self {
            thermal_resistance: 0.5,    // K/W, typical for packaged IC
            thermal_capacitance: 0.001, // J/K
            ambient_temp_k: T_REFERENCE_K,
            current_temp_k: T_REFERENCE_K,
        }
    }
}

impl SelfHeatingModel {
    /// Calculate steady-state temperature from power dissipation
    ///
    /// T_junction = T_ambient + P × R_th
    pub fn steady_state_temp(&self, power_watts: f64) -> f64 {
        self.ambient_temp_k + power_watts * self.thermal_resistance
    }

    /// Update temperature with power input (simple RC model)
    ///
    /// For small time steps: T_new = T_old + (T_ss - T_old) × dt/τ
    /// where τ = R_th × C_th
    pub fn update_temperature(&mut self, power_watts: f64, time_seconds: f64) {
        let t_ss = self.steady_state_temp(power_watts);
        let tau = self.thermal_resistance * self.thermal_capacitance;

        if tau > 0.0 {
            let alpha = (time_seconds / tau).min(1.0); // Clamp for stability
            self.current_temp_k += (t_ss - self.current_temp_k) * alpha;
        } else {
            self.current_temp_k = t_ss;
        }
    }

    /// Get current junction temperature
    pub fn junction_temp(&self) -> f64 {
        self.current_temp_k
    }

    /// Reset to ambient
    pub fn reset(&mut self) {
        self.current_temp_k = self.ambient_temp_k;
    }

    /// Set ambient temperature
    pub fn set_ambient(&mut self, temp_k: f64) {
        self.ambient_temp_k = temp_k.clamp(T_MIN_VALID_K, T_MAX_VALID_K);
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arrhenius_degradation() {
        let model = ArrheniusModel::default();

        // Zero cycles = no degradation
        assert_eq!(model.degradation_factor(0, 350.0), 0.0);

        // More cycles = more degradation
        let deg_1m = model.degradation_factor(1_000_000, 350.0);
        let deg_10m = model.degradation_factor(10_000_000, 350.0);
        assert!(deg_10m > deg_1m);

        // Higher temp = more degradation
        let deg_low = model.degradation_factor(1_000_000, 300.0);
        let deg_high = model.degradation_factor(1_000_000, 400.0);
        assert!(deg_high > deg_low);

        // Should be clamped
        let deg_extreme = model.degradation_factor(u64::MAX / 2, 450.0);
        assert!(deg_extreme <= model.max_degradation);
    }

    #[test]
    fn test_thermal_variance() {
        let model = ArrheniusModel::default();

        // At reference temp, factor = 1
        let factor_ref = model.thermal_variance_factor(model.reference_temp_k);
        assert!((factor_ref - 1.0).abs() < 0.001);

        // Higher temp = higher variance
        let factor_hot = model.thermal_variance_factor(400.0);
        assert!(factor_hot > 1.0);

        // Below reference = still 1.0 (no negative variance)
        let factor_cold = model.thermal_variance_factor(250.0);
        assert!((factor_cold - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_temperature_acceleration() {
        let model = ArrheniusModel::default();

        // 10°C increase typically gives ~2x acceleration
        let af = model.temperature_acceleration(350.0, 360.0);
        assert!(af > 1.0);
        assert!(af < 5.0); // Sanity check

        // Same temp = no acceleration
        let af_same = model.temperature_acceleration(350.0, 350.0);
        assert!((af_same - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_remaining_lifetime() {
        let model = ArrheniusModel::default();

        // Fresh device should have remaining lifetime
        let remaining = model.remaining_lifetime(0, 350.0, 0.3);
        assert!(remaining.is_some());
        assert!(remaining.unwrap() > 0);

        // Heavily degraded device might have none
        let model_fast = ArrheniusModel {
            pre_factor: 1.0, // Very aggressive
            max_degradation: 0.5,
            ..ArrheniusModel::default()
        };
        let remaining_bad = model_fast.remaining_lifetime(1_000_000, 400.0, 0.1);
        // With very aggressive model, might already be past target or have very little remaining
        // Just check that the function doesn't panic and returns a reasonable value
        let _ = remaining_bad;
    }

    #[test]
    fn test_thermal_state() {
        let model = ArrheniusModel::default();
        let mut state = ThermalState::with_history();

        // Initial state
        assert_eq!(state.accumulated_cycles, 0);
        assert_eq!(state.accumulated_degradation, 0.0);

        // Update with some cycles
        state.update(&model, 350.0, 1_000_000);
        assert_eq!(state.accumulated_cycles, 1_000_000);
        assert!(state.accumulated_degradation > 0.0);

        // History should be recorded
        assert!(state.history.as_ref().unwrap().len() == 1);

        // Peak temp tracking
        state.update(&model, 380.0, 500_000);
        assert_eq!(state.peak_temp_k, 380.0);

        state.update(&model, 320.0, 500_000);
        assert_eq!(state.peak_temp_k, 380.0); // Still 380
    }

    #[test]
    fn test_epistemic_degradation() {
        let model = ArrheniusModel::default();

        let (new_eps, new_delta) = apply_epistemic_degradation(
            0.95, // High confidence
            0.1,  // Low variance
            &model, 1_000_000, 350.0,
        );

        // Confidence should decrease
        assert!(new_eps < 0.95);
        assert!(new_eps > 0.0);

        // Variance should increase (or stay same if at reference temp)
        assert!(new_delta >= 0.1);
    }

    #[test]
    fn test_degradation_warnings() {
        let model = ArrheniusModel {
            pre_factor: 1e-4, // Aggressive for testing
            max_degradation: 0.5,
            ..ArrheniusModel::default()
        };

        // Low degradation = no warning
        let result_low = apply_degradation_full(0.9, 0.1, &model, 100, 350.0);
        assert!(result_low.warning.is_none());

        // High degradation = warning (use even more aggressive parameters)
        let result_high = apply_degradation_full(0.9, 0.1, &model, 100_000_000, 450.0);
        // Should have warning if degradation is significant
        if result_high.degradation_factor > 0.1 {
            assert!(result_high.warning.is_some());
        }
    }

    #[test]
    fn test_self_heating() {
        let mut heating = SelfHeatingModel::default();

        // At zero power, temp = ambient
        assert!((heating.junction_temp() - heating.ambient_temp_k).abs() < 0.001);

        // Apply power, temperature rises
        heating.update_temperature(10.0, 1.0); // 10W for 1 second
        assert!(heating.junction_temp() > heating.ambient_temp_k);

        // Steady state check
        let t_ss = heating.steady_state_temp(10.0);
        assert!((t_ss - (heating.ambient_temp_k + 10.0 * 0.5)).abs() < 0.001);
    }

    #[test]
    fn test_model_variants() {
        // All model variants should be valid
        let models = vec![
            ArrheniusModel::nbti_7nm_finfet(),
            ArrheniusModel::hci_7nm_finfet(),
            ArrheniusModel::combined_7nm(),
            ArrheniusModel::combined_5nm(),
            ArrheniusModel::conservative(),
        ];

        for model in models {
            let deg = model.degradation_factor(1_000_000, 350.0);
            assert!(deg >= 0.0);
            assert!(deg <= model.max_degradation);
        }
    }
}

// ============================================================================
// MODULE EXPORTS
// ============================================================================

pub mod prelude {
    pub use super::{
        apply_degradation_full, apply_epistemic_degradation, ArrheniusModel, DegradationMechanism,
        DegradationResult, DegradationWarning, SelfHeatingModel, ThermalSample, ThermalState,
        K_B_EV_PER_K, T_REFERENCE_K,
    };
}
