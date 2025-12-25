//! Quantum Noise Models with Epistemic Variance
//!
//! Noise models that propagate uncertainty through quantum operations:
//! - Depolarizing noise (random Pauli errors)
//! - Amplitude damping (T1 decay)
//! - Phase damping (T2 dephasing)
//! - Readout errors

use super::states::{Complex, QuantumState, QubitState, StateVector};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// =============================================================================
// Noise Types
// =============================================================================

/// Type of quantum noise
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseType {
    /// Depolarizing channel: ρ → (1-p)ρ + p/3(XρX + YρY + ZρZ)
    Depolarizing,
    /// Amplitude damping: models T1 decay to |0>
    AmplitudeDamping,
    /// Phase damping: models T2 dephasing
    PhaseDamping,
    /// Bit flip: X error with probability p
    BitFlip,
    /// Phase flip: Z error with probability p
    PhaseFlip,
    /// Readout error: measurement misclassification
    ReadoutError,
    /// Thermal relaxation: combined T1/T2 effects
    ThermalRelaxation,
}

// =============================================================================
// Individual Noise Channels
// =============================================================================

/// Depolarizing noise channel
#[derive(Debug, Clone)]
pub struct DepolarizingNoise {
    /// Error probability per qubit
    pub probability: f64,
    /// Qubits affected (None = all)
    pub qubits: Option<Vec<usize>>,
}

impl DepolarizingNoise {
    /// Create new depolarizing noise
    pub fn new(probability: f64) -> Self {
        Self {
            probability: probability.clamp(0.0, 1.0),
            qubits: None,
        }
    }

    /// Apply to specific qubits
    pub fn on_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.qubits = Some(qubits);
        self
    }

    /// Apply depolarizing channel to state vector (simplified)
    /// In reality, converts to density matrix; here we approximate
    pub fn apply(&self, state: &mut QubitState, seed: u64) {
        let n = state.num_qubits();
        let qubits: Vec<usize> = self.qubits.clone().unwrap_or_else(|| (0..n).collect());

        for &q in &qubits {
            // Use seed to determine if error occurs
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            q.hash(&mut hasher);
            let rand = (hasher.finish() as f64) / (u64::MAX as f64);

            if rand < self.probability {
                // Apply random Pauli error
                let error_type = ((rand / self.probability * 3.0) as usize) % 3;

                if let QuantumState::Pure(ref mut sv) = state.state {
                    match error_type {
                        0 => apply_pauli_x(sv, q),
                        1 => apply_pauli_y(sv, q),
                        _ => apply_pauli_z(sv, q),
                    }
                }

                // Track the error
                state.apply_gate_error(self.probability);
            }
        }

        // Increase variance due to noise
        state.amplitude_variance += self.probability * qubits.len() as f64;
    }

    /// Variance contribution
    pub fn variance_contribution(&self, num_qubits: usize) -> f64 {
        let affected = self.qubits.as_ref().map(|q| q.len()).unwrap_or(num_qubits);
        self.probability * affected as f64
    }
}

/// Amplitude damping channel (T1 decay)
#[derive(Debug, Clone)]
pub struct AmplitudeDamping {
    /// Damping parameter γ = 1 - e^(-t/T1)
    pub gamma: f64,
    /// T1 time (if specified)
    pub t1: Option<f64>,
    /// Qubits affected
    pub qubits: Option<Vec<usize>>,
}

impl AmplitudeDamping {
    /// Create from damping parameter
    pub fn new(gamma: f64) -> Self {
        Self {
            gamma: gamma.clamp(0.0, 1.0),
            t1: None,
            qubits: None,
        }
    }

    /// Create from T1 time and gate duration
    pub fn from_t1(t1: f64, gate_time: f64) -> Self {
        let gamma = 1.0 - (-gate_time / t1).exp();
        Self {
            gamma,
            t1: Some(t1),
            qubits: None,
        }
    }

    /// Apply to specific qubits
    pub fn on_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.qubits = Some(qubits);
        self
    }

    /// Apply amplitude damping (simplified state-vector model)
    pub fn apply(&self, state: &mut QubitState, seed: u64) {
        let n = state.num_qubits();
        let qubits: Vec<usize> = self.qubits.clone().unwrap_or_else(|| (0..n).collect());

        for &q in &qubits {
            // Probabilistic decay: |1> -> |0> with probability γ
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            q.hash(&mut hasher);
            "amplitude".hash(&mut hasher);
            let rand = (hasher.finish() as f64) / (u64::MAX as f64);

            if let QuantumState::Pure(ref mut sv) = state.state {
                // Get probability of |1> on this qubit
                let mask = 1 << (n - 1 - q);
                let mut p1 = 0.0;
                for (i, amp) in sv.amplitudes.iter().enumerate() {
                    if i & mask != 0 {
                        p1 += amp.norm_sq();
                    }
                }

                // Apply damping with probability proportional to |1> population
                if rand < p1 * self.gamma {
                    apply_partial_decay(sv, q, (1.0 - self.gamma).sqrt());
                }
            }

            state.apply_decoherence(1.0 - self.gamma);
        }

        state.amplitude_variance += self.gamma * qubits.len() as f64;
    }

    /// Variance contribution
    pub fn variance_contribution(&self, num_qubits: usize) -> f64 {
        let affected = self.qubits.as_ref().map(|q| q.len()).unwrap_or(num_qubits);
        self.gamma * affected as f64
    }
}

/// Phase damping channel (T2 dephasing)
#[derive(Debug, Clone)]
pub struct PhaseDamping {
    /// Dephasing parameter λ = 1 - e^(-t/T2)
    pub lambda: f64,
    /// T2 time (if specified)
    pub t2: Option<f64>,
    /// Qubits affected
    pub qubits: Option<Vec<usize>>,
}

impl PhaseDamping {
    /// Create from dephasing parameter
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda: lambda.clamp(0.0, 1.0),
            t2: None,
            qubits: None,
        }
    }

    /// Create from T2 time and gate duration
    pub fn from_t2(t2: f64, gate_time: f64) -> Self {
        let lambda = 1.0 - (-gate_time / t2).exp();
        Self {
            lambda,
            t2: Some(t2),
            qubits: None,
        }
    }

    /// Apply phase damping
    pub fn apply(&self, state: &mut QubitState, seed: u64) {
        let n = state.num_qubits();
        let qubits: Vec<usize> = self.qubits.clone().unwrap_or_else(|| (0..n).collect());

        for &q in &qubits {
            // Random phase flip with probability λ/2
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            q.hash(&mut hasher);
            "phase".hash(&mut hasher);
            let rand = (hasher.finish() as f64) / (u64::MAX as f64);

            if rand < self.lambda / 2.0
                && let QuantumState::Pure(ref mut sv) = state.state
            {
                apply_pauli_z(sv, q);
            }
        }

        state.amplitude_variance += self.lambda * 0.5 * qubits.len() as f64;
    }
}

/// Readout error model
#[derive(Debug, Clone)]
pub struct ReadoutError {
    /// P(measure 1 | state 0)
    pub p01: f64,
    /// P(measure 0 | state 1)
    pub p10: f64,
}

impl ReadoutError {
    /// Create symmetric readout error
    pub fn symmetric(probability: f64) -> Self {
        Self {
            p01: probability,
            p10: probability,
        }
    }

    /// Create asymmetric readout error
    pub fn asymmetric(p01: f64, p10: f64) -> Self {
        Self { p01, p10 }
    }

    /// Apply to measurement probabilities
    pub fn apply_to_probabilities(&self, probs: &mut [f64]) {
        if probs.len() < 2 {
            return;
        }

        // For single qubit: [p0, p1] -> [(1-p01)*p0 + p10*p1, p01*p0 + (1-p10)*p1]
        let p0 = probs[0];
        let p1 = probs[1];
        probs[0] = (1.0 - self.p01) * p0 + self.p10 * p1;
        probs[1] = self.p01 * p0 + (1.0 - self.p10) * p1;
    }
}

// =============================================================================
// Composite Noise Model
// =============================================================================

/// Combined noise model for realistic device simulation
#[derive(Debug, Clone)]
pub struct NoiseModel {
    /// Depolarizing noise (per gate)
    pub depolarizing: Option<DepolarizingNoise>,
    /// Amplitude damping
    pub amplitude_damping: Option<AmplitudeDamping>,
    /// Phase damping
    pub phase_damping: Option<PhaseDamping>,
    /// Readout errors
    pub readout: Option<ReadoutError>,
    /// Additional variance from calibration uncertainty
    pub calibration_variance: f64,
    /// Seed for reproducible noise
    pub seed: u64,
}

impl NoiseModel {
    /// Create empty (ideal) noise model
    pub fn ideal() -> Self {
        Self {
            depolarizing: None,
            amplitude_damping: None,
            phase_damping: None,
            readout: None,
            calibration_variance: 0.0,
            seed: 42,
        }
    }

    /// Create typical NISQ device noise model
    pub fn nisq_device() -> Self {
        Self {
            depolarizing: Some(DepolarizingNoise::new(0.001)), // 0.1% per gate
            amplitude_damping: Some(AmplitudeDamping::new(0.01)), // T1 effects
            phase_damping: Some(PhaseDamping::new(0.02)),      // T2 effects
            readout: Some(ReadoutError::symmetric(0.01)),      // 1% readout error
            calibration_variance: 0.001,
            seed: 42,
        }
    }

    /// Create aggressive noise model (for testing robustness)
    pub fn high_noise() -> Self {
        Self {
            depolarizing: Some(DepolarizingNoise::new(0.01)), // 1% per gate
            amplitude_damping: Some(AmplitudeDamping::new(0.05)),
            phase_damping: Some(PhaseDamping::new(0.1)),
            readout: Some(ReadoutError::symmetric(0.05)),
            calibration_variance: 0.01,
            seed: 42,
        }
    }

    /// Create from T1/T2 times
    pub fn from_coherence_times(t1: f64, t2: f64, gate_time: f64) -> Self {
        Self {
            depolarizing: Some(DepolarizingNoise::new(0.001)),
            amplitude_damping: Some(AmplitudeDamping::from_t1(t1, gate_time)),
            phase_damping: Some(PhaseDamping::from_t2(t2, gate_time)),
            readout: Some(ReadoutError::symmetric(0.01)),
            calibration_variance: 0.001,
            seed: 42,
        }
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Apply all noise channels to state
    pub fn apply_to_state(&self, state: &mut QubitState) {
        if let Some(ref depol) = self.depolarizing {
            depol.apply(state, self.seed);
        }

        if let Some(ref amp) = self.amplitude_damping {
            amp.apply(state, self.seed);
        }

        if let Some(ref phase) = self.phase_damping {
            phase.apply(state, self.seed);
        }

        // Add calibration variance
        state.amplitude_variance += self.calibration_variance;
    }

    /// Apply readout errors to measurement probabilities
    pub fn apply_readout(&self, probs: &mut [f64]) {
        if let Some(ref readout) = self.readout {
            readout.apply_to_probabilities(probs);
        }
    }

    /// Total variance contribution from noise
    pub fn total_variance(&self, num_qubits: usize) -> f64 {
        let mut var = self.calibration_variance;

        if let Some(ref depol) = self.depolarizing {
            var += depol.variance_contribution(num_qubits);
        }

        if let Some(ref amp) = self.amplitude_damping {
            var += amp.variance_contribution(num_qubits);
        }

        var
    }

    /// Effective fidelity after noise
    pub fn effective_fidelity(&self, num_gates: usize, num_qubits: usize) -> f64 {
        let mut fidelity = 1.0;

        if let Some(ref depol) = self.depolarizing {
            // Per-gate fidelity loss
            fidelity *= (1.0 - depol.probability).powi(num_gates as i32);
        }

        if let Some(ref amp) = self.amplitude_damping {
            fidelity *= (1.0 - amp.gamma).powi(num_qubits as i32);
        }

        if let Some(ref phase) = self.phase_damping {
            fidelity *= (1.0 - phase.lambda * 0.5).powi(num_qubits as i32);
        }

        fidelity
    }
}

// =============================================================================
// Helper functions for noise application
// =============================================================================

fn apply_pauli_x(sv: &mut StateVector, qubit: usize) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);

    for i in 0..(1 << n) {
        if i & mask == 0 {
            let j = i | mask;
            sv.amplitudes.swap(i, j);
        }
    }
}

fn apply_pauli_y(sv: &mut StateVector, qubit: usize) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);

    for i in 0..(1 << n) {
        if i & mask == 0 {
            let j = i | mask;
            let temp = sv.amplitudes[i];
            sv.amplitudes[i] = sv.amplitudes[j] * Complex::new(0.0, 1.0);
            sv.amplitudes[j] = temp * Complex::new(0.0, -1.0);
        }
    }
}

fn apply_pauli_z(sv: &mut StateVector, qubit: usize) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);

    for i in 0..(1 << n) {
        if i & mask != 0 {
            sv.amplitudes[i] = sv.amplitudes[i] * (-1.0);
        }
    }
}

fn apply_partial_decay(sv: &mut StateVector, qubit: usize, scale: f64) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);

    // Scale |1> amplitudes, transfer some to |0>
    for i in 0..(1 << n) {
        if i & mask != 0 {
            let j = i & !mask; // Corresponding |0> state
            let transfer = sv.amplitudes[i] * (1.0 - scale);
            sv.amplitudes[j] = sv.amplitudes[j] + transfer * Complex::new(0.5, 0.0);
            sv.amplitudes[i] = sv.amplitudes[i] * scale;
        }
    }

    // Renormalize
    sv.normalize();
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_depolarizing_noise() {
        let noise = DepolarizingNoise::new(0.1);
        let mut state = QubitState::zero_state(1);

        noise.apply(&mut state, 12345);

        // State should have some variance accumulated
        assert!(state.amplitude_variance > 0.0);
    }

    #[test]
    fn test_amplitude_damping() {
        let noise = AmplitudeDamping::new(0.1);
        let mut state = QubitState::plus_state(1);

        noise.apply(&mut state, 54321);

        // Decoherence factor should be reduced
        assert!(state.decoherence_factor < 1.0);
    }

    #[test]
    fn test_readout_error() {
        let readout = ReadoutError::symmetric(0.1);
        let mut probs = vec![0.0, 1.0]; // Definitely |1>

        readout.apply_to_probabilities(&mut probs);

        // Should have 10% chance of reading 0
        assert!((probs[0] - 0.1).abs() < 1e-10);
        assert!((probs[1] - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_nisq_noise_model() {
        let noise = NoiseModel::nisq_device();
        let mut state = QubitState::zero_state(2);

        noise.apply_to_state(&mut state);

        // Should have accumulated some variance
        assert!(state.amplitude_variance > 0.0);
    }

    #[test]
    fn test_noise_fidelity() {
        let noise = NoiseModel::nisq_device();
        let fidelity = noise.effective_fidelity(100, 4);

        // Fidelity should be reduced but still positive
        assert!(fidelity < 1.0);
        assert!(fidelity > 0.0);
    }

    #[test]
    fn test_ideal_noise() {
        let noise = NoiseModel::ideal();
        let mut state = QubitState::zero_state(1);
        let initial_variance = state.amplitude_variance;

        noise.apply_to_state(&mut state);

        // Ideal noise should not add variance
        assert!((state.amplitude_variance - initial_variance).abs() < 1e-10);
    }

    #[test]
    fn test_coherence_time_noise() {
        // T1 = 100μs, T2 = 50μs, gate = 100ns
        let noise = NoiseModel::from_coherence_times(100.0, 50.0, 0.1);

        assert!(noise.amplitude_damping.is_some());
        assert!(noise.phase_damping.is_some());
    }
}
