//! Epistemic Quantum States
//!
//! Quantum states with integrated epistemic uncertainty tracking.
//! Every qubit state carries its amplitude AND noise-induced variance.

use std::ops::{Add, Mul, Sub};

use crate::epistemic::bayesian::BetaConfidence;

// =============================================================================
// Complex Number Type
// =============================================================================

/// Complex number for quantum amplitudes
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn from_polar(r: f64, theta: f64) -> Self {
        Self {
            re: r * theta.cos(),
            im: r * theta.sin(),
        }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> f64 {
        self.norm_sq().sqrt()
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn phase(&self) -> f64 {
        self.im.atan2(self.re)
    }
}

impl Add for Complex {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Self {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;
    fn sub(self, other: Self) -> Self {
        Self {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        Self {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }
}

impl Mul<f64> for Complex {
    type Output = Self;
    fn mul(self, scalar: f64) -> Self {
        Self {
            re: self.re * scalar,
            im: self.im * scalar,
        }
    }
}

// =============================================================================
// State Vector
// =============================================================================

/// Pure quantum state as amplitude vector
#[derive(Debug, Clone)]
pub struct StateVector {
    /// Amplitudes (2^n for n qubits)
    pub amplitudes: Vec<Complex>,
    /// Number of qubits
    pub num_qubits: usize,
}

impl StateVector {
    /// Create |0...0> state for n qubits
    pub fn zero_state(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex::ZERO; dim];
        amplitudes[0] = Complex::ONE;
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create |1...1> state for n qubits
    pub fn one_state(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex::ZERO; dim];
        amplitudes[dim - 1] = Complex::ONE;
        Self {
            amplitudes,
            num_qubits,
        }
    }

    /// Create uniform superposition (|+>^n)
    pub fn plus_state(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let amp = Complex::new(1.0 / (dim as f64).sqrt(), 0.0);
        Self {
            amplitudes: vec![amp; dim],
            num_qubits,
        }
    }

    /// Get probability of measuring a basis state
    pub fn probability(&self, index: usize) -> f64 {
        if index < self.amplitudes.len() {
            self.amplitudes[index].norm_sq()
        } else {
            0.0
        }
    }

    /// Get probabilities for all basis states
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Check normalization
    pub fn is_normalized(&self, tolerance: f64) -> bool {
        let norm: f64 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        (norm - 1.0).abs() < tolerance
    }

    /// Normalize the state
    pub fn normalize(&mut self) {
        let norm: f64 = self.amplitudes.iter().map(|a| a.norm_sq()).sum();
        let norm = norm.sqrt();
        if norm > 1e-10 {
            for a in &mut self.amplitudes {
                *a = *a * (1.0 / norm);
            }
        }
    }

    /// Inner product <self|other>
    pub fn inner(&self, other: &StateVector) -> Complex {
        assert_eq!(self.amplitudes.len(), other.amplitudes.len());
        let mut result = Complex::ZERO;
        for (a, b) in self.amplitudes.iter().zip(other.amplitudes.iter()) {
            result = result + a.conj() * *b;
        }
        result
    }

    /// Fidelity |<self|other>|^2
    pub fn fidelity(&self, other: &StateVector) -> f64 {
        self.inner(other).norm_sq()
    }
}

// =============================================================================
// Density Matrix (for mixed states / noise)
// =============================================================================

/// Mixed quantum state as density matrix
#[derive(Debug, Clone)]
pub struct DensityMatrix {
    /// Matrix elements (row-major, dim x dim)
    pub elements: Vec<Complex>,
    /// Dimension (2^n)
    pub dim: usize,
    /// Number of qubits
    pub num_qubits: usize,
}

impl DensityMatrix {
    /// Create from pure state |psi><psi|
    pub fn from_pure(state: &StateVector) -> Self {
        let dim = state.amplitudes.len();
        let mut elements = vec![Complex::ZERO; dim * dim];

        for i in 0..dim {
            for j in 0..dim {
                elements[i * dim + j] = state.amplitudes[i] * state.amplitudes[j].conj();
            }
        }

        Self {
            elements,
            dim,
            num_qubits: state.num_qubits,
        }
    }

    /// Create maximally mixed state I/d
    pub fn maximally_mixed(num_qubits: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut elements = vec![Complex::ZERO; dim * dim];
        let diag = 1.0 / dim as f64;

        for i in 0..dim {
            elements[i * dim + i] = Complex::new(diag, 0.0);
        }

        Self {
            elements,
            dim,
            num_qubits,
        }
    }

    /// Get diagonal element (probability of basis state)
    pub fn probability(&self, index: usize) -> f64 {
        if index < self.dim {
            self.elements[index * self.dim + index].re
        } else {
            0.0
        }
    }

    /// Trace of density matrix (should be 1)
    pub fn trace(&self) -> f64 {
        let mut tr = 0.0;
        for i in 0..self.dim {
            tr += self.elements[i * self.dim + i].re;
        }
        tr
    }

    /// Purity Tr(rho^2) - 1 for pure, 1/d for maximally mixed
    pub fn purity(&self) -> f64 {
        let mut purity = 0.0;
        for i in 0..self.dim {
            for k in 0..self.dim {
                let rho_ik = self.elements[i * self.dim + k];
                let rho_ki = self.elements[k * self.dim + i];
                purity += (rho_ik * rho_ki).re;
            }
        }
        purity
    }

    /// Von Neumann entropy -Tr(rho log rho)
    pub fn entropy(&self) -> f64 {
        // Simplified: use eigenvalue decomposition
        // For now, approximate using purity
        let purity = self.purity();
        if purity > 0.9999 {
            0.0 // Pure state
        } else {
            // Linear entropy approximation
            (1.0 - purity) * (self.dim as f64).ln()
        }
    }
}

// =============================================================================
// Qubit State with Epistemic Metadata
// =============================================================================

/// A quantum state with epistemic uncertainty tracking
#[derive(Debug, Clone)]
pub struct QubitState {
    /// The quantum state (pure or mixed)
    pub state: QuantumState,
    /// Noise-induced variance in amplitudes
    pub amplitude_variance: f64,
    /// Gate error accumulated
    pub gate_error_accumulated: f64,
    /// Number of gates applied
    pub gate_count: usize,
    /// Decoherence time factor (T1/T2 effects)
    pub decoherence_factor: f64,
    /// Measurement shots for statistics
    pub measurement_shots: Option<usize>,
}

/// Either pure or mixed state
#[derive(Debug, Clone)]
pub enum QuantumState {
    Pure(StateVector),
    Mixed(DensityMatrix),
}

impl QubitState {
    /// Create a pure |0...0> state
    pub fn zero_state(num_qubits: usize) -> Self {
        Self {
            state: QuantumState::Pure(StateVector::zero_state(num_qubits)),
            amplitude_variance: 0.0,
            gate_error_accumulated: 0.0,
            gate_count: 0,
            decoherence_factor: 1.0,
            measurement_shots: None,
        }
    }

    /// Create a pure |+...+> state
    pub fn plus_state(num_qubits: usize) -> Self {
        Self {
            state: QuantumState::Pure(StateVector::plus_state(num_qubits)),
            amplitude_variance: 0.0,
            gate_error_accumulated: 0.0,
            gate_count: 0,
            decoherence_factor: 1.0,
            measurement_shots: None,
        }
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        match &self.state {
            QuantumState::Pure(sv) => sv.num_qubits,
            QuantumState::Mixed(dm) => dm.num_qubits,
        }
    }

    /// Get probability of a basis state
    pub fn probability(&self, index: usize) -> f64 {
        match &self.state {
            QuantumState::Pure(sv) => sv.probability(index),
            QuantumState::Mixed(dm) => dm.probability(index),
        }
    }

    /// Get all probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        match &self.state {
            QuantumState::Pure(sv) => sv.probabilities(),
            QuantumState::Mixed(dm) => (0..dm.dim).map(|i| dm.probability(i)).collect(),
        }
    }

    /// Get purity (1 for pure, < 1 for mixed)
    pub fn purity(&self) -> f64 {
        match &self.state {
            QuantumState::Pure(_) => 1.0,
            QuantumState::Mixed(dm) => dm.purity(),
        }
    }

    /// Estimate total epistemic variance from noise
    pub fn total_variance(&self) -> f64 {
        // Combine amplitude variance, gate errors, and decoherence
        let base_variance = self.amplitude_variance;
        let gate_variance = self.gate_error_accumulated * self.gate_count as f64;
        let decoherence_variance = 1.0 - self.decoherence_factor;

        base_variance + gate_variance + decoherence_variance
    }

    /// Apply gate error to the state
    pub fn apply_gate_error(&mut self, error_rate: f64) {
        self.gate_error_accumulated += error_rate;
        self.gate_count += 1;
        self.amplitude_variance += error_rate * error_rate;
    }

    /// Apply decoherence (T1/T2 decay)
    pub fn apply_decoherence(&mut self, decay_factor: f64) {
        self.decoherence_factor *= decay_factor;
    }

    /// Convert to density matrix (for noisy simulation)
    pub fn to_density_matrix(&self) -> DensityMatrix {
        match &self.state {
            QuantumState::Pure(sv) => DensityMatrix::from_pure(sv),
            QuantumState::Mixed(dm) => dm.clone(),
        }
    }
}

// =============================================================================
// Epistemic Qubit (Knowledge<QubitState>)
// =============================================================================

/// A qubit state with full epistemic metadata
#[derive(Debug, Clone)]
pub struct EpistemicQubit {
    /// The quantum state
    pub state: QubitState,
    /// Confidence in the state preparation
    pub confidence: BetaConfidence,
    /// Provenance hash
    pub provenance_hash: u64,
    /// Circuit depth that produced this state
    pub circuit_depth: usize,
}

impl EpistemicQubit {
    /// Create a high-confidence zero state
    pub fn zero_state(num_qubits: usize) -> Self {
        Self {
            state: QubitState::zero_state(num_qubits),
            confidence: BetaConfidence::from_confidence(0.99, 100.0), // High confidence in |0>
            provenance_hash: 0,
            circuit_depth: 0,
        }
    }

    /// Create a plus state with moderate confidence
    pub fn plus_state(num_qubits: usize) -> Self {
        Self {
            state: QubitState::plus_state(num_qubits),
            confidence: BetaConfidence::from_confidence(0.95, 50.0),
            provenance_hash: 0,
            circuit_depth: 0,
        }
    }

    /// Get the mean confidence
    pub fn confidence_mean(&self) -> f64 {
        self.confidence.mean()
    }

    /// Get the epistemic variance
    pub fn epistemic_variance(&self) -> f64 {
        // Combine confidence variance with quantum noise variance
        let confidence_var = self.confidence.variance();
        let quantum_var = self.state.total_variance();
        confidence_var + quantum_var
    }

    /// Update confidence after a noisy operation
    pub fn update_confidence_from_noise(&mut self, noise_factor: f64) {
        // Decrease confidence proportional to noise
        let new_mean = self.confidence.mean() * (1.0 - noise_factor);
        self.confidence = BetaConfidence::from_confidence(
            new_mean.max(0.01),
            self.confidence.alpha + self.confidence.beta,
        );
    }

    /// Measure with epistemic uncertainty
    pub fn measure(&self, shots: usize) -> EpistemicMeasurement {
        let probs = self.state.probabilities();
        let dim = probs.len();

        // Simulate measurement outcomes
        let mut counts = vec![0usize; dim];

        // Pseudo-random sampling based on probabilities
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        for shot in 0..shots {
            let mut hasher = DefaultHasher::new();
            shot.hash(&mut hasher);
            self.provenance_hash.hash(&mut hasher);
            let rand_val = (hasher.finish() as f64) / (u64::MAX as f64);

            let mut cumsum = 0.0;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if rand_val < cumsum {
                    counts[i] += 1;
                    break;
                }
            }
        }

        // Compute expectation value and variance
        let total = shots as f64;
        let measured_probs: Vec<f64> = counts.iter().map(|&c| c as f64 / total).collect();

        // Statistical variance from finite shots
        let shot_variance = measured_probs
            .iter()
            .map(|&p| p * (1.0 - p) / total)
            .sum::<f64>();

        // Total variance includes quantum noise and shot noise
        let total_variance = self.epistemic_variance() + shot_variance;

        EpistemicMeasurement {
            counts,
            probabilities: measured_probs,
            shots,
            confidence: BetaConfidence::from_confidence(self.confidence_mean(), shots as f64),
            variance: total_variance,
        }
    }
}

/// Result of an epistemic measurement
#[derive(Debug, Clone)]
pub struct EpistemicMeasurement {
    /// Counts for each basis state
    pub counts: Vec<usize>,
    /// Estimated probabilities
    pub probabilities: Vec<f64>,
    /// Number of shots
    pub shots: usize,
    /// Confidence in measurement
    pub confidence: BetaConfidence,
    /// Total variance (quantum + statistical)
    pub variance: f64,
}

impl EpistemicMeasurement {
    /// Get most likely outcome
    pub fn most_likely(&self) -> usize {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, c)| *c)
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get expectation value for a given observable (as probabilities)
    pub fn expectation(&self, values: &[f64]) -> f64 {
        assert_eq!(values.len(), self.probabilities.len());
        self.probabilities
            .iter()
            .zip(values.iter())
            .map(|(&p, &v)| p * v)
            .sum()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complex_operations() {
        let a = Complex::new(1.0, 2.0);
        let b = Complex::new(3.0, 4.0);

        let sum = a + b;
        assert!((sum.re - 4.0).abs() < 1e-10);
        assert!((sum.im - 6.0).abs() < 1e-10);

        let prod = a * b;
        assert!((prod.re - (-5.0)).abs() < 1e-10);
        assert!((prod.im - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_vector_zero() {
        let sv = StateVector::zero_state(2);
        assert_eq!(sv.num_qubits, 2);
        assert_eq!(sv.amplitudes.len(), 4);
        assert!((sv.probability(0) - 1.0).abs() < 1e-10);
        assert!(sv.probability(1) < 1e-10);
    }

    #[test]
    fn test_state_vector_plus() {
        let sv = StateVector::plus_state(2);
        assert!(sv.is_normalized(1e-10));

        // All probabilities should be 0.25
        for i in 0..4 {
            assert!((sv.probability(i) - 0.25).abs() < 1e-10);
        }
    }

    #[test]
    fn test_density_matrix_pure() {
        let sv = StateVector::zero_state(1);
        let dm = DensityMatrix::from_pure(&sv);

        assert!((dm.trace() - 1.0).abs() < 1e-10);
        assert!((dm.purity() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_density_matrix_mixed() {
        let dm = DensityMatrix::maximally_mixed(1);

        assert!((dm.trace() - 1.0).abs() < 1e-10);
        assert!((dm.purity() - 0.5).abs() < 1e-10); // 1/d for d=2
    }

    #[test]
    fn test_qubit_state_variance() {
        let mut qs = QubitState::zero_state(1);
        assert!(qs.total_variance() < 1e-10);

        // Apply some gate errors
        qs.apply_gate_error(0.01);
        qs.apply_gate_error(0.01);

        assert!(qs.total_variance() > 0.0);
        assert_eq!(qs.gate_count, 2);
    }

    #[test]
    fn test_epistemic_qubit() {
        let eq = EpistemicQubit::zero_state(2);

        assert_eq!(eq.state.num_qubits(), 2);
        assert!(eq.confidence_mean() > 0.9);
        assert!(eq.epistemic_variance() < 0.1);
    }

    #[test]
    fn test_epistemic_measurement() {
        let eq = EpistemicQubit::zero_state(1);
        let meas = eq.measure(1000);

        // Should mostly measure |0>
        assert!(meas.probabilities[0] > 0.9);
        assert_eq!(meas.shots, 1000);
    }

    #[test]
    fn test_fidelity() {
        let sv1 = StateVector::zero_state(1);
        let sv2 = StateVector::zero_state(1);
        let sv3 = StateVector::one_state(1);

        assert!((sv1.fidelity(&sv2) - 1.0).abs() < 1e-10);
        assert!(sv1.fidelity(&sv3) < 1e-10);
    }
}
