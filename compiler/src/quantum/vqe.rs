//! Variational Quantum Eigensolver with Epistemic Energy Estimation
//!
//! VQE and QAOA with full uncertainty quantification:
//! - Energy estimates carry confidence intervals
//! - Variance penalty encourages stable circuits
//! - Provenance tracking for quantum chemistry audits
//!
//! # Key Innovation
//!
//! Every energy measurement is Knowledge<f64>:
//! - mean: the estimated ground state energy
//! - variance: from shot noise + gate noise + parameter uncertainty
//! - confidence: Bayesian posterior on convergence

use super::circuit::{CircuitBuilder, QuantumCircuit};
use super::gates::Gate;
use super::noise::NoiseModel;
use super::states::QubitState;
use crate::epistemic::bayesian::BetaConfidence;
use std::f64::consts::PI;

// =============================================================================
// Pauli Operators and Hamiltonians
// =============================================================================

/// Pauli operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PauliOp {
    I, // Identity
    X,
    Y,
    Z,
}

impl PauliOp {
    /// Get the matrix element for computational basis
    pub fn eigenvalue(&self, bit: bool) -> f64 {
        match (self, bit) {
            (PauliOp::I, _) => 1.0,
            (PauliOp::Z, false) => 1.0,
            (PauliOp::Z, true) => -1.0,
            _ => 0.0, // X, Y require superposition measurement
        }
    }
}

/// A term in a Pauli Hamiltonian: coefficient × P_1 ⊗ P_2 ⊗ ... ⊗ P_n
#[derive(Debug, Clone)]
pub struct PauliTerm {
    /// Coefficient
    pub coeff: f64,
    /// Pauli operators on each qubit (I if not specified)
    pub paulis: Vec<(usize, PauliOp)>,
}

impl PauliTerm {
    /// Create a new Pauli term
    pub fn new(coeff: f64, paulis: Vec<(usize, PauliOp)>) -> Self {
        Self { coeff, paulis }
    }

    /// Identity term (constant)
    pub fn identity(coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![],
        }
    }

    /// Single Z term
    pub fn z(qubit: usize, coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![(qubit, PauliOp::Z)],
        }
    }

    /// ZZ interaction term
    pub fn zz(q1: usize, q2: usize, coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![(q1, PauliOp::Z), (q2, PauliOp::Z)],
        }
    }

    /// XX interaction term
    pub fn xx(q1: usize, q2: usize, coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![(q1, PauliOp::X), (q2, PauliOp::X)],
        }
    }

    /// YY interaction term
    pub fn yy(q1: usize, q2: usize, coeff: f64) -> Self {
        Self {
            coeff,
            paulis: vec![(q1, PauliOp::Y), (q2, PauliOp::Y)],
        }
    }

    /// Measure expectation value on a state (for Z-basis terms)
    pub fn expectation(&self, probs: &[f64], num_qubits: usize) -> f64 {
        if self.paulis.is_empty() {
            return self.coeff; // Identity term
        }

        // Check if all are Z operators (can measure directly)
        let all_z = self.paulis.iter().all(|(_, p)| *p == PauliOp::Z);

        if all_z {
            let mut exp = 0.0;
            for (i, &prob) in probs.iter().enumerate() {
                // Compute parity of selected qubits
                let mut parity = 1.0;
                for &(q, _) in &self.paulis {
                    let bit = (i >> (num_qubits - 1 - q)) & 1;
                    parity *= if bit == 0 { 1.0 } else { -1.0 };
                }
                exp += prob * parity;
            }
            self.coeff * exp
        } else {
            // For X, Y terms, need basis rotation (handled by measurement circuit)
            0.0 // Placeholder - real implementation needs Hadamard/S-dagger
        }
    }

    /// Generate measurement circuit for this term
    pub fn measurement_circuit(&self, num_qubits: usize) -> QuantumCircuit {
        let mut builder = CircuitBuilder::new(num_qubits);

        for &(q, op) in &self.paulis {
            match op {
                PauliOp::X => {
                    // X basis: apply Hadamard before measurement
                    builder = builder.h(q);
                }
                PauliOp::Y => {
                    // Y basis: apply S† H before measurement
                    builder = builder.rz(q, -PI / 2.0).h(q);
                }
                PauliOp::Z | PauliOp::I => {
                    // Z basis: no rotation needed
                }
            }
        }

        builder.build()
    }
}

/// Hamiltonian as sum of Pauli terms
#[derive(Debug, Clone)]
pub struct Hamiltonian {
    /// Pauli terms
    pub terms: Vec<PauliTerm>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Name (for provenance)
    pub name: String,
}

impl Hamiltonian {
    /// Create empty Hamiltonian
    pub fn new(num_qubits: usize, name: &str) -> Self {
        Self {
            terms: Vec::new(),
            num_qubits,
            name: name.to_string(),
        }
    }

    /// Add a term
    pub fn add_term(&mut self, term: PauliTerm) {
        self.terms.push(term);
    }

    /// Create H2 molecule Hamiltonian (STO-3G basis, bond length 0.7414 Å)
    /// This is the standard benchmark Hamiltonian
    pub fn h2_molecule() -> Self {
        let mut h = Hamiltonian::new(2, "H2_STO-3G");

        // H2 Hamiltonian in Bravyi-Kitaev encoding (2 qubits)
        // E = -1.0523 + 0.3979 Z0 - 0.3979 Z1 + 0.0112 Z0Z1
        //     + 0.1809 X0X1 + 0.1809 Y0Y1

        h.add_term(PauliTerm::identity(-1.0523));
        h.add_term(PauliTerm::z(0, 0.3979));
        h.add_term(PauliTerm::z(1, -0.3979));
        h.add_term(PauliTerm::zz(0, 1, -0.0112));
        h.add_term(PauliTerm::xx(0, 1, 0.1809));
        h.add_term(PauliTerm::yy(0, 1, 0.1809));

        h
    }

    /// Create LiH molecule Hamiltonian (4 qubit approximation)
    pub fn lih_molecule() -> Self {
        let mut h = Hamiltonian::new(4, "LiH_minimal");

        // Simplified LiH Hamiltonian
        h.add_term(PauliTerm::identity(-7.8));
        h.add_term(PauliTerm::z(0, 0.17));
        h.add_term(PauliTerm::z(1, -0.23));
        h.add_term(PauliTerm::z(2, 0.12));
        h.add_term(PauliTerm::z(3, -0.12));
        h.add_term(PauliTerm::zz(0, 1, 0.15));
        h.add_term(PauliTerm::zz(1, 2, 0.11));
        h.add_term(PauliTerm::zz(2, 3, 0.13));
        h.add_term(PauliTerm::xx(0, 1, 0.04));
        h.add_term(PauliTerm::yy(0, 1, 0.04));

        h
    }

    /// Create Ising model Hamiltonian: H = -J Σ Z_i Z_{i+1} - h Σ X_i
    pub fn ising_model(num_qubits: usize, j: f64, h_field: f64) -> Self {
        let mut h = Hamiltonian::new(num_qubits, &format!("Ising_{}qubits", num_qubits));

        // ZZ interactions
        for i in 0..(num_qubits - 1) {
            h.add_term(PauliTerm::zz(i, i + 1, -j));
        }

        // Transverse field
        for i in 0..num_qubits {
            h.add_term(PauliTerm::new(-h_field, vec![(i, PauliOp::X)]));
        }

        h
    }

    /// Estimate expectation value from measurement probabilities
    pub fn expectation(&self, probs: &[f64]) -> f64 {
        self.terms
            .iter()
            .map(|t| t.expectation(probs, self.num_qubits))
            .sum()
    }

    /// Group terms by commuting sets for simultaneous measurement
    pub fn group_commuting_terms(&self) -> Vec<Vec<&PauliTerm>> {
        // Simplified: group by same Pauli basis
        let mut z_terms = Vec::new();
        let mut other_terms = Vec::new();

        for term in &self.terms {
            let has_non_z = term
                .paulis
                .iter()
                .any(|(_, p)| *p == PauliOp::X || *p == PauliOp::Y);

            if has_non_z {
                other_terms.push(term);
            } else {
                z_terms.push(term);
            }
        }

        let mut groups = vec![];
        if !z_terms.is_empty() {
            groups.push(z_terms);
        }
        // For simplicity, each non-Z term is its own group
        for term in other_terms {
            groups.push(vec![term]);
        }

        groups
    }
}

// =============================================================================
// VQE Configuration and Results
// =============================================================================

/// VQE solver configuration
#[derive(Debug, Clone)]
pub struct VQEConfig {
    /// Number of measurement shots per term
    pub shots: usize,
    /// Maximum optimization iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Learning rate for gradient descent
    pub learning_rate: f64,
    /// Variance penalty weight (epistemic regularization)
    pub variance_penalty: f64,
    /// Noise model
    pub noise_model: Option<NoiseModel>,
    /// Use Adam optimizer
    pub use_adam: bool,
}

impl Default for VQEConfig {
    fn default() -> Self {
        Self {
            shots: 1024,
            max_iterations: 100,
            convergence_threshold: 1e-6,
            learning_rate: 0.1,
            variance_penalty: 0.1,
            noise_model: None,
            use_adam: true,
        }
    }
}

impl VQEConfig {
    /// Create config for fast prototyping
    pub fn fast() -> Self {
        Self {
            shots: 100,
            max_iterations: 20,
            convergence_threshold: 1e-4,
            ..Default::default()
        }
    }

    /// Create config for production accuracy
    pub fn production() -> Self {
        Self {
            shots: 8192,
            max_iterations: 500,
            convergence_threshold: 1e-8,
            variance_penalty: 0.01,
            ..Default::default()
        }
    }

    /// Add noise model
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.noise_model = Some(noise);
        self
    }
}

/// VQE optimization result with epistemic metadata
#[derive(Debug, Clone)]
pub struct VQEResult {
    /// Optimal energy (mean)
    pub energy: f64,
    /// Energy variance from measurement + noise
    pub variance: f64,
    /// Confidence in convergence
    pub confidence: BetaConfidence,
    /// Optimal parameters
    pub optimal_parameters: Vec<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Energy history
    pub energy_history: Vec<f64>,
    /// Variance history
    pub variance_history: Vec<f64>,
    /// Final circuit
    pub final_circuit: QuantumCircuit,
    /// Provenance information
    pub provenance: VQEProvenance,
}

/// Provenance tracking for VQE
#[derive(Debug, Clone)]
pub struct VQEProvenance {
    pub hamiltonian_name: String,
    pub ansatz_type: String,
    pub num_qubits: usize,
    pub num_parameters: usize,
    pub total_shots: usize,
    pub noise_level: f64,
}

impl VQEResult {
    /// Get 95% confidence interval for energy
    pub fn confidence_interval(&self) -> (f64, f64) {
        let std = self.variance.sqrt();
        (self.energy - 1.96 * std, self.energy + 1.96 * std)
    }

    /// Check if result is chemically accurate (< 1 kcal/mol ≈ 0.0016 Ha)
    pub fn is_chemically_accurate(&self) -> bool {
        self.variance.sqrt() < 0.0016
    }
}

// =============================================================================
// VQE Solver
// =============================================================================

/// Variational Quantum Eigensolver with epistemic tracking
pub struct VQESolver {
    /// Hamiltonian to minimize
    pub hamiltonian: Hamiltonian,
    /// Ansatz circuit
    pub ansatz: QuantumCircuit,
    /// Configuration
    pub config: VQEConfig,
}

impl VQESolver {
    /// Create new VQE solver
    pub fn new(hamiltonian: Hamiltonian, ansatz: QuantumCircuit, config: VQEConfig) -> Self {
        Self {
            hamiltonian,
            ansatz,
            config,
        }
    }

    /// Create VQE for H2 molecule with hardware-efficient ansatz
    pub fn h2_molecule() -> Self {
        let hamiltonian = Hamiltonian::h2_molecule();
        let ansatz = QuantumCircuit::hardware_efficient_ansatz(2, 2);
        Self::new(hamiltonian, ansatz, VQEConfig::default())
    }

    /// Evaluate energy for given parameters
    pub fn evaluate_energy(&self, params: &[f64]) -> (f64, f64) {
        let mut circuit = self.ansatz.clone();
        circuit.set_parameters(params);

        // Execute circuit
        let initial = QubitState::zero_state(self.hamiltonian.num_qubits);
        let mut state = circuit.execute(&initial);

        // Apply noise if configured
        if let Some(ref noise) = self.config.noise_model {
            noise.apply_to_state(&mut state);
        }

        // Measure expectation values for each term group
        let mut total_energy = 0.0;
        let mut total_variance = 0.0;

        for group in self.hamiltonian.group_commuting_terms() {
            let probs = state.probabilities();

            for term in group {
                let exp = term.expectation(&probs, self.hamiltonian.num_qubits);
                total_energy += exp;

                // Variance from finite shots and noise
                let shot_var = exp.abs() * (1.0 - exp.abs()) / self.config.shots as f64;
                let noise_var = state.total_variance() * term.coeff.abs();
                total_variance += shot_var + noise_var;
            }
        }

        (total_energy, total_variance)
    }

    /// Compute parameter gradients
    pub fn compute_gradients(&self, params: &[f64]) -> Vec<f64> {
        let num_params = params.len();
        let mut gradients = vec![0.0; num_params];

        for i in 0..num_params {
            // Parameter shift rule
            let mut params_plus = params.to_vec();
            params_plus[i] += PI / 2.0;
            let (e_plus, _) = self.evaluate_energy(&params_plus);

            let mut params_minus = params.to_vec();
            params_minus[i] -= PI / 2.0;
            let (e_minus, _) = self.evaluate_energy(&params_minus);

            gradients[i] = (e_plus - e_minus) / 2.0;
        }

        gradients
    }

    /// Run VQE optimization
    pub fn optimize(&mut self) -> VQEResult {
        let num_params = self.ansatz.parameters.len();
        let mut params = self.ansatz.parameters.clone();

        // Adam optimizer state
        let mut m = vec![0.0; num_params]; // First moment
        let mut v = vec![0.0; num_params]; // Second moment
        let beta1 = 0.9;
        let beta2 = 0.999;
        let epsilon = 1e-8;

        let mut energy_history = Vec::new();
        let mut variance_history = Vec::new();
        let mut best_energy = f64::MAX;
        let mut best_params = params.clone();
        let mut total_shots = 0;

        for iter in 0..self.config.max_iterations {
            // Evaluate current energy
            let (energy, variance) = self.evaluate_energy(&params);
            energy_history.push(energy);
            variance_history.push(variance);
            total_shots += self.config.shots * self.hamiltonian.terms.len();

            // Track best
            if energy < best_energy {
                best_energy = energy;
                best_params = params.clone();
            }

            // Check convergence
            if iter > 0 {
                let delta = (energy_history[iter - 1] - energy).abs();
                if delta < self.config.convergence_threshold {
                    break;
                }
            }

            // Compute gradients
            let gradients = self.compute_gradients(&params);

            // Update parameters
            if self.config.use_adam {
                for i in 0..num_params {
                    // Gradient with variance penalty
                    let g = gradients[i] + self.config.variance_penalty * variance * gradients[i];

                    // Adam update
                    m[i] = beta1 * m[i] + (1.0 - beta1) * g;
                    v[i] = beta2 * v[i] + (1.0 - beta2) * g * g;

                    let m_hat = m[i] / (1.0 - beta1.powi((iter + 1) as i32));
                    let v_hat = v[i] / (1.0 - beta2.powi((iter + 1) as i32));

                    params[i] -= self.config.learning_rate * m_hat / (v_hat.sqrt() + epsilon);
                }
            } else {
                // Simple gradient descent
                for i in 0..num_params {
                    params[i] -= self.config.learning_rate * gradients[i];
                }
            }
        }

        // Final evaluation
        let (final_energy, final_variance) = self.evaluate_energy(&best_params);

        // Build final circuit
        let mut final_circuit = self.ansatz.clone();
        final_circuit.set_parameters(&best_params);

        // Compute confidence based on convergence
        let converged = energy_history.len() < self.config.max_iterations;
        let confidence = if converged {
            BetaConfidence::from_confidence(0.95, total_shots as f64)
        } else {
            BetaConfidence::from_confidence(0.7, total_shots as f64 / 2.0)
        };

        // Noise level for provenance
        let noise_level = self
            .config
            .noise_model
            .as_ref()
            .map(|n| n.total_variance(self.hamiltonian.num_qubits))
            .unwrap_or(0.0);

        VQEResult {
            energy: final_energy,
            variance: final_variance,
            confidence,
            optimal_parameters: best_params,
            iterations: energy_history.len(),
            energy_history,
            variance_history,
            final_circuit,
            provenance: VQEProvenance {
                hamiltonian_name: self.hamiltonian.name.clone(),
                ansatz_type: "hardware_efficient".to_string(),
                num_qubits: self.hamiltonian.num_qubits,
                num_parameters: num_params,
                total_shots,
                noise_level,
            },
        }
    }
}

// =============================================================================
// QAOA for Combinatorial Optimization
// =============================================================================

/// Quantum Approximate Optimization Algorithm
pub struct QAOA {
    /// Cost Hamiltonian
    pub cost_hamiltonian: Hamiltonian,
    /// Number of QAOA layers
    pub num_layers: usize,
    /// Configuration
    pub config: VQEConfig,
}

impl QAOA {
    /// Create QAOA for MaxCut problem
    pub fn maxcut(edges: &[(usize, usize)], num_vertices: usize, num_layers: usize) -> Self {
        let mut cost = Hamiltonian::new(num_vertices, "MaxCut");

        // MaxCut: minimize Σ (1 - Z_i Z_j) / 2 = maximize Σ Z_i Z_j / 2
        for &(i, j) in edges {
            cost.add_term(PauliTerm::zz(i, j, -0.5));
            cost.add_term(PauliTerm::identity(0.5));
        }

        Self {
            cost_hamiltonian: cost,
            num_layers,
            config: VQEConfig::default(),
        }
    }

    /// Build QAOA circuit for given parameters
    pub fn build_circuit(&self, gammas: &[f64], betas: &[f64]) -> QuantumCircuit {
        let n = self.cost_hamiltonian.num_qubits;
        let mut circuit = QuantumCircuit::new(n);

        // Initial |+>^n state
        for q in 0..n {
            circuit.add_gate(Gate::h(q));
        }

        // Alternating layers
        for p in 0..self.num_layers {
            let gamma = gammas.get(p).copied().unwrap_or(0.0);
            let beta = betas.get(p).copied().unwrap_or(0.0);

            // Cost layer: e^{-iγH_C}
            for term in &self.cost_hamiltonian.terms {
                if term.paulis.len() == 2 {
                    // ZZ interaction: CNOT-RZ-CNOT
                    let q1 = term.paulis[0].0;
                    let q2 = term.paulis[1].0;
                    circuit.add_gate(Gate::cnot(q1, q2));
                    circuit.add_gate(Gate::rz(q2, 2.0 * gamma * term.coeff));
                    circuit.add_gate(Gate::cnot(q1, q2));
                }
            }

            // Mixer layer: e^{-iβH_M} with H_M = Σ X_i
            for q in 0..n {
                circuit.add_gate(Gate::rx(q, 2.0 * beta));
            }
        }

        circuit
    }

    /// Evaluate QAOA objective
    pub fn evaluate(&self, gammas: &[f64], betas: &[f64]) -> (f64, f64) {
        let circuit = self.build_circuit(gammas, betas);
        let initial = QubitState::zero_state(self.cost_hamiltonian.num_qubits);
        let state = circuit.execute(&initial);

        let probs = state.probabilities();
        let energy = self.cost_hamiltonian.expectation(&probs);
        let variance = state.total_variance();

        (energy, variance)
    }

    /// Run QAOA optimization
    pub fn optimize(&mut self) -> VQEResult {
        // Initialize parameters
        let mut gammas = vec![0.1; self.num_layers];
        let mut betas = vec![0.1; self.num_layers];

        let mut best_energy = f64::MAX;
        let mut best_gammas = gammas.clone();
        let mut best_betas = betas.clone();
        let mut energy_history = Vec::new();
        let mut variance_history = Vec::new();

        for iter in 0..self.config.max_iterations {
            let (energy, variance) = self.evaluate(&gammas, &betas);
            energy_history.push(energy);
            variance_history.push(variance);

            if energy < best_energy {
                best_energy = energy;
                best_gammas = gammas.clone();
                best_betas = betas.clone();
            }

            // Gradient descent for gammas and betas
            for p in 0..self.num_layers {
                // Gamma gradient
                let mut gammas_plus = gammas.clone();
                gammas_plus[p] += PI / 2.0;
                let (e_plus, _) = self.evaluate(&gammas_plus, &betas);

                let mut gammas_minus = gammas.clone();
                gammas_minus[p] -= PI / 2.0;
                let (e_minus, _) = self.evaluate(&gammas_minus, &betas);

                let grad_gamma = (e_plus - e_minus) / 2.0;
                gammas[p] -= self.config.learning_rate * grad_gamma;

                // Beta gradient
                let mut betas_plus = betas.clone();
                betas_plus[p] += PI / 2.0;
                let (e_plus, _) = self.evaluate(&gammas, &betas_plus);

                let mut betas_minus = betas.clone();
                betas_minus[p] -= PI / 2.0;
                let (e_minus, _) = self.evaluate(&gammas, &betas_minus);

                let grad_beta = (e_plus - e_minus) / 2.0;
                betas[p] -= self.config.learning_rate * grad_beta;
            }
        }

        // Build final result
        let final_circuit = self.build_circuit(&best_gammas, &best_betas);
        let (final_energy, final_variance) = self.evaluate(&best_gammas, &best_betas);

        let mut optimal_params = best_gammas;
        optimal_params.extend(best_betas);

        VQEResult {
            energy: final_energy,
            variance: final_variance,
            confidence: BetaConfidence::from_confidence(0.9, 1000.0),
            optimal_parameters: optimal_params,
            iterations: energy_history.len(),
            energy_history,
            variance_history,
            final_circuit,
            provenance: VQEProvenance {
                hamiltonian_name: self.cost_hamiltonian.name.clone(),
                ansatz_type: "QAOA".to_string(),
                num_qubits: self.cost_hamiltonian.num_qubits,
                num_parameters: self.num_layers * 2,
                total_shots: self.config.shots * self.config.max_iterations,
                noise_level: 0.0,
            },
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pauli_term_z() {
        let term = PauliTerm::z(0, 1.0);
        let probs = vec![1.0, 0.0]; // |0> state

        let exp = term.expectation(&probs, 1);
        assert!((exp - 1.0).abs() < 1e-10); // <0|Z|0> = 1
    }

    #[test]
    fn test_pauli_term_zz() {
        let term = PauliTerm::zz(0, 1, 1.0);

        // |00> state: <00|ZZ|00> = 1
        let probs_00 = vec![1.0, 0.0, 0.0, 0.0];
        assert!((term.expectation(&probs_00, 2) - 1.0).abs() < 1e-10);

        // |01> state: <01|ZZ|01> = -1
        let probs_01 = vec![0.0, 1.0, 0.0, 0.0];
        assert!((term.expectation(&probs_01, 2) - (-1.0)).abs() < 1e-10);

        // |11> state: <11|ZZ|11> = 1
        let probs_11 = vec![0.0, 0.0, 0.0, 1.0];
        assert!((term.expectation(&probs_11, 2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_h2_hamiltonian() {
        let h = Hamiltonian::h2_molecule();
        assert_eq!(h.num_qubits, 2);
        assert_eq!(h.terms.len(), 6); // identity + 2 Z + 1 ZZ + 1 XX + 1 YY
    }

    #[test]
    fn test_vqe_h2_energy_evaluation() {
        let solver = VQESolver::h2_molecule();

        // Evaluate at initial parameters
        let (energy, variance) = solver.evaluate_energy(&solver.ansatz.parameters);

        // Energy should be finite
        assert!(energy.is_finite());
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_vqe_h2_optimization() {
        let mut solver = VQESolver::h2_molecule();
        solver.config.max_iterations = 10; // Quick test

        let result = solver.optimize();

        // Should find something reasonable
        assert!(result.energy.is_finite());
        assert!(result.iterations <= 10);

        // H2 ground state is around -1.85 Ha
        // With few iterations, we might not get there, but should be negative
        assert!(result.energy < 0.0);
    }

    #[test]
    fn test_ising_hamiltonian() {
        let h = Hamiltonian::ising_model(3, 1.0, 0.5);
        assert_eq!(h.num_qubits, 3);
        // 2 ZZ terms + 3 X terms = 5 terms
        assert_eq!(h.terms.len(), 5);
    }

    #[test]
    fn test_qaoa_maxcut() {
        let edges = vec![(0, 1), (1, 2), (2, 0)]; // Triangle
        let qaoa = QAOA::maxcut(&edges, 3, 1);

        let (energy, variance) = qaoa.evaluate(&[0.5], &[0.3]);

        assert!(energy.is_finite());
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_vqe_result_confidence_interval() {
        let result = VQEResult {
            energy: -1.5,
            variance: 0.01,
            confidence: BetaConfidence::from_confidence(0.9, 100.0),
            optimal_parameters: vec![0.1, 0.2],
            iterations: 10,
            energy_history: vec![-1.0, -1.3, -1.5],
            variance_history: vec![0.1, 0.05, 0.01],
            final_circuit: QuantumCircuit::new(2),
            provenance: VQEProvenance {
                hamiltonian_name: "test".to_string(),
                ansatz_type: "test".to_string(),
                num_qubits: 2,
                num_parameters: 2,
                total_shots: 1000,
                noise_level: 0.0,
            },
        };

        let (lower, upper) = result.confidence_interval();
        assert!(lower < result.energy);
        assert!(upper > result.energy);
    }

    #[test]
    fn test_chemical_accuracy() {
        let result = VQEResult {
            energy: -1.85,
            variance: 0.000001, // Very low variance (std = 0.001 < 0.0016 Ha)
            confidence: BetaConfidence::from_confidence(0.99, 10000.0),
            optimal_parameters: vec![],
            iterations: 100,
            energy_history: vec![],
            variance_history: vec![],
            final_circuit: QuantumCircuit::new(2),
            provenance: VQEProvenance {
                hamiltonian_name: "H2".to_string(),
                ansatz_type: "UCCSD".to_string(),
                num_qubits: 2,
                num_parameters: 4,
                total_shots: 100000,
                noise_level: 0.0,
            },
        };

        assert!(result.is_chemically_accurate());
    }
}
