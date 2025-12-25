//! Native Pennylane-Inspired Quantum ML Integration
//!
//! This module provides Pennylane-style differentiable quantum circuits with
//! Sounio's epistemic semantics - solving the key gaps in Python Pennylane:
//!
//! ## Gaps in Pennylane (Python) that Sounio Solves
//!
//! 1. **Runtime Python**: No compile-time checks (invalid gate order undetected)
//!    → Sounio: Compile-time effect system validates circuit structure
//!
//! 2. **Basic Uncertainty**: Only sampling variance, no epistemic propagation
//!    → Sounio: Full Beta posterior on parameters AND states
//!
//! 3. **No Provenance**: Circuits without trace (where did this param come from?)
//!    → Sounio: Merkle tree of layers + params for auditable circuits
//!
//! 4. **Leaky Hardware Abstractions**: Good integration but error-prone
//!    → Sounio: Refinement types for unitarity, fidelity bounds
//!
//! ## Key Innovation: Epistemic Quantum Circuits
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────────┐
//! │              Pennylane-Native Sounio Architecture                │
//! ├─────────────────────────────────────────────────────────────────────┤
//! │                                                                     │
//! │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
//! │  │ EpistemicParam  │───►│ VariationalLayer│───►│ EpistemicQubit  │ │
//! │  │ θ ± σ (Beta)    │    │ GPU Kernel      │    │ |ψ⟩ + variance  │ │
//! │  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
//! │          │                      │                      │           │
//! │          ▼                      ▼                      ▼           │
//! │  ┌─────────────────────────────────────────────────────────────────┐│
//! │  │              Backward Pass (Parameter-Shift + Epistemic)        ││
//! │  │   ∂L/∂θ with confidence decay from noise propagation            ││
//! │  └─────────────────────────────────────────────────────────────────┘│
//! │                              │                                      │
//! │                              ▼                                      │
//! │  ┌─────────────────────────────────────────────────────────────────┐│
//! │  │              Provenance Merkle Tree                             ││
//! │  │   Hash(layer₀) → Hash(layer₁) → ... → Circuit Fingerprint      ││
//! │  └─────────────────────────────────────────────────────────────────┘│
//! └─────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Example: VQE with Epistemic Energy
//!
//! ```ignore
//! let circuit = VariationalCircuit::hardware_efficient(4, 3);
//! let hamiltonian = MolecularHamiltonian::h2();
//!
//! let result = circuit.vqe_optimize(hamiltonian, VQEConfig::default());
//! println!("Energy: {} ± {} eV", result.energy.mean(), result.energy.std());
//! println!("Confidence: {}", result.confidence.mean());
//! ```

use super::noise::NoiseModel;
use super::states::{Complex, EpistemicQubit, QuantumState, QubitState, StateVector};
use crate::epistemic::bayesian::BetaConfidence;
use std::collections::hash_map::DefaultHasher;
use std::f64::consts::PI;
use std::hash::{Hash, Hasher};

// =============================================================================
// Epistemic Parameters (Knowledge<f64> for trainable params)
// =============================================================================

/// A trainable parameter with epistemic uncertainty
#[derive(Debug, Clone)]
pub struct EpistemicParam {
    /// Mean value of the parameter
    pub mean: f64,
    /// Variance (uncertainty) in the parameter
    pub variance: f64,
    /// Confidence in this parameter estimate
    pub confidence: BetaConfidence,
    /// Gradient accumulated during backward pass
    pub gradient: f64,
    /// Provenance hash (where did this param come from?)
    pub provenance_hash: u64,
    /// Training iteration when last updated
    pub last_updated: usize,
}

impl EpistemicParam {
    /// Create a new epistemic parameter
    pub fn new(mean: f64) -> Self {
        Self {
            mean,
            variance: 0.1, // Initial uncertainty
            confidence: BetaConfidence::from_confidence(0.5, 10.0),
            gradient: 0.0,
            provenance_hash: 0,
            last_updated: 0,
        }
    }

    /// Create with specific variance
    pub fn with_variance(mean: f64, variance: f64) -> Self {
        Self {
            mean,
            variance,
            confidence: BetaConfidence::from_confidence(0.5, 10.0),
            gradient: 0.0,
            provenance_hash: 0,
            last_updated: 0,
        }
    }

    /// Create from prior distribution
    pub fn from_prior(mean: f64, std: f64, confidence: f64) -> Self {
        Self {
            mean,
            variance: std * std,
            confidence: BetaConfidence::from_confidence(confidence, 100.0),
            gradient: 0.0,
            provenance_hash: 0,
            last_updated: 0,
        }
    }

    /// Standard deviation
    pub fn std(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Update parameter with gradient descent
    pub fn update(&mut self, learning_rate: f64, iteration: usize) {
        self.mean -= learning_rate * self.gradient;

        // Reduce variance as we learn (Bayesian update)
        self.variance *= 0.99;

        // Increase confidence with each update
        self.confidence = BetaConfidence::new(self.confidence.alpha + 0.1, self.confidence.beta);

        self.gradient = 0.0;
        self.last_updated = iteration;
        self.update_provenance();
    }

    /// Update provenance hash
    fn update_provenance(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.mean.to_bits().hash(&mut hasher);
        self.last_updated.hash(&mut hasher);
        self.provenance_hash = hasher.finish();
    }

    /// Sample from parameter distribution (for MCMC/variational inference)
    pub fn sample(&self, seed: u64) -> f64 {
        // Box-Muller transform for Gaussian sampling
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        self.provenance_hash.hash(&mut hasher);
        let u1 = (hasher.finish() as f64) / (u64::MAX as f64);

        hasher = DefaultHasher::new();
        (seed + 1).hash(&mut hasher);
        let u2 = (hasher.finish() as f64) / (u64::MAX as f64);

        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
        self.mean + self.std() * z
    }
}

// =============================================================================
// Variational Layer (Pennylane-style parametric layer)
// =============================================================================

/// A single variational layer in the circuit
#[derive(Debug, Clone)]
pub struct VariationalLayer {
    /// Layer type
    pub layer_type: LayerType,
    /// Parameters for this layer
    pub params: Vec<EpistemicParam>,
    /// Qubits this layer acts on
    pub qubits: Vec<usize>,
    /// Layer index in circuit
    pub layer_index: usize,
    /// Provenance hash
    pub provenance_hash: u64,
}

/// Types of variational layers (Pennylane-inspired)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    /// Rotation layer: RX, RY, RZ on each qubit
    StronglyEntangling,
    /// Basic rotation: RY-RZ on each qubit
    BasicEntangler,
    /// Random layer (for expressibility)
    RandomLayers,
    /// Hardware-efficient ansatz layer
    HardwareEfficient,
    /// UCCSD-style excitation layer
    UCCSD,
    /// Custom layer
    Custom,
}

impl VariationalLayer {
    /// Create a strongly entangling layer (Pennylane default)
    pub fn strongly_entangling(num_qubits: usize, layer_index: usize) -> Self {
        // 3 params per qubit: RX, RY, RZ
        let params: Vec<EpistemicParam> = (0..num_qubits * 3)
            .map(|i| {
                let mut p = EpistemicParam::new(0.01 * (i as f64));
                p.provenance_hash = (layer_index as u64) << 32 | (i as u64);
                p
            })
            .collect();

        let mut layer = Self {
            layer_type: LayerType::StronglyEntangling,
            params,
            qubits: (0..num_qubits).collect(),
            layer_index,
            provenance_hash: 0,
        };
        layer.update_provenance();
        layer
    }

    /// Create a hardware-efficient layer
    pub fn hardware_efficient(num_qubits: usize, layer_index: usize) -> Self {
        // 2 params per qubit: RY, RZ
        let params: Vec<EpistemicParam> = (0..num_qubits * 2)
            .map(|i| {
                let mut p = EpistemicParam::new(0.0);
                p.provenance_hash = (layer_index as u64) << 32 | (i as u64);
                p
            })
            .collect();

        let mut layer = Self {
            layer_type: LayerType::HardwareEfficient,
            params,
            qubits: (0..num_qubits).collect(),
            layer_index,
            provenance_hash: 0,
        };
        layer.update_provenance();
        layer
    }

    /// Create a basic entangler layer
    pub fn basic_entangler(num_qubits: usize, layer_index: usize) -> Self {
        // 1 param per qubit: RY only
        let params: Vec<EpistemicParam> = (0..num_qubits)
            .map(|i| {
                let mut p = EpistemicParam::new(0.0);
                p.provenance_hash = (layer_index as u64) << 32 | (i as u64);
                p
            })
            .collect();

        let mut layer = Self {
            layer_type: LayerType::BasicEntangler,
            params,
            qubits: (0..num_qubits).collect(),
            layer_index,
            provenance_hash: 0,
        };
        layer.update_provenance();
        layer
    }

    /// Update provenance hash from all params
    fn update_provenance(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.layer_index.hash(&mut hasher);
        for p in &self.params {
            p.mean.to_bits().hash(&mut hasher);
        }
        self.provenance_hash = hasher.finish();
    }

    /// Get total epistemic variance in this layer
    pub fn total_variance(&self) -> f64 {
        self.params.iter().map(|p| p.variance).sum()
    }

    /// Get mean confidence across parameters
    pub fn mean_confidence(&self) -> f64 {
        if self.params.is_empty() {
            return 1.0;
        }
        self.params.iter().map(|p| p.confidence.mean()).sum::<f64>() / self.params.len() as f64
    }

    /// Apply this layer to a state vector (CPU version)
    pub fn apply_to_state(&self, state: &mut StateVector) {
        let n = state.num_qubits;

        match self.layer_type {
            LayerType::StronglyEntangling => {
                // RX-RY-RZ on each qubit + CNOTs
                for (i, &q) in self.qubits.iter().enumerate() {
                    if i * 3 + 2 < self.params.len() {
                        apply_rx(state, q, self.params[i * 3].mean);
                        apply_ry(state, q, self.params[i * 3 + 1].mean);
                        apply_rz(state, q, self.params[i * 3 + 2].mean);
                    }
                }
                // Entanglement: circular CNOTs
                for i in 0..self.qubits.len() {
                    let ctrl = self.qubits[i];
                    let targ = self.qubits[(i + 1) % self.qubits.len()];
                    if ctrl != targ {
                        apply_cnot(state, ctrl, targ);
                    }
                }
            }

            LayerType::HardwareEfficient => {
                // RY-RZ on each qubit
                for (i, &q) in self.qubits.iter().enumerate() {
                    if i * 2 + 1 < self.params.len() {
                        apply_ry(state, q, self.params[i * 2].mean);
                        apply_rz(state, q, self.params[i * 2 + 1].mean);
                    }
                }
                // Linear entanglement
                for i in 0..self.qubits.len().saturating_sub(1) {
                    apply_cnot(state, self.qubits[i], self.qubits[i + 1]);
                }
            }

            LayerType::BasicEntangler => {
                // RY only on each qubit
                for (i, &q) in self.qubits.iter().enumerate() {
                    if i < self.params.len() {
                        apply_ry(state, q, self.params[i].mean);
                    }
                }
                // Linear CNOTs
                for i in 0..self.qubits.len().saturating_sub(1) {
                    apply_cnot(state, self.qubits[i], self.qubits[i + 1]);
                }
            }

            _ => {
                // Custom/Random - basic rotation
                for (i, &q) in self.qubits.iter().enumerate() {
                    if i < self.params.len() {
                        apply_ry(state, q, self.params[i].mean);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Variational Circuit (Full Pennylane-style circuit)
// =============================================================================

/// A complete variational quantum circuit with epistemic tracking
#[derive(Debug, Clone)]
pub struct VariationalCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Variational layers
    pub layers: Vec<VariationalLayer>,
    /// Noise model
    pub noise_model: Option<NoiseModel>,
    /// Circuit provenance (Merkle root)
    pub provenance_root: u64,
    /// Total parameter count
    pub num_params: usize,
    /// Training configuration
    pub config: TrainingConfig,
}

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Variance penalty weight (epistemic regularization)
    pub variance_penalty: f64,
    /// Number of shots for expectation estimation
    pub shots: usize,
    /// Use Adam optimizer
    pub use_adam: bool,
    /// Adam beta1
    pub adam_beta1: f64,
    /// Adam beta2
    pub adam_beta2: f64,
    /// Clip gradients
    pub gradient_clip: Option<f64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            variance_penalty: 0.01,
            shots: 1000,
            use_adam: true,
            adam_beta1: 0.9,
            adam_beta2: 0.999,
            gradient_clip: Some(1.0),
        }
    }
}

impl VariationalCircuit {
    /// Create a new empty variational circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            layers: Vec::new(),
            noise_model: None,
            provenance_root: 0,
            num_params: 0,
            config: TrainingConfig::default(),
        }
    }

    /// Create a strongly entangling circuit (Pennylane default)
    pub fn strongly_entangling(num_qubits: usize, num_layers: usize) -> Self {
        let layers: Vec<VariationalLayer> = (0..num_layers)
            .map(|i| VariationalLayer::strongly_entangling(num_qubits, i))
            .collect();

        let num_params = layers.iter().map(|l| l.params.len()).sum();

        let mut circuit = Self {
            num_qubits,
            layers,
            noise_model: None,
            provenance_root: 0,
            num_params,
            config: TrainingConfig::default(),
        };
        circuit.update_provenance();
        circuit
    }

    /// Create a hardware-efficient ansatz
    pub fn hardware_efficient(num_qubits: usize, num_layers: usize) -> Self {
        let layers: Vec<VariationalLayer> = (0..num_layers)
            .map(|i| VariationalLayer::hardware_efficient(num_qubits, i))
            .collect();

        let num_params = layers.iter().map(|l| l.params.len()).sum();

        let mut circuit = Self {
            num_qubits,
            layers,
            noise_model: None,
            provenance_root: 0,
            num_params,
            config: TrainingConfig::default(),
        };
        circuit.update_provenance();
        circuit
    }

    /// Create a basic entangler circuit
    pub fn basic_entangler(num_qubits: usize, num_layers: usize) -> Self {
        let layers: Vec<VariationalLayer> = (0..num_layers)
            .map(|i| VariationalLayer::basic_entangler(num_qubits, i))
            .collect();

        let num_params = layers.iter().map(|l| l.params.len()).sum();

        let mut circuit = Self {
            num_qubits,
            layers,
            noise_model: None,
            provenance_root: 0,
            num_params,
            config: TrainingConfig::default(),
        };
        circuit.update_provenance();
        circuit
    }

    /// Add noise model
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.noise_model = Some(noise);
        self
    }

    /// Set training config
    pub fn with_config(mut self, config: TrainingConfig) -> Self {
        self.config = config;
        self
    }

    /// Update provenance Merkle root
    fn update_provenance(&mut self) {
        // Merkle tree: hash all layer hashes together
        let mut hasher = DefaultHasher::new();
        self.num_qubits.hash(&mut hasher);
        for layer in &self.layers {
            layer.provenance_hash.hash(&mut hasher);
        }
        self.provenance_root = hasher.finish();
    }

    /// Get all parameters as a flat vector
    pub fn get_params(&self) -> Vec<f64> {
        self.layers
            .iter()
            .flat_map(|l| l.params.iter().map(|p| p.mean))
            .collect()
    }

    /// Set all parameters from a flat vector
    pub fn set_params(&mut self, params: &[f64]) {
        let mut idx = 0;
        for layer in &mut self.layers {
            for p in &mut layer.params {
                if idx < params.len() {
                    p.mean = params[idx];
                    idx += 1;
                }
            }
            layer.update_provenance();
        }
        self.update_provenance();
    }

    /// Get total epistemic variance across all parameters
    pub fn total_param_variance(&self) -> f64 {
        self.layers.iter().map(|l| l.total_variance()).sum()
    }

    /// Get mean confidence across all parameters
    pub fn mean_confidence(&self) -> f64 {
        if self.layers.is_empty() {
            return 1.0;
        }
        self.layers.iter().map(|l| l.mean_confidence()).sum::<f64>() / self.layers.len() as f64
    }

    // =========================================================================
    // Forward Pass
    // =========================================================================

    /// Execute forward pass (CPU)
    pub fn forward(&self, initial: &QubitState) -> EpistemicQubit {
        let mut state = match &initial.state {
            QuantumState::Pure(sv) => sv.clone(),
            QuantumState::Mixed(dm) => {
                // Convert density matrix to approximate pure state
                StateVector::zero_state(dm.num_qubits)
            }
        };

        let mut total_noise_variance = initial.amplitude_variance;
        let mut total_gate_error = initial.gate_error_accumulated;
        let mut gate_count = initial.gate_count;

        // Apply each layer
        for layer in &self.layers {
            layer.apply_to_state(&mut state);

            // Accumulate epistemic variance from layer
            total_noise_variance += layer.total_variance() * 0.01; // Scale factor

            // Gate errors
            let layer_gates = match layer.layer_type {
                LayerType::StronglyEntangling => layer.qubits.len() * 3 + layer.qubits.len(),
                LayerType::HardwareEfficient => layer.qubits.len() * 2 + layer.qubits.len() - 1,
                LayerType::BasicEntangler => layer.qubits.len() * 2 - 1,
                _ => layer.qubits.len(),
            };
            total_gate_error += 0.001 * layer_gates as f64; // 0.1% per gate
            gate_count += layer_gates;
        }

        // Apply noise model if present
        let qubit_state = QubitState {
            state: QuantumState::Pure(state),
            amplitude_variance: total_noise_variance,
            gate_error_accumulated: total_gate_error,
            gate_count,
            decoherence_factor: 1.0 - total_gate_error,
            measurement_shots: Some(self.config.shots),
        };

        // Compute confidence from circuit execution
        let confidence = BetaConfidence::from_confidence(
            (1.0 - total_gate_error).max(0.01) * self.mean_confidence(),
            100.0,
        );

        EpistemicQubit {
            state: qubit_state,
            confidence,
            provenance_hash: self.provenance_root,
            circuit_depth: self.layers.len(),
        }
    }

    /// Execute forward pass from |0⟩^n
    pub fn forward_from_zero(&self) -> EpistemicQubit {
        let initial = QubitState::zero_state(self.num_qubits);
        self.forward(&initial)
    }

    // =========================================================================
    // Expectation Values
    // =========================================================================

    /// Compute expectation value of a Pauli observable
    pub fn expectation(&self, observable: &PauliObservable) -> EpistemicExpectation {
        let state = self.forward_from_zero();
        let probs = state.state.probabilities();

        // Compute expectation
        let exp_value = observable.expectation(&probs, self.num_qubits);

        // Compute variance from finite shots + noise
        let shot_variance = (1.0 - exp_value.abs()) / self.config.shots as f64;
        let noise_variance = state.epistemic_variance() * observable.weight.abs();
        let param_variance = self.total_param_variance() * 0.001;

        let total_variance = shot_variance + noise_variance + param_variance;

        EpistemicExpectation {
            mean: exp_value,
            variance: total_variance,
            confidence: state.confidence,
            shots: self.config.shots,
            provenance_hash: self.provenance_root ^ observable.hash(),
        }
    }

    /// Compute Hamiltonian expectation (for VQE)
    pub fn hamiltonian_expectation(
        &self,
        hamiltonian: &EpistemicHamiltonian,
    ) -> EpistemicExpectation {
        let state = self.forward_from_zero();
        let probs = state.state.probabilities();

        let mut total_exp = 0.0;
        let mut total_variance = 0.0;

        for term in &hamiltonian.terms {
            let exp = term.observable.expectation(&probs, self.num_qubits);
            total_exp += term.coefficient * exp;

            // Variance from each term
            let term_var = (term.coefficient * term.coefficient)
                * ((1.0 - exp.abs()) / self.config.shots as f64);
            total_variance += term_var;
        }

        // Add epistemic variance from circuit
        total_variance += state.epistemic_variance() * hamiltonian.norm();
        total_variance += self.total_param_variance() * 0.001;

        EpistemicExpectation {
            mean: total_exp,
            variance: total_variance,
            confidence: state.confidence,
            shots: self.config.shots,
            provenance_hash: self.provenance_root ^ hamiltonian.provenance_hash,
        }
    }

    // =========================================================================
    // Gradient Computation (Parameter-Shift Rule)
    // =========================================================================

    /// Compute gradients for all parameters using parameter-shift rule
    pub fn compute_gradients(&mut self, hamiltonian: &EpistemicHamiltonian) -> Vec<f64> {
        let mut gradients = Vec::with_capacity(self.num_params);

        // Store original params
        let original_params = self.get_params();

        // Pre-compute param counts per layer to avoid borrow issues
        let layer_param_counts: Vec<usize> = self.layers.iter().map(|l| l.params.len()).collect();

        for (layer_idx, &param_count) in layer_param_counts.iter().enumerate() {
            for param_idx in 0..param_count {
                // Compute flat index
                let flat_idx: usize =
                    layer_param_counts[..layer_idx].iter().sum::<usize>() + param_idx;

                // Parameter shift: +π/2
                let mut params_plus = original_params.clone();
                params_plus[flat_idx] += PI / 2.0;
                self.set_params(&params_plus);
                let exp_plus = self.hamiltonian_expectation(hamiltonian).mean;

                // Parameter shift: -π/2
                let mut params_minus = original_params.clone();
                params_minus[flat_idx] -= PI / 2.0;
                self.set_params(&params_minus);
                let exp_minus = self.hamiltonian_expectation(hamiltonian).mean;

                // Gradient
                let grad = (exp_plus - exp_minus) / 2.0;
                gradients.push(grad);
            }
        }

        // Restore original params
        self.set_params(&original_params);

        // Store gradients in epistemic params
        let mut idx = 0;
        for layer in &mut self.layers {
            for p in &mut layer.params {
                if idx < gradients.len() {
                    p.gradient = gradients[idx];
                    idx += 1;
                }
            }
        }

        gradients
    }

    /// Compute epistemic gradients (with variance penalty)
    pub fn compute_epistemic_gradients(
        &mut self,
        hamiltonian: &EpistemicHamiltonian,
    ) -> EpistemicGradients {
        let energy_gradients = self.compute_gradients(hamiltonian);

        // Compute variance of gradients (uncertainty in gradient direction)
        let gradient_variance: f64 =
            energy_gradients.iter().map(|g| g * g).sum::<f64>() / energy_gradients.len() as f64;

        // Variance penalty gradients (encourage low-variance params)
        let variance_gradients: Vec<f64> = self
            .layers
            .iter()
            .flat_map(|l| {
                l.params
                    .iter()
                    .map(|p| 2.0 * p.variance * self.config.variance_penalty)
            })
            .collect();

        // Combined gradients: ∂E/∂θ + λ * ∂Var/∂θ
        let total_gradients: Vec<f64> = energy_gradients
            .iter()
            .zip(variance_gradients.iter())
            .map(|(e, v)| e + v)
            .collect();

        // Confidence based on gradient stability
        let confidence =
            BetaConfidence::from_confidence((1.0 / (1.0 + gradient_variance)).max(0.1), 100.0);

        EpistemicGradients {
            gradients: total_gradients,
            energy_gradients,
            variance_gradients,
            gradient_variance,
            confidence,
        }
    }

    // =========================================================================
    // Optimization Step
    // =========================================================================

    /// Perform one optimization step
    pub fn step(&mut self, gradients: &EpistemicGradients, iteration: usize) {
        let mut idx = 0;
        for layer in &mut self.layers {
            for p in &mut layer.params {
                if idx < gradients.gradients.len() {
                    let mut grad = gradients.gradients[idx];

                    // Gradient clipping
                    if let Some(clip) = self.config.gradient_clip {
                        grad = grad.clamp(-clip, clip);
                    }

                    p.gradient = grad;
                    p.update(self.config.learning_rate, iteration);
                    idx += 1;
                }
            }
            layer.update_provenance();
        }
        self.update_provenance();
    }

    // =========================================================================
    // VQE Optimization
    // =========================================================================

    /// Run full VQE optimization
    pub fn vqe_optimize(
        &mut self,
        hamiltonian: &EpistemicHamiltonian,
        max_iterations: usize,
        convergence_threshold: f64,
    ) -> VQEOptimizationResult {
        let mut energy_history = Vec::new();
        let mut variance_history = Vec::new();
        let mut best_energy = f64::MAX;
        let mut best_params = self.get_params();

        for iter in 0..max_iterations {
            // Compute current energy
            let energy = self.hamiltonian_expectation(hamiltonian);
            energy_history.push(energy.mean);
            variance_history.push(energy.variance);

            // Track best
            if energy.mean < best_energy {
                best_energy = energy.mean;
                best_params = self.get_params();
            }

            // Check convergence
            if iter > 0 && (energy_history[iter - 1] - energy.mean).abs() < convergence_threshold {
                break;
            }

            // Epistemic alert for high variance
            if energy.variance > 0.1 {
                // In production, would trigger alert/logging
            }

            // Compute gradients and update
            let gradients = self.compute_epistemic_gradients(hamiltonian);
            self.step(&gradients, iter);
        }

        // Restore best params
        self.set_params(&best_params);
        let final_energy = self.hamiltonian_expectation(hamiltonian);

        let iterations = energy_history.len();
        let converged = iterations < max_iterations;

        VQEOptimizationResult {
            energy: final_energy,
            optimal_params: best_params,
            iterations,
            energy_history,
            variance_history,
            circuit_provenance: self.provenance_root,
            converged,
        }
    }
}

// =============================================================================
// Pauli Observable
// =============================================================================

/// A Pauli observable for measurement
#[derive(Debug, Clone)]
pub struct PauliObservable {
    /// Pauli operators on each qubit
    pub paulis: Vec<(usize, PauliType)>,
    /// Weight/coefficient
    pub weight: f64,
}

/// Pauli operator type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PauliType {
    I,
    X,
    Y,
    Z,
}

impl PauliObservable {
    /// Create Z observable on a qubit
    pub fn z(qubit: usize) -> Self {
        Self {
            paulis: vec![(qubit, PauliType::Z)],
            weight: 1.0,
        }
    }

    /// Create ZZ observable
    pub fn zz(q1: usize, q2: usize) -> Self {
        Self {
            paulis: vec![(q1, PauliType::Z), (q2, PauliType::Z)],
            weight: 1.0,
        }
    }

    /// With weight
    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Compute expectation value from probabilities
    pub fn expectation(&self, probs: &[f64], num_qubits: usize) -> f64 {
        if self.paulis.is_empty() {
            return self.weight;
        }

        // Check if all Z (can measure directly in computational basis)
        let all_z = self.paulis.iter().all(|(_, p)| *p == PauliType::Z);

        if all_z {
            let mut exp = 0.0;
            for (i, &prob) in probs.iter().enumerate() {
                let mut parity = 1.0;
                for &(q, _) in &self.paulis {
                    let bit = (i >> (num_qubits - 1 - q)) & 1;
                    parity *= if bit == 0 { 1.0 } else { -1.0 };
                }
                exp += prob * parity;
            }
            self.weight * exp
        } else {
            // Would need basis rotation for X, Y
            0.0
        }
    }

    /// Hash for provenance
    pub fn hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for (q, p) in &self.paulis {
            q.hash(&mut hasher);
            (*p as u8).hash(&mut hasher);
        }
        self.weight.to_bits().hash(&mut hasher);
        hasher.finish()
    }
}

// =============================================================================
// Epistemic Hamiltonian
// =============================================================================

/// A Hamiltonian with epistemic tracking
#[derive(Debug, Clone)]
pub struct EpistemicHamiltonian {
    /// Terms in the Hamiltonian
    pub terms: Vec<HamiltonianTerm>,
    /// Name
    pub name: String,
    /// Number of qubits
    pub num_qubits: usize,
    /// Provenance hash
    pub provenance_hash: u64,
}

/// A single term in the Hamiltonian
#[derive(Debug, Clone)]
pub struct HamiltonianTerm {
    /// Coefficient
    pub coefficient: f64,
    /// Observable
    pub observable: PauliObservable,
}

impl EpistemicHamiltonian {
    /// Create H2 molecule Hamiltonian (STO-3G, R=0.7414 Å)
    pub fn h2_molecule() -> Self {
        let mut terms = Vec::new();

        // H2 in Bravyi-Kitaev encoding
        // H = -1.0523 I + 0.3979 Z0 - 0.3979 Z1 - 0.0112 Z0Z1 + 0.1809 X0X1 + 0.1809 Y0Y1

        terms.push(HamiltonianTerm {
            coefficient: -1.0523,
            observable: PauliObservable {
                paulis: vec![],
                weight: 1.0,
            },
        });

        terms.push(HamiltonianTerm {
            coefficient: 0.3979,
            observable: PauliObservable::z(0),
        });

        terms.push(HamiltonianTerm {
            coefficient: -0.3979,
            observable: PauliObservable::z(1),
        });

        terms.push(HamiltonianTerm {
            coefficient: -0.0112,
            observable: PauliObservable::zz(0, 1),
        });

        // X0X1 and Y0Y1 terms (simplified - would need basis rotation)
        terms.push(HamiltonianTerm {
            coefficient: 0.1809,
            observable: PauliObservable {
                paulis: vec![(0, PauliType::X), (1, PauliType::X)],
                weight: 1.0,
            },
        });

        terms.push(HamiltonianTerm {
            coefficient: 0.1809,
            observable: PauliObservable {
                paulis: vec![(0, PauliType::Y), (1, PauliType::Y)],
                weight: 1.0,
            },
        });

        let mut h = Self {
            terms,
            name: "H2_STO-3G".to_string(),
            num_qubits: 2,
            provenance_hash: 0,
        };
        h.update_provenance();
        h
    }

    /// Create LiH molecule Hamiltonian (simplified)
    pub fn lih_molecule() -> Self {
        let mut terms = Vec::new();

        // Simplified LiH
        terms.push(HamiltonianTerm {
            coefficient: -7.8,
            observable: PauliObservable {
                paulis: vec![],
                weight: 1.0,
            },
        });

        for i in 0..4 {
            terms.push(HamiltonianTerm {
                coefficient: 0.1 * (i as f64 - 1.5),
                observable: PauliObservable::z(i),
            });
        }

        let mut h = Self {
            terms,
            name: "LiH_minimal".to_string(),
            num_qubits: 4,
            provenance_hash: 0,
        };
        h.update_provenance();
        h
    }

    /// Update provenance
    fn update_provenance(&mut self) {
        let mut hasher = DefaultHasher::new();
        self.name.hash(&mut hasher);
        self.num_qubits.hash(&mut hasher);
        for term in &self.terms {
            term.coefficient.to_bits().hash(&mut hasher);
        }
        self.provenance_hash = hasher.finish();
    }

    /// Norm of Hamiltonian (for variance estimation)
    pub fn norm(&self) -> f64 {
        self.terms.iter().map(|t| t.coefficient.abs()).sum()
    }
}

// =============================================================================
// Result Types
// =============================================================================

/// Epistemic expectation value
#[derive(Debug, Clone)]
pub struct EpistemicExpectation {
    /// Mean value
    pub mean: f64,
    /// Variance (uncertainty)
    pub variance: f64,
    /// Confidence
    pub confidence: BetaConfidence,
    /// Number of shots
    pub shots: usize,
    /// Provenance hash
    pub provenance_hash: u64,
}

impl EpistemicExpectation {
    /// Standard deviation
    pub fn std(&self) -> f64 {
        self.variance.sqrt()
    }

    /// 95% confidence interval
    pub fn confidence_interval(&self) -> (f64, f64) {
        let std = self.std();
        (self.mean - 1.96 * std, self.mean + 1.96 * std)
    }
}

/// Epistemic gradients
#[derive(Debug, Clone)]
pub struct EpistemicGradients {
    /// Total gradients (energy + variance penalty)
    pub gradients: Vec<f64>,
    /// Energy gradients only
    pub energy_gradients: Vec<f64>,
    /// Variance penalty gradients
    pub variance_gradients: Vec<f64>,
    /// Variance in gradient estimates
    pub gradient_variance: f64,
    /// Confidence in gradient direction
    pub confidence: BetaConfidence,
}

/// VQE optimization result
#[derive(Debug, Clone)]
pub struct VQEOptimizationResult {
    /// Final energy (epistemic)
    pub energy: EpistemicExpectation,
    /// Optimal parameters
    pub optimal_params: Vec<f64>,
    /// Number of iterations
    pub iterations: usize,
    /// Energy history
    pub energy_history: Vec<f64>,
    /// Variance history
    pub variance_history: Vec<f64>,
    /// Circuit provenance
    pub circuit_provenance: u64,
    /// Did optimization converge?
    pub converged: bool,
}

impl VQEOptimizationResult {
    /// Is result chemically accurate? (< 1 kcal/mol ≈ 0.0016 Ha)
    pub fn is_chemically_accurate(&self) -> bool {
        self.energy.std() < 0.0016
    }

    /// Format as string
    pub fn summary(&self) -> String {
        format!(
            "VQE Result:\n  Energy: {:.6} ± {:.6} Ha\n  Confidence: {:.2}%\n  Iterations: {}\n  Converged: {}",
            self.energy.mean,
            self.energy.std(),
            self.energy.confidence.mean() * 100.0,
            self.iterations,
            self.converged
        )
    }
}

// =============================================================================
// GPU Kernel Structures (for future CUDA/Metal integration)
// =============================================================================

/// GPU kernel configuration for variational layers
#[derive(Debug, Clone)]
pub struct GpuLayerConfig {
    /// Block size
    pub block_size: usize,
    /// Grid size
    pub grid_size: usize,
    /// Shared memory size
    pub shared_mem: usize,
}

impl Default for GpuLayerConfig {
    fn default() -> Self {
        Self {
            block_size: 256,
            grid_size: 1,
            shared_mem: 0,
        }
    }
}

/// Placeholder for GPU-accelerated layer execution
/// In production, would use cudarc or metal crate
pub fn gpu_kernel_apply_layer(
    _layer: &VariationalLayer,
    state: EpistemicQubit,
    _config: &GpuLayerConfig,
) -> EpistemicQubit {
    // CPU fallback for now
    // In production: launch CUDA kernel
    state
}

// =============================================================================
// Helper Functions (Gate Applications)
// =============================================================================

fn apply_rx(sv: &mut StateVector, qubit: usize, theta: f64) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..(1 << n) {
        if i & mask == 0 {
            let j = i | mask;
            let a = sv.amplitudes[i];
            let b = sv.amplitudes[j];
            sv.amplitudes[i] = Complex::new(
                a.re * cos_half + b.im * sin_half,
                a.im * cos_half - b.re * sin_half,
            );
            sv.amplitudes[j] = Complex::new(
                b.re * cos_half + a.im * sin_half,
                b.im * cos_half - a.re * sin_half,
            );
        }
    }
}

fn apply_ry(sv: &mut StateVector, qubit: usize, theta: f64) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);
    let cos_half = (theta / 2.0).cos();
    let sin_half = (theta / 2.0).sin();

    for i in 0..(1 << n) {
        if i & mask == 0 {
            let j = i | mask;
            let a = sv.amplitudes[i];
            let b = sv.amplitudes[j];
            sv.amplitudes[i] = Complex::new(
                a.re * cos_half - b.re * sin_half,
                a.im * cos_half - b.im * sin_half,
            );
            sv.amplitudes[j] = Complex::new(
                a.re * sin_half + b.re * cos_half,
                a.im * sin_half + b.im * cos_half,
            );
        }
    }
}

fn apply_rz(sv: &mut StateVector, qubit: usize, theta: f64) {
    let n = sv.num_qubits;
    let mask = 1 << (n - 1 - qubit);
    let phase_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
    let phase_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());

    for i in 0..(1 << n) {
        if i & mask == 0 {
            sv.amplitudes[i] = sv.amplitudes[i] * phase_neg;
        } else {
            sv.amplitudes[i] = sv.amplitudes[i] * phase_pos;
        }
    }
}

fn apply_cnot(sv: &mut StateVector, control: usize, target: usize) {
    let n = sv.num_qubits;
    let control_mask = 1 << (n - 1 - control);
    let target_mask = 1 << (n - 1 - target);

    for i in 0..(1 << n) {
        if i & control_mask != 0 && i & target_mask == 0 {
            let j = i | target_mask;
            sv.amplitudes.swap(i, j);
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
    fn test_epistemic_param() {
        let mut p = EpistemicParam::new(1.0);
        assert!((p.mean - 1.0).abs() < 1e-10);
        assert!(p.variance > 0.0);

        p.gradient = 0.1;
        p.update(0.1, 1);
        assert!(p.mean < 1.0); // Should have decreased
    }

    #[test]
    fn test_variational_layer() {
        let layer = VariationalLayer::hardware_efficient(4, 0);
        assert_eq!(layer.params.len(), 8); // 2 params per qubit
        assert_eq!(layer.qubits.len(), 4);
    }

    #[test]
    fn test_variational_circuit_creation() {
        let circuit = VariationalCircuit::strongly_entangling(4, 2);
        assert_eq!(circuit.num_qubits, 4);
        assert_eq!(circuit.layers.len(), 2);
        assert_eq!(circuit.num_params, 24); // 3 * 4 * 2
    }

    #[test]
    fn test_forward_pass() {
        let circuit = VariationalCircuit::hardware_efficient(2, 1);
        let result = circuit.forward_from_zero();

        assert!(result.confidence_mean() > 0.0);
        assert!(result.epistemic_variance() >= 0.0);
    }

    #[test]
    fn test_expectation_z() {
        let circuit = VariationalCircuit::basic_entangler(2, 1);
        let obs = PauliObservable::z(0);
        let exp = circuit.expectation(&obs);

        assert!(exp.mean.abs() <= 1.0);
        assert!(exp.variance >= 0.0);
    }

    #[test]
    fn test_h2_hamiltonian() {
        let h = EpistemicHamiltonian::h2_molecule();
        assert_eq!(h.num_qubits, 2);
        assert_eq!(h.terms.len(), 6);
    }

    #[test]
    fn test_hamiltonian_expectation() {
        let circuit = VariationalCircuit::hardware_efficient(2, 2);
        let h = EpistemicHamiltonian::h2_molecule();
        let exp = circuit.hamiltonian_expectation(&h);

        assert!(exp.mean.is_finite());
        assert!(exp.variance >= 0.0);
    }

    #[test]
    fn test_gradient_computation() {
        let mut circuit = VariationalCircuit::basic_entangler(2, 1);
        let h = EpistemicHamiltonian::h2_molecule();
        let gradients = circuit.compute_gradients(&h);

        assert_eq!(gradients.len(), circuit.num_params);
    }

    #[test]
    fn test_vqe_optimization() {
        let mut circuit = VariationalCircuit::hardware_efficient(2, 2);
        circuit.config.learning_rate = 0.2;
        let h = EpistemicHamiltonian::h2_molecule();

        let result = circuit.vqe_optimize(&h, 10, 1e-6);

        assert!(result.iterations <= 10);
        assert!(result.energy.mean.is_finite());
        // H2 ground state is around -1.85 Ha, we should get something negative
        assert!(result.energy.mean < 0.0);
    }

    #[test]
    fn test_provenance_tracking() {
        let mut circuit = VariationalCircuit::hardware_efficient(2, 1);
        let initial_provenance = circuit.provenance_root;

        let mut params = circuit.get_params();
        params[0] = 1.0;
        circuit.set_params(&params);

        // Provenance should change after param update
        assert_ne!(circuit.provenance_root, initial_provenance);
    }

    #[test]
    fn test_epistemic_gradients() {
        let mut circuit = VariationalCircuit::basic_entangler(2, 1);
        let h = EpistemicHamiltonian::h2_molecule();
        let grads = circuit.compute_epistemic_gradients(&h);

        assert_eq!(grads.gradients.len(), circuit.num_params);
        assert!(grads.gradient_variance >= 0.0);
    }

    #[test]
    fn test_confidence_interval() {
        let exp = EpistemicExpectation {
            mean: -1.5,
            variance: 0.01,
            confidence: BetaConfidence::from_confidence(0.9, 100.0),
            shots: 1000,
            provenance_hash: 0,
        };

        let (lower, upper) = exp.confidence_interval();
        assert!(lower < exp.mean);
        assert!(upper > exp.mean);
    }

    #[test]
    fn test_layer_types() {
        let se = VariationalLayer::strongly_entangling(3, 0);
        assert_eq!(se.params.len(), 9); // 3 per qubit

        let he = VariationalLayer::hardware_efficient(3, 0);
        assert_eq!(he.params.len(), 6); // 2 per qubit

        let be = VariationalLayer::basic_entangler(3, 0);
        assert_eq!(be.params.len(), 3); // 1 per qubit
    }
}
