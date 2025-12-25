//! Quantum Circuit with Differentiable Execution
//!
//! Circuits track:
//! - Gate sequence with error accumulation
//! - Parameter gradients via parameter-shift rule
//! - Epistemic variance from noise
//! - Circuit depth and two-qubit gate count

use super::gates::{Gate, ParametricGate};
use super::noise::NoiseModel;
use super::states::{EpistemicQubit, QubitState};
use crate::epistemic::bayesian::BetaConfidence;
use std::collections::HashMap;
use std::f64::consts::PI;

// =============================================================================
// Quantum Circuit
// =============================================================================

/// A quantum circuit with epistemic tracking
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    /// Number of qubits
    pub num_qubits: usize,
    /// Gate sequence
    pub gates: Vec<Gate>,
    /// Parametric gate indices (for VQE)
    pub parametric_gates: Vec<ParametricGate>,
    /// Current parameter vector
    pub parameters: Vec<f64>,
    /// Noise model (None = ideal)
    pub noise_model: Option<NoiseModel>,
    /// Circuit metadata
    pub metadata: CircuitMetadata,
}

/// Circuit metadata for analysis
#[derive(Debug, Clone, Default)]
pub struct CircuitMetadata {
    pub name: Option<String>,
    pub description: Option<String>,
    pub created_at: Option<u64>,
}

/// Statistics about a circuit
#[derive(Debug, Clone)]
pub struct CircuitStats {
    pub num_qubits: usize,
    pub depth: usize,
    pub gate_count: usize,
    pub two_qubit_gate_count: usize,
    pub three_qubit_gate_count: usize,
    pub parameter_count: usize,
    pub estimated_error: f64,
}

impl QuantumCircuit {
    /// Create an empty circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            parametric_gates: Vec::new(),
            parameters: Vec::new(),
            noise_model: None,
            metadata: CircuitMetadata::default(),
        }
    }

    /// Create circuit with noise model
    pub fn with_noise(num_qubits: usize, noise: NoiseModel) -> Self {
        Self {
            num_qubits,
            gates: Vec::new(),
            parametric_gates: Vec::new(),
            parameters: Vec::new(),
            noise_model: Some(noise),
            metadata: CircuitMetadata::default(),
        }
    }

    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: Gate) {
        // Validate qubit indices
        for &q in gate.all_qubits().iter() {
            assert!(q < self.num_qubits, "Qubit index {} out of range", q);
        }
        self.gates.push(gate);
    }

    /// Add a parametric gate with parameter tracking
    pub fn add_parametric_gate(&mut self, gate: Gate, param_indices: Vec<usize>) {
        // Extend parameters if needed
        for &idx in &param_indices {
            while self.parameters.len() <= idx {
                self.parameters.push(0.0);
            }
        }

        // Set initial parameters
        for (i, &idx) in param_indices.iter().enumerate() {
            if i < gate.params.len() {
                self.parameters[idx] = gate.params[i];
            }
        }

        let mut pg = ParametricGate::new(gate.clone(), param_indices);
        pg.gate_index = self.gates.len(); // Track where in gates array
        self.gates.push(gate);
        self.parametric_gates.push(pg);
    }

    /// Update all parameters
    pub fn set_parameters(&mut self, params: &[f64]) {
        self.parameters = params.to_vec();

        // Update parametric gates and sync to gates array
        for pg in &mut self.parametric_gates {
            pg.update_params(params);

            // Sync parameters to the actual gate in the gates array
            let gate_idx = pg.gate_index;
            if gate_idx < self.gates.len() {
                for (i, &param_idx) in pg.param_indices.iter().enumerate() {
                    if param_idx < params.len() && i < self.gates[gate_idx].params.len() {
                        self.gates[gate_idx].params[i] = params[param_idx];
                    }
                }
            }
        }
    }

    /// Get circuit statistics
    pub fn stats(&self) -> CircuitStats {
        let mut depth_map: HashMap<usize, usize> = HashMap::new();
        let mut two_qubit = 0;
        let mut three_qubit = 0;
        let mut total_error = 0.0;

        for gate in &self.gates {
            let qubits = gate.all_qubits();
            let max_depth = qubits
                .iter()
                .map(|&q| *depth_map.get(&q).unwrap_or(&0))
                .max()
                .unwrap_or(0);

            let new_depth = max_depth + 1;
            for &q in &qubits {
                depth_map.insert(q, new_depth);
            }

            match gate.gate_type.num_qubits() {
                2 => two_qubit += 1,
                3 => three_qubit += 1,
                _ => {}
            }

            total_error += gate.get_error_rate();
        }

        let depth = depth_map.values().copied().max().unwrap_or(0);

        CircuitStats {
            num_qubits: self.num_qubits,
            depth,
            gate_count: self.gates.len(),
            two_qubit_gate_count: two_qubit,
            three_qubit_gate_count: three_qubit,
            parameter_count: self.parameters.len(),
            estimated_error: total_error,
        }
    }

    /// Execute the circuit on an initial state
    pub fn execute(&self, initial_state: &QubitState) -> QubitState {
        let mut state = initial_state.clone();

        for gate in &self.gates {
            gate.apply(&mut state);
        }

        // Apply noise model if present
        if let Some(ref noise) = self.noise_model {
            noise.apply_to_state(&mut state);
        }

        state
    }

    /// Execute circuit returning epistemic qubit
    pub fn execute_epistemic(&self, initial: &EpistemicQubit) -> EpistemicQubit {
        let state = self.execute(&initial.state);
        let stats = self.stats();

        // Compute new confidence based on circuit execution
        let error_factor = 1.0 - stats.estimated_error.min(0.99);
        let new_confidence = initial.confidence_mean() * error_factor;

        EpistemicQubit {
            state,
            confidence: BetaConfidence::from_confidence(new_confidence.max(0.01), 100.0),
            provenance_hash: initial.provenance_hash ^ (stats.gate_count as u64),
            circuit_depth: initial.circuit_depth + stats.depth,
        }
    }

    /// Compute gradient via parameter-shift rule
    ///
    /// For each parameter θ:
    /// ∂f/∂θ = (f(θ + π/2) - f(θ - π/2)) / 2
    pub fn parameter_gradient<F>(&self, objective: F) -> Vec<f64>
    where
        F: Fn(&QubitState) -> f64,
    {
        let initial_state = QubitState::zero_state(self.num_qubits);
        let mut gradients = vec![0.0; self.parameters.len()];

        for (i, _) in self.parameters.iter().enumerate() {
            // Forward shift: θ + π/2
            let mut params_plus = self.parameters.clone();
            params_plus[i] += PI / 2.0;

            let mut circuit_plus = self.clone();
            circuit_plus.set_parameters(&params_plus);
            let state_plus = circuit_plus.execute(&initial_state);
            let f_plus = objective(&state_plus);

            // Backward shift: θ - π/2
            let mut params_minus = self.parameters.clone();
            params_minus[i] -= PI / 2.0;

            let mut circuit_minus = self.clone();
            circuit_minus.set_parameters(&params_minus);
            let state_minus = circuit_minus.execute(&initial_state);
            let f_minus = objective(&state_minus);

            // Parameter-shift gradient
            gradients[i] = (f_plus - f_minus) / 2.0;
        }

        gradients
    }

    /// Compute gradient with epistemic variance tracking
    pub fn epistemic_gradient<F>(&self, objective: F) -> (Vec<f64>, f64)
    where
        F: Fn(&QubitState) -> f64,
    {
        let gradients = self.parameter_gradient(&objective);

        // Estimate variance from noise and parameter uncertainty
        let stats = self.stats();
        let noise_variance = stats.estimated_error;
        let param_variance: f64 = gradients.iter().map(|g| g * g).sum::<f64>() * 0.01;
        let total_variance = noise_variance + param_variance;

        (gradients, total_variance)
    }
}

// =============================================================================
// Circuit Builder (fluent API)
// =============================================================================

/// Fluent builder for quantum circuits
pub struct CircuitBuilder {
    circuit: QuantumCircuit,
    param_counter: usize,
}

impl CircuitBuilder {
    /// Start building a circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            circuit: QuantumCircuit::new(num_qubits),
            param_counter: 0,
        }
    }

    /// Set noise model
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.circuit.noise_model = Some(noise);
        self
    }

    /// Set circuit name
    pub fn named(mut self, name: &str) -> Self {
        self.circuit.metadata.name = Some(name.to_string());
        self
    }

    // =========================================================================
    // Single-qubit gates
    // =========================================================================

    pub fn h(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::h(qubit));
        self
    }

    pub fn x(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::x(qubit));
        self
    }

    pub fn y(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::y(qubit));
        self
    }

    pub fn z(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::z(qubit));
        self
    }

    pub fn s(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::s(qubit));
        self
    }

    pub fn t(mut self, qubit: usize) -> Self {
        self.circuit.add_gate(Gate::t(qubit));
        self
    }

    // =========================================================================
    // Parametric gates (fixed angle)
    // =========================================================================

    pub fn rx(mut self, qubit: usize, theta: f64) -> Self {
        self.circuit.add_gate(Gate::rx(qubit, theta));
        self
    }

    pub fn ry(mut self, qubit: usize, theta: f64) -> Self {
        self.circuit.add_gate(Gate::ry(qubit, theta));
        self
    }

    pub fn rz(mut self, qubit: usize, theta: f64) -> Self {
        self.circuit.add_gate(Gate::rz(qubit, theta));
        self
    }

    // =========================================================================
    // Parametric gates (variational - for VQE)
    // =========================================================================

    /// Add RX with variational parameter
    pub fn rx_var(mut self, qubit: usize, initial: f64) -> Self {
        let param_idx = self.param_counter;
        self.param_counter += 1;
        self.circuit
            .add_parametric_gate(Gate::rx(qubit, initial), vec![param_idx]);
        self
    }

    /// Add RY with variational parameter
    pub fn ry_var(mut self, qubit: usize, initial: f64) -> Self {
        let param_idx = self.param_counter;
        self.param_counter += 1;
        self.circuit
            .add_parametric_gate(Gate::ry(qubit, initial), vec![param_idx]);
        self
    }

    /// Add RZ with variational parameter
    pub fn rz_var(mut self, qubit: usize, initial: f64) -> Self {
        let param_idx = self.param_counter;
        self.param_counter += 1;
        self.circuit
            .add_parametric_gate(Gate::rz(qubit, initial), vec![param_idx]);
        self
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    pub fn cnot(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(Gate::cnot(control, target));
        self
    }

    pub fn cx(self, control: usize, target: usize) -> Self {
        self.cnot(control, target)
    }

    pub fn cz(mut self, control: usize, target: usize) -> Self {
        self.circuit.add_gate(Gate::cz(control, target));
        self
    }

    pub fn swap(mut self, q1: usize, q2: usize) -> Self {
        self.circuit.add_gate(Gate::swap(q1, q2));
        self
    }

    // =========================================================================
    // Three-qubit gates
    // =========================================================================

    pub fn toffoli(mut self, c1: usize, c2: usize, target: usize) -> Self {
        self.circuit.add_gate(Gate::toffoli(c1, c2, target));
        self
    }

    pub fn ccx(self, c1: usize, c2: usize, target: usize) -> Self {
        self.toffoli(c1, c2, target)
    }

    // =========================================================================
    // Common circuit patterns
    // =========================================================================

    /// Add a layer of Hadamards
    pub fn h_all(mut self) -> Self {
        for q in 0..self.circuit.num_qubits {
            self.circuit.add_gate(Gate::h(q));
        }
        self
    }

    /// Add a layer of CNOTs in linear entanglement
    pub fn entangle_linear(mut self) -> Self {
        for q in 0..self.circuit.num_qubits - 1 {
            self.circuit.add_gate(Gate::cnot(q, q + 1));
        }
        self
    }

    /// Add a layer of CNOTs in circular entanglement
    pub fn entangle_circular(mut self) -> Self {
        for q in 0..self.circuit.num_qubits {
            self.circuit
                .add_gate(Gate::cnot(q, (q + 1) % self.circuit.num_qubits));
        }
        self
    }

    /// Add a variational layer (RY-RZ on each qubit + linear CNOTs)
    pub fn variational_layer(mut self) -> Self {
        // Rotation layer
        for q in 0..self.circuit.num_qubits {
            let idx1 = self.param_counter;
            self.param_counter += 1;
            self.circuit
                .add_parametric_gate(Gate::ry(q, 0.0), vec![idx1]);

            let idx2 = self.param_counter;
            self.param_counter += 1;
            self.circuit
                .add_parametric_gate(Gate::rz(q, 0.0), vec![idx2]);
        }

        // Entanglement layer
        for q in 0..self.circuit.num_qubits - 1 {
            self.circuit.add_gate(Gate::cnot(q, q + 1));
        }

        self
    }

    /// Build the circuit
    pub fn build(self) -> QuantumCircuit {
        self.circuit
    }
}

// =============================================================================
// Common Circuit Templates
// =============================================================================

impl QuantumCircuit {
    /// Create a Bell state preparation circuit
    pub fn bell_state(qubit1: usize, qubit2: usize) -> Self {
        CircuitBuilder::new(qubit2.max(qubit1) + 1)
            .h(qubit1)
            .cnot(qubit1, qubit2)
            .build()
    }

    /// Create a GHZ state preparation circuit
    pub fn ghz_state(num_qubits: usize) -> Self {
        let mut builder = CircuitBuilder::new(num_qubits).h(0);
        for q in 1..num_qubits {
            builder = builder.cnot(0, q);
        }
        builder.build()
    }

    /// Create a Quantum Fourier Transform circuit
    pub fn qft(num_qubits: usize) -> Self {
        let mut circuit = QuantumCircuit::new(num_qubits);

        for j in 0..num_qubits {
            circuit.add_gate(Gate::h(j));

            for k in (j + 1)..num_qubits {
                let angle = PI / (1 << (k - j)) as f64;
                circuit.add_gate(Gate::crz(k, j, angle));
            }
        }

        // Reverse qubit order
        for j in 0..(num_qubits / 2) {
            circuit.add_gate(Gate::swap(j, num_qubits - 1 - j));
        }

        circuit
    }

    /// Create a hardware-efficient ansatz for VQE
    pub fn hardware_efficient_ansatz(num_qubits: usize, num_layers: usize) -> Self {
        let mut builder = CircuitBuilder::new(num_qubits);

        for _ in 0..num_layers {
            builder = builder.variational_layer();
        }

        builder.build()
    }

    /// Create UCCSD-style ansatz for molecular simulation
    pub fn uccsd_ansatz(num_qubits: usize, num_excitations: usize) -> Self {
        let mut builder = CircuitBuilder::new(num_qubits);

        // Initialize in HF state (alternating 0s and 1s for electrons)
        for q in 0..(num_qubits / 2) {
            builder = builder.x(q);
        }

        // Single excitations
        for i in 0..num_excitations.min(num_qubits / 2) {
            let occ = i;
            let virt = num_qubits / 2 + i;

            builder = builder
                .h(occ)
                .h(virt)
                .cnot(occ, virt)
                .rz_var(virt, 0.0) // θ parameter
                .cnot(occ, virt)
                .h(occ)
                .h(virt);
        }

        // Double excitations (simplified)
        if num_qubits >= 4 && num_excitations > 0 {
            // A single double excitation for demonstration
            builder = builder
                .cnot(0, 1)
                .cnot(1, 2)
                .cnot(2, 3)
                .rz_var(3, 0.0)
                .cnot(2, 3)
                .cnot(1, 2)
                .cnot(0, 1);
        }

        builder.build()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_builder_basic() {
        let circuit = CircuitBuilder::new(2).h(0).cnot(0, 1).build();

        assert_eq!(circuit.num_qubits, 2);
        assert_eq!(circuit.gates.len(), 2);
    }

    #[test]
    fn test_circuit_stats() {
        let circuit = CircuitBuilder::new(3)
            .h_all()
            .entangle_linear()
            .h_all()
            .build();

        let stats = circuit.stats();
        assert_eq!(stats.num_qubits, 3);
        assert_eq!(stats.gate_count, 8); // 3 + 2 + 3
        assert_eq!(stats.two_qubit_gate_count, 2);
    }

    #[test]
    fn test_circuit_execution() {
        let circuit = CircuitBuilder::new(1).h(0).build();

        let initial = QubitState::zero_state(1);
        let final_state = circuit.execute(&initial);

        // Should be |+> state
        assert!((final_state.probability(0) - 0.5).abs() < 1e-10);
        assert!((final_state.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let circuit = QuantumCircuit::bell_state(0, 1);
        let initial = QubitState::zero_state(2);
        let final_state = circuit.execute(&initial);

        // Bell state: |00> + |11> with equal probability
        assert!((final_state.probability(0b00) - 0.5).abs() < 1e-10);
        assert!((final_state.probability(0b11) - 0.5).abs() < 1e-10);
        assert!(final_state.probability(0b01) < 1e-10);
        assert!(final_state.probability(0b10) < 1e-10);
    }

    #[test]
    fn test_variational_circuit() {
        let circuit = CircuitBuilder::new(2)
            .ry_var(0, 0.5)
            .ry_var(1, 0.5)
            .cnot(0, 1)
            .build();

        assert_eq!(circuit.parameters.len(), 2);
        assert_eq!(circuit.parametric_gates.len(), 2);
    }

    #[test]
    fn test_parameter_gradient() {
        // Use θ=π/4 where gradient is non-zero
        // P(|1>) = sin²(θ/2), so ∂P/∂θ = sin(θ/2)cos(θ/2) = sin(θ)/2
        // At θ=π/4: gradient ≈ sin(π/4)/2 ≈ 0.354
        let circuit = CircuitBuilder::new(1)
            .ry_var(0, std::f64::consts::PI / 4.0)
            .build();

        // Objective: probability of |1>
        let objective = |state: &QubitState| state.probability(1);

        let gradients = circuit.parameter_gradient(objective);

        assert_eq!(gradients.len(), 1);
        // At θ=π/4, gradient should be approximately 0.354
        assert!(
            gradients[0].abs() > 0.1,
            "Gradient should be non-zero at θ=π/4"
        );
    }

    #[test]
    fn test_ghz_state() {
        let circuit = QuantumCircuit::ghz_state(3);
        let initial = QubitState::zero_state(3);
        let final_state = circuit.execute(&initial);

        // GHZ: |000> + |111>
        assert!((final_state.probability(0b000) - 0.5).abs() < 1e-10);
        assert!((final_state.probability(0b111) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_hardware_efficient_ansatz() {
        let circuit = QuantumCircuit::hardware_efficient_ansatz(4, 2);
        let stats = circuit.stats();

        // 2 layers × (4 RY + 4 RZ + 3 CNOTs) = 22 gates
        assert_eq!(stats.parameter_count, 16); // 4 × 2 × 2
        assert!(stats.two_qubit_gate_count >= 6); // At least 3 per layer
    }
}
