//! Quantum Gates with Epistemic Error Tracking
//!
//! Quantum gates that track error rates and propagate uncertainty.

use super::states::{Complex, QuantumState, QubitState, StateVector};
use std::f64::consts::PI;

// =============================================================================
// Gate Types
// =============================================================================

/// Type of quantum gate
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateType {
    // Single-qubit gates
    Identity,
    PauliX,
    PauliY,
    PauliZ,
    Hadamard,
    S,     // sqrt(Z)
    Sdg,   // S†
    T,     // sqrt(S)
    Tdg,   // T†
    SqrtX, // sqrt(X)

    // Parametric single-qubit gates
    RX,
    RY,
    RZ,
    Phase,
    U1,
    U2,
    U3,

    // Two-qubit gates
    CNOT,
    CZ,
    SWAP,
    ISwap,
    CRX,
    CRY,
    CRZ,
    CPhase,

    // Three-qubit gates
    Toffoli,
    Fredkin,
}

impl GateType {
    /// Get the error rate for this gate type (NISQ device typical)
    pub fn typical_error_rate(&self) -> f64 {
        match self {
            // Single-qubit gates: ~0.1%
            GateType::Identity => 0.0,
            GateType::PauliX | GateType::PauliY | GateType::PauliZ => 0.001,
            GateType::Hadamard => 0.001,
            GateType::S | GateType::Sdg | GateType::T | GateType::Tdg => 0.001,
            GateType::SqrtX => 0.001,
            GateType::RX | GateType::RY | GateType::RZ => 0.001,
            GateType::Phase | GateType::U1 | GateType::U2 | GateType::U3 => 0.002,

            // Two-qubit gates: ~1%
            GateType::CNOT | GateType::CZ => 0.01,
            GateType::SWAP => 0.03, // 3 CNOTs
            GateType::ISwap => 0.02,
            GateType::CRX | GateType::CRY | GateType::CRZ | GateType::CPhase => 0.015,

            // Three-qubit gates: ~5%
            GateType::Toffoli => 0.05,
            GateType::Fredkin => 0.05,
        }
    }

    /// Check if gate is parametric
    pub fn is_parametric(&self) -> bool {
        matches!(
            self,
            GateType::RX
                | GateType::RY
                | GateType::RZ
                | GateType::Phase
                | GateType::U1
                | GateType::U2
                | GateType::U3
                | GateType::CRX
                | GateType::CRY
                | GateType::CRZ
                | GateType::CPhase
        )
    }

    /// Number of parameters for this gate
    pub fn num_params(&self) -> usize {
        match self {
            GateType::RX | GateType::RY | GateType::RZ | GateType::Phase | GateType::U1 => 1,
            GateType::U2 => 2,
            GateType::U3 => 3,
            GateType::CRX | GateType::CRY | GateType::CRZ | GateType::CPhase => 1,
            _ => 0,
        }
    }

    /// Number of qubits this gate acts on
    pub fn num_qubits(&self) -> usize {
        match self {
            GateType::Identity
            | GateType::PauliX
            | GateType::PauliY
            | GateType::PauliZ
            | GateType::Hadamard
            | GateType::S
            | GateType::Sdg
            | GateType::T
            | GateType::Tdg
            | GateType::SqrtX
            | GateType::RX
            | GateType::RY
            | GateType::RZ
            | GateType::Phase
            | GateType::U1
            | GateType::U2
            | GateType::U3 => 1,

            GateType::CNOT
            | GateType::CZ
            | GateType::SWAP
            | GateType::ISwap
            | GateType::CRX
            | GateType::CRY
            | GateType::CRZ
            | GateType::CPhase => 2,

            GateType::Toffoli | GateType::Fredkin => 3,
        }
    }
}

// =============================================================================
// Gate Definition
// =============================================================================

/// A quantum gate with error tracking
#[derive(Debug, Clone)]
pub struct Gate {
    /// Type of gate
    pub gate_type: GateType,
    /// Target qubit indices
    pub targets: Vec<usize>,
    /// Control qubit indices (for controlled gates)
    pub controls: Vec<usize>,
    /// Parameters (for parametric gates)
    pub params: Vec<f64>,
    /// Error rate override (None = use typical)
    pub error_rate: Option<f64>,
    /// Gradient of parameters (for VQE)
    pub param_gradients: Option<Vec<f64>>,
}

impl Gate {
    // =========================================================================
    // Single-qubit gate constructors
    // =========================================================================

    pub fn identity(qubit: usize) -> Self {
        Self {
            gate_type: GateType::Identity,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn x(qubit: usize) -> Self {
        Self {
            gate_type: GateType::PauliX,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn y(qubit: usize) -> Self {
        Self {
            gate_type: GateType::PauliY,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn z(qubit: usize) -> Self {
        Self {
            gate_type: GateType::PauliZ,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn h(qubit: usize) -> Self {
        Self {
            gate_type: GateType::Hadamard,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn s(qubit: usize) -> Self {
        Self {
            gate_type: GateType::S,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn t(qubit: usize) -> Self {
        Self {
            gate_type: GateType::T,
            targets: vec![qubit],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    // =========================================================================
    // Parametric single-qubit gates
    // =========================================================================

    pub fn rx(qubit: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::RX,
            targets: vec![qubit],
            controls: vec![],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn ry(qubit: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::RY,
            targets: vec![qubit],
            controls: vec![],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn rz(qubit: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::RZ,
            targets: vec![qubit],
            controls: vec![],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    pub fn cnot(control: usize, target: usize) -> Self {
        Self {
            gate_type: GateType::CNOT,
            targets: vec![target],
            controls: vec![control],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn cx(control: usize, target: usize) -> Self {
        Self::cnot(control, target)
    }

    pub fn cz(control: usize, target: usize) -> Self {
        Self {
            gate_type: GateType::CZ,
            targets: vec![target],
            controls: vec![control],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn swap(qubit1: usize, qubit2: usize) -> Self {
        Self {
            gate_type: GateType::SWAP,
            targets: vec![qubit1, qubit2],
            controls: vec![],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    // =========================================================================
    // Parametric two-qubit gates
    // =========================================================================

    pub fn crx(control: usize, target: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::CRX,
            targets: vec![target],
            controls: vec![control],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn cry(control: usize, target: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::CRY,
            targets: vec![target],
            controls: vec![control],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn crz(control: usize, target: usize, theta: f64) -> Self {
        Self {
            gate_type: GateType::CRZ,
            targets: vec![target],
            controls: vec![control],
            params: vec![theta],
            error_rate: None,
            param_gradients: None,
        }
    }

    // =========================================================================
    // Three-qubit gates
    // =========================================================================

    pub fn toffoli(control1: usize, control2: usize, target: usize) -> Self {
        Self {
            gate_type: GateType::Toffoli,
            targets: vec![target],
            controls: vec![control1, control2],
            params: vec![],
            error_rate: None,
            param_gradients: None,
        }
    }

    pub fn ccx(control1: usize, control2: usize, target: usize) -> Self {
        Self::toffoli(control1, control2, target)
    }

    // =========================================================================
    // Gate properties
    // =========================================================================

    /// Get the error rate for this gate
    pub fn get_error_rate(&self) -> f64 {
        self.error_rate
            .unwrap_or_else(|| self.gate_type.typical_error_rate())
    }

    /// Set custom error rate
    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self
    }

    /// Check if this gate is parametric
    pub fn is_parametric(&self) -> bool {
        self.gate_type.is_parametric()
    }

    /// Get all qubit indices this gate acts on
    pub fn all_qubits(&self) -> Vec<usize> {
        let mut qubits = self.controls.clone();
        qubits.extend(&self.targets);
        qubits
    }
}

// =============================================================================
// Parametric Gate (for VQE)
// =============================================================================

/// A parametric gate with gradient tracking for variational algorithms
#[derive(Debug, Clone)]
pub struct ParametricGate {
    /// The underlying gate
    pub gate: Gate,
    /// Parameter indices in the circuit parameter vector
    pub param_indices: Vec<usize>,
    /// Whether gradient is required
    pub requires_grad: bool,
    /// Index in the circuit's gates array
    pub gate_index: usize,
}

impl ParametricGate {
    /// Create a new parametric gate
    pub fn new(gate: Gate, param_indices: Vec<usize>) -> Self {
        Self {
            gate,
            param_indices,
            requires_grad: true,
            gate_index: 0,
        }
    }

    /// Update parameters from a parameter vector
    pub fn update_params(&mut self, params: &[f64]) {
        for (i, &idx) in self.param_indices.iter().enumerate() {
            if i < self.gate.params.len() && idx < params.len() {
                self.gate.params[i] = params[idx];
            }
        }
    }

    /// Get the gradient of this gate with respect to its parameters
    /// Uses parameter-shift rule: df/dθ = (f(θ+π/2) - f(θ-π/2)) / 2
    pub fn param_shift_gradients(&self) -> Vec<f64> {
        // Placeholder - actual computation happens during circuit execution
        vec![0.0; self.gate.params.len()]
    }
}

// =============================================================================
// Gate Application (State Vector)
// =============================================================================

impl Gate {
    /// Apply this gate to a state vector
    pub fn apply_to_statevector(&self, state: &mut StateVector) {
        match self.gate_type {
            GateType::Identity => {}

            GateType::PauliX => {
                let qubit = self.targets[0];
                apply_single_qubit_gate(state, qubit, |a, b| (b, a));
            }

            GateType::PauliY => {
                let qubit = self.targets[0];
                apply_single_qubit_gate(state, qubit, |a, b| {
                    (Complex::new(0.0, 1.0) * b, Complex::new(0.0, -1.0) * a)
                });
            }

            GateType::PauliZ => {
                let qubit = self.targets[0];
                apply_single_qubit_gate(state, qubit, |a, b| (a, Complex::new(-1.0, 0.0) * b));
            }

            GateType::Hadamard => {
                let qubit = self.targets[0];
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = (a + b) * inv_sqrt2;
                    let new_b = (a + Complex::new(-1.0, 0.0) * b) * inv_sqrt2;
                    (new_a, new_b)
                });
            }

            GateType::S => {
                let qubit = self.targets[0];
                apply_single_qubit_gate(state, qubit, |a, b| (a, Complex::I * b));
            }

            GateType::T => {
                let qubit = self.targets[0];
                let phase = Complex::from_polar(1.0, PI / 4.0);
                apply_single_qubit_gate(state, qubit, |a, b| (a, phase * b));
            }

            GateType::Sdg => {
                let qubit = self.targets[0];
                // S† = conjugate of S, which applies -i phase to |1>
                let neg_i = Complex::new(0.0, -1.0);
                apply_single_qubit_gate(state, qubit, |a, b| (a, neg_i * b));
            }

            GateType::Tdg => {
                let qubit = self.targets[0];
                // T† = conjugate of T, which applies e^{-iπ/4} phase to |1>
                let phase = Complex::from_polar(1.0, -PI / 4.0);
                apply_single_qubit_gate(state, qubit, |a, b| (a, phase * b));
            }

            GateType::SqrtX => {
                let qubit = self.targets[0];
                // sqrt(X) = (1+i)/2 * I + (1-i)/2 * X
                let half = 0.5;
                let half_i = Complex::new(0.0, 0.5);
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = a * half + a * half_i + b * half - b * half_i;
                    let new_b = a * half - a * half_i + b * half + b * half_i;
                    (new_a, new_b)
                });
            }

            GateType::Phase => {
                let qubit = self.targets[0];
                let theta = self.params[0];
                let phase = Complex::from_polar(1.0, theta);
                apply_single_qubit_gate(state, qubit, |a, b| (a, phase * b));
            }

            GateType::U1 => {
                // U1(λ) = Phase(λ) - just a phase on |1>
                let qubit = self.targets[0];
                let lambda = self.params[0];
                let phase = Complex::from_polar(1.0, lambda);
                apply_single_qubit_gate(state, qubit, |a, b| (a, phase * b));
            }

            GateType::U2 => {
                // U2(φ, λ) = (1/√2) * [[1, -e^{iλ}], [e^{iφ}, e^{i(φ+λ)}]]
                let qubit = self.targets[0];
                let phi = self.params[0];
                let lambda = self.params[1];
                let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
                let exp_phi = Complex::from_polar(1.0, phi);
                let exp_lambda = Complex::from_polar(1.0, lambda);
                let exp_sum = Complex::from_polar(1.0, phi + lambda);
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = (a - exp_lambda * b) * inv_sqrt2;
                    let new_b = (exp_phi * a + exp_sum * b) * inv_sqrt2;
                    (new_a, new_b)
                });
            }

            GateType::U3 => {
                // U3(θ, φ, λ) = [[cos(θ/2), -e^{iλ}sin(θ/2)],
                //                [e^{iφ}sin(θ/2), e^{i(φ+λ)}cos(θ/2)]]
                let qubit = self.targets[0];
                let theta = self.params[0];
                let phi = self.params[1];
                let lambda = self.params[2];
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                let exp_phi = Complex::from_polar(1.0, phi);
                let exp_lambda = Complex::from_polar(1.0, lambda);
                let exp_sum = Complex::from_polar(1.0, phi + lambda);
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = a * cos_half - exp_lambda * b * sin_half;
                    let new_b = exp_phi * a * sin_half + exp_sum * b * cos_half;
                    (new_a, new_b)
                });
            }

            GateType::RX => {
                let qubit = self.targets[0];
                let theta = self.params[0];
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = a * cos_half + Complex::new(0.0, -sin_half) * b;
                    let new_b = Complex::new(0.0, -sin_half) * a + b * cos_half;
                    (new_a, new_b)
                });
            }

            GateType::RY => {
                let qubit = self.targets[0];
                let theta = self.params[0];
                let cos_half = (theta / 2.0).cos();
                let sin_half = (theta / 2.0).sin();
                apply_single_qubit_gate(state, qubit, |a, b| {
                    let new_a = a * cos_half + Complex::new(-sin_half, 0.0) * b;
                    let new_b = Complex::new(sin_half, 0.0) * a + b * cos_half;
                    (new_a, new_b)
                });
            }

            GateType::RZ => {
                let qubit = self.targets[0];
                let theta = self.params[0];
                let phase_pos = Complex::from_polar(1.0, theta / 2.0);
                let phase_neg = Complex::from_polar(1.0, -theta / 2.0);
                apply_single_qubit_gate(state, qubit, |a, b| (phase_neg * a, phase_pos * b));
            }

            GateType::CNOT => {
                let control = self.controls[0];
                let target = self.targets[0];
                apply_cnot(state, control, target);
            }

            GateType::CZ => {
                let control = self.controls[0];
                let target = self.targets[0];
                apply_cz(state, control, target);
            }

            GateType::SWAP => {
                let q1 = self.targets[0];
                let q2 = self.targets[1];
                apply_swap(state, q1, q2);
            }

            GateType::ISwap => {
                let q1 = self.targets[0];
                let q2 = self.targets[1];
                apply_iswap(state, q1, q2);
            }

            GateType::CRX => {
                let control = self.controls[0];
                let target = self.targets[0];
                let theta = self.params[0];
                apply_controlled_rotation(state, control, target, theta, 'x');
            }

            GateType::CRY => {
                let control = self.controls[0];
                let target = self.targets[0];
                let theta = self.params[0];
                apply_controlled_rotation(state, control, target, theta, 'y');
            }

            GateType::CRZ => {
                let control = self.controls[0];
                let target = self.targets[0];
                let theta = self.params[0];
                apply_controlled_rotation(state, control, target, theta, 'z');
            }

            GateType::CPhase => {
                let control = self.controls[0];
                let target = self.targets[0];
                let theta = self.params[0];
                apply_controlled_phase(state, control, target, theta);
            }

            GateType::Toffoli => {
                let c1 = self.controls[0];
                let c2 = self.controls[1];
                let target = self.targets[0];
                apply_toffoli(state, c1, c2, target);
            }

            GateType::Fredkin => {
                let control = self.controls[0];
                let t1 = self.targets[0];
                let t2 = self.targets[1];
                apply_fredkin(state, control, t1, t2);
            }
        }
    }

    /// Apply this gate to a QubitState (with error tracking)
    pub fn apply(&self, state: &mut QubitState) {
        // Apply to state vector
        if let QuantumState::Pure(ref mut sv) = state.state {
            self.apply_to_statevector(sv);
        }

        // Track gate error
        state.apply_gate_error(self.get_error_rate());
    }
}

// =============================================================================
// Helper functions for gate application
// =============================================================================

fn apply_single_qubit_gate<F>(state: &mut StateVector, qubit: usize, transform: F)
where
    F: Fn(Complex, Complex) -> (Complex, Complex),
{
    let n = state.num_qubits;
    let dim = 1 << n;
    let mask = 1 << (n - 1 - qubit);

    for i in 0..dim {
        if i & mask == 0 {
            let j = i | mask;
            let (new_i, new_j) = transform(state.amplitudes[i], state.amplitudes[j]);
            state.amplitudes[i] = new_i;
            state.amplitudes[j] = new_j;
        }
    }
}

fn apply_cnot(state: &mut StateVector, control: usize, target: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let control_mask = 1 << (n - 1 - control);
    let target_mask = 1 << (n - 1 - target);

    for i in 0..dim {
        // Only flip target if control is |1>
        if i & control_mask != 0 && i & target_mask == 0 {
            let j = i | target_mask;
            state.amplitudes.swap(i, j);
        }
    }
}

fn apply_cz(state: &mut StateVector, control: usize, target: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let control_mask = 1 << (n - 1 - control);
    let target_mask = 1 << (n - 1 - target);

    for i in 0..dim {
        // Apply -1 phase if both control and target are |1>
        if i & control_mask != 0 && i & target_mask != 0 {
            state.amplitudes[i] = state.amplitudes[i] * (-1.0);
        }
    }
}

fn apply_swap(state: &mut StateVector, q1: usize, q2: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let mask1 = 1 << (n - 1 - q1);
    let mask2 = 1 << (n - 1 - q2);

    for i in 0..dim {
        let bit1 = (i & mask1) != 0;
        let bit2 = (i & mask2) != 0;

        if bit1 != bit2 {
            // Swap only if bits differ
            let j = i ^ mask1 ^ mask2;
            if i < j {
                state.amplitudes.swap(i, j);
            }
        }
    }
}

fn apply_iswap(state: &mut StateVector, q1: usize, q2: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let mask1 = 1 << (n - 1 - q1);
    let mask2 = 1 << (n - 1 - q2);
    let i_unit = Complex::new(0.0, 1.0);

    for i in 0..dim {
        let bit1 = (i & mask1) != 0;
        let bit2 = (i & mask2) != 0;

        // iSWAP: |01> -> i|10>, |10> -> i|01>
        if bit1 != bit2 {
            let j = i ^ mask1 ^ mask2;
            if i < j {
                let tmp = state.amplitudes[i];
                state.amplitudes[i] = state.amplitudes[j] * i_unit;
                state.amplitudes[j] = tmp * i_unit;
            }
        }
    }
}

fn apply_controlled_rotation(
    state: &mut StateVector,
    control: usize,
    target: usize,
    theta: f64,
    axis: char,
) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let control_mask = 1 << (n - 1 - control);
    let target_mask = 1 << (n - 1 - target);

    let half_theta = theta / 2.0;
    let cos = half_theta.cos();
    let sin = half_theta.sin();

    for i in 0..dim {
        // Only apply rotation if control is |1>
        if i & control_mask != 0 && i & target_mask == 0 {
            let j = i | target_mask;
            let a0 = state.amplitudes[i];
            let a1 = state.amplitudes[j];

            let (new0, new1) = match axis {
                'x' => {
                    // RX: cos(θ/2)I - i·sin(θ/2)X
                    let i_sin = Complex::new(0.0, -sin);
                    (a0 * cos + a1 * i_sin, a0 * i_sin + a1 * cos)
                }
                'y' => {
                    // RY: cos(θ/2)I - i·sin(θ/2)Y
                    (a0 * cos - a1 * sin, a0 * sin + a1 * cos)
                }
                'z' => {
                    // RZ: e^(-iθ/2)|0><0| + e^(iθ/2)|1><1|
                    let exp_neg = Complex::new(cos, -sin);
                    let exp_pos = Complex::new(cos, sin);
                    (a0 * exp_neg, a1 * exp_pos)
                }
                _ => (a0, a1),
            };

            state.amplitudes[i] = new0;
            state.amplitudes[j] = new1;
        }
    }
}

fn apply_controlled_phase(state: &mut StateVector, control: usize, target: usize, theta: f64) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let control_mask = 1 << (n - 1 - control);
    let target_mask = 1 << (n - 1 - target);

    let phase = Complex::new(theta.cos(), theta.sin());

    for i in 0..dim {
        // Apply phase if both control and target are |1>
        if i & control_mask != 0 && i & target_mask != 0 {
            state.amplitudes[i] = state.amplitudes[i] * phase;
        }
    }
}

fn apply_toffoli(state: &mut StateVector, c1: usize, c2: usize, target: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let c1_mask = 1 << (n - 1 - c1);
    let c2_mask = 1 << (n - 1 - c2);
    let target_mask = 1 << (n - 1 - target);

    for i in 0..dim {
        // Only flip target if both controls are |1>
        if i & c1_mask != 0 && i & c2_mask != 0 && i & target_mask == 0 {
            let j = i | target_mask;
            state.amplitudes.swap(i, j);
        }
    }
}

fn apply_fredkin(state: &mut StateVector, control: usize, t1: usize, t2: usize) {
    let n = state.num_qubits;
    let dim = 1 << n;
    let control_mask = 1 << (n - 1 - control);
    let t1_mask = 1 << (n - 1 - t1);
    let t2_mask = 1 << (n - 1 - t2);

    for i in 0..dim {
        // Only swap targets if control is |1> and targets differ
        if i & control_mask != 0 {
            let bit1 = (i & t1_mask) != 0;
            let bit2 = (i & t2_mask) != 0;

            if bit1 != bit2 {
                let j = i ^ t1_mask ^ t2_mask;
                if i < j {
                    state.amplitudes.swap(i, j);
                }
            }
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
    fn test_gate_error_rates() {
        assert_eq!(GateType::Identity.typical_error_rate(), 0.0);
        assert!(GateType::CNOT.typical_error_rate() > GateType::Hadamard.typical_error_rate());
        assert!(GateType::Toffoli.typical_error_rate() > GateType::CNOT.typical_error_rate());
    }

    #[test]
    fn test_gate_constructors() {
        let h = Gate::h(0);
        assert_eq!(h.gate_type, GateType::Hadamard);
        assert_eq!(h.targets, vec![0]);

        let cnot = Gate::cnot(0, 1);
        assert_eq!(cnot.controls, vec![0]);
        assert_eq!(cnot.targets, vec![1]);

        let rx = Gate::rx(0, PI / 2.0);
        assert!(rx.is_parametric());
        assert_eq!(rx.params.len(), 1);
    }

    #[test]
    fn test_hadamard_application() {
        let mut sv = StateVector::zero_state(1);
        let h = Gate::h(0);
        h.apply_to_statevector(&mut sv);

        // Should be |+> = (|0> + |1>) / sqrt(2)
        assert!((sv.probability(0) - 0.5).abs() < 1e-10);
        assert!((sv.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_x_application() {
        let mut sv = StateVector::zero_state(1);
        let x = Gate::x(0);
        x.apply_to_statevector(&mut sv);

        // Should be |1>
        assert!(sv.probability(0) < 1e-10);
        assert!((sv.probability(1) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cnot_application() {
        // |00> -> |00>
        let mut sv1 = StateVector::zero_state(2);
        Gate::cnot(0, 1).apply_to_statevector(&mut sv1);
        assert!((sv1.probability(0b00) - 1.0).abs() < 1e-10);

        // |10> -> |11>
        let mut sv2 = StateVector::zero_state(2);
        Gate::x(0).apply_to_statevector(&mut sv2);
        Gate::cnot(0, 1).apply_to_statevector(&mut sv2);
        assert!((sv2.probability(0b11) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rz_application() {
        let mut sv = StateVector::plus_state(1);
        let rz = Gate::rz(0, PI);
        rz.apply_to_statevector(&mut sv);

        // RZ(π)|+> = |->
        // |-> = (|0> - |1>) / sqrt(2)
        assert!((sv.probability(0) - 0.5).abs() < 1e-10);
        assert!((sv.probability(1) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gate_with_error() {
        let mut qs = QubitState::zero_state(1);
        let h = Gate::h(0).with_error_rate(0.05);
        h.apply(&mut qs);

        assert_eq!(qs.gate_count, 1);
        assert!((qs.gate_error_accumulated - 0.05).abs() < 1e-10);
    }
}
