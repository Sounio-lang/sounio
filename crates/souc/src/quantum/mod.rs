//! Quantum Machine Learning Module for Sounio
//!
//! Native integration of quantum computing with epistemic semantics:
//! - Epistemic quantum states (Knowledge<QubitState> with noise-aware variance)
//! - Differentiable quantum circuits with gradient tracking
//! - VQE/QAOA with full posterior energy estimation
//! - GPU kernels for parallel quantum trials
//! - Refinement types for unitarity and no-cloning
//!
//! # Key Innovation
//!
//! Every quantum measurement is `Knowledge<T>` - noise and decoherence
//! automatically propagate as epistemic variance. This enables:
//! - "How confident am I in this quantum advantage?"
//! - Variance penalty in VQE to encourage stable circuits
//! - Provenance tracking for quantum chemistry audits
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Epistemic Quantum ML                         │
//! ├─────────────────────────────────────────────────────────────────┤
//! │                                                                 │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
//! │  │ Qubit State │───►│  Circuit    │───►│ Measurement │         │
//! │  │ Knowledge   │    │ Execution   │    │ Knowledge   │         │
//! │  │ (amp+noise) │    │ (gates)     │    │ (value+var) │         │
//! │  └─────────────┘    └─────────────┘    └─────────────┘         │
//! │         │                  │                  │                 │
//! │         ▼                  ▼                  ▼                 │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │           Epistemic Variance Propagation            │       │
//! │  │   (noise model + gate errors + measurement shots)   │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! │                            │                                   │
//! │                            ▼                                   │
//! │  ┌─────────────────────────────────────────────────────┐       │
//! │  │              VQE / QAOA Optimization                │       │
//! │  │   Loss = Energy + λ * Variance (epistemic penalty)  │       │
//! │  └─────────────────────────────────────────────────────┘       │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod amplitude;
pub mod circuit;
pub mod epistemic_vqe;
pub mod gates;
pub mod gpu_quantum;
pub mod hamiltonian;
pub mod noise;
pub mod pennylane;
pub mod qir_shim;
pub mod states;
pub mod uccsd;
pub mod vqe;

pub use amplitude::EpistemicAmplitude;
pub use circuit::{CircuitBuilder, CircuitStats, QuantumCircuit};
pub use gates::{Gate, GateType, ParametricGate};
pub use hamiltonian::{
    Hamiltonian as MolecularHamiltonian, Pauli, PauliString, VQEResult as MolecularVQEResult,
};
pub use noise::{AmplitudeDamping, DepolarizingNoise, NoiseModel, NoiseType};
pub use pennylane::{
    EpistemicExpectation, EpistemicGradients, EpistemicHamiltonian, EpistemicParam, LayerType,
    PauliObservable, PauliType, TrainingConfig, VQEOptimizationResult, VariationalCircuit,
    VariationalLayer,
};
pub use qir_shim::{
    QuantumConformanceTelemetry, QuantumConformanceThresholds, QirShimArtifact,
    emit_qir_shim, evaluate_quantum_conformance, interval_coverage_95, ks_two_sample_pvalue,
};
pub use states::{DensityMatrix, EpistemicQubit, QubitState, StateVector};
pub use uccsd::{
    DoubleExcitation, FermionOp, FermionString, MolecularSystem, QubitMapping, SingleExcitation,
    UCCSDCircuit, UCCSDResult,
};
pub use vqe::{Hamiltonian, PauliTerm, VQEConfig, VQEResult, VQESolver};

// Revolutionary epistemic VQE with full Beta posteriors
pub use epistemic_vqe::{
    ActiveInferenceSummary, BetaQuantumParameter, EpistemicEnergy, EpistemicVQE,
    EpistemicVQEConfig, EpistemicVQEResult, VarianceBreakdown,
};

// GPU-accelerated quantum simulation
pub use gpu_quantum::{
    BatchedVQE, GpuBackend, GpuCircuit, GpuMemoryPool, GpuQuantumConfig, GpuStateVector,
    SingleQubitKernel, TwoQubitGateType, TwoQubitKernel,
};

// ============================================================================
// High-level QML primitives (callable from interpreter builtins)
// ============================================================================

/// Compute the tensor product of two quantum states.
///
/// For pure states |ψ₁⟩ and |ψ₂⟩ this produces |ψ₁⟩ ⊗ |ψ₂⟩, whose amplitude
/// vector has dimension dim(ψ₁) × dim(ψ₂).
pub fn tensor_product(a: &StateVector, b: &StateVector) -> StateVector {
    let dim_a = a.amplitudes.len();
    let dim_b = b.amplitudes.len();
    let mut amplitudes = Vec::with_capacity(dim_a * dim_b);
    for ai in &a.amplitudes {
        for bi in &b.amplitudes {
            amplitudes.push(*ai * *bi);
        }
    }
    StateVector {
        amplitudes,
        num_qubits: a.num_qubits + b.num_qubits,
    }
}

/// Create a strongly-entangling QNN layer descriptor.
///
/// Returns a `VariationalLayer` configured with `num_qubits` qubits at
/// `layer_index`, ready for insertion into a `VariationalCircuit`.
pub fn qnn_layer(num_qubits: usize, layer_index: usize) -> VariationalLayer {
    VariationalLayer::strongly_entangling(num_qubits, layer_index)
}
