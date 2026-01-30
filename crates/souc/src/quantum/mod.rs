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
