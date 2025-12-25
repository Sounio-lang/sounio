//! GPU Kernels for Parallel Quantum Simulation
//!
//! High-performance GPU acceleration for quantum circuit simulation:
//! - Parallel state vector evolution
//! - Batched VQE trials for variance reduction
//! - Parameter sweep optimization
//! - Monte Carlo shot simulation
//!
//! # Architecture
//!
//! ```text
//! ┌────────────────────────────────────────────────────────────────┐
//! │                    GPU QUANTUM SIMULATION                      │
//! ├────────────────────────────────────────────────────────────────┤
//! │                                                                │
//! │  Host (CPU)                    Device (GPU)                    │
//! │  ┌─────────┐                   ┌─────────────────────────┐     │
//! │  │ Circuit │──────────────────►│ State Vector Memory     │     │
//! │  │ Builder │                   │ (2^n complex amplitudes)│     │
//! │  └─────────┘                   └─────────────────────────┘     │
//! │       │                                    │                   │
//! │       ▼                                    ▼                   │
//! │  ┌─────────┐                   ┌─────────────────────────┐     │
//! │  │ Batch   │──────────────────►│ Parallel Gate Kernels  │     │
//! │  │ Config  │                   │ - Single-qubit: O(2^n) │     │
//! │  └─────────┘                   │ - Two-qubit: O(2^n)    │     │
//! │                                │ - Measurement: O(2^n)  │     │
//! │                                └─────────────────────────┘     │
//! │                                            │                   │
//! │  ┌─────────┐                              ▼                   │
//! │  │ Results │◄──────────────────┌─────────────────────────┐     │
//! │  │ Collect │                   │ Expectation Reduction   │     │
//! │  └─────────┘                   └─────────────────────────┘     │
//! └────────────────────────────────────────────────────────────────┘
//! ```

use std::f64::consts::PI;

use super::epistemic_vqe::EpistemicEnergy;
use super::noise::NoiseModel;
use super::states::{Complex, StateVector};
use super::vqe::Hamiltonian;

// =============================================================================
// GPU Configuration
// =============================================================================

/// GPU backend selection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuBackend {
    /// CUDA (NVIDIA)
    Cuda,
    /// Metal (Apple)
    Metal,
    /// WebGPU (cross-platform)
    WebGpu,
    /// CPU fallback (for testing)
    CpuSimulated,
}

/// Configuration for GPU quantum simulation
#[derive(Debug, Clone)]
pub struct GpuQuantumConfig {
    /// Backend to use
    pub backend: GpuBackend,
    /// Number of parallel circuits (batch size)
    pub batch_size: usize,
    /// Thread block size
    pub block_size: usize,
    /// Enable mixed precision (FP16 for some operations)
    pub mixed_precision: bool,
    /// Memory limit in bytes
    pub memory_limit: usize,
    /// Number of shots per circuit
    pub shots_per_circuit: usize,
}

impl Default for GpuQuantumConfig {
    fn default() -> Self {
        Self {
            backend: GpuBackend::CpuSimulated,
            batch_size: 32,
            block_size: 256,
            mixed_precision: false,
            memory_limit: 4 * 1024 * 1024 * 1024, // 4 GB
            shots_per_circuit: 1024,
        }
    }
}

impl GpuQuantumConfig {
    /// Configure for maximum performance
    pub fn high_performance() -> Self {
        Self {
            backend: GpuBackend::Cuda,
            batch_size: 128,
            block_size: 512,
            mixed_precision: true,
            memory_limit: 16 * 1024 * 1024 * 1024, // 16 GB
            shots_per_circuit: 8192,
        }
    }

    /// Configure for memory-constrained environments
    pub fn low_memory() -> Self {
        Self {
            backend: GpuBackend::CpuSimulated,
            batch_size: 8,
            block_size: 128,
            mixed_precision: true,
            memory_limit: 1024 * 1024 * 1024, // 1 GB
            shots_per_circuit: 256,
        }
    }

    /// Maximum qubits given memory limit
    pub fn max_qubits(&self) -> usize {
        // Each amplitude is 16 bytes (2 x f64)
        // State vector size = 2^n * 16 * batch_size
        let bytes_per_amplitude = 16;
        let available = self.memory_limit / (bytes_per_amplitude * self.batch_size);
        (available as f64).log2().floor() as usize
    }
}

// =============================================================================
// GPU State Vector
// =============================================================================

/// State vector stored on GPU (simulated in CPU for this implementation)
#[derive(Debug, Clone)]
pub struct GpuStateVector {
    /// Amplitudes (would be on device in real GPU implementation)
    pub amplitudes: Vec<Complex>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Which GPU it's on (for multi-GPU)
    pub device_id: usize,
}

impl GpuStateVector {
    /// Allocate zero state on GPU
    pub fn zero_state(num_qubits: usize, device_id: usize) -> Self {
        let dim = 1 << num_qubits;
        let mut amplitudes = vec![Complex::ZERO; dim];
        amplitudes[0] = Complex::ONE;

        Self {
            amplitudes,
            num_qubits,
            device_id,
        }
    }

    /// Create from CPU state vector
    pub fn from_cpu(sv: &StateVector, device_id: usize) -> Self {
        Self {
            amplitudes: sv.amplitudes.clone(),
            num_qubits: sv.num_qubits,
            device_id,
        }
    }

    /// Copy back to CPU
    pub fn to_cpu(&self) -> StateVector {
        StateVector {
            amplitudes: self.amplitudes.clone(),
            num_qubits: self.num_qubits,
        }
    }

    /// Get probabilities
    pub fn probabilities(&self) -> Vec<f64> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }
}

// =============================================================================
// GPU Gate Kernels
// =============================================================================

/// GPU kernel for single-qubit gate application
#[derive(Clone)]
pub struct SingleQubitKernel {
    /// Gate matrix (2x2)
    pub matrix: [[Complex; 2]; 2],
    /// Target qubit
    pub target: usize,
}

impl std::fmt::Debug for SingleQubitKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleQubitKernel")
            .field("target", &self.target)
            .finish()
    }
}

impl SingleQubitKernel {
    /// Create Ry rotation kernel
    pub fn ry(target: usize, theta: f64) -> Self {
        let cos_t = (theta / 2.0).cos();
        let sin_t = (theta / 2.0).sin();

        Self {
            matrix: [
                [Complex::new(cos_t, 0.0), Complex::new(-sin_t, 0.0)],
                [Complex::new(sin_t, 0.0), Complex::new(cos_t, 0.0)],
            ],
            target,
        }
    }

    /// Create Rz rotation kernel
    pub fn rz(target: usize, theta: f64) -> Self {
        let phase_neg = Complex::new((theta / 2.0).cos(), -(theta / 2.0).sin());
        let phase_pos = Complex::new((theta / 2.0).cos(), (theta / 2.0).sin());

        Self {
            matrix: [[phase_neg, Complex::ZERO], [Complex::ZERO, phase_pos]],
            target,
        }
    }

    /// Create Hadamard kernel
    pub fn hadamard(target: usize) -> Self {
        let h = Complex::new(1.0 / 2.0_f64.sqrt(), 0.0);
        Self {
            matrix: [[h, h], [h, h * (-1.0)]],
            target,
        }
    }

    /// Execute kernel on GPU state (simulated)
    pub fn execute(&self, state: &mut GpuStateVector) {
        let n = state.num_qubits;
        let mask = 1 << (n - 1 - self.target);

        // In a real GPU implementation, this would be a parallel kernel
        // Here we simulate the parallelism with a sequential loop
        for i in 0..(1 << n) {
            if i & mask == 0 {
                let j = i | mask;
                let a = state.amplitudes[i];
                let b = state.amplitudes[j];

                state.amplitudes[i] = self.matrix[0][0] * a + self.matrix[0][1] * b;
                state.amplitudes[j] = self.matrix[1][0] * a + self.matrix[1][1] * b;
            }
        }
    }
}

/// GPU kernel for two-qubit gate (CNOT)
#[derive(Debug, Clone)]
pub struct TwoQubitKernel {
    /// Control qubit
    pub control: usize,
    /// Target qubit
    pub target: usize,
    /// Gate type
    pub gate_type: TwoQubitGateType,
}

#[derive(Debug, Clone, Copy)]
pub enum TwoQubitGateType {
    Cnot,
    Cz,
    Swap,
}

impl TwoQubitKernel {
    pub fn cnot(control: usize, target: usize) -> Self {
        Self {
            control,
            target,
            gate_type: TwoQubitGateType::Cnot,
        }
    }

    pub fn cz(control: usize, target: usize) -> Self {
        Self {
            control,
            target,
            gate_type: TwoQubitGateType::Cz,
        }
    }

    pub fn execute(&self, state: &mut GpuStateVector) {
        let n = state.num_qubits;
        let control_mask = 1 << (n - 1 - self.control);
        let target_mask = 1 << (n - 1 - self.target);

        match self.gate_type {
            TwoQubitGateType::Cnot => {
                for i in 0..(1 << n) {
                    if (i & control_mask != 0) && (i & target_mask == 0) {
                        let j = i | target_mask;
                        state.amplitudes.swap(i, j);
                    }
                }
            }
            TwoQubitGateType::Cz => {
                for i in 0..(1 << n) {
                    if (i & control_mask != 0) && (i & target_mask != 0) {
                        state.amplitudes[i] = state.amplitudes[i] * (-1.0);
                    }
                }
            }
            TwoQubitGateType::Swap => {
                for i in 0..(1 << n) {
                    let c_bit = (i & control_mask != 0) as usize;
                    let t_bit = (i & target_mask != 0) as usize;
                    if c_bit != t_bit && c_bit < t_bit {
                        let j = (i & !control_mask & !target_mask)
                            | (if c_bit == 1 { target_mask } else { 0 })
                            | (if t_bit == 1 { control_mask } else { 0 });
                        state.amplitudes.swap(i, j);
                    }
                }
            }
        }
    }
}

// =============================================================================
// Batched Circuit Execution
// =============================================================================

/// A compiled circuit ready for GPU execution
#[derive(Debug, Clone)]
pub struct GpuCircuit {
    /// Single-qubit gates
    pub single_gates: Vec<SingleQubitKernel>,
    /// Two-qubit gates
    pub two_gates: Vec<TwoQubitKernel>,
    /// Number of qubits
    pub num_qubits: usize,
    /// Parameter values (for variational circuits)
    pub parameters: Vec<f64>,
}

impl GpuCircuit {
    /// Create hardware-efficient ansatz circuit
    pub fn hardware_efficient_ansatz(num_qubits: usize, num_layers: usize, params: &[f64]) -> Self {
        let mut single_gates = Vec::new();
        let mut two_gates = Vec::new();
        let mut param_idx = 0;

        for _layer in 0..num_layers {
            // Rotation layer
            for q in 0..num_qubits {
                if param_idx < params.len() {
                    single_gates.push(SingleQubitKernel::ry(q, params[param_idx]));
                    param_idx += 1;
                }
                if param_idx < params.len() {
                    single_gates.push(SingleQubitKernel::rz(q, params[param_idx]));
                    param_idx += 1;
                }
            }

            // Entangling layer
            for q in 0..(num_qubits - 1) {
                two_gates.push(TwoQubitKernel::cnot(q, q + 1));
            }
        }

        Self {
            single_gates,
            two_gates,
            num_qubits,
            parameters: params.to_vec(),
        }
    }

    /// Execute on GPU state
    pub fn execute(&self, state: &mut GpuStateVector) {
        // Interleave single and two-qubit gates
        let mut single_idx = 0;
        let mut two_idx = 0;

        let gates_per_layer = 2 * self.num_qubits + (self.num_qubits - 1);
        let rotation_gates_per_layer = 2 * self.num_qubits;

        while single_idx < self.single_gates.len() || two_idx < self.two_gates.len() {
            // Execute rotation gates for this layer
            for _ in 0..rotation_gates_per_layer {
                if single_idx < self.single_gates.len() {
                    self.single_gates[single_idx].execute(state);
                    single_idx += 1;
                }
            }

            // Execute entangling gates for this layer
            for _ in 0..(self.num_qubits - 1) {
                if two_idx < self.two_gates.len() {
                    self.two_gates[two_idx].execute(state);
                    two_idx += 1;
                }
            }
        }
    }
}

/// Batched VQE execution for variance reduction
pub struct BatchedVQE {
    /// Hamiltonian
    pub hamiltonian: Hamiltonian,
    /// GPU configuration
    pub config: GpuQuantumConfig,
    /// Noise model
    pub noise_model: NoiseModel,
}

impl BatchedVQE {
    /// Create new batched VQE
    pub fn new(hamiltonian: Hamiltonian, config: GpuQuantumConfig) -> Self {
        Self {
            hamiltonian,
            config,
            noise_model: NoiseModel::ideal(),
        }
    }

    /// Set noise model
    pub fn with_noise(mut self, noise: NoiseModel) -> Self {
        self.noise_model = noise;
        self
    }

    /// Execute batch of circuits with different parameters
    pub fn execute_batch(&self, param_sets: &[Vec<f64>]) -> Vec<EpistemicEnergy> {
        let mut results = Vec::with_capacity(param_sets.len());

        for params in param_sets {
            // Create circuit for these parameters
            let circuit =
                GpuCircuit::hardware_efficient_ansatz(self.hamiltonian.num_qubits, 2, params);

            // Initialize state
            let mut state = GpuStateVector::zero_state(self.hamiltonian.num_qubits, 0);

            // Execute circuit
            circuit.execute(&mut state);

            // Get probabilities and compute energy
            let probs = state.probabilities();
            let energy = self.hamiltonian.expectation(&probs);

            // Compute variance
            let shot_variance = compute_shot_variance(&probs, self.config.shots_per_circuit);
            let total_variance =
                shot_variance + self.noise_model.total_variance(self.hamiltonian.num_qubits);

            results.push(EpistemicEnergy::new(
                energy,
                total_variance,
                self.config.shots_per_circuit,
            ));
        }

        results
    }

    /// Parallel parameter sweep for landscape analysis
    pub fn parameter_sweep(
        &self,
        base_params: &[f64],
        param_index: usize,
        sweep_range: (f64, f64),
        num_points: usize,
    ) -> Vec<(f64, EpistemicEnergy)> {
        let mut results = Vec::with_capacity(num_points);
        let step = (sweep_range.1 - sweep_range.0) / (num_points - 1) as f64;

        for i in 0..num_points {
            let mut params = base_params.to_vec();
            let sweep_value = sweep_range.0 + i as f64 * step;
            if param_index < params.len() {
                params[param_index] = sweep_value;
            }

            // Create and execute circuit
            let circuit =
                GpuCircuit::hardware_efficient_ansatz(self.hamiltonian.num_qubits, 2, &params);
            let mut state = GpuStateVector::zero_state(self.hamiltonian.num_qubits, 0);
            circuit.execute(&mut state);

            let probs = state.probabilities();
            let energy = self.hamiltonian.expectation(&probs);
            let shot_variance = compute_shot_variance(&probs, self.config.shots_per_circuit);

            results.push((
                sweep_value,
                EpistemicEnergy::new(energy, shot_variance, self.config.shots_per_circuit),
            ));
        }

        results
    }

    /// Monte Carlo variance estimation (parallel shots)
    pub fn monte_carlo_variance(&self, params: &[f64], num_repetitions: usize) -> (f64, f64) {
        let mut energies = Vec::with_capacity(num_repetitions);

        for seed in 0..num_repetitions {
            // Create circuit with slight noise variations
            let circuit =
                GpuCircuit::hardware_efficient_ansatz(self.hamiltonian.num_qubits, 2, params);
            let mut state = GpuStateVector::zero_state(self.hamiltonian.num_qubits, 0);
            circuit.execute(&mut state);

            // Add simulated noise (would be real GPU noise in production)
            add_simulated_noise(&mut state, seed as u64, &self.noise_model);

            let probs = state.probabilities();
            let energy = self.hamiltonian.expectation(&probs);
            energies.push(energy);
        }

        // Compute mean and variance
        let mean = energies.iter().sum::<f64>() / num_repetitions as f64;
        let variance =
            energies.iter().map(|e| (e - mean).powi(2)).sum::<f64>() / (num_repetitions - 1) as f64;

        (mean, variance)
    }

    /// Combine results from multiple batches (inverse-variance weighting)
    pub fn combine_batch_results(results: &[EpistemicEnergy]) -> EpistemicEnergy {
        if results.is_empty() {
            return EpistemicEnergy::new(0.0, 1.0, 0);
        }

        if results.len() == 1 {
            return results[0].clone();
        }

        let weights: Vec<f64> = results.iter().map(|r| 1.0 / r.variance).collect();
        let total_weight: f64 = weights.iter().sum();

        let combined_mean: f64 = results
            .iter()
            .zip(weights.iter())
            .map(|(r, w)| r.mean * w)
            .sum::<f64>()
            / total_weight;

        let combined_variance = 1.0 / total_weight;
        let total_shots: usize = results.iter().map(|r| r.shots).sum();

        let mut combined = EpistemicEnergy::new(combined_mean, combined_variance, total_shots);

        // Combine variance breakdowns
        combined.variance_breakdown.shot_noise = results
            .iter()
            .zip(weights.iter())
            .map(|(r, w)| r.variance_breakdown.shot_noise * w)
            .sum::<f64>()
            / total_weight;

        combined
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn compute_shot_variance(probs: &[f64], shots: usize) -> f64 {
    probs.iter().map(|&p| p * (1.0 - p) / shots as f64).sum()
}

fn add_simulated_noise(state: &mut GpuStateVector, seed: u64, noise: &NoiseModel) {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let depol_prob = noise
        .depolarizing
        .as_ref()
        .map(|d| d.probability)
        .unwrap_or(0.0);

    for i in 0..state.amplitudes.len() {
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        i.hash(&mut hasher);
        let rand = (hasher.finish() as f64) / (u64::MAX as f64);

        if rand < depol_prob {
            // Simulate depolarizing by slightly randomizing phase
            let phase_rand = ((hasher.finish() >> 32) as f64) / (u32::MAX as f64) * 2.0 * PI;
            let phase = Complex::new(phase_rand.cos(), phase_rand.sin());
            state.amplitudes[i] = state.amplitudes[i] * phase * (1.0 - depol_prob);
        }
    }

    // Renormalize
    let norm: f64 = state.amplitudes.iter().map(|a| a.norm_sq()).sum();
    let norm = norm.sqrt();
    if norm > 1e-10 {
        for a in &mut state.amplitudes {
            *a = *a * (1.0 / norm);
        }
    }
}

// =============================================================================
// GPU Memory Management
// =============================================================================

/// GPU memory allocator for quantum states
pub struct GpuMemoryPool {
    /// Available state vectors
    pub free_states: Vec<GpuStateVector>,
    /// Total allocated memory
    pub total_allocated: usize,
    /// Peak memory usage
    pub peak_usage: usize,
    /// Configuration
    pub config: GpuQuantumConfig,
}

impl GpuMemoryPool {
    pub fn new(config: GpuQuantumConfig) -> Self {
        Self {
            free_states: Vec::new(),
            total_allocated: 0,
            peak_usage: 0,
            config,
        }
    }

    /// Allocate state vector
    pub fn allocate(&mut self, num_qubits: usize) -> GpuStateVector {
        // Check if we have a free state of the right size
        if let Some(pos) = self
            .free_states
            .iter()
            .position(|s| s.num_qubits == num_qubits)
        {
            return self.free_states.remove(pos);
        }

        // Allocate new state
        let state_size = (1 << num_qubits) * 16; // 16 bytes per complex
        self.total_allocated += state_size;
        self.peak_usage = self.peak_usage.max(self.total_allocated);

        GpuStateVector::zero_state(num_qubits, 0)
    }

    /// Free state vector (return to pool)
    pub fn free(&mut self, state: GpuStateVector) {
        self.free_states.push(state);
    }

    /// Clear pool
    pub fn clear(&mut self) {
        self.free_states.clear();
        self.total_allocated = 0;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_state_vector() {
        let state = GpuStateVector::zero_state(3, 0);
        assert_eq!(state.num_qubits, 3);
        assert_eq!(state.amplitudes.len(), 8);

        let probs = state.probabilities();
        assert!((probs[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_single_qubit_kernel_ry() {
        let kernel = SingleQubitKernel::ry(0, PI / 2.0);
        let mut state = GpuStateVector::zero_state(1, 0);

        kernel.execute(&mut state);

        // After Ry(π/2) on |0>: should be (|0> + |1>)/√2
        let probs = state.probabilities();
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_kernel_cnot() {
        let kernel = TwoQubitKernel::cnot(0, 1);

        // Test on |10> state
        let mut state = GpuStateVector::zero_state(2, 0);
        state.amplitudes[0] = Complex::ZERO;
        state.amplitudes[2] = Complex::ONE; // |10>

        kernel.execute(&mut state);

        // After CNOT: should be |11>
        let probs = state.probabilities();
        assert!(probs[2] < 1e-10); // |10> should be 0
        assert!((probs[3] - 1.0).abs() < 1e-10); // |11> should be 1
    }

    #[test]
    fn test_gpu_circuit_execution() {
        let params = vec![0.1, 0.2, 0.3, 0.4];
        let circuit = GpuCircuit::hardware_efficient_ansatz(2, 1, &params);

        let mut state = GpuStateVector::zero_state(2, 0);
        circuit.execute(&mut state);

        // State should still be normalized
        let total_prob: f64 = state.probabilities().iter().sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_batched_vqe() {
        let hamiltonian = Hamiltonian::h2_molecule();
        let config = GpuQuantumConfig::default();
        let batched = BatchedVQE::new(hamiltonian, config);

        let param_sets = vec![
            vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            vec![0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        ];

        let results = batched.execute_batch(&param_sets);

        assert_eq!(results.len(), 2);
        for result in &results {
            assert!(result.mean.is_finite());
            assert!(result.variance > 0.0);
        }
    }

    #[test]
    fn test_parameter_sweep() {
        let hamiltonian = Hamiltonian::h2_molecule();
        let config = GpuQuantumConfig::default();
        let batched = BatchedVQE::new(hamiltonian, config);

        let base_params = vec![0.0; 8];
        let sweep = batched.parameter_sweep(&base_params, 0, (-PI, PI), 5);

        assert_eq!(sweep.len(), 5);
        for (param, energy) in &sweep {
            assert!(param.is_finite());
            assert!(energy.mean.is_finite());
        }
    }

    #[test]
    fn test_monte_carlo_variance() {
        let hamiltonian = Hamiltonian::h2_molecule();
        let config = GpuQuantumConfig::default();
        let batched = BatchedVQE::new(hamiltonian, config);

        let params = vec![0.1; 8];
        let (mean, variance) = batched.monte_carlo_variance(&params, 10);

        assert!(mean.is_finite());
        assert!(variance >= 0.0);
    }

    #[test]
    fn test_combine_batch_results() {
        let r1 = EpistemicEnergy::new(-1.0, 0.01, 100);
        let r2 = EpistemicEnergy::new(-1.1, 0.02, 100);

        let combined = BatchedVQE::combine_batch_results(&[r1.clone(), r2.clone()]);

        // Combined should be weighted toward lower-variance result
        assert!(combined.mean > -1.1);
        assert!(combined.mean < -1.0);
        assert!(combined.variance < r1.variance);
    }

    #[test]
    fn test_gpu_config_max_qubits() {
        let config = GpuQuantumConfig {
            memory_limit: 1024 * 1024 * 1024, // 1 GB
            batch_size: 1,
            ..Default::default()
        };

        let max_q = config.max_qubits();
        // 1 GB / 16 bytes = 67M amplitudes = 2^26
        assert!(max_q >= 20 && max_q <= 30);
    }

    #[test]
    fn test_memory_pool() {
        let config = GpuQuantumConfig::default();
        let mut pool = GpuMemoryPool::new(config);

        let state1 = pool.allocate(3);
        assert_eq!(state1.num_qubits, 3);

        let state2 = pool.allocate(3);
        pool.free(state1);

        // Next allocation should reuse freed state
        let state3 = pool.allocate(3);
        assert_eq!(pool.free_states.len(), 0);
    }
}
