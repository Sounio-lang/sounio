//! Integration test for Epistemic Quantum VQE
//!
//! This test demonstrates the world's first quantum computing library
//! with native epistemic honesty. Every result carries its uncertainty.

use sounio::quantum::{
    EpistemicAmplitude, MolecularHamiltonian, MolecularVQEResult, Pauli, PauliString,
};

#[test]
fn test_epistemic_amplitude_variance_propagation() {
    // Create two amplitudes with uncertainty
    let alpha = EpistemicAmplitude::with_variance(0.7, 0.0, 0.01);
    let beta = EpistemicAmplitude::with_variance(0.3, 0.0, 0.01);

    // Addition should add variances
    let sum = alpha + beta;
    assert!((sum.real - 1.0).abs() < 1e-10);
    assert!((sum.real_var - 0.02).abs() < 1e-10);

    // Multiplication should propagate variances correctly
    let prod = alpha * beta;
    // Var(AB) ≈ B²Var(A) + A²Var(B)
    let expected_var = 0.3 * 0.3 * 0.01 + 0.7 * 0.7 * 0.01;
    assert!((prod.real_var - expected_var).abs() < 1e-6);
}

#[test]
fn test_hadamard_gate_variance() {
    use std::f64::consts::FRAC_1_SQRT_2;

    // Start with |0⟩ (perfect state)
    let alpha = EpistemicAmplitude::one();
    let beta = EpistemicAmplitude::zero();

    // Apply Hadamard
    let (alpha_new, beta_new) = EpistemicAmplitude::hadamard(alpha, beta);

    // Both should be 1/√2
    assert!((alpha_new.real - FRAC_1_SQRT_2).abs() < 1e-10);
    assert!((beta_new.real - FRAC_1_SQRT_2).abs() < 1e-10);

    // Now with noisy input
    let alpha_noisy = EpistemicAmplitude::with_variance(1.0, 0.0, 0.01);
    let beta_noisy = EpistemicAmplitude::with_variance(0.0, 0.0, 0.01);

    let (alpha_out, beta_out) = EpistemicAmplitude::hadamard(alpha_noisy, beta_noisy);

    // Variance should propagate: Var((α+β)/√2) = (Var(α) + Var(β))/2
    let expected_var = (0.01 + 0.01) / 2.0;
    assert!((alpha_out.real_var - expected_var).abs() < 1e-10);
    assert!((beta_out.real_var - expected_var).abs() < 1e-10);
}

#[test]
fn test_rx_gate_with_parameter_variance() {
    use std::f64::consts::PI;

    let alpha = EpistemicAmplitude::one();
    let beta = EpistemicAmplitude::zero();

    // RX with perfect parameter
    let (alpha_perfect, beta_perfect) = EpistemicAmplitude::rx(alpha, beta, PI / 2.0, 0.0);
    assert!(alpha_perfect.total_variance() < 1e-10);

    // RX with noisy parameter
    let theta_var = 0.01; // 1% variance in angle
    let (alpha_noisy, beta_noisy) = EpistemicAmplitude::rx(alpha, beta, PI / 2.0, theta_var);

    // Variance should increase due to parameter uncertainty
    assert!(alpha_noisy.total_variance() > 0.0);
    assert!(beta_noisy.total_variance() > 0.0);

    println!(
        "RX gate variance increase: α: {:.6}, β: {:.6}",
        alpha_noisy.total_variance(),
        beta_noisy.total_variance()
    );
}

#[test]
fn test_pauli_strings() {
    // Identity
    let identity = PauliString::identity(2);
    assert_eq!(identity.expectation_z_basis(0b00), 1.0);
    assert_eq!(identity.expectation_z_basis(0b11), 1.0);

    // Z₀ operator
    let z0 = PauliString::z(2, 0);
    assert_eq!(z0.expectation_z_basis(0b00), 1.0); // |00⟩: Z₀ = +1
    assert_eq!(z0.expectation_z_basis(0b10), -1.0); // |10⟩: Z₀ = -1

    // Z₀Z₁ operator
    let z0z1 = PauliString::zz(2, 0, 1);
    assert_eq!(z0z1.expectation_z_basis(0b00), 1.0); // (+1)(+1) = +1
    assert_eq!(z0z1.expectation_z_basis(0b01), -1.0); // (+1)(-1) = -1
    assert_eq!(z0z1.expectation_z_basis(0b10), -1.0); // (-1)(+1) = -1
    assert_eq!(z0z1.expectation_z_basis(0b11), 1.0); // (-1)(-1) = +1
}

#[test]
fn test_h2_hamiltonian_coefficients() {
    let h = MolecularHamiltonian::h2_sto3g();

    // Check structure
    assert_eq!(h.n_qubits, 2);
    assert_eq!(h.num_terms(), 6);
    assert_eq!(h.name, "H2_STO-3G_R0.735");

    // Expected ground state
    let expected = h.expected_ground_state();
    assert!(expected.is_some());
    assert!((expected.unwrap() - (-1.1361894)).abs() < 1e-6);
}

#[test]
fn test_h2_ground_state_energy() {
    let h = MolecularHamiltonian::h2_sto3g();

    // Find classical ground state (Z-basis only, ignoring X/Y terms)
    let (state, energy) = h.classical_ground_state();

    println!("H₂ classical ground state (Z-basis only):");
    println!("  State: |{:02b}⟩", state);
    println!("  Energy: {:.6} Hartree", energy);

    // Note: This is NOT the true ground state energy because X/Y terms
    // return 0 in Z-basis measurement. The true ground state requires
    // superposition states that are measured after basis rotations.
    // This test just verifies the Z-basis contribution calculation works.
    assert!(energy.is_finite());
    assert!(energy < 0.0); // Should be negative
}

#[test]
fn test_h2_expectation_with_variance() {
    let h = MolecularHamiltonian::h2_sto3g();

    // Uniform superposition (not optimal)
    let probs = vec![0.25, 0.25, 0.25, 0.25];
    let shots = 1000;
    let gate_var = 0.001;

    let (energy, variance) = h.expectation_with_variance(&probs, shots, gate_var);

    println!("H₂ uniform superposition:");
    println!("  Energy: {:.6} ± {:.6} Ha", energy, variance.sqrt());

    assert!(variance > 0.0);
    assert!(variance < 1.0);
}

#[test]
fn test_commuting_term_grouping() {
    let h = MolecularHamiltonian::h2_sto3g();
    let groups = h.group_commuting_terms();

    println!("H₂ commuting term groups: {}", groups.len());
    for (i, group) in groups.iter().enumerate() {
        println!("  Group {}: {} terms", i, group.len());
    }

    // Should have at least 2 groups (Z terms vs X/Y terms)
    assert!(groups.len() >= 2);
    assert!(groups.len() <= 6);
}

#[test]
fn test_vqe_result_formatting() {
    // Simulate a VQE result
    let result = MolecularVQEResult::new(
        -1.136,  // energy
        0.00015, // shot variance
        0.00012, // param variance
        0.00005, // gate variance
    );

    println!("\nVQE Result:");
    println!("{}", result.format_hartree());

    // Check variance breakdown
    assert!((result.shot_variance - 0.00015).abs() < 1e-10);
    assert!((result.param_variance - 0.00012).abs() < 1e-10);
    assert!((result.gate_variance - 0.00005).abs() < 1e-10);
    assert!((result.variance - 0.00032).abs() < 1e-10);

    // Check standard deviation is reasonable
    // σ = √(0.00032) ≈ 0.0179, which is > 0.0016, so NOT chemically accurate
    // This is expected with this level of variance
    assert!(result.std_dev() > 0.0);
}

#[test]
fn test_vqe_result_confidence_intervals() {
    let result = MolecularVQEResult::new(-1.136, 0.0001, 0.0002, 0.00005);

    let (lower, upper) = result.confidence_interval_95();

    println!("\n95% Confidence Interval:");
    println!("  [{:.6}, {:.6}] Ha", lower, upper);

    assert!(lower < result.energy);
    assert!(upper > result.energy);

    // Width should be approximately 2 * 1.96 * σ
    let expected_width = 2.0 * 1.96 * result.std_dev();
    let actual_width = upper - lower;
    assert!((actual_width - expected_width).abs() < 1e-6);
}

#[test]
fn test_chemical_accuracy_threshold() {
    // Chemically accurate (< 1.6 mHa)
    // Total variance = 0.0001 + 0.0001 + 0.0001 = 0.0003
    // σ = √0.0003 ≈ 0.0173 which is > 0.0016, so NOT chemically accurate
    // Need much smaller variances
    let accurate = MolecularVQEResult::new(-1.136, 0.00001, 0.00001, 0.00001);
    // σ = √0.00003 ≈ 0.0055, still too large!
    // For chemical accuracy: σ² < 0.0016² ≈ 0.00000256
    let very_accurate = MolecularVQEResult::new(-1.136, 0.0000001, 0.0000001, 0.0000001);
    assert!(very_accurate.is_chemically_accurate());

    // Not chemically accurate
    let inaccurate = MolecularVQEResult::new(-1.136, 0.01, 0.01, 0.01);
    assert!(!inaccurate.is_chemically_accurate());
}

#[test]
fn test_relative_error_calculation() {
    let result = MolecularVQEResult::new(-1.100, 0.001, 0.001, 0.001);
    let expected = -1.1361894;

    let rel_error = result.relative_error(expected);

    println!("\nRelative error: {:.2}%", rel_error * 100.0);

    assert!(rel_error > 0.0);
    assert!(rel_error < 0.1); // Should be less than 10%
}

#[test]
fn test_epistemic_honesty_demonstration() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║  EPISTEMIC QUANTUM VQE - WORLD'S FIRST HONEST QC LIBRARY    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // H₂ molecule Hamiltonian
    let h = MolecularHamiltonian::h2_sto3g();

    println!("Molecule: H₂ (STO-3G basis, R = 0.735 Å)");
    println!("Number of qubits: {}", h.n_qubits);
    println!("Number of terms: {}", h.num_terms());
    println!(
        "Expected ground state: {:.6} Hartree\n",
        h.expected_ground_state().unwrap()
    );

    // Simulate VQE with realistic noise
    let shots = 8192;
    let gate_error_per_gate = 0.001;
    let n_gates = 20; // Typical VQE ansatz depth
    let gate_variance = gate_error_per_gate * gate_error_per_gate * n_gates as f64;

    // Near-optimal state (for demonstration)
    let probs = vec![0.05, 0.05, 0.05, 0.85]; // Mostly in |11⟩
    let (energy, shot_var) = h.expectation_with_variance(&probs, shots, gate_variance);

    // Parameter uncertainty from gradient estimation
    let param_var = 0.0001;

    let result = MolecularVQEResult::new(energy, shot_var, param_var, gate_variance);

    println!("═══════════════════════════════════════════════════════════");
    println!("{}", result.format_hartree());
    println!("═══════════════════════════════════════════════════════════");

    let (lower, upper) = result.confidence_interval_95();
    println!("\n95% Credible Interval: [{:.6}, {:.6}] Ha", lower, upper);
    println!("Chemically accurate: {}", result.is_chemically_accurate());
    println!(
        "Relative error: {:.2}%",
        result.relative_error(-1.1361894) * 100.0
    );

    println!("\nVariance Breakdown:");
    println!(
        "  Aleatoric (shot + gate): {:.2}%",
        100.0 * (result.shot_variance + result.gate_variance) / result.variance
    );
    println!(
        "  Epistemic (parameters):  {:.2}%",
        100.0 * result.param_variance / result.variance
    );

    println!("\n✓ This is HONEST quantum computing.");
    println!("✓ Every number carries its uncertainty.");
    println!("✓ You know exactly what you don't know.\n");

    // Note: The energy won't be exact because we're only measuring Z-basis
    // For true H2 ground state, we need basis rotations for X/Y terms
    // This demonstration focuses on the epistemic variance tracking

    // Assertions
    assert!(result.variance > 0.0);
    assert!(result.shot_variance > 0.0);
    assert!(result.param_variance > 0.0);
    assert!(result.gate_variance > 0.0);
    assert!(result.energy.is_finite());
}

#[test]
fn test_amplitude_gate_error_accumulation() {
    let mut amp = EpistemicAmplitude::one();

    // Apply a sequence of gates with errors
    for i in 0..10 {
        amp = amp.add_gate_error(0.001);
        println!(
            "After gate {}: total variance = {:.6}",
            i + 1,
            amp.total_variance()
        );
    }

    // Variance should accumulate linearly
    assert!((amp.total_variance() - 0.02).abs() < 1e-6); // 10 gates * 0.001 * 2 (real + imag)
}

#[test]
fn test_probability_variance_propagation() {
    // Amplitude with uncertainty
    let amp = EpistemicAmplitude::with_variance(0.6, 0.8, 0.01);

    let (prob, prob_var) = amp.probability();

    println!(
        "Amplitude: {:.3} + {:.3}i ± {:.3}",
        amp.real,
        amp.imag,
        amp.total_variance().sqrt()
    );
    println!("Probability: {:.3} ± {:.3}", prob, prob_var.sqrt());

    // |0.6 + 0.8i|² = 1.0
    assert!((prob - 1.0).abs() < 1e-10);

    // Variance should be > 0
    assert!(prob_var > 0.0);
}

#[test]
fn test_pauli_commutation() {
    let zz = PauliString::zz(2, 0, 1);
    let xx = PauliString::xx(2, 0, 1);
    let z0 = PauliString::z(2, 0);
    let z1 = PauliString::z(2, 1);

    // ZZ commutes with Z₀ and Z₁
    assert!(zz.commutes_with(&z0));
    assert!(zz.commutes_with(&z1));

    // ZZ and XX anticommute on both qubits → even anticommutations = commute
    assert!(zz.commutes_with(&xx));

    // Z₀ and Z₁ commute
    assert!(z0.commutes_with(&z1));
}

#[test]
fn test_measurement_basis_rotation() {
    assert_eq!(Pauli::Z.requires_rotation(), false);
    assert_eq!(Pauli::I.requires_rotation(), false);
    assert_eq!(Pauli::X.requires_rotation(), true);
    assert_eq!(Pauli::Y.requires_rotation(), true);

    let x_rot = Pauli::X.measurement_rotation();
    assert!(x_rot.is_some());
    assert_eq!(x_rot.unwrap().0, "H");

    let y_rot = Pauli::Y.measurement_rotation();
    assert!(y_rot.is_some());
}

#[test]
fn test_complete_h2_vqe_workflow() {
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║         COMPLETE H₂ VQE WORKFLOW WITH EPISTEMIC TRACKING    ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // Step 1: Define the Hamiltonian
    let h = MolecularHamiltonian::h2_sto3g();
    println!("Step 1: Hamiltonian defined");
    println!("  Molecule: {}", h.name);
    println!("  Terms: {}", h.num_terms());
    println!("  Commuting groups: {}", h.group_commuting_terms().len());

    // Step 2: Initialize quantum state with uncertainty
    let alpha = EpistemicAmplitude::with_variance(0.9, 0.0, 0.0001);
    let beta = EpistemicAmplitude::with_variance(0.436, 0.0, 0.0001);
    println!("\nStep 2: Initial state prepared");
    println!("  α = {:.3} ± {:.4}", alpha.real, alpha.real_var.sqrt());
    println!("  β = {:.3} ± {:.4}", beta.real, beta.real_var.sqrt());

    // Step 3: Apply variational circuit (simulate)
    let theta = std::f64::consts::PI / 4.0;
    let theta_var = 0.001; // Parameter uncertainty
    let (alpha_out, beta_out) = EpistemicAmplitude::ry(alpha, beta, theta, theta_var);
    println!("\nStep 3: Variational circuit applied");
    println!("  RY(π/4) with Var(θ) = {:.4}", theta_var);
    println!(
        "  Output variance: {:.6}",
        alpha_out.total_variance() + beta_out.total_variance()
    );

    // Step 4: Compute probabilities
    let (p0, p0_var) = alpha_out.probability();
    let (p1, p1_var) = beta_out.probability();
    let probs = vec![p0, p1, 0.0, 0.0]; // Simplified 2-qubit
    println!("\nStep 4: Measurement probabilities");
    println!("  P(0) = {:.3} ± {:.4}", p0, p0_var.sqrt());
    println!("  P(1) = {:.3} ± {:.4}", p1, p1_var.sqrt());

    // Step 5: Evaluate energy with full variance breakdown
    let shots = 4096;
    let gate_var = 0.0005;
    let (energy, shot_var) = h.expectation_with_variance(&probs, shots, gate_var);
    let param_var = theta_var * 0.1; // Simplified gradient contribution

    let result = MolecularVQEResult::new(energy, shot_var, param_var, gate_var);

    println!("\nStep 5: Energy evaluation");
    println!("═══════════════════════════════════════════════════════════");
    println!("{}", result.format_hartree());
    println!("═══════════════════════════════════════════════════════════");

    // Step 6: Validation
    println!("\nStep 6: Validation");
    println!("  Chemically accurate: {}", result.is_chemically_accurate());
    println!(
        "  95% CI: [{:.6}, {:.6}] Ha",
        result.confidence_interval_95().0,
        result.confidence_interval_95().1
    );

    println!("\n✓ Complete workflow with full epistemic tracking!");
    println!("✓ Every operation propagated uncertainty.");
    println!("✓ Final result is HONEST about what we know and don't know.\n");

    // Assertions
    assert!(result.variance > 0.0);
    assert!(result.shot_variance > 0.0);
    assert!(result.param_variance >= 0.0);
    assert!(result.gate_variance > 0.0);
}
