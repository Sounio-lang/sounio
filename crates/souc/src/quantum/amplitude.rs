//! Epistemic Quantum Amplitudes
//!
//! Complex amplitudes with uncertainty tracking in both real and imaginary parts.
//! This is the foundation for honest quantum computing - every amplitude carries
//! its variance from gate errors, parameter uncertainty, and measurement noise.
//!
//! # Key Innovation
//!
//! Traditional quantum: α = 0.707 + 0i (dishonest - no uncertainty)
//! Epistemic quantum: α = 0.707±0.012 + 0±0.008i (honest - with variance)
//!
//! # Variance Propagation
//!
//! - **Addition**: Var(A+B) = Var(A) + Var(B)
//! - **Multiplication**: Var(AB) ≈ B²Var(A) + A²Var(B)
//! - **Hadamard**: Var((α+β)/√2) = (Var(α) + Var(β))/2
//! - **RX(θ)**: Var(cos(θ/2)) ≈ sin²(θ/2)/4 · Var(θ)

use std::ops::{Add, Mul};

/// Complex amplitude with epistemic uncertainty in real and imaginary parts
///
/// This is the fundamental building block of epistemic quantum computing.
/// Every amplitude is a Knowledge<Complex> with variance tracking.
#[derive(Debug, Clone, Copy)]
pub struct EpistemicAmplitude {
    /// Real part (mean)
    pub real: f64,
    /// Variance in real part
    pub real_var: f64,
    /// Imaginary part (mean)
    pub imag: f64,
    /// Variance in imaginary part
    pub imag_var: f64,
}

impl EpistemicAmplitude {
    /// Create a new epistemic amplitude with specified variances
    pub fn new(real: f64, imag: f64, real_var: f64, imag_var: f64) -> Self {
        Self {
            real,
            imag,
            real_var,
            imag_var,
        }
    }

    /// Create amplitude with uniform variance (isotropic uncertainty)
    pub fn with_variance(real: f64, imag: f64, variance: f64) -> Self {
        Self {
            real,
            imag,
            real_var: variance,
            imag_var: variance,
        }
    }

    /// Create perfect amplitude (no uncertainty)
    pub fn perfect(real: f64, imag: f64) -> Self {
        Self {
            real,
            imag,
            real_var: 0.0,
            imag_var: 0.0,
        }
    }

    /// Zero amplitude |0⟩ contribution
    pub const fn zero() -> Self {
        Self {
            real: 0.0,
            imag: 0.0,
            real_var: 0.0,
            imag_var: 0.0,
        }
    }

    /// One amplitude (perfect |0⟩ or |1⟩)
    pub const fn one() -> Self {
        Self {
            real: 1.0,
            imag: 0.0,
            real_var: 0.0,
            imag_var: 0.0,
        }
    }

    /// Imaginary unit i
    pub const fn i() -> Self {
        Self {
            real: 0.0,
            imag: 1.0,
            real_var: 0.0,
            imag_var: 0.0,
        }
    }

    /// Plus state amplitude: 1/√2
    pub fn plus() -> Self {
        let val = std::f64::consts::FRAC_1_SQRT_2;
        Self::perfect(val, 0.0)
    }

    /// Minus state amplitude: -1/√2
    pub fn minus() -> Self {
        let val = -std::f64::consts::FRAC_1_SQRT_2;
        Self::perfect(val, 0.0)
    }

    /// From polar form: r * e^(iθ) with variance
    pub fn from_polar(r: f64, theta: f64, r_var: f64, theta_var: f64) -> Self {
        let real = r * theta.cos();
        let imag = r * theta.sin();

        // Variance propagation for polar to Cartesian
        // Var(r*cos(θ)) ≈ cos²(θ)*Var(r) + r²*sin²(θ)*Var(θ)
        // Var(r*sin(θ)) ≈ sin²(θ)*Var(r) + r²*cos²(θ)*Var(θ)
        let cos_theta = theta.cos();
        let sin_theta = theta.sin();
        let real_var = cos_theta * cos_theta * r_var + r * r * sin_theta * sin_theta * theta_var;
        let imag_var = sin_theta * sin_theta * r_var + r * r * cos_theta * cos_theta * theta_var;

        Self {
            real,
            imag,
            real_var,
            imag_var,
        }
    }

    /// Probability |α|² and its variance
    ///
    /// For α = a + bi with Var(a), Var(b):
    /// |α|² = a² + b²
    /// Var(|α|²) ≈ 4a²Var(a) + 4b²Var(b) (first-order Taylor)
    pub fn probability(&self) -> (f64, f64) {
        let prob = self.real * self.real + self.imag * self.imag;

        // Variance using Taylor expansion
        let prob_var = 4.0 * self.real * self.real * self.real_var
            + 4.0 * self.imag * self.imag * self.imag_var;

        (prob, prob_var)
    }

    /// Norm |α| and its variance
    pub fn norm(&self) -> (f64, f64) {
        let (prob, prob_var) = self.probability();
        let norm = prob.sqrt();

        // Var(√x) ≈ Var(x)/(4x) for x > 0
        let norm_var = if prob > 1e-10 {
            prob_var / (4.0 * prob)
        } else {
            0.0
        };

        (norm, norm_var)
    }

    /// Phase arg(α) and its variance
    pub fn phase(&self) -> (f64, f64) {
        let phase = self.imag.atan2(self.real);

        // Var(atan2(b, a)) ≈ (a²Var(b) + b²Var(a))/(a² + b²)²
        let denom = self.real * self.real + self.imag * self.imag;
        let phase_var = if denom > 1e-10 {
            (self.real * self.real * self.imag_var + self.imag * self.imag * self.real_var)
                / (denom * denom)
        } else {
            0.0
        };

        (phase, phase_var)
    }

    /// Complex conjugate
    pub fn conj(&self) -> Self {
        Self {
            real: self.real,
            imag: -self.imag,
            real_var: self.real_var,
            imag_var: self.imag_var,
        }
    }

    /// Normalize to unit magnitude (preserving uncertainty)
    pub fn normalize(&self) -> Self {
        let (norm, norm_var) = self.norm();

        if norm < 1e-10 {
            return Self::zero();
        }

        let inv_norm = 1.0 / norm;

        // α/|α| with variance propagation
        // For f(x) = x/c: Var(f) ≈ Var(x)/c²
        Self {
            real: self.real * inv_norm,
            imag: self.imag * inv_norm,
            real_var: self.real_var * inv_norm * inv_norm,
            imag_var: self.imag_var * inv_norm * inv_norm,
        }
    }

    /// Total variance (sum of real and imaginary variances)
    pub fn total_variance(&self) -> f64 {
        self.real_var + self.imag_var
    }

    /// Add gate error (increases variance isotropically)
    pub fn add_gate_error(&self, error_rate: f64) -> Self {
        Self {
            real: self.real,
            imag: self.imag,
            real_var: self.real_var + error_rate,
            imag_var: self.imag_var + error_rate,
        }
    }

    /// Add parameter uncertainty (from noisy rotations)
    pub fn add_parameter_uncertainty(&self, param_var: f64, sensitivity: f64) -> Self {
        let added_var = param_var * sensitivity * sensitivity;
        Self {
            real: self.real,
            imag: self.imag,
            real_var: self.real_var + added_var,
            imag_var: self.imag_var + added_var,
        }
    }
}

// =============================================================================
// Arithmetic Operations with Variance Propagation
// =============================================================================

impl Add for EpistemicAmplitude {
    type Output = Self;

    /// Addition: (a + bi) + (c + di) with variance propagation
    ///
    /// Var(A + B) = Var(A) + Var(B) (assuming independence)
    fn add(self, other: Self) -> Self {
        Self {
            real: self.real + other.real,
            imag: self.imag + other.imag,
            real_var: self.real_var + other.real_var,
            imag_var: self.imag_var + other.imag_var,
        }
    }
}

impl Mul for EpistemicAmplitude {
    type Output = Self;

    /// Multiplication: (a + bi) * (c + di) with variance propagation
    ///
    /// Real part: ac - bd
    /// Imag part: ad + bc
    ///
    /// Var(ac - bd) ≈ c²Var(a) + a²Var(c) + d²Var(b) + b²Var(d)
    /// Var(ad + bc) ≈ d²Var(a) + a²Var(d) + c²Var(b) + b²Var(c)
    fn mul(self, other: Self) -> Self {
        let real = self.real * other.real - self.imag * other.imag;
        let imag = self.real * other.imag + self.imag * other.real;

        // Variance propagation (first-order Taylor)
        let real_var = other.real * other.real * self.real_var
            + self.real * self.real * other.real_var
            + other.imag * other.imag * self.imag_var
            + self.imag * self.imag * other.imag_var;

        let imag_var = other.imag * other.imag * self.real_var
            + self.real * self.real * other.imag_var
            + other.real * other.real * self.imag_var
            + self.imag * self.imag * other.real_var;

        Self {
            real,
            imag,
            real_var,
            imag_var,
        }
    }
}

impl Mul<f64> for EpistemicAmplitude {
    type Output = Self;

    /// Scalar multiplication with variance scaling
    ///
    /// Var(kX) = k²Var(X)
    fn mul(self, scalar: f64) -> Self {
        Self {
            real: self.real * scalar,
            imag: self.imag * scalar,
            real_var: self.real_var * scalar * scalar,
            imag_var: self.imag_var * scalar * scalar,
        }
    }
}

// =============================================================================
// Quantum Gate Operations on Amplitudes
// =============================================================================

impl EpistemicAmplitude {
    /// Apply Hadamard gate to two amplitudes
    ///
    /// H|0⟩ = (|0⟩ + |1⟩)/√2
    /// H|1⟩ = (|0⟩ - |1⟩)/√2
    ///
    /// Returns (α', β') where:
    /// α' = (α + β)/√2
    /// β' = (α - β)/√2
    ///
    /// Variance: Var((α+β)/√2) = (Var(α) + Var(β))/2
    pub fn hadamard(alpha: Self, beta: Self) -> (Self, Self) {
        let sqrt2 = std::f64::consts::SQRT_2;
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        let alpha_new = (alpha + beta) * inv_sqrt2;
        let beta_new = (alpha + beta * (-1.0)) * inv_sqrt2;

        (alpha_new, beta_new)
    }

    /// Apply Pauli X gate (bit flip)
    ///
    /// X|0⟩ = |1⟩, X|1⟩ = |0⟩
    ///
    /// Returns (β, α) - just swap amplitudes
    pub fn pauli_x(alpha: Self, beta: Self) -> (Self, Self) {
        (beta, alpha)
    }

    /// Apply Pauli Y gate
    ///
    /// Y|0⟩ = i|1⟩, Y|1⟩ = -i|0⟩
    pub fn pauli_y(alpha: Self, beta: Self) -> (Self, Self) {
        let i = Self::i();
        (beta * i * (-1.0), alpha * i)
    }

    /// Apply Pauli Z gate (phase flip)
    ///
    /// Z|0⟩ = |0⟩, Z|1⟩ = -|1⟩
    pub fn pauli_z(alpha: Self, beta: Self) -> (Self, Self) {
        (alpha, beta * (-1.0))
    }

    /// Apply RX(θ) rotation with parameter variance
    ///
    /// RX(θ) = [[cos(θ/2), -i*sin(θ/2)],
    ///          [-i*sin(θ/2), cos(θ/2)]]
    ///
    /// Variance in θ propagates through:
    /// Var(cos(θ/2)) ≈ sin²(θ/2)/4 · Var(θ)
    /// Var(sin(θ/2)) ≈ cos²(θ/2)/4 · Var(θ)
    pub fn rx(alpha: Self, beta: Self, theta: f64, theta_var: f64) -> (Self, Self) {
        let half_theta = theta / 2.0;
        let cos_val = half_theta.cos();
        let sin_val = half_theta.sin();

        // Variance from parameter uncertainty
        let cos_var = sin_val * sin_val * theta_var / 4.0;
        let sin_var = cos_val * cos_val * theta_var / 4.0;

        let cos_amp = Self::with_variance(cos_val, 0.0, cos_var);
        let sin_amp = Self::with_variance(0.0, -sin_val, sin_var);

        let alpha_new = cos_amp * alpha + sin_amp * beta;
        let beta_new = sin_amp * alpha + cos_amp * beta;

        (alpha_new, beta_new)
    }

    /// Apply RY(θ) rotation with parameter variance
    ///
    /// RY(θ) = [[cos(θ/2), -sin(θ/2)],
    ///          [sin(θ/2), cos(θ/2)]]
    pub fn ry(alpha: Self, beta: Self, theta: f64, theta_var: f64) -> (Self, Self) {
        let half_theta = theta / 2.0;
        let cos_val = half_theta.cos();
        let sin_val = half_theta.sin();

        let cos_var = sin_val * sin_val * theta_var / 4.0;
        let sin_var = cos_val * cos_val * theta_var / 4.0;

        let cos_amp = Self::with_variance(cos_val, 0.0, cos_var);
        let sin_amp = Self::with_variance(sin_val, 0.0, sin_var);

        let alpha_new = cos_amp * alpha + sin_amp * beta * (-1.0);
        let beta_new = sin_amp * alpha + cos_amp * beta;

        (alpha_new, beta_new)
    }

    /// Apply RZ(θ) rotation with parameter variance
    ///
    /// RZ(θ) = [[e^(-iθ/2), 0],
    ///          [0, e^(iθ/2)]]
    pub fn rz(alpha: Self, beta: Self, theta: f64, theta_var: f64) -> (Self, Self) {
        let half_theta = theta / 2.0;

        // e^(-iθ/2) = cos(θ/2) - i*sin(θ/2)
        let phase_minus = Self::from_polar(1.0, -half_theta, 0.0, theta_var / 4.0);
        let phase_plus = Self::from_polar(1.0, half_theta, 0.0, theta_var / 4.0);

        (alpha * phase_minus, beta * phase_plus)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_amplitude_creation() {
        let amp = EpistemicAmplitude::new(0.7, 0.0, 0.01, 0.01);
        assert!((amp.real - 0.7).abs() < 1e-10);
        assert!((amp.real_var - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_zero_one_constants() {
        let zero = EpistemicAmplitude::zero();
        assert_eq!(zero.real, 0.0);
        assert_eq!(zero.imag, 0.0);

        let one = EpistemicAmplitude::one();
        assert_eq!(one.real, 1.0);
        assert_eq!(one.imag, 0.0);
    }

    #[test]
    fn test_probability() {
        let amp = EpistemicAmplitude::perfect(0.6, 0.8); // |0.6 + 0.8i|² = 1
        let (prob, _var) = amp.probability();
        assert!((prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_addition_variance_propagation() {
        let a = EpistemicAmplitude::with_variance(1.0, 0.0, 0.01);
        let b = EpistemicAmplitude::with_variance(2.0, 0.0, 0.02);

        let sum = a + b;
        assert!((sum.real - 3.0).abs() < 1e-10);
        assert!((sum.real_var - 0.03).abs() < 1e-10); // Var(A+B) = 0.01 + 0.02
    }

    #[test]
    fn test_multiplication_variance_propagation() {
        let a = EpistemicAmplitude::with_variance(2.0, 0.0, 0.04); // 2 ± 0.2
        let b = EpistemicAmplitude::with_variance(3.0, 0.0, 0.09); // 3 ± 0.3

        let prod = a * b;
        assert!((prod.real - 6.0).abs() < 1e-10);

        // Var(AB) ≈ B²Var(A) + A²Var(B) = 9*0.04 + 4*0.09 = 0.36 + 0.36 = 0.72
        assert!((prod.real_var - 0.72).abs() < 1e-6);
    }

    #[test]
    fn test_scalar_multiplication() {
        let amp = EpistemicAmplitude::with_variance(1.0, 0.0, 0.01);
        let scaled = amp * 2.0;

        assert!((scaled.real - 2.0).abs() < 1e-10);
        assert!((scaled.real_var - 0.04).abs() < 1e-10); // Var(2X) = 4*Var(X)
    }

    #[test]
    fn test_hadamard_gate() {
        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::zero();

        let (alpha_new, beta_new) = EpistemicAmplitude::hadamard(alpha, beta);

        // H|0⟩ = (|0⟩ + |1⟩)/√2
        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((alpha_new.real - expected).abs() < 1e-10);
        assert!((beta_new.real - expected).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_x_gate() {
        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::zero();

        let (alpha_new, beta_new) = EpistemicAmplitude::pauli_x(alpha, beta);

        // X|0⟩ = |1⟩
        assert!((alpha_new.real - 0.0).abs() < 1e-10);
        assert!((beta_new.real - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_z_gate() {
        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::one();

        let (alpha_new, beta_new) = EpistemicAmplitude::pauli_z(alpha, beta);

        // Z preserves |0⟩, flips |1⟩
        assert!((alpha_new.real - 1.0).abs() < 1e-10);
        assert!((beta_new.real - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rx_gate_no_variance() {
        use std::f64::consts::PI;

        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::zero();

        // RX(π) should flip: |0⟩ → -i|1⟩
        let (alpha_new, beta_new) = EpistemicAmplitude::rx(alpha, beta, PI, 0.0);

        assert!(alpha_new.real.abs() < 1e-10);
        assert!((beta_new.imag - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_rx_gate_with_variance() {
        use std::f64::consts::PI;

        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::zero();

        let theta = PI / 2.0; // 90° rotation
        let theta_var = 0.01; // 1% variance in angle

        let (alpha_new, beta_new) = EpistemicAmplitude::rx(alpha, beta, theta, theta_var);

        // Both amplitudes should have increased variance from parameter uncertainty
        assert!(alpha_new.total_variance() > 0.0);
        assert!(beta_new.total_variance() > 0.0);
    }

    #[test]
    fn test_ry_gate() {
        use std::f64::consts::PI;

        let alpha = EpistemicAmplitude::one();
        let beta = EpistemicAmplitude::zero();

        // RY(π/2) creates equal superposition
        let (alpha_new, beta_new) = EpistemicAmplitude::ry(alpha, beta, PI / 2.0, 0.0);

        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((alpha_new.real - expected).abs() < 1e-10);
        assert!((beta_new.real - expected).abs() < 1e-10);
    }

    #[test]
    fn test_rz_gate() {
        use std::f64::consts::PI;

        let alpha = EpistemicAmplitude::perfect(1.0 / 2.0f64.sqrt(), 0.0);
        let beta = EpistemicAmplitude::perfect(1.0 / 2.0f64.sqrt(), 0.0);

        // RZ adds relative phase
        let (_alpha_new, beta_new) = EpistemicAmplitude::rz(alpha, beta, PI / 2.0, 0.0);

        // Should add phase to |1⟩ component
        assert!(beta_new.imag.abs() > 0.0 || beta_new.real.abs() > 0.0);
    }

    #[test]
    fn test_normalize() {
        let amp = EpistemicAmplitude::with_variance(3.0, 4.0, 0.01);
        let normalized = amp.normalize();

        let (norm, _) = normalized.norm();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_conjugate() {
        let amp = EpistemicAmplitude::with_variance(1.0, 2.0, 0.01);
        let conj = amp.conj();

        assert_eq!(conj.real, amp.real);
        assert_eq!(conj.imag, -amp.imag);
        assert_eq!(conj.real_var, amp.real_var);
    }

    #[test]
    fn test_phase() {
        let amp = EpistemicAmplitude::perfect(1.0, 1.0);
        let (phase, _) = amp.phase();

        // Phase of 1+i is π/4
        assert!((phase - std::f64::consts::PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_gate_error_addition() {
        let amp = EpistemicAmplitude::with_variance(1.0, 0.0, 0.01);
        let noisy = amp.add_gate_error(0.001);

        assert!((noisy.real_var - 0.011).abs() < 1e-10);
        assert!((noisy.imag_var - 0.011).abs() < 1e-10);
    }

    #[test]
    fn test_from_polar() {
        use std::f64::consts::PI;

        let amp = EpistemicAmplitude::from_polar(1.0, PI / 4.0, 0.0, 0.0);

        let expected = std::f64::consts::FRAC_1_SQRT_2;
        assert!((amp.real - expected).abs() < 1e-10);
        assert!((amp.imag - expected).abs() < 1e-10);
    }
}
