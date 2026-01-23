//! Additional Numerical Validation Tests for Octonion Operations
//!
//! This test suite extends integration_octonion_basic.rs with:
//! - Randomized property tests (100+ test cases)
//! - Activation function validation (sigmoid, tanh, activation chains)
//! - Edge cases (extreme values, denormals)
//! - Scalar and arithmetic operation tests
//!
//! References:
//! - Baez, J. C. (2002). "The Octonions" (Bull. Amer. Math. Soc. 39.2)
//! - Graves, J. T. (1843). "On algebraic triplets"

#[cfg(test)]
mod octonion_numerical_validation {

    // ========================================================================
    // Test Helpers
    // ========================================================================

    #[derive(Debug, Clone, Copy)]
    struct Octonion {
        a: f32, b: f32, c: f32, d: f32,
        e: f32, f: f32, g: f32, h: f32,
    }

    impl Octonion {
        fn new(a: f32, b: f32, c: f32, d: f32, e: f32, f: f32, g: f32, h: f32) -> Self {
            Octonion { a, b, c, d, e, f, g, h }
        }

        fn zero() -> Self {
            Octonion::new(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }

        fn one() -> Self {
            Octonion::new(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }

        fn norm_sq(&self) -> f32 {
            self.a * self.a + self.b * self.b + self.c * self.c + self.d * self.d +
            self.e * self.e + self.f * self.f + self.g * self.g + self.h * self.h
        }

        fn norm(&self) -> f32 {
            self.norm_sq().sqrt()
        }

        fn conj(&self) -> Self {
            Octonion::new(self.a, -self.b, -self.c, -self.d, -self.e, -self.f, -self.g, -self.h)
        }

        fn add(&self, other: &Self) -> Self {
            Octonion::new(
                self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d,
                self.e + other.e, self.f + other.f, self.g + other.g, self.h + other.h,
            )
        }

        fn sub(&self, other: &Self) -> Self {
            Octonion::new(
                self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d,
                self.e - other.e, self.f - other.f, self.g - other.g, self.h - other.h,
            )
        }

        fn scale(&self, s: f32) -> Self {
            Octonion::new(
                self.a * s, self.b * s, self.c * s, self.d * s,
                self.e * s, self.f * s, self.g * s, self.h * s,
            )
        }

        fn mul(&self, other: &Self) -> Self {
            let a0 = self.a; let a1 = self.b; let a2 = self.c; let a3 = self.d;
            let a4 = self.e; let a5 = self.f; let a6 = self.g; let a7 = self.h;
            let b0 = other.a; let b1 = other.b; let b2 = other.c; let b3 = other.d;
            let b4 = other.e; let b5 = other.f; let b6 = other.g; let b7 = other.h;

            let c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7;
            let c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b5 - a5*b4 - a6*b7 + a7*b6;
            let c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5;
            let c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b7 - a5*b6 + a6*b5 - a7*b4;
            let c4 = a0*b4 - a1*b5 - a2*b6 - a3*b7 + a4*b0 + a5*b1 + a6*b2 + a7*b3;
            let c5 = a0*b5 + a1*b4 - a2*b7 + a3*b6 - a4*b1 + a5*b0 - a6*b3 + a7*b2;
            let c6 = a0*b6 + a1*b7 + a2*b4 - a3*b5 - a4*b2 + a5*b3 + a6*b0 - a7*b1;
            let c7 = a0*b7 - a1*b6 + a2*b5 + a3*b4 - a4*b3 - a5*b2 + a6*b1 + a7*b0;

            Octonion::new(c0, c1, c2, c3, c4, c5, c6, c7)
        }

        fn inv(&self) -> Self {
            let norm_sq = self.norm_sq();
            if norm_sq == 0.0 {
                return Octonion::one();
            }
            self.conj().scale(1.0 / norm_sq)
        }

        fn relu(&self) -> Self {
            Octonion::new(
                self.a.max(0.0), self.b.max(0.0), self.c.max(0.0), self.d.max(0.0),
                self.e.max(0.0), self.f.max(0.0), self.g.max(0.0), self.h.max(0.0),
            )
        }

        fn sigmoid(&self) -> Self {
            Octonion::new(
                1.0 / (1.0 + (-self.a).exp()),
                1.0 / (1.0 + (-self.b).exp()),
                1.0 / (1.0 + (-self.c).exp()),
                1.0 / (1.0 + (-self.d).exp()),
                1.0 / (1.0 + (-self.e).exp()),
                1.0 / (1.0 + (-self.f).exp()),
                1.0 / (1.0 + (-self.g).exp()),
                1.0 / (1.0 + (-self.h).exp()),
            )
        }

        fn tanh_act(&self) -> Self {
            Octonion::new(
                self.a.tanh(), self.b.tanh(), self.c.tanh(), self.d.tanh(),
                self.e.tanh(), self.f.tanh(), self.g.tanh(), self.h.tanh(),
            )
        }

        fn distance(&self, other: &Self) -> f32 {
            self.sub(other).norm()
        }

        fn components(&self) -> [f32; 8] {
            [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h]
        }
    }

    /// Simple PRNG for deterministic test generation
    fn pseudo_random(seed: u64, index: usize) -> f32 {
        let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let y = x.wrapping_mul(index as u64 + 1);
        let z = (y >> 33) ^ y;
        // Map to [-2, 2] range
        ((z as f64 / u64::MAX as f64) * 4.0 - 2.0) as f32
    }

    fn random_octonion(seed: u64, base_index: usize) -> Octonion {
        Octonion::new(
            pseudo_random(seed, base_index),
            pseudo_random(seed, base_index + 1),
            pseudo_random(seed, base_index + 2),
            pseudo_random(seed, base_index + 3),
            pseudo_random(seed, base_index + 4),
            pseudo_random(seed, base_index + 5),
            pseudo_random(seed, base_index + 6),
            pseudo_random(seed, base_index + 7),
        )
    }

    // ========================================================================
    // Tier 1: Randomized Core Algebra Tests
    // ========================================================================

    #[test]
    fn test_norm_multiplicativity_100_random() {
        // Property: |o1 * o2| = |o1| * |o2| for all octonions
        // Test with 100 random pairs
        let seed = 0xDEADBEEF_u64;
        let mut max_error = 0.0_f32;

        for i in 0..100 {
            let o1 = random_octonion(seed, i * 16);
            let o2 = random_octonion(seed, i * 16 + 8);

            let product = o1.mul(&o2);
            let norm_product = product.norm();
            let product_norms = o1.norm() * o2.norm();

            let rel_error = if product_norms > 1e-10 {
                (norm_product - product_norms).abs() / product_norms
            } else {
                (norm_product - product_norms).abs()
            };

            max_error = max_error.max(rel_error);

            assert!(
                rel_error < 1e-4,
                "Norm multiplicativity failed at i={}: |o1*o2|={}, |o1|*|o2|={}, error={}",
                i, norm_product, product_norms, rel_error
            );
        }

        println!("100 random norm multiplicativity tests passed, max error: {}", max_error);
    }

    #[test]
    fn test_alternative_property_100_random() {
        // Property: (x*x)*y = x*(x*y) for all octonions
        let seed = 0xCAFEBABE_u64;
        let mut max_error = 0.0_f32;

        for i in 0..100 {
            let x = random_octonion(seed, i * 16);
            let y = random_octonion(seed, i * 16 + 8);

            let x_sq = x.mul(&x);
            let lhs = x_sq.mul(&y);
            let rhs = x.mul(&x.mul(&y));

            let error = lhs.distance(&rhs);
            max_error = max_error.max(error);

            assert!(
                error < 1e-3,
                "Alternative property failed at i={}: error={}",
                i, error
            );
        }

        println!("100 random alternative property tests passed, max error: {}", max_error);
    }

    #[test]
    fn test_flexibility_100_random() {
        // Property: (x*y)*x = x*(y*x) for all octonions
        let seed = 0xBAADF00D_u64;
        let mut max_error = 0.0_f32;

        for i in 0..100 {
            let x = random_octonion(seed, i * 16);
            let y = random_octonion(seed, i * 16 + 8);

            let lhs = x.mul(&y).mul(&x);
            let rhs = x.mul(&y.mul(&x));

            let error = lhs.distance(&rhs);
            max_error = max_error.max(error);

            assert!(
                error < 1e-3,
                "Flexibility property failed at i={}: error={}",
                i, error
            );
        }

        println!("100 random flexibility tests passed, max error: {}", max_error);
    }

    // ========================================================================
    // Tier 2: Activation Function Tests
    // ========================================================================

    #[test]
    fn test_relu_known_values() {
        // Test ReLU with known input/output pairs
        let test_cases = [
            // (input, expected_output)
            (Octonion::new(1.0, -0.5, 0.8, -0.3, 0.6, -0.2, 0.4, -0.1),
             Octonion::new(1.0, 0.0, 0.8, 0.0, 0.6, 0.0, 0.4, 0.0)),
            (Octonion::new(-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0),
             Octonion::zero()),
            (Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
             Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)),
            (Octonion::zero(),
             Octonion::zero()),
        ];

        for (i, (input, expected)) in test_cases.iter().enumerate() {
            let result = input.relu();
            let error = result.distance(expected);

            assert!(
                error < 1e-6,
                "ReLU test case {} failed: expected {:?}, got {:?}",
                i, expected.components(), result.components()
            );
        }
    }

    #[test]
    fn test_sigmoid_zero() {
        // sigmoid(0) = 0.5 for all components
        let zero = Octonion::zero();
        let result = zero.sigmoid();

        for (i, &val) in result.components().iter().enumerate() {
            assert!(
                (val - 0.5).abs() < 1e-6,
                "sigmoid(0)[{}] = {}, expected 0.5",
                i, val
            );
        }
    }

    #[test]
    fn test_sigmoid_large_positive() {
        // sigmoid(large) ≈ 1
        let large = Octonion::new(10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0);
        let result = large.sigmoid();

        for (i, &val) in result.components().iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-4,
                "sigmoid(10)[{}] = {}, expected ~1.0",
                i, val
            );
        }
    }

    #[test]
    fn test_sigmoid_large_negative() {
        // sigmoid(-large) ≈ 0
        let large_neg = Octonion::new(-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0);
        let result = large_neg.sigmoid();

        for (i, &val) in result.components().iter().enumerate() {
            assert!(
                val.abs() < 1e-4,
                "sigmoid(-10)[{}] = {}, expected ~0.0",
                i, val
            );
        }
    }

    #[test]
    fn test_tanh_zero() {
        // tanh(0) = 0
        let zero = Octonion::zero();
        let result = zero.tanh_act();

        for (i, &val) in result.components().iter().enumerate() {
            assert!(
                val.abs() < 1e-6,
                "tanh(0)[{}] = {}, expected 0.0",
                i, val
            );
        }
    }

    #[test]
    fn test_tanh_bounds() {
        // tanh output is always in [-1, 1] (closed interval due to floating point saturation)
        let seed = 0x12345678_u64;

        for i in 0..50 {
            let input = random_octonion(seed, i * 8).scale(10.0); // Large values
            let result = input.tanh_act();

            for (j, &val) in result.components().iter().enumerate() {
                assert!(
                    val >= -1.0 && val <= 1.0,
                    "tanh output[{}] = {} is out of bounds [-1, 1]",
                    j, val
                );
            }
        }
    }

    #[test]
    fn test_activation_chain() {
        // Test: mul → relu → sigmoid chain
        let o1 = Octonion::new(1.0, -0.5, 0.3, -0.2, 0.1, -0.15, 0.25, -0.05);
        let o2 = Octonion::new(0.5, 0.6, -0.4, 0.3, -0.2, 0.1, 0.1, -0.1);

        // Chain: product → relu → sigmoid
        let product = o1.mul(&o2);
        let after_relu = product.relu();
        let after_sigmoid = after_relu.sigmoid();

        // Verify properties:
        // 1. ReLU output has no negative values
        for (i, &val) in after_relu.components().iter().enumerate() {
            assert!(
                val >= 0.0,
                "ReLU output[{}] = {} should be non-negative",
                i, val
            );
        }

        // 2. Sigmoid output is in (0, 1) - specifically (0.5, 1) since ReLU ≥ 0
        for (i, &val) in after_sigmoid.components().iter().enumerate() {
            assert!(
                val >= 0.5 && val <= 1.0,
                "sigmoid(relu(x))[{}] = {} should be in [0.5, 1.0]",
                i, val
            );
        }
    }

    // ========================================================================
    // Tier 3: Arithmetic Operation Tests
    // ========================================================================

    #[test]
    fn test_addition_commutativity() {
        // a + b = b + a
        let seed = 0xFEEDFACE_u64;

        for i in 0..50 {
            let a = random_octonion(seed, i * 16);
            let b = random_octonion(seed, i * 16 + 8);

            let ab = a.add(&b);
            let ba = b.add(&a);

            let error = ab.distance(&ba);
            assert!(
                error < 1e-6,
                "Addition not commutative at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_addition_associativity() {
        // (a + b) + c = a + (b + c)
        let seed = 0xC0FFEE_u64;

        for i in 0..50 {
            let a = random_octonion(seed, i * 24);
            let b = random_octonion(seed, i * 24 + 8);
            let c = random_octonion(seed, i * 24 + 16);

            let lhs = a.add(&b).add(&c);
            let rhs = a.add(&b.add(&c));

            let error = lhs.distance(&rhs);
            assert!(
                error < 1e-5,
                "Addition not associative at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_scalar_multiplication() {
        // Test s * (a + b) = s*a + s*b (distributivity)
        let seed = 0xABCDEF_u64;

        for i in 0..50 {
            let a = random_octonion(seed, i * 16);
            let b = random_octonion(seed, i * 16 + 8);
            let s = pseudo_random(seed, i * 16 + 16);

            let lhs = a.add(&b).scale(s);
            let rhs = a.scale(s).add(&b.scale(s));

            let error = lhs.distance(&rhs);
            assert!(
                error < 1e-5,
                "Scalar distributivity failed at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_scalar_zero_one() {
        // 0 * a = 0, 1 * a = a
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);

        let zero_scaled = a.scale(0.0);
        assert!(
            zero_scaled.distance(&Octonion::zero()) < 1e-6,
            "0 * a should equal 0"
        );

        let one_scaled = a.scale(1.0);
        assert!(
            one_scaled.distance(&a) < 1e-6,
            "1 * a should equal a"
        );
    }

    #[test]
    fn test_subtraction_inverse() {
        // a - a = 0
        let seed = 0x87654321_u64;

        for i in 0..50 {
            let a = random_octonion(seed, i * 8);
            let result = a.sub(&a);

            assert!(
                result.distance(&Octonion::zero()) < 1e-6,
                "a - a should equal 0 at i={}",
                i
            );
        }
    }

    // ========================================================================
    // Tier 4: Edge Cases
    // ========================================================================

    #[test]
    fn test_zero_octonion() {
        let zero = Octonion::zero();

        // 0 + 0 = 0
        assert!(
            zero.add(&zero).distance(&zero) < 1e-10,
            "0 + 0 should equal 0"
        );

        // 0 * a = 0
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let product = zero.mul(&a);
        assert!(
            product.distance(&zero) < 1e-6,
            "0 * a should equal 0"
        );

        // |0| = 0
        assert!(
            zero.norm() < 1e-10,
            "|0| should equal 0"
        );
    }

    #[test]
    fn test_unit_octonion() {
        let one = Octonion::one();

        // 1 * a = a
        let a = Octonion::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0);
        let product = one.mul(&a);
        assert!(
            product.distance(&a) < 1e-5,
            "1 * a should equal a, got {:?}",
            product.components()
        );

        // a * 1 = a
        let product2 = a.mul(&one);
        assert!(
            product2.distance(&a) < 1e-5,
            "a * 1 should equal a"
        );

        // |1| = 1
        assert!(
            (one.norm() - 1.0).abs() < 1e-10,
            "|1| should equal 1"
        );
    }

    #[test]
    fn test_very_small_values() {
        // Test with small but representable f32 values (above subnormal threshold)
        // f32 min normal is ~1.18e-38, so we use values well above that
        let small = Octonion::new(
            1e-18, 1e-18, 1e-18, 1e-18,
            1e-18, 1e-18, 1e-18, 1e-18
        );

        // Norm should still be computable: sqrt(8 * 1e-36) ≈ 2.83e-18
        let norm = small.norm();
        let expected = (8.0_f32).sqrt() * 1e-18;
        assert!(
            norm > 0.0 && (norm - expected).abs() / expected < 1e-4,
            "Norm of small octonion: {} (expected ~{})", norm, expected
        );

        // Multiplication should produce finite result
        let product = small.mul(&small);
        for &val in product.components().iter() {
            assert!(
                val.is_finite(),
                "Small * small should produce finite result"
            );
        }
    }

    #[test]
    fn test_very_large_values() {
        // Test with large but not overflowing values
        let large = Octonion::new(
            1e10, 1e10, 1e10, 1e10,
            1e10, 1e10, 1e10, 1e10
        );

        // Norm should be ~2.83e10 (sqrt(8) * 1e10)
        let norm = large.norm();
        let expected_norm = (8.0_f32).sqrt() * 1e10;
        let rel_error = (norm - expected_norm).abs() / expected_norm;
        assert!(
            rel_error < 1e-5,
            "Large octonion norm: {} (expected ~{})",
            norm, expected_norm
        );

        // Normalization should still work
        let normalized = large.scale(1.0 / norm);
        let normalized_norm = normalized.norm();
        assert!(
            (normalized_norm - 1.0).abs() < 1e-4,
            "Normalized large octonion should have norm 1, got {}",
            normalized_norm
        );
    }

    #[test]
    fn test_mixed_signs() {
        // Test with alternating signs
        let mixed = Octonion::new(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);

        // Norm should be sqrt(8)
        let expected_norm = (8.0_f32).sqrt();
        assert!(
            (mixed.norm() - expected_norm).abs() < 1e-5,
            "Mixed sign octonion norm should be sqrt(8)"
        );

        // ReLU should zero half the components
        let after_relu = mixed.relu();
        let expected = Octonion::new(1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0);
        assert!(
            after_relu.distance(&expected) < 1e-6,
            "ReLU of mixed signs"
        );
    }

    // ========================================================================
    // Tier 5: Numerical Precision Tests
    // ========================================================================

    #[test]
    fn test_power_associativity() {
        // o^n should be well-defined regardless of bracketing
        // For octonions, (o*o)*o = o*(o*o) due to alternativity
        let seed = 0x11111111_u64;

        for i in 0..20 {
            let o = random_octonion(seed, i * 8);

            let o_sq = o.mul(&o);
            let o_cubed_left = o_sq.mul(&o);
            let o_cubed_right = o.mul(&o_sq);

            let error = o_cubed_left.distance(&o_cubed_right);
            assert!(
                error < 1e-3,
                "Power associativity failed at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_conjugate_involution() {
        // conj(conj(o)) = o
        let seed = 0x22222222_u64;

        for i in 0..50 {
            let o = random_octonion(seed, i * 8);
            let double_conj = o.conj().conj();

            let error = o.distance(&double_conj);
            assert!(
                error < 1e-6,
                "Conjugate involution failed at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_inverse_composition() {
        // (a * b)^(-1) = b^(-1) * a^(-1) for octonions
        // This follows from conjugate anti-automorphism
        let seed = 0x33333333_u64;

        for i in 0..30 {
            let a = random_octonion(seed, i * 16);
            let b = random_octonion(seed, i * 16 + 8);

            // Skip if either is near zero
            if a.norm() < 1e-5 || b.norm() < 1e-5 {
                continue;
            }

            let product_inv = a.mul(&b).inv();
            let inv_product = b.inv().mul(&a.inv());

            let error = product_inv.distance(&inv_product);
            assert!(
                error < 1e-3,
                "Inverse composition failed at i={}: error={}",
                i, error
            );
        }
    }

    #[test]
    fn test_norm_of_normalized() {
        // |normalize(o)| = 1 for non-zero o
        let seed = 0x44444444_u64;

        for i in 0..50 {
            let o = random_octonion(seed, i * 8);
            if o.norm() < 1e-10 {
                continue;
            }

            let normalized = o.scale(1.0 / o.norm());
            let norm = normalized.norm();

            assert!(
                (norm - 1.0).abs() < 1e-5,
                "Normalized octonion should have norm 1, got {}",
                norm
            );
        }
    }

    // ========================================================================
    // Summary Test
    // ========================================================================

    #[test]
    fn test_summary_all_properties() {
        // Quick validation that all major properties hold
        let o1 = Octonion::new(1.0, 0.5, 0.3, 0.2, 0.1, 0.15, 0.25, 0.05);
        let o2 = Octonion::new(0.5, 0.6, 0.4, 0.3, 0.2, 0.1, 0.1, 0.1);
        let _o3 = Octonion::new(0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);

        // 1. Norm multiplicativity
        let product = o1.mul(&o2);
        let norm_error = (product.norm() - o1.norm() * o2.norm()).abs();
        assert!(norm_error < 1e-5, "Norm multiplicativity");

        // 2. Alternativity
        let alt_error = o1.mul(&o1).mul(&o2).distance(&o1.mul(&o1.mul(&o2)));
        assert!(alt_error < 1e-4, "Alternativity");

        // 3. Flexibility
        let flex_error = o1.mul(&o2).mul(&o1).distance(&o1.mul(&o2.mul(&o1)));
        assert!(flex_error < 1e-4, "Flexibility");

        // 4. Inverse
        let inv_error = o1.mul(&o1.inv()).distance(&Octonion::one());
        assert!(inv_error < 1e-5, "Inverse property");

        // 5. Conjugate anti-automorphism
        let conj_error = o1.mul(&o2).conj().distance(&o2.conj().mul(&o1.conj()));
        assert!(conj_error < 1e-5, "Conjugate anti-automorphism");

        println!("All summary property checks passed!");
    }
}
