//! Comprehensive Moufang Identities and Mathematical Properties Validation
//!
//! This test suite validates all fundamental algebraic properties of octonion multiplication,
//! including the Moufang identities that characterize octonion algebras.
//!
//! Mathematical Foundation:
//! - Graves (1843): "On algebraic triplets" - First definition of octonions
//! - Cayley (1845): Cayley-Dickson construction
//! - Baez (2002): "The Octonions" (Bull. Amer. Math. Soc. 39.2)

#[cfg(test)]
mod octonion_moufang_validation {
    /// Simple octonion struct for validation
    #[derive(Debug, Clone, Copy)]
    struct Octonion {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
        g: f64,
        h: f64,
    }

    impl Octonion {
        fn new(a: f64, b: f64, c: f64, d: f64, e: f64, f: f64, g: f64, h: f64) -> Self {
            Octonion {
                a,
                b,
                c,
                d,
                e,
                f,
                g,
                h,
            }
        }

        fn norm_sq(&self) -> f64 {
            self.a * self.a
                + self.b * self.b
                + self.c * self.c
                + self.d * self.d
                + self.e * self.e
                + self.f * self.f
                + self.g * self.g
                + self.h * self.h
        }

        fn norm(&self) -> f64 {
            self.norm_sq().sqrt()
        }

        fn conj(&self) -> Self {
            Octonion::new(
                self.a, -self.b, -self.c, -self.d, -self.e, -self.f, -self.g, -self.h,
            )
        }

        fn mul(&self, other: &Self) -> Self {
            let (a0, a1, a2, a3, a4, a5, a6, a7) = (
                self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h,
            );
            let (b0, b1, b2, b3, b4, b5, b6, b7) = (
                other.a, other.b, other.c, other.d, other.e, other.f, other.g, other.h,
            );

            let c0 = a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3 - a4 * b4 - a5 * b5 - a6 * b6 - a7 * b7;
            let c1 = a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2 + a4 * b5 - a5 * b4 - a6 * b7 + a7 * b6;
            let c2 = a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1 + a4 * b6 + a5 * b7 - a6 * b4 - a7 * b5;
            let c3 = a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0 + a4 * b7 - a5 * b6 + a6 * b5 - a7 * b4;
            let c4 = a0 * b4 - a1 * b5 - a2 * b6 - a3 * b7 + a4 * b0 + a5 * b1 + a6 * b2 + a7 * b3;
            let c5 = a0 * b5 + a1 * b4 - a2 * b7 + a3 * b6 - a4 * b1 + a5 * b0 - a6 * b3 + a7 * b2;
            let c6 = a0 * b6 + a1 * b7 + a2 * b4 - a3 * b5 - a4 * b2 + a5 * b3 + a6 * b0 - a7 * b1;
            let c7 = a0 * b7 - a1 * b6 + a2 * b5 + a3 * b4 - a4 * b3 - a5 * b2 + a6 * b1 + a7 * b0;

            Octonion::new(c0, c1, c2, c3, c4, c5, c6, c7)
        }

        fn distance(&self, other: &Self) -> f64 {
            let d = Octonion::new(
                self.a - other.a,
                self.b - other.b,
                self.c - other.c,
                self.d - other.d,
                self.e - other.e,
                self.f - other.f,
                self.g - other.g,
                self.h - other.h,
            );
            d.norm()
        }
    }

    // Deterministic pseudo-random octonion generator for reproducible tests
    fn random_octonion(seed: u64) -> Octonion {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((x >> 33) as f64 / (1u64 << 31) as f64) - 1.0
        };
        Octonion::new(
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
            next(),
        )
    }

    // ========================================================================
    // MOUFANG IDENTITIES - The defining properties of octonion algebras
    // ========================================================================

    #[test]
    fn test_moufang_first_identity() {
        // First Moufang identity: (x*y)*(z*x) = x*((y*z)*x)
        let mut max_error: f64 = 0.0;
        for seed in 0..50 {
            let x = random_octonion(seed * 3);
            let y = random_octonion(seed * 3 + 1);
            let z = random_octonion(seed * 3 + 2);

            let lhs = x.mul(&y).mul(&z.mul(&x));
            let rhs = x.mul(&y.mul(&z).mul(&x));
            let error = lhs.distance(&rhs);
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-10,
            "First Moufang identity failed, max error: {}",
            max_error
        );
    }

    #[test]
    fn test_moufang_second_identity() {
        // Second Moufang identity: z*(x*(y*x)) = ((z*x)*y)*x
        let mut max_error: f64 = 0.0;
        for seed in 0..50 {
            let x = random_octonion(seed * 3 + 100);
            let y = random_octonion(seed * 3 + 101);
            let z = random_octonion(seed * 3 + 102);

            let lhs = z.mul(&x.mul(&y.mul(&x)));
            let rhs = z.mul(&x).mul(&y).mul(&x);
            let error = lhs.distance(&rhs);
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-10,
            "Second Moufang identity failed, max error: {}",
            max_error
        );
    }

    #[test]
    fn test_moufang_third_identity_flexibility() {
        // Third Moufang (Flexibility): (x*y)*x = x*(y*x)
        let mut max_error: f64 = 0.0;
        for seed in 0..50 {
            let x = random_octonion(seed * 2 + 200);
            let y = random_octonion(seed * 2 + 201);

            let lhs = x.mul(&y).mul(&x);
            let rhs = x.mul(&y.mul(&x));
            let error = lhs.distance(&rhs);
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-10,
            "Flexibility identity failed, max error: {}",
            max_error
        );
    }

    // ========================================================================
    // NORM MULTIPLICATIVITY - Fundamental property of division algebras
    // ========================================================================

    #[test]
    fn test_norm_multiplicativity() {
        // Property: |x * y| = |x| * |y|
        let mut max_error: f64 = 0.0;
        for seed in 0..100 {
            let x = random_octonion(seed * 2 + 300);
            let y = random_octonion(seed * 2 + 301);

            let norm_product = x.mul(&y).norm();
            let product_norms = x.norm() * y.norm();
            let error = (norm_product - product_norms).abs() / product_norms.max(1e-10);
            max_error = max_error.max(error);
        }
        assert!(
            max_error < 1e-10,
            "Norm multiplicativity failed, max error: {}",
            max_error
        );
    }

    // ========================================================================
    // ALTERNATIVE PROPERTY - Weaker than associativity
    // ========================================================================

    #[test]
    fn test_alternative_property() {
        // Left alternative: (x*x)*y = x*(x*y)
        // Right alternative: (x*y)*y = x*(y*y)
        let mut max_error: f64 = 0.0;
        for seed in 0..50 {
            let x = random_octonion(seed * 2 + 400);
            let y = random_octonion(seed * 2 + 401);

            // Left alternative
            let lhs1 = x.mul(&x).mul(&y);
            let rhs1 = x.mul(&x.mul(&y));
            max_error = max_error.max(lhs1.distance(&rhs1));

            // Right alternative
            let lhs2 = x.mul(&y).mul(&y);
            let rhs2 = x.mul(&y.mul(&y));
            max_error = max_error.max(lhs2.distance(&rhs2));
        }
        assert!(
            max_error < 1e-10,
            "Alternative property failed, max error: {}",
            max_error
        );
    }

    // ========================================================================
    // NON-ASSOCIATIVITY - What makes octonions unique
    // ========================================================================

    #[test]
    fn test_non_associativity_demonstrated() {
        // Octonions are NOT associative: (x*y)*z ≠ x*(y*z) in general
        // Use basis elements i, j, l to demonstrate
        let i = Octonion::new(0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let j = Octonion::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let l = Octonion::new(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0);

        let left = i.mul(&j).mul(&l); // (i*j)*l
        let right = i.mul(&j.mul(&l)); // i*(j*l)

        let diff = left.distance(&right);
        assert!(
            diff > 1.9,
            "Non-associativity should be demonstrated: diff = {} (expected ~2.0)",
            diff
        );
    }

    // ========================================================================
    // CONJUGATE PROPERTIES
    // ========================================================================

    #[test]
    fn test_conjugate_anti_automorphism() {
        // conj(x*y) = conj(y)*conj(x) (reverses order)
        let mut max_error: f64 = 0.0;
        for seed in 0..50 {
            let x = random_octonion(seed * 2 + 500);
            let y = random_octonion(seed * 2 + 501);

            let lhs = x.mul(&y).conj();
            let rhs = y.conj().mul(&x.conj());
            max_error = max_error.max(lhs.distance(&rhs));
        }
        assert!(
            max_error < 1e-10,
            "Conjugate anti-automorphism failed, max error: {}",
            max_error
        );
    }
}
