// normal.d - Normal (Gaussian) distribution for epistemic computing
//
// The Normal distribution is fundamental for:
// - Modeling measurement uncertainty
// - Central Limit Theorem applications
// - Bayesian inference with continuous data
//
// Implementation uses struct-based API now that field access is fixed.

// ============================================================================
// TYPES
// ============================================================================

// Normal distribution parameters
struct Normal {
    mu: f64,      // Mean
    sigma: f64    // Standard deviation (must be > 0)
}

// Confidence interval bounds
struct NormalCI {
    lo: f64,
    hi: f64
}

// ============================================================================
// NUMERIC HELPERS
// ============================================================================

// Absolute value
fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

// Clamp to range
fn clamp_f64(x: f64, lo: f64, hi: f64) -> f64 {
    if x < lo { return lo }
    if x > hi { return hi }
    return x
}

// Square root via Newton-Raphson (10 iterations)
fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    let mut y = x
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    y = 0.5 * (y + x / y)
    return y
}

// Exponential via Taylor series with range reduction
fn exp_approx(x: f64) -> f64 {
    // Range reduction: exp(x) = exp(x/2)^2 for large values
    if x > 20.0 { return exp_approx(x / 2.0) * exp_approx(x / 2.0) }
    // For negative values: exp(-x) = 1/exp(x)
    if x < 0.0 - 20.0 { return 1.0 / exp_approx(0.0 - x) }
    if x < 0.0 { return 1.0 / exp_approx(0.0 - x) }

    // Taylor series: 1 + x + x^2/2! + x^3/3! + ...
    let mut sum = 1.0
    let mut term = 1.0
    term = term * x / 1.0
    sum = sum + term
    term = term * x / 2.0
    sum = sum + term
    term = term * x / 3.0
    sum = sum + term
    term = term * x / 4.0
    sum = sum + term
    term = term * x / 5.0
    sum = sum + term
    term = term * x / 6.0
    sum = sum + term
    term = term * x / 7.0
    sum = sum + term
    term = term * x / 8.0
    sum = sum + term
    term = term * x / 9.0
    sum = sum + term
    term = term * x / 10.0
    sum = sum + term
    term = term * x / 11.0
    sum = sum + term
    term = term * x / 12.0
    sum = sum + term
    term = term * x / 13.0
    sum = sum + term
    term = term * x / 14.0
    sum = sum + term
    term = term * x / 15.0
    sum = sum + term
    return sum
}

// Pi constant (for PDF normalization)
fn pi() -> f64 {
    return 3.14159265358979323846
}

// ============================================================================
// NORMAL DISTRIBUTION API
// ============================================================================

// Minimum sigma to prevent division by zero
fn normal_epsilon() -> f64 {
    return 0.0000001
}

// Create a new Normal distribution with validated parameters
fn normal_new(mu: f64, sigma: f64) -> Normal {
    let eps = normal_epsilon()
    let safe_sigma = sigma
    if safe_sigma < eps {
        return Normal { mu: mu, sigma: eps }
    }
    return Normal { mu: mu, sigma: safe_sigma }
}

// Standard normal distribution N(0, 1)
fn normal_standard() -> Normal {
    return Normal { mu: 0.0, sigma: 1.0 }
}

// Mean of Normal distribution
fn normal_mean(n: Normal) -> f64 {
    return n.mu
}

// Variance of Normal distribution: Var = σ²
fn normal_variance(n: Normal) -> f64 {
    return n.sigma * n.sigma
}

// Standard deviation of Normal distribution
fn normal_std(n: Normal) -> f64 {
    return n.sigma
}

// Probability density function (PDF)
// f(x) = (1 / (σ√(2π))) * exp(-(x-μ)²/(2σ²))
fn normal_pdf(n: Normal, x: f64) -> f64 {
    let diff = x - n.mu
    let exponent = 0.0 - (diff * diff) / (2.0 * n.sigma * n.sigma)
    let normalization = 1.0 / (n.sigma * sqrt_f64(2.0 * pi()))
    return normalization * exp_approx(exponent)
}

// Z-score: standardize a value
fn normal_zscore(n: Normal, x: f64) -> f64 {
    return (x - n.mu) / n.sigma
}

// Inverse z-score: convert z back to x
fn normal_from_zscore(n: Normal, z: f64) -> f64 {
    return n.mu + z * n.sigma
}

// Confidence interval around mean
// CI = [μ - z*σ, μ + z*σ]
// Common z values: 1.96 for 95% CI, 1.645 for 90% CI, 2.576 for 99% CI
fn normal_ci(n: Normal, z: f64) -> NormalCI {
    let lo = n.mu - z * n.sigma
    let hi = n.mu + z * n.sigma
    return NormalCI { lo: lo, hi: hi }
}

// Risk-adjusted mean: μ - λ*σ (conservative estimate)
fn normal_risk_adjusted_mean(n: Normal, lambda: f64) -> f64 {
    return n.mu - lambda * n.sigma
}

// Shift the distribution by a constant (μ + c, σ unchanged)
fn normal_shift(n: Normal, c: f64) -> Normal {
    return Normal { mu: n.mu + c, sigma: n.sigma }
}

// Scale the distribution by a constant (c*μ, |c|*σ)
fn normal_scale(n: Normal, c: f64) -> Normal {
    return Normal { mu: c * n.mu, sigma: abs_f64(c) * n.sigma }
}

// Sum of two independent Normal distributions
// X ~ N(μ₁, σ₁²), Y ~ N(μ₂, σ₂²) => X + Y ~ N(μ₁+μ₂, σ₁²+σ₂²)
fn normal_sum(n1: Normal, n2: Normal) -> Normal {
    let new_mu = n1.mu + n2.mu
    let new_variance = n1.sigma * n1.sigma + n2.sigma * n2.sigma
    let new_sigma = sqrt_f64(new_variance)
    return Normal { mu: new_mu, sigma: new_sigma }
}

// ============================================================================
// TEST HARNESS
// ============================================================================

fn main() -> i32 {
    println("=== Normal Distribution Test ===")
    println("")

    let tol = 0.000001
    var all_passed = 1

    // Test 1: Create standard normal
    println("Test 1: Standard Normal N(0, 1)")
    let std_norm = normal_standard()
    print("  mu = ")
    println(std_norm.mu)
    print("  sigma = ")
    println(std_norm.sigma)

    if abs_f64(std_norm.mu - 0.0) < tol {
        if abs_f64(std_norm.sigma - 1.0) < tol {
            println("  PASS")
        } else {
            println("  FAIL: sigma mismatch")
            all_passed = 0
        }
    } else {
        println("  FAIL: mu mismatch")
        all_passed = 0
    }
    println("")

    // Test 2: Create custom normal
    println("Test 2: Custom Normal N(100, 15)")
    let iq = normal_new(100.0, 15.0)
    let iq_mean = normal_mean(iq)
    let iq_var = normal_variance(iq)
    let iq_std = normal_std(iq)

    print("  Mean = ")
    println(iq_mean)
    print("  Variance = ")
    println(iq_var)
    print("  Std = ")
    println(iq_std)

    if abs_f64(iq_mean - 100.0) < tol {
        if abs_f64(iq_var - 225.0) < tol {
            if abs_f64(iq_std - 15.0) < tol {
                println("  PASS")
            } else {
                println("  FAIL: std mismatch")
                all_passed = 0
            }
        } else {
            println("  FAIL: variance mismatch")
            all_passed = 0
        }
    } else {
        println("  FAIL: mean mismatch")
        all_passed = 0
    }
    println("")

    // Test 3: PDF at mean should be maximum
    println("Test 3: PDF at Mean (Standard Normal)")
    let pdf_at_mean = normal_pdf(std_norm, 0.0)
    let pdf_at_1 = normal_pdf(std_norm, 1.0)
    let pdf_at_2 = normal_pdf(std_norm, 2.0)

    print("  PDF(0) = ")
    println(pdf_at_mean)
    print("  PDF(1) = ")
    println(pdf_at_1)
    print("  PDF(2) = ")
    println(pdf_at_2)

    // PDF at mean for N(0,1) should be 1/sqrt(2*pi) ≈ 0.3989
    let expected_pdf_mean = 0.3989422804

    if abs_f64(pdf_at_mean - expected_pdf_mean) < 0.0001 {
        if pdf_at_mean > pdf_at_1 {
            if pdf_at_1 > pdf_at_2 {
                println("  PASS (PDF peaks at mean, decreases away)")
            } else {
                println("  FAIL: PDF should decrease further from mean")
                all_passed = 0
            }
        } else {
            println("  FAIL: PDF should be maximum at mean")
            all_passed = 0
        }
    } else {
        println("  FAIL: PDF(0) incorrect")
        all_passed = 0
    }
    println("")

    // Test 4: Z-score conversion
    println("Test 4: Z-score Conversion")
    let iq2 = normal_new(100.0, 15.0)
    let test4_x = 130.0
    let test4_z = normal_zscore(iq2, test4_x)
    let test4_x_back = normal_from_zscore(iq2, test4_z)

    print("  x = ")
    println(test4_x)
    print("  z-score = ")
    println(test4_z)
    print("  back to x = ")
    println(test4_x_back)

    // z = (130 - 100) / 15 = 2.0
    // Note: Using bool variables to work around nested if + function call bug
    let z_ok = abs_f64(test4_z - 2.0) < tol
    let x_back_ok = abs_f64(test4_x_back - test4_x) < tol

    if z_ok {
        if x_back_ok {
            println("  PASS")
        } else {
            println("  FAIL: inverse z-score incorrect")
            all_passed = 0
        }
    } else {
        println("  FAIL: z-score incorrect")
        all_passed = 0
    }
    println("")

    // Test 5: Confidence interval
    println("Test 5: 95% Confidence Interval")
    let ci = normal_ci(iq, 1.96)
    print("  95% CI: [")
    print(ci.lo)
    print(", ")
    print(ci.hi)
    println("]")

    // CI should be [100 - 1.96*15, 100 + 1.96*15] = [70.6, 129.4]
    let expected_lo = 100.0 - 1.96 * 15.0
    let expected_hi = 100.0 + 1.96 * 15.0

    if abs_f64(ci.lo - expected_lo) < tol {
        if abs_f64(ci.hi - expected_hi) < tol {
            if ci.lo < ci.hi {
                println("  PASS")
            } else {
                println("  FAIL: lo should be < hi")
                all_passed = 0
            }
        } else {
            println("  FAIL: hi incorrect")
            all_passed = 0
        }
    } else {
        println("  FAIL: lo incorrect")
        all_passed = 0
    }
    println("")

    // Test 6: Sum of independent normals
    println("Test 6: Sum of Independent Normals")
    let n1 = normal_new(10.0, 3.0)
    let n2 = normal_new(20.0, 4.0)
    let sum = normal_sum(n1, n2)

    print("  N(10, 3²) + N(20, 4²) = N(")
    print(sum.mu)
    print(", ")
    print(sum.sigma)
    println(")")

    // Sum should be N(30, 5) since sqrt(9 + 16) = 5
    if abs_f64(sum.mu - 30.0) < tol {
        if abs_f64(sum.sigma - 5.0) < tol {
            println("  PASS")
        } else {
            println("  FAIL: sigma incorrect (expected 5)")
            all_passed = 0
        }
    } else {
        println("  FAIL: mu incorrect (expected 30)")
        all_passed = 0
    }
    println("")

    // Summary
    println("=================================")
    if all_passed == 1 {
        println("TEST PASSED")
        return 0
    } else {
        println("TEST FAILED")
        return 1
    }
}
