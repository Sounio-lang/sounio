// beta.d - Beta distribution with variance-first API for epistemic computing
//
// Core epistemic primitive: represents probability as a full posterior
// distribution, not just a point estimate. Variance measures "ignorance."
//
// Now uses proper struct-based API with field access (compiler bug fixed).
//
// Future upgrades intentionally deferred:
// - Exact Beta quantiles (require incomplete beta function)
// - Special functions (gamma, beta, ln, exp)
// - Normal distribution conjugacy
// - MCMC sampling

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

// Beta distribution parameters
struct Beta {
    alpha: f64,
    beta: f64
}

// Confidence interval bounds
struct CI {
    lo: f64,
    hi: f64
}

// ============================================================================
// NUMERIC HELPERS
// ============================================================================

// Absolute value for f64
fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

// Clamp value to [0, 1] range
fn clamp01(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    if x > 1.0 { return 1.0 }
    return x
}

// Square root via Newton-Raphson (10 unrolled iterations)
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

// ============================================================================
// BETA DISTRIBUTION API
// ============================================================================

// Minimum epsilon to prevent invalid states (alpha, beta must be > 0)
fn beta_epsilon() -> f64 {
    return 0.0001
}

// Create a new Beta distribution with clamped parameters
fn beta_new(alpha: f64, b: f64) -> Beta {
    let eps = beta_epsilon()
    let safe_alpha = alpha
    let safe_beta = b
    if safe_alpha < eps {
        if safe_beta < eps {
            return Beta { alpha: eps, beta: eps }
        }
        return Beta { alpha: eps, beta: safe_beta }
    }
    if safe_beta < eps {
        return Beta { alpha: safe_alpha, beta: eps }
    }
    return Beta { alpha: safe_alpha, beta: safe_beta }
}

// Common priors
fn beta_uniform() -> Beta {
    return Beta { alpha: 1.0, beta: 1.0 }
}

fn beta_jeffreys() -> Beta {
    return Beta { alpha: 0.5, beta: 0.5 }
}

fn beta_weak() -> Beta {
    return Beta { alpha: 2.0, beta: 2.0 }
}

// Conjugate posterior update: Beta(a, b) + (successes, failures) -> Beta(a+s, b+f)
fn beta_posterior(prior: Beta, successes: i64, failures: i64) -> Beta {
    // Convert i64 to f64
    var sf = 0.0
    var count = 0
    if successes > 0 {
        while count < successes {
            sf = sf + 1.0
            count = count + 1
        }
    }

    var ff = 0.0
    count = 0
    if failures > 0 {
        while count < failures {
            ff = ff + 1.0
            count = count + 1
        }
    }

    return Beta { alpha: prior.alpha + sf, beta: prior.beta + ff }
}

// Mean of Beta distribution: E[X] = alpha / (alpha + beta)
fn beta_mean(b: Beta) -> f64 {
    return b.alpha / (b.alpha + b.beta)
}

// Variance: Var[X] = (alpha * beta) / ((alpha + beta)^2 * (alpha + beta + 1))
// This is the KEY epistemic quantity - measures "how much we don't know"
fn beta_variance(b: Beta) -> f64 {
    let n = b.alpha + b.beta
    return (b.alpha * b.beta) / (n * n * (n + 1.0))
}

// Standard deviation
fn beta_std(b: Beta) -> f64 {
    return sqrt_f64(beta_variance(b))
}

// Effective sample size: n = alpha + beta
fn beta_sample_size(b: Beta) -> f64 {
    return b.alpha + b.beta
}

// Mode: (alpha - 1) / (alpha + beta - 2) when alpha, beta > 1
fn beta_mode(b: Beta) -> f64 {
    if b.alpha <= 1.0 {
        return beta_mean(b)
    }
    if b.beta <= 1.0 {
        return beta_mean(b)
    }
    return (b.alpha - 1.0) / (b.alpha + b.beta - 2.0)
}

// Risk-adjusted mean: mean - lambda * std (conservative estimate)
fn beta_risk_adjusted_mean(b: Beta, lambda: f64) -> f64 {
    return beta_mean(b) - lambda * beta_std(b)
}

// Approximate CI using normal approximation around mean
// CI = [mean - z*std, mean + z*std], clamped to [0, 1]
// Common z values: 1.96 for 95%, 1.645 for 90%, 2.576 for 99%
// NOTE: This is APPROXIMATE, not exact Beta quantiles
fn beta_ci_normal(b: Beta, z: f64) -> CI {
    let m = beta_mean(b)
    let s = beta_std(b)
    let lo = clamp01(m - z * s)
    let hi = clamp01(m + z * s)
    return CI { lo: lo, hi: hi }
}

// ============================================================================
// TEST HARNESS
// ============================================================================

fn main() -> i32 {
    println("=== Beta Distribution Test ===")
    println("")

    let tol = 0.000001
    var all_passed = 1

    // Test 1: Construct prior Beta(1,1)
    println("Test 1: Uniform Prior Beta(1, 1)")
    let prior = beta_uniform()
    print("  alpha = ")
    println(prior.alpha)
    print("  beta = ")
    println(prior.beta)

    let prior_ok = abs_f64(prior.alpha - 1.0) < tol
    let prior_beta_ok = abs_f64(prior.beta - 1.0) < tol
    if prior_ok {
        if prior_beta_ok {
            println("  PASS")
        } else {
            println("  FAIL: beta mismatch")
            all_passed = 0
        }
    } else {
        println("  FAIL: alpha mismatch")
        all_passed = 0
    }
    println("")

    // Test 2: Update with successes=7, failures=3 => posterior Beta(8, 4)
    println("Test 2: Bayesian Update")
    println("  Prior: Beta(1, 1)")
    println("  Update: +7 successes, +3 failures")
    let posterior = beta_posterior(prior, 7, 3)
    print("  Posterior alpha = ")
    println(posterior.alpha)
    print("  Posterior beta = ")
    println(posterior.beta)

    let post_alpha_ok = abs_f64(posterior.alpha - 8.0) < tol
    let post_beta_ok = abs_f64(posterior.beta - 4.0) < tol
    if post_alpha_ok {
        if post_beta_ok {
            println("  PASS")
        } else {
            println("  FAIL: posterior beta mismatch")
            all_passed = 0
        }
    } else {
        println("  FAIL: posterior alpha mismatch")
        all_passed = 0
    }
    println("")

    // Test 3: Check mean == 8/12 within 1e-6
    println("Test 3: Mean Calculation")
    let post_mean = beta_mean(posterior)
    let expected_mean = 8.0 / 12.0
    print("  Mean = ")
    println(post_mean)
    print("  Expected = ")
    println(expected_mean)

    let mean_err = abs_f64(post_mean - expected_mean)
    print("  Error = ")
    println(mean_err)

    let mean_ok = mean_err < tol
    if mean_ok {
        println("  PASS")
    } else {
        println("  FAIL: mean incorrect")
        all_passed = 0
    }
    println("")

    // Test 4: Check variance == (8*4)/((12^2)*(13)) within 1e-6
    println("Test 4: Variance Calculation")
    let post_var = beta_variance(posterior)
    let expected_var = 32.0 / 1872.0
    print("  Variance = ")
    println(post_var)
    print("  Expected = ")
    println(expected_var)

    let var_err = abs_f64(post_var - expected_var)
    print("  Error = ")
    println(var_err)

    let var_ok = var_err < tol
    if var_ok {
        println("  PASS")
    } else {
        println("  FAIL: variance incorrect")
        all_passed = 0
    }
    println("")

    // Test 5: CI has lo <= hi and both within [0, 1]
    println("Test 5: Confidence Interval")
    let ci = beta_ci_normal(posterior, 1.96)
    print("  95% CI: [")
    print(ci.lo)
    print(", ")
    print(ci.hi)
    println("]")

    var ci_valid = 1
    if ci.lo > ci.hi {
        println("  FAIL: lo > hi")
        ci_valid = 0
    }
    if ci.lo < 0.0 {
        println("  FAIL: lo < 0")
        ci_valid = 0
    }
    if ci.hi > 1.0 {
        println("  FAIL: hi > 1")
        ci_valid = 0
    }

    if ci_valid == 1 {
        println("  PASS")
    } else {
        all_passed = 0
    }
    println("")

    // Test 6: Std dev
    println("Test 6: Standard Deviation")
    let post_std = beta_std(posterior)
    let expected_std = sqrt_f64(expected_var)
    print("  Std = ")
    println(post_std)
    print("  Expected = ")
    println(expected_std)

    let std_err = abs_f64(post_std - expected_std)
    let std_ok = std_err < tol
    if std_ok {
        println("  PASS")
    } else {
        println("  FAIL: std incorrect")
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
