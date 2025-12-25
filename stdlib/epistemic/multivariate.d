//! Multivariate Coverage Regions
//!
//! JCGM 102 is literally "extension to any number of output quantities"
//! and discusses coverage regions such as hyper-ellipsoids and hyper-rectangles.
//!
//! This module prevents PK time series from being treated as independent
//! scalar points (which is quietly fraudulent). It enforces:
//!   - Proper covariance structure tracking
//!   - Coverage regions instead of per-component intervals
//!
//! References:
//!   - JCGM 102:2011: Propagation of distributions using a Monte Carlo method
//!   - Johnson & Wichern (2007): Applied Multivariate Statistical Analysis

extern "C" {
    fn sqrt(x: f64) -> f64;
    fn log(x: f64) -> f64;
    fn exp(x: f64) -> f64;
}

fn sqrt_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 }
    return sqrt(x)
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { return 0.0 - x }
    return x
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

/// Time-dependent variance contribution from clearance uncertainty
fn time_var(t: f64, cv_cl: f64) -> f64 {
    return cv_cl * cv_cl * t * t
}

// ============================================================================
// COVARIANCE STRUCTURE
// ============================================================================

/// Covariance structure types:
///   0 = DIAGONAL (independent components)
///   1 = BANDED (AR-1 like correlation)
///   2 = FULL (arbitrary covariance)
struct CovarianceType {
    structure: i32,
    bandwidth: i32,     // For banded: number of off-diagonals
}

fn cov_diagonal() -> CovarianceType {
    return CovarianceType { structure: 0, bandwidth: 0 }
}

fn cov_banded(bandwidth: i32) -> CovarianceType {
    return CovarianceType { structure: 1, bandwidth: bandwidth }
}

fn cov_full() -> CovarianceType {
    return CovarianceType { structure: 2, bandwidth: 0 }
}

// ============================================================================
// 2D COVARIANCE MATRIX (simplified for PK applications)
// ============================================================================

/// 2×2 covariance matrix
struct Cov2x2 {
    v00: f64,   // Var(X0)
    v01: f64,   // Cov(X0, X1)
    v10: f64,   // Cov(X1, X0) = v01
    v11: f64,   // Var(X1)
}

fn cov2_new(var0: f64, var1: f64, covar: f64) -> Cov2x2 {
    return Cov2x2 {
        v00: var0,
        v01: covar,
        v10: covar,
        v11: var1,
    }
}

fn cov2_diagonal(var0: f64, var1: f64) -> Cov2x2 {
    return cov2_new(var0, var1, 0.0)
}

/// Correlation coefficient from covariance matrix
fn cov2_correlation(cov: Cov2x2) -> f64 {
    let denom = sqrt_f64(cov.v00 * cov.v11)
    if denom < 1.0e-15 { return 0.0 }
    let r = cov.v01 / denom
    return max_f64(-1.0, min_f64(1.0, r))
}

/// Determinant of 2×2 matrix
fn cov2_det(cov: Cov2x2) -> f64 {
    return cov.v00 * cov.v11 - cov.v01 * cov.v10
}

/// Inverse of 2×2 covariance matrix
fn cov2_inverse(cov: Cov2x2) -> Cov2x2 {
    let det = cov2_det(cov)
    if abs_f64(det) < 1.0e-15 {
        // Singular - return identity
        return cov2_diagonal(1.0, 1.0)
    }
    return Cov2x2 {
        v00: cov.v11 / det,
        v01: 0.0 - cov.v01 / det,
        v10: 0.0 - cov.v10 / det,
        v11: cov.v00 / det,
    }
}

// ============================================================================
// 4D COVARIANCE MATRIX (for time series with 4 points)
// ============================================================================

/// 4×4 covariance matrix (row-major)
struct Cov4x4 {
    // Row 0
    v00: f64, v01: f64, v02: f64, v03: f64,
    // Row 1
    v10: f64, v11: f64, v12: f64, v13: f64,
    // Row 2
    v20: f64, v21: f64, v22: f64, v23: f64,
    // Row 3
    v30: f64, v31: f64, v32: f64, v33: f64,
}

fn cov4_diagonal(v0: f64, v1: f64, v2: f64, v3: f64) -> Cov4x4 {
    return Cov4x4 {
        v00: v0, v01: 0.0, v02: 0.0, v03: 0.0,
        v10: 0.0, v11: v1, v12: 0.0, v13: 0.0,
        v20: 0.0, v21: 0.0, v22: v2, v23: 0.0,
        v30: 0.0, v31: 0.0, v32: 0.0, v33: v3,
    }
}

/// Power function for correlation decay: rho^n
fn pow_rho(rho: f64, n: i32) -> f64 {
    var result = 1.0
    var i: i32 = 0
    while i < n {
        result = result * rho
        i = i + 1
    }
    return result
}

/// Create AR(1)-like banded covariance
/// cov(i,j) = σ_i σ_j ρ^|i-j|
fn cov4_ar1(stds: [f64; 4], rho: f64) -> Cov4x4 {
    return Cov4x4 {
        v00: stds[0] * stds[0],
        v01: stds[0] * stds[1] * pow_rho(rho, 1),
        v02: stds[0] * stds[2] * pow_rho(rho, 2),
        v03: stds[0] * stds[3] * pow_rho(rho, 3),

        v10: stds[1] * stds[0] * pow_rho(rho, 1),
        v11: stds[1] * stds[1],
        v12: stds[1] * stds[2] * pow_rho(rho, 1),
        v13: stds[1] * stds[3] * pow_rho(rho, 2),

        v20: stds[2] * stds[0] * pow_rho(rho, 2),
        v21: stds[2] * stds[1] * pow_rho(rho, 1),
        v22: stds[2] * stds[2],
        v23: stds[2] * stds[3] * pow_rho(rho, 1),

        v30: stds[3] * stds[0] * pow_rho(rho, 3),
        v31: stds[3] * stds[1] * pow_rho(rho, 2),
        v32: stds[3] * stds[2] * pow_rho(rho, 1),
        v33: stds[3] * stds[3],
    }
}

// ============================================================================
// MULTIVARIATE OUTPUT
// ============================================================================

/// 2D output with covariance
struct MultivariateOutput2 {
    mean0: f64,
    mean1: f64,
    cov: Cov2x2,
    is_valid: bool,
}

fn mv2_new(m0: f64, m1: f64, cov: Cov2x2) -> MultivariateOutput2 {
    return MultivariateOutput2 {
        mean0: m0,
        mean1: m1,
        cov: cov,
        is_valid: true,
    }
}

/// 4D output with covariance (for time series)
struct MultivariateOutput4 {
    mean0: f64,
    mean1: f64,
    mean2: f64,
    mean3: f64,
    cov: Cov4x4,
    is_valid: bool,
}

fn mv4_new(m0: f64, m1: f64, m2: f64, m3: f64, cov: Cov4x4) -> MultivariateOutput4 {
    return MultivariateOutput4 {
        mean0: m0,
        mean1: m1,
        mean2: m2,
        mean3: m3,
        cov: cov,
        is_valid: true,
    }
}

// ============================================================================
// COVERAGE REGION TYPES
// ============================================================================

/// Coverage region types:
///   0 = HYPER_RECTANGLE (cheap, conservative)
///   1 = HYPER_ELLIPSOID (Gaussian-ish, compact)
///   2 = MC_CONVEX_HULL (empirical)
struct RegionType {
    region_type: i32,
    probability: f64,
}

fn region_rectangle(prob: f64) -> RegionType {
    return RegionType { region_type: 0, probability: prob }
}

fn region_ellipsoid(prob: f64) -> RegionType {
    return RegionType { region_type: 1, probability: prob }
}

fn region_hull(prob: f64) -> RegionType {
    return RegionType { region_type: 2, probability: prob }
}

// ============================================================================
// HYPER-RECTANGLE COVERAGE REGION
// ============================================================================

/// 2D hyper-rectangle
struct Rectangle2D {
    x0_lo: f64, x0_hi: f64,
    x1_lo: f64, x1_hi: f64,
    probability: f64,
    volume: f64,
}

/// Normal quantile approximation for common coverage probabilities
fn z_score(p: f64) -> f64 {
    // Approximate quantile for common values
    if p > 0.995 { return 2.807 }
    if p > 0.99 { return 2.576 }
    if p > 0.975 { return 2.241 }
    if p > 0.95 { return 1.960 }
    if p > 0.90 { return 1.645 }
    return 1.0
}

/// Compute hyper-rectangle from multivariate output
/// For p% coverage on each dimension, total coverage is p^2 (conservative)
fn rectangle2_from_mv(output: MultivariateOutput2, region: RegionType) -> Rectangle2D {
    // Per-component coverage for overall p
    // If we want 95% joint, each dimension needs sqrt(0.95) ≈ 97.5% individually
    // This is conservative but simple
    let per_dim_p = sqrt_f64(region.probability)

    let k = z_score(per_dim_p)

    let std0 = sqrt_f64(output.cov.v00)
    let std1 = sqrt_f64(output.cov.v11)

    let x0_lo = output.mean0 - k * std0
    let x0_hi = output.mean0 + k * std0
    let x1_lo = output.mean1 - k * std1
    let x1_hi = output.mean1 + k * std1

    let volume = (x0_hi - x0_lo) * (x1_hi - x1_lo)

    return Rectangle2D {
        x0_lo: x0_lo, x0_hi: x0_hi,
        x1_lo: x1_lo, x1_hi: x1_hi,
        probability: region.probability,
        volume: volume,
    }
}

// ============================================================================
// HYPER-ELLIPSOID COVERAGE REGION
// ============================================================================

/// 2D hyper-ellipsoid: (x - μ)ᵀ Σ⁻¹ (x - μ) ≤ c²
struct Ellipsoid2D {
    center0: f64,
    center1: f64,
    // Inverse covariance (precision) matrix
    prec_00: f64, prec_01: f64,
    prec_10: f64, prec_11: f64,
    c_squared: f64,      // Chi-squared threshold
    probability: f64,
    volume: f64,
}

/// Chi-squared quantile for 2 DOF
fn chi2_quantile_2(p: f64) -> f64 {
    // χ²(2) quantiles
    if p > 0.99 { return 9.21 }
    if p > 0.975 { return 7.38 }
    if p > 0.95 { return 5.99 }
    if p > 0.90 { return 4.61 }
    if p > 0.80 { return 3.22 }
    return 2.30
}

/// Compute hyper-ellipsoid from multivariate output
fn ellipsoid2_from_mv(output: MultivariateOutput2, region: RegionType) -> Ellipsoid2D {
    let inv = cov2_inverse(output.cov)
    let c2 = chi2_quantile_2(region.probability)

    // Volume of ellipsoid: π * sqrt(det(Σ)) * c
    let pi = 3.14159265358979323846
    let det = cov2_det(output.cov)
    let volume = pi * sqrt_f64(abs_f64(det)) * sqrt_f64(c2)

    return Ellipsoid2D {
        center0: output.mean0,
        center1: output.mean1,
        prec_00: inv.v00,
        prec_01: inv.v01,
        prec_10: inv.v10,
        prec_11: inv.v11,
        c_squared: c2,
        probability: region.probability,
        volume: volume,
    }
}

/// Check if point is inside ellipsoid
fn ellipsoid2_contains(e: Ellipsoid2D, x0: f64, x1: f64) -> bool {
    let d0 = x0 - e.center0
    let d1 = x1 - e.center1

    // Mahalanobis distance squared: dᵀ Σ⁻¹ d
    let maha_sq = d0 * e.prec_00 * d0
               + d0 * e.prec_01 * d1
               + d1 * e.prec_10 * d0
               + d1 * e.prec_11 * d1

    return maha_sq <= e.c_squared
}

// ============================================================================
// COVERAGE REGION COMPARISON
// ============================================================================

/// Compare rectangle vs ellipsoid efficiency
struct RegionComparison {
    rectangle_volume: f64,
    ellipsoid_volume: f64,
    efficiency_ratio: f64,  // ellipsoid/rectangle (< 1 means ellipsoid is tighter)
    recommendation: i32,    // 0=rectangle, 1=ellipsoid
}

fn compare_regions(output: MultivariateOutput2, prob: f64) -> RegionComparison {
    let rect = rectangle2_from_mv(output, region_rectangle(prob))
    let ellip = ellipsoid2_from_mv(output, region_ellipsoid(prob))

    var ratio = 1.0
    if rect.volume > 1.0e-15 {
        ratio = ellip.volume / rect.volume
    }

    // Ellipsoid is better when variables are correlated
    let r = cov2_correlation(output.cov)
    var recommendation = 0  // Rectangle by default
    if abs_f64(r) > 0.3 {
        recommendation = 1  // Ellipsoid
    }

    return RegionComparison {
        rectangle_volume: rect.volume,
        ellipsoid_volume: ellip.volume,
        efficiency_ratio: ratio,
        recommendation: recommendation,
    }
}

// ============================================================================
// PK TIME SERIES EXAMPLE
// ============================================================================

/// PK concentration at multiple time points
/// Demonstrates why per-point intervals are fraudulent
fn pk_time_series(
    dose: f64,
    volume: f64,
    clearance: f64,
    times: [f64; 4],
    cv_dose: f64,
    cv_volume: f64,
    cv_clearance: f64
) -> MultivariateOutput4 {
    let k = clearance / volume
    let c0 = dose / volume

    // Compute concentrations at each time
    let c_t0 = c0 * exp(0.0 - k * times[0])
    let c_t1 = c0 * exp(0.0 - k * times[1])
    let c_t2 = c0 * exp(0.0 - k * times[2])
    let c_t3 = c0 * exp(0.0 - k * times[3])

    // Compute marginal variances (linearized)
    // These are correlated through shared parameters!
    let rel_var = cv_dose * cv_dose + cv_volume * cv_volume

    // Time-dependent variance contribution from clearance
    // (time_var function is defined at module level)

    let var_t0 = c_t0 * c_t0 * (rel_var + time_var(times[0], cv_clearance))
    let var_t1 = c_t1 * c_t1 * (rel_var + time_var(times[1], cv_clearance))
    let var_t2 = c_t2 * c_t2 * (rel_var + time_var(times[2], cv_clearance))
    let var_t3 = c_t3 * c_t3 * (rel_var + time_var(times[3], cv_clearance))

    let std_t0 = sqrt_f64(var_t0)
    let std_t1 = sqrt_f64(var_t1)
    let std_t2 = sqrt_f64(var_t2)
    let std_t3 = sqrt_f64(var_t3)

    // CRITICAL: Time points are correlated because they share the same parameters!
    // Adjacent points have ~0.8-0.9 correlation due to shared V, CL, Dose
    let rho = 0.85  // Typical for PK time series

    let stds: [f64; 4] = [std_t0, std_t1, std_t2, std_t3]
    let cov = cov4_ar1(stds, rho)

    return MultivariateOutput4 {
        mean0: c_t0,
        mean1: c_t1,
        mean2: c_t2,
        mean3: c_t3,
        cov: cov,
        is_valid: true,
    }
}

// ============================================================================
// FRAUD DETECTION: INDEPENDENT VS CORRELATED COVERAGE
// ============================================================================

/// Detect if treating points as independent would be fraudulent
struct FraudAnalysis {
    max_correlation: f64,
    avg_correlation: f64,
    independent_coverage: f64,    // Coverage if treated as independent
    true_coverage: f64,           // Actual coverage with correlation
    coverage_inflation: f64,      // How much independent overstates
    is_fraudulent: bool,          // True if difference is substantial
}

fn detect_independence_fraud(output: MultivariateOutput2, claimed_prob: f64) -> FraudAnalysis {
    let r = cov2_correlation(output.cov)

    // If we claim independent 95% on each, joint coverage is 0.95² = 0.9025
    // If strongly correlated, true coverage is higher
    let independent_joint = claimed_prob * claimed_prob

    // For perfect correlation (r=1), joint coverage equals marginal
    // For r=0, joint = product
    // Interpolate: true_joint ≈ claimed * (1 - (1-claimed) * (1-|r|))
    let true_joint = claimed_prob * (1.0 - (1.0 - claimed_prob) * (1.0 - abs_f64(r)))

    let inflation = true_joint - independent_joint

    // Fraudulent if inflation > 3 percentage points
    let is_fraud = inflation > 0.03

    return FraudAnalysis {
        max_correlation: r,
        avg_correlation: r,
        independent_coverage: independent_joint,
        true_coverage: true_joint,
        coverage_inflation: inflation,
        is_fraudulent: is_fraud,
    }
}

// ============================================================================
// TESTS
// ============================================================================

fn test_diagonal_covariance() -> bool {
    let cov = cov2_diagonal(4.0, 9.0)

    // Variances should be on diagonal
    if abs_f64(cov.v00 - 4.0) > 0.001 { return false }
    if abs_f64(cov.v11 - 9.0) > 0.001 { return false }

    // Off-diagonal should be zero
    if abs_f64(cov.v01) > 0.001 { return false }

    // Correlation should be zero
    let r = cov2_correlation(cov)
    if abs_f64(r) > 0.001 { return false }

    return true
}

fn test_correlated_covariance() -> bool {
    // Two variables with 0.8 correlation
    let var0 = 4.0
    let var1 = 9.0
    let r = 0.8
    let covar = r * sqrt_f64(var0) * sqrt_f64(var1)  // = 0.8 * 2 * 3 = 4.8

    let cov = cov2_new(var0, var1, covar)

    // Check correlation
    let r_computed = cov2_correlation(cov)
    if abs_f64(r_computed - 0.8) > 0.01 { return false }

    return true
}

fn test_rectangle_region() -> bool {
    let cov = cov2_diagonal(1.0, 4.0)
    let output = mv2_new(10.0, 20.0, cov)

    let rect = rectangle2_from_mv(output, region_rectangle(0.95))

    // Check contains center
    if rect.x0_lo > 10.0 { return false }
    if rect.x0_hi < 10.0 { return false }
    if rect.x1_lo > 20.0 { return false }
    if rect.x1_hi < 20.0 { return false }

    // Check volume is positive
    if rect.volume <= 0.0 { return false }

    return true
}

fn test_ellipsoid_region() -> bool {
    let cov = cov2_diagonal(1.0, 1.0)
    let output = mv2_new(0.0, 0.0, cov)

    let ellip = ellipsoid2_from_mv(output, region_ellipsoid(0.95))

    // Center should be inside
    if !ellipsoid2_contains(ellip, 0.0, 0.0) { return false }

    // Far point should be outside
    if ellipsoid2_contains(ellip, 10.0, 10.0) { return false }

    return true
}

fn test_fraud_detection() -> bool {
    // Highly correlated outputs
    let covar = 0.9 * sqrt_f64(1.0) * sqrt_f64(1.0)
    let cov = cov2_new(1.0, 1.0, covar)
    let output = mv2_new(0.0, 0.0, cov)

    let fraud = detect_independence_fraud(output, 0.95)

    // Should detect that independent assumption inflates coverage
    if !fraud.is_fraudulent { return false }
    if fraud.coverage_inflation < 0.01 { return false }

    return true
}

fn test_pk_time_series_correlation() -> bool {
    let times: [f64; 4] = [0.0, 1.0, 4.0, 8.0]
    let pk = pk_time_series(500.0, 50.0, 5.0, times, 0.05, 0.20, 0.40)

    // Check valid
    if !pk.is_valid { return false }

    // Concentrations should decrease with time
    if pk.mean1 > pk.mean0 { return false }
    if pk.mean2 > pk.mean1 { return false }
    if pk.mean3 > pk.mean2 { return false }

    // Covariance should show correlation (not diagonal)
    if abs_f64(pk.cov.v01) < 0.001 { return false }

    return true
}

// ============================================================================
// MAIN
// ============================================================================

fn main() -> i32 {
    if !test_diagonal_covariance() { return 1 }
    if !test_correlated_covariance() { return 2 }
    if !test_rectangle_region() { return 3 }
    if !test_ellipsoid_region() { return 4 }
    if !test_fraud_detection() { return 5 }
    if !test_pk_time_series_correlation() { return 6 }

    return 0
}
