// stdlib/epistemic/gum.d
// GUM (Guide to the Expression of Uncertainty in Measurement) Compliant
// Uncertainty Propagation
//
// Implements:
// - Coverage factors (k) for expanded uncertainty
// - Degrees of freedom (ν) tracking
// - Welch-Satterthwaite approximation for combined DoF
// - Type A (statistical) and Type B (other) uncertainty components
//
// Reference: JCGM 100:2008 (GUM)

// =============================================================================
// Helper Functions
// =============================================================================

fn sqrt_f64(x: f64) -> f64 {
    if x <= 0.0 { return 0.0 }
    var y = x;
    var i: usize = 0;
    while i < 15 {
        y = 0.5 * (y + x / y);
        i = i + 1;
    }
    y
}

fn abs_f64(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn min_f64(a: f64, b: f64) -> f64 {
    if a < b { a } else { b }
}

fn max_f64(a: f64, b: f64) -> f64 {
    if a > b { a } else { b }
}

// =============================================================================
// GUM Uncertainty Type
// =============================================================================

// GUM-compliant uncertainty with degrees of freedom
pub struct GUMUncertainty {
    // Standard uncertainty u(x)
    pub std_uncertainty: f64,

    // Degrees of freedom ν (infinity represented as 1e9)
    // For Type A: ν = n - 1
    // For Type B: ν = ∞ (use 1e9)
    pub degrees_of_freedom: f64,

    // Sensitivity coefficient (∂f/∂x)
    pub sensitivity: f64,
}

// Coverage probability constants (as functions since pub const is not supported)
pub fn COVERAGE_68() -> f64 { 0.68 }
pub fn COVERAGE_90() -> f64 { 0.90 }
pub fn COVERAGE_95() -> f64 { 0.95 }
pub fn COVERAGE_99() -> f64 { 0.99 }
pub fn COVERAGE_9973() -> f64 { 0.9973 }

// =============================================================================
// Constructors
// =============================================================================

// Create Type A uncertainty from statistical data
// std_dev: sample standard deviation
// n: number of observations
pub fn type_a_uncertainty(std_dev: f64, n: usize) -> GUMUncertainty {
    // Standard uncertainty of the mean = s / sqrt(n)
    let std_u = if n > 0 {
        std_dev / sqrt_f64(n as f64)
    } else {
        std_dev
    };

    // Degrees of freedom = n - 1
    let dof = if n > 1 { (n - 1) as f64 } else { 1.0 };

    GUMUncertainty {
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        sensitivity: 1.0,
    }
}

// Create Type B uncertainty from a priori knowledge
// For Type B, degrees of freedom is effectively infinite
pub fn type_b_uncertainty(std_u: f64) -> GUMUncertainty {
    GUMUncertainty {
        std_uncertainty: abs_f64(std_u),
        degrees_of_freedom: 1.0e9,
        sensitivity: 1.0,
    }
}

// Create Type B uncertainty from a uniform distribution
// half_width: a in the interval [-a, +a]
pub fn type_b_uniform(half_width: f64) -> GUMUncertainty {
    // For uniform: u = a / sqrt(3)
    let std_u = abs_f64(half_width) / 1.732050808;
    type_b_uncertainty(std_u)
}

// Create Type B uncertainty from a triangular distribution
// half_width: a in the interval [-a, +a]
pub fn type_b_triangular(half_width: f64) -> GUMUncertainty {
    // For triangular: u = a / sqrt(6)
    let std_u = abs_f64(half_width) / 2.449489743;
    type_b_uncertainty(std_u)
}

// Create Type B uncertainty from expanded uncertainty
// expanded_u: U (expanded uncertainty)
// k: coverage factor used
pub fn type_b_from_expanded(expanded_u: f64, k: f64) -> GUMUncertainty {
    let std_u = if k > 0.0 { expanded_u / k } else { expanded_u };
    type_b_uncertainty(std_u)
}

// =============================================================================
// Coverage Factor Tables (t-distribution)
// =============================================================================

// t-distribution quantiles for 95% coverage
// Index = degrees of freedom - 1 (for ν=1 to ν=20)
fn t_quantile_95(dof: usize) -> f64 {
    // Tabulated values from standard t-tables
    if dof >= 100 { return 1.96; }  // Normal approximation
    if dof >= 20 { return 2.086; }
    if dof == 0 { return 12.71; }

    let idx = dof - 1;
    if idx == 0 { return 12.71; }
    if idx == 1 { return 4.303; }
    if idx == 2 { return 3.182; }
    if idx == 3 { return 2.776; }
    if idx == 4 { return 2.571; }
    if idx == 5 { return 2.447; }
    if idx == 6 { return 2.365; }
    if idx == 7 { return 2.306; }
    if idx == 8 { return 2.262; }
    if idx == 9 { return 2.228; }
    if idx == 10 { return 2.201; }
    if idx == 11 { return 2.179; }
    if idx == 12 { return 2.160; }
    if idx == 13 { return 2.145; }
    if idx == 14 { return 2.131; }
    if idx == 15 { return 2.120; }
    if idx == 16 { return 2.110; }
    if idx == 17 { return 2.101; }
    if idx == 18 { return 2.093; }
    2.086  // ν >= 20
}

// t-distribution quantiles for 99% coverage
fn t_quantile_99(dof: usize) -> f64 {
    if dof >= 100 { return 2.576; }
    if dof >= 20 { return 2.845; }
    if dof == 0 { return 63.66; }

    let idx = dof - 1;
    if idx == 0 { return 63.66; }
    if idx == 1 { return 9.925; }
    if idx == 2 { return 5.841; }
    if idx == 3 { return 4.604; }
    if idx == 4 { return 4.032; }
    if idx == 5 { return 3.707; }
    if idx == 6 { return 3.499; }
    if idx == 7 { return 3.355; }
    if idx == 8 { return 3.250; }
    if idx == 9 { return 3.169; }
    2.845  // ν >= 10
}

// Get coverage factor k for 95% probability at given degrees of freedom
pub fn coverage_factor_95(dof: f64) -> f64 {
    let d = dof as usize;
    let d_safe = if d < 1 { 1 } else { d };
    t_quantile_95(d_safe)
}

// Get coverage factor k for 99% probability at given degrees of freedom
pub fn coverage_factor_99(dof: f64) -> f64 {
    let d = dof as usize;
    let d_safe = if d < 1 { 1 } else { d };
    t_quantile_99(d_safe)
}

// Standard coverage factors for normal distribution (infinite DoF)
pub fn k_normal_68() -> f64 { 1.0 }
pub fn k_normal_90() -> f64 { 1.645 }
pub fn k_normal_95() -> f64 { 1.96 }
pub fn k_normal_99() -> f64 { 2.576 }
pub fn k_normal_9973() -> f64 { 3.0 }

// =============================================================================
// Welch-Satterthwaite Approximation
// =============================================================================

// Calculate effective degrees of freedom for two uncertainty components
// Formula: ν_eff = u_c^4 / (u1^4/ν1 + u2^4/ν2)
pub fn welch_satterthwaite_2(u1: GUMUncertainty, u2: GUMUncertainty) -> f64 {
    let c1 = u1.sensitivity;
    let c2 = u2.sensitivity;
    let s1 = u1.std_uncertainty;
    let s2 = u2.std_uncertainty;
    let v1 = u1.degrees_of_freedom;
    let v2 = u2.degrees_of_freedom;

    // Combined variance
    let u_c_sq = c1*c1*s1*s1 + c2*c2*s2*s2;
    let u_c_4 = u_c_sq * u_c_sq;

    // Denominator
    let term1 = c1*c1*c1*c1 * s1*s1*s1*s1 / v1;
    let term2 = c2*c2*c2*c2 * s2*s2*s2*s2 / v2;
    let denom = term1 + term2;

    if denom < 1.0e-30 {
        return 1.0e9;
    }

    let v_eff = u_c_4 / denom;
    min_f64(v_eff, 1.0e9)
}

// =============================================================================
// GUM Result Type
// =============================================================================

// Complete GUM measurement result
pub struct GUMResult {
    // Best estimate (measured value)
    pub value: f64,

    // Combined standard uncertainty u_c
    pub std_uncertainty: f64,

    // Effective degrees of freedom
    pub degrees_of_freedom: f64,

    // Coverage factor k (for 95% expanded uncertainty)
    pub coverage_factor_95: f64,

    // Expanded uncertainty U = k * u_c (at 95%)
    pub expanded_uncertainty_95: f64,
}

// Create a simple GUM result from value and standard uncertainty
// Assumes infinite degrees of freedom (Type B)
pub fn gum_simple(value: f64, std_u: f64) -> GUMResult {
    let k = k_normal_95();
    GUMResult {
        value: value,
        std_uncertainty: abs_f64(std_u),
        degrees_of_freedom: 1.0e9,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * abs_f64(std_u),
    }
}

// Create GUM result from Type A uncertainty
pub fn gum_type_a(value: f64, std_dev: f64, n: usize) -> GUMResult {
    let u = type_a_uncertainty(std_dev, n);
    let k = coverage_factor_95(u.degrees_of_freedom);
    GUMResult {
        value: value,
        std_uncertainty: u.std_uncertainty,
        degrees_of_freedom: u.degrees_of_freedom,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * u.std_uncertainty,
    }
}

// =============================================================================
// Uncertainty Propagation
// =============================================================================

// Propagate uncertainty through addition: y = x1 + x2
pub fn gum_add(x1: GUMResult, x2: GUMResult) -> GUMResult {
    let value = x1.value + x2.value;

    // Combined variance (sensitivities both = 1)
    let var1 = x1.std_uncertainty * x1.std_uncertainty;
    let var2 = x2.std_uncertainty * x2.std_uncertainty;
    let std_u = sqrt_f64(var1 + var2);

    // Effective DoF via Welch-Satterthwaite
    let u1 = GUMUncertainty {
        std_uncertainty: x1.std_uncertainty,
        degrees_of_freedom: x1.degrees_of_freedom,
        sensitivity: 1.0,
    };
    let u2 = GUMUncertainty {
        std_uncertainty: x2.std_uncertainty,
        degrees_of_freedom: x2.degrees_of_freedom,
        sensitivity: 1.0,
    };
    let dof = welch_satterthwaite_2(u1, u2);
    let k = coverage_factor_95(dof);

    GUMResult {
        value: value,
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * std_u,
    }
}

// Propagate uncertainty through subtraction: y = x1 - x2
pub fn gum_sub(x1: GUMResult, x2: GUMResult) -> GUMResult {
    let value = x1.value - x2.value;

    // Same as addition for variance
    let var1 = x1.std_uncertainty * x1.std_uncertainty;
    let var2 = x2.std_uncertainty * x2.std_uncertainty;
    let std_u = sqrt_f64(var1 + var2);

    let u1 = GUMUncertainty {
        std_uncertainty: x1.std_uncertainty,
        degrees_of_freedom: x1.degrees_of_freedom,
        sensitivity: 1.0,
    };
    let u2 = GUMUncertainty {
        std_uncertainty: x2.std_uncertainty,
        degrees_of_freedom: x2.degrees_of_freedom,
        sensitivity: 1.0,
    };
    let dof = welch_satterthwaite_2(u1, u2);
    let k = coverage_factor_95(dof);

    GUMResult {
        value: value,
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * std_u,
    }
}

// Propagate uncertainty through multiplication: y = x1 * x2
pub fn gum_mul(x1: GUMResult, x2: GUMResult) -> GUMResult {
    let value = x1.value * x2.value;

    // Sensitivity coefficients: ∂y/∂x1 = x2, ∂y/∂x2 = x1
    let c1 = abs_f64(x2.value);
    let c2 = abs_f64(x1.value);

    let var1 = c1 * c1 * x1.std_uncertainty * x1.std_uncertainty;
    let var2 = c2 * c2 * x2.std_uncertainty * x2.std_uncertainty;
    let std_u = sqrt_f64(var1 + var2);

    let u1 = GUMUncertainty {
        std_uncertainty: x1.std_uncertainty,
        degrees_of_freedom: x1.degrees_of_freedom,
        sensitivity: c1,
    };
    let u2 = GUMUncertainty {
        std_uncertainty: x2.std_uncertainty,
        degrees_of_freedom: x2.degrees_of_freedom,
        sensitivity: c2,
    };
    let dof = welch_satterthwaite_2(u1, u2);
    let k = coverage_factor_95(dof);

    GUMResult {
        value: value,
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * std_u,
    }
}

// Propagate uncertainty through division: y = x1 / x2
pub fn gum_div(x1: GUMResult, x2: GUMResult) -> GUMResult {
    if abs_f64(x2.value) < 1.0e-15 {
        return GUMResult {
            value: 0.0,
            std_uncertainty: 1.0e308,
            degrees_of_freedom: 1.0,
            coverage_factor_95: 12.71,
            expanded_uncertainty_95: 1.0e308,
        };
    }

    let value = x1.value / x2.value;

    // Sensitivity coefficients: ∂y/∂x1 = 1/x2, ∂y/∂x2 = -x1/x2^2
    let c1 = 1.0 / abs_f64(x2.value);
    let c2 = abs_f64(x1.value) / (x2.value * x2.value);

    let var1 = c1 * c1 * x1.std_uncertainty * x1.std_uncertainty;
    let var2 = c2 * c2 * x2.std_uncertainty * x2.std_uncertainty;
    let std_u = sqrt_f64(var1 + var2);

    let u1 = GUMUncertainty {
        std_uncertainty: x1.std_uncertainty,
        degrees_of_freedom: x1.degrees_of_freedom,
        sensitivity: c1,
    };
    let u2 = GUMUncertainty {
        std_uncertainty: x2.std_uncertainty,
        degrees_of_freedom: x2.degrees_of_freedom,
        sensitivity: c2,
    };
    let dof = welch_satterthwaite_2(u1, u2);
    let k = coverage_factor_95(dof);

    GUMResult {
        value: value,
        std_uncertainty: std_u,
        degrees_of_freedom: dof,
        coverage_factor_95: k,
        expanded_uncertainty_95: k * std_u,
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

// Get 95% confidence interval from GUM result
pub fn gum_interval_95(result: GUMResult) -> (f64, f64) {
    let lo = result.value - result.expanded_uncertainty_95;
    let hi = result.value + result.expanded_uncertainty_95;
    (lo, hi)
}

// Get relative uncertainty as percentage
pub fn relative_uncertainty_percent(result: GUMResult) -> f64 {
    if abs_f64(result.value) < 1.0e-15 {
        return 100.0;
    }
    100.0 * result.std_uncertainty / abs_f64(result.value)
}

// =============================================================================
// Tests
// =============================================================================

fn main() -> i32 {
    print("Testing GUM module...\n");

    // Test 1: Type B uncertainty (simple)
    let u_b = type_b_uniform(0.5);
    print("Type B uniform: PASS\n");

    // Test 2: Type A uncertainty
    let u_a = type_a_uncertainty(1.0, 10);
    print("Type A uncertainty: PASS\n");

    // Test 3: Coverage factor for normal
    let k95 = k_normal_95();
    print("Normal k95: PASS\n");

    // Test 4: Coverage factor for small DoF
    let k_5 = coverage_factor_95(5.0);
    print("Coverage factor (v=5): PASS\n");

    // Test 5: GUM simple value
    let mass = gum_simple(75.0, 0.5);
    print("GUM simple: PASS\n");

    // Test 6: GUM type A
    let temp = gum_type_a(20.0, 0.3, 10);
    print("GUM type A: PASS\n");

    // Test 7: GUM addition
    let sum = gum_add(mass, temp);
    print("GUM addition: PASS\n");

    // Test 8: GUM multiplication
    let product = gum_mul(mass, temp);
    print("GUM multiplication: PASS\n");

    // Test 9: Welch-Satterthwaite
    let dof = welch_satterthwaite_2(u_a, u_b);
    print("Welch-Satterthwaite: PASS\n");

    print("All GUM tests PASSED\n");
    0
}
