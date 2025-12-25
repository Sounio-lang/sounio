// fractal_blood.d
// Advanced Fractal Blood Dynamics Module
// Implements Continuous Time Random Walk (CTRW) framework for anomalous transport in vascular networks
//
// Physics Background:
// - Vascular networks exhibit fractal geometry (self-similar branching)
// - Drug transport shows subdiffusion (α < 1) due to trapping/binding
// - Standard exponential kinetics fail in fractal media
// - Mittag-Leffler function E_α(z) replaces exponential e^z
//
// References:
// - West, Brown & Enquist (1997) "A general model for the origin of allometric scaling laws"
// - Bassingthwaighte et al. (1994) "Fractal nature of regional myocardial blood flow"
// - Sokolov & Klafter (2005) "From diffusion to anomalous diffusion: A century after Einstein's Brownian motion"

module darwin_pbpk::core::fractal_blood;

use std::math;

// ============================================================================
// CONSTANTS - Universal Fractal Properties
// ============================================================================

// Fractal dimension of vascular tree (empirically measured)
// Typical range: 2.5-3.0 (Euclidean space is 3.0, pure fractal < 3.0)
const FRACTAL_DIMENSION: f64 = 2.7;

// Murray's law exponent for optimal branching
// Σ r_daughter^3 = r_parent^3 (minimizes pumping power + maintenance cost)
const MURRAY_EXPONENT: f64 = 3.0;

// Whole blood dynamic viscosity at 37°C, shear rate ~100 s^-1
const BLOOD_VISCOSITY: f64 = 0.0035; // Pa·s

// Anomalous diffusion exponent for blood (subdiffusion)
// α = 1.0: normal diffusion
// α < 1.0: subdiffusion (trapping, binding)
// α > 1.0: superdiffusion (rare in blood)
const ANOMALOUS_EXPONENT_BLOOD: f64 = 0.85;

// Maximum Taylor series terms for Mittag-Leffler function
const ML_MAX_TERMS: i32 = 100;

// Convergence tolerance for iterative calculations
const EPSILON: f64 = 1e-10;

// ============================================================================
// STRUCTURES
// ============================================================================

// Fractal parameters characterizing tissue-specific anomalous transport
struct FractalParams {
    // Fractal (Hausdorff) dimension: measure of space-filling
    // d_f = 2.0: planar network
    // d_f = 3.0: space-filling (Euclidean)
    fractal_dim: f64,

    // Anomalous diffusion exponent (0 < α ≤ 1 for subdiffusion)
    // Appears in MSD ~ t^α (mean squared displacement)
    anomalous_exponent: f64,

    // Spectral dimension: characterizes random walk return probability
    // d_s = 2 × d_f / (d_f + 2) (Alexander-Orbach conjecture)
    spectral_dim: f64,

    // Hurst exponent for fractional Brownian motion
    // H = α/2 for subdiffusion, relates to autocorrelation
    hurst_exponent: f64,
}

// Vessel segment in fractal vascular tree
struct VesselSegment {
    radius: f64,           // Vessel radius (cm)
    length: f64,           // Segment length (cm)
    flow_rate: f64@L_per_h, // Volumetric flow rate
    branching_level: i32,   // Generation number (0 = aorta)
}

// ============================================================================
// MITTAG-LEFFLER FUNCTION - Foundation of Fractional Calculus
// ============================================================================

// Approximation of Mittag-Leffler function E_α(z)
// This is the fractional generalization of exponential: E_1(z) = exp(z)
//
// Definition: E_α(z) = Σ_{k=0}^∞ z^k / Γ(αk + 1)
//
// For small |z|: use Taylor series
// For large |z|: use asymptotic expansion
//
// Critical for solving fractional ODEs:
// D^α y(t) = -λy(t) → y(t) = y₀ E_α(-λt^α)
fn mittag_leffler_approx(alpha: f64, z: f64) -> f64 {
    // Handle special cases
    if alpha == 1.0 {
        return math::exp(z);
    }

    if math::abs(z) < EPSILON {
        return 1.0;
    }

    // Taylor series for |z| < 1
    if math::abs(z) < 1.0 {
        let mut sum: f64 = 0.0;
        let mut term: f64 = 1.0;

        for k in 0..ML_MAX_TERMS {
            sum = sum + term;

            // Check convergence
            if math::abs(term) < EPSILON * math::abs(sum) {
                break;
            }

            // Next term: z^(k+1) / Γ(α(k+1) + 1)
            term = term * z / gamma_function(alpha * (k as f64 + 1.0) + 1.0);
        }

        return sum;
    }

    // Asymptotic expansion for large |z|
    // E_α(z) ~ exp(z^(1/α)) / α for z >> 1
    let z_abs = math::abs(z);
    if z > 10.0 {
        return math::exp(math::pow(z, 1.0 / alpha)) / alpha;
    }

    // For moderate z, use continued fraction (more stable)
    // Simplified approximation
    return 1.0 / (1.0 - z / gamma_function(alpha + 1.0));
}

// Gamma function Γ(x) via Stirling's approximation
// Γ(x) ≈ √(2π/x) × (x/e)^x for large x
// For small x, use recursive property Γ(x+1) = x·Γ(x)
fn gamma_function(x: f64) -> f64 {
    if x <= 0.0 {
        return 1.0; // Invalid input, return 1 to avoid NaN
    }

    // Handle integer values exactly
    if x == 1.0 {
        return 1.0;
    }
    if x == 2.0 {
        return 1.0;
    }
    if x == 3.0 {
        return 2.0;
    }
    if x == 4.0 {
        return 6.0;
    }

    // For x < 1, use recursive relation
    if x < 1.0 {
        return gamma_function(x + 1.0) / x;
    }

    // Stirling's approximation for x ≥ 1
    let sqrt_2pi = 2.5066282746310005; // √(2π)
    let log_gamma = (x - 0.5) * math::ln(x) - x + 0.5 * math::ln(sqrt_2pi);

    return math::exp(log_gamma);
}

// ============================================================================
// FRACTAL RATE CONSTANTS
// ============================================================================

// Time-dependent rate constant in fractal media
// k(t) = k₀ × t^(-h) where h = 1 - α/2
//
// Physical interpretation:
// - Reflects "slowing down" of kinetics due to trapping
// - As t → ∞, effective rate decreases (long memory)
// - For α = 1 (normal diffusion), h = 0.5
fn fractal_rate_constant(k0: f64, t: f64@h, h: f64) -> f64 {
    if t <= 0.0@h {
        return k0;
    }

    let t_numeric = t as f64; // Convert to hours (numeric)
    return k0 * math::pow(t_numeric, -h);
}

// ============================================================================
// FRACTIONAL DECAY AND ACCUMULATION
// ============================================================================

// Fractional-order decay with Mittag-Leffler kinetics
// C(t) = C₀ × E_α(-k × t^α)
//
// This replaces standard exponential decay C(t) = C₀ exp(-kt)
// For α < 1: slower than exponential (subdiffusion, trapping)
// For α = 1: reduces to standard exponential
fn fractional_decay(c0: f64@mg_per_L, k: f64, t: f64@h, alpha: f64) -> f64@mg_per_L {
    if t <= 0.0@h {
        return c0;
    }

    let t_numeric = t as f64;
    let t_alpha = math::pow(t_numeric, alpha);
    let ml_val = mittag_leffler_approx(alpha, -k * t_alpha);

    return c0 * ml_val;
}

// Fractional accumulation with input/output
// Accounts for continuous infusion into fractal compartment
//
// For constant input rate R and elimination k:
// C(t) = (R/k) × [1 - E_α(-k × t^α)]
fn fractional_accumulation(k_in: f64, k_out: f64, t: f64@h, alpha: f64) -> f64 {
    if t <= 0.0@h {
        return 0.0;
    }

    let t_numeric = t as f64;
    let t_alpha = math::pow(t_numeric, alpha);

    // Steady-state concentration
    let c_ss = k_in / k_out;

    // Time to reach steady state (fractional)
    let ml_val = mittag_leffler_approx(alpha, -k_out * t_alpha);

    return c_ss * (1.0 - ml_val);
}

// ============================================================================
// TRANSIT TIME DISTRIBUTIONS
// ============================================================================

// Power-law transit time probability density function
// P(t) ~ t^(-(1+α)) for t > t₀
//
// Heavy-tailed distribution characteristic of fractal networks
// Long transit times more probable than in normal diffusion
fn power_law_transit_pdf(t: f64@h, t0: f64@h, alpha: f64) -> f64 {
    if t < t0 {
        return 0.0;
    }

    let t_numeric = t as f64;
    let t0_numeric = t0 as f64;

    // Normalization constant (if α > 0)
    let norm = alpha / math::pow(t0_numeric, alpha);

    return norm * math::pow(t_numeric, -(1.0 + alpha));
}

// Mean transit time in fractal medium
// MTT = ∫ t × P(t) dt
//
// WARNING: For α ≤ 1, MTT may be infinite!
// Returns finite value for α > 1 only
fn mean_transit_time_fractal(t0: f64@h, alpha: f64) -> f64@h {
    if alpha <= 1.0 {
        // MTT is infinite (heavy tail dominates)
        return 1e9@h; // Return very large value
    }

    // For α > 1: MTT = t₀ × α / (α - 1)
    let t0_numeric = t0 as f64;
    let mtt = t0_numeric * alpha / (alpha - 1.0);

    return mtt@h;
}

// ============================================================================
// TISSUE-SPECIFIC FRACTAL DIMENSIONS
// ============================================================================

// Returns fractal dimension for different organs
// Based on empirical measurements of vascular architecture
//
// Organ codes:
// 0 = Liver (highly vascularized, dense capillary networks)
// 1 = Kidney (glomerular + tubular vasculature)
// 2 = Brain (blood-brain barrier, dense microcirculation)
// 3 = Muscle (parallel capillary arrays)
// 4 = Adipose (sparse vasculature)
// 5 = Heart (coronary tree, dense perfusion)
// 6 = Lung (pulmonary vasculature, gas exchange)
// 7 = Gut (villous architecture)
fn tissue_fractal_dimension(organ: i32) -> f64 {
    if organ == 0 {
        return 2.8; // Liver: highly fractal sinusoidal network
    } else if organ == 1 {
        return 2.7; // Kidney: complex glomerular structure
    } else if organ == 2 {
        return 2.6; // Brain: dense but ordered capillaries
    } else if organ == 3 {
        return 2.4; // Muscle: more parallel organization
    } else if organ == 4 {
        return 2.2; // Adipose: sparse, low metabolic demand
    } else if organ == 5 {
        return 2.75; // Heart: dense coronary network
    } else if organ == 6 {
        return 2.85; // Lung: extensive air-blood interface
    } else if organ == 7 {
        return 2.65; // Gut: villous + crypt architecture
    } else {
        return 2.5; // Default: intermediate value
    }
}

// ============================================================================
// SPECTRAL DIMENSION (Alexander-Orbach Conjecture)
// ============================================================================

// Spectral dimension d_s characterizes random walk properties
// Alexander-Orbach conjecture: d_s = 2d_f / (d_f + 2)
//
// Physical meaning:
// - Determines return probability: P(r=0, t) ~ t^(-d_s/2)
// - Lower d_s → higher return probability (more "compact" walks)
// - For Euclidean space: d_s = d (standard result)
fn spectral_dimension(fractal_dim: f64) -> f64 {
    return 2.0 * fractal_dim / (fractal_dim + 2.0);
}

// ============================================================================
// PBPK COUPLING FUNCTIONS
// ============================================================================

// Apply fractal correction to standard PBPK concentration
// Converts C_standard(t) → C_fractal(t)
//
// Method: Convolve standard solution with fractional kernel
// C_fractal(t) ≈ C_standard(t) × E_α(-t^α / τ^α)
//
// Where τ is characteristic timescale
fn apply_fractal_correction(c_standard: f64@mg_per_L, t: f64@h, alpha: f64) -> f64@mg_per_L {
    if t <= 0.0@h {
        return c_standard;
    }

    // Characteristic time (assume 1 hour for normalization)
    let tau = 1.0;

    let t_numeric = t as f64;
    let t_alpha = math::pow(t_numeric / tau, alpha);
    let correction = mittag_leffler_approx(alpha, -t_alpha);

    return c_standard * correction;
}

// Area Under Curve (AUC) with fractional kinetics
// AUC = ∫₀^∞ C₀ E_α(-kt^α) dt
//
// Analytical result: AUC = C₀ / k^(1/α) × Γ(1/α)
//
// For α = 1: recovers standard AUC = C₀/k
fn fractal_auc(c0: f64@mg_per_L, k: f64, alpha: f64) -> f64@mg_h_per_L {
    if k <= 0.0 || alpha <= 0.0 {
        return 0.0@mg_h_per_L;
    }

    // AUC = C₀ / k^(1/α) × Γ(1/α)
    let k_inv = math::pow(k, 1.0 / alpha);
    let gamma_term = gamma_function(1.0 / alpha);

    let auc = (c0 as f64) * gamma_term / k_inv;

    return auc@mg_h_per_L;
}

// ============================================================================
// FRACTAL PARAMETER INITIALIZATION
// ============================================================================

// Create FractalParams for a given organ
// Automatically computes spectral dimension and Hurst exponent
fn create_fractal_params(organ: i32, anomalous_exp: f64) -> FractalParams {
    let d_f = tissue_fractal_dimension(organ);
    let d_s = spectral_dimension(d_f);
    let h = anomalous_exp / 2.0;

    return FractalParams {
        fractal_dim: d_f,
        anomalous_exponent: anomalous_exp,
        spectral_dim: d_s,
        hurst_exponent: h,
    };
}

// ============================================================================
// VESSEL SEGMENT FUNCTIONS
// ============================================================================

// Create vessel segment with Murray's law branching
// Child vessels satisfy: Σ r_child³ = r_parent³
fn create_vessel_segment(
    radius: f64,
    length: f64,
    flow_rate: f64@L_per_h,
    level: i32
) -> VesselSegment {
    return VesselSegment {
        radius: radius,
        length: length,
        flow_rate: flow_rate,
        branching_level: level,
    };
}

// Calculate child vessel radius from parent using Murray's law
// For bifurcation: r_child = r_parent / 2^(1/3) ≈ 0.794 × r_parent
fn murray_child_radius(parent_radius: f64, num_children: i32) -> f64 {
    let exp = 1.0 / MURRAY_EXPONENT;
    return parent_radius / math::pow(num_children as f64, exp);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Calculate Reynolds number for vessel segment (dimensionless)
// Re = ρvD/μ where ρ = density, v = velocity, D = diameter, μ = viscosity
//
// Re < 2300: laminar flow
// Re > 4000: turbulent flow
fn reynolds_number(flow_rate: f64@L_per_h, radius: f64) -> f64 {
    // Blood density ≈ 1.06 g/cm³ = 1060 kg/m³
    let rho = 1060.0; // kg/m³

    // Convert flow rate to m³/s
    let q_si = (flow_rate as f64) / 3600.0 / 1000.0; // L/h → m³/s

    // Velocity: v = Q / A where A = πr²
    let area = 3.14159 * radius * radius / 10000.0; // cm² → m²
    let velocity = q_si / area; // m/s

    // Diameter in meters
    let diameter = 2.0 * radius / 100.0; // cm → m

    return rho * velocity * diameter / BLOOD_VISCOSITY;
}

// Calculate Womersley number (pulsatile flow parameter)
// Wo = r√(ωρ/μ) where ω = angular frequency (typically 2π/0.8s for heart rate)
fn womersley_number(radius: f64, heart_rate: f64) -> f64 {
    let rho = 1060.0; // kg/m³
    let omega = 2.0 * 3.14159 * heart_rate / 60.0; // Hz → rad/s
    let r_si = radius / 100.0; // cm → m

    return r_si * math::sqrt(omega * rho / BLOOD_VISCOSITY);
}

// ============================================================================
// EXPORT PUBLIC INTERFACE
// ============================================================================

export {
    // Constants
    FRACTAL_DIMENSION,
    MURRAY_EXPONENT,
    BLOOD_VISCOSITY,
    ANOMALOUS_EXPONENT_BLOOD,

    // Structures
    FractalParams,
    VesselSegment,

    // Core functions
    mittag_leffler_approx,
    gamma_function,
    fractal_rate_constant,
    fractional_decay,
    fractional_accumulation,
    power_law_transit_pdf,
    mean_transit_time_fractal,

    // Tissue properties
    tissue_fractal_dimension,
    spectral_dimension,

    // PBPK coupling
    apply_fractal_correction,
    fractal_auc,

    // Initialization
    create_fractal_params,

    // Vessel functions
    create_vessel_segment,
    murray_child_radius,
    reynolds_number,
    womersley_number,
};
