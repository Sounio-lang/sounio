//! Benchmarks for Uncertainty Propagation Backends
//!
//! Compares performance and accuracy of different propagation methods:
//! - Point (baseline)
//! - Interval arithmetic
//! - Affine arithmetic
//! - Monte Carlo
//! - Sequential Monte Carlo
//!
//! Run with: cargo bench --bench propagation_benchmark
//!
//! Requires: criterion = "0.5"

use std::time::Instant;
use std::f64::consts::PI;

// =============================================================================
// Benchmark Configuration
// =============================================================================

/// Number of iterations for timing
const TIMING_ITERATIONS: usize = 1000;

/// Number of samples for Monte Carlo
const MC_SAMPLES: usize = 10000;

/// Number of particles for SMC
const SMC_PARTICLES: usize = 1000;

/// Test functions with different characteristics
#[derive(Debug, Clone, Copy)]
pub enum TestFunction {
    /// f(x) = x + c (linear, easy)
    Linear,
    /// f(x) = x * y (bilinear)
    Bilinear,
    /// f(x) = x / y (division, harder for intervals)
    Division,
    /// f(x) = exp(-x) (nonlinear, bounded output)
    Exponential,
    /// f(x) = sin(x) (periodic, bounded)
    Sinusoidal,
    /// f(x,y,z) = F*D*ka / (V*(ka-ke)) * (exp(-ke*t) - exp(-ka*t)) (PK model)
    PKModel,
}

// =============================================================================
// Propagation Implementations
// =============================================================================

/// Point propagation (no uncertainty)
mod point {
    use super::*;

    pub fn propagate(func: TestFunction, inputs: &[f64]) -> f64 {
        match func {
            TestFunction::Linear => inputs[0] + 5.0,
            TestFunction::Bilinear => inputs[0] * inputs[1],
            TestFunction::Division => inputs[0] / inputs[1],
            TestFunction::Exponential => (-inputs[0]).exp(),
            TestFunction::Sinusoidal => inputs[0].sin(),
            TestFunction::PKModel => pk_model(inputs),
        }
    }

    fn pk_model(p: &[f64]) -> f64 {
        let (f, dose, ka, vd, cl, t) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let ke = cl / vd;
        if (ka - ke).abs() < 1e-10 { return 0.0; }
        let prefactor = f * dose * ka / (vd * (ka - ke));
        prefactor * ((-ke * t).exp() - (-ka * t).exp())
    }
}

/// Interval arithmetic propagation
mod interval {
    use super::*;

    #[derive(Debug, Clone, Copy)]
    pub struct Interval {
        pub lo: f64,
        pub hi: f64,
    }

    impl Interval {
        pub fn new(lo: f64, hi: f64) -> Self {
            Self { lo: lo.min(hi), hi: lo.max(hi) }
        }

        pub fn from_point_uncertainty(value: f64, uncertainty: f64) -> Self {
            Self::new(value - uncertainty, value + uncertainty)
        }

        pub fn midpoint(&self) -> f64 {
            (self.lo + self.hi) / 2.0
        }

        pub fn width(&self) -> f64 {
            self.hi - self.lo
        }

        pub fn add(self, other: Self) -> Self {
            Self::new(self.lo + other.lo, self.hi + other.hi)
        }

        pub fn sub(self, other: Self) -> Self {
            Self::new(self.lo - other.hi, self.hi - other.lo)
        }

        pub fn mul(self, other: Self) -> Self {
            let products = [
                self.lo * other.lo,
                self.lo * other.hi,
                self.hi * other.lo,
                self.hi * other.hi,
            ];
            Self::new(
                products.iter().copied().fold(f64::INFINITY, f64::min),
                products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        }

        pub fn div(self, other: Self) -> Self {
            if other.lo <= 0.0 && other.hi >= 0.0 {
                // Division by interval containing zero
                Self::new(f64::NEG_INFINITY, f64::INFINITY)
            } else {
                self.mul(Self::new(1.0 / other.hi, 1.0 / other.lo))
            }
        }

        pub fn exp(self) -> Self {
            Self::new(self.lo.exp(), self.hi.exp())
        }

        pub fn neg(self) -> Self {
            Self::new(-self.hi, -self.lo)
        }

        pub fn sin(self) -> Self {
            // Conservative: if width > 2π, return [-1, 1]
            if self.width() >= 2.0 * PI {
                return Self::new(-1.0, 1.0);
            }
            // Otherwise, sample endpoints and critical points
            let mut lo = self.lo.sin().min(self.hi.sin());
            let mut hi = self.lo.sin().max(self.hi.sin());

            // Check for extrema within interval
            let k_lo = (self.lo / (PI / 2.0)).ceil() as i32;
            let k_hi = (self.hi / (PI / 2.0)).floor() as i32;

            for k in k_lo..=k_hi {
                let x = k as f64 * PI / 2.0;
                if x >= self.lo && x <= self.hi {
                    let y = x.sin();
                    lo = lo.min(y);
                    hi = hi.max(y);
                }
            }

            Self::new(lo, hi)
        }
    }

    pub fn propagate(func: TestFunction, inputs: &[Interval]) -> Interval {
        match func {
            TestFunction::Linear => inputs[0].add(Interval::new(5.0, 5.0)),
            TestFunction::Bilinear => inputs[0].mul(inputs[1]),
            TestFunction::Division => inputs[0].div(inputs[1]),
            TestFunction::Exponential => inputs[0].neg().exp(),
            TestFunction::Sinusoidal => inputs[0].sin(),
            TestFunction::PKModel => pk_model(inputs),
        }
    }

    fn pk_model(p: &[Interval]) -> Interval {
        let (f, dose, ka, vd, cl, t) = (p[0], p[1], p[2], p[3], p[4], p[5]);
        let ke = cl.div(vd);
        let ka_minus_ke = ka.sub(ke);

        // Handle ka ≈ ke (could divide by zero)
        if ka_minus_ke.lo <= 0.0 && ka_minus_ke.hi >= 0.0 {
            return Interval::new(f64::NEG_INFINITY, f64::INFINITY);
        }

        let prefactor = f.mul(dose).mul(ka).div(vd.mul(ka_minus_ke));
        let exp_ke = ke.neg().mul(t).exp();
        let exp_ka = ka.neg().mul(t).exp();
        prefactor.mul(exp_ke.sub(exp_ka))
    }
}

/// Affine arithmetic propagation
mod affine {
    use super::*;

    #[derive(Debug, Clone)]
    pub struct AffineForm {
        pub center: f64,
        pub noise: Vec<(u32, f64)>, // (noise_id, coefficient)
    }

    static NEXT_NOISE_ID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

    impl AffineForm {
        pub fn new(center: f64) -> Self {
            Self { center, noise: Vec::new() }
        }

        pub fn from_interval(lo: f64, hi: f64) -> Self {
            let center = (lo + hi) / 2.0;
            let radius = (hi - lo) / 2.0;
            let id = NEXT_NOISE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Self {
                center,
                noise: vec![(id, radius)],
            }
        }

        pub fn radius(&self) -> f64 {
            self.noise.iter().map(|(_, c)| c.abs()).sum()
        }

        pub fn to_interval(&self) -> (f64, f64) {
            let r = self.radius();
            (self.center - r, self.center + r)
        }

        pub fn add(&self, other: &Self) -> Self {
            let mut noise = self.noise.clone();
            for (id, coef) in &other.noise {
                if let Some(pos) = noise.iter().position(|(i, _)| i == id) {
                    noise[pos].1 += coef;
                } else {
                    noise.push((*id, *coef));
                }
            }
            Self {
                center: self.center + other.center,
                noise,
            }
        }

        pub fn sub(&self, other: &Self) -> Self {
            let mut noise = self.noise.clone();
            for (id, coef) in &other.noise {
                if let Some(pos) = noise.iter().position(|(i, _)| i == id) {
                    noise[pos].1 -= coef;
                } else {
                    noise.push((*id, -coef));
                }
            }
            Self {
                center: self.center - other.center,
                noise,
            }
        }

        pub fn mul(&self, other: &Self) -> Self {
            // Affine * Affine introduces quadratic terms → new noise symbol
            let mut noise: Vec<(u32, f64)> = Vec::new();

            // Linear terms
            for (id, c) in &self.noise {
                noise.push((*id, c * other.center));
            }
            for (id, c) in &other.noise {
                if let Some(pos) = noise.iter().position(|(i, _)| i == id) {
                    noise[pos].1 += c * self.center;
                } else {
                    noise.push((*id, c * self.center));
                }
            }

            // Quadratic remainder (new noise symbol)
            let quad_error = self.radius() * other.radius();
            if quad_error > 1e-15 {
                let id = NEXT_NOISE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                noise.push((id, quad_error));
            }

            Self {
                center: self.center * other.center,
                noise,
            }
        }

        pub fn div(&self, other: &Self) -> Self {
            // Use interval reciprocal + affine multiply
            let (lo, hi) = other.to_interval();
            if lo <= 0.0 && hi >= 0.0 {
                // Division by interval containing zero
                return Self::from_interval(f64::NEG_INFINITY, f64::INFINITY);
            }

            let inv_center = 1.0 / other.center;
            let inv_radius = (1.0/lo - 1.0/hi).abs() / 2.0;

            let inv = Self {
                center: inv_center,
                noise: vec![(NEXT_NOISE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst), inv_radius)],
            };

            self.mul(&inv)
        }

        pub fn exp(&self) -> Self {
            // exp(x) ≈ exp(center) * (1 + noise_terms) for small deviations
            let exp_center = self.center.exp();
            let radius = self.radius();

            // First-order approximation with remainder
            let mut noise: Vec<(u32, f64)> = self.noise.iter()
                .map(|(id, c)| (*id, c * exp_center))
                .collect();

            // Add remainder for nonlinearity
            let remainder = (exp_center * radius * radius / 2.0).abs();
            if remainder > 1e-15 {
                let id = NEXT_NOISE_ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                noise.push((id, remainder));
            }

            Self {
                center: exp_center,
                noise,
            }
        }

        pub fn neg(&self) -> Self {
            Self {
                center: -self.center,
                noise: self.noise.iter().map(|(id, c)| (*id, -c)).collect(),
            }
        }
    }

    pub fn propagate(func: TestFunction, inputs: &[AffineForm]) -> AffineForm {
        match func {
            TestFunction::Linear => inputs[0].add(&AffineForm::new(5.0)),
            TestFunction::Bilinear => inputs[0].mul(&inputs[1]),
            TestFunction::Division => inputs[0].div(&inputs[1]),
            TestFunction::Exponential => inputs[0].neg().exp(),
            TestFunction::Sinusoidal => {
                // sin(x) ≈ sin(center) + cos(center) * noise
                let c = inputs[0].center;
                let sin_c = c.sin();
                let cos_c = c.cos();
                let noise: Vec<(u32, f64)> = inputs[0].noise.iter()
                    .map(|(id, coef)| (*id, cos_c * coef))
                    .collect();
                AffineForm { center: sin_c, noise }
            }
            TestFunction::PKModel => pk_model(inputs),
        }
    }

    fn pk_model(p: &[AffineForm]) -> AffineForm {
        let (f, dose, ka, vd, cl, t) = (&p[0], &p[1], &p[2], &p[3], &p[4], &p[5]);
        let ke = cl.div(vd);
        let ka_minus_ke = ka.sub(&ke);

        let prefactor = f.mul(dose).mul(ka).div(&vd.mul(&ka_minus_ke));
        let exp_ke = ke.neg().mul(t).exp();
        let exp_ka = ka.neg().mul(t).exp();
        prefactor.mul(&exp_ke.sub(&exp_ka))
    }
}

/// Monte Carlo propagation
mod montecarlo {
    use super::*;

    pub struct MCResult {
        pub mean: f64,
        pub std_dev: f64,
        pub percentile_2_5: f64,
        pub percentile_97_5: f64,
        pub samples: Vec<f64>,
    }

    pub fn propagate(
        func: TestFunction,
        input_means: &[f64],
        input_stds: &[f64],
        n_samples: usize,
    ) -> MCResult {
        let n_inputs = input_means.len();

        // Generate samples
        let mut outputs = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            // Generate input samples (quasi-random for better coverage)
            let inputs: Vec<f64> = (0..n_inputs)
                .map(|j| {
                    let u = ((i * n_inputs + j) as f64 + 0.5) / (n_samples * n_inputs) as f64;
                    let z = inverse_normal_cdf(u);
                    input_means[j] + z * input_stds[j]
                })
                .collect();

            let output = super::point::propagate(func, &inputs);
            if output.is_finite() {
                outputs.push(output);
            }
        }

        // Compute statistics
        let n = outputs.len() as f64;
        let mean = outputs.iter().sum::<f64>() / n;
        let variance = outputs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        // Sort for percentiles
        outputs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx_2_5 = ((outputs.len() as f64) * 0.025) as usize;
        let idx_97_5 = ((outputs.len() as f64) * 0.975) as usize;

        MCResult {
            mean,
            std_dev,
            percentile_2_5: outputs.get(idx_2_5).copied().unwrap_or(mean),
            percentile_97_5: outputs.get(idx_97_5.min(outputs.len()-1)).copied().unwrap_or(mean),
            samples: outputs,
        }
    }

    fn inverse_normal_cdf(p: f64) -> f64 {
        if p <= 0.0 { return -6.0; }
        if p >= 1.0 { return 6.0; }

        let p = if p > 0.5 { 1.0 - p } else { p };
        let t = (-2.0 * p.ln()).sqrt();

        let c0 = 2.515517;
        let c1 = 0.802853;
        let c2 = 0.010328;
        let d1 = 1.432788;
        let d2 = 0.189269;
        let d3 = 0.001308;

        let z = t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t);

        if p > 0.5 { -z } else { z }
    }
}

// =============================================================================
// Benchmark Runner
// =============================================================================

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub method: String,
    pub function: TestFunction,
    pub mean_time_us: f64,
    pub std_time_us: f64,
    pub result_center: f64,
    pub result_width: f64,
    pub relative_accuracy: f64,  // Compared to MC reference
}

pub fn run_benchmark(func: TestFunction, input_means: &[f64], input_stds: &[f64]) -> Vec<BenchmarkResult> {
    let mut results = Vec::new();

    // Monte Carlo reference (ground truth)
    let mc_start = Instant::now();
    let mc_result = montecarlo::propagate(func, input_means, input_stds, MC_SAMPLES);
    let mc_time = mc_start.elapsed();

    let mc_width = mc_result.percentile_97_5 - mc_result.percentile_2_5;

    results.push(BenchmarkResult {
        method: "MonteCarlo".into(),
        function: func,
        mean_time_us: mc_time.as_micros() as f64,
        std_time_us: 0.0,
        result_center: mc_result.mean,
        result_width: mc_width,
        relative_accuracy: 1.0,
    });

    // Point (baseline)
    let mut point_times = Vec::with_capacity(TIMING_ITERATIONS);
    let mut point_result = 0.0;
    for _ in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        point_result = point::propagate(func, input_means);
        point_times.push(start.elapsed().as_nanos() as f64);
    }
    let point_mean_time = point_times.iter().sum::<f64>() / point_times.len() as f64 / 1000.0;
    let point_std_time = (point_times.iter().map(|t| (t/1000.0 - point_mean_time).powi(2)).sum::<f64>() / point_times.len() as f64).sqrt();

    results.push(BenchmarkResult {
        method: "Point".into(),
        function: func,
        mean_time_us: point_mean_time,
        std_time_us: point_std_time,
        result_center: point_result,
        result_width: 0.0,
        relative_accuracy: if mc_result.std_dev > 0.0 {
            1.0 - (point_result - mc_result.mean).abs() / mc_result.std_dev
        } else { 1.0 },
    });

    // Interval
    let intervals: Vec<interval::Interval> = input_means.iter().zip(input_stds.iter())
        .map(|(&m, &s)| interval::Interval::from_point_uncertainty(m, s * 1.96))
        .collect();

    let mut interval_times = Vec::with_capacity(TIMING_ITERATIONS);
    let mut interval_result = interval::Interval::new(0.0, 0.0);
    for _ in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        interval_result = interval::propagate(func, &intervals);
        interval_times.push(start.elapsed().as_nanos() as f64);
    }
    let interval_mean_time = interval_times.iter().sum::<f64>() / interval_times.len() as f64 / 1000.0;
    let interval_std_time = (interval_times.iter().map(|t| (t/1000.0 - interval_mean_time).powi(2)).sum::<f64>() / interval_times.len() as f64).sqrt();

    let interval_covers_mc = interval_result.lo <= mc_result.percentile_2_5 && interval_result.hi >= mc_result.percentile_97_5;

    results.push(BenchmarkResult {
        method: "Interval".into(),
        function: func,
        mean_time_us: interval_mean_time,
        std_time_us: interval_std_time,
        result_center: interval_result.midpoint(),
        result_width: interval_result.width(),
        relative_accuracy: if interval_covers_mc {
            mc_width / interval_result.width().max(1e-10)
        } else { 0.0 },
    });

    // Affine
    let affines: Vec<affine::AffineForm> = input_means.iter().zip(input_stds.iter())
        .map(|(&m, &s)| affine::AffineForm::from_interval(m - s * 1.96, m + s * 1.96))
        .collect();

    let mut affine_times = Vec::with_capacity(TIMING_ITERATIONS);
    let mut affine_result = affine::AffineForm::new(0.0);
    for _ in 0..TIMING_ITERATIONS {
        let start = Instant::now();
        affine_result = affine::propagate(func, &affines);
        affine_times.push(start.elapsed().as_nanos() as f64);
    }
    let affine_mean_time = affine_times.iter().sum::<f64>() / affine_times.len() as f64 / 1000.0;
    let affine_std_time = (affine_times.iter().map(|t| (t/1000.0 - affine_mean_time).powi(2)).sum::<f64>() / affine_times.len() as f64).sqrt();

    let (aff_lo, aff_hi) = affine_result.to_interval();
    let affine_width = aff_hi - aff_lo;
    let affine_covers_mc = aff_lo <= mc_result.percentile_2_5 && aff_hi >= mc_result.percentile_97_5;

    results.push(BenchmarkResult {
        method: "Affine".into(),
        function: func,
        mean_time_us: affine_mean_time,
        std_time_us: affine_std_time,
        result_center: affine_result.center,
        result_width: affine_width,
        relative_accuracy: if affine_covers_mc {
            mc_width / affine_width.max(1e-10)
        } else { 0.0 },
    });

    results
}

/// Print benchmark results as table
pub fn print_results(results: &[BenchmarkResult]) {
    println!("\n{:=<100}", "");
    println!("PROPAGATION BENCHMARK RESULTS");
    println!("{:=<100}", "");
    println!();

    println!("{:<15} {:<12} {:>12} {:>12} {:>15} {:>15} {:>12}",
        "Method", "Function", "Time (μs)", "± Std", "Center", "Width (95%)", "Accuracy");
    println!("{:-<100}", "");

    for r in results {
        println!("{:<15} {:?} {:>12.3} {:>12.3} {:>15.6} {:>15.6} {:>11.1}%",
            r.method,
            r.function,
            r.mean_time_us,
            r.std_time_us,
            r.result_center,
            r.result_width,
            r.relative_accuracy * 100.0
        );
    }
}

/// Run all benchmarks
pub fn run_all_benchmarks() {
    println!("\n╔════════════════════════════════════════════════════════════╗");
    println!("║     SOUNIO UNCERTAINTY PROPAGATION BENCHMARK SUITE         ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    // Test 1: Linear function
    println!("\n▶ Test 1: Linear function f(x) = x + 5");
    let results = run_benchmark(
        TestFunction::Linear,
        &[10.0],
        &[1.0],
    );
    print_results(&results);

    // Test 2: Bilinear function
    println!("\n▶ Test 2: Bilinear function f(x,y) = x * y");
    let results = run_benchmark(
        TestFunction::Bilinear,
        &[10.0, 5.0],
        &[1.0, 0.5],
    );
    print_results(&results);

    // Test 3: Division
    println!("\n▶ Test 3: Division f(x,y) = x / y");
    let results = run_benchmark(
        TestFunction::Division,
        &[10.0, 5.0],
        &[1.0, 0.3],
    );
    print_results(&results);

    // Test 4: Exponential
    println!("\n▶ Test 4: Exponential f(x) = exp(-x)");
    let results = run_benchmark(
        TestFunction::Exponential,
        &[1.0],
        &[0.2],
    );
    print_results(&results);

    // Test 5: Sinusoidal
    println!("\n▶ Test 5: Sinusoidal f(x) = sin(x)");
    let results = run_benchmark(
        TestFunction::Sinusoidal,
        &[1.0],
        &[0.3],
    );
    print_results(&results);

    // Test 6: PK Model (realistic)
    println!("\n▶ Test 6: One-compartment PK model");
    println!("   C(t) = F·D·ka / (V·(ka-ke)) · (exp(-ke·t) - exp(-ka·t))");
    let results = run_benchmark(
        TestFunction::PKModel,
        &[0.85, 100.0, 1.2, 50.0, 10.0, 4.0], // F, Dose, ka, Vd, CL, t
        &[0.1, 5.0, 0.3, 10.0, 3.0, 0.1],     // Uncertainties
    );
    print_results(&results);

    println!("\n{:=<100}", "");
    println!("SUMMARY");
    println!("{:=<100}", "");
    println!("• Point: Fastest, no uncertainty information");
    println!("• Interval: Guaranteed bounds, may overestimate");
    println!("• Affine: Tighter bounds for correlated operations");
    println!("• Monte Carlo: Most accurate, slowest");
    println!("\nAccuracy = MC_width / Method_width (1.0 = same tightness as MC)");
    println!("Values > 1.0 indicate tighter-than-MC bounds (usually overconfident)");
    println!("Values < 1.0 indicate wider bounds (conservative)");
}

// =============================================================================
// Main entry point for standalone benchmark
// =============================================================================

fn main() {
    run_all_benchmarks();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_runs() {
        let results = run_benchmark(
            TestFunction::Linear,
            &[10.0],
            &[1.0],
        );
        assert_eq!(results.len(), 4);
        assert!(results[0].mean_time_us > 0.0);
    }
}
