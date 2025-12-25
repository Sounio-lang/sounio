//! Comprehensive probability distribution library

use std::math::{PI, E, sqrt, exp, ln, pow, sin, cos, erf, erfc, gamma, lgamma, beta}
use linalg::{Vector, Matrix}
use rand::{Rng, thread_rng}

/// Trait for probability distributions
pub trait Distribution<T> {
    /// Sample from the distribution
    fn sample(&self, rng: &!Rng) -> T with Prob;

    /// Probability density/mass function
    fn pdf(&self, x: T) -> f64;

    /// Log probability density
    fn log_pdf(&self, x: T) -> f64 {
        self.pdf(x).ln()
    }

    /// Cumulative distribution function
    fn cdf(&self, x: T) -> f64;

    /// Mean of the distribution
    fn mean(&self) -> T;

    /// Variance of the distribution
    fn variance(&self) -> f64;

    /// Standard deviation
    fn std(&self) -> f64 {
        self.variance().sqrt()
    }
}

/// Trait for distributions supporting quantile function
pub trait Quantile<T>: Distribution<T> {
    /// Quantile function (inverse CDF)
    fn quantile(&self, p: f64) -> T;

    /// Median (50th percentile)
    fn median(&self) -> T {
        self.quantile(0.5)
    }
}

// =============================================================================
// CONTINUOUS DISTRIBUTIONS
// =============================================================================

/// Normal (Gaussian) distribution
pub struct Normal {
    pub mu: f64,     // Mean
    pub sigma: f64,  // Standard deviation
}

impl Normal {
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        Normal { mu, sigma }
    }

    pub fn standard() -> Self {
        Normal { mu: 0.0, sigma: 1.0 }
    }
}

impl Distribution<f64> for Normal {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        // Box-Muller transform
        let u1 = rng.uniform(0.0, 1.0);
        let u2 = rng.uniform(0.0, 1.0);
        let z = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2);
        self.mu + self.sigma * z
    }

    fn pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        exp(-0.5 * z * z) / (self.sigma * sqrt(2.0 * PI))
    }

    fn log_pdf(&self, x: f64) -> f64 {
        let z = (x - self.mu) / self.sigma;
        -0.5 * z * z - ln(self.sigma) - 0.5 * ln(2.0 * PI)
    }

    fn cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + erf((x - self.mu) / (self.sigma * sqrt(2.0))))
    }

    fn mean(&self) -> f64 { self.mu }

    fn variance(&self) -> f64 { self.sigma * self.sigma }
}

impl Quantile<f64> for Normal {
    fn quantile(&self, p: f64) -> f64 {
        // Approximation using rational function
        let z = normal_quantile_approx(p);
        self.mu + self.sigma * z
    }
}

/// Multivariate Normal distribution
pub struct MultivariateNormal {
    pub mean: Vector<f64>,
    pub cov: Matrix<f64>,
    chol: Matrix<f64>,  // Cholesky factor
}

impl MultivariateNormal {
    pub fn new(mean: Vector<f64>, cov: Matrix<f64>) -> Result<Self, string> {
        let chol = linalg::cholesky(&cov)?.factor;
        Ok(MultivariateNormal { mean, cov, chol })
    }

    pub fn standard(dim: usize) -> Self {
        MultivariateNormal {
            mean: Vector::zeros(dim),
            cov: Matrix::eye(dim),
            chol: Matrix::eye(dim),
        }
    }
}

impl Distribution<Vector<f64>> for MultivariateNormal {
    fn sample(&self, rng: &!Rng) -> Vector<f64> with Prob {
        let n = self.mean.len();
        let standard = Normal::standard();

        // Sample from standard normal
        let mut z = Vector::new(n);
        for i in 0..n {
            z[i] = standard.sample(rng);
        }

        // Transform: x = mean + L * z
        &self.mean + &(&self.chol * &z)
    }

    fn pdf(&self, x: Vector<f64>) -> f64 {
        self.log_pdf(x).exp()
    }

    fn log_pdf(&self, x: Vector<f64>) -> f64 {
        let n = self.mean.len() as f64;
        let diff = &x - &self.mean;

        // Solve L * y = diff for y
        let y = linalg::solve_triangular(&self.chol, &diff, Lower).unwrap();
        let quad_form = y.dot(&y);

        // Log determinant = 2 * sum(log(diag(L)))
        let log_det: f64 = (0..self.chol.nrows())
            .map(|i| self.chol[(i, i)].ln())
            .sum::<f64>() * 2.0;

        -0.5 * (n * ln(2.0 * PI) + log_det + quad_form)
    }

    fn cdf(&self, x: Vector<f64>) -> f64 {
        // MVN CDF requires numerical integration
        // Placeholder - would use Genz algorithm
        panic!("multivariate normal CDF not implemented")
    }

    fn mean(&self) -> Vector<f64> { self.mean.clone() }

    fn variance(&self) -> f64 {
        // Return trace of covariance
        (0..self.cov.nrows()).map(|i| self.cov[(i, i)]).sum()
    }
}

/// Log-Normal distribution
pub struct LogNormal {
    pub mu: f64,     // Log-mean
    pub sigma: f64,  // Log-std
}

impl LogNormal {
    pub fn new(mu: f64, sigma: f64) -> Self {
        assert!(sigma > 0.0, "sigma must be positive");
        LogNormal { mu, sigma }
    }

    /// Create from mean and std of the log-normal (not log)
    pub fn from_mean_std(mean: f64, std: f64) -> Self {
        let variance = std * std;
        let mu = ln(mean * mean / sqrt(variance + mean * mean));
        let sigma = sqrt(ln(1.0 + variance / (mean * mean)));
        LogNormal { mu, sigma }
    }
}

impl Distribution<f64> for LogNormal {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        let normal = Normal::new(self.mu, self.sigma);
        exp(normal.sample(rng))
    }

    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        let z = (ln(x) - self.mu) / self.sigma;
        exp(-0.5 * z * z) / (x * self.sigma * sqrt(2.0 * PI))
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        0.5 * (1.0 + erf((ln(x) - self.mu) / (self.sigma * sqrt(2.0))))
    }

    fn mean(&self) -> f64 {
        exp(self.mu + 0.5 * self.sigma * self.sigma)
    }

    fn variance(&self) -> f64 {
        let s2 = self.sigma * self.sigma;
        (exp(s2) - 1.0) * exp(2.0 * self.mu + s2)
    }
}

/// Gamma distribution
pub struct Gamma {
    pub shape: f64,  // α (alpha)
    pub rate: f64,   // β (beta)
}

impl Gamma {
    pub fn new(shape: f64, rate: f64) -> Self {
        assert!(shape > 0.0 && rate > 0.0, "shape and rate must be positive");
        Gamma { shape, rate }
    }

    /// Create with shape and scale (1/rate)
    pub fn with_scale(shape: f64, scale: f64) -> Self {
        Self::new(shape, 1.0 / scale)
    }
}

impl Distribution<f64> for Gamma {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        // Marsaglia and Tsang's method
        if self.shape >= 1.0 {
            let d = self.shape - 1.0/3.0;
            let c = 1.0 / sqrt(9.0 * d);

            loop {
                let x = Normal::standard().sample(rng);
                let v = 1.0 + c * x;
                if v > 0.0 {
                    let v3 = v * v * v;
                    let u = rng.uniform(0.0, 1.0);

                    if u < 1.0 - 0.0331 * x * x * x * x
                       || ln(u) < 0.5 * x * x + d * (1.0 - v3 + ln(v3))
                    {
                        return d * v3 / self.rate;
                    }
                }
            }
        } else {
            // For shape < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
            let g = Gamma::new(self.shape + 1.0, self.rate);
            let u = rng.uniform(0.0, 1.0);
            g.sample(rng) * pow(u, 1.0 / self.shape)
        }
    }

    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }

        let log_pdf = self.shape * ln(self.rate) - lgamma(self.shape)
            + (self.shape - 1.0) * ln(x) - self.rate * x;
        exp(log_pdf)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        gamma_inc(self.shape, self.rate * x) / gamma(self.shape)
    }

    fn mean(&self) -> f64 { self.shape / self.rate }

    fn variance(&self) -> f64 { self.shape / (self.rate * self.rate) }
}

/// Beta distribution
pub struct Beta {
    pub alpha: f64,
    pub beta: f64,
}

impl Beta {
    pub fn new(alpha: f64, beta: f64) -> Self {
        assert!(alpha > 0.0 && beta > 0.0, "alpha and beta must be positive");
        Beta { alpha, beta }
    }
}

impl Distribution<f64> for Beta {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        // Use gamma ratio
        let x = Gamma::new(self.alpha, 1.0).sample(rng);
        let y = Gamma::new(self.beta, 1.0).sample(rng);
        x / (x + y)
    }

    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 || x >= 1.0 { return 0.0; }

        pow(x, self.alpha - 1.0) * pow(1.0 - x, self.beta - 1.0)
            / beta(self.alpha, self.beta)
    }

    fn cdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        if x >= 1.0 { return 1.0; }
        beta_inc(self.alpha, self.beta, x)
    }

    fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    fn variance(&self) -> f64 {
        let ab = self.alpha + self.beta;
        self.alpha * self.beta / (ab * ab * (ab + 1.0))
    }
}

/// Student's t-distribution
pub struct StudentT {
    pub df: f64,  // Degrees of freedom
}

impl StudentT {
    pub fn new(df: f64) -> Self {
        assert!(df > 0.0, "degrees of freedom must be positive");
        StudentT { df }
    }
}

impl Distribution<f64> for StudentT {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        let z = Normal::standard().sample(rng);
        let chi2 = Gamma::new(self.df / 2.0, 0.5).sample(rng);
        z / sqrt(chi2 / self.df)
    }

    fn pdf(&self, x: f64) -> f64 {
        let v = self.df;
        gamma((v + 1.0) / 2.0) / (sqrt(v * PI) * gamma(v / 2.0))
            * pow(1.0 + x * x / v, -(v + 1.0) / 2.0)
    }

    fn cdf(&self, x: f64) -> f64 {
        let v = self.df;
        let t = v / (v + x * x);

        if x >= 0.0 {
            1.0 - 0.5 * beta_inc(v / 2.0, 0.5, t)
        } else {
            0.5 * beta_inc(v / 2.0, 0.5, t)
        }
    }

    fn mean(&self) -> f64 {
        if self.df > 1.0 { 0.0 } else { f64::NAN }
    }

    fn variance(&self) -> f64 {
        if self.df > 2.0 {
            self.df / (self.df - 2.0)
        } else if self.df > 1.0 {
            f64::INFINITY
        } else {
            f64::NAN
        }
    }
}

/// Exponential distribution
pub struct Exponential {
    pub rate: f64,  // λ
}

impl Exponential {
    pub fn new(rate: f64) -> Self {
        assert!(rate > 0.0, "rate must be positive");
        Exponential { rate }
    }

    pub fn from_mean(mean: f64) -> Self {
        Self::new(1.0 / mean)
    }
}

impl Distribution<f64> for Exponential {
    fn sample(&self, rng: &!Rng) -> f64 with Prob {
        -ln(1.0 - rng.uniform(0.0, 1.0)) / self.rate
    }

    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { self.rate * exp(-self.rate * x) }
    }

    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else { 1.0 - exp(-self.rate * x) }
    }

    fn mean(&self) -> f64 { 1.0 / self.rate }

    fn variance(&self) -> f64 { 1.0 / (self.rate * self.rate) }
}

impl Quantile<f64> for Exponential {
    fn quantile(&self, p: f64) -> f64 {
        -ln(1.0 - p) / self.rate
    }
}

// =============================================================================
// DISCRETE DISTRIBUTIONS
// =============================================================================

/// Poisson distribution
pub struct Poisson {
    pub lambda: f64,  // Rate parameter
}

impl Poisson {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive");
        Poisson { lambda }
    }
}

impl Distribution<i64> for Poisson {
    fn sample(&self, rng: &!Rng) -> i64 with Prob {
        // Knuth's algorithm for small lambda
        if self.lambda < 30.0 {
            let l = exp(-self.lambda);
            let mut k = 0i64;
            let mut p = 1.0;

            loop {
                k += 1;
                p *= rng.uniform(0.0, 1.0);
                if p <= l { break; }
            }

            k - 1
        } else {
            // Normal approximation for large lambda
            let normal = Normal::new(self.lambda, sqrt(self.lambda));
            normal.sample(rng).round() as i64
        }
    }

    fn pdf(&self, x: i64) -> f64 {
        if x < 0 { return 0.0; }

        let k = x as f64;
        exp(k * ln(self.lambda) - self.lambda - lgamma(k + 1.0))
    }

    fn cdf(&self, x: i64) -> f64 {
        if x < 0 { return 0.0; }

        let mut sum = 0.0;
        for k in 0..=x {
            sum += self.pdf(k);
        }
        sum
    }

    fn mean(&self) -> i64 { self.lambda.round() as i64 }

    fn variance(&self) -> f64 { self.lambda }
}

/// Binomial distribution
pub struct Binomial {
    pub n: u64,    // Number of trials
    pub p: f64,    // Success probability
}

impl Binomial {
    pub fn new(n: u64, p: f64) -> Self {
        assert!(p >= 0.0 && p <= 1.0, "p must be in [0, 1]");
        Binomial { n, p }
    }
}

impl Distribution<u64> for Binomial {
    fn sample(&self, rng: &!Rng) -> u64 with Prob {
        // Direct method for small n
        if self.n < 25 {
            let mut successes = 0u64;
            for _ in 0..self.n {
                if rng.uniform(0.0, 1.0) < self.p {
                    successes += 1;
                }
            }
            successes
        } else {
            // Normal approximation
            let mu = self.n as f64 * self.p;
            let sigma = sqrt(mu * (1.0 - self.p));
            let x = Normal::new(mu, sigma).sample(rng);
            x.round().max(0.0).min(self.n as f64) as u64
        }
    }

    fn pdf(&self, x: u64) -> f64 {
        if x > self.n { return 0.0; }

        let k = x as f64;
        let n = self.n as f64;

        exp(lgamma(n + 1.0) - lgamma(k + 1.0) - lgamma(n - k + 1.0)
            + k * ln(self.p) + (n - k) * ln(1.0 - self.p))
    }

    fn cdf(&self, x: u64) -> f64 {
        let mut sum = 0.0;
        for k in 0..=x.min(self.n) {
            sum += self.pdf(k);
        }
        sum
    }

    fn mean(&self) -> u64 { (self.n as f64 * self.p).round() as u64 }

    fn variance(&self) -> f64 { self.n as f64 * self.p * (1.0 - self.p) }
}

/// Categorical distribution
pub struct Categorical {
    pub probs: Vector<f64>,
    cumsum: Vector<f64>,
}

impl Categorical {
    pub fn new(probs: Vector<f64>) -> Self {
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probabilities must sum to 1");

        let n = probs.len();
        let mut cumsum = Vector::new(n);
        let mut acc = 0.0;
        for i in 0..n {
            acc += probs[i];
            cumsum[i] = acc;
        }

        Categorical { probs, cumsum }
    }

    pub fn uniform(k: usize) -> Self {
        let p = 1.0 / k as f64;
        let probs = Vector::from_slice(&vec![p; k]);
        Self::new(probs)
    }
}

impl Distribution<usize> for Categorical {
    fn sample(&self, rng: &!Rng) -> usize with Prob {
        let u = rng.uniform(0.0, 1.0);

        // Binary search
        let mut lo = 0;
        let mut hi = self.cumsum.len();

        while lo < hi {
            let mid = (lo + hi) / 2;
            if self.cumsum[mid] < u {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }

        lo
    }

    fn pdf(&self, x: usize) -> f64 {
        if x >= self.probs.len() { 0.0 } else { self.probs[x] }
    }

    fn cdf(&self, x: usize) -> f64 {
        if x >= self.cumsum.len() { 1.0 } else { self.cumsum[x] }
    }

    fn mean(&self) -> usize {
        let expected: f64 = self.probs.iter()
            .enumerate()
            .map(|(i, p)| i as f64 * p)
            .sum();
        expected.round() as usize
    }

    fn variance(&self) -> f64 {
        let mu = self.mean() as f64;
        self.probs.iter()
            .enumerate()
            .map(|(i, p)| (i as f64 - mu).powi(2) * p)
            .sum()
    }
}

// Helper functions
fn normal_quantile_approx(p: f64) -> f64 {
    // Rational approximation
    let a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ];
    let b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ];
    let d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = sqrt(-2.0 * ln(p));
        (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
            (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
    } else {
        let q = sqrt(-2.0 * ln(1.0 - p));
        -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
            ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
    }
}
