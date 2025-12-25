//! Benchmark Runner
//!
//! Provides infrastructure for running and analyzing benchmarks.
//!
//! # Example
//!
//! ```d
//! #[bench]
//! fn bench_vector_push(b: &Bencher) {
//!     b.iter(|| {
//!         let v = Vec::new()
//!         for i in 0..1000 {
//!             v.push(i)
//!         }
//!     })
//! }
//! ```
//!
//! # Statistical Analysis
//!
//! The benchmark runner performs statistical analysis including:
//! - Mean and standard deviation
//! - Median and percentiles (p50, p95, p99)
//! - Min/max values
//! - Outlier detection
//! - Throughput calculation

use super::discovery::TestCase;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Configuration for benchmark execution
#[derive(Debug, Clone)]
pub struct BenchConfig {
    /// Minimum number of iterations
    pub min_iterations: u64,
    /// Maximum number of iterations
    pub max_iterations: u64,
    /// Target time for benchmarking
    pub target_time: Duration,
    /// Warmup iterations before measurement
    pub warmup_iterations: u64,
    /// Number of samples to collect
    pub sample_count: usize,
    /// Enable outlier filtering
    pub filter_outliers: bool,
    /// Confidence level for statistical analysis
    pub confidence_level: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            min_iterations: 10,
            max_iterations: 10_000_000,
            target_time: Duration::from_secs(3),
            warmup_iterations: 3,
            sample_count: 100,
            filter_outliers: true,
            confidence_level: 0.95,
        }
    }
}

/// Statistics from benchmark measurements
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchStats {
    /// Mean execution time per iteration (nanoseconds)
    pub mean_ns: f64,
    /// Standard deviation (nanoseconds)
    pub std_dev_ns: f64,
    /// Median execution time (nanoseconds)
    pub median_ns: f64,
    /// Minimum execution time (nanoseconds)
    pub min_ns: f64,
    /// Maximum execution time (nanoseconds)
    pub max_ns: f64,
    /// 50th percentile (same as median)
    pub p50_ns: f64,
    /// 95th percentile
    pub p95_ns: f64,
    /// 99th percentile
    pub p99_ns: f64,
    /// Total iterations run
    pub iterations: u64,
    /// Number of samples collected
    pub sample_count: usize,
    /// Coefficient of variation (std_dev / mean)
    pub cv: f64,
    /// Outliers removed
    pub outliers_removed: usize,
    /// Throughput (operations per second)
    pub throughput: f64,
}

impl BenchStats {
    /// Format as human-readable string
    pub fn display(&self) -> String {
        format!(
            "{}/iter (+/- {}) [{} .. {}]",
            format_duration(self.mean_ns),
            format_duration(self.std_dev_ns),
            format_duration(self.min_ns),
            format_duration(self.max_ns),
        )
    }

    /// Format as detailed report
    pub fn detailed_display(&self) -> String {
        format!(
            "  mean:   {} (+/- {})\n  median: {}\n  min:    {}\n  max:    {}\n  p95:    {}\n  p99:    {}\n  throughput: {:.2} ops/s\n  iterations: {}\n  samples: {} (outliers removed: {})",
            format_duration(self.mean_ns),
            format_duration(self.std_dev_ns),
            format_duration(self.median_ns),
            format_duration(self.min_ns),
            format_duration(self.max_ns),
            format_duration(self.p95_ns),
            format_duration(self.p99_ns),
            self.throughput,
            self.iterations,
            self.sample_count,
            self.outliers_removed,
        )
    }
}

/// Format nanoseconds as human-readable duration
fn format_duration(ns: f64) -> String {
    if ns < 1_000.0 {
        format!("{:.2} ns", ns)
    } else if ns < 1_000_000.0 {
        format!("{:.2} µs", ns / 1_000.0)
    } else if ns < 1_000_000_000.0 {
        format!("{:.2} ms", ns / 1_000_000.0)
    } else {
        format!("{:.2} s", ns / 1_000_000_000.0)
    }
}

/// Result of a benchmark run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchResult {
    /// Benchmark name
    pub name: String,
    /// Full qualified name
    pub full_name: String,
    /// Statistics
    pub stats: BenchStats,
    /// Raw sample data (optional, for detailed analysis)
    pub samples: Option<Vec<f64>>,
    /// Comparison with baseline (if available)
    pub comparison: Option<BenchComparison>,
}

impl BenchResult {
    /// Create a new benchmark result
    pub fn new(name: String, full_name: String, stats: BenchStats) -> Self {
        Self {
            name,
            full_name,
            stats,
            samples: None,
            comparison: None,
        }
    }
}

/// Comparison with a baseline benchmark
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchComparison {
    /// Baseline mean (nanoseconds)
    pub baseline_mean_ns: f64,
    /// Current mean (nanoseconds)
    pub current_mean_ns: f64,
    /// Percentage change (positive = slower, negative = faster)
    pub change_percent: f64,
    /// Whether the change is statistically significant
    pub significant: bool,
    /// Confidence interval lower bound
    pub ci_lower: f64,
    /// Confidence interval upper bound
    pub ci_upper: f64,
}

impl BenchComparison {
    /// Create a comparison
    pub fn new(baseline: f64, current: f64, significant: bool) -> Self {
        let change = ((current - baseline) / baseline) * 100.0;
        Self {
            baseline_mean_ns: baseline,
            current_mean_ns: current,
            change_percent: change,
            significant,
            ci_lower: 0.0,
            ci_upper: 0.0,
        }
    }

    /// Format as display string
    pub fn display(&self) -> String {
        let direction = if self.change_percent > 0.0 {
            "\x1b[31m+\x1b[0m" // Red for slower
        } else {
            "\x1b[32m\x1b[0m" // Green for faster
        };

        let sig = if self.significant { "*" } else { "" };

        format!(
            "{}{:.2}%{} (baseline: {}, current: {})",
            direction,
            self.change_percent.abs(),
            sig,
            format_duration(self.baseline_mean_ns),
            format_duration(self.current_mean_ns),
        )
    }
}

/// Bencher instance passed to benchmark functions
#[derive(Debug)]
pub struct Bencher {
    /// Configuration
    config: BenchConfig,
    /// Collected samples (ns per iteration)
    samples: Vec<f64>,
    /// Total iterations
    iterations: u64,
    /// Whether currently in iteration mode
    in_iter: bool,
}

impl Bencher {
    /// Create a new bencher with default config
    pub fn new() -> Self {
        Self::with_config(BenchConfig::default())
    }

    /// Create a new bencher with custom config
    pub fn with_config(config: BenchConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
            iterations: 0,
            in_iter: false,
        }
    }

    /// Run a closure repeatedly for benchmarking
    pub fn iter<F, R>(&mut self, mut f: F)
    where
        F: FnMut() -> R,
    {
        // Warmup
        for _ in 0..self.config.warmup_iterations {
            std::hint::black_box(f());
        }

        // Determine iteration count for target time
        let iters = self.calibrate(&mut f);

        // Collect samples
        for _ in 0..self.config.sample_count {
            let start = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(f());
            }
            let elapsed = start.elapsed();
            let ns_per_iter = elapsed.as_nanos() as f64 / iters as f64;
            self.samples.push(ns_per_iter);
        }

        self.iterations = iters * self.config.sample_count as u64;
    }

    /// Run a closure with setup for each iteration
    pub fn iter_with_setup<S, F, R>(&mut self, setup: S, mut f: F)
    where
        S: FnMut() -> R,
        F: FnMut(R),
    {
        let mut setup = setup;

        // Warmup
        for _ in 0..self.config.warmup_iterations {
            let input = setup();
            f(input);
            std::hint::black_box(());
        }

        // Collect samples - each sample is a single iteration with setup
        for _ in 0..self.config.sample_count {
            let input = setup();
            let start = Instant::now();
            f(input);
            std::hint::black_box(());
            let elapsed = start.elapsed();
            self.samples.push(elapsed.as_nanos() as f64);
        }

        self.iterations = self.config.sample_count as u64;
    }

    /// Run a closure with batched iterations
    pub fn iter_batched<S, F, T, R>(&mut self, mut setup: S, mut f: F, batch_size: usize)
    where
        S: FnMut() -> Vec<T>,
        F: FnMut(T) -> R,
    {
        // Collect samples
        for _ in 0..self.config.sample_count {
            let inputs = setup();
            let start = Instant::now();
            for item in inputs {
                std::hint::black_box(f(item));
            }
            let elapsed = start.elapsed();
            let ns_per_iter = elapsed.as_nanos() as f64 / batch_size as f64;
            self.samples.push(ns_per_iter);
        }

        self.iterations = self.config.sample_count as u64 * batch_size as u64;
    }

    /// Calibrate iteration count for target time
    fn calibrate<F, R>(&self, f: &mut F) -> u64
    where
        F: FnMut() -> R,
    {
        let mut iters = 1u64;

        loop {
            let start = Instant::now();
            for _ in 0..iters {
                std::hint::black_box(f());
            }
            let elapsed = start.elapsed();

            if elapsed >= Duration::from_millis(100) {
                // Calculate iterations needed for target time per sample
                let target_per_sample = self.config.target_time / self.config.sample_count as u32;
                let estimated = (target_per_sample.as_nanos() as f64 / elapsed.as_nanos() as f64
                    * iters as f64) as u64;
                return estimated
                    .max(self.config.min_iterations)
                    .min(self.config.max_iterations);
            }

            iters = iters.saturating_mul(10);
            if iters > self.config.max_iterations {
                return self.config.min_iterations;
            }
        }
    }

    /// Calculate statistics from collected samples
    pub fn stats(&self) -> BenchStats {
        let mut samples = self.samples.clone();

        // Optionally filter outliers
        let outliers_removed = if self.config.filter_outliers {
            filter_outliers(&mut samples)
        } else {
            0
        };

        if samples.is_empty() {
            return BenchStats {
                mean_ns: 0.0,
                std_dev_ns: 0.0,
                median_ns: 0.0,
                min_ns: 0.0,
                max_ns: 0.0,
                p50_ns: 0.0,
                p95_ns: 0.0,
                p99_ns: 0.0,
                iterations: 0,
                sample_count: 0,
                cv: 0.0,
                outliers_removed,
                throughput: 0.0,
            };
        }

        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = samples.len();
        let mean = samples.iter().sum::<f64>() / n as f64;
        let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        let median = percentile(&samples, 50.0);
        let min = samples[0];
        let max = samples[n - 1];
        let p50 = median;
        let p95 = percentile(&samples, 95.0);
        let p99 = percentile(&samples, 99.0);
        let cv = if mean > 0.0 { std_dev / mean } else { 0.0 };
        let throughput = if mean > 0.0 {
            1_000_000_000.0 / mean
        } else {
            0.0
        };

        BenchStats {
            mean_ns: mean,
            std_dev_ns: std_dev,
            median_ns: median,
            min_ns: min,
            max_ns: max,
            p50_ns: p50,
            p95_ns: p95,
            p99_ns: p99,
            iterations: self.iterations,
            sample_count: samples.len(),
            cv,
            outliers_removed,
            throughput,
        }
    }
}

impl Default for Bencher {
    fn default() -> Self {
        Self::new()
    }
}

/// Calculate percentile from sorted samples
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Filter outliers using IQR method, returns number removed
fn filter_outliers(samples: &mut Vec<f64>) -> usize {
    if samples.len() < 4 {
        return 0;
    }

    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let q1 = percentile(samples, 25.0);
    let q3 = percentile(samples, 75.0);
    let iqr = q3 - q1;
    let lower = q1 - 1.5 * iqr;
    let upper = q3 + 1.5 * iqr;

    let original_len = samples.len();
    samples.retain(|&x| x >= lower && x <= upper);
    original_len - samples.len()
}

/// Benchmark runner for executing benchmark suites
pub struct BenchmarkRunner {
    config: BenchConfig,
    /// Baseline results for comparison
    baselines: HashMap<String, BenchStats>,
}

impl BenchmarkRunner {
    /// Create a new benchmark runner
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            baselines: HashMap::new(),
        }
    }

    /// Load baseline results from a file
    pub fn load_baselines(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let content = std::fs::read_to_string(path)?;
        self.baselines = serde_json::from_str(&content).unwrap_or_default();
        Ok(())
    }

    /// Save baseline results to a file
    pub fn save_baselines(&self, path: &std::path::Path) -> std::io::Result<()> {
        let content = serde_json::to_string_pretty(&self.baselines)?;
        std::fs::write(path, content)
    }

    /// Run a single benchmark
    pub fn run_benchmark(&self, bench: &TestCase) -> BenchResult {
        let mut bencher = Bencher::with_config(self.config.clone());

        // Execute the benchmark function
        // In the actual implementation, this would compile and run the D code
        // For now, we simulate it
        self.execute_bench_function(bench, &mut bencher);

        let stats = bencher.stats();

        let mut result = BenchResult::new(bench.name.clone(), bench.full_name.clone(), stats);

        // Compare with baseline if available
        if let Some(baseline) = self.baselines.get(&bench.full_name) {
            let comparison = BenchComparison::new(
                baseline.mean_ns,
                result.stats.mean_ns,
                (result.stats.mean_ns - baseline.mean_ns).abs() / baseline.std_dev_ns > 2.0,
            );
            result.comparison = Some(comparison);
        }

        result
    }

    /// Execute a benchmark function
    fn execute_bench_function(&self, _bench: &TestCase, bencher: &mut Bencher) {
        // Placeholder - would execute actual D code
        bencher.iter(|| {
            // Simulated workload
            let mut sum = 0u64;
            for i in 0..1000 {
                sum = sum.wrapping_add(i);
            }
            sum
        });
    }

    /// Run all benchmarks and return results
    pub fn run_all(&self, benchmarks: &[TestCase]) -> Vec<BenchResult> {
        println!("\nrunning {} benchmarks\n", benchmarks.len());

        let results: Vec<_> = benchmarks.iter().map(|b| self.run_benchmark(b)).collect();

        // Print results
        for result in &results {
            println!(
                "test {} ... bench: {}",
                result.full_name,
                result.stats.display()
            );
            if let Some(comparison) = &result.comparison {
                println!("     change: {}", comparison.display());
            }
        }

        println!();
        results
    }

    /// Update baselines with new results
    pub fn update_baselines(&mut self, results: &[BenchResult]) {
        for result in results {
            self.baselines
                .insert(result.full_name.clone(), result.stats.clone());
        }
    }
}

impl Default for BenchmarkRunner {
    fn default() -> Self {
        Self::new(BenchConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration() {
        assert!(format_duration(500.0).contains("ns"));
        assert!(format_duration(5000.0).contains("µs"));
        assert!(format_duration(5_000_000.0).contains("ms"));
        assert!(format_duration(5_000_000_000.0).contains("s"));
    }

    #[test]
    fn test_percentile() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        // With 10 elements and rounding, 50th percentile index = round(0.5 * 9) = 5 -> value 6.0
        assert_eq!(percentile(&samples, 50.0), 6.0); // Median (rounded index)
        assert_eq!(percentile(&samples, 0.0), 1.0); // Min
        assert_eq!(percentile(&samples, 100.0), 10.0); // Max
    }

    #[test]
    fn test_filter_outliers() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier
        let removed = filter_outliers(&mut samples);
        assert_eq!(removed, 1);
        assert!(!samples.contains(&100.0));
    }

    #[test]
    #[cfg_attr(windows, ignore)] // Timing can be unreliable on Windows CI
    fn test_bencher_iter() {
        let mut bencher = Bencher::with_config(BenchConfig {
            sample_count: 10,
            warmup_iterations: 1,
            target_time: Duration::from_millis(100),
            ..Default::default()
        });

        bencher.iter(|| {
            let mut x = 0;
            for i in 0..100 {
                x += i;
            }
            x
        });

        let stats = bencher.stats();
        assert!(stats.mean_ns > 0.0);
        assert!(stats.iterations > 0);
    }

    #[test]
    fn test_bench_stats_display() {
        let stats = BenchStats {
            mean_ns: 1000.0,
            std_dev_ns: 100.0,
            median_ns: 950.0,
            min_ns: 800.0,
            max_ns: 1200.0,
            p50_ns: 950.0,
            p95_ns: 1150.0,
            p99_ns: 1190.0,
            iterations: 1000,
            sample_count: 100,
            cv: 0.1,
            outliers_removed: 2,
            throughput: 1_000_000.0,
        };

        let display = stats.display();
        assert!(display.contains("µs"));
    }

    #[test]
    fn test_bench_comparison() {
        let comp = BenchComparison::new(1000.0, 1100.0, true);
        assert_eq!(comp.change_percent, 10.0);
        assert!(comp.display().contains("10.00%"));
    }
}
