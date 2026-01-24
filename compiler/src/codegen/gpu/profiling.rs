//! GPU Performance Profiling Utilities
//!
//! Provides framework for measuring and comparing optimization feature impacts:
//! - Mixed-Precision Training
//! - Kernel Fusion
//! - Sparse Quaternion Operations
//! - Quantization-Aware Training (QAT)

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

/// Optimization features tracked for profiling
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Feature {
    MixedPrecision,
    Fusion,
    Sparsity,
    QAT,
}

impl fmt::Display for Feature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Feature::MixedPrecision => write!(f, "mixed-precision"),
            Feature::Fusion => write!(f, "fusion"),
            Feature::Sparsity => write!(f, "sparsity"),
            Feature::QAT => write!(f, "qat"),
        }
    }
}

/// Single measurement point
#[derive(Clone, Debug)]
pub struct Measurement {
    pub latency_ms: f64,      // milliseconds
    pub throughput_ops_sec: f64, // operations per second
    pub mem_bw_gbs: f64,      // GB/s
    pub energy_jop: f64,      // joules per operation
}

impl Measurement {
    pub fn new(latency_ms: f64, num_ops: usize, bytes_rw: usize, energy_j: f64) -> Self {
        let sec = latency_ms / 1000.0;
        let throughput = if sec > 0.0 { num_ops as f64 / sec } else { 0.0 };
        let gb = bytes_rw as f64 / 1e9;
        let mem_bw = if sec > 0.0 { gb / sec } else { 0.0 };
        let energy_per_op = if num_ops > 0 {
            energy_j / num_ops as f64
        } else {
            0.0
        };

        Self {
            latency_ms,
            throughput_ops_sec: throughput,
            mem_bw_gbs: mem_bw,
            energy_jop: energy_per_op,
        }
    }
}

/// Statistical summary of measurements
#[derive(Clone, Debug)]
pub struct Stats {
    pub mean: f64,
    pub stddev: f64,
    pub min: f64,
    pub max: f64,
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
}

impl Stats {
    pub fn new(values: &[f64]) -> Self {
        if values.is_empty() {
            return Stats {
                mean: 0.0,
                stddev: 0.0,
                min: 0.0,
                max: 0.0,
                p50: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let n = values.len() as f64;
        let sum: f64 = values.iter().sum();
        let mean = sum / n;

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
        let stddev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = *sorted.first().unwrap_or(&0.0);
        let max = *sorted.last().unwrap_or(&0.0);

        let p50_idx = sorted.len() / 2;
        let p50 = sorted[p50_idx];

        let p95_idx = ((0.95 * (n - 1.0)) as usize).min(sorted.len() - 1);
        let p95 = sorted[p95_idx];

        let p99_idx = ((0.99 * (n - 1.0)) as usize).min(sorted.len() - 1);
        let p99 = sorted[p99_idx];

        Stats {
            mean,
            stddev,
            min,
            max,
            p50,
            p95,
            p99,
        }
    }

    pub fn display(&self, metric_name: &str) -> String {
        format!(
            "{}: mean={:.2}, stddev={:.2}, min={:.2}, max={:.2}, p50={:.2}, p95={:.2}, p99={:.2}",
            metric_name, self.mean, self.stddev, self.min, self.max, self.p50, self.p95, self.p99
        )
    }
}

/// Result of a profiling run
#[derive(Clone, Debug)]
pub struct ProfileResult {
    pub features: Vec<Feature>,
    pub latencies: Vec<f64>,
    pub throughputs: Vec<f64>,
    pub mem_bws: Vec<f64>,
    pub energies: Vec<f64>,
    pub stats: HashMap<String, Stats>,
}

impl ProfileResult {
    pub fn new() -> Self {
        Self {
            features: vec![],
            latencies: vec![],
            throughputs: vec![],
            mem_bws: vec![],
            energies: vec![],
            stats: HashMap::new(),
        }
    }

    pub fn compute_stats(&mut self) {
        self.stats.insert(
            "latency_ms".to_string(),
            Stats::new(&self.latencies),
        );
        self.stats.insert(
            "throughput_ops_sec".to_string(),
            Stats::new(&self.throughputs),
        );
        self.stats.insert(
            "mem_bw_gbs".to_string(),
            Stats::new(&self.mem_bws),
        );
        self.stats.insert(
            "energy_jop".to_string(),
            Stats::new(&self.energies),
        );
    }

    pub fn display_summary(&self) -> String {
        let mut output = String::new();
        output.push_str("=== Profile Summary ===\n");
        output.push_str(&format!("Features: {:?}\n", self.features));

        for (metric, stats) in &self.stats {
            output.push_str(&format!("{}\n", stats.display(metric)));
        }

        output
    }
}

/// Comparison between baseline and optimized results
#[derive(Clone, Debug)]
pub struct Comparison {
    pub metric: String,
    pub baseline: f64,
    pub optimized: f64,
    pub improvement_pct: f64, // Positive = better
    pub is_better: bool,
}

impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let sign = if self.improvement_pct > 0.0 { "+" } else { "" };
        write!(
            f,
            "{}: baseline={:.2}, optimized={:.2}, {}improvement={:.1}%",
            self.metric, self.baseline, self.optimized, sign, self.improvement_pct
        )
    }
}

/// Profiler configuration
pub struct Profiler {
    warmup_runs: usize,
    measure_runs: usize,
}

impl Profiler {
    pub fn new(warmup: usize, measure: usize) -> Self {
        Self {
            warmup_runs: warmup,
            measure_runs: measure,
        }
    }

    /// Profile a generic operation with feature tracking
    pub fn profile<F>(
        &self,
        num_ops: usize,
        bytes_rw: usize,
        mut closure: F,
        features: Vec<Feature>,
    ) -> ProfileResult
    where
        F: FnMut() -> f64, // Closure returns energy in joules
    {
        // Warmup
        for _ in 0..self.warmup_runs {
            let _ = closure();
        }

        let mut result = ProfileResult::new();
        result.features = features;

        // Measurements
        for _ in 0..self.measure_runs {
            let start = Instant::now();
            let energy = closure();
            let duration = start.elapsed();

            let lat_ms = duration.as_secs_f64() * 1000.0;
            let measurement = Measurement::new(lat_ms, num_ops, bytes_rw, energy);

            result.latencies.push(measurement.latency_ms);
            result.throughputs.push(measurement.throughput_ops_sec);
            result.mem_bws.push(measurement.mem_bw_gbs);
            result.energies.push(measurement.energy_jop);
        }

        result.compute_stats();
        result
    }

    /// Compare baseline vs optimized results
    pub fn compare(
        &self,
        baseline: &ProfileResult,
        optimized: &ProfileResult,
    ) -> Vec<Comparison> {
        let mut comparisons = vec![];

        let metrics = [
            ("latency_ms", false), // Lower is better
            ("throughput_ops_sec", true), // Higher is better
            ("mem_bw_gbs", true), // Higher is better
            ("energy_jop", false), // Lower is better
        ];

        for (metric, higher_better) in &metrics {
            if let (Some(b_stats), Some(o_stats)) = (baseline.stats.get(*metric), optimized.stats.get(*metric)) {
                let b_val = b_stats.mean;
                let o_val = o_stats.mean;

                let improvement_pct = if b_val != 0.0 {
                    if *higher_better {
                        100.0 * (o_val - b_val) / b_val
                    } else {
                        100.0 * (b_val - o_val) / b_val
                    }
                } else {
                    0.0
                };

                let is_better = if *higher_better {
                    o_val > b_val
                } else {
                    o_val < b_val
                };

                comparisons.push(Comparison {
                    metric: metric.to_string(),
                    baseline: b_val,
                    optimized: o_val,
                    improvement_pct,
                    is_better,
                });
            }
        }

        comparisons
    }
}

/// Export results to CSV for analysis
pub fn export_csv(result: &ProfileResult, path: &str) -> Result<(), Box<dyn std::error::Error>> {
    use std::fs::File;
    use std::io::Write;

    let mut file = File::create(path)?;

    // Header
    writeln!(
        file,
        "metric,mean,stddev,min,max,p50,p95,p99"
    )?;

    // Data rows
    for (metric, stats) in &result.stats {
        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            metric, stats.mean, stats.stddev, stats.min, stats.max, stats.p50, stats.p95, stats.p99
        )?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stats_computation() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = Stats::new(&values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.p50, 3.0);
    }

    #[test]
    fn test_profiler_basic() {
        let profiler = Profiler::new(1, 3);
        let mut counter = 0;

        let result = profiler.profile(
            100,  // 100 ops
            1000, // 1000 bytes
            || {
                counter += 1;
                0.01 // 0.01 joules
            },
            vec![Feature::MixedPrecision],
        );

        assert_eq!(result.latencies.len(), 3);
        assert!(!result.throughputs.is_empty());
    }

    #[test]
    fn test_comparison() {
        let mut baseline = ProfileResult::new();
        baseline.latencies = vec![10.0, 10.5, 9.8];
        baseline.compute_stats();

        let mut optimized = ProfileResult::new();
        optimized.latencies = vec![5.0, 5.2, 4.9];
        optimized.compute_stats();

        let profiler = Profiler::new(1, 1);
        let comparisons = profiler.compare(&baseline, &optimized);

        let lat_comp = comparisons
            .iter()
            .find(|c| c.metric == "latency_ms")
            .unwrap();
        assert!(lat_comp.is_better);
        assert!(lat_comp.improvement_pct > 40.0); // ~50% improvement
    }
}
