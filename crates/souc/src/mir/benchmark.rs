//! MIR Benchmarking Support
//!
//! Provides benchmarking infrastructure for MIR optimization passes.

use std::time::{Duration, Instant};

/// Statistics from a benchmark run
#[derive(Debug, Clone, Default)]
pub struct BenchmarkStats {
    /// Total time spent in the pass
    pub total_time: Duration,
    /// Number of instructions processed
    pub instructions_processed: usize,
    /// Number of optimizations applied
    pub optimizations_applied: usize,
}

/// Benchmark a closure and return timing stats
pub fn benchmark<F, R>(name: &str, f: F) -> (R, BenchmarkStats)
where
    F: FnOnce() -> R,
{
    let start = Instant::now();
    let result = f();
    let elapsed = start.elapsed();

    let stats = BenchmarkStats {
        total_time: elapsed,
        instructions_processed: 0,
        optimizations_applied: 0,
    };

    if std::env::var("MIR_BENCH_VERBOSE").is_ok() {
        eprintln!("[MIR BENCH] {}: {:?}", name, elapsed);
    }

    (result, stats)
}

/// Benchmark context for tracking multiple passes
#[derive(Debug, Default)]
pub struct BenchmarkContext {
    pub pass_stats: Vec<(String, BenchmarkStats)>,
}

impl BenchmarkContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, name: impl Into<String>, stats: BenchmarkStats) {
        self.pass_stats.push((name.into(), stats));
    }

    pub fn total_time(&self) -> Duration {
        self.pass_stats.iter().map(|(_, s)| s.total_time).sum()
    }

    pub fn report(&self) -> String {
        let mut report = String::new();
        report.push_str("MIR Optimization Benchmark Report\n");
        report.push_str("=================================\n");
        for (name, stats) in &self.pass_stats {
            report.push_str(&format!(
                "{}: {:?} ({} optimizations)\n",
                name, stats.total_time, stats.optimizations_applied
            ));
        }
        report.push_str(&format!("Total: {:?}\n", self.total_time()));
        report
    }
}
