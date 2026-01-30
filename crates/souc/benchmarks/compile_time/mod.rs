//! Compile-Time Benchmarks for L0 Foundations
//!
//! This module provides benchmarks to verify the performance targets
//! for Day 51's L0 foundations:
//!
//! ## Performance Targets
//!
//! | Metric                    | Target         | Description                    |
//! |---------------------------|----------------|--------------------------------|
//! | Ontology lookup (L1)      | < 100ns        | Hot cache hit                  |
//! | Ontology lookup (L3)      | < 10μs         | Cold store access              |
//! | Term creation             | < 50ns         | CompactTerm construction       |
//! | Multiplicity check        | < 5ns          | QTT operations                 |
//! | Erasure analysis          | < 1ms/1K types | Batch erasure decisions        |
//! | Memory per term           | ≤ 64 bytes     | CompactTerm size               |
//! | L1 capacity               | 10K terms      | Hot cache size                 |
//! | L2 capacity               | 100K terms     | Warm cache size                |
//! | L3 capacity               | 15M terms      | Full ontology                  |

use std::sync::Arc;
use std::time::{Duration, Instant};

/// Benchmark result
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Name of the benchmark
    pub name: String,
    /// Number of iterations
    pub iterations: u64,
    /// Total time
    pub total_time: Duration,
    /// Time per operation
    pub time_per_op: Duration,
    /// Target time per operation
    pub target: Duration,
    /// Whether the benchmark passed
    pub passed: bool,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(
        name: impl Into<String>,
        iterations: u64,
        total_time: Duration,
        target: Duration,
    ) -> Self {
        let time_per_op = total_time / iterations as u32;
        let passed = time_per_op <= target;

        BenchmarkResult {
            name: name.into(),
            iterations,
            total_time,
            time_per_op,
            target,
            passed,
        }
    }

    /// Get speedup/slowdown factor
    pub fn factor(&self) -> f64 {
        self.target.as_nanos() as f64 / self.time_per_op.as_nanos() as f64
    }
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let status = if self.passed { "PASS" } else { "FAIL" };
        let factor = self.factor();

        write!(
            f,
            "[{}] {}: {:?}/op (target: {:?}, {:.2}x)",
            status, self.name, self.time_per_op, self.target, factor
        )
    }
}

/// Benchmark suite for L0 foundations
pub struct L0Benchmarks {
    results: Vec<BenchmarkResult>,
}

impl L0Benchmarks {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        L0Benchmarks {
            results: Vec::new(),
        }
}

    /// Run all benchmarks
    pub fn run_all(&mut self) {
        self.bench_multiplicity_operations();
        self.bench_compact_term_creation();
        self.bench_l1_cache_lookup();
        self.bench_l2_cache_lookup();
        self.bench_l3_store_lookup();
        self.bench_erasure_analysis();
        self.bench_epistemic_runtime();
        self.bench_soa_operations();
    }

    /// Benchmark multiplicity operations
    pub fn bench_multiplicity_operations(&mut self) {
        use crate::types::multiplicity::Multiplicity;

        const ITERATIONS: u64 = 1_000_000;
        const TARGET: Duration = Duration::from_nanos(5);

        let start = Instant::now();

        for _ in 0..ITERATIONS {
            let a = Multiplicity::One;
            let b = Multiplicity::Many;
            let _ = a + b;
            let _ = a * b;
            let _ = a.is_runtime_relevant();
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "multiplicity_ops",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark compact term creation
    pub fn bench_compact_term_creation(&mut self) {
        use crate::ontology::memory::compact::CompactTermBuilder;

        const ITERATIONS: u64 = 100_000;
        const TARGET: Duration = Duration::from_nanos(50);

        let start = Instant::now();

        for i in 0..ITERATIONS {
            let _ = CompactTermBuilder::new(&format!("snomed:{}", i))
                .with_label("Test term")
                .with_depth(5)
                .build();
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "compact_term_creation",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark L1 cache lookup
    pub fn bench_l1_cache_lookup(&mut self) {
        use crate::ontology::memory::{L1Cache, compact::CompactTermBuilder};

        const ITERATIONS: u64 = 100_000;
        const TARGET: Duration = Duration::from_nanos(100);

        let cache = L1Cache::with_capacity(1000);

        // Pre-populate cache
        for i in 0..1000 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            cache.insert(format!("test:{}", i), term);
        }

        let start = Instant::now();

        for i in 0..ITERATIONS {
            let key = format!("test:{}", i % 1000);
            let _ = cache.get(&key);
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "l1_cache_lookup",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark L2 cache lookup
    pub fn bench_l2_cache_lookup(&mut self) {
        use crate::ontology::memory::{L2Cache, compact::CompactTermBuilder};

        const ITERATIONS: u64 = 100_000;
        const TARGET: Duration = Duration::from_micros(1);

        let cache = L2Cache::with_capacity(10000);

        // Pre-populate cache
        for i in 0..10000 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            cache.insert(format!("test:{}", i), term);
        }

        let start = Instant::now();

        for i in 0..ITERATIONS {
            let key = format!("test:{}", i % 10000);
            let _ = cache.get(&key);
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "l2_cache_lookup",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark L3 store lookup
    pub fn bench_l3_store_lookup(&mut self) {
        use crate::ontology::memory::{L3Store, compact::CompactTermBuilder};

        const ITERATIONS: u64 = 10_000;
        const TARGET: Duration = Duration::from_micros(10);

        let store = L3Store::new();

        // Pre-populate store
        for i in 0..100000 {
            let term = Arc::new(CompactTermBuilder::new(&format!("test:{}", i)).build());
            store.insert(format!("test:{}", i), term);
        }

        let start = Instant::now();

        for i in 0..ITERATIONS {
            let key = format!("test:{}", i % 100000);
            let _ = store.get(&key);
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "l3_store_lookup",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark erasure analysis
    pub fn bench_erasure_analysis(&mut self) {
        use crate::types::erasure::ErasureAnalyzer;

        const ITERATIONS: u64 = 1000;
        const TARGET: Duration = Duration::from_millis(1);

        let start = Instant::now();

        for _ in 0..ITERATIONS {
            let mut analyzer = ErasureAnalyzer::new();

            // Analyze 1000 ontological types
            for i in 0..1000 {
                analyzer.analyze_ontological(&format!("snomed:{}", i));
            }

            let _ = analyzer.stats();
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "erasure_analysis_1k",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark epistemic runtime operations
    pub fn bench_epistemic_runtime(&mut self) {
        use crate::runtime::epistemic::{CompactKnowledge, FullKnowledge, RuntimeConfidence};

        const ITERATIONS: u64 = 100_000;
        const TARGET: Duration = Duration::from_nanos(100);

        let start = Instant::now();

        for _ in 0..ITERATIONS {
            // Create and manipulate compact knowledge
            let k1: CompactKnowledge<i32> = CompactKnowledge::new(42, 0.95);
            let k2: CompactKnowledge<i32> = CompactKnowledge::new(10, 0.9);
            let _ = k1.combine(k2, |a, b| a + b);
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "epistemic_compact_ops",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Benchmark SoA operations
    pub fn bench_soa_operations(&mut self) {
        use crate::runtime::gpu_epistemic::{SoAKnowledge, simd_ops};

        const ITERATIONS: u64 = 1000;
        const TARGET: Duration = Duration::from_micros(100);

        // Create large SoA
        let mut soa: SoAKnowledge<f64> = SoAKnowledge::with_capacity(10000);
        for i in 0..10000 {
            soa.push(i as f64, 0.9, 0, 0);
        }

        let start = Instant::now();

        for _ in 0..ITERATIONS {
            let mut confidences = soa.confidences.clone();
            simd_ops::scale_confidences(&mut confidences, 0.99);
            let _ = simd_ops::mean_confidence(&confidences);
            let _ = simd_ops::count_above_threshold(&confidences, 0.5);
        }

        let elapsed = start.elapsed();
        self.results.push(BenchmarkResult::new(
            "soa_bulk_ops_10k",
            ITERATIONS,
            elapsed,
            TARGET,
        ));
    }

    /// Get all results
    pub fn results(&self) -> &[BenchmarkResult] {
        &self.results
    }

    /// Check if all benchmarks passed
    pub fn all_passed(&self) -> bool {
        self.results.iter().all(|r| r.passed)
    }

    /// Print summary
    pub fn print_summary(&self) {
        println!("\n=== L0 Foundations Benchmark Summary ===\n");

        for result in &self.results {
            println!("{}", result);
        }

        let passed = self.results.iter().filter(|r| r.passed).count();
        let total = self.results.len();

        println!("\n{}/{} benchmarks passed", passed, total);

        if self.all_passed() {
            println!("All performance targets met!");
        } else {
            println!("Some performance targets not met.");
        }
    }
}

impl Default for L0Benchmarks {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory usage benchmarks
pub struct MemoryBenchmarks {
    results: Vec<MemoryResult>,
}

/// Memory benchmark result
#[derive(Debug, Clone)]
pub struct MemoryResult {
    pub name: String,
    pub actual_bytes: usize,
    pub target_bytes: usize,
    pub passed: bool,
}

impl MemoryBenchmarks {
    pub fn new() -> Self {
        MemoryBenchmarks {
            results: Vec::new(),
        }
}

    pub fn run_all(&mut self) {
        self.bench_compact_term_size();
        self.bench_full_knowledge_size();
        self.bench_compact_knowledge_size();
    }

    fn bench_compact_term_size(&mut self) {
        use crate::ontology::memory::compact::CompactTerm;

        let actual = std::mem::size_of::<CompactTerm>();
        let target = 64;

        self.results.push(MemoryResult {
            name: "CompactTerm".to_string(),
            actual_bytes: actual,
            target_bytes: target,
            passed: actual <= target,
        });
    }

    fn bench_full_knowledge_size(&mut self) {
        use crate::runtime::epistemic::FullKnowledge;

        // FullKnowledge<i64> should be around 64 bytes overhead
        let actual = std::mem::size_of::<FullKnowledge<i64>>();
        let target = 128; // 64 bytes overhead + value

        self.results.push(MemoryResult {
            name: "FullKnowledge<i64>".to_string(),
            actual_bytes: actual,
            target_bytes: target,
            passed: actual <= target,
        });
    }

    fn bench_compact_knowledge_size(&mut self) {
        use crate::runtime::epistemic::CompactKnowledge;

        let actual = std::mem::size_of::<CompactKnowledge<i64>>();
        let target = 24; // 16 bytes overhead + 8 byte value

        self.results.push(MemoryResult {
            name: "CompactKnowledge<i64>".to_string(),
            actual_bytes: actual,
            target_bytes: target,
            passed: actual <= target,
        });
    }

    pub fn print_summary(&self) {
        println!("\n=== Memory Usage Summary ===\n");

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            println!(
                "[{}] {}: {} bytes (target: {} bytes)",
                status, result.name, result.actual_bytes, result.target_bytes
            );
        }
    }
}

impl Default for MemoryBenchmarks {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::new(
            "test",
            1000,
            Duration::from_micros(50),
            Duration::from_nanos(100),
        );

        assert_eq!(result.time_per_op, Duration::from_nanos(50));
        assert!(result.passed);
    }

    #[test]
    fn test_memory_benchmarks() {
        let mut benches = MemoryBenchmarks::new();
        benches.run_all();

        // CompactTerm should fit in 64 bytes
        let compact_term = benches
            .results
            .iter()
            .find(|r| r.name == "CompactTerm")
            .unwrap();
        assert!(compact_term.passed, "CompactTerm should be <= 64 bytes");
    }
}
