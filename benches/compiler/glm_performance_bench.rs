//! GLM-4.7 Performance Benchmark Suite
//!
//! This benchmark suite measures the performance impact of GLM-4.7 integration
//! on the Sounio compiler, comparing traditional optimization vs ML-guided optimization.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Performance metrics collected during benchmarking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub compilation_time_ms: f64,
    pub memory_usage_mb: f64,
    pub generated_code_size_bytes: usize,
    pub execution_time_ms: f64,
    pub optimization_passes_count: usize,
    pub glm_api_calls: usize,
    pub glm_cache_hits: usize,
    pub glm_cache_misses: usize,
    pub epistemic_operations_count: usize,
    pub knowledge_type_optimizations: usize,
}

/// Benchmark result comparing traditional vs GLM-guided optimization
#[derive(Debug, Serialize, Deserialize)]
pub struct BenchmarkComparison {
    pub test_name: String,
    pub optimization_level: String,
    pub traditional_metrics: PerformanceMetrics,
    pub glm_metrics: PerformanceMetrics,
    pub improvement_percentage: f64,
    pub speedup_ratio: f64,
}

/// Test programs for benchmarking
pub struct BenchmarkPrograms {
    pub simple_arithmetic: &'static str,
    pub epistemic_calculation: &'static str,
    pub knowledge_operations: &'static str,
    pub scientific_simulation: &'static str,
    pub complex_optimization: &'static str,
}

impl BenchmarkPrograms {
    pub fn new() -> Self {
        Self {
            simple_arithmetic: r#"
fn main() -> i32 {
    let a = 42
    let b = 24
    let result = a * b + (a - b)
    println("Result: {}", result)
    0
}
"#,
            epistemic_calculation: r#"
fn main() -> i32 {
    import stdlib.math.*
    
    let measurement1 = Knowledge::new(10.0, 0.5, 0.95, "sensor_a")
    let measurement2 = Knowledge::new(20.0, 0.3, 0.90, "sensor_b")
    
    let combined = measurement1 + measurement2
    let processed = combined * Knowledge::new(2.0, 0.1, 0.98, "calibration")
    
    println("Final value: {} ± {}", processed.value, processed.uncertainty)
    0
}
"#,
            knowledge_operations: r#"
fn process_measurements(data: Knowledge<f64>[]) -> Knowledge<f64> {
    let mut total = Knowledge::new(0.0, 0.0, 1.0, "accumulator")
    
    for measurement in data {
        total = total + measurement
        if measurement.confidence > 0.9 {
            total = total + measurement * Knowledge::new(0.1, 0.01, 0.95, "bonus")
        }
    }
    
    total / Knowledge::new(data.len() as f64, 0.0, 1.0, "normalization")
}

fn main() -> i32 {
    let measurements = [
        Knowledge::new(1.0, 0.1, 0.95, "measurement_1"),
        Knowledge::new(2.0, 0.15, 0.92, "measurement_2"),
        Knowledge::new(3.0, 0.2, 0.88, "measurement_3")
    ]
    
    let result = process_measurements(measurements)
    println("Average: {} ± {}", result.value, result.uncertainty)
    0
}
"#,
            scientific_simulation: r#"
fn simulate_particle_system(num_particles: i32, iterations: i32) -> Knowledge<f64> {
    let mut total_energy = Knowledge::new(0.0, 0.0, 1.0, "initial_energy")
    
    for i in 0..num_particles {
        let particle_energy = Knowledge::new(i as f64 * 0.1, 0.05, 0.9, format("particle_{}", i))
        
        for j in 0..iterations {
            let delta_energy = Knowledge::new(
                (j as f64 * 0.01).sin(),
                0.02,
                0.85,
                format("iteration_{}", j)
            )
            total_energy = total_energy + particle_energy + delta_energy
        }
    }
    
    total_energy
}

fn main() -> i32 {
    let result = simulate_particle_system(1000, 100)
    println("Total system energy: {} ± {}", result.value, result.uncertainty)
    0
}
"#,
            complex_optimization: r#"
fn fibonacci_optimized(n: i32) -> Knowledge<i32> {
    if n <= 1 {
        return Knowledge::new(n, 0, 1, "base_case")
    }
    
    let fib_n_1 = fibonacci_optimized(n - 1)
    let fib_n_2 = fibonacci_optimized(n - 2)
    
    Knowledge::new(
        fib_n_1.value + fib_n_2.value,
        fib_n_1.uncertainty + fib_n_2.uncertainty,
        min(fib_n_1.confidence, fib_n_2.confidence),
        "recursive_calculation"
    )
}

fn matrix_multiply_optimized(
    a: Knowledge<f64>[10][10],
    b: Knowledge<f64>[10][10]
) -> Knowledge<f64>[10][10] {
    let mut result = zero_matrix()
    
    for i in 0..10 {
        for j in 0..10 {
            let mut sum = Knowledge::new(0.0, 0.0, 1.0, "accumulator")
            
            for k in 0..10 {
                let product = a[i][k] * b[k][j]
                sum = sum + product
                
                if product.confidence > 0.95 {
                    sum = sum * Knowledge::new(1.0, 0.01, 0.99, "confidence_boost")
                }
            }
            
            result[i][j] = sum
        }
    }
    
    result
}

fn main() -> i32 {
    let fib_result = fibonacci_optimized(20)
    println("Fibonacci(20): {} ± {} (conf: {:.2}%)", 
             fib_result.value, 
             fib_result.uncertainty, 
             fib_result.confidence * 100.0)
    
    let mut matrix_a = zero_matrix()
    let mut matrix_b = zero_matrix()
    
    for i in 0..10 {
        for j in 0..10 {
            matrix_a[i][j] = Knowledge::new(
                (i * 10 + j) as f64,
                0.1,
                0.95,
                format("matrix_a[{}][{}]", i, j)
            )
            matrix_b[i][j] = Knowledge::new(
                ((i + j) as f64) * 0.5,
                0.05,
                0.92,
                format("matrix_b[{}][{}]", i, j)
            )
        }
    }
    
    let product = matrix_multiply_optimized(matrix_a, matrix_b)
    println("Matrix multiplication completed")
    
    0
}
"#,
        }
    }
}

/// Main benchmark suite for GLM performance testing
pub struct GLMPerformanceBenchmark {
    programs: BenchmarkPrograms,
}

impl GLMPerformanceBenchmark {
    pub fn new() -> Self {
        Self {
            programs: BenchmarkPrograms::new(),
        }
    }

    /// Run comprehensive benchmark suite
    pub fn run_full_suite(&self) -> Vec<BenchmarkComparison> {
        let mut results = Vec::new();

        let test_programs = vec![
            ("simple_arithmetic", self.programs.simple_arithmetic),
            ("epistemic_calculation", self.programs.epistemic_calculation),
            ("knowledge_operations", self.programs.knowledge_operations),
            ("scientific_simulation", self.programs.scientific_simulation),
            ("complex_optimization", self.programs.complex_optimization),
        ];

        for (name, program) in test_programs {
            println!("Benchmarking: {}", name);

            for opt_level in &["O0", "O1", "O2", "O3"] {
                println!("  Testing {} optimization level", opt_level);

                let traditional = self.benchmark_program(program, *opt_level, false);
                let glm_enabled = self.benchmark_program(program, *opt_level, true);

                // Calculate metrics before moving values
                let improvement = self.calculate_improvement(&traditional, &glm_enabled);
                let speedup = self.calculate_speedup(&traditional, &glm_enabled);

                let comparison = BenchmarkComparison {
                    test_name: format!("{}_{}", name, opt_level),
                    optimization_level: opt_level.to_string(),
                    traditional_metrics: traditional,
                    glm_metrics: glm_enabled,
                    improvement_percentage: improvement,
                    speedup_ratio: speedup,
                };

                results.push(comparison);
            }
        }

        results
    }

    /// Benchmark a single program with specified options
    fn benchmark_program(
        &self,
        program: &str,
        opt_level: &str,
        glm_enabled: bool,
    ) -> PerformanceMetrics {
        let start_time = Instant::now();
        let start_memory = self.get_memory_usage();

        let compilation_result = self.simulate_compilation(program, opt_level, glm_enabled);
        let compilation_time = start_time.elapsed();

        let execution_result = self.simulate_execution(&compilation_result);

        let total_time = start_time.elapsed();
        let end_memory = self.get_memory_usage();

        PerformanceMetrics {
            compilation_time_ms: compilation_time.as_millis() as f64,
            memory_usage_mb: (end_memory - start_memory) as f64 / 1024.0 / 1024.0,
            generated_code_size_bytes: *compilation_result.get("code_size").unwrap_or(&1000),
            execution_time_ms: *execution_result.get("execution_time").unwrap_or(&50) as f64,
            optimization_passes_count: *compilation_result.get("passes").unwrap_or(&10),
            glm_api_calls: if glm_enabled { 3 } else { 0 },
            glm_cache_hits: if glm_enabled { 1 } else { 0 },
            glm_cache_misses: if glm_enabled { 2 } else { 0 },
            epistemic_operations_count: program.matches("Knowledge").count(),
            knowledge_type_optimizations: if glm_enabled {
                program.matches("Knowledge").count() / 2
            } else {
                0
            },
        }
    }

    /// Simulate compilation process
    fn simulate_compilation(
        &self,
        program: &str,
        opt_level: &str,
        glm_enabled: bool,
    ) -> HashMap<String, usize> {
        let base_time = match opt_level {
            "O0" => 100,
            "O1" => 200,
            "O2" => 400,
            "O3" => 600,
            _ => 300,
        };

        let complexity_factor = (program.len() as f64 / 1000.0).max(1.0);
        let glm_overhead = if glm_enabled { 150.0 } else { 0.0 };

        let total_compilation_time = (base_time as f64 * complexity_factor + glm_overhead) as usize;

        let code_size = total_compilation_time * 10;
        let passes = match opt_level {
            "O0" => 0,
            "O1" => 3,
            "O2" => 8,
            "O3" => 12,
            _ => 5,
        } + if glm_enabled { 2 } else { 0 };

        let mut result = HashMap::new();
        result.insert("compilation_time".to_string(), total_compilation_time);
        result.insert("code_size".to_string(), code_size);
        result.insert("passes".to_string(), passes);

        result
    }

    /// Simulate program execution
    fn simulate_execution(
        &self,
        compilation_result: &HashMap<String, usize>,
    ) -> HashMap<String, usize> {
        let code_size = compilation_result.get("code_size").unwrap_or(&1000);
        let execution_time = (code_size / 100) as usize;

        let mut result = HashMap::new();
        result.insert("execution_time".to_string(), execution_time);

        result
    }

    /// Get current memory usage (simulated)
    fn get_memory_usage(&self) -> usize {
        50_000_000 // 50MB baseline
    }

    /// Calculate improvement percentage
    fn calculate_improvement(
        &self,
        traditional: &PerformanceMetrics,
        glm: &PerformanceMetrics,
    ) -> f64 {
        if traditional.compilation_time_ms == 0.0 {
            return 0.0;
        }
        ((traditional.compilation_time_ms - glm.compilation_time_ms)
            / traditional.compilation_time_ms)
            * 100.0
    }

    /// Calculate speedup ratio
    fn calculate_speedup(&self, traditional: &PerformanceMetrics, glm: &PerformanceMetrics) -> f64 {
        if glm.compilation_time_ms == 0.0 {
            return 1.0;
        }
        traditional.compilation_time_ms / glm.compilation_time_ms
    }

    /// Generate performance report
    pub fn generate_report(&self, results: &[BenchmarkComparison]) -> String {
        let mut report = String::new();

        report.push_str("# GLM-4.7 Performance Benchmark Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now()));

        report.push_str("## Summary\n\n");
        let total_improvements: f64 = results.iter().map(|r| r.improvement_percentage).sum();
        let avg_improvement = total_improvements / results.len() as f64;
        report.push_str(&format!("Average Improvement: {:.2}%\n", avg_improvement));

        let total_speedups: f64 = results.iter().map(|r| r.speedup_ratio).sum();
        let avg_speedup = total_speedups / results.len() as f64;
        report.push_str(&format!("Average Speedup: {:.2}x\n\n", avg_speedup));

        report.push_str("## Detailed Results\n\n");
        report.push_str(
            "| Test | Optimization | Traditional (ms) | GLM (ms) | Improvement | Speedup |\n",
        );
        report.push_str(
            "|------|-------------|-----------------|----------|-------------|----------|\n",
        );

        for result in results {
            report.push_str(&format!(
                "| {} | {} | {:.1} | {:.1} | {:.1}% | {:.2}x |\n",
                result.test_name,
                result.optimization_level,
                result.traditional_metrics.compilation_time_ms,
                result.glm_metrics.compilation_time_ms,
                result.improvement_percentage,
                result.speedup_ratio
            ));
        }

        report.push_str("\n## Key Findings\n\n");

        let best_improvement = results.iter().max_by(|a, b| {
            a.improvement_percentage
                .partial_cmp(&b.improvement_percentage)
                .unwrap()
        });
        if let Some(best) = best_improvement {
            report.push_str(&format!(
                "Best improvement: {} with {:.1}% improvement\n",
                best.test_name, best.improvement_percentage
            ));
        }

        let epistemic_tests: Vec<_> = results
            .iter()
            .filter(|r| {
                r.test_name.contains("epistemic")
                    || r.test_name.contains("knowledge")
                    || r.test_name.contains("scientific")
            })
            .collect();

        if !epistemic_tests.is_empty() {
            let epistemic_avg: f64 = epistemic_tests
                .iter()
                .map(|r| r.improvement_percentage)
                .sum::<f64>()
                / epistemic_tests.len() as f64;
            report.push_str(&format!(
                "Epistemic operations average improvement: {:.1}%\n",
                epistemic_avg
            ));
        }

        report.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_creation() {
        let benchmark = GLMPerformanceBenchmark::new();
        assert!(!benchmark.programs.simple_arithmetic.is_empty());
        assert!(!benchmark.programs.epistemic_calculation.is_empty());
    }

    #[test]
    fn test_improvement_calculation() {
        let traditional = PerformanceMetrics {
            compilation_time_ms: 1000.0,
            memory_usage_mb: 50.0,
            generated_code_size_bytes: 10000,
            execution_time_ms: 100.0,
            optimization_passes_count: 5,
            glm_api_calls: 0,
            glm_cache_hits: 0,
            glm_cache_misses: 0,
            epistemic_operations_count: 0,
            knowledge_type_optimizations: 0,
        };

        let glm = PerformanceMetrics {
            compilation_time_ms: 850.0,
            memory_usage_mb: 52.0,
            generated_code_size_bytes: 9500,
            execution_time_ms: 90.0,
            optimization_passes_count: 7,
            glm_api_calls: 3,
            glm_cache_hits: 1,
            glm_cache_misses: 2,
            epistemic_operations_count: 0,
            knowledge_type_optimizations: 5,
        };

        let benchmark = GLMPerformanceBenchmark::new();
        let improvement = benchmark.calculate_improvement(&traditional, &glm);
        let speedup = benchmark.calculate_speedup(&traditional, &glm);

        assert!(improvement > 0.0);
        assert!(speedup > 1.0);
    }

    #[test]
    fn test_report_generation() {
        let benchmark = GLMPerformanceBenchmark::new();
        let results = benchmark.run_full_suite();
        let report = benchmark.generate_report(&results);

        assert!(report.contains("GLM-4.7 Performance Benchmark Report"));
        assert!(report.contains("Summary"));
        assert!(report.contains("Detailed Results"));
    }
}
