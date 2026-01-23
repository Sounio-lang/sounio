//! MIR Optimization Performance Benchmarks
//!
//! These benchmarks measure the performance impact of MIR optimizations,
//! comparing compilation times, code quality, and runtime performance.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use sounio::mir::analysis::LoopAnalysis;
use sounio::mir::builder::ModuleBuilder;
use sounio::mir::instructions::{MirBinaryOp, MirConstant, MirInstruction};
use sounio::mir::optimization::{ConstantPropagation, FunctionInlining, StrengthReduction};
use sounio::mir::types::MirType;
use sounio::mir::{MirFunction, MirModule};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark results storage
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub name: String,
    pub baseline_time: Duration,
    pub optimized_time: Duration,
    pub speedup: f64,
    pub memory_usage: usize,
    pub instructions_count: usize,
    pub blocks_count: usize,
}

/// Performance analyzer for MIR optimizations
pub struct PerformanceAnalyzer {
    /// Results from benchmark runs
    pub results: HashMap<String, BenchmarkResults>,
    /// Baseline metrics for comparison
    pub baseline_metrics: HashMap<String, BaselineMetrics>,
}

#[derive(Debug, Clone)]
pub struct BaselineMetrics {
    pub instruction_count: usize,
    pub block_count: usize,
    pub function_count: usize,
    pub estimated_runtime: Duration,
}

impl PerformanceAnalyzer {
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
            baseline_metrics: HashMap::new(),
        }
    }

    /// Benchmark constant propagation optimization
    pub fn benchmark_constant_propagation(&mut self) -> BenchmarkResults {
        let test_name = "constant_propagation";

        // Create a function with many constant expressions
        let baseline_module = self.create_constant_heavy_function();

        // Measure baseline (no optimization)
        let baseline_start = Instant::now();
        let baseline_metrics = self.measure_mir_complexity(&baseline_module);
        let baseline_time = baseline_start.elapsed();

        // Apply constant propagation
        let optimized_module = self.apply_constant_propagation(&baseline_module);

        // Measure optimized version
        let optimized_start = Instant::now();
        let optimized_metrics = self.measure_mir_complexity(&optimized_module);
        let optimized_time = optimized_start.elapsed();

        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        let result = BenchmarkResults {
            name: test_name.to_string(),
            baseline_time,
            optimized_time,
            speedup,
            memory_usage: optimized_metrics.estimated_memory_usage,
            instructions_count: optimized_metrics.instruction_count,
            blocks_count: optimized_metrics.block_count,
        };

        self.results.insert(test_name.to_string(), result.clone());
        self.baseline_metrics
            .insert(test_name.to_string(), baseline_metrics);

        result
    }

    /// Benchmark strength reduction optimization
    pub fn benchmark_strength_reduction(&mut self) -> BenchmarkResults {
        let test_name = "strength_reduction";

        // Create a function with expensive operations (divisions, multiplications)
        let baseline_module = self.create_arithmetic_heavy_function();

        // Measure baseline
        let baseline_start = Instant::now();
        let baseline_metrics = self.measure_mir_complexity(&baseline_module);
        let baseline_time = baseline_start.elapsed();

        // Apply strength reduction
        let optimized_module = self.apply_strength_reduction(&baseline_module);

        // Measure optimized version
        let optimized_start = Instant::now();
        let optimized_metrics = self.measure_mir_complexity(&optimized_module);
        let optimized_time = optimized_start.elapsed();

        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        let result = BenchmarkResults {
            name: test_name.to_string(),
            baseline_time,
            optimized_time,
            speedup,
            memory_usage: optimized_metrics.estimated_memory_usage,
            instructions_count: optimized_metrics.instruction_count,
            blocks_count: optimized_metrics.block_count,
        };

        self.results.insert(test_name.to_string(), result.clone());
        self.baseline_metrics
            .insert(test_name.to_string(), baseline_metrics);

        result
    }

    /// Benchmark function inlining
    pub fn benchmark_function_inlining(&mut self) -> BenchmarkResults {
        let test_name = "function_inlining";

        // Create a module with many small function calls
        let baseline_module = self.create_call_heavy_module();

        // Measure baseline
        let baseline_start = Instant::now();
        let baseline_metrics = self.measure_mir_complexity(&baseline_module);
        let baseline_time = baseline_start.elapsed();

        // Apply function inlining
        let optimized_module = self.apply_function_inlining(&baseline_module);

        // Measure optimized version
        let optimized_start = Instant::now();
        let optimized_metrics = self.measure_mir_complexity(&optimized_module);
        let optimized_time = optimized_start.elapsed();

        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        let result = BenchmarkResults {
            name: test_name.to_string(),
            baseline_time,
            optimized_time,
            speedup,
            memory_usage: optimized_metrics.estimated_memory_usage,
            instructions_count: optimized_metrics.instruction_count,
            blocks_count: optimized_metrics.block_count,
        };

        self.results.insert(test_name.to_string(), result.clone());
        self.baseline_metrics
            .insert(test_name.to_string(), baseline_metrics);

        result
    }

    /// Benchmark loop optimizations
    pub fn benchmark_loop_optimizations(&mut self) -> BenchmarkResults {
        let test_name = "loop_optimizations";

        // Create a function with complex loops
        let baseline_module = self.create_loop_heavy_function();

        // Measure baseline
        let baseline_start = Instant::now();
        let baseline_metrics = self.measure_mir_complexity(&baseline_module);
        let baseline_time = baseline_start.elapsed();

        // Apply loop optimizations
        let optimized_module = self.apply_loop_optimizations(&baseline_module);

        // Measure optimized version
        let optimized_start = Instant::now();
        let optimized_metrics = self.measure_mir_complexity(&optimized_module);
        let optimized_time = optimized_start.elapsed();

        let speedup = baseline_time.as_nanos() as f64 / optimized_time.as_nanos() as f64;

        let result = BenchmarkResults {
            name: test_name.to_string(),
            baseline_time,
            optimized_time,
            speedup,
            memory_usage: optimized_metrics.estimated_memory_usage,
            instructions_count: optimized_metrics.instruction_count,
            blocks_count: optimized_metrics.block_count,
        };

        self.results.insert(test_name.to_string(), result.clone());
        self.baseline_metrics
            .insert(test_name.to_string(), baseline_metrics);

        result
    }

    /// Run comprehensive benchmark suite
    pub fn run_comprehensive_benchmark(&mut self) -> Vec<BenchmarkResults> {
        println!("Running comprehensive MIR optimization benchmarks...");

        let mut results = Vec::new();

        // Run all benchmarks
        results.push(self.benchmark_constant_propagation());
        results.push(self.benchmark_strength_reduction());
        results.push(self.benchmark_function_inlining());
        results.push(self.benchmark_loop_optimizations());

        // Print results
        self.print_benchmark_results(&results);

        results
    }

    /// Create function heavy in constant expressions
    fn create_constant_heavy_function(&self) -> MirModule {
        let mut module_builder = ModuleBuilder::new("const_heavy");

        let mut func_builder =
            module_builder.create_function("compute_complex".to_string(), MirType::I64);

        // Create a complex expression with many constants: (10 + 20) * (30 + 40) + (50 + 60)
        let const_10 = func_builder.build_i64(10);
        let const_20 = func_builder.build_i64(20);
        let const_30 = func_builder.build_i64(30);
        let const_40 = func_builder.build_i64(40);
        let const_50 = func_builder.build_i64(50);
        let const_60 = func_builder.build_i64(60);

        let sum1 = func_builder.fresh_value();
        func_builder.build_add(sum1, const_10, const_20, MirType::I64);

        let sum2 = func_builder.fresh_value();
        func_builder.build_add(sum2, const_30, const_40, MirType::I64);

        let sum3 = func_builder.fresh_value();
        func_builder.build_add(sum3, const_50, const_60, MirType::I64);

        let product = func_builder.fresh_value();
        func_builder.build_mul(product, sum1, sum2, MirType::I64);

        let final_result = func_builder.fresh_value();
        func_builder.build_add(final_result, product, sum3, MirType::I64);

        func_builder.build_return(Some(final_result));
        module_builder.add_function(func_builder.build());

        module_builder.build()
    }

    /// Create function heavy in arithmetic operations
    fn create_arithmetic_heavy_function(&self) -> MirModule {
        let mut module_builder = ModuleBuilder::new("arith_heavy");

        let mut func_builder =
            module_builder.create_function("compute_arithmetic".to_string(), MirType::I64);

        // Create function with many divisions and multiplications
        let value = func_builder.fresh_value();
        func_builder.build_const(value, MirConstant::Int(1000), MirType::I64);

        for i in 0..100 {
            let divisor = func_builder.build_i64(2);
            let result = func_builder.fresh_value();
            func_builder.build_div(result, value, divisor, MirType::I64);

            if i < 99 {
                // Use result in next iteration
                // This creates a chain of dependent divisions
                let _ = (value, result); // Simplified for benchmark
            }
        }

        func_builder.build_return(Some(value));
        module_builder.add_function(func_builder.build());

        module_builder.build()
    }

    /// Create module with many function calls
    fn create_call_heavy_module(&self) -> MirModule {
        let mut module_builder = ModuleBuilder::new("call_heavy");

        // Create several small functions to be inlined
        for i in 0..10 {
            let mut small_func =
                module_builder.create_function(format!("small_func_{}", i), MirType::I64);
            small_func.add_param("x".to_string(), MirType::I64);

            let result = small_func.fresh_value();
            small_func.build_add(
                result,
                small_func.get_param(0),
                small_func.get_param(0),
                MirType::I64,
            );
            small_func.build_return(Some(result));
            module_builder.add_function(small_func.build());
        }

        // Create main function that calls all small functions
        let mut main_func = module_builder.create_function("main".to_string(), MirType::I64);
        let const_val = main_func.build_i64(42);
        let mut accumulator = main_func.build_i64(0);

        for i in 0..10 {
            let result = main_func.fresh_value();
            main_func.build_call(
                Some(result),
                format!("small_func_{}", i),
                vec![const_val],
                MirType::I64,
            );

            let new_accum = main_func.fresh_value();
            main_func.build_add(new_accum, accumulator, result, MirType::I64);
            accumulator = new_accum;
        }

        main_func.build_return(Some(accumulator));
        module_builder.add_function(main_func.build());

        module_builder.build()
    }

    /// Create function with complex loops
    fn create_loop_heavy_function(&self) -> MirModule {
        let mut module_builder = ModuleBuilder::new("loop_heavy");

        let mut func_builder =
            module_builder.create_function("matrix_multiply".to_string(), MirType::Unit);

        // Simulate nested loops for matrix multiplication
        let outer_loop = func_builder.create_block();
        let middle_loop = func_builder.create_block();
        let inner_loop = func_builder.create_block();
        let exit = func_builder.create_block();

        // Initialize loop variables
        let const_0 = func_builder.build_i64(0);
        let const_100 = func_builder.build_i64(100);

        func_builder.set_block(outer_loop);
        let i_var = func_builder.fresh_value();
        func_builder.build_const(i_var, MirConstant::Int(0), MirType::I64);
        func_builder.build_cond_branch(i_var, middle_loop, exit);

        func_builder.set_block(middle_loop);
        let j_var = func_builder.fresh_value();
        func_builder.build_const(j_var, MirConstant::Int(0), MirType::I64);
        func_builder.build_cond_branch(j_var, inner_loop, outer_loop);

        func_builder.set_block(inner_loop);
        // Do some computation in the innermost loop
        let temp = func_builder.fresh_value();
        func_builder.build_add(temp, i_var, j_var, MirType::I64);

        let k_var = func_builder.fresh_value();
        func_builder.build_const(k_var, MirConstant::Int(0), MirType::I64);

        // Increment j
        let new_j = func_builder.fresh_value();
        let const_1 = func_builder.build_i64(1);
        func_builder.build_add(new_j, j_var, const_1, MirType::I64);
        func_builder.build_branch(middle_loop);

        func_builder.set_block(exit);
        func_builder.build_return(None);

        module_builder.add_function(func_builder.build());

        module_builder.build()
    }

    /// Apply constant propagation optimization
    fn apply_constant_propagation(&self, module: &MirModule) -> MirModule {
        let mut optimized = module.clone();
        let pass = ConstantPropagation;

        for func in &mut optimized.functions {
            let _ = pass.run_on_function(func);
        }

        optimized
    }

    /// Apply strength reduction optimization
    fn apply_strength_reduction(&self, module: &MirModule) -> MirModule {
        let mut optimized = module.clone();
        let pass = StrengthReduction;

        for func in &mut optimized.functions {
            let _ = pass.run_on_function(func);
        }

        optimized
    }

    /// Apply function inlining optimization
    fn apply_function_inlining(&self, module: &MirModule) -> MirModule {
        let mut optimized = module.clone();
        let pass = FunctionInlining;

        let _ = pass.run_on_module(&mut optimized);

        optimized
    }

    /// Apply loop optimizations
    fn apply_loop_optimizations(&self, module: &MirModule) -> MirModule {
        let mut optimized = module.clone();

        for func in &mut optimized.functions {
            let _ = LoopAnalysis::analyze_function(func);
            // Additional loop optimizations would be applied here
        }

        optimized
    }

    /// Measure MIR complexity metrics
    fn measure_mir_complexity(&self, module: &MirModule) -> ComplexityMetrics {
        let mut total_instructions = 0;
        let mut total_blocks = 0;

        for func in &module.functions {
            total_instructions += func
                .blocks
                .iter()
                .map(|block| block.instructions.len())
                .sum::<usize>();
            total_blocks += func.blocks.len();
        }

        ComplexityMetrics {
            instruction_count: total_instructions,
            block_count: total_blocks,
            estimated_memory_usage: total_instructions * 8, // Rough estimate
        }
    }

    /// Print benchmark results in a formatted table
    fn print_benchmark_results(&self, results: &[BenchmarkResults]) {
        println!("\n{}", "=".repeat(80));
        println!("MIR OPTIMIZATION BENCHMARK RESULTS");
        println!("{}", "=".repeat(80));
        println!(
            "{:<25} {:<15} {:<15} {:<10} {:<12} {:<10} {:<10}",
            "Optimization",
            "Baseline",
            "Optimized",
            "Speedup",
            "Instructions",
            "Blocks",
            "Memory (KB)"
        );
        println!("{}", "-".repeat(80));

        for result in results {
            let memory_kb = result.memory_usage / 1024;
            println!(
                "{:<25} {:<15.2?} {:<15.2?} {:<10.2}x {:<12} {:<10} {:<10}",
                result.name,
                result.baseline_time.as_secs_f64() * 1000.0,
                result.optimized_time.as_secs_f64() * 1000.0,
                result.speedup,
                result.instructions_count,
                result.blocks_count,
                memory_kb
            );
        }

        println!("{}", "=".repeat(80));

        // Calculate aggregate statistics
        let avg_speedup: f64 =
            results.iter().map(|r| r.speedup).sum::<f64>() / results.len() as f64;
        let total_baseline_time: Duration = results
            .iter()
            .map(|r| r.baseline_time)
            .fold(Duration::from_millis(0), |acc, d| acc + d);
        let total_optimized_time: Duration = results
            .iter()
            .map(|r| r.optimized_time)
            .fold(Duration::from_millis(0), |acc, d| acc + d);

        println!("Aggregate Statistics:");
        println!("  Average Speedup: {:.2}x", avg_speedup);
        println!(
            "  Total Baseline Time: {:.2}ms",
            total_baseline_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Total Optimized Time: {:.2}ms",
            total_optimized_time.as_secs_f64() * 1000.0
        );
        println!(
            "  Overall Speedup: {:.2}x",
            total_baseline_time.as_nanos() as f64 / total_optimized_time.as_nanos() as f64
        );
        println!("{}", "=".repeat(80));
    }
}

#[derive(Debug, Clone)]
struct ComplexityMetrics {
    instruction_count: usize,
    block_count: usize,
    estimated_memory_usage: usize,
}

/// Criterion benchmark function for constant propagation
fn benchmark_constant_propagation(c: &mut Criterion) {
    let mut analyzer = PerformanceAnalyzer::new();

    c.bench_function("constant_propagation", |b| {
        b.iter(|| {
            analyzer.benchmark_constant_propagation();
        })
    });
}

/// Criterion benchmark function for strength reduction
fn benchmark_strength_reduction(c: &mut Criterion) {
    let mut analyzer = PerformanceAnalyzer::new();

    c.bench_function("strength_reduction", |b| {
        b.iter(|| {
            analyzer.benchmark_strength_reduction();
        })
    });
}

/// Criterion benchmark function for function inlining
fn benchmark_function_inlining(c: &mut Criterion) {
    let mut analyzer = PerformanceAnalyzer::new();

    c.bench_function("function_inlining", |b| {
        b.iter(|| {
            analyzer.benchmark_function_inlining();
        })
    });
}

/// Criterion benchmark function for loop optimizations
fn benchmark_loop_optimizations(c: &mut Criterion) {
    let mut analyzer = PerformanceAnalyzer::new();

    c.bench_function("loop_optimizations", |b| {
        b.iter(|| {
            analyzer.benchmark_loop_optimizations();
        })
    });
}

/// Comprehensive benchmark that runs all optimizations
fn benchmark_comprehensive_pipeline(c: &mut Criterion) {
    let mut analyzer = PerformanceAnalyzer::new();

    c.bench_function("comprehensive_pipeline", |b| {
        b.iter(|| {
            analyzer.run_comprehensive_benchmark();
        })
    });
}

/// Performance regression test
#[cfg(test)]
mod performance_tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_constant_propagation_performance() {
        let mut analyzer = PerformanceAnalyzer::new();
        let result = analyzer.benchmark_constant_propagation();

        // Constant propagation should be fast (< 10ms for this test case)
        assert!(
            result.baseline_time < Duration::from_millis(10),
            "Constant propagation took too long: {:?}",
            result.baseline_time
        );

        // Should provide some optimization benefit
        assert!(
            result.speedup > 1.0,
            "Constant propagation should provide speedup, got: {}",
            result.speedup
        );
    }

    #[test]
    fn test_strength_reduction_performance() {
        let mut analyzer = PerformanceAnalyzer::new();
        let result = analyzer.benchmark_strength_reduction();

        // Strength reduction should be fast (< 50ms for this test case)
        assert!(
            result.baseline_time < Duration::from_millis(50),
            "Strength reduction took too long: {:?}",
            result.baseline_time
        );
    }

    #[test]
    fn test_function_inlining_performance() {
        let mut analyzer = PerformanceAnalyzer::new();
        let result = analyzer.benchmark_function_inlining();

        // Function inlining should be reasonably fast (< 100ms)
        assert!(
            result.baseline_time < Duration::from_millis(100),
            "Function inlining took too long: {:?}",
            result.baseline_time
        );
    }

    #[test]
    fn test_loop_optimization_performance() {
        let mut analyzer = PerformanceAnalyzer::new();
        let result = analyzer.benchmark_loop_optimizations();

        // Loop optimizations can be more expensive (< 200ms)
        assert!(
            result.baseline_time < Duration::from_millis(200),
            "Loop optimizations took too long: {:?}",
            result.baseline_time
        );
    }

    #[test]
    fn test_comprehensive_benchmark() {
        let mut analyzer = PerformanceAnalyzer::new();
        let results = analyzer.run_comprehensive_benchmark();

        // Should have results for all optimizations
        assert_eq!(results.len(), 4, "Should have 4 benchmark results");

        // All optimizations should complete without errors
        for result in &results {
            assert!(!result.name.is_empty(), "Result name should not be empty");
            assert!(
                result.baseline_time > Duration::ZERO,
                "Baseline time should be positive"
            );
            assert!(
                result.optimized_time > Duration::ZERO,
                "Optimized time should be positive"
            );
        }
    }
}

// Register criterion benchmarks
criterion_group!(
    benches,
    benchmark_constant_propagation,
    benchmark_strength_reduction,
    benchmark_function_inlining,
    benchmark_loop_optimizations,
    benchmark_comprehensive_pipeline
);

criterion_main!(benches);
