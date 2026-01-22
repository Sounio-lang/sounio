//! MIR Optimization Benchmarking Framework
//!
//! This module provides comprehensive benchmarking for MIR optimization passes,
//! measuring effectiveness, performance, and quality metrics.

use crate::mir::{
    BlockId, MirConstant, MirFunction, MirModule, MirType, ValueId,
    analysis::ssa_validator::SSAValidator,
    builder::{FunctionBuilder, ModuleBuilder},
    optimization::*,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark results for a single optimization pass
#[derive(Debug, Clone)]
pub struct OptimizationBenchmarkResult {
    /// Name of the optimization pass
    pub pass_name: String,
    /// Time taken to run the optimization
    pub execution_time: Duration,
    /// Number of instructions before optimization
    pub instructions_before: usize,
    /// Number of instructions after optimization
    pub instructions_after: usize,
    /// Number of basic blocks before optimization
    pub blocks_before: usize,
    /// Number of basic blocks after optimization
    pub blocks_after: usize,
    /// Whether the optimization modified the code
    pub modified: bool,
    /// Memory usage (if available)
    pub memory_usage_bytes: Option<usize>,
}

/// Comprehensive benchmark results
#[derive(Debug, Clone)]
pub struct MIROptimizationBenchmark {
    /// Individual pass results
    pub pass_results: Vec<OptimizationBenchmarkResult>,
    /// Total optimization time
    pub total_optimization_time: Duration,
    /// Overall effectiveness metrics
    pub effectiveness_metrics: EffectivenessMetrics,
}

/// Metrics for measuring optimization effectiveness
#[derive(Debug, Clone)]
pub struct EffectivenessMetrics {
    /// Reduction in instruction count (percentage)
    pub instruction_reduction_percent: f64,
    /// Reduction in basic block count (percentage)
    pub block_reduction_percent: f64,
    /// Code quality score (0-100)
    pub quality_score: f64,
    /// Compilation speed impact
    pub compilation_speed_impact: f64,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of iterations for averaging
    pub iterations: usize,
    /// Whether to measure memory usage
    pub measure_memory: bool,
    /// Whether to run all passes or specific ones
    pub run_all_passes: bool,
    /// Specific passes to benchmark (empty = all)
    pub specific_passes: Vec<String>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            iterations: 5,
            measure_memory: false,
            run_all_passes: true,
            specific_passes: Vec::new(),
        }
    }
}

/// Main benchmarking interface
pub struct MIROptimizationBenchmarker {
    config: BenchmarkConfig,
    /// Track baseline metrics for comparison
    baseline_metrics: HashMap<String, usize>,
}

impl MIROptimizationBenchmarker {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            config,
            baseline_metrics: HashMap::new(),
        }
    }

    /// Benchmark a single optimization pass
    pub fn benchmark_pass(
        &mut self,
        pass_name: &str,
        module: &mut MirModule,
    ) -> Result<OptimizationBenchmarkResult, String> {
        let start_time = Instant::now();

        // Collect baseline metrics
        let instructions_before = self.count_instructions(module);
        let blocks_before = self.count_blocks(module);

        // Run the optimization
        let modified = match pass_name {
            "constant-propagation" => {
                let mut pass = ConstantPropagation::new();
                pass.run_on_module(module)?
            }
            "dead-code-elimination" => {
                let mut pass = DeadCodeElimination::new();
                pass.run_on_module(module)?
            }
            "common-subexpression-elimination" => {
                let mut pass = CommonSubexpressionElimination::new();
                pass.run_on_module(module)?
            }
            "strength-reduction" => {
                let mut pass = StrengthReduction::new();
                pass.run_on_module(module)?
            }
            "function-inlining" => {
                let mut pass = FunctionInlining::new();
                pass.run_on_module(module)?
            }
            "epistemic-optimization" => {
                let mut pass = AdvancedEpistemicOptimization::new();
                pass.run_on_module(module)?
            }
            _ => return Err(format!("Unknown optimization pass: {}", pass_name)),
        };

        let execution_time = start_time.elapsed();

        // Collect final metrics
        let instructions_after = self.count_instructions(module);
        let blocks_after = self.count_blocks(module);

        Ok(OptimizationBenchmarkResult {
            pass_name: pass_name.to_string(),
            execution_time,
            instructions_before,
            instructions_after,
            blocks_before,
            blocks_after,
            modified,
            memory_usage_bytes: None, // TODO: Implement memory tracking
        })
    }

    /// Benchmark multiple optimization passes in sequence
    pub fn benchmark_optimization_sequence(
        &mut self,
        module: &mut MirModule,
        pass_names: Vec<&str>,
    ) -> Result<MIROptimizationBenchmark, String> {
        let start_time = Instant::now();
        let mut pass_results = Vec::new();

        // Collect baseline metrics
        let baseline_instructions = self.count_instructions(module);
        let baseline_blocks = self.count_blocks(module);
        self.baseline_metrics
            .insert("instructions".to_string(), baseline_instructions);
        self.baseline_metrics
            .insert("blocks".to_string(), baseline_blocks);

        // Run each pass in sequence
        for pass_name in pass_names {
            let result = self.benchmark_pass(pass_name, module)?;
            pass_results.push(result);
        }

        let total_optimization_time = start_time.elapsed();

        // Calculate effectiveness metrics
        let effectiveness_metrics = self.calculate_effectiveness_metrics(&pass_results)?;

        Ok(MIROptimizationBenchmark {
            pass_results,
            total_optimization_time,
            effectiveness_metrics,
        })
    }

    /// Count total instructions in a module
    fn count_instructions(&self, module: &MirModule) -> usize {
        module
            .functions
            .iter()
            .map(|func| self.count_function_instructions(func))
            .sum()
    }

    /// Count instructions in a function
    fn count_function_instructions(&self, func: &MirFunction) -> usize {
        func.blocks
            .iter()
            .map(|block| block.instructions.len())
            .sum()
    }

    /// Count total blocks in a module
    fn count_blocks(&self, module: &MirModule) -> usize {
        module.functions.iter().map(|func| func.blocks.len()).sum()
    }

    /// Calculate effectiveness metrics from benchmark results
    fn calculate_effectiveness_metrics(
        &self,
        results: &[OptimizationBenchmarkResult],
    ) -> Result<EffectivenessMetrics, String> {
        if results.is_empty() {
            return Err("No benchmark results provided".to_string());
        }

        // Calculate instruction reduction
        let total_instruction_reduction = results
            .iter()
            .map(|r| {
                if r.instructions_before > 0 {
                    (r.instructions_before - r.instructions_after) as f64
                        / r.instructions_before as f64
                        * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        let instruction_reduction_percent = total_instruction_reduction / results.len() as f64;

        // Calculate block reduction
        let total_block_reduction = results
            .iter()
            .map(|r| {
                if r.blocks_before > 0 {
                    (r.blocks_before - r.blocks_after) as f64 / r.blocks_before as f64 * 100.0
                } else {
                    0.0
                }
            })
            .sum::<f64>();
        let block_reduction_percent = total_block_reduction / results.len() as f64;

        // Calculate quality score (0-100)
        let mut quality_score = 0.0;
        quality_score += (instruction_reduction_percent * 0.4).min(40.0); // 40% weight
        quality_score += (block_reduction_percent * 0.3).min(30.0); // 30% weight
        quality_score +=
            (results.iter().filter(|r| r.modified).count() as f64 / results.len() as f64 * 30.0)
                .min(30.0); // 30% weight

        // Calculate compilation speed impact
        let total_time_ms = results
            .iter()
            .map(|r| r.execution_time.as_millis())
            .sum::<u128>();
        let avg_time_per_pass = total_time_ms as f64 / results.len() as f64;
        let compilation_speed_impact = 100.0 - (avg_time_per_pass * 0.1); // Penalize slower passes

        Ok(EffectivenessMetrics {
            instruction_reduction_percent,
            block_reduction_percent,
            quality_score,
            compilation_speed_impact,
        })
    }

    /// Generate a comprehensive benchmark report
    pub fn generate_report(&self, benchmark: &MIROptimizationBenchmark) -> String {
        let mut report = String::new();

        report.push_str("# MIR Optimization Benchmark Report\n\n");
        report.push_str(&format!("Generated: {}\n\n", chrono::Utc::now()));

        // Overall metrics
        report.push_str("## Overall Effectiveness\n");
        report.push_str(&format!(
            "- Instruction Reduction: {:.2}%\n",
            benchmark
                .effectiveness_metrics
                .instruction_reduction_percent
        ));
        report.push_str(&format!(
            "- Block Reduction: {:.2}%\n",
            benchmark.effectiveness_metrics.block_reduction_percent
        ));
        report.push_str(&format!(
            "- Quality Score: {:.2}/100\n",
            benchmark.effectiveness_metrics.quality_score
        ));
        report.push_str(&format!(
            "- Compilation Speed Impact: {:.2}\n\n",
            benchmark.effectiveness_metrics.compilation_speed_impact
        ));

        // Individual pass results
        report.push_str("## Individual Pass Results\n\n");
        for result in &benchmark.pass_results {
            report.push_str(&format!("### {}\n", result.pass_name));
            report.push_str(&format!("- Execution Time: {:?}\n", result.execution_time));
            report.push_str(&format!(
                "- Instructions: {} → {} ({:+})\n",
                result.instructions_before,
                result.instructions_after,
                result.instructions_after as isize - result.instructions_before as isize
            ));
            report.push_str(&format!(
                "- Blocks: {} → {} ({:+})\n",
                result.blocks_before,
                result.blocks_after,
                result.blocks_after as isize - result.blocks_before as isize
            ));
            report.push_str(&format!("- Modified: {}\n\n", result.modified));
        }

        report
    }
}

/// Predefined benchmark test cases
pub struct BenchmarkTestCases;

impl BenchmarkTestCases {
    /// Generate a simple function for testing
    pub fn simple_arithmetic() -> MirModule {
        let mut module_builder = ModuleBuilder::new("simple_arithmetic");
        let mut func = module_builder.create_function("compute".to_string(), MirType::I64);

        // Create some arithmetic operations
        let a = func.build_i64(10);
        let b = func.build_i64(20);
        let c = func.fresh_value();
        func.build_add(c, a, b, MirType::I64);
        let two = func.build_i64(2);
        let d = func.fresh_value();
        func.build_mul(d, c, two, MirType::I64);
        let five = func.build_i64(5);
        let result = func.fresh_value();
        func.build_add(result, d, five, MirType::I64);
        func.build_return(Some(result));

        module_builder.add_function(func.build());
        module_builder.build()
    }

    /// Generate a function with common subexpressions
    pub fn common_subexpressions() -> MirModule {
        let mut module_builder = ModuleBuilder::new("common_subexpressions");
        let mut func = module_builder.create_function("compute".to_string(), MirType::I64);

        // Create common subexpressions
        let x = func.build_i64(15);
        let y = func.build_i64(25);
        let expr1 = func.fresh_value();
        func.build_add(expr1, x, y, MirType::I64);
        let expr2 = func.fresh_value();
        func.build_add(expr2, x, y, MirType::I64); // Same as expr1
        let result = func.fresh_value();
        func.build_add(result, expr1, expr2, MirType::I64);
        func.build_return(Some(result));

        module_builder.add_function(func.build());
        module_builder.build()
    }

    /// Generate a function with dead code
    pub fn dead_code() -> MirModule {
        let mut module_builder = ModuleBuilder::new("dead_code");
        let mut func = module_builder.create_function("compute".to_string(), MirType::I64);

        // Create dead code
        let dead1 = func.build_i64(42);
        let dead2 = func.build_i64(100);
        let _unused = func.fresh_value();
        func.build_add(_unused, dead1, dead2, MirType::I64); // Unused
        let result = func.build_i64(200);
        func.build_return(Some(result));

        module_builder.add_function(func.build());
        module_builder.build()
    }

    /// Generate a function with loops and constant propagation opportunities
    pub fn loop_constant_prop() -> MirModule {
        let mut module_builder = ModuleBuilder::new("loop_constant_prop");
        let mut func = module_builder.create_function("compute".to_string(), MirType::I64);

        // Create loop with constant condition
        let const_condition = func.build_bool(true);
        let loop_body_block = func.create_block("loop_body");

        // Loop body: increment a counter
        let counter = func.build_i64(0);
        let one = func.build_i64(1);
        let next_counter = func.fresh_value();
        func.build_add(next_counter, counter, one, MirType::I64);

        // Branch back to loop
        func.build_cond_branch(const_condition, loop_body_block, BlockId(1));

        // Return from loop
        let result = func.build_i64(100);
        func.build_return(Some(result));

        module_builder.add_function(func.build());
        module_builder.build()
    }

    /// Generate a large function for performance testing
    pub fn large_function(size: usize) -> MirModule {
        let mut module_builder = ModuleBuilder::new("large_function");
        let mut func = module_builder.create_function("compute".to_string(), MirType::I64);

        // Create many operations
        let mut last_value = func.build_i64(0);
        for i in 0..size {
            let const_val = func.build_i64(i as i64);
            let new_value = func.fresh_value();
            func.build_add(new_value, last_value, const_val, MirType::I64);
            last_value = new_value;
        }

        func.build_return(Some(last_value));
        module_builder.add_function(func.build());
        module_builder.build()
    }
}

/// Benchmark runner for standard test cases
pub struct StandardBenchmarkRunner {
    benchmarker: MIROptimizationBenchmarker,
}

impl StandardBenchmarkRunner {
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            benchmarker: MIROptimizationBenchmarker::new(config),
        }
    }

    /// Run benchmarks on all standard test cases
    pub fn run_all_benchmarks(
        &mut self,
    ) -> Result<HashMap<String, MIROptimizationBenchmark>, String> {
        let mut results = HashMap::new();

        // Benchmark each test case
        let test_cases = vec![
            ("simple_arithmetic", BenchmarkTestCases::simple_arithmetic()),
            (
                "common_subexpressions",
                BenchmarkTestCases::common_subexpressions(),
            ),
            ("dead_code", BenchmarkTestCases::dead_code()),
            (
                "loop_constant_prop",
                BenchmarkTestCases::loop_constant_prop(),
            ),
        ];

        for (name, mut module) in test_cases {
            let pass_names = vec![
                "constant-propagation",
                "dead-code-elimination",
                "common-subexpression-elimination",
            ];

            let benchmark = self
                .benchmarker
                .benchmark_optimization_sequence(&mut module, pass_names)?;
            results.insert(name.to_string(), benchmark);
        }

        Ok(results)
    }

    /// Run performance benchmark on large function
    pub fn run_performance_benchmark(
        &mut self,
        size: usize,
    ) -> Result<MIROptimizationBenchmark, String> {
        let mut module = BenchmarkTestCases::large_function(size);
        let pass_names = vec![
            "constant-propagation",
            "dead-code-elimination",
            "common-subexpression-elimination",
        ];

        self.benchmarker
            .benchmark_optimization_sequence(&mut module, pass_names)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmarker_creation() {
        let config = BenchmarkConfig::default();
        let benchmarker = MIROptimizationBenchmarker::new(config);
        assert!(benchmarker.baseline_metrics.is_empty());
    }

    #[test]
    fn test_simple_function_metrics() {
        let module = BenchmarkTestCases::simple_arithmetic();
        let benchmarker = MIROptimizationBenchmarker::new(BenchmarkConfig::default());

        let instructions = benchmarker.count_instructions(&module);
        let blocks = benchmarker.count_blocks(&module);

        assert!(instructions > 0);
        assert!(blocks > 0);
    }

    #[test]
    fn test_effectiveness_metrics_calculation() {
        let mut benchmarker = MIROptimizationBenchmarker::new(BenchmarkConfig::default());

        let results = vec![OptimizationBenchmarkResult {
            pass_name: "test_pass".to_string(),
            execution_time: Duration::from_millis(10),
            instructions_before: 100,
            instructions_after: 80,
            blocks_before: 10,
            blocks_after: 8,
            modified: true,
            memory_usage_bytes: None,
        }];

        let metrics = benchmarker
            .calculate_effectiveness_metrics(&results)
            .unwrap();

        assert_eq!(metrics.instruction_reduction_percent, 20.0);
        assert_eq!(metrics.block_reduction_percent, 20.0);
        assert!(metrics.quality_score > 0.0);
    }

    #[test]
    fn test_standard_benchmark_runner() {
        let mut runner = StandardBenchmarkRunner::new(BenchmarkConfig::default());
        let results = runner.run_all_benchmarks();

        assert!(results.is_ok());
        let benchmarks = results.unwrap();
        assert!(!benchmarks.is_empty());
    }
}
