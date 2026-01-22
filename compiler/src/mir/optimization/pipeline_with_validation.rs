//! MIR Optimization Pipeline with SSA Validation
//!
//! This module provides an optimization pipeline that integrates SSA validation
//! to ensure correctness throughout the optimization process.

use super::pass_manager::MIRPass;
use crate::mir::analysis::ssa_validator::SSAValidator;
use crate::mir::{MirFunction, MirModule};
use std::collections::HashMap;

/// Optimization pipeline with integrated SSA validation
pub struct ValidatedOptimizationPipeline {
    /// Base optimization passes
    passes: HashMap<String, Box<dyn MIRPass>>,
    /// SSA validator for validation
    validator: SSAValidator,
    /// Whether to validate after each pass
    validate_after_each_pass: bool,
    /// Whether to stop on first validation failure
    stop_on_failure: bool,
    /// Validation statistics
    validation_stats: ValidationStatistics,
}

/// Statistics about validation process
#[derive(Debug, Clone)]
pub struct ValidationStatistics {
    /// Total validations performed
    pub total_validations: usize,
    /// Number of validation failures
    pub failures: usize,
    /// Number of warnings
    pub warnings: usize,
    /// Time spent on validation
    pub validation_time_ms: u128,
}

impl ValidatedOptimizationPipeline {
    pub fn new() -> Self {
        Self {
            passes: HashMap::new(),
            validator: SSAValidator::new(),
            validate_after_each_pass: true,
            stop_on_failure: false,
            validation_stats: ValidationStatistics {
                total_validations: 0,
                failures: 0,
                warnings: 0,
                validation_time_ms: 0,
            },
        }
    }

    /// Add an optimization pass to the pipeline
    pub fn add_pass<P: MIRPass + 'static>(&mut self, name: String, pass: P) {
        self.passes.insert(name, Box::new(pass));
    }

    /// Run the complete pipeline with validation
    pub fn run_pipeline(&mut self, module: &mut MirModule) -> Result<PipelineResult, String> {
        let start_time = std::time::Instant::now();
        let mut total_modified = false;

        // Validate initial state
        if self.validate_after_each_pass {
            self.validate_function_before_optimization(module)?;
        }

        // Collect pass names to avoid borrow conflict
        let pass_names: Vec<String> = self.passes.keys().cloned().collect();

        // Run each pass in order
        for pass_name in pass_names {
            let pass_start = std::time::Instant::now();

            // Run the pass (get mutable reference within this scope)
            let pass_modified = {
                let pass = self.passes.get(&pass_name).ok_or_else(|| {
                    format!("Pass '{}' not found", pass_name)
                })?;
                pass.run_on_module(module)?
            };
            total_modified = total_modified || pass_modified;

            // Validate after pass
            if self.validate_after_each_pass {
                let validation_result = self.validate_function_after_pass(module, &pass_name)?;

                if !validation_result.is_valid {
                    self.validation_stats.failures += 1;

                    if self.stop_on_failure {
                        return Err(format!(
                            "SSA validation failed after pass '{}': {:?}",
                            pass_name, validation_result.errors
                        ));
                    }
                }

                self.validation_stats.warnings += validation_result.warnings.len();
            }

            let pass_time = pass_start.elapsed().as_millis();
            self.validation_stats.validation_time_ms += pass_time;
            self.validation_stats.total_validations += 1;
        }

        let total_time = start_time.elapsed();

        Ok(PipelineResult {
            modified: total_modified,
            validation_stats: self.validation_stats.clone(),
            total_time_ms: total_time.as_millis(),
        })
    }

    /// Validate function before optimization
    fn validate_function_before_optimization(&mut self, module: &MirModule) -> Result<(), String> {
        for func in &module.functions {
            let result = self.validator.validate_function(func);

            if !result.is_valid {
                return Err(format!(
                    "Initial SSA validation failed for function '{}': {:?}",
                    func.name, result.errors
                ));
            }

            if !result.warnings.is_empty() {
                eprintln!(
                    "Initial SSA warnings for function '{}': {:?}",
                    func.name, result.warnings
                );
            }
        }
        Ok(())
    }

    /// Validate function after a specific pass
    fn validate_function_after_pass(
        &mut self,
        module: &MirModule,
        pass_name: &str,
    ) -> Result<ValidationResult, String> {
        let mut all_valid = true;
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();

        for func in &module.functions {
            let result = self.validator.validate_function(func);

            if !result.is_valid {
                all_valid = false;
                all_errors.extend(result.errors.iter().map(|e| e.message.clone()));
            }

            all_warnings.extend(result.warnings.iter().map(|w| w.message.clone()));
        }

        Ok(ValidationResult {
            is_valid: all_valid,
            errors: all_errors,
            warnings: all_warnings,
            pass_name: pass_name.to_string(),
        })
    }

    /// Configure validation settings
    pub fn validate_after_each_pass(&mut self, enable: bool) {
        self.validate_after_each_pass = enable;
    }

    pub fn stop_on_failure(&mut self, enable: bool) {
        self.stop_on_failure = enable;
    }

    /// Get validation statistics
    pub fn get_statistics(&self) -> &ValidationStatistics {
        &self.validation_stats
    }

    /// Reset validation statistics
    pub fn reset_statistics(&mut self) {
        self.validation_stats = ValidationStatistics {
            total_validations: 0,
            failures: 0,
            warnings: 0,
            validation_time_ms: 0,
        };
    }

    /// Generate validation report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# MIR Optimization Pipeline Validation Report\n\n");

        report.push_str(&format!(
            "Total validations: {}\n",
            self.validation_stats.total_validations
        ));
        report.push_str(&format!(
            "Validation failures: {}\n",
            self.validation_stats.failures
        ));
        report.push_str(&format!(
            "Validation warnings: {}\n",
            self.validation_stats.warnings
        ));
        report.push_str(&format!(
            "Total validation time: {}ms\n\n",
            self.validation_stats.validation_time_ms
        ));

        if self.validation_stats.failures == 0 {
            report.push_str("✅ All validations passed successfully!\n");
        } else {
            report.push_str(&format!(
                "❌ {} validation failures detected\n",
                self.validation_stats.failures
            ));
        }

        report
    }
}

/// Result from running the validated pipeline
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub modified: bool,
    pub validation_stats: ValidationStatistics,
    pub total_time_ms: u128,
}

/// Result from validation after a specific pass
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub pass_name: String,
}

/// Create a default validated optimization pipeline
pub fn create_validated_pipeline() -> ValidatedOptimizationPipeline {
    let mut pipeline = ValidatedOptimizationPipeline::new();

    // Add default optimization passes
    pipeline.add_pass(
        "constant-propagation".to_string(),
        super::ConstantPropagation::new(),
    );
    pipeline.add_pass(
        "dead-code-elimination".to_string(),
        super::DeadCodeElimination::new(),
    );
    pipeline.add_pass(
        "common-subexpression-elimination".to_string(),
        super::CommonSubexpressionElimination::new(),
    );

    // Configure validation
    pipeline.validate_after_each_pass(true);
    pipeline.stop_on_failure(false); // Continue with warnings

    pipeline
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::{FunctionBuilder, ModuleBuilder};
    use crate::mir::{MirConstant, MirType};

    #[test]
    fn test_validated_pipeline_creation() {
        let mut pipeline = ValidatedOptimizationPipeline::new();
        assert!(pipeline.passes.is_empty());
        assert!(pipeline.validate_after_each_pass);
    }

    #[test]
    fn test_pipeline_with_validation() {
        let mut pipeline = create_validated_pipeline();

        // Create a simple function
        let mut module_builder = ModuleBuilder::new("test");
        let mut func = module_builder.create_function("test".to_string(), MirType::I64);
        let result = func.build_i64(42);
        func.build_return(Some(result));
        module_builder.add_function(func.build());
        let mut module = module_builder.build();

        // Run pipeline
        let result = pipeline.run_pipeline(&mut module);
        assert!(result.is_ok());

        let pipeline_result = result.unwrap();
        assert!(pipeline_result.validation_stats.total_validations > 0);
    }

    #[test]
    fn test_validation_settings() {
        let mut pipeline = ValidatedOptimizationPipeline::new();

        // Test configuration
        pipeline.validate_after_each_pass(false);
        assert!(!pipeline.validate_after_each_pass);

        pipeline.stop_on_failure(true);
        assert!(pipeline.stop_on_failure);
    }

    #[test]
    fn test_validation_report() {
        let pipeline = ValidatedOptimizationPipeline::new();
        let report = pipeline.generate_report();

        assert!(report.contains("MIR Optimization Pipeline Validation Report"));
        assert!(report.contains("Total validations: 0"));
    }

    #[test]
    fn test_statistics_reset() {
        let mut pipeline = ValidatedOptimizationPipeline::new();

        // Simulate some statistics
        pipeline.validation_stats.total_validations = 10;
        pipeline.validation_stats.failures = 2;
        pipeline.validation_stats.warnings = 5;

        // Reset
        pipeline.reset_statistics();

        assert_eq!(pipeline.validation_stats.total_validations, 0);
        assert_eq!(pipeline.validation_stats.failures, 0);
        assert_eq!(pipeline.validation_stats.warnings, 0);
    }
}
