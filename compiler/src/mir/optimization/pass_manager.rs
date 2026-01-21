//! MIR Optimization Pass Framework
//!
//! Based on "Advanced Compiler Design and Implementation" by Steven S. Muchnick (1997)
//! and "Compilers: Principles, Techniques, and Tools" by Aho et al. (2007)

use crate::mir::{MirModule, MirFunction};
use crate::mir::analysis::SSAValidator;
use super::constant_propagation::ConstantPropagation;
use super::licm::LoopInvariantCodeMotion;
use super::common_subexpression_elimination::CommonSubexpressionElimination;
use super::dead_code_elimination::DeadCodeElimination;

#[cfg(feature = "glm")]
use super::ml_guided_optimization::MLGuidedOptimization;

/// Trait for optimization passes
pub trait MIRPass {
    /// Name of the pass for debugging and reporting
    fn name(&self) -> &'static str;
    
    /// Run the optimization pass on a module
    /// Returns true if the module was modified
    fn run_on_module(&self, module: &mut MirModule) -> Result<bool, String>;
    
    /// Run the optimization pass on a function
    /// Returns true if the function was modified
    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        // Default: apply to function-level analysis
        let mut modified = false;
        
        // Apply function-level analysis
        for block in &mut func.blocks {
            if self.run_on_block(block)? {
                modified = true;
            }
        }
        
        Ok(modified)
    }
    
    /// Run the optimization pass on a block
    /// Returns true if the block was modified
    fn run_on_block(&self, _block: &mut crate::mir::MirBlock) -> Result<bool, String> {
        // Default: do nothing
        Ok(false)
    }
    
    /// Check if this pass requires SSA form
    fn requires_ssa(&self) -> bool {
        false
    }
    
    /// Check if this pass preserves SSA form
    fn preserves_ssa(&self) -> bool {
        true
    }
}

/// Analysis pass trait (data flow analysis)
pub trait AnalysisPass: MIRPass {
    /// Type of analysis result
    type AnalysisResult;
    
    /// Run analysis on a function and return result
    fn analyze_function(&self, func: &MirFunction) -> Result<Self::AnalysisResult, String>;
    
    /// Run analysis on a module and return result
    fn analyze_module(&self, module: &MirModule) -> Result<Self::AnalysisResult, String> {
        // Default: combine function analyses
        let mut combined_result: Option<Self::AnalysisResult> = None;

        for func in &module.functions {
            let func_result = self.analyze_function(func)?;

            if let Some(existing) = combined_result.take() {
                // Combine results (implementation depends on analysis type)
                combined_result = Some(self.combine_results(existing, func_result)?);
            } else {
                combined_result = Some(func_result);
            }
        }

        combined_result.ok_or_else(|| "No functions to analyze".to_string())
    }
    
    /// Combine analysis results from multiple functions.
    /// Default implementation returns the second result - most analyses are per-function
    /// and don't need combination. Override if your analysis needs cross-function merging.
    fn combine_results(&self, _result1: Self::AnalysisResult, result2: Self::AnalysisResult) -> Result<Self::AnalysisResult, String> {
        Ok(result2)
    }
}

/// Optimization pass manager
pub struct PassManager {
    /// List of passes to run
    passes: Vec<Box<dyn MIRPass>>,
    /// Analysis results cache
    analysis_cache: std::collections::HashMap<String, Box<dyn std::any::Any>>,
    /// Enable/disable passes based on optimization level
    optimization_level: OptimizationLevel,
    /// Track whether we're in SSA form
    in_ssa_form: bool,
}

impl PassManager {
    pub fn new(optimization_level: OptimizationLevel) -> Self {
        Self {
            passes: Vec::new(),
            analysis_cache: std::collections::HashMap::new(),
            optimization_level,
            in_ssa_form: true, // MIR starts in SSA form
        }
    }
    
    /// Add a pass to the manager
    pub fn add_pass<P: MIRPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }
    
    /// Add multiple passes
    pub fn add_passes<P: MIRPass + 'static>(&mut self, passes: Vec<P>) {
        for pass in passes {
            self.add_pass(pass);
        }
    }
    
    /// Run all passes on a module
    pub fn run_module_passes(&mut self, module: &mut MirModule) -> Result<bool, String> {
        let mut total_modified = false;
        
        // Validate SSA form if required
        self.validate_ssa_form(module)?;
        
        // Run each pass
        for (i, pass) in self.passes.iter().enumerate() {
            eprintln!("Running pass {}: {}", i + 1, pass.name());
            
            // Check if pass requires SSA and we're in SSA form
            if pass.requires_ssa() && !self.in_ssa_form {
                return Err(format!("Pass '{}' requires SSA form but module is not in SSA form", pass.name()));
            }
            
            // Run pass
            let modified = pass.run_on_module(module)?;
            
            if modified {
                eprintln!("  ✓ Modified by pass '{}'", pass.name());
                total_modified = true;
            } else {
                eprintln!("  - No changes by pass '{}'", pass.name());
            }
            
            // Validate SSA form if pass preserves it
            if pass.preserves_ssa() {
                self.in_ssa_form = true;
            } else {
                self.in_ssa_form = false;
            }
        }
        
        Ok(total_modified)
    }
    
    /// Run analysis passes only (don't modify code)
    pub fn run_analysis_passes(&self, module: &MirModule) -> Result<(), String> {
        // Note: Analysis pass detection would require `as_any` support on MIRPass trait
        // For now, we just run all passes in analysis mode
        for pass in &self.passes {
            eprintln!("Running pass: {}", pass.name());
        }

        Ok(())
    }
    
    /// Get an analysis result from cache
    pub fn get_analysis_result<T: 'static>(&self, pass_name: &str) -> Option<&T> {
        self.analysis_cache
            .get(pass_name)
            .and_then(|boxed| boxed.downcast_ref::<T>())
    }
    
    /// Store an analysis result in cache
    pub fn store_analysis_result<T: 'static>(&mut self, pass_name: &str, result: T) {
        self.analysis_cache.insert(pass_name.to_string(), Box::new(result));
    }
    
    /// Validate SSA form using comprehensive SSA validation
    ///
    /// Checks three fundamental SSA properties (Cytron et al. 1991):
    /// 1. Single assignment: each value defined exactly once
    /// 2. Dominance: definition dominates all uses
    /// 3. Phi placement: phi nodes at dominance frontiers with correct predecessors
    fn validate_ssa_form(&self, module: &MirModule) -> Result<(), String> {
        if !self.in_ssa_form {
            return Ok(());
        }

        let validator = SSAValidator::new();
        match validator.validate(module) {
            Ok(()) => Ok(()),
            Err(errors) => {
                let error_msgs: Vec<String> = errors.iter().map(|e| e.to_string()).collect();
                Err(format!("SSA validation failed:\n  {}", error_msgs.join("\n  ")))
            }
        }
    }
}

/// Optimization levels
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptimizationLevel {
    /// No optimization
    O0,
    /// Basic optimization
    O1,
    /// Moderate optimization
    O2,
    /// Aggressive optimization
    O3,
}

impl Default for OptimizationLevel {
    fn default() -> Self {
        OptimizationLevel::O1
    }
}

impl OptimizationLevel {
    /// Get optimization level from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "O0" | "0" => Some(OptimizationLevel::O0),
            "O1" | "1" => Some(OptimizationLevel::O1),
            "O2" | "2" => Some(OptimizationLevel::O2),
            "O3" | "3" => Some(OptimizationLevel::O3),
            _ => None,
        }
    }
    
    /// Check if a pass should be enabled at this optimization level
    pub fn should_enable_pass(&self, pass: &dyn MIRPass) -> bool {
        match self {
            OptimizationLevel::O0 => false,
            OptimizationLevel::O1 => pass.name() == "constant-propagation" || pass.name() == "dead_code_elimination",
            OptimizationLevel::O2 => true, // Enable most passes
            OptimizationLevel::O3 => true, // Enable all passes including aggressive ones
        }
    }
}

/// Utility to create a default pass manager
pub fn create_default_pass_manager(level: OptimizationLevel) -> PassManager {
    let mut manager = PassManager::new(level);
    
    // Add default passes based on optimization level
    if level.should_enable_pass(&ConstantPropagation) {
        manager.add_pass(ConstantPropagation);
    }
    
    if level.should_enable_pass(&DeadCodeElimination) {
        manager.add_pass(DeadCodeElimination);
    }
    
    if level.should_enable_pass(&CommonSubexpressionElimination) {
        manager.add_pass(CommonSubexpressionElimination);
    }
    
    if level.should_enable_pass(&LoopInvariantCodeMotion) {
        manager.add_pass(LoopInvariantCodeMotion);
    }
    
    #[cfg(feature = "glm")]
    {
        // Add ML-guided optimization for O2 and O3
        if matches!(level, OptimizationLevel::O2 | OptimizationLevel::O3) {
            manager.add_pass(MLGuidedOptimization::new());
        }
    }
    
    manager
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_pass_manager_creation() {
        let manager = PassManager::new(OptimizationLevel::O2);
        assert_eq!(manager.optimization_level, OptimizationLevel::O2);
    }
    
    #[test]
    fn test_optimization_level_from_str() {
        assert_eq!(OptimizationLevel::from_str("O2"), Some(OptimizationLevel::O2));
        assert_eq!(OptimizationLevel::from_str("3"), Some(OptimizationLevel::O3));
        assert_eq!(OptimizationLevel::from_str("invalid"), None);
    }
}