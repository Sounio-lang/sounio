//! Pass Manager for MIR Optimizations
//!
//! This module provides the base traits and infrastructure for optimization passes.

use crate::mir::{MirFunction, MirModule};

/// Base trait for all MIR optimization passes
pub trait MIRPass {
    /// Get the name of this pass
    fn name(&self) -> &'static str;

    /// Run the pass on a module
    fn run_on_module(&self, _module: &mut MirModule) -> Result<bool, String> {
        Ok(false)
    }

    /// Run the pass on a single function
    fn run_on_function(&self, _func: &mut MirFunction) -> Result<bool, String> {
        Ok(false)
    }

    /// Check if this pass requires SSA form
    fn requires_ssa(&self) -> bool {
        false
    }

    /// Check if this pass preserves SSA form
    fn preserves_ssa(&self) -> bool {
        false
    }
}

/// Base trait for analysis passes that don't modify the IR
pub trait AnalysisPass {
    /// Get the name of this analysis
    fn name(&self) -> &'static str;

    /// Run the analysis on a module
    fn run_on_module(&self, _module: &&mut MirModule) -> Result<(), String> {
        Ok(())
    }

    /// Run the analysis on a function
    fn run_on_function(&self, _func: &MirFunction) -> Result<(), String> {
        Ok(())
    }
}

/// Manager for running optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn MIRPass>>,
    analysis_passes: Vec<Box<dyn AnalysisPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self {
            passes: Vec::new(),
            analysis_passes: Vec::new(),
        }
    }

    pub fn add_pass<P: MIRPass + 'static>(&mut self, pass: P) {
        self.passes.push(Box::new(pass));
    }

    pub fn add_analysis<A: AnalysisPass + 'static>(&mut self, analysis: A) {
        self.analysis_passes.push(Box::new(analysis));
    }

    pub fn run_on_module(&mut self, module: &mut MirModule) -> Result<(), String> {
        for pass in &mut self.passes {
            pass.run_on_module(module)?;
        }
        for analysis in &self.analysis_passes {
            analysis.run_on_module(&mut &mut *module)?;
        }
        Ok(())
    }

    pub fn run_on_function(&mut self, func: &mut MirFunction) -> Result<(), String> {
        for pass in &mut self.passes {
            pass.run_on_function(func)?;
        }
        for analysis in &self.analysis_passes {
            analysis.run_on_function(func)?;
        }
        Ok(())
    }
}
