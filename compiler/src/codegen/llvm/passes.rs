//! LLVM optimization passes
//!
//! This module provides optimization pass management for LLVM code generation.

use inkwell::OptimizationLevel;
use inkwell::module::Module;
use inkwell::passes::{PassBuilderOptions, PassManager};
use inkwell::targets::TargetMachine;

use super::codegen::OptLevel;

/// Run optimization passes on a module using the new pass manager
pub fn optimize_module(module: &Module, opt_level: OptLevel, target: &TargetMachine) {
    let pass_options = PassBuilderOptions::create();

    // Set verification options for debug builds
    pass_options.set_verify_each(cfg!(debug_assertions));
    pass_options.set_debug_logging(false);

    // Build the pass pipeline string based on optimization level
    let passes = match opt_level {
        OptLevel::O0 => "default<O0>",
        OptLevel::O1 => "default<O1>",
        OptLevel::O2 => "default<O2>",
        OptLevel::O3 => "default<O3>",
        OptLevel::Os => "default<Os>",
        OptLevel::Oz => "default<Oz>",
    };

    // Run the pass pipeline
    let _ = module.run_passes(passes, target, pass_options);
}

/// Run custom optimization pipeline
pub fn run_custom_passes(module: &Module, opt_level: OptLevel, target: &TargetMachine) {
    let pass_options = PassBuilderOptions::create();

    // Build custom pipeline based on optimization level
    let passes = match opt_level {
        OptLevel::O0 => {
            // Minimal passes for O0
            "function(mem2reg)"
        }
        OptLevel::O1 => {
            // Basic optimizations
            concat!(
                "function(mem2reg,instcombine,simplifycfg),",
                "function(early-cse,reassociate),",
                "cgscc(inline),",
                "function(simplifycfg,instcombine)"
            )
        }
        OptLevel::O2 => {
            // Standard optimizations
            concat!(
                "function(mem2reg,instcombine,simplifycfg),",
                "function(early-cse,reassociate,loop-simplify,lcssa),",
                "function(loop(indvars,loop-idiom,loop-deletion)),",
                "cgscc(inline),",
                "function(gvn,sccp,bdce,adce),",
                "function(simplifycfg,instcombine,loop-unroll)"
            )
        }
        OptLevel::O3 => {
            // Aggressive optimizations
            concat!(
                "function(mem2reg,instcombine,simplifycfg),",
                "function(early-cse,reassociate,loop-simplify,lcssa),",
                "function(loop(indvars,loop-idiom,loop-deletion,loop-unroll)),",
                "cgscc(inline),",
                "function(gvn,sccp,bdce,adce,dse),",
                "function(slp-vectorizer,loop-vectorize),",
                "function(simplifycfg,instcombine,aggressive-instcombine)"
            )
        }
        OptLevel::Os => {
            // Size optimizations
            concat!(
                "function(mem2reg,instcombine,simplifycfg),",
                "function(early-cse),",
                "cgscc(inline),",
                "function(gvn,sccp,bdce,adce),",
                "function(simplifycfg,instcombine)"
            )
        }
        OptLevel::Oz => {
            // Aggressive size optimizations
            concat!(
                "function(mem2reg,instcombine,simplifycfg),",
                "cgscc(inline),",
                "function(sccp,bdce,adce),",
                "function(simplifycfg)"
            )
        }
    };

    let _ = module.run_passes(passes, target, pass_options);
}

/// Verify module is well-formed
pub fn verify_module(module: &Module) -> Result<(), String> {
    module.verify().map_err(|e| e.to_string())
}

/// Run only verification passes (useful for debugging)
pub fn verify_only(module: &Module, target: &TargetMachine) {
    let pass_options = PassBuilderOptions::create();
    pass_options.set_verify_each(true);

    let _ = module.run_passes("verify", target, pass_options);
}

/// Print module statistics
pub fn print_stats(module: &Module) {
    let mut func_count = 0;
    let mut block_count = 0;
    let mut instr_count = 0;

    let mut func = module.get_first_function();
    while let Some(f) = func {
        func_count += 1;

        let mut block = f.get_first_basic_block();
        while let Some(bb) = block {
            block_count += 1;

            let mut instr = bb.get_first_instruction();
            while let Some(_i) = instr {
                instr_count += 1;
                instr = _i.get_next_instruction();
            }

            block = bb.get_next_basic_block();
        }

        func = f.get_next_function();
    }

    println!("Module Statistics:");
    println!("  Functions: {}", func_count);
    println!("  Basic blocks: {}", block_count);
    println!("  Instructions: {}", instr_count);
}

/// Optimization pass configuration
#[derive(Debug, Clone)]
pub struct PassConfig {
    /// Enable inlining
    pub inline: bool,
    /// Inlining threshold
    pub inline_threshold: u32,
    /// Enable loop optimizations
    pub loop_opts: bool,
    /// Enable vectorization
    pub vectorize: bool,
    /// Enable link-time optimization preparation
    pub lto: bool,
    /// Enable debug info preservation
    pub preserve_debug: bool,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            inline: true,
            inline_threshold: 250,
            loop_opts: true,
            vectorize: false,
            lto: false,
            preserve_debug: false,
        }
    }
}

impl PassConfig {
    /// Create config for given optimization level
    pub fn for_opt_level(level: OptLevel) -> Self {
        match level {
            OptLevel::O0 => Self {
                inline: false,
                inline_threshold: 0,
                loop_opts: false,
                vectorize: false,
                lto: false,
                preserve_debug: true,
            },
            OptLevel::O1 => Self {
                inline: true,
                inline_threshold: 100,
                loop_opts: true,
                vectorize: false,
                lto: false,
                preserve_debug: false,
            },
            OptLevel::O2 => Self {
                inline: true,
                inline_threshold: 250,
                loop_opts: true,
                vectorize: false,
                lto: false,
                preserve_debug: false,
            },
            OptLevel::O3 => Self {
                inline: true,
                inline_threshold: 500,
                loop_opts: true,
                vectorize: true,
                lto: false,
                preserve_debug: false,
            },
            OptLevel::Os => Self {
                inline: true,
                inline_threshold: 50,
                loop_opts: true,
                vectorize: false,
                lto: false,
                preserve_debug: false,
            },
            OptLevel::Oz => Self {
                inline: false,
                inline_threshold: 0,
                loop_opts: false,
                vectorize: false,
                lto: false,
                preserve_debug: false,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use inkwell::context::Context;

    #[test]
    fn test_pass_config() {
        let config = PassConfig::for_opt_level(OptLevel::O2);
        assert!(config.inline);
        assert_eq!(config.inline_threshold, 250);
        assert!(config.loop_opts);
        assert!(!config.vectorize);
    }

    #[test]
    fn test_verify_empty_module() {
        let context = Context::create();
        let module = context.create_module("test");

        assert!(verify_module(&module).is_ok());
    }
}
