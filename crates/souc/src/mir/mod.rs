//! Mid-level Intermediate Representation (MIR)
//!
//! MIR is a mid-level IR between HLIR and machine code. It provides:
//! - Explicit control flow with basic blocks
//! - SSA form for register allocation
//! - Machine-independent optimizations
//! - Bridge to backend code generators

pub mod analysis;
pub mod builder;
pub mod instructions;
pub mod lower;
pub mod optimization;
pub mod types;

pub mod benchmark; // Add benchmarking support

// Re-export main types
pub use builder::{FunctionBuilder, ModuleBuilder};
pub use instructions::*;
pub use types::*;

// Integration exports for new advanced components
pub use optimization::AdvancedEpistemicOptimization;
pub use optimization::CompleteFunctionInlining;
pub use optimization::MLGuidedOptimizer;
pub use optimization::PerformanceTunedOptimizer;
pub use optimization::PipelineResult;
pub use optimization::ValidatedOptimizationPipeline;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mir_type_conversion() {
        let mir_unit = MirType::Unit;
        assert_eq!(mir_unit.is_unit(), true);
    }

    #[test]
    fn test_mir_instruction_basics() {
        // Test basic instruction creation
        let const_instr = MirInstruction::Const {
            result: ValueId(0),
            value: MirConstant::Int(42),
            ty: MirType::I32,
        };

        match const_instr {
            MirInstruction::Const {
                value: MirConstant::Int(n),
                ..
            } => {
                assert_eq!(n, 42);
            }
            _ => panic!("Expected Const instruction"),
        }
    }
}
