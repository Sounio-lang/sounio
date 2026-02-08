//! Shared types for ML-guided and heuristic optimization
//!
//! Contains code feature extraction and optimization suggestion types
//! used by both the local heuristic optimizer and the GLM-4.7 API integration.

use crate::mir::instructions::MirInstruction;
use crate::mir::{MirBlock, MirModule};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Code analysis features extracted from MIR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeFeatures {
    /// Function-level features
    pub function_count: usize,
    pub total_blocks: usize,
    pub total_instructions: usize,
    pub avg_block_size: f64,
    pub max_block_size: usize,
    pub loop_count: usize,
    pub branch_count: usize,
    pub call_count: usize,
    pub arithmetic_ops: usize,
    pub memory_ops: usize,

    /// Block-level features
    pub block_features: Vec<BlockFeatures>,

    /// Type information
    pub type_distribution: HashMap<String, usize>,
    pub epistemic_types: usize,
    pub uncertainty_ops: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockFeatures {
    pub instruction_count: usize,
    pub arithmetic_ops: usize,
    pub memory_loads: usize,
    pub memory_stores: usize,
    pub branches: usize,
    pub phi_nodes: usize,
    pub loop_depth: usize,
    pub has_uncertainty: bool,
}

/// Optimization suggestion with confidence score
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub optimization_type: OptimizationType,
    pub confidence: f32,
    pub target: String,
    pub parameters: HashMap<String, String>,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum OptimizationType {
    ConstantPropagation,
    DeadCodeElimination,
    FunctionInlining,
    LoopUnrolling,
    StrengthReduction,
    CommonSubexpressionElimination,
    LoopInvariantCodeMotion,
    AliasAnalysis,
    ScalarReplacementOfAggregates,
    // ML-specific optimizations
    PredictiveInlining,
    AdaptiveUnrolling,
    UncertaintyAwareOptimization,
}

/// Extract code features from a MIR module for a specific function
pub fn extract_features(module: &MirModule, function_name: &str) -> CodeFeatures {
    let mut features = CodeFeatures {
        function_count: module.functions.len(),
        total_blocks: 0,
        total_instructions: 0,
        avg_block_size: 0.0,
        max_block_size: 0,
        loop_count: 0,
        branch_count: 0,
        call_count: 0,
        arithmetic_ops: 0,
        memory_ops: 0,
        block_features: Vec::new(),
        type_distribution: HashMap::new(),
        epistemic_types: 0,
        uncertainty_ops: 0,
    };

    if let Some(func) = module.functions.iter().find(|f| f.name == function_name) {
        for block in &func.blocks {
            let bf = analyze_block(block);
            features.block_features.push(bf);

            features.total_blocks += 1;
            features.total_instructions += block.instructions.len();
            features.max_block_size = features.max_block_size.max(block.instructions.len());

            for inst in &block.instructions {
                match inst {
                    MirInstruction::Binary { op, .. } => {
                        if op.is_arithmetic() {
                            features.arithmetic_ops += 1;
                        }
                    }
                    MirInstruction::Load { .. } | MirInstruction::Store { .. } => {
                        features.memory_ops += 1
                    }
                    MirInstruction::Call { .. } => features.call_count += 1,
                    _ => {}
                }
            }
        }

        features.avg_block_size =
            features.total_instructions as f64 / features.total_blocks.max(1) as f64;
    }

    features
}

/// Analyze a single MIR block and extract features
pub fn analyze_block(block: &MirBlock) -> BlockFeatures {
    let mut features = BlockFeatures {
        instruction_count: block.instructions.len(),
        arithmetic_ops: 0,
        memory_loads: 0,
        memory_stores: 0,
        branches: 0,
        phi_nodes: 0,
        loop_depth: 0,
        has_uncertainty: false,
    };

    for inst in &block.instructions {
        match inst {
            MirInstruction::Binary { op, .. } => {
                if op.is_arithmetic() {
                    features.arithmetic_ops += 1;
                }
            }
            MirInstruction::Load { .. } => features.memory_loads += 1,
            MirInstruction::Store { .. } => features.memory_stores += 1,
            MirInstruction::Phi { .. } => features.phi_nodes += 1,
            _ => {}
        }
    }

    features
}
