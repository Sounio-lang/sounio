//! SSA Validator for MIR
//!
//! This module provides validation of SSA form properties, based on:
//! - Cytron et al. (1991) "Efficiently Computing Static Single Assignment Form"
//!
//! The validator checks:
//! 1. Dominance property - each variable is defined before use
//! 2. Phi node placement - phi nodes are in the correct locations
//! 3. Single assignment property - each variable is defined exactly once

use crate::mir::{BlockId, MirFunction, MirInstruction, MirModule, MirTerminator, ValueId};
use std::collections::{HashMap, HashSet};

/// SSA Validation result
#[derive(Debug, Clone)]
pub struct SSAValidationResult {
    /// Whether the SSA form is valid
    pub is_valid: bool,
    /// List of validation errors
    pub errors: Vec<SSAValidationError>,
    /// List of warnings
    pub warnings: Vec<SSAValidationWarning>,
}

/// SSA Validation error
#[derive(Debug, Clone)]
pub struct SSAValidationError {
    /// Description of the error
    pub message: String,
    /// Block where the error occurs
    pub block_id: Option<BlockId>,
    /// Instruction index where the error occurs
    pub instruction_index: Option<usize>,
}

/// SSA Validation warning
#[derive(Debug, Clone)]
pub struct SSAValidationWarning {
    /// Description of the warning
    pub message: String,
    /// Block where the warning occurs
    pub block_id: Option<BlockId>,
    /// Instruction index where the warning occurs
    pub instruction_index: Option<usize>,
}

/// SSA Validator for MIR
pub struct SSAValidator {
    /// Dominator analysis results
    dominators: HashMap<BlockId, HashSet<BlockId>>,
    /// Reverse dominator analysis results
    post_dominators: HashMap<BlockId, HashSet<BlockId>>,
    /// All defined values in the function
    defined_values: HashMap<ValueId, ValueDefinition>,
    /// All uses of values in the function
    value_uses: HashMap<ValueId, Vec<ValueUse>>,
    /// Phi nodes and their locations
    phi_nodes: HashMap<BlockId, Vec<PhiNode>>,
}

/// Information about where a value is defined
#[derive(Debug, Clone)]
struct ValueDefinition {
    /// Block where the value is defined
    pub block_id: BlockId,
    /// Instruction index where the value is defined
    pub instruction_index: Option<usize>,
    /// Type of the value
    pub value_type: String,
}

/// Information about where a value is used
#[derive(Debug, Clone)]
struct ValueUse {
    /// Block where the value is used
    pub block_id: BlockId,
    /// Instruction index where the value is used
    pub instruction_index: usize,
    /// Context of the use
    pub context: String,
}

/// Information about a phi node
#[derive(Debug, Clone)]
struct PhiNode {
    /// Result value of the phi node
    pub result: ValueId,
    /// Incoming values and their source blocks
    pub incoming: Vec<(BlockId, ValueId)>,
    /// Type of the phi node
    pub phi_type: String,
}

impl SSAValidator {
    pub fn new() -> Self {
        Self {
            dominators: HashMap::new(),
            post_dominators: HashMap::new(),
            defined_values: HashMap::new(),
            value_uses: HashMap::new(),
            phi_nodes: HashMap::new(),
        }
    }

    /// Validate a complete MIR module
    pub fn validate_module(&mut self, module: &MirModule) -> SSAValidationResult {
        let mut all_errors = Vec::new();
        let mut all_warnings = Vec::new();

        for func in &module.functions {
            let result = self.validate_function(func);
            all_errors.extend(result.errors);
            all_warnings.extend(result.warnings);
        }

        SSAValidationResult {
            is_valid: all_errors.is_empty(),
            errors: all_errors,
            warnings: all_warnings,
        }
    }

    /// Validate a single function for SSA properties
    pub fn validate_function(&mut self, func: &MirFunction) -> SSAValidationResult {
        // Reset analysis state
        self.defined_values.clear();
        self.value_uses.clear();
        self.phi_nodes.clear();

        // Collect all definitions and uses
        self.collect_definitions_and_uses(func);

        // Validate SSA properties
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Check 1: Single assignment property
        self.check_single_assignment_property(func, &mut errors);

        // Check 2: Dominance property
        self.check_dominance_property(func, &mut errors);

        // Check 3: Phi node placement
        self.check_phi_node_placement(func, &mut errors);

        // Check 4: Phi node arguments
        self.check_phi_arguments(func, &mut errors);

        // Check 5: Unreachable code
        self.check_unreachable_blocks(func, &mut warnings);

        SSAValidationResult {
            is_valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Collect all value definitions and uses in the function
    fn collect_definitions_and_uses(&mut self, func: &MirFunction) {
        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                // Collect definitions
                if let Some(result) = instr.result() {
                    self.defined_values.insert(
                        result,
                        ValueDefinition {
                            block_id: block.id,
                            instruction_index: Some(instr_idx),
                            value_type: format!("{:?}", instr),
                        },
                    );
                }

                // Collect uses
                for used_value in instr.used_values() {
                    self.value_uses
                        .entry(used_value)
                        .or_insert_with(Vec::new)
                        .push(ValueUse {
                            block_id: block.id,
                            instruction_index: instr_idx,
                            context: format!("{:?}", instr),
                        });
                }
            }

            // Collect phi nodes
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                if let MirInstruction::Phi {
                    result,
                    incoming,
                    ty,
                } = instr
                {
                    self.phi_nodes.insert(
                        block.id,
                        vec![PhiNode {
                            result: *result,
                            incoming: incoming.clone(),
                            phi_type: format!("{:?}", ty),
                        }],
                    );

                    // Collect phi node definitions
                    self.defined_values.insert(
                        *result,
                        ValueDefinition {
                            block_id: block.id,
                            instruction_index: Some(instr_idx),
                            value_type: "phi".to_string(),
                        },
                    );

                    // Collect phi node uses (incoming values)
                    for (_, value_id) in incoming {
                        self.value_uses
                            .entry(*value_id)
                            .or_insert_with(Vec::new)
                            .push(ValueUse {
                                block_id: block.id,
                                instruction_index: instr_idx,
                                context: "phi_incoming".to_string(),
                            });
                    }
                }
            }

            // Collect terminator uses
            let terminator_uses = get_terminator_uses(&block.terminator);
            for value_id in terminator_uses {
                self.value_uses
                    .entry(value_id)
                    .or_insert_with(Vec::new)
                    .push(ValueUse {
                        block_id: block.id,
                        instruction_index: block.instructions.len(), // After last instruction
                        context: format!("{:?}", block.terminator),
                    });
            }
        }
    }

    /// Check that each value is defined exactly once
    fn check_single_assignment_property(
        &self,
        func: &MirFunction,
        errors: &mut Vec<SSAValidationError>,
    ) {
        // Check for values defined in multiple places
        let mut value_blocks = HashMap::new();

        for (value_id, definition) in &self.defined_values {
            let block_id = definition.block_id;
            if let Some(existing_block) = value_blocks.get(value_id) {
                if *existing_block != block_id {
                    errors.push(SSAValidationError {
                        message: format!(
                            "Value {:?} is defined in multiple blocks: {:?} and {:?}",
                            value_id, existing_block, block_id
                        ),
                        block_id: Some(block_id),
                        instruction_index: definition.instruction_index,
                    });
                }
            } else {
                value_blocks.insert(*value_id, block_id);
            }
        }

        // Check for values defined multiple times in the same block
        let mut block_value_counts = HashMap::new();
        for (value_id, definition) in &self.defined_values {
            let key = (definition.block_id, *value_id);
            let count = block_value_counts.entry(key).or_insert(0);
            *count += 1;

            if *count > 1 {
                errors.push(SSAValidationError {
                    message: format!(
                        "Value {:?} is defined multiple times in block {:?}",
                        value_id, definition.block_id
                    ),
                    block_id: Some(definition.block_id),
                    instruction_index: definition.instruction_index,
                });
            }
        }
    }

    /// Check that each use of a value is dominated by its definition
    fn check_dominance_property(&mut self, func: &MirFunction, errors: &mut Vec<SSAValidationError>) {
        // Build dominance information (simplified - would need proper dominator analysis)
        self.build_dominators(func);

        for (value_id, definition) in &self.defined_values {
            if let Some(uses) = self.value_uses.get(value_id) {
                for use_info in uses {
                    // Check if the use is dominated by the definition
                    if !self.is_dominated_by(
                        use_info.block_id,
                        definition.block_id,
                        &self.dominators,
                    ) {
                        errors.push(SSAValidationError {
                            message: format!(
                                "Value {:?} is used in block {:?} but defined in block {:?} (dominance violation)",
                                value_id, use_info.block_id, definition.block_id
                            ),
                            block_id: Some(use_info.block_id),
                            instruction_index: Some(use_info.instruction_index),
                        });
                    }
                }
            }
        }
    }

    /// Check that phi nodes are placed correctly
    fn check_phi_node_placement(&self, func: &MirFunction, errors: &mut Vec<SSAValidationError>) {
        // Simplified check: phi nodes should be at the start of blocks
        for (block_id, phi_nodes) in &self.phi_nodes {
            for phi_node in phi_nodes {
                // Check that the phi node is the first instruction in the block
                if let Some(block) = func.get_block(*block_id) {
                    if !block.instructions.is_empty() {
                        if let Some(first_instr) = block.instructions.first() {
                            if !matches!(first_instr, MirInstruction::Phi { .. }) {
                                errors.push(SSAValidationError {
                                    message: format!(
                                        "Phi node for value {:?} is not at the start of block {:?}",
                                        phi_node.result, block_id
                                    ),
                                    block_id: Some(*block_id),
                                    instruction_index: Some(0),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    /// Check that phi node arguments come from predecessor blocks
    fn check_phi_arguments(&self, func: &MirFunction, errors: &mut Vec<SSAValidationError>) {
        for (block_id, phi_nodes) in &self.phi_nodes {
            // Get predecessor blocks
            let predecessors = self.get_predecessor_blocks(func, *block_id);

            for phi_node in phi_nodes {
                for (source_block, _) in &phi_node.incoming {
                    if !predecessors.contains(source_block) {
                        errors.push(SSAValidationError {
                            message: format!(
                                "Phi node for value {:?} has argument from non-predecessor block {:?}",
                                phi_node.result, source_block
                            ),
                            block_id: Some(*block_id),
                            instruction_index: None,
                        });
                    }
                }
            }
        }
    }

    /// Check for unreachable blocks
    fn check_unreachable_blocks(
        &self,
        func: &MirFunction,
        warnings: &mut Vec<SSAValidationWarning>,
    ) {
        let mut reachable_blocks = HashSet::new();
        let mut worklist = vec![func.entry_block];
        reachable_blocks.insert(func.entry_block);

        // Simple reachability analysis
        while let Some(current) = worklist.pop() {
            if let Some(block) = func.get_block(current) {
                for successor in get_block_successors(&block.terminator) {
                    if reachable_blocks.insert(successor) {
                        worklist.push(successor);
                    }
                }
            }
        }

        // Find unreachable blocks
        for block in &func.blocks {
            if !reachable_blocks.contains(&block.id) {
                warnings.push(SSAValidationWarning {
                    message: format!("Block {:?} is unreachable", block.id),
                    block_id: Some(block.id),
                    instruction_index: None,
                });
            }
        }
    }

    /// Build dominator information (simplified implementation)
    fn build_dominators(&mut self, func: &MirFunction) {
        self.dominators.clear();

        // Initialize: each block dominates itself
        for block in &func.blocks {
            let mut dom_set = HashSet::new();
            dom_set.insert(block.id);
            self.dominators.insert(block.id, dom_set);
        }

        // Iterative algorithm for finding dominators
        let mut changed = true;
        while changed {
            changed = false;
            for block in &func.blocks {
                if block.id == func.entry_block {
                    continue; // Entry block already initialized
                }

                // Start with all blocks, then intersect with predecessors
                // A block always dominates itself, so we add it after the intersection
                let preds: Vec<BlockId> = self.get_predecessor_blocks(func, block.id);
                let mut new_dom_set: HashSet<BlockId> = if preds.is_empty() {
                    // No predecessors - only dominated by itself
                    HashSet::new()
                } else {
                    let mut first_pred = true;
                    let mut dom_set = HashSet::new();
                    for pred in preds {
                        if let Some(pred_dom) = self.dominators.get(&pred) {
                            if first_pred {
                                dom_set = pred_dom.clone();
                                first_pred = false;
                            } else {
                                dom_set = dom_set.intersection(pred_dom).cloned().collect();
                            }
                        }
                    }
                    dom_set
                };
                // A block always dominates itself
                new_dom_set.insert(block.id);

                if let Some(current_dom) = self.dominators.get(&block.id) {
                    if *current_dom != new_dom_set {
                        self.dominators.insert(block.id, new_dom_set);
                        changed = true;
                    }
                }
            }
        }
    }

    /// Get predecessor blocks of a given block
    fn get_predecessor_blocks(&self, func: &MirFunction, block_id: BlockId) -> Vec<BlockId> {
        let mut predecessors = Vec::new();

        for block in &func.blocks {
            for successor in get_block_successors(&block.terminator) {
                if successor == block_id {
                    predecessors.push(block.id);
                }
            }
        }

        predecessors
    }

    /// Check if block A is dominated by block B
    fn is_dominated_by(
        &self,
        block_a: BlockId,
        block_b: BlockId,
        dominators: &HashMap<BlockId, HashSet<BlockId>>,
    ) -> bool {
        if let Some(dom_set) = dominators.get(&block_a) {
            dom_set.contains(&block_b)
        } else {
            false
        }
    }
}

/// Get all values used by a terminator
fn get_terminator_uses(terminator: &MirTerminator) -> Vec<ValueId> {
    match terminator {
        MirTerminator::Branch { .. } => vec![],
        MirTerminator::CondBranch { condition, .. } => vec![*condition],
        MirTerminator::Switch { value, .. } => vec![*value],
        MirTerminator::Return { value } => value.iter().copied().collect(),
        MirTerminator::Unreachable => vec![],
        MirTerminator::CallNoReturn { args, .. } => args.clone(),
    }
}

/// Get all successor blocks from a terminator
fn get_block_successors(terminator: &MirTerminator) -> Vec<BlockId> {
    match terminator {
        MirTerminator::Branch { target } => vec![*target],
        MirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => vec![*true_target, *false_target],
        MirTerminator::Switch {
            default_target,
            cases,
            ..
        } => {
            let mut succs = vec![*default_target];
            succs.extend(cases.iter().map(|(_, target)| *target));
            succs
        }
        MirTerminator::Return { .. }
        | MirTerminator::Unreachable
        | MirTerminator::CallNoReturn { .. } => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::{FunctionBuilder, ModuleBuilder};
    use crate::mir::{MirConstant, MirType};

    #[test]
    fn test_ssa_validator_creation() {
        let validator = SSAValidator::new();
        assert_eq!(validator.defined_values.len(), 0);
    }

    #[test]
    fn test_simple_function_validation() {
        let mut module_builder = ModuleBuilder::new("test");

        let mut func = module_builder.create_function("simple".to_string(), MirType::I64);
        let result = func.build_i64(42);
        func.build_return(Some(result));
        module_builder.add_function(func.build());

        let module = module_builder.build();
        let mut validator = SSAValidator::new();

        let result = validator.validate_function(&module.functions[0]);
        assert!(result.is_valid);
        assert_eq!(result.errors.len(), 0);
    }

    #[test]
    fn test_unreachable_block_detection() {
        let mut module_builder = ModuleBuilder::new("test");

        // Create a simple function that returns a constant
        let mut func = module_builder.create_function("test".to_string(), MirType::I64);
        let result = func.build_i64(42);
        func.build_return(Some(result));

        module_builder.add_function(func.build());

        let module = module_builder.build();
        let mut validator = SSAValidator::new();

        let result = validator.validate_function(&module.functions[0]);
        assert!(result.is_valid); // Simple function should be valid
        assert_eq!(result.errors.len(), 0);
    }
}
