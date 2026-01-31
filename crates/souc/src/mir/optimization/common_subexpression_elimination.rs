//! Common Subexpression Elimination (CSE)
//!
//! This module implements global CSE for MIR, based on:
//! - Muchnick (1997) "Advanced Compiler Design and Implementation", Chapter 13
//! - Briggs et al. (1997) "Value Numbering"
//! - Click (1995) "Global Code Motion, Global Value Numbering"
//!
//! CSE identifies expressions that compute the same value and replaces
//! redundant computations with references to the first computation.
//! This pass operates on SSA form and preserves it.

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use super::pass_manager::MIRPass;
use crate::mir::{
    BlockId, MirBinaryOp, MirCompareOp, MirFunction, MirInstruction, MirModule, MirTerminator,
    MirType, MirUnaryOp, ValueId,
};
use std::collections::{HashMap, HashSet};

/// Represents an expression that can be identified for CSE.
/// We use a canonical form to identify equivalent expressions.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expression {
    /// Binary operation: op(left, right)
    Binary {
        op: MirBinaryOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    },
    /// Unary operation: op(operand)
    Unary {
        op: MirUnaryOp,
        operand: ValueId,
        ty: MirType,
    },
    /// Comparison: op(left, right)
    Compare {
        op: MirCompareOp,
        left: ValueId,
        right: ValueId,
        ty: MirType,
    },
    /// Load from memory address
    Load { address: ValueId, ty: MirType },
    /// Get element pointer
    GetElementPtr {
        base: ValueId,
        indices: Vec<ValueId>,
        ty: MirType,
    },
    /// Cast operation
    Cast {
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Zero extension
    ZExt {
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Sign extension
    SExt {
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Truncation
    Trunc {
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Float to signed int
    FPToSI { source: ValueId, target_ty: MirType },
    /// Signed int to float
    SIToFP { source: ValueId, target_ty: MirType },
    /// Float to unsigned int
    FPToUI { source: ValueId, target_ty: MirType },
    /// Unsigned int to float
    UIToFP { source: ValueId, target_ty: MirType },
    /// Pointer cast
    PtrCast {
        source: ValueId,
        source_ty: MirType,
        target_ty: MirType,
    },
    /// Select (ternary conditional)
    Select {
        condition: ValueId,
        true_value: ValueId,
        false_value: ValueId,
        ty: MirType,
    },
}

impl Expression {
    /// Create an Expression from a MirInstruction if it's a pure computation.
    /// Returns None for instructions with side effects (stores, calls, etc.)
    /// or instructions that don't produce values we want to track (constants, phi).
    pub fn from_instruction(instr: &MirInstruction) -> Option<Self> {
        match instr {
            MirInstruction::Binary {
                op,
                left,
                right,
                ty,
                ..
            } => {
                // For commutative operations, normalize by ordering operands
                let (left, right) = if op.is_commutative() && right.0 < left.0 {
                    (*right, *left)
                } else {
                    (*left, *right)
                };
                Some(Expression::Binary {
                    op: *op,
                    left,
                    right,
                    ty: ty.clone(),
                })
            }
            MirInstruction::Unary {
                op, operand, ty, ..
            } => Some(Expression::Unary {
                op: *op,
                operand: *operand,
                ty: ty.clone(),
            }),
            MirInstruction::Compare {
                op,
                left,
                right,
                ty,
                ..
            } => Some(Expression::Compare {
                op: *op,
                left: *left,
                right: *right,
                ty: ty.clone(),
            }),
            MirInstruction::GetElementPtr {
                base, indices, ty, ..
            } => Some(Expression::GetElementPtr {
                base: *base,
                indices: indices.clone(),
                ty: ty.clone(),
            }),
            MirInstruction::Cast {
                source,
                source_ty,
                target_ty,
                ..
            } => Some(Expression::Cast {
                source: *source,
                source_ty: source_ty.clone(),
                target_ty: target_ty.clone(),
            }),
            MirInstruction::ZExt {
                source,
                source_ty,
                target_ty,
                ..
            } => Some(Expression::ZExt {
                source: *source,
                source_ty: source_ty.clone(),
                target_ty: target_ty.clone(),
            }),
            MirInstruction::SExt {
                source,
                source_ty,
                target_ty,
                ..
            } => Some(Expression::SExt {
                source: *source,
                source_ty: source_ty.clone(),
                target_ty: target_ty.clone(),
            }),
            MirInstruction::Trunc {
                source,
                source_ty,
                target_ty,
                ..
            } => Some(Expression::Trunc {
                source: *source,
                source_ty: source_ty.clone(),
                target_ty: target_ty.clone(),
            }),
            MirInstruction::FPToSI {
                source, target_ty, ..
            } => Some(Expression::FPToSI {
                source: *source,
                target_ty: target_ty.clone(),
            }),
            MirInstruction::SIToFP {
                source, target_ty, ..
            } => Some(Expression::SIToFP {
                source: *source,
                target_ty: target_ty.clone(),
            }),
            MirInstruction::FPToUI {
                source, target_ty, ..
            } => Some(Expression::FPToUI {
                source: *source,
                target_ty: target_ty.clone(),
            }),
            MirInstruction::UIToFP {
                source, target_ty, ..
            } => Some(Expression::UIToFP {
                source: *source,
                target_ty: target_ty.clone(),
            }),
            MirInstruction::PtrCast {
                source,
                source_ty,
                target_ty,
                ..
            } => Some(Expression::PtrCast {
                source: *source,
                source_ty: source_ty.clone(),
                target_ty: target_ty.clone(),
            }),
            MirInstruction::Select {
                condition,
                true_value,
                false_value,
                ty,
                ..
            } => Some(Expression::Select {
                condition: *condition,
                true_value: *true_value,
                false_value: *false_value,
                ty: ty.clone(),
            }),
            // Load is tricky - only safe for CSE if we can prove no aliasing stores
            // For now, we skip loads to be conservative
            MirInstruction::Load { .. } => None,
            // These instructions have side effects or are not suitable for CSE
            MirInstruction::Store { .. }
            | MirInstruction::StoreGlobal { .. }
            | MirInstruction::LoadGlobal { .. }
            | MirInstruction::Alloca { .. }
            | MirInstruction::Call { .. }
            | MirInstruction::CallIndirect { .. }
            | MirInstruction::Const { .. }
            | MirInstruction::Phi { .. } => None,
        }
    }

    /// Get the result type of this expression
    pub fn result_type(&self) -> &MirType {
        match self {
            Expression::Binary { ty, .. }
            | Expression::Unary { ty, .. }
            | Expression::Compare { ty, .. }
            | Expression::Load { ty, .. }
            | Expression::GetElementPtr { ty, .. }
            | Expression::Select { ty, .. } => ty,
            Expression::Cast { target_ty, .. }
            | Expression::ZExt { target_ty, .. }
            | Expression::SExt { target_ty, .. }
            | Expression::Trunc { target_ty, .. }
            | Expression::FPToSI { target_ty, .. }
            | Expression::SIToFP { target_ty, .. }
            | Expression::FPToUI { target_ty, .. }
            | Expression::UIToFP { target_ty, .. }
            | Expression::PtrCast { target_ty, .. } => target_ty,
        }
    }

    /// Get all values used by this expression
    pub fn used_values(&self) -> Vec<ValueId> {
        match self {
            Expression::Binary { left, right, .. } | Expression::Compare { left, right, .. } => {
                vec![*left, *right]
            }
            Expression::Unary { operand, .. }
            | Expression::Cast {
                source: operand, ..
            }
            | Expression::ZExt {
                source: operand, ..
            }
            | Expression::SExt {
                source: operand, ..
            }
            | Expression::Trunc {
                source: operand, ..
            }
            | Expression::FPToSI {
                source: operand, ..
            }
            | Expression::SIToFP {
                source: operand, ..
            }
            | Expression::FPToUI {
                source: operand, ..
            }
            | Expression::UIToFP {
                source: operand, ..
            }
            | Expression::PtrCast {
                source: operand, ..
            } => {
                vec![*operand]
            }
            Expression::Load { address, .. } => {
                vec![*address]
            }
            Expression::GetElementPtr { base, indices, .. } => {
                let mut values = vec![*base];
                values.extend(indices.iter().copied());
                values
            }
            Expression::Select {
                condition,
                true_value,
                false_value,
                ..
            } => {
                vec![*condition, *true_value, *false_value]
            }
        }
    }
}

/// Available expressions analysis for a function.
/// Maps each block to the set of expressions available at its entry.
pub struct AvailableExpressions {
    /// Available expressions at entry of each block
    pub available_in: HashMap<BlockId, HashSet<Expression>>,
    /// Available expressions at exit of each block
    pub available_out: HashMap<BlockId, HashSet<Expression>>,
    /// Expressions generated (first computed) in each block
    pub generated: HashMap<BlockId, HashSet<Expression>>,
    /// Expressions killed in each block (operands redefined - N/A in SSA)
    pub kill: HashMap<BlockId, HashSet<Expression>>,
    /// Map from expression to the ValueId that computes it (first occurrence)
    pub expr_to_value: HashMap<Expression, ValueId>,
    /// Predecessors of each block
    pub predecessors: HashMap<BlockId, Vec<BlockId>>,
}

impl AvailableExpressions {
    /// Compute available expressions analysis for a function.
    /// This is a forward data flow analysis.
    pub fn analyze(func: &MirFunction) -> Self {
        let mut analysis = Self {
            available_in: HashMap::new(),
            available_out: HashMap::new(),
            generated: HashMap::new(),
            kill: HashMap::new(),
            expr_to_value: HashMap::new(),
            predecessors: HashMap::new(),
        };

        // Build predecessor map from terminators
        analysis.build_predecessor_map(func);

        // Compute GEN sets (expressions computed in each block)
        // In SSA form, KILL sets are empty since values are never redefined
        for block in &func.blocks {
            let mut gen_set = HashSet::new();

            for instr in &block.instructions {
                if let Some(expr) = Expression::from_instruction(instr) {
                    // Record the first computation of this expression
                    if let Some(result) = get_instruction_result(instr) {
                        if !analysis.expr_to_value.contains_key(&expr) {
                            analysis.expr_to_value.insert(expr.clone(), result);
                        }
                    }
                    gen_set.insert(expr);
                }
            }

            analysis.generated.insert(block.id, gen_set);
            analysis.kill.insert(block.id, HashSet::new()); // Empty in SSA
        }

        // Initialize available_in and available_out
        // Entry block has empty available_in
        // All other blocks start with universal set (all expressions)
        let all_expressions: HashSet<Expression> = analysis
            .generated
            .values()
            .flat_map(|s| s.iter().cloned())
            .collect();

        for block in &func.blocks {
            if block.id == func.entry_block {
                analysis.available_in.insert(block.id, HashSet::new());
            } else {
                analysis
                    .available_in
                    .insert(block.id, all_expressions.clone());
            }
            analysis.available_out.insert(block.id, HashSet::new());
        }

        // Iterative data flow analysis
        // available_out[B] = gen[B] ∪ (available_in[B] - kill[B])
        // available_in[B] = ∩ available_out[P] for all predecessors P
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 1000;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                // Compute available_in as intersection of predecessors' available_out
                let preds = analysis
                    .predecessors
                    .get(&block.id)
                    .cloned()
                    .unwrap_or_default();

                let new_in = if block.id == func.entry_block {
                    HashSet::new()
                } else if preds.is_empty() {
                    HashSet::new()
                } else {
                    let mut intersection: Option<HashSet<Expression>> = None;
                    for pred in &preds {
                        if let Some(pred_out) = analysis.available_out.get(pred) {
                            match &mut intersection {
                                None => {
                                    intersection = Some(pred_out.clone());
                                }
                                Some(inter) => {
                                    *inter = inter.intersection(pred_out).cloned().collect();
                                }
                            }
                        }
                    }
                    intersection.unwrap_or_default()
                };

                // Compute available_out = generated ∪ (available_in - kill)
                // In SSA, kill is empty, so: available_out = generated ∪ available_in
                let gen_set = analysis
                    .generated
                    .get(&block.id)
                    .cloned()
                    .unwrap_or_default();
                let new_out: HashSet<Expression> = gen_set.union(&new_in).cloned().collect();

                // Check for changes
                if analysis.available_in.get(&block.id) != Some(&new_in) {
                    analysis.available_in.insert(block.id, new_in);
                    changed = true;
                }
                if analysis.available_out.get(&block.id) != Some(&new_out) {
                    analysis.available_out.insert(block.id, new_out);
                    changed = true;
                }
            }
        }

        analysis
    }

    /// Build predecessor map from block terminators
    fn build_predecessor_map(&mut self, func: &MirFunction) {
        self.predecessors.clear();

        // Initialize all blocks with empty predecessor lists
        for block in &func.blocks {
            self.predecessors.insert(block.id, Vec::new());
        }

        // Build predecessor relationships from terminators
        for block in &func.blocks {
            let successors = get_terminator_successors(&block.terminator);
            for succ in successors {
                self.predecessors.entry(succ).or_default().push(block.id);
            }
        }
    }
}

/// Common Subexpression Elimination pass
pub struct CommonSubexpressionElimination;

impl CommonSubexpressionElimination {
    /// Create a new CSE pass
    pub fn new() -> Self {
        Self
    }

    /// Apply CSE to a function, returning true if any changes were made
    fn optimize_function(&self, func: &mut MirFunction) -> bool {
        // Run available expressions analysis
        let analysis = AvailableExpressions::analyze(func);
        let dominators = compute_block_dominators(func, &analysis.predecessors);
        let value_def_blocks = compute_value_definition_blocks(func);

        let mut modified = false;
        let mut replacements: HashMap<ValueId, ValueId> = HashMap::new();

        // Process each block
        for block_idx in 0..func.blocks.len() {
            let block_id = func.blocks[block_idx].id;

            // Get available expressions at block entry
            let mut available_at_point: HashSet<Expression> = analysis
                .available_in
                .get(&block_id)
                .cloned()
                .unwrap_or_default();

            // Map from expression to the value that computes it (within this block)
            let mut local_expr_to_value: HashMap<Expression, ValueId> = HashMap::new();

            // Populate local map with globally available expressions whose values
            // are defined in a strict dominator of this block.
            for expr in &available_at_point {
                if let Some(value) = analysis.expr_to_value.get(expr) {
                    let def_block = value_def_blocks.get(value).copied();
                    let dominates_block_entry = def_block.is_some_and(|def_block| {
                        def_block != block_id
                            && dominators
                                .get(&block_id)
                                .is_some_and(|dom_set| dom_set.contains(&def_block))
                    });
                    if dominates_block_entry {
                        local_expr_to_value.insert(expr.clone(), *value);
                    }
                }
            }

            let mut new_instructions = Vec::new();

            for instr in &func.blocks[block_idx].instructions {
                // Try to create an expression from this instruction
                if let Some(expr) = Expression::from_instruction(instr) {
                    if let Some(result) = get_instruction_result(instr) {
                        // Check if this expression is already available
                        if let Some(&existing_value) = local_expr_to_value.get(&expr) {
                            // This is a redundant computation - record replacement
                            if existing_value != result {
                                replacements.insert(result, existing_value);
                                modified = true;
                                // Skip adding this instruction
                                continue;
                            }
                        } else {
                            // This is the first computation of this expression
                            available_at_point.insert(expr.clone());
                            local_expr_to_value.insert(expr, result);
                        }
                    }
                }

                new_instructions.push(instr.clone());
            }

            func.blocks[block_idx].instructions = new_instructions;
        }

        // Apply replacements to all uses
        if !replacements.is_empty() {
            apply_replacements(func, &replacements);
        }

        modified
    }
}

/// Compute classic block dominance sets for a function.
///
/// Dominators are needed to ensure we only CSE using values defined in blocks
/// that dominate the current block. This keeps the pass SSA-safe at CFG merges
/// without introducing new phi nodes.
fn compute_block_dominators(
    func: &MirFunction,
    predecessors: &HashMap<BlockId, Vec<BlockId>>,
) -> HashMap<BlockId, HashSet<BlockId>> {
    let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();
    let all_blocks: HashSet<BlockId> = blocks.iter().copied().collect();

    let mut dominators: HashMap<BlockId, HashSet<BlockId>> = HashMap::new();
    for block_id in &blocks {
        if *block_id == func.entry_block {
            dominators.insert(*block_id, HashSet::from([*block_id]));
        } else {
            dominators.insert(*block_id, all_blocks.clone());
        }
    }

    let mut changed = true;
    let mut iterations = 0;
    const MAX_ITERATIONS: usize = 1000;

    while changed && iterations < MAX_ITERATIONS {
        changed = false;
        iterations += 1;

        for block_id in &blocks {
            if *block_id == func.entry_block {
                continue;
            }

            let preds = predecessors.get(block_id).cloned().unwrap_or_default();

            let mut new_dom = if preds.is_empty() {
                // Unreachable block: treat as dominated only by itself.
                HashSet::new()
            } else {
                let mut intersection: Option<HashSet<BlockId>> = None;
                for pred in &preds {
                    if let Some(pred_dom) = dominators.get(pred) {
                        intersection = Some(match intersection {
                            None => pred_dom.clone(),
                            Some(current) => current.intersection(pred_dom).copied().collect(),
                        });
                    }
                }
                intersection.unwrap_or_default()
            };

            new_dom.insert(*block_id);

            if dominators.get(block_id) != Some(&new_dom) {
                dominators.insert(*block_id, new_dom);
                changed = true;
            }
        }
    }

    dominators
}

/// Map each SSA value to the block that defines it.
fn compute_value_definition_blocks(func: &MirFunction) -> HashMap<ValueId, BlockId> {
    let mut def_blocks: HashMap<ValueId, BlockId> = HashMap::new();

    for block in &func.blocks {
        for instr in &block.instructions {
            if let Some(result) = get_instruction_result(instr) {
                def_blocks.insert(result, block.id);
            }
        }
    }

    def_blocks
}

impl Default for CommonSubexpressionElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl MIRPass for CommonSubexpressionElimination {
    fn name(&self) -> &'static str {
        "common-subexpression-elimination"
    }

    fn run_on_module(&self, module: &mut MirModule) -> Result<bool, String> {
        let mut modified = false;

        for func in &mut module.functions {
            if self.run_on_function(func)? {
                modified = true;
            }
        }

        Ok(modified)
    }

    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        Ok(self.optimize_function(func))
    }

    fn requires_ssa(&self) -> bool {
        true
    }

    fn preserves_ssa(&self) -> bool {
        true
    }
}

/// Get the result ValueId from an instruction, if any
fn get_instruction_result(instr: &MirInstruction) -> Option<ValueId> {
    match instr {
        MirInstruction::Load { result, .. }
        | MirInstruction::GetElementPtr { result, .. }
        | MirInstruction::LoadGlobal { result, .. }
        | MirInstruction::Alloca { result, .. }
        | MirInstruction::Cast { result, .. }
        | MirInstruction::ZExt { result, .. }
        | MirInstruction::SExt { result, .. }
        | MirInstruction::Trunc { result, .. }
        | MirInstruction::FPToSI { result, .. }
        | MirInstruction::SIToFP { result, .. }
        | MirInstruction::FPToUI { result, .. }
        | MirInstruction::UIToFP { result, .. }
        | MirInstruction::PtrCast { result, .. }
        | MirInstruction::Binary { result, .. }
        | MirInstruction::Unary { result, .. }
        | MirInstruction::Compare { result, .. }
        | MirInstruction::Const { result, .. }
        | MirInstruction::Phi { result, .. }
        | MirInstruction::Select { result, .. } => Some(*result),

        MirInstruction::Call { result, .. } | MirInstruction::CallIndirect { result, .. } => {
            *result
        }

        MirInstruction::Store { .. } | MirInstruction::StoreGlobal { .. } => None,
    }
}

/// Get successor blocks from a terminator
fn get_terminator_successors(terminator: &MirTerminator) -> Vec<BlockId> {
    match terminator {
        MirTerminator::Branch { target } => vec![*target],
        MirTerminator::CondBranch {
            true_target,
            false_target,
            ..
        } => {
            vec![*true_target, *false_target]
        }
        MirTerminator::Switch {
            default_target,
            cases,
            ..
        } => {
            let mut successors = vec![*default_target];
            successors.extend(cases.iter().map(|(_, target)| *target));
            successors
        }
        MirTerminator::Return { .. }
        | MirTerminator::Unreachable
        | MirTerminator::CallNoReturn { .. } => vec![],
    }
}

/// Apply value replacements throughout the function
fn apply_replacements(func: &mut MirFunction, replacements: &HashMap<ValueId, ValueId>) {
    // Helper to replace a value if it's in the replacement map
    let replace = |v: &mut ValueId| {
        if let Some(&new_val) = replacements.get(v) {
            *v = new_val;
        }
    };

    for block in &mut func.blocks {
        for instr in &mut block.instructions {
            match instr {
                MirInstruction::Load { address, .. } => {
                    replace(address);
                }
                MirInstruction::Store { address, value } => {
                    replace(address);
                    replace(value);
                }
                MirInstruction::GetElementPtr { base, indices, .. } => {
                    replace(base);
                    for idx in indices {
                        replace(idx);
                    }
                }
                MirInstruction::StoreGlobal { value, .. } => {
                    replace(value);
                }
                MirInstruction::Cast { source, .. }
                | MirInstruction::ZExt { source, .. }
                | MirInstruction::SExt { source, .. }
                | MirInstruction::Trunc { source, .. }
                | MirInstruction::FPToSI { source, .. }
                | MirInstruction::SIToFP { source, .. }
                | MirInstruction::FPToUI { source, .. }
                | MirInstruction::UIToFP { source, .. }
                | MirInstruction::PtrCast { source, .. } => {
                    replace(source);
                }
                MirInstruction::Binary { left, right, .. } => {
                    replace(left);
                    replace(right);
                }
                MirInstruction::Unary { operand, .. } => {
                    replace(operand);
                }
                MirInstruction::Compare { left, right, .. } => {
                    replace(left);
                    replace(right);
                }
                MirInstruction::Call { args, .. } => {
                    for arg in args {
                        replace(arg);
                    }
                }
                MirInstruction::CallIndirect { func_ptr, args, .. } => {
                    replace(func_ptr);
                    for arg in args {
                        replace(arg);
                    }
                }
                MirInstruction::Phi { incoming, .. } => {
                    for (_, value) in incoming {
                        replace(value);
                    }
                }
                MirInstruction::Select {
                    condition,
                    true_value,
                    false_value,
                    ..
                } => {
                    replace(condition);
                    replace(true_value);
                    replace(false_value);
                }
                // These don't use other values
                MirInstruction::LoadGlobal { .. }
                | MirInstruction::Alloca { .. }
                | MirInstruction::Const { .. } => {}
            }
        }

        // Also update terminator uses
        match &mut block.terminator {
            MirTerminator::CondBranch { condition, .. } => {
                replace(condition);
            }
            MirTerminator::Switch { value, .. } => {
                replace(value);
            }
            MirTerminator::Return { value: Some(value) } => {
                replace(value);
            }
            MirTerminator::CallNoReturn { args, .. } => {
                for arg in args {
                    replace(arg);
                }
            }
            MirTerminator::Branch { .. }
            | MirTerminator::Return { value: None }
            | MirTerminator::Unreachable => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::analysis::SSAValidator;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::types::MirConstant;
    use crate::mir::FuncId;

    #[test]
    fn test_expression_from_binary() {
        let instr = MirInstruction::Binary {
            result: ValueId(2),
            op: MirBinaryOp::Add,
            left: ValueId(0),
            right: ValueId(1),
            ty: MirType::I32,
        };

        let expr = Expression::from_instruction(&instr);
        assert!(expr.is_some());

        if let Some(Expression::Binary {
            op,
            left,
            right,
            ty,
        }) = expr
        {
            assert_eq!(op, MirBinaryOp::Add);
            assert_eq!(left, ValueId(0));
            assert_eq!(right, ValueId(1));
            assert_eq!(ty, MirType::I32);
        } else {
            panic!("Expected Binary expression");
        }
    }

    #[test]
    fn test_commutative_normalization() {
        // For commutative ops, operands should be normalized
        let instr1 = MirInstruction::Binary {
            result: ValueId(2),
            op: MirBinaryOp::Add,
            left: ValueId(1),  // Higher
            right: ValueId(0), // Lower
            ty: MirType::I32,
        };

        let instr2 = MirInstruction::Binary {
            result: ValueId(3),
            op: MirBinaryOp::Add,
            left: ValueId(0),  // Lower
            right: ValueId(1), // Higher
            ty: MirType::I32,
        };

        let expr1 = Expression::from_instruction(&instr1).unwrap();
        let expr2 = Expression::from_instruction(&instr2).unwrap();

        // Both should normalize to the same expression (lower id first)
        assert_eq!(expr1, expr2);
    }

    #[test]
    fn test_non_commutative_no_normalization() {
        // For non-commutative ops (like Sub), operand order matters
        let instr1 = MirInstruction::Binary {
            result: ValueId(2),
            op: MirBinaryOp::Sub,
            left: ValueId(1),
            right: ValueId(0),
            ty: MirType::I32,
        };

        let instr2 = MirInstruction::Binary {
            result: ValueId(3),
            op: MirBinaryOp::Sub,
            left: ValueId(0),
            right: ValueId(1),
            ty: MirType::I32,
        };

        let expr1 = Expression::from_instruction(&instr1).unwrap();
        let expr2 = Expression::from_instruction(&instr2).unwrap();

        // Should be different expressions
        assert_ne!(expr1, expr2);
    }

    #[test]
    fn test_expression_from_store_returns_none() {
        let instr = MirInstruction::Store {
            address: ValueId(0),
            value: ValueId(1),
        };

        let expr = Expression::from_instruction(&instr);
        assert!(expr.is_none(), "Store should not produce an expression");
    }

    #[test]
    fn test_expression_from_call_returns_none() {
        let instr = MirInstruction::Call {
            result: Some(ValueId(0)),
            func_name: "foo".to_string(),
            args: vec![],
            ty: MirType::I32,
        };

        let expr = Expression::from_instruction(&instr);
        assert!(
            expr.is_none(),
            "Call should not produce an expression (side effects)"
        );
    }

    #[test]
    fn test_cse_simple() {
        // Create: x = a + b; y = a + b; return y
        // After CSE: x = a + b; return x
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::I32);

        // a = 10
        let a = builder.fresh_value();
        builder.build_const(a, MirConstant::Int(10), MirType::I32);

        // b = 20
        let b = builder.fresh_value();
        builder.build_const(b, MirConstant::Int(20), MirType::I32);

        // x = a + b
        let x = builder.fresh_value();
        builder.build_add(x, a, b, MirType::I32);

        // y = a + b (redundant)
        let y = builder.fresh_value();
        builder.build_add(y, a, b, MirType::I32);

        // return y
        builder.build_return(Some(y));

        let mut func = builder.build();

        // Run CSE
        let pass = CommonSubexpressionElimination::new();
        let modified = pass
            .run_on_function(&mut func)
            .expect("pass should succeed");

        assert!(modified, "CSE should modify the function");

        // Check that there's only one add instruction now
        let entry = func.entry();
        let add_count = entry
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    MirInstruction::Binary {
                        op: MirBinaryOp::Add,
                        ..
                    }
                )
            })
            .count();

        assert_eq!(
            add_count, 1,
            "Should have only one add instruction after CSE"
        );
    }

    #[test]
    fn test_cse_preserves_ssa_at_control_flow_merge() {
        // Diamond CFG:
        // entry -> then -> merge
        // entry -> else -> merge
        //
        // Both branches compute the same expression (a + b). The merge block also
        // computes (a + b) twice. CSE must NOT replace the merge computation with
        // a branch-local value (would violate SSA dominance), but SHOULD still
        // eliminate the redundant second computation within the merge block.

        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::I32);

        let then_block = builder.create_block("then");
        let else_block = builder.create_block("else");
        let merge_block = builder.create_block("merge");

        let a = builder.build_i32(10);
        let b = builder.build_i32(20);
        let cond = builder.build_bool(true);
        builder.build_cond_branch(cond, then_block, else_block);

        builder.switch_to_block(then_block);
        let t = builder.fresh_value();
        builder.build_add(t, a, b, MirType::I32);
        builder.build_branch(merge_block);

        builder.switch_to_block(else_block);
        let e = builder.fresh_value();
        builder.build_add(e, a, b, MirType::I32);
        builder.build_branch(merge_block);

        builder.switch_to_block(merge_block);
        let m1 = builder.fresh_value();
        builder.build_add(m1, a, b, MirType::I32);
        let m2 = builder.fresh_value();
        builder.build_add(m2, a, b, MirType::I32);
        builder.build_return(Some(m2));

        let mut func = builder.build();

        let pass = CommonSubexpressionElimination::new();
        let modified = pass
            .run_on_function(&mut func)
            .expect("pass should succeed");
        assert!(
            modified,
            "CSE should remove the redundant merge computation"
        );

        let ssa_result = SSAValidator::new().validate_function(&func);
        assert!(
            ssa_result.is_valid,
            "CSE must preserve SSA dominance:\n{}",
            ssa_result
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>()
                .join("\n")
        );

        let merge = func
            .get_block(merge_block)
            .expect("merge block should exist");
        let merge_add_count = merge
            .instructions
            .iter()
            .filter(|i| {
                matches!(
                    i,
                    MirInstruction::Binary {
                        op: MirBinaryOp::Add,
                        ..
                    }
                )
            })
            .count();
        assert_eq!(
            merge_add_count, 1,
            "merge block should have only one add after CSE"
        );
    }

    #[test]
    fn test_cse_pass_name() {
        let pass = CommonSubexpressionElimination::new();
        assert_eq!(pass.name(), "common-subexpression-elimination");
    }

    #[test]
    fn test_cse_requires_ssa() {
        let pass = CommonSubexpressionElimination::new();
        assert!(pass.requires_ssa());
        assert!(pass.preserves_ssa());
    }

    #[test]
    fn test_terminator_successors() {
        let branch = MirTerminator::Branch { target: BlockId(1) };
        assert_eq!(get_terminator_successors(&branch), vec![BlockId(1)]);

        let cond_branch = MirTerminator::CondBranch {
            condition: ValueId(0),
            true_target: BlockId(1),
            false_target: BlockId(2),
        };
        assert_eq!(
            get_terminator_successors(&cond_branch),
            vec![BlockId(1), BlockId(2)]
        );

        let ret = MirTerminator::Return { value: None };
        assert!(get_terminator_successors(&ret).is_empty());
    }
}
