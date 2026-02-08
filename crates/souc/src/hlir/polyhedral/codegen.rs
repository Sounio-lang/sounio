//! Polyhedral Code Generation
//!
//! This module generates HLIR code from polyhedral schedules. The code generator
//! takes a schedule tree and produces loop nests that implement that schedule.
//!
//! # Algorithm
//!
//! The codegen algorithm is based on:
//! - Quilleré et al. (2000) "Generation of Efficient Nested Loops from Polyhedra"
//! - Bastoul (2004) "Code Generation in the Polyhedral Model Is Easier Than You Think"
//! - Grosser et al. (2012) "Polly - Performing Polyhedral Optimizations on a Low-Level IR"
//!
//! # Key Steps
//!
//! 1. Convert schedule to AST (Abstract Syntax Tree)
//! 2. Generate loop bounds from domain constraints
//! 3. Generate array subscripts from access maps
//! 4. Emit HLIR instructions

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use std::collections::HashMap;

use crate::hlir::{
    BinaryOp, BlockId, HlirBlock, HlirConstant, HlirFunction, HlirInstr, HlirTerminator, HlirType,
    Op, ValueId,
};

use super::PolyResult;
use super::affine::{AffineExpr, AffineMap};
use super::analysis::ScopInfo;
use super::domain::Constraint;
use super::transform::{Schedule, ScheduleNode};

/// Code generator for polyhedral schedules
pub struct PolyhedralCodegen {
    /// Next available block ID
    next_block_id: u32,
    /// Next available value ID
    next_value_id: u32,
    /// Generated blocks
    blocks: Vec<HlirBlock>,
    /// Current block being built
    current_block: Option<BlockId>,
    /// Mapping from loop indices to HLIR values
    index_values: HashMap<String, ValueId>,
    /// Parameter values
    param_values: HashMap<String, ValueId>,
}

impl PolyhedralCodegen {
    /// Create a new code generator
    pub fn new() -> Self {
        Self {
            next_block_id: 0,
            next_value_id: 0,
            blocks: Vec::new(),
            current_block: None,
            index_values: HashMap::new(),
            param_values: HashMap::new(),
        }
    }

    /// Apply a schedule to a function, transforming the loop nests
    pub fn apply_schedule(
        &self,
        func: &mut HlirFunction,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<()> {
        // Find max block and value IDs
        let mut codegen = PolyhedralCodegen::new();
        codegen.init_from_function(func);

        // Generate code from schedule tree if available
        if let Some(tree) = &schedule.tree {
            let (entry, exit) = codegen.generate_from_tree(tree, scop, schedule)?;

            // Integrate generated code into function
            codegen.integrate_into_function(func, scop, entry, exit)?;
        } else {
            // No tree - generate from statement schedules
            codegen.generate_from_schedules(func, scop, schedule)?;
        }

        Ok(())
    }

    /// Initialize from an existing function
    fn init_from_function(&mut self, func: &HlirFunction) {
        // Find max IDs
        for block in &func.blocks {
            if block.id.0 >= self.next_block_id {
                self.next_block_id = block.id.0 + 1;
            }
            for instr in &block.instructions {
                if let Some(result) = instr.result {
                    if result.0 >= self.next_value_id {
                        self.next_value_id = result.0 + 1;
                    }
                }
            }
        }
    }

    /// Create a fresh block ID
    fn fresh_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Create a fresh value ID
    fn fresh_value_id(&mut self) -> ValueId {
        let id = ValueId(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Create a new block
    fn create_block(&mut self, label: &str) -> BlockId {
        let id = self.fresh_block_id();
        let block = HlirBlock::new(id, label);
        self.blocks.push(block);
        id
    }

    /// Get mutable reference to a block
    fn get_block_mut(&mut self, id: BlockId) -> Option<&mut HlirBlock> {
        self.blocks.iter_mut().find(|b| b.id == id)
    }

    /// Add instruction to current block
    fn emit(&mut self, instr: HlirInstr) {
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.get_block_mut(block_id) {
                block.instructions.push(instr);
            }
        }
    }

    /// Set current block terminator
    fn terminate(&mut self, term: HlirTerminator) {
        if let Some(block_id) = self.current_block {
            if let Some(block) = self.get_block_mut(block_id) {
                block.terminator = term;
            }
        }
    }

    /// Generate code from a schedule tree
    fn generate_from_tree(
        &mut self,
        tree: &ScheduleNode,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        match tree {
            ScheduleNode::Band {
                num_dims,
                tile_sizes,
                child,
                ..
            } => {
                if let Some(sizes) = tile_sizes {
                    // Generate tiled loop nest
                    self.generate_tiled_loops(*num_dims, sizes, child, scop, schedule)
                } else {
                    // Generate regular loop nest
                    self.generate_loop_band(*num_dims, child, scop, schedule)
                }
            }
            ScheduleNode::Sequence(children) => self.generate_sequence(children, scop, schedule),
            ScheduleNode::Parallel(children) => {
                // For now, treat parallel as sequence
                // Full parallel codegen would emit parallel constructs
                self.generate_sequence(children, scop, schedule)
            }
            ScheduleNode::Filter { statements, child } => {
                self.generate_filter(statements, child, scop, schedule)
            }
            ScheduleNode::Leaf => {
                // Generate statement body
                self.generate_statements(scop, schedule)
            }
        }
    }

    /// Generate a band of loops
    fn generate_loop_band(
        &mut self,
        num_dims: usize,
        child: &ScheduleNode,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        // Create blocks for the loop structure
        let preheader = self.create_block("loop_preheader");
        let header = self.create_block("loop_header");
        let body = self.create_block("loop_body");
        let latch = self.create_block("loop_latch");
        let exit = self.create_block("loop_exit");

        // Generate nested loops for each dimension
        // Simplified: generate a single loop for the first dimension
        // A full implementation would recursively generate nested loops

        self.current_block = Some(preheader);

        // Initialize loop index
        let init_val = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(init_val),
            op: Op::Const(HlirConstant::Int(0, HlirType::I64)),
            ty: HlirType::I64,
        });
        self.terminate(HlirTerminator::Branch(header));

        // Loop header with phi node
        self.current_block = Some(header);
        let loop_idx = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(loop_idx),
            op: Op::Phi {
                incoming: vec![
                    (preheader, init_val),
                    (latch, ValueId::UNIT), // Will be updated
                ],
            },
            ty: HlirType::I64,
        });

        // Store loop index
        self.index_values
            .insert(format!("i{}", num_dims - 1), loop_idx);

        // Get loop bound (simplified: use constant 100)
        let bound = self.get_loop_bound(scop, num_dims - 1);
        let bound_val = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(bound_val),
            op: Op::Const(HlirConstant::Int(bound, HlirType::I64)),
            ty: HlirType::I64,
        });

        // Compare: idx < bound
        let cmp_val = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(cmp_val),
            op: Op::Binary {
                op: BinaryOp::SLt,
                left: loop_idx,
                right: bound_val,
            },
            ty: HlirType::Bool,
        });

        self.terminate(HlirTerminator::CondBranch {
            condition: cmp_val,
            then_block: body,
            else_block: exit,
        });

        // Loop body
        self.current_block = Some(body);

        // Generate child (may be nested loops or statements)
        let (child_entry, child_exit) = self.generate_from_tree(child, scop, schedule)?;

        // Redirect body to child entry
        if let Some(body_block) = self.get_block_mut(body) {
            body_block.terminator = HlirTerminator::Branch(child_entry);
        }

        // Redirect child exit to latch
        if let Some(child_exit_block) = self.get_block_mut(child_exit) {
            child_exit_block.terminator = HlirTerminator::Branch(latch);
        }

        // Latch: increment and branch back
        self.current_block = Some(latch);
        let one = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(one),
            op: Op::Const(HlirConstant::Int(1, HlirType::I64)),
            ty: HlirType::I64,
        });

        let next_idx = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(next_idx),
            op: Op::Binary {
                op: BinaryOp::Add,
                left: loop_idx,
                right: one,
            },
            ty: HlirType::I64,
        });

        // Update phi incoming
        if let Some(header_block) = self.get_block_mut(header) {
            for instr in &mut header_block.instructions {
                if let Op::Phi { incoming } = &mut instr.op {
                    for (pred, val) in incoming {
                        if *pred == latch {
                            *val = next_idx;
                        }
                    }
                }
            }
        }

        self.terminate(HlirTerminator::Branch(header));

        // Exit block
        self.current_block = Some(exit);

        Ok((preheader, exit))
    }

    /// Generate tiled loops
    fn generate_tiled_loops(
        &mut self,
        num_dims: usize,
        tile_sizes: &[usize],
        child: &ScheduleNode,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        // For tiling, we generate:
        // for (ii = 0; ii < N; ii += tile_size)
        //   for (i = ii; i < min(ii + tile_size, N); i++)
        //     body

        // Simplified implementation: just generate outer tile loop
        // A full implementation would generate nested tile + point loops

        let entry = self.create_block("tile_entry");
        let exit = self.create_block("tile_exit");

        self.current_block = Some(entry);

        // For each dimension, generate tile loop
        // Here we just handle 1D case
        if num_dims > 0 && !tile_sizes.is_empty() {
            let tile_size = tile_sizes[0];

            // Generate tile iterator
            let tile_idx = self.fresh_value_id();
            self.emit(HlirInstr {
                result: Some(tile_idx),
                op: Op::Const(HlirConstant::Int(0, HlirType::I64)),
                ty: HlirType::I64,
            });

            // Store for inner loop bounds
            self.index_values
                .insert(format!("tile_i{}", num_dims - 1), tile_idx);
        }

        // Generate inner loops/body
        let (child_entry, child_exit) = self.generate_from_tree(child, scop, schedule)?;

        // Connect entry to child
        self.terminate(HlirTerminator::Branch(child_entry));

        // Connect child exit to our exit
        if let Some(child_exit_block) = self.get_block_mut(child_exit) {
            child_exit_block.terminator = HlirTerminator::Branch(exit);
        }

        Ok((entry, exit))
    }

    /// Generate sequence of children
    fn generate_sequence(
        &mut self,
        children: &[ScheduleNode],
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        if children.is_empty() {
            let block = self.create_block("empty_seq");
            return Ok((block, block));
        }

        let mut first_entry = None;
        let mut last_exit = None;

        for child in children {
            let (entry, exit) = self.generate_from_tree(child, scop, schedule)?;

            if first_entry.is_none() {
                first_entry = Some(entry);
            }

            // Connect previous exit to this entry
            if let Some(prev_exit) = last_exit {
                if let Some(block) = self.get_block_mut(prev_exit) {
                    block.terminator = HlirTerminator::Branch(entry);
                }
            }

            last_exit = Some(exit);
        }

        Ok((first_entry.unwrap(), last_exit.unwrap()))
    }

    /// Generate filtered statements
    fn generate_filter(
        &mut self,
        _statements: &std::collections::HashSet<usize>,
        child: &ScheduleNode,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        // For now, just generate the child
        // A full implementation would only include filtered statements
        self.generate_from_tree(child, scop, schedule)
    }

    /// Generate statement bodies
    fn generate_statements(
        &mut self,
        scop: &ScopInfo,
        _schedule: &Schedule,
    ) -> PolyResult<(BlockId, BlockId)> {
        let block = self.create_block("stmt_body");
        self.current_block = Some(block);

        // Generate code for each statement
        for stmt in &scop.statements {
            // Generate array accesses using the access maps
            for read in &stmt.reads {
                self.generate_array_read(read)?;
            }

            // The actual computation would be copied from the original code
            // For now, emit a placeholder

            for write in &stmt.writes {
                self.generate_array_write(write)?;
            }
        }

        Ok((block, block))
    }

    /// Generate an array read access
    fn generate_array_read(
        &mut self,
        access: &super::affine::AccessRelation,
    ) -> PolyResult<ValueId> {
        // Generate index computation
        let indices = self.generate_indices(&access.map)?;

        // Generate load (simplified)
        let result = self.fresh_value_id();

        // Placeholder: would need to look up actual array base pointer
        let base = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(base),
            op: Op::Const(HlirConstant::Null(HlirType::Ptr(Box::new(HlirType::I64)))),
            ty: HlirType::Ptr(Box::new(HlirType::I64)),
        });

        // GEP + Load
        let ptr = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(ptr),
            op: Op::GetElementPtr {
                base,
                index: indices[0],
            },
            ty: HlirType::Ptr(Box::new(HlirType::I64)),
        });

        self.emit(HlirInstr {
            result: Some(result),
            op: Op::Load { ptr },
            ty: HlirType::I64,
        });

        Ok(result)
    }

    /// Generate an array write access
    fn generate_array_write(&mut self, access: &super::affine::AccessRelation) -> PolyResult<()> {
        let indices = self.generate_indices(&access.map)?;

        // Placeholder base pointer
        let base = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(base),
            op: Op::Const(HlirConstant::Null(HlirType::Ptr(Box::new(HlirType::I64)))),
            ty: HlirType::Ptr(Box::new(HlirType::I64)),
        });

        // Compute address
        let ptr = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(ptr),
            op: Op::GetElementPtr {
                base,
                index: indices[0],
            },
            ty: HlirType::Ptr(Box::new(HlirType::I64)),
        });

        // Placeholder value to store
        let value = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(value),
            op: Op::Const(HlirConstant::Int(0, HlirType::I64)),
            ty: HlirType::I64,
        });

        self.emit(HlirInstr {
            result: None,
            op: Op::Store { ptr, value },
            ty: HlirType::Void,
        });

        Ok(())
    }

    /// Generate index expressions
    fn generate_indices(&mut self, map: &AffineMap) -> PolyResult<Vec<ValueId>> {
        let mut indices = Vec::new();

        for expr in &map.outputs {
            let idx = self.generate_affine_expr(expr)?;
            indices.push(idx);
        }

        Ok(indices)
    }

    /// Generate code for an affine expression
    fn generate_affine_expr(&mut self, expr: &AffineExpr) -> PolyResult<ValueId> {
        // Start with constant term
        let mut result = self.fresh_value_id();
        self.emit(HlirInstr {
            result: Some(result),
            op: Op::Const(HlirConstant::Int(expr.constant, HlirType::I64)),
            ty: HlirType::I64,
        });

        // Add iteration variable contributions
        for (i, coeff) in expr.iter_coeffs.iter().enumerate() {
            if *coeff == 0 {
                continue;
            }

            // Get the loop index value
            let idx_name = format!("i{}", i);
            let idx_val = self
                .index_values
                .get(&idx_name)
                .copied()
                .unwrap_or_else(|| {
                    // Create a placeholder
                    let v = self.fresh_value_id();
                    self.emit(HlirInstr {
                        result: Some(v),
                        op: Op::Const(HlirConstant::Int(0, HlirType::I64)),
                        ty: HlirType::I64,
                    });
                    v
                });

            // Multiply by coefficient
            let scaled = if *coeff == 1 {
                idx_val
            } else {
                let coeff_val = self.fresh_value_id();
                self.emit(HlirInstr {
                    result: Some(coeff_val),
                    op: Op::Const(HlirConstant::Int(*coeff, HlirType::I64)),
                    ty: HlirType::I64,
                });

                let prod = self.fresh_value_id();
                self.emit(HlirInstr {
                    result: Some(prod),
                    op: Op::Binary {
                        op: BinaryOp::Mul,
                        left: idx_val,
                        right: coeff_val,
                    },
                    ty: HlirType::I64,
                });
                prod
            };

            // Add to result
            let new_result = self.fresh_value_id();
            self.emit(HlirInstr {
                result: Some(new_result),
                op: Op::Binary {
                    op: BinaryOp::Add,
                    left: result,
                    right: scaled,
                },
                ty: HlirType::I64,
            });
            result = new_result;
        }

        // Add parameter contributions (similar pattern)
        for (i, coeff) in expr.param_coeffs.iter().enumerate() {
            if *coeff == 0 {
                continue;
            }

            let param_name = format!("p{}", i);
            let param_val = self
                .param_values
                .get(&param_name)
                .copied()
                .unwrap_or_else(|| {
                    let v = self.fresh_value_id();
                    self.emit(HlirInstr {
                        result: Some(v),
                        op: Op::Const(HlirConstant::Int(100, HlirType::I64)), // Default parameter value
                        ty: HlirType::I64,
                    });
                    v
                });

            let scaled = if *coeff == 1 {
                param_val
            } else {
                let coeff_val = self.fresh_value_id();
                self.emit(HlirInstr {
                    result: Some(coeff_val),
                    op: Op::Const(HlirConstant::Int(*coeff, HlirType::I64)),
                    ty: HlirType::I64,
                });

                let prod = self.fresh_value_id();
                self.emit(HlirInstr {
                    result: Some(prod),
                    op: Op::Binary {
                        op: BinaryOp::Mul,
                        left: param_val,
                        right: coeff_val,
                    },
                    ty: HlirType::I64,
                });
                prod
            };

            let new_result = self.fresh_value_id();
            self.emit(HlirInstr {
                result: Some(new_result),
                op: Op::Binary {
                    op: BinaryOp::Add,
                    left: result,
                    right: scaled,
                },
                ty: HlirType::I64,
            });
            result = new_result;
        }

        Ok(result)
    }

    /// Get loop bound from SCOP
    fn get_loop_bound(&self, scop: &ScopInfo, level: usize) -> i64 {
        // Try to get bound from loop nest
        for nest in &scop.loop_nests {
            if level < nest.loops.len() {
                if let Some(iv) = &nest.loops[level].induction_var {
                    if let super::analysis::InductionBound::Constant(c) = iv.final_value {
                        return c;
                    }
                }
            }
        }

        // Default bound
        100
    }

    /// Generate code from statement schedules (when no tree is available)
    fn generate_from_schedules(
        &mut self,
        func: &mut HlirFunction,
        scop: &ScopInfo,
        schedule: &Schedule,
    ) -> PolyResult<()> {
        // Simple approach: generate loops based on schedule dimensions
        // This is a simplified fallback when we don't have a schedule tree

        // For each loop nest, regenerate with the new schedule
        for nest in &scop.loop_nests {
            let depth = nest.depth();

            if depth == 0 {
                continue;
            }

            // Create blocks for transformed loop
            let (entry, exit) =
                self.generate_loop_band(depth, &ScheduleNode::Leaf, scop, schedule)?;

            // Find original loop blocks and replace
            // This is simplified - a real implementation would carefully
            // splice the new code into the function
        }

        // Add generated blocks to function
        for block in self.blocks.drain(..) {
            func.blocks.push(block);
        }

        Ok(())
    }

    /// Integrate generated code into the function
    fn integrate_into_function(
        &mut self,
        func: &mut HlirFunction,
        scop: &ScopInfo,
        entry: BlockId,
        exit: BlockId,
    ) -> PolyResult<()> {
        // Remove old SCOP blocks
        func.blocks.retain(|b| !scop.blocks.contains(&b.id));

        // Add new blocks
        for block in self.blocks.drain(..) {
            func.blocks.push(block);
        }

        // Update references from blocks outside SCOP to point to new entry
        for block in &mut func.blocks {
            redirect_terminator(&mut block.terminator, scop.entry, entry);
        }

        // Update references from exits to point to original successors
        // This requires knowing what the original exit targets were

        Ok(())
    }
}

impl Default for PolyhedralCodegen {
    fn default() -> Self {
        Self::new()
    }
}

/// Redirect terminator targets
fn redirect_terminator(term: &mut HlirTerminator, from: BlockId, to: BlockId) {
    match term {
        HlirTerminator::Branch(target) => {
            if *target == from {
                *target = to;
            }
        }
        HlirTerminator::CondBranch {
            then_block,
            else_block,
            ..
        } => {
            if *then_block == from {
                *then_block = to;
            }
            if *else_block == from {
                *else_block = to;
            }
        }
        HlirTerminator::Switch { default, cases, .. } => {
            if *default == from {
                *default = to;
            }
            for (_, target) in cases {
                if *target == from {
                    *target = to;
                }
            }
        }
        HlirTerminator::Return(_) | HlirTerminator::Unreachable => {}
    }
}

/// AST representation for code generation
///
/// This provides a higher-level representation that's easier to
/// generate loops from.
#[derive(Debug, Clone)]
pub enum CodegenAST {
    /// A loop with bounds and body
    Loop {
        /// Loop variable name
        var: String,
        /// Lower bound expression
        lower: AffineExpr,
        /// Upper bound expression
        upper: AffineExpr,
        /// Step (usually 1)
        step: i64,
        /// Loop body
        body: Box<CodegenAST>,
    },
    /// Sequence of AST nodes
    Sequence(Vec<CodegenAST>),
    /// If-then-else
    Conditional {
        condition: Vec<Constraint>,
        then_body: Box<CodegenAST>,
        else_body: Option<Box<CodegenAST>>,
    },
    /// Statement execution
    Statement {
        id: usize,
        iteration: Vec<AffineExpr>,
    },
    /// Empty node
    Empty,
}

impl CodegenAST {
    /// Create a simple loop
    pub fn simple_loop(var: &str, lower: i64, upper: i64, body: CodegenAST) -> Self {
        CodegenAST::Loop {
            var: var.to_string(),
            lower: AffineExpr::constant(lower),
            upper: AffineExpr::constant(upper),
            step: 1,
            body: Box::new(body),
        }
    }

    /// Create nested loops
    pub fn nested_loops(vars: &[&str], bounds: &[(i64, i64)], innermost: CodegenAST) -> Self {
        let mut result = innermost;

        for (var, (lower, upper)) in vars.iter().zip(bounds.iter()).rev() {
            result = CodegenAST::simple_loop(var, *lower, *upper, result);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codegen_creation() {
        let codegen = PolyhedralCodegen::new();
        assert!(codegen.blocks.is_empty());
    }

    #[test]
    fn test_fresh_ids() {
        let mut codegen = PolyhedralCodegen::new();

        let b1 = codegen.fresh_block_id();
        let b2 = codegen.fresh_block_id();
        assert_ne!(b1, b2);

        let v1 = codegen.fresh_value_id();
        let v2 = codegen.fresh_value_id();
        assert_ne!(v1, v2);
    }

    #[test]
    fn test_create_block() {
        let mut codegen = PolyhedralCodegen::new();
        let id = codegen.create_block("test_block");

        assert_eq!(codegen.blocks.len(), 1);
        assert_eq!(codegen.blocks[0].label, "test_block");
    }

    #[test]
    fn test_codegen_ast_simple_loop() {
        let body = CodegenAST::Empty;
        let loop_ast = CodegenAST::simple_loop("i", 0, 10, body);

        match loop_ast {
            CodegenAST::Loop { var, step, .. } => {
                assert_eq!(var, "i");
                assert_eq!(step, 1);
            }
            _ => panic!("Expected Loop AST"),
        }
    }

    #[test]
    fn test_codegen_ast_nested_loops() {
        let stmt = CodegenAST::Statement {
            id: 0,
            iteration: vec![],
        };

        let nested = CodegenAST::nested_loops(&["i", "j"], &[(0, 10), (0, 20)], stmt);

        // Should be i loop containing j loop
        match nested {
            CodegenAST::Loop { var, body, .. } => {
                assert_eq!(var, "i");
                match *body {
                    CodegenAST::Loop { var: inner_var, .. } => {
                        assert_eq!(inner_var, "j");
                    }
                    _ => panic!("Expected inner loop"),
                }
            }
            _ => panic!("Expected outer loop"),
        }
    }
}
