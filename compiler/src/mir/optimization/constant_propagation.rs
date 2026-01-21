//! Sparse Conditional Constant Propagation (SCCP)
//!
//! This module implements SCCP for MIR, based on:
//! - Wegman & Zadeck (1991) "Constant Propagation with Conditional Branches"
//! - Knoop et al. (1994) "Lazy Code Motion"
//!
//! SCCP is more powerful than simple constant propagation because it:
//! - Tracks values using a lattice (Top -> Constant -> Bottom)
//! - Handles phi nodes correctly
//! - Can propagate through conditional branches
//! - Detects unreachable code

use std::collections::{HashMap, HashSet, VecDeque};
use crate::mir::{
    MirModule, MirFunction, MirBlock, BlockId, ValueId,
    MirInstruction, MirTerminator, MirConstant, MirType,
    MirBinaryOp, MirUnaryOp, MirCompareOp,
};
use super::pass_manager::MIRPass;

/// Constant Propagation optimization pass using SCCP algorithm
pub struct ConstantPropagation;

/// Lattice value for SCCP
///
/// The lattice has three levels:
/// - Top: Value is unknown (not yet computed)
/// - Constant: Value is a known constant
/// - Bottom: Value is overdefined (multiple possible values)
#[derive(Clone, Debug, PartialEq)]
pub enum LatticeValue {
    /// Unknown value (top of lattice)
    Top,
    /// Known constant value
    Constant(MirConstant),
    /// Overdefined value (bottom of lattice)
    Bottom,
}

impl LatticeValue {
    /// Meet operation for the lattice
    ///
    /// Meet semantics:
    /// - Top meet X = X
    /// - X meet Top = X
    /// - Bottom meet X = Bottom
    /// - X meet Bottom = Bottom
    /// - Const(a) meet Const(b) = Const(a) if a == b, else Bottom
    pub fn meet(&self, other: &LatticeValue) -> LatticeValue {
        match (self, other) {
            (LatticeValue::Top, x) => x.clone(),
            (x, LatticeValue::Top) => x.clone(),
            (LatticeValue::Bottom, _) | (_, LatticeValue::Bottom) => LatticeValue::Bottom,
            (LatticeValue::Constant(a), LatticeValue::Constant(b)) => {
                if a == b {
                    LatticeValue::Constant(a.clone())
                } else {
                    LatticeValue::Bottom
                }
            }
        }
    }

    /// Check if this is a constant
    pub fn is_constant(&self) -> bool {
        matches!(self, LatticeValue::Constant(_))
    }

    /// Get the constant value if this is a constant
    pub fn as_constant(&self) -> Option<&MirConstant> {
        match self {
            LatticeValue::Constant(c) => Some(c),
            _ => None,
        }
    }
}

/// SCCP Analysis state
struct SCCPAnalysis {
    /// Lattice values for each SSA value
    values: HashMap<ValueId, LatticeValue>,
    /// Set of executable edges (from_block, to_block)
    executable_edges: HashSet<(BlockId, BlockId)>,
    /// Set of executable blocks
    executable_blocks: HashSet<BlockId>,
    /// Worklist of SSA values to process
    ssa_worklist: VecDeque<ValueId>,
    /// Worklist of CFG edges to process
    cfg_worklist: VecDeque<(BlockId, BlockId)>,
    /// Map from value to instructions that use it
    use_map: HashMap<ValueId, Vec<(BlockId, usize)>>,
    /// Map from block to its predecessors
    predecessors: HashMap<BlockId, Vec<BlockId>>,
}

impl SCCPAnalysis {
    fn new() -> Self {
        Self {
            values: HashMap::new(),
            executable_edges: HashSet::new(),
            executable_blocks: HashSet::new(),
            ssa_worklist: VecDeque::new(),
            cfg_worklist: VecDeque::new(),
            use_map: HashMap::new(),
            predecessors: HashMap::new(),
        }
    }

    /// Get the lattice value for a ValueId
    fn get_value(&self, value: ValueId) -> LatticeValue {
        self.values.get(&value).cloned().unwrap_or(LatticeValue::Top)
    }

    /// Set the lattice value for a ValueId, returning true if changed
    fn set_value(&mut self, value: ValueId, lattice: LatticeValue) -> bool {
        let current = self.get_value(value);
        let new_value = current.meet(&lattice);

        if new_value != current {
            self.values.insert(value, new_value);
            true
        } else {
            false
        }
    }

    /// Mark an edge as executable
    fn mark_edge_executable(&mut self, from: BlockId, to: BlockId) {
        if self.executable_edges.insert((from, to)) {
            self.cfg_worklist.push_back((from, to));
        }
    }

    /// Check if a block has been reached
    fn is_block_executable(&self, block: BlockId) -> bool {
        self.executable_blocks.contains(&block)
    }

    /// Build the use map for the function
    fn build_use_map(&mut self, func: &MirFunction) {
        self.use_map.clear();
        self.predecessors.clear();

        for block in &func.blocks {
            // Build predecessor map from terminators
            match &block.terminator {
                MirTerminator::Branch { target } => {
                    self.predecessors.entry(*target).or_default().push(block.id);
                }
                MirTerminator::CondBranch { true_target, false_target, .. } => {
                    self.predecessors.entry(*true_target).or_default().push(block.id);
                    self.predecessors.entry(*false_target).or_default().push(block.id);
                }
                MirTerminator::Switch { default_target, cases, .. } => {
                    self.predecessors.entry(*default_target).or_default().push(block.id);
                    for (_, target) in cases {
                        self.predecessors.entry(*target).or_default().push(block.id);
                    }
                }
                _ => {}
            }

            // Build use map from instructions
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                for used_value in get_instruction_uses(instr) {
                    self.use_map
                        .entry(used_value)
                        .or_default()
                        .push((block.id, instr_idx));
                }
            }
        }
    }

    /// Run SCCP analysis on a function
    fn analyze(&mut self, func: &MirFunction) {
        // Initialize
        self.values.clear();
        self.executable_edges.clear();
        self.executable_blocks.clear();
        self.ssa_worklist.clear();
        self.cfg_worklist.clear();

        self.build_use_map(func);

        // Mark entry block as executable
        if !func.blocks.is_empty() {
            let entry_id = func.entry_block;
            self.executable_blocks.insert(entry_id);

            // Initialize parameters as Bottom (unknown at compile time)
            for (idx, _) in func.params.iter().enumerate() {
                self.values.insert(ValueId(idx), LatticeValue::Bottom);
            }

            // Process entry block
            self.process_block(func, entry_id);
        }

        // Main worklist loop
        while !self.cfg_worklist.is_empty() || !self.ssa_worklist.is_empty() {
            // Process CFG worklist
            while let Some((from, to)) = self.cfg_worklist.pop_front() {
                if !self.executable_blocks.contains(&to) {
                    self.executable_blocks.insert(to);
                    self.process_block(func, to);
                } else {
                    // Block already executable, but new edge - re-evaluate phi nodes
                    if let Some(block) = func.get_block(to) {
                        for (idx, instr) in block.instructions.iter().enumerate() {
                            if let MirInstruction::Phi { result, .. } = instr {
                                let new_value = self.evaluate_phi(func, block, instr);
                                if self.set_value(*result, new_value) {
                                    self.add_uses_to_worklist(*result);
                                }
                            }
                        }
                    }
                }
            }

            // Process SSA worklist
            while let Some(value) = self.ssa_worklist.pop_front() {
                if let Some(uses) = self.use_map.get(&value).cloned() {
                    for (block_id, instr_idx) in uses {
                        if self.is_block_executable(block_id) {
                            if let Some(block) = func.get_block(block_id) {
                                if let Some(instr) = block.instructions.get(instr_idx) {
                                    self.evaluate_instruction(func, block, instr);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    /// Process all instructions in a block
    fn process_block(&mut self, func: &MirFunction, block_id: BlockId) {
        let block = match func.get_block(block_id) {
            Some(b) => b,
            None => return,
        };

        // Evaluate all instructions
        for instr in &block.instructions {
            self.evaluate_instruction(func, block, instr);
        }

        // Evaluate terminator
        self.evaluate_terminator(func, block);
    }

    /// Evaluate an instruction
    fn evaluate_instruction(&mut self, func: &MirFunction, block: &MirBlock, instr: &MirInstruction) {
        match instr {
            MirInstruction::Const { result, value, .. } => {
                if self.set_value(*result, LatticeValue::Constant(value.clone())) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Binary { result, op, left, right, ty } => {
                let left_val = self.get_value(*left);
                let right_val = self.get_value(*right);
                let new_value = self.evaluate_binary(*op, &left_val, &right_val, ty);
                if self.set_value(*result, new_value) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Unary { result, op, operand, ty } => {
                let operand_val = self.get_value(*operand);
                let new_value = self.evaluate_unary(*op, &operand_val, ty);
                if self.set_value(*result, new_value) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Compare { result, op, left, right, ty } => {
                let left_val = self.get_value(*left);
                let right_val = self.get_value(*right);
                let new_value = self.evaluate_compare(*op, &left_val, &right_val, ty);
                if self.set_value(*result, new_value) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Phi { result, .. } => {
                let new_value = self.evaluate_phi(func, block, instr);
                if self.set_value(*result, new_value) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Select { result, condition, true_value, false_value, .. } => {
                let cond_val = self.get_value(*condition);
                let true_val = self.get_value(*true_value);
                let false_val = self.get_value(*false_value);
                let new_value = self.evaluate_select(&cond_val, &true_val, &false_val);
                if self.set_value(*result, new_value) {
                    self.add_uses_to_worklist(*result);
                }
            }

            // Instructions that produce Bottom (conservative)
            MirInstruction::Load { result, .. }
            | MirInstruction::LoadGlobal { result, .. }
            | MirInstruction::Alloca { result, .. }
            | MirInstruction::GetElementPtr { result, .. }
            | MirInstruction::Cast { result, .. }
            | MirInstruction::ZExt { result, .. }
            | MirInstruction::SExt { result, .. }
            | MirInstruction::Trunc { result, .. }
            | MirInstruction::FPToSI { result, .. }
            | MirInstruction::SIToFP { result, .. }
            | MirInstruction::FPToUI { result, .. }
            | MirInstruction::UIToFP { result, .. }
            | MirInstruction::PtrCast { result, .. } => {
                if self.set_value(*result, LatticeValue::Bottom) {
                    self.add_uses_to_worklist(*result);
                }
            }

            MirInstruction::Call { result: Some(result), .. }
            | MirInstruction::CallIndirect { result: Some(result), .. } => {
                // Function calls are conservatively Bottom
                if self.set_value(*result, LatticeValue::Bottom) {
                    self.add_uses_to_worklist(*result);
                }
            }

            // Instructions without results
            MirInstruction::Store { .. }
            | MirInstruction::StoreGlobal { .. }
            | MirInstruction::Call { result: None, .. }
            | MirInstruction::CallIndirect { result: None, .. } => {}
        }
    }

    /// Evaluate a terminator to determine executable edges
    fn evaluate_terminator(&mut self, _func: &MirFunction, block: &MirBlock) {
        match &block.terminator {
            MirTerminator::Branch { target } => {
                self.mark_edge_executable(block.id, *target);
            }

            MirTerminator::CondBranch { condition, true_target, false_target } => {
                let cond_val = self.get_value(*condition);
                match cond_val {
                    LatticeValue::Constant(MirConstant::Bool(true)) => {
                        self.mark_edge_executable(block.id, *true_target);
                    }
                    LatticeValue::Constant(MirConstant::Bool(false)) => {
                        self.mark_edge_executable(block.id, *false_target);
                    }
                    _ => {
                        // Unknown or Bottom - both edges executable
                        self.mark_edge_executable(block.id, *true_target);
                        self.mark_edge_executable(block.id, *false_target);
                    }
                }
            }

            MirTerminator::Switch { value, default_target, cases } => {
                let switch_val = self.get_value(*value);
                match switch_val {
                    LatticeValue::Constant(MirConstant::Int(v)) => {
                        // Known constant - find matching case
                        let mut found = false;
                        for (case_val, target) in cases {
                            if *case_val == v {
                                self.mark_edge_executable(block.id, *target);
                                found = true;
                                break;
                            }
                        }
                        if !found {
                            self.mark_edge_executable(block.id, *default_target);
                        }
                    }
                    _ => {
                        // Unknown - all edges executable
                        self.mark_edge_executable(block.id, *default_target);
                        for (_, target) in cases {
                            self.mark_edge_executable(block.id, *target);
                        }
                    }
                }
            }

            MirTerminator::Return { .. }
            | MirTerminator::Unreachable
            | MirTerminator::CallNoReturn { .. } => {
                // No outgoing edges
            }
        }
    }

    /// Evaluate a phi instruction
    fn evaluate_phi(&self, _func: &MirFunction, block: &MirBlock, instr: &MirInstruction) -> LatticeValue {
        if let MirInstruction::Phi { incoming, .. } = instr {
            let mut result = LatticeValue::Top;

            for (pred_block, value) in incoming {
                // Only consider values from executable edges
                if self.executable_edges.contains(&(*pred_block, block.id)) {
                    let val = self.get_value(*value);
                    result = result.meet(&val);
                }
            }

            result
        } else {
            LatticeValue::Bottom
        }
    }

    /// Evaluate a select instruction
    fn evaluate_select(&self, cond: &LatticeValue, true_val: &LatticeValue, false_val: &LatticeValue) -> LatticeValue {
        match cond {
            LatticeValue::Top => LatticeValue::Top,
            LatticeValue::Constant(MirConstant::Bool(true)) => true_val.clone(),
            LatticeValue::Constant(MirConstant::Bool(false)) => false_val.clone(),
            LatticeValue::Constant(_) => LatticeValue::Bottom, // Non-bool condition
            LatticeValue::Bottom => true_val.meet(false_val),
        }
    }

    /// Evaluate a binary operation
    fn evaluate_binary(&self, op: MirBinaryOp, left: &LatticeValue, right: &LatticeValue, ty: &MirType) -> LatticeValue {
        match (left, right) {
            (LatticeValue::Top, _) | (_, LatticeValue::Top) => LatticeValue::Top,
            (LatticeValue::Bottom, _) | (_, LatticeValue::Bottom) => LatticeValue::Bottom,
            (LatticeValue::Constant(l), LatticeValue::Constant(r)) => {
                self.fold_binary(op, l, r, ty)
            }
        }
    }

    /// Fold a binary operation with constant operands
    fn fold_binary(&self, op: MirBinaryOp, left: &MirConstant, right: &MirConstant, ty: &MirType) -> LatticeValue {
        // Handle integer operations
        if let (MirConstant::Int(l), MirConstant::Int(r)) = (left, right) {
            let result = match op {
                MirBinaryOp::Add => l.checked_add(*r),
                MirBinaryOp::Sub => l.checked_sub(*r),
                MirBinaryOp::Mul => l.checked_mul(*r),
                MirBinaryOp::Div => {
                    if *r == 0 {
                        return LatticeValue::Bottom; // Division by zero
                    }
                    l.checked_div(*r)
                }
                MirBinaryOp::Rem => {
                    if *r == 0 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_rem(*r)
                }
                MirBinaryOp::And => Some(l & r),
                MirBinaryOp::Or => Some(l | r),
                MirBinaryOp::Xor => Some(l ^ r),
                MirBinaryOp::Shl => {
                    if *r < 0 || *r >= 64 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_shl(*r as u32)
                }
                MirBinaryOp::LShr => {
                    if *r < 0 || *r >= 64 {
                        return LatticeValue::Bottom;
                    }
                    // Logical shift right for signed values
                    Some((*l as u64 >> *r as u32) as i64)
                }
                MirBinaryOp::AShr => {
                    if *r < 0 || *r >= 64 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_shr(*r as u32)
                }
            };

            match result {
                Some(v) => LatticeValue::Constant(MirConstant::Int(v)),
                None => LatticeValue::Bottom, // Overflow
            }
        }
        // Handle unsigned integer operations
        else if let (MirConstant::UInt(l), MirConstant::UInt(r)) = (left, right) {
            let result = match op {
                MirBinaryOp::Add => l.checked_add(*r),
                MirBinaryOp::Sub => l.checked_sub(*r),
                MirBinaryOp::Mul => l.checked_mul(*r),
                MirBinaryOp::Div => {
                    if *r == 0 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_div(*r)
                }
                MirBinaryOp::Rem => {
                    if *r == 0 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_rem(*r)
                }
                MirBinaryOp::And => Some(l & r),
                MirBinaryOp::Or => Some(l | r),
                MirBinaryOp::Xor => Some(l ^ r),
                MirBinaryOp::Shl => {
                    if *r >= 64 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_shl(*r as u32)
                }
                MirBinaryOp::LShr | MirBinaryOp::AShr => {
                    if *r >= 64 {
                        return LatticeValue::Bottom;
                    }
                    l.checked_shr(*r as u32)
                }
            };

            match result {
                Some(v) => LatticeValue::Constant(MirConstant::UInt(v)),
                None => LatticeValue::Bottom,
            }
        }
        // Handle floating point operations (Float stores f64 as String for hashing)
        else if let (MirConstant::Float(l_str), MirConstant::Float(r_str)) = (left, right) {
            let l: f64 = match l_str.parse() {
                Ok(v) => v,
                Err(_) => return LatticeValue::Bottom,
            };
            let r: f64 = match r_str.parse() {
                Ok(v) => v,
                Err(_) => return LatticeValue::Bottom,
            };

            let result = match op {
                MirBinaryOp::Add => l + r,
                MirBinaryOp::Sub => l - r,
                MirBinaryOp::Mul => l * r,
                MirBinaryOp::Div => {
                    if r == 0.0 {
                        return LatticeValue::Bottom;
                    }
                    l / r
                }
                MirBinaryOp::Rem => {
                    if r == 0.0 {
                        return LatticeValue::Bottom;
                    }
                    l % r
                }
                // Bitwise ops don't apply to floats
                _ => return LatticeValue::Bottom,
            };

            if result.is_finite() {
                LatticeValue::Constant(MirConstant::Float(result.to_string()))
            } else {
                LatticeValue::Bottom
            }
        }
        // Handle boolean operations
        else if let (MirConstant::Bool(l), MirConstant::Bool(r)) = (left, right) {
            let result = match op {
                MirBinaryOp::And => *l && *r,
                MirBinaryOp::Or => *l || *r,
                MirBinaryOp::Xor => *l ^ *r,
                _ => return LatticeValue::Bottom,
            };
            LatticeValue::Constant(MirConstant::Bool(result))
        }
        else {
            LatticeValue::Bottom
        }
    }

    /// Evaluate a unary operation
    fn evaluate_unary(&self, op: MirUnaryOp, operand: &LatticeValue, ty: &MirType) -> LatticeValue {
        match operand {
            LatticeValue::Top => LatticeValue::Top,
            LatticeValue::Bottom => LatticeValue::Bottom,
            LatticeValue::Constant(c) => self.fold_unary(op, c, ty),
        }
    }

    /// Fold a unary operation with constant operand
    fn fold_unary(&self, op: MirUnaryOp, operand: &MirConstant, _ty: &MirType) -> LatticeValue {
        match op {
            MirUnaryOp::Neg => {
                match operand {
                    MirConstant::Int(v) => {
                        match v.checked_neg() {
                            Some(r) => LatticeValue::Constant(MirConstant::Int(r)),
                            None => LatticeValue::Bottom,
                        }
                    }
                    MirConstant::Float(v_str) => {
                        if let Ok(v) = v_str.parse::<f64>() {
                            LatticeValue::Constant(MirConstant::Float((-v).to_string()))
                        } else {
                            LatticeValue::Bottom
                        }
                    }
                    _ => LatticeValue::Bottom,
                }
            }
            MirUnaryOp::FNeg => {
                match operand {
                    MirConstant::Float(v_str) => {
                        if let Ok(v) = v_str.parse::<f64>() {
                            LatticeValue::Constant(MirConstant::Float((-v).to_string()))
                        } else {
                            LatticeValue::Bottom
                        }
                    }
                    _ => LatticeValue::Bottom,
                }
            }
            MirUnaryOp::Not => {
                match operand {
                    MirConstant::Bool(v) => {
                        LatticeValue::Constant(MirConstant::Bool(!v))
                    }
                    _ => LatticeValue::Bottom,
                }
            }
            MirUnaryOp::BitNot => {
                match operand {
                    MirConstant::Int(v) => {
                        LatticeValue::Constant(MirConstant::Int(!v))
                    }
                    MirConstant::UInt(v) => {
                        LatticeValue::Constant(MirConstant::UInt(!v))
                    }
                    _ => LatticeValue::Bottom,
                }
            }
        }
    }

    /// Evaluate a comparison operation
    fn evaluate_compare(&self, op: MirCompareOp, left: &LatticeValue, right: &LatticeValue, _ty: &MirType) -> LatticeValue {
        match (left, right) {
            (LatticeValue::Top, _) | (_, LatticeValue::Top) => LatticeValue::Top,
            (LatticeValue::Bottom, _) | (_, LatticeValue::Bottom) => LatticeValue::Bottom,
            (LatticeValue::Constant(l), LatticeValue::Constant(r)) => {
                self.fold_compare(op, l, r)
            }
        }
    }

    /// Fold a comparison with constant operands
    fn fold_compare(&self, op: MirCompareOp, left: &MirConstant, right: &MirConstant) -> LatticeValue {
        // Integer comparisons
        if let (MirConstant::Int(l), MirConstant::Int(r)) = (left, right) {
            let result = match op {
                MirCompareOp::Eq | MirCompareOp::FEq => l == r,
                MirCompareOp::Ne | MirCompareOp::FNe => l != r,
                MirCompareOp::Lt | MirCompareOp::FLt => l < r,
                MirCompareOp::Le | MirCompareOp::FLe => l <= r,
                MirCompareOp::Gt | MirCompareOp::FGt => l > r,
                MirCompareOp::Ge | MirCompareOp::FGe => l >= r,
            };
            return LatticeValue::Constant(MirConstant::Bool(result));
        }

        // Unsigned integer comparisons
        if let (MirConstant::UInt(l), MirConstant::UInt(r)) = (left, right) {
            let result = match op {
                MirCompareOp::Eq | MirCompareOp::FEq => l == r,
                MirCompareOp::Ne | MirCompareOp::FNe => l != r,
                MirCompareOp::Lt | MirCompareOp::FLt => l < r,
                MirCompareOp::Le | MirCompareOp::FLe => l <= r,
                MirCompareOp::Gt | MirCompareOp::FGt => l > r,
                MirCompareOp::Ge | MirCompareOp::FGe => l >= r,
            };
            return LatticeValue::Constant(MirConstant::Bool(result));
        }

        // Float comparisons
        if let (MirConstant::Float(l), MirConstant::Float(r)) = (left, right) {
            let result = match op {
                MirCompareOp::Eq | MirCompareOp::FEq => l == r,
                MirCompareOp::Ne | MirCompareOp::FNe => l != r,
                MirCompareOp::Lt | MirCompareOp::FLt => l < r,
                MirCompareOp::Le | MirCompareOp::FLe => l <= r,
                MirCompareOp::Gt | MirCompareOp::FGt => l > r,
                MirCompareOp::Ge | MirCompareOp::FGe => l >= r,
            };
            return LatticeValue::Constant(MirConstant::Bool(result));
        }

        // Boolean comparisons
        if let (MirConstant::Bool(l), MirConstant::Bool(r)) = (left, right) {
            let result = match op {
                MirCompareOp::Eq | MirCompareOp::FEq => l == r,
                MirCompareOp::Ne | MirCompareOp::FNe => l != r,
                _ => return LatticeValue::Bottom,
            };
            return LatticeValue::Constant(MirConstant::Bool(result));
        }

        LatticeValue::Bottom
    }

    /// Add all uses of a value to the worklist
    fn add_uses_to_worklist(&mut self, value: ValueId) {
        self.ssa_worklist.push_back(value);
    }
}

/// Get all values used by an instruction
fn get_instruction_uses(instr: &MirInstruction) -> Vec<ValueId> {
    match instr {
        MirInstruction::Load { address, .. } => vec![*address],
        MirInstruction::Store { address, value } => vec![*address, *value],
        MirInstruction::GetElementPtr { base, indices, .. } => {
            let mut uses = vec![*base];
            uses.extend(indices.iter().copied());
            uses
        }
        MirInstruction::LoadGlobal { .. } => vec![],
        MirInstruction::StoreGlobal { value, .. } => vec![*value],
        MirInstruction::Alloca { .. } => vec![],
        MirInstruction::Cast { source, .. }
        | MirInstruction::ZExt { source, .. }
        | MirInstruction::SExt { source, .. }
        | MirInstruction::Trunc { source, .. }
        | MirInstruction::FPToSI { source, .. }
        | MirInstruction::SIToFP { source, .. }
        | MirInstruction::FPToUI { source, .. }
        | MirInstruction::UIToFP { source, .. }
        | MirInstruction::PtrCast { source, .. } => vec![*source],
        MirInstruction::Binary { left, right, .. } => vec![*left, *right],
        MirInstruction::Unary { operand, .. } => vec![*operand],
        MirInstruction::Compare { left, right, .. } => vec![*left, *right],
        MirInstruction::Call { args, .. } => args.clone(),
        MirInstruction::CallIndirect { func_ptr, args, .. } => {
            let mut uses = vec![*func_ptr];
            uses.extend(args.iter().copied());
            uses
        }
        MirInstruction::Const { .. } => vec![],
        MirInstruction::Phi { incoming, .. } => {
            incoming.iter().map(|(_, v)| *v).collect()
        }
        MirInstruction::Select { condition, true_value, false_value, .. } => {
            vec![*condition, *true_value, *false_value]
        }
    }
}

impl MIRPass for ConstantPropagation {
    fn name(&self) -> &'static str {
        "constant-propagation"
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
        // Run SCCP analysis
        let mut analysis = SCCPAnalysis::new();
        analysis.analyze(func);

        // Apply transformations
        let mut modified = false;

        for block in &mut func.blocks {
            // Skip unreachable blocks
            if !analysis.is_block_executable(block.id) {
                continue;
            }

            let mut new_instructions = Vec::new();

            for instr in &block.instructions {
                let replacement = replace_with_constant(&analysis, instr);

                match replacement {
                    Some(new_instr) => {
                        new_instructions.push(new_instr);
                        modified = true;
                    }
                    None => {
                        new_instructions.push(instr.clone());
                    }
                }
            }

            block.instructions = new_instructions;

            // Simplify conditional branches with known conditions
            match &block.terminator {
                MirTerminator::CondBranch { condition, true_target, false_target } => {
                    let cond_val = analysis.get_value(*condition);
                    match cond_val {
                        LatticeValue::Constant(MirConstant::Bool(true)) => {
                            block.terminator = MirTerminator::Branch { target: *true_target };
                            modified = true;
                        }
                        LatticeValue::Constant(MirConstant::Bool(false)) => {
                            block.terminator = MirTerminator::Branch { target: *false_target };
                            modified = true;
                        }
                        _ => {}
                    }
                }
                MirTerminator::Switch { value, default_target, cases } => {
                    let switch_val = analysis.get_value(*value);
                    if let LatticeValue::Constant(MirConstant::Int(v)) = switch_val {
                        let target = cases
                            .iter()
                            .find(|(case_val, _)| *case_val == v)
                            .map(|(_, target)| *target)
                            .unwrap_or(*default_target);
                        block.terminator = MirTerminator::Branch { target };
                        modified = true;
                    }
                }
                _ => {}
            }
        }

        Ok(modified)
    }

    fn requires_ssa(&self) -> bool {
        true
    }

    fn preserves_ssa(&self) -> bool {
        true
    }
}

/// Replace an instruction with a constant if its result is known
fn replace_with_constant(analysis: &SCCPAnalysis, instr: &MirInstruction) -> Option<MirInstruction> {
    // Get the result value ID
    let result = match instr {
        MirInstruction::Binary { result, .. }
        | MirInstruction::Unary { result, .. }
        | MirInstruction::Compare { result, .. }
        | MirInstruction::Phi { result, .. }
        | MirInstruction::Select { result, .. } => *result,

        // Don't replace constants with constants
        MirInstruction::Const { .. } => return None,

        // Don't replace instructions without results or with side effects
        _ => return None,
    };

    // Check if the result is a known constant
    let lattice_value = analysis.get_value(result);

    if let LatticeValue::Constant(constant) = lattice_value {
        let ty = constant.ty();
        Some(MirInstruction::Const {
            result,
            value: constant,
            ty,
        })
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::{FuncId, MirType};

    #[test]
    fn test_lattice_meet() {
        let top = LatticeValue::Top;
        let bottom = LatticeValue::Bottom;
        let const_42 = LatticeValue::Constant(MirConstant::Int(42));
        let const_10 = LatticeValue::Constant(MirConstant::Int(10));

        // Top meet X = X
        assert_eq!(top.meet(&const_42), const_42);
        assert_eq!(top.meet(&bottom), bottom);

        // X meet Top = X
        assert_eq!(const_42.meet(&top), const_42);

        // Bottom meet X = Bottom
        assert_eq!(bottom.meet(&const_42), bottom);

        // Same constants meet to same constant
        assert_eq!(const_42.meet(&const_42.clone()), const_42);

        // Different constants meet to Bottom
        assert_eq!(const_42.meet(&const_10), bottom);
    }

    #[test]
    fn test_constant_folding_add() {
        let analysis = SCCPAnalysis::new();
        let left = LatticeValue::Constant(MirConstant::Int(10));
        let right = LatticeValue::Constant(MirConstant::Int(32));

        let result = analysis.evaluate_binary(MirBinaryOp::Add, &left, &right, &MirType::I64);

        assert_eq!(result, LatticeValue::Constant(MirConstant::Int(42)));
    }

    #[test]
    fn test_constant_folding_div_by_zero() {
        let analysis = SCCPAnalysis::new();
        let left = LatticeValue::Constant(MirConstant::Int(10));
        let right = LatticeValue::Constant(MirConstant::Int(0));

        let result = analysis.evaluate_binary(MirBinaryOp::Div, &left, &right, &MirType::I64);

        assert_eq!(result, LatticeValue::Bottom);
    }

    #[test]
    fn test_constant_folding_float() {
        let analysis = SCCPAnalysis::new();
        let left = LatticeValue::Constant(MirConstant::Float("3.0".to_string()));
        let right = LatticeValue::Constant(MirConstant::Float("4.0".to_string()));

        let result = analysis.evaluate_binary(MirBinaryOp::Mul, &left, &right, &MirType::F64);

        assert_eq!(result, LatticeValue::Constant(MirConstant::Float("12".to_string())));
    }

    #[test]
    fn test_constant_folding_comparison() {
        let analysis = SCCPAnalysis::new();
        let left = LatticeValue::Constant(MirConstant::Int(10));
        let right = LatticeValue::Constant(MirConstant::Int(20));

        let result = analysis.evaluate_compare(MirCompareOp::Lt, &left, &right, &MirType::Bool);

        assert_eq!(result, LatticeValue::Constant(MirConstant::Bool(true)));
    }

    #[test]
    fn test_constant_folding_unary() {
        let analysis = SCCPAnalysis::new();
        let operand = LatticeValue::Constant(MirConstant::Int(42));

        let result = analysis.evaluate_unary(MirUnaryOp::Neg, &operand, &MirType::I64);

        assert_eq!(result, LatticeValue::Constant(MirConstant::Int(-42)));
    }

    #[test]
    fn test_constant_folding_bool_not() {
        let analysis = SCCPAnalysis::new();
        let operand = LatticeValue::Constant(MirConstant::Bool(true));

        let result = analysis.evaluate_unary(MirUnaryOp::Not, &operand, &MirType::Bool);

        assert_eq!(result, LatticeValue::Constant(MirConstant::Bool(false)));
    }

    #[test]
    fn test_constant_propagation_simple() {
        // Create a simple function: x = 10; y = 32; z = x + y; return z
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test".to_string(),
            MirType::I64,
        );

        // x = 10
        let x = builder.fresh_value();
        builder.build_const(x, MirConstant::Int(10), MirType::I64);

        // y = 32
        let y = builder.fresh_value();
        builder.build_const(y, MirConstant::Int(32), MirType::I64);

        // z = x + y
        let z = builder.fresh_value();
        builder.build_add(z, x, y, MirType::I64);

        // return z
        builder.build_return(Some(z));

        let mut func = builder.build();

        // Run constant propagation
        let pass = ConstantPropagation;
        let modified = pass.run_on_function(&mut func).expect("pass should succeed");

        assert!(modified, "function should be modified");

        // Check that z is now a constant
        let entry = func.entry();
        let z_instr = &entry.instructions[2];

        match z_instr {
            MirInstruction::Const { value: MirConstant::Int(42), .. } => {
                // Success - z was folded to 42
            }
            _ => panic!("Expected z to be folded to constant 42, got {:?}", z_instr),
        }
    }

    #[test]
    fn test_constant_propagation_with_bottom() {
        // Function with parameter (Bottom): f(x) { return x + 10 }
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test".to_string(),
            MirType::I64,
        );

        builder.add_param("x".to_string(), MirType::I64);

        // const 10
        let ten = builder.fresh_value();
        builder.build_const(ten, MirConstant::Int(10), MirType::I64);

        // result = param + 10
        let param = ValueId(0); // First param
        let result = builder.fresh_value();
        builder.build_add(result, param, ten, MirType::I64);

        builder.build_return(Some(result));

        let mut func = builder.build();

        // Run constant propagation
        let pass = ConstantPropagation;
        let modified = pass.run_on_function(&mut func).expect("pass should succeed");

        // Should not modify because param is Bottom
        assert!(!modified, "function should not be modified when operand is Bottom");
    }
}
