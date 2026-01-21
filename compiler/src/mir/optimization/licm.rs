//! Loop Invariant Code Motion (LICM)
//!
//! This module implements LICM for MIR, based on:
//! - Muchnick (1997) "Advanced Compiler Design and Implementation", Chapter 13
//! - Wolfe (1996) "High-Performance Compilers for Parallel Computing"
//! - Cytron et al. (1991) "Efficiently Computing Static Single Assignment Form"
//!
//! LICM hoists loop-invariant computations out of loops to reduce redundant work.
//! An expression is loop-invariant if:
//! 1. All its operands are defined outside the loop, OR
//! 2. All its operands are themselves loop-invariant
//!
//! The algorithm:
//! 1. Detect natural loops via back edges in the CFG
//! 2. For each loop, identify invariant instructions
//! 3. Create/use a preheader block and move invariant code there

use std::collections::{HashMap, HashSet, VecDeque};
use crate::mir::{
    MirModule, MirFunction, MirBlock, MirInstruction, MirTerminator,
    ValueId, BlockId,
};
use super::pass_manager::MIRPass;

/// Information about a natural loop
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Header block (target of back edge)
    pub header: BlockId,
    /// All blocks in the loop body (including header)
    pub blocks: HashSet<BlockId>,
    /// Back edges: (source, target) where target == header
    pub back_edges: Vec<(BlockId, BlockId)>,
    /// Preheader block ID (if one exists or was created)
    pub preheader: Option<BlockId>,
    /// Exit blocks (blocks in the loop with successors outside)
    pub exits: HashSet<BlockId>,
}

impl LoopInfo {
    /// Create a new LoopInfo for a detected loop
    fn new(header: BlockId) -> Self {
        Self {
            header,
            blocks: HashSet::from([header]),
            back_edges: Vec::new(),
            preheader: None,
            exits: HashSet::new(),
        }
    }

    /// Check if a block is in this loop
    pub fn contains(&self, block: BlockId) -> bool {
        self.blocks.contains(&block)
    }
}

/// Loop detection and analysis
struct LoopAnalysis {
    /// Detected loops, keyed by header block
    loops: HashMap<BlockId, LoopInfo>,
    /// Predecessors for each block (computed from terminators)
    predecessors: HashMap<BlockId, Vec<BlockId>>,
    /// Successors for each block (computed from terminators)
    successors: HashMap<BlockId, Vec<BlockId>>,
    /// Dominators for each block
    dominators: HashMap<BlockId, HashSet<BlockId>>,
}

impl LoopAnalysis {
    fn new() -> Self {
        Self {
            loops: HashMap::new(),
            predecessors: HashMap::new(),
            successors: HashMap::new(),
            dominators: HashMap::new(),
        }
    }

    /// Build CFG predecessor/successor maps from terminators
    fn build_cfg(&mut self, func: &MirFunction) {
        self.predecessors.clear();
        self.successors.clear();

        // Initialize empty sets for all blocks
        for block in &func.blocks {
            self.predecessors.insert(block.id, Vec::new());
            self.successors.insert(block.id, Vec::new());
        }

        // Build from terminators
        for block in &func.blocks {
            let succs = get_terminator_successors(&block.terminator);
            self.successors.insert(block.id, succs.clone());

            for succ in succs {
                self.predecessors
                    .entry(succ)
                    .or_default()
                    .push(block.id);
            }
        }
    }

    /// Compute dominators using iterative algorithm (Cytron et al.)
    fn compute_dominators(&mut self, func: &MirFunction) {
        if func.blocks.is_empty() {
            return;
        }

        self.dominators.clear();
        let entry = func.entry_block;
        let all_blocks: HashSet<BlockId> = func.blocks.iter().map(|b| b.id).collect();

        // Initialize: Dom(entry) = {entry}, Dom(n) = all blocks for n != entry
        for block in &func.blocks {
            if block.id == entry {
                self.dominators.insert(block.id, HashSet::from([entry]));
            } else {
                self.dominators.insert(block.id, all_blocks.clone());
            }
        }

        // Iterate until fixpoint
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < 1000 {
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                if block.id == entry {
                    continue;
                }

                // Dom(n) = {n} ∪ (∩ Dom(p) for all predecessors p of n)
                let preds = self.predecessors.get(&block.id).cloned().unwrap_or_default();
                let new_dom = if preds.is_empty() {
                    HashSet::from([block.id])
                } else {
                    let mut intersection: Option<HashSet<BlockId>> = None;
                    for pred in &preds {
                        if let Some(pred_dom) = self.dominators.get(pred) {
                            intersection = match intersection {
                                None => Some(pred_dom.clone()),
                                Some(current) => {
                                    Some(current.intersection(pred_dom).cloned().collect())
                                }
                            };
                        }
                    }
                    let mut result = intersection.unwrap_or_default();
                    result.insert(block.id);
                    result
                };

                if self.dominators.get(&block.id) != Some(&new_dom) {
                    self.dominators.insert(block.id, new_dom);
                    changed = true;
                }
            }
        }
    }

    /// Check if block A dominates block B
    fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        self.dominators
            .get(&b)
            .map(|doms| doms.contains(&a))
            .unwrap_or(false)
    }

    /// Detect natural loops via back edges
    /// A back edge is an edge (n, h) where h dominates n
    fn detect_loops(&mut self, func: &MirFunction) {
        self.loops.clear();

        // Find all back edges
        for block in &func.blocks {
            let succs = self.successors.get(&block.id).cloned().unwrap_or_default();
            for succ in succs {
                // Back edge: succ dominates block
                if self.dominates(succ, block.id) {
                    // succ is the loop header
                    let loop_info = self.loops.entry(succ).or_insert_with(|| LoopInfo::new(succ));
                    loop_info.back_edges.push((block.id, succ));

                    // Compute loop body using reverse DFS from back edge source
                    self.compute_loop_body(func, block.id, succ);
                }
            }
        }

        // Compute exit blocks for each loop
        for loop_info in self.loops.values_mut() {
            for &block_id in &loop_info.blocks.clone() {
                let succs = self.successors.get(&block_id).cloned().unwrap_or_default();
                for succ in succs {
                    if !loop_info.blocks.contains(&succ) {
                        loop_info.exits.insert(block_id);
                        break;
                    }
                }
            }
        }
    }

    /// Compute loop body by walking backwards from tail to header
    fn compute_loop_body(&mut self, _func: &MirFunction, tail: BlockId, header: BlockId) {
        let loop_info = match self.loops.get_mut(&header) {
            Some(info) => info,
            None => return,
        };

        // BFS backwards from tail, stopping at header
        let mut worklist = VecDeque::new();
        worklist.push_back(tail);

        while let Some(block) = worklist.pop_front() {
            if loop_info.blocks.contains(&block) {
                continue; // Already processed
            }

            loop_info.blocks.insert(block);

            // Add predecessors (except if we've reached header)
            if block != header {
                let preds = self.predecessors.get(&block).cloned().unwrap_or_default();
                for pred in preds {
                    if !loop_info.blocks.contains(&pred) {
                        worklist.push_back(pred);
                    }
                }
            }
        }
    }

    /// Analyze function for loops
    fn analyze(&mut self, func: &MirFunction) {
        self.build_cfg(func);
        self.compute_dominators(func);
        self.detect_loops(func);
    }
}

/// Loop Invariant Code Motion optimization pass
pub struct LoopInvariantCodeMotion;

/// State for LICM transformation
struct LICMState {
    /// Values defined inside the current loop
    loop_defs: HashSet<ValueId>,
    /// Values that are loop-invariant
    invariant_values: HashSet<ValueId>,
    /// Instructions to hoist (block_id, instruction_index)
    to_hoist: Vec<(BlockId, usize, MirInstruction)>,
    /// Next available block ID
    next_block_id: usize,
    /// Next available value ID
    next_value_id: usize,
}

impl LICMState {
    fn new(func: &MirFunction) -> Self {
        // Find max block and value IDs
        let mut max_block_id = 0;
        let mut max_value_id = 0;

        for block in &func.blocks {
            if block.id.0 >= max_block_id {
                max_block_id = block.id.0 + 1;
            }
            for instr in &block.instructions {
                if let Some(result) = get_instruction_result(instr) {
                    if result.0 >= max_value_id {
                        max_value_id = result.0 + 1;
                    }
                }
            }
        }

        Self {
            loop_defs: HashSet::new(),
            invariant_values: HashSet::new(),
            to_hoist: Vec::new(),
            next_block_id: max_block_id,
            next_value_id: max_value_id,
        }
    }

    fn fresh_block_id(&mut self) -> BlockId {
        let id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Collect all values defined in the loop
    fn collect_loop_definitions(&mut self, func: &MirFunction, loop_info: &LoopInfo) {
        self.loop_defs.clear();

        for block in &func.blocks {
            if !loop_info.contains(block.id) {
                continue;
            }

            for instr in &block.instructions {
                if let Some(result) = get_instruction_result(instr) {
                    self.loop_defs.insert(result);
                }
            }
        }
    }

    /// Check if a value is loop-invariant
    fn is_invariant(&self, value: ValueId) -> bool {
        // Value is invariant if:
        // 1. It's defined outside the loop, OR
        // 2. It's already been marked as invariant
        !self.loop_defs.contains(&value) || self.invariant_values.contains(&value)
    }

    /// Check if an instruction is loop-invariant
    fn is_instruction_invariant(&self, instr: &MirInstruction) -> bool {
        // Instructions with side effects cannot be hoisted
        if has_side_effects(instr) {
            return false;
        }

        // Check all operands are invariant
        let uses = get_instruction_uses(instr);
        uses.iter().all(|&v| self.is_invariant(v))
    }

    /// Find invariant instructions in a loop (iterative fixpoint)
    fn find_invariant_instructions(&mut self, func: &MirFunction, loop_info: &LoopInfo) {
        self.invariant_values.clear();
        self.to_hoist.clear();

        // Iterate until fixpoint
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < 100 {
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                if !loop_info.contains(block.id) {
                    continue;
                }

                for (idx, instr) in block.instructions.iter().enumerate() {
                    // Skip if already marked as invariant
                    if let Some(result) = get_instruction_result(instr) {
                        if self.invariant_values.contains(&result) {
                            continue;
                        }

                        // Check if instruction is invariant
                        if self.is_instruction_invariant(instr) {
                            self.invariant_values.insert(result);
                            self.to_hoist.push((block.id, idx, instr.clone()));
                            changed = true;
                        }
                    }
                }
            }
        }
    }
}

impl MIRPass for LoopInvariantCodeMotion {
    fn name(&self) -> &'static str {
        "loop-invariant-code-motion"
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
        // Analyze loops
        let mut loop_analysis = LoopAnalysis::new();
        loop_analysis.analyze(func);

        if loop_analysis.loops.is_empty() {
            return Ok(false);
        }

        let mut modified = false;
        let mut state = LICMState::new(func);

        // Process each loop (process inner loops first for better results)
        let loop_headers: Vec<BlockId> = loop_analysis.loops.keys().cloned().collect();

        for header in loop_headers {
            let loop_info = match loop_analysis.loops.get(&header) {
                Some(info) => info.clone(),
                None => continue,
            };

            // Collect definitions and find invariants
            state.collect_loop_definitions(func, &loop_info);
            state.find_invariant_instructions(func, &loop_info);

            if state.to_hoist.is_empty() {
                continue;
            }

            // Get or create preheader
            let preheader_id = self.get_or_create_preheader(
                func,
                &loop_info,
                &loop_analysis,
                &mut state,
            )?;

            // Hoist instructions to preheader
            let hoisted = self.hoist_instructions(func, preheader_id, &state)?;
            if hoisted > 0 {
                modified = true;
                eprintln!(
                    "LICM: Hoisted {} instructions from loop at {}",
                    hoisted, header.0
                );
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

impl LoopInvariantCodeMotion {
    /// Get existing preheader or create one
    fn get_or_create_preheader(
        &self,
        func: &mut MirFunction,
        loop_info: &LoopInfo,
        loop_analysis: &LoopAnalysis,
        state: &mut LICMState,
    ) -> Result<BlockId, String> {
        let header = loop_info.header;

        // Check for existing preheader:
        // A preheader is a block that:
        // 1. Has the header as its only successor
        // 2. Is the only predecessor of the header from outside the loop
        let preds = loop_analysis.predecessors.get(&header).cloned().unwrap_or_default();
        let outside_preds: Vec<BlockId> = preds
            .iter()
            .filter(|&&p| !loop_info.contains(p))
            .cloned()
            .collect();

        // If there's exactly one outside predecessor, check if it's a valid preheader
        if outside_preds.len() == 1 {
            let candidate = outside_preds[0];
            let candidate_succs = loop_analysis.successors.get(&candidate).cloned().unwrap_or_default();
            if candidate_succs.len() == 1 && candidate_succs[0] == header {
                return Ok(candidate);
            }
        }

        // Need to create a new preheader
        let preheader_id = state.fresh_block_id();

        // Create preheader block
        let preheader = MirBlock {
            id: preheader_id,
            label: format!("preheader_{}", header.0),
            instructions: Vec::new(),
            terminator: MirTerminator::Branch { target: header },
        };

        // Insert preheader before header
        let header_idx = func.blocks.iter().position(|b| b.id == header);
        match header_idx {
            Some(idx) => func.blocks.insert(idx, preheader),
            None => func.blocks.push(preheader),
        }

        // Update all outside predecessors to jump to preheader instead of header
        for &pred_id in &outside_preds {
            if let Some(pred_block) = func.get_block_mut(pred_id) {
                redirect_terminator(&mut pred_block.terminator, header, preheader_id);
            }
        }

        // Update phi nodes in header to reference preheader
        if let Some(header_block) = func.get_block_mut(header) {
            for instr in &mut header_block.instructions {
                if let MirInstruction::Phi { incoming, .. } = instr {
                    for (pred, _) in incoming.iter_mut() {
                        if outside_preds.contains(pred) {
                            *pred = preheader_id;
                        }
                    }
                }
            }
        }

        Ok(preheader_id)
    }

    /// Hoist invariant instructions to preheader
    fn hoist_instructions(
        &self,
        func: &mut MirFunction,
        preheader_id: BlockId,
        state: &LICMState,
    ) -> Result<usize, String> {
        // Group instructions to hoist by their block, sorted by index (descending for safe removal)
        let mut hoisted_count = 0;

        // First, collect instructions to add to preheader (in order)
        let mut preheader_additions: Vec<MirInstruction> = Vec::new();
        for (_, _, instr) in &state.to_hoist {
            preheader_additions.push(instr.clone());
        }

        // Add to preheader (before its terminator)
        if let Some(preheader) = func.get_block_mut(preheader_id) {
            for instr in preheader_additions {
                preheader.instructions.push(instr);
                hoisted_count += 1;
            }
        }

        // Remove from original locations (in reverse order to maintain indices)
        let mut removals: Vec<(BlockId, usize)> = state
            .to_hoist
            .iter()
            .map(|(block, idx, _)| (*block, *idx))
            .collect();

        // Sort by block, then by index descending
        removals.sort_by(|a, b| {
            if a.0 == b.0 {
                b.1.cmp(&a.1) // Descending index within same block
            } else {
                a.0 .0.cmp(&b.0 .0)
            }
        });

        // Remove duplicates and perform removals
        let mut processed: HashSet<(BlockId, usize)> = HashSet::new();
        for (block_id, idx) in removals {
            if processed.contains(&(block_id, idx)) {
                continue;
            }
            processed.insert((block_id, idx));

            if let Some(block) = func.get_block_mut(block_id) {
                if idx < block.instructions.len() {
                    block.instructions.remove(idx);
                }
            }
        }

        Ok(hoisted_count)
    }
}

/// Get successors from a terminator
fn get_terminator_successors(term: &MirTerminator) -> Vec<BlockId> {
    match term {
        MirTerminator::Branch { target } => vec![*target],
        MirTerminator::CondBranch { true_target, false_target, .. } => {
            vec![*true_target, *false_target]
        }
        MirTerminator::Switch { default_target, cases, .. } => {
            let mut succs = vec![*default_target];
            succs.extend(cases.iter().map(|(_, t)| *t));
            succs
        }
        MirTerminator::Return { .. }
        | MirTerminator::Unreachable
        | MirTerminator::CallNoReturn { .. } => vec![],
    }
}

/// Get the result value of an instruction (if any)
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
        MirInstruction::Call { result, .. }
        | MirInstruction::CallIndirect { result, .. } => *result,
        MirInstruction::Store { .. } | MirInstruction::StoreGlobal { .. } => None,
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

/// Check if an instruction has side effects (cannot be hoisted)
fn has_side_effects(instr: &MirInstruction) -> bool {
    match instr {
        // Memory operations have side effects
        MirInstruction::Store { .. }
        | MirInstruction::StoreGlobal { .. }
        | MirInstruction::Load { .. }
        | MirInstruction::LoadGlobal { .. } => true,

        // Calls may have side effects
        MirInstruction::Call { .. }
        | MirInstruction::CallIndirect { .. } => true,

        // Allocation has side effects
        MirInstruction::Alloca { .. } => true,

        // Phi nodes must stay in their block
        MirInstruction::Phi { .. } => true,

        // Pure computations can be hoisted
        MirInstruction::GetElementPtr { .. }
        | MirInstruction::Cast { .. }
        | MirInstruction::ZExt { .. }
        | MirInstruction::SExt { .. }
        | MirInstruction::Trunc { .. }
        | MirInstruction::FPToSI { .. }
        | MirInstruction::SIToFP { .. }
        | MirInstruction::FPToUI { .. }
        | MirInstruction::UIToFP { .. }
        | MirInstruction::PtrCast { .. }
        | MirInstruction::Binary { .. }
        | MirInstruction::Unary { .. }
        | MirInstruction::Compare { .. }
        | MirInstruction::Const { .. }
        | MirInstruction::Select { .. } => false,
    }
}

/// Redirect a terminator from one target to another
fn redirect_terminator(term: &mut MirTerminator, from: BlockId, to: BlockId) {
    match term {
        MirTerminator::Branch { target } => {
            if *target == from {
                *target = to;
            }
        }
        MirTerminator::CondBranch { true_target, false_target, .. } => {
            if *true_target == from {
                *true_target = to;
            }
            if *false_target == from {
                *false_target = to;
            }
        }
        MirTerminator::Switch { default_target, cases, .. } => {
            if *default_target == from {
                *default_target = to;
            }
            for (_, target) in cases {
                if *target == from {
                    *target = to;
                }
            }
        }
        MirTerminator::Return { .. }
        | MirTerminator::Unreachable
        | MirTerminator::CallNoReturn { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::{FunctionBuilder, ModuleBuilder};
    use crate::mir::{FuncId, MirConstant, MirType};

    #[test]
    fn test_loop_detection_simple() {
        // Create: entry -> loop_header -> loop_body -> loop_header (back edge)
        //                              -> exit
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test_loop".to_string(),
            MirType::I32,
        );

        // Entry block
        let loop_header = builder.create_block("loop_header");
        builder.build_branch(loop_header);

        // Loop header
        builder.switch_to_block(loop_header);
        let cond = builder.fresh_value();
        builder.build_const(cond, MirConstant::Bool(true), MirType::Bool);
        let loop_body = builder.create_block("loop_body");
        let exit = builder.create_block("exit");
        builder.build_cond_branch(cond, loop_body, exit);

        // Loop body - branches back to header
        builder.switch_to_block(loop_body);
        builder.build_branch(loop_header);

        // Exit
        builder.switch_to_block(exit);
        let result = builder.build_i32(0);
        builder.build_return(Some(result));

        let func = builder.build();

        // Run loop analysis
        let mut analysis = LoopAnalysis::new();
        analysis.analyze(&func);

        // Should detect one loop with header at loop_header
        assert_eq!(analysis.loops.len(), 1);
        assert!(analysis.loops.contains_key(&loop_header));

        let loop_info = analysis.loops.get(&loop_header).unwrap();
        assert!(loop_info.contains(loop_header));
        assert!(loop_info.contains(loop_body));
        assert!(!loop_info.contains(exit));
    }

    #[test]
    fn test_invariant_detection() {
        // Create a loop with an invariant computation:
        // entry: x = 10
        // loop_header: y = x + 5  (invariant - x is defined outside)
        //              z = y * i  (not invariant - i changes)
        //              i = i + 1
        //              if i < 10 goto loop_header else exit
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test_invariant".to_string(),
            MirType::I32,
        );

        // Entry: x = 10
        let x = builder.fresh_value();
        builder.build_const(x, MirConstant::Int(10), MirType::I64);
        let five = builder.fresh_value();
        builder.build_const(five, MirConstant::Int(5), MirType::I64);

        let loop_header = builder.create_block("loop_header");
        builder.build_branch(loop_header);

        // Loop header
        builder.switch_to_block(loop_header);

        // y = x + 5 (should be invariant)
        let y = builder.fresh_value();
        builder.build_add(y, x, five, MirType::I64);

        let exit = builder.create_block("exit");

        // Simplified: just branch back
        builder.build_branch(loop_header);

        // Exit
        builder.switch_to_block(exit);
        builder.build_return(Some(y));

        let func = builder.build();

        // Analyze
        let mut loop_analysis = LoopAnalysis::new();
        loop_analysis.analyze(&func);

        // Should have one loop
        assert!(!loop_analysis.loops.is_empty());

        let loop_info = loop_analysis.loops.values().next().unwrap();

        // Create LICM state and find invariants
        let mut state = LICMState::new(&func);
        state.collect_loop_definitions(&func, loop_info);

        // x and five should NOT be in loop_defs (defined outside)
        assert!(!state.loop_defs.contains(&x));
        assert!(!state.loop_defs.contains(&five));

        // y should be in loop_defs (defined inside)
        assert!(state.loop_defs.contains(&y));
    }

    #[test]
    fn test_licm_pass_runs() {
        // Smoke test: the pass should run without errors
        let mut module_builder = ModuleBuilder::new("test_module");

        let mut func_builder = module_builder.create_function(
            "simple".to_string(),
            MirType::I32,
        );

        let result = func_builder.build_i32(42);
        func_builder.build_return(Some(result));

        module_builder.add_function(func_builder.build());
        let mut module = module_builder.build();

        let pass = LoopInvariantCodeMotion;
        let modified = pass.run_on_module(&mut module).expect("pass should succeed");

        // No loops, so should not be modified
        assert!(!modified);
    }

    #[test]
    fn test_dominator_computation() {
        // Test dominance: entry dominates all blocks
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test_dom".to_string(),
            MirType::Unit,
        );

        let block_a = builder.create_block("block_a");
        let block_b = builder.create_block("block_b");

        builder.build_branch(block_a);

        builder.switch_to_block(block_a);
        builder.build_branch(block_b);

        builder.switch_to_block(block_b);
        builder.build_return(None);

        let func = builder.build();
        let entry = func.entry_block;

        let mut analysis = LoopAnalysis::new();
        analysis.build_cfg(&func);
        analysis.compute_dominators(&func);

        // Entry dominates all blocks
        assert!(analysis.dominates(entry, entry));
        assert!(analysis.dominates(entry, block_a));
        assert!(analysis.dominates(entry, block_b));

        // block_a dominates block_b
        assert!(analysis.dominates(block_a, block_b));

        // block_b does not dominate block_a
        assert!(!analysis.dominates(block_b, block_a));
    }
}
