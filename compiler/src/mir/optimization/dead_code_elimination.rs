//! Dead Code Elimination Optimization
//!
//! Based on "Advanced Compiler Design and Implementation" by Steven S. Muchnick (1997)
//! and "Compilers: Principles, Techniques, and Tools" by Aho et al. (2007)
//!
//! This implements both:
//! 1. Dead Assignment Elimination (DAE)
//! 2. Dead Basic Block Elimination (DBBE)

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::LazyLock;
use crate::mir::{
    MirModule, MirFunction, MirInstruction,
    ValueId, BlockId
};
use super::pass_manager::MIRPass;

static EMPTY_VALUE_SET: LazyLock<HashSet<ValueId>> = LazyLock::new(HashSet::new);

/// Dead Code Elimination pass
pub struct DeadCodeElimination;

/// Liveness analysis data
struct LivenessAnalysis {
    /// Live_in[block] = set of values live at entry to block
    live_in: HashMap<BlockId, HashSet<ValueId>>,
    /// Live_out[block] = set of values live at exit from block
    live_out: HashMap<BlockId, HashSet<ValueId>>,
    /// Use[block] = set of values used in block
    use_set: HashMap<BlockId, HashSet<ValueId>>,
    /// Def[block] = set of values defined in block
    def_set: HashMap<BlockId, HashSet<ValueId>>,
    /// Use map: value -> instructions that use it
    use_map: HashMap<ValueId, Vec<(BlockId, usize)>>,
}

impl LivenessAnalysis {
    fn new() -> Self {
        Self {
            live_in: HashMap::new(),
            live_out: HashMap::new(),
            use_set: HashMap::new(),
            def_set: HashMap::new(),
            use_map: HashMap::new(),
        }
    }

    /// Run liveness analysis on a function
    fn analyze(&mut self, func: &MirFunction) -> Result<(), String> {
        self.build_use_def_maps(func)?;
        self.compute_liveness(func)?;
        Ok(())
    }

    /// Build use and def maps for each block
    fn build_use_def_maps(&mut self, func: &MirFunction) -> Result<(), String> {
        self.use_map.clear();

        for block in &func.blocks {
            let mut use_set = HashSet::new();
            let mut def_set = HashSet::new();

            // Process instructions
            for (instr_idx, instr) in block.instructions.iter().enumerate() {
                // Collect uses
                for used_value in get_instruction_uses(instr) {
                    use_set.insert(used_value);
                    
                    // Build use map
                    self.use_map
                        .entry(used_value)
                        .or_insert_with(Vec::new)
                        .push((block.id, instr_idx));
                }

                // Collect definitions
                if let Some(result) = instr.result() {
                    def_set.insert(result);
                }
            }

            // Also capture uses in the terminator (return, conditional branch, etc.)
            for used_value in get_terminator_uses(&block.terminator) {
                use_set.insert(used_value);
                // Use usize::MAX as instruction index to indicate terminator use
                self.use_map
                    .entry(used_value)
                    .or_insert_with(Vec::new)
                    .push((block.id, usize::MAX));
            }

            self.use_set.insert(block.id, use_set);
            self.def_set.insert(block.id, def_set);
        }

        Ok(())
    }

    /// Compute liveness using iterative data flow analysis
    fn compute_liveness(&mut self, func: &MirFunction) -> Result<(), String> {
        // Initialize live_in and live_out sets
        for block in &func.blocks {
            self.live_in.insert(block.id, HashSet::new());
            self.live_out.insert(block.id, HashSet::new());
        }

        // Iterative algorithm
        let mut changed = true;
        let mut iterations = 0;

        while changed && iterations < 1000 { // Safety bound
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                let block_id = block.id;

                // live_out[block] = union of live_in[successors]
                let mut new_live_out = HashSet::new();
                let successors = get_block_successors(&block.terminator);
                for successor in successors {
                    if let Some(live_in_succ) = self.live_in.get(&successor) {
                        new_live_out.extend(live_in_succ.iter().copied());
                    }
                }

                // live_in[block] = use[block] ∪ (live_out[block] - def[block])
                let mut new_live_in = self.use_set.get(&block_id).cloned().unwrap_or_default();
                
                // Remove definitions
                if let Some(def_set) = self.def_set.get(&block_id) {
                    for defined_value in def_set {
                        new_live_out.remove(defined_value);
                    }
                }
                
                new_live_in.extend(new_live_out.iter().copied());

                // Update if changed
                if self.live_in.get(&block_id) != Some(&new_live_in) {
                    self.live_in.insert(block_id, new_live_in.clone());
                    changed = true;
                }
                if self.live_out.get(&block_id) != Some(&new_live_out) {
                    self.live_out.insert(block_id, new_live_out);
                    changed = true;
                }
            }
        }

        eprintln!("Liveness analysis completed in {} iterations", iterations);
        Ok(())
    }

    /// Check if a value is live at a specific point
    pub fn is_value_live(&self, block: BlockId, value: ValueId) -> bool {
        self.live_in.get(&block)
            .map(|live_in| live_in.contains(&value))
            .unwrap_or(false)
    }

    /// Get all live values at block entry
    pub fn get_live_in(&self, block: BlockId) -> &HashSet<ValueId> {
        self.live_in.get(&block).unwrap_or(&EMPTY_VALUE_SET)
    }

    /// Get all live values at block exit
    pub fn get_live_out(&self, block: BlockId) -> &HashSet<ValueId> {
        self.live_out.get(&block).unwrap_or(&EMPTY_VALUE_SET)
    }
}

/// Find dead code in a function
fn find_dead_code(func: &MirFunction, analysis: &LivenessAnalysis) -> (Vec<(usize, usize)>, HashSet<BlockId>) {
    let mut dead_instructions = Vec::new();
    let mut dead_blocks = HashSet::new();

    // Check for dead instructions and unreachable blocks
    for (block_idx, block) in func.blocks.iter().enumerate() {
        // Check if block is unreachable (entry block is always reachable)
        if block.id != func.entry_block {
            let is_reachable = func.blocks.iter()
                .any(|b| get_block_successors(&b.terminator).contains(&block.id));

            if !is_reachable {
                dead_blocks.insert(block.id);
                continue;
            }
        }

        // Check each instruction for dead assignments (including entry block)
        for (instr_idx, instr) in block.instructions.iter().enumerate() {
            if let Some(result) = instr.result() {
                // Check if result is live
                if !analysis.is_value_live(block.id, result) {
                    // Additional check: ensure no other instructions use this value
                    let uses = analysis.use_map.get(&result);
                    
                    // Count uses within this block
                    let mut uses_in_block = 0;
                    if let Some(uses_list) = uses {
                        for &(use_block, use_idx) in uses_list {
                            if use_block == block.id && use_idx > instr_idx {
                                uses_in_block += 1;
                            }
                        }
                    }

                    // If no uses and not live, it's dead code
                    if uses_in_block == 0 && !analysis.is_value_live(block.id, result) {
                        dead_instructions.push((block_idx, instr_idx));
                    }
                }
            }
        }
    }

    (dead_instructions, dead_blocks)
}

impl MIRPass for DeadCodeElimination {
    fn name(&self) -> &'static str {
        "dead-code-elimination"
    }

    fn run_on_module(&self, module: &mut MirModule) -> Result<bool, String> {
        let mut total_modified = false;

        for func in &mut module.functions {
            if self.run_on_function(func)? {
                total_modified = true;
            }
        }

        Ok(total_modified)
    }

    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        // Run liveness analysis
        let mut analysis = LivenessAnalysis::new();
        analysis.analyze(func)?;

        // Find dead code
        let (dead_instructions, dead_blocks) = find_dead_code(func, &analysis);

        let mut modified = false;

        // Remove dead blocks
        if !dead_blocks.is_empty() {
            let original_len = func.blocks.len();
            func.blocks.retain(|block| !dead_blocks.contains(&block.id));
            let removed_count = original_len - func.blocks.len();
            
            if removed_count > 0 {
                eprintln!("Removed {} dead blocks", removed_count);
                modified = true;
            }
        }

        // Remove dead instructions
        if !dead_instructions.is_empty() {
            // Sort by block index in reverse order to maintain correctness
            let mut dead_instructions = dead_instructions;
            dead_instructions.sort_by(|a, b| {
                if a.0 == b.0 {
                    b.1.cmp(&a.1) // Reverse order within same block
                } else {
                    b.0.cmp(&a.0) // Reverse order of blocks
                }
            });

            for (block_idx, instr_idx) in dead_instructions {
                if let Some(block) = func.blocks.get_mut(block_idx) {
                    if instr_idx < block.instructions.len() {
                        block.instructions.remove(instr_idx);
                        eprintln!("Removed dead instruction at block {}, index {}", block_idx, instr_idx);
                        modified = true;
                    }
                }
            }
        }

        // Simple dead basic block elimination
        if let Some(optimized) = eliminate_unreachable_blocks(func) {
            if optimized {
                modified = true;
            }
        }

        Ok(modified)
    }

    fn requires_ssa(&self) -> bool {
        true // DCE works best with SSA form
    }

    fn preserves_ssa(&self) -> bool {
        true // DCE preserves SSA form
    }
}

/// Eliminate unreachable blocks using a simple reachability analysis
fn eliminate_unreachable_blocks(func: &mut MirFunction) -> Option<bool> {
    // Find reachable blocks starting from entry
    let mut reachable = HashSet::new();
    let mut worklist = VecDeque::new();
    
    worklist.push_back(func.entry_block);
    reachable.insert(func.entry_block);

    while let Some(current) = worklist.pop_front() {
        if let Some(block) = func.blocks.iter().find(|b| b.id == current) {
            for successor in get_block_successors(&block.terminator) {
                if !reachable.contains(&successor) {
                    reachable.insert(successor);
                    worklist.push_back(successor);
                }
            }
        }
    }

    // Remove unreachable blocks
    let original_len = func.blocks.len();
    func.blocks.retain(|block| reachable.contains(&block.id));
    let removed_count = original_len - func.blocks.len();

    if removed_count > 0 {
        eprintln!("Removed {} unreachable blocks", removed_count);
        Some(true)
    } else {
        Some(false)
    }
}

/// Get all values used by an instruction
fn get_instruction_uses(instr: &MirInstruction) -> Vec<ValueId> {
    match instr {
        MirInstruction::Load { address, .. } => vec![*address],
        MirInstruction::Store { address, value, .. } => vec![*address, *value],
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

/// Get the successor block IDs from a terminator
fn get_block_successors(terminator: &crate::mir::MirTerminator) -> Vec<BlockId> {
    use crate::mir::MirTerminator;
    match terminator {
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

/// Get all values used by a terminator
fn get_terminator_uses(terminator: &crate::mir::MirTerminator) -> Vec<ValueId> {
    use crate::mir::MirTerminator;
    match terminator {
        MirTerminator::Branch { .. } => vec![],
        MirTerminator::CondBranch { condition, .. } => vec![*condition],
        MirTerminator::Switch { value, .. } => vec![*value],
        MirTerminator::Return { value } => value.iter().copied().collect(),
        MirTerminator::Unreachable => vec![],
        MirTerminator::CallNoReturn { args, .. } => args.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::{FuncId, MirType, MirConstant};

    #[test]
    fn test_dead_assignment_elimination() {
        // Create a function with dead assignment: x = 10; y = 20; z = x; (y is unused)
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test".to_string(),
            MirType::I64,
        );

        // x = 10 (used)
        let x = builder.fresh_value();
        builder.build_const(x, MirConstant::Int(10), MirType::I64);

        // y = 20 (unused - should be eliminated)
        let y = builder.fresh_value();
        builder.build_const(y, MirConstant::Int(20), MirType::I64);

        // z = x (used)
        let z = builder.fresh_value();
        builder.build_add(z, x, x, MirType::I64);

        // return z
        builder.build_return(Some(z));

        let mut func = builder.build();

        // Run DCE
        let pass = DeadCodeElimination;
        let modified = pass.run_on_function(&mut func).expect("pass should succeed");

        assert!(modified, "function should be modified by DCE");

        // Should have removed the y = 20 assignment
        // Remaining: x=10, z=x+x (return is a terminator, not instruction)
        assert_eq!(func.blocks[0].instructions.len(), 2);
    }

    #[test]
    fn test_unreachable_block_elimination() {
        // This test would need a function with unreachable blocks
        // For now, just test that the pass runs without error
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test".to_string(),
            MirType::I64,
        );

        let x = builder.fresh_value();
        builder.build_const(x, MirConstant::Int(42), MirType::I64);
        builder.build_return(Some(x));

        let mut func = builder.build();

        let pass = DeadCodeElimination;
        let _modified = pass.run_on_function(&mut func).expect("pass should succeed");
    }

    #[test]
    fn test_liveness_analysis() {
        // Create a simple function: x = 10; y = x + 5; return y
        let mut builder = FunctionBuilder::new(
            FuncId(0),
            "test".to_string(),
            MirType::I64,
        );

        let x = builder.fresh_value();
        builder.build_const(x, MirConstant::Int(10), MirType::I64);

        let five = builder.fresh_value();
        builder.build_const(five, MirConstant::Int(5), MirType::I64);

        let y = builder.fresh_value();
        builder.build_add(y, x, five, MirType::I64);

        builder.build_return(Some(y));

        let func = builder.build();

        // Run liveness analysis
        let mut analysis = LivenessAnalysis::new();
        analysis.analyze(&func).expect("analysis should succeed");

        // x should be live at its definition and use
        // This is a simplified test
        let entry_block = func.blocks[0].id;
        let live_in = analysis.get_live_in(entry_block);
        
        // The exact values depend on parameter handling
        println!("Live in set size: {}", live_in.len());
        assert!(!live_in.is_empty()); // Should have some live values
    }
}
