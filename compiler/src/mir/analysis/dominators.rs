//! SSA Dominance Analysis
//!
//! Based on "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph"
//! by Cytron et al. (1991)

use crate::mir::{MirFunction, BlockId};
use std::collections::{HashMap, HashSet, BTreeSet};
use std::sync::LazyLock;

static EMPTY_BLOCK_SET: LazyLock<HashSet<BlockId>> = LazyLock::new(HashSet::new);

/// Dominator analysis for SSA validation
pub struct DominatorAnalysis {
    /// Immediate dominators for each block
    idom: HashMap<BlockId, BlockId>,
    /// Dominance frontier for each block
    df: HashMap<BlockId, HashSet<BlockId>>,
    /// Dominator tree children
    dom_tree: HashMap<BlockId, HashSet<BlockId>>,
}

impl DominatorAnalysis {
    pub fn new() -> Self {
        Self {
            idom: HashMap::new(),
            df: HashMap::new(),
            dom_tree: HashMap::new(),
        }
    }

    /// Compute dominators for a function
    /// Based on Cytron et al. algorithm
    pub fn compute_dominators(&mut self, func: &MirFunction) -> Result<(), String> {
        if func.blocks.is_empty() {
            return Ok(());
        }

        let blocks: Vec<BlockId> = func.blocks.iter().map(|b| b.id).collect();
        let entry_block = func.blocks.first().ok_or("Function has no entry block")?.id;
        
        // Initialize dominator sets
        let mut dominators: HashMap<BlockId, BTreeSet<BlockId>> = HashMap::new();
        
        // Entry block dominates itself
        for &block in &blocks {
            if block == entry_block {
                dominators.insert(block, BTreeSet::from([block]));
            } else {
                dominators.insert(block, BTreeSet::from_iter(blocks.iter().cloned()));
            }
        }

        // Iterative algorithm (Cytron et al.)
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < 1000 { // Safety bound
            changed = false;
            iterations += 1;
            
            for &block in &blocks {
                if block == entry_block {
                    continue;
                }

                // New dominators = intersection of all predecessors' dominators ∪ {block}
                let mut new_dom: BTreeSet<BlockId> = BTreeSet::from([block]);
                
                let predecessors = self.get_predecessors(func, block);
                
                if !predecessors.is_empty() {
                    let mut intersection: Option<BTreeSet<BlockId>> = None;

                    for pred in predecessors {
                        if let Some(pred_dom) = dominators.get(&pred) {
                            intersection = match intersection {
                                None => Some(pred_dom.clone()),
                                Some(current) => {
                                    Some(current.intersection(pred_dom).cloned().collect())
                                }
                            };
                        }
                    }

                    if let Some(inter) = intersection {
                        new_dom.extend(inter.iter().cloned());
                    }
                }

                if dominators.get(&block) != Some(&new_dom) {
                    dominators.insert(block, new_dom);
                    changed = true;
                }
            }
        }

        // Compute immediate dominators
        self.compute_immediate_dominators(&dominators, entry_block);
        
        // Compute dominance frontier
        self.compute_dominance_frontier(func);
        
        // Build dominator tree
        self.build_dominator_tree(entry_block);

        Ok(())
    }

    /// Compute immediate dominators from dominator sets
    fn compute_immediate_dominators(&mut self, dominators: &HashMap<BlockId, BTreeSet<BlockId>>, entry: BlockId) {
        self.idom.clear();
        
        for (block, dom_set) in dominators {
            if *block == entry {
                continue;
            }

            // Find the closest dominator (excluding the block itself)
            let mut immediate = None;
            let mut max_distance = 0;

            for candidate in dom_set {
                if *candidate == *block {
                    continue;
                }

                // Check if candidate dominates all other candidates
                let mut dominates_all = true;
                let candidate_dom = dominators.get(candidate).unwrap();

                for other_candidate in dom_set {
                    if *other_candidate == *block || *other_candidate == *candidate {
                        continue;
                    }
                    if !candidate_dom.contains(other_candidate) {
                        dominates_all = false;
                        break;
                    }
                }

                if dominates_all {
                    // Find distance (depth in dominator tree)
                    let distance = self.compute_dominance_depth(dominators, *candidate, entry);
                    if immediate.is_none() || distance > max_distance {
                        immediate = Some(*candidate);
                        max_distance = distance;
                    }
                }
            }

            if let Some(idom) = immediate {
                self.idom.insert(*block, idom);
            }
        }
    }

    /// Compute dominance frontier
    /// Based on Cytron et al. algorithm
    fn compute_dominance_frontier(&mut self, func: &MirFunction) {
        self.df.clear();

        for block in &func.blocks {
            self.df.insert(block.id, HashSet::new());
        }

        // Compute predecessors for each block
        let predecessors = self.compute_predecessors(func);

        for block in &func.blocks {
            let preds = predecessors.get(&block.id).cloned().unwrap_or_default();
            if preds.len() <= 1 {
                continue;
            }

            for pred_id in preds {
                let mut runner = pred_id;

                // Find all nodes where pred doesn't dominate
                while runner != block.id {
                    if let Some(runner_idom) = self.idom.get(&runner) {
                        if *runner_idom == block.id {
                            break;
                        }
                        self.df.get_mut(&runner).unwrap().insert(block.id);
                        runner = *runner_idom;
                    } else {
                        break;
                    }
                }
            }
        }
    }

    /// Build dominator tree
    fn build_dominator_tree(&mut self, entry: BlockId) {
        self.dom_tree.clear();

        // Initialize all blocks as children of entry if no idom found
        for (block, _) in &self.idom {
            self.dom_tree.entry(*block).or_insert_with(HashSet::new);
        }

        for (block, idom) in &self.idom {
            self.dom_tree.entry(*idom).or_insert_with(HashSet::new).insert(*block);
        }

        // Ensure entry has an entry in the tree
        self.dom_tree.entry(entry).or_insert_with(HashSet::new);
    }

    /// Compute predecessors map from terminators
    fn compute_predecessors(&self, func: &MirFunction) -> HashMap<BlockId, Vec<BlockId>> {
        use crate::mir::MirTerminator;
        let mut predecessors: HashMap<BlockId, Vec<BlockId>> = HashMap::new();

        // Initialize empty predecessor lists for all blocks
        for block in &func.blocks {
            predecessors.insert(block.id, Vec::new());
        }

        // For each block, add it as a predecessor of its successors
        for block in &func.blocks {
            let successors = match &block.terminator {
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
            };

            for succ in successors {
                predecessors.entry(succ).or_default().push(block.id);
            }
        }

        predecessors
    }

    /// Helper: Get predecessors of a block
    fn get_predecessors(&self, func: &MirFunction, block: BlockId) -> Vec<BlockId> {
        self.compute_predecessors(func).get(&block).cloned().unwrap_or_default()
    }

    /// Helper: Compute dominance depth
    fn compute_dominance_depth(&self, dominators: &HashMap<BlockId, BTreeSet<BlockId>>, block: BlockId, entry: BlockId) -> u32 {
        let mut depth = 0;
        let mut current = block;
        
        while current != entry {
            if let Some(idom) = self.idom.get(&current) {
                current = *idom;
                depth += 1;
            } else {
                break;
            }
        }
        
        depth
    }

    /// Check if block1 dominates block2
    pub fn dominates(&self, block1: BlockId, block2: BlockId, dominators: &HashMap<BlockId, BTreeSet<BlockId>>) -> bool {
        dominators.get(&block2)
            .map(|dom_set| dom_set.contains(&block1))
            .unwrap_or(false)
    }

    /// Get immediate dominator of a block
    pub fn immediate_dominator(&self, block: BlockId) -> Option<BlockId> {
        self.idom.get(&block).copied()
    }

    /// Get dominance frontier of a block
    pub fn dominance_frontier(&self, block: BlockId) -> &HashSet<BlockId> {
        self.df.get(&block).unwrap_or(&EMPTY_BLOCK_SET)
    }

    /// Get dominator tree children
    pub fn dom_tree_children(&self, block: BlockId) -> &HashSet<BlockId> {
        self.dom_tree.get(&block).unwrap_or(&EMPTY_BLOCK_SET)
    }
}

impl Default for DominatorAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::{FuncId, MirType};

    #[test]
    fn test_dominator_analysis() {
        // Create a simple function with control flow
        let mut func_builder = FunctionBuilder::new(
            FuncId(0),
            "test_func".to_string(),
            MirType::Unit,
        );

        let then_block = func_builder.create_block("then");
        let else_block = func_builder.create_block("else");
        let merge_block = func_builder.create_block("merge");

        // Entry -> then_block -> merge_block
        // Entry -> else_block -> merge_block
        func_builder.build_branch(then_block);

        func_builder.switch_to_block(then_block);
        func_builder.build_branch(merge_block);

        func_builder.switch_to_block(else_block);
        func_builder.build_branch(merge_block);

        func_builder.switch_to_block(merge_block);
        func_builder.build_return(None);

        let func = func_builder.build();

        let mut dom_analysis = DominatorAnalysis::new();
        dom_analysis.compute_dominators(&func).unwrap();

        // Entry should dominate all blocks
        // Merge block should dominate itself
        // Then and else blocks should dominate themselves and merge

        println!("Dominator analysis completed successfully");
    }
}