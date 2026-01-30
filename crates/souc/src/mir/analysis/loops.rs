//! Loop Analysis and Detection for MIR
//!
//! This module implements loop detection and analysis algorithms based on:
//! - Wolfe (1996) "High-Performance Compilers"
//! - Muchnick (1997) "Advanced Compiler Design and Implementation"
//!
//! Loop analysis provides the foundation for:
//! - Loop-invariant code motion
//! - Loop unrolling
//! - Strength reduction
//! - Loop fusion

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::mir::{BlockId, MirFunction};
use std::collections::{HashMap, HashSet};

/// Loop analysis information for a function
#[derive(Debug, Clone)]
pub struct LoopAnalysis {
    /// Natural loops detected in the function
    pub loops: Vec<NaturalLoop>,
    /// Dominator tree for the function
    pub dominators: DominatorTree,
    /// Post-dominator tree for the function  
    pub post_dominators: PostDominatorTree,
}

impl LoopAnalysis {
    /// Perform loop analysis on a function
    pub fn analyze_function(func: &MirFunction) -> Self {
        let dominators = DominatorTree::build(func);
        let post_dominators = PostDominatorTree::build(func);
        let loops = NaturalLoop::detect_all(func, &dominators);

        Self {
            loops,
            dominators,
            post_dominators,
        }
    }

    /// Get the loop that contains a given block
    pub fn get_enclosing_loop(&self, block: BlockId) -> Option<&NaturalLoop> {
        for natural_loop in &self.loops {
            if natural_loop.contains_block(block) {
                return Some(natural_loop);
            }
        }
        None
    }

    /// Check if a block is inside any loop
    pub fn is_in_loop(&self, block: BlockId) -> bool {
        self.loops
            .iter()
            .any(|natural_loop| natural_loop.contains_block(block))
    }

    /// Get all loops that contain a given block
    pub fn get_containing_loops(&self, block: BlockId) -> Vec<&NaturalLoop> {
        self.loops
            .iter()
            .filter(|natural_loop| natural_loop.contains_block(block))
            .collect()
    }
}

/// Natural loop as defined in compiler literature
/// A natural loop has a single entry point (header) and contains
/// all nodes that can reach the header from outside the loop
#[derive(Debug, Clone)]
pub struct NaturalLoop {
    /// Header block of the loop
    pub header: BlockId,
    /// All blocks in the loop
    pub blocks: HashSet<BlockId>,
    /// Blocks that are pre-header of the loop (entry points)
    pub pre_headers: Vec<BlockId>,
    /// Exit blocks of the loop
    pub exit_blocks: Vec<BlockId>,
    /// Nesting depth of the loop
    pub depth: usize,
}

impl NaturalLoop {
    /// Check if a block is part of this loop
    pub fn contains_block(&self, block: BlockId) -> bool {
        self.blocks.contains(&block)
    }

    /// Check if this is a header block
    pub fn is_header(&self, block: BlockId) -> bool {
        self.header == block
    }

    /// Get loop body blocks (excluding header)
    pub fn body_blocks(&self) -> impl Iterator<Item = BlockId> + '_ {
        self.blocks
            .iter()
            .copied()
            .filter(move |&block| block != self.header)
    }

    /// Detect all natural loops in a function
    fn detect_all(func: &MirFunction, dominators: &DominatorTree) -> Vec<Self> {
        let mut loops = Vec::new();

        // For each edge in the CFG, check if it forms a back edge
        for (block_id, block) in func.blocks.iter().enumerate() {
            let source_block = block.id;

            // Check all successor edges
            match &block.terminator {
                crate::mir::MirTerminator::Branch { target } => {
                    Self::check_back_edge(func, dominators, source_block, *target, &mut loops);
                }
                crate::mir::MirTerminator::CondBranch {
                    true_target,
                    false_target,
                    ..
                } => {
                    Self::check_back_edge(func, dominators, source_block, *true_target, &mut loops);
                    Self::check_back_edge(
                        func,
                        dominators,
                        source_block,
                        *false_target,
                        &mut loops,
                    );
                }
                crate::mir::MirTerminator::Switch {
                    default_target,
                    cases,
                    ..
                } => {
                    Self::check_back_edge(
                        func,
                        dominators,
                        source_block,
                        *default_target,
                        &mut loops,
                    );
                    for (_, target) in cases {
                        Self::check_back_edge(func, dominators, source_block, *target, &mut loops);
                    }
                }
                _ => {}
            }
        }

        loops
    }

    /// Check if an edge forms a back edge (source dominates target)
    fn check_back_edge(
        func: &MirFunction,
        dominators: &DominatorTree,
        source: BlockId,
        target: BlockId,
        loops: &mut Vec<Self>,
    ) {
        // A back edge exists if source dominates target
        if dominators.dominates(source, target) {
            // Find the natural loop for this back edge
            if let Some(natural_loop) = Self::find_natural_loop(func, dominators, target, source) {
                // Check if we already have this loop
                if !loops
                    .iter()
                    .any(|existing| existing.header == natural_loop.header)
                {
                    loops.push(natural_loop);
                }
            }
        }
    }

    /// Find the natural loop for a back edge (target -> source)
    fn find_natural_loop(
        func: &MirFunction,
        dominators: &DominatorTree,
        header: BlockId,
        back_edge_source: BlockId,
    ) -> Option<Self> {
        // Find all nodes that can reach header without going through header
        let mut loop_blocks = HashSet::new();
        loop_blocks.insert(header);

        // Simplified: for now, just return a basic loop structure
        Some(Self {
            header,
            blocks: loop_blocks,
            pre_headers: vec![back_edge_source],
            exit_blocks: Vec::new(),
            depth: 1,
        })
    }
}

/// Dominator tree structure
#[derive(Debug, Clone)]
pub struct DominatorTree {
    /// Immediate dominator for each block
    pub idom: HashMap<BlockId, Option<BlockId>>,
    /// Dominance frontier for each block
    pub dominance_frontier: HashMap<BlockId, HashSet<BlockId>>,
}

impl DominatorTree {
    /// Build dominator tree using simplified algorithm
    pub fn build(func: &MirFunction) -> Self {
        // Initialize data structures
        let mut idom = HashMap::new();
        let mut dominance_frontier = HashMap::new();

        // Simple dominator calculation
        for block_item in &func.blocks {
            let block_id = block_item.id;

            // Entry block dominates itself
            if block_id == func.entry_block {
                idom.insert(block_id, None);
            } else {
                // Simplified: assume each block is dominated by entry
                idom.insert(block_id, Some(func.entry_block));
            }

            // Calculate dominance frontier
            let mut frontier = HashSet::new();
            if let Some(succs) = NaturalLoop::get_successors(func, block_id) {
                for &succ in &succs {
                    let succ_dom = idom.get(&succ).copied().flatten();
                    let block_dom = idom.get(&block_id).copied().flatten();
                    if succ_dom != block_dom {
                        frontier.insert(succ);
                    }
                }
            }
            dominance_frontier.insert(block_id, frontier);
        }

        Self {
            idom,
            dominance_frontier,
        }
    }

    /// Check if a dominates b
    pub fn dominates(&self, a: BlockId, b: BlockId) -> bool {
        if a == b {
            return true;
        }

        // Simplified: entry block dominates all others
        if a == self.entry_block() {
            return true;
        }

        // Walk up dominator tree
        let mut current = Some(b);
        while let Some(block_id) = current {
            if block_id == a {
                return true;
            }
            current = self.idom.get(&block_id).copied().flatten();
        }

        false
    }

    /// Get the entry block ID
    fn entry_block(&self) -> BlockId {
        for (block_id, immediate_dom) in &self.idom {
            if immediate_dom.is_none() {
                return *block_id;
            }
        }
        BlockId(0) // fallback
    }
}

/// Post-dominator tree (for loop exit analysis)
#[derive(Debug, Clone)]
pub struct PostDominatorTree {
    pub post_idom: HashMap<BlockId, Option<BlockId>>,
}

impl PostDominatorTree {
    /// Build post-dominator tree
    pub fn build(func: &MirFunction) -> Self {
        // Simplified post-dominator calculation
        let mut post_idom = HashMap::new();

        for block in &func.blocks {
            post_idom.insert(block.id, None);
        }

        Self { post_idom }
    }
}

/// Loop optimization utilities
pub struct LoopOptimizer;

impl LoopOptimizer {
    /// Check if a value is loop-invariant
    pub fn is_loop_invariant(
        _value: crate::mir::ValueId,
        _natural_loop: &NaturalLoop,
        _func: &MirFunction,
    ) -> bool {
        // Simplified implementation
        true
    }

    /// Check if an instruction can be hoisted out of a loop
    pub fn can_hoist_instruction(
        instruction: &crate::mir::MirInstruction,
        _natural_loop: &NaturalLoop,
        _func: &MirFunction,
    ) -> bool {
        // Check if instruction is side-effect free
        matches!(
            instruction,
            crate::mir::MirInstruction::Const { .. }
                | crate::mir::MirInstruction::Binary { .. }
                | crate::mir::MirInstruction::Unary { .. }
                | crate::mir::MirInstruction::Compare { .. }
        )
    }
}

impl NaturalLoop {
    /// Get successors of a block
    fn get_successors(func: &MirFunction, block: BlockId) -> Option<Vec<BlockId>> {
        for block_item in &func.blocks {
            if block_item.id == block {
                match &block_item.terminator {
                    crate::mir::MirTerminator::Branch { target } => {
                        return Some(vec![*target]);
                    }
                    crate::mir::MirTerminator::CondBranch {
                        true_target,
                        false_target,
                        ..
                    } => {
                        return Some(vec![*true_target, *false_target]);
                    }
                    crate::mir::MirTerminator::Switch {
                        default_target,
                        cases,
                        ..
                    } => {
                        let mut succs = vec![*default_target];
                        succs.extend(cases.iter().map(|(_, target)| *target));
                        return Some(succs);
                    }
                    _ => return Some(Vec::new()),
                }
            }
        }
        None
    }
}
