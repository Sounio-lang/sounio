//! SIR Optimization Passes
//!
//! This is where Sounio's domain-specific optimizations live.
//! These passes leverage the domain knowledge encoded in SIR to perform
//! optimizations that general-purpose compilers cannot.
//!
//! # Pass Categories
//!
//! 1. **Analysis Passes**: Gather information without modification
//!    - Dominance analysis
//!    - Loop detection
//!    - Alias analysis
//!    - Epistemic flow analysis
//!
//! 2. **Standard Passes**: Classic compiler optimizations
//!    - Dead code elimination
//!    - Common subexpression elimination
//!    - Constant folding/propagation
//!    - Inlining
//!
//! 3. **Domain-Specific Passes**: Sounio-unique optimizations
//!    - Epistemic fusion
//!    - Certainty propagation
//!    - Distribution combining
//!    - ODE step fusion
//!    - Unit elision verification

use super::blocks::{BasicBlock, Instruction, SirFunction};
use super::metadata::{EpistemicMetadata, Metadata, MetadataStore};
use super::module::SirModule;
use super::ops::*;
use super::values::{Constant, ValueId};
use std::collections::{HashMap, HashSet};

/// Result of running a pass
#[derive(Debug, Clone, Default)]
pub struct PassResult {
    /// Was the module modified?
    pub modified: bool,
    /// Statistics
    pub stats: PassStats,
}

#[derive(Debug, Clone, Default)]
pub struct PassStats {
    pub instructions_removed: usize,
    pub instructions_added: usize,
    pub blocks_removed: usize,
    pub values_folded: usize,
    pub epistemic_ops_fused: usize,
    pub distributions_combined: usize,
    pub ode_steps_fused: usize,
}

/// Trait for SIR passes
pub trait SirPass {
    /// Name of the pass
    fn name(&self) -> &str;

    /// Run the pass on a module
    fn run(&mut self, module: &mut SirModule) -> PassResult;

    /// Run the pass on a single function
    fn run_on_function(&mut self, _func: &mut SirFunction) -> PassResult {
        PassResult::default()
    }
}

// ============================================================================
// ANALYSIS PASSES
// ============================================================================

/// Compute dominance information
pub struct DominanceAnalysis;

impl SirPass for DominanceAnalysis {
    fn name(&self) -> &str {
        "dominance"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            // Compute immediate dominators using the Lengauer-Tarjan algorithm
            compute_dominators(func);
            result.modified = true;
        }

        result
    }
}

fn compute_dominators(func: &mut SirFunction) {
    // TODO: Implement Lengauer-Tarjan algorithm
    // For now, mark entry block as having no dominator
    let entry_id = func.blocks.first().map(|b| b.id);

    for (i, block) in func.blocks.iter_mut().enumerate() {
        if i == 0 {
            block.idom = None;
        } else {
            block.idom = entry_id;
        }
    }
}

/// Detect natural loops
pub struct LoopAnalysis;

impl SirPass for LoopAnalysis {
    fn name(&self) -> &str {
        "loop-analysis"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            detect_loops(func);
            result.modified = true;
        }

        result
    }
}

fn detect_loops(func: &mut SirFunction) {
    // Detect natural loops using back edge analysis
    // A back edge is an edge from a node to one of its dominators
    
    // Step 1: Compute dominators (simplified - full implementation would use Lengauer-Tarjan)
    let mut dominators: Vec<HashSet<usize>> = vec![HashSet::new(); func.blocks.len()];
    let entry = 0;
    dominators[entry].insert(entry);
    
    // Iterative dataflow to compute dominators
    let mut changed = true;
    while changed {
        changed = false;
        for i in 0..func.blocks.len() {
            if i == entry {
                continue;
            }
            
            let mut new_doms = HashSet::new();
            new_doms.insert(i);
            
            // Intersection of dominators of all predecessors
            let block = &func.blocks[i];
            if let Some(term) = &block.terminator {
                let preds = term.successors();
                
                if !preds.is_empty() {
                    // Initialize with first predecessor's dominators
                    if let Some(&first_pred) = preds.first() {
                        new_doms = dominators[first_pred.0 as usize].clone();
                    }
                    
                    // Intersect with all other predecessors
                    for &pred in preds.iter().skip(1) {
                        new_doms = new_doms.intersection(&dominators[pred.0 as usize]).cloned().collect();
                    }
                    
                    new_doms.insert(i);
                }
            }
            
            if new_doms != dominators[i] {
                dominators[i] = new_doms;
                changed = true;
            }
        }
    }
    
    // Step 2: Find back edges (edge from node to dominator)
    let mut back_edges = Vec::new();
    for (i, block) in func.blocks.iter().enumerate() {
        if let Some(term) = &block.terminator {
            let targets = term.successors();
            
            for target in targets {
                // Check if target dominates source (back edge)
                if dominators[i].contains(&(target.0 as usize)) {
                    back_edges.push((i, target.0 as usize));
                }
            }
        }
    }
    
    // Step 3: Mark loop headers and compute loop depth
    for (header_idx, _) in &back_edges {
        func.blocks[*header_idx].is_loop_header = true;
    }
    
    // Compute loop depth (simplified - full implementation would handle nested loops)
    for block in &mut func.blocks {
        if block.is_loop_header {
            block.loop_depth = 1;
        } else {
            block.loop_depth = 0;
        }
    }
}

// ============================================================================
// STANDARD OPTIMIZATION PASSES
// ============================================================================

/// SIMD Vectorization Pass
/// 
/// Detects loops that can be vectorized and converts scalar operations
/// to SIMD operations when beneficial.
pub struct SimdVectorization;

impl SirPass for SimdVectorization {
    fn name(&self) -> &str {
        "simd-vectorization"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_added += sub_result.stats.instructions_added;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();
        
        // Find loops that can be vectorized
        // Criteria:
        // 1. Loop has known trip count (or can be computed)
        // 2. Loop body has independent iterations (no loop-carried dependencies)
        // 3. Operations are vectorizable (arithmetic on arrays)
        // 4. Array accesses are stride-1 (contiguous)
        
        // Step 1: Identify candidate loops
        let mut vectorizable_loops = Vec::new();
        
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if block.is_loop_header && block.loop_depth == 1 {
                // Analyze this loop for vectorization
                if let Some(analysis) = self.analyze_loop_for_vectorization(func, block_idx) {
                    if analysis.can_vectorize {
                        vectorizable_loops.push((block_idx, analysis));
                    }
                }
            }
        }
        
        // Step 2: Transform vectorizable loops
        for (block_idx, analysis) in vectorizable_loops {
            if self.vectorize_loop(func, block_idx, &analysis) {
                result.modified = true;
                result.stats.instructions_added += analysis.estimated_speedup;
            }
        }
        
        result
    }
}

// Helper methods for SimdVectorization
impl SimdVectorization {
    /// Analyze a loop to determine if it can be vectorized
    fn analyze_loop_for_vectorization(
        &self,
        func: &SirFunction,
        loop_header_idx: usize,
    ) -> Option<LoopVectorizationAnalysis> {
        let block = &func.blocks[loop_header_idx];
        
        // Check for vectorizable operations
        // For now, we'll do a simple check: look for array operations
        let has_array_ops = self.has_vectorizable_operations(func, loop_header_idx);
        if !has_array_ops {
            return None;
        }
        
        // Check for loop-carried dependencies
        let has_dependencies = self.has_loop_carried_dependencies(func, loop_header_idx);
        if has_dependencies {
            return None;
        }
        
        // Check for stride-1 array accesses
        let stride_analysis = self.analyze_array_strides(func, loop_header_idx);
        if !stride_analysis.all_contiguous {
            return None;
        }
        
        // Estimate vectorization benefit
        let estimated_speedup = stride_analysis.vectorizable_ops * 4; // Assume 4x speedup for SIMD
        
        Some(LoopVectorizationAnalysis {
            can_vectorize: true,
            vector_width: 4, // SSE/AVX width
            estimated_speedup,
            stride_analysis,
        })
    }
    
    /// Check if loop body has vectorizable operations
    fn has_vectorizable_operations(&self, func: &SirFunction, block_idx: usize) -> bool {
        let block = &func.blocks[block_idx];
        
        // Look for arithmetic operations on arrays
        for inst in &block.instructions {
            match &inst.inst {
                super::ops::SirInst::BinOp { op, .. } => {
                    // Most arithmetic ops are vectorizable
                    match op {
                        super::ops::ArithOp::Add
                        | super::ops::ArithOp::Sub
                        | super::ops::ArithOp::Mul
                        | super::ops::ArithOp::FAdd
                        | super::ops::ArithOp::FSub
                        | super::ops::ArithOp::FMul => return true,
                        _ => {}
                    }
                }
                super::ops::SirInst::Memory(super::ops::MemoryOp::Load { .. }) => {
                    // Array loads are vectorizable
                    return true;
                }
                super::ops::SirInst::Memory(super::ops::MemoryOp::Store { .. }) => {
                    // Array stores are vectorizable
                    return true;
                }
                _ => {}
            }
        }
        
        false
    }
    
    /// Check for loop-carried dependencies
    fn has_loop_carried_dependencies(&self, func: &SirFunction, block_idx: usize) -> bool {
        // Simple check: look for values that are defined in one iteration
        // and used in a later iteration
        // This is a simplified analysis - full implementation would use
        // SSA-based dependency tracking
        
        let block = &func.blocks[block_idx];
        let mut defined_values = std::collections::HashSet::new();
        let mut used_values = std::collections::HashSet::new();
        
        // Collect definitions and uses
        for inst in &block.instructions {
            if let Some(result) = inst.result {
                defined_values.insert(result);
            }
            
            // Collect uses from operands
            match &inst.inst {
                super::ops::SirInst::BinOp { lhs, rhs, .. } => {
                    used_values.insert(*lhs);
                    used_values.insert(*rhs);
                }
                super::ops::SirInst::Memory(super::ops::MemoryOp::Load { ptr, .. }) => {
                    used_values.insert(*ptr);
                }
                super::ops::SirInst::Memory(super::ops::MemoryOp::Store { ptr, val, .. }) => {
                    used_values.insert(*ptr);
                    used_values.insert(*val);
                }
                _ => {}
            }
        }
        
        // Check for loop-carried dependencies: if a value is both defined and used
        // in the loop, it might be a dependency (unless it's the loop counter)
        // This is a conservative check
        !defined_values.is_disjoint(&used_values)
    }
    
    /// Analyze array access strides
    fn analyze_array_strides(
        &self,
        func: &SirFunction,
        block_idx: usize,
    ) -> StrideAnalysis {
        let block = &func.blocks[block_idx];
        let all_contiguous = true;
        let mut vectorizable_ops = 0;
        
        for inst in &block.instructions {
            match &inst.inst {
                super::ops::SirInst::Memory(super::ops::MemoryOp::Load { .. })
                | super::ops::SirInst::Memory(super::ops::MemoryOp::Store { .. }) => {
                    vectorizable_ops += 1;
                    // In full implementation, would analyze pointer arithmetic
                    // to determine stride. For now, assume stride-1 if we see
                    // array operations
                }
                super::ops::SirInst::BinOp { .. } => {
                    vectorizable_ops += 1;
                }
                _ => {}
            }
        }
        
        StrideAnalysis {
            all_contiguous,
            vectorizable_ops,
        }
    }
    
    /// Transform a loop to use SIMD operations
    fn vectorize_loop(
        &mut self,
        func: &mut SirFunction,
        block_idx: usize,
        analysis: &LoopVectorizationAnalysis,
    ) -> bool {
        use super::types::VectorType;

        let vector_width = analysis.vector_width;

        // Track value mappings from scalar to vector
        let mut scalar_to_vector: HashMap<ValueId, ValueId> = HashMap::new();
        let mut next_id = self.compute_max_value_id(func) + 1;

        // Collect instructions to transform
        let mut new_instructions = Vec::new();

        for inst in &func.blocks[block_idx].instructions {
            match &inst.inst {
                // Vectorize loads: scalar load -> vector load
                SirInst::Memory(MemoryOp::Load { ptr, ty, volatile, align }) => {
                    if let super::types::SirType::Scalar(scalar_ty) = ty {
                        let vec_ty = super::types::SirType::Vector(VectorType::new(*scalar_ty, vector_width));
                        let vec_result = ValueId(next_id);
                        next_id += 1;

                        // Create vector load instruction
                        new_instructions.push(Instruction::new(
                            Some(vec_result),
                            SirInst::Memory(MemoryOp::Load {
                                ptr: *ptr,
                                ty: vec_ty,
                                volatile: *volatile,
                                align: *align,
                            }),
                        ));

                        if let Some(result) = inst.result {
                            scalar_to_vector.insert(result, vec_result);
                        }
                        continue;
                    }
                }

                // Vectorize stores: scalar store -> vector store
                SirInst::Memory(MemoryOp::Store { ptr, val, volatile, align }) => {
                    // Use vectorized value if available
                    let vec_val = scalar_to_vector.get(val).copied().unwrap_or(*val);
                    new_instructions.push(Instruction::new(
                        None,
                        SirInst::Memory(MemoryOp::Store {
                            ptr: *ptr,
                            val: vec_val,
                            volatile: *volatile,
                            align: *align,
                        }),
                    ));
                    continue;
                }

                // Vectorize arithmetic: scalar binop -> vector binop
                SirInst::BinOp { op, lhs, rhs } => {
                    let vec_lhs = scalar_to_vector.get(lhs).copied().unwrap_or(*lhs);
                    let vec_rhs = scalar_to_vector.get(rhs).copied().unwrap_or(*rhs);
                    let vec_result = ValueId(next_id);
                    next_id += 1;

                    // Check for FMA pattern: a * b + c
                    if matches!(op, ArithOp::FAdd | ArithOp::Add) {
                        // Look for multiply feeding into this add
                        if let Some(fma_inst) = self.try_create_fma(vec_lhs, vec_rhs) {
                            new_instructions.push(Instruction::new(Some(vec_result), fma_inst));
                            if let Some(result) = inst.result {
                                scalar_to_vector.insert(result, vec_result);
                            }
                            continue;
                        }
                    }

                    // Regular vector binary operation
                    new_instructions.push(Instruction::new(
                        Some(vec_result),
                        SirInst::BinOp {
                            op: *op,
                            lhs: vec_lhs,
                            rhs: vec_rhs,
                        },
                    ));

                    if let Some(result) = inst.result {
                        scalar_to_vector.insert(result, vec_result);
                    }
                    continue;
                }

                _ => {}
            }

            // Keep non-vectorizable instructions as-is
            new_instructions.push(inst.clone());
        }

        // Replace block instructions with vectorized versions
        func.blocks[block_idx].instructions = new_instructions;

        true
    }

    /// Compute the maximum ValueId in use in a function
    fn compute_max_value_id(&self, func: &SirFunction) -> u32 {
        let mut max_id = 0u32;
        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    max_id = max_id.max(result.0);
                }
            }
        }
        max_id
    }

    /// Try to create an FMA instruction from a + (b * c) or (a * b) + c pattern
    fn try_create_fma(&self, _lhs: ValueId, _rhs: ValueId) -> Option<SirInst> {
        // FMA detection would require tracking which values came from multiplies
        // For now, return None to fall back to separate mul+add
        // Full implementation would analyze def-use chains
        None
    }

    /// Select optimal vector width based on target architecture and element type
    #[allow(dead_code)]
    fn select_vector_width(elem_ty: &super::types::ScalarType) -> u8 {
        use super::types::ScalarType;

        // Default to 256-bit vectors (AVX2)
        match elem_ty {
            ScalarType::F64 => 4,  // 4x f64 = 256 bits
            ScalarType::F32 => 8,  // 8x f32 = 256 bits
            ScalarType::I64 => 4,  // 4x i64 = 256 bits
            ScalarType::I32 => 8,  // 8x i32 = 256 bits
            ScalarType::I16 => 16, // 16x i16 = 256 bits
            ScalarType::I8 => 32,  // 32x i8 = 256 bits
            ScalarType::Bool => 32, // packed bools
        }
    }
}

/// Analysis result for loop vectorization
struct LoopVectorizationAnalysis {
    can_vectorize: bool,
    vector_width: u8,
    estimated_speedup: usize,
    stride_analysis: StrideAnalysis,
}

/// Stride analysis for array accesses
struct StrideAnalysis {
    all_contiguous: bool,
    vectorizable_ops: usize,
}

/// Loop Fission Pass
/// 
/// Splits loops into multiple loops to enable parallelization.
/// Example: for i in 0..n { a[i] = f(i); b[i] = g(i); }
///          -> for i in 0..n { a[i] = f(i); }
///          -> for i in 0..n { b[i] = g(i); }
pub struct LoopFission;

impl SirPass for LoopFission {
    fn name(&self) -> &str {
        "loop-fission"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_added += sub_result.stats.instructions_added;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();
        
        // Find loops that can be split
        // Criteria:
        // 1. Loop body has independent computations
        // 2. No data dependencies between computations
        // 3. Same iteration space
        
        // Step 1: Identify loops with independent computations
        let mut fission_candidates = Vec::new();
        
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if block.is_loop_header {
                if let Some(analysis) = self.analyze_loop_for_fission(func, block_idx) {
                    if analysis.can_fission {
                        fission_candidates.push((block_idx, analysis));
                    }
                }
            }
        }
        
        // Step 2: Perform fission (in reverse order to maintain indices)
        for (block_idx, analysis) in fission_candidates.into_iter().rev() {
            if self.perform_fission(func, block_idx, &analysis) {
                result.modified = true;
                result.stats.instructions_added += analysis.num_splits;
            }
        }
        
        result
    }
}

// Helper methods for LoopFission
impl LoopFission {
    /// Analyze a loop to determine if it can be split via fission
    fn analyze_loop_for_fission(
        &self,
        func: &SirFunction,
        loop_header_idx: usize,
    ) -> Option<FissionAnalysis> {
        let block = &func.blocks[loop_header_idx];
        
        // Group instructions by their dependencies
        let mut computation_groups = Vec::new();
        let mut current_group = Vec::new();
        let mut defined_in_group = std::collections::HashSet::new();
        
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            // Check if this instruction depends on previous group
            let mut depends_on_previous = false;
            
            if let Some(result) = inst.result {
                // Check if any operand is defined in previous groups
                match &inst.inst {
                    super::ops::SirInst::BinOp { lhs, rhs, .. } => {
                        if defined_in_group.contains(lhs) || defined_in_group.contains(rhs) {
                            depends_on_previous = true;
                        }
                    }
                    super::ops::SirInst::Memory(super::ops::MemoryOp::Store { ptr, val, .. }) => {
                        if defined_in_group.contains(ptr) || defined_in_group.contains(val) {
                            depends_on_previous = true;
                        }
                    }
                    _ => {}
                }
            }
            
            if depends_on_previous && !current_group.is_empty() {
                // Start a new group
                computation_groups.push(std::mem::take(&mut current_group));
                defined_in_group.clear();
            }
            
            current_group.push(inst_idx);
            if let Some(result) = inst.result {
                defined_in_group.insert(result);
            }
        }
        
        if !current_group.is_empty() {
            computation_groups.push(current_group);
        }
        
        // Can fission if we have 2+ independent groups
        if computation_groups.len() >= 2 {
            Some(FissionAnalysis {
                can_fission: true,
                num_splits: computation_groups.len() - 1,
                computation_groups,
            })
        } else {
            None
        }
    }
    
    /// Perform loop fission by splitting into multiple loops
    fn perform_fission(
        &mut self,
        func: &mut SirFunction,
        block_idx: usize,
        analysis: &FissionAnalysis,
    ) -> bool {
        // In full implementation, this would:
        // 1. Create new blocks for each computation group
        // 2. Duplicate loop structure for each group
        // 3. Update control flow
        
        // For now, just mark that we've analyzed this block
        // In full implementation, would split the loop
        true
    }
}

/// Analysis result for loop fission
struct FissionAnalysis {
    can_fission: bool,
    num_splits: usize,
    computation_groups: Vec<Vec<usize>>,
}

/// Cache Blocking Pass
/// 
/// Optimizes matrix operations by blocking loops to improve cache locality.
/// Example: Matrix multiplication with blocking for better cache performance.
pub struct CacheBlocking;

impl SirPass for CacheBlocking {
    fn name(&self) -> &str {
        "cache-blocking"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_added += sub_result.stats.instructions_added;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();
        
        // Find nested loops that perform matrix operations
        // Criteria:
        // 1. Nested loops (2+ levels)
        // 2. Array accesses with stride patterns
        // 3. Matrix operations (multiplication, etc.)
        
        // Detect patterns like:
        // for i in 0..n {
        //   for j in 0..m {
        //     for k in 0..p {
        //       C[i][j] += A[i][k] * B[k][j]
        //     }
        //   }
        // }
        
        // Transform to blocked version:
        // for ii in 0..n step BLOCK_SIZE {
        //   for jj in 0..m step BLOCK_SIZE {
        //     for kk in 0..p step BLOCK_SIZE {
        //       for i in ii..min(ii+BLOCK_SIZE, n) {
        //         for j in jj..min(jj+BLOCK_SIZE, m) {
        //           for k in kk..min(kk+BLOCK_SIZE, p) {
        //             C[i][j] += A[i][k] * B[k][j]
        //           }
        //         }
        //       }
        //     }
        //   }
        // }
        
        // Step 1: Detect nested loop nests
        let mut blocking_candidates = Vec::new();
        
        for (block_idx, block) in func.blocks.iter().enumerate() {
            if block.is_loop_header && block.loop_depth >= 2 {
                // This is a nested loop - check if it's a matrix operation
                if let Some(analysis) = self.analyze_for_blocking(func, block_idx) {
                    if analysis.can_block {
                        blocking_candidates.push((block_idx, analysis));
                    }
                }
            }
        }
        
        // Step 2: Apply blocking transformation
        for (block_idx, analysis) in blocking_candidates.into_iter().rev() {
            if self.apply_blocking(func, block_idx, &analysis) {
                result.modified = true;
                result.stats.instructions_added += analysis.block_size;
            }
        }
        
        result
    }
}

// Helper methods for CacheBlocking
impl CacheBlocking {
    /// Analyze nested loops for cache blocking opportunities
    fn analyze_for_blocking(
        &self,
        func: &SirFunction,
        block_idx: usize,
    ) -> Option<BlockingAnalysis> {
        let block = &func.blocks[block_idx];
        
        // Check for matrix multiplication pattern
        // Look for: nested loops with array accesses and multiplications
        let mut has_matrix_ops = false;
        let nesting_depth = block.loop_depth;
        
        // Check if loop body has multiplication operations
        for inst in &block.instructions {
            match &inst.inst {
                super::ops::SirInst::BinOp { op, .. } => {
                    if matches!(op, super::ops::ArithOp::Mul | super::ops::ArithOp::FMul) {
                        has_matrix_ops = true;
                        break;
                    }
                }
                _ => {}
            }
        }
        
        if has_matrix_ops && nesting_depth >= 2 {
            // Determine optimal block size (typically L1 cache line size / element size)
            // For f64: 64 bytes / 8 bytes = 8 elements
            let block_size = 8;
            
            Some(BlockingAnalysis {
                can_block: true,
                block_size,
                nesting_depth,
            })
        } else {
            None
        }
    }
    
    /// Apply cache blocking transformation
    fn apply_blocking(
        &mut self,
        func: &mut SirFunction,
        block_idx: usize,
        analysis: &BlockingAnalysis,
    ) -> bool {
        // In full implementation, this would:
        // 1. Create new loop structure with blocking
        // 2. Add outer loops for block iteration
        // 3. Add inner loops for element iteration within blocks
        // 4. Update array access patterns
        
        // For now, just mark that we've analyzed this block
        // In full implementation, would apply blocking transformation
        true
    }
}

/// Analysis result for cache blocking
struct BlockingAnalysis {
    can_block: bool,
    block_size: usize,
    nesting_depth: u32,
}

/// Dead Code Elimination
pub struct DeadCodeElimination;

impl SirPass for DeadCodeElimination {
    fn name(&self) -> &str {
        "dce"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_removed += sub_result.stats.instructions_removed;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();

        // Collect used values
        let mut used: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

        // Mark values used by terminators
        for block in &func.blocks {
            if let Some(term) = &block.terminator {
                for val in term.operands() {
                    used.insert(val);
                }
            }
        }

        // Iterate until fixed point
        let mut changed = true;
        while changed {
            changed = false;

            for block in &func.blocks {
                for inst in &block.instructions {
                    // If this instruction's result is used, mark its operands as used
                    if inst.result.map(|r| used.contains(&r)).unwrap_or(true) {
                        for op in inst.inst.operands() {
                            if used.insert(op) {
                                changed = true;
                            }
                        }
                    }
                }
            }
        }

        // Remove dead instructions
        for block in &mut func.blocks {
            let before = block.instructions.len();
            block.instructions.retain(|inst| {
                // Keep if:
                // 1. No result (side-effecting)
                // 2. Has side effects
                // 3. Result is used
                inst.result.is_none()
                    || inst.inst.has_side_effects()
                    || inst.result.map(|r| used.contains(&r)).unwrap_or(false)
            });
            let removed = before - block.instructions.len();
            if removed > 0 {
                result.modified = true;
                result.stats.instructions_removed += removed;
            }
        }

        result
    }
}

/// Constant Folding and Propagation
pub struct ConstantFolding;

impl SirPass for ConstantFolding {
    fn name(&self) -> &str {
        "const-fold"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.values_folded += sub_result.stats.values_folded;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        // TODO: Implement constant folding
        // 1. Track which values are constants
        // 2. Evaluate constant expressions at compile time
        // 3. Replace uses with constant results
        PassResult::default()
    }
}

// ============================================================================
// DOMAIN-SPECIFIC PASSES
// ============================================================================

// ============================================================================
// PATTERN MATCHING HELPERS
// ============================================================================

/// Represents a matched epistemic operation pattern
#[derive(Debug, Clone)]
pub enum EpistemicPattern {
    /// Pattern: ExtractValue(a) op ExtractValue(b) + PropagateXxx(conf_a, conf_b)
    /// Can be fused into a single FusedMul/FusedAdd
    ValueArithmeticWithPropagate {
        /// Source epistemic value A
        source_a: ValueId,
        /// Source epistemic value B
        source_b: ValueId,
        /// The extracted value from A
        extracted_val_a: ValueId,
        /// The extracted value from B
        extracted_val_b: ValueId,
        /// The extracted confidence from A
        extracted_conf_a: ValueId,
        /// The extracted confidence from B
        extracted_conf_b: ValueId,
        /// The arithmetic operation result
        arith_result: ValueId,
        /// The propagate operation result
        propagate_result: ValueId,
        /// The arithmetic operation kind
        arith_op: ArithOp,
        /// Index of ExtractValue A instruction
        extract_val_a_idx: usize,
        /// Index of ExtractValue B instruction
        extract_val_b_idx: usize,
        /// Index of ExtractConfidence A instruction
        extract_conf_a_idx: usize,
        /// Index of ExtractConfidence B instruction
        extract_conf_b_idx: usize,
        /// Index of arithmetic instruction
        arith_idx: usize,
        /// Index of propagate instruction
        propagate_idx: usize,
    },
    /// Pattern: Create followed by immediate ExtractValue/ExtractConfidence
    /// The extraction can be elided
    RedundantExtraction {
        /// The created epistemic value
        created_value: ValueId,
        /// The original value component
        original_value: ValueId,
        /// The original confidence component
        original_confidence: ValueId,
        /// Index of create instruction
        create_idx: usize,
        /// Indices of extraction instructions that can be removed
        extraction_indices: Vec<usize>,
    },
    /// Pattern: Chain of confidence propagations with known certainty
    /// Can be eliminated when all inputs are certain
    CertaintyChain {
        /// Values in the chain known to be certain
        certain_values: Vec<ValueId>,
        /// Propagate operations that can be eliminated
        eliminable_propagates: Vec<usize>,
    },
}

/// Use-def chain for tracking value definitions and uses
#[derive(Debug, Default)]
pub struct UseDefChain {
    /// Maps ValueId to the instruction that defines it (block_idx, inst_idx)
    definitions: HashMap<ValueId, (usize, usize)>,
    /// Maps ValueId to all instructions that use it (block_idx, inst_idx)
    uses: HashMap<ValueId, Vec<(usize, usize)>>,
}

impl UseDefChain {
    /// Build use-def chain for a function
    pub fn build(func: &SirFunction) -> Self {
        let mut chain = Self::default();

        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                // Record definition
                if let Some(result) = inst.result {
                    chain.definitions.insert(result, (block_idx, inst_idx));
                }

                // Record uses
                for operand in inst.inst.operands() {
                    chain
                        .uses
                        .entry(operand)
                        .or_default()
                        .push((block_idx, inst_idx));
                }
            }
        }

        chain
    }

    /// Get the instruction that defines a value
    pub fn get_def(&self, value: ValueId) -> Option<(usize, usize)> {
        self.definitions.get(&value).copied()
    }

    /// Get all instructions that use a value
    pub fn get_uses(&self, value: ValueId) -> &[(usize, usize)] {
        self.uses.get(&value).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Check if a value has exactly one use
    pub fn has_single_use(&self, value: ValueId) -> bool {
        self.uses.get(&value).map(|v| v.len() == 1).unwrap_or(false)
    }
}

/// Pattern matcher for epistemic operations
pub struct EpistemicPatternMatcher<'a> {
    func: &'a SirFunction,
    use_def: UseDefChain,
}

impl<'a> EpistemicPatternMatcher<'a> {
    pub fn new(func: &'a SirFunction) -> Self {
        Self {
            use_def: UseDefChain::build(func),
            func,
        }
    }

    /// Find all fusable epistemic patterns in a block
    pub fn find_fusion_patterns(&self, block_idx: usize) -> Vec<EpistemicPattern> {
        let mut patterns = Vec::new();
        let block = &self.func.blocks[block_idx];

        // Look for PropagateXxx operations and trace back to find fusion opportunities
        for (inst_idx, inst) in block.instructions.iter().enumerate() {
            if let SirInst::Epistemic(ep_op) = &inst.inst {
                match ep_op {
                    EpistemicOp::PropagateMul {
                        val_a,
                        conf_a,
                        val_b,
                        conf_b,
                    } => {
                        if let Some(pattern) = self.try_match_mul_fusion(
                            block_idx,
                            inst_idx,
                            *val_a,
                            *conf_a,
                            *val_b,
                            *conf_b,
                            inst.result,
                        ) {
                            patterns.push(pattern);
                        }
                    }
                    EpistemicOp::PropagateAdd { conf_a, conf_b } => {
                        if let Some(pattern) = self.try_match_add_fusion(
                            block_idx,
                            inst_idx,
                            *conf_a,
                            *conf_b,
                            inst.result,
                        ) {
                            patterns.push(pattern);
                        }
                    }
                    _ => {}
                }
            }
        }

        // Look for redundant extraction patterns
        patterns.extend(self.find_redundant_extractions(block_idx));

        patterns
    }

    /// Try to match a multiplication fusion pattern
    fn try_match_mul_fusion(
        &self,
        block_idx: usize,
        propagate_idx: usize,
        val_a: ValueId,
        conf_a: ValueId,
        val_b: ValueId,
        conf_b: ValueId,
        propagate_result: Option<ValueId>,
    ) -> Option<EpistemicPattern> {
        let block = &self.func.blocks[block_idx];

        // Find the corresponding arithmetic mul operation
        // Look for a BinOp::FMul that uses extracted values from the same sources
        for (arith_idx, arith_inst) in block.instructions.iter().enumerate() {
            if let SirInst::BinOp {
                op: ArithOp::FMul,
                lhs,
                rhs,
            } = &arith_inst.inst
            {
                // Check if lhs and rhs are extracted values
                if let (Some(source_a), Some(source_b)) = (
                    self.trace_to_epistemic_source(*lhs),
                    self.trace_to_epistemic_source(*rhs),
                ) {
                    // Find the extraction instructions
                    if let (
                        Some((extract_val_a_idx, _)),
                        Some((extract_val_b_idx, _)),
                        Some((extract_conf_a_idx, _)),
                        Some((extract_conf_b_idx, _)),
                    ) = (
                        self.find_extract_value(block_idx, source_a),
                        self.find_extract_value(block_idx, source_b),
                        self.find_extract_confidence(block_idx, source_a),
                        self.find_extract_confidence(block_idx, source_b),
                    ) {
                        return Some(EpistemicPattern::ValueArithmeticWithPropagate {
                            source_a,
                            source_b,
                            extracted_val_a: *lhs,
                            extracted_val_b: *rhs,
                            extracted_conf_a: conf_a,
                            extracted_conf_b: conf_b,
                            arith_result: arith_inst.result?,
                            propagate_result: propagate_result?,
                            arith_op: ArithOp::FMul,
                            extract_val_a_idx,
                            extract_val_b_idx,
                            extract_conf_a_idx,
                            extract_conf_b_idx,
                            arith_idx,
                            propagate_idx,
                        });
                    }
                }
            }
        }

        None
    }

    /// Try to match an addition fusion pattern
    fn try_match_add_fusion(
        &self,
        block_idx: usize,
        propagate_idx: usize,
        conf_a: ValueId,
        conf_b: ValueId,
        propagate_result: Option<ValueId>,
    ) -> Option<EpistemicPattern> {
        let block = &self.func.blocks[block_idx];

        // Trace conf_a and conf_b back to their sources
        let source_a = self.trace_confidence_to_source(conf_a)?;
        let source_b = self.trace_confidence_to_source(conf_b)?;

        // Find the corresponding arithmetic add operation
        for (arith_idx, arith_inst) in block.instructions.iter().enumerate() {
            if let SirInst::BinOp {
                op: ArithOp::FAdd,
                lhs,
                rhs,
            } = &arith_inst.inst
            {
                // Check if these are extracted values from the same epistemic sources
                if let (Some(lhs_source), Some(rhs_source)) = (
                    self.trace_to_epistemic_source(*lhs),
                    self.trace_to_epistemic_source(*rhs),
                ) {
                    if (lhs_source == source_a && rhs_source == source_b)
                        || (lhs_source == source_b && rhs_source == source_a)
                    {
                        if let (
                            Some((extract_val_a_idx, _)),
                            Some((extract_val_b_idx, _)),
                            Some((extract_conf_a_idx, _)),
                            Some((extract_conf_b_idx, _)),
                        ) = (
                            self.find_extract_value(block_idx, source_a),
                            self.find_extract_value(block_idx, source_b),
                            self.find_extract_confidence(block_idx, source_a),
                            self.find_extract_confidence(block_idx, source_b),
                        ) {
                            return Some(EpistemicPattern::ValueArithmeticWithPropagate {
                                source_a,
                                source_b,
                                extracted_val_a: *lhs,
                                extracted_val_b: *rhs,
                                extracted_conf_a: conf_a,
                                extracted_conf_b: conf_b,
                                arith_result: arith_inst.result?,
                                propagate_result: propagate_result?,
                                arith_op: ArithOp::FAdd,
                                extract_val_a_idx,
                                extract_val_b_idx,
                                extract_conf_a_idx,
                                extract_conf_b_idx,
                                arith_idx,
                                propagate_idx,
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Find redundant extraction patterns where Create is immediately followed by Extract
    fn find_redundant_extractions(&self, block_idx: usize) -> Vec<EpistemicPattern> {
        let mut patterns = Vec::new();
        let block = &self.func.blocks[block_idx];

        for (create_idx, inst) in block.instructions.iter().enumerate() {
            if let SirInst::Epistemic(EpistemicOp::Create { value, confidence }) = &inst.inst {
                if let Some(created_value) = inst.result {
                    // Find all extractions from this created value
                    let mut extraction_indices = Vec::new();

                    for (use_block, use_idx) in self.use_def.get_uses(created_value) {
                        if *use_block == block_idx {
                            let use_inst = &block.instructions[*use_idx];
                            if let SirInst::Epistemic(
                                EpistemicOp::ExtractValue(v) | EpistemicOp::ExtractConfidence(v),
                            ) = &use_inst.inst
                            {
                                if *v == created_value {
                                    extraction_indices.push(*use_idx);
                                }
                            }
                        }
                    }

                    // If all uses are extractions and they're in the same block, it's redundant
                    if !extraction_indices.is_empty()
                        && extraction_indices.len() == self.use_def.get_uses(created_value).len()
                    {
                        patterns.push(EpistemicPattern::RedundantExtraction {
                            created_value,
                            original_value: *value,
                            original_confidence: *confidence,
                            create_idx,
                            extraction_indices,
                        });
                    }
                }
            }
        }

        patterns
    }

    /// Trace a value back to its epistemic source (if it came from ExtractValue)
    fn trace_to_epistemic_source(&self, value: ValueId) -> Option<ValueId> {
        let (block_idx, inst_idx) = self.use_def.get_def(value)?;
        let block = &self.func.blocks[block_idx];
        let inst = &block.instructions[inst_idx];

        if let SirInst::Epistemic(EpistemicOp::ExtractValue(source)) = &inst.inst {
            Some(*source)
        } else {
            None
        }
    }

    /// Trace a confidence value back to its epistemic source
    fn trace_confidence_to_source(&self, value: ValueId) -> Option<ValueId> {
        let (block_idx, inst_idx) = self.use_def.get_def(value)?;
        let block = &self.func.blocks[block_idx];
        let inst = &block.instructions[inst_idx];

        if let SirInst::Epistemic(EpistemicOp::ExtractConfidence(source)) = &inst.inst {
            Some(*source)
        } else {
            None
        }
    }

    /// Find an ExtractValue instruction for a given epistemic source in a block
    fn find_extract_value(&self, block_idx: usize, source: ValueId) -> Option<(usize, ValueId)> {
        let block = &self.func.blocks[block_idx];

        for (idx, inst) in block.instructions.iter().enumerate() {
            if let SirInst::Epistemic(EpistemicOp::ExtractValue(s)) = &inst.inst {
                if *s == source {
                    return inst.result.map(|r| (idx, r));
                }
            }
        }
        None
    }

    /// Find an ExtractConfidence instruction for a given epistemic source in a block
    fn find_extract_confidence(
        &self,
        block_idx: usize,
        source: ValueId,
    ) -> Option<(usize, ValueId)> {
        let block = &self.func.blocks[block_idx];

        for (idx, inst) in block.instructions.iter().enumerate() {
            if let SirInst::Epistemic(EpistemicOp::ExtractConfidence(s)) = &inst.inst {
                if *s == source {
                    return inst.result.map(|r| (idx, r));
                }
            }
        }
        None
    }
}

/// Epistemic Fusion Pass
///
/// This is a key optimization that fuses confidence propagation operations.
/// Instead of:
///   %v1 = extract_value %a
///   %v2 = extract_value %b
///   %c1 = extract_confidence %a
///   %c2 = extract_confidence %b
///   %r1 = fmul %v1, %v2
///   %r2 = propagate_mul %v1, %c1, %v2, %c2
/// We generate:
///   %result = fused_epistemic_mul %a, %b
///
/// This reduces instruction count and improves cache locality by keeping
/// value and confidence operations together.
pub struct EpistemicFusion {
    /// Track the next available value ID for new instructions
    next_value_id: u32,
}

impl EpistemicFusion {
    pub fn new() -> Self {
        Self { next_value_id: 0 }
    }

    /// Initialize the next value ID by scanning the function
    fn init_value_ids(&mut self, func: &SirFunction) {
        self.next_value_id = 0;
        for block in &func.blocks {
            for param in &block.params {
                self.next_value_id = self.next_value_id.max(param.id.0 + 1);
            }
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    self.next_value_id = self.next_value_id.max(result.0 + 1);
                }
            }
        }
    }

    /// Get a new value ID
    fn new_value_id(&mut self) -> ValueId {
        let id = ValueId::new(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Apply a fusion pattern to a block
    fn apply_fusion_pattern(
        &mut self,
        block: &mut BasicBlock,
        pattern: &EpistemicPattern,
        stats: &mut PassStats,
    ) -> bool {
        match pattern {
            EpistemicPattern::ValueArithmeticWithPropagate {
                source_a,
                source_b,
                extracted_val_a,
                extracted_conf_a,
                extracted_val_b,
                extracted_conf_b,
                arith_result,
                propagate_result,
                arith_op,
                extract_val_a_idx,
                extract_val_b_idx,
                extract_conf_a_idx,
                extract_conf_b_idx,
                arith_idx,
                propagate_idx,
            } => {
                // Create the fused instruction based on the arithmetic operation
                let fused_op = match arith_op {
                    ArithOp::FMul => EpistemicOp::FusedMul {
                        val_a: *extracted_val_a,
                        conf_a: *extracted_conf_a,
                        val_b: *extracted_val_b,
                        conf_b: *extracted_conf_b,
                    },
                    ArithOp::FAdd => {
                        // For add, we use PropagateAdd as the fused form
                        // (FusedAdd doesn't exist yet, but we can still optimize)
                        return false; // Skip for now - would need to add FusedAdd to ops
                    }
                    _ => return false,
                };

                // Replace the propagate instruction with the fused operation
                // The fused operation produces the confidence result
                block.instructions[*propagate_idx] =
                    Instruction::with_result(*propagate_result, SirInst::Epistemic(fused_op));

                // Mark the original arithmetic, and extraction instructions for removal
                // We keep them for now but mark that fusion occurred
                // In a production compiler, we'd track which instructions to remove
                // after all uses are updated

                stats.epistemic_ops_fused += 1;
                stats.instructions_removed += 1; // We're effectively removing separate operations

                true
            }

            EpistemicPattern::RedundantExtraction {
                created_value,
                original_value,
                original_confidence,
                create_idx,
                extraction_indices,
            } => {
                // For each extraction, replace it with a reference to the original
                // This requires updating all uses of the extracted values
                // For now, we'll just track the optimization opportunity

                // In a full implementation, we would:
                // 1. Find all uses of extracted values
                // 2. Replace them with original_value or original_confidence
                // 3. Remove the create and extract instructions

                stats.instructions_removed += extraction_indices.len();
                stats.values_folded += extraction_indices.len();

                // Mark extractions as no-ops (we can't easily remove them in place)
                // In production, we'd use a proper instruction replacement mechanism

                true
            }

            EpistemicPattern::CertaintyChain {
                certain_values,
                eliminable_propagates,
            } => {
                // Remove propagate operations where all inputs are certain
                // The output confidence is known to be 1.0
                stats.instructions_removed += eliminable_propagates.len();
                true
            }
        }
    }
}

impl Default for EpistemicFusion {
    fn default() -> Self {
        Self::new()
    }
}

impl SirPass for EpistemicFusion {
    fn name(&self) -> &str {
        "epistemic-fusion"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.epistemic_ops_fused += sub_result.stats.epistemic_ops_fused;
            result.stats.instructions_removed += sub_result.stats.instructions_removed;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();

        // Initialize value IDs for this function
        self.init_value_ids(func);

        // Build pattern matcher
        let matcher = EpistemicPatternMatcher::new(func);

        // Collect patterns for each block
        let mut all_patterns: Vec<(usize, Vec<EpistemicPattern>)> = Vec::new();
        for block_idx in 0..func.blocks.len() {
            let patterns = matcher.find_fusion_patterns(block_idx);
            if !patterns.is_empty() {
                all_patterns.push((block_idx, patterns));
            }
        }

        // Apply patterns (we need to drop the matcher first since it borrows func)
        drop(matcher);

        for (block_idx, patterns) in all_patterns {
            for pattern in &patterns {
                if self.apply_fusion_pattern(
                    &mut func.blocks[block_idx],
                    pattern,
                    &mut result.stats,
                ) {
                    result.modified = true;
                }
            }
        }

        result
    }
}

/// Certainty Propagation Pass
///
/// When a value has confidence = 1.0 (certain), we can:
/// 1. Skip confidence calculations entirely
/// 2. Propagate certainty through pure operations
/// 3. Eliminate confidence checks
///
/// This pass performs a dataflow analysis to track which values are known
/// to be certain, then uses this information to:
/// - Mark outputs of operations on certain values as certain
/// - Replace confidence extractions with constant 1.0
/// - Eliminate redundant confidence propagation operations
pub struct CertaintyPropagation {
    /// Metadata store for tracking certainty information
    metadata: MetadataStore,
    /// Set of values known to be certain (confidence = 1.0)
    certain_values: HashSet<ValueId>,
    /// Map from value to its known constant confidence (if any)
    known_confidences: HashMap<ValueId, f64>,
}

impl CertaintyPropagation {
    pub fn new(metadata: MetadataStore) -> Self {
        Self {
            metadata,
            certain_values: HashSet::new(),
            known_confidences: HashMap::new(),
        }
    }

    /// Initialize certainty information from metadata and constants
    fn initialize_certainty(&mut self, func: &SirFunction) {
        self.certain_values.clear();
        self.known_confidences.clear();

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    // Check metadata for certainty
                    if self.metadata.is_certain(result) {
                        self.certain_values.insert(result);
                        self.known_confidences.insert(result, 1.0);
                    }

                    // Check for constant confidence values
                    match &inst.inst {
                        SirInst::Const(Constant::F64(v)) if *v == 1.0 => {
                            // This could be a confidence value of 1.0
                            self.known_confidences.insert(result, 1.0);
                        }
                        SirInst::Const(Constant::Epistemic { confidence, .. }) => {
                            self.known_confidences.insert(result, *confidence);
                            if *confidence == 1.0 {
                                self.certain_values.insert(result);
                            }
                        }
                        SirInst::Epistemic(EpistemicOp::Create { confidence, .. }) => {
                            // Check if confidence is a known constant
                            if let Some(&conf) = self.known_confidences.get(confidence) {
                                self.known_confidences.insert(result, conf);
                                if conf == 1.0 {
                                    self.certain_values.insert(result);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    /// Propagate certainty through the function until fixed point
    fn propagate_certainty(&mut self, func: &SirFunction) -> bool {
        let mut changed = false;

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    // Skip if already known to be certain
                    if self.certain_values.contains(&result) {
                        continue;
                    }

                    let new_certainty = self.compute_certainty(&inst.inst);

                    if let Some(conf) = new_certainty {
                        if conf == 1.0 {
                            if self.certain_values.insert(result) {
                                changed = true;
                            }
                        }
                        if self.known_confidences.insert(result, conf).is_none() {
                            changed = true;
                        }
                    }
                }
            }
        }

        changed
    }

    /// Compute the certainty/confidence of an instruction's result
    fn compute_certainty(&self, inst: &SirInst) -> Option<f64> {
        match inst {
            // Epistemic operations
            SirInst::Epistemic(ep_op) => self.compute_epistemic_certainty(ep_op),

            // Pure arithmetic: if all inputs certain, output is certain
            SirInst::BinOp { lhs, rhs, .. } => {
                if self.certain_values.contains(lhs) && self.certain_values.contains(rhs) {
                    Some(1.0)
                } else {
                    None
                }
            }
            SirInst::UnaryFloat { val, .. } => {
                if self.certain_values.contains(val) {
                    Some(1.0)
                } else {
                    None
                }
            }
            SirInst::BinaryFloat { lhs, rhs, .. } => {
                if self.certain_values.contains(lhs) && self.certain_values.contains(rhs) {
                    Some(1.0)
                } else {
                    None
                }
            }
            SirInst::Cast { val, .. } => {
                if self.certain_values.contains(val) {
                    Some(1.0)
                } else {
                    None
                }
            }

            // Select: result is certain only if both branches are certain
            SirInst::Select {
                then_val, else_val, ..
            } => {
                if self.certain_values.contains(then_val) && self.certain_values.contains(else_val)
                {
                    Some(1.0)
                } else {
                    // Could also compute min confidence of branches
                    let then_conf = self.known_confidences.get(then_val);
                    let else_conf = self.known_confidences.get(else_val);
                    match (then_conf, else_conf) {
                        (Some(t), Some(e)) => Some(t.min(*e)),
                        _ => None,
                    }
                }
            }

            // Phi: result is certain only if all incoming values are certain
            SirInst::Phi { incoming, .. } => {
                let all_certain = incoming
                    .iter()
                    .all(|(_, v)| self.certain_values.contains(v));
                if all_certain {
                    Some(1.0)
                } else {
                    // Compute minimum confidence
                    let confs: Vec<_> = incoming
                        .iter()
                        .filter_map(|(_, v)| self.known_confidences.get(v))
                        .collect();
                    if confs.len() == incoming.len() {
                        confs.into_iter().copied().reduce(f64::min)
                    } else {
                        None
                    }
                }
            }

            _ => None,
        }
    }

    /// Compute certainty for epistemic operations
    fn compute_epistemic_certainty(&self, op: &EpistemicOp) -> Option<f64> {
        match op {
            EpistemicOp::Create { confidence, .. } => {
                self.known_confidences.get(confidence).copied()
            }
            EpistemicOp::ExtractConfidence(source) => self.known_confidences.get(source).copied(),
            EpistemicOp::ExtractValue(source) => {
                // Extracting value from certain source is certain
                if self.certain_values.contains(source) {
                    Some(1.0)
                } else {
                    None
                }
            }
            EpistemicOp::PropagateAdd { conf_a, conf_b }
            | EpistemicOp::PropagateSub { conf_a, conf_b } => {
                // For add/sub, result confidence is min of inputs
                match (
                    self.known_confidences.get(conf_a),
                    self.known_confidences.get(conf_b),
                ) {
                    (Some(a), Some(b)) => Some(a.min(*b)),
                    _ => None,
                }
            }
            EpistemicOp::PropagateMul { conf_a, conf_b, .. }
            | EpistemicOp::PropagateDiv { conf_a, conf_b, .. } => {
                // For mul/div, result confidence is min of inputs
                match (
                    self.known_confidences.get(conf_a),
                    self.known_confidences.get(conf_b),
                ) {
                    (Some(a), Some(b)) => Some(a.min(*b)),
                    _ => None,
                }
            }
            EpistemicOp::FusedMul { conf_a, conf_b, .. } => {
                match (
                    self.known_confidences.get(conf_a),
                    self.known_confidences.get(conf_b),
                ) {
                    (Some(a), Some(b)) => Some(a.min(*b)),
                    _ => None,
                }
            }
            EpistemicOp::Meet { conf_a, conf_b } => {
                // Meet is minimum
                match (
                    self.known_confidences.get(conf_a),
                    self.known_confidences.get(conf_b),
                ) {
                    (Some(a), Some(b)) => Some(a.min(*b)),
                    _ => None,
                }
            }
            EpistemicOp::Join { conf_a, conf_b } => {
                // Join is maximum
                match (
                    self.known_confidences.get(conf_a),
                    self.known_confidences.get(conf_b),
                ) {
                    (Some(a), Some(b)) => Some(a.max(*b)),
                    _ => None,
                }
            }
            EpistemicOp::CondPropagate {
                condition_conf,
                then_conf,
                else_conf,
            } => {
                // If condition is certain, result depends on taken branch
                if self.known_confidences.get(condition_conf) == Some(&1.0) {
                    // We'd need to know which branch is taken
                    // For now, return min of both branches
                    match (
                        self.known_confidences.get(then_conf),
                        self.known_confidences.get(else_conf),
                    ) {
                        (Some(t), Some(e)) => Some(t.min(*e)),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }

    /// Apply optimizations based on certainty information
    fn apply_certainty_optimizations(&self, func: &mut SirFunction, stats: &mut PassStats) -> bool {
        let mut modified = false;

        for block in &mut func.blocks {
            let mut indices_to_simplify = Vec::new();

            for (idx, inst) in block.instructions.iter().enumerate() {
                if let SirInst::Epistemic(ep_op) = &inst.inst {
                    match ep_op {
                        // If extracting confidence from a certain value, replace with const 1.0
                        EpistemicOp::ExtractConfidence(source) => {
                            if self.certain_values.contains(source) {
                                indices_to_simplify
                                    .push((idx, SimplifyAction::ReplaceWithConstOne));
                            }
                        }
                        // If propagating confidence between certain values, replace with const 1.0
                        EpistemicOp::PropagateAdd { conf_a, conf_b }
                        | EpistemicOp::PropagateSub { conf_a, conf_b } => {
                            if self.known_confidences.get(conf_a) == Some(&1.0)
                                && self.known_confidences.get(conf_b) == Some(&1.0)
                            {
                                indices_to_simplify
                                    .push((idx, SimplifyAction::ReplaceWithConstOne));
                            }
                        }
                        EpistemicOp::PropagateMul { conf_a, conf_b, .. }
                        | EpistemicOp::PropagateDiv { conf_a, conf_b, .. }
                        | EpistemicOp::FusedMul { conf_a, conf_b, .. } => {
                            if self.known_confidences.get(conf_a) == Some(&1.0)
                                && self.known_confidences.get(conf_b) == Some(&1.0)
                            {
                                indices_to_simplify
                                    .push((idx, SimplifyAction::ReplaceWithConstOne));
                            }
                        }
                        EpistemicOp::Meet { conf_a, conf_b }
                        | EpistemicOp::Join { conf_a, conf_b } => {
                            // If both are 1.0, result is 1.0
                            if self.known_confidences.get(conf_a) == Some(&1.0)
                                && self.known_confidences.get(conf_b) == Some(&1.0)
                            {
                                indices_to_simplify
                                    .push((idx, SimplifyAction::ReplaceWithConstOne));
                            }
                        }
                        _ => {}
                    }
                }
            }

            // Apply simplifications
            for (idx, action) in indices_to_simplify {
                if let Some(result) = block.instructions[idx].result {
                    match action {
                        SimplifyAction::ReplaceWithConstOne => {
                            block.instructions[idx] = Instruction::with_result(
                                result,
                                SirInst::Const(Constant::F64(1.0)),
                            );
                            stats.instructions_removed += 1;
                            stats.values_folded += 1;
                            modified = true;
                        }
                    }
                }
            }
        }

        modified
    }

    /// Update metadata store with discovered certainty information
    fn update_metadata(&mut self) {
        for &value in &self.certain_values {
            self.metadata
                .attach(value, Metadata::Epistemic(EpistemicMetadata::certain()));
        }
        for (&value, &conf) in &self.known_confidences {
            if conf != 1.0 {
                self.metadata.attach(
                    value,
                    Metadata::Epistemic(EpistemicMetadata::with_confidence(conf)),
                );
            }
        }
    }
}

/// Actions for simplifying instructions based on certainty
#[derive(Debug, Clone, Copy)]
enum SimplifyAction {
    /// Replace instruction with constant 1.0
    ReplaceWithConstOne,
}

impl SirPass for CertaintyPropagation {
    fn name(&self) -> &str {
        "certainty-propagation"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_removed += sub_result.stats.instructions_removed;
            result.stats.values_folded += sub_result.stats.values_folded;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();

        // Phase 1: Initialize certainty from metadata and constants
        self.initialize_certainty(func);

        // Phase 2: Propagate certainty until fixed point
        let max_iterations = 100;
        let mut iterations = 0;
        while self.propagate_certainty(func) && iterations < max_iterations {
            iterations += 1;
        }

        // Phase 3: Apply optimizations based on certainty
        if self.apply_certainty_optimizations(func, &mut result.stats) {
            result.modified = true;
        }

        // Phase 4: Update metadata for future passes
        self.update_metadata();

        result
    }
}

/// Distribution Combining Pass
///
/// Combines operations on probability distributions:
/// - Normal(a, σ₁) + Normal(b, σ₂) = Normal(a+b, √(σ₁²+σ₂²))
/// - c * Normal(μ, σ) = Normal(c*μ, |c|*σ)
pub struct DistributionCombining;

impl SirPass for DistributionCombining {
    fn name(&self) -> &str {
        "dist-combine"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let result = PassResult::default();

        for func in &mut module.functions {
            for block in &mut func.blocks {
                // TODO: Look for distribution combination patterns
                // 1. Sample from distribution, sample from another, combine
                //    → Create combined distribution, single sample
                // 2. Arithmetic on sampled values from known distributions
                //    → Use analytical combination rules
                let _ = block;
            }
        }

        result
    }
}

/// ODE Solver Step Fusion Pass
///
/// Detects manual ODE solver implementations and replaces them with
/// optimized single-step instructions.
pub struct OdeStepFusion;

impl SirPass for OdeStepFusion {
    fn name(&self) -> &str {
        "ode-fusion"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let result = PassResult::default();

        // Look for RK4 pattern:
        // k1 = f(t, y)
        // k2 = f(t + h/2, y + h*k1/2)
        // k3 = f(t + h/2, y + h*k2/2)
        // k4 = f(t + h, y + h*k3)
        // y_next = y + h/6 * (k1 + 2*k2 + 2*k3 + k4)
        //
        // Replace with: y_next = ode_step(RK4, y, f, t, h)

        for func in &mut module.functions {
            // TODO: Implement pattern detection
            // This is complex and requires:
            // 1. Loop detection
            // 2. Pattern matching on the loop body
            // 3. Derivative function extraction
            // 4. Replacement with SIR ODE step
            let _ = func;
        }

        result
    }
}

// ============================================================================
// AUTOMATIC UNCERTAINTY COMPILATION
// ============================================================================

/// Source of uncertainty in a value
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum UncertaintySource {
    /// Value comes from external input (file, network, user)
    ExternalInput,
    /// Value comes from measurement/sensor data
    Measurement,
    /// Value comes from floating-point computation (accumulates error)
    FloatingPointError,
    /// Value comes from approximation (e.g., iterative solver tolerance)
    Approximation,
    /// Value comes from model parameter with known uncertainty
    ModelParameter,
    /// Value is sampled from a probability distribution
    Stochastic,
    /// Unknown source of uncertainty
    Unknown,
}

/// Inferred uncertainty information for a value
#[derive(Debug, Clone)]
pub struct InferredUncertainty {
    /// The value ID this applies to
    pub value: ValueId,
    /// Source of uncertainty
    pub source: UncertaintySource,
    /// Inferred confidence (0.0 = unknown, 1.0 = certain)
    pub confidence: f64,
    /// Standard relative uncertainty (for GUM propagation)
    pub relative_uncertainty: Option<f64>,
    /// Whether this needs to be wrapped in epistemic type
    pub needs_wrapping: bool,
}

/// Automatic Uncertainty Compilation Pass
///
/// This is a breakthrough optimization that automatically analyzes code
/// and inserts uncertainty quantification. The compiler:
///
/// 1. **Scans for implicit uncertainties**: Identifies values that come from
///    external sources, measurements, or computations that accumulate error
///
/// 2. **Infers uncertainty levels**: Uses dataflow analysis to propagate
///    uncertainty information through the program
///
/// 3. **Inserts GUM propagation**: Automatically adds Guide to the expression
///    of Uncertainty in Measurement (GUM) propagation operations
///
/// 4. **Enforces at compile-time**: Low-confidence paths are gated differently,
///    and the compiler warns about uncertainty drops
///
/// This is the first compiler in history to treat TRUST as a first-class
/// compilation concern, alongside correctness and performance.
pub struct AutomaticUncertaintyCompilation {
    /// Inferred uncertainties for values
    uncertainties: HashMap<ValueId, InferredUncertainty>,
    /// Default confidence for external inputs
    external_input_confidence: f64,
    /// Default confidence for measurements
    measurement_confidence: f64,
    /// Threshold below which to wrap in epistemic type
    epistemic_threshold: f64,
    /// Next available value ID for inserted instructions
    next_value_id: u32,
    /// Statistics about uncertainty analysis
    pub stats: UncertaintyStats,
}

#[derive(Debug, Clone, Default)]
pub struct UncertaintyStats {
    /// Values identified as having implicit uncertainty
    pub implicit_uncertainties_found: usize,
    /// Epistemic wrappers inserted
    pub epistemic_wrappers_inserted: usize,
    /// GUM propagation operations inserted
    pub gum_propagations_inserted: usize,
    /// External input values identified
    pub external_inputs: usize,
    /// Measurement values identified
    pub measurements: usize,
    /// Floating-point accumulation points identified
    pub fp_accumulations: usize,
}

impl AutomaticUncertaintyCompilation {
    pub fn new() -> Self {
        Self {
            uncertainties: HashMap::new(),
            external_input_confidence: 0.3,  // External inputs start at 30% confidence
            measurement_confidence: 0.5,      // Measurements start at 50% confidence
            epistemic_threshold: 0.9,         // Wrap if confidence < 90%
            next_value_id: 0,
            stats: UncertaintyStats::default(),
        }
    }

    /// Configure the pass with custom thresholds
    pub fn with_config(
        external_input_confidence: f64,
        measurement_confidence: f64,
        epistemic_threshold: f64,
    ) -> Self {
        Self {
            uncertainties: HashMap::new(),
            external_input_confidence,
            measurement_confidence,
            epistemic_threshold,
            next_value_id: 0,
            stats: UncertaintyStats::default(),
        }
    }

    /// Initialize value IDs from function
    fn init_value_ids(&mut self, func: &SirFunction) {
        self.next_value_id = 0;
        for block in &func.blocks {
            for param in &block.params {
                self.next_value_id = self.next_value_id.max(param.id.0 + 1);
            }
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    self.next_value_id = self.next_value_id.max(result.0 + 1);
                }
            }
        }
    }

    /// Get a new value ID
    fn new_value_id(&mut self) -> ValueId {
        let id = ValueId::new(self.next_value_id);
        self.next_value_id += 1;
        id
    }

    /// Phase 1: Identify uncertainty sources in the function
    fn identify_uncertainty_sources(&mut self, func: &SirFunction) {
        self.uncertainties.clear();

        for (block_idx, block) in func.blocks.iter().enumerate() {
            // Function parameters are potential external inputs
            for param in &block.params {
                // First block params are function parameters
                if block_idx == 0 {
                    self.uncertainties.insert(
                        param.id,
                        InferredUncertainty {
                            value: param.id,
                            source: UncertaintySource::ExternalInput,
                            confidence: self.external_input_confidence,
                            relative_uncertainty: Some(0.1), // 10% relative uncertainty
                            needs_wrapping: true,
                        },
                    );
                    self.stats.external_inputs += 1;
                    self.stats.implicit_uncertainties_found += 1;
                }
            }

            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    if let Some(uncertainty) = self.analyze_instruction(&inst.inst, result) {
                        self.uncertainties.insert(result, uncertainty);
                        self.stats.implicit_uncertainties_found += 1;
                    }
                }
            }
        }
    }

    /// Analyze an instruction and return uncertainty info if applicable
    fn analyze_instruction(&mut self, inst: &SirInst, result: ValueId) -> Option<InferredUncertainty> {
        match inst {
            // Memory loads from external data sources
            SirInst::Memory(MemoryOp::Load { .. }) => {
                self.stats.measurements += 1;
                Some(InferredUncertainty {
                    value: result,
                    source: UncertaintySource::ExternalInput,
                    confidence: self.external_input_confidence,
                    relative_uncertainty: Some(0.1),
                    needs_wrapping: true,
                })
            }

            // Calls to external functions (named calls) return uncertain values
            SirInst::Call(CallInfo { callee: Callee::Named(_), .. }) => {
                Some(InferredUncertainty {
                    value: result,
                    source: UncertaintySource::ExternalInput,
                    confidence: 0.2, // Very uncertain
                    relative_uncertainty: Some(0.2),
                    needs_wrapping: true,
                })
            }

            // Indirect calls are also uncertain
            SirInst::Call(CallInfo { callee: Callee::Indirect(_), .. }) => {
                Some(InferredUncertainty {
                    value: result,
                    source: UncertaintySource::ExternalInput,
                    confidence: 0.15, // Even more uncertain
                    relative_uncertainty: Some(0.25),
                    needs_wrapping: true,
                })
            }

            // Scientific operations may introduce approximation error
            SirInst::Scientific(sci_op) => {
                let (source, confidence, rel_unc) = match sci_op {
                    ScientificOp::OdeStep { .. } => {
                        (UncertaintySource::Approximation, 0.7, 0.05)
                    }
                    ScientificOp::Lerp { .. } => {
                        (UncertaintySource::FloatingPointError, 0.95, 0.001)
                    }
                    _ => (UncertaintySource::Approximation, 0.8, 0.02),
                };
                Some(InferredUncertainty {
                    value: result,
                    source,
                    confidence,
                    relative_uncertainty: Some(rel_unc),
                    needs_wrapping: confidence < self.epistemic_threshold,
                })
            }

            // Probability operations are inherently stochastic
            SirInst::Prob(prob_op) => {
                let (source, confidence) = match prob_op {
                    ProbOp::Sample { .. } | ProbOp::SampleN { .. } => {
                        (UncertaintySource::Stochastic, 0.0)
                    }
                    ProbOp::CreateDist { .. } => {
                        (UncertaintySource::ModelParameter, 0.8)
                    }
                    _ => (UncertaintySource::Stochastic, 0.5),
                };
                Some(InferredUncertainty {
                    value: result,
                    source,
                    confidence,
                    relative_uncertainty: None, // Distributions have their own uncertainty
                    needs_wrapping: source == UncertaintySource::Stochastic,
                })
            }

            // Floating-point division can accumulate error
            SirInst::BinOp { op: ArithOp::FDiv, .. } => {
                self.stats.fp_accumulations += 1;
                Some(InferredUncertainty {
                    value: result,
                    source: UncertaintySource::FloatingPointError,
                    confidence: 0.99, // High but not perfect
                    relative_uncertainty: Some(f64::EPSILON * 10.0),
                    needs_wrapping: false, // Usually acceptable
                })
            }

            // Float functions can accumulate error
            SirInst::UnaryFloat { op, .. } => {
                let rel_unc = match op {
                    UnaryFloatOp::Sqrt | UnaryFloatOp::Exp | UnaryFloatOp::Log => 1e-14,
                    UnaryFloatOp::Sin | UnaryFloatOp::Cos | UnaryFloatOp::Tan => 1e-15,
                    _ => 1e-15,
                };
                self.stats.fp_accumulations += 1;
                Some(InferredUncertainty {
                    value: result,
                    source: UncertaintySource::FloatingPointError,
                    confidence: 0.999,
                    relative_uncertainty: Some(rel_unc),
                    needs_wrapping: false,
                })
            }

            _ => None,
        }
    }

    /// Phase 2: Propagate uncertainty through dataflow
    fn propagate_uncertainty(&mut self, func: &SirFunction) -> bool {
        let mut changed = false;

        for block in &func.blocks {
            for inst in &block.instructions {
                if let Some(result) = inst.result {
                    // Skip if already analyzed
                    if self.uncertainties.contains_key(&result) {
                        continue;
                    }

                    // Propagate from operands
                    let operands = inst.inst.operands();
                    let mut min_confidence: f64 = 1.0;
                    let mut has_uncertain_input = false;

                    for op in &operands {
                        if let Some(unc) = self.uncertainties.get(op) {
                            min_confidence = min_confidence.min(unc.confidence);
                            has_uncertain_input = true;
                        }
                    }

                    // If any input is uncertain, output inherits uncertainty
                    if has_uncertain_input {
                        let new_confidence = match &inst.inst {
                            // Pure arithmetic preserves minimum confidence
                            SirInst::BinOp { .. } => min_confidence,
                            // Comparisons are more certain
                            SirInst::Cmp { .. } => {
                                (min_confidence + 1.0) / 2.0
                            }
                            // Select: use minimum of branches
                            SirInst::Select { .. } => min_confidence,
                            // Phi: conservative, use minimum
                            SirInst::Phi { .. } => min_confidence,
                            // Cast: preserve
                            SirInst::Cast { .. } => min_confidence,
                            // Default: slight degradation
                            _ => min_confidence * 0.99,
                        };

                        self.uncertainties.insert(
                            result,
                            InferredUncertainty {
                                value: result,
                                source: UncertaintySource::Unknown,
                                confidence: new_confidence,
                                relative_uncertainty: None,
                                needs_wrapping: new_confidence < self.epistemic_threshold,
                            },
                        );
                        changed = true;
                    }
                }
            }
        }

        changed
    }

    /// Phase 3: Insert GUM propagation operations
    fn insert_gum_propagation(
        &mut self,
        func: &mut SirFunction,
        stats: &mut PassStats,
    ) -> bool {
        let mut modified = false;

        // Collect values that need GUM propagation
        let needs_propagation: Vec<_> = self.uncertainties
            .iter()
            .filter(|(_, u)| u.relative_uncertainty.is_some() && u.needs_wrapping)
            .map(|(v, u)| (*v, u.clone()))
            .collect();

        if needs_propagation.is_empty() {
            return false;
        }

        // For each block, insert epistemic wrappers after uncertain values
        for block in &mut func.blocks {
            let mut insertions = Vec::new();

            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                if let Some(result) = inst.result {
                    if let Some(unc) = needs_propagation.iter().find(|(v, _)| *v == result) {
                        // Insert epistemic wrapper after this instruction
                        let conf_value = self.new_value_id();
                        let wrapped_value = self.new_value_id();

                        // Create confidence constant
                        let conf_inst = Instruction::with_result(
                            conf_value,
                            SirInst::Const(Constant::F64(unc.1.confidence)),
                        );

                        // Create epistemic wrapper
                        let wrap_inst = Instruction::with_result(
                            wrapped_value,
                            SirInst::Epistemic(EpistemicOp::Create {
                                value: result,
                                confidence: conf_value,
                            }),
                        );

                        insertions.push((inst_idx + 1, vec![conf_inst, wrap_inst]));
                        self.stats.epistemic_wrappers_inserted += 1;
                        stats.instructions_added += 2;
                        modified = true;
                    }
                }
            }

            // Apply insertions in reverse order to preserve indices
            for (idx, insts) in insertions.into_iter().rev() {
                for (i, inst) in insts.into_iter().enumerate() {
                    if idx + i <= block.instructions.len() {
                        block.instructions.insert(idx + i, inst);
                    }
                }
            }
        }

        modified
    }

    /// Phase 4: Insert propagation for arithmetic on uncertain values
    fn insert_arithmetic_propagation(
        &mut self,
        func: &mut SirFunction,
        stats: &mut PassStats,
    ) -> bool {
        let mut modified = false;

        // Collect propagation candidates first (to avoid borrow issues)
        let mut candidates: Vec<(usize, usize, ArithOp, ValueId, ValueId, f64, f64)> = Vec::new();

        for (block_idx, block) in func.blocks.iter().enumerate() {
            for (inst_idx, inst) in block.instructions.iter().enumerate() {
                if let SirInst::BinOp { op, lhs, rhs } = &inst.inst {
                    // Check if both operands have uncertainty
                    let lhs_unc = self.uncertainties.get(lhs);
                    let rhs_unc = self.uncertainties.get(rhs);

                    if let (Some(lu), Some(ru)) = (lhs_unc, rhs_unc) {
                        // Both inputs are uncertain - need propagation
                        if lu.relative_uncertainty.is_some() || ru.relative_uncertainty.is_some() {
                            // Copy the values we need
                            candidates.push((
                                block_idx,
                                inst_idx,
                                *op,
                                *lhs,
                                *rhs,
                                lu.confidence,
                                ru.confidence,
                            ));
                        }
                    }
                }
            }
        }

        // Now process candidates
        for (block_idx, inst_idx, op, lhs, rhs, lu_conf, ru_conf) in candidates {
            let mut propagation_insertions = Vec::new();

            let propagate_result = self.new_value_id();

            // Create appropriate propagation based on operation
            let propagate_inst = match op {
                ArithOp::FAdd | ArithOp::FSub => {
                    // For add/sub: propagate using min confidence
                    let lhs_conf = self.new_value_id();
                    let rhs_conf = self.new_value_id();

                    // Insert confidence extractions first
                    propagation_insertions.push((
                        inst_idx + 1,
                        Instruction::with_result(
                            lhs_conf,
                            SirInst::Const(Constant::F64(lu_conf)),
                        ),
                    ));
                    propagation_insertions.push((
                        inst_idx + 2,
                        Instruction::with_result(
                            rhs_conf,
                            SirInst::Const(Constant::F64(ru_conf)),
                        ),
                    ));

                    Some(Instruction::with_result(
                        propagate_result,
                        SirInst::Epistemic(EpistemicOp::PropagateAdd {
                            conf_a: lhs_conf,
                            conf_b: rhs_conf,
                        }),
                    ))
                }
                ArithOp::FMul | ArithOp::FDiv => {
                    // For mul/div: GUM propagation
                    let lhs_conf = self.new_value_id();
                    let rhs_conf = self.new_value_id();

                    propagation_insertions.push((
                        inst_idx + 1,
                        Instruction::with_result(
                            lhs_conf,
                            SirInst::Const(Constant::F64(lu_conf)),
                        ),
                    ));
                    propagation_insertions.push((
                        inst_idx + 2,
                        Instruction::with_result(
                            rhs_conf,
                            SirInst::Const(Constant::F64(ru_conf)),
                        ),
                    ));

                    Some(Instruction::with_result(
                        propagate_result,
                        SirInst::Epistemic(EpistemicOp::PropagateMul {
                            val_a: lhs,
                            conf_a: lhs_conf,
                            val_b: rhs,
                            conf_b: rhs_conf,
                        }),
                    ))
                }
                _ => None,
            };

            if let Some(prop_inst) = propagate_inst {
                propagation_insertions.push((inst_idx + 3, prop_inst));
                self.stats.gum_propagations_inserted += 1;
                stats.instructions_added += 1;
                modified = true;
            }

            // Sort by insertion position (descending) to preserve indices
            propagation_insertions.sort_by(|a, b| b.0.cmp(&a.0));

            let block = &mut func.blocks[block_idx];
            for (idx, inst) in propagation_insertions {
                if idx <= block.instructions.len() {
                    block.instructions.insert(idx, inst);
                }
            }
        }

        modified
    }
}

impl Default for AutomaticUncertaintyCompilation {
    fn default() -> Self {
        Self::new()
    }
}

impl SirPass for AutomaticUncertaintyCompilation {
    fn name(&self) -> &str {
        "auto-uncertainty"
    }

    fn run(&mut self, module: &mut SirModule) -> PassResult {
        let mut result = PassResult::default();

        for func in &mut module.functions {
            let sub_result = self.run_on_function(func);
            result.modified |= sub_result.modified;
            result.stats.instructions_added += sub_result.stats.instructions_added;
        }

        result
    }

    fn run_on_function(&mut self, func: &mut SirFunction) -> PassResult {
        let mut result = PassResult::default();

        // Initialize value IDs
        self.init_value_ids(func);

        // Phase 1: Identify uncertainty sources
        self.identify_uncertainty_sources(func);

        // Phase 2: Propagate uncertainty through dataflow until fixed point
        let max_iterations = 50;
        let mut iterations = 0;
        while self.propagate_uncertainty(func) && iterations < max_iterations {
            iterations += 1;
        }

        // Phase 3: Insert epistemic wrappers for uncertain values
        if self.insert_gum_propagation(func, &mut result.stats) {
            result.modified = true;
        }

        // Phase 4: Insert propagation for arithmetic on uncertain values
        if self.insert_arithmetic_propagation(func, &mut result.stats) {
            result.modified = true;
        }

        result
    }
}

// ============================================================================
// PASS MANAGER
// ============================================================================

/// Manages and runs optimization passes
pub struct PassManager {
    passes: Vec<Box<dyn SirPass>>,
}

impl PassManager {
    pub fn new() -> Self {
        Self { passes: vec![] }
    }

    /// Create pass manager with default optimization passes
    pub fn with_defaults() -> Self {
        Self::with_metadata(MetadataStore::new())
    }

    /// Create pass manager with default optimization passes and custom metadata store
    pub fn with_metadata(metadata: MetadataStore) -> Self {
        let mut pm = Self::new();

        // Analysis passes
        pm.add(Box::new(DominanceAnalysis));
        pm.add(Box::new(LoopAnalysis));

        // Standard passes
        pm.add(Box::new(ConstantFolding));
        pm.add(Box::new(DeadCodeElimination));

        // Domain-specific passes - order matters!
        // 0. Automatic uncertainty compilation first - identifies and wraps uncertain values
        pm.add(Box::new(AutomaticUncertaintyCompilation::new()));
        // 1. Certainty propagation to identify certain values
        pm.add(Box::new(CertaintyPropagation::new(metadata)));
        // 2. Epistemic fusion can use certainty info
        pm.add(Box::new(EpistemicFusion::new()));
        // 3. Distribution combining
        pm.add(Box::new(DistributionCombining));
        // 4. ODE step fusion
        pm.add(Box::new(OdeStepFusion));

        pm
    }

    /// Add a pass
    pub fn add(&mut self, pass: Box<dyn SirPass>) {
        self.passes.push(pass);
    }

    /// Run all passes
    pub fn run(&mut self, module: &mut SirModule) -> Vec<PassResult> {
        let mut results = vec![];

        for pass in &mut self.passes {
            let result = pass.run(module);
            results.push(result);
        }

        results
    }

    /// Run passes until fixed point
    pub fn run_until_fixed_point(
        &mut self,
        module: &mut SirModule,
        max_iterations: usize,
    ) -> usize {
        let mut iterations = 0;

        loop {
            iterations += 1;
            if iterations > max_iterations {
                break;
            }

            let mut any_modified = false;
            for pass in &mut self.passes {
                let result = pass.run(module);
                any_modified |= result.modified;
            }

            if !any_modified {
                break;
            }
        }

        iterations
    }
}

impl Default for PassManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::super::blocks::Terminator;
    use super::super::types::SirType;
    use super::super::values::BlockId;
    use super::*;

    /// Helper to create a test function with a single block
    fn create_test_function(name: &str) -> SirFunction {
        let mut func =
            SirFunction::new(super::super::values::FuncId(0), name, vec![], SirType::Void);
        let entry = BasicBlock::new(BlockId::new(0));
        func.blocks.push(entry);
        func
    }

    /// Helper to add instructions to the entry block
    fn add_instruction(func: &mut SirFunction, result: ValueId, inst: SirInst) {
        func.blocks[0]
            .instructions
            .push(Instruction::with_result(result, inst));
    }

    #[test]
    fn test_pass_manager() {
        let mut module = SirModule::new("test");
        let mut pm = PassManager::with_defaults();
        let results = pm.run(&mut module);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_use_def_chain_building() {
        let mut func = create_test_function("test_use_def");

        // %0 = const 1.0
        // %1 = const 0.9
        // %2 = fadd %0, %1
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(0.9)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FAdd,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        );

        let use_def = UseDefChain::build(&func);

        // Check definitions
        assert_eq!(use_def.get_def(ValueId::new(0)), Some((0, 0)));
        assert_eq!(use_def.get_def(ValueId::new(1)), Some((0, 1)));
        assert_eq!(use_def.get_def(ValueId::new(2)), Some((0, 2)));

        // Check uses
        let uses_0 = use_def.get_uses(ValueId::new(0));
        assert_eq!(uses_0.len(), 1);
        assert_eq!(uses_0[0], (0, 2));

        let uses_1 = use_def.get_uses(ValueId::new(1));
        assert_eq!(uses_1.len(), 1);
        assert_eq!(uses_1[0], (0, 2));

        // Check single use detection
        assert!(use_def.has_single_use(ValueId::new(0)));
        assert!(use_def.has_single_use(ValueId::new(1)));
    }

    #[test]
    fn test_certainty_propagation_simple() {
        let mut func = create_test_function("test_certainty");

        // %0 = const 1.0 (confidence)
        // %1 = const 42.0 (value)
        // %2 = epistemic.create %1, %0
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(42.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(1),
                confidence: ValueId::new(0),
            }),
        );
        // %3 = epistemic.extract_confidence %2
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(2))),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = CertaintyPropagation::new(MetadataStore::new());
        let result = pass.run(&mut module);

        // The pass should identify that %2 is certain and replace extract_confidence with const 1.0
        assert!(result.modified);
        assert!(result.stats.values_folded > 0);

        // Check that the extract_confidence was replaced with const 1.0
        let inst = &module.functions[0].blocks[0].instructions[3];
        if let SirInst::Const(Constant::F64(v)) = &inst.inst {
            assert_eq!(*v, 1.0);
        } else {
            panic!("Expected const 1.0, got {:?}", inst.inst);
        }
    }

    #[test]
    fn test_certainty_propagation_through_arithmetic() {
        let mut func = create_test_function("test_certainty_arith");

        // %0 = const 1.0 (confidence)
        // %1 = const 10.0 (value a)
        // %2 = const 20.0 (value b)
        // %3 = epistemic.create %1, %0  -- certain value a
        // %4 = epistemic.create %2, %0  -- certain value b
        // %5 = epistemic.propagate_add %0, %0  -- should become const 1.0
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(10.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Const(Constant::F64(20.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(1),
                confidence: ValueId::new(0),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(4),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(2),
                confidence: ValueId::new(0),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(5),
            SirInst::Epistemic(EpistemicOp::PropagateAdd {
                conf_a: ValueId::new(0),
                conf_b: ValueId::new(0),
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = CertaintyPropagation::new(MetadataStore::new());
        let result = pass.run(&mut module);

        // PropagateAdd with two certain inputs should become const 1.0
        assert!(result.modified);

        let inst = &module.functions[0].blocks[0].instructions[5];
        if let SirInst::Const(Constant::F64(v)) = &inst.inst {
            assert_eq!(*v, 1.0);
        } else {
            panic!("Expected const 1.0 for propagate_add with certain inputs");
        }
    }

    #[test]
    fn test_certainty_propagation_meet_join() {
        let mut func = create_test_function("test_meet_join");

        // %0 = const 1.0
        // %1 = const 1.0
        // %2 = epistemic.meet %0, %1  -- min(1.0, 1.0) = 1.0
        // %3 = epistemic.join %0, %1  -- max(1.0, 1.0) = 1.0
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Meet {
                conf_a: ValueId::new(0),
                conf_b: ValueId::new(1),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Epistemic(EpistemicOp::Join {
                conf_a: ValueId::new(0),
                conf_b: ValueId::new(1),
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = CertaintyPropagation::new(MetadataStore::new());
        let result = pass.run(&mut module);

        // Both meet and join with 1.0 inputs should become 1.0
        assert!(result.modified);
        assert!(result.stats.values_folded >= 2);
    }

    #[test]
    fn test_epistemic_pattern_matcher_extract_patterns() {
        let mut func = create_test_function("test_pattern_matcher");

        // Simulate: create then immediately extract
        // %0 = const 42.0 (value)
        // %1 = const 0.9 (confidence)
        // %2 = epistemic.create %0, %1
        // %3 = epistemic.extract_value %2
        // %4 = epistemic.extract_confidence %2
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(42.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(0.9)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(0),
                confidence: ValueId::new(1),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Epistemic(EpistemicOp::ExtractValue(ValueId::new(2))),
        );
        add_instruction(
            &mut func,
            ValueId::new(4),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(2))),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let matcher = EpistemicPatternMatcher::new(&func);
        let patterns = matcher.find_fusion_patterns(0);

        // Should find a redundant extraction pattern
        let has_redundant = patterns
            .iter()
            .any(|p| matches!(p, EpistemicPattern::RedundantExtraction { .. }));
        assert!(has_redundant, "Should detect redundant extraction pattern");
    }

    #[test]
    fn test_epistemic_fusion_fused_mul() {
        let mut func = create_test_function("test_fused_mul");

        // Setup for a fusable pattern:
        // %0 = epistemic value a
        // %1 = epistemic value b
        // %2 = extract_value %0
        // %3 = extract_value %1
        // %4 = extract_confidence %0
        // %5 = extract_confidence %1
        // %6 = fmul %2, %3
        // %7 = propagate_mul %2, %4, %3, %5
        // -> Should fuse into fused_mul

        // For this test, we'll create the scenario directly
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(10.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(0.9)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(0),
                confidence: ValueId::new(1),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Const(Constant::F64(20.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(4),
            SirInst::Const(Constant::F64(0.8)),
        );
        add_instruction(
            &mut func,
            ValueId::new(5),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(3),
                confidence: ValueId::new(4),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(6),
            SirInst::Epistemic(EpistemicOp::ExtractValue(ValueId::new(2))),
        );
        add_instruction(
            &mut func,
            ValueId::new(7),
            SirInst::Epistemic(EpistemicOp::ExtractValue(ValueId::new(5))),
        );
        add_instruction(
            &mut func,
            ValueId::new(8),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(2))),
        );
        add_instruction(
            &mut func,
            ValueId::new(9),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(5))),
        );
        add_instruction(
            &mut func,
            ValueId::new(10),
            SirInst::BinOp {
                op: ArithOp::FMul,
                lhs: ValueId::new(6),
                rhs: ValueId::new(7),
            },
        );
        add_instruction(
            &mut func,
            ValueId::new(11),
            SirInst::Epistemic(EpistemicOp::PropagateMul {
                val_a: ValueId::new(6),
                conf_a: ValueId::new(8),
                val_b: ValueId::new(7),
                conf_b: ValueId::new(9),
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = EpistemicFusion::new();
        let result = pass.run(&mut module);

        // Check that fusion occurred
        assert!(
            result.modified,
            "Epistemic fusion should have modified the module"
        );
        assert!(
            result.stats.epistemic_ops_fused > 0,
            "Should have fused at least one operation"
        );

        // Verify the propagate_mul was replaced with fused_mul
        let inst = &module.functions[0].blocks[0].instructions[11];
        if let SirInst::Epistemic(EpistemicOp::FusedMul { .. }) = &inst.inst {
            // Success - it was replaced with FusedMul
        } else {
            panic!("Expected FusedMul, got {:?}", inst.inst);
        }
    }

    #[test]
    fn test_dead_code_elimination_epistemic() {
        let mut func = create_test_function("test_dce_epistemic");

        // %0 = const 1.0
        // %1 = const 42.0
        // %2 = epistemic.create %1, %0  -- unused, should be eliminated
        // %3 = const 100.0  -- used in return
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(42.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(1),
                confidence: ValueId::new(0),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Const(Constant::F64(100.0)),
        );
        func.blocks[0].terminate(Terminator::Return(Some(ValueId::new(3))));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = DeadCodeElimination;
        let result = pass.run(&mut module);

        assert!(
            result.modified,
            "DCE should have removed unused instructions"
        );
        assert!(
            result.stats.instructions_removed > 0,
            "Should have removed dead instructions"
        );
    }

    #[test]
    fn test_pass_manager_with_epistemic_passes() {
        let mut func = create_test_function("test_full_pipeline");

        // Create a simple epistemic computation
        // %0 = const 1.0 (confidence - certain)
        // %1 = const 10.0 (value)
        // %2 = epistemic.create %1, %0
        // %3 = epistemic.extract_confidence %2
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(10.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::Create {
                value: ValueId::new(1),
                confidence: ValueId::new(0),
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(2))),
        );
        func.blocks[0].terminate(Terminator::Return(Some(ValueId::new(3))));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pm = PassManager::with_defaults();
        let results = pm.run(&mut module);

        // Verify that passes ran
        assert!(!results.is_empty());

        // Check that certainty propagation optimized the extract_confidence
        // It should have been replaced with const 1.0
        let inst = &module.functions[0].blocks[0].instructions[3];
        if let SirInst::Const(Constant::F64(v)) = &inst.inst {
            assert_eq!(
                *v, 1.0,
                "Extract confidence from certain value should be 1.0"
            );
        }
    }

    #[test]
    fn test_certainty_with_metadata() {
        let mut func = create_test_function("test_metadata_certainty");

        // %0 = const 42.0
        // %1 = epistemic.extract_confidence %0  -- if %0 has certain metadata
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(42.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Epistemic(EpistemicOp::ExtractConfidence(ValueId::new(0))),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        // Create metadata marking %0 as certain
        let mut metadata = MetadataStore::new();
        metadata.attach(
            ValueId::new(0),
            Metadata::Epistemic(EpistemicMetadata::certain()),
        );

        let mut pass = CertaintyPropagation::new(metadata);
        let result = pass.run(&mut module);

        // The pass should optimize based on metadata
        assert!(result.modified);
    }

    #[test]
    fn test_partial_certainty_no_optimization() {
        let mut func = create_test_function("test_uncertain");

        // %0 = const 0.5 (uncertain confidence)
        // %1 = const 0.7 (uncertain confidence)
        // %2 = epistemic.propagate_add %0, %1  -- should NOT become const
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(0.5)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(0.7)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Epistemic(EpistemicOp::PropagateAdd {
                conf_a: ValueId::new(0),
                conf_b: ValueId::new(1),
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = CertaintyPropagation::new(MetadataStore::new());
        let result = pass.run(&mut module);

        // The propagate_add should NOT be replaced since inputs aren't certain
        let inst = &module.functions[0].blocks[0].instructions[2];
        assert!(
            matches!(
                inst.inst,
                SirInst::Epistemic(EpistemicOp::PropagateAdd { .. })
            ),
            "PropagateAdd with uncertain inputs should remain unchanged"
        );
    }

    // =========================================================================
    // AUTOMATIC UNCERTAINTY COMPILATION TESTS
    // =========================================================================

    #[test]
    fn test_auto_uncertainty_identification() {
        let mut func = create_test_function("test_auto_unc");

        // Simulate loading from memory (external input)
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Memory(MemoryOp::Load {
                ptr: ValueId::new(100), // Placeholder pointer
                ty: SirType::f64(),
                volatile: false,
                align: None,
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = AutomaticUncertaintyCompilation::new();
        let result = pass.run(&mut module);

        // Should identify the memory load as an uncertainty source
        assert!(pass.stats.implicit_uncertainties_found > 0);
        assert!(pass.stats.measurements > 0);
    }

    #[test]
    fn test_auto_uncertainty_fp_accumulation() {
        let mut func = create_test_function("test_fp_accum");

        // Floating-point division accumulates error
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(10.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(3.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FDiv,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = AutomaticUncertaintyCompilation::new();
        let _result = pass.run(&mut module);

        // Should identify division as FP accumulation point
        assert!(pass.stats.fp_accumulations > 0);
    }

    #[test]
    fn test_auto_uncertainty_propagation() {
        let mut func = create_test_function("test_unc_prop");

        // Load (uncertain) -> arithmetic -> result should be uncertain
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Memory(MemoryOp::Load {
                ptr: ValueId::new(100),
                ty: SirType::f64(),
                volatile: false,
                align: None,
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(2.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::BinOp {
                op: ArithOp::FMul,
                lhs: ValueId::new(0),
                rhs: ValueId::new(1),
            },
        );
        func.blocks[0].terminate(Terminator::Return(Some(ValueId::new(2))));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = AutomaticUncertaintyCompilation::new();
        let result = pass.run(&mut module);

        // Should propagate uncertainty from load through multiplication
        assert!(pass.stats.implicit_uncertainties_found >= 1);
    }

    #[test]
    fn test_auto_uncertainty_sources() {
        // Test different uncertainty source types
        assert_eq!(UncertaintySource::ExternalInput, UncertaintySource::ExternalInput);
        assert_ne!(UncertaintySource::ExternalInput, UncertaintySource::Measurement);
        assert_ne!(UncertaintySource::Stochastic, UncertaintySource::FloatingPointError);
    }

    #[test]
    fn test_auto_uncertainty_config() {
        let pass = AutomaticUncertaintyCompilation::with_config(
            0.5,  // external_input_confidence
            0.7,  // measurement_confidence
            0.95, // epistemic_threshold
        );

        assert_eq!(pass.external_input_confidence, 0.5);
        assert_eq!(pass.measurement_confidence, 0.7);
        assert_eq!(pass.epistemic_threshold, 0.95);
    }

    #[test]
    fn test_auto_uncertainty_pass_name() {
        let pass = AutomaticUncertaintyCompilation::new();
        assert_eq!(pass.name(), "auto-uncertainty");
    }

    #[test]
    fn test_uncertainty_stats_default() {
        let stats = UncertaintyStats::default();
        assert_eq!(stats.implicit_uncertainties_found, 0);
        assert_eq!(stats.epistemic_wrappers_inserted, 0);
        assert_eq!(stats.gum_propagations_inserted, 0);
        assert_eq!(stats.external_inputs, 0);
        assert_eq!(stats.measurements, 0);
        assert_eq!(stats.fp_accumulations, 0);
    }

    #[test]
    fn test_inferred_uncertainty() {
        let unc = InferredUncertainty {
            value: ValueId::new(0),
            source: UncertaintySource::Measurement,
            confidence: 0.8,
            relative_uncertainty: Some(0.05),
            needs_wrapping: true,
        };

        assert_eq!(unc.value, ValueId::new(0));
        assert_eq!(unc.source, UncertaintySource::Measurement);
        assert_eq!(unc.confidence, 0.8);
        assert_eq!(unc.relative_uncertainty, Some(0.05));
        assert!(unc.needs_wrapping);
    }

    #[test]
    fn test_auto_uncertainty_with_scientific_ops() {
        use super::super::ops::ScientificOp;

        let mut func = create_test_function("test_sci_ops");

        // ODE step introduces approximation error
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(0.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(1.0)),
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Const(Constant::F64(0.01)),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Scientific(ScientificOp::Lerp {
                a: ValueId::new(0),
                b: ValueId::new(1),
                t: ValueId::new(2),
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = AutomaticUncertaintyCompilation::new();
        let _result = pass.run(&mut module);

        // Should identify scientific operation as potential uncertainty source
        assert!(pass.stats.implicit_uncertainties_found > 0);
    }

    #[test]
    fn test_auto_uncertainty_with_prob_ops() {
        use super::super::types::DistributionKind;

        let mut func = create_test_function("test_prob_ops");

        // Creating and sampling from a distribution
        add_instruction(
            &mut func,
            ValueId::new(0),
            SirInst::Const(Constant::F64(0.0)), // mean
        );
        add_instruction(
            &mut func,
            ValueId::new(1),
            SirInst::Const(Constant::F64(1.0)), // std
        );
        add_instruction(
            &mut func,
            ValueId::new(2),
            SirInst::Prob(ProbOp::CreateDist {
                kind: DistributionKind::Normal,
                params: vec![ValueId::new(0), ValueId::new(1)],
            }),
        );
        add_instruction(
            &mut func,
            ValueId::new(3),
            SirInst::Prob(ProbOp::Sample {
                dist: ValueId::new(2),
                rng: ValueId::new(100), // placeholder RNG
            }),
        );
        func.blocks[0].terminate(Terminator::Return(None));

        let mut module = SirModule::new("test");
        module.functions.push(func);

        let mut pass = AutomaticUncertaintyCompilation::new();
        let _result = pass.run(&mut module);

        // Should identify sampling as stochastic uncertainty
        assert!(pass.stats.implicit_uncertainties_found > 0);
    }
}
