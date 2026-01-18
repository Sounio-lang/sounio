//! Polyhedral Analysis
//!
//! This module provides analysis passes to extract polyhedral representations
//! from HLIR. The key analyses are:
//!
//! 1. **SCOP Detection**: Find Static Control Parts - regions where all
//!    loop bounds and array accesses are affine functions.
//!
//! 2. **Dependence Analysis**: Compute data dependencies between statements
//!    using the polyhedral model.
//!
//! # References
//!
//! - Feautrier (1991) "Dataflow analysis of array and scalar references"
//! - Maydan et al. (1991) "Data dependence and data-flow analysis of arrays"
//! - Pugh & Wonnacott (1994) "An exact method for analysis of value-based
//!   array data dependences"

use std::collections::{HashMap, HashSet, VecDeque};

use crate::hlir::{
    BlockId, BinaryOp, HlirConstant, HlirFunction, HlirInstr,
    HlirTerminator, HlirType, Op, ValueId,
};

use super::affine::{AccessRelation, AffineExpr, AffineMap};
use super::domain::{Constraint, IntegerSet, IterationDomain};
use super::{PolyError, PolyResult};

/// Information about a detected loop
#[derive(Debug, Clone)]
pub struct LoopInfo {
    /// Header block ID
    pub header: BlockId,
    /// Preheader block (predecessor outside loop)
    pub preheader: Option<BlockId>,
    /// Back edge source block
    pub latch: BlockId,
    /// All blocks in the loop
    pub blocks: HashSet<BlockId>,
    /// Exit blocks (blocks with successors outside the loop)
    pub exits: HashSet<BlockId>,
    /// Loop induction variable (if detected)
    pub induction_var: Option<InductionVariable>,
    /// Nesting depth (0 = outermost)
    pub depth: usize,
    /// Parent loop (if this is a nested loop)
    pub parent: Option<BlockId>,
    /// Child loops (nested within this loop)
    pub children: Vec<BlockId>,
}

/// Information about a loop induction variable
#[derive(Debug, Clone)]
pub struct InductionVariable {
    /// The SSA value representing the induction variable
    pub value: ValueId,
    /// Initial value
    pub init: InductionInit,
    /// Step value (increment per iteration)
    pub step: i64,
    /// Final value (upper bound)
    pub final_value: InductionBound,
    /// Comparison operator for the exit condition
    pub cmp_op: BoundComparison,
}

/// Initial value of an induction variable
#[derive(Debug, Clone)]
pub enum InductionInit {
    Constant(i64),
    Value(ValueId),
    Parameter(String),
}

/// Bound for an induction variable
#[derive(Debug, Clone)]
pub enum InductionBound {
    Constant(i64),
    Value(ValueId),
    Parameter(String),
    /// Affine expression in terms of outer loop indices and parameters
    Affine(AffineExpr),
}

/// Comparison type for loop bounds
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundComparison {
    LessThan,
    LessEqual,
    GreaterThan,
    GreaterEqual,
    NotEqual,
}

/// A loop nest - a collection of perfectly nested loops
#[derive(Debug, Clone)]
pub struct LoopNest {
    /// Loops in the nest, from outermost to innermost
    pub loops: Vec<LoopInfo>,
    /// Names of loop indices
    pub index_names: Vec<String>,
    /// Combined iteration domain
    pub domain: IterationDomain,
}

impl LoopNest {
    /// Get the nesting depth
    pub fn depth(&self) -> usize {
        self.loops.len()
    }

    /// Check if the loop nest is perfectly nested
    pub fn is_perfectly_nested(&self) -> bool {
        // A loop nest is perfectly nested if each loop body
        // contains only the next inner loop (plus statements at innermost level)
        // Simplified check: verify parent-child relationships
        for (i, loop_info) in self.loops.iter().enumerate() {
            if i < self.loops.len() - 1 {
                // Non-innermost loop should have exactly one child
                if loop_info.children.len() != 1 {
                    return false;
                }
            }
        }
        true
    }
}

/// Information about a statement in a SCOP
#[derive(Debug, Clone)]
pub struct StatementInfo {
    /// Unique identifier for this statement
    pub id: usize,
    /// Block containing the statement
    pub block: BlockId,
    /// Instruction index within the block
    pub instr_index: usize,
    /// Iteration domain (all iterations where this statement executes)
    pub domain: IterationDomain,
    /// Read accesses performed by this statement
    pub reads: Vec<AccessRelation>,
    /// Write accesses performed by this statement
    pub writes: Vec<AccessRelation>,
    /// Loop depth (0 = outside all loops)
    pub loop_depth: usize,
}

/// Static Control Part (SCOP) information
///
/// A SCOP is a maximal region where:
/// - All loop bounds are affine functions of outer iterators and parameters
/// - All array subscripts are affine functions of iterators and parameters
/// - Control flow is statically predictable (no data-dependent branches)
#[derive(Debug, Clone)]
pub struct ScopInfo {
    /// Entry block of the SCOP
    pub entry: BlockId,
    /// Exit blocks of the SCOP
    pub exits: HashSet<BlockId>,
    /// All blocks in the SCOP
    pub blocks: HashSet<BlockId>,
    /// Loop nests in this SCOP
    pub loop_nests: Vec<LoopNest>,
    /// Statements in this SCOP
    pub statements: Vec<StatementInfo>,
    /// Parameters (symbolic constants) used in bounds/subscripts
    pub parameters: Vec<String>,
    /// Arrays accessed in this SCOP
    pub arrays: HashMap<String, ArrayInfo>,
}

/// Information about an array in a SCOP
#[derive(Debug, Clone)]
pub struct ArrayInfo {
    /// Array name
    pub name: String,
    /// Element type
    pub element_type: HlirType,
    /// Number of dimensions
    pub num_dims: usize,
    /// Size of each dimension (if known)
    pub dim_sizes: Vec<Option<i64>>,
}

/// Type of data dependence
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DependenceType {
    /// Read after Write (RAW / true dependence / flow dependence)
    Flow,
    /// Write after Read (WAR / anti-dependence)
    Anti,
    /// Write after Write (WAW / output dependence)
    Output,
    /// Read after Read (RAR / input dependence - not a true dependence)
    Input,
}

impl DependenceType {
    /// Check if this is a true dependence that must be respected
    pub fn is_true_dependence(&self) -> bool {
        matches!(self, DependenceType::Flow | DependenceType::Output)
    }
}

/// A data dependence between two statement instances
#[derive(Debug, Clone)]
pub struct DependenceInfo {
    /// Source statement ID
    pub source_stmt: usize,
    /// Sink statement ID
    pub sink_stmt: usize,
    /// Type of dependence
    pub dep_type: DependenceType,
    /// Array involved in the dependence
    pub array: String,
    /// Dependence polyhedron (constraints on iteration vectors)
    pub polyhedron: IntegerSet,
    /// Dependence distance vector (if constant)
    pub distance_vector: Option<Vec<i64>>,
    /// Direction vector (for non-constant distances)
    pub direction_vector: Option<Vec<Direction>>,
    /// Whether the dependence is loop-carried (vs loop-independent)
    pub is_loop_carried: bool,
    /// The loop level at which this dependence is carried (0 = outermost)
    pub carried_at_level: Option<usize>,
}

/// Direction of a dependence for one loop dimension
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Positive direction (<)
    Positive,
    /// Zero (=)
    Zero,
    /// Negative direction (>)
    Negative,
    /// Unknown (*)
    Any,
}

impl Direction {
    /// Create from a constant distance
    pub fn from_distance(d: i64) -> Self {
        match d.cmp(&0) {
            std::cmp::Ordering::Less => Direction::Negative,
            std::cmp::Ordering::Equal => Direction::Zero,
            std::cmp::Ordering::Greater => Direction::Positive,
        }
    }
}

/// Main polyhedral analysis pass
pub struct PolyhedralAnalysis {
    /// Detected loops (header -> info)
    loops: HashMap<BlockId, LoopInfo>,
    /// CFG predecessors
    predecessors: HashMap<BlockId, Vec<BlockId>>,
    /// CFG successors
    successors: HashMap<BlockId, Vec<BlockId>>,
    /// Dominator tree
    dominators: HashMap<BlockId, HashSet<BlockId>>,
    /// Value definitions (value -> defining block)
    value_defs: HashMap<ValueId, BlockId>,
}

impl PolyhedralAnalysis {
    /// Create a new polyhedral analysis instance
    pub fn new() -> Self {
        Self {
            loops: HashMap::new(),
            predecessors: HashMap::new(),
            successors: HashMap::new(),
            dominators: HashMap::new(),
            value_defs: HashMap::new(),
        }
    }

    /// Find all SCOPs in a function
    pub fn find_scops(&self, func: &HlirFunction) -> PolyResult<Vec<ScopInfo>> {
        let mut analysis = PolyhedralAnalysis::new();
        analysis.analyze_function(func)?;

        let mut scops = Vec::new();

        // For each detected loop, try to form a SCOP
        for (header, loop_info) in &analysis.loops {
            // Skip inner loops (they're part of outer loop's SCOP)
            if loop_info.parent.is_some() {
                continue;
            }

            if let Ok(scop) = analysis.extract_scop(func, *header) {
                scops.push(scop);
            }
        }

        Ok(scops)
    }

    /// Compute data dependencies for a SCOP
    pub fn compute_dependencies(&self, scop: &ScopInfo) -> PolyResult<Vec<DependenceInfo>> {
        let mut dependencies = Vec::new();

        // For each pair of statements
        for (i, stmt1) in scop.statements.iter().enumerate() {
            for (j, stmt2) in scop.statements.iter().enumerate() {
                // Check all access pairs
                for write in &stmt1.writes {
                    // Flow dependence: write -> read
                    for read in &stmt2.reads {
                        if let Some(dep) = self.check_dependence(
                            stmt1,
                            stmt2,
                            write,
                            read,
                            DependenceType::Flow,
                        ) {
                            dependencies.push(dep);
                        }
                    }

                    // Output dependence: write -> write
                    for write2 in &stmt2.writes {
                        if i != j || write.statement_id != write2.statement_id {
                            if let Some(dep) = self.check_dependence(
                                stmt1,
                                stmt2,
                                write,
                                write2,
                                DependenceType::Output,
                            ) {
                                dependencies.push(dep);
                            }
                        }
                    }
                }

                // Anti-dependence: read -> write
                for read in &stmt1.reads {
                    for write in &stmt2.writes {
                        if i != j || read.statement_id != write.statement_id {
                            if let Some(dep) = self.check_dependence(
                                stmt1,
                                stmt2,
                                read,
                                write,
                                DependenceType::Anti,
                            ) {
                                dependencies.push(dep);
                            }
                        }
                    }
                }
            }
        }

        Ok(dependencies)
    }

    /// Analyze a function to detect loops and build CFG
    fn analyze_function(&mut self, func: &HlirFunction) -> PolyResult<()> {
        self.build_cfg(func);
        self.compute_dominators(func);
        self.detect_loops(func);
        self.collect_value_definitions(func);
        self.classify_loops();
        Ok(())
    }

    /// Build CFG from function blocks
    fn build_cfg(&mut self, func: &HlirFunction) {
        self.predecessors.clear();
        self.successors.clear();

        // Initialize empty sets
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

    /// Compute dominators using iterative algorithm
    fn compute_dominators(&mut self, func: &HlirFunction) {
        if func.blocks.is_empty() {
            return;
        }

        self.dominators.clear();
        let entry = BlockId::ENTRY;
        let all_blocks: HashSet<BlockId> = func.blocks.iter().map(|b| b.id).collect();

        // Initialize
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
    fn detect_loops(&mut self, func: &HlirFunction) {
        self.loops.clear();

        // Find back edges
        for block in &func.blocks {
            let succs = self.successors.get(&block.id).cloned().unwrap_or_default();
            for succ in succs {
                if self.dominates(succ, block.id) {
                    // Found a back edge: block -> succ
                    // succ is the loop header
                    let mut loop_info = LoopInfo {
                        header: succ,
                        preheader: None,
                        latch: block.id,
                        blocks: HashSet::from([succ]),
                        exits: HashSet::new(),
                        induction_var: None,
                        depth: 0,
                        parent: None,
                        children: Vec::new(),
                    };

                    // Compute loop body
                    self.compute_loop_body(func, &mut loop_info, block.id);

                    // Find preheader
                    let preds = self.predecessors.get(&succ).cloned().unwrap_or_default();
                    let outside_preds: Vec<BlockId> = preds
                        .iter()
                        .filter(|p| !loop_info.blocks.contains(p))
                        .cloned()
                        .collect();
                    if outside_preds.len() == 1 {
                        loop_info.preheader = Some(outside_preds[0]);
                    }

                    // Find exit blocks
                    for &body_block in &loop_info.blocks {
                        let body_succs = self.successors.get(&body_block).cloned().unwrap_or_default();
                        for succ in body_succs {
                            if !loop_info.blocks.contains(&succ) {
                                loop_info.exits.insert(body_block);
                                break;
                            }
                        }
                    }

                    // Try to detect induction variable
                    loop_info.induction_var = self.detect_induction_variable(func, &loop_info);

                    self.loops.insert(succ, loop_info);
                }
            }
        }
    }

    /// Compute loop body by walking backwards from latch to header
    fn compute_loop_body(&self, _func: &HlirFunction, loop_info: &mut LoopInfo, latch: BlockId) {
        let mut worklist = VecDeque::new();
        worklist.push_back(latch);

        while let Some(block) = worklist.pop_front() {
            if loop_info.blocks.contains(&block) {
                continue;
            }

            loop_info.blocks.insert(block);

            if block != loop_info.header {
                let preds = self.predecessors.get(&block).cloned().unwrap_or_default();
                for pred in preds {
                    if !loop_info.blocks.contains(&pred) {
                        worklist.push_back(pred);
                    }
                }
            }
        }
    }

    /// Classify loops (determine nesting structure)
    fn classify_loops(&mut self) {
        let headers: Vec<BlockId> = self.loops.keys().cloned().collect();

        // Determine parent-child relationships
        for &header1 in &headers {
            for &header2 in &headers {
                if header1 == header2 {
                    continue;
                }

                let loop1_blocks = self.loops.get(&header1).map(|l| l.blocks.clone()).unwrap_or_default();
                let loop2_blocks = self.loops.get(&header2).map(|l| l.blocks.clone()).unwrap_or_default();

                // Check if loop2 is nested inside loop1
                if loop1_blocks.is_superset(&loop2_blocks) {
                    // loop2 is nested in loop1
                    if let Some(loop2) = self.loops.get_mut(&header2) {
                        // Only set parent if loop1 is the immediate parent
                        if loop2.parent.is_none() || loop2_blocks.len() < loop1_blocks.len() {
                            loop2.parent = Some(header1);
                        }
                    }
                    if let Some(loop1) = self.loops.get_mut(&header1) {
                        if !loop1.children.contains(&header2) {
                            loop1.children.push(header2);
                        }
                    }
                }
            }
        }

        // Compute depths
        for &header in &headers {
            let mut depth = 0;
            let mut current = header;
            while let Some(parent) = self.loops.get(&current).and_then(|l| l.parent) {
                depth += 1;
                current = parent;
            }
            if let Some(loop_info) = self.loops.get_mut(&header) {
                loop_info.depth = depth;
            }
        }
    }

    /// Collect value definitions
    fn collect_value_definitions(&mut self, func: &HlirFunction) {
        self.value_defs.clear();

        for block in &func.blocks {
            for instr in &block.instructions {
                if let Some(result) = instr.result {
                    self.value_defs.insert(result, block.id);
                }
            }
        }
    }

    /// Try to detect the induction variable for a loop
    fn detect_induction_variable(
        &self,
        func: &HlirFunction,
        loop_info: &LoopInfo,
    ) -> Option<InductionVariable> {
        // Look for phi nodes in the header that could be induction variables
        let header_block = func.get_block(loop_info.header)?;

        for instr in &header_block.instructions {
            if let Op::Phi { incoming } = &instr.op {
                if let Some(result) = instr.result {
                    // Check if this phi has the pattern:
                    // - One incoming value from outside the loop (init)
                    // - One incoming value from the loop body (updated value)

                    let mut init_value = None;
                    let mut loop_value = None;

                    for (pred, value) in incoming {
                        if loop_info.blocks.contains(pred) {
                            loop_value = Some(*value);
                        } else {
                            init_value = Some(*value);
                        }
                    }

                    if let (Some(init), Some(loop_val)) = (init_value, loop_value) {
                        // Try to find the increment operation
                        if let Some(iv) = self.check_induction_pattern(
                            func,
                            loop_info,
                            result,
                            init,
                            loop_val,
                        ) {
                            return Some(iv);
                        }
                    }
                }
            }
        }

        None
    }

    /// Check if values follow an induction variable pattern
    fn check_induction_pattern(
        &self,
        func: &HlirFunction,
        loop_info: &LoopInfo,
        phi_result: ValueId,
        init_value: ValueId,
        loop_value: ValueId,
    ) -> Option<InductionVariable> {
        // Look for: loop_value = phi_result + constant (or constant + phi_result)
        for block in &func.blocks {
            if !loop_info.blocks.contains(&block.id) {
                continue;
            }

            for instr in &block.instructions {
                if instr.result != Some(loop_value) {
                    continue;
                }

                if let Op::Binary { op, left, right } = &instr.op {
                    if *op != BinaryOp::Add {
                        continue;
                    }

                    // Check if one operand is phi_result and the other is constant
                    let (is_match, step) = if *left == phi_result {
                        (true, self.get_constant_value(func, *right))
                    } else if *right == phi_result {
                        (true, self.get_constant_value(func, *left))
                    } else {
                        (false, None)
                    };

                    if is_match {
                        if let Some(step_val) = step {
                            // Found induction variable pattern
                            let init = self.get_induction_init(func, init_value);
                            let bound = self.get_loop_bound(func, loop_info, phi_result);

                            return Some(InductionVariable {
                                value: phi_result,
                                init,
                                step: step_val,
                                final_value: bound.0,
                                cmp_op: bound.1,
                            });
                        }
                    }
                }
            }
        }

        None
    }

    /// Get constant value if the value is a constant
    fn get_constant_value(&self, func: &HlirFunction, value: ValueId) -> Option<i64> {
        for block in &func.blocks {
            for instr in &block.instructions {
                if instr.result != Some(value) {
                    continue;
                }

                if let Op::Const(constant) = &instr.op {
                    return match constant {
                        HlirConstant::Int(n, _) => Some(*n),
                        HlirConstant::Bool(b) => Some(if *b { 1 } else { 0 }),
                        _ => None,
                    };
                }
            }
        }
        None
    }

    /// Get the induction variable init value
    fn get_induction_init(&self, func: &HlirFunction, init_value: ValueId) -> InductionInit {
        if let Some(c) = self.get_constant_value(func, init_value) {
            InductionInit::Constant(c)
        } else {
            InductionInit::Value(init_value)
        }
    }

    /// Get the loop bound from the exit condition
    fn get_loop_bound(
        &self,
        func: &HlirFunction,
        loop_info: &LoopInfo,
        iv: ValueId,
    ) -> (InductionBound, BoundComparison) {
        // Look for comparison in loop header or latch
        for &block_id in &[loop_info.header, loop_info.latch] {
            if let Some(block) = func.get_block(block_id) {
                // Look for conditional branch
                if let HlirTerminator::CondBranch { condition, .. } = &block.terminator {
                    // Find the comparison instruction
                    for instr in &block.instructions {
                        if instr.result != Some(*condition) {
                            continue;
                        }

                        if let Op::Binary { op, left, right } = &instr.op {
                            let cmp_op = match op {
                                BinaryOp::SLt | BinaryOp::ULt | BinaryOp::FOLt => {
                                    Some(BoundComparison::LessThan)
                                }
                                BinaryOp::SLe | BinaryOp::ULe | BinaryOp::FOLe => {
                                    Some(BoundComparison::LessEqual)
                                }
                                BinaryOp::SGt | BinaryOp::UGt | BinaryOp::FOGt => {
                                    Some(BoundComparison::GreaterThan)
                                }
                                BinaryOp::SGe | BinaryOp::UGe | BinaryOp::FOGe => {
                                    Some(BoundComparison::GreaterEqual)
                                }
                                BinaryOp::Ne | BinaryOp::FONe => Some(BoundComparison::NotEqual),
                                _ => None,
                            };

                            if let Some(cmp) = cmp_op {
                                // Check which operand is the IV
                                let bound_value = if *left == iv {
                                    *right
                                } else if *right == iv {
                                    *left
                                } else {
                                    continue;
                                };

                                if let Some(c) = self.get_constant_value(func, bound_value) {
                                    return (InductionBound::Constant(c), cmp);
                                } else {
                                    return (InductionBound::Value(bound_value), cmp);
                                }
                            }
                        }
                    }
                }
            }
        }

        // Default if we can't determine the bound
        (InductionBound::Value(ValueId::UNIT), BoundComparison::LessThan)
    }

    /// Extract a SCOP from a loop
    fn extract_scop(&self, func: &HlirFunction, header: BlockId) -> PolyResult<ScopInfo> {
        let loop_info = self.loops.get(&header).ok_or_else(|| PolyError::Internal {
            reason: "Loop not found".to_string(),
        })?;

        // Build loop nest
        let mut loop_nests = Vec::new();
        let loop_nest = self.build_loop_nest(func, header)?;
        loop_nests.push(loop_nest);

        // Extract statements
        let statements = self.extract_statements(func, loop_info)?;

        // Collect arrays
        let mut arrays = HashMap::new();
        for stmt in &statements {
            for access in stmt.reads.iter().chain(stmt.writes.iter()) {
                if !arrays.contains_key(&access.array_name) {
                    arrays.insert(
                        access.array_name.clone(),
                        ArrayInfo {
                            name: access.array_name.clone(),
                            element_type: HlirType::I64, // TODO: Infer from context
                            num_dims: access.map.num_outputs,
                            dim_sizes: vec![None; access.map.num_outputs],
                        },
                    );
                }
            }
        }

        Ok(ScopInfo {
            entry: header,
            exits: loop_info.exits.clone(),
            blocks: loop_info.blocks.clone(),
            loop_nests,
            statements,
            parameters: Vec::new(), // TODO: Extract parameters
            arrays,
        })
    }

    /// Build a loop nest from a header
    fn build_loop_nest(&self, func: &HlirFunction, header: BlockId) -> PolyResult<LoopNest> {
        let mut loops = Vec::new();
        let mut index_names = Vec::new();
        let mut current = Some(header);

        while let Some(h) = current {
            if let Some(loop_info) = self.loops.get(&h) {
                loops.push(loop_info.clone());
                index_names.push(format!("i{}", loops.len() - 1));

                // Find nested loop
                current = loop_info.children.first().cloned();
            } else {
                break;
            }
        }

        // Build combined domain
        let domain = self.build_domain(&loops)?;

        Ok(LoopNest {
            loops,
            index_names,
            domain,
        })
    }

    /// Build iteration domain from loop info
    fn build_domain(&self, loops: &[LoopInfo]) -> PolyResult<IterationDomain> {
        let num_loops = loops.len();
        let index_names: Vec<String> = (0..num_loops).map(|i| format!("i{}", i)).collect();
        let mut domain = IterationDomain::new(index_names, vec![]);

        for (i, loop_info) in loops.iter().enumerate() {
            if let Some(iv) = &loop_info.induction_var {
                // Add lower bound
                match &iv.init {
                    InductionInit::Constant(c) => {
                        domain.set.add_lower_bound(i, *c);
                    }
                    _ => {
                        // Non-constant init - add parameter
                        // Simplified: assume 0 for now
                        domain.set.add_lower_bound(i, 0);
                    }
                }

                // Add upper bound
                match &iv.final_value {
                    InductionBound::Constant(c) => {
                        match iv.cmp_op {
                            BoundComparison::LessThan => {
                                domain.set.add_upper_bound_strict(i, *c);
                            }
                            BoundComparison::LessEqual => {
                                domain.set.add_upper_bound(i, *c);
                            }
                            _ => {
                                // Conservative: use arbitrary upper bound
                                domain.set.add_upper_bound_strict(i, 1000);
                            }
                        }
                    }
                    _ => {
                        // Non-constant bound - would need parametric representation
                        domain.set.add_upper_bound_strict(i, 1000);
                    }
                }
            } else {
                // No induction variable detected - use defaults
                domain.set.add_lower_bound(i, 0);
                domain.set.add_upper_bound_strict(i, 1000);
            }
        }

        Ok(domain)
    }

    /// Extract statements from loop body
    fn extract_statements(
        &self,
        func: &HlirFunction,
        loop_info: &LoopInfo,
    ) -> PolyResult<Vec<StatementInfo>> {
        let mut statements = Vec::new();
        let mut stmt_id = 0;

        for &block_id in &loop_info.blocks {
            if let Some(block) = func.get_block(block_id) {
                for (instr_idx, instr) in block.instructions.iter().enumerate() {
                    // Skip phi nodes (they're part of the loop structure)
                    if matches!(instr.op, Op::Phi { .. }) {
                        continue;
                    }

                    // Skip pure computations without memory effects
                    // (we're interested in memory accesses for dependence analysis)
                    let (reads, writes) = self.extract_accesses(func, instr, stmt_id);

                    if !reads.is_empty() || !writes.is_empty() {
                        let domain = self.build_domain(&[loop_info.clone()])?;

                        statements.push(StatementInfo {
                            id: stmt_id,
                            block: block_id,
                            instr_index: instr_idx,
                            domain,
                            reads,
                            writes,
                            loop_depth: loop_info.depth + 1,
                        });

                        stmt_id += 1;
                    }
                }
            }
        }

        Ok(statements)
    }

    /// Extract memory accesses from an instruction
    fn extract_accesses(
        &self,
        _func: &HlirFunction,
        instr: &HlirInstr,
        stmt_id: usize,
    ) -> (Vec<AccessRelation>, Vec<AccessRelation>) {
        let mut reads = Vec::new();
        let mut writes = Vec::new();

        match &instr.op {
            Op::Load { ptr } => {
                // Simplified: create a generic access
                let access = AccessRelation {
                    array_name: format!("mem_{}", ptr.0),
                    map: AffineMap::identity(1),
                    is_write: false,
                    statement_id: stmt_id,
                };
                reads.push(access);
            }
            Op::Store { ptr, .. } => {
                let access = AccessRelation {
                    array_name: format!("mem_{}", ptr.0),
                    map: AffineMap::identity(1),
                    is_write: true,
                    statement_id: stmt_id,
                };
                writes.push(access);
            }
            Op::GetElementPtr { base, index, .. } => {
                // This gives us index information
                let access = AccessRelation {
                    array_name: format!("arr_{}", base.0),
                    map: AffineMap::identity(1),
                    is_write: false,
                    statement_id: stmt_id,
                };
                reads.push(access);
            }
            _ => {}
        }

        (reads, writes)
    }

    /// Check for dependence between two accesses
    fn check_dependence(
        &self,
        source_stmt: &StatementInfo,
        sink_stmt: &StatementInfo,
        source_access: &AccessRelation,
        sink_access: &AccessRelation,
        dep_type: DependenceType,
    ) -> Option<DependenceInfo> {
        // Different arrays can't have dependence
        if source_access.array_name != sink_access.array_name {
            return None;
        }

        // Build dependence polyhedron
        // This is a simplified version - a full implementation would use
        // the Omega test or ISL for precise analysis

        let num_dims = source_stmt.domain.depth() + sink_stmt.domain.depth();
        let mut poly = IntegerSet::new(num_dims, 0);

        // Add domain constraints for source and sink
        for c in &source_stmt.domain.set.constraints {
            poly.add_constraint(c.clone());
        }

        // Shift sink constraints to use dimensions [n, 2n)
        for c in &sink_stmt.domain.set.constraints {
            let mut shifted_coeffs = vec![0; source_stmt.domain.depth()];
            shifted_coeffs.extend(c.iter_coeffs.clone());
            poly.add_constraint(Constraint {
                iter_coeffs: shifted_coeffs,
                param_coeffs: c.param_coeffs.clone(),
                constant: c.constant,
                is_equality: c.is_equality,
            });
        }

        // Add access equality constraints
        // source_access(source_iters) = sink_access(sink_iters)
        for dim in 0..source_access.map.num_outputs.min(sink_access.map.num_outputs) {
            let source_expr = &source_access.map.outputs[dim];
            let sink_expr = &sink_access.map.outputs[dim];

            // Build equality: source_expr - sink_expr = 0
            let mut coeffs = source_expr.iter_coeffs.clone();
            coeffs.resize(num_dims, 0);
            for (i, coeff) in sink_expr.iter_coeffs.iter().enumerate() {
                coeffs[source_stmt.domain.depth() + i] -= coeff;
            }

            poly.add_constraint(Constraint::equality(
                coeffs,
                vec![],
                source_expr.constant - sink_expr.constant,
            ));
        }

        // Check if polyhedron is empty (no dependence)
        if poly.is_empty() {
            return None;
        }

        // Determine if loop-carried
        let is_loop_carried = source_stmt.id == sink_stmt.id
            || !Self::is_same_iteration(&source_stmt.domain, &sink_stmt.domain);

        Some(DependenceInfo {
            source_stmt: source_stmt.id,
            sink_stmt: sink_stmt.id,
            dep_type,
            array: source_access.array_name.clone(),
            polyhedron: poly,
            distance_vector: None, // TODO: Compute distance vector
            direction_vector: None,
            is_loop_carried,
            carried_at_level: None, // TODO: Determine carrying level
        })
    }

    /// Check if two domains represent the same iteration
    fn is_same_iteration(domain1: &IterationDomain, domain2: &IterationDomain) -> bool {
        // Simplified check
        domain1.set.constraints == domain2.set.constraints
    }
}

impl Default for PolyhedralAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Get successors from a terminator
fn get_terminator_successors(term: &HlirTerminator) -> Vec<BlockId> {
    match term {
        HlirTerminator::Branch(target) => vec![*target],
        HlirTerminator::CondBranch {
            then_block,
            else_block,
            ..
        } => vec![*then_block, *else_block],
        HlirTerminator::Switch { default, cases, .. } => {
            let mut succs = vec![*default];
            succs.extend(cases.iter().map(|(_, t)| *t));
            succs
        }
        HlirTerminator::Return(_) | HlirTerminator::Unreachable => vec![],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dependence_type() {
        assert!(DependenceType::Flow.is_true_dependence());
        assert!(DependenceType::Output.is_true_dependence());
        assert!(!DependenceType::Anti.is_true_dependence());
        assert!(!DependenceType::Input.is_true_dependence());
    }

    #[test]
    fn test_direction_from_distance() {
        assert_eq!(Direction::from_distance(5), Direction::Positive);
        assert_eq!(Direction::from_distance(0), Direction::Zero);
        assert_eq!(Direction::from_distance(-3), Direction::Negative);
    }

    #[test]
    fn test_bound_comparison() {
        let bound = InductionBound::Constant(10);
        assert!(matches!(bound, InductionBound::Constant(10)));
    }

    #[test]
    fn test_induction_init() {
        let init = InductionInit::Constant(0);
        assert!(matches!(init, InductionInit::Constant(0)));
    }

    #[test]
    fn test_polyhedral_analysis_creation() {
        let analysis = PolyhedralAnalysis::new();
        assert!(analysis.loops.is_empty());
    }
}
