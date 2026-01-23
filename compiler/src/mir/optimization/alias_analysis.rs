//! Alias Analysis for MIR
//!
//! This module implements Steensgaard's points-to analysis algorithm, a flow-insensitive,
//! context-insensitive alias analysis that runs in almost-linear time.
//!
//! Based on:
//! - Steensgaard (1996) "Points-to Analysis in Almost Linear Time"
//! - Hind (2001) "Pointer Analysis: Haven't We Solved This Problem Yet?"
//!
//! The algorithm uses union-find (disjoint set) data structure to efficiently
//! track equivalence classes of pointers that may point to the same memory.
//!
//! ## Algorithm Overview
//!
//! Steensgaard's analysis is based on subset constraints:
//! - For `p = &x`: p points-to x
//! - For `p = q`: p and q are in the same equivalence class
//! - For `p = *q`: p may point to what q's target points to
//! - For `*p = q`: q may point to what p points to
//!
//! The key insight is that we use unification instead of subset constraints,
//! which makes the analysis faster but less precise.

use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;

use crate::mir::{MirFunction, MirInstruction, ValueId};

use super::pass_manager::AnalysisPass;

/// Abstract location in memory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Location {
    /// A local variable/value in SSA
    Value(ValueId),
    /// A stack allocation (from Alloca)
    Stack(ValueId),
    /// A global variable
    Global(u32),
    /// The heap (abstract location for all heap allocations)
    Heap,
    /// Unknown/external memory
    Unknown,
    /// Null pointer
    Null,
}

/// Points-to set for a location
#[derive(Debug, Clone, Default)]
pub struct PointsToSet {
    /// Locations this pointer may point to
    pub targets: HashSet<Location>,
}

impl PointsToSet {
    /// Create an empty points-to set
    pub fn new() -> Self {
        Self {
            targets: HashSet::new(),
        }
    }

    /// Create a points-to set with a single target
    pub fn singleton(target: Location) -> Self {
        Self {
            targets: HashSet::from([target]),
        }
    }

    /// Check if this set may alias with another
    pub fn may_alias(&self, other: &PointsToSet) -> bool {
        // Two pointers may alias if their points-to sets intersect
        // or if either contains Unknown
        if self.targets.contains(&Location::Unknown) || other.targets.contains(&Location::Unknown) {
            return true;
        }

        for target in &self.targets {
            if other.targets.contains(target) {
                return true;
            }
        }

        false
    }

    /// Merge another set into this one
    pub fn union(&mut self, other: &PointsToSet) {
        self.targets.extend(other.targets.iter().cloned());
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.targets.is_empty()
    }
}

/// Union-Find data structure for Steensgaard's algorithm
struct UnionFind {
    /// Parent pointers (self-loop = root)
    parent: HashMap<Location, Location>,
    /// Rank for union by rank
    rank: HashMap<Location, usize>,
}

impl UnionFind {
    fn new() -> Self {
        Self {
            parent: HashMap::new(),
            rank: HashMap::new(),
        }
    }

    /// Make a new set containing just this location
    fn make_set(&mut self, loc: Location) {
        if !self.parent.contains_key(&loc) {
            self.parent.insert(loc, loc);
            self.rank.insert(loc, 0);
        }
    }

    /// Find the representative of the set containing loc (with path compression)
    fn find(&mut self, loc: Location) -> Location {
        self.make_set(loc);

        let parent = *self.parent.get(&loc).unwrap();
        if parent != loc {
            let root = self.find(parent);
            self.parent.insert(loc, root);
            root
        } else {
            loc
        }
    }

    /// Union two sets
    fn union(&mut self, a: Location, b: Location) {
        let root_a = self.find(a);
        let root_b = self.find(b);

        if root_a == root_b {
            return;
        }

        let rank_a = *self.rank.get(&root_a).unwrap_or(&0);
        let rank_b = *self.rank.get(&root_b).unwrap_or(&0);

        if rank_a < rank_b {
            self.parent.insert(root_a, root_b);
        } else if rank_a > rank_b {
            self.parent.insert(root_b, root_a);
        } else {
            self.parent.insert(root_b, root_a);
            self.rank.insert(root_a, rank_a + 1);
        }
    }

    /// Get all locations in the same equivalence class
    fn get_class(&mut self, loc: Location) -> HashSet<Location> {
        let root = self.find(loc);
        let all_locs: Vec<Location> = self.parent.keys().cloned().collect();

        all_locs
            .into_iter()
            .filter(|&l| self.find(l) == root)
            .collect()
    }
}

/// Alias analysis result for a function
#[derive(Debug)]
pub struct AliasAnalysisResult {
    /// Points-to information for each value
    points_to: HashMap<ValueId, PointsToSet>,
    /// Equivalence classes from Steensgaard's analysis
    equivalence_classes: HashMap<Location, HashSet<Location>>,
}

impl AliasAnalysisResult {
    /// Create an empty result
    fn new() -> Self {
        Self {
            points_to: HashMap::new(),
            equivalence_classes: HashMap::new(),
        }
    }

    /// Get the points-to set for a value
    pub fn get_points_to(&self, value: ValueId) -> &PointsToSet {
        static EMPTY: LazyLock<PointsToSet> = LazyLock::new(|| PointsToSet {
            targets: HashSet::new(),
        });
        self.points_to.get(&value).unwrap_or(&EMPTY)
    }

    /// Check if two values may alias
    pub fn may_alias(&self, a: ValueId, b: ValueId) -> bool {
        // If both are the same value, they definitely alias
        if a == b {
            return true;
        }

        let pts_a = self.get_points_to(a);
        let pts_b = self.get_points_to(b);

        pts_a.may_alias(pts_b)
    }

    /// Check if two values must alias (conservative: returns true only if certain)
    pub fn must_alias(&self, a: ValueId, b: ValueId) -> bool {
        // Must-alias is harder to determine; we only report it for identical values
        // or if both point to exactly the same singleton location
        if a == b {
            return true;
        }

        let pts_a = self.get_points_to(a);
        let pts_b = self.get_points_to(b);

        // Must alias if both have exactly one target and it's the same
        if pts_a.targets.len() == 1
            && pts_b.targets.len() == 1
            && !pts_a.targets.contains(&Location::Unknown)
        {
            return pts_a.targets == pts_b.targets;
        }

        false
    }

    /// Check if a value may point to a stack location
    pub fn may_escape_to_heap(&self, value: ValueId) -> bool {
        let pts = self.get_points_to(value);
        pts.targets.contains(&Location::Heap) || pts.targets.contains(&Location::Unknown)
    }
}

/// Alias Analysis pass using Steensgaard's algorithm
pub struct AliasAnalysis {
    /// Results from the most recent analysis
    results: HashMap<String, AliasAnalysisResult>,
}

impl AliasAnalysis {
    /// Create a new alias analysis pass
    pub fn new() -> Self {
        Self {
            results: HashMap::new(),
        }
    }

    /// Get the analysis results for a function
    pub fn get_result(&self, func_name: &str) -> Option<&AliasAnalysisResult> {
        self.results.get(func_name)
    }

    /// Analyze a single function
    pub fn analyze_function(&mut self, func: &MirFunction) {
        let mut uf = UnionFind::new();
        let mut points_to: HashMap<ValueId, PointsToSet> = HashMap::new();

        // First pass: collect all values and create initial points-to information
        for block in &func.blocks {
            for instr in &block.instructions {
                self.process_instruction(instr, &mut uf, &mut points_to);
            }
        }

        // Build equivalence classes
        let mut equivalence_classes: HashMap<Location, HashSet<Location>> = HashMap::new();
        for (&value, pts) in &points_to {
            for &target in &pts.targets {
                let class = uf.get_class(target);
                equivalence_classes.insert(target, class);
            }
        }

        // Store results
        let result = AliasAnalysisResult {
            points_to,
            equivalence_classes,
        };
        self.results.insert(func.name.clone(), result);
    }

    /// Process a single instruction for alias analysis
    fn process_instruction(
        &self,
        instr: &MirInstruction,
        uf: &mut UnionFind,
        points_to: &mut HashMap<ValueId, PointsToSet>,
    ) {
        match instr {
            // Alloca: result points to a new stack location
            MirInstruction::Alloca { result, .. } => {
                let stack_loc = Location::Stack(*result);
                uf.make_set(stack_loc);
                points_to.insert(*result, PointsToSet::singleton(stack_loc));
            }

            // Load: result = *address
            // result points to what address's target points to
            MirInstruction::Load { result, address, .. } => {
                // Conservative: result may point to anything address points to
                let addr_pts = points_to.get(address).cloned().unwrap_or_default();

                // For each location address may point to, unify with result
                let mut result_pts = PointsToSet::new();
                for target in &addr_pts.targets {
                    // If we're loading from a known location, we need to find
                    // what that location points to (which we track conservatively)
                    result_pts.targets.insert(*target);
                    uf.union(Location::Value(*result), *target);
                }

                // If we don't know what address points to, be conservative
                if addr_pts.is_empty() {
                    result_pts.targets.insert(Location::Unknown);
                }

                points_to.insert(*result, result_pts);
            }

            // Store: *address = value
            // What address points to now aliases with what value points to
            MirInstruction::Store { address, value } => {
                let addr_pts = points_to.get(address).cloned().unwrap_or_default();
                let val_pts = points_to.get(value).cloned().unwrap_or_default();

                // Unify the targets of address with the targets of value
                for addr_target in &addr_pts.targets {
                    for val_target in &val_pts.targets {
                        uf.union(*addr_target, *val_target);
                    }
                }
            }

            // GEP: result = &base[indices]
            // result points to the same object as base (field-insensitive)
            MirInstruction::GetElementPtr { result, base, .. } => {
                let base_pts = points_to.get(base).cloned().unwrap_or_default();
                points_to.insert(*result, base_pts.clone());

                // Unify result with base's targets
                for target in &base_pts.targets {
                    uf.union(Location::Value(*result), *target);
                }
            }

            // LoadGlobal: result points to global memory
            MirInstruction::LoadGlobal { result, .. } => {
                let global_loc = Location::Global(result.0 as u32);
                uf.make_set(global_loc);
                points_to.insert(*result, PointsToSet::singleton(global_loc));
            }

            // StoreGlobal: similar to Store but for globals
            MirInstruction::StoreGlobal { value, .. } => {
                // The global may now point to what value points to
                let val_pts = points_to.get(value).cloned().unwrap_or_default();
                for target in &val_pts.targets {
                    uf.union(Location::Unknown, *target);
                }
            }

            // Call: Conservative - return value may alias anything
            MirInstruction::Call { result: Some(result), args, .. }
            | MirInstruction::CallIndirect { result: Some(result), args, .. } => {
                // Arguments may escape
                for arg in args {
                    if let Some(arg_pts) = points_to.get(arg) {
                        for target in &arg_pts.targets {
                            uf.union(*target, Location::Heap);
                        }
                    }
                }

                // Result may alias heap or unknown
                let mut result_pts = PointsToSet::new();
                result_pts.targets.insert(Location::Heap);
                result_pts.targets.insert(Location::Unknown);
                points_to.insert(*result, result_pts);
            }

            // Pointer cast: result aliases source
            MirInstruction::PtrCast { result, source, .. }
            | MirInstruction::Cast { result, source, .. } => {
                let src_pts = points_to.get(source).cloned().unwrap_or_default();
                points_to.insert(*result, src_pts.clone());

                for target in &src_pts.targets {
                    uf.union(Location::Value(*result), *target);
                }
            }

            // Phi: result may alias any of the incoming values
            MirInstruction::Phi { result, incoming, .. } => {
                let mut result_pts = PointsToSet::new();
                for (_, value) in incoming {
                    let val_pts = points_to.get(value).cloned().unwrap_or_default();
                    result_pts.union(&val_pts);

                    for target in &val_pts.targets {
                        uf.union(Location::Value(*result), *target);
                    }
                }
                points_to.insert(*result, result_pts);
            }

            // Select: result may alias either true_value or false_value
            MirInstruction::Select { result, true_value, false_value, .. } => {
                let true_pts = points_to.get(true_value).cloned().unwrap_or_default();
                let false_pts = points_to.get(false_value).cloned().unwrap_or_default();

                let mut result_pts = PointsToSet::new();
                result_pts.union(&true_pts);
                result_pts.union(&false_pts);

                for target in &true_pts.targets {
                    uf.union(Location::Value(*result), *target);
                }
                for target in &false_pts.targets {
                    uf.union(Location::Value(*result), *target);
                }

                points_to.insert(*result, result_pts);
            }

            // Const: non-pointer constants don't affect alias analysis
            // However, null pointers are tracked
            MirInstruction::Const { result, value, ty } => {
                if ty.is_pointer() {
                    match value {
                        crate::mir::MirConstant::Null => {
                            points_to.insert(*result, PointsToSet::singleton(Location::Null));
                        }
                        crate::mir::MirConstant::GlobalRef(_) => {
                            let global_loc = Location::Global(result.0 as u32);
                            points_to.insert(*result, PointsToSet::singleton(global_loc));
                        }
                        _ => {
                            // Other pointer constants - conservative
                            points_to.insert(*result, PointsToSet::singleton(Location::Unknown));
                        }
                    }
                }
            }

            // Other instructions don't affect pointer aliasing
            _ => {}
        }
    }
}

impl Default for AliasAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl AnalysisPass for AliasAnalysis {
    fn name(&self) -> &'static str {
        "alias-analysis"
    }

    fn run_on_function(&self, func: &MirFunction) -> Result<(), String> {
        // Note: AnalysisPass takes &self, but we need to mutate.
        // In practice, you would use interior mutability or a different pattern.
        // For now, this is a no-op; use analyze_function directly.
        Ok(())
    }
}

/// Query interface for alias analysis
pub struct AliasQuery<'a> {
    result: &'a AliasAnalysisResult,
}

impl<'a> AliasQuery<'a> {
    /// Create a new query interface
    pub fn new(result: &'a AliasAnalysisResult) -> Self {
        Self { result }
    }

    /// Check if two memory operations may alias
    pub fn may_alias(&self, a: ValueId, b: ValueId) -> AliasResult {
        if a == b {
            return AliasResult::MustAlias;
        }

        let pts_a = self.result.get_points_to(a);
        let pts_b = self.result.get_points_to(b);

        // Check for must-alias (both point to exactly the same singleton)
        if pts_a.targets.len() == 1
            && pts_b.targets.len() == 1
            && pts_a.targets == pts_b.targets
            && !pts_a.targets.contains(&Location::Unknown)
        {
            return AliasResult::MustAlias;
        }

        // Check for may-alias
        if pts_a.may_alias(pts_b) {
            return AliasResult::MayAlias;
        }

        AliasResult::NoAlias
    }

    /// Check if a store to `store_addr` may affect a load from `load_addr`
    pub fn store_may_affect_load(&self, store_addr: ValueId, load_addr: ValueId) -> bool {
        matches!(
            self.may_alias(store_addr, load_addr),
            AliasResult::MayAlias | AliasResult::MustAlias
        )
    }
}

/// Result of an alias query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasResult {
    /// The two pointers definitely do not alias
    NoAlias,
    /// The two pointers may alias
    MayAlias,
    /// The two pointers definitely alias
    MustAlias,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::{BlockId, FuncId, MirConstant, MirType};

    #[test]
    fn test_points_to_set() {
        let mut pts_a = PointsToSet::new();
        pts_a.targets.insert(Location::Stack(ValueId(0)));

        let mut pts_b = PointsToSet::new();
        pts_b.targets.insert(Location::Stack(ValueId(1)));

        // Different stack locations should not alias
        assert!(!pts_a.may_alias(&pts_b));

        // Same stack location should alias
        let mut pts_c = PointsToSet::new();
        pts_c.targets.insert(Location::Stack(ValueId(0)));
        assert!(pts_a.may_alias(&pts_c));

        // Unknown should alias with anything
        let mut pts_unknown = PointsToSet::new();
        pts_unknown.targets.insert(Location::Unknown);
        assert!(pts_a.may_alias(&pts_unknown));
    }

    #[test]
    fn test_union_find() {
        let mut uf = UnionFind::new();

        let a = Location::Stack(ValueId(0));
        let b = Location::Stack(ValueId(1));
        let c = Location::Stack(ValueId(2));

        uf.make_set(a);
        uf.make_set(b);
        uf.make_set(c);

        // Initially all separate
        assert_ne!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(b), uf.find(c));

        // Union a and b
        uf.union(a, b);
        assert_eq!(uf.find(a), uf.find(b));
        assert_ne!(uf.find(a), uf.find(c));

        // Union b and c (should merge all three)
        uf.union(b, c);
        assert_eq!(uf.find(a), uf.find(c));
    }

    #[test]
    fn test_alloca_analysis() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit);

        // Create two allocas
        let alloc_a = builder.fresh_value();
        builder.build_alloca(alloc_a, MirType::I64);

        let alloc_b = builder.fresh_value();
        builder.build_alloca(alloc_b, MirType::I64);

        builder.build_return(None);

        let func = builder.build();

        // Run analysis
        let mut analysis = AliasAnalysis::new();
        analysis.analyze_function(&func);

        let result = analysis.get_result("test").unwrap();

        // Two different allocas should not alias
        assert!(!result.may_alias(alloc_a, alloc_b));

        // Same alloca should alias with itself
        assert!(result.may_alias(alloc_a, alloc_a));
    }

    #[test]
    fn test_alias_query() {
        let mut result = AliasAnalysisResult::new();

        // Set up: a and b point to different locations
        let mut pts_a = PointsToSet::new();
        pts_a.targets.insert(Location::Stack(ValueId(100)));
        result.points_to.insert(ValueId(0), pts_a);

        let mut pts_b = PointsToSet::new();
        pts_b.targets.insert(Location::Stack(ValueId(101)));
        result.points_to.insert(ValueId(1), pts_b);

        let query = AliasQuery::new(&result);

        // Different targets = no alias
        assert_eq!(query.may_alias(ValueId(0), ValueId(1)), AliasResult::NoAlias);

        // Same value = must alias
        assert_eq!(query.may_alias(ValueId(0), ValueId(0)), AliasResult::MustAlias);
    }

    #[test]
    fn test_phi_analysis() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit);

        // Create allocas in entry
        let alloc_a = builder.fresh_value();
        builder.build_alloca(alloc_a, MirType::Ptr(Box::new(MirType::I64)));

        let alloc_b = builder.fresh_value();
        builder.build_alloca(alloc_b, MirType::Ptr(Box::new(MirType::I64)));

        let merge_block = builder.create_block("merge");
        builder.build_branch(merge_block);

        // Merge block with phi
        builder.switch_to_block(merge_block);
        let phi_result = builder.fresh_value();
        builder.build_phi(
            phi_result,
            vec![(BlockId(0), alloc_a), (BlockId(0), alloc_b)],
            MirType::Ptr(Box::new(MirType::I64)),
        );

        builder.build_return(None);

        let func = builder.build();

        // Run analysis
        let mut analysis = AliasAnalysis::new();
        analysis.analyze_function(&func);

        let result = analysis.get_result("test").unwrap();

        // phi_result may alias with both alloc_a and alloc_b
        assert!(result.may_alias(phi_result, alloc_a));
        assert!(result.may_alias(phi_result, alloc_b));
    }

    #[test]
    fn test_load_store_analysis() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit);

        // p = alloca i64
        let p = builder.fresh_value();
        builder.build_alloca(p, MirType::I64);

        // q = alloca i64
        let q = builder.fresh_value();
        builder.build_alloca(q, MirType::I64);

        // *p = 42
        let forty_two = builder.fresh_value();
        builder.build_const(forty_two, MirConstant::Int(42), MirType::I64);
        builder.build_store(p, forty_two);

        // x = *p
        let x = builder.fresh_value();
        builder.build_load(x, p, MirType::I64);

        builder.build_return(None);

        let func = builder.build();

        // Run analysis
        let mut analysis = AliasAnalysis::new();
        analysis.analyze_function(&func);

        let result = analysis.get_result("test").unwrap();
        let query = AliasQuery::new(result);

        // Store to p may affect load from p
        assert!(query.store_may_affect_load(p, p));

        // Store to p should not affect load from q (different allocas)
        assert!(!query.store_may_affect_load(p, q));
    }
}
