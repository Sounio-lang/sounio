//! Alias Analysis for MIR
//!
//! This module provides alias analysis to determine whether two memory
//! references may refer to the same location. This enables more aggressive
//! optimizations by proving memory operations are independent.
//!
//! Based on:
//! - Andersen (1994) "Program Analysis and Specialization for C"
//! - Steensgaard (1996) "Points-to Analysis in Almost Linear Time"
//! - Muchnick (1997) "Advanced Compiler Design and Implementation", Chapter 10

// Compiler internals often require deep nesting for pattern matching and control flow.
#![allow(clippy::excessive_nesting)]

use crate::mir::{MirFunction, MirInstruction, MirModule, MirType, ValueId};
use std::collections::{HashMap, HashSet, VecDeque};

/// Result of alias query
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasResult {
    /// Definitely the same location
    MustAlias,
    /// Possibly the same location (conservative)
    MayAlias,
    /// Definitely different locations
    NoAlias,
}

impl AliasResult {
    /// Returns true if the locations may alias
    pub fn may_alias(&self) -> bool {
        matches!(self, AliasResult::MustAlias | AliasResult::MayAlias)
    }

    /// Combine two alias results (meet operation)
    pub fn meet(self, other: AliasResult) -> AliasResult {
        match (self, other) {
            (AliasResult::MustAlias, AliasResult::MustAlias) => AliasResult::MustAlias,
            (AliasResult::NoAlias, AliasResult::NoAlias) => AliasResult::NoAlias,
            _ => AliasResult::MayAlias,
        }
    }
}

/// Memory location representation
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MemoryLocation {
    /// Stack-allocated variable
    Stack { alloca_id: ValueId },
    /// Global variable
    Global { name: String },
    /// Heap-allocated memory
    Heap { alloc_site: ValueId },
    /// Function parameter (pointer)
    Parameter { param_index: usize },
    /// Field of a struct
    Field {
        base: Box<MemoryLocation>,
        field_index: usize,
    },
    /// Array element
    ArrayElement {
        base: Box<MemoryLocation>,
        index: Option<i64>, // None = unknown index
    },
    /// Unknown location (conservative)
    Unknown,
}

impl MemoryLocation {
    /// Get the base location (stripping field/array accesses)
    pub fn base(&self) -> &MemoryLocation {
        match self {
            MemoryLocation::Field { base, .. } | MemoryLocation::ArrayElement { base, .. } => {
                base.base()
            }
            _ => self,
        }
    }
}

/// Points-to set for a pointer value
#[derive(Debug, Clone, Default)]
pub struct PointsToSet {
    locations: HashSet<MemoryLocation>,
}

impl PointsToSet {
    pub fn new() -> Self {
        Self {
            locations: HashSet::new(),
        }
    }

    pub fn singleton(loc: MemoryLocation) -> Self {
        let mut locations = HashSet::new();
        locations.insert(loc);
        Self { locations }
    }

    pub fn add(&mut self, loc: MemoryLocation) {
        self.locations.insert(loc);
    }

    pub fn union(&mut self, other: &PointsToSet) -> bool {
        let old_len = self.locations.len();
        self.locations.extend(other.locations.iter().cloned());
        self.locations.len() > old_len
    }

    pub fn contains(&self, loc: &MemoryLocation) -> bool {
        self.locations.contains(loc)
    }

    pub fn is_empty(&self) -> bool {
        self.locations.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MemoryLocation> {
        self.locations.iter()
    }

    /// Check if this set may overlap with another
    pub fn may_overlap(&self, other: &PointsToSet) -> bool {
        // Unknown overlaps with everything
        if self.locations.contains(&MemoryLocation::Unknown)
            || other.locations.contains(&MemoryLocation::Unknown)
        {
            return true;
        }

        // Check for any common locations
        for loc in &self.locations {
            for other_loc in &other.locations {
                if locations_may_alias(loc, other_loc) {
                    return true;
                }
            }
        }
        false
    }
}

/// Check if two memory locations may alias
fn locations_may_alias(loc1: &MemoryLocation, loc2: &MemoryLocation) -> bool {
    use MemoryLocation::*;

    match (loc1, loc2) {
        // Unknown aliases with everything
        (Unknown, _) | (_, Unknown) => true,

        // Same exact location
        (a, b) if a == b => true,

        // Different stack allocations never alias
        (Stack { alloca_id: id1 }, Stack { alloca_id: id2 }) => id1 == id2,

        // Different globals never alias
        (Global { name: n1 }, Global { name: n2 }) => n1 == n2,

        // Stack and globals never alias
        (Stack { .. }, Global { .. }) | (Global { .. }, Stack { .. }) => false,

        // Different heap allocations never alias (if tracked separately)
        (Heap { alloc_site: s1 }, Heap { alloc_site: s2 }) => s1 == s2,

        // Parameters may alias with heap but not stack/globals
        (Parameter { .. }, Heap { .. }) | (Heap { .. }, Parameter { .. }) => true,
        (Parameter { .. }, Stack { .. }) | (Stack { .. }, Parameter { .. }) => false,
        (Parameter { .. }, Global { .. }) | (Global { .. }, Parameter { .. }) => true,
        (Parameter { param_index: i1 }, Parameter { param_index: i2 }) => i1 == i2,

        // Field accesses: check base first
        (
            Field {
                base: b1,
                field_index: f1,
            },
            Field {
                base: b2,
                field_index: f2,
            },
        ) => {
            if f1 != f2 {
                false // Different fields
            } else {
                locations_may_alias(b1, b2) // Same field, check base
            }
        }

        // Array elements: conservative if same base
        (
            ArrayElement {
                base: b1,
                index: i1,
            },
            ArrayElement {
                base: b2,
                index: i2,
            },
        ) => {
            if !locations_may_alias(b1, b2) {
                false
            } else {
                match (i1, i2) {
                    (Some(idx1), Some(idx2)) => idx1 == idx2, // Known different indices
                    _ => true,                                // Unknown index - may alias
                }
            }
        }

        // Mixed field/array accesses - conservative
        (Field { base: b1, .. }, ArrayElement { base: b2, .. })
        | (ArrayElement { base: b1, .. }, Field { base: b2, .. }) => locations_may_alias(b1, b2),

        // Everything else - conservative may alias
        _ => true,
    }
}

/// Alias analysis for a function
pub struct AliasAnalysis {
    /// Points-to information for each value
    points_to: HashMap<ValueId, PointsToSet>,
    /// Escape analysis: values that may escape the function
    escaped_values: HashSet<ValueId>,
    /// Memory locations that may be modified
    modified_locations: HashSet<MemoryLocation>,
}

impl AliasAnalysis {
    /// Analyze a function for alias information
    pub fn analyze(func: &MirFunction) -> Self {
        let mut analysis = Self {
            points_to: HashMap::new(),
            escaped_values: HashSet::new(),
            modified_locations: HashSet::new(),
        };

        // Initialize points-to sets for parameters
        for (i, (param_name, param_type)) in func.params.iter().enumerate() {
            if is_pointer_type(param_type) {
                let param_value = ValueId(i);
                let mut pts = PointsToSet::new();
                pts.add(MemoryLocation::Parameter { param_index: i });
                analysis.points_to.insert(param_value, pts);
                // Parameters may escape
                analysis.escaped_values.insert(param_value);
            }
        }

        // Fixed-point iteration for points-to analysis
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for block in &func.blocks {
                for instr in &block.instructions {
                    if analysis.process_instruction(instr) {
                        changed = true;
                    }
                }
            }
        }

        analysis
    }

    /// Process a single instruction for points-to analysis
    fn process_instruction(&mut self, instr: &MirInstruction) -> bool {
        let mut changed = false;

        match instr {
            // Alloca creates a new stack location
            MirInstruction::Alloca { result, ty, .. } => {
                let loc = MemoryLocation::Stack { alloca_id: *result };
                let pts = PointsToSet::singleton(loc);
                if self.points_to.insert(*result, pts).is_none() {
                    changed = true;
                }
            }

            // Load dereferences a pointer
            MirInstruction::Load {
                result, address, ..
            } => {
                if let Some(addr_pts) = self.points_to.get(address).cloned() {
                    // Result points to what the address points to (transitively)
                    // For simplicity, mark as unknown
                    let mut pts = PointsToSet::new();
                    pts.add(MemoryLocation::Unknown);
                    if let Some(existing) = self.points_to.get_mut(result) {
                        changed |= existing.union(&pts);
                    } else {
                        self.points_to.insert(*result, pts);
                        changed = true;
                    }
                }
            }

            // Store modifies memory
            MirInstruction::Store { address, value, .. } => {
                if let Some(addr_pts) = self.points_to.get(address) {
                    for loc in addr_pts.iter() {
                        self.modified_locations.insert(loc.clone());
                    }
                }
                // If storing a pointer, propagate points-to
                if let Some(val_pts) = self.points_to.get(value).cloned() {
                    if let Some(addr_pts) = self.points_to.get(address).cloned() {
                        for loc in addr_pts.iter() {
                            // Update points-to through the store
                            // This is a simplification - full analysis tracks this
                        }
                    }
                }
            }

            // GEP (GetElementPtr) computes derived pointer
            MirInstruction::GetElementPtr {
                result,
                base,
                indices,
                ..
            } => {
                if let Some(base_pts) = self.points_to.get(base).cloned() {
                    let mut new_pts = PointsToSet::new();
                    for loc in base_pts.iter() {
                        // Create field/array access based on indices
                        // Simplified: just track that it derives from base
                        if indices.is_empty() {
                            new_pts.add(loc.clone());
                        } else {
                            // First index is array element, rest are fields
                            let derived = MemoryLocation::ArrayElement {
                                base: Box::new(loc.clone()),
                                index: None, // Could extract constant index
                            };
                            new_pts.add(derived);
                        }
                    }
                    if let Some(existing) = self.points_to.get_mut(result) {
                        changed |= existing.union(&new_pts);
                    } else {
                        self.points_to.insert(*result, new_pts);
                        changed = true;
                    }
                }
            }

            // Function calls may modify anything through pointers
            MirInstruction::Call { result, args, .. } => {
                // Mark pointer arguments as escaped
                for arg in args {
                    if self.points_to.contains_key(arg) {
                        self.escaped_values.insert(*arg);
                    }
                }
                // Result may point to heap or anything returned
                if let Some(result_val) = result {
                    let mut pts = PointsToSet::new();
                    pts.add(MemoryLocation::Heap {
                        alloc_site: *result_val,
                    });
                    pts.add(MemoryLocation::Unknown); // Conservative
                    self.points_to.insert(*result_val, pts);
                    changed = true;
                }
            }

            // Phi nodes merge points-to sets
            MirInstruction::Phi {
                result, incoming, ..
            } => {
                let mut merged = PointsToSet::new();
                for (_, val) in incoming {
                    if let Some(val_pts) = self.points_to.get(val) {
                        merged.union(val_pts);
                    }
                }
                if !merged.is_empty() {
                    if let Some(existing) = self.points_to.get_mut(result) {
                        changed |= existing.union(&merged);
                    } else {
                        self.points_to.insert(*result, merged);
                        changed = true;
                    }
                }
            }

            // Casts preserve points-to
            MirInstruction::PtrCast { result, source, .. }
            | MirInstruction::Cast { result, source, .. } => {
                if let Some(src_pts) = self.points_to.get(source).cloned() {
                    if let Some(existing) = self.points_to.get_mut(result) {
                        changed |= existing.union(&src_pts);
                    } else {
                        self.points_to.insert(*result, src_pts);
                        changed = true;
                    }
                }
            }

            _ => {}
        }

        changed
    }

    /// Query whether two values may alias
    pub fn query(&self, ptr1: ValueId, ptr2: ValueId) -> AliasResult {
        let pts1 = self.points_to.get(&ptr1);
        let pts2 = self.points_to.get(&ptr2);

        match (pts1, pts2) {
            (None, _) | (_, None) => AliasResult::NoAlias, // Non-pointer values
            (Some(pts1), Some(pts2)) => {
                if pts1.may_overlap(pts2) {
                    // Check for must-alias (singleton sets with same location)
                    if pts1.locations.len() == 1
                        && pts2.locations.len() == 1
                        && pts1.locations == pts2.locations
                    {
                        AliasResult::MustAlias
                    } else {
                        AliasResult::MayAlias
                    }
                } else {
                    AliasResult::NoAlias
                }
            }
        }
    }

    /// Check if a store to one location may affect a load from another
    pub fn store_may_affect_load(&self, store_addr: ValueId, load_addr: ValueId) -> bool {
        self.query(store_addr, load_addr).may_alias()
    }

    /// Check if a value has escaped (may be accessible outside function)
    pub fn has_escaped(&self, value: ValueId) -> bool {
        self.escaped_values.contains(&value)
    }

    /// Get locations that may be modified by the function
    pub fn get_modified_locations(&self) -> &HashSet<MemoryLocation> {
        &self.modified_locations
    }

    /// Get points-to set for a value
    pub fn get_points_to(&self, value: ValueId) -> Option<&PointsToSet> {
        self.points_to.get(&value)
    }
}

/// Check if a type is a pointer type
fn is_pointer_type(ty: &MirType) -> bool {
    matches!(ty, MirType::Ptr(_))
}

/// Inter-procedural alias analysis for a module
pub struct ModuleAliasAnalysis {
    /// Per-function analysis results
    function_analyses: HashMap<String, AliasAnalysis>,
    /// Call graph for inter-procedural analysis
    call_graph: HashMap<String, HashSet<String>>,
}

impl ModuleAliasAnalysis {
    /// Analyze entire module
    pub fn analyze(module: &MirModule) -> Self {
        let mut analysis = Self {
            function_analyses: HashMap::new(),
            call_graph: HashMap::new(),
        };

        // Build call graph
        for func in &module.functions {
            let mut callees = HashSet::new();
            for block in &func.blocks {
                for instr in &block.instructions {
                    if let MirInstruction::Call { func_name, .. } = instr {
                        callees.insert(func_name.clone());
                    }
                }
            }
            analysis.call_graph.insert(func.name.clone(), callees);
        }

        // Analyze each function
        for func in &module.functions {
            let func_analysis = AliasAnalysis::analyze(func);
            analysis
                .function_analyses
                .insert(func.name.clone(), func_analysis);
        }

        analysis
    }

    /// Get analysis for a specific function
    pub fn get_function_analysis(&self, func_name: &str) -> Option<&AliasAnalysis> {
        self.function_analyses.get(func_name)
    }

    /// Check if a function may modify a memory location
    pub fn function_may_modify(&self, func_name: &str, loc: &MemoryLocation) -> bool {
        // Check the function and all callees transitively
        let mut visited = HashSet::new();
        let mut worklist = VecDeque::new();
        worklist.push_back(func_name.to_string());

        while let Some(fname) = worklist.pop_front() {
            if !visited.insert(fname.clone()) {
                continue;
            }

            if let Some(analysis) = self.function_analyses.get(&fname) {
                for modified_loc in analysis.get_modified_locations() {
                    if locations_may_alias(modified_loc, loc) {
                        return true;
                    }
                }
            }

            // Add callees
            if let Some(callees) = self.call_graph.get(&fname) {
                for callee in callees {
                    worklist.push_back(callee.clone());
                }
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::FuncId;
    use crate::mir::builder::{FunctionBuilder, ModuleBuilder};

    #[test]
    fn test_alias_result_meet() {
        assert_eq!(
            AliasResult::MustAlias.meet(AliasResult::MustAlias),
            AliasResult::MustAlias
        );
        assert_eq!(
            AliasResult::NoAlias.meet(AliasResult::NoAlias),
            AliasResult::NoAlias
        );
        assert_eq!(
            AliasResult::MustAlias.meet(AliasResult::NoAlias),
            AliasResult::MayAlias
        );
        assert_eq!(
            AliasResult::MayAlias.meet(AliasResult::NoAlias),
            AliasResult::MayAlias
        );
    }

    #[test]
    fn test_points_to_set() {
        let mut pts = PointsToSet::new();
        assert!(pts.is_empty());

        pts.add(MemoryLocation::Stack {
            alloca_id: ValueId(0),
        });
        assert!(!pts.is_empty());

        let pts2 = PointsToSet::singleton(MemoryLocation::Stack {
            alloca_id: ValueId(1),
        });
        assert!(!pts.may_overlap(&pts2)); // Different allocas don't overlap
    }

    #[test]
    fn test_memory_location_aliasing() {
        let stack1 = MemoryLocation::Stack {
            alloca_id: ValueId(0),
        };
        let stack2 = MemoryLocation::Stack {
            alloca_id: ValueId(1),
        };
        let global1 = MemoryLocation::Global {
            name: "g1".to_string(),
        };

        // Same stack location aliases with itself
        assert!(locations_may_alias(&stack1, &stack1));

        // Different stack locations don't alias
        assert!(!locations_may_alias(&stack1, &stack2));

        // Stack and global don't alias
        assert!(!locations_may_alias(&stack1, &global1));

        // Unknown aliases with everything
        assert!(locations_may_alias(&MemoryLocation::Unknown, &stack1));
    }

    #[test]
    fn test_alias_analysis_basic() {
        let func = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit).build();

        let analysis = AliasAnalysis::analyze(&func);

        // Empty function should have no points-to information
        assert!(analysis.points_to.is_empty());
    }

    #[test]
    fn test_module_alias_analysis() {
        let mut builder = ModuleBuilder::new("test");

        let mut func = builder.create_function("main".to_string(), MirType::Unit);
        func.build_return(None);
        builder.add_function(func.build());

        let module = builder.build();
        let analysis = ModuleAliasAnalysis::analyze(&module);

        assert!(analysis.get_function_analysis("main").is_some());
    }
}
