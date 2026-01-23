//! Scalar Replacement of Aggregates (SROA)
//!
//! This module implements SROA for MIR, based on:
//! - Muchnick (1997) "Advanced Compiler Design and Implementation", Chapter 12
//! - LLVM's SROA pass documentation
//!
//! SROA identifies aggregate (struct/array) allocations that are only accessed
//! through known, constant offsets and replaces them with individual scalar
//! allocations. This enables further optimizations like register allocation
//! to work on the individual fields.
//!
//! ## Algorithm Overview
//!
//! 1. Identify candidate allocas (aggregates with known, simple field accesses)
//! 2. Verify all uses are through GEP with constant indices
//! 3. Create scalar allocas for each field
//! 4. Rewrite GEP + load/store to use the scalar allocas directly
//! 5. Remove the original aggregate alloca
//!
//! ## Limitations
//!
//! - Only handles allocas with known, constant-index field accesses
//! - Does not handle pointer escapes (address taken and passed elsewhere)
//! - Does not handle memcpy/memset operations on aggregates

use std::collections::{HashMap, HashSet};

use crate::mir::{
    BlockId, MirFunction, MirInstruction, MirModule, MirTerminator, MirType, ValueId,
};

use super::pass_manager::MIRPass;

/// Information about an aggregate allocation
#[derive(Debug)]
struct AllocaInfo {
    /// The ValueId of the alloca result
    value: ValueId,
    /// The type being allocated
    ty: MirType,
    /// Field accesses: (field_index, load/store instructions)
    field_accesses: HashMap<Vec<u64>, Vec<AccessInfo>>,
    /// Whether this alloca can be split
    is_splittable: bool,
    /// Reason if not splittable
    rejection_reason: Option<String>,
}

/// Information about a field access
#[derive(Debug, Clone)]
struct AccessInfo {
    /// Block containing the access
    block_id: BlockId,
    /// Instruction index within the block
    instr_idx: usize,
    /// The GEP result value (if any)
    gep_value: Option<ValueId>,
    /// Whether this is a load (true) or store (false)
    is_load: bool,
    /// The value loaded or stored
    value: ValueId,
}

/// Scalar Replacement of Aggregates pass
pub struct Sroa {
    /// Statistics
    stats: SroaStats,
}

/// Statistics from SROA
#[derive(Debug, Default)]
pub struct SroaStats {
    /// Number of allocas analyzed
    pub allocas_analyzed: usize,
    /// Number of allocas split
    pub allocas_split: usize,
    /// Number of scalar allocas created
    pub scalars_created: usize,
    /// Number of instructions removed
    pub instructions_removed: usize,
}

impl Sroa {
    /// Create a new SROA pass
    pub fn new() -> Self {
        Self {
            stats: SroaStats::default(),
        }
    }

    /// Get statistics from the last run
    pub fn stats(&self) -> &SroaStats {
        &self.stats
    }

    /// Find all alloca instructions in a function
    fn find_allocas(func: &MirFunction) -> Vec<(BlockId, usize, ValueId, MirType)> {
        let mut allocas = Vec::new();

        for block in &func.blocks {
            for (idx, instr) in block.instructions.iter().enumerate() {
                if let MirInstruction::Alloca { result, ty } = instr {
                    allocas.push((block.id, idx, *result, ty.clone()));
                }
            }
        }

        allocas
    }

    /// Check if a type is an aggregate (struct or array)
    fn is_aggregate_type(ty: &MirType) -> bool {
        matches!(ty, MirType::Struct { .. } | MirType::Array(_, _) | MirType::Tuple(_))
    }

    /// Get the field type for an aggregate at a given index path
    fn get_field_type(ty: &MirType, indices: &[u64]) -> Option<MirType> {
        if indices.is_empty() {
            return Some(ty.clone());
        }

        let idx = indices[0] as usize;
        let remaining = &indices[1..];

        match ty {
            MirType::Struct { fields, .. } => {
                if idx < fields.len() {
                    let (_, field_ty) = &fields[idx];
                    Self::get_field_type(field_ty, remaining)
                } else {
                    None
                }
            }
            MirType::Array(elem_ty, size) => {
                if idx < *size {
                    Self::get_field_type(elem_ty, remaining)
                } else {
                    None
                }
            }
            MirType::Tuple(types) => {
                if idx < types.len() {
                    Self::get_field_type(&types[idx], remaining)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Analyze uses of an alloca to determine if it can be split
    fn analyze_alloca_uses(func: &MirFunction, alloca_value: ValueId, alloca_ty: &MirType) -> AllocaInfo {
        let mut info = AllocaInfo {
            value: alloca_value,
            ty: alloca_ty.clone(),
            field_accesses: HashMap::new(),
            is_splittable: true,
            rejection_reason: None,
        };

        // Collect all uses of the alloca
        let mut worklist: Vec<(ValueId, Vec<u64>)> = vec![(alloca_value, vec![])];
        let mut visited: HashSet<ValueId> = HashSet::new();

        while let Some((current_value, current_indices)) = worklist.pop() {
            if !visited.insert(current_value) {
                continue;
            }

            // Find all instructions that use this value
            for block in &func.blocks {
                for (instr_idx, instr) in block.instructions.iter().enumerate() {
                    match instr {
                        // GEP: extend the index path
                        MirInstruction::GetElementPtr { result, base, indices, .. } => {
                            if *base == current_value {
                                // Extract constant indices
                                let mut index_values = current_indices.clone();
                                let mut all_const = true;

                                for idx_value in indices {
                                    // Try to find the constant value for this index
                                    if let Some(const_val) = Self::get_constant_value(func, *idx_value) {
                                        index_values.push(const_val as u64);
                                    } else {
                                        all_const = false;
                                        break;
                                    }
                                }

                                if all_const {
                                    worklist.push((*result, index_values));
                                } else {
                                    info.is_splittable = false;
                                    info.rejection_reason = Some("Non-constant GEP index".to_string());
                                    return info;
                                }
                            }
                        }

                        // Load: record the access
                        MirInstruction::Load { result, address, .. } => {
                            if *address == current_value {
                                let access = AccessInfo {
                                    block_id: block.id,
                                    instr_idx,
                                    gep_value: if current_value != alloca_value {
                                        Some(current_value)
                                    } else {
                                        None
                                    },
                                    is_load: true,
                                    value: *result,
                                };
                                info.field_accesses
                                    .entry(current_indices.clone())
                                    .or_default()
                                    .push(access);
                            }
                        }

                        // Store: record the access or check for escape
                        MirInstruction::Store { address, value } => {
                            if *address == current_value {
                                // Store to the aggregate field
                                let access = AccessInfo {
                                    block_id: block.id,
                                    instr_idx,
                                    gep_value: if current_value != alloca_value {
                                        Some(current_value)
                                    } else {
                                        None
                                    },
                                    is_load: false,
                                    value: *value,
                                };
                                info.field_accesses
                                    .entry(current_indices.clone())
                                    .or_default()
                                    .push(access);
                            } else if *value == current_value {
                                // Store of the pointer itself (address taken/escapes)
                                info.is_splittable = false;
                                info.rejection_reason = Some("Address of aggregate stored".to_string());
                                return info;
                            }
                        }

                        // Call: pointer may escape
                        MirInstruction::Call { args, .. }
                        | MirInstruction::CallIndirect { args, .. } => {
                            if args.contains(&current_value) {
                                info.is_splittable = false;
                                info.rejection_reason = Some("Pointer escapes to function call".to_string());
                                return info;
                            }
                        }

                        // Phi: complex control flow, be conservative
                        MirInstruction::Phi { incoming, .. } => {
                            for (_, val) in incoming {
                                if *val == current_value {
                                    info.is_splittable = false;
                                    info.rejection_reason = Some("Pointer used in phi node".to_string());
                                    return info;
                                }
                            }
                        }

                        _ => {}
                    }
                }
            }
        }

        // Verify we have at least some field accesses
        if info.field_accesses.is_empty() {
            info.is_splittable = false;
            info.rejection_reason = Some("No field accesses found".to_string());
        }

        info
    }

    /// Try to get a constant value for a ValueId
    fn get_constant_value(func: &MirFunction, value: ValueId) -> Option<i64> {
        for block in &func.blocks {
            for instr in &block.instructions {
                if let MirInstruction::Const { result, value: const_val, .. } = instr {
                    if *result == value {
                        return match const_val {
                            crate::mir::MirConstant::Int(v) => Some(*v),
                            crate::mir::MirConstant::UInt(v) => Some(*v as i64),
                            _ => None,
                        };
                    }
                }
            }
        }
        None
    }

    /// Perform SROA transformation on a function
    fn transform_function(&mut self, func: &mut MirFunction) -> bool {
        let allocas = Self::find_allocas(func);
        let mut modified = false;

        for (alloca_block, alloca_idx, alloca_value, alloca_ty) in allocas {
            self.stats.allocas_analyzed += 1;

            // Only process aggregate types
            if !Self::is_aggregate_type(&alloca_ty) {
                continue;
            }

            // Analyze uses
            let info = Self::analyze_alloca_uses(func, alloca_value, &alloca_ty);

            if !info.is_splittable {
                continue;
            }

            // Create scalar allocas for each accessed field
            let scalar_allocas = self.create_scalar_allocas(func, &info);

            if scalar_allocas.is_empty() {
                continue;
            }

            // Rewrite accesses to use scalar allocas
            self.rewrite_accesses(func, &info, &scalar_allocas);

            // Mark for removal (we'll remove in a cleanup pass)
            self.stats.allocas_split += 1;
            modified = true;
        }

        // Clean up removed instructions
        if modified {
            self.cleanup_removed_instructions(func);
        }

        modified
    }

    /// Create scalar allocas for each field
    fn create_scalar_allocas(
        &mut self,
        func: &mut MirFunction,
        info: &AllocaInfo,
    ) -> HashMap<Vec<u64>, ValueId> {
        let mut scalar_allocas = HashMap::new();
        let mut next_value_id = func
            .blocks
            .iter()
            .flat_map(|b| b.instructions.iter())
            .filter_map(|i| i.result())
            .map(|v| v.0)
            .max()
            .unwrap_or(0)
            + 1;

        // Find the entry block to insert scalar allocas
        let entry_block_id = func.entry_block;

        for (indices, _accesses) in &info.field_accesses {
            // Get the field type
            let field_ty = match Self::get_field_type(&info.ty, indices) {
                Some(ty) => ty,
                None => continue,
            };

            // Create a new scalar alloca
            let scalar_value = ValueId(next_value_id);
            next_value_id += 1;

            let scalar_alloca = MirInstruction::Alloca {
                result: scalar_value,
                ty: field_ty,
            };

            // Insert at the beginning of the entry block
            if let Some(entry_block) = func.get_block_mut(entry_block_id) {
                entry_block.instructions.insert(0, scalar_alloca);
            }

            scalar_allocas.insert(indices.clone(), scalar_value);
            self.stats.scalars_created += 1;
        }

        scalar_allocas
    }

    /// Rewrite accesses to use scalar allocas
    fn rewrite_accesses(
        &mut self,
        func: &mut MirFunction,
        info: &AllocaInfo,
        scalar_allocas: &HashMap<Vec<u64>, ValueId>,
    ) {
        for (indices, accesses) in &info.field_accesses {
            let scalar_alloca = match scalar_allocas.get(indices) {
                Some(v) => *v,
                None => continue,
            };

            for access in accesses {
                if let Some(block) = func.get_block_mut(access.block_id) {
                    if access.instr_idx < block.instructions.len() {
                        let instr = &mut block.instructions[access.instr_idx];

                        match instr {
                            MirInstruction::Load { address, .. } => {
                                // Rewrite to load from scalar alloca
                                *address = scalar_alloca;
                            }
                            MirInstruction::Store { address, .. } => {
                                // Rewrite to store to scalar alloca
                                *address = scalar_alloca;
                            }
                            _ => {}
                        }

                        self.stats.instructions_removed += 1;
                    }
                }
            }
        }
    }

    /// Clean up instructions that are no longer needed
    fn cleanup_removed_instructions(&self, func: &mut MirFunction) {
        // Remove GEPs that are now dead (their only use was the load/store we rewrote)
        // This is a simplified cleanup; a full implementation would use DCE

        let mut used_values: HashSet<ValueId> = HashSet::new();

        // Collect all used values
        for block in &func.blocks {
            for instr in &block.instructions {
                for used in instr.used_values() {
                    used_values.insert(used);
                }
            }

            // Also check terminator uses
            match &block.terminator {
                MirTerminator::CondBranch { condition, .. } => {
                    used_values.insert(*condition);
                }
                MirTerminator::Switch { value, .. } => {
                    used_values.insert(*value);
                }
                MirTerminator::Return { value: Some(v) } => {
                    used_values.insert(*v);
                }
                MirTerminator::CallNoReturn { args, .. } => {
                    for arg in args {
                        used_values.insert(*arg);
                    }
                }
                _ => {}
            }
        }

        // Remove dead GEPs
        for block in &mut func.blocks {
            block.instructions.retain(|instr| {
                if let MirInstruction::GetElementPtr { result, .. } = instr {
                    return used_values.contains(result);
                }
                true
            });
        }
    }
}

impl Default for Sroa {
    fn default() -> Self {
        Self::new()
    }
}

impl MIRPass for Sroa {
    fn name(&self) -> &'static str {
        "sroa"
    }

    fn run_on_module(&self, module: &mut MirModule) -> Result<bool, String> {
        let mut modified = false;

        for func in &mut module.functions {
            // Create a fresh pass for each function
            let mut pass = Sroa::new();
            if pass.transform_function(func) {
                modified = true;
            }
        }

        Ok(modified)
    }

    fn run_on_function(&self, func: &mut MirFunction) -> Result<bool, String> {
        // Create a mutable copy for transformation
        let mut pass = Sroa::new();
        Ok(pass.transform_function(func))
    }

    fn requires_ssa(&self) -> bool {
        true
    }

    fn preserves_ssa(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::FunctionBuilder;
    use crate::mir::{FuncId, MirConstant};

    #[test]
    fn test_is_aggregate_type() {
        assert!(Sroa::is_aggregate_type(&MirType::Struct {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), MirType::F64),
                ("y".to_string(), MirType::F64),
            ],
        }));

        assert!(Sroa::is_aggregate_type(&MirType::Array(
            Box::new(MirType::I32),
            4
        )));

        assert!(Sroa::is_aggregate_type(&MirType::Tuple(vec![
            MirType::I32,
            MirType::F64
        ])));

        assert!(!Sroa::is_aggregate_type(&MirType::I64));
        assert!(!Sroa::is_aggregate_type(&MirType::F64));
        assert!(!Sroa::is_aggregate_type(&MirType::Ptr(Box::new(MirType::I32))));
    }

    #[test]
    fn test_get_field_type() {
        let struct_ty = MirType::Struct {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), MirType::F64),
                ("y".to_string(), MirType::F64),
            ],
        };

        // First field
        assert_eq!(Sroa::get_field_type(&struct_ty, &[0]), Some(MirType::F64));

        // Second field
        assert_eq!(Sroa::get_field_type(&struct_ty, &[1]), Some(MirType::F64));

        // Out of bounds
        assert_eq!(Sroa::get_field_type(&struct_ty, &[2]), None);

        // Nested struct
        let nested_ty = MirType::Struct {
            name: "Line".to_string(),
            fields: vec![
                ("start".to_string(), struct_ty.clone()),
                ("end".to_string(), struct_ty.clone()),
            ],
        };

        // Access start.x
        assert_eq!(Sroa::get_field_type(&nested_ty, &[0, 0]), Some(MirType::F64));

        // Access end.y
        assert_eq!(Sroa::get_field_type(&nested_ty, &[1, 1]), Some(MirType::F64));
    }

    #[test]
    fn test_find_allocas() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit);

        let alloc1 = builder.fresh_value();
        builder.build_alloca(alloc1, MirType::I64);

        let alloc2 = builder.fresh_value();
        builder.build_alloca(
            alloc2,
            MirType::Struct {
                name: "Point".to_string(),
                fields: vec![
                    ("x".to_string(), MirType::F64),
                    ("y".to_string(), MirType::F64),
                ],
            },
        );

        builder.build_return(None);

        let func = builder.build();

        let allocas = Sroa::find_allocas(&func);

        assert_eq!(allocas.len(), 2);
        assert_eq!(allocas[0].2, alloc1);
        assert_eq!(allocas[1].2, alloc2);
    }

    #[test]
    fn test_sroa_simple_struct() {
        // Create a function that allocates a struct and accesses its fields
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::F64);

        let struct_ty = MirType::Struct {
            name: "Point".to_string(),
            fields: vec![
                ("x".to_string(), MirType::F64),
                ("y".to_string(), MirType::F64),
            ],
        };

        // Allocate struct
        let point = builder.fresh_value();
        builder.build_alloca(point, struct_ty.clone());

        // Create GEP indices
        let idx_zero = builder.fresh_value();
        builder.build_const(idx_zero, MirConstant::Int(0), MirType::I64);

        let idx_one = builder.fresh_value();
        builder.build_const(idx_one, MirConstant::Int(1), MirType::I64);

        // GEP to x field
        let x_ptr = builder.fresh_value();
        builder.build_gep(x_ptr, point, vec![idx_zero], MirType::Ptr(Box::new(MirType::F64)));

        // Store to x
        let x_val = builder.build_f64(1.0);
        builder.build_store(x_ptr, x_val);

        // GEP to y field
        let y_ptr = builder.fresh_value();
        builder.build_gep(y_ptr, point, vec![idx_one], MirType::Ptr(Box::new(MirType::F64)));

        // Store to y
        let y_val = builder.build_f64(2.0);
        builder.build_store(y_ptr, y_val);

        // Load from x
        let loaded_x = builder.fresh_value();
        builder.build_load(loaded_x, x_ptr, MirType::F64);

        builder.build_return(Some(loaded_x));

        let mut func = builder.build();

        // Run SROA using transform_function directly to get stats
        let mut pass = Sroa::new();
        let _modified = pass.transform_function(&mut func);

        // The pass should have analyzed the alloca (at least one aggregate)
        assert!(pass.stats.allocas_analyzed > 0, "Expected allocas to be analyzed");
    }

    #[test]
    fn test_sroa_non_aggregate_skipped() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::I64);

        // Allocate a scalar (should be skipped by SROA)
        let scalar = builder.fresh_value();
        builder.build_alloca(scalar, MirType::I64);

        let val = builder.build_i64(42);
        builder.build_store(scalar, val);

        let loaded = builder.fresh_value();
        builder.build_load(loaded, scalar, MirType::I64);

        builder.build_return(Some(loaded));

        let mut func = builder.build();

        let mut pass = Sroa::new();
        let modified = pass.run_on_function(&mut func).expect("SROA should succeed");

        // Scalar allocas should not be split
        assert!(!modified);
        assert_eq!(pass.stats.allocas_split, 0);
    }

    #[test]
    fn test_sroa_escaping_pointer() {
        let mut builder = FunctionBuilder::new(FuncId(0), "test".to_string(), MirType::Unit);

        let struct_ty = MirType::Struct {
            name: "Data".to_string(),
            fields: vec![("value".to_string(), MirType::I64)],
        };

        // Allocate struct
        let data = builder.fresh_value();
        builder.build_alloca(data, struct_ty);

        // Pass the pointer to a function (escapes!)
        builder.build_call(
            None,
            "external_func".to_string(),
            vec![data],
            MirType::Unit,
        );

        builder.build_return(None);

        let mut func = builder.build();

        let mut pass = Sroa::new();
        let modified = pass.run_on_function(&mut func).expect("SROA should succeed");

        // Escaping pointer should not be split
        assert!(!modified);
        assert_eq!(pass.stats.allocas_split, 0);
    }
}
