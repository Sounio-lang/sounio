//! HLIR to SIR Lowering
//!
//! Transforms high-level HLIR into low-level SIR while preserving
//! domain-specific information for optimization.
//!
//! # Lowering Strategy
//!
//! 1. **Type Lowering**: Sounio types -> SIR types
//!    - Primitives map directly
//!    - Structs get explicit layouts
//!    - Epistemic<T> -> struct { T, f64 }
//!    - Refinement types -> base type + metadata
//!
//! 2. **Instruction Lowering**: HLIR ops -> SIR ops
//!    - Most ops map directly
//!    - Effect handlers are lowered to explicit control flow
//!    - Epistemic ops get special SIR instructions
//!
//! 3. **Metadata Preservation**: Domain info -> SIR metadata
//!    - Confidence bounds
//!    - Provenance
//!    - Physical units
//!    - Refinement predicates

use crate::hlir::{
    BinaryOp, BlockId as HlirBlockId, HlirBlock, HlirConstant, HlirFunction, HlirGlobal, HlirInstr,
    HlirModule, HlirTerminator, HlirType, HlirTypeDef, HlirTypeDefKind, Op, UnaryOp,
    ValueId as HlirValueId,
};

use super::blocks::{BasicBlock, Instruction, Linkage, Terminator};
use super::metadata::{EpistemicMetadata, Metadata, MetadataStore};
use super::module::{ExternFunction, SirGlobal, SirModule};
use super::ops::{
    ArithOp, CallInfo, Callee, CastOp, CmpOp, EpistemicOp, GepIndex, MemoryOp, SirInst,
    UnaryFloatOp, UnaryIntOp,
};
use super::types::{ArrayType, CallingConv, ScalarType, SirType, StructType};
use super::values::{
    BlockId, Constant, FuncId, PhysicalUnit, Value, ValueBounds, ValueId, ValueMetadata,
};

use std::collections::HashMap;

/// Lowering context that tracks mappings during the lowering process
pub struct LoweringContext {
    /// HLIR value -> SIR value mapping
    value_map: HashMap<HlirValueId, ValueId>,
    /// HLIR block -> SIR block mapping
    block_map: HashMap<HlirBlockId, BlockId>,
    /// Type cache for efficiency
    type_cache: HashMap<HlirType, SirType>,
    /// Next SIR value ID
    next_value_id: u32,
    /// Next SIR block ID
    next_block_id: u32,
    /// Function name -> FuncId mapping for call resolution
    func_map: HashMap<String, FuncId>,
    /// Metadata store for domain-specific information
    pub metadata: MetadataStore,
    /// Value type cache (for looking up types during lowering)
    value_types: HashMap<ValueId, SirType>,
    /// Epistemic tracking: values with associated confidence
    epistemic_values: HashMap<ValueId, EpistemicInfo>,
    /// Metadata tracker for Phase 2 - captures domain-specific information
    pub metadata_tracker: MetadataTracker,
}

/// Information about epistemic values during lowering
#[derive(Debug, Clone)]
struct EpistemicInfo {
    /// The confidence value (if separate)
    confidence: Option<ValueId>,
    /// Known confidence bound
    known_confidence: Option<f64>,
    /// Has uncertainty tracking
    has_uncertainty: bool,
}

/// Metadata tracker for Phase 2 - captures domain-specific information during lowering
#[derive(Debug, Clone)]
pub struct MetadataTracker {
    /// Epistemic info: value -> (epsilon_bound, provenance_id)
    epistemic_info: HashMap<ValueId, (f64, Option<u32>)>,
    /// Physical units: value -> unit
    unit_info: HashMap<ValueId, PhysicalUnit>,
    /// Field-level metadata: (struct_value, field_index) -> metadata
    field_metadata: HashMap<(ValueId, usize), Vec<Metadata>>,
}

impl MetadataTracker {
    pub fn new() -> Self {
        Self {
            epistemic_info: HashMap::new(),
            unit_info: HashMap::new(),
            field_metadata: HashMap::new(),
        }
    }

    /// Record epistemic information for a value
    pub fn record_epistemic(&mut self, value: ValueId, epsilon_bound: f64, provenance_id: Option<u32>) {
        self.epistemic_info.insert(value, (epsilon_bound, provenance_id));
    }

    /// Record physical unit for a value
    pub fn record_unit(&mut self, value: ValueId, unit: PhysicalUnit) {
        self.unit_info.insert(value, unit);
    }

    /// Record field-level metadata
    pub fn record_field_metadata(&mut self, value: ValueId, field_index: usize, metadata: Metadata) {
        self.field_metadata
            .entry((value, field_index))
            .or_default()
            .push(metadata);
    }

    /// Extract Knowledge<T> wrapper information from a type
    pub fn extract_knowledge_wrapper(ty: &HlirType) -> Option<(f64, Option<u32>)> {
        match ty {
            HlirType::Knowledge { inner, mode, .. } => {
                use crate::runtime::EpistemicMode;
                // Extract epsilon bound based on mode
                // Epsilon = 1 - confidence, so 95% confidence = 0.05 epsilon
                let epsilon_bound = match mode {
                    EpistemicMode::Full | EpistemicMode::Compact => 0.05, // 95% confidence
                    EpistemicMode::Erased => return None, // No runtime tracking
                };
                
                // Provenance ID only tracked in Full mode
                let provenance_id = match mode {
                    EpistemicMode::Full => Some(0), // TODO: extract actual provenance from metadata
                    _ => None,
                };
                
                Some((epsilon_bound, provenance_id))
            }
            // Fallback for old-style struct names (backward compatibility)
            HlirType::Struct(name) if name.contains("Knowledge") => {
                Some((0.05, None)) // Default 95% confidence
            }
            _ => None,
        }
    }

    /// Extract Quantity<T> wrapper information from a type
    pub fn extract_quantity_wrapper(ty: &HlirType) -> Option<PhysicalUnit> {
        match ty {
            HlirType::Struct(name) if name.starts_with("Quantity_") => {
                // Extract unit string from struct name
                // Format: Quantity_<unit_str> where operators are encoded as _DIV_, _MUL_, _POW_
                let unit_encoded = name.strip_prefix("Quantity_")?;
                
                // Decode operators back to standard form
                let unit_str = unit_encoded
                    .replace("_DIV_", "/")
                    .replace("_MUL_", "*")
                    .replace("_POW_", "^");
                
                // Parse the unit string
                Self::parse_unit_string(&unit_str)
            }
            _ => None,
        }
    }

    /// Parse a unit string and return PhysicalUnit
    /// Handles simple units like "mg", "L", "h"
    /// Handles compound units like "mg/L", "L/h", "mg/h"
    fn parse_unit_string(unit_str: &str) -> Option<PhysicalUnit> {
        use crate::units::convert::parse_unit_expression;
        
        // Split by division operator
        if let Some(div_pos) = unit_str.find('/') {
            let (numerator, denominator) = unit_str.split_at(div_pos);
            let denominator = &denominator[1..]; // Skip the '/' character
            
            // Parse numerator and denominator separately
            let (num_dim, num_scale) = parse_unit_expression(numerator).ok()?;
            let (den_dim, den_scale) = parse_unit_expression(denominator).ok()?;
            
            // Divide dimensions (subtract exponents)
            let result_dim = num_dim.div(&den_dim);
            // Divide scales
            let result_scale = num_scale / den_scale;
            
            Some(PhysicalUnit {
                dimensions: Self::dimension_to_array(&result_dim),
                scale: result_scale,
            })
        } else {
            // Simple unit
            let (dim, scale) = parse_unit_expression(unit_str).ok()?;
            Some(PhysicalUnit {
                dimensions: Self::dimension_to_array(&dim),
                scale,
            })
        }
    }

    /// Convert Dimension struct to 7-element array [mass, length, time, current, temp, amount, luminosity]
    fn dimension_to_array(dim: &crate::units::dimension::Dimension) -> [i8; 7] {
        [
            dim.mass,
            dim.length,
            dim.time,
            dim.current,
            dim.temperature,
            dim.amount,
            dim.luminosity,
        ]
    }


    /// Finalize and attach collected metadata to the module's metadata store
    pub fn finalize_metadata(self, metadata_store: &mut MetadataStore) {
        for (value, (epsilon_bound, provenance_id)) in self.epistemic_info {
            let epistemic = EpistemicMetadata {
                known_confidence: Some(epsilon_bound),
                confidence_invariant: false,
                certain: epsilon_bound >= 0.9999,
                provenance_id,
            };
            metadata_store.attach(value, Metadata::Epistemic(epistemic));
        }

        for (value, unit) in self.unit_info {
            metadata_store.attach(value, Metadata::Unit(unit));
        }

        for ((value, _field_idx), field_metas) in self.field_metadata {
            for meta in field_metas {
                metadata_store.attach(value, meta);
            }
        }
    }
}

impl Default for MetadataTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl LoweringContext {
    pub fn new() -> Self {
        Self {
            value_map: HashMap::new(),
            block_map: HashMap::new(),
            type_cache: HashMap::new(),
            next_value_id: 0,
            next_block_id: 0,
            func_map: HashMap::new(),
            metadata: MetadataStore::new(),
            value_types: HashMap::new(),
            epistemic_values: HashMap::new(),
            metadata_tracker: MetadataTracker::new(),
        }
    }

    /// Allocate a new SIR value ID
    fn alloc_value(&mut self, ty: SirType) -> ValueId {
        let id = ValueId::new(self.next_value_id);
        self.next_value_id += 1;
        self.value_types.insert(id, ty);
        id
    }

    /// Allocate a new SIR block ID
    fn alloc_block(&mut self) -> BlockId {
        let id = BlockId::new(self.next_block_id);
        self.next_block_id += 1;
        id
    }

    /// Map an HLIR value to a SIR value
    fn map_value(&mut self, hlir: HlirValueId, sir: ValueId) {
        self.value_map.insert(hlir, sir);
    }

    /// Get the SIR value for an HLIR value
    fn get_value(&self, hlir: HlirValueId) -> Option<ValueId> {
        self.value_map.get(&hlir).copied()
    }

    /// Get or create value mapping (for forward references)
    fn get_or_create_value(&mut self, hlir: HlirValueId, ty: SirType) -> ValueId {
        if let Some(v) = self.get_value(hlir) {
            v
        } else {
            let v = self.alloc_value(ty);
            self.map_value(hlir, v);
            v
        }
    }

    /// Map an HLIR block to a SIR block
    fn map_block(&mut self, hlir: HlirBlockId, sir: BlockId) {
        self.block_map.insert(hlir, sir);
    }

    /// Get the SIR block for an HLIR block
    fn get_block(&self, hlir: HlirBlockId) -> Option<BlockId> {
        self.block_map.get(&hlir).copied()
    }

    /// Get the type of a SIR value
    fn get_value_type(&self, val: ValueId) -> SirType {
        self.value_types.get(&val).cloned().unwrap_or(SirType::Void)
    }

    /// Attach epistemic metadata to a value
    fn attach_epistemic(&mut self, value: ValueId, info: EpistemicInfo) {
        let meta = EpistemicMetadata {
            known_confidence: info.known_confidence,
            confidence_invariant: false,
            certain: info.known_confidence == Some(1.0),
            provenance_id: None,
        };
        self.metadata.attach(value, Metadata::Epistemic(meta));
        self.epistemic_values.insert(value, info);
    }

    /// Attach value bounds metadata
    fn attach_bounds(&mut self, value: ValueId, bounds: ValueBounds) {
        let mut value_meta = ValueMetadata::default();
        value_meta.value_bounds = Some(bounds);
        // Store in metadata store would require extending - for now we track separately
    }

    /// Attach physical unit metadata
    fn attach_unit(&mut self, value: ValueId, unit: PhysicalUnit) {
        self.metadata.attach(value, Metadata::Unit(unit));
    }

    /// Reset state for a new function
    fn reset_for_function(&mut self) {
        self.value_map.clear();
        self.block_map.clear();
        self.value_types.clear();
        // Reset IDs so each function has ValueIds starting from 0
        self.next_value_id = 0;
        self.next_block_id = 0;
        // Keep type_cache and func_map across functions
        // Keep metadata_tracker across functions - it accumulates metadata
    }
}

impl Default for LoweringContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Lower an HLIR module to SIR
/// Returns both the module and the lowering context (which contains metadata)
pub fn lower_module(hlir: &HlirModule) -> (SirModule, LoweringContext, MetadataStore) {
    let mut ctx = LoweringContext::new();
    let mut module = SirModule::new(&hlir.name);

    // Step 1: Lower all type definitions
    for type_def in &hlir.types {
        lower_type_def(&mut ctx, &mut module, type_def);
    }

    // Step 2: Lower global variables
    for global in &hlir.globals {
        lower_global(&mut ctx, &mut module, global);
    }

    // Step 3: Create function declarations first (for forward references)
    for (idx, func) in hlir.functions.iter().enumerate() {
        let func_id = FuncId(idx as u32);
        ctx.func_map.insert(func.name.clone(), func_id);

        // If this is an external function without blocks, add as extern
        if func.blocks.is_empty() {
            let params: Vec<SirType> = func.params.iter().map(|p| lower_type(&p.ty)).collect();
            let ret_ty = lower_type(&func.return_type);
            let ext = ExternFunction::new(&func.name, params, ret_ty);
            module.add_extern(ext);
        }
    }

    // Step 4: Lower function bodies
    for func in &hlir.functions {
        if !func.blocks.is_empty() {
            lower_function(&mut ctx, &mut module, func);
        }
    }

    // Step 5: Finalize metadata - attach all collected metadata to the store
    let metadata_tracker = ctx.metadata_tracker.clone();
    metadata_tracker.finalize_metadata(&mut ctx.metadata);

    // Step 6: Populate handler IDs for unified effect dispatch
    populate_handler_ids(&mut module);

    let final_metadata = ctx.metadata.clone();
    (module, ctx, final_metadata)
}

/// Lower a type definition to SIR
fn lower_type_def(ctx: &mut LoweringContext, module: &mut SirModule, type_def: &HlirTypeDef) {
    match &type_def.kind {
        HlirTypeDefKind::Struct(fields) => {
            let sir_fields: Vec<(Option<String>, SirType)> = fields
                .iter()
                .map(|(name, ty)| (Some(name.clone()), lower_type(ty)))
                .collect();
            let struct_ty = StructType::new(sir_fields).named(&type_def.name);
            module.add_type(&type_def.name, SirType::Struct(struct_ty));
        }
        HlirTypeDefKind::Enum(variants) => {
            // Enums are lowered to a tagged union
            // For now, create a struct with tag + payload
            let tag_field = (Some("tag".to_string()), SirType::i32());

            // Find the largest variant for union size
            let max_size: usize = variants
                .iter()
                .map(|(_, tys)| tys.iter().map(|t| lower_type(t).size_bytes()).sum())
                .max()
                .unwrap_or(0);

            // Payload is an array of bytes (simplified)
            let payload_ty = SirType::Array(ArrayType::new(
                SirType::Scalar(ScalarType::I8),
                max_size.max(1),
            ));
            let payload_field = (Some("payload".to_string()), payload_ty);

            let struct_ty = StructType::new(vec![tag_field, payload_field]).named(&type_def.name);
            module.add_type(&type_def.name, SirType::Struct(struct_ty));
        }
    }
}

/// Lower a global variable to SIR
fn lower_global(ctx: &mut LoweringContext, module: &mut SirModule, global: &HlirGlobal) {
    let ty = lower_type(&global.ty);
    let initializer = global.init.as_ref().map(|c| lower_constant(ctx, c));

    let mut sir_global = SirGlobal::new(&global.name, ty);
    if let Some(init) = initializer {
        sir_global = sir_global.with_initializer(init);
    }
    if global.is_const {
        sir_global = sir_global.constant();
    }

    module.add_global(sir_global);
}

/// Lower a function to SIR
fn lower_function(ctx: &mut LoweringContext, module: &mut SirModule, func: &HlirFunction) {
    ctx.reset_for_function();

    // Create SIR function signature
    let params: Vec<(String, SirType)> = func
        .params
        .iter()
        .map(|p| (p.name.clone(), lower_type(&p.ty)))
        .collect();
    let ret_ty = lower_type(&func.return_type);

    let func_id = module.create_function(&func.name, params.clone(), ret_ty.clone());

    // Create parameter values and map them
    for (idx, param) in func.params.iter().enumerate() {
        let sir_val = ctx.alloc_value(lower_type(&param.ty));
        ctx.map_value(param.value, sir_val);
    }

    // Pre-create all blocks for forward references
    for block in &func.blocks {
        let sir_block_id = ctx.alloc_block();
        ctx.map_block(block.id, sir_block_id);
    }

    // Lower each block
    let mut sir_blocks = Vec::new();
    for block in &func.blocks {
        let sir_block = lower_block(ctx, block, &func.locals);
        sir_blocks.push(sir_block);
    }

    // Add blocks to function
    if let Some(sir_func) = module.function_mut(func_id) {
        sir_func.blocks = sir_blocks;

        // Set function attributes
        if func.is_exported {
            sir_func.linkage = Linkage::External;
        } else {
            sir_func.linkage = Linkage::Internal;
        }

        // Set calling convention based on ABI
        sir_func.cc = match func.abi {
            crate::ast::Abi::Rust => CallingConv::Default,
            crate::ast::Abi::C | crate::ast::Abi::CUnwind => CallingConv::C,
            crate::ast::Abi::System | crate::ast::Abi::SystemUnwind => CallingConv::Default,
            // All other ABIs (Stdcall, Fastcall, Vectorcall, Aapcs, SysV64, Win64)
            _ => CallingConv::C,
        };
    }
}

/// Lower a basic block to SIR
fn lower_block(
    ctx: &mut LoweringContext,
    block: &HlirBlock,
    locals: &HashMap<HlirValueId, HlirType>,
) -> BasicBlock {
    let sir_block_id = ctx.get_block(block.id).expect("Block not pre-created");

    let mut sir_block = BasicBlock::new(sir_block_id);
    if !block.label.is_empty() {
        sir_block = sir_block.named(&block.label);
    }

    // Lower block parameters (for phi nodes)
    let params: Vec<Value> = block
        .params
        .iter()
        .map(|(val_id, ty)| {
            let sir_ty = lower_type(ty);
            let sir_val = ctx.get_or_create_value(*val_id, sir_ty.clone());
            Value::new(sir_val, sir_ty)
        })
        .collect();
    sir_block = sir_block.with_params(params);

    // Lower instructions
    for instr in &block.instructions {
        if let Some(inst) = lower_instruction(ctx, instr, locals) {
            sir_block.push(inst);
        }
    }

    // Lower terminator
    let term = lower_terminator(ctx, &block.terminator);
    sir_block.terminate(term);

    sir_block
}

/// Lower an instruction to SIR
fn lower_instruction(
    ctx: &mut LoweringContext,
    instr: &HlirInstr,
    locals: &HashMap<HlirValueId, HlirType>,
) -> Option<Instruction> {
    let result_ty = lower_type(&instr.ty);

    match &instr.op {
        Op::Const(constant) => {
            let sir_const = lower_constant(ctx, constant);
            let result = instr.result.map(|r| ctx.get_or_create_value(r, result_ty));
            result.map(|r| Instruction::with_result(r, SirInst::Const(sir_const)))
        }

        Op::Copy(val) => {
            // Copy is a no-op in SSA - just update the mapping
            if let (Some(result), Some(src)) = (instr.result, ctx.get_value(*val)) {
                ctx.map_value(result, src);
            }
            None
        }

        Op::Binary { op, left, right } => {
            let lhs = ctx.get_value(*left)?;
            let rhs = ctx.get_value(*right)?;
            let sir_op = lower_binary_op(*op, &instr.ty);
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            match sir_op {
                LoweredBinOp::Arith(arith_op) => Some(Instruction::with_result(
                    result,
                    SirInst::BinOp {
                        op: arith_op,
                        lhs,
                        rhs,
                    },
                )),
                LoweredBinOp::Cmp(cmp_op) => Some(Instruction::with_result(
                    result,
                    SirInst::Cmp {
                        op: cmp_op,
                        lhs,
                        rhs,
                    },
                )),
            }
        }

        Op::Unary { op, operand } => {
            let val = ctx.get_value(*operand)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let sir_inst = match op {
                UnaryOp::Neg => {
                    if instr.ty.is_float() {
                        SirInst::UnaryFloat {
                            op: UnaryFloatOp::Neg,
                            val,
                        }
                    } else {
                        // Integer negation using two's complement
                        SirInst::UnaryInt {
                            op: UnaryIntOp::Neg,
                            val,
                        }
                    }
                }
                UnaryOp::FNeg => SirInst::UnaryFloat {
                    op: UnaryFloatOp::Neg,
                    val,
                },
                UnaryOp::Not => {
                    // Bitwise not (one's complement)
                    SirInst::UnaryInt {
                        op: UnaryIntOp::Not,
                        val,
                    }
                }
            };

            Some(Instruction::with_result(result, sir_inst))
        }

        Op::Call { func: _, args } | Op::CallDirect { name: _, args } => {
            // Check for builtin intrinsics first
            if let Op::CallDirect { name, .. } = &instr.op {
                // Handle slice_from_raw_parts intrinsic
                // Creates a slice (fat pointer) from ptr and len
                if name == "__builtin_slice_from_raw_parts" || name == "std::slice::from_raw_parts"
                {
                    if args.len() == 2 {
                        let ptr = ctx.get_value(args[0])?;
                        let len = ctx.get_value(args[1])?;
                        let result = instr
                            .result
                            .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

                        // Build slice struct: { ptr, len }
                        return Some(Instruction::with_result(
                            result,
                            SirInst::BuildAggregate {
                                fields: vec![ptr, len],
                                ty: result_ty,
                            },
                        ));
                    }
                }
                // Handle mutable slice variant
                if name == "__builtin_slice_from_raw_parts_mut"
                    || name == "std::slice::from_raw_parts_mut"
                {
                    if args.len() == 2 {
                        let ptr = ctx.get_value(args[0])?;
                        let len = ctx.get_value(args[1])?;
                        let result = instr
                            .result
                            .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

                        return Some(Instruction::with_result(
                            result,
                            SirInst::BuildAggregate {
                                fields: vec![ptr, len],
                                ty: result_ty,
                            },
                        ));
                    }
                }
            }

            let sir_args: Vec<ValueId> = args.iter().filter_map(|a| ctx.get_value(*a)).collect();

            let callee = match &instr.op {
                Op::Call { func, .. } => {
                    if let Some(func_val) = ctx.get_value(*func) {
                        Callee::Indirect(func_val)
                    } else {
                        return None;
                    }
                }
                Op::CallDirect { name, .. } => {
                    if let Some(&func_id) = ctx.func_map.get(name) {
                        Callee::Direct(func_id)
                    } else {
                        Callee::Named(name.clone())
                    }
                }
                _ => unreachable!(),
            };

            let call_info = CallInfo {
                callee,
                args: sir_args,
                ret_ty: result_ty.clone(),
                cc: CallingConv::Default,
                tail: false,
                nounwind: false,
            };

            let result = instr.result.map(|r| ctx.get_or_create_value(r, result_ty));
            if let Some(r) = result {
                Some(Instruction::with_result(r, SirInst::Call(call_info)))
            } else {
                Some(Instruction::void(SirInst::Call(call_info)))
            }
        }

        Op::Load { ptr } => {
            let ptr_val = ctx.get_value(*ptr)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            Some(Instruction::with_result(
                result,
                SirInst::Memory(MemoryOp::Load {
                    ptr: ptr_val,
                    ty: result_ty,
                    volatile: false,
                    align: None,
                }),
            ))
        }

        Op::Store { ptr, value } => {
            let ptr_val = ctx.get_value(*ptr)?;
            let val = ctx.get_value(*value)?;

            Some(Instruction::void(SirInst::Memory(MemoryOp::Store {
                ptr: ptr_val,
                val,
                volatile: false,
                align: None,
            })))
        }

        Op::GetFieldPtr { base, field } => {
            let base_val = ctx.get_value(*base)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            // Determine the base type for GEP
            let base_ty = ctx.get_value_type(base_val);
            let pointee_ty = match &base_ty {
                SirType::Pointer(ptr) => (*ptr.pointee).clone(),
                _ => SirType::Void,
            };

            Some(Instruction::with_result(
                result,
                SirInst::Memory(MemoryOp::GetElementPtr {
                    ptr: base_val,
                    ty: pointee_ty,
                    indices: vec![GepIndex::Const(0), GepIndex::Const(*field as i64)],
                }),
            ))
        }

        Op::GetElementPtr { base, index } => {
            let base_val = ctx.get_value(*base)?;
            let idx_val = ctx.get_value(*index)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let base_ty = ctx.get_value_type(base_val);
            let pointee_ty = match &base_ty {
                SirType::Pointer(ptr) => (*ptr.pointee).clone(),
                _ => SirType::Void,
            };

            Some(Instruction::with_result(
                result,
                SirInst::Memory(MemoryOp::GetElementPtr {
                    ptr: base_val,
                    ty: pointee_ty,
                    indices: vec![GepIndex::Value(idx_val)],
                }),
            ))
        }

        Op::Alloca { ty } => {
            let alloc_ty = lower_type(ty);
            let ptr_ty = SirType::ptr(alloc_ty.clone());
            let result = instr.result.map(|r| ctx.get_or_create_value(r, ptr_ty))?;

            Some(Instruction::with_result(
                result,
                SirInst::Memory(MemoryOp::Alloca {
                    ty: alloc_ty,
                    count: None,
                    align: None,
                }),
            ))
        }

        Op::Cast {
            value,
            source,
            target,
        } => {
            let val = ctx.get_value(*value)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let cast_op = determine_cast_op(source, target);
            let to_ty = lower_type(target);

            Some(Instruction::with_result(
                result,
                SirInst::Cast {
                    op: cast_op,
                    val,
                    to_ty,
                },
            ))
        }

        Op::Phi { incoming } => {
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let sir_incoming: Vec<(BlockId, ValueId)> = incoming
                .iter()
                .filter_map(|(block, val)| {
                    let sir_block = ctx.get_block(*block)?;
                    let sir_val = ctx.get_value(*val)?;
                    Some((sir_block, sir_val))
                })
                .collect();

            Some(Instruction::with_result(
                result,
                SirInst::Phi {
                    ty: result_ty,
                    incoming: sir_incoming,
                },
            ))
        }

        Op::ExtractValue { base, index } => {
            let base_val = ctx.get_value(*base)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            // Lower to GEP + Load for struct access
            let base_ty = ctx.get_value_type(base_val);

            // First create a GEP
            let ptr_ty = SirType::ptr(result_ty.clone());
            let gep_result = ctx.alloc_value(ptr_ty);

            // This is simplified - in practice we'd need to handle aggregates properly
            Some(Instruction::with_result(
                result,
                SirInst::Memory(MemoryOp::Load {
                    ptr: base_val, // Simplified
                    ty: result_ty,
                    volatile: false,
                    align: None,
                }),
            ))
        }

        Op::InsertValue { base, value, index } => {
            // Simplified handling - in practice needs full aggregate support
            let base_val = ctx.get_value(*base)?;
            let val = ctx.get_value(*value)?;
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            // For now, just return the base (incomplete implementation)
            None
        }

        Op::Tuple(values) => {
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let sir_values: Vec<ValueId> =
                values.iter().filter_map(|v| ctx.get_value(*v)).collect();

            for (field_idx, (hlir_val, sir_val)) in values.iter().zip(sir_values.iter()).enumerate() {
                if let Some((epsilon_bound, provenance_id)) = 
                    MetadataTracker::extract_knowledge_wrapper(&locals.get(hlir_val).map(|t| t.clone()).unwrap_or(HlirType::Void)) {
                    ctx.metadata_tracker.record_epistemic(*sir_val, epsilon_bound, provenance_id);
                }
                if let Some(unit) = 
                    MetadataTracker::extract_quantity_wrapper(&locals.get(hlir_val).map(|t| t.clone()).unwrap_or(HlirType::Void)) {
                    ctx.metadata_tracker.record_unit(*sir_val, unit);
                }
            }

            Some(Instruction::with_result(
                result,
                SirInst::BuildAggregate {
                    fields: sir_values,
                    ty: result_ty,
                },
            ))
        }

        Op::Array(values) => {
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let sir_values: Vec<ValueId> =
                values.iter().filter_map(|v| ctx.get_value(*v)).collect();

            Some(Instruction::with_result(
                result,
                SirInst::BuildAggregate {
                    fields: sir_values,
                    ty: result_ty,
                },
            ))
        }

        Op::Struct { name, fields } => {
            let result = instr
                .result
                .map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

            let sir_values: Vec<ValueId> =
                fields.iter().filter_map(|(_field_name, v)| ctx.get_value(*v)).collect();

            for (field_idx, (_field_name, hlir_field_val)) in fields.iter().enumerate() {
                if let Some(sir_val) = ctx.get_value(*hlir_field_val) {
                    if let Some((epsilon_bound, provenance_id)) = 
                        MetadataTracker::extract_knowledge_wrapper(&locals.get(hlir_field_val).map(|t| t.clone()).unwrap_or(HlirType::Void)) {
                        ctx.metadata_tracker.record_epistemic(sir_val, epsilon_bound, provenance_id);
                    }
                    if let Some(unit) = 
                        MetadataTracker::extract_quantity_wrapper(&locals.get(hlir_field_val).map(|t| t.clone()).unwrap_or(HlirType::Void)) {
                        ctx.metadata_tracker.record_unit(sir_val, unit);
                    }
                }
            }

            Some(Instruction::with_result(
                result,
                SirInst::BuildAggregate {
                    fields: sir_values,
                    ty: result_ty,
                },
            ))
        }

        Op::PerformEffect { effect, op, args } => {
            // Effects are lowered to explicit calls or intrinsics
            // This is domain-specific handling

            let sir_args: Vec<ValueId> = args.iter().filter_map(|a| ctx.get_value(*a)).collect();

            // Handle epistemic effects specially
            if effect == "Epistemic" || effect == "epistemic" {
                return lower_epistemic_effect(ctx, op, &sir_args, instr.result, &result_ty);
            }

            // For other effects, lower to a call to a runtime function
            let call_name = format!("__sounio_effect_{}_{}", effect.to_lowercase(), op);
            let call_info = CallInfo {
                callee: Callee::Named(call_name),
                args: sir_args,
                ret_ty: result_ty.clone(),
                cc: CallingConv::C,
                tail: false,
                nounwind: false,
            };

            let result = instr.result.map(|r| ctx.get_or_create_value(r, result_ty));
            if let Some(r) = result {
                Some(Instruction::with_result(r, SirInst::Call(call_info)))
            } else {
                Some(Instruction::void(SirInst::Call(call_info)))
            }
        }

        Op::PushHandler {
            effect,
            handler_name,
            handler_id,
        } => {
            // Lower to a runtime call that pushes a handler onto the effect stack
            let call_name = format!("__sounio_push_handler_{}", effect.to_lowercase());
            let call_info = CallInfo {
                callee: Callee::Named(call_name),
                args: vec![], // Handler ID is embedded in the call target
                ret_ty: SirType::Void,
                cc: CallingConv::C,
                tail: false,
                nounwind: false,
            };
            Some(Instruction::void(SirInst::Call(call_info)))
        }

        Op::PopHandler => {
            // Lower to a runtime call that pops the topmost handler
            let call_info = CallInfo {
                callee: Callee::Named("__sounio_pop_handler".to_string()),
                args: vec![],
                ret_ty: SirType::Void,
                cc: CallingConv::C,
                tail: false,
                nounwind: false,
            };
            Some(Instruction::void(SirInst::Call(call_info)))
        }

        Op::DispatchEffect { effect, op, args } => {
            // Lower to a runtime call that dispatches through the handler stack
            let sir_args: Vec<ValueId> = args.iter().filter_map(|a| ctx.get_value(*a)).collect();

            let call_name = format!("__sounio_dispatch_{}_{}", effect.to_lowercase(), op);
            let call_info = CallInfo {
                callee: Callee::Named(call_name),
                args: sir_args,
                ret_ty: result_ty.clone(),
                cc: CallingConv::C,
                tail: false,
                nounwind: false,
            };

            let result = instr.result.map(|r| ctx.get_or_create_value(r, result_ty));
            if let Some(r) = result {
                Some(Instruction::with_result(r, SirInst::Call(call_info)))
            } else {
                Some(Instruction::void(SirInst::Call(call_info)))
            }
        }
    }
}

/// Lower epistemic effect operations to SIR epistemic instructions
fn lower_epistemic_effect(
    ctx: &mut LoweringContext,
    op: &str,
    args: &[ValueId],
    result: Option<HlirValueId>,
    result_ty: &SirType,
) -> Option<Instruction> {
    let sir_result = result.map(|r| ctx.get_or_create_value(r, result_ty.clone()))?;

    match op {
        "create" | "Create" => {
            // epistemic.create(value, confidence)
            if args.len() >= 2 {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::Create {
                        value: args[0],
                        confidence: args[1],
                    }),
                ))
            } else {
                None
            }
        }

        "extract_value" | "ExtractValue" => {
            if !args.is_empty() {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::ExtractValue(args[0])),
                ))
            } else {
                None
            }
        }

        "extract_confidence" | "ExtractConfidence" => {
            if !args.is_empty() {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::ExtractConfidence(args[0])),
                ))
            } else {
                None
            }
        }

        "propagate_add" | "PropagateAdd" => {
            if args.len() >= 2 {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::PropagateAdd {
                        conf_a: args[0],
                        conf_b: args[1],
                    }),
                ))
            } else {
                None
            }
        }

        "propagate_mul" | "PropagateMul" => {
            if args.len() >= 4 {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::PropagateMul {
                        val_a: args[0],
                        conf_a: args[1],
                        val_b: args[2],
                        conf_b: args[3],
                    }),
                ))
            } else {
                None
            }
        }

        "meet" | "Meet" => {
            if args.len() >= 2 {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::Meet {
                        conf_a: args[0],
                        conf_b: args[1],
                    }),
                ))
            } else {
                None
            }
        }

        "join" | "Join" => {
            if args.len() >= 2 {
                Some(Instruction::with_result(
                    sir_result,
                    SirInst::Epistemic(EpistemicOp::Join {
                        conf_a: args[0],
                        conf_b: args[1],
                    }),
                ))
            } else {
                None
            }
        }

        _ => {
            // Unknown epistemic operation - lower to a call
            let call_info = CallInfo {
                callee: Callee::Named(format!("__sounio_epistemic_{}", op)),
                args: args.to_vec(),
                ret_ty: result_ty.clone(),
                cc: CallingConv::C,
                tail: false,
                nounwind: false,
            };
            Some(Instruction::with_result(
                sir_result,
                SirInst::Call(call_info),
            ))
        }
    }
}

/// Lower a terminator to SIR
fn lower_terminator(ctx: &LoweringContext, term: &HlirTerminator) -> Terminator {
    match term {
        HlirTerminator::Return(None) => Terminator::Return(None),

        HlirTerminator::Return(Some(val)) => {
            let sir_val = ctx.get_value(*val);
            Terminator::Return(sir_val)
        }

        HlirTerminator::Branch(block) => {
            let sir_block = ctx.get_block(*block).unwrap_or(BlockId::new(0));
            Terminator::Br(sir_block)
        }

        HlirTerminator::CondBranch {
            condition,
            then_block,
            else_block,
        } => {
            let cond = ctx.get_value(*condition).unwrap_or(ValueId::new(0));
            let then_blk = ctx.get_block(*then_block).unwrap_or(BlockId::new(0));
            let else_blk = ctx.get_block(*else_block).unwrap_or(BlockId::new(0));

            Terminator::CondBr {
                cond,
                then_block: then_blk,
                else_block: else_blk,
            }
        }

        HlirTerminator::Switch {
            value,
            default,
            cases,
        } => {
            let val = ctx.get_value(*value).unwrap_or(ValueId::new(0));
            let default_blk = ctx.get_block(*default).unwrap_or(BlockId::new(0));

            let sir_cases: Vec<(i64, BlockId)> = cases
                .iter()
                .filter_map(|(case_val, block)| ctx.get_block(*block).map(|blk| (*case_val, blk)))
                .collect();

            Terminator::Switch {
                val,
                default: default_blk,
                cases: sir_cases,
            }
        }

        HlirTerminator::Unreachable => Terminator::Unreachable,
    }
}

/// Helper enum for lowered binary operations
enum LoweredBinOp {
    Arith(ArithOp),
    Cmp(CmpOp),
}

/// Lower an HLIR binary operator to SIR
fn lower_binary_op(op: BinaryOp, ty: &HlirType) -> LoweredBinOp {
    use LoweredBinOp::*;

    match op {
        // Arithmetic
        BinaryOp::Add => {
            if ty.is_float() {
                Arith(ArithOp::FAdd)
            } else {
                Arith(ArithOp::Add)
            }
        }
        BinaryOp::Sub => {
            if ty.is_float() {
                Arith(ArithOp::FSub)
            } else {
                Arith(ArithOp::Sub)
            }
        }
        BinaryOp::Mul => {
            if ty.is_float() {
                Arith(ArithOp::FMul)
            } else {
                Arith(ArithOp::Mul)
            }
        }
        BinaryOp::FAdd => Arith(ArithOp::FAdd),
        BinaryOp::FSub => Arith(ArithOp::FSub),
        BinaryOp::FMul => Arith(ArithOp::FMul),
        BinaryOp::FDiv => Arith(ArithOp::FDiv),
        BinaryOp::FRem => Arith(ArithOp::FRem),
        BinaryOp::SDiv => Arith(ArithOp::SDiv),
        BinaryOp::UDiv => Arith(ArithOp::UDiv),
        BinaryOp::SRem => Arith(ArithOp::SRem),
        BinaryOp::URem => Arith(ArithOp::URem),

        // Bitwise
        BinaryOp::And => Arith(ArithOp::And),
        BinaryOp::Or => Arith(ArithOp::Or),
        BinaryOp::Xor => Arith(ArithOp::Xor),
        BinaryOp::Shl => Arith(ArithOp::Shl),
        BinaryOp::AShr => Arith(ArithOp::AShr),
        BinaryOp::LShr => Arith(ArithOp::LShr),

        // Integer comparisons
        BinaryOp::Eq => Cmp(CmpOp::Eq),
        BinaryOp::Ne => Cmp(CmpOp::Ne),
        BinaryOp::SLt => Cmp(CmpOp::SLt),
        BinaryOp::SLe => Cmp(CmpOp::SLe),
        BinaryOp::SGt => Cmp(CmpOp::SGt),
        BinaryOp::SGe => Cmp(CmpOp::SGe),
        BinaryOp::ULt => Cmp(CmpOp::ULt),
        BinaryOp::ULe => Cmp(CmpOp::ULe),
        BinaryOp::UGt => Cmp(CmpOp::UGt),
        BinaryOp::UGe => Cmp(CmpOp::UGe),

        // Float comparisons
        BinaryOp::FOEq => Cmp(CmpOp::FOEq),
        BinaryOp::FONe => Cmp(CmpOp::FONe),
        BinaryOp::FOLt => Cmp(CmpOp::FOLt),
        BinaryOp::FOLe => Cmp(CmpOp::FOLe),
        BinaryOp::FOGt => Cmp(CmpOp::FOGt),
        BinaryOp::FOGe => Cmp(CmpOp::FOGe),

        // Array operations - lower to calls
        BinaryOp::Concat => {
            // This would be lowered to a runtime call
            Arith(ArithOp::Add) // Placeholder
        }
    }
}

/// Determine the appropriate cast operation
fn determine_cast_op(source: &HlirType, target: &HlirType) -> CastOp {
    match (source, target) {
        // Integer -> Integer
        (s, t) if s.is_integer() && t.is_integer() => {
            let src_bits = s.size_bits();
            let tgt_bits = t.size_bits();
            if src_bits < tgt_bits {
                if s.is_signed() {
                    CastOp::SExt
                } else {
                    CastOp::ZExt
                }
            } else if src_bits > tgt_bits {
                CastOp::Trunc
            } else {
                CastOp::BitCast
            }
        }

        // Float -> Float
        (HlirType::F32, HlirType::F64) => CastOp::FPExt,
        (HlirType::F64, HlirType::F32) => CastOp::FPTrunc,

        // Integer -> Float
        (s, HlirType::F32 | HlirType::F64) if s.is_integer() => {
            if s.is_signed() {
                CastOp::SIToFP
            } else {
                CastOp::UIToFP
            }
        }

        // Float -> Integer
        (HlirType::F32 | HlirType::F64, t) if t.is_integer() => {
            if t.is_signed() {
                CastOp::FPToSI
            } else {
                CastOp::FPToUI
            }
        }

        // Pointer casts
        (HlirType::Ptr(_), t) if t.is_integer() => CastOp::PtrToInt,
        (s, HlirType::Ptr(_)) if s.is_integer() => CastOp::IntToPtr,
        (HlirType::Ptr(_), HlirType::Ptr(_)) => CastOp::BitCast,

        // Default
        _ => CastOp::BitCast,
    }
}

/// Lower an HLIR constant to SIR
fn lower_constant(ctx: &LoweringContext, constant: &HlirConstant) -> Constant {
    match constant {
        HlirConstant::Unit => Constant::ZeroInit(SirType::Void),
        HlirConstant::Bool(b) => Constant::Bool(*b),
        HlirConstant::Int(val, ty) => match ty {
            HlirType::I8 => Constant::I8(*val as i8),
            HlirType::I16 => Constant::I16(*val as i16),
            HlirType::I32 => Constant::I32(*val as i32),
            HlirType::I64 | HlirType::I128 => Constant::I64(*val),
            HlirType::U8 => Constant::I8(*val as i8),
            HlirType::U16 => Constant::I16(*val as i16),
            HlirType::U32 => Constant::I32(*val as i32),
            HlirType::U64 | HlirType::U128 => Constant::I64(*val),
            _ => Constant::I64(*val),
        },
        HlirConstant::Float(val, ty) => match ty {
            HlirType::F32 => Constant::F32(*val as f32),
            HlirType::F64 => Constant::F64(*val),
            _ => Constant::F64(*val),
        },
        HlirConstant::String(s) => {
            // Strings become arrays of bytes
            let bytes: Vec<Constant> = s.bytes().map(|b| Constant::I8(b as i8)).collect();
            Constant::Aggregate(bytes)
        }
        HlirConstant::CString(s) => {
            // C strings become arrays of bytes with null terminator
            let mut bytes: Vec<Constant> = s.bytes().map(|b| Constant::I8(b as i8)).collect();
            bytes.push(Constant::I8(0)); // Null terminator
            Constant::Aggregate(bytes)
        }
        HlirConstant::Array(elems) => {
            let sir_elems: Vec<Constant> = elems.iter().map(|e| lower_constant(ctx, e)).collect();
            Constant::Aggregate(sir_elems)
        }
        HlirConstant::Struct(fields) => {
            let sir_fields: Vec<Constant> = fields.iter().map(|f| lower_constant(ctx, f)).collect();
            Constant::Aggregate(sir_fields)
        }
        HlirConstant::Null(ty) => Constant::NullPtr,
        HlirConstant::Undef(ty) => Constant::Undef(lower_type(ty)),
        HlirConstant::FunctionRef(name) => {
            // Function references become function pointers
            Constant::NullPtr // Simplified - would need proper handling
        }
        HlirConstant::GlobalRef(name) => {
            // Global references become pointers
            Constant::NullPtr // Simplified
        }
    }
}

/// Lower an HLIR type to SIR type
pub fn lower_type(hlir_ty: &HlirType) -> SirType {
    match hlir_ty {
        HlirType::Void => SirType::Void,
        HlirType::Bool => SirType::bool(),
        HlirType::I8 => SirType::Scalar(ScalarType::I8),
        HlirType::I16 => SirType::Scalar(ScalarType::I16),
        HlirType::I32 => SirType::i32(),
        HlirType::I64 | HlirType::I128 => SirType::i64(),
        HlirType::U8 => SirType::Scalar(ScalarType::I8), // SIR uses signed for storage
        HlirType::U16 => SirType::Scalar(ScalarType::I16),
        HlirType::U32 => SirType::i32(),
        HlirType::U64 | HlirType::U128 => SirType::i64(),
        HlirType::F32 => SirType::f32(),
        HlirType::F64 => SirType::f64(),

        HlirType::Ptr(inner) => SirType::ptr(lower_type(inner)),

        HlirType::Array(elem, size) => SirType::Array(ArrayType::new(lower_type(elem), *size)),

        HlirType::Struct(name) => {
            // Reference to a named struct
            let struct_ty = StructType::new(vec![]).named(name);
            SirType::Struct(struct_ty)
        }

        HlirType::Tuple(elems) => {
            if elems.is_empty() {
                SirType::Void
            } else {
                // Lower tuple to anonymous struct
                let fields: Vec<(Option<String>, SirType)> = elems
                    .iter()
                    .enumerate()
                    .map(|(i, ty)| (Some(format!("_{}", i)), lower_type(ty)))
                    .collect();
                SirType::Struct(StructType::new(fields))
            }
        }

        HlirType::Function {
            params,
            return_type,
        } => {
            let param_tys: Vec<SirType> = params.iter().map(lower_type).collect();
            let ret_ty = lower_type(return_type);
            SirType::Function(super::types::FunctionType {
                params: param_tys,
                ret: Box::new(ret_ty),
                cc: CallingConv::Default,
            })
        }

        // Linear algebra primitives - lower to structs with SIMD-friendly layout
        HlirType::Vec2 => {
            let fields = vec![
                (Some("x".to_string()), SirType::f32()),
                (Some("y".to_string()), SirType::f32()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec2"))
        }
        HlirType::Vec3 => {
            // Padded to 4 floats for SIMD
            let fields = vec![
                (Some("x".to_string()), SirType::f32()),
                (Some("y".to_string()), SirType::f32()),
                (Some("z".to_string()), SirType::f32()),
                (Some("_pad".to_string()), SirType::f32()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec3"))
        }
        HlirType::Vec4 => {
            let fields = vec![
                (Some("x".to_string()), SirType::f32()),
                (Some("y".to_string()), SirType::f32()),
                (Some("z".to_string()), SirType::f32()),
                (Some("w".to_string()), SirType::f32()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec4"))
        }
        HlirType::Mat2 => {
            // 2x2 matrix as 4 floats
            SirType::Array(ArrayType::new(SirType::f32(), 4))
        }
        HlirType::Mat3 => {
            // 3x3 matrix as 9 floats
            SirType::Array(ArrayType::new(SirType::f32(), 9))
        }
        HlirType::Mat4 => {
            // 4x4 matrix as 16 floats
            SirType::Array(ArrayType::new(SirType::f32(), 16))
        }
        HlirType::Quat => {
            // Quaternion as 4 floats (x, y, z, w)
            let fields = vec![
                (Some("x".to_string()), SirType::f32()),
                (Some("y".to_string()), SirType::f32()),
                (Some("z".to_string()), SirType::f32()),
                (Some("w".to_string()), SirType::f32()),
            ];
            SirType::Struct(StructType::new(fields).named("Quat"))
        }
        // f64 vector types
        HlirType::Vec2d => {
            let fields = vec![
                (Some("x".to_string()), SirType::f64()),
                (Some("y".to_string()), SirType::f64()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec2d"))
        }
        HlirType::Vec3d => {
            // Padded to 4 doubles for SIMD
            let fields = vec![
                (Some("x".to_string()), SirType::f64()),
                (Some("y".to_string()), SirType::f64()),
                (Some("z".to_string()), SirType::f64()),
                (Some("_pad".to_string()), SirType::f64()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec3d"))
        }
        HlirType::Vec4d => {
            let fields = vec![
                (Some("x".to_string()), SirType::f64()),
                (Some("y".to_string()), SirType::f64()),
                (Some("z".to_string()), SirType::f64()),
                (Some("w".to_string()), SirType::f64()),
            ];
            SirType::Struct(StructType::new(fields).named("Vec4d"))
        }
        // Octonion: 8 x f32 struct
        HlirType::Octonion => {
            let fields: Vec<(Option<String>, SirType)> = (0..8)
                .map(|i| (Some(format!("c{}", i)), SirType::f32()))
                .collect();
            SirType::Struct(StructType::new(fields).named("Octonion"))
        }
        // Quaternionic Neural Network types
        HlirType::QuatLinear => SirType::Struct(StructType::new(vec![]).named("QuatLinear")),
        HlirType::QuatConv2d => SirType::Struct(StructType::new(vec![]).named("QuatConv2d")),
        HlirType::QuatRnnState => SirType::Struct(StructType::new(vec![]).named("QuatRnnState")),
        HlirType::QuatGate => SirType::Struct(StructType::new(vec![]).named("QuatGate")),
        HlirType::Dual => {
            // Dual number for automatic differentiation
            let fields = vec![
                (Some("value".to_string()), SirType::f64()),
                (Some("derivative".to_string()), SirType::f64()),
            ];
            SirType::Struct(StructType::new(fields).named("Dual"))
        }
        HlirType::Knowledge { inner, .. } => lower_type(inner),
    }
}

/// Lower type with epistemic awareness
pub fn lower_type_epistemic(hlir_ty: &HlirType, has_uncertainty: bool) -> SirType {
    let base_ty = lower_type(hlir_ty);

    if has_uncertainty {
        SirType::Epistemic(Box::new(base_ty))
    } else {
        base_ty
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_type_lowering() {
        assert_eq!(lower_type(&HlirType::I32), SirType::i32());
        assert_eq!(lower_type(&HlirType::F64), SirType::f64());
        assert_eq!(lower_type(&HlirType::Bool), SirType::bool());
    }

    #[test]
    fn test_type_lowering_ptr() {
        let ptr_ty = HlirType::Ptr(Box::new(HlirType::I32));
        let lowered = lower_type(&ptr_ty);
        match lowered {
            SirType::Pointer(ptr) => {
                assert_eq!(*ptr.pointee, SirType::i32());
            }
            _ => panic!("Expected pointer type"),
        }
    }

    #[test]
    fn test_type_lowering_array() {
        let arr_ty = HlirType::Array(Box::new(HlirType::F64), 10);
        let lowered = lower_type(&arr_ty);
        match lowered {
            SirType::Array(arr) => {
                assert_eq!(*arr.elem, SirType::f64());
                assert_eq!(arr.len, 10);
            }
            _ => panic!("Expected array type"),
        }
    }

    #[test]
    fn test_type_lowering_tuple() {
        let tuple_ty = HlirType::Tuple(vec![HlirType::I32, HlirType::F64]);
        let lowered = lower_type(&tuple_ty);
        match lowered {
            SirType::Struct(s) => {
                assert_eq!(s.fields.len(), 2);
            }
            _ => panic!("Expected struct type for tuple"),
        }
    }

    #[test]
    fn test_type_lowering_epistemic() {
        let base_ty = lower_type(&HlirType::F64);
        let epistemic_ty = lower_type_epistemic(&HlirType::F64, true);
        match epistemic_ty {
            SirType::Epistemic(inner) => {
                assert_eq!(*inner, base_ty);
            }
            _ => panic!("Expected epistemic type"),
        }
    }

    #[test]
    fn test_type_lowering_vec3() {
        let lowered = lower_type(&HlirType::Vec3);
        match lowered {
            SirType::Struct(s) => {
                assert_eq!(s.fields.len(), 4); // x, y, z, _pad
                assert_eq!(s.name, Some("Vec3".to_string()));
            }
            _ => panic!("Expected struct type for Vec3"),
        }
    }

    #[test]
    fn test_type_lowering_dual() {
        let lowered = lower_type(&HlirType::Dual);
        match lowered {
            SirType::Struct(s) => {
                assert_eq!(s.fields.len(), 2); // value, derivative
                assert_eq!(s.name, Some("Dual".to_string()));
            }
            _ => panic!("Expected struct type for Dual"),
        }
    }

    #[test]
    fn test_empty_module_lowering() {
        let hlir = HlirModule::new("test");
        let (sir, _ctx, _metadata) = lower_module(&hlir);
        assert_eq!(sir.name, "test");
        assert!(sir.functions.is_empty());
    }

    #[test]
    fn test_constant_lowering() {
        let ctx = LoweringContext::new();

        let int_const = lower_constant(&ctx, &HlirConstant::Int(42, HlirType::I32));
        assert_eq!(int_const, Constant::I32(42));

        let float_const = lower_constant(&ctx, &HlirConstant::Float(3.14, HlirType::F64));
        assert_eq!(float_const, Constant::F64(3.14));

        let bool_const = lower_constant(&ctx, &HlirConstant::Bool(true));
        assert_eq!(bool_const, Constant::Bool(true));
    }

    #[test]
    fn test_cast_op_determination() {
        // Integer extension
        assert!(matches!(
            determine_cast_op(&HlirType::I32, &HlirType::I64),
            CastOp::SExt
        ));

        // Integer truncation
        assert!(matches!(
            determine_cast_op(&HlirType::I64, &HlirType::I32),
            CastOp::Trunc
        ));

        // Float extension
        assert!(matches!(
            determine_cast_op(&HlirType::F32, &HlirType::F64),
            CastOp::FPExt
        ));

        // Float to int
        assert!(matches!(
            determine_cast_op(&HlirType::F64, &HlirType::I32),
            CastOp::FPToSI
        ));

        // Int to float
        assert!(matches!(
            determine_cast_op(&HlirType::I32, &HlirType::F64),
            CastOp::SIToFP
        ));
    }

    #[test]
    fn test_binary_op_lowering() {
        // Integer add
        let add_op = lower_binary_op(BinaryOp::Add, &HlirType::I32);
        assert!(matches!(add_op, LoweredBinOp::Arith(ArithOp::Add)));

        // Float add
        let fadd_op = lower_binary_op(BinaryOp::Add, &HlirType::F64);
        assert!(matches!(fadd_op, LoweredBinOp::Arith(ArithOp::FAdd)));

        // Comparison
        let cmp_op = lower_binary_op(BinaryOp::Eq, &HlirType::I32);
        assert!(matches!(cmp_op, LoweredBinOp::Cmp(CmpOp::Eq)));
    }

    #[test]
    fn test_lowering_context() {
        let mut ctx = LoweringContext::new();

        // Test value allocation
        let v1 = ctx.alloc_value(SirType::i32());
        let v2 = ctx.alloc_value(SirType::f64());
        assert_ne!(v1, v2);

        // Test value mapping
        let hlir_val = HlirValueId(0);
        ctx.map_value(hlir_val, v1);
        assert_eq!(ctx.get_value(hlir_val), Some(v1));

        // Test block allocation and mapping
        let b1 = ctx.alloc_block();
        let hlir_block = HlirBlockId(0);
        ctx.map_block(hlir_block, b1);
        assert_eq!(ctx.get_block(hlir_block), Some(b1));
    }

    #[test]
    fn test_extract_quantity_mg() {
        let ty = HlirType::Struct("Quantity_mg".to_string());
        let unit = MetadataTracker::extract_quantity_wrapper(&ty);
        
        assert!(unit.is_some());
        let unit = unit.unwrap();
        
        // mg should have dimensions [1,0,0,0,0,0,0] (mass) and scale 1e-6
        assert_eq!(unit.dimensions, [1, 0, 0, 0, 0, 0, 0]);
        assert_eq!(unit.scale, 1e-6);
    }

    #[test]
    fn test_extract_quantity_mg_per_l() {
        let ty = HlirType::Struct("Quantity_mg_DIV_L".to_string());
        let unit = MetadataTracker::extract_quantity_wrapper(&ty);
        
        assert!(unit.is_some());
        let unit = unit.unwrap();
        
        // mg/L should have dimensions [1,-3,0,0,0,0,0] (mass/volume)
        // mg is [1,0,0,0,0,0,0] scale 1e-6
        // L is [0,3,0,0,0,0,0] scale 1e-3
        // mg/L = [1,-3,0,0,0,0,0] scale 1e-6/1e-3 = 1e-3
        assert_eq!(unit.dimensions, [1, -3, 0, 0, 0, 0, 0]);
        assert_eq!(unit.scale, 1e-3);
    }

    #[test]
    fn test_extract_non_quantity() {
        let ty = HlirType::I32;
        let unit = MetadataTracker::extract_quantity_wrapper(&ty);
        assert!(unit.is_none());
    }

    #[test]
    fn test_extract_quantity_simple_units() {
        // Test various simple units
        let test_cases = vec![
            ("kg", [1, 0, 0, 0, 0, 0, 0], 1.0),
            ("g", [1, 0, 0, 0, 0, 0, 0], 1e-3),
            ("L", [0, 3, 0, 0, 0, 0, 0], 1e-3),
            ("mL", [0, 3, 0, 0, 0, 0, 0], 1e-6),
            ("h", [0, 0, 1, 0, 0, 0, 0], 3600.0),
            ("min", [0, 0, 1, 0, 0, 0, 0], 60.0),
        ];

        for (unit_symbol, expected_dims, expected_scale) in test_cases {
            let ty = HlirType::Struct(format!("Quantity_{}", unit_symbol));
            let unit = MetadataTracker::extract_quantity_wrapper(&ty);
            assert!(unit.is_some(), "Failed to extract unit for {}", unit_symbol);
            let unit = unit.unwrap();
            assert_eq!(unit.dimensions, expected_dims, "Wrong dimensions for {}", unit_symbol);
            assert_eq!(unit.scale, expected_scale, "Wrong scale for {}", unit_symbol);
        }
    }

    #[test]
    fn test_extract_quantity_clearance() {
        // Test L/h (clearance unit)
        let ty = HlirType::Struct("Quantity_L_DIV_h".to_string());
        let unit = MetadataTracker::extract_quantity_wrapper(&ty);
        
        assert!(unit.is_some());
        let unit = unit.unwrap();
        
        // L/h should have dimensions [0,3,-1,0,0,0,0] (volume/time = clearance)
        // L is [0,3,0,0,0,0,0] scale 1e-3
        // h is [0,0,1,0,0,0,0] scale 3600
        // L/h = [0,3,-1,0,0,0,0] scale 1e-3/3600 ≈ 2.78e-7
        assert_eq!(unit.dimensions, [0, 3, -1, 0, 0, 0, 0]);
        assert!((unit.scale - 1e-3 / 3600.0).abs() < 1e-10);
    }
}

// ==================== HANDLER ID PROPAGATION ====================

use crate::sir::module::{HandlerMetadata, EFFECT_HANDLER_MAP};

/// Populate effect handler IDs and metadata in the SIR module
/// This creates the unified dispatch mechanism for all effects
fn populate_handler_ids(module: &mut SirModule) {
    let mut effect_ids: HashMap<String, u32> = HashMap::new();
    let mut handler_metas: HashMap<u32, HandlerMetadata> = HashMap::new();
    
    // Register all known effects with their predefined handler IDs
    for (effect_name, handler_id) in EFFECT_HANDLER_MAP {
        effect_ids.insert(effect_name.to_string(), *handler_id);
        
        // Create metadata for this handler
        let meta = HandlerMetadata::new(effect_name.to_string(), *handler_id);
        let meta = populate_effect_operations(meta, effect_name);
        
        handler_metas.insert(*handler_id, meta);
    }
    
    module.effect_handler_ids = effect_ids;
    // Note: handler_metadata is no longer stored in SirModule struct
    let _ = handler_metas; // Suppress unused variable warning
}

/// Populate default operations and metadata for each effect type
fn populate_effect_operations(mut meta: HandlerMetadata, effect_name: &str) -> HandlerMetadata {
    match effect_name {
        "IO" => {
            meta = meta.with_operation("print");
            meta = meta.with_operation("println");
            meta = meta.with_operation("read");
            meta = meta.with_operation("read_line");
            meta = meta.with_operation("write");
        }
        "Mut" => {
            meta = meta.with_state();
            meta = meta.with_operation("get");
            meta = meta.with_operation("set");
            meta = meta.with_operation("modify");
        }
        "Alloc" => {
            meta = meta.with_state();
            meta = meta.with_operation("malloc");
            meta = meta.with_operation("free");
            meta = meta.with_operation("realloc");
        }
        "Panic" => {
            meta = meta.with_operation("panic");
            meta = meta.with_operation("assert");
        }
        "Async" => {
            meta = meta.with_async();
            meta = meta.with_operation("await");
            meta = meta.with_operation("spawn");
            meta = meta.with_operation("join");
        }
        "GPU" => {
            meta = meta.with_async();
            meta = meta.with_operation("kernel_call");
            meta = meta.with_operation("memcpy_h2d");
            meta = meta.with_operation("memcpy_d2h");
        }
        "Prob" => {
            meta = meta.with_operation("sample");
            meta = meta.with_operation("observe");
            meta = meta.with_operation("score");
        }
        "Grad" => {
            meta = meta.with_operation("gradient");
            meta = meta.with_operation("backprop");
        }
        "Causal" => {
            meta = meta.with_operation("intervene");
            meta = meta.with_operation("counterfactual");
        }
        "Epistemic" => {
            meta = meta.with_operation("measure");
            meta = meta.with_operation("refine");
            meta = meta.with_operation("confidence");
        }
        "FFI" => {
            meta = meta.with_operation("call_c");
            meta = meta.with_operation("call_external");
        }
        "Network" => {
            meta = meta.with_async();
            meta = meta.with_operation("send");
            meta = meta.with_operation("recv");
        }
        "Sensor" => {
            meta = meta.with_operation("read_sensor");
            meta = meta.with_operation("calibrate");
        }
        _ => {}
    }
    meta
}

/// Validate that all effects in use have handler IDs
/// Warns if handler IDs are missing (fallback to generic dispatch)
fn validate_effect_handlers(module: &SirModule) {
    // Check all functions for effects that might not have handlers
    for func in &module.functions {
        for block in &func.blocks {
            for instr in &block.instructions {
                // Check if this instruction uses effects
                match instr {
                    _ => {
                        // In a full implementation, we'd check call instructions
                        // and verify the callee has proper handler IDs
                    }
                }
            }
        }
    }
}
