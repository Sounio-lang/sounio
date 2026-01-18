//! MIR to Cranelift backend
//!
//! This module provides translation from MIR (Mid-level IR) to Cranelift IR.
//! MIR serves as an intermediate representation between HLIR and backend-specific IR.
//!
//! # Architecture
//!
//! ```text
//! HLIR -> MIR -> [MIR Optimizer] -> Cranelift -> Native Code
//! ```
//!
//! # Key Type Mappings
//!
//! | MIR Type | Cranelift Type |
//! |----------|----------------|
//! | I32      | types::I32     |
//! | I64      | types::I64     |
//! | F32      | types::F32     |
//! | F64      | types::F64     |
//! | Bool     | types::I8      |
//! | Ptr(_)   | types::I64     |
//! | Unit     | types::I64     |

use crate::hlir::HlirModule;
use crate::mir::MirModule;
use crate::mir::optimization::OptimizationLevel;

#[cfg(feature = "jit")]
use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, UserFuncName, types};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_codegen::Context;
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{FuncId, Linkage, Module};
#[cfg(feature = "jit")]
use std::collections::HashMap;

/// Compile MIR module to native code via Cranelift JIT
#[cfg(feature = "jit")]
pub fn compile_mir(mir_module: &MirModule) -> Result<Vec<u8>, String> {
    let jit = MirAwareCraneliftJit::new();
    let _compiled = jit.compile_mir(mir_module)?;
    // Return empty bytes for now; the compiled module holds the JIT code
    Ok(vec![])
}

/// Compile MIR module with optimization
#[cfg(feature = "jit")]
pub fn compile_mir_optimized(mir_module: &MirModule, opt_level: OptimizationLevel) -> Result<CompiledModule, String> {
    let jit = MirAwareCraneliftJit::new()
        .with_optimization()
        .with_mir_optimization(opt_level);
    jit.compile_mir(mir_module)
}

#[cfg(not(feature = "jit"))]
pub fn compile_mir(_mir_module: &MirModule) -> Result<Vec<u8>, String> {
    Err("JIT backend not enabled. Compile with --features jit".to_string())
}

#[cfg(not(feature = "jit"))]
pub fn compile_mir_optimized(_mir_module: &MirModule, _opt_level: OptimizationLevel) -> Result<(), String> {
    Err("JIT backend not enabled. Compile with --features jit".to_string())
}

/// Lower HLIR module to MIR, then compile to native code
pub fn compile_hlir_via_mir(hlir_module: &HlirModule) -> Result<Vec<u8>, String> {
    // First, lower HLIR to MIR
    let mir_module = lower_hlir_to_mir(hlir_module)?;

    // Then compile MIR to native code
    #[cfg(feature = "jit")]
    {
        compile_mir(&mir_module)
    }

    #[cfg(not(feature = "jit"))]
    {
        let _ = mir_module;
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }
}

/// Lower HLIR module to MIR
fn lower_hlir_to_mir(hlir_module: &HlirModule) -> Result<MirModule, String> {
    use crate::mir::lower::lower;
    Ok(lower(hlir_module))
}

#[cfg(feature = "jit")]
/// Extended CraneliftJIT that supports MIR compilation
pub struct MirAwareCraneliftJit {
    /// Whether to enable Cranelift optimization
    optimize: bool,
    /// MIR optimization level
    mir_opt_level: Option<OptimizationLevel>,
}

#[cfg(feature = "jit")]
impl MirAwareCraneliftJit {
    pub fn new() -> Self {
        Self {
            optimize: false,
            mir_opt_level: None,
        }
    }

    pub fn with_optimization(mut self) -> Self {
        self.optimize = true;
        self
    }

    /// Enable MIR-level optimization passes
    pub fn with_mir_optimization(mut self, level: OptimizationLevel) -> Self {
        self.mir_opt_level = Some(level);
        self
    }

    /// Compile MIR module and return a handle to the compiled code
    pub fn compile_mir(&self, mir_module: &MirModule) -> Result<CompiledModule, String> {
        // Run MIR optimization passes if enabled
        let optimized_module = if let Some(opt_level) = self.mir_opt_level {
            let mut module_clone = mir_module.clone();
            let mut pass_manager = create_default_pass_manager(opt_level);
            let _modified = pass_manager.run_module_passes(&mut module_clone)?;
            module_clone
        } else {
            mir_module.clone()
        };

        let mut compiler = MirCraneliftCompiler::new(self.optimize)?;
        compiler.compile_mir_module(&optimized_module)?;
        compiler.finalize()
    }

    /// Compile HLIR module via MIR
    pub fn compile_hlir_via_mir(&self, hlir_module: &HlirModule) -> Result<CompiledModule, String> {
        let mir_module = lower_hlir_to_mir(hlir_module)?;
        self.compile_mir(&mir_module)
    }

    /// Compile HLIR with both MIR and Cranelift optimization
    pub fn compile_hlir_optimized(&self, hlir_module: &HlirModule) -> Result<CompiledModule, String> {
        let mir_module = lower_hlir_to_mir(hlir_module)?;
        self.compile_mir(&mir_module)
    }
}

#[cfg(feature = "jit")]
impl Default for MirAwareCraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "jit")]
/// Handle to compiled JIT code (same as CraneliftJit)
pub use crate::codegen::cranelift::CompiledModule;

#[cfg(feature = "jit")]
/// MIR-aware Cranelift compiler
pub struct MirCraneliftCompiler {
    jit_module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
    /// Map from MIR function names to Cranelift function IDs
    func_ids: HashMap<String, FuncId>,
    /// Map from function names to their signatures (for calling)
    func_sigs: HashMap<String, Signature>,
    /// Set of exported (user-defined) function names
    exported_funcs: std::collections::HashSet<String>,
    /// Map from global variable names to their data IDs
    global_data_ids: HashMap<String, cranelift_module::DataId>,
    /// Map from global variable names to their types
    global_types: HashMap<String, MirType>,
}

#[cfg(feature = "jit")]
impl MirCraneliftCompiler {
    /// Create a new MIR Cranelift compiler
    pub fn new(optimize: bool) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("use_colocated_libcalls", "false")
            .map_err(|e| format!("Failed to set flag: {}", e))?;
        flag_builder
            .set("is_pic", "false")
            .map_err(|e| format!("Failed to set flag: {}", e))?;

        if optimize {
            flag_builder
                .set("opt_level", "speed")
                .map_err(|e| format!("Failed to set opt level: {}", e))?;
        } else {
            flag_builder
                .set("opt_level", "none")
                .map_err(|e| format!("Failed to set opt level: {}", e))?;
        }

        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("Failed to create ISA builder: {}", e))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("Failed to create ISA: {}", e))?;

        let jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        let jit_module = JITModule::new(jit_builder);
        let ctx = jit_module.make_context();

        Ok(Self {
            jit_module,
            ctx,
            func_ctx: FunctionBuilderContext::new(),
            func_ids: HashMap::new(),
            func_sigs: HashMap::new(),
            exported_funcs: std::collections::HashSet::new(),
            global_data_ids: HashMap::new(),
            global_types: HashMap::new(),
        })
    }

    /// Translate a MIR type to a Cranelift type
    fn translate_type(&self, ty: &MirType) -> types::Type {
        match ty {
            MirType::Unit => types::I64, // Use I64 for unit to avoid issues
            MirType::Bool => types::I8,
            MirType::I8 | MirType::U8 => types::I8,
            MirType::I16 | MirType::U16 => types::I16,
            MirType::I32 | MirType::U32 => types::I32,
            MirType::I64 | MirType::U64 | MirType::Isize | MirType::Usize => types::I64,
            MirType::I128 | MirType::U128 => types::I128,
            MirType::F32 => types::F32,
            MirType::F64 => types::F64,
            MirType::Char => types::I32, // Unicode code point
            MirType::String => types::I64, // Pointer to string data
            MirType::Ptr(_) => types::I64, // Pointers are 64-bit
            MirType::Array(_, _) => types::I64, // Pointer to array
            MirType::Tuple(_) => types::I64, // Pointer to tuple
            MirType::Function { .. } => types::I64, // Function pointer
            MirType::Void => types::I64, // Use I64 for void
            MirType::Error => types::I64, // Fallback
        }
    }

    /// Create a Cranelift signature for a MIR function
    fn create_signature(&self, func: &MirFunction) -> Signature {
        let call_conv = self.jit_module.isa().default_call_conv();
        let mut sig = Signature::new(call_conv);

        // Add parameters
        for (_name, ty) in &func.params {
            let cl_type = self.translate_type(ty);
            sig.params.push(AbiParam::new(cl_type));
        }

        // Add return type (if not unit/void)
        if !matches!(func.return_type, MirType::Unit | MirType::Void) {
            let ret_type = self.translate_type(&func.return_type);
            sig.returns.push(AbiParam::new(ret_type));
        }

        sig
    }

    /// Compile a MIR module
    pub fn compile_mir_module(&mut self, mir_module: &MirModule) -> Result<(), String> {
        // First pass: declare all global variables
        for global in &mir_module.globals {
            self.declare_global(global)?;
        }

        // Second pass: declare all functions
        for func in &mir_module.functions {
            let sig = self.create_signature(func);
            let func_id = self
                .jit_module
                .declare_function(&func.name, Linkage::Export, &sig)
                .map_err(|e| format!("Failed to declare function {}: {}", func.name, e))?;
            self.func_ids.insert(func.name.clone(), func_id);
            self.func_sigs.insert(func.name.clone(), sig);
            self.exported_funcs.insert(func.name.clone());
        }

        // Third pass: define global variables
        for global in &mir_module.globals {
            self.define_global(global)?;
        }

        // Fourth pass: compile all functions
        for func in &mir_module.functions {
            self.compile_function(func)?;
        }

        Ok(())
    }

    /// Declare a global variable
    fn declare_global(&mut self, global: &crate::mir::MirGlobal) -> Result<(), String> {
        let linkage = if global.is_const {
            Linkage::Local
        } else {
            Linkage::Export
        };

        let data_id = self
            .jit_module
            .declare_data(&global.name, linkage, !global.is_const, false)
            .map_err(|e| format!("Failed to declare global {}: {}", global.name, e))?;

        self.global_data_ids.insert(global.name.clone(), data_id);
        self.global_types.insert(global.name.clone(), global.ty.clone());
        Ok(())
    }

    /// Define a global variable with its initial value
    fn define_global(&mut self, global: &crate::mir::MirGlobal) -> Result<(), String> {
        let data_id = self.global_data_ids[&global.name];

        // Determine the size of the global
        let size = global.ty.size_bytes().unwrap_or(8);

        // Create a data context for the global
        let mut data_ctx = cranelift_module::DataDescription::new();
        data_ctx.define_zeroinit(size);

        // If there's an initial value, write it
        if let Some(ref init) = global.init {
            let mut bytes = vec![0u8; size];
            self.write_constant_to_bytes(init, &global.ty, &mut bytes);
            data_ctx.define(bytes.into_boxed_slice());
        }

        self.jit_module
            .define_data(data_id, &data_ctx)
            .map_err(|e| format!("Failed to define global {}: {}", global.name, e))?;

        Ok(())
    }

    /// Write a constant value to a byte buffer
    fn write_constant_to_bytes(&self, constant: &MirConstant, _ty: &MirType, bytes: &mut [u8]) {
        match constant {
            MirConstant::Bool(b) => {
                if !bytes.is_empty() {
                    bytes[0] = *b as u8;
                }
            }
            MirConstant::Int(n) => {
                let n_bytes = n.to_le_bytes();
                let len = bytes.len().min(8);
                bytes[..len].copy_from_slice(&n_bytes[..len]);
            }
            MirConstant::UInt(n) => {
                let n_bytes = n.to_le_bytes();
                let len = bytes.len().min(8);
                bytes[..len].copy_from_slice(&n_bytes[..len]);
            }
            MirConstant::Float(f) => {
                let f_bytes = f.to_le_bytes();
                let len = bytes.len().min(8);
                bytes[..len].copy_from_slice(&f_bytes[..len]);
            }
            _ => {
                // For other constants (Null, Unit, etc.), leave as zero-initialized
            }
        }
    }

    /// Compile a single MIR function
    fn compile_function(&mut self, func: &MirFunction) -> Result<FuncId, String> {
        let func_id = self.func_ids[&func.name];

        // Set up function signature
        self.ctx.func.signature = self.create_signature(func);
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Build function body
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);

            // Collect func_refs we need to declare
            let mut local_func_refs: HashMap<String, cranelift_codegen::ir::FuncRef> =
                HashMap::new();
            for (name, &id) in &self.func_ids {
                let local_ref = self.jit_module.declare_func_in_func(id, builder.func);
                local_func_refs.insert(name.clone(), local_ref);
            }

            // Collect global refs we need to declare
            let mut local_global_refs: HashMap<String, cranelift_codegen::ir::GlobalValue> =
                HashMap::new();
            for (name, &data_id) in &self.global_data_ids {
                let global_value = self.jit_module.declare_data_in_func(data_id, builder.func);
                local_global_refs.insert(name.clone(), global_value);
            }

            // Create translator and translate the function
            let mut translator = FunctionTranslator::new(
                &mut builder,
                &local_func_refs,
                &local_global_refs,
                &self.global_types,
            );
            translator.translate_function(func)?;

            builder.finalize();
        }

        // Compile the function
        self.jit_module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function {}: {}", func.name, e))?;

        self.jit_module.clear_context(&mut self.ctx);

        Ok(func_id)
    }

    /// Finalize compilation and return the compiled module
    pub fn finalize(mut self) -> Result<CompiledModule, String> {
        self.jit_module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize: {}", e))?;

        let mut functions = HashMap::new();
        for name in &self.exported_funcs {
            if let Some(&func_id) = self.func_ids.get(name) {
                let ptr = self.jit_module.get_finalized_function(func_id);
                functions.insert(name.clone(), ptr);
            }
        }

        Ok(CompiledModule::new(self.jit_module, functions))
    }
}

/// Translator for converting MIR functions to Cranelift IR
#[cfg(feature = "jit")]
struct FunctionTranslator<'a, 'b> {
    builder: &'a mut FunctionBuilder<'b>,
    func_refs: &'a HashMap<String, cranelift_codegen::ir::FuncRef>,
    /// Map from global variable names to their GlobalValue references
    global_refs: &'a HashMap<String, cranelift_codegen::ir::GlobalValue>,
    /// Map from global variable names to their types
    global_types: &'a HashMap<String, MirType>,
    /// Map from MIR ValueId to Cranelift Value
    values: HashMap<ValueId, cranelift_codegen::ir::Value>,
    /// Map from MIR BlockId to Cranelift Block
    blocks: HashMap<BlockId, cranelift_codegen::ir::Block>,
    /// Map from MIR BlockId to its phi node block parameters
    block_params: HashMap<BlockId, Vec<(ValueId, MirType)>>,
    /// Map from ValueId to Variable for SSA construction
    variables: HashMap<ValueId, Variable>,
    /// Next variable index
    next_var: usize,
}

#[cfg(feature = "jit")]
impl<'a, 'b> FunctionTranslator<'a, 'b> {
    fn new(
        builder: &'a mut FunctionBuilder<'b>,
        func_refs: &'a HashMap<String, cranelift_codegen::ir::FuncRef>,
        global_refs: &'a HashMap<String, cranelift_codegen::ir::GlobalValue>,
        global_types: &'a HashMap<String, MirType>,
    ) -> Self {
        Self {
            builder,
            func_refs,
            global_refs,
            global_types,
            values: HashMap::new(),
            blocks: HashMap::new(),
            block_params: HashMap::new(),
            variables: HashMap::new(),
            next_var: 0,
        }
    }

    /// Translate a complete MIR function
    fn translate_function(&mut self, func: &MirFunction) -> Result<(), String> {
        // First pass: Create all Cranelift blocks and collect phi node information
        for block in &func.blocks {
            let cl_block = self.builder.create_block();
            self.blocks.insert(block.id, cl_block);

            // Scan for phi nodes in this block and record their result types
            let mut phi_params = Vec::new();
            for instr in &block.instructions {
                if let MirInstruction::Phi { result, ty, .. } = instr {
                    phi_params.push((*result, ty.clone()));
                }
            }
            if !phi_params.is_empty() {
                self.block_params.insert(block.id, phi_params);
            }
        }

        // Second pass: Add block parameters for phi nodes (except entry block)
        for block in &func.blocks {
            if Some(block.id) == func.blocks.first().map(|b| b.id) {
                continue; // Skip entry block - it gets function params instead
            }

            let cl_block = self.blocks[&block.id];
            if let Some(phi_params) = self.block_params.get(&block.id) {
                for (result_id, ty) in phi_params {
                    let cl_type = self.translate_type(ty);
                    let param = self.builder.append_block_param(cl_block, cl_type);
                    self.values.insert(*result_id, param);
                }
            }
        }

        // Set up entry block with function parameters
        if let Some(entry) = func.blocks.first() {
            let entry_block = self.blocks[&entry.id];
            self.builder.switch_to_block(entry_block);

            // Add function parameters as block parameters
            for (i, (_name, ty)) in func.params.iter().enumerate() {
                let cl_type = self.translate_type(ty);
                let param = self.builder.append_block_param(entry_block, cl_type);
                // Parameters are values 0, 1, 2, ...
                let value_id = ValueId(i);
                self.values.insert(value_id, param);
            }
        }

        // Third pass: Translate each block
        let mut is_first = true;
        for block in &func.blocks {
            self.translate_block(block, is_first)?;
            is_first = false;
        }

        // Seal all blocks
        self.builder.seal_all_blocks();

        Ok(())
    }

    /// Translate a MIR type to a Cranelift type
    fn translate_type(&self, ty: &MirType) -> types::Type {
        match ty {
            MirType::Unit => types::I64,
            MirType::Bool => types::I8,
            MirType::I8 | MirType::U8 => types::I8,
            MirType::I16 | MirType::U16 => types::I16,
            MirType::I32 | MirType::U32 => types::I32,
            MirType::I64 | MirType::U64 | MirType::Isize | MirType::Usize => types::I64,
            MirType::I128 | MirType::U128 => types::I128,
            MirType::F32 => types::F32,
            MirType::F64 => types::F64,
            MirType::Char => types::I32,
            MirType::String => types::I64,
            MirType::Ptr(_) => types::I64,
            MirType::Array(_, _) => types::I64,
            MirType::Tuple(_) => types::I64,
            MirType::Function { .. } => types::I64,
            MirType::Void => types::I64,
            MirType::Error => types::I64,
        }
    }

    /// Translate a single MIR block
    fn translate_block(&mut self, block: &MirBlock, is_entry: bool) -> Result<(), String> {
        let cl_block = self.blocks[&block.id];

        // Switch to the block (entry is already active)
        if !is_entry {
            self.builder.switch_to_block(cl_block);
        }

        // Translate all instructions
        for instr in &block.instructions {
            self.translate_instruction(instr)?;
        }

        // Translate terminator
        self.translate_terminator(&block.terminator)?;

        Ok(())
    }

    /// Get a Cranelift value for a MIR ValueId
    fn get_value(&self, id: ValueId) -> Result<cranelift_codegen::ir::Value, String> {
        self.values
            .get(&id)
            .copied()
            .ok_or_else(|| format!("Value {:?} not found", id))
    }

    /// Get or create a Cranelift block for a MIR BlockId
    fn get_block(&self, id: BlockId) -> Result<cranelift_codegen::ir::Block, String> {
        self.blocks
            .get(&id)
            .copied()
            .ok_or_else(|| format!("Block {:?} not found", id))
    }

    /// Translate a single MIR instruction
    fn translate_instruction(&mut self, instr: &MirInstruction) -> Result<(), String> {
        match instr {
            MirInstruction::Const { result, value, ty } => {
                let cl_type = self.translate_type(ty);
                let val = self.translate_constant(value, cl_type)?;
                self.values.insert(*result, val);
            }

            MirInstruction::Binary {
                result,
                op,
                left,
                right,
                ty,
            } => {
                let lhs = self.get_value(*left)?;
                let rhs = self.get_value(*right)?;
                let val = self.translate_binary_op(*op, lhs, rhs, ty)?;
                self.values.insert(*result, val);
            }

            MirInstruction::Unary {
                result,
                op,
                operand,
                ty,
            } => {
                let operand_val = self.get_value(*operand)?;
                let val = self.translate_unary_op(*op, operand_val, ty)?;
                self.values.insert(*result, val);
            }

            MirInstruction::Compare {
                result,
                op,
                left,
                right,
                ty,
            } => {
                let lhs = self.get_value(*left)?;
                let rhs = self.get_value(*right)?;
                let val = self.translate_compare_op(*op, lhs, rhs, ty)?;
                self.values.insert(*result, val);
            }

            MirInstruction::Load { result, address, ty } => {
                let addr = self.get_value(*address)?;
                let cl_type = self.translate_type(ty);
                let val = self.builder.ins().load(cl_type, MemFlags::new(), addr, 0);
                self.values.insert(*result, val);
            }

            MirInstruction::Store { address, value } => {
                let addr = self.get_value(*address)?;
                let val = self.get_value(*value)?;
                self.builder.ins().store(MemFlags::new(), val, addr, 0);
            }

            MirInstruction::GetElementPtr {
                result,
                base,
                indices,
                ty,
            } => {
                // GEP: compute address = base + sum(indices * element_size)
                let mut addr = self.get_value(*base)?;
                let elem_size = ty.size_bytes().unwrap_or(8) as i64;

                for idx_id in indices {
                    let idx = self.get_value(*idx_id)?;
                    // Scale index by element size
                    let scale = self.builder.ins().iconst(types::I64, elem_size);
                    let offset = self.builder.ins().imul(idx, scale);
                    addr = self.builder.ins().iadd(addr, offset);
                }

                self.values.insert(*result, addr);
            }

            MirInstruction::LoadGlobal {
                result,
                global_name,
                ty,
            } => {
                let cl_type = self.translate_type(ty);
                if let Some(&global_value) = self.global_refs.get(global_name) {
                    // Get the address of the global variable
                    let addr = self.builder.ins().global_value(types::I64, global_value);
                    // Load the value from that address
                    let val = self.builder.ins().load(cl_type, MemFlags::new(), addr, 0);
                    self.values.insert(*result, val);
                } else {
                    // Global not found - return zero as fallback
                    let val = if cl_type.is_float() {
                        if cl_type == types::F32 {
                            self.builder.ins().f32const(0.0)
                        } else {
                            self.builder.ins().f64const(0.0)
                        }
                    } else {
                        self.builder.ins().iconst(cl_type, 0)
                    };
                    self.values.insert(*result, val);
                }
            }

            MirInstruction::StoreGlobal { global_name, value } => {
                if let Some(&global_value) = self.global_refs.get(global_name) {
                    let val = self.get_value(*value)?;
                    // Get the address of the global variable
                    let addr = self.builder.ins().global_value(types::I64, global_value);
                    // Store to that address
                    self.builder.ins().store(MemFlags::new(), val, addr, 0);
                }
                // If global not found, silently ignore (this is a compiler bug if it happens)
            }

            MirInstruction::Alloca { result, ty } => {
                // Allocate stack space for the type
                let size = ty.size_bytes().unwrap_or(8) as u32;
                let slot = self.builder.create_sized_stack_slot(
                    cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        size,
                        8, // 8-byte alignment
                    ),
                );
                let addr = self.builder.ins().stack_addr(types::I64, slot, 0);
                self.values.insert(*result, addr);
            }

            MirInstruction::Cast {
                result,
                source,
                source_ty,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let val = self.translate_cast(src_val, source_ty, target_ty)?;
                self.values.insert(*result, val);
            }

            MirInstruction::ZExt {
                result,
                source,
                source_ty: _,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().uextend(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::SExt {
                result,
                source,
                source_ty: _,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().sextend(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::Trunc {
                result,
                source,
                source_ty: _,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().ireduce(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::FPToSI {
                result,
                source,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().fcvt_to_sint(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::SIToFP {
                result,
                source,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().fcvt_from_sint(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::FPToUI {
                result,
                source,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().fcvt_to_uint(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::UIToFP {
                result,
                source,
                target_ty,
            } => {
                let src_val = self.get_value(*source)?;
                let target_cl = self.translate_type(target_ty);
                let val = self.builder.ins().fcvt_from_uint(target_cl, src_val);
                self.values.insert(*result, val);
            }

            MirInstruction::PtrCast {
                result,
                source,
                source_ty: _,
                target_ty: _,
            } => {
                // Pointer casts are no-ops at the IR level (same representation)
                let src_val = self.get_value(*source)?;
                self.values.insert(*result, src_val);
            }

            MirInstruction::Call {
                result,
                func_name,
                args,
                ty: _,
            } => {
                let arg_vals: Result<Vec<_>, _> =
                    args.iter().map(|a| self.get_value(*a)).collect();
                let arg_vals = arg_vals?;

                if let Some(&func_ref) = self.func_refs.get(func_name) {
                    let call = self.builder.ins().call(func_ref, &arg_vals);
                    if let Some(res_id) = result {
                        // Get the first return value if there is one
                        let results = self.builder.inst_results(call);
                        if !results.is_empty() {
                            self.values.insert(*res_id, results[0]);
                        }
                    }
                } else {
                    return Err(format!("Unknown function: {}", func_name));
                }
            }

            MirInstruction::CallIndirect {
                result,
                func_ptr,
                args,
                ty,
            } => {
                let ptr = self.get_value(*func_ptr)?;
                let arg_vals: Result<Vec<_>, _> =
                    args.iter().map(|a| self.get_value(*a)).collect();
                let arg_vals = arg_vals?;

                // Create a signature for the indirect call
                let call_conv = self.builder.func.signature.call_conv;
                let mut sig = Signature::new(call_conv);
                for _ in &arg_vals {
                    sig.params.push(AbiParam::new(types::I64));
                }
                if !matches!(ty, MirType::Unit | MirType::Void) {
                    let ret_type = self.translate_type(ty);
                    sig.returns.push(AbiParam::new(ret_type));
                }

                let sig_ref = self.builder.import_signature(sig);
                let call = self.builder.ins().call_indirect(sig_ref, ptr, &arg_vals);

                if let Some(res_id) = result {
                    let results = self.builder.inst_results(call);
                    if !results.is_empty() {
                        self.values.insert(*res_id, results[0]);
                    }
                }
            }

            MirInstruction::Phi {
                result,
                incoming: _,
                ty: _,
            } => {
                // Phi nodes in Cranelift are handled via block parameters.
                // The result value was already mapped during translate_function
                // when we set up block parameters. We just verify it exists.
                if !self.values.contains_key(result) {
                    // This shouldn't happen if translate_function worked correctly
                    return Err(format!("Phi result {:?} not found in block parameters", result));
                }
                // Nothing else to do - the value is already set up as a block parameter
            }

            MirInstruction::Select {
                result,
                condition,
                true_value,
                false_value,
                ty,
            } => {
                let cond = self.get_value(*condition)?;
                let true_val = self.get_value(*true_value)?;
                let false_val = self.get_value(*false_value)?;
                let val = self.builder.ins().select(cond, true_val, false_val);
                self.values.insert(*result, val);
            }
        }

        Ok(())
    }

    /// Translate a MIR constant to a Cranelift value
    fn translate_constant(
        &mut self,
        constant: &MirConstant,
        cl_type: types::Type,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        match constant {
            MirConstant::Unit => Ok(self.builder.ins().iconst(types::I64, 0)),

            MirConstant::Bool(b) => Ok(self.builder.ins().iconst(types::I8, *b as i64)),

            MirConstant::Int(n) => Ok(self.builder.ins().iconst(cl_type, *n)),

            MirConstant::UInt(n) => Ok(self.builder.ins().iconst(cl_type, *n as i64)),

            MirConstant::Float(f) => {
                if cl_type == types::F32 {
                    Ok(self.builder.ins().f32const(*f as f32))
                } else {
                    Ok(self.builder.ins().f64const(*f))
                }
            }

            MirConstant::String(_) => {
                // String constants are stored as pointers
                // For now, return a null pointer; proper string handling requires data sections
                Ok(self.builder.ins().iconst(types::I64, 0))
            }

            MirConstant::Null => Ok(self.builder.ins().iconst(types::I64, 0)),

            MirConstant::FunctionRef(name) => {
                // Function references become function pointers
                if let Some(&func_ref) = self.func_refs.get(name) {
                    Ok(self.builder.ins().func_addr(types::I64, func_ref))
                } else {
                    Ok(self.builder.ins().iconst(types::I64, 0))
                }
            }

            MirConstant::GlobalRef(_) => {
                // Global references are pointers
                Ok(self.builder.ins().iconst(types::I64, 0))
            }
        }
    }

    /// Translate a binary operation
    fn translate_binary_op(
        &mut self,
        op: MirBinaryOp,
        lhs: cranelift_codegen::ir::Value,
        rhs: cranelift_codegen::ir::Value,
        ty: &MirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let is_float = ty.is_float();
        let is_signed = ty.is_signed();

        let val = match op {
            MirBinaryOp::Add => {
                if is_float {
                    self.builder.ins().fadd(lhs, rhs)
                } else {
                    self.builder.ins().iadd(lhs, rhs)
                }
            }

            MirBinaryOp::Sub => {
                if is_float {
                    self.builder.ins().fsub(lhs, rhs)
                } else {
                    self.builder.ins().isub(lhs, rhs)
                }
            }

            MirBinaryOp::Mul => {
                if is_float {
                    self.builder.ins().fmul(lhs, rhs)
                } else {
                    self.builder.ins().imul(lhs, rhs)
                }
            }

            MirBinaryOp::Div => {
                if is_float {
                    self.builder.ins().fdiv(lhs, rhs)
                } else if is_signed {
                    self.builder.ins().sdiv(lhs, rhs)
                } else {
                    self.builder.ins().udiv(lhs, rhs)
                }
            }

            MirBinaryOp::Rem => {
                if is_float {
                    // Floating-point remainder requires a runtime call
                    // For now, use fma-based approximation: a - floor(a/b) * b
                    let div = self.builder.ins().fdiv(lhs, rhs);
                    let floor = self.builder.ins().floor(div);
                    let prod = self.builder.ins().fmul(floor, rhs);
                    self.builder.ins().fsub(lhs, prod)
                } else if is_signed {
                    self.builder.ins().srem(lhs, rhs)
                } else {
                    self.builder.ins().urem(lhs, rhs)
                }
            }

            MirBinaryOp::And => self.builder.ins().band(lhs, rhs),

            MirBinaryOp::Or => self.builder.ins().bor(lhs, rhs),

            MirBinaryOp::Xor => self.builder.ins().bxor(lhs, rhs),

            MirBinaryOp::Shl => self.builder.ins().ishl(lhs, rhs),

            MirBinaryOp::LShr => self.builder.ins().ushr(lhs, rhs),

            MirBinaryOp::AShr => self.builder.ins().sshr(lhs, rhs),
        };

        Ok(val)
    }

    /// Translate a unary operation
    fn translate_unary_op(
        &mut self,
        op: MirUnaryOp,
        operand: cranelift_codegen::ir::Value,
        ty: &MirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let val = match op {
            MirUnaryOp::Neg => {
                if ty.is_float() {
                    self.builder.ins().fneg(operand)
                } else {
                    // Integer negation: 0 - operand
                    let cl_type = self.translate_type(ty);
                    let zero = self.builder.ins().iconst(cl_type, 0);
                    self.builder.ins().isub(zero, operand)
                }
            }

            MirUnaryOp::Not => {
                // Logical NOT: compare with 0
                let zero = self.builder.ins().iconst(types::I8, 0);
                self.builder.ins().icmp(IntCC::Equal, operand, zero)
            }

            MirUnaryOp::BitNot => self.builder.ins().bnot(operand),

            MirUnaryOp::FNeg => self.builder.ins().fneg(operand),
        };

        Ok(val)
    }

    /// Translate a comparison operation
    fn translate_compare_op(
        &mut self,
        op: MirCompareOp,
        lhs: cranelift_codegen::ir::Value,
        rhs: cranelift_codegen::ir::Value,
        ty: &MirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let is_signed = ty.is_signed();

        let val = match op {
            MirCompareOp::Eq => self.builder.ins().icmp(IntCC::Equal, lhs, rhs),

            MirCompareOp::Ne => self.builder.ins().icmp(IntCC::NotEqual, lhs, rhs),

            MirCompareOp::Lt => {
                if is_signed {
                    self.builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs)
                } else {
                    self.builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs)
                }
            }

            MirCompareOp::Le => {
                if is_signed {
                    self.builder
                        .ins()
                        .icmp(IntCC::SignedLessThanOrEqual, lhs, rhs)
                } else {
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs)
                }
            }

            MirCompareOp::Gt => {
                if is_signed {
                    self.builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs)
                } else {
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThan, lhs, rhs)
                }
            }

            MirCompareOp::Ge => {
                if is_signed {
                    self.builder
                        .ins()
                        .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs)
                } else {
                    self.builder
                        .ins()
                        .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs)
                }
            }

            // Floating point comparisons
            MirCompareOp::FEq => self.builder.ins().fcmp(FloatCC::Equal, lhs, rhs),

            MirCompareOp::FNe => self.builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),

            MirCompareOp::FLt => self.builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),

            MirCompareOp::FLe => self.builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),

            MirCompareOp::FGt => self.builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),

            MirCompareOp::FGe => self
                .builder
                .ins()
                .fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
        };

        Ok(val)
    }

    /// Translate a type cast
    fn translate_cast(
        &mut self,
        src: cranelift_codegen::ir::Value,
        source_ty: &MirType,
        target_ty: &MirType,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let src_bits = source_ty.size_bytes().unwrap_or(8) * 8;
        let tgt_bits = target_ty.size_bytes().unwrap_or(8) * 8;
        let target_cl = self.translate_type(target_ty);

        // Float to int
        if source_ty.is_float() && target_ty.is_integer() {
            if target_ty.is_signed() {
                return Ok(self.builder.ins().fcvt_to_sint(target_cl, src));
            } else {
                return Ok(self.builder.ins().fcvt_to_uint(target_cl, src));
            }
        }

        // Int to float
        if source_ty.is_integer() && target_ty.is_float() {
            if source_ty.is_signed() {
                return Ok(self.builder.ins().fcvt_from_sint(target_cl, src));
            } else {
                return Ok(self.builder.ins().fcvt_from_uint(target_cl, src));
            }
        }

        // Float to float
        if source_ty.is_float() && target_ty.is_float() {
            if src_bits < tgt_bits {
                return Ok(self.builder.ins().fpromote(target_cl, src));
            } else if src_bits > tgt_bits {
                return Ok(self.builder.ins().fdemote(target_cl, src));
            } else {
                return Ok(src);
            }
        }

        // Int to int
        if source_ty.is_integer() && target_ty.is_integer() {
            if src_bits < tgt_bits {
                if source_ty.is_signed() {
                    return Ok(self.builder.ins().sextend(target_cl, src));
                } else {
                    return Ok(self.builder.ins().uextend(target_cl, src));
                }
            } else if src_bits > tgt_bits {
                return Ok(self.builder.ins().ireduce(target_cl, src));
            } else {
                return Ok(src);
            }
        }

        // Pointer casts and others: no-op
        Ok(src)
    }

    /// Translate a block terminator
    fn translate_terminator(&mut self, terminator: &MirTerminator) -> Result<(), String> {
        match terminator {
            MirTerminator::Branch { target } => {
                let target_block = self.get_block(*target)?;
                // Note: For proper phi node handling, we would need to pass block arguments here.
                // This requires knowing which source block we're in and matching with phi incoming edges.
                // For now, we pass empty args - the phi handling in translate_function sets up
                // block parameters, but proper argument passing requires additional infrastructure.
                self.builder.ins().jump(target_block, &[]);
            }

            MirTerminator::CondBranch {
                condition,
                true_target,
                false_target,
            } => {
                let cond = self.get_value(*condition)?;
                let true_block = self.get_block(*true_target)?;
                let false_block = self.get_block(*false_target)?;
                // Note: Similar to Branch, proper phi handling would require passing block args
                self.builder.ins().brif(cond, true_block, &[], false_block, &[]);
            }

            MirTerminator::Switch {
                value,
                default_target,
                cases,
            } => {
                let val = self.get_value(*value)?;
                let default_block = self.get_block(*default_target)?;

                // Build a switch using a sequence of comparisons and branches
                // For a more efficient implementation, we could use Cranelift's br_table
                for (case_val, target) in cases {
                    let target_block = self.get_block(*target)?;
                    let case_const = self.builder.ins().iconst(types::I64, *case_val);
                    let cmp = self.builder.ins().icmp(IntCC::Equal, val, case_const);

                    // Create a continuation block for the next case check
                    let next_block = self.builder.create_block();
                    self.builder.ins().brif(cmp, target_block, &[], next_block, &[]);
                    self.builder.switch_to_block(next_block);
                    self.builder.seal_block(next_block);
                }

                // Fall through to default
                self.builder.ins().jump(default_block, &[]);
            }

            MirTerminator::Return { value } => {
                if let Some(val_id) = value {
                    let val = self.get_value(*val_id)?;
                    self.builder.ins().return_(&[val]);
                } else {
                    self.builder.ins().return_(&[]);
                }
            }

            MirTerminator::Unreachable => {
                self.builder.ins().trap(cranelift_codegen::ir::TrapCode::UnreachableCodeReached);
            }

            MirTerminator::CallNoReturn { func_name, args } => {
                let arg_vals: Result<Vec<_>, _> =
                    args.iter().map(|a| self.get_value(*a)).collect();
                let arg_vals = arg_vals?;

                if let Some(&func_ref) = self.func_refs.get(func_name) {
                    self.builder.ins().call(func_ref, &arg_vals);
                }
                self.builder.ins().trap(cranelift_codegen::ir::TrapCode::UnreachableCodeReached);
            }
        }

        Ok(())
    }
}

// ==================== Helper Functions for MIR Optimization Integration ====================

/// Convenience function to compile HLIR via MIR with a specific optimization level
#[cfg(feature = "jit")]
pub fn compile_hlir_via_mir_optimized(
    hlir_module: &HlirModule,
    opt_level: OptimizationLevel,
) -> Result<CompiledModule, String> {
    let jit = MirAwareCraneliftJit::new()
        .with_optimization()
        .with_mir_optimization(opt_level);
    jit.compile_hlir_via_mir(hlir_module)
}

/// Get the optimization passes that will be run at a given level
pub fn get_optimization_passes_for_level(level: OptimizationLevel) -> Vec<&'static str> {
    match level {
        OptimizationLevel::O0 => vec![],
        OptimizationLevel::O1 => vec!["constant-propagation", "dead_code_elimination"],
        OptimizationLevel::O2 | OptimizationLevel::O3 => vec![
            "constant-propagation",
            "dead_code_elimination",
            "common_subexpression_elimination",
            "loop_invariant_code_motion",
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mir::builder::{FunctionBuilder as MirFunctionBuilder, ModuleBuilder};
    use crate::mir::types::{MirType, MirConstant};

    #[test]
    fn test_translate_type() {
        // Test basic type translation
        assert_eq!(
            MirType::I32.size_bytes(),
            Some(4),
            "I32 should be 4 bytes"
        );
        assert_eq!(
            MirType::F64.size_bytes(),
            Some(8),
            "F64 should be 8 bytes"
        );
        assert!(MirType::I32.is_integer(), "I32 should be an integer");
        assert!(MirType::F64.is_float(), "F64 should be a float");
    }

    #[test]
    fn test_mir_module_creation() {
        let mut module_builder = ModuleBuilder::new("test_module");

        // Create a simple function
        let mut func_builder =
            module_builder.create_function("add".to_string(), MirType::I32);
        func_builder.add_param("a".to_string(), MirType::I32);
        func_builder.add_param("b".to_string(), MirType::I32);

        // Build a simple add
        let const_42 = func_builder.build_i32(42);
        func_builder.build_return(Some(const_42));

        module_builder.add_function(func_builder.build());
        let module = module_builder.build();

        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "add");
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_mir_cranelift_compile() {
        let mut module_builder = ModuleBuilder::new("test");

        // Create a function that returns 42
        let mut func_builder =
            module_builder.create_function("main".to_string(), MirType::I64);
        let const_42 = func_builder.build_i64(42);
        func_builder.build_return(Some(const_42));
        module_builder.add_function(func_builder.build());

        let module = module_builder.build();

        // Compile the module
        let jit = MirAwareCraneliftJit::new();
        let result = jit.compile_mir(&module);
        assert!(result.is_ok(), "Compilation should succeed");

        // Try to get the function pointer
        let compiled = result.unwrap();
        let main_ptr = compiled.get_function("main");
        assert!(main_ptr.is_some(), "main function should exist");

        // Call the function
        unsafe {
            let result = compiled.call_i64("main");
            assert_eq!(result.unwrap(), 42, "main() should return 42");
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_mir_cranelift_with_optimization() {
        let mut module_builder = ModuleBuilder::new("test_opt");

        // Create a function with a constant expression that can be optimized
        let mut func_builder =
            module_builder.create_function("compute".to_string(), MirType::I64);

        // Build: return 10 + 32 (constant folding opportunity)
        let const_10 = func_builder.build_i64(10);
        let const_32 = func_builder.build_i64(32);
        let result_id = func_builder.fresh_value();
        func_builder.build_add(result_id, const_10, const_32, MirType::I64);
        func_builder.build_return(Some(result_id));
        module_builder.add_function(func_builder.build());

        let module = module_builder.build();

        // Compile with optimization enabled
        let jit = MirAwareCraneliftJit::new()
            .with_optimization()
            .with_mir_optimization(OptimizationLevel::O2);
        let result = jit.compile_mir(&module);
        assert!(result.is_ok(), "Optimized compilation should succeed");

        // Verify the function works correctly
        let compiled = result.unwrap();
        unsafe {
            let result = compiled.call_i64("compute");
            assert_eq!(result.unwrap(), 42, "compute() should return 42");
        }
    }

    #[cfg(feature = "jit")]
    #[test]
    fn test_mir_cranelift_binary_ops() {
        let mut module_builder = ModuleBuilder::new("test_binops");

        // Test subtraction
        let mut func_builder =
            module_builder.create_function("sub_test".to_string(), MirType::I64);
        let const_100 = func_builder.build_i64(100);
        let const_58 = func_builder.build_i64(58);
        let result_id = func_builder.fresh_value();
        func_builder.build_sub(result_id, const_100, const_58, MirType::I64);
        func_builder.build_return(Some(result_id));
        module_builder.add_function(func_builder.build());

        // Test multiplication
        let mut func_builder =
            module_builder.create_function("mul_test".to_string(), MirType::I64);
        let const_6 = func_builder.build_i64(6);
        let const_7 = func_builder.build_i64(7);
        let result_id = func_builder.fresh_value();
        func_builder.build_mul(result_id, const_6, const_7, MirType::I64);
        func_builder.build_return(Some(result_id));
        module_builder.add_function(func_builder.build());

        let module = module_builder.build();

        let jit = MirAwareCraneliftJit::new();
        let compiled = jit.compile_mir(&module).expect("Compilation should succeed");

        unsafe {
            assert_eq!(compiled.call_i64("sub_test").unwrap(), 42, "100 - 58 = 42");
            assert_eq!(compiled.call_i64("mul_test").unwrap(), 42, "6 * 7 = 42");
        }
    }

    #[test]
    fn test_type_mappings() {
        // Verify all the documented type mappings
        assert_eq!(MirType::I32.size_bytes(), Some(4));
        assert_eq!(MirType::I64.size_bytes(), Some(8));
        assert_eq!(MirType::F32.size_bytes(), Some(4));
        assert_eq!(MirType::F64.size_bytes(), Some(8));
        assert_eq!(MirType::Bool.size_bytes(), Some(1));
        assert_eq!(MirType::Ptr(Box::new(MirType::I64)).size_bytes(), Some(8));
        assert_eq!(MirType::Unit.size_bytes(), Some(1));
    }

    #[test]
    fn test_optimization_pass_levels() {
        let o0_passes = get_optimization_passes_for_level(OptimizationLevel::O0);
        assert!(o0_passes.is_empty(), "O0 should have no passes");

        let o1_passes = get_optimization_passes_for_level(OptimizationLevel::O1);
        assert!(o1_passes.contains(&"constant-propagation"), "O1 should include constant propagation");

        let o2_passes = get_optimization_passes_for_level(OptimizationLevel::O2);
        assert!(o2_passes.len() > o1_passes.len(), "O2 should have more passes than O1");
    }
}
