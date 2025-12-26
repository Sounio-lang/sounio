//! Cranelift JIT backend
//!
//! This module provides fast JIT compilation using Cranelift.
//! Cranelift is optimized for fast compilation rather than peak runtime performance,
//! making it ideal for development and scripting use cases.

use crate::hlir::HlirModule;
#[cfg(feature = "jit")]
use std::io::Write;

// ==================== Native Runtime Functions ====================
// These are called from JIT-compiled code via FFI

/// Print an i64 value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_i64(val: i64) {
    print!("{}", val);
    let _ = std::io::stdout().flush();
}

/// Print an f64 value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_f64(val: f64) {
    print!("{}", val);
    let _ = std::io::stdout().flush();
}

/// Print a newline
#[cfg(feature = "jit")]
extern "C" fn runtime_print_newline() {
    println!();
}

/// Print a string (pointer + length)
#[cfg(feature = "jit")]
extern "C" fn runtime_print_str(ptr: *const u8, len: usize) {
    if !ptr.is_null() && len > 0 {
        let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
        if let Ok(s) = std::str::from_utf8(slice) {
            print!("{}", s);
            let _ = std::io::stdout().flush();
        }
    }
}

/// Print a null-terminated C string
#[cfg(feature = "jit")]
extern "C" fn runtime_print_cstr(ptr: *const u8) {
    if !ptr.is_null() {
        let cstr = unsafe { std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char) };
        if let Ok(s) = cstr.to_str() {
            print!("{}", s);
            let _ = std::io::stdout().flush();
        }
    }
}

// Global storage for string constants during JIT execution
#[cfg(feature = "jit")]
use std::sync::Mutex;
#[cfg(feature = "jit")]
static STRING_STORAGE: Mutex<Vec<std::ffi::CString>> = Mutex::new(Vec::new());

/// Print a boolean value
#[cfg(feature = "jit")]
extern "C" fn runtime_print_bool(val: i8) {
    print!("{}", if val != 0 { "true" } else { "false" });
    let _ = std::io::stdout().flush();
}

/// Debug test function - just returns 99 to verify FFI works
#[cfg(feature = "jit")]
extern "C" fn runtime_debug_test() -> i64 {
    99
}

#[cfg(feature = "jit")]
use crate::hlir::{
    BinaryOp, BlockId, HlirBlock, HlirConstant, HlirFunction, HlirTerminator, HlirType, Op,
    UnaryOp, ValueId,
};
use std::collections::HashMap;

#[cfg(feature = "jit")]
use cranelift_codegen::Context;
#[cfg(feature = "jit")]
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Signature, UserFuncName, types};
#[cfg(feature = "jit")]
use cranelift_codegen::settings::{self, Configurable};
#[cfg(feature = "jit")]
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
#[cfg(feature = "jit")]
use cranelift_jit::{JITBuilder, JITModule};
#[cfg(feature = "jit")]
use cranelift_module::{FuncId, Linkage, Module};

/// Compile HLIR module to native code via Cranelift JIT
#[cfg(feature = "jit")]
pub fn compile(module: &HlirModule) -> Result<Vec<u8>, String> {
    let jit = CraneliftJit::new();
    let compiled = jit.compile(module)?;
    // Return empty vec - the compiled code is held in memory by JITModule
    Ok(vec![])
}

#[cfg(not(feature = "jit"))]
pub fn compile(_module: &HlirModule) -> Result<Vec<u8>, String> {
    Err("JIT backend not enabled. Compile with --features jit".to_string())
}

/// Cranelift JIT compiler
pub struct CraneliftJit {
    /// Whether to enable optimization
    optimize: bool,
}

impl CraneliftJit {
    pub fn new() -> Self {
        Self { optimize: false }
    }

    pub fn with_optimization(mut self) -> Self {
        self.optimize = true;
        self
    }

    /// Compile and immediately run the module, returning the result of main()
    #[cfg(feature = "jit")]
    pub fn compile_and_run(&self, module: &HlirModule) -> Result<i64, String> {
        let compiled = self.compile(module)?;
        unsafe { compiled.call_i64("main") }
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile_and_run(&self, _module: &HlirModule) -> Result<i64, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }

    /// Compile the module and return a handle to the compiled code
    #[cfg(feature = "jit")]
    pub fn compile(&self, module: &HlirModule) -> Result<CompiledModule, String> {
        let mut compiler = JitCompiler::new(self.optimize)?;
        compiler.compile_module(module)?;
        compiler.finalize()
    }

    #[cfg(not(feature = "jit"))]
    pub fn compile(&self, _module: &HlirModule) -> Result<CompiledModule, String> {
        Err("JIT backend not enabled. Compile with --features jit".to_string())
    }
}

impl Default for CraneliftJit {
    fn default() -> Self {
        Self::new()
    }
}

/// Handle to compiled JIT code
pub struct CompiledModule {
    #[cfg(feature = "jit")]
    #[allow(dead_code)]
    jit_module: JITModule,
    /// Function entry points
    functions: HashMap<String, *const u8>,
}

// SAFETY: The JIT module owns the compiled code and manages its lifetime
unsafe impl Send for CompiledModule {}
unsafe impl Sync for CompiledModule {}

impl CompiledModule {
    /// Get a function pointer by name
    pub fn get_function(&self, name: &str) -> Option<*const u8> {
        self.functions.get(name).copied()
    }

    /// Call a function with no arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    pub unsafe fn call_i64(&self, name: &str) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn() -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func())
    }

    /// Call a function with one i64 argument returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    #[allow(dead_code)]
    pub unsafe fn call_i64_i64(&self, name: &str, arg: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(arg))
    }

    /// Call a function with two i64 arguments returning i64
    ///
    /// # Safety
    /// The caller must ensure the function signature matches.
    #[allow(dead_code)]
    pub unsafe fn call_i64_i64_i64(&self, name: &str, a: i64, b: i64) -> Result<i64, String> {
        let ptr = self
            .get_function(name)
            .ok_or_else(|| format!("Function not found: {}", name))?;

        let func: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
        Ok(func(a, b))
    }
}

/// JIT compilation settings
#[allow(dead_code)]
pub struct JitSettings {
    /// Enable basic optimizations
    pub optimize: bool,
    /// Enable bounds checking
    pub bounds_check: bool,
    /// Enable overflow checking
    pub overflow_check: bool,
    /// Stack size in bytes
    pub stack_size: usize,
}

impl Default for JitSettings {
    fn default() -> Self {
        Self {
            optimize: false,
            bounds_check: true,
            overflow_check: true,
            stack_size: 1024 * 1024, // 1 MB
        }
    }
}

impl JitSettings {
    #[allow(dead_code)]
    pub fn release() -> Self {
        Self {
            optimize: true,
            bounds_check: false,
            overflow_check: false,
            stack_size: 8 * 1024 * 1024, // 8 MB
        }
    }
}

// ==================== JIT Compiler Implementation ====================

#[cfg(feature = "jit")]
struct JitCompiler {
    jit_module: JITModule,
    ctx: Context,
    func_ctx: FunctionBuilderContext,
    /// Map from HLIR function names to Cranelift function IDs
    func_ids: HashMap<String, FuncId>,
    /// Map from function names to their signatures (for calling)
    func_sigs: HashMap<String, Signature>,
    /// Set of exported (user-defined) function names
    exported_funcs: std::collections::HashSet<String>,
}

#[cfg(feature = "jit")]
impl JitCompiler {
    fn new(optimize: bool) -> Result<Self, String> {
        let mut flag_builder = settings::builder();
        flag_builder.set("use_colocated_libcalls", "false").unwrap();
        flag_builder.set("is_pic", "false").unwrap();

        if optimize {
            flag_builder.set("opt_level", "speed").unwrap();
        } else {
            flag_builder.set("opt_level", "none").unwrap();
        }

        let isa_builder = cranelift_native::builder()
            .map_err(|e| format!("Failed to create ISA builder: {}", e))?;

        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| format!("Failed to create ISA: {}", e))?;

        let mut jit_builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

        // Register runtime functions
        jit_builder.symbol("runtime_print_i64", runtime_print_i64 as *const u8);
        jit_builder.symbol("runtime_print_f64", runtime_print_f64 as *const u8);
        jit_builder.symbol("runtime_print_newline", runtime_print_newline as *const u8);
        jit_builder.symbol("runtime_print_str", runtime_print_str as *const u8);
        jit_builder.symbol("runtime_print_cstr", runtime_print_cstr as *const u8);
        jit_builder.symbol("runtime_print_bool", runtime_print_bool as *const u8);
        jit_builder.symbol("runtime_debug_test", runtime_debug_test as *const u8);

        let jit_module = JITModule::new(jit_builder);
        let ctx = jit_module.make_context();

        Ok(Self {
            jit_module,
            ctx,
            func_ctx: FunctionBuilderContext::new(),
            func_ids: HashMap::new(),
            func_sigs: HashMap::new(),
            exported_funcs: std::collections::HashSet::new(),
        })
    }

    fn compile_module(&mut self, module: &HlirModule) -> Result<(), String> {
        // Declare runtime functions first
        self.declare_runtime_functions()?;

        // First pass: declare all user functions
        for func in &module.functions {
            // Declaration-only functions are treated as imports (e.g., `extern { fn ...; }`).
            let is_import = func.blocks.is_empty();
            let sig = self.create_signature(func);
            let symbol_name = func.link_name.as_deref().unwrap_or(&func.name);
            let linkage = if is_import {
                Linkage::Import
            } else {
                Linkage::Export
            };
            let func_id = self
                .jit_module
                .declare_function(symbol_name, linkage, &sig)
                .map_err(|e| format!("Failed to declare function {}: {}", func.name, e))?;
            self.func_ids.insert(func.name.clone(), func_id);
            self.func_sigs.insert(func.name.clone(), sig);
            if !is_import {
                self.exported_funcs.insert(func.name.clone());
            }
        }

        // Second pass: compile all functions
        for func in &module.functions {
            if func.blocks.is_empty() {
                // Import-only declarations have no body to compile/define.
                continue;
            }
            self.compile_function(func)?;
        }

        Ok(())
    }

    fn declare_runtime_functions(&mut self) -> Result<(), String> {
        let call_conv = self.jit_module.isa().default_call_conv();

        // runtime_print_i64(i64) -> void
        let mut sig_print_i64 = Signature::new(call_conv);
        sig_print_i64.params.push(AbiParam::new(types::I64));
        let id = self
            .jit_module
            .declare_function("runtime_print_i64", Linkage::Import, &sig_print_i64)
            .map_err(|e| format!("Failed to declare runtime_print_i64: {}", e))?;
        self.func_ids.insert("runtime_print_i64".to_string(), id);
        self.func_sigs
            .insert("runtime_print_i64".to_string(), sig_print_i64);

        // runtime_print_f64(f64) -> void
        let mut sig_print_f64 = Signature::new(call_conv);
        sig_print_f64.params.push(AbiParam::new(types::F64));
        let id = self
            .jit_module
            .declare_function("runtime_print_f64", Linkage::Import, &sig_print_f64)
            .map_err(|e| format!("Failed to declare runtime_print_f64: {}", e))?;
        self.func_ids.insert("runtime_print_f64".to_string(), id);
        self.func_sigs
            .insert("runtime_print_f64".to_string(), sig_print_f64);

        // runtime_print_newline() -> void
        let sig_print_newline = Signature::new(call_conv);
        let id = self
            .jit_module
            .declare_function("runtime_print_newline", Linkage::Import, &sig_print_newline)
            .map_err(|e| format!("Failed to declare runtime_print_newline: {}", e))?;
        self.func_ids
            .insert("runtime_print_newline".to_string(), id);
        self.func_sigs
            .insert("runtime_print_newline".to_string(), sig_print_newline);

        // runtime_print_str(ptr, len) -> void
        let mut sig_print_str = Signature::new(call_conv);
        sig_print_str.params.push(AbiParam::new(types::I64)); // ptr
        sig_print_str.params.push(AbiParam::new(types::I64)); // len
        let id = self
            .jit_module
            .declare_function("runtime_print_str", Linkage::Import, &sig_print_str)
            .map_err(|e| format!("Failed to declare runtime_print_str: {}", e))?;
        self.func_ids.insert("runtime_print_str".to_string(), id);
        self.func_sigs
            .insert("runtime_print_str".to_string(), sig_print_str);

        // runtime_print_cstr(ptr) -> void  (null-terminated C string)
        let mut sig_print_cstr = Signature::new(call_conv);
        sig_print_cstr.params.push(AbiParam::new(types::I64)); // ptr
        let id = self
            .jit_module
            .declare_function("runtime_print_cstr", Linkage::Import, &sig_print_cstr)
            .map_err(|e| format!("Failed to declare runtime_print_cstr: {}", e))?;
        self.func_ids.insert("runtime_print_cstr".to_string(), id);
        self.func_sigs
            .insert("runtime_print_cstr".to_string(), sig_print_cstr);

        // runtime_print_bool(i8) -> void
        let mut sig_print_bool = Signature::new(call_conv);
        sig_print_bool.params.push(AbiParam::new(types::I8));
        let id = self
            .jit_module
            .declare_function("runtime_print_bool", Linkage::Import, &sig_print_bool)
            .map_err(|e| format!("Failed to declare runtime_print_bool: {}", e))?;
        self.func_ids.insert("runtime_print_bool".to_string(), id);
        self.func_sigs
            .insert("runtime_print_bool".to_string(), sig_print_bool);

        Ok(())
    }

    fn create_signature(&self, func: &HlirFunction) -> Signature {
        let call_conv = self.jit_module.isa().default_call_conv();
        let mut sig = Signature::new(call_conv);

        for param in &func.params {
            sig.params
                .push(AbiParam::new(hlir_to_cranelift_type(&param.ty)));
        }

        if func.return_type != HlirType::Void {
            sig.returns
                .push(AbiParam::new(hlir_to_cranelift_type(&func.return_type)));
        }

        sig
    }

    fn compile_function(&mut self, func: &HlirFunction) -> Result<(), String> {
        let func_id = self.func_ids[&func.name];

        // Create function signature
        self.ctx.func.signature = self.create_signature(func);
        self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

        // Build function body
        {
            let mut builder = FunctionBuilder::new(&mut self.ctx.func, &mut self.func_ctx);

            // Collect func_refs we need to declare
            let mut needed_funcs: Vec<(String, FuncId)> = Vec::new();
            for (name, &id) in &self.func_ids {
                needed_funcs.push((name.clone(), id));
            }

            // Declare functions in this function's namespace
            let mut local_func_refs = HashMap::new();
            for (name, id) in &needed_funcs {
                if let Some(sig) = self.func_sigs.get(name) {
                    let local_ref = self.jit_module.declare_func_in_func(*id, builder.func);
                    local_func_refs.insert(name.clone(), local_ref);
                }
            }

            translate_function(&mut builder, func, &local_func_refs)?;
            builder.finalize();
        }

        // Compile the function
        self.jit_module
            .define_function(func_id, &mut self.ctx)
            .map_err(|e| format!("Failed to define function {}: {}", func.name, e))?;

        self.jit_module.clear_context(&mut self.ctx);

        Ok(())
    }

    fn finalize(mut self) -> Result<CompiledModule, String> {
        self.jit_module
            .finalize_definitions()
            .map_err(|e| format!("Failed to finalize: {}", e))?;

        let mut functions = HashMap::new();
        // Only get function pointers for exported (user-defined) functions, not imported runtime functions
        for name in &self.exported_funcs {
            if let Some(&func_id) = self.func_ids.get(name) {
                let ptr = self.jit_module.get_finalized_function(func_id);
                functions.insert(name.clone(), ptr);
            }
        }

        Ok(CompiledModule {
            jit_module: self.jit_module,
            functions,
        })
    }
}

/// Convert HLIR type to Cranelift type
#[cfg(feature = "jit")]
fn hlir_to_cranelift_type(ty: &HlirType) -> types::Type {
    match ty {
        HlirType::Void => types::I64, // Use I64 for void to avoid issues
        HlirType::Bool => types::I8,
        HlirType::I8 | HlirType::U8 => types::I8,
        HlirType::I16 | HlirType::U16 => types::I16,
        HlirType::I32 | HlirType::U32 => types::I32,
        HlirType::I64 | HlirType::U64 => types::I64,
        HlirType::I128 | HlirType::U128 => types::I128,
        HlirType::F32 => types::F32,
        HlirType::F64 => types::F64,
        HlirType::Ptr(_) => types::I64,
        HlirType::Array(_, _) => types::I64, // Pointer to array
        HlirType::Struct(_) => types::I64,   // Pointer to struct
        HlirType::Tuple(_) => types::I64,    // Pointer to tuple or packed
        HlirType::Function { .. } => types::I64, // Function pointer
        // Linear algebra types - use SIMD vector types for performance
        // vec2: 2x f32 = 64 bits, but we use F32X4 for alignment
        HlirType::Vec2 => types::F32X4,
        // vec3: 3x f32, padded to 4x f32 for SIMD (128 bits)
        HlirType::Vec3 => types::F32X4,
        // vec4: 4x f32 = 128 bits
        HlirType::Vec4 => types::F32X4,
        // mat2: 2x2 = 4 floats, fits in F32X4
        HlirType::Mat2 => types::F32X4,
        // mat3: 3x3 = 9 floats, use 3x F32X4 (12 floats, 3 wasted)
        // For now, represent as pointer to data
        HlirType::Mat3 => types::I64,
        // mat4: 4x4 = 16 floats = 4x F32X4
        // For now, represent as pointer to data
        HlirType::Mat4 => types::I64,
        // quat: 4x f32 = 128 bits, same as vec4
        HlirType::Quat => types::F32X4,
        // dual: 2x f64 = 128 bits (value, derivative)
        // Use F64X2 for SIMD operations on dual numbers
        HlirType::Dual => types::F64X2,
    }
}

/// Translate an entire HLIR function to Cranelift IR
#[cfg(feature = "jit")]
fn translate_function(
    builder: &mut FunctionBuilder,
    func: &HlirFunction,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<(), String> {
    let mut values: HashMap<ValueId, cranelift_codegen::ir::Value> = HashMap::new();
    let mut blocks: HashMap<BlockId, cranelift_codegen::ir::Block> = HashMap::new();
    // Track which ValueIds are string constants (for print handling)
    let mut string_values: std::collections::HashSet<ValueId> = std::collections::HashSet::new();

    // Create all blocks first
    for block in &func.blocks {
        let cl_block = builder.create_block();
        blocks.insert(block.id, cl_block);
    }

    // Entry block parameters (function arguments)
    if let Some(entry) = func.blocks.first() {
        let entry_block = blocks[&entry.id];
        builder.switch_to_block(entry_block);

        // Add function parameters
        for param in &func.params {
            let ty = hlir_to_cranelift_type(&param.ty);
            let val = builder.append_block_param(entry_block, ty);
            values.insert(param.value, val);
        }
    }

    // Translate each block (skip entry block switch since we're already there)
    let mut first = true;
    for block in &func.blocks {
        translate_block(builder, block, &blocks, &mut values, &mut string_values, func_refs, first)?;
        first = false;
    }

    // Seal all blocks at the end
    builder.seal_all_blocks();

    Ok(())
}

#[cfg(feature = "jit")]
fn translate_block(
    builder: &mut FunctionBuilder,
    block: &HlirBlock,
    blocks: &HashMap<BlockId, cranelift_codegen::ir::Block>,
    values: &mut HashMap<ValueId, cranelift_codegen::ir::Value>,
    string_values: &mut std::collections::HashSet<ValueId>,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
    is_entry: bool,
) -> Result<(), String> {
    let cl_block = blocks[&block.id];

    // Only switch if not already on this block (entry block is already active)
    if !is_entry {
        builder.switch_to_block(cl_block);
    }

    // Translate instructions
    for instr in &block.instructions {
        let result = translate_instruction(builder, instr, values, string_values, func_refs)?;
        if let (Some(res_id), Some(val)) = (instr.result, result) {
            values.insert(res_id, val);
        }
    }

    // Translate terminator
    translate_terminator(builder, &block.terminator, blocks, values)?;

    Ok(())
}

#[cfg(feature = "jit")]
fn translate_instruction(
    builder: &mut FunctionBuilder,
    instr: &crate::hlir::HlirInstr,
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
    string_values: &mut std::collections::HashSet<ValueId>,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<Option<cranelift_codegen::ir::Value>, String> {
    let ty = hlir_to_cranelift_type(&instr.ty);

    match &instr.op {
        Op::Const(constant) => {
            // Track string constants for proper print handling
            if let HlirConstant::String(_) = constant {
                if let Some(result_id) = instr.result {
                    string_values.insert(result_id);
                }
            }
            let val = translate_constant(builder, constant, &instr.ty, func_refs)?;
            Ok(Some(val))
        }

        Op::Copy(src) => {
            let src_val = get_value(values, *src)?;
            Ok(Some(src_val))
        }

        Op::Binary { op, left, right } => {
            let lhs = get_value(values, *left)?;
            let rhs = get_value(values, *right)?;
            let result = translate_binary_op(builder, *op, lhs, rhs, &instr.ty)?;
            Ok(Some(result))
        }

        Op::Unary { op, operand } => {
            let val = get_value(values, *operand)?;
            let result = translate_unary_op(builder, *op, val, &instr.ty)?;
            Ok(Some(result))
        }

        Op::CallDirect { name, args } => {
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            // Handle print/println specially by routing to runtime functions
            if name == "print" || name == "println" {
                for (i, arg_val) in arg_vals.iter().enumerate() {
                    let arg_type = builder.func.dfg.value_type(*arg_val);
                    let arg_id = args[i];

                    // Check if this argument is a string constant
                    if string_values.contains(&arg_id) {
                        // Use runtime_print_cstr for string constants
                        if let Some(&func_ref) = func_refs.get("runtime_print_cstr") {
                            builder.ins().call(func_ref, &[*arg_val]);
                        }
                    } else {
                        // Choose runtime function based on argument type
                        let runtime_func = if arg_type == types::F64 || arg_type == types::F32 {
                            "runtime_print_f64"
                        } else if arg_type == types::I8 {
                            "runtime_print_bool"
                        } else {
                            "runtime_print_i64"
                        };

                        if let Some(&func_ref) = func_refs.get(runtime_func) {
                            // Convert argument to expected type if needed
                            let converted_arg = if runtime_func == "runtime_print_f64"
                                && arg_type == types::F32
                            {
                                builder.ins().fpromote(types::F64, *arg_val)
                            } else if runtime_func == "runtime_print_i64" && arg_type != types::I64 {
                                if arg_type.is_int() && arg_type.bits() < 64 {
                                    builder.ins().sextend(types::I64, *arg_val)
                                } else {
                                    *arg_val
                                }
                            } else {
                                *arg_val
                            };
                            builder.ins().call(func_ref, &[converted_arg]);
                        }
                    }
                }

                // Add newline for println
                if name == "println" {
                    if let Some(&func_ref) = func_refs.get("runtime_print_newline") {
                        builder.ins().call(func_ref, &[]);
                    }
                }

                return Ok(None);
            }

            if let Some(&func_ref) = func_refs.get(name) {
                let call = builder.ins().call(func_ref, &arg_vals);
                let results = builder.inst_results(call);
                if results.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(results[0]))
                }
            } else {
                // Unknown function - return zero
                let zero = builder.ins().iconst(ty, 0);
                Ok(Some(zero))
            }
        }

        Op::Call { func, args } => {
            let func_val = get_value(values, *func)?;
            let arg_vals: Vec<_> = args
                .iter()
                .map(|a| get_value(values, *a))
                .collect::<Result<_, _>>()?;

            // Indirect call - need a signature
            // For now, assume simple i64 -> i64 signature
            let mut sig = Signature::new(cranelift_codegen::isa::CallConv::SystemV);
            for _ in &arg_vals {
                sig.params.push(AbiParam::new(types::I64));
            }
            sig.returns.push(AbiParam::new(types::I64));

            let sig_ref = builder.import_signature(sig);
            let call = builder.ins().call_indirect(sig_ref, func_val, &arg_vals);
            let results = builder.inst_results(call);
            if results.is_empty() {
                Ok(None)
            } else {
                Ok(Some(results[0]))
            }
        }

        Op::Load { ptr } => {
            let ptr_val = get_value(values, *ptr)?;
            let loaded = builder.ins().load(ty, MemFlags::new(), ptr_val, 0);
            Ok(Some(loaded))
        }

        Op::Store { ptr, value } => {
            let ptr_val = get_value(values, *ptr)?;
            let val = get_value(values, *value)?;
            builder.ins().store(MemFlags::new(), val, ptr_val, 0);
            Ok(None)
        }

        Op::Alloca { ty: alloc_ty } => {
            let size = alloc_ty.size_bits() / 8;
            let size = if size == 0 { 8 } else { size };
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size as u32,
                0,
            ));
            let addr = builder.ins().stack_addr(types::I64, slot, 0);
            Ok(Some(addr))
        }

        Op::GetFieldPtr { base, field } => {
            let base_val = get_value(values, *base)?;
            let offset = (*field * 8) as i64; // Assume 8-byte fields
            let ptr = builder.ins().iadd_imm(base_val, offset);
            Ok(Some(ptr))
        }

        Op::GetElementPtr { base, index } => {
            let base_val = get_value(values, *base)?;
            let idx_val = get_value(values, *index)?;
            let idx_val = match builder.func.dfg.value_type(idx_val) {
                types::I64 => idx_val,
                t if t.is_int() && t.bits() < 64 => builder.ins().sextend(types::I64, idx_val),
                _ => idx_val,
            };
            // Extract element size from pointer type (fixes issue #11)
            let elem_size = match &instr.ty {
                HlirType::Ptr(inner) => {
                    let bits = inner.size_bits();
                    if bits == 0 { 8 } else { bits / 8 }
                }
                _ => 8, // Fallback for non-pointer types
            } as i64;
            let offset = builder.ins().imul_imm(idx_val, elem_size);
            let ptr = builder.ins().iadd(base_val, offset);
            Ok(Some(ptr))
        }

        Op::Cast {
            value,
            source,
            target,
        } => {
            let val = get_value(values, *value)?;
            let target_ty = hlir_to_cranelift_type(target);
            let val_ty = builder.func.dfg.value_type(val);

            if val_ty == target_ty {
                Ok(Some(val))
            } else if val_ty.is_int() && target_ty.is_int() && val_ty.bits() < target_ty.bits() {
                // Integer widening: preserve signedness of the *source* type.
                let extended = if matches!(
                    source,
                    HlirType::I8 | HlirType::I16 | HlirType::I32 | HlirType::I64 | HlirType::I128
                ) {
                    builder.ins().sextend(target_ty, val)
                } else {
                    builder.ins().uextend(target_ty, val)
                };
                Ok(Some(extended))
            } else if !val_ty.is_int() && !target_ty.is_int() && val_ty.bits() < target_ty.bits() {
                Ok(Some(builder.ins().fpromote(target_ty, val)))
            } else {
                // Truncate / demote
                if target_ty.is_int() {
                    Ok(Some(builder.ins().ireduce(target_ty, val)))
                } else {
                    Ok(Some(builder.ins().fdemote(target_ty, val)))
                }
            }
        }

        Op::Phi { incoming } => {
            // Phi nodes should be handled as block parameters in Cranelift
            // For now, return first incoming value if available
            if let Some((_, first_val)) = incoming.first() {
                let val = get_value(values, *first_val)?;
                Ok(Some(val))
            } else {
                let zero = builder.ins().iconst(ty, 0);
                Ok(Some(zero))
            }
        }

        Op::ExtractValue { base, index } => {
            // For tuples/structs stored as aggregates
            let base_val = get_value(values, *base)?;
            // Simplified: treat as field access
            let offset = (*index * 8) as i32;
            let ptr = builder.ins().iadd_imm(base_val, offset as i64);
            let loaded = builder.ins().load(ty, MemFlags::new(), ptr, 0);
            Ok(Some(loaded))
        }

        Op::InsertValue { base, value, index } => {
            let base_val = get_value(values, *base)?;
            let val = get_value(values, *value)?;
            // Simplified: treat as field store
            let offset = (*index * 8) as i32;
            let ptr = builder.ins().iadd_imm(base_val, offset as i64);
            builder.ins().store(MemFlags::new(), val, ptr, 0);
            Ok(Some(base_val))
        }

        Op::Tuple(vals) | Op::Array(vals) => {
            // Allocate space and store values
            let size = (vals.len() * 8) as u32;
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size,
                0,
            ));
            let base = builder.ins().stack_addr(types::I64, slot, 0);

            for (i, v) in vals.iter().enumerate() {
                let val = get_value(values, *v)?;
                let offset = (i * 8) as i32;
                builder.ins().store(MemFlags::new(), val, base, offset);
            }

            Ok(Some(base))
        }

        Op::Struct { name: _, fields } => {
            // Similar to tuple
            let size = (fields.len() * 8) as u32;
            let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                size,
                0,
            ));
            let base = builder.ins().stack_addr(types::I64, slot, 0);

            for (i, (_, v)) in fields.iter().enumerate() {
                let val = get_value(values, *v)?;
                let offset = (i * 8) as i32;
                builder.ins().store(MemFlags::new(), val, base, offset);
            }

            Ok(Some(base))
        }

        Op::PerformEffect { .. } => {
            // Effects not supported in JIT yet
            let zero = builder.ins().iconst(ty, 0);
            Ok(Some(zero))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_constant(
    builder: &mut FunctionBuilder,
    constant: &HlirConstant,
    ty: &HlirType,
    func_refs: &HashMap<String, cranelift_codegen::ir::FuncRef>,
) -> Result<cranelift_codegen::ir::Value, String> {
    let cl_ty = hlir_to_cranelift_type(ty);

    match constant {
        HlirConstant::Unit => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Bool(b) => Ok(builder.ins().iconst(types::I8, *b as i64)),
        HlirConstant::Int(i, _) => Ok(builder.ins().iconst(cl_ty, *i)),
        HlirConstant::Float(f, _) => {
            if cl_ty == types::F32 {
                Ok(builder.ins().f32const(*f as f32))
            } else {
                Ok(builder.ins().f64const(*f))
            }
        }
        HlirConstant::String(s) => {
            // Store the string in global storage so it survives during JIT execution
            // Use CString for null-termination so runtime_print_cstr can use it
            let cstring = std::ffi::CString::new(s.as_str())
                .unwrap_or_else(|_| std::ffi::CString::new("").unwrap());

            // Store in global storage to keep alive, then get pointer from stored CString
            if let Ok(mut storage) = STRING_STORAGE.lock() {
                storage.push(cstring);
                // Get pointer from the stored CString (it's now at the end of the vec)
                let stored_ptr = storage.last().unwrap().as_ptr() as i64;
                Ok(builder.ins().iconst(types::I64, stored_ptr))
            } else {
                // Fallback to null if lock fails
                Ok(builder.ins().iconst(types::I64, 0))
            }
        }
        HlirConstant::Null(_) => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Undef(_) => Ok(builder.ins().iconst(cl_ty, 0)),
        HlirConstant::FunctionRef(name) => {
            if let Some(&func_ref) = func_refs.get(name) {
                Ok(builder.ins().func_addr(types::I64, func_ref))
            } else {
                Ok(builder.ins().iconst(types::I64, 0))
            }
        }
        HlirConstant::GlobalRef(_) => Ok(builder.ins().iconst(types::I64, 0)),
        HlirConstant::Array(_) | HlirConstant::Struct(_) => {
            // Complex constants - return null for now
            Ok(builder.ins().iconst(types::I64, 0))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_binary_op(
    builder: &mut FunctionBuilder,
    op: BinaryOp,
    lhs: cranelift_codegen::ir::Value,
    rhs: cranelift_codegen::ir::Value,
    result_ty: &HlirType,
) -> Result<cranelift_codegen::ir::Value, String> {
    use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};

    let result = match op {
        // Integer arithmetic
        BinaryOp::Add => builder.ins().iadd(lhs, rhs),
        BinaryOp::Sub => builder.ins().isub(lhs, rhs),
        BinaryOp::Mul => builder.ins().imul(lhs, rhs),
        BinaryOp::SDiv => builder.ins().sdiv(lhs, rhs),
        BinaryOp::UDiv => builder.ins().udiv(lhs, rhs),
        BinaryOp::SRem => builder.ins().srem(lhs, rhs),
        BinaryOp::URem => builder.ins().urem(lhs, rhs),

        // Float arithmetic
        BinaryOp::FAdd => builder.ins().fadd(lhs, rhs),
        BinaryOp::FSub => builder.ins().fsub(lhs, rhs),
        BinaryOp::FMul => builder.ins().fmul(lhs, rhs),
        BinaryOp::FDiv => builder.ins().fdiv(lhs, rhs),
        BinaryOp::FRem => {
            // Cranelift doesn't have frem, use a workaround
            let div = builder.ins().fdiv(lhs, rhs);
            let trunc = builder.ins().trunc(div);
            let mul = builder.ins().fmul(trunc, rhs);
            builder.ins().fsub(lhs, mul)
        }

        // Bitwise
        BinaryOp::And => builder.ins().band(lhs, rhs),
        BinaryOp::Or => builder.ins().bor(lhs, rhs),
        BinaryOp::Xor => builder.ins().bxor(lhs, rhs),
        BinaryOp::Shl => builder.ins().ishl(lhs, rhs),
        BinaryOp::AShr => builder.ins().sshr(lhs, rhs),
        BinaryOp::LShr => builder.ins().ushr(lhs, rhs),

        // Integer comparison
        BinaryOp::Eq => builder.ins().icmp(IntCC::Equal, lhs, rhs),
        BinaryOp::Ne => builder.ins().icmp(IntCC::NotEqual, lhs, rhs),
        BinaryOp::SLt => builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs),
        BinaryOp::SLe => builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs),
        BinaryOp::SGt => builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs),
        BinaryOp::SGe => builder
            .ins()
            .icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs),
        BinaryOp::ULt => builder.ins().icmp(IntCC::UnsignedLessThan, lhs, rhs),
        BinaryOp::ULe => builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, lhs, rhs),
        BinaryOp::UGt => builder.ins().icmp(IntCC::UnsignedGreaterThan, lhs, rhs),
        BinaryOp::UGe => builder
            .ins()
            .icmp(IntCC::UnsignedGreaterThanOrEqual, lhs, rhs),

        // Float comparison
        BinaryOp::FOEq => builder.ins().fcmp(FloatCC::Equal, lhs, rhs),
        BinaryOp::FONe => builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs),
        BinaryOp::FOLt => builder.ins().fcmp(FloatCC::LessThan, lhs, rhs),
        BinaryOp::FOLe => builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs),
        BinaryOp::FOGt => builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs),
        BinaryOp::FOGe => builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs),
        BinaryOp::Concat => {
            // String concatenation - not supported in JIT, would need runtime call
            return Err("String concatenation not supported in JIT".to_string());
        }
    };

    // Comparisons return i8, may need to extend for result type
    let result_cl_ty = hlir_to_cranelift_type(result_ty);
    let result_ty_current = builder.func.dfg.value_type(result);

    if result_ty_current != result_cl_ty && result_cl_ty.is_int() {
        if result_ty_current.bits() < result_cl_ty.bits() {
            Ok(builder.ins().uextend(result_cl_ty, result))
        } else {
            Ok(result)
        }
    } else {
        Ok(result)
    }
}

#[cfg(feature = "jit")]
fn translate_unary_op(
    builder: &mut FunctionBuilder,
    op: UnaryOp,
    val: cranelift_codegen::ir::Value,
    ty: &HlirType,
) -> Result<cranelift_codegen::ir::Value, String> {
    match op {
        UnaryOp::Neg => Ok(builder.ins().ineg(val)),
        UnaryOp::FNeg => Ok(builder.ins().fneg(val)),
        UnaryOp::Not => {
            // Logical not: xor with all 1s
            let ones = builder.ins().iconst(hlir_to_cranelift_type(ty), -1);
            Ok(builder.ins().bxor(val, ones))
        }
    }
}

#[cfg(feature = "jit")]
fn translate_terminator(
    builder: &mut FunctionBuilder,
    term: &HlirTerminator,
    blocks: &HashMap<BlockId, cranelift_codegen::ir::Block>,
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
) -> Result<(), String> {
    match term {
        HlirTerminator::Return(val) => {
            if let Some(v) = val {
                let ret_val = get_value(values, *v)?;

                // Get expected return type from function signature
                let expected_ret_ty = builder
                    .func
                    .signature
                    .returns
                    .first()
                    .map(|p| p.value_type)
                    .unwrap_or(types::I64);
                let actual_ty = builder.func.dfg.value_type(ret_val);

                // Insert cast if types don't match
                let final_val = if actual_ty != expected_ret_ty {
                    if actual_ty.bits() > expected_ret_ty.bits() {
                        // Truncate (e.g., I64 -> I32)
                        builder.ins().ireduce(expected_ret_ty, ret_val)
                    } else if actual_ty.bits() < expected_ret_ty.bits() {
                        // Extend (e.g., I32 -> I64)
                        builder.ins().sextend(expected_ret_ty, ret_val)
                    } else {
                        ret_val
                    }
                } else {
                    ret_val
                };

                builder.ins().return_(&[final_val]);
            } else {
                builder.ins().return_(&[]);
            }
        }

        HlirTerminator::Branch(target) => {
            let target_block = blocks[target];
            builder.ins().jump(target_block, &[]);
        }

        HlirTerminator::CondBranch {
            condition,
            then_block,
            else_block,
        } => {
            let cond = get_value(values, *condition)?;
            let then_b = blocks[then_block];
            let else_b = blocks[else_block];
            builder.ins().brif(cond, then_b, &[], else_b, &[]);
        }

        HlirTerminator::Switch {
            value,
            default,
            cases,
        } => {
            let val = get_value(values, *value)?;
            let default_block = blocks[default];

            // Build switch using a chain of conditionals
            for (case_val, target) in cases {
                let target_block = blocks[target];
                let case_const = builder.ins().iconst(types::I64, *case_val);
                let cmp = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    val,
                    case_const,
                );

                let next_block = builder.create_block();
                builder.ins().brif(cmp, target_block, &[], next_block, &[]);
                builder.seal_block(next_block);
                builder.switch_to_block(next_block);
            }

            builder.ins().jump(default_block, &[]);
        }

        HlirTerminator::Unreachable => {
            builder
                .ins()
                .trap(cranelift_codegen::ir::TrapCode::unwrap_user(0));
        }
    }

    Ok(())
}

#[cfg(feature = "jit")]
fn get_value(
    values: &HashMap<ValueId, cranelift_codegen::ir::Value>,
    id: ValueId,
) -> Result<cranelift_codegen::ir::Value, String> {
    values
        .get(&id)
        .copied()
        .ok_or_else(|| format!("Value not found: {:?}", id))
}

// Non-JIT placeholder implementation
#[cfg(not(feature = "jit"))]
impl CompiledModule {
    #[allow(dead_code)]
    fn new_stub() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }
}
