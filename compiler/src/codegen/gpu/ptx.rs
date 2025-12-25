//! PTX (Parallel Thread Execution) Code Generator
//!
//! Generates NVIDIA PTX assembly from GPU IR.
//!
//! References:
//! - PTX ISA: https://docs.nvidia.com/cuda/parallel-thread-execution/
//! - CUDA C Programming Guide

use std::fmt::Write;

use super::ir::*;

/// PTX code generator
pub struct PtxCodegen {
    /// Output buffer
    output: String,

    /// Target compute capability
    sm_version: (u32, u32),

    /// PTX version
    ptx_version: (u32, u32),

    /// Current indentation level
    indent: usize,

    /// Value to register mapping
    registers: Vec<String>,

    /// Next register number per type
    reg_counters: RegCounters,

    /// Type tracking for values
    value_types: Vec<GpuType>,
}

#[derive(Default)]
struct RegCounters {
    pred: u32, // Predicate registers
    b16: u32,  // 16-bit
    b32: u32,  // 32-bit
    b64: u32,  // 64-bit
    f32: u32,  // 32-bit float
    f64: u32,  // 64-bit float
}

impl PtxCodegen {
    pub fn new(sm_version: (u32, u32)) -> Self {
        // Automatically select PTX version based on compute capability
        let ptx_version = Self::recommended_ptx_version(sm_version);
        Self {
            output: String::new(),
            sm_version,
            ptx_version,
            indent: 0,
            registers: Vec::new(),
            reg_counters: RegCounters::default(),
            value_types: Vec::new(),
        }
    }

    /// Get recommended PTX version for a compute capability
    pub fn recommended_ptx_version(sm_version: (u32, u32)) -> (u32, u32) {
        let (major, minor) = sm_version;
        let sm = major * 10 + minor;
        match sm {
            ..=69 => (6, 0),   // sm_60-69: Volta and older
            70..=74 => (6, 3), // sm_70-74: Volta
            75 => (6, 4),      // sm_75: Turing
            80..=86 => (7, 1), // sm_80-86: Ampere
            87 => (7, 4),      // sm_87: Ampere (Jetson)
            89 => (8, 1),      // sm_89: Ada Lovelace
            90 => (8, 3),      // sm_90: Hopper
            100 => (8, 5),     // sm_100: Blackwell
            120 => (8, 6),     // sm_120: Blackwell Ultra
            _ => (8, 5),       // Future architectures
        }
    }

    /// Check if the current target supports a feature
    pub fn supports_feature(&self, feature: &str) -> bool {
        let sm = self.sm_version.0 * 10 + self.sm_version.1;
        match feature {
            "bf16" => sm >= 80,
            "fp8" => sm >= 89,
            "tma" => sm >= 90,
            "clusters" => sm >= 90,
            "wgmma" => sm >= 90,
            "fp4" => sm >= 100,
            "tensor_gen5" => sm >= 100,
            "decompression" => sm >= 100,
            "nvlink5" => sm >= 100,
            _ => false,
        }
    }

    /// Generate PTX code from GPU module
    pub fn generate(&mut self, module: &GpuModule) -> String {
        self.output.clear();
        self.emit_header(module);

        // Emit constants
        for constant in &module.constants {
            self.emit_constant(constant);
        }

        // Emit device functions
        for func in module.device_functions.values() {
            self.emit_device_function(func);
        }

        // Emit kernels
        for kernel in module.kernels.values() {
            self.emit_kernel(kernel);
        }

        self.output.clone()
    }

    fn emit_header(&mut self, _module: &GpuModule) {
        writeln!(
            self.output,
            ".version {}.{}",
            self.ptx_version.0, self.ptx_version.1
        )
        .unwrap();

        writeln!(
            self.output,
            ".target sm_{}{}",
            self.sm_version.0, self.sm_version.1
        )
        .unwrap();

        writeln!(self.output, ".address_size 64").unwrap();

        writeln!(self.output).unwrap();
    }

    fn emit_kernel(&mut self, kernel: &GpuKernel) {
        // Reset registers
        self.registers.clear();
        self.reg_counters = RegCounters::default();
        self.value_types.clear();

        // Kernel entry
        writeln!(self.output, ".visible .entry {}(", kernel.name).unwrap();

        // Parameters
        for (i, param) in kernel.params.iter().enumerate() {
            let ptx_type = self.gpu_type_to_ptx(&param.ty);
            let comma = if i < kernel.params.len() - 1 { "," } else { "" };
            writeln!(
                self.output,
                "\t.param {} param_{}{}",
                ptx_type, param.name, comma
            )
            .unwrap();
        }

        writeln!(self.output, ")").unwrap();

        // Max threads hint
        if let Some(max_threads) = kernel.max_threads {
            writeln!(self.output, ".maxntid {}, 1, 1", max_threads).unwrap();
        }

        writeln!(self.output, "{{").unwrap();

        self.indent = 1;

        // Declare registers
        self.emit_register_declarations(kernel);

        // Shared memory declarations
        for shared in &kernel.shared_memory {
            self.emit_shared_memory(shared);
        }

        writeln!(self.output).unwrap();

        // Basic blocks
        for block in &kernel.blocks {
            self.emit_block(block);
        }

        self.indent = 0;
        writeln!(self.output, "}}").unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_device_function(&mut self, func: &GpuFunction) {
        self.registers.clear();
        self.reg_counters = RegCounters::default();
        self.value_types.clear();

        let ret_type = self.gpu_type_to_ptx(&func.return_type);

        if func.return_type != GpuType::Void {
            writeln!(self.output, ".func ({} retval) {}(", ret_type, func.name).unwrap();
        } else {
            writeln!(self.output, ".func {}(", func.name).unwrap();
        }

        for (i, param) in func.params.iter().enumerate() {
            let ptx_type = self.gpu_type_to_ptx(&param.ty);
            let comma = if i < func.params.len() - 1 { "," } else { "" };
            writeln!(
                self.output,
                "\t.param {} param_{}{}",
                ptx_type, param.name, comma
            )
            .unwrap();
        }

        writeln!(self.output, ")").unwrap();
        writeln!(self.output, "{{").unwrap();

        self.indent = 1;
        self.emit_register_declarations_func(func);

        for block in &func.blocks {
            self.emit_block(block);
        }

        self.indent = 0;
        writeln!(self.output, "}}").unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_block(&mut self, block: &GpuBlock) {
        // Block label
        writeln!(self.output, "{}:", block.label).unwrap();

        // Instructions
        for (value_id, op) in &block.instructions {
            self.emit_instruction(*value_id, op);
        }

        // Terminator
        self.emit_terminator(&block.terminator);
    }

    fn emit_instruction(&mut self, _value_id: ValueId, op: &GpuOp) {
        let indent = "\t".repeat(self.indent);

        match op {
            // Constants
            GpuOp::ConstInt(n, ty) => {
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(ty);
                writeln!(self.output, "{}mov.{} {}, {};", indent, suffix, reg, n).unwrap();
            }

            GpuOp::ConstFloat(n, ty) => {
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F32) {
                    "f32"
                } else {
                    "f64"
                };
                // Format float with proper PTX representation
                if n.is_nan() {
                    writeln!(self.output, "{}mov.{} {}, 0x7FC00000;", indent, suffix, reg).unwrap();
                } else if n.is_infinite() {
                    let val = if *n > 0.0 { "0x7F800000" } else { "0xFF800000" };
                    writeln!(self.output, "{}mov.{} {}, {};", indent, suffix, reg, val).unwrap();
                } else {
                    writeln!(
                        self.output,
                        "{}mov.{} {}, 0F{:08X};",
                        indent,
                        suffix,
                        reg,
                        (*n as f32).to_bits()
                    )
                    .unwrap();
                }
            }

            GpuOp::ConstBool(b) => {
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let val = if *b { 1 } else { 0 };
                writeln!(self.output, "{}setp.eq.u32 {}, {}, 1;", indent, reg, val).unwrap();
            }

            // Integer Arithmetic
            GpuOp::Add(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}add.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Sub(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}sub.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Mul(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}mul.lo.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Div(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}div.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Rem(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}rem.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Neg(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, reg, v).unwrap();
            }

            // Float arithmetic
            GpuOp::FAdd(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}add.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FSub(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}sub.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FMul(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}mul.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FDiv(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}div.approx.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FNeg(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, reg, v).unwrap();
            }

            GpuOp::FMulAdd(a, b, c) => {
                let ra = self.get_register(*a);
                let rb = self.get_register(*b);
                let rc = self.get_register(*c);
                let ty = self.get_value_type(*a);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}fma.rn.{} {}, {}, {}, {};",
                    indent, suffix, reg, ra, rb, rc
                )
                .unwrap();
            }

            // Fast math
            GpuOp::FastSin(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}sin.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastCos(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}cos.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastExp(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}ex2.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastLog(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}lg2.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastSqrt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}sqrt.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FastRsqrt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}rsqrt.approx.f32 {}, {};", indent, reg, v).unwrap();
            }

            // Integer Comparisons
            GpuOp::Lt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.lt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Le(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.le.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Gt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.gt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Ge(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.ge.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Eq(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.eq.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Ne(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}setp.ne.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            // Float Comparisons
            GpuOp::FLt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.lt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FLe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.le.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FGt(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.gt.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FGe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.ge.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FEq(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.eq.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::FNe(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                let suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}setp.ne.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            // Logical operations
            GpuOp::And(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}and.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Or(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}or.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Xor(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}xor.pred {}, {}, {};", indent, reg, l, r).unwrap();
            }

            GpuOp::Not(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_pred_register();
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}not.pred {}, {};", indent, reg, v).unwrap();
            }

            // Bit operations
            GpuOp::Shl(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}shl.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::Shr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}shr.{} {}, {}, {};",
                    indent, suffix, reg, l, r
                )
                .unwrap();
            }

            GpuOp::LShr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}shr.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitAnd(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}and.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitOr(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}or.b{} {}, {}, {};", indent, bits, reg, l, r).unwrap();
            }

            GpuOp::BitXor(lhs, rhs) => {
                let l = self.get_register(*lhs);
                let r = self.get_register(*rhs);
                let ty = self.get_value_type(*lhs);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}xor.b{} {}, {}, {};",
                    indent, bits, reg, l, r
                )
                .unwrap();
            }

            GpuOp::BitNot(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}not.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::PopCount(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}popc.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::Clz(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}clz.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            GpuOp::Ctz(val) => {
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                // PTX doesn't have ctz, emulate with bfind
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}bfind.u{} {}, {};", indent, bits, reg, v).unwrap();
            }

            // Conversions
            GpuOp::Trunc(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.s64 {}, {};",
                    indent, dst_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::ZExt(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::SExt(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::FpTrunc(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}cvt.rn.f32.f64 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FpExt(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}cvt.f64.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::FpToSi(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = if matches!(src_ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}cvt.rzi.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::FpToUi(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = self.type_suffix(ty);
                let src_suffix = if matches!(src_ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                writeln!(
                    self.output,
                    "{}cvt.rzi.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::SiToFp(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.rn.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::UiToFp(val, ty) => {
                let v = self.get_register(*val);
                let src_ty = self.get_value_type(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let dst_suffix = if matches!(ty, GpuType::F64) {
                    "f64"
                } else {
                    "f32"
                };
                let src_suffix = self.type_suffix(&src_ty);
                writeln!(
                    self.output,
                    "{}cvt.rn.{}.{} {}, {};",
                    indent, dst_suffix, src_suffix, reg, v
                )
                .unwrap();
            }

            GpuOp::Bitcast(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(self.output, "{}mov.b{} {}, {};", indent, bits, reg, v).unwrap();
            }

            // === Modern ML Type Conversions (BF16/FP8/F4) ===
            GpuOp::F32ToBF16(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::BF16);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::BF16);
                // PTX 8.0+: cvt.rn.bf16.f32 (round to nearest)
                writeln!(self.output, "{}cvt.rn.bf16.f32 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::BF16ToF32(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                // PTX 8.0+: cvt.f32.bf16
                writeln!(self.output, "{}cvt.f32.bf16 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::F32ToF8E4M3(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F8E4M3);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F8E4M3);
                // PTX 8.1+ (sm_89+): cvt.rn.satfinite.e4m3x2.f32
                // Single value version via packed conversion
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(
                    self.output,
                    "{}cvt.rn.satfinite.e4m3x2.f32 {}, {}, 0f00000000;",
                    indent, tmp, v
                )
                .unwrap();
                writeln!(self.output, "{}and.b16 {}, {}, 0x00FF;", indent, reg, tmp).unwrap();
            }

            GpuOp::F8E4M3ToF32(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                // PTX 8.1+ (sm_89+): cvt.f32.e4m3
                // Extend to 16-bit, then convert
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(self.output, "{}cvt.u16.u8 {}, {};", indent, tmp, v).unwrap();
                writeln!(
                    self.output,
                    "{}cvt.f32.e4m3x2 {}, {{_, {}}};",
                    indent, reg, tmp
                )
                .unwrap();
            }

            GpuOp::F32ToF8E5M2(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F8E5M2);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F8E5M2);
                // PTX 8.1+ (sm_89+): cvt.rn.satfinite.e5m2x2.f32
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(
                    self.output,
                    "{}cvt.rn.satfinite.e5m2x2.f32 {}, {}, 0f00000000;",
                    indent, tmp, v
                )
                .unwrap();
                writeln!(self.output, "{}and.b16 {}, {}, 0x00FF;", indent, reg, tmp).unwrap();
            }

            GpuOp::F8E5M2ToF32(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                // PTX 8.1+ (sm_89+): cvt.f32.e5m2
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(self.output, "{}cvt.u16.u8 {}, {};", indent, tmp, v).unwrap();
                writeln!(
                    self.output,
                    "{}cvt.f32.e5m2x2 {}, {{_, {}}};",
                    indent, reg, tmp
                )
                .unwrap();
            }

            GpuOp::F32ToF4(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F4);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F4);
                // F4 requires software emulation - no native PTX support
                // Simplified: quantize to 4-bit via truncation
                // Format: 1 sign, 2 exp, 1 mantissa (similar to FP8 E4M3 but half precision)
                writeln!(
                    self.output,
                    "{}// F4 quantization (software emulation)",
                    indent
                )
                .unwrap();
                let tmp = self.alloc_register(&GpuType::U32);
                writeln!(self.output, "{}mov.b32 {}, {};", indent, tmp, v).unwrap();
                // Extract sign (bit 31), exp (bits 30-23), mantissa (bit 22)
                writeln!(self.output, "{}shr.u32 {}, {}, 28;", indent, reg, tmp).unwrap();
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, reg, reg).unwrap();
            }

            GpuOp::F4ToF32(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                // F4 dequantization (software emulation)
                writeln!(
                    self.output,
                    "{}// F4 dequantization (software emulation)",
                    indent
                )
                .unwrap();
                let tmp = self.alloc_register(&GpuType::U32);
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, tmp, v).unwrap();
                writeln!(self.output, "{}shl.b32 {}, {}, 28;", indent, tmp, tmp).unwrap();
                writeln!(self.output, "{}mov.b32 {}, {};", indent, reg, tmp).unwrap();
            }

            // === Packed ML Type Operations ===
            GpuOp::PackF8x2(lo, hi) => {
                let lo_v = self.get_register(*lo);
                let hi_v = self.get_register(*hi);
                let reg = self.alloc_register(&GpuType::U16);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U16);
                // Pack two u8 values into u16: (hi << 8) | lo
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(self.output, "{}cvt.u16.u8 {}, {};", indent, reg, lo_v).unwrap();
                writeln!(self.output, "{}cvt.u16.u8 {}, {};", indent, tmp, hi_v).unwrap();
                writeln!(self.output, "{}shl.b16 {}, {}, 8;", indent, tmp, tmp).unwrap();
                writeln!(self.output, "{}or.b16 {}, {}, {};", indent, reg, reg, tmp).unwrap();
            }

            GpuOp::UnpackF8x2Low(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Extract low byte: val & 0xFF
                writeln!(self.output, "{}and.b16 {}, {}, 0x00FF;", indent, reg, v).unwrap();
            }

            GpuOp::UnpackF8x2High(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Extract high byte: (val >> 8) & 0xFF
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(self.output, "{}shr.b16 {}, {}, 8;", indent, tmp, v).unwrap();
                writeln!(self.output, "{}and.b16 {}, {}, 0x00FF;", indent, reg, tmp).unwrap();
            }

            GpuOp::PackF4x2(lo, hi) => {
                let lo_v = self.get_register(*lo);
                let hi_v = self.get_register(*hi);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Pack two 4-bit values into byte: (hi << 4) | (lo & 0x0F)
                let tmp = self.alloc_register(&GpuType::U8);
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, reg, lo_v).unwrap();
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, tmp, hi_v).unwrap();
                writeln!(self.output, "{}shl.b32 {}, {}, 4;", indent, tmp, tmp).unwrap();
                writeln!(self.output, "{}or.b32 {}, {}, {};", indent, reg, reg, tmp).unwrap();
            }

            GpuOp::UnpackF4x2Low(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Extract low nibble: val & 0x0F
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, reg, v).unwrap();
            }

            GpuOp::UnpackF4x2High(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Extract high nibble: (val >> 4) & 0x0F
                let tmp = self.alloc_register(&GpuType::U8);
                writeln!(self.output, "{}shr.b32 {}, {}, 4;", indent, tmp, v).unwrap();
                writeln!(self.output, "{}and.b32 {}, {}, 0x0F;", indent, reg, tmp).unwrap();
            }

            // === Quantization Utilities ===
            GpuOp::QuantizeF32ToF8(val, mode) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // Use E4M3 format by default with specified rounding mode
                let rnd = match mode {
                    QuantizeMode::RoundNearestEven => "rn",
                    QuantizeMode::RoundTowardZero => "rz",
                    QuantizeMode::RoundTowardPosInf => "rp",
                    QuantizeMode::RoundTowardNegInf => "rm",
                    QuantizeMode::Stochastic => "rn", // Fallback to RNE for stochastic
                };
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(
                    self.output,
                    "{}cvt.{}.satfinite.e4m3x2.f32 {}, {}, 0f00000000;",
                    indent, rnd, tmp, v
                )
                .unwrap();
                writeln!(self.output, "{}and.b16 {}, {}, 0x00FF;", indent, reg, tmp).unwrap();
            }

            GpuOp::DequantizeF8ToF32(val, scale) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                // Convert F8 to F32
                let tmp = self.alloc_register(&GpuType::U16);
                writeln!(self.output, "{}cvt.u16.u8 {}, {};", indent, tmp, v).unwrap();
                writeln!(
                    self.output,
                    "{}cvt.f32.e4m3x2 {}, {{_, {}}};",
                    indent, reg, tmp
                )
                .unwrap();
                // Apply optional scale factor
                if let Some(scale_val) = scale {
                    let s = self.get_register(*scale_val);
                    writeln!(self.output, "{}mul.f32 {}, {}, {};", indent, reg, reg, s).unwrap();
                }
            }

            // === INT8/INT4 Quantization (Phase 11) ===
            GpuOp::QuantizeF32ToInt8 {
                value,
                scale,
                zero_point,
                symmetric,
            } => {
                let v = self.get_register(*value);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::I8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I8);

                // q = clamp(round(x / scale) + zero_point, -128, 127)
                let tmp_f32 = self.alloc_register(&GpuType::F32);
                let tmp_i32 = self.alloc_register(&GpuType::I32);

                writeln!(self.output, "{}// Quantize F32 to INT8", indent).unwrap();
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32, v, s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, tmp_i32, tmp_f32
                )
                .unwrap();
                if !symmetric {
                    // Add zero_point for asymmetric quantization
                    writeln!(
                        self.output,
                        "{}add.s32 {}, {}, {};",
                        indent, tmp_i32, tmp_i32, zp
                    )
                    .unwrap();
                }
                // Clamp to [-128, 127]
                writeln!(
                    self.output,
                    "{}max.s32 {}, {}, -128;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}min.s32 {}, {}, 127;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(self.output, "{}cvt.s8.s32 {}, {};", indent, reg, tmp_i32).unwrap();
            }

            GpuOp::DequantizeInt8ToF32 {
                value,
                scale,
                zero_point,
            } => {
                let v = self.get_register(*value);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                // x = (q - zero_point) * scale
                let tmp_i32 = self.alloc_register(&GpuType::I32);
                let tmp_f32 = self.alloc_register(&GpuType::F32);

                writeln!(self.output, "{}// Dequantize INT8 to F32", indent).unwrap();
                writeln!(self.output, "{}cvt.s32.s8 {}, {};", indent, tmp_i32, v).unwrap();
                writeln!(
                    self.output,
                    "{}sub.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rn.f32.s32 {}, {};",
                    indent, tmp_f32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mul.f32 {}, {}, {};",
                    indent, reg, tmp_f32, s
                )
                .unwrap();
            }

            GpuOp::QuantizeF32ToUint8 {
                value,
                scale,
                zero_point,
            } => {
                let v = self.get_register(*value);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);

                // q = clamp(round(x / scale) + zero_point, 0, 255)
                let tmp_f32 = self.alloc_register(&GpuType::F32);
                let tmp_i32 = self.alloc_register(&GpuType::I32);

                writeln!(self.output, "{}// Quantize F32 to UINT8", indent).unwrap();
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32, v, s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, tmp_i32, tmp_f32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}add.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, zp
                )
                .unwrap();
                // Clamp to [0, 255]
                writeln!(
                    self.output,
                    "{}max.s32 {}, {}, 0;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}min.s32 {}, {}, 255;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(self.output, "{}cvt.u8.s32 {}, {};", indent, reg, tmp_i32).unwrap();
            }

            GpuOp::DequantizeUint8ToF32 {
                value,
                scale,
                zero_point,
            } => {
                let v = self.get_register(*value);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                // x = (q - zero_point) * scale
                let tmp_i32 = self.alloc_register(&GpuType::I32);
                let tmp_f32 = self.alloc_register(&GpuType::F32);

                writeln!(self.output, "{}// Dequantize UINT8 to F32", indent).unwrap();
                writeln!(self.output, "{}cvt.s32.u8 {}, {};", indent, tmp_i32, v).unwrap();
                writeln!(
                    self.output,
                    "{}sub.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rn.f32.s32 {}, {};",
                    indent, tmp_f32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mul.f32 {}, {}, {};",
                    indent, reg, tmp_f32, s
                )
                .unwrap();
            }

            GpuOp::QuantizeF32ToInt4 {
                value_lo,
                value_hi,
                scale,
                zero_point,
            } => {
                let v_lo = self.get_register(*value_lo);
                let v_hi = self.get_register(*value_hi);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);

                // Pack two INT4 values into one byte
                let tmp_f32 = self.alloc_register(&GpuType::F32);
                let tmp_i32_lo = self.alloc_register(&GpuType::I32);
                let tmp_i32_hi = self.alloc_register(&GpuType::I32);

                writeln!(self.output, "{}// Quantize F32 to INT4 (packed)", indent).unwrap();
                // Quantize low nibble
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32, v_lo, s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, tmp_i32_lo, tmp_f32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}add.s32 {}, {}, {};",
                    indent, tmp_i32_lo, tmp_i32_lo, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}max.s32 {}, {}, -8;",
                    indent, tmp_i32_lo, tmp_i32_lo
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}min.s32 {}, {}, 7;",
                    indent, tmp_i32_lo, tmp_i32_lo
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}and.b32 {}, {}, 0x0F;",
                    indent, tmp_i32_lo, tmp_i32_lo
                )
                .unwrap();

                // Quantize high nibble
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32, v_hi, s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, tmp_i32_hi, tmp_f32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}add.s32 {}, {}, {};",
                    indent, tmp_i32_hi, tmp_i32_hi, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}max.s32 {}, {}, -8;",
                    indent, tmp_i32_hi, tmp_i32_hi
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}min.s32 {}, {}, 7;",
                    indent, tmp_i32_hi, tmp_i32_hi
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}shl.b32 {}, {}, 4;",
                    indent, tmp_i32_hi, tmp_i32_hi
                )
                .unwrap();

                // Pack into single byte
                writeln!(
                    self.output,
                    "{}or.b32 {}, {}, {};",
                    indent, tmp_i32_lo, tmp_i32_lo, tmp_i32_hi
                )
                .unwrap();
                writeln!(self.output, "{}cvt.u8.s32 {}, {};", indent, reg, tmp_i32_lo).unwrap();
            }

            GpuOp::DequantizeInt4ToF32Lo {
                packed,
                scale,
                zero_point,
            } => {
                let p = self.get_register(*packed);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                let tmp_i32 = self.alloc_register(&GpuType::I32);
                let tmp_f32 = self.alloc_register(&GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Dequantize INT4 (low nibble) to F32",
                    indent
                )
                .unwrap();
                // Extract low nibble and sign-extend
                writeln!(self.output, "{}cvt.s32.u8 {}, {};", indent, tmp_i32, p).unwrap();
                writeln!(
                    self.output,
                    "{}and.b32 {}, {}, 0x0F;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                // Sign extend from 4-bit
                writeln!(
                    self.output,
                    "{}shl.b32 {}, {}, 28;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}shr.s32 {}, {}, 28;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                // Dequantize: (q - zp) * scale
                writeln!(
                    self.output,
                    "{}sub.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rn.f32.s32 {}, {};",
                    indent, tmp_f32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mul.f32 {}, {}, {};",
                    indent, reg, tmp_f32, s
                )
                .unwrap();
            }

            GpuOp::DequantizeInt4ToF32Hi {
                packed,
                scale,
                zero_point,
            } => {
                let p = self.get_register(*packed);
                let s = self.get_register(*scale);
                let zp = self.get_register(*zero_point);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                let tmp_i32 = self.alloc_register(&GpuType::I32);
                let tmp_f32 = self.alloc_register(&GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Dequantize INT4 (high nibble) to F32",
                    indent
                )
                .unwrap();
                // Extract high nibble and sign-extend
                writeln!(self.output, "{}cvt.s32.u8 {}, {};", indent, tmp_i32, p).unwrap();
                writeln!(
                    self.output,
                    "{}shr.b32 {}, {}, 4;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                // Sign extend from 4-bit
                writeln!(
                    self.output,
                    "{}shl.b32 {}, {}, 28;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}shr.s32 {}, {}, 28;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                // Dequantize: (q - zp) * scale
                writeln!(
                    self.output,
                    "{}sub.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rn.f32.s32 {}, {};",
                    indent, tmp_f32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mul.f32 {}, {}, {};",
                    indent, reg, tmp_f32, s
                )
                .unwrap();
            }

            // dp4a - INT8 dot product (sm_61+, Pascal and later)
            GpuOp::Dp4a { a, b, c } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);

                writeln!(
                    self.output,
                    "{}// dp4a: c + dot(a[0:3], b[0:3]) (sm_61+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}dp4a.s32.s32 {}, {}, {}, {};",
                    indent, reg, a_reg, b_reg, c_reg
                )
                .unwrap();
            }

            GpuOp::Dp4aUnsigned { a, b, c } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);

                writeln!(
                    self.output,
                    "{}// dp4a.u32: unsigned INT8 dot product (sm_61+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}dp4a.u32.u32 {}, {}, {}, {};",
                    indent, reg, a_reg, b_reg, c_reg
                )
                .unwrap();
            }

            GpuOp::Dp4aSU { a, b, c } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);

                writeln!(
                    self.output,
                    "{}// dp4a.s32.u32: mixed signed/unsigned dot product (sm_61+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}dp4a.s32.u32 {}, {}, {}, {};",
                    indent, reg, a_reg, b_reg, c_reg
                )
                .unwrap();
            }

            GpuOp::Int8MatMul {
                a,
                b,
                c,
                m,
                n,
                k,
                a_scale,
                b_scale,
            } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let a_s = self.get_register(*a_scale);
                let b_s = self.get_register(*b_scale);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);

                writeln!(
                    self.output,
                    "{}// INT8 Matrix Multiply with Tensor Cores (sm_75+)",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}// M={}, N={}, K={}", indent, m, n, k).unwrap();
                writeln!(
                    self.output,
                    "{}// Note: Requires WMMA API or mma.sync instruction",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mma.sync.aligned.m{}n{}k{}.s32.s8.s8.s32",
                    indent, m, n, k
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                    indent, c_reg, a_reg, b_reg, c_reg
                )
                .unwrap();
                // Apply dequantization scales
                writeln!(
                    self.output,
                    "{}// Apply dequant scales: scale_a={}, scale_b={}",
                    indent, a_s, b_s
                )
                .unwrap();
                writeln!(self.output, "{}mov.s32 {}, {};", indent, reg, c_reg).unwrap();
            }

            GpuOp::QuantizePerChannel {
                values,
                scales,
                zero_points,
                axis,
                num_channels,
                signed,
            } => {
                let v = self.get_register(*values);
                let s = self.get_register(*scales);
                let zp = self.get_register(*zero_points);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);

                writeln!(
                    self.output,
                    "{}// Per-channel quantization (axis={}, channels={})",
                    indent, axis, num_channels
                )
                .unwrap();
                if *signed {
                    writeln!(self.output, "{}// Output: INT8 (signed)", indent).unwrap();
                } else {
                    writeln!(self.output, "{}// Output: UINT8 (unsigned)", indent).unwrap();
                }
                writeln!(
                    self.output,
                    "{}// Scale and zero_point arrays at: {}, {}",
                    indent, s, zp
                )
                .unwrap();
                // Per-channel quantization is typically done in a loop at the IR level
                writeln!(
                    self.output,
                    "{}// Placeholder: actual implementation depends on tensor layout",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}mov.u32 {}, 0;", indent, reg).unwrap();
            }

            GpuOp::DequantizePerChannel {
                values,
                scales,
                zero_points,
                axis,
                num_channels,
            } => {
                let v = self.get_register(*values);
                let s = self.get_register(*scales);
                let zp = self.get_register(*zero_points);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Per-channel dequantization (axis={}, channels={})",
                    indent, axis, num_channels
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Scale and zero_point arrays at: {}, {}",
                    indent, s, zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Placeholder: actual implementation depends on tensor layout",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}mov.f32 {}, 0f00000000;", indent, reg).unwrap();
            }

            GpuOp::ComputeQuantScale {
                min_val,
                max_val,
                num_bits,
                symmetric,
            } => {
                let min_v = self.get_register(*min_val);
                let max_v = self.get_register(*max_val);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);

                let tmp_range = self.alloc_register(&GpuType::F32);
                let tmp_qrange = self.alloc_register(&GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Compute quantization scale ({}-bit, symmetric={})",
                    indent, num_bits, symmetric
                )
                .unwrap();
                if *symmetric {
                    // scale = max(|min|, |max|) / 127 (for INT8)
                    let qmax = (1 << (num_bits - 1)) - 1;
                    let tmp_abs = self.alloc_register(&GpuType::F32);
                    writeln!(self.output, "{}abs.f32 {}, {};", indent, tmp_abs, min_v).unwrap();
                    writeln!(self.output, "{}abs.f32 {}, {};", indent, tmp_range, max_v).unwrap();
                    writeln!(
                        self.output,
                        "{}max.f32 {}, {}, {};",
                        indent, tmp_range, tmp_range, tmp_abs
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}mov.f32 {}, 0f{:08X};",
                        indent,
                        tmp_qrange,
                        (qmax as f32).to_bits()
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}div.rn.f32 {}, {}, {};",
                        indent, reg, tmp_range, tmp_qrange
                    )
                    .unwrap();
                } else {
                    // scale = (max - min) / (qmax - qmin)
                    let qmax = (1 << *num_bits) - 1;
                    writeln!(
                        self.output,
                        "{}sub.f32 {}, {}, {};",
                        indent, tmp_range, max_v, min_v
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}mov.f32 {}, 0f{:08X};",
                        indent,
                        tmp_qrange,
                        (qmax as f32).to_bits()
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}div.rn.f32 {}, {}, {};",
                        indent, reg, tmp_range, tmp_qrange
                    )
                    .unwrap();
                }
            }

            GpuOp::ComputeZeroPoint {
                min_val,
                scale,
                num_bits,
            } => {
                let min_v = self.get_register(*min_val);
                let s = self.get_register(*scale);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);

                let tmp_f32 = self.alloc_register(&GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Compute zero point ({}-bit)",
                    indent, num_bits
                )
                .unwrap();
                // zero_point = round(-min / scale)
                writeln!(self.output, "{}neg.f32 {}, {};", indent, tmp_f32, min_v).unwrap();
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32, tmp_f32, s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, reg, tmp_f32
                )
                .unwrap();
            }

            GpuOp::FindMinMax { values, count } => {
                let v = self.get_register(*values);
                let cnt = self.get_register(*count);
                // Returns a vec2 containing (min, max)
                let reg = self.alloc_register(&GpuType::Vec2(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec2(Box::new(GpuType::F32)));

                writeln!(
                    self.output,
                    "{}// Find min/max in tensor (reduction)",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}// Input: {} values at {}", indent, cnt, v).unwrap();
                writeln!(
                    self.output,
                    "{}// Placeholder: reduction implemented at higher level",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mov.v2.f32 {}, {{0f00000000, 0f00000000}};",
                    indent, reg
                )
                .unwrap();
            }

            GpuOp::Requantize {
                value,
                in_scale,
                in_zero_point,
                out_scale,
                out_zero_point,
            } => {
                let v = self.get_register(*value);
                let in_s = self.get_register(*in_scale);
                let in_zp = self.get_register(*in_zero_point);
                let out_s = self.get_register(*out_scale);
                let out_zp = self.get_register(*out_zero_point);
                let reg = self.alloc_register(&GpuType::I8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I8);

                let tmp_i32 = self.alloc_register(&GpuType::I32);
                let tmp_f32 = self.alloc_register(&GpuType::F32);
                let tmp_f32_2 = self.alloc_register(&GpuType::F32);

                writeln!(
                    self.output,
                    "{}// Requantize from one scale to another",
                    indent
                )
                .unwrap();
                // Dequantize: x = (q - in_zp) * in_scale
                writeln!(self.output, "{}cvt.s32.s8 {}, {};", indent, tmp_i32, v).unwrap();
                writeln!(
                    self.output,
                    "{}sub.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, in_zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rn.f32.s32 {}, {};",
                    indent, tmp_f32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mul.f32 {}, {}, {};",
                    indent, tmp_f32, tmp_f32, in_s
                )
                .unwrap();
                // Quantize: q = round(x / out_scale) + out_zp
                writeln!(
                    self.output,
                    "{}div.rn.f32 {}, {}, {};",
                    indent, tmp_f32_2, tmp_f32, out_s
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}cvt.rni.s32.f32 {}, {};",
                    indent, tmp_i32, tmp_f32_2
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}add.s32 {}, {}, {};",
                    indent, tmp_i32, tmp_i32, out_zp
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}max.s32 {}, {}, -128;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}min.s32 {}, {}, 127;",
                    indent, tmp_i32, tmp_i32
                )
                .unwrap();
                writeln!(self.output, "{}cvt.s8.s32 {}, {};", indent, reg, tmp_i32).unwrap();
            }

            // === Blackwell Features (sm_100+) ===
            GpuOp::TmaLoadAsync {
                dst_shared,
                src_global,
                size,
                barrier,
            } => {
                let dst = self.get_register(*dst_shared);
                let src = self.get_register(*src_global);
                let bar = self.get_register(*barrier);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// TMA async load (sm_90+)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    [{}, {{{}}}], [{}], [{}];",
                    indent, dst, size, src, bar
                )
                .unwrap();
            }

            GpuOp::TmaStoreAsync {
                dst_global,
                src_shared,
                size,
            } => {
                let dst = self.get_register(*dst_global);
                let src = self.get_register(*src_shared);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// TMA async store (sm_90+)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}cp.async.bulk.tensor.1d.global.shared::cta.bulk_group",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    [{}], [{}, {{{}}}];",
                    indent, dst, src, size
                )
                .unwrap();
            }

            GpuOp::TmaMulticastLoad {
                dst_shared,
                src_global,
                size,
                cluster_mask,
                barrier,
            } => {
                let dst = self.get_register(*dst_shared);
                let src = self.get_register(*src_global);
                let bar = self.get_register(*barrier);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// TMA multicast load (sm_90+)", indent).unwrap();
                writeln!(self.output, "{}cp.async.bulk.tensor.1d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster",
                         indent).unwrap();
                writeln!(
                    self.output,
                    "{}    [{}, {{{}}}], [{}], [{}], 0x{:x};",
                    indent, dst, size, src, bar, cluster_mask
                )
                .unwrap();
            }

            GpuOp::TmaReduceAsync {
                dst_global,
                src_shared,
                size,
                reduce_op,
            } => {
                let dst = self.get_register(*dst_global);
                let src = self.get_register(*src_shared);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// TMA reduce async (sm_100+)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}cp.reduce.async.bulk.tensor.1d.global.shared::cta.{}",
                    indent, reduce_op
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    [{}], [{}, {{{}}}];",
                    indent, dst, src, size
                )
                .unwrap();
            }

            // 5th-gen Tensor Core operations (sm_100+)
            GpuOp::WgmmaFp4 {
                a,
                b,
                c,
                m,
                n,
                k,
                scale_a,
                scale_b,
            } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let sa = self.get_register(*scale_a);
                let sb = self.get_register(*scale_b);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(
                    self.output,
                    "{}// WGMMA FP4 (sm_100+ 5th-gen Tensor Cores)",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}// M={}, N={}, K={}", indent, m, n, k).unwrap();
                writeln!(
                    self.output,
                    "{}wgmma.mma_async.sync.aligned.m{}n{}k{}.f32.e2m1.e2m1",
                    indent, m, n, k
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    {{{}}}, {{{}}}, {{{}}}, {}, {};",
                    indent, c_reg, a_reg, b_reg, sa, sb
                )
                .unwrap();
                writeln!(self.output, "{}mov.f32 {}, {};", indent, reg, c_reg).unwrap();
            }

            GpuOp::WgmmaFp8 {
                a,
                b,
                c,
                m,
                n,
                k,
                format,
            } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                let fmt = match format {
                    Fp8Format::E4M3 => "e4m3",
                    Fp8Format::E5M2 => "e5m2",
                };
                writeln!(self.output, "{}// WGMMA FP8 {} (sm_89+)", indent, fmt).unwrap();
                writeln!(
                    self.output,
                    "{}wgmma.mma_async.sync.aligned.m{}n{}k{}.f32.{}.{}",
                    indent, m, n, k, fmt, fmt
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    {{{}}}, {{{}}}, {{{}}};",
                    indent, c_reg, a_reg, b_reg
                )
                .unwrap();
                writeln!(self.output, "{}mov.f32 {}, {};", indent, reg, c_reg).unwrap();
            }

            GpuOp::WgmmaBf16 { a, b, c, m, n, k } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}// WGMMA BF16 (sm_90+)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}wgmma.mma_async.sync.aligned.m{}n{}k{}.f32.bf16.bf16",
                    indent, m, n, k
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}    {{{}}}, {{{}}}, {{{}}};",
                    indent, c_reg, a_reg, b_reg
                )
                .unwrap();
                writeln!(self.output, "{}mov.f32 {}, {};", indent, reg, c_reg).unwrap();
            }

            // Transformer Engine v2 (sm_100+)
            GpuOp::TransformerEngineFusedAttention {
                q,
                k,
                v,
                scale,
                output,
                format,
            } => {
                let q_reg = self.get_register(*q);
                let k_reg = self.get_register(*k);
                let v_reg = self.get_register(*v);
                let s_reg = self.get_register(*scale);
                let o_reg = self.get_register(*output);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// Transformer Engine Fused Attention (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Format: {} - This maps to cuDNN fused attention",
                    indent, format
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Placeholder: call external TE library",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// te.fused_attention({}, {}, {}, {}, {});",
                    indent, q_reg, k_reg, v_reg, s_reg, o_reg
                )
                .unwrap();
            }

            GpuOp::TransformerEngineFp8Gemm {
                a,
                b,
                c,
                amax_out,
                format,
            } => {
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);
                let c_reg = self.get_register(*c);
                let amax = self.get_register(*amax_out);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// Transformer Engine FP8 GEMM with amax (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Format: {} - Dynamic scaling",
                    indent, format
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// te.fp8_gemm({}, {}, {}, amax={});",
                    indent, a_reg, b_reg, c_reg, amax
                )
                .unwrap();
            }

            // Decompression Engine (sm_100+)
            GpuOp::DecompressLz4 {
                dst,
                src,
                compressed_size,
                uncompressed_size,
            } => {
                let d = self.get_register(*dst);
                let s = self.get_register(*src);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// Hardware LZ4 decompression (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}decompress.lz4 [{}, {}], [{}, {}];",
                    indent, d, uncompressed_size, s, compressed_size
                )
                .unwrap();
            }

            GpuOp::DecompressSnappy {
                dst,
                src,
                compressed_size,
                uncompressed_size,
            } => {
                let d = self.get_register(*dst);
                let s = self.get_register(*src);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// Hardware Snappy decompression (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}decompress.snappy [{}, {}], [{}, {}];",
                    indent, d, uncompressed_size, s, compressed_size
                )
                .unwrap();
            }

            GpuOp::DecompressDeflate {
                dst,
                src,
                compressed_size,
                uncompressed_size,
            } => {
                let d = self.get_register(*dst);
                let s = self.get_register(*src);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// Hardware Deflate decompression (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}decompress.deflate [{}, {}], [{}, {}];",
                    indent, d, uncompressed_size, s, compressed_size
                )
                .unwrap();
            }

            // Cluster operations (sm_90+, enhanced in sm_100)
            GpuOp::ClusterId => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %clusterid;", indent, reg).unwrap();
            }

            GpuOp::ClusterDim => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nclusterid;", indent, reg).unwrap();
            }

            GpuOp::BlockIdInCluster => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %cluster_ctaid;", indent, reg).unwrap();
            }

            GpuOp::ClusterBarrier => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}barrier.cluster.sync.aligned;", indent).unwrap();
            }

            GpuOp::ClusterArrive(barrier) => {
                let bar = self.get_register(*barrier);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}mbarrier.arrive.shared::cluster [{}];",
                    indent, bar
                )
                .unwrap();
            }

            GpuOp::ClusterWait(barrier) => {
                let bar = self.get_register(*barrier);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}mbarrier.wait.shared::cluster [{}];",
                    indent, bar
                )
                .unwrap();
            }

            // NVLink 5.0 Operations (sm_100+)
            GpuOp::NvlinkRead {
                dst,
                src_gpu,
                src_addr,
                size,
            } => {
                let d = self.get_register(*dst);
                let sa = self.get_register(*src_addr);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// NVLink 5.0 remote read (sm_100+)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}// rdma.read gpu={} size={}",
                    indent, src_gpu, size
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}ld.global.nc.b8 {}, [{} + gpu{}];",
                    indent, d, sa, src_gpu
                )
                .unwrap();
            }

            GpuOp::NvlinkWrite {
                dst_gpu,
                dst_addr,
                src,
                size,
            } => {
                let da = self.get_register(*dst_addr);
                let s = self.get_register(*src);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(
                    self.output,
                    "{}// NVLink 5.0 remote write (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// rdma.write gpu={} size={}",
                    indent, dst_gpu, size
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}st.global.b8 [{} + gpu{}], {};",
                    indent, da, dst_gpu, s
                )
                .unwrap();
            }

            GpuOp::NvlinkAtomicAdd {
                dst_gpu,
                dst_addr,
                value,
            } => {
                let da = self.get_register(*dst_addr);
                let v = self.get_register(*value);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(
                    self.output,
                    "{}// NVLink 5.0 remote atomic (sm_100+)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}atom.global.add.u64 {}, [{} + gpu{}], {};",
                    indent, reg, da, dst_gpu, v
                )
                .unwrap();
            }

            // Memory operations
            GpuOp::Load(ptr, space) => {
                let p = self.get_register(*ptr);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                let space_str = self.memory_space_to_ptx(*space);
                writeln!(
                    self.output,
                    "{}ld{}.u64 {}, [{}];",
                    indent, space_str, reg, p
                )
                .unwrap();
            }

            GpuOp::Store(ptr, val, space) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let space_str = self.memory_space_to_ptx(*space);
                self.registers.push("_".to_string()); // Dummy for void op
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}st{}.u64 [{}], {};", indent, space_str, p, v).unwrap();
            }

            // Atomic operations
            GpuOp::AtomicAdd(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.add.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicSub(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let neg_reg = self.alloc_register(&ty);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                // Atomic sub via negation + add
                writeln!(self.output, "{}neg.{} {}, {};", indent, suffix, neg_reg, v).unwrap();
                writeln!(
                    self.output,
                    "{}atom.global.add.{} {}, [{}], {};",
                    indent, suffix, reg, p, neg_reg
                )
                .unwrap();
            }

            GpuOp::AtomicMin(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.min.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicMax(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}atom.global.max.{} {}, [{}], {};",
                    indent, suffix, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicAnd(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.and.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicOr(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.or.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicXor(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.xor.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicExch(ptr, val) => {
                let p = self.get_register(*ptr);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.exch.b{} {}, [{}], {};",
                    indent, bits, reg, p, v
                )
                .unwrap();
            }

            GpuOp::AtomicCas(ptr, cmp, val) => {
                let p = self.get_register(*ptr);
                let c = self.get_register(*cmp);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let bits = ty.size_bytes() * 8;
                writeln!(
                    self.output,
                    "{}atom.global.cas.b{} {}, [{}], {}, {};",
                    indent, bits, reg, p, c, v
                )
                .unwrap();
            }

            // Address computation
            GpuOp::GetElementPtr(ptr, indices) => {
                let p = self.get_register(*ptr);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types
                    .push(GpuType::Ptr(Box::new(GpuType::U8), MemorySpace::Global));
                // Simple offset calculation
                if indices.is_empty() {
                    writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, p).unwrap();
                } else {
                    let idx = self.get_register(indices[0]);
                    writeln!(self.output, "{}add.u64 {}, {}, {};", indent, reg, p, idx).unwrap();
                }
            }

            GpuOp::PtrToInt(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, v).unwrap();
            }

            GpuOp::IntToPtr(val, ty) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, v).unwrap();
            }

            // GPU intrinsics
            GpuOp::ThreadIdX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.x;", indent, reg).unwrap();
            }

            GpuOp::ThreadIdY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.y;", indent, reg).unwrap();
            }

            GpuOp::ThreadIdZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %tid.z;", indent, reg).unwrap();
            }

            GpuOp::BlockIdX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.x;", indent, reg).unwrap();
            }

            GpuOp::BlockIdY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.y;", indent, reg).unwrap();
            }

            GpuOp::BlockIdZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ctaid.z;", indent, reg).unwrap();
            }

            GpuOp::BlockDimX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.x;", indent, reg).unwrap();
            }

            GpuOp::BlockDimY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.y;", indent, reg).unwrap();
            }

            GpuOp::BlockDimZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %ntid.z;", indent, reg).unwrap();
            }

            GpuOp::GridDimX => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.x;", indent, reg).unwrap();
            }

            GpuOp::GridDimY => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.y;", indent, reg).unwrap();
            }

            GpuOp::GridDimZ => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %nctaid.z;", indent, reg).unwrap();
            }

            GpuOp::WarpId => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %warpid;", indent, reg).unwrap();
            }

            GpuOp::LaneId => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, %laneid;", indent, reg).unwrap();
            }

            GpuOp::WarpSize => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}mov.u32 {}, WARP_SZ;", indent, reg).unwrap();
            }

            // Synchronization
            GpuOp::SyncThreads => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}bar.sync 0;", indent).unwrap();
            }

            GpuOp::SyncWarp(mask) => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}bar.warp.sync 0x{:08x};", indent, mask).unwrap();
            }

            GpuOp::MemoryFence(space) => {
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                let fence_type = match space {
                    MemorySpace::Global => "membar.gl;",
                    MemorySpace::Shared => "membar.cta;",
                    _ => "membar.sys;",
                };
                writeln!(self.output, "{}{}", indent, fence_type).unwrap();
            }

            // Warp operations
            GpuOp::WarpShuffle(val, lane) => {
                let v = self.get_register(*val);
                let l = self.get_register(*lane);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.idx.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, l
                )
                .unwrap();
            }

            GpuOp::WarpShuffleUp(val, delta) => {
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.up.b32 {}, {}, {}, 0, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            GpuOp::WarpShuffleDown(val, delta) => {
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.down.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            GpuOp::WarpShuffleXor(val, mask) => {
                let v = self.get_register(*val);
                let m = self.get_register(*mask);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                writeln!(
                    self.output,
                    "{}shfl.sync.bfly.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, m
                )
                .unwrap();
            }

            GpuOp::WarpVote(vote_op, val) => {
                let v = self.get_register(*val);
                let reg = match vote_op {
                    WarpVoteOp::Ballot => {
                        let r = self.alloc_register(&GpuType::U32);
                        writeln!(
                            self.output,
                            "{}vote.sync.ballot.b32 {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::U32);
                        r
                    }
                    WarpVoteOp::All => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.all.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                    WarpVoteOp::Any => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.any.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                    WarpVoteOp::Eq => {
                        let r = self.alloc_pred_register();
                        writeln!(
                            self.output,
                            "{}vote.sync.uni.pred {}, {}, 0xffffffff;",
                            indent, r, v
                        )
                        .unwrap();
                        self.value_types.push(GpuType::Bool);
                        r
                    }
                };
                self.registers.push(reg);
            }

            GpuOp::WarpReduce(reduce_op, val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::I32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I32);
                let op_name = match reduce_op {
                    WarpReduceOp::Add => "add",
                    WarpReduceOp::Min => "min",
                    WarpReduceOp::Max => "max",
                    WarpReduceOp::And => "and",
                    WarpReduceOp::Or => "or",
                    WarpReduceOp::Xor => "xor",
                };
                writeln!(
                    self.output,
                    "{}redux.sync.{}.s32 {}, {}, 0xffffffff;",
                    indent, op_name, reg, v
                )
                .unwrap();
            }

            GpuOp::WarpMatch(val) => {
                let v = self.get_register(*val);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(
                    self.output,
                    "{}match.sync.any.b32 {}, {}, 0xffffffff;",
                    indent, reg, v
                )
                .unwrap();
            }

            // Texture operations (simplified)
            GpuOp::TexFetch(tex, coord) => {
                let _t = self.get_register(*tex);
                let _c = self.get_register(*coord);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}// tex.1d.v4.f32 not implemented", indent).unwrap();
            }

            GpuOp::TexFetch2D(tex, x, y) => {
                let _t = self.get_register(*tex);
                let _xr = self.get_register(*x);
                let _yr = self.get_register(*y);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(self.output, "{}// tex.2d.v4.f32 not implemented", indent).unwrap();
            }

            GpuOp::SurfRead(surf, coord) => {
                let _s = self.get_register(*surf);
                let _c = self.get_register(*coord);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// suld.b.1d not implemented", indent).unwrap();
            }

            GpuOp::SurfWrite(surf, coord, val) => {
                let _s = self.get_register(*surf);
                let _c = self.get_register(*coord);
                let _v = self.get_register(*val);
                self.registers.push("_".to_string());
                self.value_types.push(GpuType::Void);
                writeln!(self.output, "{}// sust.b.1d not implemented", indent).unwrap();
            }

            // Select
            GpuOp::Select(cond, t, f) => {
                let c = self.get_register(*cond);
                let tv = self.get_register(*t);
                let fv = self.get_register(*f);
                let ty = self.get_value_type(*t);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(
                    self.output,
                    "{}selp.{} {}, {}, {}, {};",
                    indent, suffix, reg, tv, fv, c
                )
                .unwrap();
            }

            // Function call
            GpuOp::Call(name, args) => {
                let arg_regs: Vec<_> = args
                    .iter()
                    .map(|a| self.get_register(*a).to_string())
                    .collect();
                let reg = self.alloc_register(&GpuType::I64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::I64);

                writeln!(self.output, "{}{{", indent).unwrap();
                writeln!(self.output, "{}\t.param .b64 retval;", indent).unwrap();
                for (i, arg) in arg_regs.iter().enumerate() {
                    writeln!(self.output, "{}\t.param .b64 arg{};", indent, i).unwrap();
                    writeln!(self.output, "{}\tst.param.b64 [arg{}], {};", indent, i, arg).unwrap();
                }
                write!(self.output, "{}\tcall (retval), {}, (", indent, name).unwrap();
                for (i, _) in args.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    write!(self.output, "arg{}", i).unwrap();
                }
                writeln!(self.output, ");").unwrap();
                writeln!(self.output, "{}\tld.param.b64 {}, [retval];", indent, reg).unwrap();
                writeln!(self.output, "{}}}", indent).unwrap();
            }

            // Parameter
            GpuOp::Param(idx) => {
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(
                    self.output,
                    "{}ld.param.u64 {}, [param_{}];",
                    indent, reg, idx
                )
                .unwrap();
            }

            // Shared memory address
            GpuOp::SharedAddr(name) => {
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types
                    .push(GpuType::Ptr(Box::new(GpuType::U8), MemorySpace::Shared));
                writeln!(self.output, "{}mov.u64 {}, {};", indent, reg, name).unwrap();
            }

            // Phi (should be lowered before PTX emission)
            GpuOp::Phi(_) => {
                // Phi nodes should be eliminated before PTX generation
                self.registers.push("phi_placeholder".to_string());
                self.value_types.push(GpuType::I64);
            }

            // === Bio/Quaternion Operations (from Quaternionic Syntax preprint) ===

            // Quaternion multiplication (Hamilton product)
            // q1 * q2 = (w1w2 - x1x2 - y1y2 - z1z2)
            //         + (w1x2 + x1w2 + y1z2 - z1y2)i
            //         + (w1y2 - x1z2 + y1w2 + z1x2)j
            //         + (w1z2 + x1y2 - y1x2 + z1w2)k
            GpuOp::QuatMul(q1, q2) => {
                let _r1 = self.get_register(*q1);
                let _r2 = self.get_register(*q2);
                // Output is vec4<f32> stored in 4 consecutive f32 registers
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                // Emit inline quaternion multiply code
                writeln!(
                    self.output,
                    "{}// QuatMul: Hamilton product (noncommutative)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Result = q1 * q2 stored in {}",
                    indent, reg
                )
                .unwrap();
                writeln!(self.output, "{}call.uni __quat_mul, ({});", indent, reg).unwrap();
            }

            // Quaternion conjugate: q* = w - xi - yj - zk
            GpuOp::QuatConj(q) => {
                let _r = self.get_register(*q);
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                writeln!(self.output, "{}// QuatConj: negate imaginary parts", indent).unwrap();
                writeln!(self.output, "{}call.uni __quat_conj, ({});", indent, reg).unwrap();
            }

            // Quaternion norm squared: |q|² = w² + x² + y² + z²
            GpuOp::QuatNormSq(q) => {
                let _r = self.get_register(*q);
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::F32);
                writeln!(
                    self.output,
                    "{}// QuatNormSq: |q|² = w² + x² + y² + z²",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}call.uni __quat_norm_sq, ({});", indent, reg).unwrap();
            }

            // Quaternion normalize to unit: q / |q|
            GpuOp::QuatNormalize(q) => {
                let _r = self.get_register(*q);
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                writeln!(self.output, "{}// QuatNormalize: q / |q| for SU(2)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}call.uni __quat_normalize, ({});",
                    indent, reg
                )
                .unwrap();
            }

            // Quaternion SLERP (spherical linear interpolation)
            GpuOp::QuatSlerp(q1, q2, t) => {
                let _r1 = self.get_register(*q1);
                let _r2 = self.get_register(*q2);
                let _rt = self.get_register(*t);
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                writeln!(
                    self.output,
                    "{}// QuatSlerp: spherical interpolation",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}call.uni __quat_slerp, ({});", indent, reg).unwrap();
            }

            // DNA base complement (A↔T, C↔G)
            // Uses XOR with 3: complement(x) = x ^ 3
            GpuOp::DnaComplement(base) => {
                let r = self.get_register(*base);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                writeln!(
                    self.output,
                    "{}// DnaComplement: A↔T (0↔3), C↔G (1↔2)",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}xor.b32 {}, {}, 3;", indent, reg, r).unwrap();
            }

            // GF(4) addition (characteristic 2 field, uses lookup table)
            GpuOp::Gf4Add(a, b) => {
                let ra = self.get_register(*a);
                let rb = self.get_register(*b);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                // GF(4) addition is XOR of indices into addition table
                // But for efficiency, we inline the table lookup
                writeln!(self.output, "{}// GF(4) addition: characteristic 2", indent).unwrap();
                writeln!(
                    self.output,
                    "{}call.uni __gf4_add, ({}, {}, {});",
                    indent, reg, ra, rb
                )
                .unwrap();
            }

            // GF(4) multiplication (uses α² + α + 1 = 0)
            GpuOp::Gf4Mul(a, b) => {
                let ra = self.get_register(*a);
                let rb = self.get_register(*b);
                let reg = self.alloc_register(&GpuType::U8);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U8);
                writeln!(self.output, "{}// GF(4) multiply: α² + α + 1 = 0", indent).unwrap();
                writeln!(
                    self.output,
                    "{}call.uni __gf4_mul, ({}, {}, {});",
                    indent, reg, ra, rb
                )
                .unwrap();
            }

            // Transmission channel composition (quaternion product + renormalize)
            GpuOp::TransmissionCompose(t1, t2) => {
                let _r1 = self.get_register(*t1);
                let _r2 = self.get_register(*t2);
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                writeln!(
                    self.output,
                    "{}// TransmissionCompose: (g,t,p,e) quaternion product",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}call.uni __transmission_compose, ({});",
                    indent, reg
                )
                .unwrap();
            }

            // Transmission distortion with renormalization
            GpuOp::TransmissionDistort(trans, dg, dt, dp, de) => {
                let _rt = self.get_register(*trans);
                let _rdg = self.get_register(*dg);
                let _rdt = self.get_register(*dt);
                let _rdp = self.get_register(*dp);
                let _rde = self.get_register(*de);
                let reg = self.alloc_register(&GpuType::Vec4(Box::new(GpuType::F32)));
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Vec4(Box::new(GpuType::F32)));
                writeln!(
                    self.output,
                    "{}// TransmissionDistort: perturb + renormalize",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}call.uni __transmission_distort, ({});",
                    indent, reg
                )
                .unwrap();
            }

            // ================================================================
            // Cooperative Groups (CUDA 9.0+ / PTX 6.0+)
            // ================================================================

            // Get cooperative group handle at specified scope
            GpuOp::CoopThisGroup(scope) => {
                // Group handle is conceptual - store scope info for subsequent ops
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let scope_val = match scope {
                    CooperativeScope::Thread => 0,
                    CooperativeScope::Warp => 1,
                    CooperativeScope::Block => 2,
                    CooperativeScope::Cluster => 3,
                    CooperativeScope::Grid => 4,
                    CooperativeScope::Coalesced => 5,
                    CooperativeScope::TiledPartition(n) => 0x100 | *n,
                };
                writeln!(self.output, "{}// CoopThisGroup scope={:?}", indent, scope).unwrap();
                writeln!(self.output, "{}mov.u32 {}, {};", indent, reg, scope_val).unwrap();
            }

            // Get group size
            GpuOp::CoopGroupSize(group) => {
                let grp = self.get_register(*group);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// CoopGroupSize", indent).unwrap();
                // Check scope from group handle
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 1;", indent, grp).unwrap(); // warp?
                writeln!(self.output, "{}@p0 mov.u32 {}, 32;", indent, reg).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 2;", indent, grp).unwrap(); // block?
                writeln!(self.output, "{}@p0 mov.u32 {}, %ntid.x;", indent, reg).unwrap();
            }

            // Get thread rank within group
            GpuOp::CoopThreadRank(group) => {
                let grp = self.get_register(*group);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// CoopThreadRank", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 1;", indent, grp).unwrap(); // warp?
                writeln!(self.output, "{}@p0 mov.u32 {}, %laneid;", indent, reg).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 2;", indent, grp).unwrap(); // block?
                writeln!(self.output, "{}@p0 mov.u32 {}, %tid.x;", indent, reg).unwrap();
            }

            // Check if this thread is group leader
            GpuOp::CoopIsLeader(group) => {
                let grp = self.get_register(*group);
                let reg = self.alloc_register(&GpuType::Bool);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}// CoopIsLeader (rank == 0)", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p1, {}, 1;", indent, grp).unwrap();
                writeln!(
                    self.output,
                    "{}@p1 setp.eq.u32 {}, %laneid, 0;",
                    indent, reg
                )
                .unwrap();
                writeln!(self.output, "{}setp.eq.u32 p1, {}, 2;", indent, grp).unwrap();
                writeln!(self.output, "{}@p1 setp.eq.u32 {}, %tid.x, 0;", indent, reg).unwrap();
            }

            // Synchronize group
            GpuOp::CoopSync(group) => {
                let grp = self.get_register(*group);
                writeln!(self.output, "{}// CoopSync", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 1;", indent, grp).unwrap(); // warp?
                writeln!(self.output, "{}@p0 bar.warp.sync 0xffffffff;", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 2;", indent, grp).unwrap(); // block?
                writeln!(self.output, "{}@p0 bar.sync 0;", indent).unwrap();
                // For cluster (sm_90+)
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 3;", indent, grp).unwrap();
                writeln!(self.output, "{}@p0 barrier.cluster.arrive.release;", indent).unwrap();
                writeln!(self.output, "{}@p0 barrier.cluster.wait.acquire;", indent).unwrap();
                // No-op for thread scope
            }

            // Shuffle broadcast (all threads get value from src_rank)
            GpuOp::CoopShfl(group, val, src_rank) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let src = self.get_register(*src_rank);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let suffix = self.type_suffix(&ty);
                writeln!(self.output, "{}// CoopShfl broadcast", indent).unwrap();
                writeln!(
                    self.output,
                    "{}shfl.sync.idx.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, src
                )
                .unwrap();
            }

            // Shuffle with index
            GpuOp::CoopShflIdx(group, val, idx) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let i = self.get_register(*idx);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}// CoopShflIdx", indent).unwrap();
                writeln!(
                    self.output,
                    "{}shfl.sync.idx.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, i
                )
                .unwrap();
            }

            // Shuffle up (get from thread rank - delta)
            GpuOp::CoopShflUp(group, val, delta) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}// CoopShflUp", indent).unwrap();
                writeln!(
                    self.output,
                    "{}shfl.sync.up.b32 {}, {}, {}, 0, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            // Shuffle down (get from thread rank + delta)
            GpuOp::CoopShflDown(group, val, delta) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let d = self.get_register(*delta);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}// CoopShflDown", indent).unwrap();
                writeln!(
                    self.output,
                    "{}shfl.sync.down.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, d
                )
                .unwrap();
            }

            // Shuffle XOR (butterfly pattern)
            GpuOp::CoopShflXor(group, val, mask) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let m = self.get_register(*mask);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(self.output, "{}// CoopShflXor (butterfly)", indent).unwrap();
                writeln!(
                    self.output,
                    "{}shfl.sync.bfly.b32 {}, {}, {}, 31, 0xffffffff;",
                    indent, reg, v, m
                )
                .unwrap();
            }

            // Collective reduce
            GpuOp::CoopReduce(group, val, op) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                let op_str = match op {
                    CoopReduceOp::Add => "add",
                    CoopReduceOp::Min => "min",
                    CoopReduceOp::Max => "max",
                    CoopReduceOp::And => "and",
                    CoopReduceOp::Or => "or",
                    CoopReduceOp::Xor => "xor",
                    CoopReduceOp::Mul => "add", // No native mul, use add as fallback
                };
                let type_suffix = if ty.is_float() { ".f32" } else { ".s32" };
                writeln!(self.output, "{}// CoopReduce {:?}", indent, op).unwrap();
                // Use redux.sync for warp-level reduction (sm_80+)
                writeln!(
                    self.output,
                    "{}redux.sync.{}{} {}, {}, 0xffffffff;",
                    indent, op_str, type_suffix, reg, v
                )
                .unwrap();
            }

            // Inclusive scan
            GpuOp::CoopInclusiveScan(group, val, op) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(
                    self.output,
                    "{}// CoopInclusiveScan {:?} - Kogge-Stone pattern",
                    indent, op
                )
                .unwrap();
                // Implement via shuffle tree (Kogge-Stone)
                writeln!(self.output, "{}mov.b32 {}, {};", indent, reg, v).unwrap();
                for delta in [1, 2, 4, 8, 16] {
                    let tmp = format!("tmp_scan_{}", delta);
                    writeln!(self.output, "{}{{\n{}\t.reg .b32 {};", indent, indent, tmp).unwrap();
                    writeln!(
                        self.output,
                        "{}\tshfl.sync.up.b32 {}, {}, {}, 0, 0xffffffff;",
                        indent, tmp, reg, delta
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tadd.s32 {}, {}, {};",
                        indent, reg, reg, tmp
                    )
                    .unwrap();
                    writeln!(self.output, "{}}}", indent).unwrap();
                }
            }

            // Exclusive scan
            GpuOp::CoopExclusiveScan(group, val, op) => {
                let _grp = self.get_register(*group);
                let v = self.get_register(*val);
                let ty = self.get_value_type(*val);
                let reg = self.alloc_register(&ty);
                self.registers.push(reg.clone());
                self.value_types.push(ty.clone());
                writeln!(
                    self.output,
                    "{}// CoopExclusiveScan {:?} - shift + inclusive",
                    indent, op
                )
                .unwrap();
                // Exclusive = shift(inclusive, 1) with identity at lane 0
                writeln!(self.output, "{}mov.b32 {}, {};", indent, reg, v).unwrap();
                for delta in [1, 2, 4, 8, 16] {
                    let tmp = format!("tmp_escan_{}", delta);
                    writeln!(self.output, "{}{{\n{}\t.reg .b32 {};", indent, indent, tmp).unwrap();
                    writeln!(
                        self.output,
                        "{}\tshfl.sync.up.b32 {}, {}, {}, 0, 0xffffffff;",
                        indent, tmp, reg, delta
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tadd.s32 {}, {}, {};",
                        indent, reg, reg, tmp
                    )
                    .unwrap();
                    writeln!(self.output, "{}}}", indent).unwrap();
                }
                // Shift result down by 1, put 0 in lane 0
                writeln!(
                    self.output,
                    "{}shfl.sync.up.b32 {}, {}, 1, 0, 0xffffffff;",
                    indent, reg, reg
                )
                .unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, %laneid, 0;", indent).unwrap();
                writeln!(self.output, "{}@p0 mov.b32 {}, 0;", indent, reg).unwrap();
            }

            // Collective ballot
            GpuOp::CoopBallot(group, pred) => {
                let _grp = self.get_register(*group);
                let p = self.get_register(*pred);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// CoopBallot", indent).unwrap();
                writeln!(
                    self.output,
                    "{}vote.sync.ballot.b32 {}, {}, 0xffffffff;",
                    indent, reg, p
                )
                .unwrap();
            }

            // All threads predicate true
            GpuOp::CoopAll(group, pred) => {
                let _grp = self.get_register(*group);
                let p = self.get_register(*pred);
                let reg = self.alloc_register(&GpuType::Bool);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}// CoopAll", indent).unwrap();
                writeln!(
                    self.output,
                    "{}vote.sync.all.pred {}, {}, 0xffffffff;",
                    indent, reg, p
                )
                .unwrap();
            }

            // Any thread predicate true
            GpuOp::CoopAny(group, pred) => {
                let _grp = self.get_register(*group);
                let p = self.get_register(*pred);
                let reg = self.alloc_register(&GpuType::Bool);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}// CoopAny", indent).unwrap();
                writeln!(
                    self.output,
                    "{}vote.sync.any.pred {}, {}, 0xffffffff;",
                    indent, reg, p
                )
                .unwrap();
            }

            // Partition group into tiles
            GpuOp::CoopPartitionTiled(group, tile_size) => {
                let _grp = self.get_register(*group);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                let scope_val = 0x100 | *tile_size;
                writeln!(
                    self.output,
                    "{}// CoopPartitionTiled size={}",
                    indent, tile_size
                )
                .unwrap();
                writeln!(self.output, "{}mov.u32 {}, {};", indent, reg, scope_val).unwrap();
            }

            // Binary partition by predicate
            GpuOp::CoopPartitionBinary(group, pred) => {
                let _grp = self.get_register(*group);
                let p = self.get_register(*pred);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// CoopPartitionBinary", indent).unwrap();
                // Use match.sync to get mask of matching threads
                writeln!(
                    self.output,
                    "{}match.sync.any.b32 {}, {}, 0xffffffff;",
                    indent, reg, p
                )
                .unwrap();
            }

            // Labeled partition
            GpuOp::CoopPartitionLabeled(group, label) => {
                let _grp = self.get_register(*group);
                let l = self.get_register(*label);
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(self.output, "{}// CoopPartitionLabeled", indent).unwrap();
                // match.sync groups threads by label value
                writeln!(
                    self.output,
                    "{}match.sync.any.b32 {}, {}, 0xffffffff;",
                    indent, reg, l
                )
                .unwrap();
            }

            // Get coalesced threads (active mask)
            GpuOp::CoopCoalescedThreads => {
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(
                    self.output,
                    "{}// CoopCoalescedThreads (activemask)",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}activemask.b32 {};", indent, reg).unwrap();
            }

            // Elect a single leader
            GpuOp::CoopElect(group) => {
                let _grp = self.get_register(*group);
                let reg = self.alloc_register(&GpuType::Bool);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::Bool);
                writeln!(self.output, "{}// CoopElect (first active thread)", indent).unwrap();
                // Get active mask, find first set bit, compare with laneid
                writeln!(
                    self.output,
                    "{}{{\n{}\t.reg .b32 mask, first;",
                    indent, indent
                )
                .unwrap();
                writeln!(self.output, "{}\tactivemask.b32 mask;", indent).unwrap();
                writeln!(self.output, "{}\tbrev.b32 first, mask;", indent).unwrap();
                writeln!(self.output, "{}\tclz.b32 first, first;", indent).unwrap();
                writeln!(
                    self.output,
                    "{}\tsetp.eq.u32 {}, first, %laneid;",
                    indent, reg
                )
                .unwrap();
                writeln!(self.output, "{}}}", indent).unwrap();
            }

            // Memory fence at group scope
            GpuOp::CoopMemoryFence(group) => {
                let grp = self.get_register(*group);
                writeln!(self.output, "{}// CoopMemoryFence", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 1;", indent, grp).unwrap();
                writeln!(self.output, "{}@p0 fence.acq_rel.cta;", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 2;", indent, grp).unwrap();
                writeln!(self.output, "{}@p0 fence.acq_rel.cta;", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 3;", indent, grp).unwrap();
                writeln!(self.output, "{}@p0 fence.acq_rel.cluster;", indent).unwrap();
                writeln!(self.output, "{}setp.eq.u32 p0, {}, 4;", indent, grp).unwrap();
                writeln!(self.output, "{}@p0 fence.acq_rel.sys;", indent).unwrap();
            }

            // === Debug/Profiling Operations ===

            // Printf from GPU using vprintf
            // PTX vprintf requires: format string pointer, argument buffer pointer
            GpuOp::Printf(fmt_id, args) => {
                writeln!(
                    self.output,
                    "{}// gpu.printf (format_id={})",
                    indent, fmt_id
                )
                .unwrap();

                // Allocate result register for return value
                let ret_reg = self.alloc_register(&GpuType::I32);
                self.registers.push(ret_reg.clone());
                self.value_types.push(GpuType::I32);

                // Build argument buffer in local memory (each arg is 8 bytes for alignment)
                let arg_buf_size = args.len() * 8;
                if arg_buf_size > 0 {
                    writeln!(
                        self.output,
                        "{}{{\n{}\t.local .align 8 .b8 __printf_args[{}];",
                        indent, indent, arg_buf_size
                    )
                    .unwrap();
                    writeln!(self.output, "{}\t.reg .b64 buf_addr;", indent).unwrap();
                    writeln!(self.output, "{}\tmov.u64 buf_addr, __printf_args;", indent).unwrap();

                    // Store each argument to the buffer
                    for (i, arg) in args.iter().enumerate() {
                        let arg_reg = self.get_register(*arg);
                        let offset = i * 8;
                        // Use generic store since we don't know exact type
                        writeln!(
                            self.output,
                            "{}\tst.local.b64 [buf_addr+{}], {};",
                            indent, offset, arg_reg
                        )
                        .unwrap();
                    }

                    // Call vprintf with format string and argument buffer
                    writeln!(self.output, "{}\t.param .b64 fmt_param;", indent).unwrap();
                    writeln!(self.output, "{}\t.param .b64 buf_param;", indent).unwrap();
                    writeln!(self.output, "{}\t.param .b32 ret_param;", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}\tst.param.b64 [fmt_param], __printf_fmt_{};",
                        indent, fmt_id
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tst.param.b64 [buf_param], buf_addr;",
                        indent
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tcall.uni (ret_param), vprintf, (fmt_param, buf_param);",
                        indent
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tld.param.b32 {}, [ret_param];",
                        indent, ret_reg
                    )
                    .unwrap();
                    writeln!(self.output, "{}}}", indent).unwrap();
                } else {
                    // No args - pass null buffer
                    writeln!(
                        self.output,
                        "{}{{\n{}\t.param .b64 fmt_param;",
                        indent, indent
                    )
                    .unwrap();
                    writeln!(self.output, "{}\t.param .b64 buf_param;", indent).unwrap();
                    writeln!(self.output, "{}\t.param .b32 ret_param;", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}\tst.param.b64 [fmt_param], __printf_fmt_{};",
                        indent, fmt_id
                    )
                    .unwrap();
                    writeln!(self.output, "{}\tmov.u64 %rd0, 0;", indent).unwrap();
                    writeln!(self.output, "{}\tst.param.b64 [buf_param], %rd0;", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}\tcall.uni (ret_param), vprintf, (fmt_param, buf_param);",
                        indent
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}\tld.param.b32 {}, [ret_param];",
                        indent, ret_reg
                    )
                    .unwrap();
                    writeln!(self.output, "{}}}", indent).unwrap();
                }
            }

            // Assert condition (trap if false)
            GpuOp::Assert(cond, msg_id) => {
                let c = self.get_register(*cond);
                writeln!(self.output, "{}// gpu.assert", indent).unwrap();
                writeln!(self.output, "{}setp.eq.s32 p_assert, {}, 0;", indent, c).unwrap();
                if let Some(msg) = msg_id {
                    writeln!(
                        self.output,
                        "{}@p_assert {{ // assertion failed: msg_id={}\n{}\ttrap;\n{}}}",
                        indent, msg, indent, indent
                    )
                    .unwrap();
                } else {
                    writeln!(self.output, "{}@p_assert trap;", indent).unwrap();
                }
            }

            // Unconditional trap
            GpuOp::Trap => {
                writeln!(self.output, "{}trap;", indent).unwrap();
            }

            // Software breakpoint
            GpuOp::Brkpt => {
                writeln!(self.output, "{}brkpt;", indent).unwrap();
            }

            // Read clock counter (64-bit)
            GpuOp::Clock => {
                let ret_reg = self.alloc_register(&GpuType::U64);
                self.registers.push(ret_reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(self.output, "{}mov.u64 {}, %clock64;", indent, ret_reg).unwrap();
            }

            // Read global timer (64-bit nanoseconds)
            GpuOp::GlobalTimer => {
                let ret_reg = self.alloc_register(&GpuType::U64);
                self.registers.push(ret_reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(self.output, "{}mov.u64 {}, %globaltimer;", indent, ret_reg).unwrap();
            }

            // Performance monitoring event (requires profiler)
            GpuOp::PmEvent(event_id) => {
                writeln!(
                    self.output,
                    "{}// pmevent {} (enabled only under profiler)",
                    indent, event_id
                )
                .unwrap();
                // pmevent instruction is only meaningful when running under Nsight
                // Uncomment to enable: writeln!(self.output, "{}pmevent {};", indent, event_id).unwrap();
            }

            // ========================================
            // Tile Programming Operations (CUDA 13)
            // ========================================
            GpuOp::TileCreate {
                tile_m,
                tile_n,
                element_type,
                layout,
                ..
            } => {
                // Allocate shared memory for tile with proper alignment
                let elem_size = crate::codegen::gpu::tile::element_size_bytes(element_type);
                let smem_bytes =
                    crate::codegen::gpu::tile::shared_memory_bytes(*tile_m, *tile_n, element_type);
                let layout_str = match layout {
                    TileLayout::RowMajor => "row_major",
                    TileLayout::ColMajor => "col_major",
                    TileLayout::Swizzled { .. } => "swizzled",
                };
                writeln!(
                    self.output,
                    "{}// TileCreate {}x{} {:?} ({})",
                    indent, tile_m, tile_n, element_type, layout_str
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}.shared .align {} .b8 tile_smem[{}];",
                    indent,
                    elem_size.max(16),
                    smem_bytes
                )
                .unwrap();
                // Result is pointer to shared memory
                let reg = self.alloc_register(&GpuType::U64);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U64);
                writeln!(self.output, "{}mov.u64 {}, tile_smem;", indent, reg).unwrap();
            }

            GpuOp::TileLoad {
                tile,
                src_ptr,
                stride,
                barrier,
            } => {
                let tile_reg = self.get_register(*tile);
                let src_reg = self.get_register(*src_ptr);
                let stride_reg = self.get_register(*stride);

                if self.sm_version.0 >= 9 && barrier.is_some() {
                    // Use TMA on Hopper+ for async bulk load
                    let barrier_reg = self.get_register(barrier.unwrap());
                    writeln!(self.output, "{}// TileLoad via TMA (sm_90+)", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}cp.async.bulk.shared.global [{}, 0], [{}, {}], {};",
                        indent, tile_reg, src_reg, stride_reg, barrier_reg
                    )
                    .unwrap();
                    writeln!(self.output, "{}cp.async.bulk.commit_group;", indent).unwrap();
                } else {
                    // Fallback: cooperative coalesced loads
                    writeln!(self.output, "{}// TileLoad via coalesced loads", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}// Each thread loads its element from global to shared",
                        indent
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}ld.global.b32 %r_tmp, [{} + %tid.x * 4];",
                        indent, src_reg
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}st.shared.b32 [{} + %tid.x * 4], %r_tmp;",
                        indent, tile_reg
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}bar.sync 0; // Ensure all loads complete",
                        indent
                    )
                    .unwrap();
                }
                // No result register for store-like operations
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileStore {
                tile,
                dst_ptr,
                stride,
                barrier,
            } => {
                let tile_reg = self.get_register(*tile);
                let dst_reg = self.get_register(*dst_ptr);
                let stride_reg = self.get_register(*stride);

                if self.sm_version.0 >= 9 && barrier.is_some() {
                    let barrier_reg = self.get_register(barrier.unwrap());
                    writeln!(self.output, "{}// TileStore via TMA (sm_90+)", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}cp.async.bulk.global.shared [{}, {}], [{}, 0], {};",
                        indent, dst_reg, stride_reg, tile_reg, barrier_reg
                    )
                    .unwrap();
                } else {
                    writeln!(self.output, "{}// TileStore via coalesced stores", indent).unwrap();
                    writeln!(
                        self.output,
                        "{}bar.sync 0; // Ensure tile data ready",
                        indent
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}ld.shared.b32 %r_tmp, [{} + %tid.x * 4];",
                        indent, tile_reg
                    )
                    .unwrap();
                    writeln!(
                        self.output,
                        "{}st.global.b32 [{} + %tid.x * 4], %r_tmp;",
                        indent, dst_reg
                    )
                    .unwrap();
                }
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileMma {
                c,
                a,
                b,
                tile_m,
                tile_n,
                tile_k,
            } => {
                let c_reg = self.get_register(*c);
                let a_reg = self.get_register(*a);
                let b_reg = self.get_register(*b);

                writeln!(
                    self.output,
                    "{}// TileMma {}x{}x{}",
                    indent, tile_m, tile_n, tile_k
                )
                .unwrap();
                if self.sm_version.0 >= 10 {
                    // Blackwell: Use WGMMA (warpgroup-scoped)
                    writeln!(self.output, "{}wgmma.mma_async.sync.aligned.m{}n{}k{}.f32.bf16.bf16 {{{}}}, {{{}}}, {{{}}};",
                        indent, tile_m, tile_n, tile_k, c_reg, a_reg, b_reg).unwrap();
                } else if self.sm_version.0 >= 8 {
                    // Ampere+: Use MMA (warp-scoped)
                    writeln!(self.output, "{}mma.sync.aligned.m{}n{}k{}.row.col.f32.bf16.bf16.f32 {{{}}}, {{{}}}, {{{}}}, {{{}}};",
                        indent, tile_m, tile_n, tile_k, c_reg, a_reg, b_reg, c_reg).unwrap();
                } else {
                    writeln!(
                        self.output,
                        "{}// TileMma requires sm_80+ (Ampere or newer)",
                        indent
                    )
                    .unwrap();
                }
                let reg = self.alloc_register(&GpuType::F32);
                self.registers.push(reg);
                self.value_types.push(GpuType::F32);
            }

            GpuOp::TileSync(tile) => {
                let _ = self.get_register(*tile);
                writeln!(self.output, "{}bar.sync 0; // Tile synchronization", indent).unwrap();
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileGetElement { tile, row, col } => {
                let tile_reg = self.get_register(*tile);
                let row_reg = self.get_register(*row);
                let col_reg = self.get_register(*col);

                writeln!(self.output, "{}// TileGetElement", indent).unwrap();
                let offset_reg = self.alloc_register(&GpuType::U32);
                let addr_reg = self.alloc_register(&GpuType::U64);
                let result_reg = self.alloc_register(&GpuType::F32);
                writeln!(
                    self.output,
                    "{}mad.lo.u32 {}, {}, %tile_n, {};",
                    indent, offset_reg, row_reg, col_reg
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mad.wide.u32 {}, {}, 4, {};",
                    indent, addr_reg, offset_reg, tile_reg
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}ld.shared.f32 {}, [{}];",
                    indent, result_reg, addr_reg
                )
                .unwrap();
                self.registers.push(result_reg);
                self.value_types.push(GpuType::F32);
            }

            GpuOp::TileSetElement {
                tile,
                row,
                col,
                value,
            } => {
                let tile_reg = self.get_register(*tile);
                let row_reg = self.get_register(*row);
                let col_reg = self.get_register(*col);
                let val_reg = self.get_register(*value);

                writeln!(self.output, "{}// TileSetElement", indent).unwrap();
                let offset_reg = self.alloc_register(&GpuType::U32);
                let addr_reg = self.alloc_register(&GpuType::U64);
                writeln!(
                    self.output,
                    "{}mad.lo.u32 {}, {}, %tile_n, {};",
                    indent, offset_reg, row_reg, col_reg
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}mad.wide.u32 {}, {}, 4, {};",
                    indent, addr_reg, offset_reg, tile_reg
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}st.shared.f32 [{}], {};",
                    indent, addr_reg, val_reg
                )
                .unwrap();
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileFill { tile, value } => {
                let tile_reg = self.get_register(*tile);
                let val_reg = self.get_register(*value);

                writeln!(
                    self.output,
                    "{}// TileFill - broadcast scalar to all elements",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}// Each thread fills its element", indent).unwrap();
                writeln!(
                    self.output,
                    "{}st.shared.f32 [{} + %tid.x * 4], {};",
                    indent, tile_reg, val_reg
                )
                .unwrap();
                writeln!(self.output, "{}bar.sync 0;", indent).unwrap();
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileReduce { tile, reduce_op } => {
                let tile_reg = self.get_register(*tile);
                let op_name = match reduce_op {
                    CoopReduceOp::Add => "add",
                    CoopReduceOp::Mul => "mul",
                    CoopReduceOp::Min => "min",
                    CoopReduceOp::Max => "max",
                    CoopReduceOp::And => "and",
                    CoopReduceOp::Or => "or",
                    CoopReduceOp::Xor => "xor",
                };

                writeln!(self.output, "{}// TileReduce ({})", indent, op_name).unwrap();
                // Load element, then perform warp reduction
                let val_reg = self.alloc_register(&GpuType::F32);
                writeln!(
                    self.output,
                    "{}ld.shared.f32 {}, [{} + %tid.x * 4];",
                    indent, val_reg, tile_reg
                )
                .unwrap();
                // Warp shuffle reduction
                writeln!(
                    self.output,
                    "{}redux.sync.{}.b32 {}, {}, 0xffffffff;",
                    indent, op_name, val_reg, val_reg
                )
                .unwrap();
                self.registers.push(val_reg);
                self.value_types.push(GpuType::F32);
            }

            GpuOp::TileTranspose(tile) => {
                let tile_reg = self.get_register(*tile);

                writeln!(
                    self.output,
                    "{}// TileTranspose - requires diagonal copy through shared memory",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}bar.sync 0;", indent).unwrap();
                writeln!(
                    self.output,
                    "{}// Read from (row,col), write to (col,row)",
                    indent
                )
                .unwrap();
                writeln!(
                    self.output,
                    "{}// Actual implementation requires auxiliary shared memory",
                    indent
                )
                .unwrap();
                let _ = tile_reg;
                self.registers.push(String::new());
                self.value_types.push(GpuType::Void);
            }

            GpuOp::TileM(_tile) => {
                // Should be constant-folded; emit placeholder
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(
                    self.output,
                    "{}// TileM - should be constant-folded",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}mov.u32 {}, 0; // placeholder", indent, reg).unwrap();
            }

            GpuOp::TileN(_tile) => {
                // Should be constant-folded; emit placeholder
                let reg = self.alloc_register(&GpuType::U32);
                self.registers.push(reg.clone());
                self.value_types.push(GpuType::U32);
                writeln!(
                    self.output,
                    "{}// TileN - should be constant-folded",
                    indent
                )
                .unwrap();
                writeln!(self.output, "{}mov.u32 {}, 0; // placeholder", indent, reg).unwrap();
            }
        }
    }

    fn emit_terminator(&mut self, term: &GpuTerminator) {
        let indent = "\t".repeat(self.indent);

        match term {
            GpuTerminator::Br(target) => {
                writeln!(self.output, "{}bra BB{};", indent, target.0).unwrap();
            }

            GpuTerminator::CondBr(cond, then_block, else_block) => {
                let c = self.get_register(*cond);
                writeln!(self.output, "{}@{} bra BB{};", indent, c, then_block.0).unwrap();
                writeln!(self.output, "{}bra BB{};", indent, else_block.0).unwrap();
            }

            GpuTerminator::ReturnVoid => {
                writeln!(self.output, "{}ret;", indent).unwrap();
            }

            GpuTerminator::Return(val) => {
                let v = self.get_register(*val);
                writeln!(self.output, "{}st.param.b64 [retval], {};", indent, v).unwrap();
                writeln!(self.output, "{}ret;", indent).unwrap();
            }

            GpuTerminator::Unreachable => {
                writeln!(self.output, "{}trap;", indent).unwrap();
            }
        }
    }

    fn emit_shared_memory(&mut self, shared: &SharedMemDecl) {
        let indent = "\t".repeat(self.indent);
        let ptx_type = self.gpu_type_to_ptx(&shared.elem_type);

        writeln!(
            self.output,
            "{}.shared .align {} {} {}[{}];",
            indent, shared.align, ptx_type, shared.name, shared.size
        )
        .unwrap();
    }

    fn emit_register_declarations(&mut self, _kernel: &GpuKernel) {
        let indent = "\t".repeat(self.indent);

        writeln!(self.output, "{}// Register declarations", indent).unwrap();
        writeln!(self.output, "{}.reg .pred p<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b16 r16_<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b32 r32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b64 r64_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f32 f32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f64 f64_<64>;", indent).unwrap();
    }

    fn emit_register_declarations_func(&mut self, _func: &GpuFunction) {
        let indent = "\t".repeat(self.indent);

        writeln!(self.output, "{}.reg .pred p<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b16 r16_<64>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b32 r32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .b64 r64_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f32 f32_<128>;", indent).unwrap();
        writeln!(self.output, "{}.reg .f64 f64_<64>;", indent).unwrap();
        writeln!(self.output).unwrap();
    }

    fn emit_constant(&mut self, constant: &GpuConstant) {
        let ptx_type = self.gpu_type_to_ptx(&constant.ty);

        write!(self.output, ".const {} {} = ", ptx_type, constant.name).unwrap();
        self.emit_const_value(&constant.value);
        writeln!(self.output, ";").unwrap();
    }

    fn emit_const_value(&mut self, value: &GpuConstValue) {
        match value {
            GpuConstValue::Int(n) => write!(self.output, "{}", n).unwrap(),
            GpuConstValue::Float(n) => write!(self.output, "{:.15e}", n).unwrap(),
            GpuConstValue::Bool(b) => write!(self.output, "{}", if *b { 1 } else { 0 }).unwrap(),
            GpuConstValue::Array(elems) => {
                write!(self.output, "{{").unwrap();
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    self.emit_const_value(elem);
                }
                write!(self.output, "}}").unwrap();
            }
            GpuConstValue::Struct(fields) => {
                write!(self.output, "{{").unwrap();
                for (i, field) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(self.output, ", ").unwrap();
                    }
                    self.emit_const_value(field);
                }
                write!(self.output, "}}").unwrap();
            }
        }
    }

    fn alloc_register(&mut self, ty: &GpuType) -> String {
        match ty {
            GpuType::I16 | GpuType::U16 | GpuType::F16 => {
                let n = self.reg_counters.b16;
                self.reg_counters.b16 += 1;
                format!("r16_{}", n)
            }
            GpuType::I32 | GpuType::U32 => {
                let n = self.reg_counters.b32;
                self.reg_counters.b32 += 1;
                format!("r32_{}", n)
            }
            GpuType::I64 | GpuType::U64 | GpuType::Ptr(_, _) => {
                let n = self.reg_counters.b64;
                self.reg_counters.b64 += 1;
                format!("r64_{}", n)
            }
            GpuType::F32 => {
                let n = self.reg_counters.f32;
                self.reg_counters.f32 += 1;
                format!("f32_{}", n)
            }
            GpuType::F64 => {
                let n = self.reg_counters.f64;
                self.reg_counters.f64 += 1;
                format!("f64_{}", n)
            }
            _ => {
                let n = self.reg_counters.b64;
                self.reg_counters.b64 += 1;
                format!("r64_{}", n)
            }
        }
    }

    fn alloc_pred_register(&mut self) -> String {
        let n = self.reg_counters.pred;
        self.reg_counters.pred += 1;
        format!("p{}", n)
    }

    fn get_register(&self, id: ValueId) -> String {
        self.registers[id.0 as usize].clone()
    }

    fn get_value_type(&self, id: ValueId) -> GpuType {
        self.value_types
            .get(id.0 as usize)
            .cloned()
            .unwrap_or(GpuType::I64)
    }

    fn gpu_type_to_ptx(&self, ty: &GpuType) -> &'static str {
        match ty {
            GpuType::Void => ".b32",
            GpuType::Bool => ".pred",
            GpuType::I8 | GpuType::U8 => ".b8",
            GpuType::I16 | GpuType::U16 => ".b16",
            GpuType::I32 | GpuType::U32 => ".b32",
            GpuType::I64 | GpuType::U64 => ".b64",
            GpuType::F16 => ".f16",
            GpuType::F32 => ".f32",
            GpuType::F64 => ".f64",
            // Modern ML types (PTX 8.x+, Blackwell architecture)
            GpuType::BF16 => ".bf16", // BFloat16
            GpuType::F8E4M3 => ".b8", // FP8 E4M3 stored as byte
            GpuType::F8E5M2 => ".b8", // FP8 E5M2 stored as byte
            GpuType::F4 => ".b8",     // FP4 stored as byte (2 packed)
            GpuType::Ptr(_, _) => ".b64",
            _ => ".b64",
        }
    }

    fn type_suffix(&self, ty: &GpuType) -> &'static str {
        match ty {
            GpuType::I8 => "s8",
            GpuType::I16 => "s16",
            GpuType::I32 => "s32",
            GpuType::I64 => "s64",
            GpuType::U8 => "u8",
            GpuType::U16 => "u16",
            GpuType::U32 => "u32",
            GpuType::U64 => "u64",
            GpuType::F16 => "f16",
            GpuType::F32 => "f32",
            GpuType::F64 => "f64",
            // Modern ML types (PTX 8.x+, Blackwell architecture)
            GpuType::BF16 => "bf16",   // BFloat16
            GpuType::F8E4M3 => "e4m3", // FP8 E4M3 format
            GpuType::F8E5M2 => "e5m2", // FP8 E5M2 format
            GpuType::F4 => "b8",       // 4-bit packed (stored as byte)
            _ => "b64",
        }
    }

    fn memory_space_to_ptx(&self, space: MemorySpace) -> &'static str {
        match space {
            MemorySpace::Global => ".global",
            MemorySpace::Shared => ".shared",
            MemorySpace::Local => ".local",
            MemorySpace::Constant => ".const",
            MemorySpace::Generic => "",
            MemorySpace::Texture => ".tex",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ptx_header() {
        let mut codegen = PtxCodegen::new((7, 5));
        let module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".version 6.4")); // PTX 6.4 for Turing (sm_75)
        assert!(ptx.contains(".target sm_75"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_ptx_simple_kernel() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("add_one");
        kernel.add_param(GpuParam {
            name: "data".to_string(),
            ty: GpuType::Ptr(Box::new(GpuType::F32), MemorySpace::Global),
            space: MemorySpace::Global,
            restrict: true,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.add_instruction(ValueId(1), GpuOp::BlockIdX);
        block.add_instruction(ValueId(2), GpuOp::BlockDimX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".visible .entry add_one"));
        assert!(ptx.contains("%tid.x"));
        assert!(ptx.contains("%ctaid.x"));
        assert!(ptx.contains("%ntid.x"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_ptx_shared_memory() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("reduce");
        kernel.add_shared_memory(SharedMemDecl {
            name: "cache".to_string(),
            elem_type: GpuType::F32,
            size: 256,
            align: 4,
        });

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::SyncThreads);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".shared"));
        assert!(ptx.contains("cache"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn test_ptx_arithmetic() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("math");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(10, GpuType::I32));
        block.add_instruction(ValueId(1), GpuOp::ConstInt(20, GpuType::I32));
        block.add_instruction(ValueId(2), GpuOp::Add(ValueId(0), ValueId(1)));
        block.add_instruction(ValueId(3), GpuOp::ConstFloat(3.14, GpuType::F32));
        block.add_instruction(ValueId(4), GpuOp::ConstFloat(2.0, GpuType::F32));
        block.add_instruction(ValueId(5), GpuOp::FMul(ValueId(3), ValueId(4)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("add.s32"));
        assert!(ptx.contains("mul.f32"));
    }

    #[test]
    fn test_ptx_cooperative_groups_warp() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (8, 0), // sm_80 for redux.sync
            },
        );

        let mut kernel = GpuKernel::new("warp_reduce");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Get warp group handle
        block.add_instruction(ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Warp));
        // Get lane ID (thread rank in warp)
        block.add_instruction(ValueId(1), GpuOp::CoopThreadRank(ValueId(0)));
        // Check if leader
        block.add_instruction(ValueId(2), GpuOp::CoopIsLeader(ValueId(0)));
        // Synchronize warp
        block.add_instruction(ValueId(3), GpuOp::CoopSync(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("CoopThisGroup scope=Warp"));
        assert!(ptx.contains("CoopThreadRank"));
        assert!(ptx.contains("CoopIsLeader"));
        assert!(ptx.contains("CoopSync"));
    }

    #[test]
    fn test_ptx_cooperative_shuffle() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("warp_shuffle");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Get warp group
        block.add_instruction(ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Warp));
        // Some value to shuffle
        block.add_instruction(ValueId(1), GpuOp::ConstFloat(1.0, GpuType::F32));
        // Shuffle source lane
        block.add_instruction(ValueId(2), GpuOp::ConstInt(0, GpuType::U32));
        // Broadcast from lane 0
        block.add_instruction(
            ValueId(3),
            GpuOp::CoopShfl(ValueId(0), ValueId(1), ValueId(2)),
        );
        // Delta for shuffle down
        block.add_instruction(ValueId(4), GpuOp::ConstInt(16, GpuType::U32));
        // Shuffle down by 16
        block.add_instruction(
            ValueId(5),
            GpuOp::CoopShflDown(ValueId(0), ValueId(1), ValueId(4)),
        );
        // Shuffle XOR with mask
        block.add_instruction(
            ValueId(6),
            GpuOp::CoopShflXor(ValueId(0), ValueId(1), ValueId(4)),
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("shfl.sync.idx.b32"));
        assert!(ptx.contains("shfl.sync.down.b32"));
        assert!(ptx.contains("shfl.sync.bfly.b32"));
    }

    #[test]
    fn test_ptx_cooperative_reduce() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (8, 0), // sm_80 for redux.sync
            },
        );

        let mut kernel = GpuKernel::new("warp_collective");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Get warp group
        block.add_instruction(ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Warp));
        // Value to reduce
        block.add_instruction(ValueId(1), GpuOp::ConstFloat(1.0, GpuType::F32));
        // Reduce with add
        block.add_instruction(
            ValueId(2),
            GpuOp::CoopReduce(ValueId(0), ValueId(1), CoopReduceOp::Add),
        );
        // Reduce with max
        block.add_instruction(
            ValueId(3),
            GpuOp::CoopReduce(ValueId(0), ValueId(1), CoopReduceOp::Max),
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("redux.sync.add"));
        assert!(ptx.contains("redux.sync.max"));
    }

    #[test]
    fn test_ptx_cooperative_vote() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("warp_vote");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Get warp group
        block.add_instruction(ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Warp));
        // Predicate
        block.add_instruction(ValueId(1), GpuOp::ConstBool(true));
        // Ballot
        block.add_instruction(ValueId(2), GpuOp::CoopBallot(ValueId(0), ValueId(1)));
        // All
        block.add_instruction(ValueId(3), GpuOp::CoopAll(ValueId(0), ValueId(1)));
        // Any
        block.add_instruction(ValueId(4), GpuOp::CoopAny(ValueId(0), ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("vote.sync.ballot.b32"));
        assert!(ptx.contains("vote.sync.all.pred"));
        assert!(ptx.contains("vote.sync.any.pred"));
    }

    #[test]
    fn test_ptx_cooperative_partition() {
        let mut module = GpuModule::new(
            "test",
            GpuTarget::Cuda {
                compute_capability: (7, 5),
            },
        );

        let mut kernel = GpuKernel::new("partition_test");

        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Get warp group
        block.add_instruction(ValueId(0), GpuOp::CoopThisGroup(CooperativeScope::Warp));
        // Partition into 16-thread tiles
        block.add_instruction(ValueId(1), GpuOp::CoopPartitionTiled(ValueId(0), 16));
        // Get coalesced threads
        block.add_instruction(ValueId(2), GpuOp::CoopCoalescedThreads);
        // Elect a leader
        block.add_instruction(ValueId(3), GpuOp::CoopElect(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);

        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("CoopPartitionTiled size=16"));
        assert!(ptx.contains("activemask.b32"));
        assert!(ptx.contains("CoopElect"));
    }

    // =========================================================================
    // Debug/Profiling Operation Tests
    // =========================================================================

    #[test]
    fn test_debug_trap() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("trap_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::Trap);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("trap;"));
    }

    #[test]
    fn test_debug_brkpt() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("brkpt_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::Brkpt);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("brkpt;"));
    }

    #[test]
    fn test_debug_clock() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("clock_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::Clock);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("%clock64"));
    }

    #[test]
    fn test_debug_globaltimer() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("timer_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::GlobalTimer);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("%globaltimer"));
    }

    #[test]
    fn test_debug_assert() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("assert_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstBool(true));
        block.add_instruction(ValueId(1), GpuOp::Assert(ValueId(0), Some(42)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("gpu.assert"));
        assert!(ptx.contains("setp.eq"));
        assert!(ptx.contains("trap"));
    }

    #[test]
    fn test_debug_printf() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("printf_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create some values to print
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.add_instruction(ValueId(1), GpuOp::ConstFloat(3.14, GpuType::F32));
        // Printf with format string id 0 and two arguments
        block.add_instruction(ValueId(2), GpuOp::Printf(0, vec![ValueId(0), ValueId(1)]));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("gpu.printf"));
        assert!(ptx.contains("vprintf"));
        assert!(ptx.contains("__printf_args"));
    }

    #[test]
    fn test_debug_pmevent() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("debug_test", target);

        let mut kernel = GpuKernel::new("pmevent_test");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::PmEvent(123));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("pmevent 123"));
    }

    // =========================================================================
    // BF16/FP8/F4 Conversion Tests
    // =========================================================================

    #[test]
    fn test_f32_to_bf16_conversion() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 0), // sm_80+ for BF16
        };
        let mut module = GpuModule::new("bf16_test", target);

        let mut kernel = GpuKernel::new("bf16_convert");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create an f32 value and convert to bf16
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(3.14, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToBF16(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("cvt.rn.bf16.f32"));
    }

    #[test]
    fn test_bf16_to_f32_conversion() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 0),
        };
        let mut module = GpuModule::new("bf16_test", target);

        let mut kernel = GpuKernel::new("bf16_to_f32");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create a bf16 value (as const) and convert to f32
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(3.14, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToBF16(ValueId(0)));
        block.add_instruction(ValueId(2), GpuOp::BF16ToF32(ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("cvt.f32.bf16"));
    }

    #[test]
    fn test_f32_to_f8e4m3_conversion() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9), // sm_89+ for FP8
        };
        let mut module = GpuModule::new("fp8_test", target);

        let mut kernel = GpuKernel::new("fp8_convert");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.5, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToF8E4M3(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("e4m3"));
    }

    #[test]
    fn test_f8e4m3_to_f32_conversion() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("fp8_test", target);

        let mut kernel = GpuKernel::new("fp8_to_f32");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(2.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToF8E4M3(ValueId(0)));
        block.add_instruction(ValueId(2), GpuOp::F8E4M3ToF32(ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("cvt.f32.e4m3"));
    }

    #[test]
    fn test_f32_to_f8e5m2_conversion() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("fp8_test", target);

        let mut kernel = GpuKernel::new("fp8_e5m2_convert");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(100.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToF8E5M2(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("e5m2"));
    }

    #[test]
    fn test_f4_quantization() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("f4_test", target);

        let mut kernel = GpuKernel::new("f4_quant");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToF4(ValueId(0)));
        block.add_instruction(ValueId(2), GpuOp::F4ToF32(ValueId(1)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("F4 quantization"));
        assert!(ptx.contains("F4 dequantization"));
    }

    #[test]
    fn test_pack_f8x2() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("pack_test", target);

        let mut kernel = GpuKernel::new("pack_f8");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create two f32 values and convert to fp8, then pack
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::ConstFloat(2.0, GpuType::F32));
        block.add_instruction(ValueId(2), GpuOp::F32ToF8E4M3(ValueId(0)));
        block.add_instruction(ValueId(3), GpuOp::F32ToF8E4M3(ValueId(1)));
        block.add_instruction(ValueId(4), GpuOp::PackF8x2(ValueId(2), ValueId(3)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("shl.b16"));
        assert!(ptx.contains("or.b16"));
    }

    #[test]
    fn test_unpack_f8x2() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("unpack_test", target);

        let mut kernel = GpuKernel::new("unpack_f8");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create packed value and unpack
        block.add_instruction(ValueId(0), GpuOp::ConstInt(0x1234, GpuType::U16));
        block.add_instruction(ValueId(1), GpuOp::UnpackF8x2Low(ValueId(0)));
        block.add_instruction(ValueId(2), GpuOp::UnpackF8x2High(ValueId(0)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("and.b16"));
        assert!(ptx.contains("0x00FF"));
    }

    #[test]
    fn test_pack_f4x2() {
        let target = GpuTarget::Cuda {
            compute_capability: (7, 5),
        };
        let mut module = GpuModule::new("pack_f4_test", target);

        let mut kernel = GpuKernel::new("pack_f4");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.0, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::ConstFloat(2.0, GpuType::F32));
        block.add_instruction(ValueId(2), GpuOp::F32ToF4(ValueId(0)));
        block.add_instruction(ValueId(3), GpuOp::F32ToF4(ValueId(1)));
        block.add_instruction(ValueId(4), GpuOp::PackF4x2(ValueId(2), ValueId(3)));
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((7, 5));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("0x0F"));
        assert!(ptx.contains("or.b32"));
    }

    #[test]
    fn test_quantize_with_mode() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("quant_mode_test", target);

        let mut kernel = GpuKernel::new("quantize_rne");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.5, GpuType::F32));
        block.add_instruction(
            ValueId(1),
            GpuOp::QuantizeF32ToF8(ValueId(0), QuantizeMode::RoundNearestEven),
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("cvt.rn.satfinite"));
    }

    #[test]
    fn test_dequantize_with_scale() {
        let target = GpuTarget::Cuda {
            compute_capability: (8, 9),
        };
        let mut module = GpuModule::new("dequant_test", target);

        let mut kernel = GpuKernel::new("dequantize");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Create quantized value
        block.add_instruction(ValueId(0), GpuOp::ConstFloat(1.5, GpuType::F32));
        block.add_instruction(ValueId(1), GpuOp::F32ToF8E4M3(ValueId(0)));
        // Scale factor
        block.add_instruction(ValueId(2), GpuOp::ConstFloat(0.5, GpuType::F32));
        // Dequantize with scale
        block.add_instruction(
            ValueId(3),
            GpuOp::DequantizeF8ToF32(ValueId(1), Some(ValueId(2))),
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((8, 9));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("cvt.f32.e4m3"));
        assert!(ptx.contains("mul.f32"));
    }

    // =========================================================================
    // Blackwell (sm_100) Tests
    // =========================================================================

    #[test]
    fn test_blackwell_ptx_version() {
        // Verify correct PTX version selection for Blackwell
        let ptx_ver = PtxCodegen::recommended_ptx_version((10, 0));
        assert_eq!(ptx_ver, (8, 5));

        let ptx_ver_ultra = PtxCodegen::recommended_ptx_version((12, 0));
        assert_eq!(ptx_ver_ultra, (8, 6));
    }

    #[test]
    fn test_blackwell_target_header() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("blackwell_test", target);

        let mut kernel = GpuKernel::new("blackwell_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ThreadIdX);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains(".target sm_100"));
        assert!(ptx.contains(".version 8.5"));
    }

    #[test]
    fn test_blackwell_cluster_ops() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("cluster_test", target);

        let mut kernel = GpuKernel::new("cluster_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ClusterId);
        block.add_instruction(ValueId(1), GpuOp::ClusterDim);
        block.add_instruction(ValueId(2), GpuOp::BlockIdInCluster);
        block.add_instruction(ValueId(3), GpuOp::ClusterBarrier);
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("%clusterid"));
        assert!(ptx.contains("%nclusterid"));
        assert!(ptx.contains("%cluster_ctaid"));
        assert!(ptx.contains("barrier.cluster.sync.aligned"));
    }

    #[test]
    fn test_blackwell_wgmma_bf16() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("wgmma_test", target);

        let mut kernel = GpuKernel::new("wgmma_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        // Set up matrix operands
        block.add_instruction(ValueId(0), GpuOp::ConstInt(0, GpuType::U64)); // A ptr
        block.add_instruction(ValueId(1), GpuOp::ConstInt(0, GpuType::U64)); // B ptr
        block.add_instruction(ValueId(2), GpuOp::ConstFloat(0.0, GpuType::F32)); // C accumulator
        block.add_instruction(
            ValueId(3),
            GpuOp::WgmmaBf16 {
                a: ValueId(0),
                b: ValueId(1),
                c: ValueId(2),
                m: 64,
                n: 128,
                k: 16,
            },
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("WGMMA BF16"));
        assert!(ptx.contains("wgmma.mma_async"));
        assert!(ptx.contains("bf16.bf16"));
    }

    #[test]
    fn test_blackwell_wgmma_fp4() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("wgmma_fp4_test", target);

        let mut kernel = GpuKernel::new("wgmma_fp4_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(0, GpuType::U64)); // A
        block.add_instruction(ValueId(1), GpuOp::ConstInt(0, GpuType::U64)); // B
        block.add_instruction(ValueId(2), GpuOp::ConstFloat(0.0, GpuType::F32)); // C
        block.add_instruction(ValueId(3), GpuOp::ConstFloat(1.0, GpuType::F32)); // scale_a
        block.add_instruction(ValueId(4), GpuOp::ConstFloat(1.0, GpuType::F32)); // scale_b
        block.add_instruction(
            ValueId(5),
            GpuOp::WgmmaFp4 {
                a: ValueId(0),
                b: ValueId(1),
                c: ValueId(2),
                m: 64,
                n: 128,
                k: 32,
                scale_a: ValueId(3),
                scale_b: ValueId(4),
            },
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("WGMMA FP4"));
        assert!(ptx.contains("5th-gen Tensor Cores"));
        assert!(ptx.contains("e2m1"));
    }

    #[test]
    fn test_blackwell_tma_operations() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("tma_test", target);

        let mut kernel = GpuKernel::new("tma_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(0, GpuType::U64)); // shared ptr
        block.add_instruction(ValueId(1), GpuOp::ConstInt(0, GpuType::U64)); // global ptr
        block.add_instruction(ValueId(2), GpuOp::ConstInt(0, GpuType::U64)); // barrier
        block.add_instruction(
            ValueId(3),
            GpuOp::TmaLoadAsync {
                dst_shared: ValueId(0),
                src_global: ValueId(1),
                size: 4096,
                barrier: ValueId(2),
            },
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("TMA async load"));
        assert!(ptx.contains("cp.async.bulk.tensor"));
    }

    #[test]
    fn test_blackwell_decompression() {
        let target = GpuTarget::Cuda {
            compute_capability: (10, 0),
        };
        let mut module = GpuModule::new("decompress_test", target);

        let mut kernel = GpuKernel::new("decompress_kernel");
        let mut block = GpuBlock::new(BlockId(0), "entry");
        block.add_instruction(ValueId(0), GpuOp::ConstInt(0, GpuType::U64)); // dst
        block.add_instruction(ValueId(1), GpuOp::ConstInt(0, GpuType::U64)); // src
        block.add_instruction(
            ValueId(2),
            GpuOp::DecompressLz4 {
                dst: ValueId(0),
                src: ValueId(1),
                compressed_size: 1024,
                uncompressed_size: 4096,
            },
        );
        block.set_terminator(GpuTerminator::ReturnVoid);
        kernel.add_block(block);
        module.add_kernel(kernel);

        let mut codegen = PtxCodegen::new((10, 0));
        let ptx = codegen.generate(&module);

        assert!(ptx.contains("Hardware LZ4 decompression"));
        assert!(ptx.contains("decompress.lz4"));
    }

    #[test]
    fn test_cuda_arch_features() {
        // CudaArch and CudaFeatures imported via super::* from ir module

        // Test Blackwell features
        let blackwell = CudaArch::Blackwell;
        assert_eq!(blackwell.compute_capability(), (10, 0));
        assert_eq!(blackwell.tensor_core_gen(), 5);
        assert_eq!(blackwell.name(), "Blackwell");

        // Test feature detection
        let features = CudaFeatures::from_compute_capability((10, 0));
        assert!(features.bf16);
        assert!(features.fp8);
        assert!(features.tma);
        assert!(features.clusters);
        assert!(features.tensor_fp4);
        assert!(features.tensor_core_gen5);
        assert!(features.nvlink5);
        assert!(features.is_blackwell());

        // Test Hopper doesn't have Blackwell features
        let hopper_features = CudaFeatures::from_compute_capability((9, 0));
        assert!(hopper_features.tma);
        assert!(!hopper_features.tensor_fp4);
        assert!(!hopper_features.tensor_core_gen5);
        assert!(hopper_features.is_hopper());
    }

    #[test]
    fn test_supports_feature() {
        let codegen = PtxCodegen::new((10, 0));
        assert!(codegen.supports_feature("bf16"));
        assert!(codegen.supports_feature("fp8"));
        assert!(codegen.supports_feature("tma"));
        assert!(codegen.supports_feature("fp4"));
        assert!(codegen.supports_feature("tensor_gen5"));
        assert!(codegen.supports_feature("decompression"));
        assert!(codegen.supports_feature("nvlink5"));

        let turing = PtxCodegen::new((7, 5));
        assert!(!turing.supports_feature("bf16"));
        assert!(!turing.supports_feature("fp8"));
        assert!(!turing.supports_feature("tma"));
    }
}
