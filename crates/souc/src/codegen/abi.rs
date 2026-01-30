//! ABI (Application Binary Interface) handling for MIR
//!
//! This module provides calling convention handling and ABI support
//! based on modern x86-64 and ARM64 calling conventions.

use crate::mir::{MirFunction, MirType, ValueId};
use cranelift_codegen::ir::{AbiParam, CallConv, types};

/// Calling convention support for different architectures
pub struct CallingConvention {
    /// Architecture-specific register allocation
    pub register_classes: Vec<RegisterClass>,
    /// Stack alignment requirements
    pub stack_alignment: usize,
    /// Size of general purpose registers
    pub gp_register_size: usize,
    /// Size of floating point registers
    pub fp_register_size: usize,
}

/// Register class for ABI parameter classification
#[derive(Debug, Clone)]
pub struct RegisterClass {
    /// Class name (e.g., "integer", "float")
    pub name: String,
    /// Registers in this class
    pub registers: Vec<Register>,
    /// Maximum parameters that can be passed in registers
    pub max_params: usize,
}

/// Register representation
#[derive(Debug, Clone)]
pub struct Register {
    /// Register name (e.g., "rax", "xmm0")
    pub name: String,
    /// Register encoding
    pub encoding: u8,
    /// Is this a general purpose or floating point register
    pub is_gp: bool,
}

impl CallingConvention {
    /// Create x86-64 System V ABI calling convention
    pub fn x86_64_sysv() -> Self {
        let integer_regs = vec![
            Register {
                name: "rdi".to_string(),
                encoding: 5,
                is_gp: true,
            },
            Register {
                name: "rsi".to_string(),
                encoding: 4,
                is_gp: true,
            },
            Register {
                name: "rdx".to_string(),
                encoding: 3,
                is_gp: true,
            },
            Register {
                name: "rcx".to_string(),
                encoding: 2,
                is_ggp: true,
            },
            Register {
                name: "r8".to_string(),
                encoding: 0,
                is_gp: true,
            },
            Register {
                name: "r9".to_string(),
                encoding: 1,
                is_gp: true,
            },
            Register {
                name: "rax".to_string(),
                encoding: 0,
                is_gp: true,
            },
            Register {
                name: "rbx".to_string(),
                encoding: 3,
                is_gp: true,
            },
            Register {
                name: "r12".to_string(),
                encoding: 5,
                is_gp: true,
            },
            Register {
                name: "r13".to_string(),
                encoding: 6,
                is_gp: true,
            },
            Register {
                name: "r14".to_string(),
                encoding: 7,
                is_gp: true,
            },
            Register {
                name: "r15".to_string(),
                encoding: 8,
                is_gp: true,
            },
        ];

        let float_regs = vec![
            Register {
                name: "xmm0".to_string(),
                encoding: 0,
                is_gp: false,
            },
            Register {
                name: "xmm1".to_string(),
                encoding: 1,
                is_gp: false,
            },
            Register {
                name: "xmm2".to_string(),
                encoding: 2,
                is_gp: false,
            },
            Register {
                name: "xmm3".to_string(),
                encoding: 3,
                is_gp: false,
            },
            Register {
                name: "xmm4".to_string(),
                encoding: 4,
                is_gp: false,
            },
            Register {
                name: "xmm5".to_string(),
                encoding: 5,
                is_gp: false,
            },
            Register {
                name: "xmm6".to_string(),
                encoding: 6,
                is_gp: false,
            },
            Register {
                name: "xmm7".to_string(),
                encoding: 7,
                is_gp: false,
            },
        ];

        Self {
            register_classes: vec![
                RegisterClass {
                    name: "integer".to_string(),
                    registers: integer_regs,
                    max_params: 6,
                },
                RegisterClass {
                    name: "float".to_string(),
                    registers: float_regs,
                    max_params: 8,
                },
            ],
            stack_alignment: 16,
            gp_register_size: 8,
            fp_register_size: 16,
        }
    }

    /// Create ARM64 AAPCS64 calling convention
    pub fn arm64_aapcs64() -> Self {
        let integer_regs = vec![
            Register {
                name: "x0".to_string(),
                encoding: 0,
                is_gp: true,
            },
            Register {
                name: "x1".to_string(),
                encoding: 1,
                is_gp: true,
            },
            Register {
                name: "x2".to_string(),
                encoding: 2,
                is_gp: true,
            },
            Register {
                name: "x3".to_string(),
                encoding: 3,
                is_gp: true,
            },
            Register {
                name: "x4".to_string(),
                encoding: 4,
                is_gp: true,
            },
            Register {
                name: "x5".to_string(),
                encoding: 5,
                is_gp: true,
            },
            Register {
                name: "x6".to_string(),
                encoding: 6,
                is_gp: true,
            },
            Register {
                name: "x7".to_string(),
                encoding: 7,
                is_gp: true,
            },
            Register {
                name: "x8".to_string(),
                encoding: 8,
                is_gp: true,
            },
            Register {
                name: "x9".to_string(),
                encoding: 9,
                is_gp: true,
            },
            Register {
                name: "x10".to_string(),
                encoding: 10,
                is_gp: true,
            },
            Register {
                name: "x11".to_string(),
                encoding: 11,
                is_gp: true,
            },
            Register {
                name: "x12".to_string(),
                encoding: 12,
                is_gp: true,
            },
            Register {
                name: "x13".to_string(),
                encoding: 13,
                is_gp: true,
            },
            Register {
                name: "x14".to_string(),
                encoding: 14,
                is_gp: true,
            },
            Register {
                name: "x15".to_string(),
                encoding: 15,
                is_gp: true,
            },
            Register {
                name: "x16".to_string(),
                encoding: 16,
                is_gp: true,
            },
            Register {
                name: "x17".to_string(),
                encoding: 17,
                is_gp: true,
            },
            Register {
                name: "x18".to_string(),
                encoding: 18,
                is_gp: true,
            },
            Register {
                name: "x19".to_string(),
                encoding: 19,
                is_gp: true,
            },
            Register {
                name: "x20".to_string(),
                encoding: 20,
                is_gp: true,
            },
            Register {
                name: "x21".to_string(),
                encoding: 21,
                is_gp: true,
            },
            Register {
                name: "x22".to_string(),
                encoding: 22,
                is_gp: true,
            },
            Register {
                name: "x23".to_string(),
                encoding: 23,
                is_gp: true,
            },
            Register {
                name: "x24".to_string(),
                encoding: 24,
                is_gp: true,
            },
            Register {
                name: "x25".to_string(),
                encoding: 25,
                is_gp: true,
            },
            Register {
                name: "x26".to_string(),
                encoding: 26,
                is_gp: true,
            },
            Register {
                name: "x27".to_string(),
                encoding: 27,
                is_gp: true,
            },
            Register {
                name: "x28".to_string(),
                encoding: 28,
                is_gp: true,
            },
            Register {
                name: "x29".to_string(),
                encoding: 29,
                is_gp: true,
            },
        ];

        let float_regs = vec![
            Register {
                name: "v0".to_string(),
                encoding: 0,
                is_gp: false,
            },
            Register {
                name: "v1".to_string(),
                encoding: 1,
                is_gp: false,
            },
            Register {
                name: "v2".to_string(),
                encoding: 2,
                is_gp: false,
            },
            Register {
                name: "v3".to_string(),
                encoding: 3,
                is_gp: false,
            },
            Register {
                name: "v4".to_string(),
                encoding: 4,
                is_gp: false,
            },
            Register {
                name: "v5".to_string(),
                encoding: 5,
                is_gp: false,
            },
            Register {
                name: "v6".to_string(),
                encoding: 6,
                is_gp: false,
            },
            Register {
                name: "v7".to_string(),
                encoding: 7,
                is_gp: false,
            },
        ];

        Self {
            register_classes: vec![
                RegisterClass {
                    name: "integer".to_string(),
                    registers: integer_regs,
                    max_params: 8,
                },
                RegisterClass {
                    name: "float".to_string(),
                    registers: float_regs,
                    max_params: 8,
                },
            ],
            stack_alignment: 16,
            gp_register_size: 8,
            fp_register_size: 16,
        }
    }

    /// Classify a MIR type for ABI parameter passing
    pub fn classify_type(&self, ty: &MirType) -> AbiParamClass {
        match ty {
            MirType::I8
            | MirType::U8
            | MirType::I16
            | MirType::U16
            | MirType::I32
            | MirType::U32
            | MirType::I64
            | MirType::U64
            | MirType::Isize
            | MirType::Usize
            | MirType::Ptr(_)
            | MirType::Bool
            | MirType::Char => AbiParamClass::Integer,
            MirType::F32 | MirType::F64 => AbiParamClass::Float,
            MirType::Struct { fields, .. } => {
                // Classify struct based on its fields
                self.classify_struct(fields)
            }
            MirType::Array(_, _) | MirType::Tuple(_) => {
                // Arrays and tuples are passed by reference or as aggregates
                AbiParamClass::Memory
            }
            _ => AbiParamClass::Integer, // Default to integer for unknown types
        }
    }

    /// Classify struct parameters
    fn classify_struct(&self, fields: &[(String, MirType)]) -> AbiParamClass {
        let mut has_float = false;
        let mut has_integer = false;
        let mut total_size = 0;

        for (_, field_ty) in fields {
            match field_ty {
                MirType::F32 | MirType::F64 => has_float = true,
                MirType::I8
                | MirType::U8
                | MirType::I16
                | MirType::U16
                | MirType::I32
                | MirType::U32
                | MirType::I64
                | MirType::U64
                | MirType::Isize
                | MirType::Usize
                | MirType::Ptr(_)
                | MirType::Bool
                | MirType::Char => {
                    has_integer = true;
                }
                _ => {}
            }
            total_size += field_ty.size_bytes().unwrap_or(8);
        }

        // Simple classification: if all fields are same class, use that class
        if has_float && !has_integer {
            AbiParamClass::Float
        } else if has_integer && !has_float {
            AbiParamClass::Integer
        } else if has_float && has_integer {
            // Mixed types - classify as memory
            AbiParamClass::Memory
        } else {
            // Empty or unknown struct
            AbiParamClass::Integer
        }
    }

    /// Generate Cranelift ABI parameters from MIR function signature
    pub fn generate_abi_params(&self, func: &MirFunction) -> Vec<AbiParam> {
        let mut params = Vec::new();
        let mut int_reg_count = 0;
        let mut float_reg_count = 0;

        for (param_name, param_ty) in &func.params {
            let abi_param = self.classify_type(param_ty);

            match abi_param {
                AbiParamClass::Integer => {
                    if int_reg_count < self.get_integer_register_class().max_params {
                        // Pass in register
                        params.push(AbiParam::new(types::I64)); // All integers are I64 in Cranelift
                        int_reg_count += 1;
                    } else {
                        // Pass on stack
                        params.push(AbiParam::new(types::I64).stack_only());
                    }
                }
                AbiParamClass::Float => {
                    if float_reg_count < self.get_float_register_class().max_params {
                        // Pass in register
                        let cranelift_type = match param_ty {
                            MirType::F32 => types::F32,
                            MirType::F64 => types::F64,
                            _ => types::F64,
                        };
                        params.push(AbiParam::new(cranelift_type));
                        float_reg_count += 1;
                    } else {
                        // Pass on stack
                        let cranelift_type = match param_ty {
                            MirType::F32 => types::F32,
                            MirType::F64 => types::F64,
                            _ => types::F64,
                        };
                        params.push(AbiParam::new(cranelift_type).stack_only());
                    }
                }
                AbiParamClass::Memory => {
                    // Pass by reference
                    params.push(AbiParam::new(types::I64));
                }
            }
        }

        params
    }

    /// Generate return type ABI parameters
    pub fn generate_return_params(&self, func: &MirFunction) -> Vec<AbiParam> {
        let mut returns = Vec::new();

        match &func.return_type {
            MirType::Unit | MirType::Void => {
                // No return parameters for void/unit
            }
            MirType::F32 | MirType::F64 => {
                // Floating point return in register
                let cranelift_type = match &func.return_type {
                    MirType::F32 => types::F32,
                    MirType::F64 => types::F64,
                    _ => types::F64,
                };
                returns.push(AbiParam::new(cranelift_type));
            }
            MirType::Struct { fields, .. } if fields.len() <= 2 => {
                // Small structs can be returned in registers
                for (_, field_ty) in fields {
                    let cranelift_type = self.mir_type_to_cranelift_type(field_ty);
                    returns.push(AbiParam::new(cranelift_type));
                }
            }
            _ => {
                // All other types returned in single register (pointer to result)
                returns.push(AbiParam::new(types::I64));
            }
        }

        returns
    }

    /// Convert MIR type to Cranelift type
    fn mir_type_to_cranelift_type(&self, ty: &MirType) -> types::Type {
        match ty {
            MirType::I8 | MirType::U8 => types::I8,
            MirType::I16 | MirType::U16 => types::I16,
            MirType::I32 | MirType::U32 => types::I32,
            MirType::I64 | MirType::U64 | MirType::Isize | MirType::Usize => types::I64,
            MirType::F32 => types::F32,
            MirType::F64 => types::F64,
            MirType::Bool => types::I8,
            MirType::Char => types::I32,
            _ => types::I64, // Default to I64 for unknown types
        }
    }

    /// Get integer register class
    fn get_integer_register_class(&self) -> &RegisterClass {
        self.register_classes
            .iter()
            .find(|class| class.name == "integer")
            .expect("Integer register class not found")
    }

    /// Get float register class
    fn get_float_register_class(&self) -> &RegisterClass {
        self.register_classes
            .iter()
            .find(|class| class.name == "float")
            .expect("Float register class not found")
    }
}

/// ABI parameter classification
#[derive(Debug, Clone, PartialEq)]
pub enum AbiParamClass {
    /// Integer class (general purpose registers)
    Integer,
    /// Float class (floating point registers)
    Float,
    /// Memory class (passed by reference or on stack)
    Memory,
}

/// Stack slot optimization
pub struct StackSlotOptimizer {
    /// Maximum stack slot size before using heap allocation
    max_stack_size: usize,
    /// Stack alignment requirements
    stack_alignment: usize,
}

impl StackSlotOptimizer {
    pub fn new() -> Self {
        Self {
            max_stack_size: 4096, // 4KB max stack allocation
            stack_alignment: 16,
        }
    }

    /// Optimize stack slot allocation for a function
    pub fn optimize_stack_slots(&self, func: &MirFunction) -> StackSlotPlan {
        let mut stack_slots = Vec::new();
        let mut current_offset = 0;

        // Analyze all stack allocations in the function
        for block in &func.blocks {
            for instruction in &block.instructions {
                if let crate::mir::MirInstruction::Alloca { result, ty } = instruction {
                    let size = ty.size_bytes().unwrap_or(8);
                    let alignment = self.get_alignment_for_type(ty);

                    // Align current offset
                    current_offset = self.align_offset(current_offset, alignment);

                    // Check if allocation should be on stack or heap
                    if current_offset + size <= self.max_stack_size {
                        stack_slots.push(StackSlot {
                            value_id: *result,
                            size,
                            alignment,
                            offset: current_offset,
                            is_heap: false,
                        });
                        current_offset += size;
                    } else {
                        // Use heap allocation for large objects
                        stack_slots.push(StackSlot {
                            value_id: *result,
                            size,
                            alignment,
                            offset: 0,
                            is_heap: true,
                        });
                    }
                }
            }
        }

        StackSlotPlan {
            slots: stack_slots,
            total_stack_size: current_offset,
            max_stack_size: self.max_stack_size,
        }
    }

    /// Get alignment requirement for a type
    fn get_alignment_for_type(&self, ty: &MirType) -> usize {
        match ty {
            MirType::I8 | MirType::U8 => 1,
            MirType::I16 | MirType::U16 => 2,
            MirType::I32 | MirType::U32 | MirType::F32 => 4,
            MirType::I64
            | MirType::U64
            | MirType::Isize
            | MirType::Usize
            | MirType::F64
            | MirType::Ptr(_) => 8,
            MirType::Struct { .. } => 16, // Default to 16-byte alignment for structs
            _ => 8,
        }
    }

    /// Align an offset to the required boundary
    fn align_offset(&self, offset: usize, alignment: usize) -> usize {
        (offset + alignment - 1) & !(alignment - 1)
    }
}

/// Stack slot allocation plan
#[derive(Debug, Clone)]
pub struct StackSlotPlan {
    pub slots: Vec<StackSlot>,
    pub total_stack_size: usize,
    pub max_stack_size: usize,
}

/// Individual stack slot
#[derive(Debug, Clone)]
pub struct StackSlot {
    pub value_id: ValueId,
    pub size: usize,
    pub alignment: usize,
    pub offset: usize,
    pub is_heap: bool,
}

/// Register allocation preparation
pub struct RegisterAllocator {
    /// Available registers for allocation
    available_registers: Vec<Register>,
    /// Reserved registers (callee-saved, etc.)
    reserved_registers: Vec<String>,
    /// Number of virtual registers allocated
    next_virtual_reg: u32,
}

impl RegisterAllocator {
    pub fn new(abi: &CallingConvention) -> Self {
        let mut available_registers = Vec::new();

        // Add all available registers from ABI
        for class in &abi.register_classes {
            available_registers.extend(class.registers.clone());
        }

        // Mark callee-saved registers as reserved
        let reserved_registers = vec![
            "rbx".to_string(),
            "r12".to_string(),
            "r13".to_string(),
            "r14".to_string(),
            "r15".to_string(), // x86-64 callee-saved
        ];

        Self {
            available_registers,
            reserved_registers,
            next_virtual_reg: 0,
        }
    }

    /// Allocate a virtual register
    pub fn allocate_virtual_reg(&mut self) -> VirtualRegister {
        let reg = VirtualRegister {
            id: self.next_virtual_reg,
            class: RegisterClass::General,
            is_virtual: true,
        };
        self.next_virtual_reg += 1;
        reg
    }

    /// Allocate a physical register
    pub fn allocate_physical_reg(&mut self, class: RegisterClass) -> Option<Register> {
        // Filter out reserved registers
        for reg in &self.available_registers {
            if !self.reserved_registers.contains(&reg.name) {
                if (class == RegisterClass::General && reg.is_gp)
                    || (class == RegisterClass::Float && !reg.is_gp)
                {
                    return Some(reg.clone());
                }
            }
        }
        None
    }
}

/// Virtual register representation
#[derive(Debug, Clone)]
pub struct VirtualRegister {
    pub id: u32,
    pub class: RegisterClass,
    pub is_virtual: bool,
}

/// Register classes
#[derive(Debug, Clone, PartialEq)]
pub enum RegisterClass {
    General,
    Float,
    Vector,
}

/// Profiling hooks for performance analysis
pub struct ProfilingHooks {
    /// Enable function profiling
    pub function_profiling: bool,
    /// Enable block profiling
    pub block_profiling: bool,
    /// Enable instruction profiling
    pub instruction_profiling: bool,
}

impl ProfilingHooks {
    pub fn new() -> Self {
        Self {
            function_profiling: false,
            block_profiling: false,
            instruction_profiling: false,
        }
    }

    /// Enable comprehensive profiling
    pub fn enable_full_profiling(&mut self) {
        self.function_profiling = true;
        self.block_profiling = true;
        self.instruction_profiling = true;
    }

    /// Generate profiling instrumentation for a function
    pub fn generate_function_entry_hook(&self, func_name: &str) -> Vec<String> {
        if !self.function_profiling {
            return Vec::new();
        }

        vec![
            format!("// Function entry profiling for {}", func_name),
            "push %rax".to_string(),
            "push %rcx".to_string(),
            "push %rdx".to_string(),
            format!("lea .LPROF_{}(%rip), %rax", func_name),
            "call __llvm_profile_instrument_function".to_string(),
            "pop %rdx".to_string(),
            "pop %rcx".to_string(),
            "pop %rax".to_string(),
        ]
    }

    /// Generate block profiling instrumentation
    pub fn generate_block_profiling(&self, block_id: u32) -> Vec<String> {
        if !self.block_profiling {
            return Vec::new();
        }

        vec![
            format!("// Block {} profiling", block_id),
            "push %rax".to_string(),
            "mov ${}, %rdi".to_string(),
            "call __llvm_profile_instrument_branch".to_string(),
            "pop %rax".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x86_64_abi_classification() {
        let abi = CallingConvention::x86_64_sysv();

        assert_eq!(abi.classify_type(&MirType::I32), AbiParamClass::Integer);
        assert_eq!(abi.classify_type(&MirType::F64), AbiParamClass::Float);
        assert_eq!(abi.classify_type(&MirType::Bool), AbiParamClass::Integer);
    }

    #[test]
    fn test_stack_slot_optimization() {
        let optimizer = StackSlotOptimizer::new();
        let func = crate::mir::MirFunction::new(
            crate::mir::FuncId(0),
            "test_func".to_string(),
            MirType::Unit,
        );

        let plan = optimizer.optimize_stack_slots(&func);
        assert!(plan.total_stack_size <= optimizer.max_stack_size);
    }

    #[test]
    fn test_register_allocation() {
        let abi = CallingConvention::x86_64_sysv();
        let mut allocator = RegisterAllocator::new(&abi);

        let virtual_reg = allocator.allocate_virtual_reg();
        assert!(virtual_reg.is_virtual);
        assert_eq!(virtual_reg.class, RegisterClass::General);
    }
}
