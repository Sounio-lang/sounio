//! Compare normalized IR modules for semantic equivalence

use crate::{IrFunction, IrInstr, SoirModule};

/// Compare two normalized IR modules for semantic equivalence
///
/// Returns `true` if the modules are semantically equivalent (after normalization),
/// `false` otherwise.
///
/// This function should be called on modules that have already been normalized
/// via the `normalize()` function. If comparing unnormalized modules, differences
/// in register numbering, label order, or function order will cause false negatives.
pub fn compare(a: &SoirModule, b: &SoirModule) -> bool {
    // Compare function counts
    if a.fn_count != b.fn_count {
        return false;
    }

    // Compare string counts
    if a.string_count != b.string_count {
        return false;
    }

    // Compare each function
    for i in 0..(a.fn_count as usize) {
        if !compare_function(&a.functions[i], &b.functions[i]) {
            return false;
        }
    }

    // Compare string table
    for i in 0..(a.string_count as usize) {
        if a.string_table[i] != b.string_table[i] {
            return false;
        }
    }

    true
}

fn compare_function(a: &IrFunction, b: &IrFunction) -> bool {
    // Compare names
    if a.name != b.name {
        return false;
    }

    // Compare counts
    if a.instr_count != b.instr_count
        || a.reg_count != b.reg_count
        || a.label_count != b.label_count
        || a.param_count != b.param_count
    {
        return false;
    }

    // Compare param_regs
    for i in 0..(a.param_count as usize) {
        if a.param_regs[i] != b.param_regs[i] {
            return false;
        }
    }

    // Compare instructions
    for i in 0..(a.instr_count as usize) {
        if !compare_instr(&a.instrs[i], &b.instrs[i]) {
            return false;
        }
    }

    true
}

fn compare_instr(a: &IrInstr, b: &IrInstr) -> bool {
    // Compare opcode
    if a.op != b.op {
        return false;
    }

    // Compare registers
    if a.dst != b.dst || a.src1 != b.src1 || a.src2 != b.src2 {
        return false;
    }

    // Compare immediates
    if a.imm_i64 != b.imm_i64 {
        return false;
    }

    // For floats, use bitwise comparison to handle NaN correctly
    if a.imm_f64.to_bits() != b.imm_f64.to_bits() {
        return false;
    }

    // Compare IDs
    if a.label_id != b.label_id || a.fn_id != b.fn_id || a.field_idx != b.field_idx {
        return false;
    }

    // Compare operators
    if a.bin_op != b.bin_op || a.un_op != b.un_op {
        return false;
    }

    // Compare name
    if a.name != b.name {
        return false;
    }

    // Compare arg_count
    if a.arg_count != b.arg_count {
        return false;
    }

    true
}

/// Detailed comparison result with diagnostic information
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompareResult {
    Equal,
    DifferentFunctionCount {
        a: i64,
        b: i64,
    },
    DifferentStringCount {
        a: i64,
        b: i64,
    },
    DifferentFunctionName {
        index: usize,
        a: String,
        b: String,
    },
    DifferentInstrCount {
        func_index: usize,
        a: i64,
        b: i64,
    },
    DifferentOpcode {
        func_index: usize,
        instr_index: usize,
    },
    DifferentRegisters {
        func_index: usize,
        instr_index: usize,
    },
    DifferentImmediate {
        func_index: usize,
        instr_index: usize,
    },
}

/// Compare two modules with detailed diagnostics
pub fn compare_detailed(a: &SoirModule, b: &SoirModule) -> CompareResult {
    if a.fn_count != b.fn_count {
        return CompareResult::DifferentFunctionCount {
            a: a.fn_count,
            b: b.fn_count,
        };
    }

    if a.string_count != b.string_count {
        return CompareResult::DifferentStringCount {
            a: a.string_count,
            b: b.string_count,
        };
    }

    for i in 0..(a.fn_count as usize) {
        let fa = &a.functions[i];
        let fb = &b.functions[i];

        if fa.name != fb.name {
            return CompareResult::DifferentFunctionName {
                index: i,
                a: fa.name.as_str().unwrap_or("<invalid>").to_string(),
                b: fb.name.as_str().unwrap_or("<invalid>").to_string(),
            };
        }

        if fa.instr_count != fb.instr_count {
            return CompareResult::DifferentInstrCount {
                func_index: i,
                a: fa.instr_count,
                b: fb.instr_count,
            };
        }

        for j in 0..(fa.instr_count as usize) {
            let ia = &fa.instrs[j];
            let ib = &fb.instrs[j];

            if ia.op != ib.op {
                return CompareResult::DifferentOpcode {
                    func_index: i,
                    instr_index: j,
                };
            }

            if ia.dst != ib.dst || ia.src1 != ib.src1 || ia.src2 != ib.src2 {
                return CompareResult::DifferentRegisters {
                    func_index: i,
                    instr_index: j,
                };
            }

            if ia.imm_i64 != ib.imm_i64 || ia.imm_f64.to_bits() != ib.imm_f64.to_bits() {
                return CompareResult::DifferentImmediate {
                    func_index: i,
                    instr_index: j,
                };
            }
        }
    }

    CompareResult::Equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::normalize;
    use crate::types::*;

    #[test]
    fn test_compare_equal_modules() {
        let mut module = SoirModule::empty();
        let mut func = IrFunction::empty();
        func.name = Name::from("test");
        func.instr_count = 1;
        func.instrs[0] = IrInstr {
            op: IrOpcode::IrReturn,
            src1: 0,
            ..IrInstr::nop()
        };

        module.functions[0] = func;
        module.fn_count = 1;

        let normalized_a = normalize(&module);
        let normalized_b = normalize(&module);

        assert!(compare(&normalized_a, &normalized_b));
    }

    #[test]
    fn test_compare_different_function_count() {
        let mut module_a = SoirModule::empty();
        module_a.fn_count = 1;

        let mut module_b = SoirModule::empty();
        module_b.fn_count = 2;

        assert!(!compare(&module_a, &module_b));
    }

    #[test]
    fn test_compare_detailed() {
        let mut module_a = SoirModule::empty();
        module_a.fn_count = 1;

        let mut module_b = SoirModule::empty();
        module_b.fn_count = 2;

        let result = compare_detailed(&module_a, &module_b);
        assert!(matches!(
            result,
            CompareResult::DifferentFunctionCount { .. }
        ));
    }
}
