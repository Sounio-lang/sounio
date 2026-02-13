//! IR normalization for deterministic semantic equivalence
//!
//! Normalization transforms IR into canonical form:
//! 1. Sort functions by name (alphabetically)
//! 2. Renumber labels by first definition order
//! 3. Renumber registers by first use order
//! 4. Sort string table alphabetically

use crate::{SoirModule, IrFunction, IrOpcode, IR_INVALID_REG, IR_INVALID_LABEL};

/// Normalize a SOIR module for deterministic comparison
pub fn normalize(module: &SoirModule) -> SoirModule {
    let mut normalized = module.clone();

    // Step 1: Sort functions by name
    let fn_count = normalized.fn_count as usize;
    normalized.functions[..fn_count].sort_by(|a, b| a.name.cmp(&b.name));

    // Step 2: Normalize each function
    for i in 0..fn_count {
        normalized.functions[i] = normalize_function(&normalized.functions[i]);
    }

    // Step 3: Sort string table
    let str_count = normalized.string_count as usize;
    normalized.string_table[..str_count].sort();

    normalized
}

fn normalize_function(func: &IrFunction) -> IrFunction {
    let mut normalized = func.clone();

    // Build label renumbering map
    let label_map = build_label_map(&normalized);

    // Apply label renumbering
    for i in 0..(normalized.instr_count as usize) {
        let instr = &mut normalized.instrs[i];
        if instr.label_id >= 0 && (instr.label_id as usize) < label_map.len() {
            instr.label_id = label_map[instr.label_id as usize];
        }
    }

    // Build register renumbering map
    let reg_map = build_register_map(&normalized);

    // Apply register renumbering to param_regs
    for i in 0..normalized.param_regs.len() {
        if normalized.param_regs[i] >= 0 && (normalized.param_regs[i] as usize) < reg_map.len() {
            normalized.param_regs[i] = reg_map[normalized.param_regs[i] as usize];
        }
    }

    // Apply register renumbering to instructions
    for i in 0..(normalized.instr_count as usize) {
        let instr = &mut normalized.instrs[i];
        instr.dst = map_register(instr.dst, &reg_map);
        instr.src1 = map_register(instr.src1, &reg_map);
        instr.src2 = map_register(instr.src2, &reg_map);
    }

    normalized
}

fn build_label_map(func: &IrFunction) -> Vec<i64> {
    let mut label_map = vec![IR_INVALID_LABEL; 256];
    let mut next_label = 0i64;

    for i in 0..(func.instr_count as usize) {
        let instr = &func.instrs[i];
        if instr.op == IrOpcode::IrLabel {
            let old_label = instr.label_id;
            if old_label >= 0 && (old_label as usize) < 256 {
                if label_map[old_label as usize] == IR_INVALID_LABEL {
                    label_map[old_label as usize] = next_label;
                    next_label += 1;
                }
            }
        }
    }

    label_map
}

fn build_register_map(func: &IrFunction) -> Vec<i64> {
    let mut reg_map = vec![IR_INVALID_REG; 1024];
    let mut next_reg = 0i64;

    for i in 0..(func.instr_count as usize) {
        let instr = &func.instrs[i];

        // Record dst register (definition)
        if instr.dst >= 0 && (instr.dst as usize) < 1024 {
            if reg_map[instr.dst as usize] == IR_INVALID_REG {
                reg_map[instr.dst as usize] = next_reg;
                next_reg += 1;
            }
        }

        // Record src1 register (use)
        if instr.src1 >= 0 && (instr.src1 as usize) < 1024 {
            if reg_map[instr.src1 as usize] == IR_INVALID_REG {
                reg_map[instr.src1 as usize] = next_reg;
                next_reg += 1;
            }
        }

        // Record src2 register (use)
        if instr.src2 >= 0 && (instr.src2 as usize) < 1024 {
            if reg_map[instr.src2 as usize] == IR_INVALID_REG {
                reg_map[instr.src2 as usize] = next_reg;
                next_reg += 1;
            }
        }
    }

    reg_map
}

fn map_register(old_reg: i64, reg_map: &[i64]) -> i64 {
    if old_reg >= 0 && (old_reg as usize) < reg_map.len() {
        let mapped = reg_map[old_reg as usize];
        if mapped != IR_INVALID_REG {
            return mapped;
        }
    }
    old_reg
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn test_normalize_sorts_functions() {
        let mut module = SoirModule::empty();

        let mut f1 = IrFunction::empty();
        f1.name = Name::from("zebra");

        let mut f2 = IrFunction::empty();
        f2.name = Name::from("apple");

        module.functions[0] = f1;
        module.functions[1] = f2;
        module.fn_count = 2;

        let normalized = normalize(&module);

        assert_eq!(normalized.functions[0].name.as_str().unwrap(), "apple");
        assert_eq!(normalized.functions[1].name.as_str().unwrap(), "zebra");
    }

    #[test]
    fn test_normalize_renumbers_labels() {
        let mut func = IrFunction::empty();
        func.instr_count = 4;
        func.instrs[0] = IrInstr {
            op: IrOpcode::IrLabel,
            label_id: 5,
            ..IrInstr::nop()
        };
        func.instrs[1] = IrInstr {
            op: IrOpcode::IrJump,
            label_id: 5,
            ..IrInstr::nop()
        };
        func.instrs[2] = IrInstr {
            op: IrOpcode::IrLabel,
            label_id: 3,
            ..IrInstr::nop()
        };
        func.instrs[3] = IrInstr {
            op: IrOpcode::IrJump,
            label_id: 3,
            ..IrInstr::nop()
        };

        let normalized = normalize_function(&func);

        // Labels should be renumbered to 0, 1 in order of first appearance
        assert_eq!(normalized.instrs[0].label_id, 0);
        assert_eq!(normalized.instrs[1].label_id, 0);
        assert_eq!(normalized.instrs[2].label_id, 1);
        assert_eq!(normalized.instrs[3].label_id, 1);
    }

    #[test]
    fn test_normalize_renumbers_registers() {
        let mut func = IrFunction::empty();
        func.instr_count = 2;
        func.instrs[0] = IrInstr {
            op: IrOpcode::IrLoadImm,
            dst: 10,
            imm_i64: 42,
            ..IrInstr::nop()
        };
        func.instrs[1] = IrInstr {
            op: IrOpcode::IrCopy,
            dst: 5,
            src1: 10,
            ..IrInstr::nop()
        };

        let normalized = normalize_function(&func);

        // R10 appears first (dst), should be R0
        // R5 appears second (dst), should be R1
        assert_eq!(normalized.instrs[0].dst, 0);
        assert_eq!(normalized.instrs[1].dst, 1);
        assert_eq!(normalized.instrs[1].src1, 0);
    }
}
