//! Integration tests for SOIR serialization roundtrip
//!
//! These tests verify that programs can be:
//! 1. Compiled to IR
//! 2. Serialized to SOIR binary format
//! 3. Deserialized back
//! 4. Normalized for comparison
//!
//! This ensures the entire pipeline works correctly end-to-end.

use soir::{deserialize, normalize, serialize};
use std::path::PathBuf;

fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

#[test]
fn simple_program_roundtrip() {
    // Create a minimal SOIR module directly
    use soir::{IrFunction, IrInstr, IrOpcode, Name, SoirModule, IR_INVALID_REG, IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS};

    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Create simple main function that returns 42
    let mut func = IrFunction {
        name: Name {
            buf: *b"main\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 4,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 2,
        reg_count: 1,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    func.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 42,
        ..Default::default()
    };

    func.instrs[1] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 0,
        ..Default::default()
    };

    module.functions[0] = func;

    // Serialize
    let bytes = serialize(&module).expect("Failed to serialize simple program");

    // Deserialize
    let restored = deserialize(&bytes).expect("Failed to deserialize simple program");

    // Verify structure
    assert_eq!(restored.fn_count, 1);
    assert_eq!(restored.functions[0].instr_count, 2);
    assert_eq!(restored.functions[0].instrs[0].op, IrOpcode::IrLoadImm);
    assert_eq!(restored.functions[0].instrs[0].imm_i64, 42);
    assert_eq!(restored.functions[0].instrs[1].op, IrOpcode::IrReturn);
}

#[test]
fn normalization_equivalence() {
    // Create two syntactically different but semantically equivalent modules
    use soir::{BinaryOp, IrFunction, IrInstr, IrOpcode, Name, SoirModule, IR_INVALID_REG, IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS};

    // Module A: Uses registers R5 and R10
    let mut module_a = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func_a = IrFunction {
        name: Name {
            buf: *b"add\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 3,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 4,
        reg_count: 3,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    func_a.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 5,
        imm_i64: 10,
        ..Default::default()
    };
    func_a.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 10,
        imm_i64: 20,
        ..Default::default()
    };
    func_a.instrs[2] = IrInstr {
        op: IrOpcode::IrBinOp,
        dst: 15,
        src1: 5,
        src2: 10,
        bin_op: BinaryOp::OpAdd,
        ..Default::default()
    };
    func_a.instrs[3] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 15,
        ..Default::default()
    };

    module_a.functions[0] = func_a;

    // Module B: Uses registers R2 and R7 (same logic, different registers)
    let mut module_b = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func_b = IrFunction {
        name: Name {
            buf: *b"add\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 3,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 4,
        reg_count: 3,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    func_b.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 2,
        imm_i64: 10,
        ..Default::default()
    };
    func_b.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 7,
        imm_i64: 20,
        ..Default::default()
    };
    func_b.instrs[2] = IrInstr {
        op: IrOpcode::IrBinOp,
        dst: 42,
        src1: 2,
        src2: 7,
        bin_op: BinaryOp::OpAdd,
        ..Default::default()
    };
    func_b.instrs[3] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 42,
        ..Default::default()
    };

    module_b.functions[0] = func_b;

    // Normalize both
    let norm_a = normalize(&module_a);
    let norm_b = normalize(&module_b);

    // Serialize both normalized modules
    let bytes_a = serialize(&norm_a).expect("Failed to serialize norm_a");
    let bytes_b = serialize(&norm_b).expect("Failed to serialize norm_b");

    // They should produce identical bytes
    assert_eq!(
        bytes_a, bytes_b,
        "Normalized modules should serialize to identical bytes"
    );
}

#[test]
fn multi_function_module() {
    use soir::{IrFunction, IrInstr, IrOpcode, Name, SoirModule, IR_INVALID_REG, IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS};

    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 3,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Function 1: helper
    let mut func1 = IrFunction {
        name: Name {
            buf: *b"helper\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 6,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 2,
        reg_count: 1,
        label_count: 0,
        param_count: 1,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };
    func1.param_regs[0] = 0;
    func1.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1,
        imm_i64: 1,
        ..Default::default()
    };
    func1.instrs[1] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 1,
        ..Default::default()
    };

    // Function 2: main
    let mut func2 = IrFunction {
        name: Name {
            buf: *b"main\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 4,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 3,
        reg_count: 2,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };
    func2.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 5,
        ..Default::default()
    };
    func2.instrs[1] = IrInstr {
        op: IrOpcode::IrCall,
        dst: 1,
        fn_id: 0,
        arg_count: 1,
        name: Name {
            buf: *b"helper\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 6,
        },
        ..Default::default()
    };
    func2.instrs[2] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 1,
        ..Default::default()
    };

    // Function 3: utility
    let mut func3 = IrFunction {
        name: Name {
            buf: *b"utility\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 7,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 1,
        reg_count: 0,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };
    func3.instrs[0] = IrInstr {
        op: IrOpcode::IrNop,
        ..Default::default()
    };

    module.functions[0] = func1;
    module.functions[1] = func2;
    module.functions[2] = func3;

    // Roundtrip
    let bytes = serialize(&module).expect("Failed to serialize multi-function module");
    let restored = deserialize(&bytes).expect("Failed to deserialize multi-function module");

    assert_eq!(restored.fn_count, 3);

    // Verify function names are preserved
    let name0 = std::str::from_utf8(&restored.functions[0].name.buf[..restored.functions[0].name.len]).unwrap();
    let name1 = std::str::from_utf8(&restored.functions[1].name.buf[..restored.functions[1].name.len]).unwrap();
    let name2 = std::str::from_utf8(&restored.functions[2].name.buf[..restored.functions[2].name.len]).unwrap();

    assert_eq!(name0, "helper");
    assert_eq!(name1, "main");
    assert_eq!(name2, "utility");
}

#[test]
fn control_flow_preservation() {
    use soir::{IrFunction, IrInstr, IrOpcode, Name, SoirModule, IR_INVALID_REG, IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS};

    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Function with if-else control flow
    let mut func = IrFunction {
        name: Name {
            buf: *b"conditional\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 11,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 8,
        reg_count: 3,
        label_count: 2,
        param_count: 1,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };
    func.param_regs[0] = 0;

    func.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadBool,
        dst: 1,
        imm_i64: 1,
        ..Default::default()
    };
    func.instrs[1] = IrInstr {
        op: IrOpcode::IrBranchFalse,
        src1: 1,
        label_id: 0,
        ..Default::default()
    };
    func.instrs[2] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 2,
        imm_i64: 10,
        ..Default::default()
    };
    func.instrs[3] = IrInstr {
        op: IrOpcode::IrJump,
        label_id: 1,
        ..Default::default()
    };
    func.instrs[4] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 0,
        ..Default::default()
    };
    func.instrs[5] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 2,
        imm_i64: 20,
        ..Default::default()
    };
    func.instrs[6] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 1,
        ..Default::default()
    };
    func.instrs[7] = IrInstr {
        op: IrOpcode::IrReturn,
        src1: 2,
        ..Default::default()
    };

    module.functions[0] = func;

    // Roundtrip
    let bytes = serialize(&module).expect("Failed to serialize control flow");
    let restored = deserialize(&bytes).expect("Failed to deserialize control flow");

    assert_eq!(restored.functions[0].instr_count, 8);
    assert_eq!(restored.functions[0].label_count, 2);

    // Verify control flow structure
    assert_eq!(restored.functions[0].instrs[1].op, IrOpcode::IrBranchFalse);
    assert_eq!(restored.functions[0].instrs[3].op, IrOpcode::IrJump);
    assert_eq!(restored.functions[0].instrs[4].op, IrOpcode::IrLabel);
    assert_eq!(restored.functions[0].instrs[6].op, IrOpcode::IrLabel);
}
