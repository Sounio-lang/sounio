//! Edge case and boundary condition tests for SOIR

use soir::{
    BinaryOp, IR_INVALID_REG, IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS,
    IrFunction, IrInstr, IrOpcode, Name, SOIR_MAGIC, SOIR_VERSION, SoirError, SoirModule,
    deserialize, normalize, serialize,
};

#[test]
fn empty_module() {
    let module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 0,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let bytes = serialize(&module).expect("Failed to serialize empty module");
    let restored = deserialize(&bytes).expect("Failed to deserialize empty module");

    assert_eq!(restored.fn_count, 0);
    assert_eq!(restored.string_count, 0);
}

#[test]
fn maximum_functions() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: IR_MAX_FUNCS as i64,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Create simple function for each slot
    for i in 0..IR_MAX_FUNCS {
        let mut name = Name::default();
        let name_str = format!("fn_{}", i);
        let bytes = name_str.as_bytes();
        name.len = bytes.len().min(128);
        name.buf[..name.len].copy_from_slice(&bytes[..name.len]);

        module.functions[i] = IrFunction {
            name,
            instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
            instr_count: 1,
            reg_count: 1,
            label_count: 0,
            param_count: 0,
            param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
        };
        module.functions[i].instrs[0] = IrInstr {
            op: IrOpcode::IrReturn,
            src1: 0,
            ..Default::default()
        };
    }

    let bytes = serialize(&module).expect("Failed to serialize max functions");
    let restored = deserialize(&bytes).expect("Failed to deserialize max functions");

    assert_eq!(restored.fn_count, IR_MAX_FUNCS as i64);
}

#[test]
fn maximum_instructions() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Use 500 instructions (safely under 128KB limit but still large)
    // Each instruction is ~128 bytes, so 500 * 128 = 64KB
    const LARGE_INSTR_COUNT: usize = 500;

    let mut func = IrFunction {
        name: Name {
            buf: *b"test\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 4,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: LARGE_INSTR_COUNT as i64,
        reg_count: 100,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    // Fill with simple nop instructions
    for i in 0..LARGE_INSTR_COUNT {
        func.instrs[i] = IrInstr {
            op: IrOpcode::IrNop,
            ..Default::default()
        };
    }

    module.functions[0] = func;

    let bytes = serialize(&module).expect("Failed to serialize many instructions");
    let restored = deserialize(&bytes).expect("Failed to deserialize many instructions");

    assert_eq!(restored.functions[0].instr_count, LARGE_INSTR_COUNT as i64);
}

#[test]
fn maximum_string_table() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 0,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: IR_MAX_STRINGS as i64,
    };

    // Fill string table with unique strings
    for i in 0..IR_MAX_STRINGS {
        let mut name = Name::default();
        let name_str = format!("string_{}", i);
        let bytes = name_str.as_bytes();
        name.len = bytes.len().min(128);
        name.buf[..name.len].copy_from_slice(&bytes[..name.len]);
        module.string_table[i] = name;
    }

    let bytes = serialize(&module).expect("Failed to serialize max strings");
    let restored = deserialize(&bytes).expect("Failed to deserialize max strings");

    assert_eq!(restored.string_count, IR_MAX_STRINGS as i64);
}

#[test]
fn all_opcodes() {
    let opcodes = vec![
        IrOpcode::IrLoadImm,
        IrOpcode::IrLoadFloat,
        IrOpcode::IrLoadBool,
        IrOpcode::IrLoadString,
        IrOpcode::IrCopy,
        IrOpcode::IrBinOp,
        IrOpcode::IrUnaryOp,
        IrOpcode::IrCall,
        IrOpcode::IrReturn,
        IrOpcode::IrJump,
        IrOpcode::IrBranchTrue,
        IrOpcode::IrBranchFalse,
        IrOpcode::IrFieldGet,
        IrOpcode::IrFieldSet,
        IrOpcode::IrIndexGet,
        IrOpcode::IrIndexSet,
        IrOpcode::IrAlloc,
        IrOpcode::IrLabel,
        IrOpcode::IrNop,
        IrOpcode::IrPhi,
    ];

    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func = IrFunction {
        name: Name {
            buf: *b"test_all_ops\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 12,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: opcodes.len() as i64,
        reg_count: 10,
        label_count: 2,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    for (i, opcode) in opcodes.iter().enumerate() {
        func.instrs[i] = IrInstr {
            op: *opcode,
            dst: i as i64,
            src1: 0,
            src2: 1,
            imm_i64: 42,
            imm_f64: 3.14,
            label_id: if matches!(opcode, IrOpcode::IrLabel) {
                i as i64
            } else {
                0
            },
            ..Default::default()
        };
    }

    module.functions[0] = func;

    let bytes = serialize(&module).expect("Failed to serialize all opcodes");
    let restored = deserialize(&bytes).expect("Failed to deserialize all opcodes");

    assert_eq!(restored.functions[0].instr_count, opcodes.len() as i64);

    // Verify all opcodes are preserved
    for (i, expected_op) in opcodes.iter().enumerate() {
        assert_eq!(
            restored.functions[0].instrs[i].op, *expected_op,
            "Opcode mismatch at index {}",
            i
        );
    }
}

#[test]
fn register_boundary_conditions() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func = IrFunction {
        name: Name {
            buf: *b"reg_test\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 8,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 4,
        reg_count: 1024,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    // Test register 0
    func.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 1,
        ..Default::default()
    };

    // Test high register number
    func.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 1023,
        imm_i64: 2,
        ..Default::default()
    };

    // Test invalid register (should be preserved)
    func.instrs[2] = IrInstr {
        op: IrOpcode::IrNop,
        dst: IR_INVALID_REG,
        ..Default::default()
    };

    // Test copy between registers
    func.instrs[3] = IrInstr {
        op: IrOpcode::IrCopy,
        dst: 0,
        src1: 1023,
        ..Default::default()
    };

    module.functions[0] = func;

    let bytes = serialize(&module).expect("Failed to serialize register boundaries");
    let restored = deserialize(&bytes).expect("Failed to deserialize register boundaries");

    assert_eq!(restored.functions[0].instrs[0].dst, 0);
    assert_eq!(restored.functions[0].instrs[1].dst, 1023);
    assert_eq!(restored.functions[0].instrs[2].dst, IR_INVALID_REG);
    assert_eq!(restored.functions[0].instrs[3].src1, 1023);
}

#[test]
fn invalid_magic_bytes() {
    let mut bad_data = vec![0u8; 16];
    bad_data[0..4].copy_from_slice(b"XXXX"); // Wrong magic
    bad_data[4] = SOIR_VERSION;

    let result = deserialize(&bad_data);
    assert!(matches!(result, Err(SoirError::InvalidMagic(_))));
}

#[test]
fn unsupported_version() {
    let mut bad_data = vec![0u8; 16];
    bad_data[0..4].copy_from_slice(&SOIR_MAGIC);
    bad_data[4] = 99; // Unsupported version

    let result = deserialize(&bad_data);
    assert!(matches!(result, Err(SoirError::UnsupportedVersion(99, _))));
}

#[test]
fn truncated_header() {
    let bad_data = vec![0x53, 0x4F, 0x49]; // Only 3 bytes

    let result = deserialize(&bad_data);
    assert!(result.is_err());
}

#[test]
fn empty_data() {
    let result = deserialize(&[]);
    assert!(result.is_err());
}

#[test]
fn normalize_sorts_functions_alphabetically() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 3,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    // Add functions in reverse alphabetical order
    module.functions[0] = IrFunction {
        name: Name {
            buf: *b"zebra\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 5,
        },
        ..Default::default()
    };

    module.functions[1] = IrFunction {
        name: Name {
            buf: *b"middle\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 6,
        },
        ..Default::default()
    };

    module.functions[2] = IrFunction {
        name: Name {
            buf: *b"alpha\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 5,
        },
        ..Default::default()
    };

    let normalized = normalize(&module);

    // After normalization, should be alphabetically sorted
    let name0 =
        std::str::from_utf8(&normalized.functions[0].name.buf[..normalized.functions[0].name.len])
            .unwrap();
    let name1 =
        std::str::from_utf8(&normalized.functions[1].name.buf[..normalized.functions[1].name.len])
            .unwrap();
    let name2 =
        std::str::from_utf8(&normalized.functions[2].name.buf[..normalized.functions[2].name.len])
            .unwrap();

    assert_eq!(name0, "alpha");
    assert_eq!(name1, "middle");
    assert_eq!(name2, "zebra");
}

#[test]
fn normalize_renumbers_labels() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func = IrFunction {
        name: Name {
            buf: *b"test\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 4,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 4,
        reg_count: 2,
        label_count: 2,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    // Use labels out of order (L7, L2)
    func.instrs[0] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 7,
        ..Default::default()
    };
    func.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 0,
        imm_i64: 1,
        ..Default::default()
    };
    func.instrs[2] = IrInstr {
        op: IrOpcode::IrLabel,
        label_id: 2,
        ..Default::default()
    };
    func.instrs[3] = IrInstr {
        op: IrOpcode::IrJump,
        label_id: 7,
        ..Default::default()
    };

    module.functions[0] = func;

    let normalized = normalize(&module);

    // Labels should be renumbered in order of first appearance (L0, L1)
    assert_eq!(normalized.functions[0].instrs[0].label_id, 0);
    assert_eq!(normalized.functions[0].instrs[2].label_id, 1);
    assert_eq!(normalized.functions[0].instrs[3].label_id, 0); // Jump to first label
}

#[test]
fn normalize_renumbers_registers() {
    let mut module = SoirModule {
        functions: vec![IrFunction::default(); IR_MAX_FUNCS],
        fn_count: 1,
        string_table: vec![Name::default(); IR_MAX_STRINGS],
        string_count: 0,
    };

    let mut func = IrFunction {
        name: Name {
            buf: *b"test\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0\0",
            len: 4,
        },
        instrs: vec![IrInstr::default(); IR_MAX_INSTRS],
        instr_count: 3,
        reg_count: 3,
        label_count: 0,
        param_count: 0,
        param_regs: [IR_INVALID_REG; IR_MAX_PARAMS],
    };

    // Use registers with gaps (R100, R5, R42)
    func.instrs[0] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 100,
        imm_i64: 1,
        ..Default::default()
    };
    func.instrs[1] = IrInstr {
        op: IrOpcode::IrLoadImm,
        dst: 5,
        imm_i64: 2,
        ..Default::default()
    };
    func.instrs[2] = IrInstr {
        op: IrOpcode::IrBinOp,
        dst: 42,
        src1: 100,
        src2: 5,
        bin_op: BinaryOp::OpAdd,
        ..Default::default()
    };

    module.functions[0] = func;

    let normalized = normalize(&module);

    // Registers should be renumbered in order of first use
    // R100 -> R0, R5 -> R1, R42 -> R2
    assert_eq!(normalized.functions[0].instrs[0].dst, 0);
    assert_eq!(normalized.functions[0].instrs[1].dst, 1);
    assert_eq!(normalized.functions[0].instrs[2].dst, 2);
    assert_eq!(normalized.functions[0].instrs[2].src1, 0);
    assert_eq!(normalized.functions[0].instrs[2].src2, 1);
}
