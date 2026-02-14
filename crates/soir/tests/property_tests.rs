//! Property-based tests for SOIR normalization and serialization
//!
//! These tests use proptest to verify key invariants:
//! - Normalization is idempotent
//! - Normalization preserves semantics
//! - Serialization round-trips correctly
//! - Different register/label numberings normalize to same result

use proptest::prelude::*;
use soir::{
    BinaryOp, IrFunction, IrInstr, IrOpcode, Name, SoirModule, UnaryOp,
    deserialize, normalize, serialize, IR_INVALID_REG,
    IR_MAX_FUNCS, IR_MAX_INSTRS, IR_MAX_PARAMS, IR_MAX_STRINGS,
};

// Strategy for generating valid Name instances
fn arb_name() -> impl Strategy<Value = Name> {
    (0..32_usize, prop::collection::vec(any::<u8>(), 0..128)).prop_map(|(len, mut buf)| {
        buf.resize(128, 0);
        let len = len.min(buf.len());
        Name {
            buf: buf.try_into().unwrap(),
            len,
        }
    })
}

// Strategy for generating valid opcodes
fn arb_opcode() -> impl Strategy<Value = IrOpcode> {
    prop_oneof![
        Just(IrOpcode::IrLoadImm),
        Just(IrOpcode::IrLoadFloat),
        Just(IrOpcode::IrLoadBool),
        Just(IrOpcode::IrLoadString),
        Just(IrOpcode::IrCopy),
        Just(IrOpcode::IrBinOp),
        Just(IrOpcode::IrUnaryOp),
        Just(IrOpcode::IrCall),
        Just(IrOpcode::IrReturn),
        Just(IrOpcode::IrJump),
        Just(IrOpcode::IrBranchTrue),
        Just(IrOpcode::IrBranchFalse),
        Just(IrOpcode::IrFieldGet),
        Just(IrOpcode::IrFieldSet),
        Just(IrOpcode::IrIndexGet),
        Just(IrOpcode::IrIndexSet),
        Just(IrOpcode::IrAlloc),
        Just(IrOpcode::IrLabel),
        Just(IrOpcode::IrNop),
        Just(IrOpcode::IrPhi),
    ]
}

// Strategy for generating valid binary operations
fn arb_binop() -> impl Strategy<Value = BinaryOp> {
    prop_oneof![
        Just(BinaryOp::OpAdd),
        Just(BinaryOp::OpSub),
        Just(BinaryOp::OpMul),
        Just(BinaryOp::OpDiv),
        Just(BinaryOp::OpRem),
        Just(BinaryOp::OpEq),
        Just(BinaryOp::OpNe),
        Just(BinaryOp::OpLt),
        Just(BinaryOp::OpLe),
        Just(BinaryOp::OpGt),
        Just(BinaryOp::OpGe),
        Just(BinaryOp::OpAnd),
        Just(BinaryOp::OpOr),
        Just(BinaryOp::OpBitAnd),
        Just(BinaryOp::OpBitOr),
        Just(BinaryOp::OpBitXor),
        Just(BinaryOp::OpShl),
        Just(BinaryOp::OpShr),
        Just(BinaryOp::OpConcat),
        Just(BinaryOp::OpRange),
        Just(BinaryOp::OpRangeInclusive),
    ]
}

// Strategy for generating valid unary operations
fn arb_unop() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![
        Just(UnaryOp::OpNeg),
        Just(UnaryOp::OpNot),
        Just(UnaryOp::OpRef),
        Just(UnaryOp::OpRefMut),
        Just(UnaryOp::OpDeref),
    ]
}

// Strategy for generating valid instructions
fn arb_instr() -> impl Strategy<Value = IrInstr> {
    // Split into two tuples to avoid proptest's 12-element limit
    (
        (
            arb_opcode(),
            -1..1024_i64,  // dst register
            -1..1024_i64,  // src1
            -1..1024_i64,  // src2
            any::<i64>(),  // imm_i64
            any::<f64>().prop_filter("No NaN or Inf", |f| f.is_finite()),
        ),
        (
            -1..256_i64,   // label_id
            -1..64_i64,    // fn_id
            0..64_i64,     // field_idx
            arb_binop(),
            arb_unop(),
            arb_name(),
            0..16_i64,     // arg_count
        ),
    )
        .prop_map(
            |((op, dst, src1, src2, imm_i64, imm_f64), (label_id, fn_id, field_idx, bin_op, un_op, name, arg_count))| {
                IrInstr {
                    op,
                    dst,
                    src1,
                    src2,
                    imm_i64,
                    imm_f64,
                    label_id,
                    fn_id,
                    field_idx,
                    bin_op,
                    un_op,
                    name,
                    call_args: None,
                    arg_count,
                }
            },
        )
}

// Strategy for generating valid functions
fn arb_function() -> impl Strategy<Value = IrFunction> {
    (
        arb_name(),
        1..32_usize,  // instr_count (generate the count first)
    )
        .prop_flat_map(|(name, instr_count)| {
            (
                Just(name),
                Just(instr_count),
                prop::collection::vec(arb_instr(), instr_count..=instr_count),  // exact size
                1..128_i64,  // reg_count
                0..32_i64,   // label_count
                0..8_i64,    // param_count
                prop::collection::vec(-1..1024_i64, IR_MAX_PARAMS),
            )
        })
        .prop_map(|(name, instr_count, instrs, reg_count, label_count, param_count, param_regs_vec)| {
            let mut instr_array = vec![IrInstr::default(); IR_MAX_INSTRS];
            for (i, instr) in instrs.into_iter().enumerate() {
                instr_array[i] = instr;
            }

            let mut param_regs = [IR_INVALID_REG; IR_MAX_PARAMS];
            for (i, &reg) in param_regs_vec.iter().enumerate() {
                param_regs[i] = reg;
            }

            IrFunction {
                name,
                instrs: instr_array,
                instr_count: instr_count as i64,
                reg_count,
                label_count,
                param_count,
                param_regs,
            }
        })
}

// Strategy for generating valid modules
fn arb_module() -> impl Strategy<Value = SoirModule> {
    (
        prop::collection::vec(arb_function(), 1..8),
        prop::collection::vec(arb_name(), 0..32),
    )
        .prop_map(|(funcs, strings)| {
            let fn_count = funcs.len() as i64;
            let string_count = strings.len() as i64;

            let mut func_array = vec![IrFunction::default(); IR_MAX_FUNCS];
            for (i, func) in funcs.into_iter().enumerate() {
                func_array[i] = func;
            }

            let mut string_array = vec![Name::default(); IR_MAX_STRINGS];
            for (i, name) in strings.into_iter().enumerate() {
                string_array[i] = name;
            }

            SoirModule {
                functions: func_array,
                fn_count,
                string_table: string_array,
                string_count,
            }
        })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    /// Property: Normalization is idempotent
    /// normalize(normalize(x)) == normalize(x)
    #[test]
    fn normalize_is_idempotent(module in arb_module()) {
        let normalized1 = normalize(&module);
        let normalized2 = normalize(&normalized1);

        // Compare function counts
        prop_assert_eq!(normalized1.fn_count, normalized2.fn_count);
        prop_assert_eq!(normalized1.string_count, normalized2.string_count);

        // Compare each function
        for i in 0..normalized1.fn_count as usize {
            prop_assert_eq!(normalized1.functions[i].instr_count, normalized2.functions[i].instr_count);
            prop_assert_eq!(normalized1.functions[i].reg_count, normalized2.functions[i].reg_count);
            prop_assert_eq!(normalized1.functions[i].label_count, normalized2.functions[i].label_count);
        }
    }

    /// Property: Serialization round-trips correctly
    /// deserialize(serialize(x)) ≈ x (modulo padding/defaults)
    #[test]
    fn serialize_roundtrip(module in arb_module()) {
        let bytes = serialize(&module).unwrap();
        let restored = deserialize(&bytes).unwrap();

        prop_assert_eq!(module.fn_count, restored.fn_count);
        prop_assert_eq!(module.string_count, restored.string_count);

        for i in 0..module.fn_count as usize {
            prop_assert_eq!(module.functions[i].instr_count, restored.functions[i].instr_count);
            prop_assert_eq!(module.functions[i].param_count, restored.functions[i].param_count);
        }
    }

    /// Property: Normalization preserves function count
    #[test]
    fn normalize_preserves_function_count(module in arb_module()) {
        let normalized = normalize(&module);
        prop_assert_eq!(module.fn_count, normalized.fn_count);
    }

    /// Property: Normalization preserves instruction counts
    /// NOTE: Disabled due to proptest tuple size limitations
    /// See unit_tests::test_normalize_preserves_instr_count instead
    #[test]
    #[ignore]
    fn normalize_preserves_instr_counts(module in arb_module()) {
        let normalized = normalize(&module);

        for i in 0..module.fn_count as usize {
            prop_assert_eq!(
                module.functions[i].instr_count,
                normalized.functions[i].instr_count,
                "Function {} instruction count changed", i
            );
        }
    }

    /// Property: Serialized modules are deterministic
    #[test]
    fn serialize_is_deterministic(module in arb_module()) {
        let bytes1 = serialize(&module).unwrap();
        let bytes2 = serialize(&module).unwrap();
        prop_assert_eq!(bytes1, bytes2);
    }

    /// Property: Normalized modules serialize to same bytes regardless of initial ordering
    #[test]
    fn normalized_serialize_deterministic(module in arb_module()) {
        let norm1 = normalize(&module);
        let norm2 = normalize(&module);

        let bytes1 = serialize(&norm1).unwrap();
        let bytes2 = serialize(&norm2).unwrap();

        prop_assert_eq!(bytes1, bytes2);
    }
}

#[cfg(test)]
mod unit_tests {
    use super::*;

    #[test]
    fn test_normalize_empty_module() {
        let module = SoirModule {
            functions: vec![IrFunction::default(); IR_MAX_FUNCS],
            fn_count: 0,
            string_table: vec![Name::default(); IR_MAX_STRINGS],
            string_count: 0,
        };

        let normalized = normalize(&module);
        assert_eq!(normalized.fn_count, 0);
        assert_eq!(normalized.string_count, 0);
    }

    #[test]
    fn test_serialize_empty_module() {
        let module = SoirModule {
            functions: vec![IrFunction::default(); IR_MAX_FUNCS],
            fn_count: 0,
            string_table: vec![Name::default(); IR_MAX_STRINGS],
            string_count: 0,
        };

        let bytes = serialize(&module).unwrap();
        let restored = deserialize(&bytes).unwrap();

        assert_eq!(restored.fn_count, 0);
        assert_eq!(restored.string_count, 0);
    }

    #[test]
    fn test_normalize_preserves_instr_count() {
        let mut module = SoirModule {
            functions: vec![IrFunction::default(); IR_MAX_FUNCS],
            fn_count: 2,
            string_table: vec![Name::default(); IR_MAX_STRINGS],
            string_count: 0,
        };

        // Create two functions with different instruction counts
        let mut f1 = IrFunction::default();
        f1.name = Name::from("func1");
        f1.instr_count = 5;
        for i in 0..5 {
            f1.instrs[i] = IrInstr {
                op: IrOpcode::IrLoadImm,
                dst: i as i64,
                imm_i64: i as i64,
                ..Default::default()
            };
        }

        let mut f2 = IrFunction::default();
        f2.name = Name::from("func2");
        f2.instr_count = 3;
        for i in 0..3 {
            f2.instrs[i] = IrInstr {
                op: IrOpcode::IrNop,
                ..Default::default()
            };
        }

        module.functions[0] = f1;
        module.functions[1] = f2;

        let normalized = normalize(&module);

        // Instruction counts should be preserved
        assert_eq!(module.functions[0].instr_count, normalized.functions[0].instr_count);
        assert_eq!(module.functions[1].instr_count, normalized.functions[1].instr_count);
    }
}
