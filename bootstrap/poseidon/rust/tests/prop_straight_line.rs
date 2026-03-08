mod common;

use common::{branch_program, eval_binop, eval_unop, straight_line_program, unary_program, BinOp, UnaryOp};
use poseidon_wrapper::{Module, VmConfig};
use proptest::prelude::*;

fn binop_case_strategy() -> impl Strategy<Value = (BinOp, i64, i64)> {
    prop_oneof![
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Add, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Sub, lhs, rhs)),
        (-100_i64..100_i64, -100_i64..100_i64).prop_map(|(lhs, rhs)| (BinOp::Mul, lhs, rhs)),
        (-1000_i64..1000_i64, -100_i64..100_i64).prop_map(|(lhs, rhs)| (BinOp::Div, lhs, rhs)),
        (-1000_i64..1000_i64, -100_i64..100_i64).prop_map(|(lhs, rhs)| (BinOp::Mod, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Eq, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Ne, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Lt, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Le, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Gt, lhs, rhs)),
        (-1000_i64..1000_i64, -1000_i64..1000_i64).prop_map(|(lhs, rhs)| (BinOp::Ge, lhs, rhs)),
        (-8_i64..8_i64, -8_i64..8_i64).prop_map(|(lhs, rhs)| (BinOp::And, lhs, rhs)),
        (-8_i64..8_i64, -8_i64..8_i64).prop_map(|(lhs, rhs)| (BinOp::Or, lhs, rhs)),
        (-256_i64..256_i64, -256_i64..256_i64).prop_map(|(lhs, rhs)| (BinOp::BitAnd, lhs, rhs)),
        (-256_i64..256_i64, -256_i64..256_i64).prop_map(|(lhs, rhs)| (BinOp::BitOr, lhs, rhs)),
        (-256_i64..256_i64, -256_i64..256_i64).prop_map(|(lhs, rhs)| (BinOp::BitXor, lhs, rhs)),
        (0_i64..1024_i64, 0_i64..8_i64).prop_map(|(lhs, rhs)| (BinOp::Shl, lhs, rhs)),
        (0_i64..1024_i64, 0_i64..8_i64).prop_map(|(lhs, rhs)| (BinOp::Shr, lhs, rhs)),
    ]
}

fn unary_strategy() -> impl Strategy<Value = UnaryOp> {
    prop_oneof![Just(UnaryOp::Neg), Just(UnaryOp::Not)]
}

proptest! {
    #[test]
    fn straight_line_programs_match_host_eval(
        (op, lhs, rhs) in binop_case_strategy()
    ) {
        let bytes = straight_line_program(lhs, rhs, op);
        let module = Module::from_bytes(&bytes).expect("module");
        let result = module.run(&VmConfig::default()).expect("run");
        prop_assert_eq!(result.exit_code, eval_binop(lhs, rhs, op));
        prop_assert!(result.steps_executed > 0);
    }

    #[test]
    fn unary_programs_match_host_eval(
        src in -1000_i64..1000_i64,
        op in unary_strategy()
    ) {
        let bytes = unary_program(src, op);
        let module = Module::from_bytes(&bytes).expect("module");
        let result = module.run(&VmConfig::default()).expect("run");
        prop_assert_eq!(result.exit_code, eval_unop(src, op));
        prop_assert!(result.steps_executed > 0);
    }

    #[test]
    fn branch_programs_match_host_eval(
        lhs in -1000_i64..1000_i64,
        rhs in -1000_i64..1000_i64,
        then_value in -1000_i64..1000_i64,
        else_value in -1000_i64..1000_i64,
        pred in prop_oneof![Just(BinOp::Eq), Just(BinOp::Lt), Just(BinOp::Ge)]
    ) {
        let bytes = branch_program(lhs, rhs, pred, then_value, else_value);
        let module = Module::from_bytes(&bytes).expect("module");
        let result = module.run(&VmConfig::default()).expect("run");
        let expected = if eval_binop(lhs, rhs, pred) != 0 { then_value } else { else_value };
        prop_assert_eq!(result.exit_code, expected);
    }
}
