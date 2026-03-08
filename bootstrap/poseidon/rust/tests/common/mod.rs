#![allow(dead_code)]

use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const SOIR_MAGIC: u32 = 0x5249_4f53;
const SOIR_VERSION: u8 = 1;

const OP_LOAD_IMM: i8 = 0;
const OP_COPY: i8 = 4;
const OP_BINOP: i8 = 5;
const OP_UNARYOP: i8 = 6;
const OP_RETURN: i8 = 8;
const OP_JUMP: i8 = 9;
const OP_BRANCH_TRUE: i8 = 10;
const OP_LABEL: i8 = 17;

const BINOP_ADD: i8 = 0;
const BINOP_SUB: i8 = 1;
const BINOP_MUL: i8 = 2;
const BINOP_DIV: i8 = 3;
const BINOP_MOD: i8 = 4;
const BINOP_EQ: i8 = 5;
const BINOP_NE: i8 = 6;
const BINOP_LT: i8 = 7;
const BINOP_LE: i8 = 8;
const BINOP_GT: i8 = 9;
const BINOP_GE: i8 = 10;
const BINOP_AND: i8 = 11;
const BINOP_OR: i8 = 12;
const BINOP_BITAND: i8 = 13;
const BINOP_BITOR: i8 = 14;
const BINOP_BITXOR: i8 = 15;
const BINOP_SHL: i8 = 16;
const BINOP_SHR: i8 = 17;

const UNOP_NEG: i8 = 0;
const UNOP_NOT: i8 = 1;

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureManifest {
    pub schema: String,
    pub fixtures: Vec<FixtureSpec>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct FixtureSpec {
    pub id: String,
    pub filename: String,
    pub kind: String,
    pub tags: Vec<String>,
    pub expected_status: String,
    pub expected_exit_code: Option<i64>,
    pub max_steps: Option<i64>,
}

static TESTS_DIR: OnceLock<PathBuf> = OnceLock::new();
static FIXTURE_MANIFEST: OnceLock<FixtureManifest> = OnceLock::new();

fn write_i64(buf: &mut Vec<u8>, value: i64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_i8(buf: &mut Vec<u8>, value: i8) {
    buf.push(value as u8);
}

fn write_f64(buf: &mut Vec<u8>, value: f64) {
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_name(buf: &mut Vec<u8>, name: &str) {
    let mut bytes = [0_u8; 128];
    let raw = name.as_bytes();
    let len = raw.len().min(bytes.len());
    bytes[..len].copy_from_slice(&raw[..len]);
    write_i64(buf, len as i64);
    buf.extend_from_slice(&bytes);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
}

impl BinOp {
    fn code(self) -> i8 {
        match self {
            Self::Add => BINOP_ADD,
            Self::Sub => BINOP_SUB,
            Self::Mul => BINOP_MUL,
            Self::Div => BINOP_DIV,
            Self::Mod => BINOP_MOD,
            Self::Eq => BINOP_EQ,
            Self::Ne => BINOP_NE,
            Self::Lt => BINOP_LT,
            Self::Le => BINOP_LE,
            Self::Gt => BINOP_GT,
            Self::Ge => BINOP_GE,
            Self::And => BINOP_AND,
            Self::Or => BINOP_OR,
            Self::BitAnd => BINOP_BITAND,
            Self::BitOr => BINOP_BITOR,
            Self::BitXor => BINOP_BITXOR,
            Self::Shl => BINOP_SHL,
            Self::Shr => BINOP_SHR,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UnaryOp {
    Neg,
    Not,
}

impl UnaryOp {
    fn code(self) -> i8 {
        match self {
            Self::Neg => UNOP_NEG,
            Self::Not => UNOP_NOT,
        }
    }
}

#[derive(Clone, Debug)]
struct InstructionSpec {
    op: i8,
    dst: i64,
    src1: i64,
    src2: i64,
    imm_i64: i64,
    imm_f64: f64,
    label_id: i64,
    fn_id: i64,
    field_idx: i64,
    bin_op: i8,
    un_op: i8,
    name: &'static str,
    arg_count: i64,
}

#[derive(Clone, Debug)]
struct FunctionSpec {
    name: &'static str,
    instrs: Vec<InstructionSpec>,
    reg_count: i64,
    label_count: i64,
    param_regs: Vec<i64>,
}

fn write_instruction(buf: &mut Vec<u8>, instr: &InstructionSpec) {
    write_i8(buf, instr.op);
    buf.extend_from_slice(&[0; 7]);
    write_i64(buf, instr.dst);
    write_i64(buf, instr.src1);
    write_i64(buf, instr.src2);
    write_i64(buf, instr.imm_i64);
    write_f64(buf, instr.imm_f64);
    write_i64(buf, instr.label_id);
    write_i64(buf, instr.fn_id);
    write_i64(buf, instr.field_idx);
    write_i8(buf, instr.bin_op);
    buf.extend_from_slice(&[0; 7]);
    write_i8(buf, instr.un_op);
    buf.extend_from_slice(&[0; 7]);
    write_name(buf, instr.name);
    write_i64(buf, instr.arg_count);
}

fn write_function(buf: &mut Vec<u8>, func: &FunctionSpec) {
    write_name(buf, func.name);
    write_i64(buf, func.instrs.len() as i64);
    write_i64(buf, func.reg_count);
    write_i64(buf, func.label_count);
    write_i64(buf, func.param_regs.len() as i64);
    for i in 0..64 {
        let reg = func.param_regs.get(i).copied().unwrap_or(-1);
        write_i64(buf, reg);
    }
    for instr in &func.instrs {
        write_instruction(buf, instr);
    }
}

fn encode_module(functions: &[FunctionSpec]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&SOIR_MAGIC.to_le_bytes());
    out.push(SOIR_VERSION);
    out.extend_from_slice(&[0; 3]);
    write_i64(&mut out, functions.len() as i64);
    for func in functions {
        write_function(&mut out, func);
    }
    write_i64(&mut out, 0);
    out
}

pub fn eval_binop(lhs: i64, rhs: i64, op: BinOp) -> i64 {
    match op {
        BinOp::Add => lhs + rhs,
        BinOp::Sub => lhs - rhs,
        BinOp::Mul => lhs * rhs,
        BinOp::Div => {
            if rhs == 0 {
                0
            } else {
                lhs / rhs
            }
        }
        BinOp::Mod => {
            if rhs == 0 {
                0
            } else {
                lhs % rhs
            }
        }
        BinOp::Eq => i64::from(lhs == rhs),
        BinOp::Ne => i64::from(lhs != rhs),
        BinOp::Lt => i64::from(lhs < rhs),
        BinOp::Le => i64::from(lhs <= rhs),
        BinOp::Gt => i64::from(lhs > rhs),
        BinOp::Ge => i64::from(lhs >= rhs),
        BinOp::And => i64::from(lhs != 0 && rhs != 0),
        BinOp::Or => i64::from(lhs != 0 || rhs != 0),
        BinOp::BitAnd => lhs & rhs,
        BinOp::BitOr => lhs | rhs,
        BinOp::BitXor => lhs ^ rhs,
        BinOp::Shl => lhs << rhs,
        BinOp::Shr => lhs >> rhs,
    }
}

pub fn eval_unop(src: i64, op: UnaryOp) -> i64 {
    match op {
        UnaryOp::Neg => -src,
        UnaryOp::Not => i64::from(src == 0),
    }
}

pub fn straight_line_program(lhs: i64, rhs: i64, op: BinOp) -> Vec<u8> {
    encode_module(&[FunctionSpec {
        name: "main",
        instrs: vec![
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 0,
                src1: -1,
                src2: -1,
                imm_i64: lhs,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 1,
                src1: -1,
                src2: -1,
                imm_i64: rhs,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_BINOP,
                dst: 2,
                src1: 0,
                src2: 1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: op.code(),
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_RETURN,
                dst: -1,
                src1: 2,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
        ],
        reg_count: 3,
        label_count: 0,
        param_regs: Vec::new(),
    }])
}

pub fn unary_program(src: i64, op: UnaryOp) -> Vec<u8> {
    encode_module(&[FunctionSpec {
        name: "main",
        instrs: vec![
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 0,
                src1: -1,
                src2: -1,
                imm_i64: src,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_UNARYOP,
                dst: 1,
                src1: 0,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: op.code(),
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_RETURN,
                dst: -1,
                src1: 1,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
        ],
        reg_count: 2,
        label_count: 0,
        param_regs: Vec::new(),
    }])
}

pub fn branch_program(lhs: i64, rhs: i64, pred: BinOp, then_value: i64, else_value: i64) -> Vec<u8> {
    encode_module(&[FunctionSpec {
        name: "main",
        instrs: vec![
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 0,
                src1: -1,
                src2: -1,
                imm_i64: lhs,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 1,
                src1: -1,
                src2: -1,
                imm_i64: rhs,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_BINOP,
                dst: 2,
                src1: 0,
                src2: 1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: pred.code(),
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_BRANCH_TRUE,
                dst: -1,
                src1: 2,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: 0,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 3,
                src1: -1,
                src2: -1,
                imm_i64: else_value,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_JUMP,
                dst: -1,
                src1: -1,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: 1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LABEL,
                dst: -1,
                src1: -1,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: 0,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LOAD_IMM,
                dst: 3,
                src1: -1,
                src2: -1,
                imm_i64: then_value,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_LABEL,
                dst: -1,
                src1: -1,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: 1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
            InstructionSpec {
                op: OP_RETURN,
                dst: -1,
                src1: 3,
                src2: -1,
                imm_i64: 0,
                imm_f64: 0.0,
                label_id: -1,
                fn_id: -1,
                field_idx: -1,
                bin_op: 0,
                un_op: 0,
                name: "",
                arg_count: 0,
            },
        ],
        reg_count: 4,
        label_count: 2,
        param_regs: Vec::new(),
    }])
}

pub fn tests_dir() -> &'static Path {
    TESTS_DIR
        .get_or_init(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests"))
        .as_path()
}

pub fn fixture_manifest() -> &'static FixtureManifest {
    FIXTURE_MANIFEST.get_or_init(|| {
        let path = tests_dir().join("fixtures_manifest.v1.json");
        let raw = fs::read_to_string(&path).unwrap_or_else(|err| panic!("read fixture manifest {}: {err}", path.display()));
        let manifest: FixtureManifest =
            serde_json::from_str(&raw).unwrap_or_else(|err| panic!("parse fixture manifest {}: {err}", path.display()));
        assert_eq!(manifest.schema, "sounio.poseidon.fixtures.v1");
        manifest
    })
}

pub fn fixture_by_id(id: &str) -> FixtureSpec {
    fixture_manifest()
        .fixtures
        .iter()
        .find(|fixture| fixture.id == id)
        .cloned()
        .unwrap_or_else(|| panic!("missing fixture id {id}"))
}

pub fn fixtures_by_kind(kind: &str) -> Vec<FixtureSpec> {
    fixture_manifest()
        .fixtures
        .iter()
        .filter(|fixture| fixture.kind == kind)
        .cloned()
        .collect()
}

pub fn fixtures_with_tag(tag: &str) -> Vec<FixtureSpec> {
    fixture_manifest()
        .fixtures
        .iter()
        .filter(|fixture| fixture.tags.iter().any(|entry| entry == tag))
        .cloned()
        .collect()
}

pub fn fixture_path(spec: &FixtureSpec) -> PathBuf {
    tests_dir().join(&spec.filename)
}

pub fn read_fixture_bytes(spec: &FixtureSpec) -> Vec<u8> {
    let path = fixture_path(spec);
    fs::read(&path).unwrap_or_else(|err| panic!("read fixture {}: {err}", path.display()))
}

pub fn bench_fixtures() -> Vec<(String, Vec<u8>)> {
    fixtures_with_tag("bench")
        .into_iter()
        .map(|fixture| {
            let bytes = read_fixture_bytes(&fixture);
            (fixture.id, bytes)
        })
        .collect()
}
