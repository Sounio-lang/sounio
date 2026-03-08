#!/usr/bin/env python3
"""Generate canonical Poseidon SOIR fixtures and manifest."""

from __future__ import annotations

import json
import struct
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
FIXTURE_DIR = SCRIPT_DIR / "fixtures"
MANIFEST_PATH = SCRIPT_DIR / "fixtures_manifest.v1.json"

SOIR_MAGIC = 0x52494F53
SOIR_VERSION = 1
NAME_BUF_SIZE = 128
MAX_FUNCTIONS = 64
MAX_INSTRUCTIONS = 2048
MAX_PARAMS = 64
MAX_STRING_TABLE_ENTRIES = 4096
DEFAULT_MAX_STEPS = 10000

FUNCTION_HEADER_SIZE = 136 + 32 + (8 * 64)
INSTRUCTION_SIZE = 232

OP_LOAD_IMM = 0
OP_LOAD_BOOL = 2
OP_COPY = 4
OP_BINOP = 5
OP_UNARYOP = 6
OP_CALL = 7
OP_RETURN = 8
OP_JUMP = 9
OP_BRANCH_TRUE = 10
OP_BRANCH_FALSE = 11
OP_LABEL = 17
OP_NOP = 18

BINOP_ADD = 0
BINOP_SUB = 1
BINOP_MUL = 2
BINOP_DIV = 3
BINOP_MOD = 4
BINOP_EQ = 5
BINOP_NE = 6
BINOP_LT = 7
BINOP_LE = 8
BINOP_GT = 9
BINOP_GE = 10
BINOP_AND = 11
BINOP_OR = 12
BINOP_BITAND = 13
BINOP_BITOR = 14
BINOP_BITXOR = 15
BINOP_SHL = 16
BINOP_SHR = 17

UNOP_NEG = 0
UNOP_NOT = 1

ALLOWED_KINDS = {"positive", "runtime_contract", "loader_invalid"}
ALLOWED_TAGS = {"bench", "c_api", "c_cli", "fuzz_seed", "proptest_seed", "rust_api"}

REQUIRED_CORE_POSITIVES = {
    "return_const",
    "copy_chain",
    "load_bool_true",
    "unary_neg",
    "unary_not",
    "jump",
    "branch_true_taken",
    "branch_false_taken",
    "nop_passthrough",
    "call_user_single",
    "call_user_nested",
    "call_builtin_print_int",
    "loop_sum",
}

REQUIRED_BINARY_POSITIVES = {
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "eq",
    "ne",
    "lt",
    "le",
    "gt",
    "ge",
    "and",
    "or",
    "bitand",
    "bitor",
    "bitxor",
    "shl",
    "shr",
}

REQUIRED_RUNTIME_CONTRACTS = {
    "missing_main",
    "timeout_loop",
    "div_zero_is_zero",
    "mod_zero_is_zero",
    "invalid_fn_call_returns_zero",
    "unknown_opcode_skips",
}

REQUIRED_LOADER_INVALID = {
    "bad_magic",
    "unsupported_version",
    "truncated_header",
    "truncated_function_record",
    "truncated_instruction_record",
    "negative_fn_count",
    "fn_count_overflow",
    "instr_count_overflow",
    "param_count_overflow",
    "name_len_overflow",
    "string_count_overflow",
}

REQUIRED_BENCH_FIXTURES = {
    "add",
    "branch_true_taken",
    "call_user_single",
    "loop_sum",
    "return_const",
}


def instr(
    op,
    dst=-1,
    src1=-1,
    src2=-1,
    imm_i64=0,
    imm_f64=0.0,
    label_id=-1,
    fn_id=-1,
    field_idx=-1,
    bin_op=0,
    un_op=0,
    name="",
    arg_count=0,
):
    return {
        "op": op,
        "dst": dst,
        "src1": src1,
        "src2": src2,
        "imm_i64": imm_i64,
        "imm_f64": imm_f64,
        "label_id": label_id,
        "fn_id": fn_id,
        "field_idx": field_idx,
        "bin_op": bin_op,
        "un_op": un_op,
        "name": name,
        "arg_count": arg_count,
    }


def function(name, instrs, reg_count, label_count=0, param_regs=None):
    return {
        "name": name,
        "instrs": instrs,
        "reg_count": reg_count,
        "label_count": label_count,
        "param_regs": list(param_regs or []),
    }


def write_i64(buf, value):
    buf.extend(struct.pack("<q", int(value)))


def write_i8(buf, value):
    buf.extend(struct.pack("<b", int(value)))


def write_f64(buf, value):
    buf.extend(struct.pack("<d", float(value)))


def write_name(buf, name):
    raw = name.encode("utf-8")
    if len(raw) > NAME_BUF_SIZE:
        raise ValueError(f"name too long for SOIR Name buffer: {name!r}")
    write_i64(buf, len(raw))
    buf.extend(raw)
    buf.extend(b"\x00" * (NAME_BUF_SIZE - len(raw)))


def write_instruction(buf, spec):
    write_i8(buf, spec["op"])
    buf.extend(b"\x00" * 7)
    write_i64(buf, spec["dst"])
    write_i64(buf, spec["src1"])
    write_i64(buf, spec["src2"])
    write_i64(buf, spec["imm_i64"])
    write_f64(buf, spec["imm_f64"])
    write_i64(buf, spec["label_id"])
    write_i64(buf, spec["fn_id"])
    write_i64(buf, spec["field_idx"])
    write_i8(buf, spec["bin_op"])
    buf.extend(b"\x00" * 7)
    write_i8(buf, spec["un_op"])
    buf.extend(b"\x00" * 7)
    write_name(buf, spec["name"])
    write_i64(buf, spec["arg_count"])


def write_function(buf, spec):
    write_name(buf, spec["name"])
    write_i64(buf, len(spec["instrs"]))
    write_i64(buf, spec["reg_count"])
    write_i64(buf, spec["label_count"])
    write_i64(buf, len(spec["param_regs"]))
    for idx in range(MAX_PARAMS):
        value = spec["param_regs"][idx] if idx < len(spec["param_regs"]) else -1
        write_i64(buf, value)
    for instruction in spec["instrs"]:
        write_instruction(buf, instruction)


def encode_module(functions, string_table=()):
    buf = bytearray()
    buf.extend(struct.pack("<I", SOIR_MAGIC))
    buf.extend(struct.pack("<B", SOIR_VERSION))
    buf.extend(b"\x00" * 3)
    write_i64(buf, len(functions))
    for spec in functions:
        write_function(buf, spec)
    write_i64(buf, len(string_table))
    for entry in string_table:
        write_name(buf, entry)
    return bytes(buf)


def make_binary_fixture(bin_op, lhs, rhs):
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=lhs),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=rhs),
                    instr(OP_BINOP, dst=2, src1=0, src2=1, bin_op=bin_op),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
            )
        ]
    )


def make_unary_fixture(un_op, src):
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=src),
                    instr(OP_UNARYOP, dst=1, src1=0, un_op=un_op),
                    instr(OP_RETURN, src1=1),
                ],
                reg_count=2,
            )
        ]
    )


def make_return_const_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=42),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def make_copy_chain_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=42),
                    instr(OP_COPY, dst=1, src1=0),
                    instr(OP_COPY, dst=2, src1=1),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
            )
        ]
    )


def make_load_bool_true_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_BOOL, dst=0, imm_i64=1),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def make_jump_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_JUMP, label_id=0),
                    instr(OP_LOAD_IMM, dst=0, imm_i64=7),
                    instr(OP_LABEL, label_id=0),
                    instr(OP_LOAD_IMM, dst=0, imm_i64=42),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
                label_count=1,
            )
        ]
    )


def make_branch_true_taken_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_BOOL, dst=0, imm_i64=1),
                    instr(OP_BRANCH_TRUE, src1=0, label_id=0),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=0),
                    instr(OP_RETURN, src1=1),
                    instr(OP_LABEL, label_id=0),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=42),
                    instr(OP_RETURN, src1=1),
                ],
                reg_count=2,
                label_count=1,
            )
        ]
    )


def make_branch_false_taken_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_BOOL, dst=0, imm_i64=0),
                    instr(OP_BRANCH_FALSE, src1=0, label_id=0),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=0),
                    instr(OP_RETURN, src1=1),
                    instr(OP_LABEL, label_id=0),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=42),
                    instr(OP_RETURN, src1=1),
                ],
                reg_count=2,
                label_count=1,
            )
        ]
    )


def make_nop_passthrough_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=42),
                    instr(OP_NOP),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def make_call_user_single_fixture():
    return encode_module(
        [
            function(
                "add",
                [
                    instr(OP_BINOP, dst=2, src1=0, src2=1, bin_op=BINOP_ADD),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
                param_regs=[0, 1],
            ),
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=10),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=32),
                    instr(OP_CALL, dst=2, src1=0, src2=1, fn_id=0, name="add", arg_count=2),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
            ),
        ]
    )


def make_call_user_nested_fixture():
    return encode_module(
        [
            function(
                "add",
                [
                    instr(OP_BINOP, dst=2, src1=0, src2=1, bin_op=BINOP_ADD),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
                param_regs=[0, 1],
            ),
            function(
                "wrap",
                [
                    instr(OP_LOAD_IMM, dst=1, imm_i64=1),
                    instr(OP_CALL, dst=2, src1=0, src2=1, fn_id=0, name="add", arg_count=2),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
                param_regs=[0],
            ),
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=41),
                    instr(OP_CALL, dst=1, src1=0, fn_id=1, name="wrap", arg_count=1),
                    instr(OP_RETURN, src1=1),
                ],
                reg_count=2,
            ),
        ]
    )


def make_call_builtin_print_int_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=42),
                    instr(OP_CALL, dst=1, src1=0, name="print_int", arg_count=1),
                    instr(OP_LOAD_IMM, dst=2, imm_i64=0),
                    instr(OP_RETURN, src1=2),
                ],
                reg_count=3,
            )
        ]
    )


def make_loop_sum_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=0),
                    instr(OP_LOAD_IMM, dst=1, imm_i64=0),
                    instr(OP_LABEL, label_id=0),
                    instr(OP_LOAD_IMM, dst=2, imm_i64=5),
                    instr(OP_BINOP, dst=3, src1=1, src2=2, bin_op=BINOP_LT),
                    instr(OP_BRANCH_FALSE, src1=3, label_id=1),
                    instr(OP_BINOP, dst=4, src1=0, src2=1, bin_op=BINOP_ADD),
                    instr(OP_COPY, dst=0, src1=4),
                    instr(OP_LOAD_IMM, dst=5, imm_i64=1),
                    instr(OP_BINOP, dst=6, src1=1, src2=5, bin_op=BINOP_ADD),
                    instr(OP_COPY, dst=1, src1=6),
                    instr(OP_JUMP, label_id=0),
                    instr(OP_LABEL, label_id=1),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=7,
                label_count=2,
            )
        ]
    )


def make_missing_main_fixture():
    return encode_module(
        [
            function(
                "helper",
                [
                    instr(OP_LOAD_IMM, dst=0, imm_i64=7),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def make_invalid_fn_call_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(OP_CALL, dst=0, fn_id=99, name="missing", arg_count=0),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def make_unknown_opcode_fixture():
    return encode_module(
        [
            function(
                "main",
                [
                    instr(99),
                    instr(OP_LOAD_IMM, dst=0, imm_i64=7),
                    instr(OP_RETURN, src1=0),
                ],
                reg_count=1,
            )
        ]
    )


def encode_header_only(fn_count):
    buf = bytearray()
    buf.extend(struct.pack("<I", SOIR_MAGIC))
    buf.extend(struct.pack("<B", SOIR_VERSION))
    buf.extend(b"\x00" * 3)
    write_i64(buf, fn_count)
    return bytes(buf)


def invalid_loader_fixtures(valid_module_bytes):
    function_record_cut = 16 + 32
    instruction_record_cut = 16 + FUNCTION_HEADER_SIZE + 48

    name_len_overflow = bytearray(encode_header_only(1))
    write_i64(name_len_overflow, NAME_BUF_SIZE + 1)

    instr_count_overflow = bytearray(encode_header_only(1))
    write_name(instr_count_overflow, "main")
    write_i64(instr_count_overflow, MAX_INSTRUCTIONS + 1)
    write_i64(instr_count_overflow, 1)
    write_i64(instr_count_overflow, 0)
    write_i64(instr_count_overflow, 0)

    param_count_overflow = bytearray(encode_header_only(1))
    write_name(param_count_overflow, "main")
    write_i64(param_count_overflow, 1)
    write_i64(param_count_overflow, 1)
    write_i64(param_count_overflow, 0)
    write_i64(param_count_overflow, MAX_PARAMS + 1)

    return {
        "bad_magic": b"FAIL" + valid_module_bytes[4:],
        "unsupported_version": valid_module_bytes[:4] + b"\x02" + valid_module_bytes[5:],
        "truncated_header": valid_module_bytes[:7],
        "truncated_function_record": valid_module_bytes[:function_record_cut],
        "truncated_instruction_record": valid_module_bytes[:instruction_record_cut],
        "negative_fn_count": encode_header_only(-1),
        "fn_count_overflow": encode_header_only(MAX_FUNCTIONS + 1),
        "instr_count_overflow": bytes(instr_count_overflow),
        "param_count_overflow": bytes(param_count_overflow),
        "name_len_overflow": bytes(name_len_overflow),
        "string_count_overflow": encode_header_only(0) + struct.pack("<q", MAX_STRING_TABLE_ENTRIES + 1),
    }


def build_fixture_entries():
    fixtures = []

    def add_entry(fixture_id, kind, tags, expected_status, expected_exit_code, max_steps, payload):
        fixtures.append(
            {
                "id": fixture_id,
                "filename": f"fixtures/{fixture_id}.soir",
                "kind": kind,
                "tags": tags,
                "expected_status": expected_status,
                "expected_exit_code": expected_exit_code,
                "max_steps": max_steps,
                "payload": payload,
            }
        )

    def add_positive(fixture_id, payload, exit_code, extra_tags=None):
        tags = ["c_api", "rust_api", "fuzz_seed", "proptest_seed", "c_cli"]
        if extra_tags:
            tags.extend(extra_tags)
        add_entry(fixture_id, "positive", tags, "ok", exit_code, DEFAULT_MAX_STEPS, payload)

    binary_cases = [
        ("add", BINOP_ADD, 10, 32, 42),
        ("sub", BINOP_SUB, 50, 8, 42),
        ("mul", BINOP_MUL, 6, 7, 42),
        ("div", BINOP_DIV, 84, 2, 42),
        ("mod", BINOP_MOD, 86, 44, 42),
        ("eq", BINOP_EQ, 9, 9, 1),
        ("ne", BINOP_NE, 9, 8, 1),
        ("lt", BINOP_LT, 7, 9, 1),
        ("le", BINOP_LE, 9, 9, 1),
        ("gt", BINOP_GT, 9, 7, 1),
        ("ge", BINOP_GE, 9, 9, 1),
        ("and", BINOP_AND, 1, 1, 1),
        ("or", BINOP_OR, 0, 7, 1),
        ("bitand", BINOP_BITAND, 58, 47, 42),
        ("bitor", BINOP_BITOR, 40, 2, 42),
        ("bitxor", BINOP_BITXOR, 58, 16, 42),
        ("shl", BINOP_SHL, 21, 1, 42),
        ("shr", BINOP_SHR, 168, 2, 42),
    ]

    positive_payloads = {
        "return_const": make_return_const_fixture(),
        "copy_chain": make_copy_chain_fixture(),
        "load_bool_true": make_load_bool_true_fixture(),
        "unary_neg": make_unary_fixture(UNOP_NEG, -42),
        "unary_not": make_unary_fixture(UNOP_NOT, 0),
        "jump": make_jump_fixture(),
        "branch_true_taken": make_branch_true_taken_fixture(),
        "branch_false_taken": make_branch_false_taken_fixture(),
        "nop_passthrough": make_nop_passthrough_fixture(),
        "call_user_single": make_call_user_single_fixture(),
        "call_user_nested": make_call_user_nested_fixture(),
        "call_builtin_print_int": make_call_builtin_print_int_fixture(),
        "loop_sum": make_loop_sum_fixture(),
    }

    add_positive("return_const", positive_payloads["return_const"], 42, extra_tags=["bench"])
    add_positive("copy_chain", positive_payloads["copy_chain"], 42)
    add_positive("load_bool_true", positive_payloads["load_bool_true"], 1)
    add_positive("unary_neg", positive_payloads["unary_neg"], 42)
    add_positive("unary_not", positive_payloads["unary_not"], 1)
    add_positive("jump", positive_payloads["jump"], 42)
    add_positive("branch_true_taken", positive_payloads["branch_true_taken"], 42, extra_tags=["bench"])
    add_positive("branch_false_taken", positive_payloads["branch_false_taken"], 42)
    add_positive("nop_passthrough", positive_payloads["nop_passthrough"], 42)
    add_positive("call_user_single", positive_payloads["call_user_single"], 42, extra_tags=["bench"])
    add_positive("call_user_nested", positive_payloads["call_user_nested"], 42)
    add_entry(
        "call_builtin_print_int",
        "positive",
        ["c_api", "rust_api", "fuzz_seed"],
        "ok",
        0,
        DEFAULT_MAX_STEPS,
        positive_payloads["call_builtin_print_int"],
    )
    add_positive("loop_sum", positive_payloads["loop_sum"], 10, extra_tags=["bench"])

    for fixture_id, bin_op, lhs, rhs, exit_code in binary_cases:
        extra_tags = ["bench"] if fixture_id == "add" else None
        add_positive(fixture_id, make_binary_fixture(bin_op, lhs, rhs), exit_code, extra_tags=extra_tags)

    runtime_cases = {
        "missing_main": ("no_main_function", None, DEFAULT_MAX_STEPS, make_missing_main_fixture()),
        "timeout_loop": ("execution_timeout", None, 3, make_loop_sum_fixture()),
        "div_zero_is_zero": ("ok", 0, DEFAULT_MAX_STEPS, make_binary_fixture(BINOP_DIV, 9, 0)),
        "mod_zero_is_zero": ("ok", 0, DEFAULT_MAX_STEPS, make_binary_fixture(BINOP_MOD, 9, 0)),
        "invalid_fn_call_returns_zero": ("ok", 0, DEFAULT_MAX_STEPS, make_invalid_fn_call_fixture()),
        "unknown_opcode_skips": ("ok", 7, DEFAULT_MAX_STEPS, make_unknown_opcode_fixture()),
    }
    for fixture_id, (status, exit_code, max_steps, payload) in runtime_cases.items():
        add_entry(fixture_id, "runtime_contract", ["c_api", "rust_api", "fuzz_seed"], status, exit_code, max_steps, payload)

    loader_cases = invalid_loader_fixtures(positive_payloads["return_const"])
    for fixture_id, payload in loader_cases.items():
        expected_status = "unsupported_version" if fixture_id == "unsupported_version" else "invalid_bytecode"
        add_entry(fixture_id, "loader_invalid", ["fuzz_seed"], expected_status, None, None, payload)

    fixtures.sort(key=lambda entry: entry["id"])
    return fixtures


def validate_fixture_entries(fixtures):
    ids = [entry["id"] for entry in fixtures]
    filenames = [entry["filename"] for entry in fixtures]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate fixture ids generated")
    if len(filenames) != len(set(filenames)):
        raise ValueError("duplicate fixture filenames generated")

    by_kind = {}
    bench_ids = set()
    cli_positive_ids = set()

    for entry in fixtures:
        kind = entry["kind"]
        tags = set(entry["tags"])
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported fixture kind {kind!r} for {entry['id']}")
        unknown_tags = tags - ALLOWED_TAGS
        if unknown_tags:
            raise ValueError(f"unsupported fixture tags {sorted(unknown_tags)!r} for {entry['id']}")
        by_kind.setdefault(kind, set()).add(entry["id"])

        if "bench" in tags:
            bench_ids.add(entry["id"])
        if kind == "positive" and "c_cli" in tags:
            cli_positive_ids.add(entry["id"])

        if kind == "loader_invalid":
            if entry["expected_exit_code"] is not None:
                raise ValueError(f"loader-invalid fixture {entry['id']} must not declare an exit code")
            if "fuzz_seed" not in tags:
                raise ValueError(f"loader-invalid fixture {entry['id']} must be tagged fuzz_seed")
        if kind == "runtime_contract" and "c_cli" in tags:
            raise ValueError(f"runtime-contract fixture {entry['id']} must not be tagged c_cli")

    expected_positive_ids = REQUIRED_CORE_POSITIVES | REQUIRED_BINARY_POSITIVES
    expected_cli_ids = expected_positive_ids - {"call_builtin_print_int"}

    if by_kind.get("positive", set()) != expected_positive_ids:
        raise ValueError(
            "positive fixture set drifted: "
            f"expected {sorted(expected_positive_ids)!r}, got {sorted(by_kind.get('positive', set()))!r}"
        )
    if by_kind.get("runtime_contract", set()) != REQUIRED_RUNTIME_CONTRACTS:
        raise ValueError(
            "runtime-contract fixture set drifted: "
            f"expected {sorted(REQUIRED_RUNTIME_CONTRACTS)!r}, got {sorted(by_kind.get('runtime_contract', set()))!r}"
        )
    if by_kind.get("loader_invalid", set()) != REQUIRED_LOADER_INVALID:
        raise ValueError(
            "loader-invalid fixture set drifted: "
            f"expected {sorted(REQUIRED_LOADER_INVALID)!r}, got {sorted(by_kind.get('loader_invalid', set()))!r}"
        )
    if bench_ids != REQUIRED_BENCH_FIXTURES:
        raise ValueError(
            f"benchmark fixture set drifted: expected {sorted(REQUIRED_BENCH_FIXTURES)!r}, got {sorted(bench_ids)!r}"
        )
    if cli_positive_ids != expected_cli_ids:
        raise ValueError(
            f"CLI fixture set drifted: expected {sorted(expected_cli_ids)!r}, got {sorted(cli_positive_ids)!r}"
        )


def write_outputs(fixtures):
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    expected_names = set()
    manifest_fixtures = []

    for entry in fixtures:
        filename = entry["filename"]
        payload = entry["payload"]
        path = SCRIPT_DIR / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        expected_names.add(path.name)

        manifest_entry = {
            "id": entry["id"],
            "filename": entry["filename"],
            "kind": entry["kind"],
            "tags": entry["tags"],
            "expected_status": entry["expected_status"],
            "expected_exit_code": entry["expected_exit_code"],
            "max_steps": entry["max_steps"],
        }
        manifest_fixtures.append(manifest_entry)

    for stale in FIXTURE_DIR.glob("*.soir"):
        if stale.name not in expected_names:
            stale.unlink()

    payload = {
        "schema": "sounio.poseidon.fixtures.v1",
        "fixtures": manifest_fixtures,
    }
    MANIFEST_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    fixtures = build_fixture_entries()
    validate_fixture_entries(fixtures)
    write_outputs(fixtures)
    print(f"Generated {len(fixtures)} fixtures in {FIXTURE_DIR.name}/ and {MANIFEST_PATH.name}")


if __name__ == "__main__":
    main()
