#!/usr/bin/env python3
"""Independent verifier for E2B finite v1 Source -> ENIR CFG lowering."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CASES = [
    "v1_loop",
    "v1_if_both",
    "v1_i6",
    "v1_highreg",
    "v1e_frail",
    "v1e_emov_negzero",
    "v1_arith_high",
    "v1_branch_high",
]
EXPECTED_OBSERVATIONS = {
    "v1_loop": 1,
    "v1_if_both": 2,
    "v1_i6": 1,
    "v1_highreg": 1,
    "v1e_frail": 1,
    "v1e_emov_negzero": 2,
    "v1_arith_high": 1,
    "v1_branch_high": 2,
}
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E2BError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2BError(message)


def load_e2a(root: Path):
    path = root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py"
    spec = importlib.util.spec_from_file_location("e2a_verify", path)
    if spec is None or spec.loader is None:
        fail("cannot load E2A independent numeric oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if check and result.returncode != 0:
        fail(
            f"command failed rc={result.returncode}: {' '.join(command)}\n"
            f"{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}"
        )
    return result


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}_src\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"source builder not found: {name}_src")
    start, depth, index = match.end(), 1, match.end()
    in_string = escaped = False
    while index < len(source):
        ch = source[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
        index += 1
    fail(f"unterminated source builder: {name}_src")
    raise AssertionError


def extract_sources(corpus: Path) -> dict[str, str]:
    text = corpus.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for name in CASES:
        body = function_body(text, name)
        encoded = re.findall(r'add_line\s*\(&s,\s*"([^"\\]*(?:\\.[^"\\]*)*)",\s*&n\)', body)
        if not encoded:
            fail(f"source builder has no lines: {name}")
        lines = [bytes(line, "utf-8").decode("unicode_escape") for line in encoded]
        source = "\n".join(lines) + "\n"
        header = re.fullmatch(rf"epistemic fn {re.escape(name)}\(\) \{{", lines[0])
        if not header or lines[-1] != "}":
            fail(f"source identity mismatch for {name}")
        if len(source.encode("ascii")) > 256:
            fail(f"source exceeds EISA Str cap: {name}")
        result[name] = source
    if sum(line.startswith("gate ") for source in result.values() for line in source.splitlines()) != 11:
        fail("E2B source manifest is not exactly 8 programs / 11 gates")
    return result


@dataclass
class ExpectedOp:
    opcode: int
    result: int = -1
    type_id: int = -1
    a: int = -1
    b: int = -1
    target: int = -1
    policy: int = -1
    slot: int = -1
    line: int = 0


@dataclass
class ExpectedModule:
    name: str
    ops: list[ExpectedOp]
    values: list[tuple[int, int]]
    provenance: list[tuple[int, int]]
    observations: int


def independently_lower(source: str, e2a) -> ExpectedModule:
    lines = source.splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0].strip())
    if not header:
        fail("independent v1 parser rejected header")
    symbols: list[tuple[str, int]] = []
    slots: dict[str, int] = {}
    controls: list[dict[str, int]] = []
    ops: list[ExpectedOp] = []
    values: list[tuple[int, int]] = []
    provenance: list[tuple[int, int]] = []
    observations = 0
    root_closed = False

    def lookup(name: str) -> int:
        for symbol, value in reversed(symbols):
            if symbol == name:
                return value
        return -1

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, known: int = 0, bits: int = 0) -> int:
        value = len(values)
        values.append((known, bits))
        provenance.append((line, opcode))
        ops.append(ExpectedOp(opcode, value, 0, a, b, line=line))
        return value

    def operand(token: str, line: int) -> int:
        token = token.strip()
        value = lookup(token)
        if value >= 0:
            return value
        number = e2a.decimal_like_source(token)
        return emit(0, line, known=1, bits=e2a.fbits(number))

    for line_number, raw in enumerate(lines[1:], 2):
        line = raw.strip()
        if root_closed:
            if line:
                fail(f"content after root close at line {line_number}")
            continue
        if line == "}":
            if controls:
                control = controls.pop()
                del symbols[control["symbols"] :]
                if control["kind"] == 1:
                    if len(ops) != control["body_start"]:
                        fail(f"non-empty E2B while body closed at line {line_number}")
                    ops.append(ExpectedOp(10, target=control["test"], line=line_number))
                    ops[control["patch"]].target = len(ops)
                else:
                    if len(values) != control["values"]:
                        fail(f"conditional E2B definition requires block arguments at line {line_number}")
                    if observations != control["observations"]:
                        fail(f"conditional E2B gate requires path-sensitive observations at line {line_number}")
                    ops[control["patch"]].target = len(ops)
            else:
                root_closed = True
            continue
        if line.startswith("fuel "):
            fail(f"fuel requires E2C receipt semantics at line {line_number}")
        while_match = re.fullmatch(r"while ([A-Za-z][A-Za-z0-9_]*) != 0\.0 \{", line)
        if while_match:
            condition = lookup(while_match.group(1))
            if condition < 0:
                fail(f"undefined E2B while condition at line {line_number}")
            test = len(ops)
            ops.append(ExpectedOp(11, a=condition, target=-1, policy=0, line=line_number))
            controls.append({"kind": 1, "patch": test, "test": test, "body_start": len(ops), "symbols": len(symbols), "values": len(values), "observations": observations})
            continue
        if_match = re.fullmatch(r"if ([A-Za-z][A-Za-z0-9_]*) < 0\.0 \{", line)
        if if_match:
            condition = lookup(if_match.group(1))
            if condition < 0:
                fail(f"undefined independent if condition at line {line_number}")
            ops.append(ExpectedOp(12, a=condition, target=len(ops) + 2, policy=0, line=line_number))
            ops.append(ExpectedOp(10, target=-1, line=line_number))
            controls.append({"kind": 2, "patch": len(ops) - 1, "test": -1, "body_start": len(ops), "symbols": len(symbols), "values": len(values), "observations": observations})
            continue
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        if let_match:
            name, expr = let_match.groups()
            if lookup(name) >= 0:
                fail(f"duplicate independent symbol at line {line_number}")
            sqrt_match = re.fullmatch(r"sqrt\(([^()]+)\)", expr)
            binary_match = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expr)
            if sqrt_match:
                result = emit(6, line_number, a=operand(sqrt_match.group(1), line_number))
            elif binary_match:
                left, operator, right = binary_match.groups()
                result = emit({"+": 2, "-": 3, "*": 4, "/": 5}[operator], line_number, operand(left, line_number), operand(right, line_number))
            elif lookup(expr.strip()) >= 0:
                result = emit(9, line_number, a=lookup(expr.strip()))
            else:
                result = emit(0, line_number, known=1, bits=e2a.fbits(e2a.decimal_like_source(expr.strip())))
            symbols.append((name, result))
            continue
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if gate_match:
            value = lookup(gate_match.group(1))
            if value < 0:
                fail(f"undefined independent gate at line {line_number}")
            ops.append(ExpectedOp(7, a=value, policy=0, line=line_number))
            observations += 1
            continue
        store_match = re.fullmatch(r"store \[([A-Za-z][A-Za-z0-9_]*)\] <- ([A-Za-z][A-Za-z0-9_]*)", line)
        if store_match:
            slot, name = store_match.groups()
            value = lookup(name)
            if value < 0:
                fail(f"undefined independent store at line {line_number}")
            slot_id = slots.setdefault(slot, len(slots))
            ops.append(ExpectedOp(8, a=value, slot=slot_id, line=line_number))
            continue
        fail(f"unsupported independent source line {line_number}: {line}")

    if not root_closed or controls or observations != EXPECTED_OBSERVATIONS[header.group(1)]:
        fail("independent structured source did not close or observation count drifted")
    ops.append(ExpectedOp(13, line=len(lines) + 1))
    return ExpectedModule(header.group(1), ops, values, provenance, observations)


def verify_lowered(parsed: dict[str, object], expected: ExpectedModule) -> None:
    grouped = parsed["grouped"]
    if grouped["enir"] != [["enir", "1", "2", expected.name, "1"]]:
        fail(f"v1 ENIR header mismatch for {expected.name}")
    if grouped["type"] != [["type", "0", "4", "1", "1", "1", "1", "1"]]:
        fail(f"v1 ENIR type profile mismatch for {expected.name}")
    if grouped["policy"] != [["policy", "0", "eisa_10_100_count_only", "0", "0"]]:
        fail(f"v1 ENIR policy mismatch for {expected.name}")
    rows = grouped["op"]
    if len(rows) != len(expected.ops):
        fail(f"op count mismatch for {expected.name}: {len(rows)} != {len(expected.ops)}")
    for index, (row, op) in enumerate(zip(rows, expected.ops)):
        actual = list(map(int, row[1:]))
        wanted = [index, op.opcode, op.result, op.type_id, op.a, op.b, op.target, op.policy, op.slot, 1]
        if actual != wanted:
            fail(f"op mismatch for {expected.name} at {index}: {actual} != {wanted}")
    value_rows = grouped["value"]
    prov_rows = grouped["prov"]
    if len(value_rows) != len(expected.values) or len(prov_rows) != len(expected.provenance):
        fail(f"value/provenance count mismatch for {expected.name}")
    for index, ((known, bits), row) in enumerate(zip(expected.values, value_rows)):
        if int(row[1]) != index or int(row[2]) != 0 or int(row[3]) != known or int(row[4]) != bits or int(row[13]) != index:
            fail(f"value descriptor mismatch for {expected.name} value {index}")
    for index, ((line, opcode), row) in enumerate(zip(expected.provenance, prov_rows)):
        if list(map(int, (row[1], row[2], row[4], row[5], row[6]))) != [index, line, opcode, -1, -1]:
            fail(f"provenance mismatch for {expected.name} value {index}")
    observations = grouped["obs"]
    if len(observations) != expected.observations:
        fail(f"observation count mismatch for {expected.name}")
    for ordinal, row in enumerate(observations):
        if row != ["obs", str(ordinal), expected.name, str(ordinal), "0"]:
            fail(f"observation descriptor mismatch for {expected.name}: {row}")
    footer = grouped["end"]
    wanted_footer = [["end", "1", str(len(expected.values)), str(len(expected.provenance)), "1", str(len(expected.ops)), str(expected.observations), "1"]]
    if footer != wanted_footer:
        fail(f"footer mismatch for {expected.name}: {footer}")


def parse_native(output: bytes) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    observations: list[dict[str, int]] = []
    controls: list[dict[str, int]] = []
    for line in output.decode("ascii").splitlines():
        if line.startswith("enir-exec|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = (
                "site", "value_id", "value_bits", "error0_bits", "error1_bits",
                "uncertainty_bits", "status", "gate_class", "branch_poisoned",
                "frail_branches", "source_span",
            )
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith("enir-control|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            controls.append({key: int(fields[key]) for key in ("site", "opcode", "condition", "target", "taken", "poisoned", "frail", "next")})
    return observations, controls


def independent_execute(parsed: dict[str, object], e2a) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    grouped = parsed["grouped"]
    values_desc = grouped["value"]
    provenance = grouped["prov"]
    ops = grouped["op"]
    runtime: dict[int, object] = {}
    observations: list[dict[str, int]] = []
    controls: list[dict[str, int]] = []
    branch_poisoned = 0
    frail_branches = 0
    pc = steps = 0
    while pc < len(ops):
        if steps >= len(ops) * 4:
            fail("independent CFG interpreter exceeded step bound")
        row = ops[pc]
        site, opcode, result, a, b, target = map(int, (row[1], row[2], row[3], row[5], row[6], row[7]))
        next_pc = pc + 1
        if opcode == 0:
            runtime[result] = e2a.Value(e2a.bits_float(int(values_desc[result][4])))
        elif opcode in {2, 3, 4, 5}:
            x, y = runtime[a], runtime[b]
            if x.poisoned or y.poisoned or (opcode == 5 and y.value == 0.0):
                runtime[result] = e2a.poison()
            elif opcode in {2, 3}:
                value, residual = e2a.two_sum(x.value, y.value if opcode == 2 else -y.value)
                error = e2a.dd_add((x.error_hi, x.error_lo), (y.error_hi, y.error_lo)) if opcode == 2 else e2a.dd_sub((x.error_hi, x.error_lo), (y.error_hi, y.error_lo))
                error = e2a.dd_add(error, (residual, 0.0))
                runtime[result] = e2a.Value(value, error[0], error[1], e2a.sqrt40(x.uncertainty**2 + y.uncertainty**2), 0)
            elif opcode == 4:
                value, residual = e2a.two_prod(x.value, y.value)
                error = e2a.dd_add(e2a.dd_add(e2a.dd_mul_f64((x.error_hi, x.error_lo), y.value), e2a.dd_mul_f64((y.error_hi, y.error_lo), x.value)), (residual, 0.0))
                uncertainty = e2a.sqrt40((y.value * x.uncertainty) ** 2 + (x.value * y.uncertainty) ** 2)
                runtime[result] = e2a.Value(value, error[0], error[1], uncertainty, 0)
            else:
                value = x.value / y.value
                product = e2a.two_prod(value, y.value)
                residual_dd = e2a.dd_sub((x.value, 0.0), product)
                residual = (residual_dd[0] + residual_dd[1]) / y.value
                error = e2a.dd_add(e2a.dd_sub(e2a.dd_mul_f64((x.error_hi, x.error_lo), 1.0 / y.value), e2a.dd_mul_f64((y.error_hi, y.error_lo), value / y.value)), (residual, 0.0))
                uncertainty = (1.0 / abs(y.value)) * e2a.sqrt40(x.uncertainty**2 + (value * y.uncertainty) ** 2)
                runtime[result] = e2a.Value(value, error[0], error[1], uncertainty, 0)
        elif opcode == 6:
            x = runtime[a]
            if x.poisoned:
                runtime[result] = e2a.poison()
            elif x.value <= 0.0:
                runtime[result] = e2a.Value()
            else:
                root = e2a.sqrt40(x.value)
                product = e2a.two_prod(root, root)
                residual_dd = e2a.dd_sub((x.value, 0.0), product)
                residual = (residual_dd[0] + residual_dd[1]) / (2.0 * root)
                error = e2a.dd_add(e2a.dd_mul_f64((x.error_hi, x.error_lo), 1.0 / (2.0 * root)), (residual, 0.0))
                runtime[result] = e2a.Value(root, error[0], error[1], x.uncertainty / (2.0 * root), 0)
        elif opcode == 9:
            runtime[result] = runtime[a]
        elif opcode == 7:
            value = runtime[a]
            observations.append({
                "site": site,
                "value_id": a,
                "value_bits": POISON_VALUE_BITS if value.poisoned else e2a.fbits(value.value),
                "error0_bits": e2a.fbits(value.error_hi),
                "error1_bits": e2a.fbits(value.error_lo),
                "uncertainty_bits": POISON_U_BITS if value.poisoned else e2a.fbits(value.uncertainty),
                "status": value.poisoned,
                "gate_class": 2 if branch_poisoned else e2a.gate_class(value),
                "branch_poisoned": branch_poisoned,
                "frail_branches": frail_branches,
                "source_span": int(provenance[a][2]),
            })
        elif opcode == 8:
            pass
        elif opcode == 10:
            next_pc = target
            controls.append({"site": site, "opcode": opcode, "condition": -1, "target": target, "taken": 1, "poisoned": 0, "frail": 0, "next": next_pc})
        elif opcode in {11, 12}:
            condition = runtime[a]
            taken = poisoned = frail = 0
            if condition.poisoned:
                branch_poisoned = poisoned = 1
            else:
                band = max(abs(condition.error_hi), condition.uncertainty)
                if band != 0.0 and abs(condition.value) <= band:
                    frail = 1
                    frail_branches += 1
                if (opcode == 11 and condition.value == 0.0) or (opcode == 12 and condition.value < 0.0):
                    taken = 1
                    next_pc = target
            controls.append({"site": site, "opcode": opcode, "condition": a, "target": target, "taken": taken, "poisoned": poisoned, "frail": frail, "next": next_pc})
        elif opcode == 13:
            steps += 1
            break
        else:
            fail(f"unsupported independent CFG opcode: {opcode}")
        pc = next_pc
        steps += 1
    return observations, controls


def parts_bits(token: str) -> int:
    match = re.fullmatch(r"s([01])e([0-9]+)m([0-9]+)", token)
    if not match:
        fail(f"invalid EVM parts token: {token}")
    sign, exponent, mantissa = map(int, match.groups())
    raw = (sign << 63) | (exponent << 52) | mantissa
    return raw - (1 << 64) if raw >= (1 << 63) else raw


def probe_bits(fields: dict[str, str], prefix: str) -> int:
    sign = int(fields[f"{prefix}_sign"])
    exponent = int(fields[f"{prefix}_exp"])
    mantissa = int(fields[f"{prefix}_mant"])
    raw = (sign << 63) | (exponent << 52) | mantissa
    return raw - (1 << 64) if raw >= (1 << 63) else raw


def parse_evm_oracle(path: Path) -> dict[str, dict[str, object]]:
    cases: dict[str, dict[str, object]] = {}
    current: str | None = None
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("e2b-case-begin|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            current = fields["name"]
            cases[current] = {"receipts": [], "begin": fields}
        elif line.startswith("eisa-receipt:") and current is not None:
            match = re.fullmatch(r"eisa-receipt: v=2 prog=[0-9]+ gate=[0-9]+ reg=e([0-9]+) val=(\S+) roundoff=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)", line)
            if not match:
                fail(f"invalid E2B EVM receipt: {line}")
            reg, value, error, uncertainty, poisoned, frail = match.groups()
            cases[current]["receipts"].append({
                "reg": int(reg),
                "value_bits": parts_bits(value),
                "error0_bits": parts_bits(error),
                "uncertainty_bits": parts_bits(uncertainty),
                "status": int(poisoned),
                "frail_branches": int(frail),
            })
        elif line.startswith("e2b-case-end|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            name = fields["name"]
            if current != name or name not in cases:
                fail("EVM case marker ordering mismatch")
            cases[name]["end"] = fields
            current = None
    if list(cases) != CASES:
        fail(f"EVM case manifest mismatch: {list(cases)}")
    for name, case in cases.items():
        end = case.get("end")
        receipts = case["receipts"]
        if not isinstance(end, dict) or int(end["rc"]) != 0 or len(receipts) != EXPECTED_OBSERVATIONS[name]:
            fail(f"EVM oracle case failed or receipt count drifted: {name}")
        if name == "v1e_emov_negzero":
            receipts[0]["value_bits"] = probe_bits(end, "probe0")
            receipts[1]["value_bits"] = probe_bits(end, "probe1")
    return cases


def compare_evm(name: str, native: list[dict[str, int]], oracle: dict[str, object]) -> None:
    receipts = oracle["receipts"]
    end = oracle["end"]
    assert isinstance(receipts, list) and isinstance(end, dict)
    for index, (actual, expected) in enumerate(zip(native, receipts)):
        combined_status = max(actual["status"], actual["branch_poisoned"])
        for key in ("value_bits", "error0_bits", "uncertainty_bits", "frail_branches"):
            if actual[key] != expected[key]:
                fail(f"EVM/ENIR mismatch {name} observation {index} field {key}: {actual[key]} != {expected[key]}")
        if combined_status != expected["status"]:
            fail(f"EVM/ENIR poison mismatch {name} observation {index}")
    if native and native[-1]["branch_poisoned"] != int(end["branch_poisoned"]):
        fail(f"final branch poison mismatch for {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    e2a = load_e2a(args.root)
    sources = extract_sources(args.corpus)
    evm = parse_evm_oracle(args.oracle)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_cases = []
    aggregate_opcodes: set[int] = set()
    aggregate_control: list[dict[str, int]] = []

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio"))
    for forbidden in ("enir_shadow_fixture(", "use eisa::", "eisax_encode(", "eisa_compile(", "eisa_compile_v1("):
        if forbidden in implementation:
            fail(f"forbidden fixture/oracle dependency in ENIR implementation: {forbidden}")

    for name in CASES:
        source_path = args.out_dir / f"{name}.eisa"
        artifact_path = args.out_dir / f"{name}.enir"
        source_path.write_text(sources[name], encoding="ascii")
        lowered = run([str(args.driver), "lower-v1", str(source_path)]).stdout
        artifact_path.write_bytes(lowered)
        parsed = e2a.parse_enir(lowered)
        expected = independently_lower(sources[name], e2a)
        verify_lowered(parsed, expected)
        native_output = run([str(args.driver), "run", str(artifact_path)]).stdout
        native, controls = parse_native(native_output)
        independent, independent_controls = independent_execute(parsed, e2a)
        if native != independent or controls != independent_controls:
            fail(f"native/independent CFG execution mismatch for {name}")
        if len(native) != EXPECTED_OBSERVATIONS[name]:
            fail(f"native observation count mismatch for {name}")
        compare_evm(name, native, evm[name])
        opcodes = {int(row[2]) for row in parsed["grouped"]["op"]}
        aggregate_opcodes.update(opcodes)
        aggregate_control.extend(controls)
        max_result = max((int(row[3]) for row in parsed["grouped"]["op"]), default=-1)
        receipt_cases.append({
            "name": name,
            "source_sha256": hashlib.sha256(sources[name].encode("ascii")).hexdigest(),
            "enir_sha256": hashlib.sha256(lowered).hexdigest(),
            "ops": len(parsed["grouped"]["op"]),
            "opcodes": sorted(opcodes),
            "observations": len(native),
            "control_events": len(controls),
            "max_result_id": max_result,
            "branch_poisoned": native[-1]["branch_poisoned"],
            "frail_branches": native[-1]["frail_branches"],
        })

    if not {10, 11, 12, 13}.issubset(aggregate_opcodes):
        fail(f"missing v1 control opcode coverage: {sorted(aggregate_opcodes)}")
    if not any(row["taken"] == 1 for row in aggregate_control) or not any(row["taken"] == 0 for row in aggregate_control):
        fail("control trace lacks taken/not-taken coverage")
    if not any(row["poisoned"] == 1 for row in aggregate_control):
        fail("control trace lacks poisoned-branch coverage")
    if not any(row["frail"] == 1 for row in aggregate_control):
        fail("control trace lacks frail-branch coverage")
    high_ids = {case["name"]: case["max_result_id"] for case in receipt_cases}
    if high_ids["v1_highreg"] < 20 or high_ids["v1_arith_high"] < 23 or high_ids["v1_branch_high"] < 20:
        fail(f"high value-ID coverage regressed: {high_ids}")

    original = sources["v1_loop"]
    tampered = original.replace("let three=3.0", "let three=4.0", 1)
    tamper_source = args.out_dir / "v1_loop_tamper.eisa"
    tamper_artifact = args.out_dir / "v1_loop_tamper.enir"
    tamper_source.write_text(tampered, encoding="ascii")
    tamper_bytes = run([str(args.driver), "lower-v1", str(tamper_source)]).stdout
    tamper_artifact.write_bytes(tamper_bytes)
    base_native, base_control = parse_native(run([str(args.driver), "run", str(args.out_dir / "v1_loop.enir")]).stdout)
    tamper_native, tamper_control = parse_native(run([str(args.driver), "run", str(tamper_artifact)]).stdout)
    if hashlib.sha256(tamper_bytes).digest() == hashlib.sha256((args.out_dir / "v1_loop.enir").read_bytes()).digest():
        fail("causal source tamper did not change ENIR bytes")
    if len(base_native) != 1 or len(tamper_native) != 1 or base_native[0]["value_bits"] == tamper_native[0]["value_bits"]:
        fail("causal source tamper did not change exactly the v1_loop value observation")
    if base_control != tamper_control:
        fail("data-only source tamper changed control trace")

    receipt = {
        "schema": "madaros-v2-e2b-enir-finite-cfg-receipt-v1",
        "scope": "E2B_v1_finite_cfg_FULL_not_E2_umbrella",
        "evidence_level": "R1_replay_and_three_way_differential",
        "programs": 8,
        "observations": 11,
        "cumulative_programs": 13,
        "cumulative_observations": 17,
        "source_to_enir": "compiler_owned_direct_no_eisax",
        "interpreter": "compiler_owned_independent_no_eisa_core",
        "opcodes": sorted(aggregate_opcodes),
        "control_coverage": {"taken": True, "not_taken": True, "poisoned": True, "frail": True},
        "high_value_ids": high_ids,
        "native_independent_exact": True,
        "evm_observable_exact": True,
        "causal_source_tamper": True,
        "shadow_fixture_used": False,
        "fallback": "none",
        "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "cases": receipt_cases,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("E2B_ENIR_V1_FINITE_CFG_VERIFY_PASS programs=8 observations=11 cumulative=13/30,17/39 control=taken,not-taken,poisoned,frail native_independent=exact evm_observable=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2BError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
        print(f"E2B_ENIR_V1_FINITE_CFG_VERIFY_FAIL: {exc}")
        raise SystemExit(1)
