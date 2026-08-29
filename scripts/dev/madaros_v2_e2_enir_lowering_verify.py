#!/usr/bin/env python3
"""Independent verifier for E2 direct source lowering and ENIR execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


CASES = [
    "golden_mul_src",
    "golden_add_src",
    "golden_sqrt_src",
    "golden_poison_src",
    "e5_cancellation_src",
]
OPCODE = {"+": 2, "-": 3, "*": 4, "/": 5}
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E2Error(ValueError):
    pass


def fail(message: str) -> None:
    raise E2Error(message)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if check and result.returncode != 0:
        fail(f"command failed rc={result.returncode}: {' '.join(command)}\n{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}")
    return result


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"function not found: {name}")
    start, depth, index, in_string, escaped = match.end(), 1, match.end(), False, False
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
    fail(f"unterminated function: {name}")
    raise AssertionError


def extract_sources(corpus: Path) -> list[tuple[str, str]]:
    text = corpus.read_text(encoding="utf-8")
    cases: list[tuple[str, str]] = []
    for builder in CASES:
        body = function_body(text, builder)
        lines = re.findall(r'add_line\s*\(&s,\s*"([^"\\]*(?:\\.[^"\\]*)*)",\s*&n\)', body)
        if not lines:
            fail(f"no source lines extracted from {builder}")
        decoded = [bytes(line, "utf-8").decode("unicode_escape") for line in lines]
        cases.append((builder.removesuffix("_src"), "\n".join(decoded) + "\n"))
    if sum(line.startswith("gate ") for _, source in cases for line in source.splitlines()) != 6:
        fail("source-derived E2 slice is not exactly 5 programs / 6 gates")
    return cases


def signed_u64(value: int) -> int:
    return value - (1 << 64) if value >= (1 << 63) else value


def fbits(value: float) -> int:
    return struct.unpack(">q", struct.pack(">d", value))[0]


def bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">q", bits))[0]


def decimal_like_source(token: str) -> float:
    negative = token.startswith("-")
    if negative:
        token = token[1:]
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", token):
        fail(f"unsupported numeric token in independent parser: {token}")
    integer, dot, fraction = token.partition(".")
    value = 0.0
    for ch in integer:
        value = value * 10.0 + float(ord(ch) - 48)
    if dot:
        for ch in fraction:
            value = value * 10.0 + float(ord(ch) - 48)
        value = value / (10.0 ** len(fraction))
    return -value if negative else value


def rolling_hash(raw: bytes) -> int:
    value = 14695981
    for byte in raw:
        value = (value * 257 + byte) % 1_000_000_007
    return value


def parse_enir(raw: bytes) -> dict[str, object]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail("ENIR artifact is not canonical LF text")
    lines = raw.decode("ascii").splitlines()
    rows = [line.split("|") for line in lines]
    expected_fields = {"enir": 5, "type": 8, "value": 14, "prov": 7, "policy": 5, "op": 11, "obs": 5, "end": 8}
    order = {name: index for index, name in enumerate(expected_fields)}
    prior = -1
    for number, row in enumerate(rows, 1):
        if not row or row[0] not in expected_fields or len(row) != expected_fields[row[0]] or any(field == "" for field in row):
            fail(f"malformed ENIR row {number}")
        current = order[row[0]]
        if current < prior:
            fail(f"non-canonical section order at row {number}")
        prior = current
    if rows[0][0] != "enir" or rows[-1][0] != "end":
        fail("missing ENIR header/footer")
    grouped = {tag: [row for row in rows if row[0] == tag] for tag in expected_fields}
    return {"raw": raw, "rows": rows, "grouped": grouped}


@dataclass
class ExpectedLowering:
    module_name: str
    ops: list[tuple[int, int, int, int, int, int]]
    values: list[tuple[int, int]]
    provenance_lines: list[int]
    observations: int


def independently_lower(source: str) -> ExpectedLowering:
    lines = source.splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0].strip())
    if not header or lines[-1].strip() != "}":
        fail("independent parser rejected source header/close")
    symbols: dict[str, int] = {}
    slots: dict[str, int] = {}
    ops: list[tuple[int, int, int, int, int, int]] = []
    values: list[tuple[int, int]] = []  # known, bits
    provenance: list[int] = []
    observations = 0

    def emit(opcode: int, a: int, b: int, line: int, known: int = 0, bits: int = 0) -> int:
        value_id = len(values)
        values.append((known, bits))
        provenance.append(line)
        ops.append((opcode, value_id, a, b, -1, line))
        return value_id

    def operand(token: str, line: int, origin: int) -> int:
        token = token.strip()
        if token in symbols:
            return symbols[token]
        try:
            number = decimal_like_source(token)
        except E2Error:
            fail(f"undefined/invalid operand at line {line}: {token}")
        return emit(0, -1, -1, line, 1, fbits(number))

    for line_number, raw_line in enumerate(lines[1:-1], 2):
        line = raw_line.strip()
        if line.startswith("let "):
            match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*) = (.+)", line)
            if not match or match.group(1) in symbols:
                fail(f"invalid independent let at line {line_number}")
            name, expr = match.groups()
            sqrt_match = re.fullmatch(r"sqrt\(([^()]+)\)", expr)
            binary_match = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expr)
            if sqrt_match:
                result = emit(6, operand(sqrt_match.group(1), line_number, 0), -1, line_number)
            elif binary_match:
                left, operator, right = binary_match.groups()
                result = emit(OPCODE[operator], operand(left, line_number, 0), operand(right, line_number, 1), line_number)
            elif expr in symbols:
                result = emit(9, symbols[expr], -1, line_number)
            else:
                result = emit(0, -1, -1, line_number, 1, fbits(decimal_like_source(expr)))
            symbols[name] = result
        elif line.startswith("gate "):
            name = line[5:].strip()
            if name not in symbols:
                fail(f"undefined gate at line {line_number}")
            ops.append((7, -1, symbols[name], -1, -1, line_number))
            observations += 1
        elif line.startswith("store "):
            match = re.fullmatch(r"store \[([A-Za-z][A-Za-z0-9_]*)\] <- ([A-Za-z][A-Za-z0-9_]*)", line)
            if not match or match.group(2) not in symbols:
                fail(f"invalid store at line {line_number}")
            slot, name = match.groups()
            slot_id = slots.setdefault(slot, len(slots))
            ops.append((8, -1, symbols[name], -1, slot_id, line_number))
        else:
            fail(f"unsupported source line {line_number}: {line}")
    return ExpectedLowering(header.group(1), ops, values, provenance, observations)


def verify_lowered(parsed: dict[str, object], expected: ExpectedLowering) -> None:
    grouped = parsed["grouped"]
    assert isinstance(grouped, dict)
    header = grouped["enir"][0]
    if header != ["enir", "1", "2", expected.module_name, "0"]:
        fail(f"wrong lowered header: {header}")
    if grouped["type"] != [["type", "0", "4", "1", "1", "1", "1", "0"]]:
        fail("lowered type is not EISA v0 DD64+GUM1")
    if grouped["policy"] != [["policy", "0", "eisa_10_100_count_only", "0", "0"]]:
        fail("lowered policy mismatch")
    values = grouped["value"]
    provenance = grouped["prov"]
    ops = grouped["op"]
    observations = grouped["obs"]
    if len(values) != len(expected.values) or len(provenance) != len(expected.values) or len(ops) != len(expected.ops):
        fail("lowered value/provenance/op counts differ from independent lowering")
    for index, ((known, bits), row) in enumerate(zip(expected.values, values)):
        if int(row[1]) != index or int(row[2]) != 0 or int(row[3]) != known or int(row[4]) != bits:
            fail(f"value mismatch at {index}: {row}")
    for index, (line, row) in enumerate(zip(expected.provenance_lines, provenance)):
        if int(row[1]) != index or int(row[2]) != line:
            fail(f"provenance mismatch at {index}: {row}")
    for index, (want, row) in enumerate(zip(expected.ops, ops)):
        opcode, result, a, b, slot, line = want
        got = (int(row[2]), int(row[3]), int(row[5]), int(row[6]), int(row[9]))
        if got != (opcode, result, a, b, slot):
            fail(f"op mismatch at {index}: expected={(opcode, result, a, b, slot)} got={got}")
        if result >= 0:
            prov_row = provenance[result]
            if int(prov_row[2]) != line or int(prov_row[4]) != opcode:
                fail(f"producer provenance mismatch for value {result}")
    if len(observations) != expected.observations:
        fail("observation-site count mismatch")
    for ordinal, row in enumerate(observations):
        if row != ["obs", str(ordinal), expected.module_name, str(ordinal), "0"]:
            fail(f"observation descriptor mismatch: {row}")
    footer = grouped["end"][0]
    expected_footer = ["end", "1", str(len(values)), str(len(provenance)), "1", str(len(ops)), str(len(observations)), "1"]
    if footer != expected_footer:
        fail(f"footer mismatch: {footer}")


@dataclass
class Value:
    value: float = 0.0
    error_hi: float = 0.0
    error_lo: float = 0.0
    uncertainty: float = 0.0
    poisoned: int = 0


def two_sum(a: float, b: float) -> tuple[float, float]:
    total = a + b
    virtual = total - a
    return total, (a - (total - virtual)) + (b - virtual)


def quick_two_sum(a: float, b: float) -> tuple[float, float]:
    total = a + b
    return total, b - (total - a)


def two_prod(a: float, b: float) -> tuple[float, float]:
    product = a * b
    c = 134217729.0 * a
    ah = c - (c - a)
    al = a - ah
    c = 134217729.0 * b
    bh = c - (c - b)
    bl = b - bh
    return product, ((ah * bh - product) + ah * bl + al * bh) + al * bl


def dd_add(x: tuple[float, float], y: tuple[float, float]) -> tuple[float, float]:
    sh, sl = two_sum(x[0], y[0])
    th, tl = two_sum(x[1], y[1])
    vh, vl = quick_two_sum(sh, sl + th)
    return quick_two_sum(vh, tl + vl)


def dd_sub(x: tuple[float, float], y: tuple[float, float]) -> tuple[float, float]:
    return dd_add(x, (-y[0], -y[1]))


def dd_mul_f64(x: tuple[float, float], y: float) -> tuple[float, float]:
    ph, pl = two_prod(x[0], y)
    return quick_two_sum(ph, pl + x[1] * y)


def sqrt40(x: float) -> float:
    if x <= 0.0:
        return 0.0
    guess = x * 0.5 if x > 1.0 else 1.0
    for _ in range(40):
        guess = 0.5 * (guess + x / guess)
    return guess


def poison() -> Value:
    return Value(poisoned=1)


def gate_class(value: Value) -> int:
    if value.poisoned:
        return 2
    roundoff = abs(value.error_hi + value.error_lo)
    if value.uncertainty <= 0.0:
        return 0 if roundoff == 0.0 else 2
    if roundoff * 10.0 >= value.uncertainty:
        return 2
    if roundoff * 100.0 >= value.uncertainty:
        return 1
    return 0


def execute_independent(parsed: dict[str, object]) -> tuple[list[dict[str, int]], list[dict[str, int]]]:
    grouped = parsed["grouped"]
    values_desc = grouped["value"]
    provenance = grouped["prov"]
    runtime: dict[int, Value] = {}
    memory: dict[int, tuple[int, Value]] = {}
    observations: list[dict[str, int]] = []
    for row in grouped["op"]:
        site, opcode, result, a, b, slot = map(int, (row[1], row[2], row[3], row[5], row[6], row[9]))
        if opcode == 0:
            runtime[result] = Value(bits_float(int(values_desc[result][4])))
        elif opcode in {2, 3, 4, 5}:
            x, y = runtime[a], runtime[b]
            if x.poisoned or y.poisoned or (opcode == 5 and y.value == 0.0):
                runtime[result] = poison()
                continue
            if opcode in {2, 3}:
                vh, residual = two_sum(x.value, y.value if opcode == 2 else -y.value)
                error = dd_add((x.error_hi, x.error_lo), (y.error_hi, y.error_lo)) if opcode == 2 else dd_sub((x.error_hi, x.error_lo), (y.error_hi, y.error_lo))
                error = dd_add(error, (residual, 0.0))
                u = sqrt40(x.uncertainty**2 + y.uncertainty**2)
            elif opcode == 4:
                vh, residual = two_prod(x.value, y.value)
                error = dd_add(dd_add(dd_mul_f64((x.error_hi, x.error_lo), y.value), dd_mul_f64((y.error_hi, y.error_lo), x.value)), (residual, 0.0))
                u = sqrt40((y.value * x.uncertainty) ** 2 + (x.value * y.uncertainty) ** 2)
            else:
                vh = x.value / y.value
                product = two_prod(vh, y.value)
                residual_dd = dd_sub((x.value, 0.0), product)
                residual = (residual_dd[0] + residual_dd[1]) / y.value
                error = dd_add(dd_sub(dd_mul_f64((x.error_hi, x.error_lo), 1.0 / y.value), dd_mul_f64((y.error_hi, y.error_lo), vh / y.value)), (residual, 0.0))
                u = (1.0 / abs(y.value)) * sqrt40(x.uncertainty**2 + (vh * y.uncertainty) ** 2)
            runtime[result] = Value(vh, error[0], error[1], u, 0)
        elif opcode == 6:
            x = runtime[a]
            if x.poisoned:
                runtime[result] = poison()
            elif x.value <= 0.0:
                runtime[result] = Value()
            else:
                root = sqrt40(x.value)
                product = two_prod(root, root)
                residual_dd = dd_sub((x.value, 0.0), product)
                residual = (residual_dd[0] + residual_dd[1]) / (2.0 * root)
                error = dd_add(dd_mul_f64((x.error_hi, x.error_lo), 1.0 / (2.0 * root)), (residual, 0.0))
                runtime[result] = Value(root, error[0], error[1], x.uncertainty / (2.0 * root), 0)
        elif opcode == 9:
            runtime[result] = runtime[a]
        elif opcode == 8:
            memory[slot] = (site, runtime[a])
        elif opcode == 7:
            value = runtime[a]
            observations.append({
                "site": site,
                "value_id": a,
                "value_bits": POISON_VALUE_BITS if value.poisoned else fbits(value.value),
                "error0_bits": fbits(value.error_hi),
                "error1_bits": fbits(value.error_lo),
                "uncertainty_bits": POISON_U_BITS if value.poisoned else fbits(value.uncertainty),
                "status": value.poisoned,
                "gate_class": gate_class(value),
                "source_span": int(provenance[a][2]),
            })
        else:
            fail(f"independent interpreter saw unsupported opcode {opcode}")
    memory_rows = []
    for slot, (site, value) in sorted(memory.items()):
        memory_rows.append({
            "slot": slot,
            "site": site,
            "value_bits": POISON_VALUE_BITS if value.poisoned else fbits(value.value),
            "error0_bits": fbits(value.error_hi),
            "error1_bits": fbits(value.error_lo),
            "uncertainty_bits": POISON_U_BITS if value.poisoned else fbits(value.uncertainty),
            "status": value.poisoned,
        })
    return observations, memory_rows


def parse_native_receipts(output: bytes) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in output.decode("ascii").splitlines():
        if not line.startswith("enir-exec|"):
            continue
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        rows.append({key: int(fields[key]) for key in ("site", "value_id", "value_bits", "error0_bits", "error1_bits", "uncertainty_bits", "status", "gate_class", "source_span")})
    return rows


def parse_native_memory(output: bytes) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in output.decode("ascii").splitlines():
        if not line.startswith("enir-memory|"):
            continue
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        rows.append({key: int(fields[key]) for key in ("slot", "site", "value_bits", "error0_bits", "error1_bits", "uncertainty_bits", "status")})
    return rows


def parts_bits(token: str) -> int:
    match = re.fullmatch(r"s([01])e([0-9]+)m([0-9]+)", token)
    if not match:
        fail(f"invalid EVM parts token: {token}")
    sign, exponent, mantissa = map(int, match.groups())
    return signed_u64((sign << 63) | (exponent << 52) | mantissa)


def parse_evm_receipts(path: Path) -> list[dict[str, int]]:
    rows = []
    for line in path.read_text(encoding="ascii").splitlines():
        if not line.startswith("eisa-receipt: v=1 "):
            continue
        match = re.fullmatch(r"eisa-receipt: v=1 prog=[0-9]+ gate=[0-9]+ reg=e[0-9]+ val=(\S+) roundoff=(\S+) u=(\S+) poisoned=([01])", line)
        if not match:
            fail(f"invalid v1 EVM receipt: {line}")
        value, error, uncertainty, poisoned = match.groups()
        rows.append({"value_bits": parts_bits(value), "error0_bits": parts_bits(error), "uncertainty_bits": parts_bits(uncertainty), "status": int(poisoned)})
        if len(rows) == 6:
            break
    if len(rows) != 6:
        fail(f"expected first six v1 EVM receipts, found {len(rows)}")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for implementation in (args.root / "self-hosted/enir/source_lower.sio", args.root / "self-hosted/enir/interpreter.sio"):
        text = implementation.read_text(encoding="utf-8")
        forbidden = ("enir_shadow_fixture(", "use eisa::", "use math::dd64", "eisax_encode(", "eisa_compile(")
        if any(token in text for token in forbidden) or any(case.removesuffix("_src") in text for case in CASES):
            fail(f"fixture/case-specific dependency in {implementation}")

    oracle = parse_evm_receipts(args.oracle)
    oracle_cursor = 0
    receipts = []
    aggregate_opcodes: set[int] = set()
    for case_id, source in extract_sources(args.corpus):
        source_path = args.out_dir / f"{case_id}.eisa"
        artifact_path = args.out_dir / f"{case_id}.enir"
        source_path.write_text(source, encoding="ascii")
        first = run([str(args.driver), "lower", str(source_path)]).stdout
        second = run([str(args.driver), "lower", str(source_path)]).stdout
        if first != second:
            fail(f"nondeterministic lowering for {case_id}")
        artifact_path.write_bytes(first)
        parsed = parse_enir(first)
        expected = independently_lower(source)
        verify_lowered(parsed, expected)
        roundtrip = run([str(args.driver), "roundtrip", str(artifact_path)]).stdout
        if roundtrip != first:
            fail(f"non-identical canonical roundtrip for {case_id}")
        native_output = run([str(args.driver), "run", str(artifact_path)]).stdout
        native = parse_native_receipts(native_output)
        native_memory = parse_native_memory(native_output)
        independent, independent_memory = execute_independent(parsed)
        if native != independent or native_memory != independent_memory:
            fail(f"native/independent ENIR execution mismatch for {case_id}: observations={native}/{independent} memory={native_memory}/{independent_memory}")
        expected_oracle = oracle[oracle_cursor : oracle_cursor + len(native)]
        observed_projection = [{key: row[key] for key in ("value_bits", "error0_bits", "uncertainty_bits", "status")} for row in native]
        if observed_projection != expected_oracle:
            fail(f"ENIR/EVM observable mismatch for {case_id}: enir={observed_projection} evm={expected_oracle}")
        oracle_cursor += len(native)
        aggregate_opcodes.update(int(row[2]) for row in parsed["grouped"]["op"])
        receipts.append({
            "case_id": case_id,
            "module": expected.module_name,
            "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
            "enir_sha256": hashlib.sha256(first).hexdigest(),
            "enir_l64": rolling_hash(first),
            "ops": len(expected.ops),
            "values": len(expected.values),
            "observations": len(native),
            "memory_events": len(native_memory),
            "native_independent_exact": True,
            "evm_observable_exact": True,
        })
    if oracle_cursor != 6 or aggregate_opcodes != {0, 2, 3, 4, 5, 6, 7, 8}:
        fail(f"incomplete E2 coverage: observations={oracle_cursor} opcodes={sorted(aggregate_opcodes)}")

    # Causal source tamper: one literal changes artifact and the single dependent
    # golden_mul observation, while the source identity remains derived.
    original = (args.out_dir / "golden_mul.eisa").read_text(encoding="ascii")
    tampered = original.replace("7.25", "7.5", 1)
    tamper_source = args.out_dir / "golden_mul_tamper.eisa"
    tamper_artifact = args.out_dir / "golden_mul_tamper.enir"
    tamper_source.write_text(tampered, encoding="ascii")
    tamper_bytes = run([str(args.driver), "lower", str(tamper_source)]).stdout
    tamper_artifact.write_bytes(tamper_bytes)
    tamper_native = parse_native_receipts(run([str(args.driver), "run", str(tamper_artifact)]).stdout)
    base_native = parse_native_receipts(run([str(args.driver), "run", str(args.out_dir / "golden_mul.enir")]).stdout)
    if hashlib.sha256(tamper_bytes).digest() == hashlib.sha256((args.out_dir / "golden_mul.enir").read_bytes()).digest() or len(tamper_native) != 1 or tamper_native == base_native:
        fail("causal literal tamper did not change artifact and exactly one observation")

    receipt = {
        "schema": "madaros.v2.e2.enir-lowering-receipt/1",
        "stage": "E2A-ENIR-V0-STRAIGHT-LINE-FULL",
        "stage_from": "eisa_v0_source",
        "stage_to": "enir",
        "profile": "eisa_v0_straight_line",
        "evidence_level": "R1_replay_and_differential",
        "source_to_enir": "compiler_owned_direct_no_eisax",
        "interpreter": "compiler_owned_independent_no_eisa_core",
        "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "cases": receipts,
        "programs": 5,
        "observations": 6,
        "memory_events": sum(case["memory_events"] for case in receipts),
        "opcodes": sorted(aggregate_opcodes),
        "native_independent_exact": True,
        "evm_observable_exact": True,
        "causal_source_tamper": True,
        "shadow_fixture_used": False,
        "fallback": "none",
        "codegen_changed": False,
    }
    args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("E2A_ENIR_V0_STRAIGHT_LINE_VERIFY_PASS programs=5 observations=6 opcodes=0,2,3,4,5,6,7,8 native_independent=exact evm_observable=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2Error, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2A_ENIR_V0_STRAIGHT_LINE_VERIFY_FAIL: {exc}")
        raise SystemExit(1)
