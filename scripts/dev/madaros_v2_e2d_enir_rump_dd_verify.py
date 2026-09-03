#!/usr/bin/env python3
"""Independent E2D verifier for the source-authored v1 Rump DD64 witness."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


TAGS = {
    "enir": 5, "resource": 2, "type": 8, "value": 14, "prov": 7,
    "policy": 5, "block": 11, "barg": 7, "edge": 9, "op": 11,
    "obs": 5, "end2": 11,
}
ORDER = {tag: index for index, tag in enumerate(TAGS)}
OPCODE = {"+": 2, "-": 3, "*": 4, "/": 5}
IMAGE_OPCODE = {"econst": 0, "eadd": 2, "esub": 3, "emul": 4, "ediv": 5, "egate": 7, "ehalt": 13}


class E2DError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2DError(message)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    if check and result.returncode != 0:
        fail(f"command failed rc={result.returncode}: {' '.join(command)}\n{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}")
    return result


def load_e2a(root: Path):
    path = root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py"
    spec = importlib.util.spec_from_file_location("e2a_verify", path)
    if spec is None or spec.loader is None:
        fail("cannot load independent E2A DD64 primitives")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"function not found: {name}")
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
    fail(f"unterminated function: {name}")
    raise AssertionError


def fbits(value: float) -> int:
    return struct.unpack(">q", struct.pack(">d", value))[0]


def bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">q", bits))[0]


def parts_bits(token: str) -> int:
    match = re.fullmatch(r"s([01])e([0-9]+)m([0-9]+)", token)
    if not match:
        fail(f"invalid EISA parts token: {token}")
    sign, exponent, mantissa = map(int, match.groups())
    raw = (sign << 63) | (exponent << 52) | mantissa
    return raw - (1 << 64) if raw >= (1 << 63) else raw


def parse_enir(raw: bytes) -> dict[str, list[list[str]]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail("ENIR is not canonical LF text")
    grouped = {tag: [] for tag in TAGS}
    prior = -1
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        row = line.split("|")
        if not row or row[0] not in TAGS or len(row) != TAGS[row[0]] or any(field == "" for field in row):
            fail(f"malformed schema-v2 row {number}")
        order = ORDER[row[0]]
        if order < prior:
            fail(f"non-canonical row order at {number}")
        prior = order
        grouped[row[0]].append(row)
    return grouped


@dataclass
class Op:
    opcode: int
    result: int = -1
    a: int = -1
    b: int = -1
    policy: int = -1
    line: int = 0


@dataclass
class SourceModel:
    name: str
    fuel: int
    ops: list[Op]
    values: list[tuple[int, int, int, int, int]]  # known, bits, line, transform, origin
    gates: list[int]


def parse_source(path: Path, e2a) -> SourceModel:
    lines = path.read_text(encoding="ascii").splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    fuel = re.fullmatch(r"fuel ([0-9]+)", lines[1])
    if not header or not fuel or lines[-1] != "}":
        fail("Rump source framing mismatch")
    symbols: dict[str, int] = {}
    ops: list[Op] = []
    values: list[tuple[int, int, int, int, int]] = []
    gates: list[int] = []

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, known: int = 0, bits: int = 0, origin: int = -1) -> int:
        value_id = len(values)
        values.append((known, bits, line, opcode, origin))
        ops.append(Op(opcode, value_id, a, b, -1, line))
        return value_id

    def operand(token: str, line: int, origin: int) -> int:
        token = token.strip()
        if token in symbols:
            return symbols[token]
        number = e2a.decimal_like_source(token)
        return emit(0, line, known=1, bits=e2a.fbits(number), origin=origin)

    for line_number, line in enumerate(lines[2:-1], 3):
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if let_match:
            name, expression = let_match.groups()
            if name in symbols:
                fail(f"duplicate independent symbol at line {line_number}")
            binary = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expression)
            if binary:
                left, operator, right = binary.groups()
                value = emit(OPCODE[operator], line_number, operand(left, line_number, line_number * 16), operand(right, line_number, line_number * 16 + 1))
            elif expression.strip() in symbols:
                value = emit(9, line_number, symbols[expression.strip()])
            else:
                number = e2a.decimal_like_source(expression.strip())
                value = emit(0, line_number, known=1, bits=e2a.fbits(number), origin=line_number * 16)
            symbols[name] = value
        elif gate_match:
            name = gate_match.group(1)
            if name not in symbols:
                fail(f"undefined independent gate at line {line_number}")
            value = symbols[name]
            ops.append(Op(7, -1, value, -1, 0, line_number))
            gates.append(value)
        else:
            fail(f"unsupported independent source line {line_number}: {line}")
    model = SourceModel(header.group(1), int(fuel.group(1)), ops, values, gates)
    if model.name != "v1_rump_dd" or model.fuel != 64 or len(model.ops) != 29 or len(model.values) != 26 or len(model.gates) != 3:
        fail("Rump source manifest drifted from 29 ops / 26 values / 3 gates")
    return model


def parse_rump_image(corpus: Path, e2a) -> list[Op]:
    body = function_body(corpus.read_text(encoding="utf-8"), "rump_build")
    constants = {int(index): e2a.fbits(e2a.decimal_like_source(value)) for index, value in re.findall(r"b\.consts\[([0-9]+)\]\s*=\s*([0-9]+(?:\.[0-9]+)?)", body)}
    fields: dict[int, dict[str, int]] = {}
    for index, opcode in re.findall(r"b\.ops\[([0-9]+)\]\s*=\s*op_([a-z0-9]+)\(\)", body):
        fields.setdefault(int(index), {})["opcode"] = IMAGE_OPCODE[opcode]
    for source_field, target in (("dsts", "dst"), ("as_", "a"), ("bs_", "b")):
        for index, value in re.findall(rf"b\.{source_field}\[([0-9]+)\]\s*=\s*([0-9]+)", body):
            fields.setdefault(int(index), {})[target] = int(value)
    if re.search(r"b\.n_code\s*=\s*30", body) is None or len(fields) != 30:
        fail("frozen rump_build is not exactly 30 instructions")

    registers: dict[int, int] = {}
    values_bits: list[int] = []
    graph: list[Op] = []
    for pc in range(30):
        row = fields[pc]
        opcode = row["opcode"]
        if opcode == 0:
            value_id = len(values_bits)
            values_bits.append(constants[row["a"]])
            registers[row["dst"]] = value_id
            graph.append(Op(0, value_id))
        elif opcode in {2, 3, 4, 5}:
            value_id = len(values_bits)
            values_bits.append(0)
            graph.append(Op(opcode, value_id, registers[row["a"]], registers[row["b"]]))
            registers[row["dst"]] = value_id
        elif opcode == 7:
            graph.append(Op(7, -1, registers[row["dst"]], -1, 0))
        elif opcode == 13:
            break
        else:
            fail(f"unsupported frozen Rump opcode {opcode}")
    if len(graph) != 29:
        fail("frozen Rump graph did not normalize to 29 ENIR operations")
    return graph


def verify_source_matches_image(source: SourceModel, image: list[Op]) -> None:
    for index, (actual, expected) in enumerate(zip(source.ops, image)):
        if (actual.opcode, actual.result, actual.a, actual.b, actual.policy) != (expected.opcode, expected.result, expected.a, expected.b, expected.policy):
            fail(f"source/frozen-image graph mismatch at operation {index}")


def verify_artifact(grouped: dict[str, list[list[str]]], model: SourceModel) -> None:
    if grouped["enir"] != [["enir", "2", "2", model.name, "1"]] or grouped["resource"] != [["resource", "64"]]:
        fail("schema/resource header mismatch")
    if grouped["type"] != [["type", "0", "4", "1", "1", "1", "1", "1"]] or grouped["policy"] != [["policy", "0", "eisa_10_100_count_only", "0", "0"]]:
        fail("type/policy mismatch")
    if grouped["block"] != [["block", "0", "0", "29", "0", "0", "0", "-1", "-1", "-1", "1"]] or grouped["barg"] or grouped["edge"]:
        fail("straight-line CFG shape mismatch")
    if len(grouped["op"]) != len(model.ops):
        fail("operation count mismatch")
    for index, (row, op) in enumerate(zip(grouped["op"], model.ops)):
        wanted = [index, op.opcode, op.result, 0 if op.result >= 0 else -1, op.a, op.b, -1, op.policy, -1, 1]
        if list(map(int, row[1:])) != wanted:
            fail(f"operation mismatch at {index}: {row}")
    if len(grouped["value"]) != len(model.values) or len(grouped["prov"]) != len(model.values):
        fail("value/provenance count mismatch")
    for index, ((known, bits, line, transform, origin), value, provenance) in enumerate(zip(model.values, grouped["value"], grouped["prov"])):
        if list(map(int, (value[1], value[2], value[3], value[4], value[13]))) != [index, 0, known, bits, index]:
            fail(f"value descriptor mismatch at {index}")
        if list(map(int, provenance[1:])) != [index, line, origin, transform, -1, -1]:
            fail(f"provenance mismatch at {index}")
    observations = grouped["obs"]
    if observations != [["obs", str(i), model.name, str(i), "0"] for i in range(3)]:
        fail("ordered observation descriptors mismatch")
    footer = [1, 26, 26, 1, 1, 0, 0, 29, 3, 1]
    if grouped["end2"] != [["end2", *map(str, footer)]]:
        fail("schema-v2 footer mismatch")


def evaluate(model: SourceModel, e2a) -> list[object]:
    runtime: dict[int, object] = {}
    observations: list[object] = []
    values = model.values
    for op in model.ops:
        if op.opcode == 0:
            runtime[op.result] = e2a.Value(bits_float(values[op.result][1]))
        elif op.opcode in {2, 3}:
            x, y = runtime[op.a], runtime[op.b]
            value, residual = e2a.two_sum(x.value, y.value if op.opcode == 2 else -y.value)
            error = e2a.dd_add((x.error_hi, x.error_lo), (y.error_hi, y.error_lo)) if op.opcode == 2 else e2a.dd_sub((x.error_hi, x.error_lo), (y.error_hi, y.error_lo))
            error = e2a.dd_add(error, (residual, 0.0))
            runtime[op.result] = e2a.Value(value, error[0], error[1], e2a.sqrt40(x.uncertainty**2 + y.uncertainty**2), 0)
        elif op.opcode == 4:
            x, y = runtime[op.a], runtime[op.b]
            value, residual = e2a.two_prod(x.value, y.value)
            error = e2a.dd_add(e2a.dd_add(e2a.dd_mul_f64((x.error_hi, x.error_lo), y.value), e2a.dd_mul_f64((y.error_hi, y.error_lo), x.value)), (residual, 0.0))
            uncertainty = e2a.sqrt40((y.value * x.uncertainty) ** 2 + (x.value * y.uncertainty) ** 2)
            runtime[op.result] = e2a.Value(value, error[0], error[1], uncertainty, 0)
        elif op.opcode == 5:
            x, y = runtime[op.a], runtime[op.b]
            value = x.value / y.value
            product = e2a.two_prod(value, y.value)
            residual_dd = e2a.dd_sub((x.value, 0.0), product)
            residual = (residual_dd[0] + residual_dd[1]) / y.value
            error = e2a.dd_add(e2a.dd_sub(e2a.dd_mul_f64((x.error_hi, x.error_lo), 1.0 / y.value), e2a.dd_mul_f64((y.error_hi, y.error_lo), value / y.value)), (residual, 0.0))
            uncertainty = (1.0 / abs(y.value)) * e2a.sqrt40(x.uncertainty**2 + (value * y.uncertainty) ** 2)
            runtime[op.result] = e2a.Value(value, error[0], error[1], uncertainty, 0)
        elif op.opcode == 7:
            observations.append(runtime[op.a])
        else:
            fail(f"unsupported replay opcode {op.opcode}")
    return observations


def corpus_manifest(corpus: Path) -> list[tuple[str, int]]:
    source = corpus.read_text(encoding="utf-8")
    main = function_body(source, "main")
    calls = re.findall(r"run_(one|img)\(\s*([A-Za-z0-9_]+)\(\)\s*\)", main)
    manifest: list[tuple[str, int]] = []
    for runner, function in calls:
        name = function.removesuffix("_src" if runner == "one" else "_img")
        if runner == "one":
            count = len(re.findall(r'add_line\s*\([^\n]*"gate\s', function_body(source, function)))
        elif "rump_build(" in function_body(source, function):
            count = len(re.findall(r"\bop_egate\s*\(", function_body(source, "rump_build")))
        else:
            count = len(re.findall(r"\bop_egate\s*\(", function_body(source, function)))
            if name in {"v1_fuel", "v1_fuel_high", "v2_fuel"} and count == 0:
                count = 1
        manifest.extend((name, ordinal) for ordinal in range(count))
    if len(calls) != 30 or len(manifest) != 39:
        fail("frozen METRON corpus manifest drifted")
    return manifest


def parse_metron(path: Path, corpus: Path) -> list[dict[str, int]]:
    receipts = [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("eisa-receipt:")]
    manifest = corpus_manifest(corpus)
    selected = [line for line, (name, _) in zip(receipts, manifest) if name == "v1_rump_dd"]
    result: list[dict[str, int]] = []
    for line in selected:
        match = re.fullmatch(r"eisa-receipt: v=2 prog=[0-9]+ gate=([0-9]+) reg=e([0-9]+) val=(\S+) roundoff=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)", line)
        if not match:
            fail(f"malformed v1 Rump METRON receipt: {line}")
        gate, register, value, roundoff, uncertainty, status, frail = match.groups()
        result.append({"gate": int(gate), "register": int(register), "value_bits": parts_bits(value), "roundoff_bits": parts_bits(roundoff), "uncertainty_bits": parts_bits(uncertainty), "status": int(status), "frail": int(frail)})
    if len(result) != 3 or [row["gate"] for row in result] != [1, 2, 3] or [row["register"] for row in result] != [13, 12, 14]:
        fail("v1 Rump METRON observation identity drifted")
    return result


def parse_native(output: bytes) -> tuple[list[dict[str, int]], dict[str, int]]:
    observations: list[dict[str, int]] = []
    final: dict[str, int] | None = None
    for line in output.decode("ascii").splitlines():
        if line.startswith("enir-exec|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("ordinal", "site", "value_id", "value_bits", "error0_bits", "error1_bits", "uncertainty_bits", "status", "gate_class", "branch_poisoned", "frail_branches", "source_span")
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith("enir-exec-ok|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if len(observations) != 3 or final is None:
        fail("native Rump receipts missing")
    return observations, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    e2a = load_e2a(args.root)

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio"))
    for forbidden in ("enir_shadow_fixture(", "use eisa::", "eisax_encode(", "eisa_compile(", "eisa_compile_v1("):
        if forbidden in implementation:
            fail(f"forbidden oracle dependency in ENIR implementation: {forbidden}")

    model = parse_source(args.source, e2a)
    image = parse_rump_image(args.corpus, e2a)
    verify_source_matches_image(model, image)
    metron = parse_metron(args.oracle, args.corpus)
    replay = evaluate(model, e2a)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = args.out_dir / "v1_rump_dd.enir"
    artifact = run([str(args.driver), "lower-v1", str(args.source)]).stdout
    artifact_path.write_bytes(artifact)
    grouped = parse_enir(artifact)
    verify_artifact(grouped, model)
    if run([str(args.driver), "roundtrip", str(artifact_path)]).stdout != artifact:
        fail("schema-v2 Rump roundtrip changed bytes")
    native, final = parse_native(run([str(args.driver), "run", str(artifact_path)]).stdout)

    for ordinal, (actual, expected, oracle) in enumerate(zip(native, replay, metron)):
        replay_fields = {
            "value_bits": e2a.fbits(expected.value),
            "error0_bits": e2a.fbits(expected.error_hi),
            "error1_bits": e2a.fbits(expected.error_lo),
            "uncertainty_bits": e2a.fbits(expected.uncertainty),
            "status": expected.poisoned,
        }
        for key, value in replay_fields.items():
            if actual[key] != value:
                fail(f"native/independent DD64 mismatch observation {ordinal} field {key}")
        if actual["value_bits"] != oracle["value_bits"] or e2a.fbits(expected.error_hi + expected.error_lo) != oracle["roundoff_bits"] or actual["uncertainty_bits"] != oracle["uncertainty_bits"] or actual["status"] != oracle["status"] or actual["frail_branches"] != oracle["frail"]:
            fail(f"ENIR/METRON public receipt mismatch at observation {ordinal}")
        if actual["ordinal"] != ordinal or actual["site"] != 26 + ordinal or actual["value_id"] != model.gates[ordinal]:
            fail(f"native observation identity mismatch at {ordinal}")
    wanted_final = {"executed_ops": 30, "observations": 3, "fuel_initial": 64, "fuel_left": 34, "stop_kind": 0, "last_write": 25, "branch_poisoned": 0, "frail_branches": 0}
    if final != wanted_final:
        fail(f"normal-termination resource state mismatch: {final}")

    payload = {
        "schema": "madaros.e2d.enir-rump-dd.v1",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "program": "v1_rump_dd",
        "operations": 29,
        "semantic_instructions": 30,
        "values": 26,
        "observations": 3,
        "fuel_initial": 64,
        "fuel_left": 34,
        "source_matches_frozen_image_graph": True,
        "independent_dd64_words": True,
        "metron_public_receipts": True,
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E2D_ENIR_V1_RUMP_DD_VERIFY_PASS programs=1 observations=3 ops=29 semantic_instructions=30 values=26 fuel=64->34 graph=source==frozen-image dd64_words=independent metron_receipts=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2DError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2D_ENIR_V1_RUMP_DD_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
