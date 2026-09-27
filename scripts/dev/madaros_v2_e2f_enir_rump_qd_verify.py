#!/usr/bin/env python3
"""Independent E2F verifier for the source-authored v2 Rump qd128 witness."""

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
from fractions import Fraction
from pathlib import Path


OPCODE = {"+": 2, "-": 3, "*": 4, "/": 5}


class E2FError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2FError(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load verifier dependency: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run(command: list[str]) -> bytes:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode != 0:
        fail(f"command failed rc={result.returncode}: {' '.join(command)}\n{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}")
    return result.stdout


def fbits(value: float) -> int:
    return struct.unpack(">q", struct.pack(">d", value))[0]


def bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">q", bits))[0]


@dataclass
class Op:
    opcode: int
    result: int = -1
    a: int = -1
    b: int = -1
    policy: int = -1
    line: int = 0


@dataclass
class Model:
    name: str
    fuel: int
    ops: list[Op]
    values: list[tuple[int, int, int, int, int]]
    gates: list[int]


def parse_source(path: Path, decimal_parser) -> Model:
    lines = path.read_text(encoding="ascii").splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    fuel = re.fullmatch(r"fuel ([0-9]+)", lines[1])
    if not header or not fuel or lines[-1] != "}":
        fail("v2 Rump source framing mismatch")
    symbols: dict[str, int] = {}
    ops: list[Op] = []
    values: list[tuple[int, int, int, int, int]] = []
    gates: list[int] = []

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, bits: int | None = None, origin: int = -1) -> int:
        result = len(values)
        values.append((1 if bits is not None else 0, bits or 0, line, opcode, origin))
        ops.append(Op(opcode, result, a, b, -1, line))
        return result

    def operand(token: str, line: int, origin: int) -> int:
        token = token.strip()
        if token in symbols:
            return symbols[token]
        return emit(0, line, bits=fbits(decimal_parser(token)), origin=origin)

    for line_number, raw in enumerate(lines[2:-1], 3):
        line = raw.strip()
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if let_match:
            name, expression = let_match.groups()
            if name in symbols:
                fail(f"duplicate symbol at line {line_number}")
            binary = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expression)
            if binary:
                left, operator, right = binary.groups()
                value = emit(OPCODE[operator], line_number, operand(left, line_number, line_number * 16), operand(right, line_number, line_number * 16 + 1))
            elif expression in symbols:
                value = emit(9, line_number, symbols[expression])
            else:
                value = emit(0, line_number, bits=fbits(decimal_parser(expression)), origin=line_number * 16)
            symbols[name] = value
        elif gate_match:
            name = gate_match.group(1)
            if name not in symbols:
                fail(f"undefined gate at line {line_number}")
            gates.append(symbols[name])
            ops.append(Op(7, -1, symbols[name], -1, 0, line_number))
        else:
            fail(f"unsupported source line {line_number}: {line}")
    model = Model(header.group(1), int(fuel.group(1)), ops, values, gates)
    if model.name != "v2_rump_qd" or model.fuel != 64 or len(ops) != 29 or len(values) != 26 or gates != [24, 22, 25]:
        fail("v2 Rump source manifest drifted from 29 ops / 26 values / gates 24,22,25")
    return model


def verify_graph(model: Model, frozen: list[object]) -> None:
    if len(frozen) != len(model.ops):
        fail("source/frozen graph length mismatch")
    for index, (actual, expected) in enumerate(zip(model.ops, frozen)):
        if (actual.opcode, actual.result, actual.a, actual.b, actual.policy) != (expected.opcode, expected.result, expected.a, expected.b, expected.policy):
            fail(f"source/frozen graph mismatch at op {index}")


def verify_artifact(grouped: dict[str, list[list[str]]], model: Model) -> None:
    if grouped["enir"] != [["enir", "2", "2", "v2_rump_qd", "2"]] or grouped["resource"] != [["resource", "64"]]:
        fail("v2 Rump schema/resource mismatch")
    if grouped["type"] != [["type", "0", "4", "2", "1", "1", "1", "2"]]:
        fail("v2 Rump qd128 type mismatch")
    if grouped["block"] != [["block", "0", "0", "29", "0", "0", "0", "-1", "-1", "-1", "1"]] or grouped["barg"] or grouped["edge"]:
        fail("v2 Rump one-block shape mismatch")
    if len(grouped["op"]) != 29 or len(grouped["value"]) != 26 or len(grouped["prov"]) != 26:
        fail("v2 Rump operation/value counts mismatch")
    for index, (row, op) in enumerate(zip(grouped["op"], model.ops)):
        wanted = [index, op.opcode, op.result, 0 if op.result >= 0 else -1, op.a, op.b, -1, op.policy, -1, 1]
        if list(map(int, row[1:])) != wanted:
            fail(f"v2 Rump operation mismatch at {index}")
    for index, ((known, bits, line, transform, origin), value, provenance) in enumerate(zip(model.values, grouped["value"], grouped["prov"])):
        fields = list(map(int, value[1:]))
        if fields[:4] != [index, 0, known, bits] or fields[5:10] != [0, 0, 0, 0, 0] or fields[-1] != index:
            fail(f"v2 Rump all-word descriptor mismatch at value {index}")
        if list(map(int, provenance[1:])) != [index, line, origin, transform, -1, -1]:
            fail(f"v2 Rump provenance mismatch at value {index}")
    if grouped["obs"] != [["obs", str(i), "v2_rump_qd", str(i), "0"] for i in range(3)]:
        fail("v2 Rump ordered observations mismatch")
    if grouped["end2"] != [["end2", "1", "26", "26", "1", "1", "0", "0", "29", "3", "1"]]:
        fail("v2 Rump footer mismatch")


def evaluate(model: Model, qd) -> list[object]:
    runtime: dict[int, object] = {}
    observations: list[object] = []
    for op in model.ops:
        if op.opcode == 0:
            runtime[op.result] = qd.Value(bits_float(model.values[op.result][1]))
        elif op.opcode in {2, 3, 4, 5}:
            x, y = runtime[op.a], runtime[op.b]
            val = {2: lambda: x.val + y.val, 3: lambda: x.val - y.val, 4: lambda: x.val * y.val, 5: lambda: x.val / y.val}[op.opcode]()
            tx = qd.qd_add(qd.qd_from_f64(x.val), x.err)
            ty = qd.qd_add(qd.qd_from_f64(y.val), y.err)
            truth = {2: qd.qd_add, 3: qd.qd_sub, 4: qd.qd_mul, 5: qd.qd_div}[op.opcode](tx, ty)
            runtime[op.result] = qd.Value(val, qd.qd_sub(truth, qd.qd_from_f64(val)))
        elif op.opcode == 7:
            observations.append(runtime[op.a])
        else:
            fail(f"unsupported qd replay opcode {op.opcode}")
    if len(observations) != 3:
        fail("v2 Rump replay observation count mismatch")
    return observations


def parse_metron(path: Path, corpus: Path, helper) -> list[dict[str, int]]:
    lines = [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("eisa-receipt:")]
    selected = [line for line, (name, _) in zip(lines, helper.corpus_manifest(corpus)) if name == "v2_rump_qd"]
    parsed: list[dict[str, int]] = []
    for line in selected:
        match = re.fullmatch(r"eisa-receipt: v=3 prog=[0-9]+ gate=([0-9]+) reg=e([0-9]+) val=(\S+) roundoff0=(\S+) roundoff1=(\S+) roundoff2=(\S+) roundoff3=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)", line)
        if not match:
            fail(f"malformed METRON v2 Rump receipt: {line}")
        gate, register, value, e0, e1, e2, e3, uncertainty, status, frail = match.groups()
        parsed.append({
            "gate": int(gate), "register": int(register), "value_bits": helper.parts_bits(value),
            "error0_bits": helper.parts_bits(e0), "error1_bits": helper.parts_bits(e1),
            "error2_bits": helper.parts_bits(e2), "error3_bits": helper.parts_bits(e3),
            "uncertainty_bits": helper.parts_bits(uncertainty), "status": int(status), "frail": int(frail),
        })
    if len(parsed) != 3 or [row["gate"] for row in parsed] != [1, 2, 3] or [row["register"] for row in parsed] != [13, 12, 14]:
        fail("METRON v2 Rump observation identities drifted")
    return parsed


def parse_native(raw: bytes) -> tuple[list[dict[str, int]], dict[str, int]]:
    observations: list[dict[str, int]] = []
    final: dict[str, int] | None = None
    for line in raw.decode("ascii").splitlines():
        if line.startswith("enir-exec|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("ordinal", "site", "value_id", "value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status", "gate_class", "branch_poisoned", "frail_branches", "source_span")
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith("enir-exec-ok|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if len(observations) != 3 or final is None:
        fail("native v2 Rump receipts missing")
    return observations, final


def true_qd(value, qd):
    return qd.qd_add(qd.qd_from_f64(value.val), value.err)


def verify_reconstruction(observations: list[object], qd) -> None:
    first, second, final = (true_qd(value, qd) for value in observations)
    target = qd.qd_div(qd.qd_from_f64(-54767.0), qd.qd_from_f64(66192.0))
    source_pair = qd.qd_add(first, second)
    if tuple(map(fbits, first)) != (fbits(-2.0), 0, 0, 0):
        fail("first gated source register does not reconstruct exactly -2")
    if tuple(map(fbits, source_pair)) != tuple(map(fbits, target)):
        fail("two gated source registers do not reconstruct target qd words")
    if tuple(map(fbits, final[:3])) != tuple(map(fbits, target[:3])) or fbits(final[3]) != 0 or fbits(target[3]) == 0:
        fail("single-register Rump precision boundary drifted")
    reconstructed = sum((Fraction.from_float(component) for component in target), Fraction())
    final_reconstructed = sum((Fraction.from_float(component) for component in final), Fraction())
    exact = Fraction(-54767, 66192)
    if abs(reconstructed - exact) > abs(exact) * Fraction(1, 1 << 210):
        fail("high-precision Rump target closure bound failed")
    if abs(final_reconstructed - exact) > abs(exact) * Fraction(1, 1 << 162):
        fail("single-register Rump relative-error bound failed")


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
    e2a = load_module(args.root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py", "e2a_helper")
    e2d = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e2d_helper")
    e2e = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e2e_helper")

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio", "self-hosted/enir/qd.sio"))
    for forbidden in ("use eisa::", "use math::qd128", "eisax_encode(", "eisa_compile(", "enir_shadow_fixture("):
        if forbidden in implementation:
            fail(f"forbidden shared-oracle dependency: {forbidden}")

    model = parse_source(args.source, e2a.decimal_like_source)
    verify_graph(model, e2d.parse_rump_image(args.corpus, e2a))
    replay = evaluate(model, e2e)
    metron = parse_metron(args.oracle, args.corpus, e2d)
    verify_reconstruction(replay, e2e)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifact = run([str(args.driver), "lower-v2", str(args.source)])
    artifact_path = args.out_dir / "v2_rump_qd.enir"
    artifact_path.write_bytes(artifact)
    verify_artifact(e2d.parse_enir(artifact), model)
    if run([str(args.driver), "roundtrip", str(artifact_path)]) != artifact:
        fail("v2 Rump canonical roundtrip changed bytes")
    native, final = parse_native(run([str(args.driver), "run", str(artifact_path)]))

    for ordinal, (actual, expected, oracle) in enumerate(zip(native, replay, metron)):
        fields = {
            "value_bits": fbits(expected.val), "error0_bits": fbits(expected.err[0]),
            "error1_bits": fbits(expected.err[1]), "error2_bits": fbits(expected.err[2]),
            "error3_bits": fbits(expected.err[3]), "uncertainty_bits": fbits(expected.uncertainty),
            "status": expected.poisoned,
        }
        for key, value in fields.items():
            if actual[key] != value:
                fail(f"native/independent qd mismatch observation {ordinal} field {key}")
            if actual[key] != oracle[key]:
                fail(f"ENIR/METRON mismatch observation {ordinal} field {key}")
        if actual["ordinal"] != ordinal or actual["site"] != 26 + ordinal or actual["value_id"] != model.gates[ordinal] or actual["frail_branches"] != oracle["frail"]:
            fail(f"v2 Rump observation identity mismatch at {ordinal}")
    wanted_final = {"executed_ops": 30, "observations": 3, "fuel_initial": 64, "fuel_left": 34, "stop_kind": 0, "last_write": 25, "branch_poisoned": 0, "frail_branches": 0}
    if final != wanted_final:
        fail(f"v2 Rump resource state mismatch: {final}")

    payload = {
        "schema": "madaros.e2f.enir-rump-qd.v1", "program": "v2_rump_qd",
        "source_sha256": hashlib.sha256(args.source.read_bytes()).hexdigest(),
        "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "qd_semantics_sha256": hashlib.sha256((args.root / "self-hosted/enir/qd.sio").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "operations": 29, "semantic_instructions": 30, "values": 26,
        "observations": 3, "fuel_initial": 64, "fuel_left": 34,
        "source_matches_frozen_graph": True, "independent_qd128_words": True,
        "two_register_reconstruction_exact": True, "single_register_boundary_honest": True,
        "target_relative_error_at_most_2^-210": True,
        "final_relative_error_at_most_2^-162": True, "metron_receipts_exact": True,
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E2F_ENIR_V2_RUMP_QD_VERIFY_PASS programs=1 observations=3 ops=29 semantic_instructions=30 values=26 fuel=64->34 graph=source==frozen-image qd128_words=independent pair_reconstruction=exact single_register_boundary=honest target_relative_bound=2^-210 final_relative_bound=2^-162 metron_receipts=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2FError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2F_ENIR_V2_RUMP_QD_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
