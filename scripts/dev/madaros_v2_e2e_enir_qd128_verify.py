#!/usr/bin/env python3
"""Independent verifier for E2E source-to-ENIR qd128 arithmetic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import struct
import subprocess
import sys
from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path


PROGRAMS = (
    "v2_const_gate", "v2_add", "v2_sub", "v2_mul", "v2_div", "v2_sqrt",
)
EXPECTED_CONSTANT_BITS = {
    "v2_const_gate": (4619848792751996928,),
    "v2_add": (4591870180066957722, 4596373779694328218),
    "v2_sub": (4599075939470750515, 4591870180066957722),
    "v2_mul": (4591870180066957722, 4596373779694328218),
    "v2_div": (4607182418800017408, 4613937818241073152),
    "v2_sqrt": (4611686018427387904,),
}
OPCODE = {"+": 2, "-": 3, "*": 4, "/": 5}
IMAGE_OPCODE = {"econst": 0, "eadd": 2, "esub": 3, "emul": 4, "ediv": 5, "esqrt": 6, "egate": 7, "ehalt": 13}


class E2EError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2EError(message)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        fail(f"cannot load verifier dependency {path}")
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


def two_sum(a: float, b: float) -> tuple[float, float]:
    total = a + b
    virtual = total - a
    return total, (a - (total - virtual)) + (b - virtual)


def quick_two_sum(a: float, b: float) -> tuple[float, float]:
    total = a + b
    return total, b - (total - a)


def two_prod(a: float, b: float) -> tuple[float, float]:
    product = a * b
    ca = 134217729.0 * a
    cb = 134217729.0 * b
    ahi, bhi = ca - (ca - a), cb - (cb - b)
    alo, blo = a - ahi, b - bhi
    error = ((ahi * bhi - product) + ahi * blo + alo * bhi) + alo * blo
    return product, error


def dd_add(x: tuple[float, float], y: tuple[float, float]) -> tuple[float, float]:
    s = two_sum(x[0], y[0])
    t = two_sum(x[1], y[1])
    v = quick_two_sum(s[0], s[1] + t[0])
    return quick_two_sum(v[0], t[1] + v[1])


def dd_add_f64(x: tuple[float, float], y: float) -> tuple[float, float]:
    s = two_sum(x[0], y)
    return quick_two_sum(s[0], s[1] + x[1])


Qd = tuple[float, float, float, float]


def qd_zero() -> Qd:
    return 0.0, 0.0, 0.0, 0.0


def qd_from_f64(value: float) -> Qd:
    return value, 0.0, 0.0, 0.0


def qd_renorm5(a0: float, a1: float, a2: float, a3: float, a4: float) -> Qd:
    s4 = quick_two_sum(a3, a4)
    s3 = quick_two_sum(a2, s4[0])
    s2 = quick_two_sum(a1, s3[0])
    s1 = quick_two_sum(a0, s2[0])
    current = s1[0]
    output: list[float] = []
    for term in (s1[1], s2[1], s3[1], s4[1]):
        result = quick_two_sum(current, term)
        current = result[0]
        if result[1] != 0.0:
            output.append(current)
            current = result[1]
    if len(output) < 4:
        output.append(current)
    return tuple((output + [0.0] * 4)[:4])  # type: ignore[return-value]


def qd_renorm4(a0: float, a1: float, a2: float, a3: float) -> Qd:
    return qd_renorm5(a0, a1, a2, a3, 0.0)


def qd_add_f64(a: Qd, b: float) -> Qd:
    s0 = two_sum(a[0], b)
    s1 = two_sum(a[1], s0[1])
    s2 = two_sum(a[2], s1[1])
    s3 = two_sum(a[3], s2[1])
    return qd_renorm5(s0[0], s1[0], s2[0], s3[0], s3[1])


def qd_double_accumulate(u: float, v: float, x: float) -> tuple[float, float, float]:
    s0 = two_sum(v, x)
    s1 = two_sum(u, s0[0])
    s, uu, vv = s1[0], s1[1], s0[1]
    if uu == 0.0:
        uu, s = s, 0.0
    if vv == 0.0:
        vv, uu, s = uu, s, 0.0
    return s, uu, vv


def qd_add(a: Qd, b: Qd) -> Qd:
    components = [*a, *b]
    # Stable insertion-sort semantics equal Python's stable sort for ties.
    components.sort(key=abs, reverse=True)
    u = v = 0.0
    output: list[float] = []
    for component in components:
        if len(output) >= 4:
            break
        s, u, v = qd_double_accumulate(u, v, component)
        if s != 0.0:
            output.append(s)
    if len(output) < 4:
        output.append(u)
    if len(output) < 4:
        output.append(v)
    output += [0.0] * (4 - len(output))
    return qd_renorm4(*output[:4])


def qd_neg(a: Qd) -> Qd:
    return tuple(-x for x in a)  # type: ignore[return-value]


def qd_sub(a: Qd, b: Qd) -> Qd:
    return qd_add(a, qd_neg(b))


def qd_mul_f64(a: Qd, b: float) -> Qd:
    terms = [*two_prod(a[0], b), *two_prod(a[1], b), *two_prod(a[2], b), a[3] * b]
    result = qd_zero()
    for term in terms:
        result = qd_add_f64(result, term)
    return result


def qd_three_sum(a: float, b: float, c: float) -> tuple[float, float, float]:
    s0 = two_sum(a, b)
    s1 = two_sum(c, s0[0])
    s2 = two_sum(s0[1], s1[1])
    return s1[0], s2[0], s2[1]


def qd_six_three_sum(a: tuple[float, float, float], b: tuple[float, float, float]) -> tuple[float, float, float]:
    result = qd_zero()
    for term in (*qd_three_sum(*a), *qd_three_sum(*b)):
        result = qd_add_f64(result, term)
    return result[0], result[1], result[2]


def qd_nine_two_sum(values: tuple[float, ...]) -> tuple[float, float]:
    d0 = dd_add(two_sum(values[0], values[1]), two_sum(values[2], values[3]))
    d1 = dd_add(two_sum(values[4], values[5]), two_sum(values[6], values[7]))
    return dd_add_f64(dd_add(d0, d1), values[8])


def qd_mul(a: Qd, b: Qd) -> Qd:
    p00 = two_prod(a[0], b[0])
    p01, p10 = two_prod(a[0], b[1]), two_prod(a[1], b[0])
    p02, p11, p20 = two_prod(a[0], b[2]), two_prod(a[1], b[1]), two_prod(a[2], b[0])
    p03, p12, p21, p30 = two_prod(a[0], b[3]), two_prod(a[1], b[2]), two_prod(a[2], b[1]), two_prod(a[3], b[0])
    e1 = qd_three_sum(p01[0], p10[0], p00[1])
    e2 = qd_six_three_sum((p02[0], p11[0], p20[0]), (p01[1], p10[1], 0.0))
    e3 = qd_nine_two_sum((p03[0], p12[0], p21[0], p30[0], p02[1], p11[1], p20[1], 0.0, 0.0))
    e4 = ((((((((p03[1] + p12[1]) + p21[1]) + p30[1]) + a[1] * b[3]) + a[2] * b[2]) + a[3] * b[1]) + 0.0) + 0.0)
    result = qd_zero()
    for term in (p00[0], *e1, *e2, *e3, e4):
        result = qd_add_f64(result, term)
    return result


def qd_div(a: Qd, b: Qd) -> Qd:
    quotient: list[float] = []
    remainder = a
    for _ in range(5):
        q = remainder[0] / b[0]
        quotient.append(q)
        remainder = qd_sub(remainder, qd_mul_f64(b, q))
    return qd_renorm5(*quotient)


def sqrt40(value: float) -> float:
    if value <= 0.0:
        return 0.0
    guess = value * 0.5 if value > 1.0 else 1.0
    for _ in range(40):
        guess = 0.5 * (guess + value / guess)
    return guess


def qd_sqrt(a: Qd) -> Qd:
    if a[0] <= 0.0:
        return qd_zero()
    x = qd_from_f64(1.0 / sqrt40(a[0]))
    for _ in range(3):
        one_minus = qd_sub(qd_from_f64(1.0), qd_mul(a, qd_mul(x, x)))
        x = qd_add(x, qd_mul_f64(qd_mul(x, one_minus), 0.5))
    return qd_mul(a, x)


@dataclass
class Op:
    opcode: int
    result: int = -1
    a: int = -1
    b: int = -1
    line: int = 0


@dataclass
class Model:
    name: str
    fuel: int
    ops: list[Op]
    values: list[tuple[int, int, int, int, int]]
    gate: int


def parse_source(path: Path, decimal_parser) -> Model:
    lines = path.read_text(encoding="ascii").splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    fuel = re.fullmatch(r"\s*fuel ([0-9]+)", lines[1])
    if not header or not fuel or lines[-1] != "}":
        fail(f"source framing mismatch: {path}")
    symbols: dict[str, int] = {}
    ops: list[Op] = []
    values: list[tuple[int, int, int, int, int]] = []
    gate = -1

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, bits: int | None = None, origin: int = -1) -> int:
        result = len(values)
        values.append((1 if bits is not None else 0, bits or 0, line, opcode, origin))
        ops.append(Op(opcode, result, a, b, line))
        return result

    def operand(token: str, line: int, origin: int) -> int:
        token = token.strip()
        if token in symbols:
            return symbols[token]
        return emit(0, line, bits=fbits(decimal_parser(token)), origin=origin)

    for line_number, raw in enumerate(lines[2:-1], 3):
        line = raw.strip()
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*) = (.+)", line)
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if let_match:
            name, expression = let_match.groups()
            if name in symbols:
                fail(f"duplicate symbol in {path}:{line_number}")
            sqrt_match = re.fullmatch(r"sqrt\(([A-Za-z][A-Za-z0-9_]*)\)", expression)
            binary = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expression)
            if sqrt_match:
                if sqrt_match.group(1) not in symbols:
                    fail(f"undefined sqrt operand in {path}:{line_number}")
                value = emit(6, line_number, symbols[sqrt_match.group(1)])
            elif binary:
                left, operator, right = binary.groups()
                value = emit(OPCODE[operator], line_number, operand(left, line_number, line_number * 16), operand(right, line_number, line_number * 16 + 1))
            elif expression in symbols:
                value = emit(9, line_number, symbols[expression])
            else:
                value = emit(0, line_number, bits=fbits(decimal_parser(expression)), origin=line_number * 16)
            symbols[name] = value
        elif gate_match:
            if gate_match.group(1) not in symbols or gate >= 0:
                fail(f"invalid gate in {path}:{line_number}")
            gate = symbols[gate_match.group(1)]
            ops.append(Op(7, -1, gate, -1, line_number))
        else:
            fail(f"unsupported source line {path}:{line_number}: {line}")
    model = Model(header.group(1), int(fuel.group(1)), ops, values, gate)
    if model.name not in PROGRAMS or model.fuel != 12 or gate < 0:
        fail(f"source manifest mismatch: {path}")
    return model


def parse_image(corpus: Path, name: str, helper) -> list[Op]:
    body = helper.function_body(corpus.read_text(encoding="utf-8"), f"{name}_img")
    constants = {int(index): fbits(float(value)) for index, value in re.findall(r"b\.consts\[([0-9]+)\]\s*=\s*([0-9]+(?:\.[0-9]+)?)", body)}
    rows: dict[int, dict[str, int]] = {}
    for index, opcode in re.findall(r"b\.ops\[([0-9]+)\]\s*=\s*op_([a-z0-9]+)\(\)", body):
        rows.setdefault(int(index), {})["opcode"] = IMAGE_OPCODE[opcode]
    for field, target in (("dsts", "dst"), ("as_", "a"), ("bs_", "b")):
        for index, value in re.findall(rf"b\.{field}\[([0-9]+)\]\s*=\s*([0-9]+)", body):
            rows.setdefault(int(index), {})[target] = int(value)
    count_match = re.search(r"b\.n_code\s*=\s*([0-9]+)", body)
    if not count_match or len(rows) != int(count_match.group(1)):
        fail(f"frozen image shape drifted for {name}")
    registers: dict[int, int] = {}
    graph: list[Op] = []
    for pc in range(len(rows)):
        row, opcode = rows[pc], rows[pc]["opcode"]
        if opcode == 0:
            value = len(registers)
            registers[row["dst"]] = value
            graph.append(Op(0, value))
        elif opcode in {2, 3, 4, 5}:
            value = len(registers)
            graph.append(Op(opcode, value, registers[row["a"]], registers[row["b"]]))
            registers[row["dst"]] = value
        elif opcode == 6:
            value = len(registers)
            graph.append(Op(6, value, registers[row["a"]]))
            registers[row["dst"]] = value
        elif opcode == 7:
            graph.append(Op(7, -1, registers[row["dst"]]))
        elif opcode == 13:
            break
        else:
            fail(f"unsupported frozen opcode {opcode} for {name}")
    return graph


def verify_artifact(grouped: dict[str, list[list[str]]], model: Model) -> None:
    op_count = len(model.ops)
    if grouped["enir"] != [["enir", "2", "2", model.name, "2"]] or grouped["resource"] != [["resource", "12"]]:
        fail(f"schema/profile mismatch for {model.name}")
    if grouped["type"] != [["type", "0", "4", "2", "1", "1", "1", "2"]]:
        fail(f"qd128 type mismatch for {model.name}")
    if grouped["block"] != [["block", "0", "0", str(op_count), "0", "0", "0", "-1", "-1", "-1", "1"]] or grouped["barg"] or grouped["edge"]:
        fail(f"straight-line block mismatch for {model.name}")
    if len(grouped["op"]) != op_count or len(grouped["value"]) != len(model.values):
        fail(f"operation/value count mismatch for {model.name}")
    for index, (row, op) in enumerate(zip(grouped["op"], model.ops)):
        wanted = [index, op.opcode, op.result, 0 if op.result >= 0 else -1, op.a, op.b, -1, 0 if op.opcode == 7 else -1, -1, 1]
        if list(map(int, row[1:])) != wanted:
            fail(f"operation mismatch for {model.name} at {index}")
    for index, ((known, bits, line, transform, origin), value, provenance) in enumerate(zip(model.values, grouped["value"], grouped["prov"])):
        fields = list(map(int, value[1:]))
        if fields[:4] != [index, 0, known, bits] or fields[5:9] != [0, 0, 0, 0] or fields[-1] != index:
            fail(f"all-word value descriptor mismatch for {model.name}:{index}")
        if list(map(int, provenance[1:])) != [index, line, origin, transform, -1, -1]:
            fail(f"provenance mismatch for {model.name}:{index}")
    if grouped["obs"] != [["obs", "0", model.name, "0", "0"]]:
        fail(f"observation mismatch for {model.name}")
    footer = [1, len(model.values), len(model.values), 1, 1, 0, 0, op_count, 1, 1]
    if grouped["end2"] != [["end2", *map(str, footer)]]:
        fail(f"footer mismatch for {model.name}")


@dataclass
class Value:
    val: float
    err: Qd = (0.0, 0.0, 0.0, 0.0)
    uncertainty: float = 0.0
    poisoned: int = 0


def evaluate(model: Model) -> Value:
    runtime: dict[int, Value] = {}
    observed: Value | None = None
    for op in model.ops:
        if op.opcode == 0:
            runtime[op.result] = Value(bits_float(model.values[op.result][1]))
        elif op.opcode in {2, 3, 4, 5}:
            x, y = runtime[op.a], runtime[op.b]
            val = {2: lambda: x.val + y.val, 3: lambda: x.val - y.val, 4: lambda: x.val * y.val, 5: lambda: x.val / y.val}[op.opcode]()
            tx, ty = qd_add(qd_from_f64(x.val), x.err), qd_add(qd_from_f64(y.val), y.err)
            truth = {2: qd_add, 3: qd_sub, 4: qd_mul, 5: qd_div}[op.opcode](tx, ty)
            runtime[op.result] = Value(val, qd_sub(truth, qd_from_f64(val)))
        elif op.opcode == 6:
            x = runtime[op.a]
            val = sqrt40(x.val)
            runtime[op.result] = Value(val, qd_sub(qd_sqrt(qd_add(qd_from_f64(x.val), x.err)), qd_from_f64(val)))
        elif op.opcode == 7:
            observed = runtime[op.a]
        else:
            fail(f"unsupported replay opcode {op.opcode}")
    if observed is None:
        fail(f"independent replay emitted no observation for {model.name}")
    return observed


def verify_high_precision(model: Model, observed: Value) -> None:
    constants = tuple(bits for known, bits, *_ in model.values if known == 1)
    if constants != EXPECTED_CONSTANT_BITS[model.name]:
        fail(f"source literal bits drifted for {model.name}: {constants}")
    components = (observed.val, *observed.err)
    reconstructed = sum((Fraction.from_float(value) for value in components), Fraction())
    inputs = [Fraction.from_float(bits_float(bits)) for bits in constants]
    if model.name == "v2_const_gate":
        exact = inputs[0]
    elif model.name == "v2_add":
        exact = inputs[0] + inputs[1]
    elif model.name == "v2_sub":
        exact = inputs[0] - inputs[1]
    elif model.name == "v2_mul":
        exact = inputs[0] * inputs[1]
    elif model.name == "v2_div":
        exact = inputs[0] / inputs[1]
    else:
        with localcontext() as context:
            context.prec = 300
            exact_decimal = Decimal.from_float(bits_float(constants[0])).sqrt()
            reconstructed_decimal = sum((Decimal.from_float(value) for value in components), Decimal(0))
            relative = abs(reconstructed_decimal - exact_decimal) / exact_decimal
            if relative > Decimal(2) ** Decimal(-210):
                fail(f"high-precision sqrt closure bound failed for {model.name}: {relative}")
        return
    error = abs(reconstructed - exact)
    bound = abs(exact) * Fraction(1, 1 << 210)
    if error > bound:
        fail(f"high-precision rational closure bound failed for {model.name}: {error} > {bound}")


def parse_metron(path: Path, corpus: Path, helper) -> dict[str, dict[str, int]]:
    lines = [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("eisa-receipt:")]
    manifest = helper.corpus_manifest(corpus)
    selected: dict[str, dict[str, int]] = {}
    for line, (name, ordinal) in zip(lines, manifest):
        if name not in PROGRAMS:
            continue
        match = re.fullmatch(r"eisa-receipt: v=3 prog=[0-9]+ gate=1 reg=e([0-9]+) val=(\S+) roundoff0=(\S+) roundoff1=(\S+) roundoff2=(\S+) roundoff3=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)", line)
        if not match or ordinal != 0:
            fail(f"malformed METRON v2 arithmetic receipt: {line}")
        register, value, e0, e1, e2, e3, uncertainty, status, frail = match.groups()
        selected[name] = {
            "register": int(register), "value_bits": helper.parts_bits(value),
            "error0_bits": helper.parts_bits(e0), "error1_bits": helper.parts_bits(e1),
            "error2_bits": helper.parts_bits(e2), "error3_bits": helper.parts_bits(e3),
            "uncertainty_bits": helper.parts_bits(uncertainty), "status": int(status), "frail": int(frail),
        }
    if tuple(name for name in PROGRAMS if name in selected) != PROGRAMS:
        fail("METRON v2 arithmetic manifest incomplete")
    return selected


def parse_native(raw: bytes) -> tuple[dict[str, int], dict[str, int]]:
    observation: dict[str, int] | None = None
    final: dict[str, int] | None = None
    for line in raw.decode("ascii").splitlines():
        if line.startswith("enir-exec|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("ordinal", "site", "value_id", "value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status", "gate_class", "branch_poisoned", "frail_branches", "source_span")
            observation = {key: int(fields[key]) for key in keys}
        elif line.startswith("enir-exec-ok|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if observation is None or final is None:
        fail("native qd128 receipts missing")
    return observation, final


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    helper = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e2d_helper")
    e2a = load_module(args.root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py", "e2a_helper")

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio", "self-hosted/enir/qd.sio"))
    for forbidden in ("use eisa::", "use math::qd128", "eisax_encode(", "eisa_compile(", "enir_shadow_fixture("):
        if forbidden in implementation:
            fail(f"forbidden shared-oracle dependency: {forbidden}")

    metron = parse_metron(args.oracle, args.corpus, helper)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_programs: list[dict[str, object]] = []
    total_ops = total_values = total_fuel_left = 0
    for name in PROGRAMS:
        source = args.source_dir / f"eisa_enir_{name}.eisa"
        model = parse_source(source, e2a.decimal_like_source)
        frozen = parse_image(args.corpus, name, helper)
        if [(o.opcode, o.result, o.a, o.b) for o in model.ops] != [(o.opcode, o.result, o.a, o.b) for o in frozen if o.opcode != 13]:
            fail(f"source/frozen-image graph mismatch for {name}")
        replay = evaluate(model)
        verify_high_precision(model, replay)
        artifact = run([str(args.driver), "lower-v2", str(source)])
        artifact_path = args.out_dir / f"{name}.enir"
        artifact_path.write_bytes(artifact)
        grouped = helper.parse_enir(artifact)
        verify_artifact(grouped, model)
        if run([str(args.driver), "roundtrip", str(artifact_path)]) != artifact:
            fail(f"canonical roundtrip changed bytes for {name}")
        native, final = parse_native(run([str(args.driver), "run", str(artifact_path)]))
        expected = {
            "value_bits": fbits(replay.val), "error0_bits": fbits(replay.err[0]),
            "error1_bits": fbits(replay.err[1]), "error2_bits": fbits(replay.err[2]),
            "error3_bits": fbits(replay.err[3]), "uncertainty_bits": fbits(replay.uncertainty),
            "status": replay.poisoned,
        }
        for key, value in expected.items():
            if native[key] != value:
                fail(f"native/independent qd128 mismatch {name} field {key}: {native[key]} != {value}")
            if native[key] != metron[name][key]:
                fail(f"ENIR/METRON mismatch {name} field {key}")
        if native["ordinal"] != 0 or native["site"] != len(model.ops) - 1 or native["value_id"] != model.gate or native["frail_branches"] != metron[name]["frail"]:
            fail(f"observation identity mismatch for {name}")
        wanted_final = {
            "executed_ops": len(model.ops) + 1, "observations": 1, "fuel_initial": 12,
            "fuel_left": 12 - len(model.ops) - 1, "stop_kind": 0,
            "last_write": len(model.values) - 1, "branch_poisoned": 0, "frail_branches": 0,
        }
        if final != wanted_final:
            fail(f"resource state mismatch for {name}: {final}")
        total_ops += len(model.ops)
        total_values += len(model.values)
        total_fuel_left += final["fuel_left"]
        receipt_programs.append({
            "name": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(), "ops": len(model.ops),
            "values": len(model.values), "fuel_left": final["fuel_left"],
            "words": [native[f"error{i}_bits"] for i in range(4)],
        })

    payload = {
        "schema": "madaros.e2e.enir-qd128-arithmetic.v1", "programs": receipt_programs,
        "program_count": 6, "observation_count": 6, "total_ops": total_ops,
        "total_values": total_values, "total_fuel_left": total_fuel_left,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "qd_semantics_sha256": hashlib.sha256((args.root / "self-hosted/enir/qd.sio").read_bytes()).hexdigest(),
        "independent_checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_matches_frozen_graph": True, "independent_qd128_words": True,
        "high_precision_semantic_oracle": True, "metron_receipts_exact": True,
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print(f"E2E_ENIR_QD128_VERIFY_PASS programs=6 observations=6 ops={total_ops} values={total_values} fuel_left_sum={total_fuel_left} graph=source==frozen-image qd128_words=independent high_precision=pass metron_receipts=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2EError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2E_ENIR_QD128_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
