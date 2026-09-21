#!/usr/bin/env python3
"""Independent E2H verifier for qd128 memory, move, and poison flow."""

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


PROGRAMS = ("v2_mem", "v2_emov", "v2_mem_poison")
FUELS = {"v2_mem": 12, "v2_emov": 30, "v2_mem_poison": 20}
IMAGE_OPCODE = {
    "econst": 0, "eload": 1, "eadd": 2, "esub": 3, "emul": 4,
    "ediv": 5, "esqrt": 6, "egate": 7, "estore": 8, "emov": 9,
    "ehalt": 13,
}
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E2HError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2HError(message)


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


@dataclass
class Op:
    opcode: int
    result: int = -1
    a: int = -1
    b: int = -1
    slot: int = -1
    line: int = 0


@dataclass
class Model:
    name: str
    fuel: int
    ops: list[Op]
    values: list[tuple[int, int, int, int, int]]
    observations: list[int]


def parse_source(path: Path, decimal_parser, fbits) -> Model:
    lines = path.read_text(encoding="ascii").splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    fuel = re.fullmatch(r"fuel ([0-9]+)", lines[1])
    if not header or not fuel or lines[-1] != "}":
        fail(f"source framing mismatch: {path}")
    symbols: dict[str, int] = {}
    slots: dict[str, int] = {}
    store_sites: dict[int, int] = {}
    ops: list[Op] = []
    values: list[tuple[int, int, int, int, int]] = []
    observations: list[int] = []

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, slot: int = -1,
             bits: int | None = None, origin: int = -1) -> int:
        result = len(values)
        values.append((int(bits is not None), bits or 0, line, opcode, origin))
        ops.append(Op(opcode, result, a, b, slot, line))
        return result

    def operand(token: str, line: int, origin: int) -> int:
        token = token.strip()
        if token in symbols:
            return symbols[token]
        return emit(0, line, bits=fbits(decimal_parser(token)), origin=origin)

    for line_number, raw in enumerate(lines[2:-1], 3):
        line = raw.strip()
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*)=(.+)", line)
        store_match = re.fullmatch(r"store \[([A-Za-z][A-Za-z0-9_]*)\] <- ([A-Za-z][A-Za-z0-9_]*)", line)
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if let_match:
            name, expression = let_match.groups()
            if name in symbols:
                fail(f"duplicate symbol in {path}:{line_number}")
            load_match = re.fullmatch(r"load \[([A-Za-z][A-Za-z0-9_]*)\]", expression)
            binary = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expression)
            if load_match:
                slot_name = load_match.group(1)
                if slot_name not in slots or slots[slot_name] not in store_sites:
                    fail(f"load before store in {path}:{line_number}")
                slot = slots[slot_name]
                value = emit(1, line_number, slot=slot, origin=store_sites[slot])
            elif binary:
                left, operator, right = binary.groups()
                opcode = {"+": 2, "-": 3, "*": 4, "/": 5}[operator]
                value = emit(opcode, line_number, operand(left, line_number, line_number * 16), operand(right, line_number, line_number * 16 + 1))
            elif expression in symbols:
                value = emit(9, line_number, symbols[expression])
            else:
                value = emit(0, line_number, bits=fbits(decimal_parser(expression)), origin=line_number * 16)
            symbols[name] = value
        elif store_match:
            slot_name, value_name = store_match.groups()
            if value_name not in symbols:
                fail(f"undefined store operand in {path}:{line_number}")
            slot = slots.setdefault(slot_name, len(slots))
            ops.append(Op(8, a=symbols[value_name], slot=slot, line=line_number))
            store_sites[slot] = len(ops) - 1
        elif gate_match:
            name = gate_match.group(1)
            if name not in symbols:
                fail(f"undefined gate in {path}:{line_number}")
            observations.append(symbols[name])
            ops.append(Op(7, a=symbols[name], line=line_number))
        else:
            fail(f"unsupported source line {path}:{line_number}: {line}")
    model = Model(header.group(1), int(fuel.group(1)), ops, values, observations)
    if model.name not in PROGRAMS or model.fuel != FUELS[model.name]:
        fail(f"source manifest mismatch: {path}")
    return model


def parse_frozen(corpus: Path, name: str, helper, fbits) -> tuple[int, list[Op]]:
    body = helper.function_body(corpus.read_text(encoding="utf-8"), f"{name}_img")
    fuel_match = re.search(r"eisax_build_set_fuel\(b,\s*([0-9]+)\)", body)
    count_match = re.search(r"b\.n_code\s*=\s*([0-9]+)", body)
    if not fuel_match or not count_match:
        fail(f"frozen image framing mismatch: {name}")
    constants = {int(i): fbits(float(v)) for i, v in re.findall(r"b\.consts\[([0-9]+)\]\s*=\s*([0-9]+(?:\.[0-9]+)?)", body)}
    rows: dict[int, dict[str, int]] = {}
    for index, opcode in re.findall(r"b\.ops\[([0-9]+)\]\s*=\s*op_([a-z0-9]+)\(\)", body):
        if opcode not in IMAGE_OPCODE:
            fail(f"unsupported frozen opcode {opcode}: {name}")
        rows.setdefault(int(index), {})["opcode"] = IMAGE_OPCODE[opcode]
    for source, target in (("dsts", "dst"), ("as_", "a"), ("bs_", "b")):
        for index, value in re.findall(rf"b\.{source}\[([0-9]+)\]\s*=\s*([0-9]+)", body):
            rows.setdefault(int(index), {})[target] = int(value)
    if sorted(rows) != list(range(int(count_match.group(1)))):
        fail(f"frozen image instruction range mismatch: {name}")
    registers: dict[int, int] = {}
    graph: list[Op] = []
    for pc in range(len(rows)):
        row, opcode = rows[pc], rows[pc]["opcode"]
        if opcode == 13:
            break
        if opcode == 0:
            result = len(registers)
            registers[row["dst"]] = result
            graph.append(Op(0, result, slot=constants[row["a"]]))
        elif opcode in {2, 3, 4, 5}:
            result = len(registers)
            graph.append(Op(opcode, result, registers[row["a"]], registers[row["b"]]))
            registers[row["dst"]] = result
        elif opcode == 8:
            graph.append(Op(8, a=registers[row["dst"]], slot=row["a"]))
        elif opcode == 1:
            result = len(registers)
            graph.append(Op(1, result, slot=row["a"]))
            registers[row["dst"]] = result
        elif opcode == 9:
            result = len(registers)
            graph.append(Op(9, result, registers[row["a"]]))
            registers[row["dst"]] = result
        elif opcode == 7:
            graph.append(Op(7, a=registers[row["dst"]]))
        else:
            fail(f"unsupported frozen opcode {opcode}: {name}")
    return int(fuel_match.group(1)), graph


def graph_tuple(op: Op, constant_bits: int | None = None) -> tuple[int, int, int, int, int]:
    payload = constant_bits if op.opcode == 0 else op.slot
    return op.opcode, op.result, op.a, op.b, payload


def verify_artifact(grouped: dict[str, list[list[str]]], model: Model) -> None:
    if grouped["enir"] != [["enir", "2", "2", model.name, "2"]] or grouped["resource"] != [["resource", str(model.fuel)]]:
        fail(f"schema/profile/resource mismatch: {model.name}")
    if grouped["type"] != [["type", "0", "4", "2", "1", "1", "1", "2"]]:
        fail(f"qd128 type mismatch: {model.name}")
    if grouped["block"] != [["block", "0", "0", str(len(model.ops)), "0", "0", "0", "-1", "-1", "-1", "1"]] or grouped["barg"] or grouped["edge"]:
        fail(f"straight-line CFG mismatch: {model.name}")
    if len(grouped["op"]) != len(model.ops) or len(grouped["value"]) != len(model.values) or len(grouped["prov"]) != len(model.values):
        fail(f"descriptor count mismatch: {model.name}")
    for index, (op, row) in enumerate(zip(model.ops, grouped["op"])):
        wanted = [index, op.opcode, op.result, 0 if op.result >= 0 else -1, op.a, op.b, -1, 0 if op.opcode == 7 else -1, op.slot, 1]
        if list(map(int, row[1:])) != wanted:
            fail(f"operation mismatch {model.name}:{index}")
    for index, ((known, bits, line, transform, origin), value, prov) in enumerate(zip(model.values, grouped["value"], grouped["prov"])):
        fields = list(map(int, value[1:]))
        if fields[:4] != [index, 0, known, bits] or fields[5:10] != [0, 0, 0, 0, 0] or fields[-1] != index:
            fail(f"all-word value descriptor mismatch {model.name}:{index}")
        if list(map(int, prov[1:])) != [index, line, origin, transform, -1, -1]:
            fail(f"provenance mismatch {model.name}:{index}")
    wanted_obs = [["obs", str(i), model.name, str(i), "0"] for i in range(len(model.observations))]
    if grouped["obs"] != wanted_obs:
        fail(f"observation descriptor mismatch: {model.name}")
    footer = [1, len(model.values), len(model.values), 1, 1, 0, 0, len(model.ops), len(model.observations), 1]
    if grouped["end2"] != [["end2", *map(str, footer)]]:
        fail(f"footer mismatch: {model.name}")


def canonical(value, qd):
    if value.poisoned:
        return value
    value.val = 0.0 if value.val == 0.0 else value.val
    return value


def poison(qd):
    return qd.Value(0.0, qd.qd_zero(), float("inf"), 1)


def evaluate(model: Model, qd) -> tuple[list[object], dict[int, tuple[int, object]], int]:
    runtime: dict[int, object] = {}
    memory: dict[int, tuple[int, object]] = {}
    observations: list[object] = []
    last_write = -1
    for site, op in enumerate(model.ops):
        if op.opcode == 0:
            runtime[op.result] = qd.Value(qd.bits_float(model.values[op.result][1]))
            last_write = op.result
        elif op.opcode in {2, 3, 4, 5}:
            x, y = runtime[op.a], runtime[op.b]
            if x.poisoned or y.poisoned or (op.opcode == 5 and y.val == 0.0):
                runtime[op.result] = poison(qd)
            else:
                raw = {2: lambda: x.val + y.val, 3: lambda: x.val - y.val, 4: lambda: x.val * y.val, 5: lambda: x.val / y.val}[op.opcode]()
                value = 0.0 if raw == 0.0 else raw
                tx, ty = qd.qd_add(qd.qd_from_f64(x.val), x.err), qd.qd_add(qd.qd_from_f64(y.val), y.err)
                truth = {2: qd.qd_add, 3: qd.qd_sub, 4: qd.qd_mul, 5: qd.qd_div}[op.opcode](tx, ty)
                runtime[op.result] = canonical(qd.Value(value, qd.qd_sub(truth, qd.qd_from_f64(value))), qd)
            last_write = op.result
        elif op.opcode == 8:
            value = runtime[op.a]
            memory[op.slot] = (site, qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned))
        elif op.opcode == 1:
            _, value = memory[op.slot]
            runtime[op.result] = qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned)
            last_write = op.result
        elif op.opcode == 9:
            value = runtime[op.a]
            runtime[op.result] = qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned)
            last_write = op.result
        elif op.opcode == 7:
            observations.append(runtime[op.a])
        else:
            fail(f"unsupported replay opcode {op.opcode}")
    return observations, memory, last_write


def value_words(value, qd) -> dict[str, int]:
    return {
        "value_bits": POISON_VALUE_BITS if value.poisoned else qd.fbits(value.val),
        **{f"error{i}_bits": qd.fbits(value.err[i]) for i in range(4)},
        "uncertainty_bits": POISON_U_BITS if value.poisoned else qd.fbits(value.uncertainty),
        "status": value.poisoned,
    }


def parse_metron(path: Path, corpus: Path, helper) -> dict[str, list[dict[str, int]]]:
    selected = {name: [] for name in PROGRAMS}
    lines = [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("eisa-receipt:")]
    for line, (name, _) in zip(lines, helper.corpus_manifest(corpus)):
        if name not in selected:
            continue
        match = re.fullmatch(r"eisa-receipt: v=3 prog=[0-9]+ gate=[0-9]+ reg=e[0-9]+ val=(\S+) roundoff0=(\S+) roundoff1=(\S+) roundoff2=(\S+) roundoff3=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)", line)
        if not match:
            fail(f"malformed METRON E2H receipt: {line}")
        value, e0, e1, e2, e3, uncertainty, status, frail = match.groups()
        selected[name].append({
            "value_bits": helper.parts_bits(value), "error0_bits": helper.parts_bits(e0),
            "error1_bits": helper.parts_bits(e1), "error2_bits": helper.parts_bits(e2),
            "error3_bits": helper.parts_bits(e3), "uncertainty_bits": helper.parts_bits(uncertainty),
            "status": int(status), "frail": int(frail),
        })
    if [len(selected[name]) for name in PROGRAMS] != [1, 2, 1]:
        fail(f"METRON E2H manifest incomplete: {[len(selected[name]) for name in PROGRAMS]}")
    return selected


def parse_native(raw: bytes) -> tuple[list[dict[str, int]], list[dict[str, int]], dict[str, int]]:
    observations: list[dict[str, int]] = []
    memory: list[dict[str, int]] = []
    final = None
    word_keys = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith("enir-exec|"):
            keys = ("ordinal", "site", "value_id", *word_keys, "gate_class", "branch_poisoned", "frail_branches", "source_span")
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith("enir-memory|"):
            keys = ("slot", "site", *word_keys)
            memory.append({key: int(fields[key]) for key in keys})
        elif line.startswith("enir-exec-ok|"):
            keys = ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if final is None:
        fail("native final receipt missing")
    return observations, memory, final


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
    helper = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e2h_helper")
    e2a = load_module(args.root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py", "e2h_e2a")
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e2h_qd")

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio", "self-hosted/enir/verify.sio"))
    for forbidden in ("enir_shadow_fixture(", "use eisa::", "eisax_encode(", "eisa_compile("):
        if forbidden in implementation:
            fail(f"forbidden shared-oracle dependency: {forbidden}")
    if any(name in implementation for name in PROGRAMS):
        fail("case-specific E2H dependency in compiler-owned implementation")

    metron = parse_metron(args.oracle, args.corpus, helper)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_programs = []
    total_observations = 0
    for name in PROGRAMS:
        source = args.source_dir / f"eisa_enir_{name}.eisa"
        model = parse_source(source, e2a.decimal_like_source, qd.fbits)
        frozen_fuel, frozen = parse_frozen(args.corpus, name, helper, qd.fbits)
        source_graph = [graph_tuple(op, model.values[op.result][1] if op.opcode == 0 else None) for op in model.ops]
        frozen_graph = [graph_tuple(op, op.slot if op.opcode == 0 else None) for op in frozen]
        if frozen_fuel != model.fuel or source_graph != frozen_graph:
            fail(f"source/frozen-image graph mismatch: {name}")
        replay_observations, replay_memory, last_write = evaluate(model, qd)
        artifact = run([str(args.driver), "lower-v2", str(source)])
        if run([str(args.driver), "lower-v2", str(source)]) != artifact:
            fail(f"nondeterministic lowering: {name}")
        artifact_path = args.out_dir / f"{name}.enir"
        artifact_path.write_bytes(artifact)
        grouped = helper.parse_enir(artifact)
        verify_artifact(grouped, model)
        if run([str(args.driver), "roundtrip", str(artifact_path)]) != artifact:
            fail(f"non-identical roundtrip: {name}")
        native, native_memory, final = parse_native(run([str(args.driver), "run", str(artifact_path)]))
        expected_words = [value_words(value, qd) for value in replay_observations]
        native_words = [{key: row[key] for key in expected_words[0]} for row in native]
        metron_words = [{key: row[key] for key in expected_words[0]} for row in metron[name]]
        if native_words != expected_words or native_words != metron_words:
            fail(f"native/independent/METRON observation mismatch: {name}")
        expected_memory = [{"slot": slot, "site": site, **value_words(value, qd)} for slot, (site, value) in sorted(replay_memory.items())]
        if native_memory != expected_memory:
            fail(f"native/independent atomic memory mismatch: {name}")
        if name in {"v2_mem", "v2_mem_poison"} and {key: native_memory[0][key] for key in expected_words[0]} != native_words[0]:
            fail(f"stored product did not survive load/move/gate word-for-word: {name}")
        if name == "v2_emov" and (native_words[0] != native_words[1] or native_words[0]["value_bits"] != 0):
            fail("emov did not preserve canonical positive zero word-for-word")
        wanted_final = {
            "executed_ops": len(model.ops) + 1, "observations": len(model.observations),
            "fuel_initial": model.fuel, "fuel_left": model.fuel - len(model.ops) - 1,
            "stop_kind": 0, "last_write": last_write, "branch_poisoned": 0, "frail_branches": 0,
        }
        if final != wanted_final:
            fail(f"resource/final-state mismatch {name}: {final}")
        total_observations += len(native)
        receipt_programs.append({
            "name": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(), "ops": len(model.ops),
            "values": len(model.values), "observations": len(native), "memory_slots": len(native_memory),
            "source_matches_frozen_graph": True, "native_independent_exact": True,
            "metron_receipts_exact": True, "atomic_full_product": True,
        })

    payload = {
        "schema": "madaros.e2h.enir-memory-move-poison.v1", "stage": "E2H-ENIR-QD128-MEMORY-MOVE-POISON-FULL",
        "programs": receipt_programs, "program_count": 3, "observation_count": total_observations,
        "cumulative_programs": 30, "cumulative_observations": 39,
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "independent_checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_matches_frozen_graph": True, "independent_qd128_words": True,
        "atomic_full_product_memory": True, "negative_zero_canonical": True,
        "poison_survives_store_load_move": True, "metron_receipts_exact": True,
        "shadow_fixture_used": False, "fallback": "none", "enir_to_mir": "not_implemented_separate_stage",
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E2H_ENIR_MEMORY_MOVE_POISON_VERIFY_PASS programs=3 observations=4 cumulative=30/30,39/39 graph=source==frozen-image qd128_words=independent memory=atomic-full-product negzero=canonical poison=preserved metron_receipts=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2HError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2H_ENIR_MEMORY_MOVE_POISON_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
