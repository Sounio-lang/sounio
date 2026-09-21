#!/usr/bin/env python3
"""Independent E2G verifier for qd128 fuel, control, and frailty."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


PROGRAMS = ("v2_fuel", "v2_loop", "v2_frail")
IMAGE_OPCODE = {
    "econst": 0, "eadd": 2, "esub": 3, "egate": 7,
    "ebr": 10, "ebrz": 11, "ehalt": 13,
}


class E2GError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2GError(message)


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


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"frozen builder not found: {name}")
    start, depth, index = match.end(), 1, match.end()
    while index < len(source):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index]
        index += 1
    fail(f"unterminated frozen builder: {name}")
    raise AssertionError


def parse_frozen_image(corpus: Path, name: str, e2a) -> dict[str, object]:
    body = function_body(corpus.read_text(encoding="utf-8"), f"{name}_img")
    fuel_match = re.search(r"eisax_build_set_fuel\(b,\s*([0-9]+)\)", body)
    count_match = re.search(r"b\.n_code\s*=\s*([0-9]+)", body)
    if not fuel_match or not count_match:
        fail(f"frozen image framing mismatch: {name}")
    constants = {
        int(index): e2a.fbits(float(value))
        for index, value in re.findall(r"b\.consts\[([0-9]+)\]\s*=\s*([0-9]+(?:\.[0-9]+)?(?:e[+-]?[0-9]+)?)", body)
    }
    fields: dict[int, dict[str, int]] = {}
    for index, opcode in re.findall(r"b\.ops\[([0-9]+)\]\s*=\s*op_([a-z0-9]+)\(\)", body):
        if opcode not in IMAGE_OPCODE:
            fail(f"unsupported frozen opcode {opcode}: {name}")
        fields.setdefault(int(index), {})["opcode"] = IMAGE_OPCODE[opcode]
    for source, target in (("dsts", "dst"), ("as_", "a"), ("bs_", "b")):
        for index, value in re.findall(rf"b\.{source}\[([0-9]+)\]\s*=\s*([0-9]+)", body):
            fields.setdefault(int(index), {})[target] = int(value)
    count = int(count_match.group(1))
    if sorted(fields) != list(range(count)):
        fail(f"frozen instruction range mismatch: {name}")
    return {"fuel": int(fuel_match.group(1)), "constants": constants, "rows": [fields[i] for i in range(count)]}


def verify_frozen_shapes(images: dict[str, dict[str, object]], e2a) -> None:
    expected = {
        "v2_fuel": {
            "fuel": 5, "constants": {0: e2a.fbits(1.0)},
            "rows": [{"opcode": 0, "dst": 5, "a": 0}, {"opcode": 10, "a": 1}],
        },
        "v2_loop": {
            "fuel": 24, "constants": {0: e2a.fbits(0.0), 1: e2a.fbits(3.0)},
            "rows": [
                {"opcode": 0, "dst": 0, "a": 0}, {"opcode": 0, "dst": 2, "a": 1},
                {"opcode": 11, "a": 0, "b": 4}, {"opcode": 13},
                {"opcode": 7, "dst": 2}, {"opcode": 13},
            ],
        },
        "v2_frail": {
            "fuel": 30, "constants": {0: e2a.fbits(1.0e16), 1: e2a.fbits(1.0)},
            "rows": [
                {"opcode": 0, "dst": 0, "a": 0}, {"opcode": 0, "dst": 1, "a": 1},
                {"opcode": 2, "dst": 2, "a": 0, "b": 1},
                {"opcode": 3, "dst": 3, "a": 2, "b": 0},
                {"opcode": 11, "a": 3, "b": 6}, {"opcode": 13},
                {"opcode": 7, "dst": 3}, {"opcode": 13},
            ],
        },
    }
    if images != expected:
        fail("frozen v2 fuel/control/frail images drifted")


def verify_artifact(grouped: dict[str, list[list[str]]], expected, e2c) -> None:
    expected_blocks = [row.copy() for row in expected.blocks]
    expected_blocks[3][-1] = 1
    if grouped["enir"] != [["enir", "2", "2", expected.name, "2"]]:
        fail(f"schema/profile mismatch: {expected.name}")
    if grouped["resource"] != [["resource", str(expected.fuel)]]:
        fail(f"resource mismatch: {expected.name}")
    if grouped["type"] != [["type", "0", "4", "2", "1", "1", "1", "2"]]:
        fail(f"qd128 type mismatch: {expected.name}")
    if grouped["policy"] != [["policy", "0", "eisa_10_100_count_only", "0", "0"]]:
        fail(f"policy mismatch: {expected.name}")
    for tag, wanted in (("block", expected_blocks), ("barg", expected.block_args), ("edge", expected.edges)):
        actual = [list(map(int, row[1:])) for row in grouped[tag]]
        if actual != wanted:
            fail(f"{tag} mismatch for {expected.name}")
    if len(grouped["op"]) != len(expected.ops):
        fail(f"operation count mismatch: {expected.name}")
    for index, (row, op) in enumerate(zip(grouped["op"], expected.ops)):
        wanted = [index, op.opcode, op.result, op.type_id, op.a, op.b, -1, op.policy, -1, 1]
        if list(map(int, row[1:])) != wanted:
            fail(f"operation mismatch {expected.name} at {index}")
    if len(grouped["value"]) != len(expected.value_meta) or len(grouped["prov"]) != len(expected.value_meta):
        fail(f"value/provenance count mismatch: {expected.name}")
    for index, ((known, bits, line, transform), value, provenance) in enumerate(zip(expected.value_meta, grouped["value"], grouped["prov"])):
        fields = list(map(int, value[1:]))
        if fields[:4] != [index, 0, known, bits] or fields[5:10] != [0, 0, 0, 0, 0] or fields[-1] != index:
            fail(f"all-word value descriptor mismatch {expected.name} at {index}")
        if list(map(int, (provenance[1], provenance[2], provenance[4], provenance[5], provenance[6]))) != [index, line, transform, -1, -1]:
            fail(f"provenance mismatch {expected.name} at {index}")
    if grouped["obs"] != [["obs", "0", expected.name, "0", str(expected.observation_kind)]]:
        fail(f"observation descriptor mismatch: {expected.name}")
    footer = [1, len(expected.value_meta), len(expected.value_meta), 1, 4, len(expected.block_args), 4, len(expected.ops), 1, 1]
    if [list(map(int, row[1:])) for row in grouped["end2"]] != [footer]:
        fail(f"schema-v2 footer mismatch: {expected.name}")


def true_qd(value, qd):
    return qd.qd_add(qd.qd_from_f64(value.val), value.err)


def qd_error_x0(value) -> float:
    """Leading component named x0 by the pinned qd128 semantic profile."""
    return value.err[0]


def execute_qd(grouped: dict[str, list[list[str]]], qd) -> dict[str, object]:
    profile = int(grouped["enir"][0][4])
    if profile != 2:
        fail("independent qd replay received non-v2 artifact")
    fuel = int(grouped["resource"][0][1])
    descriptors = grouped["value"]
    ops = [list(map(int, row[1:])) for row in grouped["op"]]
    blocks = [list(map(int, row[1:])) for row in grouped["block"]]
    bargs = [list(map(int, row[1:])) for row in grouped["barg"]]
    edges = [list(map(int, row[1:])) for row in grouped["edge"]]
    runtime: dict[int, object] = {}
    block_id = 0
    executed = observations = last_write = frail_count = 0
    last_write = -1
    controls: list[tuple[int, int, int, int]] = []
    observed = None
    while True:
        block = blocks[block_id]
        for pc in range(block[1], block[1] + block[2]):
            if fuel == 0:
                if last_write < 0:
                    fail("fuel stop without last write")
                return {"fuel": 0, "stop": 1, "executed": executed, "observations": 1, "last_write": last_write, "value": runtime[last_write], "frail": frail_count, "controls": controls}
            fuel -= 1
            executed += 1
            _, opcode, result, _, a, b, _, _, _, _ = ops[pc]
            if opcode == 0:
                runtime[result] = qd.Value(qd.bits_float(int(descriptors[result][4])))
                last_write = result
            elif opcode in {2, 3, 4, 5}:
                x, y = runtime[a], runtime[b]
                value = {2: lambda: x.val + y.val, 3: lambda: x.val - y.val, 4: lambda: x.val * y.val, 5: lambda: x.val / y.val}[opcode]()
                truth = {2: qd.qd_add, 3: qd.qd_sub, 4: qd.qd_mul, 5: qd.qd_div}[opcode](true_qd(x, qd), true_qd(y, qd))
                runtime[result] = qd.Value(value, qd.qd_sub(truth, qd.qd_from_f64(value)))
                last_write = result
            elif opcode == 9:
                runtime[result] = runtime[a]
                last_write = result
            elif opcode == 7:
                observed = runtime[a]
                observations += 1
            else:
                fail(f"unsupported independent qd opcode {opcode}")
        term, condition, edge0, edge1, cost = block[5], block[6], block[7], block[8], block[9]
        if cost:
            if fuel == 0:
                if last_write < 0:
                    fail("terminator fuel stop without last write")
                return {"fuel": 0, "stop": 1, "executed": executed, "observations": 1, "last_write": last_write, "value": runtime[last_write], "frail": frail_count, "controls": controls}
            fuel -= 1
            executed += 1
        if term == 0:
            if observed is None or observations != 1:
                fail("halt without exactly one gate")
            return {"fuel": fuel, "stop": 0, "executed": executed, "observations": observations, "last_write": last_write, "value": observed, "frail": frail_count, "controls": controls}
        edge_id, taken, frail = edge0, 1, 0
        if term in {2, 3}:
            condition_value = runtime[condition]
            band = max(abs(qd_error_x0(condition_value)), abs(condition_value.uncertainty))
            frail = int(band != 0.0 and abs(condition_value.val) <= band)
            frail_count += frail
            taken = int(condition_value.val == 0.0) if term == 2 else int(condition_value.val < 0.0)
            edge_id = edge0 if taken else edge1
        controls.append((block_id, edge_id, taken, frail))
        edge = edges[edge_id]
        incoming = [runtime[edge[4 + ordinal]] for ordinal in range(edge[3])]
        target = blocks[edge[2]]
        for ordinal, value in enumerate(incoming):
            runtime[bargs[target[3] + ordinal][3]] = value
        block_id = edge[2]


def parse_native(raw: bytes) -> dict[str, object]:
    controls: list[tuple[int, int, int, int]] = []
    observation = final = None
    for line in raw.decode("ascii").splitlines():
        if line.startswith("enir-control|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            controls.append(tuple(int(fields[key]) for key in ("block", "edge", "taken", "frail")))
        elif line.startswith("enir-exec|") or line.startswith("enir-fuel|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status", "branch_poisoned", "frail_branches")
            observation = {key: int(fields[key]) for key in keys}
        elif line.startswith("enir-exec-ok|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            keys = ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if observation is None or final is None:
        fail("native execution omitted observation or final state")
    return {"controls": controls, "observation": observation, "final": final}


def parse_metron(path: Path, corpus: Path, e2d) -> dict[str, dict[str, int]]:
    receipts = [line for line in path.read_text(encoding="ascii").splitlines() if line.startswith("eisa-receipt:")]
    selected = {
        name: line for line, (name, _) in zip(receipts, e2d.corpus_manifest(corpus))
        if name in PROGRAMS
    }
    result: dict[str, dict[str, int]] = {}
    pattern = r"eisa-receipt: v=3 prog=[0-9]+ gate=([0-9]+) reg=e([0-9]+) val=(\S+) roundoff0=(\S+) roundoff1=(\S+) roundoff2=(\S+) roundoff3=(\S+) u=(\S+) poisoned=([01]) frail=([0-9]+)(?: stop=fuel)?"
    for name in PROGRAMS:
        match = re.fullmatch(pattern, selected.get(name, ""))
        if not match:
            fail(f"malformed or missing METRON receipt: {name}")
        gate, register, value, e0, e1, e2, e3, uncertainty, status, frail = match.groups()
        result[name] = {
            "gate": int(gate), "register": int(register), "value_bits": e2d.parts_bits(value),
            "error0_bits": e2d.parts_bits(e0), "error1_bits": e2d.parts_bits(e1),
            "error2_bits": e2d.parts_bits(e2), "error3_bits": e2d.parts_bits(e3),
            "uncertainty_bits": e2d.parts_bits(uncertainty), "status": int(status), "frail": int(frail),
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    e2a = load_module(args.root / "scripts/dev/madaros_v2_e2_enir_lowering_verify.py", "e2g_e2a")
    e2c = load_module(args.root / "scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py", "e2g_e2c")
    e2d = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e2g_e2d")
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e2g_qd")

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio"))
    for forbidden in ("enir_shadow_fixture(", "use eisa::", "use math::qd128", "eisax_encode(", "eisa_compile("):
        if forbidden in implementation:
            fail(f"forbidden shared-oracle dependency: {forbidden}")

    images = {name: parse_frozen_image(args.corpus, name, e2a) for name in PROGRAMS}
    verify_frozen_shapes(images, e2a)
    metron = parse_metron(args.oracle, args.corpus, e2d)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_cases = []
    for name in PROGRAMS:
        source_path = args.root / "tools/eisa" / f"eisa_enir_{name}.eisa"
        source = source_path.read_text(encoding="ascii")
        expected = e2c.independently_lower(source)
        expected_opcodes = {
            "v2_fuel": [0], "v2_loop": [0, 0, 7], "v2_frail": [0, 0, 2, 3, 7],
        }[name]
        if expected.name != name or expected.fuel != images[name]["fuel"] or [op.opcode for op in expected.ops] != expected_opcodes:
            fail(f"source/frozen normalized graph mismatch: {name}")
        artifact = run([str(args.driver), "lower-v2", str(source_path)])
        artifact_path = args.out_dir / f"{name}.enir"
        artifact_path.write_bytes(artifact)
        grouped = e2c.parse_enir(artifact)
        verify_artifact(grouped, expected, e2c)
        if run([str(args.driver), "roundtrip", str(artifact_path)]) != artifact:
            fail(f"canonical roundtrip changed bytes: {name}")
        replay = execute_qd(grouped, qd)
        native = parse_native(run([str(args.driver), "run", str(artifact_path)]))
        final = native["final"]
        observation = native["observation"]
        for actual_key, replay_key in (("fuel_left", "fuel"), ("stop_kind", "stop"), ("executed_ops", "executed"), ("last_write", "last_write"), ("frail_branches", "frail")):
            if final[actual_key] != replay[replay_key]:
                fail(f"native/independent replay mismatch {name} field {actual_key}")
        if native["controls"] != replay["controls"]:
            fail(f"native/independent control trace mismatch: {name}")
        value = replay["value"]
        replay_fields = {
            "value_bits": qd.fbits(value.val), "error0_bits": qd.fbits(value.err[0]),
            "error1_bits": qd.fbits(value.err[1]), "error2_bits": qd.fbits(value.err[2]),
            "error3_bits": qd.fbits(value.err[3]), "uncertainty_bits": qd.fbits(value.uncertainty),
            "status": 1 if replay["stop"] == 1 else value.poisoned,
        }
        for key, wanted in replay_fields.items():
            if observation[key] != wanted or observation[key] != metron[name][key]:
                fail(f"ENIR/replay/METRON mismatch {name} field {key}")
        if observation["frail_branches"] != metron[name]["frail"] or final["frail_branches"] != metron[name]["frail"]:
            fail(f"frailty mismatch: {name}")
        if observation["branch_poisoned"] != 0 or final["branch_poisoned"] != 0:
            fail(f"unexpected poisoned branch: {name}")
        if name == "v2_frail":
            exact = sum((Fraction.from_float(component) for component in true_qd(value, qd)), Fraction())
            if exact != 1:
                fail("v2_frail high-precision cancellation did not reconstruct exactly 1")
        receipt_cases.append({
            "name": name, "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "fuel_initial": expected.fuel, "fuel_left": replay["fuel"], "executed": replay["executed"],
            "stop_kind": replay["stop"], "frail_branches": replay["frail"],
            "blocks": len(grouped["block"]), "edges": len(grouped["edge"]),
            "all_qd_words_exact": True,
        })
    payload = {
        "schema": "madaros.e2g.enir-fuel-control-frail.v1",
        "cases": receipt_cases,
        "compiler_semantics_sha256": hashlib.sha256((args.root / "self-hosted/enir/interpreter.sio").read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "summary": {"programs": 3, "observations": 3, "cumulative_programs": 27, "cumulative_observations": 35, "independent_qd_replay": True, "source_matches_frozen_images": True},
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E2G_ENIR_V2_FUEL_CONTROL_FRAIL_VERIFY_PASS programs=3 observations=3 cumulative=27/30,35/39 blocks=12 edges=12 fuel=5->0,24->19,30->23 control=zero,nonzero,frail qd128_words=independent frail_true_value=exact1 metron_receipts=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2GError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2G_ENIR_V2_FUEL_CONTROL_FRAIL_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
