#!/usr/bin/env python3
"""Independent E3D source/ENIR/Join-MIR validator and qd128 replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import subprocess
import sys
from pathlib import Path


PROGRAMS = ("v2_join_then", "v2_join_else")
WORD_KEYS = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
ENIR_TAGS = {"enir": 5, "resource": 2, "type": 8, "value": 14, "prov": 7, "policy": 5, "block": 11, "barg": 7, "edge": 9, "op": 11, "obs": 5, "end3": 11}
JMIR_TAGS = {"jmir": 7, "jtype": 8, "jvalue": 15, "jprov": 8, "jblock": 12, "jbarg": 8, "jedge": 10, "jsphi": 9, "jmver": 7, "jmphi": 10, "jinstr": 15, "jobs": 6, "jend": 12}
MIR_OPCODE = {0: 20, 1: 27, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26, 8: 28, 9: 29}


class E3DError(ValueError):
    pass


def fail(message: str) -> None:
    raise E3DError(message)


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


def rolling_hash(raw: bytes) -> int:
    value = 14695981
    for byte in raw:
        value = (value * 257 + byte) % 1_000_000_007
    return value


def fbits(value: float) -> int:
    return struct.unpack(">q", struct.pack(">d", value))[0]


def parse_rows(raw: bytes, tags: dict[str, int], label: str) -> dict[str, list[list[str]]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail(f"{label} is not canonical LF text")
    order = {tag: index for index, tag in enumerate(tags)}
    grouped = {tag: [] for tag in tags}
    prior = -1
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        row = line.split("|")
        if row[0] not in tags or len(row) != tags[row[0]] or any(field == "" for field in row):
            fail(f"malformed {label} row {number}")
        if order[row[0]] < prior:
            fail(f"non-canonical {label} section order at row {number}")
        prior = order[row[0]]
        grouped[row[0]].append(row)
    return grouped


def ints(row: list[str]) -> list[int]:
    return list(map(int, row[1:]))


def source_number(token: str) -> float:
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", token):
        fail(f"unsupported source number: {token}")
    return float(token)


def eval_expr(expr: str, values: dict[str, float]) -> float:
    for operator in ("+", "-", "*", "/"):
        if operator in expr:
            left, right = expr.split(operator, 1)
            x = values[left] if left in values else source_number(left)
            y = values[right] if right in values else source_number(right)
            return {"+": x + y, "-": x - y, "*": x * y, "/": x / y}[operator]
    return values[expr] if expr in values else source_number(expr)


def source_oracle(path: Path) -> dict[str, object]:
    lines = [line.strip() for line in path.read_text(encoding="ascii").splitlines() if line.strip()]
    match = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    if not match or lines[-1] != "}":
        fail(f"malformed source envelope: {path.name}")
    name = match.group(1)
    values: dict[str, float] = {}
    then_memory: dict[str, float] = {}
    else_memory: dict[str, float] = {}
    then_yield = else_yield = join_name = gate_name = None
    condition_name = None
    phase = "entry"
    fuel = -1
    for line in lines[1:-1]:
        if line.startswith("fuel "):
            fuel = int(line[5:])
        elif line.startswith("if_zero "):
            branch = re.fullmatch(r"if_zero ([A-Za-z][A-Za-z0-9_]*) == 0\.0 \{", line)
            if not branch or phase != "entry":
                fail(f"malformed source branch: {name}")
            condition_name = branch.group(1)
            phase = "then"
        elif line == "} else {":
            if phase != "then":
                fail(f"misplaced else: {name}")
            phase = "else"
        elif line == "}":
            if phase != "else":
                fail(f"misplaced branch close: {name}")
            phase = "closed"
        elif line.startswith("yield "):
            symbol = line[6:]
            if symbol not in values:
                fail(f"unknown yield value: {name}:{symbol}")
            if phase == "then":
                then_yield = values[symbol]
            elif phase == "else":
                else_yield = values[symbol]
            else:
                fail(f"yield outside branch: {name}")
        elif line.startswith("store "):
            store = re.fullmatch(r"store \[([A-Za-z][A-Za-z0-9_]*)\] <- ([A-Za-z][A-Za-z0-9_]*)", line)
            if not store or store.group(2) not in values:
                fail(f"malformed source store: {name}")
            target = then_memory if phase == "then" else else_memory if phase == "else" else None
            if target is None or store.group(1) in target:
                fail(f"invalid source store placement: {name}")
            target[store.group(1)] = values[store.group(2)]
        elif line.startswith("join "):
            if phase != "closed":
                fail(f"join before branch closure: {name}")
            join_name = line[5:]
            phase = "join"
        elif line.startswith("let "):
            lhs, expr = line[4:].split("=", 1)
            if lhs in values:
                fail(f"duplicate source binding: {name}:{lhs}")
            if expr.startswith("load ["):
                slot = expr[6:-1]
                memory = then_memory if values[condition_name] == 0.0 else else_memory
                if phase != "join" or slot not in memory:
                    fail(f"invalid source load: {name}:{slot}")
                values[lhs] = memory[slot]
            else:
                values[lhs] = eval_expr(expr, values)
        elif line.startswith("gate "):
            gate_name = line[5:]
        else:
            fail(f"unsupported source line: {name}: {line}")
        if phase == "join" and join_name is not None and join_name not in values:
            selected = then_yield if values[condition_name] == 0.0 else else_yield
            if selected is None:
                fail(f"missing source yield: {name}")
            values[join_name] = selected
    selected_then = values[condition_name] == 0.0
    memory = then_memory if selected_then else else_memory
    if fuel <= 0 or set(then_memory) != set(else_memory) or len(memory) != 2 or gate_name not in values or join_name is None:
        fail(f"incomplete source contract: {name}")
    return {"name": name, "fuel": fuel, "then": selected_then, "edge": 2 if selected_then else 3, "value": values[gate_name], "memory": memory}


def verify_relation(enir, jmir, enir_raw: bytes, name: str) -> None:
    if enir["enir"][0] != ["enir", "3", "2", name, "2"] or enir["end3"][0][1:] != ["1", "14", "14", "1", "4", "1", "4", "18", "1", "1"]:
        fail(f"ENIR join profile mismatch: {name}")
    if jmir["jmir"][0] != ["jmir", "4", "4", name, "2", enir["resource"][0][1], str(rolling_hash(enir_raw))]:
        fail(f"Join MIR header/source binding mismatch: {name}")
    if jmir["jtype"][0][1:] != enir["type"][0][1:]:
        fail(f"logical type changed: {name}")
    for target, source in (("jvalue", "value"), ("jprov", "prov"), ("jblock", "block"), ("jbarg", "barg"), ("jedge", "edge")):
        if len(jmir[target]) != len(enir[source]):
            fail(f"{target} cardinality mismatch: {name}")
        for index, (before, after) in enumerate(zip(enir[source], jmir[target])):
            if after[1:-1] != before[1:] or int(after[-1]) != index:
                fail(f"{target} relation mismatch {name}:{index}")
    blocks = [ints(row) for row in enir["block"]]
    stores: list[tuple[int, int, int]] = []
    for index, (source, lowered) in enumerate(zip(enir["op"], jmir["jinstr"])):
        op, got = ints(source), ints(lowered)
        source_opcode = op[1]
        block_id = next((block[0] for block in blocks if block[1] <= index < block[1] + block[2]), -1)
        effect = {1: 2, 7: 1, 8: 3}.get(source_opcode, 0)
        trap = int(2 <= source_opcode <= 6)
        memory_in = memory_out = -1
        if source_opcode == 8:
            memory_out = len(stores)
            stores.append((index, block_id, op[8]))
        elif source_opcode == 1:
            memory_in = 4 + op[8]
        wanted = [index, MIR_OPCODE[source_opcode], op[2], op[3], op[4], op[5], effect, trap, op[7], op[0], op[9], op[8], memory_in, memory_out]
        if got != wanted:
            fail(f"instruction relation mismatch {name}:{index}: {got} != {wanted}")
    if [(block, slot) for _, block, slot in stores] != [(1, 0), (1, 1), (2, 0), (2, 1)]:
        fail(f"source store graph mismatch: {name}:{stores}")
    source_arg = ints(enir["barg"][0])
    source_edges = [ints(row) for row in enir["edge"]]
    wanted_scalar_phi = [[0, source_arg[1], source_arg[3], source_edges[2][0], source_edges[2][4], source_edges[3][0], source_edges[3][4], source_arg[0]]]
    if [ints(row) for row in jmir["jsphi"]] != wanted_scalar_phi:
        fail(f"scalar SSA phi mismatch: {name}")
    wanted_versions = [[version, slot, 0, block, site, site] for version, (site, block, slot) in enumerate(stores)]
    wanted_versions += [[4, 0, 1, 3, -1, -1], [5, 1, 1, 3, -1, -1]]
    wanted_phis = [[0, 3, 0, 4, 2, 0, 3, 2, 3], [1, 3, 1, 5, 2, 1, 3, 3, 3]]
    if [ints(row) for row in jmir["jmver"]] != wanted_versions or [ints(row) for row in jmir["jmphi"]] != wanted_phis:
        fail(f"Memory SSA graph mismatch: {name}")
    if jmir["jobs"][0][1:5] != enir["obs"][0][1:] or jmir["jobs"][0][5] != "0":
        fail(f"observation relation mismatch: {name}")
    footer = ints(jmir["jend"][0])
    wanted_footer = [len(jmir["jtype"]), len(jmir["jvalue"]), len(jmir["jprov"]), len(jmir["jblock"]), len(jmir["jbarg"]), len(jmir["jedge"]), len(jmir["jsphi"]), len(jmir["jmver"]), len(jmir["jmphi"]), len(jmir["jinstr"]), len(jmir["jobs"])]
    if footer != wanted_footer:
        fail(f"Join MIR footer mismatch: {name}")


def clone(value, qd):
    return qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned)


def words(value, qd) -> dict[str, int]:
    return {"value_bits": qd.fbits(value.val), **{f"error{i}_bits": qd.fbits(value.err[i]) for i in range(4)}, "uncertainty_bits": qd.fbits(value.uncertainty), "status": value.poisoned}


def replay(jmir, qd) -> dict[str, object]:
    descriptors = jmir["jvalue"]
    blocks = [ints(row) for row in jmir["jblock"]]
    bargs = [ints(row) for row in jmir["jbarg"]]
    edges = [ints(row) for row in jmir["jedge"]]
    scalar_phis = [ints(row) for row in jmir["jsphi"]]
    memory_phis = [ints(row) for row in jmir["jmphi"]]
    instructions = [ints(row) for row in jmir["jinstr"]]
    runtime: dict[int, object] = {}
    memory: dict[int, object] = {}
    memory_version: dict[int, int] = {}
    memory_site: dict[int, int] = {}
    controls, scalar_trace, memory_trace = [], [], []
    fuel = int(jmir["jmir"][0][5])
    block_id = executed = observations = frail_count = 0
    last_write = -1
    observed = None
    while True:
        block = blocks[block_id]
        for pc in range(block[1], block[1] + block[2]):
            fuel -= 1
            executed += 1
            _, opcode, result, _, a, b, _, trap, _, source_op, tick, slot, memory_in, memory_out = instructions[pc]
            if fuel < 0 or source_op != pc or tick != 1:
                fail(f"independent replay identity/fuel mismatch at {pc}")
            if opcode == 20:
                runtime[result] = qd.Value(qd.bits_float(int(descriptors[result][4])))
                last_write = result
            elif opcode in {21, 22, 23, 24}:
                x, y = runtime[a], runtime[b]
                value = {21: x.val + y.val, 22: x.val - y.val, 23: x.val * y.val, 24: x.val / y.val}[opcode]
                truth = {21: qd.qd_add, 22: qd.qd_sub, 23: qd.qd_mul, 24: qd.qd_div}[opcode](qd.qd_add(qd.qd_from_f64(x.val), x.err), qd.qd_add(qd.qd_from_f64(y.val), y.err))
                runtime[result] = qd.Value(value, qd.qd_sub(truth, qd.qd_from_f64(value)))
                if trap != 1:
                    fail(f"arithmetic trap policy missing at {pc}")
                last_write = result
            elif opcode == 28:
                if memory_version.get(slot, -1) != memory_in:
                    fail(f"STORE version mismatch at {pc}")
                memory[slot], memory_version[slot], memory_site[slot] = clone(runtime[a], qd), memory_out, pc
            elif opcode == 27:
                if memory_version.get(slot) != memory_in:
                    fail(f"LOAD version mismatch at {pc}")
                runtime[result] = clone(memory[slot], qd)
                last_write = result
            elif opcode == 26:
                observed = clone(runtime[a], qd)
                observations += 1
            else:
                fail(f"unsupported replay opcode {opcode}")
        fuel -= block[9]
        executed += block[9]
        if block[5] == 0:
            if observed is None or observations != 1:
                fail("halt without one observation")
            return {"value": observed, "fuel": fuel, "executed": executed, "last_write": last_write, "controls": controls, "scalar": scalar_trace, "memory_phis": memory_trace, "memory": memory, "memory_version": memory_version, "memory_site": memory_site}
        edge_id, taken, frail = block[7], 1, 0
        if block[5] == 2:
            condition = runtime[block[6]]
            band = max(abs(condition.err[0]), abs(condition.uncertainty))
            frail = int(band != 0.0 and abs(condition.val) <= band)
            frail_count += frail
            taken = int(condition.val == 0.0)
            edge_id = block[7] if taken else block[8]
        edge = edges[edge_id]
        controls.append((block_id, block[10], block[5], block[6], edge_id, edge[8], edge[2], taken, 0, frail))
        for phi in scalar_phis:
            if phi[1] == edge[2]:
                incoming = phi[4] if phi[3] == edge_id else phi[6] if phi[5] == edge_id else -1
                if incoming != edge[4]:
                    fail(f"scalar phi mismatch on edge {edge_id}")
                scalar_trace.append((int(jmir["jmir"][0][6]), phi[0], phi[1], edge_id, incoming, phi[2]))
        for phi in memory_phis:
            if phi[1] == edge[2]:
                incoming = phi[5] if phi[4] == edge_id else phi[7] if phi[6] == edge_id else -1
                if memory_version.get(phi[2]) != incoming:
                    fail(f"memory phi mismatch on edge {edge_id}")
                memory_trace.append((int(jmir["jmir"][0][6]), phi[0], phi[1], phi[2], edge_id, incoming, phi[3]))
                memory_version[phi[2]] = phi[3]
        incoming_values = [clone(runtime[edge[4 + ordinal]], qd) for ordinal in range(edge[3])]
        target = blocks[edge[2]]
        for ordinal, value in enumerate(incoming_values):
            runtime[bargs[target[3] + ordinal][3]] = value
        block_id = edge[2]


def parse_execution(raw: bytes, prefix: str) -> dict[str, object]:
    controls, scalar, memory_phis, memories = [], [], [], []
    observation = final = None
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith(prefix + "-control|"):
            controls.append(tuple(int(fields[key]) for key in ("block", "source_block", "term", "condition", "edge", "source_edge", "to", "taken", "poisoned", "frail")))
        elif line.startswith(prefix + "-scalar-phi|"):
            scalar.append(tuple(int(fields[key]) for key in ("source_enir_hash", "phi", "block", "edge", "incoming_value", "result_value")))
        elif line.startswith(prefix + "-memory-phi|"):
            memory_phis.append(tuple(int(fields[key]) for key in ("source_enir_hash", "phi", "block", "slot", "edge", "incoming_version", "result_version")))
        elif line.startswith(prefix + "-exec|"):
            observation = {key: int(fields[key]) for key in ("source_enir_hash", "ordinal", "site", "source_op", "value_id", *WORD_KEYS, "branch_poisoned", "frail_branches", "source_span")}
        elif line.startswith(prefix + "-memory|"):
            memories.append({key: int(fields[key]) for key in ("source_enir_hash", "slot", "version", "site", "source_op", *WORD_KEYS)})
        elif line.startswith(prefix + "-exec-ok|"):
            final = {key: int(fields[key]) for key in ("jmir_hash", "source_enir_hash", "executed_instrs", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")}
    if observation is None or final is None:
        fail(f"{prefix} execution omitted observation/final receipt")
    return {"controls": controls, "scalar": scalar, "memory_phis": memory_phis, "memories": memories, "observation": observation, "final": final}


def parse_enir_execution(raw: bytes) -> dict[str, object]:
    observation = final = None
    controls, memories = [], []
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith("enir-control|"):
            controls.append(tuple(int(fields[key]) for key in ("block", "term", "condition", "edge", "to", "taken", "poisoned", "frail")))
        elif line.startswith("enir-exec|"):
            observation = {key: int(fields[key]) for key in WORD_KEYS}
        elif line.startswith("enir-memory|"):
            memories.append({key: int(fields[key]) for key in ("slot", "site", *WORD_KEYS)})
        elif line.startswith("enir-exec-ok|"):
            final = {key: int(fields[key]) for key in ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")}
    if observation is None or final is None:
        fail("ENIR execution omitted receipts")
    return {"controls": controls, "memories": memories, "observation": observation, "final": final}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e3d_qd")
    implementation = (args.root / "self-hosted/enir/mir_join.sio").read_text(encoding="utf-8")
    for forbidden in ("enir_interpret(", "use eisa::", "native::machine_ir"):
        if forbidden in implementation:
            fail(f"Join MIR implementation depends on forbidden surface: {forbidden}")
    if any(name in implementation for name in PROGRAMS):
        fail("Join MIR implementation contains case-specific program names")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_programs = []
    for name in PROGRAMS:
        source = args.source_dir / f"eisa_enir_{name}.eisa"
        oracle = source_oracle(source)
        enir_raw = run([str(args.driver), "lower-join-v2", str(source)])
        enir = parse_rows(enir_raw, ENIR_TAGS, "ENIR schema 3")
        enir_path = args.out_dir / f"{name}.enir"
        enir_path.write_bytes(enir_raw)
        jmir_raw = run([str(args.driver), "lower-join-mir", str(enir_path)])
        if run([str(args.driver), "lower-join-mir", str(enir_path)]) != jmir_raw:
            fail(f"nondeterministic Join MIR lowering: {name}")
        jmir = parse_rows(jmir_raw, JMIR_TAGS, "Join MIR")
        jmir_path = args.out_dir / f"{name}.jmir"
        jmir_path.write_bytes(jmir_raw)
        verify_relation(enir, jmir, enir_raw, name)
        relation = run([str(args.driver), "validate-join-mir", str(enir_path), str(jmir_path)])
        run([str(args.driver), "verify-join-mir", str(jmir_path)])
        enir_exec_raw = run([str(args.driver), "run", str(enir_path)])
        jmir_exec_raw = run([str(args.driver), "run-join-mir", str(jmir_path)])
        enir_exec = parse_enir_execution(enir_exec_raw)
        jmir_exec = parse_execution(jmir_exec_raw, "jmir")
        independent = replay(jmir, qd)
        expected_words = words(independent["value"], qd)
        if expected_words["value_bits"] != fbits(oracle["value"]):
            fail(f"source oracle/independent replay mismatch: {name}")
        if enir_exec["observation"] != expected_words or {key: jmir_exec["observation"][key] for key in WORD_KEYS} != expected_words:
            fail(f"source/ENIR/Join-MIR observable mismatch: {name}")
        source_hash = rolling_hash(enir_raw)
        observation = jmir_exec["observation"]
        gate = next(row for row in jmir["jinstr"] if int(row[2]) == 26)
        gate_value = int(gate[5])
        expected_span = int(jmir["jprov"][gate_value][2])
        if observation["source_enir_hash"] != source_hash or observation["ordinal"] != 0 or observation["site"] != int(gate[1]) or observation["source_op"] != int(gate[10]) or observation["value_id"] != gate_value or observation["source_span"] != expected_span or observation["branch_poisoned"] != 0 or observation["frail_branches"] != 0:
            fail(f"Join MIR observation identity mismatch: {name}")
        scalar_phi = ints(jmir["jsphi"][0])
        scalar_incoming = scalar_phi[4] if scalar_phi[3] == oracle["edge"] else scalar_phi[6]
        expected_scalar = [(source_hash, scalar_phi[0], scalar_phi[1], oracle["edge"], scalar_incoming, scalar_phi[2])]
        expected_memory_phis = [(source_hash, slot, 3, slot, oracle["edge"], slot if oracle["then"] else 2 + slot, 4 + slot) for slot in range(2)]
        if independent["scalar"] != expected_scalar or jmir_exec["scalar"] != expected_scalar:
            fail(f"scalar phi trace mismatch: {name}")
        if independent["memory_phis"] != expected_memory_phis or jmir_exec["memory_phis"] != expected_memory_phis:
            fail(f"Memory SSA phi trace mismatch: {name}")
        for slot, expected_value in enumerate(oracle["memory"].values()):
            expected = words(qd.Value(expected_value), qd)
            native = next((row for row in jmir_exec["memories"] if row["slot"] == slot), None)
            enir_native = next((row for row in enir_exec["memories"] if row["slot"] == slot), None)
            selected_store_version = slot if oracle["then"] else 2 + slot
            selected_store = ints(jmir["jmver"][selected_store_version])
            expected_site = selected_store[4]
            if native is None or enir_native is None or native["source_enir_hash"] != source_hash or native["version"] != 4 + slot or native["site"] != expected_site or native["source_op"] != expected_site or enir_native["site"] != expected_site or {key: native[key] for key in WORD_KEYS} != expected or {key: enir_native[key] for key in WORD_KEYS} != expected:
                fail(f"multi-slot final memory mismatch: {name}:{slot}")
        if jmir_exec["controls"] != independent["controls"]:
            fail(f"control trace mismatch: {name}")
        final = jmir_exec["final"]
        wanted_final = {"jmir_hash": rolling_hash(jmir_raw), "source_enir_hash": source_hash, "executed_instrs": independent["executed"], "observations": 1, "fuel_initial": oracle["fuel"], "fuel_left": independent["fuel"], "stop_kind": 0, "last_write": independent["last_write"], "branch_poisoned": 0, "frail_branches": 0}
        if final != wanted_final:
            fail(f"final receipt mismatch: {name}")
        receipt_programs.append({"name": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "enir_sha256": hashlib.sha256(enir_raw).hexdigest(), "jmir_sha256": hashlib.sha256(jmir_raw).hexdigest(), "relation_sha256": hashlib.sha256(relation).hexdigest(), "enir_execution_sha256": hashlib.sha256(enir_exec_raw).hexdigest(), "jmir_execution_sha256": hashlib.sha256(jmir_exec_raw).hexdigest(), "selected_edge": oracle["edge"], "scalar_incoming": scalar_incoming, "observable_bits": expected_words["value_bits"], "translation_valid": True, "independent_replay_exact": True})
    payload = {"schema": "madaros.e3d.multipred-scalar-memory-ssa.v1", "stage": "E3D-MULTIPRED-SCALAR-MEMORY-SSA-FULL", "programs": receipt_programs, "program_count": 2, "observation_count": 2, "diamond_cfg": True, "multiple_predecessors": True, "scalar_ssa_phi": True, "memory_ssa_phis": 2, "memory_slots": 2, "both_predecessors_executed": True, "independent_source_oracle": True, "independent_relation_validator": True, "independent_replay": True, "alias_analysis_supported": False, "machine_ir_used": False, "fallback": "none", "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(), "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "implementation_sha256": hashlib.sha256((args.root / "self-hosted/enir/mir_join.sio").read_bytes()).hexdigest()}
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E3D_MULTIPRED_SCALAR_MEMORY_SSA_VERIFY_PASS programs=2 observations=2 paths=then+else scalar_phi=explicit memory_slots=2 memory_phis=2 execution=source==enir==jmir==independent")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E3DError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E3D_MULTIPRED_SCALAR_MEMORY_SSA_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
