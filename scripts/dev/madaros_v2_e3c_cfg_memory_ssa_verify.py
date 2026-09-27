#!/usr/bin/env python3
"""Independent E3C CFG/Memory-SSA relation validator and replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROGRAMS = ("v2_fuel", "v2_loop", "v2_frail", "v2_mem_phi_zero", "v2_mem_phi_once")
MEMORY_PROGRAMS = {"v2_mem_phi_zero", "v2_mem_phi_once"}
MIR_OPCODE = {0: 20, 1: 27, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26, 8: 28, 9: 29}
WORD_KEYS = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
TAGS = {"cmir": 7, "ctype": 8, "cvalue": 15, "cprov": 8, "cblock": 12, "cbarg": 8, "cedge": 10, "cmver": 7, "cmphi": 10, "cinstr": 15, "cobs": 6, "cend": 11}
ORDER = {tag: index for index, tag in enumerate(TAGS)}
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E3CError(ValueError):
    pass


def fail(message: str) -> None:
    raise E3CError(message)


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


def parse_cmir(raw: bytes) -> dict[str, list[list[str]]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail("CFG MIR artifact is not canonical LF text")
    grouped = {tag: [] for tag in TAGS}
    prior = -1
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        row = line.split("|")
        if row[0] not in TAGS or len(row) != TAGS[row[0]] or any(field == "" for field in row):
            fail(f"malformed CFG MIR row {number}")
        order = ORDER[row[0]]
        if order < prior:
            fail(f"non-canonical CFG MIR section order at row {number}")
        prior = order
        grouped[row[0]].append(row)
    if len(grouped["cmir"]) != 1 or len(grouped["ctype"]) != 1 or len(grouped["cend"]) != 1:
        fail("CFG MIR singleton sections missing or duplicated")
    return grouped


def ints(row: list[str]) -> list[int]:
    return list(map(int, row[1:]))


def verify_relation(enir, cmir, enir_raw: bytes, name: str) -> None:
    if cmir["cmir"][0] != ["cmir", "3", "4", name, "2", enir["resource"][0][1], str(rolling_hash(enir_raw))]:
        fail(f"CFG MIR header/source binding mismatch: {name}")
    if cmir["ctype"][0][1:] != enir["type"][0][1:]:
        fail(f"logical type changed across CFG lowering: {name}")
    if len(cmir["cvalue"]) != len(enir["value"]) or len(cmir["cprov"]) != len(enir["prov"]):
        fail(f"value/provenance cardinality changed: {name}")
    for index, (source, lowered) in enumerate(zip(enir["value"], cmir["cvalue"])):
        if lowered[1:14] != source[1:] or int(lowered[14]) != index:
            fail(f"value relation mismatch {name}:{index}")
    for index, (source, lowered) in enumerate(zip(enir["prov"], cmir["cprov"])):
        if lowered[1:7] != source[1:] or int(lowered[7]) != index:
            fail(f"provenance relation mismatch {name}:{index}")
    for tag, source_tag in (("cblock", "block"), ("cbarg", "barg"), ("cedge", "edge")):
        if len(cmir[tag]) != len(enir[source_tag]):
            fail(f"{tag} cardinality mismatch: {name}")
        for index, (source, lowered) in enumerate(zip(enir[source_tag], cmir[tag])):
            if lowered[1:-1] != source[1:] or int(lowered[-1]) != index:
                fail(f"{tag} relation mismatch {name}:{index}")
    if len(cmir["cinstr"]) != len(enir["op"]):
        fail(f"instruction cardinality changed: {name}")
    stores: list[tuple[int, int]] = []
    loads: list[int] = []
    blocks = [ints(row) for row in enir["block"]]
    for index, (source, lowered) in enumerate(zip(enir["op"], cmir["cinstr"])):
        op = ints(source)
        got = ints(lowered)
        source_opcode = op[1]
        if source_opcode not in MIR_OPCODE:
            fail(f"unsupported opcode reached independent CFG validator: {source_opcode}")
        block_id = next((block[0] for block in blocks if block[1] <= index < block[1] + block[2]), -1)
        effect = {1: 2, 7: 1, 8: 3}.get(source_opcode, 0)
        trap = int(2 <= source_opcode <= 6)
        memory_in = memory_out = -1
        if source_opcode == 8:
            memory_out = len(stores)
            stores.append((index, block_id))
            if block_id == 2:
                memory_in = 2
        elif source_opcode == 1:
            memory_in = 2
            loads.append(index)
        wanted = [index, MIR_OPCODE[source_opcode], op[2], op[3], op[4], op[5], effect, trap, op[7], op[0], op[9], op[8], memory_in, memory_out]
        if got != wanted:
            fail(f"instruction relation mismatch {name}:{index}: {got} != {wanted}")
    if name in MEMORY_PROGRAMS:
        if len(stores) != 2 or [block for _, block in stores] != [0, 2] or len(loads) != 1 or not (blocks[3][1] <= loads[0] < blocks[3][1] + blocks[3][2]):
            fail(f"source memory graph mismatch: {name}: stores={stores} loads={loads}")
        wanted_versions = [[0, 0, 0, 0, stores[0][0], stores[0][0]], [1, 0, 0, 2, stores[1][0], stores[1][0]], [2, 0, 1, 1, -1, -1]]
        wanted_phi = [[0, 1, 0, 2, 0, 0, 3, 1, 1]]
        if [ints(row) for row in cmir["cmver"]] != wanted_versions or [ints(row) for row in cmir["cmphi"]] != wanted_phi:
            fail(f"Memory SSA graph mismatch: {name}")
    elif stores or loads or cmir["cmver"] or cmir["cmphi"]:
        fail(f"memory operation or metadata appeared in memory-free CFG: {name}")
    if len(cmir["cobs"]) != len(enir["obs"]) or cmir["cobs"][0][1:5] != enir["obs"][0][1:] or cmir["cobs"][0][5] != "0":
        fail(f"observation relation mismatch: {name}")
    footer = ints(cmir["cend"][0])
    wanted_footer = [1, len(cmir["cvalue"]), len(cmir["cprov"]), len(cmir["cblock"]), len(cmir["cbarg"]), len(cmir["cedge"]), len(cmir["cmver"]), len(cmir["cmphi"]), len(cmir["cinstr"]), len(cmir["cobs"])]
    if footer != wanted_footer:
        fail(f"CFG MIR footer mismatch: {name}")


def clone(value, qd):
    return qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned)


def words(value, qd, fuel_stop: bool = False) -> dict[str, int]:
    return {
        "value_bits": POISON_VALUE_BITS if value.poisoned else qd.fbits(value.val),
        **{f"error{i}_bits": qd.fbits(value.err[i]) for i in range(4)},
        "uncertainty_bits": POISON_U_BITS if value.poisoned else qd.fbits(value.uncertainty),
        "status": 1 if fuel_stop else value.poisoned,
    }


def replay(cmir, qd) -> dict[str, object]:
    descriptors = cmir["cvalue"]
    blocks = [ints(row) for row in cmir["cblock"]]
    bargs = [ints(row) for row in cmir["cbarg"]]
    edges = [ints(row) for row in cmir["cedge"]]
    phis = [ints(row) for row in cmir["cmphi"]]
    instructions = [ints(row) for row in cmir["cinstr"]]
    fuel = int(cmir["cmir"][0][5])
    runtime: dict[int, object] = {}
    memory: dict[int, object] = {}
    memory_version: dict[int, int] = {}
    memory_site: dict[int, int] = {}
    controls: list[tuple[int, int, int, int]] = []
    phi_trace: list[tuple[int, int, int]] = []
    block_id = 0
    executed = observations = frail_count = 0
    last_write = -1
    observed = None
    while True:
        block = blocks[block_id]
        for pc in range(block[1], block[1] + block[2]):
            if fuel == 0:
                return {"fuel": 0, "stop": 1, "executed": executed, "observations": 1, "last_write": last_write, "value": runtime[last_write], "frail": frail_count, "controls": controls, "phis": phi_trace, "memory": memory, "memory_version": memory_version, "memory_site": memory_site}
            fuel -= 1
            executed += 1
            _, opcode, result, _, a, b, _, trap, _, source_op, tick, slot, memory_in, memory_out = instructions[pc]
            if source_op != pc or tick != 1:
                fail(f"replay identity/tick mismatch at {pc}")
            if opcode == 20:
                value = qd.bits_float(int(descriptors[result][4]))
                runtime[result] = qd.Value(0.0 if value == 0.0 else value)
                last_write = result
            elif opcode in {21, 22, 23, 24}:
                x, y = runtime[a], runtime[b]
                if x.poisoned or y.poisoned or (opcode == 24 and y.val == 0.0):
                    runtime[result] = qd.Value(0.0, qd.qd_zero(), float("inf"), 1)
                else:
                    value = {21: lambda: x.val + y.val, 22: lambda: x.val - y.val, 23: lambda: x.val * y.val, 24: lambda: x.val / y.val}[opcode]()
                    value = 0.0 if value == 0.0 else value
                    tx = qd.qd_add(qd.qd_from_f64(x.val), x.err)
                    ty = qd.qd_add(qd.qd_from_f64(y.val), y.err)
                    truth = {21: qd.qd_add, 22: qd.qd_sub, 23: qd.qd_mul, 24: qd.qd_div}[opcode](tx, ty)
                    runtime[result] = qd.Value(value, qd.qd_sub(truth, qd.qd_from_f64(value)))
                if trap != 1:
                    fail(f"arithmetic trap policy missing at {pc}")
                last_write = result
            elif opcode == 29:
                runtime[result] = clone(runtime[a], qd)
                last_write = result
            elif opcode == 28:
                if memory_version.get(slot, -1) != memory_in:
                    fail(f"STORE version mismatch during replay at {pc}")
                memory[slot] = clone(runtime[a], qd)
                memory_version[slot] = memory_out
                memory_site[slot] = pc
            elif opcode == 27:
                if slot not in memory or memory_version.get(slot) != memory_in:
                    fail(f"LOAD version mismatch during replay at {pc}")
                runtime[result] = clone(memory[slot], qd)
                last_write = result
            elif opcode == 26:
                observed = clone(runtime[a], qd)
                observations += 1
            else:
                fail(f"unsupported replay opcode {opcode}")
        term, condition, edge0, edge1, tick = block[5], block[6], block[7], block[8], block[9]
        if tick:
            if fuel == 0:
                return {"fuel": 0, "stop": 1, "executed": executed, "observations": 1, "last_write": last_write, "value": runtime[last_write], "frail": frail_count, "controls": controls, "phis": phi_trace, "memory": memory, "memory_version": memory_version, "memory_site": memory_site}
            fuel -= 1
            executed += 1
        if term == 0:
            if observed is None or observations != 1:
                fail("halt without exactly one observation")
            return {"fuel": fuel, "stop": 0, "executed": executed, "observations": observations, "last_write": last_write, "value": observed, "frail": frail_count, "controls": controls, "phis": phi_trace, "memory": memory, "memory_version": memory_version, "memory_site": memory_site}
        edge_id, taken, frail = edge0, 1, 0
        if term == 2:
            condition_value = runtime[condition]
            band = max(abs(condition_value.err[0]), abs(condition_value.uncertainty))
            frail = int(band != 0.0 and abs(condition_value.val) <= band)
            frail_count += frail
            taken = int(condition_value.val == 0.0)
            edge_id = edge0 if taken else edge1
        edge = edges[edge_id]
        controls.append((block_id, block[10], term, condition, edge_id, edge[8], edge[2], taken, 0, frail))
        for phi in phis:
            if phi[1] == edge[2]:
                incoming = phi[5] if phi[4] == edge_id else phi[7] if phi[6] == edge_id else -1
                if incoming < 0 or memory_version.get(phi[2]) != incoming:
                    fail(f"phi incoming version mismatch at edge {edge_id}")
                phi_trace.append((int(cmir["cmir"][0][6]), phi[0], phi[1], phi[2], edge_id, incoming, phi[3]))
                memory_version[phi[2]] = phi[3]
        incoming_values = [clone(runtime[edge[4 + ordinal]], qd) for ordinal in range(edge[3])]
        target = blocks[edge[2]]
        for ordinal, value in enumerate(incoming_values):
            runtime[bargs[target[3] + ordinal][3]] = value
        block_id = edge[2]


def parse_execution(raw: bytes, prefix: str) -> dict[str, object]:
    controls: list[tuple[int, ...]] = []
    phis: list[tuple[int, ...]] = []
    observation = memory = final = None
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith(prefix + "-control|"):
            controls.append(tuple(int(fields[key]) for key in ("block", "source_block", "term", "condition", "edge", "source_edge", "to", "taken", "poisoned", "frail")))
        elif line.startswith(prefix + "-memory-phi|"):
            phis.append(tuple(int(fields[key]) for key in ("source_enir_hash", "phi", "block", "slot", "edge", "incoming_version", "result_version")))
        elif line.startswith(prefix + "-exec|") or line.startswith(prefix + "-fuel|"):
            identity = ("source_enir_hash", "ordinal", "site", "source_op", "value_id", "source_span") if line.startswith(prefix + "-exec|") else ("source_enir_hash", "ordinal", "last_value_id", "source_span")
            observation = {key: int(fields[key]) for key in (*identity, *WORD_KEYS, "branch_poisoned", "frail_branches")}
        elif line.startswith(prefix + "-memory|"):
            memory = {key: int(fields[key]) for key in ("source_enir_hash", "slot", "version", "site", "source_op", *WORD_KEYS)}
        elif line.startswith(prefix + "-exec-ok|"):
            keys = ("cmir_hash", "source_enir_hash", "executed_instrs", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches") if prefix == "cmir" else ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")
            final = {key: int(fields[key]) for key in keys}
    if observation is None or final is None:
        fail(f"{prefix} execution omitted observation/final receipt")
    return {"controls": controls, "phis": phis, "observation": observation, "memory": memory, "final": final}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    e2c = load_module(args.root / "scripts/dev/madaros_v2_e2c_enir_fuel_blockargs_verify.py", "e3c_e2c")
    e2d = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e3c_e2d")
    e2g = load_module(args.root / "scripts/dev/madaros_v2_e2g_enir_fuel_control_frail_verify.py", "e3c_e2g")
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e3c_qd")
    implementation = (args.root / "self-hosted/enir/mir_cfg.sio").read_text(encoding="utf-8")
    for forbidden in ("enir_interpret(", "use eisa::", "eisax_encode(", "eisa_compile(", "native::machine_ir"):
        if forbidden in implementation:
            fail(f"CFG MIR implementation depends on forbidden surface: {forbidden}")
    if any(name in implementation for name in PROGRAMS):
        fail("CFG MIR implementation contains case-specific program names")
    metron = e2g.parse_metron(args.oracle, args.corpus, e2d)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_programs = []
    for name in PROGRAMS:
        source = args.root / "tools/eisa" / f"eisa_enir_{name}.eisa"
        enir_raw = run([str(args.driver), "lower-v2", str(source)])
        enir_path = args.out_dir / f"{name}.enir"
        enir_path.write_bytes(enir_raw)
        enir = e2c.parse_enir(enir_raw)
        cmir_raw = run([str(args.driver), "lower-cfg-mir", str(enir_path)])
        if run([str(args.driver), "lower-cfg-mir", str(enir_path)]) != cmir_raw:
            fail(f"nondeterministic CFG MIR lowering: {name}")
        cmir_path = args.out_dir / f"{name}.cmir"
        cmir_path.write_bytes(cmir_raw)
        cmir = parse_cmir(cmir_raw)
        verify_relation(enir, cmir, enir_raw, name)
        relation = run([str(args.driver), "validate-cfg-mir", str(enir_path), str(cmir_path)])
        run([str(args.driver), "verify-cfg-mir", str(cmir_path)])
        enir_exec_raw = run([str(args.driver), "run", str(enir_path)])
        cmir_exec_raw = run([str(args.driver), "run-cfg-mir", str(cmir_path)])
        enir_exec = e2g.parse_native(enir_exec_raw)
        cmir_exec = parse_execution(cmir_exec_raw, "cmir")
        independent = replay(cmir, qd)
        expected_words = words(independent["value"], qd, independent["stop"] == 1)
        if {key: cmir_exec["observation"][key] for key in WORD_KEYS} != expected_words:
            fail(f"CFG MIR/independent observable mismatch: {name}")
        if name not in MEMORY_PROGRAMS:
            if {key: enir_exec["observation"][key] for key in WORD_KEYS} != expected_words:
                fail(f"ENIR/CFG MIR observable mismatch: {name}")
            if {key: metron[name][key] for key in WORD_KEYS} != expected_words:
                fail(f"METRON/CFG MIR observable mismatch: {name}")
            enir_control_projection = [(row[0], row[4], row[7], row[9]) for row in independent["controls"]]
            if cmir_exec["controls"] != independent["controls"] or enir_exec["controls"] != enir_control_projection:
                fail(f"control trace mismatch: {name}")
        else:
            expected_value = 7.25 if name.endswith("zero") else 8.5
            if expected_words["value_bits"] != qd.fbits(expected_value):
                fail(f"Memory SSA selected wrong predecessor value: {name}")
            if {key: enir_exec["observation"][key] for key in WORD_KEYS} != expected_words:
                fail(f"ENIR/CFG MIR memory observable mismatch: {name}")
            source_hash = rolling_hash(enir_raw)
            expected_phis = [(source_hash, 0, 1, 0, 0, 0, 2)] if name.endswith("zero") else [(source_hash, 0, 1, 0, 0, 0, 2), (source_hash, 0, 1, 0, 3, 1, 2)]
            if cmir_exec["phis"] != expected_phis or independent["phis"] != expected_phis:
                fail(f"Memory phi trace mismatch: {name}")
            memory = cmir_exec["memory"]
            selected_version = 0 if name.endswith("zero") else 1
            expected_site = ints(cmir["cmver"][selected_version])[4]
            if memory is None or memory["source_enir_hash"] != source_hash or memory["slot"] != 0 or memory["version"] != 2 or memory["site"] != expected_site or memory["source_op"] != expected_site or {key: memory[key] for key in WORD_KEYS} != expected_words:
                fail(f"final Memory SSA receipt mismatch: {name}")
        source_hash = rolling_hash(enir_raw)
        observation = cmir_exec["observation"]
        if observation["source_enir_hash"] != source_hash or observation["ordinal"] != 0 or observation["branch_poisoned"] != 0 or observation["frail_branches"] != independent["frail"]:
            fail(f"CFG MIR observation source identity mismatch: {name}")
        if independent["stop"] == 0:
            gate = next(row for row in cmir["cinstr"] if int(row[2]) == 26)
            value_id = int(gate[5])
            if observation["site"] != int(gate[1]) or observation["source_op"] != int(gate[10]) or observation["value_id"] != value_id or observation["source_span"] != int(cmir["cprov"][value_id][2]):
                fail(f"CFG MIR gate identity mismatch: {name}")
        else:
            value_id = independent["last_write"]
            if observation["last_value_id"] != value_id or observation["source_span"] != int(cmir["cprov"][value_id][2]):
                fail(f"CFG MIR fuel identity mismatch: {name}")
        final = cmir_exec["final"]
        wanted_final = {"cmir_hash": rolling_hash(cmir_raw), "source_enir_hash": source_hash, "executed_instrs": independent["executed"], "observations": independent["observations"], "fuel_initial": int(cmir["cmir"][0][5]), "fuel_left": independent["fuel"], "stop_kind": independent["stop"], "last_write": independent["last_write"], "branch_poisoned": 0, "frail_branches": independent["frail"]}
        if final != wanted_final:
            fail(f"CFG MIR final state mismatch: {name}: {final} != {wanted_final}")
        receipt_programs.append({"name": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "enir_sha256": hashlib.sha256(enir_raw).hexdigest(), "cmir_sha256": hashlib.sha256(cmir_raw).hexdigest(), "relation_receipt_sha256": hashlib.sha256(relation).hexdigest(), "enir_execution_sha256": hashlib.sha256(enir_exec_raw).hexdigest(), "cmir_execution_sha256": hashlib.sha256(cmir_exec_raw).hexdigest(), "blocks": len(cmir["cblock"]), "edges": len(cmir["cedge"]), "memory_versions": len(cmir["cmver"]), "memory_phis": len(cmir["cmphi"]), "translation_valid": True, "independent_replay_exact": True})
    payload = {"schema": "madaros.e3c.cfg-memory-ssa.v1", "stage": "E3C-CFG-MEMORY-SSA-FULL", "programs": receipt_programs, "program_count": 5, "observation_count": 5, "explicit_cfg": True, "block_arguments": True, "explicit_edges": True, "memory_ssa": True, "loop_header_phi": True, "zero_trip_and_backedge": True, "independent_relation_validator": True, "independent_replay": True, "enir_cmir_exact": True, "metron_exact_for_canonical_e2g": True, "abi_independent": True, "machine_ir_used": False, "irreducible_cfg_supported": False, "alias_analysis_supported": False, "fallback": "none", "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(), "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "implementation_sha256": hashlib.sha256((args.root / "self-hosted/enir/mir_cfg.sio").read_bytes()).hexdigest()}
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E3C_CFG_MEMORY_SSA_VERIFY_PASS programs=5 observations=5 cfg=4-block+blockargs+edges memory_ssa=store-versions+loop-phi execution=enir==cmir==independent metron=e2g-exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E3CError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E3C_CFG_MEMORY_SSA_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
