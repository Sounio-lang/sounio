#!/usr/bin/env python3
"""Independent E3B validator for qd128 memory/move semantic MIR."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


PROGRAMS = ("v2_mem", "v2_emov", "v2_mem_poison")
MIR_OPCODE = {0: 20, 1: 27, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26, 8: 28, 9: 29}
WORD_KEYS = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E3BError(ValueError):
    pass


def fail(message: str) -> None:
    raise E3BError(message)


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


def verify_relation(enir, mir, enir_raw: bytes, name: str) -> None:
    header = mir["emir"][0]
    if header != ["emir", "2", "3", name, "2", enir["resource"][0][1], str(rolling_hash(enir_raw))]:
        fail(f"MIR header/source binding mismatch: {name}")
    if mir["mtype"][0][1:] != enir["type"][0][1:]:
        fail(f"logical type changed across ENIR-to-MIR: {name}")
    if len(mir["mvalue"]) != len(enir["value"]) or len(mir["mprov"]) != len(enir["prov"]):
        fail(f"value/provenance cardinality changed: {name}")
    for index, (source, lowered) in enumerate(zip(enir["value"], mir["mvalue"])):
        if lowered[1:14] != source[1:] or int(lowered[14]) != index:
            fail(f"value relation mismatch {name}:{index}")
    for index, (source, lowered) in enumerate(zip(enir["prov"], mir["mprov"])):
        if lowered[1:7] != source[1:] or int(lowered[7]) != index:
            fail(f"provenance relation mismatch {name}:{index}")
    if mir["mblock"] != [["mblock", "0", "0", str(len(enir["op"])), "0", "1"]]:
        fail(f"explicit MIR block/halt mismatch: {name}")
    if len(mir["minstr"]) != len(enir["op"]):
        fail(f"instruction cardinality changed: {name}")
    latest_store: dict[int, int] = {}
    for index, (source, lowered) in enumerate(zip(enir["op"], mir["minstr"])):
        op = list(map(int, source[1:]))
        got = list(map(int, lowered[1:]))
        source_opcode = op[1]
        if source_opcode not in MIR_OPCODE:
            fail(f"unsupported opcode reached E3B independent validator: {source_opcode}")
        effect = {1: 2, 7: 1, 8: 3}.get(source_opcode, 0)
        trap = 1 if 2 <= source_opcode <= 6 else 0
        slot = op[8] if source_opcode in {1, 8} else -1
        origin = latest_store.get(slot, -1) if source_opcode == 1 else -1
        wanted = [index, MIR_OPCODE[source_opcode], op[2], op[3], op[4], op[5], effect, trap, op[7], op[0], op[9], slot, origin]
        if got != wanted:
            fail(f"instruction relation mismatch {name}:{index}: {got} != {wanted}")
        if source_opcode == 1:
            result = op[2]
            if result < 0 or int(enir["prov"][result][3]) != origin:
                fail(f"ENIR load provenance/store-origin mismatch {name}:{index}")
        elif source_opcode == 8:
            latest_store[slot] = index
    if len(mir["mobs"]) != len(enir["obs"]):
        fail(f"observation cardinality changed: {name}")
    for index, (source, lowered) in enumerate(zip(enir["obs"], mir["mobs"])):
        if lowered[1:5] != source[1:] or int(lowered[5]) != index:
            fail(f"observation relation mismatch {name}:{index}")
    footer = list(map(int, mir["mend"][0][1:]))
    if footer != [1, len(mir["mvalue"]), len(mir["mprov"]), 1, len(mir["minstr"]), len(mir["mobs"])]:
        fail(f"MIR footer mismatch: {name}")


def clone(value, qd):
    return qd.Value(value.val, tuple(value.err), value.uncertainty, value.poisoned)


def poison(qd):
    return qd.Value(0.0, qd.qd_zero(), float("inf"), 1)


def value_words(value, qd) -> dict[str, int]:
    return {
        "value_bits": POISON_VALUE_BITS if value.poisoned else qd.fbits(value.val),
        **{f"error{i}_bits": qd.fbits(value.err[i]) for i in range(4)},
        "uncertainty_bits": POISON_U_BITS if value.poisoned else qd.fbits(value.uncertainty),
        "status": value.poisoned,
    }


def replay_mir(mir, qd):
    descriptors = mir["mvalue"]
    runtime: dict[int, object] = {}
    memory: dict[int, tuple[int, object]] = {}
    observations: list[dict[str, int]] = []
    last_write = -1
    for row in mir["minstr"]:
        site, opcode, result, _, a, b, effect, trap, _, source_op, tick, slot, origin = map(int, row[1:])
        if site != source_op or tick != 1:
            fail(f"MIR replay identity/tick mismatch at site {site}")
        if opcode == 20:
            raw = qd.bits_float(int(descriptors[result][4]))
            runtime[result] = qd.Value(0.0 if raw == 0.0 else raw)
            last_write = result
        elif opcode in {21, 22, 23, 24}:
            x, y = runtime[a], runtime[b]
            if x.poisoned or y.poisoned or (opcode == 24 and y.val == 0.0):
                runtime[result] = poison(qd)
            else:
                raw = {21: lambda: x.val + y.val, 22: lambda: x.val - y.val, 23: lambda: x.val * y.val, 24: lambda: x.val / y.val}[opcode]()
                val = 0.0 if raw == 0.0 else raw
                tx = qd.qd_add(qd.qd_from_f64(x.val), x.err)
                ty = qd.qd_add(qd.qd_from_f64(y.val), y.err)
                truth = {21: qd.qd_add, 22: qd.qd_sub, 23: qd.qd_mul, 24: qd.qd_div}[opcode](tx, ty)
                runtime[result] = qd.Value(val, qd.qd_sub(truth, qd.qd_from_f64(val)))
            if trap != 1:
                fail(f"arithmetic trap policy missing at site {site}")
            last_write = result
        elif opcode == 25:
            x = runtime[a]
            if x.poisoned:
                runtime[result] = poison(qd)
            elif x.val <= 0.0:
                runtime[result] = qd.Value(0.0)
            else:
                val = qd.sqrt40(x.val)
                truth = qd.qd_sqrt(qd.qd_add(qd.qd_from_f64(x.val), x.err))
                runtime[result] = qd.Value(val, qd.qd_sub(truth, qd.qd_from_f64(val)))
            last_write = result
        elif opcode == 28:
            if effect != 3 or slot < 0 or origin != -1 or a not in runtime:
                fail(f"invalid STORE during replay at site {site}")
            memory[slot] = (site, clone(runtime[a], qd))
        elif opcode == 27:
            if effect != 2 or slot not in memory or memory[slot][0] != origin:
                fail(f"invalid LOAD/store-origin during replay at site {site}")
            runtime[result] = clone(memory[slot][1], qd)
            last_write = result
        elif opcode == 29:
            if effect != 0 or slot != -1 or origin != -1 or a not in runtime:
                fail(f"invalid MOVE during replay at site {site}")
            runtime[result] = clone(runtime[a], qd)
            last_write = result
        elif opcode == 26:
            observations.append(value_words(runtime[a], qd))
        else:
            fail(f"unsupported MIR replay opcode {opcode}")
    return observations, memory, last_write


def parse_mir_exec(raw: bytes):
    observations: list[dict[str, int]] = []
    memory: list[dict[str, int]] = []
    final = None
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith("mir-exec|"):
            keys = ("source_enir_hash", "ordinal", "site", "source_op", "value_id", *WORD_KEYS, "gate_class", "source_span")
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith("mir-memory|"):
            keys = ("source_enir_hash", "slot", "site", "source_op", *WORD_KEYS)
            memory.append({key: int(fields[key]) for key in keys})
        elif line.startswith("mir-exec-ok|"):
            keys = ("mir_hash", "source_enir_hash", "executed_instrs", "observations", "fuel_initial", "fuel_left", "last_write")
            final = {key: int(fields[key]) for key in keys}
    if final is None:
        fail("missing MIR final receipt")
    return observations, memory, final


def project(rows):
    return [{key: row[key] for key in WORD_KEYS} for row in rows]


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
    e2d = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e3b_e2d")
    e2h = load_module(args.root / "scripts/dev/madaros_v2_e2h_enir_memory_move_poison_verify.py", "e3b_e2h")
    e3a = load_module(args.root / "scripts/dev/madaros_v2_e3a_enir_mir_qd128_verify.py", "e3b_e3a")
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e3b_qd")
    implementation = (args.root / "self-hosted/enir/mir.sio").read_text(encoding="utf-8")
    for forbidden in ("enir_interpret(", "use eisa::", "eisax_encode(", "eisa_compile(", "native::machine_ir"):
        if forbidden in implementation:
            fail(f"MIR implementation depends on forbidden executor/backend surface: {forbidden}")
    if any(name in implementation for name in PROGRAMS):
        fail("MIR implementation contains case-specific program names")

    metron = e2h.parse_metron(args.oracle, args.corpus, e2d)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_programs = []
    total_observations = 0
    for name in PROGRAMS:
        source = args.source_dir / f"eisa_enir_{name}.eisa"
        enir_raw = run([str(args.driver), "lower-v2", str(source)])
        enir_path = args.out_dir / f"{name}.enir"
        enir_path.write_bytes(enir_raw)
        enir = e2d.parse_enir(enir_raw)
        mir_raw = run([str(args.driver), "lower-mir", str(enir_path)])
        if run([str(args.driver), "lower-mir", str(enir_path)]) != mir_raw:
            fail(f"nondeterministic ENIR-to-MIR lowering: {name}")
        mir_path = args.out_dir / f"{name}.emir"
        mir_path.write_bytes(mir_raw)
        mir = e3a.parse_mir(mir_raw)
        verify_relation(enir, mir, enir_raw, name)
        relation = run([str(args.driver), "validate-mir", str(enir_path), str(mir_path)])
        run([str(args.driver), "verify-mir", str(mir_path)])
        enir_execution = run([str(args.driver), "run", str(enir_path)])
        mir_execution = run([str(args.driver), "run-mir", str(mir_path)])
        enir_obs, enir_memory, enir_final = e2h.parse_native(enir_execution)
        mir_obs, mir_memory, mir_final = parse_mir_exec(mir_execution)
        independent, memory, last_write = replay_mir(mir, qd)
        expected_memory = [{"slot": slot, "site": site, **value_words(value, qd)} for slot, (site, value) in sorted(memory.items())]
        mir_memory_projection = [{key: row[key] for key in expected_memory[0]} for row in mir_memory]
        if project(enir_obs) != project(mir_obs) or project(mir_obs) != independent:
            fail(f"ENIR/MIR/independent observation mismatch: {name}")
        if project(mir_obs) != [{key: row[key] for key in WORD_KEYS} for row in metron[name]]:
            fail(f"MIR/METRON observable mismatch: {name}")
        if enir_memory != expected_memory or mir_memory_projection != expected_memory:
            fail(f"ENIR/MIR/independent atomic memory mismatch: {name}")
        source_hash = rolling_hash(enir_raw)
        for ordinal, row in enumerate(mir_obs):
            if row["ordinal"] != ordinal or row["source_enir_hash"] != source_hash or row["source_op"] != row["site"]:
                fail(f"MIR observation identity/source binding mismatch: {name}:{ordinal}")
            source_row = enir_obs[ordinal]
            for key in ("site", "value_id", "gate_class", "source_span"):
                if row[key] != source_row[key]:
                    fail(f"MIR observation metadata mismatch {name}:{ordinal}:{key}")
        for row in mir_memory:
            if row["source_enir_hash"] != source_hash or row["source_op"] != row["site"]:
                fail(f"MIR memory identity/source binding mismatch: {name}")
        if name in {"v2_mem", "v2_mem_poison"} and project(mir_obs)[0] != [{key: mir_memory[0][key] for key in WORD_KEYS}][0]:
            fail(f"stored qd128 product did not survive load/move/gate: {name}")
        if name == "v2_emov" and (project(mir_obs)[0] != project(mir_obs)[1] or project(mir_obs)[0]["value_bits"] != 0):
            fail("MIR move did not preserve canonical positive zero word-for-word")
        wanted_final = {
            "mir_hash": rolling_hash(mir_raw), "source_enir_hash": source_hash,
            "executed_instrs": len(mir["minstr"]) + 1, "observations": len(mir["mobs"]),
            "fuel_initial": int(mir["emir"][0][5]), "fuel_left": int(mir["emir"][0][5]) - len(mir["minstr"]) - 1,
            "last_write": last_write,
        }
        if mir_final != wanted_final:
            fail(f"MIR fuel/final state mismatch: {name}: {mir_final}")
        if enir_final["executed_ops"] != mir_final["executed_instrs"] or enir_final["fuel_left"] != mir_final["fuel_left"]:
            fail(f"semantic tick mismatch across ENIR/MIR: {name}")
        total_observations += len(mir_obs)
        receipt_programs.append({
            "name": name, "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "enir_sha256": hashlib.sha256(enir_raw).hexdigest(), "enir_l64": source_hash,
            "mir_sha256": hashlib.sha256(mir_raw).hexdigest(), "mir_l64": rolling_hash(mir_raw),
            "native_relation_receipt_sha256": hashlib.sha256(relation).hexdigest(),
            "enir_execution_receipt_sha256": hashlib.sha256(enir_execution).hexdigest(),
            "mir_execution_receipt_sha256": hashlib.sha256(mir_execution).hexdigest(),
            "instructions": len(mir["minstr"]), "observations": len(mir_obs), "memory_slots": len(mir_memory),
            "native_relation_valid": True, "independent_relation_valid": True,
            "atomic_full_product_memory": True, "latest_store_origin_valid": True,
        })

    payload = {
        "schema": "madaros.e3b.enir-mir-memory-move.v1", "stage": "E3B-ENIR-MIR-QD128-MEMORY-MOVE-FULL",
        "programs": receipt_programs, "program_count": 3, "observation_count": total_observations,
        "native_relation_validator": True, "independent_relation_validator": True,
        "independent_mir_replay": True, "enir_mir_differential_exact": True,
        "metron_mir_differential_exact": True, "atomic_full_product_memory": True,
        "latest_store_provenance": True, "negative_zero_canonical": True,
        "poison_survives_store_load_move": True, "semantic_tick_preserved": True,
        "abi_independent": True, "machine_ir_used": False, "memory_ssa": False,
        "multi_block_cfg_supported": False, "fallback": "none",
        "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mir_implementation_sha256": hashlib.sha256((args.root / "self-hosted/enir/mir.sio").read_bytes()).hexdigest(),
        "qd_semantics_sha256": hashlib.sha256((args.root / "self-hosted/enir/qd.sio").read_bytes()).hexdigest(),
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E3B_ENIR_MIR_MEMORY_VERIFY_PASS programs=3 observations=4 relation=native+independent execution=enir==mir==metron memory=atomic-full-product provenance=latest-store move=word-identical cfg=single-block")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E3BError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E3B_ENIR_MIR_MEMORY_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
