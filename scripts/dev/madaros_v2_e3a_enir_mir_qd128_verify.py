#!/usr/bin/env python3
"""Independent E3A ENIR-to-MIR translation validator and differential replay."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROGRAMS = ("v2_const_gate", "v2_add", "v2_sub", "v2_mul", "v2_div", "v2_sqrt")
MIR_OPCODE = {0: 20, 2: 21, 3: 22, 4: 23, 5: 24, 6: 25, 7: 26}
TAGS = {"emir": 7, "mtype": 8, "mvalue": 15, "mprov": 8, "mblock": 6, "minstr": 14, "mobs": 6, "mend": 7}
ORDER = {tag: index for index, tag in enumerate(TAGS)}
POISON_VALUE_BITS = 0x7FF0000000000001
POISON_U_BITS = 0x7FF0000000000000


class E3AError(ValueError):
    pass


def fail(message: str) -> None:
    raise E3AError(message)


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


def parse_mir(raw: bytes) -> dict[str, list[list[str]]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail("MIR artifact is not canonical LF text")
    grouped = {tag: [] for tag in TAGS}
    prior = -1
    for number, line in enumerate(raw.decode("ascii").splitlines(), 1):
        row = line.split("|")
        if row[0] not in TAGS or len(row) != TAGS[row[0]] or any(field == "" for field in row):
            fail(f"malformed MIR row {number}")
        order = ORDER[row[0]]
        if order < prior:
            fail(f"non-canonical MIR section order at row {number}")
        prior = order
        grouped[row[0]].append(row)
    if len(grouped["emir"]) != 1 or len(grouped["mtype"]) != 1 or len(grouped["mblock"]) != 1 or len(grouped["mend"]) != 1:
        fail("MIR singleton sections missing or duplicated")
    return grouped


def verify_relation(enir: dict[str, list[list[str]]], mir: dict[str, list[list[str]]], enir_raw: bytes, name: str) -> None:
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
    for index, (source, lowered) in enumerate(zip(enir["op"], mir["minstr"])):
        op = list(map(int, source[1:]))
        got = list(map(int, lowered[1:]))
        source_opcode = op[1]
        if source_opcode not in MIR_OPCODE:
            fail(f"unsupported opcode reached E3A independent validator: {source_opcode}")
        effect = 1 if source_opcode == 7 else 0
        trap = 1 if 2 <= source_opcode <= 6 else 0
        wanted = [index, MIR_OPCODE[source_opcode], op[2], op[3], op[4], op[5], effect, trap, op[7], op[0], op[9], -1, -1]
        if got != wanted:
            fail(f"instruction relation mismatch {name}:{index}: {got} != {wanted}")
    if len(mir["mobs"]) != len(enir["obs"]):
        fail(f"observation cardinality changed: {name}")
    for index, (source, lowered) in enumerate(zip(enir["obs"], mir["mobs"])):
        if lowered[1:5] != source[1:] or int(lowered[5]) != index:
            fail(f"observation relation mismatch {name}:{index}")
    footer = list(map(int, mir["mend"][0][1:]))
    wanted_footer = [1, len(mir["mvalue"]), len(mir["mprov"]), 1, len(mir["minstr"]), len(mir["mobs"])]
    if footer != wanted_footer:
        fail(f"MIR footer mismatch: {name}")


@dataclass
class Value:
    val: float
    err: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    uncertainty: float = 0.0
    poisoned: int = 0


def words(value: Value, qd) -> dict[str, int]:
    return {
        "value_bits": POISON_VALUE_BITS if value.poisoned else qd.fbits(value.val),
        **{f"error{i}_bits": qd.fbits(value.err[i]) for i in range(4)},
        "uncertainty_bits": POISON_U_BITS if value.poisoned else qd.fbits(value.uncertainty),
        "status": value.poisoned,
    }


def true_qd(value: Value, qd):
    return qd.qd_add(qd.qd_from_f64(value.val), value.err)


def replay_mir(mir: dict[str, list[list[str]]], qd) -> tuple[list[dict[str, int]], int]:
    descriptors = mir["mvalue"]
    runtime: dict[int, Value] = {}
    observations: list[dict[str, int]] = []
    last_write = -1
    for row in mir["minstr"]:
        _, opcode, result, _, a, b, _, trap, _, _, tick, slot, origin = map(int, row[1:])
        if slot != -1 or origin != -1:
            fail("E3A arithmetic replay saw memory metadata")
        if tick != 1:
            fail("independent MIR replay saw non-unit semantic tick")
        if opcode == 20:
            val = qd.bits_float(int(descriptors[result][4]))
            runtime[result] = Value(0.0 if val == 0.0 else val)
            last_write = result
        elif opcode in {21, 22, 23, 24}:
            x, y = runtime[a], runtime[b]
            if x.poisoned or y.poisoned or (opcode == 24 and y.val == 0.0):
                runtime[result] = Value(0.0, poisoned=1)
            else:
                raw = {21: lambda: x.val + y.val, 22: lambda: x.val - y.val, 23: lambda: x.val * y.val, 24: lambda: x.val / y.val}[opcode]()
                val = 0.0 if raw == 0.0 else raw
                truth = {21: qd.qd_add, 22: qd.qd_sub, 23: qd.qd_mul, 24: qd.qd_div}[opcode](true_qd(x, qd), true_qd(y, qd))
                runtime[result] = Value(val, qd.qd_sub(truth, qd.qd_from_f64(val)))
            if trap != 1:
                fail("arithmetic instruction lacks explicit poison trap policy")
            last_write = result
        elif opcode == 25:
            x = runtime[a]
            if x.poisoned:
                runtime[result] = Value(0.0, poisoned=1)
            elif x.val <= 0.0:
                runtime[result] = Value(0.0)
            else:
                val = qd.sqrt40(x.val)
                runtime[result] = Value(val, qd.qd_sub(qd.qd_sqrt(true_qd(x, qd)), qd.qd_from_f64(val)))
            if trap != 1:
                fail("sqrt lacks explicit poison trap policy")
            last_write = result
        elif opcode == 26:
            observations.append(words(runtime[a], qd))
        else:
            fail(f"unsupported MIR replay opcode {opcode}")
    return observations, last_write


def parse_exec(raw: bytes, prefix: str) -> tuple[list[dict[str, int]], dict[str, int]]:
    observations: list[dict[str, int]] = []
    final = None
    for line in raw.decode("ascii").splitlines():
        fields = dict(part.split("=", 1) for part in line.split("|")[1:])
        if line.startswith(prefix + "|"):
            keys = ("ordinal", "site", "value_id", "value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status", "gate_class", "source_span")
            if prefix == "mir-exec":
                keys = (*keys, "source_enir_hash", "source_op")
            observations.append({key: int(fields[key]) for key in keys})
        elif line.startswith(prefix + "-ok|"):
            final = {key: int(fields[key]) for key in fields if key not in {"module"}}
    if final is None:
        fail(f"missing {prefix} final receipt")
    return observations, final


def projection(rows: list[dict[str, int]]) -> list[dict[str, int]]:
    keys = ("value_bits", "error0_bits", "error1_bits", "error2_bits", "error3_bits", "uncertainty_bits", "status")
    return [{key: row[key] for key in keys} for row in rows]


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
    e2d = load_module(args.root / "scripts/dev/madaros_v2_e2d_enir_rump_dd_verify.py", "e3a_e2d")
    qd = load_module(args.root / "scripts/dev/madaros_v2_e2e_enir_qd128_verify.py", "e3a_qd")
    implementation = (args.root / "self-hosted/enir/mir.sio").read_text(encoding="utf-8")
    for forbidden in ("enir_interpret(", "use eisa::", "eisax_encode(", "eisa_compile(", "native::machine_ir"):
        if forbidden in implementation:
            fail(f"MIR implementation depends on forbidden executor/backend surface: {forbidden}")
    if any(name in implementation for name in PROGRAMS):
        fail("MIR implementation contains case-specific program names")

    metron = qd.parse_metron(args.oracle, args.corpus, e2d)
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
        mir = parse_mir(mir_raw)
        verify_relation(enir, mir, enir_raw, name)
        relation = run([str(args.driver), "validate-mir", str(enir_path), str(mir_path)])
        if not relation.startswith(b"mir-relation-ok|"):
            fail(f"native relational validator emitted no receipt: {name}")
        run([str(args.driver), "verify-mir", str(mir_path)])
        enir_execution = run([str(args.driver), "run", str(enir_path)])
        mir_execution = run([str(args.driver), "run-mir", str(mir_path)])
        enir_obs, enir_final = parse_exec(enir_execution, "enir-exec")
        mir_obs, mir_final = parse_exec(mir_execution, "mir-exec")
        independent, last_write = replay_mir(mir, qd)
        for ordinal, row in enumerate(mir_obs):
            if row["ordinal"] != ordinal or row["source_enir_hash"] != rolling_hash(enir_raw) or row["source_op"] != row["site"]:
                fail(f"MIR runtime identity/source binding mismatch: {name}:{ordinal}")
        if projection(enir_obs) != projection(mir_obs) or projection(mir_obs) != independent:
            fail(f"ENIR/MIR/independent execution mismatch: {name}")
        metron_words = [{key: metron[name][key] for key in independent[0]}]
        if independent != metron_words:
            fail(f"MIR/METRON observable mismatch: {name}")
        if mir_final["executed_instrs"] != len(mir["minstr"]) + 1 or mir_final["fuel_left"] != int(mir["emir"][0][5]) - len(mir["minstr"]) - 1 or mir_final["last_write"] != last_write:
            fail(f"MIR fuel/final state mismatch: {name}")
        if mir_final["mir_hash"] != rolling_hash(mir_raw) or mir_final["source_enir_hash"] != rolling_hash(enir_raw):
            fail(f"MIR final receipt hash binding mismatch: {name}")
        if enir_final["executed_ops"] != mir_final["executed_instrs"] or enir_final["fuel_left"] != mir_final["fuel_left"]:
            fail(f"semantic tick mismatch across ENIR/MIR: {name}")
        total_observations += len(mir_obs)
        receipt_programs.append({
            "name": name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "enir_sha256": hashlib.sha256(enir_raw).hexdigest(),
            "enir_l64": rolling_hash(enir_raw),
            "mir_sha256": hashlib.sha256(mir_raw).hexdigest(),
            "mir_l64": rolling_hash(mir_raw),
            "native_relation_receipt_sha256": hashlib.sha256(relation).hexdigest(),
            "enir_execution_receipt_sha256": hashlib.sha256(enir_execution).hexdigest(),
            "mir_execution_receipt_sha256": hashlib.sha256(mir_execution).hexdigest(),
            "instructions": len(mir["minstr"]),
            "values": len(mir["mvalue"]),
            "observations": len(mir_obs),
            "native_relation_valid": True,
            "independent_relation_valid": True,
            "enir_mir_execution_exact": True,
            "mir_metron_execution_exact": True,
        })

    payload = {
        "schema": "madaros.e3a.enir-mir-qd128-arithmetic.v1",
        "stage": "E3A-ENIR-MIR-QD128-ARITHMETIC-FULL",
        "programs": receipt_programs,
        "program_count": len(receipt_programs),
        "observation_count": total_observations,
        "relation": "bit_identical_logical_epistemic_product",
        "source_enir_hash_bound": True,
        "native_relation_validator": True,
        "independent_relation_validator": True,
        "independent_mir_replay": True,
        "enir_mir_differential_exact": True,
        "metron_mir_differential_exact": True,
        "semantic_tick_preserved": True,
        "abi_independent": True,
        "machine_ir_used": False,
        "memory_covered": False,
        "multi_block_cfg_supported": False,
        "fallback": "none",
        "compiler_sha256": hashlib.sha256(args.driver.read_bytes()).hexdigest(),
        "corpus_sha256": hashlib.sha256(args.corpus.read_bytes()).hexdigest(),
        "oracle_sha256": hashlib.sha256(args.oracle.read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "mir_implementation_sha256": hashlib.sha256((args.root / "self-hosted/enir/mir.sio").read_bytes()).hexdigest(),
        "mir_implementation_components": ["ir", "canonical_parser_printer", "lowerer", "verifier", "relation_validator", "interpreter"],
        "qd_semantics_sha256": hashlib.sha256((args.root / "self-hosted/enir/qd.sio").read_bytes()).hexdigest(),
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E3A_ENIR_MIR_QD128_VERIFY_PASS programs=6 observations=6 relation=native+independent execution=enir==mir==metron semantic_ticks=exact abi=independent memory=e3b-separate cfg=single-block")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E3AError, ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E3A_ENIR_MIR_QD128_VERIFY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
