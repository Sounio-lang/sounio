#!/usr/bin/env python3
"""Independent E2C verifier for fuel, block arguments, and loop execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path


CASES = ["v1_fuel", "v1e_fixedpoint", "v1_fuel_high"]
TAGS = {
    "enir": 5, "resource": 2, "type": 8, "value": 14, "prov": 7,
    "policy": 5, "block": 11, "barg": 7, "edge": 9, "op": 11,
    "obs": 5, "end2": 11,
}
ORDER = {tag: index for index, tag in enumerate(TAGS)}


class E2CError(ValueError):
    pass


def fail(message: str) -> None:
    raise E2CError(message)


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    if check and result.returncode != 0:
        fail(f"command failed rc={result.returncode}: {' '.join(command)}\n{result.stdout.decode(errors='replace')}\n{result.stderr.decode(errors='replace')}")
    return result


def function_body(source: str, name: str) -> str:
    match = re.search(rf"\bfn\s+{re.escape(name)}_src\s*\([^)]*\)[^{{]*\{{", source)
    if not match:
        fail(f"source builder not found: {name}_src")
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
    fail(f"unterminated source builder: {name}_src")
    raise AssertionError


def extract_sources(corpus: Path) -> dict[str, str]:
    text = corpus.read_text(encoding="utf-8")
    result: dict[str, str] = {}
    for name in CASES:
        body = function_body(text, name)
        encoded = re.findall(r'add_line\s*\(&s,\s*"([^"\\]*(?:\\.[^"\\]*)*)",\s*&n\)', body)
        lines = [bytes(line, "utf-8").decode("unicode_escape") for line in encoded]
        if not lines or lines[0] != f"epistemic fn {name}() {{" or lines[-1] != "}":
            fail(f"source identity mismatch: {name}")
        source = "\n".join(lines) + "\n"
        if len(source.encode("ascii")) > 256:
            fail(f"source exceeds EISA Str cap: {name}")
        result[name] = source
    return result


def signed_bits(sign: int, exponent: int, mantissa: int) -> int:
    raw = (sign << 63) | (exponent << 52) | mantissa
    return raw - (1 << 64) if raw >= (1 << 63) else raw


def fbits(value: float) -> int:
    return struct.unpack(">q", struct.pack(">d", value))[0]


def bits_float(bits: int) -> float:
    return struct.unpack(">d", struct.pack(">q", bits))[0]


def source_number(token: str) -> float:
    if not re.fullmatch(r"-?[0-9]+(?:\.[0-9]+)?", token):
        fail(f"unsupported source number: {token}")
    negative = token.startswith("-")
    if negative:
        token = token[1:]
    integer, dot, fraction = token.partition(".")
    value = 0.0
    for ch in integer:
        value = value * 10.0 + float(ord(ch) - 48)
    scale = 0.1
    if dot:
        for ch in fraction:
            value += float(ord(ch) - 48) * scale
            scale *= 0.1
    return -value if negative else value


def parse_oracle(path: Path) -> dict[str, dict[str, int]]:
    cases: dict[str, dict[str, int]] = {}
    current: str | None = None
    receipt_status: dict[str, int] = {}
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("e2c-case-begin|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            current = fields["name"]
            cases[current] = {"source_len": int(fields["source_len"])}
        elif line.startswith("eisa-receipt:"):
            if current is None:
                fail("orphan EISA receipt")
            match = re.search(r" val=s([01])e([0-9]+)m([0-9]+).* poisoned=([01])", line)
            if not match:
                fail(f"malformed EISA receipt: {line}")
            sign, exponent, mantissa, status = map(int, match.groups())
            cases[current]["receipt_bits"] = signed_bits(sign, exponent, mantissa)
            receipt_status[current] = status
        elif line.startswith("e2c-case-end|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            name = fields["name"]
            if current != name:
                fail("oracle marker ordering mismatch")
            for key in ("rc", "gates", "fuel_initial", "fuel_left", "stop_kind", "last_reg", "branch_poisoned", "frail_branches"):
                cases[name][key] = int(fields[key])
            cases[name]["last_bits"] = signed_bits(int(fields["last_sign"]), int(fields["last_exp"]), int(fields["last_mant"]))
            cases[name]["receipt_status"] = receipt_status[name]
            current = None
    if list(cases) != CASES:
        fail(f"oracle case manifest drifted: {list(cases)}")
    if any(case.get("rc") != 0 or case.get("last_bits") != case.get("receipt_bits") for case in cases.values()):
        fail("oracle failed or receipt/register payload diverged")
    return cases


def parse_enir(raw: bytes) -> dict[str, list[list[str]]]:
    if not raw.endswith(b"\n") or b"\r" in raw:
        fail("ENIR is not canonical LF text")
    rows = [line.split("|") for line in raw.decode("ascii").splitlines()]
    prior = -1
    grouped = {tag: [] for tag in TAGS}
    for number, row in enumerate(rows, 1):
        if not row or row[0] not in TAGS or len(row) != TAGS[row[0]] or any(field == "" for field in row):
            fail(f"malformed schema-v2 row {number}")
        current = ORDER[row[0]]
        if current < prior:
            fail(f"non-canonical section order at row {number}")
        prior = current
        grouped[row[0]].append(row)
    if len(grouped["enir"]) != 1 or len(grouped["resource"]) != 1 or len(grouped["end2"]) != 1:
        fail("schema-v2 singleton row mismatch")
    return grouped


@dataclass
class ExpectedOp:
    opcode: int
    result: int = -1
    type_id: int = -1
    a: int = -1
    b: int = -1
    policy: int = -1
    line: int = 0
    known: int = 0
    bits: int = 0


@dataclass
class ExpectedCfg:
    name: str
    fuel: int
    ops: list[ExpectedOp]
    value_meta: list[tuple[int, int, int, int]]
    blocks: list[list[int]]
    block_args: list[list[int]]
    edges: list[list[int]]
    observation_kind: int


def independently_lower(source: str) -> ExpectedCfg:
    lines = source.splitlines()
    header = re.fullmatch(r"epistemic fn ([A-Za-z][A-Za-z0-9_]*)\(\) \{", lines[0])
    if not header:
        fail("independent parser rejected header")
    name = header.group(1)
    fuel_match = re.fullmatch(r"fuel ([0-9]+)", lines[1])
    if not fuel_match:
        fail("independent parser rejected fuel")
    fuel = int(fuel_match.group(1))
    symbols: list[list[object]] = []
    ops: list[ExpectedOp] = []
    values: list[tuple[int, int, int, int]] = []  # known, bits, line, transform
    inside = False
    loop_closed = False
    root_count = carry_count = entry_ops = body_first = -1
    carried: list[int] = []
    initial: list[int] = []
    header_values: list[int] = []
    exit_values: list[int] = []
    back_values: list[int] = []
    condition_value = -1
    gate_count = 0

    def lookup(symbol: str) -> int:
        for index in range(len(symbols) - 1, -1, -1):
            if symbols[index][0] == symbol:
                return index
        return -1

    def emit(opcode: int, line: int, a: int = -1, b: int = -1, known: int = 0, bits: int = 0) -> int:
        value_id = len(values)
        values.append((known, bits, line, opcode))
        ops.append(ExpectedOp(opcode, value_id, 0, a, b, -1, line, known, bits))
        return value_id

    def block_value(line: int) -> int:
        value_id = len(values)
        values.append((0, 0, line, 100))
        return value_id

    def operand(token: str, line: int) -> int:
        token = token.strip()
        index = lookup(token)
        if index >= 0:
            return int(symbols[index][1])
        number = source_number(token)
        return emit(0, line, known=1, bits=fbits(number))

    def expression(expr: str, line: int) -> int:
        expr = expr.strip()
        binary = re.fullmatch(r"(.+?)\s*([+*/-])\s*(.+)", expr)
        if binary:
            left, operator, right = binary.groups()
            return emit({"+": 2, "-": 3, "*": 4, "/": 5}[operator], line, operand(left, line), operand(right, line))
        index = lookup(expr)
        if index >= 0:
            return emit(9, line, int(symbols[index][1]))
        number = source_number(expr)
        return emit(0, line, known=1, bits=fbits(number))

    loop_line = -1
    for line_number, line in enumerate(lines[2:], 3):
        if line == "}":
            if inside:
                inside = False
                loop_closed = True
                back_values = [int(symbols[index][1]) for index in carried]
                symbols = symbols[:root_count]
                for ordinal, symbol_index in enumerate(carried):
                    symbols[symbol_index][1] = exit_values[ordinal]
                continue
            if line_number != len(lines):
                fail("premature root close")
            continue
        while_match = re.fullmatch(r"while ([A-Za-z][A-Za-z0-9_]*) != 0\.0 \{", line)
        if while_match:
            condition_index = lookup(while_match.group(1))
            if condition_index < 0 or inside or loop_closed:
                fail("invalid independent while")
            root_count = len(symbols)
            carried = list(range(root_count)) if root_count <= 4 else [condition_index]
            carry_count = len(carried)
            initial = [int(symbols[index][1]) for index in carried]
            loop_line = line_number
            for index in carried:
                header_value = block_value(line_number)
                exit_value = block_value(line_number)
                header_values.append(header_value)
                exit_values.append(exit_value)
                symbols[index][1] = header_value
                if index == condition_index:
                    condition_value = header_value
            entry_ops = len(ops)
            body_first = len(ops)
            inside = True
            continue
        let_match = re.fullmatch(r"let ([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        if let_match:
            symbol, expr = let_match.groups()
            if lookup(symbol) >= 0:
                fail("duplicate independent symbol")
            symbols.append([symbol, expression(expr, line_number)])
            continue
        set_match = re.fullmatch(r"set ([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.+)", line)
        if set_match:
            symbol, expr = set_match.groups()
            index = lookup(symbol)
            if not inside or index < 0 or index >= root_count or index not in carried:
                fail("invalid independent set")
            symbols[index][1] = emit(9, line_number, expression(expr, line_number))
            continue
        gate_match = re.fullmatch(r"gate ([A-Za-z][A-Za-z0-9_]*)", line)
        if gate_match:
            index = lookup(gate_match.group(1))
            if not loop_closed or index < 0:
                fail("invalid independent gate")
            ops.append(ExpectedOp(7, a=int(symbols[index][1]), policy=0, line=line_number))
            gate_count += 1
            continue
        fail(f"unsupported independent line {line_number}: {line}")
    if loop_line < 0 or not loop_closed or condition_value < 0 or gate_count > 1:
        fail("independent CFG did not close")
    body_count = (entry_ops if False else len(ops) - gate_count) - body_first
    exit_first = body_first + body_count
    blocks = [
        [0, 0, entry_ops, 0, 0, 1, -1, 0, -1, 0],
        [1, entry_ops, 0, 0, carry_count, 2, condition_value, 1, 2, 1],
        [2, body_first, body_count, carry_count, 0, 1, -1, 3, -1, 1],
        [3, exit_first, gate_count, carry_count, carry_count, 0, -1, -1, -1, 0],
    ]
    block_args: list[list[int]] = []
    for ordinal, value_id in enumerate(header_values):
        block_args.append([ordinal, 1, ordinal, value_id, 0, value_id])
    for ordinal, value_id in enumerate(exit_values):
        block_args.append([carry_count + ordinal, 3, ordinal, value_id, 0, value_id])
    pad = lambda args: args + [-1] * (4 - len(args))
    edges = [
        [0, 0, 1, carry_count, *pad(initial)],
        [1, 1, 3, carry_count, *pad(header_values)],
        [2, 1, 2, 0, -1, -1, -1, -1],
        [3, 2, 1, carry_count, *pad(back_values)],
    ]
    return ExpectedCfg(name, fuel, ops, values, blocks, block_args, edges, 0 if gate_count else 1)


def verify_artifact(grouped: dict[str, list[list[str]]], expected: ExpectedCfg) -> None:
    if grouped["enir"] != [["enir", "2", "2", expected.name, "1"]] or grouped["resource"] != [["resource", str(expected.fuel)]]:
        fail(f"header/resource mismatch: {expected.name}")
    if grouped["type"] != [["type", "0", "4", "1", "1", "1", "1", "1"]] or grouped["policy"] != [["policy", "0", "eisa_10_100_count_only", "0", "0"]]:
        fail(f"type/policy mismatch: {expected.name}")
    for tag, wanted in (("block", expected.blocks), ("barg", expected.block_args), ("edge", expected.edges)):
        actual = [list(map(int, row[1:])) for row in grouped[tag]]
        if actual != wanted:
            fail(f"{tag} mismatch for {expected.name}: {actual} != {wanted}")
    if len(grouped["op"]) != len(expected.ops):
        fail(f"op count mismatch: {expected.name}")
    for index, (row, op) in enumerate(zip(grouped["op"], expected.ops)):
        actual = list(map(int, row[1:]))
        wanted = [index, op.opcode, op.result, op.type_id, op.a, op.b, -1, op.policy, -1, 1]
        if actual != wanted:
            fail(f"op mismatch {expected.name} at {index}: {actual} != {wanted}")
    if len(grouped["value"]) != len(expected.value_meta) or len(grouped["prov"]) != len(expected.value_meta):
        fail(f"value/provenance count mismatch: {expected.name}")
    for index, ((known, bits, line, transform), value, provenance) in enumerate(zip(expected.value_meta, grouped["value"], grouped["prov"])):
        if int(value[1]) != index or int(value[2]) != 0 or int(value[3]) != known or int(value[4]) != bits or int(value[13]) != index:
            fail(f"value mismatch {expected.name} at {index}")
        if list(map(int, (provenance[1], provenance[2], provenance[4], provenance[5], provenance[6]))) != [index, line, transform, -1, -1]:
            fail(f"provenance mismatch {expected.name} at {index}")
    if grouped["obs"] != [["obs", "0", expected.name, "0", str(expected.observation_kind)]]:
        fail(f"observation mismatch: {expected.name}")
    footer = [1, len(expected.value_meta), len(expected.value_meta), 1, 4, len(expected.block_args), 4, len(expected.ops), 1, 1]
    if [list(map(int, row[1:])) for row in grouped["end2"]] != [footer]:
        fail(f"footer mismatch: {expected.name}")


def edge_arg(edge: list[int], ordinal: int) -> int:
    return edge[4 + ordinal]


def independent_execute(grouped: dict[str, list[list[str]]]) -> dict[str, object]:
    fuel = int(grouped["resource"][0][1])
    values_desc = grouped["value"]
    ops = [list(map(int, row[1:])) for row in grouped["op"]]
    blocks = [list(map(int, row[1:])) for row in grouped["block"]]
    bargs = [list(map(int, row[1:])) for row in grouped["barg"]]
    edges = [list(map(int, row[1:])) for row in grouped["edge"]]
    runtime: dict[int, float] = {}
    block_id = 0
    executed = gates = 0
    last_write = -1
    controls: list[tuple[int, int, int]] = []
    gate_bits: list[int] = []
    stop_kind = 0
    while True:
        block = blocks[block_id]
        for pc in range(block[1], block[1] + block[2]):
            if fuel == 0:
                stop_kind = 1
                return {"fuel_left": fuel, "stop_kind": stop_kind, "executed": executed, "gates": gates, "last_write": last_write, "last_bits": fbits(runtime[last_write]), "gate_bits": gate_bits, "controls": controls}
            fuel -= 1
            executed += 1
            _, opcode, result, _, a, b, _, _, _, _ = ops[pc]
            if opcode == 0:
                runtime[result] = bits_float(int(values_desc[result][4]))
                last_write = result
            elif opcode in {2, 3, 4, 5}:
                runtime[result] = {2: runtime[a] + runtime[b], 3: runtime[a] - runtime[b], 4: runtime[a] * runtime[b], 5: runtime[a] / runtime[b]}[opcode]
                last_write = result
            elif opcode == 9:
                runtime[result] = runtime[a]
                last_write = result
            elif opcode == 7:
                gate_bits.append(fbits(runtime[a]))
                gates += 1
            else:
                fail(f"independent execution unsupported opcode {opcode}")
        term, condition, edge0, edge1, term_cost = block[5], block[6], block[7], block[8], block[9]
        if term_cost:
            if fuel == 0:
                stop_kind = 1
                return {"fuel_left": fuel, "stop_kind": stop_kind, "executed": executed, "gates": gates, "last_write": last_write, "last_bits": fbits(runtime[last_write]), "gate_bits": gate_bits, "controls": controls}
            fuel -= 1
            executed += 1
        if term == 0:
            return {"fuel_left": fuel, "stop_kind": 0, "executed": executed, "gates": gates, "last_write": last_write, "last_bits": fbits(runtime[last_write]), "gate_bits": gate_bits, "controls": controls}
        edge_id = edge0
        taken = 1
        if term in {2, 3}:
            taken = int(runtime[condition] == 0.0) if term == 2 else int(runtime[condition] < 0.0)
            edge_id = edge0 if taken else edge1
        edge = edges[edge_id]
        controls.append((block_id, edge_id, taken))
        incoming = [runtime[edge_arg(edge, ordinal)] for ordinal in range(edge[3])]
        target = blocks[edge[2]]
        for ordinal, value in enumerate(incoming):
            runtime[bargs[target[3] + ordinal][3]] = value
        block_id = edge[2]


def parse_native(output: bytes) -> dict[str, object]:
    controls: list[tuple[int, int, int]] = []
    observation: dict[str, int] | None = None
    final: dict[str, int] | None = None
    for line in output.decode("ascii").splitlines():
        if line.startswith("enir-control|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            controls.append((int(fields["block"]), int(fields["edge"]), int(fields["taken"])))
        elif line.startswith("enir-exec|") or line.startswith("enir-fuel|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            observation = {key: int(fields[key]) for key in ("value_bits", "error0_bits", "error1_bits", "uncertainty_bits", "status", "branch_poisoned", "frail_branches")}
        elif line.startswith("enir-exec-ok|"):
            fields = dict(part.split("=", 1) for part in line.split("|")[1:])
            final = {key: int(fields[key]) for key in ("executed_ops", "observations", "fuel_initial", "fuel_left", "stop_kind", "last_write", "branch_poisoned", "frail_branches")}
    if observation is None or final is None:
        fail("native interpreter omitted observation/final receipt")
    return {"controls": controls, "observation": observation, "final": final}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    implementation = "\n".join((args.root / path).read_text(encoding="utf-8") for path in ("self-hosted/enir/source_lower.sio", "self-hosted/enir/interpreter.sio"))
    for forbidden in ("enir_shadow_fixture(", "use eisa::", "eisax_encode(", "eisa_compile(", "eisa_compile_v1("):
        if forbidden in implementation:
            fail(f"forbidden oracle dependency in ENIR implementation: {forbidden}")

    sources = extract_sources(args.corpus)
    oracle = parse_oracle(args.oracle)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    receipt_cases = []
    for name in CASES:
        source = sources[name]
        if len(source.encode("ascii")) != oracle[name]["source_len"]:
            fail(f"source/oracle length mismatch: {name}")
        source_path = args.out_dir / f"{name}.eisa"
        artifact_path = args.out_dir / f"{name}.enir"
        source_path.write_text(source, encoding="ascii")
        artifact = run([str(args.driver), "lower-v1", str(source_path)]).stdout
        artifact_path.write_bytes(artifact)
        grouped = parse_enir(artifact)
        expected = independently_lower(source)
        verify_artifact(grouped, expected)
        roundtrip = run([str(args.driver), "roundtrip", str(artifact_path)]).stdout
        if roundtrip != artifact:
            fail(f"native roundtrip changed bytes: {name}")
        replay = independent_execute(grouped)
        native = parse_native(run([str(args.driver), "run", str(artifact_path)]).stdout)
        final = native["final"]
        observation = native["observation"]
        for key, replay_key in (("fuel_left", "fuel_left"), ("stop_kind", "stop_kind"), ("executed_ops", "executed"), ("last_write", "last_write")):
            if final[key] != replay[replay_key]:
                fail(f"native/replay mismatch {name} field {key}")
        if native["controls"] != replay["controls"]:
            fail(f"native/replay control mismatch: {name}")
        if replay["fuel_left"] != oracle[name]["fuel_left"] or replay["stop_kind"] != oracle[name]["stop_kind"] or replay["gates"] != oracle[name]["gates"] or replay["last_bits"] != oracle[name]["last_bits"]:
            fail(f"ENIR/EISA state mismatch: {name}")
        if observation["value_bits"] != oracle[name]["receipt_bits"] or observation["status"] != oracle[name]["receipt_status"]:
            fail(f"ENIR/EISA receipt mismatch: {name}")
        if any(observation[key] != 0 for key in ("error0_bits", "error1_bits", "uncertainty_bits", "branch_poisoned", "frail_branches")):
            fail(f"unexpected numeric/control residue: {name}")
        receipt_cases.append({
            "name": name,
            "source_sha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
            "fuel_initial": expected.fuel,
            "fuel_left": replay["fuel_left"],
            "stop_kind": replay["stop_kind"],
            "executed": replay["executed"],
            "last_value_id": replay["last_write"],
            "last_bits": replay["last_bits"],
            "blocks": len(expected.blocks),
            "block_args": len(expected.block_args),
            "edges": len(expected.edges),
        })
    payload = {
        "schema": "madaros.e2c.enir-fuel-blockargs.v1",
        "cases": receipt_cases,
        "summary": {"programs": 3, "observations": 3, "blocks": 12, "edges": 12, "source_oracle": "EISA/METRON", "independent_replay": True},
    }
    args.receipt.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="ascii")
    print("E2C_ENIR_FUEL_BLOCKARGS_VERIFY_PASS programs=3 observations=3 blocks=12 edges=12 fuel=5,100,25 outcomes=stop,gate,stop independent_replay=exact evm_observable=exact")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (E2CError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"E2C_ENIR_FUEL_BLOCKARGS_VERIFY_FAIL: {exc}", file=__import__("sys").stderr)
        raise SystemExit(1)
