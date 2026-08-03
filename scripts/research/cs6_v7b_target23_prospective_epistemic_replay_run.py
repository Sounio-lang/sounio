#!/usr/bin/env python3
"""Run the predeclared target-23 adaptive cover under fresh challenges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shlex
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Sequence

from cs6_v7b_full_hpg_bridge_run import digest, digest_bytes
from cs6_v7b_subdivision_ladder_run import CARRIERS, LEDGER_REL, classify, input_bytes, node_id


CONTRACT_REL = Path("scripts/research/cs6_v7b_target23_prospective_epistemic_replay_contract_v1.txt")
MANIFEST_REL = Path("scripts/research/cs6_v7b_target23_prospective_epistemic_replay_coordinates_v1.tsv")
WORKER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
EXPECTED_LEAVES = 331
EXPECTED_ATTEMPTS = 662
CHALLENGE_DOMAIN = b"sounio.cs6.v7b-target23-prospective-epistemic-replay-challenge.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-target23-prospective-epistemic-replay-attempt.v1\0"
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
INTERVAL_RE = re.compile(r"(?:^| )([A-Z0-9_]+)=\[([^,]+),([^\]]+)\]")
BOOL_RE = re.compile(r"(?:^| )([A-Z0-9_]+)=(true|false)(?= |$)")
RAW_INTERVALS = {
    "C1_P2_CONTROL": ("c1", "DET"),
    "C2_FULL_P2": ("c2", "HULL_DET"),
    "AFFINE_CARRIER": ("affine", "DET"),
    "HOMOGENEOUS_LOCAL_P2": ("resident", "RECON_DET"),
    "PLUCKER_COCYCLE": ("homogeneous", "DET"),
    "LIOUVILLE": ("liouville", "DET"),
}
MANIFEST_COLUMNS = (
    "LAYER", "LEAF_ID", "PARENT_DEPTH4_CELL_INDEX", "DEPTH4_U_OFFSET",
    "DEPTH4_S_OFFSET", "SUB_U_OFFSET", "SUB_S_OFFSET",
)
RESULT_COLUMNS = (
    "ATTEMPT_INDEX", "LEAF_INDEX", "LAYER", "LEAF_ID",
    "PARENT_DEPTH4_CELL_INDEX", "DEPTH4_U_OFFSET", "DEPTH4_S_OFFSET",
    "SUB_U_OFFSET", "SUB_S_OFFSET", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX",
    "CARRIER", "STATUS", "WORKER_RC", "ELAPSED_MS", "INPUT_SHA256",
    "RUN_CHALLENGE", "ATTEMPT_BINDING", "STDOUT_SHA256", "STDERR_SHA256",
    "SUMMARY_SHA256", "LEGACY_CERTIFICATE_PASS", "TERMINAL_CERTIFIED",
    "STRUCTURAL_PASS", "HOMOGENEOUS_COMPUTATION_VALID", "PROBE_PASS",
    "LIOUVILLE_LOWER", "LIOUVILLE_UPPER", "JOINT_LOWER", "JOINT_UPPER",
    "JOINT_EQUALS_LIOUVILLE", "EPISTEMIC_CERTIFICATE_PASS",
)


def die(message: str) -> None:
    raise SystemExit(f"prospective replay run error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        die(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def exact(token: str) -> Fraction:
    try:
        numerator, denominator = float.fromhex(token).as_integer_ratio()
    except (ValueError, OverflowError) as error:
        raise ValueError(f"invalid binary64 endpoint: {token}") from error
    return Fraction(numerator, denominator)


@dataclass(frozen=True)
class Leaf:
    index: int
    layer: str
    leaf_id: str
    parent_cell: int
    depth4_u: int
    depth4_s: int
    sub_u: str
    sub_s: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    row_sha256: str
    input_sha256: str


@dataclass(frozen=True)
class Attempt:
    index: int
    leaf: Leaf
    carrier: str
    challenge: str
    binding: str

    @property
    def identity(self) -> str:
        return f"A{self.index:04d}_L{self.leaf.index:03d}_{self.carrier}"


def load_leaves(root: Path) -> list[Leaf]:
    text = canonical(root / MANIFEST_REL)
    rows = list(csv.DictReader(text.splitlines(), delimiter="\t"))
    if len(rows) != EXPECTED_LEAVES or not rows or tuple(rows[0]) != MANIFEST_COLUMNS:
        die("frozen manifest shape drifted")
    leaves: list[Leaf] = []
    d4_cells: set[int] = set()
    d5_children: dict[int, set[tuple[int, int]]] = defaultdict(set)
    for index, row in enumerate(rows, 1):
        parent = int(row["PARENT_DEPTH4_CELL_INDEX"])
        u4, s4 = int(row["DEPTH4_U_OFFSET"]), int(row["DEPTH4_S_OFFSET"])
        if parent != u4 * 16 + s4 + 1 or not (0 <= u4 < 16 and 0 <= s4 < 16):
            die(f"depth-4 parent coordinate drift: leaf {index}")
        if row["LAYER"] == "D4":
            if row["SUB_U_OFFSET"] != "NA" or row["SUB_S_OFFSET"] != "NA":
                die(f"D4 leaf has sub-offsets: leaf {index}")
            u_depth, u_index, s_depth, s_index = 7, 96 + u4, 8, 160 + s4
            d4_cells.add(parent)
        elif row["LAYER"] == "D5":
            du, ds = int(row["SUB_U_OFFSET"]), int(row["SUB_S_OFFSET"])
            if du not in (0, 1) or ds not in (0, 1):
                die(f"D5 sub-offset outside partition: leaf {index}")
            u_depth, u_index = 8, 2 * (96 + u4) + du
            s_depth, s_index = 9, 2 * (160 + s4) + ds
            d5_children[parent].add((du, ds))
        else:
            die(f"unknown layer: leaf {index}")
        expected_node = node_id(u_depth, u_index, s_depth, s_index)
        if row["LEAF_ID"] != expected_node:
            die(f"node identity drift: leaf {index}")
        row_bytes = ("\t".join(row[column] for column in MANIFEST_COLUMNS) + "\n").encode("ascii")
        leaves.append(Leaf(
            index, row["LAYER"], expected_node, parent, u4, s4,
            row["SUB_U_OFFSET"], row["SUB_S_OFFSET"], u_depth, u_index,
            s_depth, s_index, digest_bytes(row_bytes),
            digest_bytes(input_bytes(u_depth, u_index, s_depth, s_index)),
        ))
    expected_children = {(0, 0), (0, 1), (1, 0), (1, 1)}
    if len(d4_cells) != 231 or len(d5_children) != 25:
        die("adaptive topology population drifted")
    if d4_cells & set(d5_children) or len(d4_cells | set(d5_children)) != 256:
        die("adaptive topology is not a disjoint full parent partition")
    if any(children != expected_children for children in d5_children.values()):
        die("refined parent lacks its exact four-child partition")
    if len({leaf.leaf_id for leaf in leaves}) != EXPECTED_LEAVES:
        die("duplicate leaf identity")
    return leaves


def build_attempts(root: Path, source_commit: str) -> list[Attempt]:
    if not COMMIT_RE.fullmatch(source_commit):
        die("source commit must be a lowercase 40-hex Git identity")
    contract_sha, manifest_sha = digest(root / CONTRACT_REL), digest(root / MANIFEST_REL)
    attempts: list[Attempt] = []
    for leaf in load_leaves(root):
        challenge = digest_bytes(
            CHALLENGE_DOMAIN + bytes.fromhex(contract_sha) + bytes.fromhex(manifest_sha)
            + bytes.fromhex(source_commit) + leaf.index.to_bytes(2, "big")
            + bytes.fromhex(leaf.row_sha256)
        )
        for carrier in CARRIERS:
            binding = digest_bytes(
                ATTEMPT_DOMAIN + bytes.fromhex(challenge)
                + bytes.fromhex(leaf.input_sha256) + carrier.encode("ascii")
                + bytes.fromhex(contract_sha) + bytes.fromhex(manifest_sha)
            )
            attempts.append(Attempt(len(attempts) + 1, leaf, carrier, challenge, binding))
    if len(attempts) != EXPECTED_ATTEMPTS:
        die("attempt population drifted")
    return attempts


def raw_certificate(stdout: bytes, status: str, result_probe: str) -> dict[str, str]:
    try:
        text = stdout.decode("ascii")
    except UnicodeError:
        return {key: "UNEMITTED" for key in RESULT_COLUMNS[23:]}
    intervals: dict[str, tuple[str, str]] = {}
    flags: dict[str, bool] = {}
    for line in text.splitlines():
        head = line.split(" ", 1)[0]
        if head in RAW_INTERVALS:
            name, label = RAW_INTERVALS[head]
            found = {m.group(1): (m.group(2), m.group(3)) for m in INTERVAL_RE.finditer(line)}
            if label in found and name not in intervals:
                intervals[name] = found[label]
        if head in {"LEAF_RESULT", "SUMMARY"}:
            flags.update((m.group(1), m.group(2) == "true") for m in BOOL_RE.finditer(line))
    required_flags = {
        "TERMINAL_CERTIFIED", "CERTIFICATE_PASS", "STRUCTURAL_PASS",
        "HOMOGENEOUS_COMPUTATION_VALID", "PROBE_PASS",
    }
    complete = len(intervals) == 6 and required_flags <= set(flags)
    liouville = intervals.get("liouville", ("UNEMITTED", "UNEMITTED"))
    joint_lower = joint_upper = "UNEMITTED"
    equals_liouville = False
    negative = False
    if complete:
        lowers = [(exact(v[0]), v[0]) for v in intervals.values()]
        uppers = [(exact(v[1]), v[1]) for v in intervals.values()]
        low, high = max(lowers), min(uppers)
        joint_lower, joint_upper = low[1], high[1]
        equals_liouville = low[0] == exact(liouville[0]) and high[0] == exact(liouville[1])
        negative = low[0] <= high[0] and high[0] < 0 and exact(liouville[1]) < 0
    legacy = flags.get("CERTIFICATE_PASS") is True
    terminal = flags.get("TERMINAL_CERTIFIED") is True
    passed = (
        complete and status == "DESCENDANT_PROBE_PASS" and result_probe == "true"
        and flags["PROBE_PASS"] and flags["STRUCTURAL_PASS"]
        and flags["HOMOGENEOUS_COMPUTATION_VALID"] and not legacy and not terminal
        and negative
    )
    return {
        "LEGACY_CERTIFICATE_PASS": str(legacy).lower() if complete else "UNEMITTED",
        "TERMINAL_CERTIFIED": str(terminal).lower() if complete else "UNEMITTED",
        "STRUCTURAL_PASS": str(flags.get("STRUCTURAL_PASS", False)).lower() if complete else "UNEMITTED",
        "HOMOGENEOUS_COMPUTATION_VALID": str(flags.get("HOMOGENEOUS_COMPUTATION_VALID", False)).lower() if complete else "UNEMITTED",
        "PROBE_PASS": str(flags.get("PROBE_PASS", False)).lower() if complete else result_probe,
        "LIOUVILLE_LOWER": liouville[0], "LIOUVILLE_UPPER": liouville[1],
        "JOINT_LOWER": joint_lower, "JOINT_UPPER": joint_upper,
        "JOINT_EQUALS_LIOUVILLE": str(equals_liouville).lower(),
        "EPISTEMIC_CERTIFICATE_PASS": str(passed).lower(),
    }


def execute(root: Path, binary: Path, out_dir: Path, attempt: Attempt,
            timeout: float, contract_sha: str, ledger_sha: str) -> dict[str, str]:
    leaf = attempt.leaf
    command = [
        str(binary), str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth),
        str(leaf.s_index), leaf.input_sha256, attempt.challenge, attempt.carrier,
        contract_sha, ledger_sha, contract_sha,
        digest_bytes(leaf.leaf_id.encode("ascii") + b"\n"), attempt.binding,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
        rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout = 124, error.stdout or b""
        stderr = (error.stderr or b"") + b"\nTIMEOUT\n"
    elapsed = int((time.monotonic() - started) * 1000)
    attempt_dir = out_dir / "attempts" / attempt.identity
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "stdout.txt").write_bytes(stdout)
    (attempt_dir / "stderr.txt").write_bytes(stderr)
    (attempt_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n", encoding="ascii"
    )
    status, summary_sha, fields = classify(rc, stdout, stderr)
    row = {
        "ATTEMPT_INDEX": str(attempt.index), "LEAF_INDEX": str(leaf.index),
        "LAYER": leaf.layer, "LEAF_ID": leaf.leaf_id,
        "PARENT_DEPTH4_CELL_INDEX": str(leaf.parent_cell),
        "DEPTH4_U_OFFSET": str(leaf.depth4_u), "DEPTH4_S_OFFSET": str(leaf.depth4_s),
        "SUB_U_OFFSET": leaf.sub_u, "SUB_S_OFFSET": leaf.sub_s,
        "U_DEPTH": str(leaf.u_depth), "U_INDEX": str(leaf.u_index),
        "S_DEPTH": str(leaf.s_depth), "S_INDEX": str(leaf.s_index),
        "CARRIER": attempt.carrier, "STATUS": status, "WORKER_RC": str(rc),
        "ELAPSED_MS": str(elapsed), "INPUT_SHA256": leaf.input_sha256,
        "RUN_CHALLENGE": attempt.challenge, "ATTEMPT_BINDING": attempt.binding,
        "STDOUT_SHA256": digest_bytes(stdout), "STDERR_SHA256": digest_bytes(stderr),
        "SUMMARY_SHA256": summary_sha,
    }
    row.update(raw_certificate(stdout, status, fields.get("PROBE_PASS", "UNEMITTED")))
    return row


def write_tsv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: Sequence[dict[str, str]], source_commit: str) -> None:
    pairs = [rows[i:i + 2] for i in range(0, len(rows), 2)]
    certified = sum(row["EPISTEMIC_CERTIFICATE_PASS"] == "true" for row in rows)
    pair_pass = sum(all(row["EPISTEMIC_CERTIFICATE_PASS"] == "true" for row in pair) for pair in pairs)
    complete = len(rows) == EXPECTED_ATTEMPTS
    fields = (
        ("SCHEMA", "sounio.cs6.v7b-target23-prospective-epistemic-replay-summary.v1"),
        ("PRE_EXECUTION_GIT_COMMIT", source_commit),
        ("RUN_COMPLETE", str(complete).lower()),
        ("FRESH_ATTEMPT_PROCESSES", str(len(rows))),
        ("LEAVES_EVALUATED", str(len(pairs))),
        ("ATTEMPTS_COMPLETED", str(len(rows))),
        ("EPISTEMIC_CERTIFICATE_PASS_ATTEMPTS", str(certified)),
        ("PAIR_CERTIFICATE_PASS_LEAVES", str(pair_pass)),
        ("ALL_ATTEMPTS_CERTIFIED", str(complete and certified == EXPECTED_ATTEMPTS).lower()),
        ("ALL_PAIRS_CERTIFIED", str(complete and pair_pass == EXPECTED_LEAVES).lower()),
        ("PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED", "false"),
        ("GLOBAL_HPG_CERTIFICATE", "false"), ("V7_B_ELIGIBILITY", "false"),
        ("V7_B_WINNER", "NONE"), ("PROMOTION_ELIGIBLE", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"), ("NOVELTY_OR_PRIORITY_CLAIMED", "false"),
        ("FPGA_EXECUTION", "false"),
    )
    path.write_text("".join(f"{k}={v}\n" for k, v in fields), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--jobs", default=32, type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        die("jobs must be in [1,64]")
    root, out_dir = Path.cwd(), args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = build_attempts(root, args.source_commit)
    if args.plan_only:
        (out_dir / "plan-summary.txt").write_text(
            "SCHEMA=sounio.cs6.v7b-target23-prospective-epistemic-replay-plan.v1\n"
            "PLAN_VALID=true\nFROZEN_LEAVES=331\nATTEMPTS=662\n"
            f"PRE_EXECUTION_GIT_COMMIT={args.source_commit}\n"
            "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=false\n",
            encoding="ascii",
        )
        print((out_dir / "plan-summary.txt").read_text(encoding="ascii"), end="")
        return
    if args.binary is None or not args.binary.is_file():
        die("--binary must name the freshly built CAPD worker")
    binary = args.binary.resolve()
    (out_dir / "attempts").mkdir(exist_ok=True)
    contract_sha, ledger_sha = digest(root / CONTRACT_REL), digest(root / LEDGER_REL)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(
            lambda item: execute(root, binary, out_dir, item, args.timeout, contract_sha, ledger_sha),
            attempts,
        ))
    rows.sort(key=lambda row: int(row["ATTEMPT_INDEX"]))
    write_tsv(out_dir / "results.tsv", RESULT_COLUMNS, rows)
    write_summary(out_dir / "summary.txt", rows, args.source_commit)
    files = (root / CONTRACT_REL, root / MANIFEST_REL, root / LEDGER_REL,
             root / WORKER_REL, out_dir / "results.tsv", out_dir / "summary.txt", binary)
    (out_dir / "files.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
