#!/usr/bin/env python3
"""Run the independent Decimal center-orbit replay over 331 frozen leaves."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Sequence

from cs6_v7b_full_hpg_bridge_run import digest, digest_bytes
from cs6_v7b_target23_prospective_epistemic_replay_run import MANIFEST_REL, load_leaves


CONTRACT_REL = Path("scripts/research/cs6_v7b_target23_decimal_center_replay_contract_v1.txt")
CAPD_RESULT_REL = Path("scripts/research/receipts/cs6_v7b_target23_prospective_epistemic_replay_v1/results.tsv")
WORKER_REL = Path("scripts/research/cs6_v7b_target23_decimal_center_replay_worker.py")
EXPECTED_LEAVES = 331
CHALLENGE_DOMAIN = b"sounio.cs6.v7b-target23-decimal-center-replay-challenge.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-target23-decimal-center-replay-attempt.v1\0"
CARRIERS = ("C0HORect2Set", "C0Rect2Set")
RESULT_COLUMNS = (
    "LEAF_INDEX", "LAYER", "LEAF_ID", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX",
    "INPUT_SHA256", "RUN_CHALLENGE", "ATTEMPT_BINDING", "WORKER_RC", "ELAPSED_MS",
    "STDOUT_SHA256", "STDERR_SHA256", "COARSE_RETURN1_TIME", "COARSE_RETURN2_TIME",
    "FINE_RETURN1_TIME", "FINE_RETURN2_TIME", "COARSE_ELL", "FINE_ELL",
    "COARSE_INITIAL_NORMAL", "COARSE_FINAL_NORMAL", "FINE_INITIAL_NORMAL",
    "FINE_FINAL_NORMAL", "Q0_AREA", "COARSE_DETERMINANT", "FINE_DETERMINANT",
    "ABSOLUTE_DETERMINANT_DELTA", "C0HORECT2_LOWER", "C0HORECT2_UPPER",
    "C0RECT2_LOWER", "C0RECT2_UPPER", "COARSE_INSIDE_BOTH_CAPD",
    "FINE_INSIDE_BOTH_CAPD", "CENTER_REPLAY_SELF_CONSISTENT", "POINTWISE_LEAF_PASS",
)


def fail(message: str) -> None:
    raise SystemExit(f"decimal center replay run error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def parse_fields(raw: bytes, identity: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical worker output: {identity}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII worker output: {identity}") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            fail(f"malformed worker output: {identity}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            fail(f"empty or duplicate worker field: {identity}:{key}")
        fields[key] = value
    return fields


def binary64_decimal(token: str) -> Decimal:
    return Decimal.from_float(float.fromhex(token))


@dataclass(frozen=True)
class Plan:
    leaf: object
    challenge: str
    binding: str
    capd: dict[str, tuple[str, str]]

    @property
    def identity(self) -> str:
        return f"L{self.leaf.index:03d}_{self.leaf.leaf_id}"


def build_plan(root: Path, source_commit: str) -> list[Plan]:
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        fail("source commit must be lowercase 40-hex")
    leaves = load_leaves(root)
    capd_rows = list(csv.DictReader(canonical(root / CAPD_RESULT_REL).splitlines(), delimiter="\t"))
    by_leaf: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in capd_rows:
        by_leaf[row["LEAF_ID"]].append(row)
    contract_sha, manifest_sha = digest(root / CONTRACT_REL), digest(root / MANIFEST_REL)
    capd_sha, worker_sha = digest(root / CAPD_RESULT_REL), digest(root / WORKER_REL)
    plans: list[Plan] = []
    for leaf in leaves:
        rows = by_leaf.get(leaf.leaf_id, [])
        if len(rows) != 2 or tuple(row["CARRIER"] for row in rows) != CARRIERS:
            fail(f"CAPD carrier pair drifted: {leaf.leaf_id}")
        if any(row["EPISTEMIC_CERTIFICATE_PASS"] != "true" for row in rows):
            fail(f"frozen CAPD source is not certified: {leaf.leaf_id}")
        capd = {row["CARRIER"]: (row["LIOUVILLE_LOWER"], row["LIOUVILLE_UPPER"]) for row in rows}
        challenge = digest_bytes(
            CHALLENGE_DOMAIN + bytes.fromhex(contract_sha) + bytes.fromhex(manifest_sha)
            + bytes.fromhex(capd_sha) + bytes.fromhex(source_commit)
            + leaf.index.to_bytes(2, "big") + leaf.leaf_id.encode("ascii")
        )
        binding = digest_bytes(
            ATTEMPT_DOMAIN + bytes.fromhex(challenge) + bytes.fromhex(leaf.input_sha256)
            + bytes.fromhex(worker_sha)
        )
        plans.append(Plan(leaf, challenge, binding, capd))
    if len(plans) != EXPECTED_LEAVES:
        fail("plan cardinality drifted")
    return plans


def execute(root: Path, out_dir: Path, plan: Plan, timeout: float) -> dict[str, str]:
    leaf = plan.leaf
    command = [
        sys.executable, "-B", str((root / WORKER_REL).resolve()),
        str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth), str(leaf.s_index),
        plan.challenge, plan.binding,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
        rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
    except subprocess.TimeoutExpired as error:
        rc, stdout = 124, error.stdout or b""
        stderr = (error.stderr or b"") + b"\nTIMEOUT\n"
    elapsed_ms = int((time.monotonic() - started) * 1000)
    attempt_dir = out_dir / "attempts" / plan.identity
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "stdout.txt").write_bytes(stdout)
    (attempt_dir / "stderr.txt").write_bytes(stderr)
    (attempt_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n", encoding="ascii"
    )
    fields = parse_fields(stdout, plan.identity) if rc == 0 else {}
    coarse = Decimal(fields.get("COARSE_DETERMINANT", "NaN"))
    fine = Decimal(fields.get("FINE_DETERMINANT", "NaN"))
    intervals = [
        tuple(binary64_decimal(endpoint) for endpoint in plan.capd[carrier])
        for carrier in CARRIERS
    ]
    coarse_inside = coarse.is_finite() and all(lower < coarse < upper for lower, upper in intervals)
    fine_inside = fine.is_finite() and all(lower < fine < upper for lower, upper in intervals)
    self_consistent = fields.get("CENTER_REPLAY_SELF_CONSISTENT") == "true"
    passed = rc == 0 and self_consistent and coarse_inside and fine_inside
    return {
        "LEAF_INDEX": str(leaf.index), "LAYER": leaf.layer, "LEAF_ID": leaf.leaf_id,
        "U_DEPTH": str(leaf.u_depth), "U_INDEX": str(leaf.u_index),
        "S_DEPTH": str(leaf.s_depth), "S_INDEX": str(leaf.s_index),
        "INPUT_SHA256": leaf.input_sha256, "RUN_CHALLENGE": plan.challenge,
        "ATTEMPT_BINDING": plan.binding, "WORKER_RC": str(rc),
        "ELAPSED_MS": str(elapsed_ms), "STDOUT_SHA256": digest_bytes(stdout),
        "STDERR_SHA256": digest_bytes(stderr),
        "COARSE_RETURN1_TIME": fields.get("COARSE_RETURN1_TIME", "UNEMITTED"),
        "COARSE_RETURN2_TIME": fields.get("COARSE_RETURN2_TIME", "UNEMITTED"),
        "FINE_RETURN1_TIME": fields.get("FINE_RETURN1_TIME", "UNEMITTED"),
        "FINE_RETURN2_TIME": fields.get("FINE_RETURN2_TIME", "UNEMITTED"),
        "COARSE_ELL": fields.get("COARSE_ELL", "UNEMITTED"),
        "FINE_ELL": fields.get("FINE_ELL", "UNEMITTED"),
        "COARSE_INITIAL_NORMAL": fields.get("COARSE_INITIAL_NORMAL", "UNEMITTED"),
        "COARSE_FINAL_NORMAL": fields.get("COARSE_FINAL_NORMAL", "UNEMITTED"),
        "FINE_INITIAL_NORMAL": fields.get("FINE_INITIAL_NORMAL", "UNEMITTED"),
        "FINE_FINAL_NORMAL": fields.get("FINE_FINAL_NORMAL", "UNEMITTED"),
        "Q0_AREA": fields.get("FINE_Q0_AREA", "UNEMITTED"),
        "COARSE_DETERMINANT": fields.get("COARSE_DETERMINANT", "UNEMITTED"),
        "FINE_DETERMINANT": fields.get("FINE_DETERMINANT", "UNEMITTED"),
        "ABSOLUTE_DETERMINANT_DELTA": fields.get("ABSOLUTE_DETERMINANT_DELTA", "UNEMITTED"),
        "C0HORECT2_LOWER": plan.capd[CARRIERS[0]][0], "C0HORECT2_UPPER": plan.capd[CARRIERS[0]][1],
        "C0RECT2_LOWER": plan.capd[CARRIERS[1]][0], "C0RECT2_UPPER": plan.capd[CARRIERS[1]][1],
        "COARSE_INSIDE_BOTH_CAPD": str(coarse_inside).lower(),
        "FINE_INSIDE_BOTH_CAPD": str(fine_inside).lower(),
        "CENTER_REPLAY_SELF_CONSISTENT": str(self_consistent).lower(),
        "POINTWISE_LEAF_PASS": str(passed).lower(),
    }


def write_tsv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def write_summary(path: Path, rows: Sequence[dict[str, str]], source_commit: str) -> None:
    passed = sum(row["POINTWISE_LEAF_PASS"] == "true" for row in rows)
    fields = (
        ("SCHEMA", "sounio.cs6.v7b-target23-decimal-center-replay-summary.v1"),
        ("PRE_EXECUTION_GIT_COMMIT", source_commit), ("RUN_COMPLETE", str(len(rows) == 331).lower()),
        ("POINTWISE_ORBITS", str(len(rows))), ("POINTWISE_LEAVES_PASS", str(passed)),
        ("ALL_POINTWISE_LEAVES_PASS", str(passed == 331).lower()),
        ("INDEPENDENT_POINTWISE_SCOUT_COMPLETED", "false"),
        ("RIGOROUS_INTERVAL_CERTIFICATE", "false"), ("INDEPENDENT_INTERVAL_ENGINE", "false"),
        ("GLOBAL_HPG_CERTIFICATE", "false"), ("V7_B_ELIGIBILITY", "false"),
        ("PROMOTION_ELIGIBLE", "false"), ("OPEN_PROBLEM_SOLVED", "false"),
        ("NOVELTY_OR_PRIORITY_CLAIMED", "false"), ("FPGA_EXECUTION", "false"),
    )
    path.write_text("".join(f"{key}={value}\n" for key, value in fields), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--jobs", default=32, type=int)
    parser.add_argument("--timeout", default=60.0, type=float)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        fail("jobs must be in [1,64]")
    root, out_dir = Path.cwd(), args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = build_plan(root, args.source_commit)
    if args.plan_only:
        (out_dir / "plan-summary.txt").write_text(
            "SCHEMA=sounio.cs6.v7b-target23-decimal-center-replay-plan.v1\n"
            "PLAN_VALID=true\nPOINTWISE_ORBITS=331\nCAPD_CARRIERS_COMPARED_PER_ORBIT=2\n"
            f"PRE_EXECUTION_GIT_COMMIT={args.source_commit}\n"
            "INDEPENDENT_POINTWISE_SCOUT_COMPLETED=false\nRIGOROUS_INTERVAL_CERTIFICATE=false\n",
            encoding="ascii",
        )
        print((out_dir / "plan-summary.txt").read_text(encoding="ascii"), end=""); return
    (out_dir / "attempts").mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        rows = list(pool.map(lambda plan: execute(root, out_dir, plan, args.timeout), plans))
    rows.sort(key=lambda row: int(row["LEAF_INDEX"]))
    write_tsv(out_dir / "results.tsv", rows)
    write_summary(out_dir / "summary.txt", rows, args.source_commit)
    files = (root / CONTRACT_REL, root / MANIFEST_REL, root / CAPD_RESULT_REL,
             root / WORKER_REL, out_dir / "results.tsv", out_dir / "summary.txt")
    (out_dir / "files.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
