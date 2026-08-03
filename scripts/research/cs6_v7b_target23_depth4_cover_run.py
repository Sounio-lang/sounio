#!/usr/bin/env python3
"""Execute the complete depth-4 sibling cover of frozen V7-B target 23."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cs6_v7b_full_hpg_bridge_run import digest, digest_bytes
from cs6_v7b_subdivision_ladder_run import (
    CARRIERS,
    LEDGER_REL,
    PARENT_RESULTS_REL,
    Parent,
    classify,
    input_bytes,
    load_parents,
    node_id,
    parse_node,
)


CONTRACT_REL = Path("scripts/research/cs6_v7b_target23_depth4_cover_contract_v1.txt")
WORKER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
PARENT_ORDINAL = "23"
DEPTH_DELTA = 4
OFFSET_DOMAIN = tuple(range(1 << DEPTH_DELTA))
EXPECTED_CELLS = len(OFFSET_DOMAIN) ** 2
EXPECTED_ATTEMPTS = EXPECTED_CELLS * len(CARRIERS)
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-target23-depth4-cover-attempt.v1\0"
ZERO_SHA256 = "0" * 64

RESULT_COLUMNS = (
    "ATTEMPT_INDEX",
    "PARENT_V7_ORDINAL",
    "CHECKPOINT_ROLE",
    "DEPTH_DELTA",
    "CHILD_U_OFFSET",
    "CHILD_S_OFFSET",
    "NODE_ID",
    "CARRIER",
    "STATUS",
    "WORKER_RC",
    "ELAPSED_MS",
    "INPUT_SHA256",
    "RUN_CHALLENGE",
    "ATTEMPT_BINDING",
    "STDOUT_SHA256",
    "STDERR_SHA256",
    "SUMMARY_SHA256",
    "C1_ORIENTATION_UNRESOLVED",
    "C2_HULL_ORIENTATION_UNRESOLVED",
    "EVENT1_CHARTS_CERTIFIED",
    "EVENT2_CHARTS_CERTIFIED",
    "HOMOGENEOUS_COMPUTATION_VALID",
    "CERTIFICATE_PASS",
    "PROBE_PASS",
)

COORDINATE_COLUMNS = (
    "CELL_INDEX",
    "CHILD_U_OFFSET",
    "CHILD_S_OFFSET",
    "NODE_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "INPUT_SHA256",
)


def die(message: str) -> None:
    raise SystemExit(message)


@dataclass(frozen=True)
class CoverAttempt:
    index: int
    parent: Parent
    depth_delta: int
    child_u_offset: int
    child_s_offset: int
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    carrier: str
    input_sha256: str
    run_challenge: str
    attempt_binding: str

    @property
    def node(self) -> str:
        return node_id(self.u_depth, self.u_index, self.s_depth, self.s_index)

    @property
    def identity(self) -> str:
        return (
            f"A{self.index:04d}_U{self.child_u_offset:02d}_"
            f"S{self.child_s_offset:02d}_{self.carrier}"
        )


def target_parent(root: Path) -> Parent:
    selected = [
        parent
        for parent in load_parents(root / PARENT_RESULTS_REL)
        if parent.ordinal == PARENT_ORDINAL
    ]
    if len(selected) != 1:
        die("frozen target parent 23 is missing or duplicated")
    if selected[0].node != "U03-0000000006_S04-0000000010":
        die("frozen target node drifted")
    return selected[0]


def build_attempts(root: Path) -> list[CoverAttempt]:
    parent = target_parent(root)
    parent_u_depth, parent_u_index, parent_s_depth, parent_s_index = parse_node(
        parent.node
    )
    contract_sha = digest(root / CONTRACT_REL)
    parent_results_sha = digest(root / PARENT_RESULTS_REL)
    ledger_sha = digest(root / LEDGER_REL)
    attempts: list[CoverAttempt] = []
    for u_offset in OFFSET_DOMAIN:
        for s_offset in OFFSET_DOMAIN:
            u_depth = parent_u_depth + DEPTH_DELTA
            u_index = (parent_u_index << DEPTH_DELTA) + u_offset
            s_depth = parent_s_depth + DEPTH_DELTA
            s_index = (parent_s_index << DEPTH_DELTA) + s_offset
            input_sha = digest_bytes(input_bytes(u_depth, u_index, s_depth, s_index))
            for carrier in CARRIERS:
                challenge = digest_bytes(
                    b"sounio.cs6.v7b-target23-depth4-cover-challenge.v1\0"
                    + bytes.fromhex(contract_sha)
                    + bytes.fromhex(parent_results_sha)
                    + bytes.fromhex(ledger_sha)
                    + bytes([u_offset, s_offset])
                )
                binding = digest_bytes(
                    ATTEMPT_DOMAIN
                    + bytes.fromhex(challenge)
                    + bytes.fromhex(input_sha)
                    + carrier.encode("ascii")
                )
                attempts.append(
                    CoverAttempt(
                        index=len(attempts) + 1,
                        parent=parent,
                        depth_delta=DEPTH_DELTA,
                        child_u_offset=u_offset,
                        child_s_offset=s_offset,
                        u_depth=u_depth,
                        u_index=u_index,
                        s_depth=s_depth,
                        s_index=s_index,
                        carrier=carrier,
                        input_sha256=input_sha,
                        run_challenge=challenge,
                        attempt_binding=binding,
                    )
                )
    if len(attempts) != EXPECTED_ATTEMPTS:
        die("expected exactly 512 target-cover attempts")
    return attempts


def coordinate_rows(attempts: Sequence[CoverAttempt]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for cell_index, attempt in enumerate(attempts[:: len(CARRIERS)], 1):
        rows.append(
            {
                "CELL_INDEX": str(cell_index),
                "CHILD_U_OFFSET": str(attempt.child_u_offset),
                "CHILD_S_OFFSET": str(attempt.child_s_offset),
                "NODE_ID": attempt.node,
                "U_DEPTH": str(attempt.u_depth),
                "U_INDEX": str(attempt.u_index),
                "S_DEPTH": str(attempt.s_depth),
                "S_INDEX": str(attempt.s_index),
                "INPUT_SHA256": attempt.input_sha256,
            }
        )
    if len(rows) != EXPECTED_CELLS:
        die("expected exactly 256 coordinate rows")
    return rows


def write_tsv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def run_attempt(
    root: Path,
    binary: Path,
    attempt: CoverAttempt,
    timeout: float,
    out_dir: Path,
    contract_sha: str,
    ledger_sha: str,
) -> dict[str, str]:
    command = [
        str(binary),
        str(attempt.u_depth),
        str(attempt.u_index),
        str(attempt.s_depth),
        str(attempt.s_index),
        attempt.input_sha256,
        attempt.run_challenge,
        attempt.carrier,
        contract_sha,
        ledger_sha,
        contract_sha,
        digest_bytes(attempt.node.encode("ascii") + b"\n"),
        attempt.attempt_binding,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
        worker_rc = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except subprocess.TimeoutExpired as error:
        worker_rc = 124
        stdout = error.stdout or b""
        stderr = (error.stderr or b"") + b"\nTIMEOUT\n"
    elapsed_ms = int((time.monotonic() - started) * 1000)
    attempt_dir = out_dir / "attempts" / attempt.identity
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "stdout.txt").write_bytes(stdout)
    (attempt_dir / "stderr.txt").write_bytes(stderr)
    (attempt_dir / "command.txt").write_text(
        " ".join(shlex.quote(part) for part in command) + "\n", encoding="ascii"
    )
    status, summary_sha, fields = classify(worker_rc, stdout, stderr)

    def field(name: str) -> str:
        return fields.get(name, "UNEMITTED")

    return {
        "ATTEMPT_INDEX": str(attempt.index),
        "PARENT_V7_ORDINAL": attempt.parent.ordinal,
        "CHECKPOINT_ROLE": attempt.parent.role,
        "DEPTH_DELTA": str(attempt.depth_delta),
        "CHILD_U_OFFSET": str(attempt.child_u_offset),
        "CHILD_S_OFFSET": str(attempt.child_s_offset),
        "NODE_ID": attempt.node,
        "CARRIER": attempt.carrier,
        "STATUS": status,
        "WORKER_RC": str(worker_rc),
        "ELAPSED_MS": str(elapsed_ms),
        "INPUT_SHA256": attempt.input_sha256,
        "RUN_CHALLENGE": attempt.run_challenge,
        "ATTEMPT_BINDING": attempt.attempt_binding,
        "STDOUT_SHA256": digest_bytes(stdout),
        "STDERR_SHA256": digest_bytes(stderr),
        "SUMMARY_SHA256": summary_sha,
        "C1_ORIENTATION_UNRESOLVED": field("C1_ORIENTATION_UNRESOLVED"),
        "C2_HULL_ORIENTATION_UNRESOLVED": field("C2_HULL_ORIENTATION_UNRESOLVED"),
        "EVENT1_CHARTS_CERTIFIED": field("EVENT1_CHARTS_CERTIFIED"),
        "EVENT2_CHARTS_CERTIFIED": field("EVENT2_CHARTS_CERTIFIED"),
        "HOMOGENEOUS_COMPUTATION_VALID": field("HOMOGENEOUS_COMPUTATION_VALID"),
        "CERTIFICATE_PASS": field("CERTIFICATE_PASS"),
        "PROBE_PASS": field("PROBE_PASS"),
    }


def paired_cells(rows: Sequence[dict[str, str]]) -> list[list[dict[str, str]]]:
    pairs: list[list[dict[str, str]]] = []
    for offset in range(0, len(rows), len(CARRIERS)):
        pair = list(rows[offset : offset + len(CARRIERS)])
        if len(pair) != len(CARRIERS):
            die("incomplete carrier pair")
        pairs.append(pair)
    return pairs


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    pairs = paired_cells(rows)
    probe_passes = sum(row["PROBE_PASS"] == "true" for row in rows)
    certificate_passes = sum(row["CERTIFICATE_PASS"] == "true" for row in rows)
    rejected = sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in rows)
    crossing = sum(
        row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in rows
    )
    unknown = sum(row["STATUS"] == "UNKNOWN_FAILURE" for row in rows)
    both_probe = sum(all(row["PROBE_PASS"] == "true" for row in pair) for pair in pairs)
    both_rejected = sum(all(row["PROBE_PASS"] == "false" for row in pair) for pair in pairs)
    mixed = len(pairs) - both_probe - both_rejected
    carrier_agreement = sum(
        len({row["STATUS"] for row in pair}) == 1 for pair in pairs
    )
    run_valid = len(rows) == EXPECTED_ATTEMPTS and unknown == 0
    probe_cover = probe_passes == EXPECTED_ATTEMPTS
    certificate_cover = certificate_passes == EXPECTED_ATTEMPTS
    summary = [
        ("SCHEMA", "sounio.cs6.v7b-target23-depth4-cover-summary.v1"),
        ("RUN_COMPLETE", str(len(rows) == EXPECTED_ATTEMPTS).lower()),
        ("RUN_VALID", str(run_valid).lower()),
        ("CHILD_CELLS_EVALUATED", str(len(pairs))),
        ("ATTEMPTS_COMPLETED", str(len(rows))),
        ("PROBE_PASS_ATTEMPTS", str(probe_passes)),
        ("PROBE_REJECTED_ATTEMPTS", str(rejected)),
        ("SECTION_RESIDENT_CROSSING_UNAVAILABLE", str(crossing)),
        ("UNKNOWN_FAILURE", str(unknown)),
        ("CERTIFICATE_PASS_ATTEMPTS", str(certificate_passes)),
        ("BOTH_CARRIERS_PROBE_PASS_CELLS", str(both_probe)),
        ("BOTH_CARRIERS_PROBE_REJECT_CELLS", str(both_rejected)),
        ("MIXED_CARRIER_CELLS", str(mixed)),
        ("CARRIER_STATUS_AGREEMENT_CELLS", str(carrier_agreement)),
        ("PARENT_COVER_EVALUATED", str(run_valid).lower()),
        ("PARENT_PROBE_COVER_PASS", str(run_valid and probe_cover).lower()),
        (
            "PARENT_CERTIFICATE_COVER_PASS",
            str(run_valid and certificate_cover).lower(),
        ),
        ("C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", "false"),
        ("V7_B_ELIGIBILITY", "false"),
        ("V7_B_WINNER", "NONE"),
        ("PROMOTION_ELIGIBLE", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"),
        ("FPGA_EXECUTION", "false"),
    ]
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in summary), encoding="ascii"
    )


def write_plan_summary(path: Path, attempts: Sequence[CoverAttempt]) -> None:
    fields = [
        ("SCHEMA", "sounio.cs6.v7b-target23-depth4-cover-plan.v1"),
        ("PLAN_VALID", str(len(attempts) == EXPECTED_ATTEMPTS).lower()),
        ("CHILD_CELLS", str(len(attempts) // len(CARRIERS))),
        ("ATTEMPTS", str(len(attempts))),
        ("PARENT_COVER_EVALUATED", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"),
    ]
    path.write_text(
        "".join(f"{key}={value}\n" for key, value in fields), encoding="ascii"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--jobs", default=32, type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 64:
        die("jobs must be in [1,64]")
    root = Path.cwd()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = build_attempts(root)
    write_tsv(
        out_dir / "coordinate-manifest.tsv",
        COORDINATE_COLUMNS,
        coordinate_rows(attempts),
    )
    if args.plan_only:
        write_plan_summary(out_dir / "plan-summary.txt", attempts)
        print((out_dir / "plan-summary.txt").read_text(encoding="ascii"), end="")
        return
    if args.binary is None or not args.binary.is_file():
        die("--binary must name the prebuilt CAPD worker")
    binary = args.binary.resolve()
    (out_dir / "attempts").mkdir(exist_ok=True)
    contract_sha = digest(root / CONTRACT_REL)
    ledger_sha = digest(root / LEDGER_REL)

    def execute(attempt: CoverAttempt) -> dict[str, str]:
        return run_attempt(
            root, binary, attempt, args.timeout, out_dir, contract_sha, ledger_sha
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(execute, attempts))
    results.sort(key=lambda row: int(row["ATTEMPT_INDEX"]))
    write_tsv(out_dir / "results.tsv", RESULT_COLUMNS, results)
    write_summary(out_dir / "summary.txt", results)
    files = (
        root / CONTRACT_REL,
        root / PARENT_RESULTS_REL,
        root / LEDGER_REL,
        root / WORKER_REL,
        out_dir / "coordinate-manifest.tsv",
        out_dir / "results.tsv",
        out_dir / "summary.txt",
        binary,
    )
    (out_dir / "files.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files),
        encoding="ascii",
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
