#!/usr/bin/env python3
"""Run a bounded descendant-depth scout over the frozen V7-B cells."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cs6_v7b_full_hpg_bridge_run import compile_worker, digest, digest_bytes, parse_kv_line


PARENT_RESULTS_REL = Path(
    "scripts/research/receipts/cs6_v7b_full_hpg_bridge_execution_v1/results.tsv"
)
LEDGER_REL = Path("scripts/research/cs6_v7b_full_hpg_bridge_ledger_v1.tsv")
WORKER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
PARENTS = ("22", "23", "24")
CARRIERS = ("C0HORect2Set", "C0Rect2Set")
DEPTH_DELTAS = (1, 2, 3, 4)
EXPECTED_ATTEMPTS = len(PARENTS) * len(CARRIERS) * len(DEPTH_DELTAS)
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-subdivision-ladder-attempt.v1\0"
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
ZERO_SHA256 = "0" * 64

RESULT_COLUMNS = (
    "ATTEMPT_INDEX",
    "PARENT_V7_ORDINAL",
    "CHECKPOINT_ROLE",
    "DEPTH_DELTA",
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


def die(message: str) -> None:
    raise SystemExit(message)


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        die(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def parse_node(node: str) -> tuple[int, int, int, int]:
    try:
        u_part, s_part = node.split("_")
        return (
            int(u_part[1:3]),
            int(u_part.split("-")[1]),
            int(s_part[1:3]),
            int(s_part.split("-")[1]),
        )
    except (IndexError, ValueError) as error:
        raise SystemExit(f"cannot parse node id: {node}") from error


def node_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


@dataclass(frozen=True)
class Parent:
    ordinal: str
    role: str
    node: str


@dataclass(frozen=True)
class Attempt:
    index: int
    parent: Parent
    depth_delta: int
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
            f"A{self.index:04d}_P{self.parent.ordinal}_D{self.depth_delta}_"
            f"{self.carrier}"
        )


def load_parents(path: Path) -> list[Parent]:
    rows = list(csv.DictReader(canonical(path).splitlines(), delimiter="\t"))
    parents: dict[str, Parent] = {}
    for row in rows:
        ordinal = row["PARENT_V7_ORDINAL"]
        candidate = Parent(ordinal, row["CHECKPOINT_ROLE"], row["NODE_ID"])
        if ordinal in parents and parents[ordinal] != candidate:
            die(f"parent metadata drifted for ordinal {ordinal}")
        parents[ordinal] = candidate
    if tuple(parents) != PARENTS:
        die("frozen parent order drifted")
    return [parents[ordinal] for ordinal in PARENTS]


def build_attempts(root: Path) -> list[Attempt]:
    parent_results_sha = digest(root / PARENT_RESULTS_REL)
    ledger_sha = digest(root / LEDGER_REL)
    attempts: list[Attempt] = []
    for parent in load_parents(root / PARENT_RESULTS_REL):
        u_depth, u_index, s_depth, s_index = parse_node(parent.node)
        for depth_delta in DEPTH_DELTAS:
            child_u_depth = u_depth + depth_delta
            child_u_index = u_index << depth_delta
            child_s_depth = s_depth + depth_delta
            child_s_index = s_index << depth_delta
            raw_input = input_bytes(
                child_u_depth, child_u_index, child_s_depth, child_s_index
            )
            input_sha = digest_bytes(raw_input)
            for carrier in CARRIERS:
                challenge = digest_bytes(
                    b"sounio.cs6.v7b-subdivision-ladder-challenge.v1\0"
                    + bytes.fromhex(parent_results_sha)
                    + bytes.fromhex(ledger_sha)
                    + parent.ordinal.encode("ascii")
                    + bytes([depth_delta])
                )
                binding = digest_bytes(
                    ATTEMPT_DOMAIN
                    + bytes.fromhex(challenge)
                    + bytes.fromhex(input_sha)
                    + carrier.encode("ascii")
                )
                attempts.append(
                    Attempt(
                        index=len(attempts) + 1,
                        parent=parent,
                        depth_delta=depth_delta,
                        u_depth=child_u_depth,
                        u_index=child_u_index,
                        s_depth=child_s_depth,
                        s_index=child_s_index,
                        carrier=carrier,
                        input_sha256=input_sha,
                        run_challenge=challenge,
                        attempt_binding=binding,
                    )
                )
    if len(attempts) != EXPECTED_ATTEMPTS:
        die("expected exactly 24 ladder attempts")
    return attempts


def extract_summary(raw: bytes) -> tuple[str, dict[str, str]]:
    try:
        text = raw.decode("ascii")
    except UnicodeError:
        return ZERO_SHA256, {}
    lines = [line for line in text.splitlines() if line.startswith("SUMMARY ")]
    if len(lines) != 1:
        return ZERO_SHA256, {}
    marker, fields = parse_kv_line(lines[0])
    if marker != "SUMMARY":
        return ZERO_SHA256, {}
    return digest_bytes((lines[0] + "\n").encode("ascii")), fields


def classify(worker_rc: int, stdout: bytes, stderr: bytes) -> tuple[str, str, dict[str, str]]:
    summary_sha, fields = extract_summary(stdout)
    if worker_rc == 0 and fields.get("PROBE_PASS") == "true":
        return "DESCENDANT_PROBE_PASS", summary_sha, fields
    if b"one-step Newton crossing was not available" in stderr:
        return "SECTION_RESIDENT_CROSSING_UNAVAILABLE", summary_sha, fields
    if worker_rc == 0 and fields.get("PROBE_PASS") == "false":
        return "DESCENDANT_PROBE_REJECTED", summary_sha, fields
    return "UNKNOWN_FAILURE", summary_sha, fields


def run_attempt(
    root: Path,
    binary: Path,
    attempt: Attempt,
    timeout: float,
    out_dir: Path,
    ledger_sha: str,
    parent_results_sha: str,
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
        parent_results_sha,
        ledger_sha,
        parent_results_sha,
        digest_bytes(attempt.parent.node.encode("ascii") + b"\n"),
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


def write_tsv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def first_target_delta(rows: Sequence[dict[str, str]], key: str, value: str) -> str:
    for depth_delta in DEPTH_DELTAS:
        selected = [
            row
            for row in rows
            if row["PARENT_V7_ORDINAL"] == "23"
            and int(row["DEPTH_DELTA"]) == depth_delta
        ]
        if len(selected) == len(CARRIERS) and all(row[key] == value for row in selected):
            return str(depth_delta)
    return "NONE"


def first_target_crossing_recovery(rows: Sequence[dict[str, str]]) -> str:
    for depth_delta in DEPTH_DELTAS:
        selected = [
            row
            for row in rows
            if row["PARENT_V7_ORDINAL"] == "23"
            and int(row["DEPTH_DELTA"]) == depth_delta
        ]
        if len(selected) == len(CARRIERS) and all(
            row["STATUS"]
            in {"DESCENDANT_PROBE_REJECTED", "DESCENDANT_PROBE_PASS"}
            for row in selected
        ):
            return str(depth_delta)
    return "NONE"


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    passes = sum(row["STATUS"] == "DESCENDANT_PROBE_PASS" for row in rows)
    rejected = sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in rows)
    crossing = sum(
        row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in rows
    )
    unknown = sum(row["STATUS"] == "UNKNOWN_FAILURE" for row in rows)
    certificates = sum(row["CERTIFICATE_PASS"] == "true" for row in rows)
    parent_carrier_candidates = all(
        any(
            row["PARENT_V7_ORDINAL"] == parent
            and row["CARRIER"] == carrier
            and row["PROBE_PASS"] == "true"
            for row in rows
        )
        for parent in PARENTS
        for carrier in CARRIERS
    )
    summary = [
        ("SCHEMA", "sounio.cs6.v7b-subdivision-ladder-summary.v1"),
        ("RUN_COMPLETE", str(len(rows) == EXPECTED_ATTEMPTS).lower()),
        ("RUN_VALID", str(len(rows) == EXPECTED_ATTEMPTS and unknown == 0).lower()),
        ("ATTEMPTS_COMPLETED", str(len(rows))),
        ("DESCENDANT_PROBE_PASS", str(passes)),
        ("DESCENDANT_PROBE_REJECTED", str(rejected)),
        ("SECTION_RESIDENT_CROSSING_UNAVAILABLE", str(crossing)),
        ("UNKNOWN_FAILURE", str(unknown)),
        ("CERTIFICATE_PASS", str(certificates)),
        (
            "TARGET_FIRST_CROSSING_RECOVERY_DELTA",
            first_target_crossing_recovery(rows),
        ),
        (
            "TARGET_FIRST_PROBE_PASS_DELTA",
            first_target_delta(rows, "PROBE_PASS", "true"),
        ),
        ("ALL_PARENT_CARRIERS_HAVE_CANDIDATE", str(parent_carrier_candidates).lower()),
        ("DESCENDANT_CANDIDATE_DISCOVERED", str(passes > 0).lower()),
        ("PARENT_COVER_EVALUATED", "false"),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument(
        "--capd-config", default="/tmp/capd-build/bin/capd-config", type=Path
    )
    parser.add_argument(
        "--compiler",
        default=os.environ.get("CXX", "/usr/bin/x86_64-linux-gnu-g++-13"),
    )
    parser.add_argument("--timeout", default=120.0, type=float)
    parser.add_argument("--jobs", default=4, type=int)
    args = parser.parse_args()
    if args.jobs < 1 or args.jobs > 16:
        die("jobs must be in [1,16]")

    root = Path.cwd()
    if not args.capd_config.exists():
        die(f"missing CAPD config: {args.capd_config}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attempts").mkdir(exist_ok=True)
    attempts = build_attempts(root)
    binary = compile_worker(root, out_dir, args.capd_config, args.compiler)
    ledger_sha = digest(root / LEDGER_REL)
    parent_results_sha = digest(root / PARENT_RESULTS_REL)

    def execute(attempt: Attempt) -> dict[str, str]:
        return run_attempt(
            root,
            binary,
            attempt,
            args.timeout,
            out_dir,
            ledger_sha,
            parent_results_sha,
        )

    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        results = list(executor.map(execute, attempts))
    results.sort(key=lambda row: int(row["ATTEMPT_INDEX"]))
    write_tsv(out_dir / "results.tsv", results)
    write_summary(out_dir / "summary.txt", results)
    files = (
        root / PARENT_RESULTS_REL,
        root / LEDGER_REL,
        root / WORKER_REL,
        out_dir / "results.tsv",
        out_dir / "summary.txt",
    )
    (out_dir / "files.sha256").write_text(
        "".join(
            f"{digest(path)}  "
            f"{path.relative_to(root) if path.is_relative_to(root) else path.name}\n"
            for path in files
        ),
        encoding="ascii",
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
