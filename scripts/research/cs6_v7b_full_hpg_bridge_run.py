#!/usr/bin/env python3
"""Run the prospective six-attempt V7-B bridge matrix.

This runner reuses the existing V7-A carrier-ablation CAPD worker as the
read-only computation engine. It narrows the matrix to the two candidate
carriers and the three V7-A.1 cells, then classifies every downstream result
without promoting partial evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


CONTRACT_REL = Path("scripts/research/cs6_v7b_full_hpg_bridge_contract_v1.txt")
LEDGER_REL = Path("scripts/research/cs6_v7b_full_hpg_bridge_ledger_v1.tsv")
WORKER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-full-hpg-bridge-attempt.v1\0"
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CANDIDATE_CARRIERS = ("C0HORect2Set", "C0Rect2Set")
EXPECTED_CELLS = ("22", "23", "24")
EXPECTED_ATTEMPTS = 6
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ZERO_SHA256 = "0" * 64

RESULT_COLUMNS = (
    "ATTEMPT_INDEX",
    "PARENT_V7_ORDINAL",
    "CHECKPOINT_ROLE",
    "NODE_ID",
    "CARRIER",
    "STATUS",
    "WORKER_RC",
    "ELAPSED_MS",
    "INPUT_SHA256",
    "CELL_CHALLENGE",
    "RUN_CONTRACT_SHA256",
    "MANIFEST_ROW_SHA256",
    "ATTEMPT_BINDING",
    "STDOUT_SHA256",
    "STDERR_SHA256",
    "SUMMARY_SHA256",
    "C1_BOUNDARY",
    "C2_BOUNDARY",
    "SECTION_RESIDENT_CROSSING",
    "DETERMINANT_COMPATIBILITY",
    "PROBE_PASS",
)


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical(path: Path) -> bytes:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        die(f"noncanonical text file: {path}")
    try:
        raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text file: {path}") from error
    return raw


def parse_prefaced_tsv(path: Path, header_prefix: str) -> list[dict[str, str]]:
    lines = canonical(path).decode("ascii").splitlines()
    header_index = next((index for index, line in enumerate(lines) if line.startswith(header_prefix)), -1)
    if header_index < 0:
        die(f"header missing in {path}")
    return list(csv.DictReader(lines[header_index:], delimiter="\t"))


def input_bytes(row: dict[str, str]) -> bytes:
    node = row["NODE_ID"]
    try:
        u_part, s_part = node.split("_")
        u_depth = int(u_part[1:3])
        u_index = int(u_part.split("-")[1])
        s_depth = int(s_part[1:3])
        s_index = int(s_part.split("-")[1])
    except (IndexError, ValueError) as error:
        raise SystemExit(f"cannot parse node id: {node}") from error
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def parse_node(row: dict[str, str]) -> tuple[int, int, int, int]:
    node = row["NODE_ID"]
    u_part, s_part = node.split("_")
    return (
        int(u_part[1:3]),
        int(u_part.split("-")[1]),
        int(s_part[1:3]),
        int(s_part.split("-")[1]),
    )


def parse_kv_line(line: str) -> tuple[str, dict[str, str]]:
    tokens = line.split()
    if not tokens:
        raise ValueError("empty line")
    fields: dict[str, str] = {}
    for token in tokens[1:]:
        if token.count("=") != 1:
            raise ValueError(f"malformed token: {token}")
        key, value = token.split("=", 1)
        fields[key] = value
    return tokens[0], fields


def extract_summary(raw: bytes) -> tuple[str, str, bool, bool, bool]:
    if not raw:
        return ZERO_SHA256, "false", False, False, False
    try:
        text = raw.decode("ascii")
    except UnicodeError:
        return ZERO_SHA256, "false", False, False, False
    summary_lines = [line for line in text.splitlines() if line.startswith("SUMMARY ")]
    if len(summary_lines) != 1:
        return ZERO_SHA256, "false", False, False, False
    marker, fields = parse_kv_line(summary_lines[0])
    if marker != "SUMMARY":
        return ZERO_SHA256, "false", False, False, False
    probe_pass = fields.get("PROBE_PASS", "false")
    c1_ok = fields.get("C1_C2_DP_OVERLAP") == "true"
    c2_ok = fields.get("C2_HULL_ORIENTATION_UNRESOLVED") in {"true", "false"}
    section_ok = (
        fields.get("HOMOGENEOUS_COMPUTATION_VALID") == "true"
        and fields.get("EVENT_ORDER_CERTIFIED") == "true"
        and fields.get("POSTSECTION_PLUS_SIDE") == "true"
    )
    return digest_bytes((summary_lines[0] + "\n").encode("ascii")), probe_pass, c1_ok, c2_ok, section_ok


def classify(worker_rc: int, stdout: bytes, stderr: bytes) -> tuple[str, str, str, str, str, str]:
    summary_sha, probe_pass, c1_ok, c2_ok, section_ok = extract_summary(stdout)
    if worker_rc == 0 and probe_pass == "true":
        determinant = "JOINT_COMPATIBILITY_NOT_VERIFIED_BY_THIS_RUNNER"
        return (
            "FULL_BRIDGE_PROBE_PASS",
            "SATISFIED",
            "SATISFIED",
            "SATISFIED",
            determinant,
            summary_sha,
        )
    if b"one-step Newton crossing was not available" in stderr:
        return (
            "SECTION_RESIDENT_CROSSING_UNAVAILABLE",
            "UNEMITTED",
            "UNEMITTED",
            "CLASSIFIED_NEGATIVE",
            "NOT_EVALUATED",
            ZERO_SHA256,
        )
    if worker_rc == 0:
        return (
            "FULL_BRIDGE_PROBE_REJECTED",
            "SATISFIED" if c1_ok else "REJECTED",
            "SATISFIED" if c2_ok else "REJECTED",
            "SATISFIED" if section_ok else "REJECTED",
            "NOT_EVALUATED",
            summary_sha,
        )
    return ("UNKNOWN_FAILURE", "UNKNOWN", "UNKNOWN", "UNKNOWN", "UNKNOWN", summary_sha)


@dataclass(frozen=True)
class Attempt:
    index: int
    row: dict[str, str]
    input_sha256: str
    cell_challenge: str
    run_contract_sha256: str
    manifest_row_sha256: str
    attempt_binding: str

    @property
    def identity(self) -> str:
        return f"A{self.index:04d}_{self.row['PARENT_V7_ORDINAL']}_{self.row['CARRIER']}"


def build_attempts(ledger_rows: Sequence[dict[str, str]], contract_sha: str, ledger_sha: str) -> list[Attempt]:
    liouville_rows = [
        row
        for row in ledger_rows
        if row["BOUNDARY"] == "LIOUVILLE_CHECKPOINT"
        and row["STATUS"] == "SATISFIED_BY_V7A1"
        and row["CARRIER"] in CANDIDATE_CARRIERS
    ]
    liouville_rows.sort(key=lambda row: (int(row["PARENT_V7_ORDINAL"]), CANDIDATE_CARRIERS.index(row["CARRIER"])))
    if [row["PARENT_V7_ORDINAL"] for row in liouville_rows[::2]] != list(EXPECTED_CELLS):
        die("candidate cell order drifted")
    attempts: list[Attempt] = []
    for index, row in enumerate(liouville_rows, 1):
        raw_input = input_bytes(row)
        input_sha = digest_bytes(raw_input)
        row_binding = "\t".join(row[column] for column in row.keys()).encode("ascii")
        manifest_row_sha = digest_bytes(row_binding + b"\n")
        cell_challenge = digest_bytes(
            b"sounio.cs6.v7b-full-hpg-bridge-cell.v1\0"
            + bytes.fromhex(ledger_sha)
            + b"\0"
            + row["PARENT_V7_ORDINAL"].encode("ascii")
        )
        attempt_binding = digest_bytes(
            ATTEMPT_DOMAIN
            + bytes.fromhex(cell_challenge)
            + b"\0"
            + row["CARRIER"].encode("ascii")
            + b"\0"
            + bytes.fromhex(contract_sha)
        )
        attempts.append(
            Attempt(
                index=index,
                row=row,
                input_sha256=input_sha,
                cell_challenge=cell_challenge,
                run_contract_sha256=contract_sha,
                manifest_row_sha256=manifest_row_sha,
                attempt_binding=attempt_binding,
            )
        )
    if len(attempts) != EXPECTED_ATTEMPTS:
        die("expected exactly six candidate attempts")
    return attempts


def compile_worker(root: Path, out_dir: Path, capd_config: Path, compiler: str) -> Path:
    source = root / WORKER_REL
    worker_sha = digest(source)
    cflags = subprocess.check_output([str(capd_config), "--cflags"], text=True).strip()
    libs = subprocess.check_output([str(capd_config), "--libs"], text=True).strip()
    binary = out_dir / "worker-binary"
    command = [
        compiler,
        "-std=c++17",
        *shlex.split(cflags),
        "-O0",
        f'-DCS6_WORKER_SOURCE_SHA256="{worker_sha}"',
        str(source),
        "-o",
        str(binary),
        *shlex.split(libs),
    ]
    (out_dir / "compile-command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="ascii")
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    (out_dir / "compile-stdout.txt").write_text(completed.stdout, encoding="ascii", errors="replace")
    (out_dir / "compile-stderr.txt").write_text(completed.stderr, encoding="ascii", errors="replace")
    if completed.returncode != 0:
        die(f"worker compile failed rc={completed.returncode}")
    (out_dir / "worker-source.sha256").write_text(f"{worker_sha}  {WORKER_REL}\n", encoding="ascii")
    (out_dir / "worker-binary.sha256").write_text(f"{digest(binary)}  worker-binary\n", encoding="ascii")
    return binary


def run_attempt(root: Path, binary: Path, attempt: Attempt, timeout: float, out_dir: Path) -> dict[str, str]:
    u_depth, u_index, s_depth, s_index = parse_node(attempt.row)
    command = [
        str(binary),
        str(u_depth),
        str(u_index),
        str(s_depth),
        str(s_index),
        attempt.input_sha256,
        attempt.cell_challenge,
        attempt.row["CARRIER"],
        attempt.run_contract_sha256,
        digest(root / LEDGER_REL),
        attempt.run_contract_sha256,
        attempt.manifest_row_sha256,
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
    (attempt_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="ascii")
    status, c1, c2, section, determinant, summary_sha = classify(worker_rc, stdout, stderr)
    return {
        "ATTEMPT_INDEX": str(attempt.index),
        "PARENT_V7_ORDINAL": attempt.row["PARENT_V7_ORDINAL"],
        "CHECKPOINT_ROLE": attempt.row["CHECKPOINT_ROLE"],
        "NODE_ID": attempt.row["NODE_ID"],
        "CARRIER": attempt.row["CARRIER"],
        "STATUS": status,
        "WORKER_RC": str(worker_rc),
        "ELAPSED_MS": str(elapsed_ms),
        "INPUT_SHA256": attempt.input_sha256,
        "CELL_CHALLENGE": attempt.cell_challenge,
        "RUN_CONTRACT_SHA256": attempt.run_contract_sha256,
        "MANIFEST_ROW_SHA256": attempt.manifest_row_sha256,
        "ATTEMPT_BINDING": attempt.attempt_binding,
        "STDOUT_SHA256": digest_bytes(stdout),
        "STDERR_SHA256": digest_bytes(stderr),
        "SUMMARY_SHA256": summary_sha,
        "C1_BOUNDARY": c1,
        "C2_BOUNDARY": c2,
        "SECTION_RESIDENT_CROSSING": section,
        "DETERMINANT_COMPATIBILITY": determinant,
        "PROBE_PASS": "true" if status == "FULL_BRIDGE_PROBE_PASS" else "false",
    }


def write_tsv(path: Path, rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status_counts[row["STATUS"]] = status_counts.get(row["STATUS"], 0) + 1
    unknown = status_counts.get("UNKNOWN_FAILURE", 0)
    passes = status_counts.get("FULL_BRIDGE_PROBE_PASS", 0)
    classified_negatives = status_counts.get("SECTION_RESIDENT_CROSSING_UNAVAILABLE", 0)
    run_valid = unknown == 0 and len(rows) == EXPECTED_ATTEMPTS
    eligible = passes == EXPECTED_ATTEMPTS
    summary = [
        ("SCHEMA", "sounio.cs6.v7b-full-hpg-bridge-execution-summary.v1"),
        ("RUN_COMPLETE", str(len(rows) == EXPECTED_ATTEMPTS).lower()),
        ("RUN_VALID", str(run_valid).lower()),
        ("ATTEMPTS_COMPLETED", str(len(rows))),
        ("FULL_BRIDGE_PROBE_PASS", str(passes)),
        ("SECTION_RESIDENT_CROSSING_UNAVAILABLE", str(classified_negatives)),
        ("UNKNOWN_FAILURE", str(unknown)),
        ("C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", str(eligible).lower()),
        ("FULL_HPG_PIPELINE_EVALUATED", str(eligible).lower()),
        ("V7_B_ELIGIBILITY", str(eligible).lower()),
        ("V7_B_WINNER", "NONE"),
        ("PROMOTION_ELIGIBLE", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"),
        ("FPGA_EXECUTION", "false"),
    ]
    path.write_text("".join(f"{key}={value}\n" for key, value in summary), encoding="ascii")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--capd-config", default="/tmp/capd-build/bin/capd-config", type=Path)
    parser.add_argument("--compiler", default=os.environ.get("CXX", "/usr/bin/x86_64-linux-gnu-g++-13"))
    parser.add_argument("--timeout", default=90.0, type=float)
    args = parser.parse_args()

    root = Path.cwd()
    if not args.capd_config.exists():
        die(f"missing CAPD config: {args.capd_config}")
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "attempts").mkdir(exist_ok=True)

    contract_sha = digest(root / CONTRACT_REL)
    ledger_sha = digest(root / LEDGER_REL)
    ledger_rows = parse_prefaced_tsv(root / LEDGER_REL, "ROW_ID\t")
    attempts = build_attempts(ledger_rows, contract_sha, ledger_sha)
    binary = compile_worker(root, out_dir, args.capd_config, args.compiler)
    results = [run_attempt(root, binary, attempt, args.timeout, out_dir) for attempt in attempts]
    write_tsv(out_dir / "results.tsv", results)
    write_summary(out_dir / "summary.txt", results)
    files = [
        root / CONTRACT_REL,
        root / LEDGER_REL,
        root / WORKER_REL,
        out_dir / "results.tsv",
        out_dir / "summary.txt",
    ]
    (out_dir / "files.sha256").write_text(
        "".join(f"{digest(path)}  {path.relative_to(root) if path.is_relative_to(root) else path.name}\n" for path in files),
        encoding="ascii",
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
