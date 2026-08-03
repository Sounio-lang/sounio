#!/usr/bin/env python3
"""Verify a prospective target-23 replay from raw per-attempt evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from pathlib import Path

from cs6_v7b_subdivision_ladder_run import CARRIERS, extract_summary
from cs6_v7b_target23_prospective_epistemic_replay_run import (
    CONTRACT_REL,
    EXPECTED_ATTEMPTS,
    EXPECTED_LEAVES,
    LEDGER_REL,
    RESULT_COLUMNS,
    build_attempts,
    digest,
    digest_bytes,
    raw_certificate,
)


SUMMARY_KEYS = (
    "SCHEMA", "PRE_EXECUTION_GIT_COMMIT", "RUN_COMPLETE", "FRESH_ATTEMPT_PROCESSES",
    "LEAVES_EVALUATED", "ATTEMPTS_COMPLETED", "EPISTEMIC_CERTIFICATE_PASS_ATTEMPTS",
    "PAIR_CERTIFICATE_PASS_LEAVES", "ALL_ATTEMPTS_CERTIFIED", "ALL_PAIRS_CERTIFIED",
    "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED", "GLOBAL_HPG_CERTIFICATE",
    "V7_B_ELIGIBILITY", "V7_B_WINNER", "PROMOTION_ELIGIBLE", "OPEN_PROBLEM_SOLVED",
    "NOVELTY_OR_PRIORITY_CLAIMED", "FPGA_EXECUTION",
)
KEY_RE = re.compile(r"[A-Z][A-Z0-9_]*")


def fail(message: str) -> None:
    raise SystemExit(f"prospective replay verify error: {message}")


def canonical(path: Path) -> str:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical text: {path}")
    try:
        return raw.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {path}") from error


def fields(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in canonical(path).splitlines():
        if line.count("=") != 1:
            fail(f"malformed field line: {path}:{line}")
        key, value = line.split("=", 1)
        if not KEY_RE.fullmatch(key) or not value or key in result:
            fail(f"empty or duplicate field: {path}:{key}")
        result[key] = value
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify(receipt: Path, source_commit: str) -> dict[str, str]:
    root = Path.cwd()
    summary = fields(receipt / "summary.txt")
    if tuple(summary) != SUMMARY_KEYS:
        fail("summary key order or population drifted")
    if summary["SCHEMA"] != "sounio.cs6.v7b-target23-prospective-epistemic-replay-summary.v1":
        fail("summary schema drifted")
    if summary["PRE_EXECUTION_GIT_COMMIT"] != source_commit:
        fail("pre-execution commit mismatch")
    rows = list(csv.DictReader(canonical(receipt / "results.tsv").splitlines(), delimiter="\t"))
    if len(rows) != EXPECTED_ATTEMPTS or not rows or tuple(rows[0]) != RESULT_COLUMNS:
        fail("result shape drifted")
    try:
        planned = build_attempts(root, source_commit)
    except (SystemExit, ValueError) as error:
        fail(f"could not reconstruct frozen attempt plan: {error}")
    certified = 0
    pairs: dict[int, list[dict[str, str]]] = {}
    for index, (row, expected) in enumerate(zip(rows, planned, strict=True), 1):
        leaf = expected.leaf
        expected_fields = {
            "ATTEMPT_INDEX": str(index), "LEAF_INDEX": str(leaf.index),
            "LAYER": leaf.layer, "LEAF_ID": leaf.leaf_id,
            "PARENT_DEPTH4_CELL_INDEX": str(leaf.parent_cell),
            "DEPTH4_U_OFFSET": str(leaf.depth4_u), "DEPTH4_S_OFFSET": str(leaf.depth4_s),
            "SUB_U_OFFSET": leaf.sub_u, "SUB_S_OFFSET": leaf.sub_s,
            "U_DEPTH": str(leaf.u_depth), "U_INDEX": str(leaf.u_index),
            "S_DEPTH": str(leaf.s_depth), "S_INDEX": str(leaf.s_index),
            "CARRIER": expected.carrier, "INPUT_SHA256": leaf.input_sha256,
            "RUN_CHALLENGE": expected.challenge, "ATTEMPT_BINDING": expected.binding,
        }
        for key, value in expected_fields.items():
            if row[key] != value:
                fail(f"attempt {index} field mismatch: {key}")
        attempt_dir = receipt / "attempts" / expected.identity
        stdout_path, stderr_path = attempt_dir / "stdout.txt", attempt_dir / "stderr.txt"
        command_path = attempt_dir / "command.txt"
        if not stdout_path.is_file() or not stderr_path.is_file() or not command_path.is_file():
            fail(f"attempt evidence missing: {index}")
        stdout, stderr = stdout_path.read_bytes(), stderr_path.read_bytes()
        if row["STDOUT_SHA256"] != hashlib.sha256(stdout).hexdigest():
            fail(f"stdout hash mismatch: {index}")
        if row["STDERR_SHA256"] != hashlib.sha256(stderr).hexdigest():
            fail(f"stderr hash mismatch: {index}")
        summary_sha, worker_fields = extract_summary(stdout)
        if row["SUMMARY_SHA256"] != summary_sha:
            fail(f"worker summary hash mismatch: {index}")
        if row["STATUS"] != "DESCENDANT_PROBE_PASS" or row["WORKER_RC"] != "0":
            fail(f"prospective attempt did not pass probe: {index}:{row['STATUS']}")
        expected_certificate = raw_certificate(stdout, row["STATUS"], worker_fields.get("PROBE_PASS", "UNEMITTED"))
        for key, value in expected_certificate.items():
            if row[key] != value:
                fail(f"attempt {index} certificate mismatch: {key}")
        if row["EPISTEMIC_CERTIFICATE_PASS"] != "true":
            fail(f"epistemic certificate failed: {index}")
        if row["LEGACY_CERTIFICATE_PASS"] != "false" or row["TERMINAL_CERTIFIED"] != "false":
            fail(f"legacy flags not explicitly false: {index}")
        command = canonical(command_path).split()
        expected_command_tail = (
            str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth), str(leaf.s_index),
            leaf.input_sha256, expected.challenge, expected.carrier,
            digest(root / CONTRACT_REL), digest(root / LEDGER_REL), digest(root / CONTRACT_REL),
            digest_bytes(leaf.leaf_id.encode("ascii") + b"\n"), expected.binding,
        )
        if len(command) != 13 or Path(command[0]).name != "worker-binary":
            fail(f"command executable or arity mismatch: {index}")
        if tuple(command[1:]) != expected_command_tail:
            fail(f"command binding mismatch: {index}")
        certified += 1
        pairs.setdefault(leaf.index, []).append(row)
    if len(pairs) != EXPECTED_LEAVES:
        fail("leaf-pair population drifted")
    pair_pass = 0
    for leaf_index, pair in pairs.items():
        if len(pair) != 2 or tuple(row["CARRIER"] for row in pair) != CARRIERS:
            fail(f"carrier pair drifted: leaf {leaf_index}")
        if all(row["EPISTEMIC_CERTIFICATE_PASS"] == "true" for row in pair):
            pair_pass += 1
    expected_summary = {
        "RUN_COMPLETE": "true", "FRESH_ATTEMPT_PROCESSES": str(EXPECTED_ATTEMPTS),
        "LEAVES_EVALUATED": str(EXPECTED_LEAVES),
        "ATTEMPTS_COMPLETED": str(EXPECTED_ATTEMPTS),
        "EPISTEMIC_CERTIFICATE_PASS_ATTEMPTS": str(certified),
        "PAIR_CERTIFICATE_PASS_LEAVES": str(pair_pass),
        "ALL_ATTEMPTS_CERTIFIED": "true", "ALL_PAIRS_CERTIFIED": "true",
        "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED": "false",
        "GLOBAL_HPG_CERTIFICATE": "false", "V7_B_ELIGIBILITY": "false",
        "V7_B_WINNER": "NONE", "PROMOTION_ELIGIBLE": "false",
        "OPEN_PROBLEM_SOLVED": "false", "NOVELTY_OR_PRIORITY_CLAIMED": "false",
        "FPGA_EXECUTION": "false",
    }
    for key, value in expected_summary.items():
        if summary[key] != value:
            fail(f"summary mismatch: {key}")

    provenance = receipt / "provenance"
    git_head = canonical(provenance / "git-head.txt").strip()
    if git_head != source_commit:
        fail("Slurm payload Git identity mismatch")
    slurm = fields(provenance / "slurm-context.txt")
    if not slurm.get("SLURM_JOB_ID", "").isdigit() or slurm.get("SLURM_CPUS_PER_TASK") != "32":
        fail("Slurm execution attestation invalid")
    build = fields(provenance / "fresh-build-attestation.txt")
    source_path = root / "scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp"
    if build.get("WORKER_SOURCE_SHA256") != sha256(source_path):
        fail("fresh build source hash mismatch")
    files_text = canonical(receipt / "files.sha256")
    binary_lines = [line for line in files_text.splitlines() if line.endswith("  worker-binary")]
    if len(binary_lines) != 1 or build.get("WORKER_BINARY_SHA256") != binary_lines[0].split()[0]:
        fail("fresh build binary hash mismatch")
    if build.get("BUILD_AFTER_PRE_EXECUTION_COMMIT") != "true":
        fail("fresh build ordering not attested")
    return {
        "SLURM_JOB_ID": slurm["SLURM_JOB_ID"],
        "ATTEMPTS_VERIFIED": str(certified),
        "LEAF_PAIRS_VERIFIED": str(pair_pass),
        "SOURCE_FRESH_SLURM_REPLAY_VERIFIED": "true",
        "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED": "true",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    result = verify(args.receipt, args.source_commit)
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-prospective-epistemic-replay-verification.v1")
    print(f"SLURM_JOB_ID={result['SLURM_JOB_ID']}")
    print(f"ATTEMPTS_VERIFIED={result['ATTEMPTS_VERIFIED']}")
    print(f"LEAF_PAIRS_VERIFIED={result['LEAF_PAIRS_VERIFIED']}")
    print(f"SOURCE_FRESH_SLURM_REPLAY_VERIFIED={result['SOURCE_FRESH_SLURM_REPLAY_VERIFIED']}")
    print(f"PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED={result['PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED']}")
    print("GLOBAL_HPG_CERTIFICATE=false")
    print("V7_B_ELIGIBILITY=false")
    print("OPEN_PROBLEM_SOLVED=false")
    print("FPGA_EXECUTION=false")


if __name__ == "__main__":
    main()
