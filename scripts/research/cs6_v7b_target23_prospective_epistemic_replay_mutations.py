#!/usr/bin/env python3
"""Require the prospective replay verifier to reject retained-evidence tampering."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from cs6_v7b_target23_prospective_epistemic_replay_verify import verify


Mutation = tuple[str, str, Callable[[Path], None]]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise ValueError(f"mutation anchor count is {text.count(old)}: {old}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def mutate_first_result(path: Path, field: str, value: str) -> None:
    with path.open("r", encoding="ascii", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
        columns = tuple(rows[0])
    rows[0][field] = value
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def delete_first_result(path: Path) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    path.write_text("\n".join((lines[0], *lines[2:])) + "\n", encoding="ascii")


def append_stdout(path: Path) -> None:
    with path.open("ab") as stream:
        stream.write(b"MUTATED=true\n")


def mutations() -> list[Mutation]:
    return [
        ("M01", "summary_attempt_count", lambda root: replace_once(
            root / "summary.txt", "ATTEMPTS_COMPLETED=662", "ATTEMPTS_COMPLETED=661")),
        ("M02", "summary_pair_verdict", lambda root: replace_once(
            root / "summary.txt", "ALL_PAIRS_CERTIFIED=true", "ALL_PAIRS_CERTIFIED=false")),
        ("M03", "result_challenge", lambda root: mutate_first_result(
            root / "results.tsv", "RUN_CHALLENGE", "0" * 64)),
        ("M04", "result_binding", lambda root: mutate_first_result(
            root / "results.tsv", "ATTEMPT_BINDING", "0" * 64)),
        ("M05", "result_certificate", lambda root: mutate_first_result(
            root / "results.tsv", "EPISTEMIC_CERTIFICATE_PASS", "false")),
        ("M06", "result_joint_upper", lambda root: mutate_first_result(
            root / "results.tsv", "JOINT_UPPER", "0x0p+0")),
        ("M07", "result_carrier", lambda root: mutate_first_result(
            root / "results.tsv", "CARRIER", "C0HOTripletonSet")),
        ("M08", "result_row_deleted", lambda root: delete_first_result(root / "results.tsv")),
        ("M09", "raw_stdout", lambda root: append_stdout(
            root / "attempts/A0001_L001_C0HORect2Set/stdout.txt")),
        ("M10", "slurm_job_identity", lambda root: replace_once(
            root / "provenance/slurm-context.txt", "SLURM_JOB_ID=", "SLURM_JOB_ID=X")),
        ("M11", "fresh_build_order", lambda root: replace_once(
            root / "provenance/fresh-build-attestation.txt",
            "BUILD_AFTER_PRE_EXECUTION_COMMIT=true", "BUILD_AFTER_PRE_EXECUTION_COMMIT=false")),
        ("M12", "source_commit", lambda root: replace_once(
            root / "provenance/git-head.txt", root.joinpath("provenance/git-head.txt").read_text(encoding="ascii").strip(), "0" * 40)),
        ("M13", "forbidden_v7b_claim", lambda root: replace_once(
            root / "summary.txt", "V7_B_ELIGIBILITY=false", "V7_B_ELIGIBILITY=true")),
        ("M14", "forbidden_open_problem_claim", lambda root: replace_once(
            root / "summary.txt", "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows: list[tuple[str, str, str, str]] = []
    for mutation_id, target, mutate in mutations():
        with tempfile.TemporaryDirectory(prefix=f"cs6-prospective-{mutation_id}-") as temporary:
            candidate = Path(temporary) / "receipt"
            shutil.copytree(args.receipt, candidate)
            mutate(candidate)
            try:
                verify(candidate, args.source_commit)
            except SystemExit as error:
                signature = str(error.code)
                rows.append((mutation_id, target, "rejected", hashlib.sha256(signature.encode()).hexdigest()))
            else:
                rows.append((mutation_id, target, "escaped", "NA"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with (args.out_dir / "mutations.tsv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("MUTATION_ID", "TARGET", "OUTCOME", "FAILURE_SIGNATURE_SHA256"))
        writer.writerows(rows)
    rejected = sum(row[2] == "rejected" for row in rows)
    (args.out_dir / "mutation-summary.txt").write_text(
        "SCHEMA=sounio.cs6.v7b-target23-prospective-epistemic-replay-mutations.v1\n"
        f"MUTATION_TESTS={len(rows)}\nMUTATIONS_REJECTED={rejected}\n"
        f"MUTATIONS_ESCAPED={len(rows) - rejected}\n"
        f"MUTATION_GATE_PASS={str(rejected == len(rows)).lower()}\n",
        encoding="ascii",
    )
    if rejected != len(rows):
        raise SystemExit("prospective replay mutation escaped")


if __name__ == "__main__":
    main()
