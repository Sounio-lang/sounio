#!/usr/bin/env python3
"""Require the Arb validated-center verifier to reject evidence tampering."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from cs6_v7b_target23_arb_validated_minmargin_verify import verify


Mutation = tuple[str, str, Callable[[Path], None]]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise ValueError(f"mutation anchor count is {text.count(old)}: {old}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def rehash_stdout(root: Path) -> None:
    stdout = root / "worker.stdout.txt"
    summary = root / "execution-summary.txt"
    text = summary.read_text(encoding="ascii")
    lines = text.splitlines()
    updated = []
    for line in lines:
        if line.startswith("STDOUT_SHA256="):
            line = f"STDOUT_SHA256={hashlib.sha256(stdout.read_bytes()).hexdigest()}"
        updated.append(line)
    summary.write_text("\n".join(updated) + "\n", encoding="ascii")


def mutate_stdout(root: Path, old: str, new: str) -> None:
    replace_once(root / "worker.stdout.txt", old, new)
    rehash_stdout(root)


def append_stdout(root: Path) -> None:
    with (root / "worker.stdout.txt").open("ab") as stream:
        stream.write(b"UNEXPECTED_FIELD=true\n")
    rehash_stdout(root)


def replace_git_head(root: Path) -> None:
    path = root / "provenance/git-head.txt"
    replace_once(path, path.read_text(encoding="ascii").strip(), "0" * 40)


def mutations(receipt: Path) -> list[Mutation]:
    fields = {}
    for line in (receipt / "worker.stdout.txt").read_text(encoding="ascii").splitlines():
        key, value = line.split("=", 1)
        fields[key] = value
    return [
        ("M01", "summary_completion", lambda root: replace_once(
            root / "execution-summary.txt", "RUN_COMPLETE=true", "RUN_COMPLETE=false")),
        ("M02", "challenge", lambda root: mutate_stdout(
            root, f"RUN_CHALLENGE={fields['RUN_CHALLENGE']}", f"RUN_CHALLENGE={'0' * 64}")),
        ("M03", "attempt_binding", lambda root: mutate_stdout(
            root, f"ATTEMPT_BINDING={fields['ATTEMPT_BINDING']}", f"ATTEMPT_BINDING={'0' * 64}")),
        ("M04", "picard_accounting", lambda root: mutate_stdout(
            root, f"PICARD_CONTAINMENTS={fields['PICARD_CONTAINMENTS']}", "PICARD_CONTAINMENTS=1")),
        ("M05", "determinant_upper", lambda root: mutate_stdout(
            root, f"DETERMINANT_UPPER_Q={fields['DETERMINANT_UPPER_Q']}", "DETERMINANT_UPPER_Q=0")),
        ("M06", "determinant_lower", lambda root: mutate_stdout(
            root, f"DETERMINANT_LOWER_Q={fields['DETERMINANT_LOWER_Q']}", "DETERMINANT_LOWER_Q=0")),
        ("M07", "initial_normal", lambda root: mutate_stdout(
            root, f"INITIAL_NORMAL_LOWER_Q={fields['INITIAL_NORMAL_LOWER_Q']}", "INITIAL_NORMAL_LOWER_Q=0")),
        ("M08", "validated_certificate", lambda root: mutate_stdout(
            root, "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE=true",
            "INDEPENDENT_VALIDATED_CENTER_ORBIT_CERTIFICATE=false")),
        ("M09", "forbidden_leaf_wide_claim", lambda root: mutate_stdout(
            root, "LEAF_WIDE_CERTIFICATE=false", "LEAF_WIDE_CERTIFICATE=true")),
        ("M10", "forbidden_open_problem_claim", lambda root: mutate_stdout(
            root, "OPEN_PROBLEM_SOLVED=false", "OPEN_PROBLEM_SOLVED=true")),
        ("M11", "unexpected_worker_field", append_stdout),
        ("M12", "worker_source_identity", lambda root: mutate_stdout(
            root, f"WORKER_SOURCE_SHA256={fields['WORKER_SOURCE_SHA256']}",
            f"WORKER_SOURCE_SHA256={'0' * 64}")),
        ("M13", "wheel_identity", lambda root: replace_once(
            root / "provenance/dependency-attestation.txt",
            "PYTHON_FLINT_WHEEL_SHA256=", "PYTHON_FLINT_WHEEL_SHA256=0")),
        ("M14", "slurm_identity", lambda root: replace_once(
            root / "provenance/slurm-context.txt", "SLURM_JOB_ID=", "SLURM_JOB_ID=X")),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    rows: list[tuple[str, str, str, str]] = []
    all_mutations = mutations(args.receipt)
    for mutation_id, target, mutate in all_mutations:
        with tempfile.TemporaryDirectory(prefix=f"cs6-arb-{mutation_id}-") as temporary:
            candidate = Path(temporary) / "receipt"
            shutil.copytree(args.receipt, candidate)
            mutate(candidate)
            try:
                verify(candidate, args.source_commit, args.wheel_sha256)
            except SystemExit as error:
                signature = str(error.code)
                rows.append((mutation_id, target, "rejected", hashlib.sha256(signature.encode()).hexdigest()))
            else:
                rows.append((mutation_id, target, "escaped", "NA"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "mutations.tsv").write_text(
        "MUTATION_ID\tTARGET\tOUTCOME\tFAILURE_SIGNATURE_SHA256\n"
        + "".join("\t".join(row) + "\n" for row in rows),
        encoding="ascii",
    )
    rejected = sum(row[2] == "rejected" for row in rows)
    (args.out_dir / "mutation-summary.txt").write_text(
        "SCHEMA=sounio.cs6.v7b-target23-arb-validated-minmargin-mutations.v1\n"
        f"MUTATION_TESTS={len(rows)}\nMUTATIONS_REJECTED={rejected}\n"
        f"MUTATIONS_ESCAPED={len(rows) - rejected}\n"
        f"MUTATION_GATE_PASS={str(rejected == len(rows)).lower()}\n",
        encoding="ascii",
    )
    if rejected != len(rows):
        raise SystemExit("Arb validated minimum-margin mutation escaped")


if __name__ == "__main__":
    main()
