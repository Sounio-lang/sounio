#!/usr/bin/env python3
"""Require the adaptive epistemic-cover verifier to reject receipt mutations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Callable

from cs6_v7b_target23_adaptive_epistemic_cover_verify import verify


Mutation = tuple[str, str, Callable[[Path], None]]


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="ascii")
    if text.count(old) != 1:
        raise ValueError(f"mutation anchor count for {old!r} is {text.count(old)}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def mutate_tsv(path: Path, field: str, value: str) -> None:
    with path.open("r", encoding="ascii", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
        columns = tuple(rows[0])
    rows[0][field] = value
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def delete_first_row(path: Path) -> None:
    lines = path.read_text(encoding="ascii").splitlines()
    if len(lines) < 3:
        raise ValueError("not enough rows to delete")
    path.write_text("\n".join((lines[0], *lines[2:])) + "\n", encoding="ascii")


def mutations() -> list[Mutation]:
    return [
        ("M01", "summary_cover_verdict", lambda root: replace_once(
            root / "summary.txt", "ADAPTIVE_EPISTEMIC_COVER_PASS=true",
            "ADAPTIVE_EPISTEMIC_COVER_PASS=false")),
        ("M02", "summary_attempt_count", lambda root: replace_once(
            root / "summary.txt", "SELECTED_ATTEMPTS=662", "SELECTED_ATTEMPTS=661")),
        ("M03", "summary_depth4_archive_hash", lambda root: replace_once(
            root / "summary.txt",
            "DEPTH4_ARCHIVE_SHA256=c5fd1d6d2dc528c90364c72793dbb4af2a694b2a367d339df158085a263f1fdf",
            "DEPTH4_ARCHIVE_SHA256=05fd1d6d2dc528c90364c72793dbb4af2a694b2a367d339df158085a263f1fdf")),
        ("M04", "certificate_verdict", lambda root: mutate_tsv(
            root / "certificates.tsv", "EPISTEMIC_CERTIFICATE_PASS", "false")),
        ("M05", "certificate_joint_upper", lambda root: mutate_tsv(
            root / "certificates.tsv", "JOINT_UPPER", "0x0p+0")),
        ("M06", "certificate_carrier", lambda root: mutate_tsv(
            root / "certificates.tsv", "CARRIER", "C0HOTripletonSet")),
        ("M07", "certificate_leaf_identity", lambda root: mutate_tsv(
            root / "certificates.tsv", "LEAF_ID", "U00-0000000000_S00-0000000000")),
        ("M08", "certificate_row_deleted", lambda root: delete_first_row(
            root / "certificates.tsv")),
        ("M09", "leaf_pair_verdict", lambda root: mutate_tsv(
            root / "leaves.tsv", "PAIR_CERTIFICATE_PASS", "false")),
        ("M10", "leaf_certified_attempt_count", lambda root: mutate_tsv(
            root / "leaves.tsv", "CERTIFIED_ATTEMPTS", "1")),
        ("M11", "leaf_row_deleted", lambda root: delete_first_row(root / "leaves.tsv")),
        ("M12", "forbidden_v7b_promotion", lambda root: replace_once(
            root / "summary.txt", "V7_B_ELIGIBILITY=false", "V7_B_ELIGIBILITY=true")),
        ("M13", "false_prospective_replay", lambda root: replace_once(
            root / "summary.txt", "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=false",
            "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=true")),
        ("M14", "certificate_stdout_hash", lambda root: mutate_tsv(
            root / "certificates.tsv", "STDOUT_SHA256", "0" * 64)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("depth4_archive", type=Path)
    parser.add_argument("depth5_archive", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    rows: list[tuple[str, str, str, str]] = []
    for mutation_id, target, mutate in mutations():
        with tempfile.TemporaryDirectory(prefix=f"cs6-adaptive-{mutation_id}-") as temporary:
            candidate = Path(temporary) / "receipt"
            shutil.copytree(args.receipt, candidate)
            mutate(candidate)
            try:
                verify(args.depth4_archive, args.depth5_archive, candidate)
            except SystemExit as error:
                signature = str(error.code)
                rows.append((mutation_id, target, "rejected", hashlib.sha256(
                    signature.encode("utf-8")).hexdigest()))
            else:
                rows.append((mutation_id, target, "escaped", "NA"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "mutations.tsv").open("w", encoding="ascii", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("MUTATION_ID", "TARGET", "OUTCOME", "FAILURE_SIGNATURE_SHA256"))
        writer.writerows(rows)
    rejected = sum(row[2] == "rejected" for row in rows)
    summary = (
        "SCHEMA=sounio.cs6.v7b-target23-adaptive-epistemic-cover-mutations.v1\n"
        f"MUTATION_TESTS={len(rows)}\n"
        f"MUTATIONS_REJECTED={rejected}\n"
        f"MUTATIONS_ESCAPED={len(rows) - rejected}\n"
        f"MUTATION_GATE_PASS={str(rejected == len(rows)).lower()}\n"
    )
    (args.output_dir / "mutation-summary.txt").write_text(summary, encoding="ascii")
    if rejected != len(rows):
        raise SystemExit("adaptive epistemic cover mutation escaped")


if __name__ == "__main__":
    main()
