#!/usr/bin/env python3
"""Independently verify the retained adaptive epistemic-cover receipt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
import tarfile
from collections import defaultdict
from fractions import Fraction
from pathlib import Path


CARRIERS = {"C0HORect2Set", "C0Rect2Set"}
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
CERTIFICATE_COLUMNS = (
    "LAYER", "ATTEMPT_INDEX", "LEAF_ID", "PARENT_DEPTH4_CELL_INDEX", "CARRIER",
    "LEGACY_CERTIFICATE_PASS", "STRUCTURAL_PASS", "HOMOGENEOUS_COMPUTATION_VALID",
    "PROBE_PASS", "LIOUVILLE_LOWER", "LIOUVILLE_UPPER", "JOINT_LOWER", "JOINT_UPPER",
    "JOINT_EQUALS_LIOUVILLE", "EPISTEMIC_CERTIFICATE_PASS", "STDOUT_SHA256",
)
LEAF_COLUMNS = (
    "LAYER", "LEAF_ID", "PARENT_DEPTH4_CELL_INDEX", "DEPTH4_U_OFFSET",
    "DEPTH4_S_OFFSET", "SUB_U_OFFSET", "SUB_S_OFFSET", "CARRIERS_PRESENT",
    "CERTIFIED_ATTEMPTS", "PAIR_CERTIFICATE_PASS",
)


def fail(message: str) -> None:
    raise SystemExit(f"adaptive epistemic cover verify error: {message}")


def canonical_bytes(data: bytes, source: str) -> str:
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        fail(f"noncanonical text: {source}")
    try:
        return data.decode("ascii")
    except UnicodeError as error:
        raise SystemExit(f"non-ASCII text: {source}") from error


def canonical_file(path: Path) -> str:
    return canonical_bytes(path.read_bytes(), str(path))


def parse_tsv_text(text: str, columns: tuple[str, ...], source: str) -> list[dict[str, str]]:
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    if not rows or tuple(rows[0]) != columns:
        fail(f"column mismatch: {source}")
    return rows


def parse_tsv_file(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    return parse_tsv_text(canonical_file(path), columns, str(path))


def archive_bytes(bundle: tarfile.TarFile, name: str) -> bytes:
    cache = getattr(bundle, "_cs6_data_cache", None)
    if cache is None:
        cache = {}
        for member in bundle.getmembers():
            if member.name == "./results.tsv" or member.name.endswith("/stdout.txt"):
                if member.name in cache:
                    fail(f"duplicate archive member: {member.name}")
                stream = bundle.extractfile(member)
                if stream is None:
                    fail(f"archive member unreadable: {member.name}")
                cache[member.name] = stream.read()
        setattr(bundle, "_cs6_data_cache", cache)
    try:
        return cache[f"./{name}"]
    except KeyError:
        fail(f"archive member missing: {name}")


def archive_results(bundle: tarfile.TarFile) -> list[dict[str, str]]:
    text = canonical_bytes(archive_bytes(bundle, "results.tsv"), "archive:results.tsv")
    rows = list(csv.DictReader(io.StringIO(text), delimiter="\t"))
    if not rows:
        fail("empty archive results")
    return rows


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact(token: str) -> Fraction:
    try:
        numerator, denominator = float.fromhex(token).as_integer_ratio()
    except (ValueError, OverflowError) as error:
        raise SystemExit(f"invalid binary64 endpoint: {token}") from error
    return Fraction(numerator, denominator)


def raw_determinants(stdout: bytes, identity: str) -> tuple[dict[str, tuple[str, str]], dict[str, bool]]:
    intervals: dict[str, tuple[str, str]] = {}
    flags: dict[str, bool] = {}
    for line in canonical_bytes(stdout, identity).splitlines():
        head = line.split(" ", 1)[0]
        if head in RAW_INTERVALS:
            name, label = RAW_INTERVALS[head]
            fields = {
                match.group(1): (match.group(2), match.group(3))
                for match in INTERVAL_RE.finditer(line)
            }
            if label not in fields or name in intervals:
                fail(f"raw determinant field mismatch: {identity}:{name}")
            lower, upper = fields[label]
            if exact(lower) > exact(upper):
                fail(f"inverted raw interval: {identity}:{name}")
            intervals[name] = (lower, upper)
        if head in {"LEAF_RESULT", "SUMMARY"}:
            flags.update(
                (match.group(1), match.group(2) == "true")
                for match in BOOL_RE.finditer(line)
            )
    expected = {name for name, _ in RAW_INTERVALS.values()}
    if set(intervals) != expected:
        fail(f"raw determinant population mismatch: {identity}")
    required_flags = {
        "TERMINAL_CERTIFIED", "CERTIFICATE_PASS", "STRUCTURAL_PASS",
        "HOMOGENEOUS_COMPUTATION_VALID", "PROBE_PASS",
    }
    if not required_flags <= set(flags):
        fail(f"required worker flags missing: {identity}")
    return intervals, flags


def d4_identity(row: dict[str, str]) -> str:
    return (
        f"A{int(row['ATTEMPT_INDEX']):04d}_U{int(row['CHILD_U_OFFSET']):02d}_"
        f"S{int(row['CHILD_S_OFFSET']):02d}_{row['CARRIER']}"
    )


def d5_identity(row: dict[str, str]) -> str:
    return (
        f"A{int(row['ATTEMPT_INDEX']):04d}_P{int(row['SOURCE_CELL_INDEX']):03d}_"
        f"DU{row['SUB_U_OFFSET']}_DS{row['SUB_S_OFFSET']}_{row['CARRIER']}"
    )


def grouped(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    result: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        result[tuple(row[key] for key in keys)].append(row)
    for key, pair in result.items():
        if len(pair) != 2 or {row["CARRIER"] for row in pair} != CARRIERS:
            fail(f"unpaired carriers: {key}")
    return result


def expected_certificate(
    bundle: tarfile.TarFile,
    result: dict[str, str],
    layer: str,
    leaf_id: str,
    parent: str,
    identity: str,
) -> dict[str, str]:
    stdout = archive_bytes(bundle, f"attempts/{identity}/stdout.txt")
    if digest_bytes(stdout) != result["STDOUT_SHA256"]:
        fail(f"raw stdout hash mismatch: {identity}")
    intervals, flags = raw_determinants(stdout, identity)
    lowers = [(exact(value[0]), value[0], name) for name, value in intervals.items()]
    uppers = [(exact(value[1]), value[1], name) for name, value in intervals.items()]
    joint_lower = max(lowers, key=lambda item: item[0])
    joint_upper = min(uppers, key=lambda item: item[0])
    if joint_lower[0] > joint_upper[0]:
        fail(f"selected attempt has empty six-way intersection: {identity}")
    liouville_lower, liouville_upper = intervals["liouville"]
    legacy = flags.get("CERTIFICATE_PASS", True) or flags.get("TERMINAL_CERTIFIED", True)
    equals_liouville = (
        joint_lower[0] == exact(liouville_lower)
        and joint_upper[0] == exact(liouville_upper)
    )
    certificate = (
        result["PROBE_PASS"] == "true"
        and flags.get("PROBE_PASS", False)
        and flags.get("STRUCTURAL_PASS", False)
        and flags.get("HOMOGENEOUS_COMPUTATION_VALID", False)
        and not legacy
        and exact(liouville_upper) < 0
        and joint_upper[0] < 0
    )
    return {
        "LAYER": layer,
        "ATTEMPT_INDEX": result["ATTEMPT_INDEX"],
        "LEAF_ID": leaf_id,
        "PARENT_DEPTH4_CELL_INDEX": parent,
        "CARRIER": result["CARRIER"],
        "LEGACY_CERTIFICATE_PASS": str(legacy).lower(),
        "STRUCTURAL_PASS": str(flags.get("STRUCTURAL_PASS", False)).lower(),
        "HOMOGENEOUS_COMPUTATION_VALID": str(
            flags.get("HOMOGENEOUS_COMPUTATION_VALID", False)
        ).lower(),
        "PROBE_PASS": str(flags.get("PROBE_PASS", False)).lower(),
        "LIOUVILLE_LOWER": liouville_lower,
        "LIOUVILLE_UPPER": liouville_upper,
        "JOINT_LOWER": joint_lower[1],
        "JOINT_UPPER": joint_upper[1],
        "JOINT_EQUALS_LIOUVILLE": str(equals_liouville).lower(),
        "EPISTEMIC_CERTIFICATE_PASS": str(certificate).lower(),
        "STDOUT_SHA256": result["STDOUT_SHA256"],
    }


def compare_row(actual: dict[str, str], expected: dict[str, str], source: str) -> None:
    for key, value in expected.items():
        if actual.get(key) != value:
            fail(f"{source} field {key}: {actual.get(key)!r} != {value!r}")


def parse_summary(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in canonical_file(path).splitlines():
        if line.count("=") != 1:
            fail(f"malformed summary line: {line}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            fail(f"invalid summary field: {key}")
        fields[key] = value
    return fields


def verify(depth4_archive: Path, depth5_archive: Path, receipt: Path) -> None:
    certificates = parse_tsv_file(receipt / "certificates.tsv", CERTIFICATE_COLUMNS)
    leaves = parse_tsv_file(receipt / "leaves.tsv", LEAF_COLUMNS)
    summary = parse_summary(receipt / "summary.txt")
    certificate_index = {
        (row["LAYER"], row["ATTEMPT_INDEX"]): row for row in certificates
    }
    if len(certificate_index) != len(certificates):
        fail("duplicate certificate row")
    leaf_index = {(row["LAYER"], row["LEAF_ID"]): row for row in leaves}
    if len(leaf_index) != len(leaves):
        fail("duplicate leaf row")

    expected_certificates: dict[tuple[str, str], dict[str, str]] = {}
    expected_leaves: dict[tuple[str, str], dict[str, str]] = {}
    with tarfile.open(depth4_archive, "r:gz") as d4, tarfile.open(depth5_archive, "r:gz") as d5:
        d4_rows, d5_rows = archive_results(d4), archive_results(d5)
        if len(d4_rows) != 512 or len(d5_rows) != 200:
            fail("input attempt cardinality mismatch")
        d4_pairs = grouped(d4_rows, ("CHILD_U_OFFSET", "CHILD_S_OFFSET"))
        d5_pairs = grouped(d5_rows, ("SOURCE_CELL_INDEX", "SUB_U_OFFSET", "SUB_S_OFFSET"))
        if len(d4_pairs) != 256 or len(d5_pairs) != 100:
            fail("input cell cardinality mismatch")

        passed: dict[int, list[dict[str, str]]] = {}
        rejected: dict[int, list[dict[str, str]]] = {}
        for pair in d4_pairs.values():
            cell = (int(pair[0]["ATTEMPT_INDEX"]) + 1) // 2
            statuses = {row["PROBE_PASS"] for row in pair}
            if statuses == {"true"}:
                passed[cell] = pair
            elif statuses == {"false"}:
                rejected[cell] = pair
            else:
                fail(f"mixed depth4 carrier status: {cell}")
        if len(passed) != 231 or len(rejected) != 25:
            fail("depth4 pass/reject mismatch")

        refined: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for pair in d5_pairs.values():
            source = int(pair[0]["SOURCE_CELL_INDEX"])
            refined[source].add((int(pair[0]["SUB_U_OFFSET"]), int(pair[0]["SUB_S_OFFSET"])))
            if any(row["PROBE_PASS"] != "true" for row in pair):
                fail(f"nonpassing selected depth5 pair: {source}")
        if set(refined) != set(rejected):
            fail("refined source set mismatch")
        four = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if any(children != four for children in refined.values()):
            fail("refined child partition mismatch")

        for cell, pair in sorted(passed.items()):
            representative = pair[0]
            leaf_id = representative["NODE_ID"]
            certified = 0
            for result in pair:
                expected = expected_certificate(
                    d4, result, "D4", leaf_id, str(cell), d4_identity(result)
                )
                expected_certificates[("D4", result["ATTEMPT_INDEX"])] = expected
                certified += expected["EPISTEMIC_CERTIFICATE_PASS"] == "true"
            expected_leaves[("D4", leaf_id)] = {
                "LAYER": "D4", "LEAF_ID": leaf_id,
                "PARENT_DEPTH4_CELL_INDEX": str(cell),
                "DEPTH4_U_OFFSET": representative["CHILD_U_OFFSET"],
                "DEPTH4_S_OFFSET": representative["CHILD_S_OFFSET"],
                "SUB_U_OFFSET": "NA", "SUB_S_OFFSET": "NA",
                "CARRIERS_PRESENT": "2", "CERTIFIED_ATTEMPTS": str(certified),
                "PAIR_CERTIFICATE_PASS": str(certified == 2).lower(),
            }

        for key, pair in sorted(d5_pairs.items(), key=lambda item: tuple(map(int, item[0]))):
            representative = pair[0]
            source = int(representative["SOURCE_CELL_INDEX"])
            source_rep = rejected[source][0]
            if (
                representative["SOURCE_U_OFFSET"] != source_rep["CHILD_U_OFFSET"]
                or representative["SOURCE_S_OFFSET"] != source_rep["CHILD_S_OFFSET"]
            ):
                fail(f"depth5 source coordinate mismatch: {source}")
            leaf_id = representative["NODE_ID"]
            certified = 0
            for result in pair:
                expected = expected_certificate(
                    d5, result, "D5", leaf_id, str(source), d5_identity(result)
                )
                expected_certificates[("D5", result["ATTEMPT_INDEX"])] = expected
                certified += expected["EPISTEMIC_CERTIFICATE_PASS"] == "true"
            expected_leaves[("D5", leaf_id)] = {
                "LAYER": "D5", "LEAF_ID": leaf_id,
                "PARENT_DEPTH4_CELL_INDEX": str(source),
                "DEPTH4_U_OFFSET": representative["SOURCE_U_OFFSET"],
                "DEPTH4_S_OFFSET": representative["SOURCE_S_OFFSET"],
                "SUB_U_OFFSET": representative["SUB_U_OFFSET"],
                "SUB_S_OFFSET": representative["SUB_S_OFFSET"],
                "CARRIERS_PRESENT": "2", "CERTIFIED_ATTEMPTS": str(certified),
                "PAIR_CERTIFICATE_PASS": str(certified == 2).lower(),
            }

    if set(certificate_index) != set(expected_certificates):
        fail("certificate key set mismatch")
    if set(leaf_index) != set(expected_leaves):
        fail("leaf key set mismatch")
    for key, expected in expected_certificates.items():
        compare_row(certificate_index[key], expected, f"certificate {key}")
    for key, expected in expected_leaves.items():
        compare_row(leaf_index[key], expected, f"leaf {key}")

    counts = {
        "DEPTH4_SOURCE_CELLS": 256,
        "DEPTH4_SELECTED_LEAVES": 231,
        "DEPTH4_REFINED_PARENTS": 25,
        "DEPTH4_SELECTED_ATTEMPTS": sum(row["LAYER"] == "D4" for row in certificates),
        "DEPTH5_SELECTED_LEAVES": 100,
        "DEPTH5_SELECTED_ATTEMPTS": sum(row["LAYER"] == "D5" for row in certificates),
        "ADAPTIVE_LEAVES": len(leaves),
        "SELECTED_ATTEMPTS": len(certificates),
        "LEGACY_CERTIFICATE_FALSE": sum(
            row["LEGACY_CERTIFICATE_PASS"] == "false" for row in certificates
        ),
        "STRUCTURAL_PASS": sum(row["STRUCTURAL_PASS"] == "true" for row in certificates),
        "HOMOGENEOUS_COMPUTATION_VALID": sum(
            row["HOMOGENEOUS_COMPUTATION_VALID"] == "true" for row in certificates
        ),
        "PROBE_PASS": sum(row["PROBE_PASS"] == "true" for row in certificates),
        "JOINT_EQUALS_LIOUVILLE": sum(
            row["JOINT_EQUALS_LIOUVILLE"] == "true" for row in certificates
        ),
        "EPISTEMIC_CERTIFICATE_PASS": sum(
            row["EPISTEMIC_CERTIFICATE_PASS"] == "true" for row in certificates
        ),
        "PAIRED_LEAF_CERTIFICATE_PASS": sum(
            row["PAIR_CERTIFICATE_PASS"] == "true" for row in leaves
        ),
        "CARRIER_C0HORECT2SET": sum(row["CARRIER"] == "C0HORect2Set" for row in certificates),
        "CARRIER_C0RECT2SET": sum(row["CARRIER"] == "C0Rect2Set" for row in certificates),
    }
    expected_summary = {
        "SCHEMA": "sounio.cs6.v7b-target23-adaptive-epistemic-cover-summary.v1",
        "DEPTH4_ARCHIVE_SHA256": digest_file(depth4_archive),
        "DEPTH5_ARCHIVE_SHA256": digest_file(depth5_archive),
        "ANALYSIS_MODE": "RETROSPECTIVE_RETAINED_RECEIPT_AUDIT",
        **{key: str(value) for key, value in counts.items()},
        "ADAPTIVE_TOPOLOGY_PASS": "true",
        "ADAPTIVE_EPISTEMIC_COVER_PASS": "true",
        "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED": "false",
        "LEGACY_CERTIFICATE_RECLASSIFIED": "false",
        "GLOBAL_HPG_CERTIFICATE": "false",
        "V7_B_ELIGIBILITY": "false", "V7_B_WINNER": "NONE",
        "PROMOTION_ELIGIBLE": "false", "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false", "FPGA_EXECUTION": "false",
    }
    if summary != expected_summary:
        missing = sorted(set(expected_summary) - set(summary))
        extra = sorted(set(summary) - set(expected_summary))
        mismatched = sorted(
            key for key in set(summary) & set(expected_summary)
            if summary[key] != expected_summary[key]
        )
        fail(f"summary mismatch: missing={missing} extra={extra} values={mismatched}")
    if any(value != 662 for key, value in counts.items() if key in {
        "SELECTED_ATTEMPTS", "LEGACY_CERTIFICATE_FALSE", "STRUCTURAL_PASS",
        "HOMOGENEOUS_COMPUTATION_VALID", "PROBE_PASS", "JOINT_EQUALS_LIOUVILLE",
        "EPISTEMIC_CERTIFICATE_PASS",
    }):
        fail("attempt-level acceptance count mismatch")
    if counts["PAIRED_LEAF_CERTIFICATE_PASS"] != 331:
        fail("paired leaf acceptance count mismatch")
    print("VERIFY_SCHEMA=sounio.cs6.v7b-target23-adaptive-epistemic-cover-verification.v1")
    print("INPUT_ARCHIVES_VERIFIED=true")
    print("ADAPTIVE_TOPOLOGY_VERIFIED=true")
    print("SELECTED_ATTEMPTS_VERIFIED=662")
    print("PAIRED_LEAVES_VERIFIED=331")
    print("ADAPTIVE_EPISTEMIC_COVER_PASS=true")
    print("PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=false")
    print("V7_B_ELIGIBILITY=false")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("depth4_archive", type=Path)
    parser.add_argument("depth5_archive", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    verify(args.depth4_archive, args.depth5_archive, args.receipt)


if __name__ == "__main__":
    main()
