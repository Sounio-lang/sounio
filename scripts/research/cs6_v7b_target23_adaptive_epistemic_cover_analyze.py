#!/usr/bin/env python3
"""Build a retrospective adaptive epistemic-cover receipt from retained runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import tarfile
from collections import defaultdict
from pathlib import Path

from cs6_v7b_target23_epistemic_intersection_analyze import intersection, parse_attempt


CARRIERS = ("C0HORect2Set", "C0Rect2Set")
INTERSECTION_NAMES = ("c1", "c2", "affine", "resident", "homogeneous", "liouville")
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


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def archive_member(bundle: tarfile.TarFile, name: str) -> bytes:
    cache = getattr(bundle, "_cs6_data_cache", None)
    if cache is None:
        cache = {}
        for member in bundle.getmembers():
            if member.name == "./results.tsv" or member.name.endswith("/stdout.txt"):
                if member.name in cache:
                    raise ValueError(f"duplicate archive member: {member.name}")
                stream = bundle.extractfile(member)
                if stream is None:
                    raise ValueError(f"archive member is unreadable: {member.name}")
                cache[member.name] = stream.read()
        setattr(bundle, "_cs6_data_cache", cache)
    try:
        return cache[f"./{name}"]
    except KeyError as error:
        raise ValueError(f"archive member is missing: {name}") from error


def parse_results(bundle: tarfile.TarFile) -> list[dict[str, str]]:
    data = archive_member(bundle, "results.tsv")
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise ValueError("noncanonical results.tsv")
    return list(csv.DictReader(io.StringIO(data.decode("ascii")), delimiter="\t"))


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


def group_pairs(rows: list[dict[str, str]], keys: tuple[str, ...]) -> dict[tuple[str, ...], list[dict[str, str]]]:
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    for key, group in groups.items():
        if len(group) != 2 or {row["CARRIER"] for row in group} != set(CARRIERS):
            raise ValueError(f"carrier pair mismatch: {key}")
    return groups


def certificate_row(
    bundle: tarfile.TarFile,
    row: dict[str, str],
    layer: str,
    leaf_id: str,
    parent_cell: str,
    identity: str,
) -> dict[str, str]:
    stdout = archive_member(bundle, f"attempts/{identity}/stdout.txt")
    if sha256_bytes(stdout) != row["STDOUT_SHA256"]:
        raise ValueError(f"stdout digest mismatch: {identity}")
    intervals, flags = parse_attempt(stdout.decode("ascii"), identity)
    required_flags = {
        "TERMINAL_CERTIFIED", "CERTIFICATE_PASS", "STRUCTURAL_PASS",
        "HOMOGENEOUS_COMPUTATION_VALID", "PROBE_PASS",
    }
    if not required_flags <= set(flags):
        raise ValueError(f"required worker flags missing: {identity}")
    joint = intersection(intervals, INTERSECTION_NAMES)
    liouville = intervals["liouville"]
    legacy = flags.get("CERTIFICATE_PASS", True) or flags.get("TERMINAL_CERTIFIED", True)
    joint_equals = bool(
        joint is not None
        and joint.lower.value == liouville.lower.value
        and joint.upper.value == liouville.upper.value
    )
    certificate = bool(
        row["PROBE_PASS"] == "true"
        and flags.get("PROBE_PASS", False)
        and flags.get("STRUCTURAL_PASS", False)
        and flags.get("HOMOGENEOUS_COMPUTATION_VALID", False)
        and not legacy
        and liouville.upper.value < 0
        and joint is not None
        and joint.upper.value < 0
    )
    return {
        "LAYER": layer,
        "ATTEMPT_INDEX": row["ATTEMPT_INDEX"],
        "LEAF_ID": leaf_id,
        "PARENT_DEPTH4_CELL_INDEX": parent_cell,
        "CARRIER": row["CARRIER"],
        "LEGACY_CERTIFICATE_PASS": str(legacy).lower(),
        "STRUCTURAL_PASS": str(flags.get("STRUCTURAL_PASS", False)).lower(),
        "HOMOGENEOUS_COMPUTATION_VALID": str(
            flags.get("HOMOGENEOUS_COMPUTATION_VALID", False)
        ).lower(),
        "PROBE_PASS": str(flags.get("PROBE_PASS", False)).lower(),
        "LIOUVILLE_LOWER": liouville.lower.token,
        "LIOUVILLE_UPPER": liouville.upper.token,
        "JOINT_LOWER": joint.lower.token if joint else "NA",
        "JOINT_UPPER": joint.upper.token if joint else "NA",
        "JOINT_EQUALS_LIOUVILLE": str(joint_equals).lower(),
        "EPISTEMIC_CERTIFICATE_PASS": str(certificate).lower(),
        "STDOUT_SHA256": row["STDOUT_SHA256"],
    }


def write_tsv(path: Path, columns: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def analyze(depth4_archive: Path, depth5_archive: Path, output_dir: Path) -> None:
    certificates: list[dict[str, str]] = []
    leaves: list[dict[str, str]] = []
    with tarfile.open(depth4_archive, "r:gz") as d4, tarfile.open(depth5_archive, "r:gz") as d5:
        d4_rows = parse_results(d4)
        d5_rows = parse_results(d5)
        if len(d4_rows) != 512 or len(d5_rows) != 200:
            raise ValueError("input attempt cardinality mismatch")
        d4_pairs = group_pairs(d4_rows, ("CHILD_U_OFFSET", "CHILD_S_OFFSET"))
        d5_pairs = group_pairs(
            d5_rows, ("SOURCE_CELL_INDEX", "SUB_U_OFFSET", "SUB_S_OFFSET")
        )
        if len(d4_pairs) != 256 or len(d5_pairs) != 100:
            raise ValueError("input leaf cardinality mismatch")

        d4_pass: dict[int, list[dict[str, str]]] = {}
        d4_reject: dict[int, list[dict[str, str]]] = {}
        for pair in d4_pairs.values():
            cell_index = (int(pair[0]["ATTEMPT_INDEX"]) + 1) // 2
            statuses = {row["PROBE_PASS"] for row in pair}
            if statuses == {"true"}:
                d4_pass[cell_index] = pair
            elif statuses == {"false"}:
                d4_reject[cell_index] = pair
            else:
                raise ValueError(f"mixed carrier status in depth4 cell {cell_index}")
        if len(d4_pass) != 231 or len(d4_reject) != 25:
            raise ValueError("depth4 pass/reject population mismatch")

        refined_children: dict[int, set[tuple[int, int]]] = defaultdict(set)
        for pair in d5_pairs.values():
            source = int(pair[0]["SOURCE_CELL_INDEX"])
            refined_children[source].add(
                (int(pair[0]["SUB_U_OFFSET"]), int(pair[0]["SUB_S_OFFSET"]))
            )
            if any(row["PROBE_PASS"] != "true" for row in pair):
                raise ValueError(f"depth5 leaf is not a paired probe pass: {source}")
        if set(refined_children) != set(d4_reject):
            raise ValueError("depth5 refined-parent set does not match depth4 rejected set")
        expected_children = {(0, 0), (0, 1), (1, 0), (1, 1)}
        if any(children != expected_children for children in refined_children.values()):
            raise ValueError("depth5 child partition is incomplete")

        for cell_index, pair in sorted(d4_pass.items()):
            representative = pair[0]
            leaf_id = representative["NODE_ID"]
            leaf_certificates = []
            for row in sorted(pair, key=lambda item: CARRIERS.index(item["CARRIER"])):
                certificate = certificate_row(
                    d4, row, "D4", leaf_id, str(cell_index), d4_identity(row)
                )
                certificates.append(certificate)
                leaf_certificates.append(certificate)
            certified = sum(
                item["EPISTEMIC_CERTIFICATE_PASS"] == "true" for item in leaf_certificates
            )
            leaves.append({
                "LAYER": "D4", "LEAF_ID": leaf_id,
                "PARENT_DEPTH4_CELL_INDEX": str(cell_index),
                "DEPTH4_U_OFFSET": representative["CHILD_U_OFFSET"],
                "DEPTH4_S_OFFSET": representative["CHILD_S_OFFSET"],
                "SUB_U_OFFSET": "NA", "SUB_S_OFFSET": "NA",
                "CARRIERS_PRESENT": str(len(leaf_certificates)),
                "CERTIFIED_ATTEMPTS": str(certified),
                "PAIR_CERTIFICATE_PASS": str(certified == 2).lower(),
            })

        for key, pair in sorted(
            d5_pairs.items(), key=lambda item: tuple(map(int, item[0]))
        ):
            representative = pair[0]
            source = int(representative["SOURCE_CELL_INDEX"])
            rejected_rep = d4_reject[source][0]
            if (
                representative["SOURCE_U_OFFSET"] != rejected_rep["CHILD_U_OFFSET"]
                or representative["SOURCE_S_OFFSET"] != rejected_rep["CHILD_S_OFFSET"]
            ):
                raise ValueError(f"depth5 source-coordinate mismatch: {source}")
            leaf_id = representative["NODE_ID"]
            leaf_certificates = []
            for row in sorted(pair, key=lambda item: CARRIERS.index(item["CARRIER"])):
                certificate = certificate_row(
                    d5, row, "D5", leaf_id, str(source), d5_identity(row)
                )
                certificates.append(certificate)
                leaf_certificates.append(certificate)
            certified = sum(
                item["EPISTEMIC_CERTIFICATE_PASS"] == "true" for item in leaf_certificates
            )
            leaves.append({
                "LAYER": "D5", "LEAF_ID": leaf_id,
                "PARENT_DEPTH4_CELL_INDEX": str(source),
                "DEPTH4_U_OFFSET": representative["SOURCE_U_OFFSET"],
                "DEPTH4_S_OFFSET": representative["SOURCE_S_OFFSET"],
                "SUB_U_OFFSET": representative["SUB_U_OFFSET"],
                "SUB_S_OFFSET": representative["SUB_S_OFFSET"],
                "CARRIERS_PRESENT": str(len(leaf_certificates)),
                "CERTIFIED_ATTEMPTS": str(certified),
                "PAIR_CERTIFICATE_PASS": str(certified == 2).lower(),
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_tsv(output_dir / "certificates.tsv", CERTIFICATE_COLUMNS, certificates)
    write_tsv(output_dir / "leaves.tsv", LEAF_COLUMNS, leaves)
    counts = {
        "DEPTH4_SOURCE_CELLS": 256,
        "DEPTH4_SELECTED_LEAVES": len(d4_pass),
        "DEPTH4_REFINED_PARENTS": len(d4_reject),
        "DEPTH4_SELECTED_ATTEMPTS": sum(row["LAYER"] == "D4" for row in certificates),
        "DEPTH5_SELECTED_LEAVES": len(d5_pairs),
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
        "CARRIER_C0HORECT2SET": sum(
            row["CARRIER"] == "C0HORect2Set" for row in certificates
        ),
        "CARRIER_C0RECT2SET": sum(row["CARRIER"] == "C0Rect2Set" for row in certificates),
    }
    adaptive_pass = (
        counts["DEPTH4_SELECTED_LEAVES"] == 231
        and counts["DEPTH4_REFINED_PARENTS"] == 25
        and counts["DEPTH5_SELECTED_LEAVES"] == 100
        and counts["ADAPTIVE_LEAVES"] == 331
        and counts["SELECTED_ATTEMPTS"] == 662
        and counts["LEGACY_CERTIFICATE_FALSE"] == 662
        and counts["STRUCTURAL_PASS"] == 662
        and counts["HOMOGENEOUS_COMPUTATION_VALID"] == 662
        and counts["PROBE_PASS"] == 662
        and counts["JOINT_EQUALS_LIOUVILLE"] == 662
        and counts["EPISTEMIC_CERTIFICATE_PASS"] == 662
        and counts["PAIRED_LEAF_CERTIFICATE_PASS"] == 331
        and counts["CARRIER_C0HORECT2SET"] == 331
        and counts["CARRIER_C0RECT2SET"] == 331
    )
    summary = [
        "SCHEMA=sounio.cs6.v7b-target23-adaptive-epistemic-cover-summary.v1",
        f"DEPTH4_ARCHIVE_SHA256={sha256_file(depth4_archive)}",
        f"DEPTH5_ARCHIVE_SHA256={sha256_file(depth5_archive)}",
        "ANALYSIS_MODE=RETROSPECTIVE_RETAINED_RECEIPT_AUDIT",
    ]
    summary.extend(f"{key}={value}" for key, value in counts.items())
    summary.extend((
        "ADAPTIVE_TOPOLOGY_PASS=true",
        f"ADAPTIVE_EPISTEMIC_COVER_PASS={str(adaptive_pass).lower()}",
        "PROSPECTIVE_INDEPENDENT_REPLAY_COMPLETED=false",
        "LEGACY_CERTIFICATE_RECLASSIFIED=false",
        "GLOBAL_HPG_CERTIFICATE=false",
        "V7_B_ELIGIBILITY=false",
        "V7_B_WINNER=NONE",
        "PROMOTION_ELIGIBLE=false",
        "OPEN_PROBLEM_SOLVED=false",
        "NOVELTY_OR_PRIORITY_CLAIMED=false",
        "FPGA_EXECUTION=false",
    ))
    (output_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="ascii")
    if not adaptive_pass:
        raise SystemExit("adaptive epistemic cover audit failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("depth4_archive", type=Path)
    parser.add_argument("depth5_archive", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    analyze(args.depth4_archive, args.depth5_archive, args.output_dir)


if __name__ == "__main__":
    main()
