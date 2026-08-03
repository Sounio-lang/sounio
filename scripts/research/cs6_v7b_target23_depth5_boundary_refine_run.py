#!/usr/bin/env python3
"""Refine every rejected depth-4 target-23 boundary cell by one level."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cs6_v7b_full_hpg_bridge_run import digest, digest_bytes
from cs6_v7b_subdivision_ladder_run import CARRIERS, LEDGER_REL, classify, input_bytes, node_id


CONTRACT_REL = Path("scripts/research/cs6_v7b_target23_depth5_boundary_refine_contract_v1.txt")
SOURCE_RESULTS_REL = Path("scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/results.tsv")
SOURCE_COORDINATES_REL = Path("scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/coordinate-manifest.tsv")
SOURCE_BOUNDARY_MAP_REL = Path("scripts/research/receipts/cs6_v7b_target23_depth4_cover_v1/boundary-map.txt")
WORKER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
SOURCE_REJECTED_PARENTS = 25
SOURCE_PASS_CELLS = 231
SUB_OFFSETS = (0, 1)
EXPECTED_CELLS = SOURCE_REJECTED_PARENTS * len(SUB_OFFSETS) ** 2
EXPECTED_ATTEMPTS = EXPECTED_CELLS * len(CARRIERS)
ATTEMPT_DOMAIN = b"sounio.cs6.v7b-target23-depth5-boundary-attempt.v1\0"

RESULT_COLUMNS = (
    "ATTEMPT_INDEX", "SOURCE_CELL_INDEX", "SOURCE_U_OFFSET", "SOURCE_S_OFFSET",
    "SUB_U_OFFSET", "SUB_S_OFFSET", "ORIGINAL_PARENT_DEPTH_DELTA", "NODE_ID", "CARRIER",
    "STATUS", "WORKER_RC", "ELAPSED_MS", "INPUT_SHA256", "RUN_CHALLENGE",
    "ATTEMPT_BINDING", "STDOUT_SHA256", "STDERR_SHA256", "SUMMARY_SHA256",
    "C1_ORIENTATION_UNRESOLVED", "C2_HULL_ORIENTATION_UNRESOLVED",
    "EVENT1_CHARTS_CERTIFIED", "EVENT2_CHARTS_CERTIFIED",
    "HOMOGENEOUS_COMPUTATION_VALID", "CERTIFICATE_PASS", "PROBE_PASS",
)

COORDINATE_COLUMNS = (
    "CELL_INDEX", "SOURCE_CELL_INDEX", "SOURCE_U_OFFSET", "SOURCE_S_OFFSET",
    "SUB_U_OFFSET", "SUB_S_OFFSET", "NODE_ID", "U_DEPTH", "U_INDEX",
    "S_DEPTH", "S_INDEX", "INPUT_SHA256",
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


def read_tsv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(canonical(path).splitlines(), delimiter="\t"))


@dataclass(frozen=True)
class BoundaryParent:
    cell_index: int
    u_offset: int
    s_offset: int
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    node: str


@dataclass(frozen=True)
class RefineAttempt:
    index: int
    source: BoundaryParent
    sub_u_offset: int
    sub_s_offset: int
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
            f"A{self.index:04d}_P{self.source.cell_index:03d}_"
            f"DU{self.sub_u_offset}_DS{self.sub_s_offset}_{self.carrier}"
        )


def load_boundary(root: Path) -> list[BoundaryParent]:
    coordinates = read_tsv(root / SOURCE_COORDINATES_REL)
    results = read_tsv(root / SOURCE_RESULTS_REL)
    if len(coordinates) != 256 or len(results) != 512:
        die("source cover cardinality drifted")
    by_node: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in results:
        by_node[row["NODE_ID"]].append(row)
    parents: list[BoundaryParent] = []
    pass_cells = 0
    for row in coordinates:
        pair = by_node.get(row["NODE_ID"], [])
        if len(pair) != len(CARRIERS) or tuple(item["CARRIER"] for item in pair) != CARRIERS:
            die("source carrier pair drifted")
        statuses = {item["STATUS"] for item in pair}
        probes = {item["PROBE_PASS"] for item in pair}
        if statuses == {"DESCENDANT_PROBE_PASS"} and probes == {"true"}:
            pass_cells += 1
            continue
        if statuses != {"DESCENDANT_PROBE_REJECTED"} or probes != {"false"}:
            die("source cell is neither agreed pass nor agreed rejection")
        parent = BoundaryParent(
            cell_index=int(row["CELL_INDEX"]),
            u_offset=int(row["CHILD_U_OFFSET"]),
            s_offset=int(row["CHILD_S_OFFSET"]),
            u_depth=int(row["U_DEPTH"]),
            u_index=int(row["U_INDEX"]),
            s_depth=int(row["S_DEPTH"]),
            s_index=int(row["S_INDEX"]),
            node=row["NODE_ID"],
        )
        if parent.node != node_id(parent.u_depth, parent.u_index, parent.s_depth, parent.s_index):
            die("source node coordinates drifted")
        parents.append(parent)
    if pass_cells != SOURCE_PASS_CELLS or len(parents) != SOURCE_REJECTED_PARENTS:
        die("source pass/rejection partition drifted")
    return parents


def build_attempts(root: Path) -> list[RefineAttempt]:
    contract_sha = digest(root / CONTRACT_REL)
    source_results_sha = digest(root / SOURCE_RESULTS_REL)
    source_coordinates_sha = digest(root / SOURCE_COORDINATES_REL)
    boundary_map_sha = digest(root / SOURCE_BOUNDARY_MAP_REL)
    attempts: list[RefineAttempt] = []
    for parent in load_boundary(root):
        for du in SUB_OFFSETS:
            for ds in SUB_OFFSETS:
                u_depth = parent.u_depth + 1
                u_index = parent.u_index * 2 + du
                s_depth = parent.s_depth + 1
                s_index = parent.s_index * 2 + ds
                input_sha = digest_bytes(input_bytes(u_depth, u_index, s_depth, s_index))
                for carrier in CARRIERS:
                    challenge = digest_bytes(
                        b"sounio.cs6.v7b-target23-depth5-boundary-challenge.v1\0"
                        + bytes.fromhex(contract_sha)
                        + bytes.fromhex(source_results_sha)
                        + bytes.fromhex(source_coordinates_sha)
                        + bytes.fromhex(boundary_map_sha)
                        + parent.cell_index.to_bytes(2, "big")
                        + bytes([du, ds])
                    )
                    binding = digest_bytes(
                        ATTEMPT_DOMAIN + bytes.fromhex(challenge) + bytes.fromhex(input_sha)
                        + carrier.encode("ascii")
                    )
                    attempts.append(
                        RefineAttempt(
                            len(attempts) + 1, parent, du, ds, u_depth, u_index,
                            s_depth, s_index, carrier, input_sha, challenge, binding
                        )
                    )
    if len(attempts) != EXPECTED_ATTEMPTS:
        die("expected exactly 200 boundary-refinement attempts")
    return attempts


def write_tsv(path: Path, columns: Sequence[str], rows: Sequence[dict[str, str]]) -> None:
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def coordinate_rows(attempts: Sequence[RefineAttempt]) -> list[dict[str, str]]:
    rows = []
    for index, attempt in enumerate(attempts[:: len(CARRIERS)], 1):
        rows.append({
            "CELL_INDEX": str(index),
            "SOURCE_CELL_INDEX": str(attempt.source.cell_index),
            "SOURCE_U_OFFSET": str(attempt.source.u_offset),
            "SOURCE_S_OFFSET": str(attempt.source.s_offset),
            "SUB_U_OFFSET": str(attempt.sub_u_offset),
            "SUB_S_OFFSET": str(attempt.sub_s_offset),
            "NODE_ID": attempt.node,
            "U_DEPTH": str(attempt.u_depth), "U_INDEX": str(attempt.u_index),
            "S_DEPTH": str(attempt.s_depth), "S_INDEX": str(attempt.s_index),
            "INPUT_SHA256": attempt.input_sha256,
        })
    if len(rows) != EXPECTED_CELLS:
        die("expected exactly 100 coordinate rows")
    return rows


def execute_attempt(root: Path, binary: Path, attempt: RefineAttempt, timeout: float,
                    out_dir: Path, contract_sha: str, ledger_sha: str) -> dict[str, str]:
    command = [
        str(binary), str(attempt.u_depth), str(attempt.u_index), str(attempt.s_depth),
        str(attempt.s_index), attempt.input_sha256, attempt.run_challenge,
        attempt.carrier, contract_sha, ledger_sha, contract_sha,
        digest_bytes(attempt.node.encode("ascii") + b"\n"), attempt.attempt_binding,
    ]
    started = time.monotonic()
    try:
        completed = subprocess.run(command, cwd=root, capture_output=True, timeout=timeout)
        worker_rc, stdout, stderr = completed.returncode, completed.stdout, completed.stderr
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
    field = lambda name: fields.get(name, "UNEMITTED")
    return {
        "ATTEMPT_INDEX": str(attempt.index),
        "SOURCE_CELL_INDEX": str(attempt.source.cell_index),
        "SOURCE_U_OFFSET": str(attempt.source.u_offset),
        "SOURCE_S_OFFSET": str(attempt.source.s_offset),
        "SUB_U_OFFSET": str(attempt.sub_u_offset), "SUB_S_OFFSET": str(attempt.sub_s_offset),
        "ORIGINAL_PARENT_DEPTH_DELTA": "5", "NODE_ID": attempt.node, "CARRIER": attempt.carrier,
        "STATUS": status, "WORKER_RC": str(worker_rc), "ELAPSED_MS": str(elapsed_ms),
        "INPUT_SHA256": attempt.input_sha256, "RUN_CHALLENGE": attempt.run_challenge,
        "ATTEMPT_BINDING": attempt.attempt_binding, "STDOUT_SHA256": digest_bytes(stdout),
        "STDERR_SHA256": digest_bytes(stderr), "SUMMARY_SHA256": summary_sha,
        "C1_ORIENTATION_UNRESOLVED": field("C1_ORIENTATION_UNRESOLVED"),
        "C2_HULL_ORIENTATION_UNRESOLVED": field("C2_HULL_ORIENTATION_UNRESOLVED"),
        "EVENT1_CHARTS_CERTIFIED": field("EVENT1_CHARTS_CERTIFIED"),
        "EVENT2_CHARTS_CERTIFIED": field("EVENT2_CHARTS_CERTIFIED"),
        "HOMOGENEOUS_COMPUTATION_VALID": field("HOMOGENEOUS_COMPUTATION_VALID"),
        "CERTIFICATE_PASS": field("CERTIFICATE_PASS"), "PROBE_PASS": field("PROBE_PASS"),
    }


def write_summary(path: Path, rows: Sequence[dict[str, str]]) -> None:
    pairs = [rows[i:i + len(CARRIERS)] for i in range(0, len(rows), len(CARRIERS))]
    by_source: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_source[row["SOURCE_CELL_INDEX"]].append(row)
    probe_passes = sum(row["PROBE_PASS"] == "true" for row in rows)
    rejected = sum(row["STATUS"] == "DESCENDANT_PROBE_REJECTED" for row in rows)
    crossing = sum(row["STATUS"] == "SECTION_RESIDENT_CROSSING_UNAVAILABLE" for row in rows)
    unknown = sum(row["STATUS"] == "UNKNOWN_FAILURE" for row in rows)
    certificates = sum(row["CERTIFICATE_PASS"] == "true" for row in rows)
    both_pass = sum(all(row["PROBE_PASS"] == "true" for row in pair) for pair in pairs)
    both_reject = sum(all(row["PROBE_PASS"] == "false" for row in pair) for pair in pairs)
    mixed = sum(len({row["PROBE_PASS"] for row in pair}) != 1 for pair in pairs)
    agreement = sum(len({row["STATUS"] for row in pair}) == 1 for pair in pairs)
    refined_full = sum(all(row["PROBE_PASS"] == "true" for row in group) for group in by_source.values())
    refined_with_rejection = sum(any(row["PROBE_PASS"] != "true" for row in group) for group in by_source.values())
    run_complete = len(rows) == EXPECTED_ATTEMPTS
    run_valid = run_complete and unknown == 0 and len(by_source) == SOURCE_REJECTED_PARENTS
    refined_cover = run_valid and probe_passes == EXPECTED_ATTEMPTS
    fields = [
        ("SCHEMA", "sounio.cs6.v7b-target23-depth5-boundary-refine-summary.v1"),
        ("RUN_COMPLETE", str(run_complete).lower()), ("RUN_VALID", str(run_valid).lower()),
        ("SOURCE_DEPTH4_PASS_CELLS", str(SOURCE_PASS_CELLS)),
        ("SOURCE_REJECTED_PARENT_CELLS", str(len(by_source))),
        ("GRANDCHILD_CELLS_EVALUATED", str(len(pairs))),
        ("ATTEMPTS_COMPLETED", str(len(rows))), ("PROBE_PASS_ATTEMPTS", str(probe_passes)),
        ("PROBE_REJECTED_ATTEMPTS", str(rejected)),
        ("SECTION_RESIDENT_CROSSING_UNAVAILABLE", str(crossing)),
        ("UNKNOWN_FAILURE", str(unknown)), ("CERTIFICATE_PASS_ATTEMPTS", str(certificates)),
        ("BOTH_CARRIERS_PROBE_PASS_CELLS", str(both_pass)),
        ("BOTH_CARRIERS_PROBE_REJECT_CELLS", str(both_reject)),
        ("MIXED_CARRIER_CELLS", str(mixed)), ("CARRIER_STATUS_AGREEMENT_CELLS", str(agreement)),
        ("REFINED_PARENTS_FULL_PROBE_COVER", str(refined_full)),
        ("REFINED_PARENTS_WITH_REJECTION", str(refined_with_rejection)),
        ("REFINED_REGION_EVALUATED", str(run_valid).lower()),
        ("REFINED_REGION_PROBE_COVER_PASS", str(refined_cover).lower()),
        ("ADAPTIVE_PARENT_PROBE_COVER_EVALUATED", str(run_valid).lower()),
        ("ADAPTIVE_PARENT_PROBE_COVER_PASS", str(refined_cover).lower()),
        ("ADAPTIVE_COVER_LEAF_CELLS", str(SOURCE_PASS_CELLS + EXPECTED_CELLS)),
        ("ADAPTIVE_PARENT_CERTIFICATE_COVER_PASS", "false"),
        ("C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", "false"),
        ("V7_B_ELIGIBILITY", "false"), ("V7_B_WINNER", "NONE"),
        ("PROMOTION_ELIGIBLE", "false"), ("OPEN_PROBLEM_SOLVED", "false"),
        ("FPGA_EXECUTION", "false"),
    ]
    path.write_text("".join(f"{key}={value}\n" for key, value in fields), encoding="ascii")


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
    root, out_dir = Path.cwd(), args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = build_attempts(root)
    write_tsv(out_dir / "coordinate-manifest.tsv", COORDINATE_COLUMNS, coordinate_rows(attempts))
    if args.plan_only:
        fields = [
            ("SCHEMA", "sounio.cs6.v7b-target23-depth5-boundary-refine-plan.v1"),
            ("PLAN_VALID", str(len(attempts) == EXPECTED_ATTEMPTS).lower()),
            ("SOURCE_REJECTED_PARENT_CELLS", str(SOURCE_REJECTED_PARENTS)),
            ("GRANDCHILD_CELLS", str(EXPECTED_CELLS)), ("ATTEMPTS", str(len(attempts))),
            ("ADAPTIVE_PARENT_PROBE_COVER_EVALUATED", "false"),
            ("OPEN_PROBLEM_SOLVED", "false"),
        ]
        (out_dir / "plan-summary.txt").write_text(
            "".join(f"{key}={value}\n" for key, value in fields), encoding="ascii"
        )
        print((out_dir / "plan-summary.txt").read_text(encoding="ascii"), end="")
        return
    if args.binary is None or not args.binary.is_file():
        die("--binary must name the prebuilt CAPD worker")
    binary = args.binary.resolve()
    (out_dir / "attempts").mkdir(exist_ok=True)
    contract_sha, ledger_sha = digest(root / CONTRACT_REL), digest(root / LEDGER_REL)
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        rows = list(executor.map(
            lambda attempt: execute_attempt(root, binary, attempt, args.timeout, out_dir,
                                            contract_sha, ledger_sha), attempts
        ))
    rows.sort(key=lambda row: int(row["ATTEMPT_INDEX"]))
    write_tsv(out_dir / "results.tsv", RESULT_COLUMNS, rows)
    write_summary(out_dir / "summary.txt", rows)
    files = (
        root / CONTRACT_REL, root / SOURCE_RESULTS_REL, root / SOURCE_COORDINATES_REL,
        root / SOURCE_BOUNDARY_MAP_REL, root / LEDGER_REL, root / WORKER_REL,
        out_dir / "coordinate-manifest.tsv", out_dir / "results.tsv",
        out_dir / "summary.txt", binary,
    )
    (out_dir / "files.sha256").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in files), encoding="ascii"
    )
    print((out_dir / "summary.txt").read_text(encoding="ascii"), end="")


if __name__ == "__main__":
    main()
