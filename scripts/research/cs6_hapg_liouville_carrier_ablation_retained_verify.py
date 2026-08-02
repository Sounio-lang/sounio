#!/usr/bin/env python3
"""Independently replay and reduce a retained V7-A result directory."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Mapping, Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
FROZEN_CONTRACT_SHA256 = "decf9089e1dc9aae513f48c48a00e1c815a585b6ba7e9cd1c09b0b514fd58481"
COORDINATE_MANIFEST_SHA256 = "df665eceee8a45ea687a9f0bb643fe9fef28c800650482092be113f24fa41fdd"
CONTRACT_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_contract_v1.txt")
COORDINATES_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_coordinates_v1.tsv")
SOURCE_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_probe.cpp")
VERIFIER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_verify.py")
RUNNER_REL = Path("scripts/research/cs6_hapg_liouville_carrier_ablation_run.py")
PARENT_VERIFIER_REL = Path("scripts/research/cs6_plucker_cocycle_verify.py")
KAT_REL = Path("scripts/research/receipts/cs6_hapg_full_source_cover_v6_jobs_8469_8470_v1/kat/coordinates.tsv")
CELL_DOMAIN = b"sounio.cs6.hapg-liouville-carrier-ablation-cell.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.hapg-liouville-carrier-ablation-attempt.v1\0"
CARRIERS = ("C0HOTripletonSet", "C0HORect2Set", "C0Rect2Set")
BASELINE = CARRIERS[0]
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
RESULT_COLUMNS = (
    "ATTEMPT_INDEX", "ORDINAL", "SAMPLE_CLASS", "STRATUM", "NODE_ID",
    "LIOUVILLE_CARRIER", "STATUS", "WORKER_RC", "ELAPSED_MS",
    "INPUT_SHA256", "MANIFEST_ROW_SHA256", "CELL_CHALLENGE",
    "ATTEMPT_BINDING", "STDOUT_SHA256", "STDERR_SHA256",
    "VERIFICATION_SHA256", "CORE_RECEIPT_SHA256", "PHYSICAL_SHA256",
    "REFERENCE_PHYSICAL_SHA256", "INITIAL_HULL_SHA256", "MUTATION_TESTS",
    "MUTATIONS_REJECTED", "ALL_FINITE", "DETERMINANT_JOINT_COMPATIBLE",
    "C1_DET", "C2_HULL_DET", "LIOUVILLE_DET", "CHART_TUPLE",
    "PROBE_PASS", "CERTIFICATE_PASS", "SUBDIVISION_REQUIRED",
)


class AuditError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AuditError(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} is not a regular file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns
    ) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != before.st_size:
        fail(f"{label} changed while being read")
    return raw


def digest(path: Path) -> str:
    return digest_bytes(stable_bytes(path, str(path)))


def parse_kv_bytes(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AuditError(f"{label} is not ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail(f"{label} is not canonical LF text")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail(f"malformed KV row in {label}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            fail(f"duplicate or empty KV field in {label}")
        fields[key] = value
    return fields


def parse_kv(path: Path, label: str) -> dict[str, str]:
    return parse_kv_bytes(stable_bytes(path, label), label)


def parse_bool(token: str, label: str) -> bool:
    if token not in {"true", "false"}:
        fail(f"noncanonical boolean: {label}")
    return token == "true"


def canonical_uint(token: str, label: str) -> int:
    if not token.isdigit() or str(int(token)) != token:
        fail(f"noncanonical integer: {label}")
    return int(token)


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def table(path: Path, label: str, columns: Sequence[str]) -> list[dict[str, str]]:
    raw = stable_bytes(path, label)
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AuditError(f"{label} is not ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail(f"{label} is not canonical")
    reader = csv.DictReader(text.splitlines(), delimiter="\t")
    if tuple(reader.fieldnames or ()) != tuple(columns):
        fail(f"{label} columns differ from schema")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        fail(f"{label} has a malformed row")
    return rows


def coordinate_rows(path: Path) -> list[dict[str, str]]:
    raw = stable_bytes(path, "coordinates")
    if digest_bytes(raw) != COORDINATE_MANIFEST_SHA256:
        fail("coordinate manifest digest drift")
    text = raw.decode("ascii")
    header = (
        "ORDINAL\tSAMPLE_CLASS\tSTRATUM\tNODE_ID\tU_DEPTH\tU_INDEX\t"
        "S_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256"
    )
    lines = text.splitlines()
    if header not in lines:
        fail("coordinate header missing")
    index = lines.index(header)
    reader = csv.DictReader(lines[index:], delimiter="\t")
    rows = list(reader)
    if len(rows) != 40:
        fail("coordinate cardinality mismatch")
    for ordinal, row in enumerate(rows, 1):
        if canonical_uint(row["ORDINAL"], "coordinate ordinal") != ordinal:
            fail("coordinate order drift")
        numbers = [
            canonical_uint(row[key], f"coordinate {key}")
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        ]
        raw_input = leaf_input_bytes(*numbers)
        if digest_bytes(raw_input) != row["PARENT_INPUT_SHA256"]:
            fail("coordinate input digest mismatch")
        row["ROW_SHA256"] = digest_bytes(
            ("\t".join(row[key] for key in reader.fieldnames or ()) + "\n").encode("ascii")
        )
    return rows


def expected_attempts(
    coordinates: Sequence[Mapping[str, str]], run_contract_sha: str,
    coordinate_sha: str, root_challenge: str,
) -> list[dict[str, str]]:
    attempts: list[dict[str, str]] = []
    for coordinate in coordinates:
        challenge = digest_bytes(
            CELL_DOMAIN
            + bytes.fromhex(root_challenge) + b"\0"
            + bytes.fromhex(run_contract_sha) + b"\0"
            + bytes.fromhex(coordinate_sha) + b"\0"
            + bytes.fromhex(coordinate["ROW_SHA256"]) + b"\0"
            + bytes.fromhex(coordinate["PARENT_INPUT_SHA256"])
        )
        for carrier in CARRIERS:
            binding = digest_bytes(
                ATTEMPT_DOMAIN
                + bytes.fromhex(challenge) + b"\0"
                + carrier.encode("ascii") + b"\0"
                + bytes.fromhex(run_contract_sha)
            )
            attempts.append({
                "ATTEMPT_INDEX": str(len(attempts) + 1),
                "ORDINAL": coordinate["ORDINAL"],
                "SAMPLE_CLASS": coordinate["SAMPLE_CLASS"],
                "STRATUM": coordinate["STRATUM"],
                "NODE_ID": coordinate["NODE_ID"],
                "U_DEPTH": coordinate["U_DEPTH"],
                "U_INDEX": coordinate["U_INDEX"],
                "S_DEPTH": coordinate["S_DEPTH"],
                "S_INDEX": coordinate["S_INDEX"],
                "INPUT_SHA256": coordinate["PARENT_INPUT_SHA256"],
                "MANIFEST_ROW_SHA256": coordinate["ROW_SHA256"],
                "CELL_CHALLENGE": challenge,
                "LIOUVILLE_CARRIER": carrier,
                "ATTEMPT_BINDING": binding,
            })
    return attempts


def index_bytes(root: Path, excluded: set[str]) -> bytes:
    rows: list[str] = []
    for path in sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            fail("result tree contains a symlink")
        rows.append(f"{digest(path)}  {relative}\n")
    return "".join(rows).encode("ascii")


def historical_charts(path: Path, selected: set[str]) -> dict[str, str]:
    lines = stable_bytes(path, "historical KAT coordinates").decode("ascii").splitlines()
    index = next(
        (position for position, line in enumerate(lines) if line.startswith("LEAF_ID\tU_DEPTH\t")),
        None,
    )
    if index is None:
        fail("historical KAT header missing")
    charts: dict[str, str] = {}
    for row in csv.DictReader(lines[index:], delimiter="\t"):
        if row["LEAF_ID"] not in selected:
            continue
        parts = []
        for event in ("E1_R0", "E1_R1", "E2_R0", "E2_R1"):
            sign = "+1" if row[f"{event}_SIGN"] == "1" else row[f"{event}_SIGN"]
            parts.append(f"{row[f'{event}_CHART']}:{sign}")
        charts[row["LEAF_ID"]] = ",".join(parts)
    if set(charts) != selected:
        fail("historical KAT controls are incomplete")
    return charts


def interval_fraction(token: str) -> tuple[Fraction, Fraction]:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        fail("malformed determinant interval")
    try:
        lower = Fraction.from_float(float.fromhex(match.group(1)))
        upper = Fraction.from_float(float.fromhex(match.group(2)))
    except (ValueError, OverflowError) as error:
        raise AuditError("nonfinite determinant interval") from error
    if lower > upper:
        fail("reversed determinant interval")
    return lower, upper


def width_score(rows: Sequence[Mapping[str, str]]) -> tuple[Fraction, Fraction]:
    scores = []
    for row in rows:
        lower, upper = interval_fraction(row["LIOUVILLE_DET"])
        if upper >= 0:
            fail("winner score received nonnegative determinant")
        scores.append((upper - lower) / (-upper))
    return max(scores), sum(scores, Fraction(0))


def fraction_token(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def classify_negative(stderr: bytes, carrier: str, binding: str) -> bool:
    preamble = (
        f"V7_FAILURE_BINDING LIOUVILLE_CARRIER={carrier} "
        f"ATTEMPT_BINDING={binding}\n"
    ).encode("ascii")
    if not stderr.startswith(preamble):
        return False
    core = stderr[len(preamble):].lower()
    return (
        core.startswith(
            b"probe error: centeredtripletonset::evalaffinefunctional - empty intersection of rb and rq."
        )
        and b"\nrb=[" in core
        and core.endswith(b"\nrq=[-nan, -nan]\n\n")
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args(argv)
    root = args.result_dir.resolve()
    if not root.is_dir() or root.is_symlink():
        fail("result directory is invalid")
    repo = args.repo.resolve()
    worker = args.worker.resolve()
    if not repo.is_dir() or repo.is_symlink() or not worker.is_file() or worker.is_symlink():
        fail("external repository or worker anchor is invalid")
    git_head = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    git_status = subprocess.run(
        ["git", "-C", repo, "status", "--short", "--untracked-files=all"],
        check=True, capture_output=True,
    ).stdout
    if not re.fullmatch(r"[0-9a-f]{40}", git_head) or git_status:
        fail("external repository anchor is not a clean exact commit")
    descendants = list(root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        fail("result tree contains a symlink")

    if stable_bytes(root / "result-files.sha256", "result index") != index_bytes(
        root, {"result-files.sha256", "run-manifest.txt", "files.sha256"}
    ):
        fail("result-files index mismatch")
    if stable_bytes(root / "files.sha256", "final index") != index_bytes(
        root, {"files.sha256"}
    ):
        fail("files index mismatch")

    if digest(root / "frozen-contract.txt") != FROZEN_CONTRACT_SHA256:
        fail("frozen contract snapshot digest drift")
    if stable_bytes(root / "frozen-contract.txt", "frozen contract") != stable_bytes(
        repo / CONTRACT_REL, "repository frozen contract"
    ):
        fail("frozen contract snapshot differs from external checkout")
    frozen = parse_kv(root / "frozen-contract.txt", "frozen contract")
    snapshot_pairs = (
        ("coordinates.tsv", COORDINATES_REL),
        ("worker-source.cpp", SOURCE_REL),
        ("attempt-verifier.py", VERIFIER_REL),
        ("cs6_plucker_cocycle_verify.py", PARENT_VERIFIER_REL),
        ("runner.py", RUNNER_REL),
        ("historical-kat-coordinates.tsv", KAT_REL),
    )
    for result_name, repository_name in snapshot_pairs:
        if stable_bytes(root / result_name, result_name) != stable_bytes(
            repo / repository_name, str(repository_name)
        ):
            fail(f"result snapshot differs from external checkout: {result_name}")
    if stable_bytes(root / "git-head.txt", "git head") != (git_head + "\n").encode("ascii"):
        fail("result Git head differs from external checkout")
    if stable_bytes(root / "git-status.txt", "git status"):
        fail("result repository status was dirty")
    coordinates = coordinate_rows(root / "coordinates.tsv")
    run_contract = parse_kv(root / "run-contract.txt", "run contract")
    run_contract_sha = digest(root / "run-contract.txt")
    root_challenge = run_contract.get("ROOT_CHALLENGE", "")
    source_sha = digest(repo / SOURCE_REL)
    worker_sha = digest(worker)
    provenance_index = index_bytes(root / "provenance", set())
    if stable_bytes(root / "provenance.sha256", "provenance index") != provenance_index:
        fail("provenance index mismatch")
    if stable_bytes(root / "provenance" / "worker-source.cpp", "provenance source") != stable_bytes(
        repo / SOURCE_REL, "repository source"
    ):
        fail("prebuilt source snapshot differs from external checkout")
    if stable_bytes(root / "provenance" / "worker-source.sha256", "source sidecar") != (
        source_sha + "\n"
    ).encode("ascii"):
        fail("prebuilt source sidecar mismatch")
    if stable_bytes(root / "provenance" / "worker-binary.sha256", "binary sidecar") != (
        worker_sha + "\n"
    ).encode("ascii"):
        fail("prebuilt binary sidecar mismatch")
    if stable_bytes(root / "provenance" / "capd-version.txt", "CAPD version") != b"5.3.0\n":
        fail("retained CAPD version mismatch")
    cflags = stable_bytes(root / "provenance" / "capd-cflags.txt", "CAPD flags").decode("ascii")
    if "-D__USE_FILIB__" not in cflags.split() or "-frounding-math" not in cflags.split():
        fail("retained CAPD flags do not bind FILIB rounding")
    slurm_context = parse_kv(root / "provenance" / "slurm-context.txt", "Slurm context")
    jobs = canonical_uint(run_contract.get("JOBS", ""), "run jobs")
    timeout = canonical_uint(run_contract.get("TIMEOUT_SECONDS", ""), "run timeout")
    allocated = canonical_uint(slurm_context.get("SLURM_CPUS_PER_TASK", ""), "Slurm CPUs per task")
    if not (1 <= jobs <= min(120, allocated)) or not (1 <= timeout <= 3600):
        fail("run concurrency or timeout exceeds frozen envelope")
    expected_run_contract = {
        "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-run-contract.v1",
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "ROOT_CHALLENGE": frozen["ROOT_CHALLENGE"],
        "GIT_HEAD": git_head,
        "WORKER_SOURCE_SHA256": source_sha,
        "WORKER_BINARY_SHA256": worker_sha,
        "ATTEMPT_VERIFIER_SHA256": digest(repo / VERIFIER_REL),
        "PARENT_VERIFIER_SHA256": digest(repo / PARENT_VERIFIER_REL),
        "RUNNER_SHA256": digest(repo / RUNNER_REL),
        "PROVENANCE_INDEX_SHA256": digest_bytes(provenance_index),
        "HISTORICAL_KAT_COORDINATES_SHA256": digest(repo / KAT_REL),
        "CELL_COUNT": "40", "CARRIER_COUNT": "3", "ATTEMPT_COUNT": "120",
        "ATTEMPT_FREEZE_ORDER": "EXACT_120_ATTEMPTS_BEFORE_ANY_WORKER",
        "JOBS": str(jobs), "TIMEOUT_SECONDS": str(timeout),
        "SELF_TEST_MUTATIONS": "true", "CAPD_VERSION": "5.3.0",
        "INTERVAL_BACKEND": "FILIB", "OPTIMIZATION_LEVEL": "O0",
        "FPGA_EXECUTION": "false", "EXECUTION_PROVENANCE_ATTESTED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    if run_contract != expected_run_contract:
        fail("run contract differs from exact external/frozen anchors")
    if digest(repo / KAT_REL) != frozen["V6_KAT_COORDINATES_SHA256"]:
        fail("historical KAT digest differs from frozen contract")
    expected = expected_attempts(
        coordinates, run_contract_sha, COORDINATE_MANIFEST_SHA256, root_challenge
    )
    attempt_columns = (
        "ATTEMPT_INDEX", "ORDINAL", "SAMPLE_CLASS", "STRATUM", "NODE_ID",
        "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "INPUT_SHA256",
        "MANIFEST_ROW_SHA256", "CELL_CHALLENGE", "LIOUVILLE_CARRIER",
        "ATTEMPT_BINDING",
    )
    attempts = table(root / "attempt-contract.tsv", "attempt contract", attempt_columns)
    if attempts != expected:
        fail("attempt contract does not reconstruct exact 120 matrix")
    results = table(root / "results.tsv", "results", RESULT_COLUMNS)
    if len(results) != 120:
        fail("result cardinality mismatch")

    verifier = root / "attempt-verifier.py"
    replay_count = 0
    negative_count = 0
    for attempt, result in zip(attempts, results, strict=True):
        for key in (
            "ATTEMPT_INDEX", "ORDINAL", "SAMPLE_CLASS", "STRATUM", "NODE_ID",
            "LIOUVILLE_CARRIER", "INPUT_SHA256", "MANIFEST_ROW_SHA256",
            "CELL_CHALLENGE", "ATTEMPT_BINDING",
        ):
            if result[key] != attempt[key]:
                fail(f"result/attempt join mismatch: {attempt['ATTEMPT_INDEX']} {key}")
        identity = f"A{int(attempt['ATTEMPT_INDEX']):04d}"
        receipt = root / "receipts" / f"{identity}.txt"
        stderr_path = root / "stderr" / f"{identity}.txt"
        receipt_raw = stable_bytes(receipt, f"receipt {identity}")
        stderr_raw = stable_bytes(stderr_path, f"stderr {identity}")
        if digest_bytes(receipt_raw) != result["STDOUT_SHA256"]:
            fail(f"stdout digest mismatch: {identity}")
        if digest_bytes(stderr_raw) != result["STDERR_SHA256"]:
            fail(f"stderr digest mismatch: {identity}")
        input_path = root / "inputs" / f"{attempt['NODE_ID']}.txt"
        if stable_bytes(input_path, f"input {identity}") != leaf_input_bytes(
            int(attempt["U_DEPTH"]), int(attempt["U_INDEX"]),
            int(attempt["S_DEPTH"]), int(attempt["S_INDEX"]),
        ):
            fail(f"input reconstruction mismatch: {identity}")

        if result["STATUS"] == "VERIFIED_COMPLETE":
            if stderr_raw or result["WORKER_RC"] != "0":
                fail(f"verified worker channel mismatch: {identity}")
            saved_path = root / "verifications" / f"{identity}.txt"
            saved_raw = stable_bytes(saved_path, f"verification {identity}")
            if digest_bytes(saved_raw) != result["VERIFICATION_SHA256"]:
                fail(f"verification digest mismatch: {identity}")
            saved = parse_kv_bytes(saved_raw, f"verification {identity}")
            if saved.get("MUTATION_TESTS") != "93" or saved.get("MUTATIONS_REJECTED") != "93":
                fail(f"mutation audit mismatch: {identity}")
            command = [
                sys.executable, "-B", str(verifier), str(receipt),
                "--source-sha", source_sha, "--input", str(input_path),
                "--challenge", attempt["CELL_CHALLENGE"],
                "--carrier", attempt["LIOUVILLE_CARRIER"],
                "--frozen-contract-sha", FROZEN_CONTRACT_SHA256,
                "--coordinate-manifest-sha", COORDINATE_MANIFEST_SHA256,
                "--run-contract-sha", run_contract_sha,
                "--manifest-row-sha", attempt["MANIFEST_ROW_SHA256"],
                "--attempt-binding", attempt["ATTEMPT_BINDING"],
                "--self-test-mutations",
            ]
            replay = subprocess.run(command, capture_output=True)
            if replay.returncode != 0 or replay.stderr:
                fail(f"fresh verifier replay failed: {identity}")
            live = parse_kv_bytes(replay.stdout, f"fresh verification {identity}")
            for key, value in saved.items():
                if live.get(key) != value:
                    fail(f"fresh verifier field mismatch: {identity} {key}")
            if (
                result["MUTATION_TESTS"] != saved["MUTATION_TESTS"]
                or result["MUTATIONS_REJECTED"] != saved["MUTATIONS_REJECTED"]
            ):
                fail(f"result/verification mutation mismatch: {identity}")
            result_map = {
                "CORE_RECEIPT_SHA256": "CORE_RECEIPT_SHA256",
                "PHYSICAL_SHA256": "PHYSICAL_SHA256",
                "REFERENCE_PHYSICAL_SHA256": "REFERENCE_PHYSICAL_SHA256",
                "INITIAL_HULL_SHA256": "INITIAL_HULL_SHA256",
                "ALL_FINITE": "ALL_FINITE",
                "DETERMINANT_JOINT_COMPATIBLE": "DETERMINANT_JOINT_COMPATIBLE",
                "C1_DET": "C1_DET", "C2_HULL_DET": "C2_HULL_DET",
                "LIOUVILLE_DET": "LIOUVILLE_DET", "CHART_TUPLE": "CHART_TUPLE",
                "PROBE_PASS": "PROBE_PASS", "CERTIFICATE_PASS": "CERTIFICATE_PASS",
                "SUBDIVISION_REQUIRED": "SUBDIVISION_REQUIRED",
            }
            for result_key, verification_key in result_map.items():
                if result[result_key] != saved[verification_key]:
                    fail(f"result/verification mismatch: {identity} {result_key}")
            replay_count += 1
        elif result["STATUS"] == "CAPD_SET_RQ_NAN":
            if receipt_raw or result["WORKER_RC"] != "1" or not classify_negative(
                stderr_raw, attempt["LIOUVILLE_CARRIER"], attempt["ATTEMPT_BINDING"]
            ):
                fail(f"scientific negative is not exact and bound: {identity}")
            negative = parse_kv(
                root / "negative-receipts" / f"{identity}.txt",
                f"negative receipt {identity}",
            )
            expected_negative = {
                "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-negative.v1",
                "ATTEMPT_INDEX": attempt["ATTEMPT_INDEX"],
                "NODE_ID": attempt["NODE_ID"],
                "SAMPLE_CLASS": attempt["SAMPLE_CLASS"],
                "LIOUVILLE_CARRIER": attempt["LIOUVILLE_CARRIER"],
                "INPUT_SHA256": attempt["INPUT_SHA256"],
                "CELL_CHALLENGE": attempt["CELL_CHALLENGE"],
                "MANIFEST_ROW_SHA256": attempt["MANIFEST_ROW_SHA256"],
                "ATTEMPT_BINDING": attempt["ATTEMPT_BINDING"],
                "WORKER_RC": "1", "STDOUT_SHA256": EMPTY_SHA256,
                "STDERR_SHA256": result["STDERR_SHA256"],
                "CLASS": "CAPD_SET_RQ_NAN",
                "FAILURE_BINDING_AUTHENTICATED": "true",
                "SCIENTIFIC_NEGATIVE": "true",
            }
            if negative != expected_negative:
                fail(f"negative receipt differs from exact bound result: {identity}")
            negative_count += 1
        elif result["STATUS"] not in {"UNKNOWN_FAILURE", "TIMEOUT", "VERIFIER_FAILURE"}:
            fail(f"unknown result status: {identity}")

    matrix = {
        (row["NODE_ID"], row["LIOUVILLE_CARRIER"]): row for row in results
    }
    targets = [row for row in coordinates if row["SAMPLE_CLASS"] == "CAPD_SET_TARGET"]
    no_charts = [row for row in coordinates if row["SAMPLE_CLASS"] == "NO_SIGNED_CHART_CONTROL"]
    positives = [row for row in coordinates if row["SAMPLE_CLASS"] == "HPG_POSITIVE_CONTROL"]
    historical = historical_charts(
        root / "historical-kat-coordinates.tsv", {row["NODE_ID"] for row in positives}
    )
    unknown = [
        row for row in results
        if row["STATUS"] in {"UNKNOWN_FAILURE", "TIMEOUT", "VERIFIER_FAILURE"}
    ]
    baseline_valid = (
        all(matrix[(row["NODE_ID"], BASELINE)]["STATUS"] == "CAPD_SET_RQ_NAN" for row in targets)
        and all(
            matrix[(row["NODE_ID"], BASELINE)]["STATUS"] == "VERIFIED_COMPLETE"
            and not parse_bool(matrix[(row["NODE_ID"], BASELINE)]["PROBE_PASS"], "baseline no-chart")
            for row in no_charts
        )
        and all(
            matrix[(row["NODE_ID"], BASELINE)]["STATUS"] == "VERIFIED_COMPLETE"
            and parse_bool(matrix[(row["NODE_ID"], BASELINE)]["PROBE_PASS"], "baseline positive")
            and matrix[(row["NODE_ID"], BASELINE)]["CHART_TUPLE"] == historical[row["NODE_ID"]]
            for row in positives
        )
    )
    group_invariance: dict[str, bool] = {}
    group_mismatch: dict[str, bool] = {}
    for coordinate in coordinates:
        successes = [
            matrix[(coordinate["NODE_ID"], carrier)] for carrier in CARRIERS
            if matrix[(coordinate["NODE_ID"], carrier)]["STATUS"] == "VERIFIED_COMPLETE"
        ]
        invariant = (
            len(successes) >= 2
            and len({row["REFERENCE_PHYSICAL_SHA256"] for row in successes}) == 1
            and len({row["INITIAL_HULL_SHA256"] for row in successes}) == 1
        )
        group_invariance[coordinate["NODE_ID"]] = invariant
        group_mismatch[coordinate["NODE_ID"]] = len(successes) >= 2 and not invariant
    complete = [row for row in results if row["STATUS"] == "VERIFIED_COMPLETE"]
    protocol_invalid = (
        bool(unknown)
        or any(
            canonical_uint(row["MUTATION_TESTS"], "mutation tests") == 0
            or row["MUTATION_TESTS"] != row["MUTATIONS_REJECTED"]
            for row in complete
        )
        or any(
            not parse_bool(row["ALL_FINITE"], "all finite")
            or not parse_bool(row["DETERMINANT_JOINT_COMPATIBLE"], "determinant")
            for row in complete
        )
        or any(group_mismatch.values())
    )

    decisions: dict[str, str] = {BASELINE: "BASELINE_VALID" if baseline_valid else "RUN_INVALID"}
    repairs: dict[str, int] = {BASELINE: 0}
    scores: dict[str, tuple[Fraction, Fraction]] = {}
    for carrier in CARRIERS[1:]:
        carrier_rows = [matrix[(row["NODE_ID"], carrier)] for row in coordinates]
        target_rows = [matrix[(row["NODE_ID"], carrier)] for row in targets]
        repair_count = sum(row["STATUS"] == "VERIFIED_COMPLETE" for row in target_rows)
        repairs[carrier] = repair_count
        control_ok = (
            all(
                matrix[(row["NODE_ID"], carrier)]["STATUS"] == "VERIFIED_COMPLETE"
                and not parse_bool(matrix[(row["NODE_ID"], carrier)]["PROBE_PASS"], "no-chart")
                and matrix[(row["NODE_ID"], carrier)]["CHART_TUPLE"]
                == matrix[(row["NODE_ID"], BASELINE)]["CHART_TUPLE"]
                for row in no_charts
            )
            and all(
                matrix[(row["NODE_ID"], carrier)]["STATUS"] == "VERIFIED_COMPLETE"
                and parse_bool(matrix[(row["NODE_ID"], carrier)]["PROBE_PASS"], "positive")
                and matrix[(row["NODE_ID"], carrier)]["CHART_TUPLE"] == historical[row["NODE_ID"]]
                for row in positives
            )
        )
        if not baseline_valid or protocol_invalid:
            decision = "RUN_INVALID"
        elif not control_ok:
            decision = "NO_GO_CONTROL_REGRESSION"
        elif repair_count == 24 and all(row["STATUS"] == "VERIFIED_COMPLETE" for row in carrier_rows):
            decision = "GO" if all(group_invariance.values()) else "RUN_INVALID"
        elif repair_count == 0 and all(row["STATUS"] == "CAPD_SET_RQ_NAN" for row in target_rows):
            decision = "NO_GO_ALL_FAILURES"
        elif 1 <= repair_count <= 23 and all(
            row["STATUS"] in {"VERIFIED_COMPLETE", "CAPD_SET_RQ_NAN"} for row in target_rows
        ):
            decision = "INCONCLUSIVE_PARTIAL"
        else:
            decision = "RUN_INVALID"
        decisions[carrier] = decision
        if decision == "GO":
            scores[carrier] = width_score(carrier_rows)

    go_carriers = [carrier for carrier in CARRIERS[1:] if decisions[carrier] == "GO"]
    rank = {"C0HORect2Set": 0, "C0Rect2Set": 1}
    winner = "NONE" if not go_carriers else min(
        go_carriers, key=lambda carrier: (*scores[carrier], rank[carrier])
    )
    run_valid = baseline_valid and not protocol_invalid and all(
        decisions[carrier] != "RUN_INVALID" for carrier in CARRIERS[1:]
    )
    roles = {
        "C0HOTripletonSet": "BASELINE_HO_TRIPLETON",
        "C0HORect2Set": "CAUSAL_HO_DOUBLETON_ABLATION",
        "C0Rect2Set": "TAYLOR_DOUBLETON_ROBUSTNESS_CONTROL",
    }
    decision_lines = [
        "LIOUVILLE_CARRIER\tROLE\tDECISION\tCAPD_REPAIRS\tMAX_RELATIVE_WIDTH\tSUM_RELATIVE_WIDTH"
    ]
    for carrier in CARRIERS:
        score = scores.get(carrier)
        decision_lines.append("\t".join((
            carrier, roles[carrier], decisions[carrier], str(repairs[carrier]),
            "-" if score is None else fraction_token(score[0]),
            "-" if score is None else fraction_token(score[1]),
        )))
    if stable_bytes(root / "decisions.tsv", "decisions") != (
        "\n".join(decision_lines) + "\n"
    ).encode("ascii"):
        fail("decision table differs from independent reduction")

    summary = parse_kv(root / "summary.txt", "summary")
    summary_expected = {
        "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-summary.v1",
        "RUN_COMPLETE": "true", "RUN_VALID": str(run_valid).lower(),
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "RUN_CONTRACT_SHA256": run_contract_sha,
        "ATTEMPT_CONTRACT_SHA256": digest(root / "attempt-contract.tsv"),
        "CELL_COUNT": "40", "ATTEMPT_COUNT": "120",
        "VERIFIED_COMPLETE_COUNT": str(replay_count),
        "CAPD_SET_RQ_NAN_COUNT": str(negative_count),
        "UNKNOWN_FAILURE_COUNT": str(len(unknown)),
        "BASELINE_PREREQUISITE_VALID": str(baseline_valid).lower(),
        "PROTOCOL_INVALID": str(protocol_invalid).lower(),
        "C0HOTripletonSet_DECISION": decisions[BASELINE],
        "C0HORect2Set_DECISION": decisions["C0HORect2Set"],
        "C0Rect2Set_DECISION": decisions["C0Rect2Set"],
        "C0HORect2Set_CAPD_REPAIRS": str(repairs["C0HORect2Set"]),
        "C0Rect2Set_CAPD_REPAIRS": str(repairs["C0Rect2Set"]),
        "GO_CARRIER_COUNT": str(len(go_carriers)), "V7_B_WINNER": winner,
        "MUTATION_TESTS": str(sum(int(row["MUTATION_TESTS"]) for row in results)),
        "MUTATIONS_REJECTED": str(sum(int(row["MUTATIONS_REJECTED"]) for row in results)),
        "REFERENCE_INVARIANCE_CELL_COUNT": str(sum(group_invariance.values())),
        "REFERENCE_INVARIANCE_UNWITNESSED_COUNT": str(sum(not value for value in group_invariance.values())),
        "TOTAL_WORKER_ELAPSED_MS": str(sum(canonical_uint(row["ELAPSED_MS"], "elapsed") for row in results)),
        "FPGA_EXECUTION": "false", "EXECUTION_PROVENANCE_ATTESTED": "false",
        "FULL_SOURCE_CARRIER_PROVED": "false", "HYPERBOLICITY_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false", "OPEN_PROBLEM_SOLVED": "false",
        "NOVELTY_OR_PRIORITY_CLAIMED": "false", "PROMOTION_ELIGIBLE": "false",
    }
    if summary != summary_expected:
        fail("summary differs from exact independent reduction or claim boundary")

    manifest = parse_kv(root / "run-manifest.txt", "run manifest")
    manifest_expected = {
        "SCHEMA": "sounio.cs6.hapg-liouville-carrier-ablation-run-manifest.v1",
        "RUN_COMPLETE": "true", "RUN_VALID": str(run_valid).lower(),
        "GIT_HEAD": git_head,
        "ROOT_CHALLENGE": root_challenge,
        "WORKER_SOURCE_SHA256": source_sha,
        "WORKER_BINARY_SHA256": worker_sha,
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "RUN_CONTRACT_SHA256": run_contract_sha,
        "ATTEMPT_CONTRACT_SHA256": digest(root / "attempt-contract.tsv"),
        "RESULTS_SHA256": digest(root / "results.tsv"),
        "DECISIONS_SHA256": digest(root / "decisions.tsv"),
        "SUMMARY_SHA256": digest(root / "summary.txt"),
        "RESULT_FILES_INDEX_SHA256": digest(root / "result-files.sha256"),
        "ATTEMPT_COUNT": "120", "GO_CARRIER_COUNT": str(len(go_carriers)),
        "V7_B_WINNER": winner, "PROMOTION_ELIGIBLE": "false",
    }
    if manifest != manifest_expected:
        fail("run manifest differs from exact independent reduction")

    print("AUDIT_SCHEMA=sounio.cs6.hapg-liouville-carrier-ablation-retained-audit.v1")
    print("AUDIT_PASS=true")
    print("ATTEMPTS_RECONSTRUCTED=120")
    print(f"VERIFIER_REPLAYS={replay_count}")
    print(f"BOUND_NEGATIVES={negative_count}")
    print(f"RUN_VALID={str(run_valid).lower()}")
    print(f"C0HORect2Set_DECISION={decisions['C0HORect2Set']}")
    print(f"C0Rect2Set_DECISION={decisions['C0Rect2Set']}")
    print(f"V7_B_WINNER={winner}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, subprocess.SubprocessError) as error:
        print(f"retained audit error: {error}", file=sys.stderr)
        raise SystemExit(1)
