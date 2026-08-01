#!/usr/bin/env python3
"""Verify the retained 53-coordinate CS6 homogeneous-cocycle evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import math
import os
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
CHALLENGE_DOMAIN = b"sounio.cs6.plucker-cocycle-leaf-challenge.v1\0"
ZERO_SHA256 = "0" * 64
RETAINED_ROOT_CHALLENGE = "455738b87f7d2d44f9b611183ecc95f84fc423822ed5dd0b58d759a1026d5ab6"
EXCLUDED_INDEX_FILES = {"files.sha256", "retained-manifest.txt"}
LEAF_COLUMNS = (
    "LEAF_ID", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "STATUS",
    "METHOD", "PROBE_PASS", "AFFINE_PASS", "PROJECTIVE_X_PASS",
    "PROJECTIVE_Y_PASS", "PROJECTIVE_PLUS_PASS", "PROJECTIVE_MINUS_PASS",
    "HOMOGENEOUS_PASS", "CERTIFICATE_PASS", "SUBDIVISION_REQUIRED",
    "INPUT_SHA256", "LEAF_CHALLENGE", "RECEIPT_SHA256", "STDERR_SHA256",
    "VERIFICATION_SHA256", "PHYSICAL_SHA256", "WORKER_RC", "ELAPSED_MS",
    "C1_DET_WIDTH", "AFFINE_DET_WIDTH", "FIXED_BEST_DET_WIDTH",
    "HOMOGENEOUS_DET_WIDTH", "MINIMUM_PIVOT_SCORE", "CHART_SWITCHES",
    "RECEIPT_BYTES",
)
METRIC_COLUMNS = (
    "C1_DET_WIDTH", "AFFINE_DET_WIDTH", "FIXED_BEST_DET_WIDTH",
    "HOMOGENEOUS_DET_WIDTH", "MINIMUM_PIVOT_SCORE", "CHART_SWITCHES",
    "RECEIPT_BYTES",
)
RUN_ARTIFACTS = (
    "capd-cflags.txt", "capd-libs.txt", "capd-version.txt",
    "compile-command.txt", "compile-stderr.txt", "compile-stdout.txt",
    "compiler-version.txt", "coordinates.tsv", "dependencies.sha256",
    "git-head.txt", "git-status.txt", "leaf-verifier.py", "leaves.tsv",
    "link-inputs.sha256", "mutation-audit-stderr.txt", "mutation-audit.txt",
    "python-version.txt", "run-contract.txt", "runner.py",
    "runtime-libraries.sha256", "runtime-linkage.txt", "summary.txt",
    "worker-source.cpp",
)


class RetainedError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise RetainedError(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(raw) != before.st_size:
        fail(f"{label} changed while being read")
    return raw


def parse_kv(path: Path, label: str) -> dict[str, str]:
    raw = stable_bytes(path, label)
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError(f"{label} must be ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail(f"{label} is not canonical LF-terminated text")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail(f"malformed field in {label}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            fail(f"duplicate or empty field in {label}: {key}")
        result[key] = value
    return result


def require_fields(actual: dict[str, str], expected: Sequence[tuple[str, str]], label: str) -> None:
    for key, value in expected:
        if actual.get(key) != value:
            fail(f"{label} field mismatch: {key}")


def parse_index(root: Path) -> dict[str, str]:
    raw = stable_bytes(root / "files.sha256", "files index")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError("files index must be ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail("files index is noncanonical")
    result: dict[str, str] = {}
    previous = ""
    for line in text.splitlines():
        if line.count("  ") != 1:
            fail("malformed files index row")
        expected, relative = line.split("  ", 1)
        if SHA_RE.fullmatch(expected) is None or not relative or relative <= previous:
            fail("files index digest or order mismatch")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in EXCLUDED_INDEX_FILES:
            fail("unsafe or recursive files index path")
        if relative in result:
            fail("duplicate files index path")
        result[relative] = expected
        previous = relative
    retained_paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in retained_paths):
        fail("retained tree contains a symlink")
    actual_directories = {
        path.relative_to(root).as_posix()
        for path in retained_paths
        if path.is_dir()
    }
    if actual_directories != {"inputs", "receipts", "stderr", "verifications"}:
        fail("retained directory schema mismatch")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in retained_paths
        if path.is_file() and path.relative_to(root).as_posix() not in EXCLUDED_INDEX_FILES
    }
    if set(result) != actual_files:
        fail("files index does not enumerate the exact retained file set")
    for relative, expected in result.items():
        path = root / relative
        if path.is_symlink() or not path.is_file():
            fail(f"retained path is not a regular file: {relative}")
        if digest_bytes(stable_bytes(path, f"indexed file {relative}")) != expected:
            fail(f"retained file digest mismatch: {relative}")
    if len(result) != 235:
        fail("files index cardinality mismatch")
    return result


def parse_coordinates(path: Path) -> dict[str, tuple[int, int, int, int, str]]:
    raw = stable_bytes(path, "coordinates")
    lines = raw.decode("ascii").splitlines()
    if not raw.endswith(b"\n") or lines[:4] != [
        "SCHEMA=sounio.cs6.plucker-cocycle-coordinates.v1",
        "PARENT_COORDINATE_SET=CS6_C1_FULL_SOURCE_COVER_RETAINED_53",
        "LEAF_COUNT=53",
        "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256",
    ]:
        fail("coordinate manifest header mismatch")
    result: dict[str, tuple[int, int, int, int, str]] = {}
    for line in lines[4:]:
        fields = line.split("\t")
        if len(fields) != 6:
            fail("coordinate manifest row width mismatch")
        identity, u_depth, u_index, s_depth, s_index, input_sha = fields
        numbers = tuple(
            canonical_uint(token, f"coordinate {identity}")
            for token in (u_depth, u_index, s_depth, s_index)
        )
        ud, ui, sd, si = numbers
        if ud > 30 or sd > 30 or ui >= 1 << ud or si >= 1 << sd:
            fail("coordinate manifest dyadic range mismatch")
        if identity != leaf_id(ud, ui, sd, si):
            fail("coordinate manifest identity mismatch")
        if SHA_RE.fullmatch(input_sha) is None or identity in result:
            fail("coordinate manifest digest or uniqueness mismatch")
        if digest_bytes(leaf_input_bytes(ud, ui, sd, si)) != input_sha:
            fail("coordinate manifest input digest mismatch")
        result[identity] = (ud, ui, sd, si, input_sha)
    if len(result) != 53 or list(result) != sorted(result):
        fail("coordinate manifest is not the canonical 53-row set")
    return result


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def challenge(root: str, identity: str, input_sha: str) -> str:
    return digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha)
    )


def parse_boolean(token: str, label: str) -> bool:
    if token not in {"true", "false"}:
        fail(f"noncanonical boolean: {label}")
    return token == "true"


def decimal(value: float) -> str:
    return format(value, ".17g")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def canonical_uint(token: str, label: str) -> int:
    if not token.isdigit() or str(int(token)) != token:
        fail(f"noncanonical unsigned integer: {label}")
    return int(token)


def record_values(raw: bytes, marker: str) -> dict[str, str]:
    prefix = marker.encode("ascii") + b" "
    matches = [line for line in raw.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        fail(f"receipt record cardinality mismatch: {marker}")
    try:
        tokens = matches[0].decode("ascii").split(" ")[1:]
    except UnicodeError as error:
        raise RetainedError(f"non-ASCII receipt record: {marker}") from error
    result: dict[str, str] = {}
    for token in tokens:
        if token.count("=") != 1:
            fail(f"malformed receipt token: {marker}")
        key, value = token.split("=", 1)
        if not key or not value or key in result:
            fail(f"duplicate or empty receipt token: {marker} {key}")
        result[key] = value
    return result


def interval_bounds(token: str, label: str) -> tuple[float, float]:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        fail(f"malformed interval: {label}")
    try:
        lower, upper = (float.fromhex(value) for value in match.groups())
    except ValueError as error:
        raise RetainedError(f"invalid hexadecimal interval: {label}") from error
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        fail(f"nonfinite or inverted interval: {label}")
    return lower, upper


def interval_width(record: dict[str, str], key: str, label: str) -> float:
    lower, upper = interval_bounds(record.get(key, ""), f"{label} {key}")
    width = upper - lower
    if not math.isfinite(width) or width <= 0.0:
        fail(f"nonfinite or nonpositive interval width: {label} {key}")
    return width


def receipt_metrics(raw: bytes) -> tuple[str, ...]:
    c1_width = interval_width(record_values(raw, "C1_P2_CONTROL"), "DET", "C1")
    affine_width = interval_width(
        record_values(raw, "AFFINE_CARRIER"), "DET", "affine"
    )
    fixed_widths: list[float] = []
    for marker in (
        "PROJECTIVE_X", "PROJECTIVE_Y", "PROJECTIVE_PLUS", "PROJECTIVE_MINUS"
    ):
        record = record_values(raw, marker)
        if record.get("ELIGIBLE") == "true":
            fixed_widths.append(interval_width(record, "DET", marker))
        elif record.get("ELIGIBLE") != "false":
            fail(f"noncanonical fixed-chart eligibility: {marker}")
    if not fixed_widths:
        fail("no eligible fixed final projective chart")
    homogeneous_width = interval_width(
        record_values(raw, "PLUCKER_COCYCLE"), "DET", "homogeneous"
    )
    ray_markers = (
        "HOMOGENEOUS_EVENT1_RAY0", "HOMOGENEOUS_EVENT1_RAY1",
        "HOMOGENEOUS_EVENT2_RAY0", "HOMOGENEOUS_EVENT2_RAY1",
    )
    rays = [record_values(raw, marker) for marker in ray_markers]
    scores: list[float] = []
    for marker, ray in zip(ray_markers, rays, strict=True):
        if ray.get("ELIGIBLE") != "true" or ray.get("CHART") not in {
            "X", "Y", "PLUS", "MINUS"
        }:
            fail(f"invalid retained dynamic chart: {marker}")
        lower, upper = interval_bounds(ray.get("PIVOT", ""), f"{marker} pivot")
        if lower <= 0.0 <= upper:
            fail(f"retained dynamic pivot contains zero: {marker}")
        margin = min(abs(lower), abs(upper))
        norm_squared = 2.0 if ray["CHART"] in {"PLUS", "MINUS"} else 1.0
        score = margin * margin / norm_squared
        if not math.isfinite(score) or score <= 0.0:
            fail(f"nonfinite or nonpositive chart score: {marker}")
        scores.append(score)
    chart_switches = sum(
        rays[index]["CHART"] != rays[index + 2]["CHART"] for index in range(2)
    )
    return (
        decimal(c1_width), decimal(affine_width), decimal(min(fixed_widths)),
        decimal(homogeneous_width), decimal(min(scores)), str(chart_switches),
        str(len(raw)),
    )


def verify_summary(summary: dict[str, str], rows: list[dict[str, str]]) -> None:
    valid = [row for row in rows if parse_boolean(row["PROBE_PASS"], "PROBE_PASS")]
    affine_count = sum(parse_boolean(row["AFFINE_PASS"], "AFFINE_PASS") for row in valid)
    homogeneous_count = sum(
        parse_boolean(row["HOMOGENEOUS_PASS"], "HOMOGENEOUS_PASS") for row in valid
    )
    rescue_count = sum(
        parse_boolean(row["HOMOGENEOUS_PASS"], "HOMOGENEOUS_PASS")
        and not parse_boolean(row["AFFINE_PASS"], "AFFINE_PASS")
        for row in valid
    )
    loss_count = sum(
        parse_boolean(row["AFFINE_PASS"], "AFFINE_PASS")
        and not parse_boolean(row["HOMOGENEOUS_PASS"], "HOMOGENEOUS_PASS")
        for row in valid
    )
    ratios_c1 = [
        float(row["HOMOGENEOUS_DET_WIDTH"]) / float(row["C1_DET_WIDTH"])
        for row in valid
    ]
    ratios_affine = [
        float(row["HOMOGENEOUS_DET_WIDTH"]) / float(row["AFFINE_DET_WIDTH"])
        for row in valid
    ]
    ratios_fixed = [
        float(row["HOMOGENEOUS_DET_WIDTH"]) / float(row["FIXED_BEST_DET_WIDTH"])
        for row in valid
    ]
    expected = (
        ("SCHEMA", "sounio.cs6.plucker-cocycle-summary.v1"),
        ("COORDINATE_COUNT", str(len(rows))),
        ("PROBE_VALID_COUNT", str(len(valid))),
        ("COMPUTATION_UNRESOLVED_COUNT", str(len(rows) - len(valid))),
        ("AFFINE_CERTIFIED_COUNT", str(affine_count)),
        ("HOMOGENEOUS_CERTIFIED_COUNT", str(homogeneous_count)),
        ("HOMOGENEOUS_RESCUE_COUNT", str(rescue_count)),
        ("AFFINE_LOSS_COUNT", str(loss_count)),
        (
            "HOMOGENEOUS_STRICT_SUPERSET_OF_AFFINE",
            str(loss_count == 0 and rescue_count > 0).lower(),
        ),
        (
            "PROJECTIVE_FIXED_ANY_CERTIFIED_COUNT",
            str(sum(any(parse_boolean(row[key], key) for key in (
                "PROJECTIVE_X_PASS", "PROJECTIVE_Y_PASS",
                "PROJECTIVE_PLUS_PASS", "PROJECTIVE_MINUS_PASS",
            )) for row in valid)),
        ),
        (
            "HOMOGENEOUS_NARROWER_THAN_AFFINE_COUNT",
            str(sum(float(row["HOMOGENEOUS_DET_WIDTH"]) < float(row["AFFINE_DET_WIDTH"]) for row in valid)),
        ),
        (
            "HOMOGENEOUS_NARROWER_THAN_FIXED_BEST_COUNT",
            str(sum(float(row["HOMOGENEOUS_DET_WIDTH"]) < float(row["FIXED_BEST_DET_WIDTH"]) for row in valid)),
        ),
        (
            "HOMOGENEOUS_NARROWER_THAN_C1_COUNT",
            str(sum(float(row["HOMOGENEOUS_DET_WIDTH"]) < float(row["C1_DET_WIDTH"]) for row in valid)),
        ),
        ("MIN_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(min(ratios_c1))),
        (
            "MEDIAN_HOMOGENEOUS_TO_C1_WIDTH_RATIO",
            decimal(statistics.median(ratios_c1)),
        ),
        ("MEAN_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(mean(ratios_c1))),
        ("MAX_HOMOGENEOUS_TO_C1_WIDTH_RATIO", decimal(max(ratios_c1))),
        (
            "MIN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO",
            decimal(min(ratios_affine)),
        ),
        (
            "MEDIAN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO",
            decimal(statistics.median(ratios_affine)),
        ),
        ("MEAN_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO", decimal(mean(ratios_affine))),
        (
            "MAX_HOMOGENEOUS_TO_AFFINE_WIDTH_RATIO",
            decimal(max(ratios_affine)),
        ),
        (
            "MIN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO",
            decimal(min(ratios_fixed)),
        ),
        (
            "MEDIAN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO",
            decimal(statistics.median(ratios_fixed)),
        ),
        ("MEAN_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO", decimal(mean(ratios_fixed))),
        (
            "MAX_HOMOGENEOUS_TO_FIXED_BEST_WIDTH_RATIO",
            decimal(max(ratios_fixed)),
        ),
        (
            "MINIMUM_CERTIFIED_PIVOT_SCORE",
            decimal(min(float(row["MINIMUM_PIVOT_SCORE"]) for row in valid)),
        ),
        ("CHART_SWITCH_COUNT", str(sum(int(row["CHART_SWITCHES"]) for row in valid))),
        (
            "ONE_RAY_CHART_SWITCH_LEAF_COUNT",
            str(sum(int(row["CHART_SWITCHES"]) == 1 for row in valid)),
        ),
        (
            "TWO_RAY_CHART_SWITCH_LEAF_COUNT",
            str(sum(int(row["CHART_SWITCHES"]) == 2 for row in valid)),
        ),
        ("TOTAL_WORKER_ELAPSED_MS", str(sum(int(row["ELAPSED_MS"]) for row in rows))),
        (
            "MEAN_VALID_WORKER_ELAPSED_MS",
            decimal(mean([float(row["ELAPSED_MS"]) for row in valid])),
        ),
        (
            "MEAN_RECEIPT_BYTES",
            decimal(mean([float(row["RECEIPT_BYTES"]) for row in valid])),
        ),
        (
            "H_PG_CS6_SUPPORTED_ON_RETAINED_COORDINATES",
            str(loss_count == 0 and rescue_count > 0).lower(),
        ),
    )
    require_fields(summary, expected, "summary")
    if (
        summary["H_PG_CS6_SUPPORTED_ON_RETAINED_COORDINATES"]
        != summary["HOMOGENEOUS_STRICT_SUPERSET_OF_AFFINE"]
    ):
        fail("H-PG support and strict-superset predicates disagree")
    boundary = (
        ("MUTATION_TESTS", "76"),
        ("MUTATIONS_REJECTED", "76"),
        ("DISCRETE_POINCARE_COCYCLE", "true"),
        ("CONTINUOUS_RICCATI_INTEGRATED", "false"),
        ("GENERAL_GRASSMANN_PLUCKER_INTEGRATOR", "false"),
        ("EXECUTION_TRUST_MODEL", "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION"),
        ("EXECUTION_PROVENANCE_ATTESTED", "false"),
        ("U250_USED", "false"),
        ("FULL_SOURCE_CARRIER_PROVED", "false"),
        ("HYPERBOLICITY_PROVED", "false"),
        ("CHAOTIC_ATTRACTOR_PROVED", "false"),
        ("NOVELTY_OR_PRIORITY_CLAIM", "false"),
        ("PROMOTION_ELIGIBLE", "false"),
    )
    require_fields(summary, boundary, "summary boundary")
    require_fields(
        summary,
        (
            ("AFFINE_CERTIFIED_COUNT", "28"),
            ("HOMOGENEOUS_CERTIFIED_COUNT", "0"),
            ("HOMOGENEOUS_RESCUE_COUNT", "0"),
            ("AFFINE_LOSS_COUNT", "28"),
            ("HOMOGENEOUS_STRICT_SUPERSET_OF_AFFINE", "false"),
            ("PROJECTIVE_FIXED_ANY_CERTIFIED_COUNT", "0"),
            ("HOMOGENEOUS_NARROWER_THAN_AFFINE_COUNT", "0"),
            ("HOMOGENEOUS_NARROWER_THAN_FIXED_BEST_COUNT", "0"),
            ("HOMOGENEOUS_NARROWER_THAN_C1_COUNT", "43"),
            ("ONE_RAY_CHART_SWITCH_LEAF_COUNT", "12"),
            ("TWO_RAY_CHART_SWITCH_LEAF_COUNT", "40"),
            ("H_PG_CS6_SUPPORTED_ON_RETAINED_COORDINATES", "false"),
        ),
        "retained result",
    )
    if set(summary) != {key for key, _ in (*expected, *boundary)}:
        fail("summary field set mismatch")


def verify_retained(root: Path, python: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        fail("retained root must be a regular directory")
    index = parse_index(root)
    retained = parse_kv(root / "retained-manifest.txt", "retained manifest")
    require_fields(
        retained,
        (
            ("SCHEMA", "sounio.cs6.plucker-cocycle-retained.v1"),
            ("FILES_INDEX_SHA256", digest(root / "files.sha256")),
            ("FILE_COUNT", "235"),
            ("RETAINED_VERIFIER_SHA256", digest(Path(__file__).resolve())),
            ("EVIDENCE_CLASS", "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION"),
            ("COORDINATE_COUNT", "53"),
            ("PROBE_VALID_COUNT", "52"),
            ("COMPUTATION_UNRESOLVED_COUNT", "1"),
            ("PROMOTION_ELIGIBLE", "false"),
        ),
        "retained manifest",
    )
    if set(retained) != {
        "SCHEMA", "FILES_INDEX_SHA256", "FILE_COUNT", "RUN_MANIFEST_SHA256",
        "RETAINED_VERIFIER_SHA256", "EVIDENCE_CLASS", "COORDINATE_COUNT",
        "PROBE_VALID_COUNT", "COMPUTATION_UNRESOLVED_COUNT",
        "PROMOTION_ELIGIBLE",
    }:
        fail("retained manifest field set mismatch")
    run_manifest = parse_kv(root / "run-manifest.txt", "run manifest")
    require_fields(
        run_manifest,
        (
            ("SCHEMA", "sounio.cs6.plucker-cocycle-run-manifest.v1"),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", "5.3.0"),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("ROOT_CHALLENGE", RETAINED_ROOT_CHALLENGE),
            ("LEAF_COUNT", "53"),
            ("PROBE_VALID_COUNT", "52"),
            ("MUTATION_TESTS", "76"),
            ("PROMOTION_ELIGIBLE", "false"),
        ),
        "run manifest",
    )
    if SHA_RE.fullmatch(run_manifest.get("ROOT_CHALLENGE", "")) is None:
        fail("run manifest root challenge is not a canonical SHA-256")
    run_manifest_keys = {
        "SCHEMA", "RUN_COMPLETE", "CAPD_VERSION", "INTERVAL_BACKEND",
        "OPTIMIZATION_LEVEL", "ROOT_CHALLENGE", "SOURCE_SHA256",
        "VERIFIER_SHA256", "RUNNER_SHA256", "COORDINATE_MANIFEST_SHA256",
        "LEAF_COUNT", "PROBE_VALID_COUNT", "MUTATION_TESTS",
        "PROMOTION_ELIGIBLE",
    }
    run_manifest_keys.update(
        relative.upper().replace("-", "_").replace(".", "_") + "_SHA256"
        for relative in RUN_ARTIFACTS
    )
    if set(run_manifest) != run_manifest_keys:
        fail("run manifest field set mismatch")
    if digest(root / "run-manifest.txt") != retained.get("RUN_MANIFEST_SHA256"):
        fail("retained manifest does not bind run manifest")
    for key, relative in (
        ("SOURCE_SHA256", "worker-source.cpp"),
        ("VERIFIER_SHA256", "leaf-verifier.py"),
        ("RUNNER_SHA256", "runner.py"),
        ("COORDINATE_MANIFEST_SHA256", "coordinates.tsv"),
    ):
        if run_manifest.get(key) != digest(root / relative):
            fail(f"run manifest source binding mismatch: {key}")
    for relative in RUN_ARTIFACTS:
        key = relative.upper().replace("-", "_").replace(".", "_") + "_SHA256"
        if run_manifest.get(key) != digest(root / relative):
            fail(f"run manifest artifact binding mismatch: {key}")
    if stable_bytes(root / "mutation-audit-stderr.txt", "mutation audit stderr"):
        fail("retained mutation audit emitted stderr")

    coordinates = parse_coordinates(root / "coordinates.tsv")
    expected_index = set(RUN_ARTIFACTS) | {"run-manifest.txt"}
    for identity in coordinates:
        expected_index.update(
            {
                f"inputs/{identity}.txt",
                f"receipts/{identity}.txt",
                f"stderr/{identity}.txt",
            }
        )
        if identity != "U00-0000000000_S00-0000000000":
            expected_index.add(f"verifications/{identity}.txt")
    if set(index) != expected_index:
        fail("files index path schema mismatch")
    leaves_path = root / "leaves.tsv"
    leaves_raw = stable_bytes(leaves_path, "leaves table")
    try:
        leaves_text = leaves_raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError("leaves table must be ASCII") from error
    if not leaves_raw.endswith(b"\n") or "\r" in leaves_text or "\0" in leaves_text:
        fail("leaves table is not canonical LF-terminated text")
    reader = csv.DictReader(io.StringIO(leaves_text, newline=""), delimiter="\t")
    if tuple(reader.fieldnames or ()) != LEAF_COLUMNS:
        fail("leaves table header mismatch")
    rows = list(reader)
    if any(None in row or any(value is None or value == "" for value in row.values()) for row in rows):
        fail("leaves table contains missing or extra fields")
    if len(rows) != 53 or [row["LEAF_ID"] for row in rows] != sorted(coordinates):
        fail("leaves table differs from coordinate manifest")
    root_challenge = run_manifest["ROOT_CHALLENGE"]
    source_sha = run_manifest["SOURCE_SHA256"]
    valid_rows: list[dict[str, str]] = []
    for row in rows:
        identity = row["LEAF_ID"]
        u_depth, u_index, s_depth, s_index, input_sha = coordinates[identity]
        if tuple(
            canonical_uint(row[key], f"{identity} {key}")
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        ) != (
            u_depth, u_index, s_depth, s_index
        ):
            fail(f"leaf coordinate mismatch: {identity}")
        if row["INPUT_SHA256"] != input_sha:
            fail(f"leaf input hash mismatch: {identity}")
        input_path = root / "inputs" / f"{identity}.txt"
        canonical_input = leaf_input_bytes(u_depth, u_index, s_depth, s_index)
        if stable_bytes(input_path, f"input {identity}") != canonical_input:
            fail(f"noncanonical retained input: {identity}")
        expected_challenge = challenge(root_challenge, identity, input_sha)
        if row["LEAF_CHALLENGE"] != expected_challenge:
            fail(f"leaf challenge mismatch: {identity}")
        elapsed_ms = canonical_uint(row["ELAPSED_MS"], f"{identity} elapsed")
        if elapsed_ms == 0:
            fail(f"leaf elapsed time must be positive: {identity}")
        receipt_path = root / "receipts" / f"{identity}.txt"
        stderr_path = root / "stderr" / f"{identity}.txt"
        if digest(receipt_path) != row["RECEIPT_SHA256"]:
            fail(f"receipt digest mismatch: {identity}")
        if digest(stderr_path) != row["STDERR_SHA256"]:
            fail(f"stderr digest mismatch: {identity}")
        probe_pass = parse_boolean(row["PROBE_PASS"], "PROBE_PASS")
        if not probe_pass:
            if identity != "U00-0000000000_S00-0000000000":
                fail("only the canonical root may be computation-unresolved")
            expected_root = {
                "STATUS": "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN",
                "METHOD": "NONE",
                "PROBE_PASS": "false",
                "AFFINE_PASS": "false",
                "PROJECTIVE_X_PASS": "false",
                "PROJECTIVE_Y_PASS": "false",
                "PROJECTIVE_PLUS_PASS": "false",
                "PROJECTIVE_MINUS_PASS": "false",
                "HOMOGENEOUS_PASS": "false",
                "CERTIFICATE_PASS": "false",
                "SUBDIVISION_REQUIRED": "true",
                "VERIFICATION_SHA256": ZERO_SHA256,
                "PHYSICAL_SHA256": ZERO_SHA256,
                "WORKER_RC": "1",
            }
            if any(row[key] != value for key, value in expected_root.items()):
                fail("root semantic row mismatch")
            if any(row[key] != "-" for key in METRIC_COLUMNS):
                fail("unresolved root carries fabricated metrics")
            if stable_bytes(receipt_path, "root receipt"):
                fail("root interval-domain failure payload mismatch")
            stderr = stable_bytes(stderr_path, "root stderr").lower()
            if b"interval error:" not in stderr or b"division by 0" not in stderr:
                fail("root failure is not the retained interval-domain class")
            continue

        if row["WORKER_RC"] != "0":
            fail(f"computed leaf has nonzero worker status: {identity}")
        if stable_bytes(stderr_path, f"stderr {identity}"):
            fail(f"computed worker emitted stderr: {identity}")
        receipt_raw = stable_bytes(receipt_path, f"receipt {identity}")
        if not receipt_raw:
            fail(f"computed leaf has empty receipt: {identity}")
        verification_path = root / "verifications" / f"{identity}.txt"
        replay = subprocess.run(
            [
                python,
                root / "leaf-verifier.py",
                receipt_path,
                "--source-sha",
                source_sha,
                "--input",
                input_path,
                "--challenge",
                expected_challenge,
                "--require-probe",
            ],
            capture_output=True,
        )
        if replay.returncode != 0 or replay.stderr:
            fail(f"retained leaf verifier failed: {identity}")
        if replay.stdout != stable_bytes(verification_path, f"verification {identity}"):
            fail(f"retained verifier output drift: {identity}")
        verification = parse_kv(verification_path, f"verification {identity}")
        if verification.get("RECEIPT_SHA256") != row["RECEIPT_SHA256"]:
            fail(f"verification receipt binding mismatch: {identity}")
        if verification.get("PHYSICAL_SHA256") != row["PHYSICAL_SHA256"]:
            fail(f"verification physical binding mismatch: {identity}")
        if digest(verification_path) != row["VERIFICATION_SHA256"]:
            fail(f"verification digest mismatch: {identity}")
        if row["METHOD"] != verification.get("LEAF_METHOD"):
            fail(f"leaf table verifier mismatch: {identity} METHOD")
        for table_key, verification_key in (
            ("PROBE_PASS", "PROBE_PASS"),
            ("AFFINE_PASS", "AFFINE_CERTIFICATE_PASS"),
            ("PROJECTIVE_X_PASS", "PROJECTIVE_X_CERTIFICATE_PASS"),
            ("PROJECTIVE_Y_PASS", "PROJECTIVE_Y_CERTIFICATE_PASS"),
            ("PROJECTIVE_PLUS_PASS", "PROJECTIVE_PLUS_CERTIFICATE_PASS"),
            ("PROJECTIVE_MINUS_PASS", "PROJECTIVE_MINUS_CERTIFICATE_PASS"),
            ("HOMOGENEOUS_PASS", "HOMOGENEOUS_CERTIFICATE_PASS"),
            ("CERTIFICATE_PASS", "CERTIFICATE_PASS"),
            ("SUBDIVISION_REQUIRED", "SUBDIVISION_REQUIRED"),
        ):
            if row[table_key] != verification.get(verification_key):
                fail(f"leaf table verifier mismatch: {identity} {table_key}")
        expected_status = (
            "PROBE_VALID_CERTIFIED"
            if parse_boolean(row["CERTIFICATE_PASS"], "CERTIFICATE_PASS")
            else "PROBE_VALID_UNCERTIFIED"
        )
        if row["STATUS"] != expected_status:
            fail(f"computed leaf status mismatch: {identity}")
        expected_metrics = receipt_metrics(receipt_raw)
        actual_metrics = tuple(row[key] for key in METRIC_COLUMNS)
        if actual_metrics != expected_metrics:
            fail(f"computed leaf metrics mismatch: {identity}")
        valid_rows.append(row)

    if len(valid_rows) != 52:
        fail("retained valid-leaf cardinality mismatch")
    audit_row = valid_rows[0]
    audit_id = audit_row["LEAF_ID"]
    audit = subprocess.run(
        [
            python,
            root / "leaf-verifier.py",
            root / "receipts" / f"{audit_id}.txt",
            "--source-sha",
            source_sha,
            "--input",
            root / "inputs" / f"{audit_id}.txt",
            "--challenge",
            audit_row["LEAF_CHALLENGE"],
            "--self-test-mutations",
            "--require-probe",
        ],
        capture_output=True,
    )
    if audit.returncode != 0 or audit.stderr or audit.stdout != stable_bytes(
        root / "mutation-audit.txt", "mutation audit"
    ):
        fail("retained mutation audit does not replay")
    verify_summary(parse_kv(root / "summary.txt", "summary"), rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retained_dir", type=Path)
    args = parser.parse_args(argv)
    python = Path(sys.executable).resolve()
    verify_retained(args.retained_dir.resolve(), python)
    print("RETAINED_SCHEMA=sounio.cs6.plucker-cocycle-retained.v1")
    print("COORDINATE_COUNT=53")
    print("PROBE_VALID_COUNT=52")
    print("LEAF_REPLAY_MATCH=52/52")
    print("MUTATION_AUDIT_MATCH=true")
    print("FILES_INDEX_MATCH=true")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetainedError as error:
        print(f"retained verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
