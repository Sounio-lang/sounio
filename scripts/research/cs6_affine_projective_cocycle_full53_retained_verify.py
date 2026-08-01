#!/usr/bin/env python3
"""Verify retained evidence for the predeclared 53-coordinate CS6 H-APG run."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import math
import os
import re
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
CHALLENGE_DOMAIN = b"sounio.cs6.affine-projective-cocycle-full53-leaf-challenge.v1\0"
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
PREDECLARED_COMMIT = "9dcf1fca964d7e54e1109f9210689809666b2a54"
PREDECLARATION_REPORT_SHA256 = "74d8f596c9258eab49775192e8b244e032dfb5e3481a23804075c66ff316618c"
PARENT_RUN_MANIFEST_SHA256 = "21f0ca224fc79d1cfc815f203100e091bf27dbb070cb3a2c69f6056b7a6fee27"
PARENT_FILES_INDEX_SHA256 = "740ae39070ff90604048c37d25e2f0de954be9ea55e020a529e094699a82802d"
PARENT_COORDINATES_SHA256 = "6169dd7705ca4f01180f65d13d620320845684f0a9fc28411c881cfae7e4f2d8"
PARENT_LEAVES_SHA256 = "98c639a616ac640f1983209530f8fe30d769a0e4b0136665201c9bc57320e37f"
PARENT_AFFINE_OBLIGATION_SHA256 = "9f36931e672aba1b9735c45eef286fbca97da67b911c64daa0b3b8c8affecd6c"
CONTRACT_FROZEN_IN_COMMIT = "58905019754cf66f077a0db228f1a99a4a7612eb"
FROZEN_MANIFEST_SHA256 = "61b2b0649983a332b5abb530443a3ff14a19e62514ef9b1d3175d8e9a6bbfd9c"
ROOT_LEAF_ID = "U00-0000000000_S00-0000000000"
PARENT_RUN = "cs6_plucker_cocycle_retained_53_v1"
MUTATIONS_PER_LEAF = 112
MUTATION_TOTAL = 52 * MUTATIONS_PER_LEAF
EXCLUDED_INDEX_FILES = {"files.sha256", "retained-manifest.txt"}
LEAF_COLUMNS = (
    "LEAF_ID", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "STATUS",
    "PARENT_STATUS", "PARENT_AFFINE_PASS", "PARENT_RECEIPT_SHA256",
    "E1_R0_CHART", "E1_R0_SIGN", "E1_R1_CHART", "E1_R1_SIGN",
    "E2_R0_CHART", "E2_R0_SIGN", "E2_R1_CHART", "E2_R1_SIGN",
    "METHOD", "PROBE_PASS", "AFFINE_PASS", "BOXED_HOMOGENEOUS_PASS",
    "APG_VALID", "APG_PASS", "APG_RESCUE", "APG_NARROWER_BOXED",
    "APG_NARROWER_AFFINE", "APG_NARROWER_SHARED", "CERTIFICATE_PASS",
    "SUBDIVISION_REQUIRED",
    "INPUT_SHA256", "LEAF_CHALLENGE", "RECEIPT_SHA256", "STDERR_SHA256",
    "VERIFICATION_SHA256", "PHYSICAL_SHA256", "WORKER_RC", "ELAPSED_MS",
    "C1_DET_WIDTH", "AFFINE_DET_WIDTH", "BOXED_DET_WIDTH",
    "SHARED_DET_WIDTH", "APG_PRIMARY_DET_WIDTH",
    "MINIMUM_APG_PIVOT_MARGIN", "RECEIPT_BYTES",
)
METRIC_COLUMNS = (
    "C1_DET_WIDTH", "AFFINE_DET_WIDTH", "BOXED_DET_WIDTH",
    "SHARED_DET_WIDTH", "APG_PRIMARY_DET_WIDTH",
    "MINIMUM_APG_PIVOT_MARGIN", "RECEIPT_BYTES",
)
RUN_ARTIFACTS = (
    "capd-cflags.txt", "capd-libs.txt", "capd-version.txt",
    "compile-command.txt", "compile-stderr.txt", "compile-stdout.txt",
    "compiler-version.txt", "coordinates.tsv", "dependencies.sha256",
    "executed-worker.sha256", "git-head.txt", "git-status.txt",
    "leaf-verifier.py", "leaves.tsv",
    "link-inputs.sha256", "mutation-audit-stderr.txt", "mutation-audit.txt",
    "mutation-audits.tsv", "python-version.txt", "run-contract.txt", "runner.py",
    "runtime-libraries.sha256", "runtime-linkage.txt", "summary.txt",
    "worker-source.cpp",
)
MINIMUM_FILE_COUNT = len(RUN_ARTIFACTS) + 1 + 53 * 3
CHARTS = {"X", "Y", "PLUS", "MINUS"}
MANIFEST_HEADER = (
    "SCHEMA=sounio.cs6.affine-projective-cocycle-full53-coordinates.v1",
    "PARENT_COORDINATE_SET=CS6_PLUCKER_COCYCLE_RETAINED_53_V1",
    "LEAF_COUNT=53",
    "PAIRED_ELIGIBLE_COUNT=52",
    "PARENT_AFFINE_PASS_COUNT=28",
    "SELECTION=ALL_PARENT_COORDINATES_NO_POSTHOC_FILTER",
    f"PREDECLARED_IN_COMMIT={PREDECLARED_COMMIT}",
    f"PREDECLARATION_REPORT_SHA256={PREDECLARATION_REPORT_SHA256}",
    f"PARENT_RUN_MANIFEST_SHA256={PARENT_RUN_MANIFEST_SHA256}",
    f"PARENT_FILES_INDEX_SHA256={PARENT_FILES_INDEX_SHA256}",
    f"PARENT_COORDINATES_SHA256={PARENT_COORDINATES_SHA256}",
    f"PARENT_LEAVES_SHA256={PARENT_LEAVES_SHA256}",
    "PARENT_AFFINE_OBLIGATION_CANONICALIZATION=SORTED_LEAF_ID_UTF8_LF_WITH_FINAL_LF",
    f"PARENT_AFFINE_OBLIGATION_SHA256={PARENT_AFFINE_OBLIGATION_SHA256}",
    "CHART_PROVENANCE=PARENT_H_PG_SELECTED_CHART",
    "CHART_POLICY=FIXED_PER_LEAF_NO_RUNTIME_RESELECTION",
    "ROOT_POLICY=ATTEMPT_AND_REQUIRE_SAME_UNRESOLVED_CLASS",
    "AFFINE_LOSS_POLICY=ANY_CHILD_UNRESOLVED_INVALID_OR_UNCERTIFIED_ON_PARENT_AFFINE_PASS_LEAF",
    "ACCEPTANCE=53_ATTEMPTED_52_PAIRED_VALID_NO_NEW_UNRESOLVED_ZERO_AFFINE_LOSS_AT_LEAST_ONE_RESCUE_EXACT_REPLAY_ALL_MUTATIONS_REJECTED",
    "LEAF_CHALLENGE_BINDING=PARENT_INPUT_SHA256_PARENT_RECEIPT_SHA256_CHART_SIGN_TUPLE_MANIFEST_SHA256",
    "PROMOTION_REQUIRES_INDEPENDENT_ATTESTATION=true",
    "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256\tPARENT_STATUS\tPARENT_RECEIPT_SHA256\tE1_R0_CHART\tE1_R0_SIGN\tE1_R1_CHART\tE1_R1_SIGN\tE2_R0_CHART\tE2_R0_SIGN\tE2_R1_CHART\tE2_R1_SIGN",
)
VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA", "RECEIPT_SHA256", "PHYSICAL_SHA256",
    "COORDINATE_MANIFEST_SHA256", "PARENT_RECEIPT_SHA256", "LEAF_CHALLENGE",
    "MUTATION_TESTS", "MUTATIONS_REJECTED", "LEAF_METHOD", "PROBE_PASS",
    "AFFINE_CERTIFICATE_PASS", "PROJECTIVE_X_CERTIFICATE_PASS",
    "PROJECTIVE_Y_CERTIFICATE_PASS", "PROJECTIVE_PLUS_CERTIFICATE_PASS",
    "PROJECTIVE_MINUS_CERTIFICATE_PASS", "HOMOGENEOUS_CERTIFICATE_PASS",
    "APG_COMPUTATION_VALID", "APG_CERTIFICATE_PASS", "APG_RESCUE",
    "APG_STRICTLY_NARROWER_THAN_BOXED", "APG_STRICTLY_NARROWER_THAN_AFFINE",
    "APG_STRICTLY_NARROWER_THAN_SHARED", "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)
MUTATION_COLUMNS = (
    "LEAF_ID", "RECEIPT_SHA256", "VERIFICATION_SHA256",
    "AUDIT_STDOUT_SHA256", "MUTATION_TESTS", "MUTATIONS_REJECTED",
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
    if len(result) < MINIMUM_FILE_COUNT:
        fail("files index cardinality is below the complete-run minimum")
    return result


@dataclass(frozen=True)
class Coordinate:
    identity: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    input_sha: str
    parent_status: str
    parent_receipt_sha: str
    chart_signs: tuple[tuple[str, int], ...]


def git_blob(repo: Path, commit: str, relative: str, label: str) -> bytes:
    process = subprocess.run(
        ["git", "-C", repo, "show", f"{commit}:{relative}"], capture_output=True
    )
    if process.returncode != 0 or process.stderr:
        fail(f"cannot read committed {label}")
    return process.stdout


def parse_coordinates(path: Path, repo: Path) -> dict[str, Coordinate]:
    raw = stable_bytes(path, "coordinates")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RetainedError("coordinates must be ASCII") from error
    if (
        not raw.endswith(b"\n")
        or b"\r" in raw
        or b"\0" in raw
        or len(lines) != len(MANIFEST_HEADER) + 53
        or tuple(lines[: len(MANIFEST_HEADER)]) != MANIFEST_HEADER
    ):
        fail("coordinate manifest header mismatch")
    if digest_bytes(raw) != FROZEN_MANIFEST_SHA256:
        fail("coordinate manifest differs from the frozen hash")
    frozen = git_blob(
        repo,
        CONTRACT_FROZEN_IN_COMMIT,
        "scripts/research/cs6_affine_projective_cocycle_full53_coordinates_v1.tsv",
        "coordinate contract",
    )
    if raw != frozen:
        fail("coordinate manifest differs from the committed frozen contract")
    result: dict[str, Coordinate] = {}
    for line in lines[len(MANIFEST_HEADER) :]:
        fields = line.split("\t")
        if len(fields) != 16:
            fail("coordinate manifest row width mismatch")
        (
            identity, u_depth, u_index, s_depth, s_index, input_sha,
            parent_status, parent_receipt_sha,
            e1_r0_chart, e1_r0_sign, e1_r1_chart, e1_r1_sign,
            e2_r0_chart, e2_r0_sign, e2_r1_chart, e2_r1_sign,
        ) = fields
        numbers = tuple(
            canonical_uint(token, f"coordinate {identity}")
            for token in (u_depth, u_index, s_depth, s_index)
        )
        ud, ui, sd, si = numbers
        if ud > 30 or sd > 30 or ui >= 1 << ud or si >= 1 << sd:
            fail("coordinate manifest dyadic range mismatch")
        if identity != leaf_id(ud, ui, sd, si):
            fail("coordinate manifest identity mismatch")
        if (
            SHA_RE.fullmatch(input_sha) is None
            or SHA_RE.fullmatch(parent_receipt_sha) is None
            or not parent_status
            or identity in result
        ):
            fail("coordinate manifest digest or uniqueness mismatch")
        if digest_bytes(leaf_input_bytes(ud, ui, sd, si)) != input_sha:
            fail("coordinate manifest input digest mismatch")
        raw_pairs = (
            (e1_r0_chart, e1_r0_sign), (e1_r1_chart, e1_r1_sign),
            (e2_r0_chart, e2_r0_sign), (e2_r1_chart, e2_r1_sign),
        )
        if identity == ROOT_LEAF_ID:
            if numbers != (0, 0, 0, 0) or any(
                chart != "NONE" or sign != "0" for chart, sign in raw_pairs
            ):
                fail("coordinate manifest root sentinel mismatch")
        elif any(chart not in CHARTS or sign not in {"-1", "1"} for chart, sign in raw_pairs):
            fail("coordinate manifest chart/sign grammar mismatch")
        result[identity] = Coordinate(
            identity, ud, ui, sd, si, input_sha, parent_status,
            parent_receipt_sha,
            tuple((chart, int(sign)) for chart, sign in raw_pairs),
        )
    if len(result) != 53 or list(result) != sorted(result):
        fail("coordinate manifest is not the canonical ordered 53-row set")
    if ROOT_LEAF_ID not in result:
        fail("coordinate manifest omits the root")
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


def challenge(root: str, manifest_sha: str, coordinate: Coordinate) -> str:
    chart_fields: list[str] = []
    for chart, sign in coordinate.chart_signs:
        chart_fields.extend((chart, str(sign)))
    chart_tuple = ("\t".join(chart_fields) + "\n").encode("ascii")
    return digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + coordinate.identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(coordinate.input_sha)
        + b"\0"
        + bytes.fromhex(coordinate.parent_receipt_sha)
        + b"\0"
        + chart_tuple
        + b"\0"
        + bytes.fromhex(manifest_sha)
    )


def parse_boolean(token: str, label: str) -> bool:
    if token not in {"true", "false"}:
        fail(f"noncanonical boolean: {label}")
    return token == "true"


def decimal(value: float) -> str:
    return format(value, ".17g")


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def minimum(values: Sequence[float]) -> float:
    return min(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def maximum(values: Sequence[float]) -> float:
    return max(values) if values else float("nan")


def canonical_uint(token: str, label: str) -> int:
    if not token.isdigit() or str(int(token)) != token:
        fail(f"noncanonical unsigned integer: {label}")
    return int(token)


def canonical_int(token: str, label: str) -> int:
    if re.fullmatch(r"0|-?[1-9][0-9]*", token) is None:
        fail(f"noncanonical integer: {label}")
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
    boxed_width = interval_width(
        record_values(raw, "PLUCKER_COCYCLE"), "DET", "boxed homogeneous"
    )
    shared_width = interval_width(
        record_values(raw, "APG_SHARED_COMPOSITION_TM2"),
        "DET_HULL",
        "shared composition",
    )
    apg_width = interval_width(
        record_values(raw, "APG_FACTORED_EXTERIOR_TM2"),
        "PRIMARY_DET",
        "factored APG",
    )
    ray_markers = (
        "APG_EVENT1_RAY0", "APG_EVENT1_RAY1",
        "APG_EVENT2_RAY0", "APG_EVENT2_RAY1",
    )
    rays = [record_values(raw, marker) for marker in ray_markers]
    margins: list[float] = []
    for marker, ray in zip(ray_markers, rays, strict=True):
        if ray.get("PIVOT_SIGN_CERTIFIED") != "true" or ray.get("CHART") not in {
            "X", "Y", "PLUS", "MINUS"
        }:
            fail(f"invalid retained frozen APG chart: {marker}")
        lower, upper = interval_bounds(ray.get("P_HULL", ""), f"{marker} pivot")
        if lower <= 0.0 <= upper:
            fail(f"retained APG pivot contains zero: {marker}")
        margin = min(abs(lower), abs(upper))
        if not math.isfinite(margin) or margin <= 0.0:
            fail(f"nonfinite or nonpositive APG pivot margin: {marker}")
        margins.append(margin)
    return (
        decimal(c1_width), decimal(affine_width), decimal(boxed_width),
        decimal(shared_width), decimal(apg_width), decimal(min(margins)), str(len(raw)),
    )


def canonical_interval_domain_failure(stderr: bytes) -> bool:
    try:
        lines = stderr.decode("ascii").splitlines()
    except UnicodeError:
        return False
    if (
        len(lines) != 2
        or lines[0]
        != "probe error: Interval error: Division by 0 in operator/(Interval, Interval)"
        or not lines[1].startswith("   left=")
        or "  right=" not in lines[1]
        or not stderr.endswith(b"\n")
    ):
        return False
    left, right = lines[1].removeprefix("   left=").split("  right=", 1)
    try:
        return math.isfinite(float(left)) and math.isfinite(float(right))
    except ValueError:
        return False


def known_interval_domain_failure(stderr: bytes) -> bool:
    lowered = stderr.lower()
    return b"interval error:" in lowered and (
        b"division by 0" in lowered or b"division by zero" in lowered
    )


def known_frozen_chart_failure(stderr: bytes) -> bool:
    return b"frozen TM2 pivot sign was not certified" in stderr


def unresolved_status_matches(status: str, worker_rc: int, stderr: bytes) -> bool:
    interval_domain = known_interval_domain_failure(stderr)
    frozen_chart = known_frozen_chart_failure(stderr)
    if status == "COMPUTATION_UNRESOLVED_TIMEOUT":
        return worker_rc == 124
    if status == "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN":
        return worker_rc != 124 and interval_domain
    if status == "COMPUTATION_UNRESOLVED_FROZEN_CHART":
        return worker_rc != 124 and not interval_domain and frozen_chart
    if status == "COMPUTATION_UNRESOLVED_UNEXPECTED_WORKER_FAILURE":
        return worker_rc != 124 and not interval_domain and not frozen_chart
    return False


def parse_tsv(path: Path, columns: Sequence[str], label: str) -> list[dict[str, str]]:
    raw = stable_bytes(path, label)
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or "\r" in text or "\0" in text:
        fail(f"{label} is not canonical LF-terminated text")
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    if tuple(reader.fieldnames or ()) != tuple(columns):
        fail(f"{label} header mismatch")
    rows = list(reader)
    if any(
        None in row
        or any(value is None or value == "" for value in row.values())
        for row in rows
    ):
        fail(f"{label} contains missing or extra fields")
    return rows


def parse_verification(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or "\r" in text or "\0" in text:
        fail(f"{label} is not canonical LF-terminated text")
    lines = text.splitlines()
    if len(lines) != len(VERIFICATION_KEYS):
        fail(f"{label} field count mismatch")
    result: dict[str, str] = {}
    for line, expected_key in zip(lines, VERIFICATION_KEYS, strict=True):
        if line.count("=") != 1:
            fail(f"malformed field in {label}")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            fail(f"{label} field order mismatch: {expected_key}")
        result[key] = value
    return result


def parent_chart_sign(raw: bytes, event: int, ray: int) -> tuple[str, int]:
    record = record_values(raw, f"HOMOGENEOUS_EVENT{event}_RAY{ray}")
    chart = record.get("CHART", "")
    lower, upper = interval_bounds(record.get("PIVOT", ""), "parent pivot")
    sign = -1 if upper < 0.0 else 1 if lower > 0.0 else 0
    if chart not in CHARTS or sign == 0:
        fail("parent chart does not have a certified signed pivot")
    return chart, sign


def verify_parent_contract(
    repo: Path, coordinates: dict[str, Coordinate], python: Path
) -> set[str]:
    parent = repo / "scripts/research/receipts" / PARENT_RUN
    for relative, expected in (
        ("run-manifest.txt", PARENT_RUN_MANIFEST_SHA256),
        ("files.sha256", PARENT_FILES_INDEX_SHA256),
        ("coordinates.tsv", PARENT_COORDINATES_SHA256),
        ("leaves.tsv", PARENT_LEAVES_SHA256),
    ):
        if digest(parent / relative) != expected:
            fail(f"parent anchor mismatch: {relative}")
    parent_check = subprocess.run(
        [
            python,
            repo / "scripts/research/cs6_plucker_cocycle_retained_verify.py",
            parent,
        ],
        capture_output=True,
    )
    if parent_check.returncode != 0 or parent_check.stderr:
        fail("anchored parent retained bundle does not independently verify")

    raw = stable_bytes(parent / "leaves.tsv", "parent leaves")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RetainedError("parent leaves must be ASCII") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), delimiter="\t")
    rows = list(reader)
    if (
        len(rows) != 53
        or [row.get("LEAF_ID") for row in rows] != list(coordinates)
        or any(None in row for row in rows)
    ):
        fail("parent leaves population differs from the frozen coordinates")
    parent_by_id = {row["LEAF_ID"]: row for row in rows}
    affine_ids = sorted(
        identity
        for identity, row in parent_by_id.items()
        if row.get("AFFINE_PASS") == "true"
    )
    affine_raw = ("\n".join(affine_ids) + "\n").encode("ascii")
    if (
        len(affine_ids) != 28
        or digest_bytes(affine_raw) != PARENT_AFFINE_OBLIGATION_SHA256
    ):
        fail("parent affine-obligation set differs from the frozen set")

    for identity, coordinate in coordinates.items():
        row = parent_by_id[identity]
        if (
            row.get("STATUS") != coordinate.parent_status
            or row.get("INPUT_SHA256") != coordinate.input_sha
            or row.get("RECEIPT_SHA256") != coordinate.parent_receipt_sha
        ):
            fail(f"coordinate parent binding mismatch: {identity}")
        parent_input = parent / "inputs" / f"{identity}.txt"
        parent_receipt = parent / "receipts" / f"{identity}.txt"
        if (
            digest(parent_input) != coordinate.input_sha
            or digest(parent_receipt) != coordinate.parent_receipt_sha
        ):
            fail(f"coordinate parent payload mismatch: {identity}")
        if identity == ROOT_LEAF_ID:
            expected_pairs = (("NONE", 0),) * 4
        else:
            receipt_raw = stable_bytes(parent_receipt, f"parent receipt {identity}")
            expected_pairs = tuple(
                parent_chart_sign(receipt_raw, event, ray)
                for event, ray in ((1, 0), (1, 1), (2, 0), (2, 1))
            )
        if coordinate.chart_signs != expected_pairs:
            fail(f"coordinate chart/sign tuple differs from parent: {identity}")
    return set(affine_ids)


def verifier_command(
    python: Path,
    root: Path,
    repo: Path,
    row: dict[str, str],
    source_sha: str,
    *,
    mutations: bool,
) -> list[Path | str]:
    identity = row["LEAF_ID"]
    command: list[Path | str] = [
        python,
        root / "leaf-verifier.py",
        root / "receipts" / f"{identity}.txt",
        "--source-sha",
        source_sha,
        "--input",
        root / "inputs" / f"{identity}.txt",
        "--coordinate-manifest",
        root / "coordinates.tsv",
        "--parent-receipt",
        repo / "scripts/research/receipts" / PARENT_RUN / "receipts"
        / f"{identity}.txt",
        "--root-challenge",
        row["ROOT_CHALLENGE"],
    ]
    if mutations:
        command.append("--self-test-mutations")
    return command


def verify_summary(
    summary: dict[str, str],
    rows: list[dict[str, str]],
    audit_rows: list[dict[str, str]],
    root_class_match: bool,
) -> dict[str, str]:
    valid = [row for row in rows if parse_boolean(row["PROBE_PASS"], "PROBE_PASS")]
    paired_valid = [row for row in valid if row["LEAF_ID"] != ROOT_LEAF_ID]
    affine_count = sum(
        parse_boolean(row["AFFINE_PASS"], "AFFINE_PASS") for row in paired_valid
    )
    apg_count = sum(
        parse_boolean(row["APG_PASS"], "APG_PASS") for row in paired_valid
    )
    parent_nonaffine = [
        row
        for row in rows
        if row["LEAF_ID"] != ROOT_LEAF_ID
        and not parse_boolean(row["PARENT_AFFINE_PASS"], "PARENT_AFFINE_PASS")
    ]
    rescue_count = sum(
        parse_boolean(row["APG_RESCUE"], "APG_RESCUE")
        for row in parent_nonaffine
    )
    obligations = [
        row
        for row in rows
        if parse_boolean(row["PARENT_AFFINE_PASS"], "PARENT_AFFINE_PASS")
    ]
    affine_loss_count = sum(
        not (
            parse_boolean(row["PROBE_PASS"], "PROBE_PASS")
            and parse_boolean(row["APG_VALID"], "APG_VALID")
            and parse_boolean(row["APG_PASS"], "APG_PASS")
        )
        for row in obligations
    )
    narrower_boxed_count = sum(
        parse_boolean(row["APG_NARROWER_BOXED"], "APG_NARROWER_BOXED")
        for row in valid
    )
    narrower_affine_count = sum(
        parse_boolean(row["APG_NARROWER_AFFINE"], "APG_NARROWER_AFFINE")
        for row in valid
    )
    narrower_shared_count = sum(
        parse_boolean(row["APG_NARROWER_SHARED"], "APG_NARROWER_SHARED")
        for row in valid
    )
    ratios_boxed = [
        float(row["APG_PRIMARY_DET_WIDTH"]) / float(row["BOXED_DET_WIDTH"])
        for row in paired_valid
    ]
    ratios_affine = [
        float(row["APG_PRIMARY_DET_WIDTH"]) / float(row["AFFINE_DET_WIDTH"])
        for row in paired_valid
    ]
    ratios_shared = [
        float(row["APG_PRIMARY_DET_WIDTH"]) / float(row["SHARED_DET_WIDTH"])
        for row in paired_valid
    ]
    for label, values in (
        ("boxed", ratios_boxed),
        ("affine", ratios_affine),
        ("shared", ratios_shared),
    ):
        if any(not math.isfinite(value) or value <= 0.0 for value in values):
            fail(f"nonfinite or nonpositive APG width ratio: {label}")
    suite_sizes = {
        canonical_uint(row["MUTATION_TESTS"], "mutation suite size")
        for row in audit_rows
    }
    if len(suite_sizes) > 1:
        fail("mutation suite size differs across retained leaves")
    suite_size = next(iter(suite_sizes), 0)
    mutation_tests = sum(
        canonical_uint(row["MUTATION_TESTS"], "mutation tests")
        for row in audit_rows
    )
    mutations_rejected = sum(
        canonical_uint(row["MUTATIONS_REJECTED"], "mutations rejected")
        for row in audit_rows
    )
    new_unresolved = sum(
        not parse_boolean(row["PROBE_PASS"], "PROBE_PASS")
        for row in rows
        if row["LEAF_ID"] != ROOT_LEAF_ID
    )
    full53_supported = (
        len(rows) == 53
        and len(paired_valid) == 52
        and root_class_match
        and new_unresolved == 0
        and all(parse_boolean(row["APG_VALID"], "APG_VALID") for row in paired_valid)
        and affine_loss_count == 0
        and rescue_count > 0
        and len(audit_rows) == 52
        and suite_size == MUTATIONS_PER_LEAF
        and mutation_tests == MUTATION_TOTAL
        and mutations_rejected == MUTATION_TOTAL
    )
    expected = (
        ("SCHEMA", "sounio.cs6.affine-projective-cocycle-full53-summary.v1"),
        ("COORDINATE_COUNT", str(len(rows))),
        ("PAIRED_ELIGIBLE_COUNT", "52"),
        ("PAIRED_VALID_COUNT", str(len(paired_valid))),
        ("PROBE_VALID_COUNT", str(len(paired_valid))),
        ("ROOT_INTERVAL_DOMAIN_CLASS_MATCH", str(root_class_match).lower()),
        ("NEW_UNRESOLVED_COUNT", str(new_unresolved)),
        (
            "COMPUTATION_UNRESOLVED_COUNT",
            str(sum(row["STATUS"].startswith("COMPUTATION_UNRESOLVED_") for row in rows)),
        ),
        ("PARENT_AFFINE_OBLIGATION_COUNT", str(len(obligations))),
        ("PARENT_AFFINE_PRESERVED_COUNT", str(len(obligations) - affine_loss_count)),
        ("PARENT_AFFINE_LOSS_COUNT", str(affine_loss_count)),
        ("PARENT_NONAFFINE_COUNT", str(len(parent_nonaffine))),
        ("AFFINE_CERTIFIED_COUNT", str(affine_count)),
        ("APG_CERTIFIED_COUNT", str(apg_count)),
        ("APG_RESCUE_ON_PARENT_NONAFFINE_COUNT", str(rescue_count)),
        ("APG_RESCUE_COUNT", str(rescue_count)),
        ("AFFINE_LOSS_COUNT", str(affine_loss_count)),
        ("APG_NARROWER_THAN_BOXED_COUNT", str(narrower_boxed_count)),
        ("APG_NARROWER_THAN_AFFINE_COUNT", str(narrower_affine_count)),
        ("APG_NARROWER_THAN_SHARED_COUNT", str(narrower_shared_count)),
        ("MIN_APG_TO_BOXED_WIDTH_RATIO", decimal(minimum(ratios_boxed))),
        ("MEDIAN_APG_TO_BOXED_WIDTH_RATIO", decimal(median(ratios_boxed))),
        ("MEAN_APG_TO_BOXED_WIDTH_RATIO", decimal(mean(ratios_boxed))),
        ("MAX_APG_TO_BOXED_WIDTH_RATIO", decimal(maximum(ratios_boxed))),
        ("MIN_APG_TO_AFFINE_WIDTH_RATIO", decimal(minimum(ratios_affine))),
        ("MEDIAN_APG_TO_AFFINE_WIDTH_RATIO", decimal(median(ratios_affine))),
        ("MEAN_APG_TO_AFFINE_WIDTH_RATIO", decimal(mean(ratios_affine))),
        ("MAX_APG_TO_AFFINE_WIDTH_RATIO", decimal(maximum(ratios_affine))),
        ("MIN_APG_TO_SHARED_WIDTH_RATIO", decimal(minimum(ratios_shared))),
        ("MEDIAN_APG_TO_SHARED_WIDTH_RATIO", decimal(median(ratios_shared))),
        ("MEAN_APG_TO_SHARED_WIDTH_RATIO", decimal(mean(ratios_shared))),
        ("MAX_APG_TO_SHARED_WIDTH_RATIO", decimal(maximum(ratios_shared))),
        (
            "MINIMUM_APG_PIVOT_MARGIN",
            decimal(
                minimum(
                    [float(row["MINIMUM_APG_PIVOT_MARGIN"]) for row in paired_valid]
                )
            ),
        ),
        (
            "TOTAL_WORKER_ELAPSED_MS",
            str(sum(canonical_uint(row["ELAPSED_MS"], "elapsed") for row in rows)),
        ),
        (
            "MEAN_VALID_WORKER_ELAPSED_MS",
            decimal(mean([float(row["ELAPSED_MS"]) for row in valid])),
        ),
        (
            "MEAN_RECEIPT_BYTES",
            decimal(mean([float(row["RECEIPT_BYTES"]) for row in paired_valid])),
        ),
        (
            "LEAF_VERIFICATION_COUNT",
            str(sum(row["VERIFICATION_SHA256"] != ZERO_SHA256 for row in rows)),
        ),
        ("MUTATION_AUDITED_LEAF_COUNT", str(len(audit_rows))),
        ("MUTATION_SUITE_SIZE_PER_LEAF", str(suite_size)),
        ("MUTATION_TESTS", str(mutation_tests)),
        ("MUTATIONS_REJECTED", str(mutations_rejected)),
        ("H_APG_CS6_FULL53_SUPPORTED", str(full53_supported).lower()),
    )
    boundary = (
        ("COMMON_SOURCE_SYMBOLS_PRESERVED", "true"),
        ("PRIMARY_INTERVAL_INTERSECTED_WITH_CONTROLS", "false"),
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
    require_fields(summary, (*expected, *boundary), "summary")
    if set(summary) != {key for key, _ in (*expected, *boundary)}:
        fail("summary field set mismatch")
    return dict((*expected, *boundary))


def verify_retained(root: Path, python: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        fail("retained root must be a regular directory")
    repo = Path(__file__).resolve().parents[2]
    index = parse_index(root)
    retained = parse_kv(root / "retained-manifest.txt", "retained manifest")
    retained_keys = {
        "SCHEMA", "FILES_INDEX_SHA256", "FILE_COUNT", "RUN_MANIFEST_SHA256",
        "RETAINED_VERIFIER_SHA256", "EVIDENCE_CLASS", "COORDINATE_COUNT",
        "CONTRACT_FROZEN_IN_COMMIT", "FROZEN_MANIFEST_SHA256",
        "IMPLEMENTATION_COMMIT", "PAIRED_VALID_COUNT",
        "ROOT_INTERVAL_DOMAIN_CLASS_MATCH", "NEW_UNRESOLVED_COUNT",
        "COMPUTATION_UNRESOLVED_COUNT",
        "PARENT_AFFINE_OBLIGATION_COUNT", "PARENT_AFFINE_LOSS_COUNT",
        "MUTATION_AUDITED_LEAF_COUNT", "MUTATION_SUITE_SIZE_PER_LEAF",
        "MUTATION_TESTS",
        "MUTATIONS_REJECTED", "H_APG_CS6_FULL53_SUPPORTED",
        "PROMOTION_ELIGIBLE",
    }
    if set(retained) != retained_keys:
        fail("retained manifest field set mismatch")
    require_fields(
        retained,
        (
            (
                "SCHEMA",
                "sounio.cs6.affine-projective-cocycle-full53-retained.v1",
            ),
            ("FILES_INDEX_SHA256", digest(root / "files.sha256")),
            ("FILE_COUNT", str(len(index))),
            ("RETAINED_VERIFIER_SHA256", digest(Path(__file__).resolve())),
            ("EVIDENCE_CLASS", "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION"),
            ("CONTRACT_FROZEN_IN_COMMIT", CONTRACT_FROZEN_IN_COMMIT),
            ("FROZEN_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
            ("PROMOTION_ELIGIBLE", "false"),
        ),
        "retained manifest",
    )

    run_manifest = parse_kv(root / "run-manifest.txt", "run manifest")
    run_manifest_keys = {
        "SCHEMA", "RUN_COMPLETE", "CAPD_VERSION", "INTERVAL_BACKEND",
        "OPTIMIZATION_LEVEL", "ROOT_CHALLENGE", "SOURCE_SHA256",
        "VERIFIER_SHA256", "RUNNER_SHA256", "COORDINATE_MANIFEST_SHA256",
        "PREDECLARED_IN_COMMIT", "PREDECLARATION_REPORT_SHA256",
        "CONTRACT_FROZEN_IN_COMMIT", "FROZEN_MANIFEST_SHA256",
        "IMPLEMENTATION_COMMIT", "PARENT_RUN_MANIFEST_SHA256",
        "PARENT_FILES_INDEX_SHA256", "PARENT_COORDINATES_SHA256",
        "PARENT_LEAVES_SHA256", "PARENT_AFFINE_OBLIGATION_SHA256",
        "LEAF_COUNT", "PAIRED_VALID_COUNT", "ROOT_INTERVAL_DOMAIN_CLASS_MATCH",
        "NEW_UNRESOLVED_COUNT", "PARENT_AFFINE_OBLIGATION_COUNT",
        "PARENT_AFFINE_LOSS_COUNT", "MUTATION_AUDITED_LEAF_COUNT",
        "MUTATION_SUITE_SIZE_PER_LEAF", "MUTATION_TESTS",
        "MUTATIONS_REJECTED", "H_APG_CS6_FULL53_SUPPORTED",
        "PROMOTION_ELIGIBLE",
    }
    run_manifest_keys.update(
        relative.upper().replace("-", "_").replace(".", "_") + "_SHA256"
        for relative in RUN_ARTIFACTS
    )
    if set(run_manifest) != run_manifest_keys:
        fail("run manifest field set mismatch")
    require_fields(
        run_manifest,
        (
            (
                "SCHEMA",
                "sounio.cs6.affine-projective-cocycle-full53-run-manifest.v1",
            ),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", "5.3.0"),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("PREDECLARED_IN_COMMIT", PREDECLARED_COMMIT),
            ("PREDECLARATION_REPORT_SHA256", PREDECLARATION_REPORT_SHA256),
            ("CONTRACT_FROZEN_IN_COMMIT", CONTRACT_FROZEN_IN_COMMIT),
            ("FROZEN_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
            ("PARENT_RUN_MANIFEST_SHA256", PARENT_RUN_MANIFEST_SHA256),
            ("PARENT_FILES_INDEX_SHA256", PARENT_FILES_INDEX_SHA256),
            ("PARENT_COORDINATES_SHA256", PARENT_COORDINATES_SHA256),
            ("PARENT_LEAVES_SHA256", PARENT_LEAVES_SHA256),
            ("PARENT_AFFINE_OBLIGATION_SHA256", PARENT_AFFINE_OBLIGATION_SHA256),
            ("LEAF_COUNT", "53"),
            ("PARENT_AFFINE_OBLIGATION_COUNT", "28"),
            ("PROMOTION_ELIGIBLE", "false"),
        ),
        "run manifest",
    )
    if SHA_RE.fullmatch(run_manifest.get("ROOT_CHALLENGE", "")) is None:
        fail("run manifest root challenge is not a canonical SHA-256")
    implementation_commit = run_manifest.get("IMPLEMENTATION_COMMIT", "")
    if COMMIT_RE.fullmatch(implementation_commit) is None:
        fail("run manifest implementation commit is malformed")
    if digest(root / "run-manifest.txt") != retained["RUN_MANIFEST_SHA256"]:
        fail("retained manifest does not bind run manifest")
    for key, relative in (
        ("SOURCE_SHA256", "worker-source.cpp"),
        ("VERIFIER_SHA256", "leaf-verifier.py"),
        ("RUNNER_SHA256", "runner.py"),
        ("COORDINATE_MANIFEST_SHA256", "coordinates.tsv"),
    ):
        if run_manifest[key] != digest(root / relative):
            fail(f"run manifest source binding mismatch: {key}")
    for relative in RUN_ARTIFACTS:
        key = relative.upper().replace("-", "_").replace(".", "_") + "_SHA256"
        if run_manifest[key] != digest(root / relative):
            fail(f"run manifest artifact binding mismatch: {key}")

    committed_inputs = (
        ("worker-source.cpp", "scripts/research/cs6_affine_projective_cocycle_full53_probe.cpp"),
        ("leaf-verifier.py", "scripts/research/cs6_affine_projective_cocycle_full53_verify.py"),
        ("runner.py", "scripts/research/cs6_affine_projective_cocycle_full53_run.py"),
        (
            "coordinates.tsv",
            "scripts/research/cs6_affine_projective_cocycle_full53_coordinates_v1.tsv",
        ),
    )
    for bundled, relative in committed_inputs:
        if stable_bytes(root / bundled, bundled) != git_blob(
            repo, implementation_commit, relative, relative
        ):
            fail(f"bundled run input differs from implementation commit: {bundled}")
    if stable_bytes(root / "git-head.txt", "git head") != (
        implementation_commit + "\n"
    ).encode("ascii"):
        fail("git-head artifact differs from implementation commit")
    if stable_bytes(root / "git-status.txt", "git status"):
        fail("evidence run did not start from a clean worktree")

    executed = stable_bytes(root / "executed-worker.sha256", "executed worker digest")
    try:
        executed_fields = executed.decode("ascii").removesuffix("\n").split("  ")
    except UnicodeError as error:
        raise RetainedError("executed worker digest must be ASCII") from error
    if (
        not executed.endswith(b"\n")
        or len(executed_fields) != 2
        or SHA_RE.fullmatch(executed_fields[0]) is None
        or executed_fields[1] != "worker-binary"
    ):
        fail("executed worker digest row is malformed")

    run_contract = parse_kv(root / "run-contract.txt", "run contract")
    contract_keys = {
        "SCHEMA", "SOURCE", "ROOT_CHALLENGE", "COORDINATE_MANIFEST_SHA256",
        "PREDECLARED_IN_COMMIT", "PREDECLARATION_REPORT_SHA256",
        "CONTRACT_FROZEN_IN_COMMIT", "FROZEN_MANIFEST_SHA256",
        "IMPLEMENTATION_COMMIT", "PARENT_RUN_MANIFEST_SHA256",
        "PARENT_FILES_INDEX_SHA256", "PARENT_COORDINATES_SHA256",
        "PARENT_LEAVES_SHA256", "PARENT_AFFINE_OBLIGATION_SHA256",
        "EXPECTED_LEAF_COUNT", "EXPECTED_PAIRED_VALID_COUNT",
        "PARENT_AFFINE_OBLIGATION_COUNT", "EXPECTED_PARENT_NONAFFINE_COUNT",
        "ROOT_POLICY", "AFFINE_LOSS_POLICY", "MUTATION_AUDIT_POLICY",
        "NEGATIVE_SCIENTIFIC_OUTCOMES_RETAINED", "JOBS", "TIMEOUT_SECONDS",
        "DISCRETE_POINCARE_COCYCLE", "COMMON_SOURCE_SYMBOLS_PRESERVED",
        "PRIMARY_INTERVAL_INTERSECTED_WITH_CONTROLS",
        "FULL_SOURCE_CARRIER_PROVED",
    }
    if set(run_contract) != contract_keys:
        fail("run contract field set mismatch")
    require_fields(
        run_contract,
        (
            (
                "SCHEMA",
                "sounio.cs6.affine-projective-cocycle-full53-run-contract.v1",
            ),
            ("SOURCE", "N0"),
            ("ROOT_CHALLENGE", run_manifest["ROOT_CHALLENGE"]),
            ("COORDINATE_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
            ("PREDECLARED_IN_COMMIT", PREDECLARED_COMMIT),
            ("PREDECLARATION_REPORT_SHA256", PREDECLARATION_REPORT_SHA256),
            ("CONTRACT_FROZEN_IN_COMMIT", CONTRACT_FROZEN_IN_COMMIT),
            ("FROZEN_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
            ("IMPLEMENTATION_COMMIT", implementation_commit),
            ("PARENT_RUN_MANIFEST_SHA256", PARENT_RUN_MANIFEST_SHA256),
            ("PARENT_FILES_INDEX_SHA256", PARENT_FILES_INDEX_SHA256),
            ("PARENT_COORDINATES_SHA256", PARENT_COORDINATES_SHA256),
            ("PARENT_LEAVES_SHA256", PARENT_LEAVES_SHA256),
            ("PARENT_AFFINE_OBLIGATION_SHA256", PARENT_AFFINE_OBLIGATION_SHA256),
            ("EXPECTED_LEAF_COUNT", "53"),
            ("EXPECTED_PAIRED_VALID_COUNT", "52"),
            ("PARENT_AFFINE_OBLIGATION_COUNT", "28"),
            ("EXPECTED_PARENT_NONAFFINE_COUNT", "24"),
            ("ROOT_POLICY", "ATTEMPT_AND_REQUIRE_SAME_UNRESOLVED_CLASS"),
            (
                "AFFINE_LOSS_POLICY",
                "ANY_CHILD_UNRESOLVED_INVALID_OR_UNCERTIFIED_ON_PARENT_AFFINE_PASS_LEAF",
            ),
            (
                "MUTATION_AUDIT_POLICY",
                "EVERY_PAIRED_VALID_LEAF_ALL_MUTATIONS_REJECTED",
            ),
            ("NEGATIVE_SCIENTIFIC_OUTCOMES_RETAINED", "true"),
            ("DISCRETE_POINCARE_COCYCLE", "true"),
            ("COMMON_SOURCE_SYMBOLS_PRESERVED", "true"),
            ("PRIMARY_INTERVAL_INTERSECTED_WITH_CONTROLS", "false"),
            ("FULL_SOURCE_CARRIER_PROVED", "false"),
        ),
        "run contract",
    )
    jobs = canonical_uint(run_contract["JOBS"], "run jobs")
    timeout = canonical_uint(run_contract["TIMEOUT_SECONDS"], "run timeout")
    if not 1 <= jobs <= 32 or not 1 <= timeout <= 3600:
        fail("run contract resource bounds are invalid")

    coordinates = parse_coordinates(root / "coordinates.tsv", repo)
    affine_ids = verify_parent_contract(repo, coordinates, python)
    rows = parse_tsv(root / "leaves.tsv", LEAF_COLUMNS, "leaves table")
    if len(rows) != 53 or [row["LEAF_ID"] for row in rows] != list(coordinates):
        fail("leaves table differs from the frozen coordinate population")
    source_sha = run_manifest["SOURCE_SHA256"]
    root_challenge = run_manifest["ROOT_CHALLENGE"]
    verified_rows: list[dict[str, str]] = []
    unresolved_statuses = {
        "COMPUTATION_UNRESOLVED_TIMEOUT",
        "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN",
        "COMPUTATION_UNRESOLVED_FROZEN_CHART",
        "COMPUTATION_UNRESOLVED_UNEXPECTED_WORKER_FAILURE",
    }
    boolean_columns = (
        "PARENT_AFFINE_PASS", "PROBE_PASS", "AFFINE_PASS",
        "BOXED_HOMOGENEOUS_PASS", "APG_VALID", "APG_PASS", "APG_RESCUE",
        "APG_NARROWER_BOXED", "APG_NARROWER_AFFINE",
        "APG_NARROWER_SHARED", "CERTIFICATE_PASS", "SUBDIVISION_REQUIRED",
    )
    table_to_verification = (
        ("PROBE_PASS", "PROBE_PASS"),
        ("AFFINE_PASS", "AFFINE_CERTIFICATE_PASS"),
        ("BOXED_HOMOGENEOUS_PASS", "HOMOGENEOUS_CERTIFICATE_PASS"),
        ("APG_VALID", "APG_COMPUTATION_VALID"),
        ("APG_PASS", "APG_CERTIFICATE_PASS"),
        ("APG_RESCUE", "APG_RESCUE"),
        ("APG_NARROWER_BOXED", "APG_STRICTLY_NARROWER_THAN_BOXED"),
        ("APG_NARROWER_AFFINE", "APG_STRICTLY_NARROWER_THAN_AFFINE"),
        ("APG_NARROWER_SHARED", "APG_STRICTLY_NARROWER_THAN_SHARED"),
        ("CERTIFICATE_PASS", "CERTIFICATE_PASS"),
        ("SUBDIVISION_REQUIRED", "SUBDIVISION_REQUIRED"),
    )

    def replay_leaf(row: dict[str, str]) -> tuple[str, bytes, dict[str, str]]:
        identity = row["LEAF_ID"]
        process = subprocess.run(
            verifier_command(
                python, root, repo, {**row, "ROOT_CHALLENGE": root_challenge},
                source_sha, mutations=False
            ),
            capture_output=True,
        )
        if process.returncode != 0 or process.stderr:
            fail(f"retained leaf verifier failed: {identity}")
        values = parse_verification(process.stdout, f"verification replay {identity}")
        return identity, process.stdout, values

    replay_futures: dict[str, tuple[bytes, dict[str, str]]] = {}
    for row in rows:
        identity = row["LEAF_ID"]
        coordinate = coordinates[identity]
        if tuple(
            canonical_uint(row[key], f"{identity} {key}")
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        ) != (
            coordinate.u_depth,
            coordinate.u_index,
            coordinate.s_depth,
            coordinate.s_index,
        ):
            fail(f"leaf coordinate mismatch: {identity}")
        if (
            row["INPUT_SHA256"] != coordinate.input_sha
            or row["PARENT_STATUS"] != coordinate.parent_status
            or row["PARENT_RECEIPT_SHA256"] != coordinate.parent_receipt_sha
            or parse_boolean(
                row["PARENT_AFFINE_PASS"], f"{identity} parent affine"
            )
            != (identity in affine_ids)
        ):
            fail(f"leaf parent binding mismatch: {identity}")
        chart_fields: list[str] = []
        for prefix, (chart, sign) in zip(
            ("E1_R0", "E1_R1", "E2_R0", "E2_R1"),
            coordinate.chart_signs,
            strict=True,
        ):
            chart_fields.extend((row[f"{prefix}_CHART"], row[f"{prefix}_SIGN"]))
            if chart_fields[-2:] != [chart, str(sign)]:
                fail(f"leaf chart/sign mismatch: {identity} {prefix}")
        for key in boolean_columns:
            parse_boolean(row[key], f"{identity} {key}")
        for key in (
            "INPUT_SHA256", "LEAF_CHALLENGE", "RECEIPT_SHA256",
            "STDERR_SHA256", "VERIFICATION_SHA256", "PHYSICAL_SHA256",
        ):
            if SHA_RE.fullmatch(row[key]) is None:
                fail(f"leaf digest grammar mismatch: {identity} {key}")
        worker_rc = canonical_int(row["WORKER_RC"], f"{identity} worker rc")
        canonical_uint(row["ELAPSED_MS"], f"{identity} elapsed")
        input_path = root / "inputs" / f"{identity}.txt"
        receipt_path = root / "receipts" / f"{identity}.txt"
        stderr_path = root / "stderr" / f"{identity}.txt"
        if stable_bytes(input_path, f"input {identity}") != leaf_input_bytes(
            coordinate.u_depth,
            coordinate.u_index,
            coordinate.s_depth,
            coordinate.s_index,
        ):
            fail(f"noncanonical retained input: {identity}")
        expected_challenge = challenge(
            root_challenge, FROZEN_MANIFEST_SHA256, coordinate
        )
        if row["LEAF_CHALLENGE"] != expected_challenge:
            fail(f"leaf challenge mismatch: {identity}")
        stderr_raw = stable_bytes(stderr_path, f"stderr {identity}")
        if (
            digest(receipt_path) != row["RECEIPT_SHA256"]
            or digest_bytes(stderr_raw) != row["STDERR_SHA256"]
        ):
            fail(f"leaf payload digest mismatch: {identity}")
        has_verification = row["VERIFICATION_SHA256"] != ZERO_SHA256
        if not has_verification:
            false_columns = (
                "PROBE_PASS", "AFFINE_PASS", "BOXED_HOMOGENEOUS_PASS",
                "APG_VALID", "APG_PASS", "APG_RESCUE", "APG_NARROWER_BOXED",
                "APG_NARROWER_AFFINE", "APG_NARROWER_SHARED",
                "CERTIFICATE_PASS",
            )
            root_computed = (
                identity == ROOT_LEAF_ID
                and row["STATUS"] == "ROOT_POLICY_VIOLATION_COMPUTED"
                and worker_rc == 0
            )
            classified_unresolved = (
                row["STATUS"] in unresolved_statuses
                and worker_rc != 0
                and unresolved_status_matches(row["STATUS"], worker_rc, stderr_raw)
            )
            if (
                not (root_computed or classified_unresolved)
                or row["METHOD"] != "NONE"
                or any(parse_boolean(row[key], key) for key in false_columns)
                or not parse_boolean(
                    row["SUBDIVISION_REQUIRED"], "SUBDIVISION_REQUIRED"
                )
                or row["PHYSICAL_SHA256"] != ZERO_SHA256
                or tuple(row[key] for key in METRIC_COLUMNS) != ("-",) * 7
            ):
                fail(f"unresolved leaf row is inconsistent: {identity}")
            continue
        if identity == ROOT_LEAF_ID:
            fail("root unexpectedly has a verifier receipt")
        if (
            worker_rc != 0
            or stderr_raw
            or not stable_bytes(receipt_path, f"receipt {identity}")
        ):
            fail(f"computed leaf payload is inconsistent: {identity}")
        verified_rows.append(row)

    root_row = rows[0]
    root_stderr = stable_bytes(
        root / "stderr" / f"{ROOT_LEAF_ID}.txt", "root stderr"
    )
    root_class_match = (
        root_row["LEAF_ID"] == ROOT_LEAF_ID
        and root_row["STATUS"] == "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN"
        and root_row["WORKER_RC"] == "1"
        and root_row["RECEIPT_SHA256"] == EMPTY_SHA256
        and not parse_boolean(root_row["PROBE_PASS"], "root probe")
        and canonical_interval_domain_failure(root_stderr)
    )

    expected_index = set(RUN_ARTIFACTS) | {"run-manifest.txt"}
    for identity in coordinates:
        expected_index.update(
            {
                f"inputs/{identity}.txt",
                f"receipts/{identity}.txt",
                f"stderr/{identity}.txt",
            }
        )
    expected_index.update(
        f"verifications/{row['LEAF_ID']}.txt" for row in verified_rows
    )
    if set(index) != expected_index:
        fail("files index path schema mismatch")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(replay_leaf, row) for row in verified_rows]
        for future in concurrent.futures.as_completed(futures):
            identity, replay_raw, verification = future.result()
            replay_futures[identity] = (replay_raw, verification)
    for row in verified_rows:
        identity = row["LEAF_ID"]
        replay_raw, verification = replay_futures[identity]
        verification_path = root / "verifications" / f"{identity}.txt"
        if replay_raw != stable_bytes(
            verification_path, f"verification {identity}"
        ):
            fail(f"retained verifier output drift: {identity}")
        if digest(verification_path) != row["VERIFICATION_SHA256"]:
            fail(f"verification digest mismatch: {identity}")
        require_fields(
            verification,
            (
                (
                    "VERIFICATION_SCHEMA",
                    "sounio.cs6.affine-projective-cocycle-full53-leaf-verification.v1",
                ),
                ("RECEIPT_SHA256", row["RECEIPT_SHA256"]),
                ("PHYSICAL_SHA256", row["PHYSICAL_SHA256"]),
                ("COORDINATE_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
                ("PARENT_RECEIPT_SHA256", row["PARENT_RECEIPT_SHA256"]),
                ("LEAF_CHALLENGE", row["LEAF_CHALLENGE"]),
                ("MUTATION_TESTS", "0"),
                ("MUTATIONS_REJECTED", "0"),
                ("LEAF_METHOD", row["METHOD"]),
            ),
            f"verification {identity}",
        )
        for table_key, verification_key in table_to_verification:
            if row[table_key] != verification[verification_key]:
                fail(f"leaf table verifier mismatch: {identity} {table_key}")
        certificate = parse_boolean(row["CERTIFICATE_PASS"], "certificate")
        probe = parse_boolean(row["PROBE_PASS"], "probe")
        expected_status = (
            "PROBE_VALID_CERTIFIED"
            if certificate
            else "PROBE_VALID_UNCERTIFIED"
            if probe
            else "PROBE_VERIFIED_STRUCTURALLY_INVALID"
        )
        if (
            row["STATUS"] != expected_status
            or certificate
            == parse_boolean(row["SUBDIVISION_REQUIRED"], "subdivision")
        ):
            fail(f"computed leaf status mismatch: {identity}")
        expected_metrics = receipt_metrics(
            stable_bytes(root / "receipts" / f"{identity}.txt", f"receipt {identity}")
        )
        if tuple(row[key] for key in METRIC_COLUMNS) != expected_metrics:
            fail(f"computed leaf metrics mismatch: {identity}")

    audit_rows = parse_tsv(
        root / "mutation-audits.tsv", MUTATION_COLUMNS, "mutation audits"
    )
    if [row["LEAF_ID"] for row in audit_rows] != [
        row["LEAF_ID"] for row in verified_rows
    ]:
        fail("mutation audit population differs from verified paired leaves")
    for audit_row, leaf_row in zip(audit_rows, verified_rows, strict=True):
        if (
            audit_row["RECEIPT_SHA256"] != leaf_row["RECEIPT_SHA256"]
            or audit_row["VERIFICATION_SHA256"] != leaf_row["VERIFICATION_SHA256"]
            or SHA_RE.fullmatch(audit_row["AUDIT_STDOUT_SHA256"]) is None
            or canonical_uint(audit_row["MUTATION_TESTS"], "mutation tests")
            != MUTATIONS_PER_LEAF
            or canonical_uint(
                audit_row["MUTATIONS_REJECTED"], "mutations rejected"
            )
            != MUTATIONS_PER_LEAF
        ):
            fail(f"mutation audit row mismatch: {leaf_row['LEAF_ID']}")

    def replay_mutations(
        audit_row: dict[str, str], leaf_row: dict[str, str]
    ) -> tuple[str, bytes]:
        identity = leaf_row["LEAF_ID"]
        process = subprocess.run(
            verifier_command(
                python,
                root,
                repo,
                {**leaf_row, "ROOT_CHALLENGE": root_challenge},
                source_sha,
                mutations=True,
            ),
            capture_output=True,
        )
        if process.returncode != 0 or process.stderr:
            fail(f"mutation audit replay failed: {identity}")
        values = parse_verification(process.stdout, f"mutation replay {identity}")
        require_fields(
            values,
            (
                ("RECEIPT_SHA256", leaf_row["RECEIPT_SHA256"]),
                ("COORDINATE_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
                ("PARENT_RECEIPT_SHA256", leaf_row["PARENT_RECEIPT_SHA256"]),
                ("LEAF_CHALLENGE", leaf_row["LEAF_CHALLENGE"]),
                ("MUTATION_TESTS", str(MUTATIONS_PER_LEAF)),
                ("MUTATIONS_REJECTED", str(MUTATIONS_PER_LEAF)),
            ),
            f"mutation replay {identity}",
        )
        if digest_bytes(process.stdout) != audit_row["AUDIT_STDOUT_SHA256"]:
            fail(f"mutation audit output digest mismatch: {identity}")
        return identity, process.stdout

    audit_outputs: dict[str, bytes] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(replay_mutations, audit_row, leaf_row)
            for audit_row, leaf_row in zip(audit_rows, verified_rows, strict=True)
        ]
        for future in concurrent.futures.as_completed(futures):
            identity, raw = future.result()
            audit_outputs[identity] = raw
    if stable_bytes(root / "mutation-audit-stderr.txt", "mutation audit stderr"):
        fail("retained mutation audit emitted stderr")
    first_audit = audit_outputs[verified_rows[0]["LEAF_ID"]] if verified_rows else b""
    if stable_bytes(root / "mutation-audit.txt", "mutation audit") != first_audit:
        fail("retained first mutation audit does not replay")

    summary = verify_summary(
        parse_kv(root / "summary.txt", "summary"),
        rows,
        audit_rows,
        root_class_match,
    )
    run_summary_bindings = (
        ("LEAF_COUNT", "COORDINATE_COUNT"),
        ("PAIRED_VALID_COUNT", "PAIRED_VALID_COUNT"),
        ("ROOT_INTERVAL_DOMAIN_CLASS_MATCH", "ROOT_INTERVAL_DOMAIN_CLASS_MATCH"),
        ("NEW_UNRESOLVED_COUNT", "NEW_UNRESOLVED_COUNT"),
        ("PARENT_AFFINE_OBLIGATION_COUNT", "PARENT_AFFINE_OBLIGATION_COUNT"),
        ("PARENT_AFFINE_LOSS_COUNT", "PARENT_AFFINE_LOSS_COUNT"),
        ("MUTATION_AUDITED_LEAF_COUNT", "MUTATION_AUDITED_LEAF_COUNT"),
        ("MUTATION_SUITE_SIZE_PER_LEAF", "MUTATION_SUITE_SIZE_PER_LEAF"),
        ("MUTATION_TESTS", "MUTATION_TESTS"),
        ("MUTATIONS_REJECTED", "MUTATIONS_REJECTED"),
        ("H_APG_CS6_FULL53_SUPPORTED", "H_APG_CS6_FULL53_SUPPORTED"),
        ("PROMOTION_ELIGIBLE", "PROMOTION_ELIGIBLE"),
    )
    for run_key, summary_key in run_summary_bindings:
        if run_manifest[run_key] != summary[summary_key]:
            fail(f"run-manifest summary binding mismatch: {run_key}")
    retained_summary_bindings = (
        ("COORDINATE_COUNT", "COORDINATE_COUNT"),
        ("PAIRED_VALID_COUNT", "PAIRED_VALID_COUNT"),
        ("ROOT_INTERVAL_DOMAIN_CLASS_MATCH", "ROOT_INTERVAL_DOMAIN_CLASS_MATCH"),
        ("NEW_UNRESOLVED_COUNT", "NEW_UNRESOLVED_COUNT"),
        ("COMPUTATION_UNRESOLVED_COUNT", "COMPUTATION_UNRESOLVED_COUNT"),
        ("PARENT_AFFINE_OBLIGATION_COUNT", "PARENT_AFFINE_OBLIGATION_COUNT"),
        ("PARENT_AFFINE_LOSS_COUNT", "PARENT_AFFINE_LOSS_COUNT"),
        ("MUTATION_AUDITED_LEAF_COUNT", "MUTATION_AUDITED_LEAF_COUNT"),
        ("MUTATION_SUITE_SIZE_PER_LEAF", "MUTATION_SUITE_SIZE_PER_LEAF"),
        ("MUTATION_TESTS", "MUTATION_TESTS"),
        ("MUTATIONS_REJECTED", "MUTATIONS_REJECTED"),
        ("H_APG_CS6_FULL53_SUPPORTED", "H_APG_CS6_FULL53_SUPPORTED"),
        ("PROMOTION_ELIGIBLE", "PROMOTION_ELIGIBLE"),
    )
    for retained_key, summary_key in retained_summary_bindings:
        if retained[retained_key] != summary[summary_key]:
            fail(f"retained-manifest summary binding mismatch: {retained_key}")
    if retained["IMPLEMENTATION_COMMIT"] != implementation_commit:
        fail("retained-manifest implementation commit mismatch")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("retained_dir", type=Path)
    args = parser.parse_args(argv)
    summary = verify_retained(
        args.retained_dir.resolve(), Path(sys.executable).resolve()
    )
    print(
        "RETAINED_SCHEMA="
        "sounio.cs6.affine-projective-cocycle-full53-retained.v1"
    )
    print(f"COORDINATE_COUNT={summary['COORDINATE_COUNT']}")
    print(f"PAIRED_VALID_COUNT={summary['PAIRED_VALID_COUNT']}")
    print("LEAF_REPLAY_MATCH=true")
    print(
        "MUTATION_REPLAY_MATCH="
        f"{summary['MUTATIONS_REJECTED']}/{summary['MUTATION_TESTS']}"
    )
    print("FILES_INDEX_MATCH=true")
    print(
        "H_APG_CS6_FULL53_SUPPORTED="
        f"{summary['H_APG_CS6_FULL53_SUPPORTED']}"
    )
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RetainedError as error:
        print(f"retained verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
