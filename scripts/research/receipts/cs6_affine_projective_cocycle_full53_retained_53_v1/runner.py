#!/usr/bin/env python3
"""Run the predeclared 53-leaf CS6 affine-projective cocycle experiment."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CHALLENGE_DOMAIN = (
    b"sounio.cs6.affine-projective-cocycle-full53-leaf-challenge.v1\0"
)
ZERO_SHA256 = "0" * 64
PARENT_RUN = "cs6_plucker_cocycle_retained_53_v1"
PREDECLARED_COMMIT = "9dcf1fca964d7e54e1109f9210689809666b2a54"
PREDECLARATION_REPORT = (
    "docs/research/cs6_affine_projective_cocycle_2026-08-01.md"
)
PREDECLARATION_REPORT_SHA256 = (
    "74d8f596c9258eab49775192e8b244e032dfb5e3481a23804075c66ff316618c"
)
PARENT_RUN_MANIFEST_SHA256 = "21f0ca224fc79d1cfc815f203100e091bf27dbb070cb3a2c69f6056b7a6fee27"
PARENT_FILES_INDEX_SHA256 = "740ae39070ff90604048c37d25e2f0de954be9ea55e020a529e094699a82802d"
PARENT_COORDINATES_SHA256 = "6169dd7705ca4f01180f65d13d620320845684f0a9fc28411c881cfae7e4f2d8"
PARENT_LEAVES_SHA256 = "98c639a616ac640f1983209530f8fe30d769a0e4b0136665201c9bc57320e37f"
PARENT_AFFINE_OBLIGATION_SHA256 = (
    "9f36931e672aba1b9735c45eef286fbca97da67b911c64daa0b3b8c8affecd6c"
)
CONTRACT_FROZEN_IN_COMMIT = "58905019754cf66f077a0db228f1a99a4a7612eb"
FROZEN_MANIFEST_SHA256 = (
    "61b2b0649983a332b5abb530443a3ff14a19e62514ef9b1d3175d8e9a6bbfd9c"
)
ROOT_LEAF_ID = "U00-0000000000_S00-0000000000"
MUTATIONS_PER_LEAF = 112
MUTATION_TOTAL = 52 * MUTATIONS_PER_LEAF
CHALLENGE_SELFTEST_LEAF_ID = "U08-0000000064_S08-0000000064"
CHALLENGE_SELFTEST_EXPECTED = (
    "b154751d6f07c5a63b0d16adcd4f3a07e8f55555aa35216f732183df254f3869"
)
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
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "COORDINATE_MANIFEST_SHA256",
    "PARENT_RECEIPT_SHA256",
    "LEAF_CHALLENGE",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "LEAF_METHOD",
    "PROBE_PASS",
    "AFFINE_CERTIFICATE_PASS",
    "PROJECTIVE_X_CERTIFICATE_PASS",
    "PROJECTIVE_Y_CERTIFICATE_PASS",
    "PROJECTIVE_PLUS_CERTIFICATE_PASS",
    "PROJECTIVE_MINUS_CERTIFICATE_PASS",
    "HOMOGENEOUS_CERTIFICATE_PASS",
    "APG_COMPUTATION_VALID",
    "APG_CERTIFICATE_PASS",
    "APG_RESCUE",
    "APG_STRICTLY_NARROWER_THAN_BOXED",
    "APG_STRICTLY_NARROWER_THAN_AFFINE",
    "APG_STRICTLY_NARROWER_THAN_SHARED",
    "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def canonical_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    path.write_bytes(
        "".join(f"{key}={value}\n" for key, value in fields).encode("ascii")
    )


def parse_kv_bytes(raw: bytes, expected_keys: Sequence[str]) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError("non-ASCII verifier output") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise RuntimeError("noncanonical verifier output")
    lines = text.splitlines()
    if len(lines) != len(expected_keys):
        raise RuntimeError("verifier output line count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, expected_keys, strict=True):
        if line.count("=") != 1:
            raise RuntimeError("malformed verifier output")
        key, value = line.split("=", 1)
        if key != expected or not value:
            raise RuntimeError(f"verifier output key mismatch: {expected}")
        result[key] = value
    return result


def leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def leaf_challenge(root: str, manifest_sha: str, leaf: "Leaf") -> str:
    chart_sign_fields: list[str] = []
    for chart, sign in leaf.chart_sign_pairs:
        chart_sign_fields.extend((chart, str(sign)))
    chart_sign_tuple = ("\t".join(chart_sign_fields) + "\n").encode("ascii")
    return digest_bytes(
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + leaf.identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(leaf.input_sha)
        + b"\0"
        + bytes.fromhex(leaf.parent_receipt_sha)
        + b"\0"
        + chart_sign_tuple
        + b"\0"
        + bytes.fromhex(manifest_sha)
    )


@dataclass(frozen=True, order=True)
class Leaf:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    input_sha: str
    parent_status: str
    parent_receipt_sha: str
    e1_r0_chart: str
    e1_r0_sign: int
    e1_r1_chart: str
    e1_r1_sign: int
    e2_r0_chart: str
    e2_r0_sign: int
    e2_r1_chart: str
    e2_r1_sign: int
    parent_affine_pass: bool

    @property
    def identity(self) -> str:
        return leaf_id(self.u_depth, self.u_index, self.s_depth, self.s_index)

    @property
    def chart_sign_pairs(self) -> tuple[tuple[str, int], ...]:
        return (
            (self.e1_r0_chart, self.e1_r0_sign),
            (self.e1_r1_chart, self.e1_r1_sign),
            (self.e2_r0_chart, self.e2_r0_sign),
            (self.e2_r1_chart, self.e2_r1_sign),
        )


@dataclass(frozen=True)
class Metrics:
    c1_width: float
    affine_width: float
    boxed_width: float
    shared_width: float
    apg_width: float
    minimum_apg_pivot_margin: float
    receipt_bytes: int


@dataclass(frozen=True)
class LeafResult:
    leaf: Leaf
    status: str
    method: str
    probe_pass: bool
    affine: bool
    projective_x: bool
    projective_y: bool
    projective_plus: bool
    projective_minus: bool
    homogeneous: bool
    apg_valid: bool
    apg: bool
    rescue: bool
    narrower_boxed: bool
    narrower_affine: bool
    narrower_shared: bool
    certificate: bool
    subdivision: bool
    challenge: str
    receipt_sha: str
    verification_sha: str
    physical_sha: str
    worker_rc: int
    elapsed_ms: int
    metrics: Metrics | None


def parse_coordinate_manifest(path: Path, repo: Path) -> list[Leaf]:
    raw = path.read_bytes()
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError("coordinate manifest must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw or len(lines) != 75:
        raise RuntimeError("coordinate manifest line count or terminator mismatch")
    if tuple(lines[: len(MANIFEST_HEADER)]) != MANIFEST_HEADER:
        raise RuntimeError("coordinate manifest header mismatch")
    manifest_sha = digest_bytes(raw)
    if manifest_sha != FROZEN_MANIFEST_SHA256:
        raise RuntimeError("coordinate manifest differs from the frozen contract")

    frozen_manifest = subprocess.run(
        [
            "git",
            "-C",
            repo,
            "show",
            f"{CONTRACT_FROZEN_IN_COMMIT}:scripts/research/"
            "cs6_affine_projective_cocycle_full53_coordinates_v1.tsv",
        ],
        check=True,
        capture_output=True,
    ).stdout
    if frozen_manifest != raw:
        raise RuntimeError("coordinate manifest differs from the committed contract")

    leaves: list[Leaf] = []
    manifest_rows: dict[str, tuple[str, ...]] = {}
    for line in lines[len(MANIFEST_HEADER) :]:
        fields = line.split("\t")
        if len(fields) != 16:
            raise RuntimeError("coordinate manifest row width mismatch")
        (
            identity,
            u_depth,
            u_index,
            s_depth,
            s_index,
            input_sha,
            parent_status,
            parent_receipt_sha,
            e1_r0_chart,
            e1_r0_sign,
            e1_r1_chart,
            e1_r1_sign,
            e2_r0_chart,
            e2_r0_sign,
            e2_r1_chart,
            e2_r1_sign,
        ) = fields
        numbers = (u_depth, u_index, s_depth, s_index)
        if any(
            not token.isdigit() or str(int(token)) != token for token in numbers
        ):
            raise RuntimeError("coordinate manifest row grammar mismatch")
        coordinates = tuple(int(token) for token in numbers)
        if identity != leaf_id(*coordinates):
            raise RuntimeError("coordinate manifest identity mismatch")
        if any(
            (
                chart not in CHARTS
                or sign not in {"-1", "1"}
            )
            for chart, sign in (
                (e1_r0_chart, e1_r0_sign),
                (e1_r1_chart, e1_r1_sign),
                (e2_r0_chart, e2_r0_sign),
                (e2_r1_chart, e2_r1_sign),
            )
        ) and identity != ROOT_LEAF_ID:
            raise RuntimeError("coordinate manifest chart/sign grammar mismatch")
        if identity == ROOT_LEAF_ID and (
            coordinates != (0, 0, 0, 0)
            or any(
                chart != "NONE" or sign != "0"
                for chart, sign in (
                    (e1_r0_chart, e1_r0_sign),
                    (e1_r1_chart, e1_r1_sign),
                    (e2_r0_chart, e2_r0_sign),
                    (e2_r1_chart, e2_r1_sign),
                )
            )
        ):
            raise RuntimeError("coordinate manifest root sentinel mismatch")
        if (
            SHA_RE.fullmatch(input_sha) is None
            or SHA_RE.fullmatch(parent_receipt_sha) is None
            or identity in manifest_rows
        ):
            raise RuntimeError("coordinate manifest digest or uniqueness mismatch")
        if digest_bytes(leaf_input_bytes(*coordinates)) != input_sha:
            raise RuntimeError("coordinate manifest input hash mismatch")
        manifest_rows[identity] = tuple(fields)
    if len(manifest_rows) != 53 or list(manifest_rows) != sorted(manifest_rows):
        raise RuntimeError("coordinate manifest is not the canonical 53-leaf set")

    parent_root = repo / "scripts/research/receipts" / PARENT_RUN
    if digest(parent_root / "run-manifest.txt") != PARENT_RUN_MANIFEST_SHA256:
        raise RuntimeError("parent run manifest differs from the predeclared anchor")
    if digest(parent_root / "files.sha256") != PARENT_FILES_INDEX_SHA256:
        raise RuntimeError("parent files index differs from the predeclared anchor")
    if digest(parent_root / "coordinates.tsv") != PARENT_COORDINATES_SHA256:
        raise RuntimeError("parent coordinates differ from the predeclared anchor")
    if digest(parent_root / "leaves.tsv") != PARENT_LEAVES_SHA256:
        raise RuntimeError("parent leaves differ from the predeclared anchor")

    parent_retained_verifier = (
        repo / "scripts/research/cs6_plucker_cocycle_retained_verify.py"
    )
    parent_check = subprocess.run(
        [sys.executable, parent_retained_verifier, parent_root], capture_output=True
    )
    if parent_check.returncode != 0 or parent_check.stderr:
        raise RuntimeError("anchored parent retained bundle does not verify")

    predeclaration = subprocess.run(
        ["git", "-C", repo, "show", f"{PREDECLARED_COMMIT}:{PREDECLARATION_REPORT}"],
        check=True,
        capture_output=True,
    ).stdout
    if digest_bytes(predeclaration) != PREDECLARATION_REPORT_SHA256:
        raise RuntimeError("predeclaration report differs from the committed anchor")

    parent_coordinate_lines = (parent_root / "coordinates.tsv").read_text(
        encoding="ascii"
    ).splitlines()
    if parent_coordinate_lines[:4] != [
        "SCHEMA=sounio.cs6.plucker-cocycle-coordinates.v1",
        "PARENT_COORDINATE_SET=CS6_C1_FULL_SOURCE_COVER_RETAINED_53",
        "LEAF_COUNT=53",
        "LEAF_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256",
    ]:
        raise RuntimeError("parent coordinate schema mismatch")
    parent_coordinates: dict[str, tuple[str, ...]] = {}
    for line in parent_coordinate_lines[4:]:
        fields = tuple(line.split("\t"))
        if len(fields) != 6 or fields[0] in parent_coordinates:
            raise RuntimeError("parent coordinate row mismatch")
        parent_coordinates[fields[0]] = fields
    if len(parent_coordinates) != 53 or list(parent_coordinates) != sorted(
        parent_coordinates
    ):
        raise RuntimeError("parent coordinate order/cardinality mismatch")

    parent_leaves_raw = (parent_root / "leaves.tsv").read_text(encoding="ascii")
    parent_reader = csv.DictReader(io.StringIO(parent_leaves_raw), delimiter="\t")
    parent_rows = list(parent_reader)
    if len(parent_rows) != 53 or [row["LEAF_ID"] for row in parent_rows] != list(
        parent_coordinates
    ):
        raise RuntimeError("parent leaves differ from parent coordinates")
    parent_by_id = {row["LEAF_ID"]: row for row in parent_rows}
    affine_ids = sorted(
        identity
        for identity, row in parent_by_id.items()
        if row["AFFINE_PASS"] == "true"
    )
    affine_canonical = ("\n".join(affine_ids) + "\n").encode("utf-8")
    if (
        len(affine_ids) != 28
        or digest_bytes(affine_canonical) != PARENT_AFFINE_OBLIGATION_SHA256
    ):
        raise RuntimeError("parent affine obligation set differs from the frozen set")

    parent_files = (parent_root / "files.sha256").read_text(
        encoding="ascii"
    ).splitlines()
    parent_index = {
        relative: expected
        for expected, relative in (line.split("  ", 1) for line in parent_files)
    }
    for identity, fields in manifest_rows.items():
        (
            _, u_depth, u_index, s_depth, s_index, input_sha, parent_status,
            parent_receipt_sha, e1_r0_chart, e1_r0_sign, e1_r1_chart,
            e1_r1_sign, e2_r0_chart, e2_r0_sign, e2_r1_chart, e2_r1_sign,
        ) = fields
        expected_parent_coordinate = (
            identity, u_depth, u_index, s_depth, s_index, input_sha
        )
        if parent_coordinates.get(identity) != expected_parent_coordinate:
            raise RuntimeError("full53 manifest differs from parent coordinate set")
        parent_row = parent_by_id[identity]
        if parent_status != parent_row["STATUS"]:
            raise RuntimeError("full53 manifest parent status mismatch")
        parent_input_path = parent_root / "inputs" / f"{identity}.txt"
        parent_receipt_path = parent_root / "receipts" / f"{identity}.txt"
        if (
            digest(parent_input_path) != input_sha
            or parent_index.get(f"inputs/{identity}.txt") != input_sha
            or digest(parent_receipt_path) != parent_receipt_sha
            or parent_row["RECEIPT_SHA256"] != parent_receipt_sha
            or parent_index.get(f"receipts/{identity}.txt") != parent_receipt_sha
        ):
            raise RuntimeError("full53 manifest parent payload binding mismatch")

        expected_chart_signs: list[tuple[str, int]] = []
        if identity == ROOT_LEAF_ID:
            if parent_status != "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN":
                raise RuntimeError("parent root failure class mismatch")
            expected_chart_signs = [("NONE", 0)] * 4
        else:
            receipt_raw = parent_receipt_path.read_bytes()
            for event, column in ((1, 0), (1, 1), (2, 0), (2, 1)):
                record = record_values(
                    receipt_raw, f"HOMOGENEOUS_EVENT{event}_RAY{column}"
                )
                lower, upper = interval_bounds(record["PIVOT"])
                sign = -1 if upper < 0.0 else 1 if lower > 0.0 else 0
                if record.get("CHART") not in CHARTS or sign == 0:
                    raise RuntimeError("parent chart lacks a signed pivot")
                expected_chart_signs.append((record["CHART"], sign))
        actual_chart_signs = [
            (e1_r0_chart, int(e1_r0_sign)),
            (e1_r1_chart, int(e1_r1_sign)),
            (e2_r0_chart, int(e2_r0_sign)),
            (e2_r1_chart, int(e2_r1_sign)),
        ]
        if actual_chart_signs != expected_chart_signs:
            raise RuntimeError("full53 chart/sign tuple differs from parent receipt")

        leaves.append(
            Leaf(
                int(u_depth), int(u_index), int(s_depth), int(s_index),
                input_sha, parent_status, parent_receipt_sha,
                e1_r0_chart, int(e1_r0_sign), e1_r1_chart, int(e1_r1_sign),
                e2_r0_chart, int(e2_r0_sign), e2_r1_chart, int(e2_r1_sign),
                identity in affine_ids,
            )
        )
    if [leaf.identity for leaf in leaves] != list(parent_coordinates):
        raise RuntimeError("full53 leaf order differs from the parent bundle")
    return leaves


def record_values(raw: bytes, marker: str) -> dict[str, str]:
    prefix = marker.encode("ascii") + b" "
    matches = [line for line in raw.splitlines() if line.startswith(prefix)]
    if len(matches) != 1:
        raise RuntimeError(f"receipt record cardinality mismatch: {marker}")
    values: dict[str, str] = {}
    for token in matches[0].decode("ascii").split(" ")[1:]:
        if token.count("=") != 1:
            raise RuntimeError(f"malformed receipt token: {marker}")
        key, value = token.split("=", 1)
        if key in values:
            raise RuntimeError(f"duplicate receipt token: {marker} {key}")
        values[key] = value
    return values


def interval_bounds(token: str) -> tuple[float, float]:
    match = INTERVAL_RE.fullmatch(token)
    if match is None:
        raise RuntimeError("malformed retained interval")
    lower, upper = (float.fromhex(value) for value in match.groups())
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise RuntimeError("nonfinite or inverted retained interval")
    return lower, upper


def interval_width(record: dict[str, str], key: str) -> float:
    lower, upper = interval_bounds(record[key])
    width = upper - lower
    if not math.isfinite(width) or width <= 0.0:
        raise RuntimeError("retained diagnostic interval has nonpositive width")
    return width


def extract_metrics(receipt: Path) -> Metrics:
    raw = receipt.read_bytes()
    c1_width = interval_width(record_values(raw, "C1_P2_CONTROL"), "DET")
    affine_width = interval_width(record_values(raw, "AFFINE_CARRIER"), "DET")
    boxed_width = interval_width(record_values(raw, "PLUCKER_COCYCLE"), "DET")
    shared_width = interval_width(record_values(raw, "APG_SHARED_COMPOSITION_TM2"), "DET_HULL")
    apg_width = interval_width(record_values(raw, "APG_FACTORED_EXTERIOR_TM2"), "PRIMARY_DET")
    ray_markers = ("APG_EVENT1_RAY0", "APG_EVENT1_RAY1", "APG_EVENT2_RAY0", "APG_EVENT2_RAY1")
    rays = [record_values(raw, marker) for marker in ray_markers]
    margins: list[float] = []
    for ray in rays:
        if ray["PIVOT_SIGN_CERTIFIED"] != "true":
            raise RuntimeError("computed leaf has an uncertified frozen APG chart")
        lower, upper = interval_bounds(ray["P_HULL"])
        margin = min(abs(lower), abs(upper))
        if not math.isfinite(margin) or margin <= 0.0:
            raise RuntimeError("APG pivot margin is nonfinite or nonpositive")
        margins.append(margin)
    return Metrics(
        c1_width,
        affine_width,
        boxed_width,
        shared_width,
        apg_width,
        min(margins),
        len(raw),
    )


def dependency_paths(path: Path) -> list[Path]:
    text = path.read_text(encoding="utf-8").replace("\\\n", " ")
    if ":" not in text:
        raise RuntimeError("compiler dependency file is malformed")
    return sorted({Path(item) for item in shlex.split(text.split(":", 1)[1])})


def dependency_manifest(paths: Sequence[Path], source: Path) -> bytes:
    rows: list[str] = []
    for path in paths:
        if path == source:
            rows.append(f"{digest(path)}  BUNDLE/worker-source.cpp")
        elif path.is_file():
            rows.append(f"{digest(path)}  {path}")
    if not rows:
        raise RuntimeError("compiler emitted no hashable dependencies")
    return ("\n".join(sorted(set(rows))) + "\n").encode("ascii")


def known_interval_domain_failure(stderr: bytes) -> bool:
    lowered = stderr.lower()
    return b"interval error:" in lowered and (
        b"division by 0" in lowered or b"division by zero" in lowered
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
    left_token, right_token = lines[1].removeprefix("   left=").split(
        "  right=", 1
    )
    try:
        return math.isfinite(float(left_token)) and math.isfinite(float(right_token))
    except ValueError:
        return False


def known_frozen_chart_failure(stderr: bytes) -> bool:
    return b"frozen TM2 pivot sign was not certified" in stderr


def bool_value(values: dict[str, str], key: str) -> bool:
    if values[key] not in {"true", "false"}:
        raise RuntimeError(f"noncanonical verifier boolean: {key}")
    return values[key] == "true"


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def minimum(values: Sequence[float]) -> float:
    return min(values) if values else float("nan")


def median(values: Sequence[float]) -> float:
    return statistics.median(values) if values else float("nan")


def maximum(values: Sequence[float]) -> float:
    return max(values) if values else float("nan")


def decimal(value: float) -> str:
    return format(value, ".17g")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capd-config", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--coordinate-manifest", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--keep-failed", action="store_true")
    args = parser.parse_args(argv)
    if SHA_RE.fullmatch(args.root_challenge) is None:
        die("root challenge must be lowercase SHA-256")
    if not 1 <= args.jobs <= 32:
        die("jobs must be in [1,32]")
    if not 1 <= args.timeout_seconds <= 3600:
        die("timeout must be in [1,3600]")

    repo = Path(__file__).resolve().parents[2]
    source = repo / "scripts/research/cs6_affine_projective_cocycle_full53_probe.cpp"
    verifier = repo / "scripts/research/cs6_affine_projective_cocycle_full53_verify.py"
    runner = Path(__file__).resolve()
    coordinates = args.coordinate_manifest or (
        repo
        / "scripts/research/cs6_affine_projective_cocycle_full53_coordinates_v1.tsv"
    )
    for required in (source, verifier, runner, coordinates):
        if not required.is_file():
            die(f"missing runner input: {required}")
    leaves = parse_coordinate_manifest(coordinates, repo)
    selftest_leaf = next(
        leaf for leaf in leaves if leaf.identity == CHALLENGE_SELFTEST_LEAF_ID
    )
    if (
        leaf_challenge("0" * 64, FROZEN_MANIFEST_SHA256, selftest_leaf)
        != CHALLENGE_SELFTEST_EXPECTED
    ):
        die("full53 challenge helper self-test failed")

    git_head = subprocess.run(
        ["git", "-C", repo, "rev-parse", "HEAD"], check=True, capture_output=True
    ).stdout
    git_status = subprocess.run(
        ["git", "-C", repo, "status", "--short", "--untracked-files=all"],
        check=True,
        capture_output=True,
    ).stdout
    if git_status:
        die("full53 evidence run requires a clean committed worktree")
    for tracked_path in (source, verifier, runner, coordinates):
        relative = tracked_path.relative_to(repo).as_posix()
        committed = subprocess.run(
            ["git", "-C", repo, "show", f"HEAD:{relative}"],
            check=True,
            capture_output=True,
        ).stdout
        if committed != tracked_path.read_bytes():
            die(f"full53 evidence input differs from HEAD: {relative}")

    capd_config = args.capd_config.resolve()
    if not capd_config.is_file() or not os.access(capd_config, os.X_OK):
        die("capd-config is not executable")
    cxx_found = shutil.which(args.cxx)
    if cxx_found is None:
        die(f"C++ compiler not found: {args.cxx}")
    cxx = Path(cxx_found).resolve()
    python = Path(sys.executable).resolve()
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        die("run directory already exists")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(
        tempfile.mkdtemp(
            prefix=".cs6-affine-projective-cocycle-full53.", dir=run_dir.parent
        )
    )
    complete = False
    try:
        for directory in ("inputs", "receipts", "verifications", "stderr"):
            (work / directory).mkdir()
        source_snapshot = work / "worker-source.cpp"
        verifier_snapshot = work / "leaf-verifier.py"
        runner_snapshot = work / "runner.py"
        coordinate_snapshot = work / "coordinates.tsv"
        for source_path, target in (
            (source, source_snapshot),
            (verifier, verifier_snapshot),
            (runner, runner_snapshot),
            (coordinates, coordinate_snapshot),
        ):
            shutil.copy2(source_path, target)

        def capd(option: str) -> str:
            result = subprocess.run(
                [capd_config, option], check=True, capture_output=True, text=True
            )
            return result.stdout.strip()

        capd_version = capd("--modversion")
        capd_cflags = capd("--cflags")
        capd_libs = capd("--libs")
        if capd_version != "5.3.0":
            die(f"unsupported CAPD version: {capd_version}")
        flags = shlex.split(capd_cflags)
        if "-D__USE_FILIB__" not in flags or "-frounding-math" not in flags:
            die("CAPD config lacks FILIB outward-rounding flags")
        canonical_kv(
            work / "run-contract.txt",
            (
                (
                    "SCHEMA",
                    "sounio.cs6.affine-projective-cocycle-full53-run-contract.v1",
                ),
                ("SOURCE", "N0"),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("COORDINATE_MANIFEST_SHA256", digest(coordinate_snapshot)),
                ("PREDECLARED_IN_COMMIT", PREDECLARED_COMMIT),
                ("PREDECLARATION_REPORT_SHA256", PREDECLARATION_REPORT_SHA256),
                ("CONTRACT_FROZEN_IN_COMMIT", CONTRACT_FROZEN_IN_COMMIT),
                ("FROZEN_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
                ("IMPLEMENTATION_COMMIT", git_head.decode("ascii").strip()),
                ("PARENT_RUN_MANIFEST_SHA256", PARENT_RUN_MANIFEST_SHA256),
                ("PARENT_FILES_INDEX_SHA256", PARENT_FILES_INDEX_SHA256),
                ("PARENT_COORDINATES_SHA256", PARENT_COORDINATES_SHA256),
                ("PARENT_LEAVES_SHA256", PARENT_LEAVES_SHA256),
                (
                    "PARENT_AFFINE_OBLIGATION_SHA256",
                    PARENT_AFFINE_OBLIGATION_SHA256,
                ),
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
                ("JOBS", str(args.jobs)),
                ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
                ("DISCRETE_POINCARE_COCYCLE", "true"),
                ("COMMON_SOURCE_SYMBOLS_PRESERVED", "true"),
                ("PRIMARY_INTERVAL_INTERSECTED_WITH_CONTROLS", "false"),
                ("FULL_SOURCE_CARRIER_PROVED", "false"),
            ),
        )
        (work / "capd-cflags.txt").write_text(capd_cflags + "\n", encoding="ascii")
        (work / "capd-libs.txt").write_text(capd_libs + "\n", encoding="ascii")
        (work / "capd-version.txt").write_text(capd_version + "\n", encoding="ascii")
        compiler_version = subprocess.run(
            [cxx, "--version"], check=True, capture_output=True
        ).stdout.rstrip(b"\n")
        (work / "compiler-version.txt").write_bytes(compiler_version + b"\n")
        (work / "python-version.txt").write_bytes(
            subprocess.run([python, "--version"], check=True, capture_output=True).stdout
        )
        (work / "git-head.txt").write_bytes(git_head)
        (work / "git-status.txt").write_bytes(git_status)

        source_sha = digest(source_snapshot)
        dependency_file = work / "dependencies.d"
        binary = work / "worker-binary"
        compile_command = [
            str(cxx),
            "-std=c++17",
            *flags,
            "-O0",
            f'-DCS6_WORKER_SOURCE_SHA256="{source_sha}"',
            str(source_snapshot),
            "-MD",
            "-MF",
            str(dependency_file),
            "-o",
            str(binary),
            *shlex.split(capd_libs),
        ]
        (work / "compile-command.txt").write_text(
            shlex.join(compile_command) + "\n", encoding="ascii"
        )
        compiled = subprocess.run(compile_command, capture_output=True)
        (work / "compile-stdout.txt").write_bytes(compiled.stdout)
        (work / "compile-stderr.txt").write_bytes(compiled.stderr)
        if compiled.returncode != 0:
            die(f"worker compilation failed: {compiled.returncode}")
        worker_sha = digest(binary)
        (work / "executed-worker.sha256").write_text(
            f"{worker_sha}  worker-binary\n", encoding="ascii"
        )
        dependencies_before = dependency_manifest(
            dependency_paths(dependency_file), source_snapshot
        )
        (work / "dependencies.sha256").write_bytes(dependencies_before)
        link_rows = sorted(
            f"{digest(Path(item))}  {Path(item)}"
            for item in shlex.split(capd_libs)
            if Path(item).is_file()
        )
        if not link_rows:
            die("CAPD link flags contain no hashable files")
        (work / "link-inputs.sha256").write_text(
            "\n".join(link_rows) + "\n", encoding="ascii"
        )
        linkage = subprocess.run(["ldd", binary], check=True, capture_output=True, text=True)
        (work / "runtime-linkage.txt").write_text(linkage.stdout, encoding="ascii")
        runtime_paths: set[Path] = set()
        for line in linkage.stdout.splitlines():
            fields = line.split()
            candidate = None
            if "=>" in fields and fields.index("=>") + 1 < len(fields):
                candidate = fields[fields.index("=>") + 1]
            elif fields and fields[0].startswith("/"):
                candidate = fields[0]
            if candidate and candidate.startswith("/") and Path(candidate).is_file():
                runtime_paths.add(Path(candidate))
        (work / "runtime-libraries.sha256").write_text(
            "".join(f"{digest(path)}  {path}\n" for path in sorted(runtime_paths)),
            encoding="ascii",
        )

        def unresolved(leaf: Leaf, status: str, challenge: str, receipt_sha: str, rc: int, elapsed: int) -> LeafResult:
            return LeafResult(
                leaf=leaf,
                status=status,
                method="NONE",
                probe_pass=False,
                affine=False,
                projective_x=False,
                projective_y=False,
                projective_plus=False,
                projective_minus=False,
                homogeneous=False,
                apg_valid=False,
                apg=False,
                rescue=False,
                narrower_boxed=False,
                narrower_affine=False,
                narrower_shared=False,
                certificate=False,
                subdivision=True,
                challenge=challenge,
                receipt_sha=receipt_sha,
                verification_sha=ZERO_SHA256,
                physical_sha=ZERO_SHA256,
                worker_rc=rc,
                elapsed_ms=elapsed,
                metrics=None,
            )

        def run_leaf(leaf: Leaf) -> LeafResult:
            identity = leaf.identity
            input_path = work / "inputs" / f"{identity}.txt"
            receipt_path = work / "receipts" / f"{identity}.txt"
            verification_path = work / "verifications" / f"{identity}.txt"
            stderr_path = work / "stderr" / f"{identity}.txt"
            input_raw = leaf_input_bytes(
                leaf.u_depth, leaf.u_index, leaf.s_depth, leaf.s_index
            )
            input_path.write_bytes(input_raw)
            if digest_bytes(input_raw) != leaf.input_sha:
                raise RuntimeError(f"canonical input drift: {identity}")
            manifest_sha = digest(coordinate_snapshot)
            challenge = leaf_challenge(args.root_challenge, manifest_sha, leaf)
            command = [
                str(binary),
                str(leaf.u_depth),
                str(leaf.u_index),
                str(leaf.s_depth),
                str(leaf.s_index),
                leaf.input_sha,
                leaf.parent_receipt_sha,
                leaf.e1_r0_chart,
                str(leaf.e1_r0_sign),
                leaf.e1_r1_chart,
                str(leaf.e1_r1_sign),
                leaf.e2_r0_chart,
                str(leaf.e2_r0_sign),
                leaf.e2_r1_chart,
                str(leaf.e2_r1_sign),
                manifest_sha,
                challenge,
            ]
            started = time.monotonic_ns()
            try:
                worker = subprocess.run(command, capture_output=True, timeout=args.timeout_seconds)
            except subprocess.TimeoutExpired as error:
                elapsed = (time.monotonic_ns() - started) // 1_000_000
                receipt_path.write_bytes(error.stdout or b"")
                stderr_path.write_bytes(error.stderr or b"")
                return unresolved(leaf, "COMPUTATION_UNRESOLVED_TIMEOUT", challenge, digest(receipt_path), 124, elapsed)
            elapsed = (time.monotonic_ns() - started) // 1_000_000
            receipt_path.write_bytes(worker.stdout)
            stderr_path.write_bytes(worker.stderr)
            if worker.returncode != 0:
                if known_interval_domain_failure(worker.stderr):
                    return unresolved(
                        leaf, "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN", challenge,
                        digest(receipt_path), worker.returncode, elapsed,
                    )
                if known_frozen_chart_failure(worker.stderr):
                    status = "COMPUTATION_UNRESOLVED_FROZEN_CHART"
                else:
                    status = "COMPUTATION_UNRESOLVED_UNEXPECTED_WORKER_FAILURE"
                return unresolved(
                    leaf, status, challenge, digest(receipt_path),
                    worker.returncode, elapsed,
                )
            if identity == ROOT_LEAF_ID:
                return unresolved(
                    leaf,
                    "ROOT_POLICY_VIOLATION_COMPUTED",
                    challenge,
                    digest(receipt_path),
                    worker.returncode,
                    elapsed,
                )
            if worker.stderr:
                raise RuntimeError(f"worker emitted stderr for {identity}")
            verification = subprocess.run(
                [
                    python,
                    verifier_snapshot,
                    receipt_path,
                    "--source-sha",
                    source_sha,
                    "--input",
                    input_path,
                    "--coordinate-manifest",
                    coordinate_snapshot,
                    "--parent-receipt",
                    repo
                    / "scripts/research/receipts"
                    / PARENT_RUN
                    / "receipts"
                    / f"{identity}.txt",
                    "--root-challenge",
                    args.root_challenge,
                ],
                capture_output=True,
            )
            verification_path.write_bytes(verification.stdout)
            if verification.returncode != 0 or verification.stderr:
                (work / "stderr" / f"{identity}.verifier.txt").write_bytes(verification.stderr)
                raise RuntimeError(f"leaf verification failed for {identity}")
            values = parse_kv_bytes(verification.stdout, VERIFICATION_KEYS)
            probe_pass = bool_value(values, "PROBE_PASS")
            certificate = bool_value(values, "CERTIFICATE_PASS")
            subdivision = bool_value(values, "SUBDIVISION_REQUIRED")
            if certificate == subdivision:
                raise RuntimeError(f"inconsistent verified status for {identity}")
            if (
                values["COORDINATE_MANIFEST_SHA256"] != manifest_sha
                or values["PARENT_RECEIPT_SHA256"] != leaf.parent_receipt_sha
                or values["LEAF_CHALLENGE"] != challenge
            ):
                raise RuntimeError(f"verifier contract binding mismatch for {identity}")
            metrics = extract_metrics(receipt_path)
            return LeafResult(
                leaf=leaf,
                status=(
                    "PROBE_VALID_CERTIFIED"
                    if certificate
                    else "PROBE_VALID_UNCERTIFIED"
                    if probe_pass
                    else "PROBE_VERIFIED_STRUCTURALLY_INVALID"
                ),
                method=values["LEAF_METHOD"],
                probe_pass=probe_pass,
                affine=bool_value(values, "AFFINE_CERTIFICATE_PASS"),
                projective_x=bool_value(values, "PROJECTIVE_X_CERTIFICATE_PASS"),
                projective_y=bool_value(values, "PROJECTIVE_Y_CERTIFICATE_PASS"),
                projective_plus=bool_value(values, "PROJECTIVE_PLUS_CERTIFICATE_PASS"),
                projective_minus=bool_value(values, "PROJECTIVE_MINUS_CERTIFICATE_PASS"),
                homogeneous=bool_value(values, "HOMOGENEOUS_CERTIFICATE_PASS"),
                apg_valid=bool_value(values, "APG_COMPUTATION_VALID"),
                apg=bool_value(values, "APG_CERTIFICATE_PASS"),
                rescue=bool_value(values, "APG_RESCUE"),
                narrower_boxed=bool_value(values, "APG_STRICTLY_NARROWER_THAN_BOXED"),
                narrower_affine=bool_value(values, "APG_STRICTLY_NARROWER_THAN_AFFINE"),
                narrower_shared=bool_value(values, "APG_STRICTLY_NARROWER_THAN_SHARED"),
                certificate=certificate,
                subdivision=subdivision,
                challenge=challenge,
                receipt_sha=values["RECEIPT_SHA256"],
                verification_sha=digest(verification_path),
                physical_sha=values["PHYSICAL_SHA256"],
                worker_rc=worker.returncode,
                elapsed_ms=elapsed,
                metrics=metrics,
            )

        results: list[LeafResult] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = {executor.submit(run_leaf, leaf): leaf for leaf in leaves}
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result.leaf.identity)

        valid = [result for result in results if result.probe_pass]
        auditable = [
            result
            for result in results
            if result.leaf.identity != ROOT_LEAF_ID
            and result.verification_sha != ZERO_SHA256
        ]

        def mutation_audit(
            result: LeafResult,
        ) -> tuple[LeafResult, bytes, bytes, dict[str, str]]:
            identity = result.leaf.identity
            parent_receipt = (
                repo
                / "scripts/research/receipts"
                / PARENT_RUN
                / "receipts"
                / f"{identity}.txt"
            )
            process = subprocess.run(
                [
                    python,
                    verifier_snapshot,
                    work / "receipts" / f"{identity}.txt",
                    "--source-sha",
                    source_sha,
                    "--input",
                    work / "inputs" / f"{identity}.txt",
                    "--coordinate-manifest",
                    coordinate_snapshot,
                    "--parent-receipt",
                    parent_receipt,
                    "--root-challenge",
                    args.root_challenge,
                    "--self-test-mutations",
                ],
                capture_output=True,
            )
            if process.returncode != 0 or process.stderr:
                raise RuntimeError(f"mutation audit failed for {identity}")
            values = parse_kv_bytes(process.stdout, VERIFICATION_KEYS)
            tests = int(values["MUTATION_TESTS"])
            rejected = int(values["MUTATIONS_REJECTED"])
            if tests == 0 or tests != rejected:
                raise RuntimeError(
                    f"mutation audit did not reject every mutation for {identity}"
                )
            if (
                values["RECEIPT_SHA256"] != result.receipt_sha
                or values["COORDINATE_MANIFEST_SHA256"]
                != FROZEN_MANIFEST_SHA256
                or values["PARENT_RECEIPT_SHA256"]
                != result.leaf.parent_receipt_sha
                or values["LEAF_CHALLENGE"] != result.challenge
            ):
                raise RuntimeError(f"mutation audit binding mismatch for {identity}")
            return result, process.stdout, process.stderr, values

        audits: list[tuple[LeafResult, bytes, bytes, dict[str, str]]] = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs
        ) as executor:
            futures = [executor.submit(mutation_audit, result) for result in auditable]
            for future in concurrent.futures.as_completed(futures):
                audits.append(future.result())
        audits.sort(key=lambda item: item[0].leaf.identity)

        mutation_rows = [
            "LEAF_ID\tRECEIPT_SHA256\tVERIFICATION_SHA256\tAUDIT_STDOUT_SHA256\tMUTATION_TESTS\tMUTATIONS_REJECTED"
        ]
        mutation_tests = 0
        mutation_rejected = 0
        suite_sizes: set[int] = set()
        for result, stdout, _, values in audits:
            tests = int(values["MUTATION_TESTS"])
            rejected = int(values["MUTATIONS_REJECTED"])
            suite_sizes.add(tests)
            mutation_tests += tests
            mutation_rejected += rejected
            mutation_rows.append(
                "\t".join(
                    (
                        result.leaf.identity,
                        result.receipt_sha,
                        result.verification_sha,
                        digest_bytes(stdout),
                        str(tests),
                        str(rejected),
                    )
                )
            )
        if len(suite_sizes) > 1:
            raise RuntimeError("mutation suite size differs across paired leaves")
        (work / "mutation-audits.tsv").write_text(
            "\n".join(mutation_rows) + "\n", encoding="ascii"
        )
        if audits:
            (work / "mutation-audit.txt").write_bytes(audits[0][1])
            (work / "mutation-audit-stderr.txt").write_bytes(audits[0][2])
        else:
            (work / "mutation-audit.txt").write_bytes(b"")
            (work / "mutation-audit-stderr.txt").write_bytes(b"")

        columns = (
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
        rows = ["\t".join(columns)]
        for result in results:
            metric_values = ("-",) * 7 if result.metrics is None else (
                decimal(result.metrics.c1_width),
                decimal(result.metrics.affine_width),
                decimal(result.metrics.boxed_width),
                decimal(result.metrics.shared_width),
                decimal(result.metrics.apg_width),
                decimal(result.metrics.minimum_apg_pivot_margin),
                str(result.metrics.receipt_bytes),
            )
            leaf = result.leaf
            stderr_sha = digest(work / "stderr" / f"{leaf.identity}.txt")
            rows.append("\t".join((
                leaf.identity, str(leaf.u_depth), str(leaf.u_index), str(leaf.s_depth),
                str(leaf.s_index), result.status, leaf.parent_status,
                str(leaf.parent_affine_pass).lower(), leaf.parent_receipt_sha,
                leaf.e1_r0_chart, str(leaf.e1_r0_sign),
                leaf.e1_r1_chart, str(leaf.e1_r1_sign),
                leaf.e2_r0_chart, str(leaf.e2_r0_sign),
                leaf.e2_r1_chart, str(leaf.e2_r1_sign), result.method,
                str(result.probe_pass).lower(), str(result.affine).lower(),
                str(result.homogeneous).lower(), str(result.apg_valid).lower(),
                str(result.apg).lower(), str(result.rescue).lower(),
                str(result.narrower_boxed).lower(), str(result.narrower_affine).lower(),
                str(result.narrower_shared).lower(), str(result.certificate).lower(),
                str(result.subdivision).lower(), leaf.input_sha, result.challenge,
                result.receipt_sha, stderr_sha, result.verification_sha, result.physical_sha,
                str(result.worker_rc), str(result.elapsed_ms), *metric_values,
            )))
        (work / "leaves.tsv").write_text("\n".join(rows) + "\n", encoding="ascii")

        paired_valid = [
            result
            for result in valid
            if result.leaf.identity != ROOT_LEAF_ID
        ]
        metrics = [
            result.metrics for result in paired_valid if result.metrics is not None
        ]
        affine_count = sum(result.affine for result in paired_valid)
        apg_count = sum(result.apg for result in paired_valid)
        parent_nonaffine = [
            result for result in results if not result.leaf.parent_affine_pass
            and result.leaf.identity != ROOT_LEAF_ID
        ]
        rescue_count = sum(result.rescue for result in parent_nonaffine)
        affine_obligations = [
            result for result in results if result.leaf.parent_affine_pass
        ]
        affine_loss_count = sum(
            not (result.probe_pass and result.apg_valid and result.apg)
            for result in affine_obligations
        )
        affine_preserved_count = len(affine_obligations) - affine_loss_count
        narrower_boxed_count = sum(result.narrower_boxed for result in valid)
        narrower_affine_count = sum(result.narrower_affine for result in valid)
        narrower_shared_count = sum(result.narrower_shared for result in valid)
        apg_to_boxed = [item.apg_width / item.boxed_width for item in metrics]
        apg_to_affine = [item.apg_width / item.affine_width for item in metrics]
        apg_to_shared = [item.apg_width / item.shared_width for item in metrics]
        root_result = next(
            result for result in results if result.leaf.identity == ROOT_LEAF_ID
        )
        root_class_match = (
            root_result.status == "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN"
            and not root_result.probe_pass
            and root_result.worker_rc == 1
            and root_result.receipt_sha == digest_bytes(b"")
            and canonical_interval_domain_failure(
                (work / "stderr" / f"{ROOT_LEAF_ID}.txt").read_bytes()
            )
        )
        new_unresolved_count = sum(
            not result.probe_pass
            for result in results
            if result.leaf.identity != ROOT_LEAF_ID
        )
        mutation_suite_size = next(iter(suite_sizes), 0)
        full53_supported = (
            len(results) == 53
            and len(paired_valid) == 52
            and root_class_match
            and new_unresolved_count == 0
            and all(result.apg_valid for result in paired_valid)
            and affine_loss_count == 0
            and rescue_count > 0
            and len(audits) == 52
            and mutation_suite_size == MUTATIONS_PER_LEAF
            and mutation_tests == MUTATION_TOTAL
            and mutation_rejected == MUTATION_TOTAL
        )
        summary_fields = (
            (
                "SCHEMA",
                "sounio.cs6.affine-projective-cocycle-full53-summary.v1",
            ),
            ("COORDINATE_COUNT", str(len(results))),
            ("PAIRED_ELIGIBLE_COUNT", "52"),
            ("PAIRED_VALID_COUNT", str(len(paired_valid))),
            ("PROBE_VALID_COUNT", str(len(paired_valid))),
            ("ROOT_INTERVAL_DOMAIN_CLASS_MATCH", str(root_class_match).lower()),
            ("NEW_UNRESOLVED_COUNT", str(new_unresolved_count)),
            (
                "COMPUTATION_UNRESOLVED_COUNT",
                str(sum(result.status.startswith("COMPUTATION_UNRESOLVED_") for result in results)),
            ),
            ("PARENT_AFFINE_OBLIGATION_COUNT", str(len(affine_obligations))),
            ("PARENT_AFFINE_PRESERVED_COUNT", str(affine_preserved_count)),
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
            ("MIN_APG_TO_BOXED_WIDTH_RATIO", decimal(minimum(apg_to_boxed))),
            ("MEDIAN_APG_TO_BOXED_WIDTH_RATIO", decimal(median(apg_to_boxed))),
            ("MEAN_APG_TO_BOXED_WIDTH_RATIO", decimal(mean(apg_to_boxed))),
            ("MAX_APG_TO_BOXED_WIDTH_RATIO", decimal(maximum(apg_to_boxed))),
            ("MIN_APG_TO_AFFINE_WIDTH_RATIO", decimal(minimum(apg_to_affine))),
            ("MEDIAN_APG_TO_AFFINE_WIDTH_RATIO", decimal(median(apg_to_affine))),
            ("MEAN_APG_TO_AFFINE_WIDTH_RATIO", decimal(mean(apg_to_affine))),
            ("MAX_APG_TO_AFFINE_WIDTH_RATIO", decimal(maximum(apg_to_affine))),
            ("MIN_APG_TO_SHARED_WIDTH_RATIO", decimal(minimum(apg_to_shared))),
            ("MEDIAN_APG_TO_SHARED_WIDTH_RATIO", decimal(median(apg_to_shared))),
            ("MEAN_APG_TO_SHARED_WIDTH_RATIO", decimal(mean(apg_to_shared))),
            ("MAX_APG_TO_SHARED_WIDTH_RATIO", decimal(maximum(apg_to_shared))),
            ("MINIMUM_APG_PIVOT_MARGIN", decimal(minimum([item.minimum_apg_pivot_margin for item in metrics]))),
            ("TOTAL_WORKER_ELAPSED_MS", str(sum(result.elapsed_ms for result in results))),
            ("MEAN_VALID_WORKER_ELAPSED_MS", decimal(mean([float(result.elapsed_ms) for result in valid]))),
            ("MEAN_RECEIPT_BYTES", decimal(mean([float(item.receipt_bytes) for item in metrics]))),
            ("LEAF_VERIFICATION_COUNT", str(len(auditable))),
            ("MUTATION_AUDITED_LEAF_COUNT", str(len(audits))),
            ("MUTATION_SUITE_SIZE_PER_LEAF", str(mutation_suite_size)),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("MUTATIONS_REJECTED", str(mutation_rejected)),
            ("H_APG_CS6_FULL53_SUPPORTED", str(full53_supported).lower()),
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
        canonical_kv(work / "summary.txt", summary_fields)

        dependencies_after = dependency_manifest(dependency_paths(dependency_file), source_snapshot)
        if dependencies_before != dependencies_after:
            raise RuntimeError("compile dependency changed during execution")
        if digest(binary) != worker_sha:
            raise RuntimeError("worker binary changed during execution")
        binary.unlink()
        dependency_file.unlink()
        artifacts = (
            "capd-cflags.txt", "capd-libs.txt", "capd-version.txt",
            "compile-command.txt", "compile-stderr.txt", "compile-stdout.txt",
            "compiler-version.txt", "coordinates.tsv", "dependencies.sha256",
            "executed-worker.sha256", "git-head.txt", "git-status.txt",
            "leaf-verifier.py", "leaves.tsv",
            "link-inputs.sha256", "mutation-audit-stderr.txt",
            "mutation-audit.txt", "mutation-audits.tsv", "python-version.txt",
            "run-contract.txt",
            "runner.py", "runtime-libraries.sha256", "runtime-linkage.txt",
            "summary.txt", "worker-source.cpp",
        )
        manifest_fields: list[tuple[str, str]] = [
            (
                "SCHEMA",
                "sounio.cs6.affine-projective-cocycle-full53-run-manifest.v1",
            ),
            ("RUN_COMPLETE", "true"),
            ("CAPD_VERSION", capd_version),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("SOURCE_SHA256", source_sha),
            ("VERIFIER_SHA256", digest(verifier_snapshot)),
            ("RUNNER_SHA256", digest(runner_snapshot)),
            ("COORDINATE_MANIFEST_SHA256", digest(coordinate_snapshot)),
            ("PREDECLARED_IN_COMMIT", PREDECLARED_COMMIT),
            ("PREDECLARATION_REPORT_SHA256", PREDECLARATION_REPORT_SHA256),
            ("CONTRACT_FROZEN_IN_COMMIT", CONTRACT_FROZEN_IN_COMMIT),
            ("FROZEN_MANIFEST_SHA256", FROZEN_MANIFEST_SHA256),
            ("IMPLEMENTATION_COMMIT", git_head.decode("ascii").strip()),
            ("PARENT_RUN_MANIFEST_SHA256", PARENT_RUN_MANIFEST_SHA256),
            ("PARENT_FILES_INDEX_SHA256", PARENT_FILES_INDEX_SHA256),
            ("PARENT_COORDINATES_SHA256", PARENT_COORDINATES_SHA256),
            ("PARENT_LEAVES_SHA256", PARENT_LEAVES_SHA256),
            (
                "PARENT_AFFINE_OBLIGATION_SHA256",
                PARENT_AFFINE_OBLIGATION_SHA256,
            ),
            ("LEAF_COUNT", str(len(results))),
            ("PAIRED_VALID_COUNT", str(len(paired_valid))),
            ("ROOT_INTERVAL_DOMAIN_CLASS_MATCH", str(root_class_match).lower()),
            ("NEW_UNRESOLVED_COUNT", str(new_unresolved_count)),
            ("PARENT_AFFINE_OBLIGATION_COUNT", str(len(affine_obligations))),
            ("PARENT_AFFINE_LOSS_COUNT", str(affine_loss_count)),
            ("MUTATION_AUDITED_LEAF_COUNT", str(len(audits))),
            ("MUTATION_SUITE_SIZE_PER_LEAF", str(mutation_suite_size)),
            ("MUTATION_TESTS", str(mutation_tests)),
            ("MUTATIONS_REJECTED", str(mutation_rejected)),
            ("H_APG_CS6_FULL53_SUPPORTED", str(full53_supported).lower()),
            ("PROMOTION_ELIGIBLE", "false"),
        ]
        for name in artifacts:
            manifest_fields.append((
                name.upper().replace("-", "_").replace(".", "_") + "_SHA256",
                digest(work / name),
            ))
        canonical_kv(work / "run-manifest.txt", manifest_fields)
        os.replace(work, run_dir)
        complete = True
    finally:
        if not complete:
            if args.keep_failed:
                print(f"FAILED_WORK_DIR={work}", file=sys.stderr)
            else:
                shutil.rmtree(work, ignore_errors=True)

    print(f"RUN_DIR={run_dir}")
    print("COORDINATE_COUNT=53")
    print(f"PAIRED_VALID_COUNT={len(paired_valid)}")
    print(f"NEW_UNRESOLVED_COUNT={new_unresolved_count}")
    print(f"PARENT_AFFINE_LOSS_COUNT={affine_loss_count}")
    print(f"AFFINE_CERTIFIED_COUNT={affine_count}")
    print(f"APG_CERTIFIED_COUNT={apg_count}")
    print(f"APG_RESCUE_ON_PARENT_NONAFFINE_COUNT={rescue_count}")
    print(f"MUTATIONS_REJECTED={mutation_rejected}/{mutation_tests}")
    print(f"H_APG_CS6_FULL53_SUPPORTED={str(full53_supported).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
