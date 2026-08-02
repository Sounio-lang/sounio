#!/usr/bin/env python3
"""Independently verify an adaptive H-PG to fixed-chart H-APG cover run."""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Mapping, Sequence


sys.dont_write_bytecode = True

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ROOT_ID = "U00-0000000000_S00-0000000000"
KAT_CERTIFICATE_SCHEMA = "sounio.cs6.hapg-full-source-cover-kat-prerequisite.v2"
KAT_RUN_CONTRACT_EVIDENCE_KEYS = (
    "KAT_COORDINATE_MANIFEST_SHA256",
    "KAT_EXPECTED_RESULTS_SHA256",
    "KAT_WAVE_CONTRACT_SHA256",
    "KAT_WAVE_RESULT_SHA256",
    "KAT_LEAF_EVIDENCE_VALID",
    "KAT_HPG_VERIFIER_REPLAY_COUNT",
    "KAT_HAPG_VERIFIER_REPLAY_COUNT",
    "KAT_EVALUATED_NODE_COUNT",
    "KAT_HPG_SIGNED_CHART_COUNT",
    "KAT_HAPG_ATTEMPTED_COUNT",
    "KAT_HAPG_CERTIFIED_COUNT",
    "KAT_HAPG_UNCERTIFIED_COUNT",
    "KAT_HAPG_RESCUE_COUNT",
    "KAT_HPG_MUTATION_TESTS",
    "KAT_HPG_MUTATIONS_REJECTED",
    "KAT_HAPG_MUTATION_TESTS",
    "KAT_HAPG_MUTATIONS_REJECTED",
)
RUN_MANIFEST_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "MODE",
    "ROOT_CHALLENGE",
    "CAPD_VERSION",
    "INTERVAL_BACKEND",
    "OPTIMIZATION_LEVEL",
    "RUN_CONTRACT_SHA256",
    "FILES_INDEX_SHA256",
    "FILE_COUNT",
    "EVALUATED_NODE_COUNT",
    "WAVE_COUNT",
    "LOCAL_PROCESS_ORDERED_HASH_CHAIN",
    "EXECUTION_PROVENANCE_ATTESTED",
    "PROMOTION_ELIGIBLE",
)
RUN_CONTRACT_KEYS = (
    "SCHEMA",
    "FROZEN_CONTRACT_SHA256",
    "MODE",
    "SOURCE",
    "ROOT_CHALLENGE",
    "TRAVERSAL",
    "SPLIT_RULE",
    "TERMINAL_PREDICATE",
    "HPG_WORKER_SOURCE_SHA256",
    "HPG_VERIFIER_SOURCE_SHA256",
    "HAPG_WORKER_SOURCE_SHA256",
    "HAPG_KERNEL_SOURCE_SHA256",
    "HAPG_VERIFIER_ADAPTER_SHA256",
    "HAPG_NUMERIC_VERIFIER_SHA256",
    "KAT_ANCHOR_SHA256",
    "SLURM_JOB_SCRIPT_SHA256",
    "BUILD_MODE",
    "PREBUILT_RUN_MANIFEST_SHA256",
    "SLURM_JOB_ID",
    "EXECUTION_NODE",
    "SLURM_JOB_VERIFIED",
    "SLURM_JOB_RECORD_SHA256",
    "WORKING_FILESYSTEM_POLICY",
    "JOBS",
    "TIMEOUT_SECONDS",
    "MUTATION_AUDIT",
    "LOCAL_PROCESS_ORDERED_HASH_CHAIN",
    "EXECUTION_PROVENANCE_ATTESTED",
    "PROMOTION_ELIGIBLE",
    "MAX_NODES",
    "MAX_WAVES",
    "MAX_U_DEPTH",
    "MAX_S_DEPTH",
    "ALL_OR_NONE_WAVE_ADMISSION",
    "FRESH_REPLAY_ROOT_CHALLENGE",
    "KAT_PREREQUISITE_REQUIRED",
    "KAT_PREREQUISITE_CERTIFICATE_SCHEMA",
    "KAT_PREREQUISITE_CERTIFICATE_SHA256",
    "KAT_PREREQUISITE_SACCT_SHA256",
    "KAT_JOB_ID",
    "KAT_ARCHIVE_SHA256",
    "KAT_GIT_HEAD",
    "KAT_FROZEN_CONTRACT_SHA256",
    "KAT_BASE_REPO_BUNDLE_SHA256",
    "KAT_BASE_GIT_HEAD",
    "KAT_REPO_DELTA_BUNDLE_SHA256",
    "KAT_PREBUILT_ARCHIVE_SHA256",
    "KAT_PREBUILT_RUN_MANIFEST_SHA256",
    "KAT_SLURM_JOB_SCRIPT_SHA256",
    "KAT_END_UTC",
    *KAT_RUN_CONTRACT_EVIDENCE_KEYS,
    "ADAPTIVE_SUBMIT_UTC",
    "KAT_PREREQUISITE_VALID",
)
SUMMARY_KEYS = (
    "SCHEMA",
    "MODE",
    "BOUNDED_RUN_COMPLETE",
    "INFRASTRUCTURE_VALID",
    "EVALUATED_NODE_COUNT",
    "WAVE_COUNT",
    "HPG_SIGNED_CHART_COUNT",
    "HAPG_ATTEMPTED_COUNT",
    "HAPG_CERTIFIED_COUNT",
    "HAPG_RESCUE_COUNT",
    "HPG_MUTATION_TESTS",
    "HPG_MUTATIONS_REJECTED",
    "HAPG_MUTATION_TESTS",
    "HAPG_MUTATIONS_REJECTED",
    "FRESH_REPLAY_TERMINAL_COUNT",
    "FRESH_REPLAY_WAVE_COUNT",
    "FRESH_REPLAY_COMPLETE",
    "TREE_NODE_COUNT",
    "CERTIFIED_TERMINAL_COUNT",
    "UNRESOLVED_TERMINAL_COUNT",
    "UNRESOLVED_AREA_NUMERATOR",
    "UNRESOLVED_AREA_DENOMINATOR",
    "HAPG_FULL_SOURCE_COVER_CANDIDATE",
    "AGGREGATION_REQUIRED",
    "EXECUTION_PROVENANCE_ATTESTED",
    "FULL_SOURCE_CARRIER_PROVED",
    "HYPERBOLICITY_PROVED",
    "CHAOTIC_ATTRACTOR_PROVED",
    "OPEN_PROBLEM_SOLVED",
    "PROMOTION_ELIGIBLE",
)
NODE_COLUMNS = (
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "WAVE_INDEX",
    "ACTION",
    "TERMINAL_REASON",
    "WAVE_CONTRACT_SHA256",
)
EVALUATION_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "WAVE_CONTRACT_SHA256",
    "HPG_STATUS",
    "HPG_RECEIPT_SHA256",
    "HPG_VERIFICATION_SHA256",
    "HAPG_STATUS",
    "HAPG_RECEIPT_SHA256",
    "HAPG_VERIFICATION_SHA256",
    "APG_VALID",
    "APG_PASS",
    "APG_RESCUE",
    "GENERIC_CERTIFICATE_PASS",
    "DECISION",
    "TERMINAL_REASON",
)
NEGATIVE_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "HPG_STATUS",
    "HAPG_STATUS",
    "DECISION",
    "TERMINAL_REASON",
)
WAVES_COLUMNS = (
    "WAVE_INDEX",
    "FRONTIER_SHA256",
    "WAVE_CONTRACT_SHA256",
    "WAVE_RESULT_SHA256",
    "NEXT_FRONTIER_SHA256",
)
RESULT_HEADERS: tuple[tuple[str, str | None], ...] = (
    ("SCHEMA", "sounio.cs6.hapg-full-source-cover-wave-result.v1"),
    ("WAVE_INDEX", None),
    ("WAVE_CONTRACT_SHA256", None),
    ("NODE_COUNT", None),
    ("NEXT_FRONTIER_SHA256", None),
    (
        "DECISION_POLICY",
        "H_APG_ONLY_S_BIASED_BALANCED_ALL_OR_NONE_WAVE_ADMISSION",
    ),
    (
        "CAP_PRECEDENCE",
        "TIMEOUT_THEN_AXIS_DEPTH_THEN_WAVE_LIMIT_THEN_NODE_BUDGET",
    ),
)
RESULT_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "HPG_STATUS",
    "HAPG_ATTEMPTED",
    "HAPG_STATUS",
    "HAPG_RC",
    "HAPG_CHALLENGE",
    "HAPG_RECEIPT_SHA256",
    "HAPG_STDERR_SHA256",
    "HAPG_VERIFICATION_SHA256",
    "HAPG_PHYSICAL_SHA256",
    "HAPG_PROBE_PASS",
    "AFFINE_PASS",
    "PROJECTIVE_X_PASS",
    "PROJECTIVE_Y_PASS",
    "PROJECTIVE_PLUS_PASS",
    "PROJECTIVE_MINUS_PASS",
    "HOMOGENEOUS_PASS",
    "APG_VALID",
    "APG_PASS",
    "APG_RESCUE",
    "GENERIC_CERTIFICATE_PASS",
    "DECISION",
    "TERMINAL_REASON",
)
FRESH_REPLAY_CONTRACT_KEYS = (
    "SCHEMA",
    "PARENT_RUN_CONTRACT_SHA256",
    "ORIGINAL_ROOT_CHALLENGE",
    "FRESH_REPLAY_ROOT_CHALLENGE",
    "CERTIFIED_TERMINAL_COUNT",
    "CERTIFIED_TERMINALS_SHA256",
    "POLICY",
    "PROMOTION_ELIGIBLE",
)
PREBUILT_MANIFEST_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "MODE",
    "CAPD_VERSION",
    "INTERVAL_BACKEND",
    "OPTIMIZATION_LEVEL",
    "FROZEN_CONTRACT_SHA256",
    "HPG_WORKER_SOURCE_SHA256",
    "HPG_VERIFIER_SOURCE_SHA256",
    "HAPG_WORKER_SOURCE_SHA256",
    "HAPG_KERNEL_SOURCE_SHA256",
    "HAPG_VERIFIER_ADAPTER_SHA256",
    "HAPG_NUMERIC_VERIFIER_SHA256",
    "RUNNER_SHA256",
    "AGGREGATOR_SHA256",
    "KAT_ANCHOR_SHA256",
    "EXACT_TREE_KERNEL_SHA256",
    "GATE_SHA256",
    "SLURM_JOB_SCRIPT_SHA256",
    "HPG_WORKER_BINARY_SHA256",
    "HAPG_WORKER_BINARY_SHA256",
    "FILES_INDEX_SHA256",
    "FILE_COUNT",
    "PROMOTION_ELIGIBLE",
)


class AggregateError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise AggregateError(message)


def load_adjacent(name: str, filename: str) -> ModuleType:
    path = Path(__file__).resolve().with_name(filename)
    if not path.is_file():
        fail(f"missing adjacent verifier: {filename}")
    raw = path.read_bytes()
    module = ModuleType(name)
    module.__file__ = str(path)
    module.__source_sha256__ = hashlib.sha256(raw).hexdigest()
    sys.modules[name] = module
    exec(compile(raw, str(path), "exec"), module.__dict__)
    return module


LEAF_VERIFY = load_adjacent(
    "cs6_hapg_cover_aggregate_leaf_verify",
    "cs6_hapg_full_source_cover_verify.py",
)
C1 = load_adjacent(
    "cs6_hapg_cover_exact_tree_kernel",
    "cs6_c1_full_source_cover_aggregate.py",
)
KAT_ANCHOR = load_adjacent(
    "cs6_hapg_cover_aggregate_kat_anchor",
    "cs6_hapg_full_source_cover_kat_anchor.py",
)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        fail(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or len(raw) != before.st_size:
        fail(f"{label} changed while being read")
    return raw


def digest(path: Path) -> str:
    return digest_bytes(stable_bytes(path, str(path)))


def parse_int(token: str, label: str) -> int:
    if INT_RE.fullmatch(token) is None:
        fail(f"noncanonical integer: {label}")
    return int(token)


def parse_bool(token: str, label: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    fail(f"noncanonical boolean: {label}")


def validate_kat_prerequisite_certificate(
    fields: Mapping[str, str], frozen: Mapping[str, str]
) -> None:
    expected = {
        "SCHEMA": KAT_CERTIFICATE_SCHEMA,
        "CERTIFICATE_SCOPE": "AUTHORITATIVE_V6_ADAPTIVE_PREREQUISITE",
        "KAT_SCHEMA_PROFILE": "v6",
        "KAT_PREREQUISITE_VALID": "true",
        "KAT_ROOT_CHALLENGE": frozen.get("KAT_ROOT_CHALLENGE"),
        "KAT_COORDINATE_MANIFEST_SHA256": frozen.get(
            "KAT_COORDINATE_MANIFEST_SHA256"
        ),
        "KAT_EXPECTED_RESULTS_SHA256": frozen.get("KAT_EXPECTED_RESULTS_SHA256"),
        "KAT_LEAF_EVIDENCE_VALID": "true",
        "KAT_HPG_VERIFIER_REPLAY_COUNT": "52",
        "KAT_HAPG_VERIFIER_REPLAY_COUNT": "52",
        "KAT_EVALUATED_NODE_COUNT": frozen.get("KAT_EXPECTED_ATTEMPTED"),
        "KAT_HPG_SIGNED_CHART_COUNT": frozen.get("KAT_EXPECTED_H_PG_VALID"),
        "KAT_HAPG_ATTEMPTED_COUNT": frozen.get("KAT_EXPECTED_H_APG_VALID"),
        "KAT_HAPG_CERTIFIED_COUNT": frozen.get("KAT_EXPECTED_H_APG_CERTIFIED"),
        "KAT_HAPG_UNCERTIFIED_COUNT": frozen.get(
            "KAT_EXPECTED_H_APG_UNCERTIFIED"
        ),
        "KAT_HAPG_RESCUE_COUNT": frozen.get("KAT_EXPECTED_H_APG_RESCUES"),
        "KAT_HPG_MUTATION_TESTS": frozen.get("KAT_EXPECTED_HPG_MUTATION_TESTS"),
        "KAT_HPG_MUTATIONS_REJECTED": frozen.get(
            "KAT_EXPECTED_HPG_MUTATIONS_REJECTED"
        ),
        "KAT_HAPG_MUTATION_TESTS": frozen.get(
            "KAT_EXPECTED_HAPG_MUTATION_TESTS"
        ),
        "KAT_HAPG_MUTATIONS_REJECTED": frozen.get(
            "KAT_EXPECTED_HAPG_MUTATIONS_REJECTED"
        ),
        "KAT_ANCHOR_SOURCE_SHA256": frozen.get("KAT_ANCHOR_SHA256"),
        "KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT": "true",
    }
    if (
        frozen.get("KAT_PREREQUISITE_CERTIFICATE_SCHEMA")
        != KAT_CERTIFICATE_SCHEMA
        or any(fields.get(key) != value for key, value in expected.items())
        or any(
            SHA_RE.fullmatch(fields.get(key, "")) is None
            or fields.get(key) == ZERO_SHA256
            for key in ("KAT_WAVE_CONTRACT_SHA256", "KAT_WAVE_RESULT_SHA256")
        )
    ):
        fail("KAT prerequisite lacks the frozen v6 leaf-evidence bindings")


def parse_kv(raw: bytes, keys: Sequence[str], label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AggregateError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"{label} is noncanonical")
    lines = text.splitlines()
    if len(lines) != len(keys):
        fail(f"{label} field count mismatch")
    result: dict[str, str] = {}
    for line, expected in zip(lines, keys, strict=True):
        if line.count("=") != 1:
            fail(f"malformed {label} field")
        key, value = line.split("=", 1)
        if key != expected or not value:
            fail(f"{label} key mismatch: {expected}")
        result[key] = value
    return result


def read_kv(path: Path, keys: Sequence[str], label: str) -> dict[str, str]:
    return parse_kv(stable_bytes(path, label), keys, label)


def read_generic_kv(path: Path, label: str) -> dict[str, str]:
    raw = stable_bytes(path, label)
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AggregateError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"{label} is noncanonical")
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail(f"malformed {label} field")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            fail(f"duplicate or empty {label} field")
        result[key] = value
    return result


def read_table(
    path: Path,
    columns: Sequence[str],
    label: str,
    *,
    allow_empty: bool = True,
) -> list[dict[str, str]]:
    raw = stable_bytes(path, label)
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AggregateError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"{label} is noncanonical")
    lines = text.splitlines()
    if not lines or tuple(lines[0].split("\t")) != tuple(columns):
        fail(f"{label} column schema mismatch")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(columns) or any(not field for field in fields):
            fail(f"malformed {label} row")
        rows.append(dict(zip(columns, fields, strict=True)))
    if not allow_empty and not rows:
        fail(f"{label} is empty")
    return rows


def safe_file(root: Path, token: str) -> Path:
    pure = PurePosixPath(token)
    if pure.is_absolute() or ".." in pure.parts or not pure.parts:
        fail(f"unsafe bundle path: {token}")
    path = root.joinpath(*pure.parts)
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            fail(f"symlink bundle path is forbidden: {token}")
    if not path.is_file():
        fail(f"missing bundle file: {token}")
    return path


def verify_file_index(root: Path, manifest: Mapping[str, str]) -> int:
    index_path = root / "files.sha256"
    raw = stable_bytes(index_path, "files index")
    if digest_bytes(raw) != manifest["FILES_INDEX_SHA256"]:
        fail("run manifest files-index digest mismatch")
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("files index is noncanonical")
    indexed: dict[str, str] = {}
    for line in raw.decode("ascii").splitlines():
        if line.count("  ") != 1:
            fail("malformed files-index row")
        sha256, token = line.split("  ", 1)
        pure = PurePosixPath(token)
        if (
            SHA_RE.fullmatch(sha256) is None
            or pure.is_absolute()
            or ".." in pure.parts
            or not pure.parts
            or pure.as_posix() != token
            or token in indexed
        ):
            fail("unsafe or duplicate files-index row")
        indexed[token] = sha256
    if list(indexed) != sorted(indexed):
        fail("files index is not sorted")
    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail("run bundle contains a symlink")
        if path.suffix == ".pyc" or "__pycache__" in path.parts:
            fail("run bundle contains forbidden Python bytecode")
        if not path.is_file() or path in {
            root / "files.sha256",
            root / "run-manifest.txt",
        }:
            continue
        token = path.relative_to(root).as_posix()
        actual[token] = digest(path)
    if indexed != actual:
        fail("files index differs from the exact regular-file set")
    if manifest["FILE_COUNT"] != str(len(indexed)):
        fail("run manifest file count mismatch")
    return len(indexed)


def snapshot_bundle(
    source: Path,
    manifest: Mapping[str, str],
    manifest_raw: bytes,
    expected_count: int,
) -> tuple[tempfile.TemporaryDirectory[str], Path]:
    owner = tempfile.TemporaryDirectory(prefix="cs6-hapg-cover-aggregate.")
    snapshot = Path(owner.name) / "bundle"
    try:
        shutil.copytree(source, snapshot, symlinks=True)
        if stable_bytes(snapshot / "run-manifest.txt", "snapshotted run manifest") != manifest_raw:
            fail("private snapshot run manifest differs from the parsed source")
        if verify_file_index(snapshot, manifest) != expected_count:
            fail("private bundle snapshot differs from the indexed source")
        for path in sorted(snapshot.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() else 0o444)
        snapshot.chmod(0o555)
    except Exception:
        owner.cleanup()
        raise
    return owner, snapshot


@dataclass(frozen=True)
class RichNode:
    identity: str
    parent: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    wave_index: int
    action: str
    terminal_reason: str
    wave_contract_sha: str

    @property
    def area(self) -> Fraction:
        return Fraction(1, 1 << (self.u_depth + self.s_depth))


@dataclass(frozen=True)
class ParsedResult:
    headers: Mapping[str, str]
    rows: Mapping[str, Mapping[str, str]]
    sha256: str


@dataclass
class ReplayCounts:
    hpg: int = 0
    hapg: int = 0
    hpg_mutations: int = 0
    hpg_rejected: int = 0
    hapg_mutations: int = 0
    hapg_rejected: int = 0


def parse_nodes(path: Path) -> dict[str, RichNode]:
    rows = read_table(path, NODE_COLUMNS, "nodes", allow_empty=False)
    nodes: dict[str, RichNode] = {}
    for row in rows:
        coordinates = tuple(
            parse_int(row[key], key)
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        )
        u_depth, u_index, s_depth, s_index = coordinates
        if u_depth > 30 or s_depth > 30:
            fail("node exceeds worker depth contract")
        if not (u_index < 1 << u_depth and s_index < 1 << s_depth):
            fail("node index is outside its dyadic depth")
        identity = LEAF_VERIFY.canonical_leaf_id(*coordinates)
        if row["NODE_ID"] != identity or identity in nodes:
            fail("node identity is noncanonical or duplicated")
        wave_index = parse_int(row["WAVE_INDEX"], "node WAVE_INDEX")
        if wave_index != u_depth + s_depth:
            fail("node wave index differs from total dyadic depth")
        if row["ACTION"] not in {"SPLIT_U", "SPLIT_S", "CERTIFIED", "UNRESOLVED"}:
            fail("unknown node action")
        if SHA_RE.fullmatch(row["WAVE_CONTRACT_SHA256"]) is None:
            fail("node wave-contract digest is malformed")
        nodes[identity] = RichNode(
            identity,
            row["PARENT_ID"],
            u_depth,
            u_index,
            s_depth,
            s_index,
            wave_index,
            row["ACTION"],
            row["TERMINAL_REASON"],
            row["WAVE_CONTRACT_SHA256"],
        )
    if list(nodes) != sorted(nodes):
        fail("nodes are not sorted by node ID")
    return nodes


def parse_wave_result(path: Path) -> ParsedResult:
    raw = stable_bytes(path, "wave result")
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AggregateError("wave result must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail("wave result is noncanonical")
    lines = text.splitlines()
    if len(lines) < len(RESULT_HEADERS) + 1:
        fail("wave result is truncated")
    headers: dict[str, str] = {}
    for line, (expected_key, expected_value) in zip(
        lines[: len(RESULT_HEADERS)], RESULT_HEADERS, strict=True
    ):
        if line.count("=") != 1:
            fail("malformed wave-result header")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            fail(f"wave-result header mismatch: {expected_key}")
        if expected_value is not None and value != expected_value:
            fail(f"wave-result policy mismatch: {expected_key}")
        headers[key] = value
    if tuple(lines[len(RESULT_HEADERS)].split("\t")) != RESULT_COLUMNS:
        fail("wave-result column schema mismatch")
    wave_index = parse_int(headers["WAVE_INDEX"], "wave-result WAVE_INDEX")
    count = parse_int(headers["NODE_COUNT"], "wave-result NODE_COUNT")
    for key in ("WAVE_CONTRACT_SHA256", "NEXT_FRONTIER_SHA256"):
        if SHA_RE.fullmatch(headers[key]) is None:
            fail(f"wave-result digest is malformed: {key}")
    data_lines = lines[len(RESULT_HEADERS) + 1 :]
    if len(data_lines) != count or count == 0:
        fail("wave-result row count mismatch")
    rows: dict[str, Mapping[str, str]] = {}
    for line in data_lines:
        fields = line.split("\t")
        if len(fields) != len(RESULT_COLUMNS) or any(not field for field in fields):
            fail("malformed wave-result row")
        values = dict(zip(RESULT_COLUMNS, fields, strict=True))
        if parse_int(values["WAVE_INDEX"], "result row WAVE_INDEX") != wave_index:
            fail("wave-result row wave mismatch")
        identity = values["NODE_ID"]
        if identity in rows:
            fail("duplicate wave-result node")
        for key in (
            "HAPG_CHALLENGE",
            "HAPG_RECEIPT_SHA256",
            "HAPG_STDERR_SHA256",
            "HAPG_VERIFICATION_SHA256",
            "HAPG_PHYSICAL_SHA256",
        ):
            if SHA_RE.fullmatch(values[key]) is None:
                fail(f"malformed wave-result digest: {key}")
        parse_int(values["HAPG_RC"], "HAPG_RC")
        for key in RESULT_COLUMNS[11:22]:
            parse_bool(values[key], key)
        rows[identity] = values
    if list(rows) != sorted(rows):
        fail("wave-result rows are not sorted")
    return ParsedResult(headers, rows, digest_bytes(raw))


@dataclass(frozen=True)
class ExpectedLeaf:
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    parent: str
    wave_index: int

    @property
    def identity(self) -> str:
        return LEAF_VERIFY.canonical_leaf_id(
            self.u_depth, self.u_index, self.s_depth, self.s_index
        )


def leaf_input_bytes(leaf: ExpectedLeaf) -> bytes:
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={leaf.u_depth}\n"
        f"U_INDEX={leaf.u_index}\n"
        f"S_DEPTH={leaf.s_depth}\n"
        f"S_INDEX={leaf.s_index}\n"
    ).encode("ascii")


def frontier_bytes(leaves: Sequence[ExpectedLeaf]) -> bytes:
    rows = ["NODE_ID\tPARENT_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256"]
    for leaf in sorted(leaves, key=lambda item: item.identity):
        rows.append(
            "\t".join(
                (
                    leaf.identity,
                    leaf.parent,
                    str(leaf.u_depth),
                    str(leaf.u_index),
                    str(leaf.s_depth),
                    str(leaf.s_index),
                    digest_bytes(leaf_input_bytes(leaf)),
                )
            )
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def split_leaf(leaf: ExpectedLeaf) -> tuple[str, tuple[ExpectedLeaf, ExpectedLeaf]]:
    next_wave = leaf.wave_index + 1
    if leaf.s_depth <= leaf.u_depth:
        return (
            "SPLIT_S",
            (
                ExpectedLeaf(
                    leaf.u_depth,
                    leaf.u_index,
                    leaf.s_depth + 1,
                    2 * leaf.s_index,
                    leaf.identity,
                    next_wave,
                ),
                ExpectedLeaf(
                    leaf.u_depth,
                    leaf.u_index,
                    leaf.s_depth + 1,
                    2 * leaf.s_index + 1,
                    leaf.identity,
                    next_wave,
                ),
            ),
        )
    return (
        "SPLIT_U",
        (
            ExpectedLeaf(
                leaf.u_depth + 1,
                2 * leaf.u_index,
                leaf.s_depth,
                leaf.s_index,
                leaf.identity,
                next_wave,
            ),
            ExpectedLeaf(
                leaf.u_depth + 1,
                2 * leaf.u_index + 1,
                leaf.s_depth,
                leaf.s_index,
                leaf.identity,
                next_wave,
            ),
        ),
    )


def classify_failure(stderr: bytes, prefix: str) -> str | None:
    lowered = stderr.lower()
    if prefix == "H_PG" and b"interval error:" in lowered and (
        b"division by 0" in lowered or b"division by zero" in lowered
    ):
        return "H_PG_INTERVAL_DOMAIN"
    if prefix == "H_PG" and (
        b"one-step newton crossing was not available" in lowered
        or (
            lowered.startswith(
                b"probe error: poincaremap error: possible nontransversal return to the section"
            )
            and b"\ninner product of vector field and section gradient: [" in lowered
        )
    ):
        return "H_PG_CROSSING"
    if prefix == "H_PG" and (
        b"centeredtripletonset::evalaffinefunctional - empty intersection" in lowered
        and b"rq=[-nan, -nan]" in lowered
    ):
        return "H_PG_CAPD_SET"
    if prefix == "H_APG" and any(
        marker in lowered
        for marker in (
            b"frozen tm2 pivot sign was not certified",
            b"tm2 reciprocal pivot contains zero",
            b"tm2 reciprocal center contains zero",
            b"tm2 reciprocal is noncontractive",
        )
    ):
        return "H_APG_FROZEN_CHART"
    return None


def verification_values(raw: bytes, label: str) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise AggregateError(f"{label} must be ASCII") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"{label} is noncanonical")
    values: dict[str, str] = {}
    for line in text.splitlines():
        if line.count("=") != 1:
            fail(f"malformed {label} line")
        key, value = line.split("=", 1)
        if not key or not value or key in values:
            fail(f"duplicate or empty {label} field")
        values[key] = value
    return values


def run_exact_verifier(
    command: Sequence[str], expected: Path, timeout: int, label: str
) -> dict[str, str]:
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode != 0 or result.stderr:
        fail(f"{label} replay failed with rc={result.returncode}")
    expected_raw = stable_bytes(expected, f"stored {label}")
    if result.stdout != expected_raw:
        fail(f"{label} replay differs byte-for-byte from stored verification")
    return verification_values(result.stdout, label)


def verify_leaf_artifacts(
    root: Path,
    source_root: Path,
    wave_path: Path,
    wave: object,
    leaf: ExpectedLeaf,
    result: Mapping[str, str],
    hpg_source_sha: str,
    hapg_source_sha: str,
    root_challenge: str,
    mutation_audit: bool,
    timeout: int,
) -> ReplayCounts:
    identity = leaf.identity
    row = wave.rows[identity].values
    input_path = safe_file(root, f"inputs/{identity}.txt")
    input_raw = stable_bytes(input_path, "leaf input")
    if input_raw != leaf_input_bytes(leaf) or digest_bytes(input_raw) != row["INPUT_SHA256"]:
        fail(f"leaf input binding mismatch: {identity}")

    hpg_receipt = safe_file(root, f"hpg-receipts/{identity}.txt")
    hpg_stderr = safe_file(root, f"hpg-stderr/{identity}.txt")
    if digest(hpg_receipt) != row["HPG_RECEIPT_SHA256"]:
        fail(f"H-PG receipt digest mismatch: {identity}")
    if digest(hpg_stderr) != row["HPG_STDERR_SHA256"]:
        fail(f"H-PG stderr digest mismatch: {identity}")
    hpg_rc = parse_int(row["HPG_RC"], "HPG_RC")
    counts = ReplayCounts()
    if hpg_rc == 0:
        if digest(hpg_stderr) != EMPTY_SHA256:
            fail(f"successful H-PG attempt emitted stderr: {identity}")
        if row["HPG_VERIFICATION_SHA256"] == ZERO_SHA256:
            fail(f"successful H-PG attempt lacks verification: {identity}")
        hpg_verification = safe_file(root, f"hpg-verifications/{identity}.txt")
        if digest(hpg_verification) != row["HPG_VERIFICATION_SHA256"]:
            fail(f"H-PG verification digest mismatch: {identity}")
        command = [
            sys.executable,
            "-B",
            str(safe_file(source_root, "cs6_plucker_cocycle_verify.py")),
            str(hpg_receipt),
            "--source-sha",
            hpg_source_sha,
            "--input",
            str(input_path),
            "--challenge",
            row["HPG_CHALLENGE"],
        ]
        if mutation_audit:
            command.append("--self-test-mutations")
        values = run_exact_verifier(command, hpg_verification, timeout, "H-PG verifier")
        if (
            values["VERIFICATION_SCHEMA"]
            != "sounio.cs6.plucker-cocycle-leaf-verification.v1"
            or values["RECEIPT_SHA256"] != row["HPG_RECEIPT_SHA256"]
        ):
            fail(f"H-PG verification binding mismatch: {identity}")
        counts.hpg += 1
        counts.hpg_mutations += parse_int(values["MUTATION_TESTS"], "HPG mutations")
        counts.hpg_rejected += parse_int(values["MUTATIONS_REJECTED"], "HPG rejected")
        if mutation_audit and (
            counts.hpg_mutations == 0 or counts.hpg_mutations != counts.hpg_rejected
        ):
            fail(f"H-PG mutation audit mismatch: {identity}")
        ledger = LEAF_VERIFY.HPG_CORE.parse_ledger(hpg_receipt)
        chart_signs: list[tuple[str, int]] = []
        signed = True
        for marker in LEAF_VERIFY.CHART_MARKERS:
            record = ledger.records[marker]
            chart = LEAF_VERIFY.HPG_CORE.string_value(record, "CHART")
            pivot = LEAF_VERIFY.HPG_CORE.interval(record, "PIVOT")
            sign = -1 if pivot.upper < 0 else 1 if pivot.lower > 0 else 0
            if chart not in LEAF_VERIFY.HAPG_CORE.FULL53_CHARTS or sign == 0:
                signed = False
            chart_signs.append((chart, sign))
        probe = parse_bool(values["PROBE_PASS"], "HPG PROBE_PASS")
        eligible = probe and signed
        if not eligible:
            chart_signs = [("NONE", 0)] * 4
        expected_hpg = {
            "HPG_STATUS": (
                LEAF_VERIFY.SIGNED_CHART_STATUS
                if eligible
                else "H_PG_INVALID_NO_SIGNED_CHART"
            ),
            "HPG_PHYSICAL_SHA256": values["PHYSICAL_SHA256"],
            "HPG_PROBE_PASS": values["PROBE_PASS"],
            "HPG_CERTIFICATE_PASS": values["CERTIFICATE_PASS"],
            "HAPG_ELIGIBLE": "true" if eligible else "false",
        }
        for (event, ray), (chart, sign) in zip(
            ((1, 0), (1, 1), (2, 0), (2, 1)), chart_signs, strict=True
        ):
            expected_hpg[f"E{event}_R{ray}_CHART"] = chart
            expected_hpg[f"E{event}_R{ray}_SIGN"] = str(sign)
        if any(row[key] != value for key, value in expected_hpg.items()):
            fail(f"wave contract differs from H-PG recomputation: {identity}")
    else:
        failure_class = (
            "H_PG_TIMEOUT"
            if hpg_rc == 124 and row["HPG_STATUS"] == "H_PG_TIMEOUT"
            else classify_failure(stable_bytes(hpg_stderr, "H-PG stderr"), "H_PG")
        )
        if (
            row["HPG_VERIFICATION_SHA256"] != ZERO_SHA256
            or row["HPG_PHYSICAL_SHA256"] != ZERO_SHA256
            or failure_class != row["HPG_STATUS"]
        ):
            fail(f"H-PG failure classification mismatch: {identity}")

    eligible = parse_bool(row["HAPG_ELIGIBLE"], "HAPG_ELIGIBLE")
    attempted = parse_bool(result["HAPG_ATTEMPTED"], "HAPG_ATTEMPTED")
    if attempted != eligible:
        fail(f"H-APG attempt differs from frozen eligibility: {identity}")
    if result["HPG_STATUS"] != row["HPG_STATUS"]:
        fail(f"wave result changed H-PG status: {identity}")
    boolean_keys = RESULT_COLUMNS[11:22]
    if not attempted:
        if (
            result["HAPG_STATUS"] != "H_APG_NOT_ELIGIBLE"
            or result["HAPG_RC"] != "0"
            or result["HAPG_CHALLENGE"] != ZERO_SHA256
            or result["HAPG_RECEIPT_SHA256"] != EMPTY_SHA256
            or result["HAPG_STDERR_SHA256"] != EMPTY_SHA256
            or result["HAPG_VERIFICATION_SHA256"] != ZERO_SHA256
            or result["HAPG_PHYSICAL_SHA256"] != ZERO_SHA256
            or any(parse_bool(result[key], key) for key in boolean_keys)
        ):
            fail(f"ineligible H-APG sentinel mismatch: {identity}")
        return counts

    contract = LEAF_VERIFY.HAPG_CORE.Full53LeafContract(
        leaf_id=identity,
        u_depth=leaf.u_depth,
        u_index=leaf.u_index,
        s_depth=leaf.s_depth,
        s_index=leaf.s_index,
        parent_input_sha256=row["INPUT_SHA256"],
        parent_status=row["HPG_STATUS"],
        parent_receipt_sha256=row["HPG_RECEIPT_SHA256"],
        chart_signs=wave.rows[identity].chart_signs,
        manifest_sha256=wave.sha256,
    )
    challenge = LEAF_VERIFY.HAPG_CORE.full53_leaf_challenge(root_challenge, contract)
    if result["HAPG_CHALLENGE"] != challenge:
        fail(f"H-APG challenge mismatch: {identity}")
    hapg_receipt = safe_file(root, f"hapg-receipts/{identity}.txt")
    hapg_stderr = safe_file(root, f"hapg-stderr/{identity}.txt")
    if digest(hapg_receipt) != result["HAPG_RECEIPT_SHA256"]:
        fail(f"H-APG receipt digest mismatch: {identity}")
    if digest(hapg_stderr) != result["HAPG_STDERR_SHA256"]:
        fail(f"H-APG stderr digest mismatch: {identity}")
    hapg_rc = parse_int(result["HAPG_RC"], "HAPG_RC")
    if hapg_rc != 0:
        failure_class = (
            "H_APG_TIMEOUT"
            if hapg_rc == 124 and result["HAPG_STATUS"] == "H_APG_TIMEOUT"
            else classify_failure(stable_bytes(hapg_stderr, "H-APG stderr"), "H_APG")
        )
        if (
            result["HAPG_VERIFICATION_SHA256"] != ZERO_SHA256
            or result["HAPG_PHYSICAL_SHA256"] != ZERO_SHA256
            or any(parse_bool(result[key], key) for key in boolean_keys)
            or failure_class != result["HAPG_STATUS"]
        ):
            fail(f"H-APG failure classification mismatch: {identity}")
        return counts

    hapg_verification = safe_file(root, f"hapg-verifications/{identity}.txt")
    if digest(hapg_verification) != result["HAPG_VERIFICATION_SHA256"]:
        fail(f"H-APG verification digest mismatch: {identity}")
    command = [
        sys.executable,
        "-B",
        str(safe_file(source_root, "cs6_hapg_full_source_cover_verify.py")),
        str(hapg_receipt),
        "--hapg-source-sha",
        hapg_source_sha,
        "--hpg-source-sha",
        hpg_source_sha,
        "--input",
        str(input_path),
        "--wave-contract",
        str(wave_path),
        "--hpg-receipt",
        str(hpg_receipt),
        "--hpg-verification",
        str(safe_file(root, f"hpg-verifications/{identity}.txt")),
        "--root-challenge",
        root_challenge,
    ]
    if mutation_audit:
        command.append("--self-test-mutations")
    values = run_exact_verifier(command, hapg_verification, timeout, "H-APG adapter")
    counts.hapg += 1
    counts.hapg_mutations += parse_int(values["MUTATION_TESTS"], "HAPG mutations")
    counts.hapg_rejected += parse_int(values["MUTATIONS_REJECTED"], "HAPG rejected")
    if mutation_audit and (
        counts.hapg_mutations == 0 or counts.hapg_mutations != counts.hapg_rejected
    ):
        fail(f"H-APG mutation audit mismatch: {identity}")
    expected_values = {
        "HAPG_PHYSICAL_SHA256": values["PHYSICAL_SHA256"],
        "HAPG_PROBE_PASS": values["PROBE_PASS"],
        "AFFINE_PASS": values["AFFINE_CERTIFICATE_PASS"],
        "PROJECTIVE_X_PASS": values["PROJECTIVE_X_CERTIFICATE_PASS"],
        "PROJECTIVE_Y_PASS": values["PROJECTIVE_Y_CERTIFICATE_PASS"],
        "PROJECTIVE_PLUS_PASS": values["PROJECTIVE_PLUS_CERTIFICATE_PASS"],
        "PROJECTIVE_MINUS_PASS": values["PROJECTIVE_MINUS_CERTIFICATE_PASS"],
        "HOMOGENEOUS_PASS": values["HOMOGENEOUS_CERTIFICATE_PASS"],
        "APG_VALID": values["APG_COMPUTATION_VALID"],
        "APG_PASS": values["APG_CERTIFICATE_PASS"],
        "APG_RESCUE": values["APG_RESCUE"],
        "GENERIC_CERTIFICATE_PASS": values["GENERIC_CERTIFICATE_PASS"],
    }
    if any(result[key] != value for key, value in expected_values.items()):
        fail(f"wave result differs from H-APG recomputation: {identity}")
    terminal = parse_bool(values["HAPG_TERMINAL_CERTIFIED"], "HAPG terminal")
    apg_valid = parse_bool(values["APG_COMPUTATION_VALID"], "APG valid")
    apg_pass = parse_bool(values["APG_CERTIFICATE_PASS"], "APG pass")
    expected_status = (
        "H_APG_CERTIFIED"
        if terminal
        else "H_APG_UNCERTIFIED"
        if apg_valid
        else "H_APG_INVALID"
    )
    if terminal != (apg_valid and apg_pass) or result["HAPG_STATUS"] != expected_status:
        fail(f"H-APG terminal/status predicate mismatch: {identity}")
    return counts


def add_counts(total: ReplayCounts, delta: ReplayCounts) -> None:
    total.hpg += delta.hpg
    total.hapg += delta.hapg
    total.hpg_mutations += delta.hpg_mutations
    total.hpg_rejected += delta.hpg_rejected
    total.hapg_mutations += delta.hapg_mutations
    total.hapg_rejected += delta.hapg_rejected


def evaluation_projection(
    leaf: ExpectedLeaf,
    contract_sha: str,
    hpg: Mapping[str, str],
    result: Mapping[str, str],
) -> dict[str, str]:
    return {
        "WAVE_INDEX": str(leaf.wave_index),
        "NODE_ID": leaf.identity,
        "PARENT_ID": leaf.parent,
        "U_DEPTH": str(leaf.u_depth),
        "U_INDEX": str(leaf.u_index),
        "S_DEPTH": str(leaf.s_depth),
        "S_INDEX": str(leaf.s_index),
        "WAVE_CONTRACT_SHA256": contract_sha,
        "HPG_STATUS": hpg["HPG_STATUS"],
        "HPG_RECEIPT_SHA256": hpg["HPG_RECEIPT_SHA256"],
        "HPG_VERIFICATION_SHA256": hpg["HPG_VERIFICATION_SHA256"],
        "HAPG_STATUS": result["HAPG_STATUS"],
        "HAPG_RECEIPT_SHA256": result["HAPG_RECEIPT_SHA256"],
        "HAPG_VERIFICATION_SHA256": result["HAPG_VERIFICATION_SHA256"],
        "APG_VALID": result["APG_VALID"],
        "APG_PASS": result["APG_PASS"],
        "APG_RESCUE": result["APG_RESCUE"],
        "GENERIC_CERTIFICATE_PASS": result["GENERIC_CERTIFICATE_PASS"],
        "DECISION": result["DECISION"],
        "TERMINAL_REASON": result["TERMINAL_REASON"],
    }


def negative_projection(
    leaf: ExpectedLeaf, hpg: Mapping[str, str], result: Mapping[str, str]
) -> dict[str, str]:
    return {
        "WAVE_INDEX": str(leaf.wave_index),
        "NODE_ID": leaf.identity,
        "HPG_STATUS": hpg["HPG_STATUS"],
        "HAPG_STATUS": result["HAPG_STATUS"],
        "DECISION": result["DECISION"],
        "TERMINAL_REASON": result["TERMINAL_REASON"],
    }


def verify_frozen_sources(
    bundle: Path,
    run_contract: Mapping[str, str],
    expected_contract_sha: str,
) -> Mapping[str, str]:
    frozen_path = safe_file(bundle, "cs6_hapg_full_source_cover_contract_v6.txt")
    frozen = read_generic_kv(frozen_path, "frozen v6 contract")
    if (
        frozen.get("SCHEMA") != "sounio.cs6.hapg-full-source-cover-contract.v6"
        or frozen.get("CONTRACT_STATE") != "PRE_RESULT_FROZEN"
        or frozen.get("SUPERSEDES_V4_SHA256")
        != "a308b4f0d32b4179ed17f1ffd7bbd4827fa81d9cb66162318ddecbc926a43293"
        or frozen.get("SUPERSEDES_V3_SHA256")
        != "3e5f1c560356771e9d33582cab31b9776cf6f21d4eabcbc6e292523a2e9010e2"
        or frozen.get("V2_ABORTED_SLURM_JOB_ID") != "8451"
        or frozen.get("V2_ABORT_SCIENTIFIC_EVALUATIONS") != "0"
        or digest(frozen_path) != run_contract["FROZEN_CONTRACT_SHA256"]
        or digest(frozen_path) != expected_contract_sha
    ):
        fail("frozen v6 contract envelope mismatch")
    abort_bindings = {
        "SUPERSEDES_V5_SHA256": "v5-executed-contract.txt",
        "SUPERSEDES_V4_SHA256": "v4-executed-contract.txt",
        "SUPERSEDES_V3_SHA256": "v3-executed-contract.txt",
        "V2_ABORT_RECEIPT_MANIFEST_SHA256": "v2-abort-manifest.txt",
        "V2_ABORT_SACCT_SHA256": "v2-abort-sacct.txt",
        "V2_ABORT_CONFIG_SHA256": "v2-abort-config.txt",
        "V2_ABORT_STDERR_SHA256": "v2-abort-stderr.txt",
        "V3_ABORT_RECEIPT_MANIFEST_SHA256": "v3-abort-manifest.txt",
        "V3_ABORT_SACCT_SHA256": "v3-abort-sacct.txt",
        "V3_ABORT_CONFIG_SHA256": "v3-abort-config.txt",
        "V3_ABORT_SLURM_STDERR_SHA256": "v3-abort-slurm-stderr.txt",
        "V3_ABORT_REPRO_S0_STDOUT_SHA256": "v3-abort-repro-s0-stdout.txt",
        "V3_ABORT_REPRO_S0_STDERR_SHA256": "v3-abort-repro-s0-stderr.txt",
        "V3_ABORT_REPRO_S1_STDOUT_SHA256": "v3-abort-repro-s1-stdout.txt",
        "V3_ABORT_REPRO_S1_STDERR_SHA256": "v3-abort-repro-s1-stderr.txt",
        "V3_ABORT_HPG_FULL255_CENSUS_SHA256": "v3-abort-hpg-full255-census.tsv",
        "V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256": "v3-abort-hpg-full255-census-summary.txt",
        "V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256": "v3-abort-hpg-full255-stderr.jsonl",
        "V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256": "v3-abort-challenge-spotcheck.json",
        "V4_ABORT_RECEIPT_MANIFEST_SHA256": "v4-abort-manifest.txt",
        "V4_ABORT_FILES_INDEX_SHA256": "v4-abort-files.sha256",
        "V4_ABORT_SACCT_SHA256": "v4-abort-sacct.txt",
        "V4_ABORT_CONFIG_SHA256": "v4-abort-config.txt",
        "V4_ABORT_SLURM_STDOUT_SHA256": "v4-abort-slurm-stdout.txt",
        "V4_ABORT_HPG_RC0_CORPUS_SHA256": "v4-abort-hpg-rc0-corpus.tar",
        "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256": "v4-abort-corpus-files.sha256",
        "V4_ABORT_HPG_RC0_CENSUS_SHA256": "v4-abort-hpg-rc0-verifier-census.tsv",
        "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256": "v4-abort-hpg-rc0-verifier-census-summary.txt",
        "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256": "v4-abort-hpg-v5-kat-compat.tsv",
        "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256": "v4-abort-hpg-v4-kat-corpus.tar",
        "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256": "v4-abort-hpg-v4-kat-corpus-files.sha256",
        "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256": "v4-abort-midpoint-discrete-negative-test.txt",
        "V4_ABORT_LOCAL_REPRO_SHA256": "v4-abort-local-repro.tar",
        "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256": "v4-abort-v4-hpg-verifier.py",
        "V5_ABORT_RECEIPT_MANIFEST_SHA256": "v5-abort-manifest.txt",
        "V5_ABORT_FILES_INDEX_SHA256": "v5-abort-files.sha256",
        "V5_ABORT_JOBS_SACCT_SHA256": "v5-abort-sacct.psv",
    }
    for key, filename in abort_bindings.items():
        if frozen.get(key) != digest(safe_file(bundle, filename)):
            fail(f"frozen abort evidence mismatch: {filename}")
    bindings = (
        ("PREPASS_WORKER_SHA256", "cs6_plucker_cocycle_probe.cpp", "HPG_WORKER_SOURCE_SHA256"),
        ("PREPASS_VERIFIER_SHA256", "cs6_plucker_cocycle_verify.py", "HPG_VERIFIER_SOURCE_SHA256"),
        ("H_APG_WRAPPER_SHA256", "cs6_hapg_full_source_cover_worker.cpp", "HAPG_WORKER_SOURCE_SHA256"),
        ("H_APG_KERNEL_SHA256", "cs6_affine_projective_cocycle_full53_probe.cpp", "HAPG_KERNEL_SOURCE_SHA256"),
        ("H_APG_ADAPTER_SHA256", "cs6_hapg_full_source_cover_verify.py", "HAPG_VERIFIER_ADAPTER_SHA256"),
        ("H_APG_NUMERIC_VERIFIER_SHA256", "cs6_affine_projective_cocycle_full53_verify.py", "HAPG_NUMERIC_VERIFIER_SHA256"),
    )
    for frozen_key, filename, run_key in bindings:
        value = digest(safe_file(bundle, filename))
        if frozen.get(frozen_key) != value or run_contract[run_key] != value:
            fail(f"frozen source binding mismatch: {filename}")
    executed_bindings = (
        (
            "cs6_plucker_cocycle_verify.py",
            Path(LEAF_VERIFY.HPG_CORE.__file__).resolve(),
            LEAF_VERIFY.HPG_CORE,
        ),
        (
            "cs6_hapg_full_source_cover_verify.py",
            Path(LEAF_VERIFY.__file__).resolve(),
            LEAF_VERIFY,
        ),
        (
            "cs6_affine_projective_cocycle_full53_verify.py",
            Path(LEAF_VERIFY.HAPG_CORE.__file__).resolve(),
            LEAF_VERIFY.HAPG_CORE,
        ),
        (
            "cs6_affine_projective_cocycle_full53_probe.cpp",
            Path(LEAF_VERIFY.__file__).resolve().with_name(
                "cs6_affine_projective_cocycle_full53_probe.cpp"
            ),
            None,
        ),
    )
    for filename, executed, module in executed_bindings:
        bundled_sha = digest(safe_file(bundle, filename))
        if digest(executed) != bundled_sha or (
            module is not None
            and getattr(module, "__source_sha256__", None) != bundled_sha
        ):
            fail(f"executed verifier dependency differs from bundle: {filename}")
    if frozen.get("RUNNER_SHA256") != digest(
        safe_file(bundle, "cs6_hapg_full_source_cover_run.py")
    ):
        fail("frozen runner source binding mismatch")
    bundled_aggregator = safe_file(bundle, "cs6_hapg_full_source_cover_aggregate.py")
    if (
        frozen.get("AGGREGATOR_SHA256") != digest(bundled_aggregator)
        or digest(bundled_aggregator) != digest(Path(__file__).resolve())
        or globals().get("__source_sha256__") not in {None, digest(bundled_aggregator)}
    ):
        fail("executed aggregator differs from the frozen contract")
    bundled_kat_anchor = safe_file(bundle, "cs6_hapg_full_source_cover_kat_anchor.py")
    if (
        frozen.get("KAT_ANCHOR_SHA256") != digest(bundled_kat_anchor)
        or run_contract["KAT_ANCHOR_SHA256"] != digest(bundled_kat_anchor)
        or digest(bundled_kat_anchor) != digest(Path(KAT_ANCHOR.__file__).resolve())
        or getattr(KAT_ANCHOR, "__source_sha256__", None) != digest(bundled_kat_anchor)
    ):
        fail("executed KAT anchor validator differs from the frozen contract")
    bundled_tree = safe_file(bundle, "cs6_c1_full_source_cover_aggregate.py")
    if (
        frozen.get("EXACT_TREE_KERNEL_SHA256") != digest(bundled_tree)
        or digest(bundled_tree) != digest(Path(C1.__file__).resolve())
        or getattr(C1, "__source_sha256__", None) != digest(bundled_tree)
    ):
        fail("exact tree kernel differs from the frozen contract")
    if frozen.get("GATE_SHA256") != digest(
        safe_file(bundle, "cs6_hapg_full_source_cover_gate.sh")
    ):
        fail("frozen gate source binding mismatch")
    if (
        frozen.get("SLURM_JOB_SCRIPT_SHA256")
        != run_contract["SLURM_JOB_SCRIPT_SHA256"]
        or frozen.get("SLURM_JOB_SCRIPT_SHA256")
        != digest(safe_file(bundle, "cs6_hapg_full_source_cover_slurm_job.sh"))
    ):
        fail("frozen Slurm job script binding mismatch")
    if (
        frozen.get("WAVE_CONTRACT_CHAIN")
        != "SHA256_PREVIOUS_WAVE_RESULT_AND_EXACT_NEXT_FRONTIER"
        or frozen.get("FRESH_REPLAY_SEMANTICS")
        != "INDEPENDENT_RECERTIFICATION_SAME_CHARTS_DISTINCT_CHALLENGES_NOT_BITWISE_RECEIPT_REPRODUCTION"
        or frozen.get("BOUNDED_PILOT_REPLAY_ROOT_CHALLENGE")
        != run_contract["FRESH_REPLAY_ROOT_CHALLENGE"]
    ):
        fail("frozen causal-chain or replay policy mismatch")
    return frozen


def verify_prebuilt_origin(
    bundle: Path,
    run_contract: Mapping[str, str],
    frozen: Mapping[str, str],
) -> None:
    origin = bundle / "prebuilt-origin"
    if origin.is_symlink() or not origin.is_dir():
        fail("prebuilt origin directory is missing or unsafe")
    manifest_path = safe_file(origin, "run-manifest.txt")
    manifest = read_kv(manifest_path, PREBUILT_MANIFEST_KEYS, "prebuilt manifest")
    if (
        manifest["SCHEMA"] != "sounio.cs6.hapg-full-source-cover-prebuilt.v2"
        or manifest["RUN_COMPLETE"] != "true"
        or manifest["MODE"] != "prepare"
        or manifest["CAPD_VERSION"] != "5.3.0"
        or manifest["INTERVAL_BACKEND"] != "FILIB"
        or manifest["OPTIMIZATION_LEVEL"] != "O0"
        or manifest["PROMOTION_ELIGIBLE"] != "false"
        or digest(manifest_path) != run_contract["PREBUILT_RUN_MANIFEST_SHA256"]
    ):
        fail("prebuilt origin manifest mismatch")
    verify_file_index(origin, manifest)
    bindings = {
        "FROZEN_CONTRACT_SHA256": "cs6_hapg_full_source_cover_contract_v6.txt",
        "HPG_WORKER_SOURCE_SHA256": "cs6_plucker_cocycle_probe.cpp",
        "HPG_VERIFIER_SOURCE_SHA256": "cs6_plucker_cocycle_verify.py",
        "HAPG_WORKER_SOURCE_SHA256": "cs6_hapg_full_source_cover_worker.cpp",
        "HAPG_KERNEL_SOURCE_SHA256": "cs6_affine_projective_cocycle_full53_probe.cpp",
        "HAPG_VERIFIER_ADAPTER_SHA256": "cs6_hapg_full_source_cover_verify.py",
        "HAPG_NUMERIC_VERIFIER_SHA256": "cs6_affine_projective_cocycle_full53_verify.py",
        "RUNNER_SHA256": "cs6_hapg_full_source_cover_run.py",
        "AGGREGATOR_SHA256": "cs6_hapg_full_source_cover_aggregate.py",
        "KAT_ANCHOR_SHA256": "cs6_hapg_full_source_cover_kat_anchor.py",
        "EXACT_TREE_KERNEL_SHA256": "cs6_c1_full_source_cover_aggregate.py",
        "GATE_SHA256": "cs6_hapg_full_source_cover_gate.sh",
        "SLURM_JOB_SCRIPT_SHA256": "cs6_hapg_full_source_cover_slurm_job.sh",
        "HPG_WORKER_BINARY_SHA256": "hpg-worker-binary",
        "HAPG_WORKER_BINARY_SHA256": "hapg-worker-binary",
    }
    for key, filename in bindings.items():
        if manifest[key] != digest(safe_file(origin, filename)):
            fail(f"prebuilt origin declaration differs from bytes: {filename}")
    abort_bindings = {
        "SUPERSEDES_V5_SHA256": "v5-executed-contract.txt",
        "SUPERSEDES_V4_SHA256": "v4-executed-contract.txt",
        "SUPERSEDES_V3_SHA256": "v3-executed-contract.txt",
        "V2_ABORT_RECEIPT_MANIFEST_SHA256": "v2-abort-manifest.txt",
        "V2_ABORT_SACCT_SHA256": "v2-abort-sacct.txt",
        "V2_ABORT_CONFIG_SHA256": "v2-abort-config.txt",
        "V2_ABORT_STDERR_SHA256": "v2-abort-stderr.txt",
        "V3_ABORT_RECEIPT_MANIFEST_SHA256": "v3-abort-manifest.txt",
        "V3_ABORT_SACCT_SHA256": "v3-abort-sacct.txt",
        "V3_ABORT_CONFIG_SHA256": "v3-abort-config.txt",
        "V3_ABORT_SLURM_STDERR_SHA256": "v3-abort-slurm-stderr.txt",
        "V3_ABORT_REPRO_S0_STDOUT_SHA256": "v3-abort-repro-s0-stdout.txt",
        "V3_ABORT_REPRO_S0_STDERR_SHA256": "v3-abort-repro-s0-stderr.txt",
        "V3_ABORT_REPRO_S1_STDOUT_SHA256": "v3-abort-repro-s1-stdout.txt",
        "V3_ABORT_REPRO_S1_STDERR_SHA256": "v3-abort-repro-s1-stderr.txt",
        "V3_ABORT_HPG_FULL255_CENSUS_SHA256": "v3-abort-hpg-full255-census.tsv",
        "V3_ABORT_HPG_FULL255_CENSUS_SUMMARY_SHA256": "v3-abort-hpg-full255-census-summary.txt",
        "V3_ABORT_HPG_FULL255_STDERR_JSONL_SHA256": "v3-abort-hpg-full255-stderr.jsonl",
        "V3_ABORT_HPG_CHALLENGE_SPOTCHECK_SHA256": "v3-abort-challenge-spotcheck.json",
        "V4_ABORT_RECEIPT_MANIFEST_SHA256": "v4-abort-manifest.txt",
        "V4_ABORT_FILES_INDEX_SHA256": "v4-abort-files.sha256",
        "V4_ABORT_SACCT_SHA256": "v4-abort-sacct.txt",
        "V4_ABORT_CONFIG_SHA256": "v4-abort-config.txt",
        "V4_ABORT_SLURM_STDOUT_SHA256": "v4-abort-slurm-stdout.txt",
        "V4_ABORT_HPG_RC0_CORPUS_SHA256": "v4-abort-hpg-rc0-corpus.tar",
        "V4_ABORT_HPG_RC0_CORPUS_FILES_SHA256": "v4-abort-corpus-files.sha256",
        "V4_ABORT_HPG_RC0_CENSUS_SHA256": "v4-abort-hpg-rc0-verifier-census.tsv",
        "V4_ABORT_HPG_RC0_CENSUS_SUMMARY_SHA256": "v4-abort-hpg-rc0-verifier-census-summary.txt",
        "V4_ABORT_HPG_V5_KAT_COMPAT_SHA256": "v4-abort-hpg-v5-kat-compat.tsv",
        "V4_ABORT_HPG_V4_KAT_CORPUS_SHA256": "v4-abort-hpg-v4-kat-corpus.tar",
        "V4_ABORT_HPG_V4_KAT_CORPUS_FILES_SHA256": "v4-abort-hpg-v4-kat-corpus-files.sha256",
        "V4_ABORT_MIDPOINT_DISCRETE_TEST_SHA256": "v4-abort-midpoint-discrete-negative-test.txt",
        "V4_ABORT_LOCAL_REPRO_SHA256": "v4-abort-local-repro.tar",
        "V4_ABORT_EXECUTED_HPG_VERIFIER_SHA256": "v4-abort-v4-hpg-verifier.py",
        "V5_ABORT_RECEIPT_MANIFEST_SHA256": "v5-abort-manifest.txt",
        "V5_ABORT_FILES_INDEX_SHA256": "v5-abort-files.sha256",
        "V5_ABORT_JOBS_SACCT_SHA256": "v5-abort-sacct.psv",
    }
    for key, filename in abort_bindings.items():
        if frozen.get(key) != digest(safe_file(origin, filename)):
            fail(f"prebuilt origin abort evidence mismatch: {filename}")
    run_source_fields = (
        "HPG_WORKER_SOURCE_SHA256",
        "HPG_VERIFIER_SOURCE_SHA256",
        "HAPG_WORKER_SOURCE_SHA256",
        "HAPG_KERNEL_SOURCE_SHA256",
        "HAPG_VERIFIER_ADAPTER_SHA256",
        "HAPG_NUMERIC_VERIFIER_SHA256",
        "KAT_ANCHOR_SHA256",
        "SLURM_JOB_SCRIPT_SHA256",
    )
    if any(manifest[key] != run_contract[key] for key in run_source_fields):
        fail("prebuilt origin sources differ from scientific run sources")
    frozen_control_fields = (
        "RUNNER_SHA256",
        "AGGREGATOR_SHA256",
        "KAT_ANCHOR_SHA256",
        "EXACT_TREE_KERNEL_SHA256",
        "GATE_SHA256",
        "SLURM_JOB_SCRIPT_SHA256",
    )
    if any(manifest[key] != frozen[key] for key in frozen_control_fields):
        fail("prebuilt origin control sources differ from the frozen contract")
    if (
        manifest["FROZEN_CONTRACT_SHA256"] != run_contract["FROZEN_CONTRACT_SHA256"]
        or manifest["HPG_WORKER_BINARY_SHA256"]
        != frozen["PREBUILT_HPG_BINARY_SHA256"]
        or manifest["HAPG_WORKER_BINARY_SHA256"]
        != frozen["PREBUILT_HAPG_BINARY_SHA256"]
        or digest(safe_file(bundle, "hpg-worker-binary"))
        != manifest["HPG_WORKER_BINARY_SHA256"]
        or digest(safe_file(bundle, "hapg-worker-binary"))
        != manifest["HAPG_WORKER_BINARY_SHA256"]
        or stable_bytes(safe_file(origin, "git-status.txt"), "prebuilt git status")
        != b""
    ):
        fail("prebuilt binary, contract, or clean-source binding mismatch")


@dataclass(frozen=True)
class AdaptiveVerification:
    nodes: Mapping[str, RichNode]
    evaluations: Mapping[str, Mapping[str, str]]
    terminals: Sequence[object]
    accepted_area: Fraction
    unresolved_area: Fraction
    replay_counts: ReplayCounts
    wave_count: int
    hpg_signed: int
    hapg_attempted: int
    hapg_certified: int
    hapg_rescues: int


def verify_adaptive(
    bundle: Path,
    run_contract: Mapping[str, str],
    manifest: Mapping[str, str],
) -> AdaptiveVerification:
    nodes = parse_nodes(safe_file(bundle, "nodes.tsv"))
    evaluation_rows = read_table(
        safe_file(bundle, "evaluations.tsv"),
        EVALUATION_COLUMNS,
        "evaluations",
        allow_empty=False,
    )
    evaluation_map: dict[str, Mapping[str, str]] = {}
    for row in evaluation_rows:
        identity = row["NODE_ID"]
        if identity in evaluation_map:
            fail("duplicate global evaluation")
        evaluation_map[identity] = row
    if [(row["WAVE_INDEX"], row["NODE_ID"]) for row in evaluation_rows] != sorted(
        (row["WAVE_INDEX"], row["NODE_ID"]) for row in evaluation_rows
    ):
        fail("global evaluations are not canonically ordered")
    negative_rows = read_table(
        safe_file(bundle, "negative-outcomes.tsv"),
        NEGATIVE_COLUMNS,
        "negative outcomes",
    )
    wave_ledger = read_table(
        safe_file(bundle, "waves.tsv"), WAVES_COLUMNS, "waves", allow_empty=False
    )
    if [parse_int(row["WAVE_INDEX"], "waves WAVE_INDEX") for row in wave_ledger] != list(
        range(len(wave_ledger))
    ):
        fail("adaptive wave ledger is not contiguous from zero")
    max_nodes = parse_int(run_contract["MAX_NODES"], "MAX_NODES")
    max_waves = parse_int(run_contract["MAX_WAVES"], "MAX_WAVES")
    max_u_depth = parse_int(run_contract["MAX_U_DEPTH"], "MAX_U_DEPTH")
    max_s_depth = parse_int(run_contract["MAX_S_DEPTH"], "MAX_S_DEPTH")
    timeout = parse_int(run_contract["TIMEOUT_SECONDS"], "TIMEOUT_SECONDS")
    mutation_audit = parse_bool(run_contract["MUTATION_AUDIT"], "MUTATION_AUDIT")
    root_challenge = run_contract["ROOT_CHALLENGE"]
    run_contract_sha = digest(safe_file(bundle, "run-contract.txt"))
    frontier = [ExpectedLeaf(0, 0, 0, 0, "-", 0)]
    previous_result_sha = ZERO_SHA256
    allocated = 1
    derived_evaluations: dict[str, Mapping[str, str]] = {}
    derived_negatives: list[Mapping[str, str]] = []
    replay_counts = ReplayCounts()
    hpg_signed = hapg_attempted = hapg_certified = hapg_rescues = 0

    for wave_index, ledger_row in enumerate(wave_ledger):
        if not frontier or any(leaf.wave_index != wave_index for leaf in frontier):
            fail("adaptive frontier violates BFS wave depth")
        wave_path = safe_file(bundle, f"wave-contracts/W{wave_index:04d}.tsv")
        result_path = safe_file(bundle, f"wave-results/W{wave_index:04d}.tsv")
        wave = LEAF_VERIFY.parse_wave_contract(wave_path)
        result = parse_wave_result(result_path)
        frontier_sha = digest_bytes(frontier_bytes(frontier))
        if (
            wave.headers["RUN_CONTRACT_SHA256"] != run_contract_sha
            or wave.headers["ROOT_CHALLENGE"] != root_challenge
            or parse_int(wave.headers["WAVE_INDEX"], "wave WAVE_INDEX") != wave_index
            or wave.headers["PREVIOUS_WAVE_RESULT_SHA256"] != previous_result_sha
            or wave.headers["FRONTIER_SHA256"] != frontier_sha
            or wave.headers["HPG_WORKER_SOURCE_SHA256"]
            != run_contract["HPG_WORKER_SOURCE_SHA256"]
            or wave.headers["HPG_VERIFIER_SOURCE_SHA256"]
            != run_contract["HPG_VERIFIER_SOURCE_SHA256"]
            or wave.headers["HAPG_WORKER_SOURCE_SHA256"]
            != run_contract["HAPG_WORKER_SOURCE_SHA256"]
            or wave.headers["HAPG_KERNEL_SOURCE_SHA256"]
            != run_contract["HAPG_KERNEL_SOURCE_SHA256"]
            or wave.headers["HAPG_VERIFIER_ADAPTER_SHA256"]
            != run_contract["HAPG_VERIFIER_ADAPTER_SHA256"]
            or wave.headers["HAPG_NUMERIC_VERIFIER_SHA256"]
            != run_contract["HAPG_NUMERIC_VERIFIER_SHA256"]
        ):
            fail(f"wave contract causal/source binding mismatch: W{wave_index:04d}")
        expected_by_id = {leaf.identity: leaf for leaf in frontier}
        if set(wave.rows) != set(expected_by_id) or set(result.rows) != set(expected_by_id):
            fail(f"wave population differs from exact frontier: W{wave_index:04d}")
        if (
            result.headers["WAVE_INDEX"] != str(wave_index)
            or result.headers["WAVE_CONTRACT_SHA256"] != wave.sha256
            or result.headers["NODE_COUNT"] != str(len(frontier))
        ):
            fail(f"wave result envelope mismatch: W{wave_index:04d}")
        for identity, leaf in expected_by_id.items():
            row = wave.rows[identity].values
            if tuple(
                row[key]
                for key in (
                    "PARENT_ID",
                    "U_DEPTH",
                    "U_INDEX",
                    "S_DEPTH",
                    "S_INDEX",
                    "INPUT_SHA256",
                )
            ) != (
                leaf.parent,
                str(leaf.u_depth),
                str(leaf.u_index),
                str(leaf.s_depth),
                str(leaf.s_index),
                digest_bytes(leaf_input_bytes(leaf)),
            ):
                fail(f"wave row differs from expected frontier leaf: {identity}")
            add_counts(
                replay_counts,
                verify_leaf_artifacts(
                    bundle,
                    bundle,
                    wave_path,
                    wave,
                    leaf,
                    result.rows[identity],
                    run_contract["HPG_WORKER_SOURCE_SHA256"],
                    run_contract["HAPG_WORKER_SOURCE_SHA256"],
                    root_challenge,
                    mutation_audit,
                    timeout,
                ),
            )

        fixed: dict[str, tuple[str, str]] = {}
        candidates: list[tuple[ExpectedLeaf, str, tuple[ExpectedLeaf, ExpectedLeaf]]] = []
        for leaf in frontier:
            hpg = wave.rows[leaf.identity].values
            row = result.rows[leaf.identity]
            apg_valid = parse_bool(row["APG_VALID"], "APG_VALID")
            apg_pass = parse_bool(row["APG_PASS"], "APG_PASS")
            if apg_valid and apg_pass:
                fixed[leaf.identity] = ("CERTIFIED", "H_APG")
                continue
            if hpg["HPG_STATUS"] == "H_PG_TIMEOUT" or row["HAPG_STATUS"] == "H_APG_TIMEOUT":
                fixed[leaf.identity] = ("UNRESOLVED", "TIMEOUT")
                continue
            action, children = split_leaf(leaf)
            axis_limited = (
                action == "SPLIT_S" and leaf.s_depth >= max_s_depth
            ) or (action == "SPLIT_U" and leaf.u_depth >= max_u_depth)
            if axis_limited:
                fixed[leaf.identity] = ("UNRESOLVED", "AXIS_DEPTH")
            else:
                candidates.append((leaf, action, children))
        if candidates:
            if wave_index + 1 >= max_waves:
                fixed.update(
                    (leaf.identity, ("UNRESOLVED", "WAVE_LIMIT"))
                    for leaf, _, _ in candidates
                )
                candidates = []
            elif allocated + 2 * len(candidates) > max_nodes:
                fixed.update(
                    (leaf.identity, ("UNRESOLVED", "NODE_BUDGET"))
                    for leaf, _, _ in candidates
                )
                candidates = []
        split_by_id = {
            leaf.identity: (action, children) for leaf, action, children in candidates
        }
        next_frontier: list[ExpectedLeaf] = []
        for leaf in frontier:
            hpg = wave.rows[leaf.identity].values
            row = result.rows[leaf.identity]
            if leaf.identity in split_by_id:
                action, children = split_by_id[leaf.identity]
                reason = "-"
                next_frontier.extend(children)
            else:
                action, reason = fixed[leaf.identity]
            if row["DECISION"] != action or row["TERMINAL_REASON"] != reason:
                fail(f"recorded decision differs from APG-only policy: {leaf.identity}")
            node = nodes.get(leaf.identity)
            if node is None or (
                node.parent,
                node.u_depth,
                node.u_index,
                node.s_depth,
                node.s_index,
                node.wave_index,
                node.action,
                node.terminal_reason,
                node.wave_contract_sha,
            ) != (
                leaf.parent,
                leaf.u_depth,
                leaf.u_index,
                leaf.s_depth,
                leaf.s_index,
                leaf.wave_index,
                action,
                reason,
                wave.sha256,
            ):
                fail(f"nodes ledger differs from reconstructed decision: {leaf.identity}")
            projection = evaluation_projection(leaf, wave.sha256, hpg, row)
            if evaluation_map.get(leaf.identity) != projection:
                fail(f"global evaluation differs from wave evidence: {leaf.identity}")
            derived_evaluations[leaf.identity] = projection
            if not parse_bool(row["APG_PASS"], "APG_PASS"):
                derived_negatives.append(negative_projection(leaf, hpg, row))
            hpg_signed += parse_bool(hpg["HAPG_ELIGIBLE"], "HAPG_ELIGIBLE")
            hapg_attempted += parse_bool(row["HAPG_ATTEMPTED"], "HAPG_ATTEMPTED")
            hapg_certified += parse_bool(row["APG_PASS"], "APG_PASS")
            hapg_rescues += parse_bool(row["APG_RESCUE"], "APG_RESCUE")
        next_frontier.sort(key=lambda item: item.identity)
        next_sha = digest_bytes(frontier_bytes(next_frontier))
        if result.headers["NEXT_FRONTIER_SHA256"] != next_sha:
            fail(f"wave result next-frontier digest mismatch: W{wave_index:04d}")
        if ledger_row != {
            "WAVE_INDEX": str(wave_index),
            "FRONTIER_SHA256": frontier_sha,
            "WAVE_CONTRACT_SHA256": wave.sha256,
            "WAVE_RESULT_SHA256": result.sha256,
            "NEXT_FRONTIER_SHA256": next_sha,
        }:
            fail(f"global wave ledger mismatch: W{wave_index:04d}")
        allocated += len(next_frontier)
        if allocated > max_nodes:
            fail("reconstructed tree exceeds node budget")
        previous_result_sha = result.sha256
        frontier = next_frontier

    if frontier:
        fail("adaptive wave ledger ended with an unpublished frontier")
    if set(nodes) != set(derived_evaluations) or set(evaluation_map) != set(nodes):
        fail("tree, wave, and global evaluation node sets differ")
    if negative_rows != derived_negatives:
        fail("negative-outcomes ledger is not exact equality of APG negatives")
    projected = {
        identity: C1.Node(
            identity,
            node.parent,
            node.u_depth,
            node.u_index,
            node.s_depth,
            node.s_index,
            node.action,
        )
        for identity, node in nodes.items()
    }
    try:
        terminals, accepted_area, unresolved_area = C1.verify_structure(projected)
    except C1.CoverError as error:
        raise AggregateError(f"exact dyadic tree rejected: {error}") from error
    if len(nodes) != allocated:
        fail("adaptive allocated-node accounting mismatch")
    if manifest["EVALUATED_NODE_COUNT"] != str(len(nodes)):
        fail("run manifest evaluated-node count mismatch")
    if manifest["WAVE_COUNT"] != str(len(wave_ledger)):
        fail("run manifest wave count mismatch")
    return AdaptiveVerification(
        nodes,
        evaluation_map,
        terminals,
        accepted_area,
        unresolved_area,
        replay_counts,
        len(wave_ledger),
        hpg_signed,
        hapg_attempted,
        hapg_certified,
        hapg_rescues,
    )


@dataclass(frozen=True)
class FreshReplayVerification:
    terminal_count: int
    wave_count: int
    complete: bool
    replay_counts: ReplayCounts


def verify_fresh_replay(
    bundle: Path,
    run_contract: Mapping[str, str],
    adaptive: AdaptiveVerification,
) -> FreshReplayVerification:
    run_contract_sha = digest(safe_file(bundle, "run-contract.txt"))
    replay_contract_path = safe_file(bundle, "fresh-replay-contract.txt")
    replay_contract = read_kv(
        replay_contract_path,
        FRESH_REPLAY_CONTRACT_KEYS,
        "fresh replay contract",
    )
    terminal_columns = (
        "NODE_ID",
        "WAVE_INDEX",
        "ORIGINAL_HPG_RECEIPT_SHA256",
        "ORIGINAL_HAPG_RECEIPT_SHA256",
    )
    terminal_path = safe_file(bundle, "fresh-replay-terminals.tsv")
    terminal_rows = read_table(
        terminal_path, terminal_columns, "fresh replay terminals"
    )
    certified_nodes = sorted(
        (node for node in adaptive.nodes.values() if node.action == "CERTIFIED"),
        key=lambda node: (node.wave_index, node.identity),
    )
    expected_terminal_rows = [
        {
            "NODE_ID": node.identity,
            "WAVE_INDEX": str(node.wave_index),
            "ORIGINAL_HPG_RECEIPT_SHA256": adaptive.evaluations[node.identity][
                "HPG_RECEIPT_SHA256"
            ],
            "ORIGINAL_HAPG_RECEIPT_SHA256": adaptive.evaluations[node.identity][
                "HAPG_RECEIPT_SHA256"
            ],
        }
        for node in certified_nodes
    ]
    if terminal_rows != expected_terminal_rows:
        fail("fresh replay terminal set differs from certified tree terminals")
    replay_root = run_contract["FRESH_REPLAY_ROOT_CHALLENGE"]
    if (
        replay_contract["SCHEMA"]
        != "sounio.cs6.hapg-full-source-cover-fresh-replay-contract.v1"
        or replay_contract["PARENT_RUN_CONTRACT_SHA256"] != run_contract_sha
        or replay_contract["ORIGINAL_ROOT_CHALLENGE"] != run_contract["ROOT_CHALLENGE"]
        or replay_contract["FRESH_REPLAY_ROOT_CHALLENGE"] != replay_root
        or replay_root == run_contract["ROOT_CHALLENGE"]
        or replay_contract["CERTIFIED_TERMINAL_COUNT"] != str(len(certified_nodes))
        or replay_contract["CERTIFIED_TERMINALS_SHA256"] != digest(terminal_path)
        or replay_contract["POLICY"]
        != "EVERY_CERTIFIED_TERMINAL_FRESH_HPG_FREEZE_HAPG"
        or replay_contract["PROMOTION_ELIGIBLE"] != "false"
    ):
        fail("fresh replay contract binding mismatch")

    groups: list[list[RichNode]] = []
    for node in certified_nodes:
        if not groups or groups[-1][0].wave_index != node.wave_index:
            groups.append([])
        groups[-1].append(node)
    replay_root_dir = bundle / "fresh-replay"
    if replay_root_dir.is_symlink() or not replay_root_dir.is_dir():
        fail("fresh replay evidence directory is missing or unsafe")
    wave_rows = read_table(
        safe_file(replay_root_dir, "waves.tsv"),
        WAVES_COLUMNS,
        "fresh replay waves",
    )
    evaluation_rows = read_table(
        safe_file(replay_root_dir, "evaluations.tsv"),
        EVALUATION_COLUMNS,
        "fresh replay evaluations",
    )
    negative_rows = read_table(
        safe_file(replay_root_dir, "negative-outcomes.tsv"),
        NEGATIVE_COLUMNS,
        "fresh replay negatives",
    )
    if negative_rows:
        fail("fresh replay contains an APG-negative result")
    if len(wave_rows) != len(groups):
        fail("fresh replay wave count differs from terminal depth groups")
    replay_evaluation_map = {row["NODE_ID"]: row for row in evaluation_rows}
    if len(replay_evaluation_map) != len(evaluation_rows):
        fail("fresh replay evaluations contain duplicate nodes")
    previous_result_sha = ZERO_SHA256
    derived_ids: set[str] = set()
    counts = ReplayCounts()
    replay_contract_sha = digest(replay_contract_path)
    timeout = parse_int(run_contract["TIMEOUT_SECONDS"], "TIMEOUT_SECONDS")
    for group_index, group in enumerate(groups):
        wave_index = group[0].wave_index
        leaves = [
            ExpectedLeaf(
                node.u_depth,
                node.u_index,
                node.s_depth,
                node.s_index,
                node.parent,
                node.wave_index,
            )
            for node in group
        ]
        wave_path = safe_file(
            replay_root_dir, f"wave-contracts/W{wave_index:04d}.tsv"
        )
        result_path = safe_file(
            replay_root_dir, f"wave-results/W{wave_index:04d}.tsv"
        )
        wave = LEAF_VERIFY.parse_wave_contract(wave_path)
        result = parse_wave_result(result_path)
        frontier_sha = digest_bytes(frontier_bytes(leaves))
        if (
            wave.headers["RUN_CONTRACT_SHA256"] != replay_contract_sha
            or wave.headers["ROOT_CHALLENGE"] != replay_root
            or wave.headers["WAVE_INDEX"] != str(wave_index)
            or wave.headers["PREVIOUS_WAVE_RESULT_SHA256"] != previous_result_sha
            or wave.headers["FRONTIER_SHA256"] != frontier_sha
            or wave.headers["HPG_WORKER_SOURCE_SHA256"]
            != run_contract["HPG_WORKER_SOURCE_SHA256"]
            or wave.headers["HPG_VERIFIER_SOURCE_SHA256"]
            != run_contract["HPG_VERIFIER_SOURCE_SHA256"]
            or wave.headers["HAPG_WORKER_SOURCE_SHA256"]
            != run_contract["HAPG_WORKER_SOURCE_SHA256"]
            or wave.headers["HAPG_KERNEL_SOURCE_SHA256"]
            != run_contract["HAPG_KERNEL_SOURCE_SHA256"]
            or wave.headers["HAPG_VERIFIER_ADAPTER_SHA256"]
            != run_contract["HAPG_VERIFIER_ADAPTER_SHA256"]
            or wave.headers["HAPG_NUMERIC_VERIFIER_SHA256"]
            != run_contract["HAPG_NUMERIC_VERIFIER_SHA256"]
            or set(wave.rows) != {leaf.identity for leaf in leaves}
            or set(result.rows) != set(wave.rows)
            or result.headers["WAVE_INDEX"] != str(wave_index)
            or result.headers["NODE_COUNT"] != str(len(leaves))
            or result.headers["WAVE_CONTRACT_SHA256"] != wave.sha256
        ):
            fail("fresh replay wave causal binding mismatch")
        next_leaves = (
            [
                ExpectedLeaf(
                    node.u_depth,
                    node.u_index,
                    node.s_depth,
                    node.s_index,
                    node.parent,
                    node.wave_index,
                )
                for node in groups[group_index + 1]
            ]
            if group_index + 1 < len(groups)
            else []
        )
        next_sha = digest_bytes(frontier_bytes(next_leaves))
        if result.headers["NEXT_FRONTIER_SHA256"] != next_sha:
            fail("fresh replay next-frontier binding mismatch")
        if wave_rows[group_index] != {
            "WAVE_INDEX": str(wave_index),
            "FRONTIER_SHA256": frontier_sha,
            "WAVE_CONTRACT_SHA256": wave.sha256,
            "WAVE_RESULT_SHA256": result.sha256,
            "NEXT_FRONTIER_SHA256": next_sha,
        }:
            fail("fresh replay global wave ledger mismatch")
        original_wave = LEAF_VERIFY.parse_wave_contract(
            safe_file(bundle, f"wave-contracts/W{wave_index:04d}.tsv")
        )
        for leaf in leaves:
            hpg = wave.rows[leaf.identity].values
            row = result.rows[leaf.identity]
            if (
                not parse_bool(hpg["HAPG_ELIGIBLE"], "fresh HAPG_ELIGIBLE")
                or hpg["HPG_STATUS"] != LEAF_VERIFY.SIGNED_CHART_STATUS
                or wave.rows[leaf.identity].chart_signs
                != original_wave.rows[leaf.identity].chart_signs
                or row["DECISION"] != "REPLAY_CERTIFIED"
                or row["TERMINAL_REASON"] != "-"
                or not parse_bool(row["HAPG_ATTEMPTED"], "fresh HAPG attempted")
                or not parse_bool(row["APG_VALID"], "fresh APG valid")
                or not parse_bool(row["APG_PASS"], "fresh APG pass")
            ):
                fail(f"fresh replay did not reproduce certified terminal: {leaf.identity}")
            add_counts(
                counts,
                verify_leaf_artifacts(
                    replay_root_dir,
                    bundle,
                    wave_path,
                    wave,
                    leaf,
                    row,
                    run_contract["HPG_WORKER_SOURCE_SHA256"],
                    run_contract["HAPG_WORKER_SOURCE_SHA256"],
                    replay_root,
                    False,
                    timeout,
                ),
            )
            projection = evaluation_projection(leaf, wave.sha256, hpg, row)
            if replay_evaluation_map.get(leaf.identity) != projection:
                fail("fresh replay global evaluation mismatch")
            derived_ids.add(leaf.identity)
        previous_result_sha = result.sha256
    if (
        derived_ids != {node.identity for node in certified_nodes}
        or set(replay_evaluation_map) != derived_ids
    ):
        fail("fresh replay did not cover every certified terminal exactly once")
    complete = len(derived_ids) == len(certified_nodes)
    return FreshReplayVerification(len(certified_nodes), len(groups), complete, counts)


def validate_summary(
    summary: Mapping[str, str],
    adaptive: AdaptiveVerification,
    fresh: FreshReplayVerification,
) -> bool:
    terminals = list(adaptive.terminals)
    certified_terminals = sum(node.action == "CERTIFIED" for node in terminals)
    unresolved_terminals = sum(node.action == "UNRESOLVED" for node in terminals)
    expected = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-summary.v1",
        "MODE": "adaptive",
        "BOUNDED_RUN_COMPLETE": "true",
        "INFRASTRUCTURE_VALID": "true",
        "EVALUATED_NODE_COUNT": str(len(adaptive.nodes)),
        "WAVE_COUNT": str(adaptive.wave_count),
        "HPG_SIGNED_CHART_COUNT": str(adaptive.hpg_signed),
        "HAPG_ATTEMPTED_COUNT": str(adaptive.hapg_attempted),
        "HAPG_CERTIFIED_COUNT": str(adaptive.hapg_certified),
        "HAPG_RESCUE_COUNT": str(adaptive.hapg_rescues),
        "HPG_MUTATION_TESTS": str(adaptive.replay_counts.hpg_mutations),
        "HPG_MUTATIONS_REJECTED": str(adaptive.replay_counts.hpg_rejected),
        "HAPG_MUTATION_TESTS": str(adaptive.replay_counts.hapg_mutations),
        "HAPG_MUTATIONS_REJECTED": str(adaptive.replay_counts.hapg_rejected),
        "FRESH_REPLAY_TERMINAL_COUNT": str(fresh.terminal_count),
        "FRESH_REPLAY_WAVE_COUNT": str(fresh.wave_count),
        "FRESH_REPLAY_COMPLETE": str(fresh.complete).lower(),
        "TREE_NODE_COUNT": str(len(adaptive.nodes)),
        "CERTIFIED_TERMINAL_COUNT": str(certified_terminals),
        "UNRESOLVED_TERMINAL_COUNT": str(unresolved_terminals),
        "UNRESOLVED_AREA_NUMERATOR": str(adaptive.unresolved_area.numerator),
        "UNRESOLVED_AREA_DENOMINATOR": str(adaptive.unresolved_area.denominator),
        "HAPG_FULL_SOURCE_COVER_CANDIDATE": str(
            adaptive.unresolved_area == 0 and fresh.complete
        ).lower(),
        "AGGREGATION_REQUIRED": "true",
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "FULL_SOURCE_CARRIER_PROVED": "false",
        "HYPERBOLICITY_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false",
        "OPEN_PROBLEM_SOLVED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    if dict(summary) != expected:
        fail("summary differs from independent reconstruction")
    return adaptive.unresolved_area == 0 and fresh.complete


def self_test_mutations() -> tuple[int, int]:
    c1_total, c1_rejected = C1.self_test_mutations()
    if c1_total != c1_rejected:
        fail("exact tree kernel self-test did not reject every mutation")
    checks = 0
    rejected = 0

    root = ExpectedLeaf(0, 0, 0, 0, "-", 0)
    action, children = split_leaf(root)
    checks += 1
    if action == "SPLIT_S" and len(children) == 2:
        rejected += 1
    checks += 1
    if split_leaf(children[0])[0] == "SPLIT_U":
        rejected += 1
    checks += 1
    if not (True and False):
        rejected += 1
    checks += 1
    if 1 + 2 * 3 > 5:
        rejected += 1
    for raw in (
        b"probe error: frozen TM2 pivot sign was not certified\n",
        b"probe error: TM2 reciprocal pivot contains zero\n",
        b"probe error: TM2 reciprocal center contains zero\n",
        b"probe error: TM2 reciprocal is noncontractive\n",
    ):
        checks += 1
        if classify_failure(raw, "H_APG") == "H_APG_FROZEN_CHART":
            rejected += 1
    checks += 1
    if classify_failure(b"probe error: unsupported frozen chart\n", "H_APG") is None:
        rejected += 1
    if checks != rejected:
        fail("H-APG policy self-test did not reject every mutation")
    return c1_total + checks, c1_rejected + rejected


def emit_certificate(
    path: Path | None, fields: Sequence[tuple[str, str]]
) -> None:
    if path is not None:
        try:
            C1.write_certificate(path, fields)
        except C1.CoverError as error:
            raise AggregateError(f"certificate publication failed: {error}") from error
    for key, value in fields:
        print(f"{key}={value}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--expected-contract-sha", required=True)
    parser.add_argument("--expected-git-head", required=True)
    parser.add_argument("--kat-archive", required=True, type=Path)
    parser.add_argument("--kat-archive-sha256", required=True)
    parser.add_argument("--kat-job-id", required=True)
    parser.add_argument("--transport-repo-delta-sha256", required=True)
    parser.add_argument("--transport-prebuilt-archive-sha256", required=True)
    parser.add_argument("--kat-sacct-file", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--require-local-cover", action="store_true")
    parser.add_argument("--require-promotion", action="store_true")
    args = parser.parse_args(argv)
    if SHA_RE.fullmatch(args.expected_contract_sha) is None:
        fail("expected contract anchor must be lowercase SHA-256")
    if re.fullmatch(r"[0-9a-f]{40}", args.expected_git_head) is None:
        fail("expected Git head anchor must be a lowercase 40-hex commit")
    if (
        SHA_RE.fullmatch(args.kat_archive_sha256) is None
        or re.fullmatch(r"[1-9][0-9]*", args.kat_job_id) is None
        or SHA_RE.fullmatch(args.transport_repo_delta_sha256) is None
        or SHA_RE.fullmatch(args.transport_prebuilt_archive_sha256) is None
    ):
        fail("KAT prerequisite arguments are noncanonical")
    kat_archive = args.kat_archive.resolve()
    if args.kat_archive.is_symlink() or not kat_archive.is_file():
        fail("KAT prerequisite archive must be a regular non-symlink file")
    bundle = args.bundle.resolve()
    if args.bundle.is_symlink() or not bundle.is_dir():
        fail("run bundle must be a regular non-symlink directory")
    output = args.output.resolve() if args.output is not None else None
    if output is not None and (output == bundle or output.is_relative_to(bundle)):
        fail("aggregation output must be outside the immutable run bundle")
    manifest_path = safe_file(bundle, "run-manifest.txt")
    manifest_raw = stable_bytes(manifest_path, "run manifest")
    manifest = parse_kv(manifest_raw, RUN_MANIFEST_KEYS, "run manifest")
    if (
        manifest["SCHEMA"] != "sounio.cs6.hapg-full-source-cover-run-manifest.v1"
        or manifest["RUN_COMPLETE"] != "true"
        or manifest["MODE"] != "adaptive"
        or manifest["CAPD_VERSION"] != "5.3.0"
        or manifest["INTERVAL_BACKEND"] != "FILIB"
        or manifest["OPTIMIZATION_LEVEL"] != "O0"
        or manifest["LOCAL_PROCESS_ORDERED_HASH_CHAIN"] != "true"
        or manifest["EXECUTION_PROVENANCE_ATTESTED"] != "false"
        or manifest["PROMOTION_ELIGIBLE"] != "false"
    ):
        fail("run manifest policy mismatch")
    file_count = verify_file_index(bundle, manifest)
    source_bundle = bundle
    snapshot_owner, bundle = snapshot_bundle(
        source_bundle, manifest, manifest_raw, file_count
    )
    run_contract_path = safe_file(bundle, "run-contract.txt")
    run_contract = read_kv(run_contract_path, RUN_CONTRACT_KEYS, "run contract")
    if (
        run_contract["SCHEMA"] != "sounio.cs6.hapg-full-source-cover-run-contract.v2"
        or run_contract["MODE"] != "adaptive"
        or run_contract["SOURCE"] != "N0"
        or run_contract["ROOT_CHALLENGE"] != manifest["ROOT_CHALLENGE"]
        or run_contract["TRAVERSAL"] != "DETERMINISTIC_BREADTH_FIRST"
        or run_contract["SPLIT_RULE"] != "S_IF_S_DEPTH_LE_U_DEPTH_ELSE_U"
        or run_contract["TERMINAL_PREDICATE"]
        != "APG_COMPUTATION_VALID_AND_APG_CERTIFICATE_PASS"
        or run_contract["BUILD_MODE"] != "VERIFIED_PREBUILT_BUNDLE"
        or SHA_RE.fullmatch(run_contract["PREBUILT_RUN_MANIFEST_SHA256"]) is None
        or not run_contract["SLURM_JOB_ID"].isdigit()
        or run_contract["EXECUTION_NODE"] == "NONE"
        or run_contract["SLURM_JOB_VERIFIED"] != "true"
        or digest(safe_file(bundle, "slurm-job-record.txt"))
        != run_contract["SLURM_JOB_RECORD_SHA256"]
        or run_contract["WORKING_FILESYSTEM_POLICY"]
        != "NODE_LOCAL_TMP_THEN_HASHED_ARCHIVE_TRANSPORT"
        or run_contract["MUTATION_AUDIT"] != "true"
        or run_contract["LOCAL_PROCESS_ORDERED_HASH_CHAIN"] != "true"
        or run_contract["EXECUTION_PROVENANCE_ATTESTED"] != "false"
        or run_contract["PROMOTION_ELIGIBLE"] != "false"
        or run_contract["ALL_OR_NONE_WAVE_ADMISSION"] != "true"
        or digest(run_contract_path) != manifest["RUN_CONTRACT_SHA256"]
    ):
        fail("run contract policy or manifest binding mismatch")
    frozen = verify_frozen_sources(bundle, run_contract, args.expected_contract_sha)
    if (
        frozen.get("KAT_PREREQUISITE_CERTIFICATE_SCHEMA")
        != KAT_CERTIFICATE_SCHEMA
        or run_contract["KAT_PREREQUISITE_CERTIFICATE_SCHEMA"]
        != KAT_CERTIFICATE_SCHEMA
    ):
        fail("frozen or executed contract does not require KAT certificate v2")
    verify_prebuilt_origin(bundle, run_contract, frozen)
    slurm_record_raw = stable_bytes(
        safe_file(bundle, "slurm-job-record.txt"), "Slurm job record"
    )
    try:
        slurm_record_text = slurm_record_raw.decode("ascii").strip()
    except UnicodeError as error:
        raise AggregateError("Slurm job record must be ASCII") from error
    slurm_fields = {
        key: value
        for token in shlex.split(slurm_record_text)
        if "=" in token
        for key, value in (token.split("=", 1),)
    }
    adaptive_submit_utc = slurm_fields.get("SubmitTime", "")
    if (
        slurm_fields.get("JobId") != run_contract["SLURM_JOB_ID"]
        or not adaptive_submit_utc
        or adaptive_submit_utc == "Unknown"
        or adaptive_submit_utc != run_contract["ADAPTIVE_SUBMIT_UTC"]
    ):
        fail("adaptive Slurm submission anchor mismatch")
    bundled_kat_sacct = stable_bytes(
        safe_file(bundle, "kat-prerequisite-sacct.txt"),
        "bundled KAT sacct record",
    )
    if args.kat_sacct_file is None:
        independent_kat_sacct = KAT_ANCHOR.query_live_sacct(args.kat_job_id)
    else:
        independent_kat_sacct = stable_bytes(
            args.kat_sacct_file, "independent KAT sacct record"
        )
    if independent_kat_sacct != bundled_kat_sacct:
        fail("independent KAT sacct record differs from bundled bytes")
    kat_expectations = KAT_ANCHOR.KatAnchorExpectations(
        kat_job_id=args.kat_job_id,
        kat_archive_sha256=args.kat_archive_sha256,
        expected_git_head=args.expected_git_head,
        expected_contract_sha256=args.expected_contract_sha,
        expected_base_repo_bundle_sha256=frozen["BASE_REPO_BUNDLE_SHA256"],
        expected_base_git_head=frozen["BASE_REPO_BUNDLE_GIT_HEAD"],
        expected_repo_delta_bundle_sha256=args.transport_repo_delta_sha256,
        expected_prebuilt_archive_sha256=args.transport_prebuilt_archive_sha256,
        expected_prebuilt_run_manifest_sha256=run_contract[
            "PREBUILT_RUN_MANIFEST_SHA256"
        ],
        expected_slurm_job_script_sha256=frozen["SLURM_JOB_SCRIPT_SHA256"],
        schema_profile="v6",
    )
    try:
        kat_certificate = KAT_ANCHOR.certify_kat_anchor(
            archive_path=kat_archive,
            sidecar_path=Path(f"{kat_archive}.sha256"),
            sacct_bytes=independent_kat_sacct,
            adaptive_job_id=run_contract["SLURM_JOB_ID"],
            adaptive_submit_utc=adaptive_submit_utc,
            expectations=kat_expectations,
        )
    except KAT_ANCHOR.VerificationError as error:
        raise AggregateError(f"KAT prerequisite verification failed: {error}") from error
    kat_fields = kat_certificate.as_dict()
    validate_kat_prerequisite_certificate(kat_fields, frozen)
    bundled_kat_certificate = stable_bytes(
        safe_file(bundle, "kat-prerequisite-certificate.txt"),
        "bundled KAT prerequisite certificate",
    )
    run_kat_bindings = {
        "KAT_PREREQUISITE_CERTIFICATE_SCHEMA": kat_fields["SCHEMA"],
        "KAT_PREREQUISITE_CERTIFICATE_SHA256": kat_certificate.sha256,
        "KAT_PREREQUISITE_SACCT_SHA256": kat_fields["KAT_SACCT_SHA256"],
        "KAT_JOB_ID": kat_fields["KAT_JOB_ID"],
        "KAT_ARCHIVE_SHA256": kat_fields["KAT_ARCHIVE_SHA256"],
        "KAT_GIT_HEAD": kat_fields["KAT_EXPECTED_GIT_HEAD"],
        "KAT_FROZEN_CONTRACT_SHA256": kat_fields["KAT_FROZEN_CONTRACT_SHA256"],
        "KAT_BASE_REPO_BUNDLE_SHA256": kat_fields["KAT_BASE_REPO_BUNDLE_SHA256"],
        "KAT_BASE_GIT_HEAD": kat_fields["KAT_BASE_GIT_HEAD"],
        "KAT_REPO_DELTA_BUNDLE_SHA256": kat_fields["KAT_REPO_DELTA_BUNDLE_SHA256"],
        "KAT_PREBUILT_ARCHIVE_SHA256": kat_fields["KAT_PREBUILT_ARCHIVE_SHA256"],
        "KAT_PREBUILT_RUN_MANIFEST_SHA256": kat_fields[
            "KAT_PREBUILT_RUN_MANIFEST_SHA256"
        ],
        "KAT_SLURM_JOB_SCRIPT_SHA256": kat_fields["KAT_SLURM_JOB_SCRIPT_SHA256"],
        "KAT_END_UTC": kat_fields["KAT_END_UTC"],
        **{key: kat_fields[key] for key in KAT_RUN_CONTRACT_EVIDENCE_KEYS},
        "ADAPTIVE_SUBMIT_UTC": kat_fields["ADAPTIVE_SUBMIT_UTC"],
        "KAT_PREREQUISITE_VALID": kat_fields["KAT_PREREQUISITE_VALID"],
    }
    if (
        run_contract["KAT_PREREQUISITE_REQUIRED"] != "true"
        or run_contract["KAT_PREREQUISITE_VALID"] != "true"
        or kat_fields["KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT"] != "true"
        or bundled_kat_certificate != kat_certificate.as_bytes()
        or any(run_contract[key] != value for key, value in run_kat_bindings.items())
    ):
        fail("KAT prerequisite certificate or run-contract binding mismatch")
    if (
        run_contract["ROOT_CHALLENGE"] != frozen.get("BOUNDED_PILOT_ROOT_CHALLENGE")
        or run_contract["MAX_NODES"] != frozen.get("BOUNDED_PILOT_MAX_NODES")
        or run_contract["MAX_WAVES"] != frozen.get("BOUNDED_PILOT_MAX_WAVES")
        or run_contract["MAX_U_DEPTH"] != frozen.get("BOUNDED_PILOT_MAX_U_DEPTH")
        or run_contract["MAX_S_DEPTH"] != frozen.get("BOUNDED_PILOT_MAX_S_DEPTH")
        or run_contract["JOBS"] != frozen.get("BOUNDED_PILOT_JOBS")
        or run_contract["TIMEOUT_SECONDS"]
        != frozen.get("BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS")
    ):
        fail("adaptive parameters differ from the frozen bounded pilot")
    if (
        stable_bytes(safe_file(bundle, "git-status.txt"), "git status") != b""
        or stable_bytes(safe_file(bundle, "git-head.txt"), "git head")
        != f"{args.expected_git_head}\n".encode("ascii")
    ):
        fail("authoritative run was produced from a dirty source checkout")
    adaptive = verify_adaptive(bundle, run_contract, manifest)
    fresh = verify_fresh_replay(bundle, run_contract, adaptive)
    summary = read_kv(safe_file(bundle, "summary.txt"), SUMMARY_KEYS, "summary")
    local_cover = validate_summary(summary, adaptive, fresh)
    mutation_tests = mutation_rejected = 0
    if args.self_test_mutations:
        mutation_tests, mutation_rejected = self_test_mutations()
    fields = (
        ("SCHEMA", "sounio.cs6.hapg-full-source-cover-aggregation.v2"),
        ("RUN_MANIFEST_SHA256", digest(safe_file(bundle, "run-manifest.txt"))),
        ("FILES_INDEX_SHA256", manifest["FILES_INDEX_SHA256"]),
        ("FILE_COUNT", str(file_count)),
        ("RUN_CONTRACT_SHA256", manifest["RUN_CONTRACT_SHA256"]),
        ("FROZEN_CONTRACT_SHA256", run_contract["FROZEN_CONTRACT_SHA256"]),
        ("EXECUTION_GIT_HEAD", args.expected_git_head),
        ("EXACT_TREE_KERNEL_SHA256", str(frozen["EXACT_TREE_KERNEL_SHA256"])),
        ("KAT_PREREQUISITE_VALID", "true"),
        ("KAT_JOB_ID", run_contract["KAT_JOB_ID"]),
        ("KAT_ARCHIVE_SHA256", run_contract["KAT_ARCHIVE_SHA256"]),
        (
            "KAT_PREREQUISITE_CERTIFICATE_SHA256",
            run_contract["KAT_PREREQUISITE_CERTIFICATE_SHA256"],
        ),
        (
            "KAT_PREREQUISITE_SACCT_SHA256",
            run_contract["KAT_PREREQUISITE_SACCT_SHA256"],
        ),
        ("NODES_SHA256", digest(safe_file(bundle, "nodes.tsv"))),
        ("EVALUATIONS_SHA256", digest(safe_file(bundle, "evaluations.tsv"))),
        ("NEGATIVE_OUTCOMES_SHA256", digest(safe_file(bundle, "negative-outcomes.tsv"))),
        ("WAVES_SHA256", digest(safe_file(bundle, "waves.tsv"))),
        ("WAVE_CHAIN_VALID", "true"),
        ("TREE_STRUCTURE_VALID", "true"),
        ("TREE_NODE_COUNT", str(len(adaptive.nodes))),
        ("CERTIFIED_TERMINAL_COUNT", str(fresh.terminal_count)),
        (
            "UNRESOLVED_TERMINAL_COUNT",
            str(sum(node.action == "UNRESOLVED" for node in adaptive.terminals)),
        ),
        ("ACCEPTED_AREA_NUMERATOR", str(adaptive.accepted_area.numerator)),
        ("ACCEPTED_AREA_DENOMINATOR", str(adaptive.accepted_area.denominator)),
        ("UNRESOLVED_AREA_NUMERATOR", str(adaptive.unresolved_area.numerator)),
        ("UNRESOLVED_AREA_DENOMINATOR", str(adaptive.unresolved_area.denominator)),
        ("STORED_HPG_REPLAY_COUNT", str(adaptive.replay_counts.hpg)),
        ("STORED_HAPG_REPLAY_COUNT", str(adaptive.replay_counts.hapg)),
        ("FRESH_REPLAY_COUNT", str(fresh.terminal_count)),
        ("FRESH_REPLAY_COMPLETE", str(fresh.complete).lower()),
        ("AGGREGATOR_MUTATION_TESTS", str(mutation_tests)),
        ("AGGREGATOR_MUTATIONS_REJECTED", str(mutation_rejected)),
        ("BOUNDED_RUN_VALID", "true"),
        ("LOCAL_COMPLETE_HAPG_COVER", str(local_cover).lower()),
        ("EXECUTION_PROVENANCE_ATTESTED", "false"),
        ("FULL_SOURCE_CARRIER_PROVED", "false"),
        ("HYPERBOLICITY_PROVED", "false"),
        ("CHAOTIC_ATTRACTOR_PROVED", "false"),
        ("OPEN_PROBLEM_SOLVED", "false"),
        ("PROMOTION_ELIGIBLE", "false"),
    )
    if (
        stable_bytes(bundle / "run-manifest.txt", "snapshotted run manifest")
        != manifest_raw
        or stable_bytes(source_bundle / "run-manifest.txt", "source run manifest")
        != manifest_raw
        or verify_file_index(bundle, manifest) != file_count
        or verify_file_index(source_bundle, manifest) != file_count
    ):
        fail("run bundle changed during independent aggregation")
    emit_certificate(output, fields)
    snapshot_owner.cleanup()
    if args.require_promotion:
        return 3
    if args.require_local_cover and not local_cover:
        return 2
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AggregateError,
        LEAF_VERIFY.CoverVerificationError,
        LEAF_VERIFY.HPG_CORE.VerificationError,
        LEAF_VERIFY.HAPG_CORE.VerificationError,
        KAT_ANCHOR.VerificationError,
    ) as error:
        print(f"aggregation error: {error}", file=sys.stderr)
        raise SystemExit(1)
