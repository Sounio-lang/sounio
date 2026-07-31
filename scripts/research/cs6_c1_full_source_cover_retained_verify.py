#!/usr/bin/env python3
"""Verify the retained CS6 C1 scout evidence without rebuilding its worker."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import os
import re
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


class VerificationError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise VerificationError(message)


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
UINT_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")
PAIR_RE = re.compile(r"^(0|[1-9][0-9]*):(0|[1-9][0-9]*)$")
LEAF_ID_RE = re.compile(r"^U([0-9]{2})-([0-9]{10})_S([0-9]{2})-([0-9]{10})$")
ZERO_SHA = "0" * 64
CHALLENGE_DOMAIN = b"sounio.cs6.c1-cover-leaf-challenge.v1\0"

SOURCE_SHA = "d9009effa2fc7ebd399b46df4d02ce34736ef512dfab6b991366d3e7bc1aa9b4"
RUNNER_SHA = "7a7f96cf52530b3cce7854ad45e8c3a156ad8dda3ed778a9d3bbef52f127ac2f"
REPLAY_VERIFIER_SHA = "003cd4b3daa0222fe6ba2c234e46ee7e41fd23d2f0d806bacea29d045302e32e"
WORKER_BINARY_SHA = "5dc27a4dc696bbec157b34cbd7339f39119fc1dfad5446114b8f0db135eddf94"

PROVENANCE_REL = Path(
    "scripts/research/receipts/cs6_c1_full_source_cover_provenance_v1"
)

RETAINED_KEYS = (
    "SCHEMA",
    "EVIDENCE_CLASS",
    "PROMOTION_ELIGIBLE",
    "RAW_RUN_MANIFEST_SHA256",
    "RETAINED_FILES_INDEX_SHA256",
    "RAW_RUN_MANIFEST_TRUST",
    "ORIGINAL_SOURCE_SHA256",
    "ORIGINAL_SOURCE_SNAPSHOT_STATUS",
    "ORIGINAL_SOURCE_SNAPSHOT_PATH",
    "ORIGINAL_RUNNER_SHA256",
    "ORIGINAL_RUNNER_SNAPSHOT_STATUS",
    "ORIGINAL_RUNNER_SNAPSHOT_PATH",
    "ORIGINAL_VERIFIER_SHA256",
    "ORIGINAL_VERIFIER_SNAPSHOT_STATUS",
    "ORIGINAL_VERIFIER_SNAPSHOT_PATH",
    "ORIGINAL_WORKER_BINARY_SHA256",
    "ORIGINAL_WORKER_BINARY_RETAINED",
    "EXACT_EXECUTION_REPLAYABLE",
    "REPLAY_VERIFIER_SHA256",
    "REPLAY_VERIFIER_SNAPSHOT_PATH",
    "REPLAYABLE_RECEIPT_COUNT",
    "REPLAY_OUTPUT_MATCH_COUNT",
    "MUTATION_AUDIT_REPLAY_MATCH",
)

RUN_MANIFEST_KEYS = (
    "SCHEMA",
    "RUN_COMPLETE",
    "CAPD_VERSION",
    "INTERVAL_BACKEND",
    "OPTIMIZATION_LEVEL",
    "ROOT_CHALLENGE",
    "SOURCE_SHA256",
    "LEAF_COUNT",
    "CERTIFIED_COUNT",
    "MUTATION_TESTS",
    "SCOUT_ONLY",
    "EXECUTION_TRUST_MODEL",
    "REMOTE_ATTESTATION_PRESENT",
    "INDEPENDENT_REPLAY_REQUIRED",
    "PROMOTION_ELIGIBLE",
    "FULL_SOURCE_CARRIER_PROVED",
    "RUN_CONTRACT_TXT_SHA256",
    "WORKER_SOURCE_CPP_SHA256",
    "LEAF_VERIFIER_PY_SHA256",
    "RUNNER_PY_SHA256",
    "WORKER_BINARY_SHA256",
    "COMPILE_COMMAND_TXT_SHA256",
    "DEPENDENCIES_SHA256_SHA256",
    "LINK_INPUTS_SHA256_SHA256",
    "RUNTIME_LIBRARIES_SHA256_SHA256",
    "SCOUT_TSV_SHA256",
    "SUMMARY_TXT_SHA256",
    "MUTATION_AUDIT_TXT_SHA256",
)

RUN_CONTRACT_KEYS = (
    "SCHEMA",
    "SOURCE",
    "ROOT_CHALLENGE",
    "DEPTH_PAIRS",
    "GRID",
    "INCLUDE_ROOT",
    "JOBS",
    "TIMEOUT_SECONDS",
    "SCOUT_ONLY",
    "FULL_SOURCE_CARRIER_PROVED",
)

SUMMARY_KEYS = (
    "SCHEMA",
    "SCOUT_ONLY",
    "LEAF_COUNT",
    "CERTIFIED_COUNT",
    "SUBDIVISION_REQUIRED_COUNT",
    "COMPUTATION_UNRESOLVED_COUNT",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "FULL_SOURCE_CARRIER_PROVED",
    "PROJECTIVE_RICCATI_INTEGRATED",
    "HYPERBOLICITY_PROVED",
    "CHAOTIC_ATTRACTOR_PROVED",
    "U250_USED",
)

VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "LEAF_METHOD",
    "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)

SCOUT_HEADER = (
    "LEAF_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "STATUS",
    "METHOD",
    "CERTIFICATE_PASS",
    "SUBDIVISION_REQUIRED",
    "INPUT_SHA256",
    "LEAF_CHALLENGE",
    "RECEIPT_SHA256",
    "VERIFICATION_SHA256",
    "PHYSICAL_SHA256",
    "WORKER_RC",
    "ELAPSED_MS",
)

RETAINED_HASH_FILES = {
    "RUN_CONTRACT_TXT_SHA256": "run-contract.txt",
    "COMPILE_COMMAND_TXT_SHA256": "compile-command.txt",
    "DEPENDENCIES_SHA256_SHA256": "dependencies.sha256",
    "LINK_INPUTS_SHA256_SHA256": "link-inputs.sha256",
    "RUNTIME_LIBRARIES_SHA256_SHA256": "runtime-libraries.sha256",
    "SCOUT_TSV_SHA256": "scout.tsv",
    "SUMMARY_TXT_SHA256": "summary.txt",
    "MUTATION_AUDIT_TXT_SHA256": "mutation-audit.txt",
}

FORBIDDEN_NAMES = {"worker-binary"}
FORBIDDEN_SUFFIXES = {".a", ".bin", ".elf", ".exe", ".o", ".pyc", ".so"}


@dataclass(frozen=True)
class RunSpec:
    directory: str
    raw_manifest_sha: str
    retained_index_sha: str
    original_verifier_sha: str
    leaf_count: int
    certified_count: int
    replayable_count: int
    unresolved_id: str | None = None
    unresolved_stderr_sha: str | None = None


RUN_SPECS = (
    RunSpec(
        "cs6_c1_full_source_cover_core_scout_v1",
        "5c867298ad3abc4dedfcbcfaeec64bb9da80aaa1ce0f1d4614c589f501d1b2f1",
        "aafd44991eb2ad918b875ca63d75592e35333063741ab6ddca3cf401162a2406",
        "65cd0e5adbddeffadd8688a58058ae6775cd37c36ec1274c2217998964f9cdc0",
        25,
        8,
        24,
        "U00-0000000000_S00-0000000000",
        "d14d17a0aab50829d93f23257bfbbe81a6c98517fe72137143ec59a731e55d1d",
    ),
    RunSpec(
        "cs6_c1_full_source_cover_boundary_scout_v1",
        "cde98863eb1c9b3e43cc892596b0da70175c2f589fc5035a243140a801c24909",
        "30f041741e59ca50126d614dc81f8fe4e75f3d80aa50bb11a6bfa66bccbcf180",
        REPLAY_VERIFIER_SHA,
        12,
        4,
        12,
    ),
    RunSpec(
        "cs6_c1_full_source_cover_dense_boundary_scout_v1",
        "b8bddb53d49e24bcd7642d271fd8d9992e0b3e5e4d0da14aa449ae5745a19b27",
        "a3d6e6068a08709562f85c6eafa5aa001386118fd702b44ed74190fef90b8340",
        REPLAY_VERIFIER_SHA,
        16,
        16,
        16,
    ),
)


@dataclass(frozen=True)
class LeafRow:
    identity: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    status: str
    method: str
    certificate: bool
    subdivision: bool
    input_sha: str
    challenge: str
    receipt_sha: str
    verification_sha: str
    physical_sha: str
    worker_rc: int
    elapsed_ms: int


@dataclass(frozen=True)
class RunEvidence:
    spec: RunSpec
    directory: Path
    manifest: Mapping[str, str]
    retained: Mapping[str, str]
    rows: tuple[LeafRow, ...]
    replay_verifier: Path


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest(path: Path) -> str:
    try:
        return digest_bytes(path.read_bytes())
    except OSError as error:
        raise VerificationError(f"cannot hash {path}") from error


def require_sha(value: str, label: str) -> str:
    if SHA_RE.fullmatch(value) is None:
        fail(f"{label} is not a lowercase SHA-256")
    return value


def parse_uint(value: str, label: str) -> int:
    if UINT_RE.fullmatch(value) is None:
        fail(f"{label} is not a canonical unsigned integer")
    return int(value)


def parse_bool(value: str, label: str) -> bool:
    if value == "true":
        return True
    if value == "false":
        return False
    fail(f"{label} is not a canonical boolean")


def canonical_bytes(path: Path, *, ascii_only: bool = True) -> bytes:
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise VerificationError(f"cannot read {path}") from error
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        fail(f"noncanonical line endings in {path}")
    try:
        raw.decode("ascii" if ascii_only else "utf-8")
    except UnicodeError as error:
        raise VerificationError(f"invalid text encoding in {path}") from error
    return raw


def parse_kv(path: Path, expected_keys: Sequence[str]) -> dict[str, str]:
    raw = canonical_bytes(path)
    lines = raw.decode("ascii").splitlines()
    if len(lines) != len(expected_keys):
        fail(f"key count mismatch in {path}: {len(lines)} != {len(expected_keys)}")
    result: dict[str, str] = {}
    for line, expected in zip(lines, expected_keys, strict=True):
        if line.count("=") != 1:
            fail(f"malformed key/value line in {path}")
        key, value = line.split("=", 1)
        if key != expected or not value or key in result:
            fail(f"key order mismatch in {path}: expected {expected}")
        result[key] = value
    return result


def require_text_file(path: Path) -> bytes:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise VerificationError(f"missing retained file: {path}") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        fail(f"retained artifact is not a regular file: {path}")
    if path.name in FORBIDDEN_NAMES or path.suffix.lower() in FORBIDDEN_SUFFIXES:
        fail(f"forbidden binary artifact name: {path}")
    raw = path.read_bytes()
    if raw.startswith(b"\x7fELF") or b"\0" in raw:
        fail(f"binary artifact retained in Git evidence: {path}")
    try:
        text = raw.decode("utf-8")
    except UnicodeError as error:
        raise VerificationError(f"non-text retained artifact: {path}") from error
    if any(ord(char) < 32 and char not in "\t\n\r" for char in text):
        fail(f"control byte in retained text artifact: {path}")
    return raw


def scan_text_tree(root: Path, excluded: set[str]) -> tuple[dict[str, str], str]:
    if root.is_symlink() or not root.is_dir():
        fail(f"retained root is not a real directory: {root}")
    entries: dict[str, str] = {}
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in tuple(directory_names):
            path = current_path / name
            if path.is_symlink():
                fail(f"symlink directory in retained evidence: {path}")
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            require_text_file(path)
            if relative in excluded:
                continue
            entries[relative] = digest(path)
    material = "".join(
        f"{entries[relative]}  {relative}\n" for relative in sorted(entries)
    ).encode("ascii")
    return entries, digest_bytes(material)


def repo_relative_path(repo: Path, token: str, label: str) -> Path:
    candidate = Path(token)
    if candidate.is_absolute() or ".." in candidate.parts or token != candidate.as_posix():
        fail(f"{label} is not a canonical repository-relative path")
    resolved = (repo / candidate).resolve()
    try:
        resolved.relative_to(repo.resolve())
    except ValueError:
        fail(f"{label} escapes the repository")
    return resolved


def provenance_path(repo: Path, sha: str, role: str) -> Path:
    suffix = {
        "source": ".cpp",
        "runner": ".runner.py",
        "verifier": ".leaf_verify.py",
    }[role]
    return repo / PROVENANCE_REL / f"{sha}{suffix}"


def validate_snapshot(
    repo: Path,
    retained: Mapping[str, str],
    prefix: str,
    expected_sha: str,
    role: str,
) -> Path:
    if retained[f"{prefix}_SHA256"] != expected_sha:
        fail(f"{prefix} hash claim mismatch")
    if retained[f"{prefix}_SNAPSHOT_STATUS"] != "PRESENT":
        fail(f"{prefix} snapshot must be PRESENT")
    expected_path = provenance_path(repo, expected_sha, role)
    expected_token = expected_path.relative_to(repo).as_posix()
    if retained[f"{prefix}_SNAPSHOT_PATH"] != expected_token:
        fail(f"{prefix} snapshot path claim is not the canonical content address")
    claimed = repo_relative_path(
        repo, retained[f"{prefix}_SNAPSHOT_PATH"], f"{prefix} snapshot path"
    )
    if claimed != expected_path.resolve():
        fail(f"{prefix} snapshot path is not content-addressed as expected")
    require_text_file(claimed)
    if digest(claimed) != expected_sha:
        fail(f"{prefix} snapshot content hash mismatch")
    return claimed


def validate_provenance_tree(repo: Path) -> None:
    expected = {
        provenance_path(repo, SOURCE_SHA, "source").name,
        provenance_path(repo, RUNNER_SHA, "runner").name,
        provenance_path(repo, REPLAY_VERIFIER_SHA, "verifier").name,
        provenance_path(repo, RUN_SPECS[0].original_verifier_sha, "verifier").name,
    }
    entries, _ = scan_text_tree(repo / PROVENANCE_REL, set())
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        fail(f"shared provenance file set mismatch: missing={missing}, extra={extra}")
    for relative, actual in entries.items():
        prefix = relative.split(".", 1)[0]
        if actual != prefix:
            fail(f"content-addressed provenance name mismatch: {relative}")


def leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def leaf_challenge(root: str, identity: str, input_sha: str) -> str:
    material = (
        CHALLENGE_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + identity.encode("ascii")
        + b"\0"
        + bytes.fromhex(input_sha)
    )
    return digest_bytes(material)


def expected_leaves(contract: Mapping[str, str]) -> set[str]:
    grid = parse_uint(contract["GRID"], "GRID")
    if grid < 1:
        fail("GRID must be positive")
    pairs: list[tuple[int, int]] = []
    for token in contract["DEPTH_PAIRS"].split(","):
        match = PAIR_RE.fullmatch(token)
        if match is None:
            fail(f"malformed depth pair: {token}")
        pair = (int(match.group(1)), int(match.group(2)))
        if pair in pairs or pair[0] > 30 or pair[1] > 30:
            fail(f"invalid or duplicate depth pair: {token}")
        if grid > (1 << pair[0]) or grid > (1 << pair[1]):
            fail(f"grid exceeds dyadic depth for pair: {token}")
        pairs.append(pair)
    if not pairs:
        fail("empty depth-pair contract")
    leaves = {
        leaf_id(
            u_depth,
            ((2 * u_position + 1) * (1 << u_depth)) // (2 * grid),
            s_depth,
            ((2 * s_position + 1) * (1 << s_depth)) // (2 * grid),
        )
        for u_depth, s_depth in pairs
        for u_position in range(grid)
        for s_position in range(grid)
    }
    include_root = parse_bool(contract["INCLUDE_ROOT"], "INCLUDE_ROOT")
    if include_root:
        root = leaf_id(0, 0, 0, 0)
        if root in leaves:
            fail("root duplicates a scout stratum")
        leaves.add(root)
    return leaves


def parse_scout(path: Path) -> tuple[LeafRow, ...]:
    raw = canonical_bytes(path)
    lines = raw.decode("ascii").splitlines()
    if not lines or tuple(lines[0].split("\t")) != SCOUT_HEADER:
        fail(f"scout header mismatch in {path}")
    rows: list[LeafRow] = []
    seen: set[str] = set()
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(SCOUT_HEADER):
            fail(f"scout row width mismatch in {path}")
        values = dict(zip(SCOUT_HEADER, fields, strict=True))
        identity = values["LEAF_ID"]
        match = LEAF_ID_RE.fullmatch(identity)
        if match is None or identity in seen:
            fail(f"malformed or duplicate leaf identity: {identity}")
        seen.add(identity)
        u_depth = parse_uint(values["U_DEPTH"], f"{identity} U_DEPTH")
        u_index = parse_uint(values["U_INDEX"], f"{identity} U_INDEX")
        s_depth = parse_uint(values["S_DEPTH"], f"{identity} S_DEPTH")
        s_index = parse_uint(values["S_INDEX"], f"{identity} S_INDEX")
        if identity != leaf_id(u_depth, u_index, s_depth, s_index):
            fail(f"leaf identity does not match its coordinates: {identity}")
        for value, label in (
            (values["INPUT_SHA256"], "input"),
            (values["LEAF_CHALLENGE"], "challenge"),
            (values["RECEIPT_SHA256"], "receipt"),
            (values["VERIFICATION_SHA256"], "verification"),
            (values["PHYSICAL_SHA256"], "physical"),
        ):
            require_sha(value, f"{identity} {label} hash")
        status = values["STATUS"]
        if status not in {
            "CERTIFIED",
            "SUBDIVISION_REQUIRED",
            "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN",
            "COMPUTATION_UNRESOLVED_TIMEOUT",
        }:
            fail(f"unknown leaf status: {identity} {status}")
        method = values["METHOD"]
        if method not in {
            "NONE",
            "AFFINE",
            "PROJECTIVE_X",
            "PROJECTIVE_Y",
            "PROJECTIVE_PLUS",
            "PROJECTIVE_MINUS",
        }:
            fail(f"unknown leaf method: {identity} {method}")
        rows.append(
            LeafRow(
                identity,
                u_depth,
                u_index,
                s_depth,
                s_index,
                status,
                method,
                parse_bool(values["CERTIFICATE_PASS"], f"{identity} certificate"),
                parse_bool(values["SUBDIVISION_REQUIRED"], f"{identity} subdivision"),
                values["INPUT_SHA256"],
                values["LEAF_CHALLENGE"],
                values["RECEIPT_SHA256"],
                values["VERIFICATION_SHA256"],
                values["PHYSICAL_SHA256"],
                parse_uint(values["WORKER_RC"], f"{identity} WORKER_RC"),
                parse_uint(values["ELAPSED_MS"], f"{identity} ELAPSED_MS"),
            )
        )
    return tuple(rows)


def child_txt_ids(directory: Path) -> set[str]:
    if directory.is_symlink() or not directory.is_dir():
        fail(f"missing retained subdirectory: {directory}")
    result: set[str] = set()
    for child in directory.iterdir():
        if child.is_symlink() or not child.is_file() or child.suffix != ".txt":
            fail(f"unexpected retained leaf artifact: {child}")
        result.add(child.stem)
    return result


def validate_sidecar_and_manifest(
    repo: Path, receipts_root: Path, spec: RunSpec
) -> tuple[Path, dict[str, str], dict[str, str], Path]:
    directory = receipts_root / spec.directory
    retained_path = directory / "retained-manifest.txt"
    retained = parse_kv(retained_path, RETAINED_KEYS)
    expected_fixed = {
        "SCHEMA": "sounio.cs6.c1-full-source-cover-retained-scout.v1",
        "EVIDENCE_CLASS": "LOCAL_UNATTESTED_HISTORICAL_SCOUT",
        "PROMOTION_ELIGIBLE": "false",
        "RAW_RUN_MANIFEST_SHA256": spec.raw_manifest_sha,
        "RETAINED_FILES_INDEX_SHA256": spec.retained_index_sha,
        "RAW_RUN_MANIFEST_TRUST": "SELF_REPORTED_NO_ATTESTATION",
        "ORIGINAL_WORKER_BINARY_SHA256": WORKER_BINARY_SHA,
        "ORIGINAL_WORKER_BINARY_RETAINED": "false",
        "EXACT_EXECUTION_REPLAYABLE": "false",
        "REPLAY_VERIFIER_SHA256": REPLAY_VERIFIER_SHA,
        "REPLAYABLE_RECEIPT_COUNT": str(spec.replayable_count),
        "REPLAY_OUTPUT_MATCH_COUNT": str(spec.replayable_count),
        "MUTATION_AUDIT_REPLAY_MATCH": "true",
    }
    for key, expected in expected_fixed.items():
        if retained[key] != expected:
            fail(f"retained sidecar claim mismatch: {spec.directory} {key}")

    source_snapshot = validate_snapshot(
        repo, retained, "ORIGINAL_SOURCE", SOURCE_SHA, "source"
    )
    validate_snapshot(repo, retained, "ORIGINAL_RUNNER", RUNNER_SHA, "runner")
    validate_snapshot(
        repo,
        retained,
        "ORIGINAL_VERIFIER",
        spec.original_verifier_sha,
        "verifier",
    )
    replay_path = repo_relative_path(
        repo,
        retained["REPLAY_VERIFIER_SNAPSHOT_PATH"],
        "replay verifier snapshot path",
    )
    expected_replay = provenance_path(repo, REPLAY_VERIFIER_SHA, "verifier").resolve()
    expected_replay_token = expected_replay.relative_to(repo.resolve()).as_posix()
    if (
        retained["REPLAY_VERIFIER_SNAPSHOT_PATH"] != expected_replay_token
        or replay_path != expected_replay
    ):
        fail(f"replay verifier path mismatch: {spec.directory}")
    require_text_file(replay_path)
    if digest(replay_path) != REPLAY_VERIFIER_SHA:
        fail("replay verifier content hash mismatch")

    raw_entries, raw_index = scan_text_tree(directory, {"retained-manifest.txt"})
    if raw_index != spec.retained_index_sha:
        fail(f"retained raw-file index mismatch: {spec.directory}")
    if "worker-binary" in raw_entries:
        fail(f"worker binary unexpectedly retained: {spec.directory}")

    manifest_path = directory / "run-manifest.txt"
    if digest(manifest_path) != spec.raw_manifest_sha:
        fail(f"raw run manifest hash mismatch: {spec.directory}")
    manifest = parse_kv(manifest_path, RUN_MANIFEST_KEYS)
    fixed_manifest = {
        "SCHEMA": "sounio.cs6.c1-full-source-cover-scout-manifest.v1",
        "RUN_COMPLETE": "true",
        "CAPD_VERSION": "5.3.0",
        "INTERVAL_BACKEND": "FILIB",
        "OPTIMIZATION_LEVEL": "O0",
        "SOURCE_SHA256": SOURCE_SHA,
        "LEAF_COUNT": str(spec.leaf_count),
        "CERTIFIED_COUNT": str(spec.certified_count),
        "MUTATION_TESTS": "56",
        "SCOUT_ONLY": "true",
        "EXECUTION_TRUST_MODEL": "LOCAL_BOUNDED_CAPD_CPU_NO_ATTESTATION",
        "REMOTE_ATTESTATION_PRESENT": "false",
        "INDEPENDENT_REPLAY_REQUIRED": "true",
        "PROMOTION_ELIGIBLE": "false",
        "FULL_SOURCE_CARRIER_PROVED": "false",
        "WORKER_SOURCE_CPP_SHA256": SOURCE_SHA,
        "LEAF_VERIFIER_PY_SHA256": spec.original_verifier_sha,
        "RUNNER_PY_SHA256": RUNNER_SHA,
        "WORKER_BINARY_SHA256": WORKER_BINARY_SHA,
    }
    for key, expected in fixed_manifest.items():
        if manifest[key] != expected:
            fail(f"raw manifest claim mismatch: {spec.directory} {key}")
    require_sha(manifest["ROOT_CHALLENGE"], f"{spec.directory} root challenge")
    for key, file_name in RETAINED_HASH_FILES.items():
        if digest(directory / file_name) != manifest[key]:
            fail(f"raw manifest artifact hash mismatch: {spec.directory} {file_name}")
    if digest(source_snapshot) != manifest["WORKER_SOURCE_CPP_SHA256"]:
        fail(f"source provenance mismatch: {spec.directory}")
    return directory, retained, manifest, replay_path


def validate_run_structure(
    spec: RunSpec,
    directory: Path,
    retained: Mapping[str, str],
    manifest: Mapping[str, str],
    replay_path: Path,
) -> RunEvidence:
    contract = parse_kv(directory / "run-contract.txt", RUN_CONTRACT_KEYS)
    fixed_contract = {
        "SCHEMA": "sounio.cs6.c1-full-source-cover-scout-contract.v1",
        "SOURCE": "N0",
        "ROOT_CHALLENGE": manifest["ROOT_CHALLENGE"],
        "SCOUT_ONLY": "true",
        "FULL_SOURCE_CARRIER_PROVED": "false",
    }
    for key, expected in fixed_contract.items():
        if contract[key] != expected:
            fail(f"run contract mismatch: {spec.directory} {key}")
    if parse_uint(contract["JOBS"], "JOBS") < 1:
        fail(f"non-positive JOBS in {spec.directory}")
    if parse_uint(contract["TIMEOUT_SECONDS"], "TIMEOUT_SECONDS") < 1:
        fail(f"non-positive timeout in {spec.directory}")

    rows = parse_scout(directory / "scout.tsv")
    expected = expected_leaves(contract)
    identities = {row.identity for row in rows}
    if len(rows) != spec.leaf_count or identities != expected:
        fail(f"scout cardinality or leaf set mismatch: {spec.directory}")

    expected_verifications = {
        row.identity for row in rows if row.verification_sha != ZERO_SHA
    }
    for child, expected_ids in (
        ("inputs", identities),
        ("receipts", identities),
        ("stderr", identities),
        ("verifications", expected_verifications),
    ):
        actual_ids = child_txt_ids(directory / child)
        if actual_ids != expected_ids:
            fail(f"{child} file set mismatch: {spec.directory}")

    certified = 0
    subdivision = 0
    unresolved = 0
    for row in rows:
        if row.elapsed_ms < 1:
            fail(f"non-positive elapsed time: {row.identity}")
        input_path = directory / "inputs" / f"{row.identity}.txt"
        expected_input = leaf_input_bytes(
            row.u_depth, row.u_index, row.s_depth, row.s_index
        )
        if input_path.read_bytes() != expected_input or digest_bytes(expected_input) != row.input_sha:
            fail(f"leaf input mismatch: {row.identity}")
        expected_challenge = leaf_challenge(
            manifest["ROOT_CHALLENGE"], row.identity, row.input_sha
        )
        if row.challenge != expected_challenge:
            fail(f"leaf challenge mismatch: {row.identity}")
        receipt_path = directory / "receipts" / f"{row.identity}.txt"
        if digest(receipt_path) != row.receipt_sha:
            fail(f"receipt hash mismatch: {row.identity}")
        stderr_path = directory / "stderr" / f"{row.identity}.txt"

        if row.certificate == row.subdivision:
            fail(f"inconsistent certificate/subdivision flags: {row.identity}")
        if row.status == "CERTIFIED":
            certified += 1
            if not row.certificate or row.method == "NONE" or row.worker_rc != 0:
                fail(f"invalid certified row: {row.identity}")
        elif row.status == "SUBDIVISION_REQUIRED":
            subdivision += 1
            if row.certificate or not row.subdivision or row.method != "NONE" or row.worker_rc != 0:
                fail(f"invalid subdivision row: {row.identity}")
        else:
            unresolved += 1
            if row.certificate or not row.subdivision or row.method != "NONE" or row.worker_rc == 0:
                fail(f"invalid unresolved row: {row.identity}")

        if row.verification_sha == ZERO_SHA:
            if (
                row.physical_sha != ZERO_SHA
                or not row.status.startswith("COMPUTATION_UNRESOLVED")
                or receipt_path.stat().st_size != 0
            ):
                fail(f"invalid absent-verification sentinel: {row.identity}")
        else:
            verification_path = directory / "verifications" / f"{row.identity}.txt"
            if digest(verification_path) != row.verification_sha:
                fail(f"verification output hash mismatch: {row.identity}")
            verification = parse_kv(verification_path, VERIFICATION_KEYS)
            expected_verification = {
                "VERIFICATION_SCHEMA": "sounio.cs6.c1-full-source-cover-leaf-verification.v1",
                "RECEIPT_SHA256": row.receipt_sha,
                "PHYSICAL_SHA256": row.physical_sha,
                "MUTATION_TESTS": "0",
                "MUTATIONS_REJECTED": "0",
                "LEAF_METHOD": row.method,
                "SUBDIVISION_REQUIRED": str(row.subdivision).lower(),
                "CERTIFICATE_PASS": str(row.certificate).lower(),
            }
            if verification != expected_verification:
                fail(f"verification/index semantic mismatch: {row.identity}")
        if row.worker_rc == 0 and stderr_path.stat().st_size != 0:
            fail(f"successful worker retained stderr: {row.identity}")

    if certified != spec.certified_count:
        fail(f"certified count mismatch: {spec.directory}")
    if len(rows) - unresolved != spec.replayable_count:
        fail(f"replayable count mismatch: {spec.directory}")

    summary = parse_kv(directory / "summary.txt", SUMMARY_KEYS)
    expected_summary = {
        "SCHEMA": "sounio.cs6.c1-full-source-cover-scout-summary.v1",
        "SCOUT_ONLY": "true",
        "LEAF_COUNT": str(len(rows)),
        "CERTIFIED_COUNT": str(certified),
        "SUBDIVISION_REQUIRED_COUNT": str(subdivision),
        "COMPUTATION_UNRESOLVED_COUNT": str(unresolved),
        "MUTATION_TESTS": "56",
        "MUTATIONS_REJECTED": "56",
        "FULL_SOURCE_CARRIER_PROVED": "false",
        "PROJECTIVE_RICCATI_INTEGRATED": "false",
        "HYPERBOLICITY_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false",
        "U250_USED": "false",
    }
    if summary != expected_summary:
        fail(f"summary/index semantic mismatch: {spec.directory}")

    nonempty_stderr = [
        row
        for row in rows
        if (directory / "stderr" / f"{row.identity}.txt").stat().st_size != 0
    ]
    if spec.unresolved_id is None:
        if nonempty_stderr:
            fail(f"unexpected nonempty leaf stderr: {spec.directory}")
    else:
        if [row.identity for row in nonempty_stderr] != [spec.unresolved_id]:
            fail(f"root stderr identity mismatch: {spec.directory}")
        row = nonempty_stderr[0]
        stderr_path = directory / "stderr" / f"{row.identity}.txt"
        if digest(stderr_path) != spec.unresolved_stderr_sha:
            fail("root interval-domain stderr hash mismatch")
        stderr = stderr_path.read_bytes().lower()
        if b"interval error:" not in stderr or b"division by 0" not in stderr:
            fail("root stderr is not the retained interval-domain failure")
        if row.status != "COMPUTATION_UNRESOLVED_INTERVAL_DOMAIN":
            fail("root stderr classification mismatch")

    if parse_uint(retained["REPLAYABLE_RECEIPT_COUNT"], "replayable count") != spec.replayable_count:
        fail(f"sidecar replayable count mismatch: {spec.directory}")
    if parse_uint(retained["REPLAY_OUTPUT_MATCH_COUNT"], "replay output count") != spec.replayable_count:
        fail(f"sidecar replay output count mismatch: {spec.directory}")
    return RunEvidence(spec, directory, manifest, retained, rows, replay_path)


def replay_leaf(evidence: RunEvidence, row: LeafRow) -> str:
    command = [
        sys.executable,
        str(evidence.replay_verifier),
        str(evidence.directory / "receipts" / f"{row.identity}.txt"),
        "--source-sha",
        SOURCE_SHA,
        "--input",
        str(evidence.directory / "inputs" / f"{row.identity}.txt"),
        "--challenge",
        row.challenge,
    ]
    try:
        replay = subprocess.run(command, capture_output=True, timeout=180)
    except subprocess.TimeoutExpired as error:
        raise VerificationError(f"current-verifier replay timed out: {row.identity}") from error
    if replay.returncode != 0 or replay.stderr:
        fail(f"current-verifier replay failed: {row.identity}")
    expected = evidence.directory / "verifications" / f"{row.identity}.txt"
    if replay.stdout != expected.read_bytes():
        fail(f"current-verifier output is not byte-identical: {row.identity}")
    return row.identity


def replay_mutation_audit(evidence: RunEvidence) -> None:
    audit_path = evidence.directory / "mutation-audit.txt"
    audit = parse_kv(audit_path, VERIFICATION_KEYS)
    expected_fixed = {
        "VERIFICATION_SCHEMA": "sounio.cs6.c1-full-source-cover-leaf-verification.v1",
        "MUTATION_TESTS": "56",
        "MUTATIONS_REJECTED": "56",
        "SUBDIVISION_REQUIRED": "false",
        "CERTIFICATE_PASS": "true",
    }
    for key, expected in expected_fixed.items():
        if audit[key] != expected:
            fail(f"mutation audit field mismatch: {evidence.spec.directory} {key}")
    candidates = [row for row in evidence.rows if row.receipt_sha == audit["RECEIPT_SHA256"]]
    if len(candidates) != 1 or not candidates[0].certificate:
        fail(f"mutation audit receipt is not a unique certified leaf: {evidence.spec.directory}")
    row = candidates[0]
    if row.physical_sha != audit["PHYSICAL_SHA256"] or row.method != audit["LEAF_METHOD"]:
        fail(f"mutation audit/index mismatch: {evidence.spec.directory}")
    command = [
        sys.executable,
        str(evidence.replay_verifier),
        str(evidence.directory / "receipts" / f"{row.identity}.txt"),
        "--source-sha",
        SOURCE_SHA,
        "--input",
        str(evidence.directory / "inputs" / f"{row.identity}.txt"),
        "--challenge",
        row.challenge,
        "--self-test-mutations",
        "--require-terminal",
    ]
    try:
        replay = subprocess.run(command, capture_output=True, timeout=300)
    except subprocess.TimeoutExpired as error:
        raise VerificationError(
            f"mutation audit replay timed out: {evidence.spec.directory}"
        ) from error
    if replay.returncode != 0 or replay.stderr or replay.stdout != audit_path.read_bytes():
        fail(f"mutation audit is not byte-identical: {evidence.spec.directory}")
    if evidence.retained["MUTATION_AUDIT_REPLAY_MATCH"] != "true":
        fail(f"sidecar mutation replay claim mismatch: {evidence.spec.directory}")


def verify_all(repo: Path, jobs: int) -> tuple[int, int]:
    receipts_root = repo / "scripts/research/receipts"
    validate_provenance_tree(repo)
    evidence_runs: list[RunEvidence] = []
    for spec in RUN_SPECS:
        directory, retained, manifest, replay_path = validate_sidecar_and_manifest(
            repo, receipts_root, spec
        )
        evidence_runs.append(
            validate_run_structure(
                spec, directory, retained, manifest, replay_path
            )
        )

    tasks = [
        (evidence, row)
        for evidence in evidence_runs
        for row in evidence.rows
        if row.verification_sha != ZERO_SHA
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(replay_leaf, evidence, row) for evidence, row in tasks]
        for future in futures:
            future.result()
    for evidence in evidence_runs:
        replay_mutation_audit(evidence)
    return sum(len(evidence.rows) for evidence in evidence_runs), len(tasks)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="parallel current-verifier replay processes",
    )
    args = parser.parse_args(argv)
    if args.jobs < 1:
        fail("--jobs must be positive")
    repo = Path(__file__).resolve().parents[2]
    leaf_count, replay_count = verify_all(repo, args.jobs)
    print("SCHEMA=sounio.cs6.c1-full-source-cover-retained-verification.v1")
    print(f"RUN_COUNT={len(RUN_SPECS)}")
    print(f"LEAF_COUNT={leaf_count}")
    print(f"REPLAYABLE_RECEIPT_COUNT={replay_count}")
    print(f"REPLAY_OUTPUT_MATCH_COUNT={replay_count}")
    print(f"MUTATION_AUDIT_MATCH_COUNT={len(RUN_SPECS)}")
    print("PROVENANCE_SNAPSHOT_COUNT=4")
    print("PROMOTION_ELIGIBLE=false")
    print("CERTIFICATE_PASS=true")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"retained verification error: {error}", file=sys.stderr)
        raise SystemExit(1)
