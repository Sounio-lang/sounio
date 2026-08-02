#!/usr/bin/env python3
"""Verify that a completed KAT run authorizes one later adaptive run."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import io
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Mapping, Sequence


sys.dont_write_bytecode = True

SHA_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
JOB_RE = re.compile(r"^[1-9][0-9]*$")
ZERO_SHA256 = "0" * 64
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
CERTIFICATE_SCHEMA = "sounio.cs6.hapg-full-source-cover-kat-prerequisite.v2"
EMPTY_DIRS = (
    "inputs",
    "hpg-receipts",
    "hpg-stderr",
    "hpg-verifications",
    "hapg-receipts",
    "hapg-stderr",
    "hapg-verifications",
    "wave-contracts",
    "wave-results",
)

SOURCE_SNAPSHOT_FILES = frozenset(
    {
        "cs6_plucker_cocycle_probe.cpp",
        "cs6_plucker_cocycle_verify.py",
        "cs6_hapg_full_source_cover_worker.cpp",
        "cs6_affine_projective_cocycle_full53_probe.cpp",
        "cs6_affine_projective_cocycle_full53_verify.py",
        "cs6_hapg_full_source_cover_verify.py",
        "cs6_hapg_full_source_cover_run.py",
        "cs6_hapg_full_source_cover_aggregate.py",
        "cs6_hapg_full_source_cover_kat_anchor.py",
        "cs6_c1_full_source_cover_aggregate.py",
        "cs6_hapg_full_source_cover_gate.sh",
        "cs6_hapg_full_source_cover_slurm_job.sh",
        "cs6_hapg_full_source_cover_contract_v6.txt",
    }
)

HISTORICAL_SNAPSHOT_FILES = frozenset(
    {
        "v5-executed-contract.txt",
        "v4-executed-contract.txt",
        "v3-executed-contract.txt",
        "v2-abort-manifest.txt",
        "v2-abort-sacct.txt",
        "v2-abort-config.txt",
        "v2-abort-stderr.txt",
        "v3-abort-manifest.txt",
        "v3-abort-sacct.txt",
        "v3-abort-config.txt",
        "v3-abort-slurm-stderr.txt",
        "v3-abort-repro-s0-stdout.txt",
        "v3-abort-repro-s0-stderr.txt",
        "v3-abort-repro-s1-stdout.txt",
        "v3-abort-repro-s1-stderr.txt",
        "v3-abort-hpg-full255-census.tsv",
        "v3-abort-hpg-full255-census-summary.txt",
        "v3-abort-hpg-full255-stderr.jsonl",
        "v3-abort-challenge-spotcheck.json",
        "v4-abort-manifest.txt",
        "v4-abort-files.sha256",
        "v4-abort-sacct.txt",
        "v4-abort-config.txt",
        "v4-abort-slurm-stdout.txt",
        "v4-abort-hpg-rc0-corpus.tar",
        "v4-abort-corpus-files.sha256",
        "v4-abort-hpg-rc0-verifier-census.tsv",
        "v4-abort-hpg-rc0-verifier-census-summary.txt",
        "v4-abort-hpg-v5-kat-compat.tsv",
        "v4-abort-hpg-v4-kat-corpus.tar",
        "v4-abort-hpg-v4-kat-corpus-files.sha256",
        "v4-abort-midpoint-discrete-negative-test.txt",
        "v4-abort-local-repro.tar",
        "v4-abort-v4-hpg-verifier.py",
        "v5-abort-manifest.txt",
        "v5-abort-files.sha256",
        "v5-abort-sacct.psv",
    }
)

KAT_RESULT_DIRECT_FILES = SOURCE_SNAPSHOT_FILES | HISTORICAL_SNAPSHOT_FILES | frozenset(
    {
        "build-mode.txt",
        "evaluations.tsv",
        "files.sha256",
        "git-head.txt",
        "git-status.txt",
        "hapg-worker-binary",
        "hpg-worker-binary",
        "kat-coordinates.tsv",
        "kat-expected-results.tsv",
        "negative-outcomes.tsv",
        "python-version.txt",
        "run-contract.txt",
        "run-manifest.txt",
        "runtime-libraries.sha256",
        "runtime-linkage.txt",
        "slurm-job-record.txt",
        "summary.txt",
        "timings.tsv",
        "waves.tsv",
    }
)

PREBUILT_DIRECT_FILES = SOURCE_SNAPSHOT_FILES | HISTORICAL_SNAPSHOT_FILES | frozenset(
    {
        "build-mode.txt",
        "capd-cflags.txt",
        "capd-libs.txt",
        "capd-version.txt",
        "compiler-version.txt",
        "files.sha256",
        "git-head.txt",
        "git-status.txt",
        "hapg-compile-command.txt",
        "hapg-compile-stderr.txt",
        "hapg-compile-stdout.txt",
        "hapg-dependencies.d",
        "hapg-dependencies.sha256",
        "hapg-worker-binary",
        "hpg-compile-command.txt",
        "hpg-compile-stderr.txt",
        "hpg-compile-stdout.txt",
        "hpg-dependencies.d",
        "hpg-dependencies.sha256",
        "hpg-worker-binary",
        "link-inputs.sha256",
        "python-version.txt",
        "run-manifest.txt",
        "runtime-libraries.sha256",
        "runtime-linkage.txt",
    }
)

CERTIFICATE_KEYS = (
    "SCHEMA",
    "CERTIFICATE_SCOPE",
    "KAT_SCHEMA_PROFILE",
    "KAT_PREREQUISITE_VALID",
    "KAT_JOB_ID",
    "KAT_ARCHIVE_BASENAME",
    "KAT_ARCHIVE_SHA256",
    "KAT_ARCHIVE_SIDECAR_SHA256",
    "KAT_ARCHIVE_MEMBER_COUNT",
    "KAT_ARCHIVE_REGULAR_FILE_COUNT",
    "KAT_ARCHIVE_DIRECTORY_COUNT",
    "KAT_TRANSPORT_MANIFEST_SHA256",
    "KAT_TRANSPORT_JOB_RECORD_SHA256",
    "KAT_RESULT_RUN_MANIFEST_SHA256",
    "KAT_RESULT_FILES_INDEX_SHA256",
    "KAT_RESULT_FILES_INDEX_ENTRY_COUNT",
    "KAT_RESULT_RUN_CONTRACT_SHA256",
    "KAT_RESULT_SUMMARY_SHA256",
    "KAT_RESULT_JOB_RECORD_SHA256",
    "KAT_PREBUILT_RUN_MANIFEST_SHA256",
    "KAT_PREBUILT_FILES_INDEX_SHA256",
    "KAT_PREBUILT_FILES_INDEX_ENTRY_COUNT",
    "KAT_GIT_HEAD_FILE_SHA256",
    "KAT_GIT_STATUS_FILE_SHA256",
    "KAT_SACCT_SHA256",
    "KAT_SACCT_STATE",
    "KAT_SACCT_EXIT_CODE",
    "KAT_SUBMIT_UTC",
    "KAT_START_UTC",
    "KAT_END_UTC",
    "KAT_CLUSTER",
    "KAT_ACCOUNT",
    "KAT_QOS",
    "KAT_USER",
    "KAT_UID",
    "KAT_RESTARTS",
    "KAT_NODE",
    "KAT_ALLOC_NODES",
    "KAT_ALLOC_TASKS",
    "KAT_ALLOC_CPUS",
    "KAT_REQ_CPUS",
    "KAT_CONFIG_SHA256",
    "KAT_ROOT_CHALLENGE",
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
    "KAT_EXPECTED_GIT_HEAD",
    "KAT_FROZEN_CONTRACT_SHA256",
    "KAT_BASE_REPO_BUNDLE_SHA256",
    "KAT_BASE_GIT_HEAD",
    "KAT_REPO_DELTA_BUNDLE_SHA256",
    "KAT_PREBUILT_ARCHIVE_SHA256",
    "KAT_SLURM_JOB_SCRIPT_SHA256",
    "KAT_ANCHOR_SOURCE_SHA256",
    "ADAPTIVE_JOB_ID",
    "ADAPTIVE_SUBMIT_UTC",
    "KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT",
    "EXECUTION_PROVENANCE_ATTESTED",
    "PROMOTION_ELIGIBLE",
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

PREBUILT_MANIFEST_BASE_KEYS = (
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
    "EXACT_TREE_KERNEL_SHA256",
    "GATE_SHA256",
    "SLURM_JOB_SCRIPT_SHA256",
    "HPG_WORKER_BINARY_SHA256",
    "HAPG_WORKER_BINARY_SHA256",
    "FILES_INDEX_SHA256",
    "FILE_COUNT",
    "PROMOTION_ELIGIBLE",
)

RUN_CONTRACT_BASE_KEYS = (
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
    "KAT_COORDINATE_MANIFEST_SHA256",
    "KAT_EXPECTED_RESULTS_SHA256",
)

TRANSPORT_V2_KEYS = (
    "SCHEMA",
    "MODE",
    "SLURM_JOB_ID",
    "SLURM_NODE",
    "EXPECTED_GIT_HEAD",
    "EXPECTED_CONTRACT_SHA256",
    "SLURM_JOB_SCRIPT_SHA256",
    "CONFIG_SHA256",
    "PYTHON_EXECUTABLE_REALPATH",
    "BASE_REPO_BUNDLE_SHA256",
    "BASE_GIT_HEAD",
    "REPO_DELTA_BUNDLE_SHA256",
    "PREBUILT_ARCHIVE_SHA256",
    "RESULT_RUN_MANIFEST_SHA256",
    "RESULT_FILES_INDEX_SHA256",
    "AGGREGATION_SHA256",
    "EXECUTION_PROVENANCE_ATTESTED",
    "PROMOTION_ELIGIBLE",
)

TRANSPORT_V3_KEYS = TRANSPORT_V2_KEYS[:13] + (
    "KAT_PREREQUISITE_REQUIRED",
    "KAT_JOB_ID",
    "KAT_ARCHIVE_SHA256",
    "KAT_CERTIFICATE_SHA256",
) + TRANSPORT_V2_KEYS[13:16] + (
    "POST_RUN_GATE_PASS",
    "FAILURE_STAGE",
    "FAILURE_RC",
    "EXECUTION_PROVENANCE_ATTESTED",
    "PROMOTION_ELIGIBLE",
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

SLURM_CONFIG_KEYS = (
    "SCHEMA",
    "MODE",
    "BASE_REPO_BUNDLE_PATH",
    "BASE_REPO_BUNDLE_SHA256",
    "BASE_GIT_HEAD",
    "REPO_DELTA_BUNDLE_PATH",
    "REPO_DELTA_BUNDLE_SHA256",
    "PREBUILT_ARCHIVE_PATH",
    "PREBUILT_ARCHIVE_SHA256",
    "EXPECTED_GIT_HEAD",
    "EXPECTED_CONTRACT_SHA256",
    "OUTPUT_DIRECTORY",
)

WAVE_HEADERS: tuple[tuple[str, str | None], ...] = (
    ("SCHEMA", "sounio.cs6.hapg-full-source-cover-wave-contract.v1"),
    ("RUN_CONTRACT_SHA256", None),
    ("ROOT_CHALLENGE", None),
    ("WAVE_INDEX", None),
    ("PREVIOUS_WAVE_RESULT_SHA256", None),
    ("FRONTIER_SHA256", None),
    ("NODE_COUNT", None),
    ("HPG_WORKER_SOURCE_SHA256", None),
    ("HPG_VERIFIER_SOURCE_SHA256", None),
    ("HAPG_WORKER_SOURCE_SHA256", None),
    ("HAPG_KERNEL_SOURCE_SHA256", None),
    ("HAPG_VERIFIER_ADAPTER_SHA256", None),
    ("HAPG_NUMERIC_VERIFIER_SHA256", None),
    (
        "FREEZE_ORDER",
        "ALL_HPG_ATTEMPTS_VERIFIED_THEN_ATOMIC_WAVE_CONTRACT_THEN_ANY_HAPG",
    ),
)

WAVE_COLUMNS = (
    "WAVE_INDEX",
    "NODE_ID",
    "PARENT_ID",
    "U_DEPTH",
    "U_INDEX",
    "S_DEPTH",
    "S_INDEX",
    "INPUT_SHA256",
    "HPG_CHALLENGE",
    "HPG_STATUS",
    "HPG_RC",
    "HPG_RECEIPT_SHA256",
    "HPG_STDERR_SHA256",
    "HPG_VERIFICATION_SHA256",
    "HPG_PHYSICAL_SHA256",
    "HPG_PROBE_PASS",
    "HPG_CERTIFICATE_PASS",
    "E1_R0_CHART",
    "E1_R0_SIGN",
    "E1_R1_CHART",
    "E1_R1_SIGN",
    "E2_R0_CHART",
    "E2_R0_SIGN",
    "E2_R1_CHART",
    "E2_R1_SIGN",
    "HAPG_ELIGIBLE",
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

HPG_VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
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
    "SUBDIVISION_REQUIRED",
    "CERTIFICATE_PASS",
)

HAPG_VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "RECEIPT_SHA256",
    "PHYSICAL_SHA256",
    "WAVE_CONTRACT_SHA256",
    "HPG_RECEIPT_SHA256",
    "HPG_VERIFICATION_SHA256",
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
    "GENERIC_CERTIFICATE_PASS",
    "HAPG_TERMINAL_CERTIFIED",
    "HAPG_SUBDIVISION_REQUIRED",
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

TIMING_COLUMNS = ("WAVE_INDEX", "NODE_ID", "HPG_ELAPSED_MS", "HAPG_ELAPSED_MS")
ROOT_ID = "U00-0000000000_S00-0000000000"
MAX_ARCHIVE_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBERS = 2048
MAX_ARCHIVE_EXPANDED_BYTES = 512 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 128 * 1024 * 1024
SACCT_FORMAT = (
    "JobIDRaw%64,JobName%64,Cluster%64,Partition%64,Account%64,QOS%64,User%64,"
    "UID,Restarts,State%64,ExitCode,Submit,Start,End,ElapsedRaw,NodeList%256,"
    "NNodes,NTasks,AllocCPUS,ReqCPUS"
)


class VerificationError(RuntimeError):
    pass


def _fail(message: str) -> None:
    raise VerificationError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        _fail(f"{label} must be a regular non-symlink file")
    before = path.stat()
    raw = path.read_bytes()
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        _fail(f"{label} changed while being read")
    return raw


def _canonical_kv(fields: Sequence[tuple[str, str]]) -> bytes:
    seen: set[str] = set()
    rows: list[str] = []
    for key, value in fields:
        if (
            not re.fullmatch(r"[A-Z0-9_]+", key)
            or not value
            or any(char in value for char in "\r\n\0=")
            or key in seen
        ):
            _fail("noncanonical certificate field")
        seen.add(key)
        rows.append(f"{key}={value}")
    return ("\n".join(rows) + "\n").encode("ascii")


def _parse_kv(raw: bytes, expected: Sequence[str] | None, label: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail(f"noncanonical {label}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError(f"non-ASCII {label}") from error
    result: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            _fail(f"malformed {label}")
        key, value = line.split("=", 1)
        if not key or not value or key in result:
            _fail(f"duplicate or empty {label} field")
        result[key] = value
    if expected is not None and tuple(result) != tuple(expected):
        _fail(f"{label} field set or order mismatch")
    return result


def _parse_timestamp(token: str, label: str) -> datetime:
    try:
        return datetime.strptime(token, "%Y-%m-%dT%H:%M:%S")
    except ValueError as error:
        raise VerificationError(f"invalid {label} timestamp") from error


def _parse_sacct(
    raw: bytes, expected_job_id: str, *, allow_legacy_v5: bool = False
) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail("noncanonical KAT sacct bytes")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError("non-ASCII KAT sacct bytes") from error
    if len(lines) != 1:
        _fail("KAT sacct must contain exactly one root allocation row")
    columns = lines[0].split("|")
    names = (
        "JOB_ID",
        "JOB_NAME",
        "CLUSTER",
        "PARTITION",
        "ACCOUNT",
        "QOS",
        "USER",
        "UID",
        "RESTARTS",
        "STATE",
        "EXIT_CODE",
        "SUBMIT_UTC",
        "START_UTC",
        "END_UTC",
        "ELAPSED_RAW",
        "NODE",
        "ALLOC_NODES",
        "ALLOC_TASKS",
        "ALLOC_CPUS",
        "REQ_CPUS",
    )
    if len(columns) == 21 and columns[-1] == "":
        result = dict(zip(names, columns[:-1], strict=True))
    elif allow_legacy_v5 and len(columns) == 13 and columns[-1] == "":
        legacy_names = (
            "JOB_ID",
            "JOB_NAME",
            "PARTITION",
            "STATE",
            "EXIT_CODE",
            "SUBMIT_UTC",
            "START_UTC",
            "END_UTC",
            "ELAPSED_RAW",
            "NODE",
            "ALLOC_CPUS",
            "REQ_CPUS",
        )
        legacy = dict(zip(legacy_names, columns[:-1], strict=True))
        result = {
            **legacy,
            "CLUSTER": "HISTORICAL_UNAVAILABLE",
            "ACCOUNT": "HISTORICAL_UNAVAILABLE",
            "QOS": "HISTORICAL_UNAVAILABLE",
            "USER": "HISTORICAL_UNAVAILABLE",
            "UID": "0",
            "RESTARTS": "0",
            "ALLOC_NODES": "0",
            "ALLOC_TASKS": "0",
        }
    else:
        _fail("malformed KAT sacct row")
    if result["JOB_ID"] != expected_job_id or not JOB_RE.fullmatch(result["JOB_ID"]):
        _fail("KAT sacct job identity mismatch")
    if result["STATE"] != "COMPLETED" or result["EXIT_CODE"] != "0:0":
        _fail("KAT sacct is not COMPLETED with zero exit")
    submit = _parse_timestamp(result["SUBMIT_UTC"], "KAT submit")
    start = _parse_timestamp(result["START_UTC"], "KAT start")
    end = _parse_timestamp(result["END_UTC"], "KAT end")
    if not submit <= start <= end:
        _fail("KAT sacct chronology is invalid")
    for key in (
        "UID",
        "RESTARTS",
        "ELAPSED_RAW",
        "ALLOC_NODES",
        "ALLOC_CPUS",
        "REQ_CPUS",
    ):
        if not re.fullmatch(r"(?:0|[1-9][0-9]*)", result[key]):
            _fail(f"KAT sacct {key} is noncanonical")
    if result["ALLOC_TASKS"] and re.fullmatch(
        r"(?:0|[1-9][0-9]*)", result["ALLOC_TASKS"]
    ) is None:
        _fail("KAT sacct ALLOC_TASKS is noncanonical")
    return result


def query_live_sacct(
    kat_job_id: str,
    *,
    sacct_bin: str = "sacct",
    start_date: str = "1970-01-01",
    timeout_seconds: int = 30,
) -> bytes:
    if not JOB_RE.fullmatch(kat_job_id):
        _fail("invalid KAT job id")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", start_date) is None:
        _fail("invalid KAT sacct start date")
    command = [
        sacct_bin,
        "--jobs",
        kat_job_id,
        "--allocations",
        "--noheader",
        "--parsable",
        f"--format={SACCT_FORMAT}",
        "--starttime",
        start_date,
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=timeout_seconds,
            env={
                **os.environ,
                "TZ": "UTC",
                "SLURM_TIME_FORMAT": "standard",
            },
        )
    except subprocess.CalledProcessError as error:
        stderr = (error.stderr or b"").decode("utf-8", errors="replace").strip()
        raise VerificationError(
            f"KAT sacct query failed with rc={error.returncode}: {stderr or 'no stderr'}"
        ) from error
    except (OSError, subprocess.SubprocessError) as error:
        raise VerificationError(f"KAT sacct query failed: {error}") from error
    try:
        rows = [line.strip() for line in completed.stdout.decode("ascii").splitlines()]
    except UnicodeError as error:
        raise VerificationError("KAT sacct output is not ASCII") from error
    roots = []
    for row in rows:
        if not row:
            continue
        columns = [column.strip() for column in row.split("|")]
        if len(columns) == 21 and columns[-1] == "" and columns[0] == kat_job_id:
            roots.append("|".join(columns))
    if len(roots) != 1:
        _fail("KAT sacct query did not return exactly one root allocation")
    raw = (roots[0] + "\n").encode("ascii")
    _parse_sacct(raw, kat_job_id)
    return raw


@dataclass(frozen=True)
class KatAnchorExpectations:
    kat_job_id: str
    kat_archive_sha256: str
    expected_git_head: str
    expected_contract_sha256: str
    expected_base_repo_bundle_sha256: str
    expected_base_git_head: str
    expected_repo_delta_bundle_sha256: str
    expected_prebuilt_archive_sha256: str
    expected_prebuilt_run_manifest_sha256: str
    expected_slurm_job_script_sha256: str
    schema_profile: str = "v6"

    def validate(self) -> None:
        if self.schema_profile not in {
            "v6",
            "historical-v5-fixture",
            "synthetic-self-test",
        }:
            _fail("unknown KAT schema profile")
        if not JOB_RE.fullmatch(self.kat_job_id):
            _fail("invalid expected KAT job id")
        if not COMMIT_RE.fullmatch(self.expected_git_head):
            _fail("invalid expected Git head")
        if not COMMIT_RE.fullmatch(self.expected_base_git_head):
            _fail("invalid expected base Git head")
        for value in (
            self.kat_archive_sha256,
            self.expected_contract_sha256,
            self.expected_base_repo_bundle_sha256,
            self.expected_repo_delta_bundle_sha256,
            self.expected_prebuilt_archive_sha256,
            self.expected_prebuilt_run_manifest_sha256,
            self.expected_slurm_job_script_sha256,
        ):
            if not SHA_RE.fullmatch(value):
                _fail("invalid expected SHA-256 anchor")


@dataclass(frozen=True)
class KatAnchorCertificate:
    fields: tuple[tuple[str, str], ...]

    def as_dict(self) -> dict[str, str]:
        return dict(self.fields)

    def as_bytes(self) -> bytes:
        return _canonical_kv(self.fields)

    @property
    def sha256(self) -> str:
        return _sha(self.as_bytes())


def parse_kat_anchor_certificate(raw: bytes) -> KatAnchorCertificate:
    parsed = _parse_kv(raw, CERTIFICATE_KEYS, "KAT prerequisite certificate")
    if parsed["SCHEMA"] != CERTIFICATE_SCHEMA:
        _fail("KAT prerequisite certificate schema mismatch")
    for key in (
        "KAT_ARCHIVE_SHA256",
        "KAT_ARCHIVE_SIDECAR_SHA256",
        "KAT_TRANSPORT_MANIFEST_SHA256",
        "KAT_TRANSPORT_JOB_RECORD_SHA256",
        "KAT_RESULT_RUN_MANIFEST_SHA256",
        "KAT_RESULT_FILES_INDEX_SHA256",
        "KAT_RESULT_RUN_CONTRACT_SHA256",
        "KAT_RESULT_SUMMARY_SHA256",
        "KAT_RESULT_JOB_RECORD_SHA256",
        "KAT_PREBUILT_RUN_MANIFEST_SHA256",
        "KAT_PREBUILT_FILES_INDEX_SHA256",
        "KAT_GIT_HEAD_FILE_SHA256",
        "KAT_GIT_STATUS_FILE_SHA256",
        "KAT_SACCT_SHA256",
        "KAT_FROZEN_CONTRACT_SHA256",
        "KAT_BASE_REPO_BUNDLE_SHA256",
        "KAT_REPO_DELTA_BUNDLE_SHA256",
        "KAT_PREBUILT_ARCHIVE_SHA256",
        "KAT_SLURM_JOB_SCRIPT_SHA256",
        "KAT_ANCHOR_SOURCE_SHA256",
        "KAT_COORDINATE_MANIFEST_SHA256",
        "KAT_EXPECTED_RESULTS_SHA256",
        "KAT_WAVE_CONTRACT_SHA256",
        "KAT_WAVE_RESULT_SHA256",
    ):
        if not SHA_RE.fullmatch(parsed[key]):
            _fail(f"invalid certificate SHA-256 field: {key}")
    if not JOB_RE.fullmatch(parsed["KAT_JOB_ID"]) or not JOB_RE.fullmatch(
        parsed["ADAPTIVE_JOB_ID"]
    ):
        _fail("invalid certificate job id")
    for key in (
        "KAT_PREREQUISITE_VALID",
        "KAT_LEAF_EVIDENCE_VALID",
        "KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT",
        "EXECUTION_PROVENANCE_ATTESTED",
        "PROMOTION_ELIGIBLE",
    ):
        if parsed[key] not in {"true", "false"}:
            _fail(f"invalid certificate boolean field: {key}")
    for key in (
        "KAT_ARCHIVE_MEMBER_COUNT",
        "KAT_ARCHIVE_REGULAR_FILE_COUNT",
        "KAT_ARCHIVE_DIRECTORY_COUNT",
        "KAT_RESULT_FILES_INDEX_ENTRY_COUNT",
        "KAT_PREBUILT_FILES_INDEX_ENTRY_COUNT",
        "KAT_UID",
        "KAT_RESTARTS",
        "KAT_ALLOC_NODES",
        "KAT_ALLOC_TASKS",
        "KAT_ALLOC_CPUS",
        "KAT_REQ_CPUS",
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
    ):
        if re.fullmatch(r"(?:0|[1-9][0-9]*)", parsed[key]) is None:
            _fail(f"invalid certificate count field: {key}")
    if parsed["KAT_PREREQUISITE_VALID"] == "true" and (
        parsed["CERTIFICATE_SCOPE"] != "AUTHORITATIVE_V6_ADAPTIVE_PREREQUISITE"
        or parsed["KAT_SCHEMA_PROFILE"] != "v6"
        or parsed["KAT_LEAF_EVIDENCE_VALID"] != "true"
        or parsed["KAT_HPG_VERIFIER_REPLAY_COUNT"] != "52"
        or parsed["KAT_HAPG_VERIFIER_REPLAY_COUNT"] != "52"
    ):
        _fail("authoritative KAT prerequisite lacks complete leaf evidence")
    return KatAnchorCertificate(tuple(parsed.items()))


@dataclass(frozen=True)
class _ArchiveView:
    raw: bytes
    files: Mapping[str, bytes]
    directories: frozenset[str]
    member_count: int


def _read_archive(path: Path) -> _ArchiveView:
    try:
        size = path.stat(follow_symlinks=False).st_size
    except OSError as error:
        raise VerificationError(f"cannot stat KAT archive: {error}") from error
    if size <= 0 or size > MAX_ARCHIVE_BYTES:
        _fail("KAT archive byte size exceeds the bounded verifier policy")
    raw = _stable_bytes(path, "KAT archive")
    files: dict[str, bytes] = {}
    directories: set[str] = set()
    names: set[str] = set()
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
            expanded_bytes = 0
            for member in archive:
                if len(names) >= MAX_ARCHIVE_MEMBERS:
                    _fail("KAT archive member count exceeds the bounded verifier policy")
                name = member.name.rstrip("/")
                pure = PurePosixPath(name)
                if (
                    pure.is_absolute()
                    or not pure.parts
                    or any(part in {"", ".", ".."} for part in pure.parts)
                    or name != pure.as_posix()
                    or name in names
                    or not (member.isdir() or member.isfile())
                ):
                    _fail(f"unsafe KAT archive member: {member.name}")
                names.add(name)
                if member.isdir():
                    directories.add(name)
                    continue
                if member.size < 0 or member.size > MAX_ARCHIVE_MEMBER_BYTES:
                    _fail("KAT archive member size is invalid")
                expanded_bytes += member.size
                if expanded_bytes > MAX_ARCHIVE_EXPANDED_BYTES:
                    _fail("KAT archive expanded size exceeds the bounded verifier policy")
                stream = archive.extractfile(member)
                if stream is None:
                    _fail(f"missing KAT archive payload: {name}")
                payload = stream.read()
                if len(payload) != member.size:
                    _fail(f"truncated KAT archive payload: {name}")
                files[name] = payload
            if not names:
                _fail("KAT archive is empty")
            if (
                len(raw) % 512 != 0
                or len(raw) - archive.offset < 1024
                or any(raw[archive.offset:])
            ):
                _fail("KAT archive has noncanonical or opaque trailing bytes")
    except (tarfile.TarError, OSError) as error:
        raise VerificationError(f"invalid KAT archive: {error}") from error
    if set(files) & directories:
        _fail("KAT archive file/directory collision")
    return _ArchiveView(raw, files, frozenset(directories), len(names))


def _parse_index(raw: bytes, label: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail(f"noncanonical {label}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError(f"non-ASCII {label}") from error
    indexed: dict[str, str] = {}
    for line in lines:
        if line.count("  ") != 1:
            _fail(f"malformed {label}")
        digest, token = line.split("  ", 1)
        pure = PurePosixPath(token)
        if (
            not SHA_RE.fullmatch(digest)
            or pure.is_absolute()
            or not pure.parts
            or ".." in pure.parts
            or token != pure.as_posix()
            or token in indexed
        ):
            _fail(f"unsafe or duplicate {label} row")
        indexed[token] = digest
    if list(indexed) != sorted(indexed):
        _fail(f"{label} is not sorted by POSIX relative token")
    return indexed


def _verify_indexed_tree(
    view: _ArchiveView,
    prefix: str,
    manifest_name: str,
) -> tuple[dict[str, str], dict[str, str]]:
    index_name = f"{prefix}/files.sha256"
    manifest_path = f"{prefix}/{manifest_name}"
    if index_name not in view.files or manifest_path not in view.files:
        _fail(f"missing indexed envelope under {prefix}")
    indexed = _parse_index(view.files[index_name], f"{prefix} file index")
    actual = {
        name[len(prefix) + 1 :]: _sha(raw)
        for name, raw in view.files.items()
        if name.startswith(prefix + "/")
        and name not in {index_name, manifest_path}
    }
    if actual != indexed:
        _fail(f"{prefix} file index differs from the exact regular-file set")
    return indexed, actual


def _validate_archive_directory_set(view: _ArchiveView) -> None:
    expected = {"result", "result/prebuilt-origin"}
    expected.update(f"result/{name}" for name in EMPTY_DIRS)
    expected.update(f"result/prebuilt-origin/{name}" for name in EMPTY_DIRS)
    if view.directories != expected:
        _fail("KAT archive directory set differs from the frozen envelope")


def _validate_v6_direct_file_sets(view: _ArchiveView) -> None:
    result_direct = {
        name.removeprefix("result/")
        for name in view.files
        if name.startswith("result/")
        and "/" not in name.removeprefix("result/")
    }
    prebuilt_prefix = "result/prebuilt-origin/"
    prebuilt_direct = {
        name.removeprefix(prebuilt_prefix)
        for name in view.files
        if name.startswith(prebuilt_prefix)
        and "/" not in name.removeprefix(prebuilt_prefix)
    }
    prebuilt_nested = {
        name
        for name in view.files
        if name.startswith(prebuilt_prefix)
        and "/" in name.removeprefix(prebuilt_prefix)
    }
    if result_direct != KAT_RESULT_DIRECT_FILES:
        _fail("KAT result direct-file set differs from the frozen v6 envelope")
    if prebuilt_direct != PREBUILT_DIRECT_FILES or prebuilt_nested:
        _fail("KAT prebuilt file set differs from the frozen v6 envelope")


def _file(view: _ArchiveView, name: str) -> bytes:
    try:
        return view.files[name]
    except KeyError as error:
        raise VerificationError(f"missing KAT archive file: {name}") from error


def _validate_evaluations(raw: bytes) -> dict[str, int]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail("noncanonical KAT evaluations TSV")
    try:
        reader = csv.DictReader(raw.decode("ascii").splitlines(), delimiter="\t")
    except UnicodeError as error:
        raise VerificationError("non-ASCII KAT evaluations TSV") from error
    if tuple(reader.fieldnames or ()) != EVALUATION_COLUMNS:
        _fail("KAT evaluations TSV header mismatch")
    rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        _fail("malformed KAT evaluations TSV row")
    identities = [row["NODE_ID"] for row in rows]
    if len(set(identities)) != len(identities):
        _fail("duplicate KAT evaluation identity")
    return {
        "EVALUATED": len(rows),
        "HPG_SIGNED": sum(row["HPG_STATUS"] == "HPG_VERIFIED_SIGNED_CHARTS" for row in rows),
        "HAPG_ATTEMPTED": sum(
            row["HAPG_STATUS"] != "H_APG_NOT_ELIGIBLE" for row in rows
        ),
        "HAPG_CERTIFIED": sum(
            row["APG_VALID"] == "true" and row["APG_PASS"] == "true" for row in rows
        ),
        "HAPG_RESCUE": sum(row["APG_RESCUE"] == "true" for row in rows),
    }


def _parse_int(token: str, label: str) -> int:
    if re.fullmatch(r"(?:0|[1-9][0-9]*)", token) is None:
        _fail(f"noncanonical integer: {label}")
    return int(token)


def _parse_bool(token: str, label: str) -> bool:
    if token == "true":
        return True
    if token == "false":
        return False
    _fail(f"noncanonical boolean: {label}")


def _canonical_leaf_id(u_depth: int, u_index: int, s_depth: int, s_index: int) -> str:
    return f"U{u_depth:02d}-{u_index:010d}_S{s_depth:02d}-{s_index:010d}"


def _leaf_input_bytes(coordinates: tuple[int, int, int, int]) -> bytes:
    u_depth, u_index, s_depth, s_index = coordinates
    return (
        "SCHEMA=sounio.cs6.c1-full-source-cover-leaf-input.v1\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def _frontier_bytes(population: Mapping[str, Mapping[str, object]]) -> bytes:
    rows = ["NODE_ID\tPARENT_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tINPUT_SHA256"]
    for identity in sorted(population):
        coordinates = population[identity]["coordinates"]
        assert isinstance(coordinates, tuple)
        rows.append(
            "\t".join(
                (
                    identity,
                    "-",
                    *(str(value) for value in coordinates),
                    _sha(_leaf_input_bytes(coordinates)),
                )
            )
        )
    return ("\n".join(rows) + "\n").encode("ascii")


def _table_rows(
    raw: bytes,
    columns: Sequence[str],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[dict[str, str]]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail(f"noncanonical {label}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError(f"non-ASCII {label}") from error
    if not lines or tuple(lines[0].split("\t")) != tuple(columns):
        _fail(f"{label} column schema mismatch")
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) != len(columns) or any(not field for field in fields):
            _fail(f"malformed {label} row")
        rows.append(dict(zip(columns, fields, strict=True)))
    if not allow_empty and not rows:
        _fail(f"empty {label}")
    return rows


def _header_table(
    raw: bytes,
    headers_spec: Sequence[tuple[str, str | None]],
    columns: Sequence[str],
    label: str,
) -> tuple[dict[str, str], dict[str, dict[str, str]], str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail(f"noncanonical {label}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError(f"non-ASCII {label}") from error
    if len(lines) < len(headers_spec) + 2:
        _fail(f"truncated {label}")
    headers: dict[str, str] = {}
    for line, (expected_key, expected_value) in zip(
        lines[: len(headers_spec)], headers_spec, strict=True
    ):
        if line.count("=") != 1:
            _fail(f"malformed {label} header")
        key, value = line.split("=", 1)
        if key != expected_key or not value or (
            expected_value is not None and value != expected_value
        ):
            _fail(f"{label} header mismatch: {expected_key}")
        headers[key] = value
    if tuple(lines[len(headers_spec)].split("\t")) != tuple(columns):
        _fail(f"{label} column schema mismatch")
    rows: dict[str, dict[str, str]] = {}
    for line in lines[len(headers_spec) + 1 :]:
        fields = line.split("\t")
        if len(fields) != len(columns) or any(not field for field in fields):
            _fail(f"malformed {label} row")
        row = dict(zip(columns, fields, strict=True))
        identity = row["NODE_ID"]
        if identity in rows:
            _fail(f"duplicate {label} node")
        rows[identity] = row
    if not rows or list(rows) != sorted(rows):
        _fail(f"{label} rows are empty or unsorted")
    if headers.get("NODE_COUNT") != str(len(rows)):
        _fail(f"{label} node count mismatch")
    return headers, rows, _sha(raw)


def _parse_kat_population(
    coordinate_raw: bytes, expected_raw: bytes
) -> dict[str, dict[str, object]]:
    if (
        not coordinate_raw.endswith(b"\n")
        or b"\r" in coordinate_raw
        or b"\0" in coordinate_raw
    ):
        _fail("noncanonical KAT coordinate manifest")
    try:
        coordinate_lines = coordinate_raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise VerificationError("non-ASCII KAT coordinate manifest") from error
    coordinate_columns = (
        "LEAF_ID",
        "U_DEPTH",
        "U_INDEX",
        "S_DEPTH",
        "S_INDEX",
        "PARENT_INPUT_SHA256",
        "PARENT_STATUS",
        "PARENT_RECEIPT_SHA256",
        "E1_R0_CHART",
        "E1_R0_SIGN",
        "E1_R1_CHART",
        "E1_R1_SIGN",
        "E2_R0_CHART",
        "E2_R0_SIGN",
        "E2_R1_CHART",
        "E2_R1_SIGN",
    )
    header = "\t".join(coordinate_columns)
    if coordinate_lines.count(header) != 1:
        _fail("KAT coordinate manifest header mismatch")
    coordinate_rows = coordinate_lines[coordinate_lines.index(header) + 1 :]
    if len(coordinate_rows) != 53:
        _fail("KAT coordinate population is not exactly 53 leaves")

    expected_columns_raw = expected_raw.splitlines()[0] if expected_raw else b""
    try:
        expected_columns = tuple(expected_columns_raw.decode("ascii").split("\t"))
    except UnicodeError as error:
        raise VerificationError("non-ASCII KAT expected results") from error
    expected_rows = _table_rows(
        expected_raw, expected_columns, "KAT expected results"
    )
    if (
        len(expected_rows) != 53
        or "LEAF_ID" not in expected_columns
        or "APG_PASS" not in expected_columns
        or "APG_RESCUE" not in expected_columns
    ):
        _fail("KAT expected-result population mismatch")
    expected_by_id: dict[str, dict[str, str]] = {}
    for row in expected_rows:
        identity = row["LEAF_ID"]
        if identity in expected_by_id:
            _fail("duplicate KAT expected-result identity")
        expected_by_id[identity] = row

    population: dict[str, dict[str, object]] = {}
    for line in coordinate_rows:
        fields = line.split("\t")
        if len(fields) != len(coordinate_columns) or any(not field for field in fields):
            _fail("malformed KAT coordinate row")
        row = dict(zip(coordinate_columns, fields, strict=True))
        coordinates = tuple(
            _parse_int(row[key], f"KAT coordinate {key}")
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        )
        u_depth, u_index, s_depth, s_index = coordinates
        identity = _canonical_leaf_id(*coordinates)
        if (
            identity != row["LEAF_ID"]
            or identity in population
            or u_depth > 30
            or s_depth > 30
            or u_index >= 1 << u_depth
            or s_index >= 1 << s_depth
            or row["PARENT_INPUT_SHA256"] != _sha(_leaf_input_bytes(coordinates))
        ):
            _fail("KAT coordinate identity or input binding mismatch")
        chart_signs = tuple(
            (row[f"E{event}_R{ray}_CHART"], row[f"E{event}_R{ray}_SIGN"])
            for event, ray in ((1, 0), (1, 1), (2, 0), (2, 1))
        )
        if identity == ROOT_ID:
            if chart_signs != (("NONE", "0"),) * 4:
                _fail("KAT root chart sentinel mismatch")
        elif any(
            chart not in {"X", "Y", "PLUS", "MINUS"} or sign not in {"-1", "1"}
            for chart, sign in chart_signs
        ):
            _fail("KAT coordinate chart/sign tuple is invalid")
        expected = expected_by_id.get(identity)
        if expected is None:
            _fail("KAT coordinate is absent from expected results")
        population[identity] = {
            "coordinates": coordinates,
            "chart_signs": chart_signs,
            "apg_pass": _parse_bool(expected["APG_PASS"], "expected APG_PASS"),
            "apg_rescue": _parse_bool(expected["APG_RESCUE"], "expected APG_RESCUE"),
        }
    if set(population) != set(expected_by_id) or list(population) != sorted(population):
        _fail("KAT coordinate and expected-result sets or order differ")
    return population


def _science_ids(view: _ArchiveView, directory: str) -> set[str]:
    prefix = f"result/{directory}/"
    identities: set[str] = set()
    for name in view.files:
        if not name.startswith(prefix):
            continue
        token = name[len(prefix) :]
        if "/" in token or not token.endswith(".txt"):
            _fail(f"unexpected nested KAT leaf artifact: {name}")
        identity = token.removesuffix(".txt")
        if identity in identities:
            _fail(f"duplicate KAT leaf artifact: {name}")
        identities.add(identity)
    return identities


def _run_exact_verifier(
    command: Sequence[str], expected_raw: bytes, timeout: int, label: str
) -> dict[str, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            timeout=timeout,
            env={**os.environ, "TZ": "UTC"},
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise VerificationError(f"{label} replay failed: {error}") from error
    if completed.returncode != 0 or completed.stderr:
        _fail(f"{label} replay failed with rc={completed.returncode}")
    if completed.stdout != expected_raw:
        _fail(f"{label} replay differs byte-for-byte from stored verification")
    keys = HPG_VERIFICATION_KEYS if label == "H-PG verifier" else HAPG_VERIFICATION_KEYS
    return _parse_kv(completed.stdout, keys, label)


def _materialize_science(view: _ArchiveView, destination: Path) -> None:
    science_directories = set(EMPTY_DIRS)
    verifier_sources = {
        "cs6_plucker_cocycle_verify.py",
        "cs6_hapg_full_source_cover_verify.py",
        "cs6_affine_projective_cocycle_full53_probe.cpp",
        "cs6_affine_projective_cocycle_full53_verify.py",
    }
    for name, raw in view.files.items():
        if not name.startswith("result/"):
            continue
        token = name.removeprefix("result/")
        if token.split("/", 1)[0] not in science_directories and token not in verifier_sources:
            continue
        target = destination.joinpath(*PurePosixPath(token).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
        target.chmod(0o444)


def _validate_leaf_evidence(
    view: _ArchiveView,
    run_contract: Mapping[str, str],
    frozen: Mapping[str, str],
    summary: Mapping[str, str],
    transport: Mapping[str, str],
) -> dict[str, str]:
    coordinate_raw = _file(view, "result/kat-coordinates.tsv")
    expected_raw = _file(view, "result/kat-expected-results.tsv")
    for run_key, frozen_key, raw in (
        ("KAT_COORDINATE_MANIFEST_SHA256", "KAT_COORDINATE_MANIFEST_SHA256", coordinate_raw),
        ("KAT_EXPECTED_RESULTS_SHA256", "KAT_EXPECTED_RESULTS_SHA256", expected_raw),
    ):
        if run_contract[run_key] != _sha(raw) or frozen.get(frozen_key) != _sha(raw):
            _fail(f"KAT population binding mismatch: {run_key}")
    population = _parse_kat_population(coordinate_raw, expected_raw)
    identities = set(population)
    nonroot = identities - {ROOT_ID}
    if ROOT_ID not in identities or len(nonroot) != 52:
        _fail("KAT population root/nonroot cardinality mismatch")

    exact_topology = {
        "inputs": identities,
        "hpg-receipts": identities,
        "hpg-stderr": identities,
        "hpg-verifications": nonroot,
        "hapg-receipts": nonroot,
        "hapg-stderr": nonroot,
        "hapg-verifications": nonroot,
    }
    for directory, expected_ids in exact_topology.items():
        if _science_ids(view, directory) != expected_ids:
            _fail(f"KAT leaf artifact topology mismatch: {directory}")
    if {
        name.removeprefix("result/wave-contracts/")
        for name in view.files
        if name.startswith("result/wave-contracts/")
    } != {"W0000.tsv"} or {
        name.removeprefix("result/wave-results/")
        for name in view.files
        if name.startswith("result/wave-results/")
    } != {"W0000.tsv"}:
        _fail("KAT wave artifact topology mismatch")

    wave_raw = _file(view, "result/wave-contracts/W0000.tsv")
    result_raw = _file(view, "result/wave-results/W0000.tsv")
    wave_headers, wave_rows, wave_sha = _header_table(
        wave_raw, WAVE_HEADERS, WAVE_COLUMNS, "KAT wave contract"
    )
    result_headers, result_rows, result_sha = _header_table(
        result_raw, RESULT_HEADERS, RESULT_COLUMNS, "KAT wave result"
    )
    frontier_sha = _sha(_frontier_bytes(population))
    empty_frontier_sha = _sha(_frontier_bytes({}))
    run_contract_sha = _sha(_file(view, "result/run-contract.txt"))
    source_fields = (
        "HPG_WORKER_SOURCE_SHA256",
        "HPG_VERIFIER_SOURCE_SHA256",
        "HAPG_WORKER_SOURCE_SHA256",
        "HAPG_KERNEL_SOURCE_SHA256",
        "HAPG_VERIFIER_ADAPTER_SHA256",
        "HAPG_NUMERIC_VERIFIER_SHA256",
    )
    if (
        wave_headers["RUN_CONTRACT_SHA256"] != run_contract_sha
        or wave_headers["ROOT_CHALLENGE"] != run_contract["ROOT_CHALLENGE"]
        or wave_headers["WAVE_INDEX"] != "0"
        or wave_headers["PREVIOUS_WAVE_RESULT_SHA256"] != ZERO_SHA256
        or wave_headers["FRONTIER_SHA256"] != frontier_sha
        or any(wave_headers[key] != run_contract[key] for key in source_fields)
        or set(wave_rows) != identities
        or result_headers["WAVE_INDEX"] != "0"
        or result_headers["WAVE_CONTRACT_SHA256"] != wave_sha
        or result_headers["NEXT_FRONTIER_SHA256"] != empty_frontier_sha
        or set(result_rows) != identities
    ):
        _fail("KAT wave causal, source, or population binding mismatch")

    for identity in sorted(identities):
        population_row = population[identity]
        coordinates = population_row["coordinates"]
        chart_signs = population_row["chart_signs"]
        assert isinstance(coordinates, tuple) and isinstance(chart_signs, tuple)
        wave_row = wave_rows[identity]
        result_row = result_rows[identity]
        if tuple(
            wave_row[key]
            for key in ("PARENT_ID", "U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX", "INPUT_SHA256")
        ) != ("-", *(str(value) for value in coordinates), _sha(_leaf_input_bytes(coordinates))):
            _fail(f"KAT wave row differs from frozen coordinate: {identity}")
        actual_charts = tuple(
            (wave_row[f"E{event}_R{ray}_CHART"], wave_row[f"E{event}_R{ray}_SIGN"])
            for event, ray in ((1, 0), (1, 1), (2, 0), (2, 1))
        )
        if actual_charts != chart_signs:
            _fail(f"KAT wave chart tuple differs from coordinate manifest: {identity}")
        if (
            result_row["DECISION"] != "KAT_ONLY"
            or result_row["TERMINAL_REASON"] != "-"
            or result_row["HPG_STATUS"] != wave_row["HPG_STATUS"]
        ):
            _fail(f"KAT wave-result policy mismatch: {identity}")

    evaluation_rows = _table_rows(
        _file(view, "result/evaluations.tsv"), EVALUATION_COLUMNS, "KAT evaluations"
    )
    expected_evaluations = [
        {
            "WAVE_INDEX": "0",
            "NODE_ID": identity,
            "PARENT_ID": "-",
            "U_DEPTH": str(population[identity]["coordinates"][0]),
            "U_INDEX": str(population[identity]["coordinates"][1]),
            "S_DEPTH": str(population[identity]["coordinates"][2]),
            "S_INDEX": str(population[identity]["coordinates"][3]),
            "WAVE_CONTRACT_SHA256": wave_sha,
            "HPG_STATUS": wave_rows[identity]["HPG_STATUS"],
            "HPG_RECEIPT_SHA256": wave_rows[identity]["HPG_RECEIPT_SHA256"],
            "HPG_VERIFICATION_SHA256": wave_rows[identity]["HPG_VERIFICATION_SHA256"],
            "HAPG_STATUS": result_rows[identity]["HAPG_STATUS"],
            "HAPG_RECEIPT_SHA256": result_rows[identity]["HAPG_RECEIPT_SHA256"],
            "HAPG_VERIFICATION_SHA256": result_rows[identity]["HAPG_VERIFICATION_SHA256"],
            "APG_VALID": result_rows[identity]["APG_VALID"],
            "APG_PASS": result_rows[identity]["APG_PASS"],
            "APG_RESCUE": result_rows[identity]["APG_RESCUE"],
            "GENERIC_CERTIFICATE_PASS": result_rows[identity]["GENERIC_CERTIFICATE_PASS"],
            "DECISION": "KAT_ONLY",
            "TERMINAL_REASON": "-",
        }
        for identity in sorted(identities)
    ]
    if evaluation_rows != expected_evaluations:
        _fail("KAT evaluations are not the exact wave/result projection")

    negative_rows = _table_rows(
        _file(view, "result/negative-outcomes.tsv"),
        NEGATIVE_COLUMNS,
        "KAT negative outcomes",
    )
    expected_negatives = [
        {
            "WAVE_INDEX": "0",
            "NODE_ID": identity,
            "HPG_STATUS": wave_rows[identity]["HPG_STATUS"],
            "HAPG_STATUS": result_rows[identity]["HAPG_STATUS"],
            "DECISION": "KAT_ONLY",
            "TERMINAL_REASON": "-",
        }
        for identity in sorted(identities)
        if result_rows[identity]["APG_PASS"] == "false"
    ]
    if negative_rows != expected_negatives or len(negative_rows) != 5:
        _fail("KAT negative-outcome ledger is not the exact projection")
    wave_ledger = _table_rows(
        _file(view, "result/waves.tsv"), WAVES_COLUMNS, "KAT wave ledger"
    )
    if wave_ledger != [
        {
            "WAVE_INDEX": "0",
            "FRONTIER_SHA256": frontier_sha,
            "WAVE_CONTRACT_SHA256": wave_sha,
            "WAVE_RESULT_SHA256": result_sha,
            "NEXT_FRONTIER_SHA256": empty_frontier_sha,
        }
    ]:
        _fail("KAT wave ledger mismatch")
    timing_rows = _table_rows(
        _file(view, "result/timings.tsv"), TIMING_COLUMNS, "KAT timings"
    )
    if [row["NODE_ID"] for row in timing_rows] != sorted(identities) or any(
        row["WAVE_INDEX"] != "0"
        or _parse_int(row["HPG_ELAPSED_MS"], "HPG elapsed") < 0
        or _parse_int(row["HAPG_ELAPSED_MS"], "HAPG elapsed") < 0
        for row in timing_rows
    ):
        _fail("KAT timing ledger population mismatch")

    timeout = _parse_int(run_contract["TIMEOUT_SECONDS"], "KAT timeout")
    jobs = _parse_int(run_contract["JOBS"], "KAT jobs")
    replay_counts = {
        "hpg": 0,
        "hapg": 0,
        "hpg_mutations": 0,
        "hpg_rejected": 0,
        "hapg_mutations": 0,
        "hapg_rejected": 0,
    }
    with tempfile.TemporaryDirectory(prefix="cs6-hapg-kat-leaf-replay.") as directory:
        root = Path(directory)
        _materialize_science(view, root)
        wave_path = root / "wave-contracts/W0000.tsv"

        def replay(identity: str) -> dict[str, int]:
            wave_row = wave_rows[identity]
            result_row = result_rows[identity]
            input_path = root / f"inputs/{identity}.txt"
            hpg_receipt = root / f"hpg-receipts/{identity}.txt"
            hpg_stderr = root / f"hpg-stderr/{identity}.txt"
            if (
                _sha(input_path.read_bytes()) != wave_row["INPUT_SHA256"]
                or _sha(hpg_receipt.read_bytes()) != wave_row["HPG_RECEIPT_SHA256"]
                or _sha(hpg_stderr.read_bytes()) != wave_row["HPG_STDERR_SHA256"]
            ):
                _fail(f"KAT H-PG artifact digest mismatch: {identity}")
            if identity == ROOT_ID:
                lowered = hpg_stderr.read_bytes().lower()
                if (
                    hpg_receipt.read_bytes() != b""
                    or b"interval error:" not in lowered
                    or not (b"division by 0" in lowered or b"division by zero" in lowered)
                    or wave_row["HPG_STATUS"] != "H_PG_INTERVAL_DOMAIN"
                    or wave_row["HPG_RC"] == "0"
                    or wave_row["HPG_RECEIPT_SHA256"] != EMPTY_SHA256
                    or wave_row["HPG_VERIFICATION_SHA256"] != ZERO_SHA256
                    or wave_row["HPG_PHYSICAL_SHA256"] != ZERO_SHA256
                    or wave_row["HPG_PROBE_PASS"] != "false"
                    or wave_row["HPG_CERTIFICATE_PASS"] != "false"
                    or wave_row["HAPG_ELIGIBLE"] != "false"
                    or result_row["HAPG_ATTEMPTED"] != "false"
                    or result_row["HAPG_STATUS"] != "H_APG_NOT_ELIGIBLE"
                    or result_row["HAPG_RC"] != "0"
                    or result_row["HAPG_CHALLENGE"] != ZERO_SHA256
                    or result_row["HAPG_RECEIPT_SHA256"] != EMPTY_SHA256
                    or result_row["HAPG_STDERR_SHA256"] != EMPTY_SHA256
                    or result_row["HAPG_VERIFICATION_SHA256"] != ZERO_SHA256
                    or result_row["HAPG_PHYSICAL_SHA256"] != ZERO_SHA256
                    or any(
                        result_row[key] != "false"
                        for key in (
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
                        )
                    )
                ):
                    _fail("KAT root failure evidence mismatch")
                return {key: 0 for key in replay_counts}

            if (
                wave_row["HPG_STATUS"] != "HPG_VERIFIED_SIGNED_CHARTS"
                or wave_row["HPG_RC"] != "0"
                or wave_row["HAPG_ELIGIBLE"] != "true"
                or hpg_stderr.read_bytes() != b""
                or result_row["HAPG_ATTEMPTED"] != "true"
                or result_row["HAPG_RC"] != "0"
            ):
                _fail(f"KAT nonroot execution status mismatch: {identity}")
            hpg_verification = root / f"hpg-verifications/{identity}.txt"
            hpg_raw = hpg_verification.read_bytes()
            if _sha(hpg_raw) != wave_row["HPG_VERIFICATION_SHA256"]:
                _fail(f"KAT H-PG verification digest mismatch: {identity}")
            hpg_values = _run_exact_verifier(
                (
                    sys.executable,
                    "-B",
                    str(root / "cs6_plucker_cocycle_verify.py"),
                    str(hpg_receipt),
                    "--source-sha",
                    run_contract["HPG_WORKER_SOURCE_SHA256"],
                    "--input",
                    str(input_path),
                    "--challenge",
                    wave_row["HPG_CHALLENGE"],
                    "--self-test-mutations",
                ),
                hpg_raw,
                timeout,
                "H-PG verifier",
            )
            hapg_receipt = root / f"hapg-receipts/{identity}.txt"
            hapg_stderr = root / f"hapg-stderr/{identity}.txt"
            hapg_verification = root / f"hapg-verifications/{identity}.txt"
            hapg_raw = hapg_verification.read_bytes()
            if (
                _sha(hapg_receipt.read_bytes()) != result_row["HAPG_RECEIPT_SHA256"]
                or _sha(hapg_stderr.read_bytes()) != result_row["HAPG_STDERR_SHA256"]
                or hapg_stderr.read_bytes() != b""
                or _sha(hapg_raw) != result_row["HAPG_VERIFICATION_SHA256"]
            ):
                _fail(f"KAT H-APG artifact digest mismatch: {identity}")
            hapg_values = _run_exact_verifier(
                (
                    sys.executable,
                    "-B",
                    str(root / "cs6_hapg_full_source_cover_verify.py"),
                    str(hapg_receipt),
                    "--hapg-source-sha",
                    run_contract["HAPG_WORKER_SOURCE_SHA256"],
                    "--hpg-source-sha",
                    run_contract["HPG_WORKER_SOURCE_SHA256"],
                    "--input",
                    str(input_path),
                    "--wave-contract",
                    str(wave_path),
                    "--hpg-receipt",
                    str(hpg_receipt),
                    "--hpg-verification",
                    str(hpg_verification),
                    "--root-challenge",
                    run_contract["ROOT_CHALLENGE"],
                    "--self-test-mutations",
                ),
                hapg_raw,
                timeout,
                "H-APG adapter",
            )
            expected = population[identity]
            result_projection = {
                "HAPG_CHALLENGE": hapg_values["LEAF_CHALLENGE"],
                "HAPG_PHYSICAL_SHA256": hapg_values["PHYSICAL_SHA256"],
                "HAPG_PROBE_PASS": hapg_values["PROBE_PASS"],
                "AFFINE_PASS": hapg_values["AFFINE_CERTIFICATE_PASS"],
                "PROJECTIVE_X_PASS": hapg_values["PROJECTIVE_X_CERTIFICATE_PASS"],
                "PROJECTIVE_Y_PASS": hapg_values["PROJECTIVE_Y_CERTIFICATE_PASS"],
                "PROJECTIVE_PLUS_PASS": hapg_values["PROJECTIVE_PLUS_CERTIFICATE_PASS"],
                "PROJECTIVE_MINUS_PASS": hapg_values["PROJECTIVE_MINUS_CERTIFICATE_PASS"],
                "HOMOGENEOUS_PASS": hapg_values["HOMOGENEOUS_CERTIFICATE_PASS"],
                "APG_VALID": hapg_values["APG_COMPUTATION_VALID"],
                "APG_PASS": hapg_values["APG_CERTIFICATE_PASS"],
                "APG_RESCUE": hapg_values["APG_RESCUE"],
                "GENERIC_CERTIFICATE_PASS": hapg_values["GENERIC_CERTIFICATE_PASS"],
            }
            hapg_terminal = _parse_bool(
                hapg_values["HAPG_TERMINAL_CERTIFIED"], "replayed HAPG terminal"
            )
            hapg_valid = _parse_bool(
                hapg_values["APG_COMPUTATION_VALID"], "replayed APG valid"
            )
            expected_status = (
                "H_APG_CERTIFIED"
                if hapg_terminal
                else "H_APG_UNCERTIFIED"
                if hapg_valid
                else "H_APG_INVALID"
            )
            if (
                hpg_values["RECEIPT_SHA256"] != wave_row["HPG_RECEIPT_SHA256"]
                or hpg_values["PHYSICAL_SHA256"] != wave_row["HPG_PHYSICAL_SHA256"]
                or hpg_values["PROBE_PASS"] != wave_row["HPG_PROBE_PASS"]
                or hpg_values["CERTIFICATE_PASS"] != wave_row["HPG_CERTIFICATE_PASS"]
                or hapg_values["RECEIPT_SHA256"] != result_row["HAPG_RECEIPT_SHA256"]
                or hapg_values["WAVE_CONTRACT_SHA256"] != wave_sha
                or hapg_values["HPG_RECEIPT_SHA256"] != wave_row["HPG_RECEIPT_SHA256"]
                or hapg_values["HPG_VERIFICATION_SHA256"]
                != wave_row["HPG_VERIFICATION_SHA256"]
                or hapg_terminal
                != (
                    hapg_valid
                    and _parse_bool(
                        hapg_values["APG_CERTIFICATE_PASS"], "replayed APG pass"
                    )
                )
                or result_row["HAPG_STATUS"] != expected_status
                or any(result_row[key] != value for key, value in result_projection.items())
                or _parse_bool(hapg_values["APG_CERTIFICATE_PASS"], "replayed APG_PASS")
                != expected["apg_pass"]
                or _parse_bool(hapg_values["APG_RESCUE"], "replayed APG_RESCUE")
                != expected["apg_rescue"]
            ):
                _fail(f"KAT replay differs from frozen leaf outcome: {identity}")
            return {
                "hpg": 1,
                "hapg": 1,
                "hpg_mutations": _parse_int(hpg_values["MUTATION_TESTS"], "HPG mutations"),
                "hpg_rejected": _parse_int(hpg_values["MUTATIONS_REJECTED"], "HPG rejected"),
                "hapg_mutations": _parse_int(hapg_values["MUTATION_TESTS"], "HAPG mutations"),
                "hapg_rejected": _parse_int(hapg_values["MUTATIONS_REJECTED"], "HAPG rejected"),
            }

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(jobs, 32)) as executor:
            for counts in executor.map(replay, sorted(identities)):
                for key, value in counts.items():
                    replay_counts[key] += value

    expected_replays = {
        "hpg": 52,
        "hapg": 52,
        "hpg_mutations": _parse_int(
            frozen.get("KAT_EXPECTED_HPG_MUTATION_TESTS", ""), "frozen HPG mutations"
        ),
        "hpg_rejected": _parse_int(
            frozen.get("KAT_EXPECTED_HPG_MUTATIONS_REJECTED", ""),
            "frozen HPG rejected",
        ),
        "hapg_mutations": _parse_int(
            frozen.get("KAT_EXPECTED_HAPG_MUTATION_TESTS", ""), "frozen HAPG mutations"
        ),
        "hapg_rejected": _parse_int(
            frozen.get("KAT_EXPECTED_HAPG_MUTATIONS_REJECTED", ""),
            "frozen HAPG rejected",
        ),
    }
    if replay_counts != expected_replays or any(
        summary[key] != str(replay_counts[count_key])
        for key, count_key in (
            ("HPG_MUTATION_TESTS", "hpg_mutations"),
            ("HPG_MUTATIONS_REJECTED", "hpg_rejected"),
            ("HAPG_MUTATION_TESTS", "hapg_mutations"),
            ("HAPG_MUTATIONS_REJECTED", "hapg_rejected"),
        )
    ):
        _fail("KAT verifier replay or mutation totals differ from frozen evidence")
    return {
        "coordinate_sha": _sha(coordinate_raw),
        "expected_sha": _sha(expected_raw),
        "wave_sha": wave_sha,
        "result_sha": result_sha,
        "hpg_replays": str(replay_counts["hpg"]),
        "hapg_replays": str(replay_counts["hapg"]),
    }


def _control_fields(raw: bytes, label: str) -> dict[str, str]:
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        _fail(f"noncanonical {label}")
    try:
        text = raw.decode("ascii").strip()
    except UnicodeError as error:
        raise VerificationError(f"non-ASCII {label}") from error
    result: dict[str, str] = {}
    for token in shlex.split(text):
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        if key in result:
            _fail(f"duplicate {label} field")
        result[key] = value
    return result


def _require_exact(mapping: Mapping[str, str], expected: Mapping[str, str], label: str) -> None:
    for key, value in expected.items():
        if mapping.get(key) != value:
            _fail(f"{label} mismatch for {key}")


def certify_kat_anchor(
    *,
    archive_path: str | Path,
    sacct_bytes: bytes,
    adaptive_job_id: str,
    adaptive_submit_utc: str,
    expectations: KatAnchorExpectations,
    sidecar_path: str | Path | None = None,
) -> KatAnchorCertificate:
    expectations.validate()
    if not JOB_RE.fullmatch(adaptive_job_id) or adaptive_job_id == expectations.kat_job_id:
        _fail("adaptive job id must be numeric and distinct from the KAT job")
    adaptive_submit = _parse_timestamp(adaptive_submit_utc, "adaptive submit")
    archive = Path(archive_path)
    sidecar = Path(sidecar_path) if sidecar_path is not None else Path(f"{archive}.sha256")
    view = _read_archive(archive)
    archive_sha = _sha(view.raw)
    if archive_sha != expectations.kat_archive_sha256:
        _fail("KAT archive digest differs from the expected anchor")
    sidecar_raw = _stable_bytes(sidecar, "KAT archive sidecar")
    expected_sidecar = f"{archive_sha}  {archive.name}\n".encode("ascii")
    if sidecar_raw != expected_sidecar:
        _fail("KAT archive sidecar differs from exact canonical bytes")

    result_index, _ = _verify_indexed_tree(view, "result", "run-manifest.txt")
    historical = expectations.schema_profile == "historical-v5-fixture"
    synthetic = expectations.schema_profile == "synthetic-self-test"
    expected_files = {
        f"result/{token}" for token in result_index
    } | {
        "result/files.sha256",
        "result/run-manifest.txt",
        "transport-manifest.txt",
        "transport-slurm-job-record.txt",
    }
    if not historical:
        expected_files.add("transport-config.txt")
    if set(view.files) != expected_files:
        _fail("KAT archive regular-file set differs from the indexed envelope")
    _validate_archive_directory_set(view)

    run_manifest_raw = _file(view, "result/run-manifest.txt")
    run_manifest = _parse_kv(run_manifest_raw, RUN_MANIFEST_KEYS, "KAT run manifest")
    run_contract_raw = _file(view, "result/run-contract.txt")
    run_contract_keys = RUN_CONTRACT_BASE_KEYS
    if not historical:
        run_contract_keys = (
            RUN_CONTRACT_BASE_KEYS[:14]
            + ("KAT_ANCHOR_SHA256",)
            + RUN_CONTRACT_BASE_KEYS[14:]
        )
    run_contract = _parse_kv(run_contract_raw, run_contract_keys, "KAT run contract")
    summary_raw = _file(view, "result/summary.txt")
    summary = _parse_kv(summary_raw, SUMMARY_KEYS, "KAT summary")
    transport_raw = _file(view, "transport-manifest.txt")
    transport_keys = TRANSPORT_V2_KEYS if historical else TRANSPORT_V3_KEYS
    transport = _parse_kv(transport_raw, transport_keys, "KAT transport manifest")
    transport_job_raw = _file(view, "transport-slurm-job-record.txt")
    result_job_raw = _file(view, "result/slurm-job-record.txt")
    transport_job = _control_fields(transport_job_raw, "transport Slurm job record")
    result_job = _control_fields(result_job_raw, "result Slurm job record")

    contract_candidates = sorted(
        name
        for name in view.files
        if re.fullmatch(
            r"result/cs6_hapg_full_source_cover_contract_v[0-9]+\.txt", name
        )
    )
    expected_contract_name = (
        "result/cs6_hapg_full_source_cover_contract_v5.txt"
        if historical
        else "result/cs6_hapg_full_source_cover_contract_v6.txt"
    )
    if expected_contract_name not in contract_candidates:
        _fail("expected frozen KAT contract is absent from the archive")
    frozen_raw = _file(view, expected_contract_name)
    frozen = _parse_kv(frozen_raw, None, "frozen KAT contract")
    frozen_sha = _sha(frozen_raw)
    anchor_source_sha = ZERO_SHA256
    if not historical:
        anchor_source_sha = _sha(
            _file(view, "result/cs6_hapg_full_source_cover_kat_anchor.py")
        )

    prebuilt_prefix = "result/prebuilt-origin"
    prebuilt_index, _ = _verify_indexed_tree(view, prebuilt_prefix, "run-manifest.txt")
    prebuilt_manifest_raw = _file(view, f"{prebuilt_prefix}/run-manifest.txt")
    prebuilt_keys = PREBUILT_MANIFEST_BASE_KEYS
    if not historical:
        prebuilt_keys = (
            PREBUILT_MANIFEST_BASE_KEYS[:15]
            + ("KAT_ANCHOR_SHA256",)
            + PREBUILT_MANIFEST_BASE_KEYS[15:]
        )
    prebuilt = _parse_kv(
        prebuilt_manifest_raw, prebuilt_keys, "KAT prebuilt manifest"
    )

    _require_exact(
        run_manifest,
        {
            "SCHEMA": "sounio.cs6.hapg-full-source-cover-run-manifest.v1",
            "RUN_COMPLETE": "true",
            "MODE": "kat",
            "ROOT_CHALLENGE": frozen.get("KAT_ROOT_CHALLENGE", ""),
            "CAPD_VERSION": "5.3.0",
            "INTERVAL_BACKEND": "FILIB",
            "OPTIMIZATION_LEVEL": "O0",
            "RUN_CONTRACT_SHA256": _sha(run_contract_raw),
            "FILES_INDEX_SHA256": _sha(_file(view, "result/files.sha256")),
            "FILE_COUNT": str(len(result_index)),
            "WAVE_COUNT": "1",
            "LOCAL_PROCESS_ORDERED_HASH_CHAIN": "true",
            "EXECUTION_PROVENANCE_ATTESTED": "false",
            "PROMOTION_ELIGIBLE": "false",
        },
        "KAT run manifest",
    )
    expected_run_schema = (
        "sounio.cs6.hapg-full-source-cover-run-contract.v1"
        if historical
        else "sounio.cs6.hapg-full-source-cover-run-contract.v2"
    )
    _require_exact(
        run_contract,
        {
            "SCHEMA": expected_run_schema,
            "FROZEN_CONTRACT_SHA256": expectations.expected_contract_sha256,
            "MODE": "kat",
            "SOURCE": "N0",
            "ROOT_CHALLENGE": frozen.get("KAT_ROOT_CHALLENGE", ""),
            "TRAVERSAL": "DETERMINISTIC_BREADTH_FIRST",
            "SPLIT_RULE": "S_IF_S_DEPTH_LE_U_DEPTH_ELSE_U",
            "TERMINAL_PREDICATE": "APG_COMPUTATION_VALID_AND_APG_CERTIFICATE_PASS",
            "SLURM_JOB_SCRIPT_SHA256": expectations.expected_slurm_job_script_sha256,
            "BUILD_MODE": "VERIFIED_PREBUILT_BUNDLE",
            "PREBUILT_RUN_MANIFEST_SHA256": expectations.expected_prebuilt_run_manifest_sha256,
            "SLURM_JOB_ID": expectations.kat_job_id,
            "SLURM_JOB_VERIFIED": "true",
            "WORKING_FILESYSTEM_POLICY": "NODE_LOCAL_TMP_THEN_HASHED_ARCHIVE_TRANSPORT",
            "MUTATION_AUDIT": "true",
            "LOCAL_PROCESS_ORDERED_HASH_CHAIN": "true",
            "EXECUTION_PROVENANCE_ATTESTED": "false",
            "PROMOTION_ELIGIBLE": "false",
        },
        "KAT run contract",
    )
    if not historical and run_contract["KAT_ANCHOR_SHA256"] != anchor_source_sha:
        _fail("KAT run contract anchor-source binding mismatch")
    if run_contract["SLURM_JOB_RECORD_SHA256"] != _sha(result_job_raw):
        _fail("KAT result Slurm job record binding mismatch")

    expected_transport_schema = (
        "sounio.cs6.hapg-full-source-cover-transport.v2"
        if historical
        else "sounio.cs6.hapg-full-source-cover-transport.v3"
    )
    transport_expected = {
        "SCHEMA": expected_transport_schema,
        "MODE": "kat",
        "SLURM_JOB_ID": expectations.kat_job_id,
        "EXPECTED_GIT_HEAD": expectations.expected_git_head,
        "EXPECTED_CONTRACT_SHA256": expectations.expected_contract_sha256,
        "SLURM_JOB_SCRIPT_SHA256": expectations.expected_slurm_job_script_sha256,
        "BASE_REPO_BUNDLE_SHA256": expectations.expected_base_repo_bundle_sha256,
        "BASE_GIT_HEAD": expectations.expected_base_git_head,
        "REPO_DELTA_BUNDLE_SHA256": expectations.expected_repo_delta_bundle_sha256,
        "PREBUILT_ARCHIVE_SHA256": expectations.expected_prebuilt_archive_sha256,
        "RESULT_RUN_MANIFEST_SHA256": _sha(run_manifest_raw),
        "RESULT_FILES_INDEX_SHA256": _sha(_file(view, "result/files.sha256")),
        "AGGREGATION_SHA256": ZERO_SHA256,
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    if not historical:
        transport_expected.update(
            {
                "KAT_PREREQUISITE_REQUIRED": "false",
                "KAT_JOB_ID": "0",
                "KAT_ARCHIVE_SHA256": ZERO_SHA256,
                "KAT_CERTIFICATE_SHA256": ZERO_SHA256,
                "POST_RUN_GATE_PASS": "true",
                "FAILURE_STAGE": "NONE",
                "FAILURE_RC": "0",
            }
        )
    _require_exact(transport, transport_expected, "KAT transport manifest")

    if (
        frozen_sha != expectations.expected_contract_sha256
        or frozen.get("BASE_REPO_BUNDLE_SHA256")
        != expectations.expected_base_repo_bundle_sha256
        or frozen.get("BASE_REPO_BUNDLE_GIT_HEAD") != expectations.expected_base_git_head
        or frozen.get("SLURM_JOB_SCRIPT_SHA256")
        != expectations.expected_slurm_job_script_sha256
        or _file(view, "result/git-head.txt")
        != f"{expectations.expected_git_head}\n".encode("ascii")
        or _file(view, "result/git-status.txt") != b""
    ):
        _fail("KAT frozen source or Git anchor mismatch")
    if not historical and frozen.get("KAT_ANCHOR_SHA256") != anchor_source_sha:
        _fail("frozen KAT anchor source mismatch")

    expected_prebuilt_schema = (
        "sounio.cs6.hapg-full-source-cover-prebuilt.v1"
        if historical
        else "sounio.cs6.hapg-full-source-cover-prebuilt.v2"
    )
    _require_exact(
        prebuilt,
        {
            "SCHEMA": expected_prebuilt_schema,
            "RUN_COMPLETE": "true",
            "MODE": "prepare",
            "CAPD_VERSION": "5.3.0",
            "INTERVAL_BACKEND": "FILIB",
            "OPTIMIZATION_LEVEL": "O0",
            "FROZEN_CONTRACT_SHA256": expectations.expected_contract_sha256,
            "FILES_INDEX_SHA256": _sha(
                _file(view, f"{prebuilt_prefix}/files.sha256")
            ),
            "FILE_COUNT": str(len(prebuilt_index)),
            "PROMOTION_ELIGIBLE": "false",
        },
        "KAT prebuilt manifest",
    )
    if _sha(prebuilt_manifest_raw) != expectations.expected_prebuilt_run_manifest_sha256:
        _fail("KAT prebuilt run-manifest digest mismatch")
    if expectations.schema_profile == "v6":
        _validate_v6_direct_file_sets(view)
    prebuilt_declarations = {
        "FROZEN_CONTRACT_SHA256": expected_contract_name.removeprefix("result/"),
        "HPG_WORKER_SOURCE_SHA256": "cs6_plucker_cocycle_probe.cpp",
        "HPG_VERIFIER_SOURCE_SHA256": "cs6_plucker_cocycle_verify.py",
        "HAPG_WORKER_SOURCE_SHA256": "cs6_hapg_full_source_cover_worker.cpp",
        "HAPG_KERNEL_SOURCE_SHA256": "cs6_affine_projective_cocycle_full53_probe.cpp",
        "HAPG_VERIFIER_ADAPTER_SHA256": "cs6_hapg_full_source_cover_verify.py",
        "HAPG_NUMERIC_VERIFIER_SHA256": "cs6_affine_projective_cocycle_full53_verify.py",
        "RUNNER_SHA256": "cs6_hapg_full_source_cover_run.py",
        "AGGREGATOR_SHA256": "cs6_hapg_full_source_cover_aggregate.py",
        "EXACT_TREE_KERNEL_SHA256": "cs6_c1_full_source_cover_aggregate.py",
        "GATE_SHA256": "cs6_hapg_full_source_cover_gate.sh",
        "SLURM_JOB_SCRIPT_SHA256": "cs6_hapg_full_source_cover_slurm_job.sh",
        "HPG_WORKER_BINARY_SHA256": "hpg-worker-binary",
        "HAPG_WORKER_BINARY_SHA256": "hapg-worker-binary",
    }
    if not historical:
        prebuilt_declarations["KAT_ANCHOR_SHA256"] = (
            "cs6_hapg_full_source_cover_kat_anchor.py"
        )
    for key, filename in prebuilt_declarations.items():
        if prebuilt.get(key) != _sha(_file(view, f"{prebuilt_prefix}/{filename}")):
            _fail(f"KAT prebuilt declaration mismatch: {filename}")

    source_declarations = {
        "HPG_WORKER_SOURCE_SHA256": "cs6_plucker_cocycle_probe.cpp",
        "HPG_VERIFIER_SOURCE_SHA256": "cs6_plucker_cocycle_verify.py",
        "HAPG_WORKER_SOURCE_SHA256": "cs6_hapg_full_source_cover_worker.cpp",
        "HAPG_KERNEL_SOURCE_SHA256": "cs6_affine_projective_cocycle_full53_probe.cpp",
        "HAPG_VERIFIER_ADAPTER_SHA256": "cs6_hapg_full_source_cover_verify.py",
        "HAPG_NUMERIC_VERIFIER_SHA256": "cs6_affine_projective_cocycle_full53_verify.py",
        "SLURM_JOB_SCRIPT_SHA256": "cs6_hapg_full_source_cover_slurm_job.sh",
    }
    for key, filename in source_declarations.items():
        if run_contract[key] != _sha(_file(view, f"result/{filename}")):
            _fail(f"KAT run source declaration mismatch: {filename}")
    if not historical:
        frozen_source_keys = {
            "HPG_WORKER_SOURCE_SHA256": "PREPASS_WORKER_SHA256",
            "HPG_VERIFIER_SOURCE_SHA256": "PREPASS_VERIFIER_SHA256",
            "HAPG_WORKER_SOURCE_SHA256": "H_APG_WRAPPER_SHA256",
            "HAPG_KERNEL_SOURCE_SHA256": "H_APG_KERNEL_SHA256",
            "HAPG_VERIFIER_ADAPTER_SHA256": "H_APG_ADAPTER_SHA256",
            "HAPG_NUMERIC_VERIFIER_SHA256": "H_APG_NUMERIC_VERIFIER_SHA256",
            "KAT_ANCHOR_SHA256": "KAT_ANCHOR_SHA256",
            "SLURM_JOB_SCRIPT_SHA256": "SLURM_JOB_SCRIPT_SHA256",
        }
        if any(
            run_contract[run_key] != frozen.get(frozen_key)
            or run_contract[run_key] != prebuilt.get(run_key)
            for run_key, frozen_key in frozen_source_keys.items()
        ):
            _fail("KAT run, prebuilt, and frozen source declarations differ")
        if (
            run_contract["JOBS"] != frozen.get("BOUNDED_PILOT_JOBS")
            or run_contract["TIMEOUT_SECONDS"]
            != frozen.get("BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS")
        ):
            _fail("KAT execution parameters differ from the frozen contract")

    config_sha = transport["CONFIG_SHA256"]
    if not historical:
        config_raw = _file(view, "transport-config.txt")
        config = _parse_kv(config_raw, SLURM_CONFIG_KEYS, "KAT transport config")
        _require_exact(
            config,
            {
                "SCHEMA": "sounio.cs6.hapg-full-source-cover-slurm-config.v3",
                "MODE": "kat",
                "BASE_REPO_BUNDLE_SHA256": expectations.expected_base_repo_bundle_sha256,
                "BASE_GIT_HEAD": expectations.expected_base_git_head,
                "REPO_DELTA_BUNDLE_SHA256": expectations.expected_repo_delta_bundle_sha256,
                "PREBUILT_ARCHIVE_SHA256": expectations.expected_prebuilt_archive_sha256,
                "EXPECTED_GIT_HEAD": expectations.expected_git_head,
                "EXPECTED_CONTRACT_SHA256": expectations.expected_contract_sha256,
            },
            "KAT transport config",
        )
        if (
            _sha(config_raw) != config_sha
            or any(
                not config[key].startswith("/orangefs/training/")
                for key in (
                    "BASE_REPO_BUNDLE_PATH",
                    "REPO_DELTA_BUNDLE_PATH",
                    "PREBUILT_ARCHIVE_PATH",
                    "OUTPUT_DIRECTORY",
                )
            )
        ):
            _fail("KAT transport config digest or path policy mismatch")

    evaluations = _validate_evaluations(_file(view, "result/evaluations.tsv"))
    expected_counts = {
        "EVALUATED": int(frozen.get("KAT_EXPECTED_ATTEMPTED", "-1")),
        "HPG_SIGNED": int(frozen.get("KAT_EXPECTED_H_PG_VALID", "-1")),
        "HAPG_ATTEMPTED": int(frozen.get("KAT_EXPECTED_H_APG_VALID", "-1")),
        "HAPG_CERTIFIED": int(frozen.get("KAT_EXPECTED_H_APG_CERTIFIED", "-1")),
        "HAPG_RESCUE": int(frozen.get("KAT_EXPECTED_H_APG_RESCUES", "-1")),
    }
    if evaluations != expected_counts:
        _fail("KAT evaluation ledger counts differ from the frozen contract")
    summary_expected = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-summary.v1",
        "MODE": "kat",
        "BOUNDED_RUN_COMPLETE": "true",
        "INFRASTRUCTURE_VALID": "true",
        "EVALUATED_NODE_COUNT": str(evaluations["EVALUATED"]),
        "WAVE_COUNT": "1",
        "HPG_SIGNED_CHART_COUNT": str(evaluations["HPG_SIGNED"]),
        "HAPG_ATTEMPTED_COUNT": str(evaluations["HAPG_ATTEMPTED"]),
        "HAPG_CERTIFIED_COUNT": str(evaluations["HAPG_CERTIFIED"]),
        "HAPG_RESCUE_COUNT": str(evaluations["HAPG_RESCUE"]),
        "FRESH_REPLAY_TERMINAL_COUNT": "0",
        "FRESH_REPLAY_WAVE_COUNT": "0",
        "FRESH_REPLAY_COMPLETE": "false",
        "TREE_NODE_COUNT": "0",
        "CERTIFIED_TERMINAL_COUNT": "0",
        "UNRESOLVED_TERMINAL_COUNT": "0",
        "UNRESOLVED_AREA_NUMERATOR": "0",
        "UNRESOLVED_AREA_DENOMINATOR": "1",
        "HAPG_FULL_SOURCE_COVER_CANDIDATE": "false",
        "AGGREGATION_REQUIRED": "false",
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "FULL_SOURCE_CARRIER_PROVED": "false",
        "HYPERBOLICITY_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false",
        "OPEN_PROBLEM_SOLVED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    _require_exact(summary, summary_expected, "KAT summary")
    if (
        summary["HPG_MUTATION_TESTS"] != summary["HPG_MUTATIONS_REJECTED"]
        or summary["HAPG_MUTATION_TESTS"] != summary["HAPG_MUTATIONS_REJECTED"]
        or int(summary["HPG_MUTATION_TESTS"]) <= 0
        or int(summary["HAPG_MUTATION_TESTS"]) <= 0
        or run_manifest["EVALUATED_NODE_COUNT"] != summary["EVALUATED_NODE_COUNT"]
    ):
        _fail("KAT mutation audit or run/summary count binding mismatch")

    leaf_evidence = {
        "coordinate_sha": run_contract["KAT_COORDINATE_MANIFEST_SHA256"],
        "expected_sha": run_contract["KAT_EXPECTED_RESULTS_SHA256"],
        "wave_sha": ZERO_SHA256,
        "result_sha": ZERO_SHA256,
        "hpg_replays": "0",
        "hapg_replays": "0",
    }
    leaf_evidence_valid = expectations.schema_profile == "v6"
    if leaf_evidence_valid:
        leaf_evidence = _validate_leaf_evidence(
            view, run_contract, frozen, summary, transport
        )

    sacct = _parse_sacct(
        sacct_bytes,
        expectations.kat_job_id,
        allow_legacy_v5=historical,
    )
    kat_end = _parse_timestamp(sacct["END_UTC"], "KAT end")
    if kat_end > adaptive_submit:
        _fail("KAT ended after adaptive submission")
    if (
        sacct["PARTITION"] != frozen.get("BOUNDED_PILOT_SLURM_PARTITION")
        or sacct["NODE"] != frozen.get("BOUNDED_PILOT_SLURM_NODE")
        or sacct["NODE"] != transport.get("SLURM_NODE")
        or sacct["ALLOC_CPUS"] != frozen.get("BOUNDED_PILOT_SLURM_ALLOCATED_CPUS")
        or sacct["REQ_CPUS"] != frozen.get("BOUNDED_PILOT_SLURM_CPUS_PER_TASK")
        or run_contract["EXECUTION_NODE"] != sacct["NODE"]
        or result_job.get("JobId") != expectations.kat_job_id
        or transport_job.get("JobId") != expectations.kat_job_id
        or result_job.get("JobState") != "RUNNING"
        or transport_job.get("JobState") != "RUNNING"
    ):
        _fail("KAT scheduler or allocation binding mismatch")
    if not historical:
        expected_job_name = f"cs6-hapg-kat-v6-{config_sha[:12]}"
        user_match = re.fullmatch(r"([^()]+)\(([0-9]+)\)", result_job.get("UserId", ""))
        expected_control = {
            "JobId": expectations.kat_job_id,
            "JobName": expected_job_name,
            "Account": str(frozen.get("BOUNDED_PILOT_SLURM_ACCOUNT", "")),
            "QOS": str(frozen.get("BOUNDED_PILOT_SLURM_QOS", "")),
            "JobState": "RUNNING",
            "Partition": str(frozen.get("BOUNDED_PILOT_SLURM_PARTITION", "")),
            "SubmitTime": sacct["SUBMIT_UTC"],
            "StartTime": sacct["START_UTC"],
            "NodeList": str(frozen.get("BOUNDED_PILOT_SLURM_NODE", "")),
            "NumNodes": str(frozen.get("BOUNDED_PILOT_SLURM_NODES", "")),
            "NumCPUs": str(frozen.get("BOUNDED_PILOT_SLURM_ALLOCATED_CPUS", "")),
            "NumTasks": str(frozen.get("BOUNDED_PILOT_SLURM_TASKS", "")),
            "CPUs/Task": str(frozen.get("BOUNDED_PILOT_SLURM_CPUS_PER_TASK", "")),
            "Restarts": "0",
        }
        if (
            user_match is None
            or any(
                record.get(key) != value
                for record in (result_job, transport_job)
                for key, value in expected_control.items()
            )
            or transport_job.get("UserId") != result_job.get("UserId")
            or sacct["JOB_NAME"] != expected_job_name
            or sacct["CLUSTER"] != frozen.get("BOUNDED_PILOT_SLURM_CLUSTER")
            or sacct["ACCOUNT"] != frozen.get("BOUNDED_PILOT_SLURM_ACCOUNT")
            or sacct["QOS"] != frozen.get("BOUNDED_PILOT_SLURM_QOS")
            or sacct["USER"] != user_match.group(1)
            or sacct["UID"] != user_match.group(2)
            or sacct["RESTARTS"] != "0"
            or sacct["ALLOC_NODES"] != frozen.get("BOUNDED_PILOT_SLURM_NODES")
            or sacct["ALLOC_TASKS"]
            not in {"", frozen.get("BOUNDED_PILOT_SLURM_TASKS")}
        ):
            _fail("KAT external scheduler identity or chronology mismatch")

    prerequisite_valid = expectations.schema_profile == "v6"
    if historical:
        certificate_scope = "HISTORICAL_FIXTURE_ONLY"
    elif synthetic:
        certificate_scope = "SYNTHETIC_SELF_TEST_ONLY"
    else:
        certificate_scope = "AUTHORITATIVE_V6_ADAPTIVE_PREREQUISITE"
    fields = (
        ("SCHEMA", CERTIFICATE_SCHEMA),
        ("CERTIFICATE_SCOPE", certificate_scope),
        ("KAT_SCHEMA_PROFILE", expectations.schema_profile),
        ("KAT_PREREQUISITE_VALID", str(prerequisite_valid).lower()),
        ("KAT_JOB_ID", expectations.kat_job_id),
        ("KAT_ARCHIVE_BASENAME", archive.name),
        ("KAT_ARCHIVE_SHA256", archive_sha),
        ("KAT_ARCHIVE_SIDECAR_SHA256", _sha(sidecar_raw)),
        ("KAT_ARCHIVE_MEMBER_COUNT", str(view.member_count)),
        ("KAT_ARCHIVE_REGULAR_FILE_COUNT", str(len(view.files))),
        ("KAT_ARCHIVE_DIRECTORY_COUNT", str(len(view.directories))),
        ("KAT_TRANSPORT_MANIFEST_SHA256", _sha(transport_raw)),
        ("KAT_TRANSPORT_JOB_RECORD_SHA256", _sha(transport_job_raw)),
        ("KAT_RESULT_RUN_MANIFEST_SHA256", _sha(run_manifest_raw)),
        ("KAT_RESULT_FILES_INDEX_SHA256", _sha(_file(view, "result/files.sha256"))),
        ("KAT_RESULT_FILES_INDEX_ENTRY_COUNT", str(len(result_index))),
        ("KAT_RESULT_RUN_CONTRACT_SHA256", _sha(run_contract_raw)),
        ("KAT_RESULT_SUMMARY_SHA256", _sha(summary_raw)),
        ("KAT_RESULT_JOB_RECORD_SHA256", _sha(result_job_raw)),
        ("KAT_PREBUILT_RUN_MANIFEST_SHA256", _sha(prebuilt_manifest_raw)),
        (
            "KAT_PREBUILT_FILES_INDEX_SHA256",
            _sha(_file(view, f"{prebuilt_prefix}/files.sha256")),
        ),
        ("KAT_PREBUILT_FILES_INDEX_ENTRY_COUNT", str(len(prebuilt_index))),
        ("KAT_GIT_HEAD_FILE_SHA256", _sha(_file(view, "result/git-head.txt"))),
        ("KAT_GIT_STATUS_FILE_SHA256", _sha(_file(view, "result/git-status.txt"))),
        ("KAT_SACCT_SHA256", _sha(sacct_bytes)),
        ("KAT_SACCT_STATE", sacct["STATE"]),
        ("KAT_SACCT_EXIT_CODE", sacct["EXIT_CODE"]),
        ("KAT_SUBMIT_UTC", sacct["SUBMIT_UTC"]),
        ("KAT_START_UTC", sacct["START_UTC"]),
        ("KAT_END_UTC", sacct["END_UTC"]),
        ("KAT_CLUSTER", sacct["CLUSTER"]),
        ("KAT_ACCOUNT", sacct["ACCOUNT"]),
        ("KAT_QOS", sacct["QOS"]),
        ("KAT_USER", sacct["USER"]),
        ("KAT_UID", sacct["UID"]),
        ("KAT_RESTARTS", sacct["RESTARTS"]),
        ("KAT_NODE", sacct["NODE"]),
        ("KAT_ALLOC_NODES", sacct["ALLOC_NODES"]),
        ("KAT_ALLOC_TASKS", result_job["NumTasks"]),
        ("KAT_ALLOC_CPUS", sacct["ALLOC_CPUS"]),
        ("KAT_REQ_CPUS", sacct["REQ_CPUS"]),
        ("KAT_CONFIG_SHA256", transport["CONFIG_SHA256"]),
        ("KAT_ROOT_CHALLENGE", run_contract["ROOT_CHALLENGE"]),
        ("KAT_COORDINATE_MANIFEST_SHA256", leaf_evidence["coordinate_sha"]),
        ("KAT_EXPECTED_RESULTS_SHA256", leaf_evidence["expected_sha"]),
        ("KAT_WAVE_CONTRACT_SHA256", leaf_evidence["wave_sha"]),
        ("KAT_WAVE_RESULT_SHA256", leaf_evidence["result_sha"]),
        ("KAT_LEAF_EVIDENCE_VALID", str(leaf_evidence_valid).lower()),
        ("KAT_HPG_VERIFIER_REPLAY_COUNT", leaf_evidence["hpg_replays"]),
        ("KAT_HAPG_VERIFIER_REPLAY_COUNT", leaf_evidence["hapg_replays"]),
        ("KAT_EVALUATED_NODE_COUNT", str(evaluations["EVALUATED"])),
        ("KAT_HPG_SIGNED_CHART_COUNT", str(evaluations["HPG_SIGNED"])),
        ("KAT_HAPG_ATTEMPTED_COUNT", str(evaluations["HAPG_ATTEMPTED"])),
        ("KAT_HAPG_CERTIFIED_COUNT", str(evaluations["HAPG_CERTIFIED"])),
        (
            "KAT_HAPG_UNCERTIFIED_COUNT",
            str(evaluations["HAPG_ATTEMPTED"] - evaluations["HAPG_CERTIFIED"]),
        ),
        ("KAT_HAPG_RESCUE_COUNT", str(evaluations["HAPG_RESCUE"])),
        ("KAT_HPG_MUTATION_TESTS", summary["HPG_MUTATION_TESTS"]),
        ("KAT_HPG_MUTATIONS_REJECTED", summary["HPG_MUTATIONS_REJECTED"]),
        ("KAT_HAPG_MUTATION_TESTS", summary["HAPG_MUTATION_TESTS"]),
        ("KAT_HAPG_MUTATIONS_REJECTED", summary["HAPG_MUTATIONS_REJECTED"]),
        ("KAT_EXPECTED_GIT_HEAD", expectations.expected_git_head),
        ("KAT_FROZEN_CONTRACT_SHA256", expectations.expected_contract_sha256),
        ("KAT_BASE_REPO_BUNDLE_SHA256", expectations.expected_base_repo_bundle_sha256),
        ("KAT_BASE_GIT_HEAD", expectations.expected_base_git_head),
        ("KAT_REPO_DELTA_BUNDLE_SHA256", expectations.expected_repo_delta_bundle_sha256),
        ("KAT_PREBUILT_ARCHIVE_SHA256", expectations.expected_prebuilt_archive_sha256),
        ("KAT_SLURM_JOB_SCRIPT_SHA256", expectations.expected_slurm_job_script_sha256),
        ("KAT_ANCHOR_SOURCE_SHA256", anchor_source_sha),
        ("ADAPTIVE_JOB_ID", adaptive_job_id),
        ("ADAPTIVE_SUBMIT_UTC", adaptive_submit_utc),
        ("KAT_END_NOT_AFTER_ADAPTIVE_SUBMIT", "true"),
        ("EXECUTION_PROVENANCE_ATTESTED", "false"),
        ("PROMOTION_ELIGIBLE", "false"),
    )
    certificate = KatAnchorCertificate(fields)
    if tuple(certificate.as_dict()) != CERTIFICATE_KEYS:
        _fail("internal KAT certificate field order mismatch")
    parse_kat_anchor_certificate(certificate.as_bytes())
    return certificate


def _write_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    path.write_bytes(_canonical_kv(fields))


def _directory_index(root: Path) -> bytes:
    rows = []
    for path in sorted(
        root.rglob("*"), key=lambda candidate: candidate.relative_to(root).as_posix()
    ):
        if not path.is_file() or path in {
            root / "files.sha256",
            root / "run-manifest.txt",
        }:
            continue
        rows.append(f"{_sha(path.read_bytes())}  {path.relative_to(root).as_posix()}")
    return ("\n".join(rows) + "\n").encode("ascii")


def _self_test_evaluations() -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=EVALUATION_COLUMNS,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for index in range(53):
        signed = index > 0
        certified = 1 <= index <= 48
        rescue = 1 <= index <= 20
        writer.writerow(
            {
                "WAVE_INDEX": "1",
                "NODE_ID": f"KAT-{index:03d}",
                "PARENT_ID": "-",
                "U_DEPTH": "0",
                "U_INDEX": str(index),
                "S_DEPTH": "0",
                "S_INDEX": "0",
                "WAVE_CONTRACT_SHA256": "1" * 64,
                "HPG_STATUS": (
                    "HPG_VERIFIED_SIGNED_CHARTS" if signed else "H_PG_INTERVAL_DOMAIN"
                ),
                "HPG_RECEIPT_SHA256": "2" * 64,
                "HPG_VERIFICATION_SHA256": "3" * 64,
                "HAPG_STATUS": "HAPG_VERIFIED" if signed else "H_APG_NOT_ELIGIBLE",
                "HAPG_RECEIPT_SHA256": "4" * 64 if signed else ZERO_SHA256,
                "HAPG_VERIFICATION_SHA256": "5" * 64 if signed else ZERO_SHA256,
                "APG_VALID": str(signed).lower(),
                "APG_PASS": str(certified).lower(),
                "APG_RESCUE": str(rescue).lower(),
                "GENERIC_CERTIFICATE_PASS": "false",
                "DECISION": "KAT",
                "TERMINAL_REASON": "KAT",
            }
        )
    return output.getvalue().encode("ascii")


def _build_self_test_fixture(root: Path) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    result = root / "result"
    prebuilt = result / "prebuilt-origin"
    result.mkdir()
    prebuilt.mkdir()
    for directory in EMPTY_DIRS:
        (result / directory).mkdir()
        (prebuilt / directory).mkdir()

    source_names = (
        "cs6_plucker_cocycle_probe.cpp",
        "cs6_plucker_cocycle_verify.py",
        "cs6_hapg_full_source_cover_worker.cpp",
        "cs6_affine_projective_cocycle_full53_probe.cpp",
        "cs6_hapg_full_source_cover_verify.py",
        "cs6_affine_projective_cocycle_full53_verify.py",
        "cs6_hapg_full_source_cover_run.py",
        "cs6_hapg_full_source_cover_aggregate.py",
        "cs6_c1_full_source_cover_aggregate.py",
        "cs6_hapg_full_source_cover_gate.sh",
        "cs6_hapg_full_source_cover_slurm_job.sh",
    )
    source_bytes = {
        name: f"self-test-source:{name}\n".encode("ascii") for name in source_names
    }
    anchor_name = "cs6_hapg_full_source_cover_kat_anchor.py"
    anchor_bytes = Path(__file__).resolve().read_bytes()
    git_head = "a" * 40
    base_head = "b" * 40
    base_sha = "c" * 64
    delta_sha = "d" * 64
    prebuilt_archive_sha = "e" * 64
    kat_root = "f" * 64
    contract_fields = (
        ("SCHEMA", "sounio.cs6.hapg-full-source-cover-contract.v6"),
        ("BASE_REPO_BUNDLE_SHA256", base_sha),
        ("BASE_REPO_BUNDLE_GIT_HEAD", base_head),
        (
            "PREPASS_WORKER_SHA256",
            _sha(source_bytes["cs6_plucker_cocycle_probe.cpp"]),
        ),
        (
            "PREPASS_VERIFIER_SHA256",
            _sha(source_bytes["cs6_plucker_cocycle_verify.py"]),
        ),
        (
            "H_APG_WRAPPER_SHA256",
            _sha(source_bytes["cs6_hapg_full_source_cover_worker.cpp"]),
        ),
        (
            "H_APG_KERNEL_SHA256",
            _sha(source_bytes["cs6_affine_projective_cocycle_full53_probe.cpp"]),
        ),
        (
            "H_APG_ADAPTER_SHA256",
            _sha(source_bytes["cs6_hapg_full_source_cover_verify.py"]),
        ),
        (
            "H_APG_NUMERIC_VERIFIER_SHA256",
            _sha(source_bytes["cs6_affine_projective_cocycle_full53_verify.py"]),
        ),
        (
            "SLURM_JOB_SCRIPT_SHA256",
            _sha(source_bytes["cs6_hapg_full_source_cover_slurm_job.sh"]),
        ),
        ("KAT_ANCHOR_SHA256", _sha(anchor_bytes)),
        ("KAT_ROOT_CHALLENGE", kat_root),
        ("KAT_EXPECTED_ATTEMPTED", "53"),
        ("KAT_EXPECTED_H_PG_VALID", "52"),
        ("KAT_EXPECTED_H_APG_VALID", "52"),
        ("KAT_EXPECTED_H_APG_CERTIFIED", "48"),
        ("KAT_EXPECTED_H_APG_UNCERTIFIED", "4"),
        ("KAT_EXPECTED_H_APG_RESCUES", "20"),
        ("KAT_EXPECTED_HPG_MUTATION_TESTS", "4108"),
        ("KAT_EXPECTED_HPG_MUTATIONS_REJECTED", "4108"),
        ("KAT_EXPECTED_HAPG_MUTATION_TESTS", "5824"),
        ("KAT_EXPECTED_HAPG_MUTATIONS_REJECTED", "5824"),
        ("BOUNDED_PILOT_JOBS", "32"),
        ("BOUNDED_PILOT_LEAF_TIMEOUT_SECONDS", "300"),
        ("BOUNDED_PILOT_SLURM_CLUSTER", "beagle-slurm-pilot"),
        ("BOUNDED_PILOT_SLURM_PARTITION", "gpu-orangefs"),
        ("BOUNDED_PILOT_SLURM_ACCOUNT", "lab"),
        ("BOUNDED_PILOT_SLURM_QOS", "normal"),
        ("BOUNDED_PILOT_SLURM_NODES", "1"),
        ("BOUNDED_PILOT_SLURM_TASKS", "1"),
        ("BOUNDED_PILOT_SLURM_NODE", "gpuorangefs-r770-proxmox"),
        ("BOUNDED_PILOT_SLURM_ALLOCATED_CPUS", "120"),
        ("BOUNDED_PILOT_SLURM_CPUS_PER_TASK", "32"),
    )
    contract_name = "cs6_hapg_full_source_cover_contract_v6.txt"
    contract_raw = _canonical_kv(contract_fields)
    contract_sha = _sha(contract_raw)

    for directory in (result, prebuilt):
        for name, raw in source_bytes.items():
            (directory / name).write_bytes(raw)
        (directory / anchor_name).write_bytes(anchor_bytes)
        (directory / contract_name).write_bytes(contract_raw)
    for name in ("hpg-worker-binary", "hapg-worker-binary"):
        (prebuilt / name).write_bytes(f"self-test-binary:{name}\n".encode("ascii"))

    prebuilt_index_raw = _directory_index(prebuilt)
    (prebuilt / "files.sha256").write_bytes(prebuilt_index_raw)
    prebuilt_index = _parse_index(prebuilt_index_raw, "self-test prebuilt index")
    prebuilt_fields = (
        ("SCHEMA", "sounio.cs6.hapg-full-source-cover-prebuilt.v2"),
        ("RUN_COMPLETE", "true"),
        ("MODE", "prepare"),
        ("CAPD_VERSION", "5.3.0"),
        ("INTERVAL_BACKEND", "FILIB"),
        ("OPTIMIZATION_LEVEL", "O0"),
        ("FROZEN_CONTRACT_SHA256", contract_sha),
        ("HPG_WORKER_SOURCE_SHA256", prebuilt_index["cs6_plucker_cocycle_probe.cpp"]),
        ("HPG_VERIFIER_SOURCE_SHA256", prebuilt_index["cs6_plucker_cocycle_verify.py"]),
        ("HAPG_WORKER_SOURCE_SHA256", prebuilt_index["cs6_hapg_full_source_cover_worker.cpp"]),
        ("HAPG_KERNEL_SOURCE_SHA256", prebuilt_index["cs6_affine_projective_cocycle_full53_probe.cpp"]),
        ("HAPG_VERIFIER_ADAPTER_SHA256", prebuilt_index["cs6_hapg_full_source_cover_verify.py"]),
        ("HAPG_NUMERIC_VERIFIER_SHA256", prebuilt_index["cs6_affine_projective_cocycle_full53_verify.py"]),
        ("RUNNER_SHA256", prebuilt_index["cs6_hapg_full_source_cover_run.py"]),
        ("AGGREGATOR_SHA256", prebuilt_index["cs6_hapg_full_source_cover_aggregate.py"]),
        ("KAT_ANCHOR_SHA256", prebuilt_index[anchor_name]),
        ("EXACT_TREE_KERNEL_SHA256", prebuilt_index["cs6_c1_full_source_cover_aggregate.py"]),
        ("GATE_SHA256", prebuilt_index["cs6_hapg_full_source_cover_gate.sh"]),
        ("SLURM_JOB_SCRIPT_SHA256", prebuilt_index["cs6_hapg_full_source_cover_slurm_job.sh"]),
        ("HPG_WORKER_BINARY_SHA256", prebuilt_index["hpg-worker-binary"]),
        ("HAPG_WORKER_BINARY_SHA256", prebuilt_index["hapg-worker-binary"]),
        ("FILES_INDEX_SHA256", _sha(prebuilt_index_raw)),
        ("FILE_COUNT", str(len(prebuilt_index))),
        ("PROMOTION_ELIGIBLE", "false"),
    )
    _write_kv(prebuilt / "run-manifest.txt", prebuilt_fields)
    prebuilt_manifest_sha = _sha((prebuilt / "run-manifest.txt").read_bytes())

    config_values = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-slurm-config.v3",
        "MODE": "kat",
        "BASE_REPO_BUNDLE_PATH": "/orangefs/training/cs6/base.bundle",
        "BASE_REPO_BUNDLE_SHA256": base_sha,
        "BASE_GIT_HEAD": base_head,
        "REPO_DELTA_BUNDLE_PATH": "/orangefs/training/cs6/delta.bundle",
        "REPO_DELTA_BUNDLE_SHA256": delta_sha,
        "PREBUILT_ARCHIVE_PATH": "/orangefs/training/cs6/prebuilt.tar",
        "PREBUILT_ARCHIVE_SHA256": prebuilt_archive_sha,
        "EXPECTED_GIT_HEAD": git_head,
        "EXPECTED_CONTRACT_SHA256": contract_sha,
        "OUTPUT_DIRECTORY": "/orangefs/training/cs6/output",
    }
    config_path = root / "transport-config.txt"
    _write_kv(
        config_path,
        tuple((key, config_values[key]) for key in SLURM_CONFIG_KEYS),
    )
    config_sha = _sha(config_path.read_bytes())

    job_id = "101"
    node = "gpuorangefs-r770-proxmox"
    job_name = f"cs6-hapg-kat-v6-{config_sha[:12]}"
    slurm_record = (
        f"JobId={job_id} JobName={job_name} UserId=tester(1000) Account=lab "
        "QOS=normal JobState=RUNNING Partition=gpu-orangefs "
        "SubmitTime=2026-08-02T00:00:00 StartTime=2026-08-02T00:00:01 "
        f"NodeList={node} NumNodes=1 NumCPUs=120 NumTasks=1 CPUs/Task=32 Restarts=0\n"
    ).encode("ascii")
    (result / "git-head.txt").write_text(git_head + "\n", encoding="ascii")
    (result / "git-status.txt").write_bytes(b"")
    (result / "slurm-job-record.txt").write_bytes(slurm_record)
    (result / "evaluations.tsv").write_bytes(_self_test_evaluations())
    summary_values = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-summary.v1",
        "MODE": "kat",
        "BOUNDED_RUN_COMPLETE": "true",
        "INFRASTRUCTURE_VALID": "true",
        "EVALUATED_NODE_COUNT": "53",
        "WAVE_COUNT": "1",
        "HPG_SIGNED_CHART_COUNT": "52",
        "HAPG_ATTEMPTED_COUNT": "52",
        "HAPG_CERTIFIED_COUNT": "48",
        "HAPG_RESCUE_COUNT": "20",
        "HPG_MUTATION_TESTS": "4108",
        "HPG_MUTATIONS_REJECTED": "4108",
        "HAPG_MUTATION_TESTS": "5824",
        "HAPG_MUTATIONS_REJECTED": "5824",
        "FRESH_REPLAY_TERMINAL_COUNT": "0",
        "FRESH_REPLAY_WAVE_COUNT": "0",
        "FRESH_REPLAY_COMPLETE": "false",
        "TREE_NODE_COUNT": "0",
        "CERTIFIED_TERMINAL_COUNT": "0",
        "UNRESOLVED_TERMINAL_COUNT": "0",
        "UNRESOLVED_AREA_NUMERATOR": "0",
        "UNRESOLVED_AREA_DENOMINATOR": "1",
        "HAPG_FULL_SOURCE_COVER_CANDIDATE": "false",
        "AGGREGATION_REQUIRED": "false",
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "FULL_SOURCE_CARRIER_PROVED": "false",
        "HYPERBOLICITY_PROVED": "false",
        "CHAOTIC_ATTRACTOR_PROVED": "false",
        "OPEN_PROBLEM_SOLVED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    _write_kv(result / "summary.txt", tuple(summary_values.items()))

    run_source = {
        "HPG_WORKER_SOURCE_SHA256": "cs6_plucker_cocycle_probe.cpp",
        "HPG_VERIFIER_SOURCE_SHA256": "cs6_plucker_cocycle_verify.py",
        "HAPG_WORKER_SOURCE_SHA256": "cs6_hapg_full_source_cover_worker.cpp",
        "HAPG_KERNEL_SOURCE_SHA256": "cs6_affine_projective_cocycle_full53_probe.cpp",
        "HAPG_VERIFIER_ADAPTER_SHA256": "cs6_hapg_full_source_cover_verify.py",
        "HAPG_NUMERIC_VERIFIER_SHA256": "cs6_affine_projective_cocycle_full53_verify.py",
    }
    run_values = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-run-contract.v2",
        "FROZEN_CONTRACT_SHA256": contract_sha,
        "MODE": "kat",
        "SOURCE": "N0",
        "ROOT_CHALLENGE": kat_root,
        "TRAVERSAL": "DETERMINISTIC_BREADTH_FIRST",
        "SPLIT_RULE": "S_IF_S_DEPTH_LE_U_DEPTH_ELSE_U",
        "TERMINAL_PREDICATE": "APG_COMPUTATION_VALID_AND_APG_CERTIFICATE_PASS",
    }
    run_values.update(
        {key: _sha((result / filename).read_bytes()) for key, filename in run_source.items()}
    )
    run_values.update(
        {
            "KAT_ANCHOR_SHA256": _sha(anchor_bytes),
            "SLURM_JOB_SCRIPT_SHA256": _sha(
                source_bytes["cs6_hapg_full_source_cover_slurm_job.sh"]
            ),
            "BUILD_MODE": "VERIFIED_PREBUILT_BUNDLE",
            "PREBUILT_RUN_MANIFEST_SHA256": prebuilt_manifest_sha,
            "SLURM_JOB_ID": job_id,
            "EXECUTION_NODE": node,
            "SLURM_JOB_VERIFIED": "true",
            "SLURM_JOB_RECORD_SHA256": _sha(slurm_record),
            "WORKING_FILESYSTEM_POLICY": "NODE_LOCAL_TMP_THEN_HASHED_ARCHIVE_TRANSPORT",
            "JOBS": "32",
            "TIMEOUT_SECONDS": "300",
            "MUTATION_AUDIT": "true",
            "LOCAL_PROCESS_ORDERED_HASH_CHAIN": "true",
            "EXECUTION_PROVENANCE_ATTESTED": "false",
            "PROMOTION_ELIGIBLE": "false",
            "KAT_COORDINATE_MANIFEST_SHA256": "6" * 64,
            "KAT_EXPECTED_RESULTS_SHA256": "7" * 64,
        }
    )
    run_order = RUN_CONTRACT_BASE_KEYS[:14] + ("KAT_ANCHOR_SHA256",) + RUN_CONTRACT_BASE_KEYS[14:]
    _write_kv(result / "run-contract.txt", tuple((key, run_values[key]) for key in run_order))

    result_index_raw = _directory_index(result)
    (result / "files.sha256").write_bytes(result_index_raw)
    result_index = _parse_index(result_index_raw, "self-test result index")
    run_manifest_values = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-run-manifest.v1",
        "RUN_COMPLETE": "true",
        "MODE": "kat",
        "ROOT_CHALLENGE": kat_root,
        "CAPD_VERSION": "5.3.0",
        "INTERVAL_BACKEND": "FILIB",
        "OPTIMIZATION_LEVEL": "O0",
        "RUN_CONTRACT_SHA256": _sha((result / "run-contract.txt").read_bytes()),
        "FILES_INDEX_SHA256": _sha(result_index_raw),
        "FILE_COUNT": str(len(result_index)),
        "EVALUATED_NODE_COUNT": "53",
        "WAVE_COUNT": "1",
        "LOCAL_PROCESS_ORDERED_HASH_CHAIN": "true",
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    _write_kv(
        result / "run-manifest.txt",
        tuple((key, run_manifest_values[key]) for key in RUN_MANIFEST_KEYS),
    )

    transport_record = root / "transport-slurm-job-record.txt"
    transport_record.write_bytes(slurm_record)
    transport_values = {
        "SCHEMA": "sounio.cs6.hapg-full-source-cover-transport.v3",
        "MODE": "kat",
        "SLURM_JOB_ID": job_id,
        "SLURM_NODE": node,
        "EXPECTED_GIT_HEAD": git_head,
        "EXPECTED_CONTRACT_SHA256": contract_sha,
        "SLURM_JOB_SCRIPT_SHA256": _sha(
            source_bytes["cs6_hapg_full_source_cover_slurm_job.sh"]
        ),
        "CONFIG_SHA256": config_sha,
        "PYTHON_EXECUTABLE_REALPATH": "/usr/bin/python3",
        "BASE_REPO_BUNDLE_SHA256": base_sha,
        "BASE_GIT_HEAD": base_head,
        "REPO_DELTA_BUNDLE_SHA256": delta_sha,
        "PREBUILT_ARCHIVE_SHA256": prebuilt_archive_sha,
        "KAT_PREREQUISITE_REQUIRED": "false",
        "KAT_JOB_ID": "0",
        "KAT_ARCHIVE_SHA256": ZERO_SHA256,
        "KAT_CERTIFICATE_SHA256": ZERO_SHA256,
        "RESULT_RUN_MANIFEST_SHA256": _sha((result / "run-manifest.txt").read_bytes()),
        "RESULT_FILES_INDEX_SHA256": _sha(result_index_raw),
        "AGGREGATION_SHA256": ZERO_SHA256,
        "POST_RUN_GATE_PASS": "true",
        "FAILURE_STAGE": "NONE",
        "FAILURE_RC": "0",
        "EXECUTION_PROVENANCE_ATTESTED": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    transport = root / "transport-manifest.txt"
    _write_kv(
        transport,
        tuple((key, transport_values[key]) for key in TRANSPORT_V3_KEYS),
    )

    archive = root / "kat-self-test.tar"
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as output:
        output.add(result, arcname="result", recursive=True)
        output.add(transport, arcname="transport-manifest.txt", recursive=False)
        output.add(
            transport_record,
            arcname="transport-slurm-job-record.txt",
            recursive=False,
        )
        output.add(config_path, arcname="transport-config.txt", recursive=False)
    archive_sha = _sha(archive.read_bytes())
    sidecar = Path(f"{archive}.sha256")
    sidecar.write_text(f"{archive_sha}  {archive.name}\n", encoding="ascii")
    sacct = (
        f"{job_id}|{job_name}|beagle-slurm-pilot|gpu-orangefs|lab|normal|"
        "tester|1000|0|COMPLETED|0:0|2026-08-02T00:00:00|"
        f"2026-08-02T00:00:01|2026-08-02T00:01:00|59|{node}|1||120|32|\n"
    ).encode("ascii")
    expectations = KatAnchorExpectations(
        kat_job_id=job_id,
        kat_archive_sha256=archive_sha,
        expected_git_head=git_head,
        expected_contract_sha256=contract_sha,
        expected_base_repo_bundle_sha256=base_sha,
        expected_base_git_head=base_head,
        expected_repo_delta_bundle_sha256=delta_sha,
        expected_prebuilt_archive_sha256=prebuilt_archive_sha,
        expected_prebuilt_run_manifest_sha256=prebuilt_manifest_sha,
        expected_slurm_job_script_sha256=_sha(
            source_bytes["cs6_hapg_full_source_cover_slurm_job.sh"]
        ),
        schema_profile="synthetic-self-test",
    )
    return {
        "archive": archive,
        "sidecar": sidecar,
        "sacct": sacct,
        "expectations": expectations,
        "adaptive_job_id": "202",
        "adaptive_submit_utc": "2026-08-02T00:02:00",
    }


def run_self_test() -> int:
    with tempfile.TemporaryDirectory(prefix="cs6-hapg-kat-anchor-selftest.") as directory:
        root = Path(directory)
        fixture = _build_self_test_fixture(root)
        certificate = certify_kat_anchor(
            archive_path=fixture["archive"],
            sidecar_path=fixture["sidecar"],
            sacct_bytes=fixture["sacct"],
            adaptive_job_id=str(fixture["adaptive_job_id"]),
            adaptive_submit_utc=str(fixture["adaptive_submit_utc"]),
            expectations=fixture["expectations"],
        )
        parsed = parse_kat_anchor_certificate(certificate.as_bytes())
        if parsed.as_bytes() != certificate.as_bytes() or parsed.sha256 != certificate.sha256:
            _fail("KAT anchor self-test certificate round trip failed")
        parsed_fields = parsed.as_dict()
        if (
            parsed_fields["CERTIFICATE_SCOPE"] != "SYNTHETIC_SELF_TEST_ONLY"
            or parsed_fields["KAT_PREREQUISITE_VALID"] != "false"
            or parsed_fields["KAT_LEAF_EVIDENCE_VALID"] != "false"
            or parsed_fields["KAT_HPG_VERIFIER_REPLAY_COUNT"] != "0"
            or parsed_fields["KAT_HAPG_VERIFIER_REPLAY_COUNT"] != "0"
            or parsed_fields["KAT_ALLOC_TASKS"] != "1"
        ):
            _fail("synthetic KAT fixture escaped into an authoritative certificate")
        materialized = root / "materialized-science"
        _materialize_science(_read_archive(Path(fixture["archive"])), materialized)
        if not (
            materialized / "cs6_affine_projective_cocycle_full53_probe.cpp"
        ).is_file():
            _fail("KAT anchor self-test omitted the adjacent H-APG kernel")
        expected_args = (
            "--jobs",
            "101",
            "--allocations",
            "--noheader",
            "--parsable",
            f"--format={SACCT_FORMAT}",
            "--starttime",
            "1970-01-01",
        )
        fake_sacct = root / "fake-sacct"
        fake_sacct.write_text(
            "#!/usr/bin/env bash\nset -eu\n"
            f"expected=({' '.join(shlex.quote(token) for token in expected_args)})\n"
            "[[ \"$#\" -eq \"${#expected[@]}\" ]] || exit 9\n"
            "for token in \"${expected[@]}\"; do\n"
            "  [[ \"$1\" == \"$token\" ]] || exit 9\n"
            "  shift\n"
            "done\n"
            f"printf '%s' '{fixture['sacct'].decode('ascii')}'\n",
            encoding="ascii",
        )
        fake_sacct.chmod(0o755)
        if query_live_sacct("101", sacct_bin=str(fake_sacct)) != fixture["sacct"]:
            _fail("KAT anchor live sacct canonicalization self-test failed")
    return 2


def _expect_rejected(action) -> bool:
    try:
        action()
    except VerificationError:
        return True
    return False


def run_mutation_self_test() -> tuple[int, int]:
    tests = 0
    rejected = 0

    def check(action) -> None:
        nonlocal tests, rejected
        tests += 1
        rejected += int(_expect_rejected(action))

    with tempfile.TemporaryDirectory(prefix="cs6-hapg-kat-anchor-mutations.") as directory:
        root = Path(directory)
        fixture = _build_self_test_fixture(root / "valid")
        archive = fixture["archive"]
        sidecar = fixture["sidecar"]
        sacct = fixture["sacct"]
        expectations = fixture["expectations"]
        assert isinstance(expectations, KatAnchorExpectations)

        def certify(**overrides):
            values = {
                "archive_path": archive,
                "sidecar_path": sidecar,
                "sacct_bytes": sacct,
                "adaptive_job_id": fixture["adaptive_job_id"],
                "adaptive_submit_utc": fixture["adaptive_submit_utc"],
                "expectations": expectations,
            }
            values.update(overrides)
            return certify_kat_anchor(**values)

        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "kat_archive_sha256": "0" * 64,
                    }
                )
            )
        )
        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "expected_git_head": "0" * 40,
                    }
                )
            )
        )
        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "expected_contract_sha256": "1" * 64,
                    }
                )
            )
        )
        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "expected_repo_delta_bundle_sha256": "1" * 64,
                    }
                )
            )
        )
        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "expected_prebuilt_archive_sha256": "1" * 64,
                    }
                )
            )
        )
        check(
            lambda: certify(
                expectations=KatAnchorExpectations(
                    **{
                        **expectations.__dict__,
                        "expected_slurm_job_script_sha256": "1" * 64,
                    }
                )
            )
        )
        bad_sidecar = root / "bad.sha256"
        bad_sidecar.write_bytes(b"0" * 64 + b"  wrong.tar\n")
        check(lambda: certify(sidecar_path=bad_sidecar))
        check(lambda: certify(sacct_bytes=sacct.replace(b"COMPLETED", b"FAILED")))
        check(lambda: certify(sacct_bytes=sacct.replace(b"0:0", b"1:0")))
        check(
            lambda: certify(
                sacct_bytes=sacct.replace(
                    b"gpuorangefs-r770-proxmox", b"gpuorangefs-other"
                )
            )
        )
        check(lambda: certify(adaptive_submit_utc="2026-08-01T23:59:59"))
        check(lambda: certify(adaptive_job_id=expectations.kat_job_id))

        check(
            lambda: _parse_index(
                b"2" * 64 + b"  nested/z\n" + b"1" * 64 + b"  adjacent.txt\n",
                "mutated index",
            )
        )
        valid_certificate = certify()
        certificate_raw = valid_certificate.as_bytes()
        check(lambda: parse_kat_anchor_certificate(certificate_raw.rstrip(b"\n")))
        check(lambda: parse_kat_anchor_certificate(certificate_raw.replace(b"\n", b"\r\n")))
        check(
            lambda: parse_kat_anchor_certificate(
                certificate_raw + b"EXTRA_FIELD=false\n"
            )
        )
        first_line = certificate_raw.splitlines(keepends=True)[0]
        check(lambda: parse_kat_anchor_certificate(first_line + certificate_raw))

        for name, kind in (
            ("../escape", "file"),
            ("result/link", "symlink"),
            ("duplicate", "duplicate"),
        ):
            mutated = root / f"unsafe-{kind}.tar"
            with tarfile.open(mutated, "w") as output:
                if kind == "duplicate":
                    for payload in (b"one", b"two"):
                        member = tarfile.TarInfo(name)
                        member.size = len(payload)
                        output.addfile(member, io.BytesIO(payload))
                elif kind == "symlink":
                    member = tarfile.TarInfo(name)
                    member.type = tarfile.SYMTYPE
                    member.linkname = "/etc/passwd"
                    output.addfile(member)
                else:
                    payload = b"escape"
                    member = tarfile.TarInfo(name)
                    member.size = len(payload)
                    output.addfile(member, io.BytesIO(payload))
            check(lambda mutated=mutated: _read_archive(mutated))

        opaque = root / "opaque-trailer.tar"
        opaque.write_bytes(archive.read_bytes() + b"UNINDEXED-OPAQUE-PAYLOAD".ljust(512, b"\0"))
        check(lambda: _read_archive(opaque))

        envelope_files = {
            **{f"result/{name}": b"" for name in KAT_RESULT_DIRECT_FILES},
            **{
                f"result/prebuilt-origin/{name}": b""
                for name in PREBUILT_DIRECT_FILES
            },
        }
        exact_envelope = _ArchiveView(
            b"", envelope_files, frozenset(), len(envelope_files)
        )
        _validate_v6_direct_file_sets(exact_envelope)
        forged_files = {**envelope_files, "result/unfrozen-extra.txt": b"opaque"}
        forged_envelope = _ArchiveView(
            b"", forged_files, frozenset(), len(forged_files)
        )
        check(lambda: _validate_v6_direct_file_sets(forged_envelope))

    if tests != rejected:
        _fail(f"KAT anchor mutation self-test escaped: {rejected}/{tests}")
    return tests, rejected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--sidecar", type=Path)
    parser.add_argument("--sacct-file", type=Path)
    parser.add_argument("--live-sacct", action="store_true")
    parser.add_argument("--kat-job-id")
    parser.add_argument("--kat-archive-sha256")
    parser.add_argument("--expected-git-head")
    parser.add_argument("--expected-contract-sha256")
    parser.add_argument("--expected-base-repo-bundle-sha256")
    parser.add_argument("--expected-base-git-head")
    parser.add_argument("--expected-repo-delta-bundle-sha256")
    parser.add_argument("--expected-prebuilt-archive-sha256")
    parser.add_argument("--expected-prebuilt-run-manifest-sha256")
    parser.add_argument("--expected-slurm-job-script-sha256")
    parser.add_argument("--adaptive-job-id")
    parser.add_argument("--adaptive-submit-utc")
    parser.add_argument(
        "--schema-profile",
        choices=("v6", "historical-v5-fixture"),
        default="v6",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--self-test-mutations", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test or args.self_test_mutations:
        if any(
            value is not None
            for value in (
                args.archive,
                args.sidecar,
                args.sacct_file,
                args.kat_job_id,
                args.output,
            )
        ) or args.live_sacct:
            _fail("self-tests cannot be combined with an archive verification")
        if args.self_test:
            print(f"KAT_ANCHOR_SELF_TEST={run_self_test()}/2")
        if args.self_test_mutations:
            tests, rejected = run_mutation_self_test()
            print(f"KAT_ANCHOR_MUTATIONS={rejected}/{tests}")
        return 0

    required = (
        args.archive,
        args.kat_job_id,
        args.kat_archive_sha256,
        args.expected_git_head,
        args.expected_contract_sha256,
        args.expected_base_repo_bundle_sha256,
        args.expected_base_git_head,
        args.expected_repo_delta_bundle_sha256,
        args.expected_prebuilt_archive_sha256,
        args.expected_prebuilt_run_manifest_sha256,
        args.expected_slurm_job_script_sha256,
        args.adaptive_job_id,
        args.adaptive_submit_utc,
    )
    if any(value is None for value in required):
        _fail("archive verification is missing a required anchor")
    if args.live_sacct == (args.sacct_file is not None):
        _fail("choose exactly one of --live-sacct or --sacct-file")
    assert args.archive is not None
    assert args.kat_job_id is not None
    assert args.kat_archive_sha256 is not None
    if args.live_sacct:
        sacct = query_live_sacct(args.kat_job_id)
    else:
        assert args.sacct_file is not None
        sacct = _stable_bytes(args.sacct_file, "KAT sacct input")
    expectations = KatAnchorExpectations(
        kat_job_id=args.kat_job_id,
        kat_archive_sha256=args.kat_archive_sha256,
        expected_git_head=str(args.expected_git_head),
        expected_contract_sha256=str(args.expected_contract_sha256),
        expected_base_repo_bundle_sha256=str(args.expected_base_repo_bundle_sha256),
        expected_base_git_head=str(args.expected_base_git_head),
        expected_repo_delta_bundle_sha256=str(args.expected_repo_delta_bundle_sha256),
        expected_prebuilt_archive_sha256=str(args.expected_prebuilt_archive_sha256),
        expected_prebuilt_run_manifest_sha256=str(
            args.expected_prebuilt_run_manifest_sha256
        ),
        expected_slurm_job_script_sha256=str(
            args.expected_slurm_job_script_sha256
        ),
        schema_profile=args.schema_profile,
    )
    certificate = certify_kat_anchor(
        archive_path=args.archive,
        sidecar_path=args.sidecar,
        sacct_bytes=sacct,
        adaptive_job_id=str(args.adaptive_job_id),
        adaptive_submit_utc=str(args.adaptive_submit_utc),
        expectations=expectations,
    )
    raw = certificate.as_bytes()
    if args.output is not None:
        if args.output.exists() or args.output.is_symlink():
            _fail("certificate output already exists")
        descriptor = os.open(
            args.output,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o444,
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    sys.stdout.buffer.write(raw)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as error:
        print(f"KAT anchor error: {error}", file=sys.stderr)
        raise SystemExit(1)
