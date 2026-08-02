#!/usr/bin/env python3
"""Execute the frozen 3 x 3 V7-A.1 Liouville checkpoint matrix."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence


FROZEN_CONTRACT_SHA256 = "3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce"
COORDINATE_MANIFEST_SHA256 = "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7"
ROOT_CHALLENGE = "ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
INTERVAL_RE = re.compile(r"^\[([^,]+),([^\]]+)\]$")
CELL_DOMAIN = b"sounio.cs6.hapg-liouville-checkpoint-cell.v1\0"
ATTEMPT_DOMAIN = b"sounio.cs6.hapg-liouville-checkpoint-attempt.v1\0"
INPUT_SCHEMA = "sounio.cs6.c1-full-source-cover-leaf-input.v1"
CARRIERS = ("C0HOTripletonSet", "C0HORect2Set", "C0Rect2Set")
BASELINE = CARRIERS[0]
ALTERNATIVES = CARRIERS[1:]
EXPECTED_MUTATIONS_PER_CHECKPOINT = 46
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
ZERO_SHA256 = "0" * 64

CONTRACT_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_contract_v1.txt")
COORDINATES_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_coordinates_v1.tsv")
SOURCE_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_probe.cpp")
VERIFIER_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_verify.py")
RUNNER_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_run.py")
INTERVAL_VERIFIER_REL = Path("scripts/research/cs6_plucker_cocycle_verify.py")
SLURM_JOB_REL = Path("scripts/research/cs6_hapg_liouville_checkpoint_slurm_job.sh")

AUTHORITATIVE_ALLOCATION = {
    "SLURM_PARTITION": "gpu-orangefs",
    "SLURM_ACCOUNT": "lab",
    "SLURM_QOS": "normal",
    "SLURM_NODELIST": "gpuorangefs-r770-proxmox",
    "SLURM_NODES": "1",
    "SLURM_NTASKS": "1",
    "SLURM_CPUS_PER_TASK": "9",
    "SLURM_JOB_NAME": "cs6-v7a1-checkpoint",
    "SLURM_TIME_LIMIT": "00:20:00",
    "SLURM_MIN_MEMORY_NODE": "8G",
    "SLURM_EXCLUSIVE": "NODE",
}
REPOSITORY_ARCHIVE_FILES = {
    CONTRACT_REL,
    COORDINATES_REL,
    SOURCE_REL,
    VERIFIER_REL,
    RUNNER_REL,
    INTERVAL_VERIFIER_REL,
    SLURM_JOB_REL,
    Path("scripts/research/cs6_hapg_liouville_checkpoint_retained_verify.py"),
}

VERIFICATION_KEYS = (
    "VERIFICATION_SCHEMA",
    "LIOUVILLE_CARRIER",
    "ATTEMPT_BINDING",
    "RECEIPT_SHA256",
    "INITIAL_HULL_SHA256",
    "LIOUVILLE_RECORD_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "ALL_FINITE",
    "SOURCE_TILE_RECONSTRUCTED",
    "INITIAL_HULL_RECONSTRUCTED",
    "EXP_ELL_RECOMPUTED",
    "NORMAL_VELOCITIES_RECOMPUTED",
    "LIOUVILLE_IDENTITY_VERIFIED",
    "SECTION_CONTAINS_ZERO",
    "LIOUVILLE_DET",
    "PARENT_KAT_STATUS",
    "CHECKPOINT_PASS",
    "PROMOTION_ELIGIBLE",
)
RESULT_COLUMNS = (
    "ATTEMPT_INDEX",
    "ORDINAL",
    "CHECKPOINT_ROLE",
    "PARENT_V7_ORDINAL",
    "NODE_ID",
    "LIOUVILLE_CARRIER",
    "STATUS",
    "WORKER_RC",
    "ELAPSED_MS",
    "INPUT_SHA256",
    "MANIFEST_ROW_SHA256",
    "CELL_CHALLENGE",
    "ATTEMPT_BINDING",
    "STDOUT_SHA256",
    "STDERR_SHA256",
    "VERIFICATION_SHA256",
    "INITIAL_HULL_SHA256",
    "LIOUVILLE_RECORD_SHA256",
    "MUTATION_TESTS",
    "MUTATIONS_REJECTED",
    "LIOUVILLE_DET",
    "PARENT_KAT_STATUS",
    "CHECKPOINT_PASS",
)


def die(message: str) -> None:
    raise SystemExit(message)


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def digest(path: Path) -> str:
    return digest_bytes(path.read_bytes())


def bool_token(value: bool) -> str:
    return str(value).lower()


def canonical_kv(path: Path, fields: Sequence[tuple[str, str]]) -> None:
    seen: set[str] = set()
    lines = []
    for key, value in fields:
        if not key or not value or key in seen or "=" in key or "\n" in value:
            raise RuntimeError(f"noncanonical KV field: {key}")
        seen.add(key)
        lines.append(f"{key}={value}\n")
    path.write_bytes("".join(lines).encode("ascii"))


def parse_kv(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError(f"noncanonical KV file: {path}")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError(f"non-ASCII KV file: {path}") from error
    fields: dict[str, str] = {}
    for line in lines:
        if line.count("=") != 1:
            raise RuntimeError(f"malformed KV line: {path}")
        key, value = line.split("=", 1)
        if not key or not value or key in fields:
            raise RuntimeError(f"duplicate or empty KV field: {path}")
        fields[key] = value
    return fields


def parse_verification(raw: bytes) -> dict[str, str]:
    try:
        text = raw.decode("ascii")
    except UnicodeError as error:
        raise RuntimeError("verifier output is not ASCII") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        raise RuntimeError("verifier output is not canonical")
    lines = text.splitlines()
    if len(lines) != len(VERIFICATION_KEYS):
        raise RuntimeError("verifier output line count mismatch")
    fields: dict[str, str] = {}
    for line, expected_key in zip(lines, VERIFICATION_KEYS, strict=True):
        if line.count("=") != 1:
            raise RuntimeError("malformed verifier output")
        key, value = line.split("=", 1)
        if key != expected_key or not value:
            raise RuntimeError(f"verifier output key mismatch: {expected_key}")
        fields[key] = value
    return fields


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


@dataclass(frozen=True)
class Coordinate:
    ordinal: int
    role: str
    parent_ordinal: int
    parent_attempts: str
    node_id: str
    u_depth: int
    u_index: int
    s_depth: int
    s_index: int
    input_sha256: str
    parent_initial_sha256: str | None
    parent_ho_det: str | None
    parent_rect_det: str | None
    row_sha256: str

    def expected_det(self, carrier: str) -> str | None:
        if carrier == "C0HORect2Set":
            return self.parent_ho_det
        if carrier == "C0Rect2Set":
            return self.parent_rect_det
        return None


@dataclass(frozen=True)
class Attempt:
    index: int
    coordinate: Coordinate
    carrier: str
    cell_challenge: str
    binding: str

    @property
    def identity(self) -> str:
        return f"A{self.index:04d}"


@dataclass(frozen=True)
class Result:
    attempt: Attempt
    status: str
    worker_rc: int
    elapsed_ms: int
    stdout_sha256: str
    stderr_sha256: str
    verification_sha256: str = ZERO_SHA256
    initial_hull_sha256: str = ZERO_SHA256
    liouville_record_sha256: str = ZERO_SHA256
    mutation_tests: int = 0
    mutations_rejected: int = 0
    liouville_det: str = "-"
    parent_kat_status: str = "NOT_APPLICABLE"
    checkpoint_pass: bool = False


def parse_coordinates(path: Path) -> list[Coordinate]:
    raw = path.read_bytes()
    if digest_bytes(raw) != COORDINATE_MANIFEST_SHA256:
        raise RuntimeError("coordinate manifest digest drift")
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError("coordinate manifest is noncanonical")
    try:
        lines = raw.decode("ascii").splitlines()
    except UnicodeError as error:
        raise RuntimeError("coordinate manifest is not ASCII") from error
    header = (
        "ORDINAL\tCHECKPOINT_ROLE\tPARENT_V7_ORDINAL\tPARENT_V7_ATTEMPTS\t"
        "NODE_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256\t"
        "PARENT_ALT_INITIAL_HULL_SHA256\tPARENT_HO_RECT2_DET\tPARENT_RECT2_DET"
    )
    try:
        header_index = lines.index(header)
    except ValueError as error:
        raise RuntimeError("coordinate table header missing") from error
    metadata = {}
    for line in lines[:header_index]:
        if line.count("=") != 1:
            raise RuntimeError("malformed coordinate metadata")
        key, value = line.split("=", 1)
        if not key or not value or key in metadata:
            raise RuntimeError("duplicate coordinate metadata")
        metadata[key] = value
    expected_metadata = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-coordinates.v1",
        "CONTRACT_STATE": "PRE_RESULT_FROZEN",
        "DATE": "2026-08-02",
        "SOURCE": "N0",
        "CELL_COUNT": "3",
        "MASKED_TARGET_COUNT": "1",
        "POSITIVE_CONTROL_COUNT": "2",
        "CARRIER_COUNT": "3",
        "MAXIMUM_EVALUATIONS": "9",
        "SELECTION_RULE": "V7A_ORDINAL_23_MASKED_PLUS_IMMEDIATE_V7A_ORDINAL_NEIGHBORS_22_AND_24",
        "CELL_SUBSTITUTION_ALLOWED": "false",
        "ROW_ORDER": "V7A_ORDINAL_22,V7A_ORDINAL_23,V7A_ORDINAL_24",
    }
    if metadata != expected_metadata:
        raise RuntimeError("coordinate metadata differs from frozen schema")

    coordinates: list[Coordinate] = []
    expected_roles = ("POSITIVE_CONTROL_LEFT", "MASKED_TARGET", "POSITIVE_CONTROL_RIGHT")
    expected_parent_ordinals = (22, 23, 24)
    expected_parent_attempts = ("64,65,66", "67,68,69", "70,71,72")
    for raw_line in lines[header_index + 1 :]:
        fields = raw_line.split("\t")
        if len(fields) != 13:
            raise RuntimeError("coordinate row width mismatch")
        (
            ordinal_text,
            role,
            parent_ordinal_text,
            parent_attempts,
            node,
            u_depth_text,
            u_index_text,
            s_depth_text,
            s_index_text,
            input_sha,
            parent_initial,
            parent_ho_det,
            parent_rect_det,
        ) = fields
        numeric = (
            ordinal_text,
            parent_ordinal_text,
            u_depth_text,
            u_index_text,
            s_depth_text,
            s_index_text,
        )
        if any(not token.isdigit() or str(int(token)) != token for token in numeric):
            raise RuntimeError("coordinate integer is noncanonical")
        ordinal = int(ordinal_text)
        parent_ordinal = int(parent_ordinal_text)
        u_depth, u_index, s_depth, s_index = map(
            int, (u_depth_text, u_index_text, s_depth_text, s_index_text)
        )
        if ordinal != len(coordinates) + 1:
            raise RuntimeError("coordinate ordinal drift")
        if role != expected_roles[ordinal - 1]:
            raise RuntimeError("coordinate role drift")
        if parent_ordinal != expected_parent_ordinals[ordinal - 1]:
            raise RuntimeError("parent V7 ordinal drift")
        if parent_attempts != expected_parent_attempts[ordinal - 1]:
            raise RuntimeError("parent V7 attempt lineage drift")
        if node != leaf_id(u_depth, u_index, s_depth, s_index):
            raise RuntimeError("coordinate node identity mismatch")
        reconstructed = leaf_input_bytes(u_depth, u_index, s_depth, s_index)
        if digest_bytes(reconstructed) != input_sha:
            raise RuntimeError("coordinate input digest mismatch")
        if role == "MASKED_TARGET":
            if (parent_initial, parent_ho_det, parent_rect_det) != ("-", "-", "-"):
                raise RuntimeError("masked coordinate unexpectedly has parent KAT")
            initial_value = ho_value = rect_value = None
        else:
            if SHA_RE.fullmatch(parent_initial) is None:
                raise RuntimeError("control initial hull SHA is malformed")
            if INTERVAL_RE.fullmatch(parent_ho_det) is None or INTERVAL_RE.fullmatch(parent_rect_det) is None:
                raise RuntimeError("control determinant KAT is malformed")
            initial_value = parent_initial
            ho_value = parent_ho_det
            rect_value = parent_rect_det
        coordinates.append(
            Coordinate(
                ordinal=ordinal,
                role=role,
                parent_ordinal=parent_ordinal,
                parent_attempts=parent_attempts,
                node_id=node,
                u_depth=u_depth,
                u_index=u_index,
                s_depth=s_depth,
                s_index=s_index,
                input_sha256=input_sha,
                parent_initial_sha256=initial_value,
                parent_ho_det=ho_value,
                parent_rect_det=rect_value,
                row_sha256=digest_bytes((raw_line + "\n").encode("ascii")),
            )
        )
    if len(coordinates) != 3 or len({item.node_id for item in coordinates}) != 3:
        raise RuntimeError("coordinate cardinality or uniqueness mismatch")
    return coordinates


def cell_challenge(
    root: str,
    run_contract_sha256: str,
    coordinate_manifest_sha256: str,
    coordinate: Coordinate,
) -> str:
    return digest_bytes(
        CELL_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
        + b"\0"
        + bytes.fromhex(coordinate_manifest_sha256)
        + b"\0"
        + bytes.fromhex(coordinate.row_sha256)
        + b"\0"
        + bytes.fromhex(coordinate.input_sha256)
    )


def attempt_binding(challenge: str, carrier: str, run_contract_sha256: str) -> str:
    return digest_bytes(
        ATTEMPT_DOMAIN
        + bytes.fromhex(challenge)
        + b"\0"
        + carrier.encode("ascii")
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
    )


def classify_capd_set(stderr: bytes, carrier: str, binding: str) -> bool:
    preamble = (
        f"V7A1_FAILURE_BINDING LIOUVILLE_CARRIER={carrier} "
        f"ATTEMPT_BINDING={binding}\n"
    ).encode("ascii")
    if not stderr.startswith(preamble):
        return False
    lowered = stderr[len(preamble) :].lower()
    return (
        lowered.startswith(
            b"checkpoint worker error: centeredtripletonset::evalaffinefunctional - empty intersection of rb and rq."
        )
        and b"\nrb=[" in lowered
        and lowered.endswith(b"\nrq=[-nan, -nan]\n\n")
    )


def write_attempt_contract(path: Path, attempts: Sequence[Attempt]) -> None:
    columns = (
        "ATTEMPT_INDEX",
        "ORDINAL",
        "CHECKPOINT_ROLE",
        "PARENT_V7_ORDINAL",
        "NODE_ID",
        "U_DEPTH",
        "U_INDEX",
        "S_DEPTH",
        "S_INDEX",
        "INPUT_SHA256",
        "MANIFEST_ROW_SHA256",
        "CELL_CHALLENGE",
        "LIOUVILLE_CARRIER",
        "ATTEMPT_BINDING",
    )
    rows = ["\t".join(columns)]
    for attempt in attempts:
        coordinate = attempt.coordinate
        rows.append(
            "\t".join(
                (
                    str(attempt.index),
                    str(coordinate.ordinal),
                    coordinate.role,
                    str(coordinate.parent_ordinal),
                    coordinate.node_id,
                    str(coordinate.u_depth),
                    str(coordinate.u_index),
                    str(coordinate.s_depth),
                    str(coordinate.s_index),
                    coordinate.input_sha256,
                    coordinate.row_sha256,
                    attempt.cell_challenge,
                    attempt.carrier,
                    attempt.binding,
                )
            )
        )
    path.write_bytes(("\n".join(rows) + "\n").encode("ascii"))


def negative_receipt(path: Path, result: Result) -> None:
    attempt = result.attempt
    canonical_kv(
        path,
        (
            ("SCHEMA", "sounio.cs6.hapg-liouville-checkpoint-negative.v1"),
            ("ATTEMPT_INDEX", str(attempt.index)),
            ("NODE_ID", attempt.coordinate.node_id),
            ("CHECKPOINT_ROLE", attempt.coordinate.role),
            ("LIOUVILLE_CARRIER", attempt.carrier),
            ("INPUT_SHA256", attempt.coordinate.input_sha256),
            ("CELL_CHALLENGE", attempt.cell_challenge),
            ("MANIFEST_ROW_SHA256", attempt.coordinate.row_sha256),
            ("ATTEMPT_BINDING", attempt.binding),
            ("WORKER_RC", str(result.worker_rc)),
            ("STDOUT_SHA256", result.stdout_sha256),
            ("STDERR_SHA256", result.stderr_sha256),
            ("CLASS", result.status),
            ("FAILURE_BINDING_AUTHENTICATED", bool_token(result.status == "CAPD_SET_RQ_NAN")),
            ("SCIENTIFIC_NEGATIVE", bool_token(result.status == "CAPD_SET_RQ_NAN")),
        ),
    )


def write_results(path: Path, results: Sequence[Result]) -> None:
    rows = ["\t".join(RESULT_COLUMNS)]
    for result in results:
        attempt = result.attempt
        coordinate = attempt.coordinate
        values = {
            "ATTEMPT_INDEX": str(attempt.index),
            "ORDINAL": str(coordinate.ordinal),
            "CHECKPOINT_ROLE": coordinate.role,
            "PARENT_V7_ORDINAL": str(coordinate.parent_ordinal),
            "NODE_ID": coordinate.node_id,
            "LIOUVILLE_CARRIER": attempt.carrier,
            "STATUS": result.status,
            "WORKER_RC": str(result.worker_rc),
            "ELAPSED_MS": str(result.elapsed_ms),
            "INPUT_SHA256": coordinate.input_sha256,
            "MANIFEST_ROW_SHA256": coordinate.row_sha256,
            "CELL_CHALLENGE": attempt.cell_challenge,
            "ATTEMPT_BINDING": attempt.binding,
            "STDOUT_SHA256": result.stdout_sha256,
            "STDERR_SHA256": result.stderr_sha256,
            "VERIFICATION_SHA256": result.verification_sha256,
            "INITIAL_HULL_SHA256": result.initial_hull_sha256,
            "LIOUVILLE_RECORD_SHA256": result.liouville_record_sha256,
            "MUTATION_TESTS": str(result.mutation_tests),
            "MUTATIONS_REJECTED": str(result.mutations_rejected),
            "LIOUVILLE_DET": result.liouville_det,
            "PARENT_KAT_STATUS": result.parent_kat_status,
            "CHECKPOINT_PASS": bool_token(result.checkpoint_pass),
        }
        rows.append("\t".join(values[column] for column in RESULT_COLUMNS))
    path.write_bytes(("\n".join(rows) + "\n").encode("ascii"))


def content_index(root: Path) -> bytes:
    excluded = {"files.sha256", "manifest.txt"}
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise RuntimeError("result tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise RuntimeError("result tree contains a non-regular node")
        if relative in excluded:
            continue
        rows.append(f"{digest(path)}  {relative}\n")
    return "".join(rows).encode("ascii")


def validate_repository_archive(path: Path, repo: Path, expected_head: str) -> None:
    expected_names = {item.as_posix() for item in REPOSITORY_ARCHIVE_FILES}
    try:
        with tarfile.open(path, "r:") as handle:
            if handle.pax_headers.get("comment") != expected_head:
                raise RuntimeError("repository archive commit differs")
            members = handle.getmembers()
            regular = {member.name for member in members if member.isfile()}
            if regular != expected_names or any(
                not member.isfile() and not member.isdir() for member in members
            ):
                raise RuntimeError("repository archive member set differs")
            for member in members:
                if not member.isfile():
                    continue
                source = handle.extractfile(member)
                if source is None or source.read() != (repo / member.name).read_bytes():
                    raise RuntimeError("repository archive bytes differ from staged source")
    except tarfile.TarError as error:
        raise RuntimeError("repository archive is malformed") from error


def parse_scontrol(path: Path) -> dict[str, str]:
    raw = path.read_bytes()
    if not raw.endswith(b"\n") or b"\r" in raw or b"\0" in raw:
        raise RuntimeError("noncanonical retained scontrol record")
    try:
        text = raw.decode("ascii").strip()
        tokens = shlex.split(text)
    except (UnicodeError, ValueError) as error:
        raise RuntimeError("malformed retained scontrol record") from error
    fields: dict[str, str] = {}
    for token in tokens:
        if token.count("=") != 1:
            continue
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            raise RuntimeError("duplicate or empty retained scontrol field")
        fields[key] = value
    return fields


def validate_execution_provenance(
    provenance: Path,
    repo: Path,
    head: str,
    jobs: int,
    timeout_seconds: int,
    allow_synthetic_gate: bool,
) -> dict[str, str]:
    config = parse_kv(provenance / "slurm-config.txt")
    expected_config_keys = {
        "SCHEMA",
        "EXECUTION_CLASS",
        "REPO_ARCHIVE",
        "REPO_ARCHIVE_SHA256",
        "GIT_HEAD",
        "PREBUILT_ARCHIVE",
        "PREBUILT_ARCHIVE_SHA256",
        "OUTPUT_ARCHIVE",
        "ROOT_CHALLENGE",
        "FROZEN_CONTRACT_SHA256",
        "COORDINATE_MANIFEST_SHA256",
        "JOB_SCRIPT_SHA256",
        "SUBMITTED_JOB_SCRIPT",
        "JOBS",
        "TIMEOUT_SECONDS",
        *AUTHORITATIVE_ALLOCATION.keys(),
    }
    if set(config) != expected_config_keys:
        raise RuntimeError("retained Slurm config schema differs")
    execution_class = config["EXECUTION_CLASS"]
    if execution_class == "AUTHORITATIVE_SLURM":
        if any(config[key] != value for key, value in AUTHORITATIVE_ALLOCATION.items()):
            raise RuntimeError("authoritative Slurm allocation differs from frozen execution")
    elif execution_class == "SYNTHETIC_GATE":
        if not allow_synthetic_gate:
            raise RuntimeError("synthetic transport provenance is not authoritative")
    else:
        raise RuntimeError("unknown execution class")
    exact_config = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-slurm-config.v2",
        "GIT_HEAD": head,
        "ROOT_CHALLENGE": ROOT_CHALLENGE,
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "JOBS": str(jobs),
        "TIMEOUT_SECONDS": str(timeout_seconds),
    }
    if any(config.get(key) != value for key, value in exact_config.items()):
        raise RuntimeError("retained Slurm config binding differs")

    repository_archive = provenance / "repo-source.tar"
    repository_archive_sha = digest(repository_archive)
    if (
        config["REPO_ARCHIVE_SHA256"] != repository_archive_sha
        or (provenance / "repo-source.sha256").read_text(encoding="ascii")
        != repository_archive_sha + "\n"
    ):
        raise RuntimeError("repository archive digest binding differs")
    validate_repository_archive(repository_archive, repo, head)

    script_path = provenance / "slurm-job-script.sh"
    config_path = provenance / "slurm-config.txt"
    control_path = provenance / "slurm-control-plane.txt"
    script_sha = digest(script_path)
    config_sha = digest(config_path)
    control_sha = digest(control_path)
    for name, value in (
        ("slurm-job-script.sha256", script_sha),
        ("slurm-config.sha256", config_sha),
        ("slurm-control-plane.sha256", control_sha),
    ):
        if (provenance / name).read_text(encoding="ascii") != value + "\n":
            raise RuntimeError(f"retained provenance digest mismatch: {name}")
    if config["JOB_SCRIPT_SHA256"] != script_sha:
        raise RuntimeError("Slurm config does not bind the executed job script")
    if script_path.read_bytes() != (repo / SLURM_JOB_REL).read_bytes():
        raise RuntimeError("executed and repository Slurm job scripts differ")

    context = parse_kv(provenance / "slurm-context.txt")
    exact_context_keys = {
        "SCHEMA",
        "EXECUTION_CLASS",
        "SLURM_JOB_ID",
        "SLURM_JOB_NODELIST",
        "SLURMD_NODENAME",
        "EXECUTION_HOST",
        "EXECUTION_UID",
        "SLURM_JOB_NUM_NODES",
        "SLURM_NTASKS",
        "SLURM_CPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "SLURM_RESTART_COUNT",
        "SLURM_EXPORT_ENV",
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_QOS",
        "SLURM_JOB_NAME",
        "SLURM_TIME_LIMIT",
        "SLURM_MIN_MEMORY_NODE",
        "SLURM_EXCLUSIVE",
        "SLURM_COMMAND",
        "SCONTROL_PATH",
        "SCONTROL_VERSION",
    }
    if set(context) != exact_context_keys:
        raise RuntimeError("retained Slurm context schema differs")
    if (
        context["SCHEMA"] != "sounio.cs6.hapg-liouville-checkpoint-slurm-context.v2"
        or context["EXECUTION_CLASS"] != execution_class
        or not context["SLURM_JOB_ID"].isdigit()
        or not context["EXECUTION_UID"].isdigit()
        or int(context["EXECUTION_UID"]) != os.getuid()
        or context["SLURM_JOB_NODELIST"] != config["SLURM_NODELIST"]
        or context["SLURMD_NODENAME"] != config["SLURM_NODELIST"]
        or context["EXECUTION_HOST"] != config["SLURM_NODELIST"]
        or context["SLURM_JOB_NUM_NODES"] != config["SLURM_NODES"]
        or context["SLURM_NTASKS"] != config["SLURM_NTASKS"]
        or context["SLURM_CPUS_PER_TASK"] != config["SLURM_CPUS_PER_TASK"]
        or context["SLURM_RESTART_COUNT"] != "0"
        or context["SLURM_EXPORT_ENV"] != "NIL"
        or context["SLURM_PARTITION"] != config["SLURM_PARTITION"]
        or context["SLURM_ACCOUNT"] != config["SLURM_ACCOUNT"]
        or context["SLURM_QOS"] != config["SLURM_QOS"]
        or context["SLURM_JOB_NAME"] != config["SLURM_JOB_NAME"]
        or context["SLURM_TIME_LIMIT"] != config["SLURM_TIME_LIMIT"]
        or context["SLURM_MIN_MEMORY_NODE"] != config["SLURM_MIN_MEMORY_NODE"]
        or context["SLURM_EXCLUSIVE"] != config["SLURM_EXCLUSIVE"]
        or context["SLURM_COMMAND"] != config["SUBMITTED_JOB_SCRIPT"]
        or not context["SLURM_COMMAND"].startswith("/")
        or (
            execution_class == "AUTHORITATIVE_SLURM"
            and context["SCONTROL_PATH"] != "/usr/bin/scontrol"
        )
        or not context["SCONTROL_VERSION"].strip()
        or not context["SLURM_CPUS_ON_NODE"].isdigit()
        or int(context["SLURM_CPUS_ON_NODE"]) < jobs
    ):
        raise RuntimeError("retained Slurm context binding differs")

    control = parse_scontrol(control_path)
    expected_control = {
        "JobId": context["SLURM_JOB_ID"],
        "JobState": "RUNNING",
        "Partition": config["SLURM_PARTITION"],
        "Account": config["SLURM_ACCOUNT"],
        "QOS": config["SLURM_QOS"],
        "NodeList": config["SLURM_NODELIST"],
        "BatchHost": config["SLURM_NODELIST"],
        "NumNodes": config["SLURM_NODES"],
        "NumTasks": config["SLURM_NTASKS"],
        "CPUs/Task": config["SLURM_CPUS_PER_TASK"],
        "Requeue": "0",
        "Restarts": "0",
        "Command": context["SLURM_COMMAND"],
        "JobName": config["SLURM_JOB_NAME"],
        "TimeLimit": config["SLURM_TIME_LIMIT"],
        "MinMemoryNode": config["SLURM_MIN_MEMORY_NODE"],
        "OverSubscribe": "NO",
    }
    user = re.fullmatch(r"[^()]+\(([0-9]+)\)", control.get("UserId", ""))
    if (
        any(control.get(key) != value for key, value in expected_control.items())
        or user is None
        or user.group(1) != context["EXECUTION_UID"]
        or control.get("NumCPUs") != context["SLURM_CPUS_ON_NODE"]
    ):
        raise RuntimeError("retained Slurm control-plane record differs")
    return {
        "EXECUTION_CLASS": execution_class,
        "EXECUTION_PATH": (
            "SLURM_CPU_PREBUILT_NODE_LOCAL_TMP"
            if execution_class == "AUTHORITATIVE_SLURM"
            else "SYNTHETIC_LOCAL_TRANSPORT_GATE"
        ),
        "SLURM_JOB_ID": context["SLURM_JOB_ID"],
        "SLURM_NODE": config["SLURM_NODELIST"],
        "SLURM_PARTITION": config["SLURM_PARTITION"],
        "SLURM_ACCOUNT": config["SLURM_ACCOUNT"],
        "SLURM_QOS": config["SLURM_QOS"],
        "SLURM_JOB_NAME": config["SLURM_JOB_NAME"],
        "SLURM_TIME_LIMIT": config["SLURM_TIME_LIMIT"],
        "SLURM_MIN_MEMORY_NODE": config["SLURM_MIN_MEMORY_NODE"],
        "SLURM_EXCLUSIVE": config["SLURM_EXCLUSIVE"],
        "SLURM_COMMAND": context["SLURM_COMMAND"],
        "SLURM_EXPORT_ENV": context["SLURM_EXPORT_ENV"],
        "SLURM_CONTEXT_SHA256": digest(provenance / "slurm-context.txt"),
        "SCONTROL_JOB_SHA256": control_sha,
        "JOB_SCRIPT_SHA256": script_sha,
        "CONFIG_SHA256": config_sha,
        "REPO_ARCHIVE_SHA256": repository_archive_sha,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--provenance-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-challenge", required=True)
    parser.add_argument("--jobs", type=int, default=9)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--self-test-mutations", action="store_true")
    parser.add_argument("--keep-failed", action="store_true")
    parser.add_argument("--allow-synthetic-gate", action="store_true")
    args = parser.parse_args(argv)
    if args.root_challenge != ROOT_CHALLENGE:
        die("root challenge differs from frozen V7-A.1 contract")
    if not (1 <= args.jobs <= 9) or not (1 <= args.timeout_seconds <= 3600):
        die("jobs or timeout is outside the frozen execution envelope")
    if not args.self_test_mutations:
        die("mutation self-tests are mandatory for V7-A.1")

    if args.worker.is_symlink():
        die("worker path must not be a symlink")
    repo = args.repo.resolve()
    worker = args.worker.resolve()
    provenance = args.provenance_dir.resolve()
    run_dir = args.run_dir.resolve()
    if not repo.is_dir() or not worker.is_file() or not os.access(worker, os.X_OK):
        die("repo or worker path is invalid")
    required_provenance = (
        "capd-cflags.txt",
        "capd-libs.txt",
        "capd-version.txt",
        "compile-command.txt",
        "compile-stderr.txt",
        "compile-stdout.txt",
        "compiler-version.txt",
        "dependencies.sha256",
        "runtime-libraries.sha256",
        "runtime-linkage.txt",
        "worker-source.cpp",
        "worker-source.sha256",
        "worker-binary",
        "worker-binary.sha256",
        "repo-source.tar",
        "repo-source.sha256",
        "slurm-context.txt",
        "slurm-config.txt",
        "slurm-config.sha256",
        "slurm-job-script.sh",
        "slurm-job-script.sha256",
        "slurm-control-plane.txt",
        "slurm-control-plane.sha256",
        "node-uname.txt",
        "node-lscpu.txt",
        "node-runtime-linkage.txt",
        "node-runtime-libraries.sha256",
    )
    if any(not (provenance / name).is_file() or (provenance / name).is_symlink() for name in required_provenance):
        die("prebuilt provenance directory is incomplete")

    contract_path = repo / CONTRACT_REL
    coordinates_path = repo / COORDINATES_REL
    source_path = repo / SOURCE_REL
    verifier_path = repo / VERIFIER_REL
    runner_path = repo / RUNNER_REL
    interval_verifier_path = repo / INTERVAL_VERIFIER_REL
    for path in (
        contract_path,
        coordinates_path,
        source_path,
        verifier_path,
        runner_path,
        interval_verifier_path,
    ):
        if not path.is_file():
            die(f"required implementation file missing: {path}")
    if digest(contract_path) != FROZEN_CONTRACT_SHA256:
        die("frozen contract digest drift")
    if digest(coordinates_path) != COORDINATE_MANIFEST_SHA256:
        die("coordinate manifest digest drift")
    contract = parse_kv(contract_path)
    if contract.get("ROOT_CHALLENGE") != args.root_challenge:
        die("root challenge differs from contract bytes")
    coordinates = parse_coordinates(coordinates_path)
    if (provenance / "capd-version.txt").read_text(encoding="ascii") != "5.3.0\n":
        die("prebuilt CAPD source version is not frozen 5.3.0")
    cflags = (provenance / "capd-cflags.txt").read_text(encoding="ascii")
    if "-D__USE_FILIB__" not in cflags or "-frounding-math" not in cflags:
        die("prebuilt CAPD flags do not bind FILIB outward rounding")
    source_sha = digest(source_path)
    worker_sha = digest(worker)
    if (provenance / "worker-source.cpp").read_bytes() != source_path.read_bytes():
        die("prebuilt source snapshot differs from repository source")
    if (provenance / "worker-source.sha256").read_text(encoding="ascii") != source_sha + "\n":
        die("prebuilt source digest mismatch")
    if (provenance / "worker-binary.sha256").read_text(encoding="ascii") != worker_sha + "\n":
        die("prebuilt worker digest mismatch")
    if digest(provenance / "worker-binary") != worker_sha:
        die("prebuilt worker snapshot differs from executable")
    head = parse_kv(provenance / "slurm-config.txt").get("GIT_HEAD", "")
    if re.fullmatch(r"[0-9a-f]{40}", head) is None:
        die("execution Git commit is malformed")
    execution = validate_execution_provenance(
        provenance,
        repo,
        head,
        args.jobs,
        args.timeout_seconds,
        args.allow_synthetic_gate,
    )

    if run_dir.exists():
        die("run directory already exists")
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".cs6-v7a1-checkpoint.", dir=run_dir.parent))
    complete = False
    try:
        for directory in (
            "inputs",
            "receipts",
            "negative-receipts",
            "stderr",
            "verifications",
            "verifier-stderr",
        ):
            (work / directory).mkdir()
        retained_provenance = work / "provenance"
        retained_provenance.mkdir()
        for name in required_provenance:
            shutil.copyfile(provenance / name, retained_provenance / name)
        retained_worker = retained_provenance / "worker-binary"
        retained_worker.chmod(0o500)
        if digest(retained_worker) != worker_sha:
            raise RuntimeError("private worker snapshot digest mismatch")
        snapshot_dir = work / "verifier-snapshot"
        snapshot_dir.mkdir()
        shutil.copyfile(verifier_path, snapshot_dir / verifier_path.name)
        shutil.copyfile(interval_verifier_path, snapshot_dir / interval_verifier_path.name)

        started_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        run_contract_fields = (
            ("SCHEMA", "sounio.cs6.hapg-liouville-checkpoint-run-contract.v1"),
            ("FROZEN_CONTRACT_SHA256", FROZEN_CONTRACT_SHA256),
            ("COORDINATE_MANIFEST_SHA256", COORDINATE_MANIFEST_SHA256),
            ("ROOT_CHALLENGE", args.root_challenge),
            ("GIT_HEAD", head),
            ("WORKER_SOURCE_SHA256", source_sha),
            ("WORKER_BINARY_SHA256", worker_sha),
            ("VERIFIER_SHA256", digest(verifier_path)),
            ("RUNNER_SHA256", digest(runner_path)),
            ("INTERVAL_VERIFIER_SHA256", digest(interval_verifier_path)),
            ("CAPD_VERSION", "5.3.0"),
            ("INTERVAL_BACKEND", "FILIB"),
            ("OPTIMIZATION_LEVEL", "O0"),
            ("EXECUTION_CLASS", execution["EXECUTION_CLASS"]),
            ("EXECUTION_PATH", execution["EXECUTION_PATH"]),
            ("SLURM_JOB_ID", execution["SLURM_JOB_ID"]),
            ("SLURM_NODE", execution["SLURM_NODE"]),
            ("SLURM_PARTITION", execution["SLURM_PARTITION"]),
            ("SLURM_ACCOUNT", execution["SLURM_ACCOUNT"]),
            ("SLURM_QOS", execution["SLURM_QOS"]),
            ("SLURM_JOB_NAME", execution["SLURM_JOB_NAME"]),
            ("SLURM_TIME_LIMIT", execution["SLURM_TIME_LIMIT"]),
            ("SLURM_MIN_MEMORY_NODE", execution["SLURM_MIN_MEMORY_NODE"]),
            ("SLURM_EXCLUSIVE", execution["SLURM_EXCLUSIVE"]),
            ("SLURM_COMMAND", execution["SLURM_COMMAND"]),
            ("SLURM_EXPORT_ENV", execution["SLURM_EXPORT_ENV"]),
            ("SLURM_CONTEXT_SHA256", execution["SLURM_CONTEXT_SHA256"]),
            ("SCONTROL_JOB_SHA256", execution["SCONTROL_JOB_SHA256"]),
            ("JOB_SCRIPT_SHA256", execution["JOB_SCRIPT_SHA256"]),
            ("CONFIG_SHA256", execution["CONFIG_SHA256"]),
            ("REPO_ARCHIVE_SHA256", execution["REPO_ARCHIVE_SHA256"]),
            ("JOBS", str(args.jobs)),
            ("TIMEOUT_SECONDS", str(args.timeout_seconds)),
            ("MUTATION_SELF_TESTS", "true"),
            ("ATTEMPT_COUNT", "9"),
            ("UTC_STARTED", started_utc),
            ("FPGA_EXECUTION", "false"),
            ("PROMOTION_ELIGIBLE", "false"),
        )
        canonical_kv(work / "run-contract.txt", run_contract_fields)
        run_contract_sha = digest(work / "run-contract.txt")

        attempts: list[Attempt] = []
        for coordinate in coordinates:
            challenge = cell_challenge(
                args.root_challenge,
                run_contract_sha,
                COORDINATE_MANIFEST_SHA256,
                coordinate,
            )
            for carrier in CARRIERS:
                attempts.append(
                    Attempt(
                        index=len(attempts) + 1,
                        coordinate=coordinate,
                        carrier=carrier,
                        cell_challenge=challenge,
                        binding=attempt_binding(challenge, carrier, run_contract_sha),
                    )
                )
        if len(attempts) != 9:
            raise RuntimeError("attempt matrix cardinality mismatch")
        write_attempt_contract(work / "attempt-contract.tsv", attempts)
        shutil.copyfile(contract_path, work / "frozen-contract.txt")
        shutil.copyfile(coordinates_path, work / "coordinates.tsv")
        for coordinate in coordinates:
            raw_input = leaf_input_bytes(
                coordinate.u_depth,
                coordinate.u_index,
                coordinate.s_depth,
                coordinate.s_index,
            )
            if digest_bytes(raw_input) != coordinate.input_sha256:
                raise RuntimeError("input reconstruction drift")
            (work / "inputs" / f"{coordinate.node_id}.txt").write_bytes(raw_input)

        python = Path(sys.executable).resolve()
        verifier_snapshot = snapshot_dir / verifier_path.name

        def run_attempt(attempt: Attempt) -> Result:
            coordinate = attempt.coordinate
            identity = attempt.identity
            receipt_path = work / "receipts" / f"{identity}.txt"
            stderr_path = work / "stderr" / f"{identity}.txt"
            verification_path = work / "verifications" / f"{identity}.txt"
            verifier_stderr_path = work / "verifier-stderr" / f"{identity}.txt"
            command = [
                str(retained_worker),
                str(coordinate.u_depth),
                str(coordinate.u_index),
                str(coordinate.s_depth),
                str(coordinate.s_index),
                coordinate.input_sha256,
                attempt.cell_challenge,
                attempt.carrier,
                FROZEN_CONTRACT_SHA256,
                COORDINATE_MANIFEST_SHA256,
                run_contract_sha,
                coordinate.row_sha256,
                attempt.binding,
            ]
            started = time.monotonic_ns()
            try:
                completed = subprocess.run(
                    command, capture_output=True, timeout=args.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                elapsed = (time.monotonic_ns() - started) // 1_000_000
                stdout = error.stdout or b""
                stderr = error.stderr or b""
                receipt_path.write_bytes(stdout)
                stderr_path.write_bytes(stderr)
                return Result(
                    attempt=attempt,
                    status="TIMEOUT",
                    worker_rc=124,
                    elapsed_ms=elapsed,
                    stdout_sha256=digest_bytes(stdout),
                    stderr_sha256=digest_bytes(stderr),
                )
            elapsed = (time.monotonic_ns() - started) // 1_000_000
            stdout = completed.stdout
            stderr = completed.stderr
            receipt_path.write_bytes(stdout)
            stderr_path.write_bytes(stderr)
            base = Result(
                attempt=attempt,
                status="UNKNOWN_FAILURE",
                worker_rc=completed.returncode,
                elapsed_ms=elapsed,
                stdout_sha256=digest_bytes(stdout),
                stderr_sha256=digest_bytes(stderr),
            )
            if completed.returncode != 0:
                status = (
                    "CAPD_SET_RQ_NAN"
                    if completed.returncode == 1
                    and not stdout
                    and classify_capd_set(stderr, attempt.carrier, attempt.binding)
                    else "UNKNOWN_FAILURE"
                )
                result = replace(base, status=status)
                negative_receipt(work / "negative-receipts" / f"{identity}.txt", result)
                return result
            if stderr or not stdout:
                return base

            verifier_command = [
                str(python),
                "-I",
                "-B",
                str(verifier_snapshot),
                str(receipt_path),
                "--source-sha",
                source_sha,
                "--input",
                str(work / "inputs" / f"{coordinate.node_id}.txt"),
                "--challenge",
                attempt.cell_challenge,
                "--carrier",
                attempt.carrier,
                "--frozen-contract-sha",
                FROZEN_CONTRACT_SHA256,
                "--coordinate-manifest-sha",
                COORDINATE_MANIFEST_SHA256,
                "--run-contract-sha",
                run_contract_sha,
                "--manifest-row-sha",
                coordinate.row_sha256,
                "--attempt-binding",
                attempt.binding,
                "--self-test-mutations",
            ]
            expected_det = coordinate.expected_det(attempt.carrier)
            if expected_det is not None:
                if coordinate.parent_initial_sha256 is None:
                    raise RuntimeError("control determinant lacks initial-hull KAT")
                verifier_command.extend(
                    [
                        "--expected-initial-sha",
                        coordinate.parent_initial_sha256,
                        "--expected-det",
                        expected_det,
                    ]
                )
            try:
                verification = subprocess.run(
                    verifier_command, capture_output=True, timeout=args.timeout_seconds
                )
            except subprocess.TimeoutExpired as error:
                verification_path.write_bytes(error.stdout or b"")
                verifier_stderr_path.write_bytes(error.stderr or b"")
                return replace(base, status="VERIFIER_FAILURE")
            verification_path.write_bytes(verification.stdout)
            verifier_stderr_path.write_bytes(verification.stderr)
            if verification.returncode != 0 or verification.stderr:
                return replace(
                    base,
                    status="VERIFIER_FAILURE",
                    verification_sha256=digest_bytes(verification.stdout),
                )
            fields = parse_verification(verification.stdout)
            if fields["VERIFICATION_SCHEMA"] != "sounio.cs6.hapg-liouville-checkpoint-verification.v1":
                raise RuntimeError("verification schema mismatch")
            if fields["LIOUVILLE_CARRIER"] != attempt.carrier:
                raise RuntimeError("verified carrier mismatch")
            if fields["ATTEMPT_BINDING"] != attempt.binding:
                raise RuntimeError("verified binding mismatch")
            if fields["RECEIPT_SHA256"] != digest_bytes(stdout):
                raise RuntimeError("verified receipt digest mismatch")
            required_true = (
                "ALL_FINITE",
                "SOURCE_TILE_RECONSTRUCTED",
                "INITIAL_HULL_RECONSTRUCTED",
                "EXP_ELL_RECOMPUTED",
                "NORMAL_VELOCITIES_RECOMPUTED",
                "LIOUVILLE_IDENTITY_VERIFIED",
                "SECTION_CONTAINS_ZERO",
                "CHECKPOINT_PASS",
            )
            if any(fields[key] != "true" for key in required_true):
                raise RuntimeError("verifier did not assert every checkpoint invariant")
            if fields["PROMOTION_ELIGIBLE"] != "false":
                raise RuntimeError("verifier enabled promotion")
            mutation_tests = int(fields["MUTATION_TESTS"])
            mutations_rejected = int(fields["MUTATIONS_REJECTED"])
            if (
                mutation_tests != EXPECTED_MUTATIONS_PER_CHECKPOINT
                or mutation_tests != mutations_rejected
            ):
                raise RuntimeError("mutation accounting mismatch")
            expected_kat_status = "PASS" if expected_det is not None else "NOT_APPLICABLE"
            if fields["PARENT_KAT_STATUS"] != expected_kat_status:
                raise RuntimeError("parent KAT status mismatch")
            return Result(
                attempt=attempt,
                status="VERIFIED_CHECKPOINT",
                worker_rc=0,
                elapsed_ms=elapsed,
                stdout_sha256=digest_bytes(stdout),
                stderr_sha256=EMPTY_SHA256,
                verification_sha256=digest_bytes(verification.stdout),
                initial_hull_sha256=fields["INITIAL_HULL_SHA256"],
                liouville_record_sha256=fields["LIOUVILLE_RECORD_SHA256"],
                mutation_tests=mutation_tests,
                mutations_rejected=mutations_rejected,
                liouville_det=fields["LIOUVILLE_DET"],
                parent_kat_status=fields["PARENT_KAT_STATUS"],
                checkpoint_pass=True,
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_attempt, attempt) for attempt in attempts]
            results = [future.result() for future in futures]
        results.sort(key=lambda result: result.attempt.index)
        if [result.attempt.index for result in results] != list(range(1, 10)):
            raise RuntimeError("result matrix order or completeness mismatch")

        baseline_results = [result for result in results if result.attempt.carrier == BASELINE]
        baseline_valid = len(baseline_results) == 3 and all(
            result.status == "CAPD_SET_RQ_NAN" for result in baseline_results
        )
        control_results = [
            result
            for result in results
            if result.attempt.carrier in ALTERNATIVES
            and result.attempt.coordinate.role != "MASKED_TARGET"
        ]
        controls_valid = len(control_results) == 4 and all(
            result.status == "VERIFIED_CHECKPOINT"
            and result.parent_kat_status == "PASS"
            for result in control_results
        )
        masked_results = {
            result.attempt.carrier: result
            for result in results
            if result.attempt.coordinate.role == "MASKED_TARGET"
            and result.attempt.carrier in ALTERNATIVES
        }
        masked_status_valid = set(masked_results) == set(ALTERNATIVES) and all(
            result.status in {"VERIFIED_CHECKPOINT", "CAPD_SET_RQ_NAN"}
            for result in masked_results.values()
        )
        initial_invariance = True
        for coordinate in coordinates:
            successful = [
                result
                for result in results
                if result.attempt.coordinate.ordinal == coordinate.ordinal
                and result.attempt.carrier in ALTERNATIVES
                and result.status == "VERIFIED_CHECKPOINT"
            ]
            if len(successful) == 2 and len({item.initial_hull_sha256 for item in successful}) != 1:
                initial_invariance = False
        run_valid = baseline_valid and controls_valid and masked_status_valid and initial_invariance
        ho_pass = (
            ALTERNATIVES[0] in masked_results
            and masked_results[ALTERNATIVES[0]].status == "VERIFIED_CHECKPOINT"
        )
        rect_pass = (
            ALTERNATIVES[1] in masked_results
            and masked_results[ALTERNATIVES[1]].status == "VERIFIED_CHECKPOINT"
        )
        if ho_pass and rect_pass:
            outcome = "BOTH_ALTERNATIVES_PASS"
        elif ho_pass:
            outcome = "ONLY_HO_RECT2_PASSES"
        elif rect_pass:
            outcome = "ONLY_RECT2_PASSES"
        else:
            outcome = "BOTH_DECLARED_FAIL" if run_valid else "RUN_INVALID"
        if not run_valid:
            outcome = "RUN_INVALID"

        decisions = [
            "LIOUVILLE_CARRIER\tCONTROL_CHECKPOINTS\tMASKED_STATUS\tDECISION",
            f"{BASELINE}\t0\t{','.join(result.status for result in baseline_results)}\t"
            + ("BASELINE_VALID" if baseline_valid else "RUN_INVALID"),
        ]
        for carrier in ALTERNATIVES:
            controls = [
                result
                for result in control_results
                if result.attempt.carrier == carrier
                and result.status == "VERIFIED_CHECKPOINT"
            ]
            masked = masked_results.get(carrier)
            masked_status = masked.status if masked is not None else "MISSING"
            if not run_valid:
                decision = "RUN_INVALID"
            elif masked_status == "VERIFIED_CHECKPOINT":
                decision = "MASKED_CHECKPOINT_PASS"
            else:
                decision = "MASKED_CHECKPOINT_FAIL_DECLARED_RQ_NAN"
            decisions.append(f"{carrier}\t{len(controls)}\t{masked_status}\t{decision}")
        (work / "decisions.tsv").write_bytes(("\n".join(decisions) + "\n").encode("ascii"))
        write_results(work / "results.tsv", results)

        verified_count = sum(result.status == "VERIFIED_CHECKPOINT" for result in results)
        bound_negative_count = sum(result.status == "CAPD_SET_RQ_NAN" for result in results)
        mutation_tests = sum(result.mutation_tests for result in results)
        mutations_rejected = sum(result.mutations_rejected for result in results)
        if digest(worker) != worker_sha or digest(retained_worker) != worker_sha:
            raise RuntimeError("worker bytes changed during the attempt matrix")
        canonical_kv(
            work / "summary.txt",
            (
                ("SCHEMA", "sounio.cs6.hapg-liouville-checkpoint-summary.v1"),
                ("RUN_COMPLETE", "true"),
                ("RUN_VALID", bool_token(run_valid)),
                ("ATTEMPTS_COMPLETED", "9"),
                ("VERIFIED_CHECKPOINTS", str(verified_count)),
                ("BOUND_RQ_NAN", str(bound_negative_count)),
                ("BASELINE_PREREQUISITE_VALID", bool_token(baseline_valid)),
                ("POSITIVE_CONTROL_KATS_VALID", bool_token(controls_valid)),
                ("MASKED_STATUS_VALID", bool_token(masked_status_valid)),
                ("INITIAL_HULL_INVARIANCE", bool_token(initial_invariance)),
                ("OUTCOME", outcome),
                ("MUTATION_TESTS", str(mutation_tests)),
                ("MUTATIONS_REJECTED", str(mutations_rejected)),
                ("V7A_RETROACTIVE_REINTERPRETATION_ALLOWED", "false"),
                ("C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED", "false"),
                ("FULL_HPG_PIPELINE_EVALUATED", "false"),
                ("V7_B_ELIGIBILITY", "false"),
                ("V7_B_WINNER", "NONE"),
                ("PROMOTION_ELIGIBLE", "false"),
                ("OPEN_PROBLEM_SOLVED", "false"),
                ("FPGA_EXECUTION", "false"),
            ),
        )
        index = content_index(work)
        (work / "files.sha256").write_bytes(index)
        canonical_kv(
            work / "manifest.txt",
            (
                ("SCHEMA", "sounio.cs6.hapg-liouville-checkpoint-manifest.v1"),
                ("RUN_CONTRACT_SHA256", run_contract_sha),
                ("FROZEN_CONTRACT_SHA256", FROZEN_CONTRACT_SHA256),
                ("COORDINATE_MANIFEST_SHA256", COORDINATE_MANIFEST_SHA256),
                ("ROOT_CHALLENGE", args.root_challenge),
                ("GIT_HEAD", head),
                ("ATTEMPT_COUNT", "9"),
                ("VERIFIED_CHECKPOINTS", str(verified_count)),
                ("BOUND_RQ_NAN", str(bound_negative_count)),
                ("RUN_COMPLETE", "true"),
                ("RUN_VALID", bool_token(run_valid)),
                ("OUTCOME", outcome),
                ("MUTATION_TESTS", str(mutation_tests)),
                ("MUTATIONS_REJECTED", str(mutations_rejected)),
                ("FILES_SHA256", digest(work / "files.sha256")),
                ("V7_B_WINNER", "NONE"),
                ("PROMOTION_ELIGIBLE", "false"),
            ),
        )
        os.replace(work, run_dir)
        complete = True
        print(f"RUN_DIR={run_dir}")
        print("RUN_COMPLETE=true")
        print(f"RUN_VALID={bool_token(run_valid)}")
        print(f"OUTCOME={outcome}")
        print("ATTEMPTS_COMPLETED=9")
        print(f"VERIFIED_CHECKPOINTS={verified_count}")
        print(f"BOUND_RQ_NAN={bound_negative_count}")
        print(f"MUTATION_TESTS={mutation_tests}")
        print(f"MUTATIONS_REJECTED={mutations_rejected}")
        print("V7_B_WINNER=NONE")
        print("PROMOTION_ELIGIBLE=false")
        return 0 if run_valid else 2
    except Exception as error:
        print(f"V7-A.1 runner error: {error}", file=sys.stderr)
        if args.keep_failed:
            print(f"FAILED_WORK_DIR={work}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        if not complete and not args.keep_failed:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
