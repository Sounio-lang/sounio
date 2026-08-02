#!/usr/bin/env python3
"""Independently replay a retained V7-A.1 Liouville checkpoint result."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Mapping, Sequence


FROZEN_CONTRACT_SHA256 = "3afc0475847ad8054234a2ddfa108b768cfd81991d0be71fc21c991f363631ce"
COORDINATE_MANIFEST_SHA256 = "527afc7c205fcf09b15a0bff91df6935f19ed2b7e7926895916ac5da33a992a7"
ROOT_CHALLENGE = "ad536f25d02182c8b2add12ef1a7e8a8a18b4eb0d12e68535ea94ddb3eff0bdf"
SHA_RE = re.compile(r"^[0-9a-f]{64}$")
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

ATTEMPT_COLUMNS = (
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


def canonical_uint(token: str, label: str) -> int:
    if not token.isdigit() or str(int(token)) != token:
        fail(f"noncanonical integer: {label}")
    return int(token)


def parse_bool(token: str, label: str) -> bool:
    if token not in {"true", "false"}:
        fail(f"noncanonical boolean: {label}")
    return token == "true"


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


def leaf_input_bytes(u_depth: int, u_index: int, s_depth: int, s_index: int) -> bytes:
    return (
        f"SCHEMA={INPUT_SCHEMA}\n"
        "SOURCE=N0\n"
        f"U_DEPTH={u_depth}\n"
        f"U_INDEX={u_index}\n"
        f"S_DEPTH={s_depth}\n"
        f"S_INDEX={s_index}\n"
    ).encode("ascii")


def coordinate_rows(path: Path) -> list[dict[str, str]]:
    raw = stable_bytes(path, "coordinate manifest")
    if digest_bytes(raw) != COORDINATE_MANIFEST_SHA256:
        fail("coordinate manifest digest drift")
    text = raw.decode("ascii")
    header = (
        "ORDINAL\tCHECKPOINT_ROLE\tPARENT_V7_ORDINAL\tPARENT_V7_ATTEMPTS\t"
        "NODE_ID\tU_DEPTH\tU_INDEX\tS_DEPTH\tS_INDEX\tPARENT_INPUT_SHA256\t"
        "PARENT_ALT_INITIAL_HULL_SHA256\tPARENT_HO_RECT2_DET\tPARENT_RECT2_DET"
    )
    lines = text.splitlines()
    if header not in lines:
        fail("coordinate header missing")
    header_index = lines.index(header)
    reader = csv.DictReader(lines[header_index:], delimiter="\t")
    rows = list(reader)
    if len(rows) != 3:
        fail("coordinate cardinality mismatch")
    for ordinal, row in enumerate(rows, 1):
        if canonical_uint(row["ORDINAL"], "coordinate ordinal") != ordinal:
            fail("coordinate ordinal drift")
        numbers = [
            canonical_uint(row[key], f"coordinate {key}")
            for key in ("U_DEPTH", "U_INDEX", "S_DEPTH", "S_INDEX")
        ]
        raw_input = leaf_input_bytes(*numbers)
        if digest_bytes(raw_input) != row["PARENT_INPUT_SHA256"]:
            fail("coordinate input digest mismatch")
        raw_line = "\t".join(row[key] for key in reader.fieldnames or ()) + "\n"
        row["ROW_SHA256"] = digest_bytes(raw_line.encode("ascii"))
    return rows


def cell_challenge(
    root: str,
    run_contract_sha256: str,
    coordinate_manifest_sha256: str,
    row: Mapping[str, str],
) -> str:
    return digest_bytes(
        CELL_DOMAIN
        + bytes.fromhex(root)
        + b"\0"
        + bytes.fromhex(run_contract_sha256)
        + b"\0"
        + bytes.fromhex(coordinate_manifest_sha256)
        + b"\0"
        + bytes.fromhex(row["ROW_SHA256"])
        + b"\0"
        + bytes.fromhex(row["PARENT_INPUT_SHA256"])
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


def expected_attempts(
    coordinates: Sequence[Mapping[str, str]], run_contract_sha: str
) -> list[dict[str, str]]:
    attempts = []
    for coordinate in coordinates:
        challenge = cell_challenge(
            ROOT_CHALLENGE,
            run_contract_sha,
            COORDINATE_MANIFEST_SHA256,
            coordinate,
        )
        for carrier in CARRIERS:
            attempts.append(
                {
                    "ATTEMPT_INDEX": str(len(attempts) + 1),
                    "ORDINAL": coordinate["ORDINAL"],
                    "CHECKPOINT_ROLE": coordinate["CHECKPOINT_ROLE"],
                    "PARENT_V7_ORDINAL": coordinate["PARENT_V7_ORDINAL"],
                    "NODE_ID": coordinate["NODE_ID"],
                    "U_DEPTH": coordinate["U_DEPTH"],
                    "U_INDEX": coordinate["U_INDEX"],
                    "S_DEPTH": coordinate["S_DEPTH"],
                    "S_INDEX": coordinate["S_INDEX"],
                    "INPUT_SHA256": coordinate["PARENT_INPUT_SHA256"],
                    "MANIFEST_ROW_SHA256": coordinate["ROW_SHA256"],
                    "CELL_CHALLENGE": challenge,
                    "LIOUVILLE_CARRIER": carrier,
                    "ATTEMPT_BINDING": attempt_binding(challenge, carrier, run_contract_sha),
                    "PARENT_INITIAL_SHA256": coordinate["PARENT_ALT_INITIAL_HULL_SHA256"],
                    "PARENT_HO_DET": coordinate["PARENT_HO_RECT2_DET"],
                    "PARENT_RECT_DET": coordinate["PARENT_RECT2_DET"],
                }
            )
    return attempts


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


def content_index(root: Path) -> bytes:
    excluded = {"files.sha256", "manifest.txt"}
    rows = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            fail("result tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            fail("result tree contains a non-regular node")
        if relative in excluded:
            continue
        rows.append(f"{digest(path)}  {relative}\n")
    return "".join(rows).encode("ascii")


def parse_scontrol(path: Path) -> dict[str, str]:
    raw = stable_bytes(path, "Slurm control-plane record")
    try:
        text = raw.decode("ascii")
        tokens = shlex.split(text.strip())
    except (UnicodeError, ValueError) as error:
        raise AuditError("malformed Slurm control-plane record") from error
    if not text.endswith("\n") or "\r" in text or "\0" in text:
        fail("Slurm control-plane record is not canonical")
    fields: dict[str, str] = {}
    for token in tokens:
        if token.count("=") != 1:
            continue
        key, value = token.split("=", 1)
        if not key or not value or key in fields:
            fail("duplicate or empty Slurm control-plane field")
        fields[key] = value
    return fields


def git_state(repo: Path) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        capture_output=True,
    )
    value = head.stdout.strip()
    if (
        head.returncode != 0
        or status.returncode != 0
        or re.fullmatch(r"[0-9a-f]{40}", value) is None
    ):
        fail("cannot resolve retained-audit Git state")
    return value, not status.stdout


def validate_execution_provenance(
    result: Path,
    repo: Path,
    run_contract: Mapping[str, str],
    allow_synthetic_gate: bool,
) -> None:
    provenance = result / "provenance"
    config = parse_kv(provenance / "slurm-config.txt", "Slurm config")
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
        fail("retained Slurm config schema differs")
    execution_class = config["EXECUTION_CLASS"]
    if execution_class == "AUTHORITATIVE_SLURM":
        if any(config[key] != value for key, value in AUTHORITATIVE_ALLOCATION.items()):
            fail("authoritative Slurm allocation differs from frozen execution")
    elif execution_class != "SYNTHETIC_GATE" or not allow_synthetic_gate:
        fail("synthetic transport provenance is not authoritative")
    if (
        config["SCHEMA"] != "sounio.cs6.hapg-liouville-checkpoint-slurm-config.v2"
        or config["GIT_HEAD"] != run_contract["GIT_HEAD"]
        or config["ROOT_CHALLENGE"] != ROOT_CHALLENGE
        or config["FROZEN_CONTRACT_SHA256"] != FROZEN_CONTRACT_SHA256
        or config["COORDINATE_MANIFEST_SHA256"] != COORDINATE_MANIFEST_SHA256
        or config["JOBS"] != run_contract["JOBS"]
        or config["TIMEOUT_SECONDS"] != run_contract["TIMEOUT_SECONDS"]
    ):
        fail("retained Slurm config binding differs")

    repository_archive = provenance / "repo-source.tar"
    repository_archive_sha = digest(repository_archive)
    if (
        config["REPO_ARCHIVE_SHA256"] != repository_archive_sha
        or stable_bytes(provenance / "repo-source.sha256", "repository archive hash")
        != (repository_archive_sha + "\n").encode("ascii")
        or run_contract.get("REPO_ARCHIVE_SHA256") != repository_archive_sha
    ):
        fail("repository archive digest binding differs")
    expected_names = {item.as_posix() for item in REPOSITORY_ARCHIVE_FILES}
    try:
        with tarfile.open(repository_archive, "r:") as handle:
            if handle.pax_headers.get("comment") != run_contract["GIT_HEAD"]:
                fail("repository archive commit differs")
            members = handle.getmembers()
            regular = {member.name for member in members if member.isfile()}
            if regular != expected_names or any(
                not member.isfile() and not member.isdir() for member in members
            ):
                fail("repository archive member set differs")
            for member in members:
                if not member.isfile():
                    continue
                source = handle.extractfile(member)
                if source is None or source.read() != stable_bytes(
                    repo / member.name, f"repository source {member.name}"
                ):
                    fail("repository archive bytes differ from audit source")
    except tarfile.TarError as error:
        raise AuditError("repository archive is malformed") from error

    script = provenance / "slurm-job-script.sh"
    control_path = provenance / "slurm-control-plane.txt"
    digests = {
        "JOB_SCRIPT_SHA256": digest(script),
        "CONFIG_SHA256": digest(provenance / "slurm-config.txt"),
        "SCONTROL_JOB_SHA256": digest(control_path),
    }
    for filename, key in (
        ("slurm-job-script.sha256", "JOB_SCRIPT_SHA256"),
        ("slurm-config.sha256", "CONFIG_SHA256"),
        ("slurm-control-plane.sha256", "SCONTROL_JOB_SHA256"),
    ):
        if stable_bytes(provenance / filename, filename).decode("ascii") != digests[key] + "\n":
            fail(f"retained provenance digest mismatch: {filename}")
    if (
        config["JOB_SCRIPT_SHA256"] != digests["JOB_SCRIPT_SHA256"]
        or any(run_contract.get(key) != value for key, value in digests.items())
        or stable_bytes(script, "executed Slurm job script")
        != stable_bytes(repo / SLURM_JOB_REL, "repository Slurm job script")
    ):
        fail("executed Slurm job-script binding differs")

    context = parse_kv(provenance / "slurm-context.txt", "Slurm context")
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
        fail("retained Slurm context schema differs")
    context_sha = digest(provenance / "slurm-context.txt")
    if (
        context["SCHEMA"] != "sounio.cs6.hapg-liouville-checkpoint-slurm-context.v2"
        or context["EXECUTION_CLASS"] != execution_class
        or run_contract.get("EXECUTION_CLASS") != execution_class
        or run_contract.get("EXECUTION_PATH")
        != (
            "SLURM_CPU_PREBUILT_NODE_LOCAL_TMP"
            if execution_class == "AUTHORITATIVE_SLURM"
            else "SYNTHETIC_LOCAL_TRANSPORT_GATE"
        )
        or run_contract.get("SLURM_CONTEXT_SHA256") != context_sha
        or not context["SLURM_JOB_ID"].isdigit()
        or not context["EXECUTION_UID"].isdigit()
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
        or int(context["SLURM_CPUS_ON_NODE"]) < int(config["JOBS"])
    ):
        fail("retained Slurm context binding differs")

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
        or run_contract.get("SLURM_JOB_ID") != context["SLURM_JOB_ID"]
        or run_contract.get("SLURM_NODE") != config["SLURM_NODELIST"]
        or run_contract.get("SLURM_PARTITION") != config["SLURM_PARTITION"]
        or run_contract.get("SLURM_ACCOUNT") != config["SLURM_ACCOUNT"]
        or run_contract.get("SLURM_QOS") != config["SLURM_QOS"]
        or run_contract.get("SLURM_JOB_NAME") != config["SLURM_JOB_NAME"]
        or run_contract.get("SLURM_TIME_LIMIT") != config["SLURM_TIME_LIMIT"]
        or run_contract.get("SLURM_MIN_MEMORY_NODE") != config["SLURM_MIN_MEMORY_NODE"]
        or run_contract.get("SLURM_EXCLUSIVE") != config["SLURM_EXCLUSIVE"]
        or run_contract.get("SLURM_COMMAND") != context["SLURM_COMMAND"]
        or run_contract.get("SLURM_EXPORT_ENV") != "NIL"
    ):
        fail("retained Slurm control-plane record differs")


def expected_control_values(attempt: Mapping[str, str]) -> tuple[str, str] | None:
    if attempt["CHECKPOINT_ROLE"] == "MASKED_TARGET" or attempt["LIOUVILLE_CARRIER"] == BASELINE:
        return None
    determinant = (
        attempt["PARENT_HO_DET"]
        if attempt["LIOUVILLE_CARRIER"] == "C0HORect2Set"
        else attempt["PARENT_RECT_DET"]
    )
    return attempt["PARENT_INITIAL_SHA256"], determinant


def negative_receipt_expected(
    attempt: Mapping[str, str], result: Mapping[str, str]
) -> dict[str, str]:
    return {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-negative.v1",
        "ATTEMPT_INDEX": attempt["ATTEMPT_INDEX"],
        "NODE_ID": attempt["NODE_ID"],
        "CHECKPOINT_ROLE": attempt["CHECKPOINT_ROLE"],
        "LIOUVILLE_CARRIER": attempt["LIOUVILLE_CARRIER"],
        "INPUT_SHA256": attempt["INPUT_SHA256"],
        "CELL_CHALLENGE": attempt["CELL_CHALLENGE"],
        "MANIFEST_ROW_SHA256": attempt["MANIFEST_ROW_SHA256"],
        "ATTEMPT_BINDING": attempt["ATTEMPT_BINDING"],
        "WORKER_RC": result["WORKER_RC"],
        "STDOUT_SHA256": result["STDOUT_SHA256"],
        "STDERR_SHA256": result["STDERR_SHA256"],
        "CLASS": result["STATUS"],
        "FAILURE_BINDING_AUTHENTICATED": str(result["STATUS"] == "CAPD_SET_RQ_NAN").lower(),
        "SCIENTIFIC_NEGATIVE": str(result["STATUS"] == "CAPD_SET_RQ_NAN").lower(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--allow-synthetic-gate", action="store_true")
    parser.add_argument("--allow-transport-archive", action="store_true")
    args = parser.parse_args(argv)
    if args.repo.is_symlink() or args.result_dir.is_symlink():
        fail("repo and result paths must not be symlinks")
    repo = args.repo.resolve()
    result = args.result_dir.resolve()
    if not repo.is_dir() or not result.is_dir():
        fail("repo or result path is invalid")

    contract_path = repo / CONTRACT_REL
    coordinate_path = repo / COORDINATES_REL
    source_path = repo / SOURCE_REL
    verifier_path = repo / VERIFIER_REL
    runner_path = repo / RUNNER_REL
    interval_verifier_path = repo / INTERVAL_VERIFIER_REL
    if digest(contract_path) != FROZEN_CONTRACT_SHA256:
        fail("frozen contract digest drift")
    if digest(coordinate_path) != COORDINATE_MANIFEST_SHA256:
        fail("coordinate manifest digest drift")
    contract = parse_kv(contract_path, "frozen contract")
    if contract.get("ROOT_CHALLENGE") != ROOT_CHALLENGE:
        fail("root challenge differs from frozen contract")
    coordinates = coordinate_rows(coordinate_path)

    if stable_bytes(result / "frozen-contract.txt", "retained contract") != stable_bytes(contract_path, "repo contract"):
        fail("retained contract differs from repo")
    if stable_bytes(result / "coordinates.tsv", "retained coordinates") != stable_bytes(coordinate_path, "repo coordinates"):
        fail("retained coordinates differ from repo")
    run_contract = parse_kv(result / "run-contract.txt", "run contract")
    run_contract_sha = digest(result / "run-contract.txt")
    worker = result / "provenance" / "worker-binary"
    if worker.is_symlink() or not worker.is_file() or not os.access(worker, os.X_OK):
        fail("retained worker is not a regular executable")
    exact_run = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-run-contract.v1",
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "ROOT_CHALLENGE": ROOT_CHALLENGE,
        "WORKER_SOURCE_SHA256": digest(source_path),
        "WORKER_BINARY_SHA256": digest(worker),
        "VERIFIER_SHA256": digest(verifier_path),
        "RUNNER_SHA256": digest(runner_path),
        "INTERVAL_VERIFIER_SHA256": digest(interval_verifier_path),
        "CAPD_VERSION": "5.3.0",
        "INTERVAL_BACKEND": "FILIB",
        "OPTIMIZATION_LEVEL": "O0",
        "MUTATION_SELF_TESTS": "true",
        "ATTEMPT_COUNT": "9",
        "FPGA_EXECUTION": "false",
        "PROMOTION_ELIGIBLE": "false",
    }
    for key, value in exact_run.items():
        if run_contract.get(key) != value:
            fail(f"run contract mismatch: {key}")
    if re.fullmatch(r"[0-9a-f]{40}", run_contract.get("GIT_HEAD", "")) is None:
        fail("run contract Git commit is malformed")
    jobs = canonical_uint(run_contract.get("JOBS", ""), "jobs")
    timeout = canonical_uint(run_contract.get("TIMEOUT_SECONDS", ""), "timeout")
    if not 1 <= jobs <= 9 or not 1 <= timeout <= 3600:
        fail("run contract execution bounds differ")
    if re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", run_contract.get("UTC_STARTED", "")) is None:
        fail("run contract UTC timestamp is malformed")
    expected_run_keys = set(exact_run) | {
        "GIT_HEAD",
        "EXECUTION_CLASS",
        "EXECUTION_PATH",
        "SLURM_JOB_ID",
        "SLURM_NODE",
        "SLURM_PARTITION",
        "SLURM_ACCOUNT",
        "SLURM_QOS",
        "SLURM_JOB_NAME",
        "SLURM_TIME_LIMIT",
        "SLURM_MIN_MEMORY_NODE",
        "SLURM_EXCLUSIVE",
        "SLURM_COMMAND",
        "SLURM_EXPORT_ENV",
        "SLURM_CONTEXT_SHA256",
        "SCONTROL_JOB_SHA256",
        "JOB_SCRIPT_SHA256",
        "CONFIG_SHA256",
        "REPO_ARCHIVE_SHA256",
        "JOBS",
        "TIMEOUT_SECONDS",
        "UTC_STARTED",
    }
    if set(run_contract) != expected_run_keys:
        fail("run contract field set differs")
    if (repo / ".git").exists():
        head, clean = git_state(repo)
        if head != run_contract["GIT_HEAD"]:
            fail("audit repository HEAD differs from execution commit")
        if not clean:
            fail("audit repository is dirty")
    elif not args.allow_transport_archive:
        fail("audit source is not a clean Git worktree")
    validate_execution_provenance(result, repo, run_contract, args.allow_synthetic_gate)

    provenance = result / "provenance"
    if stable_bytes(provenance / "worker-source.cpp", "worker source snapshot") != stable_bytes(source_path, "repo worker source"):
        fail("worker source snapshot differs")
    if parse_kv_bytes(
        b"VALUE=" + stable_bytes(provenance / "worker-source.sha256", "source hash"),
        "source hash wrapper",
    )["VALUE"] != digest(source_path):
        fail("worker source provenance hash mismatch")
    if stable_bytes(provenance / "worker-binary.sha256", "worker hash").decode("ascii") != digest(worker) + "\n":
        fail("worker binary provenance hash mismatch")
    if stable_bytes(provenance / "capd-version.txt", "CAPD version") != b"5.3.0\n":
        fail("CAPD source version differs")
    snapshot = result / "verifier-snapshot"
    if stable_bytes(snapshot / verifier_path.name, "verifier snapshot") != stable_bytes(
        verifier_path, "repo verifier"
    ):
        fail("checkpoint verifier snapshot differs")
    if stable_bytes(
        snapshot / interval_verifier_path.name, "interval verifier snapshot"
    ) != stable_bytes(interval_verifier_path, "repo interval verifier"):
        fail("interval verifier snapshot differs")

    attempts = expected_attempts(coordinates, run_contract_sha)
    retained_attempts = table(result / "attempt-contract.tsv", "attempt contract", ATTEMPT_COLUMNS)
    if len(retained_attempts) != 9:
        fail("attempt contract cardinality mismatch")
    for expected, retained in zip(attempts, retained_attempts, strict=True):
        for key in ATTEMPT_COLUMNS:
            if retained[key] != expected[key]:
                fail(f"attempt contract mismatch: {expected['ATTEMPT_INDEX']}.{key}")
        input_path = result / "inputs" / f"{expected['NODE_ID']}.txt"
        raw_input = leaf_input_bytes(
            int(expected["U_DEPTH"]),
            int(expected["U_INDEX"]),
            int(expected["S_DEPTH"]),
            int(expected["S_INDEX"]),
        )
        if stable_bytes(input_path, "leaf input") != raw_input:
            fail("retained leaf input differs")

    rows = table(result / "results.tsv", "results", RESULT_COLUMNS)
    if len(rows) != 9:
        fail("results cardinality mismatch")
    python = Path(sys.executable).resolve()
    replay_count = 0
    worker_replay_count = 0
    bound_negatives = 0
    for expected, row in zip(attempts, rows, strict=True):
        identity = f"A{int(expected['ATTEMPT_INDEX']):04d}"
        for key in (
            "ATTEMPT_INDEX",
            "ORDINAL",
            "CHECKPOINT_ROLE",
            "PARENT_V7_ORDINAL",
            "NODE_ID",
            "LIOUVILLE_CARRIER",
            "INPUT_SHA256",
            "MANIFEST_ROW_SHA256",
            "CELL_CHALLENGE",
            "ATTEMPT_BINDING",
        ):
            if row[key] != expected[key]:
                fail(f"result binding mismatch: {identity}.{key}")
        elapsed = canonical_uint(row["ELAPSED_MS"], f"{identity} elapsed")
        if elapsed > timeout * 1000 + 60_000:
            fail("elapsed time exceeds retained audit allowance")
        receipt_path = result / "receipts" / f"{identity}.txt"
        stderr_path = result / "stderr" / f"{identity}.txt"
        receipt_raw = stable_bytes(receipt_path, f"{identity} receipt")
        stderr_raw = stable_bytes(stderr_path, f"{identity} stderr")
        if digest_bytes(receipt_raw) != row["STDOUT_SHA256"]:
            fail("receipt digest mismatch")
        if digest_bytes(stderr_raw) != row["STDERR_SHA256"]:
            fail("stderr digest mismatch")
        for key in (
            "STDOUT_SHA256",
            "STDERR_SHA256",
            "VERIFICATION_SHA256",
            "INITIAL_HULL_SHA256",
            "LIOUVILLE_RECORD_SHA256",
        ):
            if SHA_RE.fullmatch(row[key]) is None:
                fail(f"malformed result digest: {identity}.{key}")
        status = row["STATUS"]
        worker_rc = canonical_uint(row["WORKER_RC"], f"{identity} worker rc")
        mutations = canonical_uint(row["MUTATION_TESTS"], f"{identity} mutations")
        rejected = canonical_uint(row["MUTATIONS_REJECTED"], f"{identity} rejected")
        checkpoint_pass = parse_bool(row["CHECKPOINT_PASS"], f"{identity} checkpoint")
        worker_command = [
            str(worker),
            expected["U_DEPTH"],
            expected["U_INDEX"],
            expected["S_DEPTH"],
            expected["S_INDEX"],
            expected["INPUT_SHA256"],
            expected["CELL_CHALLENGE"],
            expected["LIOUVILLE_CARRIER"],
            FROZEN_CONTRACT_SHA256,
            COORDINATE_MANIFEST_SHA256,
            run_contract_sha,
            expected["MANIFEST_ROW_SHA256"],
            expected["ATTEMPT_BINDING"],
        ]
        try:
            worker_replay = subprocess.run(
                worker_command, capture_output=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as error:
            raise AuditError(f"worker replay timed out: {identity}") from error
        if (
            worker_replay.returncode != worker_rc
            or worker_replay.stdout != receipt_raw
            or worker_replay.stderr != stderr_raw
        ):
            fail(f"worker replay mismatch: {identity}")
        worker_replay_count += 1
        if status == "VERIFIED_CHECKPOINT":
            if worker_rc != 0 or stderr_raw or not receipt_raw or not checkpoint_pass:
                fail("verified checkpoint has inconsistent worker evidence")
            if (
                mutations != EXPECTED_MUTATIONS_PER_CHECKPOINT
                or mutations != rejected
            ):
                fail("verified checkpoint mutation count mismatch")
            verification_path = result / "verifications" / f"{identity}.txt"
            verifier_stderr_path = result / "verifier-stderr" / f"{identity}.txt"
            retained_verification = stable_bytes(verification_path, "retained verification")
            if digest_bytes(retained_verification) != row["VERIFICATION_SHA256"]:
                fail("verification digest mismatch")
            if stable_bytes(verifier_stderr_path, "verifier stderr"):
                fail("verified checkpoint retained verifier stderr")
            verification_fields = parse_kv_bytes(
                retained_verification, f"{identity} retained verification"
            )
            expected_verification_keys = {
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
            }
            exact_verification = {
                "VERIFICATION_SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-verification.v1",
                "LIOUVILLE_CARRIER": expected["LIOUVILLE_CARRIER"],
                "ATTEMPT_BINDING": expected["ATTEMPT_BINDING"],
                "RECEIPT_SHA256": row["STDOUT_SHA256"],
                "INITIAL_HULL_SHA256": row["INITIAL_HULL_SHA256"],
                "LIOUVILLE_RECORD_SHA256": row["LIOUVILLE_RECORD_SHA256"],
                "MUTATION_TESTS": row["MUTATION_TESTS"],
                "MUTATIONS_REJECTED": row["MUTATIONS_REJECTED"],
                "LIOUVILLE_DET": row["LIOUVILLE_DET"],
                "PARENT_KAT_STATUS": row["PARENT_KAT_STATUS"],
                "CHECKPOINT_PASS": "true",
                "PROMOTION_ELIGIBLE": "false",
            }
            required_true = {
                "ALL_FINITE",
                "SOURCE_TILE_RECONSTRUCTED",
                "INITIAL_HULL_RECONSTRUCTED",
                "EXP_ELL_RECOMPUTED",
                "NORMAL_VELOCITIES_RECOMPUTED",
                "LIOUVILLE_IDENTITY_VERIFIED",
                "SECTION_CONTAINS_ZERO",
            }
            if (
                set(verification_fields) != expected_verification_keys
                or any(verification_fields.get(key) != value for key, value in exact_verification.items())
                or any(verification_fields.get(key) != "true" for key in required_true)
            ):
                fail(f"retained verification fields differ: {identity}")
            command = [
                str(python),
                "-I",
                "-B",
                str(verifier_path),
                str(receipt_path),
                "--source-sha",
                digest(source_path),
                "--input",
                str(result / "inputs" / f"{expected['NODE_ID']}.txt"),
                "--challenge",
                expected["CELL_CHALLENGE"],
                "--carrier",
                expected["LIOUVILLE_CARRIER"],
                "--frozen-contract-sha",
                FROZEN_CONTRACT_SHA256,
                "--coordinate-manifest-sha",
                COORDINATE_MANIFEST_SHA256,
                "--run-contract-sha",
                run_contract_sha,
                "--manifest-row-sha",
                expected["MANIFEST_ROW_SHA256"],
                "--attempt-binding",
                expected["ATTEMPT_BINDING"],
                "--self-test-mutations",
            ]
            control = expected_control_values(expected)
            if control is not None:
                command.extend(
                    ["--expected-initial-sha", control[0], "--expected-det", control[1]]
                )
            try:
                replay = subprocess.run(command, capture_output=True, timeout=timeout)
            except subprocess.TimeoutExpired as error:
                raise AuditError(f"verifier replay timed out: {identity}") from error
            if replay.returncode != 0 or replay.stderr or replay.stdout != retained_verification:
                fail(f"verifier replay mismatch: {identity}")
            replay_count += 1
        elif status == "CAPD_SET_RQ_NAN":
            if (
                worker_rc != 1
                or receipt_raw
                or checkpoint_pass
                or row["STDOUT_SHA256"] != EMPTY_SHA256
                or row["VERIFICATION_SHA256"] != ZERO_SHA256
                or row["INITIAL_HULL_SHA256"] != ZERO_SHA256
                or row["LIOUVILLE_RECORD_SHA256"] != ZERO_SHA256
                or mutations != 0
                or rejected != 0
                or row["LIOUVILLE_DET"] != "-"
                or row["PARENT_KAT_STATUS"] != "NOT_APPLICABLE"
                or not classify_capd_set(
                    stderr_raw, expected["LIOUVILLE_CARRIER"], expected["ATTEMPT_BINDING"]
                )
            ):
                fail("bound CAPD negative is inconsistent")
            negative = parse_kv(
                result / "negative-receipts" / f"{identity}.txt", "negative receipt"
            )
            if negative != negative_receipt_expected(expected, row):
                fail("negative receipt binding mismatch")
            bound_negatives += 1
        elif status in {"TIMEOUT", "UNKNOWN_FAILURE", "VERIFIER_FAILURE"}:
            if checkpoint_pass or mutations != 0 or rejected != 0:
                fail("invalid attempt carries accepted checkpoint fields")
        else:
            fail(f"unknown result status: {status}")

    baseline = [row for row in rows if row["LIOUVILLE_CARRIER"] == BASELINE]
    baseline_valid = len(baseline) == 3 and all(row["STATUS"] == "CAPD_SET_RQ_NAN" for row in baseline)
    controls = [
        row
        for row in rows
        if row["LIOUVILLE_CARRIER"] in ALTERNATIVES
        and row["CHECKPOINT_ROLE"] != "MASKED_TARGET"
    ]
    controls_valid = len(controls) == 4 and all(
        row["STATUS"] == "VERIFIED_CHECKPOINT" and row["PARENT_KAT_STATUS"] == "PASS"
        for row in controls
    )
    masked = {
        row["LIOUVILLE_CARRIER"]: row
        for row in rows
        if row["CHECKPOINT_ROLE"] == "MASKED_TARGET"
        and row["LIOUVILLE_CARRIER"] in ALTERNATIVES
    }
    masked_valid = set(masked) == set(ALTERNATIVES) and all(
        row["STATUS"] in {"VERIFIED_CHECKPOINT", "CAPD_SET_RQ_NAN"}
        for row in masked.values()
    )
    initial_invariance = True
    for ordinal in ("1", "2", "3"):
        successful = [
            row["INITIAL_HULL_SHA256"]
            for row in rows
            if row["ORDINAL"] == ordinal
            and row["LIOUVILLE_CARRIER"] in ALTERNATIVES
            and row["STATUS"] == "VERIFIED_CHECKPOINT"
        ]
        if len(successful) == 2 and len(set(successful)) != 1:
            initial_invariance = False
    run_valid = baseline_valid and controls_valid and masked_valid and initial_invariance
    ho_pass = masked.get(ALTERNATIVES[0], {}).get("STATUS") == "VERIFIED_CHECKPOINT"
    rect_pass = masked.get(ALTERNATIVES[1], {}).get("STATUS") == "VERIFIED_CHECKPOINT"
    if not run_valid:
        outcome = "RUN_INVALID"
    elif ho_pass and rect_pass:
        outcome = "BOTH_ALTERNATIVES_PASS"
    elif ho_pass:
        outcome = "ONLY_HO_RECT2_PASSES"
    elif rect_pass:
        outcome = "ONLY_RECT2_PASSES"
    else:
        outcome = "BOTH_DECLARED_FAIL"
    mutation_tests = sum(int(row["MUTATION_TESTS"]) for row in rows)
    mutations_rejected = sum(int(row["MUTATIONS_REJECTED"]) for row in rows)
    verified_count = sum(row["STATUS"] == "VERIFIED_CHECKPOINT" for row in rows)

    decision_lines = [
        "LIOUVILLE_CARRIER\tCONTROL_CHECKPOINTS\tMASKED_STATUS\tDECISION",
        f"{BASELINE}\t0\t{','.join(row['STATUS'] for row in baseline)}\t"
        + ("BASELINE_VALID" if baseline_valid else "RUN_INVALID"),
    ]
    for carrier in ALTERNATIVES:
        carrier_controls = sum(
            row["LIOUVILLE_CARRIER"] == carrier
            and row["STATUS"] == "VERIFIED_CHECKPOINT"
            for row in controls
        )
        masked_status = masked.get(carrier, {}).get("STATUS", "MISSING")
        if not run_valid:
            decision = "RUN_INVALID"
        elif masked_status == "VERIFIED_CHECKPOINT":
            decision = "MASKED_CHECKPOINT_PASS"
        else:
            decision = "MASKED_CHECKPOINT_FAIL_DECLARED_RQ_NAN"
        decision_lines.append(
            f"{carrier}\t{carrier_controls}\t{masked_status}\t{decision}"
        )
    expected_decisions = ("\n".join(decision_lines) + "\n").encode("ascii")
    if stable_bytes(result / "decisions.tsv", "decisions") != expected_decisions:
        fail("decisions differ from independent reduction")

    summary = parse_kv(result / "summary.txt", "summary")
    expected_summary = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-summary.v1",
        "RUN_COMPLETE": "true",
        "RUN_VALID": str(run_valid).lower(),
        "ATTEMPTS_COMPLETED": "9",
        "VERIFIED_CHECKPOINTS": str(verified_count),
        "BOUND_RQ_NAN": str(bound_negatives),
        "BASELINE_PREREQUISITE_VALID": str(baseline_valid).lower(),
        "POSITIVE_CONTROL_KATS_VALID": str(controls_valid).lower(),
        "MASKED_STATUS_VALID": str(masked_valid).lower(),
        "INITIAL_HULL_INVARIANCE": str(initial_invariance).lower(),
        "OUTCOME": outcome,
        "MUTATION_TESTS": str(mutation_tests),
        "MUTATIONS_REJECTED": str(mutations_rejected),
        "V7A_RETROACTIVE_REINTERPRETATION_ALLOWED": "false",
        "C1_C2_DETERMINANT_COMPATIBILITY_EVALUATED": "false",
        "FULL_HPG_PIPELINE_EVALUATED": "false",
        "V7_B_ELIGIBILITY": "false",
        "V7_B_WINNER": "NONE",
        "PROMOTION_ELIGIBLE": "false",
        "OPEN_PROBLEM_SOLVED": "false",
        "FPGA_EXECUTION": "false",
    }
    if summary != expected_summary:
        fail("summary differs from independent reduction")

    retained_index = stable_bytes(result / "files.sha256", "content index")
    if retained_index != content_index(result):
        fail("content index differs from result tree")
    manifest = parse_kv(result / "manifest.txt", "manifest")
    expected_manifest = {
        "SCHEMA": "sounio.cs6.hapg-liouville-checkpoint-manifest.v1",
        "RUN_CONTRACT_SHA256": run_contract_sha,
        "FROZEN_CONTRACT_SHA256": FROZEN_CONTRACT_SHA256,
        "COORDINATE_MANIFEST_SHA256": COORDINATE_MANIFEST_SHA256,
        "ROOT_CHALLENGE": ROOT_CHALLENGE,
        "GIT_HEAD": run_contract["GIT_HEAD"],
        "ATTEMPT_COUNT": "9",
        "VERIFIED_CHECKPOINTS": str(verified_count),
        "BOUND_RQ_NAN": str(bound_negatives),
        "RUN_COMPLETE": "true",
        "RUN_VALID": str(run_valid).lower(),
        "OUTCOME": outcome,
        "MUTATION_TESTS": str(mutation_tests),
        "MUTATIONS_REJECTED": str(mutations_rejected),
        "FILES_SHA256": digest(result / "files.sha256"),
        "V7_B_WINNER": "NONE",
        "PROMOTION_ELIGIBLE": "false",
    }
    if manifest != expected_manifest:
        fail("manifest differs from independent reduction")

    print("AUDIT_SCHEMA=sounio.cs6.hapg-liouville-checkpoint-retained-audit.v1")
    print("AUDIT_PASS=true")
    print("ATTEMPTS_RECONSTRUCTED=9")
    print(f"WORKER_REPLAYS={worker_replay_count}")
    print(f"VERIFIER_REPLAYS={replay_count}")
    print(f"BOUND_NEGATIVES={bound_negatives}")
    print(f"RUN_VALID={str(run_valid).lower()}")
    print(f"OUTCOME={outcome}")
    print(f"MUTATION_TESTS={mutation_tests}")
    print(f"MUTATIONS_REJECTED={mutations_rejected}")
    print("V7_B_WINNER=NONE")
    print("PROMOTION_ELIGIBLE=false")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditError, OSError, ValueError) as error:
        print(f"retained audit error: {error}", file=sys.stderr)
        raise SystemExit(1)
