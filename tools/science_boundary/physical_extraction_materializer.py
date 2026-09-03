#!/usr/bin/env python3
"""Materialize and verify approved R3 exact-file copies."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
from typing import Any

from physical_extraction_inventory import (
    INVENTORY_SCHEMA,
    PhysicalExtractionError,
    canonical_json,
    expected_inventory,
    inventory_identity,
    read_inventory,
    resolve_input,
    sha256_bytes,
    stable_file_identity,
    within_root,
)


POLICY_SCHEMA = "sounio.physical-extraction-destination-policy.v1"
POLICY_TYPE = "local-destination-approval-policy"
POLICY_AUTHORITY = "explicit-local-copy-destination-approval"
MATERIALIZATION_SCHEMA = "sounio.physical-extraction-materialization.v1"
MATERIALIZATION_TYPE = "verified-local-exact-copy"
MATERIALIZATION_AUTHORITY = "local-destination-byte-identity"
MATERIALIZATION_STATUS = "copied-and-verified"
SOURCE_REMOVAL_STATUS = "not-authorized"
ASSURANCE_LEVEL = "identity-only"
MARKER_NAME = ".sounio-destination-approval.json"
MARKER_SCHEMA = "sounio.physical-extraction-destination.v1"
MARKER_TYPE = "preexisting-approved-destination"
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
POLICY_LIMITATIONS = [
    "does_not_create_destination_containers",
    "does_not_transfer_ownership_or_maintainership",
    "does_not_authorize_source_removal",
    "does_not_assert_remote_repository_state",
    "does_not_assert_publication_or_registry_status",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
MATERIALIZATION_LIMITATIONS = [
    "does_not_delete_source_files",
    "does_not_transfer_ownership_or_maintainership",
    "does_not_assert_remote_repository_state",
    "does_not_assert_publication_or_registry_status",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_guarantee_crash_atomicity_across_multiple_destinations",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "full_verification_requires_original_sources_inventory_policy_and_destinations",
]
POLICY_FIELDS = {
    "schema",
    "policy_type",
    "authority_scope",
    "source_inventory_identity_sha256",
    "approval_status",
    "destinations",
    "limitations",
    "policy_identity_sha256",
}
DESTINATION_FIELDS = {
    "target_id",
    "target_kind",
    "target_owner",
    "destination_key",
    "content_path",
    "approval_state",
    "approved_by",
    "approval_evidence",
    "destination_marker_sha256",
}
EVIDENCE_FIELDS = {"path", "size_bytes", "sha256"}
MARKER_FIELDS = {
    "schema",
    "marker_type",
    "target_id",
    "target_owner",
    "destination_key",
    "content_path",
    "approval_state",
    "source_inventory_identity_sha256",
}


class MaterializationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def policy_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("policy_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def receipt_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("materialization_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def with_receipt_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["materialization_identity_sha256"] = receipt_identity(result)
    return result


def read_regular_bytes(path: Path, label: str, code: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise MaterializationError(code, f"{label} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise MaterializationError(code, f"cannot read {label}: {error}") from error


def read_json(path: Path, label: str, code: str) -> tuple[dict[str, Any], bytes]:
    raw = read_regular_bytes(path, label, code)
    try:
        value = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise MaterializationError(code, f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise MaterializationError(code, f"{label} must be a JSON object")
    return value, raw


def stable_identity(path: Path, label: str, code: str) -> tuple[int, str]:
    try:
        return stable_file_identity(path)
    except PhysicalExtractionError as error:
        raise MaterializationError(code, f"cannot hash {label} ({error.code}): {error}") from error


def resolve_repo_path(repo_root: Path, value: str, label: str) -> Path:
    if not value or "\\" in value:
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label} is not a normalized repository path")
    pure = PurePosixPath(value)
    if pure.is_absolute() or value in {".", ".."} or any(part in {"", ".", ".."} for part in pure.parts):
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label} is not a normalized repository path")
    candidate = repo_root / pure
    if candidate.is_symlink():
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"cannot resolve {label}: {error}") from error
    if not within_root(resolved, repo_root) or not resolved.is_file():
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label} must be a regular file inside the source root")
    return resolved


def validate_evidence(repo_root: Path, value: Any, label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label} must be a non-empty array")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict) or set(item) != EVIDENCE_FIELDS:
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label}[{index}] has invalid fields")
        path_value = item.get("path")
        size = item.get("size_bytes")
        digest = item.get("sha256")
        if not isinstance(path_value, str) or path_value in seen:
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"{label}[{index}] has an invalid or duplicate path")
        path = resolve_repo_path(repo_root, path_value, f"{label}[{index}] path")
        actual_size, actual_digest = stable_identity(path, f"{label}[{index}]", "E-SRB-MATERIALIZE-002")
        if size != actual_size or digest != actual_digest:
            raise MaterializationError("E-SRB-MATERIALIZE-002", f"{label}[{index}] identity does not match {path_value}")
        seen.add(path_value)
        result.append({"path": path_value, "size_bytes": size, "sha256": digest})
    return result


def validate_policy(
    payload: dict[str, Any],
    repo_root: Path,
    inventory: dict[str, Any],
    *,
    require_approved: bool,
) -> list[dict[str, Any]]:
    if set(payload) != POLICY_FIELDS:
        raise MaterializationError("E-SRB-MATERIALIZE-001", "destination policy fields do not match v1")
    if payload.get("schema") != POLICY_SCHEMA or payload.get("policy_type") != POLICY_TYPE:
        raise MaterializationError("E-SRB-MATERIALIZE-001", "unsupported destination policy")
    if payload.get("authority_scope") != POLICY_AUTHORITY or payload.get("limitations") != POLICY_LIMITATIONS:
        raise MaterializationError("E-SRB-MATERIALIZE-001", "destination policy authority or limitations do not match v1")
    if payload.get("policy_identity_sha256") != policy_identity(payload):
        raise MaterializationError("E-SRB-MATERIALIZE-001", "destination policy identity hash mismatch")
    if payload.get("source_inventory_identity_sha256") != inventory.get("inventory_identity_sha256"):
        raise MaterializationError("E-SRB-MATERIALIZE-002", "destination policy is bound to another source inventory")
    rows = payload.get("destinations")
    if not isinstance(rows, list) or not rows:
        raise MaterializationError("E-SRB-MATERIALIZE-001", "destination policy must contain destination rows")

    planned = {unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"}
    validated: list[dict[str, Any]] = []
    targets: set[str] = set()
    keys: set[str] = set()
    states: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != DESTINATION_FIELDS:
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"destination policy row {index} has invalid fields")
        target_id = row.get("target_id")
        if not isinstance(target_id, str) or target_id not in planned or target_id in targets:
            raise MaterializationError("E-SRB-MATERIALIZE-002", f"destination policy row {index} has an unknown or duplicate target")
        unit = planned[target_id]
        if row.get("target_kind") != unit["target_kind"] or row.get("target_owner") != unit["target_owner"]:
            raise MaterializationError("E-SRB-MATERIALIZE-002", f"destination policy target binding mismatch for {target_id}")
        key = row.get("destination_key")
        content_path = row.get("content_path")
        approved_by = row.get("approved_by")
        state = row.get("approval_state")
        if not isinstance(key, str) or not SAFE_TOKEN.fullmatch(key) or key in keys:
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"invalid or duplicate destination key for {target_id}")
        if not isinstance(content_path, str) or not SAFE_TOKEN.fullmatch(content_path):
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"invalid destination content path for {target_id}")
        if not isinstance(approved_by, str) or not SAFE_TOKEN.fullmatch(approved_by):
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"invalid approver for {target_id}")
        if state not in {"approved", "pending"}:
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"invalid approval state for {target_id}")
        marker_sha256 = row.get("destination_marker_sha256")
        if not isinstance(marker_sha256, str) or not SHA256.fullmatch(marker_sha256):
            raise MaterializationError("E-SRB-MATERIALIZE-001", f"invalid destination marker identity for {target_id}")
        evidence = validate_evidence(repo_root, row.get("approval_evidence"), f"approval evidence for {target_id}")
        targets.add(target_id)
        keys.add(key)
        states.append(state)
        validated.append({**row, "approval_evidence": evidence})

    if set(planned) != targets:
        missing = sorted(set(planned) - targets)
        extra = sorted(targets - set(planned))
        raise MaterializationError(
            "E-SRB-MATERIALIZE-002",
            f"destination policy coverage mismatch missing={','.join(missing) or '-'} extra={','.join(extra) or '-'}",
        )
    expected_status = "approved" if all(state == "approved" for state in states) else "pending"
    if payload.get("approval_status") != expected_status:
        raise MaterializationError("E-SRB-MATERIALIZE-001", "destination policy aggregate approval status is invalid")
    if require_approved and expected_status != "approved":
        raise MaterializationError("E-SRB-MATERIALIZE-002", "destination policy is not fully approved")
    return sorted(validated, key=lambda row: row["target_id"])


def expected_marker(row: dict[str, Any], inventory: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema": MARKER_SCHEMA,
        "marker_type": MARKER_TYPE,
        "target_id": row["target_id"],
        "target_owner": row["target_owner"],
        "destination_key": row["destination_key"],
        "content_path": row["content_path"],
        "approval_state": "approved",
        "source_inventory_identity_sha256": inventory["inventory_identity_sha256"],
    }


def resolve_destinations(
    destinations_root: Path,
    repo_root: Path,
    rows: list[dict[str, Any]],
    inventory: dict[str, Any],
    *,
    require_content_absent: bool,
) -> list[dict[str, Any]]:
    if destinations_root.is_symlink() or not destinations_root.is_dir():
        raise MaterializationError("E-SRB-MATERIALIZE-003", "destinations root must be a preexisting regular directory")
    try:
        resolved_root = destinations_root.resolve(strict=True)
    except OSError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-003", f"cannot resolve destinations root: {error}") from error
    if within_root(resolved_root, repo_root) or within_root(repo_root, resolved_root):
        raise MaterializationError("E-SRB-MATERIALIZE-003", "destinations root must be separate from the source repository")
    root_device = resolved_root.stat().st_dev
    result: list[dict[str, Any]] = []
    for row in rows:
        container = resolved_root / row["destination_key"]
        if container.is_symlink() or not container.is_dir():
            raise MaterializationError("E-SRB-MATERIALIZE-003", f"approved destination is absent or unsafe: {row['destination_key']}")
        resolved_container = container.resolve(strict=True)
        if resolved_container.parent != resolved_root or resolved_container.stat().st_dev != root_device:
            raise MaterializationError("E-SRB-MATERIALIZE-003", f"destination is outside the approved root or filesystem: {row['destination_key']}")
        marker_path = resolved_container / MARKER_NAME
        marker, marker_raw = read_json(marker_path, f"destination marker for {row['target_id']}", "E-SRB-MATERIALIZE-003")
        if set(marker) != MARKER_FIELDS or marker != expected_marker(row, inventory):
            raise MaterializationError("E-SRB-MATERIALIZE-003", f"destination marker binding mismatch for {row['target_id']}")
        marker_digest = sha256_bytes(marker_raw)
        if marker_digest != row["destination_marker_sha256"]:
            raise MaterializationError("E-SRB-MATERIALIZE-003", f"destination marker identity mismatch for {row['target_id']}")
        content = resolved_container / row["content_path"]
        if content.is_symlink() or (require_content_absent and content.exists()):
            raise MaterializationError("E-SRB-MATERIALIZE-005", f"destination content is occupied: {row['target_id']}")
        if not require_content_absent and not content.is_dir():
            raise MaterializationError("E-SRB-MATERIALIZE-003", f"materialized content is absent: {row['target_id']}")
        result.append({**row, "container": resolved_container, "content": content, "marker_sha256": marker_digest})
    return result


def expected_relative_files(unit: dict[str, Any]) -> list[dict[str, Any]]:
    source_root = PurePosixPath(unit["source_path"])
    result: list[dict[str, Any]] = []
    for item in unit["files"]:
        try:
            relative = PurePosixPath(item["path"]).relative_to(source_root).as_posix()
        except ValueError as error:
            raise MaterializationError("E-SRB-MATERIALIZE-004", f"inventory file escapes unit {unit['source_path']}") from error
        if relative in {"", "."}:
            raise MaterializationError("E-SRB-MATERIALIZE-004", f"invalid inventory member for {unit['source_path']}")
        result.append({"path": relative, "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    return result


def write_all(descriptor: int, data: bytes) -> None:
    remaining = memoryview(data)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError("short write")
        remaining = remaining[written:]


def copy_verified(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    destination_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        source_descriptor = os.open(source, source_flags)
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise MaterializationError("E-SRB-MATERIALIZE-004", f"source is not a regular file: {source}")
        destination_descriptor = os.open(destination, destination_flags, 0o644)
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            write_all(destination_descriptor, chunk)
        os.fsync(destination_descriptor)
        after = os.fstat(source_descriptor)
        current = source.lstat()
        before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        current_identity = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if before_identity != after_identity or after_identity != current_identity:
            raise MaterializationError("E-SRB-MATERIALIZE-004", f"source changed while copying: {source}")
        if before.st_size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
            raise MaterializationError("E-SRB-MATERIALIZE-004", f"source no longer matches inventory: {source}")
    except OSError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-004", f"cannot copy {source}: {error}") from error
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)
        if destination.exists() and (destination.stat().st_size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]):
            destination.unlink(missing_ok=True)


def expected_directories(files: list[dict[str, Any]]) -> set[str]:
    result: set[str] = set()
    for item in files:
        parent = PurePosixPath(item["path"]).parent
        while parent.as_posix() not in {".", ""}:
            result.add(parent.as_posix())
            parent = parent.parent
    return result


def scan_content(content: Path, expected: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    if content.is_symlink() or not content.is_dir():
        raise MaterializationError("E-SRB-MATERIALIZE-004", f"{label} is not a regular directory")
    actual_files: list[dict[str, Any]] = []
    actual_directories: set[str] = set()
    for current, directories, names in os.walk(content, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            child = current_path / directory
            if child.is_symlink():
                raise MaterializationError("E-SRB-MATERIALIZE-004", f"symbolic-link directory in {label}: {child}")
            actual_directories.add(child.relative_to(content).as_posix())
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise MaterializationError("E-SRB-MATERIALIZE-004", f"symbolic-link file in {label}: {path}")
            size, digest = stable_identity(path, f"destination member {path}", "E-SRB-MATERIALIZE-004")
            actual_files.append({"path": path.relative_to(content).as_posix(), "size_bytes": size, "sha256": digest})
    if actual_files != expected or actual_directories != expected_directories(expected):
        raise MaterializationError("E-SRB-MATERIALIZE-004", f"{label} does not match the exact inventory file tree")
    return actual_files


def load_verified_inventory(
    repo_root: Path,
    inventory_path: Path,
    rings_path: Path,
    ownership_path: Path,
) -> tuple[dict[str, Any], bytes]:
    try:
        actual = read_inventory(inventory_path)
        if actual.get("schema") != INVENTORY_SCHEMA or actual.get("inventory_identity_sha256") != inventory_identity(actual):
            raise PhysicalExtractionError("E-SRB-EXTRACT-004", "physical inventory identity is invalid")
        expected = expected_inventory(repo_root, rings_path, ownership_path)
    except PhysicalExtractionError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-004", f"source inventory verification refused ({error.code}): {error}") from error
    if actual != expected:
        raise MaterializationError("E-SRB-MATERIALIZE-004", "source inventory no longer matches the source repository")
    return actual, read_regular_bytes(inventory_path, "source inventory", "E-SRB-MATERIALIZE-004")


def build_receipt(
    inventory: dict[str, Any],
    inventory_raw: bytes,
    policy: dict[str, Any],
    policy_raw: bytes,
    destinations: list[dict[str, Any]],
) -> dict[str, Any]:
    planned = {unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"}
    units: list[dict[str, Any]] = []
    for destination in sorted(destinations, key=lambda item: item["target_id"]):
        unit = planned[destination["target_id"]]
        files = expected_relative_files(unit)
        units.append(
            {
                "source_path": unit["source_path"],
                "ring": unit["ring"],
                "target_id": unit["target_id"],
                "target_owner": unit["target_owner"],
                "destination_key": destination["destination_key"],
                "content_path": destination["content_path"],
                "destination_marker_sha256": destination["marker_sha256"],
                "source_tree_sha256": unit["tree_sha256"],
                "destination_tree_sha256": sha256_bytes(canonical_json(files)),
                "file_count": unit["file_count"],
                "total_bytes": unit["total_bytes"],
                "copy_status": MATERIALIZATION_STATUS,
                "files": files,
            }
        )
    payload = {
        "schema": MATERIALIZATION_SCHEMA,
        "materialization_type": MATERIALIZATION_TYPE,
        "authority_scope": MATERIALIZATION_AUTHORITY,
        "materialization_status": MATERIALIZATION_STATUS,
        "source_removal_status": SOURCE_REMOVAL_STATUS,
        "source_bindings": {
            "inventory_file_sha256": sha256_bytes(inventory_raw),
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "destination_policy_file_sha256": sha256_bytes(policy_raw),
            "destination_policy_identity_sha256": policy["policy_identity_sha256"],
        },
        "summary": {
            "materialized_unit_count": len(units),
            "file_count": sum(unit["file_count"] for unit in units),
            "total_bytes": sum(unit["total_bytes"] for unit in units),
            "retained_source_units": inventory["summary"]["retained_core_units"],
            "blocked_source_units": inventory["summary"]["blocked_units"],
        },
        "units": units,
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": MATERIALIZATION_LIMITATIONS,
    }
    return with_receipt_identity(payload)


def stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise MaterializationError("E-SRB-MATERIALIZE-005", f"materialization receipt already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_tree(root: Path) -> None:
    directories = [Path(current) for current, _children, _files in os.walk(root, topdown=False, followlinks=False)]
    for directory in directories:
        fsync_directory(directory)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise MaterializationError("E-SRB-MATERIALIZE-001", "source repository root is not a directory")
    try:
        rings = resolve_input(repo_root, args.rings, "science rings")
        ownership = resolve_input(repo_root, args.ownership, "ownership policy")
    except PhysicalExtractionError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-001", f"cannot resolve inventory policy input ({error.code}): {error}") from error
    def absolute(value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else Path.cwd() / path

    inventory = absolute(args.inventory)
    policy = absolute(args.destination_policy)
    destinations_root = absolute(args.destinations_root)
    receipt_input = Path(args.receipt).expanduser()
    receipt = receipt_input if receipt_input.is_absolute() else Path.cwd() / receipt_input
    return repo_root, rings, ownership, inventory, policy, destinations_root, receipt


def materialize_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, inventory_path, policy_path, destinations_root, receipt = resolve_inputs(args)
    inventory, inventory_raw = load_verified_inventory(repo_root, inventory_path, rings, ownership)
    policy, policy_raw = read_json(policy_path, "destination policy", "E-SRB-MATERIALIZE-001")
    policy_rows = validate_policy(policy, repo_root, inventory, require_approved=True)
    destinations = resolve_destinations(destinations_root, repo_root, policy_rows, inventory, require_content_absent=True)
    if receipt.exists() or receipt.is_symlink():
        raise MaterializationError("E-SRB-MATERIALIZE-005", f"materialization receipt already exists: {receipt}")

    lock_path = destinations_root / ".sounio-physical-extraction.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as error:
        raise MaterializationError("E-SRB-MATERIALIZE-003", f"cannot open destination lock: {error}") from error
    if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
        os.close(lock_descriptor)
        raise MaterializationError("E-SRB-MATERIALIZE-003", "destination lock is not a regular file")
    stages: list[tuple[dict[str, Any], Path]] = []
    promoted: list[tuple[dict[str, Any], Path]] = []
    receipt_stage: Path | None = None
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise MaterializationError("E-SRB-MATERIALIZE-005", "another materialization holds the destination lock") from error
        destinations = resolve_destinations(destinations_root, repo_root, policy_rows, inventory, require_content_absent=True)
        planned = {unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"}
        for destination in destinations:
            unit = planned[destination["target_id"]]
            stage = Path(tempfile.mkdtemp(prefix=f".{destination['content_path']}.", suffix=".staging", dir=destination["container"]))
            stages.append((destination, stage))
            for item in expected_relative_files(unit):
                source = repo_root / PurePosixPath(unit["source_path"]) / PurePosixPath(item["path"])
                copy_verified(source, stage / PurePosixPath(item["path"]), item)
            scan_content(stage, expected_relative_files(unit), f"staged destination {destination['target_id']}")
            fsync_tree(stage)

        current_inventory, current_raw = load_verified_inventory(repo_root, inventory_path, rings, ownership)
        if current_inventory != inventory or current_raw != inventory_raw:
            raise MaterializationError("E-SRB-MATERIALIZE-004", "source inventory changed during materialization")
        current_policy, current_policy_raw = read_json(policy_path, "destination policy", "E-SRB-MATERIALIZE-001")
        current_rows = validate_policy(current_policy, repo_root, current_inventory, require_approved=True)
        if current_policy != policy or current_policy_raw != policy_raw or current_rows != policy_rows:
            raise MaterializationError("E-SRB-MATERIALIZE-002", "destination policy changed during materialization")
        destinations = resolve_destinations(destinations_root, repo_root, policy_rows, inventory, require_content_absent=True)
        receipt_payload = build_receipt(inventory, inventory_raw, policy, policy_raw, destinations)
        receipt_stage = stage_json(receipt, receipt_payload)

        stage_by_target = {destination["target_id"]: stage for destination, stage in stages}
        for destination in destinations:
            stage = stage_by_target[destination["target_id"]]
            if destination["content"].exists() or destination["content"].is_symlink():
                raise MaterializationError("E-SRB-MATERIALIZE-005", f"destination content appeared: {destination['target_id']}")
            os.rename(stage, destination["content"])
            promoted.append((destination, stage))
            fsync_directory(destination["container"])
        stages.clear()

        for destination in destinations:
            unit = planned[destination["target_id"]]
            scan_content(destination["content"], expected_relative_files(unit), f"materialized destination {destination['target_id']}")
        final_inventory, final_inventory_raw = load_verified_inventory(repo_root, inventory_path, rings, ownership)
        if final_inventory != inventory or final_inventory_raw != inventory_raw:
            raise MaterializationError("E-SRB-MATERIALIZE-004", "source inventory changed before receipt promotion")
        final_policy, final_policy_raw = read_json(policy_path, "destination policy", "E-SRB-MATERIALIZE-001")
        final_rows = validate_policy(final_policy, repo_root, final_inventory, require_approved=True)
        if final_policy != policy or final_policy_raw != policy_raw or final_rows != policy_rows:
            raise MaterializationError("E-SRB-MATERIALIZE-002", "destination policy changed before receipt promotion")
        final_destinations = resolve_destinations(
            destinations_root,
            repo_root,
            final_rows,
            final_inventory,
            require_content_absent=False,
        )
        if build_receipt(final_inventory, final_inventory_raw, final_policy, final_policy_raw, final_destinations) != receipt_payload:
            raise MaterializationError("E-SRB-MATERIALIZE-007", "materialization bindings changed before receipt promotion")
        destinations = final_destinations
        os.rename(receipt_stage, receipt)
        receipt_stage = None
        fsync_directory(receipt.parent)
    except Exception as error:
        rollback_errors: list[str] = []
        for destination, stage in reversed(promoted):
            try:
                if destination["content"].exists() and not stage.exists():
                    os.rename(destination["content"], stage)
                shutil.rmtree(stage, ignore_errors=False)
            except OSError as rollback_error:
                rollback_errors.append(f"{destination['target_id']}: {rollback_error}")
        for _destination, stage in stages:
            shutil.rmtree(stage, ignore_errors=True)
        if receipt_stage is not None:
            receipt_stage.unlink(missing_ok=True)
        if rollback_errors:
            raise MaterializationError(
                "E-SRB-MATERIALIZE-007",
                f"materialization rollback failed after {error}: {'; '.join(rollback_errors)}",
            ) from error
        raise
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    print(
        "PHYSICAL_EXTRACTION_MATERIALIZATION_PASS "
        f"receipt={receipt} units={len(destinations)} files={receipt_payload['summary']['file_count']} "
        f"status={MATERIALIZATION_STATUS} source_removal={SOURCE_REMOVAL_STATUS}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, inventory_path, policy_path, destinations_root, receipt_path = resolve_inputs(args)
    inventory, inventory_raw = load_verified_inventory(repo_root, inventory_path, rings, ownership)
    policy, policy_raw = read_json(policy_path, "destination policy", "E-SRB-MATERIALIZE-001")
    policy_rows = validate_policy(policy, repo_root, inventory, require_approved=True)
    destinations = resolve_destinations(destinations_root, repo_root, policy_rows, inventory, require_content_absent=False)
    planned = {unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"}
    for destination in destinations:
        scan_content(
            destination["content"],
            expected_relative_files(planned[destination["target_id"]]),
            f"materialized destination {destination['target_id']}",
        )
    actual, _raw = read_json(receipt_path, "materialization receipt", "E-SRB-MATERIALIZE-006")
    if actual.get("schema") != MATERIALIZATION_SCHEMA:
        raise MaterializationError("E-SRB-MATERIALIZE-006", "unsupported materialization receipt schema")
    if actual.get("materialization_identity_sha256") != receipt_identity(actual):
        raise MaterializationError("E-SRB-MATERIALIZE-006", "materialization receipt identity hash mismatch")
    expected = build_receipt(inventory, inventory_raw, policy, policy_raw, destinations)
    if actual != expected:
        raise MaterializationError("E-SRB-MATERIALIZE-006", "materialization receipt bindings do not match inputs")
    print(
        "PHYSICAL_EXTRACTION_MATERIALIZATION_VERIFY_PASS "
        f"receipt={receipt_path} units={len(destinations)} status={MATERIALIZATION_STATUS}"
    )
    return 0


def add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--repo-root", required=True)
    command.add_argument("--rings", default="science-rings.tsv")
    command.add_argument("--ownership", default="docs/ecosystem/science-physical-extraction-ownership.tsv")
    command.add_argument("--inventory", required=True)
    command.add_argument("--destination-policy", required=True)
    command.add_argument("--destinations-root", required=True)
    command.add_argument("--receipt", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-physical-extraction-materializer")
    subparsers = result.add_subparsers(dest="command", required=True)
    materialize = subparsers.add_parser("materialize")
    add_common_arguments(materialize)
    materialize.set_defaults(handler=materialize_command)
    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except MaterializationError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_MATERIALIZATION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error[E-SRB-MATERIALIZE-008]: materialization operation failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_MATERIALIZATION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
