#!/usr/bin/env python3
"""Authorize, but never execute, an R3 physical source-removal candidate."""

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
import subprocess
import sys
import tempfile
from typing import Any

from physical_extraction_inventory import (
    PhysicalExtractionError,
    canonical_json,
    resolve_input,
    sha256_bytes,
    stable_file_identity,
    within_root,
)
from physical_extraction_materializer import (
    MATERIALIZATION_SCHEMA,
    MaterializationError,
    build_receipt as build_materialization_receipt,
    load_verified_inventory,
    read_json,
    resolve_destinations,
    scan_content,
    expected_relative_files,
    receipt_identity as materialization_identity,
    validate_policy as validate_destination_policy,
)


POLICY_SCHEMA = "sounio.physical-extraction-source-removal-policy.v1"
POLICY_TYPE = "reviewed-post-removal-candidate-policy"
POLICY_AUTHORITY = "temporary-copy-source-removal-candidate-approval"
RECEIPT_SCHEMA = "sounio.physical-extraction-source-removal-authorization.v1"
AUTHORIZATION_TYPE = "verified-post-removal-candidate-authorization"
AUTHORIZATION_AUTHORITY = "local-temporary-copy-post-removal-evidence"
AUTHORIZATION_STATUS = "authorized-not-executed"
EXECUTION_STATUS = "not-executed"
ASSURANCE_LEVEL = "identity-only"
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
POLICY_LIMITATIONS = [
    "does_not_delete_original_source_files",
    "does_not_authorize_unlisted_source_paths",
    "does_not_execute_canonical_repository_migration",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "distinct_reviewer_labels_do_not_prove_organizational_independence",
    "post_removal_environment_is_not_fully_captured",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_delete_original_source_files",
    "authorizes_only_the_bound_scope_for_a_separate_execution_interface",
    "does_not_execute_canonical_repository_migration",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "distinct_reviewer_labels_do_not_prove_organizational_independence",
    "post_removal_environment_is_not_fully_captured",
    "does_not_preserve_uninventoried_filesystem_metadata_in_candidate_copy",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "full_verification_requires_original_sources_materialization_destinations_policy_and_gates",
]
POLICY_FIELDS = {
    "schema",
    "policy_type",
    "authority_scope",
    "source_bindings",
    "approval_status",
    "removal_scope",
    "review_evidence",
    "repairs",
    "post_removal_gates",
    "limitations",
    "policy_identity_sha256",
}
SOURCE_BINDING_FIELDS = {
    "inventory_file_sha256",
    "inventory_identity_sha256",
    "materialization_file_sha256",
    "materialization_identity_sha256",
}
SCOPE_FIELDS = {"scope_identity_sha256", "units"}
SCOPE_UNIT_FIELDS = {
    "source_path",
    "ring",
    "target_id",
    "target_owner",
    "file_count",
    "total_bytes",
    "tree_sha256",
}
REVIEW_FIELDS = {"reviewer_label", "path", "size_bytes", "sha256"}
REPAIR_FIELDS = {
    "path",
    "before_size_bytes",
    "before_sha256",
    "replacement_path",
    "replacement_size_bytes",
    "replacement_sha256",
    "after_size_bytes",
    "after_sha256",
}
GATE_FIELDS = {
    "gate_id",
    "argv",
    "cwd",
    "timeout_seconds",
    "expected_exit_code",
    "expected_stdout_sha256",
    "expected_stderr_sha256",
}


class SourceRemovalError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def policy_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("policy_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def authorization_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("authorization_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def with_authorization_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["authorization_identity_sha256"] = authorization_identity(result)
    return result


def normalized_repo_path(value: Any, label: str, *, allow_dot: bool = False) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SourceRemovalError("E-SRB-REMOVE-001", f"{label} is not a normalized repository path")
    if allow_dot and value == ".":
        return value
    pure = PurePosixPath(value)
    if pure.is_absolute() or value in {".", ".."} or any(part in {"", ".", ".."} for part in pure.parts):
        raise SourceRemovalError("E-SRB-REMOVE-001", f"{label} is not a normalized repository path")
    if pure.as_posix() != value:
        raise SourceRemovalError("E-SRB-REMOVE-001", f"{label} is not canonical")
    return value


def is_under(path: str, root: str) -> bool:
    path_parts = PurePosixPath(path).parts
    root_parts = PurePosixPath(root).parts
    return path_parts[: len(root_parts)] == root_parts


def stable_identity(path: Path, label: str, code: str) -> tuple[int, str]:
    try:
        return stable_file_identity(path)
    except PhysicalExtractionError as error:
        raise SourceRemovalError(code, f"cannot hash {label} ({error.code}): {error}") from error


def scan_repository(repo_root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for current, directories, names in os.walk(repo_root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == repo_root and ".git" in directories:
            directories.remove(".git")
        for directory in directories:
            child = current_path / directory
            if child.is_symlink():
                raise SourceRemovalError("E-SRB-REMOVE-004", f"symbolic-link directory is not candidate-safe: {child}")
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            if current_path == repo_root and name == ".git":
                continue
            if path.is_symlink():
                raise SourceRemovalError("E-SRB-REMOVE-004", f"symbolic-link file is not candidate-safe: {path}")
            try:
                mode = path.lstat().st_mode
            except OSError as error:
                raise SourceRemovalError("E-SRB-REMOVE-004", f"cannot inspect source member {path}: {error}") from error
            if not stat.S_ISREG(mode):
                raise SourceRemovalError("E-SRB-REMOVE-004", f"non-regular source member is not candidate-safe: {path}")
            size, digest = stable_identity(path, f"source member {path}", "E-SRB-REMOVE-004")
            files.append({"path": path.relative_to(repo_root).as_posix(), "size_bytes": size, "sha256": digest})
    return sorted(files, key=lambda item: item["path"])


def tree_identity(files: list[dict[str, Any]]) -> str:
    return sha256_bytes(canonical_json(files))


def removal_scope(inventory: dict[str, Any]) -> dict[str, Any]:
    units = []
    for unit in inventory["units"]:
        if unit["disposition"] != "extract-planned":
            continue
        units.append(
            {
                "source_path": unit["source_path"],
                "ring": unit["ring"],
                "target_id": unit["target_id"],
                "target_owner": unit["target_owner"],
                "file_count": unit["file_count"],
                "total_bytes": unit["total_bytes"],
                "tree_sha256": unit["tree_sha256"],
            }
        )
    units.sort(key=lambda item: item["source_path"])
    return {"scope_identity_sha256": sha256_bytes(canonical_json(units)), "units": units}


def load_materialization(
    repo_root: Path,
    rings: Path,
    ownership: Path,
    inventory_path: Path,
    destination_policy_path: Path,
    destinations_root: Path,
    materialization_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    try:
        inventory, inventory_raw = load_verified_inventory(repo_root, inventory_path, rings, ownership)
        destination_policy, destination_policy_raw = read_json(
            destination_policy_path, "destination policy", "E-SRB-MATERIALIZE-001"
        )
        rows = validate_destination_policy(destination_policy, repo_root, inventory, require_approved=True)
        destinations = resolve_destinations(
            destinations_root, repo_root, rows, inventory, require_content_absent=False
        )
        planned = {unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"}
        for destination in destinations:
            scan_content(
                destination["content"],
                expected_relative_files(planned[destination["target_id"]]),
                f"materialized destination {destination['target_id']}",
            )
        actual, materialization_raw = read_json(
            materialization_path, "materialization receipt", "E-SRB-MATERIALIZE-006"
        )
        if actual.get("schema") != MATERIALIZATION_SCHEMA:
            raise MaterializationError("E-SRB-MATERIALIZE-006", "unsupported materialization receipt schema")
        if actual.get("materialization_identity_sha256") != materialization_identity(actual):
            raise MaterializationError("E-SRB-MATERIALIZE-006", "materialization receipt identity hash mismatch")
        expected = build_materialization_receipt(
            inventory, inventory_raw, destination_policy, destination_policy_raw, destinations
        )
        if actual != expected:
            raise MaterializationError("E-SRB-MATERIALIZE-006", "materialization receipt bindings do not match inputs")
    except (MaterializationError, PhysicalExtractionError) as error:
        code = getattr(error, "code", "E-SRB-MATERIALIZE-008")
        raise SourceRemovalError(
            "E-SRB-REMOVE-004", f"materialization verification refused ({code}): {error}"
        ) from error
    return inventory, inventory_raw, destination_policy, destination_policy_raw, actual, materialization_raw


def resolve_source_file(repo_root: Path, value: str, label: str) -> Path:
    try:
        return resolve_input(repo_root, value, label)
    except PhysicalExtractionError as error:
        raise SourceRemovalError("E-SRB-REMOVE-002", f"cannot resolve {label} ({error.code}): {error}") from error


def validate_policy(
    payload: dict[str, Any],
    repo_root: Path,
    inventory: dict[str, Any],
    inventory_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    source_files: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    if set(payload) != POLICY_FIELDS:
        raise SourceRemovalError("E-SRB-REMOVE-001", "source-removal policy fields do not match v1")
    if payload.get("schema") != POLICY_SCHEMA or payload.get("policy_type") != POLICY_TYPE:
        raise SourceRemovalError("E-SRB-REMOVE-001", "unsupported source-removal policy")
    if payload.get("authority_scope") != POLICY_AUTHORITY or payload.get("limitations") != POLICY_LIMITATIONS:
        raise SourceRemovalError("E-SRB-REMOVE-001", "source-removal policy authority or limitations do not match v1")
    if payload.get("policy_identity_sha256") != policy_identity(payload):
        raise SourceRemovalError("E-SRB-REMOVE-001", "source-removal policy identity hash mismatch")
    if payload.get("approval_status") != "approved":
        raise SourceRemovalError("E-SRB-REMOVE-002", "source-removal policy is not approved")

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != SOURCE_BINDING_FIELDS:
        raise SourceRemovalError("E-SRB-REMOVE-001", "source-removal policy bindings are invalid")
    expected_bindings = {
        "inventory_file_sha256": sha256_bytes(inventory_raw),
        "inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "materialization_file_sha256": sha256_bytes(materialization_raw),
        "materialization_identity_sha256": materialization["materialization_identity_sha256"],
    }
    if bindings != expected_bindings:
        raise SourceRemovalError("E-SRB-REMOVE-002", "source-removal policy is bound to other inputs")

    expected_scope = removal_scope(inventory)
    scope = payload.get("removal_scope")
    if not isinstance(scope, dict) or set(scope) != SCOPE_FIELDS or scope != expected_scope:
        raise SourceRemovalError("E-SRB-REMOVE-002", "source-removal scope is not the exact planned inventory scope")
    if not expected_scope["units"]:
        raise SourceRemovalError("E-SRB-REMOVE-002", "source-removal scope is empty")
    if any(not isinstance(unit, dict) or set(unit) != SCOPE_UNIT_FIELDS for unit in scope["units"]):
        raise SourceRemovalError("E-SRB-REMOVE-001", "source-removal scope unit shape is invalid")
    planned_roots = [unit["source_path"] for unit in expected_scope["units"]]
    source_by_path = {item["path"]: item for item in source_files}

    review_evidence = payload.get("review_evidence")
    if not isinstance(review_evidence, list) or len(review_evidence) < 2:
        raise SourceRemovalError("E-SRB-REMOVE-002", "at least two review evidence records are required")
    validated_reviews = []
    labels: set[str] = set()
    review_paths: set[str] = set()
    for index, item in enumerate(review_evidence):
        if not isinstance(item, dict) or set(item) != REVIEW_FIELDS:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"review evidence {index} has invalid fields")
        label = item.get("reviewer_label")
        path_value = normalized_repo_path(item.get("path"), f"review evidence {index} path")
        if not isinstance(label, str) or not SAFE_TOKEN.fullmatch(label) or label in labels or path_value in review_paths:
            raise SourceRemovalError("E-SRB-REMOVE-002", f"review evidence {index} has a duplicate or invalid identity")
        if any(is_under(path_value, root) for root in planned_roots):
            raise SourceRemovalError("E-SRB-REMOVE-002", f"review evidence {index} would be removed")
        path = resolve_source_file(repo_root, path_value, f"review evidence {index}")
        size, digest = stable_identity(path, f"review evidence {index}", "E-SRB-REMOVE-002")
        if item.get("size_bytes") != size or item.get("sha256") != digest:
            raise SourceRemovalError("E-SRB-REMOVE-002", f"review evidence {index} identity mismatch")
        labels.add(label)
        review_paths.add(path_value)
        validated_reviews.append(dict(item))

    repairs = payload.get("repairs")
    if not isinstance(repairs, list) or not repairs:
        raise SourceRemovalError("E-SRB-REMOVE-002", "at least one exact repository repair is required")
    validated_repairs = []
    repair_paths: set[str] = set()
    for index, item in enumerate(repairs):
        if not isinstance(item, dict) or set(item) != REPAIR_FIELDS:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"repair {index} has invalid fields")
        path_value = normalized_repo_path(item.get("path"), f"repair {index} path")
        replacement_value = normalized_repo_path(item.get("replacement_path"), f"repair {index} replacement path")
        if path_value in repair_paths or any(is_under(path_value, root) for root in planned_roots):
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} target is duplicate or inside removal scope")
        if path_value == replacement_value:
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} replacement must be a distinct evidence file")
        if any(is_under(replacement_value, root) for root in planned_roots):
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} replacement would be removed")
        before = source_by_path.get(path_value)
        if before is None or before["size_bytes"] != item.get("before_size_bytes") or before["sha256"] != item.get("before_sha256"):
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} before identity mismatch")
        replacement = resolve_source_file(repo_root, replacement_value, f"repair {index} replacement")
        replacement_size, replacement_digest = stable_identity(
            replacement, f"repair {index} replacement", "E-SRB-REMOVE-002"
        )
        expected_after = (item.get("after_size_bytes"), item.get("after_sha256"))
        declared_replacement = (item.get("replacement_size_bytes"), item.get("replacement_sha256"))
        if declared_replacement != (replacement_size, replacement_digest) or expected_after != declared_replacement:
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} replacement or after identity mismatch")
        if expected_after == (before["size_bytes"], before["sha256"]):
            raise SourceRemovalError("E-SRB-REMOVE-002", f"repair {index} does not change the target bytes")
        repair_paths.add(path_value)
        validated_repairs.append(dict(item))

    gates = payload.get("post_removal_gates")
    if not isinstance(gates, list) or not gates:
        raise SourceRemovalError("E-SRB-REMOVE-002", "at least one post-removal gate is required")
    validated_gates = []
    gate_ids: set[str] = set()
    for index, item in enumerate(gates):
        if not isinstance(item, dict) or set(item) != GATE_FIELDS:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} has invalid fields")
        gate_id = item.get("gate_id")
        argv = item.get("argv")
        cwd = normalized_repo_path(item.get("cwd"), f"post-removal gate {index} cwd", allow_dot=True)
        timeout = item.get("timeout_seconds")
        expected_exit = item.get("expected_exit_code")
        if not isinstance(gate_id, str) or not SAFE_TOKEN.fullmatch(gate_id) or gate_id in gate_ids:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} has an invalid or duplicate ID")
        if not isinstance(argv, list) or not argv or any(
            not isinstance(arg, str) or not arg or "\x00" in arg or len(arg) > 4096 for arg in argv
        ):
            raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} argv is invalid")
        if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 900:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} timeout is invalid")
        if not isinstance(expected_exit, int) or isinstance(expected_exit, bool) or expected_exit != 0:
            raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} must expect exit code zero")
        for field in ("expected_stdout_sha256", "expected_stderr_sha256"):
            if not isinstance(item.get(field), str) or not SHA256.fullmatch(item[field]):
                raise SourceRemovalError("E-SRB-REMOVE-001", f"post-removal gate {index} has an invalid {field}")
        gate_ids.add(gate_id)
        validated_gates.append({**item, "cwd": cwd})

    return validated_reviews, validated_repairs, validated_gates


def copy_verified(source: Path, destination: Path, expected: dict[str, Any]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_descriptor = -1
    destination_descriptor = -1
    digest = hashlib.sha256()
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        before = os.fstat(source_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise SourceRemovalError("E-SRB-REMOVE-005", f"source is not a regular file: {source}")
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o644,
        )
        while True:
            chunk = os.read(source_descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            remaining = memoryview(chunk)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError("short write")
                remaining = remaining[written:]
        after = os.fstat(source_descriptor)
        current = source.lstat()
        before_id = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
        after_id = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
        current_id = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
        if before_id != after_id or after_id != current_id:
            raise SourceRemovalError("E-SRB-REMOVE-004", f"source changed while copying: {source}")
        if before.st_size != expected["size_bytes"] or digest.hexdigest() != expected["sha256"]:
            raise SourceRemovalError("E-SRB-REMOVE-004", f"source no longer matches snapshot: {source}")
    except OSError as error:
        raise SourceRemovalError("E-SRB-REMOVE-005", f"cannot construct candidate copy: {error}") from error
    finally:
        if source_descriptor >= 0:
            os.close(source_descriptor)
        if destination_descriptor >= 0:
            os.close(destination_descriptor)


def scan_candidate(candidate: Path) -> list[dict[str, Any]]:
    try:
        return scan_repository(candidate)
    except SourceRemovalError as error:
        raise SourceRemovalError("E-SRB-REMOVE-005", f"candidate tree verification refused: {error}") from error


def run_simulation(
    repo_root: Path,
    workspace_root: Path,
    inventory: dict[str, Any],
    source_files: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
    gates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    candidate = Path(tempfile.mkdtemp(prefix="sounio-source-removal-candidate.", dir=workspace_root))
    try:
        for item in source_files:
            copy_verified(repo_root / PurePosixPath(item["path"]), candidate / PurePosixPath(item["path"]), item)

        planned_roots = [
            unit["source_path"] for unit in inventory["units"] if unit["disposition"] == "extract-planned"
        ]
        for root in planned_roots:
            target = candidate / PurePosixPath(root)
            if target.is_symlink() or not target.is_dir():
                raise SourceRemovalError("E-SRB-REMOVE-005", f"candidate removal root is absent or unsafe: {root}")
            shutil.rmtree(target)

        expected = [item for item in source_files if not any(is_under(item["path"], root) for root in planned_roots)]
        expected_by_path = {item["path"]: dict(item) for item in expected}
        repair_evidence = []
        for repair in repairs:
            target = candidate / PurePosixPath(repair["path"])
            replacement = repo_root / PurePosixPath(repair["replacement_path"])
            target.unlink()
            copy_verified(
                replacement,
                target,
                {"size_bytes": repair["replacement_size_bytes"], "sha256": repair["replacement_sha256"]},
            )
            expected_by_path[repair["path"]] = {
                "path": repair["path"],
                "size_bytes": repair["after_size_bytes"],
                "sha256": repair["after_sha256"],
            }
            repair_evidence.append({**repair, "repair_status": "applied-and-verified"})

        expected_files = sorted(expected_by_path.values(), key=lambda item: item["path"])
        if scan_candidate(candidate) != expected_files:
            raise SourceRemovalError("E-SRB-REMOVE-005", "candidate differs from exact removal-and-repair plan before gates")
        for root in planned_roots:
            if (candidate / PurePosixPath(root)).exists() or (candidate / PurePosixPath(root)).is_symlink():
                raise SourceRemovalError("E-SRB-REMOVE-005", f"planned root remains in candidate: {root}")

        gate_evidence = []
        base_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", str(workspace_root)),
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "PYTHONHASHSEED": "0",
            "SOUNIO_REMOVAL_CANDIDATE_ROOT": str(candidate),
        }
        for gate in gates:
            cwd = candidate if gate["cwd"] == "." else candidate / PurePosixPath(gate["cwd"])
            if cwd.is_symlink() or not cwd.is_dir() or not within_root(cwd.resolve(strict=True), candidate):
                raise SourceRemovalError("E-SRB-REMOVE-006", f"post-removal gate cwd is absent: {gate['gate_id']}")
            try:
                result = subprocess.run(
                    gate["argv"],
                    cwd=cwd,
                    env=base_env,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=gate["timeout_seconds"],
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise SourceRemovalError("E-SRB-REMOVE-006", f"post-removal gate {gate['gate_id']} could not complete: {error}") from error
            stdout_digest = sha256_bytes(result.stdout)
            stderr_digest = sha256_bytes(result.stderr)
            if (
                result.returncode != gate["expected_exit_code"]
                or stdout_digest != gate["expected_stdout_sha256"]
                or stderr_digest != gate["expected_stderr_sha256"]
            ):
                raise SourceRemovalError(
                    "E-SRB-REMOVE-006",
                    f"post-removal gate {gate['gate_id']} output or exit status mismatch",
                )
            gate_evidence.append(
                {
                    "gate_id": gate["gate_id"],
                    "argv": gate["argv"],
                    "cwd": gate["cwd"],
                    "timeout_seconds": gate["timeout_seconds"],
                    "exit_code": result.returncode,
                    "stdout_size_bytes": len(result.stdout),
                    "stdout_sha256": stdout_digest,
                    "stderr_size_bytes": len(result.stderr),
                    "stderr_sha256": stderr_digest,
                    "gate_status": "passed",
                }
            )

        final_files = scan_candidate(candidate)
        if final_files != expected_files:
            raise SourceRemovalError("E-SRB-REMOVE-006", "a post-removal gate mutated the exact candidate tree")
        return repair_evidence, gate_evidence, tree_identity(final_files)
    finally:
        shutil.rmtree(candidate, ignore_errors=True)


def build_authorization(
    inventory: dict[str, Any],
    inventory_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    policy: dict[str, Any],
    policy_raw: bytes,
    source_files: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    repairs: list[dict[str, Any]],
    repair_evidence: list[dict[str, Any]],
    gate_evidence: list[dict[str, Any]],
    candidate_tree_sha256: str,
) -> dict[str, Any]:
    scope = removal_scope(inventory)
    payload = {
        "schema": RECEIPT_SCHEMA,
        "authorization_type": AUTHORIZATION_TYPE,
        "authority_scope": AUTHORIZATION_AUTHORITY,
        "authorization_status": AUTHORIZATION_STATUS,
        "source_removal_execution_status": EXECUTION_STATUS,
        "source_bindings": {
            "inventory_file_sha256": sha256_bytes(inventory_raw),
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "materialization_file_sha256": sha256_bytes(materialization_raw),
            "materialization_identity_sha256": materialization["materialization_identity_sha256"],
            "removal_policy_file_sha256": sha256_bytes(policy_raw),
            "removal_policy_identity_sha256": policy["policy_identity_sha256"],
        },
        "removal_scope": scope,
        "summary": {
            "authorized_unit_count": len(scope["units"]),
            "authorized_file_count": sum(unit["file_count"] for unit in scope["units"]),
            "authorized_total_bytes": sum(unit["total_bytes"] for unit in scope["units"]),
            "review_evidence_count": len(reviews),
            "repair_count": len(repairs),
            "post_removal_gate_count": len(gate_evidence),
        },
        "review_evidence": reviews,
        "repairs": repair_evidence,
        "post_removal_gates": gate_evidence,
        "candidate_evidence": {
            "original_source_file_count": len(source_files),
            "original_source_tree_sha256": tree_identity(source_files),
            "candidate_tree_sha256": candidate_tree_sha256,
            "candidate_status": "removed-repaired-and-gates-passed",
            "original_source_status": "reverified-unchanged",
        },
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": RECEIPT_LIMITATIONS,
    }
    return with_authorization_identity(payload)


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise SourceRemovalError("E-SRB-REMOVE-007", f"authorization receipt already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise SourceRemovalError("E-SRB-REMOVE-007", f"authorization receipt appeared during operation: {path}")
        os.link(temporary, path)
        temporary.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def absolute(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, Path, Path, Path]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise SourceRemovalError("E-SRB-REMOVE-001", "source repository root is not a directory")
    try:
        rings = resolve_input(repo_root, args.rings, "science rings")
        ownership = resolve_input(repo_root, args.ownership, "ownership policy")
    except PhysicalExtractionError as error:
        raise SourceRemovalError("E-SRB-REMOVE-001", f"cannot resolve inventory inputs ({error.code}): {error}") from error
    inventory = absolute(args.inventory)
    destination_policy = absolute(args.destination_policy)
    destinations_root = absolute(args.destinations_root)
    materialization = absolute(args.materialization_receipt)
    policy = absolute(args.removal_policy)
    workspace_input = absolute(args.workspace_root)
    if workspace_input.is_symlink() or not workspace_input.is_dir():
        raise SourceRemovalError("E-SRB-REMOVE-003", "workspace root must be a preexisting regular directory")
    workspace = workspace_input.resolve(strict=True)
    destination_resolved = destinations_root.resolve(strict=True)
    if (
        within_root(workspace, repo_root)
        or within_root(repo_root, workspace)
        or within_root(workspace, destination_resolved)
        or within_root(destination_resolved, workspace)
    ):
        raise SourceRemovalError("E-SRB-REMOVE-003", "workspace root must be separate from sources and destinations")
    receipt = absolute(args.authorization_receipt)
    try:
        receipt_parent = receipt.parent.resolve(strict=True)
    except OSError as error:
        raise SourceRemovalError("E-SRB-REMOVE-003", f"authorization receipt parent is absent: {error}") from error
    if within_root(receipt_parent, repo_root):
        raise SourceRemovalError("E-SRB-REMOVE-003", "authorization receipt must be outside the source repository")
    return (
        repo_root,
        rings,
        ownership,
        inventory,
        destination_policy,
        destinations_root,
        materialization,
        policy,
        workspace,
    )


def operation(args: argparse.Namespace, *, verify: bool) -> int:
    (
        repo_root,
        rings,
        ownership,
        inventory_path,
        destination_policy_path,
        destinations_root,
        materialization_path,
        policy_path,
        workspace_root,
    ) = resolve_inputs(args)
    receipt_path = absolute(args.authorization_receipt)
    if not verify and (receipt_path.exists() or receipt_path.is_symlink()):
        raise SourceRemovalError("E-SRB-REMOVE-007", f"authorization receipt already exists: {receipt_path}")

    lock_path = workspace_root / ".sounio-source-removal-authorization.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0), 0o600)
    except OSError as error:
        raise SourceRemovalError("E-SRB-REMOVE-003", f"cannot open authorization workspace lock: {error}") from error
    if not stat.S_ISREG(os.fstat(lock_descriptor).st_mode):
        os.close(lock_descriptor)
        raise SourceRemovalError("E-SRB-REMOVE-003", "authorization workspace lock is not a regular file")
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise SourceRemovalError("E-SRB-REMOVE-003", "another authorization holds the workspace lock") from error

        inventory, inventory_raw, _destination_policy, _destination_raw, materialization, materialization_raw = load_materialization(
            repo_root,
            rings,
            ownership,
            inventory_path,
            destination_policy_path,
            destinations_root,
            materialization_path,
        )
        source_files = scan_repository(repo_root)
        policy, policy_raw = read_json(policy_path, "source-removal policy", "E-SRB-REMOVE-001")
        reviews, repairs, gates = validate_policy(
            policy,
            repo_root,
            inventory,
            inventory_raw,
            materialization,
            materialization_raw,
            source_files,
        )
        repair_evidence, gate_evidence, candidate_tree_sha256 = run_simulation(
            repo_root, workspace_root, inventory, source_files, repairs, gates
        )

        final_inventory, final_inventory_raw, _dp, _dpr, final_materialization, final_materialization_raw = load_materialization(
            repo_root,
            rings,
            ownership,
            inventory_path,
            destination_policy_path,
            destinations_root,
            materialization_path,
        )
        final_source_files = scan_repository(repo_root)
        final_policy, final_policy_raw = read_json(policy_path, "source-removal policy", "E-SRB-REMOVE-001")
        final_reviews, final_repairs, final_gates = validate_policy(
            final_policy,
            repo_root,
            final_inventory,
            final_inventory_raw,
            final_materialization,
            final_materialization_raw,
            final_source_files,
        )
        if (
            final_inventory != inventory
            or final_inventory_raw != inventory_raw
            or final_materialization != materialization
            or final_materialization_raw != materialization_raw
            or final_source_files != source_files
            or final_policy != policy
            or final_policy_raw != policy_raw
            or final_reviews != reviews
            or final_repairs != repairs
            or final_gates != gates
        ):
            raise SourceRemovalError("E-SRB-REMOVE-004", "bound sources changed before authorization receipt")

        expected = build_authorization(
            inventory,
            inventory_raw,
            materialization,
            materialization_raw,
            policy,
            policy_raw,
            source_files,
            reviews,
            repairs,
            repair_evidence,
            gate_evidence,
            candidate_tree_sha256,
        )
        if verify:
            actual, _actual_raw = read_json(receipt_path, "source-removal authorization", "E-SRB-REMOVE-008")
            if actual.get("schema") != RECEIPT_SCHEMA:
                raise SourceRemovalError("E-SRB-REMOVE-008", "unsupported source-removal authorization schema")
            if actual.get("authorization_identity_sha256") != authorization_identity(actual):
                raise SourceRemovalError("E-SRB-REMOVE-008", "source-removal authorization identity hash mismatch")
            if actual != expected:
                raise SourceRemovalError("E-SRB-REMOVE-008", "source-removal authorization does not match reconstructed evidence")
        else:
            write_atomic(receipt_path, expected)
    finally:
        fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        os.close(lock_descriptor)

    action = "VERIFY_PASS" if verify else "PASS"
    print(
        f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_{action} "
        f"receipt={receipt_path} units={expected['summary']['authorized_unit_count']} "
        f"files={expected['summary']['authorized_file_count']} status={AUTHORIZATION_STATUS} "
        f"execution={EXECUTION_STATUS}"
    )
    return 0


def add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--repo-root", required=True)
    command.add_argument("--rings", default="science-rings.tsv")
    command.add_argument("--ownership", default="docs/ecosystem/science-physical-extraction-ownership.tsv")
    command.add_argument("--inventory", required=True)
    command.add_argument("--destination-policy", required=True)
    command.add_argument("--destinations-root", required=True)
    command.add_argument("--materialization-receipt", required=True)
    command.add_argument("--removal-policy", required=True)
    command.add_argument("--workspace-root", required=True)
    command.add_argument("--authorization-receipt", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-physical-extraction-source-removal-authorizer")
    subparsers = result.add_subparsers(dest="command", required=True)
    authorize = subparsers.add_parser("authorize")
    add_common_arguments(authorize)
    authorize.set_defaults(handler=lambda args: operation(args, verify=False))
    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.set_defaults(handler=lambda args: operation(args, verify=True))
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except SourceRemovalError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error[E-SRB-REMOVE-008]: source-removal authorization failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
