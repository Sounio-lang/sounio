#!/usr/bin/env python3
"""Execute and verify one explicitly authorized R3 local source removal."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from physical_extraction_inventory import canonical_json, sha256_bytes, within_root
from physical_extraction_materializer import (
    MARKER_NAME,
    MATERIALIZATION_SCHEMA,
    expected_directories,
    receipt_identity as materialization_identity,
    scan_content,
)
from source_removal_authorizer import (
    AUTHORIZATION_STATUS,
    EXECUTION_STATUS as AUTHORIZATION_EXECUTION_STATUS,
    RECEIPT_SCHEMA as AUTHORIZATION_SCHEMA,
    SourceRemovalError,
    absolute,
    authorization_identity,
    build_authorization,
    copy_verified,
    is_under,
    load_materialization,
    normalized_repo_path,
    read_json,
    run_simulation,
    scan_repository,
    stable_identity,
    tree_identity,
    validate_policy as validate_removal_policy,
)


POLICY_SCHEMA = "sounio.physical-extraction-source-removal-execution-policy.v1"
POLICY_TYPE = "explicit-local-source-removal-execution-policy"
POLICY_AUTHORITY = "exact-local-repository-tree-execution-approval"
RECEIPT_SCHEMA = "sounio.physical-extraction-source-removal-execution.v1"
EXECUTION_TYPE = "policy-bound-local-source-removal"
EXECUTION_AUTHORITY = "exact-local-repository-tree"
EXECUTION_STATUS = "executed-and-verified"
SOURCE_REMOVAL_STATUS = "executed"
ASSURANCE_LEVEL = "identity-only"
POLICY_LIMITATIONS = [
    "does_not_create_or_imply_production_approval",
    "authorizes_only_one_exact_local_repository_tree",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "does_not_guarantee_crash_atomicity_across_multiple_filesystem_operations",
    "crash_recovery_requires_the_retained_transaction_workspace",
    "requires_a_quiescent_execution_root_without_nonparticipating_writers",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_assert_this_execution_is_canonical_production_cutover",
    "execution_scope_is_only_the_bound_local_repository_tree",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "does_not_guarantee_crash_atomicity_across_multiple_filesystem_operations",
    "crash_recovery_requires_the_retained_transaction_workspace_when_no_receipt_exists",
    "requires_a_quiescent_execution_root_without_nonparticipating_writers",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "post_execution_verification_uses_bound_receipts_and_materialized_copies_not_removed_sources",
]
POLICY_FIELDS = {
    "schema",
    "policy_type",
    "authority_scope",
    "source_bindings",
    "approval_status",
    "execution_root_marker",
    "execution_scope",
    "operator_approval",
    "limitations",
    "policy_identity_sha256",
}
BINDING_FIELDS = {
    "authorization_file_sha256",
    "authorization_identity_sha256",
    "materialization_file_sha256",
    "materialization_identity_sha256",
    "inventory_identity_sha256",
    "pre_execution_tree_sha256",
    "post_execution_tree_sha256",
}
EVIDENCE_FIELDS = {"path", "size_bytes", "sha256"}
APPROVAL_FIELDS = {
    "approved_by",
    "approval_evidence",
    "authorization_identity_confirmation",
    "scope_identity_confirmation",
    "pre_execution_tree_confirmation",
}
MARKER_FIELDS = {
    "schema",
    "marker_type",
    "root_key",
    "approval_state",
    "approved_by",
}
MARKER_SCHEMA = "sounio.physical-extraction-source-removal-execution-root.v1"
MARKER_TYPE = "explicit-approved-execution-root"


class SourceRemovalExecutionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def policy_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("policy_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def execution_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("execution_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def with_execution_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["execution_identity_sha256"] = execution_identity(result)
    return result


def read_authorization(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload, raw = read_json(path, "source-removal authorization", "E-SRB-EXEC-002")
    except SourceRemovalError as error:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", str(error)) from error
    if payload.get("schema") != AUTHORIZATION_SCHEMA:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "unsupported source-removal authorization schema")
    if payload.get("authorization_status") != AUTHORIZATION_STATUS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "source-removal authorization is not approved")
    if payload.get("source_removal_execution_status") != AUTHORIZATION_EXECUTION_STATUS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "authorization already claims execution")
    if payload.get("authorization_identity_sha256") != authorization_identity(payload):
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "source-removal authorization identity mismatch")
    return payload, raw


def reconstruct_authorization(
    repo_root: Path,
    workspace_root: Path,
    rings: Path,
    ownership: Path,
    inventory_path: Path,
    destination_policy_path: Path,
    destinations_root: Path,
    materialization_path: Path,
    removal_policy_path: Path,
    authorization_path: Path,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, list[dict[str, Any]]]:
    try:
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
        removal_policy, removal_policy_raw = read_json(
            removal_policy_path, "source-removal policy", "E-SRB-REMOVE-001"
        )
        reviews, repairs, gates = validate_removal_policy(
            removal_policy,
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
        expected = build_authorization(
            inventory,
            inventory_raw,
            materialization,
            materialization_raw,
            removal_policy,
            removal_policy_raw,
            source_files,
            reviews,
            repairs,
            repair_evidence,
            gate_evidence,
            candidate_tree_sha256,
        )
    except SourceRemovalError as error:
        raise SourceRemovalExecutionError(
            "E-SRB-EXEC-002", f"authorization reconstruction refused ({error.code}): {error}"
        ) from error
    actual, actual_raw = read_authorization(authorization_path)
    if actual != expected:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "authorization does not match reconstructed evidence")
    return actual, actual_raw, materialization, materialization_raw, source_files


def expected_marker(approved_by: str) -> dict[str, Any]:
    return {
        "schema": MARKER_SCHEMA,
        "marker_type": MARKER_TYPE,
        "root_key": "approved-execution-root",
        "approval_state": "approved",
        "approved_by": approved_by,
    }


def resolve_retained_file(repo_root: Path, value: Any, label: str, planned_roots: list[str]) -> tuple[str, Path]:
    path_value = normalized_repo_path(value, label)
    if any(is_under(path_value, root) for root in planned_roots):
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"{label} would be removed")
    path = repo_root / PurePosixPath(path_value)
    if path.is_symlink() or not path.is_file():
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"{label} is not a retained regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"cannot resolve {label}: {error}") from error
    if not within_root(resolved, repo_root):
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"{label} escapes execution root")
    return path_value, resolved


def validate_file_evidence(
    repo_root: Path,
    item: Any,
    label: str,
    planned_roots: list[str],
) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != EVIDENCE_FIELDS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", f"{label} has invalid fields")
    path_value, path = resolve_retained_file(repo_root, item.get("path"), label, planned_roots)
    size, digest = stable_identity(path, label, "E-SRB-EXEC-003")
    if item.get("size_bytes") != size or item.get("sha256") != digest:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"{label} identity mismatch")
    return {"path": path_value, "size_bytes": size, "sha256": digest}


def validate_execution_policy(
    payload: dict[str, Any],
    repo_root: Path,
    authorization: dict[str, Any],
    authorization_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if set(payload) != POLICY_FIELDS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution policy fields do not match v1")
    if payload.get("schema") != POLICY_SCHEMA or payload.get("policy_type") != POLICY_TYPE:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "unsupported source-removal execution policy")
    if payload.get("authority_scope") != POLICY_AUTHORITY or payload.get("limitations") != POLICY_LIMITATIONS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution policy authority or limitations do not match v1")
    if payload.get("policy_identity_sha256") != policy_identity(payload):
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution policy identity mismatch")
    if payload.get("approval_status") != "approved":
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution policy is not approved")

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != BINDING_FIELDS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution policy source bindings are invalid")
    expected_bindings = {
        "authorization_file_sha256": sha256_bytes(authorization_raw),
        "authorization_identity_sha256": authorization["authorization_identity_sha256"],
        "materialization_file_sha256": sha256_bytes(materialization_raw),
        "materialization_identity_sha256": materialization["materialization_identity_sha256"],
        "inventory_identity_sha256": authorization["source_bindings"]["inventory_identity_sha256"],
        "pre_execution_tree_sha256": authorization["candidate_evidence"]["original_source_tree_sha256"],
        "post_execution_tree_sha256": authorization["candidate_evidence"]["candidate_tree_sha256"],
    }
    if bindings != expected_bindings:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution policy is bound to other evidence")
    if payload.get("execution_scope") != authorization.get("removal_scope"):
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution scope differs from authorization")
    planned_roots = [unit["source_path"] for unit in authorization["removal_scope"]["units"]]

    approval = payload.get("operator_approval")
    if not isinstance(approval, dict) or set(approval) != APPROVAL_FIELDS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "operator approval fields are invalid")
    approved_by = approval.get("approved_by")
    if not isinstance(approved_by, str) or not approved_by or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in approved_by
    ):
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "operator approval label is invalid")
    if (
        approval.get("authorization_identity_confirmation") != authorization["authorization_identity_sha256"]
        or approval.get("scope_identity_confirmation") != authorization["removal_scope"]["scope_identity_sha256"]
        or approval.get("pre_execution_tree_confirmation")
        != authorization["candidate_evidence"]["original_source_tree_sha256"]
    ):
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "operator approval confirmations do not match authorization")
    evidence = approval.get("approval_evidence")
    if not isinstance(evidence, list) or not evidence:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "operator approval evidence is empty")
    validated_evidence = []
    evidence_paths: set[str] = set()
    for index, item in enumerate(evidence):
        validated = validate_file_evidence(repo_root, item, f"operator approval evidence {index}", planned_roots)
        if validated["path"] in evidence_paths:
            raise SourceRemovalExecutionError("E-SRB-EXEC-003", "operator approval evidence paths are duplicated")
        evidence_paths.add(validated["path"])
        validated_evidence.append(validated)

    marker_item = payload.get("execution_root_marker")
    marker = validate_file_evidence(repo_root, marker_item, "execution root marker", planned_roots)
    marker_path = repo_root / PurePosixPath(marker["path"])
    try:
        marker_payload = json.loads(marker_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", f"cannot parse execution root marker: {error}") from error
    if not isinstance(marker_payload, dict) or set(marker_payload) != MARKER_FIELDS:
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution root marker fields are invalid")
    if marker_payload != expected_marker(approved_by):
        raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution root marker binding mismatch")
    return {**approval, "approval_evidence": validated_evidence}, marker


def confirm_operation(args: argparse.Namespace, policy: dict[str, Any], authorization: dict[str, Any]) -> None:
    expected = {
        "authorization": authorization["authorization_identity_sha256"],
        "scope": authorization["removal_scope"]["scope_identity_sha256"],
        "policy": policy["policy_identity_sha256"],
        "tree": authorization["candidate_evidence"]["original_source_tree_sha256"],
    }
    actual = {
        "authorization": args.confirm_authorization_identity,
        "scope": args.confirm_scope_identity,
        "policy": args.confirm_policy_identity,
        "tree": args.confirm_pre_execution_tree,
    }
    if actual != expected:
        raise SourceRemovalExecutionError("E-SRB-EXEC-004", "explicit CLI confirmations do not match bound execution evidence")


def verify_materialized_copies(
    destinations_root: Path,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    authorization: dict[str, Any],
) -> None:
    if materialization.get("schema") != MATERIALIZATION_SCHEMA:
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "unsupported materialization receipt")
    if materialization.get("materialization_identity_sha256") != materialization_identity(materialization):
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "materialization receipt identity mismatch")
    if (
        sha256_bytes(materialization_raw) != authorization["source_bindings"]["materialization_file_sha256"]
        or materialization["materialization_identity_sha256"]
        != authorization["source_bindings"]["materialization_identity_sha256"]
    ):
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "materialization differs from authorization binding")
    if destinations_root.is_symlink() or not destinations_root.is_dir():
        raise SourceRemovalExecutionError("E-SRB-EXEC-002", "materialization destinations root is unsafe")
    root = destinations_root.resolve(strict=True)
    for unit in materialization.get("units", []):
        container = root / unit["destination_key"]
        if container.is_symlink() or not container.is_dir() or container.resolve(strict=True).parent != root:
            raise SourceRemovalExecutionError("E-SRB-EXEC-002", f"materialized destination is unsafe: {unit['target_id']}")
        marker_path = container / MARKER_NAME
        if marker_path.is_symlink() or not marker_path.is_file():
            raise SourceRemovalExecutionError("E-SRB-EXEC-002", f"materialization marker is absent: {unit['target_id']}")
        marker_raw = marker_path.read_bytes()
        if sha256_bytes(marker_raw) != unit["destination_marker_sha256"]:
            raise SourceRemovalExecutionError("E-SRB-EXEC-002", f"materialization marker changed: {unit['target_id']}")
        content = container / unit["content_path"]
        expected_files = unit["files"]
        try:
            scan_content(content, expected_files, f"materialized destination {unit['target_id']}")
        except Exception as error:
            raise SourceRemovalExecutionError(
                "E-SRB-EXEC-002", f"materialized content verification failed for {unit['target_id']}: {error}"
            ) from error
        actual_directories = set()
        for current, directories, _names in os.walk(content, topdown=True, followlinks=False):
            current_path = Path(current)
            for directory in directories:
                actual_directories.add((current_path / directory).relative_to(content).as_posix())
        if actual_directories != expected_directories(expected_files):
            raise SourceRemovalExecutionError("E-SRB-EXEC-002", f"materialized directories changed: {unit['target_id']}")


def expected_post_execution_files(
    pre_files: list[dict[str, Any]], authorization: dict[str, Any]
) -> list[dict[str, Any]]:
    planned_roots = [unit["source_path"] for unit in authorization["removal_scope"]["units"]]
    result = {
        item["path"]: dict(item)
        for item in pre_files
        if not any(is_under(item["path"], root) for root in planned_roots)
    }
    for repair in authorization["repairs"]:
        result[repair["path"]] = {
            "path": repair["path"],
            "size_bytes": repair["after_size_bytes"],
            "sha256": repair["after_sha256"],
        }
    return sorted(result.values(), key=lambda item: item["path"])


def run_execution_gates(
    repo_root: Path,
    workspace_root: Path,
    authorization: dict[str, Any],
    *,
    verification: bool = False,
) -> list[dict[str, Any]]:
    evidence = []
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(workspace_root)),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "PYTHONHASHSEED": "0",
        "SOUNIO_REMOVAL_CANDIDATE_ROOT": str(repo_root),
        "SOUNIO_REMOVAL_EXECUTION_ROOT": str(repo_root),
        "SOUNIO_REMOVAL_EXECUTION_ACTIVE": "1",
        "SOUNIO_REMOVAL_VERIFICATION_ACTIVE": "1" if verification else "0",
    }
    for gate in authorization["post_removal_gates"]:
        cwd = repo_root if gate["cwd"] == "." else repo_root / PurePosixPath(gate["cwd"])
        if cwd.is_symlink() or not cwd.is_dir() or not within_root(cwd.resolve(strict=True), repo_root):
            raise SourceRemovalExecutionError("E-SRB-EXEC-006", f"execution gate cwd is absent: {gate['gate_id']}")
        try:
            result = subprocess.run(
                gate["argv"],
                cwd=cwd,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=gate["timeout_seconds"],
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise SourceRemovalExecutionError("E-SRB-EXEC-006", f"execution gate {gate['gate_id']} failed: {error}") from error
        stdout_sha256 = sha256_bytes(result.stdout)
        stderr_sha256 = sha256_bytes(result.stderr)
        if (
            result.returncode != gate["exit_code"]
            or stdout_sha256 != gate["stdout_sha256"]
            or stderr_sha256 != gate["stderr_sha256"]
        ):
            raise SourceRemovalExecutionError(
                "E-SRB-EXEC-006", f"execution gate {gate['gate_id']} output or exit status mismatch"
            )
        evidence.append(
            {
                "gate_id": gate["gate_id"],
                "argv": gate["argv"],
                "cwd": gate["cwd"],
                "timeout_seconds": gate["timeout_seconds"],
                "exit_code": result.returncode,
                "stdout_size_bytes": len(result.stdout),
                "stdout_sha256": stdout_sha256,
                "stderr_size_bytes": len(result.stderr),
                "stderr_sha256": stderr_sha256,
                "execution_gate_status": "passed",
            }
        )
    return evidence


def run_verification_gates(
    repo_root: Path,
    workspace_root: Path,
    authorization: dict[str, Any],
    files: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    candidate = Path(tempfile.mkdtemp(prefix=".sounio-source-removal-verification.", dir=workspace_root))
    try:
        for item in files:
            copy_verified(
                repo_root / PurePosixPath(item["path"]),
                candidate / PurePosixPath(item["path"]),
                item,
            )
        if scan_repository(candidate) != files:
            raise SourceRemovalExecutionError("E-SRB-EXEC-009", "verification copy differs from executed tree")
        evidence = run_execution_gates(candidate, workspace_root, authorization, verification=True)
        if scan_repository(candidate) != files:
            raise SourceRemovalExecutionError("E-SRB-EXEC-009", "verification gates changed disposable verification tree")
        return evidence
    finally:
        shutil.rmtree(candidate, ignore_errors=True)


def clear_repository(repo_root: Path) -> None:
    for child in repo_root.iterdir():
        if child.name == ".git":
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise SourceRemovalExecutionError("E-SRB-EXEC-007", f"cannot clear non-regular root member: {child}")


def restore_from_backup(repo_root: Path, backup: Path, pre_files: list[dict[str, Any]]) -> None:
    clear_repository(repo_root)
    for item in pre_files:
        copy_verified(backup / PurePosixPath(item["path"]), repo_root / PurePosixPath(item["path"]), item)
    restored = scan_repository(repo_root)
    if restored != pre_files:
        raise SourceRemovalExecutionError("E-SRB-EXEC-007", "rollback did not restore exact pre-execution tree")


def build_execution_receipt(
    authorization: dict[str, Any],
    authorization_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    policy: dict[str, Any],
    policy_raw: bytes,
    approval: dict[str, Any],
    marker: dict[str, Any],
    gate_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "execution_type": EXECUTION_TYPE,
        "authority_scope": EXECUTION_AUTHORITY,
        "execution_status": EXECUTION_STATUS,
        "source_removal_status": SOURCE_REMOVAL_STATUS,
        "source_bindings": {
            "authorization_file_sha256": sha256_bytes(authorization_raw),
            "authorization_identity_sha256": authorization["authorization_identity_sha256"],
            "materialization_file_sha256": sha256_bytes(materialization_raw),
            "materialization_identity_sha256": materialization["materialization_identity_sha256"],
            "execution_policy_file_sha256": sha256_bytes(policy_raw),
            "execution_policy_identity_sha256": policy["policy_identity_sha256"],
        },
        "execution_scope": authorization["removal_scope"],
        "summary": {
            "executed_unit_count": authorization["summary"]["authorized_unit_count"],
            "removed_file_count": authorization["summary"]["authorized_file_count"],
            "removed_total_bytes": authorization["summary"]["authorized_total_bytes"],
            "repair_count": authorization["summary"]["repair_count"],
            "post_removal_gate_count": len(gate_evidence),
        },
        "operator_approval": approval,
        "execution_root_marker": marker,
        "repairs": [
            {**repair, "execution_status": "executed-and-verified"}
            for repair in authorization["repairs"]
        ],
        "post_removal_gates": gate_evidence,
        "tree_evidence": {
            "pre_execution_file_count": authorization["candidate_evidence"]["original_source_file_count"],
            "pre_execution_tree_sha256": authorization["candidate_evidence"]["original_source_tree_sha256"],
            "post_execution_tree_sha256": authorization["candidate_evidence"]["candidate_tree_sha256"],
            "post_execution_status": "exact-authorized-tree",
        },
        "transaction_evidence": {
            "backup_type": "full-regular-file-pre-execution-copy",
            "receipt_promotion": "promoted-after-post-execution-verification",
            "normal_failure_rollback": "exact-pre-execution-tree",
            "transaction_status": "committed",
        },
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": RECEIPT_LIMITATIONS,
    }
    return with_execution_identity(payload)


def stage_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists() or path.is_symlink():
        raise SourceRemovalExecutionError("E-SRB-EXEC-008", f"execution receipt already exists: {path}")
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


def promote_staged_json(stage: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise SourceRemovalExecutionError("E-SRB-EXEC-008", f"execution receipt appeared during operation: {final}")
    try:
        os.link(stage, final)
    except OSError as error:
        raise SourceRemovalExecutionError("E-SRB-EXEC-008", f"cannot promote execution receipt: {error}") from error


def finish_receipt_promotion(stage: Path, final: Path) -> None:
    try:
        stage.unlink(missing_ok=True)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        print(f"warning: committed execution receipt cleanup or directory sync failed: {error}", file=sys.stderr)


def resolve_common(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path]:
    repo_input = Path(args.repo_root).expanduser()
    if repo_input.is_symlink() or not repo_input.is_dir():
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution repository root is unsafe")
    repo_root = repo_input.resolve(strict=True)
    git_entry = repo_root / ".git"
    if git_entry.is_symlink():
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution repository .git entry must not be a symbolic link")
    workspace_input = absolute(args.workspace_root)
    if workspace_input.is_symlink() or not workspace_input.is_dir():
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution workspace root is unsafe")
    workspace_root = workspace_input.resolve(strict=True)
    destinations_input = absolute(args.destinations_root)
    if destinations_input.is_symlink() or not destinations_input.is_dir():
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "materialization destinations root is unsafe")
    destinations_root = destinations_input.resolve(strict=True)
    if (
        within_root(workspace_root, repo_root)
        or within_root(repo_root, workspace_root)
        or within_root(workspace_root, destinations_root)
        or within_root(destinations_root, workspace_root)
    ):
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution workspace must be separate from sources and destinations")
    if workspace_root.stat().st_dev != repo_root.stat().st_dev:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "execution workspace must share the source filesystem")
    policy_path = absolute(args.execution_policy)
    authorization_path = absolute(args.authorization_receipt)
    materialization_path = absolute(args.materialization_receipt)
    receipt_path = absolute(args.execution_receipt)
    try:
        receipt_parent = receipt_path.parent.resolve(strict=True)
    except OSError as error:
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", f"execution receipt parent is absent: {error}") from error
    if (
        within_root(receipt_parent, repo_root)
        or within_root(receipt_parent, destinations_root)
        or within_root(receipt_parent, workspace_root)
    ):
        raise SourceRemovalExecutionError(
            "E-SRB-EXEC-001", "execution receipt must remain outside sources, destinations, and workspace"
        )
    return repo_root, workspace_root, destinations_root, policy_path, authorization_path, materialization_path


def acquire_execution_root_lock(repo_root: Path) -> int:
    try:
        descriptor = os.open(
            repo_root,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return descriptor
    except BlockingIOError as error:
        os.close(descriptor)
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", "another operation holds the execution-root lock") from error
    except OSError as error:
        if "descriptor" in locals():
            os.close(descriptor)
        raise SourceRemovalExecutionError("E-SRB-EXEC-001", f"cannot lock execution root: {error}") from error


def release_execution_root_lock(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def execute_command(args: argparse.Namespace) -> int:
    repo_root, workspace_root, destinations_root, policy_path, authorization_path, materialization_path = resolve_common(args)
    receipt_path = absolute(args.execution_receipt)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise SourceRemovalExecutionError("E-SRB-EXEC-008", f"execution receipt already exists: {receipt_path}")
    rings = repo_root / PurePosixPath(args.rings)
    ownership = repo_root / PurePosixPath(args.ownership)
    inventory_path = absolute(args.inventory)
    destination_policy_path = absolute(args.destination_policy)
    removal_policy_path = absolute(args.removal_policy)

    lock_descriptor = acquire_execution_root_lock(repo_root)
    transaction: Path | None = None
    stage: Path | None = None
    receipt_promoted = False
    mutation_started = False
    pre_files: list[dict[str, Any]] = []
    backup: Path | None = None
    try:
        authorization, authorization_raw, materialization, materialization_raw, pre_files = reconstruct_authorization(
            repo_root,
            workspace_root,
            rings,
            ownership,
            inventory_path,
            destination_policy_path,
            destinations_root,
            materialization_path,
            removal_policy_path,
            authorization_path,
        )
        verify_materialized_copies(destinations_root, materialization, materialization_raw, authorization)
        policy, policy_raw = read_json(policy_path, "source-removal execution policy", "E-SRB-EXEC-001")
        approval, marker = validate_execution_policy(
            policy, repo_root, authorization, authorization_raw, materialization, materialization_raw
        )
        confirm_operation(args, policy, authorization)
        if tree_identity(pre_files) != authorization["candidate_evidence"]["original_source_tree_sha256"]:
            raise SourceRemovalExecutionError("E-SRB-EXEC-005", "execution root does not match authorized pre-execution tree")

        transaction = Path(
            tempfile.mkdtemp(
                prefix=f".sounio-source-removal-execution.{policy['policy_identity_sha256'][:12]}.",
                dir=workspace_root,
            )
        )
        backup = transaction / "backup"
        backup.mkdir()
        for item in pre_files:
            copy_verified(repo_root / PurePosixPath(item["path"]), backup / PurePosixPath(item["path"]), item)
        if scan_repository(backup) != pre_files:
            raise SourceRemovalExecutionError("E-SRB-EXEC-005", "transaction backup does not match pre-execution tree")
        if scan_repository(repo_root) != pre_files:
            raise SourceRemovalExecutionError("E-SRB-EXEC-005", "execution root changed while transaction backup was created")

        mutation_started = True
        for unit in authorization["removal_scope"]["units"]:
            target = repo_root / PurePosixPath(unit["source_path"])
            if target.is_symlink() or not target.is_dir():
                raise SourceRemovalExecutionError("E-SRB-EXEC-005", f"authorized removal root is absent: {unit['source_path']}")
            shutil.rmtree(target)
        for repair in authorization["repairs"]:
            target = repo_root / PurePosixPath(repair["path"])
            replacement = backup / PurePosixPath(repair["replacement_path"])
            target.unlink()
            copy_verified(
                replacement,
                target,
                {"size_bytes": repair["after_size_bytes"], "sha256": repair["after_sha256"]},
            )

        expected_files = expected_post_execution_files(pre_files, authorization)
        after_apply = scan_repository(repo_root)
        if after_apply != expected_files:
            raise SourceRemovalExecutionError("E-SRB-EXEC-005", "execution root differs from authorized tree after apply")
        gate_evidence = run_execution_gates(repo_root, workspace_root, authorization)
        final_files = scan_repository(repo_root)
        if final_files != expected_files or tree_identity(final_files) != authorization["candidate_evidence"]["candidate_tree_sha256"]:
            raise SourceRemovalExecutionError("E-SRB-EXEC-006", "execution gates changed the authorized final tree")
        verify_materialized_copies(destinations_root, materialization, materialization_raw, authorization)
        final_authorization, final_authorization_raw = read_authorization(authorization_path)
        final_materialization, final_materialization_raw = read_json(
            materialization_path, "materialization receipt", "E-SRB-EXEC-002"
        )
        if (
            final_authorization != authorization
            or final_authorization_raw != authorization_raw
            or final_materialization != materialization
            or final_materialization_raw != materialization_raw
        ):
            raise SourceRemovalExecutionError("E-SRB-EXEC-002", "authorization or materialization changed during execution")
        final_policy, final_policy_raw = read_json(policy_path, "source-removal execution policy", "E-SRB-EXEC-001")
        final_approval, final_marker = validate_execution_policy(
            final_policy, repo_root, authorization, authorization_raw, materialization, materialization_raw
        )
        if final_policy != policy or final_policy_raw != policy_raw or final_approval != approval or final_marker != marker:
            raise SourceRemovalExecutionError("E-SRB-EXEC-003", "execution policy evidence changed during execution")
        expected = build_execution_receipt(
            authorization,
            authorization_raw,
            materialization,
            materialization_raw,
            policy,
            policy_raw,
            approval,
            marker,
            gate_evidence,
        )
        stage = stage_json(receipt_path, expected)
        promote_staged_json(stage, receipt_path)
        receipt_promoted = True
        finish_receipt_promotion(stage, receipt_path)
        stage = None
        try:
            shutil.rmtree(transaction)
            transaction = None
        except OSError as cleanup_error:
            print(
                f"warning: committed execution transaction cleanup requires manual removal: {cleanup_error}",
                file=sys.stderr,
            )
    except Exception as error:
        if stage is not None:
            stage.unlink(missing_ok=True)
        if not receipt_promoted and mutation_started and backup is not None and backup.is_dir() and pre_files:
            try:
                restore_from_backup(repo_root, backup, pre_files)
            except Exception as rollback_error:
                raise SourceRemovalExecutionError(
                    "E-SRB-EXEC-007",
                    f"execution failed ({error}); exact rollback failed ({rollback_error}); transaction retained at {transaction}",
                ) from error
        if transaction is not None and transaction.exists():
            shutil.rmtree(transaction, ignore_errors=True)
        raise
    finally:
        release_execution_root_lock(lock_descriptor)

    print(
        "PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_PASS "
        f"receipt={receipt_path} units={expected['summary']['executed_unit_count']} "
        f"files={expected['summary']['removed_file_count']} status={EXECUTION_STATUS}"
    )
    return 0


def verify_locked(
    args: argparse.Namespace,
    repo_root: Path,
    workspace_root: Path,
    destinations_root: Path,
    policy_path: Path,
    authorization_path: Path,
    materialization_path: Path,
) -> int:
    receipt_path = absolute(args.execution_receipt)
    authorization, authorization_raw = read_authorization(authorization_path)
    materialization, materialization_raw = read_json(
        materialization_path, "materialization receipt", "E-SRB-EXEC-002"
    )
    verify_materialized_copies(destinations_root, materialization, materialization_raw, authorization)
    policy, policy_raw = read_json(policy_path, "source-removal execution policy", "E-SRB-EXEC-001")
    approval, marker = validate_execution_policy(
        policy, repo_root, authorization, authorization_raw, materialization, materialization_raw
    )
    confirm_operation(args, policy, authorization)
    files = scan_repository(repo_root)
    if tree_identity(files) != authorization["candidate_evidence"]["candidate_tree_sha256"]:
        raise SourceRemovalExecutionError("E-SRB-EXEC-009", "post-execution tree identity mismatch")
    for unit in authorization["removal_scope"]["units"]:
        target = repo_root / PurePosixPath(unit["source_path"])
        if target.exists() or target.is_symlink():
            raise SourceRemovalExecutionError("E-SRB-EXEC-009", f"executed removal root is present: {unit['source_path']}")
    gate_evidence = run_verification_gates(repo_root, workspace_root, authorization, files)
    if scan_repository(repo_root) != files:
        raise SourceRemovalExecutionError("E-SRB-EXEC-009", "executed tree changed during verification")
    expected = build_execution_receipt(
        authorization,
        authorization_raw,
        materialization,
        materialization_raw,
        policy,
        policy_raw,
        approval,
        marker,
        gate_evidence,
    )
    actual, _actual_raw = read_json(receipt_path, "source-removal execution receipt", "E-SRB-EXEC-009")
    if actual.get("schema") != RECEIPT_SCHEMA:
        raise SourceRemovalExecutionError("E-SRB-EXEC-009", "unsupported source-removal execution receipt")
    if actual.get("execution_identity_sha256") != execution_identity(actual):
        raise SourceRemovalExecutionError("E-SRB-EXEC-009", "source-removal execution identity mismatch")
    if actual != expected:
        raise SourceRemovalExecutionError("E-SRB-EXEC-009", "source-removal execution receipt does not match evidence")
    print(
        "PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_VERIFY_PASS "
        f"receipt={receipt_path} units={expected['summary']['executed_unit_count']} status={EXECUTION_STATUS}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    resolved = resolve_common(args)
    lock_descriptor = acquire_execution_root_lock(resolved[0])
    try:
        return verify_locked(args, *resolved)
    finally:
        release_execution_root_lock(lock_descriptor)


def add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--repo-root", required=True)
    command.add_argument("--destinations-root", required=True)
    command.add_argument("--materialization-receipt", required=True)
    command.add_argument("--authorization-receipt", required=True)
    command.add_argument("--execution-policy", required=True)
    command.add_argument("--workspace-root", required=True)
    command.add_argument("--execution-receipt", required=True)
    command.add_argument("--confirm-authorization-identity", required=True)
    command.add_argument("--confirm-scope-identity", required=True)
    command.add_argument("--confirm-policy-identity", required=True)
    command.add_argument("--confirm-pre-execution-tree", required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-physical-extraction-source-removal-executor")
    subparsers = result.add_subparsers(dest="command", required=True)
    execute = subparsers.add_parser("execute")
    add_common_arguments(execute)
    execute.add_argument("--rings", default="science-rings.tsv")
    execute.add_argument("--ownership", default="docs/ecosystem/science-physical-extraction-ownership.tsv")
    execute.add_argument("--inventory", required=True)
    execute.add_argument("--destination-policy", required=True)
    execute.add_argument("--removal-policy", required=True)
    execute.set_defaults(handler=execute_command)
    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except SourceRemovalExecutionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except SourceRemovalError as error:
        print(f"error[E-SRB-EXEC-009]: dependent authorization refused ({error.code}): {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error[E-SRB-EXEC-009]: source-removal execution failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
