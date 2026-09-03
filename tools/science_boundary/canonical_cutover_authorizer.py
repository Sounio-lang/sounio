#!/usr/bin/env python3
"""Authorize, but never execute, one exact R3 canonical repository cutover."""

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
from source_removal_authorizer import (
    SourceRemovalError,
    absolute,
    copy_verified,
    normalized_repo_path,
    read_json,
    scan_repository,
    tree_identity,
)
from source_removal_executor import (
    SourceRemovalExecutionError,
    acquire_execution_root_lock,
    expected_post_execution_files,
    reconstruct_authorization,
    release_execution_root_lock,
    restore_from_backup,
    run_execution_gates,
    validate_file_evidence,
    verify_materialized_copies,
)


POLICY_SCHEMA = "sounio.physical-extraction-canonical-cutover-policy.v1"
POLICY_TYPE = "explicit-canonical-cutover-approval-policy"
POLICY_AUTHORITY = "exact-git-repository-tree-cutover-approval"
RECEIPT_SCHEMA = "sounio.physical-extraction-canonical-cutover-approval.v1"
APPROVAL_TYPE = "policy-bound-canonical-cutover-approval"
APPROVAL_AUTHORITY = "exact-git-repository-tree-cutover-approval"
APPROVAL_STATUS = "approved-not-executed"
EXECUTION_STATUS = "not-executed"
SOURCE_REMOVAL_STATUS = "not-executed"
ASSURANCE_LEVEL = "identity-plus-git-remote-ref"
APPROVAL_CONTEXTS = {"disposable-fixture", "canonical-production"}

POLICY_LIMITATIONS = [
    "policy_does_not_execute_the_canonical_cutover",
    "approval_is_limited_to_the_exact_bound_git_repositories_and_refs",
    "disposable_fixture_context_is_not_canonical_production_approval",
    "git_remote_ref_observation_does_not_prove_hosting_administration_or_ownership",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "rehearsal_does_not_capture_every_production_environment_property",
    "cutover_execution_requires_a_separate_explicit_interface",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_execute_the_canonical_cutover",
    "does_not_remove_any_source_from_the_bound_canonical_repository",
    "approval_is_limited_to_the_exact_bound_git_repositories_and_refs",
    "disposable_fixture_receipt_is_not_canonical_production_approval",
    "git_remote_ref_observation_does_not_prove_hosting_administration_or_ownership",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "rehearsal_does_not_capture_every_production_environment_property",
    "git_object_ids_are_bound_as_observed_not_as_independent_signatures",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
]

POLICY_FIELDS = {
    "schema",
    "policy_type",
    "authority_scope",
    "approval_context",
    "source_bindings",
    "approval_status",
    "canonical_repository",
    "destinations",
    "recovery_plan",
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
    "pre_cutover_tree_sha256",
    "authorized_post_cutover_tree_sha256",
    "removal_scope_identity_sha256",
    "repair_set_identity_sha256",
    "gate_set_identity_sha256",
}
GIT_FIELDS = {
    "repository_id",
    "remote_name",
    "remote_url",
    "branch",
    "head_oid",
    "remote_head_oid",
}
CANONICAL_FIELDS = GIT_FIELDS | {"retained_marker"}
DESTINATION_FIELDS = {
    "source_path",
    "ring",
    "target_id",
    "target_owner",
    "checkout_path",
    "repository_id",
    "remote_name",
    "remote_url",
    "branch",
    "head_oid",
    "remote_head_oid",
    "file_count",
    "total_bytes",
    "tree_sha256",
    "owner_approval_evidence",
}
RECOVERY_BINDING_FIELDS = {
    "plan_evidence",
    "recovery_plan_identity_sha256",
    "pre_cutover_tree_confirmation",
    "authorized_post_cutover_tree_confirmation",
}
RECOVERY_PLAN_FIELDS = {
    "schema",
    "plan_type",
    "canonical_repository_id",
    "required_backup_model",
    "transaction_workspace",
    "receipt_commit_point",
    "no_receipt_recovery",
    "receipt_present_recovery",
    "crash_atomicity",
    "approved_by",
}
OPERATOR_FIELDS = {
    "approved_by",
    "approval_evidence",
    "authorization_identity_confirmation",
    "scope_identity_confirmation",
    "pre_cutover_tree_confirmation",
    "authorized_post_cutover_tree_confirmation",
    "destination_set_identity_confirmation",
    "recovery_plan_identity_confirmation",
    "repairs_reviewed",
    "gates_reviewed",
}
EVIDENCE_FIELDS = {"path", "size_bytes", "sha256"}
MARKER_FIELDS = {"schema", "marker_type", "repository_id", "approval_context"}


class CanonicalCutoverApprovalError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def identity(payload: Any, field: str | None = None) -> str:
    value = json.loads(json.dumps(payload))
    if field is not None and isinstance(value, dict):
        value.pop(field, None)
    return sha256_bytes(canonical_json(value))


def approval_identity(payload: dict[str, Any]) -> str:
    return identity(payload, "approval_identity_sha256")


def policy_identity(payload: dict[str, Any]) -> str:
    return identity(payload, "policy_identity_sha256")


def safe_token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(
        character not in "abcdefghijklmnopqrstuvwxyz0123456789._-" for character in value
    ):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"{label} is invalid")
    return value


def safe_branch(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 200
        or value.startswith("-")
        or value.startswith("/")
        or value.endswith("/")
        or ".." in value
        or "//" in value
        or "@{" in value
        or any(character.isspace() or ord(character) < 32 for character in value)
        or any(character in "~^:?*[\\" for character in value)
    ):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"{label} is invalid")
    return value


def oid(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"{label} is not a Git SHA-1 object id")
    return value


def git_run(repository: Path, arguments: list[str], label: str) -> str:
    environment = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", str(repository.parent)),
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"cannot inspect {label}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"cannot inspect {label}: {detail}")
    try:
        return result.stdout.decode("utf-8").rstrip("\n")
    except UnicodeError as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} returned non-UTF-8 output") from error


def verify_git_repository(repository: Path, expected: dict[str, Any], label: str) -> dict[str, str]:
    if not isinstance(expected, dict) or not GIT_FIELDS.issubset(expected):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"{label} Git binding fields are invalid")
    if repository.is_symlink() or not repository.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} is not a regular repository directory")
    resolved = repository.resolve(strict=True)
    git_entry = resolved / ".git"
    if git_entry.is_symlink() or not git_entry.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} is not a standalone Git worktree")
    if git_run(resolved, ["rev-parse", "--is-inside-work-tree"], label) != "true":
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} is not a Git worktree")
    if git_run(resolved, ["rev-parse", "--is-bare-repository"], label) != "false":
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} unexpectedly resolves as bare")
    if Path(git_run(resolved, ["rev-parse", "--show-toplevel"], label)).resolve(strict=True) != resolved:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} is not the Git worktree root")
    git_directory = Path(git_run(resolved, ["rev-parse", "--absolute-git-dir"], label)).resolve(strict=True)
    common_value = git_run(resolved, ["rev-parse", "--git-common-dir"], label)
    common_directory = Path(common_value)
    if not common_directory.is_absolute():
        common_directory = resolved / common_directory
    if (
        git_directory != git_entry.resolve(strict=True)
        or common_directory.resolve(strict=True) != git_directory
        or git_run(resolved, ["rev-parse", "--show-superproject-working-tree"], label)
        or git_run(resolved, ["config", "--get", "core.repositoryformatversion"], label) != "0"
    ):
        raise CanonicalCutoverApprovalError(
            "E-SRB-CUTOVER-003", f"{label} uses a linked, nested, or unsupported repository layout"
        )
    if git_run(resolved, ["status", "--porcelain=v1", "--untracked-files=all"], label):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} worktree is not clean")

    repository_id = safe_token(expected.get("repository_id"), f"{label} repository id")
    remote_name = safe_token(expected.get("remote_name"), f"{label} remote name")
    branch = safe_branch(expected.get("branch"), f"{label} branch")
    remote_url = expected.get("remote_url")
    if (
        not isinstance(remote_url, str)
        or not remote_url
        or len(remote_url) > 2048
        or any(character in "\r\n\x00" for character in remote_url)
    ):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"{label} remote URL is invalid")
    expected_head = oid(expected.get("head_oid"), f"{label} head")
    expected_remote_head = oid(expected.get("remote_head_oid"), f"{label} remote head")

    actual_branch = git_run(resolved, ["symbolic-ref", "--quiet", "--short", "HEAD"], label)
    actual_head = git_run(resolved, ["rev-parse", "HEAD"], label)
    actual_url = git_run(resolved, ["remote", "get-url", remote_name], label)
    remote_output = git_run(resolved, ["ls-remote", "--heads", remote_name, f"refs/heads/{branch}"], label)
    remote_lines = [line for line in remote_output.splitlines() if line]
    expected_ref = f"refs/heads/{branch}"
    if len(remote_lines) != 1:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} remote branch is absent or ambiguous")
    fields = remote_lines[0].split("\t")
    if len(fields) != 2 or fields[1] != expected_ref:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} remote branch response is invalid")
    actual_remote_head = fields[0]
    if (
        actual_branch != branch
        or actual_head != expected_head
        or actual_url != remote_url
        or actual_remote_head != expected_remote_head
        or actual_head != actual_remote_head
    ):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-003", f"{label} Git state differs from policy")
    return {
        "repository_id": repository_id,
        "remote_name": remote_name,
        "remote_url": remote_url,
        "branch": branch,
        "head_oid": actual_head,
        "remote_head_oid": actual_remote_head,
        "worktree_status": "clean",
        "remote_ref_status": "head-equals-remote-branch",
    }


def expected_source_bindings(
    authorization: dict[str, Any],
    authorization_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
) -> dict[str, str]:
    return {
        "authorization_file_sha256": sha256_bytes(authorization_raw),
        "authorization_identity_sha256": authorization["authorization_identity_sha256"],
        "materialization_file_sha256": sha256_bytes(materialization_raw),
        "materialization_identity_sha256": materialization["materialization_identity_sha256"],
        "inventory_identity_sha256": authorization["source_bindings"]["inventory_identity_sha256"],
        "pre_cutover_tree_sha256": authorization["candidate_evidence"]["original_source_tree_sha256"],
        "authorized_post_cutover_tree_sha256": authorization["candidate_evidence"]["candidate_tree_sha256"],
        "removal_scope_identity_sha256": authorization["removal_scope"]["scope_identity_sha256"],
        "repair_set_identity_sha256": identity(authorization["repairs"]),
        "gate_set_identity_sha256": identity(authorization["post_removal_gates"]),
    }


def validate_recovery_plan(
    repo_root: Path,
    item: Any,
    planned_roots: list[str],
    canonical_repository_id: str,
    approved_by: str,
    bindings: dict[str, str],
) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != RECOVERY_BINDING_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "recovery plan binding fields are invalid")
    try:
        evidence = validate_file_evidence(repo_root, item.get("plan_evidence"), "recovery plan evidence", planned_roots)
    except SourceRemovalExecutionError as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", str(error)) from error
    path = repo_root / PurePosixPath(evidence["path"])
    try:
        plan = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", f"cannot parse recovery plan: {error}") from error
    if not isinstance(plan, dict) or set(plan) != RECOVERY_PLAN_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "recovery plan fields are invalid")
    expected_plan = {
        "schema": "sounio.physical-extraction-canonical-cutover-recovery-plan.v1",
        "plan_type": "full-regular-file-backup-and-explicit-rollback",
        "canonical_repository_id": canonical_repository_id,
        "required_backup_model": "full-regular-file-pre-execution-copy",
        "transaction_workspace": "same-filesystem-external",
        "receipt_commit_point": "atomic-hardlink-after-verification",
        "no_receipt_recovery": "restore-and-verify-pre-cutover-tree",
        "receipt_present_recovery": "committed-state-manual-review",
        "crash_atomicity": "not-guaranteed-across-multiple-filesystem-operations",
        "approved_by": approved_by,
    }
    if plan != expected_plan:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "recovery plan content differs from v1")
    plan_identity = identity(plan)
    if (
        item.get("recovery_plan_identity_sha256") != plan_identity
        or item.get("pre_cutover_tree_confirmation") != bindings["pre_cutover_tree_sha256"]
        or item.get("authorized_post_cutover_tree_confirmation")
        != bindings["authorized_post_cutover_tree_sha256"]
    ):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "recovery plan confirmations differ from source bindings")
    return {**item, "plan_evidence": evidence, "recovery_plan": plan}


def destination_set_identity(destinations: list[dict[str, Any]]) -> str:
    return identity(destinations)


def validate_policy(
    payload: dict[str, Any],
    repo_root: Path,
    repositories_root: Path,
    authorization: dict[str, Any],
    authorization_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    if set(payload) != POLICY_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover policy fields do not match v1")
    if payload.get("schema") != POLICY_SCHEMA or payload.get("policy_type") != POLICY_TYPE:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "unsupported canonical cutover policy")
    if payload.get("authority_scope") != POLICY_AUTHORITY or payload.get("limitations") != POLICY_LIMITATIONS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover policy authority or limitations differ from v1")
    if payload.get("policy_identity_sha256") != policy_identity(payload):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover policy identity mismatch")
    if payload.get("approval_status") != "approved":
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "cutover policy is not approved")
    if payload.get("approval_context") not in APPROVAL_CONTEXTS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover approval context is invalid")

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != BINDING_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover source bindings are invalid")
    expected_bindings = expected_source_bindings(
        authorization, authorization_raw, materialization, materialization_raw
    )
    if bindings != expected_bindings:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-002", "cutover policy is bound to other source evidence")
    planned_roots = [unit["source_path"] for unit in authorization["removal_scope"]["units"]]

    canonical = payload.get("canonical_repository")
    if not isinstance(canonical, dict) or set(canonical) != CANONICAL_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "canonical repository binding fields are invalid")
    canonical_git = verify_git_repository(repo_root, canonical, "canonical repository")
    try:
        marker = validate_file_evidence(
            repo_root, canonical.get("retained_marker"), "canonical root marker", planned_roots
        )
    except SourceRemovalExecutionError as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", str(error)) from error
    marker_path = repo_root / PurePosixPath(marker["path"])
    try:
        marker_payload = json.loads(marker_path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", f"cannot parse canonical root marker: {error}") from error
    expected_marker = {
        "schema": "sounio.physical-extraction-canonical-root.v1",
        "marker_type": "explicit-canonical-cutover-approval-root",
        "repository_id": canonical_git["repository_id"],
        "approval_context": payload["approval_context"],
    }
    if not isinstance(marker_payload, dict) or set(marker_payload) != MARKER_FIELDS or marker_payload != expected_marker:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "canonical root marker differs from policy context")

    destination_rows = payload.get("destinations")
    if not isinstance(destination_rows, list) or not destination_rows:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", "cutover destinations are empty")
    materialized_by_source = {unit["source_path"]: unit for unit in materialization.get("units", [])}
    if len(materialized_by_source) != len(materialization.get("units", [])):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-002", "materialization source paths are duplicated")
    validated_destinations: list[dict[str, Any]] = []
    seen_sources: set[str] = set()
    seen_checkouts: set[str] = set()
    seen_repositories: set[str] = {canonical_git["repository_id"]}
    owner_evidence_paths: set[str] = set()
    for index, row in enumerate(destination_rows):
        if not isinstance(row, dict) or set(row) != DESTINATION_FIELDS:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"destination {index} fields are invalid")
        source_path = normalized_repo_path(row.get("source_path"), f"destination {index} source path")
        if source_path in seen_sources or source_path not in materialized_by_source:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", f"destination {index} source binding is invalid")
        seen_sources.add(source_path)
        checkout_name = normalized_repo_path(row.get("checkout_path"), f"destination {index} checkout path")
        if "/" in checkout_name or checkout_name in seen_checkouts:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", f"destination {index} checkout path is not unique direct child")
        seen_checkouts.add(checkout_name)
        checkout = repositories_root / checkout_name
        if checkout.is_symlink() or not checkout.is_dir() or checkout.resolve(strict=True).parent != repositories_root:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", f"destination {index} checkout is unsafe")
        unit = materialized_by_source[source_path]
        expected_values = {
            "source_path": unit["source_path"],
            "ring": unit["ring"],
            "target_id": unit["target_id"],
            "target_owner": unit["target_owner"],
            "file_count": unit["file_count"],
            "total_bytes": unit["total_bytes"],
            "tree_sha256": unit["destination_tree_sha256"],
        }
        if any(row.get(key) != value for key, value in expected_values.items()):
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", f"destination {index} differs from materialization")
        try:
            actual_files = scan_repository(checkout)
        except SourceRemovalError as error:
            raise CanonicalCutoverApprovalError(
                "E-SRB-CUTOVER-004", f"destination {index} content inspection refused: {error}"
            ) from error
        if actual_files != unit["files"] or tree_identity(actual_files) != unit["destination_tree_sha256"]:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", f"destination {index} content differs from materialization")
        git_expected = {key: row[key] for key in GIT_FIELDS}
        git_state = verify_git_repository(checkout, git_expected, f"destination {index}")
        if git_state["repository_id"] in seen_repositories:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", "destination repository ids are duplicated")
        seen_repositories.add(git_state["repository_id"])
        try:
            evidence = validate_file_evidence(
                repo_root, row.get("owner_approval_evidence"), f"destination {index} owner evidence", planned_roots
            )
        except SourceRemovalExecutionError as error:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", str(error)) from error
        if evidence["path"] in owner_evidence_paths:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "destination owner evidence is duplicated")
        owner_evidence_paths.add(evidence["path"])
        validated_destinations.append(
            {
                **expected_values,
                "checkout_path": checkout_name,
                **git_state,
                "owner_approval_evidence": evidence,
                "destination_status": "clean-head-equals-remote-and-content-verified",
            }
        )
    if seen_sources != set(materialized_by_source):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", "cutover destinations do not cover materialization exactly")
    actual_checkout_names = {child.name for child in repositories_root.iterdir()}
    if actual_checkout_names != seen_checkouts:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-004", "destination repositories root has unexpected members")
    validated_destinations.sort(key=lambda item: item["source_path"])

    approval = payload.get("operator_approval")
    if not isinstance(approval, dict) or set(approval) != OPERATOR_FIELDS:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "operator approval fields are invalid")
    approved_by = safe_token(approval.get("approved_by"), "operator approval label")
    evidence_items = approval.get("approval_evidence")
    if not isinstance(evidence_items, list) or not evidence_items:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "operator approval evidence is empty")
    validated_evidence = []
    evidence_paths: set[str] = set()
    for index, item in enumerate(evidence_items):
        try:
            evidence = validate_file_evidence(repo_root, item, f"operator evidence {index}", planned_roots)
        except SourceRemovalExecutionError as error:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", str(error)) from error
        if evidence["path"] in evidence_paths:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "operator approval evidence is duplicated")
        evidence_paths.add(evidence["path"])
        validated_evidence.append(evidence)

    recovery = validate_recovery_plan(
        repo_root,
        payload.get("recovery_plan"),
        planned_roots,
        canonical_git["repository_id"],
        approved_by,
        bindings,
    )
    expected_confirmations = {
        "authorization_identity_confirmation": bindings["authorization_identity_sha256"],
        "scope_identity_confirmation": bindings["removal_scope_identity_sha256"],
        "pre_cutover_tree_confirmation": bindings["pre_cutover_tree_sha256"],
        "authorized_post_cutover_tree_confirmation": bindings["authorized_post_cutover_tree_sha256"],
        "destination_set_identity_confirmation": destination_set_identity(destination_rows),
        "recovery_plan_identity_confirmation": recovery["recovery_plan_identity_sha256"],
        "repairs_reviewed": True,
        "gates_reviewed": True,
    }
    if any(approval.get(key) != value for key, value in expected_confirmations.items()):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-005", "operator confirmations differ from cutover evidence")
    validated_approval = {**approval, "approval_evidence": validated_evidence}
    validated_canonical = {
        **canonical_git,
        "retained_marker": marker,
        "canonical_repository_status": "clean-head-equals-remote",
    }
    return validated_canonical, validated_destinations, recovery, validated_approval


def run_rehearsal(
    repo_root: Path,
    workspace_root: Path,
    authorization: dict[str, Any],
    pre_files: list[dict[str, Any]],
) -> dict[str, Any]:
    transaction = Path(tempfile.mkdtemp(prefix=".sounio-canonical-cutover-rehearsal.", dir=workspace_root))
    transaction.chmod(0o700)
    candidate = transaction / "candidate"
    backup = transaction / "backup"
    candidate.mkdir()
    backup.mkdir()
    try:
        for item in pre_files:
            source = repo_root / PurePosixPath(item["path"])
            copy_verified(source, candidate / PurePosixPath(item["path"]), item)
            copy_verified(source, backup / PurePosixPath(item["path"]), item)
        if scan_repository(candidate) != pre_files or scan_repository(backup) != pre_files:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal copy or backup differs from source")

        for unit in authorization["removal_scope"]["units"]:
            target = candidate / PurePosixPath(unit["source_path"])
            if target.is_symlink() or not target.is_dir():
                raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal removal root is absent")
            shutil.rmtree(target)
        repair_evidence = []
        for repair in authorization["repairs"]:
            target = candidate / PurePosixPath(repair["path"])
            if target.is_symlink() or not target.is_file():
                raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal repair target is absent")
            target.unlink()
            copy_verified(
                repo_root / PurePosixPath(repair["replacement_path"]),
                target,
                {
                    "size_bytes": repair["replacement_size_bytes"],
                    "sha256": repair["replacement_sha256"],
                },
            )
            repair_evidence.append({
                "path": repair["path"],
                "after_sha256": repair["after_sha256"],
                "rehearsal_status": "applied-and-verified",
            })
        expected_post = expected_post_execution_files(pre_files, authorization)
        if (
            tree_identity(expected_post) != authorization["candidate_evidence"]["candidate_tree_sha256"]
            or scan_repository(candidate) != expected_post
        ):
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal tree differs before gates")
        try:
            gates = run_execution_gates(candidate, workspace_root, authorization)
        except SourceRemovalExecutionError as error:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", str(error)) from error
        if scan_repository(candidate) != expected_post:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal gates changed the candidate tree")
        try:
            restore_from_backup(candidate, backup, pre_files)
        except SourceRemovalExecutionError as error:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", str(error)) from error
        return {
            "backup_type": "full-regular-file-pre-cutover-copy",
            "pre_cutover_tree_sha256": tree_identity(pre_files),
            "authorized_post_cutover_tree_sha256": tree_identity(expected_post),
            "restored_tree_sha256": tree_identity(scan_repository(candidate)),
            "authorized_unit_count": authorization["summary"]["authorized_unit_count"],
            "authorized_file_count": authorization["summary"]["authorized_file_count"],
            "repair_evidence": repair_evidence,
            "post_removal_gates": [
                {
                    "gate_id": gate["gate_id"],
                    "exit_code": gate["exit_code"],
                    "stdout_sha256": gate["stdout_sha256"],
                    "stderr_sha256": gate["stderr_sha256"],
                    "rehearsal_status": "passed",
                }
                for gate in gates
            ],
            "rehearsal_status": "removed-repaired-gates-passed-and-pre-tree-restored",
        }
    finally:
        shutil.rmtree(transaction, ignore_errors=True)


def confirm_operation(
    args: argparse.Namespace, policy: dict[str, Any], authorization: dict[str, Any]
) -> dict[str, str]:
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
        "tree": args.confirm_pre_cutover_tree,
    }
    if actual != expected:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-007", "explicit CLI confirmations differ from cutover evidence")
    return {
        "authorization_identity_sha256": actual["authorization"],
        "scope_identity_sha256": actual["scope"],
        "policy_identity_sha256": actual["policy"],
        "pre_cutover_tree_sha256": actual["tree"],
        "confirmation_status": "matched",
    }


def build_receipt(
    policy: dict[str, Any],
    policy_raw: bytes,
    authorization: dict[str, Any],
    authorization_raw: bytes,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    canonical: dict[str, Any],
    destinations: list[dict[str, Any]],
    recovery: dict[str, Any],
    operator: dict[str, Any],
    confirmations: dict[str, str],
    rehearsal: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "schema": RECEIPT_SCHEMA,
        "approval_type": APPROVAL_TYPE,
        "authority_scope": APPROVAL_AUTHORITY,
        "approval_context": policy["approval_context"],
        "canonical_cutover_approval_status": APPROVAL_STATUS,
        "canonical_cutover_execution_status": EXECUTION_STATUS,
        "source_removal_status": SOURCE_REMOVAL_STATUS,
        "source_bindings": {
            **expected_source_bindings(authorization, authorization_raw, materialization, materialization_raw),
            "cutover_policy_file_sha256": sha256_bytes(policy_raw),
            "cutover_policy_identity_sha256": policy["policy_identity_sha256"],
        },
        "canonical_repository": canonical,
        "destinations": destinations,
        "operator_approval": operator,
        "explicit_cli_confirmations": confirmations,
        "recovery_plan": recovery,
        "rehearsal_evidence": rehearsal,
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": RECEIPT_LIMITATIONS,
    }
    payload["approval_identity_sha256"] = approval_identity(payload)
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-008", f"cutover approval receipt already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    stage = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-008", "cutover approval output appeared during operation")
        os.link(stage, path)
        stage.unlink()
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except CanonicalCutoverApprovalError:
        stage.unlink(missing_ok=True)
        raise
    except OSError as error:
        stage.unlink(missing_ok=True)
        raise CanonicalCutoverApprovalError(
            "E-SRB-CUTOVER-008", f"cannot promote cutover approval receipt: {error}"
        ) from error


def resolve_common(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo_input = Path(args.repo_root).expanduser()
    workspace_input = absolute(args.workspace_root)
    destinations_input = absolute(args.destinations_root)
    repositories_input = absolute(args.repositories_root)
    if repo_input.is_symlink() or not repo_input.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "canonical repository root is unsafe")
    if workspace_input.is_symlink() or not workspace_input.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "approval workspace root is unsafe")
    if destinations_input.is_symlink() or not destinations_input.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "materialization destinations root is unsafe")
    if repositories_input.is_symlink() or not repositories_input.is_dir():
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "destination repositories root is unsafe")
    repo_root = repo_input.resolve(strict=True)
    workspace_root = workspace_input.resolve(strict=True)
    destinations_root = destinations_input.resolve(strict=True)
    repositories_root = repositories_input.resolve(strict=True)
    roots = [repo_root, workspace_root, destinations_root, repositories_root]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if within_root(first, second) or within_root(second, first):
                raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "source, workspace and destination roots must be separate")
    receipt = absolute(args.cutover_approval_receipt)
    try:
        receipt_parent = receipt.parent.resolve(strict=True)
    except OSError as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"cutover receipt parent is absent: {error}") from error
    if any(within_root(receipt_parent, root) for root in roots):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", "cutover receipt must remain outside all bound roots")
    return repo_root, workspace_root, destinations_root, repositories_root, receipt


def reconstruct(args: argparse.Namespace, repo_root: Path, workspace_root: Path, destinations_root: Path):
    try:
        result = reconstruct_authorization(
            repo_root,
            workspace_root,
            repo_root / PurePosixPath(args.rings),
            repo_root / PurePosixPath(args.ownership),
            absolute(args.inventory),
            absolute(args.destination_policy),
            destinations_root,
            absolute(args.materialization_receipt),
            absolute(args.removal_policy),
            absolute(args.authorization_receipt),
        )
        authorization, authorization_raw, materialization, materialization_raw, pre_files = result
        verify_materialized_copies(destinations_root, materialization, materialization_raw, authorization)
        return authorization, authorization_raw, materialization, materialization_raw, pre_files
    except SourceRemovalExecutionError as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-002", f"source evidence reconstruction refused: {error}") from error


def read_policy(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        payload, raw = read_json(path, "canonical cutover policy", "E-SRB-CUTOVER-001")
    except Exception as error:
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-001", f"cannot read canonical cutover policy: {error}") from error
    return payload, raw


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], Path]:
    repo_root, workspace_root, destinations_root, repositories_root, receipt_path = resolve_common(args)
    if args.mode == "authorize" and (receipt_path.exists() or receipt_path.is_symlink()):
        raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-008", "cutover approval receipt already exists")
    lock_descriptor = acquire_execution_root_lock(repo_root)
    try:
        authorization, authorization_raw, materialization, materialization_raw, pre_files = reconstruct(
            args, repo_root, workspace_root, destinations_root
        )
        policy_path = absolute(args.cutover_policy)
        policy, policy_raw = read_policy(policy_path)
        canonical, destinations, recovery, operator = validate_policy(
            policy,
            repo_root,
            repositories_root,
            authorization,
            authorization_raw,
            materialization,
            materialization_raw,
        )
        confirmations = confirm_operation(args, policy, authorization)
        source_before = scan_repository(repo_root)
        rehearsal = run_rehearsal(repo_root, workspace_root, authorization, pre_files)
        if scan_repository(repo_root) != source_before:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-006", "rehearsal changed canonical source")

        # Re-read every permission-bearing input and every Git ref immediately before receipt construction.
        second = reconstruct(args, repo_root, workspace_root, destinations_root)
        if second[:4] != (authorization, authorization_raw, materialization, materialization_raw):
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-009", "source evidence changed during approval")
        policy_second, policy_raw_second = read_policy(policy_path)
        if policy_second != policy or policy_raw_second != policy_raw:
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-009", "cutover policy changed during approval")
        canonical_second, destinations_second, recovery_second, operator_second = validate_policy(
            policy_second,
            repo_root,
            repositories_root,
            authorization,
            authorization_raw,
            materialization,
            materialization_raw,
        )
        if (canonical_second, destinations_second, recovery_second, operator_second) != (
            canonical,
            destinations,
            recovery,
            operator,
        ):
            raise CanonicalCutoverApprovalError("E-SRB-CUTOVER-009", "cutover evidence changed during approval")
        receipt = build_receipt(
            policy,
            policy_raw,
            authorization,
            authorization_raw,
            materialization,
            materialization_raw,
            canonical,
            destinations,
            recovery,
            operator,
            confirmations,
            rehearsal,
        )
        if args.mode == "authorize":
            write_atomic(receipt_path, receipt)
        else:
            try:
                actual, _raw = read_json(
                    receipt_path, "canonical cutover approval receipt", "E-SRB-CUTOVER-009"
                )
            except Exception as error:
                raise CanonicalCutoverApprovalError(
                    "E-SRB-CUTOVER-009", f"cannot read cutover approval receipt: {error}"
                ) from error
            if actual.get("approval_identity_sha256") != approval_identity(actual) or actual != receipt:
                raise CanonicalCutoverApprovalError(
                    "E-SRB-CUTOVER-009", "cutover approval receipt differs from reconstructed evidence"
                )
        return receipt, receipt_path
    finally:
        release_execution_root_lock(lock_descriptor)


def command(args: argparse.Namespace) -> int:
    expected, _receipt_path = evaluate(args)
    if args.mode == "authorize":
        print(
            "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVED "
            f"approval_identity={expected['approval_identity_sha256']} "
            f"context={expected['approval_context']} status={APPROVAL_STATUS} execution={EXECUTION_STATUS}"
        )
        return 0
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_VERIFY_PASS "
        f"approval_identity={expected['approval_identity_sha256']} status={APPROVAL_STATUS} execution={EXECUTION_STATUS}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    subparsers = result.add_subparsers(dest="mode", required=True)
    for mode in ("authorize", "verify"):
        item = subparsers.add_parser(mode)
        item.add_argument("--repo-root", required=True)
        item.add_argument("--rings", required=True)
        item.add_argument("--ownership", required=True)
        item.add_argument("--inventory", required=True)
        item.add_argument("--destination-policy", required=True)
        item.add_argument("--destinations-root", required=True)
        item.add_argument("--materialization-receipt", required=True)
        item.add_argument("--removal-policy", required=True)
        item.add_argument("--authorization-receipt", required=True)
        item.add_argument("--repositories-root", required=True)
        item.add_argument("--cutover-policy", required=True)
        item.add_argument("--workspace-root", required=True)
        item.add_argument("--cutover-approval-receipt", required=True)
        item.add_argument("--confirm-authorization-identity", required=True)
        item.add_argument("--confirm-scope-identity", required=True)
        item.add_argument("--confirm-policy-identity", required=True)
        item.add_argument("--confirm-pre-cutover-tree", required=True)
        item.set_defaults(handler=command)
    return result


def main() -> int:
    args = parser().parse_args()
    try:
        return args.handler(args)
    except CanonicalCutoverApprovalError as error:
        print(
            f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_REFUSED {error.code}: {error}",
            file=sys.stderr,
        )
        return 1
    except SourceRemovalExecutionError as error:
        print(
            f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_REFUSED E-SRB-CUTOVER-001: {error}",
            file=sys.stderr,
        )
        return 1
    except SourceRemovalError as error:
        print(
            f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_REFUSED E-SRB-CUTOVER-001: {error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
