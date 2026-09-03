#!/usr/bin/env python3
"""Execute and verify one explicitly approved R3 canonical Git cutover."""

from __future__ import annotations

import argparse
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

import canonical_cutover_authorizer as cutover
from physical_extraction_inventory import canonical_json, sha256_bytes, within_root
import source_removal_executor as removal
from source_removal_authorizer import (
    SourceRemovalError,
    absolute,
    copy_verified,
    read_json,
    scan_repository,
    tree_identity,
)


POLICY_SCHEMA = "sounio.physical-extraction-canonical-cutover-execution-policy.v1"
POLICY_TYPE = "explicit-canonical-cutover-execution-policy"
POLICY_AUTHORITY = "exact-approved-git-cutover-execution"
RECEIPT_SCHEMA = "sounio.physical-extraction-canonical-cutover-execution.v1"
EXECUTION_TYPE = "policy-bound-canonical-git-cutover"
EXECUTION_AUTHORITY = "exact-approved-git-cutover-execution"
EXECUTION_STATUS = "executed-and-verified"
SOURCE_REMOVAL_STATUS = "executed"
ASSURANCE_LEVEL = "identity-plus-git-remote-ref-and-published-commit"
EXECUTION_CONTEXTS = {"disposable-fixture", "canonical-production"}

POLICY_LIMITATIONS = [
    "execution_is_limited_to_the_exact_approved_git_transition",
    "disposable_fixture_context_is_not_canonical_production_execution",
    "git_remote_ref_update_does_not_prove_hosting_administration_or_ownership",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "remote_ref_update_and_receipt_promotion_are_not_one_atomic_transaction",
    "crash_between_remote_update_and_receipt_promotion_requires_bound_manual_recovery",
    "requires_quiescent_nonparticipating_writers",
    "git_object_ids_and_json_hashes_are_change_detectors_not_independent_signatures",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "execution_is_limited_to_the_exact_approved_git_transition",
    "disposable_fixture_receipt_is_not_canonical_production_execution",
    "git_remote_ref_observation_does_not_prove_hosting_administration_or_ownership",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "remote_ref_update_and_receipt_promotion_were_not_one_atomic_transaction",
    "a_process_crash_between_remote_update_and_receipt_promotion_requires_bound_manual_recovery",
    "requires_quiescent_nonparticipating_writers",
    "regular_file_content_and_git_tree_identity_do_not_capture_all_filesystem_metadata",
    "git_object_ids_and_json_hashes_are_change_detectors_not_independent_signatures",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "post_execution_verification_uses_bound_receipts_not_removed_source_files",
]

POLICY_FIELDS = {
    "schema",
    "policy_type",
    "authority_scope",
    "execution_context",
    "approval_status",
    "source_bindings",
    "canonical_repository",
    "commit_plan",
    "execution_authorization",
    "limitations",
    "policy_identity_sha256",
}
SOURCE_BINDING_FIELDS = {
    "cutover_approval_file_sha256",
    "cutover_approval_identity_sha256",
    "cutover_policy_file_sha256",
    "cutover_policy_identity_sha256",
    "authorization_file_sha256",
    "authorization_identity_sha256",
    "materialization_file_sha256",
    "materialization_identity_sha256",
    "pre_cutover_tree_sha256",
    "authorized_post_cutover_tree_sha256",
    "removal_scope_identity_sha256",
    "destination_set_identity_sha256",
    "recovery_plan_identity_sha256",
}
CANONICAL_FIELDS = {
    "repository_id",
    "remote_name",
    "remote_url",
    "branch",
    "pre_cutover_head_oid",
    "pre_cutover_remote_head_oid",
    "expected_post_cutover_git_tree_oid",
    "expected_cutover_commit_oid",
}
COMMIT_PLAN_FIELDS = {
    "author_name",
    "author_email",
    "author_date",
    "committer_name",
    "committer_email",
    "committer_date",
    "message",
    "local_ref_update",
    "remote_ref_update",
}
EXECUTION_AUTHORIZATION_FIELDS = {
    "approved_by",
    "approval_evidence",
    "cutover_approval_identity_confirmation",
    "pre_cutover_head_confirmation",
    "pre_cutover_remote_head_confirmation",
    "authorized_post_cutover_tree_confirmation",
    "expected_cutover_commit_confirmation",
    "destination_set_identity_confirmation",
    "recovery_plan_identity_confirmation",
    "remote_update_reviewed",
    "recovery_plan_reviewed",
}
EVIDENCE_FIELDS = {"path", "size_bytes", "sha256"}
GIT_OID_RE = re.compile(r"^[0-9a-f]{40}$")
IDENTITY_RE = re.compile(r"^[0-9a-f]{64}$")
EMAIL_RE = re.compile(r"^[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9.-]+$")
GIT_DATE_RE = re.compile(r"^[0-9]{10} [+-][0-9]{4}$")


class CanonicalCutoverExecutionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def identity(payload: Any, field: str | None = None) -> str:
    value = json.loads(json.dumps(payload))
    if field is not None and isinstance(value, dict):
        value.pop(field, None)
    return sha256_bytes(canonical_json(value))


def policy_identity(payload: dict[str, Any]) -> str:
    return identity(payload, "policy_identity_sha256")


def execution_identity(payload: dict[str, Any]) -> str:
    return identity(payload, "execution_identity_sha256")


def git_environment(repository: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
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
    if extra:
        environment.update(extra)
    return environment


def git_command(
    repository: Path,
    arguments: list[str],
    label: str,
    *,
    environment: dict[str, str] | None = None,
    input_bytes: bytes | None = None,
    timeout: int = 60,
) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            env=git_environment(repository, environment),
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", f"cannot {label}: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", f"cannot {label}: {detail}")
    try:
        return result.stdout.decode("utf-8").rstrip("\n")
    except UnicodeError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", f"{label} returned non-UTF-8 output") from error


def git_object_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not GIT_OID_RE.fullmatch(value):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"{label} is not a Git SHA-1 object id")
    return value


def sha256_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or not IDENTITY_RE.fullmatch(value):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"{label} is not a SHA-256 identity")
    return value


def validate_commit_plan(item: Any) -> dict[str, str]:
    if not isinstance(item, dict) or set(item) != COMMIT_PLAN_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "commit plan fields are invalid")
    result: dict[str, str] = {}
    for key in ("author_name", "committer_name"):
        value = item.get(key)
        if (
            not isinstance(value, str)
            or not value
            or len(value) > 200
            or not value.isascii()
            or any(ord(character) < 32 for character in value)
        ):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"commit plan {key} is invalid")
        result[key] = value
    for key in ("author_email", "committer_email"):
        value = item.get(key)
        if not isinstance(value, str) or len(value) > 254 or not EMAIL_RE.fullmatch(value):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"commit plan {key} is invalid")
        result[key] = value
    for key in ("author_date", "committer_date"):
        value = item.get(key)
        if not isinstance(value, str) or not GIT_DATE_RE.fullmatch(value):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"commit plan {key} is invalid")
        result[key] = value
    message = item.get("message")
    if (
        not isinstance(message, str)
        or not message
        or len(message.encode("ascii", errors="ignore")) > 4096
        or not message.isascii()
        or "\x00" in message
        or "\r" in message
        or not message.endswith("\n")
    ):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "commit plan message is invalid")
    result["message"] = message
    if item.get("local_ref_update") != "compare-and-swap-pre-head-to-expected-commit":
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "local ref update strategy is invalid")
    if item.get("remote_ref_update") != "exact-force-with-lease-pre-head-to-expected-commit":
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "remote ref update strategy is invalid")
    result["local_ref_update"] = item["local_ref_update"]
    result["remote_ref_update"] = item["remote_ref_update"]
    return result


def commit_environment(plan: dict[str, str]) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": plan["author_name"],
        "GIT_AUTHOR_EMAIL": plan["author_email"],
        "GIT_AUTHOR_DATE": plan["author_date"],
        "GIT_COMMITTER_NAME": plan["committer_name"],
        "GIT_COMMITTER_EMAIL": plan["committer_email"],
        "GIT_COMMITTER_DATE": plan["committer_date"],
    }


def copy_with_mode(source: Path, destination: Path, expected: dict[str, Any]) -> int:
    try:
        mode = stat.S_IMODE(source.stat(follow_symlinks=False).st_mode)
    except OSError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", f"cannot inspect file mode: {source}: {error}") from error
    copy_verified(source, destination, expected)
    try:
        os.chmod(destination, mode, follow_symlinks=False)
    except OSError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", f"cannot preserve file mode: {destination}: {error}") from error
    return mode


def apply_cutover(
    source_root: Path,
    target_root: Path,
    authorization: dict[str, Any],
    *,
    copy_repairs: bool,
) -> None:
    for unit in authorization["removal_scope"]["units"]:
        target = target_root / PurePosixPath(unit["source_path"])
        if target.is_symlink() or not target.is_dir():
            raise CanonicalCutoverExecutionError(
                "E-SRB-CUTOVER-EXEC-005", f"authorized removal root is absent: {unit['source_path']}"
            )
        shutil.rmtree(target)
    for repair in authorization["repairs"]:
        target = target_root / PurePosixPath(repair["path"])
        replacement = source_root / PurePosixPath(repair["replacement_path"])
        if target.is_symlink() or not target.is_file():
            raise CanonicalCutoverExecutionError(
                "E-SRB-CUTOVER-EXEC-005", f"authorized repair target is absent: {repair['path']}"
            )
        target.unlink()
        expected = {"size_bytes": repair["after_size_bytes"], "sha256": repair["after_sha256"]}
        if copy_repairs:
            copy_with_mode(replacement, target, expected)
        else:
            copy_verified(replacement, target, expected)


def compute_expected_git_transition(
    repo_root: Path,
    workspace_root: Path,
    authorization: dict[str, Any],
    pre_files: list[dict[str, Any]],
    plan: dict[str, str],
    pre_head: str,
) -> tuple[str, str]:
    transaction = Path(tempfile.mkdtemp(prefix=".sounio-canonical-cutover-git-plan.", dir=workspace_root))
    candidate = transaction / "candidate"
    object_repository = transaction / "objects.git"
    candidate.mkdir()
    try:
        for item in pre_files:
            source = repo_root / PurePosixPath(item["path"])
            copy_with_mode(source, candidate / PurePosixPath(item["path"]), item)
        apply_cutover(repo_root, candidate, authorization, copy_repairs=True)
        expected_files = removal.expected_post_execution_files(pre_files, authorization)
        if scan_repository(candidate) != expected_files:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "planned post-cutover files differ from authorization")
        git_command(
            repo_root,
            ["clone", "--bare", "--no-hardlinks", "--", str(repo_root), str(object_repository)],
            "create isolated Git planning repository",
            timeout=120,
        )
        index = transaction / "index"
        alternate = {
            "GIT_DIR": str(object_repository),
            "GIT_WORK_TREE": str(candidate),
            "GIT_INDEX_FILE": str(index),
        }
        git_command(repo_root, ["read-tree", pre_head], "seed isolated planning index", environment=alternate)
        git_command(repo_root, ["add", "-A", "--", "."], "stage isolated cutover plan", environment=alternate)
        tree_oid = git_command(repo_root, ["write-tree"], "write isolated cutover tree", environment=alternate)
        commit_oid = git_command(
            repo_root,
            ["commit-tree", tree_oid, "-p", pre_head],
            "write isolated cutover commit",
            environment={**alternate, **commit_environment(plan)},
            input_bytes=plan["message"].encode("ascii"),
        )
        return git_object_id(tree_oid, "planned Git tree"), git_object_id(commit_oid, "planned Git commit")
    finally:
        shutil.rmtree(transaction, ignore_errors=True)


def expected_source_bindings(
    approval: dict[str, Any],
    approval_raw: bytes,
    cutover_policy: dict[str, Any],
    cutover_policy_raw: bytes,
) -> dict[str, str]:
    bindings = approval["source_bindings"]
    return {
        "cutover_approval_file_sha256": sha256_bytes(approval_raw),
        "cutover_approval_identity_sha256": approval["approval_identity_sha256"],
        "cutover_policy_file_sha256": sha256_bytes(cutover_policy_raw),
        "cutover_policy_identity_sha256": cutover_policy["policy_identity_sha256"],
        "authorization_file_sha256": bindings["authorization_file_sha256"],
        "authorization_identity_sha256": bindings["authorization_identity_sha256"],
        "materialization_file_sha256": bindings["materialization_file_sha256"],
        "materialization_identity_sha256": bindings["materialization_identity_sha256"],
        "pre_cutover_tree_sha256": bindings["pre_cutover_tree_sha256"],
        "authorized_post_cutover_tree_sha256": bindings["authorized_post_cutover_tree_sha256"],
        "removal_scope_identity_sha256": bindings["removal_scope_identity_sha256"],
        "destination_set_identity_sha256": identity(approval["destinations"]),
        "recovery_plan_identity_sha256": approval["recovery_plan"]["recovery_plan_identity_sha256"],
    }


def validate_evidence(
    repo_root: Path,
    item: Any,
    label: str,
    planned_roots: list[str],
) -> dict[str, Any]:
    if not isinstance(item, dict) or set(item) != EVIDENCE_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"{label} fields are invalid")
    try:
        return removal.validate_file_evidence(repo_root, item, label, planned_roots)
    except removal.SourceRemovalExecutionError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", str(error)) from error


def validate_execution_policy(
    payload: dict[str, Any],
    repo_root: Path,
    workspace_root: Path,
    approval: dict[str, Any],
    approval_raw: bytes,
    cutover_policy: dict[str, Any],
    cutover_policy_raw: bytes,
    authorization: dict[str, Any],
    pre_files: list[dict[str, Any]] | None,
) -> tuple[dict[str, Any], dict[str, str], dict[str, str]]:
    if set(payload) != POLICY_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution policy fields do not match v1")
    if payload.get("schema") != POLICY_SCHEMA or payload.get("policy_type") != POLICY_TYPE:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "unsupported canonical cutover execution policy")
    if payload.get("authority_scope") != POLICY_AUTHORITY or payload.get("limitations") != POLICY_LIMITATIONS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution policy authority or limitations differ from v1")
    if payload.get("policy_identity_sha256") != policy_identity(payload):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution policy identity mismatch")
    if payload.get("approval_status") != "approved":
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "execution policy is not approved")
    context = payload.get("execution_context")
    if context not in EXECUTION_CONTEXTS or context != approval.get("approval_context"):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "execution context differs from cutover approval")
    if approval.get("canonical_cutover_approval_status") != "approved-not-executed":
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", "cutover approval status is invalid")

    bindings = payload.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != SOURCE_BINDING_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution source bindings are invalid")
    expected_bindings = expected_source_bindings(approval, approval_raw, cutover_policy, cutover_policy_raw)
    if bindings != expected_bindings:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", "execution policy is bound to other approval evidence")

    canonical = payload.get("canonical_repository")
    if not isinstance(canonical, dict) or set(canonical) != CANONICAL_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "canonical execution repository fields are invalid")
    approved_canonical = approval["canonical_repository"]
    expected_canonical = {
        "repository_id": approved_canonical["repository_id"],
        "remote_name": approved_canonical["remote_name"],
        "remote_url": approved_canonical["remote_url"],
        "branch": approved_canonical["branch"],
        "pre_cutover_head_oid": approved_canonical["head_oid"],
        "pre_cutover_remote_head_oid": approved_canonical["remote_head_oid"],
    }
    for key, expected in expected_canonical.items():
        if canonical.get(key) != expected:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", f"canonical execution binding differs: {key}")
    pre_head = git_object_id(canonical.get("pre_cutover_head_oid"), "pre-cutover head")
    git_object_id(canonical.get("pre_cutover_remote_head_oid"), "pre-cutover remote head")
    expected_tree = git_object_id(canonical.get("expected_post_cutover_git_tree_oid"), "expected post-cutover Git tree")
    expected_commit = git_object_id(canonical.get("expected_cutover_commit_oid"), "expected cutover commit")
    if pre_head != canonical["pre_cutover_remote_head_oid"]:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", "pre-cutover local and remote heads differ")

    plan = validate_commit_plan(payload.get("commit_plan"))
    if pre_files is not None:
        actual_tree, actual_commit = compute_expected_git_transition(
            repo_root, workspace_root, authorization, pre_files, plan, pre_head
        )
        if (actual_tree, actual_commit) != (expected_tree, expected_commit):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-004", "execution policy Git transition differs from exact plan")

    execution_authorization = payload.get("execution_authorization")
    if not isinstance(execution_authorization, dict) or set(execution_authorization) != EXECUTION_AUTHORIZATION_FIELDS:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution authorization fields are invalid")
    approved_by = cutover.safe_token(execution_authorization.get("approved_by"), "execution operator label")
    planned_roots = [unit["source_path"] for unit in authorization["removal_scope"]["units"]]
    evidence_items = execution_authorization.get("approval_evidence")
    if not isinstance(evidence_items, list) or not evidence_items:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "execution approval evidence is empty")
    validated_evidence: list[dict[str, Any]] = []
    paths: set[str] = set()
    for index, item in enumerate(evidence_items):
        evidence = validate_evidence(repo_root, item, f"execution approval evidence {index}", planned_roots)
        if evidence["path"] in paths:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "execution approval evidence is duplicated")
        paths.add(evidence["path"])
        validated_evidence.append(evidence)
    expected_confirmations: dict[str, Any] = {
        "cutover_approval_identity_confirmation": approval["approval_identity_sha256"],
        "pre_cutover_head_confirmation": pre_head,
        "pre_cutover_remote_head_confirmation": canonical["pre_cutover_remote_head_oid"],
        "authorized_post_cutover_tree_confirmation": bindings["authorized_post_cutover_tree_sha256"],
        "expected_cutover_commit_confirmation": expected_commit,
        "destination_set_identity_confirmation": bindings["destination_set_identity_sha256"],
        "recovery_plan_identity_confirmation": bindings["recovery_plan_identity_sha256"],
        "remote_update_reviewed": True,
        "recovery_plan_reviewed": True,
    }
    if any(execution_authorization.get(key) != value for key, value in expected_confirmations.items()):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "execution operator confirmations differ from policy")
    validated_authorization = {
        **execution_authorization,
        "approved_by": approved_by,
        "approval_evidence": validated_evidence,
    }
    return validated_authorization, canonical, plan


def confirm_execution(
    args: argparse.Namespace,
    policy: dict[str, Any],
    approval: dict[str, Any],
    canonical: dict[str, str],
) -> dict[str, str]:
    expected = {
        "approval": approval["approval_identity_sha256"],
        "policy": policy["policy_identity_sha256"],
        "head": canonical["pre_cutover_head_oid"],
        "commit": canonical["expected_cutover_commit_oid"],
        "context": policy["execution_context"],
    }
    actual = {
        "approval": args.confirm_cutover_approval_identity,
        "policy": args.confirm_execution_policy_identity,
        "head": args.confirm_pre_cutover_head,
        "commit": args.confirm_expected_cutover_commit,
        "context": args.confirm_execution_context,
    }
    if actual != expected:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-004", "explicit execution confirmations differ from policy")
    return {
        "cutover_approval_identity_sha256": actual["approval"],
        "execution_policy_identity_sha256": actual["policy"],
        "pre_cutover_head_oid": actual["head"],
        "expected_cutover_commit_oid": actual["commit"],
        "execution_context": actual["context"],
        "confirmation_status": "matched",
    }


def verify_approval_locked(
    args: argparse.Namespace,
    repo_root: Path,
    workspace_root: Path,
    destinations_root: Path,
    repositories_root: Path,
) -> tuple[
    dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes, list[dict[str, Any]]
]:
    try:
        authorization, authorization_raw, materialization, materialization_raw, pre_files = cutover.reconstruct(
            args, repo_root, workspace_root, destinations_root
        )
        cutover_policy, cutover_policy_raw = cutover.read_policy(absolute(args.cutover_policy))
        canonical, destinations, recovery, operator = cutover.validate_policy(
            cutover_policy,
            repo_root,
            repositories_root,
            authorization,
            authorization_raw,
            materialization,
            materialization_raw,
        )
        confirmations = cutover.confirm_operation(args, cutover_policy, authorization)
        rehearsal = cutover.run_rehearsal(repo_root, workspace_root, authorization, pre_files)
        expected = cutover.build_receipt(
            cutover_policy,
            cutover_policy_raw,
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
        approval, approval_raw = read_json(
            absolute(args.cutover_approval_receipt),
            "canonical cutover approval receipt",
            "E-SRB-CUTOVER-EXEC-002",
        )
    except (cutover.CanonicalCutoverApprovalError, removal.SourceRemovalExecutionError, SourceRemovalError) as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", f"cutover approval reconstruction refused: {error}") from error
    if approval.get("approval_identity_sha256") != cutover.approval_identity(approval) or approval != expected:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", "cutover approval receipt differs from reconstructed evidence")
    return (
        approval,
        approval_raw,
        cutover_policy,
        cutover_policy_raw,
        authorization,
        authorization_raw,
        materialization,
        materialization_raw,
        pre_files,
    )


def verify_bound_receipts_after_execution(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes, dict[str, Any], bytes]:
    approval, approval_raw = read_json(
        absolute(args.cutover_approval_receipt), "canonical cutover approval receipt", "E-SRB-CUTOVER-EXEC-009"
    )
    if (
        approval.get("schema") != cutover.RECEIPT_SCHEMA
        or approval.get("approval_identity_sha256") != cutover.approval_identity(approval)
        or approval.get("canonical_cutover_approval_status") != "approved-not-executed"
        or approval.get("canonical_cutover_execution_status") != "not-executed"
    ):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "cutover approval receipt is invalid")
    cutover_policy, cutover_policy_raw = cutover.read_policy(absolute(args.cutover_policy))
    if (
        cutover_policy.get("policy_identity_sha256") != cutover.policy_identity(cutover_policy)
        or sha256_bytes(cutover_policy_raw) != approval["source_bindings"]["cutover_policy_file_sha256"]
        or cutover_policy["policy_identity_sha256"] != approval["source_bindings"]["cutover_policy_identity_sha256"]
    ):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "cutover policy differs from approval binding")
    authorization, authorization_raw = removal.read_authorization(absolute(args.authorization_receipt))
    materialization, materialization_raw = read_json(
        absolute(args.materialization_receipt), "materialization receipt", "E-SRB-CUTOVER-EXEC-009"
    )
    bindings = approval["source_bindings"]
    if (
        sha256_bytes(authorization_raw) != bindings["authorization_file_sha256"]
        or authorization.get("authorization_identity_sha256") != bindings["authorization_identity_sha256"]
        or sha256_bytes(materialization_raw) != bindings["materialization_file_sha256"]
        or materialization.get("materialization_identity_sha256") != bindings["materialization_identity_sha256"]
        or materialization.get("materialization_identity_sha256") != removal.materialization_identity(materialization)
    ):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "authorization or materialization receipt changed")
    return (
        approval,
        approval_raw,
        cutover_policy,
        cutover_policy_raw,
        authorization,
        authorization_raw,
        materialization,
        materialization_raw,
    )


def verify_retained_evidence(
    repo_root: Path,
    approval: dict[str, Any],
    execution_authorization: dict[str, Any],
    authorization: dict[str, Any],
) -> None:
    planned_roots = [unit["source_path"] for unit in authorization["removal_scope"]["units"]]
    validate_evidence(repo_root, approval["canonical_repository"]["retained_marker"], "canonical marker", planned_roots)
    validate_evidence(repo_root, approval["recovery_plan"]["plan_evidence"], "recovery plan", planned_roots)
    for index, item in enumerate(approval["operator_approval"]["approval_evidence"]):
        validate_evidence(repo_root, item, f"cutover operator evidence {index}", planned_roots)
    for index, item in enumerate(execution_authorization["approval_evidence"]):
        validate_evidence(repo_root, item, f"execution operator evidence {index}", planned_roots)
    for index, destination in enumerate(approval["destinations"]):
        validate_evidence(repo_root, destination["owner_approval_evidence"], f"destination owner evidence {index}", planned_roots)


def verify_destinations(
    repo_root: Path,
    repositories_root: Path,
    destinations_root: Path,
    materialization: dict[str, Any],
    materialization_raw: bytes,
    authorization: dict[str, Any],
    approval: dict[str, Any],
) -> list[dict[str, Any]]:
    try:
        removal.verify_materialized_copies(destinations_root, materialization, materialization_raw, authorization)
    except removal.SourceRemovalExecutionError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-006", str(error)) from error
    units = {unit["source_path"]: unit for unit in materialization["units"]}
    rows = approval.get("destinations")
    if not isinstance(rows, list) or set(units) != {row.get("source_path") for row in rows}:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-006", "destination coverage differs from approval")
    expected_checkouts = {row["checkout_path"] for row in rows}
    if {entry.name for entry in repositories_root.iterdir()} != expected_checkouts:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-006", "destination repository root has unexpected members")
    result = []
    for index, row in enumerate(sorted(rows, key=lambda item: item["source_path"])):
        checkout = repositories_root / row["checkout_path"]
        expected_git = {key: row[key] for key in cutover.GIT_FIELDS}
        try:
            git_state = cutover.verify_git_repository(checkout, expected_git, f"destination {index}")
        except cutover.CanonicalCutoverApprovalError as error:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-006", str(error)) from error
        unit = units[row["source_path"]]
        if scan_repository(checkout) != unit["files"]:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-006", f"destination content changed: {row['source_path']}")
        result.append(
            {
                "source_path": row["source_path"],
                "repository_id": row["repository_id"],
                "branch": row["branch"],
                "head_oid": git_state["head_oid"],
                "remote_head_oid": git_state["remote_head_oid"],
                "tree_sha256": row["tree_sha256"],
                "destination_status": "unchanged-head-equals-remote-and-content-verified",
            }
        )
    return result


def remote_head(repo_root: Path, remote_name: str, branch: str) -> str:
    output = git_command(
        repo_root,
        ["ls-remote", "--heads", remote_name, f"refs/heads/{branch}"],
        "inspect canonical remote branch",
    )
    lines = [line for line in output.splitlines() if line]
    expected_ref = f"refs/heads/{branch}"
    if len(lines) != 1:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", "canonical remote branch is absent or ambiguous")
    fields = lines[0].split("\t")
    if len(fields) != 2 or fields[1] != expected_ref:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", "canonical remote branch response is invalid")
    return git_object_id(fields[0], "canonical remote head")


def verify_pre_ref_state(repo_root: Path, canonical: dict[str, str]) -> None:
    branch = git_command(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"], "inspect canonical branch")
    head = git_command(repo_root, ["rev-parse", "HEAD"], "inspect canonical head")
    url = git_command(
        repo_root, ["remote", "get-url", canonical["remote_name"]], "inspect canonical remote URL"
    )
    observed_remote = remote_head(repo_root, canonical["remote_name"], canonical["branch"])
    if (
        branch != canonical["branch"]
        or head != canonical["pre_cutover_head_oid"]
        or url != canonical["remote_url"]
        or observed_remote != canonical["pre_cutover_remote_head_oid"]
    ):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-003", "canonical Git refs changed during execution")


def verify_post_git(repo_root: Path, canonical: dict[str, str]) -> dict[str, str]:
    expected = {
        "repository_id": canonical["repository_id"],
        "remote_name": canonical["remote_name"],
        "remote_url": canonical["remote_url"],
        "branch": canonical["branch"],
        "head_oid": canonical["expected_cutover_commit_oid"],
        "remote_head_oid": canonical["expected_cutover_commit_oid"],
    }
    try:
        state = cutover.verify_git_repository(repo_root, expected, "executed canonical repository")
    except cutover.CanonicalCutoverApprovalError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", str(error)) from error
    tree_oid = git_command(
        repo_root,
        ["rev-parse", f"{canonical['expected_cutover_commit_oid']}^{{tree}}"],
        "inspect executed commit tree",
    )
    parent_oid = git_command(
        repo_root,
        ["rev-parse", f"{canonical['expected_cutover_commit_oid']}^"],
        "inspect executed commit parent",
    )
    if tree_oid != canonical["expected_post_cutover_git_tree_oid"] or parent_oid != canonical["pre_cutover_head_oid"]:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "executed commit topology differs from policy")
    return state


def push_ref(repo_root: Path, canonical: dict[str, str], old_oid: str, new_oid: str, label: str) -> None:
    reference = f"refs/heads/{canonical['branch']}"
    git_command(
        repo_root,
        [
            "push",
            "--porcelain",
            "--no-verify",
            f"--force-with-lease={reference}:{old_oid}",
            canonical["remote_name"],
            f"{new_oid}:{reference}",
        ],
        label,
        timeout=120,
    )


def build_execution_receipt(
    policy: dict[str, Any],
    policy_raw: bytes,
    approval: dict[str, Any],
    approval_raw: bytes,
    cutover_policy: dict[str, Any],
    cutover_policy_raw: bytes,
    authorization: dict[str, Any],
    execution_authorization: dict[str, Any],
    confirmations: dict[str, str],
    canonical: dict[str, str],
    plan: dict[str, str],
    destinations: list[dict[str, Any]],
    gate_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "execution_type": EXECUTION_TYPE,
        "authority_scope": EXECUTION_AUTHORITY,
        "execution_context": policy["execution_context"],
        "canonical_cutover_approval_status": "consumed",
        "canonical_cutover_execution_status": EXECUTION_STATUS,
        "source_removal_status": SOURCE_REMOVAL_STATUS,
        "source_bindings": {
            **expected_source_bindings(approval, approval_raw, cutover_policy, cutover_policy_raw),
            "execution_policy_file_sha256": sha256_bytes(policy_raw),
            "execution_policy_identity_sha256": policy["policy_identity_sha256"],
        },
        "canonical_repository": {
            **canonical,
            "post_cutover_head_oid": canonical["expected_cutover_commit_oid"],
            "post_cutover_remote_head_oid": canonical["expected_cutover_commit_oid"],
            "worktree_status": "clean",
            "remote_ref_status": "head-equals-remote-branch",
            "publication_status": "exact-branch-ref-updated",
        },
        "commit_plan": plan,
        "execution_scope": authorization["removal_scope"],
        "summary": {
            "executed_unit_count": authorization["summary"]["authorized_unit_count"],
            "removed_file_count": authorization["summary"]["authorized_file_count"],
            "removed_total_bytes": authorization["summary"]["authorized_total_bytes"],
            "repair_count": authorization["summary"]["repair_count"],
            "post_removal_gate_count": len(gate_evidence),
            "destination_repository_count": len(destinations),
        },
        "execution_authorization": execution_authorization,
        "explicit_cli_confirmations": confirmations,
        "destinations": destinations,
        "recovery_plan": approval["recovery_plan"],
        "repairs": [
            {"path": repair["path"], "after_sha256": repair["after_sha256"], "execution_status": "applied-and-verified"}
            for repair in authorization["repairs"]
        ],
        "post_removal_gates": gate_evidence,
        "tree_evidence": {
            "pre_cutover_tree_sha256": approval["source_bindings"]["pre_cutover_tree_sha256"],
            "authorized_post_cutover_tree_sha256": approval["source_bindings"]["authorized_post_cutover_tree_sha256"],
            "post_cutover_git_tree_oid": canonical["expected_post_cutover_git_tree_oid"],
            "post_cutover_status": "exact-authorized-regular-file-and-git-tree",
        },
        "transaction_evidence": {
            "backup_type": "full-regular-file-and-mode-pre-cutover-copy",
            "local_ref_update": "compare-and-swap-pre-head-to-expected-commit",
            "remote_ref_update": "exact-force-with-lease-pre-head-to-expected-commit",
            "receipt_promotion": "atomic-hardlink-after-remote-ref-verification",
            "pre_receipt_failure_recovery": "exact-lease-remote-ref-rollback-then-local-tree-rollback",
            "distributed_atomicity": "not-guaranteed",
            "transaction_status": "committed",
        },
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": RECEIPT_LIMITATIONS,
    }
    payload["execution_identity_sha256"] = execution_identity(payload)
    return payload


def stage_json(path: Path, payload: dict[str, Any]) -> Path:
    if path.exists() or path.is_symlink():
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-008", f"execution receipt already exists: {path}")
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    stage = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        stage.unlink(missing_ok=True)
        raise
    return stage


def promote_stage(stage: Path, final: Path) -> None:
    if final.exists() or final.is_symlink():
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-008", "execution receipt appeared during operation")
    try:
        os.link(stage, final)
    except OSError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-008", f"cannot promote execution receipt: {error}") from error


def finish_stage(stage: Path, final: Path) -> None:
    try:
        stage.unlink(missing_ok=True)
        descriptor = os.open(final.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        print(f"warning: committed execution receipt cleanup or directory sync failed: {error}", file=sys.stderr)


def restore_transaction(
    repo_root: Path,
    backup: Path,
    pre_files: list[dict[str, Any]],
    modes: dict[str, int],
    canonical: dict[str, str],
) -> None:
    old_oid = canonical["pre_cutover_head_oid"]
    new_oid = canonical["expected_cutover_commit_oid"]
    observed_remote = remote_head(repo_root, canonical["remote_name"], canonical["branch"])
    if observed_remote == new_oid:
        push_ref(repo_root, canonical, new_oid, old_oid, "roll back canonical remote ref")
        if remote_head(repo_root, canonical["remote_name"], canonical["branch"]) != old_oid:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-007", "remote ref rollback did not restore pre-cutover head")
    elif observed_remote != old_oid:
        raise CanonicalCutoverExecutionError(
            "E-SRB-CUTOVER-EXEC-007", "remote ref is neither the pre-cutover nor expected cutover commit"
        )
    observed_local = git_command(repo_root, ["rev-parse", "HEAD"], "inspect local ref during rollback")
    if observed_local == new_oid:
        git_command(
            repo_root,
            ["update-ref", f"refs/heads/{canonical['branch']}", old_oid, new_oid],
            "roll back canonical local ref",
        )
    elif observed_local != old_oid:
        raise CanonicalCutoverExecutionError(
            "E-SRB-CUTOVER-EXEC-007", "local ref is neither the pre-cutover nor expected cutover commit"
        )
    removal.clear_repository(repo_root)
    for item in pre_files:
        destination = repo_root / PurePosixPath(item["path"])
        copy_verified(backup / PurePosixPath(item["path"]), destination, item)
        os.chmod(destination, modes[item["path"]], follow_symlinks=False)
    git_command(repo_root, ["read-tree", "--reset", old_oid], "restore canonical Git index")
    if scan_repository(repo_root) != pre_files:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-007", "rollback did not restore pre-cutover regular-file tree")
    expected = {
        "repository_id": canonical["repository_id"],
        "remote_name": canonical["remote_name"],
        "remote_url": canonical["remote_url"],
        "branch": canonical["branch"],
        "head_oid": old_oid,
        "remote_head_oid": old_oid,
    }
    try:
        cutover.verify_git_repository(repo_root, expected, "rolled-back canonical repository")
    except cutover.CanonicalCutoverApprovalError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-007", f"Git rollback verification failed: {error}") from error


def resolve_common(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = absolute(args.repo_root)
    workspace_root = absolute(args.workspace_root)
    destinations_root = absolute(args.destinations_root)
    repositories_root = absolute(args.repositories_root)
    for root, label in (
        (repo_root, "canonical root"),
        (workspace_root, "transaction workspace"),
        (destinations_root, "materialized destinations root"),
        (repositories_root, "destination repositories root"),
    ):
        if root.is_symlink() or not root.is_dir():
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"{label} must be a preexisting directory")
    roots = [root.resolve(strict=True) for root in (repo_root, workspace_root, destinations_root, repositories_root)]
    for index, first in enumerate(roots):
        for second in roots[index + 1 :]:
            if within_root(first, second) or within_root(second, first):
                raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "all execution roots must be separate")
    if repo_root.stat().st_dev != workspace_root.stat().st_dev:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "transaction workspace must share canonical-root filesystem")
    receipt = absolute(args.execution_receipt)
    try:
        receipt_parent = receipt.parent.resolve(strict=True)
    except OSError as error:
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", f"execution receipt parent is absent: {error}") from error
    if any(within_root(receipt_parent, root) for root in roots):
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-001", "execution receipt must remain outside all bound roots")
    return repo_root, workspace_root, destinations_root, repositories_root, receipt


def execute_command(args: argparse.Namespace) -> int:
    repo_root, workspace_root, destinations_root, repositories_root, receipt_path = resolve_common(args)
    if receipt_path.exists() or receipt_path.is_symlink():
        raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-008", "canonical cutover execution receipt already exists")
    lock_descriptor = removal.acquire_execution_root_lock(repo_root)
    transaction: Path | None = None
    backup: Path | None = None
    stage: Path | None = None
    receipt_promoted = False
    mutation_started = False
    pre_files: list[dict[str, Any]] = []
    modes: dict[str, int] = {}
    canonical: dict[str, str] = {}
    try:
        first = verify_approval_locked(args, repo_root, workspace_root, destinations_root, repositories_root)
        (
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            _authorization_raw,
            materialization,
            materialization_raw,
            pre_files,
        ) = first
        execution_policy_path = absolute(args.execution_policy)
        execution_policy, execution_policy_raw = read_json(
            execution_policy_path, "canonical cutover execution policy", "E-SRB-CUTOVER-EXEC-001"
        )
        execution_authorization, canonical, plan = validate_execution_policy(
            execution_policy,
            repo_root,
            workspace_root,
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            pre_files,
        )
        confirmations = confirm_execution(args, execution_policy, approval, canonical)
        verify_retained_evidence(repo_root, approval, execution_authorization, authorization)
        destination_evidence = verify_destinations(
            repo_root,
            repositories_root,
            destinations_root,
            materialization,
            materialization_raw,
            authorization,
            approval,
        )

        transaction = Path(
            tempfile.mkdtemp(
                prefix=f".sounio-canonical-cutover-execution.{execution_policy['policy_identity_sha256'][:12]}.",
                dir=workspace_root,
            )
        )
        transaction.chmod(0o700)
        backup = transaction / "backup"
        backup.mkdir()
        for item in pre_files:
            source = repo_root / PurePosixPath(item["path"])
            modes[item["path"]] = copy_with_mode(source, backup / PurePosixPath(item["path"]), item)
        if scan_repository(backup) != pre_files or scan_repository(repo_root) != pre_files:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "transaction backup or canonical root changed")

        second = verify_approval_locked(args, repo_root, workspace_root, destinations_root, repositories_root)
        if second != first:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", "approval evidence changed after transaction backup")
        policy_second, policy_raw_second = read_json(
            execution_policy_path, "canonical cutover execution policy", "E-SRB-CUTOVER-EXEC-001"
        )
        second_authorization, second_canonical, second_plan = validate_execution_policy(
            policy_second,
            repo_root,
            workspace_root,
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            pre_files,
        )
        if (
            policy_second != execution_policy
            or policy_raw_second != execution_policy_raw
            or second_authorization != execution_authorization
            or second_canonical != canonical
            or second_plan != plan
        ):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-002", "execution policy changed after transaction backup")

        mutation_started = True
        apply_cutover(repo_root, repo_root, authorization, copy_repairs=True)
        expected_files = removal.expected_post_execution_files(pre_files, authorization)
        if scan_repository(repo_root) != expected_files or tree_identity(expected_files) != approval["source_bindings"]["authorized_post_cutover_tree_sha256"]:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "canonical tree differs after approved cutover apply")
        try:
            gate_evidence = removal.run_execution_gates(repo_root, workspace_root, authorization)
        except removal.SourceRemovalExecutionError as error:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", str(error)) from error
        if scan_repository(repo_root) != expected_files:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "post-removal gates changed canonical tree")
        verify_pre_ref_state(repo_root, canonical)
        destination_evidence = verify_destinations(
            repo_root,
            repositories_root,
            destinations_root,
            materialization,
            materialization_raw,
            authorization,
            approval,
        )
        verify_retained_evidence(repo_root, approval, execution_authorization, authorization)

        git_command(repo_root, ["add", "-A", "--", "."], "stage canonical cutover")
        tree_oid = git_command(repo_root, ["write-tree"], "write canonical cutover tree")
        commit_oid = git_command(
            repo_root,
            ["commit-tree", tree_oid, "-p", canonical["pre_cutover_head_oid"]],
            "write canonical cutover commit",
            environment=commit_environment(plan),
            input_bytes=plan["message"].encode("ascii"),
        )
        if (
            tree_oid != canonical["expected_post_cutover_git_tree_oid"]
            or commit_oid != canonical["expected_cutover_commit_oid"]
        ):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "actual Git transition differs from execution policy")
        expected_receipt = build_execution_receipt(
            execution_policy,
            execution_policy_raw,
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            execution_authorization,
            confirmations,
            canonical,
            plan,
            destination_evidence,
            gate_evidence,
        )
        stage = stage_json(receipt_path, expected_receipt)

        git_command(
            repo_root,
            [
                "update-ref",
                f"refs/heads/{canonical['branch']}",
                canonical["expected_cutover_commit_oid"],
                canonical["pre_cutover_head_oid"],
            ],
            "advance canonical local ref",
        )
        if git_command(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"], "inspect cutover worktree"):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-005", "canonical worktree is not clean after local commit")
        push_ref(
            repo_root,
            canonical,
            canonical["pre_cutover_remote_head_oid"],
            canonical["expected_cutover_commit_oid"],
            "publish canonical cutover ref",
        )
        verify_post_git(repo_root, canonical)
        verify_destinations(
            repo_root,
            repositories_root,
            destinations_root,
            materialization,
            materialization_raw,
            authorization,
            approval,
        )
        verify_retained_evidence(repo_root, approval, execution_authorization, authorization)
        promote_stage(stage, receipt_path)
        receipt_promoted = True
        finish_stage(stage, receipt_path)
        stage = None
        try:
            shutil.rmtree(transaction)
            transaction = None
        except OSError as cleanup_error:
            print(f"warning: committed cutover transaction cleanup requires manual removal: {cleanup_error}", file=sys.stderr)
    except Exception as error:
        if stage is not None:
            stage.unlink(missing_ok=True)
        if not receipt_promoted and mutation_started and backup is not None and backup.is_dir() and pre_files and canonical:
            try:
                restore_transaction(
                    repo_root,
                    backup,
                    pre_files,
                    modes,
                    canonical,
                )
            except Exception as rollback_error:
                raise CanonicalCutoverExecutionError(
                    "E-SRB-CUTOVER-EXEC-007",
                    f"cutover failed ({error}); exact rollback failed ({rollback_error}); transaction retained at {transaction}",
                ) from error
        if transaction is not None and transaction.exists():
            shutil.rmtree(transaction, ignore_errors=True)
        raise
    finally:
        removal.release_execution_root_lock(lock_descriptor)

    print(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_PASS "
        f"receipt={receipt_path} commit={canonical['expected_cutover_commit_oid']} "
        f"context={expected_receipt['execution_context']} status={EXECUTION_STATUS}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    repo_root, workspace_root, destinations_root, repositories_root, receipt_path = resolve_common(args)
    lock_descriptor = removal.acquire_execution_root_lock(repo_root)
    try:
        (
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            _authorization_raw,
            materialization,
            materialization_raw,
        ) = verify_bound_receipts_after_execution(args)
        execution_policy, execution_policy_raw = read_json(
            absolute(args.execution_policy), "canonical cutover execution policy", "E-SRB-CUTOVER-EXEC-009"
        )
        execution_authorization, canonical, plan = validate_execution_policy(
            execution_policy,
            repo_root,
            workspace_root,
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            None,
        )
        confirmations = confirm_execution(args, execution_policy, approval, canonical)
        verify_post_git(repo_root, canonical)
        files = scan_repository(repo_root)
        if tree_identity(files) != approval["source_bindings"]["authorized_post_cutover_tree_sha256"]:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "post-cutover regular-file tree differs")
        for unit in authorization["removal_scope"]["units"]:
            path = repo_root / PurePosixPath(unit["source_path"])
            if path.exists() or path.is_symlink():
                raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", f"removed source root is present: {unit['source_path']}")
        try:
            gate_evidence = removal.run_verification_gates(repo_root, workspace_root, authorization, files)
        except removal.SourceRemovalExecutionError as error:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", str(error)) from error
        if scan_repository(repo_root) != files:
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "verification changed executed canonical tree")
        verify_retained_evidence(repo_root, approval, execution_authorization, authorization)
        destination_evidence = verify_destinations(
            repo_root,
            repositories_root,
            destinations_root,
            materialization,
            materialization_raw,
            authorization,
            approval,
        )
        expected = build_execution_receipt(
            execution_policy,
            execution_policy_raw,
            approval,
            approval_raw,
            cutover_policy,
            cutover_policy_raw,
            authorization,
            execution_authorization,
            confirmations,
            canonical,
            plan,
            destination_evidence,
            gate_evidence,
        )
        actual, _actual_raw = read_json(receipt_path, "canonical cutover execution receipt", "E-SRB-CUTOVER-EXEC-009")
        if (
            actual.get("schema") != RECEIPT_SCHEMA
            or actual.get("execution_identity_sha256") != execution_identity(actual)
            or actual != expected
        ):
            raise CanonicalCutoverExecutionError("E-SRB-CUTOVER-EXEC-009", "execution receipt differs from verified state")
    finally:
        removal.release_execution_root_lock(lock_descriptor)
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_VERIFY_PASS "
        f"receipt={receipt_path} commit={canonical['expected_cutover_commit_oid']} status={EXECUTION_STATUS}"
    )
    return 0


def add_common_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--repo-root", required=True)
    command.add_argument("--destinations-root", required=True)
    command.add_argument("--materialization-receipt", required=True)
    command.add_argument("--authorization-receipt", required=True)
    command.add_argument("--repositories-root", required=True)
    command.add_argument("--cutover-policy", required=True)
    command.add_argument("--cutover-approval-receipt", required=True)
    command.add_argument("--execution-policy", required=True)
    command.add_argument("--workspace-root", required=True)
    command.add_argument("--execution-receipt", required=True)
    command.add_argument("--confirm-authorization-identity", required=True)
    command.add_argument("--confirm-scope-identity", required=True)
    command.add_argument("--confirm-policy-identity", required=True)
    command.add_argument("--confirm-pre-cutover-tree", required=True)
    command.add_argument("--confirm-cutover-approval-identity", required=True)
    command.add_argument("--confirm-execution-policy-identity", required=True)
    command.add_argument("--confirm-pre-cutover-head", required=True)
    command.add_argument("--confirm-expected-cutover-commit", required=True)
    command.add_argument("--confirm-execution-context", choices=sorted(EXECUTION_CONTEXTS), required=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-physical-extraction-canonical-cutover-executor")
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
    except CanonicalCutoverExecutionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (cutover.CanonicalCutoverApprovalError, removal.SourceRemovalExecutionError, SourceRemovalError) as error:
        print(f"error[E-SRB-CUTOVER-EXEC-009]: dependent evidence refused: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error[E-SRB-CUTOVER-EXEC-009]: canonical cutover execution failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
