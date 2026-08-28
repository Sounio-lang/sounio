#!/usr/bin/env python3
"""Emit and verify non-authorizing canonical-production gap assessments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "science_boundary"))
import physical_extraction_inventory as inventory_tool  # noqa: E402


CATALOG_SCHEMA = "sounio.physical-extraction-canonical-production-repository-catalog.v1"
PROPOSAL_SCHEMA = "sounio.physical-extraction-canonical-production-mapping-proposal.v1"
ASSESSMENT_SCHEMA = "sounio.physical-extraction-canonical-production-gap-assessment.v1"
CATALOG_TYPE = "observed-hosting-repository-catalog"
PROPOSAL_TYPE = "target-repository-mapping-proposal"
ASSESSMENT_TYPE = "non-authorizing-production-prerequisite-gap-report"
AUTHORITY_SCOPE = "prerequisite-observation-only"
PROPOSAL_STATUS = "proposed-not-approved"
EXECUTION_AUTHORITY = "none"
CUTOVER_STATUS = "not-executed"
ASSURANCE_LEVEL = "identity-plus-supplied-catalog-observation"
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
GIT_OID = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
UTC_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
VISIBILITIES = {"PUBLIC", "PRIVATE", "INTERNAL"}
PERMISSIONS = {"ADMIN", "MAINTAIN", "WRITE", "TRIAGE", "READ", "NONE", "UNKNOWN"}
CATALOG_LIMITATIONS = [
    "catalog_is_a_supplied_point_in_time_observation",
    "catalog_does_not_prove_repository_ownership_or_administration",
    "catalog_does_not_prove_branch_protection_or_push_acceptance",
    "catalog_does_not_approve_target_mapping_or_source_removal",
    "catalog_does_not_assert_scientific_truth",
]
PROPOSAL_LIMITATIONS = [
    "proposal_does_not_create_or_modify_repositories",
    "proposal_is_not_destination_owner_approval",
    "proposal_is_not_canonical_production_approval",
    "proposal_does_not_authorize_source_removal_or_ref_updates",
    "proposal_does_not_assert_scientific_truth",
]
ASSESSMENT_LIMITATIONS = [
    "assessment_never_grants_execution_authority",
    "assessment_does_not_create_or_modify_repositories",
    "assessment_does_not_materialize_or_remove_source_files",
    "assessment_does_not_create_or_update_git_refs",
    "supplied_catalog_is_not_live_hosting_attestation",
    "observed_permission_does_not_prove_human_or_organizational_authority",
    "mapping_proposal_is_not_destination_owner_approval",
    "production_evidence_and_explicit_human_decision_remain_separate",
    "assessment_does_not_provide_atomic_snapshot_across_inventory_and_git_observation",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]


class ProductionGapError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def identity(payload: dict[str, Any], field: str) -> str:
    value = json.loads(json.dumps(payload))
    value.pop(field, None)
    return sha256_bytes(canonical_json(value))


def with_identity(payload: dict[str, Any], field: str) -> dict[str, Any]:
    value = json.loads(json.dumps(payload))
    value[field] = identity(value, field)
    return value


def exact_keys(value: dict[str, Any], expected: set[str], label: str, code: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ProductionGapError(
            code,
            f"{label} fields mismatch missing={','.join(sorted(expected - actual)) or '-'} "
            f"extra={','.join(sorted(actual - expected)) or '-'}",
        )


def read_regular_json(path: Path, label: str, code: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise ProductionGapError(code, f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionGapError(code, f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProductionGapError(code, f"{label} must be a JSON object")
    return value, sha256_bytes(raw)


def valid_url(value: Any) -> bool:
    return isinstance(value, str) and 1 <= len(value) <= 2048 and not any(char.isspace() for char in value)


def validate_catalog(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    code = "E-SRB-PROD-GAP-002"
    exact_keys(
        payload,
        {
            "schema",
            "catalog_type",
            "authority_scope",
            "organization",
            "observed_at_utc",
            "repositories",
            "limitations",
            "catalog_identity_sha256",
        },
        "repository catalog",
        code,
    )
    if payload["schema"] != CATALOG_SCHEMA or payload["catalog_type"] != CATALOG_TYPE:
        raise ProductionGapError(code, "unsupported repository catalog contract")
    if payload["authority_scope"] != "supplied-repository-metadata-observation":
        raise ProductionGapError(code, "repository catalog authority scope is invalid")
    if not isinstance(payload["organization"], str) or not SAFE_ID.fullmatch(payload["organization"]):
        raise ProductionGapError(code, "repository catalog organization is invalid")
    if not isinstance(payload["observed_at_utc"], str) or not UTC_TIMESTAMP.fullmatch(payload["observed_at_utc"]):
        raise ProductionGapError(code, "repository catalog timestamp is not canonical UTC")
    if payload["limitations"] != CATALOG_LIMITATIONS:
        raise ProductionGapError(code, "repository catalog limitations mismatch")
    if not isinstance(payload["catalog_identity_sha256"], str) or not SHA256.fullmatch(
        payload["catalog_identity_sha256"]
    ):
        raise ProductionGapError(code, "repository catalog identity is invalid")
    if payload["catalog_identity_sha256"] != identity(payload, "catalog_identity_sha256"):
        raise ProductionGapError(code, "repository catalog identity hash mismatch")
    rows = payload["repositories"]
    if not isinstance(rows, list) or not rows:
        raise ProductionGapError(code, "repository catalog must contain at least one repository")
    expected_fields = {
        "repository_id",
        "name_with_owner",
        "remote_url",
        "default_branch",
        "head_oid",
        "visibility",
        "archived",
        "is_empty",
        "observed_permission",
    }
    result: dict[str, dict[str, Any]] = {}
    remote_urls: set[str] = set()
    names: set[str] = set()
    previous = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProductionGapError(code, f"repository catalog row {index} is not an object")
        exact_keys(row, expected_fields, f"repository catalog row {index}", code)
        repository_id = row["repository_id"]
        if not isinstance(repository_id, str) or not SAFE_TOKEN.fullmatch(repository_id):
            raise ProductionGapError(code, f"repository catalog row {index} repository_id is invalid")
        if repository_id <= previous:
            raise ProductionGapError(code, "repository catalog rows must be strictly sorted by repository_id")
        previous = repository_id
        if repository_id in result:
            raise ProductionGapError(code, f"duplicate repository_id: {repository_id}")
        if not isinstance(row["name_with_owner"], str) or not SAFE_ID.fullmatch(row["name_with_owner"]):
            raise ProductionGapError(code, f"repository catalog row {index} name_with_owner is invalid")
        if not row["name_with_owner"].startswith(f"{payload['organization']}/"):
            raise ProductionGapError(code, f"repository catalog row {index} is outside the declared organization")
        if row["name_with_owner"] in names:
            raise ProductionGapError(code, f"duplicate name_with_owner: {row['name_with_owner']}")
        names.add(row["name_with_owner"])
        if not valid_url(row["remote_url"]) or row["remote_url"] in remote_urls:
            raise ProductionGapError(code, f"repository catalog row {index} remote_url is invalid or duplicate")
        remote_urls.add(row["remote_url"])
        if not isinstance(row["default_branch"], str) or not SAFE_ID.fullmatch(row["default_branch"]):
            raise ProductionGapError(code, f"repository catalog row {index} default_branch is invalid")
        if not isinstance(row["archived"], bool) or not isinstance(row["is_empty"], bool):
            raise ProductionGapError(code, f"repository catalog row {index} flags are invalid")
        if row["is_empty"]:
            if row["head_oid"] is not None:
                raise ProductionGapError(code, f"empty repository {repository_id} has a head object")
        elif not isinstance(row["head_oid"], str) or not GIT_OID.fullmatch(row["head_oid"]):
            raise ProductionGapError(code, f"repository {repository_id} head_oid is invalid")
        if row["visibility"] not in VISIBILITIES or row["observed_permission"] not in PERMISSIONS:
            raise ProductionGapError(code, f"repository catalog row {index} enum is invalid")
        result[repository_id] = row
    return result


def validate_proposal(
    payload: dict[str, Any],
    catalog_identity: str,
    expected_targets: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    code = "E-SRB-PROD-GAP-003"
    exact_keys(
        payload,
        {
            "schema",
            "proposal_type",
            "authority_scope",
            "proposal_status",
            "catalog_identity_sha256",
            "canonical_repository_id",
            "mappings",
            "limitations",
            "proposal_identity_sha256",
        },
        "mapping proposal",
        code,
    )
    if payload["schema"] != PROPOSAL_SCHEMA or payload["proposal_type"] != PROPOSAL_TYPE:
        raise ProductionGapError(code, "unsupported mapping proposal contract")
    if payload["authority_scope"] != "repository-target-proposal-only":
        raise ProductionGapError(code, "mapping proposal authority scope is invalid")
    if payload["proposal_status"] != PROPOSAL_STATUS:
        raise ProductionGapError(code, "mapping proposal must remain proposed-not-approved")
    if payload["catalog_identity_sha256"] != catalog_identity:
        raise ProductionGapError(code, "mapping proposal catalog identity mismatch")
    if not isinstance(payload["canonical_repository_id"], str) or not SAFE_TOKEN.fullmatch(
        payload["canonical_repository_id"]
    ):
        raise ProductionGapError(code, "mapping proposal canonical repository ID is invalid")
    if payload["limitations"] != PROPOSAL_LIMITATIONS:
        raise ProductionGapError(code, "mapping proposal limitations mismatch")
    if not isinstance(payload["proposal_identity_sha256"], str) or payload[
        "proposal_identity_sha256"
    ] != identity(payload, "proposal_identity_sha256"):
        raise ProductionGapError(code, "mapping proposal identity hash mismatch")
    mappings = payload["mappings"]
    if not isinstance(mappings, list) or not mappings:
        raise ProductionGapError(code, "mapping proposal must contain mappings")
    expected_fields = {
        "target_id",
        "target_owner",
        "repository_id",
        "remote_url",
        "branch",
        "expected_head_oid",
        "mapping_status",
    }
    result: dict[str, dict[str, Any]] = {}
    repositories: set[str] = set()
    previous = ""
    for index, row in enumerate(mappings):
        if not isinstance(row, dict):
            raise ProductionGapError(code, f"mapping proposal row {index} is not an object")
        exact_keys(row, expected_fields, f"mapping proposal row {index}", code)
        target_id = row["target_id"]
        if not isinstance(target_id, str) or target_id not in expected_targets:
            raise ProductionGapError(code, f"mapping proposal row {index} target is not planned")
        if target_id <= previous:
            raise ProductionGapError(code, "mapping proposal rows must be strictly sorted by target_id")
        previous = target_id
        if row["target_owner"] != expected_targets[target_id]["target_owner"]:
            raise ProductionGapError(code, f"mapping proposal target owner mismatch for {target_id}")
        repository_id = row["repository_id"]
        if not isinstance(repository_id, str) or not SAFE_TOKEN.fullmatch(repository_id):
            raise ProductionGapError(code, f"mapping proposal repository_id is invalid for {target_id}")
        if repository_id in repositories:
            raise ProductionGapError(code, f"mapping proposal reuses repository {repository_id}")
        repositories.add(repository_id)
        if not valid_url(row["remote_url"]):
            raise ProductionGapError(code, f"mapping proposal remote URL is invalid for {target_id}")
        if not isinstance(row["branch"], str) or not SAFE_ID.fullmatch(row["branch"]):
            raise ProductionGapError(code, f"mapping proposal branch is invalid for {target_id}")
        if not isinstance(row["expected_head_oid"], str) or not GIT_OID.fullmatch(row["expected_head_oid"]):
            raise ProductionGapError(code, f"mapping proposal head is invalid for {target_id}")
        if row["mapping_status"] != PROPOSAL_STATUS:
            raise ProductionGapError(code, f"mapping proposal row {target_id} overstates status")
        result[target_id] = row
    missing = sorted(set(expected_targets) - set(result))
    extra = sorted(set(result) - set(expected_targets))
    if missing or extra:
        raise ProductionGapError(
            code,
            f"mapping proposal coverage mismatch missing={','.join(missing) or '-'} extra={','.join(extra) or '-'}",
        )
    return result


def run_git(repo_root: Path, arguments: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=repo_root,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0", "LANG": "C", "LC_ALL": "C"},
            text=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ProductionGapError("E-SRB-PROD-GAP-004", f"Git observation failed: {error}") from error
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise ProductionGapError("E-SRB-PROD-GAP-004", f"Git observation failed: {detail}")
    return result.stdout.strip()


def git_observation(repo_root: Path, remote_name: str) -> dict[str, Any]:
    top = Path(run_git(repo_root, ["rev-parse", "--show-toplevel"])).resolve(strict=True)
    if top != repo_root:
        raise ProductionGapError("E-SRB-PROD-GAP-004", "repo root is not the Git worktree root")
    branch = run_git(repo_root, ["symbolic-ref", "--quiet", "--short", "HEAD"])
    head = run_git(repo_root, ["rev-parse", "HEAD"])
    if not SAFE_ID.fullmatch(branch) or not GIT_OID.fullmatch(head):
        raise ProductionGapError("E-SRB-PROD-GAP-004", "Git branch or head is invalid")
    remote_url = run_git(repo_root, ["remote", "get-url", remote_name])
    if not valid_url(remote_url):
        raise ProductionGapError("E-SRB-PROD-GAP-004", "Git remote URL is invalid")
    status = run_git(repo_root, ["status", "--porcelain=v1", "--untracked-files=all"])
    return {
        "remote_name": remote_name,
        "remote_url": remote_url,
        "branch": branch,
        "head_oid": head,
        "worktree_status": "clean" if not status else "dirty",
    }


def canonical_repository_assessment(
    observation: dict[str, Any],
    catalog_row: dict[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    issues: list[str] = []
    if observation["worktree_status"] != "clean":
        issues.append("canonical-worktree-dirty")
    if catalog_row is None:
        issues.append("canonical-repository-absent-from-catalog")
        catalog_head = None
        catalog_branch = None
        catalog_url = None
    else:
        catalog_head = catalog_row["head_oid"]
        catalog_branch = catalog_row["default_branch"]
        catalog_url = catalog_row["remote_url"]
        if catalog_row["archived"]:
            issues.append("canonical-repository-archived")
        if catalog_row["is_empty"]:
            issues.append("canonical-repository-empty")
        if observation["remote_url"] != catalog_url:
            issues.append("canonical-remote-url-mismatch")
        if observation["branch"] != catalog_branch:
            issues.append("canonical-branch-not-observed-default")
        if observation["head_oid"] != catalog_head:
            issues.append("canonical-head-not-observed-default-head")
    return (
        {
            **observation,
            "catalog_remote_url": catalog_url,
            "catalog_default_branch": catalog_branch,
            "catalog_head_oid": catalog_head,
            "assessment_status": "observed-matching-clean-default" if not issues else "prerequisite-gap",
            "gap_codes": issues,
        },
        not issues,
    )


def planned_targets(inventory: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for unit in inventory["units"]:
        if unit["disposition"] != "extract-planned":
            continue
        target_id = unit["target_id"]
        if target_id in result:
            raise ProductionGapError("E-SRB-PROD-GAP-001", f"duplicate planned target: {target_id}")
        result[target_id] = {
            "source_path": unit["source_path"],
            "ring": unit["ring"],
            "target_id": target_id,
            "target_owner": unit["target_owner"],
            "file_count": unit["file_count"],
            "total_bytes": unit["total_bytes"],
            "tree_sha256": unit["tree_sha256"],
        }
    if not result:
        raise ProductionGapError("E-SRB-PROD-GAP-001", "inventory has no extract-planned targets")
    return result


def destination_assessments(
    targets: dict[str, dict[str, Any]],
    proposal: dict[str, dict[str, Any]] | None,
    catalog: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    result: list[dict[str, Any]] = []
    available = 0
    for target_id in sorted(targets):
        target = targets[target_id]
        mapping = None if proposal is None else proposal[target_id]
        issues: list[str] = []
        repository_id: str | None = None
        remote_url: str | None = None
        branch: str | None = None
        head_oid: str | None = None
        if mapping is None:
            issues.append("target-repository-mapping-absent")
        else:
            repository_id = mapping["repository_id"]
            remote_url = mapping["remote_url"]
            branch = mapping["branch"]
            head_oid = mapping["expected_head_oid"]
            catalog_row = catalog.get(repository_id)
            if catalog_row is None:
                issues.append("mapped-repository-absent-from-catalog")
            else:
                if catalog_row["remote_url"] != remote_url:
                    issues.append("mapped-repository-remote-url-mismatch")
                if catalog_row["default_branch"] != branch:
                    issues.append("mapped-repository-branch-mismatch")
                if catalog_row["head_oid"] != head_oid:
                    issues.append("mapped-repository-head-mismatch")
                if catalog_row["archived"]:
                    issues.append("mapped-repository-archived")
                if catalog_row["is_empty"]:
                    issues.append("mapped-repository-empty")
                if catalog_row["observed_permission"] not in {"ADMIN", "MAINTAIN", "WRITE"}:
                    issues.append("mapped-repository-insufficient-observed-permission")
        if not issues:
            available += 1
        result.append(
            {
                **target,
                "repository_id": repository_id,
                "remote_url": remote_url,
                "branch": branch,
                "head_oid": head_oid,
                "assessment_status": "observed-available-proposal-not-approved" if not issues else "prerequisite-gap",
                "gap_codes": issues,
            }
        )
    return result, available


def prerequisite(identifier: str, status: str, detail_code: str) -> dict[str, str]:
    return {"prerequisite_id": identifier, "status": status, "detail_code": detail_code}


def expected_assessment(
    repo_root: Path,
    rings: Path,
    ownership: Path,
    catalog_path: Path,
    proposal_path: Path | None,
    canonical_repository_id: str,
    remote_name: str,
) -> dict[str, Any]:
    catalog_payload, catalog_file_sha = read_regular_json(
        catalog_path, "repository catalog", "E-SRB-PROD-GAP-002"
    )
    catalog = validate_catalog(catalog_payload)
    inventory = inventory_tool.expected_inventory(repo_root, rings, ownership)
    targets = planned_targets(inventory)
    proposal_payload: dict[str, Any] | None = None
    proposal: dict[str, dict[str, Any]] | None = None
    proposal_file_sha: str | None = None
    if proposal_path is not None:
        proposal_payload, proposal_file_sha = read_regular_json(
            proposal_path, "mapping proposal", "E-SRB-PROD-GAP-003"
        )
        proposal = validate_proposal(proposal_payload, catalog_payload["catalog_identity_sha256"], targets)
        if proposal_payload["canonical_repository_id"] != canonical_repository_id:
            raise ProductionGapError("E-SRB-PROD-GAP-003", "mapping proposal canonical repository mismatch")
    observation = git_observation(repo_root, remote_name)
    canonical, canonical_ready = canonical_repository_assessment(
        observation, catalog.get(canonical_repository_id)
    )
    destinations, available_count = destination_assessments(targets, proposal, catalog)
    target_count = len(targets)
    proposal_present = proposal is not None
    destinations_ready = proposal_present and available_count == target_count
    prerequisites = [
        prerequisite(
            "canonical-source-snapshot",
            "satisfied" if canonical_ready else "missing",
            "clean-default-head-observed" if canonical_ready else "canonical-source-snapshot-gap",
        ),
        prerequisite(
            "target-repository-mapping-proposal",
            "satisfied" if proposal_present else "missing",
            "proposal-validated-not-approved" if proposal_present else "mapping-proposal-absent",
        ),
        prerequisite(
            "destination-repositories-observed",
            "satisfied" if destinations_ready else "missing",
            "all-mapped-repositories-observed" if destinations_ready else "destination-repository-gap",
        ),
        prerequisite("production-materialization-evidence", "missing", "not-supplied-to-gap-assessment"),
        prerequisite("production-source-removal-authorization", "missing", "not-supplied-to-gap-assessment"),
        prerequisite("canonical-production-approval", "missing", "not-supplied-to-gap-assessment"),
        prerequisite("canonical-production-execution-policy", "missing", "not-supplied-to-gap-assessment"),
        prerequisite("explicit-human-cutover-decision", "missing", "cannot-be-inferred-by-assessor"),
    ]
    if not proposal_present:
        readiness_status = "mapping-proposal-required"
        next_required_action = "author-and-review-target-repository-mapping-proposal"
    elif not destinations_ready:
        readiness_status = "destination-repositories-required"
        next_required_action = "provision-and-observe-every-proposed-destination-repository"
    elif not canonical_ready:
        readiness_status = "canonical-source-snapshot-required"
        next_required_action = "reobserve-clean-canonical-default-branch-snapshot"
    else:
        readiness_status = "production-evidence-and-human-decision-required"
        next_required_action = "author-production-evidence-set-and-obtain-explicit-human-decision"
    payload = {
        "schema": ASSESSMENT_SCHEMA,
        "assessment_type": ASSESSMENT_TYPE,
        "authority_scope": AUTHORITY_SCOPE,
        "readiness_status": readiness_status,
        "next_required_action": next_required_action,
        "execution_authority": EXECUTION_AUTHORITY,
        "canonical_cutover_execution_status": CUTOVER_STATUS,
        "source_bindings": {
            "science_rings_sha256": inventory["source_documents"]["science_rings_sha256"],
            "ownership_policy_sha256": inventory["source_documents"]["ownership_policy_sha256"],
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "repository_catalog_file_sha256": catalog_file_sha,
            "repository_catalog_identity_sha256": catalog_payload["catalog_identity_sha256"],
            "mapping_proposal_file_sha256": proposal_file_sha,
            "mapping_proposal_identity_sha256": None
            if proposal_payload is None
            else proposal_payload["proposal_identity_sha256"],
        },
        "canonical_repository": {"repository_id": canonical_repository_id, **canonical},
        "targets": destinations,
        "prerequisites": prerequisites,
        "summary": {
            "planned_target_count": target_count,
            "mapped_target_count": 0 if proposal is None else len(proposal),
            "observed_available_destination_count": available_count,
            "satisfied_prerequisite_count": sum(row["status"] == "satisfied" for row in prerequisites),
            "missing_prerequisite_count": sum(row["status"] == "missing" for row in prerequisites),
        },
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": ASSESSMENT_LIMITATIONS,
    }
    return with_identity(payload, "assessment_identity_sha256")


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProductionGapError("E-SRB-PROD-GAP-005", f"assessment output already exists: {path}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise ProductionGapError("E-SRB-PROD-GAP-005", f"assessment output appeared during write: {path}")
        os.rename(temporary, path)
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path | None]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise ProductionGapError("E-SRB-PROD-GAP-001", "repo root is not a directory")
    rings = inventory_tool.resolve_input(repo_root, args.rings, "science rings")
    ownership = inventory_tool.resolve_input(repo_root, args.ownership, "ownership policy")
    catalog = Path(args.repository_catalog).expanduser().resolve(strict=True)
    proposal = None
    if args.mapping_proposal is not None:
        proposal = Path(args.mapping_proposal).expanduser().resolve(strict=True)
    if not SAFE_TOKEN.fullmatch(args.canonical_repository_id):
        raise ProductionGapError("E-SRB-PROD-GAP-001", "canonical repository ID is invalid")
    if not SAFE_TOKEN.fullmatch(args.remote_name):
        raise ProductionGapError("E-SRB-PROD-GAP-001", "remote name is invalid")
    return repo_root, rings, ownership, catalog, proposal


def assess_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, catalog, proposal = resolve_inputs(args)
    output_input = Path(args.output).expanduser()
    output = output_input if output_input.is_absolute() else Path.cwd() / output_input
    payload = expected_assessment(
        repo_root,
        rings,
        ownership,
        catalog,
        proposal,
        args.canonical_repository_id,
        args.remote_name,
    )
    write_atomic(output, payload)
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_PASS "
        f"assessment_identity={payload['assessment_identity_sha256']} "
        f"targets={payload['summary']['planned_target_count']} "
        f"status={payload['readiness_status']} authority={payload['execution_authority']}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, catalog, proposal = resolve_inputs(args)
    assessment_path = Path(args.assessment).expanduser().resolve(strict=True)
    actual, _file_sha = read_regular_json(
        assessment_path, "gap assessment", "E-SRB-PROD-GAP-006"
    )
    if actual.get("schema") != ASSESSMENT_SCHEMA:
        raise ProductionGapError("E-SRB-PROD-GAP-006", "unsupported gap assessment schema")
    if actual.get("assessment_identity_sha256") != identity(actual, "assessment_identity_sha256"):
        raise ProductionGapError("E-SRB-PROD-GAP-006", "gap assessment identity hash mismatch")
    expected = expected_assessment(
        repo_root,
        rings,
        ownership,
        catalog,
        proposal,
        args.canonical_repository_id,
        args.remote_name,
    )
    if actual != expected:
        raise ProductionGapError("E-SRB-PROD-GAP-006", "gap assessment bindings do not match inputs")
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_VERIFY_PASS "
        f"assessment_identity={actual['assessment_identity_sha256']} status={actual['readiness_status']}"
    )
    return 0


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--rings", default="science-rings.tsv")
    parser.add_argument(
        "--ownership",
        default="docs/ecosystem/science-physical-extraction-ownership.tsv",
    )
    parser.add_argument("--repository-catalog", required=True)
    parser.add_argument("--mapping-proposal")
    parser.add_argument("--canonical-repository-id", required=True)
    parser.add_argument("--remote-name", default="origin")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-canonical-production-gap-assessor")
    subparsers = result.add_subparsers(dest="command", required=True)
    assess = subparsers.add_parser("assess")
    add_common_arguments(assess)
    assess.add_argument("--output", required=True)
    assess.set_defaults(handler=assess_command)
    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.add_argument("--assessment", required=True)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except inventory_tool.PhysicalExtractionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_REFUSED reason={error}", file=sys.stderr)
        return 1
    except ProductionGapError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, subprocess.SubprocessError) as error:
        print(f"error[E-SRB-PROD-GAP-007]: production gap assessment failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
