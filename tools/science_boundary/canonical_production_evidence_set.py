#!/usr/bin/env python3
"""Emit and verify read-only canonical-production evidence sets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "science_boundary"))
import canonical_production_gap_assessor as gap_tool  # noqa: E402
import physical_extraction_inventory as inventory_tool  # noqa: E402


EVIDENCE_SCHEMA = "sounio.physical-extraction-canonical-production-evidence-set.v1"
VALIDATION_SCHEMA = "sounio.physical-extraction-production-validation-observations.v1"
EVIDENCE_TYPE = "read-only-production-evidence-draft"
VALIDATION_TYPE = "supplied-validation-observations"
AUTHORITY_SCOPE = "evidence-observation-only"
VALIDATION_AUTHORITY = "evidence-citation-only"
PROPOSAL_STATUS = "proposed-not-approved"
EXECUTION_AUTHORITY = "none"
SOURCE_REMOVAL_AUTHORITY = "none"
PRODUCTION_APPROVAL = "not-approved"
CUTOVER_STATUS = "not-executed"
ASSURANCE_LEVEL = "exact-git-and-byte-observation-plus-supplied-validation"
SAMPLE_LIMIT = 20
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GIT_OID = re.compile(r"^[0-9a-f]{40}$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
VALIDATION_LIMITATIONS = [
    "observations_are_supplied_records_not_commands_replayed_by_this_contract",
    "a_pass_result_does_not_prove_scientific_truth",
    "a_pass_result_does_not_assert_clinical_validation_or_clinical_authority",
    "validation_observations_do_not_grant_production_or_cutover_authority",
]
EVIDENCE_LIMITATIONS = [
    "evidence_set_never_grants_execution_authority",
    "evidence_set_does_not_create_or_modify_repositories",
    "evidence_set_does_not_materialize_or_remove_source_files",
    "evidence_set_does_not_create_or_update_git_refs",
    "source_and_destination_observations_are_sequential_not_atomic",
    "byte_parity_does_not_prove_scientific_truth",
    "supplied_validation_observations_are_not_replayed_by_the_verifier",
    "exact_copy_evidence_is_not_destination_owner_approval",
    "mapping_proposal_remains_proposed_not_approved",
    "source_removal_authority_and_production_approval_remain_absent",
    "canonical_cutover_remains_not_executed",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
EXECUTION_STEPS = [
    {
        "sequence": 1,
        "step_id": "reobserve-bound-snapshots",
        "required_condition": "all-source-and-destination-git-bindings-still-match",
    },
    {
        "sequence": 2,
        "step_id": "resolve-and-reverify-parity-gaps",
        "required_condition": "every-target-has-exact-byte-parity",
    },
    {
        "sequence": 3,
        "step_id": "obtain-source-removal-authorization",
        "required_condition": "separate-explicit-human-authorization-record-exists",
    },
    {
        "sequence": 4,
        "step_id": "obtain-canonical-production-approval",
        "required_condition": "separate-explicit-production-approval-record-exists",
    },
    {
        "sequence": 5,
        "step_id": "obtain-production-execution-policy",
        "required_condition": "separate-reviewed-execution-and-recovery-policy-exists",
    },
    {
        "sequence": 6,
        "step_id": "obtain-explicit-human-cutover-decision",
        "required_condition": "separate-permission-bearing-cutover-decision-exists",
    },
    {
        "sequence": 7,
        "step_id": "invoke-separate-cutover-interface",
        "required_condition": "all-prior-conditions-remain-satisfied-at-execution-time",
    },
]
ABORT_CONDITIONS = [
    "any-git-binding-drift",
    "any-dirty-source-or-destination-worktree",
    "any-byte-parity-gap",
    "any-validation-failure-or-unbound-validation-observation",
    "any-missing-permission-bearing-prerequisite",
]
ROLLBACK_EVIDENCE = [
    "pre-execution-ref-snapshot",
    "destination-content-recovery-receipt",
    "canonical-source-restoration-receipt",
    "post-rollback-byte-parity-assessment",
]


class ProductionEvidenceError(ValueError):
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
        raise ProductionEvidenceError(
            code,
            f"{label} fields mismatch missing={','.join(sorted(expected - actual)) or '-'} "
            f"extra={','.join(sorted(actual - expected)) or '-'}",
        )


def read_regular_json(path: Path, label: str, code: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ProductionEvidenceError(code, f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ProductionEvidenceError(code, f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ProductionEvidenceError(code, f"{label} must be a JSON object")
    return value, raw


def validate_validation_observations(payload: dict[str, Any], canonical_head: str) -> list[dict[str, Any]]:
    code = "E-SRB-PROD-EVIDENCE-004"
    exact_keys(
        payload,
        {
            "schema",
            "observation_type",
            "authority_scope",
            "canonical_head_oid",
            "observations",
            "limitations",
            "validation_identity_sha256",
        },
        "validation observations",
        code,
    )
    if payload["schema"] != VALIDATION_SCHEMA or payload["observation_type"] != VALIDATION_TYPE:
        raise ProductionEvidenceError(code, "unsupported validation-observation contract")
    if payload["authority_scope"] != VALIDATION_AUTHORITY:
        raise ProductionEvidenceError(code, "validation-observation authority scope is invalid")
    if payload["canonical_head_oid"] != canonical_head or not GIT_OID.fullmatch(payload["canonical_head_oid"]):
        raise ProductionEvidenceError(code, "validation observations are not bound to the canonical source head")
    if payload["limitations"] != VALIDATION_LIMITATIONS:
        raise ProductionEvidenceError(code, "validation-observation limitations mismatch")
    if payload["validation_identity_sha256"] != identity(payload, "validation_identity_sha256"):
        raise ProductionEvidenceError(code, "validation-observation identity hash mismatch")
    rows = payload["observations"]
    if not isinstance(rows, list) or not rows:
        raise ProductionEvidenceError(code, "at least one validation observation is required")
    expected = {
        "observation_id",
        "scope",
        "command",
        "result",
        "exit_code",
        "stdout_sha256",
        "stderr_sha256",
        "evidence_ref",
    }
    previous = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ProductionEvidenceError(code, f"validation observation {index} is not an object")
        exact_keys(row, expected, f"validation observation {index}", code)
        observation_id = row["observation_id"]
        if not isinstance(observation_id, str) or not SAFE_ID.fullmatch(observation_id) or observation_id <= previous:
            raise ProductionEvidenceError(code, "validation observations must have strictly sorted safe IDs")
        previous = observation_id
        if not isinstance(row["scope"], str) or not SAFE_ID.fullmatch(row["scope"]):
            raise ProductionEvidenceError(code, f"validation observation {observation_id} scope is invalid")
        if not isinstance(row["command"], str) or not 1 <= len(row["command"]) <= 4096 or "\x00" in row["command"]:
            raise ProductionEvidenceError(code, f"validation observation {observation_id} command is invalid")
        if row["result"] not in {"passed", "failed"}:
            raise ProductionEvidenceError(code, f"validation observation {observation_id} result is invalid")
        if not isinstance(row["exit_code"], int) or isinstance(row["exit_code"], bool) or row["exit_code"] < 0:
            raise ProductionEvidenceError(code, f"validation observation {observation_id} exit code is invalid")
        if (row["result"] == "passed") != (row["exit_code"] == 0):
            raise ProductionEvidenceError(code, f"validation observation {observation_id} result and exit code disagree")
        for field in ("stdout_sha256", "stderr_sha256"):
            if not isinstance(row[field], str) or not SHA256.fullmatch(row[field]):
                raise ProductionEvidenceError(code, f"validation observation {observation_id} {field} is invalid")
        if not isinstance(row["evidence_ref"], str) or not 1 <= len(row["evidence_ref"]) <= 2048:
            raise ProductionEvidenceError(code, f"validation observation {observation_id} evidence_ref is invalid")
        if any(character.isspace() for character in row["evidence_ref"]):
            raise ProductionEvidenceError(code, f"validation observation {observation_id} evidence_ref contains whitespace")
    return rows


def relative_source_files(unit: dict[str, Any]) -> list[dict[str, Any]]:
    source_root = PurePosixPath(unit["source_path"])
    result: list[dict[str, Any]] = []
    for item in unit["files"]:
        try:
            relative = PurePosixPath(item["path"]).relative_to(source_root).as_posix()
        except ValueError as error:
            raise ProductionEvidenceError(
                "E-SRB-PROD-EVIDENCE-002", f"inventory member escapes source unit {unit['source_path']}"
            ) from error
        if relative in {"", "."}:
            raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-002", "invalid relative source member")
        result.append({"path": relative, "size_bytes": item["size_bytes"], "sha256": item["sha256"]})
    return result


def tracked_files(repository: Path, pathspec: str | None = None) -> list[str]:
    arguments = ["ls-files", "-z"]
    if pathspec is not None:
        arguments.extend(["--", pathspec])
    raw = gap_tool.run_git(repository, arguments)
    return sorted(value for value in raw.split("\0") if value)


def scan_destination_tree(root: Path) -> tuple[list[dict[str, Any]], int, str]:
    if root.is_symlink() or not root.is_dir():
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-003", f"destination is not a regular directory: {root}")
    files: list[dict[str, Any]] = []
    for current, directories, names in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directories[:] = [name for name in directories if name != ".git"]
            names = [name for name in names if name != ".git"]
        for directory in directories:
            child = current_path / directory
            if child.is_symlink():
                raise ProductionEvidenceError(
                    "E-SRB-PROD-EVIDENCE-003", f"symbolic-link directory in destination: {child}"
                )
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise ProductionEvidenceError(
                    "E-SRB-PROD-EVIDENCE-003", f"symbolic-link file in destination: {path}"
                )
            try:
                size, digest = inventory_tool.stable_file_identity(path)
            except inventory_tool.PhysicalExtractionError as error:
                raise ProductionEvidenceError(
                    "E-SRB-PROD-EVIDENCE-003", f"cannot inventory destination member {path}: {error}"
                ) from error
            files.append({"path": path.relative_to(root).as_posix(), "size_bytes": size, "sha256": digest})
    total_bytes = sum(item["size_bytes"] for item in files)
    return files, total_bytes, sha256_bytes(canonical_json(files))


def compare_files(source: list[dict[str, Any]], destination: list[dict[str, Any]]) -> dict[str, Any]:
    source_by_path = {item["path"]: item for item in source}
    destination_by_path = {item["path"]: item for item in destination}
    missing = sorted(set(source_by_path) - set(destination_by_path))
    extra = sorted(set(destination_by_path) - set(source_by_path))
    changed = sorted(
        path
        for path in set(source_by_path) & set(destination_by_path)
        if source_by_path[path] != destination_by_path[path]
    )
    matching = len(source_by_path) - len(missing) - len(changed)
    exact = not missing and not extra and not changed
    return {
        "status": "exact-copy-verified" if exact else "parity-gap-observed",
        "matching_file_count": matching,
        "missing_file_count": len(missing),
        "extra_file_count": len(extra),
        "changed_file_count": len(changed),
        "sample_limit": SAMPLE_LIMIT,
        "missing_paths_sample_complete": len(missing) <= SAMPLE_LIMIT,
        "extra_paths_sample_complete": len(extra) <= SAMPLE_LIMIT,
        "changed_paths_sample_complete": len(changed) <= SAMPLE_LIMIT,
        "missing_paths_sample": missing[:SAMPLE_LIMIT],
        "extra_paths_sample": extra[:SAMPLE_LIMIT],
        "changed_paths_sample": changed[:SAMPLE_LIMIT],
    }


def destination_root(destinations_root: Path, repository_id: str, source_root: Path) -> Path:
    if destinations_root.is_symlink() or not destinations_root.is_dir():
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-001", "destinations root must be a regular directory")
    resolved_root = destinations_root.resolve(strict=True)
    if resolved_root == source_root or resolved_root in source_root.parents or source_root in resolved_root.parents:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-001", "destinations root must be separate from source")
    candidate = resolved_root / repository_id
    if candidate.is_symlink() or not candidate.is_dir():
        raise ProductionEvidenceError(
            "E-SRB-PROD-EVIDENCE-003", f"mapped destination repository is absent: {repository_id}"
        )
    resolved = candidate.resolve(strict=True)
    if resolved.parent != resolved_root:
        raise ProductionEvidenceError(
            "E-SRB-PROD-EVIDENCE-003", f"mapped destination escapes destinations root: {repository_id}"
        )
    return resolved


def exact_git_binding(
    repository: Path,
    remote_name: str,
    mapping: dict[str, Any],
    catalog_row: dict[str, Any],
) -> dict[str, Any]:
    observation = gap_tool.git_observation(repository, remote_name)
    expected = {
        "remote_url": mapping["remote_url"],
        "branch": mapping["branch"],
        "head_oid": mapping["expected_head_oid"],
    }
    catalog_expected = {
        "remote_url": catalog_row["remote_url"],
        "branch": catalog_row["default_branch"],
        "head_oid": catalog_row["head_oid"],
    }
    if catalog_row["archived"] or catalog_row["is_empty"]:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-003", "mapped destination is archived or empty")
    if catalog_row["observed_permission"] not in {"ADMIN", "MAINTAIN", "WRITE"}:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-003", "mapped destination permission observation is insufficient")
    if expected != catalog_expected:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-003", "mapping and catalog destination bindings disagree")
    if observation["worktree_status"] != "clean" or any(observation[key] != value for key, value in expected.items()):
        raise ProductionEvidenceError(
            "E-SRB-PROD-EVIDENCE-003", f"destination Git observation does not match proposal: {mapping['repository_id']}"
        )
    return {**observation, "observation_status": "exact-proposal-and-catalog-bound-clean-snapshot"}


def governance_prerequisites(exact_count: int, target_count: int) -> list[dict[str, str]]:
    parity_complete = exact_count == target_count
    return [
        {
            "prerequisite_id": "production-materialization-evidence",
            "status": "satisfied-evidence-only" if parity_complete else "observed-partial",
            "detail_code": "all-target-byte-parity-observed" if parity_complete else "target-byte-parity-gaps-observed",
        },
        {
            "prerequisite_id": "source-removal-authorization",
            "status": "missing",
            "detail_code": "separate-explicit-human-authorization-required",
        },
        {
            "prerequisite_id": "canonical-production-approval",
            "status": "missing",
            "detail_code": "separate-explicit-production-approval-required",
        },
        {
            "prerequisite_id": "canonical-production-execution-policy",
            "status": "missing",
            "detail_code": "separate-reviewed-execution-and-recovery-policy-required",
        },
        {
            "prerequisite_id": "explicit-human-cutover-decision",
            "status": "missing",
            "detail_code": "separate-permission-bearing-human-decision-required",
        },
    ]


def expected_evidence(
    repo_root: Path,
    rings: Path,
    ownership: Path,
    catalog_path: Path,
    proposal_path: Path,
    validation_path: Path,
    destinations_root: Path,
    canonical_repository_id: str,
    source_remote_name: str,
    destination_remote_name: str,
) -> dict[str, Any]:
    inventory = inventory_tool.expected_inventory(repo_root, rings, ownership)
    planned_units = {
        unit["target_id"]: unit for unit in inventory["units"] if unit["disposition"] == "extract-planned"
    }
    expected_targets = gap_tool.planned_targets(inventory)
    catalog_payload, catalog_raw = read_regular_json(
        catalog_path, "repository catalog", "E-SRB-PROD-EVIDENCE-002"
    )
    try:
        catalog = gap_tool.validate_catalog(catalog_payload)
    except gap_tool.ProductionGapError as error:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-002", f"repository catalog refused: {error}") from error
    proposal_payload, proposal_raw = read_regular_json(
        proposal_path, "mapping proposal", "E-SRB-PROD-EVIDENCE-002"
    )
    try:
        proposal = gap_tool.validate_proposal(
            proposal_payload, catalog_payload["catalog_identity_sha256"], expected_targets
        )
    except gap_tool.ProductionGapError as error:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-002", f"mapping proposal refused: {error}") from error
    if proposal_payload["canonical_repository_id"] != canonical_repository_id:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-002", "mapping proposal canonical repository mismatch")

    source_observation = gap_tool.git_observation(repo_root, source_remote_name)
    canonical_row = catalog.get(canonical_repository_id)
    canonical, canonical_ok = gap_tool.canonical_repository_assessment(source_observation, canonical_row)
    if not canonical_ok:
        raise ProductionEvidenceError(
            "E-SRB-PROD-EVIDENCE-002",
            f"canonical source is not an exact clean catalog-bound snapshot: {','.join(canonical['gap_codes'])}",
        )

    validation_payload, validation_raw = read_regular_json(
        validation_path, "validation observations", "E-SRB-PROD-EVIDENCE-004"
    )
    validation_rows = validate_validation_observations(validation_payload, source_observation["head_oid"])

    targets: list[dict[str, Any]] = []
    for target_id in sorted(planned_units):
        unit = planned_units[target_id]
        mapping = proposal[target_id]
        catalog_row = catalog.get(mapping["repository_id"])
        if catalog_row is None:
            raise ProductionEvidenceError(
                "E-SRB-PROD-EVIDENCE-003", f"mapped repository is absent from catalog: {mapping['repository_id']}"
            )
        repository = destination_root(destinations_root, mapping["repository_id"], repo_root)
        git_binding = exact_git_binding(repository, destination_remote_name, mapping, catalog_row)
        source_files = relative_source_files(unit)
        source_tracked = tracked_files(repo_root, unit["source_path"])
        source_inventory_paths = sorted(item["path"] for item in unit["files"])
        if source_tracked != source_inventory_paths:
            raise ProductionEvidenceError(
                "E-SRB-PROD-EVIDENCE-002",
                f"source inventory is not exactly tracked by canonical HEAD: {unit['source_path']}",
            )
        destination_files, destination_bytes, destination_tree = scan_destination_tree(repository)
        destination_tracked = tracked_files(repository)
        destination_inventory_paths = sorted(item["path"] for item in destination_files)
        if destination_tracked != destination_inventory_paths:
            raise ProductionEvidenceError(
                "E-SRB-PROD-EVIDENCE-003",
                f"destination inventory is not exactly tracked by destination HEAD: {mapping['repository_id']}",
            )
        parity = compare_files(source_files, destination_files)
        source_tree = sha256_bytes(canonical_json(source_files))
        targets.append(
            {
                "source_path": unit["source_path"],
                "ring": unit["ring"],
                "target_id": target_id,
                "target_owner": unit["target_owner"],
                "source_inventory": {
                    "file_count": len(source_files),
                    "total_bytes": sum(item["size_bytes"] for item in source_files),
                    "inventory_tree_sha256": unit["tree_sha256"],
                    "comparison_tree_sha256": source_tree,
                },
                "destination_repository": {
                    "repository_id": mapping["repository_id"],
                    **git_binding,
                },
                "destination_inventory": {
                    "file_count": len(destination_files),
                    "total_bytes": destination_bytes,
                    "comparison_tree_sha256": destination_tree,
                },
                "parity": parity,
            }
        )

    exact_count = sum(target["parity"]["status"] == "exact-copy-verified" for target in targets)
    failed_validations = sum(row["result"] == "failed" for row in validation_rows)
    evidence_status = (
        "production-evidence-draft-exact-parity"
        if exact_count == len(targets) and failed_validations == 0
        else "production-evidence-draft-gaps-observed"
    )
    prerequisites = governance_prerequisites(exact_count, len(targets))
    payload = {
        "schema": EVIDENCE_SCHEMA,
        "evidence_type": EVIDENCE_TYPE,
        "authority_scope": AUTHORITY_SCOPE,
        "evidence_status": evidence_status,
        "next_required_action": (
            "obtain-separate-explicit-governance-decisions"
            if evidence_status == "production-evidence-draft-exact-parity"
            else "resolve-parity-or-validation-gaps-and-reissue-evidence"
        ),
        "proposal_status": PROPOSAL_STATUS,
        "execution_authority": EXECUTION_AUTHORITY,
        "source_removal_authority": SOURCE_REMOVAL_AUTHORITY,
        "canonical_production_approval": PRODUCTION_APPROVAL,
        "canonical_cutover_execution_status": CUTOVER_STATUS,
        "source_bindings": {
            "science_rings_sha256": inventory["source_documents"]["science_rings_sha256"],
            "ownership_policy_sha256": inventory["source_documents"]["ownership_policy_sha256"],
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "repository_catalog_file_sha256": sha256_bytes(catalog_raw),
            "repository_catalog_identity_sha256": catalog_payload["catalog_identity_sha256"],
            "mapping_proposal_file_sha256": sha256_bytes(proposal_raw),
            "mapping_proposal_identity_sha256": proposal_payload["proposal_identity_sha256"],
            "validation_observations_file_sha256": sha256_bytes(validation_raw),
            "validation_observations_identity_sha256": validation_payload["validation_identity_sha256"],
        },
        "canonical_source": {
            "repository_id": canonical_repository_id,
            **canonical,
            "observation_status": "exact-catalog-bound-clean-snapshot",
        },
        "targets": targets,
        "validation": {
            "canonical_head_oid": validation_payload["canonical_head_oid"],
            "observations": validation_rows,
            "summary": {
                "observation_count": len(validation_rows),
                "passed_count": sum(row["result"] == "passed" for row in validation_rows),
                "failed_count": failed_validations,
            },
        },
        "proposed_execution_plan": {
            "plan_status": "draft-not-authorized",
            "steps": EXECUTION_STEPS,
            "abort_conditions": ABORT_CONDITIONS,
            "rollback_evidence_required": ROLLBACK_EVIDENCE,
        },
        "governance_prerequisites": prerequisites,
        "summary": {
            "target_count": len(targets),
            "exact_parity_target_count": exact_count,
            "parity_gap_target_count": len(targets) - exact_count,
            "matching_file_count": sum(target["parity"]["matching_file_count"] for target in targets),
            "missing_file_count": sum(target["parity"]["missing_file_count"] for target in targets),
            "extra_file_count": sum(target["parity"]["extra_file_count"] for target in targets),
            "changed_file_count": sum(target["parity"]["changed_file_count"] for target in targets),
            "validation_passed_count": sum(row["result"] == "passed" for row in validation_rows),
            "validation_failed_count": failed_validations,
            "permission_bearing_prerequisite_missing_count": sum(
                row["status"] == "missing" for row in prerequisites
            ),
        },
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": EVIDENCE_LIMITATIONS,
    }
    return with_identity(payload, "evidence_identity_sha256")


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-005", f"evidence output already exists: {path}")
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
            raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-005", f"evidence output appeared during write: {path}")
        os.rename(temporary, path)
        temporary = None
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-001", "repo root is not a directory")
    rings = inventory_tool.resolve_input(repo_root, args.rings, "science rings")
    ownership = inventory_tool.resolve_input(repo_root, args.ownership, "ownership policy")
    catalog = Path(args.repository_catalog).expanduser().resolve(strict=True)
    proposal = Path(args.mapping_proposal).expanduser().resolve(strict=True)
    validation = Path(args.validation_observations).expanduser().resolve(strict=True)
    destinations = Path(args.destinations_root).expanduser().resolve(strict=True)
    for value, label in (
        (args.canonical_repository_id, "canonical repository ID"),
        (args.source_remote_name, "source remote name"),
        (args.destination_remote_name, "destination remote name"),
    ):
        if not SAFE_TOKEN.fullmatch(value):
            raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-001", f"{label} is invalid")
    return repo_root, rings, ownership, catalog, proposal, validation, destinations


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--rings", default="science-rings.tsv")
    parser.add_argument("--ownership", default="docs/ecosystem/science-physical-extraction-ownership.tsv")
    parser.add_argument("--repository-catalog", required=True)
    parser.add_argument("--mapping-proposal", required=True)
    parser.add_argument("--validation-observations", required=True)
    parser.add_argument("--destinations-root", required=True)
    parser.add_argument("--canonical-repository-id", required=True)
    parser.add_argument("--source-remote-name", default="origin")
    parser.add_argument("--destination-remote-name", default="origin")


def build_command(args: argparse.Namespace) -> int:
    inputs = resolve_inputs(args)
    output_input = Path(args.output).expanduser()
    output = output_input if output_input.is_absolute() else Path.cwd() / output_input
    payload = expected_evidence(
        *inputs,
        args.canonical_repository_id,
        args.source_remote_name,
        args.destination_remote_name,
    )
    write_atomic(output, payload)
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_PASS "
        f"evidence_identity={payload['evidence_identity_sha256']} "
        f"targets={payload['summary']['target_count']} "
        f"exact={payload['summary']['exact_parity_target_count']} "
        f"gaps={payload['summary']['parity_gap_target_count']} "
        f"status={payload['evidence_status']} authority={payload['execution_authority']}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    inputs = resolve_inputs(args)
    evidence_path = Path(args.evidence).expanduser().resolve(strict=True)
    actual, _raw = read_regular_json(evidence_path, "production evidence set", "E-SRB-PROD-EVIDENCE-006")
    if actual.get("schema") != EVIDENCE_SCHEMA:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-006", "unsupported production evidence schema")
    if actual.get("evidence_identity_sha256") != identity(actual, "evidence_identity_sha256"):
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-006", "production evidence identity hash mismatch")
    expected = expected_evidence(
        *inputs,
        args.canonical_repository_id,
        args.source_remote_name,
        args.destination_remote_name,
    )
    if actual != expected:
        raise ProductionEvidenceError("E-SRB-PROD-EVIDENCE-006", "production evidence bindings do not match inputs")
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_VERIFY_PASS "
        f"evidence_identity={actual['evidence_identity_sha256']} status={actual['evidence_status']}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-canonical-production-evidence-set")
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    common_arguments(build)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=build_command)
    verify = subparsers.add_parser("verify")
    common_arguments(verify)
    verify.add_argument("--evidence", required=True)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except inventory_tool.PhysicalExtractionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_REFUSED reason={error}", file=sys.stderr)
        return 1
    except gap_tool.ProductionGapError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_REFUSED reason={error}", file=sys.stderr)
        return 1
    except ProductionEvidenceError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError) as error:
        print(f"error[E-SRB-PROD-EVIDENCE-007]: production evidence operation failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
