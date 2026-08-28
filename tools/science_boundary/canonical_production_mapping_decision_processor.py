#!/usr/bin/env python3
"""Process and verify non-authorizing canonical-production mapping selections."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "science_boundary"))
import canonical_production_gap_assessor as gap_tool  # noqa: E402
import physical_extraction_inventory as inventory_tool  # noqa: E402


DECISION_SCHEMA = "sounio.physical-extraction-canonical-production-mapping-decision.v1"
RECEIPT_SCHEMA = "sounio.physical-extraction-canonical-production-mapping-decision-receipt.v1"
DECISION_TYPE = "canonical-production-target-mapping-selection"
RECEIPT_TYPE = "non-authorizing-mapping-decision-processing-receipt"
DECISION_AUTHORITY_SCOPE = "mapping-proposal-input-only"
RECEIPT_AUTHORITY_SCOPE = "mapping-proposal-preparation-only"
DECISION_STATUS = "human-selection-recorded-not-approved"
EVIDENCE_STATUS = "transcribed-not-authenticated"
EXECUTION_AUTHORITY = "none"
CUTOVER_STATUS = "not-executed"
ASSURANCE_LEVEL = "identity-plus-bound-catalog-and-local-git-observation"
ACTIONS = {"reuse-observed", "request-new", "revise-target"}
VISIBILITIES = {"PUBLIC", "PRIVATE", "INTERNAL"}
PUSH_PERMISSIONS = {"ADMIN", "MAINTAIN", "WRITE"}
SAFE_RESPONDER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@/-]{0,127}$")
AUTHORIZED_OPERATIONS = ["draft-proposed-not-approved-mapping"]
PROHIBITED_OPERATIONS = [
    "create-or-modify-repositories",
    "materialize-or-remove-source-files",
    "create-or-update-git-refs",
    "approve-canonical-production",
    "approve-or-execute-canonical-cutover",
]
DECISION_LIMITATIONS = [
    "decision_record_does_not_authenticate_responder_identity",
    "decision_record_does_not_prove_human_or_organizational_authority",
    "decision_record_authorizes_only_draft_proposed_not_approved_mapping",
    "decision_record_does_not_create_or_modify_repositories",
    "decision_record_does_not_authorize_materialization_source_removal_or_ref_updates",
    "decision_record_does_not_approve_canonical_production_or_cutover",
    "decision_must_be_reconfirmed_after_bound_catalog_or_source_drift",
    "decision_record_does_not_assert_scientific_truth",
]
RECEIPT_LIMITATIONS = [
    "receipt_never_grants_execution_authority",
    "receipt_does_not_authenticate_responder_identity_or_authority",
    "receipt_does_not_create_or_modify_repositories",
    "receipt_does_not_materialize_or_remove_source_files",
    "receipt_does_not_create_or_update_git_refs",
    "emitted_mapping_is_proposed_not_approved",
    "supplied_catalog_is_not_live_hosting_attestation",
    "catalog_or_source_drift_requires_a_new_selection_record",
    "receipt_commit_does_not_make_multi_file_promotion_crash_atomic",
    "receipt_does_not_approve_canonical_production_or_cutover",
    "receipt_does_not_assert_scientific_truth",
]


class MappingDecisionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_regular_json(path: Path, label: str, code: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise MappingDecisionError(code, f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MappingDecisionError(code, f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise MappingDecisionError(code, f"{label} must be a JSON object")
    return value, sha256_bytes(raw)


def exact_keys(value: dict[str, Any], expected: set[str], label: str, code: str) -> None:
    actual = set(value)
    if actual != expected:
        raise MappingDecisionError(
            code,
            f"{label} fields mismatch missing={','.join(sorted(expected - actual)) or '-'} "
            f"extra={','.join(sorted(actual - expected)) or '-'}",
        )


def valid_url(value: Any, *, https_only: bool = False, git_remote: bool = False) -> bool:
    if not isinstance(value, str) or not 1 <= len(value) <= 2048 or any(char.isspace() for char in value):
        return False
    if https_only and not value.startswith("https://"):
        return False
    return not git_remote or value.endswith(".git")


def validate_source_evidence(payload: Any) -> None:
    code = "E-SRB-PROD-MAP-002"
    if not isinstance(payload, dict):
        raise MappingDecisionError(code, "source decision evidence must be an object")
    exact_keys(
        payload,
        {
            "issue_url",
            "response_url",
            "responder_label",
            "response_body_sha256",
            "submitted_at_utc",
            "evidence_status",
        },
        "source decision evidence",
        code,
    )
    if not valid_url(payload["issue_url"], https_only=True) or not valid_url(
        payload["response_url"], https_only=True
    ):
        raise MappingDecisionError(code, "source decision evidence URLs are invalid")
    if not isinstance(payload["responder_label"], str) or not SAFE_RESPONDER.fullmatch(
        payload["responder_label"]
    ):
        raise MappingDecisionError(code, "source decision responder label is invalid")
    if not isinstance(payload["response_body_sha256"], str) or not gap_tool.SHA256.fullmatch(
        payload["response_body_sha256"]
    ):
        raise MappingDecisionError(code, "source response body hash is invalid")
    if not isinstance(payload["submitted_at_utc"], str) or not gap_tool.UTC_TIMESTAMP.fullmatch(
        payload["submitted_at_utc"]
    ):
        raise MappingDecisionError(code, "source decision timestamp is not canonical UTC")
    if payload["evidence_status"] != EVIDENCE_STATUS:
        raise MappingDecisionError(code, "source decision evidence must remain transcribed-not-authenticated")


def validate_action_shape(row: dict[str, Any], target_id: str) -> None:
    code = "E-SRB-PROD-MAP-002"
    action = row["action"]
    repository_id = row["repository_id"]
    remote_url = row["remote_url"]
    branch = row["branch"]
    visibility = row["visibility"]
    rationale = row["rationale"]
    if action in {"reuse-observed", "request-new"}:
        if not isinstance(repository_id, str) or not gap_tool.SAFE_TOKEN.fullmatch(repository_id):
            raise MappingDecisionError(code, f"repository ID is invalid for {target_id}")
        if not valid_url(remote_url, git_remote=True):
            raise MappingDecisionError(code, f"remote URL is invalid for {target_id}")
        if not isinstance(branch, str) or not gap_tool.SAFE_ID.fullmatch(branch):
            raise MappingDecisionError(code, f"branch is invalid for {target_id}")
        if rationale is not None:
            raise MappingDecisionError(code, f"rationale must be null for {action} target {target_id}")
        if action == "reuse-observed" and visibility is not None:
            raise MappingDecisionError(code, f"reuse-observed visibility must come from the catalog for {target_id}")
        if action == "request-new" and visibility not in VISIBILITIES:
            raise MappingDecisionError(code, f"request-new visibility is invalid for {target_id}")
        return
    if action == "revise-target":
        if any(value is not None for value in (repository_id, remote_url, branch, visibility)):
            raise MappingDecisionError(code, f"revise-target repository fields must be null for {target_id}")
        if not isinstance(rationale, str) or not 1 <= len(rationale) <= 2048:
            raise MappingDecisionError(code, f"revise-target rationale is invalid for {target_id}")
        return
    raise MappingDecisionError(code, f"unsupported action for {target_id}: {action}")


def validate_decision(
    payload: dict[str, Any],
    targets: dict[str, dict[str, Any]],
    catalog_payload: dict[str, Any],
    catalog: dict[str, dict[str, Any]],
    observation: dict[str, Any],
    canonical_repository_id: str,
) -> dict[str, dict[str, Any]]:
    code = "E-SRB-PROD-MAP-002"
    exact_keys(
        payload,
        {
            "schema",
            "decision_type",
            "authority_scope",
            "decision_status",
            "source_decision_evidence",
            "bindings",
            "authorized_operations",
            "prohibited_operations",
            "targets",
            "limitations",
            "decision_identity_sha256",
        },
        "mapping decision",
        code,
    )
    if payload["schema"] != DECISION_SCHEMA or payload["decision_type"] != DECISION_TYPE:
        raise MappingDecisionError(code, "unsupported mapping decision contract")
    if payload["authority_scope"] != DECISION_AUTHORITY_SCOPE or payload["decision_status"] != DECISION_STATUS:
        raise MappingDecisionError(code, "mapping decision authority or status is invalid")
    if payload["authorized_operations"] != AUTHORIZED_OPERATIONS:
        raise MappingDecisionError(code, "mapping decision authorized operations mismatch")
    if payload["prohibited_operations"] != PROHIBITED_OPERATIONS:
        raise MappingDecisionError(code, "mapping decision prohibited operations mismatch")
    if payload["limitations"] != DECISION_LIMITATIONS:
        raise MappingDecisionError(code, "mapping decision limitations mismatch")
    if not isinstance(payload["decision_identity_sha256"], str) or payload[
        "decision_identity_sha256"
    ] != gap_tool.identity(payload, "decision_identity_sha256"):
        raise MappingDecisionError(code, "mapping decision identity hash mismatch")
    validate_source_evidence(payload["source_decision_evidence"])
    bindings = payload["bindings"]
    if not isinstance(bindings, dict):
        raise MappingDecisionError(code, "mapping decision bindings must be an object")
    exact_keys(
        bindings,
        {
            "repository_catalog_identity_sha256",
            "repository_catalog_observed_at_utc",
            "canonical_repository_id",
            "canonical_repository_branch",
            "canonical_repository_head_oid",
        },
        "mapping decision bindings",
        code,
    )
    if bindings["repository_catalog_identity_sha256"] != catalog_payload["catalog_identity_sha256"]:
        raise MappingDecisionError(code, "mapping decision repository catalog identity drift")
    if bindings["repository_catalog_observed_at_utc"] != catalog_payload["observed_at_utc"]:
        raise MappingDecisionError(code, "mapping decision repository catalog timestamp drift")
    if bindings["canonical_repository_id"] != canonical_repository_id:
        raise MappingDecisionError(code, "mapping decision canonical repository ID mismatch")
    if bindings["canonical_repository_branch"] != observation["branch"]:
        raise MappingDecisionError(code, "mapping decision canonical branch drift")
    if bindings["canonical_repository_head_oid"] != observation["head_oid"]:
        raise MappingDecisionError(code, "mapping decision canonical head drift")
    canonical, canonical_ready = gap_tool.canonical_repository_assessment(
        observation, catalog.get(canonical_repository_id)
    )
    if not canonical_ready:
        raise MappingDecisionError(
            code,
            f"canonical source observation is not exact: {','.join(canonical['gap_codes'])}",
        )
    rows = payload["targets"]
    if not isinstance(rows, list) or not rows:
        raise MappingDecisionError(code, "mapping decision must contain target selections")
    expected_fields = {
        "source_path",
        "target_id",
        "target_owner",
        "action",
        "repository_id",
        "remote_url",
        "branch",
        "visibility",
        "rationale",
    }
    result: dict[str, dict[str, Any]] = {}
    repository_ids: set[str] = set()
    remote_urls: set[str] = set()
    catalog_urls = {row["remote_url"] for row in catalog.values()}
    previous = ""
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise MappingDecisionError(code, f"mapping decision target {index} is not an object")
        exact_keys(row, expected_fields, f"mapping decision target {index}", code)
        target_id = row["target_id"]
        if not isinstance(target_id, str) or target_id not in targets:
            raise MappingDecisionError(code, f"mapping decision target {index} is not planned")
        if target_id <= previous:
            raise MappingDecisionError(code, "mapping decision targets must be strictly sorted by target_id")
        previous = target_id
        target = targets[target_id]
        if row["source_path"] != target["source_path"] or row["target_owner"] != target["target_owner"]:
            raise MappingDecisionError(code, f"mapping decision governed target fields mismatch for {target_id}")
        if row["action"] not in ACTIONS:
            raise MappingDecisionError(code, f"mapping decision action is invalid for {target_id}")
        validate_action_shape(row, target_id)
        repository_id = row["repository_id"]
        remote_url = row["remote_url"]
        if repository_id is not None:
            if repository_id == canonical_repository_id:
                raise MappingDecisionError(code, f"canonical repository cannot be a destination for {target_id}")
            if repository_id in repository_ids or remote_url in remote_urls:
                raise MappingDecisionError(code, f"mapping decision reuses a destination for {target_id}")
            repository_ids.add(repository_id)
            remote_urls.add(remote_url)
        if row["action"] == "reuse-observed":
            observed = catalog.get(repository_id)
            if observed is None:
                raise MappingDecisionError(code, f"reuse-observed repository is absent for {target_id}")
            if observed["remote_url"] != remote_url or observed["default_branch"] != row["branch"]:
                raise MappingDecisionError(code, f"reuse-observed repository metadata drift for {target_id}")
            if observed["archived"] or observed["is_empty"] or observed["observed_permission"] not in PUSH_PERMISSIONS:
                raise MappingDecisionError(code, f"reuse-observed repository is unavailable for {target_id}")
            if not isinstance(observed["head_oid"], str) or not gap_tool.GIT_OID.fullmatch(observed["head_oid"]):
                raise MappingDecisionError(code, f"reuse-observed repository head is invalid for {target_id}")
        elif row["action"] == "request-new":
            if repository_id in catalog or remote_url in catalog_urls:
                raise MappingDecisionError(code, f"request-new repository already exists in the bound catalog for {target_id}")
        result[target_id] = row
    missing = sorted(set(targets) - set(result))
    extra = sorted(set(result) - set(targets))
    if missing or extra:
        raise MappingDecisionError(
            code,
            f"mapping decision coverage mismatch missing={','.join(missing) or '-'} extra={','.join(extra) or '-'}",
        )
    return result


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise MappingDecisionError("E-SRB-PROD-MAP-001", "repo root is not a directory")
    rings = inventory_tool.resolve_input(repo_root, args.rings, "science rings")
    ownership = inventory_tool.resolve_input(repo_root, args.ownership, "ownership policy")
    catalog = Path(args.repository_catalog).expanduser().resolve(strict=True)
    decision = Path(args.mapping_decision).expanduser().resolve(strict=True)
    if not gap_tool.SAFE_TOKEN.fullmatch(args.canonical_repository_id):
        raise MappingDecisionError("E-SRB-PROD-MAP-001", "canonical repository ID is invalid")
    if not gap_tool.SAFE_TOKEN.fullmatch(args.remote_name):
        raise MappingDecisionError("E-SRB-PROD-MAP-001", "remote name is invalid")
    return repo_root, rings, ownership, catalog, decision


def expected_outputs(
    repo_root: Path,
    rings: Path,
    ownership: Path,
    catalog_path: Path,
    decision_path: Path,
    canonical_repository_id: str,
    remote_name: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    catalog_payload, catalog_file_sha = read_regular_json(
        catalog_path, "repository catalog", "E-SRB-PROD-MAP-003"
    )
    try:
        catalog = gap_tool.validate_catalog(catalog_payload)
    except gap_tool.ProductionGapError as error:
        raise MappingDecisionError("E-SRB-PROD-MAP-003", f"repository catalog refused: {error}") from error
    decision_payload, decision_file_sha = read_regular_json(
        decision_path, "mapping decision", "E-SRB-PROD-MAP-002"
    )
    inventory = inventory_tool.expected_inventory(repo_root, rings, ownership)
    try:
        targets = gap_tool.planned_targets(inventory)
        observation = gap_tool.git_observation(repo_root, remote_name)
    except gap_tool.ProductionGapError as error:
        raise MappingDecisionError("E-SRB-PROD-MAP-004", f"source observation refused: {error}") from error
    selections = validate_decision(
        decision_payload,
        targets,
        catalog_payload,
        catalog,
        observation,
        canonical_repository_id,
    )
    counts = {action: sum(row["action"] == action for row in selections.values()) for action in ACTIONS}
    all_reuse = counts["reuse-observed"] == len(selections)
    proposal: dict[str, Any] | None = None
    if all_reuse:
        mappings = []
        for target_id in sorted(selections):
            selection = selections[target_id]
            observed = catalog[selection["repository_id"]]
            mappings.append(
                {
                    "target_id": target_id,
                    "target_owner": selection["target_owner"],
                    "repository_id": selection["repository_id"],
                    "remote_url": selection["remote_url"],
                    "branch": selection["branch"],
                    "expected_head_oid": observed["head_oid"],
                    "mapping_status": gap_tool.PROPOSAL_STATUS,
                }
            )
        proposal = gap_tool.with_identity(
            {
                "schema": gap_tool.PROPOSAL_SCHEMA,
                "proposal_type": gap_tool.PROPOSAL_TYPE,
                "authority_scope": "repository-target-proposal-only",
                "proposal_status": gap_tool.PROPOSAL_STATUS,
                "catalog_identity_sha256": catalog_payload["catalog_identity_sha256"],
                "canonical_repository_id": canonical_repository_id,
                "mappings": mappings,
                "limitations": gap_tool.PROPOSAL_LIMITATIONS,
            },
            "proposal_identity_sha256",
        )
    target_results: list[dict[str, Any]] = []
    for target_id in sorted(selections):
        selection = selections[target_id]
        action = selection["action"]
        observed = catalog.get(selection["repository_id"]) if action == "reuse-observed" else None
        result_status = {
            "reuse-observed": "observed-repository-reuse-ready-for-proposal",
            "request-new": "repository-creation-request-required",
            "revise-target": "ownership-policy-revision-required",
        }[action]
        target_results.append(
            {
                "source_path": selection["source_path"],
                "target_id": target_id,
                "target_owner": selection["target_owner"],
                "action": action,
                "repository_id": selection["repository_id"],
                "remote_url": selection["remote_url"],
                "branch": selection["branch"],
                "visibility": observed["visibility"] if observed is not None else selection["visibility"],
                "expected_head_oid": observed["head_oid"] if observed is not None else None,
                "rationale": selection["rationale"],
                "result_status": result_status,
            }
        )
    if counts["revise-target"]:
        processing_status = "ownership-policy-review-required"
        next_required_action = "revise-governed-target-and-repeat-human-selection"
    elif counts["request-new"]:
        processing_status = "destination-repository-creation-required"
        next_required_action = "provision-repositories-reobserve-catalog-and-reconfirm-selection"
    else:
        processing_status = "proposal-input-complete"
        next_required_action = "review-proposed-not-approved-mapping"
    receipt = gap_tool.with_identity(
        {
            "schema": RECEIPT_SCHEMA,
            "receipt_type": RECEIPT_TYPE,
            "authority_scope": RECEIPT_AUTHORITY_SCOPE,
            "processing_status": processing_status,
            "next_required_action": next_required_action,
            "execution_authority": EXECUTION_AUTHORITY,
            "canonical_cutover_execution_status": CUTOVER_STATUS,
            "source_bindings": {
                "science_rings_sha256": inventory["source_documents"]["science_rings_sha256"],
                "ownership_policy_sha256": inventory["source_documents"]["ownership_policy_sha256"],
                "inventory_identity_sha256": inventory["inventory_identity_sha256"],
                "repository_catalog_file_sha256": catalog_file_sha,
                "repository_catalog_identity_sha256": catalog_payload["catalog_identity_sha256"],
                "repository_catalog_observed_at_utc": catalog_payload["observed_at_utc"],
                "mapping_decision_file_sha256": decision_file_sha,
                "mapping_decision_identity_sha256": decision_payload["decision_identity_sha256"],
                "canonical_repository_id": canonical_repository_id,
                "canonical_repository_branch": observation["branch"],
                "canonical_repository_head_oid": observation["head_oid"],
            },
            "source_decision_evidence": decision_payload["source_decision_evidence"],
            "targets": target_results,
            "summary": {
                "target_count": len(selections),
                "reuse_observed_count": counts["reuse-observed"],
                "request_new_count": counts["request-new"],
                "revise_target_count": counts["revise-target"],
                "proposal_mapping_count": 0 if proposal is None else len(proposal["mappings"]),
            },
            "proposal_output_status": "not-emitted"
            if proposal is None
            else "emitted-proposed-not-approved",
            "proposal_identity_sha256": None
            if proposal is None
            else proposal["proposal_identity_sha256"],
            "assurance_level": ASSURANCE_LEVEL,
            "limitations": RECEIPT_LIMITATIONS,
        },
        "receipt_identity_sha256",
    )
    return receipt, proposal


def output_path(value: str) -> Path:
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else Path.cwd() / candidate


def stage_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        return temporary
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_outputs(receipt_path: Path, receipt: dict[str, Any], proposal_path: Path | None, proposal: dict[str, Any] | None) -> None:
    code = "E-SRB-PROD-MAP-005"
    pairs: list[tuple[Path, dict[str, Any]]] = []
    if proposal is not None:
        if proposal_path is None:
            raise MappingDecisionError(code, "all-reuse decision requires --proposal-output")
        pairs.append((proposal_path, proposal))
    elif proposal_path is not None:
        raise MappingDecisionError(code, "non-reuse decision must not declare --proposal-output")
    pairs.append((receipt_path, receipt))
    if len({path.absolute() for path, _payload in pairs}) != len(pairs):
        raise MappingDecisionError(code, "receipt and proposal outputs must be distinct")
    for path, _payload in pairs:
        if path.exists() or path.is_symlink():
            raise MappingDecisionError(code, f"output already exists: {path}")
    staged: list[tuple[Path, Path]] = []
    promoted: list[tuple[Path, Path]] = []
    try:
        staged = [(path, stage_json(path, payload)) for path, payload in pairs]
        for path, temporary in staged:
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise MappingDecisionError(code, f"output appeared during promotion: {path}") from error
            promoted.append((path, temporary))
            descriptor = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    except BaseException:
        for path, temporary in reversed(promoted):
            try:
                if path.exists() and path.stat().st_ino == temporary.stat().st_ino:
                    path.unlink()
            except OSError:
                pass
        raise
    finally:
        for _path, temporary in staged:
            temporary.unlink(missing_ok=True)


def process_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, catalog, decision = resolve_inputs(args)
    receipt, proposal = expected_outputs(
        repo_root,
        rings,
        ownership,
        catalog,
        decision,
        args.canonical_repository_id,
        args.remote_name,
    )
    receipt_path = output_path(args.receipt_output)
    proposal_path = None if args.proposal_output is None else output_path(args.proposal_output)
    write_outputs(receipt_path, receipt, proposal_path, proposal)
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_PASS "
        f"receipt_identity={receipt['receipt_identity_sha256']} "
        f"decision_identity={receipt['source_bindings']['mapping_decision_identity_sha256']} "
        f"targets={receipt['summary']['target_count']} status={receipt['processing_status']} "
        f"proposal={receipt['proposal_output_status']} authority={receipt['execution_authority']}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership, catalog, decision = resolve_inputs(args)
    expected_receipt, expected_proposal = expected_outputs(
        repo_root,
        rings,
        ownership,
        catalog,
        decision,
        args.canonical_repository_id,
        args.remote_name,
    )
    receipt_path = Path(args.receipt).expanduser().resolve(strict=True)
    actual_receipt, _receipt_file_sha = read_regular_json(
        receipt_path, "mapping decision receipt", "E-SRB-PROD-MAP-006"
    )
    if actual_receipt.get("schema") != RECEIPT_SCHEMA:
        raise MappingDecisionError("E-SRB-PROD-MAP-006", "unsupported mapping decision receipt schema")
    if actual_receipt.get("receipt_identity_sha256") != gap_tool.identity(
        actual_receipt, "receipt_identity_sha256"
    ):
        raise MappingDecisionError("E-SRB-PROD-MAP-006", "mapping decision receipt identity hash mismatch")
    if actual_receipt != expected_receipt:
        raise MappingDecisionError("E-SRB-PROD-MAP-006", "mapping decision receipt bindings do not match inputs")
    if expected_proposal is None:
        if args.mapping_proposal is not None:
            raise MappingDecisionError("E-SRB-PROD-MAP-006", "receipt status prohibits a mapping proposal")
    else:
        if args.mapping_proposal is None:
            raise MappingDecisionError("E-SRB-PROD-MAP-006", "receipt requires a mapping proposal")
        proposal_path = Path(args.mapping_proposal).expanduser().resolve(strict=True)
        actual_proposal, _proposal_file_sha = read_regular_json(
            proposal_path, "mapping proposal", "E-SRB-PROD-MAP-006"
        )
        if actual_proposal != expected_proposal:
            raise MappingDecisionError("E-SRB-PROD-MAP-006", "mapping proposal bindings do not match decision")
        try:
            gap_tool.validate_proposal(
                actual_proposal,
                expected_receipt["source_bindings"]["repository_catalog_identity_sha256"],
                gap_tool.planned_targets(inventory_tool.expected_inventory(repo_root, rings, ownership)),
            )
        except gap_tool.ProductionGapError as error:
            raise MappingDecisionError("E-SRB-PROD-MAP-006", f"mapping proposal validation failed: {error}") from error
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_VERIFY_PASS "
        f"receipt_identity={actual_receipt['receipt_identity_sha256']} "
        f"status={actual_receipt['processing_status']} proposal={actual_receipt['proposal_output_status']}"
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
    parser.add_argument("--mapping-decision", required=True)
    parser.add_argument("--canonical-repository-id", required=True)
    parser.add_argument("--remote-name", default="origin")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-canonical-production-mapping-decision-processor")
    subparsers = result.add_subparsers(dest="command", required=True)
    process = subparsers.add_parser("process")
    add_common_arguments(process)
    process.add_argument("--receipt-output", required=True)
    process.add_argument("--proposal-output")
    process.set_defaults(handler=process_command)
    verify = subparsers.add_parser("verify")
    add_common_arguments(verify)
    verify.add_argument("--receipt", required=True)
    verify.add_argument("--mapping-proposal")
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except inventory_tool.PhysicalExtractionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except MappingDecisionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError) as error:
        print(f"error[E-SRB-PROD-MAP-007]: mapping decision processing failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
