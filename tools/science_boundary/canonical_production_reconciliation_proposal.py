#!/usr/bin/env python3
"""Build and verify a non-authorizing path-level reconciliation proposal."""

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
import canonical_production_evidence_set as evidence_tool  # noqa: E402
import canonical_production_gap_assessor as gap_tool  # noqa: E402


SCHEMA = "sounio.physical-extraction-canonical-production-reconciliation-proposal.v1"
PROPOSAL_TYPE = "path-level-exact-source-mirror-proposal"
AUTHORITY_SCOPE = "reconciliation-description-and-review-only"
PROPOSAL_STATUS = "proposed-not-approved"
EXECUTION_AUTHORITY = "none"
DESTINATION_WRITE_AUTHORITY = "none"
SOURCE_REMOVAL_AUTHORITY = "none"
PRODUCTION_APPROVAL = "not-approved"
CUTOVER_STATUS = "not-executed"
PATH_DISPOSITIONS = {
    "add-source-byte-copy",
    "remove-destination-only",
    "replace-with-source-byte-copy",
    "retain-identical",
}
PRECONDITIONS = [
    "obtain-separate-explicit-destination-write-authorization",
    "reobserve-exact-source-and-destination-git-bindings",
    "review-every-destination-only-path-disposition",
    "review-every-changed-path-replacement",
    "establish-separate-execution-and-recovery-policy",
    "generate-and-review-an-immutable-pre-execution-ref-snapshot",
]
REVIEW_DECISIONS = [
    "accept-or-reject-source-only-additions",
    "accept-or-reject-changed-path-replacements",
    "accept-reject-or-preserve-destination-only-paths",
    "reissue-after-any-source-destination-or-policy-drift",
]
LIMITATIONS = [
    "proposal_never_grants_execution_or_destination_write_authority",
    "proposal_does_not_copy_replace_or_remove_any_repository_file",
    "proposal_does_not_create_commits_push_refs_tags_releases_or_registry_entries",
    "proposal_does_not_remove_or_relocate_canonical_source_files",
    "proposal_is_bound_to_sequential_point_in_time_snapshots_not_atomic_remote_state",
    "proposal_must_be_reissued_after_any_bound_git_or_byte_drift",
    "destination_only_removals_are_described_for_review_not_authorized",
    "exact_byte_mirroring_does_not_prove_scientific_truth",
    "proposal_does_not_assert_clinical_validation_or_clinical_authority",
    "production_approval_and_canonical_cutover_remain_absent",
]
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ReconciliationProposalError(ValueError):
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


def exact_keys(value: dict[str, Any], expected: set[str], label: str, code: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ReconciliationProposalError(
            code,
            f"{label} fields mismatch missing={','.join(sorted(expected - actual)) or '-'} "
            f"extra={','.join(sorted(actual - expected)) or '-'}",
        )


def read_regular_json(path: Path, label: str, code: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise ReconciliationProposalError(code, f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReconciliationProposalError(code, f"cannot parse {label}: {error}") from error
    if not isinstance(payload, dict):
        raise ReconciliationProposalError(code, f"{label} must be a JSON object")
    return payload, raw


def regular_file_path(value: str, label: str, code: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise ReconciliationProposalError(code, f"{label} must be a regular non-symlink file: {path}")
    try:
        return path.resolve(strict=True)
    except OSError as error:
        raise ReconciliationProposalError(code, f"cannot resolve {label}: {error}") from error


def validate_evidence(
    payload: dict[str, Any], target_id: str, expected_evidence_identity: str
) -> dict[str, Any]:
    code = "E-SRB-PROD-RECONCILE-002"
    exact_keys(
        payload,
        {
            "schema",
            "evidence_type",
            "authority_scope",
            "evidence_status",
            "next_required_action",
            "proposal_status",
            "execution_authority",
            "source_removal_authority",
            "canonical_production_approval",
            "canonical_cutover_execution_status",
            "source_bindings",
            "canonical_source",
            "targets",
            "validation",
            "proposed_execution_plan",
            "governance_prerequisites",
            "summary",
            "assurance_level",
            "limitations",
            "evidence_identity_sha256",
        },
        "production evidence set",
        code,
    )
    expected_constants = {
        "schema": evidence_tool.EVIDENCE_SCHEMA,
        "evidence_type": evidence_tool.EVIDENCE_TYPE,
        "authority_scope": evidence_tool.AUTHORITY_SCOPE,
        "evidence_status": "production-evidence-draft-gaps-observed",
        "next_required_action": "resolve-parity-or-validation-gaps-and-reissue-evidence",
        "proposal_status": PROPOSAL_STATUS,
        "execution_authority": EXECUTION_AUTHORITY,
        "source_removal_authority": SOURCE_REMOVAL_AUTHORITY,
        "canonical_production_approval": PRODUCTION_APPROVAL,
        "canonical_cutover_execution_status": CUTOVER_STATUS,
    }
    if any(payload.get(field) != value for field, value in expected_constants.items()):
        raise ReconciliationProposalError(code, "production evidence state is not an eligible non-authorizing gap")
    if payload.get("evidence_identity_sha256") != evidence_tool.identity(payload, "evidence_identity_sha256"):
        raise ReconciliationProposalError(code, "production evidence identity hash mismatch")
    if not SHA256.fullmatch(expected_evidence_identity):
        raise ReconciliationProposalError(code, "expected production evidence identity is invalid")
    if payload["evidence_identity_sha256"] != expected_evidence_identity:
        raise ReconciliationProposalError(code, "production evidence identity does not match the explicit review pin")
    if not isinstance(target_id, str) or not SAFE_ID.fullmatch(target_id):
        raise ReconciliationProposalError(code, "target ID is invalid")
    rows = payload.get("targets")
    if not isinstance(rows, list):
        raise ReconciliationProposalError(code, "production evidence targets are invalid")
    matches = [row for row in rows if isinstance(row, dict) and row.get("target_id") == target_id]
    if len(matches) != 1:
        raise ReconciliationProposalError(code, "target ID must select exactly one evidence row")
    target = matches[0]
    exact_keys(
        target,
        {
            "source_path",
            "ring",
            "target_id",
            "target_owner",
            "source_inventory",
            "destination_repository",
            "destination_inventory",
            "parity",
        },
        "target evidence",
        code,
    )
    parity = target.get("parity")
    if not isinstance(parity, dict) or parity.get("status") != "parity-gap-observed":
        raise ReconciliationProposalError(code, "selected target does not contain a parity gap")
    if not any(parity.get(field, 0) for field in ("missing_file_count", "extra_file_count", "changed_file_count")):
        raise ReconciliationProposalError(code, "selected target has no path-level reconciliation operation")
    return target


def repository_root(value: str, label: str, code: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink():
        raise ReconciliationProposalError(code, f"{label} must not be a symbolic link")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ReconciliationProposalError(code, f"cannot resolve {label}: {error}") from error
    if not resolved.is_dir():
        raise ReconciliationProposalError(code, f"{label} must be a directory")
    return resolved


def validate_relative_root(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ReconciliationProposalError("E-SRB-PROD-RECONCILE-002", "source path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ReconciliationProposalError("E-SRB-PROD-RECONCILE-002", "source path is unsafe")
    return path.as_posix()


def inventory_subtree(repository: Path, source_path: str) -> tuple[list[dict[str, Any]], int, str]:
    relative = validate_relative_root(source_path)
    candidate_input = repository / relative
    if candidate_input.is_symlink():
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-003", "source subtree root must not be a symbolic link"
        )
    candidate = candidate_input.resolve(strict=True)
    try:
        candidate.relative_to(repository)
    except ValueError as error:
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-003", "source subtree escapes repository root"
        ) from error
    files, total_bytes, tree = evidence_tool.scan_destination_tree(candidate)
    tracked = evidence_tool.tracked_files(repository, relative)
    prefix = f"{relative}/"
    tracked_relative = sorted(path[len(prefix) :] for path in tracked if path.startswith(prefix))
    if len(tracked_relative) != len(tracked) or tracked_relative != sorted(item["path"] for item in files):
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-003", "source subtree inventory is not exactly its tracked-file set"
        )
    return files, total_bytes, tree


def inventory_destination(repository: Path) -> tuple[list[dict[str, Any]], int, str]:
    files, total_bytes, tree = evidence_tool.scan_destination_tree(repository)
    tracked = evidence_tool.tracked_files(repository)
    if tracked != sorted(item["path"] for item in files):
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-004", "destination inventory is not exactly its tracked-file set"
        )
    return files, total_bytes, tree


def exact_observation(repository: Path, expected: dict[str, Any], code: str, label: str) -> dict[str, Any]:
    try:
        observation = gap_tool.git_observation(repository, expected["remote_name"])
    except (KeyError, gap_tool.ProductionGapError) as error:
        raise ReconciliationProposalError(code, f"cannot observe {label}: {error}") from error
    fields = ("remote_name", "remote_url", "branch", "head_oid", "worktree_status")
    if any(observation.get(field) != expected.get(field) for field in fields) or observation.get(
        "worktree_status"
    ) != "clean":
        raise ReconciliationProposalError(code, f"{label} Git binding does not match evidence")
    return observation


def file_state(item: dict[str, Any] | None) -> dict[str, Any] | None:
    if item is None:
        return None
    return {"size_bytes": item["size_bytes"], "sha256": item["sha256"]}


def path_plan(source: list[dict[str, Any]], destination: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_by_path = {row["path"]: row for row in source}
    destination_by_path = {row["path"]: row for row in destination}
    result: list[dict[str, Any]] = []
    for path in sorted(set(source_by_path) | set(destination_by_path)):
        source_row = source_by_path.get(path)
        destination_row = destination_by_path.get(path)
        if source_row is None:
            disposition = "remove-destination-only"
            after = None
        elif destination_row is None:
            disposition = "add-source-byte-copy"
            after = source_row
        elif file_state(source_row) == file_state(destination_row):
            disposition = "retain-identical"
            after = source_row
        else:
            disposition = "replace-with-source-byte-copy"
            after = source_row
        result.append(
            {
                "path": path,
                "disposition": disposition,
                "operation_authority": "none",
                "source": file_state(source_row),
                "destination_before": file_state(destination_row),
                "destination_after_if_separately_approved": file_state(after),
            }
        )
    return result


def expected_proposal(
    evidence_path: Path,
    source_root: Path,
    destination_root: Path,
    target_id: str,
    expected_evidence_identity: str,
) -> dict[str, Any]:
    evidence, evidence_raw = read_regular_json(
        evidence_path, "production evidence set", "E-SRB-PROD-RECONCILE-002"
    )
    target = validate_evidence(evidence, target_id, expected_evidence_identity)
    source_observation = exact_observation(
        source_root, evidence["canonical_source"], "E-SRB-PROD-RECONCILE-003", "source"
    )
    destination_observation = exact_observation(
        destination_root,
        target["destination_repository"],
        "E-SRB-PROD-RECONCILE-004",
        "destination",
    )
    source_files, source_bytes, source_tree = inventory_subtree(source_root, target["source_path"])
    destination_files, destination_bytes, destination_tree = inventory_destination(destination_root)
    source_expected = target["source_inventory"]
    destination_expected = target["destination_inventory"]
    if (
        len(source_files) != source_expected.get("file_count")
        or source_bytes != source_expected.get("total_bytes")
        or source_tree != source_expected.get("comparison_tree_sha256")
    ):
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-003", "source byte inventory does not match evidence"
        )
    if (
        len(destination_files) != destination_expected.get("file_count")
        or destination_bytes != destination_expected.get("total_bytes")
        or destination_tree != destination_expected.get("comparison_tree_sha256")
    ):
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-004", "destination byte inventory does not match evidence"
        )
    parity = evidence_tool.compare_files(source_files, destination_files)
    if parity != target["parity"]:
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-002", "independent path comparison does not match evidence"
        )
    rows = path_plan(source_files, destination_files)
    counts = {disposition: 0 for disposition in PATH_DISPOSITIONS}
    for row in rows:
        counts[row["disposition"]] += 1
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "proposal_type": PROPOSAL_TYPE,
        "authority_scope": AUTHORITY_SCOPE,
        "proposal_status": PROPOSAL_STATUS,
        "execution_authority": EXECUTION_AUTHORITY,
        "destination_write_authority": DESTINATION_WRITE_AUTHORITY,
        "source_removal_authority": SOURCE_REMOVAL_AUTHORITY,
        "canonical_production_approval": PRODUCTION_APPROVAL,
        "canonical_cutover_execution_status": CUTOVER_STATUS,
        "next_required_action": "human-review-only-no-execution",
        "evidence_binding": {
            "evidence_file_sha256": sha256_bytes(evidence_raw),
            "evidence_identity_sha256": evidence["evidence_identity_sha256"],
            "target_id": target_id,
        },
        "source_snapshot": {
            **source_observation,
            "source_path": target["source_path"],
            "file_count": len(source_files),
            "total_bytes": source_bytes,
            "comparison_tree_sha256": source_tree,
        },
        "destination_snapshot": {
            "repository_id": target["destination_repository"]["repository_id"],
            **destination_observation,
            "file_count": len(destination_files),
            "total_bytes": destination_bytes,
            "comparison_tree_sha256": destination_tree,
        },
        "proposed_strategy": {
            "strategy_id": "exact-source-mirror",
            "strategy_status": "described-for-review-not-authorized",
            "desired_postcondition": "destination-tree-equals-source-subtree-by-path-size-and-sha256",
            "destination_only_path_disposition": "remove-only-if-separately-explicitly-approved",
            "mutation_authority": "none",
        },
        "summary": {
            "path_count": len(rows),
            "mutation_path_count": len(rows) - counts["retain-identical"],
            "add_path_count": counts["add-source-byte-copy"],
            "replace_path_count": counts["replace-with-source-byte-copy"],
            "remove_path_count": counts["remove-destination-only"],
            "retain_path_count": counts["retain-identical"],
            "proposed_destination_file_count": len(source_files),
            "proposed_destination_total_bytes": source_bytes,
        },
        "path_plan_sha256": sha256_bytes(canonical_json(rows)),
        "path_plan": rows,
        "review_decisions_required": REVIEW_DECISIONS,
        "execution_preconditions_if_separately_authorized": PRECONDITIONS,
        "limitations": LIMITATIONS,
    }
    payload["proposal_identity_sha256"] = identity(payload, "proposal_identity_sha256")
    return payload


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-006", f"proposal output already exists: {path}"
        )
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise ReconciliationProposalError(
                "E-SRB-PROD-RECONCILE-006", f"proposal output appeared during write: {path}"
            ) from error
        temporary.unlink()
        temporary = None
        parent_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, str, str]:
    evidence = regular_file_path(
        args.evidence, "production evidence set", "E-SRB-PROD-RECONCILE-002"
    )
    source = repository_root(args.source_root, "source root", "E-SRB-PROD-RECONCILE-003")
    destination = repository_root(
        args.destination_root, "destination root", "E-SRB-PROD-RECONCILE-004"
    )
    if source == destination or source in destination.parents or destination in source.parents:
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-001", "source and destination roots must be separate"
        )
    return evidence, source, destination, args.target_id, args.expected_evidence_identity


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--destination-root", required=True)
    parser.add_argument("--target-id", required=True)
    parser.add_argument("--expected-evidence-identity", required=True)


def build_command(args: argparse.Namespace) -> int:
    payload = expected_proposal(*resolve_inputs(args))
    output_input = Path(args.output).expanduser()
    output = output_input if output_input.is_absolute() else Path.cwd() / output_input
    write_atomic(output, payload)
    summary = payload["summary"]
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_PASS "
        f"proposal_identity={payload['proposal_identity_sha256']} target={payload['evidence_binding']['target_id']} "
        f"add={summary['add_path_count']} replace={summary['replace_path_count']} "
        f"remove={summary['remove_path_count']} retain={summary['retain_path_count']} "
        f"status={payload['proposal_status']} authority={payload['execution_authority']}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    actual, _raw = read_regular_json(
        regular_file_path(
            args.proposal, "reconciliation proposal", "E-SRB-PROD-RECONCILE-007"
        ),
        "reconciliation proposal",
        "E-SRB-PROD-RECONCILE-007",
    )
    if actual.get("schema") != SCHEMA:
        raise ReconciliationProposalError("E-SRB-PROD-RECONCILE-007", "unsupported proposal schema")
    if actual.get("proposal_identity_sha256") != identity(actual, "proposal_identity_sha256"):
        raise ReconciliationProposalError("E-SRB-PROD-RECONCILE-007", "proposal identity hash mismatch")
    expected = expected_proposal(*resolve_inputs(args))
    if actual != expected:
        raise ReconciliationProposalError(
            "E-SRB-PROD-RECONCILE-007", "reconciliation proposal does not match current bound inputs"
        )
    print(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_VERIFY_PASS "
        f"proposal_identity={actual['proposal_identity_sha256']} status={actual['proposal_status']}"
    )
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-canonical-production-reconciliation-proposal")
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    common_arguments(build)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=build_command)
    verify = subparsers.add_parser("verify")
    common_arguments(verify)
    verify.add_argument("--proposal", required=True)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except ReconciliationProposalError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(
            f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_REFUSED reason={error}",
            file=sys.stderr,
        )
        return 1
    except (KeyError, OSError, TypeError, ValueError, gap_tool.ProductionGapError) as error:
        print(f"error[E-SRB-PROD-RECONCILE-008]: reconciliation proposal operation failed: {error}", file=sys.stderr)
        print(
            f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_REFUSED reason={error}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
