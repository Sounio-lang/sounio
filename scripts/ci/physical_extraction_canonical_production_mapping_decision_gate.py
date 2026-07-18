#!/usr/bin/env python3
"""Adversarial gate for canonical-production mapping decision processing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import physical_extraction_canonical_production_gap_gate as gap_gate  # noqa: E402


TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_mapping_decision_processor.py"
GAP_TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_gap_assessor.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_canonical_production_mapping_decision_gate.sh"
DECISION_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-mapping-decision.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-mapping-decision-receipt.v1.schema.json"
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
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, expected: int | set[int] = 0, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(
        command,
        cwd=cwd,
        env={**os.environ, "LANG": "C", "LC_ALL": "C", "TZ": "UTC", "GIT_TERMINAL_PROMPT": "0"},
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.returncode not in expected_codes:
        raise AssertionError(
            f"command returned {result.returncode}, expected {sorted(expected_codes)}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def digest(payload: object, field: str | None = None) -> str:
    value = json.loads(json.dumps(payload))
    if field is not None and isinstance(value, dict):
        value.pop(field, None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    return path


def clone_json(original: Path, destination: Path, mutate, *, rehash_field: str | None) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash_field is not None:
        payload[rehash_field] = digest(payload, rehash_field)
    return write_json(destination, payload)


def git(arguments: list[str], cwd: Path) -> str:
    return run(["git", *arguments], cwd=cwd).stdout.strip()


def source_state(repo: Path) -> dict[str, str]:
    return {
        "head": git(["rev-parse", "HEAD"], repo),
        "tree": git(["rev-parse", "HEAD^{tree}"], repo),
        "index": git(["ls-files", "--stage"], repo),
        "status": git(["status", "--porcelain=v1", "--untracked-files=all"], repo),
        "remote": git(["--git-dir", "../remotes/source.git", "rev-parse", "refs/heads/main"], repo),
    }


def target_rows(fixture: dict[str, object], actions: dict[str, str]) -> list[dict[str, object]]:
    catalog = {str(row["repository_id"]): row for row in fixture["catalog_payload"]["repositories"]}
    governed = {
        "distribution:pkg": ("packages/pkg", "future-maintainers", "destination-pkg"),
        "distribution:research": ("research", "future-maintainers", "destination-research"),
    }
    rows: list[dict[str, object]] = []
    for target_id in sorted(governed):
        source_path, owner, default_repository = governed[target_id]
        action = actions[target_id]
        if action == "reuse-observed":
            destination = catalog[default_repository]
            row = {
                "source_path": source_path,
                "target_id": target_id,
                "target_owner": owner,
                "action": action,
                "repository_id": default_repository,
                "remote_url": destination["remote_url"],
                "branch": destination["default_branch"],
                "visibility": None,
                "rationale": None,
            }
        elif action == "request-new":
            suffix = "pkg" if target_id.endswith(":pkg") else "research"
            row = {
                "source_path": source_path,
                "target_id": target_id,
                "target_owner": owner,
                "action": action,
                "repository_id": f"requested-{suffix}",
                "remote_url": f"https://example.invalid/SounioFixture/requested-{suffix}.git",
                "branch": "main",
                "visibility": "PRIVATE",
                "rationale": None,
            }
        else:
            row = {
                "source_path": source_path,
                "target_id": target_id,
                "target_owner": owner,
                "action": "revise-target",
                "repository_id": None,
                "remote_url": None,
                "branch": None,
                "visibility": None,
                "rationale": "The governed distribution target needs founder review before repository selection.",
            }
        rows.append(row)
    return rows


def create_decision(
    path: Path,
    fixture: dict[str, object],
    actions: dict[str, str],
    *,
    catalog_payload: dict[str, object] | None = None,
) -> tuple[Path, dict[str, object]]:
    catalog = catalog_payload or fixture["catalog_payload"]
    repo = Path(fixture["repo"])
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-canonical-production-mapping-decision.v1",
        "decision_type": "canonical-production-target-mapping-selection",
        "authority_scope": "mapping-proposal-input-only",
        "decision_status": "human-selection-recorded-not-approved",
        "source_decision_evidence": {
            "issue_url": "https://github.com/SounioFixture/sounio/issues/1122",
            "response_url": "https://github.com/SounioFixture/sounio/issues/1122#issuecomment-fixture",
            "responder_label": "fixture-founder",
            "response_body_sha256": hashlib.sha256(b"fixture human mapping selection\n").hexdigest(),
            "submitted_at_utc": "2026-07-18T00:00:00Z",
            "evidence_status": "transcribed-not-authenticated",
        },
        "bindings": {
            "repository_catalog_identity_sha256": catalog["catalog_identity_sha256"],
            "repository_catalog_observed_at_utc": catalog["observed_at_utc"],
            "canonical_repository_id": "sounio-fixture",
            "canonical_repository_branch": git(["branch", "--show-current"], repo),
            "canonical_repository_head_oid": git(["rev-parse", "HEAD"], repo),
        },
        "authorized_operations": AUTHORIZED_OPERATIONS,
        "prohibited_operations": PROHIBITED_OPERATIONS,
        "targets": target_rows(fixture, actions),
        "limitations": DECISION_LIMITATIONS,
    }
    payload["decision_identity_sha256"] = digest(payload, "decision_identity_sha256")
    return write_json(path, payload), payload


def process_command(
    fixture: dict[str, object],
    decision: Path,
    receipt: Path,
    *,
    proposal: Path | None,
    catalog: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(TOOL),
        "process",
        "--repo-root",
        str(fixture["repo"]),
        "--rings",
        str(fixture["rings"]),
        "--ownership",
        str(fixture["ownership"]),
        "--repository-catalog",
        str(catalog or fixture["catalog"]),
        "--mapping-decision",
        str(decision),
        "--canonical-repository-id",
        "sounio-fixture",
        "--receipt-output",
        str(receipt),
    ]
    if proposal is not None:
        command.extend(["--proposal-output", str(proposal)])
    return command


def verify_command(
    fixture: dict[str, object],
    decision: Path,
    receipt: Path,
    *,
    proposal: Path | None,
    catalog: Path | None = None,
) -> list[str]:
    command = [
        sys.executable,
        str(TOOL),
        "verify",
        "--repo-root",
        str(fixture["repo"]),
        "--rings",
        str(fixture["rings"]),
        "--ownership",
        str(fixture["ownership"]),
        "--repository-catalog",
        str(catalog or fixture["catalog"]),
        "--mapping-decision",
        str(decision),
        "--canonical-repository-id",
        "sounio-fixture",
        "--receipt",
        str(receipt),
    ]
    if proposal is not None:
        command.extend(["--mapping-proposal", str(proposal)])
    return command


def assert_refusal(
    command: list[str],
    code: str,
    *outputs: Path,
) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    for output in outputs:
        check(not output.exists(), f"refused processing left output: {output}")
    check(code in result.stderr, f"refusal lacks {code}")
    check(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_REFUSED" in result.stderr,
        "refusal lacks marker",
    )
    return result


def assert_static_contracts() -> None:
    decision = json.loads(DECISION_SCHEMA.read_text(encoding="utf-8"))
    receipt = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(decision["properties"]["authority_scope"]["const"] == "mapping-proposal-input-only", "decision scope drift")
    check(decision["properties"]["authorized_operations"]["const"] == AUTHORIZED_OPERATIONS, "decision operation drift")
    check(decision["properties"]["prohibited_operations"]["const"] == PROHIBITED_OPERATIONS, "decision prohibition drift")
    check(decision["properties"]["limitations"]["const"] == DECISION_LIMITATIONS, "decision limitations drift")
    check(receipt["properties"]["execution_authority"]["const"] == "none", "receipt grants authority")
    check(receipt["properties"]["canonical_cutover_execution_status"]["const"] == "not-executed", "receipt claims cutover")
    check(receipt["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drift")
    statuses = receipt["properties"]["processing_status"]["enum"]
    check(all("approved" not in status and "authorized" not in status for status in statuses), "receipt status overstates authority")
    shell = COMPOSED_GATE.read_text(encoding="utf-8")
    check("SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_MADAROS_BIN" in shell, "composed gate omits compiler input")
    check("SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_MADAROS_BIN" in shell, "composed gate omits compiler forwarding")
    check("physical_extraction_canonical_production_gap_gate.sh" in shell, "composed gate omits prior stack")
    check("physical_extraction_canonical_production_mapping_decision_gate.py" in shell, "composed gate omits focused decision gate")


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-canonical-production-mapping-decision-") as temporary:
        work = Path(temporary)
        fixture_a = gap_gate.create_fixture(work / "a")
        fixture_b = gap_gate.create_fixture(work / "b")
        source_a = Path(fixture_a["repo"])
        before = source_state(source_a)
        reuse_actions = {"distribution:pkg": "reuse-observed", "distribution:research": "reuse-observed"}
        reuse_a, reuse_payload = create_decision(work / "reuse-a.json", fixture_a, reuse_actions)
        reuse_b, _reuse_b_payload = create_decision(work / "reuse-b.json", fixture_b, reuse_actions)
        check(reuse_a.read_bytes() == reuse_b.read_bytes(), "equivalent decision records depend on physical root")

        receipt_a = work / "reuse-receipt-a.json"
        receipt_b = work / "reuse-receipt-b.json"
        proposal_a = work / "reuse-proposal-a.json"
        proposal_b = work / "reuse-proposal-b.json"
        result_reuse = run(process_command(fixture_a, reuse_a, receipt_a, proposal=proposal_a))
        run(process_command(fixture_b, reuse_b, receipt_b, proposal=proposal_b))
        receipt_payload = json.loads(receipt_a.read_text(encoding="ascii"))
        proposal_payload = json.loads(proposal_a.read_text(encoding="ascii"))
        check("status=proposal-input-complete" in result_reuse.stdout, "reuse result marker drift")
        check(receipt_a.read_bytes() == receipt_b.read_bytes(), "equivalent receipts depend on physical root")
        check(proposal_a.read_bytes() == proposal_b.read_bytes(), "equivalent proposals depend on physical root")
        check(receipt_payload["processing_status"] == "proposal-input-complete", "reuse status is wrong")
        check(receipt_payload["next_required_action"] == "review-proposed-not-approved-mapping", "reuse next action drift")
        check(receipt_payload["execution_authority"] == "none", "reuse receipt grants authority")
        check(receipt_payload["canonical_cutover_execution_status"] == "not-executed", "reuse receipt claims cutover")
        check(receipt_payload["proposal_output_status"] == "emitted-proposed-not-approved", "reuse proposal status drift")
        check(receipt_payload["summary"] == {"proposal_mapping_count": 2, "request_new_count": 0, "reuse_observed_count": 2, "revise_target_count": 0, "target_count": 2}, "reuse summary drift")
        check(all(row["result_status"] == "observed-repository-reuse-ready-for-proposal" for row in receipt_payload["targets"]), "reuse result overstates or regresses")
        check(proposal_payload["proposal_status"] == "proposed-not-approved", "generated proposal overstates status")
        check(all(row["mapping_status"] == "proposed-not-approved" for row in proposal_payload["mappings"]), "generated mapping overstates status")
        check(receipt_payload["proposal_identity_sha256"] == proposal_payload["proposal_identity_sha256"], "receipt/proposal identity mismatch")
        run(verify_command(fixture_a, reuse_a, receipt_a, proposal=proposal_a))

        gap_assessment = work / "gap-assessment.json"
        gap_result = run(gap_gate.assess_command(fixture_a, gap_assessment, proposal=proposal_a))
        gap_payload = json.loads(gap_assessment.read_text(encoding="ascii"))
        check("status=production-evidence-and-human-decision-required" in gap_result.stdout, "generated proposal bypassed gap status")
        check(gap_payload["execution_authority"] == "none", "generated proposal granted gap authority")
        check(next(row for row in gap_payload["prerequisites"] if row["prerequisite_id"] == "explicit-human-cutover-decision")["status"] == "missing", "mapping selection became cutover decision")
        check(source_state(source_a) == before, "reuse processing changed source or refs")

        request_actions = {"distribution:pkg": "reuse-observed", "distribution:research": "request-new"}
        request_decision, _request_payload = create_decision(work / "request.json", fixture_a, request_actions)
        request_receipt = work / "request-receipt.json"
        result_request = run(process_command(fixture_a, request_decision, request_receipt, proposal=None))
        request_receipt_payload = json.loads(request_receipt.read_text(encoding="ascii"))
        check("status=destination-repository-creation-required" in result_request.stdout, "request-new marker drift")
        check(request_receipt_payload["processing_status"] == "destination-repository-creation-required", "request-new status is wrong")
        check(request_receipt_payload["proposal_output_status"] == "not-emitted", "request-new emitted proposal")
        check(request_receipt_payload["proposal_identity_sha256"] is None, "request-new synthesized proposal identity")
        check(request_receipt_payload["summary"]["request_new_count"] == 1, "request-new count is wrong")
        check(request_receipt_payload["next_required_action"] == "provision-repositories-reobserve-catalog-and-reconfirm-selection", "request-new reconfirmation missing")
        run(verify_command(fixture_a, request_decision, request_receipt, proposal=None))

        revise_actions = {"distribution:pkg": "request-new", "distribution:research": "revise-target"}
        revise_decision, _revise_payload = create_decision(work / "revise.json", fixture_a, revise_actions)
        revise_receipt = work / "revise-receipt.json"
        result_revise = run(process_command(fixture_a, revise_decision, revise_receipt, proposal=None))
        revise_receipt_payload = json.loads(revise_receipt.read_text(encoding="ascii"))
        check("status=ownership-policy-review-required" in result_revise.stdout, "revise marker drift")
        check(revise_receipt_payload["processing_status"] == "ownership-policy-review-required", "revise status is wrong")
        check(revise_receipt_payload["summary"]["request_new_count"] == 1, "mixed request count is wrong")
        check(revise_receipt_payload["summary"]["revise_target_count"] == 1, "mixed revise count is wrong")
        check(revise_receipt_payload["next_required_action"] == "revise-governed-target-and-repeat-human-selection", "revise priority drift")
        check(revise_receipt_payload["execution_authority"] == "none", "revise receipt grants authority")
        run(verify_command(fixture_a, revise_decision, revise_receipt, proposal=None))

        negative_index = 0

        def refuse_decision(name: str, mutate, code: str = "E-SRB-PROD-MAP-002", *, base: Path = reuse_a, rehash: bool = True) -> None:
            nonlocal negative_index
            negative_index += 1
            decision = clone_json(base, work / f"bad-{negative_index}-{name}.json", mutate, rehash_field="decision_identity_sha256" if rehash else None)
            receipt = work / f"bad-{negative_index}-{name}-receipt.json"
            proposal = work / f"bad-{negative_index}-{name}-proposal.json"
            assert_refusal(process_command(fixture_a, decision, receipt, proposal=proposal), code, receipt, proposal)

        refuse_decision("identity", lambda payload: payload["bindings"].__setitem__("canonical_repository_head_oid", "0" * 40), rehash=False)
        refuse_decision("authorized-ops", lambda payload: payload["authorized_operations"].append("create-or-modify-repositories"))
        refuse_decision("prohibited-ops", lambda payload: payload["prohibited_operations"].pop())
        refuse_decision("limitations", lambda payload: payload["limitations"].pop())
        refuse_decision("incomplete", lambda payload: payload["targets"].pop())
        refuse_decision("extra", lambda payload: payload["targets"].append({**payload["targets"][0], "target_id": "distribution:extra"}))
        refuse_decision("duplicate", lambda payload: payload["targets"].__setitem__(1, dict(payload["targets"][0])))
        refuse_decision("unsorted", lambda payload: payload["targets"].reverse())
        refuse_decision("owner", lambda payload: payload["targets"][0].__setitem__("target_owner", "wrong-owner"))
        refuse_decision("source", lambda payload: payload["targets"][0].__setitem__("source_path", "wrong/path"))
        refuse_decision("response-hash", lambda payload: payload["source_decision_evidence"].__setitem__("response_body_sha256", "bad"))
        refuse_decision("responder", lambda payload: payload["source_decision_evidence"].__setitem__("responder_label", "bad label"))
        refuse_decision("evidence-status", lambda payload: payload["source_decision_evidence"].__setitem__("evidence_status", "authenticated"))
        refuse_decision("catalog-binding", lambda payload: payload["bindings"].__setitem__("repository_catalog_identity_sha256", "0" * 64))
        refuse_decision("catalog-time", lambda payload: payload["bindings"].__setitem__("repository_catalog_observed_at_utc", "2026-07-19T00:00:00Z"))
        refuse_decision("canonical-id", lambda payload: payload["bindings"].__setitem__("canonical_repository_id", "wrong-source"))
        refuse_decision("canonical-branch", lambda payload: payload["bindings"].__setitem__("canonical_repository_branch", "other"))
        refuse_decision("canonical-head", lambda payload: payload["bindings"].__setitem__("canonical_repository_head_oid", "0" * 40))
        refuse_decision("reuse-visibility", lambda payload: payload["targets"][0].__setitem__("visibility", "PRIVATE"))
        refuse_decision("reuse-rationale", lambda payload: payload["targets"][0].__setitem__("rationale", "not allowed"))
        refuse_decision("reuse-url", lambda payload: payload["targets"][0].__setitem__("remote_url", "https://example.invalid/changed.git"))
        refuse_decision("reuse-branch", lambda payload: payload["targets"][0].__setitem__("branch", "changed"))
        refuse_decision("reuse-missing", lambda payload: payload["targets"][0].__setitem__("repository_id", "absent"))

        def duplicate_destination(payload: dict[str, object]) -> None:
            payload["targets"][1]["repository_id"] = payload["targets"][0]["repository_id"]
            payload["targets"][1]["remote_url"] = payload["targets"][0]["remote_url"]

        refuse_decision("destination-reuse", duplicate_destination)

        def request_shape(payload: dict[str, object]) -> None:
            payload["targets"][1]["visibility"] = None

        refuse_decision("request-shape", request_shape, base=request_decision)

        def request_id_collision(payload: dict[str, object]) -> None:
            payload["targets"][1]["repository_id"] = "destination-research"

        refuse_decision("request-id-collision", request_id_collision, base=request_decision)

        def request_url_collision(payload: dict[str, object]) -> None:
            destination = next(row for row in fixture_a["catalog_payload"]["repositories"] if row["repository_id"] == "destination-research")
            payload["targets"][1]["remote_url"] = destination["remote_url"]

        refuse_decision("request-url-collision", request_url_collision, base=request_decision)

        def revise_shape(payload: dict[str, object]) -> None:
            payload["targets"][1]["rationale"] = None

        refuse_decision("revise-shape", revise_shape, base=revise_decision)

        archived_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        for row in archived_catalog_payload["repositories"]:
            if row["repository_id"] == "destination-pkg":
                row["archived"] = True
        archived_catalog_payload["catalog_identity_sha256"] = digest(archived_catalog_payload, "catalog_identity_sha256")
        archived_catalog = write_json(work / "archived-catalog.json", archived_catalog_payload)
        archived_decision, _ = create_decision(work / "archived-decision.json", fixture_a, reuse_actions, catalog_payload=archived_catalog_payload)
        archived_receipt = work / "archived-receipt.json"
        archived_proposal = work / "archived-proposal.json"
        assert_refusal(process_command(fixture_a, archived_decision, archived_receipt, proposal=archived_proposal, catalog=archived_catalog), "E-SRB-PROD-MAP-002", archived_receipt, archived_proposal)

        empty_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        for row in empty_catalog_payload["repositories"]:
            if row["repository_id"] == "destination-pkg":
                row["is_empty"] = True
                row["head_oid"] = None
        empty_catalog_payload["catalog_identity_sha256"] = digest(empty_catalog_payload, "catalog_identity_sha256")
        empty_catalog = write_json(work / "empty-catalog.json", empty_catalog_payload)
        empty_decision, _ = create_decision(work / "empty-decision.json", fixture_a, reuse_actions, catalog_payload=empty_catalog_payload)
        empty_receipt = work / "empty-receipt.json"
        empty_proposal = work / "empty-proposal.json"
        assert_refusal(process_command(fixture_a, empty_decision, empty_receipt, proposal=empty_proposal, catalog=empty_catalog), "E-SRB-PROD-MAP-002", empty_receipt, empty_proposal)

        readonly_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        for row in readonly_catalog_payload["repositories"]:
            if row["repository_id"] == "destination-pkg":
                row["observed_permission"] = "READ"
        readonly_catalog_payload["catalog_identity_sha256"] = digest(readonly_catalog_payload, "catalog_identity_sha256")
        readonly_catalog = write_json(work / "readonly-catalog.json", readonly_catalog_payload)
        readonly_decision, _ = create_decision(work / "readonly-decision.json", fixture_a, reuse_actions, catalog_payload=readonly_catalog_payload)
        readonly_receipt = work / "readonly-receipt.json"
        readonly_proposal = work / "readonly-proposal.json"
        assert_refusal(process_command(fixture_a, readonly_decision, readonly_receipt, proposal=readonly_proposal, catalog=readonly_catalog), "E-SRB-PROD-MAP-002", readonly_receipt, readonly_proposal)

        changed_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        changed_catalog_payload["observed_at_utc"] = "2026-07-18T01:00:00Z"
        changed_catalog_payload["catalog_identity_sha256"] = digest(changed_catalog_payload, "catalog_identity_sha256")
        changed_catalog = write_json(work / "changed-catalog.json", changed_catalog_payload)
        stale_receipt = work / "stale-catalog-receipt.json"
        stale_proposal = work / "stale-catalog-proposal.json"
        assert_refusal(process_command(fixture_a, reuse_a, stale_receipt, proposal=stale_proposal, catalog=changed_catalog), "E-SRB-PROD-MAP-002", stale_receipt, stale_proposal)

        occupied_receipt = write(work / "occupied-receipt.json", "preserve receipt\n")
        occupied_proposal = work / "occupied-side-proposal.json"
        occupied_result = run(process_command(fixture_a, reuse_a, occupied_receipt, proposal=occupied_proposal), expected=1)
        check(occupied_receipt.read_text(encoding="utf-8") == "preserve receipt\n", "occupied receipt was overwritten")
        check(not occupied_proposal.exists(), "occupied receipt refusal left proposal")
        check("E-SRB-PROD-MAP-005" in occupied_result.stderr, "occupied receipt code drift")

        occupied_proposal = write(work / "occupied-proposal.json", "preserve proposal\n")
        uncreated_receipt = work / "occupied-proposal-receipt.json"
        occupied_result = run(process_command(fixture_a, reuse_a, uncreated_receipt, proposal=occupied_proposal), expected=1)
        check(occupied_proposal.read_text(encoding="utf-8") == "preserve proposal\n", "occupied proposal was overwritten")
        check(not uncreated_receipt.exists(), "occupied proposal refusal left receipt")
        check("E-SRB-PROD-MAP-005" in occupied_result.stderr, "occupied proposal code drift")

        same_output = work / "same-output.json"
        assert_refusal(process_command(fixture_a, reuse_a, same_output, proposal=same_output), "E-SRB-PROD-MAP-005", same_output)
        unwanted_proposal = work / "unwanted-proposal.json"
        unwanted_receipt = work / "unwanted-receipt.json"
        assert_refusal(process_command(fixture_a, request_decision, unwanted_receipt, proposal=unwanted_proposal), "E-SRB-PROD-MAP-005", unwanted_receipt, unwanted_proposal)
        missing_proposal_receipt = work / "missing-proposal-receipt.json"
        assert_refusal(process_command(fixture_a, reuse_a, missing_proposal_receipt, proposal=None), "E-SRB-PROD-MAP-005", missing_proposal_receipt)

        forged_receipt = clone_json(receipt_a, work / "forged-receipt.json", lambda payload: payload.__setitem__("execution_authority", "granted"), rehash_field=None)
        assert_refusal(verify_command(fixture_a, reuse_a, forged_receipt, proposal=proposal_a), "E-SRB-PROD-MAP-006")
        rehashed_receipt = clone_json(receipt_a, work / "rehashed-receipt.json", lambda payload: payload.__setitem__("execution_authority", "granted"), rehash_field="receipt_identity_sha256")
        assert_refusal(verify_command(fixture_a, reuse_a, rehashed_receipt, proposal=proposal_a), "E-SRB-PROD-MAP-006")
        forged_proposal = clone_json(proposal_a, work / "forged-proposal.json", lambda payload: payload["mappings"][0].__setitem__("mapping_status", "approved"), rehash_field=None)
        assert_refusal(verify_command(fixture_a, reuse_a, receipt_a, proposal=forged_proposal), "E-SRB-PROD-MAP-006")
        rehashed_proposal = clone_json(proposal_a, work / "rehashed-proposal.json", lambda payload: payload["mappings"][0].__setitem__("mapping_status", "approved"), rehash_field="proposal_identity_sha256")
        assert_refusal(verify_command(fixture_a, reuse_a, receipt_a, proposal=rehashed_proposal), "E-SRB-PROD-MAP-006")
        assert_refusal(verify_command(fixture_a, reuse_a, receipt_a, proposal=None), "E-SRB-PROD-MAP-006")
        assert_refusal(verify_command(fixture_a, request_decision, request_receipt, proposal=proposal_a), "E-SRB-PROD-MAP-006")

        source_file = source_a / "packages" / "pkg" / "README.md"
        original = source_file.read_bytes()
        source_file.write_bytes(original + b"drift\n")
        drift_receipt = work / "source-drift-receipt.json"
        drift_proposal = work / "source-drift-proposal.json"
        assert_refusal(process_command(fixture_a, reuse_a, drift_receipt, proposal=drift_proposal), "E-SRB-PROD-MAP-002", drift_receipt, drift_proposal)
        source_file.write_bytes(original)
        check(source_state(source_a) == before, "source drift test did not restore source state")
        check(not any(path.name.endswith(".staging") for path in work.rglob("*")), "gate left staging files")

        print(
            "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_WITNESS "
            f"decision_identity={reuse_payload['decision_identity_sha256']} "
            f"receipt_identity={receipt_payload['receipt_identity_sha256']} "
            f"proposal_identity={proposal_payload['proposal_identity_sha256']} "
            f"targets={receipt_payload['summary']['target_count']} "
            f"status={receipt_payload['processing_status']} authority={receipt_payload['execution_authority']}"
        )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_MAPPING_DECISION_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
