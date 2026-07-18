#!/usr/bin/env python3
"""Adversarial gate for the non-authorizing canonical-production gap assessor."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import physical_extraction_inventory_gate as inventory_gate  # noqa: E402
import physical_extraction_canonical_cutover_approval_gate as approval_gate  # noqa: E402


TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_gap_assessor.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_canonical_production_gap_gate.sh"
CATALOG_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-repository-catalog.v1.schema.json"
PROPOSAL_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-mapping-proposal.v1.schema.json"
ASSESSMENT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-gap-assessment.v1.schema.json"
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
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(
    command: list[str],
    *,
    expected: int | set[int] = 0,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
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


def git(arguments: list[str], cwd: Path) -> str:
    return run(["git", *arguments], cwd=cwd).stdout.strip()


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


def repository_row(
    repository_id: str,
    name_with_owner: str,
    remote_url: str,
    head_oid: str | None,
    *,
    archived: bool = False,
    is_empty: bool = False,
) -> dict[str, object]:
    return {
        "repository_id": repository_id,
        "name_with_owner": name_with_owner,
        "remote_url": remote_url,
        "default_branch": "main",
        "head_oid": head_oid,
        "visibility": "PRIVATE",
        "archived": archived,
        "is_empty": is_empty,
        "observed_permission": "ADMIN",
    }


def create_catalog(path: Path, repositories: list[dict[str, object]]) -> tuple[Path, dict[str, object]]:
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-canonical-production-repository-catalog.v1",
        "catalog_type": "observed-hosting-repository-catalog",
        "authority_scope": "supplied-repository-metadata-observation",
        "organization": "SounioFixture",
        "observed_at_utc": "2026-07-17T00:00:00Z",
        "repositories": sorted(repositories, key=lambda row: str(row["repository_id"])),
        "limitations": CATALOG_LIMITATIONS,
    }
    payload["catalog_identity_sha256"] = digest(payload, "catalog_identity_sha256")
    return write_json(path, payload), payload


def create_proposal(
    path: Path,
    catalog: dict[str, object],
    destination_rows: list[dict[str, object]],
) -> tuple[Path, dict[str, object]]:
    destinations = {str(row["repository_id"]): row for row in destination_rows}
    mappings = [
        {
            "target_id": "distribution:pkg",
            "target_owner": "future-maintainers",
            "repository_id": "destination-pkg",
            "remote_url": destinations["destination-pkg"]["remote_url"],
            "branch": "main",
            "expected_head_oid": destinations["destination-pkg"]["head_oid"],
            "mapping_status": "proposed-not-approved",
        },
        {
            "target_id": "distribution:research",
            "target_owner": "future-maintainers",
            "repository_id": "destination-research",
            "remote_url": destinations["destination-research"]["remote_url"],
            "branch": "main",
            "expected_head_oid": destinations["destination-research"]["head_oid"],
            "mapping_status": "proposed-not-approved",
        },
    ]
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-canonical-production-mapping-proposal.v1",
        "proposal_type": "target-repository-mapping-proposal",
        "authority_scope": "repository-target-proposal-only",
        "proposal_status": "proposed-not-approved",
        "catalog_identity_sha256": catalog["catalog_identity_sha256"],
        "canonical_repository_id": "sounio-fixture",
        "mappings": mappings,
        "limitations": PROPOSAL_LIMITATIONS,
    }
    payload["proposal_identity_sha256"] = digest(payload, "proposal_identity_sha256")
    return write_json(path, payload), payload


def initialize_destination(work: Path, name: str, content: str) -> tuple[Path, dict[str, object]]:
    repository = work / name
    write(repository / "README.md", content)
    remote = work / "remotes" / f"{name}.git"
    state = approval_gate.initialize_git_repository(repository, remote, f"../remotes/{name}.git")
    row = repository_row(
        name,
        f"SounioFixture/{name}",
        state["remote_url"],
        state["head_oid"],
    )
    return repository, row


def create_fixture(work: Path) -> dict[str, object]:
    repo = work / "source-repo"
    rings, ownership = inventory_gate.create_fixture(repo)
    source_state = approval_gate.initialize_git_repository(
        repo,
        work / "remotes" / "source.git",
        "../remotes/source.git",
    )
    package_repository, package_row = initialize_destination(work, "destination-pkg", "package destination\n")
    research_repository, research_row = initialize_destination(
        work, "destination-research", "research destination\n"
    )
    source_row = repository_row(
        "sounio-fixture",
        "SounioFixture/sounio",
        source_state["remote_url"],
        source_state["head_oid"],
    )
    catalog, catalog_payload = create_catalog(
        work / "repository-catalog.json",
        [package_row, research_row, source_row],
    )
    proposal, proposal_payload = create_proposal(
        work / "mapping-proposal.json",
        catalog_payload,
        [package_row, research_row],
    )
    return {
        "repo": repo,
        "rings": rings,
        "ownership": ownership,
        "catalog": catalog,
        "catalog_payload": catalog_payload,
        "proposal": proposal,
        "proposal_payload": proposal_payload,
        "destination_repositories": [package_repository, research_repository],
    }


def assess_command(fixture: dict[str, object], output: Path, *, proposal: Path | None) -> list[str]:
    command = [
        sys.executable,
        str(TOOL),
        "assess",
        "--repo-root",
        str(fixture["repo"]),
        "--rings",
        str(fixture["rings"]),
        "--ownership",
        str(fixture["ownership"]),
        "--repository-catalog",
        str(fixture["catalog"]),
        "--canonical-repository-id",
        "sounio-fixture",
        "--remote-name",
        "origin",
        "--output",
        str(output),
    ]
    if proposal is not None:
        command.extend(["--mapping-proposal", str(proposal)])
    return command


def verify_command(
    fixture: dict[str, object], assessment: Path, *, proposal: Path | None, catalog: Path | None = None
) -> list[str]:
    command = [
        sys.executable,
        str(TOOL),
        "verify",
        "--assessment",
        str(assessment),
        "--repo-root",
        str(fixture["repo"]),
        "--rings",
        str(fixture["rings"]),
        "--ownership",
        str(fixture["ownership"]),
        "--repository-catalog",
        str(catalog or fixture["catalog"]),
        "--canonical-repository-id",
        "sounio-fixture",
        "--remote-name",
        "origin",
    ]
    if proposal is not None:
        command.extend(["--mapping-proposal", str(proposal)])
    return command


def source_state(repo: Path) -> dict[str, str]:
    return {
        "head": git(["rev-parse", "HEAD"], repo),
        "tree": git(["rev-parse", "HEAD^{tree}"], repo),
        "index": git(["ls-files", "--stage"], repo),
        "status": git(["status", "--porcelain=v1", "--untracked-files=all"], repo),
        "remote": git(["--git-dir", "../remotes/source.git", "rev-parse", "refs/heads/main"], repo),
    }


def assert_refusal(command: list[str], output: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if output is not None:
        check(not output.exists(), f"refused assessment left output: {output}")
    check(code in result.stderr, f"refusal lacks {code}")
    check(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_ASSESSMENT_REFUSED" in result.stderr,
        "refusal lacks marker",
    )
    return result


def assert_static_contracts() -> None:
    catalog = json.loads(CATALOG_SCHEMA.read_text(encoding="utf-8"))
    proposal = json.loads(PROPOSAL_SCHEMA.read_text(encoding="utf-8"))
    assessment = json.loads(ASSESSMENT_SCHEMA.read_text(encoding="utf-8"))
    check(catalog["properties"]["authority_scope"]["const"] == "supplied-repository-metadata-observation", "catalog scope drift")
    check("catalog_does_not_approve_target_mapping_or_source_removal" in catalog["properties"]["limitations"]["const"], "catalog approval limitation absent")
    check(proposal["properties"]["proposal_status"]["const"] == "proposed-not-approved", "proposal status drift")
    check("proposal_is_not_canonical_production_approval" in proposal["properties"]["limitations"]["const"], "proposal approval limitation absent")
    check(assessment["properties"]["execution_authority"]["const"] == "none", "assessment grants authority")
    check(assessment["properties"]["canonical_cutover_execution_status"]["const"] == "not-executed", "assessment claims execution")
    statuses = assessment["properties"]["readiness_status"]["enum"]
    check(all("approved" not in status and "authorized" not in status and status != "ready" for status in statuses), "assessment readiness enum overstates status")
    check(assessment["properties"]["limitations"]["const"] == ASSESSMENT_LIMITATIONS, "assessment limitations drift")
    check("canonical-production" not in TOOL.read_text(encoding="utf-8").split("EXECUTION_AUTHORITY =", 1)[1].split("\n", 1)[0], "tool authority constant drift")
    shell = COMPOSED_GATE.read_text(encoding="utf-8")
    check("SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_MADAROS_BIN" in shell, "composed gate omits current compiler input")
    check("SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_MADAROS_BIN" in shell, "composed gate does not forward current compiler")
    check("physical_extraction_canonical_cutover_execution_gate.sh" in shell, "composed gate omits prior stack")
    check("physical_extraction_canonical_production_gap_gate.py" in shell, "composed gate omits focused assessment")


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-canonical-production-gap-gate-") as temporary:
        work = Path(temporary)
        fixture_a = create_fixture(work / "a")
        fixture_b = create_fixture(work / "b")
        source_a = Path(fixture_a["repo"])
        before = source_state(source_a)

        absent_a = work / "absent-a.json"
        absent_b = work / "absent-b.json"
        result_absent = run(assess_command(fixture_a, absent_a, proposal=None))
        run(assess_command(fixture_b, absent_b, proposal=None))
        absent_payload = json.loads(absent_a.read_text(encoding="ascii"))
        check("status=mapping-proposal-required" in result_absent.stdout, "missing proposal status marker drift")
        check(absent_a.read_bytes() == absent_b.read_bytes(), "absent-proposal assessment depends on physical root")
        check(absent_payload["readiness_status"] == "mapping-proposal-required", "absent proposal status is wrong")
        check(absent_payload["execution_authority"] == "none", "absent proposal assessment grants authority")
        check(absent_payload["canonical_cutover_execution_status"] == "not-executed", "absent proposal assessment claims execution")
        check(absent_payload["summary"]["planned_target_count"] == 2, "planned target count is wrong")
        check(absent_payload["summary"]["mapped_target_count"] == 0, "absent mapping count is wrong")
        check(absent_payload["summary"]["observed_available_destination_count"] == 0, "absent available count is wrong")
        check(all(target["repository_id"] is None for target in absent_payload["targets"]), "absent proposal synthesized mapping")
        check(all(target["gap_codes"] == ["target-repository-mapping-absent"] for target in absent_payload["targets"]), "absent target gaps drift")
        run(verify_command(fixture_a, absent_a, proposal=None))

        proposed_a = work / "proposed-a.json"
        proposed_b = work / "proposed-b.json"
        result_proposed = run(assess_command(fixture_a, proposed_a, proposal=Path(fixture_a["proposal"])))
        run(assess_command(fixture_b, proposed_b, proposal=Path(fixture_b["proposal"])))
        proposed_payload = json.loads(proposed_a.read_text(encoding="ascii"))
        check("status=production-evidence-and-human-decision-required" in result_proposed.stdout, "proposed status marker drift")
        check(proposed_a.read_bytes() == proposed_b.read_bytes(), "proposed assessment depends on physical root")
        check(proposed_payload["readiness_status"] == "production-evidence-and-human-decision-required", "proposed status is wrong")
        check(proposed_payload["next_required_action"] == "author-production-evidence-set-and-obtain-explicit-human-decision", "proposed next action drift")
        check(proposed_payload["execution_authority"] == "none", "proposed assessment grants authority")
        check(proposed_payload["summary"]["mapped_target_count"] == 2, "mapped target count is wrong")
        check(proposed_payload["summary"]["observed_available_destination_count"] == 2, "available target count is wrong")
        check(all(target["assessment_status"] == "observed-available-proposal-not-approved" for target in proposed_payload["targets"]), "proposed target status overstates or regresses")
        check(proposed_payload["summary"]["missing_prerequisite_count"] == 5, "mandatory evidence gaps changed")
        check(proposed_payload["summary"]["satisfied_prerequisite_count"] == 3, "satisfied prerequisite count changed")
        check(proposed_payload["limitations"] == ASSESSMENT_LIMITATIONS, "emitted assessment limitations drift")
        check(next(row for row in proposed_payload["prerequisites"] if row["prerequisite_id"] == "explicit-human-cutover-decision")["status"] == "missing", "human decision was inferred")
        run(verify_command(fixture_a, proposed_a, proposal=Path(fixture_a["proposal"])))
        check(source_state(source_a) == before, "assessment changed canonical source repository")

        bad_catalog_identity = clone_json(
            Path(fixture_a["catalog"]),
            work / "bad-catalog-identity.json",
            lambda payload: payload.__setitem__("organization", "ChangedFixture"),
            rehash_field=None,
        )
        case = dict(fixture_a)
        case["catalog"] = bad_catalog_identity
        assert_refusal(assess_command(case, work / "bad-catalog-identity-output.json", proposal=None), work / "bad-catalog-identity-output.json", "E-SRB-PROD-GAP-002")

        unsorted_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "unsorted-catalog.json",
            lambda payload: payload["repositories"].reverse(),
            rehash_field="catalog_identity_sha256",
        )
        case = dict(fixture_a)
        case["catalog"] = unsorted_catalog
        assert_refusal(assess_command(case, work / "unsorted-output.json", proposal=None), work / "unsorted-output.json", "E-SRB-PROD-GAP-002")

        duplicate_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "duplicate-catalog.json",
            lambda payload: payload["repositories"].insert(1, dict(payload["repositories"][0])),
            rehash_field="catalog_identity_sha256",
        )
        case = dict(fixture_a)
        case["catalog"] = duplicate_catalog
        assert_refusal(assess_command(case, work / "duplicate-output.json", proposal=None), work / "duplicate-output.json", "E-SRB-PROD-GAP-002")

        outside_organization_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "outside-organization-catalog.json",
            lambda payload: payload["repositories"][0].__setitem__(
                "name_with_owner", "OtherFixture/destination-pkg"
            ),
            rehash_field="catalog_identity_sha256",
        )
        case = dict(fixture_a)
        case["catalog"] = outside_organization_catalog
        assert_refusal(
            assess_command(case, work / "outside-organization-output.json", proposal=None),
            work / "outside-organization-output.json",
            "E-SRB-PROD-GAP-002",
        )

        missing_organization_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "missing-organization-catalog.json",
            lambda payload: payload.pop("organization"),
            rehash_field="catalog_identity_sha256",
        )
        case = dict(fixture_a)
        case["catalog"] = missing_organization_catalog
        assert_refusal(
            assess_command(case, work / "missing-organization-output.json", proposal=None),
            work / "missing-organization-output.json",
            "E-SRB-PROD-GAP-002",
        )

        nonstring_organization_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "nonstring-organization-catalog.json",
            lambda payload: payload.__setitem__("organization", 7),
            rehash_field="catalog_identity_sha256",
        )
        case = dict(fixture_a)
        case["catalog"] = nonstring_organization_catalog
        assert_refusal(
            assess_command(case, work / "nonstring-organization-output.json", proposal=None),
            work / "nonstring-organization-output.json",
            "E-SRB-PROD-GAP-002",
        )

        approved_proposal = clone_json(
            Path(fixture_a["proposal"]),
            work / "approved-proposal.json",
            lambda payload: payload.__setitem__("proposal_status", "approved"),
            rehash_field="proposal_identity_sha256",
        )
        assert_refusal(assess_command(fixture_a, work / "approved-output.json", proposal=approved_proposal), work / "approved-output.json", "E-SRB-PROD-GAP-003")

        authorized_mapping = clone_json(
            Path(fixture_a["proposal"]),
            work / "authorized-mapping.json",
            lambda payload: payload["mappings"][0].__setitem__("mapping_status", "authorized"),
            rehash_field="proposal_identity_sha256",
        )
        assert_refusal(assess_command(fixture_a, work / "authorized-output.json", proposal=authorized_mapping), work / "authorized-output.json", "E-SRB-PROD-GAP-003")

        incomplete_proposal = clone_json(
            Path(fixture_a["proposal"]),
            work / "incomplete-proposal.json",
            lambda payload: payload["mappings"].pop(),
            rehash_field="proposal_identity_sha256",
        )
        assert_refusal(assess_command(fixture_a, work / "incomplete-output.json", proposal=incomplete_proposal), work / "incomplete-output.json", "E-SRB-PROD-GAP-003")

        def reuse_repository(payload: dict[str, object]) -> None:
            mappings = payload["mappings"]
            mappings[1]["repository_id"] = mappings[0]["repository_id"]

        reused_proposal = clone_json(
            Path(fixture_a["proposal"]),
            work / "reused-proposal.json",
            reuse_repository,
            rehash_field="proposal_identity_sha256",
        )
        assert_refusal(assess_command(fixture_a, work / "reused-output.json", proposal=reused_proposal), work / "reused-output.json", "E-SRB-PROD-GAP-003")

        stale_proposal = clone_json(
            Path(fixture_a["proposal"]),
            work / "stale-proposal.json",
            lambda payload: payload.__setitem__("catalog_identity_sha256", "0" * 64),
            rehash_field="proposal_identity_sha256",
        )
        assert_refusal(assess_command(fixture_a, work / "stale-output.json", proposal=stale_proposal), work / "stale-output.json", "E-SRB-PROD-GAP-003")

        missing_destination_catalog, missing_catalog_payload = create_catalog(
            work / "missing-destination-catalog.json",
            [
                row
                for row in fixture_a["catalog_payload"]["repositories"]
                if row["repository_id"] != "destination-research"
            ],
        )
        missing_destination_proposal, _payload = create_proposal(
            work / "missing-destination-proposal.json",
            missing_catalog_payload,
            [
                row
                for row in fixture_a["catalog_payload"]["repositories"]
                if str(row["repository_id"]).startswith("destination-")
            ],
        )
        missing_case = dict(fixture_a)
        missing_case["catalog"] = missing_destination_catalog
        missing_output = work / "missing-destination-output.json"
        run(assess_command(missing_case, missing_output, proposal=missing_destination_proposal))
        missing_payload = json.loads(missing_output.read_text(encoding="ascii"))
        check(missing_payload["readiness_status"] == "destination-repositories-required", "missing destination did not remain a gap")
        check(missing_payload["summary"]["observed_available_destination_count"] == 1, "missing destination availability count is wrong")

        archived_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        for row in archived_catalog_payload["repositories"]:
            if row["repository_id"] == "destination-pkg":
                row["archived"] = True
        archived_catalog_payload["catalog_identity_sha256"] = digest(archived_catalog_payload, "catalog_identity_sha256")
        archived_catalog = write_json(work / "archived-catalog.json", archived_catalog_payload)
        archived_proposal_payload = json.loads(Path(fixture_a["proposal"]).read_text(encoding="ascii"))
        archived_proposal_payload["catalog_identity_sha256"] = archived_catalog_payload["catalog_identity_sha256"]
        archived_proposal_payload["proposal_identity_sha256"] = digest(archived_proposal_payload, "proposal_identity_sha256")
        archived_proposal = write_json(work / "archived-proposal.json", archived_proposal_payload)
        archived_case = dict(fixture_a)
        archived_case["catalog"] = archived_catalog
        archived_output = work / "archived-output.json"
        run(assess_command(archived_case, archived_output, proposal=archived_proposal))
        archived_payload = json.loads(archived_output.read_text(encoding="ascii"))
        check(archived_payload["readiness_status"] == "destination-repositories-required", "archived destination was treated as available")
        check("mapped-repository-archived" in archived_payload["targets"][0]["gap_codes"], "archived destination gap code absent")

        read_only_catalog_payload = json.loads(Path(fixture_a["catalog"]).read_text(encoding="ascii"))
        for row in read_only_catalog_payload["repositories"]:
            if row["repository_id"] == "destination-pkg":
                row["observed_permission"] = "READ"
        read_only_catalog_payload["catalog_identity_sha256"] = digest(
            read_only_catalog_payload, "catalog_identity_sha256"
        )
        read_only_catalog = write_json(work / "read-only-catalog.json", read_only_catalog_payload)
        read_only_proposal_payload = json.loads(Path(fixture_a["proposal"]).read_text(encoding="ascii"))
        read_only_proposal_payload["catalog_identity_sha256"] = read_only_catalog_payload[
            "catalog_identity_sha256"
        ]
        read_only_proposal_payload["proposal_identity_sha256"] = digest(
            read_only_proposal_payload, "proposal_identity_sha256"
        )
        read_only_proposal = write_json(work / "read-only-proposal.json", read_only_proposal_payload)
        read_only_case = dict(fixture_a)
        read_only_case["catalog"] = read_only_catalog
        read_only_output = work / "read-only-output.json"
        run(assess_command(read_only_case, read_only_output, proposal=read_only_proposal))
        read_only_payload = json.loads(read_only_output.read_text(encoding="ascii"))
        check(
            read_only_payload["readiness_status"] == "destination-repositories-required",
            "read-only destination was treated as available",
        )
        check(
            "mapped-repository-insufficient-observed-permission"
            in read_only_payload["targets"][0]["gap_codes"],
            "read-only destination gap code absent",
        )

        dirty_marker = write(source_a / "dirty.txt", "dirty\n")
        dirty_output = work / "dirty-output.json"
        run(assess_command(fixture_a, dirty_output, proposal=Path(fixture_a["proposal"])))
        dirty_payload = json.loads(dirty_output.read_text(encoding="ascii"))
        check(dirty_payload["readiness_status"] == "canonical-source-snapshot-required", "dirty source was treated as canonical-ready")
        check("canonical-worktree-dirty" in dirty_payload["canonical_repository"]["gap_codes"], "dirty gap code absent")
        dirty_marker.unlink()
        check(source_state(source_a) == before, "dirty assessment failed to restore external test state")

        occupied = work / "occupied.json"
        write(occupied, "preserve\n")
        result_occupied = run(assess_command(fixture_a, occupied, proposal=None), expected=1)
        check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied output was overwritten")
        check("E-SRB-PROD-GAP-005" in result_occupied.stderr, "occupied output refusal code drift")

        forged = clone_json(
            proposed_a,
            work / "forged.json",
            lambda payload: payload.__setitem__("execution_authority", "granted"),
            rehash_field=None,
        )
        assert_refusal(verify_command(fixture_a, forged, proposal=Path(fixture_a["proposal"])), None, "E-SRB-PROD-GAP-006")
        rehashed = clone_json(
            proposed_a,
            work / "rehashed.json",
            lambda payload: payload.__setitem__("execution_authority", "granted"),
            rehash_field="assessment_identity_sha256",
        )
        assert_refusal(verify_command(fixture_a, rehashed, proposal=Path(fixture_a["proposal"])), None, "E-SRB-PROD-GAP-006")

        changed_catalog = clone_json(
            Path(fixture_a["catalog"]),
            work / "changed-catalog.json",
            lambda payload: payload.__setitem__("observed_at_utc", "2026-07-18T00:00:00Z"),
            rehash_field="catalog_identity_sha256",
        )
        assert_refusal(
            verify_command(
                fixture_a,
                absent_a,
                proposal=None,
                catalog=changed_catalog,
            ),
            None,
            "E-SRB-PROD-GAP-006",
        )

        source_file = source_a / "packages" / "pkg" / "README.md"
        original = source_file.read_bytes()
        source_file.write_bytes(original + b"changed\n")
        assert_refusal(
            verify_command(fixture_a, proposed_a, proposal=Path(fixture_a["proposal"])),
            None,
            "E-SRB-PROD-GAP-006",
        )
        source_file.write_bytes(original)
        check(source_state(source_a) == before, "source mutation test changed Git state after restoration")

        check(not any(path.name.endswith(".staging") for path in work.rglob("*")), "gate left staging files")

        print(
            "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_WITNESS "
            f"assessment_identity={proposed_payload['assessment_identity_sha256']} "
            f"catalog_identity={fixture_a['catalog_payload']['catalog_identity_sha256']} "
            f"proposal_identity={fixture_a['proposal_payload']['proposal_identity_sha256']} "
            f"targets={proposed_payload['summary']['planned_target_count']} "
            f"status={proposed_payload['readiness_status']} authority={proposed_payload['execution_authority']}"
        )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_GAP_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
