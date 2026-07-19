#!/usr/bin/env python3
"""Adversarial gate for the read-only canonical-production evidence set."""

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
import physical_extraction_canonical_production_gap_gate as gap_gate  # noqa: E402


TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_evidence_set.py"
EVIDENCE_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-evidence-set.v1.schema.json"
VALIDATION_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-production-validation-observations.v1.schema.json"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_canonical_production_evidence_set_gate.sh"
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
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, cwd: Path | None = None, expected: int = 0) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env={
            **os.environ,
            "LANG": "C",
            "LC_ALL": "C",
            "TZ": "UTC",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_AUTHOR_NAME": "Sounio Fixture",
            "GIT_AUTHOR_EMAIL": "fixture@sounio.invalid",
            "GIT_COMMITTER_NAME": "Sounio Fixture",
            "GIT_COMMITTER_EMAIL": "fixture@sounio.invalid",
            "GIT_AUTHOR_DATE": "2026-07-19T00:00:00Z",
            "GIT_COMMITTER_DATE": "2026-07-19T00:00:00Z",
        },
        text=True,
        capture_output=True,
        timeout=240,
    )
    if result.returncode != expected:
        raise AssertionError(
            f"command returned {result.returncode}, expected {expected}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def git(arguments: list[str], cwd: Path) -> str:
    return run(["git", *arguments], cwd=cwd).stdout.strip()


def digest(payload: object, field: str) -> str:
    value = json.loads(json.dumps(payload))
    if isinstance(value, dict):
        value.pop(field, None)
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    return path


def remove_worktree_content(repository: Path) -> None:
    for child in repository.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_source_tree(source: Path, destination: Path) -> None:
    for child in sorted(source.iterdir(), key=lambda path: path.name):
        target = destination / child.name
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def commit_destination(repository: Path, message: str) -> str:
    git(["add", "-A"], repository)
    git(["commit", "-m", message], repository)
    git(["push", "origin", "main"], repository)
    return git(["rev-parse", "HEAD"], repository)


def refresh_contracts(fixture: dict[str, object]) -> None:
    catalog = json.loads(Path(fixture["catalog"]).read_text(encoding="ascii"))
    proposal = json.loads(Path(fixture["proposal"]).read_text(encoding="ascii"))
    heads = {
        repository.name: git(["rev-parse", "HEAD"], repository)
        for repository in fixture["destination_repositories"]
    }
    remotes = {
        repository.name: git(["remote", "get-url", "origin"], repository)
        for repository in fixture["destination_repositories"]
    }
    for row in catalog["repositories"]:
        if row["repository_id"] in heads:
            row["head_oid"] = heads[row["repository_id"]]
            row["remote_url"] = remotes[row["repository_id"]]
    catalog["catalog_identity_sha256"] = digest(catalog, "catalog_identity_sha256")
    write_json(Path(fixture["catalog"]), catalog)
    for row in proposal["mappings"]:
        row["expected_head_oid"] = heads[row["repository_id"]]
        row["remote_url"] = remotes[row["repository_id"]]
    proposal["catalog_identity_sha256"] = catalog["catalog_identity_sha256"]
    proposal["proposal_identity_sha256"] = digest(proposal, "proposal_identity_sha256")
    write_json(Path(fixture["proposal"]), proposal)
    fixture["catalog_payload"] = catalog
    fixture["proposal_payload"] = proposal


def validation_payload(source_head: str, *, result: str = "passed") -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-production-validation-observations.v1",
        "observation_type": "supplied-validation-observations",
        "authority_scope": "evidence-citation-only",
        "canonical_head_oid": source_head,
        "observations": [
            {
                "observation_id": "fixture-package-science-gate",
                "scope": "canonical-source-snapshot",
                "command": "bash scripts/ci/package_import_science_gate.sh",
                "result": result,
                "exit_code": 0 if result == "passed" else 1,
                "stdout_sha256": hashlib.sha256(b"fixture stdout\n").hexdigest(),
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "evidence_ref": "fixture:package-science-gate",
            }
        ],
        "limitations": VALIDATION_LIMITATIONS,
    }
    payload["validation_identity_sha256"] = digest(payload, "validation_identity_sha256")
    return payload


def create_exact_fixture(work: Path) -> dict[str, object]:
    fixture = gap_gate.create_fixture(work)
    sources = {
        "destination-pkg": Path(fixture["repo"]) / "packages" / "pkg",
        "destination-research": Path(fixture["repo"]) / "research",
    }
    destinations_root = work / "destinations"
    destinations_root.mkdir()
    relocated: list[Path] = []
    for repository in fixture["destination_repositories"]:
        destination = destinations_root / repository.name
        shutil.move(str(repository), destination)
        git(["remote", "set-url", "origin", f"../../remotes/{repository.name}.git"], destination)
        relocated.append(destination)
    fixture["destination_repositories"] = relocated
    for repository in relocated:
        remove_worktree_content(repository)
        copy_source_tree(sources[repository.name], repository)
        commit_destination(repository, "fixture: exact materialization")
    refresh_contracts(fixture)
    validation = write_json(
        work / "validation-observations.json",
        validation_payload(git(["rev-parse", "HEAD"], Path(fixture["repo"]))),
    )
    fixture["validation"] = validation
    fixture["destinations_root"] = destinations_root
    return fixture


def build_command(fixture: dict[str, object], output: Path, *, validation: Path | None = None) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "build",
        "--repo-root",
        str(fixture["repo"]),
        "--rings",
        str(fixture["rings"]),
        "--ownership",
        str(fixture["ownership"]),
        "--repository-catalog",
        str(fixture["catalog"]),
        "--mapping-proposal",
        str(fixture["proposal"]),
        "--validation-observations",
        str(validation or fixture["validation"]),
        "--destinations-root",
        str(fixture["destinations_root"]),
        "--canonical-repository-id",
        "sounio-fixture",
        "--output",
        str(output),
    ]


def verify_command(fixture: dict[str, object], evidence: Path, *, validation: Path | None = None) -> list[str]:
    command = build_command(fixture, Path("unused"), validation=validation)
    command[2] = "verify"
    output_index = command.index("--output")
    command[output_index:] = ["--evidence", str(evidence)]
    return command


def repository_state(repository: Path) -> tuple[str, str, str]:
    return (
        git(["rev-parse", "HEAD"], repository),
        git(["status", "--porcelain=v1", "--untracked-files=all"], repository),
        git(["remote", "get-url", "origin"], repository),
    )


def assert_refusal(command: list[str], code: str, output: Path | None = None) -> None:
    result = run(command, expected=1)
    check(code in result.stderr, f"refusal lacks {code}")
    check("PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_REFUSED" in result.stderr, "refusal marker absent")
    if output is not None:
        check(not output.exists(), f"refusal left output {output}")


def assert_static_contracts() -> None:
    evidence = json.loads(EVIDENCE_SCHEMA.read_text(encoding="utf-8"))
    validation = json.loads(VALIDATION_SCHEMA.read_text(encoding="utf-8"))
    check(evidence["properties"]["execution_authority"]["const"] == "none", "schema grants authority")
    check(evidence["properties"]["source_removal_authority"]["const"] == "none", "schema grants removal")
    check(evidence["properties"]["canonical_production_approval"]["const"] == "not-approved", "schema approves production")
    check(evidence["properties"]["canonical_cutover_execution_status"]["const"] == "not-executed", "schema claims cutover")
    check(evidence["properties"]["limitations"]["const"] == EVIDENCE_LIMITATIONS, "evidence limitations drift")
    check(validation["properties"]["limitations"]["const"] == VALIDATION_LIMITATIONS, "validation limitations drift")
    check(validation["properties"]["authority_scope"]["const"] == "evidence-citation-only", "validation scope drift")
    shell = COMPOSED_GATE.read_text(encoding="utf-8")
    check("physical_extraction_canonical_production_gap_gate.sh" in shell, "composed gate omits prior stack")
    check("physical_extraction_canonical_production_evidence_set_gate.py" in shell, "composed gate omits evidence gate")


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-production-evidence-gate-") as temporary:
        work = Path(temporary)
        fixture_a = create_exact_fixture(work / "a")
        fixture_b = create_exact_fixture(work / "b")
        source_before = repository_state(Path(fixture_a["repo"]))
        destinations_before = [repository_state(path) for path in fixture_a["destination_repositories"]]

        exact_a = work / "exact-a.json"
        exact_b = work / "exact-b.json"
        result = run(build_command(fixture_a, exact_a))
        run(build_command(fixture_b, exact_b))
        payload = json.loads(exact_a.read_text(encoding="ascii"))
        check("status=production-evidence-draft-exact-parity" in result.stdout, "exact status marker drift")
        check(exact_a.read_bytes() == exact_b.read_bytes(), "evidence depends on physical root")
        check(payload["evidence_status"] == "production-evidence-draft-exact-parity", "exact status wrong")
        check(payload["proposal_status"] == "proposed-not-approved", "proposal status overstated")
        check(payload["execution_authority"] == "none", "evidence grants execution")
        check(payload["source_removal_authority"] == "none", "evidence grants removal")
        check(payload["canonical_production_approval"] == "not-approved", "evidence grants approval")
        check(payload["canonical_cutover_execution_status"] == "not-executed", "evidence claims cutover")
        check(payload["summary"]["target_count"] == 2, "target count wrong")
        check(payload["summary"]["exact_parity_target_count"] == 2, "exact count wrong")
        check(payload["summary"]["parity_gap_target_count"] == 0, "gap count wrong")
        check(all(row["parity"]["status"] == "exact-copy-verified" for row in payload["targets"]), "exact rows drift")
        check(payload["proposed_execution_plan"]["plan_status"] == "draft-not-authorized", "plan became authorized")
        check(payload["summary"]["permission_bearing_prerequisite_missing_count"] == 4, "permission gaps drift")
        check(sum(row["status"] == "missing" for row in payload["governance_prerequisites"]) == 4, "human gaps inferred")
        run(verify_command(fixture_a, exact_a))
        check(repository_state(Path(fixture_a["repo"])) == source_before, "build changed source repository")
        check([repository_state(path) for path in fixture_a["destination_repositories"]] == destinations_before, "build changed destination repositories")

        gap_fixture = create_exact_fixture(work / "gap")
        gap_repository = Path(gap_fixture["destination_repositories"][1])
        (gap_repository / "extra.sio").write_text("fn extra() -> i64 { 4 }\n", encoding="ascii")
        (gap_repository / "study.sio").write_text("fn study() -> i64 { 99 }\n", encoding="ascii")
        commit_destination(gap_repository, "fixture: observed parity gap")
        refresh_contracts(gap_fixture)
        gap_output = work / "gap.json"
        run(build_command(gap_fixture, gap_output))
        gap_payload = json.loads(gap_output.read_text(encoding="ascii"))
        check(gap_payload["evidence_status"] == "production-evidence-draft-gaps-observed", "gap status wrong")
        check(gap_payload["summary"]["exact_parity_target_count"] == 1, "gap exact count wrong")
        check(gap_payload["summary"]["parity_gap_target_count"] == 1, "gap target count wrong")
        research = next(row for row in gap_payload["targets"] if row["target_id"] == "distribution:research")
        check(research["parity"]["changed_paths_sample"] == ["study.sio"], "changed sample wrong")
        check(research["parity"]["extra_paths_sample"] == ["extra.sio"], "extra sample wrong")
        check(research["parity"]["missing_paths_sample_complete"], "empty missing sample should be complete")
        check(research["parity"]["extra_paths_sample_complete"], "extra sample should be complete")
        check(research["parity"]["changed_paths_sample_complete"], "changed sample should be complete")
        check(gap_payload["next_required_action"] == "resolve-parity-or-validation-gaps-and-reissue-evidence", "gap action wrong")
        run(verify_command(gap_fixture, gap_output))

        failed_validation = write_json(
            work / "failed-validation.json",
            validation_payload(git(["rev-parse", "HEAD"], Path(fixture_a["repo"])), result="failed"),
        )
        failed_output = work / "failed.json"
        run(build_command(fixture_a, failed_output, validation=failed_validation))
        failed_payload = json.loads(failed_output.read_text(encoding="ascii"))
        check(failed_payload["evidence_status"] == "production-evidence-draft-gaps-observed", "failed validation ignored")
        check(failed_payload["summary"]["validation_failed_count"] == 1, "failed validation count wrong")

        occupied = work / "occupied.json"
        occupied.write_text("preserve\n", encoding="ascii")
        result_occupied = run(build_command(fixture_a, occupied), expected=1)
        check(occupied.read_text(encoding="ascii") == "preserve\n", "occupied output overwritten")
        check("E-SRB-PROD-EVIDENCE-005" in result_occupied.stderr, "occupied output code drift")

        bad_validation_payload = validation_payload("0" * 40)
        bad_validation = write_json(work / "bad-validation.json", bad_validation_payload)
        bad_output = work / "bad-validation-output.json"
        assert_refusal(build_command(fixture_a, bad_output, validation=bad_validation), "E-SRB-PROD-EVIDENCE-004", bad_output)

        tampered_payload = json.loads(Path(fixture_a["validation"]).read_text(encoding="ascii"))
        tampered_payload["observations"][0]["result"] = "failed"
        tampered = write_json(work / "tampered-validation.json", tampered_payload)
        tampered_output = work / "tampered-output.json"
        assert_refusal(build_command(fixture_a, tampered_output, validation=tampered), "E-SRB-PROD-EVIDENCE-004", tampered_output)

        dirty_repository = Path(fixture_a["destination_repositories"][0])
        dirty = dirty_repository / "dirty.txt"
        dirty.write_text("dirty\n", encoding="ascii")
        dirty_output = work / "dirty-output.json"
        assert_refusal(build_command(fixture_a, dirty_output), "E-SRB-PROD-EVIDENCE-003", dirty_output)
        dirty.unlink()

        forged = json.loads(exact_a.read_text(encoding="ascii"))
        forged["execution_authority"] = "granted"
        forged_path = write_json(work / "forged.json", forged)
        assert_refusal(verify_command(fixture_a, forged_path), "E-SRB-PROD-EVIDENCE-006")
        forged["evidence_identity_sha256"] = digest(forged, "evidence_identity_sha256")
        rehashed_path = write_json(work / "rehashed.json", forged)
        assert_refusal(verify_command(fixture_a, rehashed_path), "E-SRB-PROD-EVIDENCE-006")

        changed = Path(fixture_a["destination_repositories"][0]) / "README.md"
        original = changed.read_bytes()
        changed.write_bytes(original + b"changed\n")
        assert_refusal(verify_command(fixture_a, exact_a), "E-SRB-PROD-EVIDENCE-003")
        changed.write_bytes(original)
        check(repository_state(Path(fixture_a["repo"])) == source_before, "gate changed source repository")
        check([repository_state(path) for path in fixture_a["destination_repositories"]] == destinations_before, "gate failed to restore destinations")
        check(not any(path.name.endswith(".staging") for path in work.rglob("*")), "gate left staging files")

        print(
            "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_WITNESS "
            f"evidence_identity={payload['evidence_identity_sha256']} "
            f"targets={payload['summary']['target_count']} exact={payload['summary']['exact_parity_target_count']} "
            f"status={payload['evidence_status']} authority={payload['execution_authority']}"
        )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_EVIDENCE_SET_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
