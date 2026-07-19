#!/usr/bin/env python3
"""Adversarial gate for the path-level reconciliation proposal contract."""

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
import physical_extraction_canonical_production_evidence_set_gate as evidence_gate  # noqa: E402


TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_reconciliation_proposal.py"
SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-production-reconciliation-proposal.v1.schema.json"
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


def commit_source(repository: Path, message: str) -> str:
    git(["add", "-A"], repository)
    git(["commit", "-m", message], repository)
    git(["push", "origin", "main"], repository)
    return git(["rev-parse", "HEAD"], repository)


def refresh_source_contracts(fixture: dict[str, object]) -> None:
    source = Path(fixture["repo"])
    head = git(["rev-parse", "HEAD"], source)
    remote = git(["remote", "get-url", "origin"], source)
    catalog_path = Path(fixture["catalog"])
    proposal_path = Path(fixture["proposal"])
    catalog = json.loads(catalog_path.read_text(encoding="ascii"))
    for row in catalog["repositories"]:
        if row["repository_id"] == "sounio-fixture":
            row["head_oid"] = head
            row["remote_url"] = remote
    catalog["catalog_identity_sha256"] = digest(catalog, "catalog_identity_sha256")
    write_json(catalog_path, catalog)
    proposal = json.loads(proposal_path.read_text(encoding="ascii"))
    proposal["catalog_identity_sha256"] = catalog["catalog_identity_sha256"]
    proposal["proposal_identity_sha256"] = digest(proposal, "proposal_identity_sha256")
    write_json(proposal_path, proposal)
    write_json(Path(fixture["validation"]), evidence_gate.validation_payload(head))
    fixture["catalog_payload"] = catalog
    fixture["proposal_payload"] = proposal


def research_destination(fixture: dict[str, object]) -> Path:
    return next(
        repository
        for repository in fixture["destination_repositories"]
        if Path(repository).name == "destination-research"
    )


def create_gap_fixture(work: Path) -> dict[str, object]:
    fixture = evidence_gate.create_exact_fixture(work)
    source = Path(fixture["repo"])
    destination = research_destination(fixture)
    (source / "research" / "missing.sio").write_text("fn missing() -> i64 { 5 }\n", encoding="ascii")
    (source / "research" / "retained.sio").write_text("fn retained() -> i64 { 6 }\n", encoding="ascii")
    commit_source(source, "fixture: extend research source")
    shutil.copy2(source / "research" / "retained.sio", destination / "retained.sio")
    (destination / "study.sio").write_text("fn study() -> i64 { 99 }\n", encoding="ascii")
    (destination / "extra.sio").write_text("fn extra() -> i64 { 7 }\n", encoding="ascii")
    evidence_gate.commit_destination(destination, "fixture: observed reconciliation gap")
    evidence_gate.refresh_contracts(fixture)
    refresh_source_contracts(fixture)
    evidence = work / "production-evidence.json"
    run(evidence_gate.build_command(fixture, evidence))
    fixture["evidence"] = evidence
    return fixture


def build_command(fixture: dict[str, object], output: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "build",
        "--evidence",
        str(fixture["evidence"]),
        "--source-root",
        str(fixture["repo"]),
        "--destination-root",
        str(research_destination(fixture)),
        "--target-id",
        "distribution:research",
        "--expected-evidence-identity",
        str(json.loads(Path(fixture["evidence"]).read_text(encoding="ascii"))["evidence_identity_sha256"]),
        "--output",
        str(output),
    ]


def verify_command(fixture: dict[str, object], proposal: Path) -> list[str]:
    command = build_command(fixture, Path("unused"))
    command[2] = "verify"
    output_index = command.index("--output")
    command[output_index:] = ["--proposal", str(proposal)]
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
    check(
        "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_REFUSED" in result.stderr,
        "refusal marker absent",
    )
    if output is not None:
        check(not output.exists(), f"refusal left output {output}")


def assert_static_contracts() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    properties = schema["properties"]
    check(properties["proposal_status"]["const"] == "proposed-not-approved", "schema approves proposal")
    check(properties["execution_authority"]["const"] == "none", "schema grants execution")
    check(properties["destination_write_authority"]["const"] == "none", "schema grants writes")
    check(properties["source_removal_authority"]["const"] == "none", "schema grants source removal")
    check(properties["canonical_production_approval"]["const"] == "not-approved", "schema approves production")
    check(properties["canonical_cutover_execution_status"]["const"] == "not-executed", "schema claims cutover")
    tool = TOOL.read_text(encoding="utf-8")
    check('"git", "push"' not in tool and '"git", "commit"' not in tool, "tool contains Git mutation command")
    check("shutil" not in tool, "tool contains copy helper")
    check("os.link(temporary, path" in tool, "output publication is not atomic no-clobber")


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-production-reconciliation-gate-") as temporary:
        work = Path(temporary)
        fixture_a = create_gap_fixture(work / "a")
        fixture_b = create_gap_fixture(work / "b")
        source_before = repository_state(Path(fixture_a["repo"]))
        destination_before = repository_state(research_destination(fixture_a))
        proposal_a = work / "proposal-a.json"
        proposal_b = work / "proposal-b.json"
        result = run(build_command(fixture_a, proposal_a))
        run(build_command(fixture_b, proposal_b))
        payload = json.loads(proposal_a.read_text(encoding="ascii"))
        check(proposal_a.read_bytes() == proposal_b.read_bytes(), "proposal depends on physical root")
        check("status=proposed-not-approved" in result.stdout, "proposal status marker drift")
        check("authority=none" in result.stdout, "authority marker drift")
        check(payload["proposal_status"] == "proposed-not-approved", "proposal status overstated")
        check(payload["execution_authority"] == "none", "proposal grants execution")
        check(payload["destination_write_authority"] == "none", "proposal grants destination write")
        check(payload["source_removal_authority"] == "none", "proposal grants source removal")
        check(payload["canonical_production_approval"] == "not-approved", "proposal grants production approval")
        check(payload["canonical_cutover_execution_status"] == "not-executed", "proposal claims cutover")
        check(payload["summary"]["path_count"] == 4, "path count wrong")
        check(payload["summary"]["mutation_path_count"] == 3, "mutation count wrong")
        check(payload["summary"]["add_path_count"] == 1, "add count wrong")
        check(payload["summary"]["replace_path_count"] == 1, "replace count wrong")
        check(payload["summary"]["remove_path_count"] == 1, "remove count wrong")
        check(payload["summary"]["retain_path_count"] == 1, "retain count wrong")
        check([row["path"] for row in payload["path_plan"]] == sorted(row["path"] for row in payload["path_plan"]), "path plan unsorted")
        check(all(row["operation_authority"] == "none" for row in payload["path_plan"]), "path row grants authority")
        check(
            payload["path_plan_sha256"]
            == hashlib.sha256(
                json.dumps(payload["path_plan"], sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
            ).hexdigest(),
            "path-plan hash mismatch",
        )
        by_path = {row["path"]: row for row in payload["path_plan"]}
        check(by_path["missing.sio"]["disposition"] == "add-source-byte-copy", "add disposition wrong")
        check(by_path["study.sio"]["disposition"] == "replace-with-source-byte-copy", "replace disposition wrong")
        check(by_path["extra.sio"]["disposition"] == "remove-destination-only", "remove disposition wrong")
        check(by_path["retained.sio"]["disposition"] == "retain-identical", "retain disposition wrong")
        check(by_path["extra.sio"]["destination_after_if_separately_approved"] is None, "remove after-state wrong")
        run(verify_command(fixture_a, proposal_a))
        check(repository_state(Path(fixture_a["repo"])) == source_before, "build changed source repository")
        check(repository_state(research_destination(fixture_a)) == destination_before, "build changed destination repository")

        occupied = work / "occupied.json"
        occupied.write_text("preserve\n", encoding="ascii")
        result_occupied = run(build_command(fixture_a, occupied), expected=1)
        check(occupied.read_text(encoding="ascii") == "preserve\n", "occupied output overwritten")
        check("E-SRB-PROD-RECONCILE-006" in result_occupied.stderr, "occupied output code drift")

        forged_evidence = json.loads(Path(fixture_a["evidence"]).read_text(encoding="ascii"))
        forged_evidence["execution_authority"] = "granted"
        forged_evidence["evidence_identity_sha256"] = digest(forged_evidence, "evidence_identity_sha256")
        forged_evidence_path = write_json(work / "forged-evidence.json", forged_evidence)
        forged_output = work / "forged-output.json"
        command = build_command(fixture_a, forged_output)
        command[command.index("--evidence") + 1] = str(forged_evidence_path)
        assert_refusal(command, "E-SRB-PROD-RECONCILE-002", forged_output)

        wrong_pin_output = work / "wrong-pin-output.json"
        wrong_pin = build_command(fixture_a, wrong_pin_output)
        wrong_pin[wrong_pin.index("--expected-evidence-identity") + 1] = "0" * 64
        assert_refusal(wrong_pin, "E-SRB-PROD-RECONCILE-002", wrong_pin_output)

        evidence_link = work / "evidence-link.json"
        evidence_link.symlink_to(Path(fixture_a["evidence"]))
        link_output = work / "link-output.json"
        link_command = build_command(fixture_a, link_output)
        link_command[link_command.index("--evidence") + 1] = str(evidence_link)
        assert_refusal(link_command, "E-SRB-PROD-RECONCILE-002", link_output)

        forged_proposal = json.loads(proposal_a.read_text(encoding="ascii"))
        forged_proposal["destination_write_authority"] = "granted"
        forged_proposal["proposal_identity_sha256"] = digest(forged_proposal, "proposal_identity_sha256")
        forged_proposal_path = write_json(work / "forged-proposal.json", forged_proposal)
        assert_refusal(verify_command(fixture_a, forged_proposal_path), "E-SRB-PROD-RECONCILE-007")

        proposal_link = work / "proposal-link.json"
        proposal_link.symlink_to(proposal_a)
        assert_refusal(verify_command(fixture_a, proposal_link), "E-SRB-PROD-RECONCILE-007")

        dirty_destination = research_destination(fixture_a) / "dirty.txt"
        dirty_destination.write_text("dirty\n", encoding="ascii")
        dirty_output = work / "dirty-output.json"
        assert_refusal(build_command(fixture_a, dirty_output), "E-SRB-PROD-RECONCILE-004", dirty_output)
        dirty_destination.unlink()

        wrong_target_output = work / "wrong-target.json"
        wrong_target = build_command(fixture_a, wrong_target_output)
        wrong_target[wrong_target.index("--target-id") + 1] = "distribution:absent"
        assert_refusal(wrong_target, "E-SRB-PROD-RECONCILE-002", wrong_target_output)

        same_root_output = work / "same-root.json"
        same_root = build_command(fixture_a, same_root_output)
        same_root[same_root.index("--destination-root") + 1] = str(fixture_a["repo"])
        assert_refusal(same_root, "E-SRB-PROD-RECONCILE-001", same_root_output)

        check(repository_state(Path(fixture_a["repo"])) == source_before, "gate left source mutation")
        check(repository_state(research_destination(fixture_a)) == destination_before, "gate left destination mutation")
        check(not any(path.name.endswith(".staging") for path in work.rglob("*")), "gate left staging file")
        print(
            "PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_WITNESS "
            f"proposal_identity={payload['proposal_identity_sha256']} target={payload['evidence_binding']['target_id']} "
            f"mutations={payload['summary']['mutation_path_count']} status={payload['proposal_status']} "
            f"authority={payload['execution_authority']}"
        )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_PRODUCTION_RECONCILIATION_PROPOSAL_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
