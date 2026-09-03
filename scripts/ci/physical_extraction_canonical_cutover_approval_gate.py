#!/usr/bin/env python3
"""Adversarial acceptance gate for R3 canonical cutover approval."""

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
import physical_extraction_source_removal_execution_gate as execution_gate  # noqa: E402


AUTHORIZER = ROOT / "tools" / "science_boundary" / "canonical_cutover_authorizer.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_canonical_cutover_approval_gate.sh"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-cutover-policy.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-cutover-approval.v1.schema.json"
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
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=240)
    if result.returncode not in expected_codes:
        raise AssertionError(
            f"command returned {result.returncode}, expected {sorted(expected_codes)}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))
    return path


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def digest(value: object, field: str | None = None) -> str:
    payload = json.loads(json.dumps(value))
    if field is not None and isinstance(payload, dict):
        payload.pop(field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def binding(path: Path, repo: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(repo).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def snapshot(repo: Path) -> dict[str, str]:
    return execution_gate.snapshot(repo)


def git_environment() -> dict[str, str]:
    return {
        **os.environ,
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "GIT_AUTHOR_NAME": "Sounio Fixture",
        "GIT_AUTHOR_EMAIL": "fixture@sounio.invalid",
        "GIT_COMMITTER_NAME": "Sounio Fixture",
        "GIT_COMMITTER_EMAIL": "fixture@sounio.invalid",
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00+0000",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00+0000",
        "GIT_TERMINAL_PROMPT": "0",
    }


def initialize_git_repository(repository: Path, bare: Path, relative_remote: str) -> dict[str, str]:
    bare.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "init", "--bare", str(bare)], env=git_environment())
    run(["git", "init", "-b", "main"], cwd=repository, env=git_environment())
    run(["git", "add", "-A"], cwd=repository, env=git_environment())
    run(["git", "commit", "-m", "fixture snapshot"], cwd=repository, env=git_environment())
    run(["git", "remote", "add", "origin", relative_remote], cwd=repository, env=git_environment())
    run(["git", "push", "-u", "origin", "main"], cwd=repository, env=git_environment())
    head = run(["git", "rev-parse", "HEAD"], cwd=repository, env=git_environment()).stdout.strip()
    return {
        "remote_name": "origin",
        "remote_url": relative_remote,
        "branch": "main",
        "head_oid": head,
        "remote_head_oid": head,
    }


def add_cutover_evidence(repo: Path) -> None:
    write(repo / "cutover-approval" / "operator.txt", "fixture operator approved exact fixture cutover\n")
    write(repo / "cutover-approval" / "owners" / "pkg.txt", "fixture target owner accepted package repository\n")
    write(repo / "cutover-approval" / "owners" / "research.txt", "fixture target owner accepted research repository\n")
    write_json(
        repo / "cutover-approval" / "root-marker.json",
        {
            "schema": "sounio.physical-extraction-canonical-root.v1",
            "marker_type": "explicit-canonical-cutover-approval-root",
            "repository_id": "sounio-fixture",
            "approval_context": "disposable-fixture",
        },
    )
    write_json(
        repo / "cutover-approval" / "recovery-plan.json",
        {
            "schema": "sounio.physical-extraction-canonical-cutover-recovery-plan.v1",
            "plan_type": "full-regular-file-backup-and-explicit-rollback",
            "canonical_repository_id": "sounio-fixture",
            "required_backup_model": "full-regular-file-pre-execution-copy",
            "transaction_workspace": "same-filesystem-external",
            "receipt_commit_point": "atomic-hardlink-after-verification",
            "no_receipt_recovery": "restore-and-verify-pre-cutover-tree",
            "receipt_present_recovery": "committed-state-manual-review",
            "crash_atomicity": "not-guaranteed-across-multiple-filesystem-operations",
            "approved_by": "fixture-operator",
        },
    )


def prepare_authorization(work: Path, repo: Path, *, rehearsal_mutates: bool = False) -> dict[str, Path]:
    rings, ownership = execution_gate.create_source_fixture(repo, execution_mutates=rehearsal_mutates)
    add_cutover_evidence(repo)
    remotes = work / "remotes"
    source_git = initialize_git_repository(repo, remotes / "source.git", "../remotes/source.git")

    inventory = work / "inventory.json"
    run(execution_gate.auth_gate.material_gate.inventory_command(repo, rings, ownership, inventory))
    destination_policy = work / "destination-policy.json"
    destination_policy_payload = execution_gate.auth_gate.material_gate.create_policy(
        repo, inventory, destination_policy
    )
    destinations = work / "materialized-destinations"
    execution_gate.auth_gate.material_gate.create_destinations(destinations, inventory, destination_policy_payload)
    materialization = work / "materialization.json"
    run(
        execution_gate.auth_gate.material_gate.materialize_command(
            repo, rings, ownership, inventory, destination_policy, destinations, materialization
        )
    )
    removal_policy = work / "removal-policy.json"
    execution_gate.auth_gate.create_removal_policy(repo, inventory, materialization, removal_policy)
    authorization_workspace = work / "authorization-workspace"
    authorization_workspace.mkdir()
    authorization = work / "authorization.json"
    run(
        execution_gate.auth_gate.authorization_command(
            "authorize",
            repo,
            rings,
            ownership,
            inventory,
            destination_policy,
            destinations,
            materialization,
            removal_policy,
            authorization_workspace,
            authorization,
        )
    )
    return {
        "repo": repo,
        "rings": rings,
        "ownership": ownership,
        "inventory": inventory,
        "destination_policy": destination_policy,
        "destinations": destinations,
        "materialization": materialization,
        "removal_policy": removal_policy,
        "authorization": authorization,
        "remotes": remotes,
        "source_git": source_git,  # type: ignore[dict-item]
    }


def copy_unit(content: Path, checkout: Path, files: list[dict[str, object]]) -> None:
    checkout.mkdir(parents=True)
    for item in files:
        source = content / str(item["path"])
        destination = checkout / str(item["path"])
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def prepare_destination_repositories(work: Path, artifacts: dict[str, object]) -> tuple[Path, list[dict[str, object]]]:
    materialization = json.loads(Path(artifacts["materialization"]).read_text(encoding="ascii"))
    repositories = work / "repositories"
    repositories.mkdir()
    rows = []
    for unit in sorted(materialization["units"], key=lambda item: item["source_path"]):
        suffix = unit["target_id"].split(":", 1)[1]
        checkout_name = f"{suffix}-repository"
        checkout = repositories / checkout_name
        content = Path(artifacts["destinations"]) / unit["destination_key"] / unit["content_path"]
        copy_unit(content, checkout, unit["files"])
        git_state = initialize_git_repository(
            checkout,
            Path(artifacts["remotes"]) / f"{suffix}.git",
            f"../../remotes/{suffix}.git",
        )
        rows.append(
            {
                "source_path": unit["source_path"],
                "ring": unit["ring"],
                "target_id": unit["target_id"],
                "target_owner": unit["target_owner"],
                "checkout_path": checkout_name,
                "repository_id": f"sounio-{suffix}-fixture",
                **git_state,
                "file_count": unit["file_count"],
                "total_bytes": unit["total_bytes"],
                "tree_sha256": unit["destination_tree_sha256"],
                "owner_approval_evidence": binding(
                    Path(artifacts["repo"]) / "cutover-approval" / "owners" / f"{suffix}.txt",
                    Path(artifacts["repo"]),
                ),
            }
        )
    return repositories, rows


def create_policy(
    artifacts: dict[str, object],
    destination_rows: list[dict[str, object]],
    output: Path,
) -> dict[str, object]:
    repo = Path(artifacts["repo"])
    authorization_raw = Path(artifacts["authorization"]).read_bytes()
    authorization = json.loads(authorization_raw)
    materialization_raw = Path(artifacts["materialization"]).read_bytes()
    materialization = json.loads(materialization_raw)
    recovery_path = repo / "cutover-approval" / "recovery-plan.json"
    recovery_payload = json.loads(recovery_path.read_text(encoding="ascii"))
    source_git = artifacts["source_git"]
    source_bindings = {
        "authorization_file_sha256": hashlib.sha256(authorization_raw).hexdigest(),
        "authorization_identity_sha256": authorization["authorization_identity_sha256"],
        "materialization_file_sha256": hashlib.sha256(materialization_raw).hexdigest(),
        "materialization_identity_sha256": materialization["materialization_identity_sha256"],
        "inventory_identity_sha256": authorization["source_bindings"]["inventory_identity_sha256"],
        "pre_cutover_tree_sha256": authorization["candidate_evidence"]["original_source_tree_sha256"],
        "authorized_post_cutover_tree_sha256": authorization["candidate_evidence"]["candidate_tree_sha256"],
        "removal_scope_identity_sha256": authorization["removal_scope"]["scope_identity_sha256"],
        "repair_set_identity_sha256": digest(authorization["repairs"]),
        "gate_set_identity_sha256": digest(authorization["post_removal_gates"]),
    }
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-canonical-cutover-policy.v1",
        "policy_type": "explicit-canonical-cutover-approval-policy",
        "authority_scope": "exact-git-repository-tree-cutover-approval",
        "approval_context": "disposable-fixture",
        "source_bindings": source_bindings,
        "approval_status": "approved",
        "canonical_repository": {
            "repository_id": "sounio-fixture",
            **source_git,
            "retained_marker": binding(repo / "cutover-approval" / "root-marker.json", repo),
        },
        "destinations": destination_rows,
        "recovery_plan": {
            "plan_evidence": binding(recovery_path, repo),
            "recovery_plan_identity_sha256": digest(recovery_payload),
            "pre_cutover_tree_confirmation": source_bindings["pre_cutover_tree_sha256"],
            "authorized_post_cutover_tree_confirmation": source_bindings["authorized_post_cutover_tree_sha256"],
        },
        "operator_approval": {
            "approved_by": "fixture-operator",
            "approval_evidence": [binding(repo / "cutover-approval" / "operator.txt", repo)],
            "authorization_identity_confirmation": source_bindings["authorization_identity_sha256"],
            "scope_identity_confirmation": source_bindings["removal_scope_identity_sha256"],
            "pre_cutover_tree_confirmation": source_bindings["pre_cutover_tree_sha256"],
            "authorized_post_cutover_tree_confirmation": source_bindings["authorized_post_cutover_tree_sha256"],
            "destination_set_identity_confirmation": digest(destination_rows),
            "recovery_plan_identity_confirmation": digest(recovery_payload),
            "repairs_reviewed": True,
            "gates_reviewed": True,
        },
        "limitations": POLICY_LIMITATIONS,
    }
    payload["policy_identity_sha256"] = digest(payload, "policy_identity_sha256")
    write_json(output, payload)
    return payload


def approval_command(
    mode: str,
    artifacts: dict[str, object],
    repositories: Path,
    policy: Path,
    policy_payload: dict[str, object],
    workspace: Path,
    receipt: Path,
) -> list[str]:
    authorization = json.loads(Path(artifacts["authorization"]).read_text(encoding="ascii"))
    return [
        sys.executable,
        str(AUTHORIZER),
        mode,
        "--repo-root",
        str(artifacts["repo"]),
        "--rings",
        "science-rings.tsv",
        "--ownership",
        "ownership.tsv",
        "--inventory",
        str(artifacts["inventory"]),
        "--destination-policy",
        str(artifacts["destination_policy"]),
        "--destinations-root",
        str(artifacts["destinations"]),
        "--materialization-receipt",
        str(artifacts["materialization"]),
        "--removal-policy",
        str(artifacts["removal_policy"]),
        "--authorization-receipt",
        str(artifacts["authorization"]),
        "--repositories-root",
        str(repositories),
        "--cutover-policy",
        str(policy),
        "--workspace-root",
        str(workspace),
        "--cutover-approval-receipt",
        str(receipt),
        "--confirm-authorization-identity",
        authorization["authorization_identity_sha256"],
        "--confirm-scope-identity",
        authorization["removal_scope"]["scope_identity_sha256"],
        "--confirm-policy-identity",
        policy_payload["policy_identity_sha256"],
        "--confirm-pre-cutover-tree",
        authorization["candidate_evidence"]["original_source_tree_sha256"],
    ]


def clone_policy(original: Path, destination: Path, mutate, *, rehash: bool = True) -> tuple[Path, dict[str, object]]:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["policy_identity_sha256"] = digest(payload, "policy_identity_sha256")
    write_json(destination, payload)
    return destination, payload


def clone_receipt(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["approval_identity_sha256"] = digest(payload, "approval_identity_sha256")
    return write_json(destination, payload)


def assert_refusal(command: list[str], receipt: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if receipt is not None:
        check(not receipt.exists(), f"refused approval left receipt {receipt}")
    check(code in result.stderr, f"refusal lacks {code}: {result.stderr}")
    check(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_REFUSED" in result.stderr,
        "refusal lacks cutover approval marker",
    )
    return result


def assert_static_contracts() -> None:
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(policy_schema["properties"]["schema"]["const"] == "sounio.physical-extraction-canonical-cutover-policy.v1", "bad policy schema")
    check(policy_schema["properties"]["approval_status"]["const"] == "approved", "policy permits pending approval")
    check(policy_schema["properties"]["limitations"]["const"] == POLICY_LIMITATIONS, "policy limitations drifted")
    check(receipt_schema["properties"]["canonical_cutover_approval_status"]["const"] == "approved-not-executed", "receipt overstates approval")
    check(receipt_schema["properties"]["canonical_cutover_execution_status"]["const"] == "not-executed", "receipt claims cutover")
    check(receipt_schema["properties"]["source_removal_status"]["const"] == "not-executed", "receipt claims removal")
    check(receipt_schema["properties"]["assurance_level"]["const"] == "identity-plus-git-remote-ref", "wrong assurance")
    check(receipt_schema["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drifted")
    source = AUTHORIZER.read_text(encoding="utf-8")
    for token in [
        "approved-not-executed",
        "not-executed",
        "ls-remote",
        "run_rehearsal",
        "restore_from_backup",
        "Re-read every permission-bearing input",
    ]:
        check(token in source, f"authorizer lacks contract token {token}")
    check("shutil.rmtree(repo_root" not in source, "authorizer contains direct canonical-root removal")
    check("clear_repository(repo_root" not in source, "authorizer contains direct canonical-root clearing")
    composed = COMPOSED_GATE.read_text(encoding="utf-8")
    check(
        'export SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN"'
        in composed,
        "composed gate does not forward current-source Madaros",
    )


def prepare_complete_fixture(work: Path, *, rehearsal_mutates: bool = False):
    repo = work / "source-repo"
    repo.mkdir(parents=True)
    artifacts = prepare_authorization(work, repo, rehearsal_mutates=rehearsal_mutates)
    repositories, destination_rows = prepare_destination_repositories(work, artifacts)
    policy = work / "cutover-policy.json"
    policy_payload = create_policy(artifacts, destination_rows, policy)
    workspace = work / "cutover-workspace"
    workspace.mkdir()
    return artifacts, repositories, policy, policy_payload, workspace


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-canonical-cutover-approval-gate.") as temporary:
        work = Path(temporary)
        fixture_a = work / "fixture-a"
        fixture_b = work / "fixture-b"
        artifacts, repositories, policy, policy_payload, workspace = prepare_complete_fixture(fixture_a)
        source = Path(artifacts["repo"])
        source_before = snapshot(source)
        destination_before = {path.name: snapshot(path) for path in repositories.iterdir() if path.is_dir()}
        receipt = fixture_a / "cutover-approval.json"

        authorized = run(
            approval_command("authorize", artifacts, repositories, policy, policy_payload, workspace, receipt)
        )
        check("status=approved-not-executed" in authorized.stdout, "approval output lacks exact status")
        check("execution=not-executed" in authorized.stdout, "approval output claims cutover")
        check(receipt.is_file(), "cutover approval receipt was not emitted")
        receipt_payload = json.loads(receipt.read_text(encoding="ascii"))
        check(receipt_payload["approval_identity_sha256"] == digest(receipt_payload, "approval_identity_sha256"), "receipt identity mismatch")
        check(receipt_payload["approval_context"] == "disposable-fixture", "fixture context was widened")
        check(receipt_payload["canonical_cutover_approval_status"] == "approved-not-executed", "wrong approval status")
        check(receipt_payload["canonical_cutover_execution_status"] == "not-executed", "receipt claims cutover")
        check(receipt_payload["source_removal_status"] == "not-executed", "receipt claims source removal")
        check(receipt_payload["explicit_cli_confirmations"]["confirmation_status"] == "matched", "CLI confirmations not recorded")
        check(receipt_payload["explicit_cli_confirmations"]["policy_identity_sha256"] == policy_payload["policy_identity_sha256"], "CLI policy confirmation differs")
        check(receipt_payload["rehearsal_evidence"]["rehearsal_status"] == "removed-repaired-gates-passed-and-pre-tree-restored", "rehearsal not witnessed")
        check(receipt_payload["rehearsal_evidence"]["pre_cutover_tree_sha256"] == receipt_payload["rehearsal_evidence"]["restored_tree_sha256"], "rehearsal restoration differs")
        check(snapshot(source) == source_before, "approval changed canonical fixture")
        check((source / "packages" / "pkg" / "README.md").is_file(), "approval removed package source")
        check((source / "research" / "study.sio").is_file(), "approval removed research source")
        check({path.name: snapshot(path) for path in repositories.iterdir() if path.is_dir()} == destination_before, "approval changed destination repositories")

        verified = run(
            approval_command("verify", artifacts, repositories, policy, policy_payload, workspace, receipt)
        )
        check("CANONICAL_CUTOVER_APPROVAL_VERIFY_PASS" in verified.stdout, "verification lacks pass marker")
        check(snapshot(source) == source_before, "verification changed canonical fixture")

        artifacts_b, repositories_b, policy_b, policy_payload_b, workspace_b = prepare_complete_fixture(fixture_b)
        receipt_b = fixture_b / "cutover-approval.json"
        run(
            approval_command(
                "authorize", artifacts_b, repositories_b, policy_b, policy_payload_b, workspace_b, receipt_b
            )
        )
        check(policy.read_bytes() == policy_b.read_bytes(), "cutover policy is not deterministic across equivalent roots")
        check(receipt.read_bytes() == receipt_b.read_bytes(), "approval receipt is not deterministic across equivalent roots")

        policy_cases = [
            ("unhashed", lambda value: value.__setitem__("approval_status", "pending"), False, "E-SRB-CUTOVER-001"),
            ("pending", lambda value: value.__setitem__("approval_status", "pending"), True, "E-SRB-CUTOVER-005"),
            ("wrong-auth", lambda value: value["source_bindings"].__setitem__("authorization_identity_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-materialization", lambda value: value["source_bindings"].__setitem__("materialization_identity_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-pre-tree", lambda value: value["source_bindings"].__setitem__("pre_cutover_tree_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-post-tree", lambda value: value["source_bindings"].__setitem__("authorized_post_cutover_tree_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-repairs", lambda value: value["source_bindings"].__setitem__("repair_set_identity_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-gates", lambda value: value["source_bindings"].__setitem__("gate_set_identity_sha256", "0" * 64), True, "E-SRB-CUTOVER-002"),
            ("wrong-source-head", lambda value: value["canonical_repository"].__setitem__("head_oid", "0" * 40), True, "E-SRB-CUTOVER-003"),
            ("wrong-source-remote", lambda value: value["canonical_repository"].__setitem__("remote_url", "../remotes/other.git"), True, "E-SRB-CUTOVER-003"),
            ("missing-destination", lambda value: value["destinations"].pop(), True, "E-SRB-CUTOVER-004"),
            ("wrong-destination-head", lambda value: value["destinations"][0].__setitem__("head_oid", "0" * 40), True, "E-SRB-CUTOVER-003"),
            ("wrong-destination-tree", lambda value: value["destinations"][0].__setitem__("tree_sha256", "0" * 64), True, "E-SRB-CUTOVER-004"),
            ("wrong-owner-evidence", lambda value: value["destinations"][0]["owner_approval_evidence"].__setitem__("sha256", "0" * 64), True, "E-SRB-CUTOVER-005"),
            ("wrong-recovery", lambda value: value["recovery_plan"].__setitem__("recovery_plan_identity_sha256", "0" * 64), True, "E-SRB-CUTOVER-005"),
            ("wrong-operator-auth", lambda value: value["operator_approval"].__setitem__("authorization_identity_confirmation", "0" * 64), True, "E-SRB-CUTOVER-005"),
            ("repairs-unreviewed", lambda value: value["operator_approval"].__setitem__("repairs_reviewed", False), True, "E-SRB-CUTOVER-005"),
            ("widened-context", lambda value: value.__setitem__("approval_context", "canonical-production"), True, "E-SRB-CUTOVER-005"),
            ("wrong-limitations", lambda value: value["limitations"].pop(), True, "E-SRB-CUTOVER-001"),
            ("extra-field", lambda value: value.__setitem__("unexpected", True), True, "E-SRB-CUTOVER-001"),
        ]
        for name, mutate, rehash, code in policy_cases:
            bad_policy, bad_payload = clone_policy(policy, fixture_a / f"{name}.json", mutate, rehash=rehash)
            refused_receipt = fixture_a / f"{name}-receipt.json"
            assert_refusal(
                approval_command(
                    "authorize", artifacts, repositories, bad_policy, bad_payload, workspace, refused_receipt
                ),
                refused_receipt,
                code,
            )
            check(snapshot(source) == source_before, f"{name} refusal changed canonical fixture")

        for option in [
            "--confirm-authorization-identity",
            "--confirm-scope-identity",
            "--confirm-policy-identity",
            "--confirm-pre-cutover-tree",
        ]:
            refused_receipt = fixture_a / f"confirmation-{option[10:]}.json"
            command = approval_command(
                "authorize", artifacts, repositories, policy, policy_payload, workspace, refused_receipt
            )
            command[command.index(option) + 1] = "0" * 64
            assert_refusal(command, refused_receipt, "E-SRB-CUTOVER-007")
            check(snapshot(source) == source_before, f"{option} refusal changed canonical fixture")

        occupied = fixture_a / "occupied.json"
        occupied.write_text("preserve\n", encoding="utf-8")
        assert_refusal(
            approval_command("authorize", artifacts, repositories, policy, policy_payload, workspace, occupied),
            None,
            "E-SRB-CUTOVER-008",
        )
        check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied receipt was overwritten")

        dirty = source / "cutover-approval" / "root-marker.json"
        original_mode = dirty.stat().st_mode
        dirty.chmod(0o755)
        dirty_receipt = fixture_a / "dirty-source.json"
        assert_refusal(
            approval_command("authorize", artifacts, repositories, policy, policy_payload, workspace, dirty_receipt),
            dirty_receipt,
            "E-SRB-CUTOVER-003",
        )
        dirty.chmod(original_mode)
        check(snapshot(source) == source_before, "dirty source test did not restore fixture")

        first_destination = next(path for path in repositories.iterdir() if path.is_dir())
        destination_dirty = next(path for path in first_destination.rglob("*") if path.is_file() and ".git" not in path.parts)
        destination_original_mode = destination_dirty.stat().st_mode
        destination_dirty.chmod(0o755)
        dirty_destination_receipt = fixture_a / "dirty-destination.json"
        assert_refusal(
            approval_command(
                "authorize", artifacts, repositories, policy, policy_payload, workspace, dirty_destination_receipt
            ),
            dirty_destination_receipt,
            "E-SRB-CUTOVER-003",
        )
        destination_dirty.chmod(destination_original_mode)

        linked_source = fixture_a / "linked-source-worktree"
        run(
            ["git", "worktree", "add", "-b", "fixture-linked", str(linked_source), "HEAD"],
            cwd=source,
            env=git_environment(),
        )
        linked_artifacts = {**artifacts, "repo": linked_source}
        linked_receipt = fixture_a / "linked-worktree-receipt.json"
        assert_refusal(
            approval_command(
                "authorize",
                linked_artifacts,
                repositories,
                policy,
                policy_payload,
                workspace,
                linked_receipt,
            ),
            linked_receipt,
            "E-SRB-CUTOVER-003",
        )
        run(["git", "worktree", "remove", str(linked_source)], cwd=source, env=git_environment())

        missing_destination = first_destination.with_name(f"{first_destination.name}.absent")
        first_destination.rename(missing_destination)
        missing_receipt = fixture_a / "missing-checkout-receipt.json"
        assert_refusal(
            approval_command("authorize", artifacts, repositories, policy, policy_payload, workspace, missing_receipt),
            missing_receipt,
            "E-SRB-CUTOVER-004",
        )
        missing_destination.rename(first_destination)

        extra_checkout = repositories / "unexpected-repository"
        extra_checkout.mkdir()
        extra_receipt = fixture_a / "extra-checkout-receipt.json"
        assert_refusal(
            approval_command("authorize", artifacts, repositories, policy, policy_payload, workspace, extra_receipt),
            extra_receipt,
            "E-SRB-CUTOVER-004",
        )
        extra_checkout.rmdir()

        source_head = policy_payload["canonical_repository"]["head_oid"]
        source_tree = run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=source, env=git_environment()
        ).stdout.strip()
        source_remote = Path(artifacts["remotes"]) / "source.git"
        source_drift = run(
            ["git", "--git-dir", str(source_remote), "commit-tree", source_tree, "-m", "remote drift"],
            env=git_environment(),
        ).stdout.strip()
        run(["git", "--git-dir", str(source_remote), "update-ref", "refs/heads/main", source_drift])
        source_remote_receipt = fixture_a / "source-remote-mismatch-receipt.json"
        assert_refusal(
            approval_command(
                "authorize", artifacts, repositories, policy, policy_payload, workspace, source_remote_receipt
            ),
            source_remote_receipt,
            "E-SRB-CUTOVER-003",
        )
        run(["git", "--git-dir", str(source_remote), "update-ref", "refs/heads/main", source_head])

        first_row = policy_payload["destinations"][0]
        bound_checkout = repositories / first_row["checkout_path"]
        destination_tree = run(
            ["git", "rev-parse", "HEAD^{tree}"], cwd=bound_checkout, env=git_environment()
        ).stdout.strip()
        destination_suffix = first_row["target_id"].split(":", 1)[1]
        destination_remote = Path(artifacts["remotes"]) / f"{destination_suffix}.git"
        destination_drift = run(
            ["git", "--git-dir", str(destination_remote), "commit-tree", destination_tree, "-m", "remote drift"],
            env=git_environment(),
        ).stdout.strip()
        run(["git", "--git-dir", str(destination_remote), "update-ref", "refs/heads/main", destination_drift])
        destination_remote_receipt = fixture_a / "destination-remote-mismatch-receipt.json"
        assert_refusal(
            approval_command(
                "authorize",
                artifacts,
                repositories,
                policy,
                policy_payload,
                workspace,
                destination_remote_receipt,
            ),
            destination_remote_receipt,
            "E-SRB-CUTOVER-003",
        )
        run(
            [
                "git",
                "--git-dir",
                str(destination_remote),
                "update-ref",
                "refs/heads/main",
                first_row["head_oid"],
            ]
        )

        for name, mutate, rehash in [
            ("receipt-unhashed", lambda value: value.__setitem__("canonical_cutover_execution_status", "executed"), False),
            ("receipt-rehashed-status", lambda value: value.__setitem__("canonical_cutover_execution_status", "executed"), True),
            ("receipt-rehashed-head", lambda value: value["canonical_repository"].__setitem__("head_oid", "0" * 40), True),
            ("receipt-rehashed-rehearsal", lambda value: value["rehearsal_evidence"].__setitem__("rehearsal_status", "failed"), True),
        ]:
            bad_receipt = clone_receipt(receipt, fixture_a / f"{name}.json", mutate, rehash=rehash)
            assert_refusal(
                approval_command("verify", artifacts, repositories, policy, policy_payload, workspace, bad_receipt),
                None,
                "E-SRB-CUTOVER-009",
            )

        mutation_fixture = work / "mutation-fixture"
        mut_artifacts, mut_repositories, mut_policy, mut_payload, mut_workspace = prepare_complete_fixture(
            mutation_fixture, rehearsal_mutates=True
        )
        mutation_before = snapshot(Path(mut_artifacts["repo"]))
        mutation_receipt = mutation_fixture / "cutover-approval.json"
        assert_refusal(
            approval_command(
                "authorize",
                mut_artifacts,
                mut_repositories,
                mut_policy,
                mut_payload,
                mut_workspace,
                mutation_receipt,
            ),
            mutation_receipt,
            "E-SRB-CUTOVER-006",
        )
        check(snapshot(Path(mut_artifacts["repo"])) == mutation_before, "mutating rehearsal escaped disposable copy")

        run(approval_command("verify", artifacts, repositories, policy, policy_payload, workspace, receipt))
        check(snapshot(source) == source_before, "final verification changed canonical fixture")
        check({path.name: snapshot(path) for path in repositories.iterdir() if path.is_dir()} == destination_before, "adversarial gate changed destinations")

    print(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_WITNESS "
        f"approval_identity={receipt_payload['approval_identity_sha256']} "
        f"policy_identity={policy_payload['policy_identity_sha256']} "
        f"authorization_identity={receipt_payload['source_bindings']['authorization_identity_sha256']} "
        f"destinations={len(receipt_payload['destinations'])} "
        f"context={receipt_payload['approval_context']} "
        f"status={receipt_payload['canonical_cutover_approval_status']} "
        f"execution={receipt_payload['canonical_cutover_execution_status']}"
    )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
