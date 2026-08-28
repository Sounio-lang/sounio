#!/usr/bin/env python3
"""Adversarial acceptance gate for R3 canonical Git cutover execution."""

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
sys.path.insert(0, str(ROOT / "tools" / "science_boundary"))
import physical_extraction_canonical_cutover_approval_gate as approval_gate  # noqa: E402
import canonical_cutover_executor as executor  # noqa: E402


EXECUTOR = ROOT / "tools" / "science_boundary" / "canonical_cutover_executor.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_canonical_cutover_execution_gate.sh"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-cutover-execution-policy.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-canonical-cutover-execution.v1.schema.json"
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
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(command, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
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


def digest(value: object, field: str | None = None) -> str:
    payload = json.loads(json.dumps(value))
    if field is not None and isinstance(payload, dict):
        payload.pop(field, None)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def git(command: list[str], cwd: Path) -> str:
    return run(["git", *command], cwd=cwd, env=approval_gate.git_environment()).stdout.strip()


def git_state(repo: Path, remote: Path, branch: str = "main") -> dict[str, object]:
    return {
        "snapshot": approval_gate.snapshot(repo),
        "head": git(["rev-parse", "HEAD"], repo),
        "tree": git(["rev-parse", "HEAD^{tree}"], repo),
        "remote": git(["--git-dir", str(remote), "rev-parse", f"refs/heads/{branch}"], repo),
        "index": git(["ls-files", "--stage"], repo),
        "status": git(["status", "--porcelain=v1", "--untracked-files=all"], repo),
    }


def replace_once(source: str, token: str, replacement: str, label: str) -> str:
    if source.count(token) != 1:
        raise AssertionError(f"fixture source token drifted for {label}")
    return source.replace(token, replacement)


def create_custom_fixture(
    work: Path,
    *,
    canonical_gate_mutates: bool = False,
    verification_gate_mutates: bool = False,
    promotion_race_receipt: Path | None = None,
):
    repo = work / "source-repo"
    repo.mkdir(parents=True)
    execution_gate = approval_gate.execution_gate
    rings, ownership = execution_gate.create_source_fixture(
        repo, verification_mutates=verification_gate_mutates
    )
    if canonical_gate_mutates:
        gate_path = repo / "post_removal_gate.py"
        source = gate_path.read_text(encoding="utf-8")
        source = replace_once(
            source,
            'print("POST_REMOVAL_GATE_PASS")',
            '(root / "unexpected-canonical-mutation.txt").write_text("unexpected\\n") if (root / ".git").is_dir() else None\n'
            'print("POST_REMOVAL_GATE_PASS")',
            "canonical-only mutation",
        )
        gate_path.write_text(source, encoding="utf-8", newline="\n")
    if promotion_race_receipt is not None:
        gate_path = repo / "post_removal_gate.py"
        source = gate_path.read_text(encoding="utf-8")
        source = replace_once(
            source,
            "import os\n",
            "import os\nimport subprocess\nimport sys\n",
            "promotion-race imports",
        )
        watcher = (
            "import pathlib,sys,time\n"
            "p=pathlib.Path(sys.argv[1])\n"
            "pattern='.'+p.name+'.*.staging'\n"
            "while not list(p.parent.glob(pattern)): time.sleep(0.001)\n"
            "p.write_text('occupied-after-stage\\n', encoding='ascii')\n"
        )
        source = replace_once(
            source,
            'print("POST_REMOVAL_GATE_PASS")',
            "subprocess.Popen([sys.executable, '-c', "
            + repr(watcher)
            + ", "
            + repr(str(promotion_race_receipt))
            + "], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) "
            + "if (root / '.git').is_dir() else None\n"
            + 'print("POST_REMOVAL_GATE_PASS")',
            "promotion-race watcher",
        )
        gate_path.write_text(source, encoding="utf-8", newline="\n")
    approval_gate.add_cutover_evidence(repo)
    remotes = work / "remotes"
    source_git = approval_gate.initialize_git_repository(repo, remotes / "source.git", "../remotes/source.git")

    inventory = work / "inventory.json"
    run(execution_gate.auth_gate.material_gate.inventory_command(repo, rings, ownership, inventory))
    destination_policy = work / "destination-policy.json"
    destination_policy_payload = execution_gate.auth_gate.material_gate.create_policy(
        repo, inventory, destination_policy
    )
    destinations = work / "materialized-destinations"
    execution_gate.auth_gate.material_gate.create_destinations(
        destinations, inventory, destination_policy_payload
    )
    materialization = work / "materialization.json"
    run(
        execution_gate.auth_gate.material_gate.materialize_command(
            repo,
            rings,
            ownership,
            inventory,
            destination_policy,
            destinations,
            materialization,
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
    artifacts: dict[str, object] = {
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
        "source_git": source_git,
    }
    repositories, destination_rows = approval_gate.prepare_destination_repositories(work, artifacts)
    cutover_policy = work / "cutover-policy.json"
    cutover_policy_payload = approval_gate.create_policy(artifacts, destination_rows, cutover_policy)
    approval_workspace = work / "cutover-approval-workspace"
    approval_workspace.mkdir()
    return artifacts, repositories, cutover_policy, cutover_policy_payload, approval_workspace


def prepare_fixture(
    work: Path,
    *,
    canonical_gate_mutates: bool = False,
    verification_gate_mutates: bool = False,
    promotion_race_receipt: Path | None = None,
) -> dict[str, object]:
    if canonical_gate_mutates or verification_gate_mutates or promotion_race_receipt is not None:
        prepared = create_custom_fixture(
            work,
            canonical_gate_mutates=canonical_gate_mutates,
            verification_gate_mutates=verification_gate_mutates,
            promotion_race_receipt=promotion_race_receipt,
        )
    else:
        prepared = approval_gate.prepare_complete_fixture(work)
    artifacts, repositories, cutover_policy, cutover_policy_payload, approval_workspace = prepared
    approval_receipt = work / "cutover-approval.json"
    run(
        approval_gate.approval_command(
            "authorize",
            artifacts,
            repositories,
            cutover_policy,
            cutover_policy_payload,
            approval_workspace,
            approval_receipt,
        )
    )
    execution_workspace = work / "cutover-execution-workspace"
    execution_workspace.mkdir()
    execution_policy = work / "cutover-execution-policy.json"
    execution_policy_payload = create_execution_policy(
        artifacts,
        repositories,
        cutover_policy,
        approval_receipt,
        execution_workspace,
        execution_policy,
    )
    return {
        "work": work,
        "artifacts": artifacts,
        "repositories": repositories,
        "cutover_policy": cutover_policy,
        "cutover_policy_payload": cutover_policy_payload,
        "approval_workspace": approval_workspace,
        "approval_receipt": approval_receipt,
        "execution_workspace": execution_workspace,
        "execution_policy": execution_policy,
        "execution_policy_payload": execution_policy_payload,
    }


def create_execution_policy(
    artifacts: dict[str, object],
    repositories: Path,
    cutover_policy: Path,
    approval_receipt: Path,
    workspace: Path,
    output: Path,
) -> dict[str, object]:
    del repositories
    repo = Path(artifacts["repo"])
    approval_raw = approval_receipt.read_bytes()
    approval = json.loads(approval_raw)
    cutover_policy_raw = cutover_policy.read_bytes()
    cutover_policy_payload = json.loads(cutover_policy_raw)
    authorization = json.loads(Path(artifacts["authorization"]).read_text(encoding="ascii"))
    pre_files = executor.scan_repository(repo)
    plan = {
        "author_name": "Sounio Fixture Cutover",
        "author_email": "fixture-cutover@sounio.invalid",
        "author_date": "1767225600 +0000",
        "committer_name": "Sounio Fixture Cutover",
        "committer_email": "fixture-cutover@sounio.invalid",
        "committer_date": "1767225600 +0000",
        "message": "Execute approved physical extraction fixture cutover\n",
        "local_ref_update": "compare-and-swap-pre-head-to-expected-commit",
        "remote_ref_update": "exact-force-with-lease-pre-head-to-expected-commit",
    }
    canonical_approval = approval["canonical_repository"]
    tree_oid, commit_oid = executor.compute_expected_git_transition(
        repo,
        workspace,
        authorization,
        pre_files,
        executor.validate_commit_plan(plan),
        canonical_approval["head_oid"],
    )
    source_bindings = executor.expected_source_bindings(
        approval, approval_raw, cutover_policy_payload, cutover_policy_raw
    )
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-canonical-cutover-execution-policy.v1",
        "policy_type": "explicit-canonical-cutover-execution-policy",
        "authority_scope": "exact-approved-git-cutover-execution",
        "execution_context": approval["approval_context"],
        "approval_status": "approved",
        "source_bindings": source_bindings,
        "canonical_repository": {
            "repository_id": canonical_approval["repository_id"],
            "remote_name": canonical_approval["remote_name"],
            "remote_url": canonical_approval["remote_url"],
            "branch": canonical_approval["branch"],
            "pre_cutover_head_oid": canonical_approval["head_oid"],
            "pre_cutover_remote_head_oid": canonical_approval["remote_head_oid"],
            "expected_post_cutover_git_tree_oid": tree_oid,
            "expected_cutover_commit_oid": commit_oid,
        },
        "commit_plan": plan,
        "execution_authorization": {
            "approved_by": "fixture-operator",
            "approval_evidence": [
                approval_gate.binding(repo / "cutover-approval" / "operator.txt", repo)
            ],
            "cutover_approval_identity_confirmation": approval["approval_identity_sha256"],
            "pre_cutover_head_confirmation": canonical_approval["head_oid"],
            "pre_cutover_remote_head_confirmation": canonical_approval["remote_head_oid"],
            "authorized_post_cutover_tree_confirmation": source_bindings[
                "authorized_post_cutover_tree_sha256"
            ],
            "expected_cutover_commit_confirmation": commit_oid,
            "destination_set_identity_confirmation": source_bindings[
                "destination_set_identity_sha256"
            ],
            "recovery_plan_identity_confirmation": source_bindings[
                "recovery_plan_identity_sha256"
            ],
            "remote_update_reviewed": True,
            "recovery_plan_reviewed": True,
        },
        "limitations": POLICY_LIMITATIONS,
    }
    payload["policy_identity_sha256"] = digest(payload, "policy_identity_sha256")
    write_json(output, payload)
    return payload


def execution_command(
    mode: str,
    fixture: dict[str, object],
    receipt: Path,
    *,
    policy: Path | None = None,
    policy_payload: dict[str, object] | None = None,
    approval_receipt: Path | None = None,
) -> list[str]:
    artifacts = fixture["artifacts"]
    assert isinstance(artifacts, dict)
    selected_policy = policy or Path(fixture["execution_policy"])
    selected_payload = policy_payload or fixture["execution_policy_payload"]
    assert isinstance(selected_payload, dict)
    selected_approval = approval_receipt or Path(fixture["approval_receipt"])
    authorization = json.loads(Path(artifacts["authorization"]).read_text(encoding="ascii"))
    cutover_policy_payload = fixture["cutover_policy_payload"]
    assert isinstance(cutover_policy_payload, dict)
    canonical = selected_payload["canonical_repository"]
    assert isinstance(canonical, dict)
    command = [
        sys.executable,
        str(EXECUTOR),
        mode,
        "--repo-root",
        str(artifacts["repo"]),
        "--destinations-root",
        str(artifacts["destinations"]),
        "--materialization-receipt",
        str(artifacts["materialization"]),
        "--authorization-receipt",
        str(artifacts["authorization"]),
        "--repositories-root",
        str(fixture["repositories"]),
        "--cutover-policy",
        str(fixture["cutover_policy"]),
        "--cutover-approval-receipt",
        str(selected_approval),
        "--execution-policy",
        str(selected_policy),
        "--workspace-root",
        str(fixture["execution_workspace"]),
        "--execution-receipt",
        str(receipt),
        "--confirm-authorization-identity",
        authorization["authorization_identity_sha256"],
        "--confirm-scope-identity",
        authorization["removal_scope"]["scope_identity_sha256"],
        "--confirm-policy-identity",
        str(cutover_policy_payload["policy_identity_sha256"]),
        "--confirm-pre-cutover-tree",
        authorization["candidate_evidence"]["original_source_tree_sha256"],
        "--confirm-cutover-approval-identity",
        str(selected_payload["source_bindings"]["cutover_approval_identity_sha256"]),
        "--confirm-execution-policy-identity",
        str(selected_payload["policy_identity_sha256"]),
        "--confirm-pre-cutover-head",
        str(canonical["pre_cutover_head_oid"]),
        "--confirm-expected-cutover-commit",
        str(canonical["expected_cutover_commit_oid"]),
        "--confirm-execution-context",
        str(selected_payload["execution_context"]),
    ]
    if mode == "execute":
        command.extend(
            [
                "--rings",
                "science-rings.tsv",
                "--ownership",
                "ownership.tsv",
                "--inventory",
                str(artifacts["inventory"]),
                "--destination-policy",
                str(artifacts["destination_policy"]),
                "--removal-policy",
                str(artifacts["removal_policy"]),
            ]
        )
    return command


def clone_policy(
    original: Path,
    destination: Path,
    mutate,
    *,
    rehash: bool = True,
) -> tuple[Path, dict[str, object]]:
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
        payload["execution_identity_sha256"] = digest(payload, "execution_identity_sha256")
    return write_json(destination, payload)


def mutate_expected_commit(payload: dict[str, object]) -> None:
    canonical = payload["canonical_repository"]
    authorization = payload["execution_authorization"]
    assert isinstance(canonical, dict) and isinstance(authorization, dict)
    canonical["expected_cutover_commit_oid"] = "0" * 40
    authorization["expected_cutover_commit_confirmation"] = "0" * 40


def assert_refusal(command: list[str], receipt: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if receipt is not None:
        check(not receipt.exists(), f"refused execution left receipt {receipt}")
    check(code in result.stderr, f"refusal lacks {code}: {result.stderr}")
    check(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_REFUSED" in result.stderr,
        "refusal lacks canonical cutover execution marker",
    )
    return result


def assert_static_contracts() -> None:
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(
        policy_schema["properties"]["schema"]["const"]
        == "sounio.physical-extraction-canonical-cutover-execution-policy.v1",
        "bad execution policy schema",
    )
    check(policy_schema["properties"]["approval_status"]["const"] == "approved", "policy permits pending approval")
    check(policy_schema["properties"]["limitations"]["const"] == POLICY_LIMITATIONS, "policy limitations drifted")
    check(receipt_schema["properties"]["canonical_cutover_approval_status"]["const"] == "consumed", "receipt approval status drifted")
    check(receipt_schema["properties"]["canonical_cutover_execution_status"]["const"] == "executed-and-verified", "receipt execution status drifted")
    check(receipt_schema["properties"]["source_removal_status"]["const"] == "executed", "receipt removal status drifted")
    check(receipt_schema["properties"]["assurance_level"]["const"] == executor.ASSURANCE_LEVEL, "receipt assurance drifted")
    check(receipt_schema["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drifted")
    source = EXECUTOR.read_text(encoding="utf-8")
    for token in [
        "commit-tree",
        "write-tree",
        "update-ref",
        "force-with-lease",
        "verify_approval_locked",
        "restore_transaction",
        "atomic-hardlink-after-remote-ref-verification",
    ]:
        check(token in source, f"executor lacks contract token {token}")
    composed = COMPOSED_GATE.read_text(encoding="utf-8")
    check(
        'export SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_APPROVAL_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_MADAROS_BIN"'
        in composed,
        "composed gate does not forward current-source Madaros",
    )


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-canonical-cutover-execution-gate.") as temporary:
        work = Path(temporary)
        fixture_a = prepare_fixture(work / "fixture-a")
        fixture_b = prepare_fixture(work / "fixture-b")
        policy_a = Path(fixture_a["execution_policy"])
        policy_b = Path(fixture_b["execution_policy"])
        check(policy_a.read_bytes() == policy_b.read_bytes(), "equivalent roots produced different execution policies")
        approval_a = Path(fixture_a["approval_receipt"])
        approval_b = Path(fixture_b["approval_receipt"])
        check(approval_a.read_bytes() == approval_b.read_bytes(), "equivalent roots produced different approvals")

        artifacts_a = fixture_a["artifacts"]
        artifacts_b = fixture_b["artifacts"]
        assert isinstance(artifacts_a, dict) and isinstance(artifacts_b, dict)
        source_a = Path(artifacts_a["repo"])
        source_b = Path(artifacts_b["repo"])
        remote_a = Path(artifacts_a["remotes"]) / "source.git"
        remote_b = Path(artifacts_b["remotes"]) / "source.git"
        before_a = git_state(source_a, remote_a)
        before_b = git_state(source_b, remote_b)
        destination_before_a = {
            path.name: approval_gate.snapshot(path)
            for path in Path(fixture_a["repositories"]).iterdir()
            if path.is_dir()
        }
        receipt_a = Path(fixture_a["work"]) / "canonical-cutover-execution.json"
        receipt_b = Path(fixture_b["work"]) / "canonical-cutover-execution.json"
        result_a = run(execution_command("execute", fixture_a, receipt_a))
        result_b = run(execution_command("execute", fixture_b, receipt_b))
        check("PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_PASS" in result_a.stdout, "execution A lacks pass marker")
        check("PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_PASS" in result_b.stdout, "execution B lacks pass marker")
        check(receipt_a.read_bytes() == receipt_b.read_bytes(), "equivalent roots produced different execution receipts")
        receipt_payload = json.loads(receipt_a.read_text(encoding="ascii"))
        policy_payload = fixture_a["execution_policy_payload"]
        assert isinstance(policy_payload, dict)
        expected_commit = policy_payload["canonical_repository"]["expected_cutover_commit_oid"]
        expected_tree = policy_payload["canonical_repository"]["expected_post_cutover_git_tree_oid"]
        check(receipt_payload["execution_identity_sha256"] == digest(receipt_payload, "execution_identity_sha256"), "bad receipt identity")
        check(receipt_payload["canonical_cutover_approval_status"] == "consumed", "approval was not consumed")
        check(receipt_payload["canonical_cutover_execution_status"] == "executed-and-verified", "cutover was not executed")
        check(receipt_payload["source_removal_status"] == "executed", "source removal was not executed")
        check(receipt_payload["execution_context"] == "disposable-fixture", "fixture context drifted")
        check(receipt_payload["assurance_level"] == executor.ASSURANCE_LEVEL, "assurance drifted")
        for source, remote, artifacts, before in (
            (source_a, remote_a, artifacts_a, before_a),
            (source_b, remote_b, artifacts_b, before_b),
        ):
            check(git(["rev-parse", "HEAD"], source) == expected_commit, "local ref did not reach expected commit")
            check(git(["--git-dir", str(remote), "rev-parse", "refs/heads/main"], source) == expected_commit, "remote ref did not reach expected commit")
            check(git(["rev-parse", "HEAD^{tree}"], source) == expected_tree, "executed Git tree differs")
            check(git(["rev-parse", "HEAD^"], source) == before["head"], "cutover commit parent differs")
            check(not git(["status", "--porcelain=v1", "--untracked-files=all"], source), "executed worktree is dirty")
            authorization = json.loads(Path(artifacts["authorization"]).read_text(encoding="ascii"))
            for unit in authorization["removal_scope"]["units"]:
                check(not (source / unit["source_path"]).exists(), f"source root remains after cutover: {unit['source_path']}")
        destination_after_a = {
            path.name: approval_gate.snapshot(path)
            for path in Path(fixture_a["repositories"]).iterdir()
            if path.is_dir()
        }
        check(destination_after_a == destination_before_a, "cutover changed destination repositories")
        run(execution_command("verify", fixture_a, receipt_a))
        run(execution_command("verify", fixture_b, receipt_b))

        bad_unhashed = clone_receipt(
            receipt_a,
            work / "bad-receipt-unhashed.json",
            lambda payload: payload.__setitem__("source_removal_status", "not-executed"),
            rehash=False,
        )
        assert_refusal(execution_command("verify", fixture_a, bad_unhashed), None, "E-SRB-CUTOVER-EXEC-009")
        bad_rehashed = clone_receipt(
            receipt_a,
            work / "bad-receipt-rehashed.json",
            lambda payload: payload["canonical_repository"].__setitem__("post_cutover_head_oid", "0" * 40),
        )
        assert_refusal(execution_command("verify", fixture_a, bad_rehashed), None, "E-SRB-CUTOVER-EXEC-009")

        reintroduced = source_a / "packages" / "pkg" / "reintroduced.sio"
        reintroduced.parent.mkdir(parents=True)
        reintroduced.write_text("fn reintroduced() -> i64 { 1 }\n", encoding="utf-8")
        assert_refusal(execution_command("verify", fixture_a, receipt_a), None, "E-SRB-CUTOVER-EXEC-009")
        shutil.rmtree(reintroduced.parents[1])
        check(not git(["status", "--porcelain=v1", "--untracked-files=all"], source_a), "reintroduced-source cleanup failed")

        source_tree = git(["rev-parse", "HEAD^{tree}"], source_a)
        remote_drift = git(
            ["--git-dir", str(remote_a), "commit-tree", source_tree, "-p", str(expected_commit), "-m", "remote drift"],
            source_a,
        )
        git(["--git-dir", str(remote_a), "update-ref", "refs/heads/main", remote_drift], source_a)
        assert_refusal(execution_command("verify", fixture_a, receipt_a), None, "E-SRB-CUTOVER-EXEC-009")
        git(["--git-dir", str(remote_a), "update-ref", "refs/heads/main", str(expected_commit)], source_a)

        first_destination = next(path for path in Path(fixture_a["repositories"]).iterdir() if path.is_dir())
        destination_head = git(["rev-parse", "HEAD"], first_destination)
        destination_file = next(path for path in first_destination.rglob("*") if path.is_file() and ".git" not in path.parts)
        destination_raw = destination_file.read_bytes()
        destination_file.write_bytes(destination_raw + b"changed\n")
        assert_refusal(execution_command("verify", fixture_a, receipt_a), None, "E-SRB-CUTOVER-EXEC-006")
        destination_file.write_bytes(destination_raw)
        check(git(["rev-parse", "HEAD"], first_destination) == destination_head, "destination cleanup changed head")

        fixture_c = prepare_fixture(work / "fixture-c")
        artifacts_c = fixture_c["artifacts"]
        assert isinstance(artifacts_c, dict)
        source_c = Path(artifacts_c["repo"])
        remote_c = Path(artifacts_c["remotes"]) / "source.git"
        before_c = git_state(source_c, remote_c)
        policy_c = Path(fixture_c["execution_policy"])
        bad_context, bad_context_payload = clone_policy(
            policy_c,
            work / "bad-context-policy.json",
            lambda payload: payload.__setitem__("execution_context", "canonical-production"),
        )
        assert_refusal(
            execution_command(
                "execute",
                fixture_c,
                work / "bad-context-receipt.json",
                policy=bad_context,
                policy_payload=bad_context_payload,
            ),
            work / "bad-context-receipt.json",
            "E-SRB-CUTOVER-EXEC-005",
        )
        bad_commit, bad_commit_payload = clone_policy(
            policy_c,
            work / "bad-commit-policy.json",
            mutate_expected_commit,
        )
        assert_refusal(
            execution_command(
                "execute",
                fixture_c,
                work / "bad-commit-receipt.json",
                policy=bad_commit,
                policy_payload=bad_commit_payload,
            ),
            work / "bad-commit-receipt.json",
            "E-SRB-CUTOVER-EXEC-004",
        )
        wrong_confirmation = execution_command("execute", fixture_c, work / "wrong-confirmation.json")
        index = wrong_confirmation.index("--confirm-expected-cutover-commit") + 1
        wrong_confirmation[index] = "0" * 40
        assert_refusal(wrong_confirmation, work / "wrong-confirmation.json", "E-SRB-CUTOVER-EXEC-004")
        occupied = work / "occupied-receipt.json"
        occupied.write_text("preserve\n", encoding="ascii")
        result = run(execution_command("execute", fixture_c, occupied), expected=1)
        check("E-SRB-CUTOVER-EXEC-008" in result.stderr, "occupied output lacks refusal code")
        check(occupied.read_text(encoding="ascii") == "preserve\n", "occupied output was overwritten")
        occupied.unlink()

        mode_path = source_c / "consumer" / "config.txt"
        original_mode = mode_path.stat().st_mode & 0o777
        os.chmod(mode_path, original_mode ^ 0o111)
        assert_refusal(execution_command("execute", fixture_c, work / "dirty-mode.json"), work / "dirty-mode.json", "E-SRB-CUTOVER-EXEC-002")
        os.chmod(mode_path, original_mode)
        check(git_state(source_c, remote_c) == before_c, "pre-execution refusals changed canonical fixture")

        fixture_rollback = prepare_fixture(work / "fixture-rollback", canonical_gate_mutates=True)
        rollback_artifacts = fixture_rollback["artifacts"]
        assert isinstance(rollback_artifacts, dict)
        rollback_source = Path(rollback_artifacts["repo"])
        rollback_remote = Path(rollback_artifacts["remotes"]) / "source.git"
        rollback_before = git_state(rollback_source, rollback_remote)
        rollback_receipt = work / "rollback-receipt.json"
        assert_refusal(
            execution_command("execute", fixture_rollback, rollback_receipt),
            rollback_receipt,
            "E-SRB-CUTOVER-EXEC-005",
        )
        check(git_state(rollback_source, rollback_remote) == rollback_before, "failed cutover did not restore exact Git state")

        remote_rollback_receipt = work / "remote-rollback-receipt.json"
        fixture_remote_rollback = prepare_fixture(
            work / "fixture-remote-rollback",
            promotion_race_receipt=remote_rollback_receipt,
        )
        remote_rollback_artifacts = fixture_remote_rollback["artifacts"]
        assert isinstance(remote_rollback_artifacts, dict)
        remote_rollback_source = Path(remote_rollback_artifacts["repo"])
        remote_rollback_remote = Path(remote_rollback_artifacts["remotes"]) / "source.git"
        remote_rollback_before = git_state(remote_rollback_source, remote_rollback_remote)
        result = run(
            execution_command("execute", fixture_remote_rollback, remote_rollback_receipt),
            expected=1,
        )
        check("E-SRB-CUTOVER-EXEC-008" in result.stderr, "post-push promotion race lacks refusal code")
        check(remote_rollback_receipt.exists(), "promotion-race witness did not occupy final receipt")
        check(
            remote_rollback_receipt.read_text(encoding="ascii") == "occupied-after-stage\n",
            "executor overwrote promotion-race witness",
        )
        check(
            git_state(remote_rollback_source, remote_rollback_remote) == remote_rollback_before,
            "post-push receipt failure did not roll back local and remote Git state",
        )
        remote_rollback_receipt.unlink()

        fixture_verification = prepare_fixture(work / "fixture-verification", verification_gate_mutates=True)
        verification_artifacts = fixture_verification["artifacts"]
        assert isinstance(verification_artifacts, dict)
        verification_source = Path(verification_artifacts["repo"])
        verification_receipt = work / "verification-mutation-receipt.json"
        run(execution_command("execute", fixture_verification, verification_receipt))
        verification_before = approval_gate.snapshot(verification_source)
        assert_refusal(
            execution_command("verify", fixture_verification, verification_receipt),
            None,
            "E-SRB-CUTOVER-EXEC-009",
        )
        check(approval_gate.snapshot(verification_source) == verification_before, "mutating verification escaped disposable copy")

        run(execution_command("verify", fixture_a, receipt_a))
        check(not git(["status", "--porcelain=v1", "--untracked-files=all"], source_a), "final verification changed source")

    print(
        "PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_WITNESS "
        f"execution_identity={receipt_payload['execution_identity_sha256']} "
        f"policy_identity={policy_payload['policy_identity_sha256']} "
        f"approval_identity={receipt_payload['source_bindings']['cutover_approval_identity_sha256']} "
        f"commit={expected_commit} context={receipt_payload['execution_context']} "
        f"status={receipt_payload['canonical_cutover_execution_status']}"
    )
    print(f"PHYSICAL_EXTRACTION_CANONICAL_CUTOVER_EXECUTION_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
