#!/usr/bin/env python3
"""Adversarial acceptance gate for R3 source-removal execution."""

from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import physical_extraction_source_removal_authorization_gate as auth_gate  # noqa: E402


EXECUTOR = ROOT / "tools" / "science_boundary" / "source_removal_executor.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_source_removal_execution_gate.sh"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-source-removal-execution-policy.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-source-removal-execution.v1.schema.json"
POLICY_LIMITATIONS = [
    "does_not_create_or_imply_production_approval",
    "authorizes_only_one_exact_local_repository_tree",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "does_not_guarantee_crash_atomicity_across_multiple_filesystem_operations",
    "crash_recovery_requires_the_retained_transaction_workspace",
    "requires_a_quiescent_execution_root_without_nonparticipating_writers",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_assert_this_execution_is_canonical_production_cutover",
    "execution_scope_is_only_the_bound_local_repository_tree",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "operator_label_does_not_prove_human_identity_or_organizational_authority",
    "does_not_guarantee_crash_atomicity_across_multiple_filesystem_operations",
    "crash_recovery_requires_the_retained_transaction_workspace_when_no_receipt_exists",
    "requires_a_quiescent_execution_root_without_nonparticipating_writers",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "post_execution_verification_uses_bound_receipts_and_materialized_copies_not_removed_sources",
]
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, expected: int | set[int] = 0) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
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


def identity(payload: dict[str, object], field: str) -> str:
    value = json.loads(json.dumps(payload))
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def binding(path: Path, repo: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(repo).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def snapshot(repo: Path) -> dict[str, str]:
    result = {}
    for current, directories, names in os.walk(repo, topdown=True, followlinks=False):
        current_path = Path(current)
        if current_path == repo and ".git" in directories:
            directories.remove(".git")
        directories.sort()
        for name in sorted(names):
            if current_path == repo and name == ".git":
                continue
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise AssertionError(f"fixture member is unsafe: {path}")
            result[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def create_source_fixture(
    repo: Path,
    *,
    execution_mutates: bool = False,
    verification_mutates: bool = False,
) -> tuple[Path, Path]:
    rings, ownership = auth_gate.material_gate.create_fixture(repo)
    write(repo / "consumer" / "config.txt", "imports=packages/pkg,research\n")
    write(repo / "repairs" / "config.after.txt", "imports=external:pkg,external:research\n")
    write(repo / "reviews" / "alpha.txt", "alpha reviewed exact removal scope\n")
    write(repo / "reviews" / "beta.txt", "beta reviewed exact removal scope\n")
    write(repo / "execution-approval" / "operator.txt", "fixture operator approved exact local execution\n")
    write_json(
        repo / "execution-approval" / "root-marker.json",
        {
            "schema": "sounio.physical-extraction-source-removal-execution-root.v1",
            "marker_type": "explicit-approved-execution-root",
            "root_key": "approved-execution-root",
            "approval_state": "approved",
            "approved_by": "fixture-operator",
        },
    )
    mutation = """
if os.environ.get("SOUNIO_REMOVAL_EXECUTION_ACTIVE") == "1":
    (root / "unexpected-execution-mutation.txt").write_text("unexpected\\n")
""" if execution_mutates else ""
    verification_mutation = """
if os.environ.get("SOUNIO_REMOVAL_VERIFICATION_ACTIVE") == "1":
    (root / "unexpected-verification-mutation.txt").write_text("unexpected\\n")
""" if verification_mutates else ""
    write(
        repo / "post_removal_gate.py",
        f"""import os
from pathlib import Path
root = Path.cwd()
assert not (root / "packages" / "pkg").exists()
assert not (root / "research").exists()
assert (root / "core" / "compiler.sio").is_file()
assert (root / "stdlib" / "candidate.sio").is_file()
assert (root / "consumer" / "config.txt").read_text() == "imports=external:pkg,external:research\\n"
{mutation}{verification_mutation}print("POST_REMOVAL_GATE_PASS")
""",
    )
    return rings, ownership


def prepare_authorization(
    work: Path,
    repo: Path,
    *,
    execution_mutates: bool = False,
    verification_mutates: bool = False,
) -> dict[str, object]:
    rings, ownership = create_source_fixture(
        repo,
        execution_mutates=execution_mutates,
        verification_mutates=verification_mutates,
    )
    inventory = work / "inventory.json"
    run(auth_gate.material_gate.inventory_command(repo, rings, ownership, inventory))
    destination_policy = work / "destination-policy.json"
    destination_policy_payload = auth_gate.material_gate.create_policy(repo, inventory, destination_policy)
    destinations = work / "destinations"
    auth_gate.material_gate.create_destinations(destinations, inventory, destination_policy_payload)
    materialization = work / "materialization.json"
    run(
        auth_gate.material_gate.materialize_command(
            repo, rings, ownership, inventory, destination_policy, destinations, materialization
        )
    )
    removal_policy = work / "removal-policy.json"
    auth_gate.create_removal_policy(repo, inventory, materialization, removal_policy)
    authorization_workspace = work / "authorization-workspace"
    authorization_workspace.mkdir()
    authorization = work / "authorization.json"
    run(
        auth_gate.authorization_command(
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
    }


def create_execution_policy(repo: Path, artifacts: dict[str, object], output: Path) -> dict[str, object]:
    authorization_raw = artifacts["authorization"].read_bytes()
    authorization = json.loads(authorization_raw)
    materialization_raw = artifacts["materialization"].read_bytes()
    materialization = json.loads(materialization_raw)
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-source-removal-execution-policy.v1",
        "policy_type": "explicit-local-source-removal-execution-policy",
        "authority_scope": "exact-local-repository-tree-execution-approval",
        "source_bindings": {
            "authorization_file_sha256": hashlib.sha256(authorization_raw).hexdigest(),
            "authorization_identity_sha256": authorization["authorization_identity_sha256"],
            "materialization_file_sha256": hashlib.sha256(materialization_raw).hexdigest(),
            "materialization_identity_sha256": materialization["materialization_identity_sha256"],
            "inventory_identity_sha256": authorization["source_bindings"]["inventory_identity_sha256"],
            "pre_execution_tree_sha256": authorization["candidate_evidence"]["original_source_tree_sha256"],
            "post_execution_tree_sha256": authorization["candidate_evidence"]["candidate_tree_sha256"],
        },
        "approval_status": "approved",
        "execution_root_marker": binding(repo / "execution-approval" / "root-marker.json", repo),
        "execution_scope": authorization["removal_scope"],
        "operator_approval": {
            "approved_by": "fixture-operator",
            "approval_evidence": [binding(repo / "execution-approval" / "operator.txt", repo)],
            "authorization_identity_confirmation": authorization["authorization_identity_sha256"],
            "scope_identity_confirmation": authorization["removal_scope"]["scope_identity_sha256"],
            "pre_execution_tree_confirmation": authorization["candidate_evidence"]["original_source_tree_sha256"],
        },
        "limitations": POLICY_LIMITATIONS,
    }
    payload["policy_identity_sha256"] = identity(payload, "policy_identity_sha256")
    write_json(output, payload)
    return payload


def execution_command(
    mode: str,
    repo: Path,
    artifacts: dict[str, object],
    policy: Path,
    policy_payload: dict[str, object],
    workspace: Path,
    receipt: Path,
) -> list[str]:
    authorization = json.loads(artifacts["authorization"].read_text(encoding="ascii"))
    command = [
        sys.executable,
        str(EXECUTOR),
        mode,
        "--repo-root",
        str(repo),
        "--destinations-root",
        str(artifacts["destinations"]),
        "--materialization-receipt",
        str(artifacts["materialization"]),
        "--authorization-receipt",
        str(artifacts["authorization"]),
        "--execution-policy",
        str(policy),
        "--workspace-root",
        str(workspace),
        "--execution-receipt",
        str(receipt),
        "--confirm-authorization-identity",
        authorization["authorization_identity_sha256"],
        "--confirm-scope-identity",
        authorization["removal_scope"]["scope_identity_sha256"],
        "--confirm-policy-identity",
        policy_payload["policy_identity_sha256"],
        "--confirm-pre-execution-tree",
        authorization["candidate_evidence"]["original_source_tree_sha256"],
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


def clone_policy(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["policy_identity_sha256"] = identity(payload, "policy_identity_sha256")
    return write_json(destination, payload)


def clone_receipt(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["execution_identity_sha256"] = identity(payload, "execution_identity_sha256")
    return write_json(destination, payload)


def assert_refusal(command: list[str], receipt: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if receipt is not None:
        check(not receipt.exists(), f"refused execution left receipt {receipt}")
    check(code in result.stderr, f"refusal lacks {code}: {result.stderr}")
    check("PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_REFUSED" in result.stderr, "refusal lacks execution marker")
    return result


def assert_static_contracts() -> None:
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(policy_schema["properties"]["schema"]["const"] == "sounio.physical-extraction-source-removal-execution-policy.v1", "bad policy schema")
    check(policy_schema["properties"]["approval_status"]["const"] == "approved", "policy permits pending execution")
    check(policy_schema["properties"]["limitations"]["const"] == POLICY_LIMITATIONS, "policy limitations drifted")
    check(receipt_schema["properties"]["execution_status"]["const"] == "executed-and-verified", "receipt execution status drifted")
    check(receipt_schema["properties"]["source_removal_status"]["const"] == "executed", "receipt removal status drifted")
    check(receipt_schema["properties"]["assurance_level"]["const"] == "identity-only", "receipt assurance drifted")
    check(receipt_schema["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drifted")
    source = EXECUTOR.read_text(encoding="utf-8")
    for token in [
        "reconstruct_authorization",
        "full-regular-file-pre-execution-copy",
        "restore_from_backup",
        "promoted-after-post-execution-verification",
        "confirm_operation",
    ]:
        check(token in source, f"executor lacks contract token {token}")
    composed = COMPOSED_GATE.read_text(encoding="utf-8")
    check(
        'export SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_MADAROS_BIN"'
        in composed,
        "composed gate does not forward current-source Madaros",
    )


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-source-removal-execution-gate.") as temporary:
        work = Path(temporary)
        source = work / "source-repo"
        source.mkdir()
        artifacts = prepare_authorization(work, source)
        source_before = snapshot(source)
        source_copy = work / "source-copy"
        shutil.copytree(source, source_copy)
        pristine = work / "pristine"
        shutil.copytree(source, pristine)

        execution_policy = work / "execution-policy.json"
        execution_policy_payload = create_execution_policy(source, artifacts, execution_policy)
        workspace_a = work / "workspace-a"
        workspace_b = work / "workspace-b"
        workspace_verify = work / "workspace-verify"
        for workspace in (workspace_a, workspace_b, workspace_verify):
            workspace.mkdir()
        receipt_a = work / "execution-a.json"
        receipt_b = work / "execution-b.json"

        executed_a = run(
            execution_command(
                "execute", source, artifacts, execution_policy, execution_policy_payload, workspace_a, receipt_a
            )
        )
        check("status=executed-and-verified" in executed_a.stdout, "execution output lacks exact status")
        check(receipt_a.is_file(), "execution receipt was not emitted")
        check(not (source / "packages" / "pkg").exists(), "package source was not removed in fixture")
        check(not (source / "research").exists(), "research source was not removed in fixture")
        check((source / "core" / "compiler.sio").is_file(), "retained core was removed")
        check((source / "stdlib" / "candidate.sio").is_file(), "blocked source was removed")
        check((source / "consumer" / "config.txt").read_text(encoding="utf-8") == "imports=external:pkg,external:research\n", "repository repair was not executed")
        check(
            not any(
                path.is_dir() and path.name.startswith(".sounio-source-removal-execution.")
                for path in workspace_a.iterdir()
            ),
            "transaction workspace leaked after success",
        )
        receipt_payload = json.loads(receipt_a.read_text(encoding="ascii"))
        check(receipt_payload["execution_identity_sha256"] == identity(receipt_payload, "execution_identity_sha256"), "execution receipt identity mismatch")
        check(receipt_payload["execution_status"] == "executed-and-verified", "wrong execution status")
        check(receipt_payload["source_removal_status"] == "executed", "wrong removal status")
        check(receipt_payload["summary"]["executed_unit_count"] == 2, "wrong executed unit count")
        check(receipt_payload["summary"]["removed_file_count"] == 3, "wrong removed file count")
        check(receipt_payload["summary"]["repair_count"] == 1, "wrong repair count")
        check(receipt_payload["repairs"][0]["execution_status"] == "executed-and-verified", "repair execution not witnessed")
        check(receipt_payload["post_removal_gates"][0]["execution_gate_status"] == "passed", "execution gate not witnessed")
        check(receipt_payload["transaction_evidence"]["transaction_status"] == "committed", "transaction was not committed")

        run(
            execution_command(
                "execute", source_copy, artifacts, execution_policy, execution_policy_payload, workspace_b, receipt_b
            )
        )
        check(receipt_a.read_bytes() == receipt_b.read_bytes(), "execution receipt is not deterministic across roots")
        verified = run(
            execution_command(
                "verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a
            )
        )
        check("EXECUTION_VERIFY_PASS" in verified.stdout, "execution verification lacks pass marker")
        run(
            execution_command(
                "verify", source_copy, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_b
            )
        )

        policy_cases = [
            ("unhashed-policy", lambda value: value.__setitem__("approval_status", "pending"), False, "E-SRB-EXEC-001"),
            ("pending-policy", lambda value: value.__setitem__("approval_status", "pending"), True, "E-SRB-EXEC-003"),
            ("wrong-authorization", lambda value: value["source_bindings"].__setitem__("authorization_identity_sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("wrong-materialization", lambda value: value["source_bindings"].__setitem__("materialization_identity_sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("wrong-inventory", lambda value: value["source_bindings"].__setitem__("inventory_identity_sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("wrong-pre-tree", lambda value: value["source_bindings"].__setitem__("pre_execution_tree_sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("wrong-post-tree", lambda value: value["source_bindings"].__setitem__("post_execution_tree_sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("scope-missing", lambda value: value["execution_scope"]["units"].pop(), True, "E-SRB-EXEC-003"),
            ("approval-empty", lambda value: value["operator_approval"].__setitem__("approval_evidence", []), True, "E-SRB-EXEC-003"),
            ("approval-hash", lambda value: value["operator_approval"]["approval_evidence"][0].__setitem__("sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("approval-auth", lambda value: value["operator_approval"].__setitem__("authorization_identity_confirmation", "0" * 64), True, "E-SRB-EXEC-003"),
            ("approval-scope", lambda value: value["operator_approval"].__setitem__("scope_identity_confirmation", "0" * 64), True, "E-SRB-EXEC-003"),
            ("approval-tree", lambda value: value["operator_approval"].__setitem__("pre_execution_tree_confirmation", "0" * 64), True, "E-SRB-EXEC-003"),
            ("marker-hash", lambda value: value["execution_root_marker"].__setitem__("sha256", "0" * 64), True, "E-SRB-EXEC-003"),
            ("marker-in-scope", lambda value: value["execution_root_marker"].__setitem__("path", "packages/pkg/README.md"), True, "E-SRB-EXEC-003"),
            ("wrong-limitations", lambda value: value["limitations"].pop(), True, "E-SRB-EXEC-001"),
            ("extra-field", lambda value: value.__setitem__("unexpected", True), True, "E-SRB-EXEC-001"),
        ]
        for name, mutate, rehash, code in policy_cases:
            case_repo = work / f"case-{name}"
            shutil.copytree(pristine, case_repo)
            before = snapshot(case_repo)
            bad_policy = clone_policy(execution_policy, work / f"{name}.json", mutate, rehash=rehash)
            bad_payload = json.loads(bad_policy.read_text(encoding="ascii"))
            bad_receipt = work / f"{name}-receipt.json"
            assert_refusal(
                execution_command("execute", case_repo, artifacts, bad_policy, bad_payload, workspace_a, bad_receipt),
                bad_receipt,
                code,
            )
            check(snapshot(case_repo) == before, f"{name} refusal changed execution root")

        confirmation_names = [
            "--confirm-authorization-identity",
            "--confirm-scope-identity",
            "--confirm-policy-identity",
            "--confirm-pre-execution-tree",
        ]
        for index, option in enumerate(confirmation_names):
            case_repo = work / f"confirmation-{index}"
            shutil.copytree(pristine, case_repo)
            before = snapshot(case_repo)
            receipt = work / f"confirmation-{index}.json"
            command = execution_command(
                "execute", case_repo, artifacts, execution_policy, execution_policy_payload, workspace_a, receipt
            )
            command[command.index(option) + 1] = "0" * 64
            assert_refusal(command, receipt, "E-SRB-EXEC-004")
            check(snapshot(case_repo) == before, f"{option} refusal changed execution root")

        occupied_repo = work / "occupied-repo"
        shutil.copytree(pristine, occupied_repo)
        occupied = work / "occupied-execution.json"
        occupied.write_text("preserve\n", encoding="utf-8")
        assert_refusal(
            execution_command(
                "execute", occupied_repo, artifacts, execution_policy, execution_policy_payload, workspace_a, occupied
            ),
            None,
            "E-SRB-EXEC-008",
        )
        check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied receipt was overwritten")
        check(snapshot(occupied_repo) == snapshot(pristine), "occupied output changed root")

        mutated_repo = work / "mutated-repo"
        shutil.copytree(pristine, mutated_repo)
        write(mutated_repo / "unexpected.txt", "unexpected\n")
        mutated_before = snapshot(mutated_repo)
        assert_refusal(
            execution_command(
                "execute", mutated_repo, artifacts, execution_policy, execution_policy_payload,
                workspace_a, work / "mutated-root-receipt.json",
            ),
            work / "mutated-root-receipt.json",
            "E-SRB-EXEC-002",
        )
        check(snapshot(mutated_repo) == mutated_before, "preflight source mismatch changed root")

        receipt_in_destination_repo = work / "receipt-in-destination-repo"
        shutil.copytree(pristine, receipt_in_destination_repo)
        receipt_in_destination_before = snapshot(receipt_in_destination_repo)
        receipt_in_destination = artifacts["destinations"] / "execution-receipt.json"
        assert_refusal(
            execution_command(
                "execute",
                receipt_in_destination_repo,
                artifacts,
                execution_policy,
                execution_policy_payload,
                workspace_a,
                receipt_in_destination,
            ),
            receipt_in_destination,
            "E-SRB-EXEC-001",
        )
        check(
            snapshot(receipt_in_destination_repo) == receipt_in_destination_before,
            "destination-contained receipt refusal changed root",
        )

        locked_repo = work / "locked-repo"
        shutil.copytree(pristine, locked_repo)
        locked_before = snapshot(locked_repo)
        locked_receipt = work / "locked-receipt.json"
        lock_descriptor = os.open(locked_repo, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_refusal(
                execution_command(
                    "execute",
                    locked_repo,
                    artifacts,
                    execution_policy,
                    execution_policy_payload,
                    workspace_a,
                    locked_receipt,
                ),
                locked_receipt,
                "E-SRB-EXEC-001",
            )
        finally:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
            os.close(lock_descriptor)
        check(snapshot(locked_repo) == locked_before, "root-lock refusal changed execution root")

        rollback_work = work / "rollback-fixture"
        rollback_repo = rollback_work / "repo"
        rollback_repo.mkdir(parents=True)
        rollback_rings, rollback_ownership = create_source_fixture(rollback_repo, execution_mutates=True)
        rollback_inventory = rollback_work / "inventory.json"
        run(auth_gate.material_gate.inventory_command(rollback_repo, rollback_rings, rollback_ownership, rollback_inventory))
        rollback_destination_policy = rollback_work / "destination-policy.json"
        rollback_dp_payload = auth_gate.material_gate.create_policy(rollback_repo, rollback_inventory, rollback_destination_policy)
        rollback_destinations = rollback_work / "destinations"
        auth_gate.material_gate.create_destinations(rollback_destinations, rollback_inventory, rollback_dp_payload)
        rollback_materialization = rollback_work / "materialization.json"
        run(auth_gate.material_gate.materialize_command(rollback_repo, rollback_rings, rollback_ownership, rollback_inventory, rollback_destination_policy, rollback_destinations, rollback_materialization))
        rollback_removal_policy = rollback_work / "removal-policy.json"
        auth_gate.create_removal_policy(rollback_repo, rollback_inventory, rollback_materialization, rollback_removal_policy)
        rollback_auth_workspace = rollback_work / "auth-workspace"
        rollback_auth_workspace.mkdir()
        rollback_authorization = rollback_work / "authorization.json"
        run(auth_gate.authorization_command("authorize", rollback_repo, rollback_rings, rollback_ownership, rollback_inventory, rollback_destination_policy, rollback_destinations, rollback_materialization, rollback_removal_policy, rollback_auth_workspace, rollback_authorization))
        rollback_artifacts = {
            "repo": rollback_repo,
            "rings": rollback_rings,
            "ownership": rollback_ownership,
            "inventory": rollback_inventory,
            "destination_policy": rollback_destination_policy,
            "destinations": rollback_destinations,
            "materialization": rollback_materialization,
            "removal_policy": rollback_removal_policy,
            "authorization": rollback_authorization,
        }
        rollback_policy = rollback_work / "execution-policy.json"
        rollback_policy_payload = create_execution_policy(rollback_repo, rollback_artifacts, rollback_policy)
        rollback_workspace = rollback_work / "execution-workspace"
        rollback_workspace.mkdir()
        rollback_before = snapshot(rollback_repo)
        rollback_receipt = rollback_work / "execution.json"
        assert_refusal(
            execution_command("execute", rollback_repo, rollback_artifacts, rollback_policy, rollback_policy_payload, rollback_workspace, rollback_receipt),
            rollback_receipt,
            "E-SRB-EXEC-006",
        )
        check(snapshot(rollback_repo) == rollback_before, "failed execution did not roll back exact source tree")
        check(
            not any(
                path.is_dir() and path.name.startswith(".sounio-source-removal-execution.")
                for path in rollback_workspace.iterdir()
            ),
            "normal rollback leaked transaction workspace",
        )

        verification_work = work / "verification-mutation-fixture"
        verification_repo = verification_work / "repo"
        verification_repo.mkdir(parents=True)
        verification_artifacts = prepare_authorization(
            verification_work,
            verification_repo,
            verification_mutates=True,
        )
        verification_policy = verification_work / "execution-policy.json"
        verification_policy_payload = create_execution_policy(
            verification_repo,
            verification_artifacts,
            verification_policy,
        )
        verification_workspace = verification_work / "execution-workspace"
        verification_workspace.mkdir()
        verification_receipt = verification_work / "execution.json"
        run(
            execution_command(
                "execute",
                verification_repo,
                verification_artifacts,
                verification_policy,
                verification_policy_payload,
                verification_workspace,
                verification_receipt,
            )
        )
        verification_before = snapshot(verification_repo)
        assert_refusal(
            execution_command(
                "verify",
                verification_repo,
                verification_artifacts,
                verification_policy,
                verification_policy_payload,
                verification_workspace,
                verification_receipt,
            ),
            None,
            "E-SRB-EXEC-009",
        )
        check(
            snapshot(verification_repo) == verification_before,
            "verification gate mutation escaped disposable verification copy",
        )

        receipt_cases = [
            ("receipt-unhashed", lambda value: value.__setitem__("execution_status", "failed"), False),
            ("receipt-rehashed-status", lambda value: value.__setitem__("execution_status", "failed"), True),
            ("receipt-rehashed-tree", lambda value: value["tree_evidence"].__setitem__("post_execution_tree_sha256", "0" * 64), True),
            ("receipt-rehashed-scope", lambda value: value["execution_scope"].__setitem__("scope_identity_sha256", "0" * 64), True),
            ("receipt-rehashed-gate", lambda value: value["post_removal_gates"][0].__setitem__("execution_gate_status", "failed"), True),
        ]
        for name, mutate, rehash in receipt_cases:
            bad_receipt = clone_receipt(receipt_a, work / f"{name}.json", mutate, rehash=rehash)
            assert_refusal(
                execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, bad_receipt),
                None,
                "E-SRB-EXEC-009",
            )

        malformed = write(work / "malformed-execution.json", "not-json\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, malformed),
            None,
            "E-SRB-EXEC-009",
        )

        extra = write(source / "unexpected-after.txt", "unexpected\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a),
            None,
            "E-SRB-EXEC-009",
        )
        extra.unlink()
        repaired = source / "consumer" / "config.txt"
        repaired_original = repaired.read_bytes()
        repaired.write_bytes(repaired_original + b"changed\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a),
            None,
            "E-SRB-EXEC-009",
        )
        repaired.write_bytes(repaired_original)
        reintroduced = source / "research"
        write(reintroduced / "study.sio", "reintroduced\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a),
            None,
            "E-SRB-EXEC-009",
        )
        shutil.rmtree(reintroduced)

        first_destination = json.loads(artifacts["destination_policy"].read_text(encoding="ascii"))["destinations"][0]
        destination_file = artifacts["destinations"] / first_destination["destination_key"] / first_destination["content_path"] / "README.md"
        destination_original = destination_file.read_bytes()
        destination_file.write_bytes(destination_original + b"changed\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a),
            None,
            "E-SRB-EXEC-002",
        )
        destination_file.write_bytes(destination_original)

        approval_file = source / "execution-approval" / "operator.txt"
        approval_original = approval_file.read_bytes()
        approval_file.write_bytes(approval_original + b"changed\n")
        assert_refusal(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a),
            None,
            "E-SRB-EXEC-003",
        )
        approval_file.write_bytes(approval_original)

        run(
            execution_command("verify", source, artifacts, execution_policy, execution_policy_payload, workspace_verify, receipt_a)
        )
        check(snapshot(source) == snapshot(source_copy), "deterministic executed roots diverged")
        check(snapshot(pristine) == source_before, "canonical fixture template was changed")

    print(
        "PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_WITNESS "
        f"execution_identity={receipt_payload['execution_identity_sha256']} "
        f"policy_identity={execution_policy_payload['policy_identity_sha256']} "
        f"authorization_identity={receipt_payload['source_bindings']['authorization_identity_sha256']} "
        f"units={receipt_payload['summary']['executed_unit_count']} "
        f"files={receipt_payload['summary']['removed_file_count']} "
        f"status={receipt_payload['execution_status']}"
    )
    print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_EXECUTION_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
