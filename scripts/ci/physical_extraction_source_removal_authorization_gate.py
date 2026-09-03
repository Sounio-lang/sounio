#!/usr/bin/env python3
"""Adversarial acceptance gate for R3 source-removal authorization."""

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
import physical_extraction_materialization_gate as material_gate  # noqa: E402


AUTHORIZER = ROOT / "tools" / "science_boundary" / "source_removal_authorizer.py"
COMPOSED_GATE = ROOT / "scripts" / "ci" / "physical_extraction_source_removal_authorization_gate.sh"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-source-removal-policy.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-source-removal-authorization.v1.schema.json"
POLICY_LIMITATIONS = [
    "does_not_delete_original_source_files",
    "does_not_authorize_unlisted_source_paths",
    "does_not_execute_canonical_repository_migration",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "distinct_reviewer_labels_do_not_prove_organizational_independence",
    "post_removal_environment_is_not_fully_captured",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_delete_original_source_files",
    "authorizes_only_the_bound_scope_for_a_separate_execution_interface",
    "does_not_execute_canonical_repository_migration",
    "does_not_assert_remote_repository_or_publication_state",
    "does_not_transfer_ownership_or_maintainership",
    "distinct_reviewer_labels_do_not_prove_organizational_independence",
    "post_removal_environment_is_not_fully_captured",
    "does_not_preserve_uninventoried_filesystem_metadata_in_candidate_copy",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "full_verification_requires_original_sources_materialization_destinations_policy_and_gates",
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


def source_snapshot(repo: Path) -> dict[str, str]:
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
            check(not path.is_symlink() and path.is_file(), f"fixture source member is unsafe: {path}")
            result[path.relative_to(repo).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def scope_from_inventory(inventory: dict[str, object]) -> dict[str, object]:
    units = []
    for unit in inventory["units"]:
        if unit["disposition"] != "extract-planned":
            continue
        units.append(
            {
                "source_path": unit["source_path"],
                "ring": unit["ring"],
                "target_id": unit["target_id"],
                "target_owner": unit["target_owner"],
                "file_count": unit["file_count"],
                "total_bytes": unit["total_bytes"],
                "tree_sha256": unit["tree_sha256"],
            }
        )
    units.sort(key=lambda item: item["source_path"])
    encoded = json.dumps(units, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return {"scope_identity_sha256": hashlib.sha256(encoded).hexdigest(), "units": units}


def create_removal_policy(
    repo: Path,
    inventory_path: Path,
    materialization_path: Path,
    output: Path,
) -> dict[str, object]:
    inventory_raw = inventory_path.read_bytes()
    inventory = json.loads(inventory_raw)
    materialization_raw = materialization_path.read_bytes()
    materialization = json.loads(materialization_raw)
    before = binding(repo / "consumer" / "config.txt", repo)
    replacement = binding(repo / "repairs" / "config.after.txt", repo)
    gate_stdout = b"POST_REMOVAL_GATE_PASS\n"
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-source-removal-policy.v1",
        "policy_type": "reviewed-post-removal-candidate-policy",
        "authority_scope": "temporary-copy-source-removal-candidate-approval",
        "source_bindings": {
            "inventory_file_sha256": hashlib.sha256(inventory_raw).hexdigest(),
            "inventory_identity_sha256": inventory["inventory_identity_sha256"],
            "materialization_file_sha256": hashlib.sha256(materialization_raw).hexdigest(),
            "materialization_identity_sha256": materialization["materialization_identity_sha256"],
        },
        "approval_status": "approved",
        "removal_scope": scope_from_inventory(inventory),
        "review_evidence": [
            {"reviewer_label": "reviewer-alpha", **binding(repo / "reviews" / "alpha.txt", repo)},
            {"reviewer_label": "reviewer-beta", **binding(repo / "reviews" / "beta.txt", repo)},
        ],
        "repairs": [
            {
                "path": before["path"],
                "before_size_bytes": before["size_bytes"],
                "before_sha256": before["sha256"],
                "replacement_path": replacement["path"],
                "replacement_size_bytes": replacement["size_bytes"],
                "replacement_sha256": replacement["sha256"],
                "after_size_bytes": replacement["size_bytes"],
                "after_sha256": replacement["sha256"],
            }
        ],
        "post_removal_gates": [
            {
                "gate_id": "fixture-post-removal",
                "argv": [sys.executable, "post_removal_gate.py"],
                "cwd": ".",
                "timeout_seconds": 30,
                "expected_exit_code": 0,
                "expected_stdout_sha256": hashlib.sha256(gate_stdout).hexdigest(),
                "expected_stderr_sha256": hashlib.sha256(b"").hexdigest(),
            }
        ],
        "limitations": POLICY_LIMITATIONS,
    }
    payload["policy_identity_sha256"] = identity(payload, "policy_identity_sha256")
    write_json(output, payload)
    return payload


def authorization_command(
    mode: str,
    repo: Path,
    rings: Path,
    ownership: Path,
    inventory: Path,
    destination_policy: Path,
    destinations: Path,
    materialization: Path,
    removal_policy: Path,
    workspace: Path,
    receipt: Path,
) -> list[str]:
    return [
        sys.executable,
        str(AUTHORIZER),
        mode,
        "--repo-root",
        str(repo),
        "--rings",
        str(rings),
        "--ownership",
        str(ownership),
        "--inventory",
        str(inventory),
        "--destination-policy",
        str(destination_policy),
        "--destinations-root",
        str(destinations),
        "--materialization-receipt",
        str(materialization),
        "--removal-policy",
        str(removal_policy),
        "--workspace-root",
        str(workspace),
        "--authorization-receipt",
        str(receipt),
    ]


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
        payload["authorization_identity_sha256"] = identity(payload, "authorization_identity_sha256")
    return write_json(destination, payload)


def assert_refusal(command: list[str], receipt: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if receipt is not None:
        check(not receipt.exists(), f"refused authorization left receipt {receipt}")
    check(code in result.stderr, f"refusal lacks {code}: {result.stderr}")
    check(
        "PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_REFUSED" in result.stderr,
        "refusal lacks source-removal authorization marker",
    )
    return result


def assert_static_contracts() -> None:
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(policy_schema["properties"]["schema"]["const"] == "sounio.physical-extraction-source-removal-policy.v1", "bad policy schema")
    check(policy_schema["properties"]["approval_status"]["const"] == "approved", "policy permits pending approval")
    check(policy_schema["properties"]["review_evidence"]["minItems"] == 2, "policy permits one review")
    check(policy_schema["properties"]["repairs"]["minItems"] == 1, "policy permits no repairs")
    check(policy_schema["properties"]["post_removal_gates"]["minItems"] == 1, "policy permits no gates")
    check(policy_schema["properties"]["limitations"]["const"] == POLICY_LIMITATIONS, "policy limitations drifted")
    check(receipt_schema["properties"]["authorization_status"]["const"] == "authorized-not-executed", "receipt overstates authorization")
    check(receipt_schema["properties"]["source_removal_execution_status"]["const"] == "not-executed", "receipt claims execution")
    check(receipt_schema["properties"]["assurance_level"]["const"] == "identity-only", "receipt assurance drifted")
    check(receipt_schema["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drifted")
    source = AUTHORIZER.read_text(encoding="utf-8")
    for token in [
        "authorized-not-executed",
        "not-executed",
        "run_simulation",
        "load_materialization",
        "scan_repository",
        "distinct_reviewer_labels_do_not_prove_organizational_independence",
    ]:
        check(token in source, f"authorizer lacks contract token {token}")
    check("shutil.rmtree(repo_root" not in source, "authorizer contains direct source-root removal")
    check("os.unlink(repo_root" not in source, "authorizer contains direct source-root unlink")
    composed = COMPOSED_GATE.read_text(encoding="utf-8")
    check(
        'export SOUNIO_PHYSICAL_EXTRACTION_MATERIALIZATION_MADAROS_BIN="$SOUNIO_PHYSICAL_EXTRACTION_SOURCE_REMOVAL_MADAROS_BIN"'
        in composed,
        "composed gate does not forward the current-source Madaros to materialization",
    )


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-source-removal-auth-gate.") as temporary:
        work = Path(temporary)
        repo = work / "source-repo"
        repo.mkdir()
        rings, ownership = material_gate.create_fixture(repo)
        write(repo / "consumer" / "config.txt", "imports=packages/pkg,research\n")
        write(repo / "repairs" / "config.after.txt", "imports=external:pkg,external:research\n")
        write(repo / "reviews" / "alpha.txt", "alpha reviewed exact removal scope\n")
        write(repo / "reviews" / "beta.txt", "beta reviewed exact removal scope\n")
        write(
            repo / "post_removal_gate.py",
            """from pathlib import Path
root = Path.cwd()
assert not (root / "packages" / "pkg").exists()
assert not (root / "research").exists()
assert (root / "core" / "compiler.sio").is_file()
assert (root / "stdlib" / "candidate.sio").is_file()
assert (root / "consumer" / "config.txt").read_text() == "imports=external:pkg,external:research\\n"
print("POST_REMOVAL_GATE_PASS")
""",
        )

        inventory = work / "inventory.json"
        run(material_gate.inventory_command(repo, rings, ownership, inventory))
        inventory_payload = json.loads(inventory.read_text(encoding="ascii"))
        destination_policy = work / "destination-policy.json"
        destination_policy_payload = material_gate.create_policy(repo, inventory, destination_policy)
        destinations = work / "destinations"
        material_gate.create_destinations(destinations, inventory, destination_policy_payload)
        materialization = work / "materialization.json"
        run(material_gate.materialize_command(repo, rings, ownership, inventory, destination_policy, destinations, materialization))
        run(material_gate.verify_command(repo, rings, ownership, inventory, destination_policy, destinations, materialization))

        removal_policy = work / "removal-policy.json"
        removal_policy_payload = create_removal_policy(repo, inventory, materialization, removal_policy)
        workspace_a = work / "workspace-a"
        workspace_b = work / "workspace-b"
        workspace_c = work / "workspace-c"
        for workspace in (workspace_a, workspace_b, workspace_c):
            workspace.mkdir()
        receipt_a = work / "authorization-a.json"
        receipt_b = work / "authorization-b.json"
        source_before = source_snapshot(repo)

        authorized_a = run(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, receipt_a,
            )
        )
        check("status=authorized-not-executed" in authorized_a.stdout, "authorization output overstates status")
        check("execution=not-executed" in authorized_a.stdout, "authorization output claims execution")
        check(receipt_a.is_file(), "authorization receipt was not emitted")
        receipt_payload = json.loads(receipt_a.read_text(encoding="ascii"))
        check(receipt_payload["authorization_identity_sha256"] == identity(receipt_payload, "authorization_identity_sha256"), "receipt identity mismatch")
        check(receipt_payload["authorization_status"] == "authorized-not-executed", "receipt status mismatch")
        check(receipt_payload["source_removal_execution_status"] == "not-executed", "receipt execution mismatch")
        check(receipt_payload["removal_scope"] == removal_policy_payload["removal_scope"], "receipt scope mismatch")
        check(receipt_payload["summary"]["authorized_unit_count"] == 2, "wrong authorized unit count")
        check(receipt_payload["summary"]["authorized_file_count"] == 3, "wrong authorized file count")
        check(receipt_payload["summary"]["review_evidence_count"] == 2, "wrong review count")
        check(receipt_payload["summary"]["repair_count"] == 1, "wrong repair count")
        check(receipt_payload["summary"]["post_removal_gate_count"] == 1, "wrong gate count")
        check(receipt_payload["repairs"][0]["repair_status"] == "applied-and-verified", "repair was not witnessed")
        check(receipt_payload["post_removal_gates"][0]["gate_status"] == "passed", "gate was not witnessed")
        check(receipt_payload["candidate_evidence"]["original_source_status"] == "reverified-unchanged", "source was not reverified")
        check(source_snapshot(repo) == source_before, "positive authorization changed original source")
        check(not any(path.name.startswith("sounio-source-removal-candidate.") for path in workspace_a.iterdir()), "candidate workspace leaked")

        run(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_b, receipt_b,
            )
        )
        check(receipt_a.read_bytes() == receipt_b.read_bytes(), "authorization is not deterministic across workspace roots")
        verified = run(
            authorization_command(
                "verify", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_c, receipt_a,
            )
        )
        check("AUTHORIZATION_VERIFY_PASS" in verified.stdout, "verification lacks pass marker")
        check(source_snapshot(repo) == source_before, "verification changed original source")

        policy_cases = [
            ("unhashed-policy", lambda value: value.__setitem__("approval_status", "pending"), False, "E-SRB-REMOVE-001"),
            ("pending-policy", lambda value: value.__setitem__("approval_status", "pending"), True, "E-SRB-REMOVE-002"),
            ("wrong-inventory", lambda value: value["source_bindings"].__setitem__("inventory_identity_sha256", "0" * 64), True, "E-SRB-REMOVE-002"),
            ("wrong-materialization", lambda value: value["source_bindings"].__setitem__("materialization_identity_sha256", "0" * 64), True, "E-SRB-REMOVE-002"),
            ("wrong-scope-id", lambda value: value["removal_scope"].__setitem__("scope_identity_sha256", "0" * 64), True, "E-SRB-REMOVE-002"),
            ("missing-scope-unit", lambda value: value["removal_scope"]["units"].pop(), True, "E-SRB-REMOVE-002"),
            ("one-review", lambda value: value["review_evidence"].pop(), True, "E-SRB-REMOVE-002"),
            ("duplicate-review-label", lambda value: value["review_evidence"][1].__setitem__("reviewer_label", value["review_evidence"][0]["reviewer_label"]), True, "E-SRB-REMOVE-002"),
            ("duplicate-review-path", lambda value: value["review_evidence"][1].update({key: value["review_evidence"][0][key] for key in ("path", "size_bytes", "sha256")}), True, "E-SRB-REMOVE-002"),
            ("no-repairs", lambda value: value.__setitem__("repairs", []), True, "E-SRB-REMOVE-002"),
            ("repair-in-scope", lambda value: value["repairs"][0].__setitem__("path", "packages/pkg/README.md"), True, "E-SRB-REMOVE-002"),
            ("replacement-in-scope", lambda value: value["repairs"][0].__setitem__("replacement_path", "research/study.sio"), True, "E-SRB-REMOVE-002"),
            ("self-replacement", lambda value: value["repairs"][0].__setitem__("replacement_path", value["repairs"][0]["path"]), True, "E-SRB-REMOVE-002"),
            ("wrong-before", lambda value: value["repairs"][0].__setitem__("before_sha256", "0" * 64), True, "E-SRB-REMOVE-002"),
            ("wrong-after", lambda value: value["repairs"][0].__setitem__("after_sha256", "0" * 64), True, "E-SRB-REMOVE-002"),
            ("no-gates", lambda value: value.__setitem__("post_removal_gates", []), True, "E-SRB-REMOVE-002"),
            ("wrong-gate-output", lambda value: value["post_removal_gates"][0].__setitem__("expected_stdout_sha256", "0" * 64), True, "E-SRB-REMOVE-006"),
            ("failed-gate", lambda value: value["post_removal_gates"][0].update({"argv": [sys.executable, "-c", "raise SystemExit(1)"], "expected_stdout_sha256": hashlib.sha256(b"").hexdigest()}), True, "E-SRB-REMOVE-006"),
            ("missing-gate-cwd", lambda value: value["post_removal_gates"][0].__setitem__("cwd", "missing"), True, "E-SRB-REMOVE-006"),
            ("mutating-gate", lambda value: value["post_removal_gates"][0].update({"argv": [sys.executable, "-c", "open('unexpected.txt','w').write('x')"], "expected_stdout_sha256": hashlib.sha256(b"").hexdigest()}), True, "E-SRB-REMOVE-006"),
            ("wrong-limitations", lambda value: value["limitations"].pop(), True, "E-SRB-REMOVE-001"),
            ("extra-field", lambda value: value.__setitem__("unexpected", True), True, "E-SRB-REMOVE-001"),
        ]
        for name, mutate, rehash, code in policy_cases:
            bad_policy = clone_policy(removal_policy, work / f"{name}.json", mutate, rehash=rehash)
            bad_receipt = work / f"{name}-receipt.json"
            assert_refusal(
                authorization_command(
                    "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                    materialization, bad_policy, workspace_a, bad_receipt,
                ),
                bad_receipt,
                code,
            )
            check(source_snapshot(repo) == source_before, f"{name} refusal changed original source")

        review_path = repo / "reviews" / "alpha.txt"
        review_original = review_path.read_bytes()
        review_path.write_bytes(review_original + b"changed\n")
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, work / "review-mutated-receipt.json",
            ),
            work / "review-mutated-receipt.json",
            "E-SRB-REMOVE-002",
        )
        review_path.write_bytes(review_original)

        replacement_path = repo / "repairs" / "config.after.txt"
        replacement_original = replacement_path.read_bytes()
        replacement_path.write_bytes(replacement_original + b"changed\n")
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, work / "replacement-mutated-receipt.json",
            ),
            work / "replacement-mutated-receipt.json",
            "E-SRB-REMOVE-002",
        )
        replacement_path.write_bytes(replacement_original)

        planned_path = repo / "packages" / "pkg" / "README.md"
        planned_original = planned_path.read_bytes()
        planned_path.write_bytes(planned_original + b"changed\n")
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, work / "source-mutated-receipt.json",
            ),
            work / "source-mutated-receipt.json",
            "E-SRB-REMOVE-004",
        )
        planned_path.write_bytes(planned_original)

        materialization_bad = material_gate.clone_receipt(
            materialization,
            work / "materialization-forged.json",
            lambda value: value.__setitem__("source_removal_status", "authorized"),
            rehash=True,
        )
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization_bad, removal_policy, workspace_a, work / "materialization-forged-auth.json",
            ),
            work / "materialization-forged-auth.json",
            "E-SRB-REMOVE-004",
        )

        first_destination = destination_policy_payload["destinations"][0]
        destination_file = destinations / first_destination["destination_key"] / first_destination["content_path"] / "README.md"
        destination_original = destination_file.read_bytes()
        destination_file.write_bytes(destination_original + b"changed\n")
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, work / "destination-mutated-auth.json",
            ),
            work / "destination-mutated-auth.json",
            "E-SRB-REMOVE-004",
        )
        destination_file.write_bytes(destination_original)

        occupied = work / "occupied-auth.json"
        occupied.write_text("preserve\n", encoding="utf-8")
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_a, occupied,
            ),
            None,
            "E-SRB-REMOVE-007",
        )
        check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied receipt was overwritten")

        inside_workspace = repo / "workspace-inside"
        inside_workspace.mkdir()
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, inside_workspace, work / "inside-workspace-auth.json",
            ),
            work / "inside-workspace-auth.json",
            "E-SRB-REMOVE-003",
        )
        inside_workspace.rmdir()

        symlink_workspace = work / "workspace-symlink"
        symlink_workspace.symlink_to(workspace_a, target_is_directory=True)
        assert_refusal(
            authorization_command(
                "authorize", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, symlink_workspace, work / "symlink-workspace-auth.json",
            ),
            work / "symlink-workspace-auth.json",
            "E-SRB-REMOVE-003",
        )

        receipt_cases = [
            ("receipt-unhashed", lambda value: value.__setitem__("authorization_status", "executed"), False),
            ("receipt-rehashed-status", lambda value: value.__setitem__("authorization_status", "executed"), True),
            ("receipt-rehashed-scope", lambda value: value["removal_scope"].__setitem__("scope_identity_sha256", "0" * 64), True),
            ("receipt-rehashed-gate", lambda value: value["post_removal_gates"][0].__setitem__("gate_status", "failed"), True),
        ]
        for name, mutate, rehash in receipt_cases:
            bad_receipt = clone_receipt(receipt_a, work / f"{name}.json", mutate, rehash=rehash)
            assert_refusal(
                authorization_command(
                    "verify", repo, rings, ownership, inventory, destination_policy, destinations,
                    materialization, removal_policy, workspace_c, bad_receipt,
                ),
                None,
                "E-SRB-REMOVE-008",
            )

        malformed = write(work / "malformed-authorization.json", "not-json\n")
        assert_refusal(
            authorization_command(
                "verify", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_c, malformed,
            ),
            None,
            "E-SRB-REMOVE-008",
        )

        check(source_snapshot(repo) == source_before, "adversarial gate did not restore source fixture")
        run(
            authorization_command(
                "verify", repo, rings, ownership, inventory, destination_policy, destinations,
                materialization, removal_policy, workspace_c, receipt_a,
            )
        )
        check(all((repo / unit["source_path"]).is_dir() for unit in inventory_payload["units"]), "source roots were removed")

    print(
        "PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_WITNESS "
        f"authorization_identity={receipt_payload['authorization_identity_sha256']} "
        f"policy_identity={removal_policy_payload['policy_identity_sha256']} "
        f"scope_identity={receipt_payload['removal_scope']['scope_identity_sha256']} "
        f"units={receipt_payload['summary']['authorized_unit_count']} "
        f"files={receipt_payload['summary']['authorized_file_count']} "
        f"status={receipt_payload['authorization_status']} "
        f"execution={receipt_payload['source_removal_execution_status']}"
    )
    print(f"PHYSICAL_EXTRACTION_SOURCE_REMOVAL_AUTHORIZATION_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
