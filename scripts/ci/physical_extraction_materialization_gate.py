#!/usr/bin/env python3
"""Adversarial acceptance gate for R3 physical-extraction materialization."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_TOOL = ROOT / "tools" / "science_boundary" / "physical_extraction_inventory.py"
MATERIALIZER = ROOT / "tools" / "science_boundary" / "physical_extraction_materializer.py"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-destination-policy.v1.schema.json"
RECEIPT_SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-materialization.v1.schema.json"
SCIENCE_FIELDS = [
    "path",
    "ring",
    "evidence_status",
    "context_of_use",
    "visibility",
    "enforcement",
    "next_gate",
    "allowed_claim_classes",
    "evidence_refs",
    "declared_by",
    "declared_at",
    "review_state",
]
OWNERSHIP_FIELDS = [
    "source_path",
    "ring",
    "current_owner",
    "target_kind",
    "target_id",
    "target_owner",
    "disposition",
    "migration_state",
    "ownership_evidence",
    "extraction_gate",
]
POLICY_LIMITATIONS = [
    "does_not_create_destination_containers",
    "does_not_transfer_ownership_or_maintainership",
    "does_not_authorize_source_removal",
    "does_not_assert_remote_repository_state",
    "does_not_assert_publication_or_registry_status",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
]
RECEIPT_LIMITATIONS = [
    "does_not_delete_source_files",
    "does_not_transfer_ownership_or_maintainership",
    "does_not_assert_remote_repository_state",
    "does_not_assert_publication_or_registry_status",
    "does_not_preserve_uninventoried_filesystem_metadata",
    "does_not_guarantee_crash_atomicity_across_multiple_destinations",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "full_verification_requires_original_sources_inventory_policy_and_destinations",
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


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def write_json(path: Path, payload: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json_bytes(payload))
    return path


def canonical_identity(payload: dict[str, object], field: str) -> str:
    value = json.loads(json.dumps(payload))
    value.pop(field, None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def file_binding(path: Path, repo: Path) -> dict[str, object]:
    raw = path.read_bytes()
    return {
        "path": path.relative_to(repo).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def write_tsv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> Path:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=fields, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return write(path, buffer.getvalue())


def science_row(path: str, ring: str) -> dict[str, str]:
    candidate = ring not in {"pl-core", "scientific-package", "research"}
    return {
        "path": path,
        "ring": ring,
        "evidence_status": "implemented" if candidate else "passes-gate",
        "context_of_use": f"materialization fixture {path}",
        "visibility": "protected" if candidate else "public",
        "enforcement": "advisory",
        "next_gate": "classification-gate" if candidate else "package-boundary-receipt",
        "allowed_claim_classes": "" if candidate else "compile|runtime",
        "evidence_refs": "review:pending" if candidate else "gate:materialization-fixture",
        "declared_by": "SOUNIO-SCIENCE-RESEARCH-BOUNDARY",
        "declared_at": "2026-07-17",
        "review_state": "draft",
    }


def ownership_row(path: str, ring: str, suffix: str) -> dict[str, str]:
    if ring == "pl-core":
        values = {
            "current_owner": "core-maintainers",
            "target_kind": "same-repository",
            "target_id": "repo:sounio",
            "target_owner": "core-maintainers",
            "disposition": "retain-core",
            "migration_state": "retained",
            "extraction_gate": "none-retained",
        }
    elif ring in {"scientific-package", "research"}:
        values = {
            "current_owner": "monorepo-maintainers",
            "target_kind": "separate-distribution",
            "target_id": f"distribution:{suffix}",
            "target_owner": "future-maintainers",
            "disposition": "extract-planned",
            "migration_state": "planned",
            "extraction_gate": "r3-materialization",
        }
    else:
        values = {
            "current_owner": "core-maintainers",
            "target_kind": "unassigned",
            "target_id": "unassigned",
            "target_owner": "unassigned",
            "disposition": "hold-unresolved",
            "migration_state": "blocked-classification",
            "extraction_gate": "classification-gate",
        }
    return {
        "source_path": path,
        "ring": ring,
        **values,
        "ownership_evidence": "science-rings.tsv|gate:materialization-fixture",
    }


def create_fixture(repo: Path) -> tuple[Path, Path]:
    write(repo / "core" / "compiler.sio", "fn core() -> i64 { 1 }\n")
    write(repo / "packages" / "pkg" / "src" / "lib.sio", "fn package() -> i64 { 2 }\n")
    write(repo / "packages" / "pkg" / "README.md", "package fixture\n")
    write(repo / "research" / "study.sio", "fn study() -> i64 { 3 }\n")
    write(repo / "stdlib" / "candidate.sio", "fn candidate() -> i64 { 4 }\n")
    write(repo / "approvals" / "pkg.txt", "approved package destination\n")
    write(repo / "approvals" / "research.txt", "approved research destination\n")
    definitions = [
        ("core", "pl-core", "core"),
        ("packages/pkg", "scientific-package", "pkg"),
        ("research", "research", "research"),
        ("stdlib", "scientific-package-candidate", "stdlib"),
    ]
    rings = write_tsv(
        repo / "science-rings.tsv",
        SCIENCE_FIELDS,
        [science_row(path, ring) for path, ring, _suffix in definitions],
    )
    ownership = write_tsv(
        repo / "ownership.tsv",
        OWNERSHIP_FIELDS,
        [ownership_row(path, ring, suffix) for path, ring, suffix in definitions],
    )
    return rings, ownership


def inventory_command(repo: Path, rings: Path, ownership: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(INVENTORY_TOOL),
        "inventory",
        "--repo-root",
        str(repo),
        "--rings",
        str(rings),
        "--ownership",
        str(ownership),
        "--output",
        str(output),
    ]


def materialize_command(
    repo: Path,
    rings: Path,
    ownership: Path,
    inventory: Path,
    policy: Path,
    destinations: Path,
    receipt: Path,
) -> list[str]:
    return [
        sys.executable,
        str(MATERIALIZER),
        "materialize",
        "--repo-root",
        str(repo),
        "--rings",
        str(rings),
        "--ownership",
        str(ownership),
        "--inventory",
        str(inventory),
        "--destination-policy",
        str(policy),
        "--destinations-root",
        str(destinations),
        "--receipt",
        str(receipt),
    ]


def verify_command(
    repo: Path,
    rings: Path,
    ownership: Path,
    inventory: Path,
    policy: Path,
    destinations: Path,
    receipt: Path,
) -> list[str]:
    command = materialize_command(repo, rings, ownership, inventory, policy, destinations, receipt)
    command[2] = "verify"
    return command


def marker_payload(row: dict[str, object], inventory: dict[str, object]) -> dict[str, object]:
    return {
        "schema": "sounio.physical-extraction-destination.v1",
        "marker_type": "preexisting-approved-destination",
        "target_id": row["target_id"],
        "target_owner": row["target_owner"],
        "destination_key": row["destination_key"],
        "content_path": row["content_path"],
        "approval_state": "approved",
        "source_inventory_identity_sha256": inventory["inventory_identity_sha256"],
    }


def create_policy(repo: Path, inventory_path: Path, output: Path, *, approval_state: str = "approved") -> dict[str, object]:
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    destinations: list[dict[str, object]] = []
    evidence = {
        "distribution:pkg": file_binding(repo / "approvals" / "pkg.txt", repo),
        "distribution:research": file_binding(repo / "approvals" / "research.txt", repo),
    }
    for unit in sorted(
        (item for item in inventory["units"] if item["disposition"] == "extract-planned"),
        key=lambda item: item["target_id"],
    ):
        suffix = unit["target_id"].split(":", 1)[1]
        row: dict[str, object] = {
            "target_id": unit["target_id"],
            "target_kind": unit["target_kind"],
            "target_owner": unit["target_owner"],
            "destination_key": f"{suffix}-destination",
            "content_path": "payload",
            "approval_state": approval_state,
            "approved_by": "fixture-author",
            "approval_evidence": [evidence[unit["target_id"]]],
        }
        row["destination_marker_sha256"] = hashlib.sha256(json_bytes(marker_payload(row, inventory))).hexdigest()
        destinations.append(row)
    payload: dict[str, object] = {
        "schema": "sounio.physical-extraction-destination-policy.v1",
        "policy_type": "local-destination-approval-policy",
        "authority_scope": "explicit-local-copy-destination-approval",
        "source_inventory_identity_sha256": inventory["inventory_identity_sha256"],
        "approval_status": "approved" if approval_state == "approved" else "pending",
        "destinations": destinations,
        "limitations": POLICY_LIMITATIONS,
    }
    payload["policy_identity_sha256"] = canonical_identity(payload, "policy_identity_sha256")
    write_json(output, payload)
    return payload


def create_destinations(root: Path, inventory_path: Path, policy: dict[str, object]) -> None:
    inventory = json.loads(inventory_path.read_text(encoding="ascii"))
    root.mkdir(parents=True)
    for row in policy["destinations"]:
        container = root / row["destination_key"]
        container.mkdir()
        marker = marker_payload(row, inventory)
        marker_path = write_json(container / ".sounio-destination-approval.json", marker)
        check(hashlib.sha256(marker_path.read_bytes()).hexdigest() == row["destination_marker_sha256"], "marker fixture hash drifted")


def clone_policy(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["policy_identity_sha256"] = canonical_identity(payload, "policy_identity_sha256")
    return write_json(destination, payload)


def clone_inventory(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["inventory_identity_sha256"] = canonical_identity(payload, "inventory_identity_sha256")
    return write_json(destination, payload)


def clone_receipt(original: Path, destination: Path, mutate, *, rehash: bool = True) -> Path:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["materialization_identity_sha256"] = canonical_identity(payload, "materialization_identity_sha256")
    return write_json(destination, payload)


def assert_refusal(command: list[str], receipt: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if receipt is not None:
        check(not receipt.exists(), f"refused materialization left receipt {receipt}")
    check(code in result.stderr, f"refusal lacks {code}: {result.stderr}")
    check("PHYSICAL_EXTRACTION_MATERIALIZATION_REFUSED" in result.stderr, "refusal lacks materialization marker")
    return result


def check_no_payloads(root: Path, policy: dict[str, object]) -> None:
    for row in policy["destinations"]:
        check(not (root / row["destination_key"] / row["content_path"]).exists(), f"refusal promoted {row['target_id']}")


def assert_static_contracts() -> None:
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    receipt_schema = json.loads(RECEIPT_SCHEMA.read_text(encoding="utf-8"))
    check(policy_schema["properties"]["schema"]["const"] == "sounio.physical-extraction-destination-policy.v1", "bad policy schema")
    check(policy_schema["properties"]["authority_scope"]["const"] == "explicit-local-copy-destination-approval", "policy authority drifted")
    check(policy_schema["properties"]["limitations"]["const"] == POLICY_LIMITATIONS, "policy limitations drifted")
    check(receipt_schema["properties"]["schema"]["const"] == "sounio.physical-extraction-materialization.v1", "bad receipt schema")
    check(receipt_schema["properties"]["materialization_status"]["const"] == "copied-and-verified", "receipt status drifted")
    check(receipt_schema["properties"]["source_removal_status"]["const"] == "not-authorized", "receipt authorizes deletion")
    check(receipt_schema["properties"]["assurance_level"]["const"] == "identity-only", "receipt overstates assurance")
    check(receipt_schema["properties"]["limitations"]["const"] == RECEIPT_LIMITATIONS, "receipt limitations drifted")
    check("does_not_assert_remote_repository_state" in RECEIPT_LIMITATIONS, "remote-state limitation is absent")
    check("does_not_delete_source_files" in RECEIPT_LIMITATIONS, "source-preservation limitation is absent")
    help_text = run([sys.executable, str(MATERIALIZER), "--help"]).stdout
    check("materialize" in help_text and "verify" in help_text, "materializer lacks both commands")


def main() -> int:
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-r3-materialization-") as temporary:
        work = Path(temporary)
        repo = work / "source-repo"
        rings, ownership = create_fixture(repo)
        inventory = work / "inventory.json"
        inventory_result = run(inventory_command(repo, rings, ownership, inventory))
        check("PHYSICAL_EXTRACTION_INVENTORY_PASS" in inventory_result.stdout, "fixture inventory did not pass")
        inventory_payload = json.loads(inventory.read_text(encoding="ascii"))
        check(inventory_payload["summary"]["planned_extraction_units"] == 2, "fixture lacks two planned units")
        check(inventory_payload["summary"]["retained_core_units"] == 1, "fixture lacks retained core")
        check(inventory_payload["summary"]["blocked_units"] == 1, "fixture lacks blocked unit")

        policy = work / "policy.json"
        policy_payload = create_policy(repo, inventory, policy)
        check(policy_payload["approval_status"] == "approved", "fixture policy is not approved")
        check(len(policy_payload["destinations"]) == 2, "fixture policy coverage is wrong")

        destinations_a = work / "destinations-a"
        create_destinations(destinations_a, inventory, policy_payload)
        source_before = {item["path"]: item["sha256"] for unit in inventory_payload["units"] for item in unit["files"]}
        receipt_a = work / "materialization-a.json"
        materialized = run(materialize_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a))
        check("PHYSICAL_EXTRACTION_MATERIALIZATION_PASS" in materialized.stdout, "materialization pass marker is absent")
        check("source_removal=not-authorized" in materialized.stdout, "materialization output overstates removal authority")
        check(receipt_a.is_file(), "materialization receipt is absent")
        receipt_payload = json.loads(receipt_a.read_text(encoding="ascii"))
        check(receipt_payload["materialization_status"] == "copied-and-verified", "receipt status is wrong")
        check(receipt_payload["source_removal_status"] == "not-authorized", "receipt authorizes source removal")
        check(receipt_payload["assurance_level"] == "identity-only", "receipt assurance is wrong")
        check(receipt_payload["summary"]["materialized_unit_count"] == 2, "materialized unit count is wrong")
        check(receipt_payload["summary"]["file_count"] == 3, "materialized file count is wrong")
        check(receipt_payload["summary"]["retained_source_units"] == 1, "retained source summary is wrong")
        check(receipt_payload["summary"]["blocked_source_units"] == 1, "blocked source summary is wrong")
        check(receipt_payload["limitations"] == RECEIPT_LIMITATIONS, "receipt limitations do not match contract")
        check(receipt_payload["materialization_identity_sha256"] == canonical_identity(receipt_payload, "materialization_identity_sha256"), "receipt identity is wrong")
        check(all(unit["copy_status"] == "copied-and-verified" for unit in receipt_payload["units"]), "unit copy status is wrong")
        check(all(unit["source_tree_sha256"] != unit["destination_tree_sha256"] for unit in receipt_payload["units"]), "source and relative destination tree domains were conflated")
        for row in policy_payload["destinations"]:
            check((destinations_a / row["destination_key"] / row["content_path"]).is_dir(), f"destination payload missing for {row['target_id']}")
        source_after = {}
        for unit in inventory_payload["units"]:
            for item in unit["files"]:
                raw = (repo / item["path"]).read_bytes()
                source_after[item["path"]] = hashlib.sha256(raw).hexdigest()
        check(source_after == source_before, "materialization changed source content")
        verified = run(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a))
        check("PHYSICAL_EXTRACTION_MATERIALIZATION_VERIFY_PASS" in verified.stdout, "materialization did not verify")

        destinations_b = work / "destinations-b"
        create_destinations(destinations_b, inventory, policy_payload)
        receipt_b = work / "materialization-b.json"
        run(materialize_command(repo, rings, ownership, inventory, policy, destinations_b, receipt_b))
        check(receipt_a.read_bytes() == receipt_b.read_bytes(), "receipt depends on physical destination root")
        run(verify_command(repo, rings, ownership, inventory, policy, destinations_b, receipt_b))

        occupied_root = work / "occupied-root"
        create_destinations(occupied_root, inventory, policy_payload)
        occupied_receipt = write(work / "occupied-receipt.json", "preserve\n")
        occupied_result = run(
            materialize_command(repo, rings, ownership, inventory, policy, occupied_root, occupied_receipt),
            expected=1,
        )
        check("E-SRB-MATERIALIZE-005" in occupied_result.stderr, "occupied receipt refusal lacks diagnostic")
        check(occupied_receipt.read_text(encoding="utf-8") == "preserve\n", "occupied receipt was overwritten")
        check_no_payloads(occupied_root, policy_payload)

        content_root = work / "occupied-content-root"
        create_destinations(content_root, inventory, policy_payload)
        first_row = policy_payload["destinations"][0]
        occupied_content = content_root / first_row["destination_key"] / first_row["content_path"]
        write(occupied_content / "preserve.txt", "preserve\n")
        content_receipt = work / "occupied-content-receipt.json"
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, content_root, content_receipt),
            content_receipt,
            "E-SRB-MATERIALIZE-005",
        )
        check((occupied_content / "preserve.txt").read_text(encoding="utf-8") == "preserve\n", "occupied destination was changed")

        pending_policy = work / "pending-policy.json"
        pending_payload = create_policy(repo, inventory, pending_policy, approval_state="pending")
        pending_root = work / "pending-root"
        create_destinations(pending_root, inventory, pending_payload)
        pending_receipt = work / "pending-receipt.json"
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, pending_policy, pending_root, pending_receipt),
            pending_receipt,
            "E-SRB-MATERIALIZE-002",
        )
        check_no_payloads(pending_root, pending_payload)

        missing_root = work / "missing-root"
        create_destinations(missing_root, inventory, policy_payload)
        shutil.rmtree(missing_root / policy_payload["destinations"][1]["destination_key"])
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, missing_root, work / "missing.json"),
            work / "missing.json",
            "E-SRB-MATERIALIZE-003",
        )

        marker_root = work / "marker-root"
        create_destinations(marker_root, inventory, policy_payload)
        marker_path = marker_root / first_row["destination_key"] / ".sounio-destination-approval.json"
        marker_path.unlink()
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, marker_root, work / "marker.json"),
            work / "marker.json",
            "E-SRB-MATERIALIZE-003",
        )

        symlink_root = work / "symlink-root"
        create_destinations(symlink_root, inventory, policy_payload)
        symlink_target = symlink_root / first_row["destination_key"]
        shutil.rmtree(symlink_target)
        symlink_target.symlink_to(destinations_a / first_row["destination_key"], target_is_directory=True)
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, symlink_root, work / "symlink.json"),
            work / "symlink.json",
            "E-SRB-MATERIALIZE-003",
        )

        inside_root = repo / "destinations-inside-source"
        create_destinations(inside_root, inventory, policy_payload)
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, inside_root, work / "inside.json"),
            work / "inside.json",
            "E-SRB-MATERIALIZE-003",
        )

        policy_cases = [
            (
                "policy-identity",
                lambda value: value["destinations"][0].__setitem__("approved_by", "other-author"),
                False,
                "E-SRB-MATERIALIZE-001",
            ),
            (
                "policy-owner",
                lambda value: value["destinations"][0].__setitem__("target_owner", "wrong-owner"),
                True,
                "E-SRB-MATERIALIZE-002",
            ),
            (
                "policy-missing",
                lambda value: value["destinations"].pop(),
                True,
                "E-SRB-MATERIALIZE-002",
            ),
            (
                "policy-duplicate-key",
                lambda value: value["destinations"][1].__setitem__("destination_key", value["destinations"][0]["destination_key"]),
                True,
                "E-SRB-MATERIALIZE-001",
            ),
            (
                "policy-inventory",
                lambda value: value.__setitem__("source_inventory_identity_sha256", "0" * 64),
                True,
                "E-SRB-MATERIALIZE-002",
            ),
            (
                "policy-aggregate",
                lambda value: value.__setitem__("approval_status", "pending"),
                True,
                "E-SRB-MATERIALIZE-001",
            ),
        ]
        for name, mutate, rehash, code in policy_cases:
            bad_policy = clone_policy(policy, work / f"{name}.json", mutate, rehash=rehash)
            bad_root = work / f"{name}-root"
            create_destinations(bad_root, inventory, policy_payload)
            bad_receipt = work / f"{name}-receipt.json"
            assert_refusal(
                materialize_command(repo, rings, ownership, inventory, bad_policy, bad_root, bad_receipt),
                bad_receipt,
                code,
            )
            check_no_payloads(bad_root, policy_payload)

        evidence_path = repo / "approvals" / "pkg.txt"
        evidence_original = evidence_path.read_bytes()
        evidence_path.write_bytes(evidence_original + b"changed\n")
        evidence_root = work / "evidence-root"
        create_destinations(evidence_root, inventory, policy_payload)
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, evidence_root, work / "evidence.json"),
            work / "evidence.json",
            "E-SRB-MATERIALIZE-002",
        )
        check_no_payloads(evidence_root, policy_payload)
        evidence_path.write_bytes(evidence_original)

        forged_inventory = clone_inventory(
            inventory,
            work / "forged-inventory.json",
            lambda value: value["units"][0].__setitem__("context_of_use", "forged context"),
        )
        forged_root = work / "forged-inventory-root"
        create_destinations(forged_root, inventory, policy_payload)
        assert_refusal(
            materialize_command(repo, rings, ownership, forged_inventory, policy, forged_root, work / "forged-inventory-receipt.json"),
            work / "forged-inventory-receipt.json",
            "E-SRB-MATERIALIZE-004",
        )
        check_no_payloads(forged_root, policy_payload)

        source_file = repo / "packages" / "pkg" / "README.md"
        source_original = source_file.read_bytes()
        source_file.write_bytes(source_original + b"changed\n")
        source_root = work / "source-mutation-root"
        create_destinations(source_root, inventory, policy_payload)
        assert_refusal(
            materialize_command(repo, rings, ownership, inventory, policy, source_root, work / "source-mutation.json"),
            work / "source-mutation.json",
            "E-SRB-MATERIALIZE-004",
        )
        check_no_payloads(source_root, policy_payload)
        source_file.write_bytes(source_original)

        verify_file = destinations_a / first_row["destination_key"] / first_row["content_path"] / "README.md"
        verify_original = verify_file.read_bytes()
        verify_file.write_bytes(verify_original + b"tamper\n")
        assert_refusal(
            verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a),
            None,
            "E-SRB-MATERIALIZE-004",
        )
        verify_file.write_bytes(verify_original)
        run(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a))

        extra_file = write(verify_file.parent / "extra.sio", "extra\n")
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-004")
        extra_file.unlink()
        extra_dir = verify_file.parent / "empty-extra"
        extra_dir.mkdir()
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-004")
        extra_dir.rmdir()
        symlink_file = verify_file.parent / "symlink-extra"
        symlink_file.symlink_to(verify_file.name)
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-004")
        symlink_file.unlink()

        verify_file.unlink()
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-004")
        verify_file.write_bytes(verify_original)

        source_file.write_bytes(source_original + b"post-copy mutation\n")
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-004")
        source_file.write_bytes(source_original)

        marker_original = (destinations_a / first_row["destination_key"] / ".sounio-destination-approval.json").read_bytes()
        marker_file = destinations_a / first_row["destination_key"] / ".sounio-destination-approval.json"
        marker_file.write_bytes(marker_original + b" ")
        assert_refusal(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a), None, "E-SRB-MATERIALIZE-003")
        marker_file.write_bytes(marker_original)

        receipt_cases = [
            (
                "receipt-unhashed",
                lambda value: value.__setitem__("source_removal_status", "authorized"),
                False,
            ),
            (
                "receipt-rehashed-status",
                lambda value: value.__setitem__("source_removal_status", "authorized"),
                True,
            ),
            (
                "receipt-rehashed-file",
                lambda value: value["units"][0]["files"][0].__setitem__("sha256", "0" * 64),
                True,
            ),
            (
                "receipt-rehashed-marker",
                lambda value: value["units"][0].__setitem__("destination_marker_sha256", "0" * 64),
                True,
            ),
        ]
        for name, mutate, rehash in receipt_cases:
            bad_receipt = clone_receipt(receipt_a, work / f"{name}.json", mutate, rehash=rehash)
            assert_refusal(
                verify_command(repo, rings, ownership, inventory, policy, destinations_a, bad_receipt),
                None,
                "E-SRB-MATERIALIZE-006",
            )

        malformed_receipt = write(work / "malformed-receipt.json", "not-json\n")
        assert_refusal(
            verify_command(repo, rings, ownership, inventory, policy, destinations_a, malformed_receipt),
            None,
            "E-SRB-MATERIALIZE-006",
        )

        source_final = {}
        for unit in inventory_payload["units"]:
            for item in unit["files"]:
                raw = (repo / item["path"]).read_bytes()
                source_final[item["path"]] = hashlib.sha256(raw).hexdigest()
        check(source_final == source_before, "adversarial gate did not restore source fixture")
        run(verify_command(repo, rings, ownership, inventory, policy, destinations_a, receipt_a))

    print(
        "PHYSICAL_EXTRACTION_MATERIALIZATION_WITNESS "
        f"receipt_identity={receipt_payload['materialization_identity_sha256']} "
        f"policy_identity={policy_payload['policy_identity_sha256']} "
        f"units={receipt_payload['summary']['materialized_unit_count']} "
        f"files={receipt_payload['summary']['file_count']} "
        f"bytes={receipt_payload['summary']['total_bytes']} "
        f"status={receipt_payload['materialization_status']} "
        f"source_removal={receipt_payload['source_removal_status']}"
    )
    print(f"PHYSICAL_EXTRACTION_MATERIALIZATION_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
