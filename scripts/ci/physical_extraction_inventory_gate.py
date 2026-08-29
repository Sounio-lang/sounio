#!/usr/bin/env python3
"""Adversarial acceptance gate for the R3 physical-extraction inventory."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "science_boundary" / "physical_extraction_inventory.py"
SCHEMA = ROOT / "schemas" / "sounio.physical-extraction-inventory.v1.schema.json"
OWNERSHIP = ROOT / "docs" / "ecosystem" / "science-physical-extraction-ownership.tsv"
RINGS = ROOT / "science-rings.tsv"
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
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, expected: int | set[int] = 0) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


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
        "context_of_use": f"fixture {path}",
        "visibility": "protected" if candidate else "public",
        "enforcement": "advisory",
        "next_gate": "classification-gate" if candidate else "package-boundary-receipt",
        "allowed_claim_classes": "" if candidate else "compile|runtime",
        "evidence_refs": "review:pending" if candidate else "gate:fixture",
        "declared_by": "SOUNIO-SCIENCE-RESEARCH-BOUNDARY",
        "declared_at": "2026-07-17",
        "review_state": "draft",
    }


def ownership_row(path: str, ring: str, target_suffix: str) -> dict[str, str]:
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
            "target_id": f"distribution:{target_suffix}",
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
        "ownership_evidence": "science-rings.tsv|gate:fixture",
    }


def fixture_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    definitions = [
        ("core", "pl-core", "core"),
        ("packages/pkg", "scientific-package", "pkg"),
        ("research", "research", "research"),
        ("stdlib", "scientific-package-candidate", "stdlib"),
    ]
    return (
        [science_row(path, ring) for path, ring, _target in definitions],
        [ownership_row(path, ring, target) for path, ring, target in definitions],
    )


def create_fixture(root: Path) -> tuple[Path, Path]:
    write(root / "core" / "compiler.sio", "fn core() -> i64 { 1 }\n")
    write(root / "packages" / "pkg" / "src" / "lib.sio", "fn package() -> i64 { 2 }\n")
    write(root / "packages" / "pkg" / "README.md", "package fixture\n")
    write(root / "research" / "study.sio", "fn study() -> i64 { 3 }\n")
    write(root / "stdlib" / "candidate.sio", "fn candidate() -> i64 { 4 }\n")
    science, ownership = fixture_rows()
    return (
        write_tsv(root / "science-rings.tsv", SCIENCE_FIELDS, science),
        write_tsv(root / "ownership.tsv", OWNERSHIP_FIELDS, ownership),
    )


def inventory_command(repo: Path, rings: Path, ownership: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
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


def verify_command(inventory: Path, repo: Path, rings: Path, ownership: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "verify",
        "--inventory",
        str(inventory),
        "--repo-root",
        str(repo),
        "--rings",
        str(rings),
        "--ownership",
        str(ownership),
    ]


def canonical_identity(payload: dict[str, object]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("inventory_identity_sha256", None)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def assert_refusal(command: list[str], output: Path | None, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    if output is not None:
        check(not output.exists(), f"refused inventory left output: {output}")
    check(code in result.stderr, f"inventory refusal lacks {code}")
    check("PHYSICAL_EXTRACTION_INVENTORY_REFUSED" in result.stderr, "inventory refusal lacks marker")
    return result


def assert_policy_refusal(
    work: Path,
    name: str,
    science: list[dict[str, str]],
    ownership: list[dict[str, str]],
    code: str,
) -> None:
    case = work / name
    shutil.copytree(work / "fixture", case)
    rings = write_tsv(case / "science-rings.tsv", SCIENCE_FIELDS, science)
    owners = write_tsv(case / "ownership.tsv", OWNERSHIP_FIELDS, ownership)
    output = work / f"{name}.json"
    assert_refusal(inventory_command(case, rings, owners, output), output, code)


def assert_tamper_refusal(
    original: Path,
    destination: Path,
    fixture: Path,
    rings: Path,
    ownership: Path,
    mutate,
    *,
    rehash: bool,
) -> None:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["inventory_identity_sha256"] = canonical_identity(payload)
    write_json(destination, payload)
    result = run(verify_command(destination, fixture, rings, ownership), expected=1)
    check("E-SRB-EXTRACT-004" in result.stderr, f"tampered inventory was not rejected: {destination.name}")
    check("PHYSICAL_EXTRACTION_INVENTORY_REFUSED" in result.stderr, "tamper refusal lacks marker")


def assert_static_contracts() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    check(schema["properties"]["schema"]["const"] == "sounio.physical-extraction-inventory.v1", "bad schema ID")
    check(schema["properties"]["extraction_status"]["const"] == "not-executed", "schema overstates extraction")
    check(schema["properties"]["assurance_level"]["const"] == "identity-only", "schema overstates assurance")
    limitations = schema["properties"]["limitations"]["const"]
    check("does_not_move_or_delete_source_files" in limitations, "move limitation is absent")
    check(
        "does_not_assert_ownership_or_maintainership_was_transferred" in limitations,
        "ownership-transfer limitation is absent",
    )
    help_text = run([sys.executable, str(TOOL), "--help"]).stdout
    check("inventory" in help_text and "verify" in help_text, "tool lacks inventory and verify commands")
    check(OWNERSHIP.is_file() and RINGS.is_file(), "canonical R3 policy inputs are absent")


def assert_fixture_flow(work: Path) -> None:
    fixture = work / "fixture"
    rings, ownership = create_fixture(fixture)
    output = work / "inventory.json"
    result = run(inventory_command(fixture, rings, ownership, output))
    check("PHYSICAL_EXTRACTION_INVENTORY_PASS" in result.stdout, "inventory lacks pass marker")
    check(output.is_file(), "inventory output is absent")
    payload = json.loads(output.read_text(encoding="ascii"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    check(set(payload) == set(schema["required"]), "payload top-level fields drifted from schema")
    check(
        set(payload["source_documents"]) == set(schema["properties"]["source_documents"]["required"]),
        "source-document fields drifted from schema",
    )
    check(set(payload["summary"]) == set(schema["properties"]["summary"]["required"]), "summary fields drifted from schema")
    check(all(set(unit) == set(schema["$defs"]["unit"]["required"]) for unit in payload["units"]), "unit fields drifted from schema")
    check(
        all(set(file) == set(schema["$defs"]["file"]["required"]) for unit in payload["units"] for file in unit["files"]),
        "file fields drifted from schema",
    )
    check(payload["limitations"] == schema["properties"]["limitations"]["const"], "limitations drifted from schema")
    check(payload["schema"] == "sounio.physical-extraction-inventory.v1", "wrong inventory schema")
    check(payload["inventory_type"] == "physical-extraction-planning-snapshot", "wrong inventory type")
    check(payload["authority_scope"] == "repository-file-identity-and-ownership-plan", "wrong authority scope")
    check(payload["extraction_status"] == "not-executed", "inventory falsely reports extraction")
    check(payload["assurance_level"] == "identity-only", "inventory overstates assurance")
    check(payload["summary"]["unit_count"] == 4, "fixture unit count is wrong")
    check(payload["summary"]["file_count"] == 5, "fixture file count is wrong")
    check(payload["summary"]["retained_core_units"] == 1, "retained core count is wrong")
    check(payload["summary"]["planned_extraction_units"] == 2, "planned extraction count is wrong")
    check(payload["summary"]["blocked_units"] == 1, "blocked unit count is wrong")
    check([unit["source_path"] for unit in payload["units"]] == ["core", "packages/pkg", "research", "stdlib"], "units are not sorted")
    check(payload["units"][0]["disposition"] == "retain-core", "core disposition drifted")
    check(payload["units"][1]["disposition"] == "extract-planned", "package disposition drifted")
    check(payload["units"][2]["ring"] == "research", "research ring is absent")
    check(payload["units"][3]["migration_state"] == "blocked-classification", "candidate was not blocked")
    check(payload["inventory_identity_sha256"] == canonical_identity(payload), "inventory identity is invalid")
    serialized = output.read_text(encoding="ascii")
    check(str(work) not in serialized, "inventory contains an absolute work path")
    check("timestamp" not in serialized and "created_at" not in serialized, "inventory contains wall-clock identity")
    check("does_not_assert_target_repository_or_distribution_exists" in payload["limitations"], "target limitation absent")
    verification = run(verify_command(output, fixture, rings, ownership))
    check("PHYSICAL_EXTRACTION_INVENTORY_VERIFY_PASS" in verification.stdout, "round-trip lacks pass marker")

    second = work / "second.json"
    run(inventory_command(fixture, rings, ownership, second))
    check(output.read_bytes() == second.read_bytes(), "inventory depends on output destination")
    copied_ownership = write(fixture / "same-ownership.tsv", ownership.read_text(encoding="utf-8"))
    third = work / "third.json"
    run(inventory_command(fixture, rings, copied_ownership, third))
    check(output.read_bytes() == third.read_bytes(), "inventory depends on ownership policy path")
    copied_rings = write(fixture / "same-rings.tsv", rings.read_text(encoding="utf-8"))
    fourth = work / "fourth.json"
    run(inventory_command(fixture, copied_rings, copied_ownership, fourth))
    check(output.read_bytes() == fourth.read_bytes(), "inventory depends on science policy path")

    occupied = write(work / "occupied.json", "preserve\n")
    occupied_result = run(inventory_command(fixture, rings, ownership, occupied), expected=1)
    check("E-SRB-EXTRACT-005" in occupied_result.stderr, "occupied output lacks promotion error")
    check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied output was overwritten")

    source = fixture / "research" / "study.sio"
    original_source = source.read_bytes()
    source.write_bytes(original_source + b"// mutation\n")
    changed = run(verify_command(output, fixture, rings, ownership), expected=1)
    check("bindings do not match" in changed.stderr, "source mutation was not detected")
    source.write_bytes(original_source)
    added = write(fixture / "research" / "added.sio", "fn added() -> i64 { 5 }\n")
    added_result = run(verify_command(output, fixture, rings, ownership), expected=1)
    check("bindings do not match" in added_result.stderr, "source addition was not detected")
    added.unlink()
    deleted = fixture / "stdlib" / "candidate.sio"
    deleted_bytes = deleted.read_bytes()
    deleted.unlink()
    deleted_result = run(verify_command(output, fixture, rings, ownership), expected=1)
    check("E-SRB-EXTRACT-003" in deleted_result.stderr, "source deletion was not detected")
    deleted.write_bytes(deleted_bytes)

    changed_ownership = ownership.read_text(encoding="utf-8").replace("future-maintainers", "changed-maintainers", 1)
    ownership.write_text(changed_ownership, encoding="utf-8", newline="\n")
    policy_result = run(verify_command(output, fixture, rings, ownership), expected=1)
    check("bindings do not match" in policy_result.stderr, "ownership mutation was not detected")
    _science_rows, ownership_rows = fixture_rows()
    write_tsv(ownership, OWNERSHIP_FIELDS, ownership_rows)
    changed_rings = rings.read_text(encoding="utf-8").replace("fixture research", "changed research", 1)
    rings.write_text(changed_rings, encoding="utf-8", newline="\n")
    rings_result = run(verify_command(output, fixture, rings, ownership), expected=1)
    check("bindings do not match" in rings_result.stderr, "science policy mutation was not detected")
    science_rows, _ownership_rows = fixture_rows()
    write_tsv(rings, SCIENCE_FIELDS, science_rows)

    assert_tamper_refusal(
        output,
        work / "tampered-id.json",
        fixture,
        rings,
        ownership,
        lambda value: value["summary"].__setitem__("unit_count", 99),
        rehash=False,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-status.json",
        fixture,
        rings,
        ownership,
        lambda value: value.__setitem__("extraction_status", "executed"),
        rehash=True,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-file.json",
        fixture,
        rings,
        ownership,
        lambda value: value["units"][0]["files"][0].__setitem__("sha256", "0" * 64),
        rehash=True,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-owner.json",
        fixture,
        rings,
        ownership,
        lambda value: value["units"][1].__setitem__("target_owner", "forged-maintainers"),
        rehash=True,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-disposition.json",
        fixture,
        rings,
        ownership,
        lambda value: value["units"][1].__setitem__("disposition", "retain-core"),
        rehash=True,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-policy-binding.json",
        fixture,
        rings,
        ownership,
        lambda value: value["source_documents"].__setitem__("ownership_policy_sha256", "1" * 64),
        rehash=True,
    )
    assert_tamper_refusal(
        output,
        work / "rehashed-extra.json",
        fixture,
        rings,
        ownership,
        lambda value: value.__setitem__("transfer_complete", True),
        rehash=True,
    )
    malformed = write(work / "malformed.json", "{")
    malformed_result = run(verify_command(malformed, fixture, rings, ownership), expected=1)
    check("E-SRB-EXTRACT-004" in malformed_result.stderr, "malformed inventory lacks structured error")
    wrong_schema = json.loads(output.read_text(encoding="ascii"))
    wrong_schema["schema"] = "sounio.physical-extraction-inventory.v2"
    wrong_schema["inventory_identity_sha256"] = canonical_identity(wrong_schema)
    wrong_schema_path = work / "wrong-schema.json"
    write_json(wrong_schema_path, wrong_schema)
    schema_result = run(verify_command(wrong_schema_path, fixture, rings, ownership), expected=1)
    check("unsupported physical inventory schema" in schema_result.stderr, "wrong schema was not rejected")


def assert_policy_adversaries(work: Path) -> None:
    science, ownership = fixture_rows()
    assert_policy_refusal(work, "missing-owner", science, ownership[:-1], "E-SRB-EXTRACT-002")
    assert_policy_refusal(work, "duplicate-owner", science, ownership + [ownership[0].copy()], "E-SRB-EXTRACT-002")
    extra = ownership_row("unknown", "research", "unknown")
    assert_policy_refusal(work, "extra-owner", science, ownership + [extra], "E-SRB-EXTRACT-002")
    mismatch = [row.copy() for row in ownership]
    mismatch[1]["ring"] = "research"
    assert_policy_refusal(work, "ring-mismatch", science, mismatch, "E-SRB-EXTRACT-002")
    duplicate_science = science + [science[0].copy()]
    assert_policy_refusal(work, "duplicate-science", duplicate_science, ownership, "E-SRB-EXTRACT-002")

    nested_science = [row.copy() for row in science]
    nested_science.append(science_row("packages", "scientific-package"))
    nested_owner = [row.copy() for row in ownership]
    nested_owner.append(ownership_row("packages", "scientific-package", "packages"))
    assert_policy_refusal(work, "nested-root", nested_science, nested_owner, "E-SRB-EXTRACT-002")
    escape_science = [row.copy() for row in science]
    escape_owner = [row.copy() for row in ownership]
    escape_science[2]["path"] = "../research"
    escape_owner[2]["source_path"] = "../research"
    assert_policy_refusal(work, "root-escape", escape_science, escape_owner, "E-SRB-EXTRACT-001")
    unknown_ring = [row.copy() for row in science]
    unknown_ring[2]["ring"] = "clinical"
    assert_policy_refusal(work, "unknown-ring", unknown_ring, ownership, "E-SRB-EXTRACT-001")
    bad_visibility = [row.copy() for row in science]
    bad_visibility[2]["visibility"] = "hidden"
    assert_policy_refusal(work, "bad-visibility", bad_visibility, ownership, "E-SRB-EXTRACT-001")
    bad_refs = [row.copy() for row in science]
    bad_refs[2]["evidence_refs"] = "gate:fixture|gate:fixture"
    assert_policy_refusal(work, "duplicate-evidence", bad_refs, ownership, "E-SRB-EXTRACT-001")

    core_extract = [row.copy() for row in ownership]
    core_extract[0].update(
        target_kind="separate-distribution",
        target_id="distribution:core",
        target_owner="future-maintainers",
        disposition="extract-planned",
        migration_state="planned",
    )
    assert_policy_refusal(work, "core-extract", science, core_extract, "E-SRB-EXTRACT-002")
    package_retain = [row.copy() for row in ownership]
    package_retain[1].update(
        target_kind="same-repository",
        target_id="repo:sounio",
        target_owner="monorepo-maintainers",
        disposition="retain-core",
        migration_state="retained",
    )
    assert_policy_refusal(work, "package-retain", science, package_retain, "E-SRB-EXTRACT-002")
    candidate_extract = [row.copy() for row in ownership]
    candidate_extract[3].update(
        target_kind="separate-distribution",
        target_id="distribution:stdlib",
        target_owner="future-maintainers",
        disposition="extract-planned",
        migration_state="planned",
    )
    assert_policy_refusal(work, "candidate-extract", science, candidate_extract, "E-SRB-EXTRACT-002")
    bad_core_owner = [row.copy() for row in ownership]
    bad_core_owner[0]["target_owner"] = "other-maintainers"
    assert_policy_refusal(work, "core-owner", science, bad_core_owner, "E-SRB-EXTRACT-002")
    unassigned_package = [row.copy() for row in ownership]
    unassigned_package[1]["target_owner"] = "unassigned"
    assert_policy_refusal(work, "unassigned-package", science, unassigned_package, "E-SRB-EXTRACT-002")
    duplicate_target = [row.copy() for row in ownership]
    duplicate_target[2]["target_id"] = duplicate_target[1]["target_id"]
    assert_policy_refusal(work, "duplicate-target", science, duplicate_target, "E-SRB-EXTRACT-002")
    assigned_candidate = [row.copy() for row in ownership]
    assigned_candidate[3]["target_owner"] = "future-maintainers"
    assert_policy_refusal(work, "assigned-candidate", science, assigned_candidate, "E-SRB-EXTRACT-002")
    unsafe_owner = [row.copy() for row in ownership]
    unsafe_owner[1]["target_owner"] = "Future Maintainers"
    assert_policy_refusal(work, "unsafe-owner", science, unsafe_owner, "E-SRB-EXTRACT-001")
    unsafe_gate = [row.copy() for row in ownership]
    unsafe_gate[1]["extraction_gate"] = "bad gate"
    assert_policy_refusal(work, "unsafe-gate", science, unsafe_gate, "E-SRB-EXTRACT-001")

    malformed_case = work / "malformed-header"
    shutil.copytree(work / "fixture", malformed_case)
    malformed_rings = write(malformed_case / "science-rings.tsv", "path\tring\ncore\tpl-core\n")
    malformed_output = work / "malformed-header.json"
    assert_refusal(
        inventory_command(malformed_case, malformed_rings, malformed_case / "ownership.tsv", malformed_output),
        malformed_output,
        "E-SRB-EXTRACT-001",
    )

    nonexistent_science = [row.copy() for row in science]
    nonexistent_owner = [row.copy() for row in ownership]
    nonexistent_science[2]["path"] = "missing-research"
    nonexistent_owner[2]["source_path"] = "missing-research"
    assert_policy_refusal(work, "missing-root", nonexistent_science, nonexistent_owner, "E-SRB-EXTRACT-003")

    symlink_case = work / "symlink-member"
    shutil.copytree(work / "fixture", symlink_case)
    link = symlink_case / "research" / "linked.sio"
    try:
        link.symlink_to(symlink_case / "core" / "compiler.sio")
    except OSError:
        pass
    else:
        symlink_output = work / "symlink-member.json"
        assert_refusal(
            inventory_command(
                symlink_case,
                symlink_case / "science-rings.tsv",
                symlink_case / "ownership.tsv",
                symlink_output,
            ),
            symlink_output,
            "E-SRB-EXTRACT-003",
        )


def assert_real_repository(work: Path) -> None:
    output = work / "real-repository.json"
    result = run(inventory_command(ROOT, RINGS, OWNERSHIP, output))
    check("PHYSICAL_EXTRACTION_INVENTORY_PASS" in result.stdout, "real repository inventory failed")
    payload = json.loads(output.read_text(encoding="ascii"))
    check(payload["summary"]["unit_count"] == 7, "real ownership policy does not cover seven roots")
    check(payload["summary"]["retained_core_units"] == 1, "real core retention count is wrong")
    check(payload["summary"]["planned_extraction_units"] == 5, "real extraction-plan count is wrong")
    check(payload["summary"]["blocked_units"] == 1, "real blocked-classification count is wrong")
    check(payload["summary"]["file_count"] > 3000, "real inventory is unexpectedly narrow")
    check(payload["summary"]["file_count"] == sum(unit["file_count"] for unit in payload["units"]), "file summary drifted")
    check(payload["summary"]["total_bytes"] == sum(unit["total_bytes"] for unit in payload["units"]), "byte summary drifted")
    units = {unit["source_path"]: unit for unit in payload["units"]}
    check(units["self-hosted"]["disposition"] == "retain-core", "self-hosted is not retained")
    check(units["stdlib"]["disposition"] == "hold-unresolved", "stdlib was extracted before classification")
    check(units["examples"]["target_id"] == "distribution:sounio-research-examples", "research target drifted")
    check(
        all(file["path"].startswith(f"{unit['source_path']}/") for unit in payload["units"] for file in unit["files"]),
        "real file escaped its ownership unit",
    )
    check(payload["inventory_identity_sha256"] == canonical_identity(payload), "real inventory identity is invalid")
    verification = run(verify_command(output, ROOT, RINGS, OWNERSHIP))
    check("PHYSICAL_EXTRACTION_INVENTORY_VERIFY_PASS" in verification.stdout, "real repository round-trip failed")


def main() -> int:
    print("SOUNIO_PHYSICAL_EXTRACTION_INVENTORY_GATE_START")
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-physical-extraction-") as temporary:
        work = Path(temporary)
        assert_fixture_flow(work)
        assert_policy_adversaries(work)
        assert_real_repository(work)
    print(f"physical-extraction-inventory tests={TESTS}")
    print("SOUNIO_PHYSICAL_EXTRACTION_INVENTORY_GATE_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(f"SOUNIO_PHYSICAL_EXTRACTION_INVENTORY_GATE_FAIL reason={error}", file=sys.stderr)
        raise SystemExit(1)
