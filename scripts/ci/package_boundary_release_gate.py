#!/usr/bin/env python3
"""Adversarial acceptance gate for R2.5 package release bundles."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SOUC = ROOT / "bin" / "souc"
SCHEMA = ROOT / "schemas" / "sounio.package-release-bundle.v1.schema.json"
INVENTORY = ROOT / "docs" / "ecosystem" / "curated-package-release-inventory.tsv"
RAW_MADAROS = Path(
    os.environ.get(
        "SOUNIO_PACKAGE_BOUNDARY_MADAROS_BIN",
        os.environ.get(
            "SOUNIO_SCIENCE_BOUNDARY_MADAROS_BIN",
            str(ROOT / "artifacts" / "self-hosted" / "madaros"),
        ),
    )
).expanduser().resolve()
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
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    accepted = {expected} if isinstance(expected, int) else expected
    env = os.environ.copy()
    env["MADAROS_RAW_BIN"] = str(RAW_MADAROS)
    if environment:
        env.update(environment)
    result = subprocess.run(
        command,
        cwd=cwd or ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=300,
        check=False,
    )
    if result.returncode not in accepted:
        raise AssertionError(
            f"command returned {result.returncode}, expected {sorted(accepted)}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def write_manifest(root: Path, *, ring: str = "research") -> Path:
    return write(
        root / "sounio.toml",
        "[package]\n"
        'name = "release-fixture"\n'
        'version = "0.1.0"\n'
        'edition = "2026"\n\n'
        "[science]\n"
        'schema = "sounio.science-manifest.v1"\n'
        f'ring = "{ring}"\n'
        'evidence-status = "fixture"\n'
        'context-of-use = "package release gate fixture"\n'
        'visibility = "public"\n'
        'allowed-claim-classes = ["compile", "runtime"]\n'
        'evidence-refs = ["gate:package_boundary_release_gate"]\n'
        'next-gate = "package-boundary-receipt"\n'
        'declared-by = "SOUNIO-SCIENCE-RESEARCH-BOUNDARY"\n'
        'declared-at = "2026-07-17"\n'
        'review-state = "reviewed"\n\n'
        "[[bin]]\n"
        'name = "release-fixture"\n'
        'path = "src/main.sio"\n',
    )


def write_project(root: Path, *, ring: str = "research") -> Path:
    write_manifest(root, ring=ring)
    write(root / "src" / "greet.sio", "pub fn answer() -> i64 {\n    42\n}\n")
    write(
        root / "src" / "main.sio",
        "use greet::{answer}\n\n"
        "fn main() -> i64 with IO {\n"
        "    print_int(answer())\n"
        "    print_char(10)\n"
        "    0\n"
        "}\n",
    )
    write(root / "evidence" / "gate.txt", "package boundary release gate\n")
    return write_claim(root)


def write_claim(root: Path, *, requested: str = "compile") -> Path:
    evidence = [
        ("source", "src/main.sio", root / "src" / "main.sio"),
        ("package", "sounio.toml", root / "sounio.toml"),
        ("compiler", RAW_MADAROS.name, RAW_MADAROS),
        ("gate", "evidence/gate.txt", root / "evidence" / "gate.txt"),
    ]
    body = (
        'schema = "sounio.claim-contract.v1"\n'
        'claim-id = "gate.package-release.compile"\n'
        f'requested-class = "{requested}"\n'
        'context-of-use = "package release gate fixture"\n'
        'root-artifact = "src/main.sio"\n'
    )
    for evidence_type, reference, path in evidence:
        body += (
            "\n[[evidence]]\n"
            f'type = "{evidence_type}"\n'
            f'ref = "{reference}"\n'
            f'sha256 = "{sha256(path)}"\n'
        )
    return write(root / "claim.toml", body)


def release_command(root: Path, claim: Path | None, bundle: Path | None = None) -> list[str]:
    command = [str(SOUC), "pkg", "build", str(root), "--science-boundary", "strict"]
    if claim is not None:
        command.extend(["--claim-contract", str(claim)])
    if bundle is not None:
        command.extend(["--release-bundle", str(bundle)])
    return command


def directory_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\n")
    return digest.hexdigest()


def canonical_identity(payload: dict[str, object], field: str) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload.pop(field, None)
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def receipt_identity(payload: dict[str, object]) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload["hashes"].pop("receipt_identity_sha256", None)
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def assert_bundle_shape(bundle: Path, project: Path) -> dict[str, object]:
    expected = {
        "package-release.json",
        "artifacts/release-fixture",
        "attestations/package-boundary-receipt.json",
        "claims/claim-contract.toml",
    }
    actual = {path.relative_to(bundle).as_posix() for path in bundle.rglob("*") if path.is_file()}
    check(actual == expected, "release bundle file inventory is not exact")
    manifest = json.loads((bundle / "package-release.json").read_text(encoding="ascii"))
    check(manifest["schema"] == "sounio.package-release-bundle.v1", "bad release bundle schema")
    check(manifest["boundary_receipt"]["verdict"] == "OK", "release manifest does not bind OK")
    check(manifest["boundary_receipt"]["mode"] == "strict", "release manifest is not strict")
    check(manifest["assurance_level"] == "identity-only", "release manifest overstates assurance")
    check(
        "full_verification_requires_original_sources_policy_and_compiler" in manifest["limitations"],
        "release manifest omits its verification limitation",
    )
    receipt = json.loads(
        (bundle / "attestations" / "package-boundary-receipt.json").read_text(encoding="ascii")
    )
    check(receipt["verdict"] == "OK" and receipt["mode"] == "strict", "bundle receipt is not strict OK")
    check(receipt["engine"]["boundary_collector"] == "madaros-raw-ast-v1", "bundle used fallback closure")
    check(not receipt["graph"]["saturated"], "bundle closure is saturated")
    check(not receipt["graph"]["unresolved_imports"], "bundle closure contains unresolved imports")
    check(receipt["artifact"] == {"kind": "native-elf", "path": "artifacts/release-fixture"}, "bad receipt artifact label")
    check(receipt["hashes"]["elf_sha256"] == sha256(bundle / "artifacts" / "release-fixture"), "ELF binding mismatch")
    check(receipt["hashes"]["claim_contract_sha256"] == sha256(project / "claim.toml"), "claim binding mismatch")
    return manifest


def assert_refusal(command: list[str], bundle: Path, expected_code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected={1, 2})
    check(not bundle.exists(), f"refused release left a final bundle: {bundle}")
    check(expected_code in result.stderr, f"refusal lacks {expected_code}")
    check("PACKAGE_BOUNDARY_RELEASE_REFUSED" in result.stderr, "refusal lacks structured marker")
    return result


def assert_tamper_refusal(source_bundle: Path, project: Path, work: Path, relative: str, mutation: bytes) -> None:
    tampered = work / (relative.replace("/", "-") + ".sio-release")
    shutil.copytree(source_bundle, tampered)
    target = tampered / relative
    target.write_bytes(target.read_bytes() + mutation)
    result = run(
        [str(SOUC), "pkg", "verify", str(tampered), "--root", str(project)],
        expected=1,
    )
    check("E-SRB-RELEASE-005" in result.stderr, f"tampered {relative} lacks structured diagnostic")
    check("PACKAGE_BOUNDARY_RELEASE_REFUSED" in result.stderr, f"tampered {relative} lacks refusal marker")


def assert_static_contracts() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    check(schema["properties"]["schema"]["const"] == "sounio.package-release-bundle.v1", "bad bundle JSON schema")
    help_text = run([str(SOUC), "--help"]).stdout
    for value in ("pkg build", "pkg verify", "--release-bundle"):
        check(value in help_text, f"public help lacks {value}")

    with INVENTORY.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    check(
        set(rows[0])
        == {
            "package",
            "repo_path",
            "manifest_state",
            "ring_state",
            "claim_contract_state",
            "release_eligibility",
            "evidence",
            "next_action",
        },
        "curated release inventory header drifted",
    )
    expected = {"epistemic-core", "epistemic-stats", "darwin-pbpk", "snn-fractal", "regulatory-tools"}
    check({row["package"] for row in rows} == expected, "curated release inventory is not the bounded Phase 1 subset")
    core = next(row for row in rows if row["package"] == "epistemic-core")
    check(core["release_eligibility"] == "not-release-eligible", "draft epistemic-core became release eligible")
    check(all(row["release_eligibility"] != "release-eligible" for row in rows), "inventory inferred release authority")


def assert_release_flow(work: Path) -> None:
    project = work / "project"
    claim = write_project(project)
    default_bundle = project / "target" / "release" / "release-fixture-0.1.0.sio-release"
    result = run(release_command(project, claim))
    check("PACKAGE_BOUNDARY_RELEASE_PASS" in result.stdout, "strict pkg build lacks success marker")
    assert_bundle_shape(default_bundle, project)
    artifact = default_bundle / "artifacts" / "release-fixture"
    execution = run([str(artifact)])
    check(execution.stdout.strip() == "42", "release artifact is not runnable")

    verification = run([str(SOUC), "pkg", "verify", str(default_bundle), "--root", str(project)])
    check("PACKAGE_BOUNDARY_RELEASE_VERIFY_PASS" in verification.stdout, "round-trip verification lacks pass marker")

    second = work / "second.sio-release"
    run(release_command(project, claim, second))
    check(directory_digest(default_bundle) == directory_digest(second), "release bundle is not deterministic")
    check(
        (default_bundle / "package-release.json").read_bytes() == (second / "package-release.json").read_bytes(),
        "release identity changes with destination path",
    )

    for relative in (
        "artifacts/release-fixture",
        "attestations/package-boundary-receipt.json",
        "claims/claim-contract.toml",
        "package-release.json",
    ):
        assert_tamper_refusal(default_bundle, project, work / "tamper", relative, b"\ntampered\n")
    extra = work / "tamper" / "extra.sio-release"
    shutil.copytree(default_bundle, extra)
    write(extra / ".undeclared", "extra\n")
    extra_result = run([str(SOUC), "pkg", "verify", str(extra), "--root", str(project)], expected=1)
    check("file inventory mismatch" in extra_result.stderr, "undeclared hidden bundle entry was not refused")

    forged_package = work / "tamper" / "forged-package.sio-release"
    shutil.copytree(default_bundle, forged_package)
    forged_package_manifest_path = forged_package / "package-release.json"
    forged_package_manifest = json.loads(forged_package_manifest_path.read_text(encoding="ascii"))
    forged_package_manifest["package"]["name"] = "forged-package"
    forged_package_manifest["bundle_identity_sha256"] = canonical_identity(
        forged_package_manifest, "bundle_identity_sha256"
    )
    write_json(forged_package_manifest_path, forged_package_manifest)
    forged_package_result = run(
        [str(SOUC), "pkg", "verify", str(forged_package), "--root", str(project)],
        expected=1,
    )
    check("artifact path is not canonical" in forged_package_result.stderr, "rehashed package forgery was accepted")

    forged_claim = work / "tamper" / "forged-claim-summary.sio-release"
    shutil.copytree(default_bundle, forged_claim)
    forged_receipt_path = forged_claim / "attestations" / "package-boundary-receipt.json"
    forged_receipt = json.loads(forged_receipt_path.read_text(encoding="ascii"))
    forged_receipt["claim_contract"]["claim_id"] = "forged.claim"
    forged_receipt["hashes"]["receipt_identity_sha256"] = receipt_identity(forged_receipt)
    write_json(forged_receipt_path, forged_receipt)
    forged_bundle_manifest_path = forged_claim / "package-release.json"
    forged_bundle_manifest = json.loads(forged_bundle_manifest_path.read_text(encoding="ascii"))
    forged_bundle_manifest["boundary_receipt"]["sha256"] = sha256(forged_receipt_path)
    forged_bundle_manifest["boundary_receipt"]["identity_sha256"] = forged_receipt["hashes"][
        "receipt_identity_sha256"
    ]
    forged_bundle_manifest["claim_contract"]["claim_id"] = "forged.claim"
    forged_bundle_manifest["bundle_identity_sha256"] = canonical_identity(
        forged_bundle_manifest, "bundle_identity_sha256"
    )
    write_json(forged_bundle_manifest_path, forged_bundle_manifest)
    forged_claim_result = run(
        [str(SOUC), "pkg", "verify", str(forged_claim), "--root", str(project)],
        expected=1,
    )
    check("bundled claim content" in forged_claim_result.stderr, "rehashed claim-summary forgery was accepted")

    original_source = (project / "src" / "greet.sio").read_bytes()
    (project / "src" / "greet.sio").write_bytes(original_source + b"\n// changed\n")
    source_result = run([str(SOUC), "pkg", "verify", str(default_bundle), "--root", str(project)], expected=1)
    check("source hash mismatch" in source_result.stderr, "source mutation was not revalidated")
    (project / "src" / "greet.sio").write_bytes(original_source)

    original_claim = claim.read_bytes()
    claim.write_bytes(original_claim + b"\n# changed\n")
    claim_result = run([str(SOUC), "pkg", "verify", str(default_bundle), "--root", str(project)], expected=1)
    check("claim contract hash mismatch" in claim_result.stderr, "claim mutation was not revalidated")
    claim.write_bytes(original_claim)

    compiler_copy = work / "different-compiler"
    shutil.copyfile(RAW_MADAROS, compiler_copy)
    with compiler_copy.open("ab") as handle:
        handle.write(b"different")
    compiler_result = run(
        [
            sys.executable,
            str(ROOT / "tools" / "science_boundary" / "package_release.py"),
            "verify",
            "--bundle",
            str(default_bundle),
            "--root",
            str(project),
            "--compiler",
            str(compiler_copy),
        ],
        expected=1,
    )
    check("compiler hash mismatch" in compiler_result.stderr, "compiler mutation was not revalidated")

    missing_claim_bundle = work / "missing-claim.sio-release"
    assert_refusal(release_command(project, None, missing_claim_bundle), missing_claim_bundle, "E-SRB-RELEASE-002")

    external_receipt_bundle = work / "external-receipt.sio-release"
    external_command = release_command(project, claim, external_receipt_bundle)
    external_command.extend(["--emit-boundary-receipt", str(work / "outside.json")])
    assert_refusal(external_command, external_receipt_bundle, "E-SRB-RELEASE-002")
    check(not (work / "outside.json").exists(), "strict package refusal wrote an external receipt")

    advisory_bundle = work / "advisory.sio-release"
    advisory_command = [
        str(SOUC),
        "pkg",
        "build",
        str(project),
        "--science-boundary",
        "advisory",
        "--release-bundle",
        str(advisory_bundle),
    ]
    assert_refusal(advisory_command, advisory_bundle, "E-SRB-RELEASE-002")

    wrong_policy_bundle = work / "wrong-policy.sio-release"
    wrong_policy_command = release_command(project, claim, wrong_policy_bundle)
    wrong_policy_command.extend(["--science-manifest", str(project / "evidence" / "gate.txt")])
    assert_refusal(wrong_policy_command, wrong_policy_bundle, "E-SRB-RELEASE-002")

    unauthorized = work / "unauthorized"
    unauthorized_claim = write_project(unauthorized)
    unauthorized_claim = write_claim(unauthorized, requested="clinical")
    unauthorized_bundle = work / "unauthorized.sio-release"
    unauthorized_result = assert_refusal(
        release_command(unauthorized, unauthorized_claim, unauthorized_bundle),
        unauthorized_bundle,
        "E-SRB-RELEASE-002",
    )
    check("not allowed by the package policy" in unauthorized_result.stderr, "package preflight did not refuse the claim")

    unknown = work / "unknown"
    unknown_claim = write_project(unknown, ring="scientific-package-candidate")
    unknown_bundle = work / "unknown.sio-release"
    unknown_result = assert_refusal(
        release_command(unknown, unknown_claim, unknown_bundle),
        unknown_bundle,
        "E-SRB-RELEASE-003",
    )
    check("verdict=UNKNOWN" in unknown_result.stderr, "UNKNOWN release refusal is not explicit")

    occupied = work / "occupied.sio-release"
    occupied.mkdir()
    marker = write(occupied / "owner.txt", "preserve\n")
    occupied_result = run(release_command(project, claim, occupied), expected=1)
    check("E-SRB-RELEASE-004" in occupied_result.stderr, "existing bundle refusal lacks promotion diagnostic")
    check(marker.read_text(encoding="utf-8") == "preserve\n", "existing bundle was mutated")


def main() -> int:
    print("SOUNIO_PACKAGE_BOUNDARY_RELEASE_GATE_START")
    check(RAW_MADAROS.is_file(), f"current-source Madaros not found: {RAW_MADAROS}")
    check(b"--science-boundary-closure" in RAW_MADAROS.read_bytes(), "Madaros lacks raw AST boundary collector")
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-package-release-") as temporary:
        assert_release_flow(Path(temporary))
    print(f"package-boundary-release tests={TESTS}")
    print("SOUNIO_PACKAGE_BOUNDARY_RELEASE_GATE_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(f"SOUNIO_PACKAGE_BOUNDARY_RELEASE_GATE_FAIL reason={error}", file=sys.stderr)
        raise SystemExit(1)
