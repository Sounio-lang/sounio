#!/usr/bin/env python3
"""Adversarial acceptance gate for the R2.6 registry attestation spec."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import package_boundary_release_gate as r25


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "science_boundary" / "registry_attestation.py"
ATTESTATION_SCHEMA = ROOT / "schemas" / "sounio.registry-attestation.v1.schema.json"
POLICY_SCHEMA = ROOT / "schemas" / "sounio.registry-attestation-policy.v1.schema.json"
RAW_MADAROS = r25.RAW_MADAROS
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], *, expected: int | set[int] = 0) -> subprocess.CompletedProcess[str]:
    return r25.run(command, expected=expected)


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")
    return path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")


def canonical_identity(payload: dict[str, object]) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload.pop("attestation_identity_sha256", None)
    encoded = json.dumps(
        identity_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def write_policy(
    path: Path,
    *,
    registry_id: str = "sounio-local-preview",
    namespace: str = "science-fixtures",
    authority_scope: str = "local-catalog-index",
    publication_status: str = "disabled",
    rings: tuple[str, ...] = ("scientific-package", "research"),
    visibilities: tuple[str, ...] = ("public",),
    claim_classes: tuple[str, ...] = ("compile",),
    assurance: tuple[str, ...] = ("identity-only",),
    extra: str = "",
) -> Path:
    quote = lambda values: ", ".join(json.dumps(value) for value in values)
    return write(
        path,
        'schema = "sounio.registry-attestation-policy.v1"\n\n'
        "[registry]\n"
        f'id = "{registry_id}"\n'
        f'namespace = "{namespace}"\n'
        f'authority-scope = "{authority_scope}"\n'
        f'publication-status = "{publication_status}"\n\n'
        "[acceptance]\n"
        f"allowed-rings = [{quote(rings)}]\n"
        f"allowed-visibilities = [{quote(visibilities)}]\n"
        f"allowed-claim-classes = [{quote(claim_classes)}]\n"
        f"allowed-assurance-levels = [{quote(assurance)}]\n"
        'required-boundary-mode = "strict"\n'
        'required-boundary-verdict = "OK"\n'
        f"{extra}",
    )


def attest_command(bundle: Path, project: Path, policy: Path, output: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "attest",
        "--bundle",
        str(bundle),
        "--root",
        str(project),
        "--compiler",
        str(RAW_MADAROS),
        "--registry-policy",
        str(policy),
        "--output",
        str(output),
    ]


def verify_command(
    attestation: Path,
    bundle: Path,
    project: Path,
    policy: Path,
    compiler: Path = RAW_MADAROS,
) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "verify",
        "--attestation",
        str(attestation),
        "--bundle",
        str(bundle),
        "--root",
        str(project),
        "--compiler",
        str(compiler),
        "--registry-policy",
        str(policy),
    ]


def assert_refusal(command: list[str], output: Path, code: str) -> subprocess.CompletedProcess[str]:
    result = run(command, expected=1)
    check(not output.exists(), f"refused attestation left output: {output}")
    check(code in result.stderr, f"registry refusal lacks {code}")
    check("REGISTRY_ATTESTATION_REFUSED" in result.stderr, "registry refusal lacks structured marker")
    return result


def assert_tampered_attestation(
    original: Path,
    destination: Path,
    bundle: Path,
    project: Path,
    policy: Path,
    mutate,
    *,
    rehash: bool,
) -> subprocess.CompletedProcess[str]:
    payload = json.loads(original.read_text(encoding="ascii"))
    mutate(payload)
    if rehash:
        payload["attestation_identity_sha256"] = canonical_identity(payload)
    write_json(destination, payload)
    result = run(verify_command(destination, bundle, project, policy), expected=1)
    check("E-SRB-REGISTRY-003" in result.stderr, f"tampered attestation was not rejected: {destination.name}")
    check("REGISTRY_ATTESTATION_REFUSED" in result.stderr, "tampered attestation lacks refusal marker")
    return result


def assert_static_contracts() -> None:
    attestation_schema = json.loads(ATTESTATION_SCHEMA.read_text(encoding="utf-8"))
    policy_schema = json.loads(POLICY_SCHEMA.read_text(encoding="utf-8"))
    check(
        attestation_schema["properties"]["schema"]["const"] == "sounio.registry-attestation.v1",
        "bad registry attestation schema",
    )
    check(
        attestation_schema["properties"]["attestation_type"]["const"]
        == "unsigned-local-policy-evaluation",
        "attestation type overstates issuer authority",
    )
    check(
        attestation_schema["properties"]["registry"]["properties"]["publication_status"]["const"]
        == "disabled",
        "attestation schema enables publication",
    )
    check(
        policy_schema["properties"]["schema"]["const"] == "sounio.registry-attestation-policy.v1",
        "bad registry policy schema",
    )
    check(
        policy_schema["properties"]["registry"]["properties"]["publication-status"]["const"]
        == "disabled",
        "registry policy schema enables publication",
    )
    help_text = run([sys.executable, str(TOOL), "--help"]).stdout
    check("attest" in help_text and "verify" in help_text, "registry attestation tool lacks both commands")
    preview_server = (ROOT / "scripts" / "dev" / "registry_serve.py").read_text(encoding="utf-8")
    check("publishing is disabled" in preview_server, "local preview server enabled publishing")


def assert_flow(work: Path) -> None:
    project = work / "project"
    claim = r25.write_project(project, ring="scientific-package")
    bundle = work / "release.sio-release"
    release = run(r25.release_command(project, claim, bundle))
    check("PACKAGE_BOUNDARY_RELEASE_PASS" in release.stdout, "R2.5 prerequisite bundle was not emitted")

    policy = write_policy(work / "registry-policy.toml")
    attestation = work / "registry-attestation.json"
    result = run(attest_command(bundle, project, policy, attestation))
    check("REGISTRY_ATTESTATION_PASS" in result.stdout, "attestation lacks success marker")
    check(attestation.is_file(), "attestation output is absent")
    payload = json.loads(attestation.read_text(encoding="ascii"))
    check(payload["schema"] == "sounio.registry-attestation.v1", "wrong attestation schema")
    check(payload["attestation_type"] == "unsigned-local-policy-evaluation", "attestation type overstates authority")
    check(payload["registry"]["publication_status"] == "disabled", "attestation claims publication")
    check(payload["registry"]["authority_scope"] == "local-catalog-index", "wrong registry authority scope")
    check(payload["decision"]["verdict"] == "POLICY_MATCH", "wrong registry decision")
    check(
        payload["decision"]["checks"]
        == [
            "full_release_bundle_verification",
            "registry_policy_identity_binding",
            "conclusive_ring_allowed",
            "visibility_allowed",
            "claim_class_allowed",
            "identity_assurance_allowed",
            "strict_boundary_ok",
        ],
        "attestation does not record the executed policy predicates",
    )
    check(payload["assurance_level"] == "identity-only", "registry attestation overstates assurance")
    check(payload["science"]["ring"] == "scientific-package", "attestation lost the package ring")
    check(payload["science"]["context_of_use"] == "package release gate fixture", "attestation lost context of use")
    check(payload["release_bindings"]["claim_id"] == "gate.package-release.compile", "claim identity is not bound")
    check(
        payload["release_bindings"]["compiler_sha256"]
        == hashlib.sha256(RAW_MADAROS.read_bytes()).hexdigest(),
        "compiler identity is not bound",
    )
    check(
        "does_not_assert_namespace_ownership_or_issuer_identity" in payload["limitations"],
        "issuer limitation is absent",
    )
    check(
        "does_not_assert_remote_signature" in payload["limitations"],
        "remote signature limitation is absent",
    )
    serialized = attestation.read_text(encoding="ascii")
    check(str(work) not in serialized, "attestation contains an absolute work path")
    check("timestamp" not in serialized and "created_at" not in serialized, "attestation contains wall-clock identity")
    check(payload["attestation_identity_sha256"] == canonical_identity(payload), "attestation identity is invalid")

    verification = run(verify_command(attestation, bundle, project, policy))
    check("REGISTRY_ATTESTATION_VERIFY_PASS" in verification.stdout, "round-trip verification lacks pass marker")

    second = work / "second-attestation.json"
    run(attest_command(bundle, project, policy, second))
    check(attestation.read_bytes() == second.read_bytes(), "attestation is not deterministic across destinations")
    policy_copy = write(work / "same-policy.toml", policy.read_text(encoding="utf-8"))
    third = work / "third-attestation.json"
    run(attest_command(bundle, project, policy_copy, third))
    check(attestation.read_bytes() == third.read_bytes(), "attestation identity depends on policy path")

    occupied = write(work / "occupied.json", "preserve\n")
    occupied_result = run(attest_command(bundle, project, policy, occupied), expected=1)
    check("E-SRB-REGISTRY-004" in occupied_result.stderr, "occupied output refusal lacks promotion diagnostic")
    check(occupied.read_text(encoding="utf-8") == "preserve\n", "occupied output was overwritten")

    deny_ring = write_policy(work / "deny-ring.toml", rings=("research",))
    ring_output = work / "deny-ring.json"
    ring_result = assert_refusal(
        attest_command(bundle, project, deny_ring, ring_output),
        ring_output,
        "E-SRB-REGISTRY-002",
    )
    check("ring is not allowed" in ring_result.stderr, "ring refusal is not explicit")

    deny_visibility = write_policy(work / "deny-visibility.toml", visibilities=("protected",))
    visibility_output = work / "deny-visibility.json"
    visibility_result = assert_refusal(
        attest_command(bundle, project, deny_visibility, visibility_output),
        visibility_output,
        "E-SRB-REGISTRY-002",
    )
    check("visibility is not allowed" in visibility_result.stderr, "visibility refusal is not explicit")

    deny_claim = write_policy(work / "deny-claim.toml", claim_classes=("runtime",))
    claim_output = work / "deny-claim.json"
    claim_result = assert_refusal(
        attest_command(bundle, project, deny_claim, claim_output),
        claim_output,
        "E-SRB-REGISTRY-002",
    )
    check("claim class is not allowed" in claim_result.stderr, "claim refusal is not explicit")

    bad_publication = write_policy(work / "bad-publication.toml", publication_status="launched")
    assert_refusal(
        attest_command(bundle, project, bad_publication, work / "bad-publication.json"),
        work / "bad-publication.json",
        "E-SRB-REGISTRY-001",
    )
    bad_ring = write_policy(work / "bad-ring.toml", rings=("scientific-package-candidate",))
    assert_refusal(
        attest_command(bundle, project, bad_ring, work / "bad-ring.json"),
        work / "bad-ring.json",
        "E-SRB-REGISTRY-001",
    )
    bad_assurance = write_policy(work / "bad-assurance.toml", assurance=("signed",))
    assert_refusal(
        attest_command(bundle, project, bad_assurance, work / "bad-assurance.json"),
        work / "bad-assurance.json",
        "E-SRB-REGISTRY-001",
    )
    extra_policy = write_policy(work / "extra-policy.toml", extra='unexpected = "field"\n')
    assert_refusal(
        attest_command(bundle, project, extra_policy, work / "extra-policy.json"),
        work / "extra-policy.json",
        "E-SRB-REGISTRY-001",
    )
    malformed_policy = write(work / "malformed-policy.toml", "[registry\n")
    assert_refusal(
        attest_command(bundle, project, malformed_policy, work / "malformed-policy.json"),
        work / "malformed-policy.json",
        "E-SRB-REGISTRY-001",
    )

    assert_tampered_attestation(
        attestation,
        work / "tampered-id.json",
        bundle,
        project,
        policy,
        lambda value: value["registry"].__setitem__("id", "forged-registry"),
        rehash=False,
    )
    assert_tampered_attestation(
        attestation,
        work / "rehashed-registry.json",
        bundle,
        project,
        policy,
        lambda value: value["registry"].__setitem__("id", "forged-registry"),
        rehash=True,
    )
    assert_tampered_attestation(
        attestation,
        work / "rehashed-bundle.json",
        bundle,
        project,
        policy,
        lambda value: value["package"].__setitem__("bundle_identity_sha256", "0" * 64),
        rehash=True,
    )
    assert_tampered_attestation(
        attestation,
        work / "rehashed-claim-binding.json",
        bundle,
        project,
        policy,
        lambda value: value["release_bindings"].__setitem__("claim_contract_sha256", "1" * 64),
        rehash=True,
    )
    assert_tampered_attestation(
        attestation,
        work / "rehashed-compiler-binding.json",
        bundle,
        project,
        policy,
        lambda value: value["release_bindings"].__setitem__("compiler_sha256", "2" * 64),
        rehash=True,
    )
    assert_tampered_attestation(
        attestation,
        work / "rehashed-type.json",
        bundle,
        project,
        policy,
        lambda value: value.__setitem__("attestation_type", "remote-signed"),
        rehash=True,
    )
    assert_tampered_attestation(
        attestation,
        work / "extra-field.json",
        bundle,
        project,
        policy,
        lambda value: value.__setitem__("signature", "none"),
        rehash=True,
    )

    malformed_attestation = write(work / "malformed-attestation.json", "{")
    malformed_result = run(verify_command(malformed_attestation, bundle, project, policy), expected=1)
    check("E-SRB-REGISTRY-003" in malformed_result.stderr, "malformed attestation lacks structured diagnostic")

    original_policy = policy.read_bytes()
    write_policy(policy, registry_id="changed-local-preview")
    changed_policy = run(verify_command(attestation, bundle, project, policy), expected=1)
    check("bindings do not match" in changed_policy.stderr, "registry policy mutation was not revalidated")
    policy.write_bytes(original_policy)

    source = project / "src" / "greet.sio"
    original_source = source.read_bytes()
    source.write_bytes(original_source + b"\n// changed\n")
    changed_source = run(verify_command(attestation, bundle, project, policy), expected=1)
    check("release bundle verification failed" in changed_source.stderr, "source mutation was not revalidated")
    source.write_bytes(original_source)

    different_compiler = write(work / "different-compiler", "#!/usr/bin/env sh\nexit 1\n")
    different_compiler.chmod(0o755)
    changed_compiler = run(verify_command(attestation, bundle, project, policy, different_compiler), expected=1)
    check("release bundle verification failed" in changed_compiler.stderr, "compiler mutation was not revalidated")

    artifact = bundle / "artifacts" / "release-fixture"
    artifact_size = artifact.stat().st_size
    with artifact.open("ab") as handle:
        handle.write(b"tampered")
    tampered_bundle_output = work / "tampered-bundle.json"
    tampered_bundle = assert_refusal(
        attest_command(bundle, project, policy, tampered_bundle_output),
        tampered_bundle_output,
        "E-SRB-REGISTRY-002",
    )
    check("artifact hash mismatch" in tampered_bundle.stderr, "bundle mutation was not revalidated")
    with artifact.open("r+b") as handle:
        handle.truncate(artifact_size)


def main() -> int:
    print("SOUNIO_REGISTRY_ATTESTATION_SPEC_GATE_START")
    check(RAW_MADAROS.is_file(), f"current-source Madaros not found: {RAW_MADAROS}")
    check(b"--science-boundary-closure" in RAW_MADAROS.read_bytes(), "Madaros lacks raw AST boundary collector")
    assert_static_contracts()
    with tempfile.TemporaryDirectory(prefix="sounio-registry-attestation-") as temporary:
        assert_flow(Path(temporary))
    print(f"registry-attestation-spec tests={TESTS}")
    print("SOUNIO_REGISTRY_ATTESTATION_SPEC_GATE_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, subprocess.TimeoutExpired) as error:
        print(f"SOUNIO_REGISTRY_ATTESTATION_SPEC_GATE_FAIL reason={error}", file=sys.stderr)
        raise SystemExit(1)
