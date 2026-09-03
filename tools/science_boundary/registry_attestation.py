#!/usr/bin/env python3
"""Emit and verify deterministic local registry-policy attestations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import tomllib
from typing import Any

import package_release


ATTESTATION_SCHEMA = "sounio.registry-attestation.v1"
POLICY_SCHEMA = "sounio.registry-attestation-policy.v1"
ATTESTATION_TYPE = "unsigned-local-policy-evaluation"
AUTHORITY_SCOPE = "local-catalog-index"
PUBLICATION_STATUS = "disabled"
CONCLUSIVE_RINGS = {"pl-core", "scientific-package", "research"}
VISIBILITIES = {"public", "protected", "embargoed"}
SAFE_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LIMITATIONS = [
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_public_registry_status_or_publication",
    "does_not_assert_namespace_ownership_or_issuer_identity",
    "does_not_assert_remote_signature",
    "does_not_assert_attested_execution_or_independent_replay",
    "full_verification_requires_original_bundle_sources_policy_and_compiler",
]


class RegistryAttestationError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def absolute_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def attestation_identity(payload: dict[str, Any]) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload.pop("attestation_identity_sha256", None)
    return hashlib.sha256(canonical_json(identity_payload)).hexdigest()


def with_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["attestation_identity_sha256"] = attestation_identity(result)
    return result


def exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise RegistryAttestationError("E-SRB-REGISTRY-001", f"{label} fields do not match schema v1")
    return value


def string_list(value: Any, label: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and SAFE_TOKEN.fullmatch(item) for item in value)
        or len(set(value)) != len(value)
    ):
        raise RegistryAttestationError("E-SRB-REGISTRY-001", f"{label} must be a non-empty unique token list")
    return value


def load_policy(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RegistryAttestationError("E-SRB-REGISTRY-001", f"registry policy is not a regular file: {path}")
    try:
        with path.open("rb") as handle:
            policy = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise RegistryAttestationError("E-SRB-REGISTRY-001", f"cannot parse registry policy: {error}") from error
    exact_keys(policy, {"schema", "registry", "acceptance"}, "registry policy")
    if policy.get("schema") != POLICY_SCHEMA:
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "unsupported registry policy schema")

    registry = exact_keys(
        policy.get("registry"),
        {"id", "namespace", "authority-scope", "publication-status"},
        "registry policy [registry]",
    )
    for field in ("id", "namespace"):
        if not isinstance(registry.get(field), str) or not SAFE_TOKEN.fullmatch(registry[field]):
            raise RegistryAttestationError("E-SRB-REGISTRY-001", f"registry {field} is not a safe token")
    if registry.get("authority-scope") != AUTHORITY_SCOPE:
        raise RegistryAttestationError(
            "E-SRB-REGISTRY-001",
            f"registry authority-scope must be {AUTHORITY_SCOPE}",
        )
    if registry.get("publication-status") != PUBLICATION_STATUS:
        raise RegistryAttestationError(
            "E-SRB-REGISTRY-001",
            "registry attestation v1 requires publication-status = disabled",
        )

    acceptance = exact_keys(
        policy.get("acceptance"),
        {
            "allowed-rings",
            "allowed-visibilities",
            "allowed-claim-classes",
            "allowed-assurance-levels",
            "required-boundary-mode",
            "required-boundary-verdict",
        },
        "registry policy [acceptance]",
    )
    rings = string_list(acceptance.get("allowed-rings"), "allowed-rings")
    visibilities = string_list(acceptance.get("allowed-visibilities"), "allowed-visibilities")
    string_list(acceptance.get("allowed-claim-classes"), "allowed-claim-classes")
    assurance = string_list(acceptance.get("allowed-assurance-levels"), "allowed-assurance-levels")
    if not set(rings).issubset(CONCLUSIVE_RINGS):
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "allowed-rings contains a non-conclusive ring")
    if not set(visibilities).issubset(VISIBILITIES):
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "allowed-visibilities contains an unsupported value")
    if set(assurance) != {"identity-only"}:
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "v1 only permits identity-only assurance")
    if acceptance.get("required-boundary-mode") != "strict":
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "v1 requires strict boundary mode")
    if acceptance.get("required-boundary-verdict") != "OK":
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "v1 requires boundary verdict OK")
    return policy


def resolve_root(candidate: str) -> tuple[Path, Path, dict[str, Any]]:
    root, manifest, package_manifest, _entry = package_release.resolve_project(candidate)
    requested = Path(candidate).expanduser().resolve()
    requested_root = requested if requested.is_dir() else requested.parent
    if root != requested_root:
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "project root does not match sounio.toml")
    return root, manifest, package_manifest


def policy_refusal(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryAttestationError("E-SRB-REGISTRY-002", message)


def expected_attestation(bundle: Path, root: Path, compiler: Path, policy_path: Path) -> dict[str, Any]:
    policy = load_policy(policy_path)
    try:
        release = package_release.validate_bundle(bundle, root, compiler)
    except package_release.ReleaseError as error:
        raise RegistryAttestationError(
            "E-SRB-REGISTRY-002",
            f"release bundle verification failed ({error.code}): {error}",
        ) from error
    checks = ["full_release_bundle_verification", "registry_policy_identity_binding"]

    resolved_root, package_manifest_path, package_manifest = resolve_root(str(root))
    if resolved_root != root.resolve():
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "verification root is not canonical")
    science = package_manifest.get("science")
    required_science = {
        "schema",
        "ring",
        "evidence-status",
        "context-of-use",
        "visibility",
        "allowed-claim-classes",
        "evidence-refs",
    }
    if not isinstance(science, dict) or not required_science.issubset(science):
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "package [science] lacks required fields")
    if science.get("schema") != "sounio.science-manifest.v1":
        raise RegistryAttestationError("E-SRB-REGISTRY-001", "package science policy has the wrong schema")

    acceptance = policy["acceptance"]
    boundary = release["boundary_receipt"]
    claim = release["claim_contract"]
    ring = str(science.get("ring", ""))
    visibility = str(science.get("visibility", ""))
    evidence_status = str(science.get("evidence-status", "")).strip()
    context_of_use = str(science.get("context-of-use", "")).strip()
    requested_class = str(claim.get("requested_class", ""))
    if not evidence_status or not context_of_use:
        raise RegistryAttestationError(
            "E-SRB-REGISTRY-001",
            "package science policy requires non-empty evidence-status and context-of-use",
        )
    policy_refusal(ring in CONCLUSIVE_RINGS, f"package ring is not conclusive: {ring}")
    policy_refusal(ring in acceptance["allowed-rings"], f"package ring is not allowed by registry policy: {ring}")
    checks.append("conclusive_ring_allowed")
    policy_refusal(
        visibility in acceptance["allowed-visibilities"],
        f"package visibility is not allowed by registry policy: {visibility}",
    )
    checks.append("visibility_allowed")
    policy_refusal(
        requested_class in acceptance["allowed-claim-classes"],
        f"claim class is not allowed by registry policy: {requested_class}",
    )
    checks.append("claim_class_allowed")
    policy_refusal(
        release.get("assurance_level") in acceptance["allowed-assurance-levels"],
        f"assurance level is not allowed by registry policy: {release.get('assurance_level')}",
    )
    checks.append("identity_assurance_allowed")
    policy_refusal(
        boundary.get("mode") == acceptance["required-boundary-mode"]
        and boundary.get("verdict") == acceptance["required-boundary-verdict"],
        "release boundary does not satisfy registry policy",
    )
    checks.append("strict_boundary_ok")

    registry = policy["registry"]
    payload = {
        "schema": ATTESTATION_SCHEMA,
        "attestation_type": ATTESTATION_TYPE,
        "registry": {
            "id": registry["id"],
            "namespace": registry["namespace"],
            "authority_scope": registry["authority-scope"],
            "publication_status": registry["publication-status"],
            "policy_sha256": sha256_file(policy_path),
        },
        "package": {
            "name": release["package"]["name"],
            "version": release["package"]["version"],
            "bundle_identity_sha256": release["bundle_identity_sha256"],
        },
        "release_bindings": {
            "bundle_manifest_sha256": sha256_file(bundle / "package-release.json"),
            "artifact_sha256": release["artifact"]["sha256"],
            "boundary_receipt_sha256": boundary["sha256"],
            "boundary_receipt_identity_sha256": boundary["identity_sha256"],
            "claim_contract_sha256": claim["sha256"],
            "claim_id": claim["claim_id"],
            "requested_class": requested_class,
            "source_bundle_sha256": release["bindings"]["source_bundle_sha256"],
            "package_policy_sha256": sha256_file(package_manifest_path),
            "compiler_sha256": release["bindings"]["compiler_sha256"],
        },
        "science": {
            "ring": ring,
            "evidence_status": evidence_status,
            "context_of_use": context_of_use,
            "visibility": visibility,
        },
        "decision": {"verdict": "POLICY_MATCH", "checks": checks},
        "assurance_level": "identity-only",
        "limitations": LIMITATIONS,
    }
    return with_identity(payload)


def read_attestation(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RegistryAttestationError("E-SRB-REGISTRY-003", f"attestation is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RegistryAttestationError("E-SRB-REGISTRY-003", f"cannot parse registry attestation: {error}") from error
    if not isinstance(value, dict):
        raise RegistryAttestationError("E-SRB-REGISTRY-003", "registry attestation must be a JSON object")
    return value


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise RegistryAttestationError("E-SRB-REGISTRY-004", f"attestation output already exists: {path}")
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".staging", dir=path.parent)
        temporary = Path(name)
        with os.fdopen(descriptor, "w", encoding="ascii", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise RegistryAttestationError("E-SRB-REGISTRY-004", f"attestation output appeared during write: {path}")
        os.rename(temporary, path)
        temporary = None
        parent_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def attest_command(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    compiler = Path(args.compiler).expanduser().resolve()
    policy = absolute_path(args.registry_policy)
    output_input = Path(args.output).expanduser()
    output = (Path.cwd() / output_input if not output_input.is_absolute() else output_input)
    output = output.parent.resolve() / output.name
    payload = expected_attestation(bundle, root, compiler, policy)
    write_atomic(output, payload)
    print(f"REGISTRY_ATTESTATION_PASS attestation={output}")
    return 0


def verify_command(args: argparse.Namespace) -> int:
    attestation_path = absolute_path(args.attestation)
    bundle = Path(args.bundle).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    compiler = Path(args.compiler).expanduser().resolve()
    policy = absolute_path(args.registry_policy)
    actual = read_attestation(attestation_path)
    if actual.get("schema") != ATTESTATION_SCHEMA:
        raise RegistryAttestationError("E-SRB-REGISTRY-003", "unsupported registry attestation schema")
    if actual.get("attestation_identity_sha256") != attestation_identity(actual):
        raise RegistryAttestationError("E-SRB-REGISTRY-003", "registry attestation identity hash mismatch")
    expected = expected_attestation(bundle, root, compiler, policy)
    if actual != expected:
        raise RegistryAttestationError("E-SRB-REGISTRY-003", "registry attestation bindings do not match inputs")
    print(f"REGISTRY_ATTESTATION_VERIFY_PASS attestation={attestation_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-registry-attestation")
    subparsers = result.add_subparsers(dest="command", required=True)
    attest = subparsers.add_parser("attest")
    attest.add_argument("--bundle", required=True)
    attest.add_argument("--root", required=True)
    attest.add_argument("--compiler", required=True)
    attest.add_argument("--registry-policy", required=True)
    attest.add_argument("--output", required=True)
    attest.set_defaults(handler=attest_command)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--attestation", required=True)
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--root", required=True)
    verify.add_argument("--compiler", required=True)
    verify.add_argument("--registry-policy", required=True)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except RegistryAttestationError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"REGISTRY_ATTESTATION_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError) as error:
        print(f"error[E-SRB-REGISTRY-005]: registry attestation operation failed: {error}", file=sys.stderr)
        print(f"REGISTRY_ATTESTATION_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
