#!/usr/bin/env python3
"""Build and verify atomic, identity-only Sounio package release bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from typing import Any


BUNDLE_SCHEMA = "sounio.package-release-bundle.v1"
RECEIPT_SCHEMA = "sounio.package-boundary-receipt.v1"
ATTESTOR = Path(__file__).with_name("attestor.py")
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
LIMITATIONS = [
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_public_registry_status",
    "does_not_assert_attested_execution_or_independent_replay",
    "full_verification_requires_original_sources_policy_and_compiler",
]


class ReleaseError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def bundle_identity(payload: dict[str, Any]) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload.pop("bundle_identity_sha256", None)
    return hashlib.sha256(canonical_json(identity_payload)).hexdigest()


def write_bundle_manifest(path: Path, payload: dict[str, Any]) -> None:
    payload["bundle_identity_sha256"] = bundle_identity(payload)
    with path.open("w", encoding="ascii", newline="\n") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def safe_relative(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ReleaseError("E-SRB-RELEASE-005", f"unsafe bundle path: {value}")
    return path.as_posix()


def require_regular_file(root: Path, relative: str) -> Path:
    relative = safe_relative(relative)
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as error:
        raise ReleaseError("E-SRB-RELEASE-005", f"bundle path escapes root: {relative}") from error
    if path.is_symlink() or not path.is_file():
        raise ReleaseError("E-SRB-RELEASE-005", f"bundle entry is missing or not regular: {relative}")
    return path


def load_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError("E-SRB-RELEASE-001", f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError("E-SRB-RELEASE-001", f"{label} must contain a top-level table")
    return value


def load_verified_toml(path: Path, label: str) -> dict[str, Any]:
    try:
        with path.open("rb") as handle:
            value = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ReleaseError("E-SRB-RELEASE-005", f"cannot parse bundled {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError("E-SRB-RELEASE-005", f"bundled {label} must contain a top-level table")
    return value


def resolve_project(candidate: str) -> tuple[Path, Path, dict[str, Any], Path]:
    path = Path(candidate).expanduser() if candidate else Path.cwd()
    path = path.resolve()
    manifest = path / "sounio.toml" if path.is_dir() else path
    if manifest.name != "sounio.toml" or not manifest.is_file():
        raise ReleaseError("E-SRB-RELEASE-001", f"package manifest not found: {manifest}")
    root = manifest.parent.resolve()
    data = load_toml(manifest, "package manifest")
    package = data.get("package")
    if not isinstance(package, dict):
        raise ReleaseError("E-SRB-RELEASE-001", "sounio.toml is missing [package]")
    name = str(package.get("name", "")).strip()
    version = str(package.get("version", "")).strip()
    if not SAFE_COMPONENT.fullmatch(name) or not SAFE_COMPONENT.fullmatch(version):
        raise ReleaseError("E-SRB-RELEASE-001", "package name and version must be safe path components")

    entry_text = ""
    bins = data.get("bin", [])
    if isinstance(bins, list) and bins and isinstance(bins[0], dict):
        entry_text = str(bins[0].get("path", ""))
    if not entry_text and isinstance(data.get("project"), dict):
        entry_text = str(data["project"].get("entry", ""))
    if not entry_text and isinstance(data.get("lib"), dict):
        entry_text = str(data["lib"].get("path", ""))
    entry_text = entry_text or "src/main.sio"
    safe_entry = safe_relative(entry_text)
    entry = (root / safe_entry).resolve()
    try:
        entry.relative_to(root)
    except ValueError as error:
        raise ReleaseError("E-SRB-RELEASE-001", f"package entry escapes project root: {entry_text}") from error
    if not entry.is_file():
        raise ReleaseError("E-SRB-RELEASE-001", f"package entry not found: {safe_entry}")
    return root, manifest, data, entry


def validate_policy_argument(explicit: str, manifest: Path) -> None:
    if explicit and Path(explicit).expanduser().resolve() != manifest.resolve():
        raise ReleaseError(
            "E-SRB-RELEASE-002",
            "strict package releases require the package sounio.toml as their policy root",
        )


def validate_claim_for_package(claim_path: Path, manifest: dict[str, Any], root: Path, entry: Path) -> None:
    contract = load_toml(claim_path, "claim contract")
    science = manifest.get("science")
    if not isinstance(science, dict):
        raise ReleaseError("E-SRB-RELEASE-002", "strict package release requires a [science] policy")
    allowed = science.get("allowed-claim-classes", science.get("allowed_claim_classes", []))
    if not isinstance(allowed, list) or not all(isinstance(item, str) and item for item in allowed):
        raise ReleaseError("E-SRB-RELEASE-002", "package policy has no valid allowed-claim-classes")
    requested = str(contract.get("requested-class", "")).strip()
    if contract.get("schema") != "sounio.claim-contract.v1" or not requested:
        raise ReleaseError("E-SRB-RELEASE-002", "claim contract schema or requested-class is invalid")
    if requested not in allowed:
        raise ReleaseError(
            "E-SRB-RELEASE-002",
            f"requested claim class is not allowed by the package policy: {requested}",
        )
    context = str(contract.get("context-of-use", "")).strip()
    if context != str(science.get("context-of-use", science.get("context_of_use", ""))).strip():
        raise ReleaseError("E-SRB-RELEASE-002", "claim context-of-use does not match the package policy")
    entry_relative = entry.relative_to(root).as_posix()
    if str(contract.get("root-artifact", "")).strip() != entry_relative:
        raise ReleaseError("E-SRB-RELEASE-002", "claim root-artifact does not match the package entrypoint")


def read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReleaseError("E-SRB-RELEASE-005", f"cannot parse {label}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseError("E-SRB-RELEASE-005", f"{label} must be a JSON object")
    return value


def validate_receipt_for_release(receipt: dict[str, Any], artifact_label: str) -> None:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ReleaseError("E-SRB-RELEASE-003", "build emitted an unsupported boundary receipt")
    if receipt.get("mode") != "strict" or receipt.get("verdict") != "OK":
        raise ReleaseError("E-SRB-RELEASE-003", "release promotion requires a strict OK receipt")
    graph = receipt.get("graph")
    if not isinstance(graph, dict):
        raise ReleaseError("E-SRB-RELEASE-003", "release receipt has no closure graph")
    if graph.get("saturated") or graph.get("unresolved_imports"):
        raise ReleaseError("E-SRB-RELEASE-003", "release promotion requires a complete raw-AST closure")
    engine = receipt.get("engine")
    if not isinstance(engine, dict) or engine.get("boundary_collector") != "madaros-raw-ast-v1":
        raise ReleaseError("E-SRB-RELEASE-003", "release promotion requires the raw Madaros AST collector")
    if not isinstance(receipt.get("claim_contract"), dict):
        raise ReleaseError("E-SRB-RELEASE-003", "release receipt is not bound to a claim contract")
    artifact = receipt.get("artifact")
    if not isinstance(artifact, dict) or artifact != {"kind": "native-elf", "path": artifact_label}:
        raise ReleaseError("E-SRB-RELEASE-003", "release receipt has the wrong artifact binding")


def run_receipt_verify(receipt: Path, root: Path, compiler: Path, elf: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ATTESTOR),
            "verify",
            "--receipt",
            str(receipt),
            "--root",
            str(root),
            "--compiler",
            str(compiler),
            "--elf",
            str(elf),
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseError("E-SRB-RELEASE-005", "package boundary receipt revalidation failed")


def validate_bundle(bundle: Path, root: Path, compiler: Path) -> dict[str, Any]:
    if bundle.is_symlink() or not bundle.is_dir():
        raise ReleaseError("E-SRB-RELEASE-005", f"release bundle is not a directory: {bundle}")
    manifest_path = require_regular_file(bundle, "package-release.json")
    data = read_json(manifest_path, "package-release.json")
    required = {
        "schema",
        "bundle_identity_sha256",
        "package",
        "artifact",
        "boundary_receipt",
        "claim_contract",
        "bindings",
        "assurance_level",
        "limitations",
    }
    if set(data) != required or data.get("schema") != BUNDLE_SCHEMA:
        raise ReleaseError("E-SRB-RELEASE-005", "release manifest fields do not match schema v1")
    if data.get("bundle_identity_sha256") != bundle_identity(data):
        raise ReleaseError("E-SRB-RELEASE-005", "release bundle identity hash mismatch")
    if data.get("assurance_level") != "identity-only" or data.get("limitations") != LIMITATIONS:
        raise ReleaseError("E-SRB-RELEASE-005", "release bundle overstates its assurance level")

    package = data.get("package")
    artifact = data.get("artifact")
    boundary = data.get("boundary_receipt")
    claim = data.get("claim_contract")
    bindings = data.get("bindings")
    if not all(isinstance(item, dict) for item in (package, artifact, boundary, claim, bindings)):
        raise ReleaseError("E-SRB-RELEASE-005", "release manifest contains malformed bindings")
    if set(package) != {"name", "version", "manifest", "manifest_sha256"}:
        raise ReleaseError("E-SRB-RELEASE-005", "bad package metadata binding")
    if set(artifact) != {"kind", "path", "sha256"} or artifact.get("kind") != "native-elf":
        raise ReleaseError("E-SRB-RELEASE-005", "bad release artifact binding")
    if set(boundary) != {"path", "sha256", "identity_sha256", "mode", "verdict"}:
        raise ReleaseError("E-SRB-RELEASE-005", "bad boundary receipt binding")
    if set(claim) != {"path", "sha256", "claim_id", "requested_class"}:
        raise ReleaseError("E-SRB-RELEASE-005", "bad claim contract binding")
    if set(bindings) != {"source_bundle_sha256", "policy_sha256", "compiler_sha256"}:
        raise ReleaseError("E-SRB-RELEASE-005", "bad release identity bindings")
    expected_artifact_path = f"artifacts/{package.get('name', '')}"
    if artifact.get("path") != expected_artifact_path:
        raise ReleaseError("E-SRB-RELEASE-005", "release artifact path is not canonical")
    if boundary.get("path") != "attestations/package-boundary-receipt.json":
        raise ReleaseError("E-SRB-RELEASE-005", "boundary receipt path is not canonical")
    if claim.get("path") != "claims/claim-contract.toml":
        raise ReleaseError("E-SRB-RELEASE-005", "claim contract path is not canonical")

    artifact_path = require_regular_file(bundle, str(artifact.get("path", "")))
    receipt_path = require_regular_file(bundle, str(boundary.get("path", "")))
    claim_path = require_regular_file(bundle, str(claim.get("path", "")))
    expected_files = {
        "package-release.json",
        safe_relative(str(artifact["path"])),
        safe_relative(str(boundary["path"])),
        safe_relative(str(claim["path"])),
    }
    actual_files = {
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual_files != expected_files:
        raise ReleaseError("E-SRB-RELEASE-005", "release bundle file inventory mismatch")
    for path, expected, label in (
        (artifact_path, artifact.get("sha256"), "artifact"),
        (receipt_path, boundary.get("sha256"), "receipt"),
        (claim_path, claim.get("sha256"), "claim contract"),
    ):
        if sha256_file(path) != expected:
            raise ReleaseError("E-SRB-RELEASE-005", f"release {label} hash mismatch")
    if not os.access(artifact_path, os.X_OK):
        raise ReleaseError("E-SRB-RELEASE-005", "release native artifact is not executable")

    receipt = read_json(receipt_path, "boundary receipt")
    validate_receipt_for_release(receipt, str(artifact["path"]))
    hashes = receipt.get("hashes", {})
    summary = receipt.get("claim_contract", {})
    if boundary.get("identity_sha256") != hashes.get("receipt_identity_sha256"):
        raise ReleaseError("E-SRB-RELEASE-005", "receipt identity binding mismatch")
    if boundary.get("mode") != "strict" or boundary.get("verdict") != "OK":
        raise ReleaseError("E-SRB-RELEASE-005", "release manifest does not bind a strict OK verdict")
    if claim.get("sha256") != hashes.get("claim_contract_sha256"):
        raise ReleaseError("E-SRB-RELEASE-005", "claim-to-receipt hash mismatch")
    if claim.get("claim_id") != summary.get("claim_id") or claim.get("requested_class") != summary.get("requested_class"):
        raise ReleaseError("E-SRB-RELEASE-005", "claim summary binding mismatch")
    claim_contract = load_verified_toml(claim_path, "claim contract")
    if claim_contract.get("schema") != "sounio.claim-contract.v1":
        raise ReleaseError("E-SRB-RELEASE-005", "bundled claim contract has the wrong schema")
    if (
        claim_contract.get("claim-id") != summary.get("claim_id")
        or claim_contract.get("requested-class") != summary.get("requested_class")
        or claim_contract.get("context-of-use") != summary.get("context_of_use")
        or claim_contract.get("root-artifact") != summary.get("root_artifact")
    ):
        raise ReleaseError("E-SRB-RELEASE-005", "bundled claim content does not match the receipt summary")
    claim_evidence = claim_contract.get("evidence", [])
    expected_bindings = sorted(
        [
            {"type": str(item.get("type", "")), "ref": str(item.get("ref", "")), "sha256": str(item.get("sha256", ""))}
            for item in claim_evidence
            if isinstance(item, dict)
        ],
        key=lambda item: (item["type"], item["ref"], item["sha256"]),
    )
    if expected_bindings != summary.get("evidence_bindings"):
        raise ReleaseError("E-SRB-RELEASE-005", "bundled claim evidence does not match the receipt bindings")
    if artifact.get("sha256") != hashes.get("elf_sha256"):
        raise ReleaseError("E-SRB-RELEASE-005", "artifact-to-receipt hash mismatch")
    for key in ("source_bundle_sha256", "policy_sha256", "compiler_sha256"):
        if bindings.get(key) != hashes.get(key):
            raise ReleaseError("E-SRB-RELEASE-005", f"receipt binding mismatch: {key}")

    package_manifest = require_regular_file(root, str(package.get("manifest", "")))
    if sha256_file(package_manifest) != package.get("manifest_sha256"):
        raise ReleaseError("E-SRB-RELEASE-005", "package manifest hash mismatch")
    if package.get("manifest_sha256") != bindings.get("policy_sha256"):
        raise ReleaseError("E-SRB-RELEASE-005", "package manifest is not the bound release policy")
    package_toml = load_verified_toml(package_manifest, "package manifest")
    package_table = package_toml.get("package")
    if (
        not isinstance(package_table, dict)
        or package_table.get("name") != package.get("name")
        or package_table.get("version") != package.get("version")
    ):
        raise ReleaseError("E-SRB-RELEASE-005", "bundle package identity does not match sounio.toml")
    run_receipt_verify(receipt_path, root, compiler, artifact_path)
    return data


def fsync_tree(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())
    for path in sorted((item for item in root.rglob("*") if item.is_dir()), reverse=True):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_command(args: argparse.Namespace) -> int:
    root, manifest, data, entry = resolve_project(args.project)
    validate_policy_argument(args.manifest, manifest)
    compiler = Path(args.compiler).expanduser().resolve()
    launcher = Path(args.launcher).expanduser().resolve()
    claim_source = Path(args.claim_contract).expanduser().resolve()
    if not compiler.is_file() or not launcher.is_file():
        raise ReleaseError("E-SRB-RELEASE-001", "Madaros compiler or launcher is unavailable")
    if not claim_source.is_file():
        raise ReleaseError("E-SRB-RELEASE-002", f"claim contract not found: {claim_source}")
    try:
        claim_source.relative_to(root)
    except ValueError as error:
        raise ReleaseError("E-SRB-RELEASE-002", "claim contract must be inside the package policy root") from error
    validate_claim_for_package(claim_source, data, root, entry)

    package = data["package"]
    name = str(package["name"])
    version = str(package["version"])
    default_bundle = (root / "target" / "release" / f"{name}-{version}.sio-release").resolve()
    final_bundle = Path(args.release_bundle).expanduser().resolve() if args.release_bundle else default_bundle
    if final_bundle.exists() or final_bundle.is_symlink():
        raise ReleaseError("E-SRB-RELEASE-004", f"release bundle already exists: {final_bundle}")
    final_bundle.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final_bundle.name}.staging-", dir=final_bundle.parent))
    promoted = False
    try:
        artifact_rel = f"artifacts/{name}"
        receipt_rel = "attestations/package-boundary-receipt.json"
        claim_rel = "claims/claim-contract.toml"
        artifact = staging / artifact_rel
        receipt_path = staging / receipt_rel
        claim_copy = staging / claim_rel
        artifact.parent.mkdir(parents=True)
        receipt_path.parent.mkdir(parents=True)
        claim_copy.parent.mkdir(parents=True)

        environment = os.environ.copy()
        environment["MADAROS_RAW_BIN"] = str(compiler)
        environment["SOUNIO_INTERNAL_PACKAGE_RELEASE"] = "1"
        environment["SOUNIO_SCIENCE_ARTIFACT_LABEL"] = artifact_rel
        result = subprocess.run(
            [
                str(launcher),
                "--science-boundary",
                "strict",
                "--science-manifest",
                str(manifest),
                "--claim-contract",
                str(claim_source),
                "--emit-boundary-receipt",
                str(receipt_path),
                "build",
                str(root),
                "-o",
                str(artifact),
            ],
            cwd=root,
            env=environment,
            check=False,
        )
        if result.returncode != 0:
            raise ReleaseError("E-SRB-RELEASE-003", f"strict package build refused with status {result.returncode}")
        if artifact.is_symlink() or not artifact.is_file() or artifact.stat().st_size == 0:
            raise ReleaseError("E-SRB-RELEASE-003", "strict package build emitted no native artifact")
        if receipt_path.is_symlink() or not receipt_path.is_file():
            raise ReleaseError("E-SRB-RELEASE-003", "strict package build emitted no boundary receipt")
        shutil.copyfile(claim_source, claim_copy)

        receipt = read_json(receipt_path, "boundary receipt")
        validate_receipt_for_release(receipt, artifact_rel)
        hashes = receipt["hashes"]
        claim_summary = receipt["claim_contract"]
        payload = {
            "schema": BUNDLE_SCHEMA,
            "package": {
                "name": name,
                "version": version,
                "manifest": "sounio.toml",
                "manifest_sha256": sha256_file(manifest),
            },
            "artifact": {"kind": "native-elf", "path": artifact_rel, "sha256": sha256_file(artifact)},
            "boundary_receipt": {
                "path": receipt_rel,
                "sha256": sha256_file(receipt_path),
                "identity_sha256": hashes["receipt_identity_sha256"],
                "mode": receipt["mode"],
                "verdict": receipt["verdict"],
            },
            "claim_contract": {
                "path": claim_rel,
                "sha256": sha256_file(claim_copy),
                "claim_id": claim_summary["claim_id"],
                "requested_class": claim_summary["requested_class"],
            },
            "bindings": {
                "source_bundle_sha256": hashes["source_bundle_sha256"],
                "policy_sha256": hashes["policy_sha256"],
                "compiler_sha256": hashes["compiler_sha256"],
            },
            "assurance_level": "identity-only",
            "limitations": LIMITATIONS,
        }
        write_bundle_manifest(staging / "package-release.json", payload)
        validate_bundle(staging, root, compiler)
        fsync_tree(staging)
        if final_bundle.exists() or final_bundle.is_symlink():
            raise ReleaseError("E-SRB-RELEASE-004", f"release bundle appeared during build: {final_bundle}")
        os.rename(staging, final_bundle)
        promoted = True
        parent_descriptor = os.open(final_bundle.parent, os.O_RDONLY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)
    print(f"PACKAGE_BOUNDARY_RELEASE_PASS bundle={final_bundle}")
    return 0


def verify_command(args: argparse.Namespace) -> int:
    bundle = Path(args.bundle).expanduser().resolve()
    root = Path(args.root).expanduser().resolve()
    compiler = Path(args.compiler).expanduser().resolve()
    validate_bundle(bundle, root, compiler)
    print(f"PACKAGE_BOUNDARY_RELEASE_VERIFY_PASS bundle={bundle}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-package-release")
    subparsers = result.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--launcher", required=True)
    build.add_argument("--compiler", required=True)
    build.add_argument("--claim-contract", required=True)
    build.add_argument("--manifest", default="")
    build.add_argument("--release-bundle", default="")
    build.add_argument("project", nargs="?", default="")
    build.set_defaults(handler=build_command)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--bundle", required=True)
    verify.add_argument("--root", default=str(Path.cwd()))
    verify.add_argument("--compiler", required=True)
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except ReleaseError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PACKAGE_BOUNDARY_RELEASE_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError, subprocess.SubprocessError) as error:
        print(f"error[E-SRB-RELEASE-005]: release operation failed: {error}", file=sys.stderr)
        print(f"PACKAGE_BOUNDARY_RELEASE_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
