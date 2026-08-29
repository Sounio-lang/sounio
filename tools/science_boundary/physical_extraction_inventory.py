#!/usr/bin/env python3
"""Emit and verify deterministic R3 physical-extraction planning inventories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tempfile
from typing import Any


INVENTORY_SCHEMA = "sounio.physical-extraction-inventory.v1"
INVENTORY_TYPE = "physical-extraction-planning-snapshot"
AUTHORITY_SCOPE = "repository-file-identity-and-ownership-plan"
EXTRACTION_STATUS = "not-executed"
ASSURANCE_LEVEL = "identity-only"
CONCLUSIVE_RINGS = {"pl-core", "scientific-package", "research"}
KNOWN_RINGS = CONCLUSIVE_RINGS | {
    "scientific-package-candidate",
    "mixed-unresolved",
    "unclassified",
}
VISIBILITIES = {"public", "protected", "embargoed"}
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
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
LIMITATIONS = [
    "does_not_move_or_delete_source_files",
    "does_not_assert_target_repository_or_distribution_exists",
    "does_not_assert_ownership_or_maintainership_was_transferred",
    "does_not_assert_publication_or_registry_status",
    "does_not_assert_scientific_truth",
    "does_not_assert_clinical_validation_or_clinical_authority",
    "does_not_assert_independent_replay",
    "full_verification_requires_the_original_repository_snapshot_and_policies",
]


class PhysicalExtractionError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def inventory_identity(payload: dict[str, Any]) -> str:
    identity_payload = json.loads(json.dumps(payload))
    identity_payload.pop("inventory_identity_sha256", None)
    return sha256_bytes(canonical_json(identity_payload))


def with_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(payload))
    result["inventory_identity_sha256"] = inventory_identity(result)
    return result


def within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def resolve_input(repo_root: Path, value: str, label: str) -> Path:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    if candidate.is_symlink():
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} must not be a symbolic link")
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"cannot resolve {label}: {error}") from error
    if not within_root(resolved, repo_root) or not resolved.is_file():
        raise PhysicalExtractionError(
            "E-SRB-EXTRACT-001",
            f"{label} must be a regular file inside the repository root",
        )
    return resolved


def read_regular_bytes(path: Path, label: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} is not a regular file: {path}")
    try:
        return path.read_bytes()
    except OSError as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"cannot read {label}: {error}") from error


def load_tsv(path: Path, expected_fields: list[str], label: str) -> tuple[list[dict[str, str]], str]:
    raw = read_regular_bytes(path, label)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} must be UTF-8") from error
    reader = csv.DictReader(io.StringIO(text), delimiter="\t")
    if reader.fieldnames != expected_fields:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} header does not match v1")
    rows: list[dict[str, str]] = []
    for number, row in enumerate(reader, start=2):
        if None in row or any(value is None for value in row.values()):
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} row {number} has the wrong field count")
        normalized = {key: value.strip() for key, value in row.items()}
        if not any(normalized.values()):
            continue
        rows.append(normalized)
    if not rows:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} has no rows")
    return rows, sha256_bytes(raw)


def normalize_source_path(value: str, label: str) -> str:
    if not value or "\\" in value:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} is not a normalized repository path")
    path = PurePosixPath(value)
    if path.is_absolute() or value in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} is not a normalized repository path")
    if path.as_posix() != value:
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} is not canonical")
    return value


def split_refs(value: str, label: str, *, allow_empty: bool = False) -> list[str]:
    if not value:
        if allow_empty:
            return []
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} is empty")
    refs = value.split("|")
    if any(not SAFE_ID.fullmatch(item) for item in refs) or len(refs) != len(set(refs)):
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"{label} contains invalid or duplicate references")
    return refs


def validate_science_rows(rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for number, row in enumerate(rows, start=2):
        source_path = normalize_source_path(row["path"], f"science rings row {number} path")
        if source_path in result:
            raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"duplicate science ring root: {source_path}")
        if row["ring"] not in KNOWN_RINGS:
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"unsupported science ring: {row['ring']}")
        if row["visibility"] not in VISIBILITIES:
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"unsupported visibility: {row['visibility']}")
        required = [
            "evidence_status",
            "context_of_use",
            "enforcement",
            "next_gate",
            "evidence_refs",
            "declared_by",
            "declared_at",
            "review_state",
        ]
        if any(not row[field] for field in required):
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"science rings row {number} has an empty field")
        result[source_path] = {
            **row,
            "allowed_claim_classes": split_refs(
                row["allowed_claim_classes"],
                f"science rings row {number} allowed claim classes",
                allow_empty=True,
            ),
            "evidence_refs": split_refs(row["evidence_refs"], f"science rings row {number} evidence refs"),
        }
    paths = [PurePosixPath(path) for path in result]
    for index, left in enumerate(paths):
        for right in paths[index + 1 :]:
            if left.parts == right.parts[: len(left.parts)] or right.parts == left.parts[: len(right.parts)]:
                raise PhysicalExtractionError(
                    "E-SRB-EXTRACT-002",
                    f"science ring roots overlap: {left.as_posix()} and {right.as_posix()}",
                )
    return result


def expected_disposition(ring: str) -> tuple[str, str, str]:
    if ring == "pl-core":
        return "same-repository", "retain-core", "retained"
    if ring in {"scientific-package", "research"}:
        return "separate-distribution", "extract-planned", "planned"
    return "unassigned", "hold-unresolved", "blocked-classification"


def validate_ownership_rows(
    rows: list[dict[str, str]],
    science: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    extraction_targets: set[str] = set()
    for number, row in enumerate(rows, start=2):
        source_path = normalize_source_path(row["source_path"], f"ownership row {number} source_path")
        if source_path in result:
            raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"duplicate ownership root: {source_path}")
        if source_path not in science:
            raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"ownership root is absent from science rings: {source_path}")
        if row["ring"] != science[source_path]["ring"]:
            raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"ownership ring mismatch for {source_path}")
        for field in ("current_owner", "target_owner"):
            if not SAFE_TOKEN.fullmatch(row[field]):
                raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"invalid {field} for {source_path}")
        if not SAFE_ID.fullmatch(row["target_id"]):
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"invalid target_id for {source_path}")
        expected_kind, expected_action, expected_state = expected_disposition(row["ring"])
        actual = (row["target_kind"], row["disposition"], row["migration_state"])
        if actual != (expected_kind, expected_action, expected_state):
            raise PhysicalExtractionError(
                "E-SRB-EXTRACT-002",
                f"ownership disposition is invalid for {source_path} ring {row['ring']}",
            )
        if expected_action == "retain-core":
            if row["target_owner"] != row["current_owner"] or row["target_id"] != "repo:sounio":
                raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"retained core target is invalid for {source_path}")
        elif expected_action == "extract-planned":
            if row["target_owner"] == "unassigned" or row["target_id"] == "unassigned":
                raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"planned extraction lacks a target for {source_path}")
            if row["target_id"] in extraction_targets:
                raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"duplicate extraction target: {row['target_id']}")
            extraction_targets.add(row["target_id"])
        elif row["target_owner"] != "unassigned" or row["target_id"] != "unassigned":
            raise PhysicalExtractionError("E-SRB-EXTRACT-002", f"unresolved root has an assigned target: {source_path}")
        if not SAFE_ID.fullmatch(row["extraction_gate"]):
            raise PhysicalExtractionError("E-SRB-EXTRACT-001", f"invalid extraction_gate for {source_path}")
        result[source_path] = {
            **row,
            "ownership_evidence": split_refs(
                row["ownership_evidence"],
                f"ownership row {number} evidence",
            ),
        }
    missing = sorted(set(science) - set(result))
    extra = sorted(set(result) - set(science))
    if missing or extra:
        raise PhysicalExtractionError(
            "E-SRB-EXTRACT-002",
            f"ownership coverage mismatch missing={','.join(missing) or '-'} extra={','.join(extra) or '-'}",
        )
    return result


def stable_file_identity(path: Path) -> tuple[int, str]:
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"cannot open inventory file {path}: {error}") from error
    digest = hashlib.sha256()
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"inventory member is not a regular file: {path}")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    try:
        current = path.lstat()
    except OSError as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"inventory member disappeared: {path}") from error
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    identity_current = (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns)
    if identity_before != identity_after or identity_after != identity_current:
        raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"inventory member changed while hashing: {path}")
    return before.st_size, digest.hexdigest()


def scan_unit(repo_root: Path, source_path: str) -> tuple[list[dict[str, Any]], int, str]:
    unit_root = repo_root / PurePosixPath(source_path)
    if unit_root.is_symlink() or not unit_root.is_dir():
        raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"source root is not a regular directory: {source_path}")
    files: list[dict[str, Any]] = []
    for current, directories, names in os.walk(unit_root, topdown=True, followlinks=False):
        current_path = Path(current)
        for directory in directories:
            child = current_path / directory
            if child.is_symlink():
                raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"symbolic-link directory is not inventory-safe: {child}")
        directories.sort()
        for name in sorted(names):
            path = current_path / name
            if path.is_symlink():
                raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"symbolic-link file is not inventory-safe: {path}")
            size, digest = stable_file_identity(path)
            relative = path.relative_to(repo_root).as_posix()
            files.append({"path": relative, "size_bytes": size, "sha256": digest})
    if not files:
        raise PhysicalExtractionError("E-SRB-EXTRACT-003", f"source root has no regular files: {source_path}")
    total_bytes = sum(item["size_bytes"] for item in files)
    return files, total_bytes, sha256_bytes(canonical_json(files))


def expected_inventory(repo_root: Path, rings_path: Path, ownership_path: Path) -> dict[str, Any]:
    science_rows, rings_sha256 = load_tsv(rings_path, SCIENCE_FIELDS, "science rings")
    ownership_rows, ownership_sha256 = load_tsv(ownership_path, OWNERSHIP_FIELDS, "ownership policy")
    science = validate_science_rows(science_rows)
    ownership = validate_ownership_rows(ownership_rows, science)
    units: list[dict[str, Any]] = []
    for source_path in sorted(science):
        ring_row = science[source_path]
        owner_row = ownership[source_path]
        files, total_bytes, tree_sha256 = scan_unit(repo_root, source_path)
        units.append(
            {
                "source_path": source_path,
                "ring": ring_row["ring"],
                "evidence_status": ring_row["evidence_status"],
                "context_of_use": ring_row["context_of_use"],
                "visibility": ring_row["visibility"],
                "ring_next_gate": ring_row["next_gate"],
                "allowed_claim_classes": ring_row["allowed_claim_classes"],
                "ring_evidence_refs": ring_row["evidence_refs"],
                "current_owner": owner_row["current_owner"],
                "target_kind": owner_row["target_kind"],
                "target_id": owner_row["target_id"],
                "target_owner": owner_row["target_owner"],
                "disposition": owner_row["disposition"],
                "migration_state": owner_row["migration_state"],
                "ownership_evidence": owner_row["ownership_evidence"],
                "extraction_gate": owner_row["extraction_gate"],
                "file_count": len(files),
                "total_bytes": total_bytes,
                "tree_sha256": tree_sha256,
                "files": files,
            }
        )
    summary = {
        "unit_count": len(units),
        "file_count": sum(unit["file_count"] for unit in units),
        "total_bytes": sum(unit["total_bytes"] for unit in units),
        "retained_core_units": sum(unit["disposition"] == "retain-core" for unit in units),
        "planned_extraction_units": sum(unit["disposition"] == "extract-planned" for unit in units),
        "blocked_units": sum(unit["disposition"] == "hold-unresolved" for unit in units),
    }
    payload = {
        "schema": INVENTORY_SCHEMA,
        "inventory_type": INVENTORY_TYPE,
        "authority_scope": AUTHORITY_SCOPE,
        "extraction_status": EXTRACTION_STATUS,
        "source_documents": {
            "science_rings_sha256": rings_sha256,
            "ownership_policy_sha256": ownership_sha256,
        },
        "summary": summary,
        "units": units,
        "assurance_level": ASSURANCE_LEVEL,
        "limitations": LIMITATIONS,
    }
    return with_identity(payload)


def read_inventory(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", f"inventory is not a regular file: {path}")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", f"cannot parse physical inventory: {error}") from error
    if not isinstance(value, dict):
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", "physical inventory must be a JSON object")
    return value


def write_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise PhysicalExtractionError("E-SRB-EXTRACT-005", f"inventory output already exists: {path}")
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
            raise PhysicalExtractionError("E-SRB-EXTRACT-005", f"inventory output appeared during write: {path}")
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


def resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repo_root = Path(args.repo_root).expanduser().resolve(strict=True)
    if not repo_root.is_dir():
        raise PhysicalExtractionError("E-SRB-EXTRACT-001", "repository root is not a directory")
    rings = resolve_input(repo_root, args.rings, "science rings")
    ownership = resolve_input(repo_root, args.ownership, "ownership policy")
    return repo_root, rings, ownership


def inventory_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership = resolve_inputs(args)
    output_input = Path(args.output).expanduser()
    output = output_input if output_input.is_absolute() else Path.cwd() / output_input
    payload = expected_inventory(repo_root, rings, ownership)
    write_atomic(output, payload)
    print(
        "PHYSICAL_EXTRACTION_INVENTORY_PASS "
        f"inventory={output} units={payload['summary']['unit_count']} "
        f"files={payload['summary']['file_count']} status={payload['extraction_status']}"
    )
    return 0


def verify_command(args: argparse.Namespace) -> int:
    repo_root, rings, ownership = resolve_inputs(args)
    inventory_path = Path(args.inventory).expanduser().resolve(strict=True)
    actual = read_inventory(inventory_path)
    if actual.get("schema") != INVENTORY_SCHEMA:
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", "unsupported physical inventory schema")
    if actual.get("inventory_identity_sha256") != inventory_identity(actual):
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", "physical inventory identity hash mismatch")
    expected = expected_inventory(repo_root, rings, ownership)
    if actual != expected:
        raise PhysicalExtractionError("E-SRB-EXTRACT-004", "physical inventory bindings do not match inputs")
    print(f"PHYSICAL_EXTRACTION_INVENTORY_VERIFY_PASS inventory={inventory_path}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(prog="sounio-physical-extraction-inventory")
    subparsers = result.add_subparsers(dest="command", required=True)
    inventory = subparsers.add_parser("inventory")
    inventory.add_argument("--repo-root", required=True)
    inventory.add_argument("--rings", default="science-rings.tsv")
    inventory.add_argument(
        "--ownership",
        default="docs/ecosystem/science-physical-extraction-ownership.tsv",
    )
    inventory.add_argument("--output", required=True)
    inventory.set_defaults(handler=inventory_command)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--inventory", required=True)
    verify.add_argument("--repo-root", required=True)
    verify.add_argument("--rings", default="science-rings.tsv")
    verify.add_argument(
        "--ownership",
        default="docs/ecosystem/science-physical-extraction-ownership.tsv",
    )
    verify.set_defaults(handler=verify_command)
    return result


def main(argv: list[str]) -> int:
    try:
        args = parser().parse_args(argv[1:])
        return args.handler(args)
    except PhysicalExtractionError as error:
        print(f"error[{error.code}]: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_INVENTORY_REFUSED reason={error}", file=sys.stderr)
        return 1
    except (KeyError, OSError, TypeError) as error:
        print(f"error[E-SRB-EXTRACT-006]: physical inventory operation failed: {error}", file=sys.stderr)
        print(f"PHYSICAL_EXTRACTION_INVENTORY_REFUSED reason={error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
