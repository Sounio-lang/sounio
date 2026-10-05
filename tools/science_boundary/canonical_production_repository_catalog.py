#!/usr/bin/env python3
"""Build and verify deterministic repository catalogs from GitHub observations."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "science_boundary"))
import canonical_production_gap_assessor as gap_tool  # noqa: E402


CATALOG_SCHEMA = "sounio.physical-extraction-canonical-production-repository-catalog.v1"
CATALOG_TYPE = "observed-hosting-repository-catalog"
AUTHORITY_SCOPE = "supplied-repository-metadata-observation"
SAFE_TOKEN = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]*$")
GITHUB_LOGIN = re.compile(r"^(?=.{1,39}$)[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?$")
GIT_OID = re.compile(r"^[0-9a-f]{40}$")
VISIBILITIES = {"PUBLIC", "PRIVATE", "INTERNAL"}
PERMISSIONS = {"ADMIN", "MAINTAIN", "WRITE", "TRIAGE", "READ", "NONE", "UNKNOWN"}
LIMITATIONS = [
    "catalog_is_a_supplied_point_in_time_observation",
    "catalog_does_not_prove_repository_ownership_or_administration",
    "catalog_does_not_prove_branch_protection_or_push_acceptance",
    "catalog_does_not_approve_target_mapping_or_source_removal",
    "catalog_does_not_assert_scientific_truth",
]


class CatalogError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def catalog_identity(payload: dict[str, Any]) -> str:
    value = json.loads(json.dumps(payload))
    value.pop("catalog_identity_sha256", None)
    return sha256_bytes(canonical_json(value))


def render(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n").encode("ascii")


def exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise CatalogError(
            "E-SRB-PROD-CATALOG-002",
            f"{label} fields mismatch missing={','.join(sorted(expected - actual)) or '-'} "
            f"extra={','.join(sorted(actual - expected)) or '-'}",
        )


def read_regular_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink() or not path.is_file():
        raise CatalogError("E-SRB-PROD-CATALOG-001", f"{label} must be a regular file: {path}")
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CatalogError("E-SRB-PROD-CATALOG-001", f"cannot parse {label}: {error}") from error
    if not isinstance(payload, dict):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} must be a JSON object")
    return payload, raw


def canonical_timestamp(value: str) -> str:
    try:
        parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise CatalogError(
            "E-SRB-PROD-CATALOG-002",
            "observed timestamp must use YYYY-MM-DDTHH:MM:SSZ",
        ) from error
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise CatalogError(
            "E-SRB-PROD-CATALOG-002",
            "observed timestamp must use YYYY-MM-DDTHH:MM:SSZ",
        )
    return value


def required_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} must be an object")
    return value


def repository_row(node_value: Any, organization: str, index: int) -> dict[str, Any]:
    label = f"repository observation row {index}"
    node = required_object(node_value, label)
    exact_keys(
        node,
        {
            "name",
            "nameWithOwner",
            "url",
            "visibility",
            "isArchived",
            "isEmpty",
            "viewerPermission",
            "defaultBranchRef",
        },
        label,
    )
    name = node["name"]
    name_with_owner = node["nameWithOwner"]
    if not isinstance(name, str) or not SAFE_TOKEN.fullmatch(name):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} name is invalid")
    expected_name = f"{organization}/{name}"
    if name_with_owner != expected_name:
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} nameWithOwner mismatch")
    if node["url"] != f"https://github.com/{expected_name}":
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} URL mismatch")
    if node["visibility"] not in VISIBILITIES:
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} visibility is invalid")
    if not isinstance(node["isArchived"], bool) or not isinstance(node["isEmpty"], bool):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} flags are invalid")
    permission = node["viewerPermission"] if node["viewerPermission"] is not None else "UNKNOWN"
    if permission not in PERMISSIONS:
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} viewer permission is invalid")
    if node["isEmpty"]:
        raise CatalogError(
            "E-SRB-PROD-CATALOG-002",
            f"{label} is empty and cannot satisfy the v1 default-branch binding",
        )
    branch_ref = required_object(node["defaultBranchRef"], f"{label} defaultBranchRef")
    exact_keys(branch_ref, {"name", "target"}, f"{label} defaultBranchRef")
    branch = branch_ref["name"]
    if not isinstance(branch, str) or not SAFE_ID.fullmatch(branch):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} default branch is invalid")
    target = required_object(branch_ref["target"], f"{label} default branch target")
    exact_keys(target, {"oid"}, f"{label} default branch target")
    oid = target["oid"]
    if not isinstance(oid, str) or not GIT_OID.fullmatch(oid):
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"{label} head object is invalid")
    return {
        "repository_id": name,
        "name_with_owner": expected_name,
        "remote_url": f"https://github.com/{expected_name}.git",
        "default_branch": branch,
        "head_oid": oid,
        "visibility": node["visibility"],
        "archived": node["isArchived"],
        "is_empty": False,
        "observed_permission": permission,
    }


def expected_catalog(
    observation: dict[str, Any],
    organization: str,
    observed_at_utc: str,
) -> dict[str, Any]:
    if not GITHUB_LOGIN.fullmatch(organization) or "--" in organization:
        raise CatalogError("E-SRB-PROD-CATALOG-002", "organization is invalid")
    exact_keys(observation, {"data"}, "GraphQL response")
    data = required_object(observation["data"], "GraphQL data")
    exact_keys(data, {"organization"}, "GraphQL data")
    organization_row = required_object(data["organization"], "GraphQL organization")
    exact_keys(organization_row, {"login", "repositories"}, "GraphQL organization")
    if organization_row["login"] != organization:
        raise CatalogError("E-SRB-PROD-CATALOG-002", "GraphQL organization login mismatch")
    repositories = required_object(organization_row["repositories"], "GraphQL repositories")
    exact_keys(repositories, {"totalCount", "nodes"}, "GraphQL repositories")
    total_count = repositories["totalCount"]
    nodes = repositories["nodes"]
    if isinstance(total_count, bool) or not isinstance(total_count, int) or total_count < 1:
        raise CatalogError("E-SRB-PROD-CATALOG-002", "GraphQL totalCount is invalid")
    if not isinstance(nodes, list) or len(nodes) != total_count:
        raise CatalogError("E-SRB-PROD-CATALOG-002", "GraphQL repository observation is incomplete")
    rows = sorted(
        (repository_row(node, organization, index) for index, node in enumerate(nodes)),
        key=lambda row: row["repository_id"],
    )
    repository_ids = [row["repository_id"] for row in rows]
    if len(repository_ids) != len(set(repository_ids)):
        raise CatalogError("E-SRB-PROD-CATALOG-002", "GraphQL repository observation contains duplicates")
    payload: dict[str, Any] = {
        "schema": CATALOG_SCHEMA,
        "catalog_type": CATALOG_TYPE,
        "authority_scope": AUTHORITY_SCOPE,
        "organization": organization,
        "observed_at_utc": canonical_timestamp(observed_at_utc),
        "repositories": rows,
        "limitations": LIMITATIONS,
    }
    payload["catalog_identity_sha256"] = catalog_identity(payload)
    try:
        gap_tool.validate_catalog(payload)
    except gap_tool.ProductionGapError as error:
        raise CatalogError("E-SRB-PROD-CATALOG-002", f"emitted catalog is invalid: {error}") from error
    return payload


def publish_no_clobber(output: Path, raw: bytes) -> None:
    if output.is_symlink() or output.exists():
        raise CatalogError("E-SRB-PROD-CATALOG-003", f"output already exists: {output}")
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        raise CatalogError("E-SRB-PROD-CATALOG-003", f"cannot resolve output parent: {error}") from error
    if not parent.is_dir():
        raise CatalogError("E-SRB-PROD-CATALOG-003", "output parent is not a directory")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{output.name}.", dir=parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o644)
        os.link(temporary, output)
        directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        directory_fd = os.open(parent, directory_flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise CatalogError("E-SRB-PROD-CATALOG-003", f"output already exists: {output}") from error
    except OSError as error:
        raise CatalogError("E-SRB-PROD-CATALOG-003", f"cannot publish output: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graphql-observation", required=True)
    parser.add_argument("--organization", required=True)
    parser.add_argument("--observed-at-utc", required=True)


def command_build(args: argparse.Namespace) -> None:
    observation_path = Path(args.graphql_observation).expanduser()
    observation, _ = read_regular_json(observation_path, "GraphQL observation")
    payload = expected_catalog(observation, args.organization, args.observed_at_utc)
    publish_no_clobber(Path(args.output).expanduser(), render(payload))
    print(
        "SOUNIO_CANONICAL_PRODUCTION_REPOSITORY_CATALOG_BUILD_PASS "
        f"catalog_identity={payload['catalog_identity_sha256']} repositories={len(payload['repositories'])}"
    )


def command_verify(args: argparse.Namespace) -> None:
    observation, _ = read_regular_json(Path(args.graphql_observation).expanduser(), "GraphQL observation")
    actual, actual_raw = read_regular_json(Path(args.catalog).expanduser(), "repository catalog")
    expected = expected_catalog(observation, args.organization, args.observed_at_utc)
    if actual != expected or actual_raw != render(expected):
        raise CatalogError("E-SRB-PROD-CATALOG-004", "repository catalog does not match the bound observation")
    print(
        "SOUNIO_CANONICAL_PRODUCTION_REPOSITORY_CATALOG_VERIFY_PASS "
        f"catalog_identity={expected['catalog_identity_sha256']} repositories={len(expected['repositories'])}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="sounio-canonical-production-repository-catalog")
    commands = root.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    common_arguments(build)
    build.add_argument("--output", required=True)
    build.set_defaults(handler=command_build)
    verify = commands.add_parser("verify")
    common_arguments(verify)
    verify.add_argument("--catalog", required=True)
    verify.set_defaults(handler=command_verify)
    return root


def main() -> int:
    args = parser().parse_args()
    try:
        args.handler(args)
    except CatalogError as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
