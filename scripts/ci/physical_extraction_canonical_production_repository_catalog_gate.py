#!/usr/bin/env python3
"""Adversarial gate for deterministic repository-catalog processing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools" / "science_boundary" / "canonical_production_repository_catalog.py"
TESTS = 0


def check(condition: bool, message: str) -> None:
    global TESTS
    TESTS += 1
    if not condition:
        raise AssertionError(message)


def run(command: list[str], expected: int | set[int] = 0) -> subprocess.CompletedProcess[str]:
    expected_codes = {expected} if isinstance(expected, int) else expected
    result = subprocess.run(
        command,
        env={**os.environ, "LANG": "C", "LC_ALL": "C", "TZ": "UTC"},
        text=True,
        capture_output=True,
        timeout=120,
    )
    if result.returncode not in expected_codes:
        raise AssertionError(
            f"command returned {result.returncode}, expected {sorted(expected_codes)}: {' '.join(command)}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def write_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=True) + "\n", encoding="ascii")
    return path


def node(name: str, oid: str, *, permission: str | None = "ADMIN") -> dict[str, Any]:
    return {
        "name": name,
        "nameWithOwner": f"SounioFixture/{name}",
        "url": f"https://github.com/SounioFixture/{name}",
        "visibility": "PUBLIC",
        "isArchived": False,
        "isEmpty": False,
        "viewerPermission": permission,
        "defaultBranchRef": {"name": "main", "target": {"oid": oid}},
    }


def fixture() -> dict[str, Any]:
    return {
        "data": {
            "organization": {
                "login": "SounioFixture",
                "repositories": {
                    "totalCount": 2,
                    "nodes": [node("zeta", "2" * 40), node("alpha", "1" * 40, permission=None)],
                },
            }
        }
    }


def build_command(
    observation: Path,
    output: Path,
    *,
    timestamp: str = "2026-07-20T00:00:00Z",
    organization: str = "SounioFixture",
) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "build",
        "--graphql-observation",
        str(observation),
        "--organization",
        organization,
        "--observed-at-utc",
        timestamp,
        "--output",
        str(output),
    ]


def verify_command(observation: Path, catalog: Path) -> list[str]:
    return [
        sys.executable,
        str(TOOL),
        "verify",
        "--graphql-observation",
        str(observation),
        "--organization",
        "SounioFixture",
        "--observed-at-utc",
        "2026-07-20T00:00:00Z",
        "--catalog",
        str(catalog),
    ]


def mutated_case(root: Path, name: str, mutate: Callable[[dict[str, Any]], None]) -> Path:
    payload = fixture()
    mutate(payload)
    return write_json(root / f"{name}.json", payload)


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sounio-repository-catalog-gate.") as temporary:
        work = Path(temporary)
        observation = write_json(work / "observation.json", fixture())
        observation_before = hashlib.sha256(observation.read_bytes()).hexdigest()
        catalog = work / "catalog.json"

        built = run(build_command(observation, catalog))
        check("BUILD_PASS" in built.stdout, "build receipt missing")
        payload = json.loads(catalog.read_text(encoding="ascii"))
        check([row["repository_id"] for row in payload["repositories"]] == ["alpha", "zeta"], "rows not sorted")
        check(payload["repositories"][0]["observed_permission"] == "UNKNOWN", "null permission not normalized")
        check(payload["authority_scope"] == "supplied-repository-metadata-observation", "authority widened")
        check(payload["catalog_identity_sha256"] == hashlib.sha256(
            json.dumps(
                {key: value for key, value in payload.items() if key != "catalog_identity_sha256"},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest(), "catalog identity mismatch")
        check(hashlib.sha256(observation.read_bytes()).hexdigest() == observation_before, "build changed observation")
        verified = run(verify_command(observation, catalog))
        check("VERIFY_PASS" in verified.stdout, "verify receipt missing")
        check(run(build_command(observation, catalog), expected=1).stderr.startswith("E-SRB-PROD-CATALOG-003"), "clobber accepted")

        input_link = work / "input-link.json"
        input_link.symlink_to(observation)
        check(run(build_command(input_link, work / "from-link.json"), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-001"
        ), "input symlink accepted")
        catalog_link = work / "catalog-link.json"
        catalog_link.symlink_to(catalog)
        check(run(verify_command(observation, catalog_link), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-001"
        ), "catalog symlink accepted")
        output_link = work / "output-link.json"
        output_link.symlink_to(work / "absent.json")
        check(run(build_command(observation, output_link), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-003"
        ), "output symlink accepted")

        cases: list[tuple[str, Callable[[dict[str, Any]], None]]] = [
            ("extra-top-level", lambda value: value.__setitem__("errors", [])),
            ("missing-data", lambda value: value.pop("data")),
            ("wrong-login", lambda value: value["data"]["organization"].__setitem__("login", "Elsewhere")),
            ("count-mismatch", lambda value: value["data"]["organization"]["repositories"].__setitem__("totalCount", 3)),
            ("boolean-count", lambda value: value["data"]["organization"]["repositories"].__setitem__("totalCount", True)),
            ("null-node", lambda value: value["data"]["organization"]["repositories"]["nodes"].__setitem__(0, None)),
            ("duplicate", lambda value: value["data"]["organization"]["repositories"]["nodes"].__setitem__(1, node("zeta", "3" * 40))),
            ("bad-owner", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("nameWithOwner", "Other/zeta")),
            ("bad-url", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("url", "https://example.invalid/zeta")),
            ("bad-visibility", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("visibility", "SECRET")),
            ("bad-permission", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("viewerPermission", "OWNER")),
            ("bad-oid", lambda value: value["data"]["organization"]["repositories"]["nodes"][0]["defaultBranchRef"]["target"].__setitem__("oid", "not-an-oid")),
            ("null-branch", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("defaultBranchRef", None)),
            ("empty-repo", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].update({"isEmpty": True, "defaultBranchRef": None})),
            ("extra-node-field", lambda value: value["data"]["organization"]["repositories"]["nodes"][0].__setitem__("owner", {})),
        ]
        for name, mutate in cases:
            bad = mutated_case(work, name, mutate)
            result = run(build_command(bad, work / f"{name}-catalog.json"), expected=1)
            check(result.stderr.startswith("E-SRB-PROD-CATALOG-002"), f"{name} was not refused")

        check(run(build_command(observation, work / "bad-time.json", timestamp="2026-02-30T00:00:00Z"), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-002"
        ), "invalid timestamp accepted")
        check(run(build_command(
            observation,
            work / "bad-organization.json",
            organization="Sounio/Fixture",
        ), expected=1).stderr.startswith("E-SRB-PROD-CATALOG-002"), "invalid GitHub organization accepted")

        tampered = json.loads(catalog.read_text(encoding="ascii"))
        tampered["repositories"][0]["head_oid"] = "4" * 40
        tampered["catalog_identity_sha256"] = "0" * 64
        tampered_path = write_json(work / "tampered.json", tampered)
        check(run(verify_command(observation, tampered_path), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-004"
        ), "tampered catalog accepted")

        reformatted = work / "reformatted.json"
        reformatted.write_text(json.dumps(payload, sort_keys=True), encoding="ascii")
        check(run(verify_command(observation, reformatted), expected=1).stderr.startswith(
            "E-SRB-PROD-CATALOG-004"
        ), "non-canonical serialization accepted")

    print(f"SOUNIO_CANONICAL_PRODUCTION_REPOSITORY_CATALOG_GATE_PASS assertions={TESTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
