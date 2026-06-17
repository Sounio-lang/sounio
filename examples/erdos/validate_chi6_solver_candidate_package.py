#!/usr/bin/env python3
"""Validate a chi>=6 solver candidate source package.

The source package binds the exact graph and exact rational coordinates before
any SAT/LRAT or Lean promotion. It is a provenance contract, not a theorem: a
valid package says only that the listed edges have exact rational unit-distance
coordinates and that downstream SAT/geometry producers must consume the same
raw-byte artifacts.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from cube_sieve_batch_manifest import sha256_file
from cube_sieve_propagation_manifest import parse_edge_file
from gen_lean_rational_geometry import parse_coords, validate_geometry


SHA_RE = re.compile(r"^[0-9a-f]{64}$")
ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
REQUIRED_KEYS = {
    "schema",
    "candidate_id",
    "edge_path",
    "edge_sha256",
    "coords_path",
    "coords_sha256",
    "coordinate_domain",
    "n",
    "m",
    "k",
    "split_vertices",
    "producer_command",
    "claim_scope",
    "promotion_gate",
}


def die(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def require_str(meta: dict[str, Any], key: str) -> str:
    value = meta.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def require_int(meta: dict[str, Any], key: str) -> int:
    value = meta.get(key)
    if not isinstance(value, int):
        raise ValueError(f"{key} must be an integer")
    return value


def resolve(base: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        raise ValueError(f"candidate package paths must be relative, got absolute path: {raw}")
    root = base.resolve()
    resolved = (base / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"candidate package path escapes source directory: {raw}")
    return resolved


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: validate_chi6_solver_candidate_package.py <candidate-source.json>", file=sys.stderr)
        return 2

    source = Path(sys.argv[1])
    if not source.is_file():
        return die(f"missing candidate source JSON: {source}")
    try:
        with source.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            raise ValueError("top-level JSON value must be an object")
        keys = set(meta)
        missing = sorted(REQUIRED_KEYS - keys)
        extra = sorted(keys - REQUIRED_KEYS)
        if missing:
            raise ValueError(f"missing required keys: {','.join(missing)}")
        if extra:
            raise ValueError(f"unexpected keys: {','.join(extra)}")

        if meta.get("schema") != "chi6_solver_candidate_package.v1":
            raise ValueError("schema must be chi6_solver_candidate_package.v1")
        candidate_id = require_str(meta, "candidate_id")
        if not ID_RE.fullmatch(candidate_id):
            raise ValueError("candidate_id must use only letters, digits, '.', '_', or '-'")
        if meta.get("coordinate_domain") != "rational_xy":
            raise ValueError("coordinate_domain must be rational_xy")
        if meta.get("claim_scope") != "solver_candidate_source_only":
            raise ValueError("claim_scope must be solver_candidate_source_only")
        if meta.get("promotion_gate") != "requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge":
            raise ValueError(
                "promotion_gate must be requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge"
            )

        n_decl = require_int(meta, "n")
        m_decl = require_int(meta, "m")
        k = require_int(meta, "k")
        if n_decl < 2:
            raise ValueError("n must be at least 2")
        if m_decl < 1:
            raise ValueError("m must be positive")
        if k != 5:
            raise ValueError("k must be 5 for chi>=6 candidate packages")

        split_vertices = meta.get("split_vertices")
        if not isinstance(split_vertices, list) or not split_vertices:
            raise ValueError("split_vertices must be a non-empty list")
        if not all(isinstance(v, int) and 0 <= v < n_decl for v in split_vertices):
            raise ValueError("split_vertices must be non-negative integers below n")
        if len(set(split_vertices)) != len(split_vertices):
            raise ValueError("split_vertices must not contain duplicates")

        edge_path_raw = require_str(meta, "edge_path")
        coords_path_raw = require_str(meta, "coords_path")
        edge_sha = require_str(meta, "edge_sha256").lower()
        coords_sha = require_str(meta, "coords_sha256").lower()
        if not SHA_RE.fullmatch(edge_sha):
            raise ValueError("edge_sha256 must be a lowercase SHA256 digest")
        if not SHA_RE.fullmatch(coords_sha):
            raise ValueError("coords_sha256 must be a lowercase SHA256 digest")

        base = source.parent
        edge_path = resolve(base, edge_path_raw)
        coords_path = resolve(base, coords_path_raw)
        if not edge_path.is_file():
            raise ValueError(f"missing edge_path artifact: {edge_path}")
        if not coords_path.is_file():
            raise ValueError(f"missing coords_path artifact: {coords_path}")
        actual_edge_sha = sha256_file(edge_path)
        actual_coords_sha = sha256_file(coords_path)
        if actual_edge_sha != edge_sha:
            raise ValueError(f"edge_sha256 mismatch: got {actual_edge_sha}, expected {edge_sha}")
        if actual_coords_sha != coords_sha:
            raise ValueError(f"coords_sha256 mismatch: got {actual_coords_sha}, expected {coords_sha}")

        n, m, edges = parse_edge_file(edge_path)
        if n != n_decl or m != m_decl:
            raise ValueError(f"edge header mismatch: got n={n} m={m}, expected n={n_decl} m={m_decl}")
        coords = parse_coords(coords_path, n)
        if len(coords) != n:
            raise ValueError(f"coordinate row count mismatch: got {len(coords)}, expected {n}")
        validate_geometry(coords, edges)

        producer_command = require_str(meta, "producer_command")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return die(str(exc))

    split_csv = ",".join(str(v) for v in split_vertices)
    print("chi6_solver_candidate_package v1")
    print(f"source_json={source}")
    print(f"candidate_id={candidate_id}")
    print(f"n={n}")
    print(f"m={m}")
    print(f"k={k}")
    print(f"validated_edge_count={m}")
    print(f"coordinate_row_count={len(coords)}")
    print(f"edge_path={edge_path_raw}")
    print(f"edge_path_abs={edge_path}")
    print(f"edge_sha256={edge_sha}")
    print(f"coords_path={coords_path_raw}")
    print(f"coords_path_abs={coords_path}")
    print(f"coords_sha256={coords_sha}")
    print("coordinate_domain=rational_xy")
    print(f"split_vertices={split_csv}")
    print(f"producer_command={producer_command}")
    print("claim_scope=solver_candidate_source_only")
    print("promotion_gate=requires_cube_cover_lrat_lean_exact_geometry_and_real_bridge")
    print("geometry_claim=exact_rational_squared_distance_edges_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("promotable=0")
    print("status=VALID_SOLVER_CANDIDATE_PACKAGE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
