#!/usr/bin/env python3
"""Run the cube-propagation producer over a batch of cubes.

This is still untrusted search plumbing. It writes one replayable per-cube
manifest plus a compact batch summary; every per-cube manifest is validated
locally, but no SAT/LRAT or Euclidean geometry claim is made.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path

from validate_cube_sieve_manifest import ManifestError, parse_key, section, validate


ASSIGNMENT_RE = re.compile(r"^(?P<v>\d+)\s*[:=,]\s*(?P<c>\d+)$")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_cube_token(token: str, path: Path, lineno: int) -> tuple[int, int]:
    m = ASSIGNMENT_RE.fullmatch(token)
    if not m:
        raise ValueError(f"{path}:{lineno}: bad cube assignment token {token!r}")
    return int(m["v"]), int(m["c"])


def parse_batch(path: Path) -> list[tuple[str, list[tuple[int, int]]]]:
    cubes: list[tuple[str, list[tuple[int, int]]]] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        cube_id = f"cube_{len(cubes):04d}"
        if tokens[0].endswith(":") and not ASSIGNMENT_RE.fullmatch(tokens[0]):
            cube_id = tokens.pop(0)[:-1]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", cube_id):
            raise ValueError(f"{path}:{lineno}: invalid cube id {cube_id!r}")
        if cube_id in seen_ids:
            raise ValueError(f"{path}:{lineno}: duplicate cube id {cube_id}")
        if not tokens:
            raise ValueError(f"{path}:{lineno}: cube {cube_id} has no assignments")
        cube = [parse_cube_token(tok, path, lineno) for tok in tokens]
        seen_vertices: set[int] = set()
        for v, _c in cube:
            if v in seen_vertices:
                raise ValueError(f"{path}:{lineno}: duplicate vertex {v} in cube {cube_id}")
            seen_vertices.add(v)
        seen_ids.add(cube_id)
        cubes.append((cube_id, cube))
    if not cubes:
        raise ValueError(f"{path}: batch file contains no cubes")
    return cubes


def write_cube_file(path: Path, cube: list[tuple[int, int]]) -> None:
    path.write_text("".join(f"{v} {c}\n" for v, c in cube), encoding="ascii")


def summary_field(manifest_text: str, key: str) -> str:
    sec = section([line.rstrip("\n") for line in manifest_text.splitlines()], "dimacs_cube_propagation")
    return parse_key(sec, key)


def run_single(
    producer: Path,
    edge_file: Path,
    k: int,
    cube_id: str,
    cube: list[tuple[int, int]],
    out_dir: Path,
) -> tuple[Path, dict[str, str]]:
    cube_path = out_dir / f"{cube_id}.cube"
    manifest_path = out_dir / f"{cube_id}.manifest"
    write_cube_file(cube_path, cube)
    proc = subprocess.run(
        [sys.executable, str(producer), str(edge_file), str(k), str(cube_path)],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"producer failed for cube {cube_id} with exit {proc.returncode}: {proc.stderr.strip()}"
        )
    manifest_path.write_text(proc.stdout, encoding="ascii")
    try:
        validate(proc.stdout)
    except ManifestError as exc:
        raise RuntimeError(f"validator rejected generated cube {cube_id}: {exc}") from exc
    return manifest_path, {
        "cube_sha256": sha256_file(cube_path),
        "trail_len": summary_field(proc.stdout, "trail_len"),
        "conflict": summary_field(proc.stdout, "conflict"),
        "conflict_vertex": summary_field(proc.stdout, "conflict_vertex"),
        "hard_cube": summary_field(proc.stdout, "hard_cube"),
        "final_domains": summary_field(proc.stdout, "final_domains"),
        "manifest_sha256": sha256_file(manifest_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("edge_file", type=Path)
    parser.add_argument("k", type=int)
    parser.add_argument("cube_batch", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument(
        "--producer",
        type=Path,
        default=Path(__file__).with_name("cube_sieve_propagation_manifest.py"),
    )
    args = parser.parse_args()

    if args.k <= 0 or args.k >= 62:
        raise SystemExit("error: k must satisfy 0 < k < 62")
    if not args.edge_file.is_file():
        raise SystemExit(f"error: missing edge file: {args.edge_file}")
    if not args.producer.is_file():
        raise SystemExit(f"error: missing producer: {args.producer}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        cubes = parse_batch(args.cube_batch)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    rows: list[tuple[str, Path, dict[str, str]]] = []
    try:
        for cube_id, cube in cubes:
            manifest_path, fields = run_single(
                args.producer, args.edge_file, args.k, cube_id, cube, args.out_dir
            )
            rows.append((cube_id, manifest_path, fields))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    conflict_count = sum(1 for _cube_id, _path, fields in rows if fields["conflict"] == "1")
    hard_count = sum(1 for _cube_id, _path, fields in rows if fields["hard_cube"] == "1")
    print("cube_sieve_batch_manifest v1")
    print("trust_boundary=search_untrusted__drat_lrat_lean_verified_required")
    print("output=dimacs_cube_batch_summary")
    print(f"edge_path={args.edge_file}")
    print(f"edge_sha256={sha256_file(args.edge_file)}")
    print(f"k={args.k}")
    print(f"cube_batch_path={args.cube_batch}")
    print(f"cube_batch_sha256={sha256_file(args.cube_batch)}")
    print(f"out_dir={args.out_dir}")
    print(f"cube_count={len(rows)}")
    print(f"conflict_count={conflict_count}")
    print(f"hard_count={hard_count}")
    print("verified_claim=none")
    print("geometry_claim=none")
    print("proof_artifact_sha256=NONE")
    print("promotion_gate=REJECT_NONE_PROOF_ARTIFACT")
    print("promotable=0")
    for index, (cube_id, manifest_path, fields) in enumerate(rows):
        print(
            "cube "
            f"index={index} id={cube_id} cube={cube_id}.cube "
            f"cube_sha256={fields['cube_sha256']} manifest={manifest_path.name} "
            f"manifest_sha256={fields['manifest_sha256']} "
            f"conflict={fields['conflict']} hard_cube={fields['hard_cube']} "
            f"trail_len={fields['trail_len']} conflict_vertex={fields['conflict_vertex']} "
            f"final_domains={fields['final_domains']}"
        )
    print("status=batch_manifest_emitted_unpromotable")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
