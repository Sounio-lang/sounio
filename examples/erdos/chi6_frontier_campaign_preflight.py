#!/usr/bin/env python3
"""Plan a cube campaign from one exact-rational frontier scout artifact.

This consumes `chi6_rational_frontier_scout.v1`, validates the source package,
emits the deterministic split-product cube batch, runs cheap propagation over
the leaves, and records the exact next SAT/LRAT commands. It is a classifier and
workload planner only: propagation conflicts are not proof artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from cube_sieve_propagation_manifest import parse_edge_file


VALIDATOR = Path(__file__).with_name("validate_chi6_solver_candidate_package.py")
SPLIT = Path(__file__).with_name("cube_split_batch.py")
BATCH = Path(__file__).with_name("cube_sieve_batch_manifest.py")
REFUTE = Path(__file__).with_name("cube_sieve_refute_batch.py")
COMPLEMENT = Path(__file__).with_name("cube_cover_complement_cnf.py")
PREFLIGHT = Path(__file__).with_name("make_chi6_integrated_candidate_preflight.sh")
MAX_HARD_CUBE_SAMPLE = 1000


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_kv_output(text: str) -> dict[str, str]:
    """Extract top-level key-value status lines; cube rows are parsed separately."""
    out: dict[str, str] = {}
    for line in text.splitlines():
        if line and "=" in line and not line.startswith("cube "):
            key, value = line.split("=", 1)
            if key in out:
                raise ValueError(f"duplicate key in tool output: {key}")
            out[key] = value
    return out


def run_command(cmd: list[str], out_path: Path) -> str:
    proc = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    out_path.write_text(proc.stdout, encoding="ascii")
    out_path.with_name(out_path.name + ".stderr").write_text(proc.stderr, encoding="ascii")
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip() or "no output")
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(
            f"command failed with exit {proc.returncode}: {' '.join(cmd)}; see {out_path}{suffix}"
        )
    return proc.stdout


def require_fields(fields: dict[str, str], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in fields]
    if missing:
        raise RuntimeError(f"{label}: missing fields: {','.join(missing)}")


def parse_nonneg_int(fields: dict[str, str], key: str, label: str) -> int:
    raw = fields[key]
    if not raw.isdigit():
        raise RuntimeError(f"{label}: field {key} must be a non-negative integer, got {raw!r}")
    return int(raw)


def parse_cube_rows(text: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("cube "):
            continue
        row: dict[str, str] = {}
        for token in line.split()[1:]:
            if "=" in token:
                key, value = token.split("=", 1)
                if not value:
                    raise ValueError(f"empty cube row field {key!r}")
                row[key] = value
        if "id" not in row:
            raise ValueError(f"cube row missing id: {line}")
        rows.append(row)
    return rows


def read_cube_assignments(path: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    with path.open("r", encoding="ascii") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                raise ValueError(f"{path}:{lineno}: bad cube line")
            cube_id, units = line.split(":", 1)
            cube_id = cube_id.strip()
            if cube_id in assignments:
                raise ValueError(f"{path}:{lineno}: duplicate cube id {cube_id}")
            assignments[cube_id] = units.strip()
    return assignments


def campaign_id(candidate_id: str, edge_sha: str, split_vertices: str, k: str) -> str:
    raw = f"{candidate_id}|{edge_sha}|{split_vertices}|{k}".encode("ascii")
    return "chi6camp_" + hashlib.sha256(raw).hexdigest()[:32]


def load_sidecar(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if meta.get("schema") != "chi6_rational_frontier_scout.v1":
        raise ValueError("frontier scout schema must be chi6_rational_frontier_scout.v1")
    for key in ("candidate_id", "candidate_source", "split_vertices", "dsatur_status", "promotable"):
        if key not in meta:
            raise ValueError(f"frontier scout missing required field: {key}")
    if meta.get("promotable") != 0:
        raise ValueError("frontier scout must be non-promotable")
    if meta.get("sat_claim") != "none" or meta.get("chromatic_claim") != "none":
        raise ValueError("frontier scout must carry no SAT/chromatic claim")
    return meta


def choose_next_gate(dsatur_status: str, hard_count: int, conflict_count: int, cube_count: int) -> str:
    if cube_count <= 0:
        raise RuntimeError("propagation emitted zero cubes")
    if hard_count + conflict_count != cube_count:
        raise RuntimeError("propagation hard/conflict counts do not cover cube count")
    if dsatur_status == "K_COLORING_FOUND":
        return "reject_or_mutate_frontier"
    if hard_count > 0:
        return "prepare_cube_refute_batch"
    if conflict_count > 0:
        return "propagation_conflicts_require_lrat_lean_upgrade"
    raise RuntimeError("propagation emitted cubes with no hard cubes and no conflicts")


def split_edge_counts(
    split_vertices: list[int],
    edges: list[tuple[int, int]],
) -> tuple[int, int]:
    selected = set(split_vertices)
    induced = sum(1 for u, v in edges if u in selected and v in selected)
    incident = sum(1 for u, v in edges if u in selected or v in selected)
    return induced, incident


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("frontier_scout_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    args = parser.parse_args()

    try:
        if args.max_cubes <= 0:
            raise ValueError("--max-cubes must be positive")
        if args.sample_hard_cubes < 0:
            raise ValueError("--sample-hard-cubes must be non-negative")
        if args.sample_hard_cubes > MAX_HARD_CUBE_SAMPLE:
            raise ValueError(f"--sample-hard-cubes must be <= {MAX_HARD_CUBE_SAMPLE}")
        for tool in (VALIDATOR, SPLIT, BATCH, REFUTE, COMPLEMENT, PREFLIGHT):
            if not tool.is_file():
                raise ValueError(f"missing tool: {tool}")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        sidecar = load_sidecar(args.frontier_scout_json)
        candidate_source = Path(sidecar["candidate_source"])
        if not candidate_source.is_file():
            raise ValueError(f"missing candidate source: {candidate_source}")

        validator_out = args.out_dir / "source_validator.out"
        source_text = run_command([sys.executable, str(VALIDATOR), str(candidate_source)], validator_out)
        source = parse_kv_output(source_text)
        require_fields(
            source,
            (
                "status",
                "candidate_id",
                "n",
                "m",
                "k",
                "edge_path_abs",
                "edge_sha256",
                "coords_path_abs",
                "coords_sha256",
                "split_vertices",
            ),
            str(validator_out),
        )
        if source.get("status") != "VALID_SOLVER_CANDIDATE_PACKAGE":
            raise RuntimeError("source validator did not return VALID_SOLVER_CANDIDATE_PACKAGE")
        n = parse_nonneg_int(source, "n", str(validator_out))
        m = parse_nonneg_int(source, "m", str(validator_out))
        k = parse_nonneg_int(source, "k", str(validator_out))
        if source["candidate_id"] != sidecar["candidate_id"]:
            raise RuntimeError("source candidate_id does not match frontier scout")
        if source["split_vertices"] != ",".join(str(v) for v in sidecar["split_vertices"]):
            raise RuntimeError("source split_vertices do not match frontier scout")
        if k != 5:
            raise RuntimeError("frontier campaign preflight only supports k=5")
        if sha256_file(Path(source["edge_path_abs"])) != source["edge_sha256"]:
            raise RuntimeError("source edge_sha256 does not match edge file")
        if sha256_file(Path(source["coords_path_abs"])) != source["coords_sha256"]:
            raise RuntimeError("source coords_sha256 does not match coordinate file")
        _n_edge, _m_edge, edges = parse_edge_file(Path(source["edge_path_abs"]))
        if _n_edge != n or _m_edge != m:
            raise RuntimeError("validated edge file metadata does not match source package")
        seen_edges = set()
        for u, v in edges:
            if u == v:
                raise RuntimeError("validated edge file contains a self-loop")
            key = (u, v) if u < v else (v, u)
            if key in seen_edges:
                raise RuntimeError("validated edge file contains a duplicate edge")
            seen_edges.add(key)
        induced_count, incident_count = split_edge_counts(sidecar["split_vertices"], edges)
        if induced_count != sidecar["split_vertices_induced_edge_count"]:
            raise RuntimeError("frontier scout induced split-edge count does not match edge file")
        if incident_count != sidecar["split_vertices_incident_edge_count"]:
            raise RuntimeError("frontier scout incident split-edge count does not match edge file")

        cid = campaign_id(source["candidate_id"], source["edge_sha256"], source["split_vertices"], source["k"])
        cube_batch = args.out_dir / f"{cid}.cubes"
        split_out = args.out_dir / "cube_split.out"
        split_text = run_command(
            [
                sys.executable,
                str(SPLIT),
                source["edge_path_abs"],
                source["k"],
                str(cube_batch),
                "--split-vertices",
                source["split_vertices"],
                "--id-prefix",
                cid,
                "--max-cubes",
                str(args.max_cubes),
            ],
            split_out,
        )
        split = parse_kv_output(split_text)
        require_fields(split, ("status", "first_cube_id", "last_cube_id"), str(split_out))
        if split.get("status") != "cube_batch_emitted_unpromotable":
            raise RuntimeError("cube split did not emit the expected unpromotable status")

        propagation_dir = args.out_dir / "propagation"
        propagation_out = args.out_dir / "propagation_batch.out"
        propagation_text = run_command(
            [
                sys.executable,
                str(BATCH),
                source["edge_path_abs"],
                source["k"],
                str(cube_batch),
                str(propagation_dir),
            ],
            propagation_out,
        )
        propagation = parse_kv_output(propagation_text)
        require_fields(propagation, ("status", "cube_count", "hard_count", "conflict_count"), str(propagation_out))
        if propagation.get("status") != "batch_manifest_emitted_unpromotable":
            raise RuntimeError("propagation batch did not emit the expected unpromotable status")
        cube_rows = parse_cube_rows(propagation_text)
        hard_ids = [row["id"] for row in cube_rows if row.get("hard_cube") == "1"]
        cube_assignments = read_cube_assignments(cube_batch)
        missing_assignments = [cube_id for cube_id in hard_ids if cube_id not in cube_assignments]
        if missing_assignments:
            raise RuntimeError(f"hard cube ids missing from cube batch: {','.join(missing_assignments)}")
        hard_list = args.out_dir / f"{cid}.hard-cubes"
        hard_list.write_text("".join(f"{cube_id}\n" for cube_id in hard_ids), encoding="ascii")

        cube_count = parse_nonneg_int(propagation, "cube_count", str(propagation_out))
        hard_count = parse_nonneg_int(propagation, "hard_count", str(propagation_out))
        conflict_count = parse_nonneg_int(propagation, "conflict_count", str(propagation_out))
        recommended = choose_next_gate(str(sidecar["dsatur_status"]), hard_count, conflict_count, cube_count)
        refute_dir = args.out_dir / "refute"
        cover_cnf = args.out_dir / f"{cid}.cover-complement.cnf"
        manifest = {
            "schema": "chi6_frontier_campaign_preflight.v1",
            "candidate_id": source["candidate_id"],
            "campaign_id": cid,
            "campaign_mode": "split_product_frontier_preflight",
            "frontier_scout_path": str(args.frontier_scout_json),
            "frontier_scout_sha256": sha256_file(args.frontier_scout_json),
            "candidate_source_path": str(candidate_source),
            "candidate_source_sha256": sha256_file(candidate_source),
            "source_validator_log": str(validator_out),
            "source_status": "PASS",
            "n": n,
            "m": m,
            "k": k,
            "edge_path_abs": source["edge_path_abs"],
            "edge_sha256": source["edge_sha256"],
            "coords_path_abs": source["coords_path_abs"],
            "coords_sha256": source["coords_sha256"],
            "split_vertices": sidecar["split_vertices"],
            "split_depth": len(sidecar["split_vertices"]),
            "split_vertex_degrees": sidecar["split_vertex_degrees"],
            "split_vertices_induced_edge_count": sidecar["split_vertices_induced_edge_count"],
            "split_vertices_incident_edge_count": sidecar["split_vertices_incident_edge_count"],
            "dsatur_status": sidecar["dsatur_status"],
            "dsatur_claim_scope": sidecar["dsatur_claim_scope"],
            "cube_count": cube_count,
            "max_cubes": args.max_cubes,
            "cube_batch_path": str(cube_batch),
            "cube_batch_sha256": sha256_file(cube_batch),
            "first_cube_id": split["first_cube_id"],
            "last_cube_id": split["last_cube_id"],
            "propagation_summary_path": str(propagation_out),
            "propagation_summary_sha256": sha256_file(propagation_out),
            "propagation_conflict_count": conflict_count,
            "propagation_hard_count": hard_count,
            "hard_cube_list_path": str(hard_list),
            "hard_cube_list_sha256": sha256_file(hard_list),
            "hard_cube_prefix_sample_requested": args.sample_hard_cubes,
            "hard_cube_prefix_sample_count": min(len(hard_ids), args.sample_hard_cubes),
            "hard_cube_prefix_sample_truncated": 1 if len(hard_ids) > args.sample_hard_cubes else 0,
            "hard_cube_prefix_sample_exhaustive": 1 if len(hard_ids) <= args.sample_hard_cubes else 0,
            "hard_cube_prefix_sample": hard_ids[: args.sample_hard_cubes],
            "hard_cube_prefix_assignment_sample": [
                {"id": cube_id, "assignments": cube_assignments[cube_id]}
                for cube_id in hard_ids[: args.sample_hard_cubes]
            ],
            "estimated_repo_colourCNF_base_clause_count": n + m * k,
            "estimated_repo_colourCNF_base_clause_count_scope": (
                "repo_colourCNF_base_only_atleast_one_plus_edge_clauses_only"
                "_no_cube_units_lrat_or_cover_clauses"
            ),
            "standard_at_most_one_colour_clause_count_included": 0,
            "estimated_vars": n * k,
            "estimated_leaf_lrat_required_if_all_hard_cubes_unsat": hard_count,
            "recommended_next_action": recommended,
            "recommended_next_gate": recommended,
            "next_action_execution_status": "not_run_by_preflight",
            "foundry_handoff_recommended": 1 if hard_count > 0 else 0,
            "refute_command": f"python3 {REFUTE} {source['edge_path_abs']} {source['k']} {cube_batch} {refute_dir}",
            "complement_cnf_command": f"python3 {COMPLEMENT} {source['edge_path_abs']} {source['k']} <covered-cubes> {cover_cnf}",
            "integrated_preflight_command": f"WORK=<work> {PREFLIGHT} {candidate_source} {cube_batch} <cover-drup-or-rup>",
            "claim_scope": "deterministic_campaign_preflight_only",
            "geometry_claim": "exact_rational_squared_distance_edges_only_from_source_validator",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_leaf_lrat_cover_lrat_lean_exact_geometry_real_bridge",
        }
        manifest_path = args.out_dir / f"{cid}.campaign-preflight.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_campaign_preflight v1")
    print(f"candidate_id={source['candidate_id']}")
    print(f"campaign_id={cid}")
    print("campaign_mode=split_product_frontier_preflight")
    print(f"frontier_scout={args.frontier_scout_json}")
    print(f"frontier_scout_sha256={sha256_file(args.frontier_scout_json)}")
    print(f"candidate_source={candidate_source}")
    print(f"candidate_source_sha256={sha256_file(candidate_source)}")
    print("source_status=PASS")
    print(f"n={source['n']}")
    print(f"m={source['m']}")
    print(f"k={source['k']}")
    print(f"split_vertices={source['split_vertices']}")
    print(f"cube_count={cube_count}")
    print(f"propagation_conflict_count={conflict_count}")
    print(f"propagation_hard_count={hard_count}")
    print(f"hard_cube_list={hard_list}")
    print(f"campaign_preflight_json={manifest_path}")
    print(f"recommended_next_action={recommended}")
    print(f"recommended_next_gate={recommended}")
    print("next_action_execution_status=not_run_by_preflight")
    print("claim_scope=deterministic_campaign_preflight_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=FRONTIER_CAMPAIGN_PREFLIGHT_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
