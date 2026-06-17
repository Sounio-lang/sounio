#!/usr/bin/env python3
"""Deterministic campaign runner for rational chi>=6 frontier scouts.

The campaign fans out small exact-rational scout runs, validates each source
package through the integrated preflight, and writes a ranked campaign manifest.
It is still search plumbing: this artifact emits no no-5-colouring certificate.
Promotion requires the downstream LRAT/Lean gates.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from make_chi6_rational_unit_graph_source_package import sha256_file, validate_candidate_id


SCOUT = Path(__file__).with_name("chi6_rational_frontier_scout.py")
PREFLIGHT = Path(__file__).with_name("make_chi6_integrated_candidate_preflight.sh")
# Prioritize scouts that looked hard for the bounded CPU probe; this is only a
# solver scheduling heuristic and carries no chromatic claim.
STATUS_BONUS = {
    "NO_K_COLORING_FOUND_BY_CPU_PROBE_NONCERTIFYING": 50_000,
    "UNKNOWN_NODE_LIMIT": 25_000,
    "K_COLORING_FOUND": 0,
}
PRIORITY_CLAIM_SCOPE = "noncertifying_solver_priority_only_not_chromatic_evidence"
DSATUR_NODE_SCORE_CAP = 10_000
SCOUT_INFEASIBLE_MARKERS = (
    "below required minimum",
    "cannot choose split depth",
)


class ScoutInfeasible(RuntimeError):
    def __init__(self, row: dict[str, Any]) -> None:
        super().__init__(str(row["reason"]))
        self.row = row


def parse_int_list(raw: str, name: str) -> list[int]:
    if not raw:
        raise ValueError(f"{name} cannot be empty")
    out: list[int] = []
    for token in raw.split(","):
        if not re.fullmatch(r"[1-9][0-9]*", token):
            raise ValueError(f"bad {name} token: {token!r}")
        value = int(token)
        out.append(value)
    return out


def parse_kv_output(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate key in tool output: {key}")
        fields[key] = value
    return fields


def run_command(cmd: list[str], out_path: Path, env: dict[str, str] | None = None) -> str:
    """Persist stdout+stderr for audit, return stdout for status-line parsing."""
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    out_path.with_name(out_path.name + ".stderr").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = (proc.stderr.strip() or proc.stdout.strip())
        suffix = f": {detail}" if detail else ""
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}; see {out_path}{suffix}")
    return proc.stdout


def first_nonempty_line(text: str) -> str:
    for raw in text.splitlines():
        line = raw.strip()
        if line:
            return line
    return "no diagnostic"


def classify_infeasible_scout(
    *,
    cid: str,
    max_den: int,
    layers: int,
    max_points: int,
    scout_out: Path,
    error: RuntimeError,
) -> ScoutInfeasible | None:
    stderr_path = scout_out.with_name(scout_out.name + ".stderr")
    stdout = scout_out.read_text(encoding="utf-8") if scout_out.is_file() else ""
    stderr = stderr_path.read_text(encoding="utf-8") if stderr_path.is_file() else ""
    combined = f"{stdout}\n{stderr}\n{error}"
    if not any(marker in combined for marker in SCOUT_INFEASIBLE_MARKERS):
        return None
    return ScoutInfeasible(
        {
            "candidate_id": cid,
            "max_den": max_den,
            "layers": layers,
            "max_points": max_points,
            "status": "SCOUT_INFEASIBLE_SKIPPED",
            "reason": first_nonempty_line(stderr or stdout or str(error)),
            "scout_out": str(scout_out),
            "scout_stderr": str(stderr_path),
            "claim_scope": "scout_infeasible_parameter_row_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "promotable": 0,
        }
    )


def candidate_id(prefix: str, max_den: int, layers: int, max_points: int, index: int) -> str:
    cid = f"{prefix}_d{max_den}_l{layers}_p{max_points}_{index:03d}"
    validate_candidate_id(cid)
    return cid


def score_candidate(sidecar: dict[str, Any]) -> int:
    for key in ("candidate_id", "dsatur_status", "m", "max_degree", "dsatur_nodes"):
        if key not in sidecar:
            raise ValueError(f"scout sidecar missing scoring field: {key}")
    status = str(sidecar.get("dsatur_status", ""))
    if status not in STATUS_BONUS:
        raise ValueError(f"unknown dsatur_status from scout: {status}")
    m = int(sidecar["m"])
    max_degree = int(sidecar["max_degree"])
    dsatur_nodes = int(sidecar["dsatur_nodes"])
    if m < 0 or max_degree < 0 or dsatur_nodes < 0:
        raise ValueError("scout scoring fields must be non-negative")
    dsatur_node_score = min(dsatur_nodes, DSATUR_NODE_SCORE_CAP)
    return (
        STATUS_BONUS[status]
        + m * 100
        + max_degree * 10
        + dsatur_node_score
    )


def require_fields(fields: dict[str, str], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in fields]
    if missing:
        raise RuntimeError(f"{label}: missing fields: {','.join(missing)}")


def run_scout(
    *,
    out_dir: Path,
    cid: str,
    max_den: int,
    layers: int,
    max_points: int,
    min_vertices: int,
    min_edges: int,
    split_depth: int,
    min_split_degree: int,
    dsatur_node_limit: int,
) -> tuple[dict[str, Any], Path, Path]:
    run_dir = out_dir / cid
    if run_dir.exists():
        raise RuntimeError(f"campaign run directory already exists; remove it or choose a new out_dir: {run_dir}")
    run_dir.mkdir()
    scout_out = run_dir / "scout.out"
    cmd = [
        sys.executable,
        str(SCOUT),
        "--candidate-id",
        cid,
        "--out-dir",
        str(run_dir),
        "--max-den",
        str(max_den),
        "--layers",
        str(layers),
        "--max-points",
        str(max_points),
        "--min-vertices",
        str(min_vertices),
        "--min-edges",
        str(min_edges),
        "--split-depth",
        str(split_depth),
        "--min-split-degree",
        str(min_split_degree),
        "--dsatur-node-limit",
        str(dsatur_node_limit),
    ]
    try:
        text = run_command(cmd, scout_out)
    except RuntimeError as exc:
        infeasible = classify_infeasible_scout(
            cid=cid,
            max_den=max_den,
            layers=layers,
            max_points=max_points,
            scout_out=scout_out,
            error=exc,
        )
        if infeasible is not None:
            raise infeasible from exc
        raise
    fields = parse_kv_output(text)
    require_fields(fields, ("frontier_scout", "candidate_source"), str(scout_out))
    sidecar = Path(fields["frontier_scout"])
    source = Path(fields["candidate_source"])
    with sidecar.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if meta.get("candidate_id") != cid:
        raise RuntimeError(f"{sidecar}: mismatched candidate_id")
    if Path(meta["candidate_source"]) != source:
        raise RuntimeError(f"{sidecar}: mismatched candidate_source")
    return meta, source, scout_out


def run_preflight(source: Path, run_dir: Path) -> tuple[dict[str, str], Path]:
    preflight_dir = run_dir / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    preflight_out = run_dir / "preflight.out"
    env = dict(os.environ)
    env["WORK"] = str(preflight_dir)
    text = run_command([str(PREFLIGHT), str(source)], preflight_out, env=env)
    fields = parse_kv_output(text)
    require_fields(
        fields,
        ("source_status", "geometry_status", "sat_status", "integrated_status", "promotable"),
        str(preflight_out),
    )
    return fields, preflight_out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--candidate-prefix", default="rfrontier")
    parser.add_argument("--max-den-list", default="1,5")
    parser.add_argument("--layers-list", default="1")
    parser.add_argument("--max-points-list", default="16")
    parser.add_argument("--min-vertices", type=int, default=2)
    parser.add_argument("--min-edges", type=int, default=1)
    parser.add_argument("--split-depth", type=int, default=1)
    parser.add_argument("--min-split-degree", type=int, default=2)
    parser.add_argument("--dsatur-node-limit", type=int, default=100_000)
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()

    try:
        validate_candidate_id(args.candidate_prefix)
        if not SCOUT.is_file():
            raise ValueError(f"missing scout: {SCOUT}")
        if not args.skip_preflight and not PREFLIGHT.is_file():
            raise ValueError(f"missing preflight: {PREFLIGHT}")
        max_dens = parse_int_list(args.max_den_list, "--max-den-list")
        layers = parse_int_list(args.layers_list, "--layers-list")
        max_points = parse_int_list(args.max_points_list, "--max-points-list")
        if args.min_vertices < 2:
            raise ValueError("--min-vertices must be at least 2")
        if args.min_edges < 1:
            raise ValueError("--min-edges must be positive")
        if args.split_depth < 1:
            raise ValueError("--split-depth must be positive")
        if args.min_split_degree < 1:
            raise ValueError("--min-split-degree must be positive")
        if args.dsatur_node_limit < 1:
            raise ValueError("--dsatur-node-limit must be positive")

        args.out_dir.mkdir(parents=True, exist_ok=True)
        candidates: list[dict[str, Any]] = []
        failed_scouts: list[dict[str, Any]] = []
        index = 0
        for max_den in max_dens:
            for layer in layers:
                for max_point in max_points:
                    cid = candidate_id(args.candidate_prefix, max_den, layer, max_point, index)
                    index += 1
                    try:
                        sidecar, source, scout_out = run_scout(
                            out_dir=args.out_dir,
                            cid=cid,
                            max_den=max_den,
                            layers=layer,
                            max_points=max_point,
                            min_vertices=args.min_vertices,
                            min_edges=args.min_edges,
                            split_depth=args.split_depth,
                            min_split_degree=args.min_split_degree,
                            dsatur_node_limit=args.dsatur_node_limit,
                        )
                    except ScoutInfeasible as exc:
                        failed_scouts.append(exc.row)
                        continue
                    run_dir = args.out_dir / cid
                    if args.skip_preflight:
                        preflight: dict[str, str] = {
                            "source_status": "SKIPPED",
                            "geometry_status": "SKIPPED",
                            "sat_status": "SKIPPED",
                            "integrated_status": "SKIPPED",
                            "promotable": "0",
                            "first_blocker": "preflight_skipped",
                        }
                        preflight_out = run_dir / "preflight.out"
                    else:
                        preflight, preflight_out = run_preflight(source, run_dir)
                    candidates.append(
                        {
                            "candidate_id": cid,
                            "run_dir": str(run_dir),
                            "max_den": max_den,
                            "layers": layer,
                            "max_points": max_point,
                            "candidate_source": str(source),
                            "candidate_source_sha256": sha256_file(source),
                            "frontier_scout": str(run_dir / f"{cid}.frontier-scout.json"),
                            "frontier_scout_sha256": sha256_file(run_dir / f"{cid}.frontier-scout.json"),
                            "scout_out": str(scout_out),
                            "scout_stderr": str(scout_out.with_name(scout_out.name + ".stderr")),
                            "preflight_out": str(preflight_out),
                            "preflight_stderr": str(preflight_out.with_name(preflight_out.name + ".stderr")),
                            "preflight_skipped": 1 if args.skip_preflight else 0,
                            "n": sidecar["n"],
                            "m": sidecar["m"],
                            "max_degree": sidecar["max_degree"],
                            "split_vertices": sidecar["split_vertices"],
                            "dsatur_status": sidecar["dsatur_status"],
                            "dsatur_nodes": sidecar["dsatur_nodes"],
                            "source_status": preflight["source_status"],
                            "geometry_status": preflight["geometry_status"],
                            "sat_status": preflight["sat_status"],
                            "integrated_status": preflight["integrated_status"],
                            "first_blocker": preflight.get("first_blocker", "NONE"),
                            "promotable": int(preflight["promotable"]),
                            "solver_heuristic_priority": score_candidate(sidecar),
                        }
                    )
        if not candidates:
            raise ValueError(
                "campaign produced no viable scout candidates; "
                f"failed_scout_count={len(failed_scouts)}"
            )
        candidates.sort(key=lambda row: (-row["solver_heuristic_priority"], str(row["candidate_id"])))
        promotable_candidate_count = sum(1 for row in candidates if int(row["promotable"]) != 0)
        manifest = {
            "schema": "chi6_rational_frontier_campaign.v1",
            "candidate_prefix": args.candidate_prefix,
            "claim_scope": "solver_candidate_campaign_only",
            "priority_claim_scope": PRIORITY_CLAIM_SCOPE,
            "priority_policy": "status_bonus_plus_edge_degree_pressure_plus_capped_dsatur_nodes",
            "dsatur_node_score_cap": DSATUR_NODE_SCORE_CAP,
            "min_vertices": args.min_vertices,
            "min_edges": args.min_edges,
            "sat_claim": "none",
            "chromatic_claim": "none",
            "promotable": 1 if promotable_candidate_count else 0,
            "promotable_candidate_count": promotable_candidate_count,
            "preflight_skipped": 1 if args.skip_preflight else 0,
            "preflight_enabled": 0 if args.skip_preflight else 1,
            "attempted_scout_count": len(candidates) + len(failed_scouts),
            "failed_scout_count": len(failed_scouts),
            "failed_scouts": failed_scouts,
            "campaign_count": len(candidates),
            "ranking": candidates,
        }
        manifest_path = args.out_dir / "campaign.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_rational_frontier_campaign v1")
    print(f"out_dir={args.out_dir}")
    print(f"campaign_json={manifest_path}")
    print(f"campaign_json_sha256={sha256_file(manifest_path)}")
    print(f"candidate_prefix={args.candidate_prefix}")
    print(f"attempted_scout_count={len(candidates) + len(failed_scouts)}")
    print(f"failed_scout_count={len(failed_scouts)}")
    print(f"campaign_count={len(candidates)}")
    print(f"preflight_enabled={0 if args.skip_preflight else 1}")
    print("claim_scope=solver_candidate_campaign_only")
    print(f"priority_claim_scope={PRIORITY_CLAIM_SCOPE}")
    print("priority_policy=status_bonus_plus_edge_degree_pressure_plus_capped_dsatur_nodes")
    print(f"dsatur_node_score_cap={DSATUR_NODE_SCORE_CAP}")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print(f"promotable={1 if promotable_candidate_count else 0}")
    print(f"promotable_candidate_count={promotable_candidate_count}")
    for idx, row in enumerate(candidates):
        print(
            "candidate "
            f"rank={idx} id={row['candidate_id']} n={row['n']} m={row['m']} "
            f"priority={row['solver_heuristic_priority']} dsatur_status={row['dsatur_status']} "
            f"source_status={row['source_status']} geometry_status={row['geometry_status']} "
            f"sat_status={row['sat_status']} integrated_status={row['integrated_status']} "
            f"first_blocker={row['first_blocker']} source={row['candidate_source']} "
            f"source_sha256={row['candidate_source_sha256']}"
        )
    print("status=RATIONAL_FRONTIER_CAMPAIGN_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
