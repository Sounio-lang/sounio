#!/usr/bin/env python3
"""Run frontier-to-cube preflight for ranked rational campaign candidates.

This is queue plumbing for the chi>=6 search lane. It consumes a
`chi6_rational_frontier_campaign.v1` manifest, runs the deterministic cube
campaign preflight for a bounded prefix of ranked candidates, and emits the
exact next refutation commands. It does not run SAT refutation and carries no
chromatic claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PREFLIGHT = Path(__file__).with_name("chi6_frontier_campaign_preflight.py")
REFUTE = Path(__file__).with_name("cube_sieve_refute_batch.py")
KNOWN_DSATUR_STATUSES = {
    "K_COLORING_FOUND",
    "UNKNOWN_NODE_LIMIT",
    "NO_K_COLORING_FOUND_BY_CPU_PROBE_NONCERTIFYING",
}


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_kv_output(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key in fields:
            raise ValueError(f"duplicate key in tool output: {key}")
        fields[key] = value
    return fields


def require_fields(fields: dict[str, str], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in fields]
    if missing:
        raise RuntimeError(f"{label}: missing fields: {','.join(missing)}")


def run_command(cmd: list[str], out_path: Path) -> str:
    proc = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out_path.write_text(proc.stdout, encoding="utf-8")
    out_path.with_name(out_path.name + ".stderr").write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"command failed with exit {proc.returncode}: {' '.join(cmd)}: {detail}")
    return proc.stdout


def load_campaign(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if meta.get("schema") != "chi6_rational_frontier_campaign.v1":
        raise ValueError("campaign schema must be chi6_rational_frontier_campaign.v1")
    if meta.get("claim_scope") != "solver_candidate_campaign_only":
        raise ValueError("campaign claim_scope must be solver_candidate_campaign_only")
    if meta.get("sat_claim") != "none" or meta.get("chromatic_claim") != "none":
        raise ValueError("campaign must carry no SAT/chromatic claim")
    if not isinstance(meta.get("promotable"), int) or meta["promotable"] != 0:
        raise ValueError("campaign preflight batch only accepts non-promotable campaigns")
    if not isinstance(meta.get("ranking"), list):
        raise ValueError("campaign ranking must be a list")
    if "campaign_count" not in meta:
        raise ValueError("campaign missing campaign_count")
    if not isinstance(meta["campaign_count"], int):
        raise ValueError("campaign_count must be an integer")
    if int(meta["campaign_count"]) < 1 or not meta["ranking"]:
        raise ValueError("campaign ranking is empty")
    if int(meta["campaign_count"]) != len(meta["ranking"]):
        raise ValueError("campaign_count does not match ranking length")
    return meta


def parse_nonneg_int(fields: dict[str, str], key: str, label: str) -> int:
    raw = fields[key]
    if not raw or any(c < "0" or c > "9" for c in raw):
        raise RuntimeError(f"{label}: field {key} must be a non-negative integer, got {raw!r}")
    return int(raw)


def resolve_campaign_path(campaign_dir: Path, raw: str, label: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = campaign_dir / path
    resolved = path.resolve()
    base = campaign_dir.resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside campaign directory: {raw}") from exc
    return resolved


def refute_argv_from_preflight(preflight_meta: dict[str, Any]) -> list[str]:
    for key in ("edge_path_abs", "k", "cube_batch_path"):
        if key not in preflight_meta:
            raise RuntimeError(f"preflight JSON missing refute argv field: {key}")
    if not REFUTE.is_file():
        raise RuntimeError(f"missing refute script: {REFUTE}")
    # The legacy `refute_command` remains a human-facing command string. The
    # structured argv below is the machine-safe handoff.
    refute_path = str(REFUTE)
    cube_batch = str(preflight_meta["cube_batch_path"])
    out_dir = str(Path(cube_batch).with_suffix("").parent / "refute")
    return [
        sys.executable,
        refute_path,
        str(preflight_meta["edge_path_abs"]),
        str(preflight_meta["k"]),
        cube_batch,
        out_dir,
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--max-cubes", type=int, default=1_000_000)
    parser.add_argument("--sample-hard-cubes", type=int, default=5)
    parser.add_argument("--skip-coloring-found", action="store_true")
    args = parser.parse_args()

    try:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.max_cubes <= 0:
            raise ValueError("--max-cubes must be positive")
        if args.sample_hard_cubes < 0:
            raise ValueError("--sample-hard-cubes must be non-negative")
        if not PREFLIGHT.is_file():
            raise ValueError(f"missing preflight script: {PREFLIGHT}")
        if not REFUTE.is_file():
            raise ValueError(f"missing refute script: {REFUTE}")
        if args.sample_hard_cubes > args.max_cubes:
            raise ValueError("--sample-hard-cubes cannot exceed --max-cubes")
        campaign = load_campaign(args.campaign_json)
        campaign_dir = args.campaign_json.resolve().parent
        args.out_dir.mkdir(parents=True, exist_ok=True)

        selected: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for rank, row in enumerate(campaign["ranking"]):
            if not isinstance(row, dict):
                raise ValueError(f"campaign ranking row {rank} is not an object")
            for key in ("candidate_id", "frontier_scout", "dsatur_status"):
                if key not in row:
                    raise ValueError(f"campaign ranking row {rank} missing field: {key}")
            dsatur_status = str(row["dsatur_status"]).strip()
            if dsatur_status not in KNOWN_DSATUR_STATUSES:
                raise ValueError(f"campaign ranking row {rank} has unknown dsatur_status: {dsatur_status}")
            row = {**row, "dsatur_status": dsatur_status}
            if args.skip_coloring_found and dsatur_status == "K_COLORING_FOUND":
                skipped.append(
                    {
                        "rank": rank,
                        "candidate_id": row["candidate_id"],
                        "reason": "coloring_found_skipped_by_flag",
                    }
                )
                continue
            if len(selected) >= args.limit:
                skipped.append(
                    {
                        "rank": rank,
                        "candidate_id": row["candidate_id"],
                        "reason": "limit_reached",
                    }
                )
                continue
            selected.append({"rank": rank, **row})
        if not selected:
            raise ValueError("no campaign candidates selected for preflight")

        rows: list[dict[str, Any]] = []
        action_counts: dict[str, int] = {}
        for row in selected:
            candidate_id = str(row["candidate_id"])
            scout = resolve_campaign_path(campaign_dir, str(row["frontier_scout"]), "frontier_scout")
            if not scout.is_file():
                raise ValueError(f"missing frontier scout for {candidate_id}: {scout}")
            run_dir = args.out_dir / candidate_id
            run_dir.mkdir(parents=True, exist_ok=True)
            stdout_path = run_dir / "campaign_preflight.out"
            text = run_command(
                [
                    sys.executable,
                    str(PREFLIGHT),
                    str(scout),
                    str(run_dir / "preflight"),
                    "--max-cubes",
                    str(args.max_cubes),
                    "--sample-hard-cubes",
                    str(args.sample_hard_cubes),
                ],
                stdout_path,
            )
            fields = parse_kv_output(text)
            require_fields(
                fields,
                (
                    "candidate_id",
                    "campaign_id",
                    "campaign_preflight_json",
                    "recommended_next_action",
                    "cube_count",
                    "propagation_conflict_count",
                    "propagation_hard_count",
                    "sat_claim",
                    "chromatic_claim",
                    "promotable",
                    "status",
                ),
                str(stdout_path),
            )
            if fields["candidate_id"] != candidate_id:
                raise RuntimeError(f"{candidate_id}: preflight returned mismatched candidate_id")
            if fields["sat_claim"] != "none" or fields["chromatic_claim"] != "none":
                raise RuntimeError(f"{candidate_id}: preflight leaked a SAT/chromatic claim")
            if fields["promotable"] != "0":
                raise RuntimeError(f"{candidate_id}: preflight leaked promotable={fields['promotable']}")
            if fields["status"] != "FRONTIER_CAMPAIGN_PREFLIGHT_READY":
                raise RuntimeError(f"{candidate_id}: unexpected preflight status {fields['status']}")
            preflight_json = Path(fields["campaign_preflight_json"])
            if not preflight_json.is_file():
                raise RuntimeError(f"{candidate_id}: missing preflight JSON {preflight_json}")
            with preflight_json.open("r", encoding="ascii") as f:
                preflight_meta = json.load(f)
            if preflight_meta.get("promotable") != 0:
                raise RuntimeError(f"{candidate_id}: preflight JSON leaked promotable claim")
            action = fields["recommended_next_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
            stderr_path = stdout_path.with_name(stdout_path.name + ".stderr")
            stderr_text = stderr_path.read_text(encoding="utf-8")
            refute_command = str(preflight_meta.get("refute_command") or "NONE")
            refute_argv = refute_argv_from_preflight(preflight_meta)
            rows.append(
                {
                    "rank": row["rank"],
                    "candidate_id": candidate_id,
                    "dsatur_status": row["dsatur_status"],
                    "solver_heuristic_priority": row.get("solver_heuristic_priority"),
                    "frontier_scout": str(scout),
                    "frontier_scout_sha256": sha256_file(scout),
                    "preflight_stdout": str(stdout_path),
                    "preflight_stderr": str(stderr_path),
                    "preflight_stderr_nonempty": 1 if stderr_text else 0,
                    "preflight_stderr_excerpt": stderr_text[:500],
                    "campaign_preflight_json": str(preflight_json),
                    "campaign_preflight_sha256": sha256_file(preflight_json),
                    "campaign_id": fields["campaign_id"],
                    "cube_count": parse_nonneg_int(fields, "cube_count", str(stdout_path)),
                    "propagation_conflict_count": parse_nonneg_int(
                        fields, "propagation_conflict_count", str(stdout_path)
                    ),
                    "propagation_hard_count": parse_nonneg_int(
                        fields, "propagation_hard_count", str(stdout_path)
                    ),
                    "recommended_next_action": action,
                    "refute_command": refute_command,
                    "refute_argv": refute_argv,
                    "foundry_handoff_recommended": int(preflight_meta.get("foundry_handoff_recommended", 0)),
                }
            )

        refute_ready = [row for row in rows if row["recommended_next_action"] == "prepare_cube_refute_batch"]
        manifest = {
            "schema": "chi6_frontier_campaign_preflight_batch.v1",
            "campaign_json": str(args.campaign_json),
            "campaign_json_sha256": sha256_file(args.campaign_json),
            "campaign_count": int(campaign["campaign_count"]),
            "selected_count": len(rows),
            "skipped_count": len(skipped),
            "skipped": skipped,
            "action_counts": action_counts,
            "refute_ready_count": len(refute_ready),
            "first_refute_candidate": refute_ready[0]["candidate_id"] if refute_ready else "NONE",
            "first_refute_command": refute_ready[0]["refute_command"] if refute_ready else "NONE",
            "first_refute_argv": refute_ready[0]["refute_argv"] if refute_ready else [],
            "claim_scope": "frontier_campaign_preflight_batch_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_refute_lrat_cover_lrat_lean_exact_geometry_real_bridge",
            "preflights": rows,
        }
        manifest_path = args.out_dir / "frontier_campaign_preflight_batch.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_campaign_preflight_batch v1")
    print(f"campaign_json={args.campaign_json}")
    print(f"campaign_json_sha256={sha256_file(args.campaign_json)}")
    print(f"selected_count={len(rows)}")
    print(f"skipped_count={len(skipped)}")
    print(f"refute_ready_count={len(refute_ready)}")
    print(f"first_refute_candidate={manifest['first_refute_candidate']}")
    print(f"first_refute_command={manifest['first_refute_command']}")
    print(f"first_refute_argv_json={json.dumps(manifest['first_refute_argv'], separators=(',', ':'))}")
    print(f"preflight_batch_json={manifest_path}")
    print("claim_scope=frontier_campaign_preflight_batch_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=FRONTIER_CAMPAIGN_PREFLIGHT_BATCH_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
