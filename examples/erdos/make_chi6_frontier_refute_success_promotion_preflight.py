#!/usr/bin/env python3
"""Bind a successful frontier refute row to the integrated chi6 preflight.

This is the promotion-adjacent bridge for the search lane. It consumes the
same successful `REFUTE_SUCCESS_UNPROMOTABLE` attempt/sweep accepted by the SAT
packager, follows the selected row back to its exact rational source package,
then runs the existing integrated preflight with the selected cube batch and a
cover DRUP/RUP proof.

The output is still non-promotable. It emits
`READY_FOR_CANDIDATE_PROMOTION_WIRING` only when the source package, exact
geometry lane, and arbitrary cube-cover SAT lane all pass for the same graph
identity.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from make_chi6_frontier_refute_success_sat_manifest import (
    SUCCESS,
    as_nonneg_int,
    load_attempt_from_input,
    load_json,
    load_preflight_batch,
    require_file_hash,
    require_no_claims,
    resolve_recorded_path,
    select_success_row,
    sha256_file,
    validate_preflight_row,
    validate_refuter_stdout,
    validate_success_row,
)


ROOT = Path(__file__).resolve().parents[2]
INTEGRATED = Path(__file__).with_name("make_chi6_integrated_candidate_preflight.sh")
READY = "READY_FOR_CANDIDATE_PROMOTION_WIRING"
INCOMPLETE = "INCOMPLETE"


def parse_kv_output(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in fields:
            raise ValueError(f"duplicate key in integrated preflight output: {key}")
        fields[key] = value
    return fields


def require_fields(fields: dict[str, str], keys: tuple[str, ...], label: str) -> None:
    missing = [key for key in keys if key not in fields]
    if missing:
        raise ValueError(f"{label}: missing fields: {','.join(missing)}")


def require_same_path(actual: Path, expected: Path, label: str) -> None:
    if actual.resolve() != expected.resolve():
        raise ValueError(f"{label} mismatch: got {actual.resolve()}, expected {expected.resolve()}")


def validate_campaign_preflight(
    *,
    preflight_row: dict[str, Any],
    preflight_batch_path: Path,
    selected: dict[str, Any],
) -> tuple[dict[str, Any], Path, Path, Path]:
    campaign_path = resolve_recorded_path(
        preflight_batch_path,
        preflight_row.get("campaign_preflight_json"),
        "campaign_preflight_json",
    )
    require_file_hash(
        campaign_path,
        preflight_row.get("campaign_preflight_sha256"),
        "campaign_preflight_json",
    )
    campaign = load_json(campaign_path)
    if campaign.get("schema") != "chi6_frontier_campaign_preflight.v1":
        raise ValueError("campaign preflight schema must be chi6_frontier_campaign_preflight.v1")
    if campaign.get("claim_scope") != "deterministic_campaign_preflight_only":
        raise ValueError("campaign preflight claim_scope must be deterministic_campaign_preflight_only")
    require_no_claims(campaign, "campaign preflight")

    candidate_id = selected["candidate_id"]
    if campaign.get("candidate_id") != candidate_id:
        raise ValueError(
            f"campaign preflight candidate_id mismatch: got {campaign.get('candidate_id')!r}, "
            f"expected {candidate_id!r}"
        )
    if campaign.get("source_status") != "PASS":
        raise ValueError(f"{candidate_id}: campaign preflight source_status must be PASS")
    if campaign.get("recommended_next_action") != "prepare_cube_refute_batch":
        raise ValueError(f"{candidate_id}: campaign preflight is not refute-ready")

    raw_k = campaign.get("k")
    if raw_k != 5 and raw_k != "5":
        raise ValueError(f"{candidate_id}: campaign preflight k must be 5")

    source_path = resolve_recorded_path(
        campaign_path,
        campaign.get("candidate_source_path"),
        "candidate_source_path",
    )
    require_file_hash(source_path, campaign.get("candidate_source_sha256"), "candidate_source_path")

    edge_path = Path(str(campaign.get("edge_path_abs", ""))).resolve()
    require_same_path(edge_path, selected["edge_path"], "campaign edge_path_abs")
    if campaign.get("edge_sha256") != sha256_file(selected["edge_path"]):
        raise ValueError("campaign edge_sha256 does not match selected refute edge")

    cube_batch_path = resolve_recorded_path(
        campaign_path,
        campaign.get("cube_batch_path"),
        "cube_batch_path",
    )
    require_same_path(cube_batch_path, selected["cube_batch_path"], "campaign cube_batch_path")
    if campaign.get("cube_batch_sha256") != sha256_file(selected["cube_batch_path"]):
        raise ValueError("campaign cube_batch_sha256 does not match selected refute cube batch")

    coords_path = Path(str(campaign.get("coords_path_abs", ""))).resolve()
    if not coords_path.is_file():
        raise ValueError(f"coords_path_abs missing: {coords_path}")
    require_file_hash(coords_path, campaign.get("coords_sha256"), "coords_path_abs")

    return campaign, campaign_path, source_path, coords_path


def run_integrated_preflight(
    *,
    source_path: Path,
    cube_batch_path: Path,
    cover_proof: Path,
    work_dir: Path,
) -> tuple[dict[str, str], Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = work_dir.parent / "integrated_preflight.out"
    stderr_path = work_dir.parent / "integrated_preflight.err"
    env = os.environ.copy()
    # The integrated shell preflight uses WORK as its artifact root.
    env["WORK"] = str(work_dir)
    proc = subprocess.run(
        [
            "bash",
            str(INTEGRATED),
            str(source_path),
            str(cube_batch_path),
            str(cover_proof),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    stdout_path.write_text(proc.stdout, encoding="utf-8")
    stderr_path.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"integrated preflight failed with exit {proc.returncode}: {detail}")
    fields = parse_kv_output(proc.stdout)
    require_fields(
        fields,
        (
            "source_status",
            "geometry_status",
            "sat_status",
            "sat_route_mode",
            "integrated_status",
            "first_blocker",
            "claim_scope",
            "promotable",
            "chromatic_claim",
        ),
        str(stdout_path),
    )
    if fields["source_status"] != "PASS":
        raise ValueError("integrated preflight did not validate the source package")
    if fields["sat_route_mode"] != "arbitrary_complement_cube_cover":
        raise ValueError("integrated preflight did not use the arbitrary complement cube-cover route")
    if fields["claim_scope"] != "integrated_preflight_only":
        raise ValueError("integrated preflight emitted unexpected claim_scope")
    if fields["promotable"] != "0" or fields["chromatic_claim"] != "none":
        raise ValueError("integrated preflight leaked a promotable/chromatic claim")

    expected_status = READY if fields["geometry_status"] == "PASS" and fields["sat_status"] == "PASS" else INCOMPLETE
    if fields["integrated_status"] != expected_status:
        raise ValueError(
            "integrated preflight status is inconsistent with geometry/SAT status: "
            f"got {fields['integrated_status']}, expected {expected_status}"
        )
    return fields, stdout_path, stderr_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="frontier refute attempt JSON or sweep JSON")
    parser.add_argument("cover_drup_or_rup", type=Path, help="DRUP/RUP proof for the complement-cover CNF")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--success-candidate", help="select a specific successful candidate_id")
    parser.add_argument("--resume", action="store_true", help="allow reusing and overwriting a non-empty output directory")
    args = parser.parse_args()

    try:
        input_json = args.input_json.resolve()
        cover_proof = args.cover_drup_or_rup.resolve()
        if not input_json.is_file():
            raise ValueError(f"missing input JSON: {input_json}")
        if not cover_proof.is_file() or cover_proof.stat().st_size == 0:
            raise ValueError(f"missing/empty cover proof: {cover_proof}")
        if not INTEGRATED.is_file():
            raise ValueError(f"missing integrated preflight script: {INTEGRATED}")
        if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
            raise ValueError("out_dir already exists and is non-empty; pass --resume to reuse it")
        args.out_dir.mkdir(parents=True, exist_ok=True)

        attempt, attempt_path, lineage = load_attempt_from_input(input_json)
        preflight, preflight_path = load_preflight_batch(attempt, attempt_path)
        index, row = select_success_row(attempt, wanted_candidate=args.success_candidate)
        selected = validate_success_row(row, attempt_path)
        preflight_row = validate_preflight_row(preflight, row)
        validate_refuter_stdout(selected)
        campaign, campaign_path, source_path, coords_path = validate_campaign_preflight(
            preflight_row=preflight_row,
            preflight_batch_path=preflight_path,
            selected=selected,
        )
        integrated, integrated_stdout, integrated_stderr = run_integrated_preflight(
            source_path=source_path,
            cube_batch_path=selected["cube_batch_path"],
            cover_proof=cover_proof,
            work_dir=args.out_dir / "integrated",
        )

        promotion_ready = 1 if integrated["integrated_status"] == READY else 0
        status = (
            "FRONTIER_REFUTE_SUCCESS_PROMOTION_PREFLIGHT_READY"
            if promotion_ready
            else "FRONTIER_REFUTE_SUCCESS_PROMOTION_PREFLIGHT_INCOMPLETE"
        )
        lineage.update(
            {
                "schema": "chi6_frontier_refute_success_promotion_preflight.v1",
                "claim_scope": "frontier_refute_success_promotion_preflight_only",
                "refute_attempt_json": str(attempt_path),
                "refute_attempt_json_sha256": sha256_file(attempt_path),
                "preflight_batch_json": str(preflight_path),
                "preflight_batch_sha256": sha256_file(preflight_path),
                "selected_attempt_index": index,
                "selected_attempt_status": SUCCESS,
                "frontier_candidate_id": selected["candidate_id"],
                "candidate_id": selected["candidate_id"],
                "campaign_preflight_json": str(campaign_path),
                "campaign_preflight_sha256": sha256_file(campaign_path),
                "candidate_source_path": str(source_path),
                "candidate_source_sha256": sha256_file(source_path),
                "edge_path": str(selected["edge_path"]),
                "edge_sha256": sha256_file(selected["edge_path"]),
                "coords_path": str(coords_path),
                "coords_sha256": sha256_file(coords_path),
                "cube_batch_path": str(selected["cube_batch_path"]),
                "cube_batch_sha256": sha256_file(selected["cube_batch_path"]),
                "cover_drup_or_rup": str(cover_proof),
                "cover_drup_or_rup_sha256": sha256_file(cover_proof),
                "integrated_work_dir": str(args.out_dir / "integrated"),
                "integrated_stdout": str(integrated_stdout),
                "integrated_stdout_sha256": sha256_file(integrated_stdout),
                "integrated_stderr": str(integrated_stderr),
                "integrated_stderr_sha256": sha256_file(integrated_stderr),
                "source_status": integrated["source_status"],
                "geometry_status": integrated["geometry_status"],
                "sat_status": integrated["sat_status"],
                "integrated_status": integrated["integrated_status"],
                "first_blocker": integrated["first_blocker"],
                "promotion_ready": promotion_ready,
                "promotable": 0,
                "chromatic_claim": "none",
                "geometry_claim": "preflight_only_exact_geometry_if_geometry_status_PASS",
                "sat_claim": "preflight_only_cube_cover_if_sat_status_PASS",
                "promotion_gate": "requires_promotable_candidate_assembler_and_validator_after_READY",
                "status": status,
            }
        )
        lineage_path = args.out_dir / "frontier_refute_success_promotion_preflight.json"
        lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_refute_success_promotion_preflight v1")
    print(f"input_json={input_json}")
    print(f"input_schema={lineage['input_schema']}")
    print(f"refute_attempt_json={attempt_path}")
    print(f"preflight_batch_json={preflight_path}")
    print(f"frontier_candidate_id={selected['candidate_id']}")
    print(f"candidate_id={selected['candidate_id']}")
    print(f"campaign_preflight_json={campaign_path}")
    print(f"campaign_preflight_json_sha256={sha256_file(campaign_path)}")
    print(f"candidate_source_path={source_path}")
    print(f"candidate_source_sha256={sha256_file(source_path)}")
    print(f"edge_path={selected['edge_path']}")
    print(f"edge_sha256={sha256_file(selected['edge_path'])}")
    print(f"coords_path={coords_path}")
    print(f"coords_sha256={sha256_file(coords_path)}")
    print(f"cube_batch_path={selected['cube_batch_path']}")
    print(f"cube_batch_sha256={sha256_file(selected['cube_batch_path'])}")
    print(f"cover_drup_or_rup={cover_proof}")
    print(f"cover_drup_or_rup_sha256={sha256_file(cover_proof)}")
    print(f"integrated_stdout={integrated_stdout}")
    print(f"integrated_stderr={integrated_stderr}")
    print(f"lineage_json={lineage_path}")
    print(f"lineage_json_sha256={sha256_file(lineage_path)}")
    print(f"source_status={integrated['source_status']}")
    print(f"geometry_status={integrated['geometry_status']}")
    print(f"sat_status={integrated['sat_status']}")
    print(f"integrated_status={integrated['integrated_status']}")
    print(f"first_blocker={integrated['first_blocker']}")
    print(f"promotion_ready={promotion_ready}")
    print("claim_scope=frontier_refute_success_promotion_preflight_only")
    print("chromatic_claim=none")
    print("promotable=0")
    print(f"status={status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
