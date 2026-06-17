#!/usr/bin/env python3
"""Package a successful frontier refute attempt into a SAT candidate manifest.

This is the bridge from the search ledger to the reusable SAT-certification
lane. It consumes either a `chi6_frontier_refute_attempt.v1` manifest or a
`chi6_frontier_refute_sweep.v1` manifest containing a successful attempt,
validates the selected `REFUTE_SUCCESS_UNPROMOTABLE` row, then delegates to the
existing arbitrary cube-cover manifest maker.

The output is deliberately non-promotable: it can package a finite
`colourCNF` UNSAT certificate for the selected edge/cube batch, but it makes no
Euclidean geometry or chi(R^2) claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REFUTER = Path(__file__).with_name("cube_sieve_refute_batch.py").resolve()
MAKER = Path(__file__).with_name("make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh")
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
NO_CLAIM_KEYS = ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim")
SUCCESS = "REFUTE_SUCCESS_UNPROMOTABLE"


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return meta


def resolve_recorded_path(base_json: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be a non-empty path string")
    path = Path(raw)
    if not path.is_absolute():
        path = base_json.resolve().parent / path
    return path.resolve()


def require_file_hash(path: Path, expected: Any, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    if not isinstance(expected, str) or not HEX64_RE.fullmatch(expected.lower()):
        raise ValueError(f"{label} has malformed recorded SHA256: {expected!r}")
    actual = sha256_file(path)
    if actual != expected.lower():
        raise ValueError(f"{label} SHA256 mismatch: got {actual}, expected {expected}")


def require_no_claims(meta: dict[str, Any], label: str) -> None:
    for key in NO_CLAIM_KEYS:
        if meta.get(key) != "none":
            raise ValueError(f"{label} must carry {key}=none")
    if meta.get("promotable") != 0:
        raise ValueError(f"{label} must carry promotable=0")


def as_nonneg_int(raw: Any, label: str) -> int:
    if isinstance(raw, int) and raw >= 0:
        return raw
    if isinstance(raw, str) and raw.isdigit():
        return int(raw)
    raise ValueError(f"{label} must be a non-negative integer, got {raw!r}")


def validate_candidate_id(raw: Any, label: str) -> str:
    candidate_id = str(raw)
    if (
        not candidate_id
        or not SAFE_ID_RE.fullmatch(candidate_id)
        or candidate_id in (".", "..")
        or ".." in candidate_id
    ):
        raise ValueError(f"unsafe {label}: {candidate_id!r}")
    return candidate_id


def parse_refuter_stdout(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    top: dict[str, str] = {}
    rows: list[dict[str, str]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "cube_sieve_refute_batch v1":
        raise ValueError(f"{path}: missing cube_sieve_refute_batch v1 header")
    for lineno, raw in enumerate(lines[1:], 2):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("cube "):
            row: dict[str, str] = {}
            for token in line.split()[1:]:
                if "=" not in token:
                    raise ValueError(f"{path}:{lineno}: malformed cube token {token!r}")
                key, value = token.split("=", 1)
                if key in row:
                    raise ValueError(f"{path}:{lineno}: duplicate cube field {key}")
                row[key] = value
            rows.append(row)
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            if key in top:
                raise ValueError(f"{path}:{lineno}: duplicate field {key}")
            top[key] = value
            continue
        raise ValueError(f"{path}:{lineno}: unexpected line {line!r}")
    return top, rows


def parse_manifest(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    for lineno, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{path}:{lineno}: manifest line lacks '='")
        key, value = line.split("=", 1)
        if key in fields:
            raise ValueError(f"{path}:{lineno}: duplicate manifest key {key}")
        fields[key] = value
    return fields


def load_attempt_from_input(input_json: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    meta = load_json(input_json)
    schema = meta.get("schema")
    lineage: dict[str, Any] = {
        "input_json": str(input_json),
        "input_json_sha256": sha256_file(input_json),
        "input_schema": schema,
    }
    if schema == "chi6_frontier_refute_attempt.v1":
        return meta, input_json, lineage

    if schema != "chi6_frontier_refute_sweep.v1":
        raise ValueError("input schema must be chi6_frontier_refute_attempt.v1 or chi6_frontier_refute_sweep.v1")
    require_no_claims(meta, "frontier refute sweep")
    cells = meta.get("cells")
    if not isinstance(cells, list):
        raise ValueError("frontier refute sweep cells must be a list")
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        if as_nonneg_int(cell.get("refute_success_count", 0), "cell refute_success_count") == 0:
            continue
        attempt_path = resolve_recorded_path(input_json, cell.get("refute_attempt_json"), "cell refute_attempt_json")
        recorded_sha = cell.get("refute_attempt_sha256")
        if isinstance(recorded_sha, str) and recorded_sha != "NONE":
            require_file_hash(attempt_path, recorded_sha, "cell refute_attempt_json")
        attempt = load_json(attempt_path)
        lineage.update(
            {
                "sweep_json": str(input_json),
                "sweep_json_sha256": sha256_file(input_json),
                "sweep_cell_index": cell.get("cell_index"),
                "sweep_cell_dir": cell.get("cell_dir"),
            }
        )
        return attempt, attempt_path, lineage
    raise ValueError("frontier refute sweep has no successful refute attempt cell")


def load_preflight_batch(attempt: dict[str, Any], attempt_path: Path) -> tuple[dict[str, Any], Path]:
    preflight_path = resolve_recorded_path(attempt_path, attempt.get("preflight_batch_json"), "preflight_batch_json")
    require_file_hash(preflight_path, attempt.get("preflight_batch_sha256"), "preflight_batch_json")
    preflight = load_json(preflight_path)
    if preflight.get("schema") != "chi6_frontier_campaign_preflight_batch.v1":
        raise ValueError("preflight batch schema must be chi6_frontier_campaign_preflight_batch.v1")
    if preflight.get("claim_scope") != "frontier_campaign_preflight_batch_only":
        raise ValueError("preflight batch claim_scope must be frontier_campaign_preflight_batch_only")
    require_no_claims(preflight, "preflight batch")
    return preflight, preflight_path


def select_success_row(
    attempt: dict[str, Any],
    *,
    wanted_candidate: str | None,
) -> tuple[int, dict[str, Any]]:
    if attempt.get("schema") != "chi6_frontier_refute_attempt.v1":
        raise ValueError("refute attempt schema must be chi6_frontier_refute_attempt.v1")
    if attempt.get("claim_scope") != "frontier_refute_attempt_only":
        raise ValueError("refute attempt claim_scope must be frontier_refute_attempt_only")
    require_no_claims(attempt, "refute attempt")
    attempts = attempt.get("attempts")
    if not isinstance(attempts, list):
        raise ValueError("refute attempt attempts must be a list")
    success_count = as_nonneg_int(attempt.get("refute_success_count"), "refute_success_count")
    status_counts = attempt.get("status_counts")
    if not isinstance(status_counts, dict):
        raise ValueError("refute attempt status_counts must be an object")
    if as_nonneg_int(status_counts.get(SUCCESS, 0), f"status_counts.{SUCCESS}") != success_count:
        raise ValueError("refute_success_count does not match status_counts")
    if success_count <= 0:
        raise ValueError("refute attempt has no REFUTE_SUCCESS_UNPROMOTABLE row")

    for index, row in enumerate(attempts):
        if not isinstance(row, dict):
            raise ValueError(f"attempt row {index} is not an object")
        if row.get("classified_status") != SUCCESS:
            continue
        candidate_id = str(row.get("candidate_id", ""))
        if wanted_candidate is not None and candidate_id != wanted_candidate:
            continue
        return index, row
    if wanted_candidate is None:
        raise ValueError("no REFUTE_SUCCESS_UNPROMOTABLE row found")
    raise ValueError(f"no successful refute row for candidate_id={wanted_candidate}")


def validate_preflight_row(preflight: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    rows = preflight.get("preflights")
    if not isinstance(rows, list):
        raise ValueError("preflight batch preflights must be a list")
    candidate_id = row["candidate_id"]
    attempt_argv = row.get("argv")
    if not isinstance(attempt_argv, list):
        raise ValueError(f"{candidate_id}: attempt argv is not a list")

    def resolve_program(raw: str) -> Path:
        path = Path(raw)
        if not path.is_absolute():
            resolved = shutil.which(raw)
            if resolved is None:
                raise ValueError(f"{candidate_id}: cannot resolve executable in refute_argv: {raw}")
            path = Path(resolved)
        return path.resolve()

    for index, preflight_row in enumerate(rows):
        if not isinstance(preflight_row, dict):
            raise ValueError(f"preflight row {index} is not an object")
        if preflight_row.get("candidate_id") != candidate_id:
            continue
        if preflight_row.get("recommended_next_action") != "prepare_cube_refute_batch":
            raise ValueError(f"{candidate_id}: preflight row is not refute-ready")
        preflight_argv = preflight_row.get("refute_argv")
        if (
            not isinstance(preflight_argv, list)
            or len(preflight_argv) != len(attempt_argv)
            or not preflight_argv
            or resolve_program(str(preflight_argv[0])) != resolve_program(str(attempt_argv[0]))
            or preflight_argv[1:] != attempt_argv[1:]
        ):
            raise ValueError(f"{candidate_id}: preflight row refute_argv does not match attempt argv")
        return preflight_row
    raise ValueError(f"preflight batch does not contain selected candidate_id={candidate_id}")


def validate_success_row(row: dict[str, Any], attempt_path: Path) -> dict[str, Any]:
    candidate_id = validate_candidate_id(row.get("candidate_id", ""), "candidate_id")
    if as_nonneg_int(row.get("returncode"), f"{candidate_id}.returncode") != 0:
        raise ValueError(f"{candidate_id}: success row returncode is not 0")
    if row.get("classification_note") != "leaf_lrat_artifacts_emitted_no_global_claim":
        raise ValueError(f"{candidate_id}: unexpected classification_note")
    required_string_fields = {
        "refuter_status": "subproblem_lrat_artifacts_emitted_unpromotable",
        "formal_proof_checker": "none",
        "verified_claim": "none",
        "global_unsat_claim": "none",
        "promotable": "0",
    }
    for key, expected in required_string_fields.items():
        if row.get(key) != expected:
            raise ValueError(f"{candidate_id}: expected {key}={expected}, got {row.get(key)!r}")
    cube_count = as_nonneg_int(row.get("cube_count"), f"{candidate_id}.cube_count")
    solver_unsat_count = as_nonneg_int(row.get("solver_unsat_count"), f"{candidate_id}.solver_unsat_count")
    failed_count = as_nonneg_int(row.get("failed_count"), f"{candidate_id}.failed_count")
    lrat_count = as_nonneg_int(row.get("refuter_lrat_artifact_count"), f"{candidate_id}.refuter_lrat_artifact_count")
    disk_lrat_count = as_nonneg_int(row.get("lrat_artifact_count_on_disk"), f"{candidate_id}.lrat_artifact_count_on_disk")
    if cube_count <= 0:
        raise ValueError(f"{candidate_id}: cube_count must be positive")
    if solver_unsat_count != cube_count:
        raise ValueError(f"{candidate_id}: solver_unsat_count does not match cube_count")
    if failed_count != 0:
        raise ValueError(f"{candidate_id}: failed_count must be 0")
    if lrat_count != cube_count:
        raise ValueError(f"{candidate_id}: refuter_lrat_artifact_count does not match cube_count")
    if disk_lrat_count < lrat_count:
        raise ValueError(f"{candidate_id}: lrat_artifact_count_on_disk is smaller than declared LRAT count")

    stdout_path = resolve_recorded_path(attempt_path, row.get("stdout"), f"{candidate_id}.stdout")
    stderr_path = resolve_recorded_path(attempt_path, row.get("stderr"), f"{candidate_id}.stderr")
    require_file_hash(stdout_path, row.get("stdout_sha256"), f"{candidate_id}.stdout")
    require_file_hash(stderr_path, row.get("stderr_sha256"), f"{candidate_id}.stderr")

    argv = row.get("argv")
    if not isinstance(argv, list) or len(argv) < 6 or not all(isinstance(x, str) and x for x in argv):
        raise ValueError(f"{candidate_id}: argv must be the canonical refuter argv list")
    if Path(argv[1]).resolve() != REFUTER:
        raise ValueError(f"{candidate_id}: argv[1] must be canonical cube_sieve_refute_batch.py")
    if argv[3] != "5":
        raise ValueError(f"{candidate_id}: only k=5 frontier successes can enter the chi6 SAT manifest route")
    edge_path = Path(argv[2]).resolve()
    cube_batch_path = Path(argv[4]).resolve()
    refute_out_dir = Path(argv[5]).resolve()
    if not edge_path.is_file():
        raise ValueError(f"{candidate_id}: edge file missing: {edge_path}")
    if not cube_batch_path.is_file():
        raise ValueError(f"{candidate_id}: cube batch missing: {cube_batch_path}")
    if not refute_out_dir.is_dir():
        raise ValueError(f"{candidate_id}: refute output dir missing: {refute_out_dir}")

    return {
        "candidate_id": candidate_id,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "edge_path": edge_path,
        "cube_batch_path": cube_batch_path,
        "refute_out_dir": refute_out_dir,
        "cube_count": cube_count,
        "lrat_artifact_count": lrat_count,
    }


def validate_refuter_stdout(selected: dict[str, Any]) -> dict[str, Any]:
    stdout_path = selected["stdout_path"]
    top, rows = parse_refuter_stdout(stdout_path)
    expected = {
        "formula_kind": "colourCNF",
        "k": "5",
        "failed_count": "0",
        "formal_proof_checker": "none",
        "verified_claim": "none",
        "global_unsat_claim": "none",
        "geometry_claim": "none",
        "promotable": "0",
        "status": "subproblem_lrat_artifacts_emitted_unpromotable",
    }
    for key, value in expected.items():
        if top.get(key) != value:
            raise ValueError(f"refuter stdout expected {key}={value}, got {top.get(key)!r}")
    if Path(top.get("edge_path", "")).resolve() != selected["edge_path"]:
        raise ValueError("refuter stdout edge_path does not match selected argv")
    if Path(top.get("cube_batch_path", "")).resolve() != selected["cube_batch_path"]:
        raise ValueError("refuter stdout cube_batch_path does not match selected argv")
    if Path(top.get("out_dir", "")).resolve() != selected["refute_out_dir"]:
        raise ValueError("refuter stdout out_dir does not match selected argv")
    if top.get("edge_sha256") != sha256_file(selected["edge_path"]):
        raise ValueError("refuter stdout edge_sha256 does not match edge file")
    if top.get("cube_batch_sha256") != sha256_file(selected["cube_batch_path"]):
        raise ValueError("refuter stdout cube_batch_sha256 does not match cube batch")
    cube_count = as_nonneg_int(top.get("cube_count"), "refuter cube_count")
    solver_unsat_count = as_nonneg_int(top.get("solver_unsat_count"), "refuter solver_unsat_count")
    lrat_count = as_nonneg_int(top.get("lrat_artifact_count"), "refuter lrat_artifact_count")
    if cube_count != selected["cube_count"] or solver_unsat_count != cube_count or lrat_count != cube_count:
        raise ValueError("refuter stdout cube/LRAT counts do not match selected success row")
    if len(rows) != cube_count:
        raise ValueError(f"refuter stdout has {len(rows)} cube rows, expected {cube_count}")

    for index, row in enumerate(rows):
        cube_id = row.get("id", f"row_{index}")
        for key in ("cube", "cube_sha256", "cnf", "cnf_sha256", "drat", "drat_sha256", "lrat", "lrat_sha256"):
            if key not in row:
                raise ValueError(f"refuter cube row {cube_id} missing {key}")
        if row.get("drat_deletions") != "0":
            raise ValueError(f"refuter cube row {cube_id} contains DRAT deletions")
        if row.get("cnf_clauses") != row.get("expected_cnf_clauses"):
            raise ValueError(f"refuter cube row {cube_id} CNF clause count was not validated")
        for path_key, hash_key in (
            ("cube", "cube_sha256"),
            ("cnf", "cnf_sha256"),
            ("drat", "drat_sha256"),
            ("lrat", "lrat_sha256"),
        ):
            artifact = selected["refute_out_dir"] / row[path_key]
            require_file_hash(artifact, row[hash_key], f"refuter row {cube_id} {path_key}")
    return {"refuter_top": top, "refuter_cube_row_count": len(rows)}


def run_maker(
    *,
    out_dir: Path,
    edge_path: Path,
    candidate_id: str,
    cube_batch_path: Path,
    cover_proof: Path,
) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    maker_stdout = out_dir / "frontier_refute_success_sat_manifest.maker.out"
    maker_stderr = out_dir / "frontier_refute_success_sat_manifest.maker.err"
    env = os.environ.copy()
    env["WORK"] = str(out_dir)
    proc = subprocess.run(
        [
            "bash",
            str(MAKER),
            str(edge_path),
            candidate_id,
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
    maker_stdout.write_text(proc.stdout, encoding="utf-8")
    maker_stderr.write_text(proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip() or "no output"
        raise RuntimeError(f"arbitrary cube-cover maker failed with exit {proc.returncode}: {detail}")
    manifest = out_dir / "candidate.manifest"
    if not manifest.is_file():
        raise RuntimeError("arbitrary cube-cover maker did not emit candidate.manifest")
    return manifest, maker_stdout, maker_stderr


def validate_emitted_manifest(manifest: Path, candidate_id: str, edge_path: Path, cube_batch_path: Path) -> dict[str, str]:
    fields = parse_manifest(manifest)
    checks = {
        "promotable": "0",
        "candidate_id": candidate_id,
        "k": "5",
        "geometry_module_path": "NONE",
        "geometry_module_sha256": "NONE",
        "geometry_proof_type": "none",
        "sat_proof_route": "cube_cover_generic",
        "triangle_sb": "none",
        "cube_cover_certificate_path": "NONE",
        "cube_cover_certificate_sha256": "NONE",
        "chromatic_claim": "none",
        "geometry_claim": "none",
    }
    for key, expected in checks.items():
        if fields.get(key) != expected:
            raise ValueError(f"candidate.manifest expected {key}={expected}, got {fields.get(key)!r}")
    for key in ("cube_cover_complement_cnf_path", "cube_cover_complement_lrat_path"):
        if fields.get(key, "NONE") == "NONE":
            raise ValueError(f"candidate.manifest missing concrete {key}")
    if fields.get("edge_sha256") != sha256_file(edge_path):
        raise ValueError("candidate.manifest edge_sha256 does not match selected frontier edge")
    if fields.get("cube_batch_sha256") != sha256_file(cube_batch_path):
        raise ValueError("candidate.manifest cube_batch_sha256 does not match selected frontier cube batch")
    validator_log = manifest.parent / "manifest_validator.log"
    if not validator_log.is_file():
        raise ValueError("manifest_validator.log missing")
    expected_line = f"chi6_manifest: VALID_NONPROMOTABLE_FORMAT candidate={candidate_id}"
    if expected_line not in validator_log.read_text(encoding="utf-8"):
        raise ValueError("manifest validator log does not contain VALID_NONPROMOTABLE_FORMAT")
    return fields


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_json", type=Path, help="frontier refute attempt JSON or sweep JSON")
    parser.add_argument("cover_drup_or_rup", type=Path, help="DRUP/RUP proof for the complement-cover CNF")
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--success-candidate", help="select a specific successful candidate_id")
    parser.add_argument("--candidate-id", help="override the emitted candidate.manifest candidate_id")
    parser.add_argument("--resume", action="store_true", help="allow reusing and overwriting a non-empty output directory")
    args = parser.parse_args()

    try:
        input_json = args.input_json.resolve()
        cover_proof = args.cover_drup_or_rup.resolve()
        if not input_json.is_file():
            raise ValueError(f"missing input JSON: {input_json}")
        if not cover_proof.is_file() or cover_proof.stat().st_size == 0:
            raise ValueError(f"missing/empty cover proof: {cover_proof}")
        if not MAKER.is_file():
            raise ValueError(f"missing arbitrary cube-cover maker: {MAKER}")
        if args.out_dir.exists() and any(args.out_dir.iterdir()) and not args.resume:
            raise ValueError("out_dir already exists and is non-empty; pass --resume to reuse it")

        attempt, attempt_path, lineage = load_attempt_from_input(input_json)
        preflight, preflight_path = load_preflight_batch(attempt, attempt_path)
        index, row = select_success_row(attempt, wanted_candidate=args.success_candidate)
        selected = validate_success_row(row, attempt_path)
        validate_preflight_row(preflight, row)
        refuter_info = validate_refuter_stdout(selected)

        candidate_id = validate_candidate_id(args.candidate_id or selected["candidate_id"], "output candidate_id")

        manifest, maker_stdout, maker_stderr = run_maker(
            out_dir=args.out_dir,
            edge_path=selected["edge_path"],
            candidate_id=candidate_id,
            cube_batch_path=selected["cube_batch_path"],
            cover_proof=cover_proof,
        )
        manifest_fields = validate_emitted_manifest(
            manifest,
            candidate_id,
            selected["edge_path"],
            selected["cube_batch_path"],
        )

        lineage.update(
            {
                "schema": "chi6_frontier_refute_success_sat_manifest.v1",
                "claim_scope": "frontier_refute_success_sat_packaging_only",
                "refute_attempt_json": str(attempt_path),
                "refute_attempt_json_sha256": sha256_file(attempt_path),
                "preflight_batch_json": str(preflight_path),
                "preflight_batch_sha256": sha256_file(preflight_path),
                "selected_attempt_index": index,
                "frontier_candidate_id": selected["candidate_id"],
                "candidate_id": candidate_id,
                "edge_path": str(selected["edge_path"]),
                "edge_sha256": sha256_file(selected["edge_path"]),
                "cube_batch_path": str(selected["cube_batch_path"]),
                "cube_batch_sha256": sha256_file(selected["cube_batch_path"]),
                "refute_stdout": str(selected["stdout_path"]),
                "refute_stdout_sha256": sha256_file(selected["stdout_path"]),
                "refute_stderr": str(selected["stderr_path"]),
                "refute_stderr_sha256": sha256_file(selected["stderr_path"]),
                "refute_out_dir": str(selected["refute_out_dir"]),
                "cube_count": selected["cube_count"],
                "lrat_artifact_count": selected["lrat_artifact_count"],
                "refuter_cube_row_count": refuter_info["refuter_cube_row_count"],
                "cover_drup_or_rup": str(cover_proof),
                "cover_drup_or_rup_sha256": sha256_file(cover_proof),
                "candidate_manifest": str(manifest),
                "candidate_manifest_sha256": sha256_file(manifest),
                "maker_stdout": str(maker_stdout),
                "maker_stdout_sha256": sha256_file(maker_stdout),
                "maker_stderr": str(maker_stderr),
                "maker_stderr_sha256": sha256_file(maker_stderr),
                "lean_sat_module_path": manifest_fields["lean_sat_module_path"],
                "sat_proof_route": manifest_fields["sat_proof_route"],
                "chromatic_claim": "none",
                "geometry_claim": "none",
                "euclidean_claim": "none",
                "promotable": 0,
                "promotion_gate": "requires_exact_euclidean_geometry_real_bridge_before_chi6_claim",
                "status": "FRONTIER_REFUTE_SUCCESS_SAT_MANIFEST_PACKAGED",
            }
        )
        lineage_path = args.out_dir / "frontier_refute_success_sat_manifest.json"
        lineage_path.write_text(json.dumps(lineage, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_refute_success_sat_manifest v1")
    print(f"input_json={input_json}")
    print(f"input_schema={lineage['input_schema']}")
    print(f"refute_attempt_json={attempt_path}")
    print(f"preflight_batch_json={preflight_path}")
    print(f"frontier_candidate_id={selected['candidate_id']}")
    print(f"candidate_id={candidate_id}")
    print(f"edge_path={selected['edge_path']}")
    print(f"edge_sha256={sha256_file(selected['edge_path'])}")
    print(f"cube_batch_path={selected['cube_batch_path']}")
    print(f"cube_batch_sha256={sha256_file(selected['cube_batch_path'])}")
    print(f"cube_count={selected['cube_count']}")
    print(f"lrat_artifact_count={selected['lrat_artifact_count']}")
    print(f"cover_drup_or_rup={cover_proof}")
    print(f"cover_drup_or_rup_sha256={sha256_file(cover_proof)}")
    print(f"candidate_manifest={manifest}")
    print(f"candidate_manifest_sha256={sha256_file(manifest)}")
    print(f"lineage_json={lineage_path}")
    print(f"lineage_json_sha256={sha256_file(lineage_path)}")
    print("claim_scope=frontier_refute_success_sat_packaging_only")
    print("sat_claim_scope=finite_colourCNF_edge_only")
    print("chromatic_claim=none")
    print("geometry_claim=none")
    print("euclidean_claim=none")
    print("promotable=0")
    print("status=FRONTIER_REFUTE_SUCCESS_SAT_MANIFEST_PACKAGED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
