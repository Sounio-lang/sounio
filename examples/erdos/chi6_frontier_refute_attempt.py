#!/usr/bin/env python3
"""Run bounded cube-refutation attempts from a frontier preflight batch.

This is an execution ledger for the chi>=6 search lane. It consumes the
non-promotable `chi6_frontier_campaign_preflight_batch.v1` manifest, executes
machine-safe refute argv entries without a shell, and records exactly what the
repo-local refuter did. Even successful leaf LRAT emission remains
non-promotable here: global UNSAT still requires a checked cube cover plus Lean
LRAT replay and exact Euclidean geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SAT_COLOURING_RE = re.compile(r"\bSAT colouring=([0-9:,-]+)\b")
SUCCESS_STATUS = "subproblem_lrat_artifacts_emitted_unpromotable"
REFUTE = Path(__file__).with_name("cube_sieve_refute_batch.py").resolve()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_top_level_kv(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            continue
        if key in fields:
            raise ValueError(f"duplicate top-level key in refute stdout: {key}")
        fields[key] = value.strip()
    return fields


def parse_nonneg_int(fields: dict[str, str], key: str) -> int | None:
    raw = fields.get(key)
    if raw is None:
        return None
    raw = raw.strip()
    if not raw.isdigit():
        return None
    return int(raw)


def load_preflight_batch(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="ascii") as f:
        meta = json.load(f)
    if meta.get("schema") != "chi6_frontier_campaign_preflight_batch.v1":
        raise ValueError("preflight batch schema must be chi6_frontier_campaign_preflight_batch.v1")
    if meta.get("claim_scope") != "frontier_campaign_preflight_batch_only":
        raise ValueError("preflight batch claim_scope must be frontier_campaign_preflight_batch_only")
    for key in ("sat_claim", "chromatic_claim", "global_unsat_claim", "verified_claim"):
        if meta.get(key) != "none":
            raise ValueError(f"preflight batch must carry {key}=none")
    if not isinstance(meta.get("promotable"), int) or meta["promotable"] != 0:
        raise ValueError("refute attempts only accept non-promotable preflight batches")
    if not isinstance(meta.get("preflights"), list):
        raise ValueError("preflight batch preflights must be a list")
    ready = [row for row in meta["preflights"] if row.get("recommended_next_action") == "prepare_cube_refute_batch"]
    if meta.get("refute_ready_count") != len(ready):
        raise ValueError("preflight batch refute_ready_count does not match ready rows")
    if not ready:
        raise ValueError("preflight batch has no refute-ready rows")
    first = meta.get("first_refute_candidate")
    if not isinstance(first, str) or first != ready[0].get("candidate_id"):
        raise ValueError("preflight batch first_refute_candidate does not match ready rows")
    return meta


def validate_candidate_id(raw: Any, index: int) -> str:
    candidate_id = str(raw)
    if (
        not candidate_id
        or not SAFE_ID_RE.fullmatch(candidate_id)
        or candidate_id in (".", "..")
        or ".." in candidate_id
    ):
        raise ValueError(f"preflight row {index} has unsafe candidate_id: {candidate_id!r}")
    return candidate_id


def resolve_python(program: str) -> str | None:
    resolved = program if Path(program).is_absolute() else shutil.which(program)
    if resolved is None:
        return None
    try:
        candidate = Path(resolved).resolve(strict=True)
        expected = Path(sys.executable).resolve(strict=True)
    except OSError:
        return None
    if candidate == expected:
        return str(expected)
    return None


def validate_argv(raw: Any, candidate_id: str) -> list[str]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{candidate_id}: refute_argv must be a non-empty list")
    argv: list[str] = []
    for index, part in enumerate(raw):
        if not isinstance(part, str) or part == "":
            raise ValueError(f"{candidate_id}: refute_argv[{index}] must be a non-empty string")
        if "\x00" in part:
            raise ValueError(f"{candidate_id}: refute_argv[{index}] contains NUL")
        argv.append(part)
    if len(argv) < 6:
        raise ValueError(f"{candidate_id}: refute_argv must invoke Python, canonical refuter, edge, k, cubes, out_dir")
    python_path = resolve_python(argv[0])
    if python_path is None:
        raise ValueError(f"{candidate_id}: refute_argv[0] must be a Python executable")
    argv[0] = python_path
    try:
        refute_script = Path(argv[1]).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{candidate_id}: refute_argv[1] does not exist: {argv[1]}") from exc
    if refute_script != REFUTE:
        raise ValueError(f"{candidate_id}: refute_argv[1] must be the canonical cube_sieve_refute_batch.py")
    return argv


def count_lrat_artifacts(out_dir: Path) -> int:
    if not out_dir.is_dir():
        return 0
    return sum(1 for path in out_dir.rglob("*.lrat") if path.is_file())


def extract_sat_colouring(stdout: str, stderr: str) -> tuple[str, int]:
    match = SAT_COLOURING_RE.search(f"{stdout}\n{stderr}")
    if match is None:
        return "NONE", 0
    colouring = match.group(1)
    if not colouring:
        return "NONE", 0
    return colouring, len([part for part in colouring.split(",") if part])


def success_contract_failures(fields: dict[str, str]) -> list[str]:
    cube_count = parse_nonneg_int(fields, "cube_count")
    solver_unsat_count = parse_nonneg_int(fields, "solver_unsat_count")
    failed_count = parse_nonneg_int(fields, "failed_count")
    failures: list[str] = []
    if fields.get("status") != SUCCESS_STATUS:
        failures.append(f"status={fields.get('status', 'MISSING')}")
    if failed_count != 0:
        failures.append(f"failed_count={fields.get('failed_count', 'MISSING')}")
    if cube_count is None:
        failures.append(f"cube_count={fields.get('cube_count', 'MISSING')}")
    if solver_unsat_count != cube_count:
        failures.append(
            f"solver_unsat_count={fields.get('solver_unsat_count', 'MISSING')} cube_count={fields.get('cube_count', 'MISSING')}"
        )
    if fields.get("formal_proof_checker") != "none":
        failures.append(f"formal_proof_checker={fields.get('formal_proof_checker', 'MISSING')}")
    if fields.get("verified_claim") != "none":
        failures.append(f"verified_claim={fields.get('verified_claim', 'MISSING')}")
    if fields.get("global_unsat_claim") != "none":
        failures.append(f"global_unsat_claim={fields.get('global_unsat_claim', 'MISSING')}")
    if fields.get("promotable") != "0":
        failures.append(f"promotable={fields.get('promotable', 'MISSING')}")
    return failures


def classify_refute_result(
    returncode: int,
    fields: dict[str, str],
    stdout: str,
    stderr: str,
    lrat_artifact_count_on_disk: int,
) -> tuple[str, str]:
    if returncode == 0:
        failures = success_contract_failures(fields)
        if not failures:
            expected_lrat_count = parse_nonneg_int(fields, "lrat_artifact_count")
            if expected_lrat_count is None:
                expected_lrat_count = parse_nonneg_int(fields, "solver_unsat_count")
            if expected_lrat_count is None or lrat_artifact_count_on_disk < expected_lrat_count:
                return (
                    "REFUTE_FAILED_INFRA_SUMMARY_MISMATCH",
                    "summary_contract_failed:"
                    f"lrat_artifact_count_on_disk={lrat_artifact_count_on_disk}"
                    f" expected={expected_lrat_count}",
                )
            return "REFUTE_SUCCESS_UNPROMOTABLE", "leaf_lrat_artifacts_emitted_no_global_claim"
        return "REFUTE_FAILED_INFRA_SUMMARY_MISMATCH", "summary_contract_failed:" + ",".join(failures)

    combined = f"{stdout}\n{stderr}"
    if (
        SAT_COLOURING_RE.search(combined)
        or re.search(r"\bNORESULT code=1\b", combined, re.IGNORECASE)
        or re.search(r"(?m)^(s\s+SATISFIABLE|status=SAT)\s*$", combined)
        or re.search(r"(?<!UN)\bSAT\b", combined)
        or "exited 0 without UNSAT marker" in combined
    ):
        return "REFUTE_SAT_MUTATE_FRONTIER", "solver_did_not_refute_at_least_one_cube"
    if re.search(r"\bNORESULT\b", combined, re.IGNORECASE):
        return "REFUTE_NORESULT_MUTATE_FRONTIER", "solver_returned_noresult_for_at_least_one_cube"
    return "REFUTE_FAILED_INFRA", "refuter_failed_without_solver_noresult_marker"


def run_attempt(row: dict[str, Any], index: int, out_dir: Path, timeout_seconds: int | None) -> dict[str, Any]:
    candidate_id = validate_candidate_id(row.get("candidate_id"), index)
    argv = validate_argv(row.get("refute_argv"), candidate_id)
    attempt_dir = out_dir / candidate_id
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = attempt_dir / "refute.stdout"
    stderr_path = attempt_dir / "refute.stderr"

    timed_out = False
    try:
        proc = subprocess.run(
            argv,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_seconds,
        )
        stdout = proc.stdout
        stderr = proc.stderr
        returncode = proc.returncode
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        stderr = f"{stderr}\ntimeout after {timeout_seconds} seconds\n"
        returncode = 124

    stdout_path.write_text(stdout, encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")
    fields = parse_top_level_kv(stdout)
    sat_colouring, sat_colouring_vertex_count = extract_sat_colouring(stdout, stderr)
    lrat_artifact_count_on_disk = count_lrat_artifacts(Path(argv[5]))
    if timed_out:
        classified_status = "REFUTE_FAILED_TIMEOUT"
        classification_note = "refuter_exceeded_timeout"
    else:
        classified_status, classification_note = classify_refute_result(
            returncode,
            fields,
            stdout,
            stderr,
            lrat_artifact_count_on_disk,
        )

    return {
        "index": index,
        "candidate_id": candidate_id,
        "argv": argv,
        "returncode": returncode,
        "classified_status": classified_status,
        "classification_note": classification_note,
        "stdout": str(stdout_path),
        "stdout_sha256": sha256_file(stdout_path),
        "stdout_nonempty": 1 if stdout else 0,
        "stdout_excerpt": stdout[:1000],
        "stderr": str(stderr_path),
        "stderr_sha256": sha256_file(stderr_path),
        "stderr_nonempty": 1 if stderr else 0,
        "stderr_excerpt": stderr[:1000],
        "refuter_status": fields.get("status", "NONE"),
        "sat_colouring": sat_colouring,
        "sat_colouring_vertex_count": sat_colouring_vertex_count,
        "cube_count": parse_nonneg_int(fields, "cube_count"),
        "solver_unsat_count": parse_nonneg_int(fields, "solver_unsat_count"),
        "refuter_lrat_artifact_count": parse_nonneg_int(fields, "lrat_artifact_count"),
        "lrat_artifact_count_on_disk": lrat_artifact_count_on_disk,
        "refuter_sha256": sha256_file(REFUTE),
        "failed_count": parse_nonneg_int(fields, "failed_count"),
        "formal_proof_checker": fields.get("formal_proof_checker", "NONE"),
        "verified_claim": fields.get("verified_claim", "NONE"),
        "global_unsat_claim": fields.get("global_unsat_claim", "NONE"),
        "promotable": fields.get("promotable", "NONE"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("preflight_batch_json", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="maximum refute-ready rows to run; 0 means all")
    parser.add_argument("--timeout-seconds", type=int, default=0, help="per-attempt timeout; 0 means no timeout")
    args = parser.parse_args()

    try:
        if args.limit < 0:
            raise ValueError("--limit must be non-negative")
        if args.timeout_seconds < 0:
            raise ValueError("--timeout-seconds must be non-negative")
        batch = load_preflight_batch(args.preflight_batch_json)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ready = [
            row
            for row in batch["preflights"]
            if row.get("recommended_next_action") == "prepare_cube_refute_batch"
        ]
        if args.limit:
            ready = ready[: args.limit]
        attempts = [
            run_attempt(
                row,
                index,
                args.out_dir,
                args.timeout_seconds if args.timeout_seconds else None,
            )
            for index, row in enumerate(ready)
        ]
        success = [row for row in attempts if row["classified_status"] == "REFUTE_SUCCESS_UNPROMOTABLE"]
        failed = [row for row in attempts if row["classified_status"] != "REFUTE_SUCCESS_UNPROMOTABLE"]
        status_counts: dict[str, int] = {}
        for row in attempts:
            status = row["classified_status"]
            status_counts[status] = status_counts.get(status, 0) + 1
        manifest = {
            "schema": "chi6_frontier_refute_attempt.v1",
            "preflight_batch_json": str(args.preflight_batch_json),
            "preflight_batch_sha256": sha256_file(args.preflight_batch_json),
            "attempt_count": len(attempts),
            "refute_success_count": len(success),
            "refute_failed_count": len(failed),
            "first_success_candidate": success[0]["candidate_id"] if success else "NONE",
            "first_failed_candidate": failed[0]["candidate_id"] if failed else "NONE",
            "status_counts": status_counts,
            "claim_scope": "frontier_refute_attempt_only",
            "sat_claim": "none",
            "chromatic_claim": "none",
            "global_unsat_claim": "none",
            "verified_claim": "none",
            "promotable": 0,
            "promotion_gate": "requires_checked_cube_cover_lrat_lean_exact_geometry_real_bridge",
            "attempt_order": "preflight_batch_ready_order",
            "attempts": attempts,
        }
        manifest_path = args.out_dir / "frontier_refute_attempt.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="ascii")
    except (OSError, RuntimeError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("chi6_frontier_refute_attempt v1")
    print(f"preflight_batch_json={args.preflight_batch_json}")
    print(f"preflight_batch_sha256={sha256_file(args.preflight_batch_json)}")
    print(f"attempt_count={manifest['attempt_count']}")
    print(f"refute_success_count={manifest['refute_success_count']}")
    print(f"refute_failed_count={manifest['refute_failed_count']}")
    print(f"first_success_candidate={manifest['first_success_candidate']}")
    print(f"first_failed_candidate={manifest['first_failed_candidate']}")
    print(f"refute_attempt_json={manifest_path}")
    print("claim_scope=frontier_refute_attempt_only")
    print("sat_claim=none")
    print("chromatic_claim=none")
    print("global_unsat_claim=none")
    print("verified_claim=none")
    print("promotable=0")
    print("status=FRONTIER_REFUTE_ATTEMPT_RECORDED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
