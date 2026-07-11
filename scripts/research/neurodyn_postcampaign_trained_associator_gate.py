#!/usr/bin/env python3
"""Gate trained ROI associator extraction after an O-SSM campaign.

The previous model-free associator failure mode was: no trained checkpoint was
loaded, yet ROI statistics were still produced.  This gate refuses to proceed
unless it can select a verified, nonzero O-SSM checkpoint with balanced accuracy
above threshold.  It is dry-run by default; pass --execute-associator to run the
native trained ROI associator extractor.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


SCHEMA = "brain_ossm.neurodyn_postcampaign_trained_associator_gate.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument("--checkpoint-root", action="append", default=[])
    parser.add_argument("--run-config", default="", help="Optional run config to replay checkpoint split settings.")
    parser.add_argument("--run-id", default="", help="Optional exact checkpoint run_id filter.")
    parser.add_argument("--min-balanced-accuracy", type=float, default=51.0)
    parser.add_argument("--tolerance-pp", type=float, default=0.5)
    parser.add_argument("--scope", choices=["all", "holdout"], default="all")
    parser.add_argument("--metric", default="hidden_delta_abs")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--associator-executor", choices=["native", "python"], default="native")
    parser.add_argument("--native-souc-engine", default="", help="Optional SOUNIO_SOUC_ENGINE for native verifier/extractor.")
    parser.add_argument("--verify-checkpoint", action="store_true")
    parser.add_argument("--execute-associator", action="store_true")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    return parser.parse_args()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def candidate_paths(args: argparse.Namespace) -> list[Path]:
    paths: dict[Path, None] = {}
    for raw in args.checkpoint:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            paths[path] = None
    for raw_root in args.checkpoint_root:
        root = Path(raw_root).expanduser().resolve()
        if not root.exists():
            continue
        for pattern in ["model.ckpt", "*.ckpt"]:
            for path in root.rglob(pattern):
                if path.is_file():
                    paths[path.resolve()] = None
    return sorted(paths)


def load_checkpoint(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - diagnostic gate
        return None, f"json_load_failed:{type(exc).__name__}:{exc}"
    return payload, ""


def checkpoint_signal(payload: dict[str, Any]) -> bool:
    values = payload.get("arrays", {}).get("A", {}).get("values", [])
    return bool(values) and any(abs(float(value)) > 1.0e-12 for value in values)


def float_field(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    if value is None or value == "":
        return default
    return float(value)


def inspect_checkpoint(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    payload, error = load_checkpoint(path)
    row: dict[str, Any] = {
        "path": str(path),
        "status": "fail",
        "failures": [],
        "run_id": "",
        "balanced_accuracy_pct": 0.0,
        "delta_pp": 999.0,
        "verification_status": "",
        "has_nonzero_a_tensor": False,
    }
    if payload is None:
        row["failures"].append(error)
        return row

    row["run_id"] = str(payload.get("run_id") or "")
    meta = payload.get("meta") or {}
    if meta.get("model") != "O-SSM":
        row["failures"].append(f"model is not O-SSM: {meta.get('model')!r}")
    if args.run_id and row["run_id"] != args.run_id:
        row["failures"].append(f"run_id mismatch: {row['run_id']} != {args.run_id}")

    verification = payload.get("verification")
    if not isinstance(verification, dict):
        row["failures"].append("missing verification block")
    else:
        row["verification_status"] = str(verification.get("status") or "")
        row["balanced_accuracy_pct"] = float_field(verification, "actual_balanced_accuracy_pct", 0.0)
        row["delta_pp"] = float_field(verification, "delta_pp", 999.0)
        tolerance = float_field(verification, "tolerance_pp", args.tolerance_pp)
        if verification.get("status") != "pass":
            row["failures"].append(f"verification status is {verification.get('status')!r}")
        if row["balanced_accuracy_pct"] <= args.min_balanced_accuracy:
            row["failures"].append(
                f"balanced accuracy {row['balanced_accuracy_pct']:.6f} <= {args.min_balanced_accuracy:.6f}"
            )
        if row["delta_pp"] > tolerance:
            row["failures"].append(f"verification delta {row['delta_pp']:.6f}pp > {tolerance:.6f}pp")

    row["has_nonzero_a_tensor"] = checkpoint_signal(payload)
    if not row["has_nonzero_a_tensor"]:
        row["failures"].append("A tensor missing or all-zero")

    if not row["failures"]:
        row["status"] = "pass"
    return row


def select_checkpoint(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    passing = [row for row in rows if row.get("status") == "pass"]
    if not passing:
        return None
    return max(passing, key=lambda row: (float(row.get("balanced_accuracy_pct") or 0.0), str(row.get("path") or "")))


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-20000:],
        "stderr": proc.stderr[-20000:],
    }


def load_json_file(path: Path) -> tuple[dict[str, Any] | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), ""
    except Exception as exc:  # noqa: BLE001 - diagnostic gate
        return None, f"json_load_failed:{type(exc).__name__}:{exc}"


def validate_associator_summary(path: Path, args: argparse.Namespace) -> tuple[dict[str, Any] | None, list[str]]:
    failures: list[str] = []
    payload, error = load_json_file(path)
    if payload is None:
        return None, [f"associator summary unreadable: {error}"]
    verification = payload.get("extractor_verification") or payload.get("native_extractor_verification")
    if not isinstance(verification, dict):
        failures.append("associator summary missing extractor verification")
        return payload, failures
    actual = float_field(verification, "actual_balanced_accuracy_pct", 0.0)
    expected = float_field(verification, "expected_balanced_accuracy_pct", 0.0)
    delta = float_field(verification, "delta_pp", abs(actual - expected))
    if actual <= args.min_balanced_accuracy:
        failures.append(f"associator verifier balanced accuracy {actual:.6f} <= {args.min_balanced_accuracy:.6f}")
    if delta > args.tolerance_pp:
        failures.append(f"associator verifier delta {delta:.6f}pp > {args.tolerance_pp:.6f}pp")
    feature_rows = int(float_field(payload, "feature_rows", 0.0))
    roi_count = int(float_field(payload, "roi_count", 0.0))
    if feature_rows <= 0:
        failures.append("associator emitted no feature rows")
    if roi_count <= 0:
        failures.append("associator emitted no ROI stats")
    top10 = path.parent / "top10_roi_biomarkers.csv"
    if not top10.is_file() or top10.stat().st_size <= 0:
        failures.append(f"associator top10 missing or empty: {top10}")
    method = str(payload.get("method") or "")
    if args.associator_executor == "native" and "native" not in method:
        failures.append(f"associator method is not native: {method!r}")
    return payload, failures


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest = Path(args.manifest).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    initial_failures: list[str] = []
    if not manifest.is_file():
        initial_failures.append(f"manifest missing: {manifest}")

    candidates = candidate_paths(args)
    rows = [inspect_checkpoint(path, args) for path in candidates]
    selected = select_checkpoint(rows)
    failures: list[str] = list(initial_failures)
    commands: dict[str, Any] = {}
    selected_copy = output_dir / "selected_model.ckpt"
    staged_manifest = manifest
    staged_checkpoint = selected_copy
    staged_run_config = Path(args.run_config).expanduser().resolve() if args.run_config else None

    if not candidates:
        failures.append("no checkpoint candidates found")
    if selected is None:
        failures.append("no verified trained checkpoint passed the gate")
    else:
        shutil.copy2(Path(selected["path"]), selected_copy)

    with tempfile.TemporaryDirectory(prefix="neurodyn-assoc-gate-") as stage_raw:
        stage_dir = Path(stage_raw)
        if manifest.is_file():
            staged_manifest = stage_dir / "manifest.tsv"
            shutil.copy2(manifest, staged_manifest)
        if selected_copy.exists():
            staged_checkpoint = stage_dir / "model.ckpt"
            shutil.copy2(selected_copy, staged_checkpoint)
        if staged_run_config is not None and staged_run_config.is_file():
            staged_run_config_copy = stage_dir / "run_config.tsv"
            shutil.copy2(staged_run_config, staged_run_config_copy)
            staged_run_config = staged_run_config_copy

        if selected is not None and args.verify_checkpoint:
            verify_cmd = [
                sys.executable,
                str(repo_root / "scripts/research/abide_checkpoint_persist.py"),
                "--verify-only",
                "--checkpoint",
                str(staged_checkpoint),
                "--manifest",
                str(staged_manifest),
                "--tolerance-pp",
                str(args.tolerance_pp),
            ]
            if staged_run_config is not None:
                verify_cmd.extend(["--run-config", str(staged_run_config)])
            if args.native_souc_engine:
                verify_cmd.extend(["--native-souc-engine", args.native_souc_engine])
            commands["verify_checkpoint"] = run_command(verify_cmd, repo_root)
            if commands["verify_checkpoint"]["returncode"] != 0:
                failures.append(f"checkpoint reverify failed rc={commands['verify_checkpoint']['returncode']}")
            else:
                shutil.copy2(staged_checkpoint, selected_copy)
                selected = inspect_checkpoint(selected_copy, args)
                if selected.get("status") != "pass":
                    failures.append("checkpoint failed after reverify")

        associator_output = output_dir / "trained_roi_associator"
        if args.execute_associator and selected is not None and not failures:
            associator_cmd = [
                sys.executable,
                str(repo_root / "scripts/research/abide_trained_roi_associator.py"),
                "--checkpoint",
                str(staged_checkpoint),
                "--manifest",
                str(staged_manifest),
                "--output-dir",
                str(associator_output),
                "--scope",
                args.scope,
                "--metric",
                args.metric,
                "--alpha",
                str(args.alpha),
                "--min-balanced-accuracy",
                str(args.min_balanced_accuracy),
                "--executor",
                args.associator_executor,
                "--repo-root",
                str(repo_root),
            ]
            if args.native_souc_engine:
                associator_cmd.extend(["--native-souc-engine", args.native_souc_engine])
            commands["trained_roi_associator"] = run_command(associator_cmd, repo_root)
            if commands["trained_roi_associator"]["returncode"] != 0:
                failures.append(f"trained ROI associator failed rc={commands['trained_roi_associator']['returncode']}")
            else:
                associator_summary, associator_failures = validate_associator_summary(
                    associator_output / "run_summary.json",
                    args,
                )
                commands["trained_roi_associator"]["summary"] = associator_summary
                failures.extend(associator_failures)
        elif args.execute_associator and failures:
            failures.append("refusing to execute associator because gate failed")

    decision = "ready_for_trained_associator" if selected is not None and not failures else "blocked"
    if args.execute_associator and decision == "ready_for_trained_associator":
        decision = "associator_complete"

    summary = {
        "schema": SCHEMA,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "decision": decision,
        "failures": failures,
        "manifest": str(manifest),
        "output_dir": str(output_dir),
        "candidate_count": len(candidates),
        "candidates": rows,
        "selected_checkpoint": selected,
        "selected_checkpoint_copy": str(selected_copy) if selected_copy.exists() else "",
        "execute_associator": bool(args.execute_associator),
        "associator_executor": args.associator_executor,
        "native_souc_engine": args.native_souc_engine,
        "associator_output_dir": str(associator_output),
        "commands": commands,
        "claim_boundary": (
            "This gate permits trained associator extraction from verified checkpoints; "
            "it does not certify a scientific result by itself."
        ),
    }
    write_json(output_dir / "postcampaign_trained_associator_gate.json", summary)

    print(f"decision={decision}")
    print(f"candidate_count={len(candidates)}")
    if selected:
        print(f"selected_checkpoint={selected.get('path')}")
        print(f"selected_balanced_accuracy={float(selected.get('balanced_accuracy_pct') or 0.0):.6f}")
    print(f"summary={output_dir / 'postcampaign_trained_associator_gate.json'}")
    for failure in failures:
        print(f"BLOCKED: {failure}", file=sys.stderr)
    return 0 if decision in {"ready_for_trained_associator", "associator_complete"} else 1


if __name__ == "__main__":
    sys.exit(main())
