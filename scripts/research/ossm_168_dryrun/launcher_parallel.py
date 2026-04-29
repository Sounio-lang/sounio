#!/usr/bin/env python3
"""Parallel launcher for OSSM-168 processing using external processes.

Spawns N concurrent subprocesses, each processing one subject.
This avoids pickling issues with MNE and allows true parallelism.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class SubjectResult:
    subject_id: str
    n_epochs: int
    F1: float
    F2: float
    F3: float
    status: str


def run_single_subject(
    sid: str,
    si: int,
    raw_root: Path,
    cache_dir: Path,
    script_path: Path,
    max_epochs: int,
) -> SubjectResult:
    """Run a single subject in a subprocess."""
    cmd = [
        sys.executable,
        str(script_path),
        "--subject", sid,
        "--subject-index", str(si),
        "--raw-root", str(raw_root),
        "--cache-dir", str(cache_dir),
        "--max-epochs", str(max_epochs),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=1200,  # 20 minute timeout per subject
        )

        if result.returncode != 0:
            stderr = result.stderr[:200] if result.stderr else ""
            return SubjectResult(
                subject_id=sid, n_epochs=0, F1=0, F2=0, F3=0,
                status=f"error: {stderr}"
            )

        # Parse JSON output
        try:
            data = json.loads(result.stdout.strip().split('\n')[-1])
        except json.JSONDecodeError:
            return SubjectResult(
                subject_id=sid, n_epochs=0, F1=0, F2=0, F3=0,
                status="json_parse_error"
            )

        if data.get("status") == "success":
            return SubjectResult(
                subject_id=sid,
                n_epochs=data.get("n_epochs", 0),
                F1=data.get("F1", 0.0),
                F2=data.get("F2", 0.0),
                F3=data.get("F3", 0.0),
                status="success",
            )
        else:
            return SubjectResult(
                subject_id=sid, n_epochs=0, F1=0, F2=0, F3=0,
                status=data.get("status", "unknown_error")
            )

    except subprocess.TimeoutExpired:
        return SubjectResult(
            subject_id=sid, n_epochs=0, F1=0, F2=0, F3=0,
            status="timeout"
        )
    except Exception as e:
        return SubjectResult(
            subject_id=sid, n_epochs=0, F1=0, F2=0, F3=0,
            status=f"exception: {type(e).__name__}: {e}"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--endpoints", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--subjects-list", type=Path, default=None)
    ap.add_argument("--max-epochs", type=int, default=50)
    ap.add_argument("--workers", type=int, default=4,
                    help="Concurrent subprocesses (default: 4)")
    ap.add_argument("--script", type=Path,
                    default=Path(__file__).parent / "process_single_subject.py")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load subjects
    if args.subjects_list:
        subjects = [s.strip() for s in args.subjects_list.read_text().splitlines() if s.strip()]
    else:
        subjects = sorted(d.name for d in args.raw_root.iterdir() if d.is_dir() and d.name.startswith("sub-"))

    # Load endpoints for subject index mapping
    endpoints: dict[str, int] = {}
    with args.endpoints.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            endpoints[row["subject_id"]] = i

    print(f"[launcher] {len(subjects)} subjects total")
    print(f"[launcher] {len(endpoints)} subjects with endpoint indices")

    # Check existing features
    feat_csv = args.out_dir / "features.csv"
    existing: set[str] = set()
    if feat_csv.exists():
        with feat_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                existing.add(row["subject_id"])
        print(f"[launcher] {len(existing)} subjects already processed")

    # Filter to subjects needing processing
    to_process = [
        (sid, endpoints.get(sid, 0))
        for sid in subjects
        if sid not in existing and sid in endpoints
    ]
    print(f"[launcher] {len(to_process)} subjects to process")

    if not to_process:
        print("[launcher] Nothing to do.")
        return 0

    results: list[SubjectResult] = []
    log_entries: list[dict] = []

    print(f"[launcher] Starting with {args.workers} concurrent workers...")
    t0_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_single_subject,
                sid,
                si,
                args.raw_root,
                args.cache_dir,
                args.script,
                args.max_epochs,
            ): sid
            for sid, si in to_process
        }

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            sid = futures[future]
            completed += 1

            try:
                result = future.result()
                results.append(result)

                log_entries.append({
                    "subject": result.subject_id,
                    "status": result.status,
                    "n_epochs": result.n_epochs,
                })

                if result.status == "success":
                    print(f"[{completed}/{len(to_process)}] {sid}: "
                          f"n_ep={result.n_epochs} F1={result.F1:.4f} F2={result.F2:.4f} F3={result.F3:.4f}")
                else:
                    print(f"[{completed}/{len(to_process)}] {sid}: FAILED ({result.status})")

            except Exception as e:
                print(f"[{completed}/{len(to_process)}] {sid}: EXCEPTION {e}")
                log_entries.append({"subject": sid, "status": f"launcher_exception: {e}"})

            # Save progress every 10 subjects
            if completed % 10 == 0:
                _save_progress(results, existing, feat_csv, log_entries, args.out_dir)

    # Final save
    _save_progress(results, existing, feat_csv, log_entries, args.out_dir)

    elapsed = time.time() - t0_start
    successful = sum(1 for r in results if r.status == "success")
    print(f"\n[launcher] Completed {completed} subjects in {elapsed:.1f}s")
    print(f"[launcher] Successful: {successful}/{completed}")

    return 0


def _save_progress(
    results: list[SubjectResult],
    existing: set[str],
    feat_csv: Path,
    log_entries: list[dict],
    out_dir: Path,
) -> None:
    """Save incremental progress."""
    # Read existing features
    all_rows: list[dict] = []
    if feat_csv.exists():
        with feat_csv.open(encoding="utf-8") as fh:
            all_rows.extend(csv.DictReader(fh))

    # Add new results
    for r in results:
        if r.status == "success" and r.subject_id not in existing:
            all_rows.append({
                "subject_id": r.subject_id,
                "n_epochs": r.n_epochs,
                "F1": r.F1,
                "F2": r.F2,
                "F3": r.F3,
            })
            existing.add(r.subject_id)

    # Write features
    if all_rows:
        with feat_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["subject_id", "n_epochs", "F1", "F2", "F3"])
            w.writeheader()
            w.writerows(all_rows)

    # Write log
    (out_dir / "processing_log.json").write_text(json.dumps(log_entries, indent=2))


if __name__ == "__main__":
    sys.exit(main())
