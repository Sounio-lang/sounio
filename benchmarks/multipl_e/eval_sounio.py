#!/usr/bin/env python3
"""
MultiPL-E evaluation script for Sounio.

Runs generated .sio files through souc and reports pass@k metrics.
Compatible with the MultiPL-E evaluation harness.

Usage:
    # Evaluate a single file
    python eval_sounio.py --file generated/humaneval_0.sio

    # Evaluate a directory of generated solutions
    python eval_sounio.py --dir generated/ --k 1,10,100

    # Evaluate with a custom souc binary
    python eval_sounio.py --dir generated/ --souc /path/to/souc

    # Output results as JSONL
    python eval_sounio.py --dir generated/ --output results.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SOUC = os.environ.get(
    "SOUC", "./bin/souc"
)
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_K_VALUES = [1, 10, 100]


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating a single .sio file."""

    path: str
    status: str          # "pass", "fail", "timeout", "error"
    exit_code: int
    stdout: str
    stderr: str
    duration_ms: float
    task_id: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PassAtKResult:
    """Aggregated pass@k results for a problem."""

    task_id: str
    n: int          # total samples
    c: int          # correct samples
    pass_at_k: dict[int, float]  # k -> pass@k score


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def eval_script(
    path: str,
    souc_path: str = DEFAULT_SOUC,
    timeout: int = DEFAULT_TIMEOUT,
    mode: str = "run",
) -> EvalResult:
    """Run a .sio file through souc and return the result.

    Args:
        path: Path to the .sio file.
        souc_path: Path to the souc binary.
        timeout: Maximum execution time in seconds.
        mode: "run" to execute, "check" to type-check only.

    Returns:
        EvalResult with status, output, and timing information.
    """
    if not os.path.isfile(path):
        return EvalResult(
            path=path,
            status="error",
            exit_code=-1,
            stdout="",
            stderr=f"File not found: {path}",
            duration_ms=0.0,
        )

    if not os.path.isfile(souc_path):
        return EvalResult(
            path=path,
            status="error",
            exit_code=-1,
            stdout="",
            stderr=f"souc binary not found: {souc_path}",
            duration_ms=0.0,
        )

    cmd = [souc_path, mode, path]

    # Set SOUNIO_STDLIB_PATH if not already set
    env = os.environ.copy()
    if "SOUNIO_STDLIB_PATH" not in env:
        # Try to find stdlib relative to souc or repo root
        repo_root = Path(path).resolve()
        # Walk up to find stdlib/
        for parent in [repo_root] + list(repo_root.parents):
            stdlib_path = parent / "stdlib"
            if stdlib_path.is_dir():
                env["SOUNIO_STDLIB_PATH"] = str(stdlib_path)
                break

    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        duration_ms = (time.monotonic() - start) * 1000.0

        if proc.returncode == 0:
            status = "pass"
        else:
            status = "fail"

        return EvalResult(
            path=path,
            status=status,
            exit_code=proc.returncode,
            stdout=proc.stdout[:10000],  # cap output size
            stderr=proc.stderr[:10000],
            duration_ms=duration_ms,
        )

    except subprocess.TimeoutExpired:
        duration_ms = (time.monotonic() - start) * 1000.0
        return EvalResult(
            path=path,
            status="timeout",
            exit_code=-1,
            stdout="",
            stderr=f"Timed out after {timeout}s",
            duration_ms=duration_ms,
        )

    except Exception as e:
        duration_ms = (time.monotonic() - start) * 1000.0
        return EvalResult(
            path=path,
            status="error",
            exit_code=-1,
            stdout="",
            stderr=str(e),
            duration_ms=duration_ms,
        )


def eval_directory(
    directory: str,
    souc_path: str = DEFAULT_SOUC,
    timeout: int = DEFAULT_TIMEOUT,
    verbose: bool = False,
) -> list[EvalResult]:
    """Evaluate all .sio files in a directory.

    Args:
        directory: Path to directory containing .sio files.
        souc_path: Path to the souc binary.
        timeout: Maximum execution time per file.
        verbose: Print progress to stderr.

    Returns:
        List of EvalResult for each file.
    """
    results = []
    sio_files = sorted(Path(directory).glob("*.sio"))

    if not sio_files:
        print(f"No .sio files found in {directory}", file=sys.stderr)
        return results

    total = len(sio_files)
    for i, sio_file in enumerate(sio_files, 1):
        if verbose:
            print(
                f"  [{i}/{total}] {sio_file.name}...",
                end="",
                file=sys.stderr,
                flush=True,
            )

        result = eval_script(str(sio_file), souc_path, timeout)

        # Try to extract task_id from file content
        try:
            with open(sio_file, "r") as f:
                for line in f:
                    if "HumanEval" in line:
                        # Extract task_id from comment
                        import re
                        m = re.search(r"(HumanEval/\d+)", line)
                        if m:
                            result.task_id = m.group(1)
                        break
        except Exception:
            pass

        if not result.task_id:
            result.task_id = sio_file.stem

        results.append(result)

        if verbose:
            symbol = {
                "pass": "PASS",
                "fail": "FAIL",
                "timeout": "T/O",
                "error": "ERR",
            }.get(result.status, "???")
            print(
                f" {symbol} ({result.duration_ms:.0f}ms)",
                file=sys.stderr,
            )

    return results


# ---------------------------------------------------------------------------
# pass@k computation
# ---------------------------------------------------------------------------

def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k metric.

    Unbiased estimator from the Codex paper (Chen et al., 2021):
      pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples, k = number of draws.

    Args:
        n: Total number of generated samples.
        c: Number of correct (passing) samples.
        k: Number of draws.

    Returns:
        pass@k score in [0, 1].
    """
    if n < k:
        return 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0

    # Use log-space to avoid overflow with large combinatorics
    # pass@k = 1 - prod_{i=0}^{k-1} (n-c-i) / (n-i)
    log_prod = 0.0
    for i in range(k):
        if n - i == 0:
            return 1.0
        log_prod += math.log(max(n - c - i, 0)) - math.log(n - i)

    return 1.0 - math.exp(log_prod)


def compute_pass_at_k(
    results: list[EvalResult],
    k_values: list[int],
) -> tuple[dict[int, float], list[PassAtKResult]]:
    """Compute pass@k for all problems.

    Groups results by task_id and computes pass@k for each group,
    then averages across all problems.

    Args:
        results: List of evaluation results.
        k_values: List of k values to compute.

    Returns:
        Tuple of (global_pass_at_k, per_problem_results).
    """
    # Group by task_id
    groups: dict[str, list[EvalResult]] = {}
    for r in results:
        groups.setdefault(r.task_id, []).append(r)

    per_problem = []
    for task_id, task_results in sorted(groups.items()):
        n = len(task_results)
        c = sum(1 for r in task_results if r.status == "pass")
        pak = {k: pass_at_k(n, c, k) for k in k_values}
        per_problem.append(PassAtKResult(
            task_id=task_id, n=n, c=c, pass_at_k=pak
        ))

    # Average across problems
    global_pak = {}
    if per_problem:
        for k in k_values:
            scores = [p.pass_at_k[k] for p in per_problem]
            global_pak[k] = sum(scores) / len(scores)
    else:
        global_pak = {k: 0.0 for k in k_values}

    return global_pak, per_problem


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_summary(
    results: list[EvalResult],
    global_pak: dict[int, float],
    per_problem: list[PassAtKResult],
) -> None:
    """Print a human-readable summary of evaluation results."""
    total = len(results)
    passed = sum(1 for r in results if r.status == "pass")
    failed = sum(1 for r in results if r.status == "fail")
    timed_out = sum(1 for r in results if r.status == "timeout")
    errors = sum(1 for r in results if r.status == "error")

    durations = [r.duration_ms for r in results if r.status != "error"]
    avg_ms = sum(durations) / len(durations) if durations else 0.0

    print("\n=== Sounio MultiPL-E Evaluation Results ===\n")
    print(f"  Total files:   {total}")
    print(f"  Passed:        {passed}")
    print(f"  Failed:        {failed}")
    print(f"  Timed out:     {timed_out}")
    print(f"  Errors:        {errors}")
    print(f"  Avg duration:  {avg_ms:.0f}ms")
    print()

    if global_pak:
        print("  pass@k:")
        for k, score in sorted(global_pak.items()):
            print(f"    pass@{k:3d} = {score:.4f}")
    print()

    # Per-problem detail (if multiple problems)
    if len(per_problem) > 1:
        print("  Per-problem results:")
        for p in per_problem:
            status_str = f"{p.c}/{p.n}"
            pak_str = ", ".join(
                f"pass@{k}={v:.3f}" for k, v in sorted(p.pass_at_k.items())
            )
            print(f"    {p.task_id:30s}  {status_str:8s}  {pak_str}")
        print()

    # Failed files detail
    failed_results = [r for r in results if r.status != "pass"]
    if failed_results:
        print("  Non-passing files:")
        for r in failed_results[:20]:
            print(f"    [{r.status:7s}] {r.path}")
            if r.stderr:
                first_err = r.stderr.strip().splitlines()[0][:100]
                print(f"             {first_err}")
        if len(failed_results) > 20:
            print(f"    ... and {len(failed_results) - 20} more")
        print()


def write_results_jsonl(results: list[EvalResult], output_path: str) -> None:
    """Write results to a JSONL file for further processing."""
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r.to_dict()) + "\n")
    print(f"  Results written to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MultiPL-E evaluation script for Sounio",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--file",
        type=str,
        help="Evaluate a single .sio file",
    )
    parser.add_argument(
        "--dir",
        type=str,
        help="Evaluate all .sio files in a directory",
    )
    parser.add_argument(
        "--souc",
        type=str,
        default=DEFAULT_SOUC,
        help=f"Path to souc binary (default: {DEFAULT_SOUC})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout per file in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="1,10,100",
        help="Comma-separated k values for pass@k (default: 1,10,100)",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Write results to JSONL file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress to stderr",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only type-check (souc check) instead of running",
    )

    args = parser.parse_args()

    k_values = [int(k.strip()) for k in args.k.split(",")]
    mode = "check" if args.check_only else "run"

    if args.file:
        result = eval_script(args.file, args.souc, args.timeout, mode)
        print(json.dumps(result.to_dict(), indent=2))
        sys.exit(0 if result.status == "pass" else 1)

    elif args.dir:
        results = eval_directory(
            args.dir, args.souc, args.timeout, verbose=args.verbose
        )

        if not results:
            print("No results.", file=sys.stderr)
            sys.exit(1)

        global_pak, per_problem = compute_pass_at_k(results, k_values)
        print_summary(results, global_pak, per_problem)

        if args.output:
            write_results_jsonl(results, args.output)

        # Exit with non-zero if any failures
        passed = sum(1 for r in results if r.status == "pass")
        sys.exit(0 if passed == len(results) else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
