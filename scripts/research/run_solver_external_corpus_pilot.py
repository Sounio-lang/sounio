#!/usr/bin/env python3
"""Run the Sounio real external-corpus solver pilot."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
SAT_SOLVERS = ("z3", "cadical", "kissat")
SMT_SOLVERS = ("z3", "cvc5", "yices-smt2")
OPB_SOLVERS = ("roundingsat", "open-wbo", "open-wbo_static", "scip", "sat4j-pb", "minisat+", "minisatp", "pbsolver")


def run(cmd: list[str], timeout: int | None = None, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, env=env)


def parse_result(stdout: str, stderr: str) -> str:
    text = (stdout + "\n" + stderr).upper()
    if "UNSATISFIABLE" in text or "\nUNSAT\n" in text or text.strip() == "UNSAT":
        return "UNSAT"
    if "SATISFIABLE" in text or "\nSAT\n" in text or text.strip() == "SAT":
        return "SAT"
    if "UNKNOWN" in text:
        return "UNKNOWN"
    return "UNPARSED"


def sounio_label(exit_code: int, stdout: str) -> str:
    if exit_code == 0:
        lines = stdout.splitlines()
        for idx, line in enumerate(lines):
            if line.startswith("SOUNIO_DIMACS_RESULT "):
                if len(line.split()) > 2:
                    value = line.split()[-1]
                elif idx + 1 < len(lines):
                    value = lines[idx + 1].strip()
                else:
                    value = ""
                if value == "1":
                    return "SAT"
                if value == "0":
                    return "UNSAT"
                if value == "-1":
                    return "LIA_ERROR"
                if value == "-2":
                    return "BUDGET_EXCEEDED"
                return f"OTHER_{value}"
        return "EXIT0_UNPARSED"
    return f"EXIT_{exit_code}"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def solver_availability(out_dir: Path) -> Path:
    completed = run(["python3", "scripts/research/bootstrap_solver_external_tools.py", "--out-dir", str(out_dir)])
    (out_dir / "bootstrap_stdout.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    return out_dir / "solver_availability.csv"


def fetch_corpus(args: argparse.Namespace, out_dir: Path) -> Path:
    corpus_dir = out_dir / "corpus"
    cmd = [
        "python3",
        "scripts/research/fetch_solver_external_corpus_pilot.py",
        "--sat-count",
        str(args.sat_count),
        "--smt-count",
        str(args.smt_count),
        "--opb-count",
        str(args.opb_count),
        "--out-dir",
        str(corpus_dir),
        "--max-sat-candidates",
        str(args.max_sat_candidates),
    ]
    completed = run(cmd, timeout=args.fetch_timeout_sec)
    (out_dir / "fetch_stdout.txt").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(completed.stdout + completed.stderr)
    return corpus_dir / "corpus_manifest.csv"


def solver_command(solver: str, path: str, instance: Path) -> list[str]:
    if solver == "cvc5":
        return [path, "--lang", "smt2", str(instance)]
    if solver == "scip":
        return [path, "-c", f"read {instance}", "-c", "optimize", "-c", "quit"]
    return [path, str(instance)]


def run_external(corpus_rows: list[dict[str, str]], availability_rows: list[dict[str, str]], out_dir: Path, timeout_sec: int) -> Path:
    solvers_by_domain: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in availability_rows:
        if row["available"] == "1":
            solvers_by_domain[row["domain"]].append(row)
    rows: list[dict[str, str]] = []
    for inst in corpus_rows:
        if inst["status"] != "OK":
            continue
        domain = inst["domain"]
        if domain == "sat":
            allowed = set(SAT_SOLVERS)
        elif domain == "smt":
            allowed = set(SMT_SOLVERS)
        elif domain == "opb":
            allowed = set(OPB_SOLVERS)
        else:
            continue
        for solver in solvers_by_domain.get(domain, []):
            if solver["solver"] not in allowed:
                continue
            start = time.monotonic_ns()
            try:
                completed = run(solver_command(solver["solver"], solver["path"], Path(inst["local_path"])), timeout=timeout_sec)
                exit_code = completed.returncode
                result = parse_result(completed.stdout, completed.stderr)
            except subprocess.TimeoutExpired:
                exit_code = 124
                result = "TIMEOUT"
            elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
            rows.append(
                {
                    "domain": domain,
                    "instance_id": inst["instance_id"],
                    "expected": inst["expected"],
                    "solver": solver["solver"],
                    "solver_path": solver["path"],
                    "exit_code": str(exit_code),
                    "result": result,
                    "elapsed_ms": str(elapsed_ms),
                    "instance_sha256": inst["sha256"],
                    "local_path": inst["local_path"],
                }
            )
    path = out_dir / "external_results.csv"
    write_csv(path, rows, ["domain", "instance_id", "expected", "solver", "solver_path", "exit_code", "result", "elapsed_ms", "instance_sha256", "local_path"])
    return path


def run_sounio(corpus_manifest: Path, out_dir: Path, timeout_sec: int) -> Path:
    harness_dir = out_dir / "sounio_harness"
    generated = run(["python3", "scripts/research/generate_sounio_dimacs_harness.py", "--manifest", str(corpus_manifest), "--out-dir", str(harness_dir)])
    (out_dir / "generate_harness_stdout.txt").write_text(generated.stdout + generated.stderr, encoding="utf-8")
    rows: list[dict[str, str]] = []
    manifest = harness_dir / "sounio_dimacs_harness_manifest.csv"
    if not manifest.exists():
        path = out_dir / "sounio_dimacs_results.csv"
        write_csv(path, rows, ["instance_id", "expected", "compile_exit", "run_exit", "result", "elapsed_ms", "sio_path", "binary_path"])
        return path
    for row in read_csv(manifest):
        if row["status"] != "OK":
            continue
        sio = Path(row["sio_path"])
        binary = out_dir / "sounio_bin" / sio.with_suffix("").name
        binary.parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic_ns()
        compile_run = run(["./bin/souc", "compile", str(sio), "-o", str(binary)], timeout=timeout_sec)
        if compile_run.returncode == 0:
            try:
                exec_run = subprocess.run([str(binary)], cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_sec)
                run_exit = exec_run.returncode
                result = sounio_label(run_exit, exec_run.stdout)
                stdout = exec_run.stdout
                stderr = exec_run.stderr
            except subprocess.TimeoutExpired:
                run_exit = 124
                result = "TIMEOUT"
                stdout = ""
                stderr = "timeout"
        else:
            run_exit = -1
            result = "COMPILE_FAILED"
            stdout = compile_run.stdout
            stderr = compile_run.stderr
        elapsed_ms = (time.monotonic_ns() - start) // 1_000_000
        stem = sio.with_suffix("").name
        (out_dir / f"{stem}.sounio.log").write_text(stdout + stderr + compile_run.stdout + compile_run.stderr, encoding="utf-8")
        rows.append(
            {
                "instance_id": row["instance_id"],
                "expected": row["expected"],
                "compile_exit": str(compile_run.returncode),
                "run_exit": str(run_exit),
                "result": result,
                "elapsed_ms": str(elapsed_ms),
                "sio_path": str(sio),
                "binary_path": str(binary),
            }
        )
    path = out_dir / "sounio_dimacs_results.csv"
    write_csv(path, rows, ["instance_id", "expected", "compile_exit", "run_exit", "result", "elapsed_ms", "sio_path", "binary_path"])
    return path


def crosscheck(external_path: Path, sounio_path: Path, out_dir: Path) -> Path:
    sounio = {row["instance_id"]: row for row in read_csv(sounio_path)}
    rows: list[dict[str, str]] = []
    for ext in read_csv(external_path):
        if ext["domain"] != "sat":
            continue
        srow = sounio.get(ext["instance_id"])
        if not srow:
            continue
        definitive = ext["result"] in {"SAT", "UNSAT"} and srow["result"] in {"SAT", "UNSAT"}
        rows.append(
            {
                "instance_id": ext["instance_id"],
                "solver": ext["solver"],
                "external_result": ext["result"],
                "sounio_result": srow["result"],
                "matches": "1" if definitive and ext["result"] == srow["result"] else "0",
                "definitive": "1" if definitive else "0",
            }
        )
    path = out_dir / "crosscheck_results.csv"
    write_csv(path, rows, ["instance_id", "solver", "external_result", "sounio_result", "matches", "definitive"])
    return path


def summarize(corpus: list[dict[str, str]], availability: list[dict[str, str]], external: list[dict[str, str]], sounio: list[dict[str, str]], cross: list[dict[str, str]], out_dir: Path) -> tuple[Path, Counter[str]]:
    summary: Counter[str] = Counter()
    for row in corpus:
        summary[f"corpus_{row['domain']}_{row['status']}"] += 1
        if row["expected"] in {"SAT", "UNSAT", "UNKNOWN", "BLOCKED"}:
            summary[f"corpus_expected_{row['expected']}"] += 1
    for row in availability:
        summary[f"solver_{row['domain']}_{row['solver']}_available_{row['available']}"] += 1
    for row in external:
        summary[f"external_{row['domain']}_{row['solver']}_{row['result']}"] += 1
    for row in sounio:
        summary[f"sounio_{row['result']}"] += 1
    for row in cross:
        if row["definitive"] == "1":
            summary["crosscheck_definitive"] += 1
            summary[f"crosscheck_match_{row['matches']}"] += 1
    sat_extra_solver = any(row["domain"] == "sat" and row["available"] == "1" and row["solver"] in {"cadical", "kissat"} for row in availability)
    sat_z3 = any(row["domain"] == "sat" and row["available"] == "1" and row["solver"] == "z3" for row in availability)
    sat_supported = sum(1 for row in corpus if row["domain"] == "sat" and row["status"] == "OK")
    mismatches = summary["crosscheck_match_0"]
    summary["pilot_sat_acceptance_candidate"] = 1 if sat_supported >= 8 and sat_z3 and sat_extra_solver and mismatches == 0 else 0
    summary["pilot_evidence_incomplete"] = 0 if summary["pilot_sat_acceptance_candidate"] else 1
    rows = [{"metric": key, "value": str(summary[key])} for key in sorted(summary)]
    path = out_dir / "summary.csv"
    write_csv(path, rows, ["metric", "value"])
    return path, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sat-count", type=int, default=12)
    parser.add_argument("--smt-count", type=int, default=12)
    parser.add_argument("--opb-count", type=int, default=12)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument("--fetch-timeout-sec", type=int, default=600)
    parser.add_argument("--max-sat-candidates", type=int, default=140)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or Path(os.environ.get("SOLVER_EXTERNAL_PILOT_OUT_DIR", f"/tmp/sounio-solver-external-pilot-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        availability_path = solver_availability(out_dir)
        corpus_manifest = fetch_corpus(args, out_dir)
        root_corpus_manifest = out_dir / "corpus_manifest.csv"
        shutil.copyfile(corpus_manifest, root_corpus_manifest)
        corpus_manifest = root_corpus_manifest
        corpus_rows = read_csv(corpus_manifest)
        availability_rows = read_csv(availability_path)
        external_path = run_external(corpus_rows, availability_rows, out_dir, args.timeout_sec)
        sounio_path = run_sounio(corpus_manifest, out_dir, args.timeout_sec)
        cross_path = crosscheck(external_path, sounio_path, out_dir)
        summary_path, summary = summarize(corpus_rows, availability_rows, read_csv(external_path), read_csv(sounio_path), read_csv(cross_path), out_dir)
    except Exception as exc:
        (out_dir / "fatal_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"fatal_error={exc}", file=sys.stderr)
        return 2

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "schema=sounio.solver.external_corpus_pilot.run.v1",
                f"timestamp_utc={stamp}",
                f"repo={REPO}",
                f"corpus_manifest={corpus_manifest}",
                f"solver_availability={availability_path}",
                f"external_results={external_path}",
                f"sounio_dimacs_results={sounio_path}",
                f"crosscheck_results={cross_path}",
                f"summary={summary_path}",
                f"pilot_sat_acceptance_candidate={summary['pilot_sat_acceptance_candidate']}",
                f"pilot_evidence_incomplete={summary['pilot_evidence_incomplete']}",
                "note=Pilot real external-corpus evidence only; not Level 3 completion, not SOTA, not certificate-independent public evidence.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0 if summary["pilot_sat_acceptance_candidate"] else 1


if __name__ == "__main__":
    sys.exit(main())
