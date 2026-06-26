#!/usr/bin/env python3
"""Run the medium Sounio SMT heuristic ablation with row timing.

This is a reproducible local benchmark harness, not public solver-competition
evidence. It compiles the Sounio emitter once, streams row output, records
inter-row wall-clock timing, and writes aggregation tables for review.
"""

from __future__ import annotations

import csv
import hashlib
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SOURCE = Path("benchmarks/solver/smt_ablation_medium.sio")
SCHEMA = "sounio.solver.smt_ablation_medium.run.v1"
DATA_SCHEMA = "sounio.solver.smt_ablation_medium.v1"


@dataclass
class Stat:
    rows: int = 0
    total: float = 0.0
    min_value: float | None = None
    max_value: float | None = None

    def add(self, value: float) -> None:
        self.rows += 1
        self.total += value
        self.min_value = value if self.min_value is None else min(self.min_value, value)
        self.max_value = value if self.max_value is None else max(self.max_value, value)

    def mean(self) -> float:
        if self.rows == 0:
            return 0.0
        return self.total / self.rows


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run_compile(souc_bin: str, binary: Path, compile_log: Path) -> int:
    started = time.monotonic_ns()
    proc = subprocess.run(
        [souc_bin, "compile", str(SOURCE), "-o", str(binary)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    elapsed_ms = (time.monotonic_ns() - started) // 1_000_000
    compile_log.write_text(proc.stdout + proc.stderr + f"\ncompile_elapsed_ms={elapsed_ms}\n", encoding="utf-8")
    return proc.returncode


def run_streamed(binary: Path, raw_csv: Path, timed_csv: Path, stderr_path: Path) -> tuple[int, int, int]:
    cmd = [str(binary)]
    stdbuf = shutil.which("stdbuf")
    if stdbuf is not None:
        cmd = [stdbuf, "-oL", str(binary)]

    started = time.monotonic_ns()
    previous_row_ns = started
    rows = 0
    header: list[str] | None = None
    raw_lines: list[str] = []
    timed_rows: list[dict[str, str]] = []

    proc = subprocess.Popen(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    for raw in proc.stdout:
        now = time.monotonic_ns()
        line = raw.rstrip("\n")
        raw_lines.append(line)
        if not line or line.startswith("#"):
            continue
        if header is None:
            header = line.split(",")
            continue
        values = line.split(",")
        if len(values) != len(header):
            continue
        rows += 1
        row = dict(zip(header, values))
        row["row_index"] = str(rows)
        row["elapsed_ms_since_prev_row"] = str((now - previous_row_ns) // 1_000_000)
        row["elapsed_ms_cumulative"] = str((now - started) // 1_000_000)
        timed_rows.append(row)
        previous_row_ns = now

    stderr = proc.stderr.read()
    returncode = proc.wait()
    raw_csv.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    stderr_path.write_text(stderr, encoding="utf-8")

    if header is None:
        timed_csv.write_text("", encoding="utf-8")
        return returncode, rows, (time.monotonic_ns() - started) // 1_000_000

    with timed_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = header + ["row_index", "elapsed_ms_since_prev_row", "elapsed_ms_cumulative"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(timed_rows)

    return returncode, rows, (time.monotonic_ns() - started) // 1_000_000


def emit_stats(timed_csv: Path, config_stats: Path, instance_stats: Path) -> None:
    grouped: dict[tuple[str, str, str, str], dict[str, Stat | set[str]]] = {}
    by_instance: dict[str, dict[str, Stat | set[str]]] = {}

    with timed_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (row["instance_id"], row["score_mode"], row["phase_mode"], row["restart_budget"])
            group = grouped.setdefault(
                key,
                {
                    "results": set(),
                    "decisions": Stat(),
                    "conflicts": Stat(),
                    "restarts": Stat(),
                    "elapsed": Stat(),
                },
            )
            inst = by_instance.setdefault(
                row["instance_id"],
                {
                    "results": set(),
                    "decisions": Stat(),
                    "conflicts": Stat(),
                    "restarts": Stat(),
                    "elapsed": Stat(),
                },
            )
            for target in (group, inst):
                results = target["results"]
                assert isinstance(results, set)
                results.add(row["result"])
                for name, column in [
                    ("decisions", "decisions"),
                    ("conflicts", "conflicts"),
                    ("restarts", "restarts"),
                    ("elapsed", "elapsed_ms_since_prev_row"),
                ]:
                    stat = target[name]
                    assert isinstance(stat, Stat)
                    stat.add(float(row[column]))

    with config_stats.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "instance_id",
            "score_mode",
            "phase_mode",
            "restart_budget",
            "rows",
            "result_values",
            "elapsed_ms_mean",
            "elapsed_ms_min",
            "elapsed_ms_max",
            "decisions_mean",
            "decisions_min",
            "decisions_max",
            "conflicts_mean",
            "conflicts_min",
            "conflicts_max",
            "restarts_mean",
            "restarts_min",
            "restarts_max",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for key in sorted(grouped, key=lambda k: tuple(int(v) for v in k)):
            group = grouped[key]
            elapsed = group["elapsed"]
            decisions = group["decisions"]
            conflicts = group["conflicts"]
            restarts = group["restarts"]
            results = group["results"]
            assert isinstance(elapsed, Stat)
            assert isinstance(decisions, Stat)
            assert isinstance(conflicts, Stat)
            assert isinstance(restarts, Stat)
            assert isinstance(results, set)
            writer.writerow(
                {
                    "instance_id": key[0],
                    "score_mode": key[1],
                    "phase_mode": key[2],
                    "restart_budget": key[3],
                    "rows": elapsed.rows,
                    "result_values": "|".join(sorted(results)),
                    "elapsed_ms_mean": f"{elapsed.mean():.6f}",
                    "elapsed_ms_min": int(elapsed.min_value or 0),
                    "elapsed_ms_max": int(elapsed.max_value or 0),
                    "decisions_mean": f"{decisions.mean():.6f}",
                    "decisions_min": int(decisions.min_value or 0),
                    "decisions_max": int(decisions.max_value or 0),
                    "conflicts_mean": f"{conflicts.mean():.6f}",
                    "conflicts_min": int(conflicts.min_value or 0),
                    "conflicts_max": int(conflicts.max_value or 0),
                    "restarts_mean": f"{restarts.mean():.6f}",
                    "restarts_min": int(restarts.min_value or 0),
                    "restarts_max": int(restarts.max_value or 0),
                }
            )

    with instance_stats.open("w", newline="", encoding="utf-8") as f:
        fields = ["instance_id", "rows", "result_values", "elapsed_ms_mean", "decisions_mean", "conflicts_mean", "restarts_mean"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for instance_id in sorted(by_instance, key=int):
            group = by_instance[instance_id]
            elapsed = group["elapsed"]
            decisions = group["decisions"]
            conflicts = group["conflicts"]
            restarts = group["restarts"]
            results = group["results"]
            assert isinstance(elapsed, Stat)
            assert isinstance(decisions, Stat)
            assert isinstance(conflicts, Stat)
            assert isinstance(restarts, Stat)
            assert isinstance(results, set)
            writer.writerow(
                {
                    "instance_id": instance_id,
                    "rows": elapsed.rows,
                    "result_values": "|".join(sorted(results)),
                    "elapsed_ms_mean": f"{elapsed.mean():.6f}",
                    "decisions_mean": f"{decisions.mean():.6f}",
                    "conflicts_mean": f"{conflicts.mean():.6f}",
                    "restarts_mean": f"{restarts.mean():.6f}",
                }
            )


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_ABLATION_MEDIUM_OUT_DIR", f"/tmp/sounio-solver-ablation-medium-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    souc_bin = os.environ.get("SOUC_BIN", "./bin/souc")
    binary = out_dir / "smt_ablation_medium"
    raw_csv = out_dir / "smt_ablation_medium.raw.csv"
    timed_csv = out_dir / "smt_ablation_medium.timed.csv"
    config_stats = out_dir / "config_stats.csv"
    instance_stats = out_dir / "instance_stats.csv"
    compile_log = out_dir / "compile.log"
    stderr_path = out_dir / "stderr.txt"
    manifest = out_dir / "manifest.txt"

    compile_status = run_compile(souc_bin, binary, compile_log)
    if compile_status != 0:
        print(compile_log.read_text(encoding="utf-8"), end="")
        return compile_status

    run_status, rows, elapsed_ms = run_streamed(binary, raw_csv, timed_csv, stderr_path)
    if run_status == 0:
        emit_stats(timed_csv, config_stats, instance_stats)

    madaros_hash = "unavailable"
    madaros = Path("artifacts/self-hosted/madaros")
    if madaros.exists():
        madaros_hash = sha256(madaros)

    with manifest.open("w", encoding="utf-8") as f:
        f.write(f"schema={SCHEMA}\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"source={SOURCE}\n")
        f.write(f"source_sha256={sha256(SOURCE)}\n")
        f.write(f"data_schema={DATA_SCHEMA}\n")
        f.write(f"souc_bin={souc_bin}\n")
        f.write(f"madaros_sha256={madaros_hash}\n")
        f.write(f"binary={binary}\n")
        f.write(f"raw_csv={raw_csv}\n")
        f.write(f"timed_csv={timed_csv}\n")
        f.write(f"config_stats_csv={config_stats}\n")
        f.write(f"instance_stats_csv={instance_stats}\n")
        f.write(f"compile_log={compile_log}\n")
        f.write(f"stderr={stderr_path}\n")
        f.write(f"compile_status={compile_status}\n")
        f.write(f"run_status={run_status}\n")
        f.write(f"elapsed_ms_total={elapsed_ms}\n")
        f.write(f"rows={rows}\n")
        f.write(f"instances=4\n")
        f.write(f"repetitions=4\n")
        f.write(f"random_seeds_per_random_family=12\n")
        f.write(f"configs=7\n")
        f.write("note=Medium local ablation with inter-row wall-clock timing. This is stronger readiness evidence than the tiny smoke, but not SATComp/SMT-COMP-scale novelty evidence.\n")

    print(manifest.read_text(encoding="utf-8"), end="")
    return run_status


if __name__ == "__main__":
    sys.exit(main())
