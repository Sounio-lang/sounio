#!/usr/bin/env python3
"""Record external solver/tool availability for the real-corpus solver pilot.

The script is intentionally user-space only. It never invokes apt, sudo, or
system package managers; it records what is already on PATH plus any executable
tool found in SOUNIO_SOLVER_TOOL_ROOT.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


SAT_SOLVERS = ("z3", "cadical", "kissat", "minisat", "glucose", "cryptominisat5")
SMT_SOLVERS = ("z3", "cvc5", "yices-smt2")
OPB_SOLVERS = ("roundingsat", "open-wbo", "open-wbo_static", "scip", "sat4j-pb", "minisat+", "minisatp", "pbsolver")


@dataclass(frozen=True)
class Tool:
    domain: str
    solver: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def solver_version(path: str | None) -> str:
    if not path:
        return ""
    probes = ([path, "--version"], [path, "-version"], [path, "-h"], [path, "--help"])
    for probe in probes:
        try:
            completed = subprocess.run(probe, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def find_tool(name: str, tool_root: Path) -> tuple[str, str]:
    path = shutil.which(name)
    if path:
        return path, "path"
    for candidate in (tool_root / "bin" / name, tool_root / name):
        if candidate.exists() and os.access(candidate, os.X_OK):
            return str(candidate), "user-space-cache"
    return "", "missing"


def rows(tool_root: Path) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for domain, names in (("sat", SAT_SOLVERS), ("smt", SMT_SOLVERS), ("opb", OPB_SOLVERS)):
        for solver in names:
            key = (domain, solver)
            if key in seen:
                continue
            seen.add(key)
            path, source = find_tool(solver, tool_root)
            digest = ""
            if path:
                try:
                    digest = sha256_file(Path(path))
                except OSError:
                    digest = "sha256_unavailable"
            out.append(
                {
                    "domain": domain,
                    "solver": solver,
                    "path": path,
                    "available": "1" if path else "0",
                    "version": solver_version(path),
                    "source": source,
                    "sha256_or_commit": digest,
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=None, help="Output directory for solver_availability.csv")
    parser.add_argument("--tool-root", type=Path, default=None, help="User-space cache root")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tool_root = args.tool_root or Path(os.environ.get("SOUNIO_SOLVER_TOOL_ROOT", "~/.cache/sounio/solver-tools")).expanduser()
    out_dir = args.out_dir or Path(f"/tmp/sounio-solver-tools-{stamp}")
    tool_root.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "solver_availability.csv"
    fieldnames = ["domain", "solver", "path", "available", "version", "source", "sha256_or_commit"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows(tool_root))

    manifest = out_dir / "manifest.txt"
    manifest.write_text(
        "\n".join(
            [
                "schema=sounio.solver.external_tools.bootstrap.v1",
                f"timestamp_utc={stamp}",
                f"tool_root={tool_root}",
                f"solver_availability={csv_path}",
                "note=user-space discovery only; no apt/root/system install attempted.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(manifest.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
