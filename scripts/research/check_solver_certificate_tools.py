#!/usr/bin/env python3
"""Record availability of external proof/certificate tools for solver claims."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


TOOL_GROUPS = [
    ("sat_lrat_frat", "drat-trim"),
    ("sat_lrat_frat", "gratgen"),
    ("sat_lrat_frat", "lrat-check"),
    ("sat_lrat_frat", "frat-rs"),
    ("smt_alethe", "carcara"),
    ("smt_alethe", "cvc5"),
    ("smt_alethe", "veriT"),
    ("pb_veripb", "veripb"),
    ("pb_veripb", "cakepb"),
    ("pb_veripb", "roundingsat"),
]


def version(path: str | None) -> str:
    if not path:
        return ""
    for args in ([path, "--version"], [path, "-version"], [path, "-h"]):
        try:
            completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            continue
        lines = (completed.stdout + completed.stderr).strip().splitlines()
        if lines:
            return lines[0][:200]
    return "version_unavailable"


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(os.environ.get("SOLVER_CERT_TOOLS_OUT_DIR", f"/tmp/sounio-solver-cert-tools-{stamp}"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for group, tool in TOOL_GROUPS:
        path = shutil.which(tool)
        rows.append(
            {
                "group": group,
                "tool": tool,
                "path": path or "",
                "available": 1 if path else 0,
                "version": version(path),
            }
        )

    csv_path = out_dir / "certificate_tool_availability.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "tool", "path", "available", "version"])
        writer.writeheader()
        writer.writerows(rows)

    by_group = {}
    for row in rows:
        by_group.setdefault(row["group"], 0)
        by_group[row["group"]] += int(row["available"])

    manifest_path = out_dir / "manifest.txt"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write("schema=sounio.solver.certificate_tools.run.v1\n")
        f.write(f"timestamp_utc={stamp}\n")
        f.write(f"repo={repo}\n")
        f.write(f"availability_csv={csv_path}\n")
        for group in sorted(by_group):
            f.write(f"{group}_available_count={by_group[group]}\n")
        f.write("note=Availability only. Missing tools block Level 3 certificate claims; this does not validate any proof format.\n")

    print(manifest_path.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
