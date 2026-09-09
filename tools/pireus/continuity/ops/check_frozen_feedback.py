#!/usr/bin/env python3
"""Readiness gate only: no allocation, generation, retry, or CI mutation."""
import argparse
import json
from pathlib import Path
import subprocess
from feedback_smoke import HERE, digest
from build_control_feedback import require

FREEZE_SHA = "d1d45b4c0a2a846b6f1e62eccacb15205988c3a9deb1fc35135b8d6ea7dd433d"


def inputs(root):
    raw = (root / "execution-freeze.json").read_bytes()
    require(digest(raw) == FREEZE_SHA, "freeze identity mismatch")
    spec = json.loads(raw)
    for table, prefix in (("runtime_sha256", "runtime/"), ("files_sha256", "")):
        for name, expected in spec[table].items():
            path = root / (prefix + name)
            require(path.resolve().is_relative_to(root.resolve()), "artifact escapes attempt")
            require(digest(path.read_bytes()) == expected, "frozen artifact mismatch: " + prefix + name)
    return spec


def source_checks(spec, checks):
    selected = {}
    for name in spec["required_source_checks"]:
        rows = [r for r in checks if r["name"] == name and r["head_sha"] == spec["source_commit"]]
        require(bool(rows), "missing exact-source check: " + name)
        row = max(rows, key=lambda r: (r.get("started_at") or "", r["id"]))
        require(row["status"] == "completed" and row["conclusion"] == "success",
                "latest exact-source check not green: " + name)
        selected[name] = row
    return selected


def readiness(root):
    spec = inputs(root)
    raw = subprocess.check_output(
        ["gh", "api", "--paginate",
         "repos/Sounio-lang/sounio/commits/" + spec["source_commit"] + "/check-runs",
         "--jq", ".check_runs[] | @json"], cwd=HERE, text=True, timeout=60)
    checks = [json.loads(line) for line in raw.splitlines() if line.strip()]
    selected = source_checks(spec, checks)
    return dict(schema="pireus-feedback-readiness-v1", source_commit=spec["source_commit"],
                freeze_sha256=FREEZE_SHA, runtime_files=len(spec["runtime_sha256"]),
                input_artifacts=len(spec["files_sha256"]), source_checks=selected,
                source_and_inputs_ready=True, live_host_preflight_required=True,
                runtime_receipts_required=True, inference_completed=False,
                pilot_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = readiness(args.root)
    with args.output.open("x") as out:
        out.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
