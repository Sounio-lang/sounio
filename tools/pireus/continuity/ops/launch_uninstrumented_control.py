#!/usr/bin/env python3
"""One uninstrumented control; preserve the qualified diagnostic and failed pilot."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess

from launch_decode_probe import ROOT, digest, encoded, execute, write
from prepare_decode_probe import prepare


def stage(parent, attempt):
    qualified = ROOT / "tools/pireus/continuity/validation/decode-complete-11957"
    q = json.loads((qualified / "qualification.json").read_text())
    old_raw = (parent / "manifest.json").read_bytes()
    if old_raw != (qualified / "manifest.json").read_bytes():
        raise ValueError("parent differs from qualified diagnostic")
    old = json.loads(old_raw)
    if q["evidence"]["job"] != 11957 or q["evidence"]["paired_responses"] != 32:
        raise ValueError("parent lacks32-request diagnostic qualification")
    raw_input = (parent / "input.json").read_bytes()
    if digest(raw_input) != old["input_sha256"]:
        raise ValueError("parent frozen inputs changed")
    files = {}
    for name, sha in old["runtime_files"].items():
        if Path(name).name != name:
            raise ValueError("unsafe runtime filename")
        raw = (parent / "runtime" / name).read_bytes()
        if digest(raw) != sha:
            raise ValueError("parent runtime changed")
        files[name] = raw
    source = subprocess.check_output(["git", "show",
        "74789c0fff522c211f741992f4b05ef2b1fb3346:tools/pireus/continuity/runtime/offline_generate.py"],
        cwd=ROOT)
    if digest(source) != old["original_runtime_sha256"]:
        raise ValueError("original runtime hash differs")
    if prepare(source) != files["offline_generate.py"]:
        raise ValueError("probe removal is not the qualified reversible transformation")
    files["offline_generate.py"] = source
    changed = [name for name, data in files.items() if digest(data) != old["runtime_files"][name]]
    if changed != ["offline_generate.py"]:
        raise ValueError("control changes more than probe removal")
    manifest = dict(schema="pireus-uninstrumented-control-v1", diagnostic_only=True,
        experiment_kind="uninstrumented-control", instrumentation=False,
        parent_diagnostic_job=11957, parent_failed_job=11956,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        parent_manifest_sha256=digest(old_raw), input_sha256=digest(raw_input),
        input_items=32, native_floor_gib=32, early_stop_gib=33, minutes=120,
        runtime_files={name: digest(data) for name, data in files.items()},
        runtime_sha256=digest(source), parent_probe_sha256=old["probe_sha256"],
        changed_runtime_files=changed, automatic_retry=False, pilot_acceptance=False,
        performance_evidence=False, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    attempt.mkdir()
    (attempt / "runtime").mkdir()
    write(attempt / "input.json", raw_input)
    for name, data in files.items():
        write(attempt / "runtime" / name, data)
    write(attempt / "manifest.json", encoded(manifest))
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["stage", "execute"])
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--parent", type=Path)
    a = p.parse_args()
    if not a.attempt.is_absolute():
        p.error("attempt must be absolute")
    if a.action == "stage":
        if a.parent is None:
            p.error("parent is required")
        print(json.dumps(stage(a.parent, a.attempt), indent=2))
    else:
        raise SystemExit(execute(a.attempt))
