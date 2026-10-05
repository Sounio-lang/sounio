#!/usr/bin/env python3
"""Immutable diagnostic staging and one-shot execution; never resumes the pilot."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_decode_probe import prepare

ROOT = Path(__file__).resolve().parents[4]
RUNTIME = ROOT / "tools/pireus/continuity/runtime"
PAIR = "gpuorangefs-multi-spark-3c59,gpuorangefs-multi-spark-8e54"


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    with path.open("xb") as f:
        f.write(data)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def stage(attempt, bundle):
    # Refuse foreign/changed input by binding it to the preserved11956 receipt.
    receipt = json.loads((ROOT / "tools/pireus/continuity/validation/first-request-comparison-11939-11956.json").read_text())
    raw = bundle.read_bytes()
    if digest(raw) != receipt["sources"]["pilot_bundle"]["sha256"]:
        raise ValueError("input is not the frozen11956 pilot bundle")
    source = (RUNTIME / "offline_generate.py").read_bytes()
    probe = prepare(source)
    expected = json.loads((ROOT / "tools/pireus/continuity/validation/decode-probe-build-11956.json").read_text())
    if digest(probe) != expected["output_sha256"]:
        raise ValueError("probe differs from qualified build")
    files = subprocess.check_output(["git", "ls-files", "-z", str(RUNTIME)],
                                    cwd=ROOT).decode().split("\0")
    snapshot = {}
    for name in filter(None, files):
        path = ROOT / name
        if path.parent == RUNTIME and path.is_file():
            snapshot[path.name] = path.read_bytes()
    snapshot["offline_generate.py"] = probe
    attempt.mkdir()  # Existing attempts, successful or failed, cannot be replaced.
    (attempt / "runtime").mkdir()
    write(attempt / "input.json", raw)
    manifest = dict(schema="pireus-decode-diagnostic-launch-v1", diagnostic_only=True,
                    parent_failed_job=11956, source_commit=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    input_sha256=digest(raw), input_items=32,
                    native_floor_gib=32, early_stop_gib=33, minutes=120,
                    runtime_files={k: digest(v) for k, v in snapshot.items()},
                    probe_sha256=digest(probe), original_runtime_sha256=digest(source),
                    pilot_acceptance=False, performance_evidence=False,
                    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for name, data in snapshot.items():
        write(attempt / "runtime" / name, data)
    write(attempt / "manifest.json", encoded(manifest))
    return manifest


def execute(attempt):
    if not os.environ.get("TMUX"):
        raise ValueError("diagnostic execution requires remote tmux")
    manifest = json.loads((attempt / "manifest.json").read_text())
    if digest((attempt / "input.json").read_bytes()) != manifest["input_sha256"]:
        raise ValueError("diagnostic input changed")
    for name, sha in manifest["runtime_files"].items():
        if Path(name).name != name or digest((attempt / "runtime" / name).read_bytes()) != sha:
            raise ValueError("diagnostic runtime changed")
    # The original launcher stages shared runtime paths before requesting its
    # exclusive allocation, so refuse staging while any pair job is present.
    queue = subprocess.check_output(["squeue", "-h", "-w", PAIR, "-o", "%i"], text=True)
    if queue.strip():
        raise ValueError("Spark pair has active or pending allocations: " + queue.strip())
    command = [sys.executable, str(attempt / "runtime/launch_pair.py"),
               "offline-generate", "--minutes", "120", "--input-bundle", str(attempt / "input.json")]
    write(attempt / "intent.json", encoded(dict(command=command,
          manifest_sha256=digest((attempt / "manifest.json").read_bytes()),
          diagnostic_only=True, automatic_retry=False)))
    with (attempt / "launch.log").open("xb") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    write(attempt / "launcher-exit.json", encoded(dict(returncode=result.returncode,
          log_sha256=digest((attempt / "launch.log").read_bytes()),
          diagnostic_only=True, pilot_acceptance=False,
          ended_at=datetime.datetime.now(datetime.timezone.utc).isoformat())))
    return result.returncode


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["stage", "execute"])
    p.add_argument("--attempt", type=Path, required=True)
    p.add_argument("--bundle", type=Path)
    args = p.parse_args()
    if not args.attempt.is_absolute():
        p.error("attempt must be absolute")
    if args.action == "stage":
        if not args.bundle:
            p.error("stage requires bundle")
        print(json.dumps(stage(args.attempt, args.bundle), indent=2))
    else:
        raise SystemExit(execute(args.attempt))
