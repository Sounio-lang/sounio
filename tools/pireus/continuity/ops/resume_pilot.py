#!/usr/bin/env python3
"""Recover a finished transport stage from Slurm accounting; never resubmit it."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cycle import atomic, digest, encoded, event, verify
from pilot import paired_completion, ROOT, HERE

HELPER = Path(__file__).resolve()
POD = "slurm-pilot-controller-0"

def completed_accounting(raw, job):
    rows = [line.split("|") for line in raw.splitlines() if line.strip()]
    selected = [row for row in rows if row[0] == job]
    if len(selected) != 1 or len(selected[0]) != 6:
        raise ValueError("accounting lacks one exact job record")
    row = selected[0]
    if row[1:3] != ["COMPLETED", "0:0"]:
        raise ValueError("accounting does not confirm successful job completion")
    if any(not value or value in {"Unknown", "None", "(null)"} for value in row[3:]):
        raise ValueError("accounting lacks node/start/end identity")
    if set(row[3].split(",")) != {"gpuorangefs-multi-spark-3c59", "gpuorangefs-multi-spark-8e54"}:
        raise ValueError("accounting job belongs to a different node pair")
    start, end = (datetime.datetime.fromisoformat(v) for v in row[4:6])
    if end < start:
        raise ValueError("accounting end precedes start")
    return row

def query_accounting(job):
    uid_command = ["kubectl", "-n", "slurm-pilot", "get", "pod", POD,
                   "-o", "jsonpath={.metadata.uid}"]
    before = subprocess.check_output(uid_command, text=True, timeout=30).strip()
    command = ["kubectl", "-n", "slurm-pilot", "exec", POD, "-c", "slurmctld",
               "--", "sacct", "-j", job, "--format=JobIDRaw,State,ExitCode,NodeList,Start,End",
               "-n", "-P", "--noconvert"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=240)
    after = subprocess.check_output(uid_command, text=True, timeout=30).strip()
    if result.returncode or not before or before != after:
        raise ValueError("accounting query failed or controller identity changed")
    completed_accounting(result.stdout, job)
    return dict(schema=1, source="Slurm accounting via controller sacct",
                controller_pod=POD, controller_uid=before, command=command,
                stdout=result.stdout, stderr=result.stderr, exit_code=result.returncode,
                observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                helper_sha256=digest(HELPER.read_bytes()))

def recover_stage(folder):
    root = folder.parents[1]
    verify(root)
    intent = json.loads((folder / "intent.json").read_text())
    command = intent["command"]
    if (len(command) != 7 or command[1] != str(HERE / "runtime/launch_pair.py")
            or command[2] not in {"tokenize", "offline-generate"}
            or command[3] != "--minutes" or command[5] != "--input-bundle"):
        raise ValueError("unrecognized frozen stage intent")
    bundle = Path(command[6])
    if bundle.parent != root or digest(bundle.read_bytes()) != intent["bundle_sha256"]:
        raise ValueError("frozen stage input changed")
    log = folder / "launch.log"
    raw = log.read_bytes()
    expected = "OFFLINE_CYCLE_COMPLETE" if command[2] == "offline-generate" else "TOKENIZER_TRANSPORT_PASS"
    job, nodes = paired_completion(raw.decode(), expected)
    receipt_path = folder / "accounting-recovery.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
    else:
        receipt = query_accounting(job)
        atomic(receipt_path, encoded(receipt))
    if (receipt["helper_sha256"] != digest(HELPER.read_bytes())
            or receipt["exit_code"] != 0
            or receipt["source"] != "Slurm accounting via controller sacct"):
        raise ValueError("recovery evidence identity changed")
    event(root, "accounting-recovery-" + folder.name, receipt_path)
    row = completed_accounting(receipt["stdout"], job)
    if log.read_bytes() != raw:
        raise ValueError("launch log changed during accounting recovery")
    result = dict(job=job, nodes=nodes, log_sha256=digest(raw),
                  input_sha256=intent["bundle_sha256"],
                  slurm_state=f"JobId={job} JobState={row[1]} ExitCode={row[2]} StateSource=accounting",
                  accounting_recovery_sha256=digest(receipt_path.read_bytes()),
                  recovery_helper_sha256=digest(HELPER.read_bytes()))
    atomic(folder / "completed.json", encoded(result))
    event(root, "accounting-completion-" + folder.name, folder / "completed.json")
    print(json.dumps(dict(stage="PILOT_ACCOUNTING_RECOVERY", folder=str(folder),
                          job=job, evidence_sha256=result["accounting_recovery_sha256"])), flush=True)

def pending(root):
    return [p.parent for p in sorted(root.glob("round-*/pilot-stages/*/intent.json"))
            if not (p.parent / "completed.json").exists()]

def run(root):
    if not os.environ.get("TMUX"):
        raise ValueError("recovery and pilot must run in remote tmux")
    # Original manifests, source dependencies and CI acceptance remain unchanged.
    # Each recovery adds an independently hashed accounting receipt.
    for _ in range(19):
        stages = pending(root)
        if len(stages) > 1:
            raise ValueError("multiple unfinished launches require individual investigation")
        for folder in stages:
            recover_stage(folder)
        result = subprocess.run([sys.executable, str(HERE / "pilot.py"),
                                 "run", "--run", str(root)], cwd=ROOT)
        if result.returncode == 0:
            return
        if not pending(root):
            raise RuntimeError("pilot failed outside recoverable stage completion")
    raise RuntimeError("bounded stage recovery limit reached")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    with (root / "transport-recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(root)
