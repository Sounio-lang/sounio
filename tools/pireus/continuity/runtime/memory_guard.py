#!/usr/bin/env python3
"""Stop only this Slurm rank's process group before the host reserve is exhausted."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

RESERVE_BYTES = 36 * 1024**3
INTERVAL_SECONDS = 0.05

def available():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable is missing")

def supervise(command, read_memory=available, reserve=RESERVE_BYTES):
    before = read_memory()
    minimum = before
    job = os.environ.get("SLURM_JOB_ID")
    rank = os.environ.get("PIREUS_RANK")
    def emit(stage, **fields):
        print(json.dumps(dict(stage=stage, job=job, rank=rank, **fields)), flush=True)
    emit("MEMORY_GUARD_START", available_bytes=before, reserve_bytes=reserve)
    if before < reserve:
        emit("MEMORY_GUARD_REFUSED", available_bytes=before)
        return 75
    child = subprocess.Popen(command, start_new_session=True)
    stopped = False
    previous = {}
    def stop_group():
        nonlocal stopped
        if not stopped:
            stopped = True
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    def interrupted(signum, frame):
        raise InterruptedError("Rank guardian interrupted")
    try:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            previous[sig] = signal.signal(sig, interrupted)
        next_report = time.monotonic()
        while child.poll() is None:
            current = read_memory()
            minimum = min(minimum, current)
            if current < reserve:
                stop_group()
                child.wait(timeout=10)
                emit("MEMORY_GUARD_STOP", available_bytes=current, minimum_bytes=minimum,
                     protected_floor_bytes=32 * 1024**3)
                return 75
            now = time.monotonic()
            if now >= next_report:
                emit("MEMORY_GUARD_SAMPLE", available_bytes=current, minimum_bytes=minimum)
                next_report = now + 5
            time.sleep(INTERVAL_SECONDS)
        emit("MEMORY_GUARD_CHILD_EXIT", returncode=child.returncode, minimum_bytes=minimum)
        return child.returncode if child.returncode >= 0 else 128 - child.returncode
    finally:
        stop_group()
        child.wait(timeout=10)
        for sig, handler in previous.items():
            signal.signal(sig, handler)

def main():
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_PROCID"):
        # Rank zero is the nonempty string "0".
        raise SystemExit("An owned Slurm allocation/rank is required")
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("A child command is required")
    raise SystemExit(supervise(command))

if __name__ == "__main__":
    main()
