#!/usr/bin/env python3
"""Prove that the portable build lock serializes concurrent commands."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[2]
LOCKER = ROOT / "scripts/dev/souc_build_lock.py"


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="sounio-build-lock-selftest.") as raw_dir:
        work_dir = Path(raw_dir)
        lock = work_dir / "build.lock"
        events = work_dir / "events"
        worker = work_dir / "worker.py"
        worker.write_text(
            "import pathlib, sys, time\n"
            "events = pathlib.Path(sys.argv[1])\n"
            "label = sys.argv[2]\n"
            "delay = float(sys.argv[3])\n"
            "with events.open('a') as f: f.write(label + ':start\\n')\n"
            "time.sleep(delay)\n"
            "with events.open('a') as f: f.write(label + ':end\\n')\n",
            encoding="utf-8",
        )

        first = subprocess.Popen(
            [
                sys.executable,
                str(LOCKER),
                str(lock),
                sys.executable,
                str(worker),
                str(events),
                "first",
                "0.4",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.1)
        second = subprocess.Popen(
            [
                sys.executable,
                str(LOCKER),
                str(lock),
                sys.executable,
                str(worker),
                str(events),
                "second",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        first_out, first_err = first.communicate(timeout=5)
        second_out, second_err = second.communicate(timeout=5)

        if first.returncode != 0 or second.returncode != 0:
            print(first_out + first_err + second_out + second_err, file=sys.stderr)
            return 1
        if events.read_text(encoding="utf-8").splitlines() != [
            "first:start",
            "first:end",
            "second:start",
            "second:end",
        ]:
            print(events.read_text(encoding="utf-8"), file=sys.stderr)
            return 1
        if "another heavy build holds the lock; waiting..." not in second_err:
            print(second_err, file=sys.stderr)
            return 1

    print("SOUC_BUILD_LOCK_SELFTEST_PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
