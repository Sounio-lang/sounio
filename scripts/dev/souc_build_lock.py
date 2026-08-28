#!/usr/bin/env python3
"""Portable advisory-lock fallback for souc-build-lock.sh."""

from __future__ import annotations

import fcntl
import os
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(f"usage: {sys.argv[0]} <lock path> <build command...>", file=sys.stderr)
        return 64

    lock_path = sys.argv[1]
    command = sys.argv[2:]
    lock_file = open(lock_path, "a", encoding="utf-8")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(
            "[souc-build-lock] another heavy build holds the lock; waiting...",
            file=sys.stderr,
            flush=True,
        )
        fcntl.flock(lock_file, fcntl.LOCK_EX)

    print(
        f"[souc-build-lock] acquired ({lock_path}); running: {' '.join(command)}",
        file=sys.stderr,
        flush=True,
    )
    os.set_inheritable(lock_file.fileno(), True)
    os.execvp(command[0], command)
    return 70


if __name__ == "__main__":
    raise SystemExit(main())
