#!/usr/bin/env python3
"""Verify the latest Moonshot A phase-status artifact."""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys


def latest_status() -> pathlib.Path:
    paths = [pathlib.Path(p) for p in glob.glob("/tmp/moonshot-a-phase-status.*/moonshot_a_phase_status.v1.json")]
    if not paths:
        raise SystemExit("moonshot_a_phase_status_verify: no /tmp phase-status artifact found")
    return max(paths, key=lambda p: p.stat().st_mtime)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?")
    ap.add_argument("--latest", action="store_true")
    args = ap.parse_args()

    path = latest_status() if args.latest else pathlib.Path(args.path or "moonshot_a_phase_status.v1.json")
    data = json.loads(path.read_text())
    assert data["schema"] == "sounio.moonshot_a.phase_status.v1"
    assert data["local_evidence"]["status"] == "pass"
    assert data["status"] in {"PASS_LOCAL_GPU_PENDING", "PASS_FULL"}
    gpu_status = data["gpu_runtime"]["status"]
    if data["status"] == "PASS_FULL":
        assert gpu_status == "pass"
    else:
        assert gpu_status in {"not_run", "artifacts_present_not_passing", "artifacts_missing"}
    print(f"moonshot_a_phase_status_verify: PASS status={data['status']} path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
