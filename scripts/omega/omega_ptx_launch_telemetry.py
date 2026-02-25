#!/usr/bin/env python3
"""Emit Omega PTX launch telemetry artifact from pure-Sounio launch test output."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

PTX_LAUNCH_SCHEMA = "sounio.omega.ptx-launch-report.v1"
PASS_MARKER = "real_ptx_launch_test PASS"
SUMMARY_RE = re.compile(
    r"kernel=(?P<kernel>[^,]+), "
    r"target=(?P<target>[^,]+), "
    r"accepted=(?P<accepted>true|false), "
    r"fallback=(?P<fallback>true|false), "
    r"op_kind=(?P<op_kind>\d+), "
    r"flags=(?P<flags>\d+), "
    r"hw_log_q32_32=(?P<hw_log>\d+)"
)
KV_RE = re.compile(r"^PTX_LAUNCH_(?P<key>HW|SW|HYBRID)_LOG_Q32_32=(?P<value>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omega PTX launch telemetry emitter")
    parser.add_argument(
        "--test-file",
        default="tests/hardware/real_ptx_launch_test.sio",
        help="Sounio test file that exercises pure PTX launch path",
    )
    parser.add_argument(
        "--out-dir",
        default="artifacts/ptx/omega",
        help="Output directory for PTX launch report",
    )
    parser.add_argument(
        "--souc-bin",
        default="souc",
        help="souc binary path/name",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if launch contract is not accepted/non-fallback or score invariants break",
    )
    return parser.parse_args()


def resolve_souc_bin(raw: str) -> str:
    candidate = raw.strip()
    if candidate and Path(candidate).exists():
        return str(Path(candidate).resolve())

    root = Path(__file__).resolve().parents[2]
    default_candidates = [
        root / "souc",
        root / "target" / "debug" / "souc",
        root / "target" / "release" / "souc",
    ]
    for path in default_candidates:
        if path.exists():
            return str(path)

    which = shutil.which(candidate if candidate else "souc")
    if which:
        return which
    raise SystemExit("unable to resolve souc binary; pass --souc-bin or build target/{debug,release}/souc")


def run_test(souc_bin: str, test_file: str) -> tuple[int, str]:
    proc = subprocess.run(
        [souc_bin, "run", test_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def parse_bool(token: str) -> bool:
    return token.strip().lower() == "true"


def extract_required(output: str) -> dict:
    pass_seen = PASS_MARKER in output

    summary_match = None
    for line in output.splitlines():
        m = SUMMARY_RE.search(line.strip())
        if m:
            summary_match = m
            break
    if summary_match is None:
        raise SystemExit("missing PTX launch summary line in test output")

    logs: dict[str, int] = {}
    for line in output.splitlines():
        m = KV_RE.match(line.strip())
        if not m:
            continue
        logs[m.group("key")] = int(m.group("value"))

    missing_logs = [k for k in ("HW", "SW", "HYBRID") if k not in logs]
    if missing_logs:
        raise SystemExit(f"missing PTX launch log markers in test output: {missing_logs}")

    payload = {
        "pass_marker_seen": pass_seen,
        "kernel_name": summary_match.group("kernel").strip(),
        "target": summary_match.group("target").strip(),
        "launch_accepted": parse_bool(summary_match.group("accepted")),
        "used_fallback": parse_bool(summary_match.group("fallback")),
        "op_kind": int(summary_match.group("op_kind")),
        "replay_flags": int(summary_match.group("flags")),
        "hardware_epistemic_power_log_q32_32": logs["HW"],
        "software_epistemic_power_log_q32_32": logs["SW"],
        "hybrid_epistemic_power_log_q32_32": logs["HYBRID"],
    }
    return payload


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    args = parse_args()
    souc_bin = resolve_souc_bin(args.souc_bin)
    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"error: PTX launch test file not found: {test_path}", file=sys.stderr)
        return 2

    rc, output = run_test(souc_bin, args.test_file)
    if rc != 0:
        print(
            f"error: PTX launch test execution failed (rc={rc}, souc={souc_bin})",
            file=sys.stderr,
        )
        print(output, file=sys.stderr)
        return rc

    data = extract_required(output)
    data["schema"] = PTX_LAUNCH_SCHEMA
    data["test_file"] = str(test_path)
    data["output_sha256"] = hashlib.sha256(output.encode("utf-8")).hexdigest()

    out_dir = Path(args.out_dir)
    report_path = out_dir / "ptx_launch_report.json"
    write_json(report_path, data)

    print(
        "omega_ptx_launch_telemetry: "
        f"kernel={data['kernel_name']} target={data['target']} "
        f"accepted={str(data['launch_accepted']).lower()} "
        f"fallback={str(data['used_fallback']).lower()} "
        f"report={report_path}"
    )

    if args.strict:
        if not data["pass_marker_seen"]:
            print("omega_ptx_launch_telemetry: strict failed (pass marker missing)", file=sys.stderr)
            return 2
        if not data["launch_accepted"]:
            print("omega_ptx_launch_telemetry: strict failed (launch_accepted=false)", file=sys.stderr)
            return 2
        if data["used_fallback"]:
            print("omega_ptx_launch_telemetry: strict failed (used_fallback=true)", file=sys.stderr)
            return 2
        if data["software_epistemic_power_log_q32_32"] <= 0:
            print("omega_ptx_launch_telemetry: strict failed (software log <= 0)", file=sys.stderr)
            return 2
        if data["hybrid_epistemic_power_log_q32_32"] <= 0:
            print("omega_ptx_launch_telemetry: strict failed (hybrid log <= 0)", file=sys.stderr)
            return 2
        if (
            data["hybrid_epistemic_power_log_q32_32"]
            < data["hardware_epistemic_power_log_q32_32"]
        ):
            print(
                "omega_ptx_launch_telemetry: strict failed (hybrid log < hardware log)",
                file=sys.stderr,
            )
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
