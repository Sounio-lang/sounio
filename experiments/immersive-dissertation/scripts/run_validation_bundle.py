#!/usr/bin/env python3
"""Run the immersive dissertation validation bundle and persist evidence."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def run_check(name: str, command: list[str], env: dict[str, str] | None = None, timeout_s: int = 45) -> dict:
    started = time.time()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as error:
        return {
            "name": name,
            "command": command,
            "returncode": 124,
            "status": "fail",
            "optional_fallback": False,
            "duration_s": round(time.time() - started, 3),
            "stdout": (error.stdout or "").strip() if isinstance(error.stdout, str) else "",
            "stderr": ((error.stderr or "") if isinstance(error.stderr, str) else "") + f"\nTIMEOUT after {timeout_s}s",
        }
    output = completed.stdout + "\n" + completed.stderr
    optional_fallback = (
        "WEBGPU_RUNTIME_NOT_AVAILABLE" in output
        or "WEBGPU_PBPK_KERNEL_RUNTIME_NOT_AVAILABLE" in output
    )
    status = "pass" if completed.returncode == 0 else "fail"
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "status": status,
        "optional_fallback": optional_fallback,
        "duration_s": round(time.time() - started, 3),
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def wait_for_server(port: int, proc: subprocess.Popen[str]) -> None:
    deadline = time.time() + 10
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("http server exited before becoming ready")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise RuntimeError("http server did not become ready")


def git_value(args: list[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default="firefox")
    parser.add_argument("--out-dir", default="/tmp/sounio-immersive-recovery-phase1-validation")
    parser.add_argument("--require-webgpu", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    port = free_port()
    url = f"http://127.0.0.1:{port}/"
    env = os.environ.copy()
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", str(port), "--bind", "127.0.0.1"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    checks: list[dict] = []
    artifacts = {
        "screenshot": str(out_dir / "immersive-screenshot.png"),
        "webgpu_report": str(out_dir / "webgpu-runtime.json"),
        "webgpu_pbpk_kernel_report": str(out_dir / "webgpu-pbpk-kernel-runtime.json"),
    }
    try:
        wait_for_server(port, server)
        checks.extend(
            [
                run_check(
                    "webgpu_pbpk_kernel_contract",
                    [sys.executable, str(ROOT / "scripts/verify_webgpu_pbpk_kernel_contract.py"), str(ROOT / "data/webgpu-pbpk-kernel-contract.json")],
                    env,
                ),
                run_check(
                    "gpu_promotion_contract",
                    [sys.executable, str(ROOT / "scripts/verify_gpu_promotion_contract.py"), str(ROOT / "data/render-quality-contract.json")],
                    env,
                ),
                run_check("experience_static", [sys.executable, str(ROOT / "scripts/verify_experience_static.py")], env),
                run_check("browser_interaction", ["node", str(ROOT / "scripts/verify_browser_interaction.mjs"), url], env),
                run_check(
                    "screenshot_capture",
                    ["node", str(ROOT / "scripts/capture_screenshot.mjs"), "--browser", args.browser, "--output", artifacts["screenshot"], url],
                    env,
                ),
                run_check("screenshot_pixels", [sys.executable, str(ROOT / "scripts/verify_screenshot_pixels.py"), artifacts["screenshot"]], env),
                run_check(
                    "webgpu_runtime",
                    [
                        "node",
                        str(ROOT / "scripts/verify_webgpu_runtime.mjs"),
                        "--browser",
                        args.browser,
                        "--output",
                        artifacts["webgpu_report"],
                        *(["--require-webgpu"] if args.require_webgpu else []),
                        url,
                    ],
                    env,
                ),
                run_check(
                    "webgpu_pbpk_kernel_runtime",
                    [
                        "node",
                        str(ROOT / "scripts/verify_webgpu_pbpk_kernel_runtime.mjs"),
                        "--browser",
                        args.browser,
                        "--output",
                        artifacts["webgpu_pbpk_kernel_report"],
                        *(["--require-webgpu"] if args.require_webgpu else []),
                        url,
                    ],
                    env,
                ),
            ]
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    failures = [check for check in checks if check["status"] != "pass"]
    webgpu_promotion_eligible = (
        args.require_webgpu
        and not failures
        and all(not check.get("optional_fallback") for check in checks if "webgpu" in check.get("name", ""))
    )
    summary = {
        "schema": "sounio.immersive_dissertation.validation_summary.v1",
        "generated_date": "2026-06-27",
        "status": "pass" if not failures else "fail",
        "url": url,
        "browser": args.browser,
        "require_webgpu": args.require_webgpu,
        "webgpu_proof_required": args.require_webgpu,
        "webgpu_promotion_eligible": webgpu_promotion_eligible,
        "claim_boundary": "Fallback visibility pass is not a WebGPU or clinical calibration claim. All displayed concentration-time profiles are illustrative replays of previously published population parameters; no new parameter estimation or clinical validation is performed or claimed.",
        "source_revision": {
            "head": git_value(["rev-parse", "HEAD"]),
            "short_head": git_value(["rev-parse", "--short", "HEAD"]),
            "dirty_files": git_value(["status", "--short"]).splitlines(),
        },
        "checks": checks,
        "failures": [check["name"] for check in failures],
        "artifacts": artifacts,
    }
    summary_path = out_dir / "validation-summary.json"
    summary_path.write_text(f"{json.dumps(summary, indent=2)}\n", encoding="utf-8")

    if failures:
        print("IMMERSIVE_VALIDATION_BUNDLE_FAIL")
        print(json.dumps({"summary": str(summary_path), "failures": summary["failures"]}, indent=2))
        return 1
    print("IMMERSIVE_VALIDATION_BUNDLE_PASS")
    print(json.dumps({"summary": str(summary_path), "checks": len(checks), "artifacts": artifacts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
