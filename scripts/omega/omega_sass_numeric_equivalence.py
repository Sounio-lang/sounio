#!/usr/bin/env python3
"""Sounio Omega Sprint 1 - numeric equivalence harness for L2 kernels.

Modes:
- fallback_symbol_numeric: executes the exported fallback symbol in both baseline
  and candidate binaries for N samples (bit-exact comparison).
- text_hash_proxy: compares `.text` section hashes when direct numeric execution
  is not available (e.g. real CUDA cubin).
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True, check=False)


def detect_symbol(binary: Path, symbol: str) -> bool:
    nm = shutil.which("nm")
    if not nm:
        return False
    result = run([nm, "-g", str(binary)])
    if result.returncode != 0:
        return False
    return symbol in result.stdout


def build_shared_from_obj(obj: Path, out_so: Path) -> Tuple[bool, str]:
    cc = shutil.which("gcc") or shutil.which("cc")
    if not cc:
        return False, "missing_cc"
    result = run(
        [
            cc,
            "-shared",
            "-Wl,--whole-archive",
            str(obj),
            "-Wl,--no-whole-archive",
            "-o",
            str(out_so),
        ]
    )
    if result.returncode != 0:
        return False, f"shared_link_failed:{result.stderr.strip()[:240]}"
    return True, "ok"


def execute_symbol_numeric(
    *,
    kernel: str,
    baseline: Path,
    candidate: Path,
    samples: int,
) -> Tuple[bool, dict]:
    symbol = f"sounio_{kernel}_seed_symbol"
    if not detect_symbol(baseline, symbol):
        return False, {"reason": f"symbol_missing_baseline:{symbol}"}
    if not detect_symbol(candidate, symbol):
        return False, {"reason": f"symbol_missing_candidate:{symbol}"}

    with tempfile.TemporaryDirectory(prefix="omega-eq-") as td:
        td_path = Path(td)
        baseline_so = td_path / f"{kernel}.baseline.so"
        candidate_so = td_path / f"{kernel}.candidate.so"

        ok, reason = build_shared_from_obj(baseline, baseline_so)
        if not ok:
            return False, {"reason": reason}
        ok, reason = build_shared_from_obj(candidate, candidate_so)
        if not ok:
            return False, {"reason": reason}

        try:
            base_lib = ctypes.CDLL(str(baseline_so))
            cand_lib = ctypes.CDLL(str(candidate_so))
            base_fn = getattr(base_lib, symbol)
            cand_fn = getattr(cand_lib, symbol)
        except Exception as exc:  # pragma: no cover
            return False, {"reason": f"symbol_load_failed:{exc}"}

        base_fn.argtypes = [ctypes.c_uint64]
        base_fn.restype = ctypes.c_uint64
        cand_fn.argtypes = [ctypes.c_uint64]
        cand_fn.restype = ctypes.c_uint64

        baseline_values: list[int] = []
        candidate_values: list[int] = []
        mismatches = 0
        mismatch_indices: list[int] = []

        for i in range(samples):
            x = ctypes.c_uint64(i).value
            b = int(base_fn(x))
            c = int(cand_fn(x))
            baseline_values.append(b)
            candidate_values.append(c)
            if b != c:
                mismatches += 1
                if len(mismatch_indices) < 8:
                    mismatch_indices.append(i)

        median_baseline = int(statistics.median(baseline_values)) if baseline_values else 0
        median_candidate = int(statistics.median(candidate_values)) if candidate_values else 0
        pass_rate = 1.0 - (mismatches / max(1, samples))
        passed = mismatches == 0

        return passed, {
            "mode": "fallback_symbol_numeric",
            "symbol": symbol,
            "samples": samples,
            "mismatches": mismatches,
            "pass_rate": pass_rate,
            "median_baseline": median_baseline,
            "median_candidate": median_candidate,
            "mismatch_indices": mismatch_indices,
            "reason": "bit_exact_match" if passed else "bit_exact_mismatch",
        }


def text_section_hash(path: Path) -> Tuple[bool, str, str]:
    objcopy = shutil.which("llvm-objcopy") or shutil.which("objcopy") or shutil.which("gobjcopy")
    if objcopy:
        with tempfile.TemporaryDirectory(prefix="omega-text-") as td:
            sec_path = Path(td) / "text.bin"
            result = run([objcopy, "--dump-section", f".text={sec_path}", str(path)])
            if result.returncode == 0 and sec_path.exists():
                sha = hashlib.sha256(sec_path.read_bytes()).hexdigest()
                return True, sha, "ok_objcopy"

    # CUDA cubins can carry non-standard ELF metadata that breaks objcopy.
    # Fallback: hash a normalized readelf hex dump for the first .text* section.
    readelf = shutil.which("readelf")
    if not readelf:
        return False, "", "missing_readelf"

    sections = run([readelf, "-W", "-S", str(path)])
    if sections.returncode != 0:
        return False, "", f"readelf_sections_failed:{sections.stderr.strip()[:240]}"

    text_section = ""
    for line in sections.stdout.splitlines():
        m = re.search(r"\]\s+(\.text[^\s]*)\s", line)
        if m:
            text_section = m.group(1)
            break
    if not text_section:
        return False, "", "text_section_not_found"

    dump = run([readelf, "-W", "-x", text_section, str(path)])
    if dump.returncode != 0:
        return False, "", f"readelf_hex_failed:{dump.stderr.strip()[:240]}"

    hex_tokens: list[str] = []
    for line in dump.stdout.splitlines():
        if not line.startswith("  0x"):
            continue
        hex_tokens.extend(re.findall(r"[0-9a-fA-F]{8}", line))
    if not hex_tokens:
        return False, "", "readelf_hex_empty"

    payload = "".join(hex_tokens).lower().encode("ascii")
    return True, hashlib.sha256(payload).hexdigest(), "ok_readelf_hex"


def proxy_text_hash(baseline: Path, candidate: Path, samples: int) -> Tuple[bool, dict]:
    ok_b, sha_b, reason_b = text_section_hash(baseline)
    ok_c, sha_c, reason_c = text_section_hash(candidate)
    if not ok_b or not ok_c:
        return False, {
            "mode": "text_hash_proxy",
            "samples": samples,
            "reason": f"{reason_b}|{reason_c}",
            "mismatches": samples,
            "pass_rate": 0.0,
        }
    passed = sha_b == sha_c
    return passed, {
        "mode": "text_hash_proxy",
        "samples": samples,
        "baseline_text_sha256": sha_b,
        "candidate_text_sha256": sha_c,
        "mismatches": 0 if passed else samples,
        "pass_rate": 1.0 if passed else 0.0,
        "reason": "text_hash_match" if passed else "text_hash_mismatch",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Omega numeric equivalence harness")
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--samples", type=int, default=int(os.environ.get("OMEGA_EQ_SAMPLES", "50")))
    parser.add_argument("--report", default="")
    parser.add_argument(
        "--proxy-allowed",
        action="store_true",
        help="Allow text-hash proxy mode when direct numeric execution is unavailable",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline = Path(args.baseline)
    candidate = Path(args.candidate)

    if not baseline.exists():
        print(f"equivalence error: baseline missing: {baseline}", file=sys.stderr)
        return 2
    if not candidate.exists():
        print(f"equivalence error: candidate missing: {candidate}", file=sys.stderr)
        return 2

    passed, details = execute_symbol_numeric(
        kernel=args.kernel,
        baseline=baseline,
        candidate=candidate,
        samples=args.samples,
    )

    if not passed:
        if args.proxy_allowed:
            proxy_pass, proxy_details = proxy_text_hash(
                baseline=baseline,
                candidate=candidate,
                samples=args.samples,
            )
            passed = proxy_pass
            details = proxy_details
        else:
            details.setdefault("mode", "fallback_symbol_numeric")

    payload = {
        "schema": "sounio.omega.numeric-equivalence.v1",
        "kernel": args.kernel,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "passed": passed,
        **details,
    }

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(payload, indent=2))

    if passed:
        print(
            "numeric_equivalence pass: "
            f"kernel={args.kernel} mode={payload.get('mode')} "
            f"samples={payload.get('samples')} reason={payload.get('reason')}"
        )
        return 0

    print(
        "numeric_equivalence fail: "
        f"kernel={args.kernel} mode={payload.get('mode')} "
        f"reason={payload.get('reason')}",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
