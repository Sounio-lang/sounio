#!/usr/bin/env python3
"""Seal/verify a same-run Linux Madaros artifact; never fall back to a prebuilt.

This is integrity and build provenance, not an independent reproducibility proof.
The producer output digest is passed separately through Actions job outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def context():
    keys = ("GITHUB_SHA", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    values = {key: os.environ[key] for key in keys}
    if any(not value for value in values.values()):
        raise ValueError("empty run context")
    if git("rev-parse", "HEAD") != values["GITHUB_SHA"]:
        raise ValueError("checkout is not the event SHA")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("tracked build inputs are dirty")
    return dict(schema=1, platform="linux-x86_64", source_sha=values["GITHUB_SHA"],
                source_tree=git("rev-parse", "HEAD^{tree}"),
                repository=values["GITHUB_REPOSITORY"], run_id=values["GITHUB_RUN_ID"],
                bootstrap_sha256=sha("bin/souc-linux-x86_64"),
                builder_sha256=sha("scripts/ci/build_modular_madaros.sh"))


def check_elf(path):
    data = path.read_bytes()[:20]
    if len(data) < 20 or data[:6] != b"\x7fELF\x02\x01" or data[18:20] != b"\x3e\x00":
        raise ValueError("artifact is not an ELF64 little-endian x86_64 binary")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args()
    binary = args.directory / "madaros"
    receipt = args.directory / "provenance.json"
    check_elf(binary)
    digest = sha(binary)
    expected = context()
    if args.mode == "seal":
        expected.update(binary_sha256=digest, producer_attempt=os.environ["GITHUB_RUN_ATTEMPT"])
        receipt.write_text(json.dumps(expected, indent=2, sort_keys=True) + "\n")
        output = os.environ.get("GITHUB_OUTPUT")
        if output:
            with open(output, "a") as stream:
                stream.write(f"sha256={digest}\n")
    else:
        actual = json.loads(receipt.read_text())
        if not args.expected_sha256 or digest != args.expected_sha256:
            raise ValueError("binary differs from producer job digest")
        if actual.get("binary_sha256") != digest:
            raise ValueError("binary differs from provenance digest")
        for key, value in expected.items():
            if actual.get(key) != value:
                raise ValueError(f"provenance mismatch: {key}")
        if not str(actual.get("producer_attempt", "")).isdigit():
            raise ValueError("missing producer attempt")
        # Partial job reruns reuse the successful producer from the SAME run.
        # run_id + source SHA + digest stay strict; attempt need not be current.
        binary.chmod(0o755)
    print(f"CANONICAL_MADAROS_{args.mode.upper()} sha256={digest}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"CANONICAL_MADAROS_FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
