"""Materialize byte-pinned external inputs; verify archived parent inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[4]
INPUTS = [
    {
        "path": "/tmp/intel-sdm-vol-2c-326018-092.pdf",
        "url": "https://cdrdv2-public.intel.com/922483/326018-092-sdm-vol-2c.pdf",
        "bytes": 3298744,
        "sha256": "939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07",
    },
    {
        "path": "/tmp/pireus-xed-v2026.08.23-20260827/datafiles/avx512f/avx512-foundation-isa.xed.txt",
        "url": "https://raw.githubusercontent.com/intelxed/xed/0bcb6237345c5066726dcc08b3d87928df3b5b26/datafiles/avx512f/avx512-foundation-isa.xed.txt",
        "bytes": 458470,
        "sha256": "e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038",
    },
]


def matches(data, expected):
    return len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"]


def prepare(cache_only=False, verify_history=False):
    records = []
    for item in INPUTS:
        path = Path(item["path"])
        if path.exists():
            if not matches(path.read_bytes(), item):
                raise ValueError("existing historical input differs: " + str(path))
            action = "verified-existing"
        else:
            if cache_only:
                raise ValueError("missing historical input: " + str(path))
            with urllib.request.urlopen(item["url"], timeout=60) as response:
                data = response.read(item["bytes"] + 1)
            if not matches(data, item):
                raise ValueError("downloaded historical input differs: " + item["url"])
            path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation preserves an unexpected concurrent writer.
            with path.open("xb") as stream:
                stream.write(data)
            action = "downloaded-and-verified"
        records.append(dict(item, action=action))
    manifest = json.loads((ROOT / "tools/pireus/continuity/ci/historical-parents/manifest.json").read_text())
    for item in manifest["files"]:
        data = (ROOT / item["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise ValueError("historical parent changed: " + item["path"])
        if verify_history:
            original = subprocess.check_output(
                ["git", "show", item["original_commit"] + ":" + item["original_path"]], cwd=ROOT)
            if data != original:
                raise ValueError("historical parent differs from original commit")
    return dict(external_inputs=records, historical_parents=len(manifest["files"]),
                parent_git_history_verified=verify_history,
                current_hardware_acceptance=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-only", action="store_true")
    parser.add_argument("--verify-history", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.cache_only, args.verify_history), indent=2))
