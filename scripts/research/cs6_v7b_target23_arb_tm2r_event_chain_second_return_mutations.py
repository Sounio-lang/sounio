#!/usr/bin/env python3
"""Negative mutations for the isolated-tile event-chain receipt verifier."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


TILES = ("XLEL", "XLEH", "XHEL", "XHEH")


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if text.count(old) != 1:
        raise RuntimeError(f"expected one occurrence of {old!r} in {path}")
    path.write_text(text.replace(old, new, 1))


def rejected(verifier: Path, receipts: Path) -> bool:
    completed = subprocess.run(
        [sys.executable, str(verifier), "--receipts", str(receipts)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode != 0


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    verifier = (
        script_dir
        / "cs6_v7b_target23_arb_tm2r_event_chain_second_return_verify.py"
    )
    source = (
        script_dir
        / "receipts"
        / "cs6_v7b_target23_arb_tm2r_event_chain_second_return_v1"
    )
    scenarios = (
        ("missing-tile", lambda root: (root / "XHEH.stdout.txt").unlink()),
        (
            "duplicate-tile",
            lambda root: replace_once(
                root / "XHEH.stdout.txt",
                "SOURCE_TILE_FILTER=XHEH",
                "SOURCE_TILE_FILTER=XHEL",
            ),
        ),
        (
            "chain-flag",
            lambda root: replace_once(
                root / "XLEL.stdout.txt",
                "SELECTED_SOURCE_CHAIN_CERTIFICATE=true",
                "SELECTED_SOURCE_CHAIN_CERTIFICATE=false",
            ),
        ),
        (
            "worker-hash",
            lambda root: replace_once(
                root / "XLEH.stdout.txt",
                "WORKER_SOURCE_SHA256=edde152f2be29f37eefaf1dd859b74f1984714f0b0aceeff53162fa1336d4fb5",
                "WORKER_SOURCE_SHA256=" + "0" * 64,
            ),
        ),
        (
            "normal-sign",
            lambda root: replace_once(
                root / "XHEL.stdout.txt",
                "SECOND_RETURN_NORMAL_HULL_LOWER_Q=",
                "SECOND_RETURN_NORMAL_HULL_LOWER_Q=-",
            ),
        ),
        (
            "point-fallback",
            lambda root: replace_once(
                root / "XHEH.stdout.txt",
                "POINT_FALLBACK_USED=false",
                "POINT_FALLBACK_USED=true",
            ),
        ),
    )
    rejected_count = 0
    with tempfile.TemporaryDirectory(prefix="cs6-event-chain-mutations-") as temp:
        temp_root = Path(temp)
        for name, mutate in scenarios:
            case = temp_root / name
            case.mkdir()
            for tile in TILES:
                shutil.copy2(source / f"{tile}.stdout.txt", case)
            mutate(case)
            if not rejected(verifier, case):
                print(f"MUTATION_ACCEPTED={name}", file=sys.stderr)
                return 1
            rejected_count += 1
            print(f"MUTATION_REJECTED={name}")
    print(f"MUTATIONS_REJECTED={rejected_count}")
    print("MUTATION_AUDIT=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
