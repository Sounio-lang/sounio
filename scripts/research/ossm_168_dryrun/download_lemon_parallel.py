#!/usr/bin/env python3
"""Parallel LEMON raw EEG downloader — workspace server local execution.

Downloads .vhdr / .vmrk / .eeg for each subject from the GWDG mirror
to /workspace/data/lemon/raw_eeg/ using a thread pool (I/O bound).
"""
import concurrent.futures
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://ftp.gwdg.de/pub/misc/MPI-Leipzig_Mind-Brain-Body-LEMON/EEG_MPILMBB_LEMON/EEG_Raw_BIDS_ID"
RAW_ROOT = Path("/workspace/data/lemon/raw_eeg")
SUBJECTS_LIST = Path("/workspace/data/lemon/subjects_all.txt")


def download_file(url: str, dest: Path) -> None:
    """Download url -> dest, raising on non-2xx."""
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as response:
        if response.status != 200:
            raise urllib.error.HTTPError(url, response.status, None, None, None)
        data = response.read()
    dest.write_bytes(data)


def download_subject(sid: str) -> tuple[str, bool, str]:
    out_dir = RAW_ROOT / sid / "RSEEG"
    vhdr_path = out_dir / f"{sid}.vhdr"
    if vhdr_path.exists() and vhdr_path.stat().st_size > 1000:
        return sid, True, "already present"

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in (".vhdr", ".vmrk", ".eeg"):
        url = f"{BASE_URL}/{sid}/RSEEG/{sid}{ext}"
        dest = out_dir / f"{sid}{ext}"
        try:
            download_file(url, dest)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return sid, False, f"{ext}: 404 (no EEG for this subject)"
            return sid, False, f"{ext}: HTTP {e.code}"
        except Exception as e:
            return sid, False, f"{ext}: {type(e).__name__}: {e}"
    return sid, True, "downloaded"


def main() -> int:
    if not SUBJECTS_LIST.exists():
        print(f"ERROR: {SUBJECTS_LIST} not found", file=sys.stderr)
        return 1

    subjects = [line.strip() for line in SUBJECTS_LIST.read_text().splitlines() if line.strip()]
    missing = []
    for sid in subjects:
        vhdr = RAW_ROOT / sid / "RSEEG" / f"{sid}.vhdr"
        if not vhdr.exists() or vhdr.stat().st_size < 1000:
            missing.append(sid)

    print(f"Total subjects: {len(subjects)}")
    print(f"Already present: {len(subjects) - len(missing)}")
    print(f"Missing: {len(missing)}")

    if not missing:
        print("All subjects already downloaded.")
        return 0

    n_threads = min(32, len(missing))
    ok_count = 0
    fail_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = {executor.submit(download_subject, sid): sid for sid in missing}
        for future in concurrent.futures.as_completed(futures):
            sid, ok, msg = future.result()
            tag = "OK" if ok else "FAIL"
            print(f"[{tag}] {sid}: {msg}")
            if ok:
                ok_count += 1
            else:
                fail_count += 1

    print(f"\nSummary: {ok_count} ok, {fail_count} failed (out of {len(missing)} missing)")
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
