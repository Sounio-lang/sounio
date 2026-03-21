#!/usr/bin/env python3
"""Check URL and DOI reachability in a BibTeX file."""

from __future__ import annotations

import argparse
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Tuple


ENTRY_RE = re.compile(r"@[\w]+\{([^,]+),", re.IGNORECASE)
URL_RE = re.compile(r"\burl\s*=\s*\{([^}]+)\}", re.IGNORECASE)
DOI_RE = re.compile(r"\bdoi\s*=\s*\{([^}]+)\}", re.IGNORECASE)


def split_entries(text: str) -> List[Tuple[str, str]]:
    entries: List[Tuple[str, str]] = []
    starts = list(ENTRY_RE.finditer(text))
    for i, m in enumerate(starts):
        key = m.group(1).strip()
        start = m.start()
        end = starts[i + 1].start() if i + 1 < len(starts) else len(text)
        entries.append((key, text[start:end]))
    return entries


def check_url(url: str, timeout_s: float) -> Tuple[bool, str]:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "sounio-paper-ref-check/1.0"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            status = getattr(resp, "status", 200)
            if 200 <= status < 400 or status == 403:
                return True, f"HTTP {status}"
            return False, f"HTTP {status}"
    except urllib.error.HTTPError as e:
        if 200 <= e.code < 400 or e.code == 403:
            return True, f"HTTP {e.code}"
        return False, f"HTTP {e.code}"
    except Exception as e:  # pragma: no cover - network variability
        return False, str(e)


def check_doi(doi: str, timeout_s: float) -> Tuple[bool, str]:
    url = f"https://doi.org/{doi}"
    return check_url(url, timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("bib", type=Path, help="Path to BibTeX file")
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Network timeout in seconds (default: 8.0)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any reference check fails",
    )
    args = parser.parse_args()

    text = args.bib.read_text(encoding="utf-8")
    entries = split_entries(text)

    failures = 0
    checked = 0

    for key, body in entries:
        urls = URL_RE.findall(body)
        dois = DOI_RE.findall(body)

        for url in urls:
            checked += 1
            ok, detail = check_url(url.strip(), args.timeout)
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {key} url {url.strip()} ({detail})")
            if not ok:
                failures += 1

        for doi in dois:
            checked += 1
            ok, detail = check_doi(doi.strip(), args.timeout)
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {key} doi {doi.strip()} ({detail})")
            if not ok:
                failures += 1

    print(f"Checked {checked} url/doi fields; failures={failures}")
    if args.strict and failures > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
