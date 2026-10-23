#!/usr/bin/env python3
"""Audit journal orchestration against a Git commit; never qualify or launch."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

def audit(repo, source, expected):
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("full lowercase source commit required")
    commit = subprocess.run(["git", "cat-file", "-t", source],
                            cwd=repo, capture_output=True, text=True)
    if commit.returncode or commit.stdout.strip() != "commit":
        raise ValueError("source must identify an existing commit")
    if not expected:
        raise ValueError("empty orchestration inventory")
    rows = []
    for name, pin in sorted(expected.items()):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or not name.startswith("tools/pireus/continuity/"):
            raise ValueError("invalid orchestration path")
        if not re.fullmatch(r"[0-9a-f]{64}", pin):
            raise ValueError("invalid expected hash")
        blob = subprocess.run(["git", "show", source + ":" + name],
                              cwd=repo, capture_output=True)
        actual = hashlib.sha256(blob.stdout).hexdigest() if blob.returncode == 0 else None
        rows.append(dict(path=name, expected_sha256=pin, source_sha256=actual,
                         present=blob.returncode == 0, matches=actual == pin))
    return dict(schema="pireus-journal-source-coverage-v1", source_commit=source,
                files=rows, orchestration_content_verified=all(x["matches"] for x in rows),
                source_ci_checked=False, source_qualified=False,
                inference_submitted=False, protocol_rebound=False)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()
    import host_decode_journal_attempt as journal
    context = journal.context()
    repo = Path(__file__).resolve().parents[4]
    expected = {"tools/pireus/continuity/" + p: h
                for p, h in context.orchestration_hashes().items()}
    result = audit(repo, args.source, expected)
    result["protocol_source_commit"] = context.protocol()["source_commit"]
    result["protocol_sha256"] = context.PROTOCOL_SHA
    result["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(result, indent=2))
    return 0 if result["orchestration_content_verified"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
