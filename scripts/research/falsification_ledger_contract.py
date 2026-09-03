#!/usr/bin/env python3
"""
Falsification Ledger — scanner and validator for scientific claims in .sio files.

Companion to:
  docs/research/falsification_ledger_spec_2026-07-25.md
  docs/research/falsification_ledger_falsifiers_2026-07-25.md

Pure Python; no external dependencies.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from typing import List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EVIDENCE_LEVELS = {
    'conceived', 'implemented', 'type_check', 'compiles', 'executes',
    'gate_green', 'instrument_controlled', 'claim_ready'
}
VERDICTS = {'alive', 'negative', 'dormant', 'refuted'}
REQUIRED_FIELDS = ['claim', 'hypothesis', 'falsifier', 'evidence', 'harness', 'gate', 'verdict']
PROVENANCE_VALUES = {
    'absent', 'cancelled', 'annihilated', 'below_resolution', 'rounded', 'gated', 'unknown'
}
ZERO_KEYWORDS = {'zero', 'annihilat', 'cancel', 'round', 'resolution', 'absent', 'gated', 'unknown'}


@dataclass
class Claim:
    claim: str
    file: str
    line: int
    hypothesis: str
    falsifier: str
    evidence: str
    harness: str
    gate: str
    verdict: str
    note: Optional[str] = None
    archive_reason: Optional[str] = None
    provenance: Optional[str] = None
    sha: str = ''


def get_repo_sha():
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return 'unknown'


def scan_file(path):
    """Extract claim blocks from a .sio file. Returns list of Claim."""
    claims = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as exc:
        print(f"SCAN_ERROR {path}: {exc}")
        return claims

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m = re.match(r'^\s*//\s*@claim\s+(\S+)\s*$', line)
        if not m:
            i += 1
            continue
        claim_name = m.group(1)
        start_line = i + 1
        fields = {'claim': claim_name}
        i += 1
        while i < len(lines):
            line = lines[i].strip()
            m = re.match(r'^\s*//\s*@(\w+)\s+(.*)$', line)
            if not m:
                break
            key, value = m.group(1), m.group(2).strip()
            fields[key] = value
            i += 1
        # build claim if all required fields present
        missing = [k for k in REQUIRED_FIELDS if k not in fields]
        if missing:
            print(f"INCOMPLETE_CLAIM {claim_name} in {path}:{start_line} missing={missing}")
        else:
            claims.append(Claim(
                claim=fields['claim'],
                file=os.path.relpath(path, REPO_ROOT),
                line=start_line,
                hypothesis=fields['hypothesis'],
                falsifier=fields['falsifier'],
                evidence=fields['evidence'],
                harness=fields['harness'],
                gate=fields['gate'],
                verdict=fields['verdict'],
                note=fields.get('note'),
                archive_reason=fields.get('archive_reason'),
                provenance=fields.get('provenance'),
                sha=get_repo_sha(),
            ))
    return claims


def find_sio_files():
    """Find .sio files in the repo, excluding common build/cache dirs."""
    sio_files = []
    for root, dirs, files in os.walk(REPO_ROOT):
        # prune
        dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', '__pycache__', 'target', '.venv', 'artifacts'}]
        for f in files:
            if f.endswith('.sio'):
                sio_files.append(os.path.join(root, f))
    return sorted(sio_files)


def path_exists(rel_path):
    return os.path.exists(os.path.join(REPO_ROOT, rel_path))


def is_executable(rel_path):
    p = os.path.join(REPO_ROOT, rel_path)
    return os.path.isfile(p) and os.access(p, os.X_OK)


def check_L1_schema(claims):
    ok = True
    for c in claims:
        if c.evidence not in EVIDENCE_LEVELS:
            print(f"L1_FAIL invalid evidence '{c.evidence}' in {c.claim}")
            ok = False
        if c.verdict not in VERDICTS:
            print(f"L1_FAIL invalid verdict '{c.verdict}' in {c.claim}")
            ok = False
    print(f"L1_SCHEMA {'PASS' if ok else 'FAIL'}")
    return ok


def check_L2_harness_exists(claims):
    ok = True
    for c in claims:
        if not path_exists(c.harness):
            print(f"L2_FAIL harness missing {c.harness} for {c.claim}")
            ok = False
    print(f"L2_HARNESS_EXISTS {'PASS' if ok else 'FAIL'}")
    return ok


def check_L3_gate_exists(claims):
    ok = True
    for c in claims:
        if not path_exists(c.gate):
            print(f"L3_FAIL gate missing {c.gate} for {c.claim}")
            ok = False
        elif not is_executable(c.gate):
            print(f"L3_WARN gate not executable {c.gate} for {c.claim}")
    print(f"L3_GATE_EXISTS {'PASS' if ok else 'FAIL'}")
    return ok


def check_L4_no_orphans(claims):
    ok = True
    for c in claims:
        if not path_exists(c.file):
            print(f"L4_FAIL source file missing {c.file} for {c.claim}")
            ok = False
    print(f"L4_NO_ORPHANS {'PASS' if ok else 'FAIL'}")
    return ok


def check_L5_verdict_consistent(claims):
    ok = True
    for c in claims:
        if c.verdict == 'negative':
            if not c.falsifier.strip():
                print(f"L5_FAIL negative claim {c.claim} lacks falsifier")
                ok = False
            if not c.gate.strip():
                print(f"L5_FAIL negative claim {c.claim} lacks gate")
                ok = False
    print(f"L5_VERDICT_CONSISTENT {'PASS' if ok else 'FAIL'}")
    return ok


def check_L6_seed_rupture(claims):
    rupture_claims = [c for c in claims if 'rupture' in c.file and 'epistemic' in c.file]
    ok = len(rupture_claims) >= 8
    print(f"L6_SEED_RUPTURE found={len(rupture_claims)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def write_ledger(claims, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    branch = 'unknown'
    try:
        branch = subprocess.run(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        pass
    # sanitize branch name for filesystem
    branch_safe = branch.replace('/', '_')
    out_path = os.path.join(out_dir, f'{branch_safe}.jsonl')
    with open(out_path, 'w', encoding='utf-8') as f:
        for c in claims:
            record = {
                'claim': c.claim,
                'file': c.file,
                'line': c.line,
                'sha': c.sha,
                'hypothesis': c.hypothesis,
                'falsifier': c.falsifier,
                'evidence': c.evidence,
                'harness': c.harness,
                'gate': c.gate,
                'verdict': c.verdict,
                'note': c.note,
                'archive_reason': c.archive_reason,
                'provenance': c.provenance,
            }
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(f"LEDGER_WRITTEN {out_path} claims={len(claims)}")


def main():
    sio_files = find_sio_files()
    all_claims = []
    for path in sio_files:
        all_claims.extend(scan_file(path))

    # de-duplicate by claim name (keep first)
    seen = set()
    claims = []
    for c in all_claims:
        if c.claim not in seen:
            seen.add(c.claim)
            claims.append(c)

    print("=" * 70)
    print("FALSIFICATION LEDGER — contract")
    print("=" * 70)
    results = []
    results.append(("L1", check_L1_schema(claims)))
    results.append(("L2", check_L2_harness_exists(claims)))
    results.append(("L3", check_L3_gate_exists(claims)))
    results.append(("L4", check_L4_no_orphans(claims)))
    results.append(("L5", check_L5_verdict_consistent(claims)))
    results.append(("L6", check_L6_seed_rupture(claims)))
    print("=" * 70)

    ledger_dir = os.path.join(REPO_ROOT, '.sounio', 'claims')
    write_ledger(claims, ledger_dir)

    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"FALSIFICATION_LEDGER_VERDICT L_GREEN ({passed}/{total} clauses PASS)")
        print("FALSIFICATION_LEDGER_NOTE comment_scanned_claims; no_parser_change; seed_rupture_present")
        return 0
    else:
        print(f"FALSIFICATION_LEDGER_VERDICT L_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
