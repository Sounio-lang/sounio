#!/usr/bin/env python3
"""
Garden-to-Claim pipeline contract — Zero of Encounter instantiation.

Verifies that the Garden seed
  docs/internal/garden/seeds/2026-07-11-the-zero-of-encounter.md
is driven through Garden -> Hypothesis -> Executable -> Claim-ready
(ledger-scoped), with the claim encoded in the Falsification Ledger and
bounded by the seed's own evidence declaration.

Companion to:
  docs/research/garden_to_claim_pipeline_spec_2026-07-25.md
  docs/research/falsification_ledger_spec_2026-07-25.md

Pure Python; no external dependencies.
"""

import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import falsification_ledger_contract as ledger

SEED = 'docs/internal/garden/seeds/2026-07-11-the-zero-of-encounter.md'
CLAIM_FILE = 'stdlib/epistemic/zero_encounter_pipeline_claim.sio'
CLAIM_NAME = 'garden_zero_encounter_pipeline'
WITNESSES = [
    'stdlib/epistemic/zero_event.sio',
    'tests/known_failures/zero_provenance_native_v2_probe.sio',
]
GATES = [
    'scripts/ci/zero_event_gate.sh',
    'scripts/ci/zero_provenance_witness_gate.sh',
    'scripts/ci/zero_event_native_v2_matrix.sh',
]
PROBE = 'tests/known_failures/zero_provenance_native_v2_probe.sio'
PROBE_MARKER = 'ZERO_PROVENANCE PASS'
GARDEN_LABELS = ['Garden', 'Hypothesis', 'Executable', 'Claim-ready']


def repo_path(rel):
    return os.path.join(REPO_ROOT, rel)


def read_text(rel):
    with open(repo_path(rel), 'r', encoding='utf-8') as f:
        return f.read()


def check_P1_seed_structure():
    ok = True
    if not os.path.isfile(repo_path(SEED)):
        print(f"P1_FAIL seed missing {SEED}")
        ok = False
    else:
        text = read_text(SEED)
        if '## Evidence State' not in text:
            print("P1_FAIL seed lacks Evidence State section")
            ok = False
        for label in GARDEN_LABELS:
            if not re.search(r'\|\s*`?' + re.escape(label) + r'`?\s*\|', text):
                print(f"P1_FAIL seed Evidence State lacks row for {label}")
                ok = False
        if '## What This Is Not' not in text:
            print("P1_FAIL seed lacks What This Is Not boundary section")
            ok = False
        if 'Executable Bridge' not in text:
            print("P1_FAIL seed lacks Executable Bridge section")
            ok = False
    print(f"P1_SEED_STRUCTURE {'PASS' if ok else 'FAIL'}")
    return ok


def check_P2_witnesses_exist():
    ok = True
    for w in WITNESSES:
        if not os.path.isfile(repo_path(w)):
            print(f"P2_FAIL witness missing {w}")
            ok = False
    if ok and PROBE_MARKER not in read_text(PROBE):
        print(f"P2_FAIL probe lost pass marker {PROBE_MARKER}")
        ok = False
    print(f"P2_WITNESSES_EXIST {'PASS' if ok else 'FAIL'}")
    return ok


def check_P3_gates_executable():
    ok = True
    for g in GATES:
        p = repo_path(g)
        if not os.path.isfile(p):
            print(f"P3_FAIL gate missing {g}")
            ok = False
        elif not os.access(p, os.X_OK):
            print(f"P3_FAIL gate not executable {g}")
            ok = False
    print(f"P3_GATES_EXECUTABLE {'PASS' if ok else 'FAIL'}")
    return ok


def find_pipeline_claim():
    claims = ledger.scan_file(repo_path(CLAIM_FILE))
    for c in claims:
        if c.claim == CLAIM_NAME:
            return c
    return None


def check_P4_ledger_claim():
    ok = True
    claim = find_pipeline_claim()
    if claim is None:
        print(f"P4_FAIL claim {CLAIM_NAME} not found or incomplete in {CLAIM_FILE}")
        ok = False
    else:
        if claim.evidence not in ledger.EVIDENCE_LEVELS:
            print(f"P4_FAIL invalid evidence '{claim.evidence}'")
            ok = False
        if claim.verdict not in ledger.VERDICTS:
            print(f"P4_FAIL invalid verdict '{claim.verdict}'")
            ok = False
        for rel, kind in ((claim.harness, 'harness'), (claim.gate, 'gate')):
            if not os.path.isfile(repo_path(rel)):
                print(f"P4_FAIL {kind} path does not resolve: {rel}")
                ok = False
    print(f"P4_LEDGER_CLAIM {'PASS' if ok else 'FAIL'}")
    return ok


def check_P5_evidence_ceiling():
    ok = True
    claim = find_pipeline_claim()
    if claim is None:
        print(f"P5_FAIL claim {CLAIM_NAME} not found; cannot check ceiling")
        ok = False
    elif claim.evidence == 'claim_ready':
        row = ''
        if os.path.isfile(repo_path(SEED)):
            m = re.search(r'^\|\s*`?Claim-ready`?\s*\|(.*)\|\s*$', read_text(SEED), re.M)
            row = m.group(1).strip() if m else ''
        if not row.lower().startswith('yes'):
            print("P5_FAIL claim evidence is claim_ready but seed Claim-ready row does not open with Yes")
            ok = False
    print(f"P5_EVIDENCE_CEILING {'PASS' if ok else 'FAIL'}")
    return ok


def check_P6_engine_split_disclosed():
    ok = True
    claim = find_pipeline_claim()
    note = (claim.note or '') if claim else ''
    if 'lean_single' not in note:
        print("P6_FAIL claim note does not disclose the lean_single execution engine")
        ok = False
    gate_path = repo_path('scripts/ci/garden_to_claim_gate.sh')
    if not os.path.isfile(gate_path):
        print("P6_FAIL pipeline gate missing scripts/ci/garden_to_claim_gate.sh")
        ok = False
    else:
        gate_text = read_text('scripts/ci/garden_to_claim_gate.sh')
        for composed in ('zero_event_gate.sh', 'zero_provenance_witness_gate.sh'):
            if composed not in gate_text:
                print(f"P6_FAIL pipeline gate does not compose {composed}")
                ok = False
    print(f"P6_ENGINE_SPLIT_DISCLOSED {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("GARDEN-TO-CLAIM PIPELINE — contract (Zero of Encounter)")
    print("=" * 70)
    results = [
        ("P1", check_P1_seed_structure()),
        ("P2", check_P2_witnesses_exist()),
        ("P3", check_P3_gates_executable()),
        ("P4", check_P4_ledger_claim()),
        ("P5", check_P5_evidence_ceiling()),
        ("P6", check_P6_engine_split_disclosed()),
    ]
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"GARDEN_TO_CLAIM_VERDICT P_GREEN ({passed}/{total} clauses PASS)")
        print("GARDEN_TO_CLAIM_NOTE seed=zero-of-encounter; ledger_scoped_claim; engine_split_disclosed")
        return 0
    print(f"GARDEN_TO_CLAIM_VERDICT P_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
