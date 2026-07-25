#!/usr/bin/env python3
"""
Zero-provenance claims — zero-event taxonomy inside the Falsification Ledger.

Companion to:
  docs/research/zero_provenance_claims_spec_2026-07-25.md
  docs/research/zero_provenance_claims_falsifiers_2026-07-25.md

Reuses the scanner from falsification_ledger_contract.py.
"""

import os
import sys

# Allow importing the ledger contract as a module
LEDGER_SCRIPT = os.path.join(os.path.dirname(__file__), 'falsification_ledger_contract.py')
sys.path.insert(0, os.path.dirname(LEDGER_SCRIPT))

from falsification_ledger_contract import (
    Claim,
    PROVENANCE_VALUES,
    ZERO_KEYWORDS,
    find_sio_files,
    scan_file,
    write_ledger,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CLINICAL_TERMS = {'patient', 'diagnosis', 'treatment', 'clinical', 'therapy', 'disease', 'symptom'}


def claims_with_provenance():
    """Scan all .sio files and return claims that have a provenance field."""
    sio_files = find_sio_files()
    all_claims = []
    for path in sio_files:
        all_claims.extend(scan_file(path))
    seen = set()
    claims = []
    for c in all_claims:
        if c.claim not in seen:
            seen.add(c.claim)
            claims.append(c)
    return claims


def is_zero_claim(c: Claim) -> bool:
    """
    A claim is a zero-claim if its evidence or falsifier explicitly invokes
    the zero-event taxonomy (not merely any occurrence of the word 'zero').
    """
    text = (c.evidence + ' ' + c.falsifier + ' ' + c.hypothesis).lower()
    specific = {'zero surface', 'provenance', 'zero-event', 'annihilated', 'below_resolution', 'rounded', 'gated'}
    return any(k in text for k in specific)


def check_Z1_provenance_enum(claims):
    ok = True
    for c in claims:
        if c.provenance is not None and c.provenance not in PROVENANCE_VALUES:
            print(f"Z1_FAIL invalid provenance '{c.provenance}' in {c.claim}")
            ok = False
    print(f"Z1_PROVENANCE_ENUM {'PASS' if ok else 'FAIL'}")
    return ok


def check_Z2_zero_claims_require_provenance(claims):
    ok = True
    for c in claims:
        if is_zero_claim(c) and c.provenance is None:
            print(f"Z2_FAIL zero-claim {c.claim} lacks provenance")
            ok = False
    print(f"Z2_ZERO_CLAIMS_REQUIRE_PROVENANCE {'PASS' if ok else 'FAIL'}")
    return ok


def check_Z3_nonzero_claims_optional(claims):
    ok = True
    for c in claims:
        if not is_zero_claim(c) and c.provenance is None:
            # allowed; nothing to check
            pass
    print(f"Z3_NONZERO_CLAIMS_OPTIONAL {'PASS' if ok else 'FAIL'}")
    return ok


def check_Z4_seed_taxonomy(claims):
    found = {c.provenance for c in claims if c.provenance is not None}
    missing = PROVENANCE_VALUES - found
    ok = len(missing) == 0
    if missing:
        print(f"Z4_FAIL missing provenance categories: {sorted(missing)}")
    print(f"Z4_SEED_TAXONOMY found={sorted(found)} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_Z5_ledger_includes_provenance(claims):
    # Run the ledger writer and inspect the output
    ledger_dir = os.path.join(REPO_ROOT, '.sounio', 'claims')
    write_ledger(claims, ledger_dir)
    # find the most recent ledger file
    import glob
    files = glob.glob(os.path.join(ledger_dir, '*.jsonl'))
    if not files:
        print("Z5_FAIL no ledger file found")
        print("Z5_LEDGER_INCLUDES_PROVENANCE FAIL")
        return False
    latest = max(files, key=os.path.getmtime)
    with open(latest, 'r', encoding='utf-8') as f:
        for line in f:
            if 'provenance' not in line:
                print(f"Z5_FAIL ledger record lacks provenance key: {line[:80]}")
                print("Z5_LEDGER_INCLUDES_PROVENANCE FAIL")
                return False
    print("Z5_LEDGER_INCLUDES_PROVENANCE PASS")
    return True


def check_Z6_discharge_not_claimed(claims):
    ok = True
    for c in claims:
        text = (c.hypothesis + ' ' + c.falsifier + ' ' + (c.note or '')).lower()
        for term in CLINICAL_TERMS:
            if term in text:
                print(f"Z6_FAIL clinical term '{term}' in claim {c.claim}")
                ok = False
    print(f"Z6_DISCHARGE_NOT_CLAIMED {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    claims = claims_with_provenance()
    print("=" * 70)
    print("ZERO-PROVENANCE CLAIMS — contract")
    print("=" * 70)
    results = []
    results.append(("Z1", check_Z1_provenance_enum(claims)))
    results.append(("Z2", check_Z2_zero_claims_require_provenance(claims)))
    results.append(("Z3", check_Z3_nonzero_claims_optional(claims)))
    results.append(("Z4", check_Z4_seed_taxonomy(claims)))
    results.append(("Z5", check_Z5_ledger_includes_provenance(claims)))
    results.append(("Z6", check_Z6_discharge_not_claimed(claims)))
    print("=" * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    if passed == total:
        print(f"ZERO_PROVENANCE_CLAIMS_VERDICT Z_GREEN ({passed}/{total} clauses PASS)")
        print("ZERO_PROVENANCE_CLAIMS_NOTE provenance_optional_for_nonzero; required_for_zero; no_clinical_claim")
        return 0
    else:
        print(f"ZERO_PROVENANCE_CLAIMS_VERDICT Z_AMBER ({passed}/{total} clauses PASS)")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
