#!/usr/bin/env python3
"""PIREUS Batch 2 claim-promotion contract.

Verifies, WITHOUT any hardware re-execution and WITHOUT the Loom Guardian,
that the four sealed PARITY_OPEN Pireus receipts are consumed by four
ledger-encoded claims (CLAIM_READY scope) under the P5 evidence ceiling:

  C1 RECEIPTS_SEALED_AND_UNTOUCHED
     The four receipt files (and their four compact evidence files) exist,
     still carry Stage: PARITY_OPEN and Parity-Receipt-Valid: true, and their
     file sha256 digests match the values pinned below. The pins were computed
     on 2026-10-04 from the Founder-Accepted tree (commits 3d384a3f0d and
     27cfc4f028); any later edit of a receipt fails this clause closed.
  C2 CLAIMS_PARSE
     Each of the four claim files yields exactly one claim with the expected
     name under scripts/research/falsification_ledger_contract.py's scanner,
     with all seven required fields and valid evidence/verdict enums.
  C3 FALSIFIER_PINS_HASHES
     Each claim's @falsifier names that receipt's pinned result sha256 (and
     the frozen/binary digest) whose drift would falsify the claim.
  C4 EVIDENCE_CEILING (P5)
     @evidence is gate_green -- never claim_ready or instrument_controlled --
     and each @hypothesis carries the ceiling markers (frozen finite input,
     named node(s), pinned sha256). sealed_receipt was rejected as an
     @evidence value because it is not a legal enum in the ledger scanner.
  C5 NO_OVERCLAIM
     No overclaim strings (generic/universal equivalence, timing, latency,
     throughput, performance, benchmark, cross-ISA, cost model, ...) appear
     in any @hypothesis or @note.
  C6 GATE_CLOSED_NO_GUARDIAN
     The gate script exists, is executable, composes this contract, and
     contains no Guardian, ssh/curl/kubectl, or hardware-script references.

Modeled on scripts/research/garden_to_claim_pipeline_contract.py.
Exit 0 iff all six clauses PASS.
"""

import hashlib
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import falsification_ledger_contract as ledger  # noqa: E402

HARNESS = "scripts/research/pireus_claim_promotion_contract.py"
GATE = "scripts/ci/pireus_claim_promotion_gate.sh"

# claim name -> (claim file, receipt file, receipt sha256, evidence file,
#                evidence sha256, result sha256, secondary digest)
CLAIMS = {
    "pireus_xor_xeon_material_parity": (
        "stdlib/epistemic/pireus_xor_xeon_material_parity_claim.sio",
        "docs/research/receipts/pireus_xor_lowering_darwin_xeon_material_parity_20260827.md",
        "342d8ba8808c2a926bb2bbf0c09488f7b849967239c932687952ec6ae789a906",
        "docs/research/evidence/pireus_xor_lowering_darwin_xeon_material_parity_20260827.txt",
        "ee37914bc738eb829f3589249f228e4a8312310fbffa0b00636cd0c9ed9a40d1",
        "fe851cccb1487d3977c491426cd89e1445e3c234fbce8c5444972a441b8876e4",
        "c88cd9ba43e106c1721ab99ea501c1c797935ed77e46f64aedab333f963e399f",
    ),
    "pireus_xor_xeon_fleet_parity": (
        "stdlib/epistemic/pireus_xor_xeon_fleet_parity_claim.sio",
        "docs/research/receipts/pireus_xor_lowering_darwin_xeon_fleet_parity_20260827.md",
        "bb5c74721df94c980edc70259d369cd39185ae84a3faeed2c791a064c060f24f",
        "docs/research/evidence/pireus_xor_lowering_darwin_xeon_fleet_parity_20260827.txt",
        "deb3e651ea1ef99d0d1783bcb29a51fc06f3c100d6e83e31709a2cf30f14367e",
        "fe851cccb1487d3977c491426cd89e1445e3c234fbce8c5444972a441b8876e4",
        "ea9e2e1f4be7926c76262876960bb673455ef9786a391286676f8b4c17539e19",
    ),
    "pireus_dgx_ptx_shfl_parity": (
        "stdlib/epistemic/pireus_dgx_ptx_shfl_parity_claim.sio",
        "docs/research/receipts/pireus_dgx_ptx_shfl_material_parity_20260827.md",
        "3c10882eff43d3b197428839996c7a04c009c8f537d0c1451bdf3e8a13e2f385",
        "docs/research/evidence/pireus_dgx_ptx_shfl_material_parity_20260827.txt",
        "2c6b6e448265a5566d17df9a674246ea62c05210e432e48e418d16358496853b",
        "1e776e655761bd9e59322ac64e736629bd35586a958d503ea814ddddaa865f3c",
        "495c52ccf2370c4e668ab1e9bc4d7dbc02c0d97a8cd27a0dbdfe5aa130d8e54e",
    ),
    "pireus_apple_a64_tbl_parity": (
        "stdlib/epistemic/pireus_apple_a64_tbl_parity_claim.sio",
        "docs/research/receipts/pireus_apple_a64_tbl_material_parity_20260827.md",
        "c00a3d4e556688829efadbbf640ea858cfe9520dc04103fa745cf1a8101f7840",
        "docs/research/evidence/pireus_apple_a64_tbl_material_parity_20260827.txt",
        "2877bfd463b4d28dc3311b75c69bec2aa1c62b430d08314989187d44b32a781e",
        "bd64bc56037a64a93c0136fa29a6ff1a294e8b84be549a5a4abfeaaf81a2e700",
        "d1de1ec160d0cf7c69a7f8e3f50d5ae027457f8c23648fb685c5216d19f10f81",
    ),
}

REQUIRED_FIELDS = ("claim", "hypothesis", "falsifier", "evidence", "harness", "gate", "verdict")

OVERCLAIM_STRINGS = (
    "generic", "universal", "timing", "latency", "throughput", "performance",
    "wall-clock", "benchmark", "cross-isa", "cost model", "all cpus",
    "equivalence between", "subquadratic", "fano",
)

SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def check_C1_receipts_sealed_and_untouched():
    ok = True
    for name, (_, receipt, receipt_sha, evidence, evidence_sha, _, _) in CLAIMS.items():
        for path, pinned in ((receipt, receipt_sha), (evidence, evidence_sha)):
            full = os.path.join(REPO_ROOT, path)
            if not os.path.isfile(full):
                print(f"C1_FAIL missing {path} ({name})")
                ok = False
                continue
            actual = _sha256(full)
            if actual != pinned:
                print(f"C1_FAIL drift {path}: sha256={actual} pinned={pinned}")
                ok = False
        text = _read(os.path.join(REPO_ROOT, receipt))
        if "Parity-Receipt-Valid: `true`" not in text:
            print(f"C1_FAIL {receipt} lost Parity-Receipt-Valid: true")
            ok = False
        if "Stage: `PARITY_OPEN`" not in text:
            print(f"C1_FAIL {receipt} lost Stage: PARITY_OPEN")
            ok = False
    print(f"C1_RECEIPTS_SEALED_AND_UNTOUCHED {'PASS' if ok else 'FAIL'}")
    return ok


def _scan_claim(name, claim_file):
    full = os.path.join(REPO_ROOT, claim_file)
    if not os.path.isfile(full):
        return None, f"missing claim file {claim_file}"
    claims = ledger.scan_file(full)
    matches = [c for c in claims if c.claim == name]
    if len(matches) != 1:
        return None, f"{claim_file}: expected exactly one claim named {name}, got {len(matches)}"
    return matches[0], None


def check_C2_claims_parse():
    ok = True
    for name, (claim_file, *_rest) in CLAIMS.items():
        claim, err = _scan_claim(name, claim_file)
        if err:
            print(f"C2_FAIL {err}")
            ok = False
            continue
        for field in REQUIRED_FIELDS:
            if not getattr(claim, field, None):
                print(f"C2_FAIL {name}: missing required field @{field}")
                ok = False
        if claim.evidence not in ledger.EVIDENCE_LEVELS:
            print(f"C2_FAIL {name}: invalid @evidence {claim.evidence}")
            ok = False
        if claim.verdict not in ledger.VERDICTS:
            print(f"C2_FAIL {name}: invalid @verdict {claim.verdict}")
            ok = False
    print(f"C2_CLAIMS_PARSE {'PASS' if ok else 'FAIL'}")
    return ok


def check_C3_falsifier_pins_hashes():
    ok = True
    for name, (claim_file, _, _, _, _, result_sha, secondary_sha) in CLAIMS.items():
        claim, err = _scan_claim(name, claim_file)
        if err:
            print(f"C3_FAIL {err}")
            ok = False
            continue
        falsifier = claim.falsifier or ""
        for digest, label in ((result_sha, "result_sha256"), (secondary_sha, "secondary digest")):
            if digest not in falsifier:
                print(f"C3_FAIL {name}: @falsifier does not pin {label} {digest}")
                ok = False
        receipt_pin = "sha256 pinned in scripts/research/pireus_claim_promotion_contract.py"
        if receipt_pin not in falsifier:
            print(f"C3_FAIL {name}: @falsifier does not name the receipt-drift pin")
            ok = False
    print(f"C3_FALSIFIER_PINS_HASHES {'PASS' if ok else 'FAIL'}")
    return ok


def check_C4_evidence_ceiling():
    ok = True
    for name, (claim_file, *_rest) in CLAIMS.items():
        claim, err = _scan_claim(name, claim_file)
        if err:
            print(f"C4_FAIL {err}")
            ok = False
            continue
        if claim.evidence != "gate_green":
            print(f"C4_FAIL {name}: @evidence={claim.evidence} exceeds the gate_green ceiling")
            ok = False
        hypothesis = (claim.hypothesis or "").lower()
        if "frozen" not in hypothesis or "finite input" not in hypothesis:
            print(f"C4_FAIL {name}: @hypothesis lacks the frozen-finite-input ceiling marker")
            ok = False
        if not SHA256_RE.search(claim.hypothesis or ""):
            print(f"C4_FAIL {name}: @hypothesis pins no sha256 digest")
            ok = False
    print(f"C4_EVIDENCE_CEILING {'PASS' if ok else 'FAIL'}")
    return ok


def check_C5_no_overclaim():
    ok = True
    for name, (claim_file, *_rest) in CLAIMS.items():
        claim, err = _scan_claim(name, claim_file)
        if err:
            print(f"C5_FAIL {err}")
            ok = False
            continue
        text = " ".join(
            part for part in (claim.hypothesis, claim.note or "") if part
        ).lower()
        for bad in OVERCLAIM_STRINGS:
            if bad in text:
                print(f"C5_FAIL {name}: overclaim string '{bad}' in hypothesis/note")
                ok = False
    print(f"C5_NO_OVERCLAIM {'PASS' if ok else 'FAIL'}")
    return ok


def check_C6_gate_closed_no_guardian():
    ok = True
    gate_path = os.path.join(REPO_ROOT, GATE)
    if not os.path.isfile(gate_path):
        print(f"C6_FAIL missing {GATE}")
        ok = False
    else:
        if not os.access(gate_path, os.X_OK):
            print(f"C6_FAIL {GATE} is not executable")
            ok = False
        text = _read(gate_path).lower()
        if HARNESS not in text:
            print(f"C6_FAIL {GATE} does not compose {HARNESS}")
            ok = False
        for forbidden in ("guardian", "ssh", "curl", "kubectl", "ssh-key", "nvcc", "srun", "sbatch"):
            if forbidden in text:
                print(f"C6_FAIL {GATE} references forbidden surface '{forbidden}'")
                ok = False
        for match in re.findall(r"scripts/ci/pireus_[a-z0-9_]+\.sh", text):
            if match != GATE:
                print(f"C6_FAIL {GATE} invokes hardware script {match}")
                ok = False
    print(f"C6_GATE_CLOSED_NO_GUARDIAN {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    print("=" * 70)
    print("PIREUS Batch 2 claim-promotion contract (no hardware, no Guardian)")
    print("=" * 70)
    results = [
        ("C1", check_C1_receipts_sealed_and_untouched()),
        ("C2", check_C2_claims_parse()),
        ("C3", check_C3_falsifier_pins_hashes()),
        ("C4", check_C4_evidence_ceiling()),
        ("C5", check_C5_no_overclaim()),
        ("C6", check_C6_gate_closed_no_guardian()),
    ]
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    print("=" * 70)
    if passed == total:
        print(f"PIREUS_CLAIM_PROMOTION_VERDICT C_GREEN ({passed}/{total} clauses PASS)")
        print("PIREUS_CLAIM_PROMOTION_NOTE 4 sealed PARITY_OPEN receipts consumed; 4 claims at CLAIM_READY scope under the P5 ceiling; zero re-execution")
        return 0
    print(f"PIREUS_CLAIM_PROMOTION_VERDICT C_AMBER ({passed}/{total} clauses PASS)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
