#!/usr/bin/env python3
"""Witness-Based Compilation paper — bound to the rung evidence it cites.

Spec/artefact: docs/papers/witness_based_compilation_2026-07-28.md

A paper is where claims get restated far from the harness that measured them,
and prose drifts. This contract binds the draft to its evidence:

  W1_TOKENS_BOUND        every rung/token citation in the contribution table
                         matches the token its rung's spec declares
  W2_WITNESS_PINNED      the witness fingerprints quoted in the paper match
                         the claim in the bound-claims manifest, and the real
                         and perturbed fingerprints quoted are distinct
  W3_FIGURES_PINNED      the load-bearing measured figures are present
  W4_HONESTY_MARKERS     the measured/derived distinction and the limits
                         survive prose edits
  W5_PEER_REVIEW_FIXES   the 2026-07-28 peer-review fixes survive: setwise
                         stabiliser, threat model, prior-art engagement,
                         fail-closed capture-race analysis

Pure Python 3, no third-party imports.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PAPER = "docs/papers/witness_based_compilation_2026-07-28.md"
MANIFEST = "examples/epistemic/rupture_claims_verified.sio"

SPEC_TOKEN_RE = re.compile(
    r"^\*\*Status:\*\*\s*`[^`]*`\s*[—-]+\s*`([A-Za-z0-9_]+)`", re.MULTILINE)
CITE_RE = re.compile(r"\|\s*(R[0-9]+)\s*\|\s*`([A-Za-z0-9_]+)`\s*\|")

# Rungs this paper cites, and the spec each token must match.
RUNG_SPECS = {
    "R0": "docs/research/self_falsifying_compilation_line_2026-07-26.md",
    "R2": "docs/research/self_falsifying_compilation_line_r2_2026-07-26.md",
    "R15": "docs/research/self_falsifying_compilation_line_r15_2026-07-28.md",
    "R16": "docs/research/self_falsifying_compilation_line_r16_2026-07-28.md",
    "R17": "docs/research/self_falsifying_compilation_line_r17_2026-07-28.md",
}

REAL_WITNESS = \
    "705d0afdf8e830756f5d58eed9e6a11c7681d9e2e3a29ce7054ea67edc385757"
PERTURBED_WITNESS = \
    "e9f935cbab6f09fed4154847e071bd98466c55b1e0c8b75b58ff5826d5019424"


def read(rel: str) -> str:
    try:
        return (REPO / rel).read_text(errors="replace")
    except OSError:
        return ""


def spec_token(rel: str) -> str | None:
    m = SPEC_TOKEN_RE.search(read(rel))
    return m.group(1) if m else None


def clause_w1(paper: str) -> bool:
    cited = dict(CITE_RE.findall(paper))
    ok = bool(cited)
    if not cited:
        print("  no rung/token citations found in the paper")
    for rung, token in sorted(cited.items(), key=lambda kv: int(kv[0][1:])):
        spec = RUNG_SPECS.get(rung)
        declared = spec_token(spec) if spec else None
        if declared is None:
            print(f"  {rung}: spec declares no token (spec missing: {spec})")
            ok = False
        elif declared != token:
            print(f"  {rung}: paper cites {token}")
            print(f"      spec declares  {declared}  <- PAPER HAS DRIFTED")
            ok = False
        else:
            print(f"  {rung}: {token}")
    missing = sorted(set(RUNG_SPECS) - set(cited), key=lambda r: int(r[1:]))
    # R2 is cited in prose but not required in the contribution table.
    required = {"R0", "R15", "R16", "R17"}
    for rung in sorted(required - set(cited)):
        print(f"  {rung}: required by the contribution table, not cited")
        ok = False
    print(f"W1_TOKENS_BOUND {'PASS' if ok else 'FAIL'} — "
          f"{len(cited)} cited tokens checked against their specs")
    return ok


def clause_w2(paper: str, manifest: str) -> bool:
    ok = True
    declared = re.search(r'witness\s*=\s*"([0-9a-f]{64})"', manifest)
    if not declared:
        print("  manifest declares no witness — the R18 binding is gone")
        ok = False
    elif declared.group(1) != REAL_WITNESS:
        print(f"  manifest witness {declared.group(1)[:16]}… "
              f"!= recorded {REAL_WITNESS[:16]}… — the binding MOVED; "
              f"if intentional, update this contract and the paper together")
        ok = False
    for fp, why in [
        (REAL_WITNESS, "the real witness fingerprint quoted in the paper"),
        (PERTURBED_WITNESS, "the perturbed twin's fingerprint quoted"),
    ]:
        if fp[:16] not in paper:
            print(f"  MISSING {fp[:16]}… — {why}")
            ok = False
    if REAL_WITNESS == PERTURBED_WITNESS:
        print("  real and perturbed fingerprints coincide — the whole point "
              "of the paper is gone")
        ok = False
    print(f"W2_WITNESS_PINNED {'PASS' if ok else 'FAIL'} — "
          f"paper, manifest and recorded fingerprints agree and differ")
    return ok


def clause_w3(paper: str) -> bool:
    required = [
        ("126 of 128", "R15 fibre graphs changed"),
        ("3/6/12", "R17/R18 counts preserved under the flip"),
        ("3·2" , "the count law (unicode or latex form)"),
        ("CLAIM_WITNESS_MISMATCH", "the refusal outcome"),
        ("CLAIM_WITNESS_ABSENT", "the absent-witness outcome"),
        ("MODULE_CLOSURE_PASSES", "the module-closure walk that replaced the wall (R29)"),
        ("~295", "the corpus denominator"),
        ("86 s", "the n=8 exclusion reason"),
        ("30 s", "the executor per-gate budget"),
        ("3.4 s", "the production gate's wall-clock cost"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper and not (
                needle == "3·2" and "3 \\cdot 2" in paper):
            print(f"  MISSING figure {needle!r} — {why}")
            ok = False
    print(f"W3_FIGURES_PINNED {'PASS' if ok else 'FAIL'} — "
          f"{len(required)} load-bearing figures checked")
    return ok


def clause_w4(paper: str) -> bool:
    required = [
        ("Measured, not proved", "the measured/derived boundary of §2.3"),
        ("open lemma", "the equivariance lemma is stated as open"),
        ("Shared misinterpretation", "the R0 scope limit survives"),
        ("opt-in", "the mechanism is opt-in (W4 safety property)"),
        ("Single corpus", "the single-corpus limitation"),
        ("historically damaged this corpus", "the R4 honesty row"),
        ("to our knowledge", "the first-witness claim is hedged"),
        ("GAIDeT-ICMJE", "the AI disclosure"),
        ("one claim of", "the deployment honesty (1 of ~295)"),
    ]
    ok = True
    for needle, why in required:
        if needle.lower() not in paper.lower():
            print(f"  MISSING {needle!r} — {why}")
            ok = False
    print(f"W4_HONESTY_MARKERS {'PASS' if ok else 'FAIL'} — "
          f"the status distinctions and limits are still stated")
    return ok


def clause_w5(paper: str) -> bool:
    required = [
        ("setwise", "Prop 2.9 uses the setwise (not pointwise) stabiliser"),
        ("Threat model", "the threat model section (§4.4)"),
        ("Metamorphic", "metamorphic testing engagement (§6)"),
        ("go.sum", "hash-pinned fetching engagement (§6)"),
        ("fixed-output", "Nix fixed-output derivations engagement (§6)"),
        ("fail-closed", "the capture race is bounded as fail-closed (§4.3)"),
        ("witness update", "the false-positive protocol (§4.4)"),
    ]
    ok = True
    for needle, why in required:
        if needle not in paper:
            print(f"  MISSING {needle!r} — {why}")
            ok = False
    print(f"W5_PEER_REVIEW_FIXES {'PASS' if ok else 'FAIL'} — "
          f"the 2026-07-28 peer-review fixes survive prose edits")
    return ok


def main() -> int:
    print("WITNESS-BASED COMPILATION PAPER — bound to the rung evidence it cites")
    print("=" * 74)

    paper = read(PAPER)
    if not paper:
        print(f"W1_TOKENS_BOUND FAIL — paper missing: {PAPER}")
        print("WITNESS_PAPER_VERDICT INCOMPLETE")
        return 1
    manifest = read(MANIFEST)
    if not manifest:
        print(f"W2_WITNESS_PINNED FAIL — manifest missing: {MANIFEST}")
        print("WITNESS_PAPER_VERDICT INCOMPLETE")
        return 1

    w1 = clause_w1(paper)
    print()
    w2 = clause_w2(paper, manifest)
    print()
    w3 = clause_w3(paper)
    print()
    w4 = clause_w4(paper)
    print()
    w5 = clause_w5(paper)
    print()

    print("=" * 74)
    if not (w1 and w2 and w3 and w4 and w5):
        print("WITNESS_PAPER_VERDICT INCOMPLETE")
        return 1
    print("WITNESS_PAPER_VERDICT DRAFT_TOKEN_BOUND__WITNESS_PINNED__LIMITS_STATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
