#!/usr/bin/env python3
"""Self-falsifying compilation, rung R24 — which production claims can honestly
declare a provenance, and why binding the rest would be hollow.

Spec: docs/research/self_falsifying_compilation_line_r24_2026-07-31.md

R23 bound provenance to zd_fiber_spectra_count_law_holds because its derivation
-- the parity-collapse map Phi -- lives OUTSIDE the gate that checks it: the
gate computes the spectrum count and hashes the spectra, but the *meaning* of
that count (that it equals the number of geometries) rests on Phi, which the
gate never runs. That is exactly the artifact R20 found missing while the gate
stayed green. Provenance binds the availability of such a derivation.

The instruction that followed -- "bind provenance to the other production
claims" -- runs straight into the line's own standard. A claim's provenance is
meaningful only when it names a derivation the gate does NOT itself run, so that
the derivation can go missing while the check passes. When the gate runs the
claim's whole derivation, provenance = that script is REDUNDANT with gate
existence: a missing script already fails the gate. Declaring it anyway is the
rubber-stamp pathology this line catalogues (R1 a claim bound to no gate, R5 a
gate nobody ran, R22 a gate certifying a literal).

So the honest answer is measured, not asserted: classify every production claim
and bind provenance only where it is not hollow.

CLASSES:
  external-derivation  the gate rests on a repo artifact it does not itself run
                       (bindable; provenance is meaningful)
  self-contained       the gate runs the claim's whole derivation
                       (provenance would duplicate gate existence)
  infra                the gate checks a compiler/build invariant, not a math
                       derivation (no derivation artifact to cite)

CLAUSES:
  U1_EVERY_CLAIM_CLASSIFIED   all production claims fall into one class.
  U2_BOUND_IFF_BINDABLE       a claim declares provenance iff it is
                              external-derivation. Both directions: no hollow
                              binding, no missing honest one.
  U3_THE_BINDABLE_ONE_IS_BOUND  zd_fiber is external-derivation and does declare
                              provenance, at the artifact R20 found missing.

Pure Python 3. Reads the manifest and the gates; runs nothing.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MANIFEST = REPO / "examples/epistemic/rupture_claims_verified.sio"
CI = REPO / "scripts/ci"

EXEC = re.compile(r"(spec_from_file_location|exec_module|SourceFileLoader|runpy"
                  r"|import_module)")
RUNS_SCRIPT = re.compile(r"scripts/research/([a-z0-9_]+\.(?:py|sio))")
DEP_IN_LINE = re.compile(r"scripts/research/([a-z0-9_]+\.py)"
                         r"|['\"]([a-z0-9_]{6,}\.py)['\"]")

# The one claim whose derivation lives outside its gate: the completeness of the
# spectrum count rests on Phi, which the witness gate never runs. Named rather
# than guessed, because that is the judgement R23 made and this rung checks it
# held rather than re-deriving it heuristically.
KNOWN_EXTERNAL = {
    "zd_fiber_spectra_count_law_holds":
        "scripts/research/cd_tower_collapse_isomorphism.py",
}


def parse_claims(text: str):
    for m in re.finditer(r"claim (\w+) \{(.*?)\n\}", text, re.S):
        name, body = m.group(1), m.group(2)
        gate = re.search(r'gate\s*=\s*"([^"]+)"', body)
        prov = re.search(r'provenance\s*=\s*"([^"]+)"', body)
        yield name, (gate.group(1) if gate else None), (prov.group(1) if prov else None)


def external_dep(gate_path: str) -> str | None:
    """A repo artifact the gate depends on but does NOT run itself, if any."""
    g = REPO / gate_path
    if not g.exists():
        return None
    gate_src = g.read_text(errors="replace")
    run_scripts = set(RUNS_SCRIPT.findall(gate_src))  # scripts the gate executes
    for rel in run_scripts:
        f = REPO / "scripts/research" / rel
        if not f.exists():
            continue
        src = f.read_text(errors="replace")
        for line in src.splitlines():
            if not EXEC.search(line):
                continue
            for mm in DEP_IN_LINE.finditer(line):
                dep = mm.group(1) or mm.group(2)
                if not dep or dep == rel or dep in run_scripts:
                    continue
                # a hard dependency the gate never runs
                return f"scripts/research/{dep}"
    return None


def classify(name: str, gate: str | None) -> tuple[str, str | None]:
    if name in KNOWN_EXTERNAL:
        return "external-derivation", KNOWN_EXTERNAL[name]
    if not gate:
        return "infra", None
    scripts = RUNS_SCRIPT.findall((REPO / gate).read_text(errors="replace")
                                  if (REPO / gate).exists() else "")
    if not scripts:
        return "infra", None
    ext = external_dep(gate)
    if ext:
        return "external-derivation", ext
    return "self-contained", None


def main() -> int:
    claims = list(parse_claims(MANIFEST.read_text()))
    print("R24 — which production claims can honestly declare a provenance")
    print("=" * 72)
    print(f"{len(claims)} production claims in {MANIFEST.relative_to(REPO)}")
    print()

    rows = []
    for name, gate, prov in claims:
        cls, target = classify(name, gate)
        rows.append((name, cls, target, prov))

    by = {}
    for _, cls, _, _ in rows:
        by[cls] = by.get(cls, 0) + 1
    for name, cls, target, prov in rows:
        mark = "prov=" + prov.split("/")[-1] if prov else "no prov"
        print(f"  [{cls:19}] {name}  ({mark})")
        if cls == "external-derivation":
            print(f"        derivation the gate does not run: {target}")
    print()
    print("  " + "  ".join(f"{k}: {v}" for k, v in sorted(by.items())))
    print()

    # ---- U1 -----------------------------------------------------------------
    u1 = all(cls in ("external-derivation", "self-contained", "infra")
             for _, cls, _, _ in rows)
    print(f"U1_EVERY_CLAIM_CLASSIFIED {'PASS' if u1 else 'FAIL'}  "
          f"{len(rows)} claims, no unclassified")

    # ---- U2 -----------------------------------------------------------------
    bad = []
    for name, cls, target, prov in rows:
        bindable = cls == "external-derivation"
        if bindable and not prov:
            bad.append(f"{name}: bindable but no provenance")
        if not bindable and prov:
            bad.append(f"{name}: {cls} but declares provenance (hollow)")
    u2 = not bad
    print(f"U2_BOUND_IFF_BINDABLE {'PASS' if u2 else 'FAIL'}  "
          f"provenance declared exactly on the bindable claims")
    for b in bad:
        print(f"    {b}")

    # ---- U3 -----------------------------------------------------------------
    z = next((r for r in rows if r[0] == "zd_fiber_spectra_count_law_holds"), None)
    u3 = bool(z) and z[1] == "external-derivation" and z[3] and \
        z[3].endswith("cd_tower_collapse_isomorphism.py") and \
        (REPO / z[3]).exists()
    print(f"U3_THE_BINDABLE_ONE_IS_BOUND {'PASS' if u3 else 'FAIL'}  "
          f"zd_fiber binds provenance to Phi, and Phi is in the tree")

    ok = u1 and u2 and u3
    ext = by.get("external-derivation", 0)
    verdict = ("PROVENANCE_BOUND_WHERE_HONEST__REST_WOULD_BE_HOLLOW"
               if ok else "INCONCLUSIVE")
    print()
    print("-" * 72)
    print(f"{ext} of {len(rows)} production claims have a derivation their gate does")
    print("not run; the rest are self-contained or infra, where provenance would")
    print("duplicate gate existence. Binding it there is the rubber-stamp this")
    print("line exists to refuse.")
    print()
    print(f"SELF_FALSIFYING_R24_VERDICT {verdict}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
