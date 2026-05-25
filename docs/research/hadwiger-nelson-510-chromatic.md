# HeuleGraph510 — chromatic number = 5 (Phase C)

**Result:** χ(HeuleGraph510) = 5, on the Phase-B-certified unit-distance graph
(510 vertices, 2504 edges, exact in ℚ(√3,√5,√11)).

**This is NOT "χ=5 kernel-checked."** The two bounds rest on different trust bases,
and saying otherwise would be an overclaim:

- **χ ≤ 5 — Lean kernel.** `formal/lean4/SounioHeule510Chromatic.lean` exhibits an
  explicit proper 5-colouring of `SounioHeule510.E` and proves by `native_decide` that
  every edge is bichromatic and only colours 0..4 are used. Fully kernel-checked.
- **χ ≥ 5 — SAT + drat-trim (community standard, not the kernel).** "Not 4-colourable"
  is an UNSAT statement; `native_decide` cannot decide it (4^510 colourings). The
  4-colouring CNF — derived *directly from the Lean edge list* `SounioHeule510.E` — is
  UNSAT (Glucose 3), and the 2.2M-line DRAT proof is checked `s VERIFIED` by `drat-trim`.
  Trust base: the SAT solver + drat-trim. **Not** the Lean kernel.

Bundle + reproduction: `scripts/research/heule510_chromatic_cert/`.

## What this is, and isn't
- **Is:** a verified UNSAT certificate at the de-facto standard every Heule/de Grey
  result ships at, chained to a graph whose *geometry* is certified to the Lean kernel
  (Phase B). Both halves of "χ = 5" are machine-checked; one in the kernel, one by
  drat-trim.
- **Is not:** a single-trust-base kernel proof of χ = 5 (that needs Phase C.1: a
  formally-verified LRAT checker on `heule510_4col.lrat`). And it is **not** progress on
  the open Hadwiger–Nelson question — χ(plane) ≥ 5 has been known since de Grey (2018);
  HeuleGraph510 is the smallest known witness. χ ≥ 6 (Phase D) remains open and may be
  false.

## Phase status
- Phase A — exact certifier on the spindle (χ=4). ✓
- Phase B — HeuleGraph510 certified exact unit-distance in ℚ(√3,√5,√11), kernel. ✓
- **Phase C — χ = 5: χ≤5 kernel, χ≥5 drat-trim-VERIFIED. ✓ (this note)**
- Phase C.1 — verified LRAT checker to lift χ≥5 to kernel grade. (open)
- Phase D — search for a 6-chromatic graph (χ(plane) ≥ 6). (open, long-odds)
