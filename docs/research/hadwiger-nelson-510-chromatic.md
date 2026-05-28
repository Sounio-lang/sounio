# HeuleGraph510 — chromatic number = 5 (Phase C)

**Result:** χ(HeuleGraph510) = 5, on the Phase-B-certified unit-distance graph
(510 vertices, 2504 edges, exact in ℚ(√3,√5,√11)).

**This is NOT "χ=5 kernel-checked."** The two bounds rest on different trust bases,
and saying otherwise would be an overclaim:

- **χ ≤ 5 — Lean kernel.** `formal/lean4/SounioHeule510Chromatic.lean` exhibits an
  explicit proper 5-colouring of `SounioHeule510.E` and proves by `native_decide` that
  every edge is bichromatic and only colours 0..4 are used. Fully kernel-checked.
- **χ ≥ 5 — two independent certifications:**
  - *Phase C — SAT + drat-trim (trusted C).* "Not 4-colourable" is UNSAT; `native_decide`
    cannot decide it (4^510 colourings). The 4-colouring CNF — derived *directly from the
    Lean edge list* `SounioHeule510.E` — is UNSAT (Glucose 3); the 2.2M-line DRAT is checked
    `s VERIFIED` by `drat-trim`.
  - *Phase C.1 — `bv_decide` (Lean-verified checker).* `formal/lean4/SounioHeule510NotColorable.lean`
    proves the same non-4-colourability through Lean's `bv_decide`: a 2-bits-per-vertex
    encoding over the same `E`, bitblasted, solved by bundled CaDiCaL (untrusted — its output
    is *checked*), and validated by an **LRAT checker formally verified in Lean**. Trust base
    `[propext, Classical.choice, Quot.sound, <bv_decide reflection axiom>]` — the *same kind*
    of compiled-reflection trust as `native_decide` (which the χ≤5 half and all of Phases A–B
    already use). **Not** a zero-axiom kernel proof, but it **removes** the external trusted-C
    drat-trim (and the foreign HOL4 kernel of cake_lpr): both bounds of χ=5 now rest on the
    *same* trust the project already carries. ~9.4 min to elaborate.

**Net (post-C.1):** the lopsided trust base is resolved. χ ≤ 5 (`native_decide`) and χ ≥ 5
(`bv_decide`, verified checker) now share one trust model; drat-trim's `s VERIFIED` stands
as independent corroboration.

Bundle + reproduction: `scripts/research/heule510_chromatic_cert/`.

## What this is, and isn't
- **Is:** a verified UNSAT certificate at the de-facto standard every Heule/de Grey
  result ships at, chained to a graph whose *geometry* is certified to the Lean kernel
  (Phase B). Both halves of "χ = 5" are machine-checked; one in the kernel, one by
  drat-trim.
- **Is not:** a *zero-axiom* kernel proof of χ = 5 — both halves use compiled reflection
  (`native_decide` / `bv_decide`), which carry a reflection axiom. (A fully axiom-free
  proof would require kernel-reducing the LRAT, infeasible at this scale.) And it is **not**
  progress on the open Hadwiger–Nelson question — χ(plane) ≥ 5 has been known since de Grey
  (2018); HeuleGraph510 is the smallest known witness. χ ≥ 6 (Phase D) remains open and may
  be false.

## Phase status
- Phase A — exact certifier on the spindle (χ=4). ✓
- Phase B — HeuleGraph510 certified exact unit-distance in ℚ(√3,√5,√11), kernel. ✓
- **Phase C — χ = 5: χ≤5 kernel, χ≥5 drat-trim-VERIFIED. ✓ (this note)**
- Phase C.1 — verified LRAT checker to lift χ≥5 to kernel grade. (open)
- Phase D — search for a 6-chromatic graph (χ(plane) ≥ 6). (open, long-odds)
