# B1 — Internalising the SAT leg in Lean core (no Mathlib)

**Date:** 2026-05-29
**Status:** mechanism + soundness bridge **proven**; full chain **closed on K₇/6**;
G₅₂₉ blocked on `native_decide` term-size (path identified).

## What this is

The χ(ℝ²) ≥ 5 lower bound has two legs: *geometry* (G₅₂₉ is unit-distance) and
*SAT* (G₅₂₉ is not 4-colourable). Until now the SAT leg was discharged by the
external CakeML checker `cake_lpr` (`CAKE_LPR_RESULT.md`) and entered the Lean
reduction (`SounioDeGreyChi5.lean`) as an explicit hypothesis.

**B1 removes the external checker** for the SAT leg: a souc_sat (Sounio's own
CDCL) UNSAT certificate is re-checked *inside Lean* by Lean core's **formally
verified LRAT checker** (`Std.Tactic.BVDecide.LRAT.check` / `check_sound`), with
**no Mathlib** and no dependency outside the Lean toolchain. drat-trim/cake_lpr
are now only used to *produce* the LRAT hints; the trust anchor is Lean's
verified checker plus the standard `native_decide` reflection axiom (the same one
`bv_decide` relies on).

## Artifacts

| File | Role | Axioms |
|------|------|--------|
| `formal/lean4/SounioSatCheckSpike.lean` | minimal mechanism demo (hand LRAT) | `[propext, Classical.choice, Quot.sound, native_decide.ax]` |
| `formal/lean4/SounioSatColouringBridge.lean` | encoding-soundness bridge: `colourCNF.Unsat → ¬ proper k-colouring` | **`[propext, Quot.sound]`** (pure logic, no `native_decide`) |
| `formal/lean4/SounioSatK76.lean` | **full chain** on K₇/6 (autogen) | `[propext, Classical.choice, Quot.sound, native_decide.ax]` |
| `examples/erdos/gen_lean_sat.sh` | codegen: souc_sat CNF+LRAT → Lean chromatic theorem | — |

## The full chain (K₇/6)

```
souc_sat (Sounio CDCL)
  └─ DIMACS CNF + streamed DRAT
       └─ drat-trim -L  ──►  LRAT hints (renumbered to contiguous ids)
            └─ Lean: check_sound k76_prf (colourCNF 7 6 k76_edges)  [native_decide]
                 └─ k76_cnf.Unsat
                      └─ SounioSatColouring.not_colourable_of_unsat   [pure logic]
                           └─ k76_not_colourable : ¬ ∃ proper 6-colouring of K₇   (χ(K₇) ≥ 7)
```

`k76_not_colourable` depends only on `[propext, Classical.choice, Quot.sound,
native_decide.ax]` — no `sorry`, no Mathlib.

### Two engineering subtleties (both resolved)

1. **Variable mapping.** `CNF.convertLRAT` relabels via `relabelFin` (identity on
   variable indices) then `+1` (DIMACS is 1-based). So CNF Nat variable `v` ↔
   DIMACS literal `v+1`; LRAT ints are kept verbatim. The colouring variable
   `v*k + c` ("vertex v has colour c") matches souc_sat's emitter exactly.

2. **Contiguous clause ids.** Lean's `DefaultFormula.insert` *pushes* each added
   clause at the end of its array, so the verified checker requires LRAT addition
   ids to be **contiguous** (the k-th addition must land at index M+k).
   drat-trim's LRAT ids have gaps (57 gaps for K₇/6). `gen_lean_sat.sh` renumbers
   additions to M+1, M+2, … and remaps every hint/deletion reference. This was
   the single cause of the first (silent) `check = false`.

The CNF is expressed as `colourCNF n k edges` (the canonical encoding); it is
clause-for-clause identical (same clause order; intra-clause literal order is
irrelevant to the set-based checker) to souc_sat's DIMACS, verified by `diff`.
This lets the **generic, scale-independent** bridge apply directly.

## G₅₂₉ scaling — honest status

| instance | vars | clauses | LRAT lines | `native_decide` |
|----------|-----:|--------:|-----------:|-----------------|
| K₇/6     | 42   | 133     | 1 464      | 171 s ✅ |
| G₅₂₉ 4-col | 2 116 | 11 212 | **98 616** (31.5 MB; 66 784 core lemmas) | ✗ (see below) |

souc_sat refutes G₅₂₉ in 300 218 conflicts (66 MB streamed DRAT); drat-trim
trims to 66 784 core lemmas / 98 616 LRAT lines (`s VERIFIED`).

Embedding a 98 616-action proof as a Lean **term** and discharging
`check = true` by `native_decide` is **not feasible in the workspace**: the K₇/6
datapoint (171 s for 1 464 actions) extrapolates to multiple hours of compilation
and a ~31 MB array-of-arrays literal whose elaboration/native compilation exceeds
available RAM. This is a *term-size* wall, not a soundness or logic gap — the
bridge already proves the reduction for any `n, k, edges`.

### Path to closing G₅₂₉ in Lean (future sprint)

1. **File-loaded reflection** (the real fix). Lean's `bv_decide`/`bv_check`
   never embed the certificate as a term: they read the LRAT at elaboration and
   reflect via `Lean.ofReduceBool` over file-loaded data. A `souc_check`-style
   term/tactic that loads `g529.lrat` from disk and calls `check` the same way
   would sidestep the term-size wall entirely. This is the recommended next step.
2. **Cluster brute force.** Build `SounioSatG529` on the HPC node (≫ workspace
   RAM, hours of compile budget) — closes it without new code, but is not
   reproducible on a laptop.
3. **Further proof minimisation** (`lrat-trim`) — helps constant-factor only
   (~67× gap remains), insufficient alone.

## Reproduce

```bash
cd examples/erdos
SOUNIO_STDLIB_PATH=$PWD/../../stdlib ../../bin/souc souc_sat.sio /tmp/souc_sat.elf
/tmp/souc_sat.elf                                   # writes souc_sat_k76.cnf/.drat
/tmp/dtrim souc_sat_k76.cnf souc_sat_k76.drat -L /tmp/k76.lrat
./gen_lean_sat.sh souc_sat_k76.cnf /tmp/k76.lrat \
    ../../formal/lean4/SounioSatK76.lean k76 SounioSatK76 7 6
cd ../../formal/lean4
lake build SounioSatColouringBridge && lake build SounioSatK76   # ~3 min
```
