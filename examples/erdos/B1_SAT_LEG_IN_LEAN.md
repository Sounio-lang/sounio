# B1 — Internalising the SAT leg in Lean core (no Mathlib)

**Date:** 2026-05-29 (updated 2026-05-30)
**Status:** mechanism + soundness bridge **proven**; full chain **closed on K₇/6**
*and on G₅₂₉* — **χ(G₅₂₉) ≥ 5 fully machine-checked in Lean core, no Mathlib** (the
term-size wall was broken by file-loaded-style reflection, see §souc_check). The SAT
leg is then composed with the (already-discharged) geometry leg in
`formal/lean4/SounioDeGreyChi5Closed.lean`, closing **χ(QF²) ≥ 5** over the exact
field-plane ℚ(√3,√5,√7,√11) with **zero remaining hypotheses** — only the `QF↪ℝ`
isometry to Euclidean χ(ℝ²)≥5 is left (needs Mathlib's ℝ).

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

## G₅₂₉ scaling — CLOSED via `souc_check` (file-loaded-style reflection)

| instance | vars | clauses | LRAT actions | embedded-term route | **souc_check (string+parse)** |
|----------|-----:|--------:|-------------:|---------------------|-------------------------------|
| K₇/6     | 42   | 133     | 1 464        | 171 s ✅            | **0.8 s** ✅ |
| G₅₂₉ 4-col | 2 116 | 11 212 | **98 616** (31.5 MB) | ✗ (term-size wall) | **~11 s** ✅ |

souc_sat refutes G₅₂₉ in 300 218 conflicts (626 MB streamed DRAT); drat-trim
trims to 66 784 core lemmas / 98 616 LRAT lines (`s VERIFIED`).

### Why the embedded-term route fails — and how `souc_check` fixes it

Embedding a 98 616-action proof as a Lean **term** (one `Action.addRup id #[…] #[…]`
per line) and discharging `check = true` by `native_decide` is infeasible: the K₇/6
datapoint (171 s for 1 464 actions, almost all in *elaborating* the constructor
applications) extrapolates to hours, and the array-of-arrays literal blows past RAM.

`bv_decide`/`bv_check` never do this — they reflect over file/array data rather than a
term. We do the analogous thing with **zero new trust**:

1. **Embed the renumbered LRAT as a single `String` literal.** A string is an *atom*:
   the elaborator ingests 31 MB in milliseconds (vs minutes for 98 616 nested
   constructors).
2. **Parse it inside the reflective computation.** `SounioSatReflect.parseLRAT :
   String → Array IntAction` runs *under* `native_decide`, i.e. as compiled native
   code, alongside `check`. Parsing 31 MB + checking 98 616 actions: ~11 s total.
3. **The parser is unverified but soundness-irrelevant.** `check_sound` guarantees
   that *if the verified checker accepts the parsed actions* the CNF is UNSAT — no
   matter how they were produced. A parser bug can only make the proof *fail*, never
   make an unsound one succeed. (Confirmed: corrupting one SB unit flips
   `native_decide` to `false`/`sorryAx`.)

Result: `SounioSatG529.g529_unsat` (SB-augmented CNF UNSAT) and, via the WLOG leg,
`g529_not_colourable` (**unconditional χ(G₅₂₉) ≥ 5**). Axioms `[propext,
Classical.choice, Quot.sound, native_decide.ax]`, no `sorry`, no Mathlib, no external
checker as a trust anchor (drat-trim/cake_lpr only *emit* the LRAT hints).

### The WLOG leg (`SounioSatColouringSB.lean`)

souc_sat refutes the 4-colouring CNF *augmented* with three units fixing a triangle
`0,1,5` to colours `0,1,2` (a complete colour-symmetry break for k=4). To make the
bound unconditional we prove `not_colourable_of_unsat_tri`: for a pairwise-adjacent
triangle, that SB-augmented `Unsat` implies no proper 4-colouring exists — the standard
colour-permutation WLOG, made constructive (`relabel4` bijection, bijectivity decided by
exhaustion over `Fin 4`). Pure logic, axioms `[propext, Quot.sound]`, Grok math-review
**[OK]**.

### Reproduce (the `souc_check` codegen)

```bash
cd examples/erdos
./gen_lean_sat_reflect.sh souc_sat_worker.cnf /tmp/g529.lrat \
    ../../formal/lean4/SounioSatG529.lean g529 SounioSatG529 529 4 \
    data/degrey_529.edge 0 1 5        # last three = the precolour triangle
cd ../../formal/lean4 && lake env lean SounioSatG529.lean   # ~11 s, prints axioms
```

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
