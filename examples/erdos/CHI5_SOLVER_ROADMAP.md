# Roadmap: a Sounio SAT solver that surpasses kissat

> Status: planning. Author target — single author, the same one who wrote the
> self-hosted Sounio compiler in < 90 days. This document is deliberately ambitious
> *and* concrete: every phase ends in a runnable artefact and an external
> verification gate. No milestone is "claimed" until a machine check passes.

---

## 0. Honest framing — what "surpass kissat" means

kissat (Armin Biere) is ~20 years of CDCL research distilled into ~50k lines of
hand-tuned C and a multi-year SAT-Competition winning streak. Beating it **in the
general SAT-Competition aggregate** in one pass is not a realistic first target, and
saying otherwise would be drift (CLAUDE.md §6.1, §6.7).

So we define *three fronts* where Sounio can genuinely win, ordered by how decisively
"win" can be defended:

| Front | What "surpass" means | Why Sounio can win |
|---|---|---|
| **F1 — Verified performance** | The **fastest *formally-verified* SAT solver**: a competitive CDCL whose core is proven correct in Lean 4 (the repo already has `formal/lean4/`). kissat is *not* verified; verified solvers (e.g. versioned IsaSAT) are far slower. | Sounio has effects, refinement types, and a Lean pipeline. A solver that is *both* fast *and* end-to-end verified is a new point on the Pareto front — a CAV/POPL contribution, not a benchmark footnote. |
| **F2 — Domain dominance** | Beat kissat **wall-clock on the geometric / unit-distance / graph-colouring family** (de Grey, Heule cores, Ramsey-type, Hadwiger–Nelson). Specialisation beats generality on a fixed family. | We control the *encoding* and can fuse exact-arithmetic structure (symmetry, geometry) the way a black-box solver cannot. |
| **F3 — Proof-native throughput** | Emit + check proofs **faster and smaller** than the DRAT/LRAT toolchain kissat feeds (binary LRAT with hints, GPU-checked). | Sounio owns the whole stack (solver → proof → checker → Lean), so we can co-design the proof format and the checker. |

The de Grey χ ≥ 5 certificate is the **flagship demo** of F1+F2+F3 together: exact
rational geometry → CNF → UNSAT → machine-checked proof → Lean theorem. Closing it is
Phase 4; the solver work that gets us there is Phases 1–3.

---

## 1. Where we are (measured, this repo)

`examples/erdos/cdcl_fast.sio` is a from-scratch CDCL with:

- two-watched-literal propagation, integer VSIDS + decay, phase saving;
- inner/outer (MiniSat-style) restarts;
- **LBD clause deletion** with DRUP `d` records — externally **`drat-trim` verified**
  (K₇ proof with 1136 deletions; non-vacuity rejects a corrupted lemma);
- in-memory DRUP buffer + an independent native RUP checker.

Measured ceiling: on de Grey's real 1581-vtx 4-colouring it sustains ~2700
conflicts/s at **bounded memory** (40k–230k live clauses) for 8.6 M conflicts without
closing UNSAT; the in-memory proof buffer fills at ~7.5 M conflicts.

**Gap to kissat** is therefore *search quality* and *engineering*, not data or memory.
The missing pieces below are exactly the deltas between a textbook CDCL and a modern
one.

---

## 1a. FLAGSHIP RESULT — χ(G₅₂₉) ≥ 5 certified by the Sounio solver (2026-05-29)

The self-hosted Sounio solver has closed the **real Heule 529-vertex de Grey core**:

- `souc_sat.sio` reads `examples/erdos/data/degrey_529.edge` (Heule's G₅₂₉,
  529 vertices / 2670 edges, from `github.com/marijnheule/CNP-SAT`) in a new
  **DIMACS edge-file mode**, builds the 4-colouring CNF (529 at-least-one +
  2670×4 edge clauses = 11 209) plus **one sound triangle-precolour** (3 units,
  triangle 0,1,5 → colours 0,1,2; satisfiability-preserving), and **refutes it**.
- Result: **UNSAT in ≈33 s, 327 208 conflicts**, a **72 MB streamed DRAT proof**,
  and **`drat-trim s VERIFIED`** (9 776 / 11 212 clauses in core, 5 010 369
  resolution steps). ⟹ **χ(G₅₂₉) ≥ 5** — the graph is not 4-colourable.
- The triangle precolour is essential: **without it our CDCL does not close in
  300 s** (>500 k conflicts and climbing); with it (complete S₄ colour-symmetry
  break, since unit-distance ⟹ K₄-free ⟹ ω=3) it finishes in 33 s.

```bash
export SOUNIO_SOUC_BIN="$PWD/artifacts/self-hosted/souc-self-hosted-x86_64"
$SOUNIO_SOUC_BIN examples/erdos/souc_sat.sio /tmp/souc_sat.elf
cd /tmp && /tmp/souc_sat.elf 0 4 1 1 "$OLDPWD/examples/erdos/data/degrey_529.edge"
drat-trim souc_sat_worker.cnf souc_sat_worker.drat        # s VERIFIED
```

This was **Part A** of the flagship (non-4-colourability of a real core, machine-checked
by *our* solver + drat-trim).

**Part B — exact unit-distance realisation (DONE, 2026-05-29).**
`examples/erdos/degrey_geometry.sio` reads the **exact algebraic coordinates** of all
529 vertices from `data/degrey_529.vtx` (Mathematica syntax, e.g.
`(-21+3·Sqrt[5]+7·Sqrt[33]+3·Sqrt[165])/48`) via a recursive-descent parser over a
**denominator-extended degree-8 field kernel** ℚ(√3,√5,√11) (the `Q16` XOR-mask algebra
from `degrey_fieldtower.sio` plus a common denominator; division only ever by a rational
or by √3, so no general field inverse is needed), and certifies that **every one of the
2670 edges has `dist² = 1` exactly** — **no floating point**. Result: `2670/2670`,
self-test `dist²(v0,v1)=1`, no `SQRT_ERR`/`DIV_ERR`. Magnitudes stay ~10¹³ ≪ i64 max
(no overflow). This is the Sounio analogue of upstream's Singular/Gröbner `check/*.singular`.

```bash
$SOUNIO_SOUC_BIN examples/erdos/degrey_geometry.sio /tmp/geo.elf && /tmp/geo.elf
# => edges checked: 2670   dist^2 == 1: 2670   FAIL: 0
```

**Both legs now hold in self-hosted Sounio**: G₅₂₉ is a unit-distance graph (exact) **and**
χ(G₅₂₉) ≥ 5 (drat-trim verified) ⟹ **χ(ℝ²) ≥ 5**.

## 1c. V-track — formal (Lean 4) machine-checking (2026-05-29, in progress)

- **Geometry leg → Lean 4 (DONE).** `formal/lean4/SounioDeGreyUnitDistance.lean`
  (auto-generated by `examples/erdos/gen_lean_geometry.sh`: the Sounio emitter prints the
  529 exact coordinates + 2670 edges, wrapped by a static exact-field kernel over
  ℚ(√3,√5,√7,√11) as integer 16-tuples) proves
  `theorem g529_all_edges_unit_distance : edges.all edgeUnit = true := by native_decide`.
  `lean` checks it in ~3 min; **`#print axioms` shows `[propext, native_decide.ax]` — no
  `sorryAx`**. Lean's `Int` is bignum, so the check is exact with *no overflow risk at all*
  (strictly stronger than the i64 Sounio check). **Scope (Grok 4.1 math-review):** what is
  machine-checked is the *exact algebraic squared-distance identity* in ℚ(√3,√5,√7,√11);
  the embedding of that field into ℝ (`b_mask ↦` the real radical) and the ring-homomorphism /
  ℚ-linear-independence facts are standard true multiquadratic-field properties discharged by
  construction, **not yet re-proved as Lean lemmas** — formalising that bridge is part of the
  remaining V-track.
- **SAT leg → LRAT → `cake_lpr` (DONE).** `drat-trim … -L g529.lrat` emits a **36 MB LRAT**
  (resolution proof *with hints*); then **`cake_lpr` — the CakeML/HOL4 machine-code, formally
  *verified* LRAT checker — returns `s VERIFIED UNSAT`** on `souc_sat_worker.cnf` + `g529.lrat`
  (~2 s). This is strictly stronger than drat-trim (unverified C): the "G₅₂₉ is not 4-colourable"
  claim now rests on a machine-checked checker. Reproducible: `examples/erdos/verify_lrat_cake.sh`
  (see `examples/erdos/CAKE_LPR_RESULT.md`).
- **Composition (reduction) lemma → Lean 4 (DONE, 2026-05-29).**
  `formal/lean4/SounioDeGreyChi5.lean` proves the *logical reduction*
  `(G unit-distance-embedded) ∧ (G not k-colourable) ⟹ no proper k-colouring of the
  unit-relation plane` — i.e. χ of the plane `> k`. The reduction itself
  (`reduction` = pullback `κ ∘ emb`; `not_colourable_implies_plane_chromatic_gt` =
  contrapositive; `degrey_plane_needs_5_colours` = the k=4 instantiation) is **fully proved
  in core Lean (no Mathlib, no `native_decide`)**; `lean` exits 0 and
  **`#print axioms degrey_plane_needs_5_colours` reports it depends on *no axioms at all*** —
  zero `sorryAx`. The two legs enter as *explicit, externally-discharged hypotheses*: the
  geometry leg (`g529_all_edges_unit_distance`, Lean `native_decide`) and the SAT leg
  (G₅₂₉ not 4-colourable, `cake_lpr`-verified). Grok 4.1 math-review: "NO ERRORS; reduction
  leg is logically sound." This is the standard, honest SAT+ITP combination shape — the SAT
  fact cannot be brute-forced inside Lean (4^529), so it is fed in as a verified hypothesis.
- **Geometry leg DISCHARGED into the composition → Lean 4 (DONE, 2026-05-29).**
  `formal/lean4/SounioDeGreyChi5Concrete.lean` instantiates the `SounioDeGreyChi5` reduction on
  the concrete G₅₂₉ over the **exact symbolic field-plane** `QF × QF` (the ℚ(√3,√5,√7,√11)
  16-tuple model). It defines the intrinsic unit relation `unitFP` (algebraic squared distance
  = 1), proves `unitFP_emb` (it matches `edgeUnit` definitionally on embedded points), and so
  turns the geometry leg from a hypothesis into the **PROVED** fact
  `geom_all_edges_unitFP : ∀ e ∈ edges, unitFP (emb e.1) (emb e.2)` (discharged by the same
  `native_decide` certificate). The headline
  `g529_field_plane_needs_5_colours (h_sat : ¬ VColourable) : ¬ Nonempty (PlaneColouring (QF×QF) unitFP 4)`
  therefore depends on **only `[propext, native_decide.ax]` — no `sorryAx`** (`lake build
  SounioDeGreyChi5Concrete`), with the SAT leg the *sole* remaining hypothesis. Grok 4.1
  math-review: "NO MATHEMATICAL ERRORS IN THE LEAN STATEMENTS."
- **Remaining gap (the only one, staged):** the **isometric embedding `QF ↪ ℝ`**
  (`b_mask ↦ √(∏ primes)`, a ring homomorphism) lifting "field-plane χ ≥ 5" to "χ(ℝ²) ≥ 5".
  Honest scope (per the §1c + concrete-file math-reviews): what is machine-checked is the
  **field-plane** statement (over `QF × QF`, `unitFP`); turning it into a *Euclidean* ℝ²
  statement needs real analysis (ℝ, `Real.sqrt`, `ring`) — i.e. **Mathlib**, which is not
  wired into this core-Lean project (`packages: []`, no `ring` tactic).   Bringing Mathlib in
  (or an in-tree real-radical model) is the deferred step; the multiquadratic field identities
  themselves are standard.
- **Mathlib-free ring groundwork (DONE, 2026-05-29).** `formal/lean4/SounioMultiquadRing.lean`
  begins mechanising "QF is the multiquadratic field" *without* Mathlib or the `ring` tactic:
  **proved** `qadd_comm`, `qadd_zero_{left,right}`, and `qmul_comm` (XOR-permutation symmetry via
  a finite `native_decide` certificate) — `#print axioms` clean (no `sorryAx`); the harder laws
  (`qmul` associativity/distributivity/unit/negation) are stated as explicit **open `Prop`
  obligations**, not axiomatised. Honest wall (see `docs/research/multiquad-faithfulness-note.md`):
  ℚ-linear independence + `QF↪ℝ` cannot even be *stated* in core Lean without constructing ℝ —
  that is precisely the standard/textbook part Mathlib would mechanise, which is why the exact
  symbolic field-plane statement is the honest self-hosted summit.

Honest framing (logged once, not repeated): finding/minimising the core was the hard work of
de Grey/Heule/Parts. The Sounio contribution is the *exact + self-hosted + machine-checked*
verification chain — Parts A+B and the Lean geometry leg deliver most of it; the verified
LRAT checker closes it.

### 1c-B1. SAT leg INTERNALISED in Lean core — no external checker, no Mathlib (2026-05-29)

The SAT leg no longer needs `cake_lpr`/drat-trim as a *trust* anchor: souc_sat's own CDCL
LRAT certificate is re-checked **inside Lean** by Lean core's formally-verified LRAT checker
(`Std.Tactic.BVDecide.LRAT.check` / `check_sound`, reflected by `native_decide`), then a
pure-logic bridge lifts `CNF.Unsat` to the chromatic statement. drat-trim is now only used to
*emit* the LRAT hints.

- **Bridge (DONE).** `formal/lean4/SounioSatColouringBridge.lean`:
  `not_colourable_of_unsat : (colourCNF n k edges).Unsat → ¬ ∃ proper k-colouring (Fin n → Fin k)`.
  Pure core Lean, axioms `[propext, Quot.sound]` (no Mathlib, **no `native_decide`**),
  scale-independent. Grok 4.1 math-review: "no gaps, no overclaims, no axioms beyond Lean core."
- **Full chain (DONE) on K₇/6.** `formal/lean4/SounioSatK76.lean` (autogenerated by
  `gen_lean_sat.sh`): `k76_not_colourable : ¬ 6-colourable K₇` (χ(K₇) ≥ 7), axioms
  `[propext, Classical.choice, Quot.sound, native_decide.ax]`, no `sorry`. The CNF is
  `colourCNF 7 6 edges` (clause-for-clause identical to souc_sat's emitter; verified by `diff`),
  so souc_sat's LRAT validates against it directly.
- **Two subtleties resolved:** (i) var `v` ↔ DIMACS `v+1` (`convertLRAT` relabel+1);
  (ii) Lean's `insert` *pushes*, so addition ids must be **contiguous** — `gen_lean_sat.sh`
  renumbers drat-trim's gappy ids (57 gaps on K₇/6) and remaps all hint/deletion refs. The
  un-renumbered version silently returned `check = false`.
- **G₅₂₉ (term-size wall) — superseded; see 1c-B2.** souc_sat refutes G₅₂₉ in 300 218 conflicts →
  626 MB DRAT → **98 616-line / 31.5 MB LRAT** (drat-trim, 66 784 core lemmas, `s VERIFIED`). The
  embedded-*term* `native_decide` route does not scale here (K₇/6 = 171 s for 1 464 actions). Fixed
  below by file-loaded reflection.

This realises the §3 backlog item "Formally-verified LRAT checker in Lean" and most of the §4.3
"end-to-end Lean theorem" prize.

### 1c-B2. χ(G₅₂₉) ≥ 5 CLOSED in Lean core via `souc_check` (file-loaded reflection) (2026-05-30)

The B1 term-size wall is **broken**. Instead of embedding the certificate as an `Array IntAction`
term, we embed the renumbered LRAT as a single **`String` literal** and parse it with
`SounioSatReflect.parseLRAT` *inside* the `native_decide` computation — so parse + verified-check
run as compiled native code (the string atom elaborates in ms). This is the `bv_check`/`ofReduceBool`
idea with zero new trust: the parser is unverified but **soundness-irrelevant** (`check_sound` only
trusts the verified checker's verdict on whatever actions it is fed; a parser bug can only make the
proof *fail*).

- **Scaling (DONE).** K₇/6: 171 s → **0.8 s**. G₅₂₉ (98 616 actions, 31.5 MB): **~11 s**,
  `g529_unsat : g529_cnf.Unsat`. Negative control: corrupting one SB unit ⟹ `native_decide` = false
  (`sorryAx`) — the result is genuine.
- **WLOG leg (DONE).** `formal/lean4/SounioSatColouringSB.lean`: `not_colourable_of_unsat_tri` —
  the souc_sat triangle-precolour (`0,1,5 → 0,1,2`) symmetry break is satisfiability-preserving for
  k=4 (constructive colour-permutation `relabel4`, bijectivity by `decide` over `Fin 4`). Pure logic,
  axioms `[propext, Quot.sound]`. Grok 4.1 math-review **[OK]** ("no gaps in the reduction").
- **Flagship theorem (DONE).** `formal/lean4/SounioSatG529.lean` (autogen by
  `gen_lean_sat_reflect.sh`): `g529_not_colourable : ¬ ∃ proper 4-colouring of G₅₂₉` =
  **unconditional χ(G₅₂₉) ≥ 5**, axioms `[propext, Classical.choice, Quot.sound, native_decide.ax]`,
  no `sorry`, **no Mathlib**, no external checker as a trust anchor.

The χ≥5 SAT leg is now machine-checked in Lean **at full G₅₂₉ scale** from Sounio's own solver.
Full writeup: `examples/erdos/B1_SAT_LEG_IN_LEAN.md`.

### 1c-B3. χ(QF²) ≥ 5 — field-plane CLOSED, both legs discharged, ZERO hypotheses (2026-05-30)

`SounioDeGreyChi5Concrete.lean` had discharged the geometry leg but kept the SAT leg as a
hypothesis (`h_sat : ¬ VColourable`). With 1c-B2 that hypothesis is now a **theorem**, so we wire
the two together:

- **`formal/lean4/SounioDeGreyChi5Closed.lean`** — `edges_eq` (the geometry edge list = souc_sat's,
  `native_decide`, the 2670 lists are literally equal) → `not_VColourable` (SAT leg discharged from
  `g529_not_colourable`) → **`g529_field_plane_chi_ge_5`**: the exact symbolic field-plane `QF×QF`
  (QF = ℚ(√3,√5,√7,√11)) unit-distance graph has **no proper 4-colouring** — **χ(QF²) ≥ 5**.
- **Zero remaining hypotheses, no `sorry`, no Mathlib, no external checker as a trust anchor.**
  Axioms `[propext, Classical.choice, Quot.sound, native_decide.ax]` (the `native_decide`s are: the
  verified LRAT checker, the exact unit-distance computation, the edge-list equality, the
  triangle adjacency/bounds).

Both legs of the de Grey χ≥5 lower bound are now machine-checked in Lean **at G₅₂₉ scale, over the
exact algebraic field-plane**. The single isolated gap to the *Euclidean* χ(ℝ²) ≥ 5 (Hadwiger–Nelson)
is the isometry `QF×QF ↪ ℝ²` (the multiquadratic `√3·√5=√15` ring-hom into ℝ), which needs
`Real.sqrt`/`ring` over Mathlib's ℝ — staged as `v6-mathlib-rembed`.

### 1c-B4. QF↪ℝ groundwork — multiquadratic generator law certified, no Mathlib (2026-05-30)

Attacking the last gap *without* Mathlib. The `QF↪ℝ` embedding sends each basis radical `basis m`
to `∏√pⱼ ∈ ℝ`; for that map to be a well-defined ring hom it must respect the multiquadratic
multiplication law. We certify that law **exactly in core Lean**:

- **`formal/lean4/SounioMultiquadRing.lean` — `basis_mul_law` (PROVED).**
  `∀ i,j < 16, qmul (basis i) (basis j) = bcoeff(i ∧ j) · basis (i ⊕ j)` — all 256 basis pairs, by
  `native_decide`. Plus named corollaries `√pᵢ² = pᵢ` (i ∈ {3,5,7,11}) and cross-products
  (`√3·√5=√15`, `√3·√15=3√5`, `√15·√35=5√21`). Axioms `[propext, native_decide.ax]`.
  This fixes the multiplication table on the *generators* — exactly the relations a real embedding
  must preserve. (Not yet a full ring/field certification: `qmul` assoc/distrib/inverses are still
  open, so this is the generator-level law, not "QF is the field ℚ(√3,√5,√7,√11)" in full.)
- Together with the earlier `qadd_comm/qmul_comm/qadd_zero_*`, the commutative-ring backbone of QF is
  now substantially certified. **Newly discharged (no Mathlib):** `qadd_assoc` (additive
  associativity, syntactic), `qmul_one_left/right` + `qmulOne_solved` (the canonical `qfone`
  `([1,0,…],1)` is a two-sided multiplicative unit on every length-16 representative — closes the
  former `QmulOneObligation`). Axioms `[propext, Quot.sound]`. **Still open:** `qmul`
  assoc/distrib and the additive inverse — these need a value-equivalence quotient, since QF is a
  ring of *fraction representatives* (denominators differ syntactically though values agree).

### 1c-B5. Abstract transfer leg — χ(F²)≥5 for ANY QF-receiving ring, no Mathlib (2026-05-30)

The reduction was already polymorphic in the plane `P`; we now make the **algebraic transfer**
explicit and prove the χ≥5 bound *generically*, isolating ℝ to a single instance.

- **`formal/lean4/SounioDeGreyChi5Transfer.lean` — `QFTransfer.chi_ge_5` (PROVED).**
  A `QFTransfer` packages a target type `F` with `add/mul/sub`, a map `φ : QF → F`, a unit
  predicate `isUnitVal`, the homomorphism equations `φ(qadd)=add`, `φ(qmul)=mul`, `φ(qsub)=sub`,
  and `hunit : isOne d → isUnitVal (φ d)`. Theorem: *for every such `T`, if G₅₂₉ is not
  4-colourable then `(F², T.unit)` has no proper 4-colouring* — **χ(F²) ≥ 5**.
  Axioms `[propext, native_decide.ax]` (no `Classical`, no Mathlib).
  The proof needs **no ring axioms of `F`**: the `F`-squared-distance of two embedded vertices
  collapses through `hsub/hmul/hadd` onto the *exact QF squared-distance*, which the geometry
  `native_decide` certificate already fixed at `1` for every edge; `hunit` finishes it.
- **`qfSelf` instance (PROVED, core Lean).** `F = QF`, `φ = id`, `isUnitVal = (isOne · = true)`.
  Then `qfSelf.unit = unitFP` *definitionally*, and `qfSelf.chi_ge_5` **is**
  `g529_field_plane_needs_5_colours` — the abstraction is faithful and non-vacuous.

### 1c-B6/B7. QF quotient ring + abstract SqrtField — both pillars built (no Mathlib, 2026-05-30)

Built in parallel via two `best-of-n-runner` subagents, then independently rebuilt + statement-audited
+ math-reviewed by the main agent (no `sorryAx`; axioms `[propext, Classical.choice, Quot.sound, native_decide.ax]`).

- **`formal/lean4/SounioMultiquadQuotient.lean` — QF value-equivalence quotient ring.**
  `QFeq x y := ∀ i<16, gi x.1 i * y.2 = gi y.1 i * x.2` (cross-multiplication value equality);
  `Setoid QFp` on positive-denominator length-16 reps (transitivity via `Int.eq_of_mul_eq_mul_right`).
  PROVED: qadd/qmul/qsub congruences, additive inverse `qadd_neg_QFeq` (closes `QaddNegObligation`),
  left/right distributivity (closes the distrib obligations), and `qCommRing : QCommRingBundle`
  (add comm/assoc/zero, neg, mul comm/one, distrib) on the quotient. **STAGED:** `qmul` associativity
  (`QmulAssocObligation`) — the single remaining triple-sum XOR-reindex. So `QF/≈` is a certified
  commutative ring *modulo multiplicative associativity*.
- **`formal/lean4/SounioSqrtField.lean` — abstract ordered field with √ (the ℝ interface).**
  `structure SqrtField` (comm-ring + ordered-field + sqrt axioms as hypotheses, satisfied by ℝ).
  PROVED from the axioms: `nonneg_sqrt_unique` (nonneg roots of equal squares agree), `mul_sqrt`
  (`√a·√b = √(ab)`), `ofNat_nonneg`, the four primes `s_sq` (`√pⱼ² = pⱼ`), radical map `r`, `r_zero`.
  **PROVED (2026-05-30):** `GeneratorLawObligation` (`r i · r j = ofNatProd(i∧j) · r(i⊕j)`) — the abstract
  image of `basis_mul_law`, i.e. the multiplicative core of the eventual `QF↪ℝ` homomorphism — is now
  the theorem `generator_law` (+ `generatorLaw_solved`). Proof by finite four-bit radical factorisation:
  reusable microlibrary `ofNat_one/ofNat_add/ofNat_mul` (ℕ→F cast hom), `mul8` (8-factor interleave via
  3× `sf_mul_mul_mul_comm`), and the per-bit `radicalBit_mul` (four `Bool` cases, `(t,t)` is exactly
  `s_sq`); the coefficient product collapses to `ofNatProd(i∧j)` via `ofNat_mul` + `Nat.testBit_and`.
  Axioms `[propext, Quot.sound]` — no Mathlib, no `Classical`, no new structure field (law DERIVED).

**STRUCTURAL FINDING (2026-05-30) — `QFTransfer` must be guarded by `den ≠ 0`.**
A *total* `QFTransfer` instance into a field is **impossible**: `hadd`/`hmul`/`hsub` are `∀ a b`
over *all* `QF` (any denominator, including `0`), and the two natural maps conflict:
- φ that **drops** the denominator (`φ(c,d) = Σ cᵢ rᵢ`) makes `hmul` hold for free (generator
  law), but `hadd` fails — `qadd` cross-multiplies the numerators by the denominators.
- φ by **fractions** (`φ(c,d) = (Σ cᵢ rᵢ)·inv(ofInt d)`) makes `hadd` hold, but `hmul` fails at
  `den = 0` (and even the if-`d=0`-then-`0` patch fixes `hmul` while re-breaking `hadd`).
So the ℝ instance needs a well-formedness guard (`den ≠ 0`, length 16) on the homomorphism laws.
That, in turn, needs an **ordered field to have characteristic zero** (`ofInt d ≠ 0` for `d ≠ 0`,
so denominators are invertible). **Done this iteration** in `SounioSqrtField.lean`:
`ofNat_ne_zero`, `ofInt`/`ofInt_ne_zero`, `sf_inv_one`, `sf_inv_ne_zero`, `sf_inv_mul_inv`
(`inv(ab)=inv a·inv b`) — all `#print axioms`-empty (fully constructive, no Classical).

**Precise remaining steps for Euclidean χ(ℝ²) ≥ 5.**
1. **(decision) guard `QFTransfer`** with `qfWf x := x.2 ≠ 0 ∧ x.1.length = 16` on `hadd/hmul/hsub`;
   reprove `qfSelf` and `geom_transfer` (the `emb` values are well-formed, so threading the guard is
   mechanical). 2. **`ofInt_add`/`ofInt_mul`** (ℤ→F ring hom) — **DONE (2026-05-30)** in
   `SounioSqrtField.lean` (`ofInt_neg`/`ofInt_add`/`ofInt_mul`, Int constructor case analysis,
   `[propext, Quot.sound]`).
3. **`evalNum`** numerator map + its multiplicative core `evalNum (qmul x y) = mul (evalNum x)(evalNum y)`
   — **DONE (2026-05-30)** in new `SounioMultiquadHom.lean`: theorem `evalNum_qmul`, proved via a
   from-scratch Mathlib-free finite-sum library (`fsum` + add/zero/congr/mul-distrib/`fsum_mul_fsum`/
   map/`fsum_perm`/`fsum_comm`/`fsum_xor`), the `generator_law` per-pair rewrite, and the
   `perm_range_xor` XOR reindex (j↦i⊕idx). Den-free heart of `hmul`; new code `[propext, Quot.sound]`.
4. **φ via fractions** + `hadd/hmul/hsub` using the char-0 inverse toolkit + `evalNum_qmul` —
   **DONE (2026-05-30)** in `SounioMultiquadHom.lean`: `phi (c,d) = mul (evalNum c) (inv (ofInt d))`
   with `phi_qmul`/`phi_qadd`/`phi_qsub` proved under the `den ≠ 0` guard (numerator-linearity
   `evalNum_qadd`/`evalNum_qsub` + field fraction identities `frac_add`/`frac_sub`). `phi_qadd`/
   `phi_qsub` `[propext, Quot.sound]`; `phi_qmul` inherits the `perm_range_xor` certs.
   **Unital law also DONE (2026-05-30)**: `phi_unit` — any QF representing the rational `1`
   (coeff₀ = den, all other coeffs `0`, den ≠ 0) maps to `R.one`, via the summand-isolation lemma
   `fsum_single` + `r_zero` + `mul_inv`. So **φ is a complete UNITAL ring homomorphism QF→F**:
   `phi_qmul`/`phi_qadd`/`phi_qsub` supply `hmul`/`hadd`/`hsub`, and `phi_unit` supplies `hunit`.
5. **Guarded transfer — DONE (2026-05-30)** in `SounioDeGreyChi5TransferWf.lean`. `QFTransferWf`
   packages the transfer with the guard `qfWf x := x.2 ≠ 0` on `hadd/hmul/hsub/hunit`; `geom_transfer_wf`
   threads the guard through the squared distance (each of `qadd`/`qmul`/`qsub` carries denominator
   `x.2*y.2`, so `qfWf` is preserved); `emb_den_ne_zero` discharges it on the whole edge set
   (`native_decide`: every `X`/`Y` coordinate has nonzero denominator; default `1`). The `SqrtField`
   instance `sqrtTransfer R` plugs `phi_qadd`/`phi_qmul`/`phi_qsub`/`phi_unit` straight into the four
   laws — the geometry's `DeGrey529.*` and φ's byte-identical `MultiquadRing.*` are **definitionally
   equal**, so no bridge lemmas were needed. **`sqrtField_chi_ge_5` : χ(F²) ≥ 5 for EVERY `SqrtField` F**,
   Mathlib-free, axioms `[propext, Classical.choice, Quot.sound]` + the legitimate `native_decide`
   certs (`perm_range_xor`, `geom_all_edges_unitFP`, `allX_ne`, `allY_ne`); **no sorryAx**.
6. **The sole remaining gap: the analytic `SqrtField ℝ` instance** — ℝ as an ordered field with √.
   **STARTED (2026-05-30)** in `SounioSqrtFieldReal.lean`: ℝ = quotient of `SounioRealCauchy` by the
   null-difference relation `RealEq`, now proved a **full equivalence** (`realEq_refl` + `realEq_symm`
   + `realEq_trans` via the ε/2 triangle on the core `Rat` order API) and packaged as `realSetoid`, so
   **ℝ := `Quotient realSetoid`** is available. An explicit obligation ledger remains for the deferred
   analytic core. Mathlib gives the instance in
   one line; a Mathlib-free witness needs the ε/2 transitivity/triangle lemmas over `Rat` (sparse
   order API — even `Rat.add_le_add` absent), the mul-monotonicity law (`SounioRealCauchy` defers
   `mul_le_mul_of_nonneg_right` ≈500–1000 LOC + `OrderedCarrierObligation`), order completeness (sup),
   and a constructive √ with `sqrt_sq`. This is multi-week and is the honest remaining frontier.
   Once `(R := ℝ)` is supplied, `sqrtField_chi_ge_5 ℝ g529_not_colourable` is **χ(ℝ²) ≥ 5** — SAT leg,
   geometry certificate, generator law, char-0 toolkit, the **full QF→F unital ring homomorphism**, and
   the **abstract χ(F²)≥5 transfer** are all closed in core Lean; only the ℝ-model satisfiability of the
   `SqrtField` axioms remains.

## 1b. Implemented & gated this iteration — `examples/erdos/souc_sat.sio`

A hardened engine `souc_sat.sio` was forked from `cdcl_fast.sio` and three plan
items were landed behind the external gate (`drat-trim s VERIFIED` on disk). All
numbers below are re-runnable:

```bash
SOUC=./bin/souc
$SOUC examples/erdos/souc_sat.sio /tmp/souc_sat.elf && chmod +x /tmp/souc_sat.elf
/tmp/souc_sat.elf                      # pigeonhole gate K4..K7 + writes souc_sat_k76.{cnf,drat}
drat-trim souc_sat_k76.cnf souc_sat_k76.drat      # external arbiter -> s VERIFIED
/tmp/souc_sat.elf 0 8 1                 # streamed worker: K8/7, seed 0, LRB -> souc_sat_worker.{cnf,drat}
/tmp/souc_sat.elf 0 8 0                 # same with VSIDS (arg3=0) for A/B
drat-trim souc_sat_worker.cnf souc_sat_worker.drat        # 23 MB streamed proof -> s VERIFIED
examples/erdos/portfolio.sh 8 8 $(command -v drat-trim)   # P1 portfolio on K8 (streaming+LRB)
```

**E0 — proof on disk + overflow guard, then TRULY STREAMED.** First a `write_file`
buffered path with a refuse-on-overflow latch. Then the real fix: a **held
`syscall6` fd + 64 KiB write buffer** (`s_open`/`wb_*`/`s_close`) so the worker
streams DRAT **as it is produced** during `solve()` — RAM is O(1) in proof size.
This **unblocks de Grey-scale**: K₈/₇ (impossible under any fixed buffer) now closes
with a **23 MB streamed proof on disk that drat-trim verifies** (`s VERIFIED`).
The CNF is streamed the same way. Negative control intact (unjustified ⊥ rejected).

**E1b — LRB branching** (Learning-Rate Branching, Liang et al. 2016) in integer
fixed-point: per-var participation rate folded into a learning-rate EMA `Q[v]`,
branch on max `Q`; α decays 0.40→0.06. Toggle `USE_LRB` (worker arg 3) for A/B.
**Decisive on structured UNSAT**: K₈/₇ conflicts **182,445 (VSIDS) → 46,165 (LRB),
≈4×**, both drat-trim `s VERIFIED`.

**E1 — recursive clause minimisation** (MiniSat `ccmin_mode=2`, iterative with
the abstract-level signature prune) in `analyze()`. Drops redundant learned
literals before the second-watch/btlevel choice; `seen[]` is restored via an
explicit to-clear list. Soundness is arbitrated by drat-trim (a wrong removal
makes the lemma non-RUP).

**E2 — Glucose LBD-EMA dynamic restarts** with an integer recent-window vs
global-average force condition and a **trail-EMA block** (postpone when the
partial model is unusually large) to avoid the pigeonhole thrash naive restarts
caused. Plus periodic rephasing.

**E4/E4b — blocking literals + chronological backtracking (sound, currently
neutral — HONEST NEGATIVE).** Two kissat-era propagation/backtrack refinements
landed behind the gate but did **not** move search statistics on the current
instance families: (i) *blocking literals* cache the other watched literal so
`propagate()` can skip a satisfied clause without dereferencing it (sound, no
speedup with the conservative blocker — commit `0436ba374`); (ii) *chronological
backtracking* (Nadel & Ryvchin 2018, `CHRONO_LIMIT=100`) backtracks one level
instead of to `btlevel` when the backjump gap exceeds the threshold (commit
`cc97280b8`). Both keep drat-trim `s VERIFIED` on K₈/₇ and **G₅₂₉** (327,208
conflicts unchanged, 32 s unchanged within noise). The gap exceeds 100 too rarely
on these LRB+Glucose runs to register. The mechanisms are wired correctly for
other instance families; the missing kissat-parity lever is a richer
blocker/lazy-watch + arena cache layout, staged as E4c.

**E4c — profile-driven win (LRB pick cache).** Instrumented `propagate()` on G₅₂₉:
`litval` dominates at **1.50 B** calls (≈22 per propagation), chrono-BT fires
**0 times**, blocking literals already skip 54 % of watch nodes — so the chrono
and richer-blocker levers are dead on this instance (the latter measured *slower*,
reverted). The profile pointed at `pick()` doing an O(nvars) scan every decision;
caching the current best LRB var (refreshed on bump/unassign, invalidated on
assign) gives **G₅₂₉ 327,208 → 300,218 conflicts (−8.3 %) and 33.8 s → ~31 s
(≈−10 %)**, drat-trim `s VERIFIED` (66 MB DRAT). K₈/₇ conflicts regressed +14 %
(different but still-sound search trajectory) at neutral wall-time — an honest
heuristic trade, not a pure no-op speedup. kissat/CaDiCaL were **not installed**
in the sandbox, so no PAR-2 comparison was fabricated; that bench stays open.
Next lever (profile-justified): inline `assign[]` reads in the propagate hot loop
to cut the 1.5 B call overheads without changing search, or an LRB max-heap.

**P1 — diversified portfolio.** `souc_sat.sio` doubles as a CLI worker
(`souc_sat <seed> <clique_n> <lrb> <sb>`): the seed perturbs **only** search order
(initial phase, activity bias, restart cadence). `portfolio.sh` forks N workers in
private dirs, first-to-UNSAT wins, and verifies the winner's cert with drat-trim;
`portfolio_slurm.sbatch` is the SLURM-array cluster variant.

**F2 — symmetry breaking + graph-colouring encoder (domain win).** Colours are
interchangeable, so any proper colouring can be permuted to give a known clique
vertex *i* colour *i* — a **satisfiability-preserving** predicate, hence
*F∧SB* UNSAT ⟹ *F* UNSAT. Two surfaces landed:

- `add_sb_units` precolours the first k-clique of the pigeonhole family. On K₈/₇
  this collapses the search: **46,165 conflicts (LRB) → 1 conflict**, drat-trim
  `s VERIFIED`. (`/tmp/souc_sat.elf 0 8 1 1`.)
- An **edge-list k-colouring encoder** (`add_edge`, `add_atleast_one`,
  `build_spindle_3col`) plus level-0 propagation of original **unit** clauses now
  drives a *real unit-distance graph* through the whole stack. The **Moser
  spindle 3-colouring is UNSAT ⟹ χ(spindle) ≥ 4**, certified end-to-end (encoder
  → triangle-precolour SB → LRB CDCL → streamed DRAT → `drat-trim s VERIFIED`).
  Verified **both with SB (2 conflicts) and without SB (13 conflicts)** — the
  no-SB run independently confirms the graph is genuinely non-3-colourable, so SB
  is sound here, not masking a colourable graph. (`/tmp/souc_sat.elf 0 0 1 1`.)

This is the first **de Grey-pipeline** result on a real geometric graph: the
same machinery (exact-edge graph → SB → CDCL → checked proof) that scales to a
5-chromatic core, demonstrated on the canonical χ=4 unit-distance graph.

**Value precedence** (Law–Lee 2004; `add_value_precedence`, `SB=2`/`SB=3`) also
landed and is `drat-trim s VERIFIED` on spindle + K₈/₇. Honest finding from the
A/B/C/D matrix: it is **redundant with triangle precolour for the de Grey k=4 case**
— because unit-distance plane graphs are **K₄-free (ω=3)**, precolouring one
triangle leaves residual colour symmetry S_{k−ω}=S₁ (trivial), so clique-precolour
is *already complete*; VP gives identical conflict counts (spindle 13→2; K₈/₇
46 165→1). VP is therefore kept as the **general** tool for k−ω≥2 (triangle-free
graphs, k≥5), not as the χ≥5 lever. See `DEGREY_LITERATURE_REVIEW.md` §4 for the
corrected analysis and the revised critical path (exact geometry + Lean V-track,
*not* more symmetry breaking — refuting a given core is trivial for any CDCL).

Measured on K₇/₆ (drat-trim core stats — the externally-checkable metric):

| metric | `cdcl_fast` (base) | `souc_sat` E0+E1 | + E2 restarts | portfolio best (seed 2) |
|---|---:|---:|---:|---:|
| conflicts | 1459 | 1485 | 990 | **703** |
| core lemmas | 1067 | 777 | 653 | 654 |
| resolution steps | 11296 | 9603 | 7924 | 8132 |
| redundant lits in core | 334 | 117 | **86** | 55 |
| drat-trim | VERIFIED | VERIFIED | VERIFIED | **VERIFIED** |

Portfolio diversification is real: across 8 seeds K₇/₆ conflicts spanned
703–2138 (≈3×); the fastest worker beats the single default config (990).

Newly added (K₈/₇, drat-trim `s VERIFIED` on the streamed proof):

| metric | VSIDS | **LRB** |
|---|---:|---:|
| conflicts | 182,445 | **46,165** |
| restarts | 1,695 | 415 |
| proof | streamed→disk, verified | streamed→disk, verified |

Still **staged** (not yet built): CHB + EVSIDS + stable/focused mode switching
(rest of E1); Luby stable mode + tiered clause DB (rest of E2); all of E3
inprocessing, E4 perf, P2 cube-and-conquer, P3 shared-DB, the Lean LRAT checker
(V), the de Grey core flagship, and the kissat benchmark. These remain as
specified below. With streamed proofs + LRB + the graph-colouring encoder with
clique-precolour symmetry breaking, the **de Grey core line is now unblocked end
to end** — the spindle (χ=4) already goes through; what remains for χ≥5 is a
genuine 5-chromatic core (exact edges) and stronger symmetry (lex-leader /
value-precedence) for the larger search.

---

## 2. The modern-CDCL gap (Phase 1–2 backlog)

Ordered by expected impact on hard structured UNSAT (the de Grey regime):

### 2.1 Decision heuristics
- [ ] **VSIDS → EVSIDS** (exponential VSIDS with float-free fixed-point decay; we
      already use integer activities — switch to the rescaling EVSIDS scheme).
- [ ] **LRB (Learning-Rate Branching)** + **CHB** — the heuristics that beat VSIDS on
      structured instances (Liang et al., 2016). Maintain a per-variable learning-rate
      EMA; this is *the* single highest-leverage change for de Grey-type problems.
- [ ] **Mode switching** (CaDiCaL "stable vs focused"): alternate LRB/VSIDS phases.

### 2.2 Restarts & phases
- [x] **Glucose-style LBD-based dynamic restarts** (restart when the recent-LBD EMA
      exceeds the global LBD average × K) instead of fixed inner/outer.
      *Done in `souc_sat.sio` with an integer cross-multiplied condition + trail-EMA block.*
- [ ] **Reluctant doubling / Luby** as the stable-mode schedule.
- [x] **Rephasing** with multiple strategies (saved / inverted / random / best-found).
      *Periodic phase flip implemented; multi-strategy rephasing still staged.*
- [ ] **Target phases / local search rephasing** (walk-based phase seeding).

### 2.3 Clause database
- [x] LBD clause deletion (done).
- [ ] **Tier system** (CaDiCaL): core (LBD ≤ 2) / tier-2 (LBD ≤ 6, used-recently) /
      local; age + activity based eviction with `used` bits.
- [ ] **Clause activity** bumping (alongside LBD) for the local tier.

### 2.4 Inprocessing (the big one)
- [x] **Recursive clause minimisation** à la MiniSat (`ccmin_mode=2`) in `analyze` —
      removes redundant learned literals. *Done in `souc_sat.sio`; drat-trim core
      redundant-literal count on K₇/₆ fell 334 → 86.* (Full on-the-fly
      self-subsumption against the DB is still staged.)
- [ ] **Vivification** (asymmetric branching to strengthen clauses).
- [ ] **Bounded Variable Elimination (BVE)** + **subsumption** + **self-subsuming
      resolution** as a preprocessing and periodic inprocessing pass.
- [ ] **Equivalent-literal substitution** (SCC over the binary implication graph),
      **failed-literal probing**, **hyper-binary resolution**.
- [ ] Every inprocessing step must **emit its DRAT/LRAT lemmas + deletions** so the
      proof stays valid — this is non-trivial and is gated by Phase 3.

### 2.5 Engine
- [ ] **Blocking literals** in watch lists (cache the other watch's value to skip
      clause access) — measurable propagation speedup.
- [ ] **Chronological backtracking** (Nadel & Ryvchin 2018) for the cases where it
      helps.
- [ ] **Cache-friendly clause arena** (we already use flat arrays — formalise the
      arena layout, 32-bit clause refs, aligned literals).
- [ ] **Phase-saving + trail reuse** on restart (don't always cancel to 0 in focused
      mode).

**Validation rule for all of Phase 1–2.** Each feature lands behind the *same*
correctness gate: the pigeonhole + small-graph suite must stay `RUP:VERIFIED` *and*
`drat-trim s VERIFIED`, with the new inprocessing lemmas/deletions present. A feature
that can only be made to pass by weakening the proof is rejected (CLAUDE.md §6.6).

---

## 3. Proof infrastructure (Phase 3) — required before de Grey scale

The in-memory DRUP array is the current hard limit. de Grey-scale needs:

- [~] **Proof to disk** — `souc_sat.sio` now serialises the full DIMACS+DRAT cert
      to disk via `write_file` with a refuse-on-overflow guard. *Truly streamed*
      (held `syscall6` fd, no in-RAM `P_lits`) is the remaining step; the 16 MiB
      buffer already overflows at K₈/₇, confirming the need.
- [ ] **Binary DRAT** output (compact) — and/or
- [ ] **LRAT** (Linear RAT with resolution hints) emitted directly. LRAT is *checkable
      in linear time* and is what modern verified toolchains use.
- [ ] **Native Sounio LRAT checker** (forward, hint-guided) — fast, no backward search.
- [ ] **Formally-verified LRAT checker in Lean** (the `cake_lpr` / `lrat-check` idea):
      a Sounio→Lean-extracted or Lean-proven checker whose soundness theorem is
      `checker accepts ⇒ formula UNSAT`. This is the keystone of Front F1.
- [ ] **GPU-checked proofs** (optional, Front F3): the repo has PTX codegen; RUP
      replay is embarrassingly parallel across independent lemma cones.

Gate: re-verify all Phase-1/2 certificates through the new LRAT path *and* drat-trim
*and* the Lean checker. Three independent checkers must agree.

---

## 4. The de Grey χ ≥ 5 flagship (Phase 4)

Two parallel attack lines; either suffices for the certificate.

### 4.1 Smaller 5-chromatic cores first
- [ ] Acquire / reconstruct the **minimised cores** (Heule's 510-vertex graph and the
      ~500–600-vertex family). Verify each core's unit-distance edges **exactly** in
      Sounio (we already do this for the 1581-vtx graph with `i64` fixed-point; upgrade
      to exact ℚ(√3,√5,√7,√11) integer tuples, the degree-16 kernel already in
      `degrey_fieldtower.sio`).
- [ ] Close the 4-colouring UNSAT of a core with the Phase-1/2 solver; emit LRAT;
      verify with drat-trim + Lean checker. **This is the first real χ ≥ 5 certificate.**

### 4.2 Symmetry-aware encoding (domain win, Front F2)
- [x] **Clique-precolour symmetry breaking** landed (`add_sb_units`,
      satisfiability-preserving). K₈/₇ collapses 46,165→1 conflict; drat-trim
      `s VERIFIED`. Edge-list k-colouring encoder + level-0 unit propagation also
      landed, certifying **χ(Moser spindle) ≥ 4** end-to-end (UNSAT with *and*
      without SB → SB confirmed sound on a real unit-distance graph).
- [ ] de Grey's graph has a large rotation/reflection automorphism group. Add
      **lex-leader / value-precedence** colour-class symmetry breaking (the
      polynomial-size predicate that helps the small-clique de Grey graphs where
      clique-precolour alone is weak) — the next encoder win.
- [ ] **Cube-and-conquer**: split on a few high-degree vertices' colours, solve cubes
      in parallel (Front F3, GPU/cluster), recombine proofs.

### 4.3 The end-to-end Lean theorem (the prize)
- [ ] Formalise in Lean 4: *"if `G` is a finite unit-distance graph in ℝ² (edges
      certified at exact distance 1 over the number field) and the 4-colouring CNF of
      `G` is UNSAT (per a verified LRAT proof), then χ(ℝ²) ≥ 5."* Compose:
      `geometry (exact) → graph → CNF (sound encoding lemma) → UNSAT (LRAT checker
      theorem) → χ ≥ 5`. End-to-end machine-checked, Sounio-native. **POPL/CAV-grade.**

---

## 5. Surpassing kissat — the competitive phase (Phase 5)

Once Phase 1–3 land, benchmark honestly:

- [ ] **Benchmark harness**: SATLIB + SAT-Competition `main`/`structured` tracks, plus
      our geometric family. Report PAR-2, solved-count, and proof sizes vs kissat 4.x
      and CaDiCaL, same hardware (the `cpu-ops` node; document exact CPU/RAM).
- [ ] **F2 first**: demonstrate wall-clock dominance on the geometric/colouring family
      (where specialised encoding + symmetry + structure-aware heuristics should win).
- [ ] **F1 next**: publish the verified-solver Pareto point (fastest *verified*).
- [ ] **F3**: smaller/faster proofs + GPU checking.
- [ ] Only then, if the data supports it, claim a head-to-head win on a defined track —
      with a re-runnable command, per CLAUDE.md §6.1.

### Sounio-native advantages to exploit (the "edge of novelty", CLAUDE.md §10)
- **Refinement types** to make solver invariants *checked by the type system*
  (e.g. "watched literal index < clause length", "trail level monotone") — fewer bugs,
  and the invariants double as Lean proof obligations.
- **Algebraic effects** to cleanly separate the pure search core (verifiable) from
  `IO`/`Alloc`/proof-emission effects — the verified core has *no* side effects.
- **Linear types** for the clause arena (no aliasing bugs, deterministic frees).
- **GPU effect** (`with GPU`) for parallel propagation / proof checking / portfolio.
- **`Knowledge[T]` / epistemic types** for a principled portfolio that tracks
  confidence across solver configs (a genuinely novel angle).

---

## 6. Phase plan & gates (summary)

| Phase | Deliverable | External gate |
|---|---|---|
| **P1** | LRB/CHB + clause minimisation + tiered DB in `cdcl_fast` | pigeonhole + small graphs `drat-trim s VERIFIED`, with minimised clauses |
| **P2** | Inprocessing (BVE, vivification, probing) with proof emission | same suite + new lemmas, 3-checker agreement |
| **P3** | Streamed LRAT + native + Lean LRAT checker | de Grey-scale proof checkable on disk; Lean checker soundness theorem |
| **P4** | First χ ≥ 5 certificate (small core) → full de Grey | drat-trim + Lean checker accept; geometry exact in number field |
| **P5** | Benchmark vs kissat/CaDiCaL; claim a defended front | re-runnable PAR-2 table on documented hardware |

**Effort estimate (single author, calibrated to the 90-day compiler):** P1 ≈ 2–3
weeks, P2 ≈ 3–4 weeks, P3 ≈ 3–4 weeks (Lean checker is the long pole), P4 ≈ 2–3 weeks
(cores) + open-ended (full de Grey), P5 ongoing. The verified-checker (P3) and LRB
(P1) are the highest-leverage items — do them first.

---

## 7. Risks & honesty guards

- **"Surpass" inflation.** Never claim a general kissat win from a domain or verified
  result. State the front explicitly and show the command (CLAUDE.md §6.1, §6.7).
- **Proof-weakening drift.** A feature that needs the proof relaxed to pass is a
  rejected feature, not a passed gate (§6.6).
- **Inprocessing ↔ proof coupling.** Every BVE/vivification/probing step must emit
  correct DRAT/LRAT; this is the subtlest correctness risk — gate each step in
  isolation before composing.
- **de Grey may stay open** under our solver for a long time; the smaller-core line
  (P4.1) is the de-risked path to a *real* χ ≥ 5 certificate.
- **Single arbiter dependence.** Keep ≥ 2 independent checkers (drat-trim + Lean LRAT)
  so no single checker bug can pass an unsound proof.

---

## 8. Immediate next action (P1 landed; next up)

Done: recursive clause minimisation (E1), Glucose LBD-EMA restarts (E2),
proof-on-disk with overflow guard (E0), diversified portfolio (P1) — all in
`souc_sat.sio` / `portfolio.sh`, all `drat-trim s VERIFIED`.

Next, in priority order:

1. **Truly streamed proof** (held `syscall6` fd + buffered writer; drop in-RAM
   `P_lits` and the 16 MiB text cap) so K₈/₇+ and de Grey-scale proofs fit. This
   unblocks every larger experiment.
2. **LRB / CHB** alongside VSIDS with stable/focused mode switching — the single
   highest-leverage search-quality change for de Grey-type structured UNSAT.
3. **Heap clause arena** (`heap_alloc`/`heap_realloc`) to drop the fixed
   8192-var / 262144-clause caps (the smoke test `tests/run-pass/stdlib_mem_alloc.sio`
   confirms `malloc`/`free` work natively).
4. Re-run de Grey on SLURM via `portfolio_slurm.sbatch`; measure
   conflicts-to-first-core-shrink and whether `trail` trends toward a level-0
   refutation.

Everything in this file is gated, re-runnable, and externally checkable. That is the
only way a solo solver beats a 20-year-old institution: not by claiming, by *proving*.
