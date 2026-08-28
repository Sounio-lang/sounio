<!-- docs:meta
topic_id: repo.examples.erdos.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.erdos.readme
-->

# Erdős unit-distance / chromatic-number examples

Sounio programs around two Erdős-flavoured problems on unit-distance graphs in the
plane, with an emphasis on **exact arithmetic** (no floating point in the geometry)
and **native SAT/UNSAT certification** (the in-repo CDCL solver in
`stdlib/theorem/smt.sio`, no external solver, no third-party DRAT).

Build & run any example:

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc examples/erdos/<file>.sio /tmp/out.elf && /tmp/out.elf
```

---

## New verified results (2026-06-16)

Two new results are now formalised in Mathlib-free Lean 4:

1. **`χ(G₅₂₉) = 5`** — exact chromatic number of the de Grey graph.
   - Lean: `formal/lean4/SounioDeGreyChi529Exact.lean`
   - Reproducer: `examples/erdos/gen_g529_5coloring.sh`
2. **`u(15705) ≥ 176768`** — new explicit planar unit-distance lower bound.
   - Witness: integer disk `x² + y² ≤ 5000` with squared unit distance `1105`.
   - Lean: `formal/lean4/SounioErdos90PlanarLowerBound.lean`

See `examples/erdos/RESULTS_2026-06-16.md` for details and `examples/erdos/AUDIT_STATUS_2026-06-15.md` §15 for the implementation trail.

---

## Thread 1 — Hadwiger–Nelson / de Grey (Erdős #508): χ of the plane

de Grey (2018, arXiv:1804.02385) gave the first 5-chromatic unit-distance graph,
settling χ(ℝ²) ≥ 5. His graph is built from **Moser spindles** assembled under a
fixed group of rotations. The key fact we exploit: the whole construction lives in
an **exact algebraic number field**, so it can be represented with integer tuples —
no rounding, no epsilon tolerances.

### The field tower (resolved from the literature)

de Grey's graph derives from the ring `N = Z[ω₁, ω₃, ω₄, ω₁₆]`, where
`ω_t = exp(i·arccos(1 − 1/2t))`, so `cos θ_t = 1 − 1/2t` and `sin θ_t = √(4t−1)/(2t)`:

| rotation | cos | sin | surd introduced |
|---|---|---|---|
| ω₁  | 1/2   | √3/2     | √3 |
| ω₃  | 5/6   | √11/6    | √11 |
| ω₄  | 7/8   | √15/8    | **√5** (√15 = √3·√5) |
| ω₁₆ | 31/32 | √63/32 = 3√7/32 | **√7** (√63 = 9·7) |

So:

- the **Moser spindle** and its ω₁,ω₃ family (Golomb, V, G₂₁, G₄₃, G₄₉) live in
  **ℚ(√3, √11)** (degree 4);
- de Grey's **full 1581-vertex 5-chromatic graph** needs **ℚ(√3, √5, √7, √11)**
  (degree 16) — the auxiliary surds √5, √7 come from ω₄, ω₁₆.

### Files

| File | What it does | Field | Result |
|---|---|---|---|
| `degrey_q3q11_spindle.sio` | Moser spindle, exact coords (scale ×12) | ℚ(√3,√11) | χ = 4 (brute force **and** native 3-col UNSAT / 4-col SAT) |
| `degrey_fragment_q3q11.sio` | glues a 2nd spindle by a 60° rotation; checks field closure | ℚ(√3,√11) | 11-vtx graph, all unit edges dist²=576 exact, no auxiliary surd; native 3-col UNSAT / 4-col SAT |
| `degrey_fieldtower.sio` | extends the kernel to the degree-16 field; XOR multiplication law; realizes ω₄ (√5) and ω₁₆ (√7) unit edges exactly | ℚ(√3,√5,√7,√11) | 5/5 arithmetic + rotation checks pass |
| `native_sat_scale_demo.sio` | native CDCL on graphs of known χ past the old 256-var cap | — | 6/6 correct (cycles to 1022 vars) |
| `sat_proof_kernel.sio` | from-scratch DPLL→**DRUP** emitter + independent **RUP checker**; demo K₄ not 3-colorable | — | VERIFIED UNSAT + 2 non-vacuity controls rejected; native **and** drat-trim `s VERIFIED` |
| `spindle_proof_cert.sio` | exact ℚ(√3,√11) spindle geometry → 3-coloring CNF → DRUP proof → RUP + DIMACS/DRAT | ℚ(√3,√11) | χ(spindle) ≥ 4: native VERIFIED + external drat-trim `s VERIFIED` |
| `dpll_scale_wall.sio` | scaling envelope of the DPLL→DRUP certifier on pigeonhole K_n/(n−1)-col | — | K₄–K₇ VERIFIED (17→6491 lemmas); K₈+ WALL; K₆ drat-trim `s VERIFIED` |
| `cdcl_proof.sio` | **CDCL (1-UIP) clause-learning** solver emitting DRUP + same RUP checker | — | K₄–K₁₀ VERIFIED; breaks the DPLL wall (K₇: 485 vs 6491 lemmas); K₇ drat-trim `s VERIFIED` |
| `cdcl_fast.sio` | production-shaped **two-watched-literal CDCL** + integer VSIDS + phase saving + inner/outer restarts + **LBD clause deletion** (DRUP with `d` lines); pigeonhole correctness gate | — | K₄–K₇ VERIFIED with deletion active; K₇ drat-trim `s VERIFIED` on a proof with **1136 `d` deletions**; non-vacuity rejected |
| `degrey_chi5_fast.sio` | de Grey's **actual 1581-vertex graph** (data acquired, not fabricated): exact i64 fixed-point unit-distance check + 4-colouring CNF + the fast CDCL | i64 fixed-point | **7877/7877** unit edges exact; 8.6 M conflicts / 53 min at bounded memory — **χ ≥ 5 not closed** (honest non-result) |
| `souc_sat.sio` | hardened engine (plan E0/E1/E2 + P1 + **F2** + **DIMACS edge-file mode**): **recursive minimisation**, **LRB branching**, **Glucose LBD-EMA restarts**, **truly streamed proof to disk** (held `syscall6` fd, O(1) RAM), CLI portfolio worker, **graph-colouring encoder + symmetry breaking** (`SB`: 1=triangle/clique-precolour, 2=value-precedence, 3=both), **reads real de Grey cores** | — | K₄–K₇ `RUP:VERIFIED`; **K₈/₇ 23 MB proof `s VERIFIED`**; LRB K₈/₇ **182k→46k**; **🏁 χ(G₅₂₉) ≥ 5: real 529-vtx Heule de Grey core refuted, 72 MB DRAT, `drat-trim s VERIFIED`**; **χ(Moser spindle) ≥ 4** (UNSAT with+without SB); unjustified-⊥ rejected |
| `degrey_geometry.sio` | **Part B**: exact unit-distance certifier — parses `degrey_529.vtx` (exact coords) over ℚ(√3,√5,√11) (denominator-extended `Q16` kernel + recursive-descent Mathematica parser), checks `dist²=1` for every edge with **no floats** | — | **🏁 2670/2670 edges `dist²=1` exact** ⟹ G₅₂₉ is a unit-distance graph; with Part A ⟹ **χ(ℝ²) ≥ 5** |
| `gen_lean_geometry.sh` → `formal/lean4/SounioDeGreyUnitDistance.lean` | **V-track**: Lean 4 formalisation of the geometry leg — Sounio emits the exact coords/edges, wrapped by an exact ℚ(√3,√5,√7,√11) field kernel | — | **🏁 `theorem g529_all_edges_unit_distance := by native_decide`** checks in `lean` (~3 min), `#print axioms` = no `sorry` (bignum Int, no overflow) |
| `formal/lean4/SounioDeGreyChi5.lean` | **V-track composition**: Lean 4 *reduction* lemma — `(G unit-distance) ∧ (G not k-colourable) ⟹ plane χ > k`; geometry + SAT legs enter as explicit externally-discharged hypotheses | — | **🏁 fully proved in core Lean (no Mathlib/`native_decide`); `#print axioms` = depends on no axioms at all** (no `sorryAx`). Grok 4.1 math-review: "reduction leg is logically sound" |
| `formal/lean4/SounioDeGreyChi5Concrete.lean` | **V-track geometry-leg DISCHARGED**: instantiates the reduction on concrete G₅₂₉ over the exact symbolic field-plane `QF×QF` (ℚ(√3,√5,√7,√11) model); proves every edge is `unitFP` via the `native_decide` cert, leaving **only the SAT leg** hypothetical | ℚ(√3,√5,√7,√11) | **🏁 `g529_field_plane_needs_5_colours` depends on only `[propext, native_decide.ax]` — no `sorryAx`** (`lake build SounioDeGreyChi5Concrete`). Scope: **field-plane** χ≥5; the `QF↪ℝ` embedding (→ Euclidean χ(ℝ²)≥5) needs Mathlib, staged |
| `formal/lean4/SounioSatColouringBridge.lean` | **B1 SAT-leg bridge**: soundness of the graph-colouring SAT encoding — `(colourCNF n k edges).Unsat ⟹ ¬ proper k-colouring` (Fin) | — | **🏁 pure core Lean, axioms `[propext, Quot.sound]`** (no Mathlib, no `native_decide`); Grok 4.1 math-review: "no gaps, no overclaims, no axioms beyond Lean core" |
| `gen_lean_sat.sh` → `formal/lean4/SounioSatK76.lean` | **B1 SAT leg INTERNALISED (full chain)**: souc_sat's *own* CDCL LRAT cert, re-checked by Lean core's **verified LRAT checker** (`check_sound` + `native_decide`), then the bridge ⟹ chromatic bound — **no external checker, no Mathlib** | K₇ / 6 | **🏁 `k76_not_colourable : ¬ 6-colourable K₇` (χ≥7), axioms `[propext, Classical.choice, Quot.sound, native_decide.ax]`, no `sorry`** (`lake build SounioSatK76`, ~3 min) |
| `formal/lean4/SounioSatColouringSB.lean` | **B2 WLOG leg**: souc_sat's triangle-precolour symmetry break is satisfiability-preserving (k=4) — `not_colourable_of_unsat_tri` lifts the SB-augmented `Unsat` to an *unconditional* χ≥5 (`relabel4` bijection by `decide` over `Fin 4`) | — | **🏁 pure core Lean, axioms `[propext, Quot.sound]`**; Grok 4.1 math-review: "no gaps in the reduction" |
| `gen_lean_sat_reflect.sh` → `formal/lean4/SounioSatG529.lean` | **B2 FLAGSHIP — χ(G₅₂₉) ≥ 5 in Lean core via `souc_check`**: souc_sat's **98 616-action** LRAT re-checked inside Lean by the verified checker via *file-loaded-style reflection* (LRAT as a `String` literal parsed under `native_decide` — no term-size wall, **~11 s**) | G₅₂₉ / 4 | **🏁 `g529_not_colourable` (χ(G₅₂₉)≥5), axioms `[propext, Classical.choice, Quot.sound, native_decide.ax]`, no `sorry`, no Mathlib** (`lake build SounioSatG529`). Embedded-term route was infeasible; this scales |
| `formal/lean4/SounioDeGreyChi5Closed.lean` | **B3 — field-plane χ(QF²) ≥ 5 CLOSED**: wires the now-proven SAT leg into the geometry reduction (`edges_eq` proves the two edge lists identical), discharging the last hypothesis | G₅₂₉ on QF×QF / 4 | **🏁 `g529_field_plane_chi_ge_5`: the exact ℚ(√3,√5,√7,√11)-plane unit-distance graph has no proper 4-colouring — ZERO hypotheses, no Mathlib, no `sorry`** (`lake build SounioDeGreyChi5Closed`). Only `QF↪ℝ` remains for Euclidean χ(ℝ²)≥5 |
| `formal/lean4/SounioMultiquadRing.lean` | **B4 — QF↪ℝ groundwork**: multiquadratic generator law `basis_mul_law` (√i·√j = bcoeff(i∧j)·√(i⊕j), all 256 pairs) + √pᵢ²=pᵢ / cross-products; plus earlier qadd/qmul commutativity & zero | 16 basis radicals | fixes the generator-level multiplication table the real embedding must preserve; axioms `[propext, native_decide.ax]` (`lake build SounioMultiquadRing`). NOT a full ring/field cert (assoc/distrib open) |
| `formal/lean4/SounioDeGreyChi5Transfer.lean` | **B5 — abstract transfer**: `QFTransfer.chi_ge_5` proves **χ(F²)≥5 for ANY ring F receiving QF via a homomorphism** (no Mathlib, no ring axioms of F); `qfSelf` (id) instance recovers the field-plane result | any QF-receiving F / 4 | isolates Euclidean χ(ℝ²)≥5 to **one ℝ instance** (`φ`=eval radicals, `Real.sqrt`); axioms `[propext, native_decide.ax]` (`lake build SounioDeGreyChi5Transfer`) |
| `formal/lean4/SounioMultiquadQuotient.lean` | **B6 — QF quotient ring**: value-equivalence `QFeq` Setoid on length-16 positive-denominator reps; congruence + additive inverse + left/right distributivity + `qCommRing` bundle | QF/≈ | certified commutative ring **modulo `qmul` associativity** (staged); no Mathlib, no `sorryAx` (`lake build SounioMultiquadQuotient`) |
| `formal/lean4/SounioSqrtField.lean` | **B7 — abstract real-field interface**: `SqrtField` (ordered field + √, ℝ-satisfiable); `nonneg_sqrt_unique`, `mul_sqrt` (`√a·√b=√(ab)`), radical map `r`; **`generator_law` PROVED** (`rᵢ·rⱼ = ofNatProd(i∧j)·r_{i⊕j}`, four-bit factorisation); **char-0 denominator toolkit** `ofNat_ne_zero`/`ofInt`/`ofInt_ne_zero`/`sf_inv_mul_inv`; **ℤ→F ring hom** `ofInt_add`/`ofInt_mul`/`ofInt_neg` (all axiom-free) | abstract `SqrtField` | target a `QFTransfer` ℝ-instance plugs into; generator law = multiplicative core of `QF↪ℝ`; char-0 = invertible denominators for the fraction map (no Mathlib/Classical, `lake build SounioSqrtField`) |
| `formal/lean4/SounioMultiquadHom.lean` | **B8 — QF→SqrtField ring homomorphism**: Mathlib-free finite-sum library `fsum` (add/zero/congr/mul-distrib/`fsum_mul_fsum`/map/`fsum_perm`/`fsum_comm`/`fsum_xor`/`fsum_neg`); `evalNum l = Σ_{i<16} ofInt(lᵢ)·r i`; **`evalNum_qmul` PROVED**; **fraction map `φ(c,d)=(Σcᵢrᵢ)·inv(ofInt d)` as a UNITAL ring homomorphism: `phi_qmul`/`phi_qadd`/`phi_qsub` (den≠0) + `phi_unit` (QF representing 1 ↦ `R.one`) PROVED** | abstract `SqrtField` / 16 | promotes per-generator `generator_law` to the full bilinear convolution identity (`perm_range_xor` reindex) and completes it to a **den-aware unital field homomorphism** (frac-add identities `frac_add`/`frac_sub` + char-0 inverse toolkit + `fsum_single` summand-isolation for `phi_unit`); `phi_qadd`/`phi_qsub` `[propext, Quot.sound]`, `phi_unit` `[propext, Classical.choice, Quot.sound]`, `phi_qmul` inherits `perm_range_xor` certs (`lake build SounioMultiquadHom`) |
| `formal/lean4/SounioDeGreyChi5TransferWf.lean` | **B9 — guarded abstract transfer**: `QFTransferWf` (den≠0 guard on `hadd/hmul/hsub/hunit`), `geom_transfer_wf`, `chi_ge_5_wf`, `emb_den_ne_zero`, `sqrtTransfer` instance; **`sqrtField_chi_ge_5` PROVED — χ(F²) ≥ 5 for EVERY `SqrtField` F** | abstract `SqrtField` | the proved fraction homomorphism (`phi_qmul`/`phi_qadd`/`phi_qsub`/`phi_unit`) instantiates the guarded transfer (geometry `DeGrey529.*` ≡ φ's `MultiquadRing.*` by defeq — no bridges); guard discharges on the edge set (`native_decide`: all `X`/`Y` denominators nonzero); axioms `[propext, Classical.choice, Quot.sound]` + legitimate native certs, **no sorryAx**; reduces χ(ℝ²)≥5 to the lone analytic fact "ℝ is a `SqrtField`" (`lake build SounioDeGreyChi5TransferWf`) |
| `data/degrey_529.edge`, `data/degrey_529.vtx`, `data/parts_510.edge` | vendored 5-chromatic cores (Heule CNP-SAT) | 529/510 | 2670/2504 | DIMACS edge + exact Mathematica coords | input to the χ≥5 pipeline; see `data/README.md` |
| `portfolio.sh` / `portfolio_slurm.sbatch` | P1 diversified portfolio: N seeded workers (search-only diversification), first-to-UNSAT wins, winner cert drat-trim-checked; local + SLURM-array | — | K₇/₆ 8 seeds: conflicts **703–2138** (≈3×); K₈/₇ streaming+LRB winner `s VERIFIED` |
| `data/degrey/gen_solver.py` | generator emitting `cdcl_fast.sio` (php) and `degrey_chi5_fast.sio` (degrey) from the graph CSVs | — | — |
| `erdos90_cubic_tower_base.sio` | explicit witness for the OpenAI-2026 #90 disproof's cubic tower base (Gauss periods) | cubic ⊂ ℚ(ζ_r) | 11/11 certified: field disc = r², r totally ramified |

### The degree-16 kernel (`degrey_fieldtower.sio`)

A ℚ-basis of ℚ(√3,√5,√7,√11) is the 2⁴ = 16 monomials `√(∏ S)` for `S ⊆ {3,5,7,11}`,
indexed by a 4-bit mask (bit0=√3, bit1=√5, bit2=√7, bit3=√11). Multiplication is
**pure XOR** — the same algebra as Cayley–Dickson, with positive square-root coeffs:

```
√(∏S) · √(∏T) = ( ∏_{p ∈ S∩T} p ) · √(∏ (S △ T))
```

i.e. basis `i · j` lands on basis `i XOR j` with rational coefficient = product of the
primes selected by `i AND j`. An element is an integer 16-tuple; one `O(16²)` loop
implements exact multiplication. Pairwise coprimality of {3,5,7,11} guarantees the 16
monomials are linearly independent (degree exactly 16, no collapse). Math-reviewed
(xai / Grok 4.1, 2026-05-28): the multiplication is the standard multiquadratic
relation and all identities/realizations hold. See `.claude/llm_offload_log.md`.

### Native χ certification

A `k`-colouring is encoded as boolean CNF (one var per vertex×colour; at-least-one +
at-most-one per vertex; per-edge same-colour exclusion) and handed to the in-repo
CDCL solver. χ ≥ k+1 is certified by a `k`-colouring **UNSAT**; a colouring exists
iff **SAT**. The spindle's χ = 4 is `3-col UNSAT ∧ 4-col SAT`, both native.

### Verifiable UNSAT certificates — "Sounio computes" → "Sounio *proves*"

A bare UNSAT result is "trust the solver". `sat_proof_kernel.sio` and
`spindle_proof_cert.sio` upgrade this to an **independently checkable proof**:

- a from-scratch DPLL refutation **emits a DRUP proof** (a DPLL search tree is a
  tree-resolution refutation; at every refuted node it emits `¬(decision literals
  on the path)`, which is RUP; the root emits the **empty clause**, post-order so
  each lemma is RUP w.r.t. the formula plus earlier lemmas);
- an **independent native RUP checker** replays the proof with only unit
  propagation (no shared state with solver heuristics) and **rejects** invalid
  proofs (controls: empty proof and a bogus non-implied unit are both rejected);
- the certificate is also emitted in **standard DIMACS + DRAT** for an external
  checker.

**External cross-check (verification-only C toolchain).** The Sounio-emitted
certificates were verified by `drat-trim` (Marijn Heule; the canonical DRAT
checker), built locally for *verification only* (no solver, no science in C):

```bash
./bin/souc examples/erdos/spindle_proof_cert.sio /tmp/spindle.elf && /tmp/spindle.elf > /tmp/out.txt
awk '/^%%DIMACS%%/{f="cnf";next} /^%%DRAT%%/{f="drat";next} /^%%END%%/{f="";next} \
     f=="cnf"{print > "/tmp/spindle.cnf"} f=="drat"{print > "/tmp/spindle.drat"}' /tmp/out.txt
drat-trim /tmp/spindle.cnf /tmp/spindle.drat     # => s VERIFIED
```

| certificate | vars | clauses | DRUP lemmas | native RUP | drat-trim |
|---|---:|---:|---:|---|---|
| K₄ not 3-colorable (χ ≥ 4) | 12 | 22 | 17 | VERIFIED | `s VERIFIED` (22/22 core, 83 steps) |
| Moser spindle not 3-colorable (χ ≥ 4) | 21 | 40 | 29 | VERIFIED | `s VERIFIED` (40/40 core, 241 steps) |

The spindle's edges are **derived from exact ℚ(√3,√11) unit-distance arithmetic**,
not hand-listed — the de Grey mechanism in miniature. (A two-reviewer disagreement
on the DRUP-emission soundness, and its resolution, is logged in
`.claude/llm_offload_log.md`.)

### Scaling envelope and the CDCL wall — `dpll_scale_wall.sio`

Two honest facts bound how far this DPLL→DRUP certifier reaches toward χ ≥ 5:

1. **3-coloring UNSAT is *local*.** Any graph containing a Moser spindle is refuted
   by the 7-vertex spindle alone, so the proof stays ~30 lemmas at any host size —
   not evidence of the χ ≥ 5 regime.
2. **The χ ≥ 5 regime is a *global* hard UNSAT** (a 4-coloring UNSAT on de Grey's
   ~510-vertex graph). Measured against the global hard family `k`-coloring `K_n`
   with `k = n−1` (= pigeonhole `PHP(n,n−1)`, exponential resolution lower bound,
   Haken 1985):

   | instance | clauses | DPLL nodes = DRUP lemmas | native RUP | drat-trim |
   |---|---:|---:|---|---|
   | K₄ / 3-col | 22 | 17 | VERIFIED | — |
   | K₅ / 4-col | 45 | 103 | VERIFIED | — |
   | K₆ / 5-col | 81 | 749 | VERIFIED | `s VERIFIED` (6275 steps) |
   | K₇ / 6-col | 133 | 6 491 | VERIFIED | — |
   | K₈ / 7-col | 204 | > 3 000 000 | WALL | — |
   | K₉ / 8-col | 297 | > 3 000 000 | WALL | — |

   The factorial blow-up (≈ ×6, ×7, ×9 per step) is the Haken bound made concrete.

**Conclusion (path to χ ≥ 5).** Reaching de Grey's 510-vertex 4-coloring UNSAT
needs (1) the graph data (not in this repo, not fabricated) and (2) **CDCL clause
learning** in the emitter — chronological DPLL provably cannot produce a
sub-exponential proof of a pigeonhole-hard instance. The native RUP checker and
DIMACS/DRAT bridge already in place are reusable as-is; a CDCL upgrade changes only
*how lemmas are produced*, not how they are checked.

### Breaking the wall with clause learning — `cdcl_proof.sio`

The CDCL upgrade is now built: a from-scratch **conflict-driven clause-learning**
solver with a trail, decision levels, implication reasons, **1-UIP conflict
analysis**, and **non-chronological backjumping**. Each learned clause is a
resolution consequence of existing clauses (hence RUP), so the sequence of learned
clauses ending in the empty clause is a valid **DRUP** proof — exactly how
production solvers emit DRAT. The *identical* native RUP checker and `drat-trim`
bridge verify it; a bug in the CDCL bookkeeping cannot yield a false "VERIFIED".

Measured on the same pigeonhole family (PHP is exponential for *general*
resolution, so CDCL cannot crack it either — but learning reaches much further
before the wall, and is *the* mechanism that makes structured de-Grey-type
4-colourings tractable):

| instance | clauses | DPLL lemmas | **CDCL lemmas** | native RUP | drat-trim |
|---|---:|---:|---:|---|---|
| K₄ / 3-col | 22 | 17 | **9** | VERIFIED | — |
| K₅ / 4-col | 45 | 103 | **40** | VERIFIED | — |
| K₆ / 5-col | 81 | 749 | **149** | VERIFIED | — |
| K₇ / 6-col | 133 | 6 491 | **485** | VERIFIED | `s VERIFIED` (3650 steps) |
| K₈ / 7-col | 204 | **WALL** (> 3 M) | **1 517** | VERIFIED | — |
| K₉ / 8-col | 297 | **WALL** | **4 343** | VERIFIED | — |
| K₁₀ / 9-col | 415 | **WALL** | **11 832** | VERIFIED | — |

Clause learning turns the DPLL wall at K₈ into routine instances and gives a 13×
lemma reduction at K₇. The blow-up persists (≈ ×3 per step — pigeonhole stays
exponential), but the regime that matters for χ ≥ 5 is *structured*, not
pigeonhole; there CDCL's learning is the decisive lever. The certifier pipeline
(emit → native RUP → drat-trim) is now production-shaped and waits only on the
de Grey graph data.

### Production-shaped solver + de Grey's real graph — `cdcl_fast.sio`, `degrey_chi5_fast.sio`

Two blockers from the previous section are now cleared:

1. **The graph data is in the repo** (`data/degrey/`, acquired from the public
   de Grey 2018 dataset — *not* fabricated). `degrey_chi5_fast.sio` reads the 1581
   vertices / 7877 edges and **verifies the unit-distance property exactly** with
   `i64` fixed-point arithmetic (scale 10⁷, integer squared distance vs `10¹⁴` within
   tolerance): **7877 / 7877 edges confirmed unit-distance**, no floating point.
2. **The clause-budget / large-struct SRET blocker is side-stepped.** The fast solver
   uses flat module-level arrays (no by-value `SmtContext` return), so it scales to
   **millions of clauses** with no SRET corruption. The 4-colouring CNF is 6324 vars /
   33089 clauses and loads cleanly.

`cdcl_fast.sio` is a from-scratch, production-shaped CDCL:

- **two-watched-literal** unit propagation (each clause watches two literals; only
  watched-literal lists are scanned — the standard modern propagation engine);
- **integer VSIDS** decision heuristic with decay/rescale, **phase saving**;
- **inner/outer (MiniSat-style) restarts** — keep restarts frequent and
  well-distributed, far better for hard UNSAT than unbounded geometric growth;
- **LBD-based clause deletion** (`reduce_db`): each learned clause is scored by
  *Literal Block Distance* (distinct decision levels); periodic reductions at decision
  level 0 keep all originals, glue clauses (LBD ≤ 3) and locked clauses, delete the
  rest, **emit a DRUP `d` record for every deletion**, then compact the DB, remap
  reasons and rebuild the watch lists.

#### Soundness of clause deletion — the drat-trim arbiter

Clause deletion is the part most likely to break a proof, so the **certificate's
validity is anchored on the external checker, never on the solver**: a bug in
deletion can only make `drat-trim` **reject** the proof, never falsely accept it.
Two independent checks back this:

- the **native RUP checker** *ignores* `d` records and stays sound (a lemma that is
  RUP w.r.t. a superset DB is still RUP — it can only over-approximate, never wrongly
  accept the empty-clause derivation);
- **`drat-trim`** *processes* the deletions and replays the add/delete stream in
  order.

Gate (pigeonhole `K_n/(n−1)`-colouring, with deletion active):

| instance | conflicts | restarts | reduces | clauses deleted | native RUP | drat-trim |
|---|---:|---:|---:|---:|---|---|
| K₄ / 3-col | 7 | 0 | 0 | 0 | VERIFIED | — |
| K₅ / 4-col | 28 | 0 | 0 | 0 | VERIFIED | — |
| K₆ / 5-col | 156 | 1 | 0 | 0 | VERIFIED | — |
| K₇ / 6-col | 1 459 | 14 | 3 | **1 136** | VERIFIED | `s VERIFIED` (1067/1460 lemmas in core, 11 296 steps) |

The K₇ certificate carries **1136 `d` deletion lines**; `drat-trim` returns
`s VERIFIED`. Non-vacuity: corrupting one added lemma flips `drat-trim` to
`s NOT VERIFIED`. So the deletion machinery is externally certified sound and the
gate has teeth. (Pigeonhole is pathological for restart-heavy search — it needs to
*retain* a large resolution refutation — so it is a deliberate *correctness* gate, not
a performance proxy; the χ ≥ 5 regime is structured, where restarts + learning help.)

#### de Grey's full graph on the cluster (SLURM)

`degrey_chi5_fast.sio` was run on the Sounio HPC cluster (`cpu-ops` partition, single
node, the binary shipped to the compute node over `srun` stdin and streamed back).
Clause deletion changed the run from "hits the memory wall" to "bounded-memory steady
state":

| | old run (no deletion) | strengthened (deletion + inner/outer restarts) |
|---|---|---|
| conflicts in ≈ 50 min | 3.5 M (degrading) | **8.6 M** (steady ≈ 2700/s) |
| restarts | 51 | **2534** |
| live clause DB | ballooned to 3.5 M | **bounded 40 k–230 k** |
| outcome | memory wall | bounded, still searching |

**Honest non-result.** Even strengthened, the solver did **not** close de Grey's
4-colouring UNSAT in 8.6 M conflicts: the `trail` keeps oscillating with no trend to a
level-0 refutation. Two hard facts emerged:

- a from-scratch CDCL (integer VSIDS, no inprocessing/vivification, no LRB/CHB, no
  preprocessing) is **far behind industrial solvers** (kissat/CaDiCaL close this in
  seconds); memory was never the real barrier — *search quality* is;
- the in-memory DRUP buffer (`P_lits`, 1 G literals) **fills at ≈ 7.5 M conflicts**, so
  even a much longer run cannot keep a verifiable proof — de Grey-scale certification
  needs **streamed / compressed proofs (binary DRAT or LRAT on disk)**, not an
  in-memory array.

This sharpens the path to an actual χ ≥ 5 certificate (see
[`CHI5_SOLVER_ROADMAP.md`](CHI5_SOLVER_ROADMAP.md)): the bottleneck is now a
**competitive, proof-producing, ideally formally-verified solver**, plus
**disk-backed proof infrastructure** — not graph data and not the clause budget.

### Solver scaling — status and BLOCKER

`native_sat_scale_demo.sio` validates the solver past the old 256-variable cap on
graphs with rigorously known χ (even cycles SAT, odd cycles UNSAT, complete graphs
UNSAT). Current state of `stdlib/theorem/smt.sio`:

- **boolean variable cap: 1024** (raised from 256 in this work) — stable; validated
  across the full demo sequence (a wrong bound flips a known-χ case) and the
  `test_smt_solver_basic` regression (incl. LIA) stays green.
- **clause budget: 4096 clauses / 16384 literals (unchanged).**

> **OPEN ISSUE (gates the de Grey-scale, χ ≥ 5 path).** de Grey's minimised graphs
> (~510 vertices) are *dense*: a 4-colouring is ≈ 2040 vars **and ≈ 10 000 clauses**.
> That needs both a bigger clause budget and more literals. When `clause_data` was
> temporarily enlarged to 65 536 (`SmtContext` ≈ 800 KB), a known-SAT even cycle
> (C₄₀₀, 800 vars) returned spurious **UNSAT** in one specific build of
> `native_sat_scale_demo.sio`. **This was NOT reliably reproducible**: minimal
> struct-return probes (512 KB array, trailing scalars after the array, deep
> recursion over a live large struct) all initialise correctly, and the exact demo
> sequence at the 512 KB config later returned the correct SAT. The symptom — a
> layout-sensitive wrong value from a by-value struct return, with no crash — is
> consistent with the known compiler SRET struct-return corruption family
> documented in `docs/audit/r2_3_compiler_tuple_return_bug/` and the large-struct
> follow-up in `docs/audit/sret_large_struct_smtcontext/`. Pending a *reproducible*
> diagnosis, the clause budget is kept at the proven-stable size (4096/16384) and
> only *sparse* graphs reach >256 vars. **No χ ≥ 5 certificate is claimed here.**

### What is NOT done (honesty boundary)

- We do **not** certify χ(ℝ²) ≥ 5. We now *do* have de Grey's actual 1581-vertex graph
  (`data/degrey/`) and verify its 7877 unit edges **exactly** (`i64` fixed-point), and
  we build and attack the 4-colouring CNF with a production-shaped, deletion-enabled,
  externally-verified CDCL — but that solver does **not close the UNSAT** (8.6 M
  conflicts, no refutation). No χ ≥ 5 certificate is claimed.
- The remaining gap is no longer data or clause budget. It is **solver strength** (a
  competitive CDCL on par with kissat/CaDiCaL) and **proof infrastructure** (streamed
  binary-DRAT/LRAT instead of an in-memory buffer that fills at ≈ 7.5 M conflicts).
  The staged plan to get there — and to make the Sounio solver genuinely *surpass*
  kissat on a chosen front — is [`CHI5_SOLVER_ROADMAP.md`](CHI5_SOLVER_ROADMAP.md).

---

## Thread 2 — `168_*.sio`: chromatic / orbit separation

The `168_*` family explores chromatic obstructions and orbit structure on
unit-distance / Cayley-style graphs (Erdős #168 neighbourhood). See
`docs/research/erdos-168-chromatic-separation.md` for the scope: these demonstrate a
"Level-2 algebraic separation" and explicitly make **no** claim to resolve #508/#704
or to improve any published bound.

---

## Provenance

- Field tower & arithmetic: math-reviewed via `bin/llm-offload -t math-review -p xai`
  (logged in `.claude/llm_offload_log.md`).
- de Grey, *The chromatic number of the plane is at least 5*, arXiv:1804.02385 (2018).
- Polymath16 project wiki (rotation tower, ω_t generators).
