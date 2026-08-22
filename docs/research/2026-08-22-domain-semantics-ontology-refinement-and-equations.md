# Domain semantics in Sounio: from per-area effects to ontology-as-refinement to equations-as-conservation-law

Date: 2026-08-22
Branch: lane/fable-1/p0f-ffi-takeover
Status: design synthesis (brainstorm capture, no code changed)

This note captures a design arc that began with a founder question — *do the
science domains in stdlib (chemistry, physics, …) have their own types or
effects, and could each area have its own semantics?* — and resolved into a
single, measured thesis. Every quantitative claim below is backed by a command
re-runnable on this checkout.

---

## 1. The question and the short answer

- **Types per domain: yes, already, everywhere.** Every science domain carries a
  real type vocabulary — `FourMomentum`/`PhononState` (physics), `VancoPKParams`/
  `TacPKParams` (clinical), `EpistemicStatevector`/`Hamiltonian` (quantum),
  `EpistemicReaction`/`BigCRN`/`CHEBI` bridge (chemistry), `TypeAEval`/
  `CalCertificate` (metrology). This was never the gap.
- **A dedicated effect per area: no — and it should not exist.** One access
  effect (`Epistemic`, id 8) suffices. The proliferation of per-area effect names
  is a symptom of putting a *value* property in a *computation* slot.
- **Own semantics per area: yes — as an ontology tag drawn from ONE shared
  lattice, carried in the type (a refinement), with an algebra of composition.**

---

## 2. Measured state of the effect system

Sounio's checker uses a **closed, numbered effect registry** in
`self-hosted/check/effects.sio` — `effect_name_to_id` maps a name to an id in
**0–22**; effects are consumed by id via `has_effect_id(row, count, id)`.

```
0 IO · 1 Mut · 2 Alloc · 3 Panic · 4 Div · 5 GPU · 6 Async · 7 Prob ·
8 Epistemic · 9 Causal · 10 Network · 11 Sensor · 12 Render · 13 Observe ·
14 NonAssoc · 15 Audit · 16 Hypothesis · 17 MultiTest · 18 ZD · 19 Witness ·
20 Temporal · 21 Learn · 22 Chaotic
```

Several of these are genuine domain semantics enforced as effects: `NonAssoc`
(non-associative multiplication must be declared), `ZD` (zero-divisor, carried by
`Forgettable<T>`), `Chaotic` (forbids float reassociation on a positive-Lyapunov
integration path), `Witness` (emit a machine-checkable Lean witness), `Epistemic`
(reading `.value` on a `Knowledge<T>` requires the capability).

### 2.1 The phantom effects

Names used in real `fn … with …` signatures across `stdlib/` that the checker
does **not** recognise. Unknown names return `-1` in `effect_name_to_id` and are
**silently dropped** by `collect_effects_from_list` (`if eff_id >= 0`):

- Confirmed phantom in **both** engines (0 occurrences anywhere in
  `self-hosted/`): `GUM`, `Perturbative`, `NarrowWidthApproximation`,
  `NonUnitary`.
- `Confidence(950)` — a **parameterised** effect; the byte-name registry cannot
  even represent `E(n)`.

```bash
grep -rn "Perturbative\|NarrowWidth\|NonUnitary" self-hosted/   # → 0
grep -nE  "return (2[3-9]|[3-9][0-9])" self-hosted/check/effects.sio  # → nothing (max id 22)
```

### 2.2 The two-engine split (the decisive datum)

There are two effect representations:

- **`self-hosted/compiler/lean_single.sio` (the seed)** uses a **bitmask**;
  `Approx` is real there — `src_match(..., "Approx") { return 262144 }` (bit 18),
  and it *propagates into closures* (`lean_single.sio:14837–14909`).
- **`self-hosted/check/effects.sio` (Madaros, the default engine)** does **not**
  map `Approx` at all — the modular rewrite *lost* the recognition the seed had.

**`Approx` is the existence proof that propagation ≠ meaning.** It is recognised
(in the seed), it propagates, and it still changes no computation, because **no
rule consumes it**. Therefore "recognise `GUM`" (give it an id so it propagates)
would merely clone `Approx`'s inertia. The gap is categorically **not a missing
id — it is a missing composition rule.**

| Ontology | seed (lean_single) | Madaros | consumed by a rule? |
|---|---|---|---|
| `Approx` | bit 18, propagates to closures | dropped | **no** (inert even where it propagates) |
| `GUM` | ghost (−1) | ghost (−1) | no |
| `Perturbative` / `NWA` / `NonUnitary` | absent | absent | no |

> Measurement-methodology lessons paid for in this session: (1) strip comments
> before counting effect usage — a naive grep for `with GUM` matched 130 sites /
> 95 files that were 95% prose; the real count is 7 sites / 6 files. (2) Verify
> "recognised" at the **definition site** (`effects.sio` / the seed bitmask), not
> from a usage count — usage count is not recognition.

---

## 3. N5 — uncertainty-ontology mismatch is a type error

The claim (falsifiable, with a Lean obligation in the CEI program):

> **N5.** The effect/type of an uncertain value carries which *uncertainty
> ontology* produced its error bar (variance / interval / systematic band /
> order-dependent / …). Combining two committed, incompatible ontologies without
> an explicit reconciliation is a **type error**, not an addition.

**Honesty gate (measured):** zero files currently mix `GUM` with a rival ontology
in real code — the earlier "three coexisting files" were comment false positives.
N5 is a **hazard the type system should prevent**, not a defect now producing
wrong numbers. The honest claim is narrower and still strong: nothing prevents
it, and the vocabulary for both ontologies is already in the source.

### 3.1 The reframe: the ontology belongs in the TYPE, not the effect row

An effect is a property of the *computation*; the N5 rule fires at `a + b` and
must compare the ontology of the *two operands*, which only their **types** can
carry. This is the founder's own moduli ruling applied again — *the modulus
belongs in the type as a refinement, because a refinement carries the algebra.*

Two mechanisms were conflated under one name:

| Mechanism | Example | Correct home |
|---|---|---|
| Access capability | `Epistemic` (id 8): touching `.value` needs it | effect ✓ (already right) |
| Provenance of the error bar | `GUM`/`Perturbative`/`Approx`: which calculus made it | **refinement on the type** ✗ (currently mis-slotted) |

The CEI type `Knowledge<T, ε, Valid, Provenance>` already declares the slots and
then discards them (never persisted in `TypeEntry`). **N5 = persist the ontology
slot in `TypeEntry` and consume it at the arithmetic join.** Proof that this is
the right design is *already in the tree*: **metrology** — the discipline whose
entire subject is uncertainty — carries `TypeAEval` (the GUM term for Type A vs
Type B evaluation) and `CalCertificate` as **types**, using no effects at all.

### 3.2 The consumer: `check_ontology_join`

At the checker site that already recognises both operands are `Knowledge`
(the `.value requires Epistemic` path, `check.sio:5432` / `19273`):

```
check_ontology_join(ont_a, ont_b) -> Ont | E
```

over a **flat lattice with ⊤/⊥**:

```
              ⊤ = Approx   (unspecified; NON-absorbing)
             /    |     |        \
      Variance  Interval  Band   OrderDependent
      (GUM,     (p-box,   (Pertur- (associator,
      aleatory) Knightian, bative,  NonAssoc)
                epistemic) NWA)
             \    |     |        /
              ⊥ = Exact
```

- **equal** → ok, result inherits the ontology.
- **⊤ `Approx`** → combines with a committed ontology only via an **explicit
  down-cast** the author writes; a bare join with ⊤ is `E`. (⊤ must hurt to use,
  or it is the universal solvent that erases every commitment.)
- **two distinct committed ontologies** → **`E`**, *not* silent widening — unless
  a **reconciliation handler** exists (CEI's tail-resumptive handler machinery).
- Tags are **inferred from the constructor** (`ep_measured`→Variance,
  `interval_new`→Interval, truncated series→Band); the author writes a tag only at
  a reconciliation point — exactly where the science needs a human decision.

**The dissertation special case:** `Variance ⊗ Interval` (aleatory IIV × epistemic
parameter uncertainty) has a principled reconciliation — a two-component
`Knowledge<T, RandomEpistemic>` **product**, not a collapse to ⊤. This is the
number GUM-tools and Uncertain⟨T⟩ cannot represent, and it is the PBPK thesis.

---

## 4. Per-domain map (the paper's motivation table, measured)

| Domain | Ontologies invoked | Own types | Regime |
|---|---|---|---|
| **particle_physics** | GUM · Perturbative · NWA · NonUnitary · Approx | `FourMomentum`, `DiracMatrix`, `BLUEResult`, `DMCandidate` | **four ontologies at once** |
| **clinical** | Knightian (7) | `VancoPKParams`, `TacPKParams`, `MercyGraph` | deep / imprecise |
| **metrology** | *(none as effect)* | `TypeAEval`, `CalCertificate`, `CorrectedValue` | GUM — **put in the type** |
| **darwin_pbpk** | GUM · Confidence · Uncertainty | `BBBGUMBudget`, `BBBGateVerdict`, `BBBPCE` | Variance ⊗ epistemic |
| **chemistry** | GUM | `EpistemicReaction`, `CHEBI`, `BigCRN` | Variance |
| **physics / pbpk** | GUM | `FourMomentum`/`PhononState` · `BloodFlow`/`AdditiveError` | Variance |
| **quantum** | Uncertainty · GUM | `EpistemicStatevector`, `Hamiltonian` | NonUnitary latent |
| **genomics / neuro** | **none** | `Kmer`, `FastaRecord` · connectivity matrices | discrete — **⊥ Exact** |

Three findings:

1. **Metrology already does it right** — uncertainty carried in the type
   (`TypeAEval`), no effect. The phantom effects elsewhere are other domains
   trying to say the same thing in the wrong slot.
2. **particle_physics is the killer N5 demo** — the one domain natively using
   multiple incompatible ontologies, and where the physicist's own practice
   (never add systematic and statistical uncertainty naively; combine in
   quadrature only under conditions) *is* the composition rule N5 enforces.
3. **genomics / neuro bound the claim honestly** — discrete domains have no
   continuous uncertainty ontology (the ⊥=`Exact` case). N5 is not universal, and
   must not force a tag where there is no uncertainty.

---

## 5. Equations directly: notation as the surface of a conservation law

Measured state:

- Reactions exist as **data** (`EpistemicReaction`, `BigCRN`), and species are
  grounded in a real ontology (`h_atom_chebi()`→CHEBI:49637,
  `o_atom_chebi()`→CHEBI:25805).
- **Atom/mass balance is checked nowhere** — `grep balance|conservation|
  atom_count chemistry/` → 0. An unbalanced `BigCRN` type-checks today.
- The **physical law lives in a comment** — `sr.sio:123` `// E² − p²c² = m²c⁴`;
  the code is `invariant_mass(E, uE, p, up)` with uncertainty **threaded by hand**.

Writing an equation directly is not sugar — it is **the compiler holding the
conservation law the equation asserts**, the same "assertion ⇒ evidence" law one
more time, now at the level of notation:

| | The equation asserts | Evidence the compiler holds |
|---|---|---|
| Chemistry | `2 H2 + O2 -> 2 H2O` | **atom + charge balance** → unbalanced = compile error |
| Physics | `E² = (pc)² + (mc²)²` | **dimensional homogeneity** + **uncertainty derived from the equation** |

Chemistry is the twin of dimensional analysis, and is buildable on what exists:
species carry composition via CHEBI, balance is the nullspace of the
stoichiometric matrix (`linalg/`), the rate constant carries
`Knowledge<_, Variance>`. **No general-purpose language checks chemical balance
as a type error** — this is frontier.

**This closes the founder's original question at its sharpest:** each area's own
semantics *is* its conservation law. The semantics of a chemical equation *are*
its balances; of a physical equation, its dimensional structure plus uncertainty
propagation. Writing the equation directly is what lets the compiler enforce the
discipline's conservation law at the point the scientist writes it.

### 5.1 Builder vs. sceptic → synthesis

- **Builder:** `reaction { … }` / `law { … }` blocks; parser → AST; checker runs
  balance/dimension.
- **Sceptic:** surface syntax is the most expensive, most-contended change
  (parser; syntax changes need codex-2 sign-off); `->` already means
  function-type and match-arm (grammar collision); the substance is the *checked
  invariant*, not the glyph.
- **Synthesis (same as for ontology):** the substance is reachable **now**, with
  zero parser risk, via a refinement-typed constructor whose typecheck fails on
  imbalance:

  ```sounio
  let combustion = crn([(2, H2), (1, O2)] -> [(2, H2O)])  // E<balance> if atoms don't close
  ```

  The glyph notation is a separate, later, optional affordance that earns its
  keep only if it reads like the scientist's page and does not collide. Ship the
  invariant first (it is the novelty and the dissertation value); the glyph is
  polish.

### 5.2 The physics prize

Writing `E² = (pc)² + (mc²)²` over `Knowledge<f64, Variance>` values and having
the compiler **derive** the propagated uncertainty eliminates the hand-threaded
`(uE, up)` of `sr.sio:124`. That is **Contribution 1 (GUM-through-ODE) realised
as a language feature** rather than library boilerplate — the written equation
*is* the propagation, certified by the ontology tag N5 introduces.

---

## 6. The unifying law (the spine)

N5 and equations are instances of one move Sounio is discovering in several
places (three landed as founder rulings before this note):

| Instance | The author asserts | The compiler holds the evidence |
|---|---|---|
| **units** | `mg + mL` | dimension — refused |
| **moduli** | sum in `[0, p)` | the modulus is in the type (a refinement carries the algebra) → demands the reduction |
| **provenance** (R-ORIGIN) | a computed value claims it was measured | provenance — refused |
| **N5** | GUM variance `+` truncation band | uncertainty ontology — refused |
| **equations** | `2 H2 + O2 -> 2 H2O` / `F = m·a` | conservation law / dimensional homogeneity — refused if violated |

One law: **assertion without evidence is a type error.** Domain semantics live in
the *types*, governed by one access effect (`Epistemic`) plus composition rules on
the type tags — not a proliferation of per-area effects.

---

## 7. Open forks (next pull)

1. **Measure the load-bearing assumption for N5:** open `TypeEntry` and the
   `Knowledge<…>` generic in the checker — does the ontology slot exist-but-
   discarded (mechanical) or not-exist (research-grade to thread through
   unification)? This number decides N5's whole cost.
2. **Design the ontology product `Variance ⊗ Interval`** — the aleatory×epistemic
   reconciliation, the dissertation line, the thing no competitor represents.
3. **Build the chemical invariant** — the refinement-typed `crn(...)` constructor
   that refuses imbalance via the stoichiometric nullspace over CHEBI-grounded
   species (no parser change; most legible demo).
4. **The physics equation as propagation** — how a `law { }` over
   `Knowledge<_, ontology>` derives uncertainty automatically, folding in
   particle physics' quadrature-under-condition rule.

## 8. Fork 1 measured — N5 cost verdict: mechanical-to-medium, NOT research-grade

The load-bearing assumption ("does the ontology slot exist-but-discarded, or
not-exist?") was measured against `self-hosted/check/`.

**The slot pattern exists and threads through unification.** `TypeEntry`
(`check/types.sio:139`) already carries index-tags that flow through `compat`:
`unit_id` (units — proven), `refinement_id` (moduli-style refinement), `algebra_kind`
/`clifford_p/q`, `epistemic_meta_id` (Contest/Policy), `ontology_id` (**domain**
ontology — ChEBI/OWL, backed by `ontology_side_table_cache.sio` which loads
ontology files), and `knowledge_epsilon: f64`.

**The join site and the rejection pattern already exist.** `compat.sio` `TyKnowledge`
arm (~230) compares `knowledge_epsilon` (subsumption ε1≤ε2). The `TyModelFamily`
arm (`compat.sio:250`) already does `a.epistemic_meta_id == b.epistemic_meta_id`
— "tags must match or incompatible" — which is exactly the shape of
`check_ontology_join`, already in the tree, just applied to ModelFamily.

**The gap is precise:** `knowledge_epsilon` is *magnitude only*, and is overloaded
(`types.sio` comments: `transport_confidence_milli`, `diagram_confidence_milli`,
`fairness_slack_milli`, `grade ε∈[0,1]`). Two `Knowledge` with equal ε but
different calculi (GUM variance vs truncation band) are indistinguishable today.
The ontology KIND is missing.

**Validity/provenance are confirmed checked-then-discarded:**
`knowledge_meta_from_ty` (`epistemic.sio:496`) hard-codes `validity_always()` and
`PROVENANCE_KIND_DERIVED` regardless of the type — the CEI plan's N3 claim, verified.
But N5 needs the ontology KIND, not those; cleaner to add as its own field.

Cost:

| Piece | Cost | Note |
|---|---|---|
| New field `uncertainty_ontology_id: i64` in `TypeEntry` (default −1) | mechanical | same site as the ~6 `ontology_id: -1` inits; do NOT reuse `ontology_id` (ChEBI/domain — collision) nor `epsilon` (overloaded) |
| Lattice + `ontology_join(a,b)` | mechanical | mirrors the ModelFamily `tag==tag` arm |
| Set the tag at constructors (`measure`→Variance, interval→Interval …) | medium | locate constructor sites, write the tag |
| Extend `TyKnowledge` compat arm + binary-op join (`check.sio:18862`) | medium | site already exists; add KIND check beside the ε check |

**Verdict:** no new type parameter, no new unifier plumbing from scratch — it is
"add one more `unit_id`." N5, and by extension forks 2–4, fall on the
mechanical/medium side, not the research horizon. Caveat: because
`knowledge_epsilon` is overloaded across Transport/Diagram/Fairness/Grade kinds,
the ontology tag must be a **separate** field, with those constructors defaulting
it to −1/⊥.

## 9. Fork 2 designed — the ontology product `Variance ⊗ Interval`

The representation already exists, orphaned from the type system:
`stdlib/epistemic/knightian.sio:65` defines

```
pub struct PBox { lo_mean: f64, hi_mean: f64, variance: f64, confidence: i64 }
```

which **is** `Variance ⊗ Interval`: `[lo_mean, hi_mean]` is the epistemic
(Knightian/interval) axis on the mean (parameter uncertainty, reducible with
data); `variance` is the aleatory axis (inter-individual variability, irreducible).
`pb_add` propagates the two axes by *different* laws — interval extension on the
means (`lo+lo, hi+hi`), GUM on the variance (`var+var`). `pb_dominates` is the
containment order; `pb_dispersion = gap + 2·sd` is the conservative both-axes
collapse (final gate only). All built; disconnected from the type system.

### 9.1 The lattice has a partial monoidal product ⊗

Beyond ⊥=Exact, ⊤=Approx and the atoms {Variance, Interval, Band, OrderDependent},
there are selected **product objects**. `Variance ⊗ Interval = ProbBox` sits above
both atoms, below ⊤.

- **Join** `Variance ⊔ Interval = ⊤` — lossy (what N5 rejects).
- **Product** `Variance ⊗ Interval = ProbBox` — lossless (the reconciliation).

⊗ is **partial**: `Variance ⊗ Band` is undefined (a variance and a systematic bias
do not form a coherent p-box; they combine by the physicist's
quadrature-under-condition rule — a different reconciliation); `Variance ⊗
OrderDependent` is the affine-octonion case (associator-as-variance, N=3). **The
boundary between "has ⊗" and "no ⊗ → E" is the scientific content.**
`check_ontology_join`: equal→ok; ⊥→the other; ⊤→down-cast; pair with ⊗ defined
*under a declared reconciliation op*→⊗; else→E.

### 9.2 Three type-theoretic claims (the novelty)

- **(a) Axis non-interference.** The two components propagate by different algebras;
  the type forbids cross-axis operations — `a.variance + b.hi_mean` is ill-typed.
  No existing system types this separation.
- **(b) Collapse is non-commutative, and the order is typed.**
  `resolve_epistemic ∘ resolve_aleatory ≠ resolve_aleatory ∘ resolve_epistemic`.
  The correct nesting (aleatory inner, epistemic outer — the 2D-MC convention) is
  the only one the type permits without an explicit `assume`; the wrong order must
  be written out and carries an obligation. Reporting a single SD *is* an implicit
  collapse, usually in the wrong order. The type makes it explicit and ordered.
  (Echo of the NonAssoc theme: order matters in two places — algebra and
  epistemic/aleatory collapse.)
- **(c) Containment is the soundness certificate** (already in `pb_dominates`).
  Every ProbBox op preserves containment of the true CDF — the p-box soundness
  lemma, which is the CEI-N1 certificate for the `(Variance⊗Interval)` handler:
  "handler soundly realises the effect" = "output p-box contains the true family."
  **Unconditional** (unlike GUM's curvature-conditional soundness) — the strongest
  Lean obligation in the programme.

### 9.3 The clinical demonstration (the dissertation line)

Vancomycin / rapamycin — "is this patient therapeutic?":

| Representation | Answer | Problem |
|---|---|---|
| point estimate | "therapeutic" | ignores all uncertainty |
| variance only (GUM) | "84% therapeutic" | *which* 84% — population or my ignorance? fused |
| **ProbBox** | "**between 71% and 92%** of patients like this are therapeutic" | the width [71,92] is epistemic (shrinks with one TDM sample); the position spread is real IIV (does not shrink) |

The `WARN` fires when the epistemic lower bound crosses subtherapeutic even though
the point estimate is therapeutic — the `project_vancomycin_auc_epistemic` result,
now a type-level theorem. It separates "get more data" (shrinks the interval) from
"irreducible patient variation" (the variance) — clinically actionable.

### 9.4 Competitor kill

GUM/Measurements.jl → one variance, axes fused. Uncertain⟨T⟩ → MC, one
distribution, no epistemic axis. Stan/Pyro → *can* model it (hierarchical) but at
inference time; a posterior fuses unless hand-structured, and fusing-wrong is never
a type error. Ferson p-boxes → this *is* the math, but a library — no type, no
collapse-order checking, no compile-time WARN. **Sounio: `Knowledge<T, ProbBox>`
as a type, collapse order compiler-checked, formation = the reconciliation handler
with a containment certificate. The only system where fusing the two axes wrong is
caught at compile time.**

### 9.5 Honest boundaries

- `PBox` models only Normal-with-interval-mean (parametric). General p-boxes
  (arbitrary CDF bands) are broader; start parametric (matches `PBox`), generalise.
- **Correlation:** `pb_add` assumes independent variance addition. Under *unknown*
  correlation the aleatory axis needs Fréchet bounds; the correlation assumption
  should be **another tag** on the type — a real gap.
- **`pb_decay(c) = c*99/100` per op is heuristic, not derived.** Per principle #6
  (values derivable, not retrofitted), the confidence-decay rate needs a derivation
  or it is drift.

## References (in-tree)

- `self-hosted/check/effects.sio` — effect registry (ids 0–22)
- `self-hosted/compiler/lean_single.sio:14837–14909, 25117–25118` — seed bitmask, `Approx` = bit 18
- `self-hosted/check/check.sio:5432, 19273` — `.value requires Epistemic` site (join hook)
- `stdlib/chemistry/kinetics.sio:102, 128` — `EpistemicReaction`, `BigCRN`
- `stdlib/chemistry/ontology.sio` — CHEBI species bridge
- `stdlib/physics/sr.sio:123–161` — invariant-mass law-in-comment, hand-threaded uncertainty
- `stdlib/metrology/` — `TypeAEval`, `CalCertificate` (ontology-in-type, done right)
- `stdlib/epistemic/knightian.sio:65` — `PBox` = `Variance ⊗ Interval` (product representation, orphaned from the type system); `interval_ieee.sio`, `montecarlo.sio` — other handler adapter targets
- `self-hosted/check/types.sio:139` — `TypeEntry` (the tag-slot host); `self-hosted/check/compat.sio:225,250` — the join site (`TyKnowledge` ε; `TyModelFamily` tag==tag template)
- `.claude/plans/silly-enchanting-newt.md` — CEI program (handlers as certified interpreters; N1–N4)
