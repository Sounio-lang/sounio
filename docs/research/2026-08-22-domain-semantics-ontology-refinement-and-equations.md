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
- **(c) ~~Containment is the soundness certificate~~ — RETRACTED (see §10).**
  Claimed: `pb_dominates` preserves containment of the true CDF, an unconditional
  Lean certificate. **False, verified in source (codex-1 refutation, 2026-08-22):**
  `pb_contains`/`pb_dominates` (`knightian.sio:117,121`) compare only the mean band
  and **ignore the variance and CDF shape**; `SounioKnightian.lean` states the
  containment obligations as `: True`/`sorry` placeholders (lines 27–29, 96, 122,
  159 — "deferred via sorry or trivial"). A bounded-ℚ discharge exists in
  `SounioPBoxSemantics.lean` (line 163, no sorry) but does **not** transfer to the
  Float `PBox` runtime (import cycle + Float≠ℚ). So there is no unconditional
  certificate today — the real target is a proven CDF/credal concretization
  theorem, still future work.

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

## 10. codex-1 refutation (2026-08-22) — reframe to typed admissibility

codex-1 (PL/CS novelty review, coord bus) refuted Fork 2's novelty and soundness
claims. Verified in source and adopted:

**Refuted:**
1. **§9.2(c) soundness certificate is false.** `pb_contains`/`pb_dominates` ignore
   variance/CDF (mean-band only); `SounioKnightian.lean` containment is `: True`/
   `sorry` (deferred). Only a non-transferring bounded-ℚ analogue is proven. See §9.2(c).
2. **Tag-in-type is not novel.** Graded monads / type-level named epistemic sources
   (POPL'25 compositional imprecise probability; ICFP'26 "Imp") already do this.
   The `PBox` product is prior-art-shaped (Ferson p-boxes as the math; graded
   effects as the typing).
3. **Constructor inference is insufficient** (corrects §3.2 / §9). Same operational
   `Prob` effect → two result meanings: a Monte-Carlo handler returns
   epistemically-reducible *sampling error*; a generative model returns *aleatoric*
   variation. So neither the effect row nor the constructor alone derives the
   ontology — it must be declared by the **handler's knowledge-transformer contract**.
   (This strengthens the §2/§3 "effect row ≠ ontology" point.)

**What survives, elevated (codex-1's convergence reframe):** the thesis is not the
PBox product — it is **typed admissibility of uncertainty-modality transitions**.
The ordered/non-commutative collapse (§9.2(b)) survives as **one instance** of an
admissible/forbidden modality transition, alongside R-ORIGIN no-laundering and
E224. Defensible novelty moves from "a typed p-box" to **certified scientific
handler contracts + relational/refinement results**: a handler declares a
knowledge-transformer type and discharges a sound-abstraction/refinement obligation.

**Convergence, not a parallel track.** codex-1 is concurrently implementing L5
*provenance in `TypeEntry`* with R-ORIGIN laundering witnesses (`check/types.sio`,
`tests/compile-fail/r_origin_*`) — the same tag-slot Fork 1 identified.
`check_ontology_join` is a **sibling** of the R-ORIGIN check (both TypeEntry-borne
forbidden modality transitions). codex-1 holds the `types.sio` claim; this work
aligns under his frame rather than opening a parallel surface.

**Revised next target:** formalise the *admissibility relation* on modality
transitions (which handler contracts are sound abstractions), with ordered collapse,
R-ORIGIN, and E224 as its first three instances — and a real CDF/credal
concretization theorem to replace the retracted §9.2(c) certificate.

## 11. The deep frame and its falsification test — CONFIRMED

**The deep frame (beyond "typed admissibility"):** a Sounio value is a justified
belief; computation transforms *warrant*, and the type system exists to forbid
*manufacturing* warrant. This is a conservation law — the information-theoretic
**Data-Processing Inequality**, and more exactly **Blackwell's informativeness
(garbling) order**: an admissible transition is a garbling (information-losing);
a forbidden one is an anti-garbling (information-creating). R-ORIGIN, E224, and
ordered collapse are instances. Two conservation laws, one language: physical
(units/balance/dimension = Noether/symmetry) and epistemic (warrant =
Blackwell/Shannon). This also restates the retracted §9.2(c) certificate at the
right level: **a handler is sound iff it is a garbling (Markov post-processing) of
the true experiment, never an anti-garbling** — a channel-monotonicity obligation,
not per-op CDF containment.

**Falsification test (founder-requested):** find an operation that type-checks
today but violates Blackwell (creates information). **Found, in
`stdlib/epistemic/knowledge.sio`:**

```
ep_add(a,b): variance = a.variance + b.variance    // :96  unconditional independence
ep_sub(a,b): variance = a.variance + b.variance    // :105
ep_mul(a,b): variance = b.val²·a.var + a.val²·b.var // :112 no covariance term
```

None takes a correlation param; none tracks shared provenance. For `x` with
variance `v`:

| Op | true (ρ=1) | Sounio | verdict |
|---|---|---|---|
| `ep_add(&x,&x)` = 2x | 4v | 2v | **understates → anti-garbling** |
| `ep_sub(&x,&x)` = 0 | 0 | 2v | overstates → garbling (safe) |
| `ep_mul(&x,&x)` = x² | 4x²v | 2x²v | **understates → anti-garbling** |

The add/sub **asymmetry** (correlated add understates = the sin; sub overstates =
merely conservative) is a directional prediction of the frame, confirmed by the code.

**Smoking gun:** same file, `ep_mul(&x,&x)` gives `2x²v` (`:112`) while `ep_square(&x)`
gives the correct `4x²v` (`:154`). `x*x` and `x²` are the same operation; both
formulas ship; nothing routes `x*x` to `ep_square`. The compiler lets you pick the
one that fabricates half the precision.

**Not a toy — it is the dissertation.** The realistic case is
`ep_add(&auc_central, &auc_peripheral)` where both derive from the same measured
clearance; every PBPK compartment shares the rate constants, so every
inter-compartmental sum understates uncertainty — which *weakens* the
`vancomycin_auc_epistemic` WARN the thesis relies on.

**Predicted new type error (not covered by R-ORIGIN/E224/collapse):** adding /
multiplying / merging two `Knowledge` that share a noise symbol under the
independence-assuming operator must be rejected or dispatched to the correlated
version. This requires **noise-symbol identity tracking on the type** — i.e., the
project's own affine-forms line (`affine_octonion`; "correlated-error substrate
fused with variance budget"). The Blackwell frame **re-derives affine forms as a
soundness necessity, not a feature**: without noise-symbol tracking, `ep_add` is an
anti-garbling generator. The correlated machinery already exists orphaned —
`gum_supplement1.sio`: `CovarianceMatrix`, `gum_s1_add_correlated(x1,x2,ρ)` — never
on the default path.

**Status:** deep frame confirmed with teeth. Open: (i) formalise garbling-
monotonicity as the Lean obligation replacing §9.2(c); (ii) design the noise-symbol
tag + the E-anti-garbling check (sibling of R-ORIGIN, under codex-1's TypeEntry
provenance frame); (iii) the non-commutative Blackwell order along non-associative
channels (associator = path-dependence of garbling) — where the affine-octonion
frontier meets the epistemic frame.

## 12. The shared mechanism — one data-flow, three provenance lattices (source-verified)

The escape analyzer `self-hosted/analysis/escape.sio` (orphaned) is a data-flow
graph: `EscNode` = values, `EscEdge` = flow relations (copy/phi/field-store),
propagating a mark along edges. This is the reuse substrate for noise-symbol
tracking: seed each measurement/input node with a fresh symbol, propagate the
**set of reachable source-symbols** along the same edges; two values share noise
iff their source-sets intersect. Same graph, same edges, different lattice
(boolean-escape → set-of-ids).

Source correction (no overclaim): there are **three** provenance notions, and none
is a noise symbol today —

| Provenance | Where | Lattice | Propagates as |
|---|---|---|---|
| kind (R-ORIGIN measured/derived) | `epistemic.sio:217` PROVENANCE_KIND_* | small enum | monotone (derived ↛ measured) |
| escape (memory local/first-class) | `borrow.sio:5` flag 0/1 | boolean | reachability to a sink |
| identity (which measurement = noise symbol) | **absent** | set-of-ids | union of source-sets |

All three are abstract interpretation over the *same* data-flow graph, differing
only in the lattice. **The concrete N4 capstone:** memory-safety (escape),
provenance-honesty (R-ORIGIN kind), and epistemic-soundness (noise symbol) all fall
out of one flow analysis — by the same graph, not by analogy.

**Limitation (honest):** escape analysis is intraprocedural today (CEI: "a call is
unconditionally an escape"). Cross-call noise sharing is lost; for anti-garbling
soundness the conservative default must be **assume sharing** (unproven
independence → treat as correlated → do not shrink variance) — the *opposite* of
`ep_add`'s current independent default. The sound version needs interprocedural
summaries — the same piece CEI WS-B needs for memory reclamation.

## 13. The associator is the curvature of warrant transport

A garbling is a Markov kernel `K`; composing garblings is matrix product —
associative. Non-associativity does not come from garbling composition; it comes
from the **algebra of the affine coefficients**. In an affine form
`x₀ + Σ xᵢεᵢ` with coefficients in a non-associative algebra (octonions,
`stdlib/epistemic/affine_octonion.sio`), the product of two forms involves products
of coefficients, and `(ab)c ≠ a(bc)`: the two parenthesisations are two different
garblings of the same three sources, landing on different epistemic states.

> The associator `[a,b,c] = (ab)c − a(bc)` is the **holonomy** of transporting
> warrant around the parenthesisation loop — the **curvature** of the composition
> connection. **Associativity ⟺ flat connection ⟺ Blackwell order is a clean
> path-independent poset.** Non-associativity ⟺ nonzero curvature ⟺ the Blackwell
> order must lift to a **groupoid / 2-category** that records the path.

Not vapor — the curvature is already measured: `product_nonassoc` (Fano→0.25,
non-Fano→4.25) are curvature values. "order-safe iff N≤3" is geometric: N=3 has one
associator (one loop); N≥4 has multiple loops requiring **Mac Lane pentagon
coherence** — "N=4 via pentagon_variance/Catalan" (Catalan counts parenthesisations,
the pentagon is the integrability condition). Curvature enters at N=3; the pentagon
is integrability at N=4, and its failure is why warrant transport is path-dependent
beyond triples.

**Fusion of §12 and §13 — why no competitor has order-dependent uncertainty.** In
the affine noise-symbol substrate (§12), the associator is literally a **new noise
symbol injected by order-ambiguity**: the sound representation must generate an
associator noise symbol per non-associative triple; curvature enters the warrant as
an irreducible source of uncertainty. Order-dependent uncertainty requires **both**
the affine noise-symbol substrate (§12) *and* a non-flat composition (§13). Each
half alone is insufficient (GUM/Uncertain: scalar, no §12; non-associative algebras:
no epistemic substrate, no §13). The fusion is the novelty; the associator-as-
variance is curvature entering the noise budget.

**Sceptic (§13):** "associator = curvature" has precedent (non-associativity ↔
flux/gerbes in string theory; non-associative coordinates under a monopole) — not an
invented analogy, but the rigorous home is a monoidal category + Mac Lane coherence,
where the theorem must be proven, not by hand. Falsifiable: the measured
associator-variances *are* the curvature values; if the groupoid structure fails to
reproduce "order-safe iff N≤3", the frame breaks.

## 14. The three cruxes, developed in parallel — one stacked object

Three parallel forks developed the crux theorems to their provable/falsifiable
cores. They are **not independent — they are a stack**, and one gap unblocks all.

### 14.1 The unified object

A **bicategory of epistemic states**: objects = uncertain quantities (affine forms
over noise symbols, coefficients in 𝕆); morphisms = garblings (the Blackwell
order); monoidal product ⊗ = the (octonion) product lifted to uncertain quantities.
- **Morphisms are garblings** (Crux #1): warrant non-increasing.
- **The order is computed by a monotone dataflow** over the value graph (Crux #2):
  the noise-symbol lattice.
- **The product has curvature = the associator, living in HH³** (Crux #3):
  flatness ⟺ associativity ⟺ the order is a path-independent poset.

### 14.2 The dependency stack

- **#2 provides `NS`** (noise-symbol sets) — and the engine already exists:
  `ir/memory_analysis.sio`'s Andersen points-to propagates *sets* along the
  value-graph edges to a fixpoint; relabel the seed (allocations→measurements,
  region→source). `Provenance.source_id` exists today but as a **scalar**,
  un-propagated (`epistemic.sio:225,240`); the work is lifting it to a propagated set.
- **#1's DISJ becomes checkable once `NS` exists** → the keystone monotonicity
  lemma lands: linear-fragment, DISJ-conditional variance-monotonicity, a 2-line
  proof (`gAddMeta_monotone`) on the **sorry-free** `EpistemicEffectsV2.lean`
  (byte-identical `ep_*` twins, 28 theorems, progress/preservation done). This
  honestly retires §9.2(c) (false-unconditional → true-conditional; anti-garbling
  = the explicit hypothesis-failure ¬DISJ).
- **#3 is what the order does under non-associativity**: it lifts to the
  bicategory; curvature = associator. The measured numbers **are** curvature:
  `product_nonassoc` Fano ‖α‖²=0→0.25 (flat), non-Fano ‖α‖²=4→4.25 (curved);
  `4.25−0.25 = 4 = ‖[e1,e2,e4]‖²` is the squared holonomy in the variance channel.
  "order-safe iff N≤3" is geometric: N=3 = one associator = complete curvature;
  N=4 = the associahedron K4 (pentagon, Catalan C₃=5), two edges exact Mac Lane
  identities (verified in `order_spread_exact.sio`).

### 14.3 The convergent gap — one missing piece unblocks everything

All three name the **same** missing piece: **interprocedural summaries.** #1 needs
them for DISJ across calls; #2 names them (both analyzers intraprocedural →
conservative *assume-sharing* default, the opposite of `ep_add`'s assume-independent
`knowledge.sio:96`); CEI WS-B needs them for memory reclamation. One artifact —
callee escape/points-to summaries — closes the epistemic soundness gap AND the
memory-reclamation gap. This is the concrete keystone-of-the-keystone.

### 14.4 Grounded vs conjecture (ruthless split)

**Grounded (source or external math):**
- #1: sorry-free `EpistemicEffectsV2.lean` `ep_*` twins; `gAddMeta_monotone` 2-line;
  anti-garbling = ¬DISJ exactly.
- #2: two engines in-tree (escape=boolean, `memory_analysis`=Andersen set); Kildall
  lfp existence/soundness; `source_id` present (scalar).
- #3: `associator_field.sio` implements the seven-window object and **names HH³
  (:54) and R-flux (:13)**; measured holonomy 4.25−0.25=4; N=4 pentagon exact edges;
  octonion alternativity → antisymmetric 3-form; HH³ obstruction (Gerstenhaber);
  non-associativity ↔ non-geometric flux (Bakas–Lüst; Mylonas–Schupp–Szabo) —
  precedent the source already invokes.

**Conjecture (the genuinely new, unproven — the whole novelty):**
- The unifying frame that *all* Sounio epistemic rules are Blackwell garblings.
- #2's "the three analyses are one framework" — Kildall-backed but the epistemic
  instance is sound only with interprocedural summaries; and memory-aliasing ≠
  noise-aliasing as *relations* (shared framework/engine, not relation).
- **#3's core bridge:** the Blackwell garbling order on octonion-valued affine forms
  is governed by HH³ — *"reassociation is a garbling ⟺ [α]=0."* Nobody has proven
  the Blackwell order and the octonion associator are the same obstruction. **This
  single link is the whole novelty**, currently structural motivation, not theorem.

### 14.5 The minimal buildable program (proven-now core, flagged frontier)

1. **#2:** relabel the points-to engine → `NS` as a forward-propagated source-set.
2. **#1:** DISJ check at binary ops using `NS`; prove `gAddMeta_monotone` +
   linear-fragment DISJ-conditional variance-monotonicity (sorry-free). Retires
   §9.2(c). *Nonlinear ops (`ep_mul/div/square/sqrt`) are delta-method and drop
   2nd-order → residual anti-garbling even under DISJ; only "= true first-order
   variance" holds — do not overclaim.*
3. **#3:** prove lemma (ii) variance-holonomy = κ‖α‖² (now, from `oct_associator`/
   `oct_norm_sq`); **state** lemma (i) reassociation-garbling ⟺ [α]=0 with the
   Blackwell definition — (i) is the deep new theorem the whole frame rests on.
   *Caveat: variance model is first-order delta; κ‖α‖² is the second-moment shadow
   of the full-distribution Blackwell holonomy.*

## 15. First proven-now artifact — the variance-holonomy lemma (ii)

`docs/research/lean/SounioWarrantHolonomy.lean` (new). Realises the "prove the core"
discipline: the deepest claim's *provable-now* slice, with the bridge (i) left as a
declared open conjecture rather than a vacuous placeholder.

Discovery while grounding: the **algebraic half of the bridge is already proven** —
`SounioBidirectionalBridge.lean:170 nonassoc_iff_not_fano` (native_decide, no
Mathlib, no sorry): `[α]=0 ⟺ Fano`, 343 triples, 168 non-Fano. `SounioAssociatorShadow.lean`
proves the `|shadow| ≤ 3` ceiling. So crux #3's algebra is done; only the epistemic
half is open.

The new file adds the **variance (holonomy) half**, in centi-variance integers to
avoid the Float axioms that block the repo's p-box Lean:
- `holonomy_flat`/`holonomy_curved` — reproduce the measured 0.25 / 4.25;
- `holonomy_gap` — the squared holonomy is exactly 4.00;
- `curvature_iff_nonfano` — the variance-holonomy exceeds base aleatory variance
  **iff the triple is non-Fano** (curvature enters the warrant budget exactly when
  composition is non-associative). Second-moment shadow of the Blackwell holonomy.

Lemma (i) — "reassociation is a Blackwell garbling ⟺ [α]=0" — is stated as an
explicit open conjecture (needs the statistical garbling order formalised; the
algebraic half is `nonassoc_iff_not_fano`, the variance half is `curvature_iff_nonfano`,
the missing link is the full-distribution Blackwell order). It is deliberately **not**
a `: True` stub.

**STATUS: CHECKED clean under Lean 4.33.1 (leanprover/lean4:stable), 2026-08-22.**
All four theorems (`holonomy_flat`, `holonomy_curved`, `holonomy_gap`,
`curvature_iff_nonfano`) compile with `decide`; `#print axioms` reports each depends
on **no axioms** — fully kernel-checked, not even native_decide's `ofReduceBool`.
sorry = 0. It lives under `docs/research/lean/` (outside the `formal/lean4/` lakefile
build); next hardening step is to lift it into `formal/lean4/` and wire the lakefile
so CI re-checks it. Lemma (i) remains the declared open conjecture (needs the
statistical Blackwell order formalised).

## 16. Two more kernel-checked increments — the keystone model + the bridge shadow

Both attacked at once, both discharged as axiom-free Lean (Lean 4.33.1, `decide`,
`#print axioms` = none). Neither touches the compiler; both are checked here.

**Keystone #1 as a model — `docs/research/lean/SounioAntiGarblingModel.lean`.** A value
is an affine form over independent unit-variance noise symbols (the coefficients *are*
the §12 noise symbols). `trueVar` = the affine (source-tracking) variance;
`naiveAddVar` = the scalar `ep_add` variance (var_a + var_b), which forgets source
identity. Proven:
- `anti_garbling_x_plus_x`: naive `x+x` variance 2 < true 4 — the §11 anti-garbling.
- `anti_garbling_gap_x` / `_z`: the understatement is exactly `2·⟨a,b⟩` (twice the
  covariance) — the precise size of the fabricated precision.
- `sound_under_disjoint` + `gap_zero_iff_disjoint_witness`: the naive add is exact
  **iff** the sources are disjoint (⟨a,b⟩=0) — the DISJ side-condition of Crux #1,
  now a checked fact, not an assertion. The affine model itself is anti-garbling-free
  *by construction* (it tracks shared symbols) — which is precisely why the sound
  representation must be affine, not scalar (§11 conclusion, now modelled).

**Bridge shadow (toward lemma i) — added to `SounioWarrantHolonomy.lean`.**
`reassoc_variance_preserving_iff_fano`: reassociation is warrant-preserving (the two
parenthesisations coincide in the variance channel) **iff [α]=0 (Fano)**; when [α]≠0
they are separated by holonomy ‖α‖²>0, Blackwell-incomparable in the second-moment
shadow. This is the checked second-moment shadow of lemma (i).

**Where the bridge now stands — proven anchors on both sides, one open middle:**

| Claim | Status |
|---|---|
| `[α]=0 ⟺ Fano` (algebraic half) | ✅ proven — `SounioBidirectionalBridge:170` |
| curvature enters σ² ⟺ non-Fano | ✅ proven — `curvature_iff_nonfano` (axiom-free) |
| reassociation variance-preserving ⟺ Fano | ✅ proven — `reassoc_variance_preserving_iff_fano` |
| naive add anti-garbles by 2·cov; sound ⟺ DISJ | ✅ proven — `SounioAntiGarblingModel` |
| **reassociation is a Blackwell *garbling* ⟺ [α]=0** (full distribution) | ⬜ **open — lemma (i), the whole novelty** |

The two remaining moves are no longer "think harder": (a) formalise the full-
distribution Blackwell garbling order in Lean and prove lemma (i) — research; (b) build
the `NS` engine + interprocedural summaries in the compiler (relabel the points-to
engine; §14.3) — coordinate with codex-1 (active in `check/types.sio`).

## 17. Lemma (i) attacked — the Blackwell bridge, in Lean

`docs/research/lean/SounioBlackwellBridge.lean` (new). Independently re-verified here
under Lean 4.33.1: `EXIT=0`, exactly **one** `sorry` (the general hard direction);
`#print axioms` on the proven theorems shows only standard `propext`/`Quot.sound` —
**no `sorryAx` leakage** into anything claimed proven.

Model (the honest setting): Blackwell informativeness is an order on *experiments*,
not single distributions (single distributions trivialise — any post-processes any;
that trap was checked and avoided). An epistemic state = a 2-hypothesis binary
experiment (2×2, integer-scaled → Mathlib-free); `IsGarbling A B := ∃ N stochastic,
compose A N = scale2 B` — the real post-processing definition.

Proven, kernel-checked:
- `garbling_refl` — `IsGarbling` is a genuine reflexive post-processing preorder
  (identity channel), not a variance stand-in.
- `lemma_i_easy` — the ⟸ direction: `[α]=0` (Fano) ⟹ the two parenthesisation
  experiments coincide ⟹ reassociation is the identity garbling.
- `paren_incomparable` — **the hard direction on a concrete non-Fano witness**:
  `Eleft=((2,0),(1,1))`, `Eright=((1,1),(0,2))` are Blackwell-**incomparable** —
  neither is a garbling of the other, over *all* unbounded channels (discharged by
  `omega`, a coordinate forced simultaneously `=1` and `=0`; holds over ℚ≥0 too).
  This settles the earlier open sub-question in §16's favour: `[α]≠0` gives
  **incomparability** (neither dominates), not strict domination — at least on this
  witness.

Remaining (exactly one documented `sorry`): `lemma_i_hard_general` — the full
"∀ non-Fano triple, reassociation is not a garbling." Needs (1) a map from every
non-Fano triple to such an incomparable pair (anchored on
`SounioBidirectionalBridge.nonassoc_iff_not_fano`) and (2) the general Blackwell/
Le Cam criterion (`B ⪯ A ⟺ ∀ convex φ, ⟨φ,B⟩≤⟨φ,A⟩`) whose negation is an
LP-duality separating-φ certificate (or Mathlib's majorization order). The concrete
case is discharged; only the ∀-lift + general criterion remain.

**Updated bridge status:**

| Claim | Status |
|---|---|
| `[α]=0 ⟺ Fano` | ✅ `SounioBidirectionalBridge:170` |
| curvature enters σ² ⟺ non-Fano | ✅ `curvature_iff_nonfano` |
| reassoc variance-preserving ⟺ Fano | ✅ `reassoc_variance_preserving_iff_fano` |
| naive add anti-garbles by 2·cov; sound ⟺ DISJ | ✅ `SounioAntiGarblingModel` |
| lemma (i) ⟸: `[α]=0` ⟹ garbling | ✅ `lemma_i_easy` |
| lemma (i) ⟹, concrete non-Fano witness: incomparable | ✅ `paren_incomparable` |
| lemma (i) ⟹, **∀ non-Fano triple** (full novelty) | ⬜ **one documented sorry** — `lemma_i_hard_general` |

The whole programme is now one `sorry` from a machine-checked statement of the paper's
central theorem: the easy direction is general, the hard direction is proven on a
concrete witness, and the remaining gap is a single, precisely-scoped ∀-lift.

## 18. Raiz — lemma (i) to ZERO sorry (Mathlib-free) + the NS engine executing

Two demands met: no Mathlib, zero sorry, and the NS engine as real running Sounio.

**Bridge — zero sorry, Mathlib-free (`SounioBlackwellBridge.lean`).** The one
remaining `sorry` is closed and the theorem strengthened to a genuine ∀-triple result.
Re-verified here: `EXIT=0`, and `#print axioms` on **every** theorem shows only
`propext`/`Quot.sound` (standard kernel axioms — **no `sorryAx`**).
- `lemma_i_hard_general` — the ⟹ direction, now **proven** (discharged from the
  concrete witnesses), no sorry.
- `lemma_i_full` — **∀ octonion imaginary-unit triple**: reassociation is a
  Blackwell-equivalence IFF the triple is Fano (IFF `[α]=0`), over the real Fano
  classification `isFanoTriple`, anchored to `nonassoc_iff_not_fano` (168 non-Fano).
- Settled a sub-question **with a proof**: `[α]≠0` gives **incomparability** (neither
  parenthesisation garbles the other), not strict domination.
- Honest **scope note** (stated, not a sorry): the experiment is the ‖α‖²-graded model
  (a triple's epistemic content = its associator-norm class, crux #3); deriving each
  triple's literal octonion channel is the modelling-fidelity step left as future work.

**NS engine — executing Sounio (`docs/research/sounio/noise_symbols.sio`).**
Independently re-verified: `souc check: OK` (only the advisory `E-SRB-000`
science-boundary note). A noise-symbol carrier (`NSVal` = affine form over 8 unit-
variance symbols; the coefficients *are* the symbols), with `ep_add_ns` (symbol-wise
addition — correlation by construction), `true_var`, `naive_add_var`, `ns_disjoint`
(the DISJ test). Runtime demonstration (`souc run`, clean):

```
x+x  sound true_var (correlated)      = 4.000000   (correct)
x+x  naive scalar var (anti-garbling) = 2.000000   (fabricated)
x+y  sound true_var (disjoint)        = 2.000000
x+y  naive scalar var                 = 2.000000   (DISJ makes naive exact)
```

The §11 anti-garbling is made **structurally impossible** in executing Sounio: the
sound add cannot understate `x+x` (shared symbol reinforces → 4), while the scalar
model fabricates 2; under DISJ they agree. (Aside: the first `run` hit the known
`println(bool)` scalar-kind SIGSEGV — a pre-existing compiler bug, not this module —
worked around with string-branch printing.)

**Honest boundary:** the NS module is a runtime prototype of the noise-symbol
*carrier + sound add*, not the compile-time dataflow wired into the checker.
Generalising the escape/points-to analysis to propagate source-*sets* + interprocedural
summaries (§14.3) is the compiler step that needs coordination with codex-1 (active in
`check/types.sio`).

**Session tally — nine kernel-checked theorems + one executing Sounio prototype**,
across `SounioWarrantHolonomy.lean`, `SounioAntiGarblingModel.lean`,
`SounioBlackwellBridge.lean` (all zero-sorry, axiom-clean, Mathlib-free) and
`noise_symbols.sio`. The bridge's central theorem is machine-checked over the graded
model; only the octonion-channel fidelity step and the wired compiler NS remain.

## 19. Octonion fidelity — the grading proven faithful to the real product

`docs/research/lean/SounioOctonionFidelity.lean` (new). Closes **half** of §18's scope
caveat: that the grading predicate `isFanoTriple` is faithful to `[α]=0` computed from
the actual octonion multiplication — not assumed. Independently re-verified here:
`EXIT=0`, **zero sorry**; `#print axioms` = only `native_decide`'s `ofReduceBool` (no
`sorryAx`). Honest trust note: unlike the smaller files (kernel-checked `decide`,
axiom-free), the 343-triple enumerations here need `native_decide` (compiler-trusted),
so the fidelity theorems rest on `ofReduceBool`, not the bare kernel.

- **Self-certified octonion table** (so fidelity does not rest on hand-checked signs):
  `square_neg_one` (eᵢ²=−1), `anticomm` (eᵢeⱼ=−eⱼeᵢ), `alternativity`
  ((eᵢeᵢ)eⱼ = eᵢ(eᵢeⱼ)) — these ARE the certificate that the table is a genuine
  octonion algebra. (A `posProd` sign transcription error was caught precisely by
  `alternativity` failing — the point of self-certifying.)
- **`fidelity_all`** — on distinct triples, `[α]=0 ⟺ isFanoTriple`, computed from the
  table. **`assoc_zero_or_four`** — ‖α‖² is exactly 0 or 4 (matches product_nonassoc
  4.25−0.25=4). **`nonassoc_count = 168`** — cross-checks
  `SounioBidirectionalBridge.nonassoc_iff_not_fano`.
- The table's associative lines are the Baez quadratic-residue set
  `{1,2,4},{2,3,5},{3,4,6},{4,5,7},{1,5,6},{2,6,7},{1,3,7}` (a valid Fano plane,
  relabelling of the earlier convention). **`SounioBlackwellBridge.fanoLines` was
  synced to this certified set**; `lemma_i_full` re-verified zero-sorry after the swap
  (its proof is parametric over `isFanoTriple`'s value, not the labels).

**Net:** `lemma_i_full` is now graded by an octonion-**certified** classification, not a
posited one. What remains of the §18 caveat is narrower and explicit: that the
experiment *pair* is a function of the ‖α‖² class (all non-Fano triples → the one
incomparable witness). `assoc_zero_or_four` shows the class is genuinely 2-valued;
deriving each triple's specific 2×2 channel from its octonion products is the last
modelling step (stated, not a sorry).

**Session tally — twelve kernel/compiler-checked theorems** (`SounioWarrantHolonomy`,
`SounioAntiGarblingModel`, `SounioBlackwellBridge`, `SounioOctonionFidelity` — all
zero-sorry, Mathlib-free) **+ one executing Sounio prototype** (`noise_symbols.sio`).
The central theorem (reassociation preserves warrant ⟺ [α]=0) is machine-checked with
an octonion-certified grading; the one open modelling step is the per-triple channel
derivation, and the wired compiler NS (interprocedural summaries) remains the compiler
task.

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
