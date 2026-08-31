<!-- docs:meta
topic_id: repo.docs.research.paper-a-antigarbling-skeleton-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-antigarbling-skeleton-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — Abstract + skeleton

**Working title:** *Manufacturing Precision Is a Type Error: Compile-Time
Anti-Garbling for Uncertainty-Typed Languages*

**Alt titles:** · *Noise Symbols in the Type: Rejecting Unsound Independence in
Uncertainty Propagation* · *`x + x ≠ 2x`: A Type Discipline for Correlated
Uncertainty*

**Target venues:** PLDI or OOPSLA (bug-class-plus-type-discipline shape); ECOOP as
fallback. Artifact-evaluable.

**Positioning in one line:** affine arithmetic (Comba–Stolfi; Goubault–Putot/Fluctuat)
*computes* noise-symbol correlations as an analysis output but rejects nothing;
uncertainty-typed languages (Uncertain⟨T⟩, Measurements.jl) carry uncertainty in the
type but track no source identity; **we put the noise-symbol source-set in the type and
make the independence assumption of uncertainty arithmetic a checked precondition** —
the intersection neither neighbor occupies.

---

## Abstract (draft, ~200 words)

Libraries that propagate measurement uncertainty — `Measurements.jl`, `Uncertain⟨T⟩`,
GUM implementations — assume the operands of every arithmetic operation are
independent. When they are not, the propagated uncertainty is silently *understated*:
the program fabricates precision it has not earned. We show this is not a corner case
but a shipping defect. In one production uncertainty library, `mul(x, x)` returns
`2x²·var` while `square(x)` returns the correct `4x²·var` — the same mathematical
operation, two formulas, with nothing routing `x·x` to the sound one; the correlated
`add`/`sub` pair exhibits a matching directional asymmetry (add understates, sub stays
conservative). We recast the problem through the Blackwell / data-processing refinement
order already standard in quantitative information flow: a sound uncertainty operation
is a *garbling* (information-losing); understating variance is an *anti-garbling*
(information-creating), which no correct program may do. We give a type system for an
uncertainty-typed language that carries the **noise-symbol source-set** of each value in
its type — reusing the source-identity idea of affine arithmetic, but in the type rather
than in an external analyzer — and turns the independence assumption into a **checked
precondition**: an independence-assuming operator over operands with non-disjoint (or
unknown) source-sets is rejected unless a proved-disjoint certificate holds. We prove
the core soundness criterion in Lean (kernel-checked, axiom-free): the naive scalar
operation is sound iff the operand covariance is zero, and the disjoint-support check is
conservatively sound. We implement the discipline in the Sounio compiler and show it
eliminates the defect class while accepting correlated-aware code, evaluated on a
physiologically-based pharmacokinetic model where every inter-compartmental sum shares
measured rate constants.

---

## Skeleton

Each section is tagged with its **artifact status** — `[BUILT]` (running / kernel-checked
today), `[DESIGNED]` (specified, wire authorized §26 but not yet in the checker),
`[WRITE]` (prose only).

### 1. Introduction  `[WRITE]`
- The promise of uncertainty types: carry `± σ` in the type, propagate automatically,
  catch "unit-like" errors at compile time. The hidden assumption that breaks it:
  **independence of operands**.
- The failure is *directional and invisible*: understating uncertainty produces a
  tighter, more confident-looking answer — the failure mode that never trips a test,
  because the number looks *better*.
- Lead artifact (the hook): the `mul(x,x)=2x²v` vs `square(x)=4x²v` defect, found in a
  shipping library, reproduced. Not hypothetical.
- Thesis: **manufacturing precision is a type error**, and it is the *same* kind of
  error as manufacturing information in quantitative information flow — an
  anti-garbling. We enforce its absence in the type.
- Contributions (bulleted, each mapped to a section):
  1. A characterization of the defect class + the Blackwell/anti-garbling framing (§2–4).
  2. A type discipline: noise-symbol source-sets in the type; independence as a checked
     precondition; the E230 rejection + proved-disjoint certificate (§5).
  3. Kernel-checked soundness of the core criterion (§6).
  4. An implementation and evaluation, including the in-the-wild defect and a PBPK case
     study (§7–8).
- **Honest delta up front** (one paragraph, so no referee has to extract it): we do NOT
  claim noise-symbol tracking (affine arithmetic, 30 years) nor the Blackwell soundness
  frame (QIF); we claim their *combination as a compile-time type rule in an
  uncertainty-typed language*.

### 2. The defect class, by example  `[BUILT]` (the code is real)
- 2.1 `mul(x,x)` vs `square(x)`: the two formulas, the missing routing, the exact
  understatement `2·cov`.
- 2.2 The add/sub asymmetry: correlated `add` understates (the sin), `sub` overstates
  (merely conservative) — a *directional prediction* of the frame, confirmed in source.
  Table: op × true(ρ=1) × library × verdict.
- 2.3 Why tests don't catch it: the wrong answer is the *more precise-looking* one.
- 2.4 The realistic instance: `add(auc_central, auc_peripheral)` where both derive from
  the same measured clearance — every PBPK compartment shares rate constants, so every
  inter-compartmental sum understates, which *weakens* the safety WARN a clinician
  relies on. Sets up §8.

### 3. Background  `[WRITE]`
- 3.1 Uncertainty types (Uncertain⟨T⟩, Measurements.jl, GUM): carry uncertainty in the
  type/value; **assume independence**; no source identity.
- 3.2 Affine arithmetic (Comba–Stolfi) and static analysis (Goubault–Putot, Fluctuat):
  noise symbols `x₀ + Σxᵢεᵢ`; shared εᵢ preserve correlation so `x−x=0`. **This is where
  source-identity tracking comes from** — but it lives in an external analyzer over C/Ada,
  produces an enclosure, and *rejects nothing*.
- 3.3 The Blackwell / data-processing refinement order and QIF (McIver–Morgan–Smith;
  Alvim et al., *Science of QIF*): refinement = Blackwell informativeness; post-processing
  (garbling) is monotone; anti-garbling is forbidden. **This is where the soundness
  criterion comes from** — QIF applies it to confidentiality leakage; we apply it to
  numeric uncertainty.

### 4. Anti-garbling as the soundness criterion  `[BUILT]` (Lean)
- 4.1 Definition: an uncertainty operation is *sound* iff it is a garbling
  (information-non-creating) of the true joint experiment; *understating variance* is an
  anti-garbling.
- 4.2 The scalar `ep_*` operators as a channel; the independence assumption as the
  claim "the operand experiment factorizes."
- 4.3 **Core lemma (kernel-checked, `SounioAntiGarblingModel.lean`, axiom-free):** the
  naive scalar add is sound **iff** operand covariance `⟨a,b⟩ = 0`; the understatement is
  exactly `2⟨a,b⟩`. State it; give the affine model it's proven over.
- 4.4 **The honesty knob (codex correction, keep it visible):** zero-covariance is the
  *necessary-and-sufficient* soundness condition; **disjoint noise-symbol support is
  sufficient but not necessary** (e.g. `(1,1)·(1,−1)=0` with overlapping support). The
  type check keys on disjoint support → it is **conservatively sound**, not an iff. Name
  this now; §6 and §10 return to it.

### 5. The type system  `[DESIGNED]` (prototypes `[BUILT]`)
- 5.1 Types carry a noise-symbol source-set: `Knowledge⟨T, NS⟩`, `NS ⊆ 𝒫(Sources)`.
  Three-state handle: `−1` = unknown/⊤ (conservative), `0` = ∅, `>0` = interned set.
- 5.2 Formation: `measure`-like constructors seed a fresh symbol; the source-set of a
  literal/exact value is ∅.
- 5.3 Transfer: `add`/`mul`/`merge` propagate `NS(a) ∪ NS(b)`; copy inherits;
  the lattice is `(𝒫(Sources), ⊆, ∪)` — a standard monotone dataflow (modelled on the
  in-tree escape analyzer, lattice boolean→set).
- 5.4 **The checked precondition (the heart):** an independence-assuming operator is
  well-typed only if `NS(a) ∩ NS(b) = ∅` is *proved*; otherwise **E230 —
  "anti-garbling: independence-assuming op over non-disjoint / unknown noise-symbol
  sets."** `−1` (unknown) is **never** treated as disjoint (conservative).
- 5.5 The escape valve: a **proved-disjoint certificate** discharges the precondition;
  a correlation-aware operator (`gum_s1_add_correlated(x1,x2,ρ)`, orphaned in-tree)
  accepts non-disjoint operands by taking the covariance explicitly.
- 5.6 Interprocedural: parametric call-summaries substitute caller source-sets into
  callee NS (the §14.3 convergent piece; shares the summary machinery memory
  reclamation needs). *Flag as the main engineering dependency.*
- 5.7 Sibling, not conflated: NS vs R-ORIGIN provenance — different question (*which
  sources' uncertainty* vs *measured-or-derived*), different lattice, different
  diagnostic (E230 vs E222). One boundary table (from §24 of the synthesis).

### 6. Metatheory  `[BUILT for the model; DESIGNED for the language theorem]`
- 6.1 Statement: a well-typed program contains no anti-garbling at any
  independence-assuming operator (relative to the tracked sources).
- 6.2 Proof spine: transfer functions are monotone (Kildall lfp exists/sound); the
  precondition rejects exactly the covariance-nonzero-admitting sites; the core lemma
  (§4.3) discharges soundness at each admitted site.
- 6.3 **Two honest boundaries, stated as theorems' hypotheses, not hidden:**
  - *Conservative, not complete:* disjoint-support ⇒ zero-cov (sound) but not conversely,
    so some sound programs need the certificate/explicit-cov operator (§4.4).
  - *Linear fragment only:* nonlinear ops (`mul`/`div`/`square`/`sqrt`) are delta-method
    and drop second-order terms → a residual anti-garbling **even under disjoint
    support**; only "= true first-order variance" holds. Do **not** overclaim beyond the
    linear fragment. (This is the `gAddMeta_monotone` linear-fragment result; nonlinear
    is scoped future work.)

### 7. Implementation  `[BUILT prototypes; DESIGNED wire]`
- 7.1 Sounio / Madaros compiler; where NS lives (`self-hosted/check/noise_sets.sio`, a
  dedicated module; trailing `noise_set_id` field on `TypeEntry`; interned-handle
  representation, union/disjoint dereference through the module table).
- 7.2 What runs today: `noise_symbols.sio` (carrier + sound add), `ns_dataflow.sio`
  (value-graph fixpoint flagging shared-source adds), `ns_contract.sio` (five acceptance
  controls incl. the sabotage-causality witness) — all souc-green.
- 7.3 The wire (N1 representation → N2 seed/union/summary → N3 E230 at `ep_add`/`ep_mul`
  → N4 gate + regression). Report status honestly at submission time.

### 8. Evaluation  `[BUILT for defect + causality; DESIGNED for wired benchmarks]`
- 8.1 **The defect in the wild** (RQ1: is the problem real?): the `mul`/`square`
  discrepancy + add/sub asymmetry, reproduced, quantified (`2·cov`).
- 8.2 **Causality of the check** (RQ2: does the type rule cause the rejection?): the
  same-source-built sabotage witness — disabling *only* NS propagation makes the `x+x`
  refusal vanish while unrelated refusals (E222) survive. Rules out coincidental
  rejection.
- 8.3 **Precision — false positives/negatives** (RQ3): rejected-but-sound cases needing
  the certificate (the conservative gap, §4.4); the correlation-aware escape valve.
- 8.4 **Case study** (RQ4: does it matter?): the vancomycin/rapamycin PBPK model —
  every inter-compartmental sum shares measured clearance; show the sound propagation
  vs the naive one and the effect on the therapeutic-range WARN.

### 9. Related work  `[WRITE]`
- Affine arithmetic & zonotopic static analysis (Comba–Stolfi; Goubault–Putot; Fluctuat;
  perturbed affine arithmetic) — *closest on source-identity*; delta = in-the-type +
  rejection + certificate, not an external enclosure.
- QIF & the Blackwell refinement order (McIver–Morgan–Smith; Alvim et al.) — *closest on
  the soundness frame*; delta = numeric uncertainty propagation vs confidentiality, and
  a static type rule vs a leakage measure.
- Uncertainty-typed languages & libraries (Uncertain⟨T⟩, Measurements.jl, GUM tools,
  Ferson p-boxes) — *carry uncertainty, track no source*; delta = the whole point.
- Information-flow / taint types, gradual & refinement types — the type-machinery
  neighbors; NS is a set-valued IFC-like lattice with a covariance-soundness reading.
- Rounding-error type systems (NumFuzz, Bean, type-based rounding-error analysis) —
  adjacent "numeric error in the type" line; different invariant (roundoff vs
  correlation-soundness).

### 10. Limitations & honesty  `[WRITE]`
- Conservative default (disjoint ⇏ complete); the certificate burden.
- Linear fragment only; nonlinear residual (delta-method).
- Interprocedural summaries are the load-bearing engineering piece; intraprocedural NS
  loses cross-call sharing → must default to assume-sharing (the *opposite* of the
  library's assume-independent).
- Unknown-correlation case beyond zero/one needs Fréchet bounds — another tag, future work.
- The narrow novelty claim is asserted *only* in the "compile-time enforcement in a type"
  form the two prior-art gates (§ prior-art memo) leave standing.

### 11. Conclusion  `[WRITE]`
- Manufacturing precision is a type error; it is anti-garbling; the type carries the
  noise-symbol source-set that makes independence a checked precondition. The number that
  looks *too good* is the one the compiler now refuses to print.

---

## Grounding index (what backs each claim)

| Claim | Artifact | Status |
|---|---|---|
| Defect is real | `stdlib/epistemic/knowledge.sio:112,154` (`ep_mul` vs `ep_square`) | source |
| naive add sound ⟺ zero-cov; gap = 2·cov | `SounioAntiGarblingModel.lean` | kernel-checked, axiom-free |
| disjoint ⇒ zero-cov (conservative, not iff) | ibid. `gap_zero_iff_disjoint_witness` + codex correction | proven + noted |
| NS carrier + sound add runs | `docs/research/sounio/noise_symbols.sio` | souc-green |
| NS dataflow flags shared-source add | `docs/research/sounio/ns_dataflow.sio` | souc-green |
| Rejection is caused by NS (not coincidence) | `ns_contract.sio` control 5 (sabotage) | souc-green |
| Correlated operator exists (escape valve) | `stdlib/epistemic/gum_supplement1.sio` (`gum_s1_add_correlated`) | in-tree, orphaned |
| Compiler wire | §26 plan (E230, `noise_sets.sio`, N1–N4) | authorized, not yet built |

## Prior-art attribution (cite up front, §1 and §9)
- Comba & Stolfi 1993 — affine arithmetic (noise symbols).
- Goubault & Putot, VMCAI 2011; *Perturbed affine arithmetic* (arXiv:0807.2961) — Fluctuat,
  zonotopic static analysis tracking correlations.
- McIver, Morgan, Smith et al., POST 2014 — abstract channels & the robust
  (Blackwell) leakage order.
- Alvim, Chatzikokolakis, McIver, Morgan, Palamidessi, Smith — *The Science of
  Quantitative Information Flow*, Springer 2020.
- Blackwell 1953 — comparison of experiments (informativeness order).
