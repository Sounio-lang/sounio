<!-- docs:meta
topic_id: repo.docs.research.publishable-novelty-assessment-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.publishable-novelty-assessment-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Publishable novelty — ruthless assessment (2026-08-25)

**Question (founder):** *what do we have that is publishable novelty?*

**Method:** ranked by **survival-after-prior-art**, NOT by proof status. The fourteen
kernel-checked theorems of the 08-22 synthesis establish that the claims are *true*;
they say nothing about whether they are *new*. Publishability is entirely the second
question. Two prior-art gates were run before writing this (below); both landed.

---

## The two prior-art gates (run 2026-08-25, decisive)

### Gate 1 — affine arithmetic as static analysis (kills the bare NS claim)
Noise-symbol identity tracking to preserve correlations between uncertain variables
is the **defining feature of affine arithmetic** (Comba & Stolfi 1993) and of
**Goubault & Putot's zonotopic static analysis** — the Fluctuat analyzer
(*Static Analysis of Finite Precision Computations*, VMCAI 2011; *Perturbed affine
arithmetic*, arXiv:0807.2961). Fluctuat propagates shared noise symbols through a
program precisely so that `x - x` cancels and correlated errors do not get treated
as independent. So **"Sounio tracks the source-identity of a value's uncertainty"
is ~30 years old.** It cannot be the novelty.

**What survives the gate:** Fluctuat *computes* the correlation as an analysis output
over C/Ada; it does not **reject** a program, does not put the noise-set **in a
user-facing type**, and demands **no certificate**. Uncertainty-typed languages
(Uncertain⟨T⟩, Measurements.jl, Ferson p-box libraries) do the opposite — they carry
uncertainty in the type but **do not track source identity at all**, so
`ep_add(&x,&x)` silently understates. The narrow surviving claim lives in that gap.

### Gate 2 — Blackwell order in quantitative information flow (kills the bare frame)
The QIF community **already identifies its refinement order with the Blackwell
informativeness (garbling) order**, with data-processing / post-processing
monotonicity as the standard soundness criterion: McIver, Morgan, Smith et al.,
*Abstract channels and their robust information-leakage ordering* (POST 2014); Alvim
et al., *The Science of Quantitative Information Flow* (Springer 2020). So
**"warrant transport is a Blackwell garbling and forbidden operations are
anti-garblings" is a known frame** — the DPI/Blackwell refinement discipline, which
QIF applies to confidentiality leakage.

**What survives the gate:** applying that refinement discipline to **numeric
uncertainty propagation** (variance / error budgets) as a **compile-time typing
rule** is a new *instantiation*, not a new *frame*. §11's honest status drops from
"the deep frame we discovered" to "the known QIF/Blackwell frame, newly instantiated
in an uncertainty-typed language." Still publishable — as a re-application, correctly
attributed.

---

## Three publishable claims, ranked

### CLAIM A — Anti-garbling as a compile-time typing discipline (strongest, defense-adjacent)
**One sentence:** *An uncertainty-typed language that carries the noise-symbol
source-set of each value in its type and makes the independence assumption of
uncertainty arithmetic a **checked precondition** — an independence-assuming
`add`/`mul`/`merge` over operands with non-disjoint (or unknown) source-sets is
**rejected (E230)** unless a proved-disjoint certificate holds.*

- **Novelty after both gates:** the *intersection* — noise-symbol tracking (from
  affine arithmetic) surfaced **in the type** of an uncertainty-typed language, with
  **rejection + certificate** (from the Blackwell/QIF discipline). Neither neighbor
  has the pair. Fluctuat rejects nothing; Uncertain⟨T⟩ tracks no source.
- **Grounded now:** `SounioAntiGarblingModel.lean` (kernel-checked, axiom-free): naive
  scalar add is sound **iff zero covariance**; the understatement is exactly `2·cov`.
  Running Sounio prototypes: `noise_symbols.sio`, `ns_dataflow.sio`, `ns_contract.sio`
  (souc-green, sabotage-causality witness passes).
- **Empirical hook — the defense-shaped part.** In-source defect, reproducible:
  `stdlib/epistemic/knowledge.sio` ships `ep_mul(&x,&x) → 2x²v` (`:112`) beside
  `ep_square(&x) → 4x²v` (`:154`) — the same operation, two formulas, **nothing
  routes `x*x` to `ep_square`**; the add/sub asymmetry (correlated add understates =
  the sin; sub overstates = merely conservative) is confirmed in source. This is a
  real bug **class** with a clean frame — a PLDI/OOPSLA-shaped "here is a silent
  unsoundness in uncertainty libraries, here is a type discipline that eliminates it"
  paper.
- **Honest scope:** must attribute affine arithmetic (correlation tracking) and QIF
  (Blackwell soundness) up front; the claim is the *compile-time enforcement in a
  type*, not the tracking or the frame. The narrow-claim wording in §24 is still a
  CANDIDATE per codex's prior-art gate — this memo **runs** that gate and it survives
  in the narrowed form above.
- **Status of the wire:** the compiler NS is authorized but unbuilt (§26 N1–N4, a
  hot-file Madaros sprint). Paper A can be written on the model + prototypes + defect;
  the wired compiler strengthens it but is not a precondition.

### CLAIM B — The octonion associator as a Blackwell obstruction (highest ceiling, highest risk, NOT defense-window)
**One sentence (honest form):** *We formalize a **model** in which reassociating a
non-associative (octonion-valued) uncertain product is a Blackwell-equivalence **iff**
the associator vanishes (**iff** the triple is Fano) — identifying algebraic
non-associativity with the Blackwell/QIF information obstruction.*

- **Why this is the genuinely unclaimed one:** the intersection **non-associative
  algebra curvature × the Blackwell/QIF order** is empty in the literature. Gate 2
  kills the bare Blackwell frame, but *nobody has connected the octonion associator to
  that order*. This is the one claim that neither prior-art neighbor touches.
- **What is actually machine-checked** (`SounioBlackwellBridge.lean`,
  `SounioTripleChannel.lean`, `SounioOctonionFidelity.lean`, zero-sorry): **given** an
  encoding of a triple's epistemic content as a 2-outcome experiment derived from the
  octonion **product sign**, reassociation is a Blackwell-equivalence iff Fano. The
  algebraic half (`[α]=0 ⟺ Fano`, 168 non-Fano) and the variance-holonomy shadow
  (`κ‖α‖²`) are independently proven.
- **The referee's attack, foregrounded (not buried):** the encoding is a **modelling
  choice**. So the honest paper sentence is *"we formalize a model in which the
  associator is the Blackwell obstruction,"* **not** *"we prove the associator is the
  Blackwell obstruction."* §20's own "Residual (honest)" note and codex correction #3
  (§21) are exactly this point.
- **NOT a defense-window item.** The remaining work — the general Blackwell/Le Cam
  criterion in Lean (∀-lift beyond the concrete witness class), full per-triple channel
  fidelity — is research measured in months. Target a PL/logic or math-physics venue,
  post-defense.

### CLAIM C — ProbBox: `Variance ⊗ Interval` as a type (the dissertation line, defense-shaped)
**One sentence:** *A type that carries the aleatory (variance, irreducible) and
epistemic (interval-on-the-mean, data-reducible) axes as a **product** with
axis-non-interference and a **typed, non-commutative collapse order**, demonstrated on
therapeutic-drug-monitoring where the compile-time WARN fires where the point estimate
says "therapeutic."*

- **Novelty bar here is pharmacometrics, not PL.** p-boxes (Ferson) are known math;
  the representation `PBox` already exists (`knightian.sio:65`). The contribution is
  (a) axis-non-interference **in the type** (`a.variance + b.hi_mean` ill-typed),
  (b) collapse order compiler-checked (2D-MC nesting is the only one allowed without
  an explicit `assume`), (c) the clinical WARN that **separates "get more data"
  (shrinks the interval) from "irreducible patient variation" (the variance)** —
  clinically actionable, the `vancomycin_auc_epistemic` result as a type-level property.
- **Honest kill inside the claim:** §9.2(c) containment certificate is **RETRACTED** —
  `pb_dominates` compares only the mean band, ignores variance/CDF; the Lean obligation
  is `True`/`sorry`. So the claim is **representation + typed collapse order + clinical
  WARN**, NOT a soundness theorem. `pb_decay` is heuristic, not derived. Correlation
  needs another tag (Fréchet bounds) — a stated gap.
- **Defense-shaped:** this is the dissertation's Contribution line. The bar is "does it
  change a clinical decision," and it does. Writable in the window.

---

## What is NOT publishable-novel (honest kills, so we stop chasing)

- **Noise-symbol / correlation tracking per se** — affine arithmetic + Fluctuat (Gate 1).
- **Blackwell-as-soundness frame per se** — QIF refinement order (Gate 2).
- **Cayley–Dickson erasure ladder (ker = 2^(n−1)−4)** — Moreno 1998; our contribution
  is *machine-verification*, which is a note, not a paper.
- **The fourteen kernel-checked theorems as such** — they establish *truth*, and truth
  is the price of entry, not the novelty.
- **"associator = curvature/flux"** — has precedent (Bakas–Lüst; Mylonas–Schupp–Szabo,
  non-geometric flux). The novelty is the *Blackwell* link (Claim B), not the curvature
  reading.

---

## Defense-window triage (28 days; defense 2026-09-22)

| Item | Shape | In window? |
|---|---|---|
| **C — clinical ProbBox line** | pharmacometrics contribution, representation + WARN | **Yes** — it is the dissertation line |
| **A — empirical anti-garbling defect** (`ep_mul` vs `ep_square`, add/sub asymmetry) | reproducible in-source defect + frame | **Yes** — writable as a finding now |
| A — full compiler NS wire (§26 N1–N4) | hot-file Madaros build sprint | Partial — not required for the paper; do post-defense |
| **B — associator/Blackwell bridge** | needs general Le Cam criterion in Lean | **No** — months of research; post-defense venue |

**Bottom line:** three real claims survive prior art, each with its bar named. **A** is
the strongest PL contribution (a bug class + a type discipline that kills it, grounded
in running Sounio and a kernel-checked model), **C** is the defense-window dissertation
line, **B** is the high-ceiling bet that must be stated as a *model*, not a theorem, and
does not fit the window. The one hard gate remaining before asserting A's narrow claim
in print — adversarial prior art on affine arithmetic and QIF — is **run above and
survives** in the narrowed "compile-time enforcement in a type" form.
