<!-- docs:meta
topic_id: repo.docs.research.paper-a-sections7-9-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-sections7-9-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §7 *Implementation* + §9 *Related work* (full draft, 2026-08-25)

> Draft prose. §7 is grounded in the authorized compiler wire (synthesis §26: module
> `self-hosted/check/noise_sets.sio`, trailing `noise_set_id` on `TypeEntry`, diagnostic
> E230, phases N1–N4) and the souc-green prototypes; parts not yet built are labelled
> **[pending wire]**. §9 cites the prior art established by the two adversarial gates
> (affine arithmetic / Fluctuat; QIF / Blackwell) plus the type-machinery neighbors.

---

## 7. Implementation

The discipline is implemented in Sounio, a self-hosted language whose checker (Madaros)
is itself written in Sounio. The design reuses three things already in the compiler — the
`TypeEntry` tag mechanism, the monotone-dataflow engine, and the provenance rule it sits
beside — so the noise-symbol discipline is an *added tag with a join rule*, not a new type
parameter threaded through unification from scratch.

### 7.1 Where the source-set lives

A Knowledge type's static data is a `TypeEntry` record (`self-hosted/check/types.sio:139`)
that already carries a family of index-tags flowing through the compatibility check
`compat` — `unit_id` (dimensions), `refinement_id`, `algebra_kind`, `epistemic_meta_id`,
`ontology_id`, `knowledge_epsilon`. The source-set is one more such tag: a trailing field

    noise_set_id : i64        // −1 = ⊤ (unknown), 0 = ∅, >0 = interned nonempty set

placed **after** `provenance_id` (§7.4), defaulting to `−1` at every existing `TypeEntry`
construction site (the conservative top). The set itself is interned in a dedicated module
`self-hosted/check/noise_sets.sio`; `noise_set_id` is a handle into that table, and
`union`/`disjoint` dereference the handle through the module rather than doing bitwise
arithmetic on the id — the id is an identity, not a mask. (The prototype `ns_contract.sio`
uses an inline `i64` bitmask, a bounded 64-source stand-in that validates the *semantics*;
the compiler uses the interned-handle representation for the *scale*.)

### 7.2 The join site

The disjointness check has a template already in the tree. `compat.sio` handles Knowledge
compatibility in its `TyKnowledge` arm (~`:230`) by comparing `knowledge_epsilon`, and it
handles model-family compatibility in the `TyModelFamily` arm (`:250`) with a
`a.epistemic_meta_id == b.epistemic_meta_id` "tags must match or incompatible" test — which
is exactly the shape of a source-set join. The independence-assuming binary operators are
typed at `check.sio` (the `kadd`/`kmul` join site, ~`:18862`); the NS rule adds, beside the
existing `knowledge_epsilon` handling, the disjointness premise of §5.4: compute
`disjoint(noise_set_id(a), noise_set_id(b))` via the `noise_sets` module, and on failure
raise E230 rather than producing a result type.

### 7.3 Phasing (N1–N4)

The wire is serialized into four behavior-neutral-then-active phases:

- **N1 — representation only.** Add the `noise_set_id` field and the `noise_sets` module;
  default every site to `−1`. The bootstrap and source build are behaviorally identical to
  before (no rule consults the field yet). This is the safe, large-surface diff.
- **N2 — transfer.** Seed a fresh symbol at `measure`; union at `kadd`/`kmul`; inherit at
  copy/ident; the parametric call-summary substitution for interprocedural flow (§5.6).
- **N3 — the gate.** Raise E230 at `kadd`/`kmul` when disjointness cannot be proved; the
  same-source-built sabotage witness (disable only the NS rule → the E230 vanishes while
  E222 stays — the compiler-level form of §8.2).
- **N4 — regression.** The named CI gate `scripts/ci/ns_antigarbling_gate.sh` plus the full
  test suite: compile-fail `ns_add_shared_source_rejected.sio` (x+x),
  `ns_add_unknown_conservative.sio`; run-pass `ns_add_disjoint_ok.sio`,
  `ns_ident_preserves_source.sio`.

**Status.** The semantics, the acceptance controls, and the analysis engine run today as
the prototypes evaluated in §8 (`noise_symbols.sio`, `ns_dataflow.sio`, `ns_contract.sio`,
all souc-green). N1–N4 in the production checker are authorized and specified but not yet
landed; every claim depending on the wired checker is marked **[pending wire]** in §8.

### 7.4 Coexistence with provenance

`noise_set_id` is deliberately a *separate* field from the provenance tag `provenance_id`
and from the overloaded `knowledge_epsilon` (which already multiplexes transport/diagram/
fairness/grade confidences — reusing it would collide). The diagnostic E230 is likewise
distinct from the provenance diagnostic E222. The two rules share the `TypeEntry` mechanism
and the dataflow substrate but remain independent abstract domains (§5.7), which is what
lets the N3 sabotage witness disable one without perturbing the other.

---

## 9. Related work

Our contribution sits at the intersection of three lines, none of which occupies it. We
state each neighbor and the precise delta, so the claim is neither the tracking (30 years
old) nor the soundness frame (standard in another domain) but their combination as a
compile-time type rule in an uncertainty-typed language.

### 9.1 Affine arithmetic and zonotopic static analysis

Noise-symbol identity is the defining device of **affine arithmetic** (Comba & Stolfi
1993): a quantity is `x₀ + Σᵢ xᵢεᵢ`, and shared `εᵢ` between two quantities are exactly
their correlation, so `x − x = 0` and correlated errors do not inflate independently.
**Goubault & Putot**'s zonotopic abstract domains and the **Fluctuat** analyzer (*Static
Analysis of Finite Precision Computations*, VMCAI 2011; *Perturbed affine arithmetic for
invariant computation*, 2008; the logical-product zonotope intersection, 2010) build a
static analysis on precisely this representation, tracking correlations between program
variables through shared noise symbols to bound finite-precision error in C/Ada.

This is the closest prior work on *source identity*, and we reuse its core idea. The delta
is threefold: (i) Fluctuat's noise symbols live in an **external analyzer** producing an
enclosure; ours live **in the type** and are part of the program's interface. (ii) Fluctuat
**reports** a bound; we **reject** — an independence-assuming operator over correlated
operands is a *type error* (E230), not a wider interval. (iii) Fluctuat targets roundoff of
a fixed computation; we target the *soundness of uncertainty propagation itself*, where the
failure is a library computing the wrong variance formula, and the fix is a checked
precondition plus a correlation-aware operator. "We track source identity" is not our claim;
"we make it a typed, rejecting precondition in an uncertainty language" is.

### 9.2 Quantitative information flow and the Blackwell order

The soundness criterion — an operation may lose information but never create it — is the
**data-processing / Blackwell informativeness order** (Blackwell 1953), and the
**quantitative information flow** community has made it the backbone of its refinement
theory: McIver, Morgan, Smith, Espinoza & Meinicke, *Abstract channels and their robust
information-leakage ordering* (POST 2014), and Alvim, Chatzikokolakis, McIver, Morgan,
Palamidessi & Smith, *The Science of Quantitative Information Flow* (Springer 2020), identify
program refinement with channel garbling and use post-processing monotonicity as the
soundness condition — `A` refines `B` iff `B` is a garbling of `A`.

We adopt this frame wholesale and say so. The delta is the *domain and the mechanism*: QIF
orders **confidentiality channels** and measures **leakage**; we order **uncertainty
propagation** in the **variance channel** and enforce the anti-garbling prohibition as a
**static type rule** at arithmetic operators. Reading `ep_add`'s variance understatement as
an anti-garbling is, to our knowledge, a new instantiation of the QIF/Blackwell discipline;
it is not a new order. (The lift of this frame to non-associative composition — where
reassociation itself becomes a garbling governed by the octonion associator — is a separate
contribution and not claimed here.)

### 9.3 Uncertainty-typed languages and libraries

The systems that carry uncertainty in the type or value are our *host* setting, and are
precisely the ones that do **not** track source identity. `Uncertain⟨T⟩` (Bornholt, Mytkowicz
& McKinley, ASPLOS 2014) represents a value as a sampled distribution and computes over it,
but a Monte-Carlo product of a variable with itself does not know it is the same variable
unless the samples are shared by construction, and nothing types the fusion. `Measurements.jl`
(Giordano) propagates GUM uncertainty and *does* track correlations at runtime via a
derivative graph — but as a numeric result, not a type, and with no compile-time rejection of
an independence-assuming path. GUM implementations (ISO/IEC 98-3) and `Ferson` **p-boxes**
give the underlying arithmetic; they are libraries, and fusing two quantities wrong is a
numeric mistake, never a type error. Our contribution is exactly the missing enforcement:
the source-set in the type, the operator's independence assumption as a checked precondition.

### 9.4 Type systems for numerical error

A parallel line puts *numerical* properties in the type. **NumFuzz** (Numerical Fuzz: a type
system for rounding-error analysis, 2024) and **Bean** (a language for backward error
analysis, 2025), and the broader *type-based approaches to rounding-error analysis* (2025),
use linear/sensitivity typing to bound roundoff. These share our shape — a numeric soundness
invariant carried by the type — but a different invariant: roundoff magnitude, not
correlation-soundness of uncertainty propagation. The two are complementary; a value could in
principle carry both a sensitivity bound and a noise-set. We borrow the discipline
(numeric-property-in-the-type) and contribute a distinct invariant and its dataflow.

### 9.5 Information-flow types and probabilistic programming

The type machinery closest to ours is **information-flow / taint typing**: a set-valued
lattice propagated through the program, with a check at sinks. The noise-set is an IFC-style
lattice whose "labels" are measurement sources and whose "sink" is an independence-assuming
operator; the reading is novel (covariance-soundness rather than confidentiality), the
machinery is familiar, and we position NS as an IFC-shaped discipline for a numeric soundness
property. Finally, **probabilistic programming** (Stan, Pyro) *can* express correlated
uncertainty via hierarchical models, but at inference time: a posterior fuses correlations
correctly only when the model is hand-structured to, and fusing wrong is a modelling choice,
never a compile-time error. We target the opposite regime — cheap, first-order, static, and
*rejecting* — where the guarantee is that the unsound path does not type-check.
