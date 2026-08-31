<!-- docs:meta
topic_id: repo.docs.research.paper-a-sections3-10-11-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-sections3-10-11-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §3 *Preliminaries* + §10 *Limitations* + §11 *Conclusion* (full draft, 2026-08-25)

> Draft prose closing the paper. §3 fixes the four pieces of vocabulary the technical
> sections use; the neighbours' full treatment is §9. §10 consolidates the boundaries
> already stated locally in §4.4, §5.6, §6.5, §8.5. §11 is the close.

---

## 3. Preliminaries

We fix four notions the rest of the paper uses. None is new here; §9 places them in the
literature.

**Uncertainty types (the host).** We work in a language where a quantity carries its
uncertainty in the type: an *epistemic value* has type `Knowledge⟨T⟩` and, at runtime, a
payload of type `T` together with metadata — for our purposes a *variance* `v = Var(X) ≥ 0`
(and a confidence field, orthogonal to soundness, which we ignore). Arithmetic on
`Knowledge⟨T⟩` propagates the variance automatically. This is the setting of
`Uncertain⟨T⟩`, `Measurements.jl`, and GUM tooling (§9.3).

**GUM propagation and its hidden hypothesis.** The propagation laws these systems implement
are the ISO GUM first-order (delta-method) rules: `Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)`,
`Var(XY) ≈ Y²Var(X) + X²Var(Y) + 2XY·Cov(X,Y)`, and so on. Every implementation we study
ships the **independence special case** — the same formulas with the covariance term set to
zero — as the operator. The special case is exact iff `Cov(X,Y) = 0`; applied to correlated
operands it understates. Making that hypothesis explicit and checked is the whole paper.

**Affine forms / noise symbols.** To reason about *which* uncertainty a value carries we use
the representation of affine arithmetic (§9.1): a value is an affine form
`x = x₀ + Σᵢ xᵢεᵢ` over independent unit-variance *noise symbols* `εᵢ`, one per
independent measurement source. Then `Var(x) = Σᵢ xᵢ²`, `Cov(x,y) = Σᵢ xᵢyᵢ` (the inner
product), and two values are correlated exactly when they share a symbol with nonzero
coefficient. The **support** of `x` is the set of symbols on which it has a nonzero
coefficient; **disjoint support ⟹ zero covariance** (the converse fails — §4.4). The type
system tracks an over-approximation of each value's support (§5).

**The Blackwell / data-processing order.** An uncertain quantity is an *experiment* (a
channel from the true value to an observation). Blackwell's order compares experiments by
informativeness: `B ⪯ A` iff `B` is obtainable from `A` by post-processing through a
stochastic channel — a **garbling**, which can only lose information (the data-processing
inequality). We call an operation that reports *less* uncertainty than its operands justify
an **anti-garbling**; it manufactures information and is forbidden. §4 makes this the
soundness criterion, in the variance channel; §9.2 gives its home in quantitative
information flow.

---

## 10. Limitations

We collect the boundaries stated locally through the paper, so the guarantee's shape is in
one place.

- **Conservative, not complete (§4.4, §6.5).** The check keys on disjoint *support*, which
  is sufficient but not necessary for zero covariance. It therefore rejects the
  overlapping-but-orthogonal case (`a = x₁+x₂`, `b = x₁−x₂`, `⟨a,b⟩ = 0`). Such programs are
  recovered by the escape valve (a proved-disjoint certificate or the correlation-aware
  operator, §5.5), never by unsound admission. The guarantee is soundness, not maximality of
  the admitted set.

- **First-order / variance channel only (§6.5, §8.5).** Soundness is exact for the linear
  fragment; nonlinear operators (`mul`, `div`, `square`, `sqrt`) are delta-method
  approximations that drop second-order terms, so a residual second-order discrepancy
  survives even under disjoint support. It is a symmetric approximation error, not a
  directional anti-garbling, and the type prevents the *first-order covariance* anti-garbling
  — the entire defect class of §2 — not the truncation error itself. Non-Gaussian or
  heavy-tailed uncertainty is likewise under-described by variance; the criterion is a
  second-moment one.

- **Interprocedural summaries are the load-bearing dependency (§5.6, §7.3).** The transfer
  is intraprocedural; sound cross-call source tracking needs parametric call-summaries.
  Without them the sound default is *assume-sharing* (drop to `⊤` at call boundaries), which
  is sound but noisy. Building the summaries — shared with the compiler's memory-reclamation
  analysis — is the principal engineering cost and is part of the pending wire.

- **Unknown correlation beyond {0, 1} (§9.5).** The escape valve's correlation-aware
  operator takes a known `ρ`. When the correlation is *unknown* (not zero, not one, not a
  given value), sound propagation needs Fréchet bounds, and the correlation assumption should
  itself become a tag on the type. We do not model this; it is a stated gap.

- **Evaluation is one library plus prototypes (§8).** RQ1 quantifies one shipping library;
  the class is general to GUM-style propagation but measured on one instance. RQ2's causality
  is established at the analysis level (the sabotage control) and awaits its compiler-level
  form. The corpus false-positive rate (RQ3) and the full two-compartment clinical flip rate
  (RQ4) are **pending the checker wire** (N3–N4) and, for RQ4, the model's two-compartment
  extension.

- **Confidence decay is heuristic.** The `confidence` field's per-operation decay
  (`× 99/100`, `× 98/100`) is not derived from a principle; it is orthogonal to the variance
  soundness this paper establishes, but it is drift until derived, and we do not defend it.

---

## 11. Conclusion

Uncertainty-typed languages promise to carry `± σ` in the type and propagate it
automatically. They keep the promise only where their unstated hypothesis holds — that the
operands of every operation are independent — and they break it silently everywhere else,
because the failure mode is a *tighter* error bar, the one answer that never looks wrong. We
showed the failure is not hypothetical: in a shipping library the same operation `x·x` has
two variances, and nothing routes the program to the sound one.

The fix is to name the hypothesis and check it. Reading uncertainty propagation through the
Blackwell / data-processing order already standard in quantitative information flow,
understating variance is an *anti-garbling* — manufacturing information — and no correct
program may contain one. We carry the noise-symbol source-set of each value in its type,
reusing affine arithmetic's source identity but lifting it from an external analyzer into the
type, and make the independence assumption of arithmetic a *checked precondition*: an
independence-assuming operator over operands whose sources are not provably disjoint is a
type error, discharged only by a proof of disjointness or by switching to an operator that
takes the correlation explicitly. The core soundness criterion is kernel-checked; the
discipline runs as prototypes and is specified into the production checker.

The guarantee is deliberately narrow and honestly bounded — first-order, conservative,
variance-channel — and it eliminates exactly the defect class it targets: the number that
looks too precise to be true is the one the compiler now refuses to print. Two directions
extend it. Downward, to second order: carrying the delta-method's dropped terms turns the
first-order guarantee into a full one. Outward, to non-associative composition: when the
affine coefficients live in a non-associative algebra, *reassociating* a product becomes a
garbling in its own right, governed by the octonion associator — the point where this
discipline meets a genuinely open question about the Blackwell order and algebraic curvature,
and the subject of separate work.
