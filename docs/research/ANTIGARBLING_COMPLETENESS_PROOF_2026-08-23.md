<!-- docs:meta
topic_id: repo.docs.research.antigarbling-completeness-proof-2026-08-23
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.antigarbling-completeness-proof-2026-08-23
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Anti-Garbling Completeness — the proof (first-order regime)

**Date:** 2026-08-23 · **Author:** fable-1 (agent=claude) · companion to
`ANTIGARBLING_COMPLETENESS_2026-08-23.md`.
**Result:** the completeness claim (C) is a **theorem for the first-order
(linearized) propagation of sums and products over a normed algebra** — with one
honesty correction to the informal version (the error is not additively separable
when a certificate fails; the two certificates are *independent and jointly
sufficient*, which is the operationally load-bearing statement).
**Machine-verified (2026-08-23):** the full spine below is checked in
`docs/research/lean/SounioAntiGarblingModel.lean` (Lean 4.33.0, exit 0, zero
warnings, `sorry`-free) — see §7 for the theorem map.

---

## 1. Setup (everything defined)

- **Carrier.** `A` — a real algebra with a bilinear product `·` (not assumed
  associative or commutative) and an inner product `⟨·,·⟩` with norm
  `‖a‖² = ⟨a,a⟩`. (Octonions with the standard Euclidean form are the model case;
  scalars `A = ℝ` are the degenerate case.)
- **Noise symbols.** Independent sources `ε₁,…,ε_n`, real random variables with
  `E[εₖ] = 0` and `E[εₖεₗ] = δₖₗ`. *Orthonormality is definitional* — a "source"
  *is* an independent unit symbol. Correlation between values is represented **only**
  by **shared symbols**, never by a nondiagonal moment matrix.
- **Uncertain value (affine form).** `x = x₀ + Σₖ dₖ(x)·εₖ`, with center
  `x₀ ∈ A` and sensitivity coefficients `dₖ(x) ∈ A`. The **support** is
  `supp(x) = { k : dₖ(x) ≠ 0 }`.
- **Expression tree.** `T` — a finite tree whose leaves are input affine forms and
  whose internal nodes are `+` or `·` (binary, bilinear). A **parenthesization**
  is the shape of `T`; a **symbol-assignment** `σ` records which εₖ each leaf
  carries (i.e. which leaves share a source).
- **Variance.** For a value `w`, `Var(w) := E‖w − E[w]‖²`.

**First-order truncation.** Expand a node value to first order in the εₖ, i.e. keep
`w = w₀ + Σₖ (∂ₖw)·εₖ + O(ε²)` and drop `O(ε²)`. The dropped `O(ε²)` term is the
**curvature/nonlinearity** — it is *not* part of (C); it is the separate `C_H`
axis (exact only under Monte-Carlo / p-box). Everything below is exact **for the
first-order value**.

**Lemma 0 (first-order variance is the sensitivity norm).**
For the first-order value `w = w₀ + Σₖ (∂ₖw)εₖ`,
`E[w] = w₀` and `Var(w) = Σₖ ‖∂ₖw‖²`.
*Proof.* `E[w]-w₀ = Σₖ (∂ₖw)E[εₖ] = 0`. `‖w-w₀‖² = Σ_{k,l} ⟨∂ₖw,∂ₗw⟩ εₖεₗ`; take
`E`, use `E[εₖεₗ]=δₖₗ`: `Var(w) = Σₖ ⟨∂ₖw,∂ₖw⟩ = Σₖ‖∂ₖw‖²`. ∎

So **first-order variance is a quadratic form in the sensitivity vector**
`∂w = (∂ₖw)ₖ ∈ Aⁿ`. Everything reduces to how the propagator computes `∂w`.

---

## 2. The sensitivity map has exactly two structural inputs

**Lemma 1 (Leibniz).** The first-order sensitivity obeys, at every node,
`∂ₖ(L + R) = ∂ₖL + ∂ₖR` and `∂ₖ(L · R) = (∂ₖL)·R₀ + L₀·(∂ₖR)`
(product rule; `L₀,R₀` the centers). Hence `∂ₖw` is computed bottom-up from the
leaf coefficients by these two rules along `T`.

**Proposition 2 (two degrees of freedom).** Fix the expression tree `T` and the
leaf centers/coefficients. Then the sensitivity vector `∂w` is a function of
**exactly two** data: the **symbol-assignment `σ`** (which leaves share which εₖ)
and the **parenthesization** of the products. Nothing else in a first-order
bilinear propagation can change `∂w`.
*Proof.* By Lemma 1, `∂ₖw` is built from the leaf coefficients `dₖ(leaf)` by `+`
and `·`-with-centers. The leaf coefficient `dₖ(leaf)` is nonzero iff that leaf
carries `εₖ` — i.e. determined by `σ`. The order in which `·`-with-centers are
composed up the tree is the parenthesization. Centers are fixed; the two rules are
fixed. There is no third input. ∎

This is the **dimension count**, made precise: the first-order structural error of
a propagator is governed by a two-coordinate object `(σ, parenthesization)`.

---

## 3. The two garblings, exactly

Write `∂w_true` for the sensitivity under the true `(σ_true, T_true)` and
`∂w_naive` under the propagator's `(σ_naive, T_naive)`.

**(A) Support (axis 2).** The naive propagator that treats operands as
independent assigns **disjoint fresh symbols** to distinct subterms
(`σ_naive`), whereas `σ_true` merges shared sources. At a node `L ⊕ R`:

- `+`: `∂ₖ(L+R)_true = ∂ₖL + ∂ₖR`. Var over that node,
  `Σₖ‖∂ₖL+∂ₖR‖² = Σₖ‖∂ₖL‖² + Σₖ‖∂ₖR‖² + 2Σₖ⟨∂ₖL,∂ₖR⟩`. The naive (disjoint)
  bookkeeping sums the first two only, **understating by**
  `Δ_support = 2·Σₖ ⟨∂ₖL,∂ₖR⟩ = 2·Cov(L,R)`. Since `⟨∂ₖL,∂ₖR⟩ = 0` whenever
  `k ∉ supp(L)∩supp(R)`, `Δ_support = 2·Σ_{k∈supp(L)∩supp(R)}⟨∂ₖL,∂ₖR⟩`.

**Lemma 3 (support certificate soundness — PROVEN, this is the NS anchor).**
`supp(L) ∩ supp(R) = ∅ ⇒ Cov(L,R) = 0 ⇒ Δ_support = 0.`
*Proof.* Every summand `⟨∂ₖL,∂ₖR⟩` has `k ∉ supp(L)∩supp(R)=∅`, so at least one
factor is `0`; the sum is `0`. ∎  (This is exactly `cov_zero_of_disjoint` in
`SounioAntiGarblingModel.lean`, and exactly what `ns_disjoint`/E230 certifies.)

**(B) Order (axis 1).** For products of ≥3 factors the naive propagator is free to
re-associate. By Lemma 1 the first-order sensitivity of a re-association differs by
a sum of **associators evaluated at centers with the noise coefficient in one
slot**:

`∂ₖ[(xy)z] − ∂ₖ[x(yz)] = [dₖ(x),y₀,z₀] + [x₀,dₖ(y),z₀] + [x₀,y₀,dₖ(z)]`,
where `[a,b,c] := (a·b)·c − a·(b·c)`.
*Proof.* Apply Lemma 1 to both parenthesizations term by term and subtract; each
of the three Leibniz terms contributes one associator. ∎

**Lemma 4 (order certificate soundness).** If every triple associator over the
relevant center values vanishes (the algebra is associative on the reached
subalgebra — e.g. a Fano/associative triple in the octonions), then
`∂ₖw` is parenthesization-independent, so `Δ_order = 0`.
*Proof.* Immediate from the associator formula above: all difference terms are `0`.
∎  (When it does *not* vanish, the sound propagator must add the associator spread
`κ‖[·,·,·]‖²` — the augmentation verified in `product_nonassoc.sio`, 0.25/4.25.)

---

## 4. The theorem

**Theorem (Anti-Garbling Completeness, first-order).**
Let `w` be built from affine-form inputs over a normed algebra `A` by a fixed tree
of sums and products, propagated to first order.
1. **(Soundness — the operational core.)** If at every combining node the operand
   supports are disjoint (`Δ_support = 0`, Lemma 3) **and** every product is taken
   in its true parenthesization or its associators vanish (`Δ_order = 0`, Lemma 4),
   then `∂w_naive = ∂w_true` and hence `Var_naive(w) = Var_true(w)` — **no
   structural understatement.**
2. **(Completeness — the dimension count.)** These two certificates are the only
   structural certificates a first-order bilinear propagator needs: by
   Proposition 2 the sole inputs to `∂w` are `σ` and the parenthesization, whose
   correctness is exactly what Lemmas 3–4 certify. Any remaining discrepancy
   between `Var_naive` and `Var_true` is `O(ε²)` — curvature — which is outside the
   first-order regime by construction.

*Proof.* (1) Under the hypotheses, Lemma 3 gives `σ`-agreement of the sensitivity
(shared-symbol merging is vacuous when supports are disjoint) and Lemma 4 gives
parenthesization-agreement; so `∂ₖw_naive = ∂ₖw_true` for every `k`, and Lemma 0
gives equal variance. (2) Proposition 2 shows `∂w` depends on nothing but
`(σ, parenthesization)`; both are certified; the only excluded term is the
first-order truncation remainder, which is second-order in `ε`. ∎

---

## 5. The honesty correction (what the informal note over-claimed)

The informal companion suggested the error **decomposes additively** as
`Var_true − Var_naive = Δ_support + Δ_order`. **That is false in general**, and
saying so is part of proving (C) "direito."

Write `∂w_true = ∂w_naive + δ_sup + δ_ord`, where `δ_sup` is the sensitivity change
from merging shared symbols and `δ_ord` from re-parenthesizing. Then
`Var_true − Var_naive = Σₖ(‖∂ₖw_naive+δ_sup+δ_ord‖² − ‖∂ₖw_naive‖²)`
`= [2⟨∂w_naive,δ_sup⟩+‖δ_sup‖²] + [2⟨∂w_naive,δ_ord⟩+‖δ_ord‖²] + 2⟨δ_sup,δ_ord⟩`.
The last term `2⟨δ_sup,δ_ord⟩` is a **support×order interaction** that need not
vanish when *both* certificates fail. So:

- The correct statement is **not** "two additive error terms," but **"two
  independent certificates that are jointly sufficient for exactness"** (Theorem
  §4.1), together with the **two-degree-of-freedom characterization** (§4.2). When
  a *single* certificate holds the other's failure is clean; when *both* fail the
  errors interact.
- This is strictly stronger where it matters (soundness is unconditional on the
  interaction) and strictly more honest (no false additive separability).

---

## 6. Scope, boundary, falsifiers

- **Regime.** First-order (linearized) propagation of **sums and products**
  (bilinear operations). This is exactly the fragment where NS and the associator
  live.
- **Excluded, by construction.** Curvature/nonlinearity (`exp`, ratios,
  higher moments `E[εᵢεⱼ]`): the `O(ε²)` remainder, the `C_H`/Monte-Carlo axis.
  (C) makes **no** claim there — that is a feature: it draws the exact line between
  *structural* (identity) error and *approximation* (shape) error.
- **Falsifiers.**
  - **F1 (kills §4.1):** a bilinear first-order program with disjoint supports and
    vanishing associators whose true variance still exceeds the naive one. By
    Theorem §4.1 this is impossible; exhibiting it refutes the proof (check the
    Leibniz/Lemma-0 steps).
  - **F5 (kills §4.2 completeness):** a first-order bilinear propagator error not
    attributable to `σ` or parenthesization. By Proposition 2 impossible; a
    counterexample would expose a hidden third input.
  - **F6 (confirms §5):** a program where **both** certificates fail and
    `Var_true − Var_naive ≠ Δ_support + Δ_order` — this should *hold* (nonzero
    interaction), confirming the additive form was wrong to assert.

---

## 7. The Lean anchor — theorem map (machine-checked)

`docs/research/lean/SounioAntiGarblingModel.lean` is checked clean by Lean 4.33.0
(exit 0, zero warnings, `sorry`-free). The whole spine of this proof is verified —
in a first-order model (scalar/`Int` coefficients where the arithmetic is the
content; an abstract biadditive magma over an abelian group for the associator):

| this proof | Lean name |
|---|---|
| Lemma 3 (Axis 2 support) | `cov_zero_of_disjoint` (core `cov_pointwise_zero`) |
| Lemma 4 (Axis 1 order) | `parenthesizations_agree_of_associator_zero` (via `sub_eq_zero_imp`) |
| Theorem §4.1, sum node (Axis 2) | `antigarbling_sound_sum` (via `varSum_expand`) |
| Theorem §4.1, product node (Axis 1) | `antigarbling_sound_product`; and §3(B) sensitivity identity `fo_product_sensitivity_diff` (∂ diff = sum of 3 associators) |
| §4.1 congruence shape | `var_eq_of_sensitivity_eq` |
| Prop 2 (§4.2 dimension count) | `sens_eadd`/`sens_emul` (Leibniz-compositional), `sens_eadd_assoc`/`sens_eadd_comm` (sum-reassoc invariant), `sens_emul_reassoc` (product-reassoc = associators) |
| §5 non-separability | `antigarbling_interaction` (error = dSup+dOrd+2·interaction), `antigarbling_not_additive` (concrete witness the additive form fails) |

Mathlib-free throughout: the additive group (`AddGrp`), inverse/negation lemmas,
and polynomial expansions are self-contained, with `omega` over abstracted products
standing in for `ring`. The `.md` companions carry the general-algebra (octonion,
real-coefficient) framing that the structural Lean model instantiates.
