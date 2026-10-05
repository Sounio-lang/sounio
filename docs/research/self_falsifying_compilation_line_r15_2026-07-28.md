<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r15-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r15-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R15 — a verdict token is blind to whatever preserves the truth of its proposition

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE`
**Parents:** `self_falsifying_compilation_line_r14_2026-07-27.md` (the anomaly this resolves), `self_falsifying_compilation_line_r2_2026-07-26.md` (verdict-token binding — the contribution this rung bounds)
**Harness:** `scripts/research/self_falsifying_compilation_line_r15_contract.py` (+ `scripts/research/r15/`)
**Gate:** `scripts/ci/self_falsifying_compilation_line_r15_gate.sh`

---

## 1. Result

R14 left one perturbation unexplained: flipping the Cayley–Dickson sign of a
single product, σ(64, 192) at level 8, leaves the contract's verdict
`ZD_FIBER_SPECTRUM_COMPLETE_INVARIANT_N_LE_8` unchanged. Three explanations were
tested and refuted there. The resolution turns out not to be about that product.

> **The flip changes 126 of 128 fiber graphs and every one of their spectra.
> What it preserves is the *number* of distinct spectra — 24 before, 24 after,
> while the set of 24 is entirely replaced. The contract's claim has the form
> `#distinct spectra = 3·2^(n−5)`, so its check tests a cardinality, and a
> cardinality cannot see a transformation that swaps the things it counts.**

Verdict: `SELF_FALSIFYING_R15_VERDICT TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE`.

### 1.1 It is a family, not an accident

At every level tested, the flip **σ(H/2, H + H/2)** with H = 2^(n−1) preserves
the count, while generic flips at the same level change it:

| n | baseline | σ(H/2, H+H/2) | generic controls |
|---:|---:|---:|---|
| 5 | 3 | **3** | 5, 5 |
| 6 | 6 | **6** | 7, 10 |
| 7 | 12 | **12** | 13, 20 |
| 8 | 24 | **24** | 25, 25, 40 |

The control column is what makes this a finding rather than a robustness
observation: if every flip preserved the count, the count would simply be
sturdy and σ(64,192) would be unremarkable.

### 1.2 Why only level 8 was invisible

A wrapper on the sign function intercepts the **recursive** calls too, so a flip
aimed at level k also perturbs every deeper level computed through it. The
contract checks n = 6, 7 and 8 together. A count-preserving flip at 5, 6 or 7
therefore still betrays itself higher up — which is exactly what R14's probe saw
when the same arithmetic form killed at levels 4–7 and survived only at 8.

At level 8 — the top of the contract's own analysis — there is nowhere higher to
look. **The blind spot sits at the boundary of the claim because that is the
only place a count-preserving perturbation has no second chance to be caught.**

---

## 2. What this says about verdict-token binding

R2 was this line's contribution: bind the build not to a check's exit status
(`build.rs`) nor to its literal output (snapshot testing) but to the
**proposition** the check reports. R15 locates the next limit up, and it is
structural rather than incidental:

> **A verdict token's resolution is bounded by the invariance group of the
> proposition it states.** A claim of the form `#X = N` is invariant under
> everything that preserves |X|. Binding the proposition does not bind the
> witness.

Here the proposition is *"there are 24 distinct spectra"*. The witness is
*which* 24. The perturbation preserves the first and replaces the second
entirely. (R16 identifies the group precisely: not maps preserving |X| but maps
acting **within the blocks** of the classification. Count-preservation is a
consequence of that, not the mechanism.)

This is not the shared-misinterpretation impossibility of R0 §3 — the check is
not wrong, and neither is the claim. It is a **resolution** limit: the token is
exactly as fine as the proposition, and propositions about counts are coarse.

### 2.1 The repair, verified rather than proposed

Bind the token to the **witness**, not the predicate: a hash of the sorted set
of spectra instead of its cardinality. Measured at n = 5, 6 (live) and n = 8
(recorded): in every case `|S| = |S′|` while **`S ≠ S′`**. A witness-bound token
changes; a count-bound token does not. Two lines in a real contract, and `C3`
verifies the discrimination rather than asserting it.

The general form: **bind a witness of the proposition, not its truth value.**

---

## 3. What this is NOT

- **Not a refutation of the completeness claim.** Fixed in the harness docstring
  before any number was seen: a perturbed sign table is **not** a Cayley–Dickson
  algebra, so nothing here bears on whether the spectrum is a complete invariant
  for the real tower at n ≤ 8. What is measured is the reach of the **check**,
  not the truth of the **claim**.
- **Not a cospectral counterexample.** That was the interesting-looking
  hypothesis going in, and it is false: of 128 fibers, **zero** change adjacency
  without also changing spectrum. The spectrum tracks the graph faithfully at
  every fiber; only the aggregate is blind.
- **Not proof that σ(H/2, H+H/2) is the only count-preserving flip.** It is the
  only one found among the perturbations tested, at four levels, with controls.
  A systematic search over all pairs was not run.
- **Not an explanation of why this family preserves the count.** The regularity
  is measured across n = 5–8, not derived. **R16 answers the mechanism**
  (`self_falsifying_compilation_line_r16_2026-07-28.md`): the flip preserves the
  whole set *partition* of fibers into spectrum-classes, changing exactly two
  edges per fiber, because the flipped pair's home fiber is the one the contract
  does not examine. So the group in §2 is **partition-preserving**, wider than
  the count-preserving description used here. The regularity is still measured
  rather than proved — R16 §3 keeps that limit.
- **Not a compiler change.** Still Python-only.

---

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r15_contract.py
# expect: C1 counts 3/6/12/24, C2 the family preserving at every level with
#         controls changing, C3 sets differ at equal cardinality,
#         SELF_FALSIFYING_R15_VERDICT TOKEN_RESOLUTION_BOUNDED_BY_PROPOSITION_INVARIANCE

bash scripts/ci/self_falsifying_compilation_line_r15_gate.sh
```

`n = 5` and `n = 6` are re-derived **live** by a from-scratch re-implementation
of the committed annihilation-graph construction — which reproduces the
contract's own published counts (`C1`) before anything is concluded from it.
`n = 7` and `n = 8` are read from `scripts/research/r15/recorded.json`; the n = 8
sweep costs about five minutes per configuration on eleven cores, so it is
recorded rather than re-run in a gate. The producing scripts are alongside it.

---

## 5. AI disclosure

Reconstruction, probes, control, gate and spec drafted under human direction
(2026-07-28). The construction was re-implemented from the recursion rather than
copied, and validated against the contract's published spectrum counts before
use. The scope limit in §3 and the control design in §1.1 were fixed before the
numbers were seen; the cospectral hypothesis was pre-registered and lost. No
clinical content. GAIDeT-ICMJE 2025.
