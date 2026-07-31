<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-v1-reduction-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-v1-reduction-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — V1 reduced: ∀n spectral completeness needs only two integers

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `V1_REDUCED_TO_TWO_INTEGERS__NOT_CLOSED`
**Parents:** `cd_tower_zd_fiber_antisymmetry_lemma_spec_2026-07-31.md`, `cd_tower_zd_fiber_spectral_forall_n_progress_2026-07-26.md`, `cd_tower_zd_fiber_spectral_classifier_2026-07-26.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_v1_reduction_contract.py`

---

## 0. What this does and does not do

**V1 is not closed.** `#distinct A_σ spectra = 3·2^{n-5}` for all `n` remains **OPEN**. This
rung takes it apart and shrinks the hard half.

Two things come out:

1. **The count is orbit arithmetic, not a mystery.** Three of its four ingredients were already
   settled; only two statements carry content.
2. **The hard half needs only two integers.** What looked like a cospectrality problem — "no two
   of the `3·2^{n-5}` classes share a spectrum", the shape that is notoriously hard because
   cospectral graphs are common — is really a question about `tr(A_σ²)` and `tr(A_σ³)`.

---

## 1. The decomposition

```
#classes = (#Fano orbits) + (#seam orbits) − (#merges)
         =    2^{n-4}     +  (2^{n-4} − 1)  − (2^{n-5} − 1)
         = 2^{n-3} − 2^{n-5} = 3·2^{n-5}
```

| ingredient | status |
|---|---|
| (a) the orbit split `2^{n-4}` Fano + `2^{n-4}−1` fixed seams | **proven ∀n** — the orbit theorem |
| (b) the spectrum is constant on each orbit | **proven ∀n** — the automorphism is an algebra map, hence a graph isomorphism |
| (c) the even-weight seams merge, exactly `2^{n-5}−1` of them | parity-collapse law; explicit `Φ` verified `n ≤ 8`, ∀n reduces to two σ-lemmas — **OPEN** |
| (d) **nothing else merges** | **OPEN** — this rung reduces it |

The arithmetic between them is trivial. So **V1 ∀n = (c) ∀n and (d) ∀n**, and nothing else.
`W1` measures the whole decomposition at n = 6..9; `W2` measures (b).

---

## 2. The reduction

> **The pair `(tr A_σ², tr A_σ³)` induces exactly the spectral partition.**

`tr(A_σ²)` is the number of nonzero entries — twice the edge count, since entries are `±1`.
`tr(A_σ³)` is the signed triangle count. Both are computable without an eigendecomposition.

- **`W3`** — the partition into blocks of equal `(tr A², tr A³)` is **identical, block for
  block**, to the partition by full spectrum, at n = 6..9. This is the clause that matters:
  two partitions can have the same *number* of blocks and still differ, so an equal count would
  not have been enough.
- **`W4`** — `#distinct (tr A², tr A³) = 3·2^{n-5}` at **n = 6, 7, 8, 9, 10, 11** — six levels.

So (d) becomes: *is a two-integer invariant injective on the classes, for all n?* That is a
closed-form question, not a spectral one.

**`W5` — the pair is not padded.** `tr(A¹) = 0` for every fiber (the diagonal vanishes — itself
a theorem, `Asig_diag` in the parent rung), and `tr(A²)` **alone** gives exactly `2^{n-4}`
values, strictly fewer than `3·2^{n-5}`. Neither trace alone does the job.

**`W7` — the reduction is not vacuous.** The weaker invariant `(#vertices, tr A²)` under-
separates at every level. If a weaker invariant separated too, `W4` would carry no information.

---

## 3. Where the closed form stops — and it stops early

`W6`. On the stratum where `tr(A²)` is **constant** — the `y = 0` Fano class together with the
**weight-1 seams only** (`7 + (n−3)` fibers) — it equals

```
tr(A²) = (2^{n-1} − 2) · 4(2^{n-3} − 1)
```

i.e. the graph is `Dmax`-regular off its single isolated vertex, with `Dmax = 4(2^{n-3}−1)` the
core law already proven ∀n. Measured: `840, 3720, 15624, 64008` at n = 6,7,8,9.

**Off that stratum `tr(A²)` genuinely varies, and no closed form is derived here.** `W6` asserts
the variation as a clause so the boundary cannot be read as an oversight. Fitting the general
values appears to need the degree-histogram induction, which this lane records as **open**. Two corrections, both caught by the clause rather than by inspection:

1. An earlier draft called this closed form "evident". It is evident only on the stratum where
   the quantity does not vary — not the same thing.
2. That draft also put **all** seams in the stratum. `W6` failed (`rc=1`). The stratum is
   `y = 0` Fano plus the **weight-1** seams; the higher-weight seams take other values. The
   mistake came from reading a table whose top block happened to contain only weight-1 seams
   and generalising from it.

`tr(A³)` has no closed form here at all.

---

## 4. Not claimed

- **V1 is not closed.** Neither (c) nor (d) is proven ∀n.
- **(d) is reduced, not proven.** Closing it needs **two** things, and both are open: a closed
  form for `(tr A², tr A³)` in terms of the fiber label, *and* a proof that that form is
  injective on the `3·2^{n-5}` classes for all `n`. Six levels of agreement is evidence, not a
  proof — a form that collides at some larger `n` would leave (d) exactly where it was.
- **(c) is untouched** by this rung and remains `n ≤ 8`. It is plausibly the cheaper of the two:
  its two σ-lemmas (`τ` preserves resonance; the discrepancy is a coboundary) have the same
  shape as `A4_sub` in the parent rung — an F₂ sign identity provable by induction through the
  four branch reductions — and that toolkit is now in-tree.
- **Numerical certificate**, exact integer sign table, `D3` class. Nothing here is Lean-proven;
  the parent rung's Lean file covers the antisymmetry lemma, not this decomposition.

---

## 5. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_v1_reduction_contract.py
```

`W0` pins this rung's builders to the in-tree `sign_table`/`A_sig` entrywise, so every clause is
measured against the lane's own generator. `W4` at n = 11 dominates the runtime (~4 min solo);
the spectral clauses stop at n = 9 because they need eigendecompositions and the traces do not —
which is precisely the point of the reduction.
