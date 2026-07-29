<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r16-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r16-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R16 — the invariance group, identified: partition-preserving, not merely count-preserving

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING`
**Parents:** `self_falsifying_compilation_line_r15_2026-07-28.md` (the open question this answers, and the description it sharpens)
**Harness:** `scripts/research/self_falsifying_compilation_line_r16_contract.py` (+ `scripts/research/r16/`)
**Gate:** `scripts/ci/self_falsifying_compilation_line_r16_gate.sh`

---

## 1. Result

R15 exhibited one element of the group a verdict token cannot see: the sign flip
**σ(H/2, H + H/2)**, H = 2^(n−1), preserves the number of distinct ZD-fiber
spectra at every level while generic flips change it. It left *why* open, calling
it the more interesting question. This rung answers it, and the answer enlarges
the group R15 described.

> **The flip does not merely preserve the count. It preserves the entire set
> partition of fibers into spectrum-classes — same blocks, same sizes — while
> replacing every spectrum that labels them. It changes exactly two edges per
> fiber, uniformly, and the reason it can do no more is arithmetic.**

Verdict: `SELF_FALSIFYING_R16_VERDICT INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING`.

### 1.1 Why the flip is minimal — arithmetic, not measurement

A fiber's vertices are pairs (lo, hi) with `lo XOR hi = L`. The flipped pair is
(h, H + h) with h = H/2, and

```
h XOR (H + h) = H        for every n
```

so that pair's own **home fiber is L = H, i.e. Llo = 0** — the single fiber the
contract does not examine, since its range is `1..H−1`. The flip therefore cannot
alter any vertex's internal product. All it can touch is adjacency *between* the
vertex whose `lo = h` and the vertex whose `hi = H + h`.

Checked as arithmetic for n = 5…12, not sampled.

### 1.2 Exactly two edges per fiber

| n | edges changed per fiber |
|---:|---|
| 5 | **2** in 14 fibers, 0 in 1 |
| 6 | **2** in 30 fibers, 0 in 1 |

The two are the sign variants of the one affected vertex pair. The same minimal
modification, uniformly, in every fiber but one.

### 1.3 The partition survives; its labels do not

| n | block sizes, before **and** after | spectra |
|---:|---|---|
| 5 | [1, 7, 7] | all differ |
| 6 | [1, 1, 7, 7, 7, 8] | all differ |
| 7 | [1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 8, 9] | all differ |

Not just the same sizes — the **identical blocks**, as set partitions of the
fiber labels. Those 7s are the size-7 Fano orbits and the 1s the fixed seams that
this corpus's own orbit theorem predicts; the flip leaves that stratification
exactly as it was and gives every stratum a new spectrum.

---

## 2. What this does to R15's statement

R15 concluded: *a verdict token's resolution is bounded by the invariance group
of the proposition it states*. That stands, but R15 characterised the group too
narrowly as "maps preserving |X|". The truth is larger and more structural:

> A check testing **|partition|** is blind to every map that acts **within
> blocks** — the partition-preserving maps. Count-preservation is a consequence,
> not the mechanism.

That matters for the repair. R15 proposed binding the *witness* — a hash of the
sorted set of spectra — and verified it discriminates. R16 explains why that
repair is the right shape: the witness is precisely the labelling the flip
destroys while the partition survives. Binding the count binds a number; binding
the witness binds the labelling; only the second sees a within-block map.

**And it locates the general hazard.** Any claim of the form *"there are exactly
N equivalence classes"* — a common shape for classification results — has this
blind spot by construction. The stronger the classification theorem, the coarser
its verdict token, because the token states a cardinality and the content is a
labelling.

---

## 3. What this is NOT

- **Not a proof.** Step 1.1 is arithmetic and holds for all n. Steps 1.2 and 1.3
  are **measurements** at n = 5, 6, 7. The inference that a uniform two-edge
  change *must* preserve the classification is supported by those measurements,
  **not established** *in this rung*: it would follow from the change being
  equivariant for whatever relation makes fibers equivalent. **R21 establishes
  exactly that equivariance** — both generating relations are F₂-linear and fix
  `h` — so the inference is now a theorem. The concession is kept because it was
  true of R16, which shipped without it.
- **Not a refutation of anything.** As in R15: a perturbed sign table is not a
  Cayley–Dickson algebra, so nothing here bears on the n ≤ 8 completeness claim
  for the real tower. This measures the reach of a **check**.
- **Not a complete description of the group.** One element is exhibited per
  level. Whether the partition-preserving maps form a group with more
  exhibitable elements — and whether σ(H/2, H+H/2) generates anything — was not
  investigated.
- **Not a compiler change.** Still Python-only.

---

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r16_contract.py
# expect: C1 the XOR identity for n=5..12, C2 two edges per fiber,
#         C3 identical blocks with differing spectra,
#         SELF_FALSIFYING_R16_VERDICT
#           INVARIANCE_GROUP_IS_PARTITION_PRESERVING_NOT_MERELY_COUNT_PRESERVING

bash scripts/ci/self_falsifying_compilation_line_r16_gate.sh
```

`n = 5` and `n = 6` are derived live; `n = 7` is read from
`scripts/research/r16/recorded.json` because its partition costs minutes. The
construction is reused from R15's contract by import — copying it would be the
exact failure R6 measures, inside the arc that measures it.

---

## 5. AI disclosure

Probes, contract, gate and spec drafted under human direction (2026-07-28). The
arithmetic in §1.1 was derived by hand and then verified mechanically for
n = 5…12. §1.2 and §1.3 are machine-measured. The limit in §3 — that the step
from uniform local change to preserved classification is inferred, not proved —
was written before the probes ran. No clinical content. GAIDeT-ICMJE 2025.
