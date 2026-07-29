<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r19-2026-07-28
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r19-2026-07-28
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R19 — R16's locality derived, and what is left reduced to one lemma

**Date:** 2026-07-28
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `LOCALITY_DERIVED__EQUIVARIANCE_REDUCED_TO_ONE_LEMMA`
**Parents:** `self_falsifying_compilation_line_r16_2026-07-28.md` (the measurement and the inference this attacks)
**Harness:** `scripts/research/self_falsifying_compilation_line_r19_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r19_gate.sh`

---

## 1. Result

R16 measured that the count-preserving flip σ(H/2, H+H/2) changes exactly two
edges per fiber, in all but one, and inferred — explicitly without proof — that
such a uniform local change preserves the classification.

> **The locality half is now a consequence of index arithmetic rather than a
> measurement, and the exceptional fiber is predicted instead of observed. Two
> candidate explanations of the other half were tested and both failed. What
> remains is one equivariance lemma, stated precisely.**

Verdict: `SELF_FALSIFYING_R19_VERDICT LOCALITY_DERIVED__EQUIVARIANCE_REDUCED_TO_ONE_LEMMA`.

### 1.1 The derivation

A fiber `L = Llo | H` has vertices `(lo, lo^L, ε)` for `lo ∈ 1..H−1`, `ε = ±1`.
`mul` evaluates σ only over the 2×2 index pairs of two vertices, so the flipped
pair `(h, H+h)`, `h = H/2`, is reachable only where one vertex carries `h` as
its **lo** and another carries `H+h` as its **hi**:

- **P** := the vertex-pair with `lo = h`, so `hi = h ^ L`
- **Q** := the vertex-pair with `hi = H+h`, so `lo = (H+h) ^ L = h ^ Llo`

| | statement | status |
|---|---|---|
| **L1** | `P = Q ⟺ h = h ^ Llo ⟺ Llo = 0` — the one fiber never examined. So `P ≠ Q` in every examined fiber. | derived |
| **L2** | `Q` exists iff `1 ≤ h ^ Llo < H`, which fails **exactly** when `Llo = h`. So exactly one fiber is untouched, and it is `Llo = H/2`. | derived |
| **L3** | The effect is to **add** the crossing matching `P_ε — Q_{−ε}`; `P` and `Q` are non-adjacent beforehand and the same-sign pairs are untouched. | derived + verified |

**L2 is the substance.** R16 reported "one fiber changes nothing" as an
observation with no account of which or why. It is a consequence of `h ^ Llo`
degenerating to `lo = 0`, and the fiber is named in advance. Verified at n = 5
(`[8]`) and n = 6 (`[16]`).

### 1.2 Two explanations tested, both refuted

Both were mine, and both are recorded because a rung that only reports what
survived is a rung that hides its search.

- **F1 — "the graph is symmetric enough that any non-adjacent vertex-pair-pair
  would do."** *False.* Adding the same crossing matching to each of the 63
  non-adjacent pairs at n = 5 gives **8 distinct spectra**, and the flip's own
  pair sits alone in its class. The pair `{h, h ^ Llo}` is special; this is not
  generic graph symmetry.
- **F2 — "the blocks are characterised by the high bit of `Llo`."** *False at
  n = 7*, where the block `[33…40, 48]` and the singleton `[56]` break any such
  reading. The clean-looking pattern at n = 5, 6 does not survive one level up.

### 1.3 An observation worth keeping

The blocks are **stable in n**: n = 5, 6, 7 agree on `Llo ∈ 1..15`, and n = 6, 7
agree on `1..31`. A fiber's class is a function of its label alone, not of the
level it is computed at.

---

## 2. What remains

> **OPEN.** Why is the assignment `Llo ↦ {h, h ^ Llo}` equivariant with the
> spectrum-block structure?

Concretely: **marking P and Q does not refine the partition** — giving those two
vertex-pairs distinct self-loop weights and taking the spectrum yields exactly
the blocks the unmarked graphs give (measured, n = 5 and 6). That is what allows
a canonical function of the *marked* graph to preserve the partition, and it is
the lemma a proof must establish.

The gap is now one statement about one assignment, rather than R16's "a uniform
local change must preserve the classification" — which was never precise enough
to attack.

**Closed in R21** (`self_falsifying_compilation_line_r21_2026-07-28.md`): both
relations generating the blocks are F₂-linear and fix `h` — the orbit action is
the identity on seam bits and `h` is one, and `τ = swap(0, lsb(Y))` cannot move
`h` because even weight forbids `lsb(Y) = n−2`. So each carries the added edge to
the added edge, and R16's inference is a theorem. The proof needed
`cd_tower_collapse_isomorphism.py`, which was **missing from this branch until
R20 restored it** — the concrete cost of that provenance defect.

---

## 3. What this is NOT

- **Not a proof of R16's inference.** The locality half is derived; the
  equivariance half is not — *in this rung*. R21 proves it; what R19 conceded
  was true of R19 and stays on the record.
- **Not a proof for all n.** L1 and L2 are arithmetic and hold for every n; L3
  and the marking result are checked at n = 5, 6 only.
- **Not a statement about the real Cayley–Dickson tower.** As in R15 and R16, a
  perturbed sign table is not a CD algebra; this concerns the reach of a check.
- **Not a compiler change.**

---

## 4. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r19_contract.py
# expect: Y1 L1/L2 arithmetic n=5..12 and L3 against built graphs,
#         Y2 untouched fiber = [H/2] predicted, Y3 marked blocks == plain blocks,
#         SELF_FALSIFYING_R19_VERDICT
#           LOCALITY_DERIVED__EQUIVARIANCE_REDUCED_TO_ONE_LEMMA
```

Runs in about a minute; the graph construction is imported from R15's contract
rather than copied.

---

## 5. AI disclosure

Derivation, probes, contract, gate and spec drafted under human direction
(2026-07-28). L1 and L2 were derived by hand and then checked mechanically for
n = 5…12; L3 and Y3 are machine-measured. F1 and F2 were hypotheses formed and
refuted in the course of this rung, and are reported as such. No clinical
content. GAIDeT-ICMJE 2025.
