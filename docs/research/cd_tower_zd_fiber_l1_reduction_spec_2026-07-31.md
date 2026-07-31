<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-l1-reduction-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-l1-reduction-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — L1 reduced to a fiber-free statement about the sign cocycle

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `L1_REDUCED_TO_SEAM_TAU_EQUIVARIANCE_OF_Q__NOT_PROVEN`
**Parents:** `cd_tower_zd_fiber_collapse_l1l2_spec_2026-07-31.md`, `cd_tower_zd_fiber_antisymmetry_lemma_spec_2026-07-31.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_l1_reduction_contract.py`

---

## 0. The result

**L1 is not proven.** It is replaced by a smaller, cleaner statement.

The previous rung showed L1 is *parity-free* and that its real hypothesis is seam-ness. This one
strips the rest of the fiber structure away:

> **(★)** For every `Y ≠ 0`, with `j = lsb(Y)` and `τ = swap(bit 0, bit j)`:
> `Q_Y(a,b) = Q_{τY}(τa, τb)` for **all** `a, b`.

where `Q_L(a,b) = σ(a,b)·σ(a⊕L,b⊕L)·σ(a,b⊕L)·σ(a⊕L,b)` is the product of `σ` over the coset
square (and `res ⟺ Q = +1`, from the parent rung).

Compared with L1, **(★)** has no fiber, no top bit, no restriction to lo-labels — and it sits
**one level down**: L1 at level `n` with label `Y|H` reduces to (★) at level `n−1` with label `Y`.
Measured at levels 5, 6, 7, 8 (`K1`), zero violations.

---

## 1. The chain, verified link by link

No link is assumed.

| clause | link | result |
|---|---|---|
| `K2` | the four branch reductions split the **second** difference into two **first** differences, exactly: `Q_{Y\|H}(a,b) = −D1_Y(a,b)·D2_Y(b⊕Y,a)` for `b ≠ Y` | 0 violations, n=6,7,8 |
| `K3` | the τ-discrepancies of the two factors **cancel**: `e1(a,b,Y) = e2(b⊕Y,a,Y)` | 0 violations, levels 5,6,7 |
| `K4` | equivalently, one statement: `e1(a,b,Y) = e1(a, b⊕Y, Y)` | 0 violations |
| `K1` | `K4` regrouped **is** (★) | 0 violations, levels 5..8 |
| `K8` | (★) needs no seam hypothesis: it holds for **every** `Y ≠ 0` | 0 violations, levels 5,6,7 |
| `K9` | base case of (★): `Q` at a single-bit label is identically `−1` | 0 violations, levels 4..7; Lean `Qgen_pow2` ∀n |
| `K15` | (★) for single-bit labels: `Q_Y(a,b) = Q_{τY}(τa,τb)` when `Y = 2^k` | 0 violations, levels 4..7; Lean `star_pow2` ∀n |
| `K16` | on the degenerate locus, `Q ≡ −1`; `Q'` is pattern-determined | measured 5..7; Lean `Qgen_degen` ∀n |
| `K17` | the gap tuples no lemma covers also give `Q = −1` | 116 064 tuples, levels 6 and 7 — the branches are exhaustive |
| `K18` | the gap lemma's central case `Q_Y(a, H) = −1`, both `Y` positions | 20 096 checks; **Lean-proven ∀n** (`Qgen_H_right_low/_hi`) |
| `K19` | the six `= H` gap conditions have only **three roots**; `Qgen` is unconditionally symmetric | measured 5..7; the `b ⊕ Y = H` pair is **Lean-proven ∀n** |

with `D1_Y(a,b) = σ(a,b)σ(a⊕Y,b)`, `D2_Y(c,y) = σ(c,y)σ(c,y⊕Y)`, and
`e1 = D1_{τY}(τa,τb)·D1_Y(a,b)`, `e2` the same for `D2`.

`D1` is the shape of `A4_sub` from the antisymmetry rung — which is the special case `Y = b`,
and is Lean-proven ∀n. The single-bit base case is `Qgen_pow2`, also Lean-proven ∀n.

---

## 2. What carries the content, and what does not

**`K5` — the cancellation is everything.** Neither `D1` nor `D2` is τ-equivariant on its own,
and their violation counts are **identical** (`1776`, `16816`, `145200` at levels 5,6,7). So L1
is a genuine cancellation between the two factors, not a factorwise property that would have
made `K3` trivial.

**`K6` — a gap in an attractive derivation, recorded rather than shipped.** By `antisym` one
expects `D2(c,y,Y) = D1(y,c,Y)`, which would yield `K4` from `K3` in one line. That identity
**fails on the degenerate locus** (`360`, `1736`, `7560` violations). `K4` is therefore
*measured directly*, not derived that way. Had the derivation been written without testing it,
the spec would have carried a false step.

**`K7` — a control whose first reading was WRONG, corrected here.** With a *mismatched* `τ` —
`j` frozen at 3 instead of `j = lsb(Y)` — the equivariance fails in bulk (`6912/28672` at level
5). That is all the measurement shows. The clause originally concluded "(★) is a statement about
seam labels"; **that was wrong** — the failure comes from using the wrong `τ`, not from `Y` being
a non-seam.

**`K8` — the seam hypothesis is not needed at all.** With the matching `τ`, (★) holds for
**every** `Y ≠ 0`, seam or not (levels 5,6,7, zero violations). So (★) is strictly more general
than L1 requires, and the seam condition can be dropped from its statement. The corrected reading
makes the target *easier*, not harder.

---

## 3. Not claimed

- **L1 is not proven, and neither is (★).** This is a reduction, verified at four levels.
- The `b = Y` boundary is excluded from `K2` and handled nowhere here. L1 itself was verified
  including it by the previous rung, so the gap is in this chain, not in L1's evidence.
- **(★) is not Lean-proven in full for multi-bit labels.** The mutual inductive
  step (16-case table, all reduction lemmas proven) is still not assembled.
  What is kernel-checked ∀n in `formal/lean4/SounioZDFiberAntisym.lean`:
  `A4_sub'`, `Qgen_coset_*`, all 16 `Qred`/`Q'red` rows, **base case**
  `Qgen_pow2`, and **single-bit equivariance** `star_pow2`.
- **The base case IS formalised (K9 / `Qgen_pow2`).** `Q` at a *single-bit* label is
  identically `−1` for all `a,b` and all bit positions `k < m` — proven ∀n.
- **(★) holds for every single-bit label (K15 / `star_pow2`).** Both sides of
  `Q_Y(a,b) = Q_{τY}(τa,τb)` are the constant `−1` when `Y = 2^k`, so the
  equivariance is a corollary of the base case.
- **Degenerate locus closed for `Q` (K16 / `Qgen_degen`).** On any of the six
  degeneracies, `Q ≡ −1` (proven ∀n). With `τ` preserving the pattern, both
  sides of (★) are that constant (`star_both_degen`). `Q'` pattern lemma not yet
  in Lean.
- **The mismatch is bridged (K17).** The reduction lemmas' hypotheses are about the
  **reduced** arguments; `Qgen_degen` is about the **current** ones, and they do not
  coincide — a tuple can be non-degenerate at level `m+2` and reduce to a degenerate one
  at `m+1` (e.g. `a = x`, `b = H`, `Y = W`: `b ≠ 0` and `b ⊕ Y ≠ 0` hold above, but the
  reduced `v = 0`). Every one of those gap tuples gives `Q = −1` — the same constant the
  degenerate branch gives. `116 064` of them at levels 6 and 7, zero exceptions.

  So the assembly has exactly **three exhaustive branches**, plus the base case:

  | branch | closes by | status |
  |---|---|---|
  | degenerate at `m+2` | `Qgen_degen`, both sides `−1` | **proven ∀n** |
  | reduces to a degenerate tuple | the same constant `−1` | `b = H` and `b ⊕ Y = H` **proven ∀n**; `a = H` and `a ⊕ b = H` not |
  | otherwise | one of the sixteen reduction lemmas + the mutual IH | **proven ∀n** |
  | base case | `Qgen_pow2` | **proven ∀n** |

- **Multi-bit non-degenerate assembly still open.** All 16 reduction cases are
  proven (K12–K14); what remains is the gap lemma, the `Q'` pattern lemma, and wiring
  them under the mutual IH.
- (c) as a whole is untouched: **L2 remains where the previous rung left it**, with its
  triangle route walled.

---

## 4. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_l1_reduction_contract.py
```

`K0` pins the builder to the in-tree `sign_table`. `K1` at level 8 dominates the runtime.
