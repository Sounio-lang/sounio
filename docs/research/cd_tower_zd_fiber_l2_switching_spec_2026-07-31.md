<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-l2-switching-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-l2-switching-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — L2: the switching function in closed form

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `L2_SWITCHING_FUNCTION_IN_CLOSED_FORM__COBOUNDARY_EXPLICIT`
**Parents:** `cd_tower_zd_fiber_l1_reduction_spec_2026-07-31.md`, `cd_tower_zd_fiber_collapse_l1l2_spec_2026-07-31.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_l2_switching_contract.py`

---

## 0. The result

L2 said: for even-weight seams `Y`, the discrepancy
`disc(a,b) = ε_fano(τa,τb)·ε_seam(a,b)` is a **coboundary** — some `λ` exists with
`disc(a,b) = λ(a)λ(b)`. Until now `λ` was obtained by *solving* (a BFS over the resonance
graph), and the previous rung walled the natural proof route: the parity does appear as a
triangle obstruction, but **triangles do not generate the cycle space**, so "all triangles `+1`"
is necessary and not sufficient.

**This rung writes `λ` down.**

> `λ(a) = ±(−1)^{p_j(a)}`, where `p_j(x)` = parity of the bits of `x` **below** `j = lsb(Y)`.

The global sign is free (`λ` is only determined up to it) and cancels in the product, so the
statement with no free parameter is

> **`disc(a,b) = (−1)^{p_j(a) + p_j(b)}`** — `M1`, all even-weight seams, **n = 6,7,8,9**, zero
> violations.

**Why this matters more than the earlier steps.** L2 was an *existence* statement — cohomological
content, `H¹` of the resonance graph. With `λ` explicit there is no cycle argument to make at
all: the triangle wall is **bypassed, not climbed**. What remains is a sign identity in the CD
cocycle, the same shape as `A4_sub`, which the branch-induction toolkit already handles.

---

## 1. Clause table

| clause | statement | result |
|---|---|---|
| `M1_CLOSED` | `disc(a,b) = (−1)^{p_j(a)+p_j(b)}` on every even-weight seam | 0 violations, n = 6..9 |
| `M2_PARITY` | the same form **fails** on odd-weight seams, in bulk | the parity mechanism |
| `M3_SWITCH` | `λ(a) = ±(−1)^{p_j(a)}` satisfies `disc = λ(a)λ(b)` for **both** global signs | confirms the sign is free |
| `M4_WALSH` | the solved `λ` has one dominant Walsh coefficient, at index `2^j − 1`, magnitude `|domain|` | how it was found |
| `M5_OVERFIT` | freezing the mask at `7` (the n=6 value) fails at n ≥ 7 | self-catch, recorded |
| `M6_NULL` | neighbouring masks `2^{j+1}−1`, `2^{j−1}−1` both fail | the mask is exactly right |
| `M0_PARITY` | builders reproduce the in-tree `sign_table` entrywise | measured |
| `M7_LEAN` | the Lean file's `tau`/`res`/`eps` are the ones measured here | measured |

---

## 2. The parity is explained, not assumed

`M2`: the same closed form fails on odd-weight `Y` — `528/1680` at n=6, up to `299088/925728`
at n=9. So even/odd is not an extra hypothesis bolted onto the collapse law; **it is exactly the
locus where this `λ` works**. That is the mechanism the lane had only observed.

---

## 3. Two errors this rung made, both caught by clauses

Recorded rather than silently fixed, because both would have shipped as false statements.

1. **An overfit to one instance** (`M5`). At n = 6 there is exactly **one** even-weight seam,
   with `j = 3`, so the mask is `7` and the first fit read "parity of the low **three** bits".
   It fails at n = 7 and n = 8, where `j = 4, 5` (`128/1752`, `1280/14904`). The mask is
   `j`-dependent. Fitting a closed form on a level with a single data point is exactly how the
   lane's earlier deflated claims began.
2. **A wrong exponent in the narrative** (`M4`). The clause first asserted the dominant Walsh
   index was `2^{j+1} − 1`; it is `2^j − 1`. `M4` failed (`rc=1`) while `M1` — which uses the
   mask directly — was unaffected. The result was right and the story about it was wrong; only
   the gate distinguished them.

---

## 4. Not claimed

- **L2 is not proven ∀n.** `M1` is measured at four levels.
- **Non-existence of any `λ` for odd weight is not proven.** `M2` shows *this* `λ` fails there;
  the non-existence rests on the previous rung's triangle obstruction, which is evidence.
- **The two identities are not Lean-proven.** The **reduction** is:
  `formal/lean4/SounioZDCollapse.lean` proves ∀n that (★) together with this rung's closed form
  imply `Φ` both **preserves and reflects** adjacency — i.e. (c) — with both identities as
  explicit hypotheses (`Phi_preserves_adj`, `Phi_reflects_adj`; no `sorry`, no `native_decide`).
  `M7` pins that file's `tau`/`res`/`eps` to the objects measured here, so the implication is
  about the same graph. Before that file, the sufficiency of the two identities was prose.
- (c) as a whole is still open. What has changed is its shape: with L1 reduced to (★) and L2's
  `λ` explicit, **(c) is now two explicit sign identities** rather than one equivariance plus one
  cohomological existence claim.

---

## 5. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_l2_switching_contract.py
```
