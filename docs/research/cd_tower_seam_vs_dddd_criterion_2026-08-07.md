<!-- docs:meta
topic_id: repo.docs.research.cd-tower-seam-vs-dddd-criterion-2026-08-07
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-seam-vs-dddd-criterion-2026-08-07
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# `seam_coincidence` against the [DDDD] criterion — settled, with a concession

**Date:** 2026-08-07. **Trigger:** the one item the literature gate of
`cd_tower_bdi_dugger_isaksen_gate_2026-08-07.md` left open. Settled by direct computation against
the CD sign cocycle at `bits = 4,5,6,7`, not by reading.

---

## Verdict in one line

**The zero-divisor clause of `seam_coincidence` is, on the bulk of the locus, a basis-level
specialisation of [DDDD] Theorem 1.5 — that must be conceded and cited. What is not in their work is
the rest of the coincidence: the TWO-TERM annihilator with an explicit witness, the operator identity
`anti0`, the closed form of the criterion, and the corners their hypotheses exclude.**

---

## 1. What our predicates actually are (checked, not assumed)

`isZD bits l u` is defined as *"`x = e_l + e_u` has a **two-term** annihilator `e_a ± e_b`"* — a
priori strictly stronger than being a zero-divisor. **Measured: on the loHi locus the two coincide.**
Computing `dim ker L_x` over ℝ for every pair:

| `bits` | loHi pairs | `dim Ann > 0` ≠ `offSeam` | two-term ≠ `offSeam` |
|---|---|---|---|
| 4 | 56 | 0 | 0 |
| 5 | 240 | 0 | 0 |
| 6 | 992 | 0 | 0 |

with `dim Ann` taking the values `0` (on-seam) and `4, 12, 20, 28` (off-seam). So our predicate does
carry genuine zero-divisor content, and the comparison with the literature is live rather than moot.

## 2. Putting our elements into [DDDD]'s parametrisation

`x = e_l + e_u` with `l < top ≤ u` is `(e_l, e_{u'})`, `u' = u ⊕ top`, inside `A_{n+1} = A_n × A_n`
with `n = bits − 1`. [DDDD] works in `H⊥_{n+1} = C⊥_n × C⊥_n` via
`{p,q} = (1/√2)(p, −i_n p) + (1/√2)(q, i_n q)`, so our `x` is `{p, q}` with

    p ∝ e_l + σ(i_n, u')·e_{i_n ⊕ u'},    q ∝ e_l − σ(i_n, u')·e_{i_n ⊕ u'}

— two-term elements of `A_n`. Their Theorem 1.3/1.5 then reads
`dim Ann{p,q} = dim Ann p + dim Ann q`, `+4` exactly on the **D-locus**.

**Verified numerically.** On every pair satisfying their hypotheses (`p, q` non-zero, both slots in
`C⊥_n`), the dimension formula holds: 182/182 at `bits = 5`, 870/870 at `bits = 6`. And:

- every **off-seam** covered pair has `p, q` **C-orthogonal** — 168/168 and 840/840 — so their
  summary *"if `a` and `b` are C-orthogonal, then `{a,b}` is always a zero-divisor"* delivers
  `off-seam ⟹ ZD` on the covered set;
- every **on-seam** covered pair falls outside the D-locus and gets `dim Ann = 0` — 14/14 and 30/30.

**So on the covered set their dichotomy IS our dichotomy.** This is the concession, and it is
measured rather than feared.

## 3. What their criterion does not reach

Their hypotheses fail exactly when `l = i_{n−1}`, or `u' = 0`, or `u' = i_{n−1}`, or `u' = l ⊕ i_{n−1}`
(the last two being where `p` or `q` degenerates to zero):

| `bits` | loHi | uncovered | share |
|---|---|---|---|
| 5 | 240 | 58 | 24% |
| 6 | 992 | 122 | 12% |
| 7 | 4032 | 250 | 6% |

A structural remark worth keeping: **their splitting has its own seam, one level below ours.** The
pairs it cannot see are precisely those meeting `i_{n−1}` — the distinguished element of the previous
doubling — while our seam is at `i_n`.

The uncovered **on-seam** pairs are still settled by the literature, just by different and more
elementary results: `u = top` gives `x = (e_l, 1)`, which has a component along `i_{n+1}` and so is
not a zero-divisor by [DDD] Lemma 9.5; `l ⊕ u = top` gives `x = (a, ±a)` with `a = e_l`, and [DDD]
Theorem 10.1 gives `Ann(αa, βa) = Ann(a) × Ann(a) = 0` since `L_{e_l}` is a signed permutation.

## 4. What is ours

1. **The two-term annihilator.** `hasXorAnnih = offSeam` says the annihilator can always be taken
   with **support 2**, with an explicit witness `e_a ± e_b` — even though the annihilator has
   dimension up to 28 at `bits = 6`. [DDD] and [DDDD] compute annihilator *dimensions* and describe
   *subspaces*; neither ever asks for, or produces, a minimal-support witness. This is a statement of
   a different kind, and nothing in their machinery yields it.
2. **`anti0 = ¬offSeam`** — the operator identity `{L_l, L_u} = 0`. Absent from their work entirely.
3. **A closed criterion instead of a recursion.** `offSeam` is `O(1)` in the two indices. [DDDD]
   Theorem 1.5 is an inductive reduction whose branch depends on D-locus membership, which the
   authors themselves describe as depending *"not just on an understanding of zero-divisors in `A_n`
   but also on a detailed understanding of annihilators in `A_n` … not as explicit as we might like."*
4. **The corners of §3, and ∀n coverage**, anchor-free and machine-checked.

## 5. Actions taken

- The paper must state that the ZD dichotomy on this locus is a basis-level specialisation of
  [DDDD] Thm 1.5 (and, at `n = 4`, of [DDD] Prop 12.1 / Moreno / Khalil–Yiu), and cite accordingly.
- The claim carried forward as ours is the **coincidence**, not the dichotomy: two-term witness +
  operator identity + closed criterion, ∀n.
- `seam_coincidence`'s standing in the arc is downgraded from "our theorem" to "our formalisation
  and our refinement of a known dichotomy".
