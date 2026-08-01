<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-l2-reduction-spec-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-l2-reduction-spec-2026-08-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — L2 reduced to a fiber-free statement about the τ-discrepancy of σ

**Date:** 2026-08-01
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `DIAMOND_IS_LEVEL_BOUNDED__PARITY_MECHANISM_IDENTIFIED`
**Parents:** `cd_tower_zd_fiber_l2_switching_spec_2026-07-31.md`, `cd_tower_zd_fiber_l1_reduction_spec_2026-07-31.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_l2_reduction_contract.py`

---

## 0. The result

**L2 is not proven.** It is replaced by a smaller statement, in the same move that worked for L1.

With the **τ-discrepancy of the cocycle**

```
g(x,y) = σ(τx, τy) · σ(x,y)
```

and `j = lsb(Y)`, `p_j(x)` = parity of the bits of `x` below `j`:

> **(♦)** For even-weight `Y`, and `a ≠ 0`, `b ≠ 0`, `b ≠ Y`:
> `Qgen'(Y,a,b) = −1` ⟹ `g(a,b) · g(a⊕Y, b⊕Y) = (−1)^{p_j(a) + p_j(b)}`

No fiber, no top bit, one level down — the same shape (★) has for L1. Measured at levels 5, 6, 7 (`N6`), zero violations in 212 712 checks — and the **reduction to
it is Lean-proven ∀n** (`l2_reduction`), so (♦) is the only measured link in L2's chain.

The previous rung had already removed the cohomology from L2 by writing λ in closed form. This
one removes the fiber.

---

## 1. The chain, verified link by link

| clause | link | result |
|---|---|---|
| `N4` | the fiber-level discrepancy **is** `g(a,b)·g(b⊕Y,a⊕Y)` at one level down — via the `R_ll`/`R_uu` branch reductions | **Lean-proven ∀n** (`l2_reduction`); the clause pins it |
| `N1` | `g` is **symmetric**, so the argument swap in `N4` disappears | 0 violations, 122 880 entries — **and already Lean-proven ∀n** |
| `N5` | the reduced resonance predicate is the proven Lean lemma `Qred_hi_ll`: `Qgen(Y+H,a,b,n+1) = −Qgen'(Y,a,b,n)` | 0 violations, 566 208 checks |
| `N6` | (♦) itself | 0 violations, levels 5,6,7 |

`N1` is not new work. `g(x,y) = g(y,x)` is equivalent to `chi(τx,τy) = chi(x,y)` for the
commutation sign `chi(x,y) = σ(x,y)σ(y,x)`, and that is **`chi_tau`, proven ∀n**. It earns its
keep at exactly one step: `R_uu` returns its arguments **swapped**, so the raw reduction gives
`g(b⊕Y, a⊕Y)`, and `gdisc_symm` is what turns that into `g(a⊕Y, b⊕Y)`.

**`N4` is proven too, not measured.** `l2_reduction` and `l2_reduction_symm` in
`formal/lean4/SounioZDFiberAntisym.lean` are kernel-checked ∀n: four branch reductions plus
`tau_seam`/`tau_xor`, with `b ⊕ Y ≠ 0` as `R_uu`'s branch condition — and it governs **both**
sides, since `τ(b⊕Y) = 0 ↔ b⊕Y = 0` by `tau_inj`. So **`L2 ⟸ (♦)` is a theorem** and (♦) is the
only measured link left. That is strictly better than where L1 stands, whose `K2`/`K3`/`K4` are
genuinely measured and unproven.

---

## 2. What carries the content, and what does not

**`N2` — `g` does not factor, and this is the whole point.** The obvious hope is that `g` is
itself a coboundary, `g(x,y) = μ(x)ν(y)`; then L2 would follow in one line with
`λ(a) = μ(a)μ(a⊕Y)`. The rectangle test refutes it in bulk (52 812 / 122 880). So the coboundary
in (♦) is **created by the pairing along `Y`**, not inherited from `g` — L2's analogue of `K5`.

The probe was worth running and it was worth running *first*: it is cheap, and had it come back
positive the rung would have been a one-liner. It also had to fail. If `g` factored, a λ would
exist for **odd-weight** `Y` too, contradicting the previous rung's triangle obstruction.

**`N3` — the F2-linear route is walled.** `R21` closed the ZD locality lemma by finding the
F2-linear identity behind it. `g` is F2-additive in each argument **only for `j ≤ 2`**; from
`j = 3` it fails in bulk. So that route does not generalise here.

**`N7` — the resonance hypothesis is essential, and this is the structural difference from (★).**
Unrestricted, (♦) **fails** (32 832 / 561 162) — and **every** failure is off resonance, none on
it. So L2 is genuinely a statement *on the resonance graph*, whereas (★) is an unrestricted
identity of the cocycle. That difference is the thing to plan around — and *not* because an induction cannot carry a
hypothesis. `star_step_low`/`star_step_hi` thread `hnd` through all four quadrants and
`star_forall` carries `Y % 2^j = 0` down the whole recursion. It is that (♦)'s hypothesis is
`Qgen'(Y,a,b) = −1`, so an induction must **re-establish that predicate at the reduced level in
each quadrant** — which means knowing how `Qgen'` reduces and whether the sign survives. `N5`
already shows one minus sign hiding in exactly that kind of reduction. Mapping it is where the
next attempt should start.

**`N5` — a pin that caught a real error.** The first draft of this rung wrote the reduced
resonance predicate **without the minus sign** in `Qred_hi_ll`. The failure locus then came out
saying every failure was *on* resonance, contradicting the cross-tabulation that says none is.
The contradiction is what surfaced the dropped sign. The clause now pins the Lean lemma to the
measured object, in the same discipline `K21` established for τ.

---

## 2b. How `Qgen'` reduces, and what that buys — the attack on (♦)

**The map, read off sixteen proven theorem statements** (not measured — `N11` parses the
`.lean`):

| | label low | label high |
|---|---|---|
| **sign** | `+1` | `−1` |
| `Qgen` becomes | `Qgen` | `Qgen'` |
| `Qgen'` becomes | `Qgen'` if `ll`/`uu`, `Qgen` if `lu`/`ul` | same rule |

> **The sign is `−1` exactly when the LABEL is high, and nothing else touches it.** Priming is
> governed separately: from `Q` by the label's half, from `Q'` by whether exactly one of `a, b`
> is upper.

Three consequences.

**(i) `N12` — the even-weight hypothesis IS the parity of the sign flips.** Descending the
resonance predicate from level `n` to level `j+2` flips sign once per level where `Y` is high,
so the accumulated sign is `(−1)^{popcount(Y ≫ (j+2))}`, whatever `a` and `b` do. With
`lsb(Y) = j`, `weight(Y) = 1 + bit_{j+1}(Y) + popcount(Y ≫ (j+2))`, so **even weight is exactly
the statement that the accumulated sign is `−(−1)^{bit_{j+1}(Y)}`**. L2's parity hypothesis is
not an extra condition bolted on; it is the descent's own bookkeeping.

**(ii) `N9` — (♦)'s conclusion is LEVEL-BOUNDED.** `G(Y,a,b) = g(a,b)·g(a⊕Y, b⊕Y)` is invariant
under dropping a level and truncating every argument:

```
G_n(Y, a, b)  =  G_{j+2}(Y mod 2^{j+2}, a mod 2^{j+2}, b mod 2^{j+2})
```

**Proven ∀n** as `G_descend` (0 violations in 1 138 688 checks besides, as a pin). It follows
from a *single* lemma — `gdisc` itself descends:

```lean
gdisc_descend : gdisc j x y (m+2) = gdisc j (x mod H) (y mod H) (m+1)
```

unconditionally, in all four quadrants. The degenerate branches never surface because `R_ul` and
`R_uu` guard on `v = 0` while the `τ` factor guards on `τv = 0` — the **same** condition, by
`tau_inj` — so the two guards fire together and their constants (`1·1` and `(−1)·(−1)`) both
multiply to `1`, which is exactly what `gdisc` is at a zero argument. That is why `N9` has no
degeneracy exceptions: they cancel in pairs.
So (♦) is not a statement about an object that grows with `n`. Its conclusion depends only on
the bottom `j+2` bits, and its target `(−1)^{p_j(a)+p_j(b)}` only on the bottom `j`. The
unbounded direction of (♦) is entirely in the *hypothesis*.

**(iii) `N10` — the hypothesis only does work at `j ≥ 3`.** At the bottom level the defect
`G·T` is **identically `+1` for `j ≤ 2`**: (♦) holds there with no hypothesis at all. From
`j = 3` it does not (192/1024, 1344/4096, 6720/16384 for `j = 3,4,5`, and the same count for
both `Y₀`). That boundary is independently where `N3` finds `g` stops being F2-bilinear — two
different probes landing on the same `j ≤ 2`.

**`N13` — two closed forms tried and refuted.** The defect is neither `−Q'(Y₀,a,b)` nor
`−Q(Y₀,a,b)`. Recorded so they are not retried.

---

## 3. Not claimed

- **L2 is not proven, and neither is (♦).** What *is* proven ∀n is the **reduction**
  (`l2_reduction`) and its symmetry ingredient (`gdisc_symm`/`chi_tau`). (♦) itself is measured
  at three levels.
- **(♦) is not a Lean statement yet, and it is not proven.** What is proven of it is that its
  conclusion is level-bounded (`G_descend`); the implication itself, with its hypothesis, is not.
- **`N12` explains the parity hypothesis; it does not discharge it.** The sign law it rests on
  is proven; the arithmetic is trivial; but (♦) itself still needs the descent of the *object*,
  where the priming alternates.
- **The reduction is weaker than L1's.** (★) dropped the fiber from *both* the hypothesis and
  the conclusion. (♦) drops it from the conclusion, and its hypothesis becomes `Qgen'(Y,a,b) = −1`
  — fiber-free, but a genuine hypothesis, which `N7` shows cannot be discarded.
- **Nothing here is claimed about odd-weight `Y`.** `N8` shows *this* identity fails there. The
  non-existence of any λ remains the previous rung's triangle obstruction: evidence, not a proof.
- **(c) is unchanged in status.** Its (★) leg is discharged in Lean
  (`SounioZDCollapse.Phi_preserves_adj_star`); its L2 leg is this. **(c) is still open.**
- **V1 is untouched.**

---

## 4. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_l2_reduction_contract.py
```

Nine clauses, `N0`–`N8`, single verdict token. Runs in well under a second.
