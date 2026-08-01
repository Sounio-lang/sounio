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
**Status:** `EXECUTABLE` — `L2_REDUCED_TO_FIBER_FREE_DIAMOND__NOT_PROVEN`
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

## 3. Not claimed

- **L2 is not proven, and neither is (♦).** What *is* proven ∀n is the **reduction**
  (`l2_reduction`) and its symmetry ingredient (`gdisc_symm`/`chi_tau`). (♦) itself is measured
  at three levels.
- **(♦) is not a Lean statement yet.**
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
