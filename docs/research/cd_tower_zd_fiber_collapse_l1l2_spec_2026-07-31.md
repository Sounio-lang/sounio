<!-- docs:meta
topic_id: repo.docs.research.cd-tower-zd-fiber-collapse-l1l2-spec-2026-07-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.cd-tower-zd-fiber-collapse-l1l2-spec-2026-07-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# CD-tower ZD fibers — attacking (c): L1 is parity-free, and the L2 triangle route is walled

**Date:** 2026-07-31
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `COLLAPSE_L1_IS_PARITY_FREE__L2_TRIANGLE_ROUTE_WALLED`
**Parents:** `cd_tower_zd_fiber_v1_reduction_spec_2026-07-31.md`, `cd_tower_zd_fiber_antisymmetry_lemma_spec_2026-07-31.md`
**Harness:** `scripts/research/cd_tower_zd_fiber_collapse_l1l2_contract.py`

---

## 0. The result

**(c) is not proven.** This rung locates it.

(c) is the parity-collapse law: even-weight seams merge into Fano classes via
`Φ(lo,s) = (τ lo, λ(lo)·s)`, `τ = swap(bit 0, bit j)`, `j = lsb(Y)`. Its ∀n proof was reduced in
2026-07-12 to two σ-lemmas, both verified `n ≤ 8`, neither proven:

- **L1** resonance preservation: `R_{L_seam}(a,b) = R_{L_fano}(τa, τb)`
- **L2** switching balance: `ε_fano(τa,τb)·ε_seam(a,b)` is a coboundary

and the only thing on record about their relative difficulty was that "(L2) is the delicate
half". Two things come out of measuring them:

> **L1 does not see the parity.** It holds for **every** seam — odd weight as well as even.
> Its real hypothesis is *seam-ness*, not parity. So the entire content of the even/odd
> distinction that the collapse law turns on lives in **L2**.

> **The obvious route into L2 is walled.** The parity does show up as a cycle obstruction —
> even-weight fibers have all triangle discrepancy-products `+1`, odd-weight ones do not — but
> **triangles do not generate the cycle space**, so that is evidence about the obstruction, not
> a proof of L2.

---

## 1. Clause table

| clause | statement | status |
|---|---|---|
| `C0_PARITY` | builders reproduce the in-tree `sign_table`/`A_sig` entrywise | measured |
| `C1_PARITYFREE` | L1 holds for every seam, **both** parities, n = 6,7,8 | measured |
| `C2_SEAMNESS` | with a **non-seam** `L` the same equivariance fails in bulk | measured (negative control) |
| `C3_NOTBILIN` | `Q` is **not** bilinear — the second-difference route is dead | measured |
| `C4_GLOBAL` | `Q` is globally `τ`-equivariant only for `j = 1,2`, not `j ≥ 3` | measured |
| `C5_OBSTRUCT` | even weight ⇒ all triangle disc-products `+1`; odd ⇒ both signs | measured |
| `C6_WALL` | triangles do **not** generate the cycle space | measured |

---

## 2. Why L1 being parity-free matters

The collapse law is a statement *about* even weight. If L1 had carried the parity, proving it
would have been the whole job. It does not: `C1` finds `0` violations for odd-weight seams as
well as even, at n = 6,7,8. `C2` is the control that makes this informative — for `L` whose
lo-part is *not* a seam, the same equivariance fails in bulk, so L1 is a real hypothesis about
seam labels rather than a general symmetry that would hold for anything.

Consequence for the ∀n attack: **L1 and L2 should be attacked separately, and L1 first.** L1 is
now a parity-free identity about seam labels — the same shape as `A4_sub` in the antisymmetry
rung, which is an F₂ sign identity proved by induction through the four branch reductions, and
that toolkit is in-tree.

---

## 3. Two routes recorded dead

Both are recorded so the next rung does not re-walk them.

**`C3` — the second-difference route.** Since `res ⟺ P1 = P3` (proven in the parent rung) and
both are `±1`, `res ⟺ Q = +1` where

```
Q_L(a,b) = σ(a,b)·σ(a⊕L,b⊕L)·σ(a,b⊕L)·σ(a⊕L,b)
```

is the product of `σ` over the coset square `{a, a⊕L} × {b, b⊕L}` — a mixed second difference of
the cocycle. Second differences of *quadratic* forms are bilinear, which would have made L1
immediate. **`Q` is not bilinear.** This is consistent with the lane's earlier finding that the
associator becomes higher-degree Boolean at `n ≥ 6` — the same wall, met from a new direction.

**`C4` — the global-symmetry route.** `Q` is `τ`-equivariant globally only for `j = 1, 2`, the
octonion bits. For `j ≥ 3` — precisely where the collapse law lives, since `j = lsb(Y)` and `Y`
is a seam — it fails in bulk. The n = 6, j = 3 violation count reproduces the `55296` already on
record for "τ is not a signed automorphism". So L1 cannot be obtained by restricting a global
symmetry; it is genuinely a statement about the restricted domain.

---

## 4. The wall in front of L2

A `±1` edge signing is a coboundary **iff every cycle** has product `+1`. `C5` measures the
triangle products and finds a perfect split: even-weight `Y` gives `{+1}`, odd-weight gives
`{+1, −1}`. That is the parity, appearing exactly where it should — as the obstruction class.

It is not enough. Triangles certify the coboundary only if they **generate** the cycle space,
and `C6` measures that they do not: the F₂ rank of the triangle vectors falls strictly below the
cycle rank `|E| − |V| + c` for several even-weight fibers (n = 6 `Y=24`: 49 < 55; n = 7 `Y=40`
and `Y=48`: 105 < 119). So "all triangles `+1`" is **necessary but not sufficient**, and `C5`
does not prove L2.

What would close L2 is a cycle-space argument that is not triangle-local — for instance an
explicit generating set for the cycle space of the resonance graph, or a direct construction of
`λ` from the fiber label rather than by solving for it.

---

## 5. Not claimed

- **(c) is not proven**, and neither is L1. `C1`/`C2` pin L1's hypothesis and show it is
  parity-free — that reduces the target, it does not prove it.
- **L2 is untouched** except that one route into it is now walled.
- The existing λ-solver verifies the collapse constructively at `n ≤ 8`; nothing here extends
  that bound.
- `C5`'s triangle scan uses the first 26 vertices per fiber, so it is a sample of triangles, not
  an exhaustive one — enough to exhibit both signs in the odd case and to fail to exhibit `−1`
  in the even case, not enough to be a proof even of the triangle statement.
- **Numerical certificate**, exact integer sign table, `D3` class. Nothing here is Lean-proven.

---

## 6. Reproduce

```sh
python3 scripts/research/cd_tower_zd_fiber_collapse_l1l2_contract.py
```

`C0` pins the builders to the in-tree ones. `C1` at n = 8 and `C4`'s exhaustive global sweep
dominate the runtime.
