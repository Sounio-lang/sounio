<!-- docs:meta
topic_id: repo.docs.research.functor-f-g2-equivariance-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-g2-equivariance-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — G₂-equivariance: an obstruction and its constructive resolution

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `H_CHARACTERISED` (obstruction + fix, not a bare green rung)
**Parent:** `docs/research/functor_f_g2_covariance_spec_2026-07-25.md` (`G_GREEN`, `weak_covariance_witness`)
**Harness:** `scripts/research/functor_f_g2_equivariance_contract.py`

---

## 0. Why this exists

The `g2_covariance` contract closed the *uniformity* question — the Functor F
assignment behaves the same on all seven basis-aligned Fano lines — and labelled
itself, honestly, `weak_covariance_witness`. Its own §4 lists the deferred step:

> **Not pointwise naturality.** We do not prove `F(g·L) = g·F(L)` for all `g ∈ G₂`.

This note executes that step. The outcome is **not** a clean upgrade to full
covariance. It is a *characterisation*: one half of F is covariant for structural
reasons, one half is provably **not**, and the non-covariant half admits a
constructive fix that restores equivariance while changing nothing observable on
the basis-aligned lines. Reporting the obstruction is the deliverable — in the
ethos of this programme (`§1 what is NOT the contribution`, `D3 FORBIDDEN`,
`FAIL_HONEST`, *halt is a deliverable*).

---

## 1. Setup

`Aut(𝕆) = G₂`. We construct genuine, **generic** automorphisms `g` (not signed
basis permutations) from a random Cayley triple `(I, J, L)` — `I ⟂ J`,
`L ⟂ ⟨1, I, J, IJ⟩` — and extend multiplicatively, matching the sign law
`e_{i⊕j} = cds(i,j)·e_i e_j` on the generators `1, 2, 4`. Each `g` is **verified**
to be an automorphism over all 64 products before use (`‖g(e_i e_j) − g(e_i)g(e_j)‖`),
which is the trap: most Fano-line-preserving signed permutations are *not* in
`Aut(𝕆)`, so an unverified `g` would produce a failed test for the wrong reason.

The worked configuration is the R3/R4 line `L = (1,2,3)` with off-line unit `e₄`;
its associator jet `α = [e₁+εe₄, e₂, e₃] = 2ε·e₅` is single-axis.

---

## 2. The three layers of F under `g`

| Layer of F | Transforms as | Covariant? | Why |
|---|---|---|---|
| associator jet `α` | `α(g·x) = g·α(x)` | ✅ **yes** | definitional: `g` automorphism ⇒ `g([x,y,z]) = [gx,gy,gz]`. No content. |
| `a = A₀ + ‖α‖²/4` | `‖g·α‖ = ‖α‖` | ✅ **yes** | `G₂ ⊂ SO(7)` ⇒ norm-preserving. No content. |
| `b` via `argmax\|α_im\|` | coordinate readout | ❌ **NO** | a generic `g` rotates a single-axis vector to full 7-axis support; `argmax` and `\|coeff\|` are not `g`-invariant. |

So the *entire* covariance question reduces to a single bit: **does `b` survive a
generic `g`?** It does not.

---

## 3. The obstruction (measured, `N = 200` independent generic `g`)

`α = 2·e₅` (support 1, `\|α\| = 2`, argmax-`b = 2`). Under 200 verified generic
automorphisms:

| Quantity | Result |
|---|---|
| `g` automorphism residual (worst / 64 products) | `3.2e-15` |
| `g` orthogonality `‖gᵀg − I‖` (worst) | `4.6e-15` |
| `g` is a signed permutation? | `0 / 200` (all generic) |
| `a`-invariance `max\|Δa\|` | `6.7e-16` (invariant) |
| support of `g·α` (mean) | `7.00` (full spread) |
| argmax-`b` shift `median\|Δb\|` | `0.657` |
| fraction of `g` with `\|Δb\| > 10⁻³` | **`1.000`** |

**`H3_ARGMAX_B_OBSTRUCTED`.** Φ_fp's polar dial `b`, as currently defined by a
coordinate `argmax`, is **not** `G₂-covariant`: every generic automorphism moves it.
The `g2_covariance` witness passed only because it stayed inside the finite subgroup
of signed basis permutations, which *do* preserve single-axis support — exactly the
sense in which it was "weak".

---

## 4. The constructive resolution

Replace the coordinate readout with a **pairing against a configuration-determined
direction** `e_m` — the axis fixed by the `(line, off-line)` data, which `g`
transports along with everything else:

```
b_cov(α) := ⟨ α , e_m ⟩          (instead of  α_im[argmax|α_im|])
```

Then equivariance holds **by construction**, because `g ∈ SO(7)` preserves the inner
product:

```
b_cov(g·α) = ⟨ g·α , g·e_m ⟩ = ⟨ α , e_m ⟩ = b_cov(α).
```

Measured `max|Δb_cov| = 1.3e-15` over the same 200 `g` (`H4_PAIRING_B_COVARIANT`).
On the basis-aligned lines `e_m = e_{argmax}`, so `b_cov ≡ b` there — **nothing the
`G_GREEN` contract observed changes**; the redefinition only fixes behaviour off the
finite subgroup, which is precisely where covariance was breaking.

**Reading.** "F is a genuine functor" is not another green clause; it is the demand
for an *invariant* formulation of its outputs. `a` was already invariant; `b` becomes
invariant once it is written as a natural pairing rather than a basis-dependent
`argmax`. That is the content.

---

## 5. Contract clauses

| Clause | Statement | PASS means |
|---|---|---|
| `H1_GENERIC_G_EXISTS` | 200 verified generic automorphisms (auto-residual `< 1e-9`, orthogonal, none a signed perm) | the test map is legitimately in `G₂` |
| `H2_A_INVARIANT` | `a = A₀ + ‖α‖²/4` invariant under every `g` | covariant half (norm) |
| `H3_ARGMAX_B_OBSTRUCTED` | argmax-`b` moved by `> 10⁻³` under `> 99 %` of `g` | **obstruction confirmed** |
| `H4_PAIRING_B_COVARIANT` | `⟨α, e_m⟩` invariant to `1e-9` under every `g` | fix restores equivariance |

Verdict token: `FUNCTOR_F_G2_EQUIV_VERDICT H_CHARACTERISED`.

---

## 6. What this is NOT

- **Not `H_GREEN`-as-covariant-out-of-the-box.** Φ_fp *as written in the green
  contract* is not `G₂`-covariant; the honest verdict is a characterisation.
- **Not a construction of `G₂`.** We sample verified elements; we do not build the
  group or its Lie algebra.
- **Not a change to the `G_GREEN` contract.** That contract remains correct for the
  uniformity claim it makes. This is a disjoint, additive rung.
- **Not D3, not clinical.**

---

## 7. Suggested next edges

1. **Propagate `b_cov` into Φ_fp** as the canonical polar coordinate (spec change to
   `functor_f_g2_covariance`), so the whole ladder is stated `G₂`-equivariantly; keep
   `argmax` only as the on-basis shortcut it provably equals.
2. **Path-class covariance.** Verify C/D end-states are `g`-covariant under `b_cov`
   (expected free, since C/D depend on `b` through its sign/scale, now invariant).
3. **Continuous `g`-orbit.** Replace the finite sample with a one-parameter subgroup
   `exp(t·𝔤)` to exhibit the obstruction as a smooth curve `b(t)` vs the flat `b_cov(t)`.

---

## 8. Reproduce

```bash
python3 scripts/research/functor_f_g2_equivariance_contract.py
# expect: H1..H4 PASS, FUNCTOR_F_G2_EQUIV_VERDICT H_CHARACTERISED
```

Pure Python (numpy), self-contained; re-implements the CD sign law for audit.

---

## 9. AI disclosure

Probe, contract, and note produced under human direction (2026-07-25), after an
advisor review that identified the single discriminating bit (`b`) and warned of the
unverified-`g` trap. Math-facing claims are bounded by the four named clauses and the
measured table in §3. Commit is gated on the mandatory math-review offload
(`bin/llm-offload -t math-review -p xai`, per CLAUDE.md §10). No clinical content.
GAIDeT-ICMJE 2025.
