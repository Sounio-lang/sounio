<!-- docs:meta
topic_id: repo.docs.research.functor-f-phi-fp-equivariant-spec-2026-07-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.functor-f-phi-fp-equivariant-spec-2026-07-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Functor F — Φ_fp made G₂-equivariant, over a continuous G₂ orbit

**Date:** 2026-07-25
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `E_GREEN` (4/4)
**Parents:** `functor_f_g2_equivariance_spec_2026-07-25.md` (`H_CHARACTERISED`), `functor_f_g2_covariance_spec_2026-07-25.md` (`G_GREEN`)
**Harness:** `scripts/research/functor_f_phi_fp_equivariant_contract.py`

---

## 0. What this closes

The equivariance characterisation (`H_CHARACTERISED`) proved that Φ_fp's polar dial
`b`, extracted by a coordinate `argmax`, is **not** G₂-covariant, and that a pairing
`b_cov := ⟨α, e_m⟩` against a configuration-determined direction restores it. Two
edges remained (that note, §7): **(2)** propagate `b_cov` into the Φ_fp ladder so the
whole C/D/Betti stack is stated equivariantly, and **(3)** replace the finite sample
of automorphisms with a *continuous* one-parameter subgroup, exhibiting the
obstruction as a smooth curve. This contract does both, and adds the decisive
observation: under the old coordinate a symmetry of the algebra can **flip which
semantic pole the model selects**.

---

## 1. The continuous orbit (item 3)

Derivations of 𝕆 are exactly `𝔤₂ = Lie(G₂)`. We build an explicit inner derivation
(Schafer) `D_{a,b}(x) = [[a,b],x] − 3(a,b,x)` from a fixed generic imaginary pair
`(a,b)`, verify numerically that `D` is a derivation (`residual = 1.8e-15`,
`D(1) = 0`), and take the one-parameter subgroup

```
g(t) = exp(t·D) ⊂ G₂,     t ∈ [0, 3]   (25 samples)
```

with a hand-rolled scaling-and-squaring matrix exponential (no scipy in-tree). Every
sampled `g(t)` is re-verified as an automorphism over all 64 products
(`worst = 6.9e-15`).

---

## 2. Result — one coordinate wiggles, one is flat

Worked jet `α = [e₁+2e₄, e₂, e₃] = 4·e₅` (so `b = 4`, `a = 3`, `x_D = −0.5961`).

| Quantity along `g(t)` | Range over the orbit | Reading |
|---|---|---|
| `a = A₀ + ‖α‖²/4` | `2.4e-14` | invariant (`G₂ ⊂ SO(7)`) |
| `b_cov = ⟨α, e_m⟩` (transported `e_m`) | `2.4e-14` | **invariant** (the fix) |
| `b_argmax` (coordinate readout) | **`7.291`** | **not invariant** (the obstruction) |
| `x_D` deepest well, **argmax**-`b` | **`1.101`** (and **sign-flips**) | path class not covariant |
| `x_D` deepest well, **`b_cov`** | `6.7e-16` | **path class covariant** |
| Betti-0 drop `2 → 1`, `b_cov` | invariant across orbit | homology witness covariant |

**Orbit sample:**

```
 t     b_argmax   b_cov      a       x_D|argmax   x_D|b_cov
 0.00   +4.0000   +4.0000  +3.0000    -0.5961      -0.5961
 0.75   -2.6808   +4.0000  +3.0000    +0.4218      -0.5961   <-- pole FLIPS under argmax
 1.50   -3.2351   +4.0000  +3.0000    +0.4980      -0.5961
 2.25   +3.2196   +4.0000  +3.0000    -0.4959      -0.5961
 3.00   +3.0638   +4.0000  +3.0000    -0.4749      -0.5961
```

At `t = 0.75` an automorphism of 𝕆 — a *symmetry of the algebra the semantics is
built on* — sends the argmax deepest well from `−0.60` to `+0.42`: the model would
land in the **opposite pole of the contradiction path**. This is not numerical
noise; it is the coordinate `argmax` reading off a different axis after the jet is
rotated to full support. Under `b_cov` the selected pole is fixed for all `t`.

---

## 3. Φ_fp restated (item 2)

Canonical polar coordinate:

```
Φ_fp:  a(α) = A₀ + ‖α‖²/4                         (unchanged; already invariant)
       b(α; cfg) = τ + ⟨α, e_m(cfg)⟩ / 2          (was: τ + α_im[argmax|α_im|]/2)
```

`e_m(cfg)` is the imaginary axis fixed by the `(line, off-line)` configuration; the
functor transports it with everything else, so `b` is a **natural pairing**, not a
basis readout. On the seven basis-aligned lines `e_m = e_{argmax}`, hence `b ≡ b_arg`
there and **every `G_GREEN` observation is preserved**. Paths C/D and the Betti
witness inherit invariance because they are functions of `(a, b)` alone.

---

## 4. Contract clauses

| Clause | Statement | PASS = |
|---|---|---|
| `E1_DERIVATION` | `D_{a,b}` verified a derivation of 𝕆; `D(1)=0` | generator lies in `𝔤₂` |
| `E2_CONTINUOUS_ORBIT` | `g(t)=exp(tD)` automorphism ∀t; `range[b_argmax] > 10⁻²`, `range[b_cov], range[a] < 10⁻⁹` | one coordinate breaks, two are flat |
| `E3_LADDER_COVARIANT` | path-D well `x_D` breaks (`range > 10⁻²`, sign-flips) under argmax-`b`, invariant under `b_cov` | the fix reaches an **observable** |
| `E4_BETTI_DROP_INVARIANT` | `b_cov` Betti witness `2→1` invariant across the orbit | homology covariant |

Verdict: `FUNCTOR_F_PHI_EQUIV_VERDICT E_GREEN`.

---

## 5. What this is NOT

- **Not a claim that the old `G_GREEN` was wrong.** It was correct for uniformity on
  the basis-aligned lines, where `b_cov ≡ b_argmax`. The continuous orbit is where
  they diverge.
- **Not a construction of `G₂`** — we use one explicit `𝔤₂` derivation and its flow.
- **Not D3, not clinical.**

---

## 6. Recommended propagation (in-place edit, gated)

Make `b_cov` the definition of `b` in `functor_f_g2_covariance_contract.py`'s
`phi_fp` (keep `argmax` only as the asserted-equal on-basis shortcut), and add a note
to `functor_f_g2_covariance_spec` §2 that uniformity there is the `e_m = e_argmax`
restriction of the equivariant statement proved here. That edit touches the green
contract; it is held pending (a) a coordination-log check against the concurrent
functor_F agent and (b) the mandatory math-review offload
(`bin/llm-offload -t math-review -p xai`, CLAUDE.md §10).

---

## 7. Reproduce

```bash
python3 scripts/research/functor_f_phi_fp_equivariant_contract.py
# expect: E1..E4 PASS, FUNCTOR_F_PHI_EQUIV_VERDICT E_GREEN
```

Pure Python (numpy); CD sign law, matrix exponential, and derivation all
self-contained and self-verified.

---

## 8. AI disclosure

Contract and note produced under human direction (2026-07-25). The single
discriminating coordinate (`b`) and the unverified-`g` trap were flagged in an
advisor review; the continuous-orbit and pole-flip framing follow from the measured
table in §2. Claims bounded by the four named clauses. Commit gated on the §10
math-review offload. No clinical content. GAIDeT-ICMJE 2025.
