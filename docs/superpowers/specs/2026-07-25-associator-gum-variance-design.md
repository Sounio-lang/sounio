<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-25-associator-gum-variance-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-25-associator-gum-variance-design
-->

# Design: Associator GUM variance numerical check (Experiment 2)

**Date:** 2026-07-25  
**Orthography:** EN-UK  
**Status:** Implementation  
**Depends on:** Experiment 1 receipt (`results/faers_fano_order_asymmetry/`)  
**Source analysis:** `docs/research/variance_of_associator.md` (β-thread)

---

## Question

Is the first-order GUM variance of  
`A = ‖[a,b,c]‖²` with `[a,b,c] = (a·b)·c − a·(b·c)`  
correctly recovered by (i) finite-difference GUM, (ii) Monte Carlo, and  
(iii) a **covariance-blind stepwise** model that mimics independent  
`Var(L−R) = Var(L)+Var(R)` component propagation?

## Cases

### Case F — Fano triple (primary)

- `a = e₁`, `b = e₂`, `c = e₄` (nonassociative generator triple)
- Unperturbed: `[a,b,c] = 2 e₇`, `A = 4`
- Only `a₁` carries uncertainty `σ²` (other components exact)
- **Truth (FO GUM / analytical):** `Var(A) = 64 σ²`
- **Stepwise blind model:** propagate component vars through mult;  
  `Var(d₇) = Var(L₇)+Var(R₇)`; then `Var(A) ≈ (2 d₇)² Var(d₇)`  
  → arithmetic-consistent value **`32 σ²`**  
  (note: the research note’s “16 σ²” line drops a factor of 2 in the last step; the experiment reports both the consistent stepwise value and the note’s printed figure)

### Case Q — Quaternion subalgebra (qualitative)

- `a,b,c ∈ ℍ ⊂ 𝕆` (components 4..7 zero)
- Truth: `A ≡ 0`, first-order `Var(A) = 0`
- Stepwise blind model must report **strictly positive** variance if covariance is ignored on `L−R`

### Case C — Scalar covariance blindness (structural)

- Pure arithmetic analogue of β: `Var(x−x)` under independence = `2σ²`, truth `0`
- Confirms the mechanism without octonion multiplication

## Decision rules

| Marker | Condition |
|---|---|
| `FANO_TRUTH_MC_OK` | `|MC_var − 64σ²| / (64σ²) < 0.10` for σ=0.05, N≥10000 |
| `FANO_STEPWISE_BIAS_OK` | `truth_fo / stepwise > 1.5` (underestimate) |
| `QUAT_TRUTH_NEAR_ZERO_OK` | FD or MC |Var| below absolute threshold |
| `QUAT_STEPWISE_POSITIVE_OK` | stepwise model `> 0` |
| `SCALAR_BLIND_OK` | blind formula `2σ²` matches expected constant |
| `ASSOC_GUM_EXPERIMENT_PASS` | all of the above |
| `ASSOC_GUM_EXPERIMENT_FAIL_HONEST` | any required check fails |

Primary σ grid: `{0.02, 0.05, 0.10}` for Fano ratio table (receipt secondary).

## Deliverables

```
experiments/associator_gum_variance/
  PROTOCOL.md
  README.md
  associator_gum_variance.sio
  run_and_receipt.sh
results/associator_gum_variance/
  receipt.v1.json
  RUNLOG.txt
scripts/ci/associator_gum_variance_gate.sh
```

## Non-goals

- Compiler fix (β Part A/B) — measurement only  
- Knowledge\<Octonion\> native `variance_of` path (ζ capacity separate)  
- Clinical language  

## Allowed claim

> On the locked Fano-triple associator probe, first-order GUM truth `64σ²` matches Monte Carlo within 10%, while a covariance-blind stepwise model underestimates by ~2× (consistent arithmetic). Quaternion subalgebra truth is zero variance; stepwise remains positive.
