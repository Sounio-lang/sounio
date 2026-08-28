# PROTOCOL — Associator GUM variance (Experiment 2)

**Locked:** 2026-07-25  
**Design:** `docs/superpowers/specs/2026-07-25-associator-gum-variance-design.md`  
**Analysis:** `docs/research/variance_of_associator.md`

## 1. Statistic

```
[a,b,c] = (a·b)·c − a·(b·c)     // octonion associator
A       = ‖[a,b,c]‖²
```

## 2. Case F (primary)

- `a=e₁`, `b=e₂`, `c=e₄`
- Only `a₁` uncertain with variance `σ²`
- Analytical FO truth: `64 σ²`
- Stepwise blind: `32 σ²` (see design note on research-doc `16σ²` slip)
- MC: N=10000, seed=20260725, Box–Muller + Park–Miller
- σ primary = 0.05

## 3. Case Q

- Fixed quaternion triple in ℍ; FD GUM over 12 real inputs at σ=0.05 → near 0
- Stepwise positive check on L−R path with σ on a₀

## 4. Case C

- Structural: independent rule predicts `Var(x−x)=2σ²`

## 5. Pass / fail

All required markers must print `1` (true) for `ASSOC_GUM_EXPERIMENT_PASS`.

## 6. Run

```bash
bash experiments/associator_gum_variance/run_and_receipt.sh
```
