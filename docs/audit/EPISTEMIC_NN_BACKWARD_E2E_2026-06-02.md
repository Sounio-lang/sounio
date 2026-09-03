<!-- docs:meta
topic_id: repo.docs.audit.epistemic-nn-backward-e2e-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epistemic-nn-backward-e2e-2026-06-02
-->

# Epistemic NN backward E2E — zero training / false timeout xfail

**Date:** 2026-06-02  
**Test:** `tests/stdlib/nn/test_epistemic_backward_e2e.sio`

## Symptom

Marked xfail for "timeout"; actual failure was **exit 1**: MSE flat at 0.25, `W[0,0]` stuck at 0.1.

## Root causes

| Bug | Effect |
|-----|--------|
| `epistemic_dense_forward` discarded activation return | Caller `out.values[*]` all zero |
| `mse_gradient() -> EpistemicVec4` SRET broken | Upstream grad zero → no weight update |
| `emat4_set(&layer.weights, …)` vs `&!` | Updates silently dropped |
| `epistemic_dense_backward() -> EpistemicLayerGrad` SRET broken | Grad buffer zero (fixed with `out: &!`) |

## Fixes

1. **`epistemic_dense_forward(out: &!EpistemicVec4, …)`** — linear map into `out`, then `epistemic_apply_activation_inplace`.
2. **`mse_gradient(out: &!EpistemicVec4, …)`** — out-parameter API.
3. **`epistemic_dense_backward(out: &!EpistemicLayerGrad, …)`** — out-parameter API.
4. **`emat4_set(&!…)`** in backward update and grad accumulation.
5. **`epistemic_mlp_forward(out: &!EpistemicVec4, …)`** — same pattern.

## Post-fix

```
Initial MSE: 0.16 → Final: 0.00024
W[0,0]: 0.1 → 0.356
PASS: epistemic backward pass e2e
PASS: relu backward gradient gating
exit=0
```

## Compiler follow-up

General fix for multi-field struct/tuple returns across bundled imports (same class as PBPK28 / PK fit).
