<!-- docs:meta
topic_id: repo.docs.audit.ode-epistemic-zero-params-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ode-epistemic-zero-params-2026-06-02
-->

# ODE epistemic E2E — zero parameters (exit 1)

**Date:** 2026-06-02  
**Tests:** `tests/stdlib/ode/test_epistemic_pk_fit_e2e.sio`, `test_epistemic_pkpd_e2e.sio`

## Symptom

Gauss-Newton printed `ka=0`, `ke=0`, `converged=0` after SIGSEGV class was fixed (`pub struct EpistemicDual`, `pub fn edual_*`).

## Root causes (two codegen bugs)

### 1. `EpistemicDual` aggregate return (4×f64)

`pk_conc_dual() -> EpistemicDual` returned zeros to callers. **Workaround:** `pk_conc_dual_val` / `pk_conc_dual_dot` return scalars only (duplicate AD body).

### 2. `PKFitResult` / `PKPDFitResult` SRET to caller

Internal fit was correct (`epistemic_pk_gauss_newton_ka` returned ≈0.23–0.5), but `let r = epistemic_pk_gauss_newton(...); r.ka` read **0** from the caller stack.

**Workaround:** out-parameter API:

```sio
var result: PKFitResult
// field init …
epistemic_pk_gauss_newton(&!result, obs_t, obs_c, obs_u, dose, vd, ka_init, ke_init)
```

Same for `epistemic_pkpd_fit(&!out, ...)`.

## Post-fix evidence

```
ka ≈ 0.506, ke ≈ 0.202, converged=1
PASS: epistemic PK fitting e2e
PASS: epistemic PK/PD inference e2e
```

## Compiler follow-up (not done here)

- Fix SRET slot count / reload for structs >2×f64 returned across bundle modules.
- Re-enable value-return API once `emit_reload_sret_ptr_to_rax` covers all aggregate sizes.

## Files touched

- `stdlib/autodiff/epistemic_dual.sio` — `pub fn edual_{const,new,add,sub,mul,div,scale,exp}`
- `stdlib/ode/epistemic_pk_fit.sio` — scalar AD helpers, `&!PKFitResult` out API
- `stdlib/ode/epistemic_pkpd_fit.sio` — uses `pk_conc_dual_val/dot`, `&!PKPDFitResult`
- Both E2E tests — removed `known-failure`, out-param call pattern
