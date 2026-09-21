<!-- docs:meta
topic_id: repo.docs.audit.madaros-correlated-eq-identity-2026-07-29
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-correlated-eq-identity-2026-07-29
-->

# Madaros: correlated equality identity (2026-07-29)

**Status:** CLOSED FIXED  
**Witness:** `tests/run-pass/correlated_eq_identity.sio`  
**Gate:** `scripts/ci/madaros_correlated_eq_gate.sh`

## Pre-fix

| Check | Madaros | lean |
|---|---|---|
| cov(x,x) | **0** | 4 |
| T1 a==a | 0.1403 | 1.0 |
| T3–T5 | FAIL | PASS |

## Root causes

1. **`get_source_id_sens` → mixed `(i64, f64)` tuple** on the imported
   `correlation.sio` path zeroed id/sensitivity in covariance. Split into
   `get_source_id` / `get_source_sens` / `get_source_u` (scalar returns).

2. **Witness style:** `var y = x; y.value = 11.0` aliases large struct copies
   under Madaros multimodule (`x.value` also becomes 11), poisoning later
   cases. Witness now constructs a fresh `correlated_from_source(11.0, …)`.

## Residual (documented, not fixed here)

Large-struct by-value copy aliasing on Madaros imported path — separate
compiler defect; do not rely on field mutation of struct copies across
modules until fixed.

## Evidence

```text
ALL PASS
```

(lean_single control: ALL PASS on the same witness.)
