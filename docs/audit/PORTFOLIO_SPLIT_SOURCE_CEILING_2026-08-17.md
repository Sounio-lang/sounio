<!-- docs:meta
topic_id: repo.docs.audit.portfolio-split-source-ceiling-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-08-17
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.portfolio-split-source-ceiling-2026-08-17
-->

# Portfolio split — under lexer source-byte ceiling

**Date:** 2026-08-17  
**Path chosen:** **A+C** (split + stop generated monolith), **not B** (raise 2097152)

## Before

| File | Bytes | Effect |
|------|------:|--------|
| `stdlib/theorem/portfolio.sio` | 2 109 065 | **OVER** CAP 2 097 152; silent clip at line 50272 |
| Importers `use theorem::portfolio::*` | 169 run-pass | known-failure / unparseable closure |

## After

Thin façade `portfolio.sio` (~1 KB) re-exports parts in dependency order. Each part ≪ CAP (largest ~420 KB `portfolio_core.sio`, largest solver band ~334 KB).

| Part | Role |
|------|------|
| `portfolio_kinds.sio` | `portfolio_kind_*` constants |
| `portfolio_checkers.sio` | `portfolio_checker_*` |
| `portfolio_known_manifests.sio` | `known_solver_portfolio_manifest_*` |
| `portfolio_core.sio` | entry/result/manifest core API |
| `portfolio_solver_v000_024.sio` … `v175_199.sio` | versioned `solver_portfolio_vN_*` bands |

Importers keep `use theorem::portfolio::*` unchanged.

## Measured

```text
./bin/souc check stdlib/theorem/portfolio.sio          → check: OK
./bin/souc check stdlib/systems/lorenz_i256_cert.sio   → check: OK
# 30/30 random sample of the 169 importers → check: OK
bash scripts/ci/stdlib_source_byte_ceiling_gate.sh     → PASS
```

## Do not

- Re-concatenate parts into one file over CAP  
- Raise CAP to “fit” a future monolith  
- Bisect with unbalanced braces — split only on complete `pub fn` items (as done here)

Related: `TOKEN_CEILING_BLOCKED_RUNPASS_CENSUS_2026-08-17.md`, E229 refusal (grok-cli5).
