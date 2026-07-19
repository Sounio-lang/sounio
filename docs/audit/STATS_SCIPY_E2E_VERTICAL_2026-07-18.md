<!-- docs:meta
topic_id: repo.docs.audit.stats-scipy-e2e-vertical-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stats-scipy-e2e-vertical-2026-07-18
-->

# Stats scipy.stats-class E2E vertical — 2026-07-18

## Scope

Thin E2E vertical proving **scipy.stats-class inference** under Sounio, plus the
structural edge SciPy cannot express: **GUM bridge on every `TestResult`**.

| | |
|---|---|
| Worktree | `/workspace/sounio-e2e-feature-vertical` |
| Branch | `feat/e2e-feature-vertical` |
| Driver | `tests/stdlib/stats/test_scipy_e2e_vertical.sio` |
| Gate | `bash scripts/stats_scipy_e2e_gate.sh` |
| Oracle | `scripts/stats_scipy_e2e_oracle.py` (pure-Python closed form; SciPy optional) |
| Receipt | `artifacts/stats/scipy_e2e_receipt.v1.json` |
| Engine | **`SOUNIO_SOUC_ENGINE=lean_single` only** |

## What is proven

1. **Welch two-sample t** (`stats::hypothesis::t_test_two_sample`) on drug vs
   placebo n=8 (arrays from `tests/stats/test_epistemic_stats.py`).
2. **Cohen's d** via both `TestResult.effect_size` and `stats::effect_size::cohens_d`.
3. **Levene W** (center = mean) for the same two groups.
4. **OLS** slope / intercept / r² via `stats::validation` (textbook x=[1..5],
   y=[2,4,5,4,5]).
5. **Bootstrap mean** seed-locked (`bootstrap_mean`, seed `20260715`).
6. **Normality pipeline** Jarque–Bera + Q–Q PPCC (explicitly **not** Shapiro–Wilk).
7. **Paired t** on trough before/after + GUM bridge.
8. **`TestResult.as_gum`**: `value == statistic`, `std_u == SE` of the mean
   difference (via `gum_simple`).

## Derived oracles (not retrofitted)

Drug / placebo closed form (sample variance, Welch–Satterthwaite, two-tail p via
regularized incomplete beta matching `hypothesis.sio`):

| Quantity | Value |
|---|---:|
| t | `6.657160298051489` |
| df | `10.141137519725348` |
| se | `1.070276165963052` |
| p | `≈ 5.292274974446240e-05` |
| Cohen's d | `3.328580149025744` |
| Levene W (mean center) | `5.629049872588284` |
| OLS slope / intercept / r² | `0.6` / `2.2` / `0.6` |

Tolerances in the driver: t/df/d/se/W abs `1e-9` (p abs `5e-7`); OLS abs
`1e-12`; bootstrap SE band `0.08` around `√(2/5)`.

## Claims **not** made

- Full `scipy.stats` API parity
- Shapiro–Wilk W/p vs `scipy.stats.shapiro`
- Default Madaros multi-module native execution
- NumPy array protocol / sklearn models (other agent lane)

## How to run

```bash
cd /workspace/sounio-e2e-feature-vertical
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
bash scripts/stats_scipy_e2e_gate.sh
# expect: STATS_SCIPY_E2E_GATE_OK
```

## Structural surpass vs SciPy

`scipy.stats.ttest_ind` returns a statistic and a p-value. Sounio's
`TestResult` also embeds `as_gum: GUMResult` so the same call site carries a
metrological standard uncertainty on the test statistic. That is the vertical's
honest "beyond SciPy" claim — not a larger function count.

## Relation to existing gates

| Gate | Role |
|---|---|
| `scripts/stats_validation_gate.sh` | OLS / descriptives run-proof only |
| `scripts/stats_epistemic_suite_selftest.sh` | ~130 module inline `ALL PASS` |
| **`scripts/stats_scipy_e2e_gate.sh`** | **Cross-module vertical + receipt + optional oracle** |

## Next options (out of this vertical)

1. Public Shapiro W/p module with SciPy parity.
2. statsmodels-style OLS full diagnostics E2E (`regression/linear` epistemic).
3. Madaros multi-module promotion after D1/D3 (see `EPISTEMIC_TRUST_MAP_2026-07-14`).
