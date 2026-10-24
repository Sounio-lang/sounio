<!-- docs:meta
topic_id: repo.docs.audit.stats-shapiro-e2e-vertical-2026-07-18
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stats-shapiro-e2e-vertical-2026-07-18
-->

# Shapiro–Wilk public W/p E2E — 2026-07-18

## Scope

| | |
|---|---|
| Module | `stdlib/stats/shapiro_wilk.sio` |
| Driver | `tests/stdlib/stats/test_shapiro_e2e.sio` |
| Gate | `bash scripts/stats_shapiro_e2e_gate.sh` → `STATS_SHAPIRO_E2E_GATE_OK` |
| Engine | `lean_single` (multi-module import of `stats::shapiro_wilk`) |
| n range | **5 ≤ n ≤ 10** only |

## Formula

The public implementation is Royston AS R94, the same approximation as tests/reference/shapiro_reference.cpp. The 1965 4 d.p. table is not what shapiro_wilk computes, so the e2e oracles follow the C++ reference.

- W from normalized normal-order weights with the Royston polynomial corrections.
- p from the Royston small-n transform for n <= 11, including n = 5.

## Fixed-sample oracles

| Sample | W | p |
|---|---:|---:|
| 1..5 | 0.98676215544771961 | 0.96717393596804047 |
| 2..11 | 0.97016461123066611 | ~0.892 (high) |
| skew 1, 1.1, ... 13 | 0.73781870358086632 | ~0.0025 (&lt; 0.05) |

## Claims not made

- Full `scipy.stats.shapiro` n-range / AS R94 exact poly for 3 ≤ n ≤ 11
- Madaros multi-module import of this module
