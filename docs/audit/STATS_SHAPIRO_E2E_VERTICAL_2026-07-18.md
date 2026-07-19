<!-- docs:meta
topic_id: repo.docs.audit.stats-shapiro-e2e-vertical-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
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

- Half-coefficients aᵢ from Shapiro & Wilk (1965) Table 5 (4 d.p.).
- W = b² / Σ(xᵢ − x̄)² with b = Σ aᵢ (x_{(n−i+1)} − x_{(i)}).
- p (n ≥ 7): Royston (1992) y = ln(1−W), μ/σ polynomials in ln n, p = 1 − Φ(z).
- p for n ∈ {5,6}: returned as **−1** (not claimed).

## Fixed-sample oracles

| Sample | W | p |
|---|---:|---:|
| 1..5 | 0.9865881 | n/a (−1) |
| 2..11 | 0.970158460121212 | ~0.941 (high) |
| skew 1..13-ish | 0.738023544970166 | ~0.0045 (&lt; 0.05) |

## Claims not made

- Full `scipy.stats.shapiro` n-range / AS R94 exact poly for 3 ≤ n ≤ 11
- Madaros multi-module import of this module
