<!-- docs:meta
topic_id: repo.docs.audit.stats-madaros-singlefile-smoke-2026-07-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stats-madaros-singlefile-smoke-2026-07-18
-->

# Madaros single-file stats smoke — 2026-07-18

## Scope

| | |
|---|---|
| Driver | `tests/stdlib/stats/test_madaros_singlefile_smoke.sio` |
| Gate | `bash scripts/stats_madaros_singlefile_smoke_gate.sh` → `STATS_MADAROS_SINGLEFILE_SMOKE_GATE_OK` |
| Engine | **default** `bin/souc` (Madaros v0.80.0) — `SOUNIO_SOUC_ENGINE` unset |
| Style | Fully inlined; **no** `use` imports |

## Proven

Welch t and Welch–Satterthwaite df on drug/placebo n=8 match closed-form oracle
t=6.657160298051489, df=10.141137519725348 (abs err &lt; 1e-9).

## Claims not made

- Madaros multi-module native path / D1–D3 trust
- Full scipy.stats API
- lean_single substitution for multi-module suite (that remains lean_single)
