<!-- docs:meta
topic_id: repo.docs.dissertation.results.runs.superseded-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.results.runs.superseded-2026-09-26
-->

# Run logs carrying the PBPK28 floor-clamp bias (2026-09-26)

The logs below are raw captures, kept unedited as historical records. Every PBPK28 bolus simulation in
them was stepped with `pbpk28_full_cn_step`, whose negativity floors injected mass into the bolus
(AUC_blood +28% at dt = 0.05 h, +52% at 0.1 h, +140% at 0.5 h). Do not quote their numbers. See
`docs/audit/PBPK28_CN_RANNACHER_MASS_BALANCE_2026-09-26.md`.

| Stale log | Regenerated log |
|---|---|
| `../pbpk28_epistemic_runs_v1.txt` (legacy prior) | `legacy_epistemic_pbpk28_v2.txt`, `legacy_hessian_pbpk28_v2.txt`, `pbpk28_sobol_pce_v2.txt` |
| `m6_epistemic_pbpk28_v1.txt` | `m6_epistemic_pbpk28_v2.txt` |
| `m6_hessian_pbpk28_v1.txt` | `m6_hessian_pbpk28_v2.txt` |
| `m6_full_stack_v1.txt` | `m6_full_stack_v2.txt` |
| `m1_copula_sweep_v1.txt` | `mc_pbpk28_rapamycin_lognormal_v3.txt` (legacy prior; includes the copula sweep) |
| `mc_pbpk28_rapamycin_lognormal_v1_repro.txt`, `mc_pbpk28_rapamycin_lognormal_v2.txt` | `mc_pbpk28_rapamycin_lognormal_v3.txt` |
| `mc_prior_family_sweep_v1_repro.txt`, `mc_prior_family_sweep_v2.txt` | `mc_prior_family_sweep_v3.txt` (legacy), `mc_prior_family_sweep_m6_v1.txt` (M6) |
| `m6_dissertation_pbpk_suite_gate_v1.txt`, `m6_dissertation_pbpk_hessian_gate_v1.txt` | not rerun; their PBPK28 members were rerun individually above |

Not affected: `m6_dissertation_pbpk28_parity_gate_v1.txt` (parity refs run their own kernel copies at
dt = 0.001 h, where the floors never fire), `m6_literature_access_v1.txt`, `m6_julia_reconciliation_v1.txt`.
