<!-- docs:meta
topic_id: repo.docs.dissertation.dossiers.figure-linkage-frozen
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.dissertation.dossiers.figure-linkage-frozen
-->

# Figure/table linkage — frozen versions (EPIC SIO-515)

Purpose: map every figure/table citation in the EPIC dossier set to the
repo-side artifact that produces it, pinned to a version and a generator
command — or mark the gap explicitly. No citation is left pointing at an
unversioned artifact.

Freezing convention: the parity gate re-runs both engines live and joins
them (no committed REC files), so a parity figure is frozen as the triple
(gate script, stdlib scenario, Node core) at the measured commit. Each
SIO-562 dossier records its own measured-at SHA; this file records the
current triple at main 3d3da08066.

## 1. Per-scenario parity figures (CI-enforced)

| Scenario | Sounio producer | Node producer | Status |
| --- | --- | --- | --- |
| rapamycin | tests/run-pass/dissertation_pbpk28_parity_ref_rapamycin.sio | scripts/dissertation/run_pbpk28_node.mjs | gate green |
| semaglutide | ..._parity_ref_semaglutide.sio | run_pbpk28_node.mjs | gate green |
| venlafaxine + ODV | ..._parity_ref_venlafaxine.sio | scripts/dissertation/run_venlafaxine_node.mjs | gate green |
| haloperidol | ..._parity_ref_haloperidol.sio (fixed-step RK4) | runHaloperidolScenario in website/src/lib/pbpk28_core.mjs | gate green |
| midazolam | ..._parity_ref_midazolam.sio | run_pbpk28_node.mjs; DDI in scripts/clinical_midazolam_ddi_e2e_gate.sh | gate green |
| tirzepatide | ..._parity_ref_tirzepatide.sio | run_pbpk28_node.mjs --drug=tirzepatide | gate green |
| everolimus | ..._parity_ref_everolimus.sio | run_pbpk28_node.mjs --drug=everolimus | gate green |
| tacrolimus | ..._parity_ref_tacrolimus.sio | run_pbpk28_node.mjs --drug=tacrolimus | gate green |
| olanzapine | ..._parity_ref_olanzapine.sio | run_pbpk28_node.mjs --drug=olanzapine | gate green |
| clozapine | ..._parity_ref_clozapine.sio | run_pbpk28_node.mjs --drug=clozapine | gate green |
| quetiapine | ..._parity_ref_quetiapine.sio | run_pbpk28_node.mjs --drug=quetiapine | gate green |
| lithium | ..._parity_ref_lithium.sio | run_pbpk28_node.mjs --drug=lithium | gate green |
| vancomycin | ..._parity_ref_vancomycin.sio | run_pbpk28_node.mjs --drug=vancomycin | gate green |
| risperidone + 9-OH | ..._parity_ref_risperidone.sio | scripts/dissertation/run_risperidone_node.mjs | gate green |
| aripiprazole | ..._parity_ref_aripiprazole.sio | scripts/dissertation/run_aripiprazole_node.mjs | gate green (PR 3460 merged) |

All producers are executed by scripts/ci/dissertation_pbpk28_parity_gate.sh
on every CI run; a figure regenerated from these triples reproduces the
gated numbers by construction. Mass residuals are printed by the same runs.

## 2. MC/GUM figures (versioned documents)

- docs/dissertation/results/mc_cross_validation_lognormal_v3.md (latest of
  3; historical GUM/MC comparison, distinct run centers noted in SIO-562).
- docs/dissertation/results/m5_gum_4th_order_v2.md (latest of 2).
- docs/dissertation/results/mc_prior_family_sweep_v3.md (latest of 3).
Status: numbers frozen in-file at their recorded versions; re-execution is
a follow-up, not claimed here.

## 3. Viewer charts

Website tour charts render website/src/lib/pbpk28_core.mjs scenario
functions (runRisperidoneScenario, runAripiprazoleScenario, infusion model,
vfx helpers) against website-pinned parameters. Frozen as the (page, core
file) pair at the measured commit; the 10 new lanes never churn the pinned
domain source (scenario-owned profiles).

## 4. Explicit gaps (not repo-side, not silently dropped)

- Appendix B .tex lives outside the repo. B.2/B.10 legends point at the
  parity producers above, but the figure output files and their generator
  commands are not in the tree (SIO-550 audit). The .tex must cite the
  triples in section 1 by commit.
- SIO-504 (bolus addendum) and SIO-505 audit notes are Notion-side; the
  repo-side counterparts are the gate clamp-0 rows and the t=0.1/0.5 h
  audit points referenced in each dossier section 9.
- ODV/V ratios, Valoteau 1996, Lampen-brain: tracked in SIO-550/SIO-562,
  not duplicated here; any figure depending on them inherits the PENDING
  status.

Versions: main 3d3da08066; gate
scripts/ci/dissertation_pbpk28_parity_gate.sh (1959 lines, 15 producers).
