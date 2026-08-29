<!-- docs:meta
topic_id: repo.docs.research.moonshot-a-orc-implementation-decision
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.moonshot-a-orc-implementation-decision
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

---
docs:meta:
  status: active
  owner: codex
  updated: 2026-05-25
status-note: "Moonshot A implementation decision log; scalar and f32-epistemic Slurm gates reached PASS_FULL."
---

# Moonshot A: Epistemic Non-Associative ORC Implementation Decision

## Decision

Execute Moonshot A in scalar-first, evidence-locked phases:

1. Keep the existing scalar Sinkhorn-16 ORC path as the runtime baseline.
2. Add an epistemic f32 shadow lane for first-order uncertainty propagation.
3. Bind local ABIDE and 168/ZD transport evidence with hash-addressed manifests.
4. Treat Slurm CUDA runtime pass as the only full-completion acceptance gate.

This avoids claiming final non-associative surgery semantics before the local
transport prototype and CUDA worker evidence both pass.

## Ground Truth

The current local evidence package is:

- `scripts/research/abide_epistemic_orc_slice.py`
- `scripts/research/abide_cohort_orc_sweep.py`
- `scripts/research/moonshot_a_abide_f32_cohort_analysis.py`
- `scripts/research/moonshot_a_abide_transport_conditioned_orc.py`
- `scripts/research/moonshot_a_abide_cohort_manifest.py`
- `scripts/research/transport_168_modulation.py`
- `scripts/ci/moonshot_a_abide_epistemic_orc_slice_gate.sh`
- `scripts/ci/moonshot_a_abide_cohort_manifest_gate.sh`
- `scripts/ci/moonshot_a_abide_epistemic_cohort_slurm_gate.sh`
- `scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh`
- `scripts/ci/moonshot_a_abide_claim_discipline_gate.sh`
- `scripts/ci/moonshot_a_abide_transport_conditioned_orc_gate.sh`
- `scripts/ci/moonshot_a_transport_168_modulation_gate.sh`
- `scripts/ci/moonshot_a_transport_168_linearity_gate.sh`
- `scripts/ci/moonshot_a_transport_168_curvature_gate.sh`
- `scripts/ci/moonshot_a_transport_168_manifest_gate.sh`
- `scripts/ci/moonshot_a_phase_status_gate.sh`
- `scripts/ci/moonshot_a_phase_status_verify.py`

The GPU runtime package is intentionally separated:

- `scripts/ci/moonshot_a_sinkhorn16_slurm_gate_common.sh`
- `scripts/ci/kretikos_kaxi_sinkhorn16_slurm_gate.sh`
- `scripts/ci/kretikos_kaxi_sinkhorn16_epistemic_slurm_gate.sh`
- `scripts/ci/moonshot_a_phase_status_adopt_runtime_artifacts.sh`

Those Slurm gates are required for `PASS_FULL`. Local Python gates can only
produce `PASS_LOCAL_GPU_PENDING`. With `SOUNIO_MOONSHOT_A_RUN_SLURM=1`, the
runtime gates submit `sbatch` jobs, build the CUDA Driver API runner on the
worker, launch the current Sinkhorn-16 PTX, and write a JSON artifact consumed
by the phase-status gate. Because the `gpu-orangefs` Slurm node accepts only
8192M per job, the gate pre-emits large Sinkhorn-16 PTX on the login/control
plane and uses the GPU worker for CUDA load/launch/copyback.
If jobs finish after the wait window, the adoption helper rebuilds phase status
from the existing runtime JSONs without submitting more jobs.

Current accepted runtime evidence:

- scalar Slurm job `1749`: `status=pass`, `runtime.reason=launch_pass`,
  `maxdu=4.000000000026205e-05`, `maxdv=4.76837e-07`, PTX SHA-256
  `dd911783f800a727c856120279007a1d035ddd4c52f1a6d393d582499c2abf10`.
- f32-epistemic Slurm job `1785`: `status=pass`,
  `runtime.reason=launch_pass`, `maxdu=4.000000000026205e-05`,
  `maxdv=4.76837e-07`, `maxvar=3.40282e+38`, `inf_count=0`,
  `nan_count=0`, PTX SHA-256
  `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`.
- phase status verifier:
  `/tmp/moonshot-a-phase-status.BIpwLA/moonshot_a_phase_status.v1.json`
  reported `PASS_FULL`.

Current accepted cohort baseline evidence:

- `scripts/ci/moonshot_a_abide_cohort_manifest_gate.sh` reports
  `PASS_COHORT_BASELINE_READY`.
- The baseline consumes the existing scalar ABIDE ORC cohort summary at
  `artifacts/research/abide_orc/cohort_summary.tsv`, SHA-256
  `aa2662a1b456b6b316631ad3a5d59df16bb72ec287a387d744170624ebf18eff`.
- Cohort baseline counts:
  - ABIDE input files: `1035`
  - ORC `.npy` files: `1034`
  - summary subjects: `1034`
  - labelled subjects: `499` (`ASD=249`, `TD=250`)
- Descriptive ASD/TD screen on scalar per-subject ORC features:
  - mean ORC Cohen's d: `0.01124003105667986`
  - std ORC Cohen's d: `0.04027584227299126`
- The phase-status gate now appends the cohort baseline manifest when the
  local cohort TSV is present, without rerunning the full GPU cohort sweep.

Current f32-epistemic cohort-campaign status:

- `scripts/research/abide_cohort_orc_sweep.py` now accepts
  `--mode f32_epistemic`, sends an `--init-var-file` to the CUDA Driver API
  runner, writes per-subject f32-epistemic variance diagnostics, supports
  `--out-dir` so epistemic outputs do not alias the scalar `.npy` cache, and
  supports `--subjects-from-report` so the full campaign follows the accepted
  scalar 1034-subject cohort rather than the raw 1035-file directory containing
  malformed `UM_1_0050284`.
- `scripts/ci/moonshot_a_abide_epistemic_cohort_slurm_gate.sh` stages the
  f32-epistemic Sinkhorn-16 PTX, builds `kaxi_ptx_runner` on the worker, runs
  the ABIDE sweep in Slurm, and emits
  `sounio.moonshot_a.abide_epistemic_cohort_slurm.v1`.
- A pilot submit-only job was accepted as Slurm job `1792` from stage
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-codex-20260525T003230`
  with run dir
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-run-20260525T003436`.
- Job `1792` was initially allocated to `gpuorangefs-r740-proxmox`, then
  cancelled by Slurm with `NODE FAILURE` at `2026-05-25T00:34:38Z` and
  requeued as pending for resources. This was a platform blocker, not runtime
  acceptance or rejection.
- The next attempts exposed and closed campaign-harness blockers:
  - worker-side PTX emission hit `OUT_OF_MEMORY`; the gate now pre-emits PTX
    on the login/control-plane side.
  - worker `python3` lacked `numpy` and `pip`; the stage now carries `./bin/uv`
    and the gate falls back to `uv run --with numpy python`.
  - worker `cc` produced a zero-filled runner placeholder; the gate now copies
    a known-good ELF `kaxi_ptx_runner` from the accepted f32-epistemic runtime
    lane and validates ELF magic before use.
- Slurm job `1797` completed the f32-epistemic ABIDE pilot:
  - `status=pass`
  - subjects reported: `2`
  - elapsed: `00:00:21`
  - node: `gpuorangefs-r770-proxmox`
  - PTX SHA-256:
    `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`
  - variance diagnostics: `inf_count=0`, `nan_count=0`,
    `negative_count=0`, `max_output_var=3.40282e+38`
  - artifact:
    `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-run-20260525T005500/moonshot_a_abide_epistemic_cohort_slurm.v1.json`
- Full-cohort f32-epistemic campaign completed after reusing the pilot-hardened
  gate:
  - run directory:
    `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745`
  - final Slurm job: `1803`
  - status: `pass`
  - subjects reported: `1034`
  - expected subjects from scalar baseline TSV: `1034`
  - ORC `.npy` artifacts: `1034`
  - f32-epistemic diagnostic TSV artifacts: `1034`
  - labelled subjects: `499` (`ASD=249`, `TD=250`)
  - PTX SHA-256:
    `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`
  - variance diagnostics: `inf_count=0`, `nan_count=0`,
    `negative_count=0`, `bad_first_index_count=0`,
    `max_output_var=3.40282e+38`
  - descriptive ASD/TD mean-ORC screen: Cohen's d `+0.0114`
  - artifact:
    `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745/moonshot_a_abide_epistemic_cohort_slurm.v1.json`
- Downstream descriptive alignment analysis is now preserved in
  `docs/research/moonshot-a-abide-f32-cohort-analysis.md` and accepted by
  `scripts/ci/moonshot_a_abide_f32_cohort_analysis_gate.sh`:
  - status: `PASS_COHORT_ANALYSIS_READY`
  - joined subjects: `1034`
  - scalar-only subjects: `0`
  - f32-only subjects: `0`
  - scalar vs f32 mean-ORC correlation: `0.9999976563086985`
  - mean ORC delta absolute max: `0.0020095770759479548`
  - scalar mean-ORC ASD/TD Cohen's d: `0.01124003105667986`
  - f32 mean-ORC ASD/TD Cohen's d: `0.011385455451838606`
  - variance diagnostics remain clean:
    `inf_count=0`, `nan_count=0`, `negative_count=0`,
    `bad_first_index_count=0`
- Transport-conditioned ABIDE x 168/ZD feature package is preserved in
  `docs/research/moonshot-a-abide-transport-conditioned-orc.md` and accepted by
  `scripts/ci/moonshot_a_abide_transport_conditioned_orc_gate.sh`:
  - status: `PASS_TRANSPORT_CONDITIONED_ORC_READY`
  - joined subjects: `1034`
  - selected 168/ZD runtime classes: `8`
  - feature rows: `8272`
  - transport-conditioned delta absolute max: `0.001085675688928034`
  - transport-delta / scalar-f32 mean-absolute-drift ratio:
    `1.0900345407798109`
  - selected class indices: `75`, `76`, `85`, `86`, `141`, `142`, `151`, `152`

## Acceptance Criteria

Local phase acceptance requires:

- ABIDE epistemic ORC slice gate passes on synthetic data.
- If local ABIDE files are present, ABIDE real-slice artifact is emitted.
- 168/ZD modulation evaluates 168 unordered classes.
- Separable modulation remains near gauge-absorbed relative to subspace modulation.
- Linearity and curvature gates enforce their declared thresholds.
- Manifest gate hashes all local transport artifacts.
- Phase-status verifier accepts the resulting status artifact.

Full phase acceptance additionally requires:

- scalar Sinkhorn-16 Slurm runtime artifact has `status=pass`.
- f32-epistemic Sinkhorn-16 Slurm runtime artifact has `status=pass`.
- phase status reports `PASS_FULL`.
- cohort baseline manifest reports `PASS_COHORT_BASELINE_READY` when existing
  scalar ABIDE cohort artifacts are present.
- f32-epistemic ABIDE full-cohort artifact reports `status=pass`,
  `limit=0`, and `subjects_reported == expected_subjects == 1034`.
- f32-epistemic downstream cohort-analysis gate reports
  `PASS_COHORT_ANALYSIS_READY`.
- f32-epistemic ABIDE claim-discipline gate reports
  `PASS_CLAIM_DISCIPLINE`.
- ABIDE transport-conditioned ORC gate reports
  `PASS_TRANSPORT_CONDITIONED_ORC_READY`.

## Boundaries

- The local ABIDE slice is not a full ABIDE cohort revalidation.
- The cohort baseline manifest uses the existing scalar ABIDE ORC cohort
  summary as a baseline; it does not rerun the full cohort.
- The f32-epistemic full-cohort campaign follows the accepted 1034-subject
  scalar baseline cohort and excludes the malformed raw input
  `UM_1_0050284`; it is not a 1035-file raw-directory claim.
- The descriptive scalar-vs-f32 cohort analysis is not a confirmatory biomarker claim
  and does not claim clinical utility, diagnosis, or external validation.
- The current transport modulation is a runtime prototype of 168/ZD-labeled perturbations, not final ZD surgery semantics.
- The transport-conditioned ABIDE feature package uses global class-probe
  deltas from the transport runtime prototype; it is not a subject-specific
  CUDA rerun under every 168/ZD modulation.
- The kappa intervals are first-order sensitivity bands, not calibrated coverage intervals.
- The reconstructed 168 runtime class list is a deterministic prefix of runtime candidate pairs and must not be cited as the Lean census itself.
- Slurm queue/resource/priority failures are platform blockers, not semantic ORC failures.

## Next TODOs

1. Keep the local Moonshot A phase gate green.
2. Preserve scalar and f32-epistemic Slurm runtime artifacts as the accepted baseline.
3. Preserve the scalar ABIDE cohort baseline manifest as the accepted campaign starting point.
4. Preserve the f32-epistemic ABIDE full-cohort artifact as the accepted
   1034-subject cohort evidence.
5. Preserve the descriptive scalar-vs-f32 cohort-analysis artifact and keep
   variance saturation language explicit: exported f32-epistemic variance is a
   finite overflow bound, not a calibrated coverage interval.
6. Keep the ABIDE claim-discipline gate green before lifting this package into
   any external-facing Moonshot A summary.
7. Preserve the ABIDE x 168/ZD transport-conditioned ORC feature package as
   dataset material for the later LoRA lane, with the global-class-probe
   boundary intact.
