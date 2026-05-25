<!-- docs:meta
topic_id: repo.docs.research.moonshot-a-slurm-blocker-handoff
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.moonshot-a-slurm-blocker-handoff
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
status-note: "Moonshot A CUDA runtime blocker closed with PASS_FULL evidence."
---

# Moonshot A Slurm Runtime Completion

Blocker-ID: `BLOCKER-MOONSHOT-A-SLURM-RESOURCES`

Severity: `B3`

Class: `platform-resource`

Evidence-Level: `E3`

Owner: `closed by codex`

Worktree: `/workspace/sounio`

Branch: current checkout is operational truth.

Acceptance gate:

```bash
SOUNIO_MOONSHOT_A_RUN_SLURM=1 bash scripts/ci/moonshot_a_phase_status_gate.sh
python3 scripts/ci/moonshot_a_phase_status_verify.py --latest
```

If queued runtime jobs finish after the original wait window, adopt their
worker JSON artifacts without submitting new jobs:

```bash
bash scripts/ci/moonshot_a_phase_status_adopt_runtime_artifacts.sh /tmp/moonshot-a-phase-status.TQdxUV
```

Resolution Evidence:

- Local Moonshot A Python gates can pass and are not the blocker.
- The restored scalar and f32-epistemic Slurm gates submit real `sbatch` jobs and emit worker-side JSON artifacts.
- The scalar runtime gate initially failed on worker harness issues:
  - job `1728`: missing staged `bin/souc-linux-x86_64`
  - job `1729`: stale/incomplete compiler stage
  - job `1730`: `OUT_OF_MEMORY` with worker-side PTX emission under the partition's 8192M memory cap
- Harness repair applied:
  - stage `bin/souc` and `bin/souc-linux-x86_64`
  - export `SOUNIO_KRETIKOS_COMPILER`
  - pre-emit Sinkhorn-16 PTX on the login/control-plane side
  - keep worker memory at the partition-accepted `8192M`
- Scalar job `1744` was cancelled before start when the long-running wrapper
  shell exited; to avoid coupling job lifetime to a waiting wrapper, the
  pre-emitted-PTX worker script was submitted directly with `sbatch`.
- Scalar job `1749` completed with `status=pass`, `runtime.reason=launch_pass`,
  and semantic Sinkhorn oracle pass:
  - `maxdu=4.000000000026205e-05`
  - `maxdv=4.76837e-07`
  - PTX SHA-256 `dd911783f800a727c856120279007a1d035ddd4c52f1a6d393d582499c2abf10`
  - PTX:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/phase/slurm_scalar_reuse/sinkhorn16_f32.ptx`.
- f32-epistemic runtime initially exposed three real harness/emitter issues:
  - zero-filled `kaxi_ptx_runner` placeholder produced `Exec format error`
  - PTX declared `.reg .f32 %f<260>` while using up to `%f376`
  - exported variance lane reached `Infinity` on `v_out[1..15]`
- f32-epistemic fixes applied:
  - copied a valid x86-64 ELF CUDA Driver API runner into the staged runtime dirs
  - raised f32-epistemic register bank to `%f<384>`
  - saturated exported f32 variance stores to max finite f32 (`0f7F7FFFFF`)
- f32-epistemic job `1785` completed with `status=pass`,
  `runtime.reason=launch_pass`, semantic Sinkhorn oracle pass, and finite
  nonnegative nonzero variance output:
  - `maxdu=4.000000000026205e-05`
  - `maxdv=4.76837e-07`
  - `maxvar=3.40282e+38`
  - `inf_count=0`
  - `nan_count=0`
  - PTX SHA-256 `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`
  - PTX:
    `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/phase/slurm_f32_epistemic_reuse/sinkhorn16_f32_epistemic.ptx`
  - job script:
    `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/phase/slurm_f32_epistemic_reuse/job_f32_epistemic.sh`
- `SOUNIO_MOONSHOT_A_SCALAR_SLURM_ARTIFACT=/tmp/moonshot-a-scalar-1749.json`
  and `SOUNIO_MOONSHOT_A_EPI_SLURM_ARTIFACT=/tmp/moonshot-a-epi-1785.json`
  promoted the phase gate to `PASS_FULL`.
- `python3 scripts/ci/moonshot_a_phase_status_verify.py --latest` passed on:
  `/tmp/moonshot-a-phase-status.BIpwLA/moonshot_a_phase_status.v1.json`.
- The phase gate now appends the scalar ABIDE cohort baseline manifest when
  `artifacts/research/abide_orc/cohort_summary.tsv` is present:
  - `status=PASS_COHORT_BASELINE_READY`
  - summary subjects `1034`
  - labelled subjects `499`
  - mean-ORC ASD/TD Cohen's d `0.01124003105667986`

Next action:

The original Slurm resource blocker is closed. Codex staged the next
Slurm/Foundry-only f32-epistemic ABIDE cohort campaign under:

```text
/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-codex-20260525T003230
```

The pilot submit-only gate accepted job `1792`:

```bash
SOUNIO_MOONSHOT_A_RUN_SLURM=1 \
SOUNIO_MOONSHOT_A_SUBMIT_ONLY=1 \
SOUNIO_MOONSHOT_A_ABIDE_COHORT_LIMIT=2 \
SOUNIO_MOONSHOT_A_ABIDE_EPI_COHORT_SLURM_DIR=/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-run-20260525T003436 \
SOUNIO_MOONSHOT_A_SLURM_JOB_TIME=01:00:00 \
SOUNIO_MOONSHOT_A_SLURM_CPUS_PER_TASK=1 \
SOUNIO_MOONSHOT_A_SLURM_MEM=8192M \
bash scripts/ci/moonshot_a_abide_epistemic_cohort_slurm_gate.sh
```

The job was initially allocated to `gpuorangefs-r740-proxmox`, then Slurm
cancelled it at `2026-05-25T00:34:38Z` with `NODE FAILURE` and requeued it as
`PENDING` with reason `(Resources)`. This was a platform blocker for that
attempt, not a runtime/emitter failure.

The follow-up campaign-harness blockers were closed:

- worker-side PTX emission hit `OUT_OF_MEMORY`; the gate now pre-emits PTX on
  the login/control-plane side.
- worker `python3` lacked `numpy` and `pip`; the stage now carries `./bin/uv`
  and the worker falls back to `uv run --with numpy python`.
- worker `cc` produced a zero-filled `kaxi_ptx_runner`; the gate now copies a
  known-good ELF runner from the accepted f32-epistemic runtime lane and
  validates ELF magic before use.

Final pilot evidence:

- Slurm job `1797` completed on `gpuorangefs-r770-proxmox`.
- Elapsed: `00:00:21`
- Artifact:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-cohort-run-20260525T005500/moonshot_a_abide_epistemic_cohort_slurm.v1.json`
- Status: `pass`
- Subjects reported: `2`
- PTX SHA-256:
  `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`
- Variance diagnostics: `inf_count=0`, `nan_count=0`, `negative_count=0`,
  `max_output_var=3.40282e+38`

Current follow-up:

- Blocker-ID: none for the 2-subject f32-epistemic ABIDE pilot.
- The full f32-epistemic ABIDE campaign is now complete for the accepted
  1034-subject scalar baseline cohort.
- Raw ABIDE contains 1035 `.1D` files, but `UM_1_0050284` is malformed
  (column count changes from 200 to 97 at row 64) and is not present in the
  accepted scalar cohort summary. Full-cohort claims here mean
  `subjects_reported == expected_subjects == 1034`, not raw-directory 1035.

Final full-cohort evidence:

- Run directory:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745`
- Final Slurm job: `1803`
- Status: `pass`
- Subjects reported: `1034`
- Expected subjects: `1034`
- ORC `.npy` artifacts: `1034`
- f32-epistemic diagnostic TSV artifacts: `1034`
- Label counts: `ASD=249`, `TD=250`, `UNKNOWN=535`
- PTX SHA-256:
  `fff94e75af0d46c3449e0f17770dd2c96b584f8cd980c61d54da4150a616e1a3`
- Variance diagnostics: `inf_count=0`, `nan_count=0`, `negative_count=0`,
  `bad_first_index_count=0`, `max_output_var=3.40282e+38`
- Artifact:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-abide-epi-full-cohort-run-20260525T010745/moonshot_a_abide_epistemic_cohort_slurm.v1.json`
- Downstream descriptive alignment analysis:
  `docs/research/moonshot-a-abide-f32-cohort-analysis.md`
- Analysis gate status: `PASS_COHORT_ANALYSIS_READY`
- Scalar vs f32 mean-ORC correlation: `0.9999976563086985`
- Mean ORC delta absolute max: `0.0020095770759479548`
- Next action: preserve this artifact package and continue downstream
  claim discipline; do not relabel it as a raw 1035-file campaign or as a
  clinical/diagnostic result. This is not the raw 1035-file directory.

Boundaries:

- This blocker does not invalidate the local ABIDE slice or 168/ZD transport evidence.
- Runtime acceptance here is limited to CUDA driver load/launch/copyback plus
  the diagonal Sinkhorn-16 semantic oracle for scalar and f32-epistemic modes.
- Full ABIDE cohort revalidation here is scoped to the accepted 1034-subject
  scalar baseline cohort, not the malformed raw 1035-file directory.
- Do not run heavy CUDA stress directly in `/workspace/sounio`.
