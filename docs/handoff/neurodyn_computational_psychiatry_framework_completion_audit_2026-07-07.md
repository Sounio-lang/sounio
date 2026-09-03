<!-- docs:meta
topic_id: repo.docs.handoff.neurodyn-computational-psychiatry-framework-completion-audit-2026-07-07
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.handoff.neurodyn-computational-psychiatry-framework-completion-audit-2026-07-07
-->

# NeuroDyn Computational Psychiatry Framework Completion Audit

Date: 2026-07-07

Audited objective: build the first scientific product as "a controlled
framework for testing path-dependent neural dynamics in computational
psychiatry," pause Algebra-C priority, and execute the six immediate steps
listed in the attached objective note.

Overall status for the immediate objective: `FRAMEWORK_EXECUTED_LOW_POWER_PILOT`.

Claim boundary: this audit closes the immediate framework/pilot objective only.
It does not assert a diagnostic, biomarker, treatment-response,
biological-mechanism, clinical-decision, broad O-SSM superiority, or "brain is
octonionic" claim.

## Requirement Audit

| requirement | status | authoritative evidence | notes |
|---|---|---|---|
| 1. Write the psychiatric preregistration | complete | `docs/research/neurodyn_ossm_adhd_dimensional_prereg_2026-07-07.md` | Prereg frames the product as computational-psychiatry instrumentation, not clinical discovery. |
| 2. Choose one primary dataset and one dimensional phenotype family | complete | Prereg plus `scripts/research/adhd200_s3_bootstrap.py` and `/tmp/adhd200_pcp_pilot24_bootstrap/adhd200_s3_bootstrap_summary.json` | Primary dataset is ADHD-200 / PCP / FCP-INDI; primary endpoints are inattention, hyperactivity/impulsivity, and ADHD total/index. |
| 3. Define baselines and controls | complete | Prereg, `scripts/research/adhd200_generic_recurrent_baseline.py`, `scripts/research/adhd200_dimensional_pilot_decision_gate.py`, and pilot summary TSVs | Controls include H-SSM, covariates, static input summaries, GRU-style reservoir, and small trained NumPy RNN. Full GRU/S4/Transformer controls remain promotion-scale follow-up, not required for the immediate smoke objective. |
| 4. Instrument O-SSM dynamic feature export | complete | `examples/brain_ossm_abide.sio` emits `STATE_TRACE`; `scripts/research/neurodyn_adhd_dimensional_state_probe.py` exports `adhd_dimensional_dynamic_features.tsv` | Real pilot emitted 960 `STATE_TRACE` rows and a 961-line dynamic feature TSV. |
| 5. Run a small site-aware pilot | complete | `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/` | Pilot used 24 real ADHD-200 PCP subjects, 2 sites, labels ADHD=12/TD=12, readiness `pass`, and leave-site-aware summaries. |
| 6. Decide whether a clinical story exists | complete | `/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24/pilot_decision/adhd_dimensional_pilot_decision.md`; `docs/handoff/neurodyn_adhd200_pcp_pilot24_2026-07-07.md` | Decision is `UNDERCONTROLLED_LOW_POWER`; no clinical story should be promoted from this pilot. |

## Executed Evidence

Bootstrap:

```bash
python3 scripts/research/adhd200_s3_bootstrap.py \
  --output-dir /tmp/adhd200_pcp_pilot24_bootstrap \
  --max-subjects 24 \
  --overwrite
```

Result: `ready`; selected subjects 24; available C-PAC subjects 162; downloaded
ROI files 24; sites `KKI=16`, `NYU=8`; labels `ADHD=12`, `TD=12`.

Access audit:

```bash
python3 scripts/research/adhd200_data_access_audit.py \
  --phenotypic-csv /tmp/adhd200_pcp_pilot24_bootstrap/adhd200_phenotypic.csv \
  --roi-dir /tmp/adhd200_pcp_pilot24_bootstrap/rois \
  --output-dir /tmp/adhd200_pcp_pilot24_bootstrap/access_audit \
  --overwrite
```

Result: `ready`.

Pilot:

```bash
PHENOTYPIC_CSV=/tmp/adhd200_pcp_pilot24_bootstrap/adhd200_phenotypic.csv \
ROI_DIR=/tmp/adhd200_pcp_pilot24_bootstrap/rois \
OUTPUT_DIR=/tmp/adhd200_pcp_pilot24_bootstrap/pilot_real24 \
NULL_PERMUTATIONS=1 \
GENERIC_BASELINE_MODELS=gru_reservoir,trained_rnn \
GENERIC_BASELINE_SEEDS=55555,11111 \
GENERIC_TRAINED_EPOCHS=4 \
GLOBAL_TRAIN_EPOCHS=2 \
OCT_TRAIN_EPOCHS=2 \
H_TRAIN_EPOCHS=2 \
SOUNIO_SOUC_ENGINE=lean_single \
scripts/research/adhd200_dimensional_pilot_smoke.sh
```

Result: exit code 0.

Key outputs:

- readiness gate: `pass`;
- dynamic feature export: 960 `STATE_TRACE` rows, 961-line TSV;
- trained generic prediction export: 151-line TSV;
- pilot decision: `UNDERCONTROLLED_LOW_POWER`;
- low-power reasons: `row_count 24 < min_decision_subjects 50` and
  `min null_permutations_mean 1 < min_decision_null_permutations 20`.

## Decision

This completes the immediate execution objective, but it does not close a
scientific claim. The answer to "is there a clinical story now?" is no. The
current evidence supports only an underpowered framework smoke: in this small
real ADHD-200 sample, O-SSM hidden trajectories did not show an interpretable
lead over the required controls, but the run is too small and uses too few
null permutations to conclude that controls suffice in general.

The scientifically correct product framing is therefore:

> A controlled framework for testing path-dependent neural dynamics in
> computational psychiatry.

The honest pilot outcome is:

> The first small ADHD-200 PCP pilot is undercontrolled for scientific
> promotion; do not promote a clinical, mechanistic, generic-control,
> or O-SSM superiority claim.

## Residual Non-Blocking Follow-Up

These items are not required to close the immediate objective, but they are
required before any promotion-scale scientific claim:

- scale beyond `n=24` and two sites;
- increase null permutations and seed count;
- run full trained GRU/S4/Transformer baselines;
- run leakage and nuisance-control audits;
- resolve or continue bypassing the Madaros f64 ABI blocker with explicit
  compiler-path disclosure;
- obtain external review on any scaled report or paper draft.
