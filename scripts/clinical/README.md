# scripts/clinical/

M4 milestone — vancomycin TDM cohort processing.

## Files

- `process_tdm_cohort.sh` — driver shell script. Reads a CSV cohort
  and invokes the Sounio Knightian pipeline per patient.
- `data_synthetic/`
  - `tdm_cohort_synthetic_v1.csv` — original 20-patient hand-crafted
    skeleton. Retained for backwards compatibility with the early M4
    pipeline smoke; **do not use for any new analysis**.
  - `tdm_cohort_synthetic_v2.csv` — 200-patient popPK-driven synthetic
    cohort (Roberts 2011 ICU vancomycin parameters; deterministic
    seed). Replaces v1 for downstream development.
  - `generate_realistic_cohort.py` — generator for v2 (and beyond).
- `etl/`
  - `mimic_iv_vancomycin.sql` — credential-gated MIMIC-IV extract
    SQL. Outputs the same schema as `tdm_cohort_synthetic_v2.csv`.
  - `extract_mimic_iv.sh` — driver wrapping the SQL for BigQuery
    or PostgreSQL deployment modes.
- `runs/<timestamp>/` — output of each pipeline invocation.

## Status

**Skeleton (M4 stage).** The pipeline structure is in place:

1. **Synthetic v2 generator (✅ landed 2026-05-01)** — popPK-based
   sampling produces 200-patient cohorts with realistic age/weight/
   CrCl/Cmin/SOFA/AKI/cure distributions. Deterministic seed
   `20260501` for CI reproducibility. Replace v1 in all new work.
2. **MIMIC-IV ETL (✅ skeleton landed 2026-05-01)** — SQL is
   schema-complete and credential-gated. Awaiting institutional
   credentialing flow (CITI training + DUA + service account
   provisioning) before first execution against real MIMIC-IV.
3. **JSON-emit mode** in `stdlib/epistemic/knightian.sio` to
   replace concatenated-stdout output with parseable per-patient
   records. Tracked as M4 follow-up.
4. **MAE / coverage analyzer** (`compute_mae.py` or Sounio-native
   equivalent) over predicted vs measured Cmin. Cannot run
   meaningfully on synthetic data (no real ground truth);
   deferred to real cohort.
5. **SOTA comparator** — wrap a NONMEM/pmetrics call from this
   script for paired-test analysis. Requires institutional
   licence; deferred to M5 with full cohort.
6. **Lean theorem closure** — the predictions emitted here will
   feed `formal/lean4/SounioVancomycinDosingSafety.lean`'s
   `cmin_within_implies_efficacy_and_safety` theorem at the
   instance level (one Lean obligation per patient-timepoint).

## Quick start (synthetic)

Generate a fresh 200-patient cohort and run the pipeline:

```bash
# Generate (deterministic seed)
python3 scripts/clinical/data_synthetic/generate_realistic_cohort.py \
    --n 200 --seed 20260501

# Run the pipeline (writes to scripts/clinical/runs/<ts>/)
bash scripts/clinical/process_tdm_cohort.sh \
    scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv
```

Expected output (from the generator's stdout):

```
  Cmin mean        ≈ 17 mg/L
  CrCl mean        ≈ 87 mL/min
  Subtherapeutic   ≈ 29%
  Therapeutic      ≈ 42%
  Toxic            ≈ 29%
  AKI (KDIGO ≥ 1)  ≈ 26%
  Clinical cure    ≈ 72%
```

These distributions are **synthetic but consistent with published
ICU vancomycin TDM cohorts** (Lodise 2009, Rybak 2020 consensus).
Use for pipeline validation only.

## MIMIC-IV credential flow

Real-data extraction requires PhysioNet credentialed access. The
process (typical 2-4 weeks):

1. **CITI training** — complete the "Data or Specimens Only
   Research" course at https://physionet.org/about/citi-course/.
   Provide certificate to PhysioNet.
2. **PhysioNet account** — register at https://physionet.org and
   submit the credentialing request with the CITI certificate.
3. **Sign DUA** — sign the MIMIC-IV Data Use Agreement.
4. **GCP project + service account** — request a GCP project
   (`sounio-research` is provisioned) and a service-account key
   with read-only access to `physionet-data.mimiciv_3_1`. Save
   the key to a path indicated by `$SOUNIO_MIMIC_KEY`.
5. **Activate the credentials**:
   ```bash
   gcloud auth activate-service-account \
       --key-file=$SOUNIO_MIMIC_KEY
   ```
6. **Run the extraction**:
   ```bash
   bash scripts/clinical/etl/extract_mimic_iv.sh
   ```

For PostgreSQL deployments (locally-hosted MIMIC-IV after running
the `mimic-code/mimic-iv-derived/` scripts):

```bash
export MIMIC_PG_URI=postgresql://mimic@localhost/mimiciv_3_1
bash scripts/clinical/etl/extract_mimic_iv.sh --mode psql
```

The output CSV
(`scripts/clinical/data_synthetic/mimic_iv_vancomycin_cohort.csv`)
shares the schema of the synthetic v2 file, so downstream tooling
needs no changes.

## Synthetic data caveat

`tdm_cohort_synthetic_v1.csv` and `tdm_cohort_synthetic_v2.csv`
are **synthetic data**. The v2 generator uses published popPK
distributions and outcome models, which produces realistic
*marginal* distributions, but the *joint* distributions (e.g.,
Cmin × cure correlation) are model-driven, not observational.

Do not draw inferential conclusions from analyses on these files.
They exist to exercise the pipeline plumbing during IRB / MIMIC-IV
lead-time. The first scientifically-valid run is with real data
from one of:
- the institutional ICU TDM cohort (pending IRB approval), or
- MIMIC-IV (pending PhysioNet credentialing).

## File-level testing

The synthetic generator is deterministic given a seed:

```bash
python3 scripts/clinical/data_synthetic/generate_realistic_cohort.py \
    --n 5 --seed 1 --output /tmp/cohort_test.csv
md5sum /tmp/cohort_test.csv  # reproducibility check
```

Re-running with the same seed must produce a byte-identical CSV.
This is exercised in CI as a smoke for the popPK pipeline.
