<!-- docs:meta
topic_id: repo.docs.governance.data-access-credentials
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.governance.data-access-credentials
-->

# Data Access Credentials — Concrete Issuance Flow

**Owner**: Demetrios Chiuratto Agourakis
**Status**: Living document (updated as credentials are issued)
**Last updated**: 2026-05-01

## Scope

This document captures the **concrete steps** required to obtain
credentials for each real-world data source the Sounio M4
pipeline (`scripts/clinical/`) supports. It is the operational
companion to:

  - `scripts/clinical/etl/mimic_iv_vancomycin.sql` (the SQL
    extraction)
  - `scripts/clinical/etl/extract_mimic_iv.sh` (the driver)
  - `scripts/clinical/validate_cohort.py` (the validator)
  - `scripts/clinical/smoke_pipeline.sh` (the integration smoke)

When a credential issuance is complete (or stalled), this
document records the date, owning party, and current blocking
step.

## Data sources currently supported

### 1. MIMIC-IV (PhysioNet, MIT)

The primary public ICU dataset. Suitable for vancomycin TDM
cohort extraction with the schema produced by
`scripts/clinical/etl/mimic_iv_vancomycin.sql`.

**Data scope**: 200K+ ICU stays from BIDMC (Boston) 2008–2019,
with deidentified labs, vitals, medications, demographics, and
30-day mortality.

**License**: PhysioNet Credentialed Health Data Use Agreement
(PhysioNet DUA). Free for research; requires named-user
credentialing.

**Issuance flow** (MIT/PhysioNet, ~10-15 days):

  1. **Create PhysioNet account** at <https://physionet.org/>
     - Free; requires an institutional email (verified by
       reply).
  2. **Complete CITI Data or Specimens Only Research course**
     at <https://physionet.org/about/citi-course/>
     - 1 of: "Conflicts of Interest" + "Data or Specimens
       Only Research" tracks.
     - Completion certificate uploaded to PhysioNet profile.
     - Allow 4-8 hours of study + quiz.
  3. **Sign the PhysioNet Credentialed Health Data DUA**
     - Click-through agreement on the PhysioNet site.
     - Bind the credentialed account to a specific institution
       and supervisor (free-text; verified by email).
  4. **Wait for approval** — typically 7-14 days; PhysioNet
     manually reviews each application.
  5. **Request access to MIMIC-IV** specifically
     (<https://physionet.org/content/mimiciv/3.1/>)
     - Click-through DUA acknowledgement, applies retroactively
       to all MIMIC-IV versions.

**Per-project requirements**:

  - **No IRB needed for de-identified MIMIC-IV** when used
    under the PhysioNet DUA (per HHS 45 CFR 46.102(e)(5),
    de-identified data per HIPAA Safe Harbor is non-human-
    subject research).
  - **Local data-use logging** is recommended (institutional
    requirement varies); for São Paulo / São Carlos, file a
    "non-human-subject research declaration" with the
    institutional Comitê de Ética em Pesquisa (CEP).

**Access modes**:

  - **Google Cloud BigQuery** (recommended): MIMIC-IV is
    pre-loaded at `physionet-data.mimiciv_3_1_*`. After DUA
    approval, request BigQuery access via
    <https://physionet.org/about/cloud/> (free quota of 1 TB
    queries/month with a personal GCP project).
  - **Direct download**: ~80 GB CSV/Parquet bundle from
    PhysioNet; load into local PostgreSQL via the
    [mimic-code](https://github.com/MIT-LCP/mimic-code/tree/main/mimic-iv)
    schema scripts.

**Status (2026-05-01)**: NOT YET INITIATED. Action item:
submit PhysioNet credential application as soon as the M4
pipeline goes formal-stable.

**Owner action item**: Send completed CITI certificate to
PhysioNet within 30 days of M4 freeze.

### 2. eICU Collaborative Research Database (PhysioNet, MIT/Philips)

Multi-center US ICU dataset (200+ hospitals). Useful for
**external validation** of MIMIC-IV-trained models.

**Issuance flow**: identical to MIMIC-IV (same DUA, same CITI
course satisfies both).

**Status**: identical to MIMIC-IV.

### 3. AmsterdamUMCdb (Amsterdam UMC, Netherlands)

European ICU dataset, distinct case mix from MIMIC-IV. Good
for **out-of-distribution cohort generalisation** experiments.

**Issuance flow**:

  1. Sign the AmsterdamUMCdb DUA at
     <https://amsterdammedicaldatascience.nl/amsterdamumcdb/>
  2. Submit a research proposal (1-2 pages); reviewed by the
     Amsterdam UMC data committee (~30 days).
  3. Sign individual data-use commitments per project.

**Status**: NOT YET INITIATED. M5 milestone if MIMIC-IV
results warrant external validation.

### 4. Institutional cohort (Mackenzie / São Paulo)

Local TDM cohort (vancomycin, gentamicin, amikacin) at
São Paulo / São Carlos institutional hospitals.

**Issuance flow**:

  1. **CEP approval** at the institutional Comitê de Ética em
     Pesquisa (Mackenzie / São Carlos, depending on cohort).
     - Typical timeline: 60-120 days.
     - Required documents: research protocol (~10 pages),
       PI CV, biosafety statement, data-handling plan.
  2. **CONEP (national)** approval if multi-center.
  3. **Termo de Consentimento Livre e Esclarecido (TCLE)**
     waiver request (since cohort is retrospective and
     de-identified).
  4. **Hospital data-export agreement** with IT department —
     define the export schema (matching
     `tdm_cohort_synthetic_v2.csv`).

**Status**: NOT YET INITIATED. Track-by-track:

  - [ ] Identify primary institutional partner (Mackenzie or
        São Carlos)
  - [ ] Draft CEP protocol (mirror MIMIC-IV ETL columns)
  - [ ] PI CV preparation (Demetrios Chiuratto Agourakis)
  - [ ] Biosafety statement (no patient contact, retrospective
        only)
  - [ ] Data-handling plan (deidentification, encrypted
        storage, no cross-institutional transfer except by
        approved API)

**Owner action item**: Pre-CEP draft once MIMIC-IV pilot
cohort generates a primary-outcome figure worthy of CEP
submission.

## Credentialing checkpoint (before any real-data ingest)

Before running `scripts/clinical/etl/extract_mimic_iv.sh`
against real PhysioNet data, the following must be true:

  1. PhysioNet account credentialed (DUA signed, CITI
     certificate on file, status = "Credentialed").
  2. MIMIC-IV-specific click-through complete.
  3. (For BigQuery mode) GCP project with billing enabled,
     `bq` CLI authenticated to the credentialed PhysioNet
     account.
  4. (For PostgreSQL mode) Local mimic-code schema loaded;
     `psql` connection verified.
  5. **Local data-export plan** in place (no patient data
     leaves the local secure storage; all analyses run in-
     place).

## Auditing

Every real-data run is logged at:

```
logs/clinical/runs/YYYYMMDD-HHMMSS/
    cohort.csv
    validate_cohort.log
    smoke_pipeline.log
    predictions.csv
    summary.json
```

The `summary.json` records:

  - cohort source (MIMIC-IV version, eICU, institutional)
  - SHA-256 hash of `cohort.csv`
  - validator outcome
  - smoke pipeline outcome
  - prediction-count and aggregate metrics
  - agent ID + git SHA at run time

This audit trail is **mandatory** before any clinical claim
based on the run can be made. The institutional CEP / CONEP /
PhysioNet DUA all require auditable provenance.

## Failure modes to plan for

  - **Credentialing slow** (PhysioNet approval > 14 days):
    fall back to extended synthetic-cohort experiments
    (current pathway, 200 patients) while waiting.
  - **CEP rejection**: typically due to insufficient
    deidentification plan; revise data-handling section and
    resubmit.
  - **BigQuery quota exceeded**: switch to PostgreSQL local
    mode (more setup, no quota cap).
  - **Schema drift between MIMIC-IV versions**: pin the SQL
    against a specific MIMIC-IV version (currently 3.1) and
    update only after parallel validation.

## Update protocol for this document

When a credential is issued, append to the relevant section:

```
**Issued**: 2026-MM-DD by <issuing party>
**Account**: <username/email or institutional ID>
**Expires**: <date or "no expiry">
**Validation**: pass <commit-SHA> running smoke_pipeline.sh
on the first data-ingest.
```

When a credential lapses, append:

```
**Lapsed**: 2026-MM-DD due to <reason>; renewal initiated
on <date>.
```
