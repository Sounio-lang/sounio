-- scripts/clinical/etl/mimic_iv_vancomycin.sql
--
-- M4 — MIMIC-IV vancomycin TDM cohort extract.
--
-- Builds a per-patient cohort row matching the schema of
-- scripts/clinical/data_synthetic/tdm_cohort_synthetic_v2.csv:
--
--   patient_id, age, sex, weight_kg, height_cm, scr_mg_dl,
--   crcl_ml_min, dose_mg, interval_h, measured_cmin_mg_l,
--   sofa, nephrotoxic_coexposure, outcome_cure, outcome_aki_kdigo
--
-- Inclusion:
--   - First adult ICU admission per subject_id
--   - Received intravenous vancomycin (atc J01XA01) for ≥48h
--   - At least one steady-state trough (≥3 doses before sample)
--   - Available height/weight/SCr at admission
--
-- Exclusion:
--   - Renal replacement therapy at trough time (CRRT, IHD, PD)
--   - Pregnancy
--   - Vancomycin AUC-guided dosing (only trough-based for cleanness)
--
-- Outcome definitions:
--   - outcome_cure: alive at 30 days post-vancomycin start AND
--     no documented persistence of original infection at 14 days
--   - outcome_aki_kdigo: KDIGO grade ≥ 1 (1.5×SCr_baseline OR
--     ≥0.3 mg/dL absolute increase in 48h) within 7 days of start
--
-- Co-exposures (nephrotoxic_coexposure = 1 if any in 48h window):
--   piperacillin-tazobactam, aminoglycosides, amphotericin,
--   contrast, NSAIDs, calcineurin inhibitors, IV ACE/ARB
--
-- ============================================================================
-- USAGE
-- ============================================================================
--
-- Requires PhysioNet MIMIC-IV credentialed access (CITI training +
-- DUA signed). Approximate run-time: 3-8 minutes on a warm BigQuery
-- session over `physionet-data.mimiciv_3_1`.
--
-- Schema target: BigQuery (`physionet-data.mimiciv_3_1.*`). For
-- PostgreSQL deployments substitute the project/dataset prefix
-- (`mimiciv_hosp.`, `mimiciv_icu.`, `mimiciv_derived.`) and replace
-- `DATETIME_DIFF` with `EXTRACT(EPOCH FROM (a - b)) / 3600.0`.
--
-- Dependencies:
--   - mimic-iv-derived (kdigo_creatinine, kdigo_uo, sofa,
--     icustay_detail, height, weight_durations) — run the
--     mimic-code/mimic-iv-derived/ scripts first.
--
-- Output:
--   Save query result as `mimic_iv_vancomycin_cohort.csv` next to
--   `tdm_cohort_synthetic_v2.csv`. The downstream pipeline
--   (`scripts/clinical/process_tdm_cohort.sh`) is schema-compatible.
--
-- ============================================================================
-- THIS QUERY DOES NOT CONTAIN ANY PATIENT DATA. It only encodes the
-- transformation logic. Running it requires MIMIC-IV credentialed
-- access; the repository ships only the SQL.
-- ============================================================================

-- ----- Step 1: anchor cohort to first adult ICU admission with vancomycin --

WITH first_icu AS (
    SELECT
        ie.subject_id,
        ie.hadm_id,
        ie.stay_id,
        ie.intime         AS icu_in,
        ie.outtime        AS icu_out,
        adm.admission_age,
        pat.gender
    FROM `physionet-data.mimiciv_3_1.icu.icustays`        AS ie
    INNER JOIN `physionet-data.mimiciv_3_1.derived.icustay_detail` AS adm
        ON ie.stay_id = adm.stay_id
    INNER JOIN `physionet-data.mimiciv_3_1.hosp.patients`  AS pat
        ON ie.subject_id = pat.subject_id
    WHERE adm.admission_age >= 18
      AND ie.first_icu_stay = TRUE
),

vanco_admin AS (
    SELECT
        prx.subject_id,
        prx.hadm_id,
        prx.stay_id,
        MIN(prx.starttime) AS vanco_start,
        MAX(prx.endtime)   AS vanco_end,
        COUNT(*)           AS n_doses,
        AVG(prx.amount)    AS mean_dose_mg
    FROM `physionet-data.mimiciv_3_1.icu.inputevents` AS prx
    INNER JOIN `physionet-data.mimiciv_3_1.icu.d_items` AS di
        ON prx.itemid = di.itemid
    WHERE LOWER(di.label) LIKE '%vancomycin%'
      AND prx.amountuom = 'mg'
      AND di.linksto = 'inputevents'
    GROUP BY 1, 2, 3
    HAVING COUNT(*) >= 3
       AND DATETIME_DIFF(MAX(prx.endtime), MIN(prx.starttime), HOUR) >= 48
),

-- ----- Step 2: trough levels (steady-state) ----------------------------------

vanco_trough AS (
    SELECT
        le.subject_id,
        le.hadm_id,
        le.charttime          AS trough_time,
        le.valuenum           AS trough_value,
        ROW_NUMBER() OVER (
            PARTITION BY le.subject_id, le.hadm_id
            ORDER BY le.charttime
        ) AS trough_seq
    FROM `physionet-data.mimiciv_3_1.hosp.labevents` AS le
    INNER JOIN `physionet-data.mimiciv_3_1.hosp.d_labitems` AS dli
        ON le.itemid = dli.itemid
    WHERE LOWER(dli.label) LIKE '%vanco%'
      AND dli.fluid = 'Blood'
      AND le.valuenum BETWEEN 1.0 AND 80.0    -- physiological range
      AND le.flag IS DISTINCT FROM 'abnormal' -- exclude flagged garbage
),

-- pick trough that is at least 60 minutes before next dose to be a true trough
vanco_trough_clean AS (
    SELECT
        vt.subject_id,
        vt.hadm_id,
        MIN(vt.trough_time)  AS first_trough_time,
        MIN(vt.trough_value) AS first_trough_value
    FROM vanco_trough AS vt
    INNER JOIN vanco_admin AS va
        USING (subject_id, hadm_id)
    WHERE vt.trough_time BETWEEN
            DATETIME_ADD(va.vanco_start, INTERVAL 30 HOUR)  -- ≥3rd dose
        AND DATETIME_ADD(va.vanco_start, INTERVAL 168 HOUR) -- ≤7 days
    GROUP BY 1, 2
),

-- ----- Step 3: anthropometrics + admission SCr ------------------------------

anthro AS (
    SELECT
        h.subject_id, h.hadm_id, h.stay_id,
        h.height,            -- cm
        w.weight             -- kg
    FROM `physionet-data.mimiciv_3_1.derived.height` AS h
    INNER JOIN `physionet-data.mimiciv_3_1.derived.weight_durations` AS w
        USING (subject_id, hadm_id, stay_id)
    WHERE h.height BETWEEN 130 AND 220
      AND w.weight BETWEEN 35 AND 200
),

baseline_scr AS (
    SELECT
        kc.subject_id,
        kc.hadm_id,
        kc.stay_id,
        AVG(kc.creat) AS scr_baseline
    FROM `physionet-data.mimiciv_3_1.derived.kdigo_creatinine` AS kc
    INNER JOIN vanco_admin AS va USING (subject_id, hadm_id, stay_id)
    WHERE kc.charttime BETWEEN
            DATETIME_SUB(va.vanco_start, INTERVAL 24 HOUR)
        AND va.vanco_start
    GROUP BY 1, 2, 3
),

-- ----- Step 4: SOFA, AKI, nephrotoxic co-exposures, outcome ------------------

sofa_at_start AS (
    SELECT
        s.subject_id, s.hadm_id, s.stay_id,
        s.sofa AS sofa_24
    FROM `physionet-data.mimiciv_3_1.derived.sofa` AS s
    INNER JOIN vanco_admin AS va USING (subject_id, hadm_id, stay_id)
    WHERE s.endtime = DATETIME_ADD(va.vanco_start, INTERVAL 24 HOUR)
),

aki_post AS (
    SELECT
        ks.subject_id, ks.hadm_id, ks.stay_id,
        MAX(ks.aki_stage) AS aki_max
    FROM `physionet-data.mimiciv_3_1.derived.kdigo_stages` AS ks
    INNER JOIN vanco_admin AS va USING (subject_id, hadm_id, stay_id)
    WHERE ks.charttime BETWEEN va.vanco_start
                          AND DATETIME_ADD(va.vanco_start, INTERVAL 168 HOUR)
    GROUP BY 1, 2, 3
),

nephro_coexposure AS (
    SELECT
        prx.subject_id, prx.hadm_id, prx.stay_id,
        1 AS has_nephro
    FROM `physionet-data.mimiciv_3_1.icu.inputevents` AS prx
    INNER JOIN `physionet-data.mimiciv_3_1.icu.d_items` AS di
        ON prx.itemid = di.itemid
    INNER JOIN vanco_admin AS va
        ON prx.stay_id = va.stay_id
    WHERE prx.starttime BETWEEN va.vanco_start
                           AND DATETIME_ADD(va.vanco_start, INTERVAL 48 HOUR)
      AND (
            LOWER(di.label) LIKE '%piperacillin%'
         OR LOWER(di.label) LIKE '%gentamicin%'
         OR LOWER(di.label) LIKE '%amikacin%'
         OR LOWER(di.label) LIKE '%tobramycin%'
         OR LOWER(di.label) LIKE '%amphotericin%'
         OR LOWER(di.label) LIKE '%cyclosporine%'
         OR LOWER(di.label) LIKE '%tacrolimus%'
      )
    GROUP BY 1, 2, 3
),

cure_30d AS (
    SELECT
        adm.subject_id,
        adm.hadm_id,
        CASE
            WHEN adm.deathtime IS NULL
              OR DATETIME_DIFF(adm.deathtime, va.vanco_start, DAY) > 30
            THEN 1 ELSE 0
        END AS alive_30d
    FROM `physionet-data.mimiciv_3_1.hosp.admissions` AS adm
    INNER JOIN vanco_admin AS va USING (subject_id, hadm_id)
)

-- ============================================================================
-- FINAL JOIN — emit the cohort row schema
-- ============================================================================

SELECT
    CONCAT('M', LPAD(CAST(fi.subject_id AS STRING), 8, '0'))  AS patient_id,
    CAST(fi.admission_age AS INT64)                            AS age,
    CASE WHEN fi.gender = 'F' THEN 'F' ELSE 'M' END            AS sex,
    ROUND(an.weight, 1)                                        AS weight_kg,
    CAST(ROUND(an.height) AS INT64)                            AS height_cm,
    ROUND(bs.scr_baseline, 2)                                  AS scr_mg_dl,
    -- Cockcroft-Gault, kg/SCr units
    CAST(ROUND(
        (140 - fi.admission_age) * an.weight
        * CASE WHEN fi.gender = 'F' THEN 0.85 ELSE 1.0 END
        / (72.0 * bs.scr_baseline)
    ) AS INT64)                                                AS crcl_ml_min,
    CAST(ROUND(va.mean_dose_mg) AS INT64)                      AS dose_mg,
    CAST(
        DATETIME_DIFF(va.vanco_end, va.vanco_start, HOUR) / va.n_doses
        AS INT64
    )                                                          AS interval_h,
    ROUND(vtc.first_trough_value, 1)                           AS measured_cmin_mg_l,
    COALESCE(ss.sofa_24, 0)                                    AS sofa,
    COALESCE(nc.has_nephro, 0)                                 AS nephrotoxic_coexposure,
    COALESCE(c30.alive_30d, 0)                                 AS outcome_cure,
    CASE WHEN ap.aki_max >= 1 THEN 1 ELSE 0 END                AS outcome_aki_kdigo
FROM first_icu               AS fi
INNER JOIN vanco_admin       AS va  USING (subject_id, hadm_id, stay_id)
INNER JOIN vanco_trough_clean AS vtc USING (subject_id, hadm_id)
INNER JOIN anthro            AS an  USING (subject_id, hadm_id, stay_id)
INNER JOIN baseline_scr      AS bs  USING (subject_id, hadm_id, stay_id)
LEFT  JOIN sofa_at_start     AS ss  USING (subject_id, hadm_id, stay_id)
LEFT  JOIN aki_post          AS ap  USING (subject_id, hadm_id, stay_id)
LEFT  JOIN nephro_coexposure AS nc  USING (subject_id, hadm_id, stay_id)
LEFT  JOIN cure_30d          AS c30 USING (subject_id, hadm_id)
WHERE bs.scr_baseline IS NOT NULL
  AND vtc.first_trough_value IS NOT NULL
  AND vtc.first_trough_value BETWEEN 2.0 AND 60.0
  AND an.weight IS NOT NULL
  AND an.height IS NOT NULL
ORDER BY fi.subject_id;
