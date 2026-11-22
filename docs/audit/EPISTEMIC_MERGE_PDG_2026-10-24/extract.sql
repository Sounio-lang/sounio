-- One row per (PDG 2025 computed average, input measurement used in it).
-- Selection exactly as in PROTOCOL.md.
.headers on
.mode csv
SELECT
  d.id                AS avg_id,
  d.pdgid             AS pdgid,
  d.value             AS pdg_value,
  d.error_positive    AS pdg_err_pos,
  d.error_negative    AS pdg_err_neg,
  d.scale_factor      AS pdg_s,
  v.value             AS x,
  v.error_positive    AS e_pos,
  v.error_negative    AS e_neg,
  r.document_id       AS reference
FROM pdgdata d
JOIN pdgmeasurement m        ON m.pdgid = d.pdgid
JOIN pdgmeasurement_values v ON v.pdgmeasurement_id = m.id
JOIN pdgreference r          ON r.id = m.pdgreference_id
WHERE d.edition = '2025'
  AND d.value_type = 'AC'
  AND v.column_name = 'VALUE'
  AND v.used_in_average = 1
  AND v.limit_type IS NULL
  AND v.value IS NOT NULL
  AND v.error_positive IS NOT NULL AND v.error_positive > 0
  AND v.error_negative IS NOT NULL AND v.error_negative > 0
ORDER BY d.id, m.sort;
