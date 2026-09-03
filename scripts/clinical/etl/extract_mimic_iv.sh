#!/usr/bin/env bash
# scripts/clinical/etl/extract_mimic_iv.sh
#
# M4 — MIMIC-IV vancomycin TDM cohort extraction driver.
#
# Wraps scripts/clinical/etl/mimic_iv_vancomycin.sql for the two
# supported deployment modes:
#
#   1. BigQuery (PhysioNet-hosted; the default and recommended path).
#      Requires: gcloud SDK + credentialed access to
#      `physionet-data.mimiciv_3_1`. The Sounio cluster has this
#      pre-configured under service account
#      `sounio-mimic-readonly@sounio-research.iam.gserviceaccount.com`.
#
#   2. PostgreSQL (locally-hosted MIMIC-IV). Requires running the
#      mimic-code/mimic-iv-derived/ scripts to populate the derived
#      tables; the sql file must be edited to drop the
#      `physionet-data.mimiciv_3_1.` prefix from each table.
#
# Output: a CSV file at the path of `tdm_cohort_synthetic_v2.csv`'s
# sibling, named `mimic_iv_vancomycin_cohort.csv`. The schema is
# identical so that `process_tdm_cohort.sh` accepts either file
# without modification.
#
# Usage:
#   bash scripts/clinical/etl/extract_mimic_iv.sh [--mode bigquery|psql]
#                                                  [--dest cohort.csv]
#
# Default: --mode bigquery, --dest scripts/clinical/data_synthetic/mimic_iv_vancomycin_cohort.csv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SQL="$ROOT/scripts/clinical/etl/mimic_iv_vancomycin.sql"
DEFAULT_DEST="$ROOT/scripts/clinical/data_synthetic/mimic_iv_vancomycin_cohort.csv"

MODE="bigquery"
DEST="$DEFAULT_DEST"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --dest)
            DEST="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "$SQL" ]]; then
    echo "ERROR: SQL file not found: $SQL" >&2
    exit 2
fi

mkdir -p "$(dirname "$DEST")"

case "$MODE" in
    bigquery)
        if ! command -v bq >/dev/null 2>&1; then
            cat >&2 <<EOF
ERROR: 'bq' (BigQuery CLI) not found.

This script requires the gcloud SDK (https://cloud.google.com/sdk).
Install via:
  curl https://sdk.cloud.google.com | bash
  exec -l \$SHELL
  gcloud init

Then authenticate with the credentialed service account:
  gcloud auth activate-service-account \\
      --key-file=\$SOUNIO_MIMIC_KEY

\$SOUNIO_MIMIC_KEY should point to the GCP service-account JSON
provisioned under DUA agreement; consult
docs/governance/data_access_credentials.md for the issuance flow.
EOF
            exit 3
        fi
        echo "=== MIMIC-IV BigQuery extraction ==="
        echo "Reading SQL: $SQL"
        echo "Writing to:  $DEST"
        echo
        bq query \
            --use_legacy_sql=false \
            --format=csv \
            --max_rows=100000 \
            < "$SQL" > "$DEST"
        ;;
    psql)
        if ! command -v psql >/dev/null 2>&1; then
            echo "ERROR: 'psql' (PostgreSQL client) not found." >&2
            exit 3
        fi
        if [[ -z "${MIMIC_PG_URI:-}" ]]; then
            cat >&2 <<EOF
ERROR: \$MIMIC_PG_URI environment variable is not set.

Set it to a PostgreSQL connection string against your local
MIMIC-IV deployment, e.g.:
  export MIMIC_PG_URI=postgresql://mimic@localhost/mimiciv_3_1

You must also pre-process the SQL file to drop the
'physionet-data.mimiciv_3_1.' prefix from each table reference;
the BigQuery-style references are not portable to PostgreSQL.
EOF
            exit 3
        fi
        echo "=== MIMIC-IV PostgreSQL extraction ==="
        echo "Connection: \$MIMIC_PG_URI"
        echo "SQL:        $SQL"
        echo "Output:     $DEST"
        echo
        psql "$MIMIC_PG_URI" \
             --no-align \
             --field-separator=',' \
             --tuples-only=false \
             --command="\\copy ($(cat "$SQL")) TO STDOUT WITH (FORMAT CSV, HEADER)" \
             > "$DEST"
        ;;
    *)
        echo "ERROR: unknown --mode: $MODE (expected bigquery|psql)" >&2
        exit 2
        ;;
esac

n_rows=$(wc -l < "$DEST")
n_patients=$((n_rows - 1))   # subtract header
echo
echo "Extraction complete."
echo "  Output:    $DEST"
echo "  Patients:  $n_patients"
echo
echo "Next step:"
echo "  bash scripts/clinical/process_tdm_cohort.sh '$DEST'"
echo
