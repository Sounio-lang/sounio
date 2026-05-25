#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
WARNING_LOG="$TMP_DIR/internal-label-authority-warning.json"

python3 scripts/ontology/validate_unit_metadata.py \
  --accept-no-clinical-safety \
  --authority-warning-log "$WARNING_LOG"

if ! grep -q '@sounio/safety:no-clinical-dimension' "$WARNING_LOG"; then
  printf 'FAIL  authority warning log missing safety marker\n' >&2
  cat "$WARNING_LOG" >&2
  exit 1
fi

cat <<'MSG'
Ontology internal dimension-label metadata gate passed.
This proves local fixture observation terms and optional local fixture analytes
are connected to auditable internal dimension-label metadata as fixture
well-formedness data. The gate explicitly passes --accept-no-clinical-safety
because checker_authority=false is intentional. It writes and verifies a
structured authority-warning log with @sounio/safety:no-clinical-dimension. It
does not make the metadata a checker, clinical, conversion, UCUM, LOINC, ChEBI,
dosing, or regulatory-exchange authority; unit semantics still come from Sounio
unit types and the checker UnitRegistry.
MSG
