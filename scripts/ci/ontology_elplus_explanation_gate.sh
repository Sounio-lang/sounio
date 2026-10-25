#!/usr/bin/env bash
# Proof-carrying EL+ closure: validate all twelve seed/completion rule types
# on both engines and independently verify every reconstructed premise.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/elplus-explain.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

[[ -x "$SOUC" ]] || { echo "ELPLUS_EXPLAIN_GATE_FAIL: compiler not executable: $SOUC" >&2; exit 1; }

"$SOUC" check tests/run-pass/ontology_elplus_explanation.sio >"$OUT/madaros-check.log" 2>&1
"$SOUC" run tests/run-pass/ontology_elplus_explanation.sio >"$OUT/madaros-run.log" 2>&1
grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_OK' "$OUT/madaros-run.log"
python3 scripts/ci/ontology_elplus_explanation_oracle.py <"$OUT/madaros-run.log"

SOUNIO_SOUC_ENGINE=lean_single "$SOUC" check tests/run-pass/ontology_elplus_explanation.sio >"$OUT/lean-check.log" 2>&1
SOUNIO_SOUC_ENGINE=lean_single "$SOUC" run tests/run-pass/ontology_elplus_explanation.sio >"$OUT/lean-run.log" 2>&1
grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_OK' "$OUT/lean-run.log"
python3 scripts/ci/ontology_elplus_explanation_oracle.py <"$OUT/lean-run.log"

# The legacy engine must remain independent; its standard temporal consumer
# detects accidental changes to the sensitive imported-module code layout.
"$SOUC" run examples/clinical/pathway_temporal_demo.sio >"$OUT/temporal.log" 2>&1
grep -Fxq 'ALL PASS' "$OUT/temporal.log"

echo 'ONTOLOGY_ELPLUS_EXPLAIN_GATE_OK traces=7 rules=0..11 engines=madaros+lean_single legacy=temporal'
