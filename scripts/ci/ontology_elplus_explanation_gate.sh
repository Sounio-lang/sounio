#!/usr/bin/env bash
# Proof-carrying EL+ closure: independently validate reconstructed proof DAGs
# on both engines, prove the engines really differ, compare their derivations
# exactly, and exercise the shared-premise DAG regression.
#
# Engine selection belongs to this gate alone: an inherited compiler override
# can otherwise collapse both legs onto one engine and make the parity claim
# vacuous. Workflow wiring is intentionally deferred; run locally or from a
# maintainer-wired CI job as `bash scripts/ci/ontology_elplus_explanation_gate.sh`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
OUT="$(mktemp -d "${TMPDIR:-/tmp}/elplus-explain.XXXXXX")"
trap 'rm -rf "$OUT"' EXIT

fail() { echo "ELPLUS_EXPLAIN_GATE_FAIL: $1" >&2; exit 1; }

for var in SOUC SOUC_BIN SOUNIO_SOUC_BIN SOUNIO_SOUC_ENGINE MADAROS_RAW_BIN SOUNIO_MADAROS_BIN; do
  [[ -z "${!var:-}" ]] || fail "inherited override $var is set; engine selection belongs to this gate"
done
SOUC="$ROOT_DIR/bin/souc"
[[ -x "$SOUC" ]] || fail "compiler launcher not executable: $SOUC"

run_leg() { # <engine-label> <engine-selector>
  local label="$1" engine="$2"
  local check_log="$OUT/$label.check.log" run_log="$OUT/$label.run.log" shared_log="$OUT/$label.shared.log"
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" check tests/run-pass/ontology_elplus_explanation.sio >"$check_log" 2>&1 \
    || { cat "$check_log" >&2; fail "engine=$label: check failed"; }
  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation.sio >"$run_log" 2>&1 \
    || { cat "$run_log" >&2; fail "engine=$label: fixture run failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_EXPLANATION_OK' "$run_log" \
    || { cat "$run_log" >&2; fail "engine=$label: success marker missing"; }
  python3 scripts/ci/ontology_elplus_explanation_oracle.py <"$run_log" >"$OUT/$label.oracle.log" \
    || { cat "$OUT/$label.oracle.log" >&2; fail "engine=$label: independent proof validation failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_PROOF_ORACLE_NEGATIVE_OK rtos_source=refused' "$OUT/$label.oracle.log" \
    || fail "engine=$label: oracle RtoS negative control missing"
  grep -Fq 'ONTOLOGY_ELPLUS_PROOF_ORACLE_OK traces=7 steps=31 rules=0..11' "$OUT/$label.oracle.log" \
    || { cat "$OUT/$label.oracle.log" >&2; fail "engine=$label: unexpected proof census"; }
  cat "$OUT/$label.oracle.log"

  env SOUNIO_SOUC_ENGINE="$engine" "$SOUC" run tests/run-pass/ontology_elplus_explanation_shared_premise.sio >"$shared_log" 2>&1 \
    || { cat "$shared_log" >&2; fail "engine=$label: shared-premise regression failed"; }
  grep -Fxq 'ONTOLOGY_ELPLUS_SHARED_PREMISE_OK facts=8' "$shared_log" \
    || { cat "$shared_log" >&2; fail "engine=$label: shared-premise DAG was not minimal"; }

  grep -E '^(TRACE_|STEP |ONTOLOGY_ELPLUS_EXPLANATION_OK)' "$run_log" >"$OUT/$label.normalized"
}

run_leg madaros madaros
grep -Fq 'run_check_mode: verdict=0' "$OUT/madaros.check.log" \
  || { cat "$OUT/madaros.check.log" >&2; fail "madaros leg: Madaros check marker missing"; }
grep -Fq 'Compilation successful!' "$OUT/madaros.run.log" \
  || { cat "$OUT/madaros.run.log" >&2; fail "madaros leg: Madaros run marker missing"; }
if grep -Eq '^compile: fns=' "$OUT/madaros.check.log" "$OUT/madaros.run.log" "$OUT/madaros.shared.log"; then
  fail "madaros leg: lean_single compile marker present"
fi

run_leg lean_single lean_single
grep -Eq '^compile: fns=' "$OUT/lean_single.check.log" \
  || { cat "$OUT/lean_single.check.log" >&2; fail "lean_single leg: lean compile marker missing"; }
if grep -Fq 'run_check_mode:' "$OUT/lean_single.check.log" \
  || grep -Fq 'Compilation successful!' "$OUT/lean_single.run.log" "$OUT/lean_single.shared.log"; then
  fail "lean_single leg: Madaros marker present"
fi

# Both engines must accept the same derivations, not merely print the same
# success marker. Compare every trace and every emitted proof step exactly.
if ! cmp -s "$OUT/madaros.normalized" "$OUT/lean_single.normalized"; then
  diff -u "$OUT/madaros.normalized" "$OUT/lean_single.normalized" >&2 || true
  fail "Madaros and lean_single derivations differ"
fi

# The legacy engine remains independent: its standard temporal consumer
# detects accidental changes to the sensitive imported-module code layout.
env SOUNIO_SOUC_ENGINE=madaros "$SOUC" run examples/clinical/pathway_temporal_demo.sio >"$OUT/temporal.log" 2>&1 \
  || { cat "$OUT/temporal.log" >&2; fail "legacy temporal consumer failed"; }
grep -Fxq 'ALL PASS' "$OUT/temporal.log" \
  || { cat "$OUT/temporal.log" >&2; fail "legacy temporal marker missing"; }

echo 'ONTOLOGY_ELPLUS_EXPLAIN_GATE_OK traces=7 steps=31 rules=0..11 engines=madaros+lean_single shared_premise=8 legacy=temporal'
