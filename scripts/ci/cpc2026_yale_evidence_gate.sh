#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCIENTIFIC_REPO="${CPC2026_SCIENTIFIC_REPO:-/workspace/hyperbolic-semantic-networks}"
SOUNIO_SOURCE="$ROOT/examples/cognitive_ossm/run_ossm_native_reference.sio"
SOUNIO_INPUT="$SCIENTIFIC_REPO/data/cpc2026/sounio_input"
FROZEN="$SCIENTIFIC_REPO/results/cpc2026/ossm_statistical_summary.json"
LEGACY="$ROOT/examples/cognitive_ossm/results/ossm_sounio_native_n1000.json"
WORK="${CPC2026_GATE_WORK:-$(mktemp -d /tmp/cpc2026-yale-gate.XXXXXX)}"
MADAROS_RAW="${CPC2026_MADAROS_RAW_BIN:-}"

trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'CPC2026_YALE_EVIDENCE_FAIL reason=%s\n' "$*" >&2
  exit 1
}

souc() {
  if [[ -n "$MADAROS_RAW" ]]; then
    MADAROS_RAW_BIN="$MADAROS_RAW" "$ROOT/bin/souc" "$@"
  else
    "$ROOT/bin/souc" "$@"
  fi
}

[[ -f "$FROZEN" ]] || fail "missing_frozen_summary:$FROZEN"
[[ -f "$SOUNIO_SOURCE" ]] || fail "missing_sounio_source:$SOUNIO_SOURCE"
[[ -d "$SOUNIO_INPUT" ]] || fail "missing_sounio_input:$SOUNIO_INPUT"
if [[ -n "$MADAROS_RAW" ]]; then
  [[ -x "$MADAROS_RAW" ]] || fail "madaros_raw_not_executable:$MADAROS_RAW"
fi

jq -e '
  ((.comparisons.normative_vs_anxious_hidden_entropy_production_rate.cohens_d - 11.650868078041157) | fabs) < 1e-12 and
  ((.comparisons.normative_vs_anxious_mean_associator_norm.cohens_d + 2.781365869022835) | fabs) < 1e-12 and
  .per_regime.normative.n_trajectories == 10000 and
  .per_regime.anxious.n_trajectories == 10000
' "$FROZEN" >/dev/null || fail "frozen_ossm_numbers_drifted"
printf 'CPC2026_FROZEN_OSSM_OK entropy_d=11.650868078041157 associator_d=-2.781365869022835\n'

jq -e '
  .artifact_status == "historical_native_rerun_pre_parser_fix_excluded_from_parity"
' "$LEGACY" >/dev/null || fail "legacy_native_artifact_boundary_missing"
printf 'CPC2026_LEGACY_NATIVE_BOUNDARY_OK\n'

souc check "$SOUNIO_SOURCE" >"$WORK/madaros-check.log" 2>&1 || {
  tail -n 80 "$WORK/madaros-check.log" >&2
  fail "madaros_check_failed"
}
grep -Fq 'check: OK' "$WORK/madaros-check.log" || fail "madaros_check_marker_missing"
printf 'CPC2026_SOUNIO_SOURCE_CHECK_OK engine=default_madaros\n'

souc compile "$ROOT/tests/native-v2/bss_global_array_index_runtime.sio" \
  -o "$WORK/bss-global-index" >"$WORK/bss-global-index-compile.log" 2>&1 || {
    tail -n 80 "$WORK/bss-global-index-compile.log" >&2
    fail "bss_global_index_compile_failed"
  }
"$WORK/bss-global-index" || fail "bss_global_index_runtime_failed"
printf 'CPC2026_NATIVE_BSS_INDEX_OK element_sizes=i8,i64 local_array=true\n'

souc compile "$SOUNIO_SOURCE" -o "$WORK/ossm-native" >"$WORK/madaros-compile.log" 2>&1 || {
  tail -n 80 "$WORK/madaros-compile.log" >&2
  fail "native_v2_compile_failed"
}
[[ -x "$WORK/ossm-native" ]] || fail "native_v2_output_missing"

timeout 60 "$WORK/ossm-native" \
  --input-root "$SOUNIO_INPUT" \
  --max-trajectories 2 \
  --max-steps 8 \
  --output "$WORK/ossm-native-summary.json" >"$WORK/madaros-run.log" 2>&1 || {
    tail -n 80 "$WORK/madaros-run.log" >&2
    fail "native_v2_bounded_runtime_failed"
  }
jq -e '
  .max_trajectories == 2 and
  .max_steps == 8 and
  .normative.n == 2 and
  .anxious.n == 2 and
  .ruminative.n == 2 and
  .psychotic.n == 2 and
  (.normative.mean_hidden_entropy_production_rate > 0) and
  (.anxious.mean_hidden_entropy_production_rate > 0) and
  (.normative.mean_h_entropy > 0) and
  (.anxious.mean_h_entropy > 0)
' "$WORK/ossm-native-summary.json" >/dev/null || fail "native_v2_bounded_summary_invalid"
printf 'CPC2026_SOUNIO_NATIVE_OK status=bounded_runtime trajectories=2 steps=8 parity_claim=false\n'

grep -Fq 'if b == 101 || b == 69' "$SOUNIO_SOURCE" || fail "source_scientific_exponent_branch_missing"
grep -Fq 'while e < exponent' "$SOUNIO_SOURCE" || fail "source_scientific_exponent_application_missing"
souc run "$ROOT/tests/run-pass/cpc2026_scientific_float_parser.sio" >"$WORK/scientific-parser.log" 2>&1 || {
  tail -n 80 "$WORK/scientific-parser.log" >&2
  fail "scientific_float_parser_failed"
}
grep -Fq 'CPC2026_SCIENTIFIC_FLOAT_OK' "$WORK/scientific-parser.log" || fail "scientific_float_parser_marker_missing"
printf 'CPC2026_SCIENTIFIC_FLOAT_OK engine=default_madaros\n'

run_receipt() {
  local source="$1" marker="$2" label="$3"
  SOUNIO_SOUC_ENGINE=lean_single "$ROOT/bin/souc" run "$ROOT/$source" >"$WORK/$label.log" 2>&1 || {
    tail -n 80 "$WORK/$label.log" >&2
    fail "receipt_failed:$label"
  }
  grep -Fq "$marker" "$WORK/$label.log" || fail "receipt_marker_missing:$label"
  printf 'CPC2026_SOUNIO_RECEIPT_OK label=%s engine=lean_single\n' "$label"
}

run_receipt tests/run-pass/order_spread_exact_n4.sio 'exact spread = 2.044226: PASS' order_spread
run_receipt tests/run-pass/octonion_associator_gum_validation.sio 'compiler variance = 0.640000' associator_gum
run_receipt tests/run-pass/associator_variance_mc.sio 'MC var(A)    = 0.643979' associator_mc

printf 'CPC2026_YALE_EVIDENCE_OK scientific_repo=%s\n' "$SCIENTIFIC_REPO"
