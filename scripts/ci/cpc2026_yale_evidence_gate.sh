#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCIENTIFIC_REPO="${CPC2026_SCIENTIFIC_REPO:-/workspace/hyperbolic-semantic-networks}"
SOUNIO_SOURCE="$ROOT/examples/cognitive_ossm/run_ossm_native_reference.sio"
FROZEN="$SCIENTIFIC_REPO/results/cpc2026/ossm_statistical_summary.json"
LEGACY="$ROOT/examples/cognitive_ossm/results/ossm_sounio_native_n1000.json"
WORK="${CPC2026_GATE_WORK:-$(mktemp -d /tmp/cpc2026-yale-gate.XXXXXX)}"

trap 'rm -rf "$WORK"' EXIT

fail() {
  printf 'CPC2026_YALE_EVIDENCE_FAIL reason=%s\n' "$*" >&2
  exit 1
}

[[ -f "$FROZEN" ]] || fail "missing_frozen_summary:$FROZEN"
[[ -f "$SOUNIO_SOURCE" ]] || fail "missing_sounio_source:$SOUNIO_SOURCE"

jq -e '
  ((.comparisons.normative_vs_anxious_hidden_entropy_production_rate.cohens_d - 11.650868078041157) | fabs) < 1e-12 and
  ((.comparisons.normative_vs_anxious_mean_associator_norm.cohens_d + 2.781365869022835) | fabs) < 1e-12 and
  .per_regime.normative.n_trajectories == 10000 and
  .per_regime.anxious.n_trajectories == 10000
' "$FROZEN" >/dev/null || fail "frozen_ossm_numbers_drifted"
printf 'CPC2026_FROZEN_OSSM_OK entropy_d=11.650868078041157 associator_d=-2.781365869022835\n'

jq -e '
  .artifact_status == "historical_native_rerun_pre_parser_fix_excluded_from_parity" and
  .current_main_status == "generator_source_repaired_to_check_only_native_v2_compile_blocked"
' "$LEGACY" >/dev/null || fail "legacy_native_artifact_boundary_missing"
printf 'CPC2026_LEGACY_NATIVE_BOUNDARY_OK\n'

"$ROOT/bin/souc" check "$SOUNIO_SOURCE" >"$WORK/madaros-check.log" 2>&1 || {
  tail -n 80 "$WORK/madaros-check.log" >&2
  fail "madaros_check_failed"
}
grep -Fq 'check: OK' "$WORK/madaros-check.log" || fail "madaros_check_marker_missing"
printf 'CPC2026_SOUNIO_SOURCE_CHECK_OK engine=default_madaros\n'

set +e
"$ROOT/bin/souc" compile "$SOUNIO_SOURCE" -o "$WORK/ossm-native" >"$WORK/madaros-compile.log" 2>&1
compile_rc=$?
set -e
if [[ "$compile_rc" -eq 0 ]]; then
  fail "native_v2_status_changed_to_success_revalidate_and_promote_explicitly"
fi
grep -Fq 'native-v2 bridge compilation failed' "$WORK/madaros-compile.log" || {
  tail -n 80 "$WORK/madaros-compile.log" >&2
  fail "unexpected_native_v2_failure"
}
printf 'CPC2026_SOUNIO_NATIVE_CLASSIFIED status=check_only blocker=native_v2_bridge_compilation_failed\n'

grep -Fq 'if b == 101 || b == 69' "$SOUNIO_SOURCE" || fail "source_scientific_exponent_branch_missing"
grep -Fq 'while e < exponent' "$SOUNIO_SOURCE" || fail "source_scientific_exponent_application_missing"
"$ROOT/bin/souc" run "$ROOT/tests/run-pass/cpc2026_scientific_float_parser.sio" >"$WORK/scientific-parser.log" 2>&1 || {
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
