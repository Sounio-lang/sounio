#!/usr/bin/env bash
# scripts/dev/madaros_wave13_tip_green_gate.sh
#
# Wave13 tip-green regression lock for origin/main Madaros science/compiler.
# Supersedes Wave12 tip-green by keeping the nine Wave12 locks and promoting
# cd_exact_generic_i64 e2e (PR #1392 — ZD PROVED under default Madaros) from
# claims_not_made into a REQUIRED sub-gate:
#
#   Wave12 retained:
#     1. dual            — scripts/madaros_dual_import_gate.sh
#     2. order_spread    — scripts/madaros_order_spread_native_gate.sh
#     3. knowledge_method— scripts/madaros_knowledge_method_residual_gate.sh
#     4. global_array    — scripts/dev/madaros_global_array_init_gate.sh
#     5. named_path      — scripts/dev/named_path_import_print_f64_gate.sh
#     6. unsplit_oct     — scripts/madaros_unsplit_oct_mul_gate.sh
#     7. epistemic_trust — scripts/epistemic_trust_gate.sh
#     8. global_array_ref— scripts/ci/madaros_global_array_ref_gate.sh
#     9. imported_f64    — scripts/ci/madaros_imported_f64_const_gate.sh
#
#   Wave13 promotion (PR #1392):
#    10. cd_exact        — scripts/dev/madaros_cd_exact_generic_i64_gate.sh
#                          (compile+run: ZD PROVED / SQ PASS / NONZERO PASS / 16×COMP)
#                          Prefer scripts/madaros_cd_exact_e2e_gate.sh when a
#                          rebuilt RAW ELF is available; this gate uses default
#                          bin/souc (stock prebuilt after #1392 merge).
#
# Exit 0 only when every sub-gate exits 0 and emits its expected sentinel.
# Writes a machine-readable tip receipt:
#   artifacts/compiler/madaros_wave13_tip_green_receipt.v1.json
#
# Does not rebuild Madaros. Does not pin lean_single. Uses default bin/souc.
# If stock prebuilt lags #1392 (cd_exact RED):
#   scripts/dev/souc-build-lock.sh make build-madaros
#   MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash "$0"
#
# Usage:
#   bash scripts/dev/madaros_wave13_tip_green_gate.sh
#   SOUC=./bin/souc bash scripts/dev/madaros_wave13_tip_green_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUC

RECEIPT_DIR="$ROOT/artifacts/compiler"
RECEIPT="$RECEIPT_DIR/madaros_wave13_tip_green_receipt.v1.json"
LOG_DIR="$(mktemp -d /tmp/madaros_wave13_tip_green.XXXXXX)"
trap 'rm -rf "$LOG_DIR"' EXIT

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
COMMIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

RAW=""
for cand in \
  "${MADAROS_RAW_BIN:-}" \
  "$ROOT/artifacts/self-hosted/madaros" \
  "$ROOT/bin/madaros-linux-x86_64"
do
  [[ -n "$cand" && -x "$cand" ]] || continue
  if [[ "$(head -c2 "$cand" 2>/dev/null)" != '#!' ]]; then
    RAW="$cand"
    break
  fi
done
RAW_SHA256="missing"
if [[ -n "$RAW" ]]; then
  RAW_SHA256="$(sha256sum "$RAW" | awk '{print $1}')"
fi

ENGINE_LINE="$("$SOUC" --version 2>&1 | head -1 || echo unknown)"

echo "== madaros_wave13_tip_green_gate =="
echo "git_sha=$COMMIT_SHORT"
echo "engine=$ENGINE_LINE"
echo "raw_elf=${RAW:-none}"
echo "raw_elf_sha256=$RAW_SHA256"
echo "souc=$SOUC"
echo "log_dir=$LOG_DIR"
echo "pr_cd_exact_e2e=1392"
echo

# name|script|expected_token
GATES=(
  "dual|scripts/madaros_dual_import_gate.sh|MADAROS_DUAL_IMPORT_GATE_OK"
  "order_spread|scripts/madaros_order_spread_native_gate.sh|MADAROS_ORDER_SPREAD_NATIVE_GATE_OK"
  "knowledge_method|scripts/madaros_knowledge_method_residual_gate.sh|MADAROS_KNOWLEDGE_METHOD_RESIDUAL_GATE_OK"
  "global_array|scripts/dev/madaros_global_array_init_gate.sh|GLOBAL_ARRAY_INIT_GATE_OK"
  "named_path|scripts/dev/named_path_import_print_f64_gate.sh|NAMED_PATH_IMPORT_GATE_PASS"
  "unsplit_oct|scripts/madaros_unsplit_oct_mul_gate.sh|MADAROS_UNSPLIT_OCT_MUL_GATE_OK"
  "epistemic_trust|scripts/epistemic_trust_gate.sh|EPISTEMIC_TRUST_GATE_OK"
  "global_array_ref|scripts/ci/madaros_global_array_ref_gate.sh|MADAROS_GLOBAL_ARRAY_REF_GATE_OK"
  "imported_f64|scripts/ci/madaros_imported_f64_const_gate.sh|MADAROS_IMPORTED_F64_CONST_GATE_OK"
  "cd_exact|scripts/dev/madaros_cd_exact_generic_i64_gate.sh|MADAROS_CD_EXACT_GENERIC_I64_GATE_OK"
)

declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_RC=()
declare -a RESULT_DURATION=()
declare -a RESULT_TOKEN=()

fail=0
idx=0
total=${#GATES[@]}
cd_exact_status="not_run"

run_one() {
  local name="$1" script="$2" token="$3"
  local log="$LOG_DIR/${name}.log"
  local start end dur rc
  idx=$((idx + 1))
  echo "---------- [$idx/$total] $name ----------"
  echo "script=$script"
  if [[ ! -f "$ROOT/$script" ]]; then
    echo "RED  $name reason=missing_script path=$script"
    RESULT_NAMES+=("$name")
    RESULT_STATUS+=("missing_script")
    RESULT_RC+=(127)
    RESULT_DURATION+=(0)
    RESULT_TOKEN+=("$token")
    fail=1
    if [[ "$name" == "cd_exact" ]]; then
      cd_exact_status="missing_script"
    fi
    return
  fi
  start=$(date +%s)
  set +e
  bash "$ROOT/$script" >"$log" 2>&1
  rc=$?
  set -e
  end=$(date +%s)
  dur=$((end - start))

  local status="pass"
  if [[ $rc -ne 0 ]]; then
    status="fail"
    fail=1
  elif [[ -n "$token" ]] && ! grep -qF "$token" "$log"; then
    status="token_missing"
    fail=1
    rc=2
  fi

  if [[ "$name" == "epistemic_trust" && "$status" == "pass" ]]; then
    if ! grep -qF 'k95i=2776' "$log"; then
      status="k95_token_missing"
      fail=1
      rc=3
    elif grep -qE 'k95i=1960|want 2776' "$log" && ! grep -qF 'PASS: k95i=2776' "$log"; then
      status="k95_collapse"
      fail=1
      rc=4
    fi
  fi

  if [[ "$name" == "imported_f64" && "$status" == "pass" ]]; then
    if ! grep -qF '4609434218613702656' "$log"; then
      status="a_const_bits_missing"
      fail=1
      rc=5
    elif ! grep -qF '4612811918334230528' "$log"; then
      status="b_const_bits_missing"
      fail=1
      rc=6
    elif ! grep -qF 'IMPORTED_F64_CONST_OK' "$log"; then
      status="bss_witness_token_missing"
      fail=1
      rc=7
    fi
  fi

  if [[ "$name" == "cd_exact" && "$status" == "pass" ]]; then
    if ! grep -qF 'ZD PROVED' "$log"; then
      status="zd_token_missing"
      fail=1
      rc=8
    fi
  fi

  if [[ "$status" == "pass" ]]; then
    echo "GREEN $name rc=$rc ${dur}s token=$token"
  else
    echo "RED   $name status=$status rc=$rc ${dur}s token=$token"
    echo "----- tail $name -----"
    tail -40 "$log" || true
    echo "----- end tail -----"
  fi

  if [[ "$name" == "cd_exact" ]]; then
    cd_exact_status="$status"
  fi

  RESULT_NAMES+=("$name")
  RESULT_STATUS+=("$status")
  RESULT_RC+=("$rc")
  RESULT_DURATION+=("$dur")
  RESULT_TOKEN+=("$token")
}

for entry in "${GATES[@]}"; do
  IFS='|' read -r gname gscript gtoken <<<"$entry"
  run_one "$gname" "$gscript" "$gtoken"
  echo
done

OVERALL="pass"
[[ $fail -eq 0 ]] || OVERALL="fail"

gates_json="["
for i in "${!RESULT_NAMES[@]}"; do
  [[ $i -gt 0 ]] && gates_json+=","
  gates_json+=$(printf '\n    {"name":"%s","status":"%s","rc":%s,"duration_s":%s,"expected_token":"%s"}' \
    "${RESULT_NAMES[$i]}" \
    "${RESULT_STATUS[$i]}" \
    "${RESULT_RC[$i]}" \
    "${RESULT_DURATION[$i]}" \
    "${RESULT_TOKEN[$i]}")
done
gates_json+=$'\n  ]'

relpath_under_root() {
  local p="$1"
  [[ -z "$p" ]] && { printf ''; return; }
  python3 - "$ROOT" "$p" <<'PY'
import os, sys
root = os.path.realpath(sys.argv[1])
path = os.path.realpath(sys.argv[2])
if path == root:
    print(".")
elif path.startswith(root + os.sep):
    print(os.path.relpath(path, root))
else:
    print(sys.argv[2])
PY
}

RAW_REL="$(relpath_under_root "${RAW:-}")"
SOUC_REL="$(relpath_under_root "$SOUC")"

CLAIMS_JSON='[
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "knowledge_method_form_parity",
    "global_array_init_i64_f64_i8",
    "named_path_import_print_f64_default_and_opt",
    "unsplit_oct_mul_no_reentry",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "global_array_ref_mut_bss_defect_b",
    "imported_module_f64_const_defect_a",
    "imported_module_f64_bss_offset_remap_defect_a_prime",
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392"
  ]'
CLAIMS_NOT_JSON='[
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "bare_cross_module_global_ident_from_main",
    "all_madaros_residuals_closed"
  ]'
if [[ "$cd_exact_status" != "pass" ]]; then
  CLAIMS_JSON='[
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "knowledge_method_form_parity",
    "global_array_init_i64_f64_i8",
    "named_path_import_print_f64_default_and_opt",
    "unsplit_oct_mul_no_reentry",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "global_array_ref_mut_bss_defect_b",
    "imported_module_f64_const_defect_a",
    "imported_module_f64_bss_offset_remap_defect_a_prime"
  ]'
  CLAIMS_NOT_JSON='[
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392",
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "bare_cross_module_global_ident_from_main",
    "all_madaros_residuals_closed"
  ]'
fi

mkdir -p "$RECEIPT_DIR"
cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_wave13_tip_green_receipt.v1",
  "status": "$OVERALL",
  "wave": "wave13",
  "role": "tip_green_regression_lock",
  "supersedes": "madaros_wave12_tip_green_receipt.v1",
  "engine": "madaros_default",
  "engine_line": $(printf '%s' "$ENGINE_LINE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "commit_short": "$COMMIT_SHORT",
  "utc": "$UTC",
  "raw_elf": $(printf '%s' "$RAW_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "raw_elf_sha256": "$RAW_SHA256",
  "souc": $(printf '%s' "$SOUC_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "require_cd_exact": true,
  "pr_cd_exact_e2e": 1392,
  "gates": $gates_json,
  "claims": $CLAIMS_JSON,
  "claims_not_made": $CLAIMS_NOT_JSON,
  "measurement": {
    "tip_sha": "$COMMIT_SHORT",
    "cd_exact": "$cd_exact_status",
    "wave12_tip_green_plus_cd_exact": "see_gates",
    "red_count_at_ship": "see_gates",
    "note": "Wave13 tip-green requires cd_exact_generic_i64 e2e (PR #1392 ZD PROVED). Rebuild Madaros if stock prebuilt lags."
  },
  "verdict": "$OVERALL"
}
EOF

echo "===== SUMMARY ====="
for i in "${!RESULT_NAMES[@]}"; do
  printf '  %-18s %-14s rc=%-3s %ss\n' \
    "${RESULT_NAMES[$i]}" \
    "${RESULT_STATUS[$i]}" \
    "${RESULT_RC[$i]}" \
    "${RESULT_DURATION[$i]}"
done
echo "overall=$OVERALL"
echo "cd_exact=$cd_exact_status"
echo "receipt: $RECEIPT"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_WAVE13_TIP_GREEN_GATE_OK"
  exit 0
fi
if [[ "$cd_exact_status" != "pass" ]]; then
  echo "NOTE: cd_exact RED — if stock prebuilt lags #1392, rebuild:" >&2
  echo "  scripts/dev/souc-build-lock.sh make build-madaros" >&2
  echo "  MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash $0" >&2
fi
echo "MADAROS_WAVE13_TIP_GREEN_GATE_FAIL" >&2
exit 1
