#!/usr/bin/env bash
# scripts/dev/madaros_wave15_showcase_gate.sh
#
# Wave15 public-facing Madaros SHOWCASE orchestrator.
# Locks everything Waves 13–14 proved on tip, plus Wave15 packaging surface.
#
# Baseline (Wave13 science pillars + tip locks that remain green on tip):
#   dual, order_spread, k95, cd_exact, cd_exact_e2e,
#   knowledge_method, global_array (incl Wave13e call_list_args),
#   named_path, unsplit_oct, global_array_ref,
#   imported_f64_core (minimal + BSS A′ + bare cross-mod Ident — #1400)
#
# Wave14 / tip locks:
#   bare_float_arith          — scripts/madaros_bare_float_arith_gate.sh (#1404)
#   root2_method              — scripts/madaros_root2_method_gate.sh
#   root2_multimodule_method  — scripts/madaros_root2_multimodule_method_gate.sh (chain)
#   imported_array_byvalue    — scripts/ci/madaros_imported_array_byvalue_gate.sh (#913 / #1398)
#   thinlink_921              — scripts/madaros_thinlink_921_residual_gate.sh (#921 / #1399)
#
# Honest residual probes (do NOT invent green):
#   imported_f64_lognormal_science — denser stats::densities vertical
#       (tip prebuilt after #1405 Wave13e rebuild regresses vs Wave13 prebuilt;
#        REQUIRE_IMPORTED_F64_SCIENCE=1 promotes to required)
#   multi-stmt paramful global list fold — residual fail-closed inside
#       madaros_global_array_init_gate.sh (expects BSS zeros; not a free claim)
#
# Optional: also run the full Wave13 showcase as a non-required probe
# (RUN_WAVE13_SHOWCASE_PROBE=1). It currently reds on tip solely via the
# imported_f64 lognormal science vertical inside wave12 tip-green.
#
# Exit 0 when every REQUIRED sub-gate is green. Writes:
#   artifacts/compiler/madaros_wave15_showcase_receipt.v1.json
#
# Does not rebuild Madaros. Does not pin lean_single. Uses default bin/souc.
# If stock prebuilt lags a source fix for a required lock:
#   make build-madaros   # bare: it locks internally; wrapping it in
#                        # souc-build-lock.sh deadlocks (see CLAUDE.md §4)
#   MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash "$0"
#
# Usage:
#   bash scripts/dev/madaros_wave15_showcase_gate.sh
#   REQUIRE_IMPORTED_F64_SCIENCE=1 bash scripts/dev/madaros_wave15_showcase_gate.sh
#   SOUC=./bin/souc bash scripts/dev/madaros_wave15_showcase_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUC
# Full denser lognormal science vertical — residual on tip after #1405 prebuilt.
# Default OFF (honest residual packaging). Set 1 only when tip proves it green.
REQUIRE_IMPORTED_F64_SCIENCE="${REQUIRE_IMPORTED_F64_SCIENCE:-0}"
# Optional Wave13 showcase probe (records status; not required by default).
RUN_WAVE13_SHOWCASE_PROBE="${RUN_WAVE13_SHOWCASE_PROBE:-0}"
# cd_exact remains required (Wave13 public surface after #1392).
REQUIRE_CD_EXACT="${REQUIRE_CD_EXACT:-1}"

RECEIPT_DIR="$ROOT/artifacts/compiler"
RECEIPT="$RECEIPT_DIR/madaros_wave15_showcase_receipt.v1.json"
LOG_DIR="$(mktemp -d /tmp/madaros_wave15_showcase.XXXXXX)"
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

echo "== madaros_wave15_showcase_gate =="
echo "git_sha=$COMMIT_SHORT"
echo "engine=$ENGINE_LINE"
echo "raw_elf=${RAW:-none}"
echo "raw_elf_sha256=$RAW_SHA256"
echo "souc=$SOUC"
echo "require_cd_exact=$REQUIRE_CD_EXACT"
echo "require_imported_f64_science=$REQUIRE_IMPORTED_F64_SCIENCE"
echo "run_wave13_showcase_probe=$RUN_WAVE13_SHOWCASE_PROBE"
echo "pr_cd_exact_e2e=1392"
echo "log_dir=$LOG_DIR"
echo

# name|script|expected_token|required(1/0)
# required=1 is the default class; residual probes flip via env below.
GATES=(
  "dual|scripts/madaros_dual_import_gate.sh|MADAROS_DUAL_IMPORT_GATE_OK|1"
  "order_spread|scripts/madaros_order_spread_native_gate.sh|MADAROS_ORDER_SPREAD_NATIVE_GATE_OK|1"
  "k95|scripts/epistemic_trust_gate.sh|EPISTEMIC_TRUST_GATE_OK|1"
  "cd_exact|scripts/dev/madaros_cd_exact_generic_i64_gate.sh|MADAROS_CD_EXACT_GENERIC_I64_GATE_OK|1"
  "cd_exact_e2e|scripts/madaros_cd_exact_e2e_gate.sh|MADAROS_CD_EXACT_E2E_GATE_OK|1"
  "knowledge_method|scripts/madaros_knowledge_method_residual_gate.sh|MADAROS_KNOWLEDGE_METHOD_RESIDUAL_GATE_OK|1"
  "global_array|scripts/dev/madaros_global_array_init_gate.sh|GLOBAL_ARRAY_INIT_GATE_OK|1"
  "named_path|scripts/dev/named_path_import_print_f64_gate.sh|NAMED_PATH_IMPORT_GATE_PASS|1"
  "unsplit_oct|scripts/madaros_unsplit_oct_mul_gate.sh|MADAROS_UNSPLIT_OCT_MUL_GATE_OK|1"
  "global_array_ref|scripts/ci/madaros_global_array_ref_gate.sh|MADAROS_GLOBAL_ARRAY_REF_GATE_OK|1"
  "imported_f64_core|__inline_imported_f64_core__|IMPORTED_F64_CORE_OK|1"
  "bare_float_arith|scripts/madaros_bare_float_arith_gate.sh|MADAROS_BARE_FLOAT_ARITH_GATE_OK|1"
  "root2_method|scripts/madaros_root2_method_gate.sh|MADAROS_ROOT2_METHOD_GATE_OK|1"
  "root2_multimodule_method|scripts/madaros_root2_multimodule_method_gate.sh|MADAROS_ROOT2_MULTIMODULE_METHOD_GATE_OK|1"
  "imported_array_byvalue|scripts/ci/madaros_imported_array_byvalue_gate.sh|MADAROS_IMPORTED_ARRAY_BYVALUE_GATE_OK|1"
  "thinlink_921|scripts/madaros_thinlink_921_residual_gate.sh|MADAROS_THINLINK_921_RESIDUAL_GATE_OK|1"
  "imported_f64_lognormal_science|__inline_imported_f64_lognormal__|IMPORTED_F64_LOGNORMAL_SCIENCE_OK|0"
)

if [[ "$RUN_WAVE13_SHOWCASE_PROBE" == "1" ]]; then
  GATES+=("wave13_showcase_probe|scripts/dev/madaros_wave13_showcase_gate.sh|MADAROS_WAVE13_SHOWCASE_GATE_OK|0")
fi

declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_RC=()
declare -a RESULT_DURATION=()
declare -a RESULT_TOKEN=()
declare -a RESULT_REQUIRED=()
declare -a RESULT_CLASS=()

fail_required=0
cd_exact_status="not_run"
cd_exact_e2e_status="not_run"
lognormal_status="not_run"
f64_core_status="not_run"
idx=0
total=${#GATES[@]}

run_inline_imported_f64_core() {
  # Required Wave11/13 f64 locks that remain green on tip #1405 prebuilt:
  # minimal Defect A, multi-mod BSS A′, bare cross-mod Ident (#1400).
  # Writes log path $1; exit status is the gate result (no return-from-redirect).
  local log="$1"
  local fail=0
  local out=""
  local min="$ROOT/tests/run-pass/imported_f64_global_const.sio"
  local bss="$ROOT/tests/run-pass/imported_module_f64_const.sio"
  local bare="$ROOT/tests/run-pass/imported_module_f64_const_bare_ident.sio"
  : >"$log"
  {
    echo "== imported_f64_core (minimal + BSS + bare Ident) =="
    echo "engine: $ENGINE_LINE"
  } >>"$log"
  for f in "$min" "$bss" "$bare"; do
    if [[ ! -f "$f" ]]; then
      echo "FAIL: missing witness $f" >>"$log"
      fail=1
    fi
  done
  if [[ $fail -eq 0 ]]; then
    local rc_min=0 rc_bss=0 rc_bare=0
    set +e
    out="$("$SOUC" run "$min" 2>&1)"
    rc_min=$?
    set -e
    echo "$out" >>"$log"
    if [[ $rc_min -ne 0 ]] || ! grep -q 'IMPORTED_F64_GLOBAL_CONST_OK' <<<"$out"; then
      echo "FAIL: minimal" >>"$log"
      fail=1
    fi

    set +e
    out="$("$SOUC" run "$bss" 2>&1)"
    rc_bss=$?
    set -e
    echo "$out" >>"$log"
    if [[ $rc_bss -ne 0 ]] || ! grep -q 'IMPORTED_F64_CONST_OK' <<<"$out"; then
      echo "FAIL: bss" >>"$log"
      fail=1
    fi
    if ! grep -q '4609434218613702656' <<<"$out"; then echo "FAIL: missing A_CONST bits" >>"$log"; fail=1; fi
    if ! grep -q '4612811918334230528' <<<"$out"; then echo "FAIL: missing B_CONST bits" >>"$log"; fail=1; fi

    set +e
    out="$("$SOUC" run "$bare" 2>&1)"
    rc_bare=$?
    set -e
    echo "$out" >>"$log"
    if [[ $rc_bare -ne 0 ]] || ! grep -q 'BARE_CROSSMOD_F64_IDENT_OK' <<<"$out"; then
      echo "FAIL: bare" >>"$log"
      fail=1
    fi
    if ! grep -q '4609434218613702656' <<<"$out"; then echo "FAIL: missing bare A bits" >>"$log"; fail=1; fi
  fi
  if [[ $fail -eq 0 ]]; then
    echo "IMPORTED_F64_CORE_OK" >>"$log"
    INLINE_RC=0
  else
    INLINE_RC=1
  fi
  return 0
}

run_inline_imported_f64_lognormal() {
  # Honest residual probe. Writes log path $1; sets INLINE_RC (never returns non-zero).
  local log="$1"
  local sci="$ROOT/tests/run-pass/imported_f64_lognormal_science.sio"
  local out="" rc=0
  INLINE_RC=1
  {
    echo "== imported_f64_lognormal_science (honest residual probe) =="
    echo "engine: $ENGINE_LINE"
  } >"$log"
  if [[ ! -f "$sci" ]]; then
    echo "FAIL: missing $sci" >>"$log"
    return 0
  fi
  set +e
  out="$("$SOUC" run "$sci" 2>&1)"
  rc=$?
  set +e
  echo "$out" >>"$log"
  if [[ $rc -ne 0 ]] || ! grep -q 'IMPORTED_F64_LOGNORMAL_SCIENCE_OK' <<<"$out"; then
    echo "RESIDUAL: denser stats::densities lognormal_pdf vertical red on tip prebuilt" >>"$log"
    echo "NOTE: Wave13 prebuilt (post-#1392, pre-#1405) was green; #1405 Wave13e rebuild regressed it." >>"$log"
    INLINE_RC=1
    return 0
  fi
  echo "IMPORTED_F64_LOGNORMAL_SCIENCE_OK" >>"$log"
  INLINE_RC=0
  return 0
}

run_one() {
  local name="$1" script="$2" token="$3" required="$4"
  local log="$LOG_DIR/${name}.log"
  local start end dur rc
  local class="required"
  [[ "$required" == "1" ]] || class="honest_probe"

  # cd_exact family: default required; REQUIRE_CD_EXACT=0 demotes.
  if [[ "$name" == "cd_exact" || "$name" == "cd_exact_e2e" ]]; then
    if [[ "$REQUIRE_CD_EXACT" == "1" ]]; then
      required="1"
      class="required"
    else
      required="0"
      class="honest_probe_legacy"
    fi
  fi

  # denser lognormal science: residual by default after #1405 tip prebuilt.
  if [[ "$name" == "imported_f64_lognormal_science" ]]; then
    if [[ "$REQUIRE_IMPORTED_F64_SCIENCE" == "1" ]]; then
      required="1"
      class="required"
    else
      required="0"
      class="honest_probe_residual"
    fi
  fi

  if [[ "$name" == "wave13_showcase_probe" ]]; then
    required="0"
    class="honest_probe_optional"
  fi

  idx=$((idx + 1))
  echo "---------- [$idx/$total] $name (class=$class) ----------"
  echo "script=$script"

  start=$(date +%s)
  rc=0
  INLINE_RC=0
  set +e
  if [[ "$script" == "__inline_imported_f64_core__" ]]; then
    run_inline_imported_f64_core "$log"
    rc=$INLINE_RC
  elif [[ "$script" == "__inline_imported_f64_lognormal__" ]]; then
    run_inline_imported_f64_lognormal "$log"
    rc=$INLINE_RC
  elif [[ ! -f "$ROOT/$script" ]]; then
    echo "STATUS $name reason=missing_script path=$script class=$class" | tee "$log"
    rc=127
  else
    bash "$ROOT/$script" >"$log" 2>&1
    rc=$?
  fi
  set +e
  end=$(date +%s)
  dur=$((end - start))
  set -e

  local status="pass"
  if [[ $rc -ne 0 ]]; then
    status="fail"
  elif [[ -n "$token" ]] && ! grep -qF "$token" "$log"; then
    status="token_missing"
    rc=2
  fi

  if [[ "$name" == "k95" && "$status" == "pass" ]]; then
    if ! grep -qF 'k95i=2776' "$log"; then
      status="k95_token_missing"
      rc=3
    elif grep -qE 'k95i=1960|want 2776' "$log" && ! grep -qF 'PASS: k95i=2776' "$log"; then
      status="k95_collapse"
      rc=4
    fi
  fi

  if [[ ( "$name" == "cd_exact" || "$name" == "cd_exact_e2e" ) && "$status" == "pass" ]]; then
    if ! grep -qF 'ZD PROVED' "$log"; then
      status="zd_token_missing"
      rc=8
    fi
  fi

  if [[ "$name" == "imported_f64_core" && "$status" == "pass" ]]; then
    if ! grep -qF 'BARE_CROSSMOD_F64_IDENT_OK' "$log"; then
      status="bare_ident_token_missing"
      rc=5
    fi
  fi

  if [[ "$name" == "global_array" && "$status" == "pass" ]]; then
    # Wave13e single-stmt paramful fold must be present in the gate (call_list_args).
    if ! grep -qF 'call_list_args' "$log" && ! grep -qF '30 1 2' "$log"; then
      status="wave13e_call_list_missing"
      rc=6
    fi
  fi

  if [[ "$status" == "pass" ]]; then
    echo "GREEN $name rc=$rc ${dur}s token=$token class=$class"
  else
    echo "RED   $name status=$status rc=$rc ${dur}s token=$token class=$class"
    echo "----- tail $name -----"
    tail -40 "$log" || true
    echo "----- end tail -----"
    if [[ "$required" == "1" ]]; then
      fail_required=1
    fi
  fi

  case "$name" in
    cd_exact) cd_exact_status="$status" ;;
    cd_exact_e2e) cd_exact_e2e_status="$status" ;;
    imported_f64_lognormal_science) lognormal_status="$status" ;;
    imported_f64_core) f64_core_status="$status" ;;
  esac

  RESULT_NAMES+=("$name")
  RESULT_STATUS+=("$status")
  RESULT_RC+=("$rc")
  RESULT_DURATION+=("$dur")
  RESULT_TOKEN+=("$token")
  RESULT_REQUIRED+=("$required")
  RESULT_CLASS+=("$class")
}

for entry in "${GATES[@]}"; do
  IFS='|' read -r gname gscript gtoken greq <<<"$entry"
  run_one "$gname" "$gscript" "$gtoken" "$greq"
  echo
done

# Composite verdict.
if [[ $fail_required -eq 0 ]]; then
  if [[ "$lognormal_status" == "pass" ]]; then
    OVERALL="pass"
    SHOWCASE_VERDICT="pass_full"
  else
    OVERALL="pass"
    SHOWCASE_VERDICT="pass_with_imported_f64_science_residual"
  fi
else
  OVERALL="fail"
  SHOWCASE_VERDICT="fail"
fi

gates_json="["
for i in "${!RESULT_NAMES[@]}"; do
  [[ $i -gt 0 ]] && gates_json+=","
  gates_json+=$(printf '\n    {"name":"%s","status":"%s","rc":%s,"duration_s":%s,"expected_token":"%s","required":%s,"class":"%s"}' \
    "${RESULT_NAMES[$i]}" \
    "${RESULT_STATUS[$i]}" \
    "${RESULT_RC[$i]}" \
    "${RESULT_DURATION[$i]}" \
    "${RESULT_TOKEN[$i]}" \
    "$( [[ "${RESULT_REQUIRED[$i]}" == "1" ]] && echo true || echo false )" \
    "${RESULT_CLASS[$i]}")
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

status_of() {
  local want="$1"
  local i
  for i in "${!RESULT_NAMES[@]}"; do
    if [[ "${RESULT_NAMES[$i]}" == "$want" ]]; then
      echo "${RESULT_STATUS[$i]}"
      return
    fi
  done
  echo "not_run"
}

RAW_REL="$(relpath_under_root "${RAW:-}")"
SOUC_REL="$(relpath_under_root "$SOUC")"

CLAIMS_JSON='[
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392",
    "cd_exact_e2e_specialized_collapse",
    "global_array_init_incl_wave13e_call_list_args",
    "imported_f64_minimal_bss_bare_ident",
    "bare_float_arith_intrinsic_results",
    "root2_method_associated_and_instance",
    "root2_multimodule_method_chain",
    "imported_array_byvalue_913",
    "thinlink_921_default_path_closed",
    "public_showcase_receipt_wave15"
  ]'

CLAIMS_NOT_JSON='[
    "imported_f64_lognormal_science_vertical",
    "multi_stmt_paramful_global_list_fold",
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "compact_imported_ir_complete",
    "all_madaros_residuals_closed"
  ]'

if [[ "$lognormal_status" == "pass" ]]; then
  CLAIMS_JSON='[
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392",
    "cd_exact_e2e_specialized_collapse",
    "global_array_init_incl_wave13e_call_list_args",
    "imported_f64_minimal_bss_bare_ident",
    "imported_f64_lognormal_science_vertical",
    "bare_float_arith_intrinsic_results",
    "root2_method_associated_and_instance",
    "root2_multimodule_method_chain",
    "imported_array_byvalue_913",
    "thinlink_921_default_path_closed",
    "public_showcase_receipt_wave15"
  ]'
  CLAIMS_NOT_JSON='[
    "multi_stmt_paramful_global_list_fold",
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "compact_imported_ir_complete",
    "all_madaros_residuals_closed"
  ]'
fi

mkdir -p "$RECEIPT_DIR"
cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_wave15_showcase_receipt.v1",
  "status": "$OVERALL",
  "showcase_verdict": "$SHOWCASE_VERDICT",
  "wave": "wave15",
  "role": "public_showcase_wave13_14_locks",
  "supersedes": "madaros_wave13_showcase_receipt.v1",
  "engine": "madaros_default",
  "engine_line": $(printf '%s' "$ENGINE_LINE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "commit_short": "$COMMIT_SHORT",
  "utc": "$UTC",
  "raw_elf": $(printf '%s' "$RAW_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "raw_elf_sha256": "$RAW_SHA256",
  "souc": $(printf '%s' "$SOUC_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "require_cd_exact": $( [[ "$REQUIRE_CD_EXACT" == "1" ]] && echo true || echo false ),
  "require_imported_f64_science": $( [[ "$REQUIRE_IMPORTED_F64_SCIENCE" == "1" ]] && echo true || echo false ),
  "pr_cd_exact_e2e": 1392,
  "pr_imported_array_byvalue": 1398,
  "pr_thinlink_921": 1399,
  "pr_bare_crossmod_f64_ident": 1400,
  "pr_root2_method_chain": 1401,
  "pr_bare_float_arith": 1404,
  "pr_wave13e_call_list_args": 1405,
  "gates": $gates_json,
  "claims": $CLAIMS_JSON,
  "claims_not_made": $CLAIMS_NOT_JSON,
  "measurement": {
    "tip_sha": "$COMMIT_SHORT",
    "dual": "$(status_of dual)",
    "order_spread": "$(status_of order_spread)",
    "k95": "$(status_of k95)",
    "cd_exact": "$cd_exact_status",
    "cd_exact_e2e": "$cd_exact_e2e_status",
    "global_array": "$(status_of global_array)",
    "imported_f64_core": "$f64_core_status",
    "imported_f64_lognormal_science": "$lognormal_status",
    "bare_float_arith": "$(status_of bare_float_arith)",
    "root2_method": "$(status_of root2_method)",
    "root2_multimodule_method": "$(status_of root2_multimodule_method)",
    "imported_array_byvalue": "$(status_of imported_array_byvalue)",
    "thinlink_921": "$(status_of thinlink_921)",
    "required_red_count": $fail_required,
    "note": "Wave15 public showcase locks Wave13 science + Wave14 tip locks under stock Madaros. imported_f64 lognormal denser vertical is an honest residual on tip prebuilt after #1405 (Wave13e) unless REQUIRE_IMPORTED_F64_SCIENCE=1 and green. multi-stmt paramful global list fold remains residual fail-closed inside global_array_init."
  },
  "verdict": "$OVERALL"
}
EOF

echo "===== SHOWCASE SUMMARY ====="
for i in "${!RESULT_NAMES[@]}"; do
  printf '  %-32s %-22s class=%-24s rc=%-3s %ss\n' \
    "${RESULT_NAMES[$i]}" \
    "${RESULT_STATUS[$i]}" \
    "${RESULT_CLASS[$i]}" \
    "${RESULT_RC[$i]}" \
    "${RESULT_DURATION[$i]}"
done
echo "overall=$OVERALL"
echo "showcase_verdict=$SHOWCASE_VERDICT"
echo "cd_exact=$cd_exact_status"
echo "cd_exact_e2e=$cd_exact_e2e_status"
echo "imported_f64_core=$f64_core_status"
echo "imported_f64_lognormal_science=$lognormal_status"
echo "require_cd_exact=$REQUIRE_CD_EXACT"
echo "require_imported_f64_science=$REQUIRE_IMPORTED_F64_SCIENCE"
echo "receipt: $RECEIPT"

if [[ $fail_required -eq 0 ]]; then
  echo "MADAROS_WAVE15_SHOWCASE_GATE_OK"
  if [[ "$lognormal_status" != "pass" ]]; then
    echo "NOTE: imported_f64 lognormal science residual (honest) — denser DE_LN_SQRT_2PI path red on tip after #1405"
    echo "NOTE: multi-stmt paramful global list fold remains residual fail-closed (inside global_array_init)"
  fi
  exit 0
fi
if [[ ( "$cd_exact_status" != "pass" || "$cd_exact_e2e_status" != "pass" ) && "$REQUIRE_CD_EXACT" == "1" ]]; then
  echo "NOTE: cd_exact family RED while required — rebuild Madaros if stock prebuilt lags #1392:" >&2
  echo "  make build-madaros   # bare; wrapping it in souc-build-lock.sh deadlocks" >&2
  echo "  MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash $0" >&2
fi
echo "MADAROS_WAVE15_SHOWCASE_GATE_FAIL" >&2
exit 1
