#!/usr/bin/env bash
# scripts/dev/madaros_wave13_showcase_gate.sh
#
# Wave13 public-facing Madaros FULL-GREEN SHOWCASE orchestrator.
# Supersedes Wave12 residual packaging: cd_exact is REQUIRED (not honest residual).
#
# Closed by PR #1392 (cd_exact_generic_i64 e2e under default Madaros — ZD PROVED).
#
#   REQUIRED (fail the showcase if any red):
#     1. wave12_tip_green — scripts/dev/madaros_wave12_tip_green_gate.sh
#     2. dual             — scripts/madaros_dual_import_gate.sh
#     3. order_spread     — scripts/madaros_order_spread_native_gate.sh
#     4. k95              — scripts/epistemic_trust_gate.sh
#                          (Section A + finite-dof gum k95i=2776)
#     5. cd_exact         — scripts/dev/madaros_cd_exact_generic_i64_gate.sh
#                          (REQUIRED by default after #1392)
#     6. cd_exact_e2e     — scripts/madaros_cd_exact_e2e_gate.sh
#                          (raw-ELF path with specialized_collapse; REQUIRED)
#
# REQUIRE_CD_EXACT defaults to 1. Set REQUIRE_CD_EXACT=0 only for emergency
# residual-only demos (not the public claim surface).
#
# Prebuilt lag note: #1392 ships bin/madaros-linux-x86_64. If a checkout still
# has an older stock ELF and cd_exact is RED, rebuild before claiming green:
#   make build-madaros   # bare: it locks internally; wrapping it in
#                        # souc-build-lock.sh deadlocks (see CLAUDE.md §4)
#   MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash "$0"
#
# Exit 0 when every REQUIRED sub-gate is green. Writes:
#   artifacts/compiler/madaros_wave13_showcase_receipt.v1.json
#
# Does not rebuild Madaros. Does not pin lean_single. Uses default bin/souc.
#
# Usage:
#   bash scripts/dev/madaros_wave13_showcase_gate.sh
#   REQUIRE_CD_EXACT=0 bash scripts/dev/madaros_wave13_showcase_gate.sh  # legacy residual-only
#   SOUC=./bin/souc bash scripts/dev/madaros_wave13_showcase_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUC
# Wave13: cd_exact REQUIRED by default (PR #1392).
REQUIRE_CD_EXACT="${REQUIRE_CD_EXACT:-1}"

RECEIPT_DIR="$ROOT/artifacts/compiler"
RECEIPT="$RECEIPT_DIR/madaros_wave13_showcase_receipt.v1.json"
LOG_DIR="$(mktemp -d /tmp/madaros_wave13_showcase.XXXXXX)"
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

echo "== madaros_wave13_showcase_gate =="
echo "git_sha=$COMMIT_SHORT"
echo "engine=$ENGINE_LINE"
echo "raw_elf=${RAW:-none}"
echo "raw_elf_sha256=$RAW_SHA256"
echo "souc=$SOUC"
echo "require_cd_exact=$REQUIRE_CD_EXACT"
echo "pr_cd_exact_e2e=1392"
echo "log_dir=$LOG_DIR"
echo

# name|script|expected_token|required(1/0)
GATES=(
  "wave12_tip_green|scripts/dev/madaros_wave12_tip_green_gate.sh|MADAROS_WAVE12_TIP_GREEN_GATE_OK|1"
  "dual|scripts/madaros_dual_import_gate.sh|MADAROS_DUAL_IMPORT_GATE_OK|1"
  "order_spread|scripts/madaros_order_spread_native_gate.sh|MADAROS_ORDER_SPREAD_NATIVE_GATE_OK|1"
  "k95|scripts/epistemic_trust_gate.sh|EPISTEMIC_TRUST_GATE_OK|1"
  "cd_exact|scripts/dev/madaros_cd_exact_generic_i64_gate.sh|MADAROS_CD_EXACT_GENERIC_I64_GATE_OK|1"
  "cd_exact_e2e|scripts/madaros_cd_exact_e2e_gate.sh|MADAROS_CD_EXACT_E2E_GATE_OK|1"
)

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
idx=0
total=${#GATES[@]}

run_one() {
  local name="$1" script="$2" token="$3" required="$4"
  local log="$LOG_DIR/${name}.log"
  local start end dur rc
  local class="required"
  [[ "$required" == "1" ]] || class="honest_probe"

  # cd_exact family: default required; REQUIRE_CD_EXACT=0 demotes both probes.
  if [[ "$name" == "cd_exact" || "$name" == "cd_exact_e2e" ]]; then
    if [[ "$REQUIRE_CD_EXACT" == "1" ]]; then
      required="1"
      class="required"
    else
      required="0"
      class="honest_probe_legacy"
    fi
  fi

  idx=$((idx + 1))
  echo "---------- [$idx/$total] $name (class=$class) ----------"
  echo "script=$script"

  if [[ ! -f "$ROOT/$script" ]]; then
    echo "STATUS $name reason=missing_script path=$script class=$class"
    RESULT_NAMES+=("$name")
    RESULT_STATUS+=("missing_script")
    RESULT_RC+=(127)
    RESULT_DURATION+=(0)
    RESULT_TOKEN+=("$token")
    RESULT_REQUIRED+=("$required")
    RESULT_CLASS+=("$class")
    if [[ "$required" == "1" ]]; then
      fail_required=1
    fi
    if [[ "$name" == "cd_exact" ]]; then
      cd_exact_status="missing_script"
    fi
    if [[ "$name" == "cd_exact_e2e" ]]; then
      cd_exact_e2e_status="missing_script"
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

  if [[ "$name" == "cd_exact" ]]; then
    cd_exact_status="$status"
  fi
  if [[ "$name" == "cd_exact_e2e" ]]; then
    cd_exact_e2e_status="$status"
  fi

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

# Composite verdict: full green only when cd_exact family passes under REQUIRE.
if [[ $fail_required -eq 0 ]]; then
  if [[ "$cd_exact_status" == "pass" && "$cd_exact_e2e_status" == "pass" ]]; then
    OVERALL="pass"
    SHOWCASE_VERDICT="pass_full"
  elif [[ "$REQUIRE_CD_EXACT" != "1" ]]; then
    OVERALL="pass"
    SHOWCASE_VERDICT="pass_with_cd_exact_residual"
  else
    OVERALL="pass"
    SHOWCASE_VERDICT="pass_full"
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

RAW_REL="$(relpath_under_root "${RAW:-}")"
SOUC_REL="$(relpath_under_root "$SOUC")"

CLAIMS_JSON='[
    "wave12_tip_green_superseding_wave11",
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "public_showcase_receipt_wave13"
  ]'
CLAIMS_NOT_JSON='[
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "all_madaros_residuals_closed"
  ]'
if [[ "$cd_exact_status" == "pass" && "$cd_exact_e2e_status" == "pass" ]]; then
  CLAIMS_JSON='[
    "wave12_tip_green_superseding_wave11",
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "gum_k95_f64_i64_cast_fixed",
    "epistemic_trust_section_a_native",
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392",
    "cd_exact_e2e_specialized_collapse",
    "public_showcase_receipt_wave13"
  ]'
else
  CLAIMS_NOT_JSON='[
    "cd_exact_generic_i64_elf",
    "cd_exact_zd_proved_pr1392",
    "cd_exact_e2e_specialized_collapse",
    "all_stdlib_dual_pairs",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "full_root2_census_closed",
    "f64_param_bitcast_all_call_shapes",
    "linalg_full_native_parity",
    "all_madaros_residuals_closed"
  ]'
fi

mkdir -p "$RECEIPT_DIR"
cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_wave13_showcase_receipt.v1",
  "status": "$OVERALL",
  "showcase_verdict": "$SHOWCASE_VERDICT",
  "wave": "wave13",
  "role": "public_full_green_showcase",
  "supersedes": "madaros_wave12_showcase_receipt.v1",
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
  "pr_cd_exact_e2e": 1392,
  "gates": $gates_json,
  "claims": $CLAIMS_JSON,
  "claims_not_made": $CLAIMS_NOT_JSON,
  "measurement": {
    "tip_sha": "$COMMIT_SHORT",
    "wave12_tip_green": "$(for i in "${!RESULT_NAMES[@]}"; do [[ "${RESULT_NAMES[$i]}" == "wave12_tip_green" ]] && echo "${RESULT_STATUS[$i]}"; done)",
    "dual": "$(for i in "${!RESULT_NAMES[@]}"; do [[ "${RESULT_NAMES[$i]}" == "dual" ]] && echo "${RESULT_STATUS[$i]}"; done)",
    "order_spread": "$(for i in "${!RESULT_NAMES[@]}"; do [[ "${RESULT_NAMES[$i]}" == "order_spread" ]] && echo "${RESULT_STATUS[$i]}"; done)",
    "k95": "$(for i in "${!RESULT_NAMES[@]}"; do [[ "${RESULT_NAMES[$i]}" == "k95" ]] && echo "${RESULT_STATUS[$i]}"; done)",
    "cd_exact": "$cd_exact_status",
    "cd_exact_e2e": "$cd_exact_e2e_status",
    "required_red_count": $fail_required,
    "note": "Wave13 public showcase requires cd_exact green (PR #1392 ZD PROVED). Not residual-only. Rebuild Madaros if stock prebuilt lags."
  },
  "verdict": "$OVERALL"
}
EOF

echo "===== SHOWCASE SUMMARY ====="
for i in "${!RESULT_NAMES[@]}"; do
  printf '  %-18s %-22s class=%-16s rc=%-3s %ss\n' \
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
echo "require_cd_exact=$REQUIRE_CD_EXACT"
echo "receipt: $RECEIPT"

if [[ $fail_required -eq 0 ]]; then
  echo "MADAROS_WAVE13_SHOWCASE_GATE_OK"
  if [[ "$cd_exact_status" == "pass" && "$cd_exact_e2e_status" == "pass" ]]; then
    echo "NOTE: cd_exact + cd_exact_e2e GREEN (ZD PROVED) — public claim after PR #1392"
  fi
  exit 0
fi
if [[ ( "$cd_exact_status" != "pass" || "$cd_exact_e2e_status" != "pass" ) && "$REQUIRE_CD_EXACT" == "1" ]]; then
  echo "NOTE: cd_exact family RED while required — rebuild Madaros if stock prebuilt lags #1392:" >&2
  echo "  make build-madaros   # bare; wrapping it in souc-build-lock.sh deadlocks" >&2
  echo "  MADAROS_RAW_BIN=artifacts/self-hosted/madaros bash $0" >&2
fi
echo "MADAROS_WAVE13_SHOWCASE_GATE_FAIL" >&2
exit 1
