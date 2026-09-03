#!/usr/bin/env bash
# scripts/dev/madaros_wave10_tip_green_gate.sh
#
# Wave10 tip-green regression lock for origin/main Madaros science/compiler
# residuals that landed through Wave9:
#
#   1. dual            — scripts/madaros_dual_import_gate.sh
#   2. order_spread    — scripts/madaros_order_spread_native_gate.sh
#   3. knowledge_method— scripts/madaros_knowledge_method_residual_gate.sh
#   4. global_array    — scripts/dev/madaros_global_array_init_gate.sh
#   5. named_path      — scripts/dev/named_path_import_print_f64_gate.sh
#                        (path-form import + multi-module -O full peels)
#   6. unsplit_oct     — scripts/madaros_unsplit_oct_mul_gate.sh
#
# Exit 0 only when every sub-gate exits 0. Writes a machine-readable receipt:
#   artifacts/compiler/madaros_wave10_tip_green_receipt.v1.json
#
# Does not rebuild Madaros. Does not pin lean_single. Uses default bin/souc.
# Soft stack raised for multi-module dual/knowledge lowers.
#
# Usage:
#   bash scripts/dev/madaros_wave10_tip_green_gate.sh
#   SOUC=./bin/souc bash scripts/dev/madaros_wave10_tip_green_gate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
unset SOUNIO_SOUC_ENGINE || true
# Multi-module dual/knowledge lowers need a large stack; prefer unlimited.
ulimit -s unlimited 2>/dev/null || ulimit -s 131072 2>/dev/null || true

SOUC="${SOUC:-$ROOT/bin/souc}"
export SOUC

RECEIPT_DIR="$ROOT/artifacts/compiler"
RECEIPT="$RECEIPT_DIR/madaros_wave10_tip_green_receipt.v1.json"
LOG_DIR="$(mktemp -d /tmp/madaros_wave10_tip_green.XXXXXX)"
trap 'rm -rf "$LOG_DIR"' EXIT

COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
COMMIT_SHORT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Prefer raw Madaros ELF for sha identity (wrapper scripts are not ELFs).
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

echo "== madaros_wave10_tip_green_gate =="
echo "git_sha=$COMMIT_SHORT"
echo "engine=$ENGINE_LINE"
echo "raw_elf=${RAW:-none}"
echo "raw_elf_sha256=$RAW_SHA256"
echo "souc=$SOUC"
echo "log_dir=$LOG_DIR"
echo

# name|script|expected_token (optional; empty = any rc==0)
GATES=(
  "dual|scripts/madaros_dual_import_gate.sh|MADAROS_DUAL_IMPORT_GATE_OK"
  "order_spread|scripts/madaros_order_spread_native_gate.sh|MADAROS_ORDER_SPREAD_NATIVE_GATE_OK"
  "knowledge_method|scripts/madaros_knowledge_method_residual_gate.sh|MADAROS_KNOWLEDGE_METHOD_RESIDUAL_GATE_OK"
  "global_array|scripts/dev/madaros_global_array_init_gate.sh|GLOBAL_ARRAY_INIT_GATE_OK"
  "named_path|scripts/dev/named_path_import_print_f64_gate.sh|NAMED_PATH_IMPORT_GATE_PASS"
  "unsplit_oct|scripts/madaros_unsplit_oct_mul_gate.sh|MADAROS_UNSPLIT_OCT_MUL_GATE_OK"
)

declare -a RESULT_NAMES=()
declare -a RESULT_STATUS=()
declare -a RESULT_RC=()
declare -a RESULT_DURATION=()
declare -a RESULT_TOKEN=()

fail=0
idx=0
total=${#GATES[@]}

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
    # Force non-zero classification even if sub-gate exited 0 without sentinel.
    rc=2
  fi

  if [[ "$status" == "pass" ]]; then
    echo "GREEN $name rc=$rc ${dur}s token=$token"
  else
    echo "RED   $name status=$status rc=$rc ${dur}s token=$token"
    echo "----- tail $name -----"
    tail -40 "$log" || true
    echo "----- end tail -----"
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

# Build gates JSON array without requiring jq.
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

# Prefer repo-relative paths in the receipt so committed snapshots are portable.
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

mkdir -p "$RECEIPT_DIR"
cat >"$RECEIPT" <<EOF
{
  "schema": "madaros_wave10_tip_green_receipt.v1",
  "status": "$OVERALL",
  "wave": "wave10",
  "role": "tip_green_regression_lock",
  "engine": "madaros_default",
  "engine_line": $(printf '%s' "$ENGINE_LINE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "lean_single_pin": false,
  "commit": "$COMMIT",
  "commit_short": "$COMMIT_SHORT",
  "utc": "$UTC",
  "raw_elf": $(printf '%s' "$RAW_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "raw_elf_sha256": "$RAW_SHA256",
  "souc": $(printf '%s' "$SOUC_REL" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read().rstrip("\n")))'),
  "gates": $gates_json,
  "claims": [
    "dual_gum_knowledge_import_native",
    "order_spread4_cpc_n4_native",
    "knowledge_method_form_parity",
    "global_array_init_i64_f64_i8",
    "named_path_import_print_f64_default_and_opt",
    "unsplit_oct_mul_no_reentry"
  ],
  "claims_not_made": [
    "all_stdlib_dual_pairs",
    "gum_k95_f64_i64_cast_fixed",
    "language_knowledge_t_generic_import",
    "multi_module_irmodule_memory_wall_closed",
    "cd_exact_generic_i64_elf",
    "full_root2_census_closed"
  ],
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
echo "receipt: $RECEIPT"

if [[ $fail -eq 0 ]]; then
  echo "MADAROS_WAVE10_TIP_GREEN_GATE_OK"
  exit 0
fi
echo "MADAROS_WAVE10_TIP_GREEN_GATE_FAIL" >&2
exit 1
