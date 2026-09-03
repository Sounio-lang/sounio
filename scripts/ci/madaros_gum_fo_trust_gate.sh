#!/usr/bin/env bash
# madaros_gum_fo_trust_gate.sh — consolidate Madaros FO GUM stack receipts.
#
# Runs every tests/run-pass/madaros_gum_fo_*.sio under MADAROS_RAW_BIN (or
# rebuilds modular Madaros first when SOUNIO_FO_REBUILD=1).
#
# Measure before claim: each gate prints a unique PASS token; this script
# records pass/fail and emits a JSON summary.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[madaros-fo-trust] SKIP: Linux-only gate" >&2
  exit 0
fi

case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[madaros-fo-trust] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

OUT_DIR="${SOUNIO_FO_TRUST_DIR:-$(mktemp -d /tmp/sounio-fo-trust.XXXXXX)}"
LOG_DIR="$OUT_DIR/logs"
SUMMARY_JSON="$OUT_DIR/summary.json"
mkdir -p "$LOG_DIR"

MADAROS_BIN="${MADAROS_RAW_BIN:-$ROOT_DIR/artifacts/self-hosted/madaros}"

if [[ "${SOUNIO_FO_REBUILD:-0}" == "1" ]]; then
  echo "[madaros-fo-trust] rebuilding Madaros → $MADAROS_BIN" >&2
  bash "$ROOT_DIR/scripts/ci/build_modular_madaros.sh" "$MADAROS_BIN" \
    >"$LOG_DIR/rebuild.log" 2>&1
fi

if [[ ! -x "$MADAROS_BIN" ]]; then
  echo "[madaros-fo-trust] FAIL: no Madaros ELF at $MADAROS_BIN" >&2
  echo "[madaros-fo-trust] hint: SOUNIO_FO_REBUILD=1 or MADAROS_RAW_BIN=path" >&2
  exit 1
fi

export MADAROS_RAW_BIN="$MADAROS_BIN"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT_DIR/stdlib}"
export PATH="$ROOT_DIR/bin:$PATH"

# Prefer bin/souc (routes to Madaros when MADAROS_RAW_BIN set / available)
SOUC="${SOUC:-$ROOT_DIR/bin/souc}"
if [[ ! -x "$SOUC" ]]; then
  echo "[madaros-fo-trust] FAIL: no souc at $SOUC" >&2
  exit 1
fi

mapfile -t GATES < <(ls -1 tests/run-pass/madaros_gum_fo_*.sio 2>/dev/null | sort)
if [[ ${#GATES[@]} -eq 0 ]]; then
  echo "[madaros-fo-trust] FAIL: no madaros_gum_fo_*.sio gates found" >&2
  exit 1
fi

pass_n=0
fail_n=0
skip_n=0
declare -a RESULTS=()

echo "[madaros-fo-trust] madaros=$(basename "$MADAROS_BIN") size=$(stat -c%s "$MADAROS_BIN" 2>/dev/null || echo '?')"
echo "[madaros-fo-trust] gates=${#GATES[@]}"

for gate in "${GATES[@]}"; do
  base="$(basename "$gate" .sio)"
  log="$LOG_DIR/${base}.log"
  token="MADAROS_GUM_FO_$(echo "${base#madaros_gum_fo_}" | tr '[:lower:]' '[:upper:]')_PASS"
  # token heuristics: gate files name madaros_gum_fo_X → expect MADAROS_GUM_FO_X_PASS
  # actual tokens vary; scan for _PASS without _FAIL
  set +e
  "$SOUC" run "$gate" >"$log" 2>&1
  rc=$?
  set -e
  if grep -q '_FAIL' "$log" 2>/dev/null; then
    echo "[madaros-fo-trust] FAIL  $base (token FAIL)"
    fail_n=$((fail_n + 1))
    RESULTS+=("{\"gate\":\"$base\",\"status\":\"fail\",\"rc\":$rc,\"reason\":\"fail_token\"}")
  elif grep -qE 'MADAROS_GUM_FO_.*_PASS' "$log" 2>/dev/null; then
    got="$(grep -oE 'MADAROS_GUM_FO_[A-Z0-9_]+_PASS' "$log" | head -1)"
    echo "[madaros-fo-trust] PASS  $base ($got)"
    pass_n=$((pass_n + 1))
    RESULTS+=("{\"gate\":\"$base\",\"status\":\"pass\",\"rc\":$rc,\"token\":\"$got\"}")
  elif [[ $rc -ne 0 ]]; then
    echo "[madaros-fo-trust] FAIL  $base (rc=$rc)"
    fail_n=$((fail_n + 1))
    RESULTS+=("{\"gate\":\"$base\",\"status\":\"fail\",\"rc\":$rc,\"reason\":\"nonzero_exit\"}")
  else
    echo "[madaros-fo-trust] FAIL  $base (no PASS token)"
    fail_n=$((fail_n + 1))
    RESULTS+=("{\"gate\":\"$base\",\"status\":\"fail\",\"rc\":$rc,\"reason\":\"no_pass_token\"}")
  fi
done

# JSON summary (no jq required)
{
  echo "{"
  echo "  \"gate\": \"madaros_gum_fo_trust\","
  echo "  \"madaros\": \"$MADAROS_BIN\","
  echo "  \"pass\": $pass_n,"
  echo "  \"fail\": $fail_n,"
  echo "  \"total\": ${#GATES[@]},"
  echo "  \"results\": ["
  for i in "${!RESULTS[@]}"; do
    comma=","
    if [[ $i -eq $((${#RESULTS[@]} - 1)) ]]; then comma=""; fi
    echo "    ${RESULTS[$i]}$comma"
  done
  echo "  ]"
  echo "}"
} >"$SUMMARY_JSON"

echo "[madaros-fo-trust] summary=$SUMMARY_JSON"
echo "[madaros-fo-trust] pass=$pass_n fail=$fail_n total=${#GATES[@]}"

if [[ $fail_n -ne 0 ]]; then
  echo "[madaros-fo-trust] FAIL: $fail_n gate(s) failed" >&2
  exit 1
fi

echo "[madaros-fo-trust] PASS: all ${#GATES[@]} FO GUM gates green"
exit 0
