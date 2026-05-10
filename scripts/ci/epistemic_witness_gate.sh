#!/usr/bin/env bash
# epistemic_witness_gate.sh -- non-trivial witness for the Epistemic Monotonicity Theorem.
#
# Compiles examples/pbpk_simple.sio through native_compile_driver and asserts that
# u_c_scaled > 0 in the output binary's .sounio.epistemic (SIEP) section.
#
# pbpk_simple.sio contains 2 Knowledge<f64> type occurrences, so u_c_scaled must be >= 2.
# This proves the compiler correctly measures external epistemic density -- the gate is
# not a tautology (u_c=0 for the compiler itself; u_c>=2 for an epistemic program).
#
# Theorem instantiation:
#   u_c_scaled(native_compile_driver compiled with pbpk_simple.sio as input) >= 2
#
# Gate status: pass when SIEP is present and u_c_scaled >= 2.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ "$(uname -s 2>/dev/null || echo unknown)" != "Linux" ]]; then
  echo "[epistemic-witness] SKIP: Linux-only gate" >&2
  exit 0
fi
case "$(uname -m 2>/dev/null || echo unknown)" in
  x86_64|amd64) ;;
  *)
    echo "[epistemic-witness] SKIP: x86-64 Linux-only gate" >&2
    exit 0
    ;;
esac

source "$ROOT_DIR/scripts/lib/resolve_souc.sh"

DRIVER_SRC="self-hosted/compiler/native_compile_driver.sio"
WITNESS_SRC="examples/epistemic_witness_minimal.sio"
OUT_DIR="$(mktemp -d /tmp/epistemic_witness_XXXXXX)"
OUT_ELF="$OUT_DIR/epistemic_witness.elf"
OUT_JSON="artifacts/omega/epistemic_witness_gate.v1.json"
EXPECTED_MIN_UC=3

trap 'rm -rf "$OUT_DIR"' EXIT

parse_siep() {
  "$ROOT_DIR/bin/kretikos" siep-probe "$1"
}

emit_json() {
  local status="$1" reason="$2" u_c="$3" instr="$4" siep_present="$5"
  local ts u_c_ge_expected=false json_out
  ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  [[ "$u_c" -ge "$EXPECTED_MIN_UC" ]] && u_c_ge_expected=true
  json_out="$("$ROOT_DIR/bin/kretikos" json-emit \
    --string "schema=sounio.epistemic-witness-gate.v1" \
    --string "generated_at_utc=$ts" \
    --string "witness_program=$WITNESS_SRC" \
    --int    "expected_min_u_c_scaled=$EXPECTED_MIN_UC" \
    --string "status=$status" \
    --string "reason=$reason" \
    --bool   "checks.siep_section_present=$siep_present" \
    --bool   "checks.u_c_scaled_ge_expected=$u_c_ge_expected" \
    --int    "result.instr_count=$instr" \
    --int    "result.u_c_scaled=$u_c" \
    --string "theorem=u_c_scaled >= 3 for epistemic_witness_minimal.sio compiled through native_compile_driver" \
    --string "note=u_c_scaled = 0 for the compiler itself (no Knowledge<T>); non-zero here proves external epistemic density is measured correctly")"
  printf '%s\n' "$json_out" > "$OUT_JSON"
  printf '%s\n' "$json_out"
}

printf '[epistemic-witness] souc=%s\n' "$SOUC_BIN"
printf '[epistemic-witness] witness=%s\n' "$WITNESS_SRC"
printf '[epistemic-witness] output=%s\n' "$OUT_ELF"

# ── Compile the witness program ───────────────────────────────────────────────
printf '[epistemic-witness] compiling %s...\n' "$WITNESS_SRC"
if ! "$SOUC_BIN" run "$DRIVER_SRC" -- "$WITNESS_SRC" -o "$OUT_ELF" 2>&1; then
  echo "[epistemic-witness] FAIL: native_compile_driver failed" >&2
  emit_json "fail" "compile_failed" "0" "0" "false"
  exit 1
fi

if [[ ! -f "$OUT_ELF" ]]; then
  echo "[epistemic-witness] FAIL: output ELF not produced" >&2
  emit_json "fail" "output_elf_missing" "0" "0" "false"
  exit 1
fi

printf '[epistemic-witness] output ELF: %s (%d bytes)\n' "$OUT_ELF" "$(wc -c < "$OUT_ELF")"

# ── Parse SIEP section ────────────────────────────────────────────────────────
EP="$(parse_siep "$OUT_ELF")"
EP_STATUS="$(echo "$EP" | cut -d: -f1)"
EP_INSTR="$(echo "$EP" | cut -d: -f2)"
EP_UC="$(echo "$EP" | cut -d: -f3)"

printf '[epistemic-witness] siep=%s instr=%s u_c_scaled=%s\n' "$EP_STATUS" "$EP_INSTR" "$EP_UC"

if [[ "$EP_STATUS" != "present" ]]; then
  echo "[epistemic-witness] FAIL: SIEP section absent from output ELF" >&2
  emit_json "fail" "siep_section_absent" "0" "0" "false"
  exit 1
fi

if [[ "$EP_UC" -lt "$EXPECTED_MIN_UC" ]]; then
  printf '[epistemic-witness] FAIL: u_c_scaled=%s < expected_min=%s\n' "$EP_UC" "$EXPECTED_MIN_UC" >&2
  emit_json "fail" "u_c_scaled_below_threshold" "$EP_UC" "$EP_INSTR" "true"
  exit 1
fi

# ── PASS ─────────────────────────────────────────────────────────────────────
printf '[epistemic-witness] PASS: u_c_scaled=%s >= %s (theorem non-trivially witnessed)\n' \
       "$EP_UC" "$EXPECTED_MIN_UC"
emit_json "pass" "epistemic_witness_verified" "$EP_UC" "$EP_INSTR" "true"
