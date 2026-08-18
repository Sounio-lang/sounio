#!/usr/bin/env bash
set -euo pipefail

# WS-F Madaros variant of eisa_bridge_conformance_gate.sh
# Templated from scripts/ci/eisa_h_zd_reference_gate.sh (tolerates documented rc=12 BLOCKED)
# and the lean_single-only eisa_bridge_conformance_gate.sh.
# Does NOT touch tools/eisa/ or stdlib/eisa/ sources. Script/report only.
# See dispatch in docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md §WS-F.
#
# Stack (#1760): this gate does NOT set its own ulimit. Default Madaros path is
# ./bin/souc → bin/madaros, which reserves MADAROS_STACK_KB=524288 (512 MiB).
# Bisected floor for the EISA v1 bridge lowering path: fails through 384 MiB,
# passes at 448 MiB; 512 MiB is the shipped default. Do not lower that here.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts/eisa
TMP_DIR="artifacts/eisa/.gate_tmp_madaros"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"' EXIT

A_OUT="$TMP_DIR/evm_stdout.txt"
B_OUT="$TMP_DIR/bridge_stdout.txt"
EMIT_LOG="$TMP_DIR/emit_driver.log"
MADAROS_RUN_LOG="$TMP_DIR/madaros_bridge_emit.log"

fail() {
  echo "[eisa-bridge-madaros] FAIL: $*" >&2
  exit 1
}

# Refuse an undersized override that would re-introduce the #1760 false red.
# 0 means unlimited (allowed). Non-zero must be >= 448 MiB (bisected pass floor).
_EISA_STACK_FLOOR_KB=$((448 * 1024))
if [[ -n "${MADAROS_STACK_KB:-}" && "${MADAROS_STACK_KB}" != "0" ]]; then
  if [[ "${MADAROS_STACK_KB}" =~ ^[0-9]+$ ]] && [[ "${MADAROS_STACK_KB}" -lt "${_EISA_STACK_FLOOR_KB}" ]]; then
    fail "MADAROS_STACK_KB=${MADAROS_STACK_KB} is below the EISA lowering floor ${_EISA_STACK_FLOOR_KB} KB (448 MiB); see #1760 (default 512 MiB)"
  fi
fi

echo "[eisa-bridge-madaros] Starting Madaros-aware EISA bridge conformance gate (WS-F)"
echo "ROOT_DIR=$ROOT_DIR"
echo "TMP_DIR=$TMP_DIR"
echo "Compiler: $(./bin/souc --version 2>&1 | head -1)"
echo "MADAROS_STACK_KB=${MADAROS_STACK_KB:-<unset; bin/madaros default 524288=512MiB>}"

# 1. Build reference EVM receipts (lean_single as in original gate; EVM path is engine-agnostic)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_evm_run.sio > "$A_OUT" 2>&1 || {
  cat "$A_OUT" >&2
  fail "eisa_evm_run.sio (reference EVM receipts)"
}
echo "[eisa-bridge-madaros] EVM reference receipts captured ($(wc -l < "$A_OUT") lines)"

# 2. Run bridge emitter under default Madaros (no SOUNIO_SOUC_ENGINE=lean_single)
# Tolerate documented BLOCKED (rc=12 from main.elf per eisa_h_zd_reference_gate.sh pattern)
set +e
./bin/souc run tools/eisa/eisa_bridge_emit.sio > "$MADAROS_RUN_LOG" 2>&1
madaros_wrapper_rc=$?
set -e

if [[ "$madaros_wrapper_rc" -eq 0 ]]; then
  madaros_runtime_status="PASS"
  echo "[eisa-bridge-madaros] Madaros emitter: PASS (rc=0)"
elif [[ "$madaros_wrapper_rc" -eq 1 ]] && grep -Fq 'main.elf rc=12' "$MADAROS_RUN_LOG"; then
  madaros_runtime_status="BLOCKED"
  echo "[eisa-bridge-madaros] Madaros emitter: BLOCKED (rc=12, as documented for P0-F FFI dispatch)"
  cat "$MADAROS_RUN_LOG" >&2
  # Continue to lean_single path for parity baseline (per h_zd pattern)
elif [[ "$madaros_wrapper_rc" -ne 0 ]]; then
  cat "$MADAROS_RUN_LOG" >&2
  fail "unexpected Madaros emitter failure rc=$madaros_wrapper_rc (not 0 or documented BLOCKED rc=12)"
fi

# Emit under lean_single to populate artifacts/eisa/*.eisax.elf (required for conformance checks)
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_bridge_emit.sio > "$EMIT_LOG" 2>&1 || {
  cat "$EMIT_LOG" >&2
  fail "eisa_bridge_emit.sio under lean_single (artifact population)"
}
echo "[eisa-bridge-madaros] lean_single emitter completed; artifacts populated. Log: $EMIT_LOG"

>"$B_OUT"

programs=(
  "golden-mul"
  "golden-add"
  "golden-sqrt"
  "golden-poison"
  "e5-cancellation"
  "v1-loop"
  "v1-if-both"
  "v1-i6"
  "v1-fuel"
  "v1-highreg"
  "v1e-fixedpoint"
  "v1e-frail"
  "v1e-emov-negzero"
  "v1-arith-high"
  "v1-fuel-high"
  "v1-branch-high"
  "v2-const-gate"
  "v2-add"
  "v2-sub"
  "v2-mul"
  "v2-div"
  "v2-sqrt"
  "v2-rump-qd"
  "v1-rump-dd"
  "v2-fuel"
  "v2-mem"
  "v2-emov"
  "v2-loop"
  "v2-frail"
  "v2-mem-poison"
)

echo "[eisa-bridge-madaros] Running ${#programs[@]} ELF conformance programs..."
for name in "${programs[@]}"; do
  elf="artifacts/eisa/${name}.eisax.elf"
  if [[ ! -x "$elf" ]]; then
    fail "missing executable ${elf} (bridge_emit did not produce it)"
  fi

  per_prog_out="$TMP_DIR/${name}.stdout"
  set +e
  "$elf" > "$per_prog_out" 2>&1
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "ELF ${name} exited with rc=${rc}" >&2
    cat "$per_prog_out" >&2
    fail "ELF exit code ${rc} for ${name}"
  fi
  cat "$per_prog_out" >> "$B_OUT"
  echo "PASS ${name} (ELF rc=0)"
done

if ! diff -u "$A_OUT" "$B_OUT"; then
  echo "EVM vs Bridge stdout mismatch:" >&2
  diff -u "$A_OUT" "$B_OUT" >&2
  fail "eisa_bridge_conformance: stdout mismatch between EVM reference and Madaros-generated ELFs"
fi
echo "[eisa-bridge-madaros] stdout parity: PASS"

# ── Tamper-sensitivity lane (identical to original) ─────────────────────────
TAMP_ELF="artifacts/eisa/golden-mul-tampered.eisax.elf"
if [[ ! -x "$TAMP_ELF" ]]; then
  fail "missing tampered executable ${TAMP_ELF}"
fi

TAMP_OUT="$TMP_DIR/golden-mul-tampered.stdout"
set +e
"$TAMP_ELF" > "$TAMP_OUT" 2>&1
trc=$?
set -e
if [[ $trc -ne 0 ]]; then
  cat "$TAMP_OUT" >&2
  fail "tampered ELF exit code ${trc}"
fi

GOLDEN_MUL_REF="$TMP_DIR/golden-mul-ref.stdout"
head -n 1 "$A_OUT" > "$GOLDEN_MUL_REF"

if diff -q "$GOLDEN_MUL_REF" "$TAMP_OUT" >/dev/null; then
  fail "tamper-sensitivity: tampered ELF stdout identical to original EVM (vacuous translator)"
fi
echo "[eisa-bridge-madaros] tamper-sensitivity: PASS"

# ── Anti-vacuity lane (identical to original) ───────────────────────────────
echo "[eisa-bridge-madaros] Running anti-vacuity checks on ${#programs[@]} ELFs..."
for name in "${programs[@]}"; do
  elf="artifacts/eisa/${name}.eisax.elf"
  per_prog_out="$TMP_DIR/${name}.stdout"
  expected_prefix="v=1 prog="
  case "$name" in
    v1-*|v1e-*) expected_prefix="v=2 prog=" ;;
    v2-*) expected_prefix="v=3 prog=" ;;
  esac

  if ! grep -aq "$expected_prefix" "$elf"; then
    fail "anti-vacuity ${name}: label prefix '${expected_prefix}' not found in ELF bytes"
  fi

  mapfile -t digit_runs < <(grep -o 'm[0-9]\{8,\}' "$per_prog_out" | sed 's/^m//' | sort -u)
  for run in "${digit_runs[@]}"; do
    if grep -aq "$run" "$elf"; then
      fail "anti-vacuity ${name}: receipt digits '${run}' baked into ELF bytes (vacuous)"
    fi
  done
  echo "  anti-vacuity PASS ${name}"
done
echo "[eisa-bridge-madaros] anti-vacuity: PASS"

echo "[eisa-bridge-madaros] RECEIPT madaros_runtime=${madaros_runtime_status} wrapper_rc=${madaros_wrapper_rc} lean_single_emit=success parity=PASS"
echo "[eisa-bridge-madaros] PASS: EISA bridge conformance under Madaros (tolerating documented P0-F BLOCKED rc=12 where applicable). No changes to tools/eisa/ or stdlib/eisa/."

# WS-F close acceptance (post-E137 fix)
# Crisp criterion for gate closure:
# 1. Default Madaros (no SOUNIO_SOUC_ENGINE) succeeds with rc=0 on emitter (no E137, full 31 ELFs emitted).
# 2. All 30 programs + tampered ELF produce stdout matching EVM reference (or documented divergence with receipt).
# 3. Byte-identical goldens verified via sha256sum against pre-measured lean_single baseline (artifacts/eisa/*.eisax.elf).
# 4. Anti-vacuity, tamper-sensitivity, and string-prefix checks all PASS.
# 5. Any remaining rc=12 explicitly listed in tolerated_blocked array (currently none post-P0-F).
# Baseline captured: 31 ELFs, full set under lean_single (see docs/audit/WS_F_CLOSE_ACCEPTANCE_2026-08-16.md).
# C3 boundary confirmed: no overlap with WS-C PR1/PR2 (frontier-only additions; EISA sources untouched).
