#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

mkdir -p artifacts/eisa
TMP_DIR="artifacts/eisa/.gate_tmp"
mkdir -p "$TMP_DIR"

A_OUT="$TMP_DIR/evm_stdout.txt"
B_OUT="$TMP_DIR/bridge_stdout.txt"
EMIT_LOG="$TMP_DIR/emit_driver.log"

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_evm_run.sio > "$A_OUT"
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tools/eisa/eisa_bridge_emit.sio > "$EMIT_LOG"

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

for name in "${programs[@]}"; do
  elf="artifacts/eisa/${name}.eisax.elf"
  if [[ ! -x "$elf" ]]; then
    echo "FAIL ${name}: missing executable ${elf}"
    exit 1
  fi

  per_prog_out="$TMP_DIR/${name}.stdout"
  set +e
  "$elf" > "$per_prog_out"
  rc=$?
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "FAIL ${name}: ELF exit code ${rc}"
    exit 1
  fi
  cat "$per_prog_out" >> "$B_OUT"
  echo "PASS ${name}"
done

if ! diff -u "$A_OUT" "$B_OUT"; then
  echo "FAIL eisa_bridge_conformance: stdout mismatch"
  exit 1
fi

# ── Tamper-sensitivity lane ──────────────────────────────────────────────────
# The tampered image flips golden-mul's first constant 0.5 -> 0.6 (re-encoded so
# hash + validation still pass). Its ELF must compute a DIFFERENT receipt from
# the untouched golden-mul EVM run — proving the ELF derives its output from the
# embedded constants at runtime, not from a baked receipt string.
TAMP_ELF="artifacts/eisa/golden-mul-tampered.eisax.elf"
if [[ ! -x "$TAMP_ELF" ]]; then
  echo "FAIL tamper-sensitivity: missing executable ${TAMP_ELF}"
  exit 1
fi

TAMP_OUT="$TMP_DIR/golden-mul-tampered.stdout"
set +e
"$TAMP_ELF" > "$TAMP_OUT"
trc=$?
set -e
if [[ $trc -ne 0 ]]; then
  echo "FAIL tamper-sensitivity: tampered ELF exit code ${trc}"
  exit 1
fi

# Original golden-mul EVM receipt is the first line of the EVM reference output.
GOLDEN_MUL_REF="$TMP_DIR/golden-mul-ref.stdout"
head -n 1 "$A_OUT" > "$GOLDEN_MUL_REF"

if diff -q "$GOLDEN_MUL_REF" "$TAMP_OUT" >/dev/null; then
  echo "FAIL tamper-sensitivity: tampered ELF stdout is identical to original EVM stdout"
  exit 1
fi
echo "PASS tamper-sensitivity"

# ── Anti-vacuity lane ────────────────────────────────────────────────────────
# Tamper sensitivity alone only proves translator determinism: a vacuous
# translator that bakes the receipt text at translation time would recompute
# the baked text for the tampered image too (math-review finding, 2026-07-05).
# The load-bearing witness that the ELF COMPUTES its receipts is that the
# receipt's value digits are absent from the binary's bytes, while the static
# label prefix is present (proving the strings extraction itself works).
for name in "${programs[@]}"; do
  elf="artifacts/eisa/${name}.eisax.elf"
  per_prog_out="$TMP_DIR/${name}.stdout"
  expected_prefix="v=1 prog="
  case "$name" in
    v1-*|v1e-*) expected_prefix="v=2 prog=" ;;
    v2-*) expected_prefix="v=3 prog=" ;;
  esac
  # grep -a (not `strings`): binutils `strings` is absent on the pinned compute
  # node; grep -a scans the raw ELF bytes as text, which is a strict superset of
  # the printable-run extraction for these ASCII patterns (portability audit
  # 2026-07-06, docs/audit/CI_GATE_PORTABILITY_2026-07-06.md).
  if ! grep -aq "$expected_prefix" "$elf"; then
    echo "FAIL anti-vacuity ${name}: label prefix not found in ELF bytes"
    exit 1
  fi
  # Mantissa digit runs (>= 8 digits) from val/roundoff/u fields must not
  # appear as literal bytes in the ELF.
  mapfile -t digit_runs < <(grep -o 'm[0-9]\{8,\}' "$per_prog_out" | sed 's/^m//' | sort -u)
  for run in "${digit_runs[@]}"; do
    if grep -aq "$run" "$elf"; then
      echo "FAIL anti-vacuity ${name}: receipt digits '${run}' baked into ELF bytes"
      exit 1
    fi
  done
done
echo "PASS anti-vacuity"

echo "PASS eisa_bridge_conformance"
