#!/usr/bin/env bash
# Sounio Science Flex — multi-domain computational receipt.
#
# One native binary must print SOUNIO_SCIENCE_FLEX_OK after:
#   HEP GUM (QED σ + Z width + α_s) + Knightian vancomycin + octonion GUM var
#
# Primary engine: lean_single (epistemic/scientific seed path).
# Madaros is optional best-effort (does not fail the gate on residual).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SRC="$ROOT/examples/sounio_science_flex/main.sio"

run_engine() {
  local engine="$1"
  local label="$2"
  echo "== science flex engine=$label =="
  local out
  out="$(mktemp /tmp/sounio-science-flex.XXXXXX.log)"
  if SOUNIO_SOUC_ENGINE="$engine" "$ROOT/bin/souc" run "$SRC" >"$out" 2>&1; then
    if grep -q 'SOUNIO_SCIENCE_FLEX_OK' "$out"; then
      echo "[science-flex] PASS ($label)"
      grep -E 'FLEX_|PASS |SOUNIO_SCIENCE_FLEX_' "$out" || true
      rm -f "$out"
      return 0
    fi
  fi
  echo "[science-flex] FAIL ($label)" >&2
  cat "$out" >&2
  rm -f "$out"
  return 1
}

run_engine lean_single lean_single

# Optional Madaros — product mitigations should keep particle path alive.
if [[ "${SOUNIO_SCIENCE_FLEX_REQUIRE_MADAROS:-0}" == "1" ]]; then
  run_engine madaros madaros
else
  if SOUNIO_SOUC_ENGINE=madaros "$ROOT/bin/souc" run "$SRC" >/tmp/sounio-science-flex-madaros.log 2>&1 \
    && grep -q 'SOUNIO_SCIENCE_FLEX_OK' /tmp/sounio-science-flex-madaros.log; then
    echo "[science-flex] BONUS madaros OK"
  else
    echo "[science-flex] madaros optional residual (not required)"
  fi
fi

echo "SOUNIO_SCIENCE_FLEX_GATE_OK"
