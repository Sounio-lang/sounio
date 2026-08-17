#!/usr/bin/env bash
# #862 / D4 closeout gate — named import + helper + print_f64 under Madaros.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"

SRC=tests/epistemic_trust/madaros_862_import_print_accept.sio
OUT="$(mktemp /tmp/madaros_862_out.XXXXXX)"
ERR="$(mktemp /tmp/madaros_862_err.XXXXXX)"
trap 'rm -f "$OUT" "$ERR"' EXIT

echo "== #862 acceptance (Madaros) =="
set +e
SOUNIO_SOUC_ENGINE=madaros ./bin/souc run "$SRC" >"$OUT" 2>"$ERR"
rc=$?
set -e

if ! grep -q 'MADAROS_862_ACCEPT_OK' "$OUT"; then
  echo "FAIL: #862 acceptance not OK rc=$rc" >&2
  tail -40 "$ERR" >&2
  tail -40 "$OUT" >&2
  exit 1
fi
if grep -qE 'E137|visibility preflight failed' "$ERR" "$OUT"; then
  echo "FAIL: E137 / preflight still present" >&2
  grep -E 'E137|visibility preflight' "$ERR" "$OUT" | head -20 >&2
  exit 1
fi

echo "MADAROS_862_GATE_OK"
