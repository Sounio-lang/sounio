#!/usr/bin/env bash
# Classifies Madaros thin-link rc=12 on ≥2 f64 comparisons in bool struct fields,
# with lean_single oracle green and precomp smoke green.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/known_failures/thinlink_bool_cmp_field_probe.sio"
SMOKE="tests/run-pass/thinlink_bool_cmp_field_precomp_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_thinlink_bool_cmp_field_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros_probe.log" 2>&1
MRC=$?
set -e
if [[ "$MRC" -eq 0 ]]; then
  echo "FAIL: Madaros unexpectedly succeeded on cmp-in-field probe; promote needs a green gate"
  cat "$OUT/madaros_probe.log" || true
  exit 1
fi
grep -Fq 'Failed to write native binary' "$OUT/madaros_probe.log" || {
  echo "FAIL: missing fail-closed native emit marker"
  cat "$OUT/madaros_probe.log" || true
  exit 1
}
if grep -Fq 'Segmentation fault' "$OUT/madaros_probe.log"; then
  echo "FAIL: segfault (not an allowed fail-closed mode)"
  cat "$OUT/madaros_probe.log" || true
  exit 1
fi
if grep -Fq 'BOOL_CMP_FIELD PASS' "$OUT/madaros_probe.log"; then
  echo "FAIL: Madaros printed PASS while rc!=0"
  cat "$OUT/madaros_probe.log" || true
  exit 1
fi

export SOUNIO_SOUC_ENGINE=lean_single
set +e
"$SOUC" run "$PROBE" >"$OUT/lean_probe.log" 2>&1
LRC=$?
set -e
unset SOUNIO_SOUC_ENGINE || true
[[ "$LRC" -eq 0 ]] || {
  echo "FAIL: lean_single oracle rc=$LRC"
  cat "$OUT/lean_probe.log" || true
  exit 1
}
grep -Fq 'BOOL_CMP_FIELD PASS' "$OUT/lean_probe.log" || {
  echo "FAIL: lean_single missing BOOL_CMP_FIELD PASS"
  cat "$OUT/lean_probe.log" || true
  exit 1
}

set +e
"$SOUC" run "$SMOKE" >"$OUT/smoke.log" 2>&1
SRC=$?
set -e
[[ "$SRC" -eq 0 ]] || {
  echo "FAIL: precomp smoke rc=$SRC"
  cat "$OUT/smoke.log" || true
  exit 1
}
grep -Fq 'BOOL_CMP_FIELD_PRECOMP PASS' "$OUT/smoke.log" || {
  echo "FAIL: precomp smoke missing sentinel"
  cat "$OUT/smoke.log" || true
  exit 1
}
if grep -Fq 'Segmentation fault' "$OUT/smoke.log"; then
  echo "FAIL: precomp smoke segfault"
  cat "$OUT/smoke.log" || true
  exit 1
fi

echo "MADAROS_THINLINK_BOOL_CMP_FIELD_GATE_OK"
