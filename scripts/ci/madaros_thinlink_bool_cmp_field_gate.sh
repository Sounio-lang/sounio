#!/usr/bin/env bash
# KL-12: Madaros native-v2 accepts ≥2 f64 comparisons in bool struct fields.
# lean_single oracle + precomp smoke stay green as regression anchors.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export SOUNIO_STDLIB_PATH="${SOUNIO_STDLIB_PATH:-$ROOT/stdlib}"
SOUC="${SOUC:-$ROOT/bin/souc}"
PROBE="tests/run-pass/thinlink_bool_cmp_field.sio"
SMOKE="tests/run-pass/thinlink_bool_cmp_field_precomp_smoke.sio"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT

echo "== madaros_thinlink_bool_cmp_field_gate =="

unset SOUNIO_SOUC_ENGINE || true
set +e
"$SOUC" run "$PROBE" >"$OUT/madaros_probe.log" 2>&1
MRC=$?
set -e
[[ "$MRC" -eq 0 ]] || {
  echo "FAIL: Madaros cmp-in-field probe rc=$MRC"
  cat "$OUT/madaros_probe.log" || true
  exit 1
}
grep -Fq 'BOOL_CMP_FIELD PASS' "$OUT/madaros_probe.log" || {
  echo "FAIL: Madaros missing BOOL_CMP_FIELD PASS"
  cat "$OUT/madaros_probe.log" || true
  exit 1
}
if grep -Fq 'Failed to write native binary' "$OUT/madaros_probe.log"; then
  echo "FAIL: Madaros still fail-closed on native emit (rc=12 regression)"
  cat "$OUT/madaros_probe.log" || true
  exit 1
fi
if grep -Fq 'Segmentation fault' "$OUT/madaros_probe.log"; then
  echo "FAIL: segfault"
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
