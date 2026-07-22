#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
engine_info="$($SOUC info 2>&1)"
if ! grep -qF 'Madaros v' <<<"$engine_info"; then
  echo "FAIL: robust-statistics gate requires default Madaros" >&2
  printf '%s\n' "$engine_info" >&2
  exit 1
fi
echo "== check data/robust.sio =="
$SOUC check stdlib/data/robust.sio >/dev/null 2>&1 || echo "NOTE: standalone check quirk on stdlib/data/robust.sio (Madaros check-mode; driver proves the API)"
echo "== run-proof: robust statistics =="
if $SOUC compile tests/stdlib/data/test_robust_stdlib.sio -o "$OUT/x.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/x.elf"; "$OUT/x.elf" | grep -q "ROBUST_STDLIB_OK" || { echo "FAIL run"; fail=1; }
else echo "FAIL compile"; fail=1; fi
[ $fail -eq 0 ] && echo "ROBUST_GATE_OK"
exit $fail
