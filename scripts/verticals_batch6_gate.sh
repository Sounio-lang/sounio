#!/usr/bin/env bash
# Combined gate: data::json + data::csv (serializers, output-conformance) + math::approx (values).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
echo "== json =="
$SOUC check stdlib/data/json.sio >/dev/null 2>&1 || { echo "FAIL check json"; fail=1; }
if $SOUC compile tests/stdlib/data/test_json_stdlib.sio -o "$OUT/j.elf" >/dev/null 2>&1; then
  o=$("$OUT/j.elf")
  echo "$o" | grep -qF '{"id":7,"name":"ok","active":true,"note":null}' || { echo "FAIL json output"; fail=1; }
  echo "$o" | grep -q "JSON_EMIT_OK" || { echo "FAIL json sentinel"; fail=1; }
else echo "FAIL json compile"; fail=1; fi
echo "== csv =="
$SOUC check stdlib/data/csv.sio >/dev/null 2>&1 || { echo "FAIL check csv"; fail=1; }
if $SOUC compile tests/stdlib/data/test_csv_stdlib.sio -o "$OUT/c.elf" >/dev/null 2>&1; then
  o=$("$OUT/c.elf")
  echo "$o" | grep -qF 'name,age' || { echo "FAIL csv header"; fail=1; }
  echo "$o" | grep -qF 'alice,30' || { echo "FAIL csv row"; fail=1; }
  echo "$o" | grep -q "CSV_EMIT_OK" || { echo "FAIL csv sentinel"; fail=1; }
else echo "FAIL csv compile"; fail=1; fi
echo "== approx =="
$SOUC check stdlib/math/approx.sio >/dev/null 2>&1 || { echo "FAIL check approx"; fail=1; }
if $SOUC compile tests/stdlib/math/test_approx_stdlib.sio -o "$OUT/a.elf" >/dev/null 2>&1; then
  "$OUT/a.elf" | grep -q "APPROX_STDLIB_OK" || { echo "FAIL approx"; fail=1; }
else echo "FAIL approx compile"; fail=1; fi
[ $fail -eq 0 ] && echo "VERTICALS_BATCH6_GATE_OK"
exit $fail
