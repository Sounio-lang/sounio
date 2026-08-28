#!/usr/bin/env bash
# Byte-exact + JSON-validity gate for the Data & Science I/O vertical, Trilha A
# (stdout JSON writer). Each case: compile a driver, run it, compare stdout to a
# frozen fixture with `cmp` (byte-exact), and — if jq is present — assert the
# output parses as valid JSON.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

HAVE_JQ=0
command -v jq >/dev/null 2>&1 && HAVE_JQ=1

echo "== check data::json =="
$SOUC check stdlib/data/json.sio || fail=1

check_case () {
  local name="$1" driver="$2" fixture="$3"
  echo "== $name =="
  if $SOUC compile "$driver" -o "$OUT/j.elf" >/dev/null 2>"$OUT/jerr"; then
    if "$OUT/j.elf" > "$OUT/j.out"; then
      if cmp -s "$OUT/j.out" "$fixture"; then
        echo "PASS: stdout == fixture (byte-exact)"
      else
        echo "FAIL: stdout != fixture"; diff <(od -c "$fixture") <(od -c "$OUT/j.out") || true; fail=1
      fi
      if [ "$HAVE_JQ" -eq 1 ]; then
        if jq -e . "$OUT/j.out" >/dev/null 2>&1; then echo "PASS: valid JSON (jq)"; else echo "FAIL: invalid JSON"; fail=1; fi
      fi
    else echo "FAIL: driver run"; fail=1; fi
  else echo "FAIL: driver compile"; tail -3 "$OUT/jerr"; fail=1; fi
}

check_case "flat object (int + f64)" tests/stdlib/data/test_json_obj.sio tests/stdlib/data/fixtures/pk_obj.json
check_case "string escaping (RFC 8259)" tests/stdlib/data/test_json_str.sio tests/stdlib/data/fixtures/pk_str.json
check_case "array of objects (PK table)" tests/stdlib/data/test_json_table.sio tests/stdlib/data/fixtures/pk_table.json
check_case "nested dataset + null/bool (BLQ)" tests/stdlib/data/test_json_dataset.sio tests/stdlib/data/fixtures/pk_dataset.json

# End-to-end: real GUM compute (imports epistemic::gum) -> byte-exact JSON object.
check_case "gum -> json (end-to-end)" examples/epistemic/gum_to_json.sio tests/stdlib/data/fixtures/gum_recovery.json

[ "$HAVE_JQ" -eq 0 ] && echo "(note: jq absent — JSON-validity checks skipped)"
[ $fail -eq 0 ] && echo "DATA_IO_JSON_GATE_OK"
exit $fail
