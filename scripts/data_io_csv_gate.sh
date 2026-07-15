#!/usr/bin/env bash
# Byte-exact gate for the Data & Science I/O vertical, Trilha A (stdout writer).
# Compiles the CSV driver, runs it, and compares stdout to the frozen fixture
# with `cmp` (byte-exact — no whitespace/line-ending tolerance).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
fail=0

echo "== check data::csv =="
$SOUC check stdlib/data/csv.sio || fail=1

# Each case: driver .sio + frozen fixture, compared byte-exact with cmp.
check_case () {
  local name="$1" driver="$2" fixture="$3"
  echo "== $name =="
  if $SOUC compile "$driver" -o "$OUT/c.elf" >/dev/null 2>"$OUT/cerr"; then
    if "$OUT/c.elf" > "$OUT/c.out"; then
      if cmp -s "$OUT/c.out" "$fixture"; then
        echo "PASS: stdout == fixture (byte-exact)"
      else
        echo "FAIL: stdout != fixture"; diff <(od -c "$fixture") <(od -c "$OUT/c.out") || true; fail=1
      fi
    else echo "FAIL: driver run"; fail=1; fi
  else echo "FAIL: driver compile"; tail -3 "$OUT/cerr"; fail=1; fi
}

check_case "int + str fields"     tests/stdlib/data/test_csv_stdout.sio tests/stdlib/data/fixtures/pk_basic.csv
check_case "fixed-point decimals" tests/stdlib/data/test_csv_fixed.sio  tests/stdlib/data/fixtures/pk_conc.csv
check_case "f64 fixed-point"      tests/stdlib/data/test_csv_f64.sio    tests/stdlib/data/fixtures/pk_f64.csv

# End-to-end: real GUM compute (imports epistemic::gum) -> byte-exact CSV row.
# Values pinned to the GUM run-proof (test_gum_stdlib): u_c=0.290402, U95=0.569188.
check_case "gum -> csv (end-to-end)" examples/epistemic/gum_to_csv.sio tests/stdlib/data/fixtures/gum_recovery.csv

# Multi-row PK concentration-time table, per-point GUM combined uncertainty.
check_case "pk curve (multi-row + u_c)" examples/epistemic/pk_curve_gum_to_csv.sio tests/stdlib/data/fixtures/pk_curve.csv
check_case "rfc-4180 quoting"           tests/stdlib/data/test_csv_quote.sio   tests/stdlib/data/fixtures/csv_quoting.csv

[ $fail -eq 0 ] && echo "DATA_IO_CSV_GATE_OK"
exit $fail
