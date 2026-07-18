#!/usr/bin/env bash
# CSV reader gate (Data & Science I/O, Trilha B — reader). Exercises the full
# stdlib/data/csv_reader.sio surface: the string-parser path (6 run-pass drivers under
# tests/stdlib/data/, which also pass on a stock compiler) + the file-input path
# (tests/data_io_gated/*, which need read_file and therefore the #1078-fixed Madaros)
# + cap-free evidence (>256 columns, a >64-byte field).
#
# read_file is only correct with PR #1078. Until #1078 merges, point the gate at a
# #1078-fixed Madaros:  SOUNIO_TEST_SOUC_BIN=/path/to/fixed/madaros bash scripts/data_io_csv_reader_gate.sh
# Dev-tier (not wired into ci.yml until #1078 lands).
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
ROOT="$(pwd)"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0

# compile <src> then run it in $OUT (where fixtures live); grep <sentinel>.
run() {
  local src="$1" sentinel="$2"
  if "$SOUC" compile "$src" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"
    ( cd "$OUT" && "$OUT/x.elf" 2>/dev/null ) | grep -q "$sentinel" || { echo "FAIL run $src (want $sentinel)"; fail=1; }
  else echo "FAIL compile $src"; fail=1; fi
}

# --- string-parser drivers (no file input) ---
run tests/stdlib/data/test_csv_reader_open.sio CSV_READER_OPEN_OK
run tests/stdlib/data/test_csv_reader_rows.sio ROWS_OK
run tests/stdlib/data/test_csv_reader_col.sio  COL_OK
run tests/stdlib/data/test_csv_reader_int.sio  INT_OK
run tests/stdlib/data/test_csv_reader_f64.sio  F64_OK
run tests/stdlib/data/test_csv_reader_str.sio  STR_OK

# --- file-input drivers (need read_file / #1078); fixtures written into $OUT ---
printf 'name,dose\nx,5\ny,6\n' > "$OUT/cr_fixture.csv"
run tests/data_io_gated/csv_reader_file.sio FILE_OK

printf 'k,v\nlong,%s\n' "$(printf 'a%.0s' $(seq 1 200))" > "$OUT/cr_bigfield.csv"
run tests/data_io_gated/csv_reader_bigfield.sio BIGFIELD_OK

# --- cap-free evidence: > 256 columns read correctly ---
python3 - "$OUT/cr_wide.csv" <<'PY'
import sys
n = 300
with open(sys.argv[1], "w") as f:
    f.write(",".join(f"c{i}" for i in range(n)) + "\n")
    f.write(",".join(str(i) for i in range(n)) + "\n")
PY
cat > "$OUT/wide.sio" <<'EOF'
use data::csv_reader::{csv_open_file, csv_next_row, csv_int, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open_file("cr_wide.csv")
    csv_next_row(&!r)
    if csv_int(&!r, 299) == 299 { print("WIDE_OK\n") } else { print("FAIL wide\n") }
    return 0
}
EOF
run "$OUT/wide.sio" WIDE_OK

[ $fail -eq 0 ] && echo "DATA_IO_CSV_READER_GATE_OK"
exit $fail
