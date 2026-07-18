# stdlib/data

Tabular data structures.

## Types
- `DataFrame`: Tabular data
- `Series`: Column types (Float, Int, String, Bool, Epistemic)
- CSV loader for network graphs
- Groupby, rolling, cumulative operations

## CSV I/O (`data::csv`, `data::csv_reader`)

- **`csv.sio`** — byte-exact **writer** to stdout: `csv_field_str/int/fixed/f64`, `csv_end_row`
  (streaming, caller threads the column index; RFC-4180 quoting).
- **`csv_reader.sio`** — **cap-free** row-iterator, header-aware, typed **reader**:
  - `csv_open(text) -> CsvReader` / `csv_open_file(path) -> CsvReader` (file input via `read_file`).
  - `csv_next_row(&!r) -> bool`, `csv_cols(&r)`, `csv_col(&r, name) -> i64` (column by header name).
  - Typed current-row access: `csv_int(&!r,col)`, `csv_f64(&!r,col)`, `csv_f64_scaled(&!r,col,dec)`
    (round-trips byte-exact with the writer's `csv_field_fixed`), and `csv_ok(&r)` (non-fatal parse flag).
  - Strings: `csv_str_into(&r,col,out:&![i8;64],cap)` un-escapes quoted `""` into a caller buffer
    (bounded to 64); for **arbitrarily long** fields use the raw uncapped accessors
    `csv_field_range(&r,col) -> CsvRange` + `csv_byte(&r,i)`.
  - **No caps:** columns (re-scan, no offset array), field length (raw accessors), rows (streamed),
    file size (dynamically-sized `read_file` mmap). Delimiter `,`; terminators `\n`/`\r\n`.
  - **Access strings only via `str_char_at`/`str_len`, never `s[i]`** — indexing a `read_file`
    result with `[]` SIGSEGVs (a separate compiler bug). The reader follows this internally.
  - **Requires the `read_file` fix (PR #1078)** for file input; the string-parser path works on any build.

  Gate: `scripts/data_io_csv_reader_gate.sh` (dev-tier until #1078 merges) → `DATA_IO_CSV_READER_GATE_OK`.