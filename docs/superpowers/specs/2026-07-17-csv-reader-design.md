<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-17-csv-reader-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-17-csv-reader-design
-->

# Design — `stdlib/data/csv_reader.sio` (cap-free, row-iterator, header-aware, typed)

**Date:** 2026-07-17 · **Lane:** Data & Science I/O — Trilha B (reader) ·
**Depends on:** the `read_file` fix + dynamic-mmap enhancement in PR #1078.

## Goal

A CSV **reader** to complement the byte-exact writer `stdlib/data/csv.sio`: read a file (or an
in-memory string), iterate rows, and read typed fields by column index **or by header name** — with
**no library-imposed size caps** (columns, field length, rows) and file size bounded only by the OS,
not a fixed constant.

## Non-goals

Full CSV dialect autodetection (delimiters other than `,`, non-LF line endings beyond `\r\n`),
streaming from a pipe/socket, or writing. Delimiter is `,`; terminators are `\n` and `\r\n`.

## Architecture

New module `stdlib/data/csv_reader.sio`, sibling of the writer. Same conventions: `csv_*` names,
`with IO, Mut, Div, Panic`, integer arithmetic (no `print_f64`), RFC-4180 quoting rules.

**No materialized field-offset arrays** — that is what would impose a column cap. Instead the reader
holds only cursors/ranges and **re-scans on demand**:

```
struct CsvReader {
    text: string,        // read_file result (mmap ptr) or a literal — ALWAYS accessed via str_char_at
    text_len: i64,       // str_len(text)
    hdr_start: i64,      // byte range [hdr_start, hdr_end) of the header line
    hdr_end: i64,
    row_start: i64,      // byte range of the CURRENT row (after csv_next_row)
    row_end: i64,
    pos: i64,            // scan cursor: start of the NEXT unread row
    ok: i64,             // 1 = last typed parse well-formed, 0 = malformed (queryable, non-fatal)
}
```

Row and column access are byte-range scans over `text` using `str_char_at`. **`text[i]` array-index
syntax is NEVER used** — it SIGSEGVs on a `read_file` result (separate string-index bug; see the
corrected dispatch). `str_char_at` / `str_len` are the only string accessors.

## API

| Function | Behaviour |
|---|---|
| `csv_open(text: string) -> CsvReader` | Wrap a string; parse the header line's byte range (does not materialize fields). |
| `csv_open_file(path: string) -> CsvReader` | `csv_open(read_file(path))`. Depends on #1078. |
| `csv_next_row(r: &! CsvReader) -> bool` | Advance to the next data row (skipping the header on first call); set `row_start/row_end`; `false` at EOF. |
| `csv_cols(r: &CsvReader) -> i64` | Field count of the current row (one re-scan). |
| `csv_col(r: &CsvReader, name: string) -> i64` | Index of the header column whose bytes equal `name`; `-1` if absent. |
| `csv_int(r: &! CsvReader, col: i64) -> i64` | Parse the current row's `col`-th field as a signed integer. Sets `ok`. |
| `csv_f64(r: &! CsvReader, col: i64) -> f64` | Parse it as a decimal into `f64`. Sets `ok`. |
| `csv_f64_scaled(r: &! CsvReader, col: i64, decimals: i64) -> i64` | Parse as fixed-point scaled i64 (round-trips byte-exact with the writer's `csv_field_fixed`). Sets `ok`. |
| `csv_str_into(r: &CsvReader, col: i64, out: &![i8], cap: i64) -> i64` | Copy the field's bytes into the **caller-sized** buffer (un-escaping `""`→`"`); return the byte length written (truncated to `cap`). No library cap. |
| `csv_ok(r: &CsvReader) -> bool` | True iff the last `csv_int/f64/f64_scaled` parsed a well-formed field. |

Field access is O(col) per call (re-scan from `row_start`). For science CSVs (dozens of columns)
this is negligible; it is the price of zero column cap. Sequential full-row reads are O(cols) total
if the caller reads columns 0..n in order.

## Parsing (RFC-4180)

- **Unquoted field:** bytes from the cursor up to the next `,`, `\n`, `\r`, or EOF.
- **Quoted field** (`"`): from after the opening quote to the closing quote; `""` is one literal `"`.
  Typed parsers (`int`/`f64`) operate on the raw range (numbers are never quoted); `csv_str_into`
  performs `""`→`"` un-escaping while copying.
- **Separator** `,`; **row terminator** `\n`, with a preceding `\r` absorbed (`\r\n`).

## read_file dynamic-mmap enhancement (in #1078, `codegen_x86_linux.sio`)

The current fix mmaps a fixed 1 MiB — a cap. Enhance `emit_builtin_read_file` to size the mapping to
the file: `open(path)` → `lseek(fd,0,SEEK_END)` to get size → `lseek(fd,0,SEEK_SET)` →
`mmap((size+1) rounded up to a 4096 page)` → `read(fd, buf, size)` → `close` → return buf.
The `+1`/page rounding guarantees a trailing zero byte (mmap anon zero-fill) for NUL-termination
regardless of file size. Still Linux x86-64 raw syscalls. The mmap is leaked (matches today).
This removes the file-size cap for every `read_file` caller, not just the CSV reader.

## Error handling

Non-fatal, science-friendly: a malformed numeric field yields `0` and clears `ok` (queryable via
`csv_ok`), rather than panicking — dirty data is expected in real datasets. Out-of-range `col`
(< 0 or ≥ `csv_cols`) returns `0`/`-1` and clears `ok` (no panic). `csv_col` of an unknown name
returns `-1`.

## Limits (all inherent, none arbitrary)

- File size: bounded by available memory (dynamic mmap), not a constant.
- Columns per row: unbounded (re-scan, no offset array).
- Field string length: bounded only by the **caller's** `csv_str_into` buffer.
- Rows: unbounded (streamed).
- Delimiter fixed to `,`; terminators `\n` / `\r\n` only (non-goal to generalize).

## Testing — `scripts/data_io_csv_reader_gate.sh`

Compiled+run with the #1078-fixed Madaros (file input needs the read_file fix):

1. **Parser on a literal** (works on current main too): a multi-line CSV literal → verify `csv_cols`,
   `csv_col("name")` by header, `csv_int`/`csv_f64`/`csv_f64_scaled`/`csv_str_into` per cell,
   and `csv_ok` on a deliberately malformed field.
2. **File round-trip vs the writer:** emit a known CSV with `stdlib/data/csv.sio` to a temp file →
   `csv_open_file` → read every cell back → assert values identical (and `csv_f64_scaled` byte-exact
   against the writer's `csv_field_fixed` input).
3. **Cap-free evidence:** a row with > 256 columns and a field longer than any old fixed buffer,
   both read correctly.

Sentinel `DATA_IO_CSV_READER_GATE_OK`. **CI-green only once #1078 merges** (file input uses
`read_file`); until then it is a runnable dev-tier gate, verified locally against the fixed Madaros.

## Open implementation questions — RESOLVED (verified against Madaros v0.80.0)

1. **`csv_str_into` buffer passing → range+byte escape hatch (no field cap).**
   Size-parametric array refs are **NOT supported**: `fn f<const N: usize>(out: &![i8; N])`
   parses but the type checker resolves `N` to `0` (`error[E009]: expected &![i8; 0]`), so
   option (a) is out. No stdlib-reachable heap `string` builder exists for option (c). The
   shipped design therefore keeps `csv_str_into(out: &![i8; 64], cap)` as an **un-escaping
   convenience** for short fields (RFC-4180 `""`→`"` and quote stripping), and adds two public
   **raw, uncapped** accessors so no field length is capped:
   - `csv_field_range(r, col) -> CsvRange` — the current row's `col`-th field byte range
     (`len == -1` if out of range). `CsvRange { start, len }` is `pub`.
   - `csv_byte(r, i) -> i64` — the byte at absolute index `i` (via `str_char_at`), so callers
     read `[rng.start, rng.start+rng.len)` without touching the private `text` field.
   The raw path does **not** un-escape quoting — it is the verbatim, unbounded path;
   `csv_str_into` is the escaping-but-bounded path. Evidence: `test_csv_reader_bigfield.sio`
   reads a **200-byte** field in full (`len==200`, all 200 bytes) via `csv_field_range`+`csv_byte`,
   from a file (`csv_open_file`) — the real cap-free path, since a 200-byte **string literal**
   cannot be used (Madaros caps string literals near 128 bytes; a compiler limit unrelated to
   this reader — a literal field of ~100 bytes still parses, ~120 does not).
2. **`&!` receiver on `csv_next_row` → supported.** Mutating `CsvReader` fields through
   `&! CsvReader` lowers correctly on native-v2; the reader threads `pos`/`row_*`/`ok` through
   mutation and all 8 driver tests pass.

## Delivery

- **#1078** gains the `read_file` dynamic-mmap commit (removes the 1 MiB cap).
- A new branch/PR adds `stdlib/data/csv_reader.sio` + `scripts/data_io_csv_reader_gate.sh` +
  examples, **depending on #1078**.
