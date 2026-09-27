<!-- docs:meta
topic_id: repo.docs.tutorials.data-processing
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.tutorials.data-processing
-->

# Data Processing with Sounio: CSV Parsing and Analysis

Sounio handles tabular data through fixed-size arrays and struct wrappers -- no heap allocation, no garbage collector, predictable memory layout. This tutorial shows how to parse CSV data, compute column statistics, and build data processing pipelines.

The production implementation lives in [`stdlib/csv/parser.sio`](../../stdlib/csv/parser.sio).

## Prerequisites

The programs below are built up inside this tutorial; there is no ready-made
example file to run. Paste a snippet into a file of your own and run it with the
prebuilt compiler:

```bash
SOUC=./bin/souc
$SOUC run my_csv_analysis.sio      # a file you create from the snippets below
```

## 1. The CsvTable Data Structure

Sounio uses fixed-size arrays for all data. The `CsvTable` struct holds up to 256 rows and 16 columns in a flat `[f64; 4096]` array (row-major layout).

```sio
struct CsvTable {
    data:         [f64; 4096],      // row-major: data[row * n_cols + col]
    n_rows:       i32,
    n_cols:       i32,
    max_rows:     i32,              // fixed at 256
    max_cols:     i32,              // fixed at 16
    has_header:   i32,              // 0 = no, 1 = yes
    header_bytes: [[u8; 32]; 16],   // column names (up to 32 bytes each)
    header_lens:  [i32; 16],
}

fn csv_table_new() -> CsvTable {
    CsvTable {
        data:         [0.0; 4096],
        n_rows:       0,
        n_cols:       0,
        max_rows:     256,
        max_cols:     16,
        has_header:   0,
        header_bytes: [[0u8; 32]; 16],
        header_lens:  [0; 16],
    }
}
```

Fixed-size arrays mean no allocation failures, no out-of-memory surprises, and deterministic performance. The tradeoff is a hard cap: 256 rows, 16 columns.

## 2. The CsvBuf Wrapper Pattern

Sounio's JIT has a known limitation: mutations to bare `&![u8; N]` arrays may not propagate back to the caller. The fix is to wrap byte buffers in a struct.

```sio
// WRONG -- bare array mutation may not propagate in JIT
fn fill_buffer(buf: &![u8; 4096]) with Mut, Panic {
    (*buf)[0] = 72u8   // might not be visible to caller
}

// CORRECT -- struct wrapper propagates mutations reliably
struct CsvBuf {
    bytes: [u8; 4096],
    len:   i32,
}

fn csv_buf_new() -> CsvBuf {
    CsvBuf { bytes: [0u8; 4096], len: 0 }
}

fn fill_buffer(buf: &!CsvBuf) with Mut, Panic {
    buf.bytes[0] = 72u8    // visible to caller
    buf.len = 1
}
```

This pattern appears throughout the stdlib. When you see `&!CsvBuf` instead of `&![u8; 4096]`, this is why.

## 3. Parsing CSV from Bytes

The parser takes raw bytes and fills a `CsvTable`. It auto-detects header rows (first line starting with an alphabetic character is treated as column names).

```sio
struct CsvBuf {
    bytes: [u8; 4096],
    len:   i32,
}

struct CsvTable {
    data:         [f64; 4096],
    n_rows:       i32,
    n_cols:       i32,
    max_rows:     i32,
    max_cols:     i32,
    has_header:   i32,
    header_bytes: [[u8; 32]; 16],
    header_lens:  [i32; 16],
}

fn csv_table_new() -> CsvTable {
    CsvTable {
        data:         [0.0; 4096],
        n_rows:       0,
        n_cols:       0,
        max_rows:     256,
        max_cols:     16,
        has_header:   0,
        header_bytes: [[0u8; 32]; 16],
        header_lens:  [0; 16],
    }
}

// Manually populate a byte buffer with CSV content
fn build_csv_input(buf: &!CsvBuf) with Mut, Panic {
    // "1.0,2.0,3.0\n4.0,5.0,6.0\n"
    // ASCII: '1'=49, '.'=46, '0'=48, ','=44, '\n'=10
    buf.bytes[0] = 49u8     // '1'
    buf.bytes[1] = 46u8     // '.'
    buf.bytes[2] = 48u8     // '0'
    buf.bytes[3] = 44u8     // ','
    buf.bytes[4] = 50u8     // '2'
    buf.bytes[5] = 46u8     // '.'
    buf.bytes[6] = 48u8     // '0'
    buf.bytes[7] = 44u8     // ','
    buf.bytes[8] = 51u8     // '3'
    buf.bytes[9] = 46u8     // '.'
    buf.bytes[10] = 48u8    // '0'
    buf.bytes[11] = 10u8    // '\n'
    buf.bytes[12] = 52u8    // '4'
    buf.bytes[13] = 46u8    // '.'
    buf.bytes[14] = 48u8    // '0'
    buf.bytes[15] = 44u8    // ','
    buf.bytes[16] = 53u8    // '5'
    buf.bytes[17] = 46u8    // '.'
    buf.bytes[18] = 48u8    // '0'
    buf.bytes[19] = 44u8    // ','
    buf.bytes[20] = 54u8    // '6'
    buf.bytes[21] = 46u8    // '.'
    buf.bytes[22] = 48u8    // '0'
    buf.bytes[23] = 10u8    // '\n'
    buf.len = 24
}
```

The parser is called as:
```sio
// csv_parse(input_bytes, input_len, table, delimiter)
// delimiter 44u8 = ','
let rc = csv_parse(&buf.bytes, buf.len, &!table, 44u8)
assert(rc == 0)   // 0 = success, -1 = error
```

## 4. Computing Column Statistics

Once data is in a `CsvTable`, column statistics use simple loops over the flat array.

```sio
fn csv_col_sum(t: &CsvTable, col: i32) -> f64 {
    if col < 0 || col >= t.n_cols { return 0.0 }
    var s: f64 = 0.0
    var r: i32 = 0
    while r < t.n_rows {
        s = s + t.data[(r * t.n_cols + col) as usize]
        r = r + 1
    }
    s
}

fn csv_col_mean(t: &CsvTable, col: i32) -> f64 with Div, Panic {
    if t.n_rows == 0 { return 0.0 }
    csv_col_sum(t, col) / (t.n_rows as f64)
}

fn csv_col_variance(t: &CsvTable, col: i32) -> f64 with Mut, Div, Panic {
    if t.n_rows == 0 { return 0.0 }
    let mean = csv_col_mean(t, col)
    var acc: f64 = 0.0
    var r: i32 = 0
    while r < t.n_rows {
        let d = t.data[(r * t.n_cols + col) as usize] - mean
        acc = acc + d * d
        r = r + 1
    }
    acc / (t.n_rows as f64)
}

fn csv_col_min(t: &CsvTable, col: i32) -> f64 {
    if t.n_rows == 0 { return 0.0 }
    var m: f64 = t.data[col as usize]
    var r: i32 = 1
    while r < t.n_rows {
        let v = t.data[(r * t.n_cols + col) as usize]
        if v < m { m = v }
        r = r + 1
    }
    m
}

fn csv_col_max(t: &CsvTable, col: i32) -> f64 {
    if t.n_rows == 0 { return 0.0 }
    var m: f64 = t.data[col as usize]
    var r: i32 = 1
    while r < t.n_rows {
        let v = t.data[(r * t.n_cols + col) as usize]
        if v > m { m = v }
        r = r + 1
    }
    m
}
```

Notice:
- `csv_col_sum` has **no effects** -- it only reads data. Pure function.
- `csv_col_mean` needs `with Div, Panic` because it divides.
- `csv_col_variance` adds `with Mut` because it uses `var acc`.

## 5. Complete Example: Build, Parse, and Analyze

This self-contained example creates a table manually, computes stats, and verifies results.

```sio
struct CsvTable {
    data:         [f64; 4096],
    n_rows:       i32,
    n_cols:       i32,
    max_rows:     i32,
    max_cols:     i32,
    has_header:   i32,
    header_bytes: [[u8; 32]; 16],
    header_lens:  [i32; 16],
}

fn csv_table_new() -> CsvTable {
    CsvTable {
        data:         [0.0; 4096],
        n_rows:       0,
        n_cols:       0,
        max_rows:     256,
        max_cols:     16,
        has_header:   0,
        header_bytes: [[0u8; 32]; 16],
        header_lens:  [0; 16],
    }
}

fn csv_table_set(t: &!CsvTable, row: i32, col: i32, val: f64) with Mut, Panic {
    let idx = row * t.n_cols + col
    t.data[idx as usize] = val
}

fn csv_table_get(t: &CsvTable, row: i32, col: i32) -> f64 {
    let idx = row * t.n_cols + col
    t.data[idx as usize]
}

fn csv_col_sum(t: &CsvTable, col: i32) -> f64 {
    var s: f64 = 0.0
    var r: i32 = 0
    while r < t.n_rows {
        s = s + t.data[(r * t.n_cols + col) as usize]
        r = r + 1
    }
    s
}

fn csv_col_mean(t: &CsvTable, col: i32) -> f64 with Div, Panic {
    csv_col_sum(t, col) / (t.n_rows as f64)
}

fn csv_col_min(t: &CsvTable, col: i32) -> f64 {
    var m: f64 = t.data[col as usize]
    var r: i32 = 1
    while r < t.n_rows {
        let v = t.data[(r * t.n_cols + col) as usize]
        if v < m { m = v }
        r = r + 1
    }
    m
}

fn csv_col_max(t: &CsvTable, col: i32) -> f64 {
    var m: f64 = t.data[col as usize]
    var r: i32 = 1
    while r < t.n_rows {
        let v = t.data[(r * t.n_cols + col) as usize]
        if v > m { m = v }
        r = r + 1
    }
    m
}

fn abs(x: f64) -> f64 {
    if x < 0.0 { 0.0 - x } else { x }
}

fn main() -> i32 with IO, Mut, Div, Panic {
    // Build a 5-row, 3-column table (e.g. age, weight_kg, height_cm)
    var t = csv_table_new()
    t.n_cols = 3
    t.n_rows = 5

    // Row 0: 25, 70.0, 175.0
    csv_table_set(&!t, 0, 0, 25.0)
    csv_table_set(&!t, 0, 1, 70.0)
    csv_table_set(&!t, 0, 2, 175.0)

    // Row 1: 30, 80.0, 180.0
    csv_table_set(&!t, 1, 0, 30.0)
    csv_table_set(&!t, 1, 1, 80.0)
    csv_table_set(&!t, 1, 2, 180.0)

    // Row 2: 22, 60.0, 165.0
    csv_table_set(&!t, 2, 0, 22.0)
    csv_table_set(&!t, 2, 1, 60.0)
    csv_table_set(&!t, 2, 2, 165.0)

    // Row 3: 35, 90.0, 185.0
    csv_table_set(&!t, 3, 0, 35.0)
    csv_table_set(&!t, 3, 1, 90.0)
    csv_table_set(&!t, 3, 2, 185.0)

    // Row 4: 28, 75.0, 170.0
    csv_table_set(&!t, 4, 0, 28.0)
    csv_table_set(&!t, 4, 1, 75.0)
    csv_table_set(&!t, 4, 2, 170.0)

    // Column 0 (age): mean = 28.0, min = 22, max = 35
    let age_mean = csv_col_mean(&t, 0)
    assert(abs(age_mean - 28.0) < 0.01)

    let age_min = csv_col_min(&t, 0)
    assert(abs(age_min - 22.0) < 0.01)

    let age_max = csv_col_max(&t, 0)
    assert(abs(age_max - 35.0) < 0.01)

    // Column 1 (weight): mean = 75.0, sum = 375.0
    let weight_sum = csv_col_sum(&t, 1)
    assert(abs(weight_sum - 375.0) < 0.01)

    let weight_mean = csv_col_mean(&t, 1)
    assert(abs(weight_mean - 75.0) < 0.01)

    println("all stats verified")
    0
}
```

## 6. Row-Major Indexing

The flat `data[4096]` array uses row-major indexing: `data[row * n_cols + col]`. This matches C memory layout and is cache-friendly for row scans.

```
Row 0: data[0], data[1], data[2]     (age=25, weight=70, height=175)
Row 1: data[3], data[4], data[5]     (age=30, weight=80, height=180)
Row 2: data[6], data[7], data[8]     (age=22, weight=60, height=165)
```

Column scans (like `csv_col_mean`) stride by `n_cols`, which is less cache-friendly but works well for tables that fit in L1 cache (4096 f64 values = 32 KB).

## 7. Design Tradeoffs vs. Dynamic Languages

| Feature | Sounio CsvTable | Python pandas |
|---------|----------------|---------------|
| Max rows | 256 (compile-time) | Unbounded (heap) |
| Max columns | 16 (compile-time) | Unbounded (heap) |
| Allocation | Zero -- stack only | Dynamic, GC-managed |
| Effect tracking | Compile-time verified | None |
| Parse errors | Return code (0 or -1) | Exceptions |
| Memory layout | Flat f64 array | Column-oriented objects |

Sounio's approach trades flexibility for predictability. You know exactly how much memory your program uses before it runs. For scientific workloads where data sizes are known upfront (sensor readings, experimental results, pharmacokinetic parameters), this is a feature, not a limitation.

## Production Reference

The complete CSV parser and serializer lives in:

- **`stdlib/csv/parser.sio`** -- `CsvTable`, `CsvBuf`, `csv_parse`, `csv_serialize`, `csv_col_mean`, `csv_col_variance`, `csv_col_min`, `csv_col_max`, `csv_table_add_row`, `csv_serialize_buf`, `csv_parse_buf`

Import with:
```sio
use csv::parser::{CsvTable, csv_table_new, csv_parse, csv_col_mean}
```
