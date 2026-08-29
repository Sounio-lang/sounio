<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-13-data-science-io
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-13-data-science-io
-->

# Data & Science I/O — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Sounio load, hold, and emit real-world scientific data through one canonical, epistemic-aware `DataFrame`, with pure-Sounio readers (CSV, Parquet, netCDF-classic, HDF5) and writers (CSV, JSON, table, provenance artifact) — no compiler changes.

**Architecture:** `stdlib/data/frame.sio` is the hub (extended additively with a null mask, a datetime dtype, and per-column units/provenance). Writers and readers are spokes targeting the hub's frozen API. Shared substrate is `stdlib/io/binary.sio` (byte reads) and `stdlib/compress/*` (gzip/zstd, plus a new snappy). "Real" is defined by golden fixtures emitted by reference tools (pyarrow/netCDF4/h5py).

**Tech Stack:** Sounio (`./bin/souc` → Madaros). Builtin `String` with `++` concat and `(x as String)` casts (the idiom already used in `frame.sio`). Test harness: `//@ check-only` or a `fn main() -> i32` returning `0` for pass, run via `./bin/souc run`.

**Scope of THIS plan:** Phase 0 (hub hardening) and Phase 1 (text writers) are specified in full, executable, bite-sized TDD detail. Phases 2–5 (CSV read, Parquet, netCDF, HDF5) are laid out as roadmap tasks with fixed interfaces; **their detailed step-by-step plans are authored after the Phase 0 API freeze**, because reader code must target the frozen hub API and writing exact steps now would invalidate on the first API adjustment. This is deliberate decomposition, not deferral of design — the interfaces they must hit are pinned below.

**Preamble — run once per shell before any task:**
```bash
cd /workspace/sounio
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
SOUC=./bin/souc
```

**Ground rules (from CLAUDE.md):**
- Never touch `self-hosted/` or `bootstrap/`. If a step hits a codegen wall, STOP and report it as a forensic dispatch — do not work around it in the compiler.
- Additive only: do not delete `data/dataframe.sio`, `dataframe/pure/core.sio`, or `data/csv_loader.sio`.
- EN-UK orthography. Atomic commits, one logical change each, no AI attribution in messages.
- `Column {` struct literals exist ONLY in `stdlib/data/frame.sio` (13 sites) — adding fields is contained to that file.

---

## File Structure

| File | Responsibility | Phase |
|---|---|---|
| `stdlib/data/frame.sio` (modify) | Hub: `Column`/`DataFrame` + null mask, datetime dtype, units/provenance, metadata-preserving filter/slice | 0 |
| `stdlib/data/README.md` (create) | Frozen hub API contract | 0 |
| `tests/stdlib/data/test_frame_nulls.sio` (create) | Null-mask behaviour | 0 |
| `tests/stdlib/data/test_frame_datetime.sio` (create) | Datetime dtype | 0 |
| `tests/stdlib/data/test_frame_metadata.sio` (create) | Units/provenance survive slice/filter | 0 |
| `stdlib/data/write_csv.sio` (create) | DataFrame → CSV string (RFC-4180) | 1 |
| `stdlib/data/write_json.sio` (create) | DataFrame → JSON string (records + columnar) | 1 |
| `stdlib/data/table.sio` (create) | DataFrame → aligned/markdown table string | 1 |
| `stdlib/data/artifact.sio` (create) | DataFrame → data + `.meta.json` sidecar strings | 1 |
| `tests/stdlib/data/test_write_*.sio` (create) | Writer round-trip/format gates | 1 |
| `stdlib/compress/snappy.sio` (create) | Snappy block decompress (Parquet read) | 3 |
| `stdlib/parquet/` (create) | Parquet reader + writer + Thrift codec | 3 |
| `stdlib/netcdf/` (create) | netCDF-classic reader | 4 |
| `stdlib/hdf5/` (create) | HDF5 reader (contiguous + chunked/filtered) | 5 |
| `scripts/data_io_gate.sh` (create) | Golden-fixture conformance gate wiring | 1→5 |

---

## PHASE 0 — Hub hardening

**Design decisions locked (from spec §10):**
- Null mask: field `null_mask: [bool]`, entry `true` = **missing**. Empty `[]` = nothing missing (keeps existing constructors cheap).
- Datetime: `dtype == 5`, epoch stored in `int_data` as `i64` **nanoseconds**; unit tag `dt_unit: i32` (0=ns, 1=us, 2=ms, 3=s).
- Metadata: `units: String` and `provenance: String`, default `""`.

### Task 1: Add metadata fields to `Column`

**Files:**
- Modify: `stdlib/data/frame.sio` (the `pub struct Column` + all 13 `Column {` constructor sites)
- Test: `tests/stdlib/data/test_frame_metadata.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_frame_metadata.sio`:
```sounio
//@ check-only
// tests/stdlib/data/test_frame_metadata.sio
use data::frame::*

fn assert_meta_defaults() -> bool {
    var d: [f64] = []
    d.push(1.0)
    d.push(2.0)
    let c = column_float("x", d)
    if c.units != "" { return false }
    if c.provenance != "" { return false }
    if c.null_mask.len() != 0 { return false }
    if c.dt_unit != 0 { return false }
    true
}

fn main() -> i32 {
    if !assert_meta_defaults() { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC check tests/stdlib/data/test_frame_metadata.sio`
Expected: FAIL — unknown field `units` on `Column`.

- [ ] **Step 3: Add the fields to the struct**

In `stdlib/data/frame.sio`, extend `pub struct Column` (after `conf_data`):
```sounio
    pub uncert_data: [f64],
    pub conf_data: [f64],
    // --- Phase 0 additions ---
    pub null_mask: [bool],    // entry true = missing; empty = none missing
    pub units: String,        // "" = unitless
    pub provenance: String,   // "" = no recorded provenance
    pub dt_unit: i32,         // datetime unit: 0=ns 1=us 2=ms 3=s (only meaningful when dtype==5)
```

- [ ] **Step 4: Update every constructor**

Each of the 13 `Column { ... }` literals in `frame.sio` must initialise the 4 new fields. Add these lines before the closing `}` of every `Column {` literal, and declare an empty bool array at the top of each constructor alongside the existing `empty_*`:
```sounio
    var empty_nm: [bool] = []
```
and in the literal:
```sounio
        null_mask: empty_nm,
        units: "",
        provenance: "",
        dt_unit: 0,
```
For `column_epistemic` (dtype 4) and all others the values are identical defaults. Do this for all 13 sites (`column_float`, `column_int`, `column_string`, `column_bool`, `column_epistemic`, and the 8 rebuild literals inside `dataframe_filter`/`dataframe_slice`/`dataframe_head`/`dataframe_tail` if they inline `Column {` — grep to confirm: `grep -n "Column {" stdlib/data/frame.sio`).

- [ ] **Step 5: Run to verify it passes**

Run: `$SOUC check tests/stdlib/data/test_frame_metadata.sio`
Expected: PASS (exit 0).

- [ ] **Step 6: Regression — hub still checks**

Run: `$SOUC check stdlib/data/frame.sio`
Expected: PASS.

- [ ] **Step 7: Commit**
```bash
git add stdlib/data/frame.sio tests/stdlib/data/test_frame_metadata.sio
git commit -m "feat(data): add null-mask/units/provenance/dt_unit fields to Column"
```

### Task 2: Null-mask accessors

**Files:**
- Modify: `stdlib/data/frame.sio`
- Test: `tests/stdlib/data/test_frame_nulls.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_frame_nulls.sio`:
```sounio
//@ check-only
// tests/stdlib/data/test_frame_nulls.sio
use data::frame::*

fn assert_nulls() -> bool {
    var d: [f64] = []
    d.push(1.0)
    d.push(2.0)
    d.push(3.0)
    var c = column_float("x", d)
    // no mask yet -> nothing null
    if column_is_null(c, 1) { return false }
    if column_null_count(c) != 0 { return false }
    // mark index 1 as missing
    var m: [bool] = []
    m.push(false)
    m.push(true)
    m.push(false)
    c = column_set_null_mask(c, m)
    if !column_is_null(c, 1) { return false }
    if column_is_null(c, 0) { return false }
    if column_null_count(c) != 1 { return false }
    true
}

fn main() -> i32 {
    if !assert_nulls() { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC check tests/stdlib/data/test_frame_nulls.sio`
Expected: FAIL — `column_is_null` not defined.

- [ ] **Step 3: Implement the accessors**

Append to `stdlib/data/frame.sio` (after the Column constructors, before `struct DataFrame`):
```sounio
pub fn column_set_null_mask(col: Column, mask: [bool]) -> Column {
    var c = col
    c.null_mask = mask
    c
}

pub fn column_is_null(col: Column, i: usize) -> bool {
    if col.null_mask.len() == 0 { return false }
    if i >= col.null_mask.len() { return false }
    col.null_mask[i]
}

pub fn column_null_count(col: Column) -> usize {
    var n: usize = 0
    var i: usize = 0
    while i < col.null_mask.len() {
        if col.null_mask[i] { n = n + 1 }
        i = i + 1
    }
    n
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC check tests/stdlib/data/test_frame_nulls.sio`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/frame.sio tests/stdlib/data/test_frame_nulls.sio
git commit -m "feat(data): null-mask accessors (is_null, null_count, set_null_mask)"
```

### Task 3: Datetime dtype (5)

**Files:**
- Modify: `stdlib/data/frame.sio` (`column_datetime`, extend `column_len`, `column_dtype_str`, `column_is_numeric`)
- Test: `tests/stdlib/data/test_frame_datetime.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_frame_datetime.sio`:
```sounio
//@ check-only
// tests/stdlib/data/test_frame_datetime.sio
use data::frame::*

fn assert_datetime() -> bool {
    var e: [i64] = []
    e.push(0)                       // epoch
    e.push(1000000000)              // +1 s in ns
    let c = column_datetime("t", e, 0)
    if column_dtype_str(c) != "datetime" { return false }
    if column_len(c) != 2 { return false }
    if c.dt_unit != 0 { return false }
    if column_is_numeric(c) { return false }   // datetime is not numeric
    true
}

fn main() -> i32 {
    if !assert_datetime() { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC check tests/stdlib/data/test_frame_datetime.sio`
Expected: FAIL — `column_datetime` not defined.

- [ ] **Step 3: Implement**

Add the constructor to `stdlib/data/frame.sio` (near the other constructors):
```sounio
pub fn column_datetime(name: String, epoch: [i64], unit: i32) -> Column {
    var empty_f: [f64] = []
    var empty_s: [String] = []
    var empty_b: [bool] = []
    var empty_u: [f64] = []
    var empty_c: [f64] = []
    var empty_nm: [bool] = []
    Column {
        name: name,
        dtype: 5,
        float_data: empty_f,
        int_data: epoch,
        string_data: empty_s,
        bool_data: empty_b,
        uncert_data: empty_u,
        conf_data: empty_c,
        null_mask: empty_nm,
        units: "",
        provenance: "",
        dt_unit: unit,
    }
}
```
Extend `column_len` — add before the final `0`:
```sounio
    if col.dtype == 5 { return col.int_data.len() }
```
Extend `column_dtype_str` — add before its final return:
```sounio
    if col.dtype == 5 { return "datetime" }
```
Confirm `column_is_numeric` returns `false` for dtype 5 (it should already, since it only trues on 0/1/4 — verify and, if it trues on `>= 1`, add an explicit `if col.dtype == 5 { return false }`).

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC check tests/stdlib/data/test_frame_datetime.sio`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/frame.sio tests/stdlib/data/test_frame_datetime.sio
git commit -m "feat(data): datetime dtype (epoch-ns i64, unit tag)"
```

### Task 4: Metadata + null mask survive filter/slice

**Files:**
- Modify: `stdlib/data/frame.sio` (`dataframe_filter`, `dataframe_slice`)
- Test: extend `tests/stdlib/data/test_frame_metadata.sio`

- [ ] **Step 1: Write the failing test**

Append to `tests/stdlib/data/test_frame_metadata.sio` a new function and call it from `main`:
```sounio
fn assert_meta_survives_filter() -> bool {
    var d: [f64] = []
    d.push(1.0)
    d.push(2.0)
    d.push(3.0)
    var c = column_float("x", d)
    c.units = "mg"
    c.provenance = "unit-test"
    var nm: [bool] = []
    nm.push(false)
    nm.push(true)
    nm.push(false)
    c = column_set_null_mask(c, nm)
    var df = dataframe_new()
    df = dataframe_add_column(df, c)
    var mask: [bool] = []
    mask.push(true)
    mask.push(false)
    mask.push(true)
    let out = dataframe_filter(df, mask)
    let oc = dataframe_get_column(out, "x")
    if oc.units != "mg" { return false }
    if oc.provenance != "unit-test" { return false }
    // rows 0 and 2 kept; both were non-null -> 0 nulls
    if column_null_count(oc) != 0 { return false }
    true
}
```
And in `main` add before `0`:
```sounio
    if !assert_meta_survives_filter() { return 1 }
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC run tests/stdlib/data/test_frame_metadata.sio`
Expected: exit 1 (units dropped by filter).

- [ ] **Step 3: Add a dtype-agnostic metadata post-pass**

In `stdlib/data/frame.sio`, at the END of `dataframe_filter` (after `new_cols` is fully built, before it constructs the returned `DataFrame`), insert:
```sounio
    // carry per-column metadata + filtered null mask (dtype-agnostic)
    var k: usize = 0
    while k < new_cols.len() {
        let src = df.columns[k]
        var nc = new_cols[k]
        nc.units = src.units
        nc.provenance = src.provenance
        nc.dt_unit = src.dt_unit
        if src.null_mask.len() > 0 {
            var new_nm: [bool] = []
            var j: usize = 0
            while j < src.null_mask.len() && j < mask.len() {
                if mask[j] { new_nm.push(src.null_mask[j]) }
                j = j + 1
            }
            nc.null_mask = new_nm
        }
        new_cols[k] = nc
        k = k + 1
    }
```
Add the equivalent post-pass to `dataframe_slice`, but slice the mask by `[start, end)` instead of a bool mask:
```sounio
    var k: usize = 0
    while k < new_cols.len() {
        let src = df.columns[k]
        var nc = new_cols[k]
        nc.units = src.units
        nc.provenance = src.provenance
        nc.dt_unit = src.dt_unit
        if src.null_mask.len() > 0 {
            var new_nm: [bool] = []
            var j: usize = start
            while j < end && j < src.null_mask.len() {
                new_nm.push(src.null_mask[j])
                j = j + 1
            }
            nc.null_mask = new_nm
        }
        new_cols[k] = nc
        k = k + 1
    }
```
Also add a `dtype == 5` branch to `dataframe_filter`/`dataframe_slice`'s per-dtype rebuild that mirrors the `dtype == 1` (int) branch but constructs via `column_datetime(col.name, new_data, col.dt_unit)`.

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC run tests/stdlib/data/test_frame_metadata.sio`
Expected: exit 0.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/frame.sio tests/stdlib/data/test_frame_metadata.sio
git commit -m "feat(data): preserve units/provenance/null-mask through filter and slice"
```

### Task 5: Freeze + document the hub API

**Files:**
- Create: `stdlib/data/README.md`

- [ ] **Step 1: Write the API contract**

Create `stdlib/data/README.md` documenting the frozen surface that Phases 1–5 target. Include: the `Column` field layout and dtype table (0=float,1=int,2=string,3=bool,4=epistemic,5=datetime); null-mask convention (`true` = missing, empty = none); `dt_unit` codes; and the full public function list from `frame.sio` (constructors, `column_is_null`/`column_null_count`/`column_set_null_mask`, `dataframe_*`). State the stability guarantee: **these signatures do not change without a spec amendment; readers/writers depend on them.**

- [ ] **Step 2: Verify it reflects reality**

Run: `grep -n -E "^pub fn |^pub struct " stdlib/data/frame.sio`
Cross-check every `pub` symbol appears in the README.

- [ ] **Step 3: Commit**
```bash
git add stdlib/data/README.md
git commit -m "docs(data): freeze and document the DataFrame hub API"
```

### Task 6: Dissertation-critical regression gate

**Files:** none (verification only)

- [ ] **Step 1: Run the pbpk/petab tests that depend on existing data code**

Run:
```bash
$SOUC check tests/stdlib/darwin_pbpk/test_observed_petab_fit_e2e.sio
$SOUC check tests/packages/package_pbpk_gum_workflow.sio
```
Expected: both PASS (Phase 0 was additive; these must be unaffected).

- [ ] **Step 2: If either fails**

STOP. The Phase 0 change was not additive. Do not proceed to Phase 1. Report which symbol broke and why. (Halt-with-report is a valid deliverable.)

---

## PHASE 1 — Text writers

All writers return a `String` (pure, no IO) so they are trivially testable; a thin file-writing wrapper can come later. Rendering rules:
- Null cell → CSV empty field; JSON `null`.
- `f64` → `(v as String)`; `i64` → `(v as String)`; `bool` → literal `"true"`/`"false"`; datetime → `(int_data[i] as String)` (raw epoch for now).

### Task 7: CSV writer (RFC-4180)

**Files:**
- Create: `stdlib/data/write_csv.sio`
- Test: `tests/stdlib/data/test_write_csv.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_write_csv.sio`:
```sounio
// tests/stdlib/data/test_write_csv.sio
use data::frame::*
use data::write_csv::*

fn build() -> DataFrame {
    var a: [f64] = []
    a.push(1.0)
    a.push(2.0)
    var s: [String] = []
    s.push("hi")
    s.push("a,b")            // must be quoted
    var df = dataframe_new()
    df = dataframe_add_column(df, column_float("x", a))
    df = dataframe_add_column(df, column_string("label", s))
    df
}

fn main() -> i32 with Mut {
    let df = build()
    let csv = dataframe_to_csv(df)
    let expected = "x,label\n1,hi\n2,\"a,b\"\n"
    if csv != expected { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC run tests/stdlib/data/test_write_csv.sio`
Expected: FAIL — `dataframe_to_csv` not defined.

- [ ] **Step 3: Implement the writer**

Create `stdlib/data/write_csv.sio`:
```sounio
// stdlib/data/write_csv.sio — DataFrame -> RFC-4180 CSV string
use data::frame::*

fn csv_needs_quote(s: String) -> bool {
    // quote if contains comma, quote, CR or LF
    var i: usize = 0
    let bytes = s.as_bytes()
    while i < bytes.len() {
        let b = bytes[i]
        if b == 44 { return true }      // ,
        if b == 34 { return true }      // "
        if b == 10 { return true }      // \n
        if b == 13 { return true }      // \r
        i = i + 1
    }
    false
}

fn csv_escape(s: String) -> String {
    if !csv_needs_quote(s) { return s }
    var out: String = "\""
    let bytes = s.as_bytes()
    var i: usize = 0
    while i < bytes.len() {
        let b = bytes[i]
        if b == 34 { out = out ++ "\"\"" } else { out = out ++ byte_to_string(b) }
        i = i + 1
    }
    out ++ "\""
}

fn cell_to_string(col: Column, i: usize) -> String {
    if column_is_null(col, i) { return "" }
    if col.dtype == 0 { return (col.float_data[i] as String) }
    if col.dtype == 1 { return (col.int_data[i] as String) }
    if col.dtype == 2 { return csv_escape(col.string_data[i]) }
    if col.dtype == 3 { if col.bool_data[i] { return "true" } else { return "false" } }
    if col.dtype == 4 { return (col.float_data[i] as String) }
    if col.dtype == 5 { return (col.int_data[i] as String) }
    ""
}

pub fn dataframe_to_csv(df: DataFrame) -> String {
    var out: String = ""
    // header
    let names = dataframe_column_names(df)
    var c: usize = 0
    while c < names.len() {
        if c > 0 { out = out ++ "," }
        out = out ++ csv_escape(names[c])
        c = c + 1
    }
    out = out ++ "\n"
    // rows
    let nrows = dataframe_nrows(df)
    var r: usize = 0
    while r < nrows {
        var cc: usize = 0
        while cc < df.columns.len() {
            if cc > 0 { out = out ++ "," }
            out = out ++ cell_to_string(df.columns[cc], r)
            cc = cc + 1
        }
        out = out ++ "\n"
        r = r + 1
    }
    out
}
```
> NOTE on `byte_to_string`/`as_bytes`: if `String.as_bytes()` or a single-byte→String helper is not available on the builtin `String`, replace `csv_escape`'s byte loop with `str`-module equivalents (`stdlib/str/lib.sio`: `str_from_literal`, `str_replace_char`, `str_contains`) — grep first: `grep -n "as_bytes\|fn byte_to_string" stdlib/**/*.sio`. Keep the RFC-4180 semantics identical (double embedded quotes, wrap in quotes when comma/quote/newline present).

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC run tests/stdlib/data/test_write_csv.sio`
Expected: exit 0.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/write_csv.sio tests/stdlib/data/test_write_csv.sio
git commit -m "feat(data): RFC-4180 CSV writer"
```

### Task 8: JSON writer (records + columnar)

**Files:**
- Create: `stdlib/data/write_json.sio`
- Test: `tests/stdlib/data/test_write_json.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_write_json.sio`:
```sounio
// tests/stdlib/data/test_write_json.sio
use data::frame::*
use data::write_json::*

fn build() -> DataFrame {
    var a: [f64] = []
    a.push(1.0)
    a.push(2.0)
    var df = dataframe_new()
    df = dataframe_add_column(df, column_float("x", a))
    df
}

fn main() -> i32 with Mut {
    let df = build()
    let recs = dataframe_to_json_records(df)
    if recs != "[{\"x\":1},{\"x\":2}]" { return 1 }
    let cols = dataframe_to_json_columns(df)
    if cols != "{\"x\":[1,2]}" { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC run tests/stdlib/data/test_write_json.sio`
Expected: FAIL — functions not defined.

- [ ] **Step 3: Implement**

Create `stdlib/data/write_json.sio`. Reuse a `json_cell(col, i)` that emits `null` when `column_is_null`, a JSON string (quoted, backslash-escaped) for dtype 2, `true`/`false` for dtype 3, and the numeric cast otherwise. Implement `dataframe_to_json_records` (array of `{ "name": value, ... }` objects, one per row) and `dataframe_to_json_columns` (`{ "name": [values...], ... }`). Follow the exact loop structure of `dataframe_to_csv` (header/rows), swapping delimiters/braces. Full code mirrors Task 7's structure — write it out in the file; do not abbreviate.

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC run tests/stdlib/data/test_write_json.sio`
Expected: exit 0.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/write_json.sio tests/stdlib/data/test_write_json.sio
git commit -m "feat(data): JSON writer (records + columnar)"
```

### Task 9: Table pretty-printer (terminal + markdown)

**Files:**
- Create: `stdlib/data/table.sio`
- Test: `tests/stdlib/data/test_table.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_table.sio`:
```sounio
// tests/stdlib/data/test_table.sio
use data::frame::*
use data::table::*

fn build() -> DataFrame {
    var a: [f64] = []
    a.push(1.0)
    var df = dataframe_new()
    df = dataframe_add_column(df, column_float("x", a))
    df
}

fn main() -> i32 with Mut {
    let df = build()
    let md = dataframe_to_markdown(df)
    if md != "| x |\n| --- |\n| 1 |\n" { return 1 }
    0
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC run tests/stdlib/data/test_table.sio`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `stdlib/data/table.sio` with `dataframe_to_markdown(df) -> String` (header row `| a | b |`, separator `| --- | --- |`, then data rows) and `dataframe_to_table(df) -> String` (space-aligned: compute max width per column over header+cells in a first pass, left-pad in a second pass). Reuse `cell_to_string` semantics from Task 7 (re-implement locally — the writer files are independent; do not cross-import a non-`pub` helper).

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC run tests/stdlib/data/test_table.sio`
Expected: exit 0.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/table.sio tests/stdlib/data/test_table.sio
git commit -m "feat(data): markdown + aligned table renderers"
```

### Task 10: Provenance artifact writer (data + `.meta.json` sidecar)

**Files:**
- Create: `stdlib/data/artifact.sio`
- Test: `tests/stdlib/data/test_artifact.sio`

- [ ] **Step 1: Write the failing test**

Create `tests/stdlib/data/test_artifact.sio`:
```sounio
// tests/stdlib/data/test_artifact.sio
use data::frame::*
use data::artifact::*

fn build() -> DataFrame {
    var a: [f64] = []
    a.push(1.0)
    a.push(2.0)
    var col = column_float("dose", a)
    col.units = "mg"
    col.provenance = "vancomycin_run_2026"
    var df = dataframe_new()
    df = dataframe_add_column(df, col)
    df
}

fn main() -> i32 with Mut {
    let df = build()
    let meta = artifact_meta_json(df)
    // schema records name, dtype, units, provenance, and row/col counts
    if !str_has(meta, "\"units\":\"mg\"") { return 1 }
    if !str_has(meta, "\"provenance\":\"vancomycin_run_2026\"") { return 1 }
    if !str_has(meta, "\"nrows\":2") { return 1 }
    if !str_has(meta, "\"ncols\":1") { return 1 }
    0
}

fn str_has(hay: String, needle: String) -> bool {
    // substring test via the str module; see NOTE in Task 7 for helper resolution
    let h = str_from_literal(hay)
    let n = str_from_literal(needle)
    str_contains(&h, &n)
}
```
> If `str_from_literal`/`str_contains` signatures differ, adapt the `str_has` helper to whatever `stdlib/str/lib.sio` exposes (verified present: `str_from_literal`, `str_contains`).

- [ ] **Step 2: Run to verify it fails**

Run: `$SOUC run tests/stdlib/data/test_artifact.sio`
Expected: FAIL — `artifact_meta_json` not defined.

- [ ] **Step 3: Implement**

Create `stdlib/data/artifact.sio` with:
- `artifact_meta_json(df) -> String` — emit `{"nrows":N,"ncols":M,"columns":[{"name":..,"dtype":"..","units":"..","provenance":".."},...],"checksum":"<hex>"}`. dtype uses `column_dtype_str`. Checksum: sum a simple rolling hash over the CSV body (call `dataframe_to_csv` from Task 7) — a stable content fingerprint, documented as non-cryptographic.
- `artifact_write(df, path) -> i32 with IO` — writes the CSV body to `path` and the metadata to `path ++ ".meta.json"`, using the existing file-write primitive (grep `stdlib/io` / `stdlib/os` for the write helper, e.g. `write_file`/`fs_write`; do NOT invent one).

- [ ] **Step 4: Run to verify it passes**

Run: `$SOUC run tests/stdlib/data/test_artifact.sio`
Expected: exit 0.

- [ ] **Step 5: Commit**
```bash
git add stdlib/data/artifact.sio tests/stdlib/data/test_artifact.sio
git commit -m "feat(data): provenance artifact writer (data + .meta.json sidecar)"
```

### Task 11: Phase-1 gate script

**Files:**
- Create: `scripts/data_io_gate.sh`

- [ ] **Step 1: Write the gate**

Create `scripts/data_io_gate.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC=./bin/souc
fail=0
for t in tests/stdlib/data/test_frame_nulls.sio \
         tests/stdlib/data/test_frame_datetime.sio \
         tests/stdlib/data/test_frame_metadata.sio \
         tests/stdlib/data/test_write_csv.sio \
         tests/stdlib/data/test_write_json.sio \
         tests/stdlib/data/test_table.sio \
         tests/stdlib/data/test_artifact.sio ; do
  echo "== $t =="
  if ! $SOUC run "$t"; then echo "FAIL: $t"; fail=1; fi
done
# regression: dissertation-critical path
for c in tests/stdlib/darwin_pbpk/test_observed_petab_fit_e2e.sio \
         tests/packages/package_pbpk_gum_workflow.sio ; do
  if ! $SOUC check "$c"; then echo "REGRESSION: $c"; fail=1; fi
done
exit $fail
```

- [ ] **Step 2: Run it**

Run: `bash scripts/data_io_gate.sh`
Expected: every test prints and exits 0.

- [ ] **Step 3: Commit**
```bash
chmod +x scripts/data_io_gate.sh
git add scripts/data_io_gate.sh
git commit -m "test(data): Phase 0+1 conformance gate (hub + writers)"
```

---

## PHASES 2–5 — Readers (roadmap; detailed plans authored post-freeze)

Each reader targets the **frozen hub API** from Task 5 and is verified by a **golden-fixture gate** (reference-tool output committed under `tests/stdlib/<fmt>/fixtures/`, generated once by a checked-in `gen_fixtures.py` — verification tooling, not the science path). Each gets its own dated plan document produced right before it is executed, following the same TDD granularity as Phases 0–1. Interfaces are pinned here so the phases compose.

### Phase 2 — CSV reader → typed DataFrame
- File: `stdlib/data/read_csv.sio`; entry `csv_read_to_frame(text: String, opts: CsvReadOpts) -> DataFrame`.
- Consolidate onto `stdlib/csv/parser.sio` (do not fork it). Add: type inference (int→float→bool→datetime→string precedence), missing→null-mask, RFC-4180 quoting, epistemic-column hook (`value±unc` or paired columns → `column_epistemic`).
- Gate: pandas-emitted CSVs (quoted fields, embedded newlines, blanks, mixed types) → expected typed frame.

### Phase 3 — Parquet reader + writer
- Files: `stdlib/compress/snappy.sio` (new, ~200 loc, LZ77 variant — write it first, unit-test against snappy spec vectors); `stdlib/parquet/{thrift.sio,read.sio,write.sio,mod.sio}`.
- Read: Thrift compact footer (FileMetaData/schema/row-groups/column-chunks) → plain + dictionary(RLE/bit-pack) + RLE encodings → codecs uncompressed/snappy/gzip/zstd → primitive types (BOOLEAN/INT32/INT64/FLOAT/DOUBLE/BYTE_ARRAY) + logical hints (string/date/timestamp) → hub dtypes + null-mask (definition levels).
- Write: uncompressed/gzip/zstd (no snappy needed to write); reuse the shared Thrift codec.
- Gate: pyarrow fixtures (each codec, dictionary on/off, with nulls) read → expected frame; + Sounio-written parquet re-read by pyarrow == original.
- New binary primitives likely needed in `stdlib/io/binary.sio`: `read_f32/f64_le`, LEB128/zigzag varint (additive).

### Phase 4 — netCDF-classic reader
- Files: `stdlib/netcdf/{read.sio,mod.sio}`; entry `netcdf_read_classic(bytes) -> DataFrame` (1-D vars → columns) with N-D vars as flat buffer + `shape` tag (spec §10 default).
- CDF-1/2/5 magic; header (dims/global-attrs/vars w/ offsets); big-endian decode; one unlimited (record) dim; `_FillValue` → null-mask.
- Doc caveat: classic only; netCDF-4 routes through Phase 5.
- Gate: netCDF4-emitted classic fixtures → expected frame/tensor.

### Phase 5 — HDF5 reader (contiguous + chunked/filtered)
- Files: `stdlib/hdf5/{superblock.sio,objheader.sio,btree.sio,heap.sio,filter.sio,dataset.sio,mod.sio}`.
- Split 5a (contiguous/uncompressed: superblock v0/v2/v3, object-header messages, datatype/dataspace/layout, group traversal via symbol-table + link msgs) → 5b (chunked via B-tree v1 chunk index + filter pipeline: gzip via existing `inflate`, shuffle).
- Datatypes fixed/float/string, LE + BE → hub dtypes; N-D → flat buffer + shape; 1-D → column.
- Gate: h5py fixtures — contiguous, chunked, gzip-filtered, shuffle+gzip, big-endian — → expected arrays/frame.

**Extend `scripts/data_io_gate.sh`** to include each format's fixture gate as it lands.

---

## Self-review notes
- **Spec coverage:** hub null/datetime/units/provenance (Tasks 1–4), frozen API (Task 5), additive-consolidation regression (Task 6), all writers incl. artifact (Tasks 7–10, target 4), conformance-gate wiring (Task 11), readers + fixtures (Phases 2–5), snappy (Phase 3), netCDF caveat (Phase 4), HDF5 5a/5b split (Phase 5). No spec section is unmapped.
- **Consistency:** field names (`null_mask`, `units`, `provenance`, `dt_unit`), dtype codes (5=datetime), and function names (`column_is_null`, `column_set_null_mask`, `column_null_count`, `column_datetime`, `dataframe_to_csv/json_*/markdown`, `artifact_meta_json`) are used identically across tasks.
- **Known unknowns flagged inline (not placeholders):** builtin `String` byte access (`as_bytes`/`byte_to_string`) and the file-write primitive are marked "grep to resolve, fall back to `stdlib/str`/`stdlib/io`" with the verified fallback named — because these depend on compiler-surface details that must be confirmed at the machine, not guessed.
```
