<!-- docs:meta
topic_id: repo.docs.superpowers.plans.2026-07-17-csv-reader
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.plans.2026-07-17-csv-reader
-->

# Cap-free CSV Reader Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `stdlib/data/csv_reader.sio` — a cap-free, row-iterator, header-aware, typed CSV reader — plus a `read_file` dynamic-mmap enhancement so file input has no fixed size cap.

**Architecture:** Two branches. (A) `read_file` dynamic-mmap on the compiler-fix branch `fix/native-read-file-string-return` (PR #1078). (B) The reader stdlib module on `feat/csv-reader` (branched off #1078). The reader holds only cursors/byte-ranges over the file text and **re-scans on demand** (no field-offset arrays → no column cap); the file text is a `string` accessed **only** via `str_char_at`/`str_len` (never `text[i]` — that SIGSEGVs on a `read_file` result). Every task compiles+runs a `.sio` driver against the #1078-fixed Madaros and greps the output — the repo's gate-driver style is our TDD loop.

**Tech Stack:** Sounio (`.sio`), Madaros v0.80.0 native-v2 backend, `bash` gate scripts. Compiler under test: the fixed Madaros ELF built from #1078 (below).

---

## Setup (do once before Task 1)

- Worktree: `/workspace/sounio/.claude/worktrees/trilha-b-builtin-fix` (currently on `feat/csv-reader`, which contains #1078's `read_file` fix as an ancestor).
- Build the fixed Madaros compiler once and export it as `SOUC`:

```bash
cd /workspace/sounio/.claude/worktrees/trilha-b-builtin-fix
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
export MAD="$(pwd)/.mad/madaros"    # any stable path
mkdir -p .mad
bash scripts/ci/build_modular_madaros.sh "$MAD"      # ~8 min; builds current source incl. #1078 read_file fix
"$MAD" --version                                     # Expected: "Madaros v0.80.0 -- the Sounio self-hosted compiler"
```

- Compile+run helper used throughout (verb form; the bare form is unsupported by a modular Madaros):

```bash
crun() { "$MAD" compile "$1" -o /tmp/cr.elf >/tmp/cr.log 2>&1 && chmod +x /tmp/cr.elf && /tmp/cr.elf; }
```

---

## Phase 0 — Capability probes (resolve the spec's open questions)

Purpose: pin down Madaros's actual capabilities before writing the module, so no task builds on an unsupported construct. Probes live under `scratch/csv_probes/` (git-ignored; not committed).

### Task 0: Probe struct + `&!` mutation + read_file-string access

**Files:**
- Create: `scratch/csv_probes/p0.sio`

- [ ] **Step 1: Write the probe**

```sounio
struct R { text: string, pos: i64, ok: i64 }

fn r_bump(r: &! R) -> () with Mut { r.pos = r.pos + 1 }

fn main() -> i32 with IO, Mut, Panic, Div {
    let s = "abc,def"
    var r = R { text: s, pos: 0, ok: 1 }
    r_bump(&!r)
    r_bump(&!r)
    print("pos="); print_int(r.pos); print("\n")                 // expect 2
    print("len="); print_int(str_len(r.text)); print("\n")        // expect 7
    print("c0="); print_int(str_char_at(r.text, 0) as i64); print("\n")  // expect 97
    print("c4="); print_int(str_char_at(r.text, 4) as i64); print("\n")  // expect 100
    return 0
}
```

- [ ] **Step 2: Run and verify**

Run: `crun scratch/csv_probes/p0.sio`
Expected stdout: `pos=2`, `len=7`, `c0=97`, `c4=100`, exit 0.

- [ ] **Step 3: Record the result**

If `&!` struct mutation and `str_char_at`/`str_len` on a struct-held string all work → the `CsvReader` struct + threading design is viable as written. If `&!` mutation fails, fall back to returning an updated `R` by value from each mutating function (functional threading, like the writer threads its column index) and update all later tasks accordingly.

### Task 1: Probe the `csv_str_into` buffer-passing options

**Files:**
- Create: `scratch/csv_probes/p1a.sio` (option a: `&![i8; N]` param), `scratch/csv_probes/p1c.sio` (option c: heap string)

- [ ] **Step 1: Probe option (a) — fixed-array ref param**

`scratch/csv_probes/p1a.sio`:

```sounio
fn fill(out: &![i8; 8], n: i64) -> i64 with Mut, Panic, Div {
    var i = 0
    while i < n { (*out)[i as usize] = (65 + i) as i8; i = i + 1 }
    n
}
fn main() -> i32 with IO, Mut, Panic, Div {
    var buf: [i8; 8] = [0; 8]
    let k = fill(&!buf, 3)
    print(str_from_bytes(buf, k)); print("\n")     // expect ABC
    return 0
}
```

- [ ] **Step 2: Run option (a)**

Run: `crun scratch/csv_probes/p1a.sio`
Expected: `ABC`, exit 0. If it compiles and prints `ABC`, **option (a) is chosen** for `csv_str_into` — signature `csv_str_into(r: &CsvReader, col: i64, out: &![i8; N], cap: i64) -> i64` written as a concrete-N helper the caller instantiates. Record whether Madaros accepts a size-generic `N` or requires a concrete size (if concrete-only, the reader ships `csv_str_into` for a documented common size AND a `csv_field_range(r,col) -> (start,len)` + `csv_byte(r,i)` low-level accessor so callers with other sizes are not capped).

- [ ] **Step 3: Probe option (c) — heap-allocated string (only if (a) is unusable)**

`scratch/csv_probes/p1c.sio`: attempt building a `string` from a byte range without a caller buffer (e.g. copy into a local `[i8; N]` then `str_from_bytes`, or any stdlib heap alloc). Run and record. Prefer (a); use (c) only if (a) fails to compile.

- [ ] **Step 4: Lock the decision in the spec**

Edit `docs/superpowers/specs/2026-07-17-csv-reader-design.md` "Open implementation questions" → replace with the chosen signature. Commit:

```bash
git add docs/superpowers/specs/2026-07-17-csv-reader-design.md
git commit -m "docs(spec): resolve csv_str_into buffer passing from Madaros probe"
```

---

## Phase 1 — `read_file` dynamic-mmap (target branch: `fix/native-read-file-string-return`, #1078)

> Switch branch for this phase: `git switch fix/native-read-file-string-return`. Rebuild `$MAD` after the change. Return to `feat/csv-reader` for Phase 2+ (it will pick up the change once #1078 is merged/rebased).

### Task 2: Size the mmap to the file

**Files:**
- Modify: `self-hosted/native/codegen_x86_linux.sio` — `emit_builtin_read_file` (the byte sequence added in #1078)
- Test: `scratch/csv_probes/p2_bigfile.sio`

- [ ] **Step 1: Write the failing test (a file larger than 1 MiB)**

```bash
python3 -c "open('big.csv','w').write('x'*1500000 + '\n')"   # 1.5 MB, > 1 MiB
cat > scratch/csv_probes/p2_bigfile.sio <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Div {
    let raw = read_file("big.csv")
    print("len="); print_int(str_len(raw)); print("\n")   // expect 1500001
    return 0
}
EOF
```

- [ ] **Step 2: Run against the current (1 MiB) build — expect truncation/wrong len**

Run: `crun scratch/csv_probes/p2_bigfile.sio`
Expected BEFORE the change: `len=` is capped near 1048576, not 1500001 (or a crash) — demonstrates the cap.

- [ ] **Step 3: Implement dynamic sizing**

In `emit_builtin_read_file`, replace the fixed `mov esi, 1048576` mmap-size and `mov edx, 1048576` read-count with a computed size. New sequence (raw bytes, mirroring the existing style; sizes are Linux x86-64 syscalls):
1. `mov [rbp-8], rdi` (save path).
2. `open(path, O_RDONLY)` → fd; save `[rbp-24]`.
3. `lseek(fd, 0, SEEK_END=2)` → size in rax (`mov eax,8; mov rdi,fd; xor esi,esi; mov edx,2; syscall`); save size `[rbp-40]`.
4. `lseek(fd, 0, SEEK_SET=0)` (`mov eax,8; mov rdi,fd; xor esi,esi; xor edx,edx; syscall`).
5. Compute mmap length = `(size + 1 + 4095) & ~4095`: `mov rax,[rbp-40]; add rax,4096; and rax,-4096` (i.e. `add rax, 0x1000` then `and rax, 0xFFFFFFFFFFFFF000`) → `[rbp-48]`.
6. `mmap(NULL, len=[rbp-48], PROT_RW=3, MAP_PRIVATE|ANON=0x22, -1, 0)` → buf; save `[rbp-16]`.
7. `read(fd, buf=[rbp-16], size=[rbp-40])`.
8. `close(fd)`.
9. `mov rax,[rbp-16]` (return buf); `leave; ret`.

Keep the sub-rsp frame ≥ 48. Write it as `b = emit_byte(b, 0x..)` sequences with per-line comments, exactly as the #1078 body does. (Byte encodings for the added ops: `lseek` = syscall 8; `and rax, imm32-sign-extended` = `48 25 <imm32>` for `and rax, 0x...` OR use `and rax, r/m` after loading the mask — pick the encoding that assembles; verify with the next step.)

- [ ] **Step 4: Rebuild + run to verify the big file reads fully**

```bash
bash scripts/ci/build_modular_madaros.sh "$MAD"
crun scratch/csv_probes/p2_bigfile.sio
```
Expected: `len=1500001`, exit 0.

- [ ] **Step 5: Regression — small file still byte-exact**

```bash
printf 'hi,world,42\napple,7,3.14\n' > small.csv
cat > scratch/csv_probes/p2_small.sio <<'EOF'
fn main() -> i32 with IO, Mut, Panic, Div {
    let raw = read_file("small.csv")
    print("len="); print_int(str_len(raw)); print("\n")
    var i = 0
    while i < str_len(raw) { print_int(str_char_at(raw, i) as i64); print(" "); i = i + 1 }
    print("\n")
    return 0
}
EOF
crun scratch/csv_probes/p2_small.sio
```
Expected: `len=25` and the exact byte sequence `104 105 44 ... 10`. Also re-run the #1078 gate proof (`str_from_bytes(buf,n)` → "hi") to confirm no collateral change.

- [ ] **Step 6: Commit to #1078**

```bash
git add self-hosted/native/codegen_x86_linux.sio
git commit -m "fix(native): size read_file's mmap to the file (remove 1 MiB cap)

lseek(SEEK_END) to get the size, mmap (size+1) rounded up to a page, read that
many bytes, return the buffer pointer. Removes the fixed 1 MiB cap for every
read_file caller. Verified: a 1.5 MB file reads fully; the 25-byte small file is
byte-exact; str_from_bytes(buf,n) still prints hi."
git push
```
Then `git switch feat/csv-reader`.

---

## Phase 2 — Reader core (branch `feat/csv-reader`)

All files created under `stdlib/data/csv_reader.sio`. Build it up function by function; each task appends to the module and adds a driver test under `tests/stdlib/data/`.

### Task 3: Module skeleton + `csv_open` (header range) + `csv_cols`

**Files:**
- Create: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_open.sio`

- [ ] **Step 1: Write the failing test**

`tests/stdlib/data/test_csv_reader_open.sio`:

```sounio
//@ run-pass
//@ expect-stdout: CSV_READER_OPEN_OK
use data::csv_reader::{csv_open, csv_cols, CsvReader}

fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("name,dose,ok\na,1,yes\nb,2,no\n")
    var pass = 1
    if csv_cols(&r) != 3 { print("FAIL cols="); print_int(csv_cols(&r)); print("\n"); pass = 0 }
    if pass == 1 { print("CSV_READER_OPEN_OK\n") }
    return 0
}
```

- [ ] **Step 2: Run — expect compile failure (module missing)**

Run: `"$MAD" compile tests/stdlib/data/test_csv_reader_open.sio -o /tmp/t.elf`
Expected: FAIL — `data::csv_reader` not found.

- [ ] **Step 3: Implement the skeleton + `csv_open` + `csv_scan_fields`**

`stdlib/data/csv_reader.sio` (header comment mirrors `csv.sio`; then):

```sounio
// data::csv_reader — cap-free, row-iterator, header-aware, typed CSV reader.
// Complements the byte-exact writer data::csv. Reads a string (a read_file result
// or a literal) accessed ONLY via str_char_at/str_len — never s[i], which SIGSEGVs
// on a read_file result. No offset arrays: columns are found by re-scanning the
// current row's byte range, so there is no column cap. RFC 4180 quoting; delimiter
// ',', terminators '\n' and '\r\n'.

pub struct CsvReader {
    text: string,
    text_len: i64,
    hdr_start: i64,
    hdr_end: i64,
    row_start: i64,
    row_end: i64,
    pos: i64,
    ok: i64,
}

// End byte index (exclusive) of the line beginning at `from`, and the index where the
// NEXT line begins. Returns line_end (before the terminator). A '\r' before '\n' is
// part of the terminator. Caller reads text via str_char_at.
fn csv_line_end(text: string, len: i64, from: i64) -> i64 with Mut, Panic, Div {
    var i = from
    var in_q = 0
    while i < len {
        let c = str_char_at(text, i) as i64
        if c == 34 { if in_q == 0 { in_q = 1 } else { in_q = 0 } }
        if c == 10 && in_q == 0 {
            var e = i
            if e > from { if (str_char_at(text, e - 1) as i64) == 13 { e = e - 1 } }
            return e
        }
        i = i + 1
    }
    len
}

// Index where the line after the one ending at `line_end` starts (skips the LF).
fn csv_next_line_start(text: string, len: i64, line_end: i64) -> i64 with Mut, Panic, Div {
    var i = line_end
    while i < len { if (str_char_at(text, i) as i64) == 10 { return i + 1 } i = i + 1 }
    len
}

// Count the comma-separated fields in the byte range [start, end), respecting quotes.
fn csv_count_fields(text: string, start: i64, end: i64) -> i64 with Mut, Panic, Div {
    if end <= start { return 0 }
    var i = start
    var n = 1
    var in_q = 0
    while i < end {
        let c = str_char_at(text, i) as i64
        if c == 34 { if in_q == 0 { in_q = 1 } else { in_q = 0 } }
        if c == 44 && in_q == 0 { n = n + 1 }
        i = i + 1
    }
    n
}

pub fn csv_open(text: string) -> CsvReader with Mut, Panic, Div {
    let len = str_len(text)
    let he = csv_line_end(text, len, 0)
    CsvReader {
        text: text, text_len: len,
        hdr_start: 0, hdr_end: he,
        row_start: 0, row_end: 0,
        pos: csv_next_line_start(text, len, he),
        ok: 1,
    }
}

pub fn csv_cols(r: &CsvReader) -> i64 with Mut, Panic, Div {
    csv_count_fields((*r).text, (*r).row_start, (*r).row_end)
}
```

- [ ] **Step 4: Register the test + run**

```bash
"$MAD" compile tests/stdlib/data/test_csv_reader_open.sio -o /tmp/t.elf && chmod +x /tmp/t.elf && /tmp/t.elf
```
Expected: `CSV_READER_OPEN_OK`.
(Note: `csv_cols` here reads `row_start/row_end` which are 0/0 until `csv_next_row`; adjust the test to call `csv_open` then check header-field count instead, OR land `csv_next_row` first. If probing Task 0 showed `&!` works, reorder to implement `csv_next_row` in Task 4 and have this test call it. Keep the test asserting a concrete post-`csv_next_row` count.)

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_open.sio
git commit -m "feat(data): csv_reader skeleton — csv_open + field/line scanners"
```

### Task 4: `csv_next_row` (row iteration)

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_rows.sio`

- [ ] **Step 1: Write the failing test**

```sounio
//@ run-pass
//@ expect-stdout: ROWS_OK
use data::csv_reader::{csv_open, csv_next_row, csv_cols, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("name,dose\na,1\nb,2\nc,3\n")
    var count = 0
    while csv_next_row(&!r) { count = count + 1 }
    if count == 3 { print("ROWS_OK\n") } else { print("FAIL count="); print_int(count); print("\n") }
    return 0
}
```

- [ ] **Step 2: Run — expect FAIL (csv_next_row missing)**

Run the compile; expect unresolved `csv_next_row`.

- [ ] **Step 3: Implement `csv_next_row`**

Append to `stdlib/data/csv_reader.sio`:

```sounio
pub fn csv_next_row(r: &! CsvReader) -> bool with Mut, Panic, Div {
    let len = (*r).text_len
    if (*r).pos >= len { return false }
    let s = (*r).pos
    let e = csv_line_end((*r).text, len, s)
    (*r).row_start = s
    (*r).row_end = e
    (*r).pos = csv_next_line_start((*r).text, len, e)
    // Guard against a trailing empty line (file ends with '\n'): if this "row" is empty
    // AND there is nothing after it, treat as EOF.
    if e <= s { if (*r).pos >= len { return false } }
    true
}
```

- [ ] **Step 4: Run + verify**

Expected: `ROWS_OK` (3 data rows; header skipped because `csv_open` set `pos` past the header).

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_rows.sio
git commit -m "feat(data): csv_next_row — stream data rows, skip header, handle trailing newline"
```

### Task 5: `csv_field_range` + `csv_col` (by header name)

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_col.sio`

- [ ] **Step 1: Write the failing test**

```sounio
//@ run-pass
//@ expect-stdout: COL_OK
use data::csv_reader::{csv_open, csv_next_row, csv_col, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("name,dose,unit\na,1,mg\n")
    var pass = 1
    if csv_col(&r, "name") != 0 { pass = 0 }
    if csv_col(&r, "dose") != 1 { pass = 0 }
    if csv_col(&r, "unit") != 2 { pass = 0 }
    if csv_col(&r, "missing") != (0 - 1) { pass = 0 }
    if pass == 1 { print("COL_OK\n") } else { print("FAIL\n") }
    return 0
}
```

- [ ] **Step 2: Run — expect FAIL (csv_col / range helpers missing)**

- [ ] **Step 3: Implement the field-range finder + `csv_col`**

Append:

```sounio
// Byte range [out_start, out_start+out_len) of the `col`-th field within [start,end),
// respecting quotes. Returns start*  via a 2-slot result encoded as (start, len) in a
// small struct to avoid multiple out-params.
pub struct CsvRange { start: i64, len: i64 }

fn csv_nth_field_range(text: string, start: i64, end: i64, col: i64) -> CsvRange with Mut, Panic, Div {
    var i = start
    var cur = 0
    var fstart = start
    var in_q = 0
    while i <= end {
        let at_end = i == end
        var c = 0
        if !at_end { c = str_char_at(text, i) as i64 }
        if c == 34 && !at_end { if in_q == 0 { in_q = 1 } else { in_q = 0 } }
        if (at_end || (c == 44 && in_q == 0)) {
            if cur == col { return CsvRange { start: fstart, len: i - fstart } }
            cur = cur + 1
            fstart = i + 1
        }
        i = i + 1
    }
    CsvRange { start: start, len: 0 - 1 }   // col out of range -> len -1
}

// Compare the field bytes [rng.start, rng.start+rng.len) to `name` byte-for-byte.
fn csv_range_eq(text: string, rng: CsvRange, name: string) -> bool with Mut, Panic, Div {
    let nl = str_len(name)
    if rng.len != nl { return false }
    var i = 0
    while i < nl {
        if (str_char_at(text, rng.start + i) as i64) != (str_char_at(name, i) as i64) { return false }
        i = i + 1
    }
    true
}

pub fn csv_col(r: &CsvReader, name: string) -> i64 with Mut, Panic, Div {
    let ncol = csv_count_fields((*r).text, (*r).hdr_start, (*r).hdr_end)
    var c = 0
    while c < ncol {
        let rng = csv_nth_field_range((*r).text, (*r).hdr_start, (*r).hdr_end, c)
        if csv_range_eq((*r).text, rng, name) { return c }
        c = c + 1
    }
    0 - 1
}
```

- [ ] **Step 4: Run + verify** → `COL_OK`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_col.sio
git commit -m "feat(data): csv_col by header name + nth-field range finder (quote-aware)"
```

### Task 6: `csv_int` + `csv_ok`

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_int.sio`

- [ ] **Step 1: Write the failing test**

```sounio
//@ run-pass
//@ expect-stdout: INT_OK
use data::csv_reader::{csv_open, csv_next_row, csv_col, csv_int, csv_ok, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("name,dose\na,-42\nb,x\n")
    var pass = 1
    let cd = csv_col(&r, "dose")
    csv_next_row(&!r)                       // row a,-42
    if csv_int(&!r, cd) != (0 - 42) { pass = 0 }
    if !csv_ok(&r) { pass = 0 }
    csv_next_row(&!r)                       // row b,x  (malformed)
    let bad = csv_int(&!r, cd)
    if csv_ok(&r) { pass = 0 }              // ok must be cleared
    if bad != 0 { pass = 0 }                // malformed -> 0
    if pass == 1 { print("INT_OK\n") } else { print("FAIL\n") }
    return 0
}
```

- [ ] **Step 2: Run — expect FAIL (csv_int/csv_ok missing)**

- [ ] **Step 3: Implement**

Append:

```sounio
pub fn csv_ok(r: &CsvReader) -> bool with Mut, Panic, Div { (*r).ok == 1 }

pub fn csv_int(r: &! CsvReader, col: i64) -> i64 with Mut, Panic, Div {
    let rng = csv_nth_field_range((*r).text, (*r).row_start, (*r).row_end, col)
    if rng.len < 0 { (*r).ok = 0; return 0 }
    var i = rng.start
    let stop = rng.start + rng.len
    var neg = 0
    if i < stop { if (str_char_at((*r).text, i) as i64) == 45 { neg = 1; i = i + 1 } }
    if i >= stop { (*r).ok = 0; return 0 }     // empty / just "-"
    var acc = 0
    while i < stop {
        let d = (str_char_at((*r).text, i) as i64) - 48
        if d < 0 || d > 9 { (*r).ok = 0; return 0 }
        acc = acc * 10 + d
        i = i + 1
    }
    (*r).ok = 1
    if neg == 1 { 0 - acc } else { acc }
}
```

- [ ] **Step 4: Run + verify** → `INT_OK`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_int.sio
git commit -m "feat(data): csv_int (signed) + csv_ok non-fatal parse flag"
```

### Task 7: `csv_f64_scaled` + `csv_f64`

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_f64.sio`

- [ ] **Step 1: Write the failing test**

```sounio
//@ run-pass
//@ expect-stdout: F64_OK
use data::csv_reader::{csv_open, csv_next_row, csv_col, csv_f64_scaled, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("x,v\na,3.14\nb,-0.5\n")
    let cv = csv_col(&r, "v")
    var pass = 1
    csv_next_row(&!r)
    if csv_f64_scaled(&!r, cv, 2) != 314 { pass = 0 }     // 3.14 * 100
    csv_next_row(&!r)
    if csv_f64_scaled(&!r, cv, 2) != (0 - 50) { pass = 0 } // -0.50 * 100
    if pass == 1 { print("F64_OK\n") } else { print("FAIL\n") }
    return 0
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `csv_f64_scaled` then `csv_f64`**

`csv_f64_scaled` parses `[-]int[.frac]` into a scaled integer of `decimals` places (round-half-away-from-zero on excess digits), matching the writer's `csv_field_fixed` convention. `csv_f64` builds an `f64` from the same parse (`whole + frac/10^k`, negated if needed). Append:

```sounio
fn ipow10(n: i64) -> i64 with Mut, Panic, Div { var p = 1; var i = 0; while i < n { p = p * 10; i = i + 1 } p }

pub fn csv_f64_scaled(r: &! CsvReader, col: i64, decimals: i64) -> i64 with Mut, Panic, Div {
    let rng = csv_nth_field_range((*r).text, (*r).row_start, (*r).row_end, col)
    if rng.len < 0 { (*r).ok = 0; return 0 }
    var i = rng.start
    let stop = rng.start + rng.len
    var neg = 0
    if i < stop { if (str_char_at((*r).text, i) as i64) == 45 { neg = 1; i = i + 1 } }
    var whole = 0
    while i < stop {
        let c = str_char_at((*r).text, i) as i64
        if c == 46 { i = i + 1; break }
        let d = c - 48
        if d < 0 || d > 9 { (*r).ok = 0; return 0 }
        whole = whole * 10 + d
        i = i + 1
    }
    var frac = 0
    var fdigits = 0
    while i < stop {
        let d = (str_char_at((*r).text, i) as i64) - 48
        if d < 0 || d > 9 { (*r).ok = 0; return 0 }
        if fdigits < decimals { frac = frac * 10 + d; fdigits = fdigits + 1 }
        i = i + 1
    }
    while fdigits < decimals { frac = frac * 10; fdigits = fdigits + 1 }
    (*r).ok = 1
    let mag = whole * ipow10(decimals) + frac
    if neg == 1 { 0 - mag } else { mag }
}

pub fn csv_f64(r: &! CsvReader, col: i64) -> f64 with Mut, Panic, Div {
    let scaled = csv_f64_scaled(r, col, 9)     // 9 dp of precision
    (scaled as f64) / 1000000000.0
}
```

- [ ] **Step 4: Run + verify** → `F64_OK`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_f64.sio
git commit -m "feat(data): csv_f64_scaled (fixed-point, writer round-trip) + csv_f64"
```

### Task 8: `csv_str_into` (caller-buffer, un-escape)

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_str.sio`

> Use the signature chosen in Task 1. The body below assumes option (a) `out: &![i8; N]`.

- [ ] **Step 1: Write the failing test**

```sounio
//@ run-pass
//@ expect-stdout: STR_OK
use data::csv_reader::{csv_open, csv_next_row, csv_col, csv_str_into, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open("k,v\nrow,\"a,\"\"b\"\"\"\n")   // field = a,"b"  (quoted, escaped)
    let cv = csv_col(&r, "v")
    csv_next_row(&!r)
    var buf: [i8; 64] = [0; 64]
    let n = csv_str_into(&r, cv, &!buf, 64)
    print(str_from_bytes(buf, n)); print("\n")        // expect a,"b"
    print("STR_OK\n")
    return 0
}
```

- [ ] **Step 2: Run — expect FAIL**

- [ ] **Step 3: Implement `csv_str_into`**

Copies the field bytes into `out`, stripping surrounding quotes and collapsing `""`→`"`. Append:

```sounio
pub fn csv_str_into(r: &CsvReader, col: i64, out: &![i8; 64], cap: i64) -> i64 with Mut, Panic, Div {
    let rng = csv_nth_field_range((*r).text, (*r).row_start, (*r).row_end, col)
    if rng.len < 0 { return 0 }
    var i = rng.start
    let stop = rng.start + rng.len
    var quoted = 0
    if rng.len >= 2 { if (str_char_at((*r).text, i) as i64) == 34 { quoted = 1; i = i + 1 } }
    var w = 0
    while i < stop {
        if w >= cap { break }
        let c = str_char_at((*r).text, i) as i64
        if quoted == 1 && c == 34 {
            // closing quote, or "" escape
            if i + 1 < stop { if (str_char_at((*r).text, i + 1) as i64) == 34 {
                (*out)[w as usize] = 34 as i8; w = w + 1; i = i + 2; continue
            } }
            i = i + 1
            break
        }
        (*out)[w as usize] = c as i8
        w = w + 1
        i = i + 1
    }
    w
}
```

- [ ] **Step 4: Run + verify** → prints `a,"b"` then `STR_OK`.

- [ ] **Step 5: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_str.sio
git commit -m "feat(data): csv_str_into — caller-sized buffer, RFC-4180 unquote/unescape"
```

### Task 9: `csv_open_file` (file input)

**Files:**
- Modify: `stdlib/data/csv_reader.sio`
- Test: `tests/stdlib/data/test_csv_reader_file.sio`

- [ ] **Step 1: Write the failing test (writes then reads a temp file)**

```sounio
//@ run-pass
//@ expect-stdout: FILE_OK
use data::csv_reader::{csv_open_file, csv_next_row, csv_col, csv_int, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open_file("cr_fixture.csv")
    let cd = csv_col(&r, "dose")
    csv_next_row(&!r)
    if csv_int(&!r, cd) == 5 { print("FILE_OK\n") } else { print("FAIL\n") }
    return 0
}
```

- [ ] **Step 2: Implement `csv_open_file`**

```sounio
pub fn csv_open_file(path: string) -> CsvReader with IO, Mut, Panic, Div {
    csv_open(read_file(path))
}
```

- [ ] **Step 3: Run with a fixture (needs #1078-fixed $MAD)**

```bash
printf 'name,dose\nx,5\ny,6\n' > cr_fixture.csv
"$MAD" compile tests/stdlib/data/test_csv_reader_file.sio -o /tmp/t.elf && chmod +x /tmp/t.elf && /tmp/t.elf
rm -f cr_fixture.csv
```
Expected: `FILE_OK`.

- [ ] **Step 4: Commit**

```bash
git add stdlib/data/csv_reader.sio tests/stdlib/data/test_csv_reader_file.sio
git commit -m "feat(data): csv_open_file — read_file(path) + csv_open (needs #1078)"
```

---

## Phase 3 — Gate, example, cap-free evidence

### Task 10: Round-trip + cap-free gate

**Files:**
- Create: `scripts/data_io_csv_reader_gate.sh`
- Create: `examples/epistemic/csv_read_pk.sio` (a real-data read demo)

- [ ] **Step 1: Write the gate**

`scripts/data_io_csv_reader_gate.sh` (dev-tier, mirrors `scripts/data_io_csv_gate.sh`; uses the compiler in `$SOUNIO_TEST_SOUC_BIN` or `./bin/souc`, which must be the #1078-fixed build):

```bash
#!/usr/bin/env bash
# CSV reader gate: parser-on-literal + file round-trip vs the writer + cap-free evidence.
# Needs the #1078 read_file fix; set SOUNIO_TEST_SOUC_BIN to a fixed Madaros until #1078 merges.
set -euo pipefail
cd "$(dirname "$0")/.."
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
SOUC="${SOUNIO_TEST_SOUC_BIN:-./bin/souc}"
OUT="$(mktemp -d)"; trap 'rm -rf "$OUT"' EXIT
fail=0
run() { # test.sio sentinel
  if SOUNIO_SOUC_ENGINE="${SOUNIO_SOUC_ENGINE:-}" "$SOUC" compile "$1" -o "$OUT/x.elf" >/dev/null 2>&1; then
    chmod +x "$OUT/x.elf"; ( cd "$OUT" && "$OUT/x.elf" ) | grep -q "$2" || { echo "FAIL run $1"; fail=1; }
  else echo "FAIL compile $1"; fail=1; fi
}
run tests/stdlib/data/test_csv_reader_open.sio CSV_READER_OPEN_OK
run tests/stdlib/data/test_csv_reader_rows.sio ROWS_OK
run tests/stdlib/data/test_csv_reader_col.sio  COL_OK
run tests/stdlib/data/test_csv_reader_int.sio  INT_OK
run tests/stdlib/data/test_csv_reader_f64.sio  F64_OK
run tests/stdlib/data/test_csv_reader_str.sio  STR_OK
# cap-free evidence: >256 columns read correctly
python3 - "$OUT/wide.csv" <<'PY'
import sys
n=300
open(sys.argv[1],'w').write(",".join(f"c{i}" for i in range(n))+"\n"+",".join(str(i) for i in range(n))+"\n")
PY
cat > "$OUT/wide.sio" <<'EOF'
use data::csv_reader::{csv_open_file, csv_next_row, csv_int, CsvReader}
fn main() -> i32 with IO, Mut, Panic, Div {
    var r = csv_open_file("wide.csv")
    csv_next_row(&!r)
    if csv_int(&!r, 299) == 299 { print("WIDE_OK\n") } else { print("FAIL wide\n") }
    return 0
}
EOF
if "$SOUC" compile "$OUT/wide.sio" -o "$OUT/w.elf" >/dev/null 2>&1; then
  chmod +x "$OUT/w.elf"; ( cd "$OUT" && "$OUT/w.elf" ) | grep -q WIDE_OK || { echo "FAIL wide"; fail=1; }
else echo "FAIL compile wide"; fail=1; fi
[ $fail -eq 0 ] && echo "DATA_IO_CSV_READER_GATE_OK"
exit $fail
```

- [ ] **Step 2: Run the gate with the fixed Madaros**

```bash
SOUNIO_TEST_SOUC_BIN="$MAD" bash scripts/data_io_csv_reader_gate.sh
```
Expected: `DATA_IO_CSV_READER_GATE_OK`.

- [ ] **Step 3: Write the example**

`examples/epistemic/csv_read_pk.sio`: read a small PK CSV (name,dose,conc), sum the `conc` column via `csv_f64_scaled`, print the total (integer-scaled). Include a `//@ run-pass` header only if a fixture ships alongside; otherwise keep it as a demo invoked by the gate with a temp fixture.

- [ ] **Step 4: Commit**

```bash
chmod +x scripts/data_io_csv_reader_gate.sh
git add scripts/data_io_csv_reader_gate.sh examples/epistemic/csv_read_pk.sio
git commit -m "test(data): csv_reader gate (parser, file round-trip, >256-col cap-free) + PK demo"
```

### Task 11: Wire the gate into CI + docs + PR

- [ ] **Step 1: Wire into ci.yml under the data-io/full condition** (mirror how `data_io_csv_gate.sh` is referenced), gated so it runs once #1078 is merged. If the writer gate is not in ci.yml, keep this dev-tier and note it in the PR body (do not fabricate a green CI dependency on unmerged #1078).

- [ ] **Step 2: Update `stdlib/data/README.md`** with a `csv_reader` section (API table, the `str_char_at`-only rule, the caller-buffer note, the #1078 dependency).

- [ ] **Step 3: Register docs governance + commit**

```bash
node scripts/docs/sync_governance_metadata.mjs
git add stdlib/data/README.md docs/governance/ .github/workflows/ci.yml
git commit -m "docs(data): document csv_reader; wire reader gate (post-#1078)"
```

- [ ] **Step 4: Open the PR** (base: `fix/native-read-file-string-return` if #1078 unmerged, else `main`), body stating the #1078 dependency and that CI goes green once #1078 lands.

---

## Self-Review notes (author)

- **Spec coverage:** csv_open/open_file/next_row/cols/col/int/f64/f64_scaled/str_into/ok — all tasked (Tasks 3–9); read_file dynamic-mmap (Task 2); gate incl. round-trip + cap-free (Task 10); docs/CI (Task 11). Error handling (`ok` flag, -1 col, non-fatal) covered in Tasks 5/6/7.
- **Known risk carried into execution:** exact Sounio syntax (`break`/`continue` in loops, `&!` receivers, `&![i8; N]` params) is verified by the Phase-0 probes and every task's compile+run step; if a construct is rejected, fall back to the functional-threading form (return updated `CsvReader` by value) and adjust signatures — the algorithm is unaffected.
- **`csv_str_into` size:** written for `[i8; 64]`; Task 1 decides whether Madaros allows a size-generic `N` (preferred) — if not, ship the fixed-64 form plus `csv_field_range`/`csv_byte` low-level accessors so no caller is capped.
