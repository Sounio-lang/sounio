<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-07-13-data-science-io-design
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-07-13-data-science-io-design
-->

# Design — Sounio Data & Science I/O release

**Status:** approved design, pre-implementation
**Date:** 2026-07-13
**Author:** Claude (session), for founder review
**Constraint:** No compiler changes. The compiler (`self-hosted/`, `bootstrap/`) is owned by CODEX-2. All work lands in `stdlib/`, `tools/`, `examples/`, `tests/`.
**Orthography:** EN-UK (CLAUDE.md §9).

---

## 1. Intent

Make Sounio able to load, hold, and emit real-world scientific data end-to-end. One canonical,
epistemic-aware `DataFrame` acts as the hub; real readers (CSV, Parquet, netCDF-classic, HDF5) and
writers (CSV, JSON, Parquet, formatted table, provenance artifact) are spokes. Everything is written
in Sounio (operating principle 4 — the science path is Sounio, not Python).

The differentiator over "pandas, but worse" is the through-line: **uncertainty, units, and provenance
survive a full read → transform → write round-trip.** That is the spine, not an add-on.

## 2. Scope

### In scope
- Hub hardening: null/missing mask, datetime dtype, per-column units + provenance metadata, frozen public API.
- Writers: CSV, JSON, formatted table (terminal + markdown), reproducible results-artifact, Parquet write.
- Readers: CSV → typed DataFrame; Parquet; netCDF-classic (CDF-1/2/5); HDF5 (contiguous **and** chunked+filtered).
- A new snappy decompressor (`stdlib/compress/snappy.sio`), required for Parquet read.
- Golden-fixture conformance gates per format, wired like existing stdlib gates.

### Out of scope (explicit)
- Embedded database (the `database/` stub lane) — deferred to a separate spec.
- Parquet nested/repeated types (lists/maps/structs) — primitive columns first; nested deferred.
- netCDF-4 as its own reader — netCDF-4 is HDF5-backed and is served by the HDF5 phase, not the netCDF-classic phase.
- Writing HDF5 / netCDF / deleting any existing DataFrame or CSV implementation.

## 3. Constraints & principles

- **No compiler changes.** If a phase hits a codegen wall, it is reported as a forensic dispatch
  (`docs/audit/`), not worked around in `self-hosted/`. (CLAUDE.md §8.)
- **Additive consolidation, not deletion.** `stdlib/data/frame.sio` becomes canonical; new code targets
  it. The other DataFrame impls (`data/dataframe.sio`, `dataframe/pure/core.sio`) and the second CSV
  parser (`data/csv_loader.sio`) are marked deprecated in-file but **remain** — the petab-fit e2e
  (`tests/stdlib/darwin_pbpk/test_observed_petab_fit_e2e.sio`) and pbpk workflow
  (`tests/packages/package_pbpk_gum_workflow.sio`) depend on the existing code, which is the
  dissertation-critical path (operating principles 2 & 5).
- **Conformance oracle defines "real".** A reader is complete only when it reads bytes emitted by the
  reference tool (pyarrow / netCDF4 / h5py). Fixture *generation* is verification tooling, in the same
  spirit as the existing z3 crosscheck — not Python in the science path.
- EN-UK orthography; no AI attribution in commits; atomic commits, one logical change each (principle 11).

## 4. Architecture

```
                    ┌─────────────────────────────┐
   readers ───────► │   canonical DataFrame hub    │ ───────► writers
  CSV / Parquet     │  stdlib/data/frame.sio       │   CSV / JSON / Parquet
  netCDF / HDF5     │  + null mask, datetime dtype,│   table / provenance-artifact
                    │    units + provenance meta   │
                    └─────────────────────────────┘
        shared substrate:
          stdlib/io/binary.sio    — read_uNN_le/be primitives
          stdlib/compress/*.sio   — gzip/deflate/inflate, zstd, + NEW snappy
```

### 4.1 The hub (`stdlib/data/frame.sio`)

Current state (verified): type-erased `Column` with `dtype` discriminator
(0=Float, 1=Int, 2=String, 3=Bool, 4=Epistemic), a `DataFrame` of named columns, and a working public
API (add/get/drop column, shape, slice/head/tail, filter by bool mask, row access, `col_sum`/`col_mean`,
`col_mean_epistemic`, `info`). **Missing** (all additive):

- **Null/missing mask** — per-column `[bool]` validity mask (or `null_mask`), so CSV blanks, Parquet
  definition levels, and netCDF `_FillValue` map to a first-class "missing" rather than a sentinel.
- **Datetime dtype** — new dtype (5=Datetime) stored as epoch `i64` (nanoseconds or seconds — decided in
  Phase 0) plus a unit tag; date parsing/formatting helpers.
- **Units + provenance metadata** — optional per-column `units: String` and `provenance: String`
  (source file, tool, checksum). Carried through slice/filter/select and emitted by writers.
- **Frozen public API** — the surface Phases 1–5 target. Once frozen in Phase 0, readers/writers build
  against a stable contract. Documented in `stdlib/data/README.md` (or module header).

`Column`/`DataFrame` remain value types (copy semantics), matching the current implementation. No new
effects introduced beyond what `frame.sio` already uses.

### 4.2 Shared substrate

- `stdlib/io/binary.sio` already provides `read_u16/32/64_le/be`, `read_iNN`. Extend additively with any
  missing primitives readers need (e.g. `read_f32/f64_le/be`, varint/zigzag for Parquet Thrift, LEB128).
- `stdlib/compress/` has `gzip_decompress`, `inflate_stored`, `zstd_decompress`. **New:**
  `stdlib/compress/snappy.sio` — snappy block-format decompress (LZ77 variant, ~200 loc). Snappy is
  Parquet's default codec; without it Parquet read fails on the majority of real files.

## 5. Components & phases

Each phase is a working, independently gated slice. Order is dependency-driven: writers first so the hub
API is proven before any reader targets it.

### Phase 0 — Hub hardening
- Files: `stdlib/data/frame.sio` (extend), `stdlib/data/README.md` (API contract).
- Deliverable: null mask, datetime dtype, units/provenance metadata, frozen API.
- Gate: a `.sio` test constructing frames with nulls/dates/units and asserting round-trip through
  slice/filter/select preserves them.

### Phase 1 — Text writers (target 4)
- Files: `stdlib/data/write_csv.sio`, `stdlib/data/write_json.sio`, `stdlib/data/table.sio`,
  `stdlib/data/artifact.sio`.
- CSV: RFC-4180 quoting/escaping, null rendering, configurable delimiter.
- JSON: records (array of objects) and columnar (object of arrays) modes; nulls as JSON `null`.
- Table: aligned columns for terminal; GitHub-flavoured markdown mode.
- **Results-artifact writer**: a reproducible bundle — data file + sidecar carrying schema, per-column
  units, provenance, row/col counts, and a content checksum. This is the epistemic differentiator; format
  decided in Phase 1 planning (leaning: data + `.meta.json` sidecar).
- Parquet **write** is deferred to Phase 3, where it shares the `stdlib/parquet/` Thrift codec with the
  reader (avoids building that module out of dependency order).
- Gate: round-trip in-memory frame → write → self-read (CSV/JSON) equals original, nulls/units/provenance
  preserved.

### Phase 2 — CSV reader → typed DataFrame (target 1)
- Files: consolidate onto `stdlib/csv/parser.sio`; new `stdlib/data/read_csv.sio` binding it to the hub.
- Type inference (int → float → bool → datetime → string precedence), missing-value handling (blank/`NA`/
  configurable → null mask), date parsing, quoting per RFC-4180, epistemic-column hook (e.g. `value±unc`
  or paired columns → epistemic dtype).
- Gate: golden CSVs (incl. quoted fields, embedded newlines, missing values, mixed types) → expected typed
  frame; cross-check against pandas-emitted expectations.

### Phase 3 — Parquet reader + writer (target 3)
- Files: `stdlib/parquet/` (reader + writer + shared Thrift-compact codec), `stdlib/compress/snappy.sio` (new).
- Parquet **write** (moved here from Phase 1): uncompressed / gzip / zstd — no snappy needed to *write*;
  emits files pyarrow can read (write-side conformance gate). Reuses the shared Thrift-compact codec.
- Thrift compact-protocol footer parse (FileMetaData, schema, row groups, column chunks); plain,
  dictionary (RLE/bit-packed), and RLE encodings; codecs uncompressed/snappy/gzip/zstd; primitive
  physical types (BOOLEAN, INT32/64, FLOAT, DOUBLE, BYTE_ARRAY) with logical-type hints (string, date,
  timestamp) → hub dtypes incl. datetime and nulls (via definition levels).
- Gate: read pyarrow-emitted `.parquet` fixtures (each codec, dictionary on/off, with nulls) → expected
  frame; plus write-side round-trip (Sounio-written `.parquet` re-read by pyarrow equals original).

### Phase 4 — netCDF-classic reader (target 3)
- Files: `stdlib/netcdf/` (classic reader).
- CDF-1/CDF-2/CDF-5 magic; header: dim list, global attrs, var list (name, dims, type, attrs, offset);
  big-endian numeric decode; fixed vars and one record (unlimited) dim; `_FillValue` → null mask; map to
  DataFrame (1-D vars as columns) and/or tensor for N-D vars.
- Explicit doc caveat: **classic only**; netCDF-4 files route through the HDF5 reader (Phase 5).
- Gate: read netCDF4-emitted classic-format fixtures → expected frame/tensor.

### Phase 5 — HDF5 full reader (target 3, largest)
- Files: `stdlib/hdf5/` (superblock, object header, B-tree, heap, filter pipeline, dataset reader).
- Superblock v0/v2/v3; object-header messages (dataspace, datatype, data-layout, filter-pipeline,
  attribute, link); group traversal via symbol-table (v0) and link messages (v2); **data layout**:
  contiguous and **chunked** via **B-tree v1 chunk index**; **filter pipeline**: gzip/deflate (existing
  `inflate`) and shuffle; datatypes: fixed/float/string, little- and big-endian → hub dtypes; N-D datasets
  → tensor, 1-D → column.
- May split during planning: **5a** contiguous/uncompressed (proves superblock+object-header+datatype),
  **5b** chunked + filter pipeline (B-tree + gzip/shuffle).
- Bonus (not committed): with HDF5 working, a thin netCDF-4 convention layer becomes feasible later.
- Gate: read h5py-emitted fixtures — contiguous, chunked, gzip-filtered, shuffle+gzip, big-endian — →
  expected arrays/frame.

## 6. Module layout

Top-level, matching existing convention (`json`, `csv`, `toml`, `yaml` are top-level):

```
stdlib/data/frame.sio          (hub — extended)
stdlib/data/read_csv.sio       (new — CSV→hub binding)
stdlib/data/write_csv.sio      (new)
stdlib/data/write_json.sio     (new)
stdlib/data/table.sio          (new — pretty/markdown table)
stdlib/data/artifact.sio       (new — provenance artifact)
stdlib/data/README.md          (new — frozen API contract)
stdlib/compress/snappy.sio     (new)
stdlib/parquet/                (new — reader, writer, thrift codec)
stdlib/netcdf/                 (new — classic reader)
stdlib/hdf5/                   (new — full reader)
```

Each new module follows the existing `lib.sio` / `mod.sio` convention used across `stdlib/`.

## 7. Verification — the conformance gate

"Real" is defined by the reference tool, per repo culture (z3 crosscheck; "always cross-verify vs an
independent oracle").

- **Fixtures**: generated by `pyarrow` (Parquet), `netCDF4` (netCDF-classic), `h5py` (HDF5), committed
  under `tests/stdlib/<fmt>/fixtures/`. Generation scripts live in `scripts/research/` or
  `tests/stdlib/<fmt>/gen_fixtures.py`, run once and checked in; they are verification tooling, not the
  science path.
- **Read gate**: each reader reads every fixture and asserts the resulting frame equals a committed
  expected (schema + values + nulls + units where applicable).
- **Round-trip gate**: writers → re-read (self for CSV/JSON; reference tool for Parquet) equals original,
  with uncertainty/units/provenance preserved.
- **Wiring**: one gate script per format under `scripts/`, invocable standalone and from the suite, in the
  style of `scripts/stdlib_hyper_execution_gate.sh`. `SKIP_BUILD=1` honoured.
- **Compiler-surface honesty**: gates run under `./bin/souc` (Madaros default; note lean_single fallback
  if a file needs it). Report exact command + path + evidence (principle: auditability over speed).

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| HDF5 chunked+filter is very large; may not finish in one pass | High | Split 5a/5b; 5a is independently useful; each has its own gate. Halt-with-report is a valid deliverable (principle 8). |
| Parquet Thrift-compact + dictionary/RLE encodings are fiddly | Med | Start with plain-encoded uncompressed fixtures; add dictionary/RLE/codecs incrementally, each gated. |
| Snappy decompressor bug corrupts silently | Med | Golden-fixture byte-exact gate against pyarrow output; unit tests on snappy spec vectors. |
| Codegen wall in a reader (large struct / array mutation) | Med | Known gotchas (struct wrappers, `(*arr)[i]`). If genuinely a compiler bug → forensic dispatch, not a `self-hosted/` patch. |
| Hub API churn breaks in-flight reader work | Low | Freeze API in Phase 0 before any reader starts; writers (Phase 1) shake it out first. |
| Consolidation accidentally breaks pbpk/petab tests | Low | Additive only; run those two tests as a regression gate after any `frame.sio` change. |

## 9. Success criteria

1. A Sounio program reads a real pyarrow-written `.parquet`, an h5py-written chunked+gzip `.h5`, a
   netCDF4-written classic `.nc`, and a messy quoted CSV — each into the same `DataFrame` type.
2. That frame can be written back to CSV, JSON, and Parquet, and a provenance artifact, with
   uncertainty/units/provenance preserved end-to-end.
3. Every format has a committed golden-fixture gate that passes under `./bin/souc`, cross-checked against
   the reference tool.
4. No file under `self-hosted/` or `bootstrap/` is modified. Existing pbpk/petab tests still pass.

## 10. Resolved decisions (defaults taken 2026-07-13)

- **Datetime storage** — store epoch as `i64` **nanoseconds** plus an explicit unit tag on the column
  (so netCDF "seconds/days since epoch" attrs convert into the same representation). Nanoseconds match
  Parquet/HDF5 conventions and avoid lossy sub-second truncation.
- **Artifact bundle shape** — data file + `.meta.json` **sidecar** carrying schema, per-column units,
  provenance, row/col counts, and content checksum. (Not a single packed file.)
- **N-D arrays** — **flattened data + shape in column/frame metadata** for this release; a dedicated
  `Tensor` type is not introduced (avoids a new cross-module dependency). N-D netCDF/HDF5 variables land
  as a flat buffer plus a `shape: [usize]` tag; 1-D variables land as ordinary columns.
```
