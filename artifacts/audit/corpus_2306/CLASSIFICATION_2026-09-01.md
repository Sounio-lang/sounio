# Corpus failure classification — issue #2306 — 2026-09-01

**Instrument:** `scripts/dev/corpus_failure_signature_scan.sh` — same program
filter as `madaros_corpus_regression_gate.sh`, but keeps rc + first diagnostic
per program (the gate discards error text). Run on Slurm r770 (bench), JOBS=6,
against the **committed** `bin/madaros-linux-x86_64` at branch tip (prebuilt
refreshed by `5c7b9ddec3`, after #2302). rc histogram: 133×rc=1 — zero OOM-137,
zero SEGV-139, so every row is a real compiler verdict.

Scope note: compile leg only. The issue's run (15) and stdout (8) modes were
not re-measured.

## The picture moved since the issue was filed

| | issue #2306 (filing) | this scan |
|---|---:|---:|
| compile failures outside baseline | 363 | **9** |
| total compile failures | — | 133 |
| baseline entries now passing | 90 | **147** |
| programs scanned | — | 1824 |

Between the filing and this scan, the merge wave landed (the gen2 stack,
`#2339` unsigned ops, `#2344`, …) and the committed prebuilt was refreshed
twice (`7cb1179632` #2302, then `5c7b9ddec3`). The `set difference between two
runs` discipline the issue asks for is exactly what this table is.

## The 9 new failures collapse to TWO causes — both fail-closed honesty gates

**Cause A (8 programs): `lower.sio:9712` — "cannot safely lower print/println
argument with unresolved scalar kind".** Introduced by `a2b99df071` (#1876,
"refuse unresolved print dispatch instead of dereferencing scalars"). These
programs did not regress from working to broken; they regressed from
*silently risky* to *loudly refused*. Files: `budget64_test.sio`,
`compress_huffman_fixed.sio`, `generic_struct_instantiate.sio`,
`generic_struct_nested.sio`, `println_string_array_field_element.sio`,
`smt_qflia_basic.sio`, `test_functional.sio`, `test_mor.sio`.

**Cause B (1 program): E221 on `math_atan_quadrant_reduction.sio`** —
`atan` is bound for typechecking but the native backend cannot emit it; the
call would compile and die on an illegal instruction at run time. The
diagnostic names the fix: `use math::pure::{atan}`.

So the "one segfault, not N defects" reading from July has a successor shape:
**a small number of fail-closed gates, not silent breakage**. The 364
regressions the issue warned against enshrining are no longer the live set.

## The fork for the 9 (not decided here)

- **Fix the tests**: make the printed scalar kind explicit at each site /
  `use math::pure::atan`. Correct if #1876's refusal is the intended surface.
- **Teach the printer**: resolve a Knowledge argument's scalar kind to its
  inner type in print dispatch instead of refusing. Correct if the refusal is
  over-broad for values with a printable inner. This is a language-surface
  decision — it belongs to whoever owns the print-dispatch semantics.

## Baseline recommendation (issue question 2)

The refresh that would have enshrined 364 regressions at filing is a different
operation now: it would REMOVE 147 stale entries (fixed programs — good
hygiene, the gate already reports disappearances non-blocking) and ADD the 9
honesty refusals (wrong — those are live triage). Order: decide the fork for
the 9, land it, then refresh so the baseline shrinks.

## Issue question 3 (gate UX)

Implemented in spirit here: this report leads with the set difference, not the
raw count. A gate-side patch (refuse a raw count when the baseline predates
HEAD by > N commits) remains open and unclaimed.

## Reproduce

```text
bash scripts/dev/corpus_failure_signature_scan.sh            # scan (Slurm bench)
bash scripts/dev/corpus_failure_signature_scan.sh --cluster  # re-cluster
```

Data: `artifacts/audit/corpus_2306/runs.tsv` (1824 rows),
`artifacts/audit/corpus_2306/clusters.md`.
