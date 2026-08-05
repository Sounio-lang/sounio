# EL+ role-aware closure on full GO go-plus — optimization results (round 13)

**Date:** 2026-08-05 · **Branch:** `research/zd-fiber-antisymmetry-lemma-20260731`
· **Compiler:** `bin/souc` (Madaros engine).

Round 12 ran the EL+ role-aware boolean closure on the full GO go-plus
ontology in **~3.5 min wall** (`./bin/souc run`, compile + run), 13.6x
slower than the ELK 0.6.0 reference (~15.4 s).  Round 13 rewrites the
fixpoint engine of `go_full_elplus_driver.sio` and closes the gap
entirely:

| | compile | run | wall (`souc run`) | vs ELK 15.4 s |
|---|---|---|---|---|
| round 12 (bitmask cube) | ~3 s | 204 s | **3 m 43 s** | 13.6x slower |
| round 13 (sparse rows) | ~2.5 s | **2.2 s** | **4.7 s** | **3.3x faster** |

Same input (`go_full_packed.txt`), same mirror checks, same printed
numbers — every figure is bit-identical to the python bitmask mirror
(`gen_go_full_data.py`, embedded as `go_full_expected_*()`), including
both rule-family ablations:

```
atomic projection closure edges:                  395939
atom-source role edges (= existential targets):   2135207
atomic conflicts (ordered pairs):                 792814846
role fixpoint rounds (full / noRC / noRS):        4 / 2 / 4
role edges without roleComp:                      1883813
role edges without roleSub:                       597305
ALL PASS
```

## What was slow

The round-12 driver materialized the role-edge relation as a dense
bitmask cube `fm[(r*HC + c)*WC + w]` — 92 x 39,000 x 598 i64 words,
**17.9 GB of BSS**, scanned far too often:

- `seed_and_expand`: 57,824 sub edges x 92 roles x 598 words = **3.2G
  word-ops per run, x3 runs** (full + two ablations), mostly scanning
  entirely-empty parent rows.
- `role_fixpoint`: full-cube dirty scans and 3.74M-cell dirty-set swaps
  per round; roleComp recomputed `acc = union of F[r2][f]` (~598-word OR
  per filler bit) for every non-empty cell in every round, regardless of
  whether its inputs changed.
- `count_role_edges`: 2.1G word loads per run, x3.
- `clear_f`: 2 x 17.9 GB zeroing between ablations.
- conflict counting: all 38,245 classes are actors, so the naive
  actor x actor loop is **1.46G iterations**.

Measured profile data (python prototypes `analyze_opt.py`,
`analyze_conf.py`): only **216,783 of 3.74M cells (5.8%)** are ever
non-empty, rows average ~10 fillers, and the endpoint-ancestor mask
takes only **218 distinct values**.

## What round 13 does instead

1. **Sparse sorted-list rows.**  Each non-empty cell (r, c) owns a
   segment of a 256 MB `arena` holding ascending filler ids
   (`f_off`/`f_len`/`f_cap` parallel arrays, geometric reallocation).
   Set union is a linear merge with dedup (~20 ops instead of a 598-word
   scan).  The 17.9 GB cube disappears; total BSS drops to ~0.9 GB.
2. **Non-empty cell list** (`ne_list`): expand/count/clear iterate only
   non-empty cells.
3. **Single-queue worklist** for roleSub (`dq` + `inq` dedup, per-role
   supers CSR), replacing full-cube dirty scans and swap loops.
4. **Version-skipped roleComp** (semi-naive evaluation): every row
   change bumps a global version (`ver` per cell, `rmax` per role);
   each (chain, cell) pair records the version of its last processing
   (`lpv`).  A pair re-fires only if the cell's row changed or one of
   its current fillers' r2-rows changed — a ~10-element version walk
   instead of a full acc rebuild.  This is the *complete* semi-naive
   form of the round-12b rule: the direction-2 leak (F[r2][f] gaining an
   edge must re-fire every c with f in F[r1][c]) is covered by the
   per-filler version check.  Prototyped in `analyze_sparse.py`: exact
   agreement with the round-12 mirror in all three configurations
   (full / no-roleComp / no-roleSub).
5. **Grouped conflict counting**: group classes by their endpoint mask
   (218 distinct values), so counting is H x 218 (~8.3M iterations)
   instead of actors x actors (1.46G).

Soundness/completeness are unchanged: every merge is an exact set union,
and the worklist/version machinery only skips recomputing rule outputs
whose inputs are unchanged (Gauss-Seidel chaotic iteration of the same
monotone operator — same least fixpoint; the round count is unchanged
too, 4/2/4).  All mirror cross-checks still gate `ALL PASS`.

## Compiler pitfalls surfaced (workarounds in the driver)

- **Module-level scalar initializers are unreliable** on the current
  native lane: `pub var g: i64 = 0` reads back garbage (probe: five
  scalars read 4202496..4202501).  Same family as the known array
  leading-cell pitfall.  Workaround: assign every module scalar
  explicitly in `main` before use.  This was the root cause of the
  first round-13 segfault (garbage `ne_n`/`gver`/`arena_n`).
- **`&&` does not short-circuit the array read in its RHS** (probe:
  `while v > 0 && arr[v - 1] > 3` segfaults at v = 0).  Workaround:
  guard with a flag variable, never index with a potentially
  out-of-range expression on the RHS of `&&`/`||`.
- The known `println(<computed local>)` char*-dispatch quirk
  (KNOWN_LIMITATIONS.md, SRET/println note) also bit: printing
  `let ne_full = ne_n` 200 lines later segfaulted; printing `ne_n`
  directly is fine.

## Reproduction

```bash
cd /workspace/sounio
./bin/souc check artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio   # check: OK
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio   # ALL PASS, ~4.7 s
# run-only timing:
./bin/souc compile artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio -o /tmp/go13.elf
time /tmp/go13.elf        # ~2.2 s, ALL PASS
```

Prototypes (validation evidence): `analyze_opt.py` (workload stats +
bitmask version-skip prototype), `analyze_sparse.py` (exact sparse-row
prototype, matches mirror in all three configs), `analyze_conf.py`
(grouped conflict counting cross-check).

## Remaining notes / trade-offs

- `stdlib/ontology/elplus.sio` is untouched (API preserved).  Its
  sparse variant (Anatomy profile, 3,304 classes) already runs in
  ~17.5 s wall including compile; the same sparse-row technique could
  be backported there, but it is not the ELK benchmark and was left
  alone.
- The sparse-row representation wins because rows are short (~10
  fillers).  On data with near-dense rows (thousands of fillers per
  cell), the bitmask cube would merge in O(1) per word while the sorted
  lists degenerate to O(row) per union — the arena would also grow
  towards total-bits x 2.  The GO go-plus profile is far from that
  regime.
- No parallelism was needed: the algorithmic wins (work only on
  non-empty cells, only on changed inputs) removed ~99% of the
  sequential work.
