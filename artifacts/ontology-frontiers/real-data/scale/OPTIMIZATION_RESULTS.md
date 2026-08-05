# EL+ role-aware closure on full GO go-plus — optimization results (rounds 13-14)

**Date:** 2026-08-05 · **Branch:** `research/zd-fiber-antisymmetry-lemma-20260731`
· **Compiler:** `bin/souc` (Madaros engine).

Round 12 ran the EL+ role-aware boolean closure on the full GO go-plus
ontology in **~3.5 min wall** (`./bin/souc run`, compile + run), 13.6x
slower than the ELK 0.6.0 reference (~15.4 s).  Round 13 rewrote the
fixpoint engine of `go_full_elplus_driver.sio` and closed the gap;
round 14 (same file) profiled the residue and removed the remaining
non-fixpoint hotspots:

| | compile | run | wall (`souc run`) | vs ELK 15.4 s |
|---|---|---|---|---|
| round 12 (bitmask cube) | ~3 s | 204 s | **3 m 43 s** | 13.6x slower |
| round 13 (sparse rows) | ~2.5 s | **2.2 s** | **4.7 s** | **3.3x faster** |
| round 14 (sparse anc + group conf) | ~2.3 s | **1.3 s** | **3.6 s** | **4.3x faster wall / 11.9x run-only** |

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
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio   # ALL PASS, ~3.6 s
# run-only timing:
./bin/souc compile artifacts/ontology-frontiers/real-data/scale/go_full_elplus_driver.sio -o /tmp/go14.elf
time /tmp/go14.elf        # ~1.3 s, ALL PASS
```

Prototypes (validation evidence): `analyze_opt.py` (workload stats +
bitmask version-skip prototype), `analyze_sparse.py` (exact sparse-row
prototype, matches mirror in all three configs), `analyze_conf.py`
(grouped conflict counting cross-check).

## Round 14: what was still slow, and what changed

Round-13 run-only profiling (early-exit ablation binaries, cumulative):
parse 16 ms · role CSR/topo ~7 ms · **anc bitmask closure + popcount
390 ms** · full seed+expand+fixpoint 765 ms · conflicts 175 ms ·
ablation runs ~395 ms each.  Round 14 attacked the non-fixpoint 60%:

1. **Sparse ancestor closure.**  The `anc` bitmask (H+1 bits per class,
   23.3M i64 words) is deleted.  Ancestor rows are sorted lists in a
   second arena (`a_arena`) built by the same merge machinery: row of c
   starts as `[c, H]` and merges each parent's row in topological order
   (~1M merge ops instead of 57,824 x 598 = 34.6M word iterations).
   `atomic_edges = sum(a_len) - H` replaces the 22.9M-word
   `row_popcount` sweep.  Segment time: 390 ms → ~16 ms.
2. **Seeds read the sparse ancestor row directly** (a copy into `scrb`),
   replacing the 18,791 x 598-bit extraction scan in every run
   (~80 ms → ~10 ms per run, x3 runs).
3. **Expand only seed-bearing roles.**  A role with no existential seed
   has every row empty during the expand sweep, so its 57,824
   parent-edge checks are provably no-ops; the state entering the role
   fixpoint is bit-identical (such roles are still reached through
   roleSub/roleComp exactly as before).  53 of 92 roles carry seeds on
   this data.
4. **Chain-relevant cell list (`cne_list`).**  `comp_scan` iterates only
   non-empty cells whose role is the r1 of some roleComp chain —
   118,467 of 216,783 cells (29 of 92 roles carry chains); the rest had
   an empty chain range and were pure scan overhead.  Same snapshot
   semantics, same reprocessing condition, identical `(cell, chain)`
   check stream (verified: 1,313,528 checks, 341,635 reprocessings,
   exactly as round 13).
5. **Group-level conflict machinery.**  `pm[c]` depends on c only
   through its endpoint mask `epm[c]`, so partner masks are computed
   once per distinct epm value (218 groups x 55 pairs instead of
   38,245 x 55) and the conflict sum runs over group pairs (218 x 218
   with per-group diagonal exclusion — the same sum as the per-class
   loop, member by member).  `epm` itself is built by walking the
   sparse ancestor rows (~434k visits) instead of 55 x 38,245 bitmask
   probes, and grouping uses an open-addressing hash on (epm0, epm1)
   instead of a linear group-table scan.  Segment time: 175 ms → ~5 ms.

Bit-exactness is preserved end to end: every printed number — including
`ne_n` (216,783) and `arena_n` (5,041,814), which fingerprint the exact
role-row engine state — is identical to round 13, and all mirror
cross-checks still gate `ALL PASS`.

Measured with counters compiled into the driver (then removed): the
residual ~1.0 s of fixpoint time across the three runs is dominated by
the semi-naive reprocessing machinery itself — 1.31M (cell, chain)
re-checks, 6.4M per-filler version probes, 342k accumulator rebuilds
(only 2.4M OR-ops total; rows average ~7 ops per rebuild) and 257k
roleSub queue pops.  The next big lever would be a fully event-driven
roleComp (reverse index f -> cells per chain, delta propagation instead
of accumulator rebuilds), but its estimated propagation volume is the
same order as the current rebuild cost on this data, at a large
complexity and memory cost (~250 MB of reverse-index state) — not worth
it for the ELK comparison.  A hoisting micro-edit in the version-probe
loop was tried and measured *slower* (1.38 s vs 1.30 s, consistently,
A/B alternated), so it was reverted; the remaining fixpoint time is
close to the floor set by the merge work itself (2.1M edge insertions,
5.0M arena words written per full run).

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
  regime.  Round 14 makes the same trade for ancestor rows (avg ~11
  ancestors per class here).
- Parallelism was considered and rejected for this driver: Sounio has
  no native threads, and the chaotic-iteration schedule is inherently
  sequential per monotone-operator round.  The wins came from removing
  sequential work, not spreading it.
- Universe reduction is already maximal for this application: the
  profile theorem (round 11) reduces statistics to the atom level, and
  the driver materializes exactly the stated axioms plus one
  existential layer — nothing removable without changing the
  deliverable numbers.
- Compile time (~2.3 s) is now the larger half of the `souc run` wall;
  ~0.5 s of it is compiler startup on an empty file, and the giant BSS
  splat arrays were verified to cost nothing extra (they compile as
  NOBITS).  Further wall-clock gains need compiler-pipeline work, which
  is out of scope here.
- No parallelism was needed: the algorithmic wins (work only on
  non-empty cells, only on changed inputs) removed ~99% of the
  sequential work.
