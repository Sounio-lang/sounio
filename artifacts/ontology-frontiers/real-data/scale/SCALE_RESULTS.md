# Scale push — FULL OAEI 2016 Anatomy TBox + the real scale ceiling

**Date:** 2026-08-02 · **Lane:** `kimi-swarm--scale-push-20260802` (round 7)
· **Compiler:** `bin/souc` (Madaros engine), branch
`research/zd-fiber-antisymmetry-lemma-20260731`.

Round 6 ran the verified pipeline on an ancestor-closed 1,961-class
subgraph of the human ontology.  This round removes the cap entirely:
**all 3,304 human classes, all 3,761 sub axioms, all 17 disjoint pairs,
all 6,638 candidate mappings** — and bisects the actual compiler/runtime
ceilings with synthetic probes.

**Headline:** the feared N² wall does not bind at this scale.  BOTH the
dense (round-6 algorithm, three 10.9M-cell bool matrices) and a new
sparse (no N² matrix) strategy run the FULL TBox to `ALL PASS` in
~17–19 s wall (compile + run).  The binding ceiling is elsewhere: the
**assignment-statement wall (~22.4k–24.4k per module, rc=19)**, now
verified for single-module compiles too.

## 1. Full-TBox result (real data, both strategies)

```
$ ./bin/souc run artifacts/ontology-frontiers/real-data/scale/full_scale_driver.sio   # sparse
=== OAEI 2016 Anatomy: FULL-TBox scale push (sparse) ===
human classes (H):            3304
sub axioms:                   3761
disjoint pairs:               17
distinct disjoint endpoints:  9
closure edges (full, BFS):    21859
mask fixpoint passes:         6
candidate mappings (M):       6638
derived conflicts (ordered):  736
kept:                         6392
dropped:                      246
top-5 dropped: 45, 46, 52, 56, 77  (all conf 3333 per-10000)
ALL PASS     — 17.5 s wall (compile + run)
```

The dense driver (`dense_full_driver.sio`, round-6 algorithm unchanged,
three `[bool; 10916416]` matrices) prints the same numbers with
`ALL PASS` in 19.0 s.  Both drivers re-verify all round-6 sanity checks
(closure edge count vs mirror, conflict symmetry, exact kept/dropped
counts, conflict-free repair, maximality witnesses, top-5) against an
independent python mirror embedded as `expected_*()`.

**The full-TBox conflict/repair numbers are byte-identical to the
1,961-class capped run of round 6** (736 ordered = 368 unordered
conflicts, 6,392 kept, 246 dropped, same top-5).  This is the empirical
confirmation that round 6's ancestor-closed cap was lossless for this
pipeline.  The full closure has 21,859 edges (vs 12,669 capped), 4
fixpoint passes.

## 2. Strategies

| | A. dense (round-6 algorithm) | B. sparse (this round) |
|---|---|---|
| memory | 3 × N² bools | O(N·EP + E) — no N² matrix |
| closure | naive N² fixpoint, ~passes×(N²+edges·N) | per-class BFS over packed parent adjacency (counting sort), O(total closure edges) |
| disj-reachability | N² `disj_c` expansion | bool[N·EP] endpoint-ancestor mask fixpoint over sub edges (EP = 9) |
| full TBox (N=3,304) | **ALL PASS, 19.0 s** | **ALL PASS, 17.5 s** |

Sparse correctness rests on a mirror cross-check in `gen_full_data.py`:
conflicts computed set-based (round-6 definition) and mask-based agree
exactly (736), else the generator aborts.

**Correctness note (new):** the 17 disjointWith pairs involve only **9
distinct classes**, and some endpoints are disjoint with *several*
others — the partner relation is one-to-many.  A mask implementation
must carry partner bit-*sets* (`ep_pbits`); a naive single-partner map
silently undercounts (244 vs 736 ordered conflicts — measured, caught
by the mirror cross-check before any Sounio run).

## 3. The ceiling (synthetic probes, star hierarchy + 1 disjoint pair +
200 mappings; wall = compile+run via `time ./bin/souc run`)

### 3a. Assignment-statement wall — THE binding constraint (rc=19)

| probe (data-init statements) | result |
|---|---|
| dense N=20,000 (20,413 stmts) | ALL PASS, 19.8 s |
| dense N=22,000 (22,413 stmts) | ALL PASS, 23.4 s |
| dense N=24,000 (24,413 stmts) | **FAIL** — `native-v2 bridge compilation failed`, rc=19, no ELF |
| dense N=27,000 (27,413 stmts) | FAIL rc=19 |
| dense N=30,000 (30,413 stmts) | FAIL rc=19 |
| dense N=50,000 (50,413 stmts) | FAIL rc=19 |
| sparse N=10,000 (10,402 stmts) | ALL PASS, 5.7 s |
| sparse N=30,000 (30,402 stmts) | FAIL rc=19 |

New finding: the ~24k assignment wall is **not specific to multimodule
thin-link** (round 6's framing) — single self-contained modules fail the
same way, between 22,413 and 24,413 array-assignment statements.
Workarounds (all verified this round): pair-packing, chunking ≤500
stmts/function, and **loop-initialisation** when the data is regular
(see 3b/3c).

### 3b. Dense N² array wall — NOT hit in-workspace

Loop-initialised star probes (constant statement count; only the N²
arrays grow):

| N | N² cells ×3 matrices | result |
|---|---|---|
| 30,000 | 0.9 G ×3 = 2.7 GB | ALL PASS, 22.5 s |
| 50,000 | 2.5 G ×3 = 7.5 GB | ALL PASS, 46.7 s |

The dense approach survives at least N=50,000 classes given loop-init
data.  N=100,000 (30 GB bools, multi-minute fixpoint) was NOT attempted
in the workspace (39 GB available RAM, shared machine, 2–3 min/run repo
rule) → Slurm handoff, §5.

For calibration, the round-6–feared "N=6,048 → 36.6M bools" is nothing:
probe_dense_6000 (36M cells ×3) passes in 5.4 s.

### 3c. Sparse strategy — no ceiling found

Loop-initialised probes:

| probe | N | closure edges | result |
|---|---|---|---|
| star | 100,000 | 199,999 | ALL PASS, 10.8 s |
| star | 1,000,000 | 1,999,999 | ALL PASS, 1.0 s |
| star | 10,000,000 | 19,999,999 | ALL PASS, 3.5 s |
| chain | 10,000 | 50,005,000 | ALL PASS, 1.8 s |
| chain | 30,000 | 450,015,000 | ALL PASS, 8.9 s |

Sparse cost is O(NSUB·EP·passes + total closure edges), not N²; it
counts a 450M-edge closure in 9 s.  Class count itself is irrelevant
(10M classes pass); the cost driver is total closure size × EP.

**Pitfall re-confirmed:** the first loop-init sparse probes segfaulted
(100k/1M) / miscounted by 1 (10M) because `poff[0]` (module-level splat
i64 array) kept its garbage leading cell — the round-6 fixup rule
(bool 0..2, i64 index 0) must cover **every** partially-written
module-level array.  One-line fix (`poff[0] = 0`), all probes then pass.

### 3d. Sharding (option c) — not needed

Splitting the run into per-shard .sio programs aggregated by python was
engineered around: the sparse strategy's memory is O(N·EP+E) and the
dense arrays fit to ≥50k classes, so no sharding was required.  If ever
needed, the statement wall (~22k) bounds each data shard, not the
algorithm.

## 4. Reproduction

```bash
cd artifacts/ontology-frontiers/real-data/scale
python3 gen_full_data.py        # mirror + full_data.sio, dense_full_data.sio, both drivers
python3 gen_probes.py           # synthetic probes (star, data-init)
cd /workspace/sounio
./bin/souc check artifacts/ontology-frontiers/real-data/scale/full_scale_driver.sio   # check: OK
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/full_scale_driver.sio   # ALL PASS
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/dense_full_driver.sio   # ALL PASS
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/probe_dense_22000.sio   # ALL PASS (last OK)
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/probe_dense_24000.sio   # rc=19 FAIL
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/probe_dense_arr_50000.sio  # ALL PASS
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/probe_sparse_loop_star_10000000.sio  # ALL PASS
./bin/souc run   artifacts/ontology-frontiers/real-data/scale/probe_sparse_loop_chain_30000.sio    # ALL PASS
```

The loop-init array-wall/chain probes are emitted by a heredoc in the
round-7 session log (not by `gen_probes.py`); their sources are the
committed `probe_dense_arr_*.sio` / `probe_sparse_loop_*.sio` files.

## 5. Slurm handoff (per docs/ops/foundry_slurm_handoff.md)

Workspace-safe envelopes are established above; do NOT rerun these in
the workspace:

1. **Dense array wall bisection, N = 50k → 100k+** (loop-init star):
   N=100,000 needs 3×10¹⁰ bools ≈ 30 GB and a ~10¹¹-op fixpoint
   (multi-minute).  Gate: `probe_dense_arr_100000.sio` (generate with
   the 3b template) on a Foundry node; record pass/OOM/timeout.
2. **Dense fixpoint on deep hierarchies**: the naive N² fixpoint is
   passes×(N²+edges·N); a chain at N≥5k is already 10¹⁰+ ops.  If dense
   semantics on deep real ontologies ever matter, gate a chain probe
   (dense, loop-init) at N=5k/10k on Slurm and expect minutes–hours;
   the sparse strategy is the verified answer there.

## 6. Files (all under artifacts/ontology-frontiers/real-data/scale/)

| file | role |
|---|---|
| `gen_full_data.py` | full-TBox generator + python mirror (set/bitmask cross-checked) + both driver templates |
| `full_data.sio`, `full_scale_driver.sio` | SPARSE full-TBox module + driver → ALL PASS, 17.5 s |
| `dense_full_data.sio`, `dense_full_driver.sio` | DENSE full-TBox module + driver → ALL PASS, 19.0 s |
| `gen_probes.py` | synthetic star-probe generator (data-init) |
| `probe_dense_{4000..50000}.sio` | dense statement-wall bisection (22k PASS / 24k, 27k, 30k, 50k FAIL rc=19) |
| `probe_dense_arr_{30000,50000}.sio` | dense array-wall probes (loop-init) → ALL PASS |
| `probe_sparse_{10000,30000,...}.sio` | sparse statement-wall confirmation (10k PASS / 30k FAIL) |
| `probe_sparse_loop_star_{100k,1M,10M}.sio` | sparse star probes (loop-init) → ALL PASS |
| `probe_sparse_loop_chain_{10000,30000}.sio` | sparse chain probes → ALL PASS (450M edges) |

(`probe_sparse_100000.sio` / `probe_sparse_300000.sio` are generated but
deliberately unrun: at 100k+/300k+ data-init statements they sit far
beyond the rc=19 wall; the loop-init variants supersede them.)

## 7. Limitations

- Synthetic probes use star/chain hierarchies; real ontologies are
  between the two in depth, but the anatomy TBox itself is fully
  covered by §1 (real data, both strategies).
- The dense fixpoint's pass count is data-dependent (4 on anatomy);
  probe timings are for the star's 2-pass convergence.
- All wall times are compile+run of `bin/souc run` on this 64-core
  workspace; they are not isolated runtime benchmarks.
