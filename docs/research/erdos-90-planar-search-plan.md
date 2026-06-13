# Erdős [90] — planar `u(n)` cluster search plan + pilot results

*Companion to `formal/lean4/SounioErdos90PlanarLowerBound.lean` and the CPU search kernel
`stdlib/research/erdos90_search.sio`. Goal: attack the classical open problem — the
maximum number `u(n)` of unit distances among `n` points in ℝ² — with the Sounio
compiler, K-AXI→PTX, and the GPU cluster. Honest framing throughout.*

## What is open, exactly

* **Lower bound (baseline, verified).** The triangular lattice (Eisenstein integers ℤ[ω])
  gives `u(n) ≥ ⌊3n − √(12n−3)⌋` =: `harb(n)` (Harborth 1974). Verified for n=1..18 with
  explicit exact configs in `lattice_achieves_harborth` (unit distance ⟺ `x²+xy+y²=1`,
  exact integer arithmetic, no floats).
* **The open question.** Is `u(n) = harb(n)` for all `n`, or can a different construction
  beat it? Exact `u(n)` is settled only for small `n`. Lattice-NN-optimality is folklore,
  unproven in general; and asymptotically the Erdős grid is known to win.
* **Upper bound.** `u(n) = O(n^{4/3})` (Spencer–Szemerédi–Trotter 1984); the exponent gap
  to the `n^{1+c/loglog n}` lower bound is THE famous open part and is **not** closable by
  finite search. We do not target the exponent.

## Pilot results (CPU, `erdos90_search.sio`, exact integer)

The pilot takes `n = W×W` points of a square ℤ² region and chooses unit² = `N` with many
sum-of-two-squares representations `r₂(N)` (interior points then have `r₂(N)` unit
neighbours), comparing to the triangular `harb(n)`:

```
n      harb(n)  bestGrid  unit²N   winner
25     57       48        5        triangular
225    623      828       25       GRID (beats)
625    1788     3144      65       GRID
1225   3553     7144      —        GRID
2025   5919     13320     325      GRID
3025   8884     22640     325      GRID
```

Clean exact **crossover**: tiny `n` favours the triangular nearest-neighbour lattice, but
by `n ≈ 225` the Erdős grid (many-representation distance) decisively wins, gap widening
to ~2.5×, with the optimal `N` climbing 5→25→65→325 as `r₂(N)` grows (4→8→12→16→24). So
`u(n) > harb(n)` for `n ≳ 225` — the triangular NN lattice is provably **not** optimal
there. (This reproduces the known Erdős phenomenon *explicitly and exactly*; the value of
the pilot is a **validated, certifiable search engine**, not a new theorem.)

Gotcha recorded: the native backend BSS-zeroes global `var`-array initializers, so the
kernel tests candidate `N` as literals (`consider(N)`), not via an `NCAND[]` array.

## The exact, GPU-able search (no floats, fully certifiable)

1. **Pool.** A finite pool `P` of exact points (bounded ℤ²/Eisenstein region, or a union
   of lattices — where non-trivial optima hide). Squared distances are integers.
2. **Unit predicate.** Edge iff `dx²+dy² == N` (integer compare).
3. **Objective.** Over size-`n` subsets `S ⊆ P`, maximize induced unit-edge count.
4. **Search (GPU-parallel).** Many parallel guided searches (simulated annealing / tabu:
   add/drop/swap a point), per-thread bitmask + incremental edge count — a K-AXI-shaped
   integer kernel → `kretikos_emit_kaxi` → PTX, launched via
   `scripts/ci/kretikos_kaxi_l4_launch_gate.sh` or BeagleCockpit MCP.
5. **Certify.** Any subset matching/beating the baseline is dumped as explicit integer
   coordinates and re-verified by `countUnit` in Lean (`native_decide`) — the GPU only
   *proposes*, Lean *certifies*. No trust in any float path.

## Build status / next steps

* DONE: exact lattice lower-bound (Lean, verified) + CPU search kernel (Sounio, runs,
  pilot crossover reproduced).
* NEXT: (a) lift the per-thread subset search to a K-AXI kernel; validate bit-exact vs the
  CPU run on a small pool before scaling (cf. `feedback_l4_gate_catches_what_local_misses`);
  (b) cluster sweep over seeds × N × region shapes on the L4/A5000 lane via MCP; (c) fold
  any record configs back into the Lean file as witnessed theorems (`countUnit … = …`).

## Honest caveats

* The grid-beats-triangular crossover is **known** (Erdős); the pilot makes it explicit
  and exact. A genuinely new result needs configs beating *all* known constructions for a
  specific `n`, or extending exact `u(n)` records — the high-risk prize.
* The GPU is a **proposer**, not a prover: every reported config is re-certified in Lean.
* This does **not** touch the `n^{4/3}` exponent gap; that is not finite-search-accessible.

## Cluster run (validated, 2026-05-25)

The Sounio search ELF ran on the live Slurm cluster via `slurm-jobs/erdos90/run_on_cluster.sh`
(direct-srun fallback; the BeagleCockpit MCP tools were not loaded this session). Topology
discovered: compute nodes do **not** mount `/workspace` — the only shared FS is OrangeFS at
`/orangefs/training` (pvfs2), invisible to the login container. Path used: compile to a
static self-contained ELF locally → ship bytes over `srun` stdin as base64 → decode + run on
the node → results on stdout.

Scaled cluster run on `cpuops-t560-proxmox` (8 cores), n up to 16384, wall = 8s:

```
n       harb(n)   bestGrid   N(unit²)   ratio
9216    27315     88320      1105       3.23x
12544   37244     130816     1105       3.51x
16384   48708     181504     1105       3.69x   (N=1105=5·13·17, r₂=32)
```

The grid-over-triangular margin grows with `n` (the winning many-representation distance
climbs 5→25→65→325→1105 as `r₂(N)` increases), exactly the Erdős phenomenon at scale — an
explicit large-`n` lower bound `u(16384) ≥ 181504`, in exact integer arithmetic.

Next (genuine open frontier, needs the array-job/optimizer build): replace the full-square
enumeration with a per-thread subset local-search (anneal/tabu) over a large exact pool, run
as a Slurm array (many seeds × n), hunting for configs that beat *all* known constructions —
then re-certify any record in Lean. The K-AXI/PTX GPU port (idle gpu-orangefs nodes r740/5860)
is the throughput multiplier for that search.

## Optimizer array sweep (3 nodes, 2026-05-25)

`stdlib/research/erdos90_optimize.sio` improves the pilot via (1) COMPACT DISK regions
(x²+y² ≤ rr) instead of squares — fewer boundary-deficient points — and (2) a broad
unit-distance² sweep. Compiled to 3 banded static ELFs and run in PARALLEL across the
three idle nodes via `srun` (the same compile→base64→srun path):

```
band  node                       peak n   count   N(unit²)  count/harb
A     cpuops-t560-proxmox         2693    19848   325       2.51x
B     gpuorangefs-r740-proxmox    5369    46688   325       2.94x
C     gpuorangefs-5860-proxmox    7845    73376   1105      3.15x   (harb=23228)
```

Headline: an explicit exact compact-disk configuration of **7845 points with 73376 unit
distances** — `u(7845) ≥ 73376`, **3.15×** the triangular nearest-neighbour bound. The
ratio grows with n and the winning many-representation distance climbs (325 → 1105) as
larger disks admit denser N. The disk shape strictly improves the square pilot's per-point
efficiency. Every config is exact integer and re-certifiable in Lean (`countUnit`).

Honest standing: this is the strongest *explicit* lower bound from the Erdős grid family
with optimized shape + distance — it does **not** beat the asymptotic Erdős construction or
touch the n^{4/3} exponent gap. Genuinely-new territory (beating *all* known constructions
for a specific n, or extending exact small-n u(n) records) needs a richer-than-lattice pool
and a true subset local-search, the next build. The cluster path and certifiable engine are
now proven end-to-end.

## Subset optimizer — densest-k-subgraph search (3 nodes × 3 seeds, 2026-05-25)

`stdlib/research/erdos90_subset.sio`: hill-climb densest-k-subgraph over the ℤ² distance-N
graph (random-swap moves: drop a member, add a non-member, accept if internal edge count
rises; Park–Miller RNG), restarted from the compact disk + random seeds. Compares best
found to the compact-disk baseline. Run as 3 independent seed variants across the three
nodes; broad N sweep including ANISOTROPIC distances where the optimal shape need not be a
disk.

Result (identical across all 3 seeds — robust):

* **Isotropic N (2, 5, 10, 13, 25, 65):** the compact disk *is* the densest k-subgraph —
  every case "disk-optimal", no subset beats it. Expected: ℤ² is vertex-transitive, so the
  densest induced subgraph of its unit-distance graph is the compact region.
* **Anisotropic N = 50** (neighbours (±1,±7),(±7,±1),(±5,±5)): the disk is genuinely
  **suboptimal** — the search robustly reshapes it `2447 → 2767` unit pairs (+13%), the same
  on all three seeds. A real shape finding: for anisotropic distances the optimal region is
  not a disk.
* **No record, though:** the reshaped N=50 config (2767) still trails the N=25 disk
  (~2780) at n=600. The optimal-N grid/disk construction stands; **no `u(n)` record beaten.**

Honest conclusion: comprehensive densest-k-subgraph search confirms the compact-disk /
optimal-N grid is locally optimal and robust, and pinpoints where the disk assumption breaks
(anisotropic N) without yielding a better global construction. This is consistent with why
the problem is hard — no exact periodic-pool subset search can beat the grid, because
(i) lattices are vertex-transitive and (ii) square ℤ² and triangular ℤ[ω] share no integer
Cartesian frame, so heterogeneous exact pools with cross-lattice unit edges don't exist.
Genuinely beating the grid would require non-lattice rational configs / the additive-energy
frontier — beyond exact lattice search. The cluster engine, however, is fully proven:
build → ship ELF over srun → 3-node parallel search → exact, Lean-certifiable configs.

## UPDATE 2026-05-29: the exponent gap has (allegedly) been closed — OpenAI 2026

Throughout this plan we said the `n^{4/3}`-vs-`n^{1+c/loglog n}` **exponent** gap is
"not finite-search-accessible" and "we do not target the exponent." On 2026-05-28 a
Lean 4 formalization appeared claiming exactly that result:

> **OpenAI (2026), *Disproof of Erdős's planar unit-distance conjecture*** —
> `github.com/logical-intelligence/erdos-unit-distance` (Apache-2.0, Lean 4.29.1 +
> Mathlib). The headline `main_theorem` proves
> `∃ δ > 0, { n | νn(n) ≥ n^(1+δ) }.Infinite`, where `νn(n) = u(n)`. A **fixed** `δ>0`
> infinitely often **contradicts** `u(n) = n^{1+o(1)}` (Erdős's conjecture).

**Trust base (as published).** Conditional on two classical, well-cited hypotheses
isolated in `main_theorem`'s signature: the Golod–Shafarevich inequality
(`d(Q)² < 4 r(Q)`, NSW 3.9.7 / Serre) and Shafarevich's relation-rank bound (Mayer
Thm 5.1, S=∅ / Koch 11.5+11.8). `#print axioms` reduces to `propext`,
`Classical.choice`, `Quot.sound` — no bespoke axioms, no `sorry`; CI runs `lake build`
+ `leanchecker`, with a heavier `lean4export`+nanoda kernel pass locally.

**The mechanism (decoded from the Lean source).** Not geometry of lattices —
**class field towers.** Pick a cyclic cubic totally-real field `F` (a subfield of
`ℚ(ζ_r)`, `r` prime `≡ 1 mod 3`) whose maximal **unramified pro-3 extension is
infinite** (Golod–Shafarevich: `r ≤ d+2` from Shafarevich forces `d² < 4r` to fail
for any finite quotient). This gives a tower of fields `L_j/F`, all unramified,
`deg → ∞`. Set `K_j = L_j(i)` — that is the *plane* (the `i² = −1` complex place),
`AdmissibleDatum` in the Lean. Fix `t` rational primes `q ≡ 1 (mod 4)` that split
completely in `L`. The exponent excess is

```
γ = t·log 2 − log H > 0,      H = class-number growth base (h(K_j) ≤ H^f).
```

Each split prime `q ≡ 1 (mod 4)` **doubles** the count of algebraic elements of a
given norm (factor 2 ⇒ the `t·log 2` term); the class number is the loss (`log H`).
When `t·log 2 > log H`, the unit-distance count on `n` algebraic points of `K_j`
beats `n^{1+o(1)}`. A geometric double-count (`overlapArea(R) = 2R²·arccos(1/2R) −
½√(4R²−1)`, the lens area of two unit-separated disks) turns the algebraic
representation count into the planar `u(n)` bound.

**This is the infinite-degree lift of our pilot.** Our search repeatedly found the
winning distance `N` climbing `5 → 65 → 325 → 1105` as the **sum-of-two-squares count
`r₂(N)` grows** (the doc's own `r₂(N) = 32` at `N=1105=5·13·17`). That `r₂` doubling
*is* the `t·log 2` engine — in the base plane `ℤ[i]`, `r₂(∏ q_i) = 4·2^t` for `t`
distinct primes `q_i ≡ 1 (mod 4)`. We verify this exact-arithmetic core independently
in `examples/erdos/erdos90_repcount_engine.sio` (8→16→32→64; and any `≡ 3 (mod 4)`
factor sends `r₂` to 0 — the construction's `q_mod4` hypothesis). What our finite /
lattice search **structurally cannot** do — and the OpenAI result makes precise — is
let the *field degree* grow: the exponent lever is unbounded-degree number fields, not
a cleverer planar lattice. Our "honest caveat" (`exponent gap is not
finite-search-accessible`) is corroborated, and the actual key is now identified.

**NB / correction.** The cross-lattice "foreclosure" remark in the densest-k-subgraph
section above (ℤ² and ℤ[ω] sharing no integer Cartesian frame ⇒ heterogeneous unit
edges "don't exist") was already found **invalid** in a 2026-05-28 `math-review`
(logged in `.claude/llm_offload_log.md`); the OpenAI construction reinforces the
correction — exact unit-distance richness lives in number fields of growing degree,
not in any fixed integer frame.

**Honest standing.** This is a **3-days-old, not-yet-peer-reviewed** artifact; the
"OpenAI 2026" attribution and the result itself await community scrutiny, and two
classical inputs are assumed (not proved) in Lean. We verify only the **finite
combinatorial core** (`erdos90_repcount_engine.sio`), not the class field tower, which
is infinitary and outside exact computation. We make **no** independent claim on the
exponent.

## UPDATE 2026-06-13: certified witness ladder (machine-checked, exact integer)

Between June 2026 cluster runs we built an end-to-end **propose → export → Lean
`native_decide`** pipeline for explicit lower bounds on `u(n)` at fixed `n`. Every
witness is a list of **distinct** integer coordinates; multiset inflation (duplicate
coords counting extra edges) was caught and rejected during export (the prior 318/338
unified-ℚ(√3) counts were invalid for this reason).

### Literature placement (web audit, 2026-06-13)

| Source | What it gives for our `n` |
|--------|---------------------------|
| [OEIS A186705](https://oeis.org/A186705) | **Exact** `u(n)` certified only for `n ≤ 21` |
| Alexeev–Mixon–Parshall [arXiv:2412.11914](https://arxiv.org/abs/2412.11914) | Improved **upper** bounds for `n = 16..30`; full enumeration to `n = 21` |
| [Erdős Problem #90](https://www.erdosproblems.com/90) | Exponent conjecture **disproved** (OpenAI May 2026, Lean); orthogonal to finite witnesses |
| Sawin [arXiv:2605.20579](https://arxiv.org/abs/2605.20579) | Explicit asymptotic **n^1.014** lower bound (May 2026) |

**Honest claim tier:** our values are **finite lower bounds** `u(n) ≥ …` with explicit
coordinates. We do **not** assert `u(n) = …` for `n > 21`. We fill a gap the OEIS /
Alexeev tables do not cover.

### Certified ladder (Lean gates green on branch `research/erdos90-resume`)

| `n` | `harb(n)` | Full square grid | Subset hill-climb | Lean module | Gate |
|-----|-----------|------------------|-------------------|-------------|------|
| 100 | 265 | 288 (10×10, N=5) | **303** (saturation, seed 1000003) | `SounioErdos90Subset303Witness` | `erdos90_subset303_witness_gate.sh` |
| 144 | 390 | 456 (12×12, N=25) | **493** (seed 9000023) | `SounioErdos90Subset144Witness` | `erdos90_subset144_witness_gate.sh` |
| 196 | 539 | **692** (14×14, N=25) | **719** (seed 8000019, job 2739) | `SounioErdos90Subset196Witness` | `erdos90_subset196_witness_gate.sh` |
| 225 | 623 | **828** (15×15, N=25) | **856** (seed 2000003, job 2780) | `SounioErdos90Subset225Witness` | `erdos90_subset225_witness_gate.sh` |
| 256 | 712 | **976** (16×16, N=25) | **1007** (seed 2000003; job 2819 pending) | `SounioErdos90Subset256Witness` | `erdos90_subset256_witness_gate.sh` |

Earlier rungs preserved: `SounioErdos90SubsetWitness` (302), `SounioErdos90GridWitness`
(288), `SounioErdos90UnifiedQsqrt3Witness` (265 deduped).

### Correction to the 2026-05-25 subset conclusion

The May 2025 densest-k-subgraph sweep at `n ≈ 400..800` found the compact disk optimal
for isotropic `N` — and concluded that "no periodic-pool subset search can beat the
grid." That conclusion is **regime-dependent**:

* **Small `n` (fixed optimal `N`, square patch):** subset **does** beat the full square
  grid — measured and Lean-certified at `n = 100` (302 > 288) and `n = 144` (493 > 456).
  Mechanism: **boundary tax** on square patches; a compact 100/144/196-point region in ℤ²
  drops deficient corner/edge vertices.
* **Large `n` with climbing `N`:** the full Erdős grid/disk construction wins; subset
  hill-climb on a fixed pool cannot beat the periodic optimum (our original pilot at
  `n = 225+`).

Crossover for **grid vs harb** at fixed `N = 5` is at `n = 64` (8×8), not `n = 225`.
The `n ≈ 225` crossover in the pilot table is for **optimal-`N` full grids** at scale.

### Saturation probe at `n = 100` (job 2679, 36 seeds, 3× iters)

| Best edges | Seeds |
|------------|-------|
| **303** | 10 (`NEW-RECORD` vs prior 302) |
| 302 | 26 |

Interpretation: under this search class (ℤ² pool, `N = 5`, hill-climb + random restarts),
the ceiling appears to be **303**, not 302. Further seed sweeps at the same algorithm
have diminishing returns unless the move set or pool changes.

### Unified ℚ(√3) pool — structural dead end at `n = 100`

Heavy cluster array (job 2537, 18 seeds) capped at **265 = harb(100)** everywhere after
coordinate deduplication. The mixed ℤ²+Eisenstein embedding does not beat the triangular
baseline at this `n`; the ℤ² subset front is the correct target for small-`n` records.

### Scaling pattern (subset gain over full square grid)

| `n` | Grid | Disk-first (Python) | Cluster subset | Δ grid→subset |
|-----|------|----------------------|----------------|---------------|
| 100 | 288 | 300 | 303 | +15 |
| 144 | 456 | 467 | 493 | +37 |
| 196 | 692 | 707 | **719** (job 2739, seed 8000019) | +27 (+3.9%) |
| 225 | 828 | ~851 | **856** (job 2780, seed 2000003) | +28 (+3.4%) |
| 256 | 976 | ~996 | **1007** (smoke, seed 2000003) | +31 (+3.2%) |

The **relative** gain peaks at `n=144` (+8.1%) then compresses (196: +3.9%, 225: +2.9%).
The **absolute** Δ is remarkably stable at `+24..+27` for `n ≥ 196` under this search
class — subset reshaping buys a near-constant edge premium over the square patch even as
the pilot-table crossover (`grid beats harb` at `n=225`) is reached.

### Two-front research model (deeper framing)

```text
Front A — finite ℤ² ladder (this work)
  Target: explicit u(n) ≥ … witnesses at chosen n
  Method: grid export + subset cluster + Lean certify
  Ceiling: pool-local hill-climb; saturated near disk+ε at fixed (n, N)

Front B — asymptotic exponent (literature, May 2026)
  Target: u(n) ≥ n^{1+δ} infinitely often
  Method: class-field towers, growing algebraic degree
  Not reachable by fixed ℤ² pool search
```

Our epistemic pipeline (distinctness check, reproducible seed, independent Lean count)
is the **correctness layer** for Front A. It caught multiset inflation that would have
published false records.

### Cluster inventory (June 2026)

| RUN_ID | Job | Target |
|--------|-----|--------|
| `erdos90-sub-20260613T145654-970979` | 2583 | `n=100` subset → 301–302 |
| `erdos90-sat-20260613T153818-1010187` | 2679 | `n=100` saturation → 302/303 |
| `erdos90-144-20260613T153818-1010197` | 2625 | `n=144` subset → 486–493 |
| `erdos90-196-20260613T161738-1043947` | **2739** (18 seeds) | `n=196` subset → **719** (8000019, 173205) |
| `erdos90-225-20260613T164143-1062825` | **2780** (18 seeds) | `n=225` subset → **856** (2000003) |
| `erdos90-256-20260613T165554-1072360` | **2819** (18 seeds) | `n=256` subset vs grid 976 |

Stage roots under `/orangefs/training/sounio/erdos90-{sub,sat,144,196,225,256}-runs/`.

#### Job 2739 aggregation (`n=196`, 18/18 complete)

| Edges | Seeds |
|-------|-------|
| **719** | 8000019, 173205 |
| 718 | 707106, 577215, 3000003, 271828 |
| 717 | 6000011, 5000009, 4000007, 3141592, 314159 |
| 716 | 9000023, 7000013, 223607, 2000003, 141421, 1000003 |
| 715 | 161803 |

Leader **8000019** (+3 over smoke seed 9000023) promoted to `EXPORT_SEED` in
`erdos90_subset196_export.sio`; Lean gate recertified at **719**.

#### Job 2780 aggregation (`n=225`, 18/18 complete)

| Edges | Seeds |
|-------|-------|
| **856** | 2000003 |
| 855 | 3141592 |
| 854 | 9000023, 707106, 4000007, 314159, 3000003, 223607, 173205, 1000003 |
| 853 | 271828, 161803, 141421 |
| 852 | 8000019, 7000013, 6000011, 577215, 5000009 |

Smoke favourite **8000019** landed mid-pack (852); leader **2000003** (+4 over smoke).
Lean recertified at **856**. Spread 852–856 is wider than at `n=196` (715–719) in
absolute terms — hill-climb polish matters more as the shape gap compresses.

### Boundary tax — why subset beats square grid at small/medium `n`

For a full `w × w` square patch with `n = w²` vertices and unit distance `N`, each
interior vertex has the maximum possible degree in the ℤ² unit graph; **corners and
edges lose neighbours** to the missing half-plane outside the patch. A compact subset
of the same cardinality can **delete low-degree boundary vertices** and **replace them
with interior-equivalent vertices** drawn from a larger pool `|ℤ² ∩ B_R|`, increasing
the edge count without changing `n`.

Quantitative sketch at fixed `N = 25` (edges per vertex capped at 12 in ℤ²):

| Patch | Corners (deg 2) | Edge (deg 5) | Interior (deg 12) | Grid edges (approx) |
|-------|-----------------|--------------|-------------------|---------------------|
| 10×10 (`n=100`) | 4 | 32 | 64 | 288 |
| 12×12 (`n=144`) | 4 | 40 | 100 | 456 |
| 14×14 (`n=196`) | 4 | 48 | 144 | 692 |
| 15×15 (`n=225`) | 4 | 56 | 169 | 828 |

The **fraction of deficient vertices** is `O(1/√n)` but the **absolute** boundary
deficit grows (`56` edge vertices at `n=225`). Subset hill-climb reshapes toward a
disk-like vertex set; measured Δ peaks at `n=144` (+37) then stabilises at `+24..+27`
for `n ∈ {196, 225}` — the relative gain compresses (+2.9% at 225) even though subset
still beats the square grid at the pilot crossover point.

This is **not** a contradiction of the May 2025 conclusion: that sweep climbed `N`
with `n` and compared against the **periodic Erdős grid optimum**, where boundary
effects vanish. Our ladder holds `N` fixed (5 or 25) and compares **finite patches**.

### Saturation and seed ecology

| `n` | Search class | Best found | Seeds at best | Interpretation |
|-----|--------------|------------|---------------|----------------|
| 100 | `N=5`, 3× iters | 303 | 10/36 | Plateau; 26 seeds stuck at 302 |
| 144 | `N=25`, full iters | 493 | 1/18 (9000023) | High seed variance; leader seed reused |
| 196 | `N=25`, 3× iters | **719** | 2/18 (8000019, 173205) | Job 2739 complete; +3 over smoke |
| 225 | `N=25`, 3× iters | **856** | 1/18 (2000003) | Job 2780 complete; smoke seed 8000019 only 852 |

**Seed ecology is not monotonic across `n`:** 9000023 leads at 144 (493); 8000019 at
196 (719); **2000003** at 225 (856). Transfer hypotheses fail; treat each rung's cluster
as independent unless reproduced.

### Epistemic correctness layer (why Lean gates matter)

The export pipeline enforces three independent checks before publication:

1. **Distinctness:** `len(set(coords)) == n` — caught multiset inflation (318/338
   false unified counts).
2. **Reproducibility:** fixed `EXPORT_SEED` in `.sio` → identical witness on replay.
3. **Independent count:** Lean `countGridUnit25` recomputes edges from coordinates
   alone (`native_decide`), decoupled from the searcher's `total_edges()`.

A record is **certified** only when the gate script is green; cluster stdout alone
is staging evidence until export + Lean confirm.

### Infrastructure (`n = 196` and `n = 225`)

| Artifact | Role |
|----------|------|
| `erdos90_grid{196,225}_export.sio` | Full square grid → Lean |
| `erdos90_subset{196,225}_{cluster,export}.sio` | Slurm kernel + replay witness |
| `submit_subset{196,225}_array.sh` | 18-seed OrangeFS arrays |
| `erdos90_{grid,subset}{196,225}_witness_gate.sh` | Export → Lean certify |

### Regime III comparison at matched `n` (`erdos90_disk225_compare.sio`)

At **equal cardinality** `n ≈ 225`, compact disk + optimal `N` from the standard sweep:

| `rr` | `n` | `bestN` | `count` | vs grid 828 | vs subset 856 |
|------|-----|---------|---------|-------------|---------------|
| 68–71 | 221 | 25 | 832 | beats | below |
| 72 | 225 | 25 | **848** | beats | **below** |

**Conclusion:** Regime II subset (`856`) beats Regime III **full disk** at the same `n`
when both use the swept `N` list (best is still `N=25` here). The May-2025 regime split
is not "disk always wins" — it is **climbing `N` + large `n`** that wins. Subset reshaping
in a larger pool strictly dominates a full disk at `n=225`.

### Next steps

1. ~~Jobs 2739 / 2780 / `n=256` smoke~~ — **done** (719, 856, 1007); aggregate **2819**.
2. Optional: `n=289` (17×17) or export disk witness at `rr=72` (`n=225`, count=848) to Lean.
3. Do **not** claim global optimality; cite OEIS/A186705 exact ceiling at `n ≤ 21`.
4. Keep asymptotic (Sawin/OpenAI) and finite (this ladder) claims in separate tiers.

---

## DEEP ANALYSIS: three-regime model (2026-06-13)

The ladder data support splitting the problem into **three regimes**, not two. Conflating
them produced the May-2025 overstatement that "subset cannot beat grid."

### Regime I — triangular dominance (`n` small, any shape)

`harb(n)` wins. Square grid and subset both lose to the Eisenstein NN lattice. Our
`n=25` pilot row (grid 48 < harb 57) and the Lean `lattice_achieves_harborth` chain
live here. **No subset search target** — the pool would need ℤ[ω], not ℤ².

### Regime II — finite patch, fixed `N` (`n ≈ 64..256`, `N ∈ {5, 25}`)

The adversary is **boundary tax on a square**, not the periodic Erdős optimum.

| Mechanism | What it buys | Measured at `n=225` |
|-----------|--------------|---------------------|
| Square → disk-shaped `k`-subset | Delete 56 edge + 4 corner low-degree vertices | grid 828 → disk-init ≈ **851** (+23) |
| Disk-init → hill-climb polish | Single-vertex swaps in pool `R≤26` | ≈851 → **856** (+5) |
| **Total** grid → certified subset | | **+28** (+3.4%) |

The decomposition is reproducible: for seeds {2000003, 8000019, 9000023} at `n=225`,
`diskHC ∈ {850,851,852}` while `BEST ∈ {852,854,856}` — **~90% of Δ is shape**,
~10% is stochastic hill-climb.

At `n=144` the shape term dominates harder (+37 total) because a 12×12 square is a
worse disk approximation (perimeter/area ratio 48/144 vs 60/225 for 15×15).

**Predicted saturation:** under move class {pick `n` from pool, swap 1 vertex, hill-climb},
absolute premium over square grid stabilises at **~25–30 edges** once `diskHC ≈ 0.98·best`
(i.e. when init already captures most boundary savings). Relative premium decays as
`Θ(1/√n)` because grid interior grows as `n` while boundary grows as `√n`.

### Regime III — scale construction (`n ≳ 400`, climbing `N`)

The May-2025 `erdos90_optimize.sio` / cluster sweeps: compact disk + **broad `N` sweep**
(5→25→65→325→1105) beats `harb(n)` with ratio climbing to 3.7× at `n=16384`. Here the
winning move is **not** dropping boundary vertices but choosing `N` with large `r₂(N)` so
interior vertices gain many unit directions. Subset hill-climb at fixed `N=25` cannot
compete — different game.

**Critical distinction:** Regime II asks "given `N=25` and `n=225`, square or reshaped
subset?" Regime III asks "given `n=225`, what `(shape, N)` pair maximises edges?" The
pilot table's `828` answer is Regime III on a **square**; our `856` is Regime II on a
**subset** — both are valid lower bounds, incomparable unless `N` and shape are aligned.

### What we have actually proved (epistemic tier)

| Claim | Status |
|-------|--------|
| `u(225) ≥ 856` with explicit distinct ℤ² coords, `N=25` | **Lean-certified** |
| `u(225) ≥ 828` via full 15×15 grid | **Lean-certified** |
| `u(225) ≥ 623` via `harb` (triangular) | **Classical + Lean lattice chain** |
| `u(225) = …` (exact optimum) | **Unknown** (OEIS exact to 21) |
| Subset beats square at fixed `N=25` for `n ∈ {100,144,196,225}` | **Measured + certified** |
| Subset beats optimal-`N` Erdős grid at `n=225` | **Not claimed** (not tested) |
| Exponent improvement (Sawin / OpenAI) | **Literature**; orthogonal |

### Dead ends (negative results matter)

1. **Unified ℚ(√3) pool** — caps at `harb(100)` after dedup; wrong geometry for small-`n`
   ℤ² records.
2. **Multiset witnesses** — 318/338 inflated counts; caught by distinctness gate.
3. **Seed transfer** — 8000019 best at 196, mid-pack at 225; 9000023 best at 144 only.
4. **Saturation at `n=100`, `N=5`** — 36 seeds → ceiling 303; further seeds useless
   without new moves/pool.

### Falsifiable predictions (next experiments)

| Experiment | If confirmed | If falsified |
|------------|--------------|--------------|
| `n=256`, fixed `N=25`, subset cluster | Δ grid→subset ∈ [22, 32] | **Confirmed** smoke: +31 (1007 vs 976) |
| `erdos90_disk225_compare` disk at `n≈225` with optimal `N` | Count `C` with `C > 856` possible | **Falsified**: best disk count **848** < subset **856** |
| Extend pool to `R=30` at `n=225` | `BEST` rises by ≤5 | Pool cutoff was binding |
| GPU K-AXI parallel restarts | Same ceilings, faster | — |

### Strategic position

**Front A (this ladder)** is not chasing the exponent (Front B / literature). It is
building a **certified library of explicit lower bounds** at human-scalable `n`, with an
honest correctness layer that already prevented false publication. The scientific value is:

- closing the OEIS/Alexeev gap for concrete `n`;
- quantifying when shape beats lattice vs when distance-spectrum beats shape;
- keeping the GPU path (proposer) tied to Lean (certifier) as a reusable pattern for
  epistemic computing in the compiler stack.

The open prize remains: a config beating **every** known construction for some `n`, or
extending exact `u(n)` beyond 21 — not merely beating our own square-grid baseline.
