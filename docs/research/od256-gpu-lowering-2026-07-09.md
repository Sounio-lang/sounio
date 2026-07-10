<!-- docs:meta
topic_id: repo.docs.research.od256-gpu-lowering-2026-07-09
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.od256-gpu-lowering-2026-07-09
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# od256 GPU octuple — K-AXI → PTX lowering (M3 sketch)

Date: 2026-07-09
Companion to `od256-oct-double-spec-2026-07-08.md`. This is the "GPU octuple"
design: how the software oct-double lowers through K-AXI → PTX → CUBIN.

## Parallelism model

Multi-double arithmetic is **sequential within one value** (renormalization is a
data-dependent carry chain) but **embarrassingly parallel across vector
elements**. So the GPU model (CAMPARY's) is: **one thread per od256 element**.
Each thread loads 8 limbs of `a` and 8 of `b`, runs the whole EFT+renorm chain in
its own registers, and stores 8 result limbs. No cross-thread communication; no
shared memory needed for the core op. This is exactly the axis K-AXI already
parallelizes (`GET_BID`/`GET_NTID`, warp size 8).

## The one real GPU win: FMA two_prod

CPU `dd64` computes `two_prod` with a **Dekker split** (`dd_split` + 4 mults + 3
adds ≈ 10 f64 ops) because it avoids assuming hardware FMA. GPUs have native
`fma.rn.f64`, so on GPU:

```
p = a*b ;  e = fma(a, b, -p)      // 1 mul + 1 fma = 2 ops
```

That is ~5× fewer ops per partial product, and od_mul is 64 partial products —
the dominant cost. **Caveat (must be stated):** FMA-two_prod and split-two_prod
produce the *same value* but can differ in the last bit of the error term, so a
GPU od256 result may differ from the CPU od256 result in low limbs. Decision
point: either (a) accept a documented CPU/GPU lane difference (both faithful to
~424 bit), or (b) switch CPU `dd64::two_prod` to FMA when available for
bit-parity. Recommend (a) for now; revisit if a cross-lane bit-exact gate is
wanted. `two_sum` is identical on both (6 add/sub, no FMA).

## Instruction sequences (hand-written reference)

`tests/golden/kaxi_ptx/od256/eft_primitives.ptx` is the hand-written PTX for the
two building blocks (validate on a GPU worker: `ptxas -arch=sm_50
eft_primitives.ptx -o /dev/null`):

- `two_sum(a,b)` → 6 × `add/sub.rn.f64`.
- `two_prod(a,b)` → `mul.rn.f64` + `neg.f64` + `fma.rn.f64`.

K-AXI opcodes already cover these: `ADD=1`, `SUB=13`, `MUL=2`, `FMA=4`
(`self-hosted/gpu/kaxi_backend.sio:17-47`), and `kaxi_to_ptx.sio` already emits
`add/mul/fma/sub.rn.f64` into `%fd` registers. So the primitives need **no new
K-AXI ops** — only a higher-level pattern that emits the right sequences.

## Renorm on GPU: unroll, don't loop

The CPU `od_renorm` uses a data-dependent `while` loop (Shewchuk grow-expansion).
On GPU, branchy per-thread loops hurt warp coherence. Because the limb counts are
**fixed** (od_add: 16→8; od_mul: ≤128→8), the renorm should be **fully unrolled**
into a straight-line `two_sum` sweep (K-AXI has `BRANCH`/`SETP`/`RECONVERGE` if
truly needed, but unrolled is preferred). Unrolled renorm = a fixed sequence of
`two_sum`s + a final compress — all `add/sub.rn.f64`, no divergence.

## Register budget

One thread needs: 8 (`a`) + 8 (`b`) + ~16 scratch (expansion + carries). PTX
`%fd<70>` (the current kaxi_to_ptx f64 bank) is comfortably enough; the b64
address bank `%rd<260>` covers the `param_mem + i*8` limb addressing. No spill
expected for add; mul's 128-term unroll may need staged accumulation but stays
within the bank.

## Emitter extension (the actual work item)

Add an `od256` lowering pattern to `kaxi_to_ptx.sio` (and a driver mode in
`kretikos_kaxi_to_ptx.sio`, e.g. `--od256`), analogous to the still-scaffold
`f32_assoc_gum` octonion lane (`tests/golden/kaxi_ptx/f32_assoc_gum/README`).
Concretely:
1. emit per-limb `ld.global.f64` for the 8+8 inputs,
2. emit the unrolled `two_sum`/`fma two_prod` sequences,
3. emit the unrolled renorm sweep,
4. emit per-limb `st.global.f64` for the 8 outputs.

## Validation path

1. `ptxas -arch=sm_50` accept (no GPU needed) via
   `slurm-jobs/kaxi-ptxas-accept/run_ptxas.sh` — assemble to `/dev/null`.
2. Golden PTX byte-gate: capture into `tests/golden/kaxi_ptx/od256/` once the
   emitter produces it (`scripts/ci/kaxi_ptx_golden_gate.sh` pattern).
3. Numerical gate: run the CUBIN on the Slurm GPU worker (L4 / GB10) over random
   inputs and compare to the **same mpmath oracle** as the CPU lane
   (`scripts/ci/od256_mpmath_gate.py`), asserting ≥ ~410 effective bits.

## Status

- Design + hand-written EFT PTX: this doc + `eft_primitives.ptx`. **Sketch done.**
- **Emitter STARTED (2026-07-10):** the `od256_two_sum` pattern is wired into
  `kretikos_emit_kaxi.sio` (`kaxi_emit_od256_two_sum_asm`) + the driver
  `kretikos_kaxi_to_ptx.sio`. It emits **correct f64 PTX end-to-end** (6 ×
  `add.f64`/`sub.f64` on `%fd`, `ld/st.global.f64` for data, `.u64` addressing) —
  captured golden `tests/golden/kaxi_ptx/od256/od256_two_sum.ptx`, matching the
  hand-written reference's 6-op two_sum. Key detail: value lines carry a
  `type=f64` annotation (address lines do not) so mode-0 lowers them to `.f64`.
- **`two_prod` DONE (2026-07-10) via Dekker split.** K-AXI `fma` is hardcoded
  `a*b+a` (not general), so `two_prod` uses the split path instead: `p=a*b`;
  `split(x): c=C*x; hi=c-(c-x); lo=x-hi` with `C=2^27+1`;
  `e=((ah*bh-p)+ah*bl+al*bh)+al*bl` = 7 `mul.f64` + 7 `sub.f64` + 3 `add.f64`.
  Emitted PTX (`tests/golden/kaxi_ptx/od256/od256_two_prod.ptx`) is
  **bit-identical to the mpmath-validated reference over 200k random inputs**.
  This required a small **transpiler extension**: an f64 immediate
  (`load_immf rD, fimm64=<16 hex>` → `mov.f64 %fd, 0d…`, mode 0) in
  `kaxi_to_ptx.sio` (`ptx_parse_fimm64_hex` + `kaxi_lower_load_immf` +
  dispatch). f32 patterns unchanged (verified). This f64-immediate also unblocks
  the Newton constants (1.0=`3FF0…`, 0.5=`3FE0…`) needed later for GPU div/sqrt.
- **`od256_add` renorm DESIGNED + VALIDATED (2026-07-10, branchless).** The crux
  is the 16→8 renormalization. Finding: simple fixed schemes (VecSum + truncate)
  cap at ~267 bits — the CAMPARY *compaction* step is essential for octuple. The
  compaction's data-dependent "advance output slot" branch is emulated
  **branchless via SELP**: at each step, K `SELP`s keyed on `SETP_EQ(j,cnt) &
  SETP_NE(err,0)` write the committed limb to a fixed array, and `cnt` advances
  by a predicated ADD. Validated in `scripts/ci/od256_renorm_gpu_ref.py`:
  **bit-identical to the branchy CAMPARY algorithm (8000/8000)** and **429.9
  effective bits** vs mpmath. Full pipeline = branchless merge (a,b each sorted →
  16 desc, an odd-even merge network of abs+SETP+SELP compare-exchanges) →
  VecSum (15 fixed `two_sum`) → SELP-VSEB (15 steps × [fast_two_sum + K SELP]).
  Only fixed ops: `add/sub.f64`, `setp`, `selp` — all in the K-AXI ISA
  (`SETP_NE=35`, `SETP_EQ=17`... err, `SETP_EQ` per backend, `SELP=26`). Emit is
  now mechanical (~500 ops); the hard numerical design is done.
- **`od256_add` EMITTED + VALIDATED (2026-07-10).** `kaxi_emit_od256_add_asm`
  (`kretikos_emit_kaxi.sio`) + driver dispatch. Pipeline: **interleave** a,b → 16
  (no merge network — interleave + 2× VecSum reaches octuple, so the odd-even
  merge network is unnecessary) → 2× VecSum (30 `two_sum`) → branchless SELP-VSEB.
  Emitted PTX `tests/golden/kaxi_ptx/od256/od256_add.ptx` (1889 lines: 118
  `add.f64`, 150 `sub.f64`, 263 `selp.f64`, 143 `setp`; max `%fd59 < 70`).
  **Validated by full PTX simulation** (addressing + branchless logic):
  **3000/3000 bit-identical to the interleave reference and 430.5 effective bits
  vs mpmath.** Required f64 setp/selp (added) + f64 immediate (added).
  Also fixed: `od256_two_prod` register numbering (was `%fd77` > 69, invalid PTX;
  renumbered to `%fd≤59`, re-validated 50k bit-identical).
- **`od256_mul` EMITTED + VALIDATED (2026-07-10).** 43 partial products
  `two_prod(a[i],b[j])` for `i+j<9` (rest < 2^-424; splits precomputed once per
  limb) → 86 terms → **3× VecSum (in-place) → branchless SELP-VSEB**. Emitted
  `tests/golden/kaxi_ptx/od256/od256_mul.ptx` (11407 lines: 231 `mul.f64`, 994
  `add.f64`, 1281 `sub.f64`, 1453 `selp.f64`). **PTX-simulated: 400/400
  bit-identical to the reference and 429.8 effective bits vs mpmath.**
  - **Enabler — dynamic `%fd` declaration.** The 86-term renorm needs >70 f64
    registers, but the header was a fixed `.reg .f64 %fd<70>`. `kaxi_max_f64_reg`
    scans the K-AXI asm for the max register on `type=f64` lines and the header
    now declares `max(70, N+1)`. **Non-breaking**: patterns with ≤69 f64 regs
    still emit `%fd<70>` byte-identical (verified: two_sum/two_prod/add/octonion
    goldens unchanged); od256_mul gets `%fd<168>`.
- **Follow-on:** ptxas-accept wiring, golden-gate pattern-list entry, GPU numeric
  gate on the worker (L4/GB10). All four arithmetic kernels
  (two_sum/two_prod/add/mul) are now emitted + simulation-validated.

Related: `od256-oct-double-spec-2026-07-08.md`; skills `gpu-kaxi-ptx-cubin`,
`epistemic-types-hypercomplex`.
