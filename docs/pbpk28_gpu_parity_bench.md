# PBPK-28 GPU parity and A5000 benchmark record

Branch `pbpk-28-gpu-parity`, commit `5e9551c2f8`, 2026-10-31.
Covers: GUM + population-MC kernels, per-drug parity gates with sabotage
arms, signedness and bank fixes in the K-AXI to PTX lowering, device
bitwise re-verify, and the RTX A5000 vs CPU benchmark.

## Parity evidence

- Golden PTX gate `scripts/ci/kaxi_ptx_golden_gate.sh`: 318/318 byte-identical
  after regenerating 14 goldens whose only change is `div`/`setp` u64 to s64
  plus sha256 updates. Audited: every changed PTX line is one of those ops.
- Sim parity gate `scripts/ci/dissertation_pbpk28_gpu_parity_gate.sh`:
  10/10 cases green (rapamycin and semaglutide GUM, MC substreams, draws,
  emitter, sim, both sabotage arms). Venlafaxine, haloperidol, midazolam
  GPU-GUM arms stay PENDING until their ep28 priors land.
- Device bitwise re-verify on RTX A5000, driver API, single context:
  - GUM `kaxi_kernel`, 8 threads, synthetic exact inputs: all 8 out words
    match the C model bit for bit (0.25, 2, 6.75, 16, 31.25, 54, 1.75, 18).
    Repeat run identical. Real-magnitude smoke matches bit for bit.
  - MC advance, 1 thread: out words 12345, 12345, 3023790853, 12345, 12345,
    2478282264, 545508589 equal the CPU `mrg32k3a_next_u01` reference.
- Every launch in every run returned sync rc 0. Sabotage arms (corrupted
  combine output, wrong-seed reference) fail as required.

## Root causes fixed for device correctness

1. Integer `div` and `setp` lowered as unsigned (`div.u64`, `setp.*.u64`)
   while K-AXI sim and the CPU reference are signed i64. Negative residues
   (second MRG32k3a component) diverged on device. Now `div.s64` and
   `setp.*.s64` in modes 0 and 1.
2. GUM address math used `types=2` (32-bit `%r` bank) while address
   consumers use the 64-bit `%rd` bank, so loads and stores missed.
   The typed lowerer now accepts `op_type=3` (untyped 64-bit integer) for
   ADD, SUB, MUL, DIV, and the GUM kernel uses it for address bases.
   Follow-ups, not regressions: `min`/`max` keep unsigned defaults, and
   the `i32` lane is unchanged. Neither is exercised by these kernels.

## Versions

- GPU node gpuorangefs-multi-r740-proxmox: NVIDIA RTX A5000 24564 MiB,
  cc 8.6, driver 595.71.05, CUDA driver API 13020, libcuda from the OS.
  A second idle Quadro RTX 8000 shares the node and was not used.
- PTX: transpiler emits `.version 7.0` / `.target sm_50`; assembled with
  host ptxas 12.8.61 to sm_86 cubins: gum 2728 bytes
  (sha256 6e72f7b68bac4c1609b57dc775e090bc34a785a02809504ec0be46c013f45f8d),
  mc 5096 bytes
  (sha256 44963e25bdb0a856f61024bf77ee8d57551b8d762aa10f9d6251734e9326905e).
- CPU node (same machine): Intel Xeon Gold 6148 at 2.40 GHz,
  kernel 7.0.2-5-pve, glibc 2.39.
- Toolchain: Madaros v0.80.0 via `bin/souc` (local build 2026-10-23, not
  the committed ELF; sources read at runtime, so the fix applies).
  CPU bench ELFs built with `souc compile` from this tree.

## Benchmark, per unit steady state, median of 5 (GPU) and 3 to 5 (CPU)

GUM combine, one 8-lane f64 evaluation:

- GPU launch plus sync, same cubin relaunched: 6.7 us at N=2e4 and 2e5,
  7.0 us at N=2e6. Flat with N, so launch bound.
- CPU compiled ELF slope: 6.2 us per combine (N=200 to N=2e5).
  Fixed host cost about 0.83 s per process (ELF startup plus the 17-solve
  Jacobian both paths need once per drug and endpoint).
- Verdict: near tie per unit. The GPU wins only with a batched
  multi-evaluation kernel, which is the proposed next step.

Population MC, one draw (substream advance plus polar plus lognormal):

- GPU with the current seed-baked kernel: launch floor 7.0 us, but each
  draw needs its own cubin, so the honest per-draw pipeline is 9.36 s
  souc emit plus 0.03 s ptxas plus launch, about 9.4 s per draw.
- CPU compiled ELF: 68 us per substream advance steady state.
  Triangular totals from the substream loop: 6.8 ms per draw at N=200,
  20.4 ms per draw at N=600, 68.3 ms per draw at N=2000.
- Verdict: the current kernel design is recompile bound on device and
  triangular on host. A memory-fed advance kernel (state in, state out,
  one cubin, N threads) is required for population scale on GPU.

## What lowers to PTX today (verified)

Lowered and device proven: untyped integer ADD, SUB, MUL, DIV (signed),
integer SETP and SELP, F64 MUL and LOAD/STORE GLOBAL, GET_TID, RET.
Lowered with golden coverage but not device run here: FMA, MIN, MAX, ABS,
NEG, SQRT, RECIP, LG2, EX2, SIN, COS, RSQRT, F32 lanes, epistemic dual
lane, vote, shfl, warp reduce, BAR.SYNC, BRA. Not lowered: shared-memory
load and store (TODO comments), modes without goldens.
Integer MIN and MAX still lower unsigned and need a signedness pass.

## Reproduce

Emit from the repo root with `bin/souc run` on the two emit scripts,
assemble with ptxas 12.8.61 `-arch=sm_86 -m64`, launch `kaxi_kernel`
with one device pointer (GUM 8 threads over a 24-double buffer with
d, v, out slots; MC 1 thread over a 7-word buffer), sync, copy back.
Slurm jobs in this record used `--gres=gpu:a5000:1` with output to the
shared bench dir and base64 payload injection with sha256 checks.

## Addendum: integer defect class sweep, 2026-10-31

A systematic sweep of every integer lowering followed, because the suite
had passed 318 goldens and 10 parity cases with two live miscompiles.
Findings and fixes on top of commit `ad16753b6c`:

1. `min` and `max` on the integer lane lowered unsigned (`min.u64`,
   `max.u64`) in all modes. Sounio integer min and max are signed, so
   negative operands diverged. Now `min.s64` and `max.s64` on the
   integer lanes of every mode. The float lanes are untouched.
2. The `i32` lane (`type=i32`, `%r` bank) is coherent within
   load, arithmetic, and store, and no in-tree pattern mixes it with
   setp, selp, or address consumers, which all read `%rd`. Verdict:
   keep, with the contract pinned in the transpiler header: i32 values
   never feed addresses, setp, or selp. The GUM kernel was the only
   mixer and now uses type 3.
3. Negative immediates are not representable: the sim stream packs
   16-bit unsigned fields and the assembler clamps negatives to 0,
   while the transpiler would emit a wrapping `mov`. So `load_imm`
   accepts 0 to 65535 only, and kernels must build negatives
   arithmetically (the oracle does `sub r1, r21, r20`). The larger
   MC immediates already follow this rule with hi and lo parts.
   No behavior was changed here; the contract is now documented.

Device oracle `tests/run-pass/gpu_int_oracle.sio` (also run by the
suite for sim and PTX asserts): one thread computes div, all six
setps, min, max, abs, neg, selp, add, sub, and mul on -7 and 3,
stores 16 words. Sim asserts every covered slot with negatives;
PTX asserts pin the s64 lowerings plus a sabotage arm. On the A5000
all 16 device words matched bit for bit:
-2, 0, -7, 3, 7, -3, -7, -4, -10, -21, 1, 0, 1, 0, 0, 1.
## Addendum: memory-fed population-MC kernel, 2026-10-31

The per-draw kernel baked MRG32k3a state into immediates, so every
draw paid a full recompile (about 9.4 s: 9.36 s souc emit plus 29 ms
ptxas) against a 68 us CPU step. The memory-fed emitter
(kaxi_emit_pbpk28_mc_advance_memfed_asm) takes no state arguments:
one cubin, N threads, per-thread state at mem[tid*9 .. tid*9+6),
z out at mem[tid*9+6]. The host uploads substream states and reads
back advances; one compile serves any draw count and any seed.
Emit cost is unchanged in kind (9.41 s souc emit, measured this
round) and is now paid once instead of per draw. Per-draw device
launch latency was not separately timed this round; the structural
per-draw cost is one HtoD (1152 B for 16 draws) plus one launch plus
one DtoH. The GPU is not claimed faster than the CPU here; the win
is compile-once plus batching, with timed batch kernels later work.

Correctness, all bitwise. Sim: two legs, tid 0 reads slice 0
(imm-fed equivalence) and tid 1 reads slice 1 (stride addressing),
both against mrg32k3a_next_u01. PTX: get_tid lowers to
mov.u32 plus cvt.u64.u32, state moves through ld.global.u64 and
st.global.u64, arithmetic through the signed integer path
(div.s64, setp.*.s64, selp.b64); no f32 or s32 in the integer lane.
Device (RTX A5000, cc 8.6, driver 595.71.05, PTX JIT-loaded,
sync rc=0): 16 threads, 144 of 144 words exact versus the Sounio
reference. That is 96 state words plus 16 z words equal to the CPU
next state per substream, plus 32 gap words still zero, which proves
no thread touched a neighbour slice.

The first device attempt (job 20694) ran but compared against a
packed 96-word upload while the kernel strides 9 words per thread,
so only slice 0 was valid input. The rerun (job 20777) uploads a
strided 144-word buffer with explicit zero gaps and diffs all 144
words. The misaligned run is kept as evidence that the comparison
is sensitive: wrong inputs produced visibly wrong words, including
z = 4294967087 from zero-state slices.

## Appendix: integer-op audit, 2026-10-31

Every integer lowering after the signedness fixes, with verdict and
evidence. CORRECT means sim, PTX, and device agree on the Sounio i64
semantics; FIXED means it miscompiled before this round.

- add, sub: add.u64 and sub.u64 in every mode. Two-complement
  wraparound is bit-identical to signed, so CORRECT. Oracle sim,
  PTX, and device (values include -7, 3, -4, -10).
- mul: mul.lo.u64 in every mode. CORRECT for in-range products; the
  MRG32k3a products stay below 2 to the 48. Oracle plus both MC
  kernels on device.
- div: div.s64 in every untyped mode. FIXED (was div.u64). Oracle
  covers negative dividends and divisors; device 16 of 16.
- setp lt, le, eq, ge, gt, ne: setp.*.s64. FIXED (were unsigned).
  All six oracle-covered, sim plus PTX plus device.
- selp: selp.b64 on the predicate bit. CORRECT, type-agnostic by
  construction. Oracle plus mem-fed device (z select included).
- min, max: min.s64 and max.s64 in every mode. FIXED (were u64 in
  all modes). Twelve goldens regenerated with ub64-to-s64-only
  diffs; golden gate 318 of 318; oracle negatives on device.
- abs, neg: abs.s64 and neg.s64. CORRECT already; signed from the
  start. Oracle-covered (-7 to 7, 3 to -3).
- load_imm: mov.u64, accepts 0 to 65535. CORRECT with the
  documented contract; negatives must be built arithmetically
  (the oracle does sub r1, r21, r20; MC splits mults hi and lo).
- load_global and store_global, untyped: ld.global.u64 and
  st.global.u64 on the %rd bank. CORRECT. GUM, MC, oracle, and
  mem-fed kernels all device-verified through this path.
- load_global and store_global, type i32: ld.global.s32 and
  st.global.s32 on the %r bank. Coherent within load, arithmetic,
  and store by code inspection; the pinned contract keeps i32
  values out of addresses, setp, and selp. Negative-operand and
  overflow device behavior unexercised; no in-tree kernel drives
  the i32 lane out of range.
- load_global and store_global, types f64 and f32: unchanged float
  lanes. CORRECT, pre-existing; GUM device-verified bitwise.
- get_tid, get_bid, get_ntid: special regs through u32 to u64.
  CORRECT. Mem-fed device proof addresses 16 slices through tid.
- mul_imm, add_imm: lower to load_imm (known quirk, noted in the
  source). No in-tree integer kernel depends on immediate folding.
- and, or, xor, shl: no lowering arms; they fall through to nop.
  GAP, openly: no in-tree kernel uses them, and the emitter tests
  assert nop-free text, so a future user gets a loud failure, not
  a silent wrong answer.
- integer atomics (add, sub, min, max): no lowering arms; only
  atomic_addf is reachable in the transpiler. GAP, same status as
  bitwise ops: unused, nop-guarded by the emitter asserts.
- fma on the integer lane: lowers through f32. FMA is a float op
  with no exact integer semantics claimed; no in-tree integer
  kernel emits it. Left as-is, recorded here.
- div boundary: every divisor in every proof is a nonzero
  constant. Divide-by-zero is uncompared (sim Panic versus PTX
  undefined); no kernel divides by a computed zero.

