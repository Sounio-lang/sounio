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
