<!-- docs:meta
topic_id: repo.docs.research.solver-gpu-native-path-2026-06-27
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-gpu-native-path-2026-06-27
-->

<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver lane GPU path — boundary findings (2026-06-27)

A Level-3-frontier probe toward the lane's target (showcase Sounio's native GPU
path + machine-code control for the solver). The result is a **boundary map**,
not a win: **the checked public GPU kernel subset cannot yet express a solver
kernel.** This is the honest frontier finding — recorded so the next attempt
starts from the real gap rather than re-discovering it.

Evidence tags per `docs/audit/GPU_PIPELINE_SOTA_ASSESSMENT_2026-05-30.md`:
**MEASURED** / **BOUNDED** / **SOURCE-ONLY**. No claim rests on a projection.

## Claim boundary

This note makes **no** GPU solver claim. It maps what the public, checked GPU
kernel surface (`artifacts/omega/souc-bin/souc-linux-x86_64-gpu`) can and cannot
emit, using the solver's epistemic scoring as the worked example. There is no GPU
runtime here and no parallel solver kernel.

## What MEASURED, and why it is NOT a solver kernel

`benchmarks/solver/gpu_epistemic_scoring.sio` (two `kernel fn` bodies mirroring
the `smt.sio` UCB score and Beta polarity) **checks clean and emits non-trivial
PTX** (298 lines, 2 `.visible .entry`, real `sqrt.approx.f32` / `div.approx.f32` /
`add.f32` / `mul.f32`).

But this is a scalar formula evaluated in a loop, **not** a scorer:

- All inputs (`act_mean`, `act_var`, `beta`, `phase_*`) are scalar and
  loop-invariant; `while i<n` recomputes the **same** value `n` times.
- The result is accumulated into a local `acc` that is **discarded** — no output
  parameter, no global store of the f64 result. The `st.global` lines in the PTX
  are loop/spill machinery, not score outputs.
- No per-variable data, no thread parallelism, no observable output.

So "the formulas emit valid PTX" is literally true and uninteresting. It is the
*ceiling* of the checked subset, not evidence of a GPU solver.

## The boundary (MEASURED, 2026-06-27)

A real per-variable GPU scorer needs a thread index and array I/O. Both are
**absent from the checked public surface**:

| Probe | Command | Result |
|---|---|---|
| thread-index intrinsic | `kernel fn … { let i = gpu.thread_id().x; (*out)[i] = (*mean)[i] + 1.0 }` | `check` fails: **`Undefined variable: gpu`** |
| block-indexed tiled kernel | `build examples/tile_matmul.sio --backend gpu` (uses `gpu.block_id()`) | **rc=1, no PTX** |
| richer source kernel | `build examples/kernel_source_level.sio --backend gpu` | **rc=1, no PTX** |
| loop + scalar f64 (no index/array) | `build benchmarks/solver/gpu_epistemic_scoring.sio …` | PTX OK (the subset ceiling) |

This matches `examples/gpu.sio`'s own note: "the `gpu.*` intrinsic namespace … is
not yet part of the checked public surface." The docs conflict
(`docs/compiler/GPU_KERNELS.md:58` lists `gpu.thread_id.*`); the surface, not the
doc, is authoritative here.

**Conclusion:** the native GPU path for the solver is blocked at the language
surface — without `gpu.thread_id`/`gpu.block_id` and array-param kernels in the
checked compiler, no SAT/SMT kernel (parallel BCP, batched per-variable scoring,
watch-list eval) can be expressed or emitted. That is the real frontier gap.

## Second finding: f64 → approx-f32 narrowing

The kernel parameters are `f64`, but every GPU arithmetic op emitted is **f32**:
`sqrt.approx.f32`, `div.approx.f32`, `add.f32`, `mul.f32`. For a GUM/epistemic
heuristic whose premise is *precise* uncertainty propagation, the GPU path
silently computing approximate single precision is a correctness divergence, not a
cosmetic caveat — the GPU path is not an f64 drop-in for the CPU scorer. Likely a
compiler-backend gap. Tracked separately as an issue.

## Status / next steps

- **MEASURED:** the checked GPU subset = loop + scalar f64 only; thread-index and
  array-param kernels do not emit; f64 narrows to `approx.f32`.
- **The frontier gap (next work):** land `gpu.thread_id`/`gpu.block_id` + array
  params in the checked public kernel surface; only then is a per-variable GPU
  scorer expressible. Then revisit array-batched epistemic scoring and (separately)
  GPU runtime on L4.
- This is **not** Level 3, and **not** a GPU solver — it is a boundary receipt that
  redirects the "native GPU" framing to the actual surface blocker.
