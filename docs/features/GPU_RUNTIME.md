<!-- docs:meta
topic_id: website.docs.gpu
authority: dual
audience: users
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.gpu
-->

# Sounio GPU Runtime

This document defines the current GPU runtime contract in the repository as it
actually exists today. The repo contains more GPU implementation work than the
default JIT artifact exposes, so this page splits the story into three layers:

1. the checked public GPU artifact
2. the attestation artifacts that prove backend/runtime status
3. the larger self-hosted GPU implementation tree

## 1. Public GPU artifact

The checked GPU profile in this repo is:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
```

Current public-profile facts from that artifact:

- version: `1.0.0-beta.4`
- GPU codegen: enabled
- JIT: disabled
- public PTX emission path: `build --backend gpu`
- top-level `gpu-emit` subcommand: not exposed by the checked artifact CLI

## 2. Public surface that is verified today

These commands are checked against the committed GPU artifact:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"

"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" check examples/kernel_vec_add.sio
"$SOUC_GPU_BIN" check examples/kernel_matmul.sio
"$SOUC_GPU_BIN" check examples/kernel_epistemic_vec_add.sio
"$SOUC_GPU_BIN" check tests/run-pass/gpu_launch_surface.sio

"$SOUC_GPU_BIN" build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
"$SOUC_GPU_BIN" build examples/kernel_matmul.sio --backend gpu -o /tmp/kernel_matmul.ptx
```

The checked public surface currently includes:

- `kernel fn`
- `with GPU`
- `perform GPU.launch(...)` with explicit `(x, y, z)` grid/block tuples
- `perform GPU.sync()`
- PTX emission through `build --backend gpu`

The checked public lane now has explicit regression coverage for both the
baseline 1D-default launch tuple shape and a non-unit multidimensional launch
surface in `tests/run-pass/gpu_launch_multidim_surface.sio`.

That does **not** mean every source-tree runtime/helper path is now a general
multidimensional promotion. The honest state today is:

- checked public surface: explicit 3-tuple launch syntax is accepted
- deterministic sim/reference lane: explicit multidimensional tuple launch is
  exercised without hardware dependency
- source-tree PTX helper generation: the convenience `n`-based helper path
  still defaults to 1D launch derivation unless an explicit descriptor path is
  used

The checked public artifact does **not** currently resolve the `gpu.*`
intrinsic namespace used by older docs and sketches. In practice, that means:

- `gpu.thread_id.*`: not public in the checked artifact
- `gpu.block_id.*`: not public in the checked artifact
- `gpu.block_dim.*`: not public in the checked artifact
- `gpu.alloc<T>(...)`: not public in the checked artifact
- `gpu.alloc::<T>(...)`: also not public in the checked artifact

Those fenced surfaces now have dedicated negative fixtures in
`tests/gpu/fixtures/` so each unsupported public builtin is independently
attested instead of being grouped into a single umbrella rejection.

Wave 8 adds a dedicated negative regression for non-`x` axis spellings such as
`gpu.thread_id.y`, `gpu.block_id.z`, and `gpu.block_dim.y`, keeping the
priority builtin families explicitly fenced at the checked public boundary.

Wave 9 re-checks those same priority builtin families and still finds no
checked-artifact evidence for promotion.

Wave 6 re-checked the same fixtures against the selfhost compile-proof lane and
found that the current selfhost front-end also still rejects these names at the
source surface. Internal lowering modules may model axis-sensitive builtin
semantics for future work, but that is not a public or default-selfhost support
claim today.

Wave 7 tightens the source-tree launch descriptor contract further: explicit
grid/block descriptors now fail closed when `block_dim_x * block_dim_y *
block_dim_z` exceeds the repo-local `1024` threads-per-block ceiling, and that
constraint is mirrored in the deterministic sim/reference lane.

Wave 8 tightens the host-side marshaling path as well: launch helpers now fail
closed before deriving byte counts when `n < 0`, and that guard is mirrored in
the deterministic sim/reference lane.

Wave 9 tightens that same path one step further: helper-based launches now also
fail closed when `n == 0`, keeping nonpositive element counts out of the
runtime/marshaling path and aligning the helper lane more closely with the
descriptor lane.

Those surfaces still matter for implementation work, but they should not be
presented as the default public happy path until the checked artifact accepts
them.

## 3. Runtime evidence and support tier

The GPU release lane is backed by these committed artifacts:

- `artifacts/omega/gpu_codegen_parity.v1.json`
- `artifacts/omega/gpu_binary_attestation.v1.json`
- `artifacts/omega/gpu_runtime_attest_gate.v1.json`
- `artifacts/omega/gpu_public_contract.v1.json`

Current attested compute evidence:

- CUDA lane: `cuda-sm80`
- ROCm lane: `rocm-gfx942`
- runtime attestation host: NVIDIA L4

The repository therefore has a stronger GPU story than “parser-only syntax”:

- the public GPU artifact emits PTX
- committed parity artifacts track CUDA and ROCm binary materialization
- committed runtime attestation proves the current required GPU smoke set on
  real hardware

At the same time, support remains tiered:

- stable, artifact-backed compute evidence: CUDA and ROCm lanes named above
- source-tree implementation work without equal public runtime evidence yet:
  Metal, SPIR-V, WGSL/render, tensor-core tuning, and other advanced paths

## Capability taxonomy

Repo-local GPU support is now described with explicit classes:

- `gpu-surface-supported`
- `gpu-lowering-supported`
- `gpu-compile-proof`
- `gpu-sim-runtime-supported`
- `gpu-hardware-runtime-supported`
- `gpu-explicit-unsupported`

Use `docs/implementation/GPU_CAPABILITY_MODEL.md` for the maintainer-facing
mapping from examples/tests to those support classes and for the canonical gate
entrypoints that validate each lane.

## 4. Where the implementation lives

The main self-hosted GPU surface is under `self-hosted/gpu/`:

- lowering bridge: `self-hosted/gpu/hlir_to_gpu.sio`
- IR model: `self-hosted/gpu/kernel_ir.sio`
- PTX backend: `self-hosted/gpu/ptx.sio`
- SPIR-V backend: `self-hosted/gpu/spirv.sio`
- Metal backend: `self-hosted/gpu/metal.sio`
- runtime bridge code: `self-hosted/gpu/runtime/`
- advanced paths: `self-hosted/gpu/opt/`, `self-hosted/gpu/autodiff/`,
  `self-hosted/gpu/multi/`, `self-hosted/gpu/epistemic_*`

The self-hosted CLI also contains `gpu-emit` dispatch in `self-hosted/main.sio`,
but that path is not the checked public CLI contract. In this repo snapshot, the
recommended public command for GPU emission is still:

```bash
"$SOUC_GPU_BIN" build file.sio --backend gpu -o out.ptx
```

## 5. Runtime caveats that matter

- Do not describe the default JIT artifact as GPU-enabled. It is not.
- Do not describe the checked GPU artifact as exposing the full `gpu.*`
  intrinsic namespace. It does not.
- Do not describe the public contract as general automatic multidimensional
  host-lowering support. The checked surface accepts explicit 3-tuples, but the
  source-tree convenience helper path is still 1D-default.
- Do not describe `gpu-emit` as a public top-level subcommand of the checked
  GPU artifact. It is not there today.
- Do describe the attestation artifacts when making claims about CUDA/ROCm
  support, because those artifacts are the strongest proof in the repo.

## 6. Maintenance rule

If public GPU docs, examples, or website copy change, rerun:

```bash
bash scripts/omega/omega_gpu_public_contract_gate.sh
bash scripts/gpu/gpu_surface_lowering_gate.sh
```

That gate is the machine-readable check that the shipped GPU profile, examples,
and public docs are still talking about the same thing. The second gate keeps
the new GPU capability taxonomy aligned with the checked artifact surface.
