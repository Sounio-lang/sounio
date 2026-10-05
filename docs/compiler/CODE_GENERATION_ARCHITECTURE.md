<!-- docs:meta
topic_id: website.docs.compiler.codegen
authority: dual
audience: contributors
last_validated: 2026-03-07
validated_by: A4
source_of_truth: docs/governance/topic-registry.v1.json#website.docs.compiler.codegen
-->

# Sounio Code Generation Architecture

Code generation is where the distinction between source-tree capability and
checked-artifact capability matters most. The repository contains several
backend implementations, and this checkout now ships separate checked JIT and
GPU artifacts instead of only one public backend story.

## Current backend map

The active self-hosted backend tree is split by target:

- `self-hosted/native/`: native lowering, ABI handling, frame layout, relocations, ELF/object generation, and backend test support
- `self-hosted/wasm/`: lowering, encoding, module construction, and WASM driver code
- `self-hosted/gpu/`: PTX, SPIR-V, Metal, tensor-oriented work, and GPU lowering paths
- `self-hosted/llvm/`: LLVM-specific lowering and backend support
- `self-hosted/emit/`: shared text or emission helpers used by backend flows
- `self-hosted/linker/`: linker-oriented support code

Useful native backend landmarks:

- `self-hosted/native/lower_ir.sio`
- `self-hosted/native/codegen.sio`
- `self-hosted/native/frame.sio`
- `self-hosted/native/reloc.sio`
- `self-hosted/native/elf.sio`
- `self-hosted/native/suite.sio`

Useful GPU landmarks:

- `self-hosted/gpu/hlir_to_gpu.sio`
- `self-hosted/gpu/ptx.sio`
- `self-hosted/gpu/spirv.sio`
- `self-hosted/gpu/metal.sio`
- `self-hosted/gpu/kernel_ir.sio`

## Checked-artifact reality

For the default docs-facing JIT artifact:

```bash
export SOUC_BIN="$(pwd)/bin/souc"
"$SOUC_BIN" info
```

The current `souc info` output shows:

- **Cranelift JIT NOT compiled** — the line is `[-] Cranelift JIT - rebuild with
  --features jit`. This list previously read "Cranelift JIT enabled" — not compiled; measured
  2026-08-27, no artifact enables it and none is built with `--features jit`
- LLVM not compiled, same shape
- GPU codegen enabled on the `-gpu` artifact

For the separate checked GPU artifact:

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" build examples/kernel_vec_add.sio --backend gpu -o /tmp/kernel_vec_add.ptx
```

That profile reports GPU codegen enabled, JIT disabled, and public PTX
emission through `build --backend gpu`.

That means contributor docs should make three different kinds of claims:

- implementation claim: "the repo contains native, WASM, GPU, and LLVM backend work"
- default-artifact claim: "the default published binary currently exposes the Cranelift JIT path"
- GPU-artifact claim: "the separate checked GPU binary exposes public GPU codegen"

## Validation strategy

Use different evidence for different statements:

- backend architecture: inspect the `self-hosted/` backend directories
- user-facing capability: confirm with `souc info`
- language success or refusal: run current `tests/run-pass/` or `tests/compile-fail/` fixtures with the checked artifact
- science/runtime claims: read committed gate artifacts under `artifacts/stdlib/` and `artifacts/omega/`

## Documentation rules

- Do not document LLVM or GPU codegen as part of the default user path unless you have validated a build that enables them.
- Do document the separate checked GPU profile when you are describing public GPU codegen.
- Do document source-tree backend breadth when explaining architecture to contributors.
- Do not infer backend maturity from directory presence alone.
- When a codegen claim affects the website, keep the wording aligned with `website/src/content/docs/en/compiler/codegen.mdx`.
