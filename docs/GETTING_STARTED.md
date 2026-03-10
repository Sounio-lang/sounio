<!-- docs:meta
topic_id: repo.docs.getting-started
authority: repo_only
audience: users
last_validated: 2026-03-10
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.getting-started
-->

# Getting Started with Sounio

For this repository snapshot, the public path starts from the checked compiler
artifacts already committed under `artifacts/omega/souc-bin/`. Do not begin
with `cargo build` at the repo root; there is no top-level compiler Cargo
manifest there anymore.

## 1. Default JIT workflow

```bash
export SOUC_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-jit"
"$SOUC_BIN" info
"$SOUC_BIN" check examples/hello.sio
"$SOUC_BIN" run examples/hello.sio
```

What this default artifact proves today:

- version `1.0.0-beta.4`
- Cranelift JIT enabled
- LLVM and GPU codegen disabled in this artifact
- SMT, LSP, ontology, distributed, and package-manager features disabled in this artifact

Use this profile for the ordinary docs workflow, examples, and `check`-first
validation.

## 2. GPU workflow

```bash
export SOUC_GPU_BIN="$(pwd)/artifacts/omega/souc-bin/souc-linux-x86_64-gpu"
"$SOUC_GPU_BIN" info
"$SOUC_GPU_BIN" check examples/gpu.sio
"$SOUC_GPU_BIN" build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

What this GPU artifact proves today:

- GPU codegen enabled
- Cranelift JIT disabled
- the public emission path is `build --backend gpu`
- older top-level `gpu-emit` and `gpu.*` intrinsic-heavy examples are not part of the checked public CLI contract

Use this profile only when the claim is specifically about public GPU syntax or
PTX emission.

## 3. The shortest useful command set

```bash
# Type-check without running
"$SOUC_BIN" check examples/hello.sio

# Run with the default artifact
"$SOUC_BIN" run examples/hello.sio

# Open the REPL on the default artifact
"$SOUC_BIN" repl

# Format source files
"$SOUC_BIN" fmt examples/hello.sio
```

## 4. Build-from-source note

If you are working on internal or historical build paths, treat them as
component-local workflows. The root of this repository does not expose a single
public `cargo build --features ...` contract for the compiler anymore.

When you rebuild any compiler profile yourself:

- confirm the resulting capability set with `souc info`
- document the exact binary you rebuilt
- do not describe source-tree presence alone as public feature availability

## 5. Next reads

- [docs/guide/getting-started.md](docs/guide/getting-started.md) for the current repo-facing walkthrough
- [docs/compiler/GPU_KERNELS.md](docs/compiler/GPU_KERNELS.md) for the checked GPU surface
- [docs/features/GPU_RUNTIME.md](docs/features/GPU_RUNTIME.md) for the public GPU runtime contract
- [examples/showcase/README.md](../examples/showcase/README.md) for example descriptions
