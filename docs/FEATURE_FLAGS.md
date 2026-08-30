<!-- docs:meta
topic_id: repo.docs.feature-flags
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.feature-flags
-->

# Sounio Feature Flags Reference

This file documents the real capability surface users can rely on in this
repository. The public compiler contract is artifact-based, not "run one root
Cargo build and inherit every historical feature flag".

## 1. Public profiles you can verify today

The checked compiler artifacts live under `artifacts/omega/souc-bin/`.

| Profile | Artifact | What `souc info` proves |
|---------|----------|-------------------------|
| GPU profile | `souc-linux-x86_64-gpu` | GPU codegen enabled; LLVM and Cranelift JIT **not compiled**; PTX emission via `build --backend gpu` |

> **There is no JIT profile, and there is no Cranelift backend — measured 2026-08-27.**
> This table used to open with a "Default JIT profile" row naming
> `souc-linux-x86_64-jit`. That artifact does not exist, is tracked nowhere, and no
> build script passes `--features jit`. The column header is `What souc info proves`,
> and what `souc info` actually prints is:
>
> ```
> Enabled Backends:
>   [-] LLVM - rebuild with --features llvm
>   [-] Cranelift JIT - rebuild with --features jit
>   [+] GPU codegen - PTX/SPIR-V generation
> ```
>
> The seven `cranelift` strings in the binary are the messages that say it is absent
> (`Cranelift backend not compiled. Add --features jit.`); it exports no Cranelift
> symbol. The shipped engine is native-v2 (Madaros) per `docs/RELEASE_POLICY.md`.

Recommended verification:

```bash
./bin/souc info
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu info
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

## 2. What "feature flags" mean now

For the main public compiler workflow, names such as `jit`, `llvm`, `gpu`,
`smt`, `lsp`, `ontology`, `distributed`, and `pkg` are best understood as
rebuild-only capability families reported by `souc info`.

That means:

- they are real compiler capability groups
- they are not exposed through a single root-level `Cargo.toml` in this checkout
- they must be confirmed on the exact binary you are documenting

## 3. Source-build guidance

If you are rebuilding internal components or historical Rust subtrees:

- use the local manifest that actually exists in that subtree
- treat its features as component-local, not as the public Sounio compiler contract
- verify the rebuilt compiler with `souc info` before documenting its behavior

This repository currently has Rust manifests for subcomponents such as:

- `bootstrap/poseidon/rust/Cargo.toml`
- `tests/jit/Cargo.toml`

Those are not a substitute for the public compiler artifact contract.

## 4. GPU-specific rule

For public GPU documentation, use the checked GPU profile and the public CLI
path:

```bash
./artifacts/omega/souc-bin/souc-linux-x86_64-gpu build examples/gpu.sio --backend gpu -o /tmp/sounio-gpu.ptx
```

Do not describe top-level `gpu-emit` as a public checked command, and do not
describe older `gpu.*` intrinsic-heavy examples as if they already passed in the
checked public artifact.

## 5. Documentation rule of thumb

- cite the exact artifact or binary you tested
- use `souc info` as the first proof point
- treat source-tree presence as implementation evidence, not as proof that the checked public binary exposes the same feature
