<!-- docs:meta
topic_id: repo.docs.website-features
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.website-features
-->

# Sounio Language: Website Features Overview

This page is the source-of-truth summary for what the website should claim
right now. It is intentionally narrower than the full source tree: every item
below must be supportable by a checked artifact, a committed status artifact, or
an explicitly scoped implementation note.

## 1. Public compiler profiles

The website should distinguish two checked compiler artifacts:

- default profile: `bin/souc`
- GPU profile: `artifacts/omega/souc-bin/souc-linux-x86_64-gpu`

What they prove today:

- **Cranelift JIT NOT compiled** — `souc info` prints `[-] Cranelift JIT - rebuild
  with --features jit`. Measured 2026-08-27: no artifact enables it, no build path
  passes the feature, and the binary exports no Cranelift symbol. This bullet
  previously read "Cranelift JIT enabled" — not compiled
- the checked JIT profile reports LLVM and GPU codegen disabled
- the checked GPU profile reports GPU codegen enabled and Cranelift JIT disabled
- the checked GPU profile emits PTX through `build --backend gpu`

The website should not collapse these into one "all backends enabled" story.

## 2. GPU claims the website may make

The public GPU story is real, but constrained:

- the checked GPU artifact accepts `kernel fn`
- the checked GPU artifact accepts `perform GPU.launch(...)`
- the checked GPU artifact accepts `perform GPU.sync()`
- the public checked CLI path for PTX emission is `build --backend gpu`

The website should not claim, without further artifact evidence:

- that the default JIT artifact is GPU-enabled
- that top-level `gpu-emit` is exposed by the checked public GPU CLI
- that older `gpu.thread_id.*`, `gpu.block_id.*`, `gpu.block_dim.*`, or `gpu.alloc(...)` surfaces are already part of the checked public contract
- that every backend present in `self-hosted/gpu/` is equally public and equally attested

If a page discusses backend breadth, it must explicitly separate:

- source-tree implementation work
- Omega or internal gates
- checked public artifact behavior

## 3. Scientific and epistemic claims the website may make

The website may continue to describe Sounio as an epistemic and scientific
language, but those claims should anchor to committed artifacts instead of
marketing-only prose.

Current committed signals:

- stdlib reliability totals: `81 pass / 0 fail / 1 skip / 82 total`
- stdlib inventory: `604` `.sio` files, `111` disabled files, `44` stub module files, `92` active module entrypoints
- science pipeline lanes: `2/2` required lanes passing
- hyper execution lanes: `7/7` required lanes passing
- science runtime regressions remain tracked separately, with `4` soft local failures at the current snapshot

These numbers come from committed status artifacts and should be preferred over
generic "fully production-ready" language.

## 4. Website copy rules

Every website-facing claim should follow these rules:

- name the exact artifact or gate when the claim depends on backend availability
- use `souc info` and committed status JSON as primary evidence
- describe implementation breadth separately from public CLI exposure
- remove aspirational wording if the feature is still public-facing but not artifact-backed
- preserve the `/learn/*` docs path as the canonical website docs surface

## 5. Graphics claims the website may make

The graphics showcase may claim:

- the checked JIT artifact renders `examples/render/triangle_basic.sio`,
  `examples/render/cube_wireframe.sio`,
  `examples/render/uncertainty_field.sio`,
  `examples/render/causal_dag.sio`, and
  `examples/render/quaternion_rotation.sio` to PPM
- the website SVG previews for those five examples are generated from the exact
  checked-artifact output by `website/scripts/render-assets.mjs`
- `examples/render/triangle_ppm.sio` and `examples/render/uncertainty_ppm.sio`
  remain real render fixtures in-repo, but they are not current website preview
  cards
- terminal-native demos under `examples/graphics/demos/` remain a separate, valid public surface

The website should not claim, without further evidence:

- that the current public graphics path depends on GPU rendering
- that Sounio already ships an OpenGL, Vulkan, WebGPU, or native windowed renderer as a checked public contract
- that showcase graphics images are hand-curated artwork if they are presented as compiler output

## 6. Where to point readers

Use these pages as the public website references:

- `website/src/content/docs/en/getting-started.mdx`
- `website/src/content/docs/en/feature-status.mdx`
- `website/src/content/docs/en/gpu.mdx`
- `website/src/content/showcases/gpu.mdx`
- `website/src/content/showcases/graphics.mdx`
- `docs/implementation/GPU_COMPILER_CONTRACTS.md`

Do not use this file to resurrect older marketing copy about fully exposed GPU
intrinsics, full backend parity, or root-Cargo feature toggles unless those
claims have been revalidated first.
