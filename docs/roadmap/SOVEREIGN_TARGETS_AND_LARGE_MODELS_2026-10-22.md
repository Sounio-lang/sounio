<!-- docs:meta
topic_id: repo.docs.roadmap.sovereign-targets-and-large-models-2026-10-22
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.roadmap.sovereign-targets-and-large-models-2026-10-22
-->

# Sovereign targets and the large-model roadmap

**Date:** 2026-10-22 · **Author:** Demetrios Chiuratto Agourakis

**Status: plan.** It records which emission targets the compiler can own end to
end, the rule for third-party tools, and the order in which large-model
inference should be attempted. Hardware figures are vendor specifications or the
author's lab inventory as stated on 2026-10-22, not measurements made in this
repository; where a statement needs a measurement before scheduling, it says so.

---

## 1. What "emit machine code with nothing third-party" means

Emitting instructions directly is four layers, and each target's irreducible
dependency sits in a different one:

1. **encoding** — the operation as bits;
2. **object format** — ELF, Mach-O, cubin, HSA code object, bitstream;
3. **loader / driver interface** — the kernel via syscalls, or a GPU driver via
   ioctls;
4. **firmware** — what runs inside the device. This is the legitimate vendor
   lock, the only one accepted.

Goal: **the compiler owns layers 1–3; only layer 4 belongs to the vendor.**

## 2. Target by target

| target | owning 1–3 | what remains | status in the tree |
|---|---|---|---|
| x86-64 Linux | yes: own encoder (EVEX included), own ELF writer, direct syscalls, no libc, no linker | the kernel syscall ABI | the native backend already works this way (`self-hosted/native/`; [ADR-012](../decisions/adr-012-hyper-native-v2-bridge.md) records its 186-byte single-ZMM AVX-512 Fano sequence as validated bit-exact on hardware) |
| AArch64 Linux | yes; the encoder can be **generated** from Arm's machine-readable architecture specification | — | the seed emits it (`--target aarch64-linux` in `self-hosted/compiler/lean_single.sio`) |
| AArch64 macOS | partly: the syscall ABI is not stable (Apple requires libSystem) and every arm64 binary must be signed; an ad-hoc signature is a hash structure the compiler can write without `codesign` | libSystem | — |
| RISC-V | yes: open ISA, machine-readable opcodes; on an FPGA the core itself can be owned | — | not a target yet |
| AMD GPU (RDNA) | **yes, the most open GPU target**: public ISA manuals, open kernel driver, documented KFD submission (ioctls + AQL packets); code object, load and dispatch without ROCm | GPU firmware | not a target yet |
| NVIDIA GPU | against the vendor: PTX is finished by the driver's JIT (`ptxas`), so layer 1 is not ours; owning it means emitting undocumented SASS. Mesa's NAK is an open encoding reference: it has encoders from Kepler on, with one family (`sm70`) covering Volta through consumer Blackwell; support for GB10's SM121 specifically is not established. Open kernel modules allow talking via ioctl (tinygrad does). Each generation is still its own encoding table | GSP firmware | PTX emission exists (`self-hosted/gpu/`) |
| Apple Metal | **no**: the public API accepts AIR, Apple's compiler emits the GPU ISA; Asahi's AGX reverse engineering is Linux-only | the vendor compiler | — |
| Alveo U250 | **no**: the UltraScale+ bitstream is proprietary (Project U-Ray documents part of it, far from place-and-route); Vivado is a vendor lock, XRT is open | Vivado | SAN kernels exist ([deployment paper](../papers/san_fpga_deployment_2026-08-04.md)) |

## 3. The rule: oracle yes, dependency no

**A third-party tool may enter as a test oracle, never as a build dependency.**
`objdump`, Intel XED, `nvdisasm`, `llvm-mc`, MPFR and Spike may check what the
compiler emitted; none of them may sit between a `.sio` source and its binary.
This is ADR-009's oracle principle applied to the toolchain, and it is what lets
sovereignty and independent cross-checking coexist.

## 4. Large models: bandwidth decides

In decode, each token reads every **active** weight once:

> tokens/s ≤ memory bandwidth ÷ (active parameters × bytes per parameter)

Two consequences: prefer mixture-of-experts (cost per token scales with active
parameters, memory with total), and choose hardware by bandwidth. With a ~120B
total / ~5B active MoE at 4 bits (~3 GB read per token):

| hardware (author's lab) | capacity | bandwidth | role |
|---|---|---|---|
| 2× NVIDIA GB10 | 128 GB unified each | ~273 GB/s | where a large model fits; theoretical ceiling ~10² tok/s for that MoE, less in practice |
| AMD Radeon AI PRO R9700 | 32 GB | ~640 GB/s | fast but small: a ~30B / ~3B-active MoE at 4 bits |
| CPU nodes with large RAM | up to 512 GB | DDR | capacity reserve, slow |
| 2× Alveo U250 | 64 GB DDR4 each | ~77 GB/s | too slow for decode; an epistemic and verification coprocessor, not an LLM accelerator |

**The tension:** the most sovereign target (AMD) is small; the target with
capacity (GB10) is NVIDIA Blackwell, whose SASS is undocumented. Resolve it in
time, with provenance recording which layer is not owned.

### Phases

- **A — correctness, 100% sovereign, CPU.** Qwen2.5-0.5B on a CPU node. AMX
  tile instructions are documented and can be emitted directly on Sapphire
  Rapids and later; Skylake has no AMX. **AMX ignores MXCSR, fixes
  round-to-nearest and forces DAZ/FTZ**, so it may serve only as an uncertified
  baseline; no AMX result may populate a certified enclosure
  ([interval proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md), D11).
- **B — sovereign GPU.** R9700 via KFD, without ROCm; RDNA4 matrix instructions
  (WMMA) are documented in the ISA manual. Target: a ~30B / ~3B-active MoE at
  4 bits. The RDNA rounding mode lives in a wave-global `MODE` register, to be
  checked in the RDNA4 manual before scheduling certified kernels there.
- **C — scale, on GB10.** PTX first, accepting the driver JIT, with provenance
  marking "layer 1 not sovereign"; own SASS later, Blackwell only, with NAK as
  the encoding reference. Two linked GB10s allow expert parallelism across 256 GB
  for 200B+-class models; expert streaming lets total size exceed memory.

### The kernel core is small

About ten kernels: GEMV/GEMM with int4/fp4 dequantisation; decode attention
(flash-decoding style); RMSNorm, RoPE, softmax; top-k expert routing; KV-cache
management; sampling; plus a safetensors or GGUF reader (simple formats).

### Where Sounio differs — and the naive mistake

- **Do not propagate intervals through the whole model.** The dependency
  problem makes the enclosure explode within a few layers.
- **Quantisation is a controlled lie with a known size.** `Knowledge` can carry
  per-layer weight-quantisation error; no current runtime does.
- **Certified `e` per kernel, statistical `u` across the model.**
- **The `Introspect` channel:** logits, entropy and activations as typed values
  with their own provenance
  ([research record](../research/EPISTEMIC_REASONING_AND_METACOGNITION_2026-10-22.md)).

### Acceptance

The reference (llama.cpp or the model's reference implementation) is an oracle
only. Accept when: per-token KL divergence is below a declared threshold; greedy
decode is token-identical on N prompts; the maximum logit difference is reported
with its enclosure.

## Claims Forbidden

- Do not describe RISC-V, AMD RDNA or SASS as current targets.
- Do not quote the tok/s figures as measured; they are bandwidth ceilings.
- Do not describe PTX output as sovereign at layer 1.
