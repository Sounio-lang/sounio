# MoE F32 mul_mat_id (Q4_K)

## Goal

Native Sounio hetero path for MoE expert matmul with F32 activations when
amax exceeds fp16 range (contract amax=1e5). No llama.cpp runtime dependency.

## Kernel

Source: `stdlib/hetero/kernels/mmid_q4k_dp4a_f32.cu`

Entry: `mmid_q4k_dp4a_f32` (extern C)

Algorithm (port of the measured llama.cpp dp4a path):

1. Per expert block (grid.y = expert), count matching (token, slot) ids.
2. Quantize matched F32 rows to int8 + per-32 scale in shared memory.
3. For each weight row in the tile, accumulate with `__dp4a` over Q4_K blocks,
   applying F32 scales (not half) so large activations stay accurate.
4. Write dst[token][slot][row]. Invalid ids get a quiet NaN on expert 0.

Build: `scripts/gpu/build_mmid_q4k_dp4a_f32.sh` -> PTX sm_89 + optional host lab.

## Lab gate (L4)

Shapes: m=4096, k=2048, 128 experts, 4 used, amax=1e5.

- NMSE <= 5e-4 vs CPU dequant+GEMM
- n=16 < 2500 us/run
- n=64 < 5500 us/run

Host binary: `mmid_q4k_dp4a_f32_lab` (define `MMID_LAB_HOST`).


## Measured (L4, 2026-10-28)

Host: gpuorangefs-r770-proxmox, NVIDIA L4, nvcc 12.6 (PTX .version 8.5 for driver JIT), arch sm_89.

| n | NMSE | us/run | gate |
|---|------|--------|------|
| 1 | 4.58064387e-13 | 114.43 (vec) | NMSE<=5e-4, us<=120 |
| 16 | 1.53926385e-05 | 2324.22 | NMSE<=5e-4, us<2500 |
| 64 | 1.34314301e-05 | 4638.05 | NMSE<=5e-4, us<5500 |

The one-cycle vec edit replaced byte-at-a-time Q4_K nibble reads with aligned uint32_t __ldg reads. The n=1 baseline was 308.24 us/run; the measured result is 114.43 us/run (2.69x faster).

Source artifact: `stdlib/hetero/kernels/mmid_q4k_dp4a_f32_lab_out.txt`

## Runtime (Fase B)

- `ce_launch_2d_shmem` for 2D grid + dynamic shared
- `rt_load_mmid` / `rt_mul_mat_id_q4k` on `Runtime.mmid_func`
- Demo: `examples/hetero_mmid_demo.sio`

## GraphIR MulMatId

- `op_mul_mat_id()` = 4 in `types.sio`; `device_throughput` uses `tput_matmul_f32`
- `graph_add_mmid_node` / `graph_mmid_flops` in `graph.sio`
- `rt_exec_mmid_node`: validates `node.op == op_mul_mat_id` then calls `rt_mul_mat_id_q4k`
- `rt_exec_chain` refuses non-`op_matmul` nodes with status `-22` (different buffer contract)

## n=1 tile (vec)

- Entry `mmid_q4k_vec_f32`: grid (tiles, slots), shared = k F32 activations.
- `rt_mul_mat_id_q4k` auto-picks vec when `ntokens == 1`, else dp4a.
- Lab gate: NMSE <= 5e-4 and us <= 120. Measured 114.43 us/run on L4 after packed nibble loads.

## MoE-FFN layer

- `stdlib/hetero/moe.sio`: `moe_topk`, `moe_silu_mul`, `moe_weighted_sum`, `moe_ffn_q4k`
- Demo: `examples/hetero_moe_layer_demo.sio` (router -> gate/up/down Q4_K -> combine)
- Vec n=1: uint32_t `__ldg` nibble unpack; measured 114.43 us/run on L4

Hetero Q5_K/Q8_0 mmid + moe_ffn: Plan E (below). Plan F adds Q4_K/Q5_K dispatch in `llama_moe_ffn_block` and multi-layer forward (supersedes Q4_K-only note in Plan E).

## Plan C receipts (2026-10-28)

Plan: `docs/superpowers/plans/2026-10-28-moe-f32-vec-llm.md`
Spec: `docs/superpowers/specs/2026-10-28-moe-f32-vec-llm-design.md`
Branch: `cursor/moe-f32-mmid`

1. Vec n=1 gate HIT: 114.43 us/run on L4 (NMSE 4.58e-13; gate us<=120). PTX .version 8.5 / sm_89.
2. LLM MoE FFN: `llama_moe_ffn_block` + `LlamaMoeFfnCfg` in `stdlib/llm/llama.sio` are the llama-path MoE API (router top-k + `moe_ffn_q4k`); runnable demo `examples/llama_moe_ffn_demo.sio` (L4 ok ms_gate/up/down). Plan C does NOT hook the block into `ll_forward_impl`; the dense SwiGLU path is unchanged. Wiring MoE inside `ll_forward_impl` is a deferred follow-up.
3. Docs: this file (measured table + receipts).

## Plan D: forward hook (2026-10-28)

Plan: `docs/superpowers/plans/2026-10-28-moe-f32-forward-hook.md`
Spec: `docs/superpowers/specs/2026-10-28-moe-f32-forward-hook-design.md`
Branch: `cursor/moe-f32-mmid`

Layer-0 MoE FFN is wired inside `ll_forward_impl` in `stdlib/llm/llama.sio`.
Plan C exposed `llama_moe_ffn_block` as a standalone API; Plan D calls it from
the llama forward path when the gate passes.

### Gate: `ll_moe_ready`

MoE runs only when all of the following hold:

- `use_gpu` is true and `Runtime.exec.ok`
- `ntok == 1` (vec mmid path; batch prefill stays dense SwiGLU)
- `cfg.n_experts > 0`, `cfg.n_expert_used > 0`, slots <= 16 and <= n_experts
- `rt.mmid_vec_func != 0` (PTX vec kernel loaded)
- MoE Q4_K packs and `moe_router` are populated (non-null, positive byte sizes)

If any check fails, the layer uses the existing dense gate/up/SwiGLU/down path.
That is also the fallback when `llama_moe_ffn_block` returns `!r.ok` (status in
`last_moe_status`).

### Layer-0 branch

After post-attention RMSNorm on layer index 0:

1. Router logits: `ll_proj` from `h2` through `m.moe_router`.
2. F64 -> F32 bridge for hidden state and router logits.
3. `LlamaMoeFfnCfg` from `cfg` plus pack strides/bytes on `Llama`.
4. `llama_moe_ffn_block(rt, ...)` -> F32 expert combine.
5. On success: F32 -> F64, residual add into `x`, set `last_moe_used=true`,
   `last_moe_status=0`.

Layers with `li != 0` always use dense SwiGLU. `cfg.n_experts == 0` (defaults)
keeps the full model dense.

Telemetry resets at each forward entry (`last_moe_used=false`,
`last_moe_status=0`). Public entry points `llama_forward` / `llama_forward_rt`
take `&!Llama` so telemetry can update.

### Demo and L4 receipt

Example: `examples/llama_moe_forward_demo.sio`

Synthetic 1-layer Llama (`hidden=256`, `moe_inter=512`, 8 experts, 2 slots),
Q4_K expert packs, single-token forward via `llama_forward_rt`.

Measured on NVIDIA L4 (gpuorangefs-r770-proxmox, same host family as mmid lab):

| check | value |
|-------|-------|
| last_moe_used | 1 |
| last_moe_status | 0 |
| finite_logits | 1 |
| process RC | 0 |

Stdout ends with `LLAMA MOE FORWARD DONE`. Local no-GPU path prints fallback and
exits RC=0 (dense may run if weights exist).

Regression: `examples/llama_moe_ffn_demo.sio` unchanged (block-level demo).


## Plan E: Q5_K + Q8_0 (2026-10-28)

Plan: `docs/superpowers/plans/2026-10-28-moe-f32-q5-q8.md`
Spec: `docs/superpowers/specs/2026-10-28-moe-f32-q5-q8-design.md`
Branch: `cursor/moe-f32-q5-q8`

Twin hetero stacks (dp4a batch + vec n=1) for Q5_K and Q8_0 expert matmul with F32
activations (amax=1e5). Runtime loaders, `moe_ffn_q5k` / `moe_ffn_q8_0`, and L4 demos
mirror the measured Q4_K path. No llama.cpp link.

`llama_moe_ffn_block` and the Plan D layer-0 forward hook were **Q4_K only** in the
Plan E cycle. Plan F supersedes that with `moe_quant` dispatch (Q4_K or Q5_K) and
MoE on every ready layer (not layer 0 only).

Roadmap after Plan E (spec non-goals): **4 -> 2 -> 3** means Q4_K Plans A-D ship
first; Plan E adds hetero Q5_K/Q8_0; then cycle 2 (multi-layer MoE) and cycle 3
(weight loader) as separate specs.

### Q5_K measured (L4, 2026-10-28)

Host: gpuorangefs-r770-proxmox, NVIDIA L4, nvcc 12.6 (PTX `.version 8.5`, sm_89).
Lab shapes: m=4096, k=2048, 128 experts, 4 slots, amax=1e5. Oracle: CPU dequant+GEMM.

| n | path | NMSE | us/run | gate |
|---|------|------|--------|------|
| 1 | vec | 7.13651471e-13 | 139.32 | NMSE<=5e-4; us<=160 (ceiling; aspirational 120) |
| 16 | dp4a | 1.41207309e-05 | 4161.27 | NMSE<=5e-4; us<=5000 |
| 64 | dp4a | 1.4168864e-05 | 8560.91 | NMSE<=5e-4; us<=10000 |

`LAB_EXIT:0`. Source: `stdlib/hetero/kernels/mmid_q5k_dp4a_f32_lab_out.txt`
Kernel: `stdlib/hetero/kernels/mmid_q5k_dp4a_f32.cu` / `.ptx`
Build/run: `scripts/gpu/build_mmid_q5k_dp4a_f32.sh`, `scripts/gpu/run_mmid_q5k_l4_lab.sh`
Runtime: `rt_load_mmid_q5k`, `rt_mul_mat_id_q5k` in `stdlib/hetero/runtime.sio`
MoE: `moe_ffn_q5k` in `stdlib/hetero/moe.sio`
Demo: `examples/hetero_moe_q5k_demo.sio` (L4 smoke: MOE Q5_K DONE, status 0, finite)

### Q8_0 measured (L4, 2026-10-28)

Same host and lab shapes as Q5_K.

| n | path | NMSE | us/run | gate |
|---|------|------|--------|------|
| 1 | vec | 6.69601827e-13 | 241.94 | NMSE<=5e-4; us<=280 (ceiling; aspirational 120) |
| 16 | dp4a | 1.41185482e-05 | 5528.33 | NMSE<=5e-4; us<=6500 |
| 64 | dp4a | 1.41679768e-05 | 11966.53 | NMSE<=5e-4; us<=14000 |

`LAB_EXIT:0`. Source: `stdlib/hetero/kernels/mmid_q8_0_dp4a_f32_lab_out.txt`
Kernel: `stdlib/hetero/kernels/mmid_q8_0_dp4a_f32.cu` / `.ptx`
Build/run: `scripts/gpu/build_mmid_q8_0_dp4a_f32.sh`, `scripts/gpu/run_mmid_q8_0_l4_lab.sh`
Runtime: `rt_load_mmid_q8_0`, `rt_mul_mat_id_q8_0`
MoE: `moe_ffn_q8_0`
Demo: `examples/hetero_moe_q8_0_demo.sio` (L4 smoke: MOE Q8_0 DONE, status 0, finite)

Q8_0 vec reads more weight bytes per row than Q5_K; documented us ceilings sit above
the Q4_K table targets while NMSE gates stay green.

## Plan F: Mixtral multi-layer + prefill + loader (2026-10-28)

Plan: `docs/superpowers/plans/2026-10-28-moe-f32-mixtral-forward.md`
Spec: `docs/superpowers/specs/2026-10-28-moe-f32-mixtral-forward-design.md`
Branch: `cursor/moe-f32-mixtral-forward`

Multi-layer MoE inside `ll_forward_impl`, batch prefill (`ntok>1`) via dp4a mmid,
Mixtral-shaped safetensor load plus Q4_K sidecar packs, and llama-path quant dispatch.
No llama.cpp runtime link.

### What shipped

- Per-layer MoE fields on `Llama` (routers and gate/up/down packs for up to 64 layers).
- `ll_moe_ready_layer(m, rt, use_gpu, ntok, li)` with no `li==0` gate and no `ntok==1` gate.
- Forward attempts MoE on every layer that passes ready; dense SwiGLU fallback per layer on miss or block failure.
- Telemetry: `last_moe_layers_ok` counts successful MoE layers; `last_moe_used = last_moe_layers_ok > 0`.
- `LlamaMoeFfnCfg.moe_quant` (0=Q4_K, 1=Q5_K) dispatches `moe_ffn_q4k` vs `moe_ffn_q5k` in `llama_moe_ffn_block`.
- `llama_cfg` scans `num_local_experts`, `num_experts_per_tok`, `moe_intermediate_size` from config when present.
- `llama_load_moe(dir)` loads attn/embed/norms and MoE routers plus optional sidecar expert packs.
- Demos: `examples/llama_moe_forward_demo.sio` (regression), `examples/llama_moe_multilayer_demo.sio` (synthetic 2-layer, ntok 1 and 4), `examples/llama_mixtral_forward_demo.sio` (dir-based smoke).

### Loader contract

- Router per layer: safetensor `model.layers.{i}.block_sparse_moe.gate.weight` into `moe_router[i]` (same transpose convention as other linear weights).
- Expert Q4_K sidecars (raw concat, strides from cfg):

```text
dir/moe_q4k/layer{i}_gate.bin
dir/moe_q4k/layer{i}_up.bin
dir/moe_q4k/layer{i}_down.bin
```

- If sidecars or MoE cfg are missing, attn still loads and that layer uses dense FFN fallback.
- Default demo model dir: `/workspace/llm-models/tiny-mixtral-moe` (CLI arg overrides).

### L4 synthetic multi-layer (Task 4)

Host: gpuorangefs-r770-proxmox, NVIDIA L4 (same family as mmid lab).
Demo: `examples/llama_moe_multilayer_demo.sio` (`layers=2`, `hidden=256`, `moe_inter=512`, 8 experts, 2 slots).

| run | last_moe_used | last_moe_layers_ok | last_moe_status | finite_logits | RC |
|-----|---------------|--------------------|-----------------|---------------|-----|
| ntok=1 | 1 | 2 | 0 | 1 | 0 |
| ntok=4 | 1 | 2 | 0 | 1 | 0 |

Stdout ends with `LLAMA MOE MULTILAYER DONE`. Log on workspace: `/tmp/moe_multilayer_task4_l4.log`.

### Tiny Mixtral fixture + real Mixtral

- Fixture generator: `scripts/llm/gen_tiny_mixtral_moe_fixture.py` writes `/workspace/llm-models/tiny-mixtral-moe` (config MoE keys, attn safetensors, `moe_q4k/layer{i}_*.bin`).
- Demo default dir: that fixture. L4 smoke expects `last_moe_layers_ok > 0` when packs and GPU mmid are ready.
- Full HuggingFace Mixtral weights remain optional; missing dir still prints `MIXTRAL DIR MISSING` and exits 0.


## Plan G: GGUF + fused gate/up SiLU (2026-10-29)

Plan: `docs/superpowers/plans/2026-10-29-moe-f32-gguf-silu.md`
Spec: `docs/superpowers/specs/2026-10-29-moe-f32-gguf-silu-design.md`
Branch: `cursor/moe-f32-gguf-silu`

Approach B (design spec): one CUDA kernel per `(token, slot, inter row)` dots gate and up
Q4_K expert weights against `x`, applies `silu(gate)*up` on device, and writes F32 `act`.
Down projection stays on the existing `mmid` path plus host `moe_weighted_sum`. No llama.cpp
runtime link.

### Super-kernel + runtime + `moe_ffn_q4k`

- Kernel: `stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32.cu` entry `moe_gate_up_silu_q4k_f32`
- Build/run lab: `scripts/gpu/build_moe_gate_up_silu_q4k_f32.sh`,
  `scripts/gpu/run_moe_gate_up_silu_q4k_l4_lab.sh` (`srun --gres=gpu:1`, gpuorangefs-r770-proxmox)
- Runtime: `rt_load_moe_gate_up_silu`, `rt_moe_gate_up_silu_q4k` (`Runtime.gus_q4k_func`) in
  `stdlib/hetero/runtime.sio`
- Wire: `moe_ffn_q4k` in `stdlib/hetero/moe.sio` tries fused path first; on success
  `ms_gate=gus.ms`, `ms_up=0.0`; else two `rt_mul_mat_id_q4k` + host `moe_silu_mul`
- Hetero demo: `examples/hetero_moe_gate_up_silu_demo.sio` (args: gemm-ptx, mmid-ptx, gus-ptx)

### Q4_K super-kernel measured (L4, 2026-10-29)

Host: gpuorangefs-r770-proxmox, NVIDIA L4, nvcc 12.6 (PTX sm_89). Lab shapes match mmid MoE
contract: m=4096, k=2048, 128 experts, 4 slots, amax=1e5. Oracle: CPU dequant+dot gate/up,
host SiLU*mul. Baseline: two mmid launches (vec n=1 / dp4a n>1) + host silu_mul.

Source artifact: `stdlib/hetero/kernels/moe_gate_up_silu_q4k_f32_lab_out.txt`

| n | NMSE | us/run | baseline_us | gate |
|---|------|--------|-------------|------|
| 1 | 3.34545851e-13 | 218.70 | 366.66 | NMSE<=5e-4 |
| 4 | 4.06218005e-13 | 1281.45 | 1681.16 | NMSE<=5e-4 |

`LAB_EXIT:0`. All NMSE <= 5e-4.

L4 hetero FFN smoke (`examples/hetero_moe_gate_up_silu_demo.sio`):

```text
moe gus: kernels ready
moe gus: gate_ms=0.062592 up_ms=0.000000 down_ms=0.037312
MOE GUS DONE
DEMO_EXIT:0
```

`up_ms=0` confirms super-kernel path (not dual-mmid fallback).

### GGUF reader + fixture + `llama_load_gguf`

- Reader: `stdlib/llm/gguf.sio` (GGUF v3; F32, F16, Q4_K nbytes; `gguf_open`, `gguf_find`)
- Probe: `examples/gguf_probe_demo.sio` (missing path -> `ok=false`)
- Fixture generator:

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --out /workspace/llm-models/tiny-mixtral-moe.gguf
```

Fixture shapes: hidden=256, moe_inter=512, layers=2, experts=8, slots=2, vocab=64.
Q4_K MoE tensors: `blk.N.ffn_gate_exps.weight`, `blk.N.ffn_up_exps.weight`,
`blk.N.ffn_down_exps.weight` (llama.cpp MoE naming). ~5.0 MiB on disk.

- Loader: `llama_load_gguf` in `stdlib/llm/llama.sio` fills Plan F MoE pack/strides (same
  contract as `llama_load_moe` sidecars)
- Load smoke: `examples/llama_gguf_load_smoke.sio` -> `loaded=1 n_experts=8`, `moe_pack_tensors=6`

### Prefill demo (GGUF path, ntok 1 and 4)

Example: `examples/llama_gguf_forward_demo.sio`

Default fixture: `/workspace/llm-models/tiny-mixtral-moe.gguf`. Missing file:
`GGUF FIXTURE MISSING`, `LLAMA GGUF FORWARD DONE`, exit 0.

Loads PTX: `gemm_naive_f32`, `mmid_q4k_dp4a_f32`, `moe_gate_up_silu_q4k_f32` when GPU ready.
Optional 5th argv real GGUF: `GGUF REAL SKIP` if path missing or load fails (load-only smoke;
no full forward on real weights in this cycle).

Measured on L4 (payload tar with ELF + fixture + 3 PTX; log `/tmp/llama_gguf_fwd_l4.log`):

```text
ntok=1 gpu_ready=1 last_moe_layers_ok=2 finite_logits=1
ntok=4 gpu_ready=1 last_moe_layers_ok=2 finite_logits=1
GGUF REAL SKIP
LLAMA GGUF FORWARD DONE
DEMO_EXIT:0
```

Verify locally:

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --out /workspace/llm-models/tiny-mixtral-moe.gguf
bin/souc check examples/llama_gguf_forward_demo.sio
bin/souc build examples/llama_gguf_forward_demo.sio -o /tmp/llama_gguf_forward_demo.elf
```

### Q5_K super-kernel (deferred)

Q5_K super-kernel deferred; `moe_ffn_q5k` keeps two-mmid + host SiLU.

Rationale: Q5_K block dequant differs from the Q4_K path in `moe_gate_up_silu_q4k_f32.cu`;
a twin needs a separate kernel, L4 lab gate, runtime symbol, and `moe_ffn_q5k` wire, not a
mechanical copy. Plan G SiLU fusion ships on Q4_K; Q5_K MoE remains on Plan E mmid stack
(`rt_mul_mat_id_q5k`, `examples/hetero_moe_q5k_demo.sio`).

Changelog fragment: `changelog.d/moe-f32-gguf-silu.added.md`.

## Plan H: mega-combine (2026-10-29)

Plan: `docs/superpowers/plans/2026-10-29-moe-f32-mega-combine.md`
Spec: `docs/superpowers/specs/2026-10-29-moe-f32-mega-combine-design.md`
Branch: `cursor/moe-f32-gguf-silu` (shared with Plan G; SDD Tasks 1-8; human-gated commit)

### Approach B

One CUDA block per token fuses gate/up Q*_K dots against `x`, SiLU `g/(1.f+expf(-g))`,
down projection, and router-weighted combine into F32 `dst`. When mega PTX is loaded,
`moe_ffn_q4k` / `moe_ffn_q5k` / `moe_ffn_q8_0` early-return with `ms_up=0` and
`ms_down=0`. Fallback order: Plan G GUS + mmid down (Q4) or Plan E dual-mmid + host
SiLU (Q5/Q8 when mega missing).

Kernels: `moe_ffn_mega_q4k_f32.cu`, `moe_ffn_mega_q5k_f32.cu`, `moe_ffn_mega_q8_0_f32.cu`
Runtime: `rt_load_moe_ffn_mega` (+ per-quant loaders), `rt_moe_ffn_mega_q4k` /
`rt_moe_ffn_mega_q5k` / `rt_moe_ffn_mega_q8_0` in `stdlib/hetero/runtime.sio`.

Host: gpuorangefs-r770-proxmox, NVIDIA L4, nvcc 12.6 (PTX sm_89). Lab contract (Tasks 1/4/5):
`hidden=inter=256`, `experts=4`, `slots=2`, `ntokens in {1,4}`, `amax=1e5`. Oracle: CPU
dequant + full MoE FFN + weighted combine. Gate: NMSE <= 5e-4, `LAB_EXIT:0`.

### Q4_K mega lab + microbench vs pipeline

Pipeline baseline (timed in lab host): `moe_gate_up_silu_q4k_f32` + `mmid_q4k_*` on `act`
+ host weighted sum.

Build/run: `scripts/gpu/build_moe_ffn_mega_q4k_f32.sh`,
`scripts/gpu/run_moe_ffn_mega_q4k_l4_lab.sh`

Source artifact: `stdlib/hetero/kernels/moe_ffn_mega_q4k_f32_lab_out.txt`

| n | NMSE | mega_us | pipeline_us |
|---|------|---------|-------------|
| 1 | 3.09729184e-14 | 27.83 | 24.25 |
| 4 | 5.67271065e-14 | 27.88 | 30.63 |

At this 256x256 lab shape, pipeline is slightly faster at n=1; mega wins accuracy of
full fused combine on device and drops host combine. Production shapes differ from mmid
L4 (4096x2048); compare mega vs pipeline only within the same lab binary.

### Q5_K mega lab + microbench vs pipeline

Build/run: `scripts/gpu/build_moe_ffn_mega_q5k_f32.sh`,
`scripts/gpu/run_moe_ffn_mega_q5k_l4_lab.sh`

Source artifact: `stdlib/hetero/kernels/moe_ffn_mega_q5k_f32_lab_out.txt`

| n | NMSE | mega_us | pipeline_us |
|---|------|---------|-------------|
| 1 | 2.91671503e-13 | 36.31 | 27.17 |
| 4 | 2.88811923e-13 | 36.39 | 34.74 |

`LAB_EXIT:0`. Plan G deferred Q5 GUS; Plan H mega is the Q5 SiLU+down+combine path.

### Q8_0 mega path (PATH=mega)

Task 5 chose **Approach B mega** only. Mixed path (mmid Q8 + GPU `moe_weighted_sum_f32`)
was not needed; lab artifact tags `PATH=mega`.

Build/run: `scripts/gpu/build_moe_ffn_mega_q8_0_f32.sh`,
`scripts/gpu/run_moe_ffn_mega_q8_0_l4_lab.sh`

Source artifact: `stdlib/hetero/kernels/moe_ffn_mega_q8_0_f32_lab_out.txt`

| n | NMSE | mega_us |
|---|------|---------|
| 1 | 1.3181987e-13 | 57.35 |
| 4 | 1.24388746e-13 | 57.58 |

`LAB_EXIT:0`. Lab host does not print pipeline_us for Q8 (mega-only receipt).

### Hetero demo receipts (L4)

- `examples/hetero_moe_ffn_mega_demo.sio` (Q4 mega PTX argv)
- `examples/hetero_moe_q5k_demo.sio` / `examples/hetero_moe_q8_0_demo.sio` (optional mega PTX)

Measured Q4 mega demo on L4 (`/tmp/moe_mega_demo_l4.log`):

```text
moe mega: kernels ready
moe mega: gate_ms=0.109216 up_ms=0.000000 down_ms=0.000000
MOE MEGA DONE
DEMO_EXIT:0
```

`up_ms=0` and `down_ms=0` confirm mega-kernel path (not GUS+down or dual-mmid).

### GGUF fixtures + llama prefill (Tasks 6-7)

Fixture generator (per-quant tiny Mixtral MoE GGUF):

```bash
python3 scripts/llm/gen_tiny_mixtral_gguf.py --quant q4k --out /workspace/llm-models/tiny-mixtral-moe-q4k.gguf
python3 scripts/llm/gen_tiny_mixtral_gguf.py --quant q5k --out /workspace/llm-models/tiny-mixtral-moe-q5k.gguf
python3 scripts/llm/gen_tiny_mixtral_gguf.py --quant q8_0 --out /workspace/llm-models/tiny-mixtral-moe-q8_0.gguf
```

`llama_load_gguf` sets `moe_quant` from layer-0 expert dtype (0=Q4_K, 1=Q5_K, 2=Q8_0).
All MoE layers must share one quant; mixed gate/up/down dtypes abort bind.

Example: `examples/llama_gguf_forward_demo.sio` loads mmid/gus/mega PTX by `moe_quant`,
prints mega vs mmid fallback, optional real GGUF argv (skip when missing).

L4 prefill matrix (gpuorangefs-r770-proxmox, staged ELF+PTX+fixture tar):

| quant | ntok=1 | ntok=4 |
|-------|--------|--------|
| q4k | DONE RC=0, mega, layers_ok=2, finite=1 | DONE RC=0, mega, layers_ok=2, finite=1 |
| q5k | DONE RC=0, mega, layers_ok=2, finite=1 | DONE RC=0, mega, layers_ok=2, finite=1 |
| q8_0 | DONE RC=0, mega, layers_ok=2, finite=1 | DONE RC=0, mega, layers_ok=2, finite=1 |

Load smoke: `examples/llama_gguf_load_smoke.sio` (`bin/souc check` OK for q5k/q8 fixtures).

Changelog fragment: `changelog.d/moe-f32-mega-combine.added.md`.
