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

Hetero Q5_K/Q8_0 mmid + moe_ffn: Plan E (below). `llama_moe_ffn_block` stays Q4_K.

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

`llama_moe_ffn_block` and the Plan D layer-0 forward hook remain **Q4_K only** this
cycle (no multi-quant llama API).

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
