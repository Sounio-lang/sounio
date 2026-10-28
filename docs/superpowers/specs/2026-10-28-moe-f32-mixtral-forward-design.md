# MoE multi-layer + Mixtral loader + prefill (Plan F)

**Date:** 2026-10-28
**Branch:** `cursor/moe-f32-mixtral-forward`
**Status:** Draft for review
**Depends on:** Plan D (layer-0 hook) + Plan E (Q5_K/Q8_0 hetero mmid) - PRs #3292 / #3300

## Goal

Ship a native Sounio Mixtral-capable forward path:

1. MoE FFN on **every** configured layer (not only layer 0).
2. **Prefill**: MoE works for `ntok > 1` via dp4a; `ntok == 1` keeps vec.
3. **Loader**: `llama_load_moe` reads Mixtral-shaped safetensors (attn dense + MoE experts as Qx_K packs).
4. Acceptance demos: synthetic multi-layer + optional real Mixtral L4 smoke with BPE.

Approach: **B (SOTA+++)** from the Plan F brainstorm - full decode stack plus prefill; no GGUF and no GPU SiLU in this cycle.

## Out of scope

- GGUF dual loader
- GPU SiLU / weighted-combine kernels
- Q8_0 inside `llama_moe_ffn_block` (hetero already has it; llm defaults Q4_K/Q5_K from packs)
- GraphIR multi-quant
- Changing dense attention math beyond existing `ll_proj`
- Merging to `main` (human review)

## Architecture

### Per-layer MoE storage

Replace single-layer MoE fields with per-layer arrays (cap 64, same as attn):

- `moe_router[64]: CpuBuf`
- `moe_gate_w[64] / moe_up_w[64] / moe_down_w[64]: *mut u8`
- byte sizes + expert/row strides per layer (or shared strides if identical layout)
- `moe_quant: i64` tag (0=Q4_K, 1=Q5_K) for `llama_moe_ffn_block` dispatch
- Telemetry: `last_moe_used`, `last_moe_status`, `last_moe_layers_ok: i64`

Dense default: `cfg.n_experts == 0` leaves packs null and never enters MoE.

### Forward (`ll_forward_impl`)

For each layer `li`:

1. Attention + residual unchanged.
2. `h2 = rmsnorm(post)`.
3. If `ll_moe_ready_layer(m, rt, use_gpu, ntok, li)`:
   - project router for that layer
   - `llama_moe_ffn_block` with layer packs
   - on `r.ok`: residual add, bump `last_moe_layers_ok`
   - on fail: set `last_moe_status`, fall through to dense SwiGLU for **this layer only**
4. Else dense SwiGLU as today.

`ll_moe_ready_layer` requires GPU + mmid funcs for the active quant + packs for `li` + expert cfg. **Does not** require `li == 0` or `ntok == 1`.

Runtime already picks vec vs dp4a inside `rt_mul_mat_id_*` from `ntokens`.

### `llama_moe_ffn_block`

Extend to call `moe_ffn_q4k` or `moe_ffn_q5k` from `moe_quant` (and matching `rt_load_mmid` / `rt_load_mmid_q5k` already done by the demo/runtime open path). Keep one public block API.

### Loader

`llama_cfg(dir)`:

- Scan `num_local_experts` -> `n_experts`
- Scan `num_experts_per_tok` -> `n_expert_used`
- Scan `moe_intermediate_size` / fallback -> `moe_inter`
- Existing dense fields unchanged

`llama_load_moe(dir) -> Llama`:

- Load attn/norms/embed as `llama_load` today (f64).
- Skip dense `mlp.gate/up/down` when MoE cfg active (or leave empty).
- Per layer, load:
  - `model.layers.{i}.block_sparse_moe.gate.weight` into `moe_router[i]` (transpose if needed for `ll_proj`)
  - Expert packs for w1/w3/w2 (Mixtral naming) into gate/up/down Qx_K byte buffers
- Packing contract (document in design note): either
  - (preferred) directory already contains pre-packed Qx_K expert blobs with documented strides, or
  - host packer from f16/f32 expert tensors into Qx_K (if present as dense in safetensors)

If MoE tensors missing: `loaded` may still be true for dense-only; MoE layers fall back to dense.

### Demos

1. `examples/llama_moe_multilayer_demo.sio` - synthetic `layers>=2`, packs on all layers, run `ntok=1` and `ntok=4`, assert MoE used + finite + DONE + `exit(0)`.
2. `examples/llama_mixtral_forward_demo.sio` - `llama_load_moe` + BPE + `llama_forward_rt` on a configured model dir env/path; L4: decode 1 token finite; optional short prefill.

Regression: `llama_moe_forward_demo`, `hetero_moe_*`, dense demos typecheck.

## Error handling

| Case | Behavior |
|------|----------|
| `n_experts==0` | Dense only |
| Packs missing for layer | Dense FFN that layer |
| MoE block `!ok` | Dense fallback that layer; status recorded |
| Mixtral dir absent on CI | Document blocker; synthetic gates still required green |
| slots > 16 | Reject as today |

## Success criteria

1. Multi-layer synthetic MoE green for `ntok=1` and `ntok=4` on L4 (or CPU GPU-unavailable path with honest messaging + `exit(0)` only when GPU path not claimed).
2. No regression on Plan D/E demos (`souc check`).
3. `llama_load_moe` loads Mixtral-shaped cfg + per-layer routers/packs when artifacts present.
4. Real Mixtral L4 1-token decode finite **or** written blocker with exact missing artifact path.
5. Docs Plan F section + Notion card Done when acceptance met.
6. No llama.cpp runtime link.

## Roadmap note

Program order was 4 -> 2 -> 3; Plan E closed item 4. Plan F merges items 2+3 with prefill. Later: GGUF / GPU SiLU (Plan G+).
