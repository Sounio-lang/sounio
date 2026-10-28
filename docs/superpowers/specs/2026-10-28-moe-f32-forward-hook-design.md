<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-10-28-moe-f32-forward-hook-design
authority: repo_only
audience: contributors
last_validated: unvalidated
validated_by: none
source_of_truth: docs/superpowers/specs/2026-10-28-moe-f32-forward-hook-design.md
-->

# Design: MoE FFN hook in llama forward (plan C follow-up)

**Date:** 2026-10-28
**Orthography:** EN-UK
**Status:** Approved design-base (approach C)
**Depends on:** plan C (`docs/superpowers/specs/2026-10-28-moe-f32-vec-llm-design.md`) - vec gate HIT, `llama_moe_ffn_block`, demo
**Branch:** `cursor/moe-f32-mmid`
**Hardware:** NVIDIA L4 via `srun` when exercising GPU MoE path
**Implementation plan:** `docs/superpowers/plans/2026-10-28-moe-f32-forward-hook.md`
**Backlog:** Notion Tasks prefix `SIO-moe-f32-forward`

---

## Goal

Close the gap between demo-only `llama_moe_ffn_block` and the llama forward path:

- Extend config so dense remains default (`n_experts == 0`).
- In `ll_forward_impl`, for **layer 0 only**, when MoE is configured and GPU+packs are ready, run router + `llama_moe_ffn_block` instead of dense SwiGLU.
- All other layers stay dense SwiGLU.
- Demo injects synthetic Q4_K packs + tiny router weight into a `Llama` without a full Mixtral safetensors loader.

---

## Non-goals

- Full Mixtral/Qwen-MoE safetensors load for all layers.
- MoE on every layer.
- Changing attention / KV / RoPE.
- GPU SiLU/combine (host path inside `moe_ffn_q4k` stays).
- Matching llama.cpp end-to-end.
- Auto-commit / PR (human-gated).

---

## Architecture

```
ll_forward_impl (per layer li)
  attn + residual  (unchanged)
  h2 = post_attn RMSNorm
  if li == 0 and moe_ready(m, rt, use_gpu):
       router_logits = proj(h2, m.moe_router)
       convert h2 -> f32 row(s)
       llama_moe_ffn_block(...) -> dn_f32
       convert dn_f32 -> f64 CpuBuf
       ll_add_into(x, dn)
  else:
       dense SwiGLU (existing gate/up/silu/down)
```

`moe_ready` is true when:

- `m.cfg.n_experts > 0` and `m.cfg.n_expert_used > 0`
- `use_gpu` and `rt` has mmid loaded
- MoE packs present: gate/up/down pointers non-null with matching byte sizes
- `m.moe_router` weight present for `hidden -> n_experts`

On MoE path failure (`!MoeFfnResult.ok`): fall back to dense SwiGLU for that layer (do not abort the whole forward). Demo observes `last_moe_used` / `last_moe_status`.

---

## Data model changes

### `LlamaCfg` additions

- `n_experts: i64` - 0 = dense-only (default)
- `n_expert_used: i64` - slots; must be <= 16 and <= n_experts when MoE
- `moe_inter: i64` - MoE intermediate; 0 means use `cfg.inter`

`llama_cfg()` leaves these at 0 so existing dense loads are unchanged.

### `Llama` additions (layer-0 packs only this plan)

- `moe_router: CpuBuf` - empty when unused
- `moe_gate_w` / `moe_up_w` / `moe_down_w: *mut u8` plus byte sizes and Q4_K strides
- `last_moe_used: bool` and `last_moe_status: i64` updated each forward

Null / zero bytes => MoE not ready => dense. Demo fills fields; no Mixtral tensor-name loader in this plan.

Reuse `LlamaMoeFfnCfg` built from these fields inside the forward branch.

---

## Router and numeric bridge

- Dense projection `h2` (ntok x hidden) -> logits (ntok x n_experts) via existing `ll_proj` / `ck_gemm` on f64, then cast to f32 for `moe_topk`.
- Cast `h2` to f32 for `llama_moe_ffn_block`; cast MoE dst f32 back to f64 before `ll_add_into`.
- **MoE branch requires `ntok == 1`** on first landing (vec path). If `ntok != 1`, use dense even when MoE cfg is set. Lifting this is a follow-up (dp4a already supports n>1).

---

## Demo

`examples/llama_moe_forward_demo.sio`:

1. Open runtime, `rt_load_mmid`.
2. Build a tiny `Llama` with `layers=1` (or 2 with only layer 0 MoE), compact dims (prefer hidden=256, moe_inter=512, experts=8, slots=2), synthetic dense attn/embed weights plus Q4_K packs and router.
3. Call `llama_forward_rt` with `ntok=1`.
4. Assert `last_moe_used` and `last_moe_status == 0` when GPU available; finite logits; print DONE.
5. GPU unavailable: dense fallback, DONE, exit 0.

Keep `examples/llama_moe_ffn_demo.sio` green.

---

## Error handling

| Condition | Behaviour |
|-----------|-----------|
| `n_experts == 0` | Dense only |
| MoE cfg set but packs missing / GPU down / ntok!=1 | Dense FFN for layer 0 |
| `llama_moe_ffn_block` returns `!ok` | Dense fallback; set `last_moe_status` |
| `slots > 16` or `slots > n_experts` | Block rejects (-20); forward treats as !ok |

No llama.cpp linkage.

---

## Testing

| Gate | How |
|------|-----|
| Typecheck | `bin/souc check examples/llama_moe_forward_demo.sio` |
| GPU smoke | L4 `srun`: MoE used + finite logits + DONE |
| No GPU | Fallback + DONE exit 0 |
| Regression | `llama_moe_ffn_demo` still OK |
| Dense cfg | Existing dense demos unchanged when `n_experts==0` |

---

## Success definition

1. `ll_forward_impl` runs MoE FFN on layer 0 via native hetero Q4_K F32.
2. Dense default preserved for `n_experts==0`.
3. Demo proves MoE branch on L4 with finite outputs.
4. `docs/design/moe-f32-mmid.md` updated; Notion mirror when MCP auth available.

---

## Open decisions locked

- Approach C (layer-0 only, synthetic packs, ntok==1 MoE gate).
- Fallback to dense on MoE failure (no hard abort).
- Full multi-layer MoE loader is out of scope.
