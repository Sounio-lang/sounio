# MoE GGUF loader + fused gate/up SiLU (Plan G)

**Date:** 2026-10-29
**Branch:** `cursor/moe-f32-gguf-silu`
**Status:** Draft for review
**Depends on:** Plan F (multi-layer MoE + `llama_load_moe` + Q4/Q5 dispatch) - PR #3303
**Approach:** **B** - super-kernel fused gate+up+SiLU (not device-resident two-mmid + elementwise)

## Goal

Ship SOTA+++ MoE hot path and weight ingest:

1. **Fused SiLU path:** one CUDA super-kernel computes, per `(token, slot, inter_row)`, Q4_K (and Q5_K twin if cheap) dots against gate and up expert packs, writes `act = silu(gate) * up` on device.
2. **Wire into `moe_ffn_q4k` / `moe_ffn_q5k`:** replace the two `mmid` gate/up launches + host `moe_silu_mul` with the super-kernel; keep `mmid` down + host `moe_weighted_sum`.
3. **GGUF hybrid loader:** `llama_load_gguf` maps Mixtral-shaped Q4_K MoE tensors into the same pack/stride contract as Plan F; **fixture GGUF required green**; real GGUF path optional with honest skip.
4. **Acceptance:** lab NMSE + microbench vs two-mmid+host baseline; demos with `ntok=1` and `ntok=4` (prefill); receipts in `docs/design/moe-f32-mmid.md`.

## Out of scope

- llama.cpp runtime link or headers
- GPU weighted-combine
- Fusing **down** into the super-kernel
- Full GGUF quant zoo (only Q4_K MoE packs required; other quants = documented skip)
- Auto-merge to `main`

## Architecture

### Super-kernel (approach B)

Name sketch: `moe_gate_up_silu_q4k` (vec n=1 twin if needed for decode).

For each output index in `act[n_tokens * slots * inter]`:

1. Resolve expert id from `ids[token, slot]`.
2. Dot Q4_K row of **gate** pack (k=hidden) against `x[token, :]`.
3. Dot Q4_K row of **up** pack against the same `x`.
4. Store `silu(g) * u` with the same SiLU definition as `moe_silu_mul` (`g / (1 + exp(-g))`).

Quiet-NaN / amax policy matches mmid labs (contract amax=1e5).

Q5_K twin: same grid/API if Plan F dispatch already selects Q5; otherwise defer twin with an explicit plan note (Q4_K alone still ships Plan G SiLU goal).

### Runtime

- `rt_load_moe_gate_up_silu(rt, ptx_path)` loads PTX symbols.
- `rt_moe_gate_up_silu_q4k(...)` launches with pack pointers, strides, ids, x, act.
- If load/launch fails: **fallback** to today path (two `rt_mul_mat_id_*` + host `moe_silu_mul`); telemetry must show fallback was used.

### `moe_ffn_*` change

```text
topk (host)
super-kernel -> act (device)     # was: mmid gate, mmid up, host silu_mul
mmid down(act) -> down rows
weighted_sum (host) -> dst
```

Host `moe_silu_mul` remains as oracle for labs and fallback.

### GGUF

- New module `stdlib/llm/gguf.sio` (or equivalent): read header, tensor directory, data blob; no HF dependency at runtime.
- `llama_load_gguf(path) -> Llama`: fill cfg MoE fields; bind Q4_K expert tensors into `moe_*_w` / bytes / strides; decode attn/embed into existing f64 CpuBuf path (same transpose rules as safetensors loaders where applicable).
- Fixture generator `scripts/llm/gen_tiny_mixtral_gguf.py` writes a tiny Mixtral-shaped GGUF aligned with `/workspace/llm-models/tiny-mixtral-moe` semantics (hidden/moe_inter divisible by 256).
- Demo default: fixture path. Optional argv for a real GGUF; missing real file = print skip + still require fixture success in CI/L4 scripts.

## Components

| Piece | Role |
|-------|------|
| `stdlib/hetero/kernels/moe_gate_up_silu_q4k_*.cu/.ptx` | Super-kernel |
| Optional Q5_K twin kernel | Same API if shipped |
| `rt_load_moe_gate_up_silu`, `rt_moe_gate_up_silu_q4k` | Runtime |
| `moe_ffn_q4k` / `moe_ffn_q5k` | Call super-kernel |
| Lab + `scripts/gpu/run_moe_gate_up_silu_*_l4_lab.sh` | NMSE + us vs baseline |
| `llm::gguf`, `llama_load_gguf` | Loader |
| `scripts/llm/gen_tiny_mixtral_gguf.py` | Fixture |
| `examples/hetero_moe_gate_up_silu_demo.sio` | Kernel/FFN smoke |
| `examples/llama_gguf_forward_demo.sio` | Load + prefill smoke |

## Error handling

| Case | Behavior |
|------|----------|
| Super-kernel PTX missing / launch fail | Fallback two-mmid + host SiLU; status marks fallback |
| Lab NMSE > 5e-4 | Fail lab; do not claim green |
| Fixture GGUF missing | Demo: missing message + exit 0; L4 script must generate fixture first |
| Real GGUF missing | Optional path skip; fixture still required |
| Non-Q4_K MoE tensors in GGUF | Skip/blocker text; no silent wrong dequant |

## Lab and acceptance gates (L4)

Shapes follow mmid family where applicable (document exact lab shapes in the plan). Oracle for act: CPU dequant+GEMM gate, dequant+GEMM up, host SiLU*mul.

- **NMSE(act) <= 5e-4** vs oracle; amax=1e5.
- **Microbench:** report us/run for super-kernel path vs baseline (two mmid + host SiLU) in Plan G receipts; day-1 us ceiling aspirational if NMSE green (record measured numbers; set ceilings once stable).
- **Demo SiLU/FFN:** DONE, status 0, finite outputs.
- **GGUF fixture:** `llama_forward_rt` with `ntok=1` and `ntok=4`; finite logits; `last_moe_layers_ok > 0` when packs + GPU mmid/super-kernel ready.
- **Docs:** Plan G section in `docs/design/moe-f32-mmid.md` + Notion card; changelog fragment on implementation PR.

## Success criteria

1. Super-kernel Q4_K lab green (NMSE gate) with published microbench table.
2. Default `moe_ffn_q4k` uses super-kernel; fallback works when PTX absent.
3. Fixture GGUF load + forward prefill green on L4.
4. Real GGUF optional path documented; honest skip if absent.
5. No llama.cpp dependency.
6. Human-gated commit/PR only.

## Implementation order (for the plan)

1. Kernel + lab Q4_K super-kernel
2. Runtime + `moe_ffn_q4k` wire + hetero demo
3. Q5_K twin or explicit defer note
4. GGUF reader + fixture generator
5. `llama_load_gguf` + llama demo (`ntok` 1 and 4)
6. Docs + Notion + changelog

## Roadmap note

Plan F closed multi-layer + safetensor/sidecar loader + prefill without GGUF/GPU SiLU. Plan G is fused SiLU (B) + GGUF hybrid. Later: GPU combine, broader GGUF quants, optional true end-to-end fuse including down.
