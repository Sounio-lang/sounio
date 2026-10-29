# MoE mega-kernel + GPU combine + GGUF Q5/Q8 (Plan H)

**Date:** 2026-10-29
**Branch:** `cursor/moe-f32-gguf-silu` (extends Plan G tree; split only if human asks)
**Status:** Draft for review
**Depends on:** Plan G (GUS `moe_gate_up_silu_q4k` + hybrid GGUF + Q5 GUS deferred) - uncommitted (ship gated on human bora)
**Approach:** **B** - mega-kernel: fused gate+up SiLU + down + weighted combine on device (not staged A / not host combine)

## Goal

Ship SOTA+++ MoE FFN hot path end-to-end on device and close GGUF quant gaps left by Plan G:

1. **Mega-kernel B:** one CUDA path per quant family computes, per token, top-k expert FFN: dequant gate/up -> `act = silu(gate)*up` -> down GEMM -> weighted accumulate into `dst` on GPU. No host `moe_weighted_sum` / `moe_silu_mul` on the hot path.
2. **Q5_K twin:** mandatory in this plan (Plan G deferred Q5 GUS; H ships Q5 mega).
3. **Q8_0 path:** prefer same mega mould; if Q8_0 dequant/grid does not fit cleanly in time, document **mixed path**: mega (or GUS) through act + `mmid` Q8 down + GPU elementwise weighted combine. No silent host combine on the claimed hot path.
4. **Wire:** `moe_ffn_q4k` / `moe_ffn_q5k` / `moe_ffn_q8_0` prefer mega; fallback = Plan G GUS + mmid down + host combine (or two-mmid+host if GUS absent).
5. **GGUF:** bind Q5_K and Q8_0 MoE packs; multi-quant fixture; demos with fixture green; real GGUF optional with honest skip.
6. **Acceptance:** NMSE vs host oracle; prefill demos; L4 microbench mega vs Plan G pipeline (GUS+down+host); docs + changelog + Notion.

## Out of scope

- llama.cpp runtime link or headers
- Attention, KV cache, tokenizer, serving stack
- Q2_K / Q3_K / Q6_K / IQ* zoo
- CPU mega-kernel, multi-GPU, autotune / graph compiler
- Auto-merge to `main`; commit/PR only on human "bora"

## Architecture

### Mega-kernel (approach B)

Name sketch: `moe_ffn_mega_q4k` (vec twin if decode n=1 needs it), plus `moe_ffn_mega_q5k`, and `moe_ffn_mega_q8_0` **or** mixed-path helpers.

For each token (and optionally batched tokens):

1. Read `ids[token, slot]` and `weights[token, slot]` for `slots = top_k`.
2. For each slot, resolve expert; for each intermediate row: QK dots gate and up vs `x[token, :]`; write local `act = silu(g)*u` (same SiLU as `moe_silu_mul`: `g / (1 + exp(-g))`).
3. For each output dim of down: QK dot of down pack row vs `act`; accumulate `weight[slot] * down_row` into per-token accumulator.
4. Write `dst[token, :]`.

Quiet-NaN / amax policy matches mmid labs (contract amax=1e5). Tile/shared-mem choices stay conservative; labs publish occupancy/us.

Q5_K: same API surface as Q4_K mega (mandatory green).

Q8_0: same API if dequant fits; else mixed path with an explicit note in plan + receipts (still no host combine on hot path).

### Runtime

- `rt_load_moe_ffn_mega(rt, ptx_path)` loads PTX symbols for Q4/Q5/(Q8).
- `rt_moe_ffn_mega_q4k` / `_q5k` / `_q8_0` (or `_combine` helper for mixed path) launch with pack pointers, strides, ids, weights, x, dst.
- Load/launch failure -> **fallback** Plan G path (GUS + mmid down + host weighted_sum); telemetry must mark fallback.

### `moe_ffn_*` change

```text
topk (host)
mega-kernel -> dst (device)     # was: GUS -> act; mmid down; host weighted_sum
```

Host `moe_silu_mul` / `moe_weighted_sum` remain oracles for labs and fallback.

### GGUF (extends Plan G)

- Reuse `stdlib/llm/gguf.sio` + `llama_load_gguf`.
- Bind Q5_K and Q8_0 expert tensors into the same pack/stride contract as Plan F/G.
- Extend fixture generator (or sibling) for multi-quant tiny Mixtral-shaped GGUF (hidden/moe_inter divisible by 256 where QK requires).
- Demos: fixture default; optional real GGUF argv; missing real = skip; fixture must stay green in L4 scripts.

## Components

| Piece | Role |
|-------|------|
| `stdlib/hetero/kernels/moe_ffn_mega_q4k_*.cu/.ptx` | Mega Q4_K |
| `stdlib/hetero/kernels/moe_ffn_mega_q5k_*.cu/.ptx` | Mega Q5_K twin (required) |
| `moe_ffn_mega_q8_0_*.cu/.ptx` **or** GPU combine helper + mmid Q8 | Q8_0 path |
| `rt_load_moe_ffn_mega`, `rt_moe_ffn_mega_q*` | Runtime |
| `moe_ffn_q4k` / `q5k` / `q8_0` | Prefer mega |
| Labs + `scripts/gpu/run_moe_ffn_mega_*_l4_lab.sh` | NMSE + us vs Plan G pipeline |
| `gguf` + `llama_load_gguf` Q5/Q8 bind | Loader |
| Multi-quant fixture generator | Fixture |
| `examples/hetero_moe_ffn_mega_demo.sio` | Kernel/FFN smoke |
| `examples/llama_gguf_*` with Q5/Q8 fixtures | Prefill smoke |

## Error handling

| Case | Behavior |
|------|----------|
| Mega PTX missing / launch fail | Fallback GUS + mmid down + host combine; status marks fallback |
| Lab NMSE > threshold (align Q4/Q5 mmid labs; default 5e-4 unless plan sets tighter) | Fail lab; no green claim |
| Shape / quant mismatch on GGUF load | Clear error (type + dims); no partial bind |
| Fixture GGUF missing | L4 script generates first; demo may exit 0 with missing message |
| Real GGUF missing | Optional skip; fixture still required |
| Q8_0 mega not shipping | Mixed path only if documented in plan + receipts |

## Lab and acceptance gates (L4)

Oracle: CPU/host path with same SiLU and weighted_sum semantics (Plan F/G oracles).

- **NMSE(dst) <= 5e-4** (or documented lab threshold) vs oracle; amax=1e5.
- **Microbench:** mega us/run vs Plan G pipeline (GUS + down + host); record numbers; meta: mega <= pipeline (prefer clear win).
- **Prefill:** multilayer / Mixtral-tiny / GGUF Q4+Q5+Q8 demos, `ntok=1` and `ntok=4`, finite outputs, no NaN.
- **Docs:** Plan H section in `docs/design/moe-f32-mmid.md` + Notion card; changelog.d fragment on implementation PR.

## Success criteria

1. Mega Q4_K and Q5_K labs green (NMSE) with published microbench vs Plan G pipeline.
2. Q8_0 mega green **or** documented mixed path green with GPU combine (no host combine on hot path).
3. Default `moe_ffn_q4k` / `q5k` (/ `q8_0`) prefer mega; fallback works when PTX absent.
4. GGUF Q5_K + Q8_0 fixture load + forward prefill green on L4.
5. No llama.cpp dependency.
6. Human-gated commit/PR only; Plan G ship before or with H as human decides ("bora").

## Risks

| Risk | Mitigation |
|------|------------|
| Occupancy / shared mem on mega B | Conservative tiles; lab us vs GUS pipeline |
| Q8_0 dequant unlike QK | Mixed path explicit in plan if twin slips |
| Plan G still dirty on same tree | H builds on G; ship G first or keep H commits after G stable |
| Multi-quant fixture fragile | Python generator + lab checksums / fixed seeds |

## Implementation order (for the plan)

1. Mega Q4_K kernel + L4 lab (NMSE + us vs Plan G)
2. Runtime + `moe_ffn_q4k` prefer-mega + hetero demo
3. Mega Q5_K twin + lab + `moe_ffn_q5k` wire
4. Q8_0 mega or mixed GPU-combine path + lab
5. GGUF Q5/Q8 bind + multi-quant fixture
6. Llama GGUF demos (`ntok` 1 and 4) + prefill
7. Docs + Notion + changelog

## Roadmap note

Plan F: multi-layer + safetensor/sidecar + Q4/Q5 dispatch.
Plan G: fused GUS (B) + hybrid GGUF Q4; Q5 GUS deferred; host combine remains.
Plan H: mega B (GUS+down+combine on device) + Q5 twin + GGUF Q5/Q8; closes the hot-path host round-trip for MoE FFN.
