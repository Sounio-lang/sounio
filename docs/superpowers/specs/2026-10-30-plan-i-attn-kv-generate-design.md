# Plan I: GPU attention + KV + generate (attn QK, KV cache, greedy/temp/top-p)

**Date:** 2026-10-30
**Branch:** `pbpk-28-gpu-parity`
**Status:** SDD implementation (approach C hybrid, per brainstorm approval)
**Depends on:** Plan H mega MoE FFN (hot path unchanged below attention)

## Goal

Close the hot path left out of Plan H: attention projections and KV cache on
device, plus a generate loop with greedy and temperature/top-p sampling.
Serving (`serve.sio`) is out; it adopts this forward later.

## Architecture (approach C, approved)

Per layer on GPU with honest CPU fallback:

1. RMSNorm (host) then mmid Q4_K/Q5_K for Q, K, V, O packs (MoE stride
   contract, experts=1, trivial ids row of expert 0).
2. RoPE on device in Q/K (`attn_rope_f32.cu`; GQA heads come from cfg).
3. Causal scores plus softmax (`attn_softmax_causal_f32.cu`), attention x V
   (`attn_av_f32.cu`); KV cache resident on device, append on decode
   (ntok=1), block write on prefill (ntok>1).
4. Residual plus Plan H MoE mega unchanged; LM head as today.
5. Generate: BPE encode, forward loop to max_new/EOS, BPE decode.

Day-1 truth: packs and PTX are absent on most hosts, so the module reports
`attn_gpu_status` 0/1/2 and the labs SKIP (not pass) the mmid smoke below 2.
Fallback is marked, never silent.

## Components and APIs (`stdlib/llm/attn_gpu.sio`)

- `AttnPacks` / `attn_load_sidecar` / `attn_bind_gguf_layer`: Q/K/V/O quads.
- `attn_row_stride` via `gguf_row_quant_bytes`; `attn_packs_ready` gate.
- `attn_proj_mmid`: one dense projection through `rt_mul_mat_id_q4k/q5k`.
- `AttnKernels` / `attn_load_kernels`: rope/softmax/av PTX probe.
- `llama_sample_greedy`, `llama_sample_temp_top_p` (heap-ordered top-p,
  LCG seed, loud reject on temp<=0 or top_p outside (0,1]).
- `llama_generate_from_ids`: prefill then decode loop, EOS stop.
- Demos: `examples/llama_attn_gpu_demo.sio`, `examples/llama_generate_demo.sio`.

## Data flow, errors, acceptance

- Prefill writes full KV for new positions; decode appends 1 row.
- PTX/load fail: CPU fallback, status marks it.
- Pack mismatch: loud fail, no partial bind. Lab NMSE over limit or NaN:
  fail. Bad temp/top-p: reject, never silent greedy.
- Acceptance: NMSE vs CPU `llama_forward` on fixture; prefill ntok>1 plus
  decode ntok=1 with KV append receipt; GPU us bench where a GPU is
  present; generate demo finite on greedy and temp/top-p; docs plus
  changelog plus Notion. Serving out.

## Validation receipts (this session)

- `madaros check stdlib/llm/attn_gpu.sio`: EXIT=0 (18 modules).
- Demo checks in this workspace segfault the local madaros build
  (EXIT=139), including pre-existing `llama_gpu_demo` / `llama_tiny_demo`:
  environmental, not from this change. Demos were re-read for borrow and
  arg mistakes instead; full `run` belongs on the L4/SLURM lane.
- L4 labs (NMSE, prefill/decode, microbench, generate smoke) are encoded in
  the two demos and run where a GPU is present.
