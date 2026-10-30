# Plan J: serving on the GPU forward (select pump + SSE)

**Date:** 2026-10-30
**Branch:** pbpk-28-gpu-parity
**Status:** SDD implementation (approach B rewrite, per approval)

## Goal

Wire serve.sio onto llama_forward_rt plus generate, with temp/top-p params, SSE streaming, and session isolation for overlapping connections.

## Key deviation from the approved sections

The approved design said worker pool plus GPU lock. The language has no thread primitives, so overlap is an interleaved select event loop: at most one token per active session per pump turn, with one KvCache per slot. GPU calls never overlap by construction. Session isolation and no-mixing hold exactly as approved.

## Components

- Session slots (four): fd, request bytes, ids, params, seed, logits, KvCache, text buffer, counters, timestamps.
- select pump: readable listen fd accepts into a free slot; readable session fds get one recv; GEN slots get one decode turn.
- Params: temperature, top_p, seed, max_tokens, stream. Bad values are 400.
- SSE: one data frame per token plus data DONE; non-stream body keeps the v1 byte format.
- llm_serve takes rt and use_gpu; CPU fallback when the runtime is not ready.

## Validation

- tests/run-pass/serve_j_params: defaults, clamps, rejects, SSE bytes.
- examples/llm_serve_j_demo: serving-loop parity with generate, usage counts, SSE golden bytes, TTFB via clock_gettime.
- Live overlap on NVIDIA L4 (gpuorangefs-r770, job 20457, ExitCode 0:0): STEP1 SERVJPARAMS PASS; STEP2 greedy parity=1 SERVEDONE PASS; STEP3 health True, a-status 200 (253 B non-stream JSON with usage), b-status 200 (1210 B SSE frames ending data: [DONE]), OVERLAP-OK.
- Session isolation on GPU: req=2 served on sess=0, req=3 on sess=1, each completion_tokens=8 finish=length; TTFB 6.9 s and 8.2 s with gpu=1.
- GPU lane fixes: attn_fail and ll_gguf_moe_fail print, so they declare IO (E035) with propagation to callers; SSE DONE frame is 14 bytes, test and demo expectation fixed from 15; stream sends tested nw_send_all != 0 but it returns sent bytes, so each stream send closed the slot right after the headers (200 with empty body): checks are now < 0.

## Forge receipt

- corpus-smoke run sounio-20261030T152953Z-90d38b: pass (cpu pass=3 fail=0, job 20304), snapshot of fd6ef70ddc.
- Artifacts: /orangefs/training/sounio-foundry/runs/sounio-20261030T152953Z-90d38b.
