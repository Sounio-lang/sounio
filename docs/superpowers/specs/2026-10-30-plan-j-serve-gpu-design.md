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
- Live overlap with two curl clients plus full-compiler forge run: pending a GPU lane.
