# Plan K (SOTA): batched decode + chat template — design

## Why

Plan J serves one decode row per slot per pump turn (one GEMM per
projection per slot). SOTA servers batch the decode rows of all live
slots into one GEMM per weight per layer, and speak chat templates.
K1 does the batching with the existing kernels (no new PTX).
K2 adds a mini-Jinja renderer plus `v1/chat/completions`.

Out of scope: paged KV, CUDA graphs (no graph API in the PTX
runtime), prefix cache, grammar-constrained tools.

## K1 — batched decode

`llama_forward_batch(_rt)` in `stdlib/llm/llama.sio`: stack one row
per live slot; row-wise ops (rmsnorm, Q/K/V/O/gate/up/down
projections, silu, MoE router+block) run stacked as one GEMM each;
rope, KV append, and causal attention stay per slot on its own
cache. Returns stacked `[n * vocab]` logits.

`sv_turn_batch` in `stdlib/llm/serve.sio` mirrors `sv_turn`: sample
+ send/accumulate per slot, one batched forward for the live rows,
finish + served log per done slot. The pump uses it when 2+ GEN
slots share an iteration, else the old single path (zero change to
the J receipts path).

Bit-exactness: batch rows do not interact (independent caches,
row-wise math), proven by `examples/llama_batch_demo.sio` (8 steps,
2 prompts, full-logit compare + greedy tokens, CPU and GPU).

## K1 receipt (L4, job 20523, ExitCode 0:0)

- STEP1 SERVJPARAMS PASS; STEP2 greedy parity=1 SERVEDONE PASS.
- STEP2b BATCHDONE PASS (cpu parity=1, gpu parity=1).
- STEP3 OVERLAP-OK with the batch path live (a 200 non-stream,
  b 200 SSE + DONE, sess=0/1).
- Fix during lane: logits copy used `&!logis[i].data` and overwrote
  the buffer pointer (server died after one frame); field read is
  correct (c912d22640).

## K2 — chat template + tools

New `stdlib/llm/chat.sio`: mini-Jinja renderer (`{{var}}`,
`{%for x in messages%}`, `{%if ...%}`, dotted/`['k']` access on
message maps), template from `--chat_template` file or builtin
ChatML default (no model on disk ships one). `tools` param
pass-through; `<tool_call>json</tool_call>` extraction from
generated text into `tool_calls` (protocol only; no local model
guarantees tool-shaped output).

Endpoint `POST /v1/chat/completions` (non-stream JSON + SSE reuse),
400 on bad roles/shape. Tests: template golden bytes (CPU
run-pass), tool extraction unit, chat demo parity vs completions.
## Full K receipt (L4, job 20540, ExitCode 0:0)

- STEP1 SERVJPARAMS PASS; STEP1b CHATTPL PASS + CHATBODY PASS.
- STEP2 parity=1 SERVEDONE PASS; STEP2b BATCHDONE PASS (cpu 1, gpu 1).
- STEP3 OVERLAP-OK (batch path live); STEP4 CHAT-OK (chat 200 + usage, stream delta + DONE, bad role 400, empty 400, tools 200 no lift).
- Zero regression on Plan I/J in the same run.
