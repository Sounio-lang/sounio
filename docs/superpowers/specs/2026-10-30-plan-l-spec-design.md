# Plan L (SOTA): speculative greedy decode — design

## Why

Plans I-K decode one token per forward per slot. SOTA servers cut
that cost with speculative decoding: a cheap draft proposes K tokens,
one target forward verifies them, accepted tokens cost ~0 extra
forwards. L adds the verify/rollback machinery greedy-only, proven
bit-exact vs plain greedy, with mechanical drafts (no compatible
draft weights on disk).

Out of scope: sampling inside spec (greedy only), tree drafts, a
trained draft model, server pump integration (no-pump demo only).

## L-spec — verify + rollback

`llama_forward_rows(_rt)` in `stdlib/llm/llama.sio`: same loop as the
forward impl, returns stacked per-row logits for N rows. Mechanical
copy; only the final block differs (stack instead of last-row).

`llama_generate_spec_from_ids` in `stdlib/llm/attn_gpu.sio`: seed the
KV with the prompt, then loop: take up to K draft tokens, append
them, run one verify forward over prompt+draft rows, compare each
draft token against the argmax at its row, accept the matching prefix
(plus one bonus token past a full accept), truncate `kv.len` back on
first mismatch, resample from there. KV rollback is a length cut; no
cache content moves.

Drafts are mechanical: echo (repeats the last token, exercises the
reject path) and oracle (copies the true greedy suffix, accept-all).
Both must match plain greedy token-for-token; the demo asserts it.

Gap: no draft model fits the disk weights (tiny-random-Llama,
vocab 32k); a real draft needs matching vocab + head. Server
integration (spec inside the pump) is future work.

## Receipts

CPU (`souc run examples/llama_spec_demo.sio`): SPECDONE PASS.
Baseline 16 singles. Echo: 15 singles + 1 verify, 1 accepted.
Oracle: 4 singles + 3 verifies, 12 accepted. Same token strings.

## Full L receipt (L4, job 20555, ExitCode 0:0)

- STEP1 SERVJPARAMS PASS; STEP1b CHATTPL PASS + CHATBODY PASS.
- STEP2 parity=1 SERVEDONE PASS; STEP2b BATCHDONE PASS (cpu 1, gpu 1).
- STEP3 OVERLAP-OK (a 200 non-stream, b 200 SSE + DONE, sess=0/1,
  TTFB 6.9s/8.2s first-token waits).
- STEP4 CHAT-OK (chat 200 + usage, stream delta + DONE, bad role
  400, empty 400, tools 200 no lift).
- STEP5 SPECDONE PASS with GPU forward counts identical to CPU
  (tokens=16, base 16 singles; echo 15+1v accepted 1; oracle 4+3v
  accepted 12; same token sequences).
- ALLDONE. Zero regression on Plan I/J/K in the same run.
