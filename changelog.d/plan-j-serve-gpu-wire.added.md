llm serve rewritten onto the GPU forward (Plan J, approach B): one select pump drives up to four session slots with one KvCache each, so overlapping connections interleave token by token without mixing sessions.
Completions gains temperature, top_p, seed, max_tokens, and stream for SSE; bad params are 400, never silent greedy. Health and models bytes are unchanged, and the non-stream body keeps the v1 format.
Every finished request logs tokens, finish reason, time-to-first-token, and the GPU path flag; a disconnect frees its slot and KV. llm_serve takes the runtime and a use-GPU flag with CPU fallback.
Coverage: serve_j_params run-pass test plus llm_serve_j_demo parity demo.
