# Plan M2 (SOTA): continuous batching + paged KV — design

## Why

Plans I–M1 serve at most 4 fixed sessions, each holding a full
contiguous `KvCache` (`kv_new` allocates `max_seq * kvdim` per layer
per slot upfront, whether used or not). Two SOTA gaps remain: an
arrival waits for a free slot even when the GPU is idle between
turns, and short sequences pay the memory of the longest one. M2
fixes both with paged KV plus a pump that admits work every turn.

Out of scope: preemption/recompute, host swap, a true paged GPU
kernel (Phase 2), tree drafts, multi-GPU.

## M2a — paged KV (decode)

`PagedKv` replaces `KvCache` per sequence: a per-layer block pool
(`CpuBuf` blocks of B = 16 tokens), a per-sequence block table, and
`len`. Token at position `p` lives in block `table[p / B]` at offset
`p % B`. Append writes to the tail block, allocating a fresh block
from the pool when the tail is full. Reads go through the table.
Attention math is untouched (same dot/softmax/accumulate over
positions `0..len`); only the address computation changes, so output
is bit-exact vs the contiguous cache.

Rollback (spec verify, M1) stays a length cut: blocks past `len` go
stale and are reused on regrow, freed only when the sequence
closes. When the pool is exhausted, new arrivals wait in the queue;
running sequences are never evicted (no recompute path in M2).

GPU Phase 1: gather each row's blocks into contiguous scratch, then
run the existing kernels unchanged (memory win now, same compute).
A true paged kernel is Phase 2 and is not required for the M2
receipts.

## M2b — continuous batching

The fixed `[KvCache; 4]` becomes a free-list plus a waiting queue.
Every pump turn admits queued arrivals (prefill chunk of C tokens)
alongside one decode token per running sequence. The batch forward
takes per-slot runs `(slot, start_row, nrows)` instead of one token
per slot; prefill rows attend to the cache plus earlier rows of the
same chunk (causal within chunk), decode rows attend as today.
Finish conditions are unchanged (eos / max_tokens); closing a slot
returns its blocks to the pool immediately.

M1 lookup drafts keep working per sequence: draft/verify rows are
just more rows of that sequence in the same turn.

## Receipts

- PAGEDKV PASS: paged vs contiguous token-for-token, CPU and GPU.
- BATCHCONT PASS: arrival mid-decode joins next turn; close frees
  blocks; prefill chunk + decode mix in one forward.
- ALLDONE, no regression: SERVJPARAMS, CHATTPL, CHATBODY, LOOKUP,
  SERVEDONE parity=1, BATCHDONE cpu 1 gpu 1, OVERLAP-OK, CHAT-OK,
  SPECDONE — same lane shape as M1 (L4, ExitCode 0:0).

## Implementation notes (2026-10-31, CPU parity reached)

- `rpos` contract (decided): ABSOLUTE position. `pos0 = rpos[s]`,
  `upto = pos0 + 1`. The caller (scheduler) passes the token's absolute
  cache index; rows of one slot in one call must arrive in ascending
  order with no gaps so each append lands at `len == pos0`. The earlier
  `slen[slot] + rpos` (intra-call offset) design double-counted on the
  decode step and is removed.
- Per-layer body lives in `ll_runs_layer` (helper): same math as the
  batch forward, per-row rope/writeback preserved for chunked prefill
  positions. The monolithic first version produced wrong layer-1+
  values (deterministic, e.g. 88973 vs 205211) with no logic difference
  vs the passing batch impl; moving the identical layer body to a
  helper fixed it. Suspected context-sensitive codegen issue; if it
  recurs, bisect by shrinking/extracting the hot region again.
- Wrappers/impl take `&![PagedKv; 256]` (honest: the forward mutates
  the tables via append).
- Verified CPU (forge pod): PAGEDKV, PAGEDFWD, PAGEDBATCH, LOOKUP,
  PAGEDRUNS all PASS. GPU lane (L4) still pending.

## Implementation notes (2026-10-31, BATCHCONT CPU green)

- Engine lives in `stdlib/llm/serve.sio` (`svb_qpush/qshift`,
  `svb_admit_all/build_rows/sample_rows/pump`): free-list of 4 paged
  slots plus a waiting queue of 8; every turn admits queued arrivals
  and runs one `llama_forward_batch_runs` mixing prefill-chunk rows
  (kind 0) and decode rows (kind 1); close truncates the slot tables
  and frees the slot. Receipt test `tests/run-pass/batch_cont.sio`
  replays every row against sequential singles (bit-exact, greedy
  lockstep), with mid-decode arrival, slot reuse, and a GPU arm.
- Compiler finding (RESOLVED 2026-10-31, SIO-559): the many-param
  struct failures were the f2i frontend bug, not struct access.
  Fix: lower_name_is_float_builtin now precedes the fn-table lookup
  in expr_result_is_float_ref (self-hosted/ir/lower.sio). The
  scalar-only arrays plus caller-owned token bank workaround is
  REVERTED: svb engine and batch_cont test use SvSlot and SvQueue
  structs across many-param fns; M2 9/9 green on CPU with the fixed
  compiler. sv_turn_batch (21 params) was never executed end to
  end; it is out of M2 scope and should not be trusted without a lane.
- GPU arm of BATCHCONT runs only where `rt.ready`; CPU pod skips
  it. L4 lane still pending.
