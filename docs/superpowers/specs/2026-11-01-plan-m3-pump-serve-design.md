<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-01-plan-m3-pump-serve-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-01-plan-m3-pump-serve-design
-->

# Plan M3: pump wiring (serve continuous batching) — design

## Why

M2 proved the engine: `svb_pump` runs chunked prefill plus decode
rows over paged KV with queue admission (`BATCHCONT PASS`, CPU and
RTX4000). But `llm_serve` still generates through `sv_turn`
(single) and `sv_turn_batch` (contiguous `KvCache`, batch decode
only). Arrivals still wait for a free session slot, short sessions
hold full caches, and prefill still blocks the turn. M3 wires the
pump into the serve loop so every GEN turn is a pump turn.

Out of scope: preemption/recompute, temp/top-p inside the pump
(sampling stays in the server), true paged GPU kernel (Phase 2),
multi-GPU, HTTP keep-alive changes, auth/quotas.

## Current (what the server does today)

`llm_serve` owns 4 fixed sessions: `fd/st` (socket/state), `ids`
(prompt ids), `prm` (temp/top-p/max-tokens/stream), `seed`,
`logi` (last-row logits), `kvs` (contiguous `KvCache`), `nids`
(prompt len), `gen` (generated count), `t0/ttfb`, `text/tlen`
(non-stream accumulate), `reqid`, `chats/chtols`. `sv_begin`
parses the request, prefills the prompt into `kvs[si]` in one shot,
sets `logi` to the last row, state becomes GEN (2). Each turn:
`bn == 1` runs `sv_turn` (sample, emit, forward 1 token);
`bn > 1` runs `sv_turn_batch` (sample all, emit all, ONE
`llama_forward_batch[_rt]` over live rows, scatter logits, finish
done rows). Finish (`tok == 0` or `max_tokens`) sends SSE-done and
closes, or replies the JSON body and closes, then prints the
served-req receipt line.

## M3 — pump turn in the serve loop

Session state gains a pump sidecar, kept next to the current
arrays (no struct rewrite of the hot loop): `slots: [SvSlot; 4]`
(off/nid/cur/gen/maxgen/last), `bank: [i64; 512]` (prompt token
store), `q: SvQueue` (waiting arrivals), `pks_k/pks_v`
(256 paged tables, slot s owns layers `[s*64 .. s*64+63]`),
plus the row layout buffers (`rslot/rpos/toks/kind/rowtok`,
`meta`). Session slot `s` maps 1:1 to pump slot `s`.

Prompt ingest moves out of `sv_begin`'s one-shot prefill for pump
sessions: the prompt ids go into `bank` at a free offset, a queue
entry `(off, nids, maxgen)` is pushed, and the slot starts empty
(`off = -1`). The pump admits it on the next turn and prefills in
chunks of C alongside running decodes. TTFB changes meaning: first
token after the first chunk lands, not after the full prompt.

Sampling stays in the server (pump keeps no sampler). Per turn the
server reads each live row's logits from the pump output, samples
greedy or temp/top-p exactly as `sv_turn` does today (same
`prm/seed` arrays, same helpers), emits SSE-delta or accumulates
text exactly as today, then feeds the sampled tokens back as the
next turn's `last` inputs. Finish/close/receipt logic is unchanged,
driven by the sampled token and `gen` counts. Concretely the
`svb_pump` + `svb_sample_rows` split becomes: pump builds rows and
returns stacked logits; server samples and emits; pump advances
cursors with the sampled tokens. The internal greedy `last` in
`svb_sample_rows` goes away in favor of server-supplied tokens.

Mixed mode and rollback: `bn == 1` keeps `sv_turn` (proven path,
zero behavior change for the common case). `bn > 1` takes the pump
turn. A flag (env or arg, default pump) keeps `sv_turn_batch`
reachable until the M3 lane is green; then it is deleted with its
contiguous batch path, and the single path follows in M4.

## Verification

Run-pass first (no sockets): a scripted 3-client conversation
through the pump turn with arrivals mid-generation, prompt lengths
spanning chunk boundaries, one EOS-early client and one
max-tokens client, SSE and accumulate modes, greedy only. Expected
token streams must equal sequential single forwards, and the row
layout must show prefill/decode mixing with chunk cap respected.
Marker: `SERVEPUMP PASS`.

Live lane second: `llm_serve` on tiny-random-Llama (CPU) and
SmolLM2-135M (RTX4000 + L4 when the cluster frees) with 3
concurrent curl clients (2 stream, 1 accumulate), same prompts as
the run-pass script. Responses must be token-identical to the
sequential baseline; receipt lines must show all sessions served.
Markers: `SERVEPUMP-LIVE PASS`, `parity=1` per client. Temp>0
sampling parity (seeded) is verified separately after greedy is
green and is not required for the M3 receipt.

## Receipts

- SERVEPUMP PASS (run-pass, scripted arrivals, both stream modes).
- SERVEPUMP-LIVE PASS on CPU + RTX4000 (+ L4 when available),
  token-identical to sequential, no J/K/L/M1/M2 regression in the
  same lane (SERVJPARAMS, CHATTPL, CHATBODY, LOOKUP, SERVEDONE,
  BATCHDONE, SPECDONE, plus the M2 9/9).
- `sv_turn_batch` deleted only after the live lane is green.

## Decisions (2026-11-01, user-confirmed)

- Chunk size C = 64 for serve traffic.
- Oversize prompts: enlarge the bank (no 429, no wait-on-full).
- `bn == 1` migrates to the pump once green (pump for all GEN);
  `sv_turn` stays only as the rollback path until the lane passes.

## Open questions (deferred)

- Queue depth 8 vs burst traffic: wait or reject with 429?

## Implementation notes (2026-11-01, compiler finding: 25-param cliff)

- `sv_begin` grew to 25 params and `sv_pump_turn` takes 26. Both
  SIGSEGV `madaros-f2ifix2 build` under the default 8MB stack, even
  though `check` passes. Bisect: pure `[i64; 16384]` locals build
  (512-16384), `&[i64; 16384]` params build, 640KB frames build,
  23- and 24-param fns build, 25-param fn SIGSEGVs. gdb shows the
  crash on a stack address with no backtrace (stack exhaustion).
  With `ulimit -s unlimited` (lane standard) the 25-param repro
  builds; 16MB also suffices. Root cause class: compiler stack
  hunger scaling with param count, silent SIGSEGV instead of a
  diagnostic. Filed as backlog claim (proposed SIO-561); proper
  fix (stack discipline / proper error) is open. M3 builds and
  tests run under `ulimit -s unlimited`, which is the lane
  standard, not a workaround: the code shape (25/26 params) is
  natural and matches the existing 23-param `sv_turn_batch`.

## Receipts (2026-11-01)

- SERVEPUMP PASS, CPU: split turn (fwd + host greedy sample +
  adv) lockstep vs whole `svb_pump` — same meta, same rowtok,
  same token streams, slot reuse, prefill/decode mixing.
- M2 CPU suite green with the refactored engine: PAGEDKV,
  PAGEDFWD, PAGEDBATCH, PAGEDRUNS, BATCHCONT, BUILTINF2I, LOOKUP.
- RTX4000 (5860) lane, self-contained sbatch via login-slinky,
  exit-code protocol (orangefs -o channel is deaf: quota):
  serve_pump PASS with `rt.ready` required, batch_cont PASS with
  `rt.ready` required, SERVEPUMP-LIVE PASS (real `llm_serve` +
  3 concurrent /dev/tcp clients: 2 non-stream identical, 1 SSE
  with data:/[DONE], 3 served receipts, 4 gpu=1 lines).
- Cluster notes: node has no curl (bash /dev/tcp client used);
  `-o` files stay empty (writes fail silently); big heredocs do
  not survive spool (plain base64, ~1MB scripts OK); gzip missing
  on node (do not rely on it).
- Reconciled with parallel session commit 1242b437d7 (vendored
  fixture): `batch_cont` bank 512->16384; both M2/M3 tests now
  load `tests/fixtures/tiny-random-Llama` (byte-identical).
