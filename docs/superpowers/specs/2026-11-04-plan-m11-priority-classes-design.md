<!-- docs:meta
topic_id: repo.docs.superpowers.specs.2026-11-04-plan-m11-priority-classes-design
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.superpowers.specs.2026-11-04-plan-m11-priority-classes-design
-->

# M11: priority classes for preemption

## Direction

M9/M10 preempt the smallest-gen decoding slot whenever the
table is full and the queue is non-empty. Every session is
equal: a background batch job can evict an interactive decode,
and a low-priority arrival can trigger the eviction of a
high-priority decoder. M11 adds priority classes: the victim
is the lowest-priority decoder (tie-break: smallest gen, the
M9/M10 rule), and a queued arrival only triggers preemption
when it outranks the victim. No struct layout change; pump-path fn signatures widen by one
parallel array (M10 pattern): priority travels in a
session-indexed parallel array, reusing the ss/sos session
identity M10 built.

## Slice

1. Session priority: sprio[8], session-indexed, default 0,
   higher value = more important (harder to evict). Set at
   push time per session; last write wins. Live: sv_begin
   parses optional request JSON field priority (default 0);
   plain test pushes stay 0.
2. Victim choice in svb_admit_all: among eligible slots
   (off >= 0, vvalid == 0, gen > 0) pick lowest
   (sprio[sos[s]], gen[s]) lexicographically. sos[s] < 0
   (plain occupant) reads as priority 0.
3. Trigger guard: preempt only if the head entry outranks
   the victim. Head priority = sprio[q.ss[0]] for live
   entries (ss >= 0), 0 for plain/victim entries (ss < 0;
   a staged victim head keeps the M10 restore path, which
   never preempts). If no victim is outranked, return with
   the queue held (same wait path as queue-full today).
4. Threading: sprio rides beside every call site that
   already carries sos/t_sos (engine fns, pump, turn,
   live server session array). No SvQueue/SvSlot field;
one extra parallel-array param on admit/fwd/pump/turn.

## Verification

- New tests/run-pass/serve_prio.sio: 3 sessions over
  4 slots with mixed priorities. High-priority decoder
  survives a low-priority burst (low-prio arrivals wait,
  no preempt bit); low-priority decoder is evicted for a
  high-priority arrival and recomputes bit-exact (dual-ref
  discipline from serve_preempt: main cache never reset).
- CPU 6/6 (existing 5 + prio). Live 5860 lane: burst with
  priority 1 arrivals against priority-0 decoders
  finishes 5/5 with preempt lines; inverted run (prio-0
  burst vs prio-1 decoders) shows zero preempt lines and
  all 5 served after drain. Receipt in plan-m11.
- Negative control: with all priorities equal, victim
  order is exactly smallest-gen (M9/M10 behavior kept).

## Out of scope

Starvation bounds / aging, priority inheritance across
restore, more than one preempt per fwd call, cross-node
spill, non-pump path behavior.

## Receipts 2026-11-08/09 — live r770/L4 (M11 priority classes)

Build: source 39519a5448 (origin/pbpk-28-gpu-parity tip, SHA-OK), m11e.elf
via committed bin/madaros-linux-x86_64 (md5 8ad309f6),
elf sha256 8bfb7d5e8f925968ea2148c25cb16ca5c3a2740370f9e4c345c3545e5c80a5b1.
Model SmolLM2-135M (orangefs), max_tokens 8 burst / 64 occupants,
greedy defaults (temp 0, seed 12345). Single-allocation srun choreography:
4 occupants fill 4 slots, 1.5 s later 4 burst arrivals; solos sequential first.

| lane | occ prio | burst prio | burst-vs-solo | occ done | svd preempt lines |
| prio_up_pd1 | 0 | 1 | 4/4 | 4/4 | 17 |
| prio_down_pd1 | 1 | 0 | 4/4 | 4/4 | 1 (same-prio fill churn; burst waited ~480 s, zero inversion preempts) |
| prio_up_pd0 | 0 | 1 | 4/4 | 4/4 | 22 |
| prio_down_pd0 | 1 | 0 | 4/4 | 4/4 | 0 (burst waited ~460-469 s for drain) |

Interpretation: trigger guard holds live — prio-1 burst evicts prio-0
decoders (17/22 preempt lines, every row recomputed bit-exact 4/4);
prio-0 burst never evicts prio-1 decoders (0 inversion lines), all 4
served after drain 4/4.

Deviations from spec: (1) lane is r770/L4, not 5860 (5860 drained,
Reason=Kill, needs host admin); (2) 4+4 choreography instead of a 5-way
burst: SV_MAX_SESS=8 caps concurrent sessions, so 4 occ + 4 burst = 8;
(3) comparison on raw text-payload bytes: the server emits raw control
chars/newlines inside JSON strings when the model emits them, which breaks
strict client-side json.loads; payload byte-compare remains exact.

Follow-up (not M11, no server change made): completions containing raw
control chars produce invalid JSON on the wire; harness worked around it
with tolerant field extraction.
