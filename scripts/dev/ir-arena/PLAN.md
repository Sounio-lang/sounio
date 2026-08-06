# Forward plan for the #1649 arena branch

Written 2026-08-06, after a session that fixed four defect waves on this branch
and found the fifth. Companion to `README.md`, which records what was measured;
this records what to do next and what to refuse.

## The one idea worth keeping

The branch's real invention is not struct-of-arrays. It is the
**latch → quarantine → refuse the artifact** contract around it: a violated
handle routes the write to a quarantine region, sets a flag, and
`module_frontend.sio:6158` refuses to emit a binary. That machinery is already
built and already wired.

Everything below either feeds that contract or marks where it deliberately does
not apply.

## Two classes, not one. They need different medicine.

**Truncation** — a bounded table, an index that can exceed it, a **wrong answer
at rc=0**. Members: `DCE_MAX_INSTRS` 8192, `CP_MAX_INSTRS` 8192, opt_cleanup's
256-wide register and label state, and — new, found by this review, in the
branch's own code — `ir_arena_put_name` silently writing `NAME_LEN = 0` and
`ir_arena_put_args` silently writing `ARG_COUNT = 0` on pool overflow. That
second pair is rc=12 resurrected with the alarm removed.

**Exhaustion** — a monotonic counter with no reclamation. Members: the runtime GC
handle table (`exit(182)`, #1658), and latently `IR_ARENA_TOP` plus the two pools,
since `ir_region_grow_slot` abandons the old extent rather than reusing it, so
consumption tracks *churn*, not program size.

The distinction is not academic. For truncation, refusing is the cure. For
exhaustion, refusing is **all the system already does**, and the refusing is the
problem.

### The rule that dissolves the paradox I hit

I tried twice to guard opt_cleanup and both attempts measured worse than the bug
(SIGSEGV one way, 95-instead-of-56 the other). The reason, stated generally:

> A pass may be skipped only if nothing downstream consumes its postcondition.
> Otherwise the only sound refusal is to poison the compile — never to skip the
> work and ship the artifact anyway.

DCE and CP are semantics-preserving when skipped: nothing depends on them having
run. opt_cleanup is a pipeline whose later peels assume earlier invariants, and
codegen requires `compact_nops` outright. So the guards I landed are right *for
those two specifically*, and the opt_cleanup guard was unsound in both shapes I
tried. Where a pass cannot be skipped, the legal move is the latch.

### Making it structural under Sounio's constraints

Array sizes must be literals, so `[T; SOME_CONST]` is unavailable. The affordable
substitute:

- **One blessed literal per index domain**, with CI grepping that every table in
  that domain uses it. Crude; it is the only version of a shared constant this
  language affords.
- **Bound the domain once, at the allocator.** Lowering mints registers and
  labels and already records `reg_count` / `label_count`. One check at
  end-of-lowering, latching on overflow, makes every downstream table sound by
  construction — per-site guards then protect memory, not correctness.
- **Every `*_capacity()` gets a boundary witness** at cap−1 / cap / cap+1,
  asserting *correct output or a diagnostic, never silence*. CI fails if a
  capacity function has no witness. The name-pool and arg-pool cases now latch
  rather than corrupt, so their witnesses should show a diagnostic — and until
  one exists, that latch is an untested branch. Cheapest form: a scratch copy of
  `ir.sio` with the capacity accessor reduced, the way
  `SOUNIO_IR_ARENA_VACUITY=1` already patches out the generation guard.

## Scope correction: the widening is 10 tables, not 90

I reported ~90 register-indexed `[_; 256]` arrays in opt_cleanup. Verified: 90 in
the file, but only **10 on the `ocp_mfi_*` spine**, which is the only path `-O`
executes (`module_frontend.sio:6149`/`:6217`). The other 80 are on the by-value
audit/self-test spine, which production never runs.

So: **gate the by-value spine** (a probe declining an oversized input is fine),
and **widen the ten**. Widen them as **global scalar lanes**, not locals —
16K-wide locals would put multi-megabyte frames on a seed-built stack, and global
scalar arrays are the shape this branch has already proven safe under the seed.
Reset by **epoch counter**, not by clearing loops: `ocp_mfi_dse` currently clears
its table at every label, which at 65,536 entries is a performance cliff and a
standing invitation to forget a clear.

One of the ten (`targeted`, opt_cleanup.sio:9208) is **label**-indexed, not
register-indexed. A 7057-instruction function very likely has more than 256
labels, so `ocp_mfi_dead_label` may be deleting a label some jump still targets —
a second candidate for the `-O` print deletion, alongside `ocp_mfi_dce_once` not
walking `call_args` as uses (its own comment, line 9292). **Which one actually
kills the prints is not established.** Widen both domains, re-measure, and bisect
the peels if it survives.

Do **not** renumber registers: 7088 registers over 7057 instructions is already
dense, so renumbering compresses nothing and adds a pass that can be wrong.

## Verification, in the order it should be built

Self-compile-green is a statement about the *seed*, not about the branch
compiler — this branch stayed green throughout the period when its compiler could
not compile a call with arguments.

1. **Differential output corpus.** Each program compiled by seed, by branch, and
   by branch `-O`; run all three; diff stdout and exit code. This alone would
   have caught rc=12 on the first two-function program and the `-O` print
   deletion. It must carry a **vacuity guard**: when `-O` is requested, assert the
   cleanup receipt shows nonzero pass activity, so a run whose passes never
   executed fails rather than passes. My first parity measurement was exactly
   that failure and I nearly shipped it as evidence.
2. **Activate the seals.** `ir_region_seal` has zero callers, so the backstop
   designed for the aliasing class does not exist. Seal each region after the
   last mutable pass — `module_frontend.sio:6155` already *claims* codegen sees
   an immutable module; sealing makes the claim checked.

   This is the only honest answer to *"is the alias class closed?"*. It is not
   knowable by scanning: Sweep 1 matched only syntactically-local writes and
   missed two `&!`-one-frame-down sites; a third shape — a copy stored into a
   struct field and mutated later — would evade both scans. Static scans find
   sites; only the seal tells you that you are done.
3. **A linear IR validator.** `verify.sio` checks equivalence, not invariants. An
   O(n) walk after lowering and after cleanup: every call's `arg_count` matches
   pool-resident args, every reg < `reg_count`, every jump target resolves, every
   name-requiring opcode has `name_len > 0`. This catches truncation *products*
   regardless of which table was too small — it turns "which of ten tables"
   from an investigation into an assertion.
4. **Boundary witnesses** per `*_capacity()`, as above.

**Where I disagree with the consulted design.** It placed the corpus runner last,
as the merge gate, with the widening before it. That is backwards: the widening
touches the file that has produced every transform accident on this branch, and
without the differential runner there is no way to know the widening is right.
Build the runner first and use it to validate the widening. This session produced
three cases where the obvious fix measured worse than the bug; that is the prior
to design against.

## What must be true before this merges

1. **Rewrite the 97 MB blob out of history, then rebase onto a freshly fetched
   `origin/main`.** Do the rewrite first — the branch is unmerged, so it is free
   now and only gets more entangled. `opt_cleanup.sio` and `module_frontend.sio`
   are churn-prone; every day of delay grows the conflict.
2. **Build and differentially verify the two frame-down clone fixes**
   (`const_prop.sio:1671`, `dce.sio:880` — patched, build in flight).
3. ~~**Latch the two silent pool overflows**~~ — **DONE** (`f11599deda`). Both now
   latch with distinct kinds, and `put_name` keeps the binding when the slot
   already holds that name, moving pool consumption from O(stores) to
   O(instructions). Zero violations on the compiler's own 9.5 MB self-compile and
   on all seven regression programs — and the latch is **proved to fire**:
   `tests/native-v2/ir_arena_pool_witness.sio` consumes both pools for real (no
   capacity reduced) and asserts the exact arithmetic boundary, 32768 for names
   (4194304/128) and 4096 for args (262144/64), with the right kind and a cleared
   payload. Non-vacuous: stripping just the two latch calls makes it report
   `NAME_POOL_FIRED_AT -1` and exit 12. Wired into `ir_instr_arena_gate.sh`,
   which now carries 4 witnesses and 3 vacuity checks.
4. **Seal after cleanup** — the merge-time proof that the alias sweeps closed
   anything.
5. **Cover `ir_arena_swap_slots`.** It is the only primitive here written from
   scratch and nothing in the evidence executes it; its two call sites need
   specific IR shapes. The corpus must force them.
6. **The `-O` register/label widening**, or — if it slips — the lowering-side cap
   check that latches and refuses the artifact. What must not ship is the current
   state: an in-tree `run-pass` test that miscompiles at rc=0 under `-O`.
   "Documented in known_failures" is not the same as "not shipped."

## Deliberately refused

- **Widening the 80-array audit spine.** Gate it instead; `-O` never runs it.
- **Renumbering registers.** Dense already; buys nothing, risks something.
- **Any GC collector before precise stack maps exist.** A collector with
  imprecise roots converts a loud exhaustion (`exit(182)`) into a silent
  use-after-free — it would demote the failure from the good class to the bad
  one. The staged order for #1658 is: diagnose the death (one stderr write before
  the exit at `codegen_x86_linux.sio:7101`), then widen the handle-avoidance
  class (#919's unboxed small-value path already proves allocations can bypass
  handles; every avoided handle is reclamation never needed), then precise stack
  maps, and only then a collector.
- **Wiring `ir_arena_mark`/`ir_arena_release` now.** The true ownership boundary
  for the IR arena is the whole compile — codegen reads regions until the end —
  so per-function release is wrong. The real customer is the per-object-module
  loop in `module_loader.sio:2929`, and only once a workload approaches capacity.
  A 60-function, 6304-line program compiles clean today.
- **Generic sparse-map machinery.** The epoch-tagged global lane *is* the sparse
  map Sounio can afford.

## The invariant this is all for

Reintroducing the truncation class should require writing a bounded table that
bypasses both the blessed-literal grep and the boundary-witness pairing — two
mechanical, visible failures — rather than silently indexing past a cap at rc=0.
