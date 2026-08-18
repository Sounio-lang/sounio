# Agent Handoff — the Sounio fleet

> **Rewritten 2026-08-16** (glm-cli1, lane `agent-handoff-refresh`). The previous
> edition described a six-lane activation from 2026-05-10 with worktrees that no
> longer exist; it had already declared itself stale. This edition describes the
> fleet as it actually is, grounded in the 2026-08-15→16 session, and carries the
> operational lessons that session paid for. It is **durable orientation** — live
> state always lives in `bin/sounio-coord` and `.claude/attention_p0.v1.json`,
> never here. When this file disagrees with those, those win and this file gets
> updated in the same change-set (Charter §8).

## 1. Cold start — the 60-second version

```bash
cd /workspace/.wt/<your-worktree>        # one worktree per agent; never work on /workspace/sounio
./sounio-whereami --quick                # orient
bin/sounio-coord status                  # active claims, conflicts, worktree heads
bin/sounio-coord inbox --agent <you> --lane <your-lane>   # unread messages — act, then ack
bash scripts/dev/attention_brief.sh      # P0 slots, STALE claims, freeze state
cat .claude/ATTENTION_CHARTER.md         # binding: 5 = 1 + 2, ≤1 active P0
```

Read order for a cold agent: this file → `CLAUDE.md` → `.claude/ATTENTION_CHARTER.md`
→ `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` (the active plan)
→ `.claude/MEMORY_LANES.md` (pick **only** your lane's memory files).

## 2. The fleet as it is (verified 2026-08-16T12:07Z)

**23 tmux windows** in session `fleet`, one per agent, mixed backends:

| Windows | Backend | Notes |
|---|---|---|
| 0 | GLM (glm-cli1, glm-cli2 at 18) | this lane's family |
| 1–3 | Claude (claude-1/2/3) | claude-1 = `fleet-orchestrator` coordination lane |
| 4–6 | Codex (codex-1/2/3) | |
| 7–9 | Cursor (cursor-1/2/3) | |
| 10–14 | Grok (grok-cli1..5) | |
| 15–16 | Kimi (kimi-cli1/2) | |
| 19–21 | MiniMax (minimax-cli1..3) | |
| 17 | `repo` | the canonical `/workspace/sounio` control checkout |
| 22 | fable-1 | Claude-family takeover lane (see §5) |

- **One worktree per agent** under `/workspace/.wt/<name>`; window 17's
  `/workspace/sounio` is the control/shepherd checkout — merge, brief, inbox.
  **Not** a compile bench and never a second agent's workspace.
- **Coordination bus**: `bin/sounio-coord` (`claim` / `scope` / `heartbeat` /
  `release` / `send` / `inbox` / `ack` / `status` / `prune`), state in
  `/workspace/.tmp/sounio-coord/2493585255/`. Claims carry a TTL — **heartbeat
  every working session** or your files silently free up. Same bus over MCP for
  Cursor/Claude (`scripts/mcp/sounio_coord_mcp.py`).
- **Ceilings that are binding, not advice**: ≤1 active P0 slot (Charter),
  ≤2 agents editing `self-hosted/` at once (CLAUDE.md §4). Heavy builds
  (`make build`, `souc main.sio`, bundle checks) go through
  `scripts/dev/souc-build-lock.sh` — the pod has been k8s-evicted under
  CPU stampede before (2026-05-29, load ~153). Cheap `souc check` is exempt.
  `make madaros-full-gate` / `make build-madaros` self-lock internally — do
  **not** wrap them again (nested `exec 9>` deadlocks).
- Legacy lane tables, the 2026-05 collision resolutions and merge orders that
  used to live here are history; `git log` and `artifacts/omega/agent_handoff.log.md`
  remain the archive. `integration/sounio-dev-ready-base` is stale — do not
  broad-merge it.

## 3. Attention Charter — P0 slots

The Charter (`5 = 1 + 2`: attention governance *is* compiler sovereignty plus
epistemic honesty) is binding on every lane. Only work that closes **1** or **2**
with a named gate ≥ E3 may hold P0. Machine truth:
`.claude/attention_p0.v1.json` (updated by the orchestrator; **do not hand-edit
without claiming it** — claude-1 holds it).

State at rewrite time (2026-08-16T11:15Z snapshot):

- **Slots A–E: done** (2026-08-04, D3 residual / native-v2 zero-event /
  trust-map / claim registry / hygiene — see the machine file for receipts).
- **Slot F — `extern "C"` FFI silent-noop under Madaros: done 2026-08-16T04:50Z**,
  owner fable-1 (see §5 for the full arc, including why the owner changed
  mid-flight).
- **Slot G — WS-C PR1 (MIR port, Route B): queued**, owner fable-1, blocked on
  the founder-ordered sequence *"amendments first, then WS-C PR1"*: grok-cli1
  amending `MIR_PORT_PLAN.md` (`amend/mir-port-plan-20260816`), grok-cli2
  amending `MLI_DESIGN.md` (`amend/mli-design-20260816`), codex-2 producing the
  payload census. Constraint C3: WS-F owns `tools/eisa` + `stdlib/eisa` sources —
  PR1 may add frontier files but must not modify main-side eisa files unclaimed.

Before any write-bearing dispatch, fill the Charter §4 contract (Lane / Owner /
Closes / Worktree / Branch / Write-Set / Required-Gates / Merge-Target /
Known-Blockers). "Explore and fix" without it is drift; halt with a Blocker-ID or
a `Next-Command` handoff is a deliverable.

## 4. Where the work is — the focus plan

`docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` (authored by
Fable 5 at founder request, orchestrated by claude-1) is the active operational
plan. Founder objectives, verbatim: *"Madaros E2E operacional, SOIR, MIR, MLI,
HLIR EISA, f128 e f256 implantados e verificados."* Seven workstreams:

| WS | Scope | Status at rewrite |
|---|---|---|
| A | Madaros E2E operational — fresh dated full gate, residual disposition, status regen | P0-F closed its headline defect; `MADAROS_STATUS.md` regen still owed (WS-A owns the gate marker; see slot F's owed followups) |
| B | SOIR — roundtrip gate + coverage census | planned |
| C | **MIR port — Route B** (add `self-hosted/enir/**` as a self-contained subtree, 14 files / 7310 LOC) | **founder-approved; PR1 = P0 slot G, queued behind amendments** |
| D | **MLI design** — greenfield Machine-Level IR between MIR and codegen | **Option C founder-approved**; design doc in the wave-1 tranche |
| E | HLIR re-verify | grok-cli5 writing the HLIR dispatches |
| F | EISA Madaros port (owns `tools/eisa` + `stdlib/eisa`) | grok-cli4 on close-prep (`WS_F_CLOSE_ACCEPTANCE_2026-08-16.md`) |
| G | f128/f256 from V0-A — largest single workstream | grok-cli3 on V0-B witnesses |

Dispatch runs in waves under the ≤2 self-hosted-writer ceiling; **wave 1 is out**.
The wave-1 planning tranche landed as **`6f2c4e2461` (PR #1751)**: the focus
plan, `MIR_PORT_PLAN.md` (the costed three-route study behind Route B),
`MLI_DESIGN.md`, the adversarial preflight review, and the payload census —
previously untracked files in a shared checkout, landed precisely so they could
not be lost (see lesson 4).

## 5. Grounding session — the P0-F arc (2026-08-13 → 08-16)

The event this file is anchored to, kept here because it is the fleet's clearest
recent demonstration of how reallocation, preservation and closure are supposed
to work:

1. **2026-08-13** — forensic dispatch
   `docs/audit/EXTERN_C_FFI_SILENT_NOOP_DISPATCH_2026-08-13.md`: `extern "C"`
   calls under default Madaros return a fabricated `0` without invoking the
   function (`getpid()`, `system()`). Root causes later located: the parser keeps
   only the first declaration of a multi-decl extern block; lowering never
   assigns the extern strategy, so calls lower into empty bodies.
2. **2026-08-15 → 08-16T00:43Z** — glm-cli1 executes Track B's leftovers and
   Track A: refreshes the stale seed from the fixed-point build (#725, with a
   freshness drift-guard), adds the engine-forced `system()` regression gate,
   dispatches the four secondary defects (one root-caused to `jmp rel8`
   saturation at string literals ≥127 bytes; one proven unreproducible in 11
   attempts and recorded as such), and builds the four-layer Track A port
   (parser wrapper rewrite, checker `ffi_*` binds, native registry emitters).
3. **~00:43Z** — glm-cli1 hits a **5-hour API usage limit** mid-task (8/12
   subtasks done). The slot's reset was ~8.5h away with WS-C PR1 and WS-D S1
   both held behind P0-F.
4. **01:12–01:13Z** — the orchestrator **preserves the WIP verbatim as
   `9498c533a8`** (attributed to glm-cli1, explicitly unreviewed and unmeasured)
   and the founder **reallocates the slot to fable-1** — availability handoff,
   not a quality judgement. This is the protocol working: work is preserved,
   attributed, and transferred; the replacement verifies rather than assumes.
5. **→ 04:50Z** — fable-1 closes P0-F with a four-commit stack on
   `lane/fable-1/p0f-ffi-takeover`: `7a871288ec` re-entrant extern blocks (the
   core defect — replaced the inherited aggregate-global design after measuring
   three failed variants), `637dbf751c` fail-closed externs +
   exit/abort/malloc/free intrinsics, `e2d20025c5` `system()` via
   fork/execve/wait4 + dispatch close-out, `433715ff7a` the KNOWN_LIMITATIONS
   correction. `make madaros-full-gate` PASS **against a Madaros built from
   branch tip that session, not the checked-in ELF**; suite 983 pass / 563
   baseline-fail / 103 known with **zero attributable regressions**, verified
   against a parent-commit control build. Residual:
   `tests/run-pass/ffi_system_array_arg.sio` checked in as known-failure
   (`&[i8;N]` extern args forward an empty pointer through the signatureless
   `ffi_` path — needs a follow-on dispatch giving `ffi_` builtins real
   parameter types). Owed: a forensic dispatch for the large-aggregate-in-global
   seed miscompile (repros preserved under `docs/audit/p0f_repros/`, not yet
   minimised).

## 6. Operational lessons this fleet paid for (2026-08-15/16)

These are the reusable part of the session. Each cost real hours.

1. **Build your instrument from source; never trust the checked-in ELF.** A
   stale `bin/souc-lean-single-x86_64` (three weeks old, #725) cost a full false
   investigation — including a confident misdiagnosis that the binary was "a
   different tool entirely" when it was merely stale. Worse, a Madaros built on
   a stale base produced **phantom parse failures in a file that was pristine**.
   Before believing any anomalous compiler behaviour: rebuild from the branch
   tip (`make build-madaros`, self-locking; `scripts/dev/souc-build-lock.sh make
   build` for the seed) and re-run against the fresh binary. The seed now has a
   freshness drift-guard (`make lean-seed-gate`); the habit is still yours to
   keep. Gate receipts must name the binary they ran against.
2. **Never give a module global an aggregate type.** The self-hosted seed
   miscompiles large aggregates in globals — **two independent lanes hit it the
   same night**. Parallel primitive arrays (the `parser.sio` house style) are
   the safe shape. The forensic dispatch is still owed; until it lands, treat
   any `var X: SomeStruct[…]` / `Option<Box<…>>` at module scope in
   `self-hosted/` as suspect when behaviour makes no sense.
3. **`&`-launched background jobs do not reliably fire their completion
   waiters — poll.** Ampersand-detached work has silently stranded lanes this
   session: the waiter never fires, the lane waits forever on a notification
   that will not come. Poll the output file / process state on a cadence instead
   of blocking on a promised wake-up.
4. **Uncommitted work in a worktree is one accident from loss.** The approved
   wave-1 architecture docs sat untracked in a shared checkout until the
   preflight review flagged it (finding C7) and the tranche was landed as
   `6f2c4e2461`; P0-F's mid-flight WIP survived *only* because the orchestrator
   preserved it as a commit at reallocation. Commit WIP before you run low on
   context, before a takeover, and before anything touches a shared checkout.
   A `wip(...)` commit with an honest "unreviewed, not claimed green" message
   is cheap insurance; `git clean` is not.

## 7. Durable pointers

| Source of truth | For |
|---|---|
| `bin/sounio-coord` (+ `/workspace/.tmp/sounio-coord/…`) | live claims, heartbeats, messages |
| `.claude/attention_p0.v1.json` | P0 slot state — machine-readable, orchestrator-updated |
| `.claude/ATTENTION_CHARTER.md` | the ranking everything else obeys |
| `docs/internal/coordination/MADAROS_FOCUS_PLAN_2026-08-16.md` | workstreams, waves, founder decisions |
| `docs/internal/coordination/COMPILER_LANE_CONTRACT.md` | compiler lane states |
| `.claude/PARALLEL_BLOCKER_CONTRACT.md` | blocker shape, merge contract |
| `docs/MADAROS_STATUS.md` | status — **verify its date before trusting it**; regenerating it is WS-A work |
| `.claude/MEMORY_LANES.md` | per-lane memory files (token hygiene: read only your lane) |
| `artifacts/omega/agent_handoff.log.md` | durable CLAIM/RELEASE archive |
| `docs/audit/` | forensic dispatches — the protocol before any `self-hosted/` fix |

— end of rewrite; next revision owed when the fleet shape or the P0 queue
materially changes, not on a calendar.
