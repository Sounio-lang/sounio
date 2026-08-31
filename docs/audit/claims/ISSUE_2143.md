<!-- docs:meta
topic_id: repo.docs.audit.claims.issue-2143
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.claims.issue-2143
-->

# Claims — #2143, the FO channel table leaks across functions

Produced by `fable-1`, verified by `claude-1`. This file is the pilot for
`docs/governance/PAIR_VERIFICATION_PROTOCOL.md`: it is written as numbered
atomic claims with a verdict from someone who did not produce them, rather than
as prose.

Every verdict was reached by a DIFFERENT route than the claim. That is the
point: re-running the command that produced a claim reproduces its errors too.

---

## CLAIM-1 — `begin_function_body_lowering` has no callers

`self-hosted/ir/lower.sio` defines it with the comment "D5 (#986): per-function
reset of vreg-keyed float/variance tables", and the name occurs once in the
whole `self-hosted/` tree: its own definition.

- produced-by: `grep -rn --include='*.sio' 'begin_function_body_lowering' self-hosted/`
- **VERDICT-1: CONFIRMED** by claude-1
- via: counting the *live* per-function entry instead — located
  `lowerer_lower_fn_item_mut` (4343–4675 on main) and read every line of it,
  rather than grepping for the dead name. Positive control: the neighbour
  `flush_current_func` returns 11 occurrences, so the extraction sees names.

## CLAIM-2 — `fo_nchan` and `fo_bind_count` are zeroed only in dead code and in constructors

- produced-by: an assignment audit of `lower.sio`
- **VERDICT-2: CONFIRMED** by claude-1
- via: grepping for *writes* of any value (`fo_nchan *=`), not for the zeroing:
  `6255` (inside the dead function) and `8691` (growth); `fo_bind_count` at
  `6256`, `8064`, `8807`. Then the four constructor sites separately
  (`fo_nchan: 0` at 1160, 1691, 1755, 1823).

## CLAIM-3 — those constructors run once per module, not per function

- produced-by: reading their names
- **VERDICT-3: REFUTED** by claude-1 (self-refuted; see below)
- via: a lowering-time trace under `SOUNIO_FO_SEED_TRACE=1` at both decision
  points of `fo_seed_from_variance`. Three same-shaped functions produce:

      NEW   ch=0 vreg=3      <- function 1
      REUSE ch=0 vreg=3
      NEW   ch=0 vreg=3      <- function 2, opens at ch=0 again
      REUSE ch=0 vreg=3
      NEW   ch=0 vreg=3      <- function 3, likewise
      REUSE ch=0 vreg=3

  Every function opens at channel 0, so `fo_nchan` IS zero at each function's
  start and the channel table does NOT carry across functions.

  This claim was first marked CONFIRMED by the same verifier, from counting
  CALL SITES in source and concluding execution frequency. Those are different
  quantities. Nine textual calls to `lowerer_new` say nothing about how often it
  runs, and the trace says it runs often enough that the table is fresh per
  function. The verifier's method was the defect, not the producer's claim.

  What resets it is NOT yet identified: the live entry
  `lowerer_lower_fn_item_mut` takes `&! Lowerer` in place and provably does not
  touch these fields (CLAIM-4, unaffected). Whether a fresh Lowerer is
  constructed per function, or the increments are lost across a by-value `self`
  boundary, is UNMEASURED.

## CLAIM-4 — the live per-function entry never touches the FO channel table

- **VERDICT-4: CONFIRMED** by claude-1
- via: reading the entire body of `lowerer_lower_fn_item_mut` for any mention of
  `fo_nchan|fo_sigma2|fo_bind_count|FO_COV_RHO|FO_SIGMA_REG`. The only hit is
  `FO_COV_RHO`, the reset landed by #2145.

## CLAIM-5 — there is no symbol `FO_SIGMA_REG`

- produced-by: `fable-1`, correcting claude-1's original report
- **VERDICT-5: REFUTED** by claude-1
- via: direct grep on `origin/main` — four occurrences, including the
  declaration `var FO_SIGMA_REG: [i64; 32] = [-1; 32]` at `lower.sio:616` and a
  write at `6260`. It is a module global distinct from the `fo_sigma2` Lowerer
  field, sitting in the same dead reset loop — a fourth leaking sibling, not an
  alias of the second.

## CLAIM-6 — a vreg collision across functions makes `fo_seed_from_variance` reuse a stale channel, producing wrong variance

- produced-by: inference from the reuse condition `fo_sigma2[k] == variance_reg`
- **VERDICT-6: REFUTED** by claude-1
- via: the same trace. The vregs DO collide across functions -- `vreg=3` in all
  three, so the witness's central assumption was correct and it was not inert --
  but each function opens at `ch=0`, so the reuse that fires is with the
  function's OWN earlier entry, never with another function's. No stale channel
  is reachable, and no wrong number is produced.
- superseded reasoning: a witness with three same-shaped functions
  (`tests/run-pass/fo_channel_leak_across_functions.sio`, branch
  `test/fo-channel-leak-witness`) returned the CORRECT sigma for each under
  source-built Madaros — 0.010000 / 0.250000 / 0.040000, matching lean_single.
  That is not a latency proof: the witness assumes same-shaped functions produce
  colliding virtual registers, and that assumption was never verified. If the
  regs do not collide, the witness never exercised the reuse path at all.
- what would settle it: a lowering-time trace of `(vreg, chosen channel, sigma
  now, sigma stored)` at both decision points in `fo_seed_from_variance`. In
  progress under `SOUNIO_FO_SEED_TRACE=1`.

---

## Why CLAIM-6 stays open

The structural leak (1–4) is proven. Whether it produces a wrong number is a
different claim, and in this repository the first has been true with the second
false before — #2152 took four probes to separate exactly that pair, and three
of them returned confident answers to questions they were not asking.

An investigation with five confirmed claims and one honest UNMEASURED is worth
more than six confident ones.
