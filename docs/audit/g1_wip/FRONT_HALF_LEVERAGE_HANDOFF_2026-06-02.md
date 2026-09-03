<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.front-half-leverage-handoff-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.front-half-leverage-handoff-2026-06-02
-->

# Front-half leverage handoff — the #1 unaddressed lever + 4 ready parser prereqs (2026-06-02)

For whoever drives the G1 front-half next. Crashes are essentially closed (live lane:
222→0 via *mut migration + 1GB stack). **The largest remaining corpus blocker is NOT a
crash — it is the spurious-type-error class, and it is still unaddressed.** Measured on
g1 tip `0b42cf301` (binary built this session, `ulimit -s 1048576`):

- **124/504 pass, 377 rc=1, 3 rc=139.** Pass-count is UNCHANGED from the pre-crash-fix
  census (124) — the crash work moved rc=139→rc=1 but flipped no programs to pass.

## The #1 lever: E008 spurious return-type (one bridge-state bug) — 132 programs
- **E008 "expected `()` / found <T>" — 106 programs** (100% the spurious unit-return pattern
  in the earlier census). Root: the by-value bridge (`checker_check_expr_mut` →
  `(*c).check_expr`, **check.sio:1146**) drops `current_return_type` to `TyUnit` at the
  return site (`checker_check_return_expr_inplace`, **check.sio:~2489**), so every
  value-returning fn mismatches.
- **E170 ".value requires `with Epistemic`" — 27 programs.** Same bridge drops the
  `current_effects` row.
- **Fix:** carry `current_return_type` + `current_effects` across the by-value bridge (read
  `FnSig.ret`/`.effects` via the proven `.find`/`.get` path) instead of defaulting to
  unit/empty. Clears the SOLE blocker on ~96 programs, noise on ~36 (~132 touched). This is
  the single highest-leverage front-half fix — **larger than the entire crash class was.**

## E072 + the missing `is_extern` (blocks kernel + extern buckets)
- **E072 "kernel must return unit"** (check.sio:~12687, `is_kernel_fn && ret_ty != TyUnit`)
  fires on (a) real kernels whose return-type is mis-detected and (b) **extern fns**, which
  have no `is_extern` marker so they inherit `is_kernel:true`.
- **Fix:** add `is_extern: bool` to `FnDef`; for extern fns skip the body check AND the
  kernel-unit-return check. Unblocks the 9 extern + helps the 17 kernel programs.

## 4 parser prereqs already landed (this session, pushed, 0-regression, bin/souc untouched)
These are ready and will flip the moment the front-half above lands — they are jointly
blocked, not independent:
- `parser/sci-notation-float` (`f77ae77b0`) — **+15 ALREADY** (sole parse-blocked; the one exception).
- `parser/const-decls` (`0b3918faa`) — const parses; 8 ontology_generated_* progs blocked on **E008**.
- `parser/kernel-fn` (`f9c189397`) — kernel fn parses; 17 progs blocked on **E072**.
- `parser/extern-blocks` (`74368b22b`) — extern blocks parse; 9 progs blocked on **E072 + is_extern**.

## Coordination note
E008/E072 live in `check.sio`, which the G1 lane is actively *mut-migrating (live build
observed 2026-06-02). This handoff is a doc (no check.sio edit) to avoid colliding with that
in-flight work. Whoever takes the front-half: the return-type/effect bridge propagation is
the highest-value next move; the 4 parser branches are the downstream that converts it into
a large pass-count jump. Supporting data: `MODULAR_CORPUS_FAILURE_BACKLOG_2026-06-02.md` +
`MODULAR_CORPUS_CRASH_CENSUS_2026-06-01.md`.
