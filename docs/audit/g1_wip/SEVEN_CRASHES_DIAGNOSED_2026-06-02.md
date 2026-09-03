<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.seven-crashes-diagnosed-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.seven-crashes-diagnosed-2026-06-02
-->

# Seven Crashes Diagnosed — modular corpus under g1 / *mut / move-codegen (2026-06-02)

> **STATUS UPDATE 2026-06-10** — Cluster C ("known large-SRET miscompile,
> NOT pursued") is now largely stale: the SRET-forwarding repros pass on the
> current `bin/souc` (see the banner in `SRET_FORWARDING_BUG_2026-06-02.md`;
> regression pins in `tests/run-pass/sret_forwarding_*.sio`). Residual open
> member of the family: forward-in-aggregate returns uninitialised memory
> (known-failure pin). The modular `--check` crash frontier re-measured at
> **1 crasher** in a 109-file corpus sample (`slice_fat_pointers.sio`) — see
> [`../MODULAR_COMPILER_AUDIT_2026-06-10.md`](../MODULAR_COMPILER_AUDIT_2026-06-10.md).

**Scope.** Census + adversarial classification of SIGSEGV (rc=139) failures when running the modular/move-codegen compiler (`mc.elf` built from the g1/move-codegen tree) over the `tests/run-pass` corpus (and related). Work reduced an initial large crash frontier (tens of rc=139 under the checker) via collectors and *mut-spine migrations. After shippable changes, **7 crashers remained**; they were clustered for routing.

**Decision (user explicit, this session).** "leave C documented as the known large-SRET miscompile." 5 of the 7 resolved through shippable changes (4 via cluster-A live, 1 via cluster-B committed). The remaining 2 are cluster C.

## Cluster breakdown
- **Cluster A (4 crashers)**: resolved via live *mut / spine work (e.g. typed-closure literal, if/else, match deref + heterog. tuple, etc.). Shippable, landed or in the live g1 lane.
- **Cluster B (1 crasher)**: resolved via committed codegen fix (the nested-*mut-write / boundary-check hoisting that collapsed a larger 170-crash class; see handoff "DEEPER-CRASH 2026-06-02T~23:35Z" and the E008_ROOTCAUSE... doc).
- **Cluster C (2 crashers)**: **NOT pursued**; left as documented known limitation.

## Cluster C — the known large-SRET miscompile (by-value Checker 8 MB)
C is the known large-by-value-Checker (8MB) SRET miscompile — the bug the entire modular/move-codegen *mut arc exists to avoid.

The patch's own comment (line 103) names it: holding `let t = (*c).report_*()` inflates the caller frame to 8MB and "smashes the return address."

**Three facts make the dive a trap:**
1. The direct `bin/souc` large-SRET codegen fix is intractable-without-gdb (your own recorded B-repro verdict; see also the gdb-pinned forwarding bug in `SRET_FORWARDING_BUG_2026-06-02.md` and the minimal/tuple-wrapped crash repros).
2. The only tractable fix — route the remaining by-value Checker return via `*mut` — lives inside the effect patch, which is net-negative and won't ship.
3. Fixing the crashes wouldn't make that patch shippable anyway: its blocker is the ~95 false-passes, not the 7 crashes.

**Root cause family (sibling to the G1 "SRET-smash").** Large-struct value-move / SRET-forwarding in the legacy by-value path (lean_single.sio). A return-position struct-returning call drops the caller-supplied sret pointer; the inner callee writes into its own local temp; the outer returns the (now-stale or zeroed) address region. Layout-sensitive (silent zero for many cases; sentinel deref / rc=139 when the aggregate is large or wrapped, e.g. the Checker itself with its 20k+ slots / ~168 KB per copy, dozens stacked in the check spine → multi-MB frames and return-address smash).

See:
- `SRET_FORWARDING_BUG_2026-06-02.md` + `SRET_FORWARDING_MINIMAL_REPRO_2026-06-02.sio` + `SRET_FORWARDING_CRASH_REPRO_2026-06-02.sio` (gdb pin, isolation matrix, "return-of-call" vs "return-of-local", escalation on tuple-wrapped).
- `MODULAR_COMPILER_STACK_CLASH_2026-05-29.md` (the 7.6 MB `check_expr` frame as ~46 by-value Checker copies; the `*mut` refactor of the check phase as the credible, general avoidance).
- `artifacts/omega/agent_handoff.log.md` entries on the by-value Checker truncation (mc.elf census, "copied 8MB Checker as self on EVERY user-fn call arg -> stack smash"), the E008 by-value bridge falsification, and the deeper-crash reductions.
- `SOURCE_TO_ELF_BRIDGE_PROTOTYPE_2026-06-02.md` (IrModule large-struct return also surfaces the family).
- The *mut-spine work (collectors, in-place handlers) that converted most old crashes into rc=1 and resolved A/B.

**Status.** NOT pursued. Documented known limitation of the pre-*mut / by-value Checker + legacy SRET codegen. The 2 remaining crashers are witnesses of this class. Future work on a full by-value Checker cleanup or a direct SRET forwarding fix in lean_single can use the repros and the 3 facts above; the current *mut arc (and move-codegen improvements) is the shippable path that sidesteps it.

## Supporting artifacts & cross-refs
- Census / backlog: `MODULAR_CORPUS_CRASH_CENSUS_2026-06-01.md`, `MODULAR_CORPUS_FAILURE_BACKLOG_2026-06-02.md` (update them with pointers to this doc).
- Handoff log (this tree + originating move-codegen session) for the exact 5/7, A/B/C language, and patch context.
- Repros under `docs/audit/g1_wip/` (SRET_* and any cluster-specific ones from the 7-crasher hunt).
- `FRONT_HALF_LEVERAGE_HANDOFF_2026-06-02.md` (the larger front-half context in which these crashes were triaged).

## Handoff
Diagnosis complete. Record made canonical. No further workflow / agents spun up for C (per the "leave documented" call and "trivial doc edit" assessment). Route any future large-SRET or full-Checker by-value effort here first.
