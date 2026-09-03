<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.stack-class-resolved-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.stack-class-resolved-2026-06-02
-->

# Stack-overflow class RESOLVED by the standard large stack — heap-tables refactor UNNECESSARY (2026-06-02)

## The finding (advisor-directed empirical test, definitive)

The residual ~222 "crashes" were measured under a NON-STANDARD `ulimit -s 65536` (64MB) sweep
cap. The project's OWN tooling runs the compiler with a large stack: `canonical_compiler_gate.sh`
uses `ulimit -s 1048576` (1GB); several others use `unlimited`. The OS default soft stack here
is only ~12MB; the HARD limit is unlimited.

Re-sweeping the full 847 examples at the canonical 1GB stack:

  rc=139 crashes:  @64MB = 222   ->   @1GB = 7

So **215 of the 222 were stack-overflow artifacts of the harness cap**, not compiler defects.
The frame-shrinking done this session (0c2edf193 binary tail, 373e30dfb call path → call frame
12.3MB→676KB) lowers the stack REQUIREMENT; the standard 1GB invocation covers the rest.

## The heap-allocate-Checker-tables refactor is NOT needed

It was the proposed structural fix for the stack-overflow class. But:
- The class is already resolved by running with the standard large stack (the project convention).
- The refactor is an 86-field / 49-table / 216-site change that converts inline Tables to Box,
  which SHARES tables on the pervasive by-value `c = c.method()` copy → ALIASES the save/restore
  idioms (check_if/check_match `saved = c.refine_cond_env; …; restore`) → silent wrong-type-checks
  the corpus sweep would NOT catch. High risk, large blast radius, for a solved problem.
DECISION (with advisor): do NOT do the Box refactor. Run with the standard stack.

## The 7 residual crashes @1GB are the SRET-smash class, NOT stack overflow

gdb on them (ekan_concrete_uci, ekan_energy_uci, ekan_wine_uci, epistemic_classifier,
lean_mini_compiler, lean_utils_self_host, mlp_concrete_ablation): `rip` is in the STACK region
executing zeroed bytes (return-address smash) — the bin/souc large-struct-SRET miscompile, the
SAME class the *mut migration fixes. They crash at `unlimited` too (stack-independent). Next
step for them: gdb/print-trace each to find the unmigrated by-value path it hits, *mut it
(same proven pattern). NOT a stack issue.

## Net (HEAD after this)

Modular compiler `--check` over 847 examples at the standard 1GB stack: 142 rc=0, 698 rc=1
(complete with type errors), **7 rc=139** (SRET-smash stragglers). From 481 original crashes →
7 = **474 eliminated (98.5%)**: ~259 via the *mut SRET migration (enum + 13 expr kinds + Call
arg-checker), ~215 via correct stack sizing. The sweep tool (.dbg/g1corpus/sweep.sh) now uses
the canonical 1GB stack so future measurements reflect real usage.
