<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.bare-pattern-reject-guard-readiness-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.bare-pattern-reject-guard-readiness-2026-06-01
-->

# #3 — bin/souc reject-bare-variant guard: READINESS (groundwork done, re-bootstrap STAGED)

**Goal:** make `bin/souc` emit a checker ERROR on a bare (unqualified) enum-variant
match pattern, so the whole bug class (silently miscompiled all-arms-fire) fails
loudly instead of producing runaway recursion. The advisor's call: REJECT (force
qualification) is more correct than ACCEPT (resolve+dispatch bare), and matches the
codebase convention (every file qualifies).

## Feasibility — CONFIRMED safe (2026-06-01)
- `self-hosted/compiler/lean_single.sio` (bin/souc's own source) has **0 bare match
  arms** → bin/souc can recompile ITSELF under the guard (gen1→gen2→gen3 won't choke).
- The modular closure of `self-hosted/compiler/main.sio` (72 `use`s: lexer, parser/ast,
  ir, check/*, resolve/*, io, native) has **0 remaining bare enum-variant arms** after
  this session's qualification (check/compat/resolve done) → `mc.elf` builds under the
  guard. printer/* and self-hosted/main.sio are NOT in the closure (irrelevant).
- (epistemic/dependent `PROVENANCE_KIND_*`/`DEP_CONSTRAINT_KIND_*` arms are ALL-CAPS
  const-pattern matches, a DIFFERENT category — the variant guard must NOT flag those.)

## Why this is STAGED, not executed this session
The repo's hard rule (memory + CODEGEN_ROOTFIX_PLAN): editing lean_single.sio +
re-bootstrap is the highest-risk op and **must be done in FRESH context, never at the
tail of a long build-heavy session** (a prior guess-edit near-OOM'd the pod). This
session is build #6+. #3 is PREVENTION (not unblocking) — low urgency. So: groundwork
done, execute fresh.

## Execution procedure (fresh session)
1. Find the pattern-resolution site in lean_single.sio (the checker arm that handles a
   bare identifier pattern — currently treats an unresolved CamelCase variant as a
   catch-all/binding → all-arms-fire). Likely near PatEnum / pattern-checking.
2. Add: if a bare CamelCase pattern does NOT resolve to a binding AND is a known enum
   variant reachable only via `Enum::Variant`, emit a checker error
   "unresolved variant `X` — qualify as `Enum::X`". Do NOT flag const patterns
   (ALL_CAPS) or builtins (Some/None/Ok/Err).
3. Re-bootstrap discipline: gen1 (`bin/souc lean_single /tmp/g1`), gen2 (`/tmp/g1
   lean_single /tmp/g2`), gen3 (`/tmp/g2 … /tmp/g3`); REQUIRE gen2==gen3 (md5).
4. Run the FULL canonical gate + wide run-pass/compile-fail sweep BEFORE replacing
   bin/souc. A miscompiling intermediate bricks the toolchain.
5. Only then install gen3 as bin/souc. Cap stack (never `ulimit -s unlimited`).
6. Verify: a deliberately-bare test program now ERRORS with the new message; the whole
   tree still builds.

## Note on value
The guard PREVENTS future bare-pattern bugs; it does NOT fix the ~481 example programs
that still crash on mc.elf — those are OTHER modular-frontend bugs/feature-gaps (see
the #2 diagnosis). Sequence the guard AFTER the high-value feature work if prioritizing.
