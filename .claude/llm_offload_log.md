# LLM Offload Audit Log

Append-only. One entry per non-trivial offload attempt (caught a bug,
informed a design decision, blocked a commit) or per M1-mandatory review
that could not run, per `.claude/AGENT_OFFLOAD_POLICY.md`. Newest entries at
the bottom.

Format: `## <UTC timestamp> — <agent> — <what>` followed by outcome and
reasoning.

---

## 2026-09-26T12:15Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1 math-review, equilibrium.sio test fixes (PR sounio-lang/sounio#2694)

**Trigger**: two hand-derived math corrections in `stdlib/chemistry/equilibrium.sio`
(a sign flip in a ΔG↔K test input; a quadratic-equilibrium root recomputed
from scratch), which `.claude/AGENT_OFFLOAD_POLICY.md` §M1 marks mandatory
for `bin/llm-offload -t math-review`.

**Attempted**: `bin/llm-offload --status` — reports no key file found; no
provider (xai, zai, or local) is configured in this container. Did not
attempt `bin/llm-offload -t math-review` itself since `--status` already
shows no provider can be reached; there is no second-choice provider to
retry per the policy's "Provider down / timeout" row, only "no provider at
all," which the same row's guidance ("if all fail, document ... and
proceed") is written for.

**Outcome**: WAIVED for lack of a reachable provider, not skipped by choice.
Both corrections were independently verified without an LLM: by hand
(ΔG = −RT·ln K is negative for K>1, so the test's `+5700.0` J input is
inverted; the correct value is `-5700.0`) and cross-checked numerically in
Python (`k_from_delta_g(-5700.0) ≈ 9.979`, matching the asserted K≈10 to
within the test's own tolerance). The quadratic root for `test_ab_c`
(A+B⇌C, a0=b0=1, K=10) was solved independently as
`x=(21−√41)/20=0.729844`, matching `solve_ab_c_equil`'s own computation
exactly, confirming the *function* was already correct and only the
*test's* prior expectation (0.45) was wrong. Both fixes verified by actually
running the module (`bin/souc run stdlib/chemistry/equilibrium.sio` →
`EQUILIBRIUM + ELECTRO + SOLVERS REAL ALL PASS`), not just `check`ing it.

**Flagged for re-review**: per the policy's failure-mode table, this should
get an orthogonal-provider pass once a provider is configured in a session
that has one, per the mandatory M1 checkpoint this entry stands in for.

Related, found in the same audit pass and fixed in the same PR (not itself a
math-review trigger, but recorded for continuity): a cross-module private-
symbol collision in `stdlib/chemistry/{equilibrium,acids,stoichiometry,thermochem}.sio`
— all four declared their own private `abs_f64`/`check_near`/`sqrt_approx`,
and `equilibrium`+`acids` additionally both declared `ln_approx` with
*different* bodies (acids.sio's does range-reduction; equilibrium.sio's does
not). Importing both in one file resolved `acids::ph()`'s internal
`ln_approx` call to whichever module's copy the import order favored,
silently: `ph(1e-3, 1e-5)` returned the correct `3.0` with
`use chemistry::acids::*` first, and the wrong `2.119756` with
`use chemistry::equilibrium::*` first — no error, no warning. Minimal
repro kept in this session's scratchpad
(`/tmp/claude-0/collision_repro4.sio`, ~9 lines plus the two stdlib modules).
Fixed at the stdlib level by prefixing every colliding helper with its
module name; the underlying compiler behavior (a private symbol apparently
resolved across module boundaries by declaration/import order during
multi-module lowering) is unfixed and out of scope for this PR — this is a
stdlib workaround, not a compiler fix, and is flagged here for a forensic
dispatch per CLAUDE.md §4/§8 ("do not patch self-hosted/ ad hoc").
