# LLM Offload Audit Log

Append-only. One entry per non-trivial offload attempt (caught a bug,
informed a design decision, blocked a commit) or per M1-mandatory review
that could not run, per `.claude/AGENT_OFFLOAD_POLICY.md`. Newest entries at
the bottom.

Format: `## <UTC timestamp> — <agent> — <what>` followed by outcome and
reasoning.

---

## 2026-09-26T12:15Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1 math-review, equilibrium.sio test fixes (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | equilibrium.sio, acids.sio, stoichiometry.sio, thermochem.sio, test_equilibrium_acids.sio (ΔG sign fix in test_real_delta_g; quadratic root fix in test_ab_c; cross-module private-symbol collision closed at the stdlib level) | WAIVED | No offload provider configured in this container (`bin/llm-offload --status`); verified by hand + Python instead. Full narrative below. |

Note: `scripts/dev/check_offload_policy.sh`'s `MATH_REVIEW_PATHS` regex does
not currently include `stdlib/chemistry/*`, so this row is not required by
that gate for these specific files — added anyway, both because the
underlying M1 policy trigger (a hand-derived math correction) applies
regardless of which paths the automated gate happens to cover today, and to
match this log's established row format for anyone grepping it later.

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
module name. **Correction, same session**: this was first written up as an
open compiler defect needing a forensic dispatch. It is not open — Copilot's
review on the PR pointed at `docs/audit/MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md`,
which documents this exact stdlib/chemistry pair as the motivating example
for `self-hosted/compiler/private_fn_identity.sio`, landed 2026-09-21 (5
days before this session). Verified directly rather than taking the doc's
word for it: `make build-madaros` from current source, then the same
two-import-order repro, gives the correct `3.0` both ways. The committed,
shipped `bin/madaros-linux-x86_64` this repo ships predates that fix and
still segfaults on the same repro (exit 139) rather than silently picking a
body. Lesson for this log: `bin/souc`/the committed ELF lag source
(CLAUDE.md operating principle 15) applies to reasoning about defects, not
just to benchmark numbers — "reproduces on the shipped binary" and "open in
current source" are different claims, and this entry conflated them. The
stdlib prefix rename stays: it is no longer closing an open compiler
defect, but it does mean acids::ph() doesn't depend on which Madaros a
caller happens to be running.

---

## 2026-09-26T13:45Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1 math-review, kinetics.sio's 9 failing self-tests (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | kinetics.sio (Mittag-Leffler target recomputed for test_fractional_caputo_epistemic + validate_against_literature's pass_frac; PINN training-loop learning rate re-derived; simulate_bayesian_posterior_crn's importance-sampling combination moved to log-space) | WAIVED | No offload provider configured in this container, same as the entry above. Independent Python verification for each derivation; full narrative in the commit message (b28e33b8) and in the fix comments at each site. |

Same gap as the entry above (`MATH_REVIEW_PATHS` doesn't cover
`stdlib/chemistry/*`) and same reasoning for adding the row anyway. Three
hand-derived math corrections in this pass, each independently verified in
Python before being written into the .sio file, not asserted from memory:

1. **Mittag-Leffler value.** `fractional_decay_ml(1.0, 0.1, 10.0, 0.8)` calls
   `mittag_leffler_e_alpha(0.8, -0.630957)`. The file's own test asserted
   0.35; independently implemented the identical power series (gamma-function
   terms, `sum += z^k/Γ(0.8k+1)`, converges cleanly since |z|<1) in Python:
   0.533673. Cross-checked the qualitative claim in the neighboring comment
   ("slower decay than integer order") against exp(-1)=0.367879: 0.533673 is
   indeed larger (slower decay), so the DIRECTION was right and only the
   magnitude was a guess. Both `test_fractional_caputo_epistemic` and the
   textually-identical `pass_frac` computation inside
   `validate_against_literature` used the same wrong 0.35 and got the same
   fix.
2. **PINN training-loop learning rate.** `k = k - lr * dloss_dk` starting at
   k=0.12, target k=0.1: reimplemented the exact loop (same `exp_approx`,
   same finite-difference gradient, same 5 iterations) in Python at lr=0.1
   (the file's value) and at several candidates. lr=0.1 oscillates
   (0.12/0.080/0.153/0.088/0.128/0.078, no convergence in 5 steps); lr=0.05
   converges to k=0.099995 — the change applied. Two duplicated copies of
   this loop existed (`test_epistemic_pinn_crn`,
   `bench_epistemic_pinn_crn`); both fixed identically.
3. **Bayesian importance-sampling combination.** Not a single wrong number
   but a numerically unstable combination rule:
   `w = w * likelihood_weight(...)` multiplying several
   already-small Gaussian-kernel values underflows to exactly 0.0 in f64
   once no single sampled parameter matches all of several spread-out data
   points simultaneously against tight sigmas — confirmed by instrumenting
   the function directly (added and then removed temporary debug prints
   showing `sum_w=0.0` for every sample, every call) rather than inferring it
   from the symptom alone. Replaced with the standard fix: accumulate
   log-weights, then exponentiate once per sample relative to the batch's
   own maximum log-weight (so the best sample never underflows, and the
   others are correctly small relative to it). This is a numerical-methods
   correction, not a literature-derived constant, included here because
   `.claude/AGENT_OFFLOAD_POLICY.md` §M1 lists "GUM uncertainty-propagation
   derivations" and this is the same class of hand-derived probabilistic
   math.

Also found and fixed in the same pass, not itself an M1 trigger but
recorded for continuity: a `let var = ...` binding (`var` is the
mutable-declaration keyword) in the function this same investigation was
chasing, which made every `if var <= 0.0 {...}` comparison against it take
the true branch unconditionally under `SOUNIO_SOUC_ENGINE=lean_single` (a
minimal 6-line standalone repro confirms it; Madaros correctly refuses the
same file as using a reserved identifier). This was the actual reason
`simulate_bayesian_posterior_crn` had been silently returning its
unmodified prior on every call, independent of the two other bugs
(an indexing bug and the log-space fix above) layered on top of it. Not a
math-review trigger by itself, and not something this stdlib fix can close
in the compiler — flagged here for a forensic dispatch per CLAUDE.md's rule
against patching self-hosted/ ad hoc, same posture as the private_fn_identity
entry above.
