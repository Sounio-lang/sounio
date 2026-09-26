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
`EQUILIBRIUM + ELECTRO + SOLVERS REAL ALL PASS (phase3+)`), not just
`check`ing it.

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

---

## 2026-09-26T14:45Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1/M3, demos/hydrogen/README.md p-box and VoI numeric corrections (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | demos/hydrogen/README.md (caprock_seal_pbox theta-scenario reliabilities; caprock_integrity_v2 shale/mudstone p-box bounds; sobol_voi MC convergence/seed-invariance numbers and VoI sequencing widths) | WAIVED | No offload provider configured in this container, same as both entries above; independent verification instead. |

**Trigger**: `demos/hydrogen/README.md` is explicitly prepared for a presentation
to Dr. Emmanuel Stamatakis's group (NCSR Demokritos, H2Lab) — an
external-facing artifact under `.claude/AGENT_OFFLOAD_POLICY.md` §M3 — and
the corrections touch p-box/probability-of-failure arithmetic (§M1's
"GUM/p-box uncertainty-propagation derivations").

**Attempted**: `bin/llm-offload --status` — same result as the earlier
entries in this log: no provider (xai, zai, or local) reachable in this
container. No `--raw <draft> deepseek xai gemini` fan-out attempted for the
same reason `math-review` wasn't attempted above: `--status` already shows
nothing to fan out to.

**Outcome**: WAIVED for lack of a reachable provider. The corrections
themselves are **not** a hand-derived recomputation — they are transcription
fixes: the underlying `.sio` files (`caprock_seal_pbox.sio`,
`caprock_integrity_v2.sio`, `sobol_voi.sio`) were unmodified and already
correct; the doc's prose numbers had drifted from what those files actually
print. Verification was re-running each unmodified `.sio` file and diffing
its stdout against the doc's claimed numbers directly, not an independent
derivation of the underlying p-box/VoI math — so this waiver documents an
artifact-accuracy check, not a second opinion on the probabilistic model
itself. If the underlying p-box/VoI methodology in those `.sio` files
changes in a future PR, that would be a fresh M1 trigger needing its own
review attempt.

**Flagged for re-review**: per policy, both this entry and the two above
should get a real fan-out pass once a provider is configured in a session
that has one.

---

## 2026-09-26T15:15Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, simulate_bayesian_posterior_crn's concentration-time model rewrite (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | kinetics.sio (simulate_bayesian_posterior_crn/bayesian_structural_posterior/bayesian_identifiability_crn rewritten from single-endpoint comparison to a concentration-time trajectory fit; new simulate_general_crn_checkpoints; all Bayesian call sites' synthetic data regenerated) | WAIVED | No offload provider configured in this container, same as the three entries above. Independent Python cross-check of every generated dataset against the closed-form solution before writing it into the file; full narrative below. |

**Trigger**: this is the deepest M1 trigger in this PR — not a value correction
but a replacement of the underlying observation model, done at the user's
explicit request ("entra na dissertação, quero decidir o modelo") after
Copilot flagged (and the user was shown, via AskUserQuestion, the two
possible fixes) that comparing every observation to the same simulated
endpoint was not scientifically defensible for any current caller's
synthetic data.

**Attempted**: `bin/llm-offload --status` — same result as every prior entry
in this log: no provider reachable in this container.

**Outcome**: WAIVED for lack of a reachable provider. Independent
verification instead, at each step:

1. **Rate-law derivation.** Confirmed from `compute_rates_general`/
   `general_dc` (read directly, not assumed) that reaction r0 (species0 ->
   species1, rate k0) makes species 0's ODE `dC0/dt = -k0*C0` -- a pure,
   analytically exact first-order decay, decoupled from the downstream
   reaction r1 (fixed rate 0.05) since species 0 does not participate in
   r1. This is why species 0, not species 2, is the tracked quantity in
   the new model: it is the only species in this network whose trajectory
   is unambiguous and independently checkable.
2. **RK4-vs-closed-form cross-check.** Wrote a Python replica of the exact
   RK4 stepper `simulate_general_crn_checkpoints` performs (same
   `y_{n+1} = y_n + (dt/6)(k1+2k2+2k3+k4)` update, same rate law) and
   confirmed it reproduces the closed-form `C0(t) = C0(0)*exp(-k0*t)` to 6
   decimal digits at every checkpoint scheme used (dt in {0.2, 0.25}, k in
   {0.12}, step counts up to 40) -- e.g. checkpoints [7,15,22,30] at
   dt=0.2, k=0.12 gives RK4 (0.845354, 0.697676, 0.589783, 0.486752)
   against the closed-form (0.845354, 0.697676, 0.589783, 0.486752),
   identical to the printed precision. Every data_points array written
   into every call site (validate_against_literature, test_bayesian_ident_crn,
   test_bayesian_structural_tie, bench_scale_bayes_samples, the "SUPER
   SHOWCASE QUÍMICO" demo) is this RK4 output, not a hand-picked or
   guessed number.
3. **New-function correctness.** `simulate_general_crn_checkpoints` was
   type-checked (`souc check`, clean) and its output was diffed against
   `simulate_general_crn`'s existing, already-tested final-state output at
   the same total step count for a sanity check (both report the same
   final species-0 value) before being used anywhere else.
4. **A real bug caught by this verification process itself, not by
   review**: the first version of `test_bayesian_permutation_invariant`
   permuted `checkpoint_steps` directly (e.g. to `[30,7,22,15]`), silently
   violating a precondition of `simulate_general_crn_checkpoints`
   (checkpoint_steps must be strictly ascending, since the loop runs to
   `checkpoint_steps[3]` and detects checkpoints in array order) -- the
   loop stops after 15 steps and never reaches the checkpoints that would
   have come later in a correctly-ordered array, silently leaving those
   slots at their zero-initialized default instead of erroring. Caught
   because the test itself then failed unexpectedly (`pk=0.114357` vs
   `pk2=0.099572`, nowhere near the invariance the test was supposed to
   demonstrate) -- traced to the precondition violation rather than
   patched around with a looser tolerance. Replaced with
   `test_bayesian_checkpoint_mismatch_sensitivity`, which keeps
   `checkpoint_steps` fixed and ascending and instead shuffles which VALUE
   occupies which slot -- verified to discriminate (`pk_correct=0.114357`
   vs `pk_mismatched=0.095415`, a ~0.019 gap against a 1e-3 tolerance).

Also updated `bayesian_identifiability_crn` (a "compat wrapper," verified
via repo-wide search to have zero current callers): it takes a single
scalar `data_v`/`data_u` pair, so it cannot synthesize a genuine multi-time
trajectory without inventing numbers; modeled honestly instead as 4
genuine repeat measurements of species 0 at the SAME single checkpoint
(`[T,T,T,T]`), a real, common experimental design (e.g. replicate assay
readings) that `simulate_general_crn_checkpoints` handles correctly by
construction (repeated checkpoint values are detected and recorded
identically), rather than the previous `[v, 0.7v, 0.5v, 0.3v]`
descending-spread fabrication.

**Flagged for re-review**: per policy, alongside the three entries above,
once a provider is configured in a session that has one.

## 2026-09-26T15:30Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, log_gaussian_term standardized-residual fix (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | kinetics.sio (log_gaussian_term rewritten to compute the standardized residual z=(sim-data)/data_u before squaring, instead of squaring data_u and resid separately) | WAIVED | No offload provider configured in this container, same as the four entries above. Independent Python cross-check of old vs. new formula on normal-range, underflow, and Inf/Inf inputs; full narrative below. |

**Trigger**: Copilot review on the concentration-time model rewrite above
found that the old body, `variance = data_u * data_u; ... resid*resid /
variance`, underflows `variance` to exactly `0.0` for any positive but very
tight `data_u` (e.g. `1e-200`, since `1e-200^2 = 1e-400`, below the
smallest positive f64 denormal `~4.9e-324`) — the `variance <= 0.0` guard
then fires and returns the same constant `-1e300` regardless of `resid`,
conflating "this proposal exactly matches a very precise observation"
(should be the BEST possible log-weight, `0.0`) with "this proposal is
impossible given a very precise observation" (should be `-Inf`, correctly
rejected). Also flagged: squaring `resid` and `data_u` independently can
produce an `Inf/Inf = NaN` when both are large, which is worse than either
endpoint alone since NaN is invisible to the `sum_w <= 0.0` fallback
(every IEEE comparison against NaN is false).

**Attempted**: `bin/llm-offload --status` — no provider reachable in this
container, same as every prior entry in this log.

**Outcome**: WAIVED for lack of a reachable provider. Independent
verification instead:

1. **Reproduced the underflow directly in Python** (`resid=0.5001-0.5`,
   `data_u=1e-200`): old body returns the same `-1e300` whether `resid` is
   `0.0` (perfect match) or `0.0001` (real mismatch); new body
   (`z = resid/data_u`, then `-0.5*z*z`) correctly returns `-0.0` for the
   perfect match and `-inf` for the mismatch — verified both are now
   distinguishable, and that `-inf` is caught by the existing
   `is_finite_f64` guard at the call site (proposal skipped, not folded
   into the accumulator as NaN).
2. **Reproduced the reviewer's Inf/Inf case** (`sim=1e200, data=0.0,
   data_u=1e200`): old body computes `resid*resid = inf`,
   `variance = inf`, `inf/inf = nan`; new body computes `z = 1.0` first
   (finite), giving a correct finite result (`-0.5`).
3. **Confirmed numerical equivalence on every realistic input.** For every
   `data_u` value any current caller actually passes (`0.02`-`0.05`), the
   new formula matches the old to floating-point noise (~1e-15 relative,
   from reordering the division before vs. after squaring) — e.g.
   `sim=0.5, data=0.45, data_u=0.05`: old `-0.49999999999999967`, new
   `-0.4999999999999998`. No existing call site's numeric output changes
   beyond last-bit rounding.
4. **Regression test added directly in kinetics.sio**
   (`test_log_gaussian_term_tiny_uncertainty`, next to `log_gaussian_term`
   itself since that helper is private and the external fixture can only
   see `pub` functions), wired into
   `tests/stdlib/chemistry/test_kinetics_fixed_regressions.sio`. Verified
   as a real control, not a vacuous assertion: temporarily reverted
   `log_gaussian_term` to the old body and re-ran the fixture under the
   exact CI mechanism (`SOUNIO_TEST_SOUC_BIN` at the lean_single stage2
   binary) — it no longer reports `PASS` (run exits non-zero), confirming
   the test suite does catch this regression; restored the fix and
   re-confirmed a clean `PASS`.

**Flagged for re-review**: per policy, alongside the four entries above,
once a provider is configured in a session that has one.

## 2026-09-26T15:45Z — Claude (session_01RMzxzzsE5JNGEqnnkUs9Yo) — M1, k_from_delta_g negative-uncertainty sign fix (PR sounio-lang/sounio#2694)

| 2026-09-26 | — | math-review | equilibrium.sio (k_from_delta_g's temperature-sensitivity term wrapped in absolute value so combined standard uncertainty can no longer go negative) | WAIVED | No offload provider configured in this container, same as the five entries above. Independent Python cross-check of old vs. new formula; full narrative below. |

**Trigger**: Copilot review found that `k_from_delta_g`'s GUM-style combined
uncertainty, `uk = |k| * (udg/|RT| + (dg/(RT²))*ut)`, kept `dg`'s own sign
on the second (temperature) sensitivity term instead of summing sensitivity
magnitudes. For any spontaneous reaction (`dg < 0`, the common physical
case) with `udg = 0` and `ut > 0`, this returns a negative `uk` — a
standard uncertainty can never be negative, by definition.

**Attempted**: `bin/llm-offload --status` — no provider reachable in this
container, same as every prior entry in this log.

**Outcome**: WAIVED for lack of a reachable provider. Independent
verification instead:

1. **Reproduced the sign bug directly in Python**: `k_from_delta_g(-5000.0,
   0.0, 300.0, 5.0)` returns `uk = -0.248...` under the old body.
2. **Fixed by wrapping the temperature term in absolute value**
   (`equilibrium_abs_f64(dg / (r_const()*t*t)) * ut`), matching the
   already-correct treatment of the `udg` term two operands earlier in the
   same expression. Re-ran the same Python case: `uk = +0.248...`.
3. **Confirmed no change for the sign this file's own tests already
   exercise**: `test_k_dg`/`test_real_delta_g` both pass `dg < 0` through
   `k_from_delta_g` but discard `uk` (bind it to `_`), so they never
   exercised this term's sign either way — not a coincidental prior pass.
   For a positive-`dg` input the fix is a no-op (the term was already
   positive), verified numerically identical in Python.
4. **Regression test added**: `test_k_dg_negative_uncertainty_sign` in
   `equilibrium.sio`, wired into `main()` and into
   `tests/stdlib/chemistry/test_equilibrium_acids.sio`. Verified as a real
   control: temporarily reverted the fix and confirmed
   `test_equilibrium_acids.sio` fails (`FAIL
   k_dg_negative_uncertainty_sign`) under the live suite; restored the fix
   and re-confirmed `PASS`.

**Flagged for re-review**: per policy, alongside the five entries above,
once a provider is configured in a session that has one.
