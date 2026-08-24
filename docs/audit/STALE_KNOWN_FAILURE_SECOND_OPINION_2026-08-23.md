<!-- docs:meta
topic_id: repo.docs.audit.stale-known-failure-second-opinion-2026-08-23
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.stale-known-failure-second-opinion-2026-08-23
-->

# Codex-3 independent verification of 21 stale known-failure markers

Date: 2026-08-24 UTC

Independent worktree: `/tmp/stale-kf-second-opinion`

Audited base: `867a05435865b9feeea92d8f1d2ec913c8d6fdf5`

Compared branch: `fix/stale-known-failure-markers-20260823` at
`4e8e7a3a53f3be0068fcd51e37efc95a9fa6f640`

## Measurement identity

Every local compiler invocation removed the inherited `SOUC_BIN` and
`SOUNIO_SOUC_BIN` variables and explicitly set:

```text
SOUNIO_STDLIB_PATH=/tmp/stale-kf-second-opinion/stdlib
```

This matters because the inherited environment pointed at
`/workspace/sounio/bin/souc` and `/workspace/sounio/stdlib`, which would have
made an isolated-worktree command silently measure the promoted checkout.

Two binaries were measured:

1. **Pre-compiled Madaros**: the worktree-local `./bin/souc` launcher resolving
   to `/tmp/stale-kf-second-opinion/bin/madaros-linux-x86_64`, reporting
   Madaros v0.80.0.
2. **Pre-compiled lean_single**: the checked-in
   `/tmp/stale-kf-second-opinion/bin/souc-linux-x86_64`, invoked through
   `/tmp/stale-kf-second-opinion/scripts/ci/souc-native-wrapper.sh` with
   `SOUNIO_SOUC_RAW_MODE=legacy`.

I did **not** build a compiler from source. Consequently, nothing below is
attributed to an independently measured source build. Where the compared branch
claims a fresh `souc-stage2/lean_single` source build passed, I identify that as
branch evidence rather than my measurement.

The unmodified 21-test filtered suite produced:

- Pre-compiled Madaros: 12 XPAS, 9 XFAIL.
- Pre-compiled lean_single: 19 XPAS, 2 XFAIL.

Scratch mutations lived outside both Git worktrees under
`/tmp/stale-kf-mutants`. I did not edit the compared branch or its test files.

## 1. `parser_card_a_misc_patterns.sio`

**What it really asserts:** The parser and checker accept wildcard patterns,
struct destructuring, and public-field access in a valid program. The computed
sum is ignored, so it does not assert that the extracted runtime values are
correct.

**Baseline engine and binary:** XPAS on pre-compiled Madaros via
`/tmp/stale-kf-second-opinion/bin/madaros-linux-x86_64`; XPAS on pre-compiled
lean_single via `/tmp/stale-kf-second-opinion/bin/souc-linux-x86_64`.

**Exact sabotage and result:** In the scratch copy I changed
`let ParserCardAPair { x, y }` to `let ParserCardAPair { x, z }`. Pre-compiled
Madaros rejected it with E012 (`no field z`) and E137 (`name y` undeclared),
exit code **1**.

**Derived or hardcoded:** Not a numerical claim. Syntax/type acceptance is the
assertion; the unused arithmetic result is derived but unasserted.

**Verdict:** **Marker should go.** The narrow parser/type assertion is live and
the common native-artifact reason is resolved for the CI lean_single surface.

## 2. `parser_card_a_refinement_predicates.sio`

**What it really asserts:** Compound refinement predicates parse, typecheck, and
accept the valid literals used by the program.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed `parser_card_a_bounded(42)` to
`parser_card_a_bounded(142)`. Pre-compiled Madaros rejected the scratch copy
with E042, `value does not satisfy the refinement predicate`, exit code **1**.

**Derived or hardcoded:** The boundary constants are hardcoded; satisfaction is
computed by the checker rather than printed as a constant verdict.

**Verdict:** **Marker should go.** This is not merely a parser smoke: the
predicate is enforced by the measured Madaros binary.

## 3. `pbpk28_struct_return.sio`

**What it really asserts:** Only that `ep28_rapamycin_params()` can be called and
the program exits successfully. It prints `p.cl_central`, but does not compare
that value with an expectation and has no `expect-stdout` annotation.

**Baseline engine and binary:** XFAIL with run exit code **1** on pre-compiled
Madaros `madaros-linux-x86_64`; XPAS with program exit code **0** on
pre-compiled lean_single `souc-linux-x86_64`.

**Exact sabotage and result:** I replaced `print_f64(p.cl_central)` with
`print_f64(999.0)`. Pre-compiled lean_single compiled and ran the scratch copy,
printed `cl=999.000000` followed by `pbpk28_struct_return: PASS`, and exited
**0**.

**Derived or hardcoded:** The original printed `12.4` is derived from the
returned struct. It is not asserted. The final `PASS` string is hardcoded and
unconditional.

**Verdict:** **Marker should stay.** A wrong struct-return value remains green;
the intended numerical assertion is vacuous.

## 4. `rapamycin_iso_budget.sio`

**What it really asserts:** The harness sees the hardcoded token `PASS` and the
token `inf` emitted in a GUM effective-DOF report. It does not require the
Knowledge and Budget64 uncertainty paths to agree, even though it prints a
cross-check message.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I forced both cross-check branches to run and
replaced `r_blood` and `r_brain` with `999.0`. Pre-compiled Madaros printed
`Knowledge / Budget64 ratio out of [0.9,1.1]` for both paths, then printed
`PASS`, and exited **0**.

**Derived or hardcoded:** Concentrations, uncertainty budgets, standard
uncertainties, and `inf` effective DOF are derived from hardcoded model inputs.
The asserted `PASS` token is hardcoded and unconditional. The derived agreement
ratios are not asserted.

**Verdict:** **Marker should stay.** The central scientific cross-check can be
made false without making the test red.

## 5. `rapamycin_rk4_budget.sio`

**What it really asserts:** The first-order variance remains live through the
RK4 calculation: `var_blood_k > 1e-18`, `var_brain_k > 0`, and
`var_periph_k >= 0`. Failure prints `FABRICATED_ZERO` and returns 1 before the
required `FAMILY_A_VAR_LIVE` and `PASS` tokens.

**Baseline engine and binary:** XPAS with exit code **0** on pre-compiled
Madaros `madaros-linux-x86_64`. The checked-in pre-compiled lean_single
`souc-linux-x86_64` produced XFAIL with run exit code **1**. The compared branch
reports a fresh `souc-stage2/lean_single` source-built artifact passing, but I
did not independently build or measure that artifact.

**Exact sabotage and result:** I replaced the calculated `ok_var` predicate with
`false`. Pre-compiled Madaros printed `FABRICATED_ZERO` and
`EPISTEMIC_FABRICATION`, then exited **1**.

**Derived or hardcoded:** The three variances and reported standard
uncertainties are derived from the RK4 and uncertainty calculations. Thresholds
and source measurement uncertainties are hardcoded inputs. `PASS` is hardcoded,
but is reachable only after the derived variance gate succeeds.

**Verdict:** **Marker should go from the lean_single CI manifest if the branch's
fresh source-built stage2 receipt is trusted.** The assertion itself is not
vacuous. The statement must not be generalized to the checked-in lean_single
binary, which failed my measurement.

## 6. `darwin_compartments_coronary_smc_smoke.sio`

**What it really asserts:** It intends to assert routing mass balance, split
ratio, cumulative integration, and terminal blood-state checks across three
regimes. In reality, failures print a message and return from unit `main`, which
exits zero. Its `expect-stdout-contains` annotation is not recognized by the
suite harness.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed the mass-balance check from
`bal_rel > 1.0e-9` to `bal_rel > -1.0`, forcing the failure branch. Pre-compiled
Madaros printed `FAIL: routing mass-balance drift ...` and exited **0**.

**Derived or hardcoded:** The compartment states and balance are derived;
thresholds are hardcoded. Neither the derived verdict nor the intended PASS
token is wired into a recognized harness assertion.

**Verdict:** **Marker should stay.** A broken physical invariant remains green.

## 7. `darwin_pd_coronary_smc_smoke.sio`

**What it really asserts:** It intends to assert the expected activation and
nuclear-state ranges for no-drug, partial-inhibition, and full-inhibition
scenarios plus an LLL sanity check. In reality, every failure returns from unit
`main` with exit zero, and `expect-stdout-contains` is ignored by the harness.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I replaced the first derived predicate with
`let s1_a_ok = false`. Pre-compiled Madaros printed
`FAIL: scenario 1 A not pinned at 1.0` and exited **0**.

**Derived or hardcoded:** Scenario states are derived; acceptable ranges are
hardcoded. The PASS/FAIL text is hardcoded, and neither controls test success.

**Verdict:** **Marker should stay.** The PD assertion is vacuous at the harness
boundary.

## 8. `dissertation_pbpk28_confidence_gate.sio`

**What it really asserts:** On lean_single, four `Knowledge[f64]` priors with
individual confidences down to 0.40 satisfy `with Epistemic(400)` and the gated
body runs. The printed sum is not numerically compared.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed `with Epistemic(400)` to
`with Epistemic(950)`. Pre-compiled lean_single rejected it with four
`EpistemicComplete violation` diagnostics and exit code **1**. The same scratch
copy compiled and ran under pre-compiled Madaros, printed `18.180000` and
`PBPK28_CONFIDENCE_GATE_PASS`, and exited **0**.

**Derived or hardcoded:** The four central values and four confidence values are
hardcoded literature/model inputs. The printed `18.18` is derived by summing the
central values, but is not asserted. Gate satisfiability is computed by the
lean_single checker.

**Verdict:** **Marker should go from the lean_single CI manifest.** The gate is
live there. The Madaros gate is inert and remains a separate compiler defect;
this test must not be cited as a Madaros confidence-gate witness.

## 9. `refinement_f64_return_violation.sio`

**What it really asserts:** A function returning literal `1.7` for the type
`{ p: f64 | p >= 0.0 && p <= 1.0 }` must fail compilation with a diagnostic
containing `refinement type violation`.

**Baseline engine and binary:** Pre-compiled Madaros
`madaros-linux-x86_64` accepted the invalid program, compiler exit code **0**,
so the suite reported XFAIL (`expected compile failure but passed`).
Pre-compiled lean_single `souc-linux-x86_64` rejected it with the expected
diagnostic and compiler exit code **1**, so the known marker appeared as XPAS.

**Exact sabotage and result:** I did not mutate this file. The invalid original
is already the negative witness, and the two measured engines discriminate:
Madaros exit **0**, lean_single exit **1**. The compared branch reports a
separate `1.7 -> 0.5` control, but that is branch evidence, not my sabotage.

**Derived or hardcoded:** The literal and interval bounds are hardcoded; the
refinement decision is computed by the checker.

**Verdict:** **Marker should go from the lean_single-only sidecar manifest.** It
must not be described as resolved in Madaros, where the soundness hole remains.

## 10. `turbofish_concrete_type_mismatch.sio`

**What it really asserts:** `identity::<bool>(42)` must be rejected after
substitution of `T=bool`, with a diagnostic matching the literal pattern
`Type mismatch`.

**Baseline engine and binary:** Pre-compiled Madaros
`madaros-linux-x86_64` rejected the source, compiler exit code **1**, but emitted
`argument type does not match parameter`; the harness therefore reported XFAIL
for missing the exact `Type mismatch` pattern. Pre-compiled lean_single
`souc-linux-x86_64` accepted the invalid program, compiler exit code **0**, so
the compile-fail test was XFAIL for the stated soundness gap.

**Exact sabotage and result:** I did not add a scratch mutation. The original
invalid call is the negative witness and already produces the relevant split:
Madaros exit **1**, lean_single exit **0**.

**Derived or hardcoded:** Not numerical. The concrete type argument and integer
literal are hardcoded; compatibility is a checker decision.

**Verdict:** **Marker should stay.** The checked-in lean_single binary still has
the stated soundness gap, and the fixed-point rebootstrap condition in the
annotation is not independently demonstrated. Madaros also needs diagnostic
wording reconciliation before the current harness expectation passes.

## 11. `test_pipeline_real_e2e.sio`

**What it really asserts:** The PBPK integration produces positive AUC and
finite, nonnegative, route-consistent state, with failures propagated as
nonzero exit codes before `SCIENCE_PBPK_OK`.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed the AUC failure threshold from
`auc <= 1e-12` to `auc <= 1e12`. Pre-compiled Madaros took the failure path and
exited **2**.

**Derived or hardcoded:** AUC and state invariants are derived from the PBPK
integration; tolerance thresholds are hardcoded.

**Verdict:** **Marker should go.** This is a real fail-closed end-to-end
assertion.

## 12. `fo_call_boundary_arity3.sio`

**What it really asserts:** Variance from `Knowledge<f64>` must survive a
three-parameter user-function boundary even when the function returns its first
argument unchanged.

**Baseline engine and binary:** Pre-compiled Madaros
`madaros-linux-x86_64` computed zero variance and the program exited **1**,
reported as XFAIL. Pre-compiled lean_single `souc-linux-x86_64` preserved live
variance and exited **0**, reported as XPAS.

**Exact sabotage and result:** No additional mutation was needed. The original
test is already a discriminating witness: Madaros exit **1**, lean_single exit
**0**.

**Derived or hardcoded:** The variance is derived from the measurement and call
transfer; the `1e-12` liveness threshold is hardcoded.

**Verdict:** **Marker should stay.** Its annotation explicitly names a Madaros
call-transfer defect, and that defect reproduced exactly.

## 13. `fo_call_boundary_neg.sio`

**What it really asserts:** First-order variance must survive `0.0 - x` through
a user-function call.

**Baseline engine and binary:** Pre-compiled Madaros
`madaros-linux-x86_64` produced zero variance and exited **1**, reported as
XFAIL. Pre-compiled lean_single `souc-linux-x86_64` produced live variance and
exited **0**, reported as XPAS.

**Exact sabotage and result:** No scratch mutation was needed. The original is
the engine discriminator: Madaros exit **1**, lean_single exit **0**.

**Derived or hardcoded:** Variance is derived; the liveness threshold is
hardcoded.

**Verdict:** **Marker should stay.** The stated missing Madaros `OpSub` transfer
remains live.

## 14. `test_kaxi_fuse.sio`

**What it really asserts:** It calculates an inverse-variance fused mean near
11.6 and uncertainty near 0.8944, but only prints PASS or FAIL. Unit `main` has
no `expect-stdout` and always exits zero after a completed run.

**Baseline engine and binary:** XFAIL with run exit code **1** on pre-compiled
Madaros `madaros-linux-x86_64`, before a successful verdict-bearing execution.
XPAS with exit code **0** on pre-compiled lean_single `souc-linux-x86_64`.

**Exact sabotage and result:** I changed `if m < 11.5999` to
`if m < 1000.0`, forcing `ok = 0`. Pre-compiled lean_single printed
`kaxi_fuse: FAIL m=11.600000 u=0.894427` and exited **0**.

**Derived or hardcoded:** Mean and uncertainty are derived; expected windows
are hardcoded. The printed verdict is hardcoded branch text and does not control
success.

**Verdict:** **Marker should stay.** A numerically false verdict remains green.

## 15. `test_core_e2e.sio`

**What it really asserts:** Twelve current `stdlib/math/core` operations must
match expected values within tolerance; any mismatch increments `fail` and
causes exit 1.

**Baseline engine and binary:** XFAIL with run exit code **1** on pre-compiled
Madaros `madaros-linux-x86_64`; XPAS with exit code **0** on pre-compiled
lean_single `souc-linux-x86_64`.

**Exact sabotage and result:** I changed the expected result of `sqrt(4.0)` from
`2.0` to `3.0`. Pre-compiled lean_single compiled and ran the scratch copy and
exited **1**.

**Derived or hardcoded:** Function results are derived; reference values and
tolerances are hardcoded.

**Verdict:** **Marker should go from the lean_single CI sidecar.** The test is
non-vacuous. The compared branch's claim that it also passes default Madaros
disagrees with my sanitized pre-compiled Madaros result.

## 16. `test_hyper_math_e2e.sio`

**What it really asserts:** The chosen sedenion pair has product norm below the
zero threshold, causing `pair_zero` to be true and allowing
`HYPER_MATH_OK`. The octonion-real and zero-divisor-classification values are
emitted only as metrics and do not control the exit code.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed
`pair_zero = sed_norm_sq(pair) < 0.000001` to use the impossible bound `< -1.0`.
Pre-compiled Madaros emitted `sed_pair_zero_flag 0.000000` and exited **2**.

**Derived or hardcoded:** The product and its norm are derived; the selected
pair and tolerance are hardcoded. Other printed metrics are derived but
unasserted.

**Verdict:** **Marker should go for the narrow zero-product assertion.** It must
not be described as broad validation of every emitted hypercomplex metric.

## 17. `test_distributions_e2e.sio`

**What it really asserts:** Seven analytic distribution means or variances must
match expected values within tolerance; any mismatch increments the failure
count and returns 1.

**Baseline engine and binary:** XFAIL with run exit code **1** on pre-compiled
Madaros `madaros-linux-x86_64`; XPAS with exit code **0** on pre-compiled
lean_single `souc-linux-x86_64`.

**Exact sabotage and result:** I changed the expected normal mean from `5.0` to
`6.0`. Pre-compiled lean_single printed six PASS results and
`Test 3 ... FAIL`, then exited **1**.

**Derived or hardcoded:** Distribution moments are derived by the stdlib;
analytic references and tolerances are hardcoded.

**Verdict:** **Marker should go from the lean_single CI sidecar.** The test is
non-vacuous. The compared branch's statement that it passes default Madaros
disagrees with my sanitized pre-compiled Madaros result.

## 18. `test_log_path_cmp.sio`

**What it really asserts:** A large set of log-level, buffer, path, comparison,
min/max, clamp, approximate-equality, and sign contracts must all match exact
expectations; each failure has a distinct nonzero return code.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed the first expectation from
`LOG_TRACE() != 0` to `LOG_TRACE() != 999`. Pre-compiled Madaros exited
**101**.

**Derived or hardcoded:** Library results are derived; expected constants are
hardcoded contract values.

**Verdict:** **Marker should go.** The test is strongly fail-closed.

## 19. `gum_fo_across_call.sio`

**What it really asserts:** First-order GUM variance must survive a user-function
call; derived variance must exceed `1e-12`, otherwise the program prints an
epistemic-fabrication warning and exits 1.

**Baseline engine and binary:** Pre-compiled Madaros
`madaros-linux-x86_64` produced zero call variance and exited **1**, reported as
XFAIL. Pre-compiled lean_single `souc-linux-x86_64` produced live variance and
exited **0**, reported as XPAS.

**Exact sabotage and result:** No extra mutation was needed. The original test
is already the discriminating witness: Madaros exit **1**, lean_single exit
**0**.

**Derived or hardcoded:** Variance is derived; the liveness threshold is
hardcoded.

**Verdict:** **Marker should stay.** The stated Madaros Family A call-boundary
loss remains reproduced.

## 20. `associator_field_octonion.sio`

**What it really asserts:** All Fano and non-Fano associator-field invariants
must satisfy their expected values so that the final output contains the exact
required token `ALL PASS`.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed the expected non-Fano norm squared from
`4.0` to `5.0`. Pre-compiled Madaros ran the program and exited **0**, but
printed `W5 nonfano norm_sq=4: FAIL` and ended with `SOME FAIL`. Because the
test has `//@ expect-stdout: ALL PASS`, the harness would classify this as red
despite the process exit code 0.

**Derived or hardcoded:** Associator invariants and augmented variances are
derived; expected reference values are hardcoded.

**Verdict:** **Marker should go.** The recognized exact-output assertion makes
the test non-vacuous.

## 21. `knowledge_array.sio`

**What it really asserts:** It intends to assert that three `Knowledge` values
survive array indexing, field access, struct return, and a stack-stomping call.
In reality it only prints PASS or FAIL; unit `main` has no recognized output
expectation and exits zero either way.

**Baseline engine and binary:** XPAS on pre-compiled Madaros
`madaros-linux-x86_64`; XPAS on pre-compiled lean_single
`souc-linux-x86_64`.

**Exact sabotage and result:** I changed the first expected value from
`k0.value == 12.4` to `k0.value == -999.0`. Pre-compiled Madaros printed
`KNOWLEDGE_ARRAY_FAIL` and exited **0**.

**Derived or hardcoded:** Array values are derived from the called constructors
and returns; expected numbers and PASS/FAIL strings are hardcoded. The numeric
comparison does not control process or harness success.

**Verdict:** **Marker should stay.** The intended value-preservation assertion
is vacuous.

## Agreement with `fix/stale-known-failure-markers-20260823`

I agree with the branch's **final marker decisions**:

- Remove 11 sidecar entries:
  `parser_card_a_misc_patterns`, `parser_card_a_refinement_predicates`,
  `rapamycin_rk4_budget`, `dissertation_pbpk28_confidence_gate`,
  `refinement_f64_return_violation`, `test_pipeline_real_e2e`,
  `test_core_e2e`, `test_hyper_math_e2e`, `test_distributions_e2e`,
  `test_log_path_cmp`, and `associator_field_octonion`.
- Retain the six vacuous sidecar entries:
  `pbpk28_struct_return`, `rapamycin_iso_budget`,
  `darwin_compartments_coronary_smc_smoke`,
  `darwin_pd_coronary_smc_smoke`, `test_kaxi_fuse`, and
  `knowledge_array`.
- Leave the four inline engine-specific markers in place:
  `turbofish_concrete_type_mismatch`, `fo_call_boundary_arity3`,
  `fo_call_boundary_neg`, and `gum_fo_across_call`.

That is 11 removals and 10 retained markers.

## Disagreement with `fix/stale-known-failure-markers-20260823`

I disagree with several **engine-attribution claims in the commit prose**, not
with the resulting file diff:

1. Commit `683c213821` says all eight retired full-suite tests pass on both
   `souc-stage2/lean_single` and default Madaros. My sanitized measurement with
   the worktree-local pre-compiled Madaros found `test_core_e2e` and
   `test_distributions_e2e` still exiting 1. The removal remains correct for the
   lean_single CI sidecar, but `passes on both` is not supported by my Madaros
   measurement.
2. Commit `caf4b71d49` says `rapamycin_rk4_budget` passes on both engines. My
   pre-compiled Madaros passed it, but the checked-in pre-compiled lean_single
   exited 1. A fresh source-built `souc-stage2` may differ, and the branch says
   it does, but that source-build result is not interchangeable with every
   lean_single binary.
3. The branch correctly states that `dissertation_pbpk28_confidence_gate` is
   enforced by lean_single and inert in Madaros. My direct `Epistemic(950)`
   control independently confirms that exact split.
4. Commit `9fef6e0cd9` correctly limits the f64 return-refinement closure to the
   lean_single manifest and explicitly records the continuing Madaros defect.
5. Commit `4e8e7a3a53` correctly retains the six vacuous tests. I independently
   reproduced vacuity for all six, but not always on both engines: the sabotage
   binary is stated separately in each section above. Claims of a mutation
   passing on both engines exceed my own measurements where I only ran the
   relevant scratch control on one binary.

The practical correction is therefore simple: keep the branch's 11/10 marker
decision, but make every supporting statement engine-specific and distinguish a
fresh source-built `souc-stage2` artifact from the checked-in lean_single binary
and from pre-compiled Madaros.

---

## Addendum by claude-1, after this report was written

Point 3 above needs correcting, and it is the most consequential line in the
document.

It says the branch "correctly states that `dissertation_pbpk28_confidence_gate`
is enforced by lean_single and inert in Madaros", and confirms that split with a
direct `Epistemic(950)` control. Both measurements were made with the
**pre-compiled** Madaros, as this report's own Measurement Identity section
states plainly.

Measured afterwards on a Madaros **built from origin/main source**, on Slurm:

    error[E215] in compile-fail/dissertation_pbpk28_overclaim::rapa_overclaimed_dose:
        EpistemicComplete violation

matching that fixture's own `error-pattern` exactly. A positional-argument
equivalent is refused too. **Madaros enforces the floor.** What did not was the
committed ELF at `e5032371a9` (md5 `518006cc413e`); `d97181caa6` later shipped a
rebuilt one (md5 `8923e638d025`) that refuses.

So the "enforced by lean_single, inert in Madaros" split was never a property of
the two engines. It was the age of one binary.

Three of us reached that wrong conclusion independently — the branch author,
this report, and the orchestrator relaying both — and all three had carefully
sanitised `SOUC_BIN` and `SOUNIO_STDLIB_PATH` first. Clearing the environment
disarms one trap. The committed binary is a second one, and a clean environment
running a stale binary answers yesterday with every appearance of rigour.

This report is the reason that was caught: it is the only one of the three that
wrote down which binary produced each number. Everything else here stands.

Two things came out of it:

- `scripts/ci/madaros_binary_source_drift_gate.sh` asks, per capability, whether
  the shipped binary does what the source implements.
- `souc --version` now prints the ELF path, its md5 and the tree SHA on stderr,
  so the condition travels with every log without anyone having to remember it.
