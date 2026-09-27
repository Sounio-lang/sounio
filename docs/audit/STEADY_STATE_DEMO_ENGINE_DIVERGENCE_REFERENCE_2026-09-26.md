<!-- docs:meta
topic_id: repo.docs.audit.steady-state-demo-engine-divergence-reference-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.steady-state-demo-engine-divergence-reference-2026-09-26
-->

# Steady-state demo: engine divergence versus an independent reference (2026-09-26)

**Status:** diagnosed. No compiler defect in Madaros. No `self-hosted/` or
`stdlib/` change is made by this dispatch.
**Lane:** `claude(gracious-bardeen)`. This builds on the literal-rounding finding
of `claude(confident-kilby)` (commit `2a6d0819a`, branch
`claude/confident-kilby-8d4142`, section *Steady-state divergence: root cause*
of `docs/audit/LEAN_SINGLE_IMPORTED_TYPE_ERROR_FAIL_OPEN_2026-09-26.md`), which
was re-measured here independently rather than cited.

## Question

`examples/dissertation_steady_state_demo.sio` and
`examples/dissertation_steady_state_fullvd_demo.sio`, both driven by
`stdlib/darwin_pbpk/scenarios/steady_state_runner.sio`, print different last
digits under the two engines. For the fullvd demo, `AUC_last / AUC_first` is
1.228366 under Madaros and 1.228345 under lean_single, a gap of 2.1e-5. Which
engine is numerically correct, and is there a miscompile?

## Answer

1. **Neither value is right past the fourth significant figure.** An
   independent solve of the same model converges to **1.227702** for the same
   16-checkpoint quantity. Madaros is off by +6.65e-4 and lean_single by
   +6.44e-4 (5.4e-4 and 5.2e-4 relative). All three round to 1.228 at four
   significant figures and disagree at the fifth (1.2284 and 1.2283 against
   1.2277). The engine gap is about 31 times smaller than either engine's
   discretization error.
2. **No floating-point operation is miscompiled; the gap is a literal-rounding
   defect in lean_single, amplified.** The evidence is Measurement 5: with
   exact literals the runs are byte-identical. So the gap is exactly
   lean_single's 1–2 ulp misrounding of 14 decimal literals (`rtol`, `atol`,
   Tsit5 tableau entries), amplified by the adaptive step controller's
   discrete accept/reject decisions. Measurement 4 is consistent with that
   amplification but does not by itself prove the cause: on Madaros alone,
   raising `rtol` by exactly 1 ulp moves the ratio by 9.6e-6, within a factor
   of about 2 of the engine gap.
3. **Madaros is the faithful execution of the source as written.** With the 14
   misrounded literals replaced by exact dyadic expressions, lean_single
   reproduces Madaros byte for byte on both demos. Under the operator
   directive, Madaros is the only PBPK compiler, so its values stand as the
   source's values. The source's values carry the discretization error in (1).
4. **No dissertation figure moves because of the engine choice.** The only
   consumer is the M6 suite gate (`scripts/ci/dissertation_pbpk_suite_gate.sh`),
   which checks the `SS DEMO OK` / `SS FULLVD OK` markers, not the numbers. No
   file under `docs/dissertation/` quotes any of the affected values.

## Setup

| Item | Value |
|---|---|
| Base | `main` 2e8b76d31 |
| Madaros | `make build-madaros` from that source, bare (content-cache hit on key `4784102d…`), `artifacts/self-hosted/madaros` md5 `5764851f`, invoked through `./bin/souc` |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc`, run from the worktree root (it resolves `stdlib/` relative to the CWD) |
| Env | `SOUNIO_STDLIB_PATH` pinned to the worktree's `stdlib/`; `SOUC_BIN`, `SOUNIO_SOUC_BIN`, `MADAROS_RAW_BIN`, `SOUNIO_MADAROS_BIN` unset |
| Worktree | `/workspace/worktrees/gracious-bardeen-ss` on sounio-workspace |

**On `main` the demos do not compile under Madaros.** The errors are E259 on
`SteadyStateReport` / `SSIntervalResult` / `OralBBBTrace` fields read across
modules, E137 on `print_i64` (undeclared; the builtin is `print_int`), and
E008 at the `ssr_run_one_interval` bailout, which returns `OralBBBTrace` where
`SSIntervalResult` is declared. lean_single runs them anyway (see the
fail-open dispatch above). To obtain Madaros numbers, the three stdlib-only
commits of `claude/confident-kilby-8d4142` were cherry-picked (`-x`) into the
measurement worktree: `cc292c77b` (print_int), `3bf0c926c` (pub result-record
fields), and `0ee5cd57e` (bailout returns `SSIntervalResult`). They touch no
`self-hosted/` file, so one Madaros build serves both trees. They are not part
of this dispatch's commit.

## Measurement 1: reproduction

With the cherry-picks, `./bin/souc run` on each demo gives:

| Printed value | Madaros | lean_single |
|---|---:|---:|
| fullvd `AUC_last / AUC_first` | 1.228366 | 1.228345 |
| fullvd `C_max_last / C_max_first` | 1.063169 | 1.063101 |
| fullvd per-dose row 6, C_max | 0.003727 | 0.003726 |
| 7-dose demo per-dose row 4, C_max | 0.005650 | 0.005649 |

Every other printed line is identical.

## Measurement 2: independent reference

`docs/audit/repro/ss_multidose_rk4_reference.sio` (fullvd, 14 doses) and
`docs/audit/repro/ss_multidose_rk4_reference_mean.sio` (mean params, 7 doses)
use the **same model**
(`pbpk_ode`, `rapamycin_fullvd_params` / `rapamycin_mean_params`,
`rapamycin_absorption_params`, 2 mg q24h, per-interval lag of 0.5 h). The
numerics are **different**:

- classical RK4 at a fixed step h = 0.1 / SUB hours;
- the gut depot is integrated as a state (`a' = -ka·a` after the lag);
- the depot feeds blood as a continuous rate `F·ka·a / V_blood`, so there is
  no operator splitting and no adaptive control;
- the lag (0.5 h) and the checkpoints (24/15 = 1.6 h) fall exactly on step
  boundaries.

It reports the demo's quantity: fu × trapezoid over the same 16 checkpoints,
and C_max over the same checkpoints. It also reports a fine-grid AUC: a
trapezoid over every RK4 step. That is not exact; halving h moves the fullvd ratio by 3e-7, an estimate of its
quadrature error.

The source's scheme is a first-order Lie–Trotter splitting of
y′ = A·y + b·u(t). It adds the mass the depot releases over [t, t+dt] as a
bolus at t, then steps the homogeneous ODE. For bounded u the local defect
against the Duhamel solution is O(dt²), so the scheme is consistent and its
dt → 0 limit is the continuous-input solution the reference computes. (This
does not rely on linearity in dose.) The largest local exchange rate is
brain, Q_brain/(V_brain·Kp_brain) = 50/(1.4·0.08) ≈ 446 h⁻¹. That was not
shown to be the spectral radius of A, so RK4 stability at the chosen steps
rests on the empirical check: SUB = 40 (h = 2.5e-3 h) and SUB = 80 agree to
1e-12 on the 16-point quantities.

| Quantity (fullvd, 14 doses) | SUB = 40 | SUB = 80 | Madaros = lean_single? |
|---|---:|---:|---|
| auc16 ratio | 1.227701683257 | 1.227701683257 | identical, all printed digits |
| fine-grid-AUC ratio | 1.218199363835 | 1.218199066331 | identical |
| C_max ratio | 1.062768049768 | 1.062768049768 | identical |

| Quantity (mean params, 7 doses) | SUB = 40 | SUB = 80 |
|---|---:|---:|
| auc16 ratio | 4.788650990858 | 4.788650990857 |
| fine-grid-AUC ratio | 4.762119492696 | 4.762116943166 |
| C_max ratio | 4.162524012230 | 4.162524012223 |

With a fixed step and no accept/reject branching, the two engines agree on
every printed digit. The same `pbpk_ode` arithmetic runs in both. A 1-ulp
literal difference stays a 1-ulp difference.

## Measurement 3: the source algorithm under tightened tolerance

`docs/audit/repro/ss_multidose_source_tol_probe.sio` is the runner's plasma
loop with `rtol` and `atol` exposed. It keeps the operator-split bolus, the
adaptive Tsit5 step, the `optimal_step_14` controller and the 16-checkpoint
trapezoid. BBB is dropped because it is one-way coupled and does not feed
plasma. At the default tolerance it reproduces the demo exactly (Madaros
1.228366492, lean_single 1.228345205). Madaros exhausts the handle table
(rc=182, owned by `claude(sleepy-easley)`) at rtol ≤ 1e-8, so the sweep is on
lean_single. Its literal misrounding shifts the ratio by about 2e-5
(Measurement 1), small against the errors in this table.

| rtol / atol | auc16 ratio (fullvd) | accepted steps | error vs reference | error × steps |
|---|---:|---:|---:|---:|
| 1e-6 / 1e-10 | 1.228345 (Madaros 1.228366) | 58,562 | +6.44e-4 | 37.7 |
| 1e-8 / 1e-12 | 1.228004 | 96,423 | +3.02e-4 | 29.1 |
| 1e-10 / 1e-14 | 1.227829 | 197,173 | +1.28e-4 | 25.2 |
| 1e-12 / 1e-16 | 1.227754 | 455,345 | +5.26e-5 | 23.9 |

The error times the step count falls from 37.7 and levels off near 24. The
last three rows still drift by about 18%, so pure C/N is only approximate.
The error is therefore **first order in step size**. That is the signature
of the operator-split bolus, not of the fifth-order Tsit5 truncation. A Richardson extrapolation on
the last two rows, assuming error ∝ 1/N, gives 1.2276969, against the
reference 1.2277017. For mean params the same extrapolation gives 4.788639,
against 4.788651. The source scheme converges to the reference, which
confirms the reference.

## Measurement 4: 1-ulp sensitivity on Madaros alone

Same probe, Madaros, with
`rtol = 0.000001 * (1.0 + 2.220446049250313e-16)`, exactly one ulp above the
correctly rounded `1e-6` (see the note at the end of this section).

| | rtol = 1e-6 | rtol = 1e-6 + 1 ulp |
|---|---:|---:|
| fullvd auc16 ratio | 1.228366492 | 1.228376120 |
| fullvd C_max ratio | 1.063168581 | 1.063182284 |
| accepted / rejected steps | 58,552 / 1,790 | 58,559 / 1,826 |
| mean auc16 ratio | 4.791231660 | 4.791240244 |

A 1-ulp change in one constant, on one engine, moves the ratio by 9.6e-6 and
the C_max ratio by 1.4e-5, within a factor of about 2 of the engine gap.  The
dependence cannot be smooth. Measurement 3 gives a trend of
dR/d(ln rtol) ≈ 3.4e-4 / ln(100) ≈ 7.4e-5, so a 1-ulp relative change
(2.2e-16) would move R by about 1.6e-20 on that trend. The observed 9.6e-6
is about 1e15 times larger, so a discrete branch is needed. The run has one: the
controller's accept/reject test `err_norm <= 1.0`, which the I-controller
keeps `err_norm` close to by design. The rejected-step count moves from
1,790 to 1,826 and the accepted count from 58,552 to 58,559. A different step
sequence then carries a different O(dt) splitting error, as Measurement 3
shows. This cascade is generic to the controller. An earlier draft
attributed it specifically to steps pinned at the explicit stability
boundary; that attribution is untested and is withdrawn (the two
reviewers split on it; see the math-review section). The mean step at
rtol = 1e-6 is 336 h / 58,552 ≈ 5.7e-3 h, and the step count keeps rising
as rtol tightens, so accuracy control is active. The step sequences were not
traced.

(The +1 ulp bits, 4517329193108106638 against 4517329193108106637, were
computed in IEEE-754 double arithmetic outside Sounio. Madaros parses both
decimal literals in the expression exactly, per the minimal repro below and
the 115-literal census in the confident-kilby dispatch.)

## Measurement 5: exact-literal causal test (independent replication)

The 14 literals that confident-kilby found misrounded by lean_single were
replaced by exact dyadic expressions `((M as f64) / (2^k as f64) / …)`, with M
< 2^53 and every divisor a power of two. This was done in a scratch copy of
the whole worktree `stdlib/` and in the two demos, with a substitution script
that matches whole tokens only. lean_single was run from the copy's root.

| Comparison | 7-dose demo | fullvd demo |
|---|---|---|
| lean_single (exact literals) vs Madaros (original) | **identical** | **identical** |
| Madaros (exact literals) vs Madaros (original) | identical | identical |
| lean_single (exact literals) vs lean_single (original) | differs | differs |

The third row shows lean_single did read the copy.

## Minimal repro (principle 12)

`docs/audit/repro/lean_single_float_literal_bits.sio`, 18 lines. It prints
`f64_to_bits` of six literals from this closure against their IEEE-754
round-to-nearest bits:

| Literal | Role | lean_single ulp error | Madaros ulp error |
|---|---|---:|---:|
| `0.000001` | `rtol` | +1 | 0 |
| `0.0000000001` | `atol`, `dt_min` | +2 | 0 |
| `0.041` | `v_vasc` | −1 | 0 |
| `1.379008574103742` | Tsit5 `a74`/`b4` | +1 | 0 |
| `0.9800255409045097` | Tsit5 `c5` | −1 | 0 |
| `1.0e-12` | AUC guard | +1 | 0 |

Madaros prints `LITERALS_EXACT`, lean_single `LITERALS_MISROUNDED`. The
amplification from ulps to 1e-5 is Measurement 4.

## Candidates ruled out

- **f64 miscompile (FMA contraction, operand order, i64↔f64 casts).** In
  Measurement 2 the same `pbpk_ode` arithmetic runs about 540,000 times (fullvd,
  SUB = 40), and the engines agree to every printed digit. In Measurement 5, correcting
  literals alone makes the adaptive run byte-identical. No residual arithmetic
  difference exists.
- **Loop-bound or iteration-count difference.** The accepted and rejected
  step counts do differ (58,552/1,790 against 58,562/1,840), but only as a
  consequence of the constants. With exact literals the outputs are
  identical, so no loop executes a different count for any other reason. The
  outer loops (16 checkpoints, n doses) are integer-bounded and identical.
- **Madaros struct-field aliasing (`let a0 = s.a; s = f(s)`).** lean_single
  copies correctly (hazard posted 2026-09-26 by `claude(elegant-borg)`;
  repro `/workspace/.wt/claude-vfx-probe/repro_alias.sio`). It matches
  Madaros byte for byte once literals are exact. So no Madaros-only aliasing
  changes a value on this path. The runner's aggregate moves
  (`sys_st = result.state_new`, `abs_st = abs_next`, `sys_st = res.sys_st`)
  alias only dead temporaries.

## Figures: what moves, and by how much

**Engine choice, Madaros against lean_single:** only the four printed values
in Measurement 1 move, by 1 in the sixth decimal for the per-dose C_max
values, 6.8e-5 for the C_max ratio and 2.1e-5 for the AUC ratio. No
dissertation document quotes them. The M6 gate is marker-only and passes on
both engines. **No dissertation figure moves.**

**Discretization, as printed against the converged reference.** This is not
caused by either engine and is not fixed here. The fullvd demo reports its SS
row at 0-based dose index 7, which is 1-based table row 8:

| fullvd value | printed (Madaros) | reference, same 16-pt quantity | reference, fine-grid AUC |
|---|---:|---:|---:|
| `C_max_ss` (row 8) | 0.003731 mg/L | 0.003754 (+0.63%) | — |
| `C_trough_ss` (row 8) | 0.000234 mg/L | 0.000234 (+4.6e-5 rel.) | — |
| `AUC_tau_ss_u` (row 8) | 0.001672 mg·h/L | 0.001676 (+0.25%) | 0.001735 (+3.8%) |
| `C_max_last / C_max_first` | 1.063169 | 1.062768 | — |
| `AUC_last / AUC_first` | 1.228366 | 1.227702 | 1.218199 |
| SS dose index, `t_to_90pct_h` | 7, 48 | 7, 48 (unchanged; see adjacent finding 6) | — |

| 7-dose demo, row 7 | printed (Madaros) | reference, 16-pt | reference, fine-grid AUC |
|---|---:|---:|---:|
| C_max | 0.008062 | 0.008077 (+0.19%) | — |
| C_trough | 0.006800 | 0.006800 | — |
| AUC_tau (unbound) | 0.013998 | 0.014001 (+0.02%) | 0.014021 (+0.16%) |

## Adjacent findings (reported, not fixed; outside this dispatch)

1. **Operator-split absorption is first order.** `ssr_run_one_interval` adds
   each step's released mass to blood as a bolus before the Tsit5 step. The
   printed values therefore carry an O(dt) error that tolerance cannot cure:
   the Tsit5 error estimate never sees it. It is −0.63% on fullvd C_max. The
   fix would feed `F·ka·a(t)/V_blood` into the RHS, or integrate the depot as
   a 15th state. That changes the printed numbers, so it is an operator
   decision.
2. **The 16-checkpoint trapezoid is biased low.** Measured from the reference
   run (16-point against every-step trapezoid): `AUC_tau` is 3.4–4.1% below the fine-grid interval
   AUC for fullvd (4.1% at dose 1, 3.4% at dose 14), and 0.15–0.7% below for
   mean params. Only the dose-to-dose difference in that bias moves the ratio,
   from 1.2182 (fine grid) to 1.2277. The spacing (1.6 h) exceeds 1/ka ≈ 0.91 h, so
   under-resolution of the absorption peak is the plausible mechanism, but it
   was not isolated. `C_max` is likewise a checkpoint maximum, not the true
   peak.
3. **SS dose label is 0-based.** `SS reached @ dose #7` is `dose_i = 7`, but
   the per-dose table is 1-based. The reported SS values are those of table
   row 8. Measured: `AUC_tau_ss_u` 0.001672 equals row 8, and row 7 is
   0.001670.
4. **lean_single literal conversion.** The fix was already proposed in the
   confident-kilby dispatch: emit correctly rounded bits for float literals
   and for the `const` path. It is not applied because lean_single is the
   bootstrap seed.
5. **Madaros handle exhaustion** (rc=182) blocks tighter-tolerance runs of
   this runner on Madaros. This belongs to `claude(sleepy-easley)`.
6. **`t_to_90pct_h` is tautological.** `run_oral_multidose` sets
   `cap = grew + 1e-12` and tests `grew >= 0.9 * cap`. That holds at the second
   dose whenever AUC grew by more than about 1e-11, so the field is always
   2·τ (48 h at q24h) regardless of the kinetics. Both demos print 48. Read
   from source at `steady_state_runner.sio:285-290`; no run isolates it
   further. No `docs/dissertation/` file quotes it.

## Math-review checkpoint (CLAUDE.md §10)

Input: the claims C1–C5 of this document, sent as
`/tmp/gb/math_review_ss_divergence.md` via `bin/llm-offload -t math-review`
(tooling from `chore/llm-offload-llmgateway-grok47` 754c303cb, default fan-out
`xai kimi zai local`). The outcome is recorded in the section below. It is also in the local
`.claude/llm_offload_log.md`, which is gitignored on `main`.

Run 2026-09-26 on sounio-workspace. The first default fan-out returned
nothing usable. Z.AI was rate-limited (code 1313), the local model was down,
Kimi K3 spent all 8,192 tokens on reasoning (`finish_reason: length`, empty
content), and Grok 4.7 timed out at 180 s on both the gateway and direct legs.
A rerun of the `xai` leg with `OFFLOAD_TIMEOUT=900` completed on Grok 4.7
through the LLM Gateway (raw JSON `/tmp/llm-offload-fHcqUq/`). Verdicts and
what changed in this document:

| Claim | Grok 4.7 verdict | Action taken |
|---|---|---|
| C1: split limit is the continuous solution, RK4 is the reference | OVERREACH: linearity in dose is irrelevant. The scheme is Lie–Trotter with an O(dt²) local defect, so it is consistent for any Lipschitz RHS. | Premise replaced by the consistency argument (Measurement 2). |
| C2: first-order splitting error, limit 1.22770 ± 1e-5 | OK. Recomputed err × N = 29.12, 25.18, 23.94; Richardson L = 1.2276969, \|L − R₁₆\| = 4.8e-6. Noted the ~18% drift. | Drift noted in Measurement 3. |
| C3a: neither value correct to 4 s.f., +6.4e-4 to +6.6e-4 | TIGHTENABLE. Magnitudes confirmed (6.65e-4, 6.44e-4). All three round to 1.228, so the failure is at the fifth figure. | Answer §1 reworded. |
| C3b: gap is controller sensitivity, not a miscompile | OVERREACH. The size comparison does not identify the cause; Measurement 5 does. A smooth sensitivity cannot turn 1–2 ulp into 2e-5, so a discrete branch is required. | Answer §2 now rests on Measurement 5; Measurement 4 is cited as consistency only. |
| C4: 16-point trapezoid bias 3.4–4.1% from the absorption peak | OVERREACH on mechanism. The R shift (+9.50e-3) is verified. The percentages are consistent. The peak mechanism is plausible, not proved. | Adjacent finding 2 now says measured bias, plausible mechanism, not isolated. |
| C5: chatter at the explicit stability boundary | OVERREACH. 446 h⁻¹ is a local rate, not shown to be ρ(A). The mean step is 5.7e-3 h and N rises with tighter rtol, so accuracy control is active. The accept/reject cascade is generic to the I-controller. Hairer–Wanner II §IV.2 is the wrong citation. | Stability-boundary attribution withdrawn; citation removed; RK4 stability rests on the h/(h/2) check. |

No claim was judged wrong in its numbers. Every correction narrowed an
inference.

**Second vendor, Kimi K3 (LLM Gateway).** Three attempts gave no formal
verdict. At 8,192 and 20,000 max tokens the model spent the whole budget on
reasoning (`finish_reason: length`, empty content; last run
`/tmp/llm-offload-pCEuLu/kimi.json`). At 32,000 the upstream provider
failed (`fetch_failed`). Its 62 kB reasoning trace was read as an
independent, **informal** second opinion. Treat it as such, not as a review
verdict:

- C1, C2, C4: agrees. It independently found that the error scales as
  rtol^0.19 ≈ rtol^(1/5). That is what first-order splitting error predicts
  when an I-controlled fifth-order method takes N ∝ rtol^(−1/5) steps. It
  also checked that the 4.1% / 3.35% bias pair reproduces the R shift
  (1.2182 → 1.2277).
- C3: agrees on the magnitudes. Like Grok, it flagged that "not an
  arithmetic miscompile" misleads, because the literals are miscompiled.
  Answer §2 was reworded to say so.
- C5: **disagrees with Grok.** Kimi reads the rtol = 1e-6 runs as
  stability-limited: mean step 5.7e-3 h against a Tsit5 real-axis limit of
  about 3.3/446 ≈ 7.4e-3 h. It holds that Hairer–Wanner II §IV.2 is the
  apt citation. Grok reads the same data as accuracy-limited and the
  citation as wrong. With the reviewers split and no eigenvalue computation
  or step trace, the stability-boundary mechanism stays **unresolved**, as
  Measurement 4 now says. The conclusions do not depend on it.
- Alternative Kimi raised: interpolation at checkpoints as a second
  amplifier. Excluded by construction: the runner clamps `dt_use` to land
  on each checkpoint (`if t + dt_sys > target { target - t }`), so every
  checkpoint is a step node and no dense output is used.

## Reproduce

```bash
# measurement worktree: main + cc292c77b 3bf0c926c 0ee5cd57e (cherry-picked)
make build-madaros
export SOUNIO_STDLIB_PATH=$PWD/stdlib
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_RAW_BIN SOUNIO_MADAROS_BIN
./bin/souc run examples/dissertation_steady_state_fullvd_demo.sio
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run examples/dissertation_steady_state_fullvd_demo.sio
./bin/souc run docs/audit/repro/ss_multidose_rk4_reference.sio         # 1.227701683257
./bin/souc run docs/audit/repro/ss_multidose_rk4_reference_mean.sio    # 4.788650990858
./bin/souc run docs/audit/repro/ss_multidose_source_tol_probe.sio      # 1.228366492144
./bin/souc run docs/audit/repro/lean_single_float_literal_bits.sio     # LITERALS_EXACT
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run docs/audit/repro/lean_single_float_literal_bits.sio  # LITERALS_MISROUNDED
```

The three probes and the literal repro need only `main`'s stdlib, not the
cherry-picks.
