<!-- docs:meta
topic_id: repo.docs.audit.darwin-pbpk-local-exp-ln-accuracy-dispatch-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.darwin-pbpk-local-exp-ln-accuracy-dispatch-2026-09-26
-->

# darwin_pbpk local exp/ln helpers are inaccurate — dispatch

**Date:** 2026-09-26
**Base:** `origin/main` @ `ce93ea9534` (remote worktree `/workspace/worktrees/claude-transc-helpers`, detached)
**Engine:** Madaros, md5 `5764851f3d229372e26aac1e951c95e1`, invoked through `bin/souc` with
`SOUNIO_MADAROS_BIN` pinned to it. The ELF was copied from the workspace worktree
`ab-alias-base` (HEAD `2e8b76d312`). The `self-hosted/` tree is `2a32990c94` at `2e8b76d312`,
`98315edcdb` and `ce93ea9534`, so this is a Madaros built from the source under test.
`PBPK28_CN_FLOOR_CLAMP_MASS_INJECTION_DISPATCH_2026-09-26.md` records a separate
`make build-madaros` of `98315edcdb` that produced the same md5, so the build reproduces.
`SOUNIO_STDLIB_PATH` pointed at the worktree's `stdlib/`. `SOUC_BIN`, `SOUNIO_SOUC_BIN` and
`SOUNIO_SOUC_ENGINE` were unset. lean_single was **not** used (operator directive 2026-09-26:
Madaros is the only PBPK compiler).
**Scope:** `release/biomaterial_release.sio`, `validation/{pbpk28_rapamycin_clinical,
pbpk28_semaglutide_clinical, rapamycin_clinical, tacrolimus_oral_pbpk}.sio`.
**Status:** evidence recorded; fix **proposed and measured, not applied**. This dispatch changes
no file under `stdlib/` or `self-hosted/`. The proposed diff is
[`repro/darwin_pbpk_local_exp_ln_swap.patch`](repro/darwin_pbpk_local_exp_ln_swap.patch).
**Precedent:** the same defect in `release/matrix_er.sio`
(`MATRIX_ER_TRANSCENDENTAL_ACCURACY_DISPATCH_2026-09-26.md`, branch `claude/bold-robinson-ab52d2`,
in progress and not on `main` at the time of writing).

## Summary

Five darwin_pbpk files carry private copies of two transcendental helpers instead of calling
`stdlib/math/pure.sio`:

| helper | files | algorithm |
|---|---|---|
| `rel_exp`, `pv_exp`, `sv_exp`, `cv_exp`, inline GMFE block | biomaterial_release, pbpk28_rapamycin_clinical, pbpk28_semaglutide_clinical, rapamycin_clinical, tacrolimus_oral_pbpk:357 | `(1 + x/1024)^1024` by ten squarings |
| `pv_ln`, `sv_ln`, `cv_ln`, `tco_ln` | the four validation files | divide or multiply by `e` until `sx ∈ [0.5, 2]`, then a 5-term artanh series |

The five exp copies are the same arithmetic. The four ln copies are also the same arithmetic;
`tco_ln` differs only in line layout. One probe covers all of them.

Results, measured on Madaros:

* The **exp helper** always reads low. Its relative error is −x²/2048 to leading order. The
  comment on `rel_exp` claims "relative error < 0.01% for |x| < 10". That holds only for
  |x| ≲ 0.4525. At x = −10 the error is **−4.80%**.
* On the stent's own first-order release argument at 1545 h the error is **−0.584%**. The
  printed TEST 2 release fraction moves **0.968779 → 0.968595** when `math::pure::exp` is
  swapped in.
* The **ln helper** has an absolute error of up to **1.13 × 10⁻⁶**, which is small next to the
  exp error.
* **GMFE is biased low.** Every GMFE these files compute is `exp(mean ln FE)` through the exp
  helper, so it under-reads by (ln G)²/2048: **−234 ppm at the 2.0 gate** and **−589 ppm at the
  3.0 gate**. A true GMFE in (2.0, 2.0004694] or (3.0, 3.0017691] prints at or below the gate and
  passes.
* **No PASS/FAIL verdict flips today.** The false-pass bands above are latent: no current
  input falls in them. Four printed values move, all in the 4th to 7th significant figure. The pbpk28 GMFE paths are dead until observed data land (`obs_n() = 0`).
  rapamycin_clinical's GMFE is a dead store.
* `math::pure::{exp, ln}` matches V8 libm to within 0.56 × 10⁻¹⁵ at every call-site argument
  measured (relative for exp, absolute for ln). The swap changes no constant.

## 1. The two helpers, analytically

**exp.** `(1 + x/n)^n = exp(n·ln(1 + x/n))`, and for |x| < n

    n·ln(1 + x/n) = x − x²/(2n) + x³/(3n²) − …

Therefore

    helper(x)/exp(x) − 1 = −x²/(2n) + x³/(3n²) + x⁴/(8n²) + O(n⁻³),   n = 1024.

Because ln(1 + y) < y for every y ≠ 0 with y > −1, the helper is **strictly below** exp(x) for
every x ≠ 0 with x > −1024. The bias is one-sided and does not average out.

* The "< 0.01%" claim needs x²/2048 < 10⁻⁴, i.e. |x| < √0.2048 ≈ 0.4525 to leading order.
* At x = −1024 the base 1 + x/1024 is 0. Below that it is negative and the even power is
  meaningless. In double arithmetic the result is exactly 0 wherever |1 + x/1024| < 0.4834,
  i.e. for x between about −529 and −1519. It climbs back to exactly 1 at x = −2048, and
  exceeds 1 below that.
* For the Cypher k_r in `biomaterial_release` these points fall at t ≈ 457 143 h, ≈ 914 286 h
  and beyond. Measured with the helper arithmetic: `release_cumulative` (pub) returns Q = D up
  to about 6.8 × 10⁵ h, Q = 4.5 × 10⁻⁶ mg at 914 285.7 h, −9.0 × 10⁻⁵ mg at 914 286 h,
  −3.7 × 10⁷⁵ mg at 10⁶ h and −∞ at 2 × 10⁶ h.
* The `q > total_dose` cap does not catch negative Q. No in-repo caller goes beyond 10⁵ h.

**ln.** The e-scaling reduces the argument to sx ∈ [0.5, 2]. Then u = (sx − 1)/(sx + 1) ∈ [−1/3, 1/3].

* The series stops at u⁹/9, so the truncation error is −2·Σ_{k≥5} u^{2k+1}/(2k+1). Its
  magnitude is at most 2|u|¹¹/(11(1 − u²)) = 1.1547 × 10⁻⁶ at |u| = 1/3.
* Measured worst case over a 30 001-point grid on [0.5, 2]: **1.133169 × 10⁻⁶**, at both ends.
* The divisor 2.718281828459045 is `e` rounded to double. Each scaling step adds about half an
  ulp, so 14 steps (x = 10⁻⁶) contribute about 10⁻¹⁵. The measured −7.3 × 10⁻¹³ at x = 10⁻⁶ is
  instead the series tail at the reduced argument sx = 10⁻⁶·e¹⁴ ≈ 1.2026 (u ≈ 0.092).

## 2. Helper error at the arguments the callers use

Probe: [`repro/darwin_pbpk_local_exp_ln_accuracy.sio`](repro/darwin_pbpk_local_exp_ln_accuracy.sio).
It holds verbatim copies of both helpers and compares them with `math::pure` and with V8
`Math.exp` / `Math.log` references (< 1 ulp, 17 s.f.). rc = 1 means the defect is present.

On Madaros 5764851f it exits rc = 1 with `REPRO_DARWIN_PBPK_LOCAL_EXP_LN_INACCURATE`. Every
helper figure below agrees with an independent Node evaluation of the same helper to all
printed digits.

| call site | argument | helper error | `math::pure` error |
|---|---|---:|---:|
| exp, stent FO, t = 1 h | −0.00224 | −0.0024 ppm | +2.2e-16 |
| exp, stent FO, t = 168 h (sim end) | −0.37632 | −69.16 ppm | 0 |
| exp, stent FO, t = 720 h (fit anchor) | −1.6128 | −1270.6 ppm | −3.3e-16 |
| exp, stent FO, t = 1545 h (TEST 2) | −3.4608 | **−5844.3 ppm** | 0 |
| exp, `rel_exp`'s own "\|x\| < 10" bound | −10 | **−47960 ppm** | −5.6e-16 |
| exp, GMFE = 2 gate | ln 2 | −234.46 ppm | 0 |
| exp, GMFE = 3 gate | ln 3 | −588.74 ppm | 0 |
| exp, tacrolimus TEST 7 ln_avg (≈ 0.242142 from the printed FEs) | 0.24212 | −28.62 ppm | −2.2e-16 |
| ln, reduced-domain end | 0.5 | +1.133e-6 abs | +2.2e-16 abs |
| ln, reduced-domain end | 2.0 | −1.133e-6 abs | 0 |
| ln, tail concentration | 1e-6 | −7.3e-13 abs | 0 |
| ln, tacrolimus FE(t½) | 1.673122 | −4.96e-8 abs | 0 |

**GMFE false-pass band.** Bisection on `helper(ln g) ≤ G` gives the largest true GMFE that a
gate still passes:

* G = 2.0 (pbpk28 files): **2.0004694** (+234.7 ppm)
* G = 3.0 (tacrolimus TEST 7): **3.0017691** (+589.7 ppm)

## 3. Per file: what moves on Madaros

Each file was run with `bin/souc run` at base, then with the patch applied, on the same ELF and
stdlib path. The runs used the runner described in [§6](#6-reproduce). Outputs are compared as
printed; `println(f64)` is `%f`, 6 decimals.

| file | base rc | patched rc | printed output that moves | verdict |
|---|---:|---:|---|---|
| `release/biomaterial_release.sio` | 0 | 0 | TEST 2 Q(1545 h)/D **0.968779 → 0.968595**; TEST 6 first-order AUC(0–168 h) **0.023305 → 0.023303** | 8/8 PASS both |
| `validation/tacrolimus_oral_pbpk.sio` | 0 | 0 | TEST 7 GMFE **1.273939 → 1.273975** | PASS (≤ 3.0) both |
| `validation/pbpk28_semaglutide_clinical.sio` | 0 | 0 | terminal t½ **226.711659 → 226.711520** h (λz 0.003057 unchanged at print) | `PENDING_OBSERVED` both |
| `validation/rapamycin_clinical.sio` | 182 | 182 | none; the output is byte-identical up to the same abort | n/a (aborts) |
| `validation/pbpk28_rapamycin_clinical.sio` | 1 (E001) | 1 (E001) | not reachable; the harness copy is byte-identical (t½ 3.110789, λz 0.222820) | n/a |

### 3.1 `release/biomaterial_release.sio` (SISTEMA 1, controlled-release axis)

`rel_exp` is reached only through model 2 (first-order) in `release_cumulative` and
`release_rate`.

* **TEST 2.** `release_cumulative(fo, 1545)` with x = −3.4608. The helper under-reads
  e^(−k_r t), so it **over-states** the released fraction by 1.84 × 10⁻⁴ absolute
  (0.968779 against a correct 0.968595). The > 0.96 check passes either way.
* **TEST 4.** At t = 100 000 h, x = −224: the helper gives 1.6e-110 and the true value is
  5.2e-98. Both vanish against D, so the printed 0.140000 is unchanged.
* **TEST 6.** `simulate_14_release` calls `release_step_amount` → `release_cumulative` for
  t ∈ [0, 168] h, where x ∈ [−0.376, 0] and the error is at most −69 ppm. Q(168 h) is therefore
  high by +1.5 × 10⁻⁴ relative. The first-order AUC moves in the 6th decimal; Cmax (0.000230)
  does not move at print resolution.
* **Not affected.** Higuchi and zero-order paths, and TESTS 1, 3, 5, 7, 8.
  `tests/run-pass/darwin_compartments_coronary_smc_smoke.sio` imports this module but uses
  Higuchi only.
* **`release_rate`.** First-order `release_rate` has no in-repo caller, so it moves no printed
  output. It carries the same exp error.
* **Fit anchor.** The comment derives k_r = −ln(0.2)/720 so that 80% is released by 720 h. The
  helper evaluates that anchor as 1 − 0.199075 = 80.09% released. The correct value for the
  rounded k_r = 0.00224 is 80.07%.

### 3.2 `validation/tacrolimus_oral_pbpk.sio`

TEST 7 computes `ln_avg` from three `tco_ln` calls. It then computes exp(ln_avg) inline as
`(1 + x/1024)^1024`, under a comment that calls this "Newton's method".

* **ln contribution.** The FEs printed are 1.156497, 1.673122 and 1.068589. The ln errors add
  −1.65 × 10⁻⁸ to ln_avg, which is negligible.
* **exp contribution.** −28.6 ppm, which accounts for the printed move 1.273939 → 1.273975.
* **Gate.** 1.27 is far from the 3.0 gate. The latent false-pass band is (3.0, 3.0017691].

### 3.3 `validation/pbpk28_semaglutide_clinical.sio`

* **Live path.** `obs_n() = 0`, so the only live consumer is `sv_ln` in the terminal-slope fit.
  It moves the printed t½ by −6.1 × 10⁻⁷ relative.
* **Dead path.** The GMFE branch (`sv_ln` + `sv_exp`) is dead until observed data are filled
  in. It will then read low by (mean ln FE)²/2048, with a false-pass band of (2.0, 2.0004694].

### 3.4 `validation/rapamycin_clinical.sio`

`cv_ln` and `cv_exp` are reached only through `gmfe_from_fes` → `gmfe_all`. That value is
**computed but never printed or tested**: a dead store at line 349.

* Base and patched outputs are byte-identical.
* Both abort at the same point with `madaros: handles full` (rc = 182) at the start of PART B
  (GUM budget). That is the pre-existing unreclaimed-handle defect; it is unrelated to this
  dispatch.

### 3.5 `validation/pbpk28_rapamycin_clinical.sio`

The file does not compile under Madaros at base. `error[E001] … at 13905..13996: this binding
expects a different type` points at `let bar_lbls: [str; 8] = [...]` in the plot block.

To measure the exp/ln path anyway, a **scratch harness copy** was made. It was not committed and
the committed file was not edited. Base and patched variants each had four edits:

* the `use plot::…` lines removed;
* the plot block removed;
* the `let _ont = test_pbpk28_rapamycin_ontology_integration()` call removed;
* one `print(tsit_rapa)` line removed.

Each removal clears a Madaros blocker hit in sequence. None of them involves exp/ln:

1. E001 on the `[str; 8]` binding;
2. lowering refusal "cannot safely lower print/println argument with unresolved scalar kind",
   first in `test_pbpk28_rapamycin_ontology_integration` (a print of a tuple-destructured
   `i32`) and then in `plot::epistemic::error_bar_chart`;
3. `NATIVE_REFUSAL kind=empty_stub_ud2` for `chemistry_rapamycin_chebi` and
   `chemistry_cyp3a4_metabolism_iri` (`missing_lowered_body`), which trap with rc = 132.

These are recorded here as observations for their owners. They are **not diagnosed** in this
dispatch.

With the four edits, both variants run with rc = 0 and print byte-identical output:
t½ = 3.110789 h, λz = 0.222820 h⁻¹. The `pv_ln` shift is below print resolution on this fit.
The GMFE branch is dead (`obs_n() = 0`) and carries the same latent band as §3.3.

## 4. Proposed fix (measured, not applied)

Import `math::pure::{exp, ln}`, delete the local helpers, and call `exp` / `ln` at the call
sites. The tacrolimus inline block becomes `let gmfe = exp(ln_avg)`. The full diff is
[`repro/darwin_pbpk_local_exp_ln_swap.patch`](repro/darwin_pbpk_local_exp_ln_swap.patch)
(5 files, +43 / −164). It applies cleanly to `ce93ea9534`.

The patch was checked for side effects:

* **Constants.** No literal constant changes.
* **Effects.** `pure::exp` / `pure::ln` are `with Mut, Div, Panic`. Every call site is already
  inside a function that declares all three (`release_cumulative`, `release_rate`,
  `gmfe_from_fes`, `main`). The patched files compile on Madaros with the rc values in §3.
* **Name clashes.** No glob import in these files (`tsit5_pbpk14::*`, `tsit5_pbpk28::*`,
  `pbpk28_params::*`) exports `exp` or `ln`. No local binding is named `exp` or `ln`.
* **Behaviour differences, all unreachable from the current callers.**
  * `pure::ln(x ≤ 0)` returns −1e30 instead of −999999. The callers guard with FE ≥ 1 and
    C > 1e-9.
  * `pure::exp` clamps |x| > 500. The largest |x| reached is 224.
* **Comments.** Each file's comment is replaced with one that states the measured error and
  points here. The replaced comments are the false "< 0.01%" claim, "native compiler has no
  stdlib transcendentals here" and "Newton's method".

Applying the patch is a separate, operator-authorised step. After it lands:

* the probe's helper copies still reproduce the historical defect (rc = 1 by design);
* the dissertation text that quotes 0.968779 or 1.273939 must follow.

## 5. Out of scope, not measured

* **Same ln idiom elsewhere.** The e-scaled artanh ln also appears at
  `population/pop_epistemic.sio:67`, `population/pop_pbpk_pd.sio:83`,
  `population/pop_sim.sio:89` and `validation/gum_vs_mc.sio:49`. Term counts and callers were
  not checked.
* **matrix_er.sio.** `release/matrix_er.sio:78` has the same exp and is covered by the precedent
  dispatch.
* **sqrt copies.** `cv_sqrt` / `tco_sqrt` are 15 Newton steps from y = x. Newton converges from
  y = x for every x > 0. However, about ½·log₂(x) halving steps come before quadratic
  convergence, so 15 steps are too few for x ≳ 2³⁰. Their arguments were not measured, and the
  copies are left in place.
* **CI gate engine.** `scripts/ci/dissertation_pbpk_suite_gate.sh` runs all five files through
  `scripts/ci/souc-seq-leansingle.sh`, i.e. lean_single. Under the 2026-09-26 directive that
  gate is not evidence for these files.

## 6. Reproduce

On the workspace, from a checkout of `ce93ea9534` with the Madaros ELF above:

```bash
unset SOUC_BIN SOUNIO_SOUC_BIN MADAROS_RAW_BIN SOUNIO_SOUC_ENGINE
export SOUNIO_STDLIB_PATH=$PWD/stdlib SOUNIO_MADAROS_BIN=$PWD/artifacts/self-hosted/madaros
./bin/souc run docs/audit/repro/darwin_pbpk_local_exp_ln_accuracy.sio          # rc=1 = defect
for f in stdlib/darwin_pbpk/release/biomaterial_release.sio \
         stdlib/darwin_pbpk/validation/{tacrolimus_oral_pbpk,pbpk28_semaglutide_clinical,rapamycin_clinical,pbpk28_rapamycin_clinical}.sio; do
  ./bin/souc run "$f" > "before_$(basename "$f" .sio).out"; echo "$f rc=$?"
done
git apply docs/audit/repro/darwin_pbpk_local_exp_ln_swap.patch
# rerun the loop into after_*.out, then diff before_* after_* (ignore /tmp/madaros-run.* paths)
```

Always go through `bin/souc`, never the raw ELF. The raw ELF runs at an 8 MiB stack and
SIGSEGVs on darwin_pbpk code (`MADAROS_RAW_ELF_8MIB_CHECKER_STACK_2026-09-26.md`).

## 7. Offload review

`bin/llm-offload -t math-review` ran on the canonical route. The detailed entry is in
`.claude/llm_offload_log.md`.

* **Grok 4.7** (xAI). The gateway leg failed and the driver fell back to xAI direct, which
  completed. It confirmed every load-bearing figure. It found four wrong side statements: the
  x ≤ −1024 behaviour, the 0.4525 radius, the source of the 7.3e-13 ln error, and the sqrt
  caveat. All four are corrected above.
* **Qwen 3 235B** (Alibaba, via OpenRouter). It reviewed the corrected text and marked every
  math claim OK. One emphasis point was applied.
* **Kimi K3**, the canonical second vendor, failed four times. Three times it exhausted its
  token budget on reasoning; once the gateway's upstream connection failed. It is not counted.
