<!-- docs:meta
topic_id: repo.docs.audit.lean-single-imported-type-error-fail-open-2026-09-26
authority: repo_only
audience: users
last_validated: 2026-09-26
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-imported-type-error-fail-open-2026-09-26
-->

# Dispatch: `print_i64` is declared nowhere, and lean_single's import tolerance hides it (2026-09-26)

**Status:** OPEN for F-B (lean_single). F-A applied, plus the E259 and E008
follow-ups below, under the operator directive that **Madaros is the only
compiler for PBPK** (2026-09-26). No change to `self-hosted/`. Filed under
the forensic dispatch protocol (`CLAUDE.md` §8) before any patch.

## Claim

The report that opened this dispatch read Madaros' `E137` on the BBB tests as
a name Madaros fails to resolve. The measurement says the reverse:

1. **Madaros is right.** The name at both spans is `print_i64`. It is not a
   builtin on either engine (the integer-print builtin is `print_int`), and no
   module in either test's import closure declares it. `E137` is the correct
   verdict. **No patch to `self-hosted/check` is warranted.**
2. **lean_single's `BBB_GATE_OK` is a false green.** lean_single does report
   the error (`E200 undefined identifier`) but treats it as non-fatal because
   it sits in an *imported* module. It then compiles the call to
   `xor eax, eax`. The call prints nothing, and a value-returning undefined
   call yields `0`. The run exits `0`.
3. The same undeclared `print_i64` appears in three more `darwin_pbpk`
   modules, with nine call sites in total across the five modules.

## Decoding the spans

Spans are byte offsets into the accessing module, not the test file.
Measured on `main` @ `2e8b76d312`:

| Module | Span | Bytes at span | Line |
|---|---|---|---|
| `stdlib/darwin_pbpk/bbb/bbb_gate.sio` | `4579..4588` | `print_i64` | 117 — `print_i64(n); println("")` |
| `stdlib/darwin_pbpk/bbb/bbb_voi.sio`  | `4885..4894` | `print_i64` | 146 — `print_i64(rank_k); print(",")` |

```bash
python3 -c "d=open('stdlib/darwin_pbpk/bbb/bbb_gate.sio','rb').read(); print(d[4579:4588])"   # b'print_i64'
python3 -c "d=open('stdlib/darwin_pbpk/bbb/bbb_voi.sio','rb').read();  print(d[4885:4894])"   # b'print_i64'
```

Both lines were introduced with the modules themselves
(`74275805e8`, 2026-04-22; `b4e1ca96f6`, 2026-04-23) and have never compiled
to a working call on either engine.

## Why `print_i64` is not a name

- Madaros builtin predicates: `call_expr_is_builtin_print_int`
  (`self-hosted/check/check.sio:18332`, bytes `p r i n t _ i n t`) and
  `call_expr_is_builtin_print_char`. There is no `print_i64` predicate, and
  `git grep '"print_i64"' -- self-hosted` is empty.
- lean_single matches `src_match(ns, ne - ns, "print_int")`
  (`self-hosted/compiler/lean_single.sio:14773`) and `print_f64`. There is no
  `print_i64` arm.
- The only `fn print_i64` definitions under `stdlib/` are private helpers in
  `stdlib/metrology/calibration.sio:322` and `stdlib/plot/bar.sio:301`. Neither
  is in the BBB closure, and neither is `pub`.
- `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check` over each test lists
  exactly one undefined identifier in the whole closure:

  ```
  test_bbb_gate: error[E200]: undefined identifier `print_i64` at stdlib/darwin_pbpk/bbb/bbb_gate.sio:117
  test_bbb_voi:  error[E200]: undefined identifier `print_i64` at stdlib/darwin_pbpk/bbb/bbb_voi.sio:146
  ```

  Both runs exit `0`.

## Minimal reproduction (13 lines, two files)

`print_i64_leaf.sio`:

```sounio
// Leaf module. `print_i64` is not a builtin (the builtin is `print_int`)
// and is declared nowhere in this module's closure.
pub fn leaf_print(n: i64) with IO, Mut, Panic, Div {
    print("n=")
    print_i64(n)
    println("")
}
```

`print_i64_main.sio`:

```sounio
use print_i64_leaf::{leaf_print}

fn main() with IO, Mut, Panic, Div {
    leaf_print(42)
    println("REACHED_END")
}
```

| Engine | Command | Result |
|---|---|---|
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run print_i64_main.sio` | prints `n=` (no digits), `REACHED_END`; **rc=0** |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check print_i64_main.sio` | prints `error[E200]: undefined identifier \`print_i64\``, emits an ELF; **rc=0** |
| Madaros | `./bin/souc check print_i64_main.sio` | `error[E137] in print_i64_leaf::leaf_print at 200..209`, `= name print_i64`; **rc=1** |

Control cases, same session:

- The same call in a **single-module** program is rejected by **both** engines
  (rc=1). The tolerance is specific to imported modules.
- `print_int(n)` in place of `print_i64(n)` prints `42` on both engines.
- A value-returning variant `pub fn v(n: i64) -> i64 { no_such_fn_anywhere(n) + 1 }`
  returns `1` under lean_single (the undefined call yields `0`); Madaros
  rejects it with `E137`, `= name no_such_fn_anywhere`.

## Root cause of the lean_single false green

`self-hosted/compiler/lean_single.sio:4557`:

```sounio
fn tc_mark_failed() with Mut {
    TC_FN_ERR_COUNT = TC_FN_ERR_COUNT + 1
    if MAIN_SRC_END > 0 && CURRENT_FN >= 0 && CURRENT_FN < 65536 {
        if (FN_EFFECTS[CURRENT_FN as usize] & 2048) != 0 {
            return                      // <-- imported fn: error is not fatal
        }
    }
    TYPECHECK_FAILED = 1
}
```

The undefined-identifier arm (`lean_single.sio:19086`) prints `E200`, calls
`tc_mark_failed()`, and emits `xor eax, eax` in place of the value. For a
function carrying the import bit (`FN_EFFECTS & 2048`), `TYPECHECK_FAILED` is
never set. The "CONVERGENCE FIX" block (`lean_single.sio:33754`) states the
intent outright: *"Functions with few E200s can still run — the xor rax,rax
placeholder returns 0 for undefined vars."* It stubs an imported function
only when it has more than 10 errors.

Provenance (`git blame -L 4557,4565`): the import-bit early return arrived in
`781b11a780` (2026-04-18). That is the bulk "Consolidate all language work
onto integration/sounio-dev-ready-base" commit, whose message lists no
rationale for this guard. The `MAIN_SRC_END > 0 && CURRENT_FN …` bounds line
was later edited by `b3e319cfb9` (2026-05-29).

So in lean_single, any type error in an imported module (not only `E200`;
anything routed through `tc_mark_failed`) is a warning in effect. This is the
same class of fail-open as the `tc_error` severity note at
`lean_single.sio:4580`.

## Blast radius (measured)

Modules that call `print_i64(` without declaring it
(`git grep` over `stdlib/`, then filter out files that define `fn print_i64(`):

| Module | Call sites | Direct importers |
|---|---|---|
| `stdlib/darwin_pbpk/bbb/bbb_gate.sio` | 117 | `tests/.../bbb/test_bbb_gate.sio`, `stdlib/darwin_pbpk/aggregate_confidence.sio` |
| `stdlib/darwin_pbpk/bbb/bbb_voi.sio` | 146 | `tests/.../bbb/test_bbb_voi.sio`, `stdlib/darwin_pbpk/bbb/scenario_gate.sio`, `examples/dissertation_scenario_gate_demo.sio` |
| `stdlib/darwin_pbpk/pd/pd_gate.sio` | 141, 144, 145 | `tests/stdlib/darwin_pbpk/test_pd_gate.sio` |
| `stdlib/darwin_pbpk/pd/pd_gum.sio` | 357 | `pd_gate.sio`, `test_pd_gate.sio`, `test_pd_gum_voi.sio` |
| `stdlib/darwin_pbpk/scenarios/steady_state_runner.sio` | 315, 317, 331 | `examples/dissertation_steady_state_demo.sio`, `examples/dissertation_steady_state_fullvd_demo.sio` |

Observed lean_single output with the integers silently missing:

```
test_bbb_gate:  "  n_drivers       = "                      (count absent)
test_pd_gate:   "  VERDICT: PASSED —  parameters meet confidence floor"
                "  VERDICT: FAILED —  of  parameters below floor"
```

`test_bbb_voi` loses the `rank` column of its CSV the same way:

```
rank,param,sobol_S1,confidence,VoI
,ps_bbb,0.272386,0.450000,0.149812
,fu_icf,0.185994,0.250000,0.139495
```

and still prints `BBB_VOI_OK`. **Not claimed:** `examples/dissertation_steady_state_demo.sio`
does not reach the runner's `print_i64` lines. Its visible output
(`doses_run = 7.000000`, …) is intact under lean_single. The fullvd variant
was not run.

No numerical result is affected: every site prints an integer the caller
already holds. What is affected is (a) the printed audit trail of the
BBB/PD gates, and (b) the fact that these tests, and these dissertation
examples' imports, cannot type-check under default Madaros while the call
stands.

## Confirmation on a source-built Madaros

The tables above were first measured on the committed ELF (md5 `57c015c1`).
Principle 15 requires a compiler built from the commit under test, so they
were re-run on `make build-madaros` from `main` @ `2e8b76d312`
(`artifacts/self-hosted/madaros`, md5 `5764851f`, invoked through `./bin/souc`).

| Target | E137 | E259 | rc |
|---|---:|---:|---:|
| reproduction (`print_i64_main.sio`) | 1 (`= name print_i64`) | 0 | 1 |
| `tests/stdlib/darwin_pbpk/bbb/test_bbb_gate.sio` | 1 | 8 | 1 |
| `tests/stdlib/darwin_pbpk/bbb/test_bbb_voi.sio` | 1 | 9 | 1 |
| `tests/stdlib/darwin_pbpk/test_pd_gate.sio` | 4 | 38 | 1 |

The four E137s in `test_pd_gate` are the three `pd_gate.sio` sites plus the
one in `pd_gum.sio`. No other error code appears.

**A second, independent blocker is visible on current source.** Every test
above also fails with `E259 struct field is private in its defining module`,
raised in the test's own `main`. Example: `test_bbb_gate.sio` bytes
`1697..1710` are `v1.admitted`, and `BBBGateVerdict` is declared
`pub struct` with fields that carry no `pub`
(`stdlib/darwin_pbpk/bbb/bbb_gate.sio:39`). The report that opened this
dispatch (Madaros from `98315edcdb`) quoted only the E137. Whether that build
emitted the E259s too was not re-measured here. The E259 question (make the
fields `pub`, add accessors, or revisit the default) is out of scope for this
dispatch and is not diagnosed here.

## Proposed fixes (not applied)

**F-A: stdlib (trivial, clears the E137).** Replace `print_i64(` with
`print_int(` at the nine sites above. Every argument is already `i64` or cast
`as i64`, and `print_int` is type-checked against `i64` on Madaros
(`checker_check_print_i64_expr_inplace`, `check.sio:10330`).

Trial, measured 2026-09-26 on the remote worktree with the substitution
applied uncommitted and then reverted (`git checkout -- stdlib`):

| Test | lean_single after F-A | Madaros (`5764851f`) after F-A |
|---|---|---|
| `test_bbb_gate` | rc=0, `n_drivers = 4` / `= 1`, `BBB_GATE_OK` | rc=1, E137=0, E259=8 |
| `test_bbb_voi` | rc=0, `1,ps_bbb,0.272386,…`, `BBB_VOI_OK` | rc=1, E137=0, E259=9 |
| `test_pd_gate` | rc=0, `PASSED — 10 parameters…`, `FAILED — 6 of 10…` | rc=1, E137=0, E259=38 |

So F-A does two things: it restores the missing integers on lean_single, and
it removes every E137 on Madaros. It does **not** make these tests pass on
Madaros, because the E259 blocker remains. A parallel lane
(`claude(sleepy-easley)`, coord message 2026-09-26T20:34Z) separately reports
`rc=182 handles full` for BBB `darwin_pbpk` runs under Madaros once they
compile. That is not re-measured here, but it is likely a third wall behind
the E259.

**F-B: lean_single (seed; needs measurement first).** Make `tc_mark_failed()`
fatal for imported functions, at least for `E200`. This changes the frozen
bootstrap seed and may flip currently green tests that depend on the same
tolerance. Before any patch:
1. Count `E200`/`tc_mark_failed` hits in imported modules across the suite
   under lean_single. The `check` output already names them and exits 0, so
   `grep -c '^error\['` over a suite-wide `check` sweep is the census.
2. Confirm that `make build` (the gen2 == gen3 fixed point over `lean_single.sio`)
   has zero imported-module errors, so tightening cannot break the bootstrap.
3. Only then change the early return. A reasonable first step is to make
   `souc check` exit nonzero whenever it has printed an `error[` line, which
   removes the "check prints an error and says rc=0" contradiction without
   touching codegen.

**F-C: gate (optional).** A multimodule `//@ compile-fail` pair built from the
reproduction above, marked `//@ requires: madaros`, to pin Madaros' correct
rejection. Under lean_single the same pair is the witness for F-B.

## Follow-up: PBPK under Madaros only (2026-09-26)

Operator directive, 2026-09-26: Madaros is the only compiler for PBPK. A
lean_single green on `darwin_pbpk` is not evidence. Three `stdlib/` commits
follow, each an atomic change:

1. **F-A.** `print_int` replaces `print_i64` at the nine sites.
2. **E259.** 65 fields in 13 result/record structs across 9 files become
   `pub`. This continues `b77dd4532`, whose scan skipped these files because
   they also carried the E137. Every site was traced to its owning struct
   through the producer's return type, and the only token added is `pub`.
3. **E008.** In `scenarios/steady_state_runner.sio::ssr_run_one_interval`, the
   step-rejection bailout returned the bare `OralBBBTrace` where the function
   declares `SSIntervalResult`. The loop had been copied from `oral_bbb_run`,
   where `return trace` is correct. lean_single accepted this for the same
   reason as `print_i64`. The fix wraps the carried state and leaves
   `trace.success = false` as the failure signal. At that commit
   `run_oral_multidose` did not yet read `success`. That was fixed later in
   this PR; see "Remaining PBPK targets and runner failure signal".

### Census

Scope: the 27 files then in `tests/stdlib/darwin_pbpk/` plus 11 `examples/`
matched by `git grep 'darwin_pbpk::'`, 38 targets in total. Ten of those
examples import `darwin_pbpk`. The eleventh, `examples/clinical/ddi_elplus_demo.sio`,
only mentions it in a comment; it was checked and run like the others, which
is harmless but means the scope was 27 PBPK tests, 10 PBPK examples and one
non-PBPK example. Madaros built from source, md5 `5764851f`.

The first census was taken with F-A already applied. Before F-A the E137s
were measured per test only (see the tables above).

| `souc check`, distinct sites | + F-A | + E259 | + E008 |
|---|---:|---:|---:|
| targets with rc=0 | 26 | 34 | 36 |
| E259 | 159 | 0 | 0 |
| E008 | 1 | 1 | 0 |
| E175 (`test_simulation_e2e`) | 6 | 6 | 6 |
| E137 (`tsit5_pbpk14_demo`, unrelated to `print_i64`) | 4 | 4 | 4 |

`souc run` on the final tree, Madaros against lean_single. Madaros writes
its compile log to stdout, so program output is compared from the line after
`Written to …/main.elf`:

| Outcome | Targets |
|---|---:|
| rc=0 on both engines, program output byte-identical | 22 |
| rc=0 on both engines, numeric difference in the last printed digits | 2 |
| Madaros `rc=182` (`madaros: handles full`) | 12 |
| does not compile on Madaros | 2 |

- **Numeric differences: explained, lean_single is the one that is wrong.**
  `dissertation_steady_state_demo` prints `4.000000,0.005650,…` on Madaros
  against `0.005649` on lean_single. `dissertation_steady_state_fullvd_demo`
  prints `C_max_last / C_max_first` 1.063169 against 1.063101, and
  `AUC_last / AUC_first` 1.228366 against 1.228345. The root cause is
  lean_single's float-literal conversion (next section). The Madaros values
  are the ones the source as written specifies.
- **`rc=182`.** `test_bbb_gate`, `test_bbb_gum_budget`, `test_bbb_hdmr_7d`,
  `test_bbb_pce2d_sobol`, `test_bbb_pce_vs_gum`, `test_bbb_voi`,
  `test_des_bbb_coupled`, `test_brain_plasma_tac`,
  `test_observed_petab_fit_e2e`, `test_pd_gum_voi`, `test_steady_state`, and
  `examples/dissertation_scenario_gate_demo.sio`. This is the unreclaimed
  handle table that lane `claude(sleepy-easley)` is fixing in
  `self-hosted/native/gc.sio`. The failure is runtime, not type checking.
- **Does not compile (both since fixed; see the next section).**
  - `test_simulation_e2e.sio` calls the private Butcher tableau helpers
    `tsit5_c2()` … `tsit5_a31()` of `tsit5_pbpk14.sio` (E175).
  - `examples/darwin_pbpk/tsit5_pbpk14_demo.sio` has no `use` line.
  - **Correction:** an earlier revision said the demo's four functions
    (`default_pbpk_params`, `default_ode_config`, `solve_pbpk14`,
    `pbpk_state_total_mass`) exist nowhere in `stdlib/`. That was wrong. All
    four are `pub` in `tsit5_pbpk14.sio`; the search that "found nothing"
    used `\b`, which `git grep -E` does not support. The demo's defects
    were the missing `use`, undeclared effects on `main`, reads of
    `PBPKSolution14`'s private fields, and a `println(value)` per label.

## Steady-state divergence: root cause (2026-09-26)

**Claim.** lean_single does not convert decimal float literals to the
nearest `f64`. Madaros does. In the closure of the steady-state demos, 14 of
115 distinct literals land 1–2 ulp away from the correctly rounded value
under lean_single. With those 14 literals supplied as exact values, lean_single
reproduces Madaros' output byte for byte.

**Mechanism.** `self-hosted/compiler/lean_single.sio:13131` (float literal,
token kind 53) does not emit the literal's bits. It emits code that rebuilds
the value at run time:

```
cvtsi2sd(int_part) + cvtsi2sd(frac_digits) / cvtsi2sd(10^n)
```

followed by one `mulsd` or `divsd` by `10.0` per unit of the decimal exponent.
Every step rounds. `1.0e-30`, for example, is thirty successive divisions of
`1.0` by `10.0`.

### Measurement 1: literal bits (115 literals)

All distinct float literals in the 10 modules of the demos' closure
(`absorption`, `bbb_core`, `bbb_rapamycin`, `drugs/rapamycin`,
`epistemic_pbpk14`, `oral_rapamycin_bbb`, `steady_state_runner`,
`tsit5_pbpk14` and the two demos). Each was printed through
`print_int(f64_to_bits(<literal>))` on both engines and compared, as exact
64-bit integers, with Python's correctly rounded `float()`.

| Engine | Literals off the correctly rounded value |
|---|---:|
| Madaros (md5 `5764851f`) | 0 of 115 |
| lean_single | 14 of 115 |

The 14, with lean_single's error in ulp:

| Literal | Where | ulp |
|---|---|---:|
| `1.0e-30` | `epistemic_pbpk14.sio:308` | +1 |
| `1.0e-12` | `drugs/rapamycin.sio`, `epistemic_pbpk14.sio`, `steady_state_runner.sio:291` | +1 |
| `0.0000000001` | `tsit5_pbpk14.sio:636,638` (`atol`, `dt_min`) | +2 |
| `0.0000001` | `tsit5_pbpk14.sio:615` (`dt_min`) | +2 |
| `0.000001`, `1.0e-6` | `tsit5_pbpk14.sio:635` (`rtol`), `epistemic_pbpk14.sio:159` | +1 |
| `0.000009` | `drugs/rapamycin.sio:346` | −1 |
| `0.00178001105222577714` | `tsit5_e1` | +1 |
| `0.01515151515151515` | `tsit5_e7` | −1 |
| `0.028269050394068383` | `tsit5_a65` | +1 |
| `0.041` | `bbb_rapamycin.sio:42` (`v_vasc`) | −1 |
| `0.09249506636175525` | `tsit5_a54` | −1 |
| `0.9800255409045097` | `tsit5_c5` | −1 |
| `1.379008574103742` | `tsit5_a74`, `tsit5_b4` | +1 |

### Measurement 2: causal test

In a scratch copy of `stdlib/`, not committed, each of the 21 occurrences of
those 14 literals was replaced by an expression both engines evaluate
exactly: `((M as f64) / (2^j as f64) / …)`, where `M < 2^53` and every divisor
is a power of two. Printing the bits of those expressions matched the
correctly rounded reference on both engines, for all 14.

Two points about the harness:
- lean_single resolves `stdlib/<module>` relative to the **current
  directory** and ignores `SOUNIO_STDLIB_PATH`, so the run was made from a
  directory whose `stdlib/` is the copy.
- A marker injected into the copy confirmed lean_single read it. A first
  attempt that relied on `SOUNIO_STDLIB_PATH` silently compiled the original
  stdlib and was discarded.

| Comparison | `steady_state_demo` | `steady_state_fullvd_demo` |
|---|---|---|
| lean_single (original) vs Madaros | differs, 1 line | differs, 3 lines |
| **lean_single (exact literals) vs Madaros** | **identical** | **identical** |
| Madaros (exact literals) vs Madaros | identical | identical |

The last row is expected: Madaros already parses the literals exactly, so
nothing changes for it. The literal conversion accounts for the entire
divergence. No other engine difference is involved in these two demos.

### Consequences

- Under the directive the Madaros numbers stand, and they are the ones
  faithful to the source.
- A 1–2 ulp perturbation of constants moved the printed C_max and AUC
  ratios by up to 6.4e-5 relative. The adaptive Tsit5 step control
  (`rtol = 1e-6`) is the likely amplifier, because accept/reject decisions
  near `err_norm = 1` depend on the tolerance and tableau literals. That is
  inferred, not traced step by step.
- **The engine gap is not the accuracy of these numbers.** Lane
  `claude(gracious-bardeen)` reports an independent fixed-step RK4
  reference (continuous absorption, no operator splitting) for
  `AUC_last / AUC_first` in `dissertation_steady_state_demo`: 1.227702.
  **Both** engines sit about 6.5e-4 away from it, roughly 31 times the
  engine gap. They attribute this to the runner's operator-split oral bolus,
  which is O(dt). The report is their dispatch,
  `docs/audit/STEADY_STATE_DEMO_ENGINE_DIVERGENCE_REFERENCE_2026-09-26.md`
  (commit `6db545723` on `claude/gracious-bardeen-155cb0`, not yet pushed).
  It is not re-measured here. On that evidence these ratios carry about
  three significant figures, and the engine difference is noise beneath the
  discretisation error. That lane also reports `t_to_90pct_h`,
  steady-state dose labelling and trapezoid-AUC defects in
  `steady_state_runner.sio`; those three were fixed later in this PR (see
  "Steady-state runner: three endpoint bugs" below).
- Any lean_single-versus-Madaros parity comparison involving decimal literals
  can differ at the ulp level for this reason alone. Parity checks should
  compare with a tolerance or on correctly rounded inputs.
- Proposed fix for lean_single (not applied; it is the bootstrap seed, same
  caution as F-B): compute the literal's correctly rounded bits at compile
  time and emit `mov rax, imm64`. The `const` path
  (`lean_single.sio:19030`) rebuilds the value the same way, with the
  `mulsd`/`divsd` by `10.0` loop, and would need the same change.


## Remaining PBPK targets and runner failure signal (2026-09-26)

Operator decision: make the Tsit5 helpers `pub` and rewrite the demo.

- `stdlib/darwin_pbpk/tsit5_pbpk14.sio`: the 41 Butcher tableau helpers
  (`tsit5_c*`, `tsit5_a*`, `tsit5_b*`, `tsit5_e*`) and the 6 fields of
  `PBPKSolution14` become `pub`. `stdlib/ode/tsit5_multicomp.sio` defines the
  same 41 names privately. No module imports both, so nothing collides; the
  census below confirms it.
- `examples/darwin_pbpk/tsit5_pbpk14_demo.sio` was rewritten against the
  existing API: `use` line, effects on `main`, label and value on one line.
  Its pass condition is physical bounds only (`0 < % eliminated < 100`,
  `success`, `t_final >= t_end`), not a fitted number. The old comment's
  "expect ~25–30% elimination" was wrong: the measured value is 71.591962%.
- Census with both changes (Madaros md5 `5764851f`):
  - `check` rc=0 on **38/38** targets (the same 38 as above; this
    predates the regression fixture described below).
  - `run`: 24 byte-identical with lean_single (up from 22), 2 steady-state
    demos differing by the literal rounding explained above, and 12
    `rc=182`. No regressions.

A review of PR #2698 noted that `run_oral_multidose` ignored the
`trace.success = false` bailout, so a failed integration still produced a
normal-looking report. `ssr_run_one_interval` had a second silent path as
well: exhausting `cfg.max_steps` before a checkpoint recorded the point and
reported success. Both now fail:

- The interval returns `success = false` when `t < target` after the step
  loop. That can only happen through `nsteps >= max_steps`, because the loop
  otherwise continues while `t < target`.
- `SteadyStateReport` gains `success`. `run_oral_multidose` stops at the
  first failed interval, with `n_doses_run` counting only complete intervals.
- Both steady-state demos return 1 on a failed report.

Verified on Madaros:
- Both demos print output byte-identical to before the change.
- With `tight_ode_config().max_steps = 1` in a scratch stdlib copy, the demo
  prints `integration failed after 0 complete dose interval(s)` and exits 1.

That failure is now replayable in CI. `run_oral_multidose_cfg` takes the ODE
config, and `run_oral_multidose` wraps it with `tight_ode_config()`, so its
behaviour is unchanged. `tests/run-pass/darwin_pbpk_steady_state_failure.sio`
forces one step of at most 1e-3 h per checkpoint and asserts
`success == false`, `n_doses_run == 0` and `reached_ss == false`. As a
control, the default config completes all three doses with
`success == true`. `ssr_print_report` now flags a failed report.

The fixture was first added under `tests/stdlib/darwin_pbpk/`. It was moved to
`tests/run-pass/` with `//@ requires: madaros` because
`scripts/ci/madaros_changed_tests_gate.sh` selects only changed
`tests/run-pass/*.sio` files carrying that annotation. Only there does it run
on the current-source Madaros that CI builds, rather than solely under the
lean_single full suite.


Two more current-source Madaros guards, both selected by the same gate:

- `tests/run-pass/darwin_pbpk_record_fields_madaros.sio` (`check-only`). It
  reads all 65 fields made `pub` here, plus `SteadyStateReport.success`, from
  outside their modules and constructs `HillEpParam`. Type-checking its
  closure also re-covers the E137 and E008 module bodies. It is not a
  runtime test because the BBB pipeline still hits `rc=182`.
- `tests/run-pass/darwin_pbpk_tsit5_public_surface.sio` (run). It asserts
  Butcher-tableau properties (row sums, `sum b = 1`, `sum e = 0`, exact FSAL
  `b_i == a_7i`) and `nfeval == 7 * (nsteps + nreject)`. The tolerance of
  1e-13 is derived in the file header.

Sabotage, each measured through the gate:

| Sabotage | Result |
|---|---|
| `BBBGateVerdict.admitted` made private again | check fails |
| `print_i64` reintroduced in `bbb_gate` | check fails |
| `tsit5_a52` off by 1e-10 | run fails with exit 4 |
| `tsit5_b4` changed in the last digit | run fails with exit 12 (FSAL) |

A 1e-15 change to `a52` passes, as designed: it is below the derived
tolerance.

A later review also noted that a regimen outside 1..32 doses produced
`success = true`: zero doses for `n_doses < 1`, or silent truncation to 32.
`run_oral_multidose_cfg` now fails such a request up front with
`success = false` and `n_doses_run = 0`. `darwin_pbpk_steady_state_failure`
covers `n_doses` = 0, −1 and 33.


## Steady-state runner: three endpoint bugs (2026-09-26)

Reported by lane `claude(gracious-bardeen)` (PR #2711). Fixed here at the
operator's request. Each fix has its own commit and a
`requires: madaros` regression test. Restoring the old runner makes each
test fail on its own assertion.

| Bug | Fix | Test (exit code with old runner) |
|---|---|---|
| `dose_of_ss` was the 0-based loop index, one row off the 1-based table | stores `dose_i + 1` | `darwin_pbpk_steady_state_endpoints` (4): `auc_tau_ss == auc_tau_per_dose[dose_of_ss - 1]` |
| `t_to_90pct_h` tested `grew >= 0.9 * (grew + 1e-12)`, so it returned 2·tau for every drug | first interval whose AUC_tau reaches 90% of the last interval's AUC_tau; `-1` (undefined) without a detected SS | `darwin_pbpk_steady_state_failure` (10): no t_90 for a 3-dose regimen |
| AUC_tau was a 16-point trapezoid (1.6 h spacing) | accumulated over every accepted Tsit5 step (plasma) and BBB RK4 sub-step (ISF, ICF) | `darwin_pbpk_steady_state_auc_quadrature` (2) |

The AUC check uses a reference independent of the runner's quadrature. Dose
1 is integrated inside the test with a fixed step and the same
operator-split input. That reference is first-order in dt (0.005 h to
0.0025 h moves it 404 ppm), so the Richardson extrapolation is used.
Measured against it: new runner +3.1e-4, old runner −7.4e-3; the bound is 2e-3.
A first version of this test compared the runner against itself at a tighter
tolerance. The checkpoint trapezoid cancelled on both sides, and the whole-runner
sabotage showed that version passing on the old code. It was replaced before
commit.

Effect on the demos (Madaros):

| Quantity | Before | After |
|---|---|---|
| `steady_state_demo` `t_to_90pct_h` (SS not reached) | 48 | undefined |
| `steady_state_demo` AUC_tau, dose 1 / 7 | 0.002922 / 0.013998 | 0.002945 / 0.014022 |
| `fullvd_demo` SS dose label | #7 | #8 (same interval) |
| `fullvd_demo` AUC_tau_ss | 0.001672 | 0.001735 (+3.8%) |
| `fullvd_demo` `t_to_90pct_h` | 48 | 48 (now because dose 2 reaches 91.5% of the last interval) |
| `fullvd_demo` AUC_last / AUC_first | 1.228366 | 1.218177 |

The ratio falls because the old trapezoid under-read the sharper first dose
more than the last. gracious-bardeen's reference (PR #2711) prints both
quadratures of an independent fixed-step RK4 run, and has confirmed this on
the coordination bus. Its 1.227701683 applies the same 16-checkpoint
trapezoid, so it validates only the old metric. Its trapezoid over every RK4
step (h = 2.5e-3 h; halving h moves it 3e-7) gives 1.2181994. The fixed
runner's 1.218177 is −1.8e-5 relative from that, consistent with the O(dt)
split-bolus error that remains in the runner. A math review (xai, qwen;
`.claude/llm_offload_log.md`) confirmed the quadrature and changed one
choice: the t_90 plateau is the last interval, not the AUC at SS
declaration, which can sit ~5% below the plateau.

### Open, not fixed: blood concentration used as plasma

Found during the math review and confirmed in the code; not changed here.
`pbpk_ode` integrates **blood** concentration
(`c_plasma = c_blood / rb_ratio`, `tsit5_pbpk14.sio:381`). Two places treat
blood as plasma:

- **Reported plasma exposure.** `steady_state_runner` stores `sys_st.blood`
  in `trace.plasma[]` and reports `auc_plasma_u = fu_plasma * AUC(blood)`.
- **BBB coupling.** It drives the BBB model with `c_mid`, interpolated from
  `sys_st.blood`, while `bbb_ode(st, c_plasma, prm)` documents and uses its
  argument as plasma.

`oral_bbb_run` in `scenarios/oral_rapamycin_bbb.sio` has the same pattern.
The size and sign of the error depend on `rb_ratio`, since
plasma = blood / rb:

- `rapamycin_mean_params` (`rb_ratio = 0.58`): the reported unbound plasma
  AUC and the BBB driving concentration are about 42% low (plasma is 1.72 ×
  blood).
- `rapamycin_fullvd_params` (`rb_ratio = 36.0`, Yatscoff 1995): they are
  about 36 × high.

Ratios such as AUC_last / AUC_first are scale-free and survive. Absolute
concentrations, AUCs and Kp,uu do not. The sign correction, which the
first version of this note missed, is from gracious-bardeen. It needs an operator decision
and a math review before any change.

A later review noted that the runner's header described `C_max_ss`,
`C_trough_ss` and `AUC_tau_ss` as last-interval values, while the code
freezes them at the interval where steady state is declared. The
documentation now states the actual behaviour (declaration interval,
`dose_of_ss`). Whether these endpoints should instead report the last
interval is an open operator decision. It would change reported numbers
(full-Vd AUC_tau_ss 0.001735 -> 0.001737), and the declaration can sit
below the plateau (see the math review above).

