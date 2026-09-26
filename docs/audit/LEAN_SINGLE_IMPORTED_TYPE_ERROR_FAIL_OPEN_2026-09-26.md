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
   `trace.success = false` as the failure signal. `run_oral_multidose` never
   reads `success`; that is noted here and left unchanged.

### Census

Scope: the 27 files in `tests/stdlib/darwin_pbpk/` plus the 11 `examples/`
that import `darwin_pbpk`, 38 targets in total. Madaros built from source,
md5 `5764851f`.

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

- **Numeric differences, not yet explained.**
  `examples/dissertation_steady_state_demo.sio` prints
  `4.000000,0.005650,…` on Madaros against `0.005649` on lean_single.
  `examples/dissertation_steady_state_fullvd_demo.sio` prints
  `C_max_last / C_max_first` 1.063169 against 1.063101, and
  `AUC_last / AUC_first` 1.228366 against 1.228345: relative differences of
  6.4e-5 and 1.7e-5. The cause has not been established; step-control
  divergence in the adaptive integrator is a hypothesis only. Under the
  directive the Madaros value is the one reported, but no value from these
  two demos should be quoted until the divergence is understood.
- **`rc=182`.** `test_bbb_gate`, `test_bbb_gum_budget`, `test_bbb_hdmr_7d`,
  `test_bbb_pce2d_sobol`, `test_bbb_pce_vs_gum`, `test_bbb_voi`,
  `test_des_bbb_coupled`, `test_brain_plasma_tac`,
  `test_observed_petab_fit_e2e`, `test_pd_gum_voi`, `test_steady_state`, and
  `examples/dissertation_scenario_gate_demo.sio`. This is the unreclaimed
  handle table that lane `claude(sleepy-easley)` is fixing in
  `self-hosted/native/gc.sio`. The failure is runtime, not type checking.
- **Does not compile.** `test_simulation_e2e.sio` calls the private Butcher
  tableau helpers `tsit5_c2()` … `tsit5_a31()` of `tsit5_pbpk14.sio` (E175).
  Whether to make solver internals `pub` is an API decision, left to the
  operator. `examples/darwin_pbpk/tsit5_pbpk14_demo.sio` has no `use` and
  calls four functions (`default_pbpk_params`, `default_ode_config`,
  `solve_pbpk14`, `pbpk_state_total_mass`) that exist nowhere in `stdlib/`.
  It fails on both engines and needs rewriting against the real API.

