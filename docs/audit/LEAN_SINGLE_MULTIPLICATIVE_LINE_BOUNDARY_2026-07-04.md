<!-- docs:meta
topic_id: repo.docs.audit.lean-single-multiplicative-line-boundary-2026-07-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-multiplicative-line-boundary-2026-07-04
-->

# lean_single forensic dispatch — multiplicative parser absorbs a line-leading `*` from the next statement

Date: 2026-07-04
Branch: `main` @ `85a8e9f5a`
Class: **checker/parser false-positive** (a valid dereference-assignment statement is
misparsed as a continuation of the previous statement's expression) — root-causes issue
#601's "Bug H"
Status: root-caused, fixed, verified (full test suite 1311/1311, zero regressions)

## Symptom

A `*ref = ...` dereference-assignment lexically following an earlier function-call
statement in the same function fails to compile:

```sio
fn noop() -> i64 { 0 }
fn write_only(n: &!i64) with Mut {
    noop()
    *n = 999          // error: arithmetic operands must have matching numeric types
}
```

Not specific to arithmetic, tuples, or literals — the failure is a pure literal write
with no arithmetic operator anywhere in the failing statement's own source text. The
preceding call needn't touch the reference in question, or take any arguments at all.
Originally catalogued (without a root cause) as "Bug H" in issue #601, found while
auditing `stdlib/epistemic/ode.sio` (issue #580) — `rk4_step`/`rk45_step` increment a
`n_evals: &!i64` counter inline after each `ode_rhs_dispatch(...)` call, hitting this
bug at 10 sites, worked around by extracting the increment into a dedicated helper
function.

## Reproduction

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fresh.elf
cat > /tmp/repro.sio <<'EOF'
fn noop() -> i64 { 0 }
fn write_only(n: &!i64) with Mut {
    noop()
    *n = 999
}
fn main() -> i64 {
    var x: i64 = 5
    write_only(&!x)
    x
}
EOF
/tmp/lean_fresh.elf /tmp/repro.sio /tmp/out.elf
# error: arithmetic operands must have matching numeric types at <main>:4
```

## Bisection

Progressively isolated with minimal repros:

| Variant | Compiles? |
|---|---:|
| `*n = 999` as the function's *first* statement (no preceding call) | yes |
| `let v = *n` (read) before any call, then use `v` | yes |
| A plain `var` counter (`v = v + 1`, not a dereference) incremented repeatedly after preceding calls | yes |
| `*n = 999` after a preceding call, `n` read via `.1` tuple-field access instead of destructuring | n/a (unrelated axis) |
| `*n = 999` after `noop()` (no-arg call, discarded return) | **no** |
| `let discard = noop()` (bound, not discarded) then `*n = 999` | **no** |

The second and third rows initially suggested the bug was specific to *reading* a
dereferenced value after a call; the fourth row (binding the call's result to a `let`,
which still fails identically) shows it is not about discarding a call's return value —
**any** preceding expression-valued statement triggers it, as long as the *next*
statement's first token is `*`.

## Root cause

`compile_multiplicative()` (`self-hosted/compiler/lean_single.sio:15585`) — the
`*`/`/`/`%` binary-operator continuation loop — already contained a guard for exactly
this class of bug:

```sio
fn compile_multiplicative() with IO, Mut, Panic, Div {
    compile_postfix()
    var lhs_end_line: i64 = TL[(EP - 1) as usize]
    while TK[EP as usize] == 19 || TK[EP as usize] == 20 || TK[EP as usize] == 29 {
        if EXPR_TY == 11 && TK[EP as usize] == 19 && TL[EP as usize] > lhs_end_line { return }
        ...
```

`TK[EP as usize] == 19` is `*`; `TL[EP as usize] > lhs_end_line` checks whether that `*`
starts a *new source line* relative to the just-compiled left operand. The intent is
clearly "a `*` opening a new line is a new statement's dereference prefix, not continued
multiplication" — Sounio has no semicolons, so line position is the only signal
available to disambiguate `EXPR\n* IDENT = VAL` (two statements) from `EXPR *\nIDENT`
(one continued multiplication, a style nobody uses in this codebase — confirmed via a
repo-wide grep of every line-leading `*` in `self-hosted/` and `stdlib/`: all 47 hits are
dereference-assignments, zero are continued multiplications).

The guard was scoped to `EXPR_TY == 11` (raw pointer) only. For any *other* expression
type on the left (here, `noop()`'s `i64` return, EXPR_TY == 1), the guard never fires,
and the loop happily treats the following line's `*` as "continue multiplying": `noop()`
followed by `*n = 999` is absorbed as `noop() * n`, which then fails its own type check
(`i64 * &!i64` — a reference is not numeric) with "arithmetic operands must have
matching numeric types", misattributed to the dereference-assignment's line because that
is where the manufactured `*` token sits.

`compile_multiplicative_a64()` (line 31633, the aarch64 codegen twin) carries the
identical narrowly-scoped guard and the identical bug.

## Fix

Drop the `EXPR_TY == 11` restriction in both functions — a `*` starting a new line stops
multiplicative continuation unconditionally:

```sio
if TK[EP as usize] == 19 && TL[EP as usize] > lhs_end_line { return }
```

No other change. Verified safe against the existing codebase (no line-leading `*` in
`self-hosted/` or `stdlib/` is a continued multiplication) and against a full test suite
run (1311/1311, 0 regressions, 0 new known-failures).

## Scope note: a separate, more severe runtime bug found while validating this fix

**Not fixed here — flagged for a dedicated follow-up.** Independent of this parser fix,
a bare `*n = <literal>` where `n: &!i64` (a mutable reference to a *scalar*, not a
struct or array) appears to compile without error but **silently does not write anything
at runtime**, even as the *first statement of a function with no preceding call at all*:

```sio
fn write_only_baseline(n: &!i64) with Mut { *n = 999 }
fn main() -> i64 with IO {
    var x: i64 = 5
    write_only_baseline(&!x)
    println(x)   // prints 5, not 999 — on both the original compiler and this fix
    0
}
```

This reproduces identically on the unpatched compiler (i.e. it predates and is unrelated
to this dispatch's fix — this fix only removes a *false compile-time rejection*; it does
not newly enable, and does not newly break, the underlying store). The checker's own
source (`self-hosted/compiler/lean_single.sio`, the `stmt_is_deref_store` handler) only
emits the actual store instruction (`emit_store_to_pointer_offset_x86`) when
`VAR_TY[lvi] == 11` (raw pointer); for `VAR_TY[lvi] == 10` (the `&!T` reference case),
it falls through to `tc_error(name_tok, "dereference assignment requires raw pointer
binding")` — a **non-fatal warning** (`tc_error`, not `tc_error_hard`), so compilation
proceeds but no store code is ever emitted for the reference case. This means every
`*ref_param = value` pattern for a scalar `&!T` reference parameter in the existing
codebase (including the very `ode.sio` `bump_n_evals` helper this dispatch's own Bug H
repro was extracted from) may be silently writing nothing at runtime — not confirmed
either way here; this needs its own dedicated bisection and is out of scope for a
parser-continuation fix.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1311  Fail: 0  Known failures: 127  Skip: 689  Total: 2127
```

Also confirmed: normal same-line multiplication (`a * b`) unaffected. Madaros
(`self-hosted/compiler/main.sio` + module frontend, a fully separate source tree) does
not reproduce this bug and is untouched by this fix.

## Cross-references

- `docs/audit/MADAROS_NET_MOD_SIO_STANDALONE_CHECK_SILENT_FAIL_2026-07-01.md` — the
  running dispatch where Bug H was originally catalogued (2026-07-03 update) alongside
  checker bugs A-G, all pragmatically worked around at the stdlib level rather than
  fixed at the source until now.
- GitHub issue #601 — the consolidated tracking issue for checker bugs A-H; closes the
  Bug H item.
- `stdlib/epistemic/ode.sio` (`bump_n_evals`) — the stdlib workaround this fix makes
  unnecessary (left in place; the workaround is harmless and this dispatch does not
  revert it).
