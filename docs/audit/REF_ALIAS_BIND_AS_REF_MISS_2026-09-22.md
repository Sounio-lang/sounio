<!-- docs:meta
topic_id: repo.docs.audit.ref-alias-bind-as-ref-miss-2026-09-22
authority: repo_only
audience: users
last_validated: 2026-09-23
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ref-alias-bind-as-ref-miss-2026-09-22
-->

# AUDIT: bare-identifier ref-alias loses `is_ref` — `let zs = a` silently reads/writes through the wrong indirection — 2026-09-22

- Author: Claude, investigating a Copilot review comment on
  [PR #2615](https://github.com/Sounio-lang/sounio/pull/2615#discussion_r4075442769)
  (full investigation in the reply at
  [r4075706212](https://github.com/Sounio-lang/sounio/pull/2615#discussion_r4075706212)).
- Found on `lane/cursor-1/20260826` while replying to the PR #2615 review
  thread; confirmed on unmodified `origin/main` at
  `3a50ccec92372c0b68ec623b31e34ea44ebbdb89` — **not new, not f128-specific,
  and not introduced by PR #2615 or its predecessor.**
- This fix branch (`fix/ref-alias-bind-as-ref-miss-20260922`) is cut from
  `origin/main` directly, independent of both `lane/cursor-1/20260826`'s
  unrelated in-flight `lower.sio` refactor and PR #2615's branch.
- Compiler: Madaros (self-hosted modular compiler,
  `self-hosted/compiler/main.sio` / `self-hosted/ir/lower.sio`)

## Symptom

```sio
fn read_through_alias(a: &[i64; 2]) -> i64 with Mut, Panic, Div, IO {
    let zs = a
    zs[0]
}
```

Called with `xs = [42, 99]` via `read_through_alias(&xs)`, `zs[0]` reads a
garbage value (an unrelated stack address, ~1.4e14) instead of `42`. The bug
reproduces for any `let`/`var` binding whose right-hand side is a bare
identifier that is *itself* already reference-typed
(`&T` / `&!T` / `&[T;N]` / `&![T;N]`), with no explicit type annotation on the
new binding and no `&`/`&!` written at the alias site.

## Root cause

`self-hosted/ir/lower.sio`, `Lowerer.lower_let_stmt_finalize_ref`. Whether a
`let`/`var` binding is marked as holding a reference (`is_ref`, consumed by
every `lookup_local_is_ref` call site — see below) is decided by:

```sio
var bind_as_ref = lower_opt_type_is_ref_like(&s.ty)   // explicit `let zs: &T = ...`
if !bind_as_ref {
    match s.expr {
        Some(rhs_ref) => {
            if (*rhs_ref).kind == ExprKind::ExprUnary {
                if (*rhs_ref).un_op == UnaryOp::OpRef || (*rhs_ref).un_op == UnaryOp::OpRefMut {
                    bind_as_ref = true            // RHS is literally `&x` / `&!x`
                }
            }
        }
        _ => {}
    }
}
```

Neither branch covers `let zs = a` where `a` is a bare `ExprIdent` that is
itself already reference-typed. `zs` is bound to a plain `ir_copy` of `a`'s
value register — which correctly holds the reference/address value — but
`is_ref` is never set on `zs`. Every later access through `zs` then treats
that address as an ordinary GC handle instead of dereferencing it first,
reading (or writing) through the wrong indirection level.

### Every `lookup_local_is_ref` consumer this reaches

`grep -c lookup_local_is_ref self-hosted/ir/lower.sio` finds 11 consumer call
sites as of this fix (excluding the function's own definition and the fix's
capture call — see **Fix** below). Each reads `is_ref` off the *base*
identifier of an expression, so every one of them silently mis-lowered when
that base was an unannotated bare-identifier alias:

| Site (pre-fix) | What breaks |
|---|---|
| `lower_index_expr_ref` | `zs[i]` read — wrong indirection (the reported repro) |
| index-store path in `lower_assign_stmt_ref` | `zs[i] = v` write — wrong indirection |
| field-get path in the field-access lowering fn | `zs.field` read on an aliased struct pointer |
| field-store path in `lower_assign_stmt_ref` | `zs.field = v` write on an aliased struct pointer |
| `lower_expr_as_array_handle_ref` | passing `zs` as a GC-array-handle builtin arg (`str_from_bytes`, `write_file`) |
| `lower_first_arg_is_ref_like_ref` | packing `zs` as a first call argument before a C `char*` builtin (KL-14a) |
| `lower_seq_recv_base_ref` | `zs.len()` / `zs.get(i)` etc. on an aliased `&Seq<T>` receiver |
| `.len()` receiver check | `zs.len()` on an aliased array/slice |
| `&self` auto-ref receiver check | `zs.method()` where the callee's first param is `&self` |
| `lower_for_in_array_ref` | `for v in zs { ... }` — silently falls through to the plain local-array desugar instead of correctly refusing/handling the ref case |
| `struct_local_is_value_copyable` | `var y = zs` on a struct-typed alias — wrongly treated as value-copyable instead of aliasing |

## Scope

- Affects both `let` and `var` — both statement kinds route through the same
  `lower_let_stmt_ref` → `lower_let_stmt_finalize_ref` path
  (`self-hosted/ir/lower.sio`, dispatch at `lower_stmt_ref`).
- Affects fixed-array references (`&[T;N]` / `&![T;N]`, the reported case)
  and struct-pointer references (`&S` / `&!S`) identically — both go through
  the same `bind_as_ref` computation, and both the field-access and
  index-access `lookup_local_is_ref` sites are separately confirmed above.
- Transitive aliasing (`let ws = zs` after `let zs = a`) is fixed
  automatically once the direct case is: `lookup_local_is_ref` for `zs` now
  returns `true`, so the same bare-identifier check on `ws`'s binding picks
  it up.
- Closures/captures are **not** affected by this bug: a captured variable
  becomes a synthetic function parameter with an explicit declared type,
  which was already handled correctly ("Params already get this from their
  type" — see the comment above `bind_as_ref` in `lower_let_stmt_finalize_ref`).
  Sounio has no nested-fn-declaration form separate from closures, so this
  covers the full "nested function" surface.
- **Relationship to PR #2615's SIGSEGV fix:** PR #2615 fixes a *different*
  bug where `a[i]` accessed **directly** (not through an alias) SIGSEGV'd for
  `a: &[f128;N]` params. On the base this PR is built from (`origin/main` at
  `e6cc1e4bf1`), that direct-access SIGSEGV is **still present** — confirmed
  empirically (see **f128 exclusion** below) — so PR #2615 has not landed on
  this base yet. Once it does, `let zs = a; zs[0]` for `a: &[f128;N]` should
  reach the alias path and become observable exactly as described in the
  original report: for every element type *unaffected* by that SIGSEGV, this
  alias bug was already silently returning wrong data with no crash and no
  diagnostic, with no known trigger date. `&[f128;N]` itself is a documented
  exception to that general shape — see below.

## Fix

`self-hosted/ir/lower.sio`, `lower_let_stmt_finalize_ref`: added a third
branch to the `bind_as_ref` computation — a bare-identifier RHS that is
itself already reference-typed now also sets `bind_as_ref = true`. This
reuses the existing `bind_local_ref` mechanism (a simple flag set on the
local-stack slot; see the exclusion comment above
`struct_local_is_value_copyable` — "`is_ref` — `&T` / `&!T` locals: aliasing
is what a reference MEANS") — no new representation, no new call sites, and
every consumer above now sees the correct bit without further changes.

**Same-name shadowing correction (post-review):** the first version of this
fix queried `lookup_local_is_ref` on the RHS name from *inside*
`lower_let_stmt_finalize_ref`, which runs after `lower_let_stmt_after_expr_ref`
has already called `bind_local(s.name, ...)` to create the destination's own
local-stack slot. `lookup_local_is_ref` searches newest-first, so for
same-name shadowing (`let a = a`) that query found the just-created
destination slot (always `is_ref = 0` at that point) instead of the outer,
still-in-scope `a` it needed to read — silently losing `is_ref` on exactly
the shadowing form. Fixed by capturing the RHS identifier's `is_ref` status
in `lower_let_stmt_after_expr_ref` *before* its `bind_local(s.name, ...)`
call, and threading that captured `bool` into `lower_let_stmt_finalize_ref`
as a parameter instead of re-querying after the rebind.

**`&[f128;N]` exclusion (post-review):** a second Copilot finding on the same
commit caught a distinct hazard: V0-E.5.7 force-copies every `&[f128;N]` /
`&![f128;N]` param into a fresh, real GC array handle regardless of the `&`
annotation (`lower_fn_params_ref`: `param_word_scalar` is true for any f128
array type, ref or not — `array_copy_word_scalar[a] = 1`), while `a` is
*simultaneously* stamped `is_ref = 1` from its declared reference type. For
this one case, `a`'s own local-stack `is_ref` bit does **not** describe what
`a`'s register holds (a value-copied handle, not a caller address) — the
`array_copy_word_scalar` bit is what actually governs it. Propagating
`is_ref` to `zs` in `let zs = a` would route `zs[i]` through the
ref-array/deref path (`label_id = 1`) onto a register holding a handle, not
an address.

Fixed by excluding sources that are both `is_ref` and `array_copy_word_scalar`
from the alias propagation: `rhs_ident_is_ref` in
`lower_let_stmt_after_expr_ref` is now
`lo.lookup_local_is_ref(name) && !lo.lookup_local_array_copy_word_scalar(name)`.

This exclusion is mechanically necessary but **not sufficient** for correct
*values* through `let zs = a; zs[i]` on `&[f128;N]` — empirical testing
(minimal probes, not checked in) found:

- Direct, non-aliased `a[0] + a[1] + a[2]` for `a: &[f128;3]` **already
  SIGSEGVs on this PR's unmodified base** (confirms the PR #2615 relationship
  above; out of scope here).
- With the exclusion applied, `let zs = a` (the bind alone, no indexing)
  no longer crashes — confirming the exclusion is doing its job.
- `zs[i]` **still SIGSEGVs** after the exclusion, from a second, deeper,
  separate gap: `lower_let_stmt_after_expr_ref`'s own word-scalar re-copy for
  the alias (`fixed_array_len >= 0 && fixed_array_word_scalar` →
  `emit_fixed_array_value_copy` again) produces a local that is
  `array_copy_word_scalar = 1` and `array_elem_wide_bits = 128` with
  `is_ref = 0` — a combination that, before this fix, only ever arose
  together with `is_ref = 1` on the original ref param, and appears to be
  unhandled by whatever `zs[i]` indexing path is reached with `is_ref = 0`.
  This reaches into `&[f128;N]` support's own machinery in a way that is
  outside this PR's scope (a bare-identifier alias bug fix) to chase down
  and fix blindly — doing so without the same exhaustive verification this
  PR received would risk a *new* miscompile in an actively-developed feature
  area, which is exactly the failure mode this whole investigation exists to
  avoid repeating.

**Net effect:** the exclusion is still the correct and necessary fix for the
`is_ref` propagation specifically (it removes one confirmed way `zs[i]`
could misread `a`'s value), but a fully correct `let zs = a; zs[i]` for
`&[f128;N]` remains blocked on the pre-existing direct-access bug (and,
per the above, possibly an additional array-copy/wide-bits interaction) —
neither of which this PR introduces or can responsibly resolve as a side
effect. Tracked as follow-up work, likely intersecting with PR #2615's
territory. No regression test for this exact scenario is included for that
reason — see **Regression coverage** below for what *is* covered and why.

## Regression coverage

`tests/run-pass/ref_alias_bind_as_ref.sio` (`//@ requires: madaros` — the fix
lives only in the self-hosted Madaros compiler, not the lean_single
bootstrap that builds the default suite's stage2 binary) — covers, all
through a bare-identifier alias with no explicit type annotation:

1. read through an alias of an immutable ref array param (`&[i64;2]`, the
   reported repro)
2. write through an alias of a mutable ref array param (`&![i64;3]`)
3. read through an alias of a non-array struct reference (`&Pair`)
4. write through an alias of a mutable struct reference (`&!Pair`)
5. read through a same-name shadowing alias (`let a = a`) of an immutable ref
   array param — the case that motivated the capture-before-bind correction
   above

None of these five sources are `array_copy_word_scalar` (that bit is
currently only forced for f128-element arrays), so the `&[f128;N]` exclusion
above is a no-op for all of them — re-verified after adding it: all five
still pass unchanged. The exclusion itself has no positive-value regression
test, for the reasons given above.

Wired into CI as `scripts/ci/madaros_ref_alias_bind_as_ref_gate.sh`, added to
the `madaros-witness-gate` job in `.github/workflows/ci.yml` right after
"Imported Seq lowering against the fresh build", reusing that job's single
`/tmp/madaros-ci.elf` build (`bash scripts/ci/build_modular_madaros.sh
/tmp/madaros-ci.elf`) instead of triggering its own. The gate script itself
also supports building its own compiler when `MADAROS_RAW_BIN` is unset,
following the build-then-check-then-compile-then-run convention used by
`scripts/ci/madaros_fixed_array_call_boundary_alias_gate.sh` and
`scripts/ci/madaros_f128_f256_v0e_surface_gate.sh`.

Corpus-scanned every `tests/`, `stdlib/`, and `self-hosted/` `.sio` file for
existing occurrences of the buggy pattern (bare-identifier `let`/`var` alias
of a `&`/`&!`-typed param or local, subsequently accessed through implicit
deref). None found: the only alias-of-ref idiom already in the corpus is
explicit-deref linked-list traversal (`var cur = <ref param>; match *cur {
... }`), which never goes through the affected `lookup_local_is_ref` call
sites. The fix changes no existing test's behavior.

## Verification

Built fresh via `scripts/ci/build_modular_madaros.sh` (current source,
behind `scripts/dev/souc-build-lock.sh`'s global lock) rather than trusting
`souc check` alone, since the fix is in the self-hosted compiler's own
source. Verified independently twice: once while developing the fix on
`lane/cursor-1/20260826`, and again from this fix's actual base
(`origin/main` at `e6cc1e4bf1`, this branch's parent) to confirm the fix
holds with no dependency on that branch's unrelated in-flight `lower.sio`
refactor.

Build from `origin/main` (this branch's actual base):

```
compile: fns=11664 code=125069593 bytes main=fn1465 patches=69319
elf: 125073689 bytes (bss=4388777192)
```

`MADAROS_RAW_BIN=<that ELF> bash scripts/ci/madaros_ref_alias_bind_as_ref_gate.sh`:

```
compiler_sha256=a0542301c5c441f2c03f6e6c5098875544c58c826042555acbaa69e6d9c5233b
source_sha256=59bac88375f614223e5694ee0574f0c085f8f21db29d3063e76c81efc0da30de
ref_alias_bind_as_ref: PASS
PASS: read/write through a bare-identifier alias of a &T / &!T / &![T;N] param
stays reference-typed (IndexGet/Set + FieldGet/Set)
```

Re-verified a third time from a fresh build after adding the `&[f128;N]`
exclusion (`compiler_sha256` above is that build). Also ran, against that
same build, with `ulimit -s unlimited` (stack size is a separate, known
confound for several unrelated f128 gates on this codebase — see
`ulimit -s 1048576` guards throughout `.github/workflows/ci.yml`'s Madaros
jobs) three targeted probes (not checked in — see **`&[f128;N]` exclusion**
above for what they showed): the alias *bind* `let zs = a` alone no longer
crashes; `zs[i]` still does, from a separate pre-existing gap; direct
`a[i]` (no alias) already crashes on this unmodified base independent of
this PR.

All five cases in `tests/run-pass/ref_alias_bind_as_ref.sio` (read/write
through a fixed-array-ref alias, read/write through a struct-ref alias, and
read through a same-name shadowing alias) pass against a Madaros build
compiled from the fixed source — re-verified after the capture-before-bind
correction above.
