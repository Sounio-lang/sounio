<!-- docs:meta
topic_id: repo.docs.audit.lean-single-multisegment-qualified-call-2026-07-04
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-multisegment-qualified-call-2026-07-04
-->

# lean_single forensic dispatch — multi-segment `::`-qualified calls were never resolved, only ever silently miscompiled

Date: 2026-07-04
Branch: `main` @ `6dcb3c953`
Class: **checker/parser gap** (a whole class of valid-looking call syntax was
never implemented; the compiler silently emitted `0` instead of erroring or
resolving) — root-causes and closes issue #601's "Bug A"
Status: root-caused, fixed, verified (full test suite 1310 pass / 0 fail / 128
known failures / 689 skip, net +9 tests now genuinely verified, +1 known
failure — see "Verification" for the reconciliation)

## Summary — this supersedes issue #601's original Bug A framing

Issue #601 catalogued "Bug A" as: *inline fully-qualified calls need ≥3 path
segments* — i.e. `pkg::file::fn()` (one `::`) fails with "unknown identifier",
while `pkg::sub::file::fn()` (two or more `::`) "resolves fine", citing
`distributed::pure::types::registry_new()` as a working example.

That framing does not survive controlled testing. **Every `::`-qualified call
with 2+ `::` tokens silently miscompiles to a stub `0` return, regardless of
segment count** — the "resolves fine" claim was based on absence of a compile
error, not on a runtime-correctness check. The one real-world citation
(`distributed::pure::types::registry_new()`, in
`tests/stdlib/distributed/test_distributed_core.sio`) is a `//@ check-only`
test: compiled, never executed. A repo-wide census of every 2+-`::` qualified
*call* in executable code position (excluding comments/strings) found no
executed, CI-tracked test anywhere that verifies such a call's return value —
the one file that tried (`tests/stdlib_database/test_database_e2e.sio`,
`//@ run-pass`) lives in a directory `run_sio_test_suite.sh` never globs and
independently fails to compile on the unrelated, already-catalogued issue
#601 Bug B.

## Reproduction (controlled — file loaded, syntax varied)

The single variable that matters is whether the target file is loaded into
the compilation unit at all (`resolve_imports()` only scans literal `use `
lines; a bare qualified call never triggers a load on its own). Holding that
constant via `use pkg::mod::*` isolates the actual defect:

```sio
// stdlib/bugatest2seg/foo.sio
module bugatest2seg::foo
pub fn bar() -> i64 { 42 }
```

```sio
use bugatest2seg::foo::*
fn main() -> i64 with IO {
    println(bugatest2seg::foo::bar())  // fully qualified, 2 `::`
    println(bar())                     // bare call, workaround idiom
    0
}
```

Pre-fix result: `0` then `42` — the qualified-call syntax silently resolves
to a stub, the bare call (the codebase's actual idiom) resolves correctly.
Identical result for a 3-`::` path (`bugatest3seg::sub::foo::bar()`).

Against real stdlib code (the exact call issue #601 cited), instrumented to
print rather than merely check for a compile error:

```sio
use distributed::pure::types::*
fn main() -> i64 with IO, Mut {
    var r = distributed::pure::types::registry_new()
    println(distributed::pure::types::registry_size(&r))
    // ... add two nodes via distributed::pure::types::registry_add ...
    println(distributed::pure::types::registry_size(&r))
    0
}
```

Pre-fix: prints `0`, `0` — `registry_size` never actually runs; the "add"
calls are likewise stubs. `assert_registry()`'s check-only harness happened
to type-check regardless, because it never runs to observe the wrong value.

## Root cause

`self-hosted/compiler/lean_single.sio`'s `compile_primary()` (x86-64,
originally starting at line 10957) and `compile_primary_a64()` (aarch64 twin,
originally at line 30003) only ever understood **one** `::` level following
an identifier:

- `Enum::Variant` — look up `en_variant_value(hash(Enum), hash(Variant))`
- `Type::method()` — if the token after the second segment is `(`, treat it
  as a static method call via `fn_find_method`

For a chain of 2+ `::` (`a::b::c()`), the first segment ("a") is neither a
registered enum nor immediately followed by `(` after its variant name ("b")
— the next token is another `::`, not `(` — so `looks_like_method_call` is
false. Execution falls into the final catch-all:

```sio
// disc < 0, not a method call — unregistered enum variant: consume :: + name to prevent cascade
EP = EP + 1  // skip ::
EP = EP + 1  // skip variant name
em(0x48); em(0xb8); em64(0)
EXPR_IS_F64 = 0; EXPR_TY = 0; EXPR_TY_HASH = 0
return
```

This emits a hardcoded `0` and **returns having consumed only the first
`::segment` pair** — the remaining `::c(...)` tokens are left in the stream
for whatever surrounding syntax happens to follow, which is why observed
behaviour was inconsistent across contexts (a bare `println(call())` argument
position versus an `if call() != N` condition) during initial investigation.
The aarch64 twin is even narrower — it has no `Type::method()` handling at
all and unconditionally treats any single `::` as an enum-variant lookup,
falling back to the same "emit 0, consume only one segment" behaviour on
failure.

Sounio's function namespace is flat: `fn_find(name)` looks up by bare name
only, populated by whichever files `resolve_imports()` loaded via `use
mod::*`. There was never a mechanism to resolve a *chain* of module-path
segments down to a callable — the only working call idiom has always been
`use pkg::mod::*` followed by a bare call, which is what every non-test
caller in the codebase already does. Multi-segment qualified call syntax was
simply never wired up, on either backend.

## Fix

In both `compile_primary()` and `compile_primary_a64()`, immediately before
the existing single-`::` (`Enum::Variant` / `Type::method()`) handling, flatten
a genuine 2+-`::` chain down to its terminal segment and fall through to the
ordinary identifier/call resolution that already exists below (the same path
a bare call takes):

```sio
var mseg_flattened: i64 = 0
while TK[EP as usize] == 51 && EP + 2 < TC && TK[(EP + 2) as usize] == 51 {
    EP = EP + 1  // skip ::
    ns = TS[EP as usize]
    ne = TE[EP as usize]
    EP = EP + 1  // skip segment name
    mseg_flattened = 1
}
if mseg_flattened == 1 && TK[EP as usize] == 51 {
    EP = EP + 1  // skip ::
    ns = TS[EP as usize]
    ne = TE[EP as usize]
    EP = EP + 1  // skip terminal segment name
}
```

The loop only advances while a *further* `::` follows the segment about to be
consumed (`TK[EP+2] == 51`), so it never fires for a genuine single `::`
(`Color::Red`, `Type::method()`) — `mseg_flattened` stays `0` and the
existing enum/method-call code runs completely untouched. Only when the loop
has proven a 2+-`::` chain does the final `if` consume the last `::segment`,
leaving `ns`/`ne` on the terminal identifier and `EP` positioned exactly as
if that name had been written bare — the same state the pre-existing
`fn_find` / bare-call codegen below already handles correctly.

This is a syntactic normalization, not new call semantics: `a::b::c::fn()`
now means "resolve `fn` by its flat, already-`use`-loaded name" — identical
to what `use a::b::c::*; fn()` already does. It does not implement true
nested-module name resolution (a qualified path whose *middle* segment is
itself a real struct/enum type with its own static methods is not
special-cased); no such shape exists in the current codebase's 2+-`::` call
sites, all of which are `module::submodule::free_function()`.

## Consequence: a masked bug (#601 Bug B) surfaced as a byproduct

Fixing Bug A causes multi-segment calls to actually reach `fn_find` and
arity-checking for the first time. One stdlib test,
`tests/stdlib/database/test_database_core.sio`, calls
`database::pure::engine::engine_create_table(&!db, "users")`, where
`engine_create_table`'s parameter type is itself a qualified path
(`db: &!database::pure::types::InMemoryDB`). That combination triggers issue
#601's independently-catalogued **Bug B** ("qualified-path param type + glob
import → false arity mismatch") — previously invisible because the call
never got past Bug A's silent-`0` stub to reach arity-checking at all. This
is not a regression introduced by this fix; it is a pre-existing, separately
tracked defect that this fix unmasks. The test is marked
`//@ known-failure: issue #601 Bug B ...` rather than left as an unexplained
regression, pending Bug B's own fix.

## Test changes

Nine `//@ check-only` stdlib tests called 2+-`::`-segment qualified functions
with **no `use` statement at all** — the target files were never loaded, so
after this fix (which routes multi-segment calls through the same
`fn_find`-based resolution as bare calls) they correctly report "unknown
identifier" instead of silently compiling to a stub. This is the intended,
honest behaviour: a call to a symbol that was never imported should fail
loudly, not return `0`. Fixed by adding the missing `use pkg::mod::*`
statements (the same fix pattern as issue #614/PR #615), matching how every
executed caller in the codebase already loads its dependencies:

- `tests/stdlib/cache/test_cache_core.sio`
- `tests/stdlib/cli/test_cli_core.sio`
- `tests/stdlib/database/test_database_core.sio` (also newly marked
  known-failure per the Bug B interaction above)
- `tests/stdlib/distributed/test_distributed_core.sio`
- `tests/stdlib/genomics/test_fasta_parse.sio`
- `tests/stdlib/genomics/test_gf4_gpu_e2e.sio`
- `tests/stdlib/image/test_image_core.sio`
- `tests/stdlib/infra/test_infra_core.sio`
- `tests/stdlib/queue/test_queue_core.sio`
- `tests/stdlib/wasm/test_wasm_core.sio`

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1310  Fail: 0  Known failures: 128  Skip: 689  Total: 2127
```

Reconciles against the pre-fix baseline (1311 pass / 0 fail / 127 known
failures / 689 skip / 2127 total): the 9 stdlib tests above previously
counted as "pass" only because their check-only harnesses never observed the
silent-`0` miscompile; they now genuinely execute their qualified calls and
still pass. `test_database_core.sio` moves from "pass" (silently, via the
same masking) to "known failure" (Bug B, tracked separately) — net
128 = 127 + 1, net pass count 1310 = 1311 − 1, zero unexplained deltas.

Also confirmed directly: `use distributed::pure::types::*` +
`distributed::pure::types::registry_size(&r)` now correctly prints `0`, `1`,
`2` across three registry states (previously stuck at `0`, `0`, `0`); the
original loud "unknown identifier" case (2-segment call, target file never
loaded via any `use`) is unaffected — still errors identically before and
after this fix, since flattening only changes which identifier is looked up,
not whether it must first be loaded.

## Cross-references

- `docs/audit/LEAN_SINGLE_MULTIPLICATIVE_LINE_BOUNDARY_2026-07-04.md` — Bug H
  (PR #619), same discovery week, independent bug.
- `docs/audit/LEAN_SINGLE_SCALAR_REF_DEREF_STORE_2026-07-04.md` — issue #620
  (PR #623), same discovery week, independent bug.
- GitHub issue #601 — tracks Bug A (closed by this fix) and Bug B (still
  open, unmasked but not fixed by this dispatch).
