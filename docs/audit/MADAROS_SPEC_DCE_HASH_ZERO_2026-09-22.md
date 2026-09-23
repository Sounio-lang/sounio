<!-- docs:meta
topic_id: repo.docs.audit.madaros-spec-dce-hash-zero-2026-09-22
authority: repo_only
audience: users
last_validated: 2026-09-22
validated_by: claude
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-spec-dce-hash-zero-2026-09-22
-->

# Madaros: cross-module DCE must not drop a function whose name hashes to 0

## Claim

> Under **current-source Madaros**, a top-level function is never dropped by
> cross-module dead-code elimination while a live call site to it still
> exists in the loaded program set -- including when the function's own name
> happens to hash to exactly 0 under `ast_name_hash`.

Before this change, a called function whose name hashed to 0 was silently
excluded from the reachability mark set and then deleted from the item list
by the filter, regardless of whether anything called it.

## Symptom

Single module, no imports, no collision, nothing generic:

```sio
fn fZOXITBAFRX_E() -> i64 {
    20
}

fn main() -> i64 with IO {
    print_int(fZOXITBAFRX_E())
    0
}
```

Compiled via `bin/madaros compile <file>.sio -o <out>.elf` (the wrapper CLI --
the raw ELF's `--native-compile` refuses single-module/no-import programs
outright: "native compile disabled for single-module streaming lane"; the
positional `madaros <src> -o <out>` form the wrapper actually uses does not
have that restriction and is what this gate and fixture use).

Before this fix, on unmodified `origin/main`:

```
error[E137] in <main>::main at 79..92: use of undeclared variable
   |
   = help: declare the variable before use, or import it from another module
   = name fZOXITBAFRX_E
imported_compile: lower_done
IR lowering failed during merge: epistemic_export_failed
Compilation failed!
```

Control (confirms this is specifically about the hash being 0, not this
string or this fn shape in general): the identical-shape program with the
function renamed to `twenty()` (`ast_name_hash("twenty") != 0`) compiles and
runs cleanly, printing `20`.

`ast_name_hash("fZOXITBAFRX_E") == 0` was confirmed both independently (a
standalone Python re-implementation of the djb2 algorithm --
`hash = 5381; hash = hash * 33 + byte`, clamped
`if hash < 0 { 0 - hash } else { hash }`, as defined in
`self-hosted/parser/ast.sio:1448`) and via this compiler's own `ast_name_hash`.

## Root cause

`self-hosted/check/specializer.sio`'s cross-module dead-code-elimination
reachability marker uses an open-addressing hash set, `marks: [i64; 16384]`
(`SPEC_DCE_SLOTS`), where an **empty slot** is represented by the sentinel
value `0`:

* `spec_dce_hash_find` (query side, ~line 1879): used `if h == 0 { return
  false }` as a fast path -- "a hash-0 name is never marked reachable",
  indistinguishable from the table's own empty-slot sentinel.
* `spec_dce_hash_insert` (~line 1899): used the identical `if h == 0 { return
  false }` guard, refusing to insert a hash-0 mark **at all**, before ever
  touching `marks[]`.

`ast_name_hash`'s own clamp (`if hash < 0 { 0 - hash } else { hash }`) makes
every real hash `>= 0`, so a legitimate name CAN land on exactly 0 -- and when
it does, `spec_dce_hash_insert` makes that name **permanently unmarkable**,
not merely at risk of a slot collision.

`spec_dce_scan_expr` (~lines 1928-1950) calls `spec_dce_hash_insert(marks,
count, ast_name_hash(callee))` whenever it walks a call, method-call, ident or
single-segment path expression, to mark the callee reachable. For a callee
whose name hashes to 0, this insert is a silent no-op: the mark set never
gains a mark for it, no matter how many live call sites exist.

`spec_dce_filter_items` (~line 2467) then drops a top-level `ItemFn` whenever
`spec_dce_hash_find(marks, count, ast_name_hash(it.name))` says "not marked" --
which is unconditionally true for a hash-0 name, live or not.

## Where this runs

Both `spec_dce_hash_insert` and `spec_dce_hash_find` back **every** DCE entry
point in this file, so the defect was not confined to one pipeline:

* `spec_dce_mark_across_programs` / `spec_dce_filter_with_global_marks`
  (~lines 2565, 2659) -- the ordinary path, driven from
  `module_frontend_lower_single_program_array_direct_box`
  (`self-hosted/compiler/module_frontend.sio:5569`) for a **single** module
  and from `module_frontend_lower_programs_array_direct_box_multi`
  (`module_frontend.sio:5644`, `:5668`, `:5714`) for **multiple** modules.
  This repro's single-module, no-import program goes through this path --
  contrary to what might be assumed, cross-module DCE is not gated on there
  being more than one module; it runs unconditionally whenever a program has
  a `main`.
* `spec_dce_unreachable_item_fns` (~line 2682) -- the specialized-collapse
  path, used when `module_frontend_specialized_prepare` merges every module
  into one item list because something in the program instantiates a
  generic.

A single fix to the two shared primitives therefore covers every caller;
there is no separate copy of this bug to chase down per pipeline.

### Why this fails closed here, and why that is not a guarantee

`module_frontend_lower_single_program_array_direct_box` re-typechecks the
DCE-filtered item list (`check_program_epistemic_into`) before lowering. With
`fZOXITBAFRX_E`'s `FnDef` removed from the list but the call to it still
present in `main`'s (unfiltered) body, that re-typecheck reports the call as
an undeclared variable, which is what surfaces as `error[E137]` and
`epistemic_export_failed` above -- a hard compile error, not a silently wrong
binary.

That is incidental to this program's exact shape, not a guarantee. The
sibling defect this one was found alongside -- module-aware private-function
identity, `self-hosted/compiler/private_fn_identity.sio`, PR #2598 --
hit the identical `spec_dce_hash_insert`/`spec_dce_hash_find` code through the
**multi-module merge** path (two modules, each privately defining
`fZOXITBAFRX_E`) and did not fail this gracefully: it crashed the compiler
process itself (measured `rc=139`, SIGSEGV) rather than emitting a clean
diagnostic. A differently shaped program -- the dangling call being itself
unreachable-but-syntactically-present in a way the re-check tolerates, or a
lowering path that skips the re-typecheck -- could plausibly let this
silently drop a real function instead of failing at all. This defect should
not be relied on as a general safety net for its own consequences.

## The fix

`self-hosted/check/specializer.sio`, `spec_dce_hash_insert` /
`spec_dce_hash_find`: store/compare `key = h + 1` in `marks[]` instead of the
raw hash `h`, and remove the `h == 0` early-outs entirely. Every real hash's
stored key is now `>= 1`; `0` stays an unambiguous "truly empty" sentinel for
every possible hash, including 0 itself.

This is the identical fix, to the identical bug class, already applied to a
*different* hash table in this codebase: `self-hosted/compiler/
private_fn_identity.sio`'s `pfi_census_note` / `pfi_census_mods` (2026-09-22,
PR #2598), which used the same `h == 0` shortcut against the same kind of
`0`-sentinel open-addressing table.

The capacity-refusal branch in `spec_dce_hash_insert` (`if *count >=
SPEC_DCE_MAX { return false }`) needed no change: it tests `*count`, not the
stored key, and is unaffected by the shift.

No other reader of `marks[]` / `SPEC_DCE_G_MARKS[]` makes the raw-hash-vs-0
assumption; `spec_dce_hash_insert` and `spec_dce_hash_find` are the only two
functions that read or write a hash value into the table.

## Evidence

Gate: `scripts/ci/madaros_spec_dce_hash_zero_gate.sh`, fixtures under
`tests/multimodule/spec_dce_hash_zero/`, wired into `.github/workflows/ci.yml`
immediately after the cross-module DCE reachability step (shared ELF via
`MADAROS_RAW_BIN`). Also added as `tests/run-pass/madaros_spec_dce_hash_zero.sio`
(`//@ requires: madaros`, `//@ expect-stdout: 20`), which
`scripts/ci/madaros_changed_tests_gate.sh` runs automatically against the
current-source compiler on any PR that touches it.

| check | baseline | fixed |
|---|---|---|
| `hashzero` (the reported repro), via `bin/madaros compile` (default ELF resolution, `artifacts/self-hosted/madaros`) | `error[E137]` / `epistemic_export_failed`, compiler rc=1, no ELF | compiles, runs, prints `20` |
| `hashzero`, raw ELF positional invocation against the committed `bin/madaros-linux-x86_64` | compiles ("Merged IR: 3 functions"), but `NATIVE_REFUSAL kind=empty_stub_ud2 fn=2 name=fZOXITBAFRX_E reason=missing_lowered_body`; the emitted ELF SIGILLs when run | compiles, runs, prints `20` |
| `control` (identical shape, non-zero hash) | compiles, runs, prints `20` | compiles, runs, prints `20` (unchanged -- isolates the trigger to the hash value) |

Rebuilt Madaros from this tree's patched `self-hosted/check/specializer.sio`
and reran the fixtures directly against two different pre-fix baseline
binaries -- `artifacts/self-hosted/madaros` (built 2026-09-19) and the older
committed `bin/madaros-linux-x86_64` (built 2026-09-11) -- and against the
rebuilt post-fix ELF, confirming the before/after rows above on a real
compile, not a source-level argument alone. The two pre-fix binaries fail
**differently** (a clean compile-time diagnostic vs. an emitted ELF that
traps at runtime) despite sharing the same root cause: exactly the "not a
general safety net" concern above, now with a second, concretely observed
failure shape rather than only a hypothetical one. `scripts/ci/
madaros_spec_dce_hash_zero_gate.sh` fails red against both pre-fix binaries
and passes green against the post-fix one.

## Related

* Module-aware private-function identity, `self-hosted/compiler/
  private_fn_identity.sio` (PR #2598, `docs/audit/
  MADAROS_PRIVATE_FN_IDENTITY_2026-09-21.md` once merged) -- the sibling
  defect (identical bug class, different hash table, `private_fn_identity.sio`'s
  census) whose verification surfaced this one. Unrelated to and not blocked
  by that change; this defect predates it and is reachable with
  `private_fn_identity.sio` entirely out of the picture (a single module
  never reaches that pass's `count < 2` guard).
