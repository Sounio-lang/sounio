<!-- docs:meta
topic_id: repo.docs.audit.lean-single-use-as-alias-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-use-as-alias-2026-07-05
-->

# lean_single forensic dispatch — `use pkg::mod as alias` + single-`::` call resolution

Date: 2026-07-05
Branch: `main` (post-PR #630, Bug G — closing issue #601's full A–H campaign)
Class: **two independent defects, fixed together because the feature only
works end-to-end with both fixed** — closes issues #632 and #633's sibling
(#633 is unrelated, see that issue) — specifically closes **#632**
Status: root-caused, fixed, verified (full test suite 1314 pass / 0 fail /
124 known failures / 689 skip / 2127 total — exact match to the current
baseline, zero regressions)

## Summary

Issue #632 as filed described one defect: `use pkg::mod as alias` silently
drops spaces during import-path construction, building a garbled filesystem
path (`pkg/modasalias.sio`) that never exists, so the module never loads.
That defect is real and is fixed here. But investigation found that fixing
it alone does not make the feature work: a second, independent, and more
severe gap exists in how the compiler resolves the resulting `alias::fn()`
call. This was surfaced to the user before any fix was written (see
"Design decision" below) and the user chose to fix both together, since
there is no partial fix that delivers working behavior on its own.

## Defect 1: import-path construction swallows spaces (issue #632 as filed)

`self-hosted/compiler/lean_single.sio`'s `resolve_imports()` builds a `use`
statement's target filesystem path by scanning bytes until a newline,
`;`, or `::*`/`::{` marker — nothing else was treated as a path terminator.
The path-*building* loop right after it explicitly **skipped** spaces
(`c == 32`) rather than stopping at them, on the assumption a module path
never contains one. `use pkg::mod as alias` has no `::*`/`::{` marker before
end-of-line, so the whole line `pkg::mod as alias` was scanned as one path,
producing `pkg/modasalias.sio` (spaces dropped, `::`→`/`) — a file that
never exists.

**Fix**: the path-extraction loop now recognizes a literal `" as "` byte
sequence as an unambiguous path terminator (a module path never legitimately
contains a space — confirmed by census, see below) and stops there,
disabling the unrelated Bug C segment-stripping fallback for this case since
`as`-aliasing always names a whole module, never "the last segment is
actually an exported symbol." `use pkg::mod as alias` now correctly resolves
`pkg/mod.sio`.

## Defect 2 (discovered, not originally filed): single-`::` calls never resolve a plain function

Fixing defect 1 alone was not sufficient: `alias::item()` still silently
returned `0` afterward, with no error. Investigation found this reproduces
identically for **any** single-`::` call to a plain (non-enum, non-struct)
module — not alias-specific at all:

```sio
use pkg::mod            // no alias, bare whole-file import
fn main() -> i64 { mod::item() }   // returns 0, not item()'s real value
```

`compile_primary()`'s qualified-call handling has two distinct mechanisms:
- **2+ `::` chains** (`pkg::sub::fn()`) are flattened to their terminal
  segment and resolved via an ordinary bare `fn_find()` (the fix for issue
  #601's Bug A) — Sounio's function namespace is flat, so a multi-segment
  prefix is purely decorative.
- **A single `::`** (`X::fn()`) never went through that flatten step. It was
  always treated as either an `Enum::Variant` reference or a `Type::method()`
  static method call, keyed by hashing `X` as a type/enum name. When `X`
  doesn't match any real enum or struct — which is exactly what a module
  alias is — the code silently emitted a stub `0` and returned, with no
  fallback to the same "decorative prefix, resolve the bare name" rule the
  2+-`::` case already uses. There is, and never was, a dedicated
  module-alias table anywhere in the compiler.

**Design decision** (put to the user before writing this fix, since it
changes what every single-`::` expression means for an unrecognized prefix):
should `X::fn()` fall through to a bare `fn_find()` on `fn`, exactly like the
2+-`::` case, when `X` isn't a real enum/struct? **Approved.** This needed a
decision rather than a unilateral fix because it is a language-semantics
change to an existing, exercised code path (every enum-variant and
static-method call in the language goes through this same function), not a
narrowly-scoped bug fix.

**Census before fixing** (per the same discipline as issue #601's Bug A —
verify-before-fixing, not synthetic repros in isolation): `grep -rn "use .*
as " stdlib/ examples/ tests/ self-hosted/` found exactly **one** real
`use ... as` usage in the entire codebase —
`stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio:40:
use chemistry::ontology as pbpk_chemistry_ontology;`, consumed as
`pbpk_chemistry_ontology::rapamycin_chebi()` and
`pbpk_chemistry_ontology::cyp3a4_metabolism_iri()` — both **function calls**,
the exact shape this fix addresses. That file does not currently compile at
all (fails on defect 1 first), so nothing today depends on the pre-fix
behavior of either defect.

**Fix** (`compile_primary()`, x86-64): when the single-`::` static-method
lookup (`fn_find_method`) fails to find a match, instead of draining the
call's `()` and stubbing a `0`, treat the prefix as decorative: reassign
`ns`/`ne` to the already-consumed terminal segment (`EP` is already
correctly positioned at the call's `(`) and let control fall through to the
ordinary bare-call resolution code that already exists just below (the same
code the 2+-`::` flatten path already relies on). A new `mcall_fallthrough`
flag guards the enclosing block's original "not a method call" stub so it
still fires, unchanged, for the non-call shape (`X::something` with no `(`
following) and for the genuine 2+-`::` case (unaffected, already handled
earlier). The `Type::method()` and `Enum::Variant` success paths are
untouched.

**aarch64 twin** (`compile_primary_a64()`): a smaller, analogous change.
Unlike x86, a64's single-`::` handling never had a `Type::method()` branch
at all — it unconditionally treated any single `::` as an `Enum::Variant`
lookup, stubbing `0` on a miss regardless of whether a call followed. Added
the same "if a call follows and it isn't a real enum variant, fall through
to bare resolution" branch. a64 static-method calls remain entirely
unsupported — a separate, larger, pre-existing gap not addressed here (this
only adds the "unrecognized single-`::` prefix in call position resolves
like the 2+-`::` case" behavior, for parity with x86 on this specific
shape).

## Incidental effect: a previously-silent bug now surfaces as a compile error

`Point::totally_fake_method()` for a real struct `Point` with no such
method — previously **silently compiled to a stub returning `0`, with no
diagnostic at all**. After this fix, the fallthrough reaches ordinary
bare-call resolution, which fails to find `totally_fake_method` and reports
a compile error (`error: unknown identifier` — the exact wording references
the outer name-token span, which reads a little oddly for this repurposed
path, but the failure is now loud rather than silent). This is a strict
correctness improvement — a typo'd static method call is no longer a silent
zero-stub — confirmed not to regress any currently-passing test.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Exact match to the current baseline — zero regressions.

Directly confirmed by runtime value:
- Issue #632's exact repro (`use pkg::mod as alias; alias::item()`): compiles
  and returns the real value (was a hard "unreadable import" error).
- `mod::item()` (single-`::`, no alias, real module): returns the real value
  (was `0`).
- `totallybogus::item()` (garbage prefix, real function, no alias at all):
  now also correctly resolves to the real value — confirming the fallthrough
  is general, not alias-specific.
- Real enum-variant matching (`Color::Green` in a `match`) and real
  static-method calls (`Point::origin()`): unaffected, unchanged codegen
  path.
- `stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio`: the
  `use chemistry::ontology as pbpk_chemistry_ontology;` import and both
  `pbpk_chemistry_ontology::...()` calls (lines 117–118) no longer error.
  The file still fails to compile end-to-end due to unrelated,
  pre-existing defects at lines 315/334 (arity mismatch, type mismatch) —
  out of scope for this dispatch, not touched.

## Cross-references

- GitHub issue #632 — closed by this fix.
- `docs/audit/LEAN_SINGLE_NAMED_USE_IMPORT_2026-07-05.md` — Bug C (PR #626),
  where the `use ... as alias` variant was originally noted as a follow-up
  and where `is_named_import`'s segment-stripping fallback (left disabled
  for the `as`-aliased case here) was introduced.
- `docs/audit/LEAN_SINGLE_MULTISEGMENT_QUALIFIED_CALL_2026-07-04.md` — Bug A
  (PR #624), the 2+-`::` flatten-to-terminal-segment fix this dispatch
  extends the same rationale to for the single-`::` case.
- GitHub issue #633 — the other untracked gap from the same investigation
  batch (`println(&str)`), unrelated to this fix, not addressed here.
