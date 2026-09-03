<!-- docs:meta
topic_id: repo.docs.audit.lean-single-bugf-rootcaused-not-fixed-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-bugf-rootcaused-not-fixed-2026-07-05
-->

# lean_single forensic dispatch — issue #601 Bug F root-caused, fix rejected (blocked on pre-existing `kinetics.sio` structural defect)

Date: 2026-07-05
Branch: `main` (post-PR #628, Bug E)
Class: **root cause identified, no code change landed** — the fix that
addresses issue #601's "Bug F" was built, verified correct against every
isolated repro, then rejected after it regressed 6 previously-passing tests
Status: root-caused; fix designed, tested, and deliberately not shipped;
issue #601 Bug F remains open as a known limitation

## Summary

Issue #601's Bug F ("a module-level `let CONST: T = some_fn();` doesn't
propagate its declared type to later arithmetic, because it's initialized
from a function call rather than a literal") was root-caused correctly and a
fix was implemented, but the fix causes a **verified regression** in the full
test suite (1308 pass / 6 fail / 124 known failures / 689 skip, vs. the
1314/0/124/689 baseline) with no clean way to avoid it. The regression's own
root cause is a **pre-existing, independent structural defect** in
`stdlib/chemistry/kinetics.sio` (orphaned top-level statements with no
enclosing function) that is out of this dispatch's scope. This document
records both root causes and the decision not to ship, so neither has to be
re-investigated from scratch.

## Original repro (issue #601)

```sio
fn get_const() -> f64 { 8.314 }
let R: f64 = get_const()          // module-level, from a function call
fn f(t: f64) -> f64 { R * t }     // error: arithmetic operands must have matching numeric types
```

A literal-initialized module-level `let` (`let R: f64 = 8.314`) does not have
this problem — only a function-call (or other non-literal) initializer
triggers it.

## Root cause #1 (Bug F itself): two disjoint global-registration mechanisms

`self-hosted/compiler/lean_single.sio` has always had two entirely separate
code paths for module-level bindings:

- **`var` declarations** go through `gl_add(ns, ne, esiz, alen)` (line 1397),
  which reserves a real BSS-backed slot and records it in `GL`/`GL_TY`/
  `GL_TY_HASH`. `emit_global_inits_x86()`/`_a64()` (line 7415/~7502) then run
  at program start and emit code to **evaluate and store the initializer
  expression** — any expression, not just a literal — into that slot.
- **`let`/`const` declarations** go through `scan_all_consts()` (line 6147)
  instead, which only ever does **compile-time literal folding**: it reads
  an int or float literal token directly into `CONST_VAL`/`CONST_TYPE` and
  registers it in `CONST_NS`/`CONST_NE`. For a non-literal initializer (a
  function call), the existing code silently fails to extract a value and
  the declared type never reaches `CONST_TYPE` — hence the later "arithmetic
  operands must have matching numeric types" error at every use site.

**Fix implemented** (not merged — see below): in `scan_all_consts()`'s four
near-duplicate blocks (bare `let`, bare `const`, `pub let`, `pub const`),
detect a non-literal initializer (`TK[p] != 4 && TK[p] != 53`, i.e. not an
int/float literal) and, instead of attempting constant-folding, register the
binding as a real global via `gl_add()` — the exact same mechanism `var`
already uses — then let the already-existing `emit_global_inits_{x86,a64}`
machinery run its initializer expression. This required extending
`emit_global_inits_{x86,a64}`'s dispatch (`if tk == 12 { ... }`, `var`-only)
to `if tk == 11 || tk == 12 { ... }`, plus updating the two internal
skip-to-next-declaration loops in each function accordingly.

**Independent byproduct found while implementing the fix, worth noting for
any future attempt**: `scan_all_consts()`'s type-annotation detection used
`TK[p as usize] == 15`, which is the tokenizer code for the `while` keyword,
not `:`. This is dead code today — a literal's own token already determines
its type, so the wrong check never mattered — but it would need to be `== 43`
(the actual colon token) for a non-literal initializer's declared type
annotation to be read correctly.

This design was verified correct in isolation, with real runtime values
(not just "no compile error"):

- `fn get_const() -> f64 { 8.314 }` / `let R: f64 = get_const()` / `R * 2.0`
  → `16.0` (was a compile error)
- Same pattern with an `i64` return type — correct
- `pub let Q: f64 = get_const2()` (cross-module, non-literal) — correct
- Literal-initialized `let PI: f64 = 3.14` (regression check) — unaffected

## Root cause #2 (the blocker): pre-existing orphaned top-level code in `kinetics.sio`

Applying the fix above and running the full suite produced 6 new failures:
`test_kinetics_core.sio`, `test_lib_surface.sio`, `test_pbpk_ontology.sio`,
`graphics_epistemic_advanced_smoke.sio`, `graphics_smoke.sio`,
`graphics_svg_export_smoke.sio`. The first was traced to
`stdlib/chemistry/kinetics.sio`, where `pub fn crn_result_to_audit(...)`
closes around line 1485, followed by a large Portuguese-language comment
block (~1487–1514), followed by what turns out to be **statements sitting at
the file's true top level, outside any enclosing function** — including
`let (ens_v, ens_u) = simulate_fractional_structural_ensemble(...)` at line
1524 and several more `let`s through line 1552. This is almost certainly a
missing `fn ... { ` wrapper accidentally dropped after the comment block; it
predates this dispatch entirely.

Confirmed via a tokenizer-aware brace-depth count (stripping `//` comments
and `"..."` string literals first, to avoid false brace matches from
comment/format-string text) run from the top of the file through line 1552:
depth is genuinely **0** at every one of the reported `let` lines
(1515, 1516, 1521, 1522, 1524, 1546, 1552) — this is real module-level code,
not a `scan_all_consts()` depth-tracking bug.

**Why it was previously harmless and is not now**: on `main` today,
`scan_all_consts()` already scans these orphaned top-level `let`s at
depth 0 and attempts to register them as constants — confirmed via a
side-by-side debug build of the *unmodified* `scan_all_consts()`, which
reports seeing the identical 9 lines. This has always been silently
harmless, because the old code only ever does literal constant-folding
(a no-op on a non-literal initializer) and never touches codegen. The fix
above changes that: it now runs `gl_add()` plus a real `emit_global_inits`
codegen pass over the same orphaned lines, which corrupts shared compiler
state (parsing `[0.52, 0.38, 0.27, 0.19]` and a tuple-destructuring call as
if they were legitimate module-level globals) and produces the observed
cascade — "unknown identifier `ens_v`/`ens_u`", `error[E001]: Type mismatch`,
"array index must be integer", "tail type mismatch" — none of which are real
defects in the test files themselves.

**Why there is no clean fix from this side**: the only signal that
distinguishes a genuine module-level `let` (issue #601's repro) from
`kinetics.sio`'s orphaned code is depth — and both are depth 0. Depth cannot
be the discriminator; there is no syntactic difference to key off. Scoping
the fix more narrowly (e.g. only firing for specific initializer shapes)
would be guessing against a condition this dispatch does not have full
visibility into across the stdlib, not a principled fix.

## Decision

Per repo operating principle 6 ("numerical/behavioral tolerances must be
derivable, not retrofitted") and 7 ("auditability over speed"), shipping a
fix that trades 6 confirmed-passing tests for a capability no currently
passing test exercises — and that does not even make its own motivating
files (`kinetics.sio`, `thermo.sio`) compile cleanly, since `thermo.sio`
surfaces new "tail type mismatch" and non-`pub`-call warnings even with the
fix applied — is not a good trade. **No code change is being made in this
dispatch.** The fix design above is preserved here in case a future dispatch
also fixes `kinetics.sio`'s orphaned-code defect first (which would remove
the blocker), rather than needing to be re-derived from scratch.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_verify.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Baseline (no fix, current main): Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
# With the rejected Bug F fix applied: Pass: 1308  Fail: 6  Known failures: 124  Skip: 689  Total: 2127
```

This dispatch makes no source change, so `main`'s baseline (1314/0/124/689)
is unaffected.

## Cross-references

- GitHub issue #601 — Bug F remains an open, documented known limitation.
  Bug G remains open, plus the `use ... as alias` and `println(&str)` gaps
  noted in `docs/audit/LEAN_SINGLE_NAMED_USE_IMPORT_2026-07-05.md` and
  `docs/audit/LEAN_SINGLE_LITERAL_REF_ARG_2026-07-05.md`, none yet
  issue-tracked.
- A separate, not-yet-filed defect: `stdlib/chemistry/kinetics.sio` has
  orphaned top-level code (no enclosing function) around lines 1515–1552,
  likely a dropped `fn` wrapper after the comment block ending ~1514. This
  should be reported/fixed independently of Bug F.
