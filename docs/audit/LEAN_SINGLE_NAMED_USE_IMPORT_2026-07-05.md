<!-- docs:meta
topic_id: repo.docs.audit.lean-single-named-use-import-2026-07-05
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.lean-single-named-use-import-2026-07-05
-->

# lean_single forensic dispatch — a named (non-glob) `use` import treats the imported symbol as a path segment

Date: 2026-07-05
Branch: `main` (post-PR #625, Bug B)
Class: **module-loader gap** (a syntactically valid, commonly-used `use` form
was resolved against the wrong filesystem path and failed to load at all) —
root-causes and closes issue #601's "Bug C"
Status: root-caused, fixed, verified (full test suite 1314 pass / 0 fail / 124
known failures / 689 skip — 3 net additional genuine passes, 0 regressions)

## Symptom

`use package::file::specific_symbol` (no trailing `::*`, no `::{...}`) fails
to load the target file at all, reporting the *symbol* name as if it were an
additional directory/file segment:

```sio
// stdlib/pkg/thing.sio
module pkg::thing
pub fn item() -> i64 { 1 }

// main.sio
use pkg::thing::item
fn main() -> i64 { item() }
// error: unreadable import: stdlib/pkg/thing/item.sio
```

The glob form (`use pkg::thing::*`) loads the same file correctly. Matches
issue #601's original Bug C repro exactly. Real-world impact: three tracked,
value-checked (`//@ expect-stdout: ALL PASS`) acceptance tests for the
octonion/uncertainty-propagation stdlib were carried as known failures purely
because of this — `tests/run-pass/uncertain_octonion_auto.sio`,
`tests/run-pass/propagate_nonassoc_variance.sio`, and
`tests/run-pass/perturbation_graph_order_safe.sio`, all of which import via
`use algebra::octonion::oct_basis` or similar named forms.

## Root cause

`self-hosted/compiler/lean_single.sio`'s `resolve_imports()` extracts a `use`
statement's module path with a single scan (originally at line 34744) that
recognises exactly two path-terminating markers:

```sio
while pos < SRC_LEN && sb(pos) != 10 && sb(pos) != 59 && sb(pos) != 13 {
    if sb(pos) == 58 && pos + 1 < SRC_LEN && sb(pos+1) == 58 {
        if pos + 2 < SRC_LEN && sb(pos+2) == 42 { /* ::* */ path_end = pos; break }
        if pos + 2 < SRC_LEN && sb(pos+2) == 123 { /* ::{ */ path_end = pos; break }
    }
    pos = pos + 1
    path_end = pos
}
```

`::*` and `::{...}` are unambiguous: everything before them is the file path,
everything from the marker onward is glob/brace syntax, not more path. A
**bare named import has no such marker** — `use pkg::thing::item` is
token-for-token indistinguishable from a (legitimate, and already-working)
whole-file import like `use database::pure::types` where every segment really
is a directory/file component. The scan has no way to know, from the tokens
alone, whether the *last* segment is "the file" or "a specific symbol
exported by the second-to-last segment's file" — so it always treated it as
the former, converting every `::` (including the one before the actual
symbol) into a `/` and appending `.sio`, producing `pkg/thing/item.sio` —
a path that never exists.

There was no fallback that ever tried the *other* interpretation (drop the
last segment, treat it as a symbol name, load the parent as the file) — every
existing fallback in this function (`mod.sio`, `lib.sio`,
`packages/<pkg>/src/`, `stdlib/`-prefix, `self-hosted/`-prefix) only varies
*where* the fully-joined path is rooted, never *how many trailing segments*
belong to the file path.

## Fix

The two are genuinely ambiguous from syntax alone (this is not decidable
without a filesystem probe — real code uses both `use pkg::mod::file` and
`use pkg::mod::file::symbol` identically shaped otherwise), so the fix adds a
**last-resort fallback**, tried only after every existing whole-path attempt
has failed and only for a bare (non-glob, non-brace) import — tracked via a
new `is_named_import` flag set during path extraction: strip the last
`/`-segment of the already-`.sio`-suffixed raw path and retry across the same
path roots (`BASE_DIR`-relative, `stdlib/`-prefixed, `self-hosted/`-prefixed,
and the raw path itself) the primary attempts already tried:

```sio
if copy_len <= 0 && is_named_import == 1 {
    // find the last '/' before ".sio" in the raw joined path
    // if found, truncate there + re-append ".sio", retry read_file()
    // across BASE_DIR/, stdlib/, self-hosted/, and the bare raw path
}
```

Because this only activates when the naive full-path read has *already
failed*, the existing `use pkg::mod::file` (bare whole-file, no trailing
symbol) idiom is completely unaffected — it keeps succeeding on the first,
unshortened attempt, exactly as before. Verified directly: `use
database::pure::types` (no glob, no symbol — the same shape a resolver-only
fix could have broken) still loads `database/pure/types.sio` on the first
attempt, not the shortened fallback.

## Verification

```bash
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"
scripts/dev/souc-build-lock.sh ./bin/souc-lean-single-x86_64 self-hosted/compiler/lean_single.sio /tmp/lean_fixed.elf
bash scripts/run_sio_test_suite.sh --format junit --jobs 8
# Pass: 1314  Fail: 0  Known failures: 124  Skip: 689  Total: 2127
```

Baseline was 1311 pass / 0 fail / 127 known failures / 689 skip — net +3 pass,
-3 known failures, 0 regressions. The three newly-passing tests
(`uncertain_octonion_auto.sio`, `propagate_nonassoc_variance.sio`,
`perturbation_graph_order_safe.sio`) are removed from
`tests/known_failures/hardened_diagnostics_full_suite.txt`; all three are
`//@ run-pass` + `//@ expect-stdout: ALL PASS` tests (exact-stdout-match, not
merely "compiles"), so this is a genuine correctness fix, not a masked
compile-clean regression.

Also confirmed directly:
- The exact issue #601 repro (`use pkg::thing::item; item()`) now loads
  correctly and returns `1` as defined.
- The glob form (`use pkg::thing::*`) is unaffected (still works).
- The bare whole-file-import idiom (`use database::pure::types`, no trailing
  symbol) is unaffected — succeeds on the unshortened first attempt.

## Known follow-up not fixed here

Issue #601's Bug C entry also notes a **module-alias variant**
(`use pkg::mod as alias;` then `alias::fn()`) "behaves the same way." That
form hits a different code path in the same extraction loop — the ` as
alias` suffix is not a `::`-segment at all; spaces are silently dropped
during path construction (`c == 32` is skipped, not treated as a
terminator), so `use pkg::mod as alias` currently builds the garbled path
`pkg/modasalias.sio` rather than `pkg/mod/alias.sio`. This is a related but
mechanically distinct defect (space-handling in path construction, not
`::`-segment ambiguity) and is not addressed by this fix. Left for a
dedicated follow-up.

## Cross-references

- `docs/audit/LEAN_SINGLE_MULTISEGMENT_QUALIFIED_CALL_2026-07-04.md` — Bug A
  (PR #624), the call-site sibling: a different `::`-ambiguity, in call
  position rather than the `use` statement's own path.
- `docs/audit/LEAN_SINGLE_SCAN_TYPE_QUALIFIED_PATH_2026-07-04.md` — Bug B
  (PR #625), the type-annotation sibling.
- GitHub issue #601 — tracks Bug C (closed by this fix). Bugs D–G remain
  open; the `use ... as alias` variant noted above is a newly-identified,
  not-yet-tracked follow-up.
