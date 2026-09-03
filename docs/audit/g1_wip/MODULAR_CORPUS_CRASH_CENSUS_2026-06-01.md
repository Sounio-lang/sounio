<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.modular-corpus-crash-census-2026-06-01
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.modular-corpus-crash-census-2026-06-01
-->

# Modular `--check` corpus census — g1 tip `ddc7a8b7e` (2026-06-01)

**Read-only diagnostic. No source edited.** Empirical per-program inventory of the
504 `tests/run-pass/*.sio` programs run through the modular compiler's `--check` on the
**current** g1 branch tip — *not* the binary the earlier diagnosis used.

- Branch / commit: `g1/qualify-bare-patterns` @ `ddc7a8b7e` (bare-pattern quals +
  `040ac8dba` struct collector + `ddc7a8b7e` enum-collector body fix).
- Binary: `mc_census.elf` md5 `fcaabdad1481385e2c9e05e701e21428`, built from
  `self-hosted/compiler/main.sio` with `bin/souc` md5 `e35ef063…` (untouched), in a
  detached worktree `/workspace/sounio-g1-census` (the live FIX #2 worktree was NOT
  touched).
- Run: `--check` per file, `ulimit -s 1048576` (1 GB cap, **never** unlimited), 30 s
  timeout. Harness + raw logs + `results.tsv` under `.build/census/` in that worktree.

## Headline: the crash frontier is now **6** (vs ~55 per the prior doc)

**6 crashers measured here** on `fcaabdad`. The "~55" is from
`MODULAR_FRONTEND_GAPS_2026-06-01.md` (binary `daaa5758`, pre-collector) and was **not
re-measured on this harness/stack-cap** — so treat the drop as a cross-binary
observation, not a controlled delta. The struct + enum collectors that landed after that
doc almost certainly converted most old SIGSEGV crashers into ordinary `rc=1` type/parse
failures; the frontier is now **6 crashers + a dominant spurious-error class + a parser
backlog.**

| Bucket | Count | % of 504 |
|---|---:|---:|
| **PASS** (rc=0, `check: OK`) | 124 | 24.6% |
| **FAIL** (rc=1) | 374 | 74.2% |
| **CRASH** (rc=139 SIGSEGV) | 6 | 1.2% |
| timeout (rc=124) | 0 | 0% |

## The 6 crashers (rc=139)

| Program | Triggering construct | Attribution |
|---|---|---|
| `closure_basic` | typed closure `\|x: T\| …` | **PROVEN** (repro below) |
| `closure_arity_2` | typed closure `\|x: T\| …` | proven by class |
| `approx_propagation` | typed closure `\|x: T\| …` | source-marker |
| `lsp_hover_qualified` | `if/else` (crashes after E014) | source-marker |
| `native_tokenizer` | `match` + `if/else` (crashes after E008) | source-marker |
| `sprint235_print_f64_e2e` | (no obvious marker; needs per-file gdb) | unpinned |

**See also:** `SEVEN_CRASHES_DIAGNOSED_2026-06-02.md` (the 2026-06-02 follow-up diagnosis that evolved the count to 7 crashers and introduced the A/B/C clustering; C is the known large-SRET by-value-Checker miscompile — the root the *mut arc avoids — and was explicitly left documented / NOT pursued).

**New finding (PROVEN): typed closures crash the checker (top crash cause, 3/6).**
Controlled minimal repro on `fcaabdad`, `ulimit -s 1048576`:

```
fn main() -> i32 { let inc = |x: i64| x + 1   let r = inc(5)   0 }   → rc=139 (SIGSEGV)
fn main() -> i32 { let r = 5 + 1   0 }                                → rc=0   (control)
```

Nuance: the *typed* form `|x: i64|` parses and reaches the checker, then crashes; the
*untyped* form `|x|` fails earlier at parse (rc=1 `parse_failed`) — so this only bites
programs that use the typed syntax. The closure expr kind, like `if/else`/`match`,
bridges to the by-value `check_expr` (check.sio:1146); it was not separately called out
in the prior gaps doc. The other 2 crashers are the known `if/else`/`match`-statement
by-value-spine crashes (doc §B). All 6 are the same root: **non-leaf exprs not yet on the
`*mut` spine.** (closure_arity_2 shares closure_basic's exact construct → "proven by
class"; approx_propagation/lsp_hover/native_tokenizer are source-marker attributions, not
per-file gdb.)

## The 374 `rc=1` failures decompose into 3 classes

### (1) SPURIOUS — one bug, **132 programs (35% of all failures)**
The single highest-leverage target. Caused by the by-value bridge dropping `*c` state
(check.sio:1146-1147), exactly as the gaps doc predicted (its FIX #3):

- **E008 "expected `()` / found <T>" — 105 programs, 105/105 (100%) are the spurious
  unit-return pattern.** `current_return_type` reads as `TyUnit` at
  `checker_check_return_expr_inplace` (check.sio:2489), so *every* value-returning
  function mismatches. Not a real error.
- **E170 "accessing `.value` … requires `with Epistemic` effect" — 27 programs.**
  `current_effects` ([i64;8]) is dropped when `*c` is materialized into the by-value
  `self` at the bridge, so the effect row reads empty. Not a real error.

**132 programs emit one of these spurious errors (proven: E008 105/105 exact-string +
E170 27).** Fixing the bridge state propagation (or moving return/field-access onto the
`*mut` spine) is the single highest-leverage lever — but "clears 132" is *projected*, not
proven: clearing the first error can surface a genuine second error in some programs, so
the realized pass-count gain is ≤132.

### (2) PARSE-GAPS — **~172 programs**
Parser rejects the construct before the checker runs (`parser reported … syntax errors`
/ `parse error: expected token at line N`; the parser does not name the token). Diverse;
identifiable sub-classes are small (`algebra` 9, `type-alias` 4, `loop` 3, `impl` 3,
`affine` 1); the remaining ~152 need per-line triage. This is a **separate parser-
completeness effort**, independent of the `*mut` spine. (Consistent with the doc's
"methods/impl/loop/type-alias fail at parse" + the FIX#0-falsified note.)

### (3) OTHER GENUINE-LOOKING TYPE ERRORS — **~59 programs**
E004 (18), E014 (11), E011 (8), E001 (8), E003 (5), E016 (4), **E015 (3 — down from
"ALL struct/enum usage" pre-collector**, confirming the struct collector works), E040
(1), E013 (1). Mix of real gaps and possible secondaries behind the spurious class;
worth re-checking after FIX #3 lands (some may be downstream of the dropped state).

*(11 of the 374 fell into no recognized token bucket — non-parse, non-E-coded output;
unclassified, low priority.)*

## Ranked leverage (updated from the gaps doc)

1. **Bridge `*c` state propagation / move return+field-access onto `*mut` spine** →
   clears ~132 spurious (E008+E170). **Highest leverage by far.** (doc FIX #3, now
   quantified as the #1 lever.)
2. **Move closure + if/else + match onto the `*mut` spine** → clears the 6 crashers.
   (Closures are new; if/else+match = doc FIX #2, the lane currently in-flight.)
3. **Parser completeness** (~172) → the long tail; separate from the spine, lower
   urgency per-fix but the largest raw count.

## Caveats (honest)
- Snapshot of committed tip `ddc7a8b7e`; a live session is advancing FIX #2 on
  `g1/qualify-bare-patterns` — this census will drift as that lands (expected to clear
  the if/else+match crashers and possibly the E008/E170 class). Re-run the harness
  against the new binary to refresh.
- Crasher/parse sub-classification is feature-heuristic (source-marker + first-diagnostic
  matching), not per-file gdb. The PASS/FAIL/CRASH split and the E-code histogram are
  exact (from `results.tsv`); the *attribution* of crashers to constructs is indicative.
- `sprint235_print_f64_e2e` crash cause not pinned.
