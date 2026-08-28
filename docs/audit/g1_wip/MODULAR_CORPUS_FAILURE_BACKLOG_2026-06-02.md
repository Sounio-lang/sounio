<!-- docs:meta
topic_id: repo.docs.audit.g1-wip.modular-corpus-failure-backlog-2026-06-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.g1-wip.modular-corpus-failure-backlog-2026-06-02
-->

# Modular `--check` corpus failure backlog — ranked, verified (2026-06-02)

> **⚠️ PARTIALLY STALE 2026-06-10** — re-measured in
> [`../MODULAR_COMPILER_AUDIT_2026-06-10.md`](../MODULAR_COMPILER_AUDIT_2026-06-10.md)
> on the 2026-06-07 binary: roadmap items **#1 (E008/E170 bridge state-loss),
> #2 (scientific-notation lexer) and #4 (E014 usize index) are FIXED**
> (verified by direct repro). **#3 (E004 literal width) remains OPEN** and is
> now the largest coded failure bucket. Pass rate moved 25% → ~37% (1-in-5
> sample); 23/68 sampled failures now reject with NO error code (new
> diagnostics-gap finding). Long-tail bucket counts below are dated; re-run
> the backlog workflow on the current binary before using them for routing.

Turns the census's black-box "~231 generic failures" into a **complete, leverage-ranked,
adversarially-verified root-cause backlog** of all modular-compiler `--check` corpus
failures. Built by a 22-agent workflow (classify → synthesize → verify).

- Corpus: 504 `tests/run-pass/*.sio`; **380 fail** (124 pass). g1 tip
  `3e3a239d2` (enum let-binding), binary md5 `4bcf747c`, `bin/souc e35ef063`,
  detached worktree, `ulimit -s 1048576`. Live FIX#2 worktree untouched (read-only).
- **352/380 classified** — one classify batch (~28 programs) failed to return structured
  output; those are uncategorized (not lost, just not bucketed). Counts below are of the
  352.
- Top-6 buckets **adversarially verified** with minimal repros on the same binary: **5/6
  CONFIRMED**, 1 (`misc_syntax`) correctly flagged NOT-single-cause (it's a grab-bag).

## The prioritized roadmap (do in this order)

Four fixes clear **~230 of 352** failures:

1. **[~132, ONE bug, front-half/G1] Bridge state-loss.** The by-value bridge
   (check.sio:1146 / `return` ~2489) drops the declared **return type** → E008 "expected
   `()`" on every value-returning fn, AND drops the **effect row** → E170 ".value requires
   `with Epistemic`". Sole blocker on **~96** programs, noise on ~36 more. *VERIFIED:*
   minimal correct `main` emits exactly these two spurious errors, rc=1 (not a crash).
   **Highest leverage; one fix.** (= the census's #1 lever, now confirmed to also block real
   compilation, not just `--check`.)
2. **[~54, parser] Scientific-notation float lexer gap.** `1.0e-30` / `3.0e-26` are
   mis-lexed; cascades worst as `<float>e<exp> {` before a brace. **Largest REAL
   (non-spurious) bug** — one lexer change (`[eE][+-]?[0-9]+` exponent suffix) unblocks ~54
   numeric/PBPK/ODE/algebra programs. *VERIFIED:* repro shows `parse error … actual=123`
   (the `{`).
3. **[~26, front-half/G1] E004 literal width.** Untyped int/float literals default to
   i64/f64 and won't combine with narrower operands (`i32 + 10`, `f32 + 0.5`). Fix:
   bidirectional literal-width inference at binary-op sites. *VERIFIED.*
4. **[~18, front-half/G1] E014 `usize` array index.** Index check accepts only i64;
   `arr[i as usize]` rejected. Fix: treat all integer widths/usize as valid index types.
   *VERIFIED.*

## Full ranked backlog (352 classified)

| Count | Category | Owner | Root cause | Verified |
|---:|---|---|---|---|
| 132 | spurious | front-half/G1 | E008-return + E170-effect = one bridge-state bug | CONFIRMED |
| 54 | parse_gap | parser | scientific-notation float exponent lexer gap | CONFIRMED |
| 26 | type_error | front-half/G1 | E004 int/float width mismatch | CONFIRMED |
| 18 | type_error | front-half/G1 | E014 `usize`/`as usize` array index rejected | CONFIRMED |
| 18 | parse_gap | parser | misc one-off syntax (pub field, destructure, …) | NOT-single |
| 15 | parse_gap | parser | `kernel fn … with GPU` + `&![f64]` slice args | CONFIRMED |
| 13 | parse_gap | parser | refinement types `{ x: T \| pred }` | — |
| 13 | parse_gap | parser | `algebra … over …{}` + `study{}/hypothesis{}` DSL | — |
| 11 | parse_gap | parser | first-class fn types `fn(T)->T [with E]` | — |
| 10 | parse_gap | parser | `async{}`/`spawn{}`/`.await`/tuple-destructure | — |
| 9 | parse_gap | parser | `extern "C" { … }` foreign-fn blocks | — |
| 9 | type_error | front-half/G1 | E001 unit-typed let rejects f64 literal | — |
| 9 | type_error | front-half/G1 | E011/E013 `Seq<T>` method resolution + subscript | — |
| 9 | type_error | front-half/G1 | E015 unknown struct (Knowledge ctor + cross-module) | — |
| 8 | parse_gap | parser | top-level `const NAME: T = N` decls | — |
| 7 | parse_gap | front-half/G1 | untyped closure params `\|x\|` + fn-as-value | — |
| 4 | parse_gap | parser | generic fn type-params `fn name<T>` | — |
| 4 | type_error | front-half/G1 | E003 immutable-binding modify (needs var/Mut) | — |
| 4 | crash | front-half/G1 | typed closure literal → SIGSEGV mid-check | — |
| 4 | type_error | front-half/G1 | silent reject of `&!`/`&!Outer` mutable borrow | — |
| 3 | parse_gap | parser | inherent `impl T { fn … }` blocks | — |
| 2 | crash | front-half/G1 | SIGSEGV on `match *slot` deref + heterog. tuple | — |

**See also:** `SEVEN_CRASHES_DIAGNOSED_2026-06-02.md` for the cluster breakdown of the (evolved) 7 crashers (Clusters A/B resolved via shippable *mut/codegen work; Cluster C = the known large-SRET by-value Checker miscompile, documented + NOT pursued per explicit decision — the bug the *mut arc exists to avoid).

Totals: parse_gap ~163, spurious ~132, type_error ~80, crash 6.
(Bucket counts sum to 382 vs 352 classified — a few programs double-tagged across
related sub-keys during synthesis; treat counts as ±a handful, ranking is robust.)

## What this changes vs the census
The census (`MODULAR_CORPUS_CRASH_CENSUS_2026-06-01.md`) left ~172 parse-gaps as
"generic, needs per-line triage." This cracks that open: the parse-gaps are **not a blob**
— they are ~12 distinct, self-contained grammar productions, and **one of them
(scientific-notation floats, 54) is the single largest real bug in the whole corpus**, not
visible until now. The 6 crashers are confirmed to be non-leaf exprs (typed closures,
deref-match, heterogeneous tuple) needing the same `*mut`-spine treatment as the landed
`match→if-chain`.

## Ownership split (for routing)
- **front-half/G1** (check.sio): the 132 bridge-state bug (#1), E004/E014/E001/E011/E015/
  E003 type-inference gaps, the 6 crashers, untyped-closure check. (~210)
- **parser/lexer**: float exponent (#2, biggest parser win), kernel fn, refinement, DSL
  blocks, fn-types, async, extern, const, generics, impl. (~140) — a self-contained
  parser-completeness workstream, independent of the `*mut` spine.

## Caveats (honest)
- **Dated snapshot**; g1 binary moving under the live FIX#2 lane — re-run the workflow to
  refresh (script: `modular-corpus-failure-backlog`).
- 28/380 unclassified (one batch's StructuredOutput failure). Counts are of 352.
- Sub-bucket attribution is agent-classified (read source+log); the PASS/FAIL/crash split
  and the top-bucket repros are exact/verified; long-tail counts are ±a handful.
- `misc_syntax` (18) is a grab-bag, not one cause (verifier flagged it); split further
  before treating as a single fix.
