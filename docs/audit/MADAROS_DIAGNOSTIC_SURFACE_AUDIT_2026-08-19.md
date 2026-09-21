<!-- docs:meta
topic_id: repo.docs.audit.madaros-diagnostic-surface-audit-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-diagnostic-surface-audit-2026-08-19
-->

# Madaros diagnostic surface audit (WS-B3)

**Date:** 2026-08-19
**Auditor:** Claude (lane `claude-2`, branch `lane/claude-2/20260814`)
**Scope:** the compiler's error messages *as a user interface* — not whether Madaros compiles
programs correctly, but whether it tells the truth about what it is doing when it refuses one.
**Compiler under test:** `./bin/souc` → committed prebuilt ELF `bin/madaros-linux-x86_64`
(md5 `46e7ef146a0071a15f7f8e1054947475`, mtime 2026-08-14), reporting `Madaros v0.80.0`. This is
the shipped binary, not a fresh from-source rebuild — flagged per CLAUDE.md's own warning that
the shipped binary can lag source; none of the findings below depend on that gap, since they are
about the diagnostic *text and firing behaviour* of the binary users actually run.

## 0. Semantic declaration

The founder's framing for today was: we spent the day finding that external systems (the GitHub
API, in 20/38 measured cases) report a status that doesn't match reality, and the task is to turn
that same instrument on the compiler. A compiler's diagnostic is the same kind of claim as an API
status field — it asserts "here is what is true about your program" — and it is trusted
unconditionally by a user who has no independent way to check it, more so than an API response,
because the user's whole feedback loop *is* the compiler.

The finding, stated before any count: **Madaros does not have a diagnostic system. It has three
of them, unreconciled, only one of which is documented, and none of which agree with the two
documents that describe them.** A code number does not have a stable meaning across the codebase
— `E041` alone means four different, unrelated things depending on which of three independent
reporting mechanisms happens to fire. And the corpus the project uses to assert "this code is
correct" fails to compile clean 35% of the time, for reasons that trace to five shared defects,
not five hundred separate ones.

## 1. Method

Two independent, fully re-runnable sweeps, cross-checked against each other and against
`docs/compiler/KNOWN_LIMITATIONS.md`:

1. **Emission-site inventory** — every place in `self-hosted/**/*.sio` that can produce an
   `error[E0XX]` tag, found two ways (literal-string grep, which undercounts, plus a trace of
   every numeric `code` parameter passed to `report_error_at` / `checker_report_error_at_inplace`
   / `checker_report_mismatch_inplace`, which does not), reconciled against the three dispatch
   tables `print_error_message` / `print_error_help` / `print_error_note` in
   `self-hosted/check/check.sio`. Scoped to the Madaros path (`main.sio`-reachable); the legacy
   `lean_single.sio` engine inventoried separately, not merged in, per "Madaros canónico."
2. **Correctness-corpus sweep** — `./bin/souc check <file>` (default Madaros engine; `check` skips
   codegen, no build lock needed) run over the full, un-sampled `tests/run-pass/*.sio` corpus
   (1,696 files) — code the project itself asserts is correct. Every hit cross-checked under
   `SOUNIO_SOUC_ENGINE=lean_single` as an independent oracle: if the second engine accepts the
   same file, disagreement is established without needing to hand-adjudicate right and wrong.

Reproduction:

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc --version                                    # Madaros v0.80.0

# emission-site inventory
grep -rnoE 'error\[E[0-9]+\]' self-hosted/check/check.sio self-hosted/parser/stmts.sio self-hosted/parser/types.sio
grep -rn '\.report_error_at(\|checker_report_error_at_inplace(\|checker_report_mismatch_inplace(' self-hosted --include=*.sio
sed -n '11808,11987p' self-hosted/check/check.sio | grep -oE 'code == [0-9]+'   # message table
sed -n '11990,12273p' self-hosted/check/check.sio | grep -oE 'code == [0-9]+'   # help table
sed -n '12276,12594p' self-hosted/check/check.sio | grep -oE 'code == [0-9]+'   # note table

# correctness-corpus sweep
find tests/run-pass -name '*.sio' | wc -l                                       # 1,696
find tests/run-pass -name '*.sio' | xargs -P 32 -I{} ./bin/souc check {}        # per-file rc + stderr
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check <hit>                           # cross-check
```

## 2. Q1 — how many error codes exist, and how many say what to do next

**155 distinct codes have a live emission path in Madaros**, reached through **three
uncoordinated mechanisms**, all inside `self-hosted/check/check.sio` (plus a fourth, smaller site
in the parser):

| Mechanism | Path | Codes reached |
|---|---|---|
| A — central numeric-code path | `report_error_at` (188 call sites) / `checker_report_error_at_inplace` (135) / `checker_report_mismatch_inplace` (31), all routing through `code: i64` into the three dispatch tables | 90 |
| B — hardcoded per-call reporters | ~85 functions named `report_*`/`checker_report_*_inplace`, each with its own inline `print("error[E0XX]")` and hand-typed text, **never touching the dispatch tables** | 77 |
| C/D — bare inline literals and parser-level literals | e.g. E040/E036 inline in `check.sio`; E040/E041/E218 in `self-hosted/parser/stmts.sio` and `types.sio`, pre-typecheck | (subset of the above) |

12 codes are reached through **both** A and B — this is where the collisions in §3 live.

Separately from string grep: 3 codes (`37`, `38`, `39` — shared/exclusive borrow conflict,
double-consume) are passed as a **variable**, not a literal, resolved only by tracing
`self-hosted/check/borrow.sio:144,164,181`. A plain grep for `error[E0` or for integer literals
at call sites misses these entirely — the true inventory needs both instruments used here, or it
undercounts silently.

Of the three dispatch tables (the closest thing Madaros has to a designed registry, though it
governs only mechanism A):

| Table | Arms | Coverage |
|---|---:|---|
| `print_error_message` (what's wrong) | 156 | every mechanism-A code has a slot; **zero** fall through to the generic default |
| `print_error_help` (what to do next) | 88 | **≈43% of codes with a message have no help arm** |
| `print_error_note` (context) | 101 | ≈35% have no note arm |

The gap concentrates, not scatters: the entire `E200–E207` zero-divisor/ZD-effect-gate family and
the `E150–E160` ontology/study family — a semantically important cluster, not edge cases — state
the problem and stop. 9 codes emitted only via mechanism B (`65, 66, 69, 73–78`) were never given
*any* table slot; if a future refactor ever routes them through `report_error_at`, they will
silently fall back to "unknown error" with no help or note. 11 codes have a dead table arm with no
emitter anywhere (`31, 33, 34, 52, 53, 67, 68, 138–141`) — `67/68` and `138–141` are dead because
their sole implementation, `causal.sio`'s identifiability checker, has zero callers in the whole
compiler.

**Answer: 155 live codes, 88 (57%) with genuine next-step guidance, 67 (43%) that only state what
is wrong.**

## 3. Q2 — how many are documented as an accepted limitation

**5 of 155.** `KNOWN_LIMITATIONS.md` names E019, E035, E137, E176, and the E201/203/204/207
zero-divisor family. **150 live codes are entirely undocumented** — including the whole
E065–E136 contest/audit/decision/deferral block, the E150–E160 and E200–E207 families identified
above as help-less, E040/E041/E042/E070/E072/E091 (all implicated in the collisions below), and
E170–E177/E213/E216–E218.

Worse than a coverage gap: where documentation exists, it is **wrong**, checked live against the
running binary today:

| Claim | Source | Live test result |
|---|---|---|
| E040 (`let mut x`) does not fire, compiles clean | `docs/llm-guide/explanations/E040.md`, "verified 2026-07-11" | **Fires.** Message: `Sounio uses 'var' for mutable bindings, not 'let mut'` |
| E041 (`&mut buf`) does not fire, compiles clean | `docs/llm-guide/explanations/E041.md`, same box | **Fires.** Message: `Sounio uses '&!T'...` |
| E042 (`#[derive(Debug)]`) does not fire, compiles clean | `docs/llm-guide/explanations/E042.md`, same box | **Neither.** No diagnostic text at all — a silent `AST closure incomplete` failure, exit 1. Worse than documented: the doc implies success, reality is an uninformative crash. |
| E035 (missing `with Div`) does not fire | same box, and independently `error-catalog.md` | **Confirmed accurate** — the one claim in this box that still holds |
| E035 "reporting E035 on violations" is verified/enforced | `docs/compiler/KNOWN_LIMITATIONS.md:210` | **Contradicts the row above, and the live test.** Two governance docs in the same repository disagree with each other; one of them disagrees with the compiler. |
| asin/acos are supported transcendentals | `docs/compiler/KNOWN_LIMITATIONS.md:409` | Madaros rejects `asin(x)`/`acos(x)` with `E137: use of undeclared variable` unless explicitly imported by name — lean_single ambient-injects them, Madaros does not (root cause C, §4). The doc is correct for lean_single and wrong for Madaros; it doesn't distinguish. |

Root cause, via `git log -S'E040' -- self-hosted/parser/stmts.sio`: the E040/E041 fix
(commit `fe63cfa4be`, "E040/E041 helpful diagnostics for Rust `let mut`/`&mut`") landed
**2026-07-12** — one day *after* the doc's own "verified 2026-07-11" timestamp — and the doc was
never revisited in the five weeks since. The documentation didn't fail to keep up with a moving
target; it was invalidated the day after it was written and nobody re-ran it.

## 4. Q3 — does anything fire on correct Sounio code (the one that matters)

Yes, systematically. Full sweep, not sampled: **596 of 1,696 `tests/run-pass/*.sio` files
(35.1%) fail `./bin/souc check`** under Madaros. All 1,100 passing files were separately checked
for the inverse failure mode (exit 0 while printing `error[` text) — zero found; exit code and
diagnostic text agree throughout the corpus.

`tests/known_failures/hardened_diagnostics_full_suite.txt` only pre-lists 14 of the 596 — it is a
path blocklist for the `run` (compile+execute) harness, not a check-time diagnostic registry, so
it does not shield most of this.

A more serious finding sits inside the corpus's own annotations. 338 of the 596 hits carry an
in-file `//@ known-failure: <module> typechecks, but current Madaros imported/native lowering
exits 139 at runtime` — a comment that explicitly asserts `check` should pass and only excuses a
*runtime* failure. Every one of those 338 fails `check` anyway. The annotation is not a live
description of current behaviour; it's a claim the compiler itself now falsifies every time that
file is checked.

Cross-checking the un-annotated hits (228 files, the clean false-positive candidate set, 100%
swept) against `lean_single` as an independent oracle:

| Verdict | Count | Meaning |
|---|---:|---|
| Disagree — Madaros fails, lean_single accepts | 194 | Madaros-only defect, by construction |
| Both engines reject | 34 | needs individual judgment — **not** automatically "stale test": at least 23 of these 34 (the `madaros_gum_fo_*` family) contain a proven Madaros-only sub-bug (root cause C below) masked at file level by a second, both-engines gap |

These 596 hits are not 596 independent bugs. They trace to five root causes:

| # | Defect | Symptom | Files affected | lean_single | Status |
|---|---|---|---|---:|---|
| A | Parser fails inside one 50,536-line stdlib file (`theorem/portfolio.sio`), specific function at line ~50272, mechanism not fully root-caused (best-supported hypothesis: a cumulative parser-state ceiling near function #2160, not a construct defect) | `parse error: expected token ... actual=0`, no line-level cause isolated | 337 (the 336 stale-annotation files + 1) | Whole-file: accepts, compiles 2,233 fns to a working ELF | Confirmed disagreement; mechanism open |
| B | `on` is a hardcoded reserved token (`self-hosted/lexer/tables.sio:25`) but usable as an identifier in valid Sounio | `closure parser incomplete`, **zero line/column emitted** | 24 (graphics stack + dependents) | Accepts, all 24 individually verified | Confirmed, minimal repro: `let on = true` alone breaks Madaros check |
| C | `asin`/`acos` require explicit `use stdlib::math::pure::{asin,acos}` under Madaros; lean_single ambient-injects them | `E137: use of undeclared variable` at every call site | 23 (`madaros_gum_fo_*` family) | Also rejects the same files, for an unrelated reason — isolated 5-line repro proves the asin/acos gap is real and Madaros-only, independent of that co-occurring gap | Confirmed Madaros-only regression (contradicts `KNOWN_LIMITATIONS.md:409`, §3) |
| D | Fully-qualified `use stdlib::<mod>::...` (explicit `stdlib::` prefix) fails import resolution; the same import without the prefix works | `unresolved import in authoritative closure` for a file that exists on disk | ≥1 confirmed (`trajectory_basic.sio`), more suspected | Accepts, compiles clean to ELF | Confirmed disagreement |
| E | `Seq<T>` subscript sugar (`s[i]`, `s[i]=v`) unsupported by the Madaros checker | `E011`/`E013` (no method / indexing requires array type) | ≥6 confirmed, likely explains 9 more `ontology_typed_bridge_*` files with the same signature | Accepts, compiles clean to ELF | Confirmed. The known-failures doc's own comment claims these "now PASS" — true for lean_single/native x86, **false for Madaros's checker**, which the doc doesn't distinguish |

Combining the directly-verified 194-file disagreement set with the 336 files that collapse onto
root cause A by identical parse-error signature (1 individually re-verified against lean_single,
the remainder inferred at very high confidence from the byte-identical failure line, not each
individually re-run): **≈530 of 596 check-time failures on asserted-correct code are confirmed or
high-confidence Madaros-only false positives**, traceable to five shared defects. A remaining
60–70 files across smaller error-code buckets (E175, E012, E039, E035, E036, E001/E004, E009, and
singletons) are confirmed as Madaros/lean_single disagreements but not yet clustered to a root
cause — flagged as open, not resolved by omission.

**Answer: yes, and not as a rare edge case.** A user hitting any of the five root causes above
gets a syntax- or type-shaped error message — E137 "undeclared variable" for `asin`, E011/E013 for
`Seq` indexing — that describes their code as wrong. Nothing in the message says the compiler,
not the program, is the defective party. That is exactly the class named in the day's earlier
work: a diagnostic that lies convincingly, with no signal the user could use to catch it, unless
they happen to try the same file under lean_single or file an issue and wait for someone with
this audit's tooling to trace it back to a hardcoded lexer keyword.

## 5. Secondary finding — code numbers do not have stable meanings

Independent of Q1–Q3: mechanisms A and B in §2 were built by different call sites at different
times and never reconciled against each other. The result is that a code number alone is not a
reliable diagnosis:

| Code | Meaning A | Meaning B (contradictory) |
|---|---|---|
| E041 | ontology subsumption unverifiable | (1) "use `&!T`, not `&mut T`" (parser); (2) unit mismatch in arithmetic; (3) unit mismatch in a call argument — **four unrelated meanings, live simultaneously** |
| E040 | dead table arm, never reached | "linear value not consumed" **vs.** "use `var`, not `let mut`" (parser) — three meanings, the table's own arm orphaned |
| E042 | refinement predicate violated | "multiple tests performed without correction" (statistics domain) |
| E070 | kernel function has a forbidden effect | model-family mismatch in a Contest robustness declaration |
| E072 | kernel function must return unit | Contest diagnostic missing |
| E091 | "reference escapes its scope" (live) shadows a **second, dead arm in the same table** whose text — "counterexample section is incomplete" — happens to match an unrelated live literal call site elsewhere. Two authors independently picked the same number for unrelated features. |

`E041` is the worst case: four live, simultaneously-reachable meanings for one printed tag.

## 6. lean_single — separate inventory, not merged

`self-hosted/compiler/lean_single.sio` (39,331 lines) has **no** `print_error_message`/
`print_error_help`/`print_error_note`/`report_error_at` machinery at all — no central table, no
help/note concept. Only 14 distinct coded errors exist there
(`1, 67, 70, 72, 80, 170, 171, 201–207`); the dominant style is 69 **uncoded** `print("error: ...")`
sites, including ones that reject the exact same Rust-isms Madaros codes as E040/E041/E218, but
without any code at all. Where lean_single does use a code number it generally agrees in domain
with Madaros (E070/E072 kernel constraints, E170/E171 epistemic `.value`, E201–E207 ZD family) —
these are not additional collisions.

## 7. Bottom line

- **Q1:** 155 live codes; 57% have a real next-step; 43% (concentrated in the E150–E160 and
  E200–E207 families) state the problem and nothing else.
- **Q2:** 5 of 155 codes are documented as accepted limitations; 150 are undocumented; where
  documentation does exist for a diagnostic (E035, E040–E042), it is currently wrong, and one
  case (E035) has two project documents contradicting each other as well as the compiler.
- **Q3 — the one that matters:** yes. 35.1% of the corpus the project uses to assert "this is
  correct Sounio" fails to check clean under Madaros, ~530 of those 596 failures confirmed or
  high-confidence false positives via an independent-engine oracle, collapsing to five shared
  defects (a hardcoded reserved word usable as an identifier, an ambient-vs-explicit-import
  asymmetry, an import-path resolution bug, unsupported `Seq` sugar, and one large-file parser
  failure of unproven mechanism). None of the five produce a message that tells the user the
  compiler is at fault.

An error surface with no reconciled registry, contradictory documentation, and a corpus that
fails 35% of the time on code asserted correct is not "language has diagnostics with gaps." It is
a compiler that has not yet decided what its own error codes mean, being asked, by its own
project's convention, to be believed.

## 8. Open items for follow-up

- Root cause A's exact mechanism (parser-state ceiling vs. undiscovered construct defect near
  function #~2160 of `theorem/portfolio.sio`) is not proven, only the disagreement is.
- 60–70 files in the 194-disagreement set are confirmed compiler-disagreement hits not yet
  clustered to a shared root cause.
- `self-hosted/diagnostics/mod.sio` — a fully-designed 40-code registry with severity levels and
  a `has_suggestion`/`suggestion_text` field — has zero callers anywhere in the compiler. Per
  CLAUDE.md principle 2 (stubs are not gaps), this is reported as designed-but-unwired, not
  something to delete; it is the shape a reconciled registry could grow into, not evidence nobody
  thought about the problem.
- `scheduler_machine_reorder.sio` and `ptx_maxntid.sio` (both currently outside
  `SOUNIO_STDLIB_PATH`) were not resolved to compiler-bug vs. environment-path issue; time-boxed
  out of this sweep.
