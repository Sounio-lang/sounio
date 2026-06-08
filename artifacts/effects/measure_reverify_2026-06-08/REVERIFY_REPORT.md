# Effects MEASURE phase — re-verification report

**Date:** 2026-06-08
**Branch:** `claude/effects-enforcement` (worktree `/workspace/sounio-effects`), tip `c7bbc771f`
**Instrument:** `artifacts/effects/souc-eff-warn.elf` (Madáres v0.80.0, warn-mode toggle = 1)
**Raw data:** `w035_live_sweep.tsv`, `live_module_set_107.txt` (this directory)

## Verdict

**Two robust findings, one retraction.**

1. **HOF centrality is first-order (robust).** This is the one number that could have changed the
   plan; it does not. The compiler flip needs no M4 row-poly. P1 remains the sensible default
   annotation strategy — but see the caveat below: gap *and* depth are now both unmeasured, so
   "P1 decisively beats P2" is downgraded to "P1 is the reasonable provisional default".

2. **Retraction — the headline binary instrument is UNRELIABLE.** An earlier draft of this report
   claimed the 64 isolation-checkable live files are "fully IO-annotated (binary-verified,
   W035 = 0)". **That claim is false.** Standalone `--check` **truncates / suppresses W035
   emission after an early point in large files** (demonstrated below). The OK/W035 = 0 result for
   a big file means the checker *stopped emitting warnings*, not that the file is clean. Concretely:
   `check/knowledge_context.sio:2920` is a genuine `print()` call in a function declaring
   `with Mut, Panic, Div, Alloc` (no IO) — a real violation — yet its file is OK with W035 = 0.

3. **Net on the gap:** it is **not reliably measured by ANY current tool**. The binary truncates;
   the bundle path is blocked (271-wall); grep heuristics produce both false positives *and* false
   negatives (the workflow's two heuristics gave 0 vs 461 — and the 461 had a real true-positive,
   `knowledge_context:2920`, alongside false positives like `make_hyper_expr_info`). The earlier
   "~87 % annotated / ~160-fn gap" figures were never binary-verified and remain unverified.

4. **Flip-gating soundness — one bug (now fixed) and one design-coverage gap (not a bug).**
   (a) **Parser bug — FIXED** (`parse_module_item`, items.sio:189, commit `526b0091f`): a leading
   `module NAME` declaration looped to EOF on a `Newline` token the lexer never emits, **swallowing
   the file's whole body into one discarded item** — so standalone `--check` of a module-headed file
   checked *nothing* (effects *and* type errors silently passed). Now consumes only the path. This is
   what made the `knowledge_context.sio` standalone result misleading; it is **not** the 271-wall.
   (b) **Imported bodies are not re-checked on import — this is SEPARATE COMPILATION, working as
   designed, not "under-enforcement"** (corrected diagnosis). `import_typecheck_main` seeds imported
   *interfaces/signatures* (`checker_boot4_seed_imported`) and checks the *target's* body — the
   standard model: a module is checked when it is the target; importers trust the validated
   interface. Recursively body-checking imports would re-check shared deps once per importer and
   reverse the deliberate #2 design — the wrong fix. The genuine flip requirement is **coverage**:
   every module body checked *once* with its imports seeded. Naive "`--check` each file" fails
   (a module checked in isolation TYPEFAILs on unresolved imports — `knowledge_context` → 184
   errors). The sound shape is to generalise the existing seed-imports-then-check-*main* flow to
   take **any** module as target, then a gate that loops it over all modules under `toggle = 2`.
   That is a contained **coverage gate**, not a checker change — and it is downstream of the 271-wall
   (the flip can't be self-host-verified until gen-N can re-parse `main.sio`), so not on the critical
   path. Left as a proposed dispatch, not patched (§8). **The `module`-demotion fix touches none of
   this.** (Details in §2.)

> **Reconciliation note (concurrent session).** While this re-verification ran, another session
> advanced the same topic: the `main.sio:6808` comment-scanner is **already fixed**
> (`d2d0827a1`), and fixing it **unmasked a deeper wall** — gen-N reports **271 parse errors** on
> `main.sio` + its 72 imports, root-caused to **lexer keyword collisions** (e.g. `effect` is
> reserved as `TokenKind::Effect` but used as an identifier in the compiler's own source;
> `check.sio`'s 38 errors are a separate, unpinned cause). This is **exactly the same
> phenomenon** this report measures as the unmeasurable live files (43 of 107 + `main.sio`) — independent
> corroboration from a different angle. Consequently the "fix the scanner first" framing in
> §5/§6 below is **superseded**: the scanner is done; the real prerequisite is the 271-error
> parser self-host gap, and the flip is now an A/B/C operator decision (see end of §5).

---

## 1. What was re-verified, and how

The prior MEASURE used per-function `awk`/`grep` heuristics over a loosely-scoped file set.
This pass upgraded the methodology on three axes:

1. **Instrument = the compiler's own checker, not grep.** Run `souc-eff-warn.elf --check <file>`
   and count emitted `warning[W035]` (the live `require_effect` print→IO check), instead of
   pattern-matching source text. **⚠️ This upgrade FAILED for large files — see §2: standalone
   `--check` truncates W035 emission, so the binary is not a reliable gap instrument either.**
2. **Scope = the true live module set.** Transitive closure of `use` from
   `self-hosted/compiler/main.sio`: **107 files** (fixpoint at iteration 4), vs the 64 direct
   imports the prior measure implicitly used.
3. **Instrument calibration.** Verified W035 fires **per use-site, not per function**
   (two `println` in one fn → 2 warnings), fires on both single- and multi-module paths, and
   reports location as `at 0` (no line/column in this build).

---

## 2. The sweep, and why it does NOT measure annotation

Serial sweep of the 107 live files (`stdin=/dev/null`, ~9 s total):

| Status | Files | Raw result |
|---|---:|---|
| `OK` | **64** | all reported W035 = 0 |
| `TYPEFAIL` (checker bailed early) | 36 | 26 W035 total |
| `PARSEFAIL` (never reached check) | 7 | 0 W035 |

**Do not read `OK / W035 = 0` as "annotated".** It is not "positional truncation" (an earlier
draft's guess) — the root cause is a **parser bug**: a leading `module NAME` declaration causes the
rest of the file to be **dropped from the AST**, so nothing after it is checked at all (effects
*and* type errors).

**Root cause — `parse_module_item` consumes the whole file (items.sio:189):**

```
while parser_peek(p) != TokenKind::Newline && parser_peek(p) != TokenKind::Eof { p = p.advance_raw() }
```

The lexer emits **zero `Newline` tokens** (verified: 0 emit sites in `self-hosted/lexer/`), so the
loop never sees a `Newline` and runs to **EOF**, swallowing every following declaration into one
discarded `ItemUnit`. Evidence:

| Probe | Result |
|---|---|
| `module a::b` + 3 fns | "Main file: **1** item" (the 3 fns vanish) |
| no `module` + 3 fns | "Main file: **3** items" |
| `fn x` → `module a::b` → `fn y(){print}` | "**2** items" (fn x + the module item that ate fn y); W035 = 0 |
| `module foo::bar` + `fn typeerr(){let x:i64=true …}` | **`check: OK`** — even a real **type error** is dropped |
| same without `module` | **`type checking failed`** (E001 caught) |

So a `module` header silently removes the entire file body from checking. After refuting the
obvious candidates, the bisection pinned the trigger to **line 1** (the `module` declaration):

Refuted candidates (probes, warn binary):

| Candidate | Refuting probe |
|---|---|
| Warnings-after-first-error | small file with a real type error (E001) before a print-no-IO fn → W035 **still fires** |
| Size / function-count threshold | synthetic clean files of 20/60/120/**200** fns + end-violation → all **fire** |
| Single-module / isolation artifact | clean fn referencing an *unresolved* imported type + print → **still fires** |
| The `use` block | removing all `use` lines but keeping `module` → end-violation **still suppressed** |

Confirming the trigger:

| Probe | W035 / E035 |
|---|---|
| violation inserted **before** the `module` line (top of file) | **fires** |
| violation inserted **after** the `module` line | **suppressed** |
| `knowledge_context` with the `module` line **deleted** + end violation | **fires** (W035 = 3) |
| 4-line minimal: `module foo::bar` + `fn v(){print("x")}` | **W035 = 0 / E035 = 0**, `check: OK` |
| same 4 lines **without** `module` | **W035 = 1 / E035 = 1**, check fails |

So **any file `--check`ed as a main file while headed by a `module` declaration has its entire body
dropped from checking** — effects *and* type errors — while still reporting `check: OK`.

This is a **separate bug from the 271-wall**, despite both involving `module`: the 271-wall is
`module` used as an *identifier* (parser/lexer, expression/pattern positions); this is the `module`
*declaration* path (`parse_module_item`) relying on a non-existent `Newline` token.

**Scope of the parse bug:** it bites the **standalone `--check <module-file>`** entry. The
import/build path must parse module-headed files correctly — otherwise the real compiler could not
use any function in `check/*`, `native/*`, etc. (it does), so those modules are parsed via a path
that does not trip this. The blast radius of *this bug* is therefore the standalone-check sweep
itself: the 15 / 107 live files with a `module` header (and 98 of `self-hosted/`, 112 stdlib, 20
examples) check as empty in isolation.

**Consequence for the measurement:** the W035 sweep's `OK / W035 = 0` is meaningless for any
module-headed file (the body was never parsed), and `knowledge_context:2920` proves it misses real
violations. The "instrument upgrade over awk" failed; the per-file isolated binary is *not* a
reliable gap instrument. The gap remains unmeasured.

### Does the `module`-demotion fix clear this suppression? — NO (verified)

An earlier "fix lead" guessed the suppression might share a root with the concurrent session's
271-wall `module`-keyword demotion. **Verified false:**

- The demotion (uncommitted in `/tmp/kw-demote`) touches only `parser/exprs.sio` and
  `parser/patterns.sio`, adding `TokenKind::Module|Effect|Study|Is` to the *identifier* positions —
  it lets `module` be used as a variable/field/expression (the 271-wall). Its own plan states:
  *"keep [`module`] item dispatch only at top-level item position"* — i.e. it **deliberately
  preserves** the `module NAME` declaration grammar.
- The suppression lives in `parse_module_item` (`parser/items.sio:189`) — the **declaration** path —
  which the demotion does not modify. After demotion, `module check::knowledge_context` still routes
  to `parse_module_item`, which still loops to EOF (no `Newline` token) and still swallows the file.

So the two are orthogonal: demotion fixes `module`-as-identifier; it does nothing for
`module`-declaration-eats-the-file.

### ✅ Fix applied + verified (2026-06-08)

`parse_module_item` (items.sio:189) was changed to consume only `module` + the path segments via the
shared `parse_type_path()` (plus an optional trailing `;`), instead of looping to the never-emitted
`Newline`:

```sounio
if parser_peek(p) == TokenKind::Ident || tk_is_keyword(parser_peek(p)) {
    let pair = p.parse_type_path()
    p = pair.0
}
if parser_peek(p) == TokenKind::Semi { p = parser_advance(p) }
```

Rebuilt via `souc-build-lock.sh ./bin/souc main.sio` (163 s) and verified:

| Check | Before | After |
|---|---|---|
| `module a::b` + 3 fns → item count | 1 | **4** |
| `module foo::bar` + print-no-IO → W035 | 0 | **1** |
| `module foo::bar` + type error → outcome | `check: OK` | **type checking failed** |
| `knowledge_context.sio` standalone | "1 item", `check: OK` | **"158 items", W035 = 2** (real violations now seen) + 184 type errors (its imports are unresolved in *standalone* check — the honest result) |
| no-`module` files (regression) | — | unchanged |
| **`release_gate.sh` (20 gates)** | — | **all PASS** (parser_sweep 525/525, multimodule 27/27, effects 8/8, capgate 32/32, …) |

**Scope of the fix:** it clears the **standalone `--check`** suppression (and so makes the per-file
W035 sweep actually see module-file bodies). It does **not** change what happens when a module is
*imported*: a `main` importing a `module`-headed lib with a violation still gives W035 = 0 with the
fixed binary — but (see next subsection) that is **separate compilation by design, not a bug**.
Committed as `526b0091f` on `claude/effects-enforcement` (one-function change in `parser/items.sio`).

### "Import-path under-enforcement" — re-diagnosed as separate compilation (NOT a bug)

`import_typecheck_main` (module_frontend.sio:3551) does `checker_boot4_alloc_seed_main(main)`, then
BFS-**seeds imported signatures** (`checker_boot4_seed_imported`) and checks **main only**. So an
imported module's body is never re-checked by its importer — which is the standard separate-compilation
model (a module is checked when it is the target; importers trust its validated interface). Making
imports recursively body-check would re-check shared deps once per importer and reverse the deliberate
#2 import-typecheck design — the wrong fix; not patched (§8).

> ⚠️ **Retracted data point:** an earlier draft cited a `--native-v2-compile` run of main+lib giving
> "E035 = 0" as build-path evidence. That run died with `ir_summary_failed` *before* any check could
> fire, so it shows nothing about enforcement. The real basis is the code above
> (`import_typecheck_main` checks main only), which is sound.

**The genuine flip requirement is coverage, not a checker change:** every module body checked *once*
with its imports seeded. Naive "`--check` each file" fails (isolated modules TYPEFAIL on unresolved
imports — `knowledge_context` → 184 errors). The contained, sound shape: generalise the existing
seed-imports-then-check-*main* flow to take **any** module as target, then a gate looping it over all
modules under `toggle = 2`. This reuses the #2 machinery and is downstream of the 271-wall (the flip
can't be self-host-verified until gen-N re-parses `main.sio`), so not on the critical path. Proposed
as a dispatch; not implemented this session.

> **🛑 Flip-gating soundness risk — REAL, but re-grounded (toggle = 2 rebuild, 2026-06-08).** I
> rebuilt the compiler with `effects_enforcement_mode() -> 2` →
> `artifacts/effects/souc-eff-error.elf` (retained; toggle reverted to 1 after). Error-mode enforces
> correctly on plain files (println-no-IO at top → `error[E035]`, check fails; synthetic 200-fn
> end-violation → E035). **Correction to an earlier draft:** the `knowledge_context.sio` standalone
> result (E035 = 0) is *confounded by the `parse_module_item` bug above* — that file's body is never
> parsed, so its 0 is not evidence about enforcement. Re-grounded on the **multi-module path**
> instead:
>
> | Test (error-mode binary) | Result | Meaning |
> |---|---|---|
> | main (no `module` header) imports a `module`-headed `lib` with a print-no-IO fn; lib's `lib_ok` is called and resolves | **`check: OK`, 0 × E035** | imported module body parsed (import resolves) but **not effect-checked** |
> | a print-no-IO fn directly in a non-`module` main | `error[E035]`, check fails | the root's own body **is** enforced |
>
> So for a self-host flip: `main.sio` (no `module` header) → its ~1,418 functions **would** be
> enforced as the target; the imported modules (`check/*`, `native/*`, …) are seeded as interfaces,
> not re-checked by their importer. **Re-diagnosis (see subsection above): this is separate
> compilation, not under-enforcement** — each module is enforced *when it is the target*. The flip
> requirement is therefore a per-module **coverage gate** (check every module once with imports
> seeded), not a checker change, and it is downstream of the 271-wall. So: **toggle = 2 must not be
> trusted for a self-host-verified flip until a coverage gate checks every module body** — but the
> mechanism is a gate to build, not a soundness bug to patch. Enforcement on standalone non-`module`
> programs (and on each module checked as the target) is sound.

### Coverage / unmeasurability (arithmetic)

The 107-file live set = transitive closure of `use` from `main.sio`; **`main.sio` itself is not in
the 107** (nothing imports the root). Within the 107: **43 are unmeasurable in isolation**
(7 PARSEFAIL + 36 TYPEFAIL) and 64 parse+typecheck but are subject to the truncation above. Plus
`main.sio` as a 108th unmeasurable file:

- `main.sio`: 27,712 lines, 1,418 fns, 116 declaring IO; cannot be checked standalone (271 parse
  errors) or bundled (was the `6808` scanner — now fixed — and the deeper 271 keyword-collision
  wall; see Reconciliation note + §4).

---

## 3. HOF centrality — confirmed first-order (the decision-critical number)

Across the 107 live files (≈173 k LOC):

| Construct | Count |
|---|---:|
| fn-typed params (`: fn(`) | ~0 (only a comment in `parser/types.sio`) |
| fn-pointer type aliases (`type X = fn(`) | 0 |
| closures passed as args (`\|p\| {...}`) | 0 |
| map/fold/reduce/filter combinators | 0 |
| fn return types (`-> fn(`) | 0 |

Corpus context (matches prior memory 4 / 261 / 31): self-hosted (all) = 4, stdlib = 264,
examples = 31. The 4 self-hosted hits are outside the live set.

⟹ **The live compiler is first-order. The compiler flip needs no M4 row-poly.** Only a
full-corpus flip (stdlib's 264 fn-typed params) would put M4 on the critical path. This is the
finding that could have changed the plan; it does not.

---

## 4. The gap is genuinely unmeasured — and why (CORRECTION)

Two independent grep heuristics were run this pass and **contradicted each other**:

- Heuristic A (census over the unmeasurable set): **0** fns calling print without IO.
- Heuristic B (depth proxy over all 107): **461** fns (14.2 %) calling print without IO,
  with named counterexamples *inside OK files*.

Heuristic B was checked against source — and it is **noisy in BOTH directions**, not simply refuted
(an earlier draft wrongly called it fully refuted; that was based on reading only ~4 lines of each
function body):

- `check/mod.sio:296 make_hyper_expr_info` (full span 296–313) — declares `with Mut, Panic, Div`;
  **no print call** in the full body → B **false positive**.
- `knowledge_context.sio:2920 …semantic_probe_items` (full span 2920–2933) — declares
  `with Mut, Panic, Div, Alloc`; **DOES call `print("…")` at line 2930** → B **true positive**.
  Yet the file is `OK / W035 = 0` in the sweep — the binary **missed a real violation** (this is the
  truncation of §2).

So B has both false positives and at least one confirmed true positive that the binary missed. Its
85.8 %-annotated figure uses the same noisy method as the earlier memory's "~87 %", and Heuristic
A's "0" is the same kind of text heuristic — none is trustworthy.

**Conclusion:** every available instrument is unreliable here — the binary truncates W035 on large
files; grep heuristics mis-count both ways; the bundle path is blocked. **The live-compiler IO gap
is genuinely unmeasured.** It can only be settled by a working bundle-path check (which needs the
271-wall fix), not by per-file isolation or grep.

### PARSEFAIL root-cause

The analysis agent classified ~4/7 as "6808-comment-scanner class", but that classification is
**superseded and wrong**: the 6808 scanner is already fixed (`d2d0827a1`), so these PARSEFAILs
are not scanner mis-fires. They are the **271-error gen-N self-host wall** — lexer keyword
collisions (the concurrent session pinned `effect` as one confirmed cause; `check.sio`'s 38 are a
separate unpinned cause). The PARSEFAIL parse-error counts this sweep observed
(`check/check.sio` 38, `module_native_driver` 32, `module_frontend` 26, `hlir_to_gpu` 42,
`hlir/ir` 19, `ir/normalize` 13, `ir/dce` 2) match the concurrent per-module bisect and stand
as the file-level membership of that wall.

---

## 5. Sharpened conclusion: the real prerequisite is the 271-error parser self-host gap

The scanner fix is done (`d2d0827a1`); it was *not* the deep blocker. The genuine
critical-path prerequisite for a **self-host-verified** flip is closing the **271-error
gen-N parser self-host gap** (lexer keyword collisions), because it gates, in order:

1. **True gap measurement** — only the bundle path (`main.sio` + transitive imports) yields a
   checker-verified W035 count across all 1,418 main.sio fns and the 36 TYPEFAIL modules. That
   path needs a gen-N compiler that can parse main.sio, which the keyword collisions prevent.
2. **Checker-verified annotation** — annotating any of the 44 unmeasurable files cannot be
   confirmed by re-running `--check` while they still fail to parse/type-check; the bundle path
   (an enforcing parser that reads main.sio) is required.
3. **Self-host flip verification** — toggle = 2 validated by a clean self-build needs the bundle
   path, which needs the parser-gap fix.

But note the concurrent session's important nuance (topic-file line 94): the flip is **still
shippable without** closing this gap, because `bin/souc` (gen N-1, no enforcement) does the build
so it never breaks, and gen-N then enforces mandatory effects on every program it *can* parse
(user code, examples, gate witnesses). What stays unverifiable is the annotation-completeness of
the **compiler's own source**. Hence the pending **operator A/B/C decision**:
(A) flip mode = 2 anyway — mandatory for user code, self-host unverified;
(B) keep warn-mode, treat the parser gap as a prereq, pause;
(C) pivot to fixing the 271-error parser self-host gap first (a substantial parser workstream).

---

## 6. Revised NEXT sequence

0. ~~Fix the 6808 scanner~~ — **DONE** (`d2d0827a1`).
1. **Operator A/B/C decision** (see §5): ship the flip now (A), pause in warn-mode (B), or fix the
   271-error parser self-host gap first (C). This decision, not measurement, is the live blocker.
2. *If self-host verification is wanted (A-verified or C):* close the keyword-collision parser gap
   — preferred direction is contextual keywords (reserve only in declaration position) over
   mechanical identifier renames. Then attempt the bundle check (`main.sio` + transitive imports)
   and capture the **true** W035 count + per-site list — the first trustworthy gap number.
3. Annotate from that W035 list via `effects_annotate.sh` (still UNVERIFIED), P1 iterate-rebuild
   rounds (2.7 min each); the corpus's heavy over-declaration keeps rounds few.
4. Flip the compiler toggle to 2; verify (self-host: gen2 == gen3; or A-path: effects-gate
   error-mode + `release_gate.sh` 20 gates + examples).
5. DEFER stdlib/examples + M4 row-poly (264 fn-typed params) to a later phase.

---

## 7. Caveats — what this does NOT measure

- **Transitive depth.** Nothing here measures the call-graph cascade; per-file isolation can't
  (imported signatures unresolved). "Depth shallow" is inferred from over-declaration, not measured.
- **Per-use-site, not per-fn.** W035 counts call-sites; the 26 TYPEFAIL W035 are early-bail
  partials — not a meaningful bound.
- **Heuristic census is text, not the checker.** The "0 gap in the unmeasurable set" is
  "no obvious violation found by regex", not "checker-verified gap-free".
- **No reliable per-file gap statement exists** with current tools. The binary truncates W035 on
  large files (§2), so `OK / W035 = 0` does not mean annotated; the bundle path is blocked; grep
  mis-counts both ways. The gap is unmeasured until the 271-wall is fixed and a bundle check runs.
- **The only robust finding** is HOF first-order (§3) — a negative grep result, hard to fake and
  corroborated by the prior measure's 4/261/31. Everything quantitative about the IO gap is pending.
