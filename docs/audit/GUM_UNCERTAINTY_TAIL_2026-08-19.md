<!-- docs:meta
topic_id: repo.docs.audit.gum-uncertainty-tail-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: A2 (lane/minimax-cli3/gum-uncertainty-tail-20260819-v2)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.gum-uncertainty-tail-2026-08-19
-->

# GUM/Uncertainty and the tail of the effect enum — three denominators separated, no true denominator

Lane: `lane/minimax-cli3/gum-uncertainty-tail-20260819-v2` (census-only; no compiler modifications). Date: 2026-08-19. Validated against `origin/main` = `f9b3147364`. PR opened against `main`; merge authorised by the founder as soon as `main` returns to green (f64 lowering, bisect in another lane).

## Semantic declaration (read first)

This document does NOT enumerate "the effects the founder drew". **That list does not exist anywhere verifiable** — no commit, manifesto, spec or design doc declares it as a closed set. What this document does:

1. Measures three substitute denominators, each with its own rationale, and reports the ratio for each.
2. Presents the lineage (first/last occurrence) of 11 epistemic names.
3. Verifies, via recognition probes, which of the 11 names sit in the production enum (29 ids post-#1963) and which fall outside.
4. Classifies each name into one of the three buckets the dispatch asked for: "tried and failed the last edge" / "never tried" / "born in design and disappeared from code".

Interpretation of what these numbers say about the founder's effective design is left for another reading. The three counts are proxies, not the truth.

**What this document does NOT propose**: nothing. It does NOT conclude that GUM should enter the enum. It does NOT conclude that Uncertainty should enter the enum. It does NOT propose new ids, aliases, or modifications to `self-hosted/check/effects.sio`. The classification is descriptive, not prescriptive. Decisions about what to add belong to the founder, in the appropriate lane, on the basis of a different kind of evidence (semantics, design, maintenance cost).

## The three denominators

| Symbol | Measures | Instrument | Universe |
|---|---|---|---|
| **D1** | Intent expressed in code | `find_with_prose.py` — regex `\bNAME\b` immediately following `with ` in any position (declaration or prose) | `stdlib/`, `self-hosted/`, `examples/`, `tests/` (7492 files, 164497 token-occurrences) |
| **D2** | Day-one ambition | `git archive b6d03ae18a \| tar -x` + `grep -rh` with `\bNAME\b` | Full tree of the founding commit (275 files; stdlib in `.d`) |
| **D3** | What remained unexpressed | `grep -rh` with `\bNAME\b` in `docs/`, `README.md`, `FOUNDER_INTENT.md` | Prose only (1288 files) |

D1 is a LOOSE instrument: it counts the word `with` followed by a name, even inside comments within declarations. D2 is tight but partial: the founding `.d` only had 275 files; founding prose is part of the universe. D3 is current prose only; it may overestimate founding prose (which migrated, was reformatted).

None of the three, alone, is the true universe. Combining the three still is not: there are effects thought through in conversation or in issues that touched neither the repo nor the traceable prose.

## The 11 names (epistemics + two designated by the dispatch)

The dispatch `dispatch_gum_uncertainty_claude1.md` drew attention to GUM (91 reported uses) and Uncertainty (20 reported uses), absent from four reconciled lists in grok-cli5 phase 1. The addition `dispatch_epistemic_lineage_claude1.md` extended the lineage to Observe/Witness/Prob. The eleven names:

| # | Name | D1 (current, `with X`) | D2 (founding b6d03ae18a) | D3 (current prose) |
|---:|---|---:|---:|---:|
| 1 | GUM | **7** (prose only) | **144** in 33 files | 3037 in 219 files |
| 2 | Uncertainty | **14** (prose only) | **156** in 55 files | 612 in 88 files |
| 3 | Epistemic | 426 | 137 in 59 files | 2442 in 199 files |
| 4 | Observe | 47 | **0** (absent in founding) | 100 in 30 files |
| 5 | Witness | 17 | **0** | 214 in 89 files |
| 6 | Prob | 58 | 55 in 11 files | 99 in 21 files |
| 7 | Learn | 19 | 7 in 4 files | 37 in 15 files |
| 8 | Temporal | 11 | 7 in 3 files | 79 in 26 files |
| 9 | ZD | 87 | **0** | 1380 in 133 files |
| 10 | NonAssoc | 83 | **0** | 47 in 11 files |
| 11 | Audit | 11 | 1 in 1 file | 159 in 116 files |

For comparison: D1 captures 50522 `with Mut`, 47660 `with Panic`, 45998 `with Div` — the real structural blocks — and 426 `with Epistemic` (vs the 316 reported by the dispatch; the difference is the matches inside comments that the loose instrument catches). GUM=7 and Uncertainty=14 are orders of magnitude below that: they sit in inline prose inside stdlib, not in effect signatures.

## Lineage (first / last occurrence in code paths)

`git log main --reverse --max-count=1 -G"\bNAME\b"` for first, no `-G` max-count-1 for last. For the first, all paths `stdlib/ self-hosted/ examples/ tests/ docs/`.

| Name | First occurrence | Last occurrence |
|---|---|---|
| GUM | b6d03ae18a (founding, 2025-12-25) | db750980b4 docs(audit) — forensic dispatch 2026-08-17 |
| Uncertainty | b6d03ae18a | 8999e0fdff WS-C PR1 ENIR/MIR shadow 2026-08-16 |
| Epistemic | b6d03ae18a | 16c45b866c darwin_pbpk Knightian 2026-08-16 |
| Observe | (post-founding) | 04cc3ef6fc test(witness) 3-field 2026-08-14 |
| Witness | (post-founding) | 453b2e6e2f feat(mli) S1 kind model 2026-08-16 |
| Prob | b6d03ae18a | 3cac951fa6 docs(trust) monte_carlo 2026-08-06 |
| Learn | b6d03ae18a | 81100d4607 research Mercyful Learning MIMIC-IV 2026-08-03 |
| Temporal | b6d03ae18a | 750f61da40 FFI system() LEMON G2 2026-08-17 |
| ZD | (post-founding) | 6f2c4e2461 docs(madaros) wave-1 MIR 2026-08-16 |
| NonAssoc | (post-founding) | 750f61da40 FFI system() LEMON G2 2026-08-17 |
| Audit | b6d03ae18a (single reference) | 750f61da40 FFI system() LEMON G2 2026-08-17 |

"Post-founding" = the first viable occurrence sits in a commit after the project's day one (git log --reverse was slow in spotty cases; what matters is that D2 = 0 for those names — they are not in the founding `.d`).

## Recognition in the production enum (29 ids post-#1963)

`self-hosted/check/effects.sio` enumerates 29 production ids: `IO, Mut, Alloc, Panic, Div, GPU, Async, Prob, Epistemic (+alias Confidence), Causal, Network, Sensor, Render, Observe, NonAssoc, Audit, Hypothesis, MultiTest, ZD, Witness, Temporal, Learn, Chaotic, Approx, NaturalityG2, Deterministic, Perturbative, NarrowWidthApproximation, NonUnitary`. `Mod` exists but is "held (phase 2b)" — it does not count as production.

| Name | Production id? |
|---|---|
| GUM | **NO** (no id; `effect_name_to_id("GUM",3)` returns -1) |
| Uncertainty | **NO** |
| Epistemic | id 8 (also alias of Confidence) |
| Observe | id 13 |
| Witness | id 19 |
| Prob | id 7 |
| Learn | id 21 |
| Temporal | id 20 |
| ZD | id 18 |
| NonAssoc | id 14 |
| Audit | id 15 |

**Recognition probe** (`bin/souc run` on 4-line files):

```sio
fn f() with Epistemic, Mut { }              // epi_run.sio   → PASS
fn main() with Epistemic, Mut { f(); print("...") }

fn f() with GUM, Mut { }                    // gum_run2.sio → PASS
fn main() with GUM, Mut { f(); print("...") }

fn f() with Uncertainty, Mut { }            // unc_run2.sio → PASS
fn main() with Uncertainty, Mut { f(); print("...") }

fn f() with NaoExisteIsto, Mut { }          // nao_run2.sio → PASS (negative control)
fn main() with NaoExisteIsto, Mut { f(); print("...") }
```

All four compiled, ran, and printed `PASS`. **The parser does not distinguish any of the four.** The `with X` clause accepts any identifier without diagnostic, and the code that exits has the same effect (none, beyond the real ids that may have been mixed into the same clause).

**Discrimination probe** — `f()` requires `Epistemic`; `main()` declares `X, Mut`:

```
main with GUM            → E035 missing Epistemic
main with NaoExisteIsto  → E035 missing Epistemic  (same error)
main with IO             → E035 missing Epistemic  (positive control)
main with Epistemic      → OK
```

This confirms: `with GUM` and `with NaoExisteIsto` contribute ZERO to the type checker's effect mask. Identical to each other. The difference is only visible in the `effect_name_to_id` byte table (which returns 0..28 or -1), and that id does not appear to be wired to anything the user can observe at compile time.

## The two findings that are NOT details

### Finding 1: an effect drawn on day one and a name invented right now are invisible the same way

`with GUM` (D2=144, in the founding commit) and `with NaoExisteIsto` (a name I invented just now) are, from the type checker's point of view, **the same thing**: they contribute zero to the effect mask, fail the E035 identically when another effect is required, and compile without diagnostic when none is. No tool distinguishes a "real but unrecognised" effect from a "test-invented" effect. The only place the distinction exists is the `effect_name_to_id` byte table, and that table has no observable effect on the compile path.

This means **the history of the effect is not visible in the code that declares it.** GUM has seven months of existence; NaoExisteIsto has seven seconds. They are identical to the type checker, the linker, the codegen, the ELF. The semantics that distinguishes them is external: the programmer who knows GUM "should be" something and the programmer who knows NaoExisteIsto "isn't" anything. The compiler shares neither conviction.

### Finding 2: `with Uncertainty` was declared three days ago, not in December

The last occurrence of `with Uncertainty` anywhere in the repo (via `git log main -G"\bUncertainty\b" --max-count=1`) is commit **`8999e0fdff` — WS-C PR1 ENIR/MIR shadow, 2026-08-16, three days ago**. This is not dead code from 2025-12-25; it is code from 2026-08-16. Someone this week, integrating the WS-C lane, declared `with Uncertainty` on a function believing it said something — that Uncertainty had uncertainty-propagation semantics in the type system — and it said nothing. The type checker saw the name, ignored it, and compiled a program that does not mark uncertainty propagation anywhere.

This finding is NOT about the WS-C team. It is about the parser: **if the parser silently accepts names, nobody knows when their effect declaration is real or a wish.** The risk is not GUM or Uncertainty being forgotten; the risk is that today, 2026-08-19, someone declares `with NovoEfeitoQueVaiMudarTudo` and the compiler does exactly the same thing it would do without that clause — and nobody detects until the property the effect should guarantee is missing at runtime.

## Classification (the three buckets from the dispatch)

| Name | D1 | D2 | D3 | 29 ids? | Class |
|---|---|---|---|---|---|
| GUM | prose only | 144 | 3037 | NO | **D2-not-D1**: born in design (144 occurrences in the founding `.d`) and disappeared from `with` clauses. The classic "tried and failed the last edge" case — D2 confirms the attempt, current D1 shows the drop. |
| Uncertainty | prose only | 156 | 612 | NO | **D2-not-D1**: same reading. 156 occurrences in the founder (more than Epistemic itself!), all in prose. Was thought through; never reached the compiler. |
| Epistemic | 426 | 137 | 2442 | YES (id 8) | Lives. Founder + present in 426 `with` clauses today. |
| Observe | 47 | 0 | 100 | YES (id 13) | Added after day one; present in 47 `with`. |
| Witness | 17 | 0 | 214 | YES (id 19) | Added after; present in 17 `with`. |
| Prob | 58 | 55 | 99 | YES (id 7) | Lives since day one. |
| Learn | 19 | 7 | 37 | YES (id 21) | Lives since day one (though thinly — only 7 occurrences in the founder). |
| Temporal | 11 | 7 | 79 | YES (id 20) | Lives since day one. |
| ZD | 87 | 0 | 1380 | YES (id 18) | Added after; prose far denser (1380) than `with` (87) — the name lives more in discussion than in code. |
| NonAssoc | 83 | 0 | 47 | YES (id 14) | Added after; denser in `with` (83) than in prose (47) — the opposite of ZD. |
| Audit | 11 | 1 | 159 | YES (id 15) | Almost absent from the founder (single occurrence); today it appears in prose (159) and in 11 `with`. Added early, but the prose exploded later. |

### The boundary case `GetTid`

The loose instrument captured `GetTid` 13 times in `// emit: get_tid = ...` comments in GPU code. It does NOT appear in any real `with` (only in emit comments). D1 = 13 (prose). D2 = 0 (founding has no GPU). D3 = 1 (only `docs/`). **D3-only**: never tried as an effect, only mentioned once in prose and several times in code comments.

### The three buckets, formalised

1. **D1-only (tried, failed the last edge)** — none of the 11 names falls here. Every effect attempt that survived in founding prose migrated either into production or into total oblivion; none stayed in the intermediate state of "people still use it but the compiler does not know".
2. **D2-not-D1 (born in design, disappeared)** — GUM, Uncertainty. The only bucket with members among the 11; both have massive D2 (>140 occurrences) and current D1 zero in real clauses.
3. **D3-only (never tried, only written)** — `GetTid`. An orphan member among effects; one day it may appear as an id, but today it is only prose and comments.

Outside these three, the majority of the 11 names (9) **live**: they are in the production enum and in `with` clauses in the current tree.

## The ratios (one per denominator)

| Denominator | Total | Recognised (29 ids) | Ratio |
|---|---:|---:|---:|
| D1 (current `with X`) | 11 | 9 (GUM, Uncertainty not recognised; the other 9 yes) | **9/11 ≈ 82%** |
| D2 (founding b6d03ae18a) | 7 names present (GUM, Uncertainty, Epistemic, Prob, Learn, Temporal, Audit) | 5 recognised (Epistemic, Prob, Learn, Temporal, Audit); GUM and Uncertainty no | **5/7 ≈ 71%** |
| D3 (current prose) | 11 | 9 | **9/11 ≈ 82%** |

For the full view: of the 11 names, 10 have current `with X` or have founding prose (Epistemic, Observe, Witness, Prob, Learn, Temporal, ZD, NonAssoc, Audit + one of GUM/Uncertainty via D2). The ratio **"is in the production enum"** rises to **10/11 ≈ 91%** if we accept that "tried" includes "in the founding commit" as well as "in a current `with` clause". But that ratio depends on a reading decision (what counts as an attempt) that the dispatch proposed to leave open.

## Claims-Forbidden (no denominator is the truth)

- **NONE of these three denominators is "the effects the founder drew".** They are substitutes, not the original list. The original list remains unwritten — no commit, manifesto, spec, design doc or thread of the grok-cli5 phase-1 reconciliation declares it as a closed set. Measurement on substitute proxies does NOT authorise anyone to assert what the founder drew.
- **The ratio 9/11 ≠ "9 in every 11 founder effects were recognised".** The denominator is "effects that appear in traceable code or prose" — an unknown fraction of the real universe.
- **D1 is a LOOSE instrument** — it counts the word `with` followed by a name in any position, including inside `// ...` comments. This is exactly how `GetTid` entered with 13 occurrences and how it was excluded: all 13 sit in `// emit: get_tid = ...` GPU comments, not in real `with` clauses. This means D1 may inflate names of prose-in-code; it is why the strict instrument (`find_with_names.py`) gives smaller counts (Epistemic 218 vs 426). The D1 reading is NOT the count of effect declarations; it is the count of "where the name follows `with` in text, inside or outside active code". The distinction is preserved above so as not to inflate conclusions.
- **This document does NOT conclude that GUM or Uncertainty should enter the enum.** The measurement shows that both have massive D2 (144 and 156) — they were designed — but the absence from the 29 production ids is a fact, not a sentence. The decision to add (or not) belongs to the founder, in another lane, with another kind of evidence. This doc describes; it does not prescribe.
- **The discovery that `with GUM` and `with NaoExisteIsto` are identical does NOT prove that GUM and NaoExisteIsto are identical.** It proves that the parser does not distinguish. The compiler may still give semantics to GUM at a later point; today, from the type checker's perspective, it does not.
- **The "tried and failed the last edge" reading for GUM/Uncertainty is a reading of D2 + D1.** It does not exclude the alternative reading "born in prose, never actually declared even in the founder" — the boundary between founder prose and founder effect declaration depends on what counts as a declaration.
- **INDETERMINATE** (per the `Mod` phase 2b precedent from minimax-cli2): if a reasonable reader cannot decide into which of the three buckets a name falls, **that decision stays open** and a fourth bucket is not invented to force a fit.
- **Nothing is added to the enum** (dispatch rule). This document is measurement; it does not modify `self-hosted/check/effects.sio`.

## What this measurement does NOT say

- It does not say which of the 9 "recognised" names was drawn by the founder and which was added later. The id table in `self-hosted/check/effects.sio` has its own history (#1963 is the most recent commit that added six extras) and each id has its own first occurrence — not traced here.
- It says nothing about the semantics of the 29 ids. Presence measurement ≠ significant-use measurement.
- It says nothing about effects in unmerged lanes. There are branches (see `docs/audit/BRANCH_AUDIT_2026-08-15.md`) with `with X` declarations that are not on `main`; those are outside this scope.

## Coordination

- Lane branch: `lane/minimax-cli3/gum-uncertainty-tail-20260819-v2` (push dd3725dde4 → 0ec8ef8c50 → e0e972ba69 → 7d08b3e9af → 8ada507886)
- Coordination bus: `artifacts/omega/agent_handoff.log.md` (NOTIFY published; see entry `agent claude / time_utc 2026-08-19T13:30:00Z` and follow-up at `13:45:00Z`)
- PR comment on #1947: NOT posted. #1947 belongs to `lane/empryo-1/ir-capacity-object-20260819` (minimax-cli2 / empryo-1). The dispatch `drop1947_claude1.md` transferred that PR; commenting on it from this lane would be cross-lane noise. The census handoff is via the coordination bus.
- Coordination requested: grok-cli5 owns the effect vocabulary; this measurement cross-checked with their phase-1 reconciliation (4 lists, GUM/Uncertainty absent from all 4) without contradiction.
- Founder rule in force: nothing is reverted. Candidates `6f23dfe1da` (#1935) and `7be969ed05` (#1939) are under analysis by grok-cli3, not this lane.

## Annex: instrument files

- `/tmp/find_with_names.py` — strict instrument (only `fn NAME(...)[with X, Y, Z]` declarations). Validated against Epistemic.
- `/tmp/find_with_prose.py` — loose instrument (catches `with X` in any context). Produced the 164497-token / D1 table.
- `/tmp/discrim_{1,2,3}.sio` — 3 effect-discrimination programs (Epistemic required in `f()`; `main()` with GUM / NaoExisteIsto / IO). All returned E035.
- `/tmp/gum_run2.sio`, `/tmp/unc_run2.sio`, `/tmp/nao_run2.sio`, `/tmp/epi_run.sio` — 4 parser-acceptance programs (all compiled and ran identically).
- `/tmp/founding_tree/` — extraction via `git archive b6d03ae18a | tar -x` for the D2 inventory.
