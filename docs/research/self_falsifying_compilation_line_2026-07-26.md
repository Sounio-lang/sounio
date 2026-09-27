<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation — opening a research line: the substrate is live, the corpus was unbound, and the failures it must catch are interpretive

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE`
**Token history:** opened as `…__CORPUS_UNBOUND__…`; flipped to `…__CORPUS_BOUND__…` when rung R1 bound real gates. See §1.1 — the flip was caught by this document's own drift guard, not by hand.
**Parents:** `self_falsifying_compiler_spec_2026-07-25.md` (the mechanism), `ast_native_claims_spec_2026-07-25.md` (claim syntax), `falsification_ledger_spec_2026-07-25.md` (claim schema)
**Harness:** `scripts/research/self_falsifying_compilation_line_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_gate.sh`

---

## 0. What this is, and what it deliberately does not claim

Three earlier rungs built a compiler that refuses to emit code whose scientific
premises no longer hold: a claim schema, a claim syntax, and `--verify-claims`,
which executes each claim's gate after type-check and **before codegen**. This
document opens a **research line** on top of that mechanism, and its first act
is an audit rather than a construction.

The audit found one encouraging thing and two uncomfortable ones.

> **The mechanism works. Nothing real is attached to it. And the failures it
> would need to catch are not the failures it can catch.**

Stated precisely, and each measured by the harness:

- **The substrate is live.** `scripts/ci/self_falsifying_compiler_gate.sh` was
  run in full (not `check`-only) against the prebuilt claim-aware Madaros:
  **7/7 clauses pass**, including `F5_FAIL_BLOCKS` and `F7_DEFAULT_LANE_BLOCKS`
  — a falsified claim really does abort compilation with no ELF emitted, on
  both lanes, and `F6_TIMEOUT` really does kill a hung gate.
- **The corpus was unbound** — *this is the audit finding, at the commit that
  opened the line (`2ba3ece5a`); rung R1 has since changed it, see §1.1.* The
  repository contained **9 native `claim` blocks across 4 files, every one of
  them a test or a CI fixture — 0 in production
  source**, against **295 CI gates** and **40 research contracts**. Counting
  generously (any `.sio` file mentioning a `scripts/ci/*.sh` path, including
  the older comment-form claims), **11 of 295 gates (3.7%)** are named by a
  claim at all. The empirical surface of this project is essentially
  disconnected from the mechanism built to guard it.

  > The gate and contract denominators are a **moving count over tracked files**
  > (they include this line's own gate and contract) — re-derive with the
  > command in §8 rather than quoting them; the figures above are as measured at
  > the commit that introduced this document. The `0 production claims` figure
  > is the load-bearing one and does not move with the denominator.
- **The historical failures are interpretive.** For three known
  self-corrections, at the commit where the claim was false, the spec's verdict
  token and the harness's emitted token **agreed**, and no CI gate script
  changed. **No claim gate would have fired in any of the three.**

**Verdict fixed before computing does not apply retroactively to §1.** The S1–S4
measurements were taken during this session and the harness reproduces them; it
does not pretend they were predicted. What *is* fixed in advance, in §5, is the
verdict type for the forward rungs — in particular the operational definition of
"would have caught it" for the retrospective study, written down here **before**
that study is run, so it cannot be graded on a curve.

---

## 1. Results

| Clause | Result | Status |
|---|---|---|
| `S1_SUBSTRATE_SURFACE` | claim executor, `--verify-claims` flag, text-preserving registry accessors, and mechanism gate all present | mechanism intact in source. |
| `S2_CORPUS_GAP` | 9 native claims / 4 files, **all tests or fixtures; 0 production**; 295 CI gates, 40 contracts | corpus `UNBOUND`. |
| `S3_BINDING_GAP` | **11/295 (3.7%)** CI gates named by any claim | the guard covers almost nothing. |
| `S4_RETROSPECTIVE` | **3/3** audited corrections were `SILENT`; token-agreement test **exact** in 3/3 | no claim gate would have fired while the claim was false. |

### 1.1 The drift guard fired on this document — for real

Rung R1 (`self_falsifying_compilation_line_r1_2026-07-26.md`) bound 15 real gates
to native claims in `examples/epistemic/rupture_claims_verified.sio`. The moment
those files entered the git index, this document's harness re-measured the tree
and its gate went red **on its own accord**:

```
SELF_FALSIFYING_COMPILATION_LINE_GATE_FAIL: verdict drift:
  spec says     'SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE'
  contract emits 'SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE'
```

Production claims went `0 → 15`; gates named by a claim, `11/295 → 24/296`
(8.1 %). This is the **drift** class of §3 — claim and check diverging over time
— and it is the class §3 says *is* addressable. Here it was caught
automatically, on a real change, with no synthetic test involved.

> **The 8.1 % is partly self-referential — read it with that discount.** Most of
> the `+13` is this line's own bookkeeping: R1's manifest binding gates, plus
> R1's own gate entering the denominator. One bound claim,
> `self_falsifying_compilation_line_audit_holds`, binds *this document's* gate to
> a claim inside the very corpus this document measures. The circularity is
> deliberate (the line should hold itself to its own discipline) but it means the
> binding-coverage figure overstates how much *pre-existing* science is guarded.
> The figure that does not move on its own artifacts is R1's module-closure
> result (`self_falsifying_compilation_line_r1_2026-07-26.md` §2): no library
> claim executes, whatever the coverage number says.

It also exposes a design question the line has to answer, and that R2 inherits:
**a spec is either a record or a live assertion, and a token-bound gate forces it
to be the latter.** This document is an audit — a statement about a moment — yet
its gate re-measures the current tree, so its token must track the present or the
gate fails. The audit finding is preserved in prose above; the token now reports
the live state. Whether that is the right convention for a corpus of 295 gates is
an open design question, not a settled one.

**A live instance, found while writing this.** The denominators above moved from
`294/39` to `295/40` the moment this rung's own gate and contract were committed,
while the verdict token stayed correct. That is precisely the **sub-token error**
of §2: the headline held, a supporting number underneath it went stale, and the
gate — which checks only the token — stayed green. The failure mode is not
hypothetical and it is not rare; it took under an hour to produce one.

Verdict: `SELF_FALSIFYING_LINE_VERDICT SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE`
(as audited: `…__CORPUS_UNBOUND__…`; see §1.1).

> **And a second live instance, caught by eye and not by the gate.** This very
> line still read `…__CORPUS_UNBOUND__…` after §1.1 had been written and the
> Status line corrected — the gate stayed green throughout, because it compares
> only the `Status:` line against the harness. A restatement of the verdict
> *inside the prose* is below the guard's resolution. Two sub-token instances in
> one document, in one session, is the honest measure of how sharp this failure
> mode is.

---

## 2. The retrospective audit, case by case

The predicate is about the **parent** commit — the state in which the claim was
false. A claim gate fires there only if the harness exited non-zero
(exit-code gating) or if the spec's declared verdict token disagreed with the
token the harness emits (token matching). If neither holds and no CI gate script
changed, the correction is **`SILENT`**: the compiler would have gone on emitting
code throughout.

| Correction | Token before → after | Depth | Would a claim gate fire? |
|---|---|---|---|
| `daa0635d0` ord-3 module overclaim deflated | `ORD3_MODULE_IS_2xV3` → `ORD3_IMAGES_FILL_CLASS_COORD_SPACE` | headline | **No** — spec and harness agreed on the wrong token |
| `ec579a24c` E6 bridge corrected | `PHI_IS_G2_SHADOW_OF_E6_CUBIC` → `PHI_IS_THE_E6_CUBIC_CROSSTERM` | headline | **No** — same |
| `eb38e9ce5` ord-3 symmetry-fill group id corrected | `NO_INVARIANT_FILL` → `NO_INVARIANT_FILL` | below-token-resolution | **No** — the wrong fact (`S4/24` vs `2^3:PSL(2,7)/192`) was never in the token |

Two distinct mechanisms, both invisible to the mechanism as built:

1. **Shared misinterpretation** (`daa0635d0`, `ec579a24c`). The computation was
   right; the *label* was wrong. Claim and check were authored together, from
   the same misunderstanding, and they agreed with each other perfectly.
2. **Sub-token error** (`eb38e9ce5`). The corrected fact was a supporting
   detail that the verdict token never encoded. The headline stayed true while
   a load-bearing detail underneath it was false.

**Sample honesty.** `n = 3`, and the three were *chosen because they were known
corrections*, not sampled. They are evidence that the failure mode exists and is
not rare in this corpus; they are **not** an estimate of its frequency. That is
what rung R4 is for.

**Measurement honesty.** "Spec and harness agreed" is computed by testing whether
the spec's declared token is among the tokens the harness *could* emit — an
over-approximation in general, since a harness with several verdict branches
would match more than one spec. The contract therefore reports the count: at all
three parent commits exactly **one** non-placeholder token was emittable (the
only other branch being `INCOMPLETE`), so here the test is **exact**, not
approximate. Were that count ever `> 1`, the finding would have to weaken to "the
spec's token was among those the harness could emit".

**Reachability.** All three audited commits are **branch-local** to the functor-F
research lane and are *not* reachable from `main`. `S4` fails rather than
silently passing when they cannot be resolved, so the clause cannot degrade to a
vacuous `PASS` in a fresh clone or after a rebase.

---

## 3. The limit this exposes

**Definition (drift).** A claim declares a proposition `p` and names a check
`g`. *Drift* is the state in which `g` still passes but `p` no longer describes
what `g` establishes, because one side changed without the other.

**Definition (shared misinterpretation).** The state in which `g` passes, `p` is
false, and `p` and `g` were authored together from the same misunderstanding:
`g` faithfully computes something, and `p` mislabels what that something means.

**Proposition (scope limit of self-falsification).** No compile-time procedure
whose only evidence about `p` is the behaviour of the claim's own checks can
detect shared misinterpretation.

*Argument.* The compiler observes `g`'s exit status and output. Under shared
misinterpretation `g` runs and reports exactly as it would if `p` were true —
by construction, since `g` was written to check `p` as its author understood it.
The mislabelling is a relation between `p` and the world, not a property of `g`'s
observable behaviour, so no predicate over that behaviour separates the two
cases. Detecting it requires a derivation of `p` **independent of the claim**,
which is by definition not part of the claim. ∎

This is a scoping argument, not a deep theorem — but it is the honest boundary,
and the corpus's own history is what forced it into view. It is the compile-time
analogue of the familiar fact that a test suite encodes its author's
misunderstanding as faithfully as their understanding.

**What remains addressable.** Drift is detectable, and cheaply: bind the
**verdict token**, not the exit code. Require the harness to emit a token and
the claim in source to declare one, and make disagreement a compile error. That
is strictly stronger than exit-code gating and it is what rung R3 builds. It
would not have caught any of the three cases in §2 — and saying so plainly is
the point of having measured them first.

**A weaker guard on the part that is out of reach.** The claim schema already
carries a `falsifier` field, today prose. Making falsifiers *independently
executable* — a check that must **fail** for the claim to live, authored against
the claim rather than with it — does not close the gap (the proposition above
stands), but it is non-vacuous: a shared misinterpretation need not survive an
independently-authored attempt to refute it. Whether that survives contact with
a real corpus is an open question, not a claim.

---

## 4. Why this is a research line and not a feature

The mechanism as built binds a build artifact to a **computation**. The
interesting question the audit surfaces is whether it can be made to bind a
build artifact to a **proposition**, and what that costs:

- **RQ1 — binding.** What does it take to attach a real scientific codebase's
  empirical surface (294 gates) to source-level claims? What breaks?
  Immediately known to break: the mechanism executes claims only in the **main
  source file** (mechanism spec §6), while any realistic claim lives in an
  imported module. Module-closure propagation is unbuilt.
- **RQ2 — reproducibility.** The mechanism deliberately breaks hermetic builds:
  same source, different day, different outcome, because the world changed. The
  reconciliation is an *empirical lockfile* — recorded witnesses (gate identity,
  result, environment digest, timestamp) with a replay mode and a staleness
  policy, so a build is reproducible **relative to a witness set**. Design
  deferred until RQ1 has produced real claims to witness; the right granularity
  and expiry cannot be guessed from the toy fixtures.
- **RQ3 — resolution.** §2 shows errors can sit *below* a verdict token's
  resolution. What claim granularity would have made `eb38e9ce5` visible, and
  what does finer granularity cost in authoring burden?
- **RQ4 — how often.** Over the correction history, at what rate would a claim
  gate have fired, under exit-code gating versus token binding?

---

## 5. Rungs, with the verdict type fixed in advance

| Rung | What it does | Verdict type fixed **now** |
|---|---|---|
| **R0** | this audit | done: `SUBSTRATE_LIVE__CORPUS_BOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` (opened as `…UNBOUND…`, see §1.1) |
| **R1** | bind a sample of real gates to native claims in real sources; bind one in an **imported** module deliberately | `BOUND_N__MODULE_CLOSURE_{BLOCKS,PASSES}`; the module-closure result is reported whichever way it goes. **Done:** `BOUND_15__MODULE_CLOSURE_BLOCKS` (`self_falsifying_compilation_line_r1_2026-07-26.md`); module-closure half **superseded 2026-08-01** by R29 → `BOUND_16__MODULE_CLOSURE_PASSES` |
| **R2** | verdict-token binding (harness emits, claim declares, mismatch is a compile error) | `TOKEN_BINDING_{IMPLEMENTED,BLOCKED}__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| **R3** | executable falsifiers (must fail for the claim to live) | `FALSIFIERS_EXECUTABLE__GUARD_{NONVACUOUS,VACUOUS}` |
| **R4** | retrospective over the correction history | see the operational definition below |
| **R5** | write-up (`docs/papers/oopsla2027/`) | n/a |

**Amendment to R1's verdict form.** R1's token was originally fixed as
`BOUND_N_OF_294__MODULE_CLOSURE_{BLOCKS,PASSES}`. That form embeds the
gate-population denominator, which moves whenever any gate is added — the token
would have drifted with the claim unchanged, which is this line's own
**sub-token** failure mode appearing inside its own verdict scheme. The form was
corrected to carry the bound count only. Recorded here rather than substituted
silently.

### R4's operational definition, fixed before the study runs

For a correction commit `c` with parent `c^`, three arms are evaluated at `c^` —
the state in which the claim was false:

- **Arm A — exit-code gating.** The harness named at `c^` exits non-zero at `c^`.
- **Arm B — token binding.** The verdict token declared in the spec at `c^`
  differs from the token the harness emits at `c^`.
- **Arm C — cross-version replay.** The **corrected** harness (taken from `c`)
  fails, or emits a token differing from `c^`'s declared one, when run against
  the state at `c^`.

**Arms A and B are known-blind by construction, and R4 is not an open question
about them.** §2 already establishes that on the three audited cases both are
silent, and the §3 proposition says why: a check authored together with its
claim reports identically whether or not the claim is true. Predicting a near-zero
rate for A and B and then "discovering" it would be theatre. R4 runs them only to
measure the rate across the full correction history rather than three hand-picked
commits — a frequency estimate, not a test of the mechanism.

**Arm C is the arm that can actually fire, and its outcome is genuinely
unknown.** It asks a different question: *was the error computationally reachable
at all?* If the corrected harness would have gone red at `c^`, the error was
latent in the computation and a finer or later-authored check could have caught
it — which is evidence that R3's resolution question is worth pursuing. If Arm C
is also silent, the errors were purely interpretive, no check could have reached
them, and the honest conclusion is that compile-time gating is the wrong
instrument for this failure class.

Classification, applied per correction:

| Bucket | Condition |
|---|---|
| `CAUGHT_A` / `CAUGHT_B` / `CAUGHT_C` | the corresponding arm fires |
| `SILENT` | no arm fires |
| `UNCLASSIFIABLE` | spec or harness absent at `c^`, no verdict token declared, or the corrected harness cannot be executed against `c^`'s state (import, path or data dependencies) — counted separately, **never redistributed** into another bucket |

`UNCLASSIFIABLE` is expected to be substantial for Arm C and that is not a
failure of the study; an honest denominator matters more than a large one.

**The result is reported whatever it is**, including all-silent. No threshold for
"success" is set, because none would be honest.

---

## 6. What this is NOT

- **Not a claim that the mechanism is useless.** It is verified working, and
  drift is a real failure mode it can catch. The audit says it is *unused* and
  *aimed at a different failure mode than the one this corpus exhibits*.
- **Not a frequency estimate.** `n = 3`, hand-picked. See §2.
- **Not a formal semantics.** §3's proposition is a scoping argument in prose.
- **Not a literature-positioned contribution yet.** See §7.
- **Not a compiler change.** This rung adds a spec, a harness and a gate; it
  touches no compiler source.

---

## 7. Prior-art positioning — **UNVERIFIED, do not cite**

No literature search was performed for this rung. The following is the
*hypothesis* to be checked in a dedicated rung before any write-up, and must not
be repeated as established:

| Neighbour | Conjectured difference |
|---|---|
| Runtime assertions / design-by-contract | about program state, not facts external to the program |
| Refinement types, static analysis | decidable properties of the source |
| `constexpr` / `comptime` / staging | compile-time *computation*, no external oracle |
| Build-system test gating | tests run beside or after the build; they do not condition codegen from within the compiler |
| Certified compilation, proof-carrying code | proofs about the program, not experiments about the world |
| Reproducible builds, hermetic build systems | the opposite design goal — this makes the build depend on the world on purpose |
| Literate / reproducible research tooling | closest neighbour; there the document is the artifact and nothing blocks code generation |

The conjectured distinguishing feature is that **the compiler refuses to emit an
executable when an empirical premise fails**, making empirical currency a
property of the build artifact. Whether that is genuinely unoccupied is exactly
what has not been checked.

---

## 8. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_contract.py
# expect: S1..S4 PASS, SELF_FALSIFYING_LINE_VERDICT
#         SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE

bash scripts/ci/self_falsifying_compilation_line_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_GATE_OK

# the underlying mechanism (needs the claim-aware Madaros; CPU-heavy):
SFC_SKIP_BUILD=1 SFC_TEST_TIMEOUT=1 bash scripts/ci/self_falsifying_compiler_gate.sh
# expect: F1..F7 PASS, SELF_FALSIFYING_COMPILER_GATE_OK
```

Re-derive the denominators rather than quoting §1 (per `CLAUDE.md` §1):

```bash
git ls-files 'scripts/ci/*.sh'       | grep -c gate       # CI gates
git ls-files 'scripts/research/*.py' | grep -c contract   # research contracts
git grep -c '^claim ' -- '*.sio'                          # native claim blocks
```

The line gate applies the drift guard of §3 to this document: it fails if the
verdict token declared in the Status line above disagrees with the token the
harness emits. **It does not check the numbers in §1** — which is exactly the
sub-token blind spot the §1 note records, left in place rather than papered over.

`S4` requires the audited commits to be reachable and **fails** if they are not;
they are branch-local to the functor-F lane and absent from `main`.

Pure Python 3 + git for the contract; bash for the gates.

---

## 9. AI disclosure

Spec, harness and gate drafted under human direction (2026-07-26). The audit
measurements are machine-reproducible via the harness. No clinical content.
GAIDeT-ICMJE 2025.
