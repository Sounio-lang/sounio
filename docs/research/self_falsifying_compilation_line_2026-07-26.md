# Self-falsifying compilation — opening a research line: the substrate is live, the corpus is unbound, and the failures it must catch are interpretive

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE`
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
- **The corpus is unbound.** The repository contains **9 native `claim` blocks
  across 4 files, every one of them a test or a CI fixture — 0 in production
  source**, against **294 CI gates** and **39 research contracts**. Counting
  generously (any `.sio` file mentioning a `scripts/ci/*.sh` path, including
  the older comment-form claims), **11 of 294 gates (3.7%)** are named by a
  claim at all. The empirical surface of this project is essentially
  disconnected from the mechanism built to guard it.
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
| `S2_CORPUS_GAP` | 9 native claims / 4 files, **all tests or fixtures; 0 production**; 294 CI gates, 39 contracts | corpus `UNBOUND`. |
| `S3_BINDING_GAP` | **11/294 (3.7%)** CI gates named by any claim | the guard covers almost nothing. |
| `S4_RETROSPECTIVE` | **3/3** audited corrections were `SILENT` | no claim gate would have fired while the claim was false. |

Verdict: `SELF_FALSIFYING_LINE_VERDICT SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE`.

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
| **R0** | this audit | done: `SUBSTRATE_LIVE__CORPUS_UNBOUND__HISTORICAL_FAILURES_ARE_INTERPRETIVE` |
| **R1** | bind a sample of real gates to native claims in real sources; bind one in an **imported** module deliberately | `BOUND_N_OF_294__MODULE_CLOSURE_{BLOCKS,PASSES}` — and the module-closure result is reported whichever way it goes |
| **R2** | verdict-token binding (harness emits, claim declares, mismatch is a compile error) | `TOKEN_BINDING_{IMPLEMENTED,BLOCKED}__CATCHES_DRIFT_NOT_MISINTERPRETATION` |
| **R3** | executable falsifiers (must fail for the claim to live) | `FALSIFIERS_EXECUTABLE__GUARD_{NONVACUOUS,VACUOUS}` |
| **R4** | retrospective over the correction history | see the operational definition below |
| **R5** | write-up (`docs/papers/oopsla2027/`) | n/a |

### R4's operational definition, fixed before the study runs

For a correction commit `c` with parent `c^`, a claim gate **would have caught**
the error iff, at `c^`, with the claim bound as it is at `c`:

- **under exit-code gating** — the named harness exits non-zero at `c^`; or
- **under token binding** — the verdict token declared in the spec at `c^`
  differs from the token the harness emits at `c^`.

A correction is **`SILENT`** iff neither holds. Corrections whose spec or
harness does not exist at `c^`, or which do not name a verdict token, are
reported as **`UNCLASSIFIABLE`** and counted separately — they are not
redistributed into either bucket.

**The result is reported whatever it is.** If the catch rate is 0, that is the
finding, and it is the more interesting one: it would mean compile-time claim
gating does not address the failure mode that actually damages this corpus, and
the line's value lies in R2/R3 plus a precisely-drawn negative boundary. No
threshold for "success" is set, because none would be honest.

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

The line gate applies the drift guard of §3 to this document: it fails if the
verdict token declared in the Status line above disagrees with the token the
harness emits.

Pure Python 3 + git for the contract; bash for the gates.

---

## 9. AI disclosure

Spec, harness and gate drafted under human direction (2026-07-26). The audit
measurements are machine-reproducible via the harness. No clinical content.
GAIDeT-ICMJE 2025.
