<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r4-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r4-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R4 — the retrospective: the corpus's own history cannot be graded, and where it can, nothing fires

**Date:** 2026-07-26
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `RETROSPECTIVE_RUN__SOME_ARM_FIRED`
**Parents:** `self_falsifying_compilation_line_2026-07-26.md` (R0 §5 fixed this predicate before the study ran), `self_falsifying_compilation_line_r2_2026-07-26.md` (arm B is R2's mechanism), `self_falsifying_compilation_line_r3_2026-07-26.md` (the falsifier result)
**Harness:** `scripts/research/self_falsifying_compilation_line_r4_contract.py`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r4_gate.sh`

---

## 0. Result, including the part that contradicts the verdict token

R0 §5 fixed the predicate — three arms evaluated at the parent commit `c^`,
five buckets, `UNCLASSIFIABLE` never redistributed — and stated in advance that
arms A and B were known-blind, with arm C the open question. The study ran as
specified.

> **Arms A and B fired zero times — and for the two objective corrections this
> is now EXECUTED, not assumed. Arm C fired twice, and both firings are
> tautological. Of the pairs where a prior claim even existed, 70 % could not be
> graded at all, because the spec declared no verdict token.**

| Bucket | Count |
|---|---:|
| `CAUGHT_A` (exit-code gating) | **0** |
| `CAUGHT_B` (token binding) | **0** |
| `CAUGHT_C` (cross-version replay) | **2** |
| `SILENT` | 4 |
| `UNCLASSIFIABLE` — spec declared no verdict token at `c^` | 14 |
| `NO_PRIOR_CLAIM` — the commit *created* the spec, so there was nothing to catch | 45 |

`65` (commit, spec) pairs from `51` message-flagged commits. `20` had a prior
claim; `6` of those were classifiable.

**Arm A is executed where the harness can be run standalone.** These contracts
are pure numpy computations with no repository dependencies — the same property
that makes arm C degenerate makes the historical version simply runnable. For
both objective corrections, the harness **as it stood at `c^` exits 0 and emits
exactly the token the spec declared**:

```
daa0635d0^  exit 0, emitted ORD3_MODULE_IS_2xV3          (= declared)
ec579a24c^  exit 0, emitted PHI_IS_G2_SHADOW_OF_E6_CUBIC (= declared)
```

So the check ran green and agreed with its claim *while the claim was false*.
That is R0 §3's shared misinterpretation, demonstrated by execution rather than
by static comparison. Where a harness cannot be run standalone the result is
recorded as not-executed with the reason, never as a pass.

**The verdict token says `SOME_ARM_FIRED`, and it is not being changed.** The
criteria were fixed in the harness before running, and the mechanical outcome is
that an arm fired. Retro-fitting the token to the narrative below is exactly the
failure this line studies, so the token records the mechanics and this section
records what they mean. That the two disagree is the honest state.

---

## 1. Why arm C's two firings establish nothing

Arm C asks whether the **corrected** harness, run against `c^`, disagrees with
the claim declared at `c^`. Both firings:

```
daa0635d0  corrected harness emits [ORD3_IMAGES_FILL_CLASS_COORD_SPACE]
           c^ declared ORD3_MODULE_IS_2xV3
ec579a24c  corrected harness emits [PHI_IS_THE_E6_CUBIC_CROSSTERM]
           c^ declared PHI_IS_G2_SHADOW_OF_E6_CUBIC
```

These harnesses are **pure computations** — they read no repository state, so
"run against `c^`" is the same as running them now, and they emit the corrected
token because the correction is what they encode. For a correction that changes
the token, arm C therefore fires *by construction*. It restates the definition
of a correction; it does not show the error was detectable at the time.

R0 §5 called arm C's outcome "genuinely unknown". It was — and the answer is
that the arm is degenerate for this corpus. That is a finding about the
predicate, not about the mechanism.

Arm C would be informative for corrections whose token did **not** change (the
sub-token class, e.g. `eb38e9ce5`). None of those reached a classifiable state
here.

---

## 2. Why most of the history cannot be graded

Of the `20` pairs where a prior claim existed, `14` (**70 %**) are
`UNCLASSIFIABLE` for a single reason: **the spec declared no verdict token at
`c^`**. There was no machine-readable claim to compare anything against.

This is the same wall R1 measured from the other side — only `25/270` specs
carry a parseable token, and the convention is days old. The corpus's history
predates the structure the predicate needs.

The `45` `NO_PRIOR_CLAIM` pairs are a selection artefact, not missing data: they
are commits whose message mentions a correction while *creating* a spec — the
correction being described happened elsewhere. They are bucketed separately so
they cannot inflate the unclassifiable count, and they are not evidence of
anything.

---

## 3. What the whole line now says about its own premise

Putting R0–R4 together, on this corpus:

| Question | Answer |
|---|---|
| Does the mechanism work? | Yes — verified end to end (R0 `F1–F7`, R2 `D1–D4`). |
| Can it be attached to real science? | Yes, 15 gates bound; but not inside libraries — imported claims never execute (R1). |
| Would exit-code gating have caught the corpus's errors? | **No.** 0 of 6 classifiable. |
| Would token binding have caught them? | **No.** 0 of 6. Claim and check agreed while the claim was false. |
| Can the history even be graded? | **Mostly not** — 70 % of eligible pairs lack a token. |
| Is anything left that reaches the real failure mode? | Executable falsifiers, for the minority of claims that reduce to a closed form — and they are not self-starting (R3). |

The line's premise — *a compiler that refuses to emit code whose scientific
premises no longer hold* — is **buildable and was built**. What it guards
against is drift between a claim and its check. What actually damaged this
corpus was claim and check being wrong **together**, and no amount of compiler
machinery reaches that; R0 §3 says why, and R4 is the empirical half of the
same statement.

That is a negative result about the idea's usefulness *here*, obtained without
weakening a single definition along the way. It is the result the line was set
up to be able to reach.

---

## 4. Threats to this reading

- **`n` is small.** 6 classifiable pairs. The zero for arms A and B is not a
  rate estimate; it is "every case we could grade came out silent".
- **P2 is message-matched**, so its recall is unknown: a correction whose commit
  message used none of the flagged words is invisible here. P1 — the objective,
  token-change population — is `2`, which bounds how much the objective route
  can ever see in this history.
- **Arm A is executed only where the harness is self-contained.** For the two
  objective corrections it ran and exited 0 (§0), so their zero is measured. For
  pairs whose harness cannot be run standalone the result is recorded as
  not-executed with the reason; those contribute no evidence either way rather
  than a silent pass.
- **This line's own specs are excluded** from both populations. R0's token moved
  `UNBOUND → BOUND` because R1 did the binding — a state change, not a
  correction — and its commit message mentions corrections made elsewhere.
  Including it would have manufactured a third "correction". Same
  self-reference discount R1 applied to its coverage figure.

---

## 5. What this is NOT

- **Not a demonstration that the mechanism is broken.** It works; it is aimed at
  a failure mode this corpus does not exhibit.
- **Not a general claim about scientific software.** One repository, one team,
  one author.
- **Not a reason to remove the guard.** Drift is real and cheap to prevent; R2
  costs nothing at build time when no `verdict_token` is declared.

---

## 6. Reproduce

```bash
python3 scripts/research/self_falsifying_compilation_line_r4_contract.py
# expect: CAUGHT_A 0, CAUGHT_B 0, CAUGHT_C 2, SILENT 4,
#         UNCLASSIFIABLE 14, NO_PRIOR_CLAIM 45
#         SELF_FALSIFYING_R4_VERDICT RETROSPECTIVE_RUN__SOME_ARM_FIRED

bash scripts/ci/self_falsifying_compilation_line_r4_gate.sh
# expect: SELF_FALSIFYING_COMPILATION_LINE_R4_GATE_OK
```

The scan walks the whole history and takes a few minutes. Counts move as the
history grows — re-run rather than quoting §0.

---

## 7. AI disclosure

Harness, gate and spec drafted under human direction (2026-07-26). All counts
are machine-produced from git history and re-runnable. Arm A is recorded as
not-executed with its reason. No clinical content. GAIDeT-ICMJE 2025.
