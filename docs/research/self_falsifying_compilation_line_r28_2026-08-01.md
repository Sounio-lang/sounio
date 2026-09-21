<!-- docs:meta
topic_id: repo.docs.research.self-falsifying-compilation-line-r28-2026-08-01
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.self-falsifying-compilation-line-r28-2026-08-01
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Self-falsifying compilation R28 — the confidence gate separates almost nothing

**Date:** 2026-08-01
**Orthography:** EN-UK
**Status:** `EXECUTABLE` — `CONFIDENCE_IS_GRADED_IN_PRINCIPLE__BINARY_IN_PRACTICE`
**Parents:** `self_falsifying_compilation_line_r22_2026-07-29.md` (a field shaped like a measurement), `self_falsifying_compilation_line_r27_2026-08-01.md` (declared alive, never checked)
**Harness:** `scripts/research/self_falsifying_compilation_line_r28_contract.py`
**Receipt:** `scripts/research/r28/conf_census.json`
**Gate:** `scripts/ci/self_falsifying_compilation_line_r28_gate.sh`

---

## 1. Result

The compiler carries an epistemic confidence scalar in `0..1000`, a gate at 950,
and tiers built on it (`PLATINUM == 1000`). The obvious question is whether that
number is *calibrated*. This rung asks a prior and much cheaper one, because the
answer decides whether calibration is even a meaningful thing to attempt:

**how is the scalar distributed over real source?**

> **It is graded in principle and binary in practice. Across 30.6 million
> expression tokens the scalar takes 66 distinct values — so it is not a boolean
> by construction — and 99.9933% of the mass sits at exactly 0 or exactly 1000.
> The population strictly between 0 and the gate, the only population whose
> verdict the threshold's exact position decides, is 891 tokens: 0.003% of the
> corpus.**

Verdict: `SELF_FALSIFYING_R28_VERDICT CONFIDENCE_IS_GRADED_IN_PRINCIPLE__BINARY_IN_PRACTICE`

Move the gate anywhere in `(0, 950]` and the corpus barely notices. Before asking
whether 950 is the right threshold, one has to notice that almost nothing is ever
weighed against it.

**No counts in the headline token, and R1 is the reason.** Corpus figures move
with the corpus; they live in §3 with the date they were measured, and the
contract re-measures the properties on every run.

## 2. Why this comes before calibration, not after

The external literature makes the stakes concrete. Industrial static analysers
report false-positive rates between 76% and over 90%, and the calibration
apparatus — expected calibration error, reliability diagrams, Platt scaling — is
mature but has been developed for and applied to machine-learned predictors, not
to analyser confidence. A calibrated confidence tier would therefore be a claim
nobody in that field can currently make.

It would also be unearnable on this distribution. An ECE computed here is
dominated by two point masses; the graded tail it is meant to describe is 0.007%
of the data. The reliability diagram would have two populated bins and sixty-four
values scattered between them at a rate of one token in fifteen thousand.

So the honest ordering is: make the scalar grade before asking whether its
grading is true. That is a compiler change and it belongs to its own rung.

## 3. Verified, and how

Corpus figures measured 2026-08-01 over 1,892 files and 30,588,837 expression
tokens; recorded in `scripts/research/r28/conf_census.json` and reproducible with
the command in §6.

| clause | | |
|---|---|---|
| `B1_SUPPORT_IS_GRADED` | 66 distinct values, 64 of them strictly inside `(0, 1000)` | not a boolean by construction |
| `B2_MASS_IS_BINARY` | `conf == 0`: 7,923,090 · `conf == 1000`: 22,663,691 · everything else: **2,056** (0.0067%) | 99.9933% at the two extremes |
| `B3_GATE_SEPARATES_ALMOST_NOTHING` | strictly in `(0, 950)`: **891** (0.00291%); in `[950, 1000)`: 1,165 | the threshold is very nearly inert |
| `B4_SHARED_REDIRECT_INVENTS_VALUES` | 24 concurrent compilers into one file produce **2,441** confidences above 1000 | the control |
| `B5_LIVE_CENSUS_AGREES` | fresh 24-file census, one output each: 47,965 tokens, 100.0000% at the extremes, zero impossible values | the receipt cannot go stale silently |

**B1 and B2 must be read as one sentence.** Either alone is a misleading
headline: "graded" without the mass invites the reader to imagine a populated
spectrum, and "binary" without the support asserts something false about the
implementation. The gate refuses to let them be separated.

**B4 is why B1's number can be trusted at all.** The natural way to run this
census — many compilers in parallel, output appended to one file — tears the JSON
mid-number and yields confidences like 1002, 1013, and in one case 1048984, which
is a `bss=` figure from an interleaved diagnostic line bleeding into the token
stream. Those values cannot exist: the scalar is bounded at 1000. They land in
the graded tail, which is exactly the population B1 measures, so a census taken
that way reports a support that is partly fabricated. B5 runs the same census the
correct way and finds zero impossible values.

**Small samples do not see the tail.** B5's 24-file census found the extremes at
100.0000% — not one graded token. A 50-file sample taken while preparing this
rung found exactly two, both at 998. The tail is real but so thin that its
composition changes with the sample; any headline computed on tens of files is
unstable, which is why the receipt is a corpus-scale measurement and not a
convenient subset.

## 4. What this is NOT

- **Not a calibration.** Nothing here was calibrated, and no reliability diagram
  or ECE is computed. Doing so requires labelled ground truth — verdicts known to
  be right or wrong — which this repository does not have for this scalar. The
  rung measures a distribution, not an accuracy.
- **Not a claim that the confidence is wrong.** A predictor that answers 0 or
  1000 may be answering correctly every time. Degenerate is not the same as
  incorrect, and this rung distinguishes them rather than eliding them.
- **Not a compiler change.** No `.sio` file is touched. Making the scalar grade —
  or removing the tiers it cannot populate — is a design decision about the
  epistemic model and belongs to a rung that argues for one.
- **Not a statement about the whole corpus.** 1,892 files were measured before a
  wall-clock limit stopped the sweep, out of 2,914 enumerated. The receipt says
  which.

## 5. What would change the verdict

A single measurement would overturn this: a corpus in which the graded population
is a material fraction of tokens. The contract's B2 fails if the extremes drop
below 99.9%, and B3 fails if the sub-gate population rises above one token in ten
thousand. Both are live thresholds, not prose.

## 6. Reproduce

```bash
# the corpus census, ~10 min, ONE OUTPUT FILE PER INPUT -- never a shared redirect
mkdir -p /tmp/r28 && { ls tests/run-pass/*.sio; ls tests/compile-fail/*.sio; } \
  | xargs -P8 -I{} sh -c 'o=/tmp/r28/$(echo {} | tr "/" "_").out;
      ./bin/souc-lean-single-x86_64 "{}" /tmp/e_$$.elf --dump-conf-json > "$o" 2>&1; rm -f /tmp/e_$$.elf'
cat /tmp/r28/*.out | grep -o '"conf":[0-9]*' | sed 's/.*://' | sort -n | uniq -c

# the rung itself, ~20 s, no build lock
python3 scripts/research/self_falsifying_compilation_line_r28_contract.py
bash scripts/ci/self_falsifying_compilation_line_r28_gate.sh
```

Needs `bin/souc-lean-single-x86_64`; the gate refuses rather than passing if it is
absent. Takes no global build lock and leaves the working tree unchanged.

## 7. AI disclosure

Finding, contract, gate, receipt and spec drafted under human direction
(2026-08-01). The thread was selected by a multi-agent planning workflow; a
subagent first reported a 50-file census with a different graded tail, and every
figure here was re-measured at corpus scale by hand before being written, which
is how the sample-dependence in §3 was found. The shared-redirect fabrication was
reported by a refuting agent and independently reproduced here at a larger
magnitude than reported. No clinical content. GAIDeT-ICMJE 2025.
