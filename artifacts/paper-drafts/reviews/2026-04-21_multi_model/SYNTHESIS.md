# Multi-Model Peer Review Synthesis

**Manuscript:** `artifacts/paper-drafts/hessian_biomarker_preprint.tex` (commit `f6dbb58c`)
**Date:** 2026-04-21
**Reviewers:** 4 independent LLM reviewers, same prompt, zero cross-talk
**Prompt:** `prompt.md` in this directory

| # | Model | Provider | Role framing |
|---|---|---|---|
| R1 | grok-4-1-fast-reasoning | xAI | reasoning, blunt |
| R2 | grok-4-0709 | xAI | reasoning, legacy |
| R3 | deepseek-reasoner | DeepSeek | reasoning, chain-of-thought |
| R4 | deepseek-chat | DeepSeek | generalist |

Two other providers (Groq / MiniMax) were attempted and failed at the API
gate (invalid key / plan not permitting the requested model). Output is
intentionally preserved in raw form; see the `review_*.md` files.

## 1. Consensus verdict

All four reviewers recommend **reject in its current form**. None of the
four is willing to call the preprint ready for Q1 submission.

## 2. Unanimous concerns (4 / 4 reviewers agree)

### 2.1 The central claim is overstated relative to the effect size

Every reviewer flags the title/abstract phrasing "spikes at ictal onset"
as stronger than the data support.

- Median increase PRE5 → IC is only **+22.7 %**.
- LOO AUC = 0.642 with 95 % CI **[0.513, 0.764]**. The lower bound is
  barely above chance; R3 calls the interval "implausibly narrow" for
  N=24 and asks whether the bootstrap honoured the leave-*patient*-out
  dependence structure.
- R1 and R2 both note that the per-patient appendix table shows
  **9 / 24 patients with negative spike**, directly contradicting the
  "sign-consistent" language in the abstract.

### 2.2 Treatment of the three null models is selective reporting

All four reviewers flag the same statistical flaw. The protocol
pre-registers the iid null as primary and the other two as sensitivity
checks. In the Results and Abstract, however, the paper presents
(iid 1.23e-2, circular 8.99e-5, block 3.07e-2) as three convergent
pieces of evidence. That is *p-hacking by alternative-null*:

- if iid had been non-significant, the circular result would have been
  promoted;
- the three nulls are three *assumptions* about the same test, not
  three independent tests.

Every reviewer independently concluded this must be fixed before the
paper can be submitted anywhere.

### 2.3 The sedenion algebra is ornamental, not load-bearing

Four independent critiques converge on the same point:

- The associator norm, for the chosen channel → basis embedding, is a
  specific cubic polynomial on ℝ¹⁶.
- Any permutation of the channel → basis map gives a different norm;
  the mapping has no neurophysiological justification.
- A simpler trilinear form (whitened triple product, bispectrum norm,
  quaternion tensor associator on 4×4 channel groups) would almost
  certainly capture the same signal.
- Non-associativity / zero-divisors are never shown to be necessary —
  no ablation against octonions, against a random trilinear form, or
  against a basis-permutation control.

The paper calls the associator "an empirical, data-level index of how
far the recent recurrence structure departs from an octonion subalgebra"
— R4 labels this a "just-so story".

### 2.4 N=24 CHB-MIT is underpowered and non-representative

All four reviewers reject the generic-mechanism framing:

- single-centre (Boston Children's)
- pediatric, intractable focal epilepsy
- scalp EEG (poor spatial resolution)
- 16-channel canonical montage, enriched dataset
- only first annotated seizure per patient
- ±90 s windows around a known onset, i.e. detection, not prediction

R4: *"proof-of-concept at best; not a biomarker validation study."*

### 2.5 Pilot-Hessian leakage into pre-registration

Every reviewer flags the same pre-registration defect. The Hessian
pilot (n ≤ 7) was performed first, found unstable, and the sedenion
associator was chosen *in response to* that failure. The formal
pre-registration was therefore written after a data-dependent
statistic choice, even if the N=24 inferential analysis itself
followed the PROTOCOL. R3: "the pre-registration of the final
protocol does not erase this iterative, data-influenced development."

## 3. Non-unanimous but specific-and-actionable findings

| Finding | Raised by | Impact |
|---|---|---|
| Per-window standardisation uses the *interictal II* window (t* − 300 s), which may itself carry peri-ictal drift and bias the IC value upward | R1 | high |
| T3 "100 / 100 sign preservation" is **too clean** — suggests a systemic cohort-level bias (e.g. the global standardisation), not a channel-subset robustness | R3, R4 | high |
| T3 always draws 16 of 23 channels, so the sedenion dimension is held constant; T3 therefore cannot test the channel → basis *mapping*, only the specific channel identity | R4 | medium |
| T4's inclusion in the BH-FDR family dilutes focus because T4 is classifier performance, not a direct test of the spike hypothesis | R3 | medium |
| T1 refutation (p = 0.599) effectively rescinds half the pre-registered narrative; the paper quietly narrows to T2 alone without penalty | R2, R3 | medium |

## 4. Passages flagged as desk-rejectable in their current wording

Every reviewer cited essentially the same four passages:

1. **Title** — "Pre-Registered Evidence that the Sedenion Associator Norm
   **Spikes** at Ictal Onset in Scalp EEG" (all 4).
2. **Abstract, final sentence** — "...a pre-registered, sign-consistent,
   and **spatially robust** finding; it is **not an artefact**" (all 4).
3. **Results §3.2** — "T2 ictal spike is robust **across three null
   models**" (all 4), treating three nulls as convergent evidence.
4. **Discussion opening** — "The sedenion associator norm spikes at
   ictal onset in the CHB-MIT cohort, pre-registered, 100k-permutation,
   BH-FDR-controlled" (all 4) — uses procedural rigor words to dress
   up a weak effect.
5. **Background / Sedenion lifting** — the arbitrary channel → basis
   mapping is presented as principled without justification (R3, R4).

## 5. Minimum revision package before resubmission

Derived from the union of the four reviewers' recommendations. Ordered
by marginal credibility gained per unit effort.

1. **Rewrite title / abstract / discussion opening** to drop the word
   "spike" and replace it with "small but statistically significant
   cohort-level increase". Drop "not an artefact".
2. **Pick one null model** (iid, as pre-registered), make *it* primary,
   move the other two to a short sensitivity paragraph, and explicitly
   label the circular-shift p as exploratory — not to be cited in the
   abstract. This is the single highest-leverage change.
3. **Add three control biomarkers** run through the exact same
   pipeline: (a) bispectrum norm, (b) random-basis trilinear form, and
   (c) sedenion associator with a *randomly permuted* channel → basis
   mapping. Report p-values side by side. If they match T2, the
   sedenion claim must be retracted.
4. **Re-derive the LOO AUC CI** using a patient-level cluster bootstrap
   and report the full bootstrap distribution. If the honest lower
   bound crosses 0.5, say so.
5. **Add an explicit "Scope" paragraph** near the top: pediatric,
   intractable, single centre, first seizure, detection (not
   prediction), 16-canonical-channel scalp. No extrapolation beyond
   these.
6. **Document the Hessian-pilot → associator decision** in a
   "Deviations from ideal pre-registration" paragraph, rather than
   hiding it in the Introduction.
7. **Re-label T3** as "robustness to channel-identity drop-out" rather
   than "spatial robustness" (it does not test the channel → basis
   mapping).
8. **Drop the word "evidence" from the title**; use "association" or
   "pre-registered observation".

## 6. One-sentence consensus

> A real, small, pre-registered cohort-level effect, wrapped in
> language that is one or two orders of magnitude stronger than the
> underlying data justifies, with a biomarker whose algebraic
> sophistication is doing no causal work.

## 7. Provenance

- Prompt: `prompt.md` (683 lines, includes the full preprint TeX)
- Individual reviews: `review_*.md` (4 files, verbatim model output)
- All four calls were issued in parallel from
  `/workspace/sounio` on 2026-04-21 starting 16:22 UTC.
- Temperature 0.3, no system prompt, no chain-of-reviewer prompting.
- Models saw *only* the prompt; they did not see each other's output.
