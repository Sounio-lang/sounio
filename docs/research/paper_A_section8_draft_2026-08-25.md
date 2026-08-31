<!-- docs:meta
topic_id: repo.docs.research.paper-a-section8-draft-2026-08-25
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-section8-draft-2026-08-25
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — §8 *Evaluation* (full draft, 2026-08-25)

> Draft prose for the evaluation. Every result marked "runs today" was reproduced on
> `2026-08-25` with `bin/souc`; outputs are pasted verbatim. Results that require the
> authorized-but-unbuilt compiler wire (§7.3) are labelled **[pending wire]** and not
> reported as measured — an honesty line the paper keeps rather than fabricating
> full-model numbers.

---

## 8. Evaluation

We ask four questions:

- **RQ1 — Is the defect real?** Does the anti-garbling class occur in shipping
  uncertainty code, and how large is the error?
- **RQ2 — Does the type rule cause the rejection?** When the discipline refuses a
  program, is the refusal attributable to noise-symbol propagation, or could it be an
  unrelated effect firing coincidentally?
- **RQ3 — How precise is the check?** What sound programs does the conservative rule
  reject, and does the escape valve recover them?
- **RQ4 — Does it matter?** In a clinical uncertainty model, does the understatement
  change a decision the system exists to make?

**What runs today vs. what is pending.** The kernel-checked soundness model
(`SounioAntiGarblingModel.lean`), the noise-symbol carrier and dataflow prototypes
(`noise_symbols.sio`, `ns_dataflow.sio`), and the acceptance-controls contract
(`ns_contract.sio`) all execute now; RQ1 and RQ2 are answered on them. The E230 rule
wired into the checker at the real `ep_add`/`ep_mul` sites (§7.3, N3) is authorized but
not yet built, so the corpus-level false-positive rate (RQ3) and the end-to-end
two-compartment clinical run (RQ4) are reported as mechanism-plus-controlled-instance,
with the full-scale numbers marked **[pending wire]**.

### 8.1 RQ1 — the defect is real, in shipping code

The anti-garbling class is not hypothetical. In the production uncertainty library
`stdlib/epistemic/knowledge.sio`, the same operation `x·x` has two implementations with
different variances — `ep_mul(&x,&x)` returns `2m²v`, `ep_square(&x)` returns the correct
`4m²v` (§2.1) — and nothing routes `x·x` to the sound one. Addition and multiplication
understate on any correlated operands; the error is exactly the dropped covariance term
`2·Cov` (§2.2, Lemma 1). For maximally correlated operands the understatement is a factor
of two in variance — a factor of `√2` in the reported standard deviation — in the
*optimistic* direction. The library's own test suite (`knowledge.sio:290–310`) exercises
these operators only on independent operands, so the defect ships untested (§2.3).

This establishes the target: a real, silent, safety-relevant unsoundness that current
testing does not surface and that no type in the library distinguishes from correct code.

### 8.2 RQ2 — the rejection is caused by noise-symbol propagation

A type rule that rejects `x+x` is only meaningful if the rejection is *because of* the
shared source, not an artifact of some other check firing. We establish causality with a
**sabotage control**: a single knob that disables noise-symbol set-propagation (measurement
nodes seed `∅` instead of a fresh symbol) while leaving every other rule intact. If the
`x+x` refusal is caused by NS, flipping the knob must make exactly that refusal vanish and
leave unrelated refusals standing.

`ns_contract.sio` encodes this as five acceptance controls. Run today, verbatim:

```
$ ./bin/souc run docs/research/sounio/ns_contract.sio
NS contract — five acceptance controls
1 x+x flagged (shared source): PASS
2 x+y accepted (disjoint cert): PASS
3 unknown conservative (flagged): PASS
4 ident(x)+x flagged (identity survives): PASS
5 sabotage: x+x NOT flagged (refusal vanishes): PASS
ALL FIVE CONTROLS PASS
```

Reading the controls against the type rules of §5:

| Control | Tests | §5 rule exercised |
|---|---|---|
| 1 `x+x` flagged | shared source ⇒ E230 | (Add-Indep) premise fails |
| 2 `x+y` accepted | disjoint supports ⇒ admitted | (Add-Indep) premise holds |
| 3 `x+⊤` flagged | unknown never disjoint | §5.1 `⊤`-conservatism |
| 4 `ident(x)+x` flagged | identity survives a copy | §5.3 (Copy) transfer |
| 5 sabotage ⇒ `x+x` clean | **refusal is caused by NS** | causality witness |

Control 5 is the load-bearing one: with set-propagation removed, `x+x` is no longer
flagged, so the refusal in control 1 is *attributable to* the propagated source-set and
not to a coincident effect. The independent dataflow prototype confirms the same
distinction on a value graph rather than on scalar handles:

```
$ ./bin/souc run docs/research/sounio/ns_dataflow.sio
NS dataflow analysis (source-set fixpoint over the value graph)
s1 = ADD(x, x): FLAGGED anti-garbling (inputs share a source)
s2 = ADD(x, y): clean (disjoint sources)
```

`s1 = x+x` (shared source) is flagged; `s2 = x+y` (disjoint) is clean — the same verdict,
reached by a monotone least-fixpoint over the graph (§5.3), which is the compile-time form
of the check.

**[pending wire]** The same sabotage protocol on the *wired* compiler — disable only the
NS rule on an otherwise-identical source build, confirm the E230 at `ep_add` vanishes while
E222 (R-ORIGIN) refusals remain — is N3 of §7.3, and is the causality claim's compiler-level
form. The prototype establishes it at the analysis level; the wired witness is future work.

### 8.3 RQ3 — precision: what the conservative rule costs

The check keys on disjoint *support*, which is sufficient but not necessary for zero
covariance (§4.4). It is therefore sound but incomplete: it rejects the
overlapping-but-orthogonal case — operands sharing a symbol whose signed coefficients
cancel, e.g. `a = x₁+x₂`, `b = x₁−x₂`, with `⟨a,b⟩ = 0`. On the designed control set there
are no such false positives (controls 1–4 are exactly the sound/unsound boundary), but a
controlled corpus is the honest measure and it needs the wire.

The escape valve (§5.5) bounds the cost: a rejected sound program is recovered either by a
proved-disjoint certificate (discharging the premise on the strength of `⟨a,b⟩=0`) or by
switching to the correlation-aware operator `add_correlated(a,b,ρ)`, which carries the
covariance explicitly and needs no disjointness premise. So the conservative rule never
*blocks* correct code; it forces correlated arithmetic to be written with the operator that
does not assume independence. The measurable quantity — the false-positive rate on real
uncertainty code, and how often the certificate vs. the correlated operator is the fix — is
**[pending wire]** (N4 regression corpus).

### 8.4 RQ4 — it changes a clinical decision

The stakes are set by a real, running model. `examples/vancomycin_auc_epistemic.sio`
(a `run-pass` example) propagates GUM uncertainty through a vancomycin AUC-guided
therapeutic-drug-monitoring chain for a discriminating patient (65 yr male, 70±1 kg,
SCr 1.40±0.14 mg/dL, 500 mg q12h):

```
CrCl (Cockcroft–Gault) = 52.1 mL/min,  u(CrCl) = 5.2
CL   (Matzke 1984)     = 2.22 L/h,      u(CL)   = 0.22
AUC₀₋₂₄ (q12h)         = 450 mg·h/L,    u(AUC)  = 44   ⇒  95% CI [362, 538]
```

The point estimate AUC = 450 reads **therapeutic**; the credible interval [362, 538]
**crosses the 400 subtherapeutic boundary**, and the epistemic model raises `WARN: possible
subtherapeutic`. The entire clinical value of propagating uncertainty is that this WARN
fires where the point estimate is silent — the decision-flip the deployed point-estimate
systems (InsightRx, DoseMeRx, JPKD) cannot produce.

**The anti-garbling threat to this WARN.** The width of the credible interval *is* the
propagated uncertainty. Any operation that understates variance shrinks the interval toward
the point estimate and can pull its lower bound back across 400 — silencing the WARN. The
bite lands wherever the model combines two quantities that share a measured source. On a
controlled instance: summing two AUC contributions that both descend from the same measured
clearance, `add(auc_a, auc_b)` with `Cov(auc_a, auc_b) > 0`, the independence-assuming
`ep_add` omits `2·Cov`; by Lemma 1 the reported variance is understated by exactly that
term, and the interval half-width contracts by `√(1 − 2Cov/Var_true)`. For strongly
correlated compartments (`ρ → 1`) this is the factor-of-`√2` SD contraction of §8.1 — enough
to move a lower bound of 362 above 400 and convert a `WARN` into a false `THERAPEUTIC`.

**Measured (2026-08-31).** The two-compartment extension now exists and the flip rate is
measured — `docs/research/sounio/rq4_vanco_two_compartment_flip.sio`, one deterministic cohort
of 5,000 patients (weight 45–120 kg, SCr 0.6–2.6 mg/dL, Q and Vp ±30 % about population,
u(weight) = 1 kg, u(SCr) = 10 %, u(Q) = u(Vp) = 20 %; 500 mg q12h; 909 true WARNs among 1,669
therapeutic-window point estimates), propagated three ways: first-order affine forms over the
measured sources (the truth, **T**), the shipped scalar `ep_*` chain (**N**), and exact operands
with an independence-assuming *final add only* (**S**, isolating Lemma 1's `2·Cov`). Two
shared-source sums a PK library actually performs:

| shared-source sum | true WARN | silenced by the naive add | Var ratio naive/true |
|---|---|---|---|
| **B** — interval sum `AUC(0–12) + AUC(12–24)`, same CL (ρ = 1) | 909 | **311 = 34.2 %** | **0.500** |
| **A** — two-compartment phase sum `A/α + B/β` | 909 | **0** (62 spurious instead) | 1.204 (final add); **300.7** (whole chain: 1,894 spurious WARNs, 38 % of the cohort) |

**B is the anti-garbling this section feared, at the size it feared:** with ρ = 1 and equal
terms Lemma 1 gives exactly half the variance (the √2 contraction of §8.1), and it silences one
true WARN in three. **A is an honest null in the feared direction** — and a finding: the phase
covariance is *negative* in 5,000/5,000 patients, because AUC is invariant to Q and Vp and the
decomposition into phases is a partition of that invariant — whatever Q and Vp move into one
phase they move out of the other. There the independence-assuming add *over*-states variance,
and across the whole chain the over-statement compounds to 300×: garbling rather than
anti-garbling, and a different clinical harm (alarm fatigue: 1,894 spurious WARNs) from the same
defect. The sign of the covariance decides which harm you get; the discipline does not need to
know the sign — E230 rejects the shared-source `add` either way, and exact propagation
(`exact_preservation`) is right in both directions. Full record and reproduce line:
`paper_A_rq4_two_compartment_flip_2026-08-31.md`.

### 8.5 Threats to validity

- **Construct.** The soundness criterion is enforced on the *variance* (second-moment)
  channel (§4.1). Non-Gaussian or heavy-tailed uncertainty is under-described by variance;
  the criterion catches variance understatement, not every distributional anti-garbling.
- **Internal.** RQ2's causality rests on a single sabotage knob at the analysis level; the
  compiler-level witness (E230 vanishes, E222 remains, same source build) is [pending wire].
- **External.** RQ1 quantifies one library; the *class* (independence assumed and unchecked)
  is general to GUM-style propagation, but we measure one instance. RQ4's magnitude is exact
  on a controlled instance, not a patient-cohort flip rate.
- **Scope of the guarantee.** Soundness holds on the linear fragment; nonlinear operators
  (`mul`, `div`, `square`, `sqrt`) retain a delta-method second-order residual even under
  disjoint support (§6.3) — the type prevents the *first-order* covariance anti-garbling,
  not the truncation error of the delta method itself.
