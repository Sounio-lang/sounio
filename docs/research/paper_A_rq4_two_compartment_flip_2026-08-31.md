<!-- docs:meta
topic_id: repo.docs.research.paper-a-rq4-two-compartment-flip-2026-08-31
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.paper-a-rq4-two-compartment-flip-2026-08-31
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Paper A — RQ4 closed: the two-compartment vancomycin flip rate, measured (2026-08-31)

**Closes** the last `[pending]` of §8.4: "embedding the shared-source sum in the full
two-compartment vancomycin model — and measuring how many patients flip". Artifact:
`docs/research/sounio/rq4_vanco_two_compartment_flip.sio` (Sounio, runs under the committed
Madaros v0.80.0 on `main`/lane; deterministic; identity self-check at error 0 × 10⁻⁹).

```
bin/souc run docs/research/sounio/rq4_vanco_two_compartment_flip.sio
RQ4_FLIP n=5000 true_warn=909 silenced_sum=0 silenced_naive=0 spurious_naive=1894
         var_ratio_sum_permille=1204 var_ratio_naive_permille=300662
         B_true_warn=909 B_silenced=311 B_var_ratio_permille=500
```

## What was measured

One deterministic cohort (LCG seed 20260831, N = 5000; weight 45–120 kg, SCr 0.6–2.6 mg/dL,
Q and Vp ±30 % around population; age 65 male; 500 mg q12h). Inputs carry u(weight) = 1 kg,
u(SCr) = 10 %, u(Q) = u(Vp) = 20 %. The decision rule is the example's: **WARN** iff the
point AUC₀₋₂₄ ≥ 400 and the lower 95 % bound (point − 2·u) < 400. 1669 of the 5000 point
estimates fall in the therapeutic window [400, 600]; 909 of them are true WARNs.

Three propagations of the same chain:

| | propagation | what it is |
|---|---|---|
| **T** | first-order affine forms over the four measured sources, everywhere | the truth to first order — the same object as `Aff` in `EpistemicEffectsNS.lean` |
| **N** | the shipped scalar `ep_*` chain (`stdlib/epistemic/knowledge.sio`), everywhere | the library as it is: independence assumed at every operation |
| **S** | exact (affine) operands, independence-assuming **final add only** | the controlled instance of §8.4: isolates Lemma 1's `2·Cov` at one `add` |

Two shared-source sums a PK library actually performs:

**Scenario A — the two-compartment phase decomposition.** AUC = A/α + B/β with A, α, B, β all
descending from CL (weight, SCr), Vc (weight), Q, Vp. Algebraically the sum is 2·D/CL; the
affine propagation reproduces that identity to first order at error 0 (value and variance).

**Scenario B — the interval sum.** AUC₀₋₂₄ = AUC(0–12) + AUC(12–24), each interval D/CL from
the *same* clearance estimate (ρ = 1).

## Results

| | true WARN (T) | WARN under S | **silenced** by S | WARN under N | silenced by N | spurious under N | Var ratio S/T | Var ratio N/T |
|---|---|---|---|---|---|---|---|---|
| A — phase sum | 909 | 971 | **0 / 909** | 2803 | 0 | **1894** (37.9 % of cohort) | 1.204 | **300.7** |
| B — interval sum | 909 | 598 | **311 / 909 = 34.2 %** | — | — | — | **0.500** | — |

**Scenario B is the anti-garbling the paper feared, at the size it feared.** With ρ = 1 and
equal terms, Lemma 1 gives Var_true = 4v and Var_naive = 2v: the ratio is exactly 0.500 (the
√2 SD contraction of §8.1), and it silences **one true WARN in three** — 311 patients whose
lower bound the naive sum pulls back above 400.

**Scenario A is an honest null in the feared direction — and a finding of its own.** The phase
covariance is **negative in 5000/5000 patients**, so the independence-assuming add
*over*-states variance (1.204×) and silences nothing; it manufactures 62 spurious WARNs instead.
The reason is structural, not numerical: AUC is invariant to Q and Vp, so the decomposition into
phases is a *partition* of a Q-, Vp-invariant quantity — whatever Q and Vp move into one phase
they move out of the other, and the shared-source terms enter the covariance with opposite
signs. Across the whole shipped chain (N) the effect compounds to a **300× variance
over-statement**: 2803 WARNs where 909 are warranted — 1894 spurious alarms, 38 % of the
cohort. That is garbling (information *lost*), not anti-garbling; it is the other clinical harm
— alarm fatigue — and it is produced by the same defect (independence assumed where sources
are shared), with the sign of the covariance deciding which harm you get.

## What this changes in the paper

1. §8.4's "measure how many patients flip" has a number: **34.2 %** of true WARNs silenced in
   the ρ = 1 interval sum (B); **0 %** in the phase decomposition (A), where the covariance is
   negative.
2. §2/§4's directional signature (add understates, sub stays conservative) acquires its
   covariance-sign twin: an independence-assuming `add` **understates when Cov > 0 and
   over-states when Cov < 0**; both are first-order-exact under NS (`exact_preservation`),
   and E230 rejects the shared-source `add` in both cases — the type system does not know the
   sign, and does not need to: the fix is exact propagation, not a one-sided correction.
3. A limitation for §10: the sign of the harm is model-structural. The paper's "≈√2 contraction
   for ρ → 1" is right for sums of like-signed shared-source terms (B) and wrong for partitions
   (A). The discipline is indifferent to this; the *prose* about "silencing" must be scoped to
   Cov > 0.

## Engine note (honest)

Runs under Madaros (the claim clock) — **re-verified on the committed compiler `bf1fe608` with a
byte-identical `RQ4_FLIP` line.** Under `lean_single` the program does not build (exit 1, no
diagnostic) — as the `a_mu_*` run-pass tests record, stage-2 does not execute imported
`Epistemic` graphs.

**Correction (same day).** During development the program crashed and printed wrong values, which
was attributed to a per-function scalar-slot overflow in the committed Madaros (audit note, issue
#2318). That was wrong: `bin/souc` in the lane worktree had resolved a gitignored, stale
2026-08-16 lane build (`md5 709acf97`) ahead of the committed ELF — while its provenance line
still said "COMMITTED". With the artifact removed, the 30-local probe, 100/400-local probes, the
inline shape and a 183-struct borrowed chain all pass on the committed, source-built and
lean_single engines (worktree agent + independent re-run). #2318 is closed as not reproducible;
the audit doc keeps the false alarm with the correction appended; the probes are kept as
positive-control run-pass fixtures. The stage-function structure of the RQ4 chain is kept for
readability only.

## Reproduce

```bash
bin/souc run docs/research/sounio/rq4_vanco_two_compartment_flip.sio   # exit 0; last line RQ4_FLIP …
```
Determinism: two consecutive runs on 2026-08-31 produced byte-identical `RQ4_FLIP` lines.

**Partition lemma (same day):** the negative-covariance mechanism is proved in `EpistemicEffectsNS.lean` — `partition_iff`, `inner_nonpos_of_partition`, `naive_add_conservative_of_partition`, `inner_eq_neg_var_of_full_partition` (gate: 16 theorems).

**Adequacy (same day):** a Monte Carlo of the same cohort (`rq4_vanco_mc_adequacy.sio`, 1 000
draws per patient) gives Var_MC/Var_T = 0.999 (0.857–1.158), the same WARN decision on 99.4 % of
patients, and Var_N/Var_MC = 300.9 — see `paper_A_rq4_mc_adequacy_2026-08-31.md`.
