<!-- docs:meta
topic_id: repo.docs.research.neurodyn-ossm-sota-deep-research-2026-07-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.neurodyn-ossm-sota-deep-research-2026-07-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# NeuroDyn O-SSM SOTA Deep Research Note

Date: 2026-07-05

Claim boundary: this note is a research-planning artifact. It does not make a
clinical diagnostic, mechanistic, treatment-response, or O-SSM superiority
claim. Positive claims remain gated by held-out, site-aware, checkpointed runs.

## Core Position

The useful scientific target is not "an octonion classifier for MDD/ADHD".
That framing is too generic and too easy to falsify with stronger baselines.

The useful target is:

> An octonionic state-space model for testing whether non-commutative and
> non-associative temporal composition in ROI time series carries information
> that is missed or flattened by static FC, sliding-window dFC, real-valued
> SSMs, and graph/attention baselines.

This keeps the novelty where it belongs: order, hidden state, associator, and
temporal composition. A paper is a consequence if this survives controls.

## Why Octonions Are Non-Trivial Here

Octonion multiplication is non-commutative and non-associative. In a sequence
model, that means a state update can, in principle, distinguish:

- `(h * x_t) * x_{t+1}`
- `h * (x_t * x_{t+1})`
- `(h * x_{t+1}) * x_t`

even when the same multiset of ROI events, same endpoints, same mean, same
delta, and same energy are presented. That is not a normal "more parameters"
argument. It is an algebraic order/composition argument.

The local deterministic witness supports the algebraic possibility:

- `unit_real + identity + brain`: final state distance `12.986738`
- `unit_real + mandel_hybrid + brain`: final state distance `14.947292`
- `unit_real + identity + stable_sigmoid`: final state distance near zero

Reconstruction anchors:

- witness script:
  `scripts/research/neurodyn_octonionic_order_witness.py`
- manifest generator:
  `scripts/research/neurodyn_noncommutative_temporal_manifest.py`
- hidden-state probe:
  `scripts/research/neurodyn_hidden_state_separability.py`
- model surface:
  `examples/brain_ossm_abide.sio`
- deterministic witness artifact:
`/mnt/cephfs/neurodyn/derivatives/synthetic/octonionic_order_witness_20260705T237500Z`

But the trainable Brain O-SSM path has not yet preserved that separation:

- unit-anchor fixed-pair, delta/flat disabled:
  - subjects `112`
  - seeds `20`
  - O-SSM classifier BA `50.580357`
  - O-SSM hidden nearest-centroid BA `48.080357`
- frozen-state unit-anchor control, delta/flat disabled:
  - subjects `112`
  - seeds `20`
  - O-SSM classifier BA `49.508929`
  - O-SSM hidden nearest-centroid BA `50.044643`

Therefore the honest current claim is:

> The octonionic update path can represent non-trivial temporal order, but the
> current trainable objective/readout does not yet extract that order into a
> useful hidden-state representation.

## SOTA Map

| Area | Representative SOTA | What it already does | Associator/order statistic evaluated? | Gap relevant to O-SSM |
|---|---|---|---|---|
| Deep octonion networks | Deep Octonion Networks, Wu et al. | Defines octonion convolution, batch norm, initialization; demonstrates image classification gains vs real/complex/quaternion networks. | Not found in the source summary as a primary endpoint. | Mostly vision/CNN. Not a neurodynamic state-space model; limited direct test of associator/order as the scientific object. |
| Hypercomplex neural networks | Recent HNN reviews | Establish hypercomplex networks as an active family and identify octonions as difficult because of non-associativity and training complexity. | Review-level discussion, not a benchmark endpoint. | Reviews emphasize the challenge, but applications to ROI-level fMRI dynamics and associator biomarkers remain sparse. |
| fMRI state-space models | fMRI-S4 | Uses S4/state-space machinery plus 1D convs to capture short/long dependencies in rs-fMRI and validates on MDD/ASD/sex. | Not evaluated in the reviewed abstract; needs direct baseline run. | Strong baseline. It models temporal dependence, but a controlled associator/order assay remains to be measured. |
| Dynamic FC in MDD | Whole-brain dynamic/static FC + SVM | Shows dynamic FC can outperform static FC for MDD classification in some cohorts. | No explicit non-associative order statistic. | dFC usually compresses dynamics into windowed covariance/statistics; order-sensitive event composition is not the primitive. |
| Dynamic/graph ADHD models | DeepFMRI, ASTNet, Skip-Vote-Net, dynamic GCN/GAT variants | Learn FC/dFC, adaptive spatial-temporal graphs, or segment-wise dynamics for ADHD. | Not evaluated in the searched summaries; needs controlled reimplementation. | Natural competitor space. Comparative claims require running these baselines on the same order assays. |
| Multimodal MDD prediction | fMRI/EEG GNN fusion for treatment response | Uses graph fusion and multimodal latent alignment for antidepressant/placebo response. | No O-SSM associator statistic. | Important long-term direction, but too many variables for the first O-SSM proof. O-SSM should first pass a unimodal temporal assay. |
| Current fMRI trends | Intrinsic modes, dynamic FC, DCM/effective connectivity | Field is moving beyond static correlations toward temporal modes and effective connectivity. | Different formal object. | O-SSM can be positioned as an algebraic probe only after direct baselines are run. |

## Source Notes

- Deep Octonion Networks, arXiv:1903.08478:
  https://arxiv.org/abs/1903.08478
  The paper constructs DON building blocks such as octonion convolution, batch
  normalization, and initialization, then evaluates on CIFAR-10/CIFAR-100.

- fMRI-S4, arXiv:2208.04166:
  https://arxiv.org/abs/2208.04166
  The paper targets single-subject phenotype/psychiatric classification from
  rs-fMRI timecourses and presents S4 as a plug-and-play temporal baseline for
  MDD, ASD, and sex classification.

- Dynamic/static FC for MDD, Frontiers in Psychiatry 2022:
  https://www.frontiersin.org/journals/psychiatry/articles/10.3389/fpsyt.2022.973921/full
  The study compares whole-brain dynamic and static FC with linear SVM for MDD
  and reports stronger/stabler dynamic FC performance in that cohort.

- ADHD rs-fMRI ALE meta-analysis, European Child & Adolescent Psychiatry:
  https://link.springer.com/article/10.1007/s00787-025-02906-3
  The review emphasizes ADHD heterogeneity, developmental specificity, and
  resting-state alterations across frontal, sensorimotor, and cerebellar areas.

- ASTNet for ADHD:
  https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2024.1394234/full
  The model uses adaptive dFC generation and temporal dependency mining from
  rs-fMRI windows. It is a direct ADHD dynamic baseline family.

- Skip-Vote-Net ADHD:
  https://www.nature.com/articles/s41598-024-74282-y
  Uses dynamic connectivity from ADHD-200/NYU data and majority-vote deep
  classification, including subtype discrimination.

- MDD multimodal fMRI/EEG GNN treatment-response model:
  https://www.nature.com/articles/s41380-025-02974-6
  Relevant as a future direction for treatment prediction, but not the first
  proof target for O-SSM.

- State of the Brain 2025 fMRI trends:
  https://apertureneuro.org/article/157595-reflections-on-current-trends-in-functional-mri-the-state-of-the-brain-2025
  Highlights dynamic FC, intrinsic modes, and large-scale DCM/effective
  connectivity as current fMRI directions.

## Novelty Assessment

Low novelty:

- O-SSM as a generic classifier.
- Marginal BA improvements on small, confounded clinical cohorts.
- Associator ranking from uncheckpointed or untrained states.
- Results driven by direct `delta`/`flat` readout shortcuts.

Medium novelty:

- Octonionic ROI-sequence model compared fairly to H-SSM, real SSM, dFC, S4,
  and graph baselines on MDD/ADHD.
- Hidden-state probes and associator maps under site-held-out evaluation.
- Checkpointed, reproducible ROI-level associator extraction.

High novelty:

- A falsifiable benchmark showing that only temporal composition/order differs
  while mean, delta, endpoints, energy, and unordered event multiset are held
  fixed.
- A trainable O-SSM objective that preserves this order information in hidden
  state when direct shortcuts are disabled.
- A bridge from synthetic order witnesses to real rs-fMRI segments, with
  associator statistics tested as exploratory neurodynamic signatures.
- A rigorous negative-control story: O-SSM only gets credit when state/associator
  paths, not direct summaries, carry the signal.

Current grade: high potential novelty, low-to-medium achieved evidence.

The achieved evidence is interesting because the negative controls are clean.
They disfavor an easy-but-wrong story under the current assays: high
temporal-arrow accuracy was carried by direct summary terms, not by measurable
hidden octonionic dynamics. This is not a population-level statistical
falsification claim; the current BA values near 50% need confidence intervals
or permutation tests before they can support stronger language.

## Recommended Disorder Target

ADHD should be the first real clinical target.

Reasons:

- ADHD is naturally temporal/developmental and network-dynamic.
- ADHD SOTA already uses dFC, adaptive spatial-temporal networks, and
  dynamic graph models, giving clear baselines.
- ADHD-200 gives a known multi-site benchmark surface.
- The hypothesis can be framed around dynamic control/attention-state evolution,
  without making premature treatment-response claims.

MDD remains important, but should be second:

- MDD is highly heterogeneous.
- Treatment response and subtype work quickly require multimodal or clinical
  covariates.
- Static/dynamic FC baselines can report very high within-cohort performance,
  so site/generalization controls matter even more.

## Experimental Program

### Phase 1: Synthetic order objective

Goal: prove that the trainable O-SSM can preserve non-associative order in
hidden state when shortcuts are unavailable.

Required gates:

- fixed-pair unit-anchor synthetic assay
- Fano-cycle synthetic assay
- `readout_delta_scale=0.0`
- `readout_flat_scale=0.0`
- hidden-state nearest-centroid BA clearly above chance
- H-SSM and real-state baselines remain near chance
- invariant audit passes: mean/delta/start/end/energy/multiset unchanged

Patch target:

- add an opt-in state-order auxiliary or contrastive trace objective
- keep default training unchanged
- require `trace_hidden_state=1` for assay runs

### Phase 2: Real ADHD dynamic benchmark

Goal: test whether the state objective transfers to real rs-fMRI.

Phase 2 is locked until Phase 1 passes a quantitative recovery gate:

- O-SSM hidden-state nearest-centroid BA at least `60%` on unit-anchor
  fixed-pair synthetic assay across `20` seeds.
- O-SSM classifier BA at least `60%` with `delta/flat` disabled.
- H-SSM remains below `55%` on the same assay.
- Permutation or label-swap control returns to chance.
- 95% CI or permutation p-value is reported for the hidden-state probe.

Dataset:

- ADHD-200 first, preferably site-aware split.

Models:

- static FC + linear/SVM baseline
- sliding-window dFC baseline
- H-SSM
- O-SSM without auxiliary
- O-SSM with state-order auxiliary
- fMRI-S4 or lightweight S4 baseline if feasible
- dynamic GNN/ASTNet-style baseline if feasible

Primary metrics:

- balanced accuracy
- AUROC
- calibration/ECE
- site-held-out stability
- hidden-state separability
- associator ROI/network enrichment with Holm-Bonferroni and Cliff's delta

Hard fail conditions:

- checkpoint missing
- direct readout shortcut explains performance
- site-held-out collapses to chance while random split looks positive
- associator extraction uses untrained weights
- ROI significance is not corrected for multiplicity

### Phase 3: MDD replication or treatment-response lane

Goal: only after Phase 1 and ADHD real benchmark, test MDD.

Preferred direction:

- not "MDD diagnosis" as the main claim
- instead: dynamic subtype or treatment-response exploratory signature

This aligns better with modern MDD literature, where heterogeneity and
multimodal treatment prediction are central.

## Immediate Next Work

1. Implement an opt-in state-order auxiliary in `examples/brain_ossm_abide.sio`.
2. Run only the unit-anchor fixed-pair assay first.
3. Require hidden-state probe improvement before touching clinical claims.
4. If synthetic gate passes, run ADHD-200 or available ADHD ROI data with
   site-aware split and checkpoint persistence.
5. Keep the paper out of the driver's seat. The artifact to optimize is a
   falsifiable scientific instrument.

## One-Sentence Internal North Star

Build an octonionic neurodynamic assay that earns the right to say when temporal
composition matters, and when it does not.
