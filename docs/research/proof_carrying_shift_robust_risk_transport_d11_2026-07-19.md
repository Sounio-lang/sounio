<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-shift-robust-risk-transport-d11-2026-07-19
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-shift-robust-risk-transport-d11-2026-07-19
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# D11 Research Note: Shift-Robust Risk Transport

Date: 2026-07-19

## Research Question

When does evidence supporting a bounded source-population canary remain usable
for a target population, and what information must force degradation,
suspension, or revocation instead?

The answer adopted by D11 is deliberately asymmetric: target evidence may
preserve a canary rank or move it downward. It cannot create external
validation, production permission, or patient-specific clinical authority.

## Statistical Anchors

- Tibshirani, Barber, Candes, and Ramdas show weighted conformal prediction
  under covariate shift when the source-to-target likelihood ratio is known or
  accurately estimated: https://papers.neurips.cc/paper/2019/hash/8fb21ee7a2207526da55a679f0332de2-Abstract.html
- Lipton, Wang, and Smola define black-box label-shift estimation and require an
  invertible confusion matrix for identification:
  https://proceedings.mlr.press/v80/lipton18a.html
- Qiu, Dobriban, and Tchetgen Tchetgen establish an impossibility boundary for
  informative finite-sample prediction under completely unknown covariate
  shift: https://doi.org/10.1093/jrsssb/qkad069
- Angelopoulos et al. extend conformal prediction to expected bounded monotone
  loss: https://proceedings.iclr.cc/paper_files/paper/2024/hash/f3549ef9b5ff520a7e41ff3cc306ab2b-Abstract-Conference.html
- Barber et al. quantify conformal degradation beyond exchangeability:
  https://doi.org/10.1214/23-AOS2276
- Farinhas et al. derive nonexchangeable conformal-risk penalties:
  https://proceedings.iclr.cc/paper_files/paper/2024/hash/de04896f011beff76c91e094f72727f4-Abstract-Conference.html
- Cauchois et al. study robust prediction over declared divergence balls:
  https://doi.org/10.1080/01621459.2023.2298037
- Gibbs, Cherian, and Candes provide finite-sample conditional guarantees over
  declared function classes, not universal individual-conditional coverage:
  https://doi.org/10.1093/jrsssb/qkaf008
- Hultberg, Zachariah, and Ribeiro give 2026 preprint results for anytime-valid
  conformal risk under test-time shift with given joint importance weights and
  bounded monotone right-continuous loss: https://doi.org/10.48550/arXiv.2602.04364

The D11 fixtures implement none of those general algorithms or theorems. The
papers motivate assumptions and failure boundaries; the executable claim is
only the enumerated finite arithmetic in this lane.

## Clinical And Lifecycle Anchors

- IMDRF N88 final GMLP principles address intended population, subgroups,
  environment, human-AI workflow, independent testing, and monitoring across
  the lifecycle:
  https://www.imdrf.org/sites/default/files/2025-02/IMDRF_AIML%20WG_GMLP_N88%20Final.pdf
- FDA's August 2025 final PCCP guidance recommends that a PCCP describe planned
  modifications, implementation and validation methods, and impact assessment;
  a plan alone is not authorization:
  https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
- FDA's postmarket research program explicitly distinguishes changes in input
  data, output performance, patient populations, protocols, and clinical sites:
  https://www.fda.gov/medical-devices/medical-device-regulatory-science-research-programs-conducted-osel/methods-and-tools-effective-postmarket-monitoring-artificial-intelligence-ai-enabled-medical-devices
- WHO post-market surveillance guidance provides a framework for investigating
  continued-use risk and selecting corrective or preventive action:
  https://www.who.int/publications/i/item/9789240015319
- Steingrimsson et al. derive identification conditions for transporting a
  prediction model and target-population performance:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11004796/
- Riley et al. emphasize target-representative validation, calibration,
  discrimination, subgroup checks, and decision utility as distinct analyses:
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10788734/

FDA's broader AI-device lifecycle guidance remained draft, not final guidance,
at the review date and is not treated as binding authority here.

## Frozen Contest Table

| ID | Collision | Exact result |
|---|---|---|
| W0 | D10 source boundary | synthetic canary only; zero production/clinical authority |
| W1 | covariate weighting | `(3,9)/12 -> (6,6)/12`; `1/4 -> 1/2`; weighted source risk equals target risk |
| W2 | overlap failure | identical observed source evidence; target risk in `[0,1/2]` |
| W3 | label weighting and singularity | `(3,9)/12 -> (6,6)/12`; `1/4 -> 1/2`; probe `31311` is distinct from loss `31711` |
| W4 | concept ambiguity | identical unlabeled target inputs and scores; risks `2/4` and `4/4` |
| W5 | marginal/subgroup | both `6/12`; worst subgroup `1/2` versus `1` |
| W6 | local calibration | diagnostic residual `0 -> 1/4` cannot transition state; active later residual `-1/2` supplies a private trigger |
| W7 | shifted conformal risk | source `1/4`, TV `1/4`, target `1/2`, bound tight |
| W8 | authority attenuation | ranks `3 -> 3 -> 2 -> 1 -> 0`, windows `31121 -> 31122 -> 31123 -> 31124`, no upward edge |

The positive evidence is not combined from unrelated laws that merely share an
ID. One private exact-law token fixes source atom counts `(3,3,3,3)`, target
atom counts `(6,2,2,2)`, denominator `12`, and loss `(1,0,0,0)`. Its
deterministic noncryptographic fingerprint is `92352734845403155`. Coarsening
atom 0 versus atoms 1..3 gives deterministic `X=Y`, for which the declared
covariate and label conditionals are jointly compatible. Label identification
uses a separate perfect probe rather than treating the evaluated loss as a
classifier. The private law also freezes the atom projection, probe confusion
matrix, diagnostic and active scores, and positive subgroup allocation; each
positive certifier checks its entire public observation payload against those
private fields before emitting a token. The private binder then checks the same
exact-law token, fingerprint, fixture run, model, source/target population,
evidence window, loss, calibrator, label probe, subgroup plan, and member-wise
scope proof before it can preserve the canary rank.

D10 attests the source canary boundary but does not attest the invented D11
scope relation. D11's positive subset token explicitly binds source members
`3114101..3114108` and target members `3114101..3114104`; this is a frozen
fixture proof, not a general set-inclusion implementation.

## Interpretation Boundary

The strongest supported statement is that exact source evidence is not enough:
target transport requires assumption-indexed, identity-bound evidence for all
six dimensions and an explicit risk envelope. Missing overlap, singular label
identification, unlabeled concept ambiguity, subgroup harm, local calibration
drift, or target-risk breach blocks continuation.

Actual suspension, recall, decommissioning, or clinical use remains the act of
a competent institution under the applicable jurisdiction. D11 records only a
synthetic nominal fixture trace and reserves institutional authority as an
opaque unproducible type. Its final revoked token is terminal only in that
trace: fixture producers remain replayable, values remain copyable, no runtime
canary is disabled, and global absorption or a unique execution chain is not
proved.
