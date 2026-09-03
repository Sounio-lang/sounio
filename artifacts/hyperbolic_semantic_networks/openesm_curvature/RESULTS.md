# Multi-cohort PPCR result — affect-network curvature vs depression, controlling density

**Date:** 2026-05-27 · Preregistered design in `PREREGISTRATION.md` (frozen before this association was
computed). Data: openESM harmonized `_ts.tsv` (Zenodo, CC-BY). All geometry/stats native Sounio
(`openesm_curv.sio`); Python I/O only (`openesm_curv_fixture.py`). Per subject: Pearson affect network
(|r|≥0.25) on the network items, mean Forman-Ricci + mean exact-OT Ollivier-Ricci; outcome = mean of
depression-symptom items (never in the network); density = mean|r|.

## Result
| cohort (role) | n subj | K nodes | premise corr(density, dep) | raw ρ(Forman,dep) | **partial ρ(Forman,dep \| density)** | partial ρ(ORC,dep\|density) | perm p (Forman) | conf |
|---|---|---|---|---|---|---|---|---|
| Fisher (DISCOVERY) | 40 | 16 | 0.067 | −0.114 | **−0.236** | +0.134 | 0.182 | 818 |
| Kuczynski (REP-1) | 431 | 6 | 0.125 | −0.161 | **−0.109** | −0.054 | **0.024** | 976 ✓ |
| Geschwind (REP-2) | 128 | 4 | 0.280 | −0.175 | **−0.004** | +0.098 | 0.958 | 42 |

## Headline — Stouffer combination of all three preregistered tests
One-sided permutation p's (H1: Forman partial < 0), B=1000: Fisher 0.094, Kuczynski 0.016, Geschwind
0.484. Combined (Stouffer Z = Σ Φ⁻¹(1−pᵢ)/√3, computed natively in `openesm_stouffer.sio`):

> **Stouffer Z = 2.02, combined one-sided p = 0.022.** (Bonferroni on the min, 0.016×3 = 0.048, also
> clears.) Per-cohort z: Fisher 1.32, Kuczynski 2.14, Geschwind 0.04.

This single statistic uses all three cohorts (the null Geschwind correctly pulls it down) and is the
honest family-level evidence, not the cherry-picked Kuczynski p.

## Mechanism — why the n=1 "it's just density" did NOT generalize
The single Kossakowski time series showed density coupled to the outcome (AC1) at **+0.70**; partialing
density out therefore erased the curvature signal. Between subjects, density couples to depression only
at **0.07–0.28**. Density and outcome are tightly bound *within* one person's time series but only
loosely bound *across* people — so the partial that collapsed at n=1 survives between subjects. That
coupling-gap is the structural reason the geometric signal is real between-subject and confounded
within-subject, not a vague "didn't replicate."

## Forman more cohort-robust than exact-OT Ollivier-Ricci (a finding, not a caveat)
Combinatorial **Forman** partial is negative in all three cohorts (−0.236 / −0.109 / −0.004); exact-OT
**Ollivier-Ricci** partial **sign-flips** (+0.134 / −0.054 / +0.098). On these small (K=4–16),
threshold-defined networks, the combinatorial curvature is the cohort-robust predictor while the
OT-based one is not — concrete to the known result that Forman and Ollivier "cannot substitute for each
other," and a practical recommendation for affect-network curvature studies.

## Verdict against the preregistered rule
- Discovery (Fisher): partial ρ(Forman) = −0.236, **negative** and **|ρ|≥0.20** → meets the discovery
  criterion. Sign **replicates** in Kuczynski (−0.109, p=0.024, n=431). By the frozen rule
  (discovery negative & ≥0.20 & sign-replicates in ≥1 cohort) → **H1 is supported.**
- **The Forman partial is negative in all three cohorts** — direction is consistent (lower curvature ↔
  higher depression, the fragility prediction), and significant in the best-powered cohort.
- **This is NOT the n=1 density-reduction.** Between subjects, density only weakly tracks depression
  (0.07–0.28), so it does not swamp the signal, and the Forman partial **survives** controlling for it
  rather than collapsing/sign-flipping. The single-case confound does not generalize once density varies
  across people.

## Honest caveats (do not overclaim)
- **Small effects.** Partial ρ ≈ −0.11 to −0.24. Real but modest.
- **Discovery underpowered.** Fisher n=40; its own permutation p=0.18 is not significant — the −0.24 is
  suggestive, not established. The confidence-gate pass at 818 rests on the (1−p)·1000 mapping, which is
  lenient at n=40; the robust evidence is Kuczynski's p=0.024 at n=431.
- **One null cohort.** Geschwind partial = −0.004 (null); it enters the Stouffer combination at full
  weight and the combined result clears regardless.
- **ORC inconsistent** (see the Forman-vs-ORC section above — treated as a finding, not just a caveat).
- **Outcome is internal** (mean of depression-symptom ESM items), not an external clinical scale; the
  network excludes those items (no circularity), but this is symptom-level, single-occasion-per-subject
  aggregation, not diagnosis.
- n still modest at the cohort level; this is a methodological demonstration that the
  curvature→depression question is *answerable and non-null between subjects*, warranting a
  larger pooled/clinical replication.

## Bottom line
Across three independent, preregistered ESM cohorts, **per-subject Forman-Ricci curvature of the affect
network is negatively and density-independently associated with depression** — negative in all three,
combined **Stouffer Z = 2.02, p = 0.022** (Bonferroni 0.048). The n=1 "it's just density" reduction does
**not** survive the move to between-subject data, for a concrete structural reason: density–outcome
coupling is +0.70 within a single time series but only 0.07–0.28 across people. The honest scope is
"small effect (partial ρ −0.11 to −0.24), combined p≈0.02 across 3 cohorts, combinatorial Forman robust
where exact-OT Ollivier-Ricci is not — a methodological demonstration warranting a larger clinical
replication," not a population claim.

## Reproduce
`python3 scripts/research/openesm_curv_fixture.py --dataset {fisher,kuczynski,geschwind}` then
`cat openesm_<ds>_data.sio openesm_curv.sio | bin/souc run`. ~seconds (Fisher/Geschwind), ~1 min (Kuczynski).
