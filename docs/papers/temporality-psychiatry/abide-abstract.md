<!-- docs:meta
topic_id: repo.docs.papers.temporality-psychiatry.abide-abstract
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.papers.temporality-psychiatry.abide-abstract
-->

# Provisional Abstract: Non-Associative State Composition on Real Temporal Neurodynamics

We investigate whether non-associative state composition provides measurable
practical benefit on real temporal neurodynamics rather than only on synthetic
algebraic probes. Our target model, the octonionic state-space model (O-SSM),
uses non-associative composition in the recurrent state update; the central
question is whether this extra structure helps when temporal order carries real
signal. We evaluate this claim on ABIDE-derived fMRI regional time series under
site-aware generalization, parameter-matched comparisons, and mechanism-facing
ablations. The key comparison is against an associative hypercomplex baseline
(H-SSM) and a diagonal baseline with comparable parameter budgets.

Our hypothesis is that non-associativity should help most when temporal
grouping order matters, especially under distribution shift, low-data regimes,
or transition-heavy neural sequences. We therefore do not evaluate only headline
accuracy, but also balanced accuracy, calibration, cross-seed variance,
cross-site robustness, and order-sensitive auxiliary tasks. We further track
associator-aware diagnostics over time to test whether any empirical gain is
actually tied to non-associative dynamics rather than incidental optimization
effects.

The strongest expected outcome is not merely a higher accuracy number, but a
demonstration that non-associative temporal composition yields a reproducible
advantage in at least one practically relevant regime while remaining
mechanistically interpretable. Even if the gain is conditional rather than
universal, such a result would establish a concrete domain in which
non-associativity is computationally useful on real data. Conversely, if O-SSM
fails to outperform associative baselines under site-aware controls, the study
will still sharply delimit the practical scope of non-associative sequence
models and provide a benchmark framework for future work.

In either case, the work turns a theoretical novelty into a falsifiable
real-data program: non-associativity matters if and only if it improves
inference on temporal neurodynamics in regimes where order-sensitive composition
is scientifically meaningful.
