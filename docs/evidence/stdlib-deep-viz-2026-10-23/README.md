<!-- docs:meta
topic_id: repo.docs.evidence.stdlib-deep-viz-2026-10-23.readme
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.evidence.stdlib-deep-viz-2026-10-23.readme
-->

# Scientific 2D dynamic rendering — native receipt

Slurm16121 passed the indexed numerical/visual source packet
`2526736ced258c156e2168fcb27affcb6b63b8de`: native fixtures and expected
completion markers, C++23 actual-output comparisons, and input/compiler hashes.
The source-built compiler is the protected ELF identified in COMPILER.sha256.
Remote GitHub CI and merge remain separate pending states.

Coverage includes fitted text/axis margins, forest pagination, line/interval
geometry and clipping beyond1024 points, scatter4096, strided heatmap, shared
log/symlog transforms and reduction, explicit histogram/box rendering, exact
binary64 saddle predicates and degenerate-aware bilinear contours. The renderer
consumes supplied interval and summary semantics; it does not invent confidence
intervals or statistical estimates.

The million-point reduction fixture ran0.52s at23552KiB RSS, preserving original
indices, gaps and exterior neighbors. Reduction is display-only and does not
replace observations for numerical calculations. This is one structured fixture,
not a universal renderer benchmark.

C++ compares actual SVG geometry and Canvas pixels. The saddle oracle independently
uses exact integer arithmetic for96binary64 cases; contour cases cover11panels,
14segments, masks, flat edges and exact crossings. Full actual stdout and ELFs
remain at `/scratch/sounio-stdlib-deep-20261023/viz1/artifacts`.
OUTPUTS.sha256 records their output identities. This repository receipt retains
compact results/oracle summaries rather than duplicating large pixel transcripts.
Versioned C++ emitters/checkers and Sounio fixtures permit reproduction.

Input hashes cover listed files and were verified at the endpoints. They are not
a proof of every environmental dependency or whole-tree immutability; the wrapper
may materialize its prebuilt marker. These checks establish the stated fixtures,
not general numerical, visual or publication-quality superiority.
