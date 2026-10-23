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

## Stage2 compatibility repair, 2026-10-23

PR3048 CI run39997094713/job120567131861 exposed six renderer crashes and
two stale SVG line markers. Its base run39994861909/job120559563649 passed;
the eight fixtures were unchanged. A later science CI also exposed crashes in
`viz_bca_mean` and `viz_permutation_mc`. No engine requirement was added to hide
these previously supported Stage2 programs.

The repair nests the axis tick ordering check so tick zero never evaluates an
index of minus one. Axis emitters recompute canonical label text from each
scalar tick instead of copying an indexed nested label record out of the
returned plan. Planned coordinates, fit decisions and the plan schema remain
unchanged. Forest expectations now identify the actual clipped pixel-cell
rectangles; their Canvas assertions remain unchanged.

The exact failing-CI Stage2 ELF has SHA256
`fea662a291d0b57e99bfadbdd09d0984e1f348a938a20bb8ca1c962d250c1571`.
Slurm16177 reproduced the original entropy crash and both marker mismatches;
the ordering guard alone removed the count-matrix crash but left five other
crashes. Slurm16183, with scalar label emission, executed all eight original
programs successfully, leaving only the expected old forest marker mismatch.
Slurm16184 independently verified both replacement SVG rectangle geometries in
C++23. Actual output uses literal quotes.

Final Slurm16188 passed compilation, execution and every expected stdout marker
for all ten legacy fixtures plus `viz_axis_first_tick_guard` on Stage2. It also
passed Madaros check/compile/execution/markers for the focused axis-layout,
forest and new first-tick fixtures, plus the two existing C++23 actual-output
axis/forest comparisons and the legacy forest span comparison. Madaros reused
the attested compiler SHA256
`1148eabd3905ec76911d027c578fb8d6b2bd2dde2c186ee3c02c4c3b97733029`
with the frozen updated visual stdlib and isolated uncached invocation.
These are focused controls, not a full corpus or remote CI pass.

The valid first-tick fixture also passes against the original PR head. It is
coverage-only, not a failing-before witness. An earlier version had an invalid
label budget and is not regression evidence. Slurm16175 initially produced
execution status126 because the harness omitted executable permission on
Stage2 output; Slurm16177 retried the same byte-identified ELFs with the harness
repaired. These raw statuses remain separate.

Compact phase statuses, resource receipts, source manifests, executable/output
identities and C++ summaries are retained in `stage2-compat/`. Full outputs,
compiler logs and ELFs remain under
`/scratch/sounio-stdlib-deep-20261023/pr3048-ci-controls/artifacts`.
Source inventories were verified unchanged after the final execution. No
measured speed or memory superiority is claimed.
