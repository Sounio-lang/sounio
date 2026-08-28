<!-- docs:meta
topic_id: repo.examples.semantic-orc.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A5
source_of_truth: docs/governance/topic-registry.v1.json#repo.examples.semantic-orc.readme
-->

# Semantic Entropic Transport for Ollivier-Ricci Curvature, Sounio-Native Lane

**Clinical-safety warning:** this prototype is not suitable for diagnosis,
biomarker claims, treatment-response claims, patient-level prediction, or
clinical utility claims. It is a numerical consistency gate for small
entropic-transport examples only.

Status: **prototype runtime gate**. This is a small type-checked transport
primitive, not a validated semantic-network analysis component.

ORC means **Ollivier-Ricci curvature**:
`kappa(x,y) = 1 - W1(mu_x, mu_y) / d(x,y)`, where `W1` is a Wasserstein-1
transport distance between local graph measures.

This prototype does not compute exact `W1`; it computes small
entropic-regularized transport instances that are precursors for an ORC
pipeline.

For the current SWOW16 graph-edge fixture lane, the `ORC` name marks the
intended research direction only. The current artifacts do not compute
Ollivier-Ricci curvature, do not estimate Ollivier-Ricci curvature, do not
report curvature values, do not compute or approximate Wasserstein-1 distance,
and do not make biomarker, clinical, statistical-inference, population-level,
GPU-runtime, or generalizability claims unless a specific manifest below says
otherwise.

This directory is an early Sounio-native prototype for moving
`agourakis82/hyperbolic-semantic-networks` beyond a Python/Julia mirror and
into Sounio's own epistemic compiler surface.

The first executable step is `sinkhorn_lse_orc.sio` in this same directory:

- computes matrix-dimension-bounded 2x2, 4x4, and 16x16 entropic-transport
  problems with log-domain Sinkhorn-LSE updates;
- admits the transport epsilon through `Knowledge<f64>` and
  `require_confidence`, then rejects epsilon outside the prototype range
  `[1e-3, 1e-1]`. Here epsilon is the entropic regularization parameter,
  not a confidence interval for `W1`;
- emits an explicit runtime-gate token,
  `claim_level=semantic_orc_lse_runtime_gate`;
- checks two bounded transport-ratio sanity cases: identical local support
  costs produce a positive curvature-style diagnostic token, while a 1D metric
  semantic-distance case with transport cost above the observed edge scale
  produces a negative curvature-style diagnostic token;
- checks mass plus row/column marginal constraints;
- applies 64 fixed Sinkhorn-LSE iterations to the 4x4 non-uniform marginal
  case, then checks whether the fixed-point residual is below tolerance;
- checks a 16x16 symmetric diagonal/off-diagonal cost case matching the
  K-AXI Sinkhorn16 sanity shape;
- checks a 16x16 non-uniform neighborhood proxy with decreasing/increasing
  marginals and a 1D metric cost. Here K-AXI refers to Sounio's internal
  Kretikos accelerator instruction lane for emitted Sinkhorn16 kernels.

The second executable step is a fixture-backed bridge from the CPC/SWOW export:

- `scripts/research/semantic_orc_swow16_fixture.py` reads
  `data/cpc2026/sounio_input/node_features.csv` and one
  `trajectories_<regime>_nodes_parity.csv` file from
  `agourakis82/hyperbolic-semantic-networks`;
- it selects the first 16 unique node indices available in the parity
  trajectories for a regime, then derives two positive normalized measures
  from exported semantic features;
- it derives a 16x16 semantic cost matrix from Poincare coordinates, valence,
  and correlation entropy features;
- it emits a temporary Sounio program containing those fixed arrays;
- it emits a JSON manifest with SHA-256 hashes for the generator, template,
  feature CSV, trajectory CSV, plus the selected node ids and measure/cost
  construction notes;
- `scripts/ci/semantic_orc_swow16_fixture_gate.sh` type-checks and runs that
  generated Sounio program with `bin/souc`, after checking the manifest shape.

The runtime token means exactly:

- `require_confidence(measure(0.05, uncertainty: 0.0005), 0.98)` returned an
  admitted entropic regularization parameter;
- epsilon passed `[1e-3, 1e-1]`;
- 2x2 mass error is `<= 1e-3`, each row-sum error is `<= 1e-3`, and each
  column-sum error is `<= 1e-3`;
- 4x4 mass error is `<= 1.5e-2`, row/column errors are `<= 1e-2`, and
  fixed-point residual is `<= 2e-3`;
- 16x16 **symmetric sanity-check** mass, row/column, and fixed-point residual
  errors are `<= 2e-3`;
- 16x16 **non-uniform neighborhood proxy** mass, row/column, and fixed-point
  residual errors are `<= 1e-2`, `<= 1e-2`, and `<= 3e-3`, respectively;
- the N=16 non-uniform case uses adaptive stopping and must satisfy the
  residual threshold within 128 iterations;
- the token also emits `instance_checks_passed=true`,
  `entropic_regularized_transport_only=true`,
  `clinical_claims_not_enforced=true`, and `no_convergence_theorem=true`;
- no proof of convergence to an optimal transport plan, clinical theorem, or
  general N=16 solver claim is implied.

The SWOW/CPC fixture runtime token means exactly:

- the input directory contained `node_features.csv` and the selected
  `trajectories_<regime>_nodes_parity.csv`;
- 16 unique node indices were selected from exported parity trajectories;
- a provenance manifest was created with generator, template, feature-file, and
  trajectory-file hashes;
- the generated Sounio source was accepted by `bin/souc check`;
- the generated Sounio source emitted
  `SOUNIO_SWOW16_ENTROPIC_TRANSPORT_PASS`;
- the fixture-backed N=16 entropic-transport instance satisfied mass,
  row/column marginal, fixed-point residual, positive-cost, adaptive-stop,
  epistemic epsilon, and epsilon-range runtime gates;
- the token also emits `entropic_regularized_transport_only=true`,
  `clinical_claims_not_enforced=true`, and `no_convergence_theorem=true`.

This is not yet the full Computational Psychiatry Congress depression pipeline
in Sounio. It is a first type-checked and runtime-gated transport primitive
needed to get there.

## Commands

```bash
bin/souc check examples/semantic_orc/sinkhorn_lse_orc.sio
bin/souc run examples/semantic_orc/sinkhorn_lse_orc.sio
bash scripts/ci/semantic_orc_sinkhorn_lse_gate.sh
bash scripts/ci/semantic_orc_swow16_fixture_gate.sh
SOUNIO_SEMANTIC_ORC_REGIME=anxious bash scripts/ci/semantic_orc_swow16_fixture_gate.sh
SOUNIO_SEMANTIC_ORC_REGIME=ruminative bash scripts/ci/semantic_orc_swow16_fixture_gate.sh
SOUNIO_SEMANTIC_ORC_REGIME=psychotic bash scripts/ci/semantic_orc_swow16_fixture_gate.sh
bash scripts/ci/semantic_orc_swow16_kaxi_pack_gate.sh
bash scripts/ci/semantic_orc_swow16_multisupport_gate.sh
bash scripts/ci/semantic_orc_swow16_permutation_fixture_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_kaxi_pack_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_multifixture_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_tile_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_reducer_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_edge_tile_matrix_parameter_sweep_gate.sh
bash scripts/ci/semantic_orc_swow16_graph_degree_shuffle_fixture_gate.sh
```

## Current Deep Lane Evidence

All values in this section are engineering acceptance metrics for bounded
fixtures. They are not clinical cohort comparisons, biomarker effects,
diagnostic separations, psychiatric group rankings, exact transport distances,
or ORC values. Paths under `/orangefs` are internal persistent run artifacts
for this workspace, not standalone public reproduction packages.

The Slurm GPU path passed on 2026-05-25 for four CPC/SWOW export labels using
pre-emitted `sinkhorn16` PTX and a prebuilt K-AXI runner:

- manifest:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/sounio/artifacts/semantic_orc/slurm/swow16-kaxi-four-regime-20260525T1130Z.json`
- runtime results: 4/4 `launch_pass`
- worst GPU-vs-pack-oracle differences:
  `maxdu=7.419522394691569e-06`,
  `maxdv=6.17755054155289e-06`

The multisupport Slurm GPU path also passed on 2026-05-25:

- manifest:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/sounio/artifacts/semantic_orc/slurm/swow16-kaxi-multisupport-20260525T114751Z/swow16_kaxi_slurm_multisupport_manifest.json`
- runtime results: 16/16 `launch_pass`
- tolerance: `0.005` for both `maxdu_vs_pack_oracle` and
  `maxdv_vs_pack_oracle`
- worst GPU-vs-pack-oracle differences:
  `maxdu=4.238789615307326e-05`,
  `maxdv=5.941514330753961e-06`

The edge-by-tile matrix gate passed on 2026-05-25:

- manifest:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/sounio/artifacts/semantic_orc/graph_tile_matrix/swow16-graph-tile-matrix-20260525T131852Z/swow16_graph_edge_tile_matrix_manifest.json`
- schema: `sounio.semantic_orc.swow16_graph_edge_tile_matrix_manifest.v1`
- edges: 4 deterministic real SWOW edges using edge stride `13`
- tiles per edge: 8 deterministic 16-node support tiles
- total fixtures: 32 generated Sounio/K-AXI support tiles
- edge labels in this run: `age -> old`, `amazing -> great`,
  `animal -> horse`, `area -> square`
- endpoint-free matrix enumeration: 309 unique endpoint-free nodes out of 448
  endpoint-free tile positions
- runtime result: 32/32 graph-edge support tiles passed `check`/`run` and K-AXI
  pack validation
- matrix manifest sentinel:
  `numerical_values_are_engineering_diagnostics_only=true`

The edge-by-tile matrix values are deterministic fixture-enumeration and
packaging evidence only. Edge stride and tile stride are deterministic
engineering enumeration parameters, not sampling, coverage, or
representativeness designs. The current evidence proves only the recorded
4-edge x 8-tile parameter point, not a sweep over other settings.

The Sounio-native reducer over that matrix also passed on 2026-05-25:

- manifest:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/sounio/artifacts/semantic_orc/graph_tile_matrix_reducer/swow16-graph-tile-matrix-reducer-20260525T133730Z/swow16_graph_edge_tile_matrix_reducer_manifest.json`
- schema:
  `sounio.semantic_orc.swow16_graph_edge_tile_matrix_reducer_manifest.v1`
- reducer role: generated Sounio manifest-invariant check only; not transport
  recomputation, not exact Wasserstein-1, not exact ORC, not an ORC estimator,
  not GPU runtime evidence, not statistical inference, and not biomarker or
  clinical evidence
- generated Sounio runtime result: `check` and `run` passed and emitted
  `SOUNIO_SWOW16_GRAPH_EDGE_TILE_MATRIX_REDUCER_PASS`
- matrix invariant summary: 4 deterministic real SWOW edges, 8 support tiles
  per edge, 32 total fixtures, 309 unique endpoint-free nodes out of 448
  endpoint-free tile positions

The reducer intentionally checks shape, source-manifest hash, diagnostic-only
sentinels, and claim-boundary tokens. It records upstream matrix diagnostics
but does not revalidate transport residuals and is not a scientific reducer,
exact transport solver, curvature estimator, null-model analysis, GPU result,
or clinical validation.
The reducer's endpoint-free invariant is tied to this fixture schema: every
support tile has 16 nodes, of which two are the selected SWOW edge endpoints,
leaving exactly 14 endpoint-free tile positions per fixture.

The explicit graph-edge/tile green-point parameter check passed on 2026-05-25:

- manifest:
  `/orangefs/training/sounio/moonshot-a-runtime/moonshot-a-phase-status-codex-20260524T2038/sounio/artifacts/semantic_orc/graph_tile_matrix_parameter_sweep/swow16-graph-tile-matrix-parameter-sweep-final-20260525T135220Z/swow16_graph_edge_tile_matrix_parameter_sweep_manifest.json`
- schema:
  `sounio.semantic_orc.swow16_graph_edge_tile_matrix_parameter_sweep_manifest.v1`
- explicit parameter points: `2x4`, `4x4`, and `4x8`
- point count: 3
- total edge/tile fixtures across points: 56
- upstream pack-oracle row/column tolerance used by this lane: `0.015`
- runtime result:
  `3/3 explicit parameter points passed generated Sounio reducer gates`
- worst inherited K-AXI oracle diagnostics across the passed points:
  `max_row_err=0.014439138319142952`,
  `max_col_err=1.3569757706388685e-08`
- local and OrangeFS executions both passed with the same default point set

This sweep proves only the exact edge-count x tile-count points recorded in
the manifest. It is not interpolation, extrapolation, random sampling,
coverage, representativeness, or a full sweep over the SWOW graph. It also
does not compute or approximate exact Wasserstein-1 distance, exact
Ollivier-Ricci curvature, an ORC estimator, a biomarker, a clinical result,
statistical inference, GPU-runtime evidence, population-level evidence, or
generalizability evidence.

Default execution is intentionally restricted to the recorded green points
`2x4`, `4x4`, and `4x8` because the nearby `8x4` expansion point currently
fails. Exploratory non-default points require
`SOUNIO_SEMANTIC_ORC_ALLOW_EXPERIMENTAL_POINTS=1` and, even if they pass, are
still exact-point evidence only. The aggregation gate cross-checks each point's
recorded edge count, tile count, edge stride, tile stride, manifest digests,
diagnostic-only sentinels, Sounio reducer PASS token, and inherited
pack-oracle row/column diagnostics. It delegates transport-quality checks to
the upstream graph-edge tile and matrix gates; it does not recompute transport
or independently revalidate residuals at the aggregation level.

The `max_row_err` and `max_col_err` values in this subsection are inherited
K-AXI pack-oracle diagnostics from the graph-edge tile lane, whose explicit
prototype acceptance threshold is `0.015` for row and column errors in this
lane. They are not the same tolerance family as the hand-written
`sinkhorn_lse_orc.sio` 16x16 neighborhood proxy. The row/column asymmetry is
therefore recorded as an engineering diagnostic for the fixed-iteration
pack-oracle surface, not interpreted as transport quality, curvature, or
clinical evidence.

An attempted local expansion point, `8x4`, did not pass this prototype
tolerance: it failed at `edge_index=4`, `tile_index=1` with
`K-AXI oracle row error is outside prototype tolerance`. That failed point is
the next expansion blocker, not part of the green sweep evidence above. Its
root cause has not yet been classified; the next rung is to determine whether
it is numerical tolerance pressure, graph-topology sensitivity, fixture
generation behavior, or a real limitation of the fixed-iteration K-AXI oracle
surface.

## Why This Is More Sounio Than A Port

The historical note
`docs/research/hyperbolic_semantic_networks_run.md` identified primal Sinkhorn
as the wrong algorithmic surface for the semantic-network lane. This prototype
does not yet reproduce that full comparison. It establishes a small
current-source Sounio runtime gate for the replacement surface: log-domain
updates, admitted epsilon, marginal checks, a non-uniform 4x4 fixed-point
residual check, an N=16 symmetric fixed-point gate, an N=16 non-uniform
neighborhood-proxy gate, and a generated N=16 fixture gate backed by exported
CPC/SWOW node features.

The local contract is deliberately narrow:

- no ORC runtime-gate token without an admitted entropic regularization
  parameter;
- no semantic-network prototype output from this lane without log-domain
  transport;
- no external-facing numerical claim based on this lane without including the
  runtime-gate output and the `entropic_regularized_transport_only=true`
  limitation.

This prototype does **not** yet enforce global clinical-claim discipline. The
ungated claim types are biomarker claims, diagnostic claims, treatment-response
claims, patient-level prediction claims, and any statement about clinical
utility. Those require future null-model, size-control, and per-subject graph
evidence before any marker or biomarker wording.

## Known Limitations

- The runtime gate now includes a non-uniform N=16 neighborhood proxy, but it
  still does not prove convergence for arbitrary N=16 or larger supports.
- The current-source local `exp`/`ln` approximations make the 4x4 marginal
  tolerance intentionally loose (`<= 1.5e-2` mass error, `<= 1e-2`
  row/column error). The fixed-point residual gate is tighter (`<= 2e-3`).
  These are prototype numerical tolerances, not clinical tolerances; they are
  too loose for downstream curvature interpretation.
- The `Knowledge<f64>` gate exercises Sounio's epistemic metadata path for the
  regularization parameter, but it does not by itself prove a numerical
  stability theorem or quantify W1 uncertainty. The `[1e-3, 1e-1]` range is a
  prototype numerical range, not a clinically validated range.
- The runtime token is tied to marginal and fixed-point checks, not a formal
  Lean proof object or optimality theorem.
- The fixed-point checks are post-iteration residual checks. They do not
  monitor residual monotonicity during iterations.
- The 4x4 non-uniform case uses 64 fixed Sinkhorn-LSE iterations and accepts
  only if the marginal and fixed-point residual checks pass after those
  iterations; residual monotonicity is not monitored.
- The N=16 non-uniform proxy uses adaptive stopping on the fixed-point residual,
  capped at 128 iterations.
- The hand-written 16x16 non-uniform case uses synthetic marginals/costs. The
  fixture-backed gate uses exported SWOW/CPC features, but still uses a
  generator to emit fixed Sounio arrays rather than loading CSV directly inside
  Sounio.
- The fixture-backed gate selects a bounded N=16 support from parity
  trajectories. It is not yet an all-node graph pass, a per-subject graph pass,
  or a depression-severity analysis.
- No null model, size-control, or depression severity analysis is implemented
  in this directory yet; the prototype is unsuitable for clinical
  interpretation until those gates exist and pass.

## Next Rungs

1. Classify and repair, relax, or explicitly parameterize the `8x4` expansion
   blocker before claiming a wider edge/tile parameter envelope.
2. Replace generated fixed arrays with a Sounio-owned fixture reader or a
   typed Sounio module that preserves the existing checksum manifest contract.
3. Add residual-monitoring gates for the 4x4 and N=16 non-uniform cases.
4. Add a multi-regime fixture manifest that records selected node ids, measure
   construction, cost construction, and gate output per regime.
5. Recompute the small random-regular curvature transition witness with CPU
   Sinkhorn-LSE and define the transition criterion explicitly.
6. Attach null-model and size-control gates before reintroducing depression
   severity interpretation.
