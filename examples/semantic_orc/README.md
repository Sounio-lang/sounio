<!-- docs:meta
topic_id: repo.examples.semantic-orc.readme
authority: repo_only
audience: researchers
last_validated: 2026-05-25
validated_by: Codex
source_of_truth: examples/semantic_orc/
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
- checks two bounded sanity cases: identical local support costs produce
  positive ORC, while a 1D metric semantic-distance case with transport cost
  above the observed edge scale produces negative ORC;
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
```

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

1. Replace generated fixed arrays with a Sounio-owned fixture reader or a
   canonical generated-source manifest with checksums.
2. Add residual-monitoring gates for the 4x4 and N=16 non-uniform cases.
3. Add a multi-regime fixture manifest that records selected node ids, measure
   construction, cost construction, and gate output per regime.
4. Recompute the small random-regular curvature transition witness with CPU
   Sinkhorn-LSE and define the transition criterion explicitly.
5. Pack the same N=16 inputs for the K-AXI Sinkhorn16 GPU kernel.
6. Attach null-model and size-control gates before reintroducing depression
   severity interpretation.
