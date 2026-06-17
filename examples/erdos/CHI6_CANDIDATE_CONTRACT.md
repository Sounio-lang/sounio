# CHI6 Candidate Contract

This is the promotion contract for a real unit-distance graph lower-bound
candidate toward `chi(R^2) >= 6` in the Sounio Erdos lane. Search code, GPU
kernels, cube manifests, and SAT producers are untrusted. Mathematical promotion
requires exact geometry plus a Lean 4-checked no-5-colouring certificate.
The `chi6` prefix in files and scripts names this target lane only; it is not a
claim that such a witness currently exists. Until every promotion gate below is
met, manifests must remain `promotable=0` and carry no chromatic lower-bound
claim.

The reusable Lean 4 API for this contract lives in
`formal/lean4/SounioFiniteUnitDistanceWitness.lean`: `ExactFieldLike`,
`ExactSquaredDistancePlane`, `NatEdgeExactGeometry`, and
`EuclideanNatEdgeExactGeometry`. The calibration smoke
`formal/lean4/SounioFiniteUnitDistanceEuclideanSmoke.lean` shows that the
Euclidean contract type is inhabited by a minimal two-point, one-edge exact
squared-distance object over `Rat^2` and by a rational unit square with the four
unit side edges only; the diagonals are not asserted as unit edges. It is not a
scalability demonstration, attaches no no-5-colouring certificate, and is not a
candidate witness.

## Required Artifacts

1. Candidate graph
   - Stable vertex count `n` and edge count `m`.
   - Plain DIMACS edge file with header `p edge n m`.
   - DIMACS vertex ids are 1-based; Lean/SAT generator edge lists and
     `triangle_sb` manifest ids are 0-based.
   - Edge ordering used identically by the SAT certificate and Lean geometry.
     A promotable package must expose the same ordered edge list as a Lean term,
     generated from or proved equal to the declared DIMACS artifact; an
     auditor-visible file hash alone is not enough for promotion.
   - SHA256 of the edge file recorded in the candidate manifest, computed over
     raw file bytes with `sha256sum`-compatible semantics and no line-ending or
     text encoding normalization.
   - Solver/search handoff should prefer the source-bundle contract
     `examples/erdos/schemas/chi6_solver_candidate_package.v1.schema.json`,
     validated by `examples/erdos/validate_chi6_solver_candidate_package.py`.
     This JSON binds `candidate_id`, `edge_path`, `edge_sha256`,
     `coords_path`, `coords_sha256`, `coordinate_domain=rational_xy`, `n`, `m`,
     `k=5`, `split_vertices`, producer command, and claim boundary before any
     downstream SAT or geometry maker runs. It is provenance only:
     `claim_scope=solver_candidate_source_only`, `sat_claim=none`,
     `chromatic_claim=none`, and `promotable=0`.

2. Exact planar geometry
   - A Lean point type `P` for the exact coordinate domain.
   - A Lean relation `unit : P -> P -> Prop`.
   - An embedding `emb : Nat -> P`.
   - A `UnitDistanceChromatic.EuclideanNatEdgeExactGeometry n P unit`
     object whose `ExactSquaredDistancePlane` component ties `unit p q` to
     exact two-coordinate squared distance equal to `1` over an
     `ExactFieldLike` scalar-law package.
   - Proofs:
     - `emb_injective : forall i j, i < n -> j < n -> emb i = emb j -> i = j`
     - `endpoints : forall e in edges, e.1 < n /\ e.2 < n`
     - `unit_edges : forall e in edges, unit (emb e.1) (emb e.2)`
     - metric sanity facts from `ExactSquaredDistancePlane`: zero squared
       distance iff point equality, unit symmetry, and unit irreflexivity.
   - Floating-point, GPU, or PTX output may guide search, but cannot be the
     promoted geometry proof.
   - For rational-coordinate candidates, the data-driven bridge is
     `examples/erdos/gen_lean_rational_geometry.py`: it consumes the same
     DIMACS `p edge` graph plus a zero-based `id,x,y` coordinate CSV, rejects
     duplicate/collapsed points and non-unit listed edges, and emits a Lean
     module inhabiting `EuclideanNatEdgeExactGeometry` over `Rat^2`. The same
     generated module also exposes a standard `Real × Real` unit-distance
     relation, a proof that this relation is the repository's expanded
     squared-distance formula, and per-edge Real unit proofs obtained by
     applying `qR` to the rational coordinates. Its output is still
     geometry-only (`promotable=0`, no SAT/LRAT proof, no chromatic claim)
     unless a separate candidate package attaches the no-5-colouring proof,
     edge sync to the reflected SAT module, `lean_real_final_theorem`, and
     offload review required below. The paired gate is
     `examples/erdos/test_chi6_rational_geometry_generator.sh`.
   - When a producer starts from exact rational coordinates rather than an
     already trusted edge list, use
     `examples/erdos/make_chi6_rational_unit_graph_source_package.py`: it
     derives the DIMACS edge list from all unordered coordinate pairs whose
     exact rational squared distance is `1`, then emits the same
     `chi6_solver_candidate_package.v1` source JSON consumed by the integrated
     preflight. Its coordinate IDs and split vertices are zero-based; the derived
     DIMACS edge file is one-based by DIMACS convention. The producer rejects
     sparse IDs, collapsed points, unsafe candidate IDs, weak/isolated split
     vertices, and coordinate sets below the requested exact unit-edge threshold.
     This reduces edge/coordinate drift, but remains a source package only
     (`promotable=0`, no SAT/LRAT proof, no chromatic claim). The paired gate is
     `examples/erdos/test_chi6_rational_unit_graph_source_package.sh`.
   - The first local CPU frontier scout is
     `examples/erdos/chi6_rational_frontier_scout.py`: it generates or ingests
     exact rational coordinate clouds, derives all unit edges exactly, runs a
     bounded non-certifying DSATUR 5-colourability probe for search pressure
     only, chooses high-degree zero-based split vertices, records any adjacency
     among those split vertices, and then calls the source-package producer
     above. Its scout sidecar is `solver_candidate_frontier_only` and remains
     non-promotable (`sat_claim=none`, `chromatic_claim=none`) until the
     downstream LRAT/Lean gates attach a checked no-5-colouring certificate. The
     paired gate is `examples/erdos/test_chi6_rational_frontier_scout.sh`.
   - Multi-frontier CPU campaigns use
     `examples/erdos/chi6_rational_frontier_campaign.py`: it fans out several
     scout runs, validates each source package through integrated preflight, and
     writes a ranked `chi6_rational_frontier_campaign.v1` manifest. Its
     `solver_heuristic_priority`, priority bonuses for non-certifying DSATUR
     statuses, and DSATUR pressure are search/provenance signals only, never
     chromatic evidence; DSATUR node count is capped inside the priority so
     sheer failed-search volume cannot dominate the ranking. The campaign may
     skip expected infeasible scout parameter rows, recording them in
     `failed_scouts` with no SAT/chromatic claim and `promotable=0`, while still
     carrying viable rows forward. A campaign result is
     eligible for promotion only after one concrete package reaches
     `geometry_status=PASS`, `sat_status=PASS`, and
     `integrated_status=READY_FOR_CANDIDATE_PROMOTION_WIRING`, then passes
     `examples/erdos/validate_chi6_promotable_candidate.sh` on a
     `promotable=1` manifest. The paired gate is
     `examples/erdos/test_chi6_rational_frontier_campaign.sh`.
   - Per-frontier cube campaign planning uses
     `examples/erdos/chi6_frontier_campaign_preflight.py`: it consumes one
     scout sidecar, revalidates the source package, emits the deterministic
     split-product cube batch, runs propagation-only cube classification, and
     records the exact next refutation/preflight commands. This is a workload
     planner, not a proof: propagation conflicts and hard-cube counts remain
     `verified_claim=none`, `global_unsat_claim=none`, and `promotable=0`
     until leaf LRAT plus cover LRAT/Lean replay are attached. Estimated
     `estimated_repo_colourCNF_base_clause_count` values cover only the repo
     `colourCNF` base shape, excluding standard at-most-one colour clauses, cube
     units, LRAT rows, and cover clauses; hard-cube samples are capped prefix
     diagnostics with explicit requested/count/truncated/exhaustive fields plus
     the sampled cube assignments, not exhaustive cover artifacts. The paired gate is
     `examples/erdos/test_chi6_frontier_campaign_preflight.sh`.
   - Campaign-to-preflight queueing uses
     `examples/erdos/chi6_frontier_campaign_preflight_batch.py`: it consumes one
     `chi6_rational_frontier_campaign.v1` manifest, runs bounded per-candidate
     cube preflights for the ranked prefix, and emits a
   `chi6_frontier_campaign_preflight_batch.v1` manifest with action counts,
   refute-ready candidates, and exact `cube_sieve_refute_batch.py` commands.
   This is still queue plumbing: it does not run SAT/LRAT refutation, and all
   rows remain `sat_claim=none`, `chromatic_claim=none`, `global_unsat_claim=none`,
   `verified_claim=none`, and `promotable=0`. The paired gate is
   `examples/erdos/test_chi6_frontier_campaign_preflight_batch.sh`.
  - Frontier refute attempts use
    `examples/erdos/chi6_frontier_refute_attempt.py`: it consumes the preflight
    batch manifest, executes selected machine-safe `refute_argv` rows without a
    shell, and records stdout/stderr hashes, return codes, parsed refuter
    counts, and a classification such as `REFUTE_SUCCESS_UNPROMOTABLE` or
    `REFUTE_NORESULT_MUTATE_FRONTIER`. This ledger may prove that individual
    cube leaves emitted CNF/DRAT/LRAT artifacts through the repo-local refuter,
    but it still does not prove a global obstruction: cover LRAT/Lean replay,
    exact geometry, and the Real-plane bridge remain separate promotion gates.
    Its manifest remains `sat_claim=none`, `chromatic_claim=none`,
    `global_unsat_claim=none`, `verified_claim=none`, and `promotable=0`. The
    paired gate is `examples/erdos/test_chi6_frontier_refute_attempt.sh`.
    The accepted machine argv is deliberately narrow: Python must invoke the
    canonical `examples/erdos/cube_sieve_refute_batch.py`; shell strings and
    alternate executables are rejected. A success classification requires the
    refuter stdout contract exactly: `status=subproblem_lrat_artifacts_emitted_unpromotable`,
    `failed_count=0`, `solver_unsat_count=cube_count`,
    `formal_proof_checker=none`, `verified_claim=none`,
    `global_unsat_claim=none`, and `promotable=0`; the ledger also checks that
    the refuter output directory contains at least the declared number of LRAT
    artifacts before accepting `REFUTE_SUCCESS_UNPROMOTABLE`.
  - Frontier-success SAT packaging uses
    `examples/erdos/make_chi6_frontier_refute_success_sat_manifest.py`: it
    consumes either a `chi6_frontier_refute_attempt.v1` manifest or a
    `chi6_frontier_refute_sweep.v1` manifest containing a
    `REFUTE_SUCCESS_UNPROMOTABLE` row, revalidates the selected ledger row,
    refuter stdout hash, per-cube CNF/DRAT/LRAT hashes, and originating
    preflight batch, then delegates to the arbitrary complement-cover route to
    emit a non-promotable `candidate.manifest`. This is the first packaging rung
    that may carry a checked finite SAT artifact for the selected edge/cube
    batch, but the boundary remains explicit: `chromatic_claim=none`,
    `geometry_claim=none`, `euclidean_claim=none`, and `promotable=0`. The
    paired gate is
    `examples/erdos/test_chi6_frontier_refute_success_sat_manifest.sh`.
  - Frontier-success promotion preflight uses
    `examples/erdos/make_chi6_frontier_refute_success_promotion_preflight.py`:
    it consumes the same successful attempt/sweep input, follows the selected
    row back through `campaign_preflight_json`, verifies candidate-source,
    edge, coordinate, cube-batch, and cover hashes, then runs
    `examples/erdos/make_chi6_integrated_candidate_preflight.sh` on the exact
    source package and selected cube batch. It is the first source-bound joiner
    between a successful search ledger row and the integrated exact-geometry +
    arbitrary cube-cover SAT preflight, but it still emits only pre-promotion
    lineage. `READY_FOR_CANDIDATE_PROMOTION_WIRING` is permitted only when
    `source_status=PASS`, `geometry_status=PASS`, and `sat_status=PASS` for the
    same graph identity. Otherwise the output remains incomplete, with
    `chromatic_claim=none` and `promotable=0`. The paired gate is
    `examples/erdos/test_chi6_frontier_refute_success_promotion_preflight.sh`.
  - Bounded frontier refute sweeps use
    `examples/erdos/chi6_frontier_refute_sweep.py`: it composes the existing
    rational campaign, campaign preflight batch, and refute-attempt ledger into
    one reproducible local search loop. The output records the campaign,
    preflight-batch, and refute-attempt manifests by path/hash, summarizes
    refute status counts, and emits the next search action
    (`mutate_or_expand_frontier`, `preflight_produced_no_refute_ready_cubes`, or
    `adjust_split_parameters_or_expand_frontier` for cells with no viable scouts,
    or a leaf-LRAT packaging action when a success appears). This remains a search ledger only:
    `sat_claim=none`, `chromatic_claim=none`, `global_unsat_claim=none`,
    `verified_claim=none`, and `promotable=0`. The paired gate is
    `examples/erdos/test_chi6_frontier_refute_sweep.sh`.
  - SAT-colouring-guided beam search uses
    `examples/erdos/chi6_colour_guided_beam.py` and
    `examples/erdos/chi6_colour_guided_beam_campaign.py`: it mutates exact
    rational frontiers using observed 5-colourings, scouts/preflights each child,
    and optionally runs bounded refute-ready leaves. When a child remains
    5-colourable, the beam tries to extend the parent colouring history over the
    child graph, appends the fresh scout colouring, deduplicates, and carries at
    most `max_carried_colourings` forward. This carried history is search
    pressure only, not chromatic evidence. The mutation stage may also require
    `min_neighbor_count` existing unit-neighbours and records selected
    existing-neighbour totals plus exact edge gain after mutation so sparse
    chain growth can be deprioritized. `chi6_colour_guided_density_probe.py`
    summarizes the same exact-rational candidate envelope across denominator
    budgets and neighbour-count thresholds before committing to a campaign. The
    optional edge-gain batch selector may also choose one or more bounded
    combinations of candidate points that create new-new unit edges, recording
    the requested batch count, combination offset/stride window, combination
    count, and whether the bounded search was truncated. Offset/stride windows
    let local and Slurm campaigns shard or sample the same deterministic
    combination order without treating any one window as exhaustive unless the
    non-truncation ledger says so. These density metrics and batch selectors are
    search pressure only, not a lower-bound claim.
    `chi6_colouring_sampler.py` may also enumerate a bounded deterministic set
    of symmetry-normalized proper 5-colourings for a fixed exact-rational graph
    so mutation can compare candidate points against more than one observed
    colouring. Its output is search instrumentation only:
    `claim_scope=bounded_colouring_sampling_only`, `sat_claim=none`,
    `chromatic_claim=none`, `global_unsat_claim=none`,
    `verified_claim=none`, and `promotable=0`. Beam and campaign manifests remain
    `sat_claim=none`, `chromatic_claim=none`, `global_unsat_claim=none`,
    `verified_claim=none`, and `promotable=0`; promotion still requires exact
    geometry plus checked SAT/LRAT/Lean artifacts. The paired gates are
    `examples/erdos/test_chi6_colouring_sampler.sh`,
    `examples/erdos/test_chi6_colour_guided_density_probe.sh`,
    `examples/erdos/test_chi6_colour_guided_beam.sh`,
    `examples/erdos/test_chi6_colour_guided_beam_campaign.sh`, and
    `examples/erdos/test_chi6_colour_guided_beam_slurm_job.sh`.
  - GPU-lane cube propagation must pass
    `examples/erdos/cube_sieve_gpu_parity.py` before its output is used by the
    chi>=6 search lane. The checker runs the deterministic CPU propagation
    producer and a backend producer with the same CLI, validates both manifests,
    and compares the canonical validated manifest text. A pass is only parity
    evidence: `verified_claim=none`, `geometry_claim=none`,
    `proof_artifact_sha256=NONE`, and `promotable=0`. The paired gate is
    `examples/erdos/test_cube_sieve_gpu_parity.sh`.
  - Darwin RTX 8000 search dispatch uses
    `examples/erdos/make_chi6_rtx8000_gpu_job.py`: it emits a Kubernetes job
    manifest pinned to the validated sm_75 RTX 8000 lane with OrangeFS scratch,
    requires an explicit cluster image built with the Sounio search payload,
    runs `chi6_frontier_refute_sweep.py`, and rejects node-local scratch or
    LLM-serving use by contract. The launcher is not a proof artifact; it keeps
    the same `backend_untrusted__drat_lrat_lean_verified_required` boundary and
    remains `promotable=0`. The paired gate is
    `examples/erdos/test_chi6_rtx8000_gpu_job_manifest.sh`.

3. Verified no-5-colouring certificate
   - Preferred plain path:
     - generate `(SounioSatColouring.colourCNF n 5 edges).Unsat`
     - use `examples/erdos/make_graph_reflect_certificate.sh` when the proof is
       deletion-free RUP-addition DRUP/DRAT.
   - Symmetry-break path:
     - generate `(SounioSatColouringSB.colourCNFsb5 a b c n edges).Unsat`
       (`SB_MODE=1 examples/erdos/make_graph_reflect_certificate.sh ...`)
     - prove graph-theoretically that `a,b,c` are a triangle in `edges`; because
       every listed edge has a `unit_edges` proof, this also proves the three
       embedded points are pairwise unit-distance.
      - no separate Euclidean non-collinearity proof is required for the SAT
        symmetry break: the graph-colouring argument only needs the three
        vertices to form a `K3` in the finite graph, so adjacent vertices must
        receive distinct colours. Exact geometry still proves the listed edges
        are unit edges and injectivity/irreflexivity prevents vertex collapse.
     - the generated candidate-owned Lean surface must expose the triangle
       membership proof used by the symmetry-break certificate, not just record
       `triangle_sb=a,b,c` in the manifest.
      - the proof must be over the same zero-based Lean edge list consumed by
        the SAT reflection module, so the DIMACS-to-Lean index conversion is not
        an informal side condition.
   - Large proofs with deletion records must go through `drat-trim -L` or a
     deletion-aware converter before Lean reflection.
   - The current promoted proof replay surface is LRAT-oriented. The historical
     field name `drat_or_lrat_path` may point at intermediate non-promotable
     smoke artifacts, but a `promotable=1` Lean path must provide proof bytes
     already replayable by the repository's LRAT reflection pipeline. Raw DRAT
     must be converted before promotion.

4. Lean package
   - Build a `UnitDistanceChromatic.NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit`.
   - Build a `UnitDistanceChromatic.EuclideanNatEdgeExactGeometry n P unit`
     tying `unit` to exact squared-distance geometry.
   - Expose the final obstruction through the Euclidean API:
     `EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract`
     or a direct Euclidean no-5 wrapper.
   - The candidate module must be `lake build` green under the repository
     `lean-toolchain`/`lakefile.lean` at `generator_commit`, and free of
     `sorry`/`admit`.

## Candidate Manifest Fields

Each candidate-shaped attempt should record these fields next to the artifacts.
`promotable=1` is reserved for a real exact-geometry candidate that has all
artifacts present and separately passes the Lean/offload gates. Smoke fixtures
and incomplete searches must use `promotable=0`.

```text
candidate_manifest_version=1
promotable=0 | 1
candidate_id=
n=
m=
k=5
edge_path=
edge_sha256=
cnf_path=
cnf_sha256=
drat_or_lrat_path=
drat_or_lrat_sha256=
cube_batch_path=
cube_batch_sha256=
cube_refutation_summary_path=
cube_refutation_summary_sha256=
cube_cover_certificate_path=
cube_cover_certificate_sha256=
cube_cover_complement_cnf_path=
cube_cover_complement_cnf_sha256=
cube_cover_complement_lrat_path=
cube_cover_complement_lrat_sha256=
lean_sat_module_path=
lean_sat_module_sha256=
geometry_module_path=
geometry_module_sha256=
geometry_source_path=
geometry_source_sha256=
geometry_proof_type=none | finite_smoke | euclidean
sat_proof_route=none | plain_lrat | triangle_sb5_lrat | cube_cover_split5 | cube_cover_generic
triangle_sb=none | a,b,c
generator_commit=
producer_command=
lean_build_command=
offload_review_raw=
offload_review_sha256=
lean_module=
lean_sat_edges_term=
lean_point_type=
lean_unit_term=
lean_geometry_term=
lean_edges_sync_term=
lean_no_five_witness_term=
lean_final_theorem=
lean_real_unit_term=
lean_real_emb_term=
lean_real_unit_edges_term=
lean_real_unit_iff_standard=
lean_real_final_theorem=
```

Path/hash pairs must be both concrete or both `NONE`. Hashes are SHA256 over
raw file bytes. `NONE` is allowed only for optional artifacts in `promotable=0`
manifests; the edge artifact is always required because it defines the graph
being discussed. `geometry_source_path` is optional for non-promotable packages
and should point at the exact coordinate/algebraic input used to generate a
geometry module when that source is available. `promotable=1` requires all
artifact path/hash pairs to be
concrete, including the Euclidean geometry module and the raw offload-review
record. The `sat_proof_route` field declares the SAT certificate surface:
`none` is allowed only for non-promotable incomplete manifests, `plain_lrat`
means a direct `colourCNF n 5` proof, `triangle_sb5_lrat` means a
triangle-precoloured `colourCNFsb5` proof, and `cube_cover_split5` means five
cube-unit leaf proofs composed by `SounioSatCubeCover.unsat_of_split_vertex5`.
`cube_cover_generic` means cube-augmented leaf proofs composed by
`SounioSatCubeCover.unsat_of_cube_cover` from an explicit Lean `CubeCover`
condition. In Lean, `CubeCover n k edges cubes` is the covering obligation:
every satisfying assignment of the base `colourCNF n k edges` satisfies at least
one cube's unit clauses. Cubes do not need to be pairwise disjoint; overlap is
allowed because the theorem needs coverage, not a partition. Generic cover
certificates can be supplied either by a structured Lean cover theorem such as
`split_vertices_cubes_cover`, or by a Lean-checked LRAT proof of the
complement-cover CNF `cubeCoverComplementCNF` via
`cube_cover_of_complement_unsat`. In generator terminology,
`--composition arbitrary` selects this complement-cover route: the generated
Lean module checks both the cube-augmented leaf LRATs and the complement-cover
LRAT, then obtains `CubeCover` from `cube_cover_of_complement_unsat`. For this
route, `cube_cover_certificate_path=NONE` is valid only when the manifest carries
concrete `cube_cover_complement_cnf` and `cube_cover_complement_lrat` path/hash
pairs; those complement artifacts are the cover certificate. A cube batch's
coverage is not inferred from vertex coverage or any producer heuristic: it is
trusted only when a structured Lean `CubeCover` theorem is supplied or when the
generated Lean module replays a complement LRAT against the repository's own
`cubeCoverComplementCNF n k edges cubes` term. Candidate cube batches must
contain at least one cube row; if the base `colourCNF` is proved
UNSAT without cubes, the manifest should use `plain_lrat` rather than a vacuous
cube-cover route.
The promotable join assembler recognizes `plain_lrat`, `triangle_sb5_lrat`,
`cube_cover_split5`, and `cube_cover_generic` as route-specific SAT surfaces.
For the triangle route, it parses the candidate SAT module's reflected
`colourCNFsb5 a b c n edges` declaration, checks that the manifest
`triangle_sb` and `n` fields match the Lean declaration, and exports the SB5
no-five witness through the same finite and standard-Real promotion gate used
by the other routes.
The current arbitrary-complement smoke gates are
`examples/erdos/test_cube_cover_arbitrary_complement_lean_reflect_pipeline.sh`
for the 25-leaf two-vertex K6 split cover and
`examples/erdos/test_cube_cover_arbitrary_complement_scale_pipeline.sh` for a
125-leaf three-vertex K6 cover. These are finite SAT calibration gates only:
they test the complement-LRAT cover machinery and generated Lean replay, not
search difficulty. The 125-leaf gate intentionally sends a product-shaped
fixture through the arbitrary route to exercise the generic proof surface, but
it is not Euclidean geometry evidence and not a chi(R^2) >= 6 witness.
`examples/erdos/test_cube_cover_arbitrary_complement_nonproduct_pipeline.sh`
adds a 130-leaf non-product-shaped smoke: five singleton cubes provide a simple
checked cover, and 125 redundant mixed cubes stress arbitrary membership
dispatch and reflected leaf replay.
`triangle_sb=a,b,c` uses zero-based vertex indices; the DIMACS edge file is
1-based, and the validator converts while checking that all three unordered
triangle edges appear in the edge file and that the generated Lean module uses
the matching `colourCNFsb5 a b c n` shape. The conversion is literal:
DIMACS vertex ids are parsed as 1-based, converted to zero-based ids by
subtracting 1, and then compared as unordered edges against `(a,b)`, `(b,c)`,
and `(c,a)`.

The `geometry_proof_type` field is a claim-boundary fuse:

- `none`: no geometry module is present.
- `finite_smoke`: a finite or non-Euclidean smoke module is present; it may
  exercise the Lean interface but is never promotable.
- `euclidean`: the module provides a
  `UnitDistanceChromatic.EuclideanNatEdgeExactGeometry` object, so the unit
  relation is tied in Lean to a two-coordinate squared-distance formula
  `(x1-x2)^2 + (y1-y2)^2 = 1` over an `ExactFieldLike` scalar with basic field
  laws and characteristic zero. Its `NatEdgeExactGeometry` component also
  requires injectivity of the finite embedding on `Fin n`, and its
  `ExactSquaredDistancePlane` component requires zero-distance equality, unit
  symmetry, and unit irreflexivity. `promotable=1` requires this value, concrete
  proof artifacts, a concrete hashed offload raw file for repository governance
  audit, and exposure of the Euclidean plug-in path such as
  `EuclideanNatEdgeExactGeometry.chi_ge_6_euclidean_plugin_contract` or one of
  its direct no-5-colouring wrappers.

If the candidate's internal coordinates are not literally `Real × Real`
coordinates, `promotable=1` still requires a candidate-owned Real
interpretation. For example, a rational or algebraic-field construction must
provide the Lean bridge that interprets its points in `Real × Real`, preserves
unit-distance edges, and exposes the final Real-plane obstruction through
`lean_real_unit_term`, `lean_real_unit_iff_standard`, and
`lean_real_final_theorem`. The promotable Lean gate then derives an additional
local theorem against the repository-standard `standardRealPlaneUnit` relation
from `lean_real_final_theorem` plus `lean_real_unit_iff_standard`; a custom
candidate-owned `realUnit` theorem alone is not enough. The `ExactFieldLike`
finite geometry package is not, by itself, a theorem about the standard
Euclidean plane.

For `promotable=1`, `generator_commit` must be a full 40-character hex commit
SHA in the repository, and `producer_command` must be concrete enough to rerun
from the repository root at that commit. Non-promotable development manifests
may use `generator_commit=DIRTY` or another explicit development marker, but
they are not archival evidence. `lean_build_command` is the exact local Lean
gate used for the candidate module. For promotable manifests this command must
be a simple `lake build ...` module list with no shell separators; any cluster
wrapper or orchestration script belongs outside the manifest. "No shell
separators" means no `;`, `&&`, `||`, pipes, command substitutions, or directory
changes inside the manifest field; multiple module arguments to the single
`lake build` invocation are allowed, and the command is run from `formal/lean4`.
`producer_command` should likewise be expanded enough for audit: record concrete
paths, seeds, cube files, and flags rather than relying on unrecorded
environment variables.

The validator rejects listed Lean SAT/geometry modules that contain `sorry` or
`admit`. A real candidate must also scan any candidate-owned helper modules it
imports; hiding incomplete proof in a helper is not promotion-eligible.
`validate_chi6_candidate_manifest.sh` is a manifest format/hash/route checker,
not a SAT-proof checker. It pins the declared DRAT/LRAT bytes by hash and checks
that the Lean surface matches the declared route. It also parses the `p edge`
header, counts edge rows, and rejects malformed, duplicate, out-of-range, or
self-loop edge rows. The trusted proof replay and Lean typecheck remain separate
acceptance gates.

## Acceptance Gate

Before promoting a real candidate, keep the local producer smoke honest. These
are infrastructure preflights: they prove the repository's K6 calibration,
format validators, LRAT reflection, and Euclidean geometry interfaces still
work. They are not themselves candidate-specific acceptance for a future
168-vertex or larger graph.

```bash
examples/erdos/test_cube_sieve_skeleton.sh
examples/erdos/test_cube_sieve_propagation_manifest.sh
examples/erdos/test_cube_sieve_batch_manifest.sh
examples/erdos/test_cube_sieve_gpu_parity.sh
examples/erdos/test_souc_sat_cube_units.sh
examples/erdos/test_cube_sieve_refute_batch.sh
examples/erdos/test_cube_split_batch.sh
examples/erdos/test_chi6_candidate_search_manifest.sh
examples/erdos/test_chi6_external_cube_cover_candidate_manifest.sh
examples/erdos/make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh
examples/erdos/test_chi6_external_arbitrary_cube_cover_candidate_manifest.sh
examples/erdos/test_chi6_foundry_handoff_package.sh
examples/erdos/test_chi6_solver_candidate_package_contract.sh
examples/erdos/test_chi6_integrated_candidate_preflight.sh
examples/erdos/test_chi6_rational_frontier_scout.sh
examples/erdos/test_chi6_rational_frontier_campaign.sh
examples/erdos/test_chi6_frontier_campaign_preflight.sh
examples/erdos/test_chi6_frontier_campaign_preflight_batch.sh
examples/erdos/test_chi6_frontier_refute_attempt.sh
examples/erdos/test_chi6_frontier_refute_success_sat_manifest.sh
examples/erdos/test_chi6_frontier_refute_success_promotion_preflight.sh
examples/erdos/test_chi6_frontier_refute_sweep.sh
examples/erdos/test_chi6_rtx8000_gpu_job_manifest.sh
examples/erdos/test_cube_cover_certificate.sh
examples/erdos/test_cube_cover_lean_composition.sh
examples/erdos/test_cube_cover_lean_reflect_pipeline.sh
examples/erdos/test_cube_cover_product_lean_reflect_pipeline.sh
examples/erdos/test_cube_cover_arbitrary_complement_lean_reflect_pipeline.sh
examples/erdos/test_chi6_cube_cover_smoke_candidate_manifest.sh
examples/erdos/test_chi6_cube_cover_temp_candidate_gate.sh
examples/erdos/test_chi6_euclidean_geometry_contract_gate.sh
examples/erdos/test_chi6_rational_geometry_generator.sh
examples/erdos/test_graph_reflect_certificate_sb.sh
examples/erdos/test_chi6_candidate_manifest_validator.sh
examples/erdos/test_chi6_promotable_candidate_gate.sh
examples/erdos/test_chi6_promotable_candidate_assembler_gate.sh
examples/erdos/make_chi6_smoke_candidate_manifest.sh
examples/erdos/test_chi6_temp_candidate_gate.sh
```

Adjacent baseline/vitrine sanity, not chi>=6 candidate acceptance:

```bash
examples/erdos/test_madore_q311_g529_3511_unified_vitrine_gate.sh
```

This builds `ErdosVitrine`, the permanent Lean packaging surface for the closed
Madore/Q311 `chi(R^2) >= 4` base case and the scoped current `{3,5,11}` G529
support/minimal-support surface. It also checks a separate chi>=6
interface-smoke boundary. It is deliberately not a chi>=6 lower-bound gate: the
finite no-five smoke and exact Euclidean square smoke are not connected into a
Euclidean no-five-colouring witness, and the real chi>=6 lower-bound
certificate remains absent.

This gate is only a search-lane preflight. It checks that the cube-sieve
skeleton compiles, emits the expected K6/k=5 propagation-trail manifest, and
keeps `promotable=0` while no checked proof artifact exists. The data-driven
producer gate reads a DIMACS `p edge` graph plus a zero-based cube file and emits
the same replayable propagation manifest shape for both the K6 conflict cube and
a non-K6 hard-cube smoke. The batch producer gate then runs the same producer
over a one-line-per-cube input file, writes one validated manifest per cube, and
emits a fail-closed summary with raw cube/manifest hashes and conflict/hard
counts for GPU-style fan-out. `examples/erdos/cube_split_batch.py` is the
canonical split-product batch front door: it reads a DIMACS graph and a list of
zero-based split vertices, emits every colour-product cube in the same batch
format consumed by the refuter and Lean cover generators, records raw hashes,
and fails closed above a configurable cube-count cap.
`examples/erdos/chi6_candidate_search_manifest.py` is the bounded finite-graph
search and packaging handoff. In enumeration mode it scans the declared
`all_simple_graphs(n)` slice under explicit edge/graph-count bounds and emits
either a bounded absence summary or non-promotable candidate edge/cube
artifacts. In `--edge-file` mode it packages an existing DIMACS graph, such as a
cluster-produced candidate or a known calibration graph, and deliberately makes
no non-colourability claim. The external package includes a
`*.meta.json` sidecar recording the source path, source SHA256, packaged edge
SHA256, `n`, `m`, `k`, split vertices, argv, and the provenance/promotion
boundary fields; the
manifest points at this sidecar through `source_meta_path` /
`source_meta_sha256`. The sidecar schema is
`examples/erdos/schemas/chi6_external_dimacs_edge_package.v1.schema.json`; its
`provenance_scope=edge_packaging_only` and
`promotion_gate=requires_lrat_lean_and_exact_euclidean_geometry` fields are
deliberate claim-boundary fuses. Both modes keep `geometry_claim=none` until a separate
exact Euclidean witness is supplied. This is a producer for the
SAT/cube-cover lane, not a mathematical promotion gate. Its absence status is
only `FINITE_GRAPH_CANDIDATE_ABSENT_WITHIN_BOUND` for that finite enumeration
slice; it is not evidence that a Euclidean chi>=6 witness does not exist. Its
candidate/package status is likewise an untrusted finite-graph handoff until the
downstream LRAT/Lean and geometry gates attach.
`examples/erdos/test_chi6_solver_candidate_package_contract.sh` adds the source
bundle one rung earlier than the manifest: it validates a hash-pinned
`chi6_solver_candidate_package.v1` JSON that binds the exact DIMACS graph,
exact rational coordinate CSV, coordinate domain, and split vertices. The paired
geometry manifest maker can consume this JSON directly, so smoke fixtures and
future solver outputs no longer need to pass edge and coordinate artifacts as
loose ad hoc CLI arguments. A source package is still only a candidate source;
it does not assert no-5-colourability or a Euclidean `chi >= 6` theorem.
`examples/erdos/test_chi6_integrated_candidate_preflight.sh` then runs one such
source bundle through both halves: rational exact geometry and the external
cube-cover SAT/LRAT bridge. It is a classifier, not a promotion gate. The square
fixture intentionally produces `geometry_status=PASS`, `sat_status=FAIL`, and
`first_blocker=sat_no5_cube_cover_refutation_absent`, proving that geometry
success alone cannot be mistaken for a chromatic lower bound. A real source
bundle must reach `geometry_status=PASS` and `sat_status=PASS` before any
candidate-owned promotable Lean module should be assembled. The same preflight
also accepts the GPU/search-shaped input
`<candidate-source.json> <cube-batch> <cover-drup-or-rup>`; in that mode it
runs the arbitrary complement-cover bridge rather than the split-product cube
bridge and reports `sat_route_mode=arbitrary_complement_cube_cover`. This is
the intended handoff shape for cube-and-conquer output from an untrusted solver:
the cube batch and cover proof can make the SAT half pass only after LRAT/Lean
replay succeeds, while the source package still supplies the exact rational
geometry half.
`examples/erdos/make_chi6_external_cube_cover_candidate_manifest.sh` is the
next non-promotable bridge: it takes such an external DIMACS package, runs the
split-product cubes through the per-cube LRAT refuter, checks a product cover
certificate, generates the generic Lean SAT module, and emits a
`candidate.manifest` with `sat_proof_route=cube_cover_generic`,
`source_meta_path`, and `source_meta_sha256`; the manifest validator checks that
the sidecar describes the same candidate id, `n`, `m`, `k`, packaged edge hash,
provenance scope, and promotion gate as the manifest. The paired gate uses K6
only as a calibration graph for this external package path; it does not create a
Euclidean candidate or a colouring lower-bound claim. The `souc_sat` cube-unit
gate then checks the next
proof-producing rung: `./souc_sat.elf <seed> <k> <use_lrb> <sb_mode> <edgefile>
[cubefile]` appends each zero-based cube assignment as an original unit clause
before `n_orig`, emits a CNF for exactly `CNF(edge,k) /\ cube_units`, and emits a
deletion-free DRAT smoke that the repo-local RUP-to-LRAT converter can replay to
the empty clause. The batch refutation gate compiles `souc_sat` once, runs every
cube in an isolated artifact directory, records per-cube CNF/DRAT/LRAT hashes,
and rejects satisfiable cube subproblems. This still does not prove global UNSAT:
promotion needs a checked cube-cover certificate tying the batch cubes back to the
whole colouring search space, plus Lean-checked LRAT and Euclidean geometry. The
cover-certificate smoke checks the smallest finite composition boundary: for
K6/k=5, the five one-literal cubes `0:0` through `0:4` cover the split-vertex
cases of the base `colourCNF` search space by the at-least-one colour clause for
vertex 0. It verifies that the referenced per-leaf LRAT artifact files match the
refutation-batch hashes and that each LRAT artifact contains an empty-clause row,
but still emits `verified_claim=none` and
`global_unsat_claim=none` until Lean-checked leaf UNSAT facts are attached to a
candidate-owned generated module. The Lean composition gate builds the reusable
`SounioSatCubeCover.unsat_of_split_vertex5` adapter, and the K6/k=5 reflection
smoke proves the end-to-end finite SAT shape: five cube-unit LRAT leaves are
checked by Lean core and composed back to the plain `colourCNF`. That remains a
K6 calibration artifact with `geometry_claim=none`, not evidence for a Euclidean
`chi >= 6` theorem.
The cube-cover smoke is also wrapped as a non-promotable `candidate.manifest`
with `sat_proof_route=cube_cover_split5` and, in a parallel opt-in smoke,
`sat_proof_route=cube_cover_generic`; both use `triangle_sb=none`, the generated
Lean SAT module, and hashed cube batch/refutation/cover artifacts. This is the
manifest shape a real cube search will replace, while still leaving
`geometry_module_path=NONE` and `promotable=0`. The legacy triangle-precolour
smoke uses `sat_proof_route=triangle_sb5_lrat`.
`examples/erdos/test_chi6_cube_cover_temp_candidate_gate.sh` separately checks
the candidate-owned attachment surface: a generated arbitrary-complement SAT
module is joined to a finite non-Euclidean `NatEdgeExactGeometry` whose edge list
is definitionally the generated SAT edge list and whose temporary unit relation
is the generated undirected edge relation, then routed through
`NatEdgeExactGeometry.noFiveWitnessOfCubeCoverUnsat`. This is an interface
calibration gate only; a promotable candidate still needs honest Euclidean
geometry and the Real-plane bridge fields above.
The current `cube_cover_split5` route is deliberately narrow: it composes the
five colour cases for one split vertex through
`SounioSatCubeCover.unsat_of_split_vertex5`. The Lean core also exposes the
generic theorem `SounioSatCubeCover.unsat_of_cube_cover`, plus witness adapters
`noFiveWitnessOfCubeCoverUnsat` / `noFivePlaneColouringOfCubeCoverUnsat`, for a
finite list of arbitrary cubes with a checked `CubeCover` condition. Large
cube-and-conquer searches with multi-vertex or overlapping cubes must use
`cube_cover_generic` and supply a Lean `CubeCover` proof/checker; they must not
be squeezed through the single-split route. The current generic smoke still uses
the K6 split cubes and `split_vertex5_cubes_cover` only to prove the route
surface is wired end-to-end.
The next checked cover family is a split-product cover:
`SounioSatCubeCover.splitVerticesCubes k vs` enumerates all positive colour
choices for a finite vertex list `vs`, and
`SounioSatCubeCover.split_vertices_cubes_cover` proves it is a real `CubeCover`
from the at-least-one clauses. The product-reflection smoke exercises this with
K6/k=5 and `vs=[0,1]`, producing 25 Lean-checked LRAT leaves composed through
`unsat_of_cube_cover`. For arbitrary cube families, the complement-cover route
emits `base ∧ ⋀ cube, block(cube)` as DIMACS, checks an LRAT refutation of that
CNF in Lean, converts it to `CubeCover` through
`cube_cover_of_complement_unsat`, and then composes the per-cube LRAT leaves
through `unsat_of_cube_cover`. The calibration gate exercises this on the K6
two-vertex cube family without using the split-product theorem; a real search
can replace the cube list and complement LRAT while keeping the same trusted
Lean surface. `make_chi6_external_arbitrary_cube_cover_candidate_manifest.sh`
is the reusable bridge for an external edge file plus arbitrary cube batch plus
complement DRUP/RUP proof; it emits a non-promotable `candidate.manifest` with
concrete `cube_cover_complement_cnf` and `cube_cover_complement_lrat` artifacts.
`test_chi6_external_arbitrary_cube_cover_candidate_manifest.sh` calibrates that
bridge on K6 and proves the manifest validator follows the generic
complement-cover branch instead of the older split-cover certificate branch.
The emitted complement DIMACS is producer input, not trusted evidence by itself:
in the generated module, the LRAT is replayed against Lean's own
`cubeCoverComplementCNF n k edges cubes` term, so a stale or buggy DIMACS
emitter cannot by itself establish `CubeCover`. In the same non-submitting
spirit, `make_chi6_foundry_handoff_package.sh` accepts an already-produced
`candidate.manifest`, revalidates it, copies the manifest and directly
referenced hash-checked artifacts into `chi6-package/`, emits `SHA256SUMS`, and
writes a Foundry/Slurm handoff note. The standalone
`validate_chi6_foundry_handoff_package.sh` checker verifies the package hashes,
re-runs the candidate manifest validator on the copied manifest, and compares
the handoff's candidate/route/hash fields against the manifest. This is a
transport and replay package only; it records host replay commands and the
proposed Foundry target shape, but it does not submit jobs or turn a
non-promotable package into a Euclidean theorem.
The exact Lean composition surface is:

```lean
def SounioSatCubeCover.CubeCover
    (n k : Nat) (edges : List (Nat × Nat)) (cubes : List Cube) : Prop :=
  ∀ a : Nat → Bool,
    CNF.eval a (colourCNF n k edges) = true →
      ∃ cube, cube ∈ cubes ∧ CNF.eval a (cubeCNF cube) = true

theorem SounioSatCubeCover.unsat_of_cube_cover
    {n k : Nat} {edges : List (Nat × Nat)} {cubes : List Cube}
    (hcover : CubeCover n k edges cubes)
    (hunsat : ∀ cube, cube ∈ cubes →
      (colourCNFWithCube n k edges cube).Unsat) :
    (colourCNF n k edges).Unsat

theorem SounioSatCubeCover.split_vertices_cubes_cover
    {n k : Nat} {edges : List (Nat × Nat)} {vs : List Nat}
    (hverts : ∀ v, v ∈ vs → v < n) :
    CubeCover n k edges (splitVerticesCubes k vs)

def SounioSatCubeCover.cubeCoverComplementCNF
    (n k : Nat) (edges : List (Nat × Nat)) (cubes : List Cube) : CNF Nat

theorem SounioSatCubeCover.cube_cover_of_complement_unsat
    {n k : Nat} {edges : List (Nat × Nat)} {cubes : List Cube}
    (hcomp : (cubeCoverComplementCNF n k edges cubes).Unsat) :
    CubeCover n k edges cubes
```

This is why overlapping cubes are sound: the theorem consumes only coverage
(`hcover`) and per-cube UNSAT leaves (`hunsat`), never a disjointness or
partition hypothesis.
The formal witness API also exposes the same route directly:
`NatEdgeUnitDistanceCertificate.noFiveWitnessOfSplitVertex5Unsat`,
`NatEdgeExactGeometry.noFiveWitnessOfSplitVertex5Unsat`, and
`EuclideanNatEdgeExactGeometry.noFivePlaneColouringOfSplitVertex5Unsat` turn five
Lean-checked cube leaves for one split vertex into the same no-5 witness object
used by the plain-CNF and SB5 routes.
These outputs are replayed by a producer-side validator that parses the declared
`graph_family`, `n`, `m`, `k`, edge rows, and `cube_assignment` rows, then checks
the trail steps, DIMACS literals, RUP-style reason clauses, conflict/hard-cube
summary, and final domains against those declared parameters; this still is not
a SAT proof or geometry proof. The preflight includes a negative mutation from
`k=5` to `k=6` to prove the parser is not decorative, plus fail-closed mutations
for cube count, duplicate cube colours, bad edge rows, wrong hard-cube summary,
duplicate batch cube ids, and malformed cube tokens. The preflight also checks
that the reflected-certificate producer can emit the triangle-precoloured
`colourCNFsb5` Lean shape needed by the no-5-colouring path, that manifests are
fail-closed on hashes and missing promotion artifacts, that a concrete
non-promotable `candidate.manifest` can be produced from the K6/SB5 smoke
pipeline and from the cube-cover composition pipeline, and that the public
generic no-5 obstruction shape is exercised on a temp non-Euclidean smoke.
The temp smoke is also checked to remain non-promotable under the stricter
Euclidean geometry requirement. It is not evidence for a Euclidean `chi >= 6`
theorem, and it does not build a Euclidean geometry module. It only exercises
the SAT/certificate and manifest-shape side of the future promotion path.
The Euclidean geometry smoke is the complementary geometry-only gate: it builds
exact `EuclideanNatEdgeExactGeometry` objects for a rational unit segment and a
four-edge rational unit square, proving the squared-distance contract is
executable before any candidate SAT certificate is attached.
`examples/erdos/test_chi6_rational_geometry_generator.sh` lifts that from a
handwritten smoke to a data-driven bridge: a DIMACS graph plus exact rational
CSV coordinates generates an importable Lean module, proves its listed edges
are exact unit edges over `Rat^2`, exposes `edgesSyncSelf`, exposes
`chi6rat_real_unit`, `chi6rat_real_unit_iff_standard`, and
`chi6rat_real_unit_edges` for the repository's `Real × Real` plane, and rejects
a negative coordinate mutation whose listed edge has squared distance `4`
instead of `1`. This still has `sat_claim=none`, `chromatic_claim=none`, and
`promotable=0`; it is the geometry producer rung a real solver package must
combine with the LRAT/cube-cover route and a final Real-plane no-5 theorem.

Manifest format validation is executable:

```bash
examples/erdos/validate_chi6_candidate_manifest.sh candidate.manifest
examples/erdos/validate_chi6_promotable_candidate.sh candidate.manifest
```

Passing the validator is bookkeeping only. It does not verify LRAT, build Lean,
or prove exact geometry. It prevents common promotion mistakes: missing hashes,
bad triangle metadata, smoke geometry marked as promotable, a promotable module
that lacks `EuclideanNatEdgeExactGeometry` and `ExactFieldLike`, and listed Lean
modules containing `sorry`/`admit`.

The promotable verifier is stronger but still local: it calls the manifest
validator, imports `lean_module`, and asks Lean to type-check the named geometry,
no-five witness, and final theorem at the manifest's concrete `n`, point type,
and unit relation. It rejects marker-only files that can pass textual `rg`
checks but cannot inhabit the required Lean types.

For a candidate named `Chi6CandidateFoo`, the minimum gate is:

```bash
examples/erdos/test_drup_to_lrat_rup.sh
examples/erdos/make_graph_reflect_certificate.sh graph.edge 5 formal/lean4/SounioSatFooReflect.lean foo SounioSatFooReflect /tmp/foo-cert
examples/erdos/validate_chi6_candidate_manifest.sh /tmp/foo-cert/candidate.manifest
cd formal/lean4
lake build SounioSatFooReflect Chi6CandidateFoo
cd ../..
rg -n '\b(sorry|admit)\b' formal/lean4
bin/llm-offload -t math-review -p xai -i formal/lean4/Chi6CandidateFoo.lean
```

The `rg` command must return no matches. The final theorem should be a theorem
over the exact planar point type, not over a smoke relation. A candidate-owned
helper set is not enough for final promotion: the whole Lean 4 proof surface
used by the build must be clean of `sorry`/`admit`. The listed-module textual
scan is only a preflight; a real promotable candidate must also audit
candidate-owned transitive imports or print/check the final theorem axioms so a
helper module cannot hide proof holes or candidate-owned `axiom`/`constant`
bridges.

The `bin/llm-offload` line is a repository governance review, not part of the
trusted formal kernel. It can block publication by policy, but it cannot promote
a candidate mathematically. Its raw output is recorded and hashed so the repo
policy decision is auditable; the mathematical gate remains the Lean 4 build of
exact geometry plus checked SAT certificate. Third-party theorem reproduction
depends on the Lean artifacts, the repository commit, and the raw-byte artifact
hashes, not on the offload tool. The mathematical trust boundary remains:
exact Lean Euclidean squared-distance geometry plus Lean-checked SAT/LRAT
obstruction. The validator only checks artifact shape; the `lake build` of
`EuclideanNatEdgeExactGeometry` is what checks the scalar-law bundle and the
squared-distance formula/injectivity/sanity proofs. The Lean SAT checker and
generated reflected module are pinned by `generator_commit` plus
`lean_sat_module_sha256`.
If an offload review reports BLOCKER/MAJOR issues, the candidate package must
either fix them or record an explicit waiver/rationale in
`.claude/llm_offload_log.md`; the hash field records the review evidence, not a
formal proof.
For a real candidate, the Lean edge list used by the SAT module and the edge
list used by the geometry module must be proved or generated from the same
ordered DIMACS artifact. Isomorphic edge sets are not enough for audit: the
finite graph consumed by the SAT obstruction and the finite graph embedded as
unit distances must be byte/term synchronized by the candidate package.

For `promotable=1`, the Lean term fields are mandatory:

- `lean_module`: the Lean module to import for the candidate-owned theorem
  surface.
- `lean_sat_edges_term`: the fully qualified generated SAT edge list consumed by
  the reflected SAT proof.
- `lean_point_type`: the exact point type used by the Euclidean graph.
- `lean_unit_term`: the unit-distance relation for that point type.
- `lean_geometry_term`: the fully qualified candidate-owned
  `EuclideanNatEdgeExactGeometry n P unit` object.
- `lean_edges_sync_term`: the fully qualified candidate-owned proof that
  `lean_geometry_term.exact.edges = lean_sat_edges_term`. This is an edge-drift
  fuse between the reflected SAT object and the geometry object; it does not
  replace the `EuclideanNatEdgeExactGeometry` unit-edge and Real-plane bridge
  obligations.
- `lean_no_five_witness_term`: the fully qualified candidate-owned
  `NatEdgeUnitDistanceCertificate.NoFiveColourWitness n P unit` object.
- `lean_final_theorem`: the fully qualified candidate-owned theorem exposing
  the Euclidean no-5-colouring obstruction through the plugin contract or direct
  Euclidean wrapper.
- `lean_real_unit_term`: the fully qualified unit-distance relation on
  `Real × Real` used for the public plane theorem.
- `lean_real_emb_term` and `lean_real_unit_edges_term`: explicit bridge terms
  used by generated promotion joins to map Nat vertices into `Real × Real` and
  prove the listed edges satisfy `lean_real_unit_term`. The promotable join
  assembler requires these fields rather than inferring them from naming
  conventions.
- `lean_real_unit_iff_standard`: the fully qualified theorem proving that
  `lean_real_unit_term p q` is equivalent to the repository's standard
  squared-distance formula over the repository's
  `SounioSqrt.RealCauchyField.Real` interface, exported as
  `SounioSqrt.RealCauchyField.standardRealPlaneDist2` by
  `SounioRealPlaneGeometry`.
- `lean_real_final_theorem`: the fully qualified candidate-owned theorem
  exposing the public standard-plane obstruction,
  `¬ Nonempty (PlaneColouring (Real × Real) lean_real_unit_term 5)`.

The manifest validator checks that those names are syntactically valid Lean
names and that matching `def`/`theorem`/`abbrev` declarations for the unit,
SAT edge list, geometry, edge-list synchronization proof, no-five witness, and
final theorem live in the listed SAT/geometry modules.
It also rejects candidate-owned top-level `axiom`, `constant`, or `opaque`
declarations in the listed SAT/geometry modules. This textual check is scoped to
candidate-owned files; trusted repository imports must be governed by the
separate axiom-dependency gate. This is still not a substitute for `lake build`
and `#print axioms`; it is a fail-closed preflight against fake promotion by
textual marker strings alone. The validator also requires the promotable
geometry module to expose the witness adapter matching
`sat_proof_route`: plain `colourCNF`, triangle-precoloured `colourCNFsb5`, or
five-leaf cube-cover `noFiveWitnessOfSplitVertex5Unsat` /
`noFivePlaneColouringOfSplitVertex5Unsat`, or generic cube-cover
`noFiveWitnessOfCubeCoverUnsat` / `noFivePlaneColouringOfCubeCoverUnsat`. This
keeps the manifest's declared SAT route synchronized with the candidate-owned
Lean theorem surface.

The promotable verifier gate then generates a temporary Lean module importing
the declared SAT module and `lean_module`, then checks the declared terms at
these concrete types:

```lean
EuclideanNatEdgeExactGeometry n lean_point_type lean_unit_term
lean_sat_edges_term : List (Nat × Nat)
lean_edges_sync_term :
  lean_geometry_term.exact.edges = lean_sat_edges_term
NatEdgeUnitDistanceCertificate.NoFiveColourWitness n lean_point_type lean_unit_term
¬ Nonempty (PlaneColouring lean_point_type lean_unit_term 5)
lean_real_unit_term : Real × Real → Real × Real → Prop
∀ p q : Real × Real,
  lean_real_unit_term p q ↔
    standardRealPlaneDist2 p q = qR (1 : Rat)
¬ Nonempty (PlaneColouring (Real × Real) lean_real_unit_term 5)
```

This is a Lean type check against the exported standard squared-distance formula
using the repository's `addR`/`negR`/`mulR`/`qR` Real interface; it is not a
textual string match on mathematical prose. Candidate packages must prove
equivalence to `standardRealPlaneDist2`, not to a verifier-local duplicate. The
current K6/cube-cover smokes are non-promotable and do not exercise this public
Real-plane gate.

The promotion assembler rung is executable but intentionally fail-closed:
`examples/erdos/make_chi6_promotable_candidate_manifest.sh` consumes the output
of `examples/erdos/make_chi6_integrated_candidate_preflight.sh` and refuses
unless `geometry_status=PASS`, `sat_status=PASS`, and
`integrated_status=READY_FOR_CANDIDATE_PROMOTION_WIRING`. When that gate is
eventually reached by a real solver package, it uses
`examples/erdos/gen_lean_chi6_promotable_candidate.py` to generate the
candidate-owned join module exposing edge sync, the route-specific no-five
witness, finite no-five theorem, Real unit relation, and Real no-five theorem.
The paired gate `examples/erdos/test_chi6_promotable_candidate_assembler_gate.sh`
uses the rational unit square to prove that geometry success with SAT failure is
rejected, that unrelated SAT/geometry manifests cannot be joined, that missing
Real bridge terms are fatal, and that the SB5 route rejects manifest/Lean
triangle mismatches.

For `promotable=1`, the verifier must print the axiom dependencies of the
declared final terms and reject
`sorryAx` plus any unexpected axiom outside the small accepted kernel/native
surface. `propext`, `Quot.sound`, and Lean's `native_decide` checker axioms are
expected on the reflected SAT side; `Classical.choice` is tolerated only where
the existing finite SAT/cube-cover adapters already introduce it, not as a
license for candidate-owned geometry existence shortcuts. This closes the
helper-module loophole where a listed geometry module itself contains no
`axiom`, but imports a candidate-owned axiom-backed proof.

This type check is the local candidate obstruction gate. A public
`chi(R^2) >= 6` theorem additionally requires the declared point type and
unit relation to be tied to the standard Euclidean plane, either concretely as a
Real/rational-coordinate plane interpreted in `Real^2`, or through an explicit
candidate-owned Lean bridge theorem. That bridge is represented in the manifest
by `lean_real_unit_term`, `lean_real_unit_iff_standard`, and
`lean_real_final_theorem`; without those standard-plane terms, the manifest is
at most a finite or abstract-model obstruction. Abstract finite or
non-Archimedean models remain `finite_smoke`, even when they satisfy the finite
obstruction shape.

## Non-Promotable Artifacts

These are useful scaffolds, but must not be claimed as a Euclidean plane
`chi >= 6` witness:

- `SounioFiniteUnitDistanceWitnessSmoke.lean`: finite K6/no-5 smoke over a
  complete finite relation, not a planar unit-distance embedding.
- `SounioFiniteUnitDistanceEuclideanSmoke.lean`: exact rational unit-segment
  and four-edge unit-square geometry smoke for the Euclidean contract, not a
  no-5-colouring obstruction.
- `SounioSatK65Reflect.lean`: generated finite K6/no-5 SAT certificate smoke.
- `examples/erdos/cube_sieve_skeleton.sio`: executable manifest/proof-smoke
  skeleton; it prints `promotable=0` until a proof artifact exists.
- `examples/erdos/168_orbit_chi6_proof.sio`: R16/ZD orbit computation with
  executable evidence, not an exact Lean proof of a planar unit-distance graph.

## Soundness Boundary

The cluster/GPU lane may discover candidates and produce SAT artifacts. The
trusted mathematical boundary is the Lean 4 build of exact Euclidean geometry
plus the Lean-checked SAT certificate. The graph only needs to be a finite
unit-distance subgraph, but the theorem must be routed by the contrapositive:
any 5-colouring of the ambient Euclidean unit-distance plane would restrict to a
5-colouring of the embedded finite graph, so a Lean-checked finite no-5
obstruction rules out such an ambient colouring. No GPU result, solver status
line, LLM review, or generated manifest is a theorem until it plugs into the
Lean witness shape above. For the Euclidean-plane claim, the witness shape must
model the standard two-coordinate squared-distance relation over `Real` (or an
exact rational/subfield coordinate representation with an explicit Real
interpretation); an abstract `ExactFieldLike` instance alone is not enough to
promote a `chi(R^2)` result.
