<!-- docs:meta
topic_id: repo.docs.research.solver-proofchecker-sota-2026-06-22
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.solver-proofchecker-sota-2026-06-22
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Solver + proof-checker SOTA map for Sounio

This note is a current, implementation-facing map for upgrading Sounio's SAT,
SMT, proof-checking, and high-precision dynamical-systems stack. It is not a
performance claim. Every item below still needs local gates before publication.

## External calibration checked on 2026-06-22

- SAT Competition 2024 emphasizes the same direction Sounio already needs:
  stronger solver engineering, refined implementation techniques, and public
  benchmark/proof-checker comparison. Source:
  <https://satcompetition.github.io/2024/>.
- Current SAT SOTA is still CaDiCaL/Kissat-class CDCL with heavy inprocessing,
  incremental interfaces, proof generation, and careful data layout. CaDiCaL
  2.0 is explicitly positioned around a clean library interface, testing, proof
  generation, interpolation, and advanced use cases. Source:
  <https://link.springer.com/chapter/10.1007/978-3-031-65627-9_7>.
- RustSAT's 2025 tool paper is useful as an interface lesson: separate the core
  solver API from encodings, incremental solving, assumptions, cores, phase
  control, learned-clause callbacks, stats, and external-solver adapters. Source:
  <https://arxiv.org/html/2505.15221v1>.
- For proof checking, the important direction is not raw DRAT forever. LRAT
  adds explicit hints that make efficient formally verified checkers practical;
  CLRAT/FRAT-style variants are the natural next interchange formats for large
  Sounio certificates. Source:
  <https://jix.github.io/varisat/manual/0.2.0/formats/lrat-proofs.html>.
- FRAT is especially relevant because it is designed as solver-elaborator
  communication: the solver can emit extra information that reduces the cost of
  elaboration to LRAT, while the independently checkable backend remains small.
  The published FRAT toolchain reports large median reductions in elaboration
  time and peak memory over a comparable DRAT elaboration path. Source:
  <https://www.cs.cmu.edu/~mheule/publications/FRAT-TACAS.pdf>.
- The 2025 SAT Competition checker/output material keeps that proof-checker
  split visible in practice: LRAT/LPR-style verified checking, CakePB/VeriPB
  documentation, and proof-producing solver tracks are treated as competition
  infrastructure rather than an academic side channel. Source:
  <https://satcompetition.github.io/2025/output.html>.
- VeriPB generalizes the certificate story beyond clauses to pseudo-Boolean
  cutting-planes reasoning, including sophisticated combinatorial techniques.
  It is the right model for a future Sounio PB/Cardinality/MaxSAT proof layer.
  Source: <https://veripb.org/>.
- The 2025 PB optimization line uses a two-stage proof architecture: an
  elaborator converts rich proofs to a restrictive backend format, then a
  formally verified checker validates the backend proof. This is a good shape
  for Sounio: aggressive producer, tiny checker. Source:
  <https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/html/LIPIcs.CP.2025.21/LIPIcs.CP.2025.21.html>.
- The 2026 competition line keeps that direction live: VeriPB/CakePB remains an
  active proof-checking path for SAT/PB certificates, and the PB 2026 material
  notes a new VeriPB format generation with backwards-compatible checkers.
  Sources: <https://satcompetition.github.io/2026/downloads/checkers/veripb.pdf>
  and <https://www.cril.univ-artois.fr/PB26/>.
- PBLean is the newest theorem-prover-facing pressure point for this lane: it
  imports VeriPB pseudo-Boolean kernel certificates into Lean 4 through a proved
  reflective checker, so large solver certificates become composable Lean
  theorems instead of external verdicts only. Source:
  <https://arxiv.org/html/2602.08692v2>.
- The current PB proof-checking pressure point is not just parsing a format:
  VeriPB/CakePB-style pipelines need efficient checking of cutting-planes
  derivations, strengthening/convenience rules, and propagation traces. That
  motivates tiny kernel receipts inside Sounio before a full text parser exists.
  Sources: <https://veripb.org/> and
  <https://drops.dagstuhl.de/storage/00lipics/lipics-vol340-cp2025/html/LIPIcs.CP.2025.21/LIPIcs.CP.2025.21.html>.
- For SMT proofs, the comparable direction is Alethe/LFSC-style proof objects
  and independent reconstruction/checking. cvc5 documents Alethe as an
  SMT-LIB-based proof format with coarse and fine steps, and Carcara/LFSC-style
  checkers make the external proof object independently replayable. Sources:
  <https://cvc5.github.io/docs/cvc5-1.0.0/proofs/output_alethe.html> and
  <https://github.com/cvc5/LFSC>.
- The Alethe/Carcara line also reinforces a producer/elaborator/checker split:
  Alethe proof traces are SMT-LIB-like sequences of assumptions and inferred
  clauses ending in the empty clause, while Carcara-style tooling checks or
  elaborates coarse steps into more reconstructable fine steps. Sources:
  <https://verit.gitlabpages.uliege.be/alethe/specification.pdf> and
  <https://github.com/ufmg-smite/carcara>.
- Lean-side SMT integration is moving the same way: `lean-smt` uses SMT-LIB and
  solver proofs to discharge Lean goals, while the Isabelle/cvc5 Alethe work
  shows the independent-reconstruction path for SMT-LIB benchmarks. Sources:
  <https://arxiv.org/html/2505.15796v1> and
  <https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2025.26>.
- For high-precision Lorenz and chaotic systems, the relevant verified-numerics
  north star is interval/ball arithmetic and Taylor-model style validated
  integration, not just more f64 digits. Arb's midpoint-radius ball arithmetic
  tracks errors automatically at arbitrary precision. Source:
  <https://arblib.org/>.
- The current implementation pressure from Arb/FLINT-style arithmetic is that
  the certificate should carry midpoint/radius/error status, not only a rounded
  point. FLINT documents ball arithmetic as attaching an error bound to each
  variable so rigorous computation does not depend on manual post-hoc error
  analysis. Source: <https://flintlib.org/doc/overview.html>.
- Taylor-model and flowpipe tools reinforce the same split: a producer may use
  high-order polynomial/local flowpipe machinery, but the checker-facing object
  needs explicit order, step size, remainder/radius budget, and dependency
  links. Flow* is the classic Taylor-model flowpipe reference point; recent
  proof-assistant work formalizes Taylor/power-series ODE solving with
  extractable certified exact-real programs. Sources:
  <https://plv.colorado.edu/papers/flowstar-cav13.pdf> and
  <https://drops.dagstuhl.de/storage/00lipics/lipics-vol309-itp2024/LIPIcs.ITP.2024.30/LIPIcs.ITP.2024.30.pdf>.
- The Lorenz-specific lesson from Tucker/CAPD-style rigorous numerics is that a
  high-precision point orbit is not enough: the proof object must connect the
  point computation to interval enclosures, directed rounding, and an invariant
  or return-map argument before it becomes a dynamical theorem. Sounio's current
  practical next step is therefore inclusion certificates that relate wide
  fixed-point trajectories to validated balls. Sources:
  <https://www.semanticscholar.org/paper/The-Lorenz-Attractor-Exists-%E2%80%93-An-Auto-Validated-Tucker/27f27aa1ecc85515f7c0eb4120e92b84b97f3ae6>
  and <https://ww2.ii.uj.edu.pl/~kapela/papers/capd-review.pdf>.

## Sounio state before this note

- `stdlib/theorem/smt.sio` is a bounded DPLL(T) core with propositional CNF,
  bounded LIA via Fourier-Motzkin, and epistemic solver heuristics. It is the
  small integrated SMT engine.
- `stdlib/theorem/qflra_exact.sio` is the exact rational QF_LRA layer. It already
  produces Farkas multipliers for UNSAT rows, but callers historically had to
  re-check the certificate manually.
- `examples/erdos/souc_sat.sio` is the scale solver for graph-colouring UNSAT:
  CDCL, DRAT emission, symmetry breaking, and de Grey/G529-style workflows.
- `stdlib/systems/lib.sio` currently integrates Lorenz/Rossler/Van der Pol with
  f64 forward Euler. It is useful as a demo, but it is not a validated-numerics
  engine.
- The current branch has experimental wide integer work (`i128`, `i256`, etc.)
  in the checker/native pipeline and a `madaros-wide-int-gate`. Treat it as a
  compiler capability lane, not yet as a stable stdlib numeric abstraction.

## Erdős inventory from the repo pass

- Hadwiger-Nelson is tracked in the repo as Erdős #508, not #90. The strongest
  current Sounio colourability artifact is the de Grey/Heule G529 pipeline:
  exact unit-distance geometry plus a 4-colouring UNSAT certificate for the
  529-vertex core, with the SAT leg reflected into Lean via LRAT. This supports
  the known `chi >= 5` line; it is not a `chi >= 6` result.
- Erdős #90 in this repo is the planar unit-distance count problem `u(n)`, not
  the chromatic number of the plane. The current Lean-facing artifacts include
  exact lattice and compact-disk lower-bound witnesses such as
  `u(15705) >= 176768` and `u(31417) >= 405648`.
- The 168/ZD/sedenion work is real but scoped. It proves finite algebraic
  separations and negative bridge results: the distance-based ZD/associator
  surgeries explored there do not produce a new planar chromatic lower bound,
  and the docs explicitly retract the tempting but non-faithful planar bridge.
- The `chi=6`-flavoured material in `examples/erdos/168_orbit_chi6_proof.sio`
  belongs to high-dimensional or algebraic experiment space, not a known
  Hadwiger-Nelson plane theorem. Treat it as a candidate/search lane until a
  concrete unit-distance graph plus independently checked 5-colouring UNSAT
  certificate exists.
- `stdlib/theorem/cardinality.sio` now makes that boundary executable with an
  Erdős/Hadwiger-Nelson scope gate: the current graph-colouring artifact is
  pinned as a `K3`/2-colour UNSAT replay with fingerprint `37553010`, while the
  `chi >= 6` preflight requires Erdős #508, lower bound 6, a concrete
  unit-distance graph, a 5-colour UNSAT certificate, and independent replay mask
  `31`. The current K3 witness intentionally fails that preflight.

## Upgrade direction

1. Proof kernel first: keep solvers aggressive, but make every UNSAT result
   checkable by a smaller independent kernel. For QF_LRA this means generic
   Farkas checking; for SAT this means LRAT/FRAT hints; for PB/Cardinality this
   means a VeriPB-like elaborator/backend split.
2. Interfaces before heroics: add stable assumption/core/stats/proof hooks around
   Sounio solvers, mirroring the shape of IPASIR/RustSAT without copying their
   implementation constraints.
3. Encoding layer: expose typed cardinality and pseudo-Boolean encodings as
   first-class Sounio libraries, with proof logging. This is where graph
   colouring, scheduling, and future clinical/resource constraints converge.
4. Wide numerics for dynamics: do not jump straight from f64 Lorenz to "i256
   proves chaos." Build fixed-point/ball primitives with explicit scale,
   radius, rounding, and overflow status, then put Lorenz behind interval/Taylor
   steps that return enclosures, not point guesses.
5. High precision means certified range, not pretty decimals. A Lorenz upgrade
   should report `(midpoint, radius, status)` and gate long-time claims through
   interval inclusion or shadowing-style certificates.

## Patch started in this wave

`stdlib/theorem/qflra_exact.sio` now keeps an original copy of each input
constraint row and adds `qf_verify_farkas_unsat()`, a generic exact checker for:

- `y >= 0`;
- `y * A == 0` for every variable column;
- `y * b < 0`.

`tests/run-pass/test_qflra_exact.sio` now calls this checker in addition to its
older hand-written certificate check. This moves QF_LRA UNSAT from "solver says
UNSAT and one test manually inspects y" toward "solver emits a reusable proof
object checked by a dedicated kernel."

Follow-up on the current Madaros path:

- `stdlib/theorem/smt.sio` now has the first assumption/core replay surface:
  `smt_solve_with_assumptions` temporarily appends assumption literals as unit
  clauses, solves, and restores the base clause database; `smt_check_assumption_core`
  replays a caller-provided subset of assumptions and accepts only when that
  subset is itself UNSAT. This is intentionally a checker-side contract for
  sufficient cores, not a minimal-core extractor. It is the smallest step toward
  IPASIR/RustSAT-style incremental assumptions without changing the main DPLL(T)
  search.
- `tests/run-pass/smt_assumption_core_tiny.sio` is the backend-friendly
  executable mirror for that semantics. It uses the base formula `x OR y`,
  verifies that assumptions `-x, -y` are UNSAT together, rejects either singleton
  as a sufficient core, rejects duplicate core indices, and rejects out-of-range
  core indices. It now also pins a small certificate envelope: formula
  fingerprint `796206516`, assumptions fingerprint `106325289`, core
  fingerprint `666359709`, and UNSAT-core bundle fingerprint `923659833`.
  The negative controls reject a changed assumption set, duplicate-core
  fingerprint, SAT solver result, and wrong core size. This is still a
  sufficient-core checker contract, not a minimal-core extractor. Gate:
  `bash scripts/run_sio_test_suite.sh smt_assumption_core_tiny --verbose` passes
  on this branch.
- The `theorem::smt` module itself now checks on the current Madaros path after
  narrow `i32`/`i64` literal repairs, and its intended API surface is public:
  `SmtContext`, construction/configuration/stat getters, clause/LIA loading,
  solving, and assumption/core replay. `tests/run-pass/smt_assumption_core_imported.sio`
  is the fresh importing gate; `./bin/souc check` passes for it, while native
  execution is marked known-failure because current Madaros multimodule native
  lowering exits 139 even on the official `thin_single` witness. See
  `docs/audit/MADAROS_MULTIMODULE_NATIVE_SEED_SEGFAULT_2026-06-22.md`.
  `tests/stdlib/theorem/test_smt_solver_basic.sio`
  now also frontend-checks through the public API, but its run path is marked
  known-failure at multimodule native thin-link lowering (`ir_bodies_failed`).
  Treat the remaining blocker as imported/native lowering, not solver API shape.
- `stdlib/math/rational.sio` had a parser-only compatibility issue: `rat_parse`
  used `s.len()` and direct `s[i]` indexing on a builtin string. The current
  compiler expects the established `str_len` / `str_char_at` builtin pattern
  used elsewhere in the repo. This was repaired without changing rational
  arithmetic.
- `./bin/souc check stdlib/math/rational.sio` and
  `./bin/souc check tests/run-pass/test_qflra_exact.sio` now pass.
- `tests/run-pass/qflra_farkas_checker_tiny.sio` adds a backend-friendly
  micro-gate for the proof checker idea itself. It verifies the integer Farkas
  certificate `y=[1,0,0,1]` for `x+y<=1, -x<=0, -y<=0, -x-y<=-2`, and rejects
  three negative controls: a negative multiplier, a non-contradictory `y*b`,
  and a non-cancelling column sum. It now also pins a certificate envelope:
  formula fingerprint `334817196`, Farkas-vector fingerprint `152003092`, and
  bundle fingerprint `13244761`, rejecting changed formulas, changed column
  sums, non-contradictory bounds, and invalid certificate vectors. This is a
  proof-kernel witness for exact Farkas UNSAT, not an optimization solver or a
  general proof interchange format. Gate:
  `bash scripts/run_sio_test_suite.sh qflra_farkas_checker_tiny --verbose`
  passes on this branch.
- `stdlib/theorem/cardinality.sio` adds the first proof-logged cardinality
  encoding surface: deterministic clause IDs for pairwise exactly-one
  encodings, stable literal lookup, and tiny conflict verifiers for ALO/AMO
  proof-log steps. This is the intended shared layer between graph colouring,
  future PB/Cardinality encodings, and proof checkers; it keeps encoding order
  out of individual solvers.
- `tests/run-pass/cardinality_proof_log_tiny.sio` is a backend-friendly
  self-contained executable mirror of that cardinality layer. It checks the
  `n=3` exactly-one clause layout, accepts the AMO conflict under assumptions
  `x1, x3`, rejects wrong clause IDs, duplicate literals, and signed-literal
  mistakes, and checks the ALO conflict under `-x1, -x2, -x3`. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_proof_log_tiny --verbose`
  passes on this branch.
- `stdlib/theorem/cardinality.sio` now also has the first higher-cardinality
  proof-log shape: pairwise-stable `at-most-two(x_1..x_n)` clauses, one ternary
  clause `(-x_i OR -x_j OR -x_k)` for every `i < j < k`, with deterministic
  tuple IDs, literal lookup, and conflict checking under three positive
  assumptions. This is not yet a totalizer or sequential-counter checker; it is
  the smallest proof-kernel surface that lets graph colouring, scheduling, and PB
  encoders talk about "three trues violate at-most-two" without trusting an ad
  hoc clause order.
- `tests/run-pass/cardinality_at_most_two_tiny.sio` is the backend-friendly
  executable mirror for that shape. For `n=4`, it checks the four lexicographic
  triple IDs, verifies the `(-x2 OR -x3 OR -x4)` clause layout, accepts the
  conflict under assumptions `x2, x3, x4` in multiple orders, and rejects wrong
  IDs, duplicate assumptions, negative literals, and out-of-range literals. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_at_most_two_tiny --verbose`
  passes on this branch.
- `stdlib/theorem/cardinality.sio` now also exposes a graph-colouring proof-log
  map: `card_colour_var`, stable per-vertex exactly-one clause IDs/literals,
  edge disequality clause IDs, and same-colour edge-conflict checks. This gives
  future Hadwiger-Nelson/graph-colouring encoders a deterministic bridge from
  `(vertex, colour)` and `(edge, colour)` facts to proof-kernel clause IDs,
  rather than letting each solver invent its own order.
- `tests/run-pass/cardinality_colouring_ids_tiny.sio` is the backend-friendly
  mirror for that colouring map. For two vertices and three colours it checks
  global variable numbering, vertex exactly-one clause ID blocks, lifted ALO/AMO
  literals, edge clause base/IDs, edge literals, and conflict checking for "both
  endpoints have colour 2". Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_ids_tiny --verbose`
  passes on this branch.
- `stdlib/theorem/cardinality.sio` now also exposes a graph-colouring CNF
  manifest layer: vertex-clause block size, edge-clause block size, total clause
  count, clause kind (`vertex` vs `edge`), and reverse decoding from a global
  clause ID back to `(vertex, local exactly-one clause)` or `(edge_index,
  colour)`. This is a proof-producer/checker contract: large colouring UNSAT
  certificates should be able to say "clause 473 is edge 91 colour 2" without
  trusting hidden generator order.
- `tests/run-pass/cardinality_colouring_manifest_scalar.sio` is the executable
  mirror for that manifest layout. For `K3` with two colours it checks six
  vertex clauses, six edge clauses, total clause count 12, vertex/edge boundary
  IDs, reverse decoding, and invalid-ID rejection. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_manifest_scalar --verbose`
  passes on this branch.
- The manifest layer now has a deterministic non-cryptographic fingerprint:
  `card_colour_manifest_fingerprint(vertex_count, edge_count, colour_count)`.
  It mixes the instance sizes, variable count, vertex block, edge base, edge
  block, and total clause count modulo `1000000007`. This is a lightweight
  proof-log anchor so a producer and checker can cheaply agree on "the CNF
  layout I am proving about" before replaying LRAT/PB evidence. It is not a
  collision-resistant hash and does not by itself certify satisfiability or
  UNSAT.
- `tests/run-pass/cardinality_colouring_manifest_fingerprint_tiny.sio` is the
  executable mirror for that anchor. It pins `K3` with two colours to
  fingerprint `296728038`, checks distinct fingerprints for changed edge count,
  vertex count, and colour count, and rejects invalid dimensions. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_manifest_fingerprint_tiny --verbose`
  passes on this branch.
- The colouring manifest layer now also exposes the LRAT header convention:
  `card_colour_first_derived_clause_id`, `card_colour_final_derived_clause_id`,
  and `card_colour_lrat_header_valid`. The rule is intentionally simple:
  derived proof clauses start at `total_base_clause_count + 1`, and a proof
  header is accepted only when both the manifest fingerprint and first derived
  clause ID match the instance. For K3/2-colour, the base CNF has 12 clauses,
  so the seven-step scalar LRAT smoke derives IDs 13 through 19.
- `tests/run-pass/cardinality_colouring_lrat_header_tiny.sio` is the executable
  mirror for that header. It checks K3/2-colour fingerprint `296728038`, total
  base count 12, first derived ID 13, final derived ID 19 for seven derived
  clauses, and rejects wrong fingerprints, wrong proof-start IDs, changed
  instance dimensions, invalid colours, and zero derived-count witnesses. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_lrat_header_tiny --verbose`
  passes on this branch.
- The same layer now exposes an UNSAT-certificate bundle contract:
  `card_colour_lrat_unsat_bundle_valid` and
  `card_colour_lrat_unsat_bundle_fingerprint`. The bundle binds the colouring
  manifest fingerprint, first derived LRAT ID, derived-count, final derived ID,
  and final clause length. For K3/2-colour with seven derived clauses ending in
  the empty clause, the bundle fingerprint is `500954697`. This catches the
  common proof-plumbing error class "valid-looking proof metadata attached to
  the wrong CNF or wrong final proof line"; it is still not a full LRAT parser
  or an independent proof replay by itself.
- `tests/run-pass/cardinality_colouring_unsat_bundle_tiny.sio` is the scalar
  executable mirror for that bundle. It accepts the K3/2-colour envelope
  `(manifest_fp=296728038, first=13, derived_count=7, final=19, final_len=0)`,
  pins bundle fingerprint `500954697`, and rejects wrong manifest fingerprints,
  wrong first IDs, wrong derived counts, wrong final IDs, non-empty final
  clauses, and changed instance dimensions. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_unsat_bundle_tiny --verbose`
  passes on this branch.
- `stdlib/theorem/cardinality.sio` now also exposes a small graph-instance
  anchor for three-edge colouring certificates:
  `card_colour_graph3_fingerprint`,
  `card_colour_graph3_instance_fingerprint`, and
  `card_colour_graph3_unsat_claim_fingerprint`. This separates the concrete
  graph edge list from the generic CNF layout. For `K3` with ordered edges
  `(0,1),(0,2),(1,2)` and two colours, the graph fingerprint is `786487984`,
  the graph+CNF instance fingerprint is `77085393`, and the UNSAT claim
  fingerprint over bundle `500954697` plus replay anchor `891522734` is
  `848522370`. This is a minimal graph-colouring certificate shape: graph
  identity, colouring CNF manifest, and replayed UNSAT evidence are separate
  fields. It is still only the `K3` two-colouring smoke, not a
  plane-chromatic lower bound.
- `tests/run-pass/cardinality_colouring_graph_instance_tiny.sio` mirrors that
  graph-instance layer. It accepts the ordered `K3` edge list, pins the three
  fingerprints above, and rejects loops, out-of-order edge lists, reversed edge
  endpoints, wrong manifest fingerprints, non-UNSAT result codes, and missing
  derived proof lines. Gate:
  `bash scripts/run_sio_test_suite.sh cardinality_colouring_graph_instance_tiny --verbose`
  passes on this branch.
- `tests/run-pass/cardinality_colouring_manifest_tiny.sio` is the desired
  imported-library version using `use theorem::cardinality::*`. It typechecks,
  but is marked known-failure because current imported/native lowering exits
  139 at runtime. That is the same backend class as the other imported
  theorem-module witnesses, not evidence against the manifest arithmetic.
- `stdlib/theorem/pb.sio` adds the first pseudo-Boolean checker-side layer:
  bounded three-literal inequalities with conservative unknown handling. It
  checks contradictions for `sum(w_i*x_i) <= rhs` and `sum(w_i*x_i) >= rhs`, and
  exposes exactly-one/at-most-one conflict helpers as PB bounds rather than only
  clauses. It now also checks coefficient saturation: for Boolean variables and
  non-negative `rhs`, coefficients above `rhs` can be replaced by `rhs`, a small
  but real cutting-plane style proof step. It also checks positive-integer
  division with the sound rounding direction: floor for `<=` rows and ceil for
  `>=` rows. The PB layer now has bounded non-negative scaling, same-sense row
  addition, and forced-literal propagation as well. Scaling/addition reject
  values outside a small `i64` range so the checker does not certify overflowed
  arithmetic. This is the local analogue of the VeriPB direction: rich encodings
  should elaborate into tiny independently checked arithmetic steps.
- The PB layer now names the three-literal `at-most-two` cardinality row as
  `x0 + x1 + x2 <= 2`, with `pb_card_at_most_two_row3`,
  `pb_check_at_most_two_true_conflict3`, and the row wrapper
  `pb_row_check_at_most_two_conflict3`. This is a small bridge between the
  cardinality proof-log surface and the PB checker: the clause
  `(-x_i OR -x_j OR -x_k)` and the PB row `x_i+x_j+x_k <= 2` certify the same
  three-true conflict, but through two independently shaped proof vocabularies.
- That bridge is now explicit in the public PB API. `pb_card_at_most_two_clause_matches3`
  checks that a ternary SAT clause is exactly the three negative literals for a
  proposed AM2 row; `pb_check_card_at_most_two_clause_conflict3` and
  `pb_row_check_card_at_most_two_clause_conflict3` then require both the clause
  shape and the PB contradiction under three true assignments. This is the
  first small VeriPB-like elaboration hook where a SAT encoding artifact and a
  PB row artifact must agree before the checker accepts the conflict.
- The PB layer now also exposes the first minimization optimality envelope:
  `PbObjective3`, objective-value checking, objective/row/assignment
  fingerprints, and `pb_check_min_objective_optimal3`. The certificate shape is
  deliberately small: a derived PB lower-bound row `objective >= k` plus an
  incumbent Boolean assignment whose objective value is exactly `k`. This is
  not a MaxSAT/PB optimizer and not a VeriPB parser; it is the smallest checked
  kernel shape needed before Sounio can honestly talk about proof-carrying
  objective bounds instead of only PB refutations.
- `tests/run-pass/pb_tiny_checker.sio` is the backend-friendly executable smoke.
  It verifies weighted `<=` and `>=` contradictions, exactly-one all-false
  conflict, at-most-one two-true conflict, coefficient saturation, division
  with floor/ceil rounding, and rejects invalid assignments, negative weights,
  invalid saturation witnesses, wrong rounded quotients, zero divisors, and
  negative coefficients. Gate:
  `bash scripts/run_sio_test_suite.sh pb_tiny_checker --verbose` passes on this
  branch.
- `tests/run-pass/pb_cardinality_bridge_tiny.sio` is the executable bridge
  smoke. For `n=4`, it requires both sides of the certificate: cardinality
  clause ID 4 must be the stable ternary clause `(-x2 OR -x3 OR -x4)`, and the
  PB row `x2+x3+x4 <= 2` must be contradictory under three true assumptions. It
  rejects the half-proofs: correct cardinality with only two true assignments,
  PB contradiction under a wrong clause ID, duplicate literals, negative
  literals, out-of-range literals, invalid PB assignments, and SAT/PB clause
  shape mismatches such as duplicate negative literals or the wrong variable.
  Gate:
  `bash scripts/run_sio_test_suite.sh pb_cardinality_bridge_tiny --verbose`
  passes on this branch.
- `tests/run-pass/pb_cardinality_bridge_imported.sio` is the imported-library
  smoke for the public bridge API. It frontend-checks against `theorem::pb`, but
  runtime is marked known-failure under the same imported/native lowering exit
  139 blocker as the other broad `theorem::pb` imported gate.
- `tests/run-pass/pb_linear_checker_tiny.sio` is the split executable smoke for
  PB linear-combination proof steps. It verifies non-negative row scaling,
  coefficient-wise addition, RHS addition, wrong witnesses, negative inputs, and
  uses the added row to certify a `<=` contradiction. Gate:
  `bash scripts/run_sio_test_suite.sh pb_linear_checker_tiny --verbose` passes
  on this branch. The split file is intentional: a monolithic PB smoke with all
  helpers in one file typechecked but hit the Madaros `IR lowering failed`
  surface, while the smaller gates lower and run.
- `tests/run-pass/pb_propagation_checker_tiny.sio` is the executable smoke for
  PB forced-literal propagation. For `<=` rows it verifies that an unknown
  literal is forced false when setting it true would exceed the RHS. For `>=`
  rows it verifies that an unknown literal is forced true when setting it false
  would make the RHS unreachable. It rejects invalid slots, invalid
  assignments, negative weights, and already-assigned targets. Gate:
  `bash scripts/run_sio_test_suite.sh pb_propagation_checker_tiny --verbose`
  passes on this branch.
- The PB layer now also exposes a row-shaped checker API through `PbRow3`:
  `sense=-1` for `<=`, `sense=+1` for `>=`, plus row-level wrappers for
  contradiction, forced literals, saturation, division, scaling, and same-sense
  addition. This is an interface step toward a VeriPB-like backend: producers
  should elaborate rich PB reasoning into small checked row transformations,
  not into unstructured "solver says UNSAT" statuses.
- `tests/run-pass/pb_row_chain_tiny.sio` adds a backend-friendly executable
  chain for that proof style without using struct-by-value at runtime. It checks
  `saturate -> divide -> scale -> add -> contradiction` over nonnegative
  three-literal PB rows, including negative controls for wrong saturation,
  wrong floor/ceil division, wrong scaling, wrong addition, and a
  non-contradictory assignment. It now also binds the additive contradiction
  path into a deterministic scalar certificate bundle: base rows
  `(2,1,3<=4)` and `(1,4,0<=2)` derive `(3,5,3<=6)`, assignment
  `(true,true,unknown)` yields a `<=` contradiction, and the checked bundle
  fingerprint is `624517492` (`rowA=361051473`, `rowB=46142869`,
  `rowC=617917101`, `assignment=123291979`). This is a VeriPB-shaped replay
  envelope over a tiny row chain, not a full VeriPB parser/checker. Gate:
  `bash scripts/run_sio_test_suite.sh pb_row_chain_tiny --verbose` passes on
  this branch.
- `tests/run-pass/pb_row_chain_imported.sio` is the desired imported-library
  version using `use theorem::pb::*` and the new `PbRow3` API. It typechecks,
  but runtime is marked known-failure because the broad multimodule native
  seed-lowering blocker exits 139 on the current Madaros path. This is evidence
  for the existing import backend blocker, not a PB proof-rule failure.
- `tests/run-pass/pb_min_optimality_tiny.sio` is the executable tiny
  optimization-certificate gate. It minimizes `2*x0 + 3*x1 + 5*x2`, checks the
  lower-bound row `2*x0 + 3*x1 + 5*x2 >= 3`, checks incumbent assignment
  `(false,true,false)` with objective value `3`, and pins objective fingerprint
  `910777053`, row fingerprint `177229408`, assignment fingerprint `625953239`,
  and certificate fingerprint `271184648`. Negative controls reject wrong
  bounds, wrong objective coefficients, wrong row sense, and incumbents whose
  value does not meet the bound exactly. Gate:
  `bash scripts/run_sio_test_suite.sh pb_min_optimality_tiny --verbose` passes
  on this branch.
- `stdlib/theorem/lrat.sio` adds the first native LRAT/RUP replay kernel:
  explicit clause IDs, active/deleted clause state, bounded clause storage,
  negated-lemma assignment, and hint replay that accepts only when unit
  propagation over the antecedent IDs derives conflict. It is a checker kernel,
  not a solver.
- `tests/run-pass/lrat_tiny_checker.sio` is the backend-friendly scalar smoke for
  that LRAT idea. It verifies the tiny formula `{x}, {-x}` derives the empty
  clause from hints `[1,2]`, accepts either hint order, rejects missing hints,
  rejects unknown IDs, rejects a deleted antecedent, and accepts the tautological
  lemma `x OR -x` because its negation conflicts immediately. Gate:
  `bash scripts/run_sio_test_suite.sh lrat_tiny_checker --verbose` passes on
  this branch.
- `stdlib/theorem/lrat.sio` now exposes named wrappers for common RUP proof
  steps: `lrat_check_unit_rup4`, `lrat_add_unit_rup4`,
  `lrat_check_empty_rup4`, and `lrat_add_empty_rup4`. These are thin wrappers
  over the bounded four-literal checker, but they make the proof-kernel API read
  like a certificate trace instead of an ad hoc `len=0`/`len=1` convention.
- `stdlib/theorem/lrat.sio` also exposes an explicit binary-resolution unit
  step: `lrat_check_binary_resolution_unit` and
  `lrat_add_binary_resolution_unit`. This checks the common proof-trace move
  from `(p OR r)` and `(-p OR r)` to unit `r`, while still requiring active
  clause IDs and nonzero pivot/resolvent literals. This does not replace RUP or
  make Sounio a full LRAT/LPR parser, but it adds a small direct resolution
  kernel that sits closer to the LRAT/LPR/cake_lpr direction used by modern SAT
  proof checkers.
- `tests/run-pass/lrat_binary_resolution_tiny.sio` is the backend-friendly
  executable mirror for that rule. It derives `y` from `(x OR y)` and
  `(-x OR y)`, accepts the symmetric clause order with pivot `-x`, and rejects
  wrong resolvents, wrong pivots, zero literals, deleted antecedents, reused
  IDs, and duplicate additions. Gate:
  `bash scripts/run_sio_test_suite.sh lrat_binary_resolution --verbose` passes
  on this branch.
- `tests/run-pass/lrat_chain_checker_tiny.sio` adds the next scalar LRAT/RUP
  proof-chain smoke. For the formula `(x OR y), -x, -y`, it derives unit `y`
  by RUP, then derives the empty clause from the derived unit and `-y`. The gate
  accepts both valid hint orders for the unit lemma, rejects missing/unknown
  antecedents, rejects using the empty proof before the derived unit exists, and
  rejects the proof after deleting the required `-y` clause. Gate:
  `bash scripts/run_sio_test_suite.sh lrat_chain_checker_tiny --verbose` passes
  on this branch.
- `stdlib/theorem/lrat.sio` now also exposes the first FRAT-to-LRAT elaboration
  envelope:
  `lrat_frat_elaboration_manifest_fingerprint`,
  `lrat_frat_hint_trace_fingerprint`, and
  `lrat_frat_unsat_envelope_fingerprint`. The tiny envelope models the
  `{x OR y, -x, -y} -> y -> empty` trace with three original clauses, two
  additions, one deletion marker, derived IDs `4..5`, final empty clause length
  `0`, manifest `975208343`, hint trace `459486070`, and final envelope
  `86860040`. This is the first Sounio hook for the FRAT idea that rich solver
  traces can be elaborated into a smaller LRAT-style backend certificate; it is
  not a FRAT parser and not a replacement for full LRAT replay.
- `tests/run-pass/lrat_frat_elaboration_tiny.sio` is the scalar executable
  mirror for that envelope. It accepts the canonical trace metadata and rejects
  missing deletion markers, nonempty final clauses, wrong deleted IDs, wrong
  hint counts, and missing final empty evidence. Gate:
  `bash scripts/run_sio_test_suite.sh lrat_frat_elaboration --verbose` passes
  on this branch.
- `tests/run-pass/lrat_frat_elaboration_imported.sio` is the desired public API
  smoke through `theorem::lrat::*`. It frontend-checks and remains marked
  known-failure under the existing array-backed imported/native lowering exit
  139 boundary.
- `tests/run-pass/colouring_k3_lrat_tiny.sio` is the first scalar bridge from
  graph-colouring encoding to a RUP proof chain. It uses the stable colouring
  variable/clause-ID convention for `K3` with two colours, checks the expected
  vertex and edge clause IDs, derives four colour-implication lemmas, derives
  `-v0c0` and `-v0c1`, and finally derives the empty clause from the vertex-0
  at-least-one clause. The same executable now also binds that replay to the
  colouring manifest fingerprint `296728038`, the UNSAT bundle fingerprint
  `500954697`, derived IDs `13..19`, and an integrated replay anchor
  `891522734`. This proves the tiny theorem "triangle is not 2-colourable" as a
  proof-kernel smoke while also checking that the replay metadata is attached to
  the intended CNF; it is not a Hadwiger-Nelson bound and not evidence for
  `chi >= 6`. Gate:
  `bash scripts/run_sio_test_suite.sh colouring_k3_lrat_tiny --verbose` passes
  on this branch, and the test is included in the `lrat` suite.
- `tests/run-pass/lrat_chain_imported.sio` is the desired imported-library
  version using `theorem::lrat::*` plus the new wrappers. It typechecks, but is
  marked known-failure because the current array-backed LRAT runtime still exits
  139 in Madaros native lowering. This pins a compiler/backend boundary rather
  than weakening the RUP math claim.
- `tests/run-pass/lrat_binary_resolution_imported.sio` is the corresponding
  imported-library smoke for the binary-resolution helpers. It typechecks and
  keeps the same known-failure boundary as the broader imported LRAT gates:
  runtime still depends on the current array-backed module path, which exits 139
  under Madaros native lowering.
- `stdlib/theorem/lrat.sio` now also exposes lifecycle preflight helpers:
  `lrat_id_seen`, `lrat_is_deleted`, `lrat_can_add_clause_id`,
  `lrat_can_add_clause4`, and `lrat_next_fresh_id`. These helpers do not make
  the checker a full LRAT parser; they make the existing stable-ID policy
  explicit for proof producers. In particular, deleted IDs remain seen and are
  not reusable, while `lrat_next_fresh_id` gives a simple monotone addition-ID
  convention for bounded traces.
- `tests/run-pass/lrat_deletion_lifecycle_tiny.sio` adds a scalar lifecycle
  gate for that policy. It models `(x OR y), -x, -y`, derives `y`, derives the
  empty clause, then checks the negative cases that matter for LRAT soundness:
  using deleted `-x` cannot derive `y`, using deleted `-y` cannot derive the
  empty clause, an already-used ID cannot be added again after deletion, and the
  next-fresh-ID convention advances from 4 to 5 before reporting exhaustion in
  the tiny bounded model. Gate:
  `bash scripts/run_sio_test_suite.sh lrat_deletion_lifecycle_tiny --verbose`
  passes on this branch.
- `tests/run-pass/sat_resolution_chain_tiny.sio` adds an explicit
  resolution-proof smoke alongside the RUP/LRAT gates. For base clauses
  `(x OR y)`, `(-x OR z)`, `-y`, and `-z`, it resolves on `x` to derive
  `(y OR z)`, resolves on `y` to derive `z`, and resolves on `z` to derive the
  empty clause. The checker rejects wrong pivots, tautological outputs, missing
  derived antecedents, and using the final empty step before the unit `z`
  exists. It pins instance fingerprint `829764719` and certificate fingerprint
  `177576187`. This is a tiny explicit resolution trace, not a complete DRAT or
  LRAT parser. Gate:
  `bash scripts/run_sio_test_suite.sh sat_resolution_chain_tiny --verbose`
  passes on this branch.
- `tests/run-pass/sat_resolution_rup_bridge_tiny.sio` then checks that the same
  explicit resolution trace has a RUP-style replay: `(y OR z)` is RUP over the
  base clauses, `z` is RUP after `(y OR z)` is added, and the empty clause is
  RUP after `z` is added. It pins bridge fingerprint `759705343` and rejects
  missing derived antecedents or a broken replay step. This is a
  cross-format normalization smoke, not a general resolution-to-LRAT compiler.
  Gate:
  `bash scripts/run_sio_test_suite.sh sat_resolution_rup_bridge_tiny --verbose`
  passes on this branch.
- `tests/run-pass/sat_model_witness_tiny.sio` adds the positive SAT side of the
  same certificate story. For a fixed four-clause CNF over variables
  `x, y, z`, the witness assignment `x=true, y=false, z=true` is checked
  against every clause, with negative controls for a false extra clause,
  duplicate literals, zero literals, a bad satisfying assignment, a formula
  fingerprint mismatch, and a model fingerprint mismatch. It pins formula
  fingerprint `620950027`, model fingerprint `904583820`, and SAT certificate
  fingerprint `701650234`. This is a model-witness checker for one tiny CNF,
  not a SAT solver, not a completeness proof, and not a replacement for
  UNSAT proof replay. Gate:
  `bash scripts/run_sio_test_suite.sh sat_model_witness_tiny --verbose`
  passes on this branch.
- A first array-backed self-contained LRAT run-pass shaped like the stdlib
  module typechecked but crashed native compilation as a single no-import module
  (`Loaded 1 modules`, `Type check complete`, then wrapper `exit 139`). That is
  a separate native-lowering/global-array blocker from the imported-module
  `lower_array` crash; keep the scalar smoke plus `./bin/souc check
  stdlib/theorem/lrat.sio` until the compiler lane can carry the array-backed
  checker at runtime.
- `./bin/souc run tests/run-pass/test_qflra_exact.sio`,
  `./bin/souc run tests/run-pass/test_rational_exact.sio`, and
  `bash scripts/run_sio_test_suite.sh smt_qflia_basic --verbose` still exit 139
  in the native backend after typecheck, during imported/native lowering. Treat
  this as a compiler/backend blocker, not as evidence against the Farkas
  condition. The exact Farkas math was separately reviewed by
  `bin/llm-offload -t math-review -p xai` on 2026-06-22: accepted as sound, with
  the scope limitation above also accepted. `test_qflra_exact.sio` is now
  explicitly marked known-failure for this imported/native runtime blocker while
  `qflra_farkas_checker_tiny.sio` remains the executable proof-kernel witness.

### Cross-checker portfolio manifest

The solver/proof-checker lane now has a first cross-checker manifest layer in
`stdlib/theorem/portfolio.sio`. It does not re-check LRAT, SMT, QF_LRA, PB, or
Lorenz certificates directly. Instead it binds already-checked scalar
certificate fingerprints into a single ordered manifest so a producer and a
consumer can agree on the exact portfolio of tiny checker envelopes being
composed.

`tests/run-pass/solver_portfolio_manifest_tiny.sio` is the executable scalar
mirror. Version 1 binds five existing anchors:

- `K3`/2-colour LRAT replay: instance `296728038`, certificate `891522734`,
  entry `53646245`.
- SMT assumption-core envelope: instance `796206516`, certificate `923659833`,
  entry `528026990`.
- QF_LRA/Farkas envelope: instance `334817196`, certificate `13244761`, entry
  `301444501`.
- PB row-chain envelope: instance `617917101`, certificate `624517492`, entry
  `258927862`.
- Lorenz five-step trajectory manifest: instance `236665224`, certificate
  `508132668`, entry `388915166`.

The resulting v1 portfolio fingerprint is `902469519`. Version 2 appends the
`i256` Lorenz pre-division numerator manifest below: instance `300929056`,
certificate `572148472`, entry `900589789`, yielding portfolio fingerprint
`195085206`. Version 3 also appends the `i256` pre-division bit-budget
certificate below: instance `743388522`, certificate `133303313`, entry
`205040674`, yielding portfolio fingerprint `278042210`. Version 4 appends the
SAT-resolution trace above: instance `829764719`, certificate `177576187`, entry
`580031260`, yielding portfolio fingerprint `609543650`. The v4 check lives in
`tests/run-pass/solver_portfolio_sat_resolution_v4_tiny.sio` rather than the
already-large accumulated portfolio smoke, because adding the extra v4 runtime
logic to that single file triggered the current native `exit 139` surface even
though the separated v4 gate passes. Version 5 appends the resolution-to-RUP
bridge: instance `829764719`, certificate `759705343`, entry `973127057`,
yielding portfolio fingerprint `535346450`, checked in
`tests/run-pass/solver_portfolio_sat_bridge_v5_tiny.sio`. Version 6 appends the
PB minimization optimality envelope: instance `910777053`, certificate
`271184648`, entry `438043475`, yielding portfolio fingerprint `865143107`,
checked in `tests/run-pass/solver_portfolio_pb_optimality_v6_tiny.sio`.
Version 7 appends the Lorenz `i256` division-witness envelope below: instance
`660429472`, certificate `510854875`, entry `887056435`, yielding portfolio
fingerprint `35245852`, checked in
`tests/run-pass/solver_portfolio_lorenz_i256_division_v7_tiny.sio`. Version 8
appends the two-step Lorenz `i256` witness-chain envelope: instance `660429472`,
certificate `335080767`, entry `769531137`, yielding portfolio fingerprint
`539941120`, checked in
`tests/run-pass/solver_portfolio_lorenz_i256_chain_v8_tiny.sio`. Version 9
appends the three-step Lorenz `i256` witness-chain envelope: instance
`660429472`, certificate `249889958`, entry `579271506`, yielding portfolio
fingerprint `116716563`, checked in
`tests/run-pass/solver_portfolio_lorenz_i256_chain_v9_tiny.sio`. Version 10
appends the Lorenz `i256` three-step point-trajectory manifest: instance
`660429472`, certificate `449592233`, entry `252932879`, yielding portfolio
fingerprint `292334948`, checked in
`tests/run-pass/solver_portfolio_lorenz_i256_trajectory_v10_tiny.sio`. Version 11
appends the SAT model-witness envelope above: instance/formula `620950027`,
certificate `701650234`, entry `838301653`, yielding portfolio fingerprint
`210925933`, checked in
`tests/run-pass/solver_portfolio_sat_model_v11_tiny.sio`. Version 12 appends a
four-step Lorenz `i256` quotient/remainder witness-chain envelope: instance
`660429472`, certificate/chain anchor `737039167`, entry `755389732`, yielding
portfolio fingerprint `494109851`, checked in
`tests/run-pass/solver_portfolio_lorenz_i256_chain_v12_tiny.sio`. Version 13
adds a Lorenz `i256` single-step certificate contract for the fourth point in
the chain. This gate checks the same step-4 quotient/remainder witnesses, also
checks the time-step witness `2^32 / 100 = 42949672 r96`, packages seven
sub-witness checks into mask `127`, and binds instance `924803514`,
certificate/contract `630097760`, entry `521104746`, and portfolio fingerprint
`639301048`. The checks live in
`tests/run-pass/lorenz_i256_step_certificate_tiny.sio` and
`tests/run-pass/solver_portfolio_lorenz_i256_step_v13_tiny.sio`. The same
contract is now also staged as a reusable stdlib surface in
`stdlib/systems/lorenz_i256_cert.sio`, composed with
`theorem::div_witness` rather than duplicating the quotient/remainder rule.
`./bin/souc check stdlib/systems/lorenz_i256_cert.sio` checks the composed
module. The imported runtime gate
`tests/run-pass/lorenz_i256_step_certificate_imported.sio` frontend-checks the
API but is marked known-failure because current imported/native lowering exits
139; the self-contained step gate remains the runtime evidence for the
arithmetic contract. Version 14 appends the graph-colouring `K3` UNSAT-claim
envelope above: graph+CNF instance `77085393`, claim/certificate `848522370`,
entry `537394311`, yielding portfolio fingerprint `600905571`, checked in
`tests/run-pass/solver_portfolio_graph_colouring_v14_tiny.sio`. This portfolio
entry binds the K3 graph-instance anchor, colouring CNF manifest, UNSAT bundle,
and replay anchor as proof-carrying metadata only; it is not a stronger
colouring theorem than the underlying K3/2-colour replay. Negative controls
reject wrong result categories (`UNSAT`, `SAT`, and `validated trajectory`),
mismatched checker kinds, wrong manifest version, wrong entry count, and altered
entry fingerprints. This is proof-carrying metadata for a mixed solver/dynamics
portfolio; it does not make any component certificate stronger than its own
gate. The v13 step contract is still a point-step certificate rather than a
validated enclosure or long-time Lorenz theorem, and the v14 graph-colouring
entry is still only the `K3` two-colouring smoke rather than a plane-chromatic
lower bound. Version 15 appends a fifth Lorenz `i256` quotient/remainder
witness-chain envelope: starting from the fourth-step state
`(4920371098, 8816077911, 4092986080)`, the supplied division witnesses produce
the fifth state `(5309941770, 10058731223, 4084837935)`, step fingerprint
`561641681`, chain anchor `23144051`, entry `827916944`, and portfolio
fingerprint `222692310`, checked in
`tests/run-pass/lorenz_i256_division_chain_five_step_tiny.sio` and
`tests/run-pass/solver_portfolio_lorenz_i256_chain_v15_tiny.sio`. This remains
a supplied quotient/remainder replay chain over fixed-point point states: it is
not native source-level `i256` division, not a validated enclosure, and not a
long-time Lorenz theorem. The fifth-step witness checker is now also staged in
`stdlib/systems/lorenz_i256_cert.sio` as public API:
`lorenz_i256_step5_certificate_check` replays the `dt`, derivative-scaling,
increment, and final-state quotient/remainder witnesses through
`theorem::div_witness`, returns chain anchor `23144051` only when the supplied
certificate fingerprint is `561641681`, and is frontend-checked by
`tests/run-pass/lorenz_i256_step5_certificate_imported.sio`. The imported gate
shares the current imported/native exit-139 known-failure boundary; the
self-contained v15 chain remains the runtime evidence. Version 16 appends the
corresponding five-step
point-trajectory manifest using the existing Lorenz `i256` trajectory checker:
scale `2^32`, time-step witness `42949672 r96`, five steps from
`(2^32, 2^32, 2^32)` to `(5309941770, 10058731223, 4084837935)`, chain anchor
`23144051`, final step certificate `561641681`, trajectory manifest
`214180161`, portfolio entry `626160047`, and portfolio fingerprint
`320118125`, checked in
`tests/run-pass/lorenz_i256_trajectory5_manifest_tiny.sio` and
`tests/run-pass/solver_portfolio_lorenz_i256_trajectory_v16_tiny.sio`. This
manifest binds metadata for the already checked point chain; it is not a new
integrator, not an interval/Taylor enclosure, and not a shadowing or chaos
claim. The five-step manifest helpers are also staged in
`stdlib/systems/lorenz_i256_cert.sio` as reusable API surface; the imported gate
`tests/run-pass/lorenz_i256_trajectory5_manifest_imported.sio` frontend-checks
that API but is marked known-failure for the same current imported/native
lowering exit-139 boundary as the step-contract import path. The stronger
composition helper `lorenz_i256_trajectory5_certificate_check` now connects
those two public surfaces: it first replays the fifth-step quotient/remainder
certificate, requires chain anchor `23144051`, then emits trajectory manifest
`214180161`. The runtime gate
`tests/run-pass/lorenz_i256_trajectory5_certificate_tiny.sio` mirrors that
composition self-contained; the imported gate
`tests/run-pass/lorenz_i256_trajectory5_certificate_imported.sio`
frontend-checks the public API and remains under the same imported/native
known-failure boundary. This is still a point-trajectory certificate over fixed
Euler witnesses, not adaptive integration or interval/Taylor validation. The
portfolio v16 consumer is likewise staged as
`tests/run-pass/solver_portfolio_lorenz_i256_trajectory_v16_imported.sio`; it
frontend-checks the public `theorem::portfolio` API for the v16 entry and
manifest, while the self-contained v16 smoke remains the runtime evidence until
the imported/native lowering boundary is fixed. To reduce consumer boilerplate,
`theorem::portfolio` now also exposes
`solver_portfolio_v16_lorenz_i256_trajectory5_bundle_fingerprint(instance,
trajectory_manifest, chain5_entry)`, which reconstructs the v16 entry and
manifest from the three semantically meaningful anchors and rejects altered
trajectory or chain fingerprints. The composed imported gate
`tests/run-pass/lorenz_i256_portfolio_v16_composed_imported.sio` ties the two
public APIs together: it obtains the trajectory manifest from the certificate
checker `systems::lorenz_i256_cert::lorenz_i256_trajectory5_certificate_check`,
feeds that manifest into `theorem::portfolio`, and keeps the same
imported/native known-failure boundary until multimodule native lowering is
repaired. The executable mirror
`tests/run-pass/lorenz_i256_portfolio_v16_composed_tiny.sio` now checks the same
certificate-backed composition self-contained on the current native path: it
replays the fifth-step quotient/remainder witnesses, derives manifest
`214180161`, reconstructs the v16 portfolio fingerprint `320118125`, derives
result coverage `685105937`, checker-family coverage `210011781`, acceptance
receipt `629028996`, audit receipt `932551699`, and final readiness fingerprint
`589844740`. This gives runtime evidence for the proof-carrying composition
from Lorenz quotient/remainder witnesses into the portfolio readiness envelope
even while the imported version remains a frontend-only API gate. The v16
readiness path is now promoted into reusable systems API as
`systems::lorenz_i256_cert::lorenz_i256_portfolio_v16_readiness_check`, which
takes the fifth-step quotient/remainder witness fields, derives the trajectory
manifest through the Lorenz certificate checker, then walks the public
`theorem::portfolio` envelope to readiness `589844740`. The imported smoke
`tests/run-pass/lorenz_i256_portfolio_v16_readiness_imported.sio` frontend-checks
this one-call API and keeps the same imported/native runtime known-failure
boundary. This is a consumer-facing certificate-to-receipt path over fixed
witnesses and reviewed portfolio anchors; it is not an inline replay of every
portfolio component checker and not a new Lorenz theorem.

`systems::lorenz_i256_cert` now also exposes the first inclusion bridge from a
wide fixed-point point trajectory into a validated ball enclosure:
`lorenz_i256_trajectory5_projection_inclusion_fingerprint` projects the final
`i256` point trajectory from scale `2^32` into scale `1e6` using supplied
division witnesses, then checks the projected coordinates
`(1236317, 2341980, 951075)` against the fifth-step validated enclosure
`x in [992950,1479682]`, `y in [2022339,2661619]`, and
`z in [908630,993526]`. The inclusion fingerprint is `890086395`; the companion
`lorenz_i256_trajectory5_projection_contract_fingerprint` binds that inclusion
to trajectory manifest `214180161`, radius budget `565673886`, quality profile
`70877465`, and mask `7`, yielding contract `607091799`. Runtime evidence is
`tests/run-pass/lorenz_i256_projection_inclusion_tiny.sio`; the public imported
API smoke is `tests/run-pass/lorenz_i256_projection_inclusion_imported.sio` and
remains under the imported/native known-failure boundary. This is a cross-scale
inclusion certificate for the five-step smoke, not a Taylor-model integrator or
long-time Lorenz theorem.

The v16 portfolio also now carries a result-coverage
anchor, `solver_portfolio_v16_result_coverage_fingerprint`, pinning the mixed
manifest shape to seven UNSAT entries, eleven validated numeric/trajectory
entries, one optimality entry, and one SAT model entry over the v16 manifest
`320118125`; the executable gate
`tests/run-pass/solver_portfolio_v16_coverage_tiny.sio` is the runtime evidence,
while `tests/run-pass/solver_portfolio_v16_coverage_imported.sio` frontend-checks
the public API and remains under the same imported/native known-failure boundary.
For consumers that only need a final envelope, `theorem::portfolio` also exposes
`solver_portfolio_v16_acceptance_receipt_fingerprint`, binding manifest
`320118125` and coverage `685105937` into receipt `629028996`; the tiny and
imported gates `solver_portfolio_v16_acceptance_receipt_*` mirror the usual
runtime/API split. The companion
`solver_portfolio_v16_acceptance_from_counts_fingerprint` derives the coverage
anchor from the semantic counts before emitting the same receipt, giving callers
a single final-envelope API over the manifest and counts. This remains a compact
acceptance receipt, not a substitute for replaying each component certificate.
The next ergonomic layer,
`solver_portfolio_v16_acceptance_from_entries_fingerprint`, reconstructs the
v16 manifest from the twenty reviewed portfolio entries, checks it against the
known v16 bundle, derives coverage from the counts, and emits the same final
receipt. That gives Sounio a one-call portfolio acceptance path over SAT, SMT,
PB, graph-colouring, and Lorenz certificate anchors while preserving the claim
boundary: it is portfolio integrity evidence over reviewed anchors, not a
replacement for the individual proof checkers. To make that envelope less
opaque, the portfolio also exposes
`solver_portfolio_v16_checker_family_coverage_fingerprint`, pinning the checker
mix to four SAT-family entries, two SMT/QF_LRA entries, two PB entries, one
graph-colouring entry, and eleven Lorenz numeric/trajectory entries over the
same manifest. The family fingerprint is `210011781`; the tiny/imported gates
`solver_portfolio_v16_checker_family_coverage_*` keep this as taxonomy evidence,
not a new solver-completeness claim. The current top receipt is
`solver_portfolio_v16_audit_receipt_fingerprint`, which binds manifest
`320118125`, result coverage `685105937`, family coverage `210011781`, and
acceptance receipt `629028996` into audit receipt `932551699`. This is the
compact v16 portfolio audit envelope; it proves that the already-reviewed
anchors, counts, taxonomy, and receipt agree, not that any component certificate
has been rechecked inside the portfolio layer. The derived helper
`solver_portfolio_v16_audit_from_entries_fingerprint` takes the twenty portfolio
entries, result counts, and checker-family counts, reconstructs the manifest,
result coverage, family coverage, acceptance receipt, and final audit receipt in
one path, returning the same `932551699` only when all layers agree. For
consumer diagnostics, `solver_portfolio_v16_readiness_mask` exposes which audit
layers are present, with the complete v16 mask `31`, and
`solver_portfolio_v16_readiness_fingerprint` pins the fully ready envelope to
`589844740`; a partial mask is diagnostic only and does not upgrade into an
acceptance claim. The follow-on helper
`solver_portfolio_v16_readiness_from_entries_fingerprint` derives the whole
chain from the twenty entries plus result and checker-family counts, then emits
the same readiness fingerprint only after manifest, coverage, family, acceptance,
and audit receipts all agree. This is the strongest v16 consumer API in the
portfolio layer, but it is still an integrity receipt over reviewed anchors, not
an inline replay of LRAT, SMT, PB, graph-colouring, or Lorenz certificates.
Version 17 appends the promoted `systems::ball_fixed` five-step validated-ball
trajectory manifest as its own portfolio entry instead of folding it into the
older Lorenz point-trajectory entry. The new kind/checker
`lorenz_ball_fixed_trajectory` binds instance `236665224`, manifest certificate
`508132668`, and validated result into entry `590228875`; the v17 manifest is
`247161833`. The v17 result coverage is `575168545` over twenty-one entries
with seven UNSAT, twelve validated, one optimal, and one SAT result. The checker
family coverage is `805421150`, counting twelve Lorenz numeric/trajectory
entries while leaving SAT-family, SMT/QF_LRA, PB, and graph-colouring counts
unchanged. Acceptance, audit, and readiness receipts are respectively
`154169825`, `426158721`, and `879808511`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_v17_tiny.sio` and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_v17_imported.sio` mirror the
runtime/API split: the tiny gate is executable evidence for the v17 envelope,
while the imported gate frontend-checks `theorem::portfolio` and remains under
the current imported/native known-failure boundary. This is a portfolio
integrity upgrade over the validated-ball manifest, not a claim that the
portfolio layer replays every enclosure primitive internally. The same path is
now exposed as a consumer-facing systems helper,
`systems::ball_fixed::ball_fixed_lorenz_portfolio_v17_readiness_check`, which
takes the validated-ball trajectory manifest fields, recomputes manifest
`508132668`, validates the metadata, walks the public v17 portfolio envelope,
and returns readiness `879808511`. The imported smoke
`tests/run-pass/lorenz_ball_fixed_portfolio_v17_readiness_imported.sio`
frontend-checks this one-call API and keeps the same imported/native runtime
known-failure boundary.
Version 18 appends the first explicit SMT external-proof reconstruction
envelope. The new kind/checker pair `smt_external_proof` /
`smt_alethe_reconstruction` binds the existing tiny SMT assumption-core
fingerprints, formula `796206516`, assumptions `106325289`, and core
`666359709`, to an external-proof certificate fingerprint `770769284`. The
portfolio entry is `46718256`, the v18 manifest is `265423024`, result coverage
is `184063844` over twenty-two entries with eight UNSAT, twelve validated, one
optimal, and one SAT result, checker-family coverage is `22024619` with the SMT
family count raised to three, and acceptance/audit/readiness receipts are
`563343445`, `637718997`, and `958828933`. This is deliberately an Alethe-style
reconstruction receipt over a known tiny SMT core, not a claim that Sounio now
contains a full Alethe, LFSC, or Carcara implementation. The runtime evidence is
`tests/run-pass/solver_portfolio_smt_external_proof_v18_tiny.sio`; the public
API smoke is `tests/run-pass/solver_portfolio_smt_external_proof_v18_imported.sio`,
which frontend-checks `theorem::portfolio` and remains under the current
imported/native runtime known-failure boundary.
Version 19 appends the Lorenz ball-fixed radius-budget receipt as its own
portfolio entry. The new kind/checker pair `lorenz_ball_fixed_radius_budget`
binds trajectory manifest `508132668` and budget certificate `565673886` into
entry `295040814`; the v19 manifest is `592974800`. Result coverage is
`474634520` over twenty-three entries with eight UNSAT, thirteen validated, one
optimal, and one SAT result. Checker-family coverage is `17942049`, raising the
Lorenz family count to thirteen while preserving the SAT, SMT, PB, and graph
counts from v18. Acceptance, audit, and readiness receipts are respectively
`213285430`, `845530340`, and `865010181`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_radius_budget_v19_tiny.sio`
and `tests/run-pass/solver_portfolio_lorenz_ball_fixed_radius_budget_v19_imported.sio`
mirror the existing runtime/API split. This is portfolio integrity evidence that
the reviewed radius-quality receipt is now part of the composed checker
inventory; it is still not a long-time Lorenz theorem.

Version 20 appends a Lorenz wide-precision ladder receipt. The new
kind/checker pair `lorenz_wide_precision_ladder` binds the i128 fifth-step
validated-enclosure receipt `302145212`, the i256 five-step trajectory entry
`626160047`, and the ball-fixed radius-budget entry `295040814` into ladder
instance `725615043`, ladder certificate `455456906`, and portfolio entry
`124720146`; the v20 manifest is `158082390`. Result coverage is `2761010`
over twenty-four entries with eight UNSAT, fourteen validated, one optimal, and
one SAT result. Checker-family coverage is `251415300`, raising the Lorenz
family count to fourteen. Acceptance, audit, and readiness receipts are
respectively `128854088`, `610639541`, and `511521805`. The gates
`tests/run-pass/solver_portfolio_lorenz_wide_precision_ladder_v20_tiny.sio`
and `tests/run-pass/solver_portfolio_lorenz_wide_precision_ladder_v20_imported.sio`
mirror the runtime/API split. This is a composition receipt across i128, i256,
and fixed-radius ball evidence; it is not a shadowing theorem, a Taylor-model
integrator, or a claim of long-time Lorenz correctness. The same envelope is
now available as a systems-facing helper,
`systems::ball_fixed::ball_fixed_lorenz_portfolio_v20_precision_ladder_readiness_check`.
It recomputes the ball-fixed trajectory manifest `508132668`, radius-budget
receipt `565673886`, radius-budget entry `295040814`, and i128 enclosure receipt
`302145212` from supplied Lorenz fields before accepting the i256 trajectory
entry `626160047` and returning readiness `511521805`. The public smoke
`tests/run-pass/lorenz_ball_fixed_portfolio_v20_precision_ladder_imported.sio`
frontend-checks that one-call API and remains under the current imported/native
runtime known-failure boundary.

Version 21 appends the Lorenz ball-fixed step-policy receipt. The new
kind/checker pair `lorenz_ball_fixed_step_policy` binds trajectory manifest
`508132668`, radius-budget `565673886`, quality profile `70877465`, and local
step-policy fingerprint `231786611` into step-policy instance `668252674`,
certificate `702727574`, and portfolio entry `988622529`; the v21 manifest is
`999686282`. Result coverage is `807383802` over twenty-five entries with eight
UNSAT, fifteen validated, one optimal, and one SAT result. Checker-family
coverage is `761384846`, raising the Lorenz family count to fifteen. Acceptance,
audit, and readiness receipts are respectively `780868003`, `885886120`, and
`174001552`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_v21_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_v21_imported.sio`
mirror the runtime/API split. The systems-facing helper
`systems::ball_fixed::ball_fixed_lorenz_portfolio_v21_step_policy_readiness_check`
recomputes the ball-fixed trajectory manifest, radius budget, quality profile,
and step policy from supplied Lorenz fields before returning readiness
`174001552`; its public smoke is
`tests/run-pass/lorenz_ball_fixed_portfolio_v21_step_policy_imported.sio` and
remains under the imported/native runtime known-failure boundary. This is a
portfolio-level policy receipt over the five-step local smoke, not a general
adaptive-step controller.

Version 22 appends the Lorenz ball-fixed step-policy margin receipt. The new
kind/checker pair `lorenz_ball_fixed_step_policy_margin` binds quality profile
`70877465`, step-policy fingerprint `231786611`, and margin fingerprint
`494960746` into margin instance `377517990`, certificate `250048780`, and
portfolio entry `757892608`; the v22 manifest is `420758164`. Result coverage
is `191474584` over twenty-six entries with eight UNSAT, sixteen validated, one
optimal, and one SAT result. Checker-family coverage is `850822389`, raising the
Lorenz family count to sixteen. Acceptance, audit, and readiness receipts are
respectively `338097733`, `589698637`, and `356299987`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_margin_v22_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_margin_v22_imported.sio`
mirror the runtime/API split. This is portfolio integrity evidence that policy
headroom is now part of the composed checker inventory; it is still not an
adaptive-step controller or a long-time Lorenz theorem.

Version 23 appends the Lorenz ball-fixed guard-band step-policy decision. The
new kind/checker pair `lorenz_ball_fixed_step_policy_guard` binds margin
fingerprint `494960746` and local guard decision `833881722` into guard instance
`554569097`, certificate `414588992`, and portfolio entry `3812644`; the v23
manifest is `193159026`. Result coverage is `926894353` over twenty-seven
entries with eight UNSAT, seventeen validated, one optimal, and one SAT result.
Checker-family coverage is `291588905`, raising the Lorenz family count to
seventeen. Acceptance, audit, and readiness receipts are respectively
`975493822`, `718334294`, and `698832575`. The guard decision uses shrink
threshold `0` ppm, hold threshold `2000` ppm, and grow threshold `5000` ppm; the
current minimum policy margin is `1273` ppm, so the encoded action is
`hold_same_dt` in the `hold_thin` guard band with next-`dt` ratio `1/1`. The
gates `tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_guard_v23_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_step_policy_guard_v23_imported.sio`
mirror the runtime/API split. This is a proof-carrying conservative policy
decision over the five-step smoke, not evidence that a larger step is safe.

Version 24 appends the Lorenz ball-fixed full policy-chain receipt. The new
kind/checker pair `lorenz_ball_fixed_policy_chain` binds quality profile
`70877465`, step-policy `231786611`, margin `494960746`, guard decision
`833881722`, and local chain fingerprint `21862303` into chain instance
`703368760`, certificate `606714830`, and portfolio entry `777068644`; the v24
manifest is `654522473`. Result coverage is `351276693` over twenty-eight
entries with eight UNSAT, eighteen validated, one optimal, and one SAT result.
Checker-family coverage is `421318006`, raising the Lorenz family count to
eighteen. Acceptance, audit, and readiness receipts are respectively
`977323540`, `782172696`, and `327542064`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_policy_chain_v24_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_policy_chain_v24_imported.sio`
mirror the runtime/API split. This is portfolio integrity evidence for the
entire local quality-to-guard chain, not a long-time Lorenz theorem or a general
adaptive-step controller.

Version 25 appends the SAT FRAT-to-LRAT elaboration envelope. The new
kind/checker pair `sat_frat_elaboration` binds the FRAT manifest `975208343`
and final UNSAT envelope `86860040` into portfolio entry `423305090`; the v25
manifest is `857970677`. Result coverage is `145359005` over twenty-nine
entries with nine UNSAT, eighteen validated, one optimal, and one SAT result.
Checker-family coverage is `705272357`, raising the SAT family count to five.
Acceptance, audit, and readiness receipts are respectively `697797234`,
`136376970`, and `300977997`. The gates
`tests/run-pass/solver_portfolio_sat_frat_elaboration_v25_tiny.sio` and
`tests/run-pass/solver_portfolio_sat_frat_elaboration_v25_imported.sio` mirror
the runtime/API split. This is cross-checker inventory evidence for the tiny
FRAT elaboration hook, not a full FRAT parser or large-proof checker.

Version 26 appends the Lorenz i256 projection-inclusion bridge. The new
kind/checker pair `lorenz_i256_projection_inclusion` binds trajectory manifest
`214180161` and projection contract `607091799` into portfolio entry
`63872871`; the v26 manifest is `326517892`. Result coverage is `576925127`
over thirty entries with nine UNSAT, nineteen validated, one optimal, and one
SAT result. Checker-family coverage is `842185233`, raising the Lorenz family
count to nineteen. Acceptance, audit, and readiness receipts are respectively
`345898763`, `128651545`, and `242940443`. The gates
`tests/run-pass/solver_portfolio_lorenz_i256_projection_inclusion_v26_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_i256_projection_inclusion_v26_imported.sio`
mirror the runtime/API split. This is portfolio integrity evidence that the
cross-scale i256-to-ball inclusion is part of the checker inventory; it is not
a Taylor-model integrator or long-time Lorenz theorem.

`systems::ball_fixed` now exposes an explicit enclosure receipt layer. The
generic `ball_fixed_enclosure1_*` and `ball_fixed_enclosure3_*` helpers turn
`(scale, center, radius, lo, hi)` and 3D boxes into deterministic status masks
and fingerprints. The Lorenz-specialized
`ball_fixed_lorenz_trajectory5_final_enclosure_fingerprint` binds trajectory
manifest `508132668`, quality profile `70877465`, radius budget `565673886`,
and final 3D enclosure fingerprint `166945519` into receipt `114244647`. The
gates `tests/run-pass/lorenz_ball_fixed_explicit_enclosure_tiny.sio` and
`tests/run-pass/lorenz_ball_fixed_explicit_enclosure_imported.sio` mirror the
runtime/API split. This is a cleaner certificate boundary for the already-built
five-step Lorenz enclosure, not a new integration method.

The same enclosure receipt now has a public record schema:
`BallFixedEnclosure1` and `BallFixedEnclosure3` expose named fields for the
canonical scale, centers, radii, interval endpoints, status masks, and
fingerprints. The executable validation still flows through scalar field-wise
helpers such as `ball_fixed_enclosure1_fingerprint`,
`ball_fixed_enclosure3_fingerprint`, `ball_fixed_contains_value`, and
`ball_fixed_lorenz_trajectory5_final_enclosure_fingerprint`. The gates
`tests/run-pass/lorenz_ball_fixed_enclosure_record_tiny.sio` and
`tests/run-pass/lorenz_ball_fixed_enclosure_record_imported.sio` are
`check-only` schema/API gates because this wave exposed a current native
aggregate-lowering edge: record field reads and record-return constructors for
the wide enclosure shapes do not yet have reliable runtime evidence. This is a
typed data-boundary upgrade over the scalar receipt layer, not proof that wide
record passing is mature for numerical kernels.

Version 27 appends that explicit Lorenz enclosure receipt to the common solver
portfolio. The new kind/checker pair
`lorenz_ball_fixed_explicit_enclosure` binds instance `808477419` and
certificate `414432618` into portfolio entry `547731575`; the v27 manifest is
`291678314`. Result coverage is `505104449` over thirty-one entries with nine
UNSAT, twenty validated, one optimal, and one SAT result. Checker-family
coverage is `475711309`, raising the Lorenz family count to twenty. Acceptance,
audit, and readiness receipts are respectively `609703723`, `377961680`, and
`106882974`. The gates
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_explicit_enclosure_v27_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_ball_fixed_explicit_enclosure_v27_imported.sio`
mirror the runtime/API split. This is cross-checker inventory evidence for the
explicit final enclosure boundary, not a stronger numerical theorem.

The LRAT layer now also exposes a deletion/ID lifecycle receipt. The public
`lrat_deletion_lifecycle_*` helpers bind the tiny LRAT lifecycle where original
clauses 1 through 3 derive unit clause 4 and empty clause 5, while ID reuse is
blocked, deleted antecedents are accounted for, and the final empty clause
remains the UNSAT endpoint. The lifecycle manifest is `6950141`, the audit
receipt is `109354544`, and the final lifecycle UNSAT envelope is `9126320`.
The gates `tests/run-pass/lrat_deletion_lifecycle_receipt_tiny.sio` and
`tests/run-pass/lrat_deletion_lifecycle_receipt_imported.sio` mirror the
runtime/API split. This is a proof-kernel auditability upgrade for clause-ID and
deletion policy, not a full LRAT parser.

Version 28 appends that LRAT deletion/ID lifecycle receipt to the common solver
portfolio. The new kind/checker pair `lrat_deletion_lifecycle` binds instance
`6950141` and certificate `9126320` into portfolio entry `620031252`; the v28
manifest is `380302392`. Result coverage is `184362635` over thirty-two entries
with ten UNSAT, twenty validated, one optimal, and one SAT result.
Checker-family coverage is `644841534`, raising the SAT/LRAT family count to
six while leaving SMT, PB, graph, and Lorenz counts unchanged. Acceptance,
audit, and readiness receipts are respectively `790802629`, `16072096`, and
`413118064`. The gates
`tests/run-pass/solver_portfolio_lrat_deletion_lifecycle_v28_tiny.sio` and
`tests/run-pass/solver_portfolio_lrat_deletion_lifecycle_v28_imported.sio`
mirror the runtime/API split. This is cross-checker inventory evidence for LRAT
lifecycle policy; it is not a large-proof checker or a replacement for full
LRAT replay.

The Lorenz i256 layer now also exposes an explicit precision-bridge receipt
from the five-step i256 trajectory to the ball-fixed final enclosure. The public
`lorenz_i256_ball_fixed_bridge_*` helpers bind scale `2^32`, target scale
`1e6`, trajectory manifest `214180161`, projection inclusion `890086395`,
projection contract `607091799`, final enclosure receipt `114244647`, and
enclosure fingerprint `166945519` into bridge instance `149359034`,
certificate `138609006`, and audit receipt `488107916`. The gates
`tests/run-pass/lorenz_i256_ball_fixed_bridge_tiny.sio` and
`tests/run-pass/lorenz_i256_ball_fixed_bridge_imported.sio` mirror the
runtime/API split. This is the first compact certificate that says "this exact
i256 point trajectory projects into this already-audited ball-fixed enclosure";
it is not a new integrator, a shadowing proof, or a long-time Lorenz theorem.

Version 29 appends that Lorenz i256-to-ball-fixed bridge to the common solver
portfolio. The new kind/checker pair `lorenz_i256_ball_fixed_bridge` binds
instance `149359034` and audit certificate `488107916` into portfolio entry
`916832909`; the v29 manifest is `137630154`. Result coverage is `904709304`
over thirty-three entries with ten UNSAT, twenty-one validated, one optimal,
and one SAT result. Checker-family coverage is `70534950`, raising the Lorenz
family count to twenty-one while leaving SAT/LRAT, SMT, PB, and graph counts
unchanged. Acceptance, audit, and readiness receipts are respectively
`685946948`, `493502604`, and `182721636`. The gates
`tests/run-pass/solver_portfolio_lorenz_i256_ball_fixed_bridge_v29_tiny.sio`
and
`tests/run-pass/solver_portfolio_lorenz_i256_ball_fixed_bridge_v29_imported.sio`
mirror the runtime/API split. This is cross-scale certificate inventory
evidence for the existing five-step Lorenz anchors, not a stronger numerical
theorem.

Version 30 adds an Erdős/Hadwiger-Nelson scope-audit entry rather than a new
UNSAT theorem. The entry binds the current graph-colouring scope fingerprint
`37553010` to an audit fingerprint `180094659`: Hadwiger-Nelson is tracked here
as Erdős #508, Erdős #90 is the unit-distance count problem, and the current
`K3`/2-colour replay is not accepted as a `chi >= 6` preflight. The v30 entry
fingerprint is `786321129`, manifest `856410103`, result coverage `586508146`,
family coverage `938241679`, acceptance `909830054`, audit `703699366`, and
readiness `742145517`. These portfolio acceptance/audit/readiness values are
deterministic inventory checksums over already-scoped anchors; they are not a
meta-proof that the underlying mathematics is stronger than the individual
checker gates. The result inventory now has thirty-four entries: ten UNSAT
results, twenty-two validated results, one optimality result, and one SAT
result. Graph-family coverage rises to two because the portfolio now tracks both
the actual graph-colouring UNSAT smoke and the claim-boundary audit. This is
evidence hygiene for future plane-colouring work, not evidence for
`chi(R^2) >= 6`.

`systems::lorenz_i256_cert` now also exposes a five-step range-budget receipt
for the i256 Lorenz trajectory used by the projection bridge:
`lorenz_i256_trajectory5_range_budget_fingerprint` pins scale `2^32`, five
steps, final point `(5309941770,10058731223,4084837935)`, trajectory manifest
`214180161`, and chain anchor `23144051` to state bound `2^34`, derivative/
division-product bound `2^71`, signed-i256 limit exponent `255`, and headroom
`184`. The budget fingerprint is `87220662`, and the companion audit
fingerprint `323169628` binds it to bridge instance `149359034`, projection
inclusion `890086395`, and ball-fixed enclosure `166945519`. This is a
range/headroom certificate for the existing supplied quotient/remainder
trajectory, not a source-level `i256` division fix, not an adaptive integrator,
and not a long-time Lorenz theorem.

Version 31 appends that Lorenz i256 range-budget certificate to the portfolio.
The new kind/checker `lorenz_i256_range_budget` binds entry `116461241` into
manifest `628994509`; result coverage is `322111452` over thirty-five entries
with ten UNSAT, twenty-three validated, one optimality, and one SAT result.
Checker-family coverage is `379191739`, raising the Lorenz family count to
twenty-two while leaving SAT/LRAT, SMT, PB, and graph counts unchanged.
Acceptance, audit, and readiness checksums are respectively `978689863`,
`692196152`, and `376269457`. As with the earlier portfolio receipts, these are
inventory checksums over already-scoped anchors rather than a meta-proof.

The i256-to-ball-fixed bridge now also records projection roundoff. The helper
`lorenz_i256_projection_roundoff_budget_fingerprint` binds the projection
quotients/remainders `(1236317,687511168)`, `(2341980,3715113920)`, and
`(951075,1913956800)` from source scale `2^32` into target scale `1e6`, with
maximum floor error one target-scale unit, max remainder `3715113920`, lower
slack `42445`, upper slack after one-unit roundup `42450`, and inclusion mask
`7`. The roundoff budget fingerprint is `677514412`; audit fingerprint
`159826781` links it to range budget `87220662`, bridge instance `149359034`,
and ball-fixed enclosure `166945519`. This is a projection-loss receipt for the
existing certified point trajectory and enclosure bridge, not a new Lorenz
integrator, not a tighter enclosure, and not a continuous-time theorem.

Version 32 appends the projection-roundoff receipt to the portfolio. The new
kind/checker `lorenz_i256_projection_roundoff` binds entry `775818863` into
manifest `601279541`; result coverage is `257415384` over thirty-six entries
with ten UNSAT, twenty-four validated, one optimality, and one SAT result.
Checker-family coverage is `19842425`, raising the Lorenz family count to
twenty-three while leaving SAT/LRAT, SMT, PB, and graph counts unchanged.
Acceptance, audit, and readiness checksums are respectively `240886278`,
`666559707`, and `32963031`.

The bridge now has a second projection guard:
`lorenz_i256_projection_margin_budget_fingerprint` ties that one-unit target
scale roundoff to the existing radius and policy margins. It requires the
roundoff budget `677514412`, roundoff audit `159826781`, range budget
`87220662`, ball-fixed radius budget `565673886`, step-policy margin
`494960746`, and policy-chain receipt `21862303`. The checked derived margin is
lower slack `42445 - 1 = 42444`, upper slack after one-unit roundup `42450`,
minimum policy margin `1273`, inclusion mask `7`, and safety mask `31`. The
budget fingerprint is `69322697`; audit fingerprint `397379945` ties it back to
the ball-fixed enclosure `166945519`. This is only a finite projection-margin
receipt for the already logged five-step certificate; it is not a new numerical
method, not a global attractor statement, and not an independent proof of the
Lorenz dynamics.

Version 33 appends this projection-margin receipt to the portfolio. The new
kind/checker `lorenz_i256_projection_margin` binds entry `585753742` into
manifest `502768484`; result coverage is `121923227` over thirty-seven entries
with ten UNSAT, twenty-five validated, one optimality, and one SAT result.
Checker-family coverage is `589697029`, raising the Lorenz family count to
twenty-four while leaving SAT/LRAT, SMT, PB, and graph counts unchanged.
Acceptance, audit, and readiness checksums are respectively `632669050`,
`197884610`, and `593120310`.

The quotient/remainder checker now exposes a reusable projection form:
`div_witness_projection3_i256_fingerprint` verifies three supplied i256
projection divisions of the form `n * target_scale = q * source_scale + r`,
without invoking source-level wide division. The Lorenz wrapper
`lorenz_i256_projection_div_witness_fingerprint` instantiates it for the
five-step i256 final point `(5309941770,10058731223,4084837935)` from source
scale `2^32` into target scale `1e6`, producing generic projection witness
`440693751` and Lorenz projection-div-witness fingerprint `693793961`. The
audit fingerprint `659349019` binds that witness to roundoff budget `677514412`,
projection-margin budget `69322697`, projection inclusion `890086395`, and
readiness mask `31`. This makes the projection quotient/remainder proof a
first-class checker artifact rather than a hidden duplicate inside the Lorenz
bridge; it still does not fix native source-level i256 division.

Version 34 appends the projection-div-witness receipt to the portfolio. The new
kind/checker `lorenz_i256_projection_div_witness` binds entry `44126209` into
manifest `490367409`; result coverage is `72541052` over thirty-eight entries
with ten UNSAT, twenty-six validated, one optimality, and one SAT result.
Checker-family coverage is `245661608`, raising the Lorenz family count to
twenty-five while leaving SAT/LRAT, SMT, PB, and graph counts unchanged.
Acceptance, audit, and readiness checksums are respectively `570541615`,
`623438833`, and `772074709`.

The projection witness layer now also records directed rounding. The helper
`div_witness_projection3_i256_directed_round_fingerprint` first replays the
three floor quotient/remainder witnesses and then checks each supplied ceil
value by the rule `ceil = q` when `r = 0` and `ceil = q + 1` when `r > 0`. For
the Lorenz projection all three remainders are positive, so the certified ceil
triple is `(1236318,2341981,951076)` with round mask `7` and generic directed
rounding fingerprint `505163788`. The Lorenz wrapper
`lorenz_i256_projection_directed_rounding_fingerprint` binds that directed
rounding witness to projection-div-witness `693793961`, roundoff budget
`677514412`, and projection-margin budget `69322697`, producing fingerprint
`80894555`; audit fingerprint `793284661` binds the same anchors with readiness
mask `31`. This is a directed rounding certificate for the finite projection
used by the enclosure bridge, not a native wide-division fix or a stronger
continuous Lorenz theorem.

Version 35 appends the projection directed-rounding receipt to the portfolio.
The new kind/checker `lorenz_i256_projection_directed_rounding` binds entry
`297819393` into manifest `162915880`; result coverage is `708108430` over
thirty-nine entries with ten UNSAT, twenty-seven validated, one optimality, and
one SAT result. Checker-family coverage is `586575740`, raising the Lorenz
family count to twenty-six while leaving SAT/LRAT, SMT, PB, and graph counts
unchanged. Acceptance, audit, and readiness checksums are respectively
`747375279`, `90534165`, and `4774294`.

The projection witness layer now records interval containment as well:
`div_witness_projection3_i256_interval_containment_fingerprint` replays the
directed floor/ceil witness and checks that each rounded interval `[floor,ceil]`
is contained in the ball-fixed enclosure bounds. For the Lorenz projection the
axis slacks are x `(243367,243364)`, y `(319641,319638)`, and z
`(42445,42450)`, giving containment mask `7`, minimum lower slack `42445`, and
minimum upper slack `42450`. The generic interval-containment fingerprint is
`192258040`; `lorenz_i256_projection_interval_containment_fingerprint` binds it
to directed rounding `80894555`, projection inclusion `890086395`, and
projection-margin budget `69322697`, yielding `310218620`. Its audit fingerprint
is `501185861`. This receipt says the finite rounded projection interval is
inside the existing certified enclosure; it does not strengthen the underlying
dynamics or repair native wide division.

Version 36 appends the projection interval-containment receipt to the portfolio.
The new kind/checker `lorenz_i256_projection_interval_containment` binds entry
`189652817` into manifest `622328210`; result coverage is `130539653` over
forty entries with ten UNSAT, twenty-eight validated, one optimality, and one
SAT result. Checker-family coverage is `714353724`, raising the Lorenz family
count to twenty-seven while leaving SAT/LRAT, SMT, PB, and graph counts
unchanged. Acceptance, audit, and readiness checksums are respectively
`649832602`, `86586813`, and `19419291`.

The next layer is a projection certificate envelope rather than another
standalone checksum. `lorenz_i256_projection_certificate_envelope_fingerprint`
replays the five-step i256 trajectory certificate from the supplied final-step
quotient/remainder witnesses, then requires the projection inclusion
`890086395`, projection contract `607091799`, projection-div witness
`693793961`, directed rounding `80894555`, interval containment `310218620`,
interval audit `501185861`, bridge instance/certificate/audit
`149359034`/`138609006`/`488107916`, final enclosure receipt `114244647`,
ball-fixed enclosure `166945519`, range budget `87220662`, roundoff budget
`677514412`, projection-margin budget `69322697`, and portfolio-v36 readiness
`19419291`. The resulting envelope fingerprint is `77888110`; audit fingerprint
`245178976` binds the same anchors with readiness mask `31`. This is a
proof-carrying replay envelope for the finite five-step Lorenz projection chain,
not a native `i256` division fix, a new integrator, a shadowing proof, or a
continuous-time theorem.

Version 37 appends the projection certificate-envelope receipt to the portfolio.
The new kind/checker `lorenz_i256_projection_certificate_envelope` binds entry
`836439604` into manifest `163659586`; result coverage is `634889936` over
forty-one entries with ten UNSAT, twenty-nine validated, one optimality, and one
SAT result. Checker-family coverage is `924050761`, raising the Lorenz family
count to twenty-eight while leaving SAT/LRAT, SMT, PB, and graph counts
unchanged. Acceptance, audit, and readiness checksums are respectively
`284662936`, `559786522`, and `51601364`.

The certificate envelope is now backed by a small dependency-DAG receipt.
`lorenz_i256_projection_certificate_dependency_dag_fingerprint` checks that the
envelope `77888110`/`245178976` is rooted in the five-step trajectory manifest
`214180161` and in the predecessor portfolio entries/manifests for projection
inclusion (`63872871`/`326517892`), explicit enclosure
(`547731575`/`291678314`), bridge (`916832909`/`137630154`), range budget
(`116461241`/`628994509`), roundoff (`775818863`/`601279541`), margin
(`585753742`/`502768484`), projection-div witness (`44126209`/`490367409`),
directed rounding (`297819393`/`162915880`), interval containment
(`189652817`/`622328210`), and envelope v37 (`836439604`/`163659586`). The DAG
records seven layers, ten dependencies, and complete dependency mask `127`,
yielding fingerprint `713010204`; audit fingerprint `725222703` binds the same
entry-level ancestry. This is a certificate-graph consistency layer over the
finite Lorenz projection smoke, not a proof of acyclicity for arbitrary
portfolio graphs and not a stronger dynamics theorem.

Version 38 appends the projection dependency-DAG receipt to the portfolio. The
new kind/checker `lorenz_i256_projection_dependency_dag` binds entry
`389121447` into manifest `731723710`; result coverage is `165972953` over
forty-two entries with ten UNSAT, thirty validated, one optimality, and one SAT
result. Checker-family coverage is `160480532`, raising the Lorenz family count
to twenty-nine while leaving SAT/LRAT, SMT, PB, and graph counts unchanged.
Acceptance, audit, and readiness checksums are respectively `686410113`,
`87146991`, and `845057161`.

Version 39 adds a meta-level SOTA-alignment receipt rather than another Lorenz
certificate. `solver_portfolio_v39_sota_alignment_fingerprint` binds the local
SAT/LRAT/FRAT axis (`53646245`, `423305090`, `620031252`), PB/VeriPB-shaped axis
(`258927862`, `438043475`), SMT/Alethe-shaped axis (`46718256`), and Lorenz
certificate-DAG axis (`389121447`, manifest `731723710`) into alignment
fingerprint `74298281`; audit fingerprint `963677367` binds the same axes with
alignment mask `31`. The new kind/checker `solver_sota_alignment` is counted as
a meta-family entry, not as Lorenz. Portfolio v39 binds entry `528140838` into
manifest `409028717`; result coverage is `806296867` over forty-three entries
with ten UNSAT, thirty-one validated, one optimality, and one SAT result.
Checker-family coverage is `933171094` with SAT/LRAT `6`, SMT `3`, PB `2`,
graph `2`, Lorenz `29`, and meta `1`. Acceptance, audit, and readiness
checksums are respectively `216209469`, `832122075`, and `812428526`. This is a
research-alignment receipt over local proof-checker architecture; it is not a
claim that Sounio now implements full LRAT/LPR, VeriPB/CakePB, Alethe/Carcara,
or rigorous long-time Lorenz validation.

Version 40 turns that alignment into a small proof-trace interop receipt.
`solver_portfolio_v40_proof_trace_interop_fingerprint` binds the SAT proof
trace anchors (`53646245`, `423305090`, `620031252`), PB anchors
(`258927862`, `438043475`), SMT/Alethe anchor (`46718256`), the Erdős scope
guard (`786321129`), the Lorenz certificate-DAG entry (`389121447`), and the
v39 alignment entry/manifest (`528140838`, `409028717`). The receipt records
six external trace-format directions (LRAT/LPR, FRAT, VeriPB, CakePB kernel,
Alethe, and Sounio certificate DAG ingestion policy), eight local checker
anchors, and complete interop mask `63`, yielding fingerprint `90120367`.
Audit fingerprint `336688074` additionally requires the v39 readiness
`812428526` and the Lorenz dependency-DAG readiness `845057161`. Portfolio v40
binds entry `15302992` into manifest `788938861`; result coverage is
`149225904` over forty-four entries with ten UNSAT, thirty-two validated, one
optimality, and one SAT result. Checker-family coverage is `166966176` with
SAT/LRAT `6`, SMT `3`, PB `2`, graph `2`, Lorenz `29`, and meta `2`.
Acceptance, audit, and readiness checksums are respectively `662126478`,
`255705570`, and `983550353`. This is an ingestion and trace-accounting layer:
future producers may emit rich FRAT/VeriPB/Alethe-like objects, but Sounio's
trusted surface is still the smaller checked kernels and receipts listed here.
It does not claim a full parser, a full external checker clone, or a new Erdős
or Lorenz theorem.

Version 41 adds the first PB kernel-trace receipt after the v40 interop layer.
`pb_kernel_trace_propagation_fingerprint3` in `theorem::pb` checks a bounded
VeriPB/CakePB-shaped kernel trace over three Boolean variables: saturation of a
`<=` row, sound non-negative division, same-sense row addition, and a forced
literal propagation step. The concrete checked trace starts from
`7*x0 + 5*x1 + 2*x2 <= 6`, saturates it to
`6*x0 + 5*x1 + 2*x2 <= 6`, divides by `2` to
`3*x0 + 2*x1 + x2 <= 3`, adds `x0 + 2*x1 <= 2`, and derives that `x1`
is forced false under assignment `(x0=true, x1=unknown, x2=true)`. The kernel
certificate fingerprint is `478581235`; audit fingerprint is `827870314`.
`solver_portfolio_v41_pb_kernel_trace_fingerprint` binds that PB kernel trace
to the existing PB row-chain/optimality entries and the v40 interop manifest,
yielding receipt `666560586` and audit `879678645`. Portfolio v41 binds entry
`752808158` into manifest `379464486`; result coverage is `702770436` over
forty-five entries with ten UNSAT, thirty-three validated, one optimality, and
one SAT result. Checker-family coverage is `141353377`, raising PB family count
to `3` while leaving SAT/LRAT `6`, SMT `3`, graph `2`, Lorenz `29`, and meta
`2`. Acceptance, audit, and readiness checksums are respectively `707086658`,
`860179250`, and `801644191`. This is still a bounded three-variable kernel
trace, not a full VeriPB parser, not full strengthening/redundance support, and
not a proof that arbitrary PB certificates are accepted.

Version 42 adds the corresponding SMT/Alethe micro-reconstruction receipt. The
new `smt_alethe_assumption_core_reconstruction_fingerprint` in `theorem::smt`
does not merely checksum an external label: it replays the bounded
assumption-core check for the base formula `x OR y` under assumptions `not x`
and `not y`, then records an Alethe-shaped trace summary with two assumption
steps, one input clause, one resolution step, and final empty clause. The local
reconstruction fingerprint is `580810113`; its audit fingerprint is `72583387`
and links back to the v18 external-proof certificate `770769284`.
`solver_portfolio_v42_smt_alethe_micro_reconstruction_fingerprint` binds this
micro-reconstruction to the v18 SMT external-proof entry/manifest, the v40
interop entry, and the v41 PB kernel-trace entry, yielding receipt `550418014`
and audit `181685497`. Portfolio v42 binds entry `868378560` into manifest
`977002972`; result coverage is `263327815` over forty-six entries with ten
UNSAT, thirty-four validated, one optimality, and one SAT result.
Checker-family coverage is `4917294`, raising SMT family count to `4` while
leaving SAT/LRAT `6`, PB `3`, graph `2`, Lorenz `29`, and meta `2`.
Acceptance, audit, and readiness checksums are respectively `750750705`,
`486030685`, and `584513793`. This is a bounded reconstruction witness over a
single assumption-core example; it is not a full Alethe parser, not full theory
lemma reconstruction, not Carcara, and not proof of arbitrary SMT certificates.

`tests/run-pass/lorenz_i128_fixed_step.sio` adds the first high-precision Lorenz
smoke gate over source-level `i128` fixed-point arithmetic. It checks one Euler
step at `(1,1,1)` with scale `1e6`, `dt=0.01`, `sigma=10`, `rho=28`, and
`beta=8/3`, expecting:

- `dx = 0`;
- `dy = 26000000`;
- `dz = -1666666`;
- next state `(1000000, 1260000, 983334)`.

The gate intentionally avoids passing `i128` values across user-defined function
boundaries because smoke tests in this wave showed that source-level wide-int
function ABI still needs a separate repair gate. It also avoids signed negative
wide division by computing the negative `z` step as a positive drop and
subtracting it; a dedicated signed-wide-div regression should be added before
building a public Lorenz fixed-point API.

`tests/run-pass/lorenz_i128_ball_step.sio` extends that point smoke into a first
validated-enclosure shape. The Lorenz center step remains `i128`, while the
one-step radii are carried as `i64` because they are tiny and this avoids the
current helper-heavy `i128` ABI boundary. For initial radius `10` at scale
`1e6`, the gate checks:

- derivative radii `dx_rad=200`, `dy_rad=291`, `dz_rad=49`;
- next-state radii `nx_rad=13`, `ny_rad=13`, `nz_rad=11`;
- the same midpoint as the point-valued smoke:
  `(1000000, 1260000, 983334)`.

This is still an Euler smoke, not a rigorous long-time Lorenz theorem. The
important upgrade is the data shape: `(midpoint, radius)` with explicit
rounding margin, instead of a bare point trajectory.

`tests/run-pass/lorenz_ball_fixed_primitives_tiny.sio` adds the first ABI-safe
primitive layer behind that shape. It keeps helper signatures in `i64` and
checks the reusable radius/status operations that a future `systems::ball_fixed`
module needs before exposing a public wide-int API:

- nonnegative floor/ceil division guards;
- product-radius propagation with an explicit outward rounding unit;
- tight and guarded `beta*z` radius rules;
- Euler radius propagation with an explicit time-step guard;
- bound projection from `(center, radius)` to `[lo, hi]`;
- status checks for negative radius, invalid scale, and radius limit overflow.

The gate deliberately does not pass `i128` centers through user-defined helper
functions. It is a staging contract for API shape and rounding discipline while
`wide_i128_fn_abi_known_failure.sio` remains pinned.

`stdlib/systems/ball_fixed.sio` promotes that staging contract into a small
library surface without touching the legacy `systems::lib` f64 demos. The module
exports i64-only helpers for status checks, nonnegative floor/ceil division,
product-radius outward rounding, tight/guarded `beta*z` radius bounds,
Euler-radius propagation, bound projection, and Lorenz-specific derivative
radius helpers. It intentionally does not expose a public `(center_i128,
radius_i64)` aggregate yet, because wide-int values are still unsafe across
user-defined call boundaries.

The same module now also carries the first reusable LTE arithmetic hooks:
bounded `dt^2` and `2*scale^2` helpers, `ceil(h^2*|x''|/(2*scale^2))` radius
calculation, Euler-radius-plus-LTE accumulation, absolute interval endpoint
selection, and Lorenz-shaped absolute derivative/second-derivative bounds for
the first validated-step tube. The overflow contract is intentionally narrow:
the helpers are built for the current scale-`1e6`, `dt=0.01` gates and return
`-1` outside their bounded staging envelope. This is an API foothold for
validated integration, not a general arbitrary-precision Taylor engine.

`systems::ball_fixed` now also has a minimal enclosure-certificate surface:
`ball_fixed_bounds_match`, `ball_fixed_contains_value`,
`ball_fixed_contains_ball`, `ball_fixed_enclosure_from_euler_lte`, and
`ball_fixed_validated_step3`. These helpers let a caller check that a reported
`[lo, hi]` really matches `(center, radius)`, that a final LTE-enlarged ball
contains the smaller Euler ball, that boundary points are included while outside
points are rejected, and that the x/y/z components of one validated step agree
as a single certificate. `ball_fixed_validated_step3_fingerprint` adds a
deterministic non-cryptographic anchor for that certificate; the first Lorenz
validated-step gate pins fingerprint `242670977`.
`ball_fixed_validated_step3_chain_fingerprint` then adds the proof-log style
chain anchor over previous anchor, step index, scale, `dt`, and step
fingerprint; the first link at scale `1e6`, `dt=0.01` pins `447483934`. This
moves the Lorenz lane from "compute a radius" toward "check a returned
enclosure contract", still with i64 staging signatures.

`tests/run-pass/lorenz_ball_fixed_chain_five_step_tiny.sio` extends the scalar
mirror of that contract across the five explicit validated Lorenz steps. It pins
the step fingerprints `(242670977, 649286198, 250702467, 554782556, 236665224)`
and the chained anchors `(447483934, 781811503, 265719931, 83984861,
484010018)`, with negative controls for altered bounds and broken chain
metadata. This is a certificate-chain smoke gate over existing split-step data,
not a long-time Lorenz theorem or a replacement for a future Taylor-model
integrator.

`tests/run-pass/lorenz_ball_fixed_trajectory_manifest_tiny.sio` then binds that
five-step chain to a trajectory-level manifest: initial center `(1,1,1)` at
scale `1e6`, `dt=0.01`, five steps, chain anchor `484010018`, final step
fingerprint `236665224`, and final enclosure
`x in [992950,1479682]`, `y in [2022339,2661619]`,
`z in [908630,993526]`. The manifest fingerprint is `508132668`, and negative
controls reject changed chain anchors, changed step counts, changed final-step
fingerprints, and inconsistent final bounds. This is a portable manifest for a
short validated trajectory smoke, still not a long-time Lorenz theorem.
The manifest helper is now also promoted into `systems::ball_fixed` as
`ball_fixed_lorenz_trajectory5_manifest_fingerprint` plus
`ball_fixed_lorenz_trajectory5_manifest_valid`, preserving the same public
manifest fingerprint `508132668` and metadata checks for scale, `dt`, step
count, chain anchor, and final-step fingerprint. The imported smoke
`tests/run-pass/lorenz_ball_fixed_trajectory_manifest_imported.sio`
frontend-checks that reusable API while retaining the current imported/native
runtime known-failure boundary. This reusable API is the source anchor for the
v17 portfolio entry above, avoiding duplicated manifest logic in the portfolio
consumer gate.
The trajectory surface now also has a first radius-quality receipt:
`ball_fixed_lorenz_trajectory5_radius_budget_fingerprint`. For the same
five-step manifest, it checks final radii `(243366,319640,42448)` against an
explicit budget `(250000,325000,50000)`, records slacks `(6634,5360,7552)`,
and pins the budget fingerprint `565673886`. This is a compact tube-quality
check over the validated-ball smoke, not a proof of long-time Lorenz shadowing
or a Taylor-model integrator. Runtime evidence lives in
`tests/run-pass/lorenz_ball_fixed_radius_budget_tiny.sio`; the public imported
API smoke is `tests/run-pass/lorenz_ball_fixed_radius_budget_imported.sio` and
keeps the current imported/native known-failure boundary.
The next quality layer is
`ball_fixed_lorenz_trajectory5_quality_profile_fingerprint`. It binds manifest
`508132668` and radius-budget `565673886` to a profile fingerprint
`70877465`, checking radius sum `605454`, max/min radii `319640/42448`,
integer aspect quotient/remainder `7/22504`, cap fill `968726` ppm, slack
`31273` ppm, widths `(486732,639280,84896)`, and integer box volume
`26416072366172160`. This is an enclosure-quality diagnostic for future
step-size and wrapping-control work; it is not yet an adaptive integrator.
Runtime evidence is `tests/run-pass/lorenz_ball_fixed_quality_profile_tiny.sio`;
the public imported smoke is
`tests/run-pass/lorenz_ball_fixed_quality_profile_imported.sio` and keeps the
same imported/native known-failure boundary.
That profile now feeds a first local step-policy receipt:
`ball_fixed_lorenz_trajectory5_step_policy_fingerprint`. For profile
`70877465`, thresholds fill `<=970000` ppm, slack `>=30000` ppm, aspect
quotient `<=8`, and box volume `<=27000000000000000` produce policy mask `15`,
action `accept_same_dt`, and policy fingerprint `231786611`. Runtime evidence is
`tests/run-pass/lorenz_ball_fixed_step_policy_tiny.sio`; the public imported
smoke is `tests/run-pass/lorenz_ball_fixed_step_policy_imported.sio` and remains
under the imported/native known-failure boundary. This is a local policy receipt
for the five-step smoke, not a general adaptive-step controller.
The follow-on margin receipt
`ball_fixed_lorenz_trajectory5_step_policy_margin_fingerprint` records the
actual headroom under that policy: fill margin `1274` ppm, slack margin `1273`
ppm, aspect margin `1`, volume margin `583927633827840`, margin mask `15`, and
margin fingerprint `494960746`. Runtime evidence is
`tests/run-pass/lorenz_ball_fixed_step_policy_margin_tiny.sio`; the public
imported smoke is
`tests/run-pass/lorenz_ball_fixed_step_policy_margin_imported.sio` and remains
under the imported/native known-failure boundary. This adds auditable policy
headroom for future step-size control, not a proof that a larger step is safe.
That headroom now feeds a guard-band decision receipt,
`ball_fixed_lorenz_trajectory5_step_policy_guard_decision_fingerprint`.
With margin fingerprint `494960746`, minimum ppm margin `1273`, hold threshold
`2000` ppm, and grow threshold `5000` ppm, the decision fingerprint is
`833881722`; it encodes `hold_same_dt` in a `hold_thin` guard band with
next-`dt` ratio `1/1`. Runtime evidence is
`tests/run-pass/lorenz_ball_fixed_step_policy_guard_tiny.sio`; the public
imported smoke is
`tests/run-pass/lorenz_ball_fixed_step_policy_guard_imported.sio` and remains
under the imported/native known-failure boundary. This records a conservative
decision over the local smoke, not a general adaptive controller.

That guard decision now feeds
`ball_fixed_lorenz_trajectory5_policy_chain_fingerprint`, binding quality
profile `70877465`, step-policy `231786611`, margin `494960746`, and guard
decision `833881722` into chain fingerprint `21862303`. The receipt records four
stages, policy/margin masks `15/15`, action `hold_same_dt`, guard band
`hold_thin`, and next-`dt` ratio `1/1`. Runtime evidence is
`tests/run-pass/lorenz_ball_fixed_policy_chain_tiny.sio`; the public imported
smoke is `tests/run-pass/lorenz_ball_fixed_policy_chain_imported.sio` and
remains under the imported/native known-failure boundary. This closes the local
quality-to-guard audit chain for the five-step smoke, not a general adaptive
controller.

`tests/run-pass/lorenz_ball_fixed_stdlib_tiny.sio` is the desired imported gate
for `use systems::ball_fixed::*`. It frontend-checks, but runtime is marked
known-failure because current imported/native lowering exits 139. The executable
evidence for the same arithmetic remains the scalar mirror
`lorenz_ball_fixed_primitives_tiny.sio`, with the LTE/second-derivative numbers
covered by `lorenz_ball_fixed_lte_primitives_tiny.sio` and the enclosure
containment contract covered by `lorenz_ball_fixed_enclosure_primitives_tiny.sio`.

`tests/run-pass/lorenz_i128_euler_lte_seed.sio` and
`tests/run-pass/lorenz_i128_euler_lte_z_seed.sio` add the first local
truncation-error budget seed for that same initial point. The split is
intentional: the current native backend still exits 139 when the x/y and z
wide second-derivative expressions stay live in one `main`. Together the two
gates compute the instantaneous Lorenz second derivative in fixed-point scale:

- `|x''| = 260000000`;
- `|y''| = 24333334`;
- `|z''| = 30444442`.

With `h = 0.01`, it checks the integer ceiling of `0.5*h^2*|state''|`,
yielding LTE radii `(13000, 1217, 1523)` in scale-`1e6` units. The dynamic
derivatives are computed in `i128`, while the final LTE ceiling is carried in
`i64` because the values fit and the current native path still dislikes the
large-denominator `i128` division shape. Adding those LTE radii to the
one-step ball radii from `lorenz_i128_ball_step.sio` gives `(13013, 1230,
1534)`. This is deliberately recorded as an LTE seed, not a validated
integrator theorem: a full proof still needs a supremum bound for `state''`
over the whole step enclosure, not only the initial-point value.

`tests/run-pass/lorenz_i128_euler_lte_interval_xy.sio` and
`tests/run-pass/lorenz_i128_euler_lte_interval_z.sio` take the next step:
they bound `state''` over the conservative box covering both the initial ball
and the first Euler ball:

- `x in [999987, 1000013]`;
- `y in [999990, 1260013]`;
- `z in [983323, 1000010]`.

Using termwise absolute-value interval bounds, they check:

- derivative bounds over the box `|f| <= (22600260, 28277042, 3926724)`;
- second-derivative bounds `|state''| <= (508773020, 642787743, 67225296)`;
- interval LTE radii `(25439, 32140, 3362)`;
- one-step ball radii plus interval LTE `(25452, 32153, 3373)`.

These bounds are intentionally conservative and still use scalar `i64` for the
budget arithmetic because the interval bounds fit easily. The purpose is to
move from pointwise LTE arithmetic toward a validated-step shape without
claiming a full Taylor-model or long-time Lorenz certificate yet.

`tests/run-pass/lorenz_i128_validated_first_step.sio` folds the split facts above
into a single first-step enclosure contract. It consumes the `i128` Euler center
already certified by `lorenz_i128_ball_step.sio`, combines the Euler ball radii
`(13, 13, 11)` with the interval LTE radii `(25439, 32140, 3362)`, and checks the
final conservative first-step radii:

- `(25452, 32153, 3373)`;
- final enclosure bounds:
  `x in [974548, 1025452]`,
  `y in [1227847, 1292153]`,
  `z in [979961, 986707]`.

This is the first gate that reads like a validated-step certificate rather than
separate arithmetic witnesses. It intentionally stays scalar because a monolithic
file that recomputed the full `i128` center plus interval LTE budget in one
`main` hit the current native IR-lowering boundary. Its scope is still one Euler
step over the stated box, with a termwise interval LTE bound; it is not yet a
reusable integrator API or a long-time Lorenz shadowing certificate.

`tests/run-pass/lorenz_i128_validated_second_step.sio` carries that validated
first-step enclosure forward instead of restarting from the smaller Euler-only
ball. It consumes the step-2 center already certified by
`lorenz_i128_ball_two_step.sio`, propagates the first validated radii
`(25452, 32153, 3373)` through one Euler vector-field evaluation, and checks:

- second Euler-ball radii from the validated input: `(31213, 39386, 4114)`;
- a conservative second-step LTE tube covering
  `x in [974548, 1057213]`,
  `y in [1227847, 1556952]`,
  `z in [965598, 986707]`;
- derivative bounds over that tube `(26141650, 30138074, 4277249)`;
- second-derivative bounds `(562797240, 741383914, 83969657)`;
- second-step LTE radii `(28140, 37070, 4199)`;
- final second-step radii `(59353, 76456, 8313)`;
- final second-step enclosure bounds:
  `x in [966647, 1085353]`,
  `y in [1441110, 1594022]`,
  `z in [961399, 978025]`.

This is now a two-step validated-enclosure shape, but still in split gate form.
The center arithmetic remains in the `i128` ball-transition gates; the validated
contracts stay scalar because monolithic center+LTE files currently hit native
IR-lowering boundaries.

`tests/run-pass/lorenz_i128_validated_third_step.sio` extends that chain by
carrying the second validated enclosure forward through the third center already
certified by `lorenz_i128_ball_three_step.sio`. It starts from second-step
validated radii `(59353, 76456, 8313)` and checks:

- third Euler-ball radii from the validated input: `(72934, 93355, 10266)`;
- a conservative third-step LTE tube covering
  `x in [966647, 1148090]`,
  `y in [1441110, 1873076]`,
  `z in [949158, 978025]`;
- derivative bounds over that tube `(30211660, 32929878, 4758527)`;
- second-derivative bounds `(631415380, 855643938, 107084606)`;
- third-step LTE radii `(31571, 42783, 5355)`;
- final third-step radii `(104505, 136138, 15621)`;
- final third-step enclosure bounds:
  `x in [970651, 1179661]`,
  `y in [1643583, 1915859]`,
  `z in [943803, 975045]`.

The Lorenz lane now has two parallel stories: small Euler-only ball propagation
through split transitions, and a more honest validated chain that accumulates
LTE/error radii. The latter is closer to a real validated integrator, while
still being deliberately split and scalar at the contract layer.

`tests/run-pass/lorenz_i128_validated_fourth_step.sio` brings the validated chain
to the same four-center horizon as the Euler-only split gates. It carries the
third validated radii `(104505, 136138, 15621)` through the fourth center already
certified by `lorenz_i128_ball_four_step.sio`, and checks:

- fourth Euler-ball radii from the validated input: `(128570, 165943, 19504)`;
- a conservative fourth-step LTE tube covering
  `x in [970651, 1274182]`,
  `y in [1643583, 2218595]`,
  `z in [933471, 975045]`;
- derivative bounds over that tube `(34927770, 36706280, 5427014)`;
- second-derivative bounds `(716340500, 988994784, 138733096)`;
- fourth-step LTE radii `(35818, 49450, 6937)`;
- final fourth-step radii `(164388, 215393, 26441)`;
- final fourth-step enclosure bounds:
  `x in [981224, 1310000]`,
  `y in [1837259, 2268045]`,
  `z in [926534, 979416]`.

`tests/run-pass/lorenz_i128_validated_fifth_step.sio` extends the same checked
enclosure-smoke contract one more step. It carries the fourth accumulated radii
`(164388, 215393, 26441)` through the fifth center already certified by
`lorenz_i128_ball_five_step.sio`, and checks:

- fifth Euler-ball radii from the accumulated input: `(202367, 262356, 33343)`;
- a termwise fifth-step LTE tube covering
  `x in [981224, 1438683]`,
  `y in [1837259, 2604335]`,
  `z in [917735, 984421]`;
- derivative bounds over that tube `(40430180, 41567130, 6371936)`;
- second-derivative bounds `(819973100, 1145675176, 182087487)`;
- fifth-step LTE radii `(40999, 57284, 9105)`;
- final fifth-step radii `(243366, 319640, 42448)`;
- final fifth-step enclosure bounds:
  `x in [992950, 1479682]`,
  `y in [2022339, 2661619]`,
  `z in [908630, 993526]`.

The checked enclosure chain now spans five explicit steps at fixed-point scale
`1e6`. That is still far from a Lorenz theorem: no adaptive step selection, no
Taylor-model remainder, no invariant set, no separate monotonicity/wrapping proof
for the second-derivative envelope, and no shadowing/long-time guarantee is
claimed. It is, however, a real accumulation-of-error spine rather than a set of
isolated point simulations.

`tests/run-pass/lorenz_i128_ball_two_step.sio` adds the second transition in
that same shape. It starts from the step-1 ball certified by
`lorenz_i128_ball_step.sio` and checks the step-2 center and radii:

- derivative center `(2600000, 25756666, -1362224)`;
- next center `(1026000, 1517566, 969712)`;
- derivative radii `dxr=260`, `dyr=376`, `dzr=61`;
- next-state radii `(16, 17, 12)`.

`tests/run-pass/lorenz_i128_ball_three_step.sio` extends the same certified
shape to a third transition. It starts from the step-2 ball certified by
`lorenz_i128_ball_two_step.sio` and checks:

- derivative center `(4915660, 26215509, -1028876)`;
- next center `(1075156, 1779721, 959424)`;
- derivative radii `dxr=330`, `dyr=462`, `dzr=74`;
- next-state radii `(20, 22, 13)`.

`tests/run-pass/lorenz_i128_ball_four_step.sio` adds the fourth split
transition. It starts from the step-3 ball certified by
`lorenz_i128_ball_three_step.sio` and checks:

- derivative center `(7045650, 27293116, -644987)`;
- next center `(1145612, 2052652, 952975)`;
- derivative radii `dxr=420`, `dyr=577`, `dzr=95`;
- next-state radii `(25, 28, 14)`.

`tests/run-pass/lorenz_i128_ball_five_step.sio` adds the fifth split
transition. It starts from the step-4 ball certified by
`lorenz_i128_ball_four_step.sio` and checks:

- derivative center `(9070400, 28932744, -189724)`;
- next center `(1236316, 2341979, 951078)`;
- derivative radii `dxr=530`, `dyr=721`, `dzr=122`;
- next-state radii `(31, 36, 16)`.

An earlier all-in-one file that recomputed multiple transitions in one `main`
typechecked but the emitted ELF failed. The current split gates are a deliberate
compiler-aware shape: they prove chained ball propagation through five split
transitions without pretending the current wide-int backend can carry arbitrary
helper-heavy `i128` programs yet.

`tests/run-pass/lorenz_i256_derivative_num_smoke.sio` opens the `i256` Lorenz
lane with a deliberately narrower passing gate. At fixed-point scale
`S = 2^32`, with `(x,y,z)=(S,S,S)`, `sigma=10`, `rho=28`, and `beta=8/3`, it
checks only the pre-division numerator identities:

- `rho*S - z = 27S`;
- `x*y = S^2`;
- `x*(rho*S-z) = 27S^2`;
- `8*z = 8S`;
- `27S^2 > S^2 > S`.

This gate proves that source-level `i256` can carry Lorenz-shaped numerator
products in the native path. `tests/run-pass/lorenz_i256_numerator_manifest_tiny.sio`
then turns those same pre-division identities into a certificate manifest. It
checks the five flags above at runtime, pins instance fingerprint `300929056`,
certificate fingerprint `572148472`, and rejects a changed identity flag or
wrong scale metadata. It intentionally does not claim a full Euler step,
validated enclosure, or long-time theorem.

`tests/run-pass/lorenz_i256_bit_budget_tiny.sio` adds the next `i256` gate:
not another numerical trajectory, but a bit-budget certificate for the same
pre-division products. It checks that for `S=2^32`, `S^2` uses exponent `64`,
the conservative `27S^2` numerator bound stays below `2^69`, `8S` uses exponent
`35`, and the largest numerator still has `186` exponent bits of headroom below
the signed-positive `i256` limit exponent `255`. The gate executes the same
`i256` numerator products, pins range instance fingerprint `743388522` and range
certificate fingerprint `133303313`, and rejects a changed exponent bound or
wrong range metadata. This is overflow-budget evidence for the pre-division
smoke, not a fix for the division XFAILs below.

`tests/run-pass/lorenz_i256_division_witness_tiny.sio` adds the first passing
division-shaped `i256` Lorenz gate without invoking source-level `i256`
division. Instead, every division is represented as a quotient/remainder witness
checked by `num = q*den + r` and `0 <= r < den` using `i256` multiplication,
addition, and comparison. At `S=2^32`, the gate checks:

- `S / 100 = 42949672` remainder `96` for `dt`;
- `27S^2 / S = 27S` remainder `0` for the `dy` numerator;
- `S^2 / S = S` remainder `0`;
- `8S / 3 = 11453246122` remainder `2` for the `beta*z` term;
- `dt*(26S) / S = 1116691472` remainder `0` for the `y` increment;
- `dt*(11453246122-S) / S = 71582786` remainder `2834678416` for the positive
  `z` drop.

The resulting one-step witness has `x'=4294967296`, `y'=5411658768`, and
`z'=4223384510`, pins instance fingerprint `660429472` and certificate
fingerprint `510854875`, and rejects bad remainders, bad quotients, changed
certificate metadata, and wrong instance metadata. This is not a source-level
division fix and not a validated Lorenz enclosure. It is a checker-friendly
bridge showing that the `i256` arithmetic needed to verify divisions can already
execute if the quotient/remainder pair is supplied as certificate data.

`tests/run-pass/lorenz_i256_division_chain_two_step_tiny.sio` extends that
division-witness pattern to a two-step chain. Starting from the one-step state
above, it checks the second step by quotient/remainder witnesses:

- `dy` numerator quotient `116035699778`, remainder `0`;
- `xy` quotient `5411658768`, remainder `0`;
- `8z/3` quotient `11262358693`, remainder `1`;
- `x` increment quotient `111669144`, remainder `3023657216`;
- `y` increment quotient `1106240385`, remainder `1604599760`;
- `z` drop quotient `58506997`, remainder `4047004488`.

The second witness lands at `x=4406636440`, `y=6517899153`, `z=4164877513`,
pins step-2 certificate fingerprint `899209716`, and chains with the first-step
certificate into anchor `335080767`. This is a stronger dynamic smoke than the
single-step witness, but it is still a quotient/remainder certificate chain,
not native `i256` division, not an adaptive integrator, and not a Lorenz
shadowing theorem.

`tests/run-pass/lorenz_i256_division_chain_three_step_tiny.sio` extends the
same quotient/remainder certificate chain to a third Euler step. Starting from
the two-step state above, it checks:

- `dy` numerator quotient `119112655997`, remainder `796606888`;
- `xy` quotient `6687364522`, remainder `1434262808`;
- `8z/3` quotient `11106340034`, remainder `2`;
- `x` increment quotient `211126266`, remainder `2495204624`;
- `y` increment quotient `1125947543`, remainder `1173001440`;
- `z` drop quotient `44189754`, remainder `568146880`.

The third witness lands at `x=4617762706`, `y=7643846696`, `z=4120687759`,
pins step-3 certificate fingerprint `603078026`, and chains all three
certificates into anchor `249889958`. This makes the `i256` lane a short
certificate-chain experiment rather than a single arithmetic demo, but the
claim boundary is unchanged: the chain verifies supplied quotient/remainder
witnesses and point-step arithmetic only.

`stdlib/theorem/div_witness.sio` now factors the quotient/remainder check into a
small reusable proof-checker helper for `i64` and `i256`. The executable
self-contained smoke `tests/run-pass/div_witness_checker_tiny.sio` validates
the checker rule for both ordinary integer certificates and Lorenz-shaped
`i256` certificate data, pinning bundle fingerprints `963466398` and
`813457414`. The imported smoke `tests/run-pass/div_witness_i64_imported_tiny.sio`
and the `i256` imported smoke
`tests/run-pass/div_witness_i256_imported_tiny.sio` checks three concrete
witnesses from the Lorenz `i256` lane in frontend/check mode: `S/100 =
42949672 r96`, `8*z3/3 = 10988500690 r2`, and the fourth-step `x` increment
`dt*dx/S = 302608392 r1014364768`, with bundle fingerprint `813457414`.
Runtime for the imported helper path currently exits `139`, so those imported
gates are marked known-failure and treated as a multimodule/native boundary, not
as a mathematical failure of the witness rule. This is the reusable checker
contract for certificate-supplied division; it still does not execute
source-level `i256` division.

`tests/run-pass/lorenz_i256_division_chain_four_step_tiny.sio` extends the
same quotient/remainder certificate chain to a fourth Euler step. Starting from
the three-step state above, it checks:

- `dy` numerator quotient `124866970867`, remainder `1906281842`;
- `xy` quotient `8218332706`, remainder `3252936400`;
- `8z/3` quotient `10988500690`, remainder `2`;
- `x` increment quotient `302608392`, remainder `1014364768`;
- `y` increment quotient `1172231215`, remainder `2184377272`;
- `z` drop quotient `27701679`, remainder `948411264`.

The fourth witness lands at `x=4920371098`, `y=8816077911`, `z=4092986080`,
pins step-4 certificate fingerprint `753371133`, and chains all four
certificates into anchor `737039167`. This is still a supplied
quotient/remainder witness chain, not a source-level `i256` division fix, not a
validated enclosure, and not a long-time Lorenz result.

`tests/run-pass/lorenz_i256_trajectory_manifest_tiny.sio` packages the
three-step quotient/remainder chain into a point-trajectory manifest. It binds
scale `2^32`, `dt=floor(S/100)=42949672` with remainder `96`, step count `3`,
initial state `(S,S,S)`, chain anchor `249889958`, final step certificate
`603078026`, final state `(4617762706,7643846696,4120687759)`, and manifest
fingerprint `449592233`. It also checks simple displacement sanity:
`x` increased by `322795410`, `y` increased by `3348879400`, and `z` decreased
by `174279537`. This is a point-trajectory manifest over a checked witness
chain, not an enclosure, invariant set, adaptive solver, or proof of Lorenz
dynamics beyond the stated Euler steps.

Follow-up regression pins added after isolating those boundaries:

- `tests/run-pass/wide_i128_signed_div_known_failure.sio` is an XFAIL for
  signed negative `i128` division. The current wide division lowering/codegen is
  unsigned; a real fix needs signedness to survive lowering into the wide-div IR
  or a signed wide-div opcode/checker contract.
- `tests/run-pass/wide_i128_fn_abi_known_failure.sio` is an XFAIL for passing
  and returning `i128` across user-defined call boundaries. Inline wide-int
  arithmetic is green, but public fixed-point APIs should not depend on helper
  functions taking/returning wide ints until this XPASes.
- `tests/run-pass/wide_i256_divfull_scratch_known_failure.sio` isolates the
  source-level `i256` division crash below Lorenz itself. One `i256` division can
  run, but two adjacent source-level `DivFull` operations feeding a subtraction
  currently exit 139 in the checked Madaros artifact. The source lowerer now
  reserves hidden backend scratch slots for wide multiply and full wide
  division/remainder, but the XPASS proof still requires a rebuilt Madaros
  artifact.
- `tests/run-pass/lorenz_i256_fixed_step_1e6.sio` is an XFAIL for the same
  complete Lorenz point step that passes over `i128` at scale `1e6`. It
  typechecks, compiles, and then exits 139 at runtime, so the current `i256`
  blocker is not just high-scale arithmetic; source-level `i256` fixed-point
  division is still unsafe even while all intermediate values stay small.
- `tests/run-pass/lorenz_i256_smallscale_step.sio` is an XFAIL for the
  source-level `i256` Lorenz division path even at scale `2^32`.
- `tests/run-pass/lorenz_i256_product_smoke.sio` is an XFAIL for high-scale
  `i256` Lorenz numerator products around scale `1e20`, where products reach
  roughly `1e40`.
- `tests/run-pass/lorenz_i256_fixed_step.sio` is an XFAIL for the full high-scale
  `i256` fixed-point Euler step. Together these pins separate "i256 type and
  smaller source arithmetic work" from "Lorenz-scale source `i256` products and
  divisions are not production-ready."

Current v43 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_taylor_ball_bridge_fingerprint()` and
  `lorenz_i256_taylor_ball_bridge_audit_fingerprint()`. This is a bounded
  Taylor/ball bridge receipt for the existing Lorenz five-step `i256` lane. It
  binds the checked point-trajectory manifest `214180161`, dependency-DAG
  `713010204`, final ball enclosure `166945519`, final enclosure receipt
  `114244647`, range budget `87220662`, roundoff budget `677514412`, projection
  margin budget `69322697`, local Taylor order `2`, `dt=1/100`, source width
  `i256`, scale `2^32`, bit headroom `184`, and final radius/cap slack
  `(6634, 5360, 7552)` into bridge fingerprint `979362768` and audit
  fingerprint `927174126`.
- `stdlib/theorem/portfolio.sio` registers the bridge as portfolio kind/checker
  `46` and v43. The v43 manifest has 47 entries, result coverage
  `(unsat=10, validated=35, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=30, meta=2)`, manifest fingerprint
  `130892363`, and readiness fingerprint `61654294`.
- The tiny gates
  `tests/run-pass/lorenz_i256_taylor_ball_bridge_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor_ball_bridge_v43_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- Scope boundary: this is not a Taylor-model integrator, not Flow*, not a
  long-time Lorenz theorem, and not source-level `i256` division acceptance. It
  is the next checker-shaped object: a proof-carrying bridge that says the
  high-precision point certificate, radius budgets, projection/enclosure stack,
  and local Taylor-remainder policy are mutually pinned.

Current v44 addition:

- `stdlib/systems/ball_fixed.sio` adds a tiny Taylor-order-2 remainder checker:
  `ball_fixed_taylor2_remainder_den()`,
  `ball_fixed_taylor2_remainder_radius_from_ratio()`,
  `ball_fixed_lorenz_taylor2_remainder_fingerprint()`, and
  `ball_fixed_lorenz_taylor2_remainder_audit_fingerprint()`. For the current
  Lorenz bridge step ratio `h=1/100`, it checks the denominator
  `6 * 100^3 = 6000000` and verifies:
  - `ceil(607000000 / 6000000) = 102`;
  - `ceil(809000000 / 6000000) = 135`;
  - `ceil(119000000 / 6000000) = 20`.
- The remainder checker then proves those radii still fit in the v43 final ball
  slack: x slack `6634 -> 6532`, y slack `5360 -> 5225`, z slack
  `7552 -> 7532`. The local remainder fingerprint is `320985955`, and the
  audit fingerprint is `980184978`.
- `stdlib/theorem/portfolio.sio` registers the remainder layer as
  kind/checker `47` and portfolio v44. The v44 manifest has 48 entries, result
  coverage `(unsat=10, validated=36, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=31, meta=2)`, manifest fingerprint
  `898788381`, and readiness fingerprint `101774289`.
- The tiny gates `tests/run-pass/lorenz_taylor2_remainder_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_remainder_v44_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- Scope boundary: this is still not a Taylor-model integrator and not an
  adaptive validated ODE solver. It is deliberately smaller: the first replayable
  remainder-radius micro-kernel in the Lorenz/i256 lane, shaped so a future
  Taylor/RK/flowpipe producer can emit small per-step bounds that Sounio checks
  independently.

Current v45 addition:

- `stdlib/systems/ball_fixed.sio` adds
  `ball_fixed_lorenz_taylor2_step_policy_fingerprint()` and
  `ball_fixed_lorenz_taylor2_step_policy_audit_fingerprint()`. This is a
  small step-policy guard over the v44 Taylor-2 remainder receipt, not a new
  Taylor-model integrator. It checks the final slack budget after the
  remainder:
  - total pre-remainder slack `6634 + 5360 + 7552 = 19546`;
  - total Taylor-2 remainder `102 + 135 + 20 = 257`;
  - total post-remainder slack `6532 + 5225 + 7532 = 19289`;
  - conservative consumed margin
    `ceil(257 * 1000000 / 19546) = 13149 ppm`;
  - remaining slack ratio
    `floor(19289 * 1000000 / 19546) = 986851 ppm`.
- The policy thresholds are deliberately simple: consumed remainder must be at
  most `15000 ppm`, minimum post-remainder axis slack must be at least `5000`,
  total post-remainder slack must be at least `19000`, and the v44 validation
  mask must remain `127`. Those checks produce policy mask `63`, choose
  `action_accept_same_dt=1`, `action_grow_dt=0`, and keep the next local
  step ratio at `1/1`. In other words, the current `dt=1/100` is accepted for
  this receipt, but the guard does not authorize increasing `dt`.
- The local step-policy fingerprint is `403306384`, and the audit fingerprint
  is `635367450`.
- `stdlib/theorem/portfolio.sio` registers the step-policy layer as
  kind/checker `48` and portfolio v45. The v45 manifest has 49 entries, result
  coverage `(unsat=10, validated=37, optimal=1, sat=1)`, checker-family
  coverage `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=32, meta=2)`, manifest
  fingerprint `975812377`, and readiness fingerprint `498384393`.
- The tiny gates `tests/run-pass/lorenz_taylor2_step_policy_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_policy_v45_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- Scope boundary: this is still not Flow*, CAPD, Ariadne, or a full adaptive
  validated ODE solver. It is the first replayable accept/hold decision over
  the Lorenz/i256 Taylor-remainder receipt, designed so a future producer can
  emit step candidates and Sounio can independently accept, hold, shrink, or
  reject them with integer receipts.

Current v46 addition:

- `stdlib/systems/ball_fixed.sio` adds
  `ball_fixed_adaptive_step_action_code()`,
  `ball_fixed_lorenz_taylor2_adaptive_step_decision_fingerprint()`, and
  `ball_fixed_lorenz_taylor2_adaptive_step_decision_audit_fingerprint()`.
  This is the first replayable adaptive decision layer over the v45 policy
  receipt. The action vocabulary is deliberately tiny and integer-checkable:
  `0=reject`, `1=shrink`, `2=hold`, `3=grow`.
- The decision rule rejects invalid masks or negative post-remainder slack,
  shrinks when the consumed-remainder ppm exceeds the shrink limit or the
  post-remainder slack falls below required limits, grows only when the
  consumed-remainder ppm and both slack margins are comfortably inside grow
  thresholds, and otherwise holds the current step. For the current Lorenz/i256
  receipt:
  - consumed remainder is `13149 ppm`;
  - remaining slack ratio is `986851 ppm`;
  - shrink threshold is `20000 ppm`;
  - grow threshold is `5000 ppm`;
  - minimum post-remainder axis slack is `5225` against hold limit `5000`
    and grow limit `7000`;
  - total post-remainder slack is `19289` against hold limit `19000`
    and grow limit `25000`;
  - validation mask is `127`.
- Those facts force action code `2`, meaning hold the current `dt=1/100`.
  The receipt also records candidate ratios for the future producer/consumer
  protocol: hold `1/1`, shrink `1/2`, and grow `3/2`. The local decision
  fingerprint is `120259197`, and its audit fingerprint is `471466754`.
- `stdlib/theorem/portfolio.sio` registers the adaptive decision as
  kind/checker `49` and portfolio v46. The v46 manifest has 50 entries, result
  coverage `(unsat=10, validated=38, optimal=1, sat=1)`, checker-family
  coverage `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=33, meta=2)`, manifest
  fingerprint `470000532`, and readiness fingerprint `984799877`.
- The tiny gates
  `tests/run-pass/lorenz_taylor2_adaptive_step_decision_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_adaptive_step_decision_v46_tiny.sio`
  are executable runtime checks. The first tiny gate also checks all four
  action branches: reject, shrink, hold, and grow. The imported gates typecheck
  the public APIs and are marked known-failure for the existing multimodule
  native-lowering exit-139 boundary.
- SOTA alignment: Flow* exposes adaptive step-size ranges, and Taylor-model
  based validated integration uses local error/remainder control to balance
  tight enclosures against computational cost. Sounio's current contribution is
  narrower but distinctive: the step decision is not a floating runtime
  heuristic. It is a small proof-carrying integer receipt that can be replayed
  independently by the language-side checker.
- Scope boundary: this is still not a full flowpipe engine, not a Taylor-model
  algebra, not a CAPD/Flow*/Ariadne replacement, and not a long-time Lorenz
  theorem. It is a verified control-plane primitive for a future producer that
  emits candidate steps, remainders, and enclosures.

Current v47 addition:

- `stdlib/systems/ball_fixed.sio` adds `ball_fixed_rational_leq_nonneg()`,
  `ball_fixed_lorenz_taylor2_step_schedule_fingerprint()`, and
  `ball_fixed_lorenz_taylor2_step_schedule_audit_fingerprint()`. This is the
  first rational step-schedule receipt over the v46 adaptive decision.
- The schedule follows the Flow*-style idea that adaptive step size is bounded
  by positive rational min/max limits. For the current Lorenz/i256 receipt it
  binds:
  - base step `1/100`;
  - shrink candidate `1/200`;
  - selected/hold step `1/100`;
  - grow candidate `3/200`;
  - admissible range `[1/200, 3/200]`;
  - selected denominator cap `200`;
  - v46 action code `2` and schedule mask `127`.
- The tiny rational checker proves `1/200 <= 1/100 <= 3/200` by integer
  cross-multiplication, also checking the ordered candidate ladder
  `1/200 <= 1/100 <= 3/200`. The selected next step is therefore inside the
  schedule bounds and matches the hold action. The local schedule fingerprint
  is `353612555`, and the audit fingerprint is `335541872`.
- `stdlib/theorem/portfolio.sio` registers the schedule as kind/checker `50`
  and portfolio v47. The v47 manifest has 51 entries, result coverage
  `(unsat=10, validated=39, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=34, meta=2)`, manifest fingerprint
  `189376492`, and readiness fingerprint `224324310`.
- The tiny gates `tests/run-pass/lorenz_taylor2_step_schedule_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_schedule_v47_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- Scope boundary: v47 does not compute a new Lorenz enclosure and does not prove
  that the next step will succeed. It only certifies the schedule selection
  surface: given the v46 hold decision, the selected rational step is inside
  the producer/consumer min-max envelope.

Current v48 addition:

- `stdlib/systems/ball_fixed.sio` adds
  `ball_fixed_lorenz_taylor2_step_request_fingerprint()` and
  `ball_fixed_lorenz_taylor2_step_request_audit_fingerprint()`. This is a
  producer-request certificate for the next Taylor-2 Lorenz/i256 step after the
  v47 rational schedule.
- The request binds the v47 schedule fingerprint `353612555`, v47 schedule audit
  `335541872`, v47 portfolio receipt `842631931`, v47 portfolio audit
  `13881891`, and v47 readiness `224324310`. It also binds the existing
  trajectory manifest `214180161`, final enclosure receipt `114244647`,
  dependency DAG `713010204`, step index `6`, Taylor order `2`, selected step
  `1/100`, fixed-point scale `2^32`, source width `i256`, bit headroom `184`,
  minimum bit headroom `180`, target remainder `15000` ppm, denominator cap
  `200`, schedule mask `127`, and pending request status `1`.
- The request mask is `127`: step index matches, order is Taylor-2, selected
  step is `1/100`, width/scale are `i256` at `2^32`, headroom satisfies
  `184 >= 180`, target remainder satisfies `15000 <= 15000`, and the selected
  denominator `100` is inside the v47 cap `200`.
- `stdlib/theorem/portfolio.sio` registers the step request as kind/checker
  `51` and portfolio v48. The v48 manifest has 52 entries, result coverage
  `(unsat=10, validated=40, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=35, meta=2)`, manifest fingerprint
  `59406573`, and readiness fingerprint `38526066`.
- The tiny gates `tests/run-pass/lorenz_taylor2_step_request_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_step_request_v48_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- SOTA alignment: Flow*, Ariadne, and Taylor-model reachability literature build
  flowpipes or Taylor-model reach sets by combining high-order expansions,
  remainder control, and validated enclosure propagation. Sounio is not claiming
  that engine yet. The v48 contribution is a language-native, integer-only
  control-plane receipt for a future producer: a replayable request saying which
  Taylor-2 step should be attempted and under which width, scale, headroom, and
  remainder contract.
- Scope boundary: v48 does not compute the next enclosure, does not certify the
  step succeeds, does not prove long-time Lorenz behavior, and does not replace
  CAPD/Flow*/Ariadne. It only turns the already-certified v47 schedule into a
  precise pending step request that a producer/checker pair can consume next.

Current v49 addition:

- `stdlib/systems/ball_fixed.sio` adds
  `ball_fixed_lorenz_taylor2_response_envelope_fingerprint()` and
  `ball_fixed_lorenz_taylor2_response_envelope_audit_fingerprint()`. This is a
  response-envelope contract over the v48 pending request. It says exactly what
  the next producer must attach before Sounio may talk about a candidate step:
  center data, radius data, remainder data, and proof-trace data.
- The envelope binds v48 request fingerprint `510353327`, request audit
  `465022480`, v48 portfolio receipt `185755530`, v48 portfolio audit
  `167433392`, v48 readiness `38526066`, trajectory manifest `214180161`,
  dependency DAG `713010204`, step index `6`, Taylor order `2`, selected step
  `1/100`, fixed-point scale `2^32`, source width `i256`, request mask `127`,
  candidate kind `ball enclosure`, required artifact count `4`, and response
  status `candidate-required`.
- The response mask is `127`: request/audit match, v48 receipt/readiness match,
  step/order match, selected step matches, width/scale match, the envelope asks
  for the expected four artifacts, and the response status is candidate-required
  while the v48 request mask remains complete.
- `stdlib/theorem/portfolio.sio` registers the response envelope as kind/checker
  `52` and portfolio v49. The v49 manifest has 53 entries, result coverage
  `(unsat=10, validated=41, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=36, meta=2)`, manifest fingerprint
  `678566245`, and readiness fingerprint `367410832`.
- The tiny gates `tests/run-pass/lorenz_taylor2_response_envelope_tiny.sio` and
  `tests/run-pass/solver_portfolio_lorenz_i256_taylor2_response_envelope_v49_tiny.sio`
  are executable runtime checks. The imported gates typecheck the public APIs
  and are marked known-failure for the existing multimodule native-lowering
  exit-139 boundary.
- Scope boundary: v49 is deliberately still not a computed candidate, not a
  verified next enclosure, not a validated Taylor step, and not a long-time
  Lorenz theorem. It is the typed checklist that prevents a future producer
  from smuggling an under-specified candidate into the proof portfolio.

Current v50 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds the first concrete artifact that
  satisfies one slot of the v49 response envelope:
  `lorenz_i256_step6_center_artifact_check()` plus fingerprint/audit helpers for
  a step-6 center candidate. This is still not a radius artifact, remainder
  artifact, proof-trace artifact, or validated enclosure.
- The checker starts from the five-step manifest endpoint
  `(x,y,z)=(5309941770,10058731223,4084837935)` at scale `2^32`, binds the v49
  response envelope `506104143`/`956295573`, and verifies quotient/remainder
  witnesses for:
  - `dt = floor(2^32/100) = 42949672`, remainder `96`;
  - `dy_scaled = floor(x*(28*2^32-z)/2^32) = 143628214324`,
    remainder `3606116906`;
  - `xy_scaled = floor(x*y/2^32) = 12435782019`, remainder `2421034086`;
  - `beta_z = floor(8*z/3) = 10892901160`, remainder `0`;
  - x increment `474878934`, remainder `2944751696`;
  - y increment `1335694801`, remainder `665264776`;
  - z increment `15428808`, remainder `1052865080`.
- A small but important dynamics detail is now explicit: at step 6,
  `xy_scaled > beta_z`, so the z update is an increment rather than the earlier
  drop convention. The checked center becomes
  `(5784820704,11394426024,4100266743)`.
- The local instance fingerprint is `104233294`, certificate fingerprint is
  `366755610`, center artifact fingerprint is `134624236`, and artifact audit
  is `643317565`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `53` and
  portfolio v50. The v50 manifest has 54 entries, result coverage
  `(unsat=10, validated=42, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=37, meta=2)`, manifest fingerprint
  `215425432`, and readiness fingerprint `228451054`.
- The tiny gate `tests/run-pass/lorenz_i256_step6_center_artifact_tiny.sio`
  executes the quotient/remainder witnesses with native `i256` arithmetic. The
  imported gates typecheck public APIs and remain known-failure for the current
  multimodule native-lowering exit-139 boundary.
- Scope boundary: v50 is a checked point-center artifact for one step. It is
  not a Taylor-2 high-order producer, not a ball radius, not a remainder bound,
  not a proof trace for enclosure containment, and not a rigorous Lorenz
  flowpipe. It is the first concrete payload slot under the v49 response
  envelope.

Current v51 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_radius_artifact_check()` plus fingerprint/audit helpers for
  a conservative radius artifact paired with the v50 step-6 center artifact.
- The checker converts the previous five-step target-scale enclosure radii
  `(243366,319640,42448)` at scale `1e6` into source-scale `2^32` radii
  `(1045249011,1372843347,182312772)` by upward conversion, then propagates one
  Euler-style radius step using `i256` inequalities rather than source-level wide
  division. The derivative-radius witnesses are:
  `dx_rad=24180923580`, `prod_x_rhoz=28542626778`,
  `dy_rad=29915470125`, `prod_xy=4479326365`,
  `beta_rad=486167393`, and `dz_rad=4965493758`.
- The resulting source-scale radii are
  `(1287058242,1671998042,231967709)`, guarded by explicit caps
  `(1600000000,2600000000,500000000)`. These caps are intentionally larger than
  the old `ball_fixed` helper cap, which was tuned for smaller target-scale
  helper receipts and is too tight for this source-scale i256 radius artifact.
- The local radius instance fingerprint is `399482214`, certificate fingerprint
  is `694298444`, radius artifact fingerprint is `237534229`, and artifact audit
  is `931365823`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `54` and
  portfolio v51. The v51 manifest has 55 entries, result coverage
  `(unsat=10, validated=43, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=38, meta=2)`, manifest fingerprint
  `751523371`, and readiness fingerprint `362208564`.
- The tiny gate `tests/run-pass/lorenz_i256_step6_radius_artifact_tiny.sio`
  executes the radius-conversion and propagation inequalities with native
  `i256` arithmetic. The imported gates typecheck public APIs and remain
  known-failure for the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v51 is a conservative radius artifact in `2^32` scale,
  converted from the previous `1e6` enclosure radii and propagated one
  Euler-style radius step. It is not a Taylor-2 remainder, not a proof trace, not
  a validated enclosure, and not a rigorous Lorenz flowpipe.

Current v52 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_enclosure_projection_artifact_check()` plus
  fingerprint/audit helpers for projecting the v50/v51 step-6 center-plus-radius
  enclosure from source scale `2^32` into target scale `1e6` with directed
  rounding.
- The source-scale interval endpoints are checked as exact `center ± radius`:
  `x=[4497762462,7071878946]`, `y=[9722427982,13066424066]`,
  `z=[3868299034,4332234452]`. The target-scale projection uses lower
  endpoints rounded down and upper endpoints rounded up:
  `x=[1047216,1646551]`, `y=[2263679,3042264]`,
  `z=[900658,1008677]`.
- The checker avoids source-level wide division by verifying directed-rounding
  inequalities: `q*S <= n*T < (q+1)*S` for lower floors and
  `(q-1)*S < n*T <= q*S` for upper ceils. It also checks target widths
  `(599335,778585,108019)` and containment of the directed-rounded projected
  center ranges.
- The local projection instance fingerprint is `35569102`, certificate
  fingerprint is `472466935`, projection artifact fingerprint is `516981885`,
  and artifact audit is `989033159`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `55` and
  portfolio v52. The v52 manifest has 56 entries, result coverage
  `(unsat=10, validated=44, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=39, meta=2)`, manifest fingerprint
  `924526189`, and readiness fingerprint `28339575`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_enclosure_projection_artifact_tiny.sio`
  executes the source endpoint and directed-rounding inequalities with native
  `i256` arithmetic. The imported gates typecheck public APIs and remain
  known-failure for the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v52 is a directed-rounding projection of the checked discrete
  center-plus-radius artifact into target scale. It is not a new integrator, not
  a Taylor-2 remainder, not a proof trace for continuous-time containment, not a
  validated flowpipe, and not a long-time Lorenz theorem.

Current v53 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_enclosure_candidate_bundle_check()` plus fingerprint/audit
  helpers for bundling the step-6 center, radius, and directed-projection
  artifacts under the v49 response envelope.
- The bundle intentionally records a partial state rather than pretending the
  enclosure is complete. It pins `required_artifact_mask=15`,
  `provided_artifact_mask=7`, `missing_artifact_mask=8`, and
  `candidate_status_partial=3`. The checked dependencies are the v50 center
  artifact `134624236`, v51 radius artifact `237534229`, and v52 projection
  artifact `516981885`.
- The checker proves the three dependencies are present, the required/provided
  masks combine to the missing mask, the candidate remains partial, and the
  validated-enclosure mask is still `0`. This is a proof-checker guard against
  accidentally upgrading three checked artifacts into a full validated enclosure.
- The local bundle instance fingerprint is `425169956`, certificate fingerprint
  is `54004840`, candidate bundle fingerprint is `947760031`, and artifact audit
  is `982064907`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `56` and
  portfolio v53. The v53 manifest has 57 entries, result coverage
  `(unsat=10, validated=45, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=40, meta=2)`, manifest fingerprint
  `145449`, and readiness fingerprint `878999745`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_enclosure_candidate_bundle_tiny.sio`
  executes the partial-candidate mask checks and fingerprint anchors. The
  imported gates typecheck public APIs and remain known-failure for the current
  multimodule native-lowering exit-139 boundary.
- Scope boundary: v53 is a compositional partial-candidate receipt. It is not a
  missing Taylor/remainder artifact, not a continuous-time containment proof,
  not a validated next enclosure, not a flowpipe, and not a long-time Lorenz
  theorem. Its job is precisely to keep those non-claims machine-visible.

Current v54 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_remainder_obligation_check()` plus fingerprint/audit
  helpers for the missing step-6 remainder-obligation slot. The input box is
  exactly the v52 directed projection at target scale `1_000_000`:
  `x=[1047216,1646551]`, `y=[2263679,3042264]`, and
  `z=[900658,1008677]`, with `dt=1/100`.
- The checker records conservative first-derivative bounds over that projected
  box: `|rho-z|<=27099342`, `|x'|<=46888150`,
  `ceil(x_hi*|rho-z|/S)=44620449`, `|y'|<=47662713`,
  `ceil(x_hi*y_hi/S)=5009243`, `ceil(8*z_hi/3)=2689806`, and
  `|z'|<=7699049`.
- It then records conservative second-derivative bounds:
  `|x''|<=10*(47662713+46888150)=945508630`,
  `|y''|<=1270638013+12676877+47662713=1330977603`, and
  `|z''|<=142646131+78479088+20530798=241656017`.
  The local-error budget uses `ceil(second/20000)` for `h=1/100`, producing
  LTE obligations `(47276,66549,12083)` with caps `(50000,70000,15000)` and
  remaining slack `(2724,3451,2917)`.
- The local artifact binds instance fingerprint `827494918`, certificate
  fingerprint `967484944`, remainder-obligation fingerprint `652877359`, and
  artifact audit `344816971`. The checker now replays the component sums and
  the ceiling inequalities, not just the final constants.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `57` and
  portfolio v54. The v54 manifest has 58 entries, result coverage
  `(unsat=10, validated=46, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=41, meta=2)`, manifest fingerprint
  `290740352`, acceptance receipt `897074981`, audit receipt `812192508`, and
  readiness fingerprint `903533282`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_remainder_obligation_tiny.sio` executes the
  arithmetic and ceiling checks with `i64` helper arithmetic because all
  target-scale products in this obligation fit in signed 64-bit range. The
  imported gates typecheck public APIs and remain known-failure for the current
  multimodule native-lowering exit-139 boundary.
- Scope boundary: v54 is a conservative step-6 remainder-obligation budget over
  the already projected candidate box. It is not a completed Taylor-2 proof,
  not a proof trace for continuous-time containment, not a validated next
  enclosure, not a flowpipe, and not a long-time Lorenz theorem.

Current v55 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_completed_candidate_bundle_check()` plus fingerprint/audit
  helpers for the first step-6 bundle whose required artifact mask is fully
  populated. It composes the v50 center artifact `134624236`, v51 radius
  artifact `237534229`, v52 projection artifact `516981885`, and v54
  remainder-obligation artifact `652877359` under the v49 response envelope.
- The machine-visible distinction is explicit: `required_artifact_mask=15`,
  `provided_artifact_mask=15`, `missing_artifact_mask=0`, and
  `candidate_status_complete=4`, but `validated_enclosure_mask` remains `0`.
  This closes the missing-artifact bookkeeping hole from v53 without pretending
  that the completed candidate is already a continuous-time validated enclosure.
- The local completed-bundle instance fingerprint is `175122911`, certificate
  fingerprint is `425699291`, completed-bundle fingerprint is `641464796`, and
  artifact audit is `222156824`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `58` and
  portfolio v55. The v55 manifest has 59 entries, result coverage
  `(unsat=10, validated=47, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=42, meta=2)`, manifest fingerprint
  `773887956`, acceptance receipt `204512630`, audit receipt `130046178`, and
  readiness fingerprint `260555867`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_completed_candidate_bundle_tiny.sio`
  executes the complete-mask checks, the non-validation guard, and the
  fingerprint/audit anchors. The imported gates typecheck public APIs and
  remain known-failure for the current multimodule native-lowering exit-139
  boundary.
- Scope boundary: v55 is a completed candidate-bundle receipt, not a Taylor-2
  proof trace, not a validated next enclosure, not a continuous flowpipe, not a
  shadowing/invariant-set proof, and not a long-time Lorenz theorem. It is the
  bookkeeping bridge that says the candidate now has all four requested
  artifacts available for a future proof-trace/enclosure validator.

Current v56 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_proof_trace_skeleton_check()` plus fingerprint/audit
  helpers for turning the v55 completed candidate bundle into a replay-shaped
  trace skeleton. The trace skeleton names four nodes: the v50 center artifact,
  v51 radius artifact, v52 projection artifact, and v54 remainder-obligation
  artifact.
- The skeleton fixes `trace_version=1`, `trace_kind_step6=6`,
  `trace_node_count=4`, `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, and `trace_status_skeleton=5`. It still fixes
  `validated_enclosure_mask=0`, so the trace cannot be confused with a completed
  enclosure proof.
- The local proof-trace-skeleton instance fingerprint is `524260857`,
  certificate fingerprint is `454263207`, skeleton fingerprint is `929938975`,
  and artifact audit is `531834411`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `59` and
  portfolio v56. The v56 manifest has 60 entries, result coverage
  `(unsat=10, validated=48, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=43, meta=2)`, manifest fingerprint
  `813874951`, acceptance receipt `267225966`, audit receipt `73809619`, and
  readiness fingerprint `836194547`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_proof_trace_skeleton_tiny.sio` executes the
  trace-node, replay-mask, dependency-root, and non-validation checks. The
  imported gates typecheck public APIs and remain known-failure for the current
  multimodule native-lowering exit-139 boundary.
- Scope boundary: v56 is a proof-trace skeleton receipt, not a proof-trace
  replay engine, not a Taylor-2 enclosure proof, not a validated next enclosure,
  not a continuous flowpipe, not a shadowing/invariant-set proof, and not a
  long-time Lorenz theorem. It is the next proof-checker-shaped handoff surface
  for a future enclosure validator.

Current v57 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_replay_preflight_check()` plus fingerprint/audit helpers
  for the first replay-shaped receipt over the v56 trace skeleton. The preflight
  consumes the proof-trace skeleton `929938975`, skeleton audit `531834411`, and
  completed bundle `641464796`.
- The preflight fixes `trace_version=1`, `replay_version=1`,
  `trace_node_count=4`, `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, and `replay_status_preflight=6`. It still fixes
  `validated_enclosure_mask=0`.
- The local replay-preflight instance fingerprint is `416249604`, certificate
  fingerprint is `539977104`, replay-preflight fingerprint is `490362827`, and
  artifact audit is `772468073`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `60` and
  portfolio v57. The v57 manifest has 61 entries, result coverage
  `(unsat=10, validated=49, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=44, meta=2)`, manifest fingerprint
  `181028475`, acceptance receipt `109240090`, audit receipt `898775050`, and
  readiness fingerprint `439805716`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_replay_preflight_tiny.sio` executes the
  replay-order, replayed-node, predecessor-ready, version, and non-validation
  checks. The imported gates typecheck public APIs and remain known-failure for
  the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v57 is a replay preflight receipt. It is not a proof-trace
  replay engine, not a Taylor-2 enclosure proof, not a validated next enclosure,
  not a continuous flowpipe, not a shadowing/invariant-set proof, and not a
  long-time Lorenz theorem. It moves the proof shape from named nodes to
  replayable masks, leaving the actual enclosure validator as the next missing
  semantic layer.

Current v58 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_replay_executor_check()` plus fingerprint/audit helpers
  for the first execution-shaped replay receipt over the v57 preflight. The
  executor consumes replay preflight `490362827`, replay-preflight audit
  `772468073`, proof-trace skeleton `929938975`, and completed bundle
  `641464796`.
- The executor fixes `trace_version=1`, `replay_engine_version=1`,
  `trace_node_count=4`, `dependency_edge_count=4`, `node_receipt_mask=15`,
  `edge_receipt_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, and `replay_status_executed=7`. It still fixes
  `validated_enclosure_mask=0`.
- The local replay-executor instance fingerprint is `195235667`, certificate
  fingerprint is `81302823`, replay-executor fingerprint is `975464919`, and
  artifact audit is `376630542`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `61` and
  portfolio v58. The v58 manifest has 62 entries, result coverage
  `(unsat=10, validated=50, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=45, meta=2)`, manifest fingerprint
  `15485832`, acceptance receipt `30989952`, audit receipt `6887223`, and
  readiness fingerprint `686563909`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_replay_executor_tiny.sio` executes the
  node-receipt, edge-receipt, replayed-node, predecessor-ready, version, and
  non-validation checks. The imported gates typecheck public APIs and remain
  known-failure for the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v58 is a replay executor receipt, not a Taylor-2 enclosure
  proof, not a validated next enclosure, not a continuous flowpipe, not a
  shadowing/invariant-set proof, and not a long-time Lorenz theorem. It makes
  the proof trace more replay-shaped by checking node/edge execution masks, but
  it still does not prove that the step-6 enclosure contains the true Lorenz
  flow.

Current v59 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_enclosure_validator_guard_check()` plus fingerprint/audit
  helpers for the first guarded inclusion validator over the v58 replay
  executor. It consumes replay executor `975464919`, replay-executor audit
  `376630542`, completed bundle `641464796`, projection artifact `516981885`,
  and remainder obligation `652877359`.
- The guard evaluates the explicit scalar inclusion requirements
  `need = projected_radius + LTE` at target scale `1_000_000`: `x_need=290642`,
  `y_need=386189`, `z_need=54531`. It compares them with projection margins
  `x_margin=299663`, `y_margin=389292`, `z_margin=54009`, producing
  `inclusion_pass_mask=3`, `inclusion_fail_mask=4`, and a documented
  `z` deficit of `522`.
- The local enclosure-guard instance fingerprint is `898897987`, certificate
  fingerprint is `851507956`, guard artifact fingerprint is `367562376`, and
  artifact audit is `814950326`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `62` and
  portfolio v59. The v59 manifest has 63 entries, result coverage
  `(unsat=10, validated=51, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=46, meta=2)`, manifest fingerprint
  `626294410`, acceptance receipt `634036559`, audit receipt `460279835`, and
  readiness fingerprint `774303054`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_enclosure_validator_guard_tiny.sio`
  executes the inclusion-mask, z-deficit, radius-plus-remainder, and
  non-promotion checks. The imported gates typecheck public APIs and remain
  known-failure for the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v59 is a guarded validator receipt. It does not validate the
  step-6 enclosure, because the explicit `z` inequality fails by `522` at the
  current projected margins. The important safety behavior is that
  `validated_enclosure_mask` remains `0` and `validator_status_guarded=8`.
  Closing v59 requires either a wider projection, tighter z-radius/LTE budget,
  or a recomputed candidate with enough z margin before any valid-enclosure
  claim is allowed.

Current v60 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_z_margin_repair_check()` plus fingerprint/audit helpers
  for the first conservative repair of the v59 `z` deficit. It consumes the v59
  enclosure guard `367562376`/audit `814950326` and the original v52 projection
  artifact `516981885`/audit `989033159`.
- The repair keeps the old projected `z` interval `[900658,1008677]` as a
  required sub-interval and introduces a wider superset
  `[899667,1009668]`. The old width was `108019`, old effective margin was
  `54009`; the repaired width is `110001`, repaired margin is `55000`.
  Each side is widened by `991`.
- Against the existing v59 requirement `z_need=54531`, the repaired margin has
  slack `469`. This changes the repaired scalar inclusion masks to
  `repaired_inclusion_pass_mask=7` and `repaired_inclusion_fail_mask=0`, while
  still keeping `validated_enclosure_mask=0`.
- The local z-margin repair instance fingerprint is `156039132`, certificate
  fingerprint is `264234495`, repair artifact fingerprint is `617386880`, and
  artifact audit is `478471574`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `63` and
  portfolio v60. The v60 manifest has 64 entries, result coverage
  `(unsat=10, validated=52, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=47, meta=2)`, manifest fingerprint
  `795358725`, acceptance receipt `907169025`, audit receipt `1448964`, and
  readiness fingerprint `16326305`.
- The tiny gate `tests/run-pass/lorenz_i256_step6_z_margin_repair_tiny.sio`
  executes the old-interval containment, repaired-interval width, superset,
  repaired z-slack, repaired inclusion-mask, center-anchor, and non-promotion
  checks. The imported gates typecheck public APIs and remain known-failure for
  the current multimodule native-lowering exit-139 boundary.
- Scope boundary: v60 repairs the scalar z margin as a conservative interval
  widening receipt. It is still not a validated step-6 enclosure and does not
  claim a Taylor-2 flowpipe, shadowing/invariant proof, or long-time Lorenz
  theorem. The next semantic step is a validator that replays the full
  `(x,y,z)` inclusion using this repaired projection and only then considers
  changing `validated_enclosure_mask`.

Current v61 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_repaired_projection_inclusion_validator_check()` plus
  fingerprint/audit helpers for the first scalar validation pass over the
  repaired step-6 projection. It consumes the v60 z-margin repair
  `617386880`/audit `478471574`, the v58 replay executor `975464919`, the v57
  preflight `376630542`, and the v55 completed candidate bundle `641464796`.
- The validator checks the repaired scalar margins
  `(x=299663, y=389292, z=55000)` against the existing needs
  `(x=290642, y=386189, z=54531)`. The resulting slacks are
  `(x=9021, y=3103, z=469)`, giving `inclusion_pass_mask=7` and
  `inclusion_fail_mask=0`.
- This is the first checker in the step-6 chain that sets
  `validated_enclosure_mask=7`, with `validator_status_validated=10`. The
  local instance fingerprint is `241045434`, certificate fingerprint is
  `533899248`, validator artifact fingerprint is `316597079`, and artifact
  audit is `476878195`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `64` and
  portfolio v61. The v61 manifest has 65 entries, result coverage
  `(unsat=10, validated=53, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=48, meta=2)`, manifest fingerprint
  `682420766`, acceptance receipt `221112893`, audit receipt `697905232`, and
  readiness fingerprint `731697967`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_repaired_projection_inclusion_validator_tiny.sio`
  checks the repaired margins, needs, slacks, pass/fail masks, promotion mask,
  and deterministic artifact fingerprints. The portfolio tiny gate checks the
  v61 receipt, audit receipt, entry, manifest, result/family coverage,
  acceptance, audit, and readiness fingerprints. The imported gates typecheck
  public APIs and remain known-failure for the current multimodule
  native-lowering exit-139 boundary.
- Scope boundary: v61 validates only the scalar repaired step-6 projection
  inclusion. It does not certify a Taylor-2 flowpipe, a shadowing/invariant
  theorem, or long-time Lorenz behavior. The next semantic step is to connect
  this scalar inclusion receipt to the Taylor-2 flowpipe/remainder path under a
  replayable proof trace.

Current v62 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_taylor2_flowpipe_link_preflight_check()` plus
  fingerprint/audit helpers. This is a link-preflight receipt: it connects the
  v61 repaired scalar inclusion artifact `316597079`/audit `476878195` to the
  v49 Taylor-2 response envelope `506104143`/audit `956295573`, the v54
  remainder obligation `652877359`/audit `344816971`, the v58 replay executor
  `975464919`, and the v55 completed candidate bundle `641464796`.
- The link pins the intended Taylor-2 contract for step 6:
  `taylor_order=2`, selected step `1/100`, source width `i256`, fixed-point
  scale `2^32`, scalar inclusion mask `7`, response mask `127`, remainder
  obligation mask `127`, and proof-trace replay mask `15`.
- The critical non-claim is machine-visible: `flowpipe_proof_mask=0` and
  `link_status_preflight=11`. The local instance fingerprint is `831922075`,
  certificate fingerprint is `453371086`, link-preflight artifact fingerprint
  is `956152878`, and artifact audit is `916252082`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `65` and
  portfolio v62. The v62 manifest has 66 entries, result coverage
  `(unsat=10, validated=54, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=49, meta=2)`, manifest fingerprint
  `806638802`, acceptance receipt `672036277`, audit receipt `51064202`, and
  readiness fingerprint `991960186`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_taylor2_flowpipe_link_preflight_tiny.sio`
  checks the anchor artifacts, Taylor-2 step contract, masks, preflight status,
  and deterministic fingerprints. The portfolio tiny gate checks the v62
  receipt, audit receipt, entry, manifest, result/family coverage, acceptance,
  audit, and readiness fingerprints. The imported gates typecheck public APIs
  and remain known-failure for the current multimodule native-lowering
  exit-139 boundary.
- Scope boundary: v62 is a replayable bridge from scalar inclusion into the
  Taylor-2 response/remainder lane. It does not certify a Taylor-2 flowpipe,
  a continuous-time containment proof, a shadowing/invariant theorem, or
  long-time Lorenz behavior. The next semantic step is to replace
  `flowpipe_proof_mask=0` with an actual Taylor-2 flowpipe proof obligation
  checker.

Current v63 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_taylor2_local_containment_obligation_check()` plus
  fingerprint/audit helpers. This is the first local containment obligation
  checker in the Taylor-2 lane. It consumes the v62 link preflight
  `956152878`/audit `916252082`, the v61 repaired scalar inclusion
  `316597079`/audit `476878195`, the v54 remainder obligation
  `652877359`/audit `344816971`, and the v60 z-margin repair
  `617386880`/audit `478471574`.
- The checker decomposes the repaired inclusion into explicit local
  containment obligations:
  - projected radii `(x=243366, y=319640, z=42448)`;
  - Taylor-2 LTE obligations `(x=47276, y=66549, z=12083)`;
  - total needs `(x=290642, y=386189, z=54531)`;
  - repaired margins `(x=299663, y=389292, z=55000)`;
  - positive slacks `(x=9021, y=3103, z=469)`.
- These checks produce `axis_containment_mask=7` and
  `local_flowpipe_obligation_mask=7`, while deliberately preserving
  `global_flowpipe_claim_mask=0` and `obligation_status_local=12`.
  The local instance fingerprint is `325277065`, certificate fingerprint is
  `986548614`, local-containment artifact fingerprint is `963265870`, and
  artifact audit is `344106968`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `66` and
  portfolio v63. The v63 manifest has 67 entries, result coverage
  `(unsat=10, validated=55, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=50, meta=2)`, manifest fingerprint
  `536676300`, acceptance receipt `891293978`, audit receipt `98798749`, and
  readiness fingerprint `262835863`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_taylor2_local_containment_obligation_tiny.sio`
  checks the radius-plus-LTE decomposition, axis containment mask, local
  obligation mask, global non-claim mask, z repair consumption, and
  deterministic fingerprints. The portfolio tiny gate checks the v63 receipt,
  audit receipt, entry, manifest, result/family coverage, acceptance, audit,
  and readiness fingerprints. The imported gates typecheck public APIs and
  remain known-failure for the current multimodule native-lowering exit-139
  boundary.
- Scope boundary: v63 checks local step-6 containment obligations at target
  scale `1_000_000`. It is still not a continuous-time flowpipe theorem,
  not a global invariant/shadowing proof, and not long-time Lorenz behavior.
  The next semantic step is to attach a checker for the continuous Taylor-2
  time-slab containment between the step endpoints.

Current v64 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_taylor2_time_slab_containment_check()` plus
  fingerprint/audit helpers. This checker consumes the v63 local containment
  artifact `963265870`/audit `344106968`, the v62 Taylor-2 link preflight
  `956152878`/audit `916252082`, and the v54 remainder obligation
  `652877359`/audit `344816971`.
- The checker accounts for continuous displacement inside the selected
  step by binding derivative bounds to upward-rounded sweep ceilings over
  `dt=1/100`:
  - derivative bounds `(dx=46888150, dy=47662713, dz=7699049)`;
  - sweep ceilings `(x=468882, y=476628, z=76991)`;
  - slab margins `(x=470000, y=477000, z=80000)`;
  - positive slab slacks `(x=1118, y=372, z=3009)`.
- It validates the ceiling facts by multiplication, not dynamic division:
  `(sweep - 1) * 100 < derivative_bound <= sweep * 100` on each axis.
  These checks produce `time_slab_containment_mask=7` and
  `local_time_slab_obligation_mask=7`, while deliberately preserving
  `global_flowpipe_claim_mask=0` and `time_slab_status_local=13`.
  The time-slab instance fingerprint is `612699967`, certificate fingerprint
  is `57659327`, artifact fingerprint is `230532791`, and artifact audit is
  `370048214`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `67` and
  portfolio v64. The v64 manifest has 68 entries, result coverage
  `(unsat=10, validated=56, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=51, meta=2)`, manifest fingerprint
  `827592666`, acceptance receipt `558014348`, audit receipt `638977732`, and
  readiness fingerprint `94199545`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_taylor2_time_slab_containment_tiny.sio`
  checks derivative-bound ceilings, time-slab margins, positive slacks, local
  obligation mask, global non-claim mask, v63 bridge consumption, and
  deterministic fingerprints. The portfolio tiny gate checks the v64 receipt,
  audit receipt, entry, manifest, result/family coverage, acceptance, audit,
  and readiness fingerprints. The imported gates typecheck public APIs and
  remain known-failure for the current multimodule native-lowering exit-139
  boundary.
- Scope boundary: v64 is a local continuous-time displacement containment
  obligation for the single step-6 slab under recorded derivative bounds. It
  is still not a global invariant, not a shadowing theorem, not a full Taylor-2
  flowpipe proof, and not long-time Lorenz behavior.

Current v65 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_taylor2_flowpipe_obligation_check()` plus
  fingerprint/audit helpers. This checker composes the v64 time-slab
  containment artifact `230532791`/audit `370048214`, the v63 endpoint-style
  local containment artifact `963265870`/audit `344106968`, the v49 Taylor-2
  response envelope `506104143`/audit `956295573`, the v54 remainder
  obligation `652877359`/audit `344816971`, and the v62 link preflight
  `956152878`/audit `916252082`.
- The checker binds four local prerequisites into a single flowpipe-obligation
  readiness mask:
  - endpoint containment mask `7`;
  - time-slab containment mask `7`;
  - response-envelope mask `127`;
  - remainder-obligation mask `127`;
  - resulting `composition_ready_mask=15` and
    `local_flowpipe_obligation_mask=15`.
- The critical non-claim remains machine-visible: `local_flowpipe_proof_mask=0`
  and `global_flowpipe_claim_mask=0`, with `flowpipe_obligation_status=14`.
  The v65 instance fingerprint is `854680028`, certificate fingerprint is
  `855359915`, flowpipe-obligation artifact fingerprint is `646304688`, and
  artifact audit is `65498507`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `68` and
  portfolio v65. The v65 manifest has 69 entries, result coverage
  `(unsat=10, validated=57, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=52, meta=2)`, manifest fingerprint
  `890903826`, acceptance receipt `595175716`, audit receipt `899868800`, and
  readiness fingerprint `449901436`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_taylor2_flowpipe_obligation_tiny.sio`
  checks the four prerequisite masks, composition mask, local obligation mask,
  proof/global non-claim masks, step contract, dependency closure, and
  deterministic fingerprints. The portfolio tiny gate checks the v65 receipt,
  audit receipt, entry, manifest, result/family coverage, acceptance, audit,
  and readiness fingerprints. The imported gates typecheck public APIs and
  remain known-failure for the current multimodule native-lowering exit-139
  boundary.
- Scope boundary: v65 is a local flowpipe-obligation composition receipt. It
  is stronger than the v62 preflight and v64 single-slab displacement check
  because it binds endpoint, slab, response, and remainder prerequisites
  together. It is still not a full local Taylor-2 flowpipe proof, not a global
  invariant, not a shadowing theorem, and not long-time Lorenz behavior.

Current v66 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_step6_taylor2_local_flowpipe_proof_check()` plus
  fingerprint/audit helpers. This checker sits directly above v65 and consumes
  the v65 flowpipe-obligation artifact `646304688`/audit `65498507`, the v64
  time-slab artifact `230532791`/audit `370048214`, the v63 local-containment
  artifact `963265870`/audit `344106968`, the v49 response envelope
  `506104143`/audit `956295573`, the v54 remainder obligation
  `652877359`/audit `344816971`, and the v62 Taylor-2 link preflight
  `956152878`/audit `916252082`.
- The checker upgrades the local receipt by requiring all local prerequisite
  masks before setting the proof bits:
  - endpoint containment mask `7`;
  - time-slab containment mask `7`;
  - response-envelope mask `127`;
  - remainder-obligation mask `127`;
  - resulting `composition_ready_mask=15`;
  - `local_flowpipe_obligation_mask=15`;
  - `local_flowpipe_proof_mask=15`;
  - `global_flowpipe_claim_mask=0`;
  - `local_flowpipe_status=15`.
- The v66 instance fingerprint is `820830981`, certificate fingerprint is
  `807625549`, local-flowpipe-proof artifact fingerprint is `813941448`, and
  artifact audit is `693709162`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `69` and
  portfolio v66. The v66 manifest has 70 entries, result coverage
  `(unsat=10, validated=58, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=53, meta=2)`, manifest fingerprint
  `525173872`, acceptance receipt `907656427`, audit receipt `703795364`, and
  readiness fingerprint `57670279`.
- The tiny gate
  `tests/run-pass/lorenz_i256_step6_taylor2_local_flowpipe_proof_tiny.sio`
  checks the checked local proof mask, global non-claim mask, local status,
  dependency closure, and deterministic fingerprints. The portfolio tiny gate
  checks the v66 receipt, audit receipt, entry, manifest, result/family
  coverage, acceptance, audit, and readiness fingerprints. The imported gates
  typecheck public APIs and remain known-failure for the current multimodule
  native-lowering exit-139 boundary.
- Scope boundary: v66 is a local single-step Taylor-2 flowpipe proof receipt
  under recorded prerequisite certificates. It is stronger than v65 because it
  replaces `local_flowpipe_proof_mask=0` with checked local proof bits, but it
  still deliberately preserves `global_flowpipe_claim_mask=0`. It is not a
  global invariant, not a shadowing theorem, and not long-time Lorenz behavior.

Current v67 addition:

- `stdlib/systems/lorenz_i256_cert.sio` adds
  `lorenz_i256_trajectory5_step6_local_flowpipe_bridge_check()` plus
  fingerprint/audit helpers. This checker bridges the existing five-step
  point-trajectory certificate to the v66 local step-6 Taylor-2 flowpipe proof
  receipt. It consumes:
  - trajectory5 manifest `214180161`;
  - five-step chain anchor `23144051`;
  - final step certificate `561641681`;
  - step-6 center artifact `134624236`/audit `643317565`;
  - v66 local-flowpipe-proof artifact `813941448`/audit `693709162`.
- The bridge checks the coordinate adjacency between the five-step point
  trajectory endpoint and the step-6 start point:
  - prefix final `(5309941770,10058731223,4084837935)`;
  - step-6 start `(5309941770,10058731223,4084837935)`;
  - step-6 center `(5784820704,11394426024,4100266743)`;
  - `adjacency_mask=7`;
  - `chain_extension_mask=15`;
  - `global_flowpipe_claim_mask=0`;
  - `bridge_status=16`.
- The v67 system instance fingerprint is `286435512`, certificate fingerprint
  is `352296669`, bridge artifact fingerprint is `881240986`, and bridge audit
  is `732374685`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `70` and
  portfolio v67. The v67 manifest has 71 entries, result coverage
  `(unsat=10, validated=59, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=54, meta=2)`, manifest fingerprint
  `94234647`, acceptance receipt `858480510`, audit receipt `618227054`, and
  readiness fingerprint `664541158`.
- Scope boundary: v67 is a chain-extension bridge receipt, not a proof that
  all six steps have local Taylor-2 flowpipe proofs. The first five steps are
  still represented by the older point-trajectory certificate; the only local
  Taylor-2 flowpipe proof receipt in this bridge is the v66 step-6 receipt.
  It remains non-global and keeps `global_flowpipe_claim_mask=0`.

Current v68 addition:

- Added `lorenz_i256_step5_taylor2_local_flowpipe_seed_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_local_flowpipe_seed_tiny.sio` and
  the imported API gate. This starts the earlier-step backfill requested by the
  v67 scope note, but deliberately marks the result as a seed/preflight, not a
  complete local flowpipe proof.
- The checker replays the step-5 i256 quotient/remainder witnesses for
  `(4920371098,8816077911,4092986080)` to
  `(5309941770,10058731223,4084837935)`, anchors the five-step trajectory
  manifest `214180161`, chain anchor `23144051`, final step certificate
  `561641681`, and the v67 bridge `881240986`/`732374685`.
- The v68 masks are intentionally explicit:
  `point_replay_mask=127`, `endpoint_mask=7`, `trajectory_anchor_mask=7`,
  `bridge_anchor_mask=3`, `local_flowpipe_seed_mask=15`,
  `local_flowpipe_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `seed_status=17`.
- The v68 system instance fingerprint is `629109830`, certificate fingerprint
  is `993750007`, seed artifact fingerprint is `931268564`, and seed audit is
  `900912795`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `71` and
  portfolio v68. The v68 manifest has 72 entries, result coverage
  `(unsat=10, validated=60, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=55, meta=2)`, manifest fingerprint
  `219898081`, acceptance receipt `647626897`, audit receipt `792429254`, and
  readiness fingerprint `144889913`.
- Scope boundary: v68 proves that the step-5 point update can be replayed as a
  typed i256 seed for later local Taylor-2 flowpipe work and that it is wired
  into the existing v67 chain bridge. It does not claim step-5 local
  containment, a step-5 Taylor remainder enclosure, a five-step local flowpipe
  chain, or any global Lorenz/shadowing theorem.

Current v69 addition:

- Added `lorenz_i256_step5_taylor2_point_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_point_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This promotes the v68 step-5 seed into a
  first point-time-slab containment obligation without importing the richer
  step-6 radius/enclosure assumptions.
- The checker derives integer derivative bounds from the fixed step-5 point at
  target scale `1_000_000`: `dx_bound=9070400`, `dy_bound=28932778`,
  `dz_bound=189714`. It then checks `ceil(bound / 100)` sweeps
  `(90704,289328,1898)` for `dt=1/100` against slab margins
  `(91000,290000,2000)` with slacks `(296,672,102)`.
- The v69 masks are intentionally scoped:
  `seed_anchor_mask=3`, `derivative_bound_mask=7`,
  `sweep_ceiling_mask=7`, `slab_margin_mask=7`,
  `time_slab_containment_mask=7`, `local_time_slab_seed_mask=15`,
  `local_time_slab_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `time_slab_status=18`.
- The v69 system instance fingerprint is `403599593`, certificate fingerprint
  is `883271225`, point-time-slab artifact fingerprint is `693506658`, and
  point-time-slab audit is `389624150`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `72` and
  portfolio v69. The v69 manifest has 73 entries, result coverage
  `(unsat=10, validated=61, optimal=1, sat=1)`, checker-family coverage
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=56, meta=2)`, manifest fingerprint
  `428032311`, acceptance receipt `103473156`, audit receipt `10042504`, and
  readiness fingerprint `838284677`.
- Scope boundary: v69 is a point-derived time-slab seed for step 5. It proves
  integer bound/ceiling/slack consistency for the certified point derivative,
  not a ball enclosure, not a Taylor remainder enclosure, not a local flowpipe
  proof, and not a global Lorenz/shadowing theorem.

Current v70 addition:

- Added `lorenz_i256_step5_taylor2_response_envelope_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_response_envelope_tiny.sio` and
  imported API/portfolio gates. This records the v69 point time-slab seed as a
  typed response-envelope obligation stub for the next step-5 local Taylor-2
  enclosure artifacts.
- The v70 checker anchors both immediate prerequisites: the v69 point
  time-slab artifact/audit pair `693506658`/`389624150` and the v68 local
  flowpipe seed artifact/audit pair `931268564`/`900912795`. It fixes
  `step_index=5`, `taylor_order=2`, `dt=1/100`, `scale_log2=32`, and
  `source_width_bits=256`.
- The response contract requires a ball-enclosure candidate family with four
  future artifacts: center, radius, remainder, and proof trace. The masks are
  intentionally explicit: `anchor_mask=15`, `contract_mask=15`,
  `candidate_requirement_mask=15`, `response_envelope_mask=127`,
  `local_response_seed_mask=15`, `local_response_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, `response_status=19`, and `ok_mask=255`.
- The v70 system instance fingerprint is `805286304`, certificate fingerprint
  is `160196187`, response-envelope artifact fingerprint is `422368008`, and
  response-envelope audit is `577119901`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `73` and
  portfolio v70. The v70 manifest has 74 entries; the existing portfolio result
  code is `validated`, but this is a bookkeeping classification for the
  obligation stub, not a Lorenz enclosure proof. Result counters are
  `(unsat=10, validated=62, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=57, meta=2)`, manifest fingerprint
  `880702042`, acceptance receipt `242887974`, audit receipt `225019937`, and
  readiness fingerprint `678530115`.
- Scope boundary: v70 is a candidate-required response envelope. It does not
  compute the center, radius, remainder, or proof-trace artifacts yet; it does
  not prove a local enclosure, a five-step flowpipe, or any global
  Lorenz/shadowing theorem.

Current v71 addition:

- Added `lorenz_i256_step5_taylor2_center_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_center_artifact_tiny.sio` and
  imported API/portfolio gates. This records the first concrete v70-required
  candidate artifact: the step-5 center replay from the existing i256
  quotient/remainder witnesses.
- The v71 checker anchors the v70 response-envelope artifact/audit
  `422368008`/`577119901`, the v69 point time-slab artifact/audit
  `693506658`/`389624150`, and the v68 local-flowpipe seed artifact/audit
  `931268564`/`900912795`.
- The center replay uses the step-5 point start
  `(4920371098,8816077911,4092986080)` at scale `2^32`, `dt=42949672 r96`
  for `1/100`, and the supplied division witnesses
  `dy_scaled=133081411987 r405415232`,
  `xy_scaled=10099814960 r2149068118`,
  `beta_z/3=10914629546 r2`,
  `x_inc=389570672 r2544410448`,
  `y_inc=1242653312 r4228538720`, and
  `z_drop=8148145 r2911449872`. The resulting center is
  `(5309941770,10058731223,4084837935)`.
- The v71 masks are deliberately non-global:
  `point_replay_mask=127`, `anchor_mask=63`, `final_center_mask=7`,
  `center_candidate_mask=15`, `local_center_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and `ok_mask=255`.
- The v71 system instance fingerprint is `668835069`, certificate fingerprint
  is `823414876`, center artifact fingerprint is `171451971`, and center
  artifact audit is `862108922`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `74` and
  portfolio v71. The v71 manifest has 75 entries; the existing portfolio result
  code is `validated`, but this is still a bookkeeping classification for a
  center candidate artifact, not a Lorenz enclosure proof. Result counters are
  `(unsat=10, validated=63, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=58, meta=2)`, manifest fingerprint
  `257446237`, acceptance receipt `717358762`, audit receipt `644138284`, and
  readiness fingerprint `176345240`.
- Scope boundary: v71 computes/checks the center candidate replay only. It does
  not compute the radius, remainder, or proof-trace artifacts; it does not prove
  local containment, a five-step flowpipe, or any global Lorenz/shadowing
  theorem.

Current v72 addition:

- Added `lorenz_i256_step5_taylor2_radius_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_radius_artifact_tiny.sio` and
  imported API/portfolio gates. This records the second concrete v70-required
  candidate artifact: a conservative step-5 radius propagation candidate.
- The v72 checker anchors the v71 center artifact/audit
  `171451971`/`862108922`, the v70 response-envelope artifact/audit
  `422368008`/`577119901`, and the v69 point time-slab artifact/audit
  `693506658`/`389624150`.
- The prior target-scale radii are the v69 time-slab margins
  `(91000,290000,2000)`. These convert by ceiling to source-scale radii
  `(390842024,1245540516,8589935)` at `2^32`, with conversion excesses still
  below the target scale `1_000_000`.
- The radius replay uses the step-5 start center
  `(4920371098,8816077911,4092986080)` and computes conservative derivative
  radius bounds:
  `dx_rad=16363825400`, `prod_x_rhoz=10581737366`,
  `dy_rad=11827277882`, `prod_xy=2342514896`,
  `beta_rad=22906495`, and `dz_rad=2365421391`.
  Here `beta_rad=22906495` is deliberately one tick above the exact ceiling
  `22906494`, preserving the inherited conservative guard shape from the
  step-6 radius artifact; it is an overbound, not a tight-bound claim.
  With `dt=42949672`, the next source-scale radius candidate is
  `(554480275,1363813293,32244149)`, under caps
  `(950000000,1800000000,100000000)`.
- The v72 masks remain non-global:
  `ok_mask=127`, `local_radius_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v72 system instance fingerprint is `476305583`, certificate fingerprint
  is `839289344`, radius artifact fingerprint is `815806077`, and radius
  artifact audit is `655666714`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `75` and
  portfolio v72. The v72 manifest has 76 entries; the existing portfolio result
  code is `validated`, but this remains a bookkeeping classification for a
  radius candidate artifact, not a Lorenz enclosure proof. Result counters are
  `(unsat=10, validated=64, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=59, meta=2)`, manifest fingerprint
  `86228816`, acceptance receipt `370906727`, audit receipt `294632541`, and
  readiness fingerprint `859564353`.
- Scope boundary: v72 computes/checks a radius candidate only. It does not
  compute the remainder or proof-trace artifacts; it does not prove local
  containment, a five-step flowpipe, or any global Lorenz/shadowing theorem.

Current v73 addition:

- Added `lorenz_i256_step5_taylor2_remainder_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_remainder_obligation_tiny.sio`
  and imported API/portfolio gates. This records the third concrete
  v70-required step-5 artifact: a Taylor-2 remainder obligation over the v72
  source-scale radius ball.
- The v73 checker anchors the v70 response-envelope artifact/audit
  `422368008`/`577119901`, the v71 center artifact/audit
  `171451971`/`862108922`, and the v72 radius artifact/audit
  `815806077`/`655666714`.
- The source-scale box uses start center
  `(4920371098,8816077911,4092986080)` with source radii
  `(390842024,1245540516,8589935)`, giving absolute bounds
  `(5311213122,10061618427,4101576015)` and
  `rhoz_abs_bound=116174688143`.
- The first-derivative bounds are
  `x_prime_bound=153728315490`, `y_prime_bound=153724767780`, and
  `z_prime_bound=23379865897`. The second-derivative bounds are
  `x_second_bound=3074530832700`, `y_second_bound=4340838038257`, and
  `z_second_bound=612576495421`.
- With `dt_q=42949672` at source scale `2^32`, the source LTE ceilings are
  `(153726535,217041893,30628824)`. Converted back to ppm by ceiling, the
  target LTE obligations are `(35793,50535,7132)`, under caps
  `(40000,55000,10000)` with slacks `(4207,4465,2868)`.
- The v73 masks remain non-global:
  `ok_mask=127`, `local_remainder_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v73 system instance fingerprint is `77648112`, certificate fingerprint
  is `754547684`, remainder-obligation artifact fingerprint is `69995750`,
  and artifact audit is `380914208`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `76` and
  portfolio v73. The v73 manifest has 77 entries; the existing portfolio result
  code is `validated`, but this remains a bookkeeping classification for a
  remainder-obligation replay, not a Lorenz enclosure proof. Result counters are
  `(unsat=10, validated=65, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=60, meta=2)`, manifest fingerprint
  `611181728`, acceptance receipt `245041585`, audit receipt `450254687`, and
  readiness fingerprint `256503731`.
- Scope boundary: v73 computes/checks a Taylor-2 remainder obligation only. It
  does not construct a proof trace, completed candidate bundle, local
  containment proof, five-step flowpipe, invariant/shadowing proof, or global
  Lorenz theorem.

Current v74 addition:

- Added `lorenz_i256_step5_taylor2_completed_candidate_bundle_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_completed_candidate_bundle_tiny.sio`
  and imported API/portfolio gates. This composes the four concrete step-5
  candidate artifacts into a completed candidate bundle: response envelope,
  center artifact, radius artifact, and remainder obligation.
- The v74 checker anchors the v70 response-envelope artifact/audit
  `422368008`/`577119901`, the v71 center artifact/audit
  `171451971`/`862108922`, the v72 radius artifact/audit
  `815806077`/`655666714`, and the v73 remainder-obligation artifact/audit
  `69995750`/`380914208`.
- The completed-bundle masks are:
  `required_artifact_mask=15`, `provided_artifact_mask=15`,
  `missing_artifact_mask=0`, `candidate_status_complete=4`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255` so the
  non-claim masks are first-class checked bits, not prose-only caveats.
  Here `candidate_status_complete=4` means only that the required candidate
  artifact bundle is present; it is not a proof-status code.
- The v74 system instance fingerprint is `372232202`, certificate fingerprint
  is `128376272`, completed-bundle artifact fingerprint is `989883403`, and
  artifact audit is `754841079`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `77` and
  portfolio v74. The v74 manifest has 78 entries; the existing portfolio result
  code is `validated`, but this remains a bookkeeping classification for a
  completed candidate bundle, not a local containment proof. Result counters are
  `(unsat=10, validated=66, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=61, meta=2)`, manifest fingerprint
  `522438539`, acceptance receipt `213191343`, audit receipt `457602463`, and
  readiness fingerprint `635257553`.
- Scope boundary: v74 completes the candidate-artifact bundle only. It does not
  construct a proof trace, replay executor, local containment proof, five-step
  flowpipe, invariant/shadowing proof, or global Lorenz theorem.

Current v75 addition:

- Added `lorenz_i256_step5_taylor2_proof_trace_skeleton_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_proof_trace_skeleton_tiny.sio`
  and imported API/portfolio gates. This turns the v74 completed candidate
  bundle into a replay-shaped trace skeleton for the same step-5 Taylor-2
  artifact set.
- The v75 checker anchors the completed bundle/audit
  `989883403`/`754841079`, response envelope `422368008`, center artifact
  `171451971`, radius artifact `815806077`, and remainder obligation
  `69995750`.
- The proof-trace skeleton records `trace_version=1`, `trace_kind_step5=5`,
  `trace_node_count=4`, `dependency_edge_count=4`,
  `obligation_mask=15`, `replay_order_mask=15`,
  `dependency_root_mask=15`, and `trace_status_skeleton=5`.
  It also keeps `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
  The checker uses `ok_mask=255`, including an explicit non-claim bit for the
  containment/global-flowpipe masks.
- The v75 system instance fingerprint is `991104836`, certificate fingerprint
  is `272679598`, proof-trace-skeleton artifact fingerprint is `971755585`,
  and artifact audit is `205079086`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `78` and
  portfolio v75. The v75 manifest has 79 entries; the existing portfolio result
  code is `validated`, but this remains a bookkeeping classification for a
  trace skeleton, not a replay executor or enclosure proof. Result counters are
  `(unsat=10, validated=67, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=62, meta=2)`, manifest fingerprint
  `698709746`, acceptance receipt `980834451`, audit receipt `340435167`, and
  readiness fingerprint `466346197`.
- Scope boundary: v75 constructs only the proof-trace skeleton. It does not
  execute the replay, prove local containment, assemble a five-step flowpipe,
  prove invariant/shadowing, or state any global Lorenz theorem.

Current v76 addition:

- Added `lorenz_i256_step5_taylor2_replay_preflight_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_replay_preflight_tiny.sio`
  and imported API/portfolio gates. This consumes the v75 proof-trace skeleton
  and checks the preflight masks needed before a replay executor is allowed to
  exist.
- The v76 checker anchors the proof-trace skeleton/audit
  `971755585`/`205079086` and the completed bundle `989883403`.
  It records `trace_version=1`, `replay_version=1`, `trace_node_count=4`,
  `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, and `replay_status_preflight=6`.
- The non-claim masks remain explicit:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255`, including a
  first-class non-claim bit for the containment/global-flowpipe masks.
- The v76 system instance fingerprint is `282143781`, certificate fingerprint
  is `396051824`, replay-preflight artifact fingerprint is `847188269`, and
  artifact audit is `425385087`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `79` and
  portfolio v76. The v76 manifest has 80 entries; the portfolio result code is
  still only a bookkeeping classification for a replay preflight. Result
  counters are `(unsat=10, validated=68, optimal=1, sat=1)`,
  checker-family counters are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=63,
  meta=2)`, manifest fingerprint `383424151`, acceptance receipt `573836383`,
  audit receipt `157740163`, and readiness fingerprint `320835674`.
- Scope boundary: v76 verifies replay preflight readiness only. It does not
  execute the replay, prove local containment, assemble a five-step flowpipe,
  prove invariant/shadowing, or state any global Lorenz theorem.

Current v77 addition:

- Added `lorenz_i256_step5_taylor2_replay_executor_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_replay_executor_tiny.sio`
  and imported API/portfolio gates. This consumes the v76 replay preflight and
  checks node/edge replay receipts for the step-5 Taylor-2 candidate artifacts.
- The v77 checker anchors replay preflight/audit `847188269`/`425385087`,
  proof-trace skeleton `971755585`, completed bundle `989883403`, response
  envelope `422368008`, center artifact `171451971`, radius artifact
  `815806077`, and remainder obligation `69995750`.
- It records `trace_version=1`, `replay_engine_version=1`,
  `trace_node_count=4`, `dependency_edge_count=4`,
  `node_receipt_mask=15`, `edge_receipt_mask=15`,
  `replayed_node_mask=15`, `predecessor_ready_mask=15`, and
  `replay_status_executed=7`.
- The non-claim masks remain explicit:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255`, including a
  first-class non-claim bit for the containment/global-flowpipe masks.
- The v77 system instance fingerprint is `201510806`, certificate fingerprint
  is `882863410`, replay-executor artifact fingerprint is `225293164`, and
  artifact audit is `897523976`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `80` and
  portfolio v77. The v77 manifest has 81 entries; the portfolio result code is
  still only a bookkeeping classification for a replay executor over candidate
  artifacts. Result counters are `(unsat=10, validated=69, optimal=1, sat=1)`,
  checker-family counters are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=64,
  meta=2)`, manifest fingerprint `490309693`, acceptance receipt `28811066`,
  audit receipt `963881119`, and readiness fingerprint `38408153`.
- Scope boundary: v77 executes replay bookkeeping only. It does not prove local
  containment, assemble a five-step flowpipe, prove invariant/shadowing, or
  state any global Lorenz theorem.

Current v78 addition:

- Added `lorenz_i256_step5_taylor2_enclosure_validator_guard_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_enclosure_validator_guard_tiny.sio`
  and imported API/portfolio gates. This consumes the v77 replay executor and
  checks the step-5 Taylor-2 radius-plus-remainder enclosure budget against a
  conservative guard margin.
- The v78 checker anchors replay executor/audit `225293164`/`897523976`,
  completed bundle `989883403`, radius artifact `815806077`, and remainder
  obligation `69995750`. It intentionally does not introduce a projection
  dependency for step 5.
- It records `target_scale=1000000`, radius ppm `(242489, 319183, 42444)`,
  LTE ppm `(35793, 50535, 7132)`, computed needs
  `(278282, 369718, 49576)`, and guard margins `(279000, 370000, 50000)`.
  The resulting slack is `(718, 282, 424)` ppm with inclusion masks
  `pass=7`, `fail=0`.
- The non-claim masks remain explicit:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255`, including
  first-class bits for containment blocking and non-claim status.
- The v78 system instance fingerprint is `658127438`, certificate fingerprint
  is `85740445`, enclosure-guard artifact fingerprint is `40755632`, and
  artifact audit is `556529793`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `81` and
  portfolio v78. The v78 manifest has 82 entries; the portfolio result code is
  still only a bookkeeping classification for a guarded enclosure budget over
  candidate artifacts. Result counters are `(unsat=10, validated=70,
  optimal=1, sat=1)`, checker-family counters are `(SAT=6, SMT=4, PB=3,
  graph=2, Lorenz=65, meta=2)`, manifest fingerprint `312911035`,
  acceptance receipt `237569040`, audit receipt `885583200`, and readiness
  fingerprint `501874592`.
- Scope boundary: v78 proves only that the recorded step-5 guard operands match
  the chosen scalar margins and that no local-containment or global-flowpipe
  claim was promoted. It does not prove local containment, assemble a five-step
  flowpipe, prove invariant/shadowing, or state any global Lorenz theorem.

Current v79 addition:

- Added `lorenz_i256_step5_taylor2_local_containment_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_local_containment_obligation_tiny.sio`
  and imported API/portfolio gates. This consumes the v78 enclosure validator
  guard and turns the passing scalar budget into a local-containment obligation
  receipt for the step-5 Taylor-2 candidate.
- The v79 checker anchors enclosure guard/audit `40755632`/`556529793`,
  replay executor/audit `225293164`/`897523976`, completed bundle
  `989883403`, radius artifact `815806077`, and remainder obligation
  `69995750`.
- It records Taylor order `2`, `target_scale=1000000`, selected `dt=1/100`,
  radius ppm `(242489, 319183, 42444)`, LTE ppm `(35793, 50535, 7132)`,
  computed needs `(278282, 369718, 49576)`, margins
  `(279000, 370000, 50000)`, and slacks `(718, 282, 424)`.
- The local masks are explicit: `axis_containment_mask=7`,
  `local_flowpipe_obligation_mask=7`, `obligation_status_local=12`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255`, including
  first-class bits for predecessor anchors, candidate-artifact anchors,
  radius-plus-LTE arithmetic, axis containment, Taylor contract, local
  obligation status, non-claim status, and exact slack profile.
- The v79 system instance fingerprint is `289647059`, certificate fingerprint
  is `886367159`, local-containment-obligation artifact fingerprint is
  `291436394`, and artifact audit is `876674722`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `82` and
  portfolio v79. The v79 manifest has 83 entries; the portfolio result code is
  still only a bookkeeping classification for a local obligation receipt over
  candidate artifacts. Result counters are `(unsat=10, validated=71,
  optimal=1, sat=1)`, checker-family counters are `(SAT=6, SMT=4, PB=3,
  graph=2, Lorenz=66, meta=2)`, manifest fingerprint `814116954`,
  acceptance receipt `89796164`, audit receipt `164051989`, and readiness
  fingerprint `296359452`.
- Scope boundary: v79 records a local-containment obligation under the v78
  scalar guard, but it is not yet the local flowpipe proof receipt, does not
  assemble a five-step flowpipe, and does not prove invariant/shadowing or any
  global Lorenz theorem.

Current v80 addition:

- Added `lorenz_i256_step5_taylor2_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This consumes the v79 local-containment
  obligation and the v69 point time-slab seed to record a step-5 Taylor-2
  time-slab containment receipt.
- The v80 checker anchors local containment obligation/audit
  `291436394`/`876674722`, point time-slab artifact/audit
  `693506658`/`389624150`, and remainder obligation `69995750`.
- It reuses the v69 point-derived derivative bounds and sweep ceilings:
  `dx_bound=9070400`, `dy_bound=28932778`, `dz_bound=189714`,
  `dt=1/100`, sweeps `(90704, 289328, 1898)`, slab margins
  `(91000, 290000, 2000)`, and slacks `(296, 672, 102)`.
- The time-slab masks are explicit: `time_slab_containment_mask=7`,
  `local_time_slab_obligation_mask=7`, `time_slab_status_local=13`, and
  `global_flowpipe_claim_mask=0`. The checker uses `ok_mask=255`, including
  first-class bits for predecessor anchors, remainder anchor, sweep ceilings,
  slab margins, `dt` contract, local obligation status, non-claim status, and
  the v79 bridge.
- The v80 system instance fingerprint is `513837001`, certificate fingerprint
  is `463594250`, time-slab-containment artifact fingerprint is `253054033`,
  and artifact audit is `661134385`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `83` and
  portfolio v80. The v80 manifest has 84 entries; the portfolio result code is
  still only a bookkeeping classification for a local time-slab receipt over
  candidate artifacts. Result counters are `(unsat=10, validated=72,
  optimal=1, sat=1)`, checker-family counters are `(SAT=6, SMT=4, PB=3,
  graph=2, Lorenz=67, meta=2)`, manifest fingerprint `768140968`,
  acceptance receipt `933320065`, audit receipt `516949051`, and readiness
  fingerprint `246310129`.
- Scope boundary: v80 records a local time-slab containment receipt under the
  v79 obligation and v69 point-derived slab seed. It is not yet the local
  flowpipe proof receipt, does not assemble a five-step flowpipe, and does not
  prove invariant/shadowing or any global Lorenz theorem.

Current v81 addition:

- Added `lorenz_i256_step5_taylor2_flowpipe_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_flowpipe_obligation_tiny.sio`
  and imported API/portfolio gates. This promotes the step-5 backfill from the
  v80 time-slab receipt into a local flowpipe-obligation composition receipt.
- The v81 checker anchors the v80 time-slab-containment artifact/audit
  `253054033`/`661134385`, the v79 local-containment obligation/audit
  `291436394`/`876674722`, the v70 response envelope/audit
  `422368008`/`577119901`, the v73 remainder obligation/audit
  `69995750`/`380914208`, and the v68 local-flowpipe seed/audit
  `931268564`/`900912795`.
- The local composition mask is first-class:
  `endpoint_containment_mask=7`, `time_slab_containment_mask=7`,
  `response_envelope_mask=127`, `remainder_obligation_mask=127`,
  `composition_ready_mask=15`, and `local_flowpipe_obligation_mask=15`.
  The non-claim boundary is also machine-visible:
  `local_flowpipe_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `flowpipe_obligation_status=20`.
- The v81 system instance fingerprint is `725158511`, certificate fingerprint
  is `914660586`, flowpipe-obligation artifact fingerprint is `108175917`,
  and artifact audit is `495875254`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `84` and
  portfolio v81. The v81 manifest has 85 entries; result counters are
  `(unsat=10, validated=73, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=68, meta=2)`, manifest fingerprint
  `827330719`, acceptance receipt `536898882`, audit receipt `656019738`, and
  readiness fingerprint `555645792`.
- Scope boundary: v81 is a local flowpipe-obligation composition receipt. It
  is stronger than v80 because it binds endpoint containment, time-slab
  containment, response-envelope, remainder-obligation, and seed prerequisites
  together. It is still not the local Taylor-2 flowpipe proof, not a five-step
  flowpipe, not an invariant/shadowing theorem, and not a global Lorenz claim.

Current v82 addition:

- Added `lorenz_i256_step5_taylor2_local_flowpipe_proof_check()` plus
  `tests/run-pass/lorenz_i256_step5_taylor2_local_flowpipe_proof_tiny.sio`
  and imported API/portfolio gates. This consumes the v81 step-5 flowpipe
  obligation and records a local single-step Taylor-2 flowpipe proof receipt
  for step 5.
- The v82 checker anchors the v81 flowpipe-obligation artifact/audit
  `108175917`/`495875254`, the v80 time-slab-containment artifact/audit
  `253054033`/`661134385`, the v79 local-containment obligation/audit
  `291436394`/`876674722`, the v70 response envelope/audit
  `422368008`/`577119901`, the v73 remainder obligation/audit
  `69995750`/`380914208`, and the v68 local-flowpipe seed/audit
  `931268564`/`900912795`.
- The local proof masks are explicit:
  `endpoint_proof_mask=7`, `time_slab_proof_mask=7`,
  `response_proof_mask=127`, `remainder_proof_mask=127`,
  `composition_ready_mask=15`, `local_flowpipe_obligation_mask=15`,
  `local_flowpipe_proof_mask=15`, `global_flowpipe_claim_mask=0`, and
  `local_flowpipe_status=21`.
- The v82 system instance fingerprint is `251889352`, certificate fingerprint
  is `67726687`, local-flowpipe-proof artifact fingerprint is `965829478`,
  and artifact audit is `618466007`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `85` and
  portfolio v82. The v82 manifest has 86 entries; result counters are
  `(unsat=10, validated=74, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=69, meta=2)`, manifest fingerprint
  `572498026`, acceptance receipt `940454269`, audit receipt `277255258`, and
  readiness fingerprint `271235202`.
- Scope boundary: v82 is a local single-step Taylor-2 flowpipe proof receipt
  for step 5 under recorded prerequisite certificates. It is stronger than v81
  because `local_flowpipe_proof_mask` is now `15`, but it is not a five-step
  local flowpipe chain, not an invariant/shadowing theorem, and not a global
  Lorenz claim.

Current v83 addition:

- Added `lorenz_i256_step4_taylor2_local_flowpipe_seed_check()` plus
  `tests/run-pass/lorenz_i256_step4_taylor2_local_flowpipe_seed_tiny.sio`
  and imported API/portfolio gates. This starts the step-4 backfill with the
  same seed/preflight discipline used for v68 on step 5; it deliberately does
  not skip directly to containment or a local proof.
- The checker replays the step-4 i256 quotient/remainder witnesses for
  `(4617762706,7643846696,4120687759)` to
  `(4920371098,8816077911,4092986080)`, anchors the trajectory5 manifest
  `214180161`, the four-step chain anchor `737039167`, step-4 certificate
  `753371133`, five-step chain anchor `23144051`, and step-5 certificate
  `561641681`.
- The v83 masks keep the non-claim boundary machine-visible:
  `point_replay_mask=127`, `endpoint_mask=7`, `trajectory_anchor_mask=7`,
  `suffix_anchor_mask=3`, `local_flowpipe_seed_mask=15`,
  `local_flowpipe_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `seed_status=22`.
- The v83 system instance fingerprint is `685779442`, certificate fingerprint
  is `804235029`, step-4 local-flowpipe seed artifact fingerprint is
  `563331437`, and artifact audit is `684718822`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `86` and
  portfolio v83. The v83 manifest has 87 entries; result counters are
  `(unsat=10, validated=75, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=70, meta=2)`, manifest fingerprint
  `652012103`, acceptance receipt `360030248`, audit receipt `141219813`, and
  readiness fingerprint `470514407`.
- Scope boundary: v83 proves that the step-4 point update can be replayed as a
  typed i256 seed and that its prefix/suffix chain anchors line up with the
  existing trajectory receipts. It is not step-4 time-slab containment, not a
  response/remainder/local-containment obligation, not a step-4 local
  flowpipe proof, not a five-step flowpipe chain, and not a global Lorenz
  theorem.

Current v84 addition:

- Added `lorenz_i256_step4_taylor2_point_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step4_taylor2_point_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This promotes the v83 step-4 seed into a
  point-derived time-slab containment receipt, mirroring the v69 step-5 layer
  while staying below response/remainder/local-proof claims.
- The checker derives integer derivative bounds from the fixed step-4 point at
  target scale `1_000_000`: `dx_bound=7045651`, `dy_bound=27293136`,
  `dz_bound=644980`. It then checks `ceil(bound / 100)` sweeps
  `(70457,272932,6450)` for `dt=1/100` against slab margins
  `(71000,273500,6500)` with slacks `(543,568,50)`.
- The v84 masks keep the time-slab scope explicit:
  `seed_anchor_mask=3`, `derivative_bound_mask=7`,
  `sweep_ceiling_mask=7`, `slab_margin_mask=7`,
  `time_slab_containment_mask=7`, `local_time_slab_seed_mask=15`,
  `local_time_slab_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `time_slab_status=23`.
- The v84 system instance fingerprint is `338504427`, certificate fingerprint
  is `904234352`, point-time-slab artifact fingerprint is `827089582`, and
  point-time-slab audit is `688797663`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `87` and
  portfolio v84. The v84 manifest has 88 entries; result counters are
  `(unsat=10, validated=76, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=71, meta=2)`, manifest fingerprint
  `157689400`, acceptance receipt `410135939`, audit receipt `296264027`, and
  readiness fingerprint `544039824`.
- Scope boundary: v84 is a point-derived time-slab containment receipt for
  step 4. It proves integer bound/ceiling/slack consistency for the certified
  point derivative, not a ball enclosure, not a Taylor remainder enclosure,
  not a response or local-containment obligation, not a step-4 local flowpipe
  proof, and not a global Lorenz/shadowing theorem.

Current v85 addition:

- Added `lorenz_i256_step4_taylor2_response_envelope_check()` plus
  `tests/run-pass/lorenz_i256_step4_taylor2_response_envelope_tiny.sio`
  and imported API/portfolio gates. This mirrors the v70 response-envelope
  layer for step 5, but anchors it to the step-4 v84 point-time-slab receipt
  and v83 local-flowpipe seed instead of jumping directly to a completed
  enclosure or local proof.
- The checker anchors v84 point-time-slab artifact/audit
  `827089582`/`688797663` and v83 local-flowpipe seed artifact/audit
  `563331437`/`684718822`. It fixes `step_index=4`, `taylor_order=2`,
  `dt=1/100`, `scale_log2=32`, `source_width_bits=256`, candidate kind
  `ball enclosure`, and the four required future artifacts: center, radius,
  remainder, and proof trace.
- The v85 masks keep the response scope explicit:
  `anchor_mask=15`, `contract_mask=15`,
  `candidate_requirement_mask=15`, `response_envelope_mask=127`,
  `local_response_seed_mask=15`, `local_response_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, `response_status=24`, and
  `ok_mask=255`.
- The v85 system instance fingerprint is `295689332`, certificate fingerprint
  is `596813742`, response-envelope artifact fingerprint is `575348777`, and
  response-envelope audit is `417943295`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `88` and
  portfolio v85. The v85 manifest has 89 entries; result counters are
  `(unsat=10, validated=77, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=72, meta=2)`, manifest fingerprint
  `767589932`, acceptance receipt `162558664`, audit receipt `69746315`, and
  readiness fingerprint `661481028`.
- Scope boundary: v85 is a candidate-required response envelope. It does not
  provide the step-4 center, radius, Taylor remainder, proof trace, local
  containment obligation, local flowpipe proof, multi-step Lorenz proof, or
  global shadowing theorem.

Current v86 addition:

- Added `lorenz_i256_step4_taylor2_center_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step4_taylor2_center_artifact_tiny.sio`
  and imported API/portfolio gates. This fills the first concrete artifact
  slot promised by the v85 response envelope: the step-4 center candidate.
- The checker replays the step-4 i256 quotient/remainder witnesses from the
  v83 seed, starting at `(4617762706,7643846696,4120687759)` and ending at
  `(4920371098,8816077911,4092986080)`. It checks
  `dy_scaled_q/r=124866970867/1906281842`,
  `xy_scaled_q/r=8218332706/3252936400`,
  `beta_q/r=10988500690/2`, increments
  `(302608392,1172231215,27701679)`, and increment remainders
  `(1014364768,2184377272,948411264)`.
- The v86 anchors are the v85 response envelope `575348777`/`417943295`,
  v84 point-time-slab `827089582`/`688797663`, and v83 local-flowpipe seed
  `563331437`/`684718822`.
- The v86 masks keep the candidate boundary explicit:
  `point_replay_mask=127`, `anchor_mask=63`, `final_center_mask=7`,
  `center_candidate_mask=15`, `local_center_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and `ok_mask=255`.
- The v86 system instance fingerprint is `625447576`, certificate fingerprint
  is `54748178`, center artifact fingerprint is `450696904`, and center
  artifact audit is `780071782`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `89` and
  portfolio v86. The v86 manifest has 90 entries; result counters are
  `(unsat=10, validated=78, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=73, meta=2)`, manifest fingerprint
  `620969911`, acceptance receipt `643405834`, audit receipt `824303685`,
  and readiness fingerprint `580966337`.
- Scope boundary: v86 is a center candidate artifact under the v85 response
  envelope. It is not a radius artifact, Taylor remainder enclosure, proof
  trace, local-containment obligation, local flowpipe proof, multi-step Lorenz
  proof, or global shadowing theorem.

Current v87 addition:

- Added `lorenz_i256_step4_taylor2_radius_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step4_taylor2_radius_artifact_tiny.sio`
  and imported API/portfolio gates. This fills the radius-artifact slot under
  the v85 response envelope and composes directly after the v86 center
  candidate artifact.
- The checker converts the v84 target-scale slab margins
  `(71000,273500,6500)` into source-scale radii
  `(304942679,1174673556,27917288)` at `2^32`, then derives conservative
  derivative-radius witnesses `dx_rad=14796162350`,
  `prod_x_rhoz=8277823766`, `dy_rad=9452497322`,
  `prod_xy=1889073041`, `beta_rad=74446103`, and
  `dz_rad=1963519144`.
- The v87 next-radius witnesses are
  `(452904300,1269198528,47552480)` under caps
  `(500000000,1500000000,100000000)`. These caps are candidate bounds for
  this radius artifact, not a completed containment theorem.
- The v87 anchors are the v86 center artifact `450696904`/`780071782`, the
  v85 response envelope `575348777`/`417943295`, and the v84 point-time-slab
  `827089582`/`688797663`.
- The v87 masks keep the candidate boundary explicit:
  conversion, `dx`, `x*(rho-z)`, `dy`, `z`, next-radius, and cap checks
  compose to `ok_mask=127`, with `local_radius_proof_mask=0` and
  `global_flowpipe_claim_mask=0`.
- The v87 system instance fingerprint is `696087643`, certificate fingerprint
  is `662097664`, radius artifact fingerprint is `749342117`, and radius
  artifact audit is `185341710`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `90` and
  portfolio v87. The v87 manifest has 91 entries; result counters are
  `(unsat=10, validated=79, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=74, meta=2)`, manifest fingerprint
  `370584026`, acceptance receipt `901275932`, audit receipt `860684087`,
  and readiness fingerprint `322940586`.
- Scope boundary: v87 is a radius candidate artifact under the v85/v86
  candidate chain. It is not a Taylor remainder enclosure, proof trace,
  local-containment obligation, local flowpipe proof, multi-step Lorenz proof,
  or global shadowing theorem.

### 2026-06-23 step-4 Taylor-2 remainder obligation v88

- Added a step-4 Lorenz i256 Taylor-2 remainder-obligation checker that anchors
  the v85 response envelope, v86 center artifact, and v87 radius artifact while
  preserving `local_remainder_proof_mask=0` and `global_flowpipe_claim_mask=0`.
- The v88 system instance fingerprint is `408418036`, certificate fingerprint
  is `166672715`, remainder-obligation artifact fingerprint is `700260810`, and
  artifact audit is `747893368`.
- The source-radius box remains the step-4 source radius
  `(304942679, 1174673556, 27917288)`, not the expanded v87 next-radius box.
  The checked Taylor-2 LTE ppm bounds are `(32524, 45203, 5836)` under caps
  `(40000, 55000, 10000)`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `91` and
  portfolio v88. The v88 chained manifest has 92 entries; result counters are
  `(unsat=10, validated=80, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=75, meta=2)`, manifest fingerprint
  `2529965`, acceptance receipt `229944247`, audit receipt `782243093`, and
  readiness fingerprint `328442325`.
- Scope boundary: v88 is a checked Taylor-2 remainder-obligation receipt. It is
  not a local-containment proof, proof trace, completed candidate bundle,
  multi-step Lorenz proof, or global shadowing theorem.

### 2026-06-23 step-4 Taylor-2 completed candidate bundle v89

- Added a step-4 Lorenz i256 Taylor-2 completed-candidate bundle checker that
  composes the v85 response envelope, v86 center artifact, v87 radius artifact,
  and v88 remainder obligation into a single candidate-complete receipt.
- The v89 system instance fingerprint is `387318912`, certificate fingerprint
  is `851913299`, completed-bundle artifact fingerprint is `389227567`, and
  artifact audit is `234665993`.
- The checked mask shape is intentionally conservative: required/provided
  artifact mask `15`, missing artifact mask `0`, candidate status `4`,
  dependency-ready mask `15`, and `validated_enclosure_mask=0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `92` and
  portfolio v89. The v89 chained manifest has 93 entries; result counters are
  `(unsat=10, validated=81, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=76, meta=2)`, manifest fingerprint
  `109460759`, acceptance receipt `905773864`, audit receipt `482417275`, and
  readiness fingerprint `909105361`.
- Scope boundary: v89 is a completed-candidate bundle receipt only. It is not a
  proof trace, proof-trace replay engine, local-containment proof, local
  flowpipe proof, multi-step Lorenz proof, invariant/shadowing proof, or global
  Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 proof-trace skeleton v90

- Added a step-4 Lorenz i256 Taylor-2 proof-trace skeleton checker that anchors
  the v89 completed-candidate bundle and the four candidate artifacts
  `(response, center, radius, remainder)` without replaying the trace yet.
- The v90 system instance fingerprint is `607086535`, certificate fingerprint
  is `46656432`, proof-trace skeleton artifact fingerprint is `556038402`, and
  artifact audit is `509176612`.
- The checked skeleton shape is trace version `1`, trace kind step4 `4`,
  node/edge counts `4/4`, obligation/replay/dependency-root masks `15/15/15`,
  trace status `5`, and `validated_enclosure_mask=0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `93` and
  portfolio v90. The v90 chained manifest has 94 entries; result counters are
  `(unsat=10, validated=82, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=77, meta=2)`, manifest fingerprint
  `785386925`, acceptance receipt `647953398`, audit receipt `448312604`, and
  readiness fingerprint `888891354`.
- Scope boundary: v90 is a proof-trace skeleton receipt only. It is not a
  proof-trace replay engine, local-containment proof, local flowpipe proof,
  multi-step Lorenz proof, invariant/shadowing proof, or global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 replay preflight v91

- Added a step-4 Lorenz i256 Taylor-2 replay-preflight checker that consumes
  the v90 proof-trace skeleton and verifies the replay/readiness masks required
  before a replay executor is allowed to exist.
- The v91 system instance fingerprint is `335114620`, certificate fingerprint
  is `922120978`, replay-preflight artifact fingerprint is `525394039`, and
  artifact audit is `949135313`.
- The checked preflight shape is trace version `1`, replay version `1`,
  node/edge counts `4/4`, obligation/replay/replayed/predecessor-ready masks
  `15/15/15/15`, replay status `6`, and
  `validated_enclosure_mask=0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `94` and
  portfolio v91. The v91 chained manifest has 95 entries; result counters are
  `(unsat=10, validated=83, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=78, meta=2)`, manifest fingerprint
  `271084941`, acceptance receipt `915914304`, audit receipt `520416831`, and
  readiness fingerprint `387466576`.
- Scope boundary: v91 is a replay-preflight receipt only. It is not replay
  execution, validated enclosure, local-containment proof, local flowpipe
  proof, multi-step Lorenz proof, invariant/shadowing proof, or global Lorenz
  theorem.

### 2026-06-23 step-4 Taylor-2 replay executor v92

- Added a step-4 Lorenz i256 Taylor-2 replay-executor receipt that consumes the
  v91 replay preflight, the v90 proof-trace skeleton, and the v89 completed
  candidate bundle. It checks replayed node/edge masks but still does not
  promote the enclosure to validated.
- The v92 system instance fingerprint is `434726493`, certificate fingerprint
  is `258671429`, replay-executor artifact fingerprint is `563122161`, and
  artifact audit is `308331037`.
- The checked executor shape is trace version `1`, replay engine version `1`,
  node/edge counts `4/4`, node/edge/replayed/predecessor masks
  `15/15/15/15`, replay status `7`, and
  `validated_enclosure_mask=0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `95` and
  portfolio v92. The v92 chained manifest has 96 entries; result counters are
  `(unsat=10, validated=84, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=79, meta=2)`, manifest fingerprint
  `164137752`, acceptance receipt `91721995`, audit receipt `507235414`, and
  readiness fingerprint `769570755`.
- Scope boundary: v92 is replay execution of the proof-trace receipt only. It
  is not validated enclosure, local-containment proof, local flowpipe proof,
  multi-step Lorenz proof, invariant/shadowing proof, or global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 enclosure validator guard v93

- Added a step-4 Lorenz i256 Taylor-2 enclosure-validator guard that consumes
  the v92 replay executor, the v89 completed candidate bundle, the v87 radius
  artifact, and the v88 remainder obligation. It checks a scalar radius-plus-
  remainder budget against explicit guard margins but does not validate the
  enclosure yet.
- The v93 system instance fingerprint is `265395467`, certificate fingerprint
  is `669027086`, enclosure-guard artifact fingerprint is `133410511`, and
  artifact audit is `282038720`.
- The checked scalar budget is target scale `1000000`, radius ppm
  `(71000, 273500, 6500)`, LTE ppm `(32524, 45203, 5836)`, computed needs
  `(103524, 318703, 12336)`, and guard margins `(104000, 319000, 12500)`.
  The resulting slack is `(476, 297, 164)` ppm with inclusion masks
  `pass=7`, `fail=0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `96` and
  portfolio v93. The v93 chained manifest has 97 entries; result counters are
  `(unsat=10, validated=85, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=80, meta=2)`, manifest fingerprint
  `571640663`, acceptance receipt `493872957`, audit receipt `501269187`, and
  readiness fingerprint `866875356`.
- Scope boundary: v93 is a guarded scalar enclosure-budget receipt only. It is
  not a validated enclosure, local-containment proof, local flowpipe proof,
  multi-step Lorenz proof, invariant/shadowing proof, or global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 local containment obligation v94

- Added a step-4 Lorenz i256 Taylor-2 local-containment obligation that
  consumes the v93 enclosure-validator guard, the v92 replay executor, the v89
  completed candidate bundle, the v87 radius artifact, and the v88 remainder
  obligation. It turns the already checked scalar budget into an axiswise local
  flowpipe-obligation mask without claiming a local proof yet.
- The v94 system instance fingerprint is `273233357`, certificate fingerprint
  is `255024353`, local-containment-obligation artifact fingerprint is
  `37368367`, and artifact audit is `94030837`.
- The checked obligation uses Taylor order `2`, selected step `1/100`, target
  scale `1000000`, radii `(71000, 273500, 6500)`, LTE bounds
  `(32524, 45203, 5836)`, needs `(103524, 318703, 12336)`, margins
  `(104000, 319000, 12500)`, and positive slack `(476, 297, 164)`.
  The axis containment mask and local-flowpipe-obligation mask are both `7`;
  `global_flowpipe_claim_mask` remains `0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `97` and
  portfolio v94. The v94 chained manifest has 98 entries; result counters are
  `(unsat=10, validated=86, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=81, meta=2)`, manifest fingerprint
  `212279500`, acceptance receipt `960863117`, audit receipt `526242618`, and
  readiness fingerprint `785400259`.
- Scope boundary: v94 is a local-containment obligation receipt only. It is not
  a local containment proof, local flowpipe proof, multi-step Lorenz proof,
  invariant/shadowing proof, or global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 time-slab containment v95

- Added a step-4 Lorenz i256 Taylor-2 time-slab-containment receipt that
  composes the v94 local-containment obligation with the v84 point-derived
  time-slab receipt and the v88 remainder obligation. This records the local
  time slab needed before a later flowpipe-obligation receipt.
- The v95 system instance fingerprint is `519576522`, certificate fingerprint
  is `85884916`, time-slab-containment artifact fingerprint is `110014281`,
  and artifact audit is `651883944`.
- The checked time slab uses selected step `1/100`, derivative bounds
  `(7045651, 27293136, 644980)`, sweep ceilings `(70457, 272932, 6450)`,
  slab margins `(71000, 273500, 6500)`, and positive slack `(543, 568, 50)`.
  The time-slab-containment mask and local-time-slab-obligation mask are both
  `7`; `global_flowpipe_claim_mask` remains `0`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `98` and
  portfolio v95. The v95 chained manifest has 99 entries; result counters are
  `(unsat=10, validated=87, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=82, meta=2)`, manifest fingerprint
  `138855577`, acceptance receipt `750840648`, audit receipt `12175216`, and
  readiness fingerprint `506674456`.
- Scope boundary: v95 is a local time-slab containment receipt only. It is not
  a flowpipe obligation, local flowpipe proof, multi-step Lorenz proof,
  invariant/shadowing proof, or global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 flowpipe obligation v96

- Added a step-4 Lorenz i256 Taylor-2 flowpipe-obligation receipt that composes
  the v95 time-slab containment, v94 endpoint/local-containment obligation,
  v85 response envelope, v88 remainder obligation, and v83 local-flowpipe seed
  into one explicit local obligation boundary.
- The v96 system instance fingerprint is `887487060`, certificate fingerprint
  is `311967295`, flowpipe-obligation artifact fingerprint is `555736781`, and
  artifact audit is `942295194`.
- The checked composition records endpoint containment mask `7`, time-slab
  containment mask `7`, response-envelope mask `127`, remainder-obligation mask
  `127`, and composition-ready/local-flowpipe-obligation mask `15`. It
  deliberately preserves `local_flowpipe_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and flowpipe-obligation status `20`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `99` and
  portfolio v96. The v96 chained manifest has 100 entries; result counters are
  `(unsat=10, validated=88, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=83, meta=2)`, manifest fingerprint
  `793336125`, acceptance receipt `66069664`, audit receipt `832768462`, and
  readiness fingerprint `252299410`.
- Scope boundary: v96 is a local flowpipe-obligation receipt only. It is not a
  local flowpipe proof, multi-step Lorenz proof, invariant/shadowing proof, or
  global Lorenz theorem.

### 2026-06-23 step-4 Taylor-2 local flowpipe proof v97

- Added a step-4 Lorenz i256 Taylor-2 local-flowpipe-proof receipt that
  consumes the v96 flowpipe obligation, v95 time-slab containment, v94 local
  containment obligation, v85 response envelope, v88 remainder obligation, and
  v83 local-flowpipe seed. This is the step-4 analogue of the v82 step-5 local
  proof receipt.
- The v97 system instance fingerprint is `142700099`, certificate fingerprint
  is `72413218`, local-flowpipe-proof artifact fingerprint is `354200144`, and
  artifact audit is `97568991`.
- The checked masks are explicit:
  `endpoint_proof_mask=7`, `time_slab_proof_mask=7`,
  `response_proof_mask=127`, `remainder_proof_mask=127`,
  `composition_ready_mask=15`, `local_flowpipe_obligation_mask=15`,
  `local_flowpipe_proof_mask=15`, `global_flowpipe_claim_mask=0`, and
  `local_flowpipe_status=21`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `100` and
  portfolio v97. The v97 chained manifest has 101 entries; result counters are
  `(unsat=10, validated=89, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=84, meta=2)`, manifest fingerprint
  `663928611`, acceptance receipt `446588679`, audit receipt `22871079`, and
  readiness fingerprint `2491648`.
- Scope boundary: v97 is a local single-step Taylor-2 flowpipe proof receipt
  for step 4 under recorded prerequisite certificates. It is stronger than v96
  because `local_flowpipe_proof_mask` is now `15`, but it is not a multi-step
  Lorenz proof, not an invariant/shadowing theorem, and not a global Lorenz
  claim.

### 2026-06-23 step-3 Taylor-2 local flowpipe seed v98

- Added `lorenz_i256_step3_taylor2_local_flowpipe_seed_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_local_flowpipe_seed_tiny.sio`
  and imported API/portfolio gates. This starts the step-3 backfill after the
  completed step-4 local proof receipt instead of jumping directly to
  containment or a proof claim.
- The checker replays the step-3 i256 quotient/remainder witnesses for
  `(4406636440,6517899153,4164877513)` to
  `(4617762706,7643846696,4120687759)`, anchors the three-step trajectory
  manifest `449592233`, the three-step chain anchor `249889958`, step-3
  certificate `603078026`, four-step chain anchor `737039167`, and step-4
  certificate `753371133`.
- The v98 masks keep the non-claim boundary machine-visible:
  `point_replay_mask=127`, `endpoint_mask=7`, `trajectory_anchor_mask=7`,
  `suffix_anchor_mask=3`, `local_flowpipe_seed_mask=15`,
  `local_flowpipe_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `seed_status=22`.
- The v98 system instance fingerprint is `319928500`, certificate fingerprint
  is `198035376`, step-3 local-flowpipe seed artifact fingerprint is
  `389654052`, and artifact audit is `790630993`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `101` and
  portfolio v98. The v98 chained manifest has 102 entries; result counters are
  `(unsat=10, validated=90, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=85, meta=2)`, manifest fingerprint
  `142673235`, acceptance receipt `346679172`, audit receipt `122829650`, and
  readiness fingerprint `96776847`.
- Scope boundary: v98 proves that the step-3 point update can be replayed as a
  typed i256 seed and that its prefix/suffix anchors line up with existing
  trajectory receipts. It is not step-3 time-slab containment, not a response/
  remainder/local-containment obligation, not a step-3 local flowpipe proof,
  not a multi-step flowpipe proof, and not a global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 point time-slab containment v99

- Added `lorenz_i256_step3_taylor2_point_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_point_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This consumes the v98 step-3 local-flowpipe
  seed receipt instead of reusing the step-4 containment constants.
- The step-3 point-time-slab checker derives conservative one-step derivative
  ceilings from the v98 start point `(4406636440,6517899153,4164877513)`:
  derivative bounds `(4915667,26215510,1028873)`, dt sweeps
  `(49157,262156,10289)`, slab margins `(49500,262500,10500)`, and positive
  slacks `(343,344,211)`.
- The v99 masks keep the containment level machine-visible:
  `seed_anchor_mask=3`, `derivative_bound_mask=7`, `sweep_ceiling_mask=7`,
  `slab_margin_mask=7`, `time_slab_containment_mask=7`,
  `local_time_slab_seed_mask=15`, `local_time_slab_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and `time_slab_status=23`.
- The v99 system instance fingerprint is `831120557`, certificate fingerprint
  is `546042778`, point-time-slab artifact fingerprint is `417748218`, and
  artifact audit is `109213065`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `102` and
  portfolio v99. The v99 chained manifest has 103 entries; result counters are
  `(unsat=10, validated=91, optimal=1, sat=1)`, checker-family counters are
  `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=86, meta=2)`, manifest fingerprint
  `650273736`, acceptance receipt `65010134`, audit receipt `423857093`, and
  readiness fingerprint `295692487`.
- Scope boundary: v99 proves a local point-time-slab seed for the step-3
  backfill under the v98 seed artifact. It is not a response-envelope,
  remainder, local-containment obligation, local flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 response envelope v100

- Added `lorenz_i256_step3_taylor2_response_envelope_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_response_envelope_tiny.sio`
  and imported API/portfolio gates. This consumes the v99 point-time-slab
  receipt and the v98 local-flowpipe seed receipt for the step-3 backfill.
- The checker records a candidate-required response envelope for Taylor order
  2, dt `1/100`, i256 at `scale_log2=32`, and required ball-enclosure
  artifacts `(center=1, radius=1, remainder=1, proof_trace=1)`.
- The v100 masks keep the non-proof boundary explicit:
  `anchor_mask=15`, `contract_mask=15`, `candidate_requirement_mask=15`,
  `response_envelope_mask=127`, `local_response_seed_mask=15`,
  `local_response_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `response_status=24`.
- The v100 system instance fingerprint is `834642244`, certificate fingerprint
  is `140313818`, response-envelope artifact fingerprint is `116883728`, and
  artifact audit is `84354264`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `103` and
  portfolio v100. The v100 chained manifest has 104 entries; result counters
  are `(unsat=10, validated=92, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=87, meta=2)`, manifest
  fingerprint `763506498`, acceptance receipt `764455787`, audit receipt
  `788860016`, and readiness fingerprint `779852235`.
- Scope boundary: v100 proves only that the step-3 response envelope and
  candidate artifact requirements are replayable under the v99/v98 anchors. It
  is not a center artifact, radius artifact, remainder obligation, proof trace,
  local-containment obligation, local flowpipe proof, multi-step flowpipe
  proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 center artifact v101

- Added `lorenz_i256_step3_taylor2_center_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_center_artifact_tiny.sio` and
  imported API/portfolio gates. This consumes the v100 response-envelope
  receipt, the v99 point-time-slab receipt, and the v98 local-flowpipe seed.
- The checker replays the exact i256 quotient/remainder witnesses for the
  step-3 center update from `(4406636440,6517899153,4164877513)` to
  `(4617762706,7643846696,4120687759)`: `dy_scaled_q=119112655997`,
  `xy_scaled_q=6687364522`, `beta_q=11106340034`, increments
  `(211126266,1125947543,44189754)`, and the recorded nonzero remainders.
- The v101 masks keep the non-proof boundary explicit:
  `point_replay_mask=127`, `anchor_mask=63`, `final_center_mask=7`,
  `center_candidate_mask=15`, `local_center_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v101 system instance fingerprint is `391525506`, certificate fingerprint
  is `653032557`, center artifact fingerprint is `164784481`, and artifact
  audit is `660765149`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `104` and
  portfolio v101. The v101 chained manifest has 105 entries; result counters
  are `(unsat=10, validated=93, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=88, meta=2)`, manifest
  fingerprint `109642402`, acceptance receipt `967273893`, audit receipt
  `699241021`, and readiness fingerprint `533529134`.
- Scope boundary: v101 proves only a replayable step-3 center candidate
  artifact under the v100/v99/v98 anchors. It is not a radius artifact,
  remainder obligation, proof trace, local-containment obligation, local
  flowpipe proof, multi-step flowpipe proof, invariant/shadowing theorem, or
  global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 radius artifact v102

- Added `lorenz_i256_step3_taylor2_radius_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_radius_artifact_tiny.sio` and
  imported API/portfolio gates. This consumes the v101 center-artifact receipt,
  the v100 response-envelope receipt, and the v99 point-time-slab receipt.
- The checker converts the step-3 slab margins `(49500,262500,10500)` at
  target scale `1000000` into source-scale radii
  `(212600882,1127428916,45097157)`, then replays interval-radius derivative
  bounds around center `(4617762706,7643846696,4120687759)`.
- The recorded radius witnesses are `dx_rad=13400297980`,
  `prod_x_rhoz=5799569470`, `dy_rad=6926998386`, `prod_xy=1646340856`,
  `beta_rad=120259087`, `dz_rad=1766599943`, and next source-scale radii
  `(346603859,1196698899,62763157)` under caps
  `(500000000,1500000000,100000000)`.
- The v102 masks keep the non-proof boundary explicit: `ok_mask=127`,
  `local_radius_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
- The v102 system instance fingerprint is `118650140`, certificate fingerprint
  is `413660742`, radius artifact fingerprint is `844877047`, and artifact
  audit is `492955058`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `105` and
  portfolio v102. The v102 chained manifest has 106 entries; result counters
  are `(unsat=10, validated=94, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=89, meta=2)`, manifest
  fingerprint `323635732`, acceptance receipt `175774038`, audit receipt
  `360019655`, and readiness fingerprint `584783151`.
- Scope boundary: v102 proves only a replayable step-3 radius candidate
  artifact under the v101/v100/v99 anchors. It is not a remainder obligation,
  proof trace, local-containment obligation, local flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 remainder obligation v103

- Added `lorenz_i256_step3_taylor2_remainder_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_remainder_obligation_tiny.sio` and
  imported API/portfolio gates. This consumes the v102 radius artifact, the
  v101 center artifact, and the v100 response envelope.
- The checker builds the absolute box around center
  `(4617762706,7643846696,4120687759)` using source-scale radii
  `(212600882,1127428916,45097157)`, giving bounds
  `(4830363588,8771275612,4165784916)` and
  `rhoz_abs_bound=116183493686`.
- The first- and second-derivative witnesses are
  `x_prime=136016392000`, `y_prime=139437815949`,
  `z_prime=20973433339`, `x_second=2754542079490`,
  `y_second=3842415704665`, and `z_second=490524458002`.
- The local truncation obligation records source-scale LTE ceilings
  `(137727098,192120777,24526222)`, target-scale ppm ceilings
  `(32068,44732,5711)`, caps `(40000,55000,10000)`, and positive slacks
  `(7932,10268,4289)`.
- The v103 masks keep the non-proof boundary explicit: `ok_mask=127`,
  `local_remainder_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
- The v103 system instance fingerprint is `542114330`, certificate fingerprint
  is `570828076`, remainder-obligation fingerprint is `693563916`, and
  artifact audit is `465507112`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `106` and
  portfolio v103. The v103 chained manifest has 107 entries; result counters
  are `(unsat=10, validated=95, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=90, meta=2)`, manifest
  fingerprint `502178581`, acceptance receipt `689402395`, audit receipt
  `143043359`, and readiness fingerprint `953013701`.
- Scope boundary: v103 proves only a replayable step-3 Taylor-2 remainder
  obligation under the v102/v101/v100 anchors. It is not a proof trace,
  local-containment proof, local flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 completed candidate bundle v104

- Added `lorenz_i256_step3_taylor2_completed_candidate_bundle_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_completed_candidate_bundle_tiny.sio`
  and imported API/portfolio gates. This consumes the v103 remainder
  obligation, v102 radius artifact, v101 center artifact, and v100 response
  envelope as a complete candidate-bundle dependency set.
- The checker records `required_artifact_mask=15`, `provided_artifact_mask=15`,
  `missing_artifact_mask=0`, `candidate_status_complete=4`, and
  `ok_mask=255`, while keeping `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
- The v104 system instance fingerprint is `220878249`, certificate fingerprint
  is `449726126`, completed-candidate-bundle fingerprint is `693283321`, and
  artifact audit is `160292433`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `107` and
  portfolio v104. The v104 chained manifest has 108 entries; result counters
  are `(unsat=10, validated=96, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=91, meta=2)`, manifest
  fingerprint `824838852`, acceptance receipt `527800756`, audit receipt
  `985220902`, and readiness fingerprint `829136864`.
- Scope boundary: v104 proves only that the replayable step-3 Taylor-2
  candidate bundle has all four required artifact dependencies present under
  the v103/v102/v101/v100 anchors. It is not a proof trace, replay executor,
  local-containment proof, local flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 proof-trace skeleton v105

- Added `lorenz_i256_step3_taylor2_proof_trace_skeleton_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_proof_trace_skeleton_tiny.sio` and
  imported API/portfolio gates. This consumes the v104 completed candidate
  bundle plus the four step-3 Taylor-2 dependency artifacts.
- The checker records a skeleton trace with `trace_version=1`,
  `trace_kind_step3=3`, `trace_node_count=4`, `dependency_edge_count=4`,
  `obligation_mask=15`, `replay_order_mask=15`,
  `dependency_root_mask=15`, `trace_status_skeleton=5`, and `ok_mask=255`.
- The v105 masks keep replay and validation separate:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v105 system instance fingerprint is `50354795`, certificate fingerprint
  is `17187575`, proof-trace-skeleton fingerprint is `308150621`, and artifact
  audit is `281860333`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `108` and
  portfolio v105. The v105 chained manifest has 109 entries; result counters
  are `(unsat=10, validated=97, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=92, meta=2)`, manifest
  fingerprint `543887870`, acceptance receipt `206531377`, audit receipt
  `238950575`, and readiness fingerprint `139143245`.
- Scope boundary: v105 proves only that a replayable step-3 Taylor-2 trace
  skeleton names the four candidate obligations and their replay/root masks
  under the v104/v103/v102/v101/v100 anchors. It is not a replay executor,
  validated enclosure, local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-3 Taylor-2 replay preflight v106

- Added `lorenz_i256_step3_taylor2_replay_preflight_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_replay_preflight_tiny.sio` and
  imported API/portfolio gates. This consumes the v105 proof-trace skeleton
  and the v104 completed candidate bundle.
- The checker records a pre-replay gate with `trace_version=1`,
  `replay_version=1`, `trace_node_count=4`, `dependency_edge_count=4`,
  `obligation_mask=15`, `replay_order_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, `replay_status_preflight=6`, and
  `ok_mask=255`.
- The v106 masks keep replay execution and validation separate:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v106 system instance fingerprint is `450583915`, certificate fingerprint
  is `995926925`, replay-preflight fingerprint is `488274627`, and artifact
  audit is `678589210`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `109` and
  portfolio v106. The v106 chained manifest has 110 entries; result counters
  are `(unsat=10, validated=98, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=93, meta=2)`, manifest
  fingerprint `377115992`, acceptance receipt `490704907`, audit receipt
  `12824662`, and readiness fingerprint `711276818`.
- Scope boundary: v106 proves only that the step-3 Taylor-2 proof-trace
  skeleton is ready for replay under consistent node/order/predecessor masks.
  It is not a replay executor, validated enclosure, local-containment proof,
  local flowpipe proof, multi-step flowpipe proof, invariant/shadowing theorem,
  or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 replay executor v107

- Added `lorenz_i256_step3_taylor2_replay_executor_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_replay_executor_tiny.sio` and
  imported API/portfolio gates. This consumes the v106 replay preflight, the
  v105 proof-trace skeleton, and the v104 completed candidate bundle.
- The checker records executed replay receipts for the four trace nodes:
  `trace_version=1`, `replay_engine_version=1`, `trace_node_count=4`,
  `dependency_edge_count=4`, `node_receipt_mask=15`, `edge_receipt_mask=15`,
  `replayed_node_mask=15`, `predecessor_ready_mask=15`,
  `replay_status_executed=7`, and `ok_mask=255`.
- The v107 masks continue to keep replay execution and validation separate:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v107 system instance fingerprint is `37836649`, certificate fingerprint
  is `378550558`, replay-executor fingerprint is `803328296`, and artifact
  audit is `630199225`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `110` and
  portfolio v107. The v107 chained manifest has 111 entries; result counters
  are `(unsat=10, validated=99, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=94, meta=2)`, manifest
  fingerprint `947882987`, acceptance receipt `634681668`, audit receipt
  `109328435`, and readiness fingerprint `517946998`.
- Scope boundary: v107 proves only that the step-3 Taylor-2 proof-trace has
  been replayed with complete node and dependency receipts. It is not a
  validated enclosure, local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-3 Taylor-2 enclosure validator guard v108

- Added `lorenz_i256_step3_taylor2_enclosure_validator_guard_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_enclosure_validator_guard_tiny.sio`
  and imported API/portfolio gates. This consumes the v107 replay executor,
  v104 completed bundle, v102 radius artifact, and v103 remainder obligation.
- The guard converts the v102 source-scale radii
  `(346603859,1196698899,62763157)` into target-scale ppm ceilings
  `(80700,278629,14614)`, combines them with the v103 LTE ppm ceilings
  `(32068,44732,5711)`, and records required enclosure budget
  `(112768,323361,20325)`.
- The guard margins are `(113000,323500,20500)`, giving positive slacks
  `(232,139,175)`, `inclusion_pass_mask=7`, `inclusion_fail_mask=0`,
  `validator_status_guarded=8`, and `ok_mask=255`.
- The v108 masks keep guard acceptance and validation separate:
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v108 system instance fingerprint is `166627559`, certificate fingerprint
  is `623656248`, enclosure-guard fingerprint is `940966535`, and artifact
  audit is `356178921`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `111` and
  portfolio v108. The v108 chained manifest has 112 entries; result counters
  are `(unsat=10, validated=100, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=95, meta=2)`, manifest
  fingerprint `324233200`, acceptance receipt `28915879`, audit receipt
  `620796259`, and readiness fingerprint `437221035`.
- Scope boundary: v108 proves only that the step-3 Taylor-2 replayed candidate
  has a guarded scalar enclosure budget with positive slack. It is not a
  validated enclosure, local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-3 Taylor-2 local containment obligation v109

- Added `lorenz_i256_step3_taylor2_local_containment_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_local_containment_obligation_tiny.sio`
  and imported API/portfolio gates. This consumes the v108 enclosure guard,
  v107 replay executor, v104 completed bundle, v102 radius artifact, and v103
  remainder obligation.
- The obligation keeps the v108 scalar budget fixed: target-scale radii
  `(80700,278629,14614)` plus LTE ceilings `(32068,44732,5711)` produce needs
  `(112768,323361,20325)` under margins `(113000,323500,20500)`.
- The positive containment slacks are `(232,139,175)`, with
  `axis_containment_mask=7`, `local_flowpipe_obligation_mask=7`,
  `obligation_status_local=12`, and `ok_mask=255`.
- The v109 masks keep local obligation and global proof separate:
  `global_flowpipe_claim_mask=0`. This is an obligation receipt, not a
  multi-step/global theorem.
- The v109 system instance fingerprint is `950834721`, certificate fingerprint
  is `281922457`, local-containment-obligation fingerprint is `360827572`,
  and artifact audit is `456885748`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `112` and
  portfolio v109. The v109 chained manifest has 113 entries; result counters
  are `(unsat=10, validated=101, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=96, meta=2)`, manifest
  fingerprint `427093525`, acceptance receipt `923119767`, audit receipt
  `880977452`, and readiness fingerprint `780008796`.
- Scope boundary: v109 proves only that the replayed step-3 Taylor-2 candidate
  has a local containment obligation with positive per-axis slack under the
  v108/v107/v104/v103/v102 anchors. It is not time-slab containment, local
  flowpipe proof, multi-step flowpipe proof, invariant/shadowing theorem, or
  global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 time-slab containment v110

- Added `lorenz_i256_step3_taylor2_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This consumes the v109 local containment
  obligation, v99 point-time-slab containment, and v103 remainder obligation.
- The checker reuses the step-3 point-time-slab derivative bounds
  `(4915667,26215510,1028873)` with dt `1/100`, producing sweep ceilings
  `(49157,262156,10289)`.
- The slab margins are `(49500,262500,10500)`, giving positive slacks
  `(343,344,211)`, `time_slab_containment_mask=7`,
  `local_time_slab_obligation_mask=7`, `time_slab_status_local=13`, and
  `ok_mask=255`.
- The v110 masks keep local time-slab containment separate from downstream
  flowpipe proof: `global_flowpipe_claim_mask=0`.
- The v110 system instance fingerprint is `604148523`, certificate
  fingerprint is `460379869`, time-slab-containment fingerprint is
  `113001479`, and artifact audit is `175704874`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `113` and
  portfolio v110. The v110 chained manifest has 114 entries; result counters
  are `(unsat=10, validated=102, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=97, meta=2)`, manifest
  fingerprint `767613887`, acceptance receipt `579383998`, audit receipt
  `339236300`, and readiness fingerprint `843058093`.
- Scope boundary: v110 proves only that the step-3 Taylor-2 point-time-slab
  sweep is contained inside its local slab margins under the v109/v99/v103
  anchors. It is not a flowpipe obligation, local flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 flowpipe obligation v111

- Added `lorenz_i256_step3_taylor2_flowpipe_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_flowpipe_obligation_tiny.sio`
  and imported API/portfolio gates. This composes the v110 time-slab
  containment, v109 local containment obligation, v100 response envelope, v103
  remainder obligation, and v98 local flowpipe seed.
- The checker requires endpoint and time-slab containment masks `7`, response
  and remainder masks `127`, `composition_ready_mask=15`,
  `local_flowpipe_obligation_mask=15`, `local_flowpipe_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, `flowpipe_obligation_status=20`, and
  `ok_mask=255`.
- The v111 system instance fingerprint is `437093534`, certificate
  fingerprint is `810964111`, flowpipe-obligation fingerprint is `743452338`,
  and artifact audit is `255714212`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `114` and
  portfolio v111. The v111 chained manifest has 115 entries; result counters
  are `(unsat=10, validated=103, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=98, meta=2)`, manifest
  fingerprint `133350891`, acceptance receipt `587282084`, audit receipt
  `396590211`, and readiness fingerprint `146637720`.
- Scope boundary: v111 proves only that all step-3 Taylor-2 local ingredients
  needed for a flowpipe proof are present and mutually anchored. It is not the
  local flowpipe proof, multi-step flowpipe proof, invariant/shadowing theorem,
  or global Lorenz theorem.

### 2026-06-23 step-3 Taylor-2 local flowpipe proof v112

- Added `lorenz_i256_step3_taylor2_local_flowpipe_proof_check()` plus
  `tests/run-pass/lorenz_i256_step3_taylor2_local_flowpipe_proof_tiny.sio`
  and imported API/portfolio gates. This consumes the v111 flowpipe
  obligation, v110 time-slab containment, v109 local containment obligation,
  v100 response envelope, v103 remainder obligation, and v98 local flowpipe
  seed.
- The checker turns the local proof mask on only after all four local
  ingredients are present: endpoint and time-slab proof masks `7`, response
  and remainder proof masks `127`, `composition_ready_mask=15`,
  `local_flowpipe_obligation_mask=15`, `local_flowpipe_proof_mask=15`,
  `global_flowpipe_claim_mask=0`, `local_flowpipe_status=21`, and
  `ok_mask=255`.
- The v112 system instance fingerprint is `237311165`, certificate
  fingerprint is `512177108`, local-flowpipe-proof fingerprint is `78860411`,
  and artifact audit is `972719127`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `115` and
  portfolio v112. The v112 chained manifest has 116 entries; result counters
  are `(unsat=10, validated=104, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=99, meta=2)`, manifest
  fingerprint `198078664`, acceptance receipt `18878136`, audit receipt
  `278163265`, and readiness fingerprint `48841302`.
- Scope boundary: v112 proves only the local step-3 Taylor-2 flowpipe proof
  receipt under its fixed anchors. It is not a multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 local flowpipe seed v113

- Added `lorenz_i256_step2_taylor2_local_flowpipe_seed_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_local_flowpipe_seed_tiny.sio`
  and imported API/portfolio gates. This backfills the local step-2 seed
  from the existing two-step chain anchor `335080767` and step-2 certificate
  `899209716`, with the step-3 chain/certificate as the suffix dependency.
- The checker keeps `global_flowpipe_claim_mask=0`, verifies the i256 division
  witnesses for the point replay, endpoint mask `7`, trajectory anchor mask
  `7`, suffix anchor mask `3`, `local_flowpipe_seed_mask=15`,
  `local_flowpipe_proof_mask=0`, `seed_status=22`, and `ok_mask=255`.
- The v113 system instance fingerprint is `551664923`, certificate fingerprint
  is `87899344`, local-flowpipe-seed fingerprint is `633359277`, and artifact
  audit is `421619310`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `116` and
  portfolio v113. The v113 chained manifest has 117 entries; result counters
  are `(unsat=10, validated=105, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=100, meta=2)`, manifest
  fingerprint `766195979`, acceptance receipt `839004489`, audit receipt
  `893230226`, and readiness fingerprint `908862257`.
- Scope boundary: v113 proves only the local step-2 Taylor-2 flowpipe seed
  receipt under fixed chain anchors. It is not the step-2 local flowpipe proof,
  a multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 point time-slab containment v114

- Added `lorenz_i256_step2_taylor2_point_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_point_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This consumes the v113 local flowpipe seed
  fingerprint `633359277` and audit `421619310`.
- The checker pins derivative ceilings at target scale `1_000_000`:
  `dx_bound=2600000`, `dy_bound=25756667`, and `dz_bound=1362223`.
  With `dt=1/100`, the sweep ceilings are `(26000,257567,13623)`.
  The local slab margins `(26100,257700,13700)` leave positive slacks
  `(100,133,77)`.
- The checker records `seed_anchor_mask=3`, `derivative_bound_mask=7`,
  `sweep_ceiling_mask=7`, `slab_margin_mask=7`,
  `time_slab_containment_mask=7`, `local_time_slab_seed_mask=15`,
  `local_time_slab_proof_mask=0`, `global_flowpipe_claim_mask=0`,
  `time_slab_status=23`, and `ok_mask=255`.
- The v114 system instance fingerprint is `910323405`, certificate fingerprint
  is `188845140`, point-time-slab fingerprint is `268229078`, and artifact
  audit is `223161586`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `117` and
  portfolio v114. The v114 chained manifest has 118 entries; result counters
  are `(unsat=10, validated=106, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=101, meta=2)`, manifest
  fingerprint `237552477`, acceptance receipt `569422547`, audit receipt
  `367930703`, and readiness fingerprint `373523570`.
- Scope boundary: v114 proves only the local step-2 point time-slab containment
  receipt under the fixed v113 seed. It is not the step-2 local flowpipe proof,
  a multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 response envelope v115

- Added `lorenz_i256_step2_taylor2_response_envelope_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_response_envelope_tiny.sio`
  and imported API/portfolio gates. This consumes the v114 point time-slab
  fingerprint `268229078`/audit `223161586` and the v113 local-flowpipe seed
  fingerprint `633359277`/audit `421619310`.
- The checker pins step index `2`, Taylor order `2`, `dt=1/100`, source width
  `i256`, scale log2 `32`, point-time-slab containment mask `7`, and
  local-time-slab seed mask `15`.
- The response envelope records a ball-enclosure candidate requirement with
  four required artifacts: center, radius, remainder, and proof trace. It
  records `anchor_mask=15`, `contract_mask=15`,
  `candidate_requirement_mask=15`, `response_envelope_mask=127`,
  `local_response_seed_mask=15`, `local_response_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, `response_status=24`, and `ok_mask=255`.
- The v115 system instance fingerprint is `929244479`, certificate fingerprint
  is `168526441`, response-envelope fingerprint is `482555681`, and artifact
  audit is `591288676`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `118` and
  portfolio v115. The v115 chained manifest has 119 entries; result counters
  are `(unsat=10, validated=107, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=102, meta=2)`, manifest
  fingerprint `242024859`, acceptance receipt `954945780`, audit receipt
  `916566740`, and readiness fingerprint `442225509`.
- Scope boundary: v115 proves only the step-2 response-envelope/candidate
  requirement receipt. It does not provide center/radius/remainder/proof-trace
  artifacts, a local containment proof, a local flowpipe proof, a multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 center artifact v116

- Added `lorenz_i256_step2_taylor2_center_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_center_artifact_tiny.sio` and
  imported API/portfolio gates. This consumes the v115 response-envelope
  fingerprint `482555681`/audit `591288676`, v114 point-time-slab
  fingerprint `268229078`/audit `223161586`, and v113 local-flowpipe seed
  fingerprint `633359277`/audit `421619310`.
- The checker replays the i256 division witnesses for the step-2 Taylor/Euler
  center from `(4294967296,5411658768,4223384510)` to
  `(4406636440,6517899153,4164877513)`. It pins `dt_q=42949672`,
  `dt_r=96`, `dy_scaled=(116035699778,0)`, `xy_scaled=(5411658768,0)`,
  `beta=(11262358693,1)`, `x_inc=(111669144,3023657216)`,
  `y_inc=(1106240385,1604599760)`, and
  `z_drop=(58506997,4047004488)`.
- The checker records `point_replay_mask=127`, `anchor_mask=63`,
  `final_center_mask=7`, `center_candidate_mask=15`,
  `local_center_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `ok_mask=255`.
- The v116 system instance fingerprint is `577937280`, certificate fingerprint
  is `442088371`, center-artifact fingerprint is `554654324`, and artifact
  audit is `135610689`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `119` and
  portfolio v116. The v116 chained manifest has 120 entries; result counters
  are `(unsat=10, validated=108, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=103, meta=2)`, manifest
  fingerprint `905577950`, acceptance receipt `947996951`, audit receipt
  `624876688`, and readiness fingerprint `812289972`.
- Scope boundary: v116 proves only the step-2 center candidate artifact and its
  finite division-witness replay. It is not a radius artifact, remainder
  obligation, proof trace, local containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 radius artifact v117

- Added `lorenz_i256_step2_taylor2_radius_artifact_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_radius_artifact_tiny.sio` and
  imported API/portfolio gates. This consumes the v116 center artifact
  fingerprint `554654324`/audit `135610689`, v115 response envelope
  fingerprint `482555681`/audit `591288676`, and v114 point-time-slab
  fingerprint `268229078`/audit `223161586`.
- The checker converts the v114 step-2 slab margins
  `(26100,257700,13700)` ppm to source-scale radii
  `(112098647,1106813073,58841052)`, with conversion excesses below the
  target scale. It then records derivative radius witnesses
  `dx_rad=12189117200`, `prod_x_rhoz=3091965484`,
  `dy_rad=4198778557`, `prod_xy=1334595202`,
  `beta_rad=156909473`, and `dz_rad=1491504675`.
- Applying `dt_q=42949672` produces next source radii
  `(233989817,1148800858,73756099)`, under caps
  `(500000000,1500000000,100000000)`.
- The checker records `ok_mask=127`, `local_radius_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v117 system instance fingerprint is `844704318`, certificate fingerprint
  is `427274446`, radius-artifact fingerprint is `443171640`, and artifact
  audit is `790224648`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `120` and
  portfolio v117. The v117 chained manifest has 121 entries; result counters
  are `(unsat=10, validated=109, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=104, meta=2)`, manifest
  fingerprint `213780817`, acceptance receipt `75038092`, audit receipt
  `493806823`, and readiness fingerprint `625305600`.
- Scope boundary: v117 proves only the step-2 radius candidate artifact and
  finite radius-budget propagation. It is not a remainder obligation, proof
  trace, local containment proof, local flowpipe proof, multi-step flowpipe
  proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 remainder obligation v118

- Added `lorenz_i256_step2_taylor2_remainder_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_remainder_obligation_tiny.sio`
  and imported API/portfolio gates. This consumes the v115 response envelope
  fingerprint `482555681`/audit `591288676`, the v116 center artifact
  fingerprint `554654324`/audit `135610689`, and the v117 radius artifact
  fingerprint `443171640`/audit `790224648`.
- The checker uses the v116 step-2 center
  `(4406636440,6517899153,4164877513)` and the v117 prior source radii
  `(112098647,1106813073,58841052)` to form absolute enclosure bounds
  `(4518735087,7624712226,4223718565)` and `rhoz_abs=116153047827`.
  The use of prior source radii, rather than the propagated next radii, matches
  the already-registered step-3 remainder-obligation pattern from v103.
- It records first-derivative bounds
  `(121434473130,129829333707,19285209231)` and second-derivative bounds
  `(2512638068370,3434191844255,403599237402)`.
- Applying the Taylor-2 LTE denominator `2*S*S` with `dt_q=42949672`
  produces source-scale LTE bounds `(125631898,171709585,20179961)` and
  target ppm bounds `(29251,39980,4699)`. These stay under caps
  `(40000,55000,10000)`, with slacks `(10749,15020,5301)`.
- The checker records `ok_mask=127`, `local_remainder_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v118 system instance fingerprint is `272280595`, certificate fingerprint
  is `598358642`, remainder-obligation artifact fingerprint is `444092349`,
  and artifact audit is `625429438`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `121` and
  portfolio v118. The v118 chained manifest has 122 entries; result counters
  are `(unsat=10, validated=110, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=105, meta=2)`, manifest
  fingerprint `56035538`, acceptance receipt `206065887`, audit receipt
  `838517422`, and readiness fingerprint `210859660`.
- Scope boundary: v118 proves only the step-2 finite Taylor-2 remainder
  obligation and cap slack replay. It is not a completed candidate bundle,
  proof trace, local containment proof, local flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 completed candidate bundle v119

- Added `lorenz_i256_step2_taylor2_completed_candidate_bundle_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_completed_candidate_bundle_tiny.sio`
  and imported API/portfolio gates. This consumes the v115 response envelope
  fingerprint `482555681`/audit `591288676`, the v116 center artifact
  fingerprint `554654324`/audit `135610689`, the v117 radius artifact
  fingerprint `443171640`/audit `790224648`, and the v118 remainder-obligation
  fingerprint `444092349`/audit `625429438`.
- The checker records `required_artifact_mask=15`, `provided_artifact_mask=15`,
  `missing_artifact_mask=0`, `candidate_status_complete=4`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The v119 system instance fingerprint is `785814651`, certificate fingerprint
  is `513055061`, completed-candidate-bundle fingerprint is `725374434`, and
  artifact audit is `571048339`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `122` and
  portfolio v119. The v119 chained manifest has 123 entries; result counters
  are `(unsat=10, validated=111, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=106, meta=2)`, manifest
  fingerprint `888484153`, acceptance receipt `100654283`, audit receipt
  `576074392`, and readiness fingerprint `998736898`.
- Scope boundary: v119 proves only that the step-2 candidate has the required
  response, center, radius, and remainder artifacts wired together. It is not a
  proof trace, local containment proof, local flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 proof-trace skeleton v120

- Added `lorenz_i256_step2_taylor2_proof_trace_skeleton_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_proof_trace_skeleton_tiny.sio`
  and imported API/portfolio gates. This consumes the v119 completed bundle
  fingerprint `725374434`/audit `571048339` plus step-2 response `482555681`,
  center `554654324`, radius `443171640`, and remainder `444092349`.
- The checker records `trace_version=1`, `trace_kind_step2=2`,
  `trace_node_count=4`, `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, `dependency_root_mask=15`,
  `trace_status_skeleton=5`, `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `ok_mask=255`.
- The v120 system instance fingerprint is `865392417`, certificate fingerprint
  is `259997489`, proof-trace-skeleton fingerprint is `976546207`, and artifact
  audit is `892326627`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `123` and
  portfolio v120. The v120 chained manifest has 124 entries; result counters
  are `(unsat=10, validated=112, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=107, meta=2)`, manifest
  fingerprint `787471364`, acceptance receipt `383893026`, audit receipt
  `492151843`, and readiness fingerprint `558083306`.
- Scope boundary: v120 proves only that a replayable step-2 Taylor-2 trace
  skeleton names the four candidate obligations and replay/root masks under
  the v119/v118/v117/v116/v115 anchors. It is not a replay executor, validated
  enclosure, local-containment proof, local flowpipe proof, multi-step flowpipe
  proof, invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 replay preflight v121

- Added `lorenz_i256_step2_taylor2_replay_preflight_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_replay_preflight_tiny.sio`
  and imported API/portfolio gates. This consumes the v120 proof-trace skeleton
  fingerprint `976546207`/audit `892326627` and the v119 completed bundle
  fingerprint `725374434`.
- The checker records `trace_version=1`, `replay_version=1`,
  `trace_node_count=4`, `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, `replay_status_preflight=6`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and `ok_mask=255`.
- The v121 system instance fingerprint is `688535068`, certificate fingerprint
  is `118139253`, replay-preflight artifact fingerprint is `939672637`, and
  artifact audit is `69896373`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `124` and
  portfolio v121. The v121 chained manifest has 125 entries; result counters
  are `(unsat=10, validated=113, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=108, meta=2)`, manifest
  fingerprint `802098435`, acceptance receipt `281483835`, audit receipt
  `356474349`, and readiness fingerprint `323177089`.
- Scope boundary: v121 proves only that the step-2 Taylor-2 proof-trace
  skeleton and completed-bundle predecessor are ready for a later replay
  executor. It is not replay execution, validated enclosure, local-containment
  proof, local flowpipe proof, multi-step flowpipe proof, invariant/shadowing
  theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 replay executor v122

- Added `lorenz_i256_step2_taylor2_replay_executor_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_replay_executor_tiny.sio`
  and imported API/portfolio gates. This consumes the v121 replay preflight
  fingerprint `939672637`/audit `69896373`, the v120 proof-trace skeleton
  fingerprint `976546207`, and the v119 completed bundle fingerprint
  `725374434`.
- The checker records the replayed step-2 node receipts for response
  `482555681`, center `554654324`, radius `443171640`, and remainder
  `444092349`; `trace_version=1`, `replay_engine_version=1`,
  `trace_node_count=4`, `dependency_edge_count=4`,
  `node_receipt_mask=15`, `edge_receipt_mask=15`,
  `replayed_node_mask=15`, `predecessor_ready_mask=15`,
  `replay_status_executed=7`, `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `ok_mask=255`.
- The v122 system instance fingerprint is `549025432`, certificate fingerprint
  is `561473898`, replay-executor artifact fingerprint is `424013491`, and
  artifact audit is `409402480`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `125` and
  portfolio v122. The v122 chained manifest has 126 entries; result counters
  are `(unsat=10, validated=114, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=109, meta=2)`, manifest
  fingerprint `810874704`, acceptance receipt `647110545`, audit receipt
  `151596995`, and readiness fingerprint `141264735`.
- Scope boundary: v122 proves only replay execution of the step-2 Taylor-2
  proof-trace receipt over the four candidate nodes and dependency masks. It is
  not a validated enclosure, local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 enclosure-validator guard v123

- Added `lorenz_i256_step2_taylor2_enclosure_validator_guard_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_enclosure_validator_guard_tiny.sio`
  and imported API/portfolio gates. This consumes the v122 replay executor
  fingerprint `424013491`/audit `409402480`, the v119 completed bundle
  fingerprint `725374434`, the v117 radius artifact fingerprint `443171640`,
  and the v118 remainder obligation fingerprint `444092349`.
- The guard uses the v117 propagated next-source radii, not the earlier slab
  margin tuple, converted by ceiling to target-scale ppm:
  `(233989817, 1148800858, 73756099)` over `2^32` becomes
  `(54480, 267477, 17173)`. It combines those with v118 LTE ppm
  `(29251, 39980, 4699)` to obtain need ppm `(83731, 307457, 21872)`.
  The guard margins are `(84000, 308000, 22000)`, giving positive slack
  `(269, 543, 128)`.
- The checker records `inclusion_pass_mask=7`, `inclusion_fail_mask=0`,
  `validated_enclosure_mask=0`, `validator_status_guarded=8`,
  `local_containment_proof_mask=0`, `global_flowpipe_claim_mask=0`, and
  `ok_mask=255`.
- The v123 system instance fingerprint is `672898502`, certificate fingerprint
  is `487522414`, enclosure-guard artifact fingerprint is `702220657`, and
  artifact audit is `270117363`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `126` and
  portfolio v123. The v123 chained manifest has 127 entries; result counters
  are `(unsat=10, validated=115, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=110, meta=2)`, manifest
  fingerprint `277596501`, acceptance receipt `125881296`, audit receipt
  `157705536`, and readiness fingerprint `106243394`.
- Scope boundary: v123 proves only that the replayed step-2 radius-plus-LTE
  ppm need tuple is strictly inside the chosen guard margins while all
  enclosure/containment/flowpipe claim masks remain disabled. It is not a
  validated enclosure, local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 local-containment obligation v124

- Added `lorenz_i256_step2_taylor2_local_containment_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_local_containment_obligation_tiny.sio`
  and imported API/portfolio gates. This consumes the v123 enclosure guard
  fingerprint `702220657`/audit `270117363`, the v122 replay executor
  fingerprint `424013491`/audit `409402480`, the v119 completed bundle
  fingerprint `725374434`, the v117 radius artifact fingerprint `443171640`,
  and the v118 remainder obligation fingerprint `444092349`.
- The checker carries forward the same v123 containment arithmetic: radius ppm
  `(54480, 267477, 17173)` plus LTE ppm `(29251, 39980, 4699)` gives need ppm
  `(83731, 307457, 21872)` under margins `(84000, 308000, 22000)` with slack
  `(269, 543, 128)`. It records `axis_containment_mask=7`,
  `local_flowpipe_obligation_mask=7`, `global_flowpipe_claim_mask=0`,
  `obligation_status_local=12`, and `ok_mask=255`.
- The v124 system instance fingerprint is `861466922`, certificate fingerprint
  is `375889750`, local-containment-obligation artifact fingerprint is
  `311696236`, and artifact audit is `301485983`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `127` and
  portfolio v124. The v124 chained manifest has 128 entries; result counters
  are `(unsat=10, validated=116, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=111, meta=2)`, manifest
  fingerprint `449547056`, acceptance receipt `636623594`, audit receipt
  `990540911`, and readiness fingerprint `638783577`.
- Scope boundary: v124 proves only a local-containment obligation receipt that
  replays the per-axis radius-plus-LTE containment arithmetic under the v123
  guard. It is not a local-containment proof, local flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 time-slab containment v125

- Added `lorenz_i256_step2_taylor2_time_slab_containment_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_time_slab_containment_tiny.sio`
  and imported API/portfolio gates. This consumes the v124 local-containment
  obligation fingerprint `311696236`/audit `301485983`, the v114 point
  time-slab fingerprint `268229078`/audit `223161586`, and the v118 remainder
  obligation fingerprint `444092349`.
- The checker records derivative bounds `(2600000, 25756667, 1362223)` at
  `dt=1/100`, ceil sweeps `(26000, 257567, 13623)`, margins
  `(26100, 257700, 13700)`, and slacks `(100, 133, 77)`. It preserves
  `time_slab_containment_mask=7`, `local_time_slab_obligation_mask=7`,
  `global_flowpipe_claim_mask=0`, `time_slab_status_local=13`, and
  `ok_mask=255`.
- The v125 system instance fingerprint is `121434193`, certificate fingerprint
  is `856015850`, time-slab-containment artifact fingerprint is `85766158`,
  and artifact audit is `81234415`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `128` and
  portfolio v125. The v125 chained manifest has 129 entries; result counters
  are `(unsat=10, validated=117, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=112, meta=2)`, manifest
  fingerprint `474619438`, acceptance receipt `991727494`, audit receipt
  `665682181`, and readiness fingerprint `537357117`.
- Scope boundary: v125 proves only a local time-slab-containment receipt that
  replays the derivative-bound ceil-sweep arithmetic and bridges it to the v124
  local-containment obligation. It is not a flowpipe obligation/proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

### 2026-06-23 step-2 Taylor-2 flowpipe obligation v126

- Added `lorenz_i256_step2_taylor2_flowpipe_obligation_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_flowpipe_obligation_tiny.sio`
  and imported API/portfolio gates. This consumes the v125 time-slab
  containment fingerprint `85766158`/audit `81234415`, the v124 local
  containment fingerprint `311696236`/audit `301485983`, the v115 response
  envelope fingerprint `482555681`/audit `591288676`, the v118 remainder
  obligation fingerprint `444092349`/audit `625429438`, and the v113 local
  flowpipe seed fingerprint `633359277`/audit `421619310`.
- The checker composes `endpoint_containment_mask=7`,
  `time_slab_containment_mask=7`, `response_envelope_mask=127`, and
  `remainder_obligation_mask=127` into `composition_ready_mask=15` and
  `local_flowpipe_obligation_mask=15`, while keeping
  `local_flowpipe_proof_mask=0`, `global_flowpipe_claim_mask=0`,
  `flowpipe_obligation_status=20`, and `ok_mask=255`.
- The v126 system instance fingerprint is `280255647`, certificate
  fingerprint is `378373582`, flowpipe-obligation artifact fingerprint is
  `532518987`, and artifact audit is `763325627`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `129` and
  portfolio v126. The v126 chained manifest has 130 entries; result counters
  are `(unsat=10, validated=118, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=113, meta=2)`, manifest
  fingerprint `98914593`, acceptance receipt `374640263`, audit receipt
  `190106026`, and readiness fingerprint `769602917`.
- Scope boundary: v126 proves only a local flowpipe-obligation receipt that
  composes the already-local endpoint/time-slab/response/remainder/seed
  evidence. It is not a local flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-2 Taylor-2 local-flowpipe proof v127

- Added `lorenz_i256_step2_taylor2_local_flowpipe_proof_check()` plus
  `tests/run-pass/lorenz_i256_step2_taylor2_local_flowpipe_proof_tiny.sio`
  and imported API/portfolio gates. This consumes the v126 flowpipe obligation
  fingerprint `532518987`/audit `763325627`, v125 time slab
  `85766158`/`81234415`, v124 local containment `311696236`/`301485983`,
  v115 response envelope `482555681`/`591288676`, v118 remainder obligation
  `444092349`/`625429438`, and v113 local flowpipe seed
  `633359277`/`421619310`.
- The checker composes endpoint/time-slab/response/remainder proof masks
  `(7, 7, 127, 127)` into `composition_ready_mask=15`,
  `local_flowpipe_obligation_mask=15`, and `local_flowpipe_proof_mask=15`,
  while keeping `global_flowpipe_claim_mask=0`, `local_flowpipe_status=21`,
  and `ok_mask=255`.
- The v127 system instance fingerprint is `420503797`, certificate fingerprint
  is `884187166`, local-flowpipe-proof artifact fingerprint is `418029869`,
  and artifact audit is `518930343`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `130` and
  portfolio v127. The v127 chained manifest has 131 entries; result counters
  are `(unsat=10, validated=119, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=114, meta=2)`, manifest
  fingerprint `443349153`, acceptance receipt `957093564`, audit receipt
  `978938017`, and readiness fingerprint `251844917`.
- Scope boundary: v127 proves only a local single-step Taylor-2 flowpipe proof
  receipt for step 2. It is not a multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

### 2026-06-23 step-1 Taylor-2 local-flowpipe seed v128

- Added `lorenz_i256_step1_taylor2_local_flowpipe_seed_check()` plus
  `tests/run-pass/lorenz_i256_step1_taylor2_local_flowpipe_seed_tiny.sio`
  and imported API/portfolio gates. This starts the step-1 backfill instead of
  jumping directly to a local-flowpipe proof.
- The checker replays the initial i256 step from `(2^32, 2^32, 2^32)` to
  `(4294967296, 5411658768, 4223384510)`, anchors the existing step-1 division
  certificate `510854875` and division instance `660429472`, and binds the
  suffix to the two-step chain anchor `335080767` plus step-2 certificate
  `899209716`.
- The seed composes `point_replay_mask=127`, `endpoint_mask=7`,
  `trajectory_anchor_mask=3`, and `suffix_anchor_mask=3` into
  `local_flowpipe_seed_mask=15`, while keeping `local_flowpipe_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, `seed_status=22`, and `ok_mask=255`.
- The v128 system instance fingerprint is `157170836`, certificate fingerprint
  is `689398129`, local-flowpipe-seed artifact fingerprint is `301347724`, and
  artifact audit is `777254975`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `131` and
  portfolio v128. The v128 chained manifest has 132 entries; result counters
  are `(unsat=10, validated=120, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=115, meta=2)`, manifest
  fingerprint `123836674`, acceptance receipt `497055835`, audit receipt
  `442075876`, and readiness fingerprint `506855028`.
- Scope boundary: v128 proves only the step-1 local-flowpipe seed receipt. It
  is not step-1 point-time-slab containment, response-envelope, remainder,
  flowpipe obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 point-time-slab containment v129

Implemented the next local Lorenz replay rung after v128: a step-1 Taylor-2
point-time-slab containment receipt in `stdlib/systems/lorenz_i256_cert.sio`,
plus portfolio v129 registration in `stdlib/theorem/portfolio.sio`.

What the checker now binds:

- The local-flowpipe seed anchor is the v128 step-1 seed artifact
  `301347724` with audit `777254975`.
- The step contract remains i256, scale `2^32`, `dt = 1/100`, step index `1`,
  and non-global status (`global_flowpipe_claim_mask = 0`).
- The derivative/sweep/margin witnesses are:
  - `dx_bound = 0`, `x_sweep = 0`, `x_slab_margin = 1`, `x_slab_slack = 1`.
  - `dy_bound = 111669149696`, `y_sweep = 1116691497`,
    `y_slab_margin = 1116691600`, `y_slab_slack = 103`.
  - `dz_bound = 7158278826`, `z_sweep = 71582789`,
    `z_slab_margin = 71582900`, `z_slab_slack = 111`.
- The system instance fingerprint is `835733741`, certificate fingerprint is
  `776601529`, point-time-slab artifact fingerprint is `44353214`, and artifact
  audit is `431845922`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `132` and
  portfolio v129. The v129 chained manifest has 133 entries; result counters
  are `(unsat=10, validated=121, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=116, meta=2)`, manifest
  fingerprint `706283113`, acceptance receipt `912904205`, audit receipt
  `945348140`, and readiness fingerprint `435724379`.
- Scope boundary: v129 proves only the step-1 point-time-slab containment
  receipt. It is not step-1 response-envelope, center/radius/remainder,
  flowpipe obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 response envelope v130

Implemented the next local Lorenz replay rung after v129: a step-1 Taylor-2
response-envelope receipt in `stdlib/systems/lorenz_i256_cert.sio`, plus
portfolio v130 registration in `stdlib/theorem/portfolio.sio`.

What the checker now binds:

- The point-time-slab anchor is the v129 step-1 point-time-slab artifact
  `44353214` with audit `431845922`.
- The local-flowpipe seed anchor remains the v128 step-1 seed artifact
  `301347724` with audit `777254975`.
- The step contract remains i256, Taylor order `2`, scale `2^32`,
  `dt = 1/100`, step index `1`, and non-global status
  (`global_flowpipe_claim_mask = 0`).
- The response envelope records the candidate kind as ball enclosure and
  requires four downstream local artifacts: center, radius, remainder, and
  proof trace. Those artifacts are not produced by v130.
- The masks are `anchor_mask=15`, `contract_mask=15`,
  `candidate_requirement_mask=15`, `response_envelope_mask=127`,
  `local_response_seed_mask=15`, `local_response_proof_mask=0`, and
  `ok_mask=255`.
- The system instance fingerprint is `670391666`, certificate fingerprint is
  `875065834`, response-envelope artifact fingerprint is `496675613`, and
  artifact audit is `467471545`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `133` and
  portfolio v130. The v130 chained manifest has 134 entries; result counters
  are `(unsat=10, validated=122, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=117, meta=2)`, manifest
  fingerprint `549540177`, acceptance receipt `973777941`, audit receipt
  `733264640`, and readiness fingerprint `275405819`.
- Scope boundary: v130 proves only the step-1 response-envelope receipt. It is
  not center/radius/remainder generation, local containment, flowpipe
  obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 center artifact v131

Implemented the next local Lorenz replay rung after v130: a step-1 Taylor-2
center-artifact receipt in `stdlib/systems/lorenz_i256_cert.sio`, plus
portfolio v131 registration in `stdlib/theorem/portfolio.sio`.

What the checker now binds:

- The response-envelope anchor is the v130 step-1 response-envelope artifact
  `496675613` with audit `467471545`.
- The point-time-slab anchor is the v129 step-1 point-time-slab artifact
  `44353214` with audit `431845922`.
- The local-flowpipe seed anchor remains the v128 step-1 seed artifact
  `301347724` with audit `777254975`.
- The center replay verifies the same quotient/remainder witnesses used by the
  step-1 local-flowpipe seed: `dt_q=42949672`, `dt_r=96`,
  `dy_scaled_q=115964116992`, `xy_scaled_q=4294967296`,
  `beta_q=11453246122`, `x_inc_q=0`, `y_inc_q=1116691472`,
  and `z_drop_q=71582786`.
- The final center is `(4294967296, 5411658768, 4223384510)`, with
  `point_replay_mask=127`, `anchor_mask=63`, `final_center_mask=7`,
  `center_candidate_mask=15`, `local_center_proof_mask=0`, and `ok_mask=255`.
- The system instance fingerprint is `446273096`, certificate fingerprint is
  `155258349`, center artifact fingerprint is `472667880`, and artifact audit
  is `286326819`.
- `stdlib/theorem/portfolio.sio` registers this as kind/checker `134` and
  portfolio v131. The v131 chained manifest has 135 entries; result counters
  are `(unsat=10, validated=123, optimal=1, sat=1)`, checker-family counters
  are `(SAT=6, SMT=4, PB=3, graph=2, Lorenz=118, meta=2)`, manifest
  fingerprint `360785796`, acceptance receipt `178550302`, audit receipt
  `386122134`, and readiness fingerprint `450703322`.
- Scope boundary: v131 proves only the step-1 center replay/candidate artifact.
  It is not a radius artifact, remainder obligation, local containment,
  flowpipe obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 radius artifact v132

Implemented the next local Lorenz replay rung after v131: a step-1 Taylor-2
radius candidate artifact in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v132 registration in `stdlib/theorem/portfolio.sio`.

- The center anchor is the v131 step-1 center artifact fingerprint
  `472667880` with audit `286326819`.
- The response-envelope anchor remains the v130 artifact `496675613` with
  audit `467471545`; the point-time-slab anchor remains v129 artifact
  `44353214` with audit `431845922`.
- The radius target/source bridge records target-scale radii
  `(1,260001,16667)` at `target_scale=1000000`, converted by directed
  source-scale rounding to `(4295,1116695792,71584220)` at `2^32` scale.
  This source radius covers the v129 slab margins; it is deliberately not a
  tight global enclosure claim.
- The derivative-radius witnesses from the v131 center
  `(4294967296,5411658768,4223384510)` are `dx_rad=11167000870`,
  `prod_x_rhoz=71700329`, `dy_rad=1188396121`, `prod_xy=1116702321`,
  `beta_rad=190891255`, and `dz_rad=1307593576`.
- With `dt_q=42949672`, the next source-scale radius candidate is
  `(111674302,1128579753,84660156)` under caps
  `(120000000,1200000000,90000000)`.
- The checker records `ok_mask=127`, `local_radius_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The instance fingerprint is `763744159`,
  certificate fingerprint is `224745160`, radius artifact fingerprint is
  `133154696`, and artifact audit fingerprint is `194048456`.
- Portfolio v132 adds kind/checker `135`. The radius receipt is `313127397`,
  portfolio audit is `155192722`, entry fingerprint is `840177182`, manifest
  is `567532487`, result coverage is `525558358`, family coverage is
  `695592996`, acceptance receipt is `32482341`, audit receipt is `242956216`,
  and readiness is `284405691`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh lorenz_i256_step1_taylor2_radius_artifact`
  (`Pass 2`, `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_radius_artifact_v132`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 137`, `Known failures 128`).
- Scope boundary: v132 proves only a replayable step-1 radius candidate
  artifact and finite radius-budget propagation under the v131 center. It is
  not a Taylor remainder obligation, local containment proof, flowpipe
  obligation, proof trace, multi-step flowpipe proof, invariant/shadowing
  theorem, or global Lorenz theorem.

## 2026-06-23 step-1 remainder obligation v133

Implemented the next local Lorenz replay rung after v132: a step-1 Taylor-2
remainder/LTE obligation receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v133 registration in `stdlib/theorem/portfolio.sio`.

- The response-envelope anchor remains v130 artifact `496675613` with audit
  `467471545`; the center anchor is v131 artifact `472667880` with audit
  `286326819`; the radius anchor is v132 artifact `133154696` with audit
  `194048456`.
- The obligation uses the v132 input source radii
  `(4295,1116695792,71584220)` around the v131 center
  `(4294967296,5411658768,4223384510)`.
- The absolute/Lorenz bounds are `x_abs=4294971591`,
  `y_abs=6528354560`, `z_abs=4294968730`, and
  `rhoz_abs=116107283998`.
- The first-derivative bounds are `x_prime=108233261510`,
  `y_prime=122635754667`, and `z_prime=17981611036`.
- The second-derivative bounds are `x_second=2308690161770`,
  `y_second=3066523255862`, and `z_second=335101505169`.
- With `dt_q=42949672` and denominator `2*scale^2`, the source-scale LTE
  witnesses are `(115434503,153326156,16755075)`, converted to target ppm
  `(26877,35700,3902)` under caps `(35000,45000,10000)` with slacks
  `(8123,9300,6098)`.
- The checker records `ok_mask=127`, `local_remainder_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`. The instance fingerprint is `340568604`,
  certificate fingerprint is `273027742`, remainder obligation fingerprint is
  `265853299`, and artifact audit fingerprint is `919608627`.
- Portfolio v133 adds kind/checker `136`. The remainder receipt is `993840454`,
  portfolio audit is `232141983`, entry fingerprint is `721641728`, manifest
  is `656275273`, result coverage is `577320044`, family coverage is
  `118781846`, acceptance receipt is `223667055`, audit receipt is `850590681`,
  and readiness is `551760019`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_remainder_obligation` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_remainder_obligation_v133`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 138`, `Known failures 129`).
- Scope boundary: v133 proves only a replayable local Taylor-2 remainder/LTE
  obligation under the v132 radius artifact. It is not a completed candidate
  bundle, local containment proof, flowpipe obligation, proof trace,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

## 2026-06-23 step-1 completed candidate bundle v134

Implemented the next local Lorenz replay rung after v133: a step-1 Taylor-2
completed candidate bundle receipt in `stdlib/systems/lorenz_i256_cert.sio`,
with portfolio v134 registration in `stdlib/theorem/portfolio.sio`.

- The response-envelope anchor remains v130 artifact `496675613` with audit
  `467471545`; the center anchor is v131 artifact `472667880` with audit
  `286326819`; the radius anchor is v132 artifact `133154696` with audit
  `194048456`; the remainder-obligation anchor is v133 artifact `265853299`
  with audit `919608627`.
- The checker records `required_artifact_mask=15`,
  `provided_artifact_mask=15`, `missing_artifact_mask=0`,
  `candidate_status_complete=4`, `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
  This is deliberately a completed candidate bundle, not a local-containment
  or flowpipe proof.
- The completed-bundle `ok_mask` is `255`. The instance fingerprint is
  `338339305`, certificate fingerprint is `836456562`, completed candidate
  bundle fingerprint is `620127047`, and artifact audit fingerprint is
  `956998285`.
- Portfolio v134 adds kind/checker `137`. The completed-bundle receipt is
  `980632939`, portfolio audit is `33113047`, entry fingerprint is
  `113040423`, manifest is `350381416`, result coverage is `234445087`,
  family coverage is `147334060`, acceptance receipt is `826869557`, audit
  receipt is `844106555`, and readiness is `536505063`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_completed_candidate_bundle` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_completed_candidate_bundle_v134`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 139`, `Known failures 130`).
- Scope boundary: v134 proves only that the step-1 candidate bundle has all
  four required replay artifacts present and consistently fingerprinted. It is
  not a proof-trace skeleton, replay preflight, replay executor, local
  containment proof, flowpipe obligation, local-flowpipe proof, multi-step
  flowpipe proof, invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 proof-trace skeleton v135

Implemented the next local Lorenz replay rung after v134: a step-1 Taylor-2
proof-trace skeleton receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v135 registration in `stdlib/theorem/portfolio.sio`.

- The completed-bundle anchor is v134 artifact `620127047` with audit
  `956998285`. The trace keeps the same four replay dependencies:
  response-envelope `496675613`, center `472667880`, radius `133154696`, and
  remainder-obligation `265853299`.
- The trace skeleton records `trace_version=1`, `trace_kind_step1=1`,
  `trace_node_count=4`, `dependency_edge_count=4`, `obligation_mask=15`,
  `replay_order_mask=15`, `dependency_root_mask=15`,
  `trace_status_skeleton=5`, `validated_enclosure_mask=0`,
  `local_containment_proof_mask=0`, and `global_flowpipe_claim_mask=0`.
- The skeleton `ok_mask` is `255`. The instance fingerprint is `955923870`,
  certificate fingerprint is `311805844`, proof-trace skeleton fingerprint is
  `174108453`, and artifact audit fingerprint is `516721287`.
- Portfolio v135 adds kind/checker `138`. The proof-trace-skeleton receipt is
  `487794978`, portfolio audit is `55049153`, entry fingerprint is
  `847786813`, manifest is `925645053`, result coverage is `772727624`,
  family coverage is `57043761`, acceptance receipt is `921070257`, audit
  receipt is `301067060`, and readiness is `494223181`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_proof_trace_skeleton` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_proof_trace_skeleton_v135`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 140`, `Known failures 131`).
- Scope boundary: v135 proves only that a replay-facing trace skeleton exists
  for the step-1 candidate bundle and orders the four dependency nodes. It is
  not replay preflight, replay execution, validated enclosure, local containment
  proof, flowpipe obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 replay preflight v136

Implemented the next local Lorenz replay rung after v135: a step-1 Taylor-2
replay-preflight receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v136 registration in `stdlib/theorem/portfolio.sio`.

- The proof-trace-skeleton anchor is v135 artifact `174108453` with audit
  `516721287`; the completed-candidate-bundle anchor remains v134 artifact
  `620127047`.
- The replay preflight records `step_index=1`, `trace_version=1`,
  `replay_version=1`, `trace_node_count=4`, `dependency_edge_count=4`,
  `obligation_mask=15`, `replay_order_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, `replay_status_preflight=6`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The replay-preflight `ok_mask` is `255`. The instance fingerprint is
  `674339367`, certificate fingerprint is `972235050`, replay-preflight
  fingerprint is `261545659`, and artifact audit fingerprint is `928945135`.
- Portfolio v136 adds kind/checker `139`. The replay-preflight receipt is
  `570323855`, portfolio audit is `922761694`, entry fingerprint is
  `947539911`, manifest is `665055088`, result coverage is `475156559`,
  family coverage is `130899867`, acceptance receipt is `3449148`, audit
  receipt is `441075153`, and readiness is `245102468`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_replay_preflight` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_replay_preflight_v136`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 141`, `Known failures 132`).
- Scope boundary: v136 proves only that the four replay nodes for step 1 are
  present, ordered, replay-marked, and predecessor-ready before execution. It
  is not replay execution, validated enclosure, local containment proof,
  flowpipe obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 replay executor v137

Implemented the next local Lorenz replay rung after v136: a step-1 Taylor-2
replay-executor receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v137 registration in `stdlib/theorem/portfolio.sio`.

- The replay-preflight anchor is v136 artifact `261545659` with audit
  `928945135`; the proof-trace-skeleton anchor remains v135 artifact
  `174108453`; the completed-candidate-bundle anchor remains v134 artifact
  `620127047`.
- The replay executor records the four candidate nodes: response envelope
  `496675613`, center `472667880`, radius `133154696`, and remainder
  obligation `265853299`. It pins `trace_version=1`,
  `replay_engine_version=1`, `trace_node_count=4`,
  `dependency_edge_count=4`, `node_receipt_mask=15`,
  `edge_receipt_mask=15`, `replayed_node_mask=15`,
  `predecessor_ready_mask=15`, `replay_status_executed=7`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The replay-executor `ok_mask` is `255`. The instance fingerprint is
  `388821571`, certificate fingerprint is `316195300`, replay-executor
  fingerprint is `815221512`, and artifact audit fingerprint is `981617998`.
- Portfolio v137 adds kind/checker `140`. The replay-executor receipt is
  `907464772`, portfolio audit is `901239486`, entry fingerprint is
  `954097866`, manifest is `352621120`, result coverage is `125741491`,
  family coverage is `152911970`, acceptance receipt is `110303919`, audit
  receipt is `557268780`, and readiness is `937727692`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_replay_executor` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_replay_executor_v137`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 142`, `Known failures 133`).
- Scope boundary: v137 proves only that the step-1 replay executor consumed
  all four recorded candidate-node receipts under the v136 preflight and kept
  predecessor, node, and edge masks closed. It is not a validated enclosure,
  local containment proof, flowpipe obligation, local-flowpipe proof,
  multi-step flowpipe proof, invariant/shadowing theorem, or global Lorenz
  theorem.

## 2026-06-23 step-1 enclosure validator guard v138

Implemented the next local Lorenz replay rung after v137: a step-1 Taylor-2
enclosure-validator guard receipt in `stdlib/systems/lorenz_i256_cert.sio`,
with portfolio v138 registration in `stdlib/theorem/portfolio.sio`.

- The replay-executor anchor is v137 artifact `815221512` with audit
  `981617998`; the completed-candidate-bundle anchor remains v134 artifact
  `620127047`; the radius and remainder anchors remain v132 artifact
  `133154696` and v133 artifact `265853299`.
- The guard records the target-scale inclusion operands:
  radius `(1,260001,16667)`, LTE `(26877,35700,3902)`, need
  `(26878,295701,20569)`, margin `(27000,296000,21000)`, and slack
  `(122,299,431)`. It pins `inclusion_pass_mask=7`,
  `inclusion_fail_mask=0`, `validator_status_guarded=8`,
  `validated_enclosure_mask=0`, `local_containment_proof_mask=0`, and
  `global_flowpipe_claim_mask=0`.
- The enclosure-guard `ok_mask` is `255`. The instance fingerprint is
  `741550055`, certificate fingerprint is `156734502`, guard fingerprint is
  `737436919`, and artifact audit fingerprint is `387055804`.
- Portfolio v138 adds kind/checker `141`. The enclosure-guard receipt is
  `689786971`, portfolio audit is `217168551`, entry fingerprint is
  `503206887`, manifest is `784470841`, result coverage is `520610112`,
  family coverage is `919207762`, acceptance receipt is `35987707`, audit
  receipt is `881923867`, and readiness is `766395700`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_enclosure_validator_guard` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_enclosure_validator_guard_v138`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 143`, `Known failures 134`).
- Scope boundary: v138 proves only a guarded axis-wise margin check for the
  step-1 candidate envelope. It deliberately keeps `validated_enclosure_mask=0`
  and does not claim a validated enclosure, local containment proof, flowpipe
  obligation, local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 local containment obligation v139

Implemented the next local Lorenz replay rung after v138: a step-1 Taylor-2
local-containment-obligation receipt in `stdlib/systems/lorenz_i256_cert.sio`,
with portfolio v139 registration in `stdlib/theorem/portfolio.sio`.

- The enclosure-guard anchor is v138 artifact `737436919` with audit
  `387055804`; the replay-executor anchor remains v137 artifact `815221512`
  with audit `981617998`; the completed-candidate-bundle, radius, and
  remainder anchors remain `620127047`, `133154696`, and `265853299`.
- The local obligation records the same target-scale operands from v138:
  radius `(1,260001,16667)`, LTE `(26877,35700,3902)`, need
  `(26878,295701,20569)`, margin `(27000,296000,21000)`, and positive slack
  `(122,299,431)`. It pins `axis_containment_mask=7`,
  `local_flowpipe_obligation_mask=7`, `obligation_status_local=12`, and
  `global_flowpipe_claim_mask=0`.
- The local-containment-obligation `ok_mask` is `255`. The instance fingerprint
  is `220504517`, certificate fingerprint is `778022451`, obligation artifact
  fingerprint is `473585568`, and artifact audit fingerprint is `790380830`.
- Portfolio v139 adds kind/checker `142`. The local-obligation receipt is
  `26998442`, portfolio audit is `558635697`, entry fingerprint is
  `382273466`, manifest is `893496809`, result coverage is `592654980`,
  family coverage is `362679794`, acceptance receipt is `744777140`, audit
  receipt is `56048069`, and readiness is `743392198`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_local_containment_obligation` (`Pass 2`,
  `Known failures 2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_local_containment_obligation_v139`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 144`, `Known failures 135`).
- Scope boundary: v139 proves only a local containment obligation with positive
  per-axis slack under the existing step-1 replay/enclosure guards. It is not a
  validated enclosure, time-slab containment proof, flowpipe obligation,
  local-flowpipe proof, multi-step flowpipe proof, invariant/shadowing theorem,
  or global Lorenz theorem.

## 2026-06-23 step-1 time-slab containment v140

Implemented the next local Lorenz replay rung after v139: a step-1 Taylor-2
time-slab-containment receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v140 registration in `stdlib/theorem/portfolio.sio`.

- The local-containment anchor is v139 artifact `473585568` with audit
  `790380830`; the point-time-slab anchor is v129 artifact `44353214` with
  audit `431845922`; the remainder anchor remains v133 artifact `265853299`.
- The time-slab obligation records source-scale derivative bounds
  `(0,111669149696,7158278826)`, `dt=1/100`, sweep ceilings
  `(0,1116691497,71582789)`, slab margins `(1,1116691600,71582900)`, and
  positive slacks `(1,103,111)`. It pins `time_slab_containment_mask=7`,
  `local_time_slab_obligation_mask=7`, `time_slab_status_local=13`, and
  `global_flowpipe_claim_mask=0`.
- The time-slab-containment `ok_mask` is `255`. The instance fingerprint is
  `781293646`, certificate fingerprint is `561666691`, time-slab artifact
  fingerprint is `180607828`, and artifact audit fingerprint is `551960823`.
- Portfolio v140 adds kind/checker `143`. The time-slab receipt is `62233558`,
  portfolio audit is `104592626`, entry fingerprint is `940931367`, manifest
  is `788316508`, result coverage is `450493579`, family coverage is
  `591945564`, acceptance receipt is `95823883`, audit receipt is `978175425`,
  and readiness is `191710845`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_time_slab_containment` (`Pass 2`, `Known failures
  2`), `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_time_slab_containment_v140`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 145`, `Known failures 136`).
- Scope boundary: v140 proves only a single-step local time-slab containment
  obligation under the existing point-time-slab and local-containment receipts.
  It is not a flowpipe obligation, local-flowpipe proof, multi-step flowpipe
  proof, invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 flowpipe obligation v141

Implemented the next local Lorenz replay rung after v140: a step-1 Taylor-2
flowpipe-obligation receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v141 registration in `stdlib/theorem/portfolio.sio`.

- The flowpipe-obligation checker composes the v140 time-slab containment
  artifact `180607828`/audit `551960823`, the v139 local-containment
  obligation `473585568`/audit `790380830`, the v130 response envelope
  `496675613`/audit `467471545`, the v133 remainder obligation
  `265853299`/audit `919608627`, and the v128 local-flowpipe seed
  `301347724`/audit `777254975`.
- The composition pins endpoint and time-slab containment masks `7`, response
  and remainder masks `127`, `composition_ready_mask=15`,
  `local_flowpipe_obligation_mask=15`, `local_flowpipe_proof_mask=0`,
  `global_flowpipe_claim_mask=0`, and local obligation status `20`.
- The flowpipe-obligation `ok_mask` is `255`. The instance fingerprint is
  `180287901`, certificate fingerprint is `534209661`, flowpipe-obligation
  artifact fingerprint is `90632144`, and artifact audit fingerprint is
  `922642400`.
- Portfolio v141 adds kind/checker `144`. The flowpipe-obligation receipt is
  `579621511`, portfolio audit is `392918762`, entry fingerprint is
  `246755403`, manifest is `730796448`, result coverage is `355992419`,
  family coverage is `868871575`, acceptance receipt is `853879678`, audit
  receipt is `893659579`, and readiness is `639513798`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks,
  `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_flowpipe_obligation` (`Pass 2`, `Known failures
  2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_flowpipe_obligation_v141` (`Pass
  1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 146`, `Known failures 137`).
- Scope boundary: v141 proves only that the already-local endpoint/time-slab/
  response/remainder/seed ingredients are present for a step-1 local flowpipe
  obligation. It is not a local-flowpipe proof, multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem.

## 2026-06-23 step-1 local-flowpipe proof v142

Implemented the next local Lorenz replay rung after v141: a step-1 Taylor-2
local-flowpipe-proof receipt in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v142 registration in `stdlib/theorem/portfolio.sio`.

- The local proof checker consumes the v141 flowpipe obligation
  `90632144`/`922642400`, v140 time-slab containment
  `180607828`/`551960823`, v139 local-containment obligation
  `473585568`/`790380830`, v130 response envelope
  `496675613`/`467471545`, v133 remainder obligation
  `265853299`/`919608627`, and v128 local-flowpipe seed
  `301347724`/`777254975`.
- The proof-side masks are explicit: endpoint and time-slab proof masks `7`,
  response and remainder proof masks `127`, `composition_ready_mask=15`,
  `local_flowpipe_obligation_mask=15`, `local_flowpipe_proof_mask=15`,
  `global_flowpipe_claim_mask=0`, and local proof status `21`.
- The local-flowpipe-proof `ok_mask` is `255`. The instance fingerprint is
  `531298954`, certificate fingerprint is `566562700`, local-flowpipe-proof
  artifact fingerprint is `908974219`, and artifact audit fingerprint is
  `890221848`.
- Portfolio v142 adds kind/checker `145`. The local-flowpipe-proof receipt is
  `989878921`, portfolio audit is `995615032`, entry fingerprint is
  `394497389`, manifest is `976586199`, result coverage is `564801070`,
  family coverage is `449107390`, acceptance receipt is `327543567`, audit
  receipt is `790297139`, and readiness is `97103871`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks,
  `./scripts/run_sio_test_suite.sh
  lorenz_i256_step1_taylor2_local_flowpipe_proof` (`Pass 2`, `Known failures
  2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_taylor2_local_flowpipe_proof_v142`
  (`Pass 1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh
  portfolio` (`Pass 147`, `Known failures 138`).
- Scope boundary: v142 proves only a single-step local flowpipe receipt under
  the recorded step-1 obligations. It is not a multi-step flowpipe proof,
  invariant/shadowing theorem, or global Lorenz theorem; the global claim mask
  remains zero.

## 2026-06-23 step-1-to-step-2 local-flowpipe bridge v143

Implemented the next contextual Lorenz replay rung after v142: a bridge receipt
from the step-1 local-flowpipe proof into the already-existing step-1 seed and
step-2 chain context, in `stdlib/systems/lorenz_i256_cert.sio`, with portfolio
v143 registration in `stdlib/theorem/portfolio.sio`.

- The bridge checker consumes the v142 step-1 local-flowpipe proof
  `908974219`/`890221848`, the v128 step-1 local-flowpipe seed
  `301347724`/`777254975`, the step-1 division/certificate context
  `660429472`/`510854875`, and the existing two-step chain context
  `335080767`/`899209716`.
- The bridge-side masks are explicit:
  `local_flowpipe_proof_mask=15`, `seed_anchor_mask=15`,
  `chain_context_mask=15`, `bridge_anchor_mask=15`,
  `chain_extension_mask=15`, `global_flowpipe_claim_mask=0`, and bridge
  status `22`.
- The bridge `ok_mask` is `255`. The instance fingerprint is `148904669`,
  certificate fingerprint is `278828475`, chain-bridge artifact fingerprint is
  `595466240`, and artifact audit fingerprint is `382059454`.
- Portfolio v143 adds kind/checker `146`. The bridge receipt is `881374690`,
  portfolio audit is `145539674`, entry fingerprint is `961360409`, manifest is
  `183919651`, result coverage is `735153429`, family coverage is `990886920`,
  acceptance receipt is `751938858`, audit receipt is `619775358`, and
  readiness is `659832949`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh step1_step2_local_flowpipe_bridge`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step1_step2_local_flowpipe_bridge_v143` (`Pass
  1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 148`, `Known failures 139`).
- Scope boundary: v143 is a local chain-context bridge only. It does not prove a
  two-step flowpipe, a multi-step flowpipe, an invariant/shadowing theorem, or a
  global Lorenz theorem; the global claim mask remains zero.

## 2026-06-23 step-2-to-step-3 local-flowpipe bridge v144

Implemented the next contextual Lorenz replay bridge after v143: a bridge
receipt from the already-existing step-2 local-flowpipe proof into the step-2
seed and step-3 chain context, in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v144 registration in `stdlib/theorem/portfolio.sio`.

- The bridge checker consumes the v127 step-2 local-flowpipe proof
  `418029869`/`518930343`, the v113 step-2 local-flowpipe seed
  `633359277`/`421619310`, the existing two-step chain context
  `335080767`/`899209716`, and the existing three-step suffix context
  `249889958`/`603078026`.
- The bridge-side masks are explicit:
  `local_flowpipe_proof_mask=15`, `seed_anchor_mask=15`,
  `chain_context_mask=15`, `bridge_anchor_mask=15`,
  `chain_extension_mask=15`, `global_flowpipe_claim_mask=0`, and bridge
  status `23`.
- The bridge `ok_mask` is `255`. The instance fingerprint is `991315838`,
  certificate fingerprint is `951485822`, chain-bridge artifact fingerprint is
  `158202075`, and artifact audit fingerprint is `952956886`.
- Portfolio v144 adds kind/checker `147`. The bridge receipt is `151952098`,
  portfolio audit is `395014347`, entry fingerprint is `687754680`, manifest is
  `658146531`, result coverage is `172399202`, family coverage is `799559864`,
  acceptance receipt is `543027310`, audit receipt is `612345619`, and
  readiness is `509107988`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh step2_step3_local_flowpipe_bridge`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step2_step3_local_flowpipe_bridge_v144` (`Pass
  1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 149`, `Known failures 140`).
- Scope boundary: v144 is a local chain-context bridge only. It does not prove a
  three-step flowpipe, a multi-step flowpipe, an invariant/shadowing theorem, or
  a global Lorenz theorem; the global claim mask remains zero.

## 2026-06-23 step-3-to-step-4 local-flowpipe bridge v145

Implemented the next contextual Lorenz replay bridge after v144: a bridge
receipt from the already-existing step-3 local-flowpipe proof into the step-3
seed and step-4 chain context, in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v145 registration in `stdlib/theorem/portfolio.sio`.

- The bridge checker consumes the v112 step-3 local-flowpipe proof
  `78860411`/`972719127`, the v98 step-3 local-flowpipe seed
  `389654052`/`790630993`, the existing three-step chain context
  `249889958`/`603078026`, and the existing four-step suffix context
  `737039167`/`753371133`.
- The bridge-side masks are explicit:
  `local_flowpipe_proof_mask=15`, `seed_anchor_mask=15`,
  `chain_context_mask=15`, `bridge_anchor_mask=15`,
  `chain_extension_mask=15`, `global_flowpipe_claim_mask=0`, and bridge
  status `24`.
- The bridge `ok_mask` is `255`. The instance fingerprint is `151689640`,
  certificate fingerprint is `22530645`, chain-bridge artifact fingerprint is
  `692714944`, and artifact audit fingerprint is `627937430`.
- Portfolio v145 adds kind/checker `148`. The bridge receipt is `190299596`,
  portfolio audit is `946465531`, entry fingerprint is `422116880`, manifest is
  `906661736`, result coverage is `383933307`, family coverage is `382521133`,
  acceptance receipt is `647910597`, audit receipt is `792162997`, and readiness
  is `311513640`. The Lorenz family count is cumulative portfolio coverage, not
  an additional claim made by this bridge.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh step3_step4_local_flowpipe_bridge`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step3_step4_local_flowpipe_bridge_v145` (`Pass
  1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 150`, `Known failures 141`).
- Scope boundary: v145 is a local chain-context bridge only. It does not prove a
  four-step flowpipe, a multi-step flowpipe, an invariant/shadowing theorem, or
  a global Lorenz theorem; the global claim mask remains zero.

## 2026-06-23 step-4-to-step-5 local-flowpipe bridge v146

Implemented the next contextual Lorenz replay bridge after v145: a bridge
receipt from the already-existing step-4 local-flowpipe proof into the step-4
seed and step-5 chain context, in `stdlib/systems/lorenz_i256_cert.sio`, with
portfolio v146 registration in `stdlib/theorem/portfolio.sio`.

- The bridge checker consumes the v97 step-4 local-flowpipe proof
  `354200144`/`97568991`, the v83 step-4 local-flowpipe seed
  `563331437`/`684718822`, the existing four-step chain context
  `737039167`/`753371133`, and the existing five-step suffix context
  `23144051`/`561641681`.
- The bridge-side masks are explicit:
  `local_flowpipe_proof_mask=15`, `seed_anchor_mask=15`,
  `chain_context_mask=15`, `bridge_anchor_mask=15`,
  `chain_extension_mask=15`, `global_flowpipe_claim_mask=0`, and bridge
  status `25` as a local bridge-status tag only.
- The bridge `ok_mask` is `255`. The instance fingerprint is `274535943`,
  certificate fingerprint is `889322717`, chain-bridge artifact fingerprint is
  `880867663`, and artifact audit fingerprint is `674753000`.
- Portfolio v146 adds kind/checker `149`. The bridge receipt is `984096488`,
  portfolio audit is `876128531`, entry fingerprint is `654864830`, manifest is
  `139046199`, result coverage is `579336677`, family coverage is `949351674`,
  acceptance receipt is `412537629`, audit receipt is `652056951`, and readiness
  is `514521290`. The Lorenz family count is cumulative portfolio coverage, not
  an additional claim made by this bridge.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh step4_step5_local_flowpipe_bridge`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_step4_step5_local_flowpipe_bridge_v146` (`Pass
  1`, `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 151`, `Known failures 142`).
- Scope boundary: v146 is a local chain-context bridge only. It does not prove a
  five-step flowpipe, a multi-step flowpipe, an invariant/shadowing theorem, or
  a global Lorenz theorem; the global claim mask remains zero.

## 2026-06-23 five-step local-flowpipe chain-composition gate v147

Implemented the first explicit five-step local-flowpipe chain-composition gate:
`lorenz_i256_five_step_local_flowpipe_chain_check()` in
`stdlib/systems/lorenz_i256_cert.sio`, with portfolio v147 registration in
`stdlib/theorem/portfolio.sio`.

- The chain checker consumes the five existing single-step local-flowpipe proof
  artifacts/audits:
  step 1 `908974219`/`890221848`, step 2 `418029869`/`518930343`,
  step 3 `78860411`/`972719127`, step 4 `354200144`/`97568991`,
  and step 5 `965829478`/`618466007`.
- It also consumes the four adjacent local-flowpipe bridge artifacts/audits:
  step 1->2 `595466240`/`382059454`, step 2->3
  `158202075`/`952956886`, step 3->4 `692714944`/`627937430`,
  and step 4->5 `880867663`/`674753000`, plus the five-step chain
  anchors `214180161`, `23144051`, and `561641681`.
- The chain-side masks are explicit:
  `local_proof_chain_mask=31`, `bridge_chain_mask=15`,
  `chain_anchor_mask=7`, `adjacency_chain_mask=15`,
  `chain_composition_mask=31`, `global_flowpipe_claim_mask=0`, and chain
  status `26` as a local chain-composition tag only.
- The chain `ok_mask` is `255`. The instance fingerprint is `449273642`,
  certificate fingerprint is `114764949`, local-chain artifact fingerprint is
  `911209450`, and artifact audit fingerprint is `709377850`.
- Portfolio v147 adds kind/checker `150`. The chain receipt is `527088467`,
  portfolio audit is `492682642`, entry fingerprint is `657988743`, manifest is
  `867133305`, result coverage is `270442676`, family coverage is `11884837`,
  acceptance receipt is `578754011`, audit receipt is `755711125`, and readiness
  is `31177321`. The Lorenz family count is cumulative portfolio coverage, not
  an additional mathematical claim by this gate.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh five_step_local_flowpipe_chain`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_five_step_local_flowpipe_chain_v147` (`Pass 1`,
  `Known failures 1`), and `./scripts/run_sio_test_suite.sh portfolio`
  (`Pass 152`, `Known failures 143`).
- Scope boundary: v147 is a local five-step chain-composition gate only. It does
  not prove a global flowpipe, an invariant/shadowing theorem, or a global
  Lorenz theorem; `global_flowpipe_claim_mask` remains `0`.

## 2026-06-23 global-flowpipe non-claim preflight v148

Implemented a deliberately non-global preflight gate:
`lorenz_i256_global_flowpipe_claim_preflight_check()` in
`stdlib/systems/lorenz_i256_cert.sio`, with portfolio v148 registration in
`stdlib/theorem/portfolio.sio`.

- The preflight consumes the v147 five-step local-chain artifact/audit
  `911209450`/`709377850`, portfolio entry/manifest/readiness
  `657988743`/`867133305`/`31177321`, and the existing five-step trajectory
  anchors `214180161`, `23144051`, and `561641681`.
- It records `available_evidence_mask=15` for the finite local chain and its
  portfolio binding, but also records `missing_global_obligation_mask=15` for
  the four intentionally absent global obligations: invariant certificate,
  shadowing/stability certificate, global cover certificate, and unbounded-time
  certificate.
- The global readiness side is explicitly negative:
  `global_claim_ready_mask=0`, `global_flowpipe_claim_mask=0`, blocker count
  `4`, preflight version `1`, and status `27` as a non-claim preflight tag.
- The preflight `ok_mask` is `255` because the non-claim audit is internally
  consistent. The instance fingerprint is `549978893`, certificate fingerprint
  is `696181837`, preflight artifact fingerprint is `461075499`, and artifact
  audit fingerprint is `641858696`.
- Portfolio v148 adds kind/checker `151`. The preflight receipt is `507928544`,
  portfolio audit is `772088455`, entry fingerprint is `467412992`, manifest is
  `978167597`, result coverage is `344495868`, family coverage is `457365200`,
  acceptance receipt is `184216452`, audit receipt is `36358799`, and readiness
  is `329959989`. This entry is portfolio coverage for a guardrail, not a global
  theorem.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh global_flowpipe_claim_preflight`
  (`Pass 2`, `Known failures 2`),
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_global_flowpipe_claim_preflight_v148`
  (`Pass 1`, `Known failures 1`), and
  `./scripts/run_sio_test_suite.sh portfolio` (`Pass 153`, `Known failures
  144`).
- Scope boundary: v148 is a global-claim preflight and blocker ledger only. It
  does not prove a global flowpipe, an invariant/shadowing theorem, or a global
  Lorenz theorem; `global_flowpipe_claim_mask` remains `0`.

## 2026-06-23 finite-cover candidate seed v149

The next Lorenz i256 move is a deliberately narrow finite-cover prerequisite,
not a global-cover certificate. `lorenz_i256_finite_cover_candidate_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds a single terminal enclosure cell
candidate to the v148 global-claim preflight, the projection certificate
envelope, the projection dependency DAG, the five-step local chain, and the
existing range/roundoff/margin/enclosure budget anchors.

- System anchors: the candidate consumes v148 preflight artifact/audit
  `461075499`/`641858696`, v148 portfolio entry/manifest/readiness
  `467412992`/`978167597`/`329959989`, projection certificate envelope
  `77888110` with audit `245178976`, projection dependency DAG `713010204`
  with audit `725222703`, v38 readiness `845057161`, final enclosure receipt
  `114244647`, ball-fixed enclosure `166945519`, range budget `87220662`,
  roundoff budget `677514412`, margin budget `69322697`, and five-step local
  chain `911209450` with audit `709377850`.
- The candidate intentionally records `candidate_cell_count = 1`,
  `covered_cell_count = 1`, `finite_cover_candidate_mask = 31`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
  This says "one cover cell candidate is wired and replayable"; it does not say
  "the global cover obligation is discharged."
- System fingerprints: instance `906677279`, certificate `409602997`, artifact
  `911169246`, audit `785916873`. The artifact kind is `20`, scoped to the
  finite-cover candidate seed.
- Portfolio v149 adds kind/checker `152`. The finite-cover candidate receipt is
  `41883869`, audit is `324028289`, entry fingerprint is `820916947`, manifest
  is `527721765`, result coverage is `857068943`, family coverage is
  `341365439`, acceptance receipt is `697596120`, audit receipt is
  `773584282`, and readiness is `437511874`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh finite_cover_candidate`,
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_finite_cover_candidate_v149`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v149 is a finite-cover candidate seed. It does not prove a
  global cover, invariant theorem, shadowing theorem, unbounded-time property,
  or global Lorenz theorem. The v148 missing-global-obligation mask remains the
  truthful ledger until a real cover certificate and the other global
  obligations are separately checked.

## 2026-06-23 cover-refinement ledger v150

`lorenz_i256_cover_refinement_ledger_check()` in
`stdlib/systems/lorenz_i256_cert.sio` turns the v149 single-cell candidate into
a replayable refinement ledger. The ledger records a `2 x 2 x 2` split of the
one parent cell into eight child obligations, all still pending. This is the
next cover-building data structure, not a cover certificate.

- System anchors: the ledger consumes v149 finite-cover candidate artifact/audit
  `911169246`/`785916873`, v149 portfolio entry/manifest/readiness
  `820916947`/`527721765`/`437511874`, v148 preflight artifact/audit
  `461075499`/`641858696`, projection dependency DAG `713010204` with audit
  `725222703`, projection certificate envelope `77888110` with audit
  `245178976`, final enclosure receipt `114244647`, and ball-fixed enclosure
  `166945519`.
- Refinement shape: `candidate_cell_count = 1`, `parent_cell_count = 1`,
  `split_axis_count = 3`, `split_factor_per_axis = 2`, `child_cell_count = 8`,
  `pending_child_obligation_count = 8`, `resolved_child_count = 0`, and
  `refinement_level = 1`.
- Masks: `cover_refinement_input_mask = 31`, `subdivision_plan_mask = 7`,
  `child_obligation_mask = 15`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`. The eight children are obligations to be
  discharged later, not proof that the parent has a certified global cover.
- System fingerprints: instance `236980132`, certificate `747708140`, artifact
  `239912256`, audit `937619929`. The artifact kind is `21`, scoped to the
  cover-refinement ledger.
- Portfolio v150 adds kind/checker `153`. The ledger receipt is `893139927`,
  audit is `132235218`, entry fingerprint is `974395903`, manifest is
  `568368767`, result coverage is `860734845`, family coverage is `716458512`,
  acceptance receipt is `106875278`, audit receipt is `658039681`, and
  readiness is `228495843`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh cover_refinement_ledger`,
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_refinement_ledger_v150`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v150 creates a child-obligation ledger for finite-cover
  refinement. It does not prove the eight children, a global cover, invariant
  theorem, shadowing theorem, unbounded-time property, global flowpipe, or
  global Lorenz theorem.

## 2026-06-23 cover child-0 obligation seed v151

`lorenz_i256_cover_child0_obligation_seed_check()` in
`stdlib/systems/lorenz_i256_cert.sio` selects the first child from the v150
`2 x 2 x 2` refinement ledger. The selected child is index `0`, with slots
`(0, 0, 0)`. This is the first replayable child-obligation seed, not a local
flowpipe proof and not a discharged cover child.

- System anchors: the child seed consumes the v150 cover-refinement ledger
  artifact/audit `239912256`/`937619929`, v150 portfolio
  entry/manifest/readiness `974395903`/`568368767`/`228495843`, v149 finite
  cover candidate artifact/audit `911169246`/`785916873`, projection dependency
  DAG `713010204` with audit `725222703`, and projection certificate envelope
  `77888110` with audit `245178976`.
- Child shape: `child_index = 0`, `child_x_slot = 0`, `child_y_slot = 0`,
  `child_z_slot = 0`, `split_axis_count = 3`, `split_factor_per_axis = 2`,
  `child_cell_count = 8`, `selected_child_count = 1`,
  `pending_child_obligation_count = 8`, and `resolved_child_count = 0`.
- Masks: `selected_child_mask = 1`, `child_coordinate_mask = 7`,
  `inherited_anchor_mask = 15`, `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `644624864`, certificate `702528645`, artifact
  `577517715`, audit `49794723`. The artifact kind is `22`, scoped to the
  child-0 obligation seed.
- Portfolio v151 adds kind/checker `154`. The child seed receipt is
  `402674780`, audit is `117291494`, entry fingerprint is `441058878`,
  manifest is `752370901`, result coverage is `7755872`, family coverage is
  `234906710`, acceptance receipt is `429064177`, audit receipt is
  `933996794`, and readiness is `960341230`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh cover_child0_obligation_seed`,
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_obligation_seed_v151`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v151 only selects and fingerprints child `0` as a pending
  obligation inherited from v150. It does not prove a local flowpipe for that
  child, validate that child, discharge the remaining seven children, prove a
  global cover, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 local-flowpipe preflight v152

`lorenz_i256_cover_child0_local_flowpipe_preflight_check()` in
`stdlib/systems/lorenz_i256_cert.sio` attaches the v151 child-0 obligation seed
to the already checked five-step local-flowpipe-chain machinery. This is a
preflight ledger for the first child cell, not a local-flowpipe proof for that
cell.

- System anchors: the preflight consumes child-0 seed artifact/audit
  `577517715`/`49794723`, v151 portfolio entry/manifest/readiness
  `441058878`/`752370901`/`960341230`, v150 ledger artifact/audit
  `239912256`/`937619929`, v147 five-step local-flowpipe-chain artifact/audit
  `911209450`/`709377850`, v147 portfolio entry/manifest/readiness
  `657988743`/`867133305`/`31177321`, projection dependency DAG
  `713010204` with audit `725222703`, and projection certificate envelope
  `77888110` with audit `245178976`.
- Child shape: `child_index = 0`, slots `(0, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`, and
  `resolved_child_count = 0`.
- Preflight masks: `inherited_anchor_mask = 31`,
  `local_flowpipe_preflight_mask = 31`, `proof_dependency_mask = 31`,
  `available_local_chain_mask = 31`, and `pending_local_proof_mask = 31`.
- Non-claim masks: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `338034581`, certificate `537778915`, artifact
  `687983672`, audit `401304676`. The artifact kind is `23`, scoped to the
  child-0 local-flowpipe preflight.
- Portfolio v152 adds kind/checker `155`. The preflight receipt is
  `169673338`, audit is `493703035`, entry fingerprint is `396467356`,
  manifest is `592568164`, result coverage is `810972042`, family coverage is
  `409550044`, acceptance receipt is `311843645`, audit receipt is
  `249983152`, and readiness is `266425678`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_local_flowpipe_preflight`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_preflight_v152`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v152 only confirms that child `0` has the right inherited
  anchors and local-flowpipe machinery available for a future proof attempt. It
  does not prove the child flowpipe, validate that child, discharge any finite
  cover obligation, prove an invariant, prove shadowing, or assert a global
  Lorenz theorem.

## 2026-06-23 cover child-0 local-flowpipe proof skeleton v153

`lorenz_i256_cover_child0_local_flowpipe_proof_skeleton_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the v152 child-0 preflight to the
five existing Taylor-2 proof-trace skeleton receipts. This is a replay topology
for the first child cell, not a local-flowpipe proof for that cell.

- System anchors: the proof skeleton consumes the v152 preflight artifact/audit
  `687983672`/`401304676`, v152 portfolio entry/manifest/readiness
  `396467356`/`592568164`/`266425678`, v151 child seed artifact/audit
  `577517715`/`49794723`, the v147 five-step local-flowpipe-chain artifact/audit
  `911209450`/`709377850`, and the five step proof-trace skeleton artifact/audit
  pairs `174108453`/`516721287`, `976546207`/`892326627`,
  `308150621`/`281860333`, `556038402`/`509176612`, and
  `971755585`/`205079086`.
- Child shape: `child_index = 0`, slots `(0, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`,
  `resolved_child_count = 0`, `skeleton_node_count = 5`, and
  `skeleton_edge_count = 4`.
- Skeleton masks: `child_proof_skeleton_mask = 31`,
  `step_skeleton_dependency_mask = 31`, `skeleton_topology_mask = 31`, and
  `pending_local_proof_mask = 31`.
- Non-claim masks: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `999669477`, certificate `189977756`, artifact
  `204218974`, audit `455037243`. The artifact kind is `24`, scoped to the
  child-0 local-flowpipe proof skeleton.
- Portfolio v153 adds kind/checker `156`. The proof-skeleton receipt is
  `196370208`, audit is `179567267`, entry fingerprint is `764545251`,
  manifest is `754217778`, result coverage is `935640556`, family coverage is
  `905645729`, acceptance receipt is `550055298`, audit receipt is
  `538497989`, and readiness is `718133122`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_local_flowpipe_proof_skeleton`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_proof_skeleton_v153`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v153 only records the child-0 replay/proof topology and step
  skeleton dependencies. It does not prove the child flowpipe, validate child
  `0`, discharge any finite-cover obligation, prove an invariant, prove
  shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 local-flowpipe replay executor v154

`lorenz_i256_cover_child0_local_flowpipe_replay_executor_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the v153 child-0 proof skeleton to
the five existing Taylor-2 replay-executor receipts. This is the first child-0
replay executor receipt; it is still not a containment proof, not a validated
child, and not a finite-cover discharge.

- System anchors: the replay executor consumes the v153 proof-skeleton
  artifact/audit `204218974`/`455037243`, v153 portfolio
  entry/manifest/readiness `764545251`/`754217778`/`718133122`, the v152
  preflight artifact/audit `687983672`/`401304676`, and the five replay
  executor artifact/audit pairs `815221512`/`981617998`,
  `424013491`/`409402480`, `803328296`/`630199225`,
  `563122161`/`308331037`, and `225293164`/`897523976`.
- Portfolio step anchors: v154 also records the step replay-executor
  receipt/audit pairs from v137, v122, v107, v92, and v77:
  `907464772`/`901239486`, `899233777`/`518276528`,
  `420916049`/`731007955`, `143267578`/`89119858`, and
  `362923525`/`495796683`.
- Child shape: `child_index = 0`, slots `(0, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`,
  `resolved_child_count = 0`, `replay_node_count = 5`, and
  `replay_edge_count = 4`.
- Replay masks: `replay_executor_mask = 31`,
  `step_replay_dependency_mask = 31`, `replay_topology_mask = 31`, and
  `pending_containment_obligation_mask = 31`.
- Non-claim masks: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `533801936`, certificate `512361703`, artifact
  `979954189`, audit `42062960`. The artifact kind is `25`, scoped to the
  child-0 local-flowpipe replay executor.
- Portfolio v154 adds kind/checker `157`. The replay-executor receipt is
  `605739108`, audit is `19793671`, entry fingerprint is `85478916`, manifest
  is `776180561`, result coverage is `920622239`, family coverage is
  `262054576`, acceptance receipt is `235150015`, audit receipt is
  `997585839`, and readiness is `344176956`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_local_flowpipe_replay_executor`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_local_flowpipe_replay_executor_v154`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v154 records that child `0` can point at the existing five
  step replay executors through the v153 topology. It does not prove
  containment for the child cell, validate child `0`, discharge the finite
  cover, prove an invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 containment obligation v155

`lorenz_i256_cover_child0_containment_obligation_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the v154 child-0 replay executor to
the five existing step-local containment-obligation receipts. This is a child-0
containment obligation/candidate guard, not a validated child containment proof.

- System anchors: the child containment obligation consumes the v154 replay
  executor artifact/audit `979954189`/`42062960`, v154 portfolio
  entry/manifest/readiness `85478916`/`776180561`/`344176956`, the v153
  proof-skeleton artifact/audit `204218974`/`455037243`, and five step-local
  containment-obligation artifact/audit pairs `473585568`/`790380830`,
  `311696236`/`301485983`, `360827572`/`456885748`,
  `37368367`/`94030837`, and `291436394`/`876674722`.
- Portfolio step anchors: v155 records the step containment-obligation
  receipt/audit pairs from v139, v124, v109, v94, and v79:
  `26998442`/`558635697`, `7118356`/`340320797`,
  `280440446`/`300531949`, `367295300`/`113336557`, and
  `590242925`/`820968143`.
- Child shape: `child_index = 0`, slots `(0, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`,
  `resolved_child_count = 0`, `containment_node_count = 5`, and
  `containment_edge_count = 4`.
- Containment masks: `child_replay_executor_mask = 31`,
  `step_containment_dependency_mask = 31`, `containment_obligation_mask = 31`,
  `child_containment_candidate_mask = 31`, and
  `pending_child_validation_mask = 31`.
- Non-claim masks: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `847089510`, certificate `286298325`, artifact
  `556464464`, audit `754852588`. The artifact kind is `26`, scoped to the
  child-0 containment obligation.
- Portfolio v155 adds kind/checker `158`. The containment-obligation receipt is
  `901717119`, audit is `718305675`, entry fingerprint is `216539434`,
  manifest is `65996538`, result coverage is `173457116`, family coverage is
  `886316631`, acceptance receipt is `110948779`, audit receipt is
  `307718013`, and readiness is `376386864`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_containment_obligation`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_containment_obligation_v155`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v155 records that child `0` has a replay-executor-backed
  containment obligation whose step-local components are all available. It does
  not validate child `0`, prove the child cell containment theorem, discharge
  any finite cover obligation, prove an invariant, prove shadowing, or assert a
  global Lorenz theorem.

## 2026-06-23 cover child-0 validation guard v156

`lorenz_i256_cover_child0_validation_guard_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the v155 child-0 containment
obligation to the v154 replay executor, v151 child seed, and v147 five-step
local-flowpipe chain. This is a child-0 validation guard: it records that the
inputs needed for a future validation/discharge step are available, but it does
not validate or discharge child `0`.

- System anchors: the validation guard consumes v155 containment artifact/audit
  `556464464`/`754852588`, v155 portfolio entry/manifest/readiness
  `216539434`/`65996538`/`376386864`, v154 replay-executor artifact/audit
  `979954189`/`42062960`, v154 portfolio entry/manifest/readiness
  `85478916`/`776180561`/`344176956`, v151 child-0 seed artifact/audit
  `577517715`/`49794723`, and v147 five-step local-flowpipe-chain artifact/audit
  `911209450`/`709377850`.
- Portfolio step anchors: v156 records the v155 containment receipt/audit
  `901717119`/`718305675`, v154 replay receipt/audit
  `605739108`/`19793671`, v151 child-seed receipt/audit
  `402674780`/`117291494`, and v147 chain receipt/audit
  `760101623`/`786231486`.
- Child shape: `child_index = 0`, slots `(0, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`,
  `resolved_child_count = 0`, `validation_node_count = 5`, and
  `validation_edge_count = 4`.
- Validation-guard masks: `validation_anchor_mask = 31`,
  `containment_obligation_dependency_mask = 31`,
  `local_chain_dependency_mask = 31`, `child_validation_guard_mask = 31`, and
  `pending_child_discharge_mask = 31`.
- Non-claim masks: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `270532878`, certificate `219973249`, artifact
  `122525958`, audit `667114714`. The artifact kind is `27`, scoped to the
  child-0 validation guard.
- Portfolio v156 adds kind/checker `159`. The validation-guard receipt is
  `668638514`, audit is `897511359`, entry fingerprint is `906063860`,
  manifest is `397511764`, result coverage is `467991242`, family coverage is
  `552277921`, acceptance receipt is `18414285`, audit receipt is `253759540`,
  and readiness is `812993317`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh cover_child0_validation_guard`,
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_validation_guard_v156`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v156 records that child `0` now has the seed, replay,
  containment-obligation, local-chain, and portfolio anchors needed for a future
  validation/discharge attempt. It does not validate child `0`, discharge any
  child obligation, certify a finite cover, prove an invariant, prove shadowing,
  or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 x-axis validation witness v157

`lorenz_i256_cover_child0_x_axis_validation_witness_check()` in
`stdlib/systems/lorenz_i256_cert.sio` adds the first axis-specific validation
witness surface for child `0`. It consumes the v156 validation guard, v155
containment obligation, and five step-local containment-obligation receipts, but
keeps the axis and child validation masks at zero. This is a witness checker
shape for the X axis, not an X-axis containment theorem.

- System anchors: the X-axis witness consumes v156 validation-guard artifact/audit
  `122525958`/`667114714`, v156 portfolio entry/manifest/readiness
  `906063860`/`397511764`/`812993317`, v155 containment artifact/audit
  `556464464`/`754852588`, v155 portfolio entry/manifest/readiness
  `216539434`/`65996538`/`376386864`, and five step-local
  containment-obligation artifact/audit pairs `473585568`/`790380830`,
  `311696236`/`301485983`, `360827572`/`456885748`,
  `37368367`/`94030837`, and `291436394`/`876674722`.
- Portfolio step anchors: v157 records the v156 validation-guard receipt/audit
  `668638514`/`897511359`, v155 containment receipt/audit
  `901717119`/`718305675`, and the step containment receipt/audit pairs from
  v139, v124, v109, v94, and v79: `26998442`/`558635697`,
  `7118356`/`340320797`, `280440446`/`300531949`,
  `367295300`/`113336557`, and `590242925`/`820968143`.
- Child and axis shape: `child_index = 0`, `axis_index = 0`, slots `(0, 0, 0)`,
  `child_cell_count = 8`, `selected_child_count = 1`,
  `pending_child_obligation_count = 8`, `resolved_child_count = 0`,
  `witness_node_count = 5`, and `witness_edge_count = 4`.
- Witness masks: `axis_witness_mask = 1`,
  `validation_guard_dependency_mask = 31`,
  `containment_obligation_dependency_mask = 31`,
  `step_axis_dependency_mask = 31`,
  `x_axis_witness_candidate_mask = 31`, and
  `pending_axis_validation_mask = 31`.
- Non-claim masks: `axis_validated_mask = 0`,
  `local_flowpipe_proof_mask = 0`, `child_validated_mask = 0`,
  `child_discharge_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `264482340`, certificate `233084840`, artifact
  `537883792`, audit `111967815`. The artifact kind is `28`, scoped to the
  child-0 X-axis validation witness.
- Portfolio v157 adds kind/checker `160`. The X-axis witness receipt is
  `23819133`, audit is `275376963`, entry fingerprint is `924103496`,
  manifest is `204190141`, result coverage is `237688519`, family coverage is
  `693402369`, acceptance receipt is `772999009`, audit receipt is `215241306`,
  and readiness is `710437593`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_x_axis_validation_witness`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_x_axis_validation_witness_v157`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v157 records an explicit X-axis witness-checker interface and
  dependency ledger for child `0`. It does not validate the X axis, validate
  child `0`, discharge any child obligation, certify a finite cover, prove an
  invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 y-axis validation witness v158

`lorenz_i256_cover_child0_y_axis_validation_witness_check()` in
`stdlib/systems/lorenz_i256_cert.sio` adds the Y-axis companion to the v157
X-axis witness surface for child `0`. It consumes the v156 validation guard, the
v155 containment obligation, the five step-local containment-obligation
receipts, and records v157 as the preceding axis-witness portfolio layer. This
is still a witness checker shape, not a Y-axis containment theorem.

- System anchors: the Y-axis witness consumes v156 validation-guard artifact/audit
  `122525958`/`667114714`, v156 portfolio entry/manifest/readiness
  `906063860`/`397511764`/`812993317`, v155 containment artifact/audit
  `556464464`/`754852588`, v155 portfolio entry/manifest/readiness
  `216539434`/`65996538`/`376386864`, and five step-local
  containment-obligation artifact/audit pairs `473585568`/`790380830`,
  `311696236`/`301485983`, `360827572`/`456885748`,
  `37368367`/`94030837`, and `291436394`/`876674722`.
- Portfolio step anchors: v158 records the v157 X-axis witness receipt/audit
  `23819133`/`275376963`, v157 portfolio entry/manifest/readiness
  `924103496`/`204190141`/`710437593`, v156 validation-guard receipt/audit
  `668638514`/`897511359`, and v155 containment receipt/audit
  `901717119`/`718305675`.
- Child and axis shape: `child_index = 0`, `axis_index = 1`, slots `(0, 0, 0)`,
  `child_cell_count = 8`, `selected_child_count = 1`,
  `pending_child_obligation_count = 8`, `resolved_child_count = 0`,
  `witness_node_count = 5`, and `witness_edge_count = 4`.
- Witness masks: `axis_witness_mask = 2`,
  `validation_guard_dependency_mask = 31`,
  `containment_obligation_dependency_mask = 31`,
  `step_axis_dependency_mask = 31`,
  `y_axis_witness_candidate_mask = 31`, and
  `pending_axis_validation_mask = 31`.
- Non-claim masks: `axis_validated_mask = 0`,
  `local_flowpipe_proof_mask = 0`, `child_validated_mask = 0`,
  `child_discharge_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `433988856`, certificate `967497116`, artifact
  `621968749`, audit `432340743`. The artifact kind is `29`, scoped to the
  child-0 Y-axis validation witness.
- Portfolio v158 adds kind/checker `161`. The Y-axis witness receipt is
  `215439979`, audit is `926858315`, entry fingerprint is `784178707`,
  manifest is `976587883`, result coverage is `973105161`, family coverage is
  `800246175`, acceptance receipt is `946777722`, audit receipt is `472196821`,
  and readiness is `853877827`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_y_axis_validation_witness`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_y_axis_validation_witness_v158`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v158 records an explicit Y-axis witness-checker interface and
  dependency ledger for child `0`. It does not validate the Y axis, validate
  child `0`, discharge any child obligation, certify a finite cover, prove an
  invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 z-axis validation witness v159

`lorenz_i256_cover_child0_z_axis_validation_witness_check()` in
`stdlib/systems/lorenz_i256_cert.sio` completes the coordinate-wise witness
surface for child `0` by adding the Z-axis companion to the existing X/Y witness
interfaces. It consumes the v156 validation guard, the v155 containment
obligation, the five step-local containment-obligation receipts, and records
v158 as the preceding Y-axis witness portfolio layer. This is still a witness
checker shape, not a Z-axis containment theorem.

- System anchors: the Z-axis witness consumes v156 validation-guard artifact/audit
  `122525958`/`667114714`, v156 portfolio entry/manifest/readiness
  `906063860`/`397511764`/`812993317`, v155 containment artifact/audit
  `556464464`/`754852588`, v155 portfolio entry/manifest/readiness
  `216539434`/`65996538`/`376386864`, and five step-local
  containment-obligation artifact/audit pairs `473585568`/`790380830`,
  `311696236`/`301485983`, `360827572`/`456885748`,
  `37368367`/`94030837`, and `291436394`/`876674722`.
- Portfolio step anchors: v159 records the v158 Y-axis witness receipt/audit
  `215439979`/`926858315`, v158 portfolio entry/manifest/readiness
  `784178707`/`976587883`/`853877827`, v156 validation-guard receipt/audit
  `668638514`/`897511359`, and v155 containment receipt/audit
  `901717119`/`718305675`.
- Child and axis shape: `child_index = 0`, `axis_index = 2`, slots `(0, 0, 0)`,
  `child_cell_count = 8`, `selected_child_count = 1`,
  `pending_child_obligation_count = 8`, `resolved_child_count = 0`,
  `witness_node_count = 5`, and `witness_edge_count = 4`.
- Witness masks: `axis_witness_mask = 4`,
  `validation_guard_dependency_mask = 31`,
  `containment_obligation_dependency_mask = 31`,
  `step_axis_dependency_mask = 31`,
  `z_axis_witness_candidate_mask = 31`, and
  `pending_axis_validation_mask = 31`.
- Non-claim masks: `axis_validated_mask = 0`,
  `local_flowpipe_proof_mask = 0`, `child_validated_mask = 0`,
  `child_discharge_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `642457820`, certificate `125119970`, artifact
  `725197597`, audit `434796366`. The artifact kind is `30`, scoped to the
  child-0 Z-axis validation witness.
- Portfolio v159 adds kind/checker `162`. The Z-axis witness receipt is
  `397206580`, audit is `940494663`, entry fingerprint is `825417819`,
  manifest is `659766660`, result coverage is `619302838`, family coverage is
  `817871023`, acceptance receipt is `356914642`, audit receipt is `989438473`,
  and readiness is `864603127`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_z_axis_validation_witness`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_z_axis_validation_witness_v159`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v159 records an explicit Z-axis witness-checker interface and
  dependency ledger for child `0`. It does not validate the Z axis, validate
  child `0`, discharge any child obligation, certify a finite cover, prove an
  invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 axis witness bundle v160

`lorenz_i256_cover_child0_axis_witness_bundle_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the three child-0 axis witness
surfaces into one X/Y/Z bundle. It consumes the v157 X-axis witness, v158 Y-axis
witness, v159 Z-axis witness, and their corresponding portfolio readiness
anchors. This is a bundle/dependency receipt, not an axis-validation theorem.

- System anchors: the bundle consumes X-axis witness artifact/audit
  `537883792`/`111967815`, v157 portfolio entry/manifest/readiness
  `924103496`/`204190141`/`710437593`, Y-axis witness artifact/audit
  `621968749`/`432340743`, v158 portfolio entry/manifest/readiness
  `784178707`/`976587883`/`853877827`, Z-axis witness artifact/audit
  `725197597`/`434796366`, and v159 portfolio entry/manifest/readiness
  `825417819`/`659766660`/`864603127`.
- Bundle shape: `child_index = 0`, `child_cell_count = 8`,
  `axis_count = 3`, `witness_node_count = 15`, and
  `witness_edge_count = 12`.
- Witness masks: `axis_witness_mask = 7`,
  `axis_artifact_dependency_mask = 7`, `portfolio_dependency_mask = 7`,
  `bundle_dependency_mask = 7`, and `pending_axis_validation_mask = 31`.
- Non-claim masks: `axis_validated_mask = 0`,
  `local_flowpipe_proof_mask = 0`, `child_validated_mask = 0`,
  `child_discharge_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `879555052`, certificate `801313075`, artifact
  `33958008`, audit `326688223`. The artifact kind is `31`, scoped to the
  child-0 X/Y/Z axis witness bundle.
- Portfolio v160 adds kind/checker `163`. The bundle receipt is `235058081`,
  audit is `38080034`, entry fingerprint is `467972897`, manifest is
  `539193234`, result coverage is `461748312`, family coverage is `31743661`,
  acceptance receipt is `707906712`, audit receipt is `444463345`, and
  readiness is `507615110`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_axis_witness_bundle`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_axis_witness_bundle_v160`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v160 records a complete X/Y/Z witness bundle for child `0`.
  It does not validate any axis, validate child `0`, discharge any child
  obligation, certify a finite cover, prove an invariant, prove shadowing, or
  assert a global Lorenz theorem.

## 2026-06-23 cover child-0 X-axis arithmetic validator v161

`lorenz_i256_cover_child0_x_axis_arithmetic_validator_check()` in
`stdlib/systems/lorenz_i256_cert.sio` is the first real arithmetic validator
over the child-0 axis-witness bundle. It consumes the v160 X/Y/Z witness bundle,
the v157 X-axis witness, and the five existing child-0 containment-obligation
receipts, then checks explicit X-axis `need <= margin` inequalities and slack
values for the five local steps.

- System anchors: the validator consumes the v160 axis bundle artifact/audit
  `33958008`/`326688223`, v160 portfolio receipt/audit
  `235058081`/`38080034`, v160 entry/manifest/readiness
  `467972897`/`539193234`/`507615110`, X-axis witness artifact/audit
  `537883792`/`111967815`, and v157 entry/manifest/readiness
  `924103496`/`204190141`/`710437593`.
- Containment anchors: step 1 artifact/audit `473585568`/`790380830`, step 2
  `311696236`/`301485983`, step 3 `360827572`/`456885748`, step 4
  `37368367`/`94030837`, and step 5 `291436394`/`876674722`.
- X-axis arithmetic tuple at `target_scale = 1000000`,
  `source_width_bits = 256`, `scale_log2 = 32`: step 1
  need/margin/slack `26878`/`27000`/`122`, step 2
  `83731`/`84000`/`269`, step 3 `112768`/`113000`/`232`, step 4
  `103524`/`104000`/`476`, and step 5 `278282`/`279000`/`718`.
- Arithmetic masks: `x_axis_pass_mask = 31`, `x_axis_fail_mask = 0`,
  `x_axis_slack_mask = 31`, `min_x_slack_ppm = 122`, and
  `total_x_need_ppm = 605183`.
- Promotion masks: `axis_validated_mask = 1` records only the X-axis arithmetic
  validator; `pending_axis_validation_mask = 30` and
  `remaining_axis_pending_mask = 6` keep Y/Z validation pending.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `503013755`, certificate `118280524`, artifact
  `260549564`, audit `90270943`. The artifact kind is `32`, scoped to
  child-0 X-axis arithmetic validation.
- Portfolio v161 adds kind/checker `164`. The validator receipt is
  `421391411`, audit is `195981576`, entry fingerprint is `426864254`,
  manifest is `343727158`, result coverage is `229301136`, family coverage is
  `170723656`, acceptance receipt is `645201422`, audit receipt is
  `755007029`, and readiness is `580739550`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v161 test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_x_axis_arithmetic_validator`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_x_axis_arithmetic_validator_v161`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v161 validates only the X-axis arithmetic over five already
  recorded local containment obligations for child `0`. It does not validate
  Y/Z, validate child `0`, discharge any child obligation, certify a finite
  cover, prove an invariant, prove shadowing, or assert a global Lorenz
  theorem.

## 2026-06-23 cover child-0 Y-axis arithmetic validator v162

`lorenz_i256_cover_child0_y_axis_arithmetic_validator_check()` in
`stdlib/systems/lorenz_i256_cert.sio` repeats the v161 arithmetic-validator
shape for the Y axis. It consumes the v160 axis-witness bundle, the v161
X-axis arithmetic validator readiness, the v158 Y-axis witness, and the same
five child-0 local containment obligations, then checks explicit Y-axis
`need <= margin` inequalities and slack values.

- System anchors: the validator consumes the v160 axis bundle artifact/audit
  `33958008`/`326688223`, v160 receipt/audit/entry/manifest/readiness
  `235058081`/`38080034`/`467972897`/`539193234`/`507615110`, v161
  receipt/audit/entry/manifest/readiness
  `421391411`/`195981576`/`426864254`/`343727158`/`580739550`, Y-axis witness
  artifact/audit `621968749`/`432340743`, and v158
  receipt/audit/entry/manifest/readiness
  `215439979`/`926858315`/`784178707`/`976587883`/`853877827`.
- Containment anchors: step 1 artifact/audit `473585568`/`790380830`, step 2
  `311696236`/`301485983`, step 3 `360827572`/`456885748`, step 4
  `37368367`/`94030837`, and step 5 `291436394`/`876674722`.
- Y-axis arithmetic tuple at `target_scale = 1000000`,
  `source_width_bits = 256`, `scale_log2 = 32`: step 1
  need/margin/slack `295701`/`296000`/`299`, step 2
  `307457`/`308000`/`543`, step 3 `323361`/`323500`/`139`, step 4
  `318703`/`319000`/`297`, and step 5 `369718`/`370000`/`282`.
- Arithmetic masks: `y_axis_pass_mask = 31`, `y_axis_fail_mask = 0`,
  `y_axis_slack_mask = 31`, `min_y_slack_ppm = 139`, and
  `total_y_need_ppm = 1614940`.
- Promotion masks: `axis_validated_mask = 3` records only the X/Y arithmetic
  validators; `pending_axis_validation_mask = 28` and
  `remaining_axis_pending_mask = 4` keep Z validation pending.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `278656756`, certificate `11270473`, artifact
  `99615754`, audit `105297009`. The artifact kind is `33`, scoped to child-0
  Y-axis arithmetic validation.
- Portfolio v162 adds kind/checker `165`. The validator receipt is
  `646649692`, audit is `380218523`, entry fingerprint is `487433865`,
  manifest is `477732316`, result coverage is `326325194`, family coverage is
  `639174885`, acceptance receipt is `708785011`, audit receipt is
  `138315621`, and readiness is `738520536`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v162 test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_y_axis_arithmetic_validator`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_y_axis_arithmetic_validator_v162`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v162 validates only Y-axis arithmetic in addition to the
  already recorded X-axis arithmetic validator. It does not validate Z,
  validate child `0`, discharge any child obligation, certify a finite cover,
  prove an invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 Z-axis arithmetic validator v163

`lorenz_i256_cover_child0_z_axis_arithmetic_validator_check()` in
`stdlib/systems/lorenz_i256_cert.sio` completes the X/Y/Z arithmetic-validator
surface for child `0`. It consumes the v160 axis-witness bundle, the v162
Y-axis arithmetic validator readiness, the v159 Z-axis witness, and the same
five child-0 local containment obligations, then checks explicit Z-axis
`need <= margin` inequalities and slack values.

- System anchors: the validator consumes the v160 axis bundle artifact/audit
  `33958008`/`326688223`, v160 receipt/audit/entry/manifest/readiness
  `235058081`/`38080034`/`467972897`/`539193234`/`507615110`, v162
  receipt/audit/entry/manifest/readiness
  `646649692`/`380218523`/`487433865`/`477732316`/`738520536`, Z-axis witness
  artifact/audit `725197597`/`434796366`, and v159
  receipt/audit/entry/manifest/readiness
  `397206580`/`940494663`/`825417819`/`659766660`/`864603127`.
- Containment anchors: step 1 artifact/audit `473585568`/`790380830`, step 2
  `311696236`/`301485983`, step 3 `360827572`/`456885748`, step 4
  `37368367`/`94030837`, and step 5 `291436394`/`876674722`.
- Z-axis arithmetic tuple at `target_scale = 1000000`,
  `source_width_bits = 256`, `scale_log2 = 32`: step 1
  need/margin/slack `20569`/`21000`/`431`, step 2
  `21872`/`22000`/`128`, step 3 `20325`/`20500`/`175`, step 4
  `12336`/`12500`/`164`, and step 5 `49576`/`50000`/`424`.
- Arithmetic masks: `z_axis_pass_mask = 31`, `z_axis_fail_mask = 0`,
  `z_axis_slack_mask = 31`, `min_z_slack_ppm = 128`, and
  `total_z_need_ppm = 124678`.
- Promotion masks: `axis_validated_mask = 7` records the X/Y/Z arithmetic
  validators; `pending_axis_validation_mask = 24` keeps the broader
  validation/promotion work pending, while `remaining_axis_pending_mask = 0`
  records no axis arithmetic validator left for this child-0 slice.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `348416226`, certificate `157818710`,
  artifact `21090937`, audit `972902456`. The artifact kind is `34`, scoped
  to child-0 Z-axis arithmetic validation.
- Portfolio v163 adds kind/checker `166`. The validator receipt is
  `587874571`, audit is `56865068`, entry fingerprint is `627571015`,
  manifest is `569710473`, result coverage is `381322251`, family coverage is
  `65599106`, acceptance receipt is `724786771`, audit receipt is
  `514472967`, and readiness is `891783529`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v163 test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_z_axis_arithmetic_validator`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_z_axis_arithmetic_validator_v163`,
  and `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v163 validates only Z-axis arithmetic in addition to the
  already recorded X/Y validators. It does not validate child `0`, discharge
  any child obligation, certify a finite cover, prove an invariant, prove
  shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 X/Y/Z axis arithmetic bundle v164

`lorenz_i256_cover_child0_axis_arithmetic_bundle_check()` in
`stdlib/systems/lorenz_i256_cert.sio` bundles the three child-0 axis arithmetic
validators into a single receipt. It consumes the v160 axis-witness bundle plus
the v161 X-axis, v162 Y-axis, and v163 Z-axis arithmetic validator artifacts
and portfolio readiness receipts. The bundle is intentionally only an
arithmetic-and-readiness surface; it does not discharge the child obligation.

- System anchors: v160 axis bundle artifact/audit `33958008`/`326688223`,
  v160 receipt/audit/entry/manifest/readiness
  `235058081`/`38080034`/`467972897`/`539193234`/`507615110`; X-axis validator
  artifact/audit `260549564`/`90270943`, v161
  receipt/audit/entry/manifest/readiness
  `421391411`/`195981576`/`426864254`/`343727158`/`580739550`; Y-axis
  validator artifact/audit `99615754`/`105297009`, v162
  receipt/audit/entry/manifest/readiness
  `646649692`/`380218523`/`487433865`/`477732316`/`738520536`; and Z-axis
  validator artifact/audit `21090937`/`972902456`, v163
  receipt/audit/entry/manifest/readiness
  `587874571`/`56865068`/`627571015`/`569710473`/`891783529`.
- Axis arithmetic summary: X total need/min slack `605183`/`122`, Y
  `1614940`/`139`, Z `124678`/`128`, aggregate total need `2344801`, and
  aggregate minimum slack `122`.
- Bundle masks: `axis_arithmetic_validator_mask = 7`,
  `axis_arithmetic_receipt_mask = 7`, `axis_arithmetic_audit_mask = 7`,
  `axis_arithmetic_bundle_mask = 7`, `axis_validated_mask = 7`,
  `pending_axis_validation_mask = 24`, `remaining_axis_pending_mask = 0`, and
  `pending_child_validation_mask = 31`.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `243947898`, certificate `366255612`,
  artifact `789069033`, audit `690586147`. The artifact kind is `35`, scoped
  to the child-0 X/Y/Z axis arithmetic bundle.
- Portfolio v164 adds kind/checker `167`. The bundle receipt is `549336244`,
  audit is `339078406`, entry fingerprint is `823908520`, manifest is
  `775511795`, result coverage is `550142473`, family coverage is `605846499`,
  acceptance receipt is `536988057`, audit receipt is `109146167`, and
  readiness is `99214103`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v164 test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_axis_arithmetic_bundle`, `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_axis_arithmetic_bundle_v164`, and
  `./scripts/run_sio_test_suite.sh portfolio`.
- Scope boundary: v164 records a bundle of X/Y/Z arithmetic validators over
  existing child-0 containment obligations. It still does not validate child
  `0`, discharge any child obligation, certify a finite cover, prove an
  invariant, prove shadowing, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 axis arithmetic readiness v165

`lorenz_i256_cover_child0_axis_arithmetic_readiness_check()` in
`stdlib/systems/lorenz_i256_cert.sio` connects the v164 X/Y/Z axis arithmetic
bundle to the earlier child-0 validation guard and containment-obligation
anchors. This is a readiness receipt for the next child-validation checker, not
the child validation itself.

- System anchors: child-0 containment artifact/audit `556464464`/`754852588`,
  v155 receipt/audit/entry/manifest/readiness
  `901717119`/`718305675`/`216539434`/`65996538`/`376386864`; child-0
  validation guard artifact/audit `122525958`/`667114714`, v156
  receipt/audit/entry/manifest/readiness
  `668638514`/`897511359`/`906063860`/`397511764`/`812993317`; and v164 axis
  arithmetic bundle artifact/audit `789069033`/`690586147`,
  receipt/audit/entry/manifest/readiness
  `549336244`/`339078406`/`823908520`/`775511795`/`99214103`.
- Arithmetic summary carried forward: aggregate total axis need `2344801` and
  aggregate minimum slack `122`.
- Readiness masks: `axis_arithmetic_ready_mask = 7`,
  `validation_guard_dependency_mask = 15`, `containment_dependency_mask = 15`,
  `readiness_dependency_mask = 31`, `pending_child_validation_mask = 31`, and
  `child_axis_readiness_mask = 0`.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `98645448`, certificate `770760867`, artifact
  `464483154`, audit `56112870`. The artifact kind is `36`, scoped to the
  child-0 axis arithmetic readiness receipt.
- Portfolio v165 adds kind/checker `168`. The readiness receipt is
  `291275395`, audit is `323522319`, entry fingerprint is `771918406`,
  manifest is `507007122`, result coverage is `244656700`, family coverage is
  `671787897`, acceptance receipt is `657061525`, audit receipt is
  `338343968`, and readiness is `868353620`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v165 test-file
  checks, `./scripts/run_sio_test_suite.sh
  cover_child0_axis_arithmetic_readiness`, and `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_axis_arithmetic_readiness_v165`.
- Scope boundary: v165 states only that the child-0 guard, containment anchors,
  and X/Y/Z axis arithmetic bundle are all available for a later child
  validation checker. It still does not validate child `0`, discharge any child
  obligation, certify a finite cover, prove an invariant, prove shadowing, or
  assert a global Lorenz theorem.

## 2026-06-23 cover child-0 local validation core v166

`lorenz_i256_cover_child0_validation_core_check()` in
`stdlib/systems/lorenz_i256_cert.sio` is the first child-0 local-validation
receipt after the v165 readiness gate. It binds the readiness receipt back to
the v156 validation guard, v155 containment obligation, and v164 axis
arithmetic bundle, then marks only child `0` as locally validated inside this
finite-cover candidate lane.

- System anchors: v165 axis-arithmetic readiness artifact/audit
  `464483154`/`56112870`, receipt/audit/entry/manifest/readiness
  `291275395`/`323522319`/`771918406`/`507007122`/`868353620`; v156
  validation guard artifact/audit `122525958`/`667114714`,
  receipt/audit/entry/manifest/readiness
  `668638514`/`897511359`/`906063860`/`397511764`/`812993317`; v155
  containment artifact/audit `556464464`/`754852588`,
  receipt/audit/entry/manifest/readiness
  `901717119`/`718305675`/`216539434`/`65996538`/`376386864`; and v164
  axis-arithmetic bundle artifact/audit `789069033`/`690586147`,
  receipt/audit/entry/manifest/readiness
  `549336244`/`339078406`/`823908520`/`775511795`/`99214103`.
- Carried arithmetic summary: aggregate total axis need `2344801` and
  aggregate minimum slack `122`.
- Validation masks: `axis_arithmetic_ready_mask = 7`,
  `axis_readiness_dependency_mask = 15`,
  `validation_guard_dependency_mask = 15`, `containment_dependency_mask = 15`,
  `axis_bundle_dependency_mask = 15`, `child_validation_dependency_mask = 63`,
  `pending_child_validation_mask = 30`, `child_axis_readiness_mask = 7`,
  `local_flowpipe_proof_mask = 31`, and `child_validated_mask = 1`.
- Non-discharge/global masks stay closed: `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `107508389`, certificate `167688818`,
  artifact `375395561`, audit `939132912`. The artifact kind is `37`, scoped
  to the child-0 local validation core.
- Portfolio v166 adds kind/checker `169`. The validation receipt is
  `611594783`, audit is `483258342`, entry fingerprint is `332103319`,
  manifest is `53041691`, result coverage is `753710176`, family coverage is
  `552268537`, acceptance receipt is `676280163`, audit receipt is
  `323017615`, and readiness is `329577381`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v166 test-file
  checks, `./bin/souc run` for the two v166 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child0_validation_core`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_validation_core_v166`.
- Scope boundary: v166 validates only child `0` inside the local receipt chain.
  It still does not discharge child `0`, validate the remaining children,
  certify a finite cover, prove an invariant, prove shadowing, assert
  unbounded-time behavior, or assert a global Lorenz theorem.

## 2026-06-23 cover child-0 discharge preflight v167

`lorenz_i256_cover_child0_discharge_preflight_check()` in
`stdlib/systems/lorenz_i256_cert.sio` turns the v166 child-0 local validation
into an explicit non-discharge ledger. It records that child `0` has a local
validation receipt, then enumerates the remaining cover-level blockers that
prevent any child discharge or finite-cover certificate claim.

- System anchors: v166 child-0 validation-core artifact/audit
  `375395561`/`939132912`, receipt/audit/entry/manifest/readiness
  `611594783`/`483258342`/`332103319`/`53041691`/`329577381`; v151 child-0
  obligation seed artifact/audit `577517715`/`49794723`,
  receipt/audit/entry/manifest/readiness
  `402674780`/`117291494`/`441058878`/`752370901`/`960341230`; v150 cover
  refinement ledger artifact/audit `239912256`/`937619929`,
  receipt/audit/entry/manifest/readiness
  `893139927`/`132235218`/`974395903`/`568368767`/`228495843`; and v149
  finite-cover candidate artifact/audit `911169246`/`785916873`,
  receipt/audit/entry/manifest/readiness
  `41883869`/`324028289`/`820916947`/`527721765`/`437511874`.
- Preflight masks: `child_validated_mask = 1`,
  `pending_child_validation_mask = 30`, `remaining_child_validation_mask = 30`,
  `child_validation_dependency_mask = 63`,
  `child0_validation_receipt_mask = 31`,
  `child0_discharge_preflight_mask = 1`, `discharge_blocker_mask = 15`,
  `missing_child_validation_mask = 30`, and
  `pending_child_discharge_mask = 255`.
- Non-claim masks stay closed: `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, `global_flowpipe_claim_mask = 0`, and
  `finite_cover_certificate_mask = 0`.
- System fingerprints: instance `241406543`, certificate `630674365`,
  artifact `367693213`, audit `255409323`. The artifact kind is `38`, scoped
  to the child-0 discharge preflight, not the discharge itself.
- Portfolio v167 adds kind/checker `170`. The preflight receipt is
  `411576874`, audit is `995638697`, entry fingerprint is `839232536`,
  manifest is `971077767`, result coverage is `634765145`, family coverage is
  `804750677`, acceptance receipt is `458239701`, audit receipt is
  `253329063`, and readiness is `479999245`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v167 test-file
  checks, `./bin/souc run` for the two v167 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child0_discharge_preflight`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child0_discharge_preflight_v167`.
- Scope boundary: v167 only records blockers before child-0 discharge. It does
  not discharge child `0`, validate remaining children, discharge any child,
  certify a finite cover, prove an invariant, prove shadowing, assert
  unbounded-time behavior, or assert a global Lorenz theorem.

## 2026-06-23 cover child-1 obligation seed v168

`lorenz_i256_cover_child1_obligation_seed_check()` in
`stdlib/systems/lorenz_i256_cert.sio` starts the sibling-child ladder after the
v167 child-0 discharge preflight. It selects child `1`, with slots `(1, 0, 0)`,
as a replayable pending obligation while preserving the non-claim boundary.

- System anchors: v167 child-0 discharge-preflight artifact/audit
  `367693213`/`255409323`, receipt/audit/entry/manifest/readiness
  `411576874`/`995638697`/`839232536`/`971077767`/`479999245`; v150 cover
  refinement ledger artifact/audit `239912256`/`937619929`,
  receipt/audit/entry/manifest/readiness
  `893139927`/`132235218`/`974395903`/`568368767`/`228495843`; v149
  finite-cover candidate artifact/audit `911169246`/`785916873`,
  receipt/audit/entry/manifest/readiness
  `41883869`/`324028289`/`820916947`/`527721765`/`437511874`; projection
  dependency DAG `713010204` with audit `725222703`; and projection
  certificate envelope `77888110` with audit `245178976`.
- Child shape: `child_index = 1`, slots `(1, 0, 0)`,
  `split_axis_count = 3`, `split_factor_per_axis = 2`,
  `child_cell_count = 8`, `selected_child_count = 1`,
  `pending_child_obligation_count = 8`, and `resolved_child_count = 0`.
- Masks: `selected_child_mask = 2`, `child_coordinate_mask = 7`,
  `inherited_anchor_mask = 15`, `prior_child_validated_mask = 1`,
  `pending_child_validation_mask = 30`, `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `7187617`, certificate `361973575`, artifact
  `277189357`, audit `29837118`. The artifact kind is `39`, scoped to the
  child-1 obligation seed.
- Portfolio v168 adds kind/checker `171`. The child-1 seed receipt is
  `455287687`, audit is `365381856`, entry fingerprint is `754167191`,
  manifest is `785481354`, result coverage is `412187632`, family coverage is
  `953600335`, acceptance receipt is `849717348`, audit receipt is
  `769183626`, and readiness is `438739269`.
- Counter checks: result coverage stays explicit as `10 + 160 + 1 + 1 = 172`;
  family coverage stays explicit as `6 + 4 + 3 + 2 + 155 + 2 = 172`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v168 test-file
  checks, `./bin/souc run` for the two v168 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_obligation_seed`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_obligation_seed_v168`.
- Scope boundary: v168 only selects and fingerprints child `1` as a pending
  sibling obligation. It does not prove a local flowpipe for child `1`,
  validate child `1`, discharge any child, certify a finite cover, prove an
  invariant, prove shadowing, assert unbounded-time behavior, or assert a
  global Lorenz theorem.

## 2026-06-23 cover child-1 local-flowpipe preflight v169

`lorenz_i256_cover_child1_local_flowpipe_preflight_check()` in
`stdlib/systems/lorenz_i256_cert.sio` attaches the v168 child-1 obligation seed
to the already checked five-step local-flowpipe-chain machinery. This is a
preflight receipt for child `1`, not a local-flowpipe proof, validation receipt,
or discharge receipt.

- System anchors: v168 child-1 obligation seed artifact/audit
  `277189357`/`29837118`, v168 portfolio entry/manifest/readiness
  `754167191`/`785481354`/`438739269`, v150 cover-refinement ledger
  artifact/audit `239912256`/`937619929`, v147 five-step local-flowpipe-chain
  artifact/audit `911209450`/`709377850`, v147 portfolio
  entry/manifest/readiness `657988743`/`867133305`/`31177321`, projection
  dependency DAG `713010204` with audit `725222703`, and projection
  certificate envelope `77888110` with audit `245178976`.
- Child shape: `child_index = 1`, slots `(1, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`, and
  `resolved_child_count = 0`.
- Preflight masks: `inherited_anchor_mask = 31`,
  `local_flowpipe_preflight_mask = 31`, `proof_dependency_mask = 31`,
  `available_local_chain_mask = 31`, and `pending_local_proof_mask = 31`.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `431462556`, certificate `924554309`,
  artifact `105720752`, audit `750610782`. The artifact kind is `40`, scoped
  to the child-1 local-flowpipe preflight.
- Portfolio v169 adds kind/checker `172`. The preflight receipt is
  `931182967`, audit is `466306558`, entry fingerprint is `271368334`,
  manifest is `358260902`, result coverage is `947986087`, family coverage is
  `860825954`, acceptance receipt is `960364947`, audit receipt is
  `312066006`, and readiness is `954803939`.
- Counter checks: result coverage stays explicit as `10 + 161 + 1 + 1 = 173`;
  family coverage stays explicit as `6 + 4 + 3 + 2 + 156 + 2 = 173`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v169 test-file
  checks, `./bin/souc run` for the two v169 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_local_flowpipe_preflight`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_preflight_v169`.
- Scope boundary: v169 only records child-1 local-flowpipe preflight readiness.
  It does not add a child-1 proof skeleton, execute replay, prove local
  containment, validate child `1`, discharge any child, certify a finite cover,
  prove an invariant, prove shadowing, assert unbounded-time behavior, or
  assert a global Lorenz theorem.

## 2026-06-23 cover child-1 local-flowpipe proof skeleton v170

`lorenz_i256_cover_child1_local_flowpipe_proof_skeleton_check()` in
`stdlib/systems/lorenz_i256_cert.sio` binds the v169 child-1 preflight to the
five existing step proof-trace skeleton receipts. This orders the child-1
proof dependencies, but it still does not execute replay or prove local
containment.

- System anchors: child-1 local-flowpipe preflight artifact/audit
  `105720752`/`750610782`, v169 portfolio entry/manifest/readiness
  `271368334`/`358260902`/`954803939`, child-1 obligation seed artifact/audit
  `277189357`/`29837118`, five-step local-flowpipe-chain artifact/audit
  `911209450`/`709377850`, and step proof-trace skeleton artifact/audit pairs
  `174108453`/`516721287`, `976546207`/`892326627`,
  `308150621`/`281860333`, `556038402`/`509176612`, and
  `971755585`/`205079086`.
- Child shape: `child_index = 1`, slots `(1, 0, 0)`, `child_cell_count = 8`,
  `selected_child_count = 1`, `pending_child_obligation_count = 8`, and
  `resolved_child_count = 0`.
- Skeleton masks: `inherited_anchor_mask = 31`,
  `local_flowpipe_preflight_mask = 31`, `child_proof_skeleton_mask = 31`,
  `step_skeleton_dependency_mask = 31`, `skeleton_topology_mask = 31`, and
  `pending_local_proof_mask = 31`.
- Non-claim masks stay closed: `local_flowpipe_proof_mask = 0`,
  `child_validated_mask = 0`, `child_discharge_mask = 0`,
  `global_cover_certificate_mask = 0`, and `global_flowpipe_claim_mask = 0`.
- System fingerprints: instance `826591689`, certificate `433652313`,
  artifact `259900750`, audit `190272326`. The artifact kind is `41`, scoped
  to the child-1 local-flowpipe proof skeleton.
- Portfolio v170 adds kind/checker `173`. The skeleton receipt is `281245914`,
  audit is `490120064`, entry fingerprint is `850263288`, manifest is
  `234669390`, result coverage is `787413475`, family coverage is `71680506`,
  acceptance receipt is `11676142`, audit receipt is `422416917`, and readiness
  is `779299425`.
- Counter checks: result coverage stays explicit as `10 + 162 + 1 + 1 = 174`;
  family coverage stays explicit as `6 + 4 + 3 + 2 + 157 + 2 = 174`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v170 test-file
  checks, `./bin/souc run` for the two v170 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_local_flowpipe_proof_skeleton`,
  and `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_proof_skeleton_v170`.
- Scope boundary: v170 only records a child-1 proof skeleton over existing
  step skeleton receipts. It does not execute replay, prove containment, prove
  a local flowpipe, validate child `1`, discharge any child, certify a finite
  cover, prove an invariant, prove shadowing, assert unbounded-time behavior,
  or assert a global Lorenz theorem.

## 2026-06-23 cover child-1 local-flowpipe replay executor v171

`stdlib/systems/lorenz_i256_cert.sio` now mirrors the child-0 replay-executor
receipt for child `1` at slots `(1, 0, 0)`. The new system layer binds the v170
child-1 proof skeleton to the five existing step replay-executor receipts while
keeping containment, local proof, child validation, child discharge, finite-cover
certificate, and global-flowpipe claim masks at zero.

- System anchors: child-1 proof skeleton artifact/audit `259900750`/`190272326`,
  v170 portfolio entry/manifest/readiness `850263288`/`234669390`/`779299425`,
  child-1 preflight artifact/audit `105720752`/`750610782`, and the five step
  replay-executor artifact/audit pairs already used by the child-0 v154 replay
  layer.
- System receipt: instance `348190891`, certificate `207316565`, artifact
  `405062249`, audit `948443386`, artifact kind `42`, child selector mask `2`,
  replay status `50`, and non-claim masks including `child_discharge_mask = 0`.
- Portfolio v171 adds kind/checker `174`. The replay-executor receipt is
  `325710521`, audit is `762321265`, entry fingerprint is `137019258`, manifest
  is `858101314`, result coverage is `373864292`, family coverage is
  `29558494`, acceptance receipt is `315556739`, audit receipt is `346304642`,
  and readiness is `146996447`.
- Counter checks: result coverage is explicit as `10 + 163 + 1 + 1 = 175`;
  family coverage is explicit as `6 + 4 + 3 + 2 + 158 + 2 = 175`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v171 test-file
  checks, `./bin/souc run` for the two v171 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_local_flowpipe_replay_executor`,
  and `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_local_flowpipe_replay_executor_v171`.
- Scope boundary: v171 only records a child-1 replay-executor receipt over
  existing step replay receipts. It does not prove containment, prove a local
  flowpipe, validate child `1`, discharge any child, certify a finite cover,
  prove an invariant, prove shadowing, assert unbounded-time behavior, or assert
  a global Lorenz theorem.

## 2026-06-23 cover child-1 containment obligation v172

`stdlib/systems/lorenz_i256_cert.sio` now mirrors the child-0 containment
obligation receipt for child `1` at slots `(1, 0, 0)`. The new system layer
binds the v171 child-1 replay executor to the five existing step-local
containment-obligation receipts while keeping local proof, child validation,
child discharge, finite-cover certificate, and global-flowpipe claim masks at
zero.

- System anchors: child-1 replay executor artifact/audit `405062249`/`948443386`,
  v171 portfolio entry/manifest/readiness `137019258`/`858101314`/`146996447`,
  child-1 proof skeleton artifact/audit `259900750`/`190272326`, and the five
  step containment-obligation artifact/audit pairs already used by the child-0
  v155 containment layer.
- System receipt: instance `990459874`, certificate `958375003`, artifact
  `697916891`, audit `284547222`, artifact kind `43`, child selector mask `2`,
  containment status `51`, `pending_child_validation_mask = 30`, and non-claim
  masks including `child_discharge_mask = 0`.
- Portfolio v172 adds kind/checker `175`. The containment-obligation receipt is
  `708817283`, audit is `514720626`, entry fingerprint is `227808528`, manifest
  is `413882488`, result coverage is `892664373`, family coverage is
  `919785746`, acceptance receipt is `168610802`, audit receipt is `373424726`,
  and readiness is `793575105`.
- Counter checks: result coverage is explicit as `10 + 164 + 1 + 1 = 176`;
  family coverage is explicit as `6 + 4 + 3 + 2 + 159 + 2 = 176`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v172 test-file
  checks, `./bin/souc run` for the two v172 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_containment_obligation`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_containment_obligation_v172`.
- Scope boundary: v172 only records a child-1 containment-obligation receipt
  over existing step containment-obligation receipts. It does not prove a
  completed local flowpipe, validate child `1`, discharge any child, certify a
  finite cover, prove an invariant, prove shadowing, assert unbounded-time
  behavior, or assert a global Lorenz theorem.

## 2026-06-23 cover child-1 validation guard v173

`stdlib/systems/lorenz_i256_cert.sio` now adds the child-1 analogue of the
child-0 validation guard, anchored on the v172 child-1 containment obligation,
the v171 child-1 replay executor, the child-1 obligation seed, and the existing
five-step local-flowpipe-chain receipt. This is only a guard/readiness layer:
it keeps local proof, child validation, child discharge, finite-cover
certificate, and global-flowpipe claim masks at zero.

- System anchors: child-1 containment obligation artifact/audit
  `697916891`/`284547222`, v172 portfolio entry/manifest/readiness
  `227808528`/`413882488`/`793575105`, child-1 replay executor artifact/audit
  `405062249`/`948443386`, v171 portfolio entry/manifest/readiness
  `137019258`/`858101314`/`146996447`, child-1 obligation seed artifact/audit
  `277189357`/`29837118`, and five-step local-flowpipe chain artifact/audit
  `911209450`/`709377850`.
- System receipt: instance `466835680`, certificate `894189666`, artifact
  `399287919`, audit `812182729`, artifact kind `44`, child selector mask `2`,
  validation status `52`, validation anchor/dependency masks `31`, and
  `pending_child_discharge_mask = 255`.
- Portfolio v173 adds kind/checker `176`. The validation-guard receipt is
  `703500225`, audit is `906600492`, entry fingerprint is `595323045`,
  manifest is `136253071`, result coverage is `578053856`, family coverage is
  `976602400`, acceptance receipt is `802326634`, audit receipt is
  `541445443`, and readiness is `701839377`.
- Counter checks: result coverage is explicit as `10 + 165 + 1 + 1 = 177`;
  family coverage is explicit as `6 + 4 + 3 + 2 + 160 + 2 = 177`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v173 test-file
  checks, `./bin/souc run` for the two v173 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_validation_guard`, and
  `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_validation_guard_v173`.
- Scope boundary: v173 only records a child-1 validation-guard receipt. It does
  not validate child `1`, discharge any child, certify a finite cover, prove an
  invariant, prove shadowing, assert unbounded-time behavior, or assert a
  global Lorenz theorem.

## 2026-06-23 cover child-1 X-axis validation witness v174

`stdlib/systems/lorenz_i256_cert.sio` now adds the first child-1 axis-specific
validation-witness receipt, mirroring the child-0 X-axis witness shape for
child `1` at slots `(1, 0, 0)`. The layer binds the v173 child-1 validation
guard, v172 child-1 containment obligation, and the five step-local containment
obligation receipts while keeping axis validation, child validation, child
discharge, finite-cover certificate, and global-flowpipe claim masks at zero.

- System anchors: child-1 validation guard artifact/audit
  `399287919`/`812182729`, v173 portfolio entry/manifest/readiness
  `595323045`/`136253071`/`701839377`, child-1 containment obligation
  artifact/audit `697916891`/`284547222`, v172 portfolio
  entry/manifest/readiness `227808528`/`413882488`/`793575105`, and the five
  step containment-obligation artifact/audit pairs
  `473585568`/`790380830`, `311696236`/`301485983`,
  `360827572`/`456885748`, `37368367`/`94030837`, and
  `291436394`/`876674722`.
- System receipt: instance `708562482`, certificate `227724220`, artifact
  `701185538`, audit `176921379`, artifact kind `45`, child selector mask `2`,
  axis witness mask `1`, pending-axis validation mask `31`, witness status
  `53`, and all axis-validated/child-validated/discharge/global-claim masks at
  zero.
- Portfolio v174 adds kind/checker `177`. The X-axis witness receipt is
  `494166692`, audit is `61623278`, entry fingerprint is `286151155`, manifest
  is `933676753`, result coverage is `338496431`, family coverage is
  `108472139`, acceptance receipt is `684629015`, audit receipt is `51076318`,
  and readiness is `106491107`.
- Counter checks: result coverage is explicit as `10 + 166 + 1 + 1 = 178`;
  family coverage is explicit as `6 + 4 + 3 + 2 + 161 + 2 = 178`.
- Validation gates: `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, the four new v174 test-file
  checks, `./bin/souc run` for the two v174 tiny tests,
  `./scripts/run_sio_test_suite.sh cover_child1_x_axis_validation_witness`,
  and `./scripts/run_sio_test_suite.sh
  solver_portfolio_lorenz_i256_cover_child1_x_axis_validation_witness_v174`.
- Scope boundary: v174 only records a child-1 X-axis validation-witness
  receipt. It does not validate the X axis, validate child `1`, discharge any
  child, certify a finite cover, prove an invariant, prove shadowing, assert
  unbounded-time behavior, or assert a global Lorenz theorem.

## Next concrete implementation steps

1. Continue the earlier-step Lorenz backfill as a ladder, not as a jump:
   steps 1 through 5 now have local-flowpipe proof receipts, explicit bridges
   into adjacent chain contexts, and a separate five-step local-chain
   composition gate without flipping the global claim mask. The global-claim
   preflight now records the missing global obligations explicitly, and v149
   adds one terminal finite-cover candidate seed. v150 expands that seed into
   eight pending child obligations, and v151 selects child `0` as the first
   concrete pending obligation. v152 attaches that child to the local-flowpipe
   machinery as a preflight, and v153 binds the preflight to a five-step proof
   skeleton while keeping the child proof pending. v154 binds that skeleton to
   the five existing step replay executors and still keeps containment pending.
   v155 binds the replay executor to five step-local containment obligations
   while still keeping child validation pending. v156 adds the child-0
   validation guard over those anchors while still keeping
   `child_validated_mask = 0`, `child_discharge_mask = 0`, and
   `global_flowpipe_claim_mask = 0`. v157 adds the first axis-specific
   validation-witness checker surface for the X axis while keeping
   `axis_validated_mask = 0`, v158 adds the companion Y-axis witness surface,
   v159 completes the X/Y/Z witness surface, and v160 bundles those witnesses
   under the same non-claim boundary. v161 adds the first real axis-validation
   checker with explicit X-axis arithmetic over this bundle, and v162 repeats
   the shape for Y while preserving the child/global non-claim masks. v163
   repeats the same arithmetic-validator shape for Z. v164 bundles the X/Y/Z
   arithmetic validators under one receipt while keeping child validation
   pending. v165 connects that bundle to the child-0 guard/containment anchors
   as a readiness receipt. v166 adds the first child-0 local-validation core
   and flips only `child_validated_mask = 1` for child `0`, while keeping
   `child_discharge_mask = 0` and `global_flowpipe_claim_mask = 0`. v167 adds
   the child-0 discharge preflight and makes the remaining blockers explicit:
   remaining child validation, pending child discharge, finite-cover
   certificate, and global-flowpipe claim bits stay unresolved. v168 starts the
   sibling-child ladder by selecting child `1` at slots `(1, 0, 0)` while
   keeping child-1 local-flowpipe proof, validation, discharge, and global
   cover claims pending. v169 attaches child `1` to the five-step
   local-flowpipe-chain machinery as a preflight while still keeping proof,
   validation, discharge, and global cover claims pending. v170 binds that
   preflight to the five step proof-trace skeleton receipts while still keeping
   replay execution, containment, validation, discharge, and global cover claims
   pending. v171 binds that skeleton to the five existing step replay executors
   while still keeping containment, validation, discharge, and global cover
   claims pending. v172 binds that replay executor to the five existing
   step-local containment obligations while still keeping validation, discharge,
   and global cover claims pending. v173 adds the child-1 validation guard over
   those anchors while still keeping `child_validated_mask = 0`,
   `child_discharge_mask = 0`, and `global_flowpipe_claim_mask = 0`. v174 adds
   the child-1 X-axis validation-witness receipt while still keeping
   `axis_validated_mask = 0`, `child_validated_mask = 0`,
   `child_discharge_mask = 0`, and `global_flowpipe_claim_mask = 0`. The next
   narrow move is a child-1 Y-axis validation-witness receipt, not
   discharge or global certification.
2. Repair the current Madaros multimodule native seed-lowering segfault pinned by
   `scripts/ci/madaros_multimodule_witness.sh` on `thin_single`. This broad
   blocker also blocks `test_rational_exact`, `test_qflra_exact`,
   `smt_qflia_basic`, `smt_assumption_core_imported`, and
   `test_smt_solver_basic` at native runtime or thin-link time.
   Until then, keep the Farkas micro-gate as the executable proof-kernel witness
   and use `check` gates for the imported rational/QF_LRA modules.
3. Grow `theorem::lrat` from the current bounded RUP replay, tiny proof-chain,
   colouring `K3` RUP bridge, and lifecycle/ID preflight kernel into the
   production path: real deletion actions, text/binary LRAT parsing, and direct
   `souc_sat` emission of hint lists.
4. Expand `theorem::cardinality` + `theorem::pb` beyond the current pairwise
   exactly-one, graph-colouring ID/manifest/fingerprint map, bound contradiction,
   saturation, division, scaling, addition, and forced-literal propagation checks:
   sequential counters, totalizers, stronger cutting planes, and VeriPB-style
   checked elaboration. Graph colouring should now target the stable colouring
   ID/literal/manifest/fingerprint map instead of ad hoc clause generation.
5. Grow `systems::ball_fixed` from the current i64 radius/status helper surface
   into a full validated-integrator API while the public boundary still avoids
   the current `i128` function-ABI bug. The immediate API surface now includes
   radius propagation, first-step LTE bound helpers, enclosure-containment
   checks, explicit 1D/3D enclosure receipts, a 3D validated-step checker, and
   deterministic step/chain certificate fingerprints. The next step is to
   promote these scalar receipts into a real public enclosure abstraction, either
   `(center_i128, radius_i64, scale)` records or limb-pair i64 helpers, then
   swapping the backend to native `i128`/`i256` only after both
   `madaros-wide-int-gate` and the Lorenz-shaped source-level wide arithmetic
   gates are green.
6. Fix the wide-int XFAILs above. Signed division likely starts in
   `self-hosted/ir/lower.sio` because `OpDiv` currently lowers wide integer
   division without signedness. ABI likely lives in user-call lowering/codegen,
   since inline arithmetic and synthetic wide-int emitters pass. The adjacent
   `i256` division crash has a narrower candidate fix in `self-hosted/ir/lower.sio`:
   source-level wide multiply/division results must reserve the backend scratch
   registers that `IrWideMul` and `IrWideDivFull` already use. The acceptance
   gate is a rebuilt Madaros artifact that turns
  `wide_i256_divfull_scratch_known_failure` and then
  `lorenz_i256_fixed_step_1e6` from XFAIL to XPASS. In parallel, the quotient/
  remainder witness gate above gives a checker-shaped acceptance target: even
  before source-level `/` is trusted, a solver or compiler pass can emit
  quotient/remainder data and have Sounio verify the wide division result by
  multiplication. After that, source-level wide divisions must survive native
  codegen at `2^32`, and only then at high Lorenz fixed-point scales;
  numerator-only success is not enough.
7. Promote the split Lorenz ball gates into a real validated integrator. The
   real target is Taylor/RK with interval remainders and a theorem-facing
   certificate. The v43 Taylor/ball bridge gives this lane its first
   checker-facing order/step/remainder-policy receipt, but the missing next
   layer is still a real high-order producer plus a small replay checker for
   per-step remainder bounds.

## 2026-06-23 interrupted child-1 Y-axis witness v175 handoff

An attempted v175 continuation was interrupted by a concurrent checkout/reset
of `/workspace/sounio` from `fix/equiv-theory-ci-followup` to
`fix/docs-registry-missing-audit-files` at `2026-06-23 19:03 UTC`
(`git reflog`: checkout followed by reset to `HEAD`). That reset displaced the
untracked Lorenz/portfolio lane from branch-tracked truth. The current worktree
may contain untracked copies of `stdlib/systems/lorenz_i256_cert.sio`,
`stdlib/theorem/portfolio.sio`, and related run-pass tests, but they were not
used as accepted v175 evidence in this checkout.

The v175 work below was therefore not completed or validated in the live
checkout. Treat it as a precise resumption packet, not as accepted evidence.

- Intended system layer: child `1`, slots `(1, 0, 0)`, Y axis `1`, axis witness
  mask `2`, artifact kind `46`, witness status `54`.
- Intended anchors: v173 child-1 validation-guard artifact/audit
  `399287919`/`812182729`, v173 portfolio entry/manifest/readiness
  `595323045`/`136253071`/`701839377`, v172 child-1 containment-obligation
  artifact/audit `697916891`/`284547222`, v172 portfolio
  entry/manifest/readiness `227808528`/`413882488`/`793575105`, and the five
  step containment-obligation artifact/audit pairs
  `473585568`/`790380830`, `311696236`/`301485983`,
  `360827572`/`456885748`, `37368367`/`94030837`, and
  `291436394`/`876674722`.
- Derived system constants for resumption: instance `452101076`, certificate
  `32468837`, artifact `597632301`, audit `85858631`.
- Intended portfolio v175 kind/checker: `178`.
- Derived portfolio constants for resumption: receipt `549267660`, audit
  `219936520`, entry `961194039`, manifest `398024434`, result coverage
  `765863019`, family coverage `907265898`, acceptance receipt `147594670`,
  audit receipt `473727672`, readiness `52794647`.
- Target counter values: `10 + 167 + 1 + 1 = 179` for result coverage and
  `6 + 4 + 3 + 2 + 162 + 2 = 179` for family coverage.
- Intended non-claims: `axis_validated_mask = 0`,
  `local_flowpipe_proof_mask = 0`, `child_validated_mask = 0`,
  `child_discharge_mask = 0`, `global_cover_certificate_mask = 0`, and
  `global_flowpipe_claim_mask = 0`.
- Required resumption gates: restore the v174 Lorenz/portfolio lane in an
  isolated or coordinated worktree, add the v175 system/portfolio blocks, add
  tiny/imported tests for
  `lorenz_i256_cover_child1_y_axis_validation_witness` and
  `solver_portfolio_lorenz_i256_cover_child1_y_axis_validation_witness_v175`,
  then run `./bin/souc check stdlib/systems/lorenz_i256_cert.sio`,
  `./bin/souc check stdlib/theorem/portfolio.sio`, checks for the four v175
  tests, tiny test runs, the two focused suites, offload review, and diff
  hygiene. The old full-portfolio timeout blocker remains separate unless the
  full portfolio suite is rerun and proven.

## Non-claims

- This does not claim Sounio beats Kissat, CaDiCaL, cvc5, Bitwuzla, or Z3.
- This does not claim a planar Hadwiger-Nelson chi >= 6 witness.
- This does not claim f64 Lorenz simulation is rigorous.
- This does not claim i256 is production-ready in the Sounio user language.
