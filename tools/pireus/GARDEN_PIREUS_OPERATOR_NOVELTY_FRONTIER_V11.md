# Garden: Pireus Operator Novelty Frontier V11

Status: GARDEN

Concept-ID: SOUNIO-PIREUS-OPERATOR-NOVELTY-FRONTIER

Semantic-Lane-ID: pireus-operator-novelty-frontier-v11-20260830

## Question

What is the complete bounded novelty frontier of the frozen V10
'enumerated_coefficient_delta' grammar when every one of its 7200 bilinear
operator candidates is compared against:

1. every class and action in the frozen V10 atlas quotient; and
2. every other grammar candidate under the frozen 'C2_diag' action?

No frontier count, atlas-collision count, quotient-class count, witness family,
or digest is known in this Garden. Those values must first be produced by a
matcher-free Sounio executable.

## Authority Order

    GARDEN
    -> SOUNIO_EXECUTABLE
    -> SEMANTICS_FROZEN
    -> PARITY_OPEN
    -> CLAIM_READY

Sounio is SEMANTIC_AUTHORITY. Lean 4 is FORMAL_PARITY, Koka is EFFECT_PARITY,
C++ is MATERIAL_PARITY, and Haskell is an optional denotational baseline.
Python and Rust are forbidden oracles. External LLMs are review-only and
cannot confirm a result.

No parity implementation may execute until a V11 Sounio artifact is frozen by
source and semantics hashes.

## Frozen Parent

V11 is a child of the frozen V10 discovery engine, not a retrospective rewrite
of it.

    parent_source=stdlib/hardware/pireus/operator_discovery_engine.sio
    parent_source_sha256=919b6104cbce1c5f8643f5df88b9071305d3fee854f785ac63a883bc45f16117
    parent_semantics_sha256=2640bb928740ef03f5a42725f42c62735bc2121621bb3dfd4b4cdf3572003ec5
    parent_freeze_sha256=9a83c9a4b920d41ee91bd7681f4e95ac11480d762185ec9ff003692d3c01d247
    parent_parity_open_sha256=5f109404d2a2e8e56e6cff486f871e0961f843edd2e48e2feb5f5717d1d8d39d
    parent_material_commit=51acf5795b65
    parent_material_receipt_sha256=56d0ed053a67ad1f1d3065411b48638571b2c77d2f64361090e1d9d6e21e78ab

The parent result establishes only candidate zero as internally N2 relative to
the frozen six representatives. V11 must not extrapolate that result to the
remaining 7199 candidates.

## Grammar

The grammar is definitionally finite:

    input_0 in 1..15
    input_1 in 1..15
    output  in 0..15
    delta   in {-1,+1}

    coordinate = ((input_0 * 16) + input_1) * 16 + output
    candidate  = parent + delta * basis_tensor(coordinate)

Its cardinality is 15 * 15 * 16 * 2 = 7200. The encoding from candidate ID to
(coordinate, delta) must be injective and invertible. Excluding lane zero from
both inputs preserves the frozen two-sided e0 unit by construction, but the
executable must still check this boundary.

V11 emits the entire population and selects no winner.

## Exact Frontier Algorithm

The reference equality is exact equality of all 4096 integer tensor
coefficients. The optimized classifier may avoid materializing 7200 complete
tensors only by emitting an exact sparse certificate.

For each frozen atlas representative R, Sounio first scans the complete parent
tensor P and records the difference profile:

    D_R = { cell | P[cell] != R[cell] }

The profile contains the exact mismatch count, first two mismatch cells, and
their signed coefficient differences. For a candidate
C = P + delta * e_m, collision with R is decided by:

    C == R
    iff D_R is contained in {m}
        and R[m] - P[m] == delta

The first separator is derived from the ordered difference support after the
single mutation. Sounio must cross-check the sparse classifier against direct
4096-cell comparison for deterministic canaries at the first, middle, and last
grammar IDs, plus one synthetic exact-collision control. These controls do not
define the frontier result.

For candidate-to-candidate quotienting, the identity action uses grammar
injectivity. For the nonidentity action q, Sounio scans the complete base
difference q(P) - P once and derives both the transported coordinate q(m) and
the basis character chi_q(m). The exact transport law is:

    q(P + delta * e_m) - P
      = (q(P) - P) + delta * chi_q(m) * e_q(m)

The frozen V10 action is unsigned, so its derived character is +1 on every
cell, but V11 must carry and check the character rather than silently erase it.
For each candidate, the action lands back in the grammar only when the
right-hand difference from P is exactly one signed basis tensor at a legal
grammar coordinate whose two input lanes are in 1..15. This is an exact
O(4096 * representatives + 7200 * representatives) frontier algorithm after
the complete base profiles, action permutation, and character have been
derived. It is not a probabilistic fingerprint and not a quadratic pair
sampler.

The nonidentity action is checked as an involution on the ambient tensor. A
candidate image inside the grammar must map symmetrically back to its source.
Images outside the grammar remain singleton frontier classes. Distinct
in-grammar pairs are counted once in canonical ID order; fixed points do not
reduce the class count. The candidate partition must cover all 7200 IDs.

## Required First Result

The matcher-free Sounio transcript must derive and emit:

- 7200 generated and typed-admitted candidate records;
- atlas collision and N2 counts;
- a separator for every non-collision against every representative;
- the six complete base-difference profiles;
- identity and nonidentity candidate-quotient collision counts;
- quotient-frontier class count;
- first and last IDs in every derived outcome partition;
- exact SHA-256 digests for the ordered candidate census, separator census,
  quotient map, and aggregate frontier;
- explicit positive collision and incomplete-search controls;
- explicit refusal of partial enumeration, injected expected counts, candidate
  selection, parity semantic writes, material claims, historical claims,
  Python, Rust, and raw-ELF execution.

The first executable must not contain matchers for the unknown derived counts
or digests. Structural constants such as 7200, 4096, 3, and 2 are grammar
definitions, not expected scientific results.

## Novelty Boundary

N2 means only:

    exactly distinct from every frozen V10 atlas representative
    under every frozen C2_diag action

The V11 frontier may establish exhaustive internal novelty for this finite
grammar and quotient. It does not establish:

- novelty relative to all mathematical operators;
- algebraic, algorithmic, material, or scientific novelty;
- utility, performance, lowerability, or hardware superiority;
- global, historical, publication, patent, or priority novelty;
- CLAIM_READY.

All such fields remain false unless a later, separately frozen stage discharges
its own evidence obligations.

## Falsifiers

The V11 hypothesis is falsified if any of the following occurs:

- fewer or more than 7200 unique grammar IDs are generated;
- ID decoding is not injective and invertible;
- a supposedly unit-preserving candidate touches an e0 input cell;
- sparse and direct equality disagree on any control;
- a collision is promoted to novelty;
- an incomplete comparison is promoted to novelty;
- the ambient nonidentity action is not involutive;
- a transported basis character or legal-support condition is omitted;
- an in-grammar image does not map symmetrically back to its source;
- an out-of-grammar image is counted as a pair;
- a fixed point reduces the quotient class count;
- a distinct pair is counted in both orders;
- the candidate quotient partition does not cover all 7200 IDs;
- frontier arithmetic does not partition all 7200 candidates;
- a result matcher appears before the first Sounio transcript is preserved;
- any parity, LLM, Python, Rust, material, or target process creates semantics
  or an expected result.

## Execution Boundary

Use ./bin/souc with an explicit canonical engine selection and never invoke a
raw compiler ELF. Heavy first execution belongs on Sounio Compiler Foundry or
Slurm. The first transcript and test transcript must record compiler hash,
source hash, parent hashes, toolchain, hardware, command, result hash, and
Guardian ALLOW/DENY decisions.

## Review Provenance

External review has role REVIEW_ONLY and cannot confirm any V11 result.

- `/tmp/llm-offload-BRFqqq/` returned an empty artifact and is recorded as
  `ERROR_EMPTY_NOT_A_PASS`.
- `/tmp/llm-offload-W5M5lg/` identified the missing transported basis
  character and incomplete quotient-accounting obligations; both were
  corrected before this Garden was sealed.
- `/tmp/llm-offload-bYlL8p/` found no false closed-form frontier result and no
  remaining algebraic contradiction. It left linearity, signed-monomial basis
  transport, involution, codec injectivity, and partition coverage as
  executable obligations.

    review_provider=xai:grok-4.5
    review_role=REVIEW_ONLY
    llm_confirmed_result=false
