# Garden: Pireus Proof-Carrying Operator Morphogenesis V12

Status: GARDEN

Concept-ID: SOUNIO-PIREUS-PROOF-CARRYING-OPERATOR-MORPHOGENESIS

Semantic-Lane-ID: pireus-proof-carrying-morphogenesis-v12-20260830

## Question

Can Pireus construct a sequence of new 16-lane twisted-XOR operators, rather
than enumerate a predeclared mutation list, such that every emitted operator
carries an exact machine-checkable certificate separating its complete orbit
from a frozen archive of earlier Sounio operators?

The first result is unknown in this Garden. In particular, no generated sign
table, ANF genome, orbit size, law spectrum, probe value, digest, or derived
population statistic is specified here. Those values must first be produced by
a matcher-free Sounio executable.

## Authority Order

    GARDEN
    -> SOUNIO_EXECUTABLE
    -> SEMANTICS_FROZEN
    -> PARITY_OPEN
    -> CLAIM_READY

Sounio is SEMANTIC_AUTHORITY. Lean 4 is FORMAL_PARITY, Koka is EFFECT_PARITY,
C++ is MATERIAL_PARITY, and Haskell is an optional denotational baseline.
Python and Rust are forbidden oracles. External LLMs are REVIEW_ONLY and may
criticize the construction but cannot create, derive, or confirm its result.

No parity implementation or target execution may run until the V12 Sounio
artifact and its first result have been frozen by hash.

## Frozen Parents

V12 has two append-only parents with different jobs.

The v4 cubic forge supplies a same-shape archive of 48 generated full-support
sign tables. Its frozen Sounio result exposes the complete population, including
all 12,288 sign bits, through `PireusCubicForgeResult.population`.

    cubic_source=stdlib/hardware/pireus/cubic_operator_forge.sio
    cubic_source_sha256=2c295c48bcd2de0f43a42787dcc612f78c7d40d528641e4fec890858d881c974
    cubic_semantics_sha256=e8268af20770dbf292fb39f92793b7b89d1651b2e88193e0cb6ee765dfc1f1ff
    cubic_freeze_sha256=1da425c1ff53273825a71b46850e0cd9e7d4cd5b77aa79eb65ef269aadd5a87b
    cubic_parity_open_sha256=82cdb8875783d34a903b7b599aeeb9501d73eba9ed0a2426040bd708aaf2665a
    cubic_frozen_evidence_sha256=d27915015cabda1d11211968e0bde5655757599d8dc3313fbfc0506877e49694

The v11 frontier supplies the latest exhaustive archive accounting and target
custody boundary. Its C++ receipt is material parity only and cannot define a
V12 sign, genome, witness, or expected result.

    frontier_source=stdlib/hardware/pireus/operator_novelty_frontier.sio
    frontier_source_sha256=9289cd504385e2f1f4eed095d82a963cf2e5e67124bf8d267d1bc6ccda7ac36b
    frontier_semantics_sha256=f1e339ec7bc290f412d42bba3fa1ba609fd89947408ea422ab96026cce5883dc
    frontier_freeze_sha256=b57decc8ff929640345e47edc931bdfa6cd06c738d3ff9591d3a460593dae242
    frontier_parity_open_sha256=f7cde0ed063d136bbef43cf9e820d734341f87717bb26e130a3643bc62fb31de
    frontier_material_commit=82f5d7302c35
    frontier_material_receipt_sha256=95c0a63f8833abd8636d4dd6d43097c5172e15b3992a3459a1f78c8dcbf198ad
    frontier_material_gate_sha256=5f80f5bd30093b2f9a71917586ba136aea42bcdb1dee49128ea34d02899abc33

V12 restarts at GARDEN. Neither parent's parity state transfers to V12.

## Operator Space

For 16 lanes indexed by `Z2^4`, every V12 phenotype has the exact form

    T_p(a, b)[d]
      = sum_i sigma_cd(i, i XOR d)
          * (-1)^p(i, i XOR d)
          * a[i]
          * b[i XOR d]

where `p(i,j)` is Boolean and is zero whenever `i=0` or `j=0`.

`sigma_cd` is the inherited Sounio function
`algebra::cayley_dickson::cd_sigma` at source SHA-256
`e7dd98de0644013ebf6e0d435fddb7f893720f684c96c3fbe20cc11b1f518fed`.
V12 does not assume its unit boundary from notation: the executable must check
`sigma_cd(0,j,4)=+1` and `sigma_cd(i,0,4)=+1` for all 16 lanes before admitting
any phenotype.

The canonical evaluator defines the reduction spine operationally: for each
destination `d` in ascending order, it visits `i=0..15` in ascending order and
performs one multiply followed by one addition per term, without reassociation
or contraction. Preserving that spine is an executable comparison between the
generated sign microprogram and this reference evaluator, not a consequence of
the phase formula alone.

Subject to those checked parent and evaluator obligations, every phenotype:

- preserves the XOR destination and partner map;
- has exactly one nonzero `+1` or `-1` coefficient for every basis pair;
- preserves both sides of the `e0` unit;
- preserves the exact ascending-`i` reduction spine;
- differs from another phenotype only through its normalized sign phase.

The interior domain has `15 * 15 = 225` cells. A truth table on those cells is
equivalent to a unique mixed Boolean ANF

    p(i,j) = XOR_(A nonempty, B nonempty)
               c[A,B] * product_(r in A) i_r * product_(s in B) j_s

with `(2^4 - 1)^2 = 225` coefficients. The Sounio executable must derive the
ANF by the exact Boolean Mobius transform and reconstruct all 225 truth cells
from it. The truth table is the phenotype; the ANF coefficients are its
proof-carrying genotype. Neither representation may be a probabilistic
fingerprint.

## Unit-Preserving Action

The acting group is `C2`, generated by swapping bit 0 and bit 1 in every lane
index. The set of phenotypes is quotiented into orbits under this action. The
lane map `q` fixes lane zero, preserves XOR, is an involution, and acts unsigned
on basis vectors. Its exact pullback on a sign table is

    (q . tau)(i,j) = tau(q(i), q(j))

because `q^-1=q`. The executable must also transport the XOR destination by
`q(i XOR j)=q(i) XOR q(j)`. This action is deliberately different from v11's
lane-0/lane-1 transposition, which does not preserve the V12 unit boundary.

The initial archive is built by executing the frozen v4 Sounio parent, checking
its frozen matcher, importing its 48 complete sign tables, applying the V12
involution to each, and deduplicating exact 256-cell images. The resulting
archive must be closed under the action before generation begins. Its unique
image count is a derived V12 result, not an injected expectation.

## Constructive Diagonalizer

Let the current exact archive be the ordered distinct image list

    F_0, F_1, ..., F_(m-1)

with `m <= 225`. Let `c_k` be the `k`th interior pair in destination-major
lexicographic order: sort the 225 non-axis pairs by
`(destination=i XOR j, i, j)`. For every normalized sign table `F`, define its
phase relative to the pinned Cayley-Dickson base by

    phase(F)[i,j] = 0  iff  sign(F)[i,j] = sigma_cd(i,j,4)
    phase(F)[i,j] = 1  iff  sign(F)[i,j] = -sigma_cd(i,j,4)

The sign-range and axis checks make this map total on every admitted archive
image. Pireus constructs a new truth table `P` by

    P[c_k] = 1 XOR phase(F_k)[c_k]    for 0 <= k < m
    P[c]   = 0                        for every other interior cell
    P[0,j] = P[i,0] = 0

and converts `P` to its mixed ANF genotype.

For every archived image `F_k`, the tuple

    (archive_index=k,
     witness_cell=c_k,
     archived_phase=phase(F_k)[c_k],
     generated_phase=P[c_k])

is a separator certificate because the last two bits differ exactly. Distinct
witness cells make the assignments nonconflicting. The proof is constructive:
the candidate is produced from the archive and the certificate is produced by
the same step. No post-hoc search is allowed to label an already chosen table
as novel.

The candidate certificate is lifted to its action image rather than replaced
by an unstructured scan. For every archived `F_j`, closure supplies an index
`k` with `F_k=q.F_j`. If `P` differs from `F_k` at `c_k`, then `q.P` differs
from `F_j` at `q(c_k)`. Each generated record must carry both the original and
transported separator tuples and directly verify both bits. A complete
256-cell comparison remains an independent executable cross-check.

After direct 256-cell verification, Pireus computes the new candidate's full
order-two orbit, rejects any collision with the pre-generation archive, appends
the distinct orbit images, restores canonical order, and checks closure. The
next epoch must diagonalize against the enlarged archive. This makes each
candidate novel relative to the complete orbit closure of the cubic parent and
all earlier V12 candidates.

The first executable has a structural budget of 16 generation epochs. It must
not stop early because a preferred candidate was found. It must stop closed if
the archive exceeds the 225-cell diagonal capacity. The number of appended
orbit images, fixed points, pair orbits, certificates, and final archive images
are execution outputs.

## Proof-Carrying Record

Every generated record must contain or bind:

- its epoch and immutable parent-archive digest;
- all 256 normalized signs and all 225 phase truth bits;
- all 225 mixed-ANF coefficients and an exact inverse-transform check;
- one exact separator certificate for every pre-generation archive image;
- direct equality checks against every pre-generation archive image;
- the action image, orbit kind, and canonical orbit representative;
- exact left-unit, right-unit, support, XOR-route, and reduction-order checks;
- commutator and associator spectra as diagnostics, not algebraic claims;
- exact integer probe outputs under the canonical reduction spine;
- ordered SHA-256 digests for phenotype, genotype, certificates, orbit, epoch,
  and cumulative archive.

The executable must also emit a replayable 256-entry sign microprogram. This is
a semantic artifact that later hardware ontologies may lower. It is not an
instruction schedule or a material result.

## Required First Result

The matcher-free Sounio transcript must derive and emit:

- the exact unique size and closure status of the imported cubic orbit archive;
- 16 generated operator records or a fail-closed capacity refusal;
- every separator certificate and direct comparison result;
- every ANF forward/inverse check;
- per-epoch orbit and archive accounting;
- exact structural and law-spectrum diagnostics;
- exact ordered digests for every proof-carrying record and the full run;
- negative controls for witness reuse, witness-bit agreement, omitted orbit
  images, archive duplication, broken unit axes, broken XOR routing, incomplete
  comparison, injected result fields, parity writes, LLM promotion, forbidden
  oracles, raw-ELF execution, material promotion, and claim promotion.

The first executable may contain structural matchers for 16 epochs, 16 lanes,
225 interior cells, 256 sign cells, and group order 2. It must not contain a
matcher for any generated sign, derived archive count, orbit distribution,
law-spectrum value, probe output, or digest.

## Novelty Boundary

If every obligation passes, V12 may establish only:

    constructive bounded relative operator novelty

meaning exact inequivalence under the declared order-two action relative to the
frozen same-shape cubic archive and earlier V12 generations.

It does not establish:

- exhaustion of the `2^225` normalized-twist space;
- inequivalence under full `GL(4,2)`, gauge, isotopy, or algebra isomorphism;
- cocycle, alternativity, associativity, norm, division, or zero-divisor laws;
- algorithmic usefulness, lowerability, cost, performance, or superiority;
- mathematical, scientific, global, historical, publication, patent, priority,
  or CLAIM_READY novelty.

The construction is a Pireus novelty generator only inside its explicit finite
archive, phenotype, and action contract. Broader novelty fields remain false.

## Target Boundary

The canonical future material classes remain Xeon, Apple Silicon, DGX, and the
declared dual AMD Alveo U250 target. Spark material work routes through
Kubernetes nodes `spark-3c59` and `spark-8e54`; this Garden does not reopen the
Slurm route. The generated sign microprogram may later be lowered against
processor-code ontologies, but target observations cannot alter a frozen V12
operator identity.

No V12 target process runs before parity opens. A later material receipt must
separate canonical target-class coverage from physical endpoint coverage and
must not infer two observed U250 cards from a dual-card declaration.

## Falsifiers

The V12 construction is falsified if any of the following occurs:

- the cubic parent does not match its frozen Sounio semantics;
- the initial or cumulative archive is not exact-image deduplicated and closed;
- an archived image is omitted from a generation step;
- two archived images are assigned the same witness cell;
- a generated witness bit equals its archived bit;
- a candidate or its action image equals a pre-generation archive image;
- a generated truth table and its mixed ANF disagree at any interior cell;
- a unit-axis phase is nonzero or a basis coefficient leaves `{-1,+1}`;
- XOR destination, partner, or reduction order changes;
- generation stops after selection rather than the structural epoch budget;
- the archive exceeds diagonal capacity without a fail-closed refusal;
- any expected sign, count, spectrum, probe, or digest enters before first run;
- parity, hardware, LLM, Python, Rust, or a raw compiler ELF creates semantics,
  supplies an expected result, or promotes a claim.

## Execution Boundary

Use `./bin/souc` with an explicit canonical engine selection and never execute a
raw compiler ELF. The first transcript and test transcript must bind the Garden
commit, parent hashes, source hash, wrapper, resolver, compiler, hardware,
command, Guardian decisions, and result hash. Heavy replay belongs on Sounio
Compiler Foundry or the repository's approved heavy-validation route.

## Review Boundary

External review is mandatory for the construction's math claims and remains
REVIEW_ONLY. A reviewer may expose a false proof obligation or missing
falsifier, but cannot supply a phase bit, generated operator, expected count,
digest, novelty verdict, or semantic confirmation.

The initial xAI review checked the 225-dimensional phase/ANF space, axis
boundary, involutive linear action, diagonal separator construction, capacity,
and claim scope. It correctly required the Cayley-Dickson source and unit laws
to be explicit, separated the acting group from its orbit quotient, and moved
reduction-order preservation from an implied consequence to an executable
obligation. Those findings are applied above. The correction review accepted
the pinned operator formula but returned only a short partial audit, so it is
not represented as a second full pass.

    review_provider=xai/grok-4.5
    review_role=REVIEW_ONLY
    initial_review_guardian_frame_sha256=61703d64363f34f75a822445fc4c46758011cf36fb241df2ae7ecdfe4d3d0c18
    initial_review_raw=/tmp/llm-offload-8Nq84L/
    correction_review_guardian_frame_sha256=5befcc0a55aa96c9647d8b4e37d95555fc3a9395411086621ace72ef0c2e47a2
    correction_review_raw=/tmp/llm-offload-zzcoBh/
    llm_confirmed_result=false
