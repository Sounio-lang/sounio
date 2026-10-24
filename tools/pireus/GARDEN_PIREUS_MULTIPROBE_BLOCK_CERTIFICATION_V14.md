# Garden: Pireus Multi-Probe Block Certification V14

Status: GARDEN

Concept-ID: SOUNIO-PIREUS-MULTIPROBE-BLOCK-CERTIFICATION

Semantic-Lane-ID: pireus-multiprobe-block-certification-v14-20260831

## Semantic Lane Declaration

    Semantic-Lane-ID: pireus-multiprobe-block-certification-v14-20260831
    Owner: codex under founder direction
    Concept-IDs: proposed SOUNIO-PIREUS-MULTIPROBE-BLOCK-CERTIFICATION
    Intent-Preserved: Pireus generates new operators and carries an exact,
      reusable strategy for testing them without surrendering Sounio authority.
    Transformation: add a target-neutral certification-strategy ontology that
      partitions one exact finite work function into typed probes and blocks.
    Types-Changed: none in GARDEN; proposed V14 Sounio types are listed below.
    Effects-Changed: none in GARDEN; later source must declare IO, mutation,
      hashing, and epistemic refusal effects explicitly.
    IR-Changed: none.
    Claims-Introduced: none in GARDEN; a later frozen Sounio result may claim
      only derivation and validation of the declared structural plan.
    Claims-Forbidden: hash injectivity, automatic proof, cross-trust
      composition, cache correctness, speedup, full V13 admission, novelty,
      subquadratic complexity, and CLAIM_READY.
    Assumptions: exact frozen-parent bytes; operational SHA-256 collision
      resistance only; closed deterministic work domains before reuse.
    Write-Set: this Garden seed, technical Garden, GARDEN-status contract;
      later separately committed V14 Sounio source, example, test, gate, and
      evidence artifacts.
    Read-Set: frozen V13 Garden, contract, source, transcript, receipt, freeze;
      frozen Operator Genome V3 semantics; Pireus ontology model.
    Positive-Witness: induced block 63 is [39690,40320) with width 630.
    Negative-Witness: block 63 at [39689,40319) is rejected as
      PARTITION_BOUNDARY_MISMATCH.
    Acceptance-Gate: future scripts/ci/pireus_multiprobe_block_certification.sh
      with pre-execution Guardian authorization and deliberate negative cases.
    Integration-Target: stdlib/hardware/pireus operator ontology and future
      operator-generation certification envelopes.
    Authoritative-Only-If: matcher-free Sounio executes first, its result and
      source are committed and frozen by hash, and no parity source supplied or
      repaired an expected value.

Founder direction preserved:

> Pireus must make an operator's certification strategy reusable by the
> generation of new operators, instead of replaying one opaque 39-minute
> admission every time.

## Question

Can Sounio make the certification strategy a first-class part of a Pireus
operator, derive a canonical multi-probe plan whose finite work is partitioned
into independently addressable blocks, and admit exact-domain reuse without
letting a hash, a cache, a parity language, a target, or an LLM redefine the
operator or its expected result?

The first V14 result is unknown. This Garden fixes the protocol question and
the structural V13 profile, but contains no V14 plan digest, work address,
receipt digest, expected result, aggregate, cache hit, completion count,
negative-case count, performance result, or success value.

## Authority Order

    GARDEN
    -> SOUNIO_EXECUTABLE
    -> SEMANTICS_FROZEN
    -> PARITY_OPEN
    -> CLAIM_READY

Sounio is SEMANTIC_AUTHORITY. Lean 4 is FORMAL_PARITY, Koka is EFFECT_PARITY,
C++ is MATERIAL_PARITY, and Haskell is an optional denotational baseline.
Python and Rust are forbidden oracles. Shell, Node, Ruby, awk, bc, and similar
disposable tools may perform non-semantic process control or byte hashing, but
may not construct a probe, work domain, expected digest, block result, or
aggregate. External LLMs are REVIEW_ONLY.

No parity implementation or target process may execute until the matcher-free
V14 Sounio source, first V14 transcript, and frozen V14 semantics are committed
in that order.

## Frozen Parents

### V13 semantic parent

V14 starts from the frozen Sounio V13 operator-orbit admission, not from its
later Lean replay. The exact parent identities are:

    concept=SOUNIO-PIREUS-OPERATOR-ORBIT-CANONICALIZATION
    garden_commit=86755b3027a3c5d0b7d5961e4012cab95d4c8c31
    matcher_free_executable_commit=73704f7afed6780c3a317b739cbd35fe94dbe395
    first_evidence_commit=22fbabe81cf365c0b542d8a425ec4c081f31e390
    matcher_commit=00200c2aa5a021cdc8d91de2d231f3e573d372bb
    matcher_free_module_sha256=3136968a83bbba18d56c543895d6bbd9530ccf6c59db78ac6b6f2fa3bd26c9e4
    matcher_module_sha256=7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae
    first_transcript_sha256=16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f
    first_receipt_sha256=be0ef127e7c40cf0167cb55189c39245dbfd93ffed990d64a003cadf3f19f38b
    semantics_sha256=0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c

The V13 transcript contains the 32 admitted records that seed the first V14
probe set. V14 must hash the transcript directly inside Sounio and parse the
complete records without shell preprocessing. These are frozen parent inputs,
not V14 expected results.

### Operator ontology parent

The frozen Operator Genome semantic identity is:

    concept=SOUNIO-PIREUS-OPERATOR-GENOME
    genome_sha256=99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926

V14 extends the ontology by linking an operator identity to a certification
strategy. It does not change the genome's partner map, signs, destinations,
ordinals, reduction barriers, target envelopes, or frozen semantic digest.

## Ontology

A `CertificationStrategy` is a target-neutral semantic object linked to one
operator lineage. It contains:

1. `WorkFunctionIdentity`: the exact algorithm, action space, enumeration
   order, scalar/numeric rules, and deterministic execution contract.
2. `ProbeSet`: a nonempty canonical sequence of full probe identities.
3. `PartitionSchema`: the unique rule inducing the required blocks for each
   declared finite work interval.
4. `TrustPolicy`: the receipt and proof strata allowed to fill obligations.
5. `AggregationSchema`: the domain-separated, canonical encoding used to bind
   blocks into probes and probes into an admission.
6. `ClaimBoundary`: fields that must remain false at the current stage.

The strategy is part of operator provenance and reuse policy. It is not part
of the mathematical multiplication law, and changing it does not silently
create a new operator. The operator identity and strategy identity are stored
and hashed separately, then linked by an explicit envelope.

## Four Distinct Identities

V14 must never collapse these objects:

| Object | Meaning | Not equivalent to |
| --- | --- | --- |
| `WorkDomain` | Exact semantic inputs of one deterministic job | Its result |
| `WorkAddress` | SHA-256 address of the canonical `WorkDomain` bytes | Semantic equality or proof |
| `ResultDigest` | Identity of the complete result produced for that domain | Job identity |
| `ReceiptBinding` / `ProofBinding` | Evidence plus explicit trust kind | Automatic formal proof or CLAIM_READY |

Exact field equality establishes that two serialized work domains are the same
protocol job. The recomputed address then checks the integrity binding. Digest
equality alone is never an equality predicate for domains, probes, operators,
results, or certificates.

## Closed Work Domain

Every input capable of changing a block result belongs to the canonical
`WorkDomain`:

- protocol and serialization schema versions;
- Sounio authority source hash and frozen semantics hash;
- operator lineage/genome hash and certification-strategy hash;
- work-function and action-space identifiers;
- exact enumeration order and finite interval;
- full probe identity and canonical probe bytes;
- partition schema, block index, start, and width;
- scalar, integer, Boolean, floating-point, overflow, and reduction semantics;
- determinism mode and execution-policy identity;
- toolchain identity when it can change semantic output;
- hardware, FP mode, thread model, schedule, driver, or target identity whenever
  output is material-bound rather than host-independent.

V14 has two determinism modes:

    HOST_INDEPENDENT_EXACT
    MATERIAL_BOUND

The initial V13 canonical-action scan uses only exact finite Boolean and integer
semantics and therefore requests `HOST_INDEPENDENT_EXACT`. A different host may
execute a matching block, but any different result for the same exact domain
is a hard conflict, not a new cache entry. Numeric or target work that lacks a
host-independent norm must use `MATERIAL_BOUND` and include every
result-affecting material field in its domain.

No claim that this operational contract is a universal purity theorem is made
at GARDEN. Until the work function is frozen and its deterministic composition
obligations are discharged, reuse remains closed.

## Canonical Serialization and Address

The first executable must define one total, length-delimited, domain-separated
serialization. Every field has an explicit tag, width, signedness, byte order,
and sequence length. No textual formatting, map iteration order, host word
size, pointer, path alias, or locale enters identity.

For domain-separation tag `PIREUS_V14_WORK_DOMAIN`, define structurally:

    work_address = SHA256(
        tag || schema || canonical_serialization(work_domain))

SHA-256 is computed inside Sounio. The address is operationally collision
resistant, but V14 does not assume or claim injectivity. Reuse checks the full
decoded domain field by field before recomputing and comparing its address.

## Expected and Observed Results

The executable representation must use distinct types:

    ExpectedDigest
    ObservedDigest

Only Sounio in role SEMANTIC_AUTHORITY at an admissible stage may construct an
`ExpectedDigest`. Lean, Koka, C++, Haskell, hardware processes, and LLMs can
construct only observations or their own explicit proof bindings. An observed
digest can be compared with an expected digest; it can never inhabit the
expected slot or repair a Sounio mismatch.

The first matcher-free V14 source contains neither kind as a golden. Its first
execution originates the V14 plan and protocol-validation results. Actual V13
block-result expected digests remain absent and all block obligations remain
pending in the first plan.

## Receipt and Proof Strata

`Certificate` is the protocol family name, not a claim that every member is a
formal proof. Each binding carries exactly one trust kind:

    SOUNIO_EXECUTION_RECEIPT
    LEAN_KERNEL_PROOF
    LEAN_NATIVE_DECIDE_REPORT
    KOKA_EFFECT_PARITY
    CPP_MATERIAL_PARITY
    HASKELL_DENOTATIONAL_OBSERVATION

Native-decide trust is not silently described as kernel reduction. An
execution receipt is not a proof theorem. Material parity is not semantic
authority. Bindings from different trust kinds do not compose into a stronger
kind without a separately frozen transport rule. No such transport rule exists
in this Garden.

## Induced Block Partition

The initial profile is inherited structurally from V13:

    probe_count=32
    action_view_count_per_probe=40320
    block_count_per_probe=64
    block_width=630

The partition schema induces, rather than merely permits, each block:

    block(k).index = k
    block(k).start = 630*k
    block(k).width = 630
    block(k).end = 630*(k+1)
    0 <= k < 64

Thus the first block is `[0,630)` and the last is `[39690,40320)`. The plan is

    32 probes * 64 blocks * 630 views
    = 2048 logical block jobs covering 32 * 40320 views.

This arithmetic defines work shape only. It does not establish that executing
2048 jobs is faster than the monolith, that any block is cached, or that the
current canonicalizer can already resume at block boundaries.

## Block Receipt

A completed block receipt binds:

- the full decoded `WorkDomain` and recomputed `WorkAddress`;
- producer language, producer role, stage, and trust kind;
- Sounio source and frozen semantics hashes;
- exact command, toolchain, hardware/execution context, and policy receipt;
- declared extent and independently recorded actual start, end, visit count,
  and completion status;
- result payload schema and `ResultDigest` over the complete extent product;
- explicit conflict, reuse, parity, novelty, performance, and claim flags.

A block is complete only if it is the block induced for its index, its actual
extent equals the declared half-open interval, every required view is visited
exactly once under the frozen ordering, the result digest binds the full result
payload, authority and stage are valid, and all mismatch/conflict flags are
false.

## Probe and Admission Composition

For one probe, the required block set must be exactly equal to
`PartitionInduce(schema, [0,40320))`; an arbitrary exact cover is insufficient.
The blocks must also be a disjoint exact cover, each actual extent must be full,
and every receipt must have the same compatible strategy, operator, probe,
determinism, source, semantics, and trust stratum.

The probe aggregate is a domain-separated SHA-256 over a length-delimited
sequence of pairs `(block_index, full_block_receipt_digest)` in strictly
ascending block index. Concatenating bare result digests is forbidden.

The admission probe set is nonempty and canonically serialized. Uniqueness is
checked on the full pair `(probe_id, canonical_probe_bytes)`, and duplicate
content under a second identifier is rejected. Admission completeness requires
exact equality with the probe set induced by the frozen plan, every probe
complete in one compatible trust stratum, and a domain-separated aggregate of
`(probe_index, full_probe_receipt_digest)` in canonical probe order.

These executable rules are intended composition obligations. This Garden does
not claim a Lean composition theorem, a reusable proof theorem, or completed
V13 admission.

## Reuse Protocol

A stored block may be considered for reuse only when:

1. the requested and stored canonical `WorkDomain` values are exactly equal;
2. both canonical serializations reproduce the same bytes;
3. the stored address recomputes from those bytes;
4. the receipt passes its stage, authority, extent, result, and trust checks;
5. the requested composition stratum accepts that exact trust kind;
6. no conflicting result exists for the same exact domain.

A miss schedules fresh Sounio-authorized work. A valid hit may avoid repeating
that block computation, but it does not change semantic expectations. Two
different results for one exact host-independent domain invalidate reuse and
the enclosing admission. They are not resolved by selecting the newest,
fastest, majority, or target-preferred result.

## Positive and Negative Witnesses

The exact positive structural witness is the induced final block:

    index=63
    start=39690
    width=630
    end=40320

The exact negative structural witness changes only the start:

    index=63
    start=39689
    width=630
    end=40319

It must be rejected as `PARTITION_BOUNDARY_MISMATCH` even though it has positive
width and lies inside the action universe. The first executable must also
exercise deliberate refusals for:

- missing, duplicate, reordered, empty, or content-duplicate probes;
- missing, duplicate, reordered, gapped, overlapping, zero-width,
  out-of-range, partial, or schema-divergent blocks;
- a matching schema version with different induced boundaries;
- a fixed-width interval not divisible by the declared width;
- wrong aggregate order or a bare-result-digest aggregate;
- source, semantics, operator, strategy, probe, partition, numeric,
  determinism, toolchain, hardware, FP, thread, schedule, or address drift;
- different results for one exact host-independent domain;
- incomplete actual extent marked complete;
- observed/parity data occupying an expected slot;
- cross-trust aggregation or native-decide-to-kernel promotion;
- Python or Rust oracle execution;
- C++ or LLM semantic-authority promotion;
- parity before freeze, semantic repair from parity, or CLAIM_READY promotion;
- missing policy, Guardian error, timeout, or absent receipt;
- the pending second U250 counted as installed or as an enumeration failure.

## Hardware Boundary

The canonical material targets remain Xeon, Apple Silicon, DGX through
Kubernetes, and dual AMD Alveo U250. DGX work must never route through Slurm.
The U250 inventory remains:

    declared=2
    installed=1
    pending_installation=1
    enumeration_failure=0

No target runs in this Garden. Hardware identity is receipt context for the
host-independent exact V13 profile and becomes part of `WorkDomain` for any
future material-bound strategy.

## Required First Result

After this Garden is committed, the matcher-free Sounio executable must:

- hash and parse the frozen V13 transcript directly;
- reconstruct exactly the declared 32-probe structural input set;
- derive all 2048 work domains and addresses without expected block results;
- prove operationally that the induced partition is exact by full census;
- serialize and digest the strategy, probe set, partition, and pending plan;
- execute every required positive and negative protocol case;
- emit a receipt in which every actual V13 block remains pending;
- keep reuse, formal/effect/material parity, performance, novelty, and
  claim-ready fields false.

The counts and digests produced by that run are unknown until Sounio emits
them. The first executable may contain the structural constants in this Garden
and exact frozen parent hashes, but no V14 observed matcher.

## What This Is Not

V14 does not yet claim:

- semantic hash injectivity or collision-free equality;
- a Merkle tree, authenticated data structure, or distributed consensus;
- purity or determinism for arbitrary target computations;
- proof composition across trust kinds;
- cache correctness beyond exact-domain operational validation;
- a cache hit, wall-time reduction, throughput gain, or cost result;
- a complete block-scoped V13 canonicalizer;
- formal, effect, material, or denotational parity;
- algorithmic, material, scientific, global, historical, or priority novelty;
- subquadratic twisted-XOR multiplication or canonicalization;
- admission of all V13 records under V14;
- CLAIM_READY.

## Evidence Progression

1. Commit this technical Garden and the GARDEN-status contract.
2. Commit the matcher-free Sounio protocol kernel separately.
3. Authorize and execute only that Sounio source through `./bin/souc`.
4. Commit its first transcript and receipt before adding matchers.
5. Add observed matchers and freeze the V14 semantics by hash.
6. Open parity only after that freeze.
7. Build block execution/reuse and measure performance as later, separately
   claimed stages.

Failure at any stage remains evidence. It cannot be converted into a golden,
waiver, cache hit, or parity success.
