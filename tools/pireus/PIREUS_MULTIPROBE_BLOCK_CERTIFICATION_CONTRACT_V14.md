# Pireus Multi-Probe Block Certification V14 Contract

Status: GARDEN

Concept-ID: SOUNIO-PIREUS-MULTIPROBE-BLOCK-CERTIFICATION

Semantic-Lane-ID: pireus-multiprobe-block-certification-v14-20260831

## Contract

V14 makes certification strategy a first-class, target-neutral Pireus
ontology object. It defines how one frozen operator and work function induce a
canonical probe set, block partition, work addresses, evidence bindings, and
aggregate admission. It does not change operator semantics and does not yet
claim that any actual V13 block has been completed or reused.

The mandatory order is:

    GARDEN
    -> SOUNIO_EXECUTABLE
    -> SEMANTICS_FROZEN
    -> PARITY_OPEN
    -> CLAIM_READY

The first executable producer will be
`stdlib/hardware/pireus/multiprobe_block_certification.sio`, created only after
the Garden commit. Sounio is SEMANTIC_AUTHORITY. Lean, Koka, C++, Haskell,
targets, caches, and LLMs cannot create V14 expected results.

## Parent Boundary

The semantic work profile is derived from the hash-frozen V13 Sounio result:

    parent_concept=SOUNIO-PIREUS-OPERATOR-ORBIT-CANONICALIZATION
    parent_matcher_free_source_sha256=3136968a83bbba18d56c543895d6bbd9530ccf6c59db78ac6b6f2fa3bd26c9e4
    parent_frozen_matcher_source_sha256=7ada1b17bf91fdb3f4c48877d2485f71a65bb4159d88cb7e4b288c77bfe3cdae
    parent_first_transcript_sha256=16af63f5e0f8aa7e5c899f4c395404b83fb402f6bbdb5f20dea2a3d10ad2e19f
    parent_first_receipt_sha256=be0ef127e7c40cf0167cb55189c39245dbfd93ffed990d64a003cadf3f19f38b
    parent_semantics_sha256=0a4b2c35fde7724ea687055f1d8a5628353ed35147b90ad909a7e9fd528f9f7c

The ontology parent is Operator Genome V3 semantic identity
`99d5e3417550ad3f7b8223b29f25b3d8d8616ac425d4615c37c7f2f402668926`.

The V14 executable must read, SHA-256 hash, and parse the V13 first transcript
inside Sounio. External preprocessing and parity reconstruction are forbidden.
The frozen V13 values are parent inputs; they cannot be presented as new V14
expected outputs.

## Required Types

The first Sounio representation must distinguish at least:

    CertificationStrategy
    WorkFunctionIdentity
    ProbeIdentity
    ProbeSet
    PartitionSchema
    WorkDomain
    WorkAddress
    ExpectedDigest
    ObservedDigest
    BlockObligation
    BlockReceiptBinding
    ProbeReceiptBinding
    AdmissionReceiptBinding
    ProofBinding
    TrustKind
    DeterminismMode
    ClaimBoundary

`ExpectedDigest` and `ObservedDigest` must not be aliases. Construction of an
expected digest requires producer language Sounio, role SEMANTIC_AUTHORITY,
and an admissible stage. A proof or observation may compare against expected
data but may not be cast into it.

## Identity Rules

`WorkDomain` is job identity. It contains every result-affecting semantic
input, including exact source/semantics/operator/strategy/probe/partition,
numeric, determinism, toolchain, and material-bound execution identities.

`WorkAddress` is SHA-256 over a domain-separated canonical serialization of
`WorkDomain`. It is an address and integrity binding only. Reuse requires full
decoded field equality and identical canonical bytes before the address is
recomputed. Address equality is never a semantic equality predicate.

`ResultDigest` binds the complete result payload of one fully executed extent.
It is not job identity. A block receipt binds both job and result identities.

The canonical serialization must be total, tagged, length-delimited, and
independent of textual formatting, host size, memory address, locale, map
iteration order, or path aliases. Every integer width, signedness, byte order,
sum tag, sequence length, and domain-separation tag is explicit.

## Determinism Rules

The allowed modes are:

    HOST_INDEPENDENT_EXACT
    MATERIAL_BOUND

The initial V13 finite Boolean/integer scan uses
`HOST_INDEPENDENT_EXACT`. Hardware is recorded in the receipt but cannot alter
the expected result. Divergent results for one exact domain create
`SAME_DOMAIN_RESULT_CONFLICT` and invalidate reuse and every enclosing
aggregate.

A computation without a host-independent exact norm must use
`MATERIAL_BOUND`. Every result-affecting hardware, FP, thread, schedule,
driver, and toolchain field then belongs to `WorkDomain`; omitting one is a
closed-domain failure.

Reuse remains false before the work function and these rules are frozen. This
contract does not assert a universal purity theorem.

## V13 Partition Profile

The structural profile is:

    probes=32
    work_interval=[0,40320)
    blocks_per_probe=64
    fixed_block_width=630
    logical_block_jobs=2048

For `k` in `[0,64)`, the only admissible block is:

    index=k
    start=630*k
    width=630
    end=630*(k+1)

Completeness requires equality with the block set induced by this schema, not
merely some disjoint exact cover. Each actual scanned extent must equal its
declared extent and visit every required action exactly once under the frozen
V13 order.

The required positive witness is block 63 at `[39690,40320)`. A block 63 at
`[39689,40319)` is the required negative witness and must return
`PARTITION_BOUNDARY_MISMATCH`.

## Block Completion

A block receipt is complete only when all of these hold:

- its domain is exactly the induced domain for its index;
- its canonical bytes and recomputed address match;
- its actual start, end, width, and visit count equal the declared extent;
- its result payload is complete and its digest recomputes;
- producer language, role, stage, source, semantics, and trust kind are valid;
- toolchain and material context satisfy the selected determinism mode;
- no source, semantic, domain, result, conflict, parity, or claim flag fails.

The receipt records language, role, trust kind, source and semantic hashes,
toolchain, hardware, command, policy decision, result, and claim boundary as
required by the founder contract.

## Composition

For one probe, the block set must exactly equal the schema-induced set, be a
disjoint exact cover, carry full actual extents, and use one compatible trust
stratum. The canonical probe aggregate is:

    SHA256(
      PIREUS_V14_PROBE_AGGREGATE ||
      schema || probe_identity || block_count ||
      ordered_length_delimited(
        block_index || full_block_receipt_digest))

where order is strictly ascending block index. Bare result-digest
concatenation is forbidden.

The admission probe set is nonempty, canonically ordered, and unique by full
identity `(probe_id, canonical_probe_bytes)`. Duplicate content under another
identifier is rejected. The set must exactly equal the frozen plan's induced
probe set. Its aggregate uses an analogous domain-separated encoding of
`(probe_index, full_probe_receipt_digest)` in canonical probe order.

Cross-trust aggregation is forbidden unless a later frozen transport rule
states and discharges the exact conversion. V14 initially defines no such
transport.

## Trust Strata

The protocol recognizes distinct bindings:

    SOUNIO_EXECUTION_RECEIPT
    LEAN_KERNEL_PROOF
    LEAN_NATIVE_DECIDE_REPORT
    KOKA_EFFECT_PARITY
    CPP_MATERIAL_PARITY
    HASKELL_DENOTATIONAL_OBSERVATION

The word `certificate` names the protocol family. It does not promote an
execution receipt into a proof, native decision into kernel reduction, effect
parity into semantic authority, or material parity into expected output.

## Reuse Decision

A stored block may be reused only after:

1. exact requested/stored `WorkDomain` equality;
2. canonical serialization byte equality;
3. address recomputation;
4. complete receipt validation;
5. compatible trust stratum validation;
6. absence of any same-domain result conflict.

An invalid hit is a hard refusal. It cannot be repaired by recomputing only an
aggregate, choosing a majority result, preferring a target, or importing a
parity value. A miss may schedule fresh Sounio-authorized work.

## First Executable

The matcher-free Sounio source must:

- validate the frozen V13 transcript hash and parse all 32 admitted records;
- derive the 32-probe canonical set without external preprocessing;
- induce and validate all 64 blocks per probe;
- construct all 2048 pending work domains and addresses;
- derive strategy, probe-set, partition, and plan digests;
- execute the complete positive and negative protocol surface;
- emit zero completed V13 blocks and zero reuse hits;
- emit no expected V13 block result digest;
- keep parity, performance, novelty, and CLAIM_READY false.

The structural constants and parent hashes in this contract are admissible
source inputs. Every V14 output digest, negative census, diagnostic, and result
is unknown until the first Sounio execution. No matcher for them may exist in
the first executable commit.

## Mandatory Negative Surface

The executable refuses:

- empty, missing, duplicate, reordered, or content-duplicate probes;
- missing, duplicate, reordered, gapped, overlapping, zero-width,
  out-of-range, partial, nondivisible, or wrongly induced blocks;
- matching schema labels with different boundaries;
- wrong aggregate order or aggregate encoding;
- source, semantics, operator, strategy, probe, partition, numeric,
  determinism, toolchain, material, FP, thread, or schedule drift;
- wrong canonical bytes, address, result digest, or actual extent;
- nondeterministic divergence for one host-independent exact domain;
- parity/observed data in an expected slot;
- cross-trust aggregation or proof-kind promotion;
- Python, Rust, C++ authority, LLM authority, or review promotion;
- parity before freeze, result repair from parity, or early CLAIM_READY;
- policy absence, policy error, timeout, or missing receipt;
- raw-ELF execution instead of `./bin/souc`;
- DGX routing through Slurm;
- U250 inventory other than declared 2, installed 1, pending 1, enumeration
  failures 0.

## Claim Boundary

The first V14 result must keep false:

    actual_block_execution_complete
    any_block_reused
    all_v13_probes_certified
    deterministic_purity_formally_proved
    collision_free_semantic_identity
    formal_parity_open
    effect_parity_open
    material_parity_open
    denotational_parity_open
    performance_measured
    speedup_admitted
    cache_hit_rate_admitted
    algorithmic_novelty
    material_novelty
    scientific_novelty
    global_novelty
    historical_novelty
    priority_claim
    claim_ready

The first result may establish only that Sounio derived and validated the
frozen structural plan and protocol rules. It cannot establish that the plan
has already saved work.

## Evidence Progression

1. Commit the technical Garden and this GARDEN-status contract.
2. Commit the matcher-free Sounio executable separately.
3. Obtain pre-execution Guardian ALLOW and run through `./bin/souc`.
4. Commit the first transcript and receipt before adding any expected matcher.
5. Add only observed matchers, replay, and freeze V14 semantics by hash.
6. Open Lean, Koka, C++, and optional Haskell parity only after freeze.
7. Implement actual block runners and cache materialization in a later stage.

Guardian error, timeout, missing policy, malformed parent, structural mismatch,
or negative-control failure returns a typed nonzero result and cannot be
silently converted into a plan, receipt, cache hit, waiver, or success.

## Hardware Boundary

Canonical material targets remain Xeon, Apple Silicon, DGX via Kubernetes,
and dual AMD Alveo U250. The second U250 is pending installation, not missing
from the declared topology and not an enumeration failure. No hardware target
runs during GARDEN, and no material observation creates semantic authority.

## Review Boundary

Math review is mandatory and REVIEW_ONLY. The pre-Garden xAI review identified
and caused correction of the determinism/domain-closure and schema-induced
exact-cover blockers. Z.AI provider error 1313 is logged as an incomplete
second opinion, never a pass. No reviewer supplies a V14 digest, block result,
speedup, novelty verdict, or authority decision.
