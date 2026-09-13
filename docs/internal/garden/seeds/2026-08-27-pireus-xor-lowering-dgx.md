<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-dgx
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-dgx
-->

# Garden Seed: Pireus XOR Lowering For DGX

Date: `2026-08-27`

Concept-IDs: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`,
`SOUNIO-PIREUS-PTX-PRMT-IMPORT`

Status: `GARDEN`

## Question

Can Sounio derive a DGX candidate for the frozen bits=4 `XorConvolution`
`XOR_PERMUTE` node by mapping the logical XOR action into a pinned PTX warp-lane
operation, while preserving all `f64` bits and refusing to confuse PTX,
target SASS, or DGX material support?

This is a target-specific child of the existing Pireus lowering architecture.
It does not introduce a second operation, guardian, compiler path, ontology,
or lowering contract.

## Frozen Parents

The first child must reject drift from both frozen Sounio parents:

```text
lowering_source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
lowering_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
lowering_receipt_sha256=daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
ptx_prmt_source_sha256=ca2760d539c4602c85841ac8475a9ffd8a2f760313a8169faf99a32956063bba
ptx_prmt_semantics_sha256=1454e6a212f320fbf4194b3cbb220a30abed56fbf5e8041ce076b7dee5cae697
ptx_prmt_receipt_sha256=e68f6edacfa85c48cd3cb51ab4929975a187174b0b1ab980a2c0f0868f5f38fa
```

The lowering parent owns the five-node DAG, the complete bits=4 partner table,
the twist masks, and the exact ascending-`i` `f64` fold. The PTX `prmt` parent
owns only its frozen virtual-ISA projection. It assigns no lowering role and
cannot retrospectively become the expected DGX solution.

## Pinned PTX Semantic Surface

The new bounded question consumes the same official archived PTX document as
the frozen `prmt` parent:

```text
release=CUDA 13.2.0
ptx_isa=9.2
html_url=https://docs.nvidia.com/cuda/archive/13.2.0/parallel-thread-execution/index.html
html_observed_http_status=200
html_observed_content_type=text/html
html_observed_last_modified=Sat, 04 Apr 2026 19:38:39 GMT
html_observed_etag="1cd98e8eb716453c209c1e34fad90980"
html_bytes=3428895
html_sha256=fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457
selected_section_id=data-movement-and-conversion-instructions-shfl-sync
```

The raw HTML is not added to Git. Pireus records the URL, transport metadata,
hash, structural coordinates, and later Sounio-derived projection. The vendor
license and notice boundary of the frozen PTX parent remains in force.

Human inspection selected the `shfl.sync` section because it exposes a
warp-lane coordinate question. It did not create an expected mode, source-lane
formula, member-mask rule, target threshold, payload decomposition, control,
count, or match verdict. Those values must be born in the first Sounio
execution.

## Coordinate Boundary

The frozen Pireus selector is expressed over sixteen logical `f64` elements:

```text
D = I = {0, 1, ..., 15}
partner[d, i] = i XOR d, for every (d, i) in D x I
finite_coverage_cells = 256
```

The PTX source defines its own lane, mask, bound, and payload coordinates.
Sounio must derive a typed bridge between those coordinates before testing
coverage. The bridge must establish the admitted logical-lane subset, source
lane selection, participation conditions, bounds, and bit-preserving transport
of one `f64` payload through whatever PTX operand width the source actually
defines.

Coverage equality is equality of the decoded four-bit logical source index
with `i XOR d` for all 256 cells. It is not equality of raw PTX lane encodings.
The executable must derive and prove that every frozen displacement stays
inside the admitted active-lane subset. Payload preservation is a separate
obligation over every 64-bit input pattern; decomposition may not normalize,
canonicalize, swap, or invent payload bits.

The earlier `prmt` import remains valuable adjacent ontology, but its
within-value byte-permutation geometry is not presumed to implement the
logical lane action. Sounio may prove a relationship or refuse one; the Garden
does not choose the verdict.

## First Sounio Executable

After this Garden is committed, the first child executable must:

1. bind the six frozen parent hashes and the pinned PTX transport hash;
2. read, reconstruct if chunked, and hash the complete PTX HTML in Sounio;
3. parse HTML structure and locate exactly one selected section;
4. retain raw vendor syntax, prose, identifiers, version notes, target notes,
   and source coordinates before assigning Pireus roles;
5. derive the selected operation modes, lane relation, member-mask behavior,
   bounds, payload width, and target conditions from the source;
6. derive a typed mapping from the sixteen logical indices and every frozen
   displacement into the admitted PTX lane domain;
7. derive any required `f64` payload decomposition and reconstruction, then
   prove bit preservation for every component;
8. derive every candidate control and test all 256 frozen `partner[d, i]`
   identities against the derived semantics using logical-index equality;
9. distinguish local `XOR_PERMUTE` coverage from the other four DAG nodes;
10. preserve the exact ascending-`i` reduction barrier and refuse a warp tree
    reduction unless a separate Sounio numerical contract is admitted;
11. emit unresolved facts explicitly and create the first expected result in
    Sounio;
12. emit zero SASS equivalences, DGX observations, compiler-emission claims,
    costs, or performance claims;
13. leave `PARITY_OPEN=false` and `CLAIM_READY=false`.

No expected mode, lane formula, mask, bound, decomposition, control, count,
target threshold, coverage verdict, digest, or lowering authorization may be
added before the first Sounio execution emits it.

## Required Negative Surface

At minimum, the Sounio child must reject:

1. execution before this Garden commit exists;
2. drift in either frozen parent triplet;
3. drift in the pinned PTX HTML or its chunk reconstruction;
4. duplicate, missing, malformed, or unsupported selected sections;
5. a prose substring promoted to virtual-ISA semantics;
6. the frozen `prmt` result promoted to a lane permutation without proof;
7. an Intel or Arm selector control relabeled as PTX;
8. an incomplete or aliased mapping of sixteen logical indices into lanes;
9. an invalid participation mask, inactive source lane, or out-of-bound source;
10. a payload-width conversion that loses, swaps, canonicalizes, or invents
    `f64` bits;
11. missing, duplicated, or wrong frozen selector cells;
12. a local selector match promoted to the complete five-node operation;
13. a sign, multiply, reduction, or output mechanism imported from memory;
14. a warp tree reduction promoted to exact ascending-`i` behavior;
15. PTX acceptance promoted to a SASS equivalence or DGX capability;
16. a DGX family declaration promoted to a device observation;
17. an instruction count, cost, latency, throughput, energy, or speedup claim;
18. a parity language, vendor tool, compiler, disassembler, hardware probe, or
    external model promoted to semantic authority;
19. parity or claim promotion before the Loom transitions;
20. Python or Rust as producer, oracle, or guardian.

## Authority And Material Boundary

The layers remain distinct:

```text
PTX virtual-ISA semantics
!= generated SASS
!= DGX material capability
!= executed material behavior
```

Only Sounio may create the first candidate semantics and expected result. A
later DGX material receipt must bind the frozen Sounio hash, language and role,
toolchain, exact GPU and driver/toolchain surface, command, generated artifact,
and result. It may compare or measure; it cannot amend the frozen semantics.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-xor-lowering-dgx-20260827
Owner: codex/session-01a040f3-2b73-76e2-bbf7-
Concept-IDs: SOUNIO-PIREUS-XOR-LOWERING-LEGALITY; SOUNIO-PIREUS-PTX-PRMT-IMPORT
Intent-Preserved: DGX remains canonical while Sounio derives its warp-lane selector candidate before material parity
Transformation: frozen target-independent lowering plus pinned PTX section to an unresolved DGX candidate
Types-Changed: none in Garden
Effects-Changed: none in Garden
IR-Changed: none
Claims-Introduced: none in Garden
Claims-Forbidden: expected PTX semantics; expected controls; expected counts; selector verdict; SASS equivalence; whole-operation lowering; DGX observation; compiler emission; cost; performance
Assumptions: the frozen parents and pinned vendor bytes remain byte-identical and live-importable
Write-Set: this Garden seed; generated documentation governance metadata
Read-Set: frozen XOR lowering parent; frozen PTX prmt parent; pinned PTX HTML corpus
Positive-Witness: Loom admits this Garden before the first child executable
Negative-Witness: pre-Garden execution and prohibited-oracle attempts are denied
Acceptance-Gate: Loom GARDEN admission plus docs registry and semantic coordination gates
Integration-Target: current Pireus XOR lowering-legality pipeline
Authoritative-Only-If: the first candidate and expected result are emitted by Sounio after this Garden commit and later frozen by exact hash
```

## Loom Garden Admission

The frozen Sounio language-authority runtime admitted this bounded Garden
intent before commit:

```text
intent_sha256=a1a9f39e4e99b3c8a4f85d4ac7698310b0a177c05f67e59497e4b963b8072b7c
loom_frame_sha256=5d7a2e0a06660381e90c16b03cb7a66f3724683787bf0166f649f491dc1a8e8a
decision=SOUNIO_LANGUAGE_AUTHORITY_ALLOW code=0 reason=allow next_stage=GARDEN
```

The same runtime denied a deliberate Python expected-result producer before
interpreter launch:

```text
python_oracle_frame_sha256=08def15b5225980c4870e29854fdfc1a2a0fc27738fd2ddaad377a2aaa4db507
decision=SOUNIO_LANGUAGE_AUTHORITY_DENY code=110 reason=forbidden-language next_stage=GARDEN
python_launch_count=0
```

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Lean 4, Koka, C++, Haskell, vendor tools, compilers, hardware probes, and
external models may act only after an exact Sounio artifact is frozen. External
LLMs remain `REVIEW_ONLY`. Python and Rust are prohibited. Disposable-language
substitution cannot create or confirm the expected result.

This seed establishes only `GARDEN` for the DGX slice of
`SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`.
