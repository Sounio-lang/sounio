<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-apple-silicon
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-xor-lowering-apple-silicon
-->

# Garden Seed: Pireus XOR Lowering For Apple Silicon

Date: `2026-08-27`

Concept-IDs: `SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`,
`SOUNIO-PIREUS-AARCHMRS-IMPORT`

Status: `GARDEN`

## Question

Can Sounio derive an Apple Silicon CPU candidate for the frozen bits=4
`XorConvolution` `XOR_PERMUTE` node from pinned A64 architectural semantics,
without importing Intel controls, silently changing `f64` coordinates, or
promoting architectural resemblance into an Apple material observation?

This is a target-specific child of the existing Pireus lowering architecture.
It does not introduce a second operation, guardian, compiler path, ontology,
or lowering contract. It opens the Apple Silicon CPU slice only. Apple GPU and
Metal lowering remain separate canonical surfaces.

## Frozen Parents

The first child must reject drift from both frozen Sounio parents:

```text
lowering_source_sha256=7087649a5cfdb41a884aa9a2e1b0b64bbe2d25da3ca3cd1d54d5b70429854edb
lowering_semantics_sha256=9e92f256c25a774979a1b02cb10d5d39a1ae702ccec3a273044f91930f03a970
lowering_receipt_sha256=daef832ee6370b656e93ae84c76ba6d17c98aaf5ad1dd86674dee27ba0f84346
aarchmrs_source_sha256=ce0693e51f5204f89c67b7917fd129dc1976f069675323ec73d4e2c42913078b
aarchmrs_semantics_sha256=ed66cc2e2fe27ce06842c1ef2091e2f482b8bcb2d4b84e4e649361ca957b7b14
aarchmrs_receipt_sha256=cd64c91c330c9a81e554408a10de4bccbdf9984395ec049c48dc99148aa11934
```

The lowering parent owns the five-node DAG, the complete bits=4 partner table,
the twist masks, and the exact ascending-`i` `f64` fold. The open AARCHMRS
parent owns its frozen inventory of Arm forms and encodings. It assigns no
operand semantic roles and therefore cannot by itself authorize a selector.

## Pinned A64 Semantic Surface

The new primary input is Arm's official A64 ISA XML release:

```text
archive=ISA_A64_xml_A_profile-2025-12.tar.gz
url=https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-A64-ISA/ISA_A64/ISA_A64_xml_A_profile-2025-12.tar.gz
observed_http_status=200
observed_content_length=36357340
observed_last_modified=Mon, 15 Dec 2025 13:59:29 GMT
archive_sha256=845ed227a6692ddb6b602da2ecbbac776620195a9c001ec576ced3a9a53dc26b
tbl_file=ISA_A64_xml_A_profile-2025-12/tbl_advsimd.xml
tbl_bytes=14897
tbl_sha256=48ef32ed67b9824ba39eb58518faec196472c3a574cf1bbe1f3a494811a6cbbe
tbx_file=ISA_A64_xml_A_profile-2025-12/tbx_advsimd.xml
tbx_bytes=14962
tbx_sha256=fa21f8c0784ec327ca9089552d22b55e0eb4b9dd6e0a2eeb078eeed0e203ca79
notice_file=ISA_A64_xml_A_profile-2025-12/notice.xml
notice_bytes=5212
notice_sha256=7f6e2780187dc8eb12b53d97eb435be19597b1af256a84fb44d4b5bd41846747
```

The raw archive and extracted XML are not added to Git. The vendor notice is a
retention boundary: Pireus records the URL, transport observations, hashes,
structural coordinates, and later Sounio-derived projections, but does not
redistribute the corpus.

Human inspection selected `tbl_advsimd.xml` as the first bounded question. It
did not create an expected operation, selector width, table extent, control,
group count, or match verdict. The first Sounio executable must recover those
facts from XML structure and the embedded architectural operation.

## Coordinate Boundary

The frozen Pireus selector is expressed over sixteen logical `f64` elements:

```text
D = I = {0, 1, ..., 15}
partner[d, i] = i XOR d, for every (d, i) in D x I
finite_coverage_cells = 256
```

The A64 source describes its own architectural coordinate system. Sounio must
derive a typed bridge between those coordinate systems before it can test
coverage. The bridge must make element-to-byte expansion, source grouping,
index bounds, out-of-range behavior, destination behavior, and bit preservation
explicit. A human-supplied Intel control table cannot be reinterpreted as an
Arm result.

Coverage equality is equality of the decoded four-bit logical destination
index with `i XOR d` for all 256 cells. It is not equality of raw byte indices.
Payload preservation is a separate obligation over every 64-bit input pattern;
the bridge may not normalize, canonicalize, swap, or invent payload bits.

The Garden asks whether the derived A64 candidate can realize the frozen
partner table. It deliberately does not state how many instructions, source
groups, chunks, controls, or candidate forms are required.

## First Sounio Executable

After this Garden is committed, the first child executable must:

1. bind the six frozen parent hashes and all pinned A64 transport hashes;
2. read and hash the complete selected XML inputs in Sounio;
3. parse XML structure rather than search for prose substrings;
4. retain raw vendor fields, identifiers, encodings, pseudocode, and source
   coordinates before assigning Pireus roles;
5. recover the selected architectural operation and its bounds without using
   the open AARCHMRS inventory as a semantic substitute;
6. derive the typed mapping between logical `f64` indices and the selected
   architectural coordinate system;
7. derive every candidate control and source grouping from that mapping;
8. test all 256 frozen `partner[d, i]` identities against the derived
   semantics using logical-index equality;
9. distinguish local `XOR_PERMUTE` coverage from the other four DAG nodes;
10. preserve the exact ascending-`i` reduction barrier and refuse a tree
    reduction unless a separate Sounio numerical contract is admitted;
11. emit unresolved facts explicitly and create the first expected result in
    Sounio;
12. emit zero Apple device observations, compiler-emission claims, costs, or
    performance claims;
13. leave `PARITY_OPEN=false` and `CLAIM_READY=false`.

No expected control, count, semantic role, coverage verdict, digest, or
lowering authorization may be added before the first Sounio execution emits
it.

## Required Negative Surface

At minimum, the Sounio child must reject:

1. execution before this Garden commit exists;
2. drift in either frozen parent triplet;
3. drift in the A64 archive, selected XML, or notice;
4. duplicate, missing, malformed, or unsupported selected XML structures;
5. a prose substring promoted to an architectural operation;
6. AARCHMRS encodings promoted to operand semantics;
7. an Intel selector control relabeled as Arm;
8. byte-coordinate coverage promoted to `f64` coverage without a complete
   bit-preserving mapping;
9. missing, duplicated, out-of-range, or aliased logical selector cells;
10. a local selector match promoted to the complete five-node operation;
11. a sign, multiply, reduction, or output mechanism imported from memory;
12. a tree reduction promoted to exact ascending-`i` behavior;
13. A64 architectural availability promoted to Apple Silicon support;
14. an Apple family declaration promoted to a device observation;
15. an instruction count, cost, latency, throughput, energy, or speedup claim;
16. a parity language, vendor tool, compiler, disassembler, or external model
    promoted to semantic authority;
17. parity or claim promotion before the Loom transitions;
18. Python or Rust as producer, oracle, or guardian.

## Authority And Material Boundary

The layers remain distinct:

```text
A64 architectural semantics
!= Apple Silicon implementation support
!= compiler emission
!= executed material behavior
```

Only Sounio may create the first candidate semantics and expected result. A
later Apple material receipt must bind the frozen Sounio hash, language and
role, toolchain, exact Apple hardware, command, and result. It may compare or
measure; it cannot amend the frozen semantics.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-xor-lowering-apple-silicon-20260827
Owner: codex/session-01a040f3-2b73-76e2-bbf7-
Concept-IDs: SOUNIO-PIREUS-XOR-LOWERING-LEGALITY; SOUNIO-PIREUS-AARCHMRS-IMPORT
Intent-Preserved: Apple Silicon remains canonical while Sounio derives its CPU selector candidate before material parity
Transformation: frozen target-independent lowering plus pinned A64 XML to an unresolved Apple Silicon CPU candidate
Types-Changed: none in Garden
Effects-Changed: none in Garden
IR-Changed: none
Claims-Introduced: none in Garden
Claims-Forbidden: expected controls; expected counts; selector verdict; whole-operation lowering; Apple observation; compiler emission; cost; performance; cross-ISA equivalence
Assumptions: the frozen parents and pinned vendor bytes remain byte-identical and live-importable
Write-Set: this Garden seed; generated documentation governance metadata
Read-Set: frozen XOR lowering parent; frozen AARCHMRS parent; pinned A64 XML corpus
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
intent_sha256=20ea211c7bae5a886cae1e9e85012765d00716316063204d127fb35e68326e4a
loom_frame_sha256=03617e5db660081cd38f15d94b942052dc74a039efeb178033413f6c4d5b89bd
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

This seed establishes only `GARDEN` for the Apple Silicon CPU slice of
`SOUNIO-PIREUS-XOR-LOWERING-LEGALITY`.
