<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-intel-vpermpd-selector-semantics
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-intel-vpermpd-selector-semantics
-->

# Garden Seed: Pireus Intel VPERMPD Selector Semantics

Date: `2026-08-27`

Concept-ID: `SOUNIO-PIREUS-INTEL-VPERMPD-SELECTOR-SEMANTICS`

Founder phrase:

> se nós ingeríssemos toda a ontologia de códigos dos processadores x86,
> AARch e Mac silicon... tornaria nossa vida infinitamente mais facil

Founder direction: Pireus is the ontology port. Continue the current Pireus
system and its Loom-governed authority order; do not create a second guardian,
compiler, selector semantics, or competing ontology.

Status: `GARDEN`

## Question

What selector semantics does Intel's pinned Software Developer's Manual define
for `VPERMPD`, and can Sounio itself recover a bounded normalized semantic
record from the exact vendor bytes before any parity language, vendor tool,
host PDF parser, or external model is allowed to compare the result?

This Garden admits the source-pinning and first Sounio extraction experiment.
It does not admit an expected selector rule, instruction equivalence, lowering,
cost, availability, emitted use, or speedup.

## Frozen Parents

The child must reject any parent drift from:

```text
xor_material_source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
xor_material_semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
xor_material_receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
xed_import_frozen_content_sha256=5d9a56cd05eb141b24dfa80bbab74f41306bb19a01902c25fb0feeda63265612
xed_import_current_envelope_sha256=d96d6d57ba1e296930caec5f4f0aff8e2898b3b1d5df6bfaacb96a19333266f7
xed_import_current_receipt_sha256=2dfc243381acb8d365112b3b4075ccabf944de6ff081b4626f9a4f693f136af6
```

The material parent proves a complete two-chunk XOR selector layout and keeps
`selector_semantics_receipt_present=false`. The XED parent proves accepted
form presence and preserves selector syntax. Neither parent defines the
runtime behavior of an Intel instruction form.

## Pinned Normative Source

The admitted source is:

```text
publisher=Intel Corporation
manual=Intel 64 and IA-32 Architectures Software Developer's Manual
manual_version=092
volume=2C
volume_title=Instruction Set Reference, V
landing_page=https://www.intel.com/content/www/us/en/developer/articles/technical/intel-sdm.html
landing_page_updated=2026-08-19
pdf_url=https://cdrdv2-public.intel.com/922483/326018-092-sdm-vol-2c.pdf
pdf_bytes=3298744
pdf_sha256=939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
pdf_envelope=%PDF-1.6
```

The landing page and link were selected by repository inspection plus the
official Intel publication surface. That selection establishes provenance
only. It did not create, decode, or confirm a selector-semantic value.

The raw vendor PDF is not committed. Every authoritative execution must load
the exact bytes from an explicitly supplied path and reject a length or digest
mismatch before decoding content.

## Sounio-First Extraction Boundary

After this Garden is committed, the first child executable may generalize the
existing bounded Sounio PDF machinery only enough to admit a named, pinned
Intel profile while preserving the frozen Apple profile and its witnesses.

The first Sounio executable must:

1. bind every frozen parent hash above;
2. load the exact Intel PDF and verify its byte length and SHA-256 in Sounio;
3. validate the admitted PDF envelope and bounded object/page structure;
4. decode only the content needed to identify the unique `VPERMPD` section;
5. resolve the section's text through Sounio-owned PDF font/text machinery;
6. establish unambiguous section start and end boundaries;
7. derive a normalized selector-semantic record from the vendor section;
8. emit the source page/object lineage for every normalized field;
9. emit the canonical normalized record and its digest before expectations
   are added anywhere in the repository;
10. keep instruction-form distinctions explicit;
11. leave `PARITY_OPEN=false` and `CLAIM_READY=false`.

The normalized record may have fields for instruction form, vector length,
destination lane domain, selector source, selector-field rule, selected source
domain, masking behavior, and out-of-domain behavior. This is a schema, not a
set of expected values. Fields absent or ambiguous in the selected section
must remain absent or ambiguous rather than being filled from memory.

No expected selector value, bit slice, lane table, pseudocode result, semantic
digest, instruction sufficiency verdict, or material-match verdict may be
written before the first Sounio execution emits it.

## Existing PDF Boundary

The current `document::pdf_flatedecode` executable is frozen around an Apple
PDF profile with a specific digest, length, `%PDF-1.3` envelope, classic xref
shape, page inventory, and content digest. Those facts must remain valid.

This Garden does not authorize replacing that profile with an unbounded
general-purpose parser. It authorizes a bounded profile distinction or a new
bounded Intel reader, chosen from executable evidence after the Garden commit.
Any unsupported PDF feature must fail closed and remain a classified blocker.

## Required Negative Surface

At minimum, the Sounio child must reject:

1. execution before this Garden commit exists;
2. missing material-parent source, semantics, or receipt;
3. mismatched material-parent source, semantics, or receipt;
4. missing or mismatched XED lineage;
5. missing Intel PDF;
6. wrong Intel PDF byte length;
7. wrong Intel PDF SHA-256;
8. wrong or unsupported PDF envelope;
9. unsupported xref, object-stream, filter, font, or text encoding;
10. missing `VPERMPD` section anchor;
11. multiple unresolved `VPERMPD` section anchors;
12. ambiguous section end boundary;
13. normalized fields without page/object lineage;
14. a selector value imported from memory, a Rust test, or an external model;
15. a host PDF parser result promoted to semantic authority;
16. XED form presence promoted to selector behavior;
17. one `VPERMPD` form's semantics promoted to another form without evidence;
18. a semantic record promoted to instruction equivalence;
19. a semantic record promoted to target availability or emitted use;
20. a Darwin, Apple Silicon, or DGX observation without a receipt;
21. a cost, instruction-count, lowering, or speedup claim;
22. parity or claim promotion before their Loom transitions;
23. Python or Rust as producer, parser, oracle, or guardian.

## Canonical Target Boundary

The Intel vendor semantics are relevant to the canonical Darwin Xeon target,
but a vendor manual is not a hardware observation. The executable must retain:

| Target | Status in this lane | Required later evidence |
| --- | --- | --- |
| Darwin Xeon | vendor semantics pending | exact target capability and emitted/executed instruction receipt |
| Apple Silicon | canonical, unresolved | Apple instruction/material semantic lane |
| DGX | canonical, unresolved | NVIDIA instruction/material semantic lane |

All Darwin CPUs in the frozen target profile are Xeon. Apple Silicon and DGX
remain canonical targets. Canonical declaration is not observation.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: pireus-intel-vpermpd-selector-semantics-20260827
Owner: codex/session-01a040f3-2b73-76e2-bbf7-
Concept-IDs: SOUNIO-PIREUS-INTEL-VPERMPD-SELECTOR-SEMANTICS; SOUNIO-PIREUS-XOR-MATERIAL-MATCHING; SOUNIO-PIREUS-XED-PERMUTE-IMPORT; SOUNIO-PDF-TEXT-PROJECTION
Intent-Preserved: Sounio creates the first executable selector-semantic record from exact vendor bytes without changing the frozen XOR operation
Transformation: pinned Intel PDF bytes to a bounded provenance-bearing VPERMPD selector-semantic record
Types-Changed: none in Garden
Effects-Changed: none in Garden
IR-Changed: none
Claims-Introduced: none in Garden
Claims-Forbidden: expected selector values; instruction equivalence; lowering; availability; emitted use; cost; performance; cross-ISA parity
Assumptions: exact official Intel Volume 2C version 092 bytes remain available at the declared execution path
Write-Set: this Garden seed; concept registry; generated documentation governance metadata
Read-Set: frozen material parent; frozen XED parent; Sounio PDF modules; exact Intel PDF bytes
Positive-Witness: Garden admission before the first child executable
Negative-Witness: deliberate pre-Garden child and prohibited-oracle attempts are denied
Acceptance-Gate: Loom GARDEN admission plus docs registry and semantic coordination gates
Integration-Target: current Pireus ontology and material-matching pipeline
Authoritative-Only-If: exact Intel bytes are decoded by the first Sounio executable after this Garden commit and the resulting semantics are separately frozen by hash
```

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

Lean 4, Koka, C++, Haskell, vendor tools, external models, and hardware probes
cannot create the expected selector semantics. They may compare, prove, or
measure only after the Sounio artifact is frozen and identified by hash.

External LLMs remain `REVIEW_ONLY`. Python and Rust are prohibited. Node may
run only the existing deterministic documentation metadata generator and may
not decode the PDF or compute or confirm selector semantics.

## What This Is Not

This seed is not:

- a claim that a particular selector bit rule is correct;
- a claim that `VPERMPD` realizes the frozen XOR layout;
- a claim that one-source selection is sufficient;
- a claim that another permute form is necessary or unnecessary;
- permission to use the existing Rust JIT test as evidence;
- a general PDF parser project;
- an AVX-512 capability or use observation;
- an Apple Silicon or DGX semantic result;
- formal, effect, material, or denotational parity;
- a performance, novelty, production, or claim-ready assertion.

The next executable bridge is one bounded Sounio extraction of the `VPERMPD`
semantic section from the exact pinned Intel Volume 2C bytes.
