<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-ptx-prmt-corpus
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-ptx-prmt-corpus
-->

# Pireus: The PTX `prmt` Chart

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Pireus needs an NVIDIA chart for canonical DGX targets, but it must not collapse
three different objects into one:

```text
PTX virtual instruction != target SASS instruction != DGX material capability
```

The first chart slice is PTX `prmt`. It is close enough to the permutation
question that motivated Pireus to be useful, while still forcing the ontology
to represent virtual-ISA version, target requirement, selector form, source
width, and translation boundary explicitly.

## Pinned Corpus

The first executable will consume the official archived CUDA 13.2.0 PTX ISA
9.2 HTML. The versioned archive URL is preferred over the live PTX 9.3 page so
the initial Sounio semantics can be reproduced from the same upstream bytes.

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

pdf_url=https://docs.nvidia.com/cuda/archive/13.2.0/pdf/ptx_isa_9.2.pdf
pdf_observed_http_status=200
pdf_observed_content_type=application/pdf
pdf_observed_content_length=20208675
pdf_observed_last_modified=Sat, 04 Apr 2026 19:39:05 GMT
pdf_observed_etag="7739d4ba7b8b7ec2041a064cc7beb45a"
pdf_sha256=6d136dbaa3f72bc82e42593c5a1a214977cfc4eeba88282b76f284c06f26e254
```

The HTML and PDF are two renderings of the vendor document, not independent
semantic votes. The HTML is the parser input; the PDF is a provenance cross-
reference only. Neither raw file is added to Git.

On 2026-08-27 the live NVIDIA URL reported PTX ISA 9.3. Its PDF was observed as
20,296,990 bytes with SHA-256
`769cc12e363e7bb9cd311464f2a42edbdaf4c74ca18575de8c358e93223434fc` and
`Last-Modified: Wed, 26 Aug 2026 23:53:33 GMT`. That live snapshot is not the
first importer input and cannot silently replace the pinned 9.2 bytes.

## License And Retention Boundary

The vendor notice says that the document is provided as-is, reserves changes,
does not grant an intellectual-property license, and restricts reproduction.
Pireus therefore records only source coordinates, transport metadata, hashes,
and Sounio-derived projections in Git. It does not redistribute the NVIDIA HTML
or PDF. Any later corpus repository must preserve this non-redistribution
boundary unless NVIDIA supplies different terms.

## First Harbor Slice

The first Sounio executable will inspect exactly the HTML section whose ID is:

```text
data-movement-and-conversion-instructions-prmt
```

The candidate projection includes the raw section identity, title, syntax
block, mode tokens, prose blocks, PTX-version note, target requirement, and
examples. The source visibly distinguishes a generic form from specialized
modes, but no mode count, operand role, selector semantics, minimum target, or
example count is frozen by this Garden seed. Those expected results must be
born from the Sounio parser.

This seed does not claim:

- that PTX `prmt` is a vector-lane permutation;
- that PTX `prmt` maps one-to-one to a SASS instruction;
- that a DGX GPU supports a particular physical instruction because PTX accepts
  `prmt`;
- that `prmt` is equivalent to Intel `VPERMPD`/`VPERMI2PD`/`VPERMT2PD` or Arm
  `TBL`/`TBX`;
- that `prmt` is a valid or optimal Cayley-Dickson lowering.

## Required Sounio Contract

The importer must:

1. read the complete pinned HTML byte stream;
2. verify its byte length and SHA-256 in Sounio;
3. parse HTML structure rather than count text substrings;
4. locate exactly one selected section and stop at its structural boundary;
5. retain raw vendor text before assigning Pireus semantic roles;
6. reject hash drift, duplicate selected sections, malformed or unsupported
   structure, missing required blocks, unknown selected-section shapes, empty
   digest, and capacity overflow;
7. emit the first executable inventory and expected result in Sounio;
8. keep SASS equivalence, hardware availability, cost, lowering, parity, and
   claim readiness closed until separately evidenced.

Python and Rust are prohibited. Node, Ruby, shell text processing, `awk`, `bc`,
or another disposable language may transport or inspect material bytes, but
may not create the semantic projection or expected result.

## Mandatory Order

```text
GARDEN
-> SOUNIO_EXECUTABLE
-> SEMANTICS_FROZEN
-> PARITY_OPEN
-> CLAIM_READY
```

This seed establishes only `GARDEN`.
