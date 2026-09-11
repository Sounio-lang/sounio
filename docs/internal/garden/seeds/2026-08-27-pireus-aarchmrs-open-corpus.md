<!-- docs:meta
topic_id: repo.docs.internal.garden.seeds.2026-08-27-pireus-aarchmrs-open-corpus
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.garden.seeds.2026-08-27-pireus-aarchmrs-open-corpus
-->

# Pireus: The Open Arm Chart

> **Status**: Garden | **Date**: 2026-08-27 | **Authority**: founder direction

## Butterfly

Pireus needs an Arm chart that can be ingested by Sounio without turning a
vendor example program, a disposable script, or a parity implementation into
semantic authority.

Arm publishes an open A-profile Architecture Machine Readable Specification
package containing the three structured layers Pireus needs first:

```text
Features.json     architecture feature constraints
Instructions.json A64, A32, and T32 instruction-set records
Registers.json    architectural registers and system instructions
```

## Pinned Corpus

The chosen package is the official open, non-confidential distribution:

```text
archive=AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12.tar.gz
url=https://developer.arm.com/-/cdn-downloads/permalink/Exploration-Tools-OS-Machine-Readable-Data/AARCHMRS_BSD/AARCHMRS_OPENSOURCE_A_profile_FAT-2025-12.tar.gz
observed_http_status=200
observed_content_length=5371270
observed_last_modified=Mon, 15 Dec 2025 14:05:58 GMT
archive_sha256=4dc5da62a5c856d7b1086b895075f54807f821ea21a333049cb0f40f9479cecc
```

Its embedded instruction metadata is:

```text
architecture=vFATAp1-A
build=518
ref=2025-12_rel
schema=2.7.4
timestamp=2025-12-10 16:11:30
```

The extracted authority surfaces are pinned independently:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Features.json` | 1,187,831 | `d61b4987c3ad44941a3871a578636dfdd0f45b5a909bd5cf29bcb0fb6b4f5751` |
| `Instructions.json` | 115,312,065 | `bedf5f8fc142d6232f15caaa170b9fab996a732db0b04bf4604e91fb10c3244b` |
| `Registers.json` | 93,461,804 | `fe5d92734031cb61af4c00f63b63d29c6934dbf0091730c03e9c43cdfc7c4baf` |
| `README.md` | 1,242 | `514e17170d5f0065386051553bc46d3f246d4616aa9db7e5a63af0d2045b55d1` |
| `docs/notice.html` | 3,666 | `226329bc6900d775b5fa9f1a2256354c9aab38f95f551d65a13dfb8360277291` |

The raw package is not added to Git. The pin is enough to reacquire and verify
the exact bytes while corpus storage, release cadence, and provenance remain
under design.

## License And Quality Boundary

The embedded notice licenses this open package under the BSD 3-clause license.
Its README says the architectural content has the same quality as the
equivalent XML releases, while also warning that the schema remains under
development and can change.

This package is intentionally preferred over the proprietary
`AARCHMRS_A_profile` package for the first Pireus importer. The open package
omits descriptive or not-yet-machine-readable proprietary content, but adds
machine-readable instruction and register surfaces that can be retained with
their license terms.

## First Harbor Slice

The first Sounio executable will inspect `Instructions.json` and select the
`TBL`/`TBX` permutation family. That slice is small enough to study deeply but
crosses important Arm distinctions:

- A64 Advanced SIMD versus SVE/SME forms;
- fixed-width versus scalable vector operands;
- write-only table lookup versus destination-preserving table extension;
- feature conditions, operation IDs, encodings, and assembly symbols.

No record count, operand role, family partition, or semantic equivalence is
frozen here. Raw searches found multiple `TBL` and `TBX` literals in shared
assembly rules and instruction objects; only the Sounio parser may decide which
occurrences become Pireus instruction forms.

In particular, this Garden seed does not claim that Arm `TBL` or `TBX` is
equivalent to Intel `VPERMPD`, `VPERMI2PD`, or `VPERMT2PD`, nor that either is a
valid Cayley-Dickson lowering.

## Required Sounio Contract

The importer must:

1. read the complete pinned `Instructions.json` byte stream;
2. verify its byte length and SHA-256 in Sounio;
3. parse JSON structure rather than count substrings;
4. distinguish instruction objects from shared literals and rules;
5. retain raw vendor fields before assigning Pireus semantic roles;
6. reject unsupported schema, malformed structure, duplicate required fields,
   unknown selected-record shapes, hash drift, and capacity overflow;
7. emit the first executable inventory and expected result in Sounio;
8. keep parity, performance, and lowering claims closed until the Sounio stream
   is frozen and accepted by Loom.

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
