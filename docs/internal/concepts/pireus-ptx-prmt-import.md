<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-ptx-prmt-import
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-ptx-prmt-import
-->

# Pireus PTX `prmt` Import

**Concept-ID:** `SOUNIO-PIREUS-PTX-PRMT-IMPORT`
**Status:** executable candidate; Loom acceptance required
**Owner:** Pireus hardware ontology
**Semantic authority:** Sounio

## Boundary

This concept is the first grammar-bounded NVIDIA corpus projection in Pireus.
It imports one vendor-document section from the archived CUDA 13.2.0 PTX ISA
9.2 HTML:

```text
data-movement-and-conversion-instructions-prmt
```

The concept represents a normative **virtual-ISA document record**. It does not
represent a physical NVIDIA instruction, a DGX capability observation, or a
compiler lowering.

```text
PTX form != SASS form != material capability != selected lowering
```

## Source Contract

| Field | Frozen value |
| --- | --- |
| Release | CUDA 13.2.0 |
| PTX ISA | 9.2 |
| HTML bytes | 3,428,895 |
| HTML SHA-256 | `fd013df0c9560d9f86672c379b57b30a6d5efb2eccbb0c6c487950032e6d3457` |
| Transport | four ordered chunks: 1,000,000, 1,000,000, 1,000,000, 428,895 bytes |

The importer in `stdlib/hardware/pireus/ptx_import.sio` hashes the complete
stream in Sounio before accepting the projection.

## Frozen Projection

The executable admits exactly:

- one selected `section` with one `h4` heading;
- the seven expected rubric identities carried by the selected vendor shape;
- one syntax, semantics, PTX-notes, target-notes, and examples region;
- the raw mode tokens `f4e`, `b4e`, `rc8`, `ecl`, `ecr`, and `rc16`;
- the vendor note that the instruction was introduced in PTX ISA 2.0;
- the raw target requirement `sm_20 or higher`;
- one Pireus instruction-form individual linked to one corpus, one virtual ISA,
  one raw target requirement, and six raw mode individuals.

Raw here means vendor-projected and unpromoted. No semantic operand role is
assigned by this version.

## Fail-Closed Conditions

The executable refuses:

- missing files, wrong chunk lengths, total-length drift, or SHA-256 drift;
- malformed or unbalanced HTML structure;
- duplicate selected sections or duplicate `id` attributes;
- missing or unknown selected-section shape;
- tag, paragraph, code-token, or preformatted-block capacity overflow;
- an empty or otherwise non-matching digest.

The deliberate negative witnesses execute in Sounio. Python and Rust are not
used, and disposable languages do not supply expected results.

## Evidence State

| Stage | State |
| --- | --- |
| `GARDEN` | Established by commit `71fcaa0201`. |
| `SOUNIO_EXECUTABLE` | Implemented by the importer and example witness. |
| `SEMANTICS_FROZEN` | Proposed by the paired semantics and receipt; requires Loom acceptance. |
| `PARITY_OPEN` | False. |
| `CLAIM_READY` | False. |

Registration must not promote this concept beyond the accepted evidence stage.
