<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-xed-permute-import
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-xed-permute-import
-->

# Pireus XED Permutation Import

Proposed Concept-ID: `SOUNIO-PIREUS-XED-PERMUTE-IMPORT`

Status: `SEMANTICS_FROZEN_PENDING_REGISTRY_AND_LOOM_ACCEPTANCE`

Semantic-Lane-ID: `pireus-xed-permute-import-20260827`

## Intent

Give Pireus its first vendor-corpus ingestion surface without allowing vendor
data, a convenience exporter, or a parity implementation to define Sounio
semantics.

The first slice is intentionally bounded to the eight 512-bit `f64` records for
`VPERMPD`, `VPERMI2PD`, and `VPERMT2PD` in Intel XED release `v2026.08.23`.

## Authority

The semantic authority is:

```text
stdlib/hardware/pireus/xed_import.sio
examples/pireus_xed_permute_import.sio
```

The XED file is normative vendor input. It is not an expected-result producer.
Sounio verifies the complete file digest, applies the accepted grammar, builds
the Pireus triples, evaluates the queries, and produces the expected result.

## Accepted Corpus

```text
upstream=https://github.com/intelxed/xed
release=v2026.08.23
commit=0bcb6237345c5066726dcc08b3d87928df3b5b26
path=datafiles/avx512f/avx512-foundation-isa.xed.txt
bytes=458470
sha256=e9bc0522be4c1a3a3d938eab334c47e306fe759cccf376b9dfb2b9cf7aee0038
license=Apache-2.0
```

The raw vendor file is not copied into the Sounio repository. The executable
accepts its path as its only argument and refuses any digest mismatch.

## Frozen Grammar

For the three selected `ICLASS` values, a record must contain exactly one of
each field below:

```text
ICLASS CPL CATEGORY EXTENSION ISA_SET EXCEPTIONS REAL_OPCODE
ATTRIBUTES PATTERN OPERANDS IFORM
```

The accepted field values are the exact eight record shapes present in the
pinned corpus. An unknown field, duplicate field, missing field, mismatched
`PATTERN`/`OPERANDS`/`IFORM`, wrong ISA set, or wrong complete-file digest is a
closed failure.

Records with other `ICLASS` values are outside this slice and are ignored. They
do not become negative architecture facts.

## Preserved Distinctions

The importer preserves these raw distinctions:

- `ICLASS` family;
- `ISA_SET=AVX512F_512`;
- destination access `w` versus `rw`;
- mask read operand;
- `zf64` register width and `f64` memory element spelling;
- register versus memory source form;
- `UIMM8` versus register-index syntax for `VPERMPD`;
- source release, commit, file, byte length, digest, and evidence role.

`VPERMI2PD` and `VPERMT2PD` remain `raw_selector_syntax=unassigned`. Their raw
operands are retained, but operand position is not promoted to selector,
payload, merge, or table semantics by this importer.

## Ontology Projection

Each accepted record becomes an `InstructionForm` individual connected to:

```text
VendorCorpus
ICLASS
ISASet
DestinationAccess
RawSelectorSyntax
StorageKind
NormativeVendorRecord evidence
```

The projection extends the frozen v0.1 store; it does not mutate the v0 or v0.1
models.

## Negative Witnesses

The Sounio executable deliberately verifies three denials:

1. one modified corpus byte is refused by the SHA-256 gate;
2. `CPL` renamed to an unknown selected-record field is refused;
3. a selected record with its `IFORM` line removed is refused.

No Python, Rust, Node, Ruby, `awk`, or `bc` participates in the semantic or
expected-result path.

## Claim Boundary

This concept establishes a pinned, fail-closed Sounio ingestion and ontology
projection for eight records. It does not establish instruction behavior,
encoding correctness, availability on any Darwin machine, throughput,
latency, compiler lowering, equivalence among the three instruction families,
or a Cayley-Dickson speedup.

`PARITY_OPEN` remains closed until the Concept-ID is registered and Loom
accepts the frozen Sounio receipt.
