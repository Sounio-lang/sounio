<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-xor-selector-material-admission
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-xor-selector-material-admission
-->

# Pireus XOR Selector Material Admission

Concept-ID: `SOUNIO-PIREUS-XOR-SELECTOR-MATERIAL-ADMISSION`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-xor-selector-material-admission-20260827`

## Intent

Admit three already sealed, target-local material receipts into a typed Sounio
evidence overlay without letting C++, compiler output, hardware, vendor operand
encodings, or an external reviewer revise the frozen XOR operation semantics.

The semantic producer is Sounio:

```text
stdlib/hardware/pireus/xor_selector_material_admission.sio
examples/pireus_xor_selector_material_admission.sio
tests/stdlib/hardware/test_pireus_xor_selector_material_admission.sio
```

## Causal Order

```text
GARDEN commit=b53115358687f2d660d3bc5596f07a37aa4929fb
SOUNIO_EXECUTABLE commit=fdd444afc5ba0e7529bfee532640dc0a665bfc3f
SEMANTICS_FROZEN=enclosing Git commit
PARITY_OPEN=false
CLAIM_READY=false
```

No parity implementation ran before the exact Garden and first Sounio
executable existed.

## Parent Closure

The child live-validates the frozen five-node lowering parent and live-hashes
the six files carrying the three material receipt/evidence pairs. The nine
required parents are the three lowering artifacts plus those six material
files. The ordered manifest additionally binds the Garden source, for ten
artifacts total:

```text
parent_artifacts=10
required_parent_hashes=9
parent_manifest_sha256=23eeef8d222c99674bc3a3f92ea5cb46772fc5d7a58ed74af36469a9f32ef712
```

The Garden commit is also stored as five SHA-1 words and enters the aggregate
admission digest. A file mismatch becomes a typed receipt binding error before
admission.

## Coverage Result

| Target | Receipt | Admitted nodes | Unresolved | Refused | Whole operation |
| --- | --- | ---: | ---: | ---: | --- |
| Darwin Xeon | admitted | 5 | 0 | 0 | true |
| Apple Silicon | admitted | 1 | 4 | 0 | false |
| DGX | admitted | 1 | 4 | 0 | false |

All three receipts explicitly witness `XOR_PERMUTE`. Only the Darwin Xeon
receipt names sufficient evidence for the other four semantic nodes.

The DGX receipt reports one resolved-but-unnamed node count. That count is
preserved as data but does not identify or admit any of nodes 1 through 4.
They remain `UNRESOLVED`.

## Target Identity Boundary

The Apple material parent records the tailnet identity
`sounio-language-macbook`, host `Sounio-Language-MacBook`, model `Mac17,7`,
and Apple M5 Max hardware. This admission execution only binds that sealed
receipt and evidence; it does not turn the Linux authority host into an Apple
observation.

Likewise, no DGX identity is inferred from an IP address or from the
unresolved-node count. Target identity must arrive in a sealed material
receipt.

## Encoding And Cost Boundary

The DGX material parent distinguishes the frozen abstract operand `c=15` from
the emitted PTX/SASS operand `c=4127` (`0x101f`). Coordinate parity is admitted
while operand-encoding equality remains false.

Static selector-site observations remain target-local facts:

```text
darwin_static_selector_sites=3
apple_static_selector_sites=1
dgx_ptx_selector_sites=32
dgx_sass_selector_sites=32
generic_instruction_cost=false
cross_isa_equivalence=false
```

No count becomes latency, throughput, scheduling cost, or a minimum lowering.

## Digest Contract

Each target digest binds every field emitted by its admission record,
including binding error, node statuses, aggregate node counts, whole-operation
coverage, operand encoding, static sites, and reproducibility flags.

The aggregate digest additionally binds the Garden commit, parent binding,
three target digests, six live-file matches, fourteen aggregate integers,
seven boundary booleans, and all twenty-two negative witnesses. The seven are
`material_files_valid`, `generic_instruction_cost`, `cross_isa_equivalence`,
`transform_authorized`, `review_promoted`, `parity_open`, and `claim_ready`.

## Closed Claims

This freeze does not establish:

- a new hardware measurement or compiler emission;
- cross-ISA semantic or performance equivalence;
- generic instruction cost, latency, throughput, energy, or speedup;
- a complete Apple Silicon or DGX lowering;
- operand-encoding equality for DGX;
- a transform, Walsh-Hadamard, or subquadratic result;
- Lean, Koka, C++, or Haskell parity for this admission overlay;
- authority for any external LLM review;
- `PARITY_OPEN` or `CLAIM_READY`.

The external Loom guardian remains the stage and producer-language authority.
