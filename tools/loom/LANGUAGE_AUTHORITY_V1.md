# Loom Language Authority V1

Frame `9020` is the executable boundary between the Sounio-owned language
authority policy and later native enforcement. The Sounio source is normative.
This document describes the wire form; it does not create decisions.

## Evidence Progression

| ID | Stage |
| ---: | --- |
| 1 | `GARDEN` |
| 2 | `SOUNIO_EXECUTABLE` |
| 3 | `SEMANTICS_FROZEN` |
| 4 | `PARITY_OPEN` |
| 5 | `CLAIM_READY` |

Successful transition actions are `1=GARDEN_RECORD`, `2=SOUNIO_EXECUTE`,
`3=FREEZE_SEMANTICS`, `4=PARITY_EXECUTE`, `5=REVIEW`,
`6=GUARD_ENFORCE`, `7=CLAIM_PROMOTE`, `8=RECEIPT_SEAL`, `9=WRITE`,
`10=COMMIT`, `11=CI`, and `12=RUNTIME_REALIZE`.

## Languages And Roles

Languages are `1=Sounio`, `2=Lean4`, `3=Koka`, `4=C++`, `5=Haskell`,
`6=external LLM`, `7=Python`, `8=Rust`, `9=OCaml`, and `10=other`. Roles are
`1=SEMANTIC_AUTHORITY`, `2=FORMAL_PARITY`, `3=EFFECT_PARITY`,
`4=MATERIAL_PARITY`, `5=OPTIONAL_DENOTATIONAL_BASELINE`, `6=REVIEW_ONLY`,
`7=PROHIBITED`, and `8=OPERATIONAL_REALIZATION`.

Python and Rust are refused before waiver evaluation. An indirect launcher must
therefore classify the resolved interpreter rather than its cosmetic command
name. External LLMs can execute only `REVIEW` and any attempt to promote their
receipt is refused.

OCaml may realize the existing Loom runtime and native Guardian only after a
frozen Sounio semantics hash exists and the request binds that hash as its exact
parent. It cannot create semantic or expected-result writes. Native Guardian
admission additionally requires the transitional, declarative-only, and
Sounio-fixture-matched attestations. C++ retains the same narrow Guardian
bootstrap permission, but OCaml is the selected operational implementation.

## Frame

The frame is one ASCII line of unsigned decimal fields:

```text
9020 stage action language role policy_state
semantic_write expected_result_write parity_receipt_valid review_promoted
exception_requested waiver_founder waiver_scope waiver_purpose waiver_unexpired
guardian_transitional guardian_declarative_only guardian_sounio_fixture_match
source_sha256[8] semantics_sha256[8] parent_semantics_sha256[8]
toolchain_sha256[8] hardware_sha256[8] command_sha256[8]
result_sha256[8] waiver_sha256[8]
```

Each SHA-256 value is eight big-domain `u32` limbs represented as non-negative
`i64` decimal fields. A zero vector means absent. `policy_state` is
`0=missing`, `1=available`, `2=timeout`, or `3=error`.

An execution receipt is not complete until it identifies the Sounio source,
frozen semantics, producing language and role, toolchain, hardware, command,
and result. A parity pre-exec request omits only the not-yet-produced result and
must bind `parent_semantics_sha256` exactly to `semantics_sha256`.

## Decisions

`0` is ALLOW. Denials use stable codes:

| Code | Reason |
| ---: | --- |
| 101 | policy missing |
| 102 | policy timeout |
| 103 | policy error |
| 110 | forbidden language |
| 111 | role mismatch |
| 112 | wrong stage |
| 113 | semantic authority required |
| 114 | expected-result authority required |
| 115 | Sounio source hash missing |
| 116 | semantics hash missing |
| 117 | parity parent hash mismatch |
| 118 | receipt incomplete |
| 119 | review promoted to authority |
| 120 | waiver invalid |
| 121 | native Guardian contract incomplete |
| 122 | parity receipt missing |
| 123 | action forbidden for role |
| 124 | malformed frame |

The executable prints exactly one `SOUNIO_LANGUAGE_AUTHORITY_ALLOW` or
`SOUNIO_LANGUAGE_AUTHORITY_DENY` line and exits with the decision code.
Input `0` runs the Sounio-owned 33-case expected-result suite.

## Migration Boundary

V1 is not a claim that current hooks comply. The current Codex and Claude hooks
still invoke Python. The next layer extends the existing OCaml Loom runtime with
a native Guardian bridge that consumes this frozen Sounio contract, blocks
before process/write/commit/CI effects, and appends ALLOW/DENY receipts. This is
an operational realization of Sounio decisions, not a second semantic kernel.
It remains explicitly transitional; the destination is a Guardian compiled in
Sounio. F# is intentionally outside V1 and may later serve as a scientific
workbench without acquiring semantic authority.
