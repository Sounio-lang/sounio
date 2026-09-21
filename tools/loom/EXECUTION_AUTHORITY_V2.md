# Loom Execution Authority V2

Frame `9021` is the Sounio-owned pre-execution capability policy for Loom.
It decides whether an already measured request may cross the boundary into a
process, child process, Git commit, or CI execution. The native Guardian may
measure facts, invoke this frame, enforce the result, and write a receipt. It
may not reinterpret or replace the decision.

## Evidence Order

The only valid progression remains:

```text
GARDEN -> SOUNIO_EXECUTABLE -> SEMANTICS_FROZEN -> PARITY_OPEN -> CLAIM_READY
```

V2 requires stage `SEMANTICS_FROZEN` or later for every execution capability.
Parity languages require an explicit `PARITY_OPEN` state. No receipt, waiver,
review, or later language can reconstruct a missing Sounio parent.

## Measured Request

The frame is one ASCII line of unsigned decimal fields:

```text
9021 stage surface execution_class language purpose policy_state
preexec_complete closure_attested semantic_write expected_result_write
review_promoted parity_open receipt_chain_valid exception_requested
waiver_founder waiver_scope waiver_purpose waiver_unexpired
source_sha256[8] semantics_sha256[8] parent_semantics_sha256[8]
toolchain_sha256[8] hardware_sha256[8] command_sha256[8]
result_sha256[8] waiver_sha256[8]
```

SHA-256 values are eight `u32` limbs represented as non-negative decimal
`i64` fields. The result hash may be absent before execution. Source,
semantics, exact parent, toolchain, hardware, and command hashes may not be
absent.

Surfaces are `1=EXEC`, `2=COMMIT`, `3=CI`, and `4=CHILD_EXEC`.
Execution classes are `1=STATIC`, `2=RESOLVED_INTERPRETER`,
`3=DYNAMIC_COMMAND`, `4=NATIVE_BINARY`, and `5=UNCLASSIFIED`.
Purposes are `1=MECHANICAL`, `2=SEMANTIC`, `3=PARITY`, `4=REVIEW`, and
`5=CLAIM_PROMOTION`.

Languages preserve the V1 assignments through `10=other`, with
`11=shell`, `12=Git`, and `13=unclassified`. The authoritative roles are not
encoded as caller-controlled fields: they are derived from language and
purpose by the Sounio policy.

## Fail-Closed Boundary

The request is denied before execution when policy is missing, times out, or
errors; pre-execution measurement is incomplete; the transitive execution
closure is unattested; a dynamic or unclassified command remains; the
language is unclassified; or the frozen receipt chain is incomplete.

Python and Rust are denied before waiver evaluation. Founder waivers must bind
founder identity, scope, purpose, expiry, and a nonzero waiver hash, but cannot
waive either prohibited language. Shell, Git, OCaml, and identified native
tools are mechanical transports only. Child execution must be measured again.

Only Sounio may perform semantic or expected-result writes. Lean, Koka, C++,
and Haskell may run only for parity after `PARITY_OPEN`, against the exact
frozen Sounio parent, with a valid receipt chain. External LLMs are review-only
and cannot cross commit or CI surfaces or promote their output to authority.

## Decisions

`0` is ALLOW. Denials use stable codes:

| Code | Reason |
| ---: | --- |
| 201 | policy missing |
| 202 | policy timeout |
| 203 | policy error |
| 210 | Python or Rust prohibited |
| 211 | language/purpose mismatch |
| 212 | wrong evidence stage |
| 213 | semantic authority required |
| 214 | expected-result authority required |
| 217 | parent semantics hash mismatch |
| 218 | receipt incomplete |
| 219 | review promoted to authority |
| 220 | waiver invalid |
| 222 | parity closed |
| 223 | parity purpose required |
| 224 | malformed frame |
| 225 | pre-execution measurement incomplete |
| 226 | dynamic execution unclassified |
| 227 | execution closure unattested |
| 228 | language unclassified |
| 229 | receipt chain missing |
| 230 | surface forbidden for role |

Input `0` runs the Sounio-owned 32-case expected-result suite. The external
gate additionally plants executable Python and Rust sentinels and performs a
causal sabotage control: removing only Python from the Sounio prohibition must
make the unchanged Python frame ALLOW. That control proves the observed refusal
comes from this rule rather than from an unrelated failure.

## Integration Boundary

Freezing V2 authorizes a native broker to consume frame `9021`; it does not by
itself attach that broker to every shell, commit, and CI surface. Hook rewriting
must replace the original execution with a broker-issued capability, and every
child process must be reclassified. CI must verify the same frozen hashes and
refuse a missing runtime. Until those adapters and their negative fixtures are
green, V2 remains `SEMANTICS_FROZEN`, with `PARITY_OPEN=false` and
`CLAIM_READY=false`.
