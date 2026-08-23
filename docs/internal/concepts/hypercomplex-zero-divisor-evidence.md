<!-- docs:meta
topic_id: repo.docs.internal.concepts.hypercomplex-zero-divisor-evidence
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.hypercomplex-zero-divisor-evidence
-->

# Hypercomplex Zero-Divisor Evidence


Status: **executable**

Concept-ID: `SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE`

## Founder Intent

When nonzero hypercomplex operands annihilate, the zero result must not erase
the ordered operands, signed coefficients, algebra convention, coefficient
domain, detector, or distinction between exact proof and numerical measurement.

## V0 Executable Boundary

The first executable surface is a bounded software reference for ordered pairs
of 16-component sedenions. Each operand has at most two nonzero `i64`
coefficients, each exactly `+1` or `-1`. Multiplication uses the public
`cd_sigma_exact_i64` Cayley-Dickson convention.

`ZDExactTokenV0` and `ZDMeasuredReceiptV0` are distinct types, so a measured
near-zero receipt cannot satisfy a consumer that requires an exact token.
They are not currently an opacity boundary: both compiler engines permit
external struct literals despite non-public fields. Exact-token consumers
therefore revalidate identities, tags, coefficient arrays, nonzero operands,
the ordered exact product, and the checksum. The checksum is diagnostic, not
authority. Inputs outside the bounded no-overflow envelope are classified
before multiplication. Coefficients other than `+/-1` are unsupported; a
magnitude above `1518500249` is specifically overflow risk because four
same-component products can exceed `i64`.

## Canonical Distinctions

```text
zero-divisor element       != ordered zero-divisor pair
a * b == 0                 != b * a == 0
nonzero-factor annihilation != multiplication by zero
signed coefficients        != unsigned support bitmask
exact product equality     != tolerance-gated near-zero
ordered pair               != unordered projective class
status flag                != evidence token
computational evidence     != physical or clinical mechanism
software reference         != ISA, ABI, native lowering, or RTL support
```

The existing primitive census surfaces distinguish 84 participating primitive
vectors, 336 ordered annihilation pairs, and 168 unordered projective classes.
Those counts are bound to
`docs/research/SEDENION_ZERO_DIVISOR_GEOMETRY_REPORT.md`,
`scripts/research/generate_sedenion_zero_divisor_geometry.py`,
`formal/lean4/SounioZeroDivisorBridge.lean`, and
`scripts/ci/sedenion_zd168_crosscheck_gate.sh`. This V0 lane consumes that
evidence and does not re-prove the counts. The three counts are not
interchangeable names for one object.

## Authoritative Surface

- `stdlib/eisa/hypercomplex_zd.sio`
- `tests/stdlib/eisa/test_eisa_h_zd.sio`
- `scripts/ci/eisa_h_zd_reference_gate.sh`

## Required Invariants

- Exact tokens bind signed coefficient arrays and ordered operand identities.
- Exact tokens name the multiplication convention, coefficient domain,
  detector, and orientation.
- Both operands must be nonzero before an exact token can be minted.
- Same-support sign changes are recomputed, not accepted by support alone.
- Measurement receipts never coerce to exact tokens.
- Unsupported inputs and overflow risk fail closed before multiplication.
- Caller-provided element status and token checksum are never accepted as
  arithmetic authority.
- Exact-token consumers independently recompute the bounded ordered product.
- NaN, infinities, negative norms, negative products, and nonpositive
  measurement resolution are unsupported; zero norm remains a distinct
  zero-operand classification.
- A token does not authorize cancellation, inversion, reassociation, or a
  physical interpretation.

## Pending Interface

Checker emission and sign-sensitive semantics for an IR operation, followed by
an explicit EISA/RTL profile. The current `IrSedZDCheck` placeholder is not an
authoritative implementation of this concept.

## Claims Permitted

After the focused gate passes, the named software reference may claim exact
annihilation detection for its bounded V0 domain under the named convention.

## Claims Forbidden

- General zero-divisor decision for arbitrary sedenions or hypercomplex algebras.
- Native compiler, EISA opcode, ABI, FPGA, or silicon support.
- Equating measured near-zero with exact annihilation.
- Physical, biological, psychiatric, or clinical interpretation.

## Semantic Lane Declaration

```text
Semantic-Lane-ID: EISA-H-ZD-REF-V0
Owner: Codex eisa_h_inventory lane
Concept-IDs: SOUNIO-HYPERCOMPLEX-ZD-EVIDENCE; SOUNIO-ZERO-PROVENANCE; SOUNIO-EPISTEMIC-NUMERIC-VALUE; SOUNIO-NONASSOCIATIVE-ORDER; SOUNIO-EXPLICIT-DISCHARGE; SOUNIO-PRECISION-PRESERVATION
Intent-Preserved: ordered nonzero-factor annihilation retains signed operands, identities, convention, domain, detector, and proof-versus-measurement status
Transformation: add a bounded stdlib software reference with semantic revalidation at every exact-token consumer
Types-Changed: add SedExact16V0, ZDExactAttemptV0, ZDExactTokenV0, and ZDMeasuredReceiptV0
Effects-Changed: none
IR-Changed: none
Claims-Introduced: exact ordered annihilation detection for at-most-two-support +/-1 i64 sedenions under cd_sigma_exact_i64
Claims-Forbidden: arbitrary-sedenion decision; native runtime; EISA opcode; ABI; RTL; silicon; physical or clinical interpretation
Assumptions: cd_sigma_exact_i64 is the named convention; each admitted operand has at most two nonzero coefficients, each +/-1; exact accumulation magnitude is at most four
Write-Set: stdlib/eisa/hypercomplex_zd.sio; tests/stdlib/eisa/test_eisa_h_zd.sio; scripts/ci/eisa_h_zd_reference_gate.sh; this concept and generated governance metadata; .claude/llm_offload_log.md
Read-Set: stdlib/algebra/cayley_dickson_exact_i64.sio; stdlib/epistemic/zero_event.sio; existing exact census report, Python, Lean, and gate surfaces
Positive-Witness: (e3+e10)*(e6-e15) mints a revalidated exact token
Negative-Witness: e1*e2, zero operand, same-support sign tamper, forged 2^32 coefficients, forged token, invalid measurements, and equality threshold are rejected or classified
Acceptance-Gate: bash scripts/ci/eisa_h_zd_reference_gate.sh; bash scripts/ci/zero_event_gate.sh; semantic and docs gates; xAI and ZAI math review
Integration-Target: origin/main
Authoritative-Only-If: focused gate reports check=Madaros and execution=lean_single, all adversarial witnesses pass, and no compiler, ISA, ABI, native, or hardware claim is inferred
```

## Semantic Integration Receipt

```text
Semantic-Outcome: bounded software reference revalidates exact evidence and passes focused adversarial witnesses
Concept-Status-Before: proposed
Concept-Status-After: executable
Distinctions-Added: signed support; ordered identity; exact token versus measured receipt; unsupported versus overflow-risk versus zero operand; finite measurement requirement
Distinctions-Preserved: value versus error versus uncertainty; zero provenance; nonassociative order; explicit discharge; software reference versus native or hardware support
Distinctions-Erased: none
Evidence-Run: Madaros check; lean_single execution; focused reference gate; zero-event gate; semantic/docs gates; xAI and ZAI math review
Fallback-Path: explicit lean_single execution only; default Madaros wrapper exits 1 after native driver write rc=12
Legacy-Kept: all existing EISA, algebra, compiler, IR, native, ABI, and hardware paths
Conflicting-Lanes: none in this write set
Next-Semantic-Interface: E176/private-field enforcement, then separately reviewed sign-sensitive checker emission and non-placeholder IR/native semantics
```

The durable execution receipt for this revision is:

```text
check=Madaros
execution=lean_single
default_madaros_runtime=BLOCKED
default_madaros_wrapper_rc=1
default_madaros_native_driver_rc=12
```
