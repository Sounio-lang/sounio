<!-- docs:meta
topic_id: repo.docs.internal.concepts.pireus-intel-vpermpd-selector-semantics
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.internal.concepts.pireus-intel-vpermpd-selector-semantics
-->

# Pireus Intel VPERMPD Selector Semantics

Concept-ID: `SOUNIO-PIREUS-INTEL-VPERMPD-SELECTOR-SEMANTICS`

Status: `SEMANTICS_FROZEN`

Semantic-Lane-ID: `pireus-intel-vpermpd-selector-semantics-20260827`

## Intent

Give the frozen Pireus bits=4 `XOR_PERMUTE` layout one pinned Intel selector
interpretation without claiming an emitted lowering, instruction count,
hardware observation, cost, or performance result.

The semantic producer is Sounio:

```text
stdlib/hardware/pireus/intel_vpermpd_semantics.sio
examples/pireus_intel_vpermpd_semantics.sio
tests/stdlib/hardware/test_pireus_intel_vpermpd_semantics.sio
```

## Causal Parents

The executable binds and live-runs the frozen XOR material parent:

```text
xor_material_source_sha256=eadd752fbda1f50f24bed1260c54936d710af10973653982f5687cd8a551a575
xor_material_semantics_sha256=b4791514032859acc0e8888c4d35760f549a6267e02b2cd5f30a96c0b9dee554
xor_material_receipt_sha256=cc157a7d6ba33b945bc9537be1856bf60481573067ade5f166101ca36a98c1df
```

It also consumes the same digest-pinned Intel XED AVX-512F corpus used by the
parent. XED identifies instruction forms. It does not supply selector behavior
or expected results in this lane.

## Vendor Source

The bounded Sounio reader accepts exactly Intel SDM Volume 2C, version 092:

```text
pdf_bytes=3298744
pdf_sha256=939c9543ff98eefb80f5c5a517bf6f08e864497ea8e032334849f3e39a7b3b07
pdf_envelope=%PDF-1.6
```

The raw PDF is not stored in the repository. An authority execution receives
its path explicitly and rejects length or digest drift before deriving any
semantic field.

## Frozen Rule

The Sounio projection identifies the EVEX.512 vector-control form with eight
64-bit destination elements and eight 64-bit control elements. For destination
lane `k`, the selected source lane is the low three bits of control element
`k`:

```text
selected_lane(k) = control(k) & 7
```

The rule is scoped to one eight-lane source vector. It is not a statement
about the immediate-control form or a two-source permute form.

## XOR Match

For fixed displacement `d`, output chunk `c`, and output lane `l`, the frozen
material parent defines:

```text
bits         = 4
dimension    = 16
chunk_count  = 2
chunk_lanes  = 8
d            in [0,15]
c            in [0,1]
l            in [0,7]
source_chunk = c XOR (d >> 3)
source_lane  = l XOR (d & 7)
```

For each fixed `(d,c)`, `source_chunk` is constant and all eight
`source_lane` values form a permutation of `[0,7]`. The vector-control rule
therefore covers all 256 selector cells with 32 abstract form applications,
one per `(d,c)` group. This is enough to close the one-source selector-form
question for this specific `XOR_PERMUTE` boundary.

The number 32 is not an emitted instruction count. No compiler lowering or
machine-code sequence was created or measured.

## Immediate Boundary

The immediate-control form is evaluated separately. Among the eight low
displacements, the four values `d in [0,3]` preserve lane bit 2 and repeat the
same two-bit permutation in both 256-bit halves. The four values `d in [4,7]`
flip lane bit 2 and are refused because they require cross-half selection.
Therefore:

```text
imm8_patterns_tested=8
imm8_patterns_supported=4
imm8_patterns_refused=4
imm8_complete=false
```

No immediate encoding is promoted as a complete realization of the frozen
XOR layout.

## Closed Claims

The result establishes none of the following:

- that a compiler emits `VPERMPD`;
- that Darwin Xeon exposes or executes AVX-512;
- that 32 abstract applications equal 32 machine instructions;
- a lowering for twist, multiply, horizontal reduction, or output;
- latency, throughput, scheduling, register-pressure, or speedup results;
- the earlier estimate of roughly 112 instructions for full multiplication;
- Apple Silicon or DGX selector semantics;
- cross-ISA equivalence;
- Walsh-Hadamard diagonalization or subquadratic twisted convolution;
- a Fano-plane explanation for the seven-negative-sign regularity;
- Lean 4, Koka, C++, or Haskell parity.

Darwin Xeon, Apple Silicon, and DGX remain canonical targets. Canonical target
membership is not a material observation.

The external Loom guardian remains the stage and producer-language authority.
`PARITY_OPEN=false` and `CLAIM_READY=false`.
