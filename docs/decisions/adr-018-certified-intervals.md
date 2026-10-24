<!-- docs:meta
topic_id: repo.docs.decisions.adr-018-certified-intervals
authority: repo_only
audience: users
last_validated: unvalidated
validated_by: none
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.decisions.adr-018-certified-intervals
-->

# ADR-018: Certified interval arithmetic and separate statistical uncertainty

- Status: accepted design; implementation and qualification in progress
- Decision date: 2026-10-24
- Decision owner: Demetrios Agourakis
- Tracking: SIO-177 (ABI/frontend/IR), SIO-178 (backends/Lean/qualification)
- Detailed design: [D-34 proposal](../proposals/CERTIFIED_INTERVAL_PRIMITIVE_2026-10-22.md)

## Decision

The maintainer approved the full recommended D-34 package as part of the massive
stdlib sprint. This records a language decision, not an assertion that an existing
compiler or library operation is certified.

1. `Knowledge<T>` keeps a value enclosure `e` and statistical standard uncertainty
   `u` as separate components. Conversion `e_to_typeB(dist)` is explicit and
   records its distribution and provenance. No implicit quadrature conversion.
2. Nondegenerate `e` requires a qualified interval Jacobian to propagate `u`.
   If the derivative implementation is unavailable, the certified path refuses.
   An explicit provenance-bearing `assume_point` may request an uncertified point
   approximation; it cannot manufacture a certified derivative.
3. Interval FMA is explicit. Implicit contraction and reassociation remain forbidden.
4. Comparisons have Certain/Possible/Never results and no implicit Bool conversion.
   Uncertain `if` is refused. Explicit `hull_branches` evaluates both branches
   without externally visible effects and returns an interval hull in Marco 0.
   It does not invent probabilities or statistical uncertainty for the branches.
5. Marco 0 includes x86 EVEX scalar/ZMM, RISC-V F/D static rounding and RVV in
   Spike, and PTX. No EVEX YMM embedded-rounding path. AMD is Marco 0b; Metal,
   AMX and U250 are outside this certified scope.

## Representation and arithmetic contract

`Interval<f32/f64>` is inf-sup and NaN-free. Empty is `[+inf,-inf]`; entire is
`[-inf,+inf]`; zero lower endpoints are negative zero and upper endpoints positive
zero. Decorations are ill/trv/def/dac/com. Invalid input is a refusal, distinct
from a valid empty set. Infinity is an endpoint, never a real singleton member.
Division across zero returns the hull with trv decoration; division by `[0,0]`
returns empty/trv. Endpoint `0*inf` and indeterminate infinite differences require
set-based special cases before hardware arithmetic.

Directed rounding is opcode identity. Start with opaque interval kernel regions;
FP-state writes are barriers before any optimization crosses those regions.
Each x86 kernel checks entry MXCSR against `0x1F80`, including DAZ/FTZ; RVV kernels
check their expected frm. A bad environment refuses rather than weakening the
arithmetic guarantee. Endpoint selection uses compare/blend, not ISA min/max.

Initially disable interval constant folding. Directed decimal conversion must be
host-independent and implemented in Sounio before interval decimal literals can
claim a certified enclosure. MPFR is a test oracle, never a compiler dependency.

## Proof and execution boundaries

The Lean 4 model uses Rat extended by infinities. RD/RU axioms are listed and
marked NOT ESTABLISHED against a bit-level IEEE model. Enclosure proofs for the
four operations must be sorry-free. Differential execution against a C++23/MPFR
oracle checks the hardware link; it is not a proof of that link.

Qualification requires one million deterministic cases per format/operation,
edge cases and negative controls, with no containment/tightness violations and
bit-identical endpoints/decorations across all Marco 0 targets. Keep target,
compiler, source, FP environment and stdout identities in receipts. Until those
gates pass, expose implementation status honestly; no generic certified label.

Transcendentals, affine arithmetic, midpoint-radius GEMM and general model
certification remain outside this ADR. In particular typed censored log support
may retain symbolic `log(LOQ)`; it cannot claim a certified binary endpoint from
an unqualified logarithm.

## Integration

SIO-87 typed zeros distinguishes structural, censored, absent and numerical zero;
absence-to-zero requires an explicit recorded decision. SIO-88 meet/contest uses
certified intersection plus explicit En(k)/Birge policy, retaining both
provenances on refusal. Neither silently changes current `Epistemic` arithmetic.
Shared compiler files follow the coordination owner contract; isolated structural
contracts do not substitute for primitive type/IR/backend integration.
