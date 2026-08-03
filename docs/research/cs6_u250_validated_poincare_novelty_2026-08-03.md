# Candidate novelty window: proof-carrying validated Poincare hardware

**State:** pre-execution hypothesis, not a novelty or priority claim.

## Strongest accurate idea

Build an FPGA accelerator that does not merely approximate an ODE trajectory.
It emits dyadic enclosures and a compact integer transcript from which an
independent verifier can check outward rounding, a Picard self-map, strict
contraction, event transversality, and a bounded Poincare-determinant claim.
Failure to establish any obligation must produce refusal, not a numerical value
presented as a certificate.

The intended information is the distinction between a computed center, its
arithmetic/truncation enclosure, scientific uncertainty in the initial set, and
the provenance of every certification obligation.

## What already exists

- FPGA Runge-Kutta and Adams solvers already exist. RHCOS reports a
  high-precision complex ODE solver and speedups, but evaluates relative
  numerical error rather than a validated flow enclosure:
  [Zhang et al., 2025](https://doi.org/10.1587/elex.22.20250473).
- Hardware-oriented parallel Runge-Kutta solvers and custom precision are
  established topics:
  [Kornilina et al., 2022](https://doi.org/10.3390/computation10120215).
- Rigorous interval integration, Taylor models, wrapping control, and rigorous
  Poincare maps are established in software. The modern Poincare-map literature
  explicitly treats sections, coordinates, sliding, and wrapping:
  [Kapela, Wilczak, and Zgliczynski, 2021](https://arxiv.org/abs/2104.08046).
- Formally verified ODE solvers can compute Poincare maps and derivatives in a
  proof assistant:
  [Immler, 2018](https://doi.org/10.1007/978-3-319-94821-8_18).
- Hardware primitives for interval contractors have been reported for custom
  RISC-V extensions:
  [Filiol et al., 2025](https://doi.org/10.1109/NewCAS64648.2025.11107029).
- Recent FPGA number systems provide formal arithmetic error bounds and use
  Runge-Kutta as an application, but bounded arithmetic error is not by itself a
  validated ODE/event certificate:
  [Darvishi, 2026](https://arxiv.org/abs/2603.08712).

## Candidate gap

The searched literature did not expose a system combining all of the following:

1. execution on an FPGA rather than only a software validated-numerics engine;
2. outward-rounded enclosures for the flow, not empirical error versus a
   floating-point reference;
3. rigorous event isolation and transversality for a Poincare section;
4. an independently checkable, proof-carrying integer transcript;
5. a determinant enclosure tied to the event and flow certificate;
6. fail-closed behavior when any enclosure or contraction obligation fails.

This is a candidate intersection, not evidence that no prior work exists.
Priority remains false until a systematic bibliographic review covers FPGA,
validated numerics, hybrid/event ODE integration, computer-assisted dynamics,
and proof-carrying accelerators across multiple indexes and citation graphs.

## Smallest differentiating experiment

The first experiment is deliberately below the ODE claim. Implement signed
S1.I31.F96 interval addition, subtraction, multiplication, and division by frozen
positive integers using exact 256-bit intermediates and directed endpoint
rounding. An independent Python model evaluates every input as an exact
`Fraction`; HLS C simulation must match its 128-bit endpoints exactly.

The adversarial corpus must include zero, exact-grid products, every sign
combination, products one unit above and below a rounding boundary, narrow
target-23 state boxes, wide legal boxes, and divisors `2`, `3`, `6`, and `41`.

Only after this gate passes may a second frozen contract add one target-23
Picard step. A full orbit, event, determinant, and physical U250 run are later
transitions, not implications of the arithmetic gate.

## Falsifiers

- Any exact rational result escaping an emitted interval falsifies the
  arithmetic design.
- Any legal input whose result overflows 128 bits without refusal falsifies the
  domain contract.
- Any critical mutation accepted by the independent verifier falsifies the
  audit design.
- Failure of the Picard self-map or strict-contraction check means the chosen
  precision/step/box is inadequate; it is not rounded into a pass.
- Resource use beyond one U250 SLR, or throughput below a serious CPU validated
  baseline, demotes the machine architecture even if it is mathematically sound.
- Discovery of prior work satisfying the same six-part combination removes or
  narrows the novelty window.

## Current claims

The literature matrix and arithmetic contract are implemented. No FPGA run,
dyadic arithmetic certificate, Picard certificate, rigorous ODE result,
leaf-wide result, global H-PG result, V7-B eligibility, novelty, priority,
promotion, or open-problem solution is claimed.
