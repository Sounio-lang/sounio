<!-- docs:meta
topic_id: repo.docs.research.proof-carrying-linear-target-monitor-d12-2026-07-26
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.proof-carrying-linear-target-monitor-d12-2026-07-26
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Proof-Carrying Linear Target Monitor D12

Status: frozen bounded synthetic specification, 2026-07-26.

Lineage note: the legacy branch for this stage is named
`codex/psychiatric-d12-affine-target-monitor-20260719` (never implemented; it carried zero
commits of its own). The label *affine* predates this specification. The semantics adopted here
is **linear**, not affine, for the reason given under Literature-Derived Design Constraints:
only linearity forbids silent discard, and that prohibition is the point of the stage.

## Research Question

D9 separates statistical categories, D10 separates statistical evidence from a revocable
deployment warrant, and D11 governs when source-population evidence may be transported to a
target population. All three share a structural hole that D10 states about itself:

> "Any caller can invoke the frozen fixture producers and pipe their private return values by
> inference, so the lane proves stage non-substitution, not caller authenticity or evidence origin."

Two consequences follow, and D12 addresses the second:

1. Nothing establishes **who** produced a monitoring observation (out of scope here).
2. Nothing prevents the **same** observation from being presented repeatedly, nor prevents an
   observation from being **collected and then dropped**. Evidence is, in D2–D11, a freely
   copyable and freely discardable value.

This is not a cosmetic gap. It voids the two axes the previous stages are built on: a revocable
warrant (D10) is meaningless if a stale observation can be re-presented to renew it, and
asymmetric degradation (D11) is meaningless if the evidence that would force a downgrade can be
quietly discarded.

The question:

> Can a language make monitoring evidence a **linear resource** — consumed exactly once — so that
> (a) re-presenting the same observation cannot re-authorize, and (b) collecting an observation
> and failing to account for it is a compile-time error?

D12 answers a bounded version. It gives monitoring evidence a linear type, gives the resulting
authorities uninhabited-sealed barrier types, and requires that every produced observation is
consumed exactly once — renewing, degrading, or revoking, but never vanishing.

This is a synthetic epistemic kernel. It is not a patient model, a monitoring algorithm, a
statistical procedure, or a clinical pathway.

## Literature-Derived Design Constraints

**Linear, not affine — and the distinction is load-bearing.** In substructural type systems
(Walker, *Substructural Type Systems*, in Pierce ed., ATTAPL, MIT Press 2005; Wadler, *Linear types
can change the world*, 1990), a **linear** type admits exchange but neither weakening nor
contraction (used exactly once); an **affine** type additionally admits **weakening** (used at most
once, may be discarded). The guarantee "no monitoring evidence is silently discarded" is precisely
**the absence of weakening**. An affine monitor would deliver only the anti-replay half.

**The mechanism is not new; the application is.** Using linear types to govern an epistemic budget
is an established line, beginning with Fuzz (Reed & Pierce) and continuing through DFuzz
(Gaboardi et al., POPL 2013, https://dl.acm.org/doi/10.1145/2429069.2429113), Duet
(https://arxiv.org/pdf/1909.02481), Jazz, Solo, and *Contextual Linear Types for Differential
Privacy* (TOPLAS 2023, https://dl.acm.org/doi/10.1145/3589207). This repository already contains an
instance: `stdlib/privacy/lib.sio` declares `pub linear struct PrivacyBudget`, citing
Dwork-McSherry-Nissim-Smith (2006). Linear types have likewise been used to forbid **nonce reuse**
at compile time (https://arxiv.org/pdf/2305.04138) — the anti-replay half of this stage is a known
application and is **not claimed as novel**. Revocable authority has been treated with
flow-sensitive capabilities (*Typestate via Revocable Capabilities*, PACMPL 2026,
https://arxiv.org/abs/2510.08889).

What we did **not** find in that literature is the obligation half: using the absence of weakening
to make *collected-but-unaccounted* clinical monitoring evidence a compile-time error. The
empirical problem is well documented — outcome reporting bias, where results are selected for
publication based on their direction, with odds ratios of roughly 2.2–4.7 for fully reporting
statistically significant outcomes (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6161807/;
https://catalogofbias.org/biases/outcome-reporting-bias/) — but its enforcement today is
**procedural** (prospective trial registration), not type-theoretic.

**Numeric budgets are additive and partial; linearity is structural and atomic.** Alpha-spending
(DeMets & Lan, *Interim analysis: the alpha spending function approach*, Stat Med 1994,
https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.4780131308) spends only the increment since the
last look; differential privacy composes additively. A linear value, by contrast, moves whole. The
two are reconciled by the **consume-and-return** pattern (`spend(b, ε) -> Budget`), which is what
Fuzz-lineage systems and `PrivacyBudget` already do. D12 therefore treats a monitoring
**observation** as an atom — one reading, one event — and does **not** model α as a linear atom.

**Declare the ceiling in advance.** Game-theoretic statistics (Ramdas et al., *Game-theoretic
statistics and safe anytime-valid inference*, https://arxiv.org/abs/2210.01948) permits optional
stopping provided you never risk more than the capital committed at the outset. The obligation set
must likewise be fixed a priori, or "not collecting" trivially evades the guarantee — the same
reason prospective registration exists.

## Statistical Anchors

- Alpha-spending and group-sequential monitoring: DeMets & Lan (1994), above.
- Safe anytime-valid inference and e-processes: Ramdas et al. (2023), above; already anchored by D10
  and not restated here.
- Outcome reporting bias, empirical magnitude: PMC6161807, above.

The D12 fixtures implement **none** of these algorithms or theorems. Linearity enforces a discipline
of **custody and accounting** inside a program; it does not validate the underlying statistics.

## Clinical And Lifecycle Anchors

Regulated monitoring treats validation as lifecycle- and context-bound, with change control and
real-world monitoring as distinct obligations (IMDRF N88; FDA PCCP final guidance, 2025) — these are
already cited by D10 and are inherited rather than re-derived. D12 adds only the custody rule: an
observation that has been produced must be accounted for.

## Frozen Contest Table

The stage declares, as nominally distinct types:

| Category | Type | Producible? |
|---|---|---|
| monitoring observation | `linear struct D12MonitorObservation` | yes, by a declared fixture producer |
| accounted outcome | `D12AccountedOutcome` | only by consuming an observation |
| target-population authority | `D12TargetAuthorityReceipt` | **no** (uninhabited seal) |
| external validation | `D12ExternalValidationReceipt` | **no** (uninhabited seal) |
| production permission | `D12ProductionAuthorityReceipt` | **no** (uninhabited seal) |
| patient state | `D12PatientStateReceipt` | **no** (uninhabited seal) |

Barrier receipts carry an **uninhabited** seal (`enum D12ReceiptSeal {}`), so no value of the seal
type exists: a naive forge `{ receipt_id: 1 }` is a field-type error and a nested forge
`{ receipt_id: D12ReceiptSeal {} }` is a construction error for an uninhabited type. Both are
rejected by `souc check` **and** `souc compile`, at typecheck.

The negative obligations are therefore of three kinds:

1. **Replay** — consuming the same observation twice (linear double use, `E039`).
2. **Silent discard** — producing an observation and not consuming it (`E040`).
   This is the obligation an affine type would *not* impose, and it is the distinguishing
   claim of this stage.
3. **Barrier** — an accounted outcome does not become external validation, production permission,
   target-population authority, or patient state (`E016` naive forge, `E015` nested forge of the
   uninhabited seal) — inherited from D10/D11.

### Which of these are mechanized here, and which are not

| obligation | diagnostic | test shipped | why |
|---|---|---|---|
| silent discard | `E040` | ✅ yes | distinguishable and immune to the checker defect below |
| barrier (naive forge) | `E016` | ✅ yes (×4) | rejected by both `check` and `compile` |
| barrier (nested forge) | `E015` | ✅ yes | uninhabited seal has no value |
| observation forged directly | `E012` | ✅ yes | payload type is non-`pub` |
| outcome forged directly | `E046` | ✅ yes | payload type is non-`pub` |
| **replay** | `E039` | ✅ yes | shippable since the checker defect below was fixed |

**Unforgeability is a property of the field's type, not of the struct.** Marking a struct `pub`
makes it *constructible* by importers — constructor visibility follows the struct, not its fields.
A first draft of this module declared `D12MonitorObservation` and `D12AccountedOutcome` as `pub`
structs over plain `i64` fields, and both were forgeable from outside, which silently voided the two
"producible?" claims in the table above: an importer could mint an observation without the producer
and an outcome without consuming anything. Both types now wrap a **non-`pub` payload type**, which is
the same construction D11 uses for its barrier receipts (`D11ReceiptSeal`). The two rows above are
the regression tests for that.

**Replay was initially not shippable, and is now.** Expressing replay requires naming the value
(`let o = …; use(o); use(o)`), but the modular Madaros checker used to raise a **spurious `E039` on
the single-use pattern** too, so a replay test would have passed for the wrong reason — a vacuous
test of exactly the kind the D-series audit (2026-07-25) exists to prevent.

The defect was root-caused and fixed while building this stage (issue #1464): typing an
identifier argument consumed the linear binding, and the explicit ownership-transfer step then
consumed it a second time whenever the parameter was by value. Consumption during typing is now
suppressed for every identifier argument, leaving the transfer decision to the single place that
knows the by-ref/by-value distinction. With that, `E039` distinguishes genuine replay from
first use, and the test is shipped.

The positive dual is shipped and passes: `tests/run-pass/clinical_linear_target_monitor_witness.sio`
consumes each observation exactly once and is accepted by `souc check` **and** runs, which is what
establishes that `E040` above comes from the missing consumption rather than from the defect.

## Interpretation Boundary

- Linearity governs custody **inside a program**. It does not prevent *not collecting* an
  observation, and it cannot reach fraud outside the type system. Outcome reporting bias is a
  sociotechnical problem; this stage addresses one mechanizable corner of it.
- The mechanism (linear types for an epistemic resource) is **established since Fuzz**; the
  contribution claimed here is the application to monitoring-evidence custody and its enforcement
  by a compiler, not the mechanism.
- The anti-replay property is a known use of linear types (nonce enforcement) and is not claimed
  as novel.
- Uninhabited seals prevent *forgery of the barrier receipts*. They say nothing about caller
  authenticity or evidence origin — the first consequence of the D10 hole remains open.
- No claim is made that the fixtures model any real monitoring process, patient, or clinical
  decision. The contest table is frozen and synthetic.
