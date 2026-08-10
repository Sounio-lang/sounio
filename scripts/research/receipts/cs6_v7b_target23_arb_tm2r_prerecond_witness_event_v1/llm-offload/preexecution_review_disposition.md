# Preexecution review disposition

Provider: xAI Grok 4.3, task `math-review`.

- MAJOR variable-weight scope: accepted. Keys and contract language now say
  that positive upper aggregate weight proves represented possible symbolic
  dependence only; it does not prove nonzero pointwise dependence.
- BLOCKER early acceptance: closed in code. An interval-Newton projection that
  closes before the frozen depth-10 refusal is recorded returns
  `EARLY_ACCEPTANCE_BEFORE_FROZEN_REFUSAL`; the implementation controls then
  force `IMPLEMENTATION_INCONSISTENCY`. It cannot produce
  `EVENT_REFINEMENT_BUDGET_LIMIT`.
- MINOR quadratic transversality: not applicable to the implemented sign test.
  The worker imports a projector whose derivative is evaluated over the full
  Arb Picard tube and whose projected normal is evaluated over the full carrier.
  Neither acceptance test uses only a linearized coefficient or point value.
- MINOR fifth outcome: closed by the same early-acceptance guard and the closed
  verifier classification set.

The default Z.AI second review was attempted and failed with provider code
1310 because the weekly/monthly limit is exhausted until 2026-08-15. The first
generic xAI pass returned `NO MATHEMATICAL CONTENT TO REVIEW`; the focused xAI
request above is the substantive review.

The second focused xAI pass raised a valid naming objection: closing only under
the depth-18 budget does not make the depth-10 refusal incorrect. The accepted
classification was therefore narrowed from `EVENT_CRITERION_FALSE_REFUSAL` to
`EVENT_REFINEMENT_BUDGET_LIMIT`. Its endpoint-sign objection is rejected under
the review's stated assumption that the imported interval-Newton routine is
sound: strict Newton self-inclusion in the validated Picard slab proves an
enclosed root without an endpoint sign-change premise. Its proposed
non-strict-derivative acceptance case is unreachable because the imported
upward projector itself requires a strictly positive full-tube derivative.
