# Math-review disposition

## Initial finding

The initial Z.AI reviews read `b_box[16]` as the first term of an infinite
center-series tail and correctly observed that one coefficient alone would not
bound such a tail. This would have invalidated a tail-sum interpretation.

The same review also questioned why time-independent forcing appears explicitly
only at recurrence degree zero.

## Resolution

The intended theorem is Taylor's componentwise Lagrange remainder, not an
infinite-series tail. For each component, the remainder is
`h^16*x_i^(16)(xi_i)/16!` for some `xi_i` in `[0,h]`. Because the predecessor
Picard certificate keeps the whole step trajectory in its strict self-map box,
interval evaluation of the normalized order-16 autonomous flow derivative over
that box encloses the derivative at every possible `x(xi_i)`.

The contract now states these semantics explicitly and says that the remainder
is not `sum(k=16..infinity, b_center[k])`. Generator and verifier variables and
comments were renamed accordingly.

A time-independent forcing has only a degree-zero power-series coefficient on
the right-hand side. Its effect propagates through the state coefficients, so
the existing `delta_k0` terms are correct.

## Corrected review

Focused independent reviews by XAI/Grok 4.3 and Z.AI/GLM-5.2 both accepted the
scaled recurrences, constant-term semantics, Picard trajectory binding, and
componentwise Lagrange remainder enclosure. No unresolved mathematical finding
remains for the bounded one-step claim.

This disposition does not establish a full orbit, leaf-wide certificate,
global H-PG result, novelty priority, or an open-problem solution.

## Final-result review

XAI/Grok 4.3 accepted the final report's Lagrange theorem, Picard-box binding,
exact width decomposition, and evidence boundary. Its only tightening was that
the reviewer-endorsement sentence is not needed for the proof; no downstream
claim changes. Z.AI/GLM-5.2 returned a delayed, truncated final-report analysis
without a verdict; its earlier attempt was initially empty and later appended
another incomplete fragment. Qwen, Mistral, DeepSeek, and Groq/Llama fallbacks
returned provider errors. These failures are retained rather than counted as
passes. The earlier substantive corrected-contract Z.AI review remains the
independent second opinion on the underlying mathematics, but not on the
completed hardware report.
