# Pre-execution review disposition

Date: 2026-08-09

Target: lineage-preserving transport of the accepted pre-QR XLEL carrier to
the next complete upward section return.

## Review routing

- Mandatory `math-review`: xAI/Grok 4.3 and Z.AI/GLM-5.2.
- Hostile implementation review: xAI/Grok 4.3 and Z.AI/GLM-5.2.
- Focused review after the implementation fix: xAI/Grok 4.3 and Z.AI/GLM-5.2.

Both math-review legs classified the worker as interval-orchestration without
a new standalone derivation. Both implementation reviewers nevertheless found
the same substantive audit hazard: a refused partial domain could expose
top-level time, derivative, and normal hulls built only from successful
branches.

## Addressed findings

1. The worker now omits all three aggregate hulls whenever any terminal branch
   is unresolved.
2. The verifier rejects any refused receipt that exposes one of those fields.
3. A dedicated mutation injects such a partial hull and must be rejected.
4. The worker computes aggregate bounds directly as exact rational endpoint
   minima and maxima over every terminal carrier.
5. The verifier independently recomputes those three hulls with `Fraction` and
   requires exact equality, in addition to strict upward transversality.
6. A second mutation corrupts an aggregate hull and must be rejected.

## Documented disagreements

- The monkeypatch objections do not identify a soundness failure in this
  single-threaded, one-shot Slurm process. Every replacement is restored by a
  surrounding `finally`; an unexpected exception emits no receipt.
- The missing-module reproducibility objection ignores the receipt contract:
  worker, preconditioner, centered chart, composability domain, event carrier,
  chain, adaptive splitter, event integrator, base TM2R source, and frozen prior
  receipt are all bound by SHA-256 and checked independently.
- The node/depth limits are falsifier budgets, not completeness assumptions.
  Exhaustion records every unresolved branch and forces
  `PRERECOND_NEXT_RETURN_REFUSED`.
- Focused-review objections about unchecked status and positive cover gates
  were based on the abbreviated review snippet. The complete verifier accepts
  only `COMPLETE`, `TRANSPORT_REFUSED`, or
  `FINAL_SYMBOLIC_DEPENDENCE_LOST`, derives completeness from the unresolved
  list and carriers, reconstructs every exact binary domain split from the
  unit six-variable domain, and certifies the terminal lineage cover itself.

No covering relation, recurrent graph, chaos result, or open-problem solution
is claimed by this experiment.

Raw review artifacts are retained in this directory.
