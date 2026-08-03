# Pre-execution review disposition

## Applied findings

- The contract now states that `zs` is a fixed scalar, expands the three
  divergence partials, fixes the positive orientation of the `w=0` section,
  and freezes every source-frame scalar.
- The worker evaluates the terminal normal velocity on the section as
  `x*y-zs`, rather than retaining the positive bisection residual in `w`.
- The verifier requires the exact worker field population and order.
- The verifier independently reconstructs the dyadic source center, initial
  normal velocity, and oriented frame area at both Decimal precisions.
- CAPD containment is strict at both resolutions.
- Worker Python version and executable are bound to Slurm runtime provenance.

## Documented disagreements

- Grok's initial prose printed an incorrect simplified divergence while saying
  it matched. A focused retry derived the three partials correctly and passed.
- Z.AI returned empty artifacts on both the full and focused attempts. DeepSeek
  was used as the independent fallback, so the default Z.AI leg is incomplete.
- DeepSeek's divergence BLOCKER contradicts its own displayed calculation: its
  calculation sums exactly to the implemented `ell'`. The focused Grok retry
  independently confirms the identity.
- DeepSeek claimed the worker omits resolution-prefixed `Q0_AREA` fields. The
  worker emits both from its explicit `COARSE`/`FINE` loop; the one-leaf smoke
  check also passed exact formula reconstruction at both precisions.
- DeepSeek claimed the manifest input digest could be supplied independently of
  coordinates. The imported frozen loader derives the digest from the validated
  dyadic depths and indices; it does not trust a manifest digest column.
- The runner's summary deliberately keeps completion false. Only the later
  verifier can transition the separate verification output to true after
  checking retained raw evidence and Slurm provenance.
- The SHA-256 framed transfer is an integrity and completeness check, not an
  authenticated-host claim. No authenticated-channel claim is made.
- Deterministic tar bytes are not claimed. The exact received archive hash and
  its expanded canonical contents are retained.

## Scope after review

This remains a non-rigorous point-orbit falsification scout. It is not an
interval integrator, an independent interval certificate, a global H-PG
certificate, V7-B eligibility, novelty, or a solution of an open problem.
