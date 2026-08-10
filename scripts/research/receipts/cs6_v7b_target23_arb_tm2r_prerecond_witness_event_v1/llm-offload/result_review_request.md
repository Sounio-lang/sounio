# Result-scope review request

Audit the exact receipt `witness_event.json`, its independent verifier output,
mutation log, and `result_report.md` in this directory.

Check these claims:

1. The time-depth extension from 10 to 18 reduced the derivative width by only
   `0.008843634958699553%`.
2. The terminal derivative interval has midpoint about `21.75265821188538` and
   radius about `319.0841178894043`, so a fixed-midpoint radius reduction by
   more than `14.668741391572123` is necessary for a strictly positive lower
   endpoint.
3. Thirty-eight attempts equal two endpoints at every integer time depth from
   0 through 18, and all report 256 unresolved leaves after 255 split nodes.
4. The evidence supports `WITNESS_TRANSVERSALITY_UNRESOLVED`, not an event
   criterion defect, absence of a true return, or impossibility of a better
   spatial carrier.
5. Positive upper variable weights are described only as represented possible
   dependence at the raw projection, never pointwise or final dependence.

Return BLOCKER, MAJOR, or MINOR findings. Recompute the arithmetic and flag any
claim wider than the exact receipt.
