1. [BLOCKER] The conditional `if upward_returns and not unresolved` only guards hull emission; nothing shown forces `status = "TRANSPORT_REFUSED"` (or equivalent) on every path that leaves `unresolved` true, so a worker could still emit a receipt containing the three hull fields under a different or missing status.
   <location: worker emission logic (the supplied snippet only shows the positive case)>
   <why it matters> The independent verifier only inspects the string `"TRANSPORT_REFUSED"`; any other status bypasses the new checks and re-introduces the original partial-hull leak.
   <minimal fix> Add an exhaustive status-setting rule (or assertion) that sets a refused status whenever `unresolved` is non-empty, and prove the verifier covers every refused status string.

2. [MAJOR] No evidence is supplied that `FINAL_SYMBOLIC_DEPENDENCE_LOST` is emitted with `unresolved=False` and a certified complete cover; the verifier therefore cannot be shown to accept the intended hulls for that case rather than treating it as another refused transport.
   <location: status and cover logic for FINAL_SYMBOLIC_DEPENDENCE_LOST>
   <why it matters> The question explicitly claims this case must still produce useful hulls; without the corresponding emission path and verifier rule, the fix either over- or under-accepts.
   <minimal fix> Supply the exact status string, the value written to `terminal_domain_cover_certified`, and the verifier clause that permits the hull fields for this status alone.

NO ISSUES FOUND AT REQUESTED SEVERITY
