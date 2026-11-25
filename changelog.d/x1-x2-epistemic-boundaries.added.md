Lie corpus X1 and X2, implemented as compile-time refusals with E313.
X1: a below-quantification observation is now the distinct compiler type
`Censored<T>` (built by `censored(lloq)`, read back with
`censored_bound(c)`), never a `Knowledge` with a sentinel value or an
epsilon mode: passing one where a measurement is expected is refused.
X2: the standard uncertainty `u` gains provenance in
`TypeEntry.knowledge_u_prov` — Type B (declared sigma: `measure()`,
`Knowledge(...)`) vs Type A (estimated from residual scatter:
`residual_knowledge(v, sigma)`). A Type A value does not satisfy a
Type B or unspecified requirement; it becomes Type B only through the
caller's explicit `type_b(...)` choice. Provenance lives on `u`, never
in `conf` or the epsilon slot.
