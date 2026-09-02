<!-- docs:meta
topic_id: repo.docs.audit.dimensional-typing-gap-2026-09-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dimensional-typing-gap-2026-09-02
-->

# Dispatch: the dimensional type system cannot express the GRI-Mech constants (2026-09-02)

**Status:** measured, not fixed. Issue [#2388](https://github.com/Sounio-lang/sounio/issues/2388); related [#2387](https://github.com/Sounio-lang/sounio/issues/2387) (`f128`). Filed under the forensic dispatch protocol
(`CLAUDE.md` §8): evidence and minimal reproductions first, no ad-hoc patch to
`self-hosted/`.

**Why it matters.** `benchmarks/chemistry/RESULTS.md` §6.3 (6) records a
convention constant — `R_cal = 1.9872041` against `8.31446261815324/4.184` —
that survived a 30-site alignment because a bare `f64` literal has no
syntactic signature. The remedy the section proposes is dimensional typing:
`R: cal/(mol·K)` *derived* from `R_SI: J/(mol·K)` and `4.184 J/cal`, so the
truncation becomes a checkable derivation. The three reproductions below
show that the language, on both engines, cannot express that today.

## Repro 1 — derived unit annotations do not parse (both engines)

`tests/known-gaps/units/derived_unit_annotation_unparsed.sio`

```
let c: mol/cm3 = n / v        # Madaros: parse error, expected=146 actual=135
let ea: cal/mol = 16812.0     # lean_single: unknown identifier `mol`
```

Every spelling tried fails: `mg/mL`, `cal/mol`, `mol/cm3`, `mol/cm^3`,
`cal/mol/K`, `cal/(mol*K)`, `mol/L`. `examples/macro_system_demo.sio`
contains `let concentration: mg/mL = dose / volume;` and reports `check: OK`
on Madaros — but the same function extracted verbatim into a fresh file fails
to parse. The demo is not being parsed; its green is not evidence.

## Repro 2 — lean_single drops the dimension of an inferred quotient

`tests/known-gaps/units/derived_unit_dropped_by_inference.sio`

```
let n: mol = 0.0000081
let v: cm3 = 1.0
let t: K = 1500.0
let c = n / v
let bad_inferred = c + t     # accepted  <-- mol/cm3 + K
let bad_direct   = n + t     # error: unit dimension mismatch
```

Direct addition of two annotated bindings is checked; a quotient loses its
dimension and adds to a temperature without complaint.

## Repro 3 — a unit is lost at every call boundary (lean_single)

`tests/known-gaps/units/unit_lost_at_call_boundary.sio`

```
fn takes_f64(x: f64) -> f64 { x * 2.0 }
let t_k: K = 1500.0
let b = takes_f64(t_k)       # accepted: K passes as f64, no cast needed
```

The GRI-Mech rate path is `g30_kfwd_eff(r: i64, t: f64, m_eff: f64)`. Even a
fully annotated caller loses `K` on entry, so **the constant the section is
about would not have been caught even in principle** by the checker as it
stands. This is the decisive one.

## What would close it

1. A parser rule for derived unit types (`A/B`, `A·B`, `A^n`) on both engines,
   or a documented constructor form if `/` in type position is reserved.
2. Dimension propagation through `*` and `/` in inference, not only through
   `+`/`-` between annotated bindings.
3. Unit-typed parameters, so a `K` argument to an `f64` parameter is an error
   unless cast.

With those three, `stdlib/chemistry/gri30_h2.sio` can carry
`Ea: cal/mol`, `R: cal/(mol·K)` derived from `R_SI` and the calorie, and
`c: mol/cm3`, and instance (6) becomes a compile error. Without them the
"re-type the kinetic path" step of the 2026-09-02 dispatch is not a probe
port; it is compiler work, and it is filed here rather than faked with
annotations that the checker does not read.

## Related measured limitation

`examples/numerics/f128_is_f64_probe.sio`: `f128` is refused at parse by
Madaros and accepted by lean_single **as f64** — the halving count until
`1+ε == 1` is 53, not 113. `CLAUDE.md` §13 states the opposite engine split
and is stale. Recorded in `docs/compiler/KNOWN_LIMITATIONS.md`.
