<!-- docs:meta
topic_id: repo.docs.audit.dimensional-typing-gap-2026-09-02
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dimensional-typing-gap-2026-09-02
-->

# Dispatch: the dimensional type system cannot express the GRI-Mech constants (2026-09-02)

**Status:** measured; Repro 2 CLOSED 2026-09-05 (see the correction at the end), Repros 1 and 3 still open. Issue [#2388](https://github.com/Sounio-lang/sounio/issues/2388); related [#2387](https://github.com/Sounio-lang/sounio/issues/2387) (`f128`). Filed under the forensic dispatch protocol
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
on Madaros — but that line sits inside the `/* ... */` block that wraps the
file's "aspirational example preserved below" (lines 1–263); the checked body
is the 20-line stub after it. **Correction (2026-09-02):** an earlier revision
of this dispatch read that green as "the demo is not being parsed". It is
parsed; the derived-unit annotation is simply commented out. There is no
"check OK on an unparsed body" defect here, and no gate is filed for one.
The three reproductions that stand are pinned by
`scripts/ci/language_gap_ratchet_gate.sh`, which asserts the measured
behaviour and fails on purpose when a fix lands.

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

---

## Correction, 2026-09-05 — Repro 2 is closed

`tests/known-gaps/units/derived_unit_dropped_by_inference.sio` no longer passes.
On both targets, lean_single now reports

```
error: unit dimension mismatch at <main>:8
typecheck: failed
```

so `(mol/cm3) + K` is refused and the quotient keeps its dimension.

What closed it is item **2** of this document's own "What would close it" list --
dimension propagation through `*` and `/` in inference, not only through `+`/`-`
between annotated bindings. `dim_add` / `dim_sub` at the multiplicative site now
compose the dimension of a derived quantity, and the additive check compares it.
The arm64 pass carried neither and now carries both, which is why the refusal
holds on `--target aarch64-linux` as well.

Items **1** and **3** are untouched. Derived unit types (`A/B`, `A·B`, `A^n`) are
still not in the grammar, and a `K` argument still enters an `f64` parameter
unchecked -- Repro 3, which this document calls the decisive one, still passes:

```
[ratchet] ok   unit lost at call boundary (lean_single accepts): 0
```

So the conclusion about `stdlib/chemistry/gri30_h2.sio` stands: instance (6) is
still not a compile error, and re-typing the kinetic path is still compiler work
rather than a probe port.

scripts/ci/language_gap_ratchet_gate.sh now asserts the refusal, so the closure
cannot silently reopen: the ratchet is red in both directions.
