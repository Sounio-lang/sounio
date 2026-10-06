<!-- docs:meta
topic_id: repo.docs.audit.units-promise-measurement-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: claude1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.units-promise-measurement-2026-08-19
-->

# Units: promised dimensions, delivered aliases — measurement 2026-08-19

**Why.** `stdlib/units/` (9 files, 33 882 bytes) promises a unit system
with QUDT alignment. The dispatch asked one deciding question: is `mg` a
dimension in the checker, a nominal type, or an alias of `f64`? This is
measurement only — no code was changed.

**Toolchain.** From-source build of `ffaf0f906d` (`origin/main` +
specializer DCE fix, which touches no checker path measured here).
`<bin> check` and `<bin> compile` on single-file programs under
`SOUNIO_STDLIB_PATH=stdlib`. Reproduction files: `/tmp/u1886/*.sio`.

**Instrument validation (before trusting any number).** `check` was
validated against known-bad inputs first: it reports E001 on
`let x: bool = 5.0` and parse errors on malformed input — but **exits
rc=0 in both cases**. The diagnostic text is the signal; rc is not.
Uncalled function bodies ARE checked (E008 fires in a never-called
`fn dead(x: mg) -> f64 { x }`), so a green `check: OK` covers the whole
file. Probes that parse-failed were re-run through `compile` to confirm
the same verdict.

## Semantic declaration (mandatory) — what `mg` is, exactly

There are **three different `mg`s** in this repository, with three
different semantics:

| # | Where | Definition | Semantics |
|---|---|---|---|
| 1 | `stdlib/units/pharmacological.sio:7` | `unit mg;` | Declaration consumed by `collect_unit_decl` (`self-hosted/check/check.sio:17537`), which registers **`unit_dim_dimensionless()`** — the comment in the source admits it: *"For now, user-defined units get a generic dimension."* And it is a **lookup no-op**: builtin `mg` (mass) is registered at init (`check.sio:1201`) and `UnitRegistry::find` returns the first match, so the user's entry is unreachable. |
| 2 | `stdlib/darwin_pbpk/core/pbpk_params.sio:11` | `type mg = f64;` | A plain alias. Verbatim option (a). Verifies nothing. |
| 3 | Checker builtin (`self-hosted/check/units.sio:232`) | registry entry `mg` = mass^1, scale 1/1000 g | Resolves in type position to an f64-derived type tagged with the registry index (`check.sio:16461` → `ty_with_unit(ty_f64(), idx)`). Closest to option (c) — but see the reachability table: the dimensional machinery behind it is unreachable from any program that compiles. |

The builtin registry holds 16 entries (mg, g, kg, m, km, cm, mm, s, ms,
min, h, K, mol, mg_dL, mmol_L, U_L, mm_h), capacity 32, overflow
**silently dropped** (`units.sio:193`, `if reg.count < 32` with no else).
Measured: the 40th declared unit in a file still accepts `as` casts —
because `as` never consults the registry at all.

## The decisive three programs (dispatch's test, verbatim)

| program | verdict | exact diagnostic |
|---|---|---|
| P1 (correct) `let d: mg = 500.0` | **REFUSED** | `error[E001] … expected mg / found f64` |
| P2 (wrong) `let d: mg = 500.0; let w: kg = d` | REFUSED at line 1 | same E001, ×2 |
| P3 (control) `let d: qzx = 500.0` | REFUSED, **identical shape** | `error[E001] … expected qzx / found f64` |

The negative control gives the **same diagnostic** as the real unit: at
binding position nothing distinguishes a registered unit from an
arbitrary name. The correct program does not compile, so the
wrong program cannot be distinguished from it — the two-program test
degenerates. The repo knows: `tests/compile-fail/unit_mismatch.sio`
contains exactly P1's line under `//@ compile-fail` with
`//@ description: BLOCKED - requires units of measure`.

Via the only introduction that compiles (`as`), everything is accepted:

| cast form | verdict |
|---|---|
| `500.0 as mg` | OK |
| `(500.0 as mg) as kg` | OK (mass→mass) |
| `(500.0 as mg) as m` | **OK (mass→length, unchecked)** |
| `500.0 as qzx` (declared `unit qzx;`) | OK |
| `500.0 as flurb` (declared nowhere) | OK |

The dimensional check that exists (`check.sio:8514-8520`, error 43,
`unit_dim_compatible`) sits on the `:`-ascription path — which
**parse-fails** in let and struct-literal position (`expected=131
actual=181`). The `as` path performs no dimensional check. Runtime
measurement: `((500.0 as mg) as kg) as f64` prints **`500.000000`** —
no scale factor is applied; the cast is a rebrand.

## Operator surface of a unit-typed value (all measured)

Given `let a = 200.0 as mg`:

| expression | verdict |
|---|---|
| `a + b`, `a - b`, `a * b` (b: mg) | E004 "cannot be combined with this operator" |
| `a * 2.0` (scalar) | E004 |
| `a / b` (mg/mg) | E004 |
| `a / k` (mg/kg) | E004 |
| `a == b` (mg==mg) | OK |
| `a == k` (mg==kg) | E004 (expected mg) |
| `a == 200.0` (mg==f64) | E004 |
| `print(a)` | OK |
| `let x: f64 = a` | E001 (expected f64, found mg) |
| `fn f(x: f64)` called with `a` | E009 |
| `P { w: a }` where field is `kg` | E016 |
| `let x = a as f64` | OK — the only escape, unchecked |

So a unit-typed value is **inert**: it cannot participate in any
arithmetic, cannot flow to f64, and its only legal operations are
same-unit `==`, `print`, and further unchecked `as` rebrands.

Consequently the checker's own unit machinery is **dead code**:
`check_binary_units` (`check.sio:20684`) requires registry-index
equality for `+`/`-` (an E041 that has never fired for a compiling
program — `binary_result_type` rejects unit-typed operands as
non-numeric first), and the `*`/`/` arms compute `unit_dim_mul`/
`unit_dim_div` and then **discard the result** (source comment:
*"For now, result is f64 with no named unit"*, `check.sio:20708`).
`mg/kg` produces neither `mg_kg` nor a checked f64 — it produces E004.
`units.sio`'s full exponent algebra is reachable only from the
unreachable arms.

## `unit X = expr;` derived definitions are parsed and discarded

`unit mg_per_L = mg / L` parses and checks OK — but
`collect_unit_decl(item.name)` (`check.sio:17351`) receives **only the
name**. The initializer is dropped; the unit registers dimensionless.
The pbpk example's entire derived-unit vocabulary (`mg_per_L`,
`L_per_h`, `mL_per_min`) has no semantics.

## The `//@ run-pass` unit tests are red, and the baseline agrees

All seven unit tests in `tests/run-pass/` fail to compile, each at its
own `let x: <unit> = <literal>` line:

| test | first error |
|---|---|
| `unit_cast_compatible.sio` | E001 (its line 9 is P1 verbatim) |
| `unit_cast_time.sio` | E001 |
| `unit_decl_keyword.sio` | E001 + E004 |
| `unit_div_cancel.sio` | E001 ×2 + E004 |
| `unit_energy_explicit_conversion.sio` | E001 + E008 (also: `eV`/`J` are in no registry) |
| `unit_same_add.sio` | E001 ×2 + E004 |
| `unit_scalar_mul.sio` | E001 + E004 |

`tests/madaros_corpus_baseline.txt` records all seven as `compile`
failures — checked-in run-pass tests that are known-red.

## The "serious uses" are dead text

- `examples/pbpk/darwin_pbpk_14comp.sio` — **line 1 is `/*`**. The
  entire PBPK demo (`unit kg` at :43, `pub weight: kg` at :107,
  `70.0_kg` at :493/:498, `weight: 70.0 : kg` at :624) is inside a block
  comment. The real `main` at the bottom is an integer checksum demo
  (`3·7 + 2 + 5 + 8 = 36`) that prints `36`. The unit syntax that
  parse-errors in isolation never reaches the parser. Separately:
  `70.0_kg` is E137 (`use of undeclared variable` — the suffix does not
  exist in the lexer) and `70.0 : kg` is a parse error, measured in
  isolation.
- `examples/epistemic/knowledge_units.sio:127` — `let dose_m =
  ek_measured(500.0, 2.5)   // unit: mg`. The unit is in a comment, in
  the file whose purpose is joining knowledge and units; its own header
  says it uses "a plain userspace struct (EK) instead of the built-in"
  machinery. `check: OK` — nothing unit-shaped is checked.

## The system that does work: value-level `Quantity` (runtime, not checker)

`stdlib/units/lib.sio` compiles clean, and its users run correctly:

```
force = 19.600000 ± 0.980000 N          (tests/stdlib/units/test_units_stdlib.sio)
work  = 2060.100000 ± 37.355367 J       (examples/units/dimensional_report.sio)
```

GUM propagation and dimension printing are real — as ordinary struct
code. Dimension enforcement is a **runtime `assert`**
(`lib.sio:183`, `assert(dim_eq(a.dim, b.dim))`): adding mass to length
compiles without complaint and aborts at run time (rc=1, no output,
measured). This is dimensional algebra as a library, not as a type
system.

`stdlib/units/qudt.sio`: **zero URIs** (`grep -cE "http|qudt.org|URI"`
= 0; the only "QUDT" is the header comment). It is a second, parallel
`UnitDim` struct (different field order from lib.sio's) with the same
exponent arithmetic — a name table, not ontology linkage — and it fails
its own `check` (E035 in `units/qudt::test_velocity_dim`).
`pharmacological.sio` also fails its own check (E001).

## Importer census (`^use units::`, instrument validated)

The pattern finds all known users (heliobiology, physics, chemistry,
pbpk, viz — spot-checked present). `grep -rl "^use units::"` = **26
files** (+2 `pub use units::` re-exports the pattern misses:
`stdlib/chemistry/mod.sio:5`, `stdlib/physics/mod.sio:11`).

`check` over the sources and importers:

| verdict | files |
|---|---|
| OK (9) | `units/lib.sio`; `examples/{units/dimensional_report, scientific_computing_demo, macro_system_demo, pbpk/metformin_simulation}.sio`; `packages/sounio-units/tests/test_units_e2e.sio`; `tests/packages/package_import_units_witness.sio`; `tests/stdlib/units/{test_units_stdlib, test_units_deep_stdlib, test_units_negative}.sio` |
| FAIL (13) | `units/qudt.sio` (E035), `units/pharmacological.sio` (E001), `heliobiology/units.sio` (parse), `pbpk/rapamycin_units_bridge.sio` (E137 ×37), `pbpk/regulatory.sio` (parse ×6), all 5 `stdlib/physics/*.sio` (AST closure incomplete / unresolved), `tests/stdlib/pbpk/*` (E137 ×37–40), all 4 `tests/stdlib/physics/*` |
| expected-fail (1) | `tests/compile-fail/unit_mismatch.sio` — the BLOCKED negative test |

The single-symbol import form (`use units::{…}`, `use units::qudt::{m,
kg}`) is the known-broken form (lib.sio:10-11 documents it; the physics
and heliobiology failures are that bug, not unit semantics). The glob
form compiles and runs.

## Claims-Forbidden

Per the dispatch: no conclusion that units "verify" or "don't verify"
was asserted before the three programs ran. They ran, and they showed:

- **Checker-level unit types verify nothing that compiles.** The only
  programs that compile introduce units via unchecked `as` (accepted to
  any name, any dimension, no scale factor), and the resulting values
  admit no arithmetic at all. There is no program whose acceptance or
  rejection depends on a unit dimension. This is measured, not asserted:
  P1/P2/P3 above, the operator table, and the `500.000000` runtime
  conversion.
- **The value-level library enforces dimensions at run time** (assert
  abort) and computes correctly. That is the one part of the promise
  that holds — as a library, not a language feature.
- **INDETERMINATE → measured**: the dispatch allowed INDETERMINATE as a
  legitimate answer; the three programs resolved it to the negative for
  the type system.

Not measured here (out of scope, measurement-only dispatch): fix
directions for any of the above; whether the E041 identity rule and
`unit_dim_mul/div` arms were ever reachable in some historical revision
(they are unreachable at this commit); behavior under `--optimize` or
non-native targets.

## Follow-ups surfaced (not acted on)

1. `let d: mg = 500.0` refusing a bare literal makes every checked-in
   run-pass unit test un-compilable; either literal-ascription or a
   `from_mg`-style constructor is needed before any of this is usable.
2. `unit X = expr` discarding its initializer silently is a parser
   smell — it advertises derived units that cannot exist.
3. `stdlib/units/qudt.sio` claims QUDT alignment it does not have, and
   does not compile.
4. `examples/pbpk/darwin_pbpk_14comp.sio`'s commented-out body should
   either be deleted or moved to a design doc; it reads as a working
   example of a feature that does not exist.
