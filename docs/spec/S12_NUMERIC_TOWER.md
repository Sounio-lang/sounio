<!-- docs:meta
topic_id: repo.docs.spec.s12-numeric-tower
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.spec.s12-numeric-tower
-->

---
title: S12 — Numeric tower
status: measured
date: 2026-08-20
last_validated: 2026-08-20
engines: Madaros v0.80.0 (default), lean_single
---

# S12 — Numeric tower

## 12.1 Normative

*Awaiting founder rulings — see §12.4. Nothing in this section is normative yet.*

## 12.2 What is measured today

### 12.2.1 The integer tower is open by pattern, not a closed set

`docs/spec/E2E_SPECIFICATION_FRAME.md` row 12 records the tower as `i8..i128`,
`u8..u128`, `f32`/`f64`. That is not what the compiler implements.

Every probe below is a **parameter with a live caller**, because a parameter type
alone is not interrogated — `fn f(x: T)` checks clean for every `T`, invented
ones included (§2.3 of `LANGUAGE_SPECIFICATION.md`).

| written | Madaros v0.80.0 |
|---|---|
| `i7`, `i13`, `i999999` | **`check: OK`** |
| `u3`, `u4096` | **`check: OK`** |
| `i0` | `error[E009]` |
| `f17`, `f64000` | `error[E009]` |
| `f128`, `f256` | `error[E249]` — Reserved |

So `i<n>` and `u<n>` are accepted for any **n ≥ 1**. The float side is closed:
only `f32` and `f64` exist, and the two Reserved names are refused **by name,
with a diagnostic**.

The asymmetry is worth stating plainly: **the float half of the tower is honest
about what it does not have, and the integer half is not.**

### 12.2.2 No integer width has semantics — all of them are `i64`

| type | expression | correct at the declared width | printed |
|---|---|---|---|
| `i8` | `100 + 100` | `-56` | **`200`** |
| `u8` | `200 + 200` | `144` | **`400`** |
| `i16` | `20000 + 20000` | `-25536` | **`40000`** |
| `i32` | `2000000000 + 2000000000` | `-294967296` | **`4000000000`** |
| `i7` | `100 + 100` | (does not fit) | **`200`** |
| **`i256`** | `5e18 + 5e18` | `10000000000000000000` | **`-8446744073709551616`** |

Every one is plain `i64` arithmetic. The `i256` row is the `i64` wraparound
exactly. **The declared width is a naming convention, not a representation** —
documented widths and undocumented ones alike.

### 12.2.3 `i256` is used 1,826 times, and most of it is one module

`stdlib/systems/` is 56,327 lines across 67 files and 220 importers. It is
almost entirely the **Lorenz i256 certification**: `lorenz_i256_cert_step1..6`,
`child0..4`, flowpipes, cover refinements, boundary faces, obligation seeds,
discharge preflights — computer-assisted rigorous numerics.

It carries **733** `i256` annotations, on the certificate's own quantities:
`xy_scaled_q`, `beta_q`, `dx_num`, `dy_num`, `scale`, `target_scale`, `y_inc_r`.

There is **no limb implementation to fall back on**: `fn i256_*` occurs **0**
times in the whole of `stdlib/`.

**What is measured about the consequence, and what is not.** The module's
literals stay well inside `i64` — the largest is `27000000000000000` (2.7e16),
and there are **zero** literals of 19 digits or more against an `i64` ceiling of
9.22e18. Whether intermediate **products** exceed `i64` is **not measured here**,
and it is the question that decides whether this is a naming hazard or a
soundness defect in a proof artefact. It is recorded as owed (§12.4-4) rather
than asserted in either direction.

### 12.2.6 ANSWERED — the Lorenz certificate's products do exceed `i64`

§12.2.3 left one question open and refused to answer it in either direction. It
is answered (#2046, measured on an independently source-built compiler).

**Maximum observed intermediate:**

    8,007,432,506,888,905,229,835,698,176

against a signed `i64` ceiling of `9,223,372,036,854,775,807`. The product is
**exactly `868,167,572 × 2^63`** — the ratio to the `i64` *ceiling* is
`868167572.000000` to six decimals but is **not** an integer multiple of it, and
the first receipt's phrasing *"exactly 868,167,572 times that bound"* was the
rounding, not the identity. The independent replica (#2050) caught it. It is
`y_lte_source * den` in
`lorenz_i256_step5_taylor2_remainder_obligation_check`,
`stdlib/systems/lorenz_i256_cert_step5.sio:2310`.

Since §12.2.2 measures `i256` to be `i64`, that product **wraps**, silently,
inside a computer-assisted proof artefact.

**How it was measured, because a result this size is worth its method.** Madaros
was built from source `67aa2aec12` on Slurm (`cpuops-t560-proxmox`, 32 CPUs,
`rc=0`, ELF SHA-256 recorded), 25 fixtures were executed under that ELF, and an
**exact arbitrary-precision accumulator** replayed the typed-`i256` arithmetic of
the calls actually executed — 933 intermediate values, **no floating-point
conversion at any point**. The positive control forced `10^30` and the detector
reported it above `2^63 - 1`, so the detector is known to fire.

**Coverage is bounded and declared.** Covered: centre, radius and remainder checks
for steps 1–6, the step certificate, the trajectory-5 certificate, imported
child-0 and child-1 arithmetic bundles, the refinement ledger, and two standalone
negative controls. Explicitly **`NOT EXECUTABLE`** for this receipt: children 2–4,
bridge families, long loops, proof skeletons, candidate/replay/enclosure/flowpipe
fixtures, and the trajectory-5 projection-inclusion fingerprint — because no
source-built invocation was made for them, and *static inspection is not counted
as runtime measurement*. This is a measured certificate-path result, not an
exhaustive claim about every Lorenz `i256` path.

**Independently replicated, and the replica answers more than it was asked.**
#2050 measured the same peak at the same site, from a separate from-source build,
**measuring first and reading #2046 only after the peak was in hand** — the order
is what separates a replication from an echo. Its positive control fired on `2^63`
before any Lorenz number was touched.

It also settles, for the obligation measured, the question §12.4-6 was opened to
hold: **the wrap does not overturn this conclusion.** Full-width arithmetic still
leaves `source_lte_ok = 1`. So the honest statement narrows by one notch — the
arithmetic is unsound, and *the one obligation whose verdict has been recomputed
at full width survives it.*

**What remains unknown, and it is not small.** Whether the wrap changes the
certificate's **verdict**. A product that overflows inside a comparison may still
land on the correct side by arithmetic accident. That is a different question from
this one and it is owed (§12.4-6).

### 12.2.4 The tower as the corpus actually uses it

Type annotations and return positions in versioned `.sio` outside `archive/` and
`bootstrap/`:

| type | uses | | type | uses |
|---|---:|---|---|---:|
| `i64` | 184,565 | | `u8` | 112 |
| `f64` | 49,757 | | `i128` | 137 |
| `i32` | 19,993 | | `u32` | 143 |
| `i8` | 1,959 | | `u64` | 73 |
| `i256` | 1,826 | | `u128` | 5 |
| `f32` | 1,730 | | `i16` | 6 |
| `f128` | 82 | | `u16` | **0** |
| `f256` | 69 | | | |

`u16` is in the documented tower and has **zero** uses. `f128` and `f256` are
Reserved and refused by Madaros, and appear 82 and 69 times — every one of those
sites is refused on the default engine.

### 12.2.5 The engines disagree on the Reserved names

| | Madaros v0.80.0 | lean_single |
|---|---|---|
| `f128` in a parameter, with a caller | **`error[E249]`** | accepted, ELF emitted |

`f128`/`f256` are the tree's clearest instance of the **Reserved** state — a name
taken, every use refused with a diagnostic that says why. That state exists on
one engine only.

## 12.3 What this does not claim

This section measures **acceptance and arithmetic**, not codegen strategy. It does
not claim the compiler *could* not represent these widths, nor that any specific
`stdlib/systems` result is wrong — §12.2.3 states exactly what was and was not
measured about that. It does not touch `f32`/`f64`, whose IEEE behaviour is not
in question here.

## 12.4 Rulings owed

1. **Is the integer tower closed or open?** `i7` and `i999999` typecheck today.
   If the tower is `i8..i128` plus `i256`, then everything else owes an `E009`
   with a name. If it is open, the spec owes a rule for what `i<n>` means.
2. **Does a declared width mean anything?** Today it does not. If it should, that
   is representation and wrapping work in the backend. If it should not, the
   widths are aliases for `i64` and the spec must say so, because a reader of
   `i8` today reasonably expects wrapping at 8 bits.
3. **`i256`: real or Reserved?** It is neither at present — it is accepted and it
   is `i64`. The `f128` treatment is the honest template: refuse it by name until
   it exists.
4. ~~**Owed measurement, not a ruling.**~~ **ANSWERED 2026-08-20 — it exceeds
   `i64`.** See §12.2.6. The question is now a ruling: what to do about a proof
   artefact whose arithmetic wraps.
5. **Does `Reserved` belong on both engines?** `E249` is Madaros-only; lean_single
   accepts `f128`. The Reserved state is the tree's one first-class refusal, and
   it currently holds on one engine.

6. **Owed, and now partly answered: does the wrap change the verdict?** #2050
   recomputed the step-5 remainder obligation at full width and it still yields
   `source_lte_ok = 1` — that conclusion survives its own overflow. **One
   obligation is not the certificate.** What is owed is the same recomputation
   across the obligations §12.2.6 lists as covered, and then across those it lists
   as `NOT EXECUTABLE`. Until then: *the arithmetic is unsound; one verdict has
   been recomputed and holds; the rest are unaudited.*

## Claims Forbidden

What this section does not license anyone to say:

- **Superseded 2026-08-20.** This bullet read *"not that `stdlib/systems` is
  wrong — the magnitudes are unmeasured"*. They have since been measured (§12.2.6)
  and they exceed `i64`. What is still forbidden is the next step: **not that any
  certificate conclusion is wrong.** The arithmetic is unsound; whether the
  verdicts are affected is unaudited, and §12.4-6 owes it.
- Not that the widths are unimplementable. They are unimplemented.
- Not that `f128`'s refusal is a defect. It is the one place in the tower where a
  name that does not work says so, and it is the template the rest owes.
- Not that `u16` having zero uses makes it removable. Absence of use is not
  evidence of absence of intent; §12.4-1 is the ruling that decides it.
