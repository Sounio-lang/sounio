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
| `f128`, `f256` | `error[E218]` — Reserved |

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
| `f128` in a parameter, with a caller | **`error[E218]`** | accepted, ELF emitted |

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
4. **Owed measurement, not a ruling.** Whether the Lorenz certificate's
   intermediate products exceed `i64`. Until that is measured, no claim about the
   certification's soundness should be made in either direction.
5. **Does `Reserved` belong on both engines?** `E218` is Madaros-only; lean_single
   accepts `f128`. The Reserved state is the tree's one first-class refusal, and
   it currently holds on one engine.

## Claims Forbidden

What this section does not license anyone to say:

- Not that `stdlib/systems` is wrong. The magnitudes that would decide it are
  unmeasured, and §12.4-4 names that as the missing step.
- Not that the widths are unimplementable. They are unimplemented.
- Not that `f128`'s refusal is a defect. It is the one place in the tower where a
  name that does not work says so, and it is the template the rest owes.
- Not that `u16` having zero uses makes it removable. Absence of use is not
  evidence of absence of intent; §12.4-1 is the ruling that decides it.
