<!-- docs:meta
topic_id: repo.docs.audit.madaros-imported-module-f64-cast-bitcast-2026-07-14
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-imported-module-f64-cast-bitcast-2026-07-14
-->

# Madaros v0.80.0 — `f64 as i64` in an imported-module function body is a bitcast

**Date:** 2026-07-14
**Toolchain:** `./bin/souc` → Madaros v0.80.0 (default engine)
**Owner:** CODEX-2 (`self-hosted/` — native codegen, numeric cast lowering on the imported-module path)
**Status:** **CLOSED** (2026-07-19 #983 root-cause + #1252 joint D5+D1 land; Wave10 2026-07-21 trust-map/gate promotion). Finite-dof `gum_k95` gated at k95i=2776.
**Severity:** was high — silently corrupted the flagship epistemic primitive (GUM coverage factor)

## Summary

An `f64 as i64` conversion **inside a function that lives in an imported module**
lowers to a **bit reinterpretation** (the raw IEEE-754 payload copied into the
integer register) instead of a **numeric truncation**. The same cast in the
importing program's own `main()` is correct, so the defect is specific to the
imported-module code path.

This is not a corner case: it makes `stdlib/epistemic/gum.sio` report a **95 %
coverage factor of 1.960 for every effective degrees of freedom**, so every
finite-sample expanded uncertainty `U95`/`U99` from an importing program is
**under-covered**. The combined standard uncertainty `u_c` and the effective
degrees of freedom `nu_eff` are unaffected (they never cross an `f64→i64` cast).

## Minimal repro

An importable module:

```sounio
// stdlib/castprobe.sio
pub fn f64_to_i64(x: f64) -> i64 { return x as i64 }
```

```sounio
// driver.sio
use castprobe::*
fn main() -> i32 with IO, Mut, Div, Panic {
    print_int(f64_to_i64(4.172))   // prints 4616383272838735331, expected 4
    print("\n")
    return 0
}
```

`4616383272838735331` is exactly the little-endian IEEE-754 double bit pattern of
`4.172` reinterpreted as a signed 64-bit integer
(`python3 -c "import struct; print(struct.unpack('<q', struct.pack('<d', 4.172))[0])"`).
So the lowering emits a `movq`/bit-copy where it must emit `cvttsd2si` (truncating
convert).

## Localisation — what works vs what breaks

| Case | Result |
|---|---|
| `f64 as i64` in the importer's **own `main()`** | **correct** (truncates) — see `examples/epistemic/gum_to_csv.sio`, `pk_curve_gum_to_csv.sio` |
| `f64 as i64` in an **imported-module function body** | **bitcast** (this bug) |
| imported `fn t95(dof: i64) -> f64` (i64 arg, long if-chain) | correct: `t95(2)=4.303`, `t95(4)=2.776` |
| imported `if dof > 1.0e8 { ... }` (f64 compare vs large literal) | correct branch selection |
| imported f64 arithmetic returning f64 (e.g. `ws2` → `nu_eff`) | correct |

So: general imported-function lowering, i64 args, f64 compares, and f64 arithmetic
are all sound. **Only the `f64 → i64` numeric cast in an imported-module body is
mis-lowered.** The importer-`main()` cast path is the working reference to diff
against — analogous to how the user-fn call path is the reference for the
`&local_array`→builtin defect (`DATA_IO_TRILHA_B_BUILTIN_BUFPTR_DISPATCH_2026-07-14.md`).

## Concrete downstream failure (GUM coverage factor)

`gum.sio` does, inside imported module bodies:

```
nu_eff = ws2(u1, u2)                 // correct f64, e.g. 4.173
di     = dof_to_i64(nu_eff)          // `nu_eff as i64` -> BITCAST -> 4.616e18
k95    = t95(di)                     // t95(huge) -> falls to the 1.960 tail
```

Result: `gum_k95` = 1.960 regardless of `nu_eff`; `gum_u95 = k95 * u_c` is
under-covered for finite samples. Verified: with `s=0.30,n=3,a=0.20`, `nu_eff=4.17`
(correct) but `k95=1.960`, so the reported `U95=0.408` should be `≈0.571`
(`t_{0.975}(4.17)≈2.743`, × `u_c=0.20817`). The `Knowledge<T>/GUM =
validated_research` claim does not hold for finite-dof coverage intervals until
this is fixed.

**Science check (math-review, xai grok-4.3 + zai GLM-5.2 fan-out, 2026-07-14):**
both providers confirm the coverage factor for finite `nu_eff` is the Student-t
quantile `t_{0.975}(nu_eff)`, converging to 1.960 only as `nu_eff → ∞`
(ISO/IEC Guide 98-3 / JCGM 100 Annex G; the G.6.6 `k=2` rule is a large-`nu_eff`
approximation, not an exact identity). t-table values and the Welch–Satterthwaite
worked example (`nu_eff≈4.17`) confirmed OK; the fixed-1.960 behaviour under-covers
every finite-sample case. Z.AI tightened the worked `U95` to `≈0.571`
(`k≈2.743`), adopted above.

## Proposed fix locus

Native codegen for the `f64 → i64` (and likely `f32 → iN`, `f64 → i32`, unsigned)
conversion on the imported-module lowering path — emit a truncating SSE convert
(`cvttsd2si`) rather than a register bit-copy. The importer-`main()` path already
does this; align the imported-module path to it.

## Acceptance gate (met)

1. `pub fn f(x: f64) -> i64 { x as i64 }` in an imported module returns `4` for
   `4.172` (truncation) — measured Wave10: `param=4 arith=4 local=4`.
2. `gum_k95` matches `t95(nu_eff)` for small samples — Type-A-dominant
   (`gum_type_a(0.30,5)` + tiny Type-B) → **k95i=2776**, U95≈0.372 under
   default Madaros multi-module import. Gate: `scripts/epistemic_trust_gate.sh`
   Section A (promoted from retired Section B trip-wire). **Do not** use a
   Type-B-dominant budget as the k95 trip-wire — k95=1.960 is correct there.

## AI disclosure

Repros and localisation by AI agent (Claude) under human direction, on Madaros
v0.80.0; re-runnable with `export SOUNIO_STDLIB_PATH=$(pwd)/stdlib`. No
`self-hosted/` sources were modified. Coverage-factor science confirmed by the
mandatory math-review offload. GAIDeT-ICMJE 2025.
