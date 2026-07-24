<!-- docs:meta
topic_id: repo.docs.audit.madaros-wave13-imported-f64-lognormal-defect-2026-07-24
authority: repo_only
audience: users
last_validated: 2026-07-24
validated_by: claude-1
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-wave13-imported-f64-lognormal-defect-2026-07-24
-->

# Madaros — imported module-const `DE_LN_SQRT_2PI` reads garbage in `lognormal_pdf` (multi-module native)

**Date:** 2026-07-24
**Role:** claude-1 — forensic dispatch (evidence + proposed fix; no `self-hosted/` patch applied)
**Tip measured:** `origin/main` @ `3e7ed9f52` (Wave13e)
**Engine:** default `bin/souc` → Madaros v0.80.0
**Where measured:** x86-64 host `r740-proxmox` (fresh `git clone` @ `3e7ed9f52`; the agent workspace box is aarch64 and cannot execute the x86-64 toolchain)

## TL;DR

At HEAD `3e7ed9f52`, the **Wave13 showcase gate does not reproduce full-green**. Five of six required gates pass (`dual`, `order_spread`, `k95`, `cd_exact`, `cd_exact_e2e` — the cd_exact ZD-PROVED headline is solid). The sixth, `wave12_tip_green`, is RED because its `imported_f64` sub-lock fails:

```
FAIL lognormal_pdf (imported DE_LN_SQRT_2PI likely zero)
```

The failing test's own message is **misleading**: the constant is not read as zero. It is read as **≈690.78 (garbage)**, so `lognormal_pdf(1,0,1)` returns `9.999…e-301` instead of `0.39894228…`.

## This is source-genuine, not prebuilt lag

A fresh `make build-madaros` from source at `3e7ed9f52` produced an ELF **byte-identical** to the committed `bin/madaros-linux-x86_64`:

```
fresh:    b097dc085ecd4b0d74f8368801c3e2af2607512a044c0d4009d2fa5716d5c715
prebuilt: b097dc085ecd4b0d74f8368801c3e2af2607512a044c0d4009d2fa5716d5c715
```

So the failure is a property of the source at this commit. Behavior-bisecting the *committed* prebuilt across `f22947019 → e6b8ff557 → 1318bc8e2 → 88c83a9ef → 3d0932795 → 3e7ed9f52` shows `imported_f64` **RED at every one of these committed prebuilts**, including `f22947019 "preserve imported f64 module constants (Wave11e Defect A)"`. The green `824f687d…` binary recorded in `madaros_wave12_tip_green_receipt.v1.json` was a maintainer **local RAW build that was never committed as the prebuilt**; the committed prebuilts in this window do not reproduce it.

## Localization (measured)

Witness: `tests/run-pass/imported_f64_lognormal_science.sio` — `main` does `use stats::densities::{lognormal_pdf}` and calls `lognormal_pdf(1.0, 0.0, 1.0)`.

`stdlib/stats/densities.sio`:
- line 43: `fn de_ln(x: f64) -> f64` — **local to the module** (not imported)
- line 71: `let DE_LN_SQRT_2PI: f64 = 0.9189385332046727` — **the only module-level global in the file**
- line 132–134: `lognormal_pdf` body: `z = (de_ln(x)-mu)/sigma`; `logf = 0.0 - de_ln(x) - de_ln(sigma) - DE_LN_SQRT_2PI - 0.5*z*z`; `return de_exp(logf)`

Sub-witness matrix at HEAD (committed prebuilt = fresh build):

| Witness / call | Result | Note |
|---|---|---|
| `imported_f64_global_const.sio` | **OK** | simple imported f64 const mechanism works |
| `imported_module_f64_const.sio` | **OK** | imported BSS f64 const works |
| `imported_module_f64_const_bare_ident.sio` | **OK** | bare-ident imported f64 const works |
| `lognormal_cdf(1,0,1)` → 0.5 | **OK** | uses local `de_ln` + `normal_cdf`; no `DE_LN_SQRT_2PI` |
| `gamma_pdf(1,1,1)` → 0.36788 | **OK** | uses local `de_ln`, `de_exp`, imported `lgamma`; no `DE_LN_SQRT_2PI` |
| `lognormal_pdf(1,0,1)` → 0.39894 | **FAIL** | `9.999…e-301`; the only fn reading `DE_LN_SQRT_2PI` |

Conclusion: `de_ln` / `de_exp` / `lgamma` / `normal_cdf` all compute correctly across the module import boundary. The defect is isolated to the **load of the module-level f64 const `DE_LN_SQRT_2PI` inside `lognormal_pdf`'s compiled body** under the multi-module native path.

Arithmetic check (mu=0, sigma=1 ⇒ `de_ln(x)=de_ln(1)=0`, `z=0`):
`logf = -C`, observed `logf = ln(9.999…e-301) = -690.7755` ⇒ **C reads ≈690.78** instead of `0.9189385`.

## Relationship to prior Defect A

The witness comment documents the original Defect A: multi-module parse resets *wiped* the const to zero ⇒ `logf=0` ⇒ `p=exp(0)=1.0`. Commit `f22947019` stopped the zeroing (the "collapsed to 1.0" guard no longer trips), but the const now resolves to a **garbage non-zero value**. This is an **incomplete fix**, not a fresh regression: the failure mode moved from `p=1.0` to `p=1e-301`.

## Reproduce

```bash
# on an x86-64 host, fresh clone @ 3e7ed9f52
export SOUNIO_STDLIB_PATH="$(pwd)/stdlib"; unset SOUNIO_SOUC_ENGINE; ulimit -s unlimited
./bin/souc run tests/run-pass/imported_f64_lognormal_science.sio
#   lnpdf_bits 118622047889322840   (= 9.999999999999999e-301)
#   exp_bits   4600858325139338834  (= 0.39894228040143276)
#   FAIL lognormal_pdf (imported DE_LN_SQRT_2PI likely zero)
```

## Proposed fix direction (for the assignee — not applied here)

The BSS-slot seeding for imported module-level f64 constants (see `self-hosted/compiler/module_frontend.sio` around the `GLOBAL_VAR_INIT_*` / "seed BSS slots at lower time" comment, ~L5940–5990) preserves the const's *presence* but resolves the wrong value/offset when the const is consumed inside a `pub fn` of the imported module. Suggested steps:

1. Instrument the lower-time BSS seed for `DE_LN_SQRT_2PI`: print the assigned BSS offset and the f64 word written, vs the offset read at the `DE_LN_SQRT_2PI` use-site in `lognormal_pdf`.
2. Confirm whether the seed writes `0.9189385…` to slot S but the load in `lognormal_pdf` reads slot S′ ≠ S (offset mismatch), or the seed itself writes a wrong word.
3. Contrast with `gamma_pdf`/`lognormal_cdf`, which do not reference a module-const and compile correctly — the divergence isolates the const-use lowering, not the general f64 call path.

## Scope note / claim boundary

- **Do not** describe the Wave13 showcase as "closed / full-green" at HEAD. Reproduced state is **5/6**.
- Solid claims: `cd_exact` / `cd_exact_e2e` (ZD PROVED), `dual`, `order_spread` (N=4 spread ≈ 2.044226), `k95` (finite-dof Student-t = 2776).
- Open: `imported_f64` (`DE_LN_SQRT_2PI` garbage read) blocks `wave12_tip_green`, hence the showcase.

## Related

- `docs/audit/MADAROS_WAVE13_SHOWCASE_2026-07-21.md` — claims `showcase_verdict: pass_full` (measured against the pre-#1392 local RAW `824f687d`, not reproducible at HEAD)
- `docs/audit/MADAROS_IMPORTED_MODULE_F64_CAST_BITCAST_2026-07-14.md` — D-family imported f64 native-path lineage
- `docs/audit/MADAROS_IMPORTED_MODULE_NATIVE_PATH_ESCALATION_2026-07-14.md`
