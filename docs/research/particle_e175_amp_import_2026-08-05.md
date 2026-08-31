<!-- docs:meta
topic_id: repo.docs.research.particle-e175-amp-import-2026-08-05
authority: historical
audience: researchers
last_validated: 2026-03-07
validated_by: A6
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.research.particle-e175-amp-import-2026-08-05
-->


<!-- docs:status-note:start -->
> Docs status: `historical`
> This page is preserved for lineage. Start at [Docs Authority Matrix](../governance/DOCS_AUTHORITY_MATRIX.md) and [docs index](../README.md) for the current canonical surface for this topic.
<!-- docs:status-note:end -->

# Particle E175 — Madaros import of `eemm_z_amplitude_nu` (2026-08-05)

## Claim

Under default Madaros, `examples/particle_physics/exp13_amplitude_honesty.sio`
(which imports `particle_physics::nonunitary_amp::eemm_z_amplitude_nu`) type-checks
and runs with `PARTICLE_EXP13_OK`, matching lean_single.

## Symptom (pre-fix)

```text
SOUNIO_SOUC_ENGINE=madaros ./bin/souc check examples/particle_physics/exp13_amplitude_honesty.sio
→ many error[E175] function is private in its defining module
```

Bisect:

| Program | Madaros `check` |
|---|---|
| `lorentz` alone | OK |
| `complex::lib` alone | OK |
| `complex` + `lorentz` | **E175** |
| `spinor` (pulls both) | **E175** |
| full EXP13 | **E175** |

lean_single accepted the same sources throughout.

## Root cause

`stdlib/particle_physics/lorentz.sio` (and `vertex.sio`) declared

```sio
extern "C" {
    fn sqrt(x: f64) -> f64;
    // lorentz also: sinh, cosh
}
```

Those bindings are **private** fn sigs in the multimodule table. Madaros
`fn_sig_table_find_prefer_module` prefers same-module, else any **non-private**,
else the first free-fn match. `complex::lib` calls builtin `sqrt` / `sinh` /
`cosh` with no local definition, so lookup fell through to lorentz/vertex's
private extern stubs → false E175.

Same class as the gum+knowledge helper-name collision (#1245), but the colliding
names were **private extern builtins**, not `chk`/`near`. Related: #1622 (extern
stubs return 0 under native — builtins need no declaration).

## Fix

1. Drop the private `extern "C"` blocks in `lorentz.sio` and `vertex.sio`.
2. Call builtins (`sqrt`, `sinh`, `cosh`) directly — no declaration.
3. In `spinor.sio`, import `metric_eta` from `lorentz` instead of a local private
   duplicate (hygiene; not the E175 trigger once externs were removed).

No change to Madaros checker sources (claimed elsewhere); no formula change.

## Evidence

```bash
bash scripts/research/particle_e175_amp_import_gate.sh
# → PARTICLE_E175_AMP_IMPORT_GATE_OK
# also: PARTICLE_EXP13_GATE_OK (lean_single + madaros)
```

Minimal witness after fix: complex + lorentz `check: OK` under Madaros.

## Follow-up (2026-08-06 trilogy)

Broader `particle_physics` extern sweep + #1627 checker hardening + EXP14
`eemm_z_amplitude_nu` restore: see `docs/research/particle_e175_trilogy_2026-08-06.md`.
