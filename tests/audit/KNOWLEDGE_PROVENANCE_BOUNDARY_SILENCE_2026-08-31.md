# Knowledge provenance boundary silence — 2026-08-31/09-01

**At the `Knowledge[...]` call boundary of the semantic-clock engine, only
epsilon is real.** A value annotated `Computed` flows into a `Measured`
parameter with no diagnostic, in every engine leg measured. The validity slot
— enforced by the lean_single bootstrap engine — is structurally inert under
Madaros. Provenance is enforced nowhere.

This receipt does not change the language. It makes the asymmetry audible.

Lineage: next bridge of the observer surface survey
(`artifacts/audit/observer_surface_survey_20260831.md`, static reading), which
found `check_knowledge_type` dropping validity/provenance before `TypeEntry`
(`self-hosted/check/epistemic.sio:49`, "TypeEntry does not yet persist full
validity/provenance metadata") and `knowledge_meta_from_ty` rebuilding every
meta as always-valid/DERIVED (:497-530). Sibling of the 2026-08-19 silent-drop
audit (`tests/audit/KNOWLEDGE_ANNOTATION_SILENT_DROP_2026-08-19.md`).

## Measured matrix (from-source Madaros, branch tip, Slurm r770, 246 s build)

| Slot at the call boundary | Madaros `check` | Madaros `compile` | lean_single raw |
|---|---|---|---|
| epsilon | **enforced** (E036 fires) | — | enforced (P0003, coarse) |
| validity | **inert** | **inert** | **enforced** ("Temporal validity window mismatch") |
| provenance | **inert** | **inert** | **inert** |

Capability controls (each leg must prove it *can* refuse before its silence
counts): E241 fixture refuses under Madaros `check` and `compile`;
`silence_lean` under Madaros `check` fires E036 at the call boundary, proving
the boundary machinery runs there; the covid-shaped control refuses under
lean_single. All fired.

The structural explanation is the survey's: `knowledge_call_boundary_compatible`
(`self-hosted/check/epistemic.sio:908-924`) does call `validity_subsumes` and
`provenance_subsumes` — but it feeds both from `knowledge_meta_from_ty`, which
rebuilds every meta as `validity_always()` / `PROVENANCE_KIND_DERIVED`. The
comparison runs; its inputs are pre-flattened. The sockets are wired; the
wires carry nothing.

## Engine/mode discipline (measured, do not relearn the hard way)

- Raw positional `souc <src> <out>` is the **lean_single** bootstrap engine.
  Madaros speaks verbs (`check`, `compile`, `run`). A "mode" name in this repo
  often names an *engine*.
- E241 (unknown annotation component) refuses in Madaros `check` and
  `compile`; lean_single raw mode compiles the same fixture clean.
- The committed `bin/madaros-linux-x86_64` predates E241 entirely (the E241
  fixture compiles under it, measured 2026-08-31, md5 ff69dae4). Committed-ELF
  rows are STALE-SUSPECT by construction; the authoritative rows above are
  from a from-source build (`scripts/ci/knowledge_provenance_boundary_gate.sh`,
  Slurm partition `bench`).
- lean_single's `ty_eq` is annotation-hash-based: an explicit `eps < 0.05`
  bound in the annotation trips its generic P0003 at the `let`. The lean probe
  therefore carries no eps bound; the Madaros probes carry `eps < 0.05` on
  BOTH sides, so the epsilon surface is satisfied and only the probed slot can
  disagree.

## The confound this receipt had to survive

A first version of the provenance probe (no eps bound in the annotation) was
refused under Madaros `check` — `error[E036]`, "confidence bound is not tight
enough". That refusal read as provenance enforcement and the gate's first
verdict was a false "pass". The diagnostic named epsilon, not provenance.
The gate's classifier now falsifies the accusation **only** on a refusal that
names provenance; any other refusal is reported as a confounded probe, never
as enforcement. The lesson is the audit's subject in miniature: a refusal —
like an absence — is only evidence *for the cause it names*.

## Files

| File | Role |
|---|---|
| `tests/audit/knowledge_provenance_boundary_silence.sio` | provenance-only mismatch, Madaros legs (eps satisfied both sides) |
| `tests/audit/knowledge_provenance_boundary_silence_lean.sio` | provenance-only mismatch, lean_single leg (no eps bound) + E036 capability probe under Madaros check |
| `tests/audit/knowledge_validity_boundary_control.sio` | Madaros-leg validity control (eps satisfied; only the window disagrees) |
| `tests/audit/knowledge_validity_boundary_control_lean.sio` | lean-leg validity control (covid shape) |
| `tests/audit/knowledge_literature_e241_probe.sio` | unwriteable provenance name → E241 net |
| `scripts/ci/knowledge_provenance_boundary_gate.sh` | the instrument (Slurm from-source by default; local fallback labeled STALE-SUSPECT) |

## Gate

```text
bash scripts/ci/knowledge_provenance_boundary_gate.sh
```

The gate FAILS while the silence exists; it turns green when a provenance
mismatch is refused by a diagnostic that names provenance. It also REPORTS
the validity asymmetry (Madaros legs) without verdict. Machine-readable
verdict: `artifacts/audit/knowledge_provenance_boundary/status.json`.
Not wired into CI — a red-by-design accusation belongs to whoever picks up
the fix, not to every PR.

## What closes it

`TypeEntry` persisting validity/provenance metadata (the comment at
`self-hosted/check/epistemic.sio:49` is the unwired organ), so that
`validity_subsumes` / `provenance_subsumes` receive real inputs. When that
lands, this gate is the witness that it works — and the validity half of the
matrix says the fix has TWO slots to unflatten, not one.

## What this receipt is not

- Not a claim that a wrong number has shipped; it is a claim about which
  declared boundaries do not exist in which engine.
- Not a design for `Measured<T> by O`; the observer butterfly stays
  un-promoted.
- Not a criticism of the covid compile-fail fixtures: they pin real
  lean_single behavior. This receipt adds which engine that behavior is not
  shared by.
