<!-- docs:meta
topic_id: repo.docs.audit.zd-mut-spine-2026-08-20
authority: repo_only
audience: users
last_validated: 2026-08-20
validated_by: cursor-3 (Slurm souc-build-remote --gate witness, gpuorangefs-r770-proxmox)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.zd-mut-spine-2026-08-20
-->

# ZD family on the `*mut` type spine (2026-08-20)

**Verdict: the eight ZD wrappers now refuse and accept at parameter
position.** `fn r(x: ExactlyPrivate<f64>) -> f64` without `with ZD`
emits **E201**. The same function with `with ZD` checks. Ordinary
`i64` parameters still check.

SHA built: worktree `lane/cursor-3/zd-mut-spine-20260820` on
`origin/main` `67aa2aec12`. Slurm node `gpuorangefs-r770-proxmox`.
Engine: Madaros built from this tree (`madaros.elf` 100594218 bytes).
`self-hosted/` was edited. The dissertation tree was not.

## Why match arms were theatre

`checker_lower_type_expr_mut` already had a `_ =>` for the twenty
ambitious kinds. Adding `TypeExprKind::TypeExactlyPrivate => …` arms
compiled, and the ELF grew, but a forced-refuse arm still printed
`check: OK`.

The seed miscompiles `match <enum-field>` inside a `*mut` function:
discriminant 0 lands on the first arm (`TypeNamed`). Every
`ExactlyPrivate<T>` therefore arrives as a **name plus type_args**,
the same path `Seq` and `Hyper` already use. A non-`*mut` tag helper
did not see kind 13 either.

The live interrogation is therefore name-based, next to those
existing special cases, in `checker_lower_named_type_with_args_mut`.

## Positive control (item 1)

Forced refuse, rebuild, confirm fail. Three rungs, all on this
branch, no checkout change while the remote pack ran:

| Control | What was forced | ordinary_i64 | ExactlyPrivate with ZD |
|---|---|---|---|
| Unconditional E201 at the top of `checker_lower_type_expr_mut` | every type | **refused E201** — the function is on the `check` path | — |
| Refuse unless `te.kind == TypeNamed` | non-Named kinds | pass | **pass** — the kind compares as Named |
| Refuse the **name** `ExactlyPrivate` | name path | pass | **refused E201** — this is the live path |

Only after the third rung fired was the real ZD logic written.

## Two other silences that hid the same defect

1. **Collect lowers parameters before the function's `with` clause
   was current.** `with ZD` would have looked like a missing effect.
   Collect now installs declared effects, lowers params and the
   return type, then restores.
2. **`check_items_verdict_boot4` discarded collect-time errors.** It
   inspected only the check-pass `had_error` bit, then printed
   `check: OK` / rc=0. It now fails if `error_count > 0` or
   `had_error` after collect+check. Unused functions still lower
   their parameters.

## Witnesses (17/17)

`SOUNIO_WITNESS_GLOB='tests/audit/zd_mut_spine/*.sio'` against the
from-source ELF. Each compile-fail file has a live caller from
`main` (a declaration alone is not a test).

| File | Expect | Result |
|---|---|---|
| `exactly_private_param_nozd.sio` | E201 | pass |
| `exactly_private_param_with_zd.sio` | accept | pass |
| `forgettable_param_nozd.sio` | E200 | pass |
| `forgettable_param_with_zd.sio` | accept | pass |
| `editable_param_nozd.sio` | E202 | pass |
| `editable_param_with_zd.sio` | accept | pass |
| `capability_gated_param_nozd.sio` | E203 | pass |
| `composable_param_nozd.sio` | E204 | pass |
| `audited_param_no_witness.sio` | E205 | pass |
| `audited_param_with_zd_witness.sio` | accept | pass |
| `revivable_param_no_temporal.sio` | E206 | pass |
| `revivable_param_with_zd_temporal.sio` | accept | pass |
| `interpretable_param_nozd.sio` | E207 | pass |
| `exactly_private_locus_ok.sio` | accept `e3e10` | pass |
| `exactly_private_locus_malformed.sio` | E208 `e3e3` | pass |
| `exactly_private_locus_out_of_range.sio` | E208 `e3e99` | pass |
| `ordinary_i64_param.sio` | accept | pass |

The `ExactlyPrivate<T, A>` locus interrogation from
`feat/exactly-private-ta` is on this path (parser optional second
argument + `zd_locus_is_wellformed`). Narrowing from a well-formed
index pair to one of the 84 `validPrims` is still owed.

## Ratchet

`scripts/ci/silent_type_spine_ratchet_gate.sh` is in the tree.
ZD names are off the kind list. Frozen count **19 → 10**. The ten
are `Proof` (1) plus `Counterfactual`/`Intervention` crossings in
`tests/frontend/`. Proof, causal, DP, aleatoric, and session kinds
remain scaffolding.

## What this does not claim

- It does not claim the `*mut` `match te.kind` now sees ambitious
  kinds. It does not. The name path is the one that runs.
- It does not claim lean_single changed. The gate is Madaros from
  this source.
- It does not bring the remaining twelve kinds across.
