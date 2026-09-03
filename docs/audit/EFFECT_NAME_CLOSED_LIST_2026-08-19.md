<!-- docs:meta
topic_id: repo.docs.audit.effect-name-closed-list-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.effect-name-closed-list-2026-08-19
-->

# Effect-name closed list — accusation gate

**Filed:** 2026-08-19 · **Lane:** grok-cli5 · **Status:** gate live, expected red

## Why

An unrecognised `with` name is not a diagnostic. `effect_name_to_id`
returns −1. `collect_effects_from_list` then admits a name only when
`eff_id >= 0 && n < 8`. The miss is dropped with no code. Three
independent instruments already measured the hole:

1. The checker guard itself (`self-hosted/check/effects.sio`).
2. #1953 negative control `with Foo` → `check: OK`, silence, not E035.
3. #1993 `handle<NaoExisteEsteEfeito>` is indistinguishable from
   `handle<IO>` under Madaros (compile, run, exit 0).

This gate does **not** fix the compiler. It makes the hole visible and
stops a new invented name landing without a file:line. The compiler
fix is a separate founder dispatch. `self-hosted/` is untouched.

## Closed list — derived, not written

The dispatch text said 23 names. That was the handwritten-arm era.
On this `origin/main` the function is table-driven:

- `effect_named_id_max() = 28` → ids 0–28 (29 table names)
- `name_is_confidence` aliases `Confidence` onto id 8 (Epistemic)

**Measured closed set: 30 strings.** Derived by decoding
`effect_kind_name_len` / `effect_kind_name_byte` / `name_is_confidence`
from `self-hosted/check/effects.sio`. The gate does not contain a
handwritten name list. If the table recognises N names, N is N.

| Id | Name | Notes |
|---:|---|---|
| 0–22 | IO … Chaotic | phase 2a |
| 23–28 | Approx … NonUnitary | phase 2b extras |
| 8 (alias) | Confidence | same id as Epistemic, not a new variant |

`EffVar` (discriminant 29) is not a user-facing `with` name.
`Mod` is HELD: `effect_name_to_id("Mod")` returns −1. The scan
reports every live `with Mod`. It is not added to the closed list.

Network, Sensor, and Render each have five live uses. They stay on
the list because the table recognises them, not because of use count.

## Scan (versioned `.sio`)

Exclusions: `archive/`, `bootstrap/`, `self-hosted/bootstrap/`,
`*.sio.old`, `//` line comments, `/* */` block comments, and
`"string literals"`. The extractor stops on the parser's own
`comma + Ident + Colon` boundary so `with Approx, x: f64` does not
count `x` as an effect.

Measured on this worktree after the witness landed, against
`origin/main` `482051161c`:

| | |
|---|---:|
| `with` name tokens | 165975 |
| on the derived list | 163130 |
| not on the list | 2845 |
| distinct unknown names | 12 |

| Name | Count | What it is |
|---|---:|---|
| Mod | 2813 | HELD; used and not recognised |
| Compute | 8 | `stdlib/quantum/vqe.sio` |
| Foo | 5 | #1953 silence witnesses |
| Exp | 5 | `examples/pharmacokinetic_model.sio` |
| Log | 4 | same file |
| NomeQueNaoExiste | 2 | this gate's accusation witness |
| NoSuchEffectX | 2 | archaeology drop fixture |
| Choice | 2 | handler demo |
| Counter, Fetch, Fail, Sampling | 1 each | handler / sample fixtures |

Every unknown site is in the gate artefact. Stdout prints twenty
sites per name and refers the rest to the artefact.

## Accusation

Witness: `docs/audit/repro/effect_unknown_name.sio`
(`with NomeQueNaoExiste`).

Slurm, not the pod. Partition `cpu-ops`, host `cpuops-t560-proxmox`.
Tarball stdin (the node cannot see `/workspace`). Committed
`bin/souc` → Madaros v0.80.0.

```
souc=./bin/souc host=cpuops-t560-proxmox
check: OK          # witness, rc=0
check: OK          # positive with IO, rc=0
```

The compiler accepted the invented name in silence. The gate fails
while that remains true. That is the accusation, and the message
says so.

When a named diagnostic starts refusing the witness, the accusation
arm turns green. The scan arm stays red until every used name is
either recognised by `effect_name_to_id` or removed from the corpus.
Those are two different holes. Do not close the scan by adding Mod
to a handwritten list.

## Controls

`--self-test` (must stay green):

```
positive_io_not_accused=true
negative_comment_a_the_ignored=true
negative_string_ignored=true
negative_sio_old_ignored=true
status=pass
metrics {total=4, passed=4, failed=0, not_run=0}
```

Constructed cases: English `with a` / `with the` in comments,
`"with Mut"` in a string, and `with InventedFromOld` in a `.sio.old`.
None are counted. A real `with IO` is not accused.

Full gate (expected red today):

```
status=fail
metrics {total=165976, passed=163130, failed=2846, not_run=0}
```

165975 scan tokens + 1 accusation. 2845 unknown uses + 1 silence.

## Reachability

Wired as `.github/workflows/effect-name-closed-list.yml`
(pull_request, merge_group, push to main). `--self-test` is a hard
step. The scan + accusation step uses `continue-on-error` so the
signal stays visible without making every unrelated pull request
unmergeable. `ci.yml` is under another lane's claim; a dedicated
workflow is still a reachable gate. A gate that never runs is not
a gate (#1994).

## Reproduce

```bash
env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  bash scripts/ci/effect_name_closed_list_gate.sh --self-test

env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  bash scripts/ci/effect_name_closed_list_gate.sh --scan-only

# Accusation compile: Slurm when srun is present; GHA sets
# EFFECT_NAME_CLOSED_LIST_SLURM=0 and uses the committed ELF.
env -u SOUC_BIN -u SOUNIO_SOUC_BIN \
  bash scripts/ci/effect_name_closed_list_gate.sh
```

## What this is not

- Not a compiler fix. `self-hosted/` is not edited.
- Not a licence to add Mod, Compute, Exp, or Log to the closed list.
- Not a claim that 23 is the live count. The live count is derived.
- Not a merge. The hole stays red until the founder dispatches the
  checker diagnostic.

Topic-registry insert for this receipt is blocked this turn
(governance files held by another lane). The document is the
measurement; the registry row is a follow-up, not the evidence.
