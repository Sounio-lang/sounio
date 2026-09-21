<!-- docs:meta
topic_id: repo.docs.audit.e200-knowledge-reverify-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: cursor-1 (Slurm souc check/run, both engines)
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e200-knowledge-reverify-2026-08-19
-->

# E200 reverify — `Knowledge { value, variance, epsilon, provenance }` (2026-08-19)

**Verdict: ACEITE.** Both engines typecheck the dissertation constructor.
There is no `E200`. `Knowledge` is the compiler builtin
(`TypeKind::TyKnowledge`), not `ecosystem/shared/epistemic_types.sio`.
The `kn` block is the GUM carrier, not decoration.

SHA measured: `f4dc51777e` (`origin/main`). Slurm partition `cpu-ops`
(host `cpuops-t560-proxmox`). Compiler staged to
`/orangefs/training/sounio/cursor-1-e200`. The dissertation file and
`self-hosted/` were not edited.

## Controls

| Role | File | Madaros (`bin/souc` default, v0.80.0) | lean_single (`SOUNIO_SOUC_ENGINE=lean_single`) |
|---|---|---|---|
| Positive (same folder, no Knowledge ctor) | `stdlib/darwin_pbpk/constants.sio` | **check rc=0** (`check: OK`) | check rc=1, `error: no main` (lean_single check compiles to ELF and requires `main`) |
| Positive (same folder, has `main`) | `stdlib/darwin_pbpk/test_sort_fix.sio` | not required once constants was green | **check rc=0**, ELF emitted, **0 × E200** |
| Negative (wrong fields) | `tests/audit/e200_knowledge_wrong_fields.sio` | **check rc=1**, `error[E012]` `this type has no field named` | **check rc=0** with `warning: unknown field in Knowledge literal` (twice). Fields are **not** a hard refuse on lean_single |

The command is not broken: Madaros accepts the sibling library; lean_single
accepts a sibling with `main`. The negative control shows Madaros *does*
check Knowledge field names (E012, not E200). lean_single only warns.

## Target: `souc check stdlib/darwin_pbpk/epistemic_pbpk28.sio`

| Engine | rc | Diagnostic | Line |
|---|---|---|---|
| Madaros v0.80.0 | **0** | `check: OK` / `verdict=0` (3 modules) | none |
| lean_single | **0** | compile to ELF, `knowledge_subtype: 0 violations`, **E200 count = 0** | none |

Isolate programs (no PBPK imports) also check rc=0 on **both** engines:

- `tests/audit/e200_knowledge_literal.sio` — the struct literal
- `tests/audit/e200_knowledge_call.sio` — `Knowledge(0.0, ε=1.0, prov="unused")`

## What `Knowledge` resolves to

Builtin of the compiler. Not imported. The two versioned
`struct Knowledge[T]` files (`ecosystem/shared/epistemic_types.sio`,
and test-local structs) are a different type (ecosystem: `value, ε, prov,
metadata`; ε there is confidence in [0,1], and there is no `variance`).

Madaros field access for `TypeKind::TyKnowledge`
(`self-hosted/check/check.sio`): `.value` → inner `T` (needs
`with Epistemic`, E170), `.variance` → f64, `.confidence` → f64,
`.epsilon` → f64 at the type level. IR layout is three slots
(`ir_register_knowledge_layout`): value, variance, confidence.
`prov` / `provenance` has no IR slot.

Live `souc run` of `tests/audit/e200_knowledge_fields_print.sio`:

| Field | Madaros | lean_single |
|---|---|---|
| `.value` | 12.400000 | 12.400000 |
| `.variance` | 22.202944 | 22.202944 |
| `.epsilon` | **0.000000** | 0.650000 |
| `.confidence` | 0.650000 | 0.650000 |

That matches the in-tree comment: under Madaros `.epsilon` typechecks
and reads 0.0; `.confidence` holds the construction-site ε. The GUM
path in this file reads `.confidence`, not `.epsilon`.

## Numeric path — carrier, not decoration

`m` / `v` / `c` are locals of `ep28_rapamycin_priors`. Only `kn` is
stored on `EpPrior28`. `ep28_run` reads `priors.kn[i].value`,
`.variance`, and `.confidence`.

Node-local copy with the seven `kn[i] = Knowledge { … }` lines
stripped (array fill left as `Knowledge(0.0, ε=1.0, prov="unused")`,
variance 0). Dissertation file untouched.

| Run | Engine | TEST 5 `sens[0]` | zeroed components | summary |
|---|---|---|---|---|
| Original | Madaros | 0.697376 | 0 | TEST 5 PASS; TEST 6 FAIL (`AUC confidence: 4604219396932172800` — `print_f64` bit-pattern, already named in the file) |
| Original | lean_single | 0.697376 | 0 | **ALL 9 TESTS PASSED**, `AUC confidence: 0.671038`, rc=0 |
| Stripped `kn[i]=` | Madaros | **0.000000** | **7** | 5 pass / 4 fail (`Jacobian column lost`) |

Removing the `kn` writes changes the GUM numbers. The block is the
carrier. The parallel `f64` arrays are a construction-site projection
into that builtin, not a second kernel.

## Gate / workflow

`scripts/ci/dissertation_pbpk_suite_gate.sh` **does** list
`epistemic_pbpk28` and runs `souc run` on it (PASS-marker required).
It is called from `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`,
which is called from `scripts/ci/native_v2_frontend_convergence_gate.sh`.

**No `.github/workflows/*.yml` mentions any of those three scripts.**
The gate is not reachable from CI workflows on this SHA.

## What this does not claim

- It does not claim Madaros TEST 6 is green (the printed confidence
  is the known `print_f64` fabrication; lean_single prints 0.671).
- It does not claim `provenance` is stored at runtime (no IR slot).
- It does not claim lean_single refuses unknown Knowledge fields
  (it warns and still emits an ELF).
