<!-- docs:meta
topic_id: repo.docs.audit.e200-knowledge-constructor-reverify-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.e200-knowledge-constructor-reverify-2026-08-19
-->

# E200 Knowledge constructor reverify (independent, grok-cli5)

**Decision: ACEITE** (compiler builtin `TypeKind::TyKnowledge`; both engines
`check` the dissertation literal; `kn` is the GUM carrier).

The memory claim that `souc` still refuses
`Knowledge { value, variance, epsilon, provenance }` with **E200** is
**false** on the binaries staged at `b76ba90257`. E200 did not appear on
either engine. Dissertation modules are not forced onto raw `f64` by this
constructor.

Cursor-1 filed the same ACEITE on `origin/main` as
[`E200_KNOWLEDGE_REVERIFY_2026-08-19.md`](E200_KNOWLEDGE_REVERIFY_2026-08-19.md)
(#2024, SHA `f4dc51777e`). This lane re-measured independently on
`b76ba90257` and agrees on acceptance, builtin resolution, carrier,
and workflow-unreachability. It **does not** agree that Madaros
refuses unknown Knowledge field names: #2024's negative fixture
omits `value`, so its E012 is the missing-`value` rule, not a field
allowlist. Split controls below.

This is a measurement. `stdlib/darwin_pbpk/epistemic_pbpk28.sio` and
`self-hosted/` were not edited.

---

## Instrument

| Field | Value |
|---|---|
| SHA of sources / staged binaries | `b76ba90257` (`origin/main` at branch creation) |
| Submitter | workspace pod; `/workspace` is **not** visible on the node |
| Launch | `scripts/dev/e200_knowledge_reverify.sh` → `srun`, partition `cpu-ops` |
| Node | `cpuops-t560-proxmox` |
| `workspace_visible` | no |
| `orangefs_visible` | yes |
| Stamp | `2026-08-19T20:19:27Z` |
| Madaros | `bin/souc` default → `Madaros v0.80.0` (`bin/madaros-linux-x86_64`, sha256 `437bdd8f96a2…`) |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single` → `bin/souc-lean-single-x86_64` (`Usage: mini_native <source.sio> <output>`) |
| Poison | `env -u SOUC_BIN -u SOUNIO_SOUC_BIN` on the node |
| Stack | `ulimit -s 1048576`, `MADAROS_STACK_KB=524288` |
| Table | [`E200_KNOWLEDGE_CONSTRUCTOR_REVERIFY_2026-08-19.tsv`](E200_KNOWLEDGE_CONSTRUCTOR_REVERIFY_2026-08-19.tsv) |
| Driver | [`scripts/dev/e200_knowledge_reverify.sh`](../../scripts/dev/e200_knowledge_reverify.sh) |
| Isolates | [`docs/audit/repro/e200/`](repro/e200/) |

`souc check` under `SOUNIO_SOUC_ENGINE=lean_single` is a **compile**
(`souc SRC TMP`); it is not typecheck-only. Madaros `check` is typecheck-only.
Every row names its engine.

---

## 1. Does `souc check epistemic_pbpk28.sio` pass?

| Engine | Verb | rc | Diagnostic | Line |
|---|---|---:|---|---|
| Madaros v0.80.0 | `check` | **0** | none (`check: OK`, 3 modules, `verdict=0`) | — |
| lean_single | `check` (= compile) | **0** | none (ELF 133203 bytes, 63 fns) | — |

E200 is absent. The constructor at lines 292–299 is live code and is accepted.

---

## 2. What is `Knowledge`?

A **compiler builtin**, not a stdlib struct and not the ecosystem
`struct Knowledge[T] { value, ε, prov, metadata }`.

The file imports only `darwin_pbpk::core::pbpk28_params` and
`darwin_pbpk::tsit5_pbpk28`. There is no `Knowledge` declaration in
that import closure.

Madaros resolves the ident in `self-hosted/check/check.sio`:

- call form `Knowledge(v, ε=…, prov=…)` → `call_expr_is_builtin_knowledge_ctor`
  + `checker_check_knowledge_ctor_expr_inplace` (first argument types the
  inner `T`; epsilon is left unspecified at the type level);
- struct literal `Knowledge { … }` → `checker_check_struct_lit_inplace`
  special-case on name `"Knowledge"` / `check_knowledge_struct_lit`.
  Only the field named `value` is required. Other field names are
  typechecked as expressions and otherwise ignored.

IR layout (`ir_register_knowledge_layout`) is three `f64` slots:
`value`, `variance`, `confidence`. `provenance` has no slot.
`.epsilon` typechecks as `f64` and reads **0.0** under Madaros
(pre-existing lowering gap, reproduced below).

Current Madaros **E200** is *not* “undefined identifier”. It is
“Forgettable type requires ZD effect”. lean_single E200 remains
“unknown / undefined identifier”. Neither code fired on this constructor.

---

## 3. Numeric path — carrier, not decorative

`m`, `v`, `c` are locals inside `ep28_rapamycin_priors` /
`ep28_semaglutide_priors`. They exist only to fill `kn`. `ep28_run`
reads `priors.kn[i].value`, `.variance`, `.confidence`. There is no
second GUM path that reads the raw arrays.

Removing the `kn` block would not compile: `EpPrior28` requires
`kn: [Knowledge[f64]; 8]`, and the GUM loop has no other source.
That is source, not a mutation of the dissertation file.

Isolate `numeric_carrier.sio` (same field order as the priors:
`value`, `variance`, `epsilon`, `provenance`) then print the four
accessors:

| Engine | `.value` | `.variance` | `.confidence` | `.epsilon` |
|---|---|---|---|---|
| Madaros v0.80.0 `run` | 12.400000 | 22.202944 | **0.650000** | **0.000000** |
| lean_single `run` | 12.400000 | 22.202944 | 0.650000 | 0.650000 |

Authored `ε` therefore lands in the GUM confidence slot on both
engines when the literal is written in dissertation order. Madaros
`.epsilon` reads 0.0; lean_single aliases `.epsilon` to confidence.

Reorder isolate (`value`, `epsilon`, `variance`, `provenance`):

| Engine | `.value` | `.variance` | `.confidence` |
|---|---|---|---|
| Madaros v0.80.0 `run` | 12.400000 | 22.202944 | **0.000000** |
| lean_single `run` | 12.400000 | 22.202944 | **0.650000** |

Madaros fills unknown names by **position** (`default_idx`). `epsilon`
is not a named alias of `confidence`; it only hits slot 2 when it is
the third field. lean_single maps `epsilon` by name. Dissertation
priors use the order that happens to work on both.

`provenance` is written and discarded (no IR slot). That string is
decorative. The `value` / `variance` / (positional-or-named) confidence
payload is not.

---

## 4. Controls

**Positive.** Same folder, `stdlib/darwin_pbpk/test_sort_fix.sio`,
same `souc check` command:

- Madaros: rc=0, `check: OK`
- lean_single: rc=0, ELF produced

The command is not broken. The target pass is a fact about the target.

**Negative (wrong field, `value` present).**
`Knowledge { value: 1.0, mystery: 99.0 }`:

- Madaros `check`: **rc=0**, no diagnostic
- lean_single `check`: **rc=0**, `warning: unknown field in Knowledge literal`

The compiler is **not** refusing unknown Knowledge field names when
`value` is present. That is the finding the dispatch asked for. It is
not E200.

**Negative (no `value`).** `Knowledge { mystery: 99.0, provenance: "none" }`:

- Madaros `check`: **rc=1**, `error[E012]` (“this type has no field named”)
- lean_single `check`: **rc=0**, warning only

This is the same shape as #2024's `tests/audit/e200_knowledge_wrong_fields.sio`
(`not_a_field` + `also_wrong`, no `value`). Madaros E012 there is the
missing-`value` rule, not an allowlist of field names. Extra names with
`value` present are accepted on both engines. lean_single never hard-fails
either negative.

---

## 5. Gate reachability

`scripts/ci/dissertation_pbpk_suite_gate.sh` **does** list
`epistemic_pbpk28` (`stdlib/darwin_pbpk/epistemic_pbpk28.sio`) and runs
it with `souc run`, not `check`. The gate pins
`SOUC_BIN` to `scripts/ci/souc-seq-leansingle.sh` (lean_single compile+run).

It is invoked from `scripts/ci/native_v2_cpu_compiler_umbrella_gate.sh`.
That umbrella is invoked from `scripts/ci/native_v2_frontend_convergence_gate.sh`.

None of those three names appears in `.github/workflows/`. Re-ran
`python3 scripts/dev/ci_gate_workflow_reachability.py` on this SHA:

- `dissertation_pbpk_suite_gate.sh` is in `dissertation_leftover`
- census class on the committed TSV: `manual-by-design`, workflow=`no`
- umbrella and frontend-convergence: leftover, workflow=`no`

The suite is operator-reachable (umbrella / Slurm / hand). It is
**not** workflow-reachable. That is not a forgotten wire; it is the
census class.

---

## What this does not close

- Rebuilding Madaros from `self-hosted/` (not done; prebuilt ELF used).
- Whether a future field-name check should refuse `epsilon` or accept it
  as an alias of `confidence`.
- The TEST 6 aggregator formula (already a separate receipt).
- Running the full PBPK28 GUM job. Check + isolate `run` are enough
  for constructor acceptance and slot occupancy.

---

## Files

| Path | Role |
|---|---|
| `docs/audit/E200_KNOWLEDGE_CONSTRUCTOR_REVERIFY_2026-08-19.md` | this receipt |
| `docs/audit/E200_KNOWLEDGE_CONSTRUCTOR_REVERIFY_2026-08-19.tsv` | one row per measured cell |
| `scripts/dev/e200_knowledge_reverify.sh` | Slurm packer (stdin tarball) |
| `docs/audit/repro/e200/*.sio` | isolates and controls |

Not edited: `stdlib/darwin_pbpk/epistemic_pbpk28.sio`, `self-hosted/`.
