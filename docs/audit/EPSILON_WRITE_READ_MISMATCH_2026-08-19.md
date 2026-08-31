<!-- docs:meta
topic_id: repo.docs.audit.epsilon-write-read-mismatch-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli5
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.epsilon-write-read-mismatch-2026-08-19
-->

# Written `epsilon:` is not read `.epsilon`

**Decision: CONSTRUTOR NÃO VALIDA.** Madaros `check` accepts any Knowledge
field name once `value` is present, including the invented `epsilom:`,
with no diagnostic. An invented **third** name writes the confidence
slot by position. That is larger than a swapped alias.

lean_single warns on unknown names and still emits an ELF (`rc=0`).
It does alias `epsilon` ↔ `confidence` by name. Madaros does not.

`self-hosted/` and `stdlib/darwin_pbpk/epistemic_pbpk28.sio` were not
edited.

---

## Instrument

| Field | Value |
|---|---|
| SHA | `ef69121b26` (`origin/main`, contains #2024) |
| Node | `cpuops-t560-proxmox` (`workspace_visible=no`) |
| Stamp | `2026-08-19T20:26:44Z` |
| Madaros | `bin/souc` default → `Madaros v0.80.0` |
| lean_single | `SOUNIO_SOUC_ENGINE=lean_single` |
| Driver | [`scripts/dev/epsilon_write_read.sh`](../../scripts/dev/epsilon_write_read.sh) |
| Isolates | [`docs/audit/repro/epsilon/`](repro/epsilon/) |
| Table | [`EPSILON_WRITE_READ_MISMATCH_2026-08-19.tsv`](EPSILON_WRITE_READ_MISMATCH_2026-08-19.tsv) |

---

## Controls

**Positive.** Write `value: 12.4`, read `.value`:

| Engine | check rc | run |
|---|---:|---|
| Madaros v0.80.0 | 0 | **12.400000** |
| lean_single | 0 | **12.400000** |

The isolate reads fields. A later `.epsilon` of `0.0` is not a dead printer.

**Negative.** Write `epsilom: 0.42` (invented name):

| Engine | check | run `.value` / `.confidence` / `.epsilon` |
|---|---|---|
| Madaros v0.80.0 | **rc=0, silence** | 1.000000 / 0.000000 / 0.000000 |
| lean_single | rc=0, `warning: unknown field` | 1.000000 / 1.000000 / 1.000000 |

Madaros does not validate constructor names. lean_single warns and
drops the unknown write (`.confidence` stays the default `1.0`, not
`0.42`).

---

## 1. Where does written `epsilon:` go?

**Madaros.** `checker_check_struct_lit_inplace` special-cases
`Knowledge` and only requires a field named `value`. Other names are
typechecked as expressions and otherwise ignored. Lowering
(`field_idx_from_struct_literal_name`) matches `value` / `variance` /
`confidence` against the three-slot IR layout. `epsilon` is not in
that layout, so it is stored at **`default_idx` = position in the
literal**. Dissertation order is `value`, `variance`, `epsilon`,
`provenance` → `epsilon` is index 2 → IR field 2 = `confidence`.

**lean_single.** Write of `epsilon` and `confidence` is one branch
(`lean_single.sio` ~10447, ~10710). Both store the confidence slot.
Unknown names take the `tc_error` / warning path, not a hard refuse.

Call form `Knowledge(v, ε=…, prov=…)` on Madaros lowers the second
argument into field 2 by construction (`lower_knowledge_ctor_call_ref`),
not by the struct-literal name table.

---

## 2. Where does `.confidence == 0.65` come from?

It is the authored `c[0]`, not a default.

Sentinel isolate writes `epsilon: 0.42` in dissertation field order:

| Engine | `.value` | `.variance` | `.confidence` | `.epsilon` |
|---|---|---|---|---|
| Madaros v0.80.0 | 1.000000 | 2.000000 | **0.420000** | **0.000000** |
| lean_single | 1.000000 | 2.000000 | 0.420000 | 0.420000 |

Default confidence on lean_single when the slot is not written is
**1.0** (`epsilom` negative). 0.42 is therefore the write, not a
constant. #2024's 0.65 is `c[0]`.

Invented third name `epsilom: 0.42` after `value` and `variance`:

| Engine | `.confidence` |
|---|---|
| Madaros v0.80.0 | **0.420000** (any third name fills slot 2) |
| lean_single | **1.000000** (unknown dropped; default) |

Writing the layout name `confidence: 0.42`:

| Engine | `.confidence` | `.epsilon` |
|---|---|---|
| Madaros v0.80.0 | 0.420000 | 0.000000 |
| lean_single | 0.420000 | 0.420000 |

Madaros `.epsilon` is not an alias. It typechecks as `f64` and the
field-get lowering has no layout entry, so the name hashes to a vacant
index (`'e' % 64 = 37`) and reads 0.0. lean_single loads offset 16 for
both names.

---

## 3. Write set versus read set (`TyKnowledge`)

| Name | Madaros write | Madaros read | lean_single write | lean_single read |
|---|---|---|---|---|
| `value` | required | yes (E170) | yes | yes |
| `variance` | by name | yes | yes | yes |
| `uncertainty` | by position if used | no (E, no field) | yes (σ² via square) | yes (√variance) |
| `confidence` | by name | yes | yes (alias of epsilon) | yes (offset 16) |
| `epsilon` | **position only** | typechecks; runtime 0.0 | yes (alias of confidence) | yes (offset 16) |
| `provenance` / `prov` | ignored / position | no | ignored | no |
| any other name | **accepted, stored by position** | no (no-field error on access) | warning, dropped | no (unknown field access) |

The sets differ. A constructor that accepts names the reader does not
know loses data without a Madaros diagnostic. That is the defect.

---

## 4. Dissertation dependence

`stdlib/darwin_pbpk/` contains **zero** `.epsilon` reads. The live
writers are `epistemic_pbpk28.sio` (lines 293–299, 566–572) and
`validation/pbpk28_mc_cross_validation.sio` (line 520). The GUM loop
reads `priors.kn[i].value`, `.variance`, `.confidence`
(`ep28_run` ~366–418).

TEST 5 (`sens[0]`) survives on `value`/`variance`. TEST 6
(`auc_blood_conf`) reads `.confidence`. Under dissertation field
order that slot **does** hold `c[i]` on both engines (Madaros by
position, lean_single by alias). Nothing downstream of those writers
reads `.epsilon` and therefore nothing in this tree observes the
Madaros 0.0.

The lost name is `.epsilon` as a **reader**. The authored number is
not lost from GUM, given the current literal order. Reordering
`epsilon` ahead of `variance` would zero Madaros `.confidence`
(#2027 reorder isolate). That is a silent order dependence, not a
named alias.

---

## What this does not close

- Repair. Forense only.
- Whether `epsilon` should become a real fourth slot or a named alias
  of `confidence` on Madaros.
- Rebuilding Madaros from `self-hosted/` (prebuilt ELF used).

---

## Files

| Path | Role |
|---|---|
| `docs/audit/EPSILON_WRITE_READ_MISMATCH_2026-08-19.md` | this receipt |
| `docs/audit/EPSILON_WRITE_READ_MISMATCH_2026-08-19.tsv` | one row per cell |
| `scripts/dev/epsilon_write_read.sh` | Slurm packer |
| `docs/audit/repro/epsilon/*.sio` | isolates |
