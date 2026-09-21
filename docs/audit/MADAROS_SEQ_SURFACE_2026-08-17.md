<!-- docs:meta
topic_id: repo.docs.audit.madaros-seq-surface-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.madaros-seq-surface-2026-08-17
-->

# Madaros Seq surface — checker gap close (2026-08-17)

**Lane:** grok-cli5 / `madaros-seq-20260817`  
**Parent split:** `docs/audit/DISSERTATION_PBPK_SUITE_RESIDUAL_SPLIT_2026-08-17.md`  
**Instrument rule:** build Madaros from source before claims (`bin/souc` is prebuilt).

---

## 1. Where Seq was missing

| Layer | lean_single | Madaros (pre-fix) |
|-------|-------------|---------------------|
| **Checker** | full: `seq_new`, methods, index, `acknowledge` | **absent** — E137/E011/E013 |
| **Lowering / native codegen** | full TY_SEQ=12 emit path in monolith | **absent** in modular IR/native (no Seq symbols outside lean_single) |

**Verdict:** missing from **both**, but the dissertation residual dies at **check** first. Closing the checker is necessary and sufficient for `souc check` on kaxi + `ontology/model` import. Full `souc run` still needs a separate lowering/codegen port (not this commit’s claim unless verified).

---

## 2. Engine divergence (for grok-cli1 census)

lean_single **supports** Seq end-to-end (check + run on the minimal witness). Madaros modular did not. This is a **two-engine surface divergence**, not a test-source defect.

Witness (pre-fix prebuilt Madaros):

```text
./bin/souc check tests/run-pass/madaros_seq_minimal_witness.sio
# E137 seq_new, E011 .push/.count, E013 index, E137 acknowledge

SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check tests/run-pass/madaros_seq_minimal_witness.sio
# rc=0
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc run tests/run-pass/madaros_seq_minimal_witness.sio
# PASS
```

---

## 3. Fix shape (checker only)

| File | Change |
|------|--------|
| `self-hosted/check/types.sio` | `ty_seq` / `is_seq_type` / `seq_elem_type` — TyNamed("Seq") + inner elem |
| `self-hosted/check/compat.sio` | Seq↔Seq compatibility by element type (not bare name) |
| `self-hosted/check/check.sio` | lower `Seq<T>`; bind + typecheck `seq_new`/`seq_push`/`seq_get`/`seq_set`/`seq_len`/`seq_count`/`acknowledge`; methods `.push`/`.count`/`.get`/`.set`/`.len`; index elem type |

`acknowledge` requires `with Epistemic` (E170), matching `.value` honesty.

---

## 4. What this greens (owner map)

| Fix | Greens | Does not green |
|-----|--------|----------------|
| **Madaros Seq (checker)** | `rapamycin_kaxi_fuse_prior` **check**; clinical **ontology import** path through `ontology/model.sio` | sobol Saltelli fn-effects; halo E170 test edit; clinical plot/`1_000_000` test slice; **run**/native of Seq until lowering lands |

---

## 5. Build (from source — mandatory)

```bash
bash scripts/ci/build_modular_madaros.sh artifacts/self-hosted/madaros-seq
```

Do **not** trust prebuilt `bin/souc` for post-fix claims.

### Measured 2026-08-17 (fresh `artifacts/self-hosted/madaros-seq`)

| Target | rc | errors |
|--------|---:|-------:|
| `tests/run-pass/madaros_seq_minimal_witness.sio` | **0** | 0 |
| `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` | **0** | 0 |
| `stdlib/ontology/model.sio` | **0** | 0 |
| Nested `Bag.items[0].tags.push` probe | **0** | 0 |
| `pbpk28_rapamycin_clinical.sio` | 1 | **no** ontology/model Seq cascade; residuals = plot E175/E009 + `1_000_000` E137 + chemistry E004 (test/stdlib / other) |
| Prebuilt `bin/souc` kaxi (control) | 1 | E137 `seq_new` still |

### Implementation notes (forensic)

1. **Checker only** — not lowering/codegen. `souc check` greens; `souc run` of Seq is still unproven under Madaros.
2. `checker_lower_type_expr_list_mut` returns `TypeEntryList`, not `Option` — matching it as Option silently skipped `ty_seq` and left bare `Seq` without elem (nested field methods E011).
3. `seq_new` returns `ty_seq_named(Seq, unknown)`; binding annotation supplies `T` via Seq wildcard compat.
4. Index: inlined Seq arm in `checker_check_index_inplace` (free `indexable_elem_type` path was flaky on TyNamed Seq during isolation).
5. `acknowledge` builtin + E170 Epistemic gate (lean_single surface).

---

## 6. Non-claims

- **Did not** port Seq to IR lower / native codegen — run-path still open.
- **Did not** fix clinical plot API / digit separators / chemistry E004 (test+stdlib owners per residual split).
- check.sio path overlapped cursor-3 E219 claim by file; Seq surface is disjoint from E219 refusal logic.
- GitHub outage this session — commits local only until push is possible.
