<!-- docs:meta
topic_id: repo.docs.audit.dissertation-pbpk-suite-residual-split-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dissertation-pbpk-suite-residual-split-2026-08-17
-->

# Dissertation PBPK suite — residual preflight split (test vs Madaros)

**Date:** 2026-08-17  
**Lane:** grok-cli5 / `pbpk-residual-split-20260817`  
**Parent triage:** `docs/audit/DISSERTATION_PBPK_SUITE_TRIAGE_2026-08-16.md` rows 17–22 (preflight/status-1 family)  
**Instrument:** `bin/souc` Madaros v0.80.0 vs `SOUNIO_SOUC_ENGINE=lean_single` on this worktree  
**Scope:** classification only. No `self-hosted/` edits. No science parameters changed.

Parent triage filed these four under **toolchain_defect — preflight** because job-9908 kept only 5-line tails. This note re-runs `souc check` and splits each failure into **test/stdlib source** vs **Madaros** (opposite owners).

---

## 0. Dossier branch hygiene (same session)

| Branch | Fate |
|--------|------|
| `lane/grok-cli5/dossier-root-import-20260817` | **Current.** PR **#1789** open: repo-root import fix for `scripts::` / dossier_smoke. |
| `lane/grok-cli5/dossier-import-closure` | **False start.** Same investigation; earlier incomplete landing. **Remote deleted**; local branch deleted. Do not reopen. |

Two branches for one task is unreadable backlog; only #1789 remains.

---

## 1. Method

```bash
export SOUNIO_STDLIB_PATH=$(pwd)/stdlib
./bin/souc check <path>                                          # Madaros
SOUNIO_SOUC_ENGINE=lean_single ./bin/souc check <path>           # lean_single A/B
```

Plus minimal probes for fn-type effects, `.value`+Confidence, digit separators, and `use chemistry::ontology`.

| Case | Path | Madaros | lean_single |
|------|------|---------|-------------|
| sobol_pce | `stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio` | rc=1 E009+E035 | rc=0 (check) |
| kaxi_fuse | `tests/run-pass/rapamycin_kaxi_fuse_prior.sio` | rc=1 E137/E011 | **rc=0** |
| clinical | `stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio` | rc=1 multi | rc=1 multi |
| halo_pgx | `tests/run-pass/halo_pgx_gate_pass.sio` | rc=1 E170 | **rc=0** |

---

## 2. Per-case split

### 2.1 `pbpk28_sobol_pce` — **mostly TEST/STDLIB; one Madaros secondary**

**Diagnostics (Madaros):**

- `E009` `sp28_selftest_main` / `sp28_selftest_semaglutide_main`: `saltelli_run(..., model)` — **expected fn#167, found fn#6 / fn#11**
- `E035` on imported `epistemic_pbpk28::main`: missing `Epistemic` required by `ep28_selftest_main`

**What the source does:**

```text
// stdlib/epistemic/sobol.sio
pub fn saltelli_run(..., model: fn([f64; 10]) -> f64) -> SaltelliResult

// validation callbacks
fn sobol_model_pbpk28(x: [f64; 10]) -> f64 with Mut, Div, Panic { ... }
fn sobol_model_semaglutide(...) -> f64 with Mut, Div, Panic { ... }
let sr = saltelli_run(n_samples, n_dims, seed, sobol_model_pbpk28)
```

**Probe (same worktree):**

| Probe | Madaros | lean_single |
|-------|---------|-------------|
| Effectful fn passed to bare `fn(...)` param | **E009** fn# mismatch | accepts |
| Bare fn → bare param | ok | ok |
| Effectful fn → `fn(...) with Mut, Div, Panic` param | ok | ok |

**Judgement:**

| Layer | Owner | Why |
|-------|-------|-----|
| E009 fn-type / effects | **TEST + stdlib API** | Madaros is **correctly strict**: an effectful model is not a pure `fn([f64;10])->f64`. lean_single under-checks. Fix is annotate `saltelli_run`’s `model` (and `saltelli_analyze`) with `with Mut, Div, Panic` (or make models pure — they cannot be; CN needs Mut/Div/Panic). |
| E035 Epistemic | **Madaros (secondary)** | `epistemic_pbpk28.sio` alone checks **verdict=0** under Madaros (`main` already has `Epistemic`). E035 appears only as a **dependency of sobol_pce** — multi-module effect attribution noise after the E009 failures, not a missing annotation in ep28 source. |

**Not science.** Saltelli never runs.

**Next owner:** stdlib/epistemic (signature) or a small validation-only wrapper — **not** a Madaros “fn identity is broken” bug for the primary E009.

---

### 2.2 `rapamycin_kaxi_fuse_prior` — **MADAROS (Seq / kaxi surface)**

**Diagnostics (Madaros):** `E137 seq_new`, `E011` no method (`.push` / Seq methods), `E137 acknowledge`, cascade into `epistemic/kaxi::kaxi_fuse` (`E011`, `E013` index, `E137 acknowledge`).

**lean_single:** **check PASS** (compile completes).

**What the source does (correct for the language surface lean implements):**

```sounio
var obs: Seq<Knowledge<f64>> = seq_new()
obs.push(measure(...))
let post = kaxi_fuse(obs)
let cl_post = acknowledge(post, "...")
```

`Seq<T>`, `seq_new`, Seq methods, and `acknowledge` are implemented as **lean_single builtins** (`self-hosted/compiler/lean_single.sio` Seq/TY_SEQ path). Madaros modular checker has **no equivalent** resolution for `seq_new` / Seq methods / `acknowledge` as free names.

**Judgement:**

| Layer | Owner |
|-------|-------|
| Entire failure | **Madaros** — Seq\<T\> + acknowledge surface not ported / not wired in modular check |
| Test source | **Not defective** for the intended method witness |

Same root breaks `stdlib/epistemic/kaxi.sio` and (below) `stdlib/ontology/model.sio`.

**Next owner:** Madaros Seq / epistemic builtin lane — **not** a dissertation test rewrite (rewriting to fixed arrays would abandon the Seq witness the gate documents).

---

### 2.3 `pbpk28_rapamycin_clinical` — **SPLIT: test/stdlib first; ontology leg = Seq (not codex-3 enforcement)**

Never reaches `PBPK28_RAPAMYCIN_CLINICAL_PENDING_OBSERVED`. Dies in check. **Both engines fail**, for overlapping but not identical reasons.

#### 2.3.1 Test / stdlib defects (both engines — fix without Madaros)

| Symptom | Evidence | Owner |
|---------|----------|-------|
| `1_000_000.0` in `scale_for_plot` | Both: `E137`/`E200` name `_000_000`. Spec claims digit separators (`docs/spec/LANGUAGE_SPECIFICATION.md`); Madaros lexer has an underscore path in `lexer/mod.sio` / `numparse.sio`, but the live tokenisation still splits `1_000_000.0` → `1` + `_000_000`. | **Toolchain gap (both engines)** on separators; **immediate clinical unblock = test** write `1000000.0` |
| `error_bar_entry(2.0, scale..., ...)` | API is `error_bar_entry(label: string, value, uncertainty, confidence)` (`stdlib/plot/epistemic.sio`). Call site passes **f64 first** (time), not a string label. | **TEST SOURCE** |
| `error_bar_chart(&ebs, "title")` | API is `error_bar_chart(entries, n: i64, title)` — arity 3. | **TEST SOURCE** |
| E175 `error_bar_entry` / `error_bar_chart` private | Callees are non-`pub` in `plot/epistemic.sio`. Madaros enforces module privacy. | **STDLIB** (`pub` the helpers) **and/or** test stop calling private plot demos |

These alone prevent PENDING emission even if ontology were deleted from the file.

#### 2.3.2 Ontology import cascade (Madaros-only; lean_single OK)

```text
use chemistry::ontology;
→ ontology/model.sio uses seq_new() / Seq methods throughout
→ Madaros: E137 seq_new, E011 no method, E013 index, E004 cascades
→ lean_single: chemistry::ontology import probe compiles
```

Isolated probe `use chemistry::ontology` reproduces the Seq cascade under Madaros; lean_single compiles.

**Is this codex-3’s “ontology-enforcement gap in the checker”?**

**No — not the same root.**  

- This clinical failure is **Seq infrastructure missing under Madaros**, which happens to live inside `ontology/model.sio` because that module is Seq-heavy.  
- An “ontology-enforcement” checker gap would be about **whether Madaros enforces ontology/typed-bundle rules** (nominal ChEBI, `@ ontology-bundle`, subsumption). That is a different defect class.  
- **Do not double-fix Seq under an ontology-enforcement label.**  
- **Handoff:** the Seq/Madaros hole is shared with §2.2 (kaxi). If codex-3’s lane is only enforcement policy in the checker, **leave Seq to the Seq/Madaros owner**; clinical remains blocked on §2.3.1 even after Seq lands.  
- If codex-3 has expanded to “make ontology modules check under Madaros”, then **ontology/model Seq is their consumer surface** — same physical files, still the Seq root, not enforcement.

**Judgement summary for clinical:**

| Slice | Owner | Blocks PENDING? |
|-------|-------|-----------------|
| digit separator / scale_for_plot | toolchain + easy test workaround | yes |
| plot epistemic call shape + privacy | **test + stdlib** | yes |
| chemistry::ontology → Seq | **Madaros Seq** (same as kaxi) | yes under Madaros |
| “ontology enforcement” checker policy | **not this case** — do not assign as codex-3 enforcement twin |

**Recommended order:** (1) test/stdlib plot + `1000000.0` so lean path can emit PENDING; (2) Madaros Seq unblocks ontology import and kaxi together.

---

### 2.4 `halo_pgx_gate_pass` — **TEST SOURCE (Madaros correct); lean under-enforces**

**Diagnostics (Madaros):** `E170` — accessing `.value` on epistemic type requires `with Epistemic` or `acknowledge`.

**Source:**

```sounio
fn main() with IO, Mut, Div, Panic, Confidence(750) {
    let k_cl: Knowledge<f64> = measure(40.0, uncertainty: 1.6)
    let cl = k_cl.value   // E170 under Madaros
    ...
}
```

**Probes:**

| Surface | Madaros | lean_single |
|---------|---------|-------------|
| bare `.value` (no Epistemic) | E170 | E170 |
| `Confidence(750)` + `.value` | **E170** | **PASS** |
| `Confidence(750)` + `Epistemic` + `.value` | ok | ok |
| `Confidence(750)` + `acknowledge` | E137 `acknowledge` undeclared | PASS |

**Judgement:**

| Layer | Owner | Why |
|-------|-------|-----|
| Primary | **TEST SOURCE** | Add `Epistemic` to `main` (and keep `Confidence(750)` for the PGx gate story). Madaros E170 is the honesty rule. |
| Secondary | lean_single **under-enforcement** | Treats `Confidence(N)` as enough to open `.value`; Madaros does not. Design/help text pairs `.value` with `Epistemic` / `acknowledge`, not Confidence alone. |
| acknowledge path on Madaros | **Madaros** | `acknowledge` not resolved as builtin (same family as §2.2) — but **not required** if the test uses `.value` under `with Epistemic`. |

**Not science.** Compile-time gate witness only.

---

## 3. Owner matrix (the deliverable)

| # | Case | Primary owner | Secondary | Hand off? |
|---|------|---------------|-----------|-----------|
| 19 | `pbpk28_sobol_pce` | **stdlib/test** — effectful model vs bare `fn` in `saltelli_run` | Madaros multi-module E035 noise | No |
| 20 | `rapamycin_kaxi_fuse_prior` | **Madaros** — Seq + acknowledge | — | Seq/Madaros lane |
| 22 | `pbpk28_rapamycin_clinical` | **test+stdlib** first (plot API, digit literal workaround) | **Madaros Seq** via ontology import | **Not** codex-3 ontology-**enforcement**; Seq shared with kaxi |
| 18 | `halo_pgx_gate_pass` | **test** — add `with Epistemic` | lean_single Confidence/E170 laxity | No |

Opposite owners matter:

- Fixing Madaros does **not** green sobol or halo without source/API edits.  
- Fixing tests does **not** green kaxi without Seq on Madaros.  
- Clinical needs **both** slices; ontology-enforcement work alone does not clear it.

---

## 4. Relation to parent triage buckets

Parent row bucket **toolchain_defect — preflight** still holds for job-9908 scoring (science never ran). This note **sub-classifies** that bucket:

| Parent # | Name | Sub-class |
|----------|------|-----------|
| 18 | halo_pgx_gate_pass | test_source (Madaros-correct E170) |
| 19 | pbpk28_sobol_pce | test_stdlib_api (+ Madaros E035 secondary) |
| 20 | rapamycin_kaxi_fuse_prior | madaros_seq_surface |
| 22 | pbpk28_rapamycin_clinical | mixed_test_stdlib + madaros_seq_via_ontology |

Science count from parent triage is unchanged: these four are still not science_model_defect.

---

## 5. Commands / receipts

```text
./bin/souc --version
# Madaros v0.80.0

./bin/souc check stdlib/darwin_pbpk/validation/pbpk28_sobol_pce.sio
# E009 expected fn#167 found fn#6 / fn#11; E035 ep28 main

./bin/souc check tests/run-pass/rapamycin_kaxi_fuse_prior.sio
# E137 seq_new, E011 methods, E137 acknowledge; lean_single rc=0

./bin/souc check stdlib/darwin_pbpk/validation/pbpk28_rapamycin_clinical.sio
# E004 scale_for_plot; E175/E009/E010 plot; ontology/model Seq cascade

./bin/souc check tests/run-pass/halo_pgx_gate_pass.sio
# E170 .value; lean_single rc=0

./bin/souc check stdlib/darwin_pbpk/epistemic_pbpk28.sio
# verdict=0  (isolates E035 as multi-module secondary)
```

Probes under `/tmp/fnprobe*.sio`, `/tmp/e170*.sio`, `/tmp/uscore.sio`, `/tmp/ont_import.sio` (not committed).

---

## 6. Explicit non-claims

- Did not rebuild Madaros; used checked-in `bin/souc` v0.80.0.  
- Did not re-run full `dissertation_pbpk_suite_gate.sh` (53 entries).  
- Did not edit test sources or `self-hosted/` in this pass.  
- Did not assign clinical to codex-3 as ontology-enforcement twin (see §2.3.2).
