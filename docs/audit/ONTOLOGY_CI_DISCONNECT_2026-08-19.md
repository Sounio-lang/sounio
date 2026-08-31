<!-- docs:meta
topic_id: repo.docs.audit.ontology-ci-disconnect-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-19
validated_by: grok-cli3
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.ontology-ci-disconnect-2026-08-19
-->

# Ontology CI disconnect — measurement, not rewiring

Date: 2026-08-19
Lane: `ontology-ci-audit-20260819`
Worktree: `/workspace/.wt/ontology-ci-audit`
Source: `origin/main` `f9b314736421f6cff0ca02ffe02c6cb7def71a0a`
  (`compiler(effects): Chaotic exists; six extras after id 22; Mod held (#1963)`)
This is not a rewiring. No gate was wired. No known-failure tag was changed.

Companion: [`ONTOLOGY_CI_DISCONNECT_2026-08-19.tsv`](ONTOLOGY_CI_DISCONNECT_2026-08-19.tsv)

**Escalation (dispatch item 4):** none of the fourteen unwired gates accused
Madaros versus lean_single *ontology-axiom* divergence. The known E158 split
is why `madaros_ontology_enforcement_gate.sh` is already wired; it is not one
of the fourteen. No pre-report escalation.

---

## Semantic-Lane declaration

```text
Semantic-Lane-ID: ontology-ci-audit-20260819
Owner: grok-cli3
Concept-IDs: none
Intent-Preserved: analogy != ontology; compile success != runtime parity; Madaros is the claim-oracle; lean_single is the bootstrap seed without semantic authority
Transformation: none — measurement of CI wiring against today's main
Types-Changed: none
Effects-Changed: none
IR-Changed: none
Claims-Introduced: 17 ontology-named scripts/ci gates exist; 3 are named by .github/; of the 14 unwired, 8 are GREEN today and 6 are RED-STALE; none are RED-REAL
Claims-Forbidden: "the fourteen are all green so they can be wired"; "the fourteen re-measured Madaros/lean ontology divergence"; "practically everything is already gated"; "a red rc is an axiom hole"
Assumptions: the wiring instrument is `git grep -l --fixed-strings <basename> origin/main -- .github/`; Madaros v0.80.0 is the default `bin/souc` oracle
Write-Set: docs/audit/ONTOLOGY_CI_DISCONNECT_2026-08-19.md; docs/audit/ONTOLOGY_CI_DISCONNECT_2026-08-19.tsv
Read-Set: scripts/ci/*ontolog*; scripts/ci/classify_ci_impact.sh; scripts/ontology/validate_unit_metadata.py; .github/workflows/ci.yml; self-hosted/check/units.sio; self-hosted/check/mod.sio; stdlib/ontology/**
Positive-Witness: none claimed (no rewiring)
Negative-Witness: none claimed
Acceptance-Gate: fourteen per-gate rcs against f9b3147364; each classified; map of the 17 versus founder "practically everything"; wiring order proposed; nothing wired
Integration-Target: none
Authoritative-Only-If: a later dispatch wires a named subset after the repairs listed below
```

---

## Instrument validation

Dispatch instrument: `git grep -c "<basename>" origin/main -- .github/`.
A colon-counting parse of that form already lied today (`origin/main:path:n`).
The truthful form is **files-with-matches**:

```bash
git grep -l --fixed-strings '<basename>' origin/main -- .github/
```

Positive control (known-wired today): `concept_status_gate.sh` →
`origin/main:.github/workflows/ci.yml` (ci.yml:68). Instrument valid.
The same ruler on the seventeen ontology basenames yields **3 named, 14 zero**.

Cheap gates ran on the pod worktree with `SKIP_BUILD=1` and worktree
`bin/souc` → Madaros v0.80.0. Heavy gates ran on Slurm node
`gpuorangefs-r770-proxmox` (16 CPUs) with an isolated tarball of this SHA
that shipped `bin/madaros-linux-x86_64` (99 964 676 B). First Slurm pass
omitted the Madaros ELF and lacked `make`; those rcs are discarded.
`generated_ontology` was re-run on the pod (`make` exists). Builder work
dir was isolated (`SOUNIO_VALIDATION_BUILD_DIR` under the job tree).

---

## The seventeen

Named by `.github/` today (ci.yml):

| gate | where |
|---|---|
| `run_ontology_validation.sh` | ci.yml:313, if `impact.ontology \| full` |
| `ontology_cli_smoke_gate.sh` | ci.yml:316, same impact predicate |
| `madaros_ontology_enforcement_gate.sh` | ci.yml:702, Madaros current-source job, `MADAROS_BIN=/tmp/madaros-ci.elf` |

Fourteen unwired. Per-gate rc against `f9b3147364`, not aggregated:

| # | gate | rc | class |
|---|---|---:|---|
| 1 | `generated_ontology_manifest_gate.sh` | 0 | GREEN |
| 2 | `generated_ontology_gate.sh --check` | 0 | GREEN |
| 3 | `ontology_cache_compile_gate.sh` | 0 | GREEN |
| 4 | `ontology_model_compile_gate.sh` | 0 | GREEN |
| 5 | `ontology_query_compile_gate.sh` | 0 | GREEN |
| 6 | `ontology_reasoner_compile_gate.sh` | 0 | GREEN |
| 7 | `ontology_typed_bridge_gate.sh` | 0 | GREEN |
| 8 | `build_ontology_validation_souc.sh` | 0 | GREEN |
| 9 | `ontology_unit_metadata_gate.sh` | 1 | RED-STALE |
| 10 | `ontology_bundle_directive_gate.sh` | 1 | RED-STALE |
| 11 | `knowledge_context_phase2_ontology_gate.sh` | 1 | RED-STALE |
| 12 | `ontology_cache_frontend_composition_gate.sh` | 1 | RED-STALE |
| 13 | `ontology_bundle_directive_native_scan_gate.sh` | 1 | RED-STALE |
| 14 | `ontology_hash_benchmark.sh` | 1 | RED-STALE |

Not all-green (dispatch: that would have meant the instrument was not
running). Zero RED-REAL. Zero INDETERMINATE after the isolated re-run.

---

## Classification notes

### GREEN — may be named first

- **generated_ontology_manifest.** Stable `MANIFEST.tsv` covers the nine
  public bundles (alg, chebi, go, hpo, loinc, part, phys, qm, snomed) plus
  stubs, witnesses, typed bridges, class/const/disjoint limits.
- **generated_ontology --check.** Regenerated `.dontology` + C-FFI stubs;
  git tree stayed clean. Already a *child* of the wired
  `run_ontology_validation` prepare step, but only when
  `GITHUB_ACTIONS=true`. Not named by any workflow.
- **cache / model / query / reasoner compile.** Concatenate the stdlib
  module with an exercise main, `souc compile` + run under Madaros.
  Already optional children of `run_ontology_validation`, and that parent
  **skips the whole bundle** if a `Seq<i64>` probe fails to compile
  (`return 0`, not a fail).
- **typed_bridge.** Run-pass GO coercion prints `ontology typed bridge go OK`;
  compile-fail sibling is refused with **E152**. Axiom still live.
- **build_ontology_validation_souc.** Isolated work dir, rc=0. The helper
  `build_native_souc.sh` prefers a checked-in ELF; this is "wrapper built",
  not a from-source fixed-point. Still what the gate claims.

### RED-STALE — the tree moved; the gate did not

Dispatch item 4 asked for one line each if any were RED-REAL.
There are none. The six reds, one line each, are aged instruments:

1. **unit_metadata** — `validate_unit_metadata.py` requires
   `unit_register_h(122913530894,` for `mg_dL` inside
   `self-hosted/compiler/lean_single.sio`; that name is registered in
   Madaros `self-hosted/check/units.sio:256` and is absent from lean_single.
2. **bundle_directive** — sibling subsumption is already refused
   (`error[E009] expected SNOMED_44054006 found SNOMED_46635009`); the gate
   still greps `cannot prove ontology subsumption`.
3. **knowledge_context_phase2** — compiled `lean_frontend` then died
   `Permission denied` on the ELF (no `chmod +x`); the script printed
   "Stage1 rejected positive witness". The diabetes proof-context file was
   **not** checked.
4. **cache_frontend_composition** — `bin/souc check` of the k2 probe is
   OK; `bin/souc run` dies in Madaros multimodule native thin-link
   (`Failed to write native binary … rc=12`). Cache write/reimport was
   **not** measured. Off-axis known native-path fragility, not an axiom.
5. **bundle_directive_native_scan** —
   `k2_check_mod_knowledge_context_ontocache_sidecar_probe` calls
   `check_items_verdict_boot4_with_ontocache_sidecar(items, source_path)`
   (2 args); `self-hosted/check/mod.sio:1621` now takes 3
   (`items, source_path, module_path`). **E010**.
6. **hash_benchmark** — patches `main` to call `ontology_run_cli()`.
   That symbol does not exist in `lean_single.sio`. Instant compile fail.
   This is a micro-benchmark, not a validator. Candidate to die.

### What this is not

It is not evidence that ontology axioms on main are sound. It is evidence
that the unwired redness today is gate rot, not a newly measured axiom hole.

---

## Map of the seventeen versus "practically everything"

Founder today: practically everything should be ontologically validated.

Census on this SHA (re-derived, not remembered):

| surface | count | command |
|---|---:|---|
| `self-hosted/` files mentioning ontolog | 42 | `git grep -l -i ontolog -- self-hosted` |
| `stdlib/` files mentioning ontolog | 101 | `git grep -l -i ontolog -- stdlib` |
| files under `stdlib/ontology/` | 43 | `git ls-files 'stdlib/ontology/**'` |
| files under `stdlib/compiler/ontology/` | 7 | `git ls-files 'stdlib/compiler/ontology/**'` |
| files under `stdlib/data/data/ontology/` | 24 | `git ls-files 'stdlib/data/data/ontology/**'` |
| tests with `ontolog` in the path | 161 | `git ls-files 'tests/**' \| rg -i ontolog` |
| kernel fixtures `tests/run-pass/ontology_*.sio` | 42 | prefix reserved by `run_ontology_validation` |
| kernel fixtures `tests/compile-fail/ontology_*.sio` | 111 | same |

### What the set actually watches

| layer | who | what is proved |
|---|---|---|
| Bundle freshness | generated_ontology, manifest | nine `.dontology` slices + generated stubs match source |
| Kernel fixtures | run_ontology_validation (wired) | `ontology_*` run-pass / compile-fail + `test_ontology.sio` |
| Inverse role | madaros_ontology_enforcement (wired) | forward `inverse_of` is E158 on **both** engines against a fresh ELF |
| Typed bridge | typed_bridge (unwired; optional child) | GO nominal upcast; bad subsumption is E152 |
| Stdlib compile+run | cache / model / query / reasoner | those four files compile and their exercise mains exit 0 |
| Directive expansion | bundle_directive | `//@ ontology-bundle` → C importer → checker; sibling reject |
| Native side-table | cache_frontend, native_scan | `.ontocache` write/reimport + k2 probes (currently unmeasurable) |
| Proof-context | knowledge_context_phase2 | Stage1 `Knowledge<… where {…}>` (currently unmeasurable) |
| Unit labels | unit_metadata | fixture JSON well-formedness; **explicitly not** clinical/UCUM/LOINC authority |
| CLI / hash | cli_smoke (wired), hash_benchmark | dispatch into a **deleted** `ontology_run_cli` |
| Wrapper builder | build_ontology_validation_souc | produces the rebuilt validation wrapper |

### What it does not watch

- **Impact classifier hole.** `classify_ci_impact.sh:97` marks `ontology=true`
  only for `stdlib/compiler/ontology/*`, `scripts/ci/*ontology*`,
  `docs/ontology/*`. It does **not** mark `stdlib/ontology/*`,
  `stdlib/data/data/ontology/*`, `self-hosted/check/ontology_side_table_cache.sio`,
  or `tests/**/ontology_*`. A PR that breaks `stdlib/ontology/reasoner.sio`
  is `stdlib=true` and **does not start** the two impact-gated ontology jobs.
- **Silent skip inside the wired parent.** `run_ontology_validation` skips
  its compile-gate bundle (cache/model/query/reasoner/typed_bridge/cli_smoke)
  if `Seq<i64>` fails to compile, and returns 0.
- **Wired smoke is skip-as-pass.** `ontology_cli_smoke_gate.sh:97-99` exits 0
  when the patched `ontology_run_cli` source will not compile. The symbol is
  gone. That named job is a no-op. (Not one of the fourteen; found while
  mapping. Not an axiom-divergence accusation.)
- **Most of `stdlib/ontology/`.** biomedical stubs, `proof_carrying_*`,
  `path_conditioned_partial_identification`, `shift_robust_risk_transport`,
  `endogenous_observability`, policy/relational modules — no compile gate.
- **Madaros unit registry.** unit_metadata still greps lean_single.
- **Clinical / PBPK / example binding.** Explicitly excluded by the
  validation filter comments and by unit_metadata's
  `--accept-no-clinical-safety`.
- **Full terminology.** Nine bounded bundles, not SNOMED/GO/HPO/LOINC/ChEBI
  completeness.
- **Query / reasoner soundness.** Compile+run of the module, not a decision
  procedure against the bundles.

Distance from "practically everything": the wired surface is a kernel-fixture
harness plus one inverse-role check, both easy to skip (impact predicate or
skip-as-pass). The compile, typed-bridge, cache, proof-context, unit, and
directive gates that would widen that surface are either unwired, optional
children, or aged.

---

## Proposed order — do not wire today

Dispatch item 3: propose only.

**Name first (green, cheap, no repair):**

1. `generated_ontology_manifest_gate.sh`
2. `generated_ontology_gate.sh --check` (already a GHA child; name it so a
   skipped parent cannot hide drift)
3. `ontology_cache_compile_gate.sh`
4. `ontology_model_compile_gate.sh`
5. `ontology_query_compile_gate.sh`
6. `ontology_reasoner_compile_gate.sh`
7. `ontology_typed_bridge_gate.sh`

Name them as *independent* steps. Leaving them only as children of
`run_ontology_validation` keeps the Seq-skip hole.

**Repair, then name:**

8. `ontology_bundle_directive_gate.sh` — accept `E009` (or both strings) as
   the sibling diagnostic. The axiom already holds.
9. `ontology_unit_metadata_gate.sh` — register evidence against
   `self-hosted/check/units.sio`, not lean_single.
10. `ontology_bundle_directive_native_scan_gate.sh` — give the sidecar probe
    the third `AstPath` argument.
11. `knowledge_context_phase2_ontology_gate.sh` — `chmod +x`; do not label
    `EACCES` as a Stage1 reject.
12. `ontology_cache_frontend_composition_gate.sh` — stop requiring Madaros
    `souc run` of an imported k2 probe; `check` already passed.

**Die, do not name:**

13. `ontology_hash_benchmark.sh` — dead `ontology_run_cli`, not a validator.

**Do not double-name:**

14. `build_ontology_validation_souc.sh` — builder already invoked by the
    wired parent.

**Rewiring that would actually approach the founder sentence** (later
dispatch, not this one): teach `classify_ci_impact.sh` that
`stdlib/ontology/**`, `stdlib/data/data/ontology/**`, and
`self-hosted/**/*ontology*` flip `ontology=true`. Until that happens,
naming the greens still leaves them dark on the PRs that break the soul.

**Wired hole to repair on its own ticket:** `ontology_cli_smoke_gate.sh`
skip-as-pass. Fail closed, or delete the job.

---

## Commands

```text
git grep -l --fixed-strings concept_status_gate.sh origin/main -- .github/
# → origin/main:.github/workflows/ci.yml

# cheap / small-compile (pod, SKIP_BUILD=1, Madaros v0.80.0)
bash scripts/ci/generated_ontology_manifest_gate.sh          # rc=0
bash scripts/ci/ontology_unit_metadata_gate.sh               # rc=1
bash scripts/ci/ontology_bundle_directive_gate.sh            # rc=1
bash scripts/ci/ontology_cache_compile_gate.sh               # rc=0
bash scripts/ci/ontology_model_compile_gate.sh               # rc=0
bash scripts/ci/ontology_query_compile_gate.sh               # rc=0
bash scripts/ci/ontology_reasoner_compile_gate.sh            # rc=0
bash scripts/ci/ontology_typed_bridge_gate.sh                # rc=0
bash scripts/ci/generated_ontology_gate.sh --check           # rc=0

# heavy (Slurm r770, isolated tree, madaros-linux-x86_64 shipped)
bash scripts/ci/knowledge_context_phase2_ontology_gate.sh            # rc=1
bash scripts/ci/ontology_cache_frontend_composition_gate.sh          # rc=1
bash scripts/ci/ontology_bundle_directive_native_scan_gate.sh        # rc=1
bash scripts/ci/ontology_hash_benchmark.sh                           # rc=1
bash scripts/ci/build_ontology_validation_souc.sh <isolated-wrapper> # rc=0
```

---

## Halt

Nothing wired. Nothing reverted. Dirty effect-set work on
`lane/grok-cli3/effect-set-as-data-20260819` was not mixed into this tree.
Registry / topic-registry / DOCS_AUTHORITY_MATRIX not touched.
