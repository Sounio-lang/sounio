<!-- docs:meta
topic_id: repo.docs.audit.readme
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.readme
-->

# Repository audit (2026-05)

Forensic inventory of **what the repo actually contains** vs **what public surfaces claim**.
Generated during the website honesty pass on branch `website/fixes-from-main`.

## Regenerate scale numbers

```bash
bash scripts/dev/measure_repo_scale.sh
bash scripts/dev/measure_repo_scale.sh --json artifacts/audit/repo_scale.v1.json
```

## Executive summary

| Metric | Measured value | Common mistake |
|---|---:|---|
| Tracked `.sio` files | **4,233** | Counting only `stdlib/` (~1,008 files) |
| Tracked `.sio` lines | **~1.84M** | Reporting ~200k–660k from partial trees |
| `self-hosted/` | **~542k lines** | Treating compiler as "small bootstrap" |
| `stdlib/` modules | **128** (66 works / 59 scaffold / 3 doc-only roots) | Badge "90% tested" without smoke-test audit |
| CI `*gate*.sh` scripts | **129** | Assuming all gates run on `make check` |
| Works gates wired to CI | **~10%** (see A.4) | "127 gates = green CI" |

**Canonical public-claim downgrades and closures:** `docs/serious-language/public-claim-registry.v1.tsv` (`stdlib.surface = validated_research` for the bounded support contract only, LSP/editor = validated_research for checked preview tooling, remaining prototype rows such as generics and Windows stay downgraded).

## Audit phases (machine-readable)

| Phase | Question | Artifact |
|---|---|---|
| **A** | Folder-by-folder pillars (compiler, stdlib, CI, formal, narrative) | `stdlib_module_audit.v1.json`, `gpu_subsystem_audit.v1.json`, `ci_gates_audit.v1.json` |
| **A.2** | 59 stdlib scaffold modules — depth + fake smoke tests | `stdlib_scaffold_deep_audit.v1.json`, `stdlib_module_audit_refined.v1.json` |
| **A.3** | 33 "scaffold" gates reclassified (orchestrators vs honest scaffold) | `ci_gates_scaffold_a3.v1.json`, `ci_gates_scaffold_deep_audit.v1.json` |
| **A.4** | Works gates vs Makefile / GitHub CI / hub wiring | `ci_gates_works_wiring_a4.v1.json`, `ci_gates_works_reachability_a4.v1.json` |
| **A.5** | Orphan priority matrix + wiring waves | `ci_gates_orphan_priority_a5.v1.json` |
| **B** | Regeneratable scale snapshot | `repo_scale.v1.json` (via `scripts/dev/measure_repo_scale.sh`) |

## Key findings

### Stdlib (A.2)

- **66 works**, **59 scaffold**, **3** non-module roots (`BENCHMARKS.md`, etc.).
- **32 smoke tests** under `tests/stdlib/` print `FOO_OK` without exercising module logic — inflates pass-rate narratives.
- Examples: `tests/stdlib/geometry/test_types.sio` vs `stdlib/geometry/engine.sio` (481 lines, no real test).

### CI gates (A.3–A.4)

- **94 gates** tiered "works"; only **3** on regular GitHub CI; **1** on `make check` transitive path (`stdlib_science_pipeline`).
- **52 works gates** had **zero references** before wiring waves 1–3 (commit `7c6698ab6`).
- GitHub CI runs **scaffold** gates (`selfhost_host`, `souc_v2`, `claude_operational_contract`) more often than most works gates.

### Wiring delivered (waves 1–3, `7c6698ab6`)

1. **Ontology compile bundle** in `run_ontology_validation.sh` (6 gates).
2. **`native_v2_struct_gate.sh`** orchestrator (14+ sub-gates).
3. **Umbrella** rows: `struct_orchestrator`, `kretikos_kaxi_meta` (with env skips).

### GPU (`gpu_subsystem_audit.v1.json`)

- `self-hosted/gpu/` ~83k LOC; kaxi/epistemic wired in hubs.
- `quant/quantize.sio`, `multi/multi_gpu.sio` — substantial code, **no callers**, typecheck failures.

## What to cite externally

| Safe | Unsafe |
|---|---|
| "4k+ `.sio` files, ~1.8M LOC tracked" (with `measure_repo_scale.sh`) | "660k LOC language" (stdlib-only) |
| "Self-hosted compiler ~540k LOC" | "Small experimental repo" |
| "128 stdlib modules; bounded reliability gate" | "90% stdlib tested" without gate context |
| "129 CI gate scripts; subset wired" | "127 green gates on every PR" |

## Related docs

- [public-claim-registry.v1.tsv](../serious-language/public-claim-registry.v1.tsv)
- [readiness-ledger.md](../serious-language/readiness-ledger.md)
- [KNOWN_LIMITATIONS.md](../compiler/KNOWN_LIMITATIONS.md)
- Website honest status: `website/src/components/status/HonestStatusSection.astro`
