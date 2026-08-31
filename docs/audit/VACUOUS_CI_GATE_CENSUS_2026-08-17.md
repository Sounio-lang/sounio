<!-- docs:meta
topic_id: repo.docs.audit.vacuous-ci-gate-census-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.vacuous-ci-gate-census-2026-08-17
-->

# Vacuous CI gate census — empty discovery → exit 0

**Date:** 2026-08-17  
**Scope:** `scripts/ci/**/*.sh` (~452 shells; ~387 `*gate*.sh`)  
**Question:** If a gate’s fixture list, glob, or directory walk comes back **empty**, does it still **exit 0**?  
**Method:** Structural read of discovery loops (`for … *`, `ls`/`mapfile` globs, `find` walks, git-diff → filtered lists). Not a full delete-and-run matrix. **No fixes.**

**Re-measured on rebase (2026-08-28, base `origin/main@26a348da09`).**
Scope moved: `scripts/ci/**/*.sh` **452 → 593**; `*gate*.sh` **387 → 519**.
**N = 6 still holds**: all six named scripts are present and each still shows the
documented empty-discovery→exit-0 shape at the new base — `madaros_changed_tests_gate.sh:126`
(`((${#selected[@]} == 0)) … exit 0`), `classify_ci_impact.sh:22`,
`run_pass_output_gate.sh:67` (`for src in $(ls …/tests/run-pass/*.sio 2>/dev/null | sort)`),
`check_doc_snippets.sh` (`fail=0` … `if [ "$fail" -gt 0 ]`), `sounio_validation.sh:132`
(`file_count -eq 0 → return 0`), `ui_type_deignore_audit.sh:27`. The `+1` related
forbid-scan (`heuristic_firewall_gate.sh:69`) and all four fail-closed contrast
scripts are also still present. This is a **census of a shape**, not of a fixed set:
N is the count of the six audited scripts, not a claim that no seventh exists among
the 593. `scripts/ci/gate_vacuity_gate.sh` on the same base reports the wider static
ratchet: `scanned 539 gates, flagged 22 with an unguarded extraction`.

---

## The number

**N = 6**

Six scripts still exit 0 when discovery yields nothing. Green CI that runs them can **overstate coverage by up to 6**.

| | Count |
|--|------:|
| Empty discovery → **exit 0** | **6** |
| Related forbid-scan (empty corpus also green) | +1 |
| Fail-closed discovery (contrast) | 4 |

---

## Two failure modes (anchors)

| Mode | Example | In N=6? |
|------|---------|--------|
| **Die before measuring** | ENIR / stack abort (e.g. exit 128 on missing `origin/main`) — red looks ordinary | **No** |
| **Measure nothing** | Empty list/glob/dir still exit 0 | **Yes — N=6** |

`madaros_visibility_context_gate.sh` is a **related weak gate**, not empty-list vacuity on trees that carry its hardcoded `duplicate_private_*.sio` paths. Softness there: **`baseline` / `KNOWN_BLOCKER` still exits 0**. Missing paths on some checkouts are path-missing, not the empty-glob shape.

---

## The 6 by shape (remedy differs)

### 1. Filtered list → skip-as-green — **n = 2**  
**Remedy:** count assertion when the job claims coverage; skip only with explicit opt-in.

| Script | Empty behaviour |
|--------|-----------------|
| `madaros_changed_tests_gate.sh` | `${#selected[@]}==0` after git-diff filter → `exit 0` |
| `classify_ci_impact.sh` | empty diff → all flags false → exit 0 |

### 2. Glob via `ls` / `$(…)` — **n = 1**  
**Remedy:** **count assertion** after discovery (`fail if n == 0`).

| Script | Empty behaviour |
|--------|-----------------|
| `run_pass_output_gate.sh` | `for src in $(ls tests/run-pass/*.sio 2>/dev/null)`; zero eligible → no regressions → **exit 0** |

### 3. Directory walk, then secondary glob — **n = 1**  
**Remedy:** **existence/count** after extract — `total == 0` must not pass.

| Script | Empty behaviour |
|--------|-----------------|
| `check_doc_snippets.sh` | `find docs` → extract → empty tmp glob → `fail==0` → **exit 0** |

### 4. `find` corpus walk — **n = 1**  
**Remedy:** **existence check** — zero inputs is failure.

| Script | Empty behaviour |
|--------|-----------------|
| `sounio_validation.sh` | `file_count -eq 0` → warning → **return 0** |

### 5. Bare `for f in dir/*.ext` — **n = 1**  
**Remedy:** **count assertion** on matched/effective work items before PASS.

| Script | Empty behaviour |
|--------|-----------------|
| `ui_type_deignore_audit.sh` | no files / no `//@ ignore` → still **`UI_TYPE_DEIGNORE_AUDIT_PASS`** |

*(Recursive sub-gate without propagated exit code is a third remedy class — not the bulk of this N.)*

---

## Related (+1), not in N

| Script | Shape | Remedy |
|--------|--------|--------|
| `heuristic_firewall_gate.sh` | forbid-scan | Require **non-empty corpus** before accepting empty violations |

---

## Fail-closed contrast (copy this shape)

Empty discovery → **non-zero**:

- `madaros_gum_fo_trust_gate.sh`
- `track_a_nv2_parity_inventory.sh`
- `ontology_hash_benchmark.sh`
- `journal_submission_gate.sh` (J5)

---

## Bottom line

**N = 6.** That is the finding. Green CI that includes these gates can overstate coverage by exactly that many whenever discovery yields nothing.
