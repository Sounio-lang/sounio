<!-- docs:meta
topic_id: repo.docs.audit.dossier-import-closure-2026-08-17
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.dossier-import-closure-2026-08-17
-->

# Dissertation dossier gate — import/closure root cause

**Date:** 2026-08-17  
**Lane:** grok-cli5 / `dossier-import-closure`  
**Evidence parent:** `docs/audit/DISSERTATION_SIX_GATES_AB_TRUTH_2026-08-17.md` §2.2  
**Gate:** `scripts/ci/dissertation_dossier_gate.sh`  
**Smoke:** `tests/run-pass/dossier_smoke.sio`  
**Module:** `scripts/dissertation/dossier_generator.sio` (**exists on disk**)

---

## 0. Headline

| Hypothesis | Verdict |
|------------|---------|
| Module file missing at that path | **No** — `scripts/dissertation/dossier_generator.sio` is present |
| Import **cap** silently dropping the `use` (64 vs 128) | **No** — smoke has a **single** `use`; not a count overflow |
| Authoritative-closure walker fails to follow a path shape it should | **Yes** |

**Root cause:** `module_frontend_resolve_import_relpath` never tries the
**repository-root-relative** path `scripts/dissertation/dossier_generator.sio`.
It tries `cur_dir/…`, ancestor `parent/…` while `parent` is non-empty, then
`stdlib/` and `packages/`. When walking up from `tests/run-pass`, the parent of
`tests` is empty / `.`, so the loop **stops without** probing bare `rel_path` at
the repo root. The resolver returns a non-existent local path and reports
**unresolved import** — as if the module did not exist.

This is the same honesty class as the token-ceiling silent drop: the system
**does not admit** a path it never looked for.

---

## 1. Diagnostic (prebuilt, before fix)

```text
unresolved import in authoritative closure: scripts::dissertation::dossier_generator
run_check_mode: AST closure incomplete nodes=1 unresolved=1 saturated=false
```

`use scripts::dissertation::dossier_generator::*` from `tests/run-pass/` becomes
relpath `scripts/dissertation/dossier_generator.sio`.

| Candidate | Exists? |
|-----------|---------|
| `tests/run-pass/scripts/dissertation/dossier_generator.sio` | no |
| `tests/scripts/dissertation/dossier_generator.sio` | no |
| `scripts/dissertation/dossier_generator.sio` (repo root) | **yes** — **never tried** |
| `stdlib/scripts/…` | no |

ImportList cap in `module_parse.sio` reports `overflowed` when `use` count exceeds
capacity; dossier smoke has **one** import → not the 64/128 prebuilt/source
mismatch class.

---

## 2. Fix (source-built Madaros required)

In `self-hosted/compiler/module_frontend.sio` /
`module_frontend_resolve_import_relpath`, **after** the ancestor walk and
**before** stdlib:

```sio
// Repo-root relative (scripts::, tools::, examples::, … from under tests/)
if file_exists(rel_path) {
    return rel_path
}
```

### Measurement after rebuild (`artifacts/self-hosted/madaros` from this edit)

| Step | Result |
|------|--------|
| `souc check tests/run-pass/dossier_smoke.sio` | **`check: OK`**, `about to check **2** modules`, verdict=0 |
| `souc compile` + run binary | succeeds; stdout ends with `PASS dossier_smoke` |
| Full `dissertation_dossier_gate.sh` | **PASS 4 / FAIL 1** — only remaining fail is **stdout ≠ golden** |

Import/closure class from §2.2 is **closed** under from-source Madaros with this
fix. Golden mismatch is a **separate residual** (see §3). **Golden not
touched** (dispatch order).

---

## 3. Residual (not this root class)

After import resolves, golden diff still fails:

1. **Struct field print order** in the parameter table looks scrambled vs smoke
   input (`value`/`confidence` columns) — possible layout/print bug in the
   generator path; not an import miss.
2. **Golden** contains extra §11 blocks and slightly different §10 prose than
   current generator output — snapshot drift, not closure.

Do **not** “fix” by editing
`tests/golden/dissertation/dossier_rapamycin_snapshot.md` until those residuals
are classified. Gate remains red on golden alone; the **reported** A/B failure
mode (unresolved import / nodes=1) is gone once the binary includes this fix.

---

## 4. Cap measurement note

| Surface | Value | Relevant? |
|---------|-------|-----------|
| `ImportList` paths array (`module_parse`) | capacity with `overflowed` flag | Not hit (1 import) |
| `ModuleClosure.paths` | 256 | Not hit |
| Prebuilt vs source 64 vs 128 import lore | historical | **Not** this bug |

Built Madaros from source before concluding on caps (`build_modular_madaros.sh` →
local `artifacts/self-hosted/madaros`).

---

## 5. Coordination / landing

Fix landed on `lane/grok-cli5/dossier-root-import-20260817` off `origin/main` by
explicit founder/dispatch order after diagnosis. Overlap with grok-cli3
`string-lit-128` claim on `self-hosted/compiler/**` was noted; edit is confined
to `module_frontend_resolve_import_relpath` (import path resolution only).

---

## 6. Evidence commands

```bash
# Prebuilt — still shows unresolved import until PR binary ships
./bin/souc check tests/run-pass/dossier_smoke.sio

# From-source with fix
bash scripts/ci/build_modular_madaros.sh "$PWD/artifacts/self-hosted/madaros"
MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros ./bin/souc check tests/run-pass/dossier_smoke.sio
# expect: check: OK, 2 modules

MADAROS_RAW_BIN=$PWD/artifacts/self-hosted/madaros bash scripts/ci/dissertation_dossier_gate.sh
# expect: check/compile/run PASS; golden may still FAIL
```

## 7. Landed patch

```diff
diff --git a/self-hosted/compiler/module_frontend.sio b/self-hosted/compiler/module_frontend.sio
index e5ebc58f6e..735f333851 100644
--- a/self-hosted/compiler/module_frontend.sio
+++ b/self-hosted/compiler/module_frontend.sio
@@ -495,6 +495,18 @@ fn module_frontend_resolve_import_relpath(rel_path: string, cur_dir: string) ->
             anc_depth = anc_depth + 1
         }
     }
+    // Repo-root relative. The ancestor walk stops when `parent` of a top-level
+    // directory (e.g. `tests` from `tests/run-pass`) is empty, so it never
+    // tries the bare rel_path at the repository root. That left every
+    // `use scripts::…` / `use tools::…` import from under tests/ permanently
+    // unresolved even when the file existed (dossier_smoke →
+    // scripts/dissertation/dossier_generator.sio). Not an import-cap drop:
+    // the path was never a candidate. Same honesty class as silent token
+    // truncation — absence must not look like "module missing" when the
+    // resolver simply did not look.
+    if file_exists(rel_path) {
+        return rel_path
+    }
     let stdlib_env_root = read_env("SOUNIO_STDLIB_PATH")
     if str_len(stdlib_env_root) > 0 {
         let stdlib_env_path = str_concat(str_concat(stdlib_env_root, "/"), rel_path)
```
