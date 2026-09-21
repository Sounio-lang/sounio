<!-- docs:meta
topic_id: repo.docs.audit.codegen-dead-body-deletion-2026-08-18
authority: repo_only
audience: users
last_validated: 2026-03-07
validated_by: A2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.codegen-dead-body-deletion-2026-08-18
-->

# Codegen dead-body deletion — option-3 result

**Date:** 2026-08-18
**Counsel:** minimax-cli2
**Working tree:** `codegen/v2-ref-deletion-20260818` at `/tmp/wt-codegen-deletion` (branched from `origin/main` at `2016efb8e4`).
**Companion priors:**
- `docs/audit/CODEGEN_DUPLICATION_REFUTED_HYPOTHESIS_2026-08-17.md` (homonym-cascade refutation)
- `docs/audit/CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md` (48-pair byte-diff classification)

## Scope (user's directive, restated)

Delete ONLY bodies the census proves unreachable, one body at a time with cheap `souc check` between each. Confirm first with a repo-wide `native::codegen::` importer search. Measure how many of the 25 residual E175-shaped homonyms drop. If any remain, name each: symbol, the two files, and whether the copies DIVERGE or are identical.

## Step 0 — Confirmation: importer search across the entire repo

`grep -rn 'native::codegen' --include='*.sio' .` (excluding `archive/` and `bootstrap/`) returned **24 importers** (17 glob `use native::codegen::*` and 7 qualified — see the doc that accompanies this one for the full list).

`grep -rn 'compile_ir_function_v2_ref' --include='*.sio' .` returned exactly **4 lines**:

| file | line | role |
|---|---|---|
| `self-hosted/native/codegen.sio` | 4187 | definition (`pub fn`) |
| `self-hosted/native/codegen.sio` | 4189 | internal call (from `compile_ir_function_v2_ref` body to `compile_ir_function_v2_into`) |
| `self-hosted/native/codegen_x86_linux.sio` | 8714 | definition (`pub fn`) |
| `self-hosted/native/codegen_x86_linux.sio` | 8716 | internal call (from `compile_ir_function_v2_ref` body to `compile_ir_function_v2_into`) |

**Zero external callers.** Confirmed dead in BOTH files before any edit.

## Step 0.5 — Premise verification on the user's `compile_ir_function_v2_into` claim

The dispatch stated: *"compile_ir_function_v2_into tem 7 chamadores no x86_linux e 0 directos no codegen.sio, onde só é alcançado por v2_ref, que também está morto."*

That is half right and half wrong:

- ✅ **True:** the INTERNAL codegen.sio reachability of `compile_ir_function_v2_into` is exactly one path — through `compile_ir_function_v2_ref`. Zero OTHER direct callers inside `codegen.sio`'s own body.
- ✅ **True:** `compile_ir_function_v2_ref` is dead (see Step 0).
- ❌ **Not 7, not reachable-from-outside-only via v2_ref:** `compile_ir_function_v2_into` is **externally LIVE** in BOTH files:

| external importer | imports from | call site |
|---|---|---|
| `self-hosted/compiler/module_loader.sio` (qualified, line 36) | `native::codegen` | 1845, 3742 |
| `self-hosted/native/wide_driver.sio` (glob, line 22) | `native::codegen::*` | 255 |
| `self-hosted/compiler/render_native_compile_driver_stable.sio` (glob, line 35) | `native::codegen::*` | 233 |
| `self-hosted/compiler/module_native_streaming.sio` (qualified, line 13) | `native::codegen_x86_linux` | 141 |

So in `codegen.sio`'s `compile_ir_function_v2_into` is reached from 4 distinct external call sites; in `codegen_x86_linux.sio`'s it is reached from 1.

**Operationally:** per the user's "APENAS os corpos que o censo prova inalcançáveis" rule, the only deletions authorised by the census are the two `compile_ir_function_v2_ref` bodies. **`compile_ir_function_v2_into` is NOT deletable** — it has live external callers in both modules.

## Step 1 — Delete `compile_ir_function_v2_ref` from `codegen.sio`

Body at lines 4187–4191 (5 lines). Replaced with a tombstone comment recording the deletion reason and the census evidence. Auto-compile (`souc check` on the file plus its 4 importers) showed E175 counts unchanged at 61 / 70 / 67 / 61 — confirming the prebuilt `bin/souc` couldn't see the source change (and so cannot refute the deletion), and that no importer was generating E175s about `v2_ref` even before the edit.

Commit: `dead-code(codegen): delete compile_ir_function_v2_ref from codegen.sio` (1 file, +8/-5).

## Step 2 — Delete `compile_ir_function_v2_ref` from `codegen_x86_linux.sio`

Body at lines 8714–8718. Mirror edit. Auto-compile on `codegen_x86_linux.sio` and `module_native_streaming.sio` (the only importers of `codegen_x86_linux::*` that could plausibly reach this symbol): E175 counts unchanged at 0 / 0.

Commit: `dead-code(codegen_x86_linux): delete compile_ir_function_v2_ref from codegen_x86_linux.sio` (1 file, +7/-5).

## Step 3 — Measure: how many of the 25 residual E175-shaped homonyms fell?

**Zero.** The two deleted bodies (`compile_ir_function_v2_ref` × 2) are NOT in the 25-residual set.

The 25-residual set is the 48-pair byte-diff census (§`CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md`) minus the 23 PUB-SWAP-ONLY pairs (whose bodies are byte-identical after stripping `pub`, so they're "no actual content difference" and don't behave differently as E175 generators). That leaves 24 IDENTICAL + 1 SUBSTANTIVE-DIVERGENT = **25 residual pairs whose content-shape could plausibly generate or fail to generate E175s**.

`v2_ref` is not one of those 25 names. So deleting it does not touch any residual.

**Re-census after deletion** (run on the same source files in `/tmp/wt-codegen-deletion`):

```
IDENTICAL (debt): 24
PUB_SWAP_ONLY (debt+vis): 23
SUBSTANTIVE_DIVERGENT (bug): 1

Sum of E175-residual (IDENTICAL + DIVERGENT): 25
```

**25 of 25 remain. 0 fell.**

## Step 4 — Name each of the 25 residuals (per user directive)

### 24 IDENTICAL pairs (debt — pure duplication, bodies byte-equal)

Both files literally carry the same line: e.g. `pub fn ARCH_X86_64() -> i64 { native_policy_arch_x86_64() }`. Deletion direction is blocked by glob-importers that depend on the name existing in the named module — see prior docs for the deletion-direction analysis.

| # | symbol | files | divergence |
|---:|---|---|---|
| 1 | `ARCH_RISCV64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 2 | `ARCH_UNKNOWN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 3 | `ARCH_X86_64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 4 | `ERR_BACKEND_NOT_IMPLEMENTED` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 5 | `ERR_INVALID_MATRIX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 6 | `ERR_INVALID_TARGET` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 7 | `ERR_OK` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 8 | `ERR_TARGET_FLAGS_REQUIRE_NATIVE` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 9 | `ERR_TRACE_WRITE_FAILED` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 10 | `FORMAT_ELF64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 11 | `FORMAT_MACHO64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 12 | `FORMAT_NONE` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 13 | `FORMAT_PE64` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 14 | `MACOS_SYS_OPEN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 15 | `MATRIX_AME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 16 | `MATRIX_APPLE_AMX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 17 | `MATRIX_AUTO` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 18 | `MATRIX_IME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 19 | `MATRIX_INTEL_AMX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 20 | `MATRIX_OFF` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 21 | `MATRIX_VME` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 22 | `OS_LINUX` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 23 | `OS_UNKNOWN` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |
| 24 | `OS_WINDOWS` | codegen.sio, codegen_x86_linux.sio | IDENTICAL |

### 1 SUBSTANTIVE-DIVERGENT pair (the bug — runtime divergence)

| # | symbol | files | divergence |
|---:|---|---|---|
| 25 | `name_is_get_arg_count` | codegen.sio (L948–L965, 18 lines), codegen_x86_linux.sio (L1199–L1229, 31 lines) | **SUBSTANTIVE-DIVERGENT** |

The 13-character `"get_arg_count"` byte-cascade is identical in both files. The x86_linux copy **also** matches a 9-character `"arg_count"` alias (with explicit comment: *"Source-level spelling is `arg_count`; retain the older `get_arg_count` backend alias for hand-built IR witnesses."*). The codegen copy **does not** include the 9-char alias. So `name_is_get_arg_count("arg_count")` returns `false` from `codegen.sio`'s version but `true` from `codegen_x86_linux.sio`'s version — runtime divergence depending on which module the call resolves through. This is the user-framed "bug à espera".

## Step 5 — Sync `name_is_get_arg_count` (the SUBSTANTIVE-DIVERGENT residual) — direction analysis first

User follow-up directive: *"sincroniza o name_is_get_arg_count entre os dois ficheiros… se este símbolo diverge, diz COMO diverge antes de sincronizares, porque a diferença pode ser um fix que só entrou numa das cópias — e nesse caso a sincronização tem uma direcção certa e uma errada."*

### How it diverges (forensic evidence from `git blame`)

| | codegen_x86_linux.sio (L1217–L1247) | codegen.sio (L948–L965) |
|---|---|---|
| Original commit | `781b11a780b`, 2026-04-18 (Demetrios) | same |
| Subsequent edit | **`f664766e134`, 2026-07-11** (Demetrios) | **none** (`git log -S'arg_count' -- codegen.sio` returns zero) |
| State today | 31 lines, **two branches**: `arg_count` (9 chars) + `get_arg_count` (13 chars) | 18 lines, **one branch**: only `get_arg_count` (13 chars) |

The x86_linux copy carries an explicit inline comment (L1218–1219):

```
// Source-level spelling is "arg_count"; retain the older
// "get_arg_count" backend alias for hand-built IR witnesses.
```

This comment disambiguates the sync direction: the 9-char alias was a **deliberate, documented extension** introduced on 2026-07-11 to recognize the source-level spelling `arg_count`. The 13-char `get_arg_count` was **retained** as a backend alias for hand-built IR witnesses — not deprecated. codegen.sio was missed in that propagation.

### Sync direction

**x86_linux → codegen.sio.** The opposite direction (codegen → x86_linux, removing the 9-char branch) would **delete a feature that was deliberately introduced** and break any source code that uses `arg_count`. The directional git-blame check on the inline comment is what resolves the otherwise-symmetric-looking divergence.

### Impact (what the sync fixes)

The 4 call sites of `name_is_get_arg_count` in codegen.sio (L3266, L3980, L4054, L4205) currently return `false` for `"arg_count"`. The 6 call sites in codegen_x86_linux.sio (L5899, L6655, L6794, L9035, plus the local dispatch at L6655 / 6794) return `true`. After the sync, all 10 call sites resolve consistently — dispatch is no longer split depending on which module the call resolves through.

### Verification (before commit)

- `souc-stage2 check self-hosted/native/codegen.sio` → `gate_pass=1375`, `gate=950/1000`, 21 functions refined. The new 9-char branch typechecks cleanly. (The "error: no main" tail is the standard noise for library files — it appears identically for `codegen_x86_linux.sio` which already has the 9-char branch.)
- `souc-stage2 check self-hosted/native/codegen_x86_linux.sio` → identical econf stats, confirms no regression.

Commit: `fix(native): sync name_is_get_arg_count in codegen.sio with codegen_x86_linux.sio` (1 file, +18/-0).

## Re-census after sync — how many of the 25 residuals fell?

The byte-diff census (with `stripPub`) **does not** record a fall for #25. The 9-char branch was added to `name_is_get_arg_count` in `codegen.sio`, but the body now carries a sync-provenance block (4 lines) and two `// "arg_count" = 9 chars` / `// "get_arg_count" = 13 chars` markers that `codegen_x86_linux.sio` does not have. After `stripPub`, the bodies still differ in those 5 comment lines, so the byte census still classifies #25 as SUBSTANTIVE-DIVERGENT.

What the sync *did* close is the **logic divergence**, not the byte divergence:

- Pre-sync: `name_is_get_arg_count("arg_count")` returns `false` from `codegen.sio`, `true` from `codegen_x86_linux.sio` — runtime split depending on which module resolves the call. The user-named "bug à espera".
- Post-sync: `name_is_get_arg_count("arg_count")` returns `true` from both files. No runtime divergence.

The byte-diff re-classification after the sync, under the three-category scheme that was in force at the time:

```
IDENTICAL (debt):              24
PUB_SWAP_ONLY (debt+vis):      0
SUBSTANTIVE_DIVERGENT (census): 1   <- #25: comment-only, not logic (the methodology bug)
```

**The previous "1 of 25 fell" headline in commit `a0da4d95b7` was true at the logic level but did not survive a strict byte-diff census.** Step 6 below re-measures on `origin/main` after PR #1837 merged, with the user's option-1 framing (keep the sync-provenance comment) and the methodology refinement that introduces a `COMMENT_ONLY_DIVERGENT` category so that #25 stops being listed alongside real runtime bugs. Under the four-category scheme (now shipped in `scripts/research/codegen_byte_diff_census.cjs`), the 25-residual set re-classifies to 24 IDENTICAL + 0 PUB_SWAP_ONLY + 1 COMMENT_ONLY_DIVERGENT + 0 SUBSTANTIVE_DIVERGENT — i.e. zero runtime-divergent pairs, exactly as the option-1 framing predicted.

The remaining 24 IDENTICAL pairs are still the user-named "dívida" — same name in two files, byte-equal bodies, glob-importable. Their consolidation direction analysis is recorded in `CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md` and was the prior doc's scope; this doc only records the sync, its deliberate preservation, and the census refinement it motivates.

## Halt per FLEET_CONSTRAINTS

The user's directive also asked for a measurement of how many of the 25 residuals fall from this deletion. I have done that measurement in the form available on this pod:

- The prebuilt `./bin/souc` cannot see source edits (per FLEET_CONSTRAINTS: "**`./bin/souc` is PREBUILT.** Editing compiler source does not change it."). The cheap `souc check` baseline / post-edit E175 counts are identical — confirming the deletion didn't break anything in the prebuilt's view, but also confirming this measurement cannot see source-level deltas.
- The full self-compile gate (`scripts/ci/native_v2_driver_self_compile_gate.sh`) that would measure the residual E175 count change is forbidden on this pod per FLEET_CONSTRAINTS ("the k8s liveness probe recycles the pod under CPU saturation"). Slurm path is broken (`launch failed requeued held`).
- The static byte-diff census is the available measurement, and it says **0 of 25 fell** (because the deletions don't touch any of the 25).

This is the precisely-bounded refutation the protocol asks for: the deletions were safe (zero external callers), they did not touch the 25 residual homonym names, and the available measurement cannot detect a fall. To detect a fall, route through the self-compile gate (CI or Slurm) once those are operational.

## Files referenced (this worktree)

- `self-hosted/native/codegen.sio` — line 4187 replaced (8-line tombstone, 5-line body removed)
- `self-hosted/native/codegen_x86_linux.sio` — line 8714 replaced (7-line tombstone, 5-line body removed)
- `self-hosted/compiler/module_loader.sio` (importer, unchanged)
- `self-hosted/native/wide_driver.sio` (importer, unchanged)
- `self-hosted/compiler/render_native_compile_driver_stable.sio` (importer, unchanged)
- `self-hosted/compiler/module_native_streaming.sio` (importer of codegen_x86_linux, unchanged)
- `docs/audit/CODEGEN_DUPLICATION_REFUTED_HYPOTHESIS_2026-08-17.md`
- `docs/audit/CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md`
- `scripts/ci/native_v2_driver_self_compile_gate.sh` (NOT executed on this pod)

## Branch

`codegen/v2-ref-deletion-20260818` at `/tmp/wt-codegen-deletion`. Four commits (all pushed to origin, PR #1837):

```
f81031a968 fix(native): sync name_is_get_arg_count in codegen.sio with codegen_x86_linux.sio
705722f701 docs(audit): codegen dead-body deletion — option-3 result (0/25 residual fell)
a869385512 dead-code(codegen_x86_linux): delete compile_ir_function_v2_ref from codegen_x86_linux.sio
4ca16b3b26 dead-code(codegen): delete compile_ir_function_v2_ref from codegen.sio
```

The "0/25 residual fell" headline in commit 705722f701 was true at the time it was written (the deletion in steps 1–2 doesn't touch any of the 25 names); commit f81031a968 updates the score to **1/25 fell** by closing the SUBSTANTIVE-DIVERGENT pair. See "Re-census after sync" above for the breakdown.

## Step 6 — Post-merge measurement (2026-08-18, after PR #1837 merged in `80cc1366a2`)

User follow-up directive: *"Agora que o v2_ref saiu, mede quantos caem. Por cada um que sobrar: que símbolo, em que dois ficheiros, e se as cópias DIVERGEM ou são idênticas."*

### Re-measurement on origin/main

`git show dde4b0b0d4:self-hosted/native/codegen.sio` and `git show dde4b0b0d4:self-hosted/native/codegen_x86_linux.sio` were extracted and re-classified with the byte-diff census (see `scripts/research/codegen_check_named_residuals.cjs` in this repo). Both file snapshots are byte-identical between `dde4b0b0d4` and `64924d371a` (the merge commit `80cc1366a2` of #1837 is present in both).

### How many of the 25 fell with `v2_ref`'s removal?

**Zero.** The two deleted bodies (`compile_ir_function_v2_ref` × 2) are not in the 25-residual set. Removing them is the safe option-3 deletion the user authorised; it does not touch any of the 25 homonym names. The 25-residual landscape is unchanged in count.

### The 25, named on `dde4b0b0d4`

| # | symbol | codegen.sio | codegen_x86_linux.sio | divergence |
|---:|---|---|---|---|
| 1 | `ARCH_RISCV64` | L45 | L85 | IDENTICAL |
| 2 | `ARCH_UNKNOWN` | L42 | L82 | IDENTICAL |
| 3 | `ARCH_X86_64` | L43 | L83 | IDENTICAL |
| 4 | `ERR_BACKEND_NOT_IMPLEMENTED` | L66 | L106 | IDENTICAL |
| 5 | `ERR_INVALID_MATRIX` | L68 | L108 | IDENTICAL |
| 6 | `ERR_INVALID_TARGET` | L67 | L107 | IDENTICAL |
| 7 | `ERR_OK` | L65 | L105 | IDENTICAL |
| 8 | `ERR_TARGET_FLAGS_REQUIRE_NATIVE` | L70 | L110 | IDENTICAL |
| 9 | `ERR_TRACE_WRITE_FAILED` | L69 | L109 | IDENTICAL |
| 10 | `FORMAT_ELF64` | L61 | L101 | IDENTICAL |
| 11 | `FORMAT_MACHO64` | L62 | L102 | IDENTICAL |
| 12 | `FORMAT_NONE` | L60 | L100 | IDENTICAL |
| 13 | `FORMAT_PE64` | L63 | L103 | IDENTICAL |
| 14 | `MACOS_SYS_OPEN` | L5132 | L10523 | IDENTICAL |
| 15 | `MATRIX_AME` | L58 | L98 | IDENTICAL |
| 16 | `MATRIX_APPLE_AMX` | L54 | L94 | IDENTICAL |
| 17 | `MATRIX_AUTO` | L52 | L92 | IDENTICAL |
| 18 | `MATRIX_IME` | L56 | L96 | IDENTICAL |
| 19 | `MATRIX_INTEL_AMX` | L55 | L95 | IDENTICAL |
| 20 | `MATRIX_OFF` | L53 | L93 | IDENTICAL |
| 21 | `MATRIX_VME` | L57 | L97 | IDENTICAL |
| 22 | `OS_LINUX` | L48 | L88 | IDENTICAL |
| 23 | `OS_UNKNOWN` | L47 | L87 | IDENTICAL |
| 24 | `OS_WINDOWS` | L50 | L90 | IDENTICAL |
| 25 | `name_is_get_arg_count` | L948 (36 lines) | L1217 (31 lines) | **COMMENT_ONLY_DIVERGENT** (post-refinement census; was SUBSTANTIVE_DIVERGENT under the three-category scheme; see "Census methodology refinement" below) |

### Reclassification of #25 — the byte-census mislabels this pair

The byte-diff census (with `stripPub` applied) still flags `name_is_get_arg_count` as SUBSTANTIVE-DIVERGENT. The 36-vs-31 line gap is **entirely comments**:

| what differs | codegen.sio | codegen_x86_linux.sio |
|---|---|---|
| signature | `pub fn` (visibility) | `fn` (private) |
| sync-provenance block (4 lines) | present | absent |
| `// "arg_count" = 9 chars` marker | present | absent |
| `// "get_arg_count" = 13 chars` marker | present | absent |
| 9-char `arg_count` byte cascade (9 byte-cmp + return true) | present | present |
| 13-char `get_arg_count` byte cascade (13 byte-cmp + return true) | present | present |

The byte cascades are byte-equal between the two files (modulo `pub`). The 5-line size delta is exclusively in comments.

**Runtime check:** `name_is_get_arg_count("arg_count")` returns `true` from both files post-`80cc1366a2` (the squash of #1837 carried the 9-char branch forward to codegen.sio). The user's framing "duas cópias DIVERGENTES são um bug à espera" no longer applies at the executable level — there is no logic divergence, only documentation divergence.

### User decision on #25 (option 1, not option 3)

I offered three options for closing the byte-census entry:

1. **Keep as-is** — sync-provenance preserved; #25 stays SUBSTANTIVE-DIVERGENT per byte census.
2. **Strip the sync-provenance block only** — closes the byte-census entry but loses the historical rasto.
3. **Align fully to x86_linux** — closes the byte-census entry, loses the sync-provenance AND the `// "arg_count" = 9 chars` / `// "get_arg_count" = 13 chars` markers.

User chose **option 1** with explicit reasoning:

> *"A divergência é INTEIRAMENTE em comentários: a proveniência do sync e os marcadores dos 9 e 13 chars. Isso fecha a minha pergunta — não há fix numa cópia e ausente noutra, portanto não há direcção errada possível no código. E é precisamente por isso que a opção 3 é má. Ela apaga o comentário que regista de onde veio o ramo de 9 chars, para fechar um item de byte-diff num censo. Isso é optimizar a métrica em vez do código: o número 9 é mágico, e o comentário é a única coisa que diz porquê. Daqui a três meses alguém pergunta e a resposta terá desaparecido para satisfazer um contador. O #25 não é um defeito — é um FACTO sobre o par, e o censo devia registá-lo como diferem-só-em-comentários em vez de o listar ao lado de divergências reais. Se o censo não consegue exprimir essa distinção, o defeito é do censo."*

### Census methodology refinement (shipped in this commit)

The previous byte-diff census collapsed three distinct phenomena into one category (SUBSTANTIVE-DIVERGENT). The user-identified gap was a methodology bug, not a code bug: the census was listing pairs whose bodies differed *only* in inline comments next to pairs whose byte cascades actually disagreed. The four-category scheme now in use:

```
IDENTICAL                          bodies byte-equal (modulo `pub`)
PUB_SWAP_ONLY                      bodies byte-equal after stripping `pub`
LOGIC_DIVERGENT                    logic differs (e.g. extra byte-cascade branch)
COMMENT_ONLY_DIVERGENT             bodies differ ONLY in comments — logic byte-equal
                                   (after stripPub AND stripComments)
```

`name_is_get_arg_count` on `dde4b0b0d4` (and on `64924d371a`, which is byte-identical post-`80cc1366a2`) classifies as **COMMENT_ONLY_DIVERGENT** under this scheme. `v2_ref`-style runtime divergence is **LOGIC_DIVERGENT** (which is what the original census called SUBSTANTIVE_DIVERGENT). The 24 IDENTICAL pairs above stay IDENTICAL.

This refinement changes nothing about the action set on the 24 IDENTICAL pairs (they remain pure debt, consolidation direction analysis recorded in `CODEGEN_BODY_DIFF_GLOB_HOMONYMS_2026-08-17.md`). It changes the framing of #25 from "bug à espera" to "documented divergence, deliberately preserved". That is the precise status the user's option-1 choice encodes.

#### Re-census of the 25 on the four-category scheme

Run on `git show origin/main:self-hosted/native/codegen.sio` (currently `d3ea284caf`) and `git show origin/main:self-hosted/native/codegen_x86_linux.sio`:

```
Of the 25 residual names:
  IDENTICAL: 24
  PUB_SWAP_ONLY: 0
  COMMENT_ONLY_DIVERGENT: 1   ← name_is_get_arg_count
  SUBSTANTIVE_DIVERGENT: 0
  MISSING: 0
```

`SUBSTANTIVE_DIVERGENT` went from 1 to 0. The single formerly-substantive pair is now correctly classified as `COMMENT_ONLY_DIVERGENT`, exactly as the option-1 framing predicted.

#### Re-census of the full 267-pair homonym corpus

Run on the same files at `origin/main@<current>`:

```
codegen.sio symbols: 293
codegen_x86_linux.sio symbols: 585
Common homonym symbols: 267

IDENTICAL: 183
PUB_SWAP_ONLY: 33
COMMENT_ONLY_DIVERGENT: 1   ← name_is_get_arg_count (the only one in the full corpus)
SUBSTANTIVE_DIVERGENT: 50
```

Only `name_is_get_arg_count` in the entire 267-pair corpus has a comment-only-but-logic-equal diff. The remaining 50 SUBSTANTIVE_DIVERGENT pairs are genuine logic divergence candidates and remain candidates for directional-sync analysis before any edit.

#### Implementation

The refined census now lives in the repo at:

- `scripts/research/codegen_byte_diff_census.cjs` — the full 267-pair classifier (takes the two `.sio` files as argv).
- `scripts/research/codegen_check_named_residuals.cjs` — the targeted 25-residual classifier (hardcoded `RESIDUALS` list).

Both are pure-Node CommonJS (`#!/usr/bin/env node`) with zero external deps, so they run inside the prebuilt-toolchain pod without `npm install`. Each carries a header block explaining the four categories and pointing back at this audit doc.
