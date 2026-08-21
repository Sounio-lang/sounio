<!-- docs:meta
topic_id: repo.docs.audit.input-keyword-2026-08-19
authority: repo_only
audience: users
last_validated: 2026-08-21
validated_by: cursor-2
source_of_truth: docs/governance/topic-registry.v1.json#repo.docs.audit.input-keyword-2026-08-19
-->

# Input keyword

Founder ruling 2026-08-19: withdraw example `label` vocabulary and map `asserted → Input`. `AstProvInput` already had `PROVENANCE_KIND_INPUT` and a branch in `check/epistemic.sio` `provenance_from_ast`. It lacked a token and a parser arm. This change gives `Input` the keyword.

Forensic sources (read before the edit): #2001, #2011.

## Copies measured before the edit

`fn keyword_lookup` / sibling tables, counted on `6ce6e4dafd`:

| Path | Role | Changed? |
|---|---|---|
| `self-hosted/lexer/tables.sio` | full PascalCase table (`keyword_lookup`) | **yes** |
| `self-hosted/parser/parser.sio` `pt_read_kind` | **this is the live peek path**; `parser_peek` re-derives the kind from source text and ignores stored `PT_KIND` | **yes — len-5 `"Input"`** |
| `self-hosted/lexer/mod.sio` `lex_source_to_globals` | writes token kinds; peek does not read them | **yes — now calls `keyword_lookup`** |
| `self-hosted/lexer/mod.sio` `keyword_lookup_code` | frozen i64 protocol; unknown → 123 → Ident | left in place |
| `self-hosted/compiler/render_native_compile_driver_lean.sio` `keyword_lookup_cs` | lean driver render | no |
| `stdlib/compiler/lexer/scanner.sio` | stdlib copy | no |
| `self-hosted/bootstrap/bootstrap_v0.sio` | seed-era mirror | **no — seed** |
| `bootstrap/bootstrap_stage1.sio` | older bootstrap | no |

Provenance parser loops (Derived / Computed / Measured):

| Path | Loops | Changed? |
|---|---|---|
| `self-hosted/parser/types.sio` | 2 (angle ~1107, bracket ~1378) | **yes, both** |
| `self-hosted/bootstrap/bootstrap_v0.sio` | 2 | no |
| `bootstrap/bootstrap_stage1.sio` | 2 | no |

`self-hosted/compiler/lean_single.sio` has **no** `TokenKind::Measured` and **no** `keyword_lookup`. Knowledge annotation tags are any identifier (`TK == 3`): “Accept ignorable provenance/evidence tags like Derived.” The seed does not store `PROVENANCE_KIND_*` from those tags.

## Does lean_single need the same edit?

**No, not to compile the program.** Baseline on prebuilt engines, Slurm `cpuops-t560-proxmox`, `workspace_visible=no`, SHA `6ce6e4dafd`:

| Engine | program | check rc |
|---|---|---:|
| **madaros** | `Knowledge[f64, Input]` | 0 (Ident; bracket else-arm skips it) |
| **madaros** | `Knowledge<f64, Input>` | **1** parse error `expected=147 actual=175` (`>` eaten as `CmpGt`) |
| **madaros** | `Knowledge[f64, Measured]` | 0 |
| **madaros** | `let input = 7` | 0 |
| **lean_single** | all four | 0 |

lean_single already accepts `Input` as an ignorable Ident. Adding the keyword only to Madaros does **not** create a program that compiles on one engine and fails on the other. The angle form currently **diverges the other way** (Madaros refuses, lean_single accepts). The keyword closes that gap.

The seed is not edited. That is a founder decision; this lane does not take it.

## What landed (Madaros)

1. `TokenKind::Input` next to `Measured` in `self-hosted/lexer/token.sio`, plus `tk_is_keyword`.
2. `keyword_lookup` length-5 PascalCase `"Input"` in `self-hosted/lexer/tables.sio`.
3. **Live peek:** `"Input"` in `pt_read_kind` length-5 in `self-hosted/parser/parser.sio` (same reconstructive table that already had `Measured` at length 8). Two source-built Madaros rebuilds that only patched the lexer still failed `Knowledge<f64, Input>` with `expected=Gt actual=RParen` — `parser_peek` never read the stored kind.
4. Parser arms in **both** Knowledge component loops in `self-hosted/parser/types.sio` → `AstProvInput`.
5. `print_token_kind` arms in `self-hosted/main.sio` and `self-hosted/main_bootstrap.sio` (those matches have no wildcard).
6. Tests: `tests/run-pass/knowledge_provenance_input.sio`, `knowledge_provenance_measured.sio`, `knowledge_input_lowercase_ident.sio`.

Not in this change: `Source`, `Literature`, the absent-versus-`derived` collision, the silent `else` skip, `bootstrap_v0.sio`, `lean_single.sio`.

`TypeEntry` still does not persist provenance (`check/epistemic.sio`: “TypeEntry does not yet persist full validity/provenance metadata.”). `provenance_from_ast` maps `AstProvInput` → `PROVENANCE_KIND_INPUT` (5). The user-visible proof that the token exists is that `Knowledge<f64, Input>` parses: before the keyword that form failed on Madaros because `Input` was `Ident` and `>` was read as `CmpGt`.

## Verification

Edited modules `souc check` under **prebuilt** Madaros v0.80.0: `token.sio` rc=0, `tables.sio` rc=0, `types.sio` rc=0, `parser.sio` rc=0.

Patched-compiler cells — Slurm job on `gpuorangefs-multi-r740-proxmox`, 32 CPUs, `workspace_visible=no`, source-built Madaros (`scripts/ci/build_modular_madaros.sh`, build_rc=0, elapsed=549s, ELF 100564677 bytes). lean_single remains the committed seed ELF.

| Engine | program | check rc | run rc | stdout |
|---|---|---:|---:|---|
| **madaros** (source-built) | `knowledge_provenance_input` | 0 | 0 | `INPUT_KEYWORD_OK` |
| **madaros** (source-built) | `knowledge_provenance_measured` | 0 | 0 | `MEASURED_KEYWORD_OK` |
| **madaros** (source-built) | `knowledge_input_lowercase_ident` | 0 | 7 | (return `7`; intended) |
| **lean_single** (seed ELF) | `knowledge_provenance_input` | 0 | 0 | `INPUT_KEYWORD_OK` |
| **lean_single** (seed ELF) | `knowledge_provenance_measured` | 0 | 0 | `MEASURED_KEYWORD_OK` |
| **lean_single** (seed ELF) | `knowledge_input_lowercase_ident` | 0 | 7 | (return `7`; intended) |

Two earlier source-built rebuilds that patched only the lexer/`keyword_lookup` path still failed `Knowledge<f64, Input>` (`expected=Gt actual=RParen`). The third rebuild, after `pt_read_kind` gained `"Input"`, is the green matrix above.

## What this file is not

- Not a seed / `lean_single` change.
- Not E220 for unknown components.
- Not a claim that TypeEntry now stores provenance kind.
